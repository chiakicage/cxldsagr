"""Finite per-layer residency: hits, competing sessions, tails and dense recall."""

import pytest
import torch

from models.attention_contracts import BlockSelection
from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
from operators.nosa.attention.offload.api import NosaFetchWorkspace, prefetch_cached_history
from operators.nosa.attention.offload.tests.test_nosa_offload import _inputs, hopper  # noqa: F401

pytestmark = pytest.mark.usefixtures("hopper")


def _workspace(prefix, queries, overlap):
    return NosaFetchWorkspace(
        prefix + queries, 2, 128, device="cuda", dtype=torch.bfloat16, overlap=overlap
    )


def _run(workspace, inputs, selection, prefix, tags, owner):
    q, k, v, hk, hv, cis = inputs
    return workspace.run(
        q,
        selection,
        hk,
        hv,
        k[prefix:],
        v[prefix:],
        cis,
        prefix,
        cache_tags=tags,
        cache_owner=owner,
    )


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("prefix", [70, 128, 257])
def test_cached_fetch_hits_conflicts_and_partial_pages(overlap, prefix):
    queries = 19
    inputs = _inputs(prefix, queries)
    q, k, v, _, _, cis = inputs
    ids = torch.arange((prefix + queries + 63) // 64, device="cuda").view(1, 1, -1)
    selection = BlockSelection(ids, 64)
    workspace = _workspace(prefix, queries, overlap)
    tags = torch.zeros_like(workspace._first_use, dtype=torch.int64)
    expected = nosa_block_sparse_attention(q, k, v, selection, query_start=prefix, cis_bias=cis)
    first = _run(workspace, inputs, selection, prefix, tags, 1)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert workspace.last_transfer_bytes.item() == prefix * 2 * 128 * 2 * 2
    assert torch.all(tags[: prefix // 64] == 1)
    second = _run(workspace, inputs, selection, prefix, tags, 1)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    assert workspace.last_transfer_bytes.item() == (prefix % 64) * 2 * 128 * 2 * 2
    assert torch.all(tags[prefix // 64 :] == 0)
    # A distinct session overwrites every selected historical page, even when
    # its values happen to match. Returning to A must refill those pages again.
    for owner in (2, 1):
        actual = _run(workspace, inputs, selection, prefix, tags, owner)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert workspace.last_transfer_bytes.item() == prefix * 2 * 128 * 2 * 2


@pytest.mark.parametrize("overlap", [False, True])
def test_cached_fetch_preserves_unselected_hits_and_invalidates_suffix(overlap):
    prefix, queries = 256, 64
    inputs = _inputs(prefix, queries)
    workspace = _workspace(prefix, queries, overlap)
    tags = torch.zeros_like(workspace._first_use, dtype=torch.int64)
    all_pages = BlockSelection(torch.arange(5, device="cuda").view(1, 1, -1), 64)
    _run(workspace, inputs, all_pages, prefix, tags, 1)
    # B needs page 2 only. Page zero is overwritten for physical padding and
    # must lose A's tag; other unselected pages retain their contents and owner.
    selected = BlockSelection(torch.tensor([[[2]]], device="cuda"), 64)
    _run(workspace, inputs, selected, prefix, tags, 2)
    assert tags[:, 0].tolist() == [0, 1, 2, 1, 0]
    _run(workspace, inputs, all_pages, prefix, tags, 1)
    assert workspace.last_transfer_bytes.item() == 2 * 64 * 2 * 128 * 2 * 2


@pytest.mark.parametrize("prefix", [0, 70, 256])
def test_dense_prefetch_uses_identical_page_tags_and_nondefault_stream(prefix):
    queries = 19
    inputs = _inputs(prefix, queries)
    _, k, v, hk, hv, _ = inputs
    workspace = _workspace(prefix, queries, False)
    tags = torch.zeros_like(workspace._first_use, dtype=torch.int64)
    counter = torch.empty((), device="cuda", dtype=torch.int64)
    current = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    copy_stream.wait_stream(current)
    with torch.cuda.stream(copy_stream):
        result = prefetch_cached_history(
            workspace.keys, workspace.values, hk, hv, tags, 7, prefix, out=counter
        )
    current.wait_stream(copy_stream)
    assert result is counter
    assert counter.item() == prefix * 2 * 128 * 2 * 2
    torch.testing.assert_close(workspace.keys[:prefix], k[:prefix], rtol=0, atol=0)
    torch.testing.assert_close(workspace.values[:prefix], v[:prefix], rtol=0, atol=0)
    prefetch_cached_history(workspace.keys, workspace.values, hk, hv, tags, 7, prefix, out=counter)
    assert counter.item() == (prefix % 64) * 2 * 128 * 2 * 2


def test_cached_fetch_failed_submit_invalidates_mutated_pool(monkeypatch):
    from operators.nosa.attention.offload import _fused

    prefix, queries = 128, 19
    inputs = _inputs(prefix, queries)
    workspace = _workspace(prefix, queries, False)
    tags = torch.ones_like(workspace._first_use, dtype=torch.int64)
    selection = BlockSelection(torch.arange(3, device="cuda").view(1, 1, -1), 64)
    module = _fused._module()

    class FailingModule:
        def forward(self, *args):
            raise RuntimeError("injected before attention launch")

    monkeypatch.setattr(_fused, "_module", lambda: FailingModule())
    with pytest.raises(RuntimeError, match="injected"):
        _run(workspace, inputs, selection, prefix, tags, 2)
    assert torch.count_nonzero(tags).item() == 0
    monkeypatch.setattr(_fused, "_module", lambda: module)
    _run(workspace, inputs, selection, prefix, tags, 1)
    assert workspace.last_transfer_bytes.item() == prefix * 2 * 128 * 2 * 2


@pytest.mark.parametrize("kind", ["empty_tags", "kv_alias", "cis_alias", "mask_alias"])
def test_cached_fetch_rejects_writable_aliases_before_launch(kind):
    prefix, queries = (0 if kind == "empty_tags" else 128), 19
    inputs = list(_inputs(prefix, queries))
    workspace = _workspace(prefix, queries, False)
    pages = len(workspace._first_use)
    tags = torch.zeros((pages, 2), device="cuda", dtype=torch.int64)
    ids = torch.tensor([[[0, 1, 2]]], device="cuda")
    valid = torch.ones_like(ids, dtype=torch.bool)
    if kind == "empty_tags":
        tags = tags[:0]
    elif kind == "kv_alias":
        workspace.values = workspace.keys
    elif kind == "cis_alias":
        inputs[-1] = torch.zeros((prefix + queries, 2), device="cuda", dtype=torch.float32)
        tags = inputs[-1].view(torch.int64).reshape(-1)[: pages * 2].reshape(pages, 2)
    else:
        valid = tags.view(torch.bool).reshape(-1)[:3].reshape(1, 1, 3)
    with pytest.raises(ValueError, match="cache_tags|geometry|alias"):
        _run(workspace, inputs, BlockSelection(ids, 64, valid), prefix, tags, 1)
