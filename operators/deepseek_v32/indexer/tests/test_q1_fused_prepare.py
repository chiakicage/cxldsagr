"""Current-key preparation, exact official dispatch and observed native identity."""

import os
import subprocess
import sys
from contextlib import nullcontext

import pytest
import torch

from operators.deepseek_v32.indexer import _native_cache, echo, official_decode, official_prefetch
from operators.deepseek_v32.indexer.page64 import pack_q1_keys
from operators.deepseek_v32.indexer.selection import exact_topk
from operators.deepseek_v32.indexer.tests.test_official_prefetch import _context, _lease


def test_loaded_control_does_not_claim_current_key_participation(monkeypatch):
    monkeypatch.setattr(official_prefetch, "_FUSED_ENTRY_OBSERVED", False)
    monkeypatch.setattr(official_prefetch, "_MODULE_NAME", "loaded_control")
    monkeypatch.setattr(
        _native_cache,
        "native_info",
        lambda _: pytest.fail("Unused current-key entry queried native"),
    )
    assert official_prefetch.runtime_info() is None


def test_runtime_identity_binds_observed_entry_without_loading_native(monkeypatch):
    record = {"artifact_name": "bound.so", "artifact_sha256": "digest"}
    names = []
    monkeypatch.setattr(official_prefetch, "_FUSED_ENTRY_OBSERVED", True)
    monkeypatch.setattr(official_prefetch, "_MODULE_NAME", "bound")
    monkeypatch.setattr(
        official_prefetch, "_module", lambda: pytest.fail("Must not load a provider")
    )
    monkeypatch.setattr(_native_cache, "native_info", lambda name: names.append(name) or record)
    assert official_prefetch.runtime_info() == {
        "preparation": official_prefetch.PREPARATION,
        "entry_point": "logits_from_keys",
        "native": record,
    }
    assert names == ["bound"]


@pytest.mark.parametrize("fail", [True, False])
def test_entry_marks_participation_only_after_normal_return(monkeypatch, fail):
    marker, error = object(), RuntimeError("injected official failure")
    keys = torch.empty(2, 128)
    monkeypatch.setattr(official_prefetch, "_FUSED_ENTRY_OBSERVED", False)
    monkeypatch.setattr(official_prefetch, "_prepare_keys", lambda *args: (None, None, None))
    monkeypatch.setattr(official_decode, "metadata", lambda _: None)

    def run(*args):
        assert not official_prefetch._FUSED_ENTRY_OBSERVED
        if fail:
            raise error
        return marker

    monkeypatch.setattr(official_prefetch, "_logits_prepared", run)
    if fail:
        with pytest.raises(RuntimeError) as raised:
            official_prefetch.logits_from_keys(None, keys, None, None, None, {})
        assert raised.value is error
    else:
        assert official_prefetch.logits_from_keys(None, keys, None, None, None, {}) is marker
    assert official_prefetch._FUSED_ENTRY_OBSERVED is not fail


def test_allocation_failure_precedes_cleanup_registration(monkeypatch):
    keys = torch.empty(65, 128, dtype=torch.float8_e4m3fn)
    scales = torch.ones(65)
    lease = {"device": torch.empty(97, 576, dtype=torch.bfloat16)}
    error = RuntimeError("injected packing allocation failure")
    monkeypatch.setattr(official_prefetch, "_new_staging", lambda *args: object())

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch, "empty", fail)
    with pytest.raises(RuntimeError) as raised:
        official_prefetch._prepare_keys(keys, scales, lease, None)
    assert raised.value is error
    assert official_prefetch.CLEANUP_KEY not in lease
    assert official_prefetch.STATE_KEY not in lease


def test_unaligned_key_storage_is_rejected_before_staging(monkeypatch):
    keys = torch.zeros(65 * 128 + 1, dtype=torch.uint8)[1:].view(65, 128).view(torch.float8_e4m3fn)
    scales = torch.ones(65)
    lease = {"device": torch.empty(97, 576, dtype=torch.bfloat16)}
    monkeypatch.setattr(
        official_prefetch, "_new_staging", lambda *args: pytest.fail("Invalid K reached staging")
    )
    with pytest.raises(ValueError, match="4-byte-aligned"):
        official_prefetch._prepare_keys(keys, scales, lease, None)


def _release(lease):
    lease[official_prefetch.CLEANUP_KEY]()
    torch.cuda.synchronize()
    lease.pop(official_prefetch.CLEANUP_KEY)
    lease.pop(official_prefetch.STATE_KEY)
    lease.pop("_prepared", None)


def _packing_reference(keys, scales):
    keys = keys.cpu().view(torch.uint8)
    scales = scales.cpu().view(torch.uint8)
    packed = torch.zeros(((len(keys) + 63) // 64, 8448), dtype=torch.uint8)
    for page in range(len(packed)):
        start, count = page * 64, min(64, len(keys) - page * 64)
        packed[page, : count * 128] = keys[start : start + count].flatten()
        packed[page, 8192 : 8192 + count * 4] = scales[start * 4 : (start + count) * 4]
    return packed


def _assert_prepared(keys, scales, lease, result):
    packed, blocks, state = result
    torch.cuda.synchronize()
    assert torch.equal(packed.cpu(), _packing_reference(keys, scales))
    assert blocks.cpu().tolist() == [list(range(len(packed)))]
    history = lease["history_length"]
    logical = torch.arange(history)
    expected = torch.zeros(len(keys), dtype=torch.int32)
    expected[:history] = lease["page_table"].cpu()[logical // 64] * 64 + logical % 64
    assert torch.equal(state.page_table.cpu()[0], expected)
    assert state.host_ids.eq(-1).all()
    assert state.counter.item() == 0
    assert lease["allocation_log"].eq(2**31 - 1).all()
    assert lease["prefetch_stats"].eq(0).all()


@pytest.mark.parametrize("columns", [1, 63, 64, 65, 129, 32768, 65537])
@pytest.mark.parametrize("key_offset", [0, 4])
def test_cuda_fused_bytes_and_same_address_graph_refresh(columns, key_offset):
    capacity = max(384, (columns + 63) // 64 * 64)
    lease = _lease(history=columns - 1, host_capacity=capacity)
    lease["page_table"] = torch.arange(capacity // 64, device="cuda", dtype=torch.int32).flip(0)
    keys = (
        torch.arange(columns * 128 + key_offset, device="cuda", dtype=torch.int64)
        .to(torch.uint8)[key_offset:]
        .view(columns, 128)
        .view(torch.float8_e4m3fn)
    )
    bits = torch.tensor(
        [0, -2147483648, 1, -2147483647, 0x7FC01234, 0x7F800000, -8388608],
        dtype=torch.int32,
        device="cuda",
    )
    scales = bits.repeat((columns + 6) // 7)[:columns].view(torch.float32)
    context = _context(lease)
    unchanged = {
        name: lease[name].clone()
        for name in ("device", "host_to_device", "device_to_host", "priority", "offset")
    }
    result = official_prefetch._prepare_keys(keys, scales, lease, context)
    _assert_prepared(keys, scales, lease, result)
    for name, value in unchanged.items():
        assert torch.equal(lease[name], value)
    _release(lease)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = official_prefetch._prepare_keys(keys, scales, lease, context)
    for iteration in range(3):
        keys.view(torch.uint8).fill_(1 + 97 * iteration)
        scales.view(torch.int32).fill_(0x7FA00123 + iteration)
        lease["page_table"].copy_(lease["page_table"].roll(1))
        graph.replay()
        _assert_prepared(keys, scales, lease, result)
    del graph
    _release(lease)


@pytest.mark.parametrize("transient", [False, True])
@pytest.mark.parametrize("nondefault_stream", [False, True])
@torch.inference_mode()
def test_cuda_public_fused_scores_selection_and_actual_promotion(transient, nondefault_stream):
    columns = 32769
    lease = _lease(history=columns - 1, host_capacity=32832)
    lease.update(
        page_table=torch.arange(513, dtype=torch.int32, device="cuda").flip(0),
        transient_suffix=transient,
        max_prefetch=64,
    )
    device = lease["device"].device
    torch.manual_seed(8319)
    q = torch.randn(1, 64, 128, device=device).to(torch.float8_e4m3fn)
    keys = torch.randn(columns, 128, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(columns, device=device) + 0.5
    weights = torch.rand(1, 64, device=device) / 64
    context = _context(lease)
    packed = pack_q1_keys(keys, scales)
    blocks = torch.arange(len(packed), dtype=torch.int32, device=device)[None]
    lease["offset"][1] = torch.inf
    expected = official_prefetch.logits(
        q[None], packed, weights, context, blocks, official_decode.metadata(context), columns, lease
    ).clone()
    _release(lease)
    lease["offset"][1] = 0
    before_offsets = lease["offset"].clone()
    stream = torch.cuda.Stream() if nondefault_stream else None
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        actual = echo.logits(q, keys, weights, scales, columns - 1, lease, _pad_to_stride=True)
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert actual.shape == (1, 33024)
    assert actual[:, columns:].isneginf().all()
    assert torch.equal(actual[:, :columns].view(torch.int32), expected.view(torch.int32))
    for left, right in zip(exact_topk(actual, 2048), exact_topk(expected, 2048), strict=True):
        if left.dtype == torch.float32:
            left, right = left.view(torch.int32), right.view(torch.int32)
        assert torch.equal(left, right)
    state = lease[official_prefetch.STATE_KEY]
    count = min(64, state.counter.item())
    assert count == 64
    ids = state.host_ids[0, :count].long()
    assert ids.unique().numel() == count
    logical_to_host = state.page_table[0, : columns - 1].long()
    logical = torch.where(torch.isin(logical_to_host, ids))[0]
    assert len(logical) == count and actual[0, logical].gt(0).all()
    slots = lease["host_to_device"][ids].long()
    assert (
        slots.unique().numel() == count
        and slots.gt(0).all()
        and slots.lt(len(lease["device"])).all()
    )
    assert torch.equal(lease["device_to_host"][slots], ids)
    assert torch.equal(lease["allocation_log"][slots], ids)
    assert torch.equal(lease["device"][slots].cpu(), lease["host"][ids.cpu()])
    assert torch.equal(state.records[0, :count].cpu(), lease["host"][ids.cpu()])
    assert lease["prefetch_stats"].tolist() == [count, 0, state.counter.item() - count]
    assert torch.equal(lease["offset"].view(torch.int32), before_offsets.view(torch.int32))
    assert official_prefetch.runtime_info()["entry_point"] == "logits_from_keys"
    _release(lease)


@pytest.mark.parametrize("kind", ["bounded", "full"])
def test_cuda_fused_prepared_token_and_duplicate_owner(kind):
    from operators.deepseek_v32.indexer import cache_ops

    lease = _lease(history=128)
    keys = torch.zeros(129, 128, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.ones(129, device="cuda")
    scratch = torch.empty(96, dtype=torch.int64, device="cuda")
    if kind == "bounded":
        token = cache_ops.prepare_prefetch_free(
            lease["priority"],
            lease["free_bitmap"],
            lease["device_to_host"],
            lease["free_slots"],
            lease["allocation_log"],
            lease["counter"],
            lease["prefetch_stats"],
            scratch,
            timestamp=8,
        )
    else:
        token = cache_ops.prepare_prefetch(
            lease["priority"],
            lease["free_slots"],
            lease["allocation_log"],
            lease["counter"],
            lease["prefetch_stats"],
            scratch,
            timestamp=8,
            host_capacity=len(lease["host"]),
            query_count=1,
        )
    lease.update(_prepared=token, max_prefetch=64)
    official_prefetch._prepare_keys(keys, scales, lease, _context(lease))
    assert token.used
    with pytest.raises(ValueError, match="staging already belongs"):
        official_prefetch._prepare_keys(keys, scales, lease, _context(lease))
    _release(lease)


def test_cuda_fused_entry_preserves_original_failure_and_cleanup(monkeypatch):
    from operators.deepseek_v32.indexer.tests.test_official_prefetch import _stage

    lease = _lease(history=128)
    keys = torch.zeros(129, 128, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.ones(129, device="cuda")
    error = RuntimeError("injected failure after official stage publication")
    monkeypatch.setattr(official_prefetch, "_FUSED_ENTRY_OBSERVED", False)

    def fail(*args):
        assert callable(lease[official_prefetch.CLEANUP_KEY])
        _stage(lease, lease[official_prefetch.STATE_KEY], 64)
        raise error

    monkeypatch.setattr(official_decode, "logits", fail)
    with pytest.raises(RuntimeError) as raised:
        official_prefetch.logits_from_keys(None, keys, None, scales, _context(lease), lease)
    assert raised.value is error
    assert not official_prefetch._FUSED_ENTRY_OBSERVED
    assert lease["host_to_device"][0].item() == len(lease["device"])
    _release(lease)
    assert lease["host_to_device"].eq(2**31 - 1).all()


@pytest.mark.parametrize("invalid", ["context", "negative_page", "high_page"])
def test_cuda_fused_preserves_device_assertions_in_isolated_process(invalid):
    from operators.deepseek_v32.indexer.tests.test_official_prefetch import _require_sm90

    _require_sm90()
    program = """
import sys
import torch
from operators.deepseek_v32.indexer import official_prefetch
from operators.deepseek_v32.indexer.tests.test_official_prefetch import _lease, _context
lease = _lease(history=128)
context = _context(lease)
invalid = sys.argv[1]
if invalid == 'context':
    context.add_(1)
elif invalid == 'negative_page':
    lease['page_table'][0] = -1
else:
    lease['page_table'][0] = len(lease['host']) // 64
k = torch.zeros(129, 128, dtype=torch.float8_e4m3fn, device='cuda')
scale = torch.ones(129, device='cuda')
official_prefetch._prepare_keys(k, scale, lease, context)
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, invalid],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode != 0
    assert "device-side assert" in result.stderr or "device-side assertion" in result.stderr
