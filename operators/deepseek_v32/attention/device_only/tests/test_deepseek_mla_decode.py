"""Explicit split-KV decode checks; prefill query slicing keeps its own contract."""

import hashlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from operators.deepseek_v32.attention.device_only import decode, mla
from operators.deepseek_v32.attention.device_only.tests.test_deepseek_mla import require_sm90
from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla


def test_decode_import_does_not_load_triton_or_flashmla():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['triton'] = None; sys.modules['flash_mla'] = None; "
                "from operators.deepseek_v32.attention.device_only.mla import sparse_mla_decode; "
                "from operators.deepseek_v32.attention.device_only.decode import runtime_info; "
                "assert runtime_info() is None"
            ),
        ],
        check=True,
    )


def test_runtime_info_hashes_only_retained_launch_objects(monkeypatch):
    compiled = SimpleNamespace(
        name="combine",
        metadata={"target": {"backend": "cuda", "arch": 90, "warp_size": 32}},
        asm={"ptx": "test PTX", "cubin": b"test CUBIN"},
        module=1,
        function=2,
    )
    monkeypatch.setattr(decode, "_observed", {"compile-key": compiled})
    row = decode.runtime_info()["specializations"][0]
    assert row["hash"] == "compile-key"
    assert row["artifact_sha256"]["cubin"] == hashlib.sha256(b"test CUBIN").hexdigest()
    assert row["metadata"]["target"]["arch"] == 90


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("selected", [0, 1, 127, 129, 193, 2048, 2049, 4096])
def test_decode_strides_padding_duplicates_and_invalid_ids(heads, selected):
    require_sm90()
    torch.manual_seed(418 + selected)
    q = torch.randn(1, heads * 2, 1152, device="cuda", dtype=torch.bfloat16)[:, ::2, ::2]
    kv = torch.randn(846, 1152, device="cuda", dtype=torch.bfloat16)[::2, ::2]
    ids = torch.randint(1, len(kv), (1, selected * 2), device="cuda", dtype=torch.int32)[:, ::2]
    kv[0] = float("nan")
    ids[:, ::7] = -1
    ids[:, 1::11] = len(kv) + 7
    if selected > 3:
        ids[:, 2:4] = 1
    if selected > 128:
        ids[:, 128] = 1  # Preserve repeated records across shard boundaries.
    scale = 576**-0.5
    expected = reference_sparse_mla(q, kv, ids, scale)
    original = mla.sparse_mla(q, kv, ids, scale)
    result = mla.sparse_mla_decode(q, kv, ids, scale)
    torch.testing.assert_close(result, expected, atol=4e-3, rtol=2e-2)
    torch.testing.assert_close(result, original, atol=4e-3, rtol=2e-2)


@pytest.mark.parametrize("heads", [64, 128])
def test_decode_int64_overflow_empty_shards_and_unaligned_inputs(heads):
    require_sm90()
    q = torch.zeros(heads * 576 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(1, heads, 576)
    kv = torch.full((4 * 576 + 1,), 7, device="cuda", dtype=torch.bfloat16)[1:].view(4, 576)
    kv[0] = float("nan")
    kv[1, :512] = 2
    ids = torch.full((1, 2048), 2**32 + 1, device="cuda", dtype=torch.int64)
    ids[:, 1] = -(2**32) + 1
    empty = mla.sparse_mla_decode(q, kv, ids, 1)
    assert torch.equal(empty, torch.zeros_like(empty))
    ids[:, 513] = 1
    result = mla.sparse_mla_decode(q, kv, ids, 1)
    torch.testing.assert_close(result, torch.full_like(result, 2), atol=0, rtol=0)


@pytest.mark.parametrize("scale", [-1.0, 0.0, 1.0])
def test_decode_large_logits_with_empty_shards_are_stable(scale):
    require_sm90()
    q = torch.zeros(1, 128, 576, device="cuda", dtype=torch.bfloat16)
    q[..., 575] = 100
    kv = torch.zeros(3, 576, device="cuda", dtype=torch.bfloat16)
    kv[0] = float("nan")
    kv[1, :512], kv[1, 575] = 2, 100
    kv[2, :512], kv[2, 575] = 5, -100
    ids = torch.full((1, 2048), -1, device="cuda", dtype=torch.int32)
    ids[:, 65], ids[:, 1537] = 1, 2
    result = mla.sparse_mla_decode(q, kv, ids, scale)
    expected = 2 if scale > 0 else 5 if scale < 0 else 3.5
    torch.testing.assert_close(result, torch.full_like(result, expected), atol=0, rtol=0)


@pytest.mark.parametrize("heads", [64, 128])
def test_decode_graph_reads_changed_q_kv_and_selection(heads):
    require_sm90()
    torch.manual_seed(619)
    q = torch.randn(1, heads, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(257, 576, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, len(kv), (1, 2048), device="cuda", dtype=torch.int32)
    scale = 576**-0.5
    mla.sparse_mla_decode(q, kv, ids, scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = mla.sparse_mla_decode(q, kv, ids, scale)
    for turn in range(4):
        q.normal_()
        kv.normal_()
        if turn % 2:
            ids.fill_(-1)
        else:
            ids.random_(0, len(kv))
        graph.replay()
        torch.cuda.synchronize()
        expected = reference_sparse_mla(q, kv, ids, scale)
        torch.testing.assert_close(result, expected, atol=4e-3, rtol=2e-2)
        torch.testing.assert_close(result, mla.sparse_mla_decode(q, kv, ids, scale), atol=0, rtol=0)
    info = decode.runtime_info()
    assert info["specializations"]
    assert all({"ptx", "cubin"} <= row["artifact_sha256"].keys() for row in info["specializations"])


@pytest.mark.parametrize("queries", [0, 2])
def test_decode_rejects_non_single_query(queries):
    require_sm90()
    q = torch.zeros(queries, 128, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(5, 576, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(queries, 2048, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="exactly one query"):
        mla.sparse_mla_decode(q, kv, ids, 1)


def test_decode_model_workspace_covers_observed_peak():
    require_sm90()
    from models.deepseek_v32.execution.cache_resources import execution_reservation

    q = torch.zeros(1, 128, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(4096, 576, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(2048, device="cuda", dtype=torch.int64).view(1, -1)
    mla.sparse_mla_decode(q, kv, ids, 0.1)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = mla.sparse_mla_decode(q, kv, ids, 0.1)
    torch.cuda.synchronize()
    observed = torch.cuda.max_memory_allocated() - before
    reservation = execution_reservation(1, 4096, topk=2048, width=576)
    assert observed <= reservation.indexer_bytes + reservation.attention_extra_bytes == 4_816_896
    assert result.shape == (1, 128, 512)
