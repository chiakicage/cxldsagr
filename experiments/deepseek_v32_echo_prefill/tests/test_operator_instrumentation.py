"""Validate diagnostic metadata on CPU without CUDA launches or value reads."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from experiments.deepseek_v32_echo_prefill.src.operator_instrumentation import (
    InstrumentOperators,
    OperatorScopes,
    QueryOrigin,
    mla_call_details,
)


@pytest.fixture
def nvtx(monkeypatch):
    active, visited = [], []

    @contextmanager
    def scope(label):
        active.append(label)
        visited.append(label)
        try:
            yield
        finally:
            assert active.pop() == label

    monkeypatch.setattr(torch.cuda.nvtx, "range", scope)
    return active, visited


def test_layer_misc_is_leaf_and_layer_context_restores_after_failure(nvtx):
    active, visited = nvtx
    scope = OperatorScopes("resident", "extend_annotated")
    with scope("layer_0"):
        assert scope.layer == "layer_0"
        with scope("indexer"):
            pass
    with pytest.raises(RuntimeError), scope("layer_2"):
        raise RuntimeError("diagnostic failure")
    assert scope.layer == "shared"
    assert active == []
    assert visited == [
        "echo/resident/extend_annotated/layer_0",
        "echo/resident/extend_annotated/layer_0/layer_misc",
        "echo/resident/extend_annotated/layer_0/indexer_aux",
        "echo/resident/extend_annotated/layer_2",
        "echo/resident/extend_annotated/layer_2/layer_misc",
    ]
    assert [row["stage"] for row in scope.calls] == ["layer_misc", "indexer_aux", "layer_misc"]
    assert all(row["useful_flops"] is None and row["precision"] is None for row in scope.calls)
    assert all(row["executed_matmul_flops"] is None for row in scope.calls)


def test_recursive_query_slices_resolve_from_original_view_not_storage_base():
    storage = torch.empty((19, 4, 16), dtype=torch.bfloat16)
    projected = storage[3:15]
    origin = QueryOrigin.capture(projected, 65536)
    assert origin.resolve_start(projected) == 65536
    assert origin.resolve_start(projected[6:][2:5]) == 65544
    assert origin.resolve_start(projected[:6][3:]) == 65539
    assert all(not isinstance(value, torch.Tensor) for value in vars(origin).values())
    with pytest.raises(ValueError, match="unit-step"):
        origin.resolve_start(projected[::2])
    with pytest.raises(ValueError, match="unit-step"):
        origin.resolve_start(projected.clone())
    with pytest.raises(ValueError, match="does not align"):
        origin.resolve_start(storage[2:4])
    with pytest.raises(ValueError, match="outside"):
        origin.resolve_start(storage[14:16])


def test_causal_sparse_metadata_matches_selection_and_conserves_split_work(monkeypatch):
    q = torch.empty((9, 64, 576), dtype=torch.bfloat16)
    origin = QueryOrigin.capture(q, 2)
    selection = torch.full((9, 7), -1, dtype=torch.int32)
    for row in range(9):
        # Physical remapping changes IDs, but preserves the exact causal validity mask.
        count = min(2 + row + 1, 7)
        selection[row, :count] = torch.arange(count) + 100
    observed_valid = int((selection >= 0).sum())

    def forbidden(*args, **kwargs):
        raise AssertionError("annotation must not copy or inspect tensor values")

    for name in ("item", "cpu", "numpy", "to", "sum"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    whole = mla_call_details(q, selection, origin, 512)
    left = mla_call_details(q[:4], selection[:4], origin, 512)
    right = mla_call_details(q[4:], selection[4:], origin, 512)
    assert whole["valid_selected_pairs"] == observed_valid
    assert left["query_start"] == 2 and right["query_start"] == 6
    assert left["useful_flops"] + right["useful_flops"] == whole["useful_flops"]
    assert (
        left["executed_matmul_flops"] + right["executed_matmul_flops"]
        == whole["executed_matmul_flops"]
    )
    assert whole["precision"] == "BF16"
    assert "QK:" in whole["formula"] and "PV:" in whole["formula"]


def test_shape_only_wrappers_keep_stage_names_precision_and_restore_targets(nvtx, monkeypatch):
    from models.deepseek_v32 import echo_attention, echo_block, echo_infer, echo_model
    from operators.deepseek_v32.attention.offload import mla as offload_mla
    from operators.deepseek_v32.indexer import echo

    def checkpoint_linear():
        linear = object.__new__(echo_model.CheckpointLinear)
        linear.weight = torch.ones((7, 5), dtype=torch.bfloat16)
        linear.scales = None
        return linear

    attn = SimpleNamespace(
        **{
            name: checkpoint_linear()
            for name in ("wq_a", "wq_b", "wkv_a", "index_wq", "index_wk", "wo")
        },
        wk_b=torch.ones((2, 4, 6), dtype=torch.bfloat16),
        wv_b=torch.ones((2, 6, 7), dtype=torch.bfloat16),
        index_head_weight=torch.ones((64, 5), dtype=torch.float32),
    )
    model = SimpleNamespace(
        blocks=[
            SimpleNamespace(
                attention=SimpleNamespace(attention=attn),
                mlp=SimpleNamespace(
                    **{name: checkpoint_linear() for name in ("gate", "up", "down")}
                ),
            )
        ],
        head_weight=torch.ones((13, 5), dtype=torch.bfloat16),
    )
    projected = SimpleNamespace(q=torch.empty((3, 128, 576), dtype=torch.bfloat16))
    monkeypatch.setattr(echo_model.CheckpointAttention, "project", lambda *a, **k: projected)
    monkeypatch.setattr(echo_attention, "sparse_mla", lambda q, *a: q)
    monkeypatch.setattr(echo, "logits", lambda q, *a, **k: q)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    original_bmm, original_linear = torch.bmm, F.linear
    scope = OperatorScopes("resident", "extend_annotated")
    x = torch.ones((3, 5), dtype=torch.bfloat16)
    expected_linear = attn.wq_a(x)
    with InstrumentOperators(model, scope), scope("layer_0"):
        actual_linear = attn.wq_a(x)
        torch.bmm(torch.ones((2, 3, 4), dtype=torch.bfloat16), attn.wk_b)
        F.linear(x.float(), attn.index_head_weight)
        F.linear(x[-1:], model.head_weight)
        echo_model.CheckpointAttention.project(attn, x, 128)
        q = torch.empty((3, 64, 128), dtype=torch.float8_e4m3fn)
        k = torch.empty((131, 128), dtype=torch.float8_e4m3fn)
        echo.logits(q, k, None, None, 128)
        echo.logits(q, k, None, None, 128, prefetch={})
        ids = torch.zeros((3, 3), dtype=torch.int32)
        echo_attention.sparse_mla(projected.q, None, ids, 1.0)
        offload_mla.sparse_mla_from_pool(projected.q[1:], None, ids[1:], 1.0)
        norm_weight = torch.ones(5)
        echo_model.rms_norm(x, norm_weight, 1e-6)
        echo_infer.rms_norm(x, norm_weight, 1e-6)
        echo_block.residual_rms_norm(x, x, norm_weight, 1e-6)
        echo_block.silu_mul(x, x)
        angles = torch.zeros(3, 2)
        rotary_cache = echo_model.prepare_rotary_cache(angles)
        echo_model.apply_rope_pair(
            x[:, None, :4], x[:, None, :4], angles, interleaved=True, cache=rotary_cache
        )
    assert torch.bmm is original_bmm and F.linear is original_linear
    torch.testing.assert_close(actual_linear, expected_linear)
    calls = {row["stage"]: row for row in scope.calls if row["stage"] != "mla_qk_pv"}
    assert calls["q_a_proj"]["precision"] == "BF16"
    assert calls["index_weights_proj"]["precision"] == "FP32"
    assert calls["lm_head"]["shape"] == [1, 13, 5]
    assert calls["indexer_qk"]["useful_flops"] == calls["indexer_fused"]["useful_flops"]
    assert calls["indexer_qk"]["executed_matmul_flops"] > calls["indexer_qk"]["useful_flops"]
    for stage in (
        "rms_norm",
        "residual_rms_norm",
        "silu_mul",
        "prepare_rotary_cache",
        "apply_rope_pair",
    ):
        assert calls[stage]["useful_flops"] is None
        assert calls[stage]["executed_matmul_flops"] is None
    mla = [row for row in scope.calls if row["stage"] == "mla_qk_pv"]
    assert [row["query_start"] for row in mla] == [128, 129]
    assert all("formula" in row and "notes" in row for row in scope.calls)


def test_old_control_helpers_remain_instrumentable(nvtx, monkeypatch):
    from models.deepseek_v32 import echo_block, echo_model

    for name in ("apply_rope_pair", "prepare_rotary_cache"):
        monkeypatch.delattr(echo_model, name)
    for name in ("residual_rms_norm", "silu_mul"):
        monkeypatch.delattr(echo_block, name)
    original = echo_model.rms_norm
    monkeypatch.setattr(echo_block, "rms_norm", original, raising=False)
    model = SimpleNamespace(blocks=[], head_weight=torch.ones(2, 3))
    scope = OperatorScopes("resident", "extend_annotated")
    with InstrumentOperators(model, scope):
        echo_block.rms_norm(torch.ones(2, 3), torch.ones(3), 1e-6)
    assert echo_block.rms_norm is original
    assert [row["stage"] for row in scope.calls] == ["rms_norm"]


@torch.inference_mode()
def test_packed_mlp_keeps_three_matrix_scopes_and_quantization_attribution(nvtx, monkeypatch):
    from models.deepseek_v32.echo_block import CheckpointMLP
    from models.deepseek_v32.echo_model import CheckpointLinear
    from operators.deepseek_v32.linear import fp8

    active, _ = nvtx
    mlp = object.__new__(CheckpointMLP)
    mlp.pack_gate_up = True
    labels = {}
    for name, shape in (("gate", (64, 128)), ("up", (64, 128)), ("down", (128, 64))):
        linear = object.__new__(CheckpointLinear)
        linear.weight = torch.ones(shape).to(torch.float8_e4m3fn)
        linear.scales = torch.ones(1, 1)
        setattr(mlp, name, linear)
        labels[linear.weight.data_ptr()] = name
    # Exercise call plumbing on CPU; native numerical checks live with the operators.
    monkeypatch.setattr(mlp, "_can_pack_gate_up", lambda _: True)
    prepared_records, quantizations, gemms = [], [], []

    def adapter(x, weight, scales, *, out=None, quantized=None, return_quantized=False):
        label = labels[weight.data_ptr()]
        assert active[-1].endswith("/mlp_" + label)
        if quantized is None:
            quantizations.append(label)
            quantized = object()
        else:
            assert label == "up" and quantized is prepared_records[-1]
        gemms.append(label)
        result = torch.ones(len(x), len(weight), dtype=torch.bfloat16) if out is None else out
        result.fill_(1)
        if return_quantized:
            assert label == "gate"
            prepared_records.append(quantized)
            return result, quantized
        return result

    monkeypatch.setattr(fp8, "fp8_linear", adapter)
    scopes = OperatorScopes("resident", "extend_annotated")
    model = SimpleNamespace(blocks=[], head_weight=torch.ones(1, 1))
    instrument = InstrumentOperators(model, scopes)
    instrument.linears.update({id(getattr(mlp, name)): "mlp_" + name for name in labels.values()})
    with instrument, scopes("layer_0"):
        first = mlp(torch.ones(3, 128).bfloat16())
        second = mlp(torch.ones(3, 128).bfloat16())
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    assert prepared_records[0] is not prepared_records[1]
    assert quantizations == ["gate", "down", "gate", "down"]
    assert gemms == ["gate", "up", "down", "gate", "up", "down"]
    records = [row for row in scopes.calls if row["stage"].startswith("mlp_")]
    assert [row["stage"] for row in records] == ["mlp_" + name for name in gemms]
    assert all(row["useful_flops"] == 2 * 3 * 64 * 128 for row in records)
    assert [row["shape"] for row in records[:3]] == [[3, 64, 128], [3, 64, 128], [3, 128, 64]]
    assert sum(row["stage"] == "silu_mul" for row in scopes.calls) == 2
    assert set(vars(mlp)) == {"pack_gate_up", "gate", "up", "down", "_can_pack_gate_up"}
