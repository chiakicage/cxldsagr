"""Input construction and measurement arithmetic, without GPU execution."""

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.cache_manager_performance.src import workload
from experiments.cache_manager_performance.src.analyze import category, interval_metrics
from experiments.cache_manager_performance.src.workload import Config, phase_input


def test_prefill_repeats_queries_but_truncates_captured_kv_and_labels_synthetic():
    cfg = Config(history=8, append=2, chunk=4, layers=1)
    capture = {
        "kv": torch.arange(10)[:, None],
        "index_keys": torch.arange(10)[:, None],
        "index_scales": torch.arange(10),
        "q": torch.tensor([[11], [12]]),
        "index_q": torch.tensor([[21], [22]]),
        "index_weights": torch.tensor([[31], [32]]),
    }
    prefill = phase_input(capture, "prefill_last_chunk", cfg)
    assert (prefill["start"], prefill["count"], prefill["end"]) == (4, 4, 8)
    assert prefill["q"].flatten().tolist() == [11, 12, 11, 12]
    assert len(prefill["kv"]) == 8
    assert prefill["input_kind"] == "synthetic_repeated_extend_queries"
    extend = phase_input(capture, "extend_cold", cfg)
    assert (extend["start"], extend["count"], extend["end"]) == (8, 2, 10)
    assert extend["input_kind"] == "captured_extend_queries"


def test_control_remains_overhead_and_overlapping_io_is_unioned():
    rows = [
        {"start": 0, "end": 40, "category": "indexer_compute"},
        {"start": 20, "end": 60, "category": "io"},
        {"start": 50, "end": 80, "category": "d2d_control"},
        {"start": 70, "end": 90, "category": "cache_management"},
    ]
    result = interval_metrics(rows, 0, 100)
    assert result["gpu_idle_ms"] == 10 / 1e6
    assert result["outside_indexer_and_io_ms"] == 40 / 1e6
    assert result["window_minus_known_io_ms"] == 60 / 1e6
    assert result["manager_selection_control_union_ms"] == 40 / 1e6
    assert result["whole_model_gap_ratio"] is None
    assert result["mfu"] is None


def test_fused_internal_io_is_not_invented_and_memcpy_d2d_remains_control():
    row = {
        "kind": "kernel",
        "name": "sm90_fp8_mqa_logits_fuse_prefetch",
        "scope": {"stage": "window/layer_0/indexer_prefetch"},
    }
    assert category(row) == "indexer_compute_and_io"
    result = interval_metrics([{**row, "category": category(row), "start": 0, "end": 50}], 0, 100)
    assert result["known_io_union_ms"] == 0
    assert result["fused_internal_io_unresolved"]
    assert category({**row, "kind": "memcpy", "name": "DtoD"}) == "d2d_control"
    assert category({**row, "kind": "memcpy", "name": "DtoH"}) == "io"
    assert category({**row, "kind": "memcpy", "name": "Host-to-Device"}) == "io"
    assert category({**row, "kind": "memcpy", "name": "Device-to-Host"}) == "io"
    assert category({**row, "name": "echo_native::clean"}) == "indexer_compute"
    for name in ("masked_fill_kernel", "CompareFunctor<long>"):
        assert category({**row, "name": name, "scope": {"stage": "window/indexer"}}) == (
            "indexer_compute"
        )
    assert category({**row, "name": "FillFunctor<int>"}) == "cache_management"


def test_invalid_shapes_and_profile_windows_fail():
    with pytest.raises(ValueError):
        Config(history=7, chunk=4)
    with pytest.raises(ValueError):
        interval_metrics([], 4, 4)


def test_capture_source_and_layer_mismatch_are_rejected(tmp_path, monkeypatch):
    config = Config(history=8, append=2, chunk=4, layers=1)
    source = {
        "accepted": True,
        "run_id": "fixture",
        "prefix_tokens": 8,
        "extend_tokens": 2,
        "num_layers": 1,
    }
    (tmp_path / "result.json").write_text(json.dumps(source))
    (tmp_path / "kernel_inputs_layer_0.pt").write_bytes(b"fixture")
    data = {
        "layer": 1,
        "source_run_id": "fixture",
        "query_start": 8,
        "q": torch.zeros(2, 1),
        "kv": torch.zeros(10, 1),
    }
    monkeypatch.setattr(workload, "load_inputs", lambda path: data)
    with pytest.raises(ValueError, match="layer or source"):
        workload.load_workload(tmp_path, config)
    data.update(layer=0, source_run_id="different")
    with pytest.raises(ValueError, match="layer or source"):
        workload.load_workload(tmp_path, config)
    data["source_run_id"] = "fixture"
    _, identities = workload.load_workload(tmp_path, config)
    assert identities[0]["source_result_sha256"]
    assert identities[0]["source_run_id"] == "fixture"


def _completed_replay(*, offload):
    """Small completed states exercise the untimed oracle without running MLA."""
    replay = workload.Replay.__new__(workload.Replay)
    replay.phase = "extend_cold"
    replay.device = torch.device("cpu")
    indices = torch.tensor([[1, 0, -1], [2, 0, 1]], dtype=torch.int32)
    kv = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16)
    attention = torch.arange(8, dtype=torch.bfloat16).view(2, 2, 2)
    replay.reference_indices = [indices.clone()]
    replay.inputs = [
        {
            "start": 1,
            "count": 2,
            "end": 3,
            "q": torch.zeros(2, 2, 3, dtype=torch.bfloat16),
            "kv": kv,
        }
    ]
    cache = SimpleNamespace(records=kv.clone(), metrics=dict)
    replay.pool = None
    if offload:
        # Logical IDs 0,1,2 live on host page 1 and permuted GPU slots 2,3,1.
        h2d = torch.full((128,), workload.MISSING, dtype=torch.int32)
        h2d[64:67] = torch.tensor([2, 3, 1], dtype=torch.int32)
        d2h = torch.tensor([workload.MISSING, 66, 64, 65], dtype=torch.int64)
        cache.records = torch.cat((torch.zeros_like(kv[:1]), kv[[2, 0, 1]]))
        cache.page_table = torch.tensor([1], dtype=torch.int32)
        cache.host_to_device = h2d
        replay.pool = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    host_to_device=h2d,
                    device_to_host=d2h,
                    free=torch.tensor([False, False, False, False]),
                    clock=1,
                    clock_tensor=torch.tensor(1),
                )
            ]
        )
    replay.runners = [SimpleNamespace(cache=cache, cfg=SimpleNamespace(kv_lora_rank=2))]
    replay.outputs = [(indices, attention)]
    return replay


def test_completed_attention_oracle_checks_remapped_kv_and_actual_outputs():
    _, reference = _completed_replay(offload=False).verify()
    replay = _completed_replay(offload=True)
    before = replay.runners[0].cache.host_to_device.clone()
    checks, evidence = replay.verify(reference)
    assert checks[0]["attention_matches_resident"]
    assert checks[0]["attention_finite"]
    assert checks[0]["selected_record_source"] == "post_consume_cache_maps"
    assert checks[0]["unique_selected"] == 3
    assert set(evidence) == {"indices", "attention"}
    assert torch.equal(evidence["attention"][0], reference["attention"][0])
    assert torch.equal(replay.runners[0].cache.host_to_device, before)


def test_attention_mismatch_is_rejected_even_with_correct_kv_and_maps():
    _, reference = _completed_replay(offload=False).verify()
    replay = _completed_replay(offload=True)
    replay.outputs[0][1][0, 0, 0] += 1
    with pytest.raises(AssertionError):
        replay.verify(reference)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_attention_cannot_pass_without_a_resident_reference(value):
    replay = _completed_replay(offload=False)
    replay.outputs[0][1][0, 0, 0] = value
    with pytest.raises(AssertionError, match="non-finite"):
        replay.verify()


@pytest.mark.parametrize("shape,dtype", [((2, 2, 3), torch.bfloat16), ((2, 2, 2), torch.float32)])
def test_attention_output_contract_is_checked_without_a_reference(shape, dtype):
    replay = _completed_replay(offload=False)
    replay.outputs[0] = (replay.outputs[0][0], torch.zeros(shape, dtype=dtype))
    with pytest.raises(AssertionError, match="shape or dtype"):
        replay.verify()


@pytest.mark.parametrize("slot", [workload.MISSING, -1, 0, 4])
def test_completed_selection_requires_valid_published_pool_slots(slot):
    replay = _completed_replay(offload=True)
    replay.runners[0].cache.host_to_device[64] = slot
    with pytest.raises(AssertionError, match="valid published pool slot"):
        replay.verify()


def test_selected_kv_corruption_is_rejected_independently_of_attention():
    replay = _completed_replay(offload=True)
    replay.runners[0].cache.records[2, 0] += 1
    with pytest.raises(AssertionError):
        replay.verify()


def test_inverse_mapping_corruption_is_rejected():
    replay = _completed_replay(offload=True)
    replay.pool.layers[0].device_to_host[1] = 64
    with pytest.raises(AssertionError, match="not inverse"):
        replay.verify()


def test_incomplete_attention_reference_is_rejected():
    replay = _completed_replay(offload=False)
    with pytest.raises(ValueError, match="one selection and output per layer"):
        replay.verify({"indices": replay.reference_indices})
