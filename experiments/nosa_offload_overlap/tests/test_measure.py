import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.nosa_offload_overlap.src.measure import expected_fetch_rows, prefix_transfer_bytes


def test_transfer_bytes_deduplicate_across_queries_tiles_and_handle_partial_block():
    ids = torch.tensor(
        [
            [[0, 1, 1], [2, -1, 0]],
            [[1, 2, 4], [2, 2, 4]],
            [[2, 0, 1], [0, 1, 2]],
        ]
    )
    valid = torch.ones_like(ids, dtype=torch.bool)
    total, tiles = prefix_transfer_bytes(ids, valid, 130, 4, 2, tile_size=2)
    assert total == 2 * 130 * 4 * 2 * 2
    assert tiles == [(130 + 66) * 4 * 2 * 2, 64 * 4 * 2 * 2]


def test_masked_blocks_and_suffix_are_not_transferred_and_padding_page_is_early():
    ids = torch.tensor([[[1, 2]], [[0, 0]]])
    valid = torch.tensor([[[False, True]], [[True, False]]])
    total, tiles = prefix_transfer_bytes(ids, valid, 128, 128, 2, tile_size=1)
    assert total == 64 * 128 * 2 * 2
    assert tiles == [total, 0]


def test_expected_fetch_rows_preserve_block_major_head_index_and_partial_bytes():
    ids = torch.tensor([[[0, 2, 3], [1, 2, 3]], [[2, 0, 3], [1, 2, 3]]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[:, 1, 1] = False
    assert expected_fetch_rows(ids, valid, 130) == [
        {"row": 0, "bytes": 32768},
        {"row": 3, "bytes": 32768},
        {"row": 4, "bytes": 1024},
    ]


def test_fetch_cta_cli_default_and_override():
    from experiments.nosa_offload_overlap.src.measure import parser

    args = ["--run-id", "fixture", "--output-dir", "/tmp/fixture", "--synthetic"]
    assert parser().parse_args(args).fetch_ctas == 96
    assert parser().parse_args([*args, "--fetch-ctas", "12"]).fetch_ctas == 12


def test_metadata_includes_actual_fused_build(monkeypatch):
    from experiments.nosa_offload_overlap.src import measure

    fused = {
        "fused_source_sha256": "kernel-sha",
        "fused_wrapper_sha256": "wrapper-sha",
        "fused_prefill_header_sha256": "header-sha",
        "cuda_flags": ["-gencode=arch=compute_90a,code=sm_90a"],
    }
    monkeypatch.setattr(measure, "build_info", lambda: {"selected_backend": "native"})
    monkeypatch.setitem(
        sys.modules,
        "operators.nosa.attention.offload._fused",
        SimpleNamespace(build_info=lambda: fused),
    )
    assert measure.native_build_metadata() == {"selected_backend": "native", "offload_fused": fused}


def test_source_snapshot_covers_cooperative_runtime_and_planner():
    from experiments.nosa_baseline_performance.src.dense.sources import source_hashes

    hashes = source_hashes()
    assert {
        "operators/nosa/attention/offload/_fused.py",
        "operators/nosa/attention/offload/api.py",
        "operators/nosa/attention/offload/csrc/nosa_offload_fused.cu",
        "operators/nosa/attention/offload/csrc/nosa_offload.cu",
        "operators/nosa/attention/device_only/csrc/nosa_attention_fa3.cu",
        "operators/nosa/attention/device_only/csrc/nosa_attention.cu",
        "operators/nosa/attention/device_only/csrc/nosa_attention_grouped.cuh",
    } <= hashes.keys()


def test_work_profile_stripe_geometry_comes_from_actual_native_build():
    from experiments.nosa_offload_overlap.src.analyze import work_profile_metadata
    from experiments.nosa_offload_overlap.src.measure import new_work_profile

    for stripes in (2, 8, 16):
        profile = new_work_profile("fixture", {"offload_fused": {"fetch_stripes": stripes}})
        assert work_profile_metadata(profile)["fetch_stripes"] == stripes
        assert profile["schema_version"] == 3
        assert profile["cases"] == {} and profile["records"] == []
    with pytest.raises(ValueError, match="fetch_stripes"):
        new_work_profile("fixture", {"selected_backend": "native"})


def test_clean_offload_bench_never_reads_traffic_or_runs_reference(monkeypatch):
    from experiments.nosa_offload_overlap.src import measure

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent check leaked into clean performance samples")

    class Workspace:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, q, *args):
            return q

        def synchronize(self):
            pass

        @property
        def last_transfer_bytes(self):
            forbidden()

    monkeypatch.setitem(
        sys.modules,
        "operators.nosa.attention.offload.api",
        SimpleNamespace(NosaFetchWorkspace=Workspace),
    )
    monkeypatch.setattr(measure, "copy_to_device", lambda tensor, device: tensor)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(
        measure.torch.cuda,
        "Event",
        lambda **kwargs: SimpleNamespace(
            record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 2.0
        ),
    )
    monkeypatch.setattr(measure.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(measure.torch.cuda.nvtx, "range_push", lambda name: None)
    monkeypatch.setattr(measure.torch.cuda.nvtx, "range_pop", lambda: None)
    monkeypatch.setattr(measure, "nosa_block_sparse_attention", lambda q, *args: q)
    monkeypatch.setattr(measure, "reference_acceptance", forbidden)
    monkeypatch.setattr(measure, "_validate_case", forbidden)
    monkeypatch.setattr(measure.torch.testing, "assert_close", forbidden)
    cpu = {
        "q": torch.zeros((8, 32, 128), dtype=torch.bfloat16),
        "k": torch.zeros((72, 2, 128), dtype=torch.bfloat16),
        "v": torch.zeros((72, 2, 128), dtype=torch.bfloat16),
        "cis": torch.zeros((72, 2), dtype=torch.bfloat16),
        "ids": torch.zeros((8, 2, 64), dtype=torch.int32),
        "valid_mask": torch.zeros((8, 2, 64), dtype=torch.bool),
    }
    tensor_identity = {
        name: measure.tensor_metadata(cpu[name]) for name in measure.required_order()
    }
    args = SimpleNamespace(
        mode="bench",
        queries=8,
        device="cpu",
        tile_size=8,
        fetch_ctas=2,
        profiled=False,
        warmup=2,
        repeats=3,
        synthetic=False,
        validation_cases={
            "fixture": {
                "tensors": tensor_identity,
                "acceptance": {"rows": list(range(8))},
                "serialized_overlap_exact_equal": True,
            }
        },
    )
    result = measure.benchmark_case(cpu, args, label="fixture", prefix=64)
    assert result["traffic_checks"] == "independent_check"
    assert result["numerical_acceptance_source"] == "independent_check"
    assert all(len(result["modes"][mode]["cuda_ms"]["samples"]) == 3 for mode in measure.MODES)
