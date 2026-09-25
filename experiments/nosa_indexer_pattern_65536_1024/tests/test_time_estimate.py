"""Independent pair-count, per-layer timing, and decimal-bandwidth estimates."""

import json

import numpy as np
import pytest

from experiments.nosa_indexer_pattern_65536_1024.src.estimate import estimate
from experiments.nosa_indexer_pattern_65536_1024.tests.test_pattern_analyze import (
    selection_fixture,
)


@pytest.fixture
def tiny_case():
    ids, mask = selection_fixture()
    config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "head_dim": 3,
        "hidden_size": 18,
        "intermediate_size": 24,
    }
    modules = [
        {"attention_core": 8.0, "qkv_proj": 2.0, "o_proj": 1.0, "mlp": 7.0, "norm": 0.5},
        {"attention_core": 20.0, "qkv_proj": 3.0, "o_proj": 2.0, "mlp": 10.0, "norm": 0.75},
    ]
    timing = {
        "layers": [
            {"layer": index, "modules": values, "gpu_ms": sum(values.values())}
            for index, values in enumerate(modules)
        ],
        "shared_modules": {"embedding": 0.125, "final_norm": 0.375},
    }
    return (
        ids,
        mask,
        {
            "config": config,
            "execution": {"prefix_tokens": 8, "new_tokens": 4, "total_tokens": 12},
            "dense_timing": timing,
            "block_size": 4,
            "element_size": 2,
            "peak_tflops": 100.0,
            "bandwidth_gbps": 50.0,
        },
    )


def test_partial_blocks_gqa_and_each_layers_measured_mfu(tiny_case):
    ids, mask, params = tiny_case
    report = estimate(ids, mask, **params)
    # At positions 8..11, a selected block 2 contributes 1,2,3,4
    # visible tokens. Historical blocks each contribute four tokens.
    # L0/H0: 5+4+3+8=20, L0/H1: 4+4+4+4=16; layer 1 swaps these totals.
    assert report["sparse_query_key_pairs_per_kv_head"] == [[20, 16], [16, 20]]
    # Each KV head serves three Q heads: 36 pairs * 3 * 4 * d(3) = 1296 FLOPs.
    # Dense: (9+10+11+12) pairs * 6 Q heads * 4 * d(3) = 3024 FLOPs.
    for row, attention_ms, other_ms in zip(
        report["layers"], [8.0, 20.0], [10.5, 15.75], strict=True
    ):
        assert row["sparse_attention_flops"] == 1296
        assert row["dense_attention_flops"] == 3024
        assert row["attention_flops_ratio"] == pytest.approx(3 / 7)
        assert row["dense_attention_gpu_ms"] == attention_ms
        assert row["sparse_attention_gpu_ms"] == pytest.approx(attention_ms * 3 / 7)
        assert row["other_gpu_ms"] == other_ms
        assert row["sparse_layer_gpu_ms"] == pytest.approx(other_ms + attention_ms * 3 / 7)
        assert row["dense_attention_mfu_pct"] == pytest.approx(
            3024 / (attention_ms / 1000 * 100e12) * 100
        )
        # QKV + O + gate/up + down projections, preserved for each layer.
        assert row["projection_flops"] == 17280

    totals = report["totals"]
    assert totals["sparse_attention_gpu_ms"] == pytest.approx(12.0)
    assert totals["other_gpu_ms"] == pytest.approx(26.25)
    assert totals["sparse_layer_gpu_ms"] == pytest.approx(38.25)
    assert totals["shared_gpu_ms"] == pytest.approx(0.5)
    assert totals["sparse_model_gpu_ms_including_shared"] == pytest.approx(38.75)
    assert totals["mean_sparse_layer_gpu_ms"] == pytest.approx(38.25 / 2)
    assert totals["sparse_effective_matrix_flops"] == 2 * (17280 + 1296)
    assert report["dense_shared_modules"] == {"embedding": 0.125, "final_norm": 0.375}
    # No allocation of embedding/final norm to decoder layers or their mean.
    assert sum(row["sparse_layer_gpu_ms"] for row in report["layers"]) == pytest.approx(38.25)
    json.dumps(report, allow_nan=False)


def test_decimal_bandwidth_and_heads_share_the_layer_transfer_budget(tiny_case):
    ids, mask, params = tiny_case
    report = estimate(ids, mask, **params)
    assert report["parameters"]["bytes_per_second"] == 50_000_000_000
    # 4 tokens * d(3) * (K+V) * BF16(2) = 48 bytes per block/head.
    # Distinct layer/head unions contain 2,2,1,2 blocks; full cache has 3 each.
    assert [row["sparse_union_bytes"] for row in report["heads"]] == [96, 96, 48, 96]
    for row, selected_bytes in zip(report["layers"], [192, 144], strict=True):
        assert row["sparse_union_bytes"] == selected_bytes
        assert row["full_kv_bytes"] == 288
        assert row["sparse_fetch_ms"] == pytest.approx(selected_bytes / 50_000_000)
        assert row["full_fetch_ms"] == pytest.approx(288 / 50_000_000)
        head_ms = sum(
            head["sparse_fetch_ms"] for head in report["heads"] if head["layer"] == row["layer"]
        )
        assert row["sparse_fetch_ms"] == pytest.approx(head_ms)
    assert report["totals"]["sparse_fetch_ms"] == pytest.approx(336 / 50_000_000)
    assert report["totals"]["prefix_sparse_fetch_ms"] == pytest.approx(240 / 50_000_000)
    assert report["totals"]["prefix_full_fetch_ms"] == pytest.approx(384 / 50_000_000)


def test_union_bytes_and_attention_pairs_are_independent_metrics(tiny_case):
    _, _, params = tiny_case
    historical = np.zeros((2, 4, 2, 1), dtype=np.int32)
    varied_history = historical.copy()
    varied_history[:, 1::2, :, :] = 1
    partial_current = historical.copy()
    partial_current[:, 0, :, :] = 2
    mask = np.ones_like(historical, dtype=np.bool_)
    single = estimate(historical, mask, **params)
    varied = estimate(varied_history, mask, **params)
    partial = estimate(partial_current, mask, **params)
    for first, second, third in zip(
        single["layers"], varied["layers"], partial["layers"], strict=True
    ):
        # Each query still selects four historical tokens; a larger cross-query
        # union changes transferred bytes but cannot change attention FLOPs.
        assert first["sparse_attention_flops"] == second["sparse_attention_flops"] == 1152
        assert first["sparse_attention_gpu_ms"] == second["sparse_attention_gpu_ms"]
        assert second["sparse_fetch_ms"] == pytest.approx(2 * first["sparse_fetch_ms"])
        # Two selected blocks per union in both cases, but the first query's
        # current block now contributes one token instead of four.
        assert second["sparse_union_bytes"] == third["sparse_union_bytes"] == 192
        assert second["sparse_fetch_ms"] == third["sparse_fetch_ms"]
        assert third["sparse_attention_flops"] == 936
        assert third["sparse_attention_gpu_ms"] < second["sparse_attention_gpu_ms"]


@pytest.mark.parametrize(
    "block_budget,pairs_per_head,sparse_flops,union_blocks,fetch_ms",
    [
        (32, 2_064_896, 33_831_256_064, 47, 0.06160384),
        (64, 4_162_048, 68_190_994_432, 79, 0.10354688),
    ],
)
def test_65536_plus_1024_flops_and_full_fetch_golden_values(
    tiny_case, block_budget, pairs_per_head, sparse_flops, union_blocks, fetch_ms
):
    _, _, params = tiny_case
    positions = np.arange(65536, 66560)
    # Sink + 15/47 historical top blocks, then 16 local blocks including current.
    historical_blocks = block_budget - 16
    selected = np.concatenate(
        (
            np.broadcast_to(np.arange(historical_blocks), (1024, historical_blocks)),
            positions[:, None] // 64 + np.arange(-15, 1),
        ),
        axis=1,
    )
    ids = np.broadcast_to(selected[None, :, None, :], (2, 1024, 2, block_budget)).astype(np.int32)
    params.update(
        config={
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "hidden_size": 4096,
            "intermediate_size": 14336,
        },
        execution={"prefix_tokens": 65536, "new_tokens": 1024, "total_tokens": 66560},
        block_size=64,
    )
    report = estimate(ids, np.ones_like(ids, dtype=np.bool_), **params)
    # For each KV head: 31/63 full blocks/query plus causal fractions 1..64,
    # repeated across the 16 candidate blocks.
    assert report["sparse_query_key_pairs_per_kv_head"] == [[pairs_per_head, pairs_per_head]] * 2
    ratio = ((block_budget - 1) * 64 + 32.5) / (65536 + 512.5)
    for row in report["layers"]:
        assert row["sparse_attention_flops"] == sparse_flops
        assert row["dense_attention_flops"] == 1_108_109_950_976
        assert row["attention_flops_ratio"] == pytest.approx(ratio, rel=1e-14)
        assert row["sparse_attention_gpu_ms"] == pytest.approx(
            row["dense_attention_gpu_ms"] * ratio
        )
        assert row["full_kv_bytes"] == 68_157_440
        assert row["full_fetch_ms"] == pytest.approx(1.3631488)
        # Local union is 1009..1039 (31 blocks); the disjoint historical set
        # has 16/48 blocks. This construction reaches the 47/79-block lower bound.
        assert row["sparse_union_bytes"] == union_blocks * 32768 * 2
        assert row["sparse_fetch_ms"] == pytest.approx(fetch_ms)
        assert row["attention_flops_ratio"] != pytest.approx(
            row["sparse_union_bytes"] / row["full_kv_bytes"]
        )


@pytest.mark.parametrize("bandwidth", [0.0, -1.0, float("nan"), float("inf")])
def test_bandwidth_must_be_positive_and_finite(tiny_case, bandwidth):
    ids, mask, params = tiny_case
    params["bandwidth_gbps"] = bandwidth
    with pytest.raises(ValueError, match="bandwidth"):
        estimate(ids, mask, **params)


@pytest.mark.parametrize(
    "scope,module,value",
    [
        ("layer", "attention_core", 0.0),
        ("layer", "mlp", -1.0),
        ("shared", "embedding", -1.0),
        ("shared", "final_norm", float("nan")),
    ],
)
def test_bad_module_timing_is_rejected(tiny_case, scope, module, value):
    ids, mask, params = tiny_case
    timing = params["dense_timing"]
    if scope == "layer":
        row = timing["layers"][0]
        row["modules"][module] = value
        row["gpu_ms"] = sum(row["modules"].values())
    else:
        timing["shared_modules"][module] = value
    with pytest.raises(ValueError, match="(time|duration|positive|finite|nonnegative)"):
        estimate(ids, mask, **params)
