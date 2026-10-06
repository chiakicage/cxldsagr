"""Dependency and boundary checks for the analytical single-layer schedules."""

import copy

import pytest

from experiments.deepseek_v32_motivation.src.simulate import (
    audit_simulation,
    compute_metrics,
    simulate,
    transfer_metrics,
)


def inputs(*, sparse=3, dense=30, writeback=2):
    costs = {"projection": 2, "index": 5, "topk": 1, "attention": 3, "finish": 4}
    return {
        "run_id": "test",
        "prefill": {
            "compute": {"stages_ns": costs.copy(), "total_ns": 15},
            "d2h": {"duration_ns": writeback},
        },
        "extend": {
            "compute": {"stages_ns": costs.copy(), "total_ns": 15},
            "sparse_h2d": {"duration_ns": sparse},
            "dense_h2d": {"duration_ns": dense},
            "dense_next_h2d": {"duration_ns": dense + 1},
            "dense_pipeline_context": {
                "previous_layer": 0,
                "compute": {"stages_ns": costs.copy(), "total_ns": 15},
                "h2d": {"duration_ns": dense},
            },
        },
    }


def scenarios(**kwargs):
    return {row["id"]: row for row in simulate(inputs(**kwargs))["scenarios"]}


def event(row, stage):
    return next(e for e in row["events"] if e["stage"] == stage)


def test_sparse_requires_exact_selection_unless_oracle_is_explicit():
    rows = scenarios()
    serial = rows["extend_serial_sparse"]
    oracle = rows["extend_oracle_sparse"]
    assert event(serial, "h2d")["start_ns"] == event(serial, "topk")["end_ns"]
    assert event(serial, "attention")["start_ns"] == event(serial, "h2d")["end_ns"]
    assert serial["completion_ns"] == 18
    assert event(oracle, "h2d")["start_ns"] == event(oracle, "index")["start_ns"]
    assert oracle["completion_ns"] == rows["extend_hbm"]["completion_ns"] == 15


def test_oracle_preserves_io_tail_when_transfer_exceeds_indexer():
    row = scenarios(sparse=11)["extend_oracle_sparse"]
    assert event(row, "topk")["start_ns"] == 13
    assert row["completion_ns"] == 21


@pytest.mark.parametrize(("dense", "expected"), [(4, 15), (11, 15), (16, 16), (30, 30)])
def test_dense_keeps_projection_origin_and_waits_only_for_remaining_fetch(dense, expected):
    row = scenarios(dense=dense)["extend_dense_prefetch"]
    assert event(row, "projection")["start_ns"] == 0
    current, following = row["full_transfers"]
    assert current["start_ns"] == row["previous_fetch_end_ns"]
    assert following["start_ns"] == current["end_ns"]
    assert event(row, "attention")["start_ns"] >= current["end_ns"]
    assert row["completion_ns"] == expected
    assert all(0 <= e["start_ns"] < e["end_ns"] <= expected for e in row["events"])
    if dense == 4:
        assert current["end_ns"] < 0
        assert row["io_ready_ns"] == 0
        assert not any(e["lane"] in {"wait", "io"} for e in row["events"])


def test_dense_clips_next_fetch_at_current_output_and_removes_separate_scenario():
    rows = scenarios()
    assert "extend_dense_lookahead" not in rows
    assert "extend_dense_cold" not in rows
    row = rows["extend_dense_prefetch"]
    assert row["compute_ready_ns"] == row["completion_ns"] == 30
    assert row["io_ready_ns"] == 23
    assert row["full_transfers"][1]["end_ns"] == 54
    io = [e for e in row["events"] if e["lane"] == "io"]
    assert [(e["start_ns"], e["end_ns"]) for e in io] == [(0, 23), (23, 30)]
    assert io[1]["transfer_role"] == "next_layer_context"


def test_dense_uses_measured_adjacent_layer_costs_instead_of_a_uniform_period():
    source = inputs(dense=1_374_494)
    current = dict(
        zip(
            ("projection", "index", "topk", "attention", "finish"),
            (143040, 270816, 83392, 120672, 203200),
            strict=True,
        )
    )
    previous = dict(zip(current, (141440, 268160, 84128, 121920, 206303), strict=True))
    source["extend"]["compute"] = {"stages_ns": current, "total_ns": 821120}
    source["extend"]["dense_next_h2d"]["duration_ns"] = 1_374_430
    source["extend"]["dense_pipeline_context"].update(
        {
            "compute": {"stages_ns": previous, "total_ns": 821951},
            "h2d": {"duration_ns": 1_373_758},
        }
    )
    row = next(r for r in simulate(source)["scenarios"] if r["id"] == "extend_dense_prefetch")
    assert row["prefetch_lead_ns"] == 328223
    assert row["full_transfers"][0]["start_ns"] == -328223
    assert row["io_ready_ns"] == 1046271
    assert row["wait_ns"] == 549023
    assert row["completion_ns"] == 1370143
    assert row["full_transfers"][1]["end_ns"] == 2420701


def test_dense_audit_rejects_clipped_service_times_or_shifted_origin():
    source = inputs()
    document = simulate(source)
    row = next(r for r in document["scenarios"] if r["id"] == "extend_dense_prefetch")
    row["full_transfers"][0]["duration_ns"] = 23
    with pytest.raises(ValueError, match="full-transfer"):
        audit_simulation(document, source)
    row["full_transfers"][0]["duration_ns"] = 30
    event(row, "projection")["start_ns"] = 1
    event(row, "projection")["end_ns"] = 3
    event(row, "index")["start_ns"] = 3
    event(row, "index")["end_ns"] = 8
    event(row, "topk")["start_ns"] = 8
    event(row, "topk")["end_ns"] = 9
    with pytest.raises(ValueError, match="time origin"):
        audit_simulation(document, source)


@pytest.mark.parametrize(("writeback", "expected"), [(2, 15), (20, 22)])
def test_prefill_drains_writeback_even_after_output_is_ready(writeback, expected):
    rows = scenarios(writeback=writeback)
    row = rows["prefill_overlap"]
    assert event(row, "d2h")["start_ns"] == event(row, "projection")["end_ns"]
    assert row["completion_ns"] == expected
    assert rows["prefill_serial"]["completion_ns"] == 15 + writeback


def test_zero_io_cost_has_no_overhead_and_inputs_are_unchanged():
    source = inputs(sparse=0, dense=0, writeback=0)
    before = copy.deepcopy(source)
    rows = simulate(source)["scenarios"]
    assert all(row["completion_ns"] == 15 for row in rows)
    assert source == before


def test_malformed_costs_fail_instead_of_producing_a_report():
    source = inputs()
    source["extend"]["compute"]["total_ns"] = 99
    with pytest.raises(ValueError, match="conserve"):
        simulate(source)
    source = inputs(sparse=-1)
    with pytest.raises(ValueError, match="negative"):
        simulate(source)


def compute_with_work():
    compute = inputs()["extend"]["compute"]
    work = {"dense_peaks_tflops": {"FP8": 2.0, "BF16": 1.0}, "stages": {}}
    for stage in compute["stages_ns"]:
        if stage == "topk":
            work["stages"][stage] = {"flops_by_precision": {}, "ideal_compute_ns": None}
        else:
            precision, flops = ("BF16", 1000) if stage == "attention" else ("FP8", 2000)
            work["stages"][stage] = {
                "flops_by_precision": {precision: flops},
                "ideal_compute_ns": 1,
            }
    work["total"] = {"flops_by_precision": {"FP8": 6000, "BF16": 1000}, "ideal_compute_ns": 4}
    compute["work"] = work
    return compute


def test_mixed_precision_mfu_normalizes_each_peak_and_keeps_topk_time():
    compute = compute_with_work()
    work = compute["work"]
    result = compute_metrics(compute)
    assert result["total"]["compute_mfu_percent"] == pytest.approx(100 * 4 / 15)
    assert result["topk"]["compute_mfu_percent"] is None
    assert result["topk"]["duration_ns"] == 1
    assert result["attention"]["compute_mfu_percent"] == pytest.approx(100 / 3)
    work["total"] = {"flops_by_precision": {"FP8": 6000}, "ideal_compute_ns": 3}
    with pytest.raises(ValueError, match="total FLOPs differ"):
        compute_metrics(compute)


def test_dense_counts_full_current_layer_payload_and_bandwidth_once():
    source = inputs()
    for phase in ("prefill", "extend"):
        source[phase]["compute"] = compute_with_work()
    for phase, key, byte_count in (
        ("prefill", "d2h", 100),
        ("extend", "sparse_h2d", 60),
        ("extend", "dense_h2d", 300),
        ("extend", "dense_next_h2d", 310),
    ):
        source[phase][key].update(
            {"bytes": byte_count, "source_scheme": "test", "layer": 1, "records": byte_count}
        )
    document = simulate(source)
    row = next(r for r in document["scenarios"] if r["id"] == "extend_dense_prefetch")
    assert row["io_bytes"] == 300
    assert row["standalone_io_bandwidth_GBps"] == 10
    assert row["schedule_mfu_percent"] == pytest.approx(100 * 4 / 30)
    io = [e for e in row["events"] if e["lane"] == "io"]
    assert [e["full_bytes"] for e in io] == [300, 310]
    assert all(e["full_bandwidth_GBps"] == 10 and "bytes" not in e for e in io)
    assert [e["bytes"] for e in row["full_transfers"]] == [300, 310]


def test_bandwidth_uses_full_service_time_and_decimal_gigabytes():
    result = transfer_metrics({"bytes": 5_000_000_000, "duration_ns": 100_000_000})
    assert result["bandwidth_GBps"] == 50
    assert result["MiB"] == pytest.approx(5_000_000_000 / 2**20)
    assert transfer_metrics({"bytes": 0, "duration_ns": 0})["bandwidth_GBps"] is None
    with pytest.raises(ValueError, match="transfer"):
        transfer_metrics({"bytes": 100, "duration_ns": 0})


def test_measured_echo_preserves_fusion_cleanup_and_residual_dependency():
    source = inputs()
    source["extend"]["echo"] = {
        "fused": {"duration_ns": 20},
        "cleanup": {"duration_ns": 1},
        "recall": {"duration_ns": 2},
    }
    row = next(row for row in simulate(source)["scenarios"] if row["id"] == "extend_echo_measured")
    assert row["completion_ns"] == 33
    assert event(row, "fused")["end_ns"] == event(row, "cleanup")["start_ns"]
    assert event(row, "cleanup")["end_ns"] == event(row, "topk")["start_ns"]
    assert event(row, "topk")["end_ns"] == event(row, "h2d")["start_ns"]
    assert event(row, "attention")["start_ns"] == event(row, "h2d")["end_ns"]
    assert len([e for e in row["events"] if e["lane"] == "fused"]) == 1
    assert len([e for e in row["events"] if e["lane"] == "io"]) == 1
