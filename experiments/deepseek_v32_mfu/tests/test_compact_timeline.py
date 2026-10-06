import pytest

from experiments.deepseek_v32_mfu.src import compact_timeline, launch_gap, timeline


def row(start, end, layer, stage, *, lane="Compute", source_stage=None):
    return {
        "lane": lane,
        "kind": "kernel",
        "name": "fixture",
        "stage": stage,
        "source_stage": source_stage or stage,
        "layer": f"layer_{layer}",
        "raw_start_ns": start,
        "raw_end_ns": end,
        "start_ms": start / 1e6,
        "end_ms": end / 1e6,
    }


def copy_row(start, end, stage, direction="host-to-device"):
    return {
        **row(start, end, 0, stage, lane="IO"),
        "kind": "memcpy",
        "name": direction,
        "layer": "shared",
    }


@pytest.mark.parametrize("method", ["hbm", "dense_prefetch"])
def test_last_prefill_chunk_selects_all_three_layers_and_excludes_previous_chunks(method):
    rows = [row(layer * 10, layer * 10 + 8, layer, "q_a_proj") for layer in range(3)]
    rows.append(copy_row(95, 99, "dense_history_prefetch_layer_0"))
    rows += [row(100 + layer * 10, 108 + layer * 10, layer, "q_a_proj") for layer in range(3)]
    panel = {
        "method": method,
        "chunks": [{"chunk": 0, "start_ns": 0}, {"chunk": 63, "start_ns": 90}],
        "rows": rows,
        "window": {"start_ns": 0, "end_ns": 140},
        "shared_tail": {"start_ns": 128},
    }
    clipped = compact_timeline.three_layer_panel(panel, "prefill")
    assert clipped["chunk"] == 63
    assert (clipped["window"]["start_ns"], clipped["window"]["end_ns"]) == (100, 128)
    assert len(clipped["rows"]) == 3
    assert [item["layer"] for item in clipped["layers"]] == [0, 1, 2]
    assert clipped["window"]["gap_no_io_percent"] == pytest.approx(100 * 4 / 28)


@pytest.mark.parametrize(
    ("history_h2d", "expected_start", "expected_idle"),
    [
        ((35, 80), 35, [(80, 90), (110, 115), (140, 145)]),
        ((80, 100), 80, [(110, 115), (140, 145)]),
        ((95, 120), 90, [(140, 145)]),
        (None, 90, [(110, 115), (140, 145)]),
    ],
)
def test_dense_extend_includes_only_history_h2d_leading_computation(
    history_h2d, expected_start, expected_idle
):
    rows = [
        copy_row(10, 20, "input_copy"),
        copy_row(20, 25, "dense_history_prefetch_layer_0", "device-to-host"),
        copy_row(25, 30, "dense_history_prefetch_layer_1"),
        row(90, 110, 0, "q_a_proj"),
        row(115, 140, 1, "q_a_proj"),
        row(145, 170, 2, "q_a_proj"),
    ]
    if history_h2d is not None:
        rows.append(copy_row(*history_h2d, "dense_history_prefetch_layer_0"))
    panel = {
        "method": "dense_prefetch",
        "rows": rows,
        "window": {"start_ns": 0, "end_ns": 200},
        "shared_tail": {"start_ns": 170},
    }
    clipped = compact_timeline.three_layer_panel(panel, "extend")
    assert (clipped["window"]["start_ns"], clipped["window"]["end_ns"]) == (expected_start, 170)
    assert clipped["window"]["l0_compute_start_ns"] == 90
    assert clipped["window"]["l0_history_h2d_start_ns"] == (history_h2d[0] if history_h2d else None)
    assert clipped["layers"][0] == {
        "layer": 0,
        "start_ns": expected_start,
        "end_ns": 110,
        "compute_start_ns": 90,
    }
    assert compact_timeline.idle_echo_annotations(clipped)["GPU idle"] == expected_idle
    assert all(item["raw_start_ns"] >= expected_start for item in clipped["rows"])
    if history_h2d is not None:
        transfer = next(item for item in clipped["rows"] if item["lane"] == "IO")
        assert (transfer["raw_start_ns"], transfer["raw_end_ns"]) == history_h2d
        assert transfer["start_ms"] == (history_h2d[0] - expected_start) / 1e6
        assert compact_timeline.io_direction(transfer) == "H2D: DRAM to GPU"


@pytest.mark.parametrize("method", ["hbm", "echo", "serial_sparse"])
def test_other_extend_methods_keep_first_compute_origin(method):
    panel = {
        "method": method,
        "rows": [
            copy_row(10, 20, "dense_history_prefetch_layer_0"),
            row(30, 40, 0, "q_a_proj"),
            row(45, 55, 1, "q_a_proj"),
            row(60, 70, 2, "q_a_proj"),
        ],
        "window": {"start_ns": 0, "end_ns": 80},
        "shared_tail": {"start_ns": 70},
    }
    clipped = compact_timeline.three_layer_panel(panel, "extend")
    assert clipped["window"]["start_ns"] == 30
    assert clipped["window"]["start_basis"] == "L0 first compute"
    assert clipped["window"]["l0_history_h2d_start_ns"] is None
    assert len(clipped["rows"]) == 3


def test_adjacent_layer_timestamp_overlap_preserves_raw_intervals_and_gap_unions():
    rows = [
        row(10, 50, 0, "q_a_proj"),
        row(52, 54, 0, "cache_write", lane="GPU control"),
        row(60, 100, 0, "mlp"),
        row(98, 105, 1, "input_residual_norm"),
        row(110, 150, 1, "mlp"),
        row(148, 155, 2, "input_residual_norm"),
        row(160, 200, 2, "mlp"),
    ]
    panel = {
        "method": "serial_sparse",
        "rows": rows,
        "window": {"start_ns": 0, "end_ns": 220},
        "shared_tail": {"start_ns": 200},
    }
    clipped = compact_timeline.three_layer_panel(panel, "extend")
    assert [(item["start_ns"], item["end_ns"]) for item in clipped["layers"]] == [
        (10, 100),
        (100, 150),
        (150, 200),
    ]
    assert [item["compute_start_ns"] for item in clipped["layers"]] == [10, 98, 148]
    assert [(item["raw_start_ns"], item["raw_end_ns"]) for item in clipped["rows"]] == [
        (item["raw_start_ns"], item["raw_end_ns"]) for item in rows
    ]
    assert clipped["rows"][3]["start_ms"] == 88 / 1e6
    assert clipped["window"]["compute_union_ms"] == pytest.approx(170 / 1e6)
    assert clipped["window"]["gap_ms"] == pytest.approx(20 / 1e6)
    assert clipped["window"]["gpu_idle_ms"] == pytest.approx(18 / 1e6)
    activities = [
        {"start": item["raw_start_ns"], "end": item["raw_end_ns"], "lane": item["lane"]}
        for item in clipped["rows"]
    ]
    measured = [
        launch_gap.summarize_window(
            activities,
            layer["start_ns"],
            layer["end_ns"],
            lane_classifier=lambda item: item["lane"],
        )
        for layer in clipped["layers"]
    ]
    assert [item["gap_ms"] for item in measured] == pytest.approx([10 / 1e6, 5 / 1e6, 5 / 1e6])
    for key in ("window_ms", "compute_union_ms", "gap_ms", "gpu_idle_ms", "control_only_ms"):
        assert sum(item[key] for item in measured) == pytest.approx(clipped["window"][key])


@pytest.mark.parametrize(
    "intervals", [[(10, 30), (5, 40), (50, 70)], [(10, 50), (20, 40), (60, 70)]]
)
def test_nonincreasing_layer_compute_endpoints_are_rejected(intervals):
    panel = {
        "method": "hbm",
        "rows": [
            row(start, end, layer, "q_a_proj") for layer, (start, end) in enumerate(intervals)
        ],
        "window": {"start_ns": 0, "end_ns": 80},
        "shared_tail": {"start_ns": 70},
    }
    with pytest.raises(ValueError, match="increasing layer compute start and end"):
        compact_timeline.three_layer_panel(panel, "extend")


def test_full_graph_ownership_keeps_gpu_boundaries_without_inventing_cpu_layers():
    scopes = [{"layer": "shared", "stage": "forward_misc", "start": 0, "end": 140}]
    replay_scope = {"layer": "shared", "stage": "extend_graph_replay_q_128", "label": "replay"}
    activities = [
        {
            "start": start,
            "end": end,
            "kind": "kernel",
            "name": "gemm",
            "scope": replay_scope,
            "stream_id": 1,
            "graph_layer": f"layer_{layer}",
            "graph_stage": "attention_projection",
            "graph_source_stage": "attention_projection",
            "full_extend_graph": True,
        }
        for start, end, layer in [(20, 45, 0), (50, 75, 1), (80, 105, 2)]
    ]
    first, _ = timeline.select_layer(scopes, [], activities, layer=0)
    second, visible = timeline.select_layer(scopes, [], activities, layer=1)
    assert (first["start_ns"], first["end_ns"]) == (0, 45)
    assert (second["start_ns"], second["end_ns"]) == (45, 75)
    assert visible[0]["layer"] == "layer_1"
    assert visible[0]["scope_label"] == "replay"
    assert all(activity["scope"]["layer"] == "shared" for activity in activities)
    assert launch_gap.summarize_window(activities, 0, 140)["gap_ms"] == pytest.approx(65 / 1e6)


def test_capture_source_stage_preserves_coarse_compute_colors():
    rows = [
        row(0, 1, 0, "residual_rms_norm", source_stage="input_residual_norm"),
        row(2, 3, 0, "attention_projection"),
        row(4, 5, 0, "residual_rms_norm", source_stage="post_attention_residual_norm"),
        row(6, 7, 0, "mlp", source_stage="dense_mlp"),
    ]
    assert [item["label"] for item in timeline.computation_segments(rows)] == [
        "Projection / RoPE",
        "Output / MLP",
    ]


def test_red_annotations_count_only_idle_and_echo_has_separate_management_segments():
    rows = [
        row(0, 10, 0, "q_a_proj"),
        row(10, 20, 0, "offload_prepare", lane="GPU control"),
        row(30, 40, 0, "indexer_fused", lane="Compute + IO"),
        row(40, 50, 0, "offload_finalize", lane="GPU control"),
        row(50, 60, 0, "prefetch_hint", lane="GPU control"),
    ]
    panel = {
        "method": "echo",
        "rows": rows,
        "window": {"start_ns": 0, "end_ns": 60, "gpu_idle_ms": 10 / 1e6},
    }
    annotations = compact_timeline.idle_echo_annotations(panel)
    assert annotations == {
        "GPU idle": [(20, 30)],
        "ECHO prepare": [(10, 20)],
        "ECHO finalize": [(40, 50)],
        "ECHO hint": [(50, 60)],
    }


def test_startup_comparison_retains_embedding_and_pre_layer_io_without_shared_tail():
    embedding = {**row(30, 32, 0, "embedding"), "layer": "shared"}
    rows = [
        copy_row(10, 20, "input_copy"),
        row(22, 25, 0, "input_copy", lane="GPU control"),
        embedding,
        copy_row(35, 80, "dense_history_prefetch_layer_0"),
        row(90, 110, 0, "q_a_proj"),
        row(115, 140, 1, "q_a_proj"),
        row(145, 170, 2, "q_a_proj"),
        {**row(190, 200, 0, "lm_head"), "layer": "shared"},
    ]
    panel = {
        "method": "dense_prefetch",
        "rows": rows,
        "window": {"start_ns": 0, "end_ns": 210},
        "shared_tail": {"start_ns": 170},
    }
    cropped = compact_timeline.three_layer_panel(panel, "extend")
    extended = compact_timeline.extend_startup_panel(panel)
    assert cropped["window"]["start_ns"] == 35
    assert cropped["window"]["start_basis"] == "L0 history H2D"
    assert extended["cropped_window"] == cropped["window"]
    assert extended["layers"] == cropped["layers"]
    assert (extended["window"]["start_ns"], extended["window"]["end_ns"]) == (0, 170)
    assert extended["startup"]["end_ns"] == 30
    assert extended["embedding"]["kernel_union_ms"] == pytest.approx(2 / 1e6)
    assert extended["l0_offset_ms"] == pytest.approx(90 / 1e6)
    assert len(extended["rows"]) == 7
    assert compact_timeline.idle_echo_annotations(extended)["GPU idle"] == [
        (0, 10),
        (20, 22),
        (25, 30),
        (32, 35),
        (80, 90),
        (110, 115),
        (140, 145),
    ]
    assert extended["window"]["gap_ms"] == pytest.approx(43 / 1e6)
