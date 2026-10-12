from copy import deepcopy

from experiments.deepseek_v32_echo_official.src import q1_timeline_categories as categories


def panel(source, rows):
    return {
        "source": source,
        "method": "echo",
        "rows": rows,
        "window": {"start_ns": 0, "end_ns": 100},
        "layers": [{"layer": n} for n in range(3)],
        "idle_intervals_ns": [[90, 100]],
    }


def activity(stage, name, start, end, lane="Compute", category="Indexer / top-k"):
    return {
        "layer": 0,
        "stage": stage,
        "name": name,
        "start_ns": start,
        "end_ns": end,
        "lane": lane,
        "category": category,
        "graph_node_id": start + 1,
        "stream": 7,
        "kind": "kernel",
    }


def test_official_scope_includes_topk_setup_but_excludes_metadata_and_hint():
    source = panel(
        "Official SGLang",
        [
            activity("indexer", "index q projection", 0, 15),
            activity("indexer", "index k projection", 5, 20),
            activity("indexer", "smxx_paged_mqa_logits_metadata<32>", 21, 23),
            activity("indexer_prefetch", "sm90_fp8_paged_mqa_logits_fused_v2<1>", 24, 30),
            activity("topk", "FillFunctor<float>", 31, 32, "GPU control", "GPU control"),
            activity("topk", "topk_transform_decode_kernel", 33, 50),
            activity("topk", "arange/mask", 51, 54, "GPU control", "GPU control"),
            activity("indexer", "EMA mul/add", 55, 58),
            activity("indexer", "Device-to-Device", 59, 60, "GPU control", "GPU control"),
        ],
    )
    source["rows"][1]["stream"] = 142
    source["rows"][-1]["kind"] = "memcpy"
    original = deepcopy(source)
    aligned = categories.align_panel(source)
    assert source == original
    assert [r["category"] for r in aligned["rows"]] == [
        "Projection / RoPE",
        "Projection / RoPE",
        "GPU control",
        "Indexer / top-k",
        "Indexer / top-k",
        "Indexer / top-k",
        "Indexer / top-k",
        "ECHO hint",
        "ECHO hint",
    ]
    totals = {r["category"]: r for r in categories.category_metrics(aligned) if r["layer"] == "all"}
    assert totals["Projection / RoPE"]["union_ns"] == 20
    assert totals["Projection / RoPE"]["kernel_sum_ns"] == 30
    assert totals["Indexer / top-k"]["union_ns"] == 27
    assert totals["ECHO hint"]["activity_sum_ns"] == 4
    assert totals["ECHO hint"]["kernel_sum_ns"] == 3
    for source_row, aligned_row in zip(source["rows"], aligned["rows"], strict=True):
        assert aligned_row["source_lane"] == source_row["lane"]
        assert aligned_row["source_category"] == source_row["category"]
        assert aligned_row["graph_node_id"] == source_row["graph_node_id"]


def test_local_scoring_contract_has_same_boundary_and_keeps_preparation_outside():
    aligned = categories.align_panel(
        panel(
            "Local warm",
            [
                activity("indexer_fused", "prepare_keys", 0, 10, "GPU control", "GPU control"),
                activity("indexer_fused", "smxx_paged_mqa_logits_metadata<32>", 11, 13),
                activity("indexer_fused", "sm90_fp8_paged_mqa_logits_fused_v2<1>", 14, 24),
                activity("indexer_fused", "smxx_clean_logits<1>", 25, 28),
                activity("exact_topk", "FilteredTopK", 29, 40),
                activity("exact_topk", "CUB sort/mask", 41, 50),
            ],
        )
    )
    assert [r["category"] for r in aligned["rows"]] == [
        "GPU control",
        "GPU control",
        *["Indexer / top-k"] * 4,
    ]


def test_metrics_clip_each_activity_and_ignore_activities_outside_window():
    aligned = categories.align_panel(
        panel(
            "Local warm",
            [
                activity("exact_topk", "selection", -5, 10),
                activity("exact_topk", "sort", 90, 110),
                activity("exact_topk", "outside", 120, 130),
            ],
        )
    )
    metric = next(
        row
        for row in categories.category_metrics(aligned)
        if row["layer"] == "all" and row["category"] == "Indexer / top-k"
    )
    assert metric["union_ns"] == metric["activity_sum_ns"] == metric["kernel_sum_ns"] == 20
