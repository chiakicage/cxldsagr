import copy
import hashlib
import json
import math
from collections import defaultdict

import pytest
import torch

from experiments.deepseek_v32_echo_prefill.src import report


def annotation(mode, tokens, placement, chunk_size):
    calls = []
    chunks = math.ceil(tokens / chunk_size)

    def add(stage, layer, device, count):
        calls.extend(
            {
                "stage": stage,
                "layer": layer,
                "device": device,
                "host_ms": 0.3,
                "cuda_inclusive_ms": 0.2,
                "cuda_exclusive_ms": 0.1,
            }
            for _ in range(count)
        )

    for layer, device in enumerate(placement):
        stages = [
            "input_residual_norm",
            "attention_projection",
            "cache_write",
            "offload_prepare",
            "exact_topk",
            "attention_output",
            "post_attention_residual_norm",
            "sparse_mla",
            "indexer_prefetch" if mode == "offload" else "indexer",
        ]
        stages += (
            ["dense_mlp"]
            if layer < 3
            else ["moe", "moe_routing", "moe_experts", "moe_shared_experts"]
        )
        if mode == "offload":
            stages += ["offload_exact_recall"]
        for stage in stages:
            add(stage, f"layer_{layer}", int(device.split(":")[1]), chunks)
    add("embedding", "shared", 0, chunks)
    add("hidden_transfer", "shared", 0, chunks * 61)
    add("final_norm_lm_head", "shared", 7, 1)
    totals = defaultdict(float)
    for call in calls:
        totals[call["stage"]] += call["cuda_exclusive_ms"]
    return {"wall_ms": 300.0, "stage_calls": calls, "stages_cuda_exclusive_ms": dict(totals)}


def cache(mode, tokens, slots):
    offload = mode == "offload"
    capacity, record = 66560, 1152
    row = {
        "written_records": tokens,
        "recalled_records": tokens + 42 if offload else 0,
        "evicted_records": tokens if offload else 0,
        "max_working_set": slots if offload else 0,
        "prefetched_records": 50 if offload else 0,
        "device_to_host_bytes": tokens * record if offload else 0,
        "device_record_bytes": (slots if offload else capacity) * record,
        "host_record_bytes": capacity * record if offload else 0,
        "record_bytes": record,
        "device_slots": slots if offload else capacity,
    }
    row["host_to_device_bytes"] = (row["prefetched_records"] + row["recalled_records"]) * record
    return [copy.deepcopy(row) for _ in range(61)]


def archive_sources(directory, required):
    sources = {}
    for relative in required:
        path = directory / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Unit-test source archive fixture: {relative}\n")
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "sources.json").write_text(json.dumps(sources))
    return sources


@pytest.fixture
def valid_run(tmp_path):
    run_id = "test_fixture_61_layers"
    directory = tmp_path / "output" / "data" / run_id
    directory.mkdir(parents=True)
    sources = archive_sources(directory, report.REQUIRED_SOURCES)
    request = {
        "model": "deepseek_v32",
        "prompt": "Unit-test fixture with readable history and candidates.",
        "input_ids": list(range(66560)),
        "attention_mask": [1] * 66560,
        "instruction_tokens": 23,
        "stable_prefix_tokens": 65536,
        "candidate_suffix_tokens": 1024,
        "total_input_tokens": 66560,
        "history_token_span": [23, 65536],
        "candidate_token_span": [65536, 66560],
        "history_sha256": hashlib.sha256(b"unit-test history").hexdigest(),
        "content_is_synthetic": True,
    }
    request_text = json.dumps(request, ensure_ascii=False)
    (directory / "request.json").write_text(request_text + "\n")
    placement = ["cuda:0"] * 13 + [f"cuda:{device}" for device in (1, 2, 6, 7) for _ in range(12)]
    result = {
        "run_id": run_id,
        "accepted": True,
        "scope": "complete_checkpoint_61_transformer_layers_embedding_final_norm_lm_head",
        "model": "/unit-test/checkpoint",
        "layers": 61,
        "prefix_tokens": 65536,
        "extend_tokens": 1024,
        "logits": "last_token_only",
        "checkpoint_weights": "resident_block_FP8_with_BF16_MLA_absorbed_KV_projections",
        "cache_record": "BF16_512_latent_plus_64_RoPE_1152_bytes",
        "indexer": "64x128_FP8_after_normalized_Hadamard_top2048",
        "offload": "mapped_pinned_local_DRAM_fused_indexer_prefetch_then_exact_recall",
        "placement": placement,
        "chunk_size": 8192,
        "slots": 16384,
        "warmups": 1,
        "repeats": 3,
        "prefill_repeats": 2,
        "seed": 42,
        "request_sha256": hashlib.sha256(request_text.encode()).hexdigest(),
        "source_sha256": sources,
        "dependencies": {
            name: "test-version" for name in ("torch", "triton", "safetensors", "apache-tvm-ffi")
        },
        "hardware": [f"NVIDIA H200 unit-test GPU {device}" for device in (0, 1, 2, 6, 7)],
        "correctness": {
            "bitwise_equal": True,
            "same_next_token": True,
            "max_abs": 0.0,
            "nrmse": 0.0,
        },
        "measurements": {},
    }
    for mode in report.MODES:
        factor = 1.0 if mode == "resident" else 1.5
        prefix_walls = [200.0 * factor, 220.0 * factor]
        extend_walls = [10.0 * factor, 11.0 * factor, 12.0 * factor]
        result["measurements"][mode] = {
            "prefill": [
                {"wall_ms": value, "peak_allocated_bytes": [2**30] * 5} for value in prefix_walls
            ],
            "extend": [
                {"wall_ms": value, "peak_allocated_bytes": [2**30] * 5} for value in extend_walls
            ],
            "prefill_median_ms": 210.0 * factor,
            "extend_median_ms": 11.0 * factor,
            "annotated_prefill": annotation(mode, 65536, placement, 8192),
            "annotated_extend": annotation(mode, 1024, placement, 8192),
            "prefill_cache_per_layer": cache(mode, 65536, 16384),
            "cache_per_layer": cache(mode, 1024, 16384),
        }
        torch.save(torch.arange(129280).float().reshape(1, -1), directory / f"{mode}_logits.pt")
    path = directory / "result.json"
    path.write_text(json.dumps(result))
    return path, result


def rewrite(path, result):
    path.write_text(json.dumps(result))


@pytest.mark.parametrize("layout", ["legacy", "model"])
def test_complete_source_layouts_validate_their_archived_files(valid_run, layout):
    path, result = valid_run
    required = report.LEGACY_REQUIRED_SOURCES if layout == "legacy" else report.REQUIRED_SOURCES
    result["source_sha256"] = archive_sources(path.parent, required)
    rewrite(path, result)
    assert report.make_summary(path)["provenance"]["source_sha256"] == result["source_sha256"]


@pytest.mark.parametrize(
    "layout,missing",
    [
        (layout, source)
        for layout, required in (
            ("legacy", report.LEGACY_REQUIRED_SOURCES),
            ("model", report.REQUIRED_SOURCES),
        )
        for source in sorted(required)
    ],
)
def test_each_source_layout_requires_every_runtime_dependency(valid_run, layout, missing):
    path, result = valid_run
    required = report.LEGACY_REQUIRED_SOURCES if layout == "legacy" else report.REQUIRED_SOURCES
    result["source_sha256"] = archive_sources(path.parent, required - {missing})
    with pytest.raises(report.InvalidResult, match="Required implementation snapshots"):
        report.validate_sources_and_request(result, path.parent)


def test_incomplete_source_layouts_cannot_be_combined_into_one_complete_run(valid_run):
    path, result = valid_run
    required = (report.LEGACY_REQUIRED_SOURCES | report.REQUIRED_SOURCES) - {
        "operators/sm90/deepseek_mla.py",
        "operators/deepseek_v32/attention/_validation.py",
    }
    result["source_sha256"] = archive_sources(path.parent, required)
    with pytest.raises(report.InvalidResult, match="Required implementation snapshots"):
        report.validate_sources_and_request(result, path.parent)


def test_valid_summary_keeps_independent_timing_and_byte_boundaries(valid_run):
    path, _ = valid_run
    summary = report.make_summary(path)
    assert len(summary["phases"]) == 4
    rows = {(row["mode"], row["phase"]): row for row in summary["phases"]}
    assert rows["resident", "extend"]["wall_median_ms"] == 11
    assert rows["offload", "prefill"]["wall_median_ms"] == 315
    assert rows["resident", "extend"]["annotated_wall_ms"] == 300
    assert (
        rows["offload", "extend"]["annotated_cache_totals"]["device_record_bytes"]
        == 61 * 16384 * 1152
    )
    assert (
        rows["offload", "prefill"]["annotated_cache_totals"]["device_to_host_bytes"]
        == 61 * 65536 * 1152
    )
    assert (
        "no independent prefetch duration" in summary["measurement_boundaries"]["indexer_prefetch"]
    )
    boundaries = summary["measurement_boundaries"]
    assert "cross-device overlap" in boundaries["annotated"]
    assert "host launch gaps" in boundaries["annotated"]
    assert "not summed kernel durations" in boundaries["annotated"]
    assert "source-computation dependencies" in boundaries["hidden_transfer"]
    assert "current-chunk staging" in boundaries["recalled_records"]
    assert "all gathered records" in boundaries["cache_bytes"]
    assert not (path.parent / "summary.json").exists()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("accepted", False, "not been accepted"),
        ("layers", 1, "61 layers"),
        ("prefix_tokens", 64000, "65536"),
        ("extend_tokens", 1, "1024"),
        ("warmups", 0, "warmups"),
        ("repeats", 2, "repetition count"),
        ("slots", 66560, "bounded pool"),
    ],
)
def test_rejects_wrong_scope_or_measurement_counts(valid_run, field, value, match):
    path, result = valid_run
    result[field] = value
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match=match):
        report.make_summary(path)


def test_rejects_partially_mapped_layers(valid_run):
    path, result = valid_run
    result["placement"].pop()
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match="Every transformer layer"):
        report.make_summary(path)


@pytest.mark.parametrize("target", ["sample", "event", "total"])
def test_rejects_nonfinite_or_negative_times(valid_run, target):
    path, result = valid_run
    measurement = result["measurements"]["offload"]
    if target == "sample":
        measurement["extend"][0]["wall_ms"] = float("nan")
    elif target == "event":
        measurement["annotated_extend"]["stage_calls"][0]["host_ms"] = -1
    else:
        measurement["annotated_extend"]["stages_cuda_exclusive_ms"]["cache_write"] = float("inf")
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match="finite"):
        report.make_summary(path)


def test_rejects_annotated_totals_that_do_not_match_events(valid_run):
    path, result = valid_run
    result["measurements"]["resident"]["annotated_extend"]["stages_cuda_exclusive_ms"][
        "cache_write"
    ] += 1
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match="stage total mismatch"):
        report.make_summary(path)


def test_rejects_prefix_without_complete_layer_execution(valid_run):
    path, result = valid_run
    annotated = result["measurements"]["resident"]["annotated_prefill"]
    for index, call in enumerate(annotated["stage_calls"]):
        if call["layer"] == "layer_60" and call["stage"] == "attention_projection":
            removed = annotated["stage_calls"].pop(index)
            annotated["stages_cuda_exclusive_ms"]["attention_projection"] -= removed[
                "cuda_exclusive_ms"
            ]
            break
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match="incomplete layer_60/attention_projection"):
        report.make_summary(path)


def test_rejects_annotation_that_drops_failed_union_split_attempts(valid_run):
    path, result = valid_run
    annotated = result["measurements"]["offload"]["annotated_extend"]
    originals = {
        call["stage"]: call
        for call in annotated["stage_calls"]
        if call["layer"] == "layer_0" and call["stage"] in ("sparse_mla", "offload_exact_recall")
    }
    # One root splits into two successfully consumed leaves: two MLA calls,
    # two successful recalls, and one failed recall that must also be timed.
    for stage, original in originals.items():
        annotated["stage_calls"].append(copy.deepcopy(original))
        annotated["stages_cuda_exclusive_ms"][stage] += original["cuda_exclusive_ms"]
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match="split attempts"):
        report.make_summary(path)
    failed_attempt = copy.deepcopy(originals["offload_exact_recall"])
    annotated["stage_calls"].append(failed_attempt)
    annotated["stages_cuda_exclusive_ms"]["offload_exact_recall"] += failed_attempt[
        "cuda_exclusive_ms"
    ]
    rewrite(path, result)
    report.make_summary(path)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("written_records", 1024, "writes are incomplete"),
        ("record_bytes", 656, "1152-byte"),
        ("device_slots", 66560, "slot count"),
        ("device_record_bytes", 1, "device bytes"),
        ("host_to_device_bytes", 0, "H2D"),
        ("device_to_host_bytes", 0, "D2H"),
    ],
)
def test_rejects_fake_history_or_inconsistent_cache_accounting(valid_run, field, value, match):
    path, result = valid_run
    result["measurements"]["offload"]["prefill_cache_per_layer"][60][field] = value
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match=match):
        report.make_summary(path)


def test_rejects_unverified_or_tampered_sources(valid_run):
    path, result = valid_run
    source = path.parent / "source" / "models/deepseek_v32/echo_infer.py"
    source.write_text("# Changed after measurement\n")
    with pytest.raises(report.InvalidResult, match="snapshot SHA256 mismatch"):
        report.make_summary(path)
    result["source_sha256"].pop("models/deepseek_v32/echo_infer.py")
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match="Required implementation snapshots"):
        report.make_summary(path)


def test_rejects_request_hash_or_semantic_boundary_mismatch(valid_run):
    path, result = valid_run
    request_path = path.parent / "request.json"
    request = json.loads(request_path.read_text())
    request["candidate_token_span"] = [65535, 66560]
    request_path.write_text(json.dumps(request) + "\n")
    with pytest.raises(report.InvalidResult, match="Request SHA256"):
        report.make_summary(path)
    result["request_sha256"] = hashlib.sha256(
        request_path.read_bytes().removesuffix(b"\n")
    ).hexdigest()
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match="Candidate span"):
        report.make_summary(path)


def test_rejects_numerically_close_but_nonidentical_logits(valid_run):
    path, result = valid_run
    result["correctness"]["bitwise_equal"] = False
    rewrite(path, result)
    with pytest.raises(report.InvalidResult, match="bitwise_equal=true"):
        report.make_summary(path)
    result["correctness"]["bitwise_equal"] = True
    rewrite(path, result)
    value = torch.load(path.parent / "offload_logits.pt", weights_only=True)
    value[0, 0] = 1e-5
    torch.save(value, path.parent / "offload_logits.pt")
    with pytest.raises(report.InvalidResult, match="Saved resident/offload logits differ"):
        report.make_summary(path)


@pytest.fixture
def stub_figures(monkeypatch):
    # Publication must stay covered without the optional analysis dependencies.
    def create_figures(summary, directory):
        for name in ("latency.png", "latency.svg", "annotated_scopes.png", "annotated_scopes.svg"):
            (directory / name).write_text(f"{summary['run_id']}: {name}\n")

    monkeypatch.setattr(report, "plot", create_figures)


def test_publish_copies_artifacts_and_preserves_unmarked_readme(valid_run, tmp_path, stub_figures):
    path, _ = valid_run
    report_dir = tmp_path / "experiment" / "report"
    report_dir.parent.mkdir()
    readme = report_dir.parent / "README.md"
    text = "Existing purpose, method, status and results.\n"
    readme.write_text(text)
    status = report.generate_report(path, publish=True, report_dir=report_dir)
    assert status["published"] and not status["readme_updated"]
    assert readme.read_text() == text
    for name in (
        "summary.json",
        "summary.csv",
        "latency.png",
        "latency.svg",
        "annotated_scopes.png",
        "annotated_scopes.svg",
    ):
        assert (report_dir / name).read_bytes() == (path.parent / name).read_bytes()
        assert (report_dir / name).stat().st_size > 0
    assert (
        json.loads((report_dir / "summary.json").read_text())["run_id"] == "test_fixture_61_layers"
    )
    fragment = (path.parent / "report_fragment.md").read_text()
    assert "逐元素核验 bitwise" in fragment
    assert "不能全部解释为 residual recall" in fragment
    assert "不能相加作端到端分解" in fragment
    assert "不能当成独立 NVLink copy" in fragment
    assert "output/data/test_fixture_61_layers/`" in fragment
    assert "](output/" not in fragment


def test_plot_generates_figures_with_measurement_boundaries(valid_run):
    pytest.importorskip(
        "matplotlib", reason="Figure rendering requires the optional analysis group"
    )
    path, _ = valid_run
    report.plot(report.make_summary(path), path.parent)
    for name in ("latency.png", "latency.svg", "annotated_scopes.png", "annotated_scopes.svg"):
        assert (path.parent / name).stat().st_size > 100
    assert "test_fixture_61_layers" in (path.parent / "latency.svg").read_text()
    scope_svg = (path.parent / "annotated_scopes.svg").read_text()
    assert "not uninstrumented end-to-end latency" in scope_svg
    assert "Offload preparation" in scope_svg
    assert "Exact recall" in scope_svg
    assert "Hidden transfer / dependency wait" in scope_svg
    assert "host launch gaps" in scope_svg


def test_publish_only_replaces_explicit_readme_markers(valid_run, tmp_path, stub_figures):
    path, _ = valid_run
    report_dir = tmp_path / "experiment" / "report"
    report_dir.parent.mkdir()
    readme = report_dir.parent / "README.md"
    readme.write_text("Before.\n" + report.BEGIN + "\nOld.\n" + report.END + "\nAfter.\n")

    status = report.generate_report(path, publish=True, report_dir=report_dir)
    assert status["readme_updated"]
    assert readme.read_text().startswith("Before.\n" + report.BEGIN)
    assert readme.read_text().endswith(report.END + "\nAfter.\n")
    assert "Old." not in readme.read_text()


def test_rejected_result_never_overwrites_existing_report(valid_run, tmp_path):
    path, result = valid_run
    result["accepted"] = False
    rewrite(path, result)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    old = report_dir / "summary.json"
    old.write_text("previous accepted report\n")
    with pytest.raises(report.InvalidResult):
        report.generate_report(path, publish=True, report_dir=report_dir)
    assert old.read_text() == "previous accepted report\n"
    assert not (path.parent / "summary.json").exists()


def test_malformed_readme_markers_fail_before_publication(valid_run, tmp_path):
    path, _ = valid_run
    report_dir = tmp_path / "experiment" / "report"
    report_dir.mkdir(parents=True)
    (report_dir.parent / "README.md").write_text("Existing report\n" + report.BEGIN)
    old = report_dir / "summary.json"
    old.write_text("previous accepted report\n")
    with pytest.raises(report.InvalidResult, match="markers are malformed"):
        report.generate_report(path, publish=True, report_dir=report_dir)
    assert old.read_text() == "previous accepted report\n"
    assert not (path.parent / "summary.json").exists()
