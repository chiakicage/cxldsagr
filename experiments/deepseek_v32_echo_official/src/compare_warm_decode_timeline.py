"""Compare signed official decode and local warm ECHO windows without GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import compare_decode_timeline as decode
from experiments.deepseek_v32_echo_official.src import compare_timeline as comparison
from experiments.deepseek_v32_echo_official.src import q1_timeline_categories as categories

engine = comparison.engine
EXPERIMENT = comparison.EXPERIMENT
PUBLISHED_FILES = (
    "timeline_decode.svg",
    "timeline_decode.png",
    "windows.csv",
    "summary.json",
    "provenance.json",
    "publication_manifest.json",
    "report.md",
    "aligned_panels.json",
    "category_metrics.csv",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_manifest(directory, filename="publication_manifest.json", hash_key="files_sha256"):
    """Verify every payload, including figures that are not used for the comparison."""
    manifest = comparison.read(directory / filename)
    hashes = manifest[hash_key]
    require(bool(hashes), "Publication has no payload hashes")
    for name, digest in hashes.items():
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()), "Payload escapes publication")
        comparison.require_hash(path, digest)
    return manifest


def validate_warm(directory):
    """Authenticate the display, original publication and input-audit source chain."""
    publication = validate_manifest(directory)
    require(publication["schema"] == "q1-warm-display-artifacts-v1", "Expected warm display")
    display = comparison.read(directory / "display_provenance.json")
    require(
        display["schema"] == "q1-warm-timeline-display-v1"
        and {
            "summary.json",
            "window_rows.json",
            "provenance.json",
            "display_provenance.json",
            "original_publication_manifest.json",
            "input_recall_audit/identity_recall.json",
            "input_recall_audit/publication_manifest.json",
            "timeline.svg",
            "timeline.png",
            "results.md",
            "timing_samples.csv",
            "cache_samples.csv",
        }.issubset(publication["files_sha256"]),
        "Warm display payload coverage differs",
    )
    original_path = directory / "original_publication_manifest.json"
    original_hash = engine.sha256(original_path)
    original_directory = Path(display["source_report"])
    comparison.require_hash(original_directory / "publication_manifest.json", original_hash)
    original = validate_manifest(original_directory)
    require(
        original == comparison.read(original_path)
        and original["schema"] == "q1-warm-artifacts-v1"
        and display["source_publication_sha256"] == original_hash,
        "Warm original publication chain differs",
    )
    for name, digest in display["unchanged_data_sha256"].items():
        require(original["files_sha256"][name] == digest, "Warm unchanged-data chain differs")
        comparison.require_hash(directory / name, digest)
    require(
        {
            "summary.json",
            "window_rows.json",
            "provenance.json",
            "cache_samples.csv",
            "timing_samples.csv",
        }.issubset(display["unchanged_data_sha256"]),
        "Warm unchanged-data coverage differs",
    )

    audit_directory = directory / "input_recall_audit"
    audit_publication = validate_manifest(audit_directory)
    audit = comparison.read(audit_directory / "identity_recall.json")
    raw_audit_path = Path(audit_publication["raw_publication"]["path"])
    comparison.require_hash(raw_audit_path, audit_publication["raw_publication"]["sha256"])
    raw_audit = validate_manifest(raw_audit_path.parent)
    generator_path = Path(audit_publication["source_audit"]["path"])
    comparison.require_hash(generator_path, audit_publication["source_audit"]["sha256"])
    require(
        audit_publication["schema"] == "q1-warm-selected-identity-recall-supplement-v1"
        and audit["schema"] == "deepseek-q1-official-warm-identity-recall-v1"
        and audit["passed"] is True
        and audit["official_numerical_acceptance"] is False
        and raw_audit["files_sha256"]["identity_recall.json"]
        == engine.sha256(audit_directory / "identity_recall.json")
        and audit["generator_sha256"] == engine.sha256(generator_path)
        and audit_publication["warm_original_publication_manifest_sha256"] == original_hash
        and raw_audit["warm_original_publication_manifest_sha256"] == original_hash
        and audit["warm_cohort"]["publication_manifest_sha256"] == original_hash
        and audit_publication["official_observation_sha256"]
        == raw_audit["official_observation_sha256"]
        == audit["official_observation_sha256"],
        "Warm input-audit provenance chain differs",
    )

    summary_path = directory / "summary.json"
    summary = comparison.read(summary_path)
    shape = summary["configuration"]
    expected = {
        "prefix_tokens": 65536,
        "extend_tokens": 1,
        "num_layers": 3,
        "chunk_size": 1024,
        "extend_chunk_size": 1,
        "extend_residency": "warm",
        "method_isolation": "fresh-process-one-method-v1",
    }
    require(
        summary["schema"] == "deepseek-v32-q1-warm-report-v1"
        and summary["passed"] is True
        and all(shape.get(key) == value for key, value in expected.items())
        and summary["run_id"] == display["measurement_run_id"] == audit["warm_cohort"]["run_id"]
        and engine.sha256(summary_path) == audit["warm_cohort"]["summary_sha256"],
        "Expected an accepted warm H65536/A1 three-layer report",
    )
    saved = comparison.read(directory / "window_rows.json")
    require(
        list(saved) == ["extend"]
        and len(saved["extend"]) == 1
        and saved["extend"][0]["method"] == "echo",
        "Expected one warm ECHO extend panel",
    )
    panel = saved["extend"][0]
    profile = panel["provenance"]
    provenance = comparison.read(directory / "provenance.json")
    child = summary["children"]["echo"]["profile"]
    bench = summary["children"]["echo"]["bench"]
    check = summary["children"]["echo"]["check"]
    require(
        panel["phase"] == "extend"
        and panel["chunk"] is None
        and panel["shape"] == comparison.compact.profile_shape(shape)
        and panel["window"]
        == summary["window"]
        == display["rendered_metrics"]["extend/echo"]["window"]
        and profile == provenance["profile"]
        and profile["profile_run_id"]
        == child["run_id"]
        == audit["warm_cohort"]["bound_profile_run_id"]
        and profile["profile_directory"] == child["directory"]
        and profile["result_sha256"]
        == child["result_sha256"]
        == audit["warm_cohort"]["profile_result_sha256"]
        and profile["sqlite"] == str(Path(child["directory"]) / "capture_4.sqlite")
        and profile["sqlite_sha256"] == child["files"]["data"]["capture_4.sqlite"]
        and profile["validation_receipt"]["receipt_path"] == check["receipt_path"]
        and profile["validation_receipt"]["receipt_sha256"] == summary["receipts"]["echo"]
        and all(
            profile["benchmark"][key] == bench[key]
            for key in ("run_id", "directory", "result_sha256")
        )
        and profile["benchmark"]["execution_identity_sha256"]
        == summary["execution_identity_sha256"]
        == audit["warm_cohort"]["execution_identity_sha256"],
        "Warm panel profile, receipt, benchmark or window provenance differs",
    )
    for path, digest in (
        (str(Path(child["directory"]) / "result.json"), child["result_sha256"]),
        (profile["sqlite"], profile["sqlite_sha256"]),
    ):
        require(provenance["input_sha256"].get(path) == digest, "Unbound warm profile input")
    counts = audit["warm_cohort"]["counts"]
    profile_counts = [row for row in counts if row["phase"] == "profile"]
    require(
        [row["layer"] for row in profile_counts] == [0, 1, 2]
        and len(summary["profile_cache_per_layer"]) == 3
        and len(summary["cache_samples"]) == shape["repeats"] * 3,
        "Warm per-layer traffic coverage differs",
    )
    for row in counts:
        require(
            row["host_to_device_bytes"] == 0
            and row["device_to_host_bytes"] == 1152
            and row["prefetched_records"] == row["recalled_records"] == 0,
            "Warm measured traffic differs",
        )
    for signed, measured in zip(profile_counts, summary["profile_cache_per_layer"], strict=True):
        require(
            all(
                signed[key] == measured[key]
                for key in (
                    "host_to_device_bytes",
                    "device_to_host_bytes",
                    "prefetched_records",
                    "recalled_records",
                )
            ),
            "Warm profile traffic binding differs",
        )
    bench_counts = [row for row in counts if row["phase"] == "bench"]
    require(
        len(bench_counts) == shape["repeats"] * 3
        and len(counts) == len(bench_counts) + 3
        and {(row["sample"], row["layer"]) for row in bench_counts}
        == {(sample, layer) for sample in range(shape["repeats"]) for layer in range(3)},
        "Warm benchmark traffic coverage differs",
    )
    for signed, measured in zip(bench_counts, summary["cache_samples"], strict=True):
        require(
            all(
                signed[key] == measured[key]
                for key in (
                    "sample",
                    "layer",
                    "host_to_device_bytes",
                    "device_to_host_bytes",
                    "prefetched_records",
                    "recalled_records",
                )
            ),
            "Warm benchmark traffic binding differs",
        )
    normalized, categories = comparison.normalize_mfu_panel(panel, profile["profile_run_id"])
    require(
        dict(categories) == display["rendered_metrics"]["extend/echo"]["compute_categories"]
        and normalized["idle_intervals_ns"]
        == display["rendered_metrics"]["extend/echo"]["annotation_intervals_ns"]["GPU idle"],
        "Warm display intervals or categories differ",
    )
    normalized.update(source="Local warm", condition="same supplied token; warm history KV")
    paths = [
        directory / "publication_manifest.json",
        *[directory / name for name in publication["files_sha256"]],
        original_directory / "publication_manifest.json",
        *[original_directory / name for name in original["files_sha256"]],
        audit_directory / "publication_manifest.json",
        raw_audit_path,
        *[raw_audit_path.parent / name for name in raw_audit["files_sha256"]],
        generator_path,
    ]
    return normalized, list(dict.fromkeys(paths)), {"summary": summary, "audit": audit}


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def validate_input_identity(official_report, warm, official_run_directory):
    """Bind the old profile's actual completion to the warm supplied token array."""
    run_path = official_run_directory / "run.json"
    source = official_report["sources"]["echo_profile"]
    comparison.require_hash(run_path, source["run_sha256"])
    run = comparison.read(run_path)
    complete_path = official_run_directory / "sample_00/engine/complete.json"
    complete = comparison.read(complete_path)
    require(
        run["run_id"] == source["run_id"]
        and run["status"] == complete["status"] == "completed"
        and run["case"] == complete["case"] == "echo"
        and run["mode"] == complete["mode"] == "profile"
        and complete == run["workers"][0]["completion"]
        and run["protocol"] == complete["protocol"]
        and run["protocol"]["history_tokens"] == 65536
        and run["protocol"]["decode_forwards"] == 1
        and complete["row"]["output_ids"] == [57841, 39556]
        and complete["row"]["prompt_tokens"] == 65536
        and complete["row"]["completion_tokens"] == 2
        and complete["prefix_sha256"] == complete["row"]["input_sha256"] == run["prompt_sha256"],
        "Official profile completion or decode input differs",
    )
    local = warm["audit"]["local_input"]
    request_path = Path(local["path"])
    comparison.require_hash(request_path, local["file_sha256"])
    request = comparison.read(request_path)
    tokens = request["input_ids"]
    require(
        local["exact_history_arrays_equal"] is True
        and local["exact_decode_token_equal"] is True
        and local["local_sampling_included"] is False
        and local["prefix_tokens"] == 65536
        and local["query_tokens"] == 1
        and len(tokens) == 65537
        and token_hash(tokens[:65536]) == local["history_token_sha256"] == run["prompt_sha256"]
        and token_hash(tokens) == local["full_token_sha256"]
        and tokens[65536]
        == local["decode_input_token"]
        == complete["row"]["output_ids"][0]
        == warm["summary"]["configuration"]["append_token"]
        and warm["summary"]["children"]["echo"]["profile"]["files"]["data"]["request.json"]
        == local["file_sha256"],
        "Warm supplied history or decode token differs from the official profile",
    )
    return {
        "history_token_sha256": run["prompt_sha256"],
        "history_tokens": 65536,
        "decode_input_token": tokens[65536],
        "official_output_ids": complete["row"]["output_ids"],
        "local_full_token_sha256": local["full_token_sha256"],
        "exact_history_arrays_equal": True,
        "exact_decode_token_equal": True,
        "local_sampling_included": False,
    }, [run_path, complete_path, request_path]


def load_comparison(official_directory, warm_directory, *, official_run_directory=None):
    official_publication = validate_manifest(official_directory, "publication.json", "files")
    official, official_paths = decode.official_panels(official_directory)
    official = next(panel for panel in official if panel["method"] == "echo")
    local, local_paths, warm = validate_warm(warm_directory)
    if official_run_directory is None:
        official_run_directory = EXPERIMENT / "output/data" / official["profile_run_id"]
    identity, identity_paths = validate_input_identity(
        comparison.read(official_directory / "report.json"), warm, official_run_directory
    )
    identity["separate_recall_observation"] = {
        "run_id": warm["audit"]["official_run_id"],
        "observation_sha256": warm["audit"]["official_observation_sha256"],
        "residual_recall": warm["audit"]["official_residual_recall"],
        "boundary": "Separate observer run; never assigned to the old profile's traffic.",
    }
    require(
        warm["audit"]["official_residual_recall"]
        == [{"d2d_records": 0, "h2d_records": 0, "layer": layer} for layer in range(3)],
        "Separate recall observation differs",
    )
    official_paths.extend(official_directory / name for name in official_publication["files"])
    return (
        [official, local],
        list(dict.fromkeys([*official_paths, *local_paths, *identity_paths])),
        identity,
    )


def interval_metrics(panel):
    """Make a disjoint elapsed partition; retain lane unions separately from it."""
    window = panel["window"]
    start, end = window["start_ns"], window["end_ns"]
    rows = panel["rows"]

    def intervals(predicate):
        return [
            (max(start, row["start_ns"]), min(end, row["end_ns"]))
            for row in rows
            if predicate(row) and row["end_ns"] > start and row["start_ns"] < end
        ]

    all_intervals = intervals(lambda row: True)
    compute = intervals(lambda row: row["lane"] == "Compute")
    # Display-only orange recall has unknown bytes and is retained as control here.
    io = intervals(
        lambda row: row["lane"] in {"H2D", "D2H"} and row["category"] != "ECHO recall (IO unknown)"
    )
    busy = engine.duration(all_intervals)
    compute_ns = engine.duration(compute)
    compute_io_ns = engine.duration([*compute, *io])
    totals = {
        "window": end - start,
        "idle": end - start - busy,
        "compute_union": compute_ns,
        "io_only": compute_io_ns - compute_ns,
        "control_only": busy - compute_io_ns,
        "original_control_lane_union": engine.duration(
            intervals(lambda row: row.get("source_lane", row["lane"]) == "GPU control")
        ),
        "source_control_category_union": engine.duration(
            intervals(lambda row: row.get("source_category", row["category"]) == "GPU control")
        ),
    }
    require(
        totals["window"] == round(window["window_ms"] * 1e6)
        and totals["idle"] == round(window["gpu_idle_ms"] * 1e6)
        and totals["window"]
        == sum(totals[key] for key in ("idle", "compute_union", "io_only", "control_only")),
        "GPU elapsed partition does not conserve the published window",
    )
    return {
        **{f"{key}_ns": value for key, value in totals.items()},
        **{f"{key}_ms": value / 1e6 for key, value in totals.items()},
    }


def report_text(summary):
    rows = summary["windows"]
    contribution = summary["control_and_idle_share_of_window_difference_percent"]
    contribution_text = (
        f"两者合计占窗口差的 {contribution:.2f}%。\n\n"
        if contribution is not None
        else "两侧窗口相同，不计算贡献比例。\n\n"
    )
    table = "\n".join(
        f"| {row['source']} | {row['window_ms']:.6f} | {row['compute_union_ms']:.6f} | {row['control_only_ms']:.6f} | {row['io_only_ms']:.6f} | {row['idle_ms']:.6f} | {row['original_control_lane_union_ms']:.6f} |"
        for row in rows
    )
    return (
        "# 官方正常 decode 与本地 warm ECHO\n\n"
        f"展示运行 `{summary['run_id']}` 复用已发布的 L0–L2 profile，未新增 GPU 测量。"
        "两侧 history token 数组一致，H=65,536、Q=1，输入 token 均为 57841；本地显式输入该 token，不包含采样。"
        "本地保留 prefill 后的主 KV 驻留，官方保留正常 radix 与自然驻留。\n\n"
        "两侧使用同一绘图函数、颜色、泳道高度和时间轴。蓝色 Indexer 统一包含评分、因果清理及完整 top-k 输出流程，"
        "包括初始化、选择、转换、排序和 mask；投影、RoPE、量化与 head weights 归 Projection，"
        "调度元数据、索引打包、缓存管理和 hint 更新归 control。"
        "按所有 stream 的 GPU 区间并集统计，并裁到相同的三层窗口；融合 kernel 保留完整时长。"
        "[逐层分类统计](category_metrics.csv)采用同一口径。\n\n"
        "![两侧 ECHO 时间线](timeline_decode.svg)\n\n"
        "两侧融合评分/预取 kernel 都画斜线，即使实际搬运为 0 B。"
        "斜线表示 kernel 包含 mapped-host 预取路径，不能当作实际 H2D 时间或字节数；"
        "本地 warm 的预取计数仍为 0。\n\n"
        "| 来源 | 窗口 ms | Compute 并集 ms | Control-only ms | 明确 IO 独占 ms | Idle ms | 原 Control lane 并集 ms |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + table
        + "\n\nControl-only 是 Compute 和明确 IO 之外仍有 GPU 活动的区间。"
        "前三项工作区间与 idle 构成互不重叠的窗口分解；原 control lane 并集单列，不能与它们重复相加。"
        "融合 indexer 保留整段 Compute 时间，图中的橙色 recall 条按功能展示，不能当作纯 H2D。\n\n"
        f"本地窗口少 {summary['official_minus_local_ms']['window'] * 1000:.3f} µs，"
        f"其中 control 独占区间少 {summary['official_minus_local_ms']['control_only'] * 1000:.3f} µs、"
        f"idle 少 {summary['official_minus_local_ms']['idle'] * 1000:.3f} µs，"
        + contribution_text
        + f"官方旧 trace 的 recall 活动共 {summary['official_profile_recall_activity_us']:.3f} µs，实际 mapped-host 字节未知。"
        "另一次独立 observer 运行观测到三层 residual recall 都为 0 条、0 B，"
        "这一计数不回填旧 trace，也不覆盖融合 indexer 的预测流量。"
        "本地 warm 的正式计时与 profile 计数均为每层 H2D 0 B、D2H 1,152 B。\n\n"
        "两侧使用不同的框架环境、GPU、pool 容量及缓存操作顺序。"
        "官方跨实现数值验收仍为 `false`；输入一致不证明模型数值或缓存策略等价。"
        "这些侵入式窗口不等于完整请求延迟，不用于计算配对加速比，也不能把窗口差直接归为缓存策略收益。"
        "原始 `panels.json` 保留已认证活动，`aligned_panels.json` 只更新功能分类，保留原类别与逐项分类理由。"
        "输入报告副本和源码快照保留在本次 `output/data/` 运行目录；"
        "[来源](provenance.json)、[窗口](windows.csv)和[发布清单](publication_manifest.json)绑定展示素材。\n"
    )


def build_report(official_directory, warm_directory, output, publish_dir=None):
    output = output.resolve()
    require(
        output.parent == (EXPERIMENT / "output/data").resolve(),
        "Output must be a new run ID under this experiment's output/data",
    )
    if publish_dir is not None:
        publish_dir = publish_dir.resolve()
        require(
            publish_dir.is_relative_to((EXPERIMENT / "report").resolve()),
            "Publication must be under this experiment's report",
        )
    if output.exists() or (publish_dir is not None and publish_dir.exists()):
        raise FileExistsError("Use new output and publication directories")
    panels, inputs, identity = load_comparison(official_directory, warm_directory)
    aligned = [categories.align_panel(panel) for panel in panels]
    sources = list(
        dict.fromkeys(
            [
                Path(__file__),
                Path(decode.__file__),
                Path(comparison.__file__),
                Path(engine.__file__),
                Path(comparison.compact.__file__),
                Path(comparison.timeline.__file__),
                Path(decode.decode_display.__file__),
                Path(categories.__file__),
            ]
        )
    )
    paths = list(dict.fromkeys([*inputs, *sources]))
    before = {str(path.resolve()): engine.sha256(path) for path in paths}
    output.mkdir(parents=True)
    decode.draw(
        aligned,
        output,
        xlim_ms=(0, 2),
        figure_title="H64K · Q1 · L0–L2 | Official decode and local warm ECHO",
        compact_style=True,
    )
    windows = [
        {
            "source": panel["source"],
            "method": panel["method"],
            "condition": panel["condition"],
            "profile_run_id": panel["profile_run_id"],
            "start_ns": panel["window"]["start_ns"],
            "end_ns": panel["window"]["end_ns"],
            **interval_metrics(panel),
        }
        for panel in aligned
    ]
    engine.write_csv(output / "windows.csv", windows)
    write_json(output / "panels.json", panels)
    write_json(output / "aligned_panels.json", aligned)
    engine.write_csv(
        output / "category_metrics.csv",
        [row for panel in aligned for row in categories.category_metrics(panel)],
    )
    differences = {
        key: (windows[0][f"{key}_ns"] - windows[1][f"{key}_ns"]) / 1e6
        for key in ("window", "compute_union", "control_only", "io_only", "idle")
    }
    summary = {
        "schema": "official-local-warm-echo-comparison-v1",
        "run_id": output.name,
        "passed": True,
        "numerical_acceptance_between_implementations": False,
        "measurement": "Existing independent intrusive NSYS L0-L2 windows; no new GPU execution.",
        "category_policy": categories.POLICY,
        "windows": windows,
        "official_minus_local_ms": differences,
        "control_and_idle_share_of_window_difference_percent": (
            100 * (differences["control_only"] + differences["idle"]) / differences["window"]
            if differences["window"]
            else None
        ),
        "input_identity": identity,
        "official_profile_actual_mapped_host_bytes": None,
        "official_profile_recall_activity_us": engine.duration(
            [
                (row["start_ns"], row["end_ns"])
                for row in panels[0]["rows"]
                if row["category"] == "ECHO recall (IO unknown)"
            ]
        )
        / 1000,
        "local_warm_per_layer_bytes": {"host_to_device": 0, "device_to_host": 1152},
    }
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(report_text(summary))
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    copies = output / "input_reports"
    copies.mkdir()
    shutil.copytree(official_directory, copies / "official")
    shutil.copytree(warm_directory, copies / "warm")
    display = comparison.read(warm_directory / "display_provenance.json")
    shutil.copytree(Path(display["source_report"]), copies / "warm_original")
    audit_publication = comparison.read(
        warm_directory / "input_recall_audit/publication_manifest.json"
    )
    shutil.copytree(
        Path(audit_publication["raw_publication"]["path"]).parent, copies / "input_audit_original"
    )
    identity_copies = copies / "token_identity"
    identity_copies.mkdir()
    for name, path in zip(
        ("official_run.json", "official_complete.json", "local_request.json"),
        inputs[-3:],
        strict=True,
    ):
        shutil.copyfile(path, identity_copies / name)
    require(
        before == {str(path.resolve()): engine.sha256(path) for path in paths},
        "Input or renderer changed during rendering",
    )
    provenance = {
        "schema": "official-local-warm-echo-comparison-provenance-v1",
        "run_id": output.name,
        "inputs_and_sources_sha256": before,
        "profile_run_ids": [panel["profile_run_id"] for panel in panels],
        "local_panel_provenance": panels[1]["provenance"],
        "shared_xlim_ms": [0, 2],
        "category_policy": categories.POLICY,
        "source_snapshot_sha256": {path.name: engine.sha256(path) for path in snapshot.iterdir()},
        "output_evidence_sha256": {
            path.relative_to(output).as_posix(): engine.sha256(path)
            for path in output.rglob("*")
            if path.is_file()
        },
        "audit": {
            "all_publication_payload_hashes": True,
            "original_warm_and_input_audit_source_chain": True,
            "official_profile_input_identity": True,
            "original_windows_and_profile_ids": True,
            "disjoint_interval_conservation": True,
            "source_to_aligned_activity_identity": True,
        },
        "io_display_policy": decode.decode_display.RECALL_DISPLAY,
        "fused_display_policy": decode.decode_display.FUSED_DISPLAY,
    }
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "publication_manifest.json",
        {
            "schema": "official-local-warm-echo-display-artifacts-v1",
            "run_id": output.name,
            "files_sha256": {
                name: engine.sha256(output / name)
                for name in PUBLISHED_FILES
                if name != "publication_manifest.json"
            },
        },
    )
    if publish_dir is not None:
        publish_dir.mkdir(parents=True)
        for name in PUBLISHED_FILES:
            shutil.copyfile(output / name, publish_dir / name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-report", type=Path, default=EXPERIMENT / "report/sglang_decode")
    parser.add_argument("--warm-report", type=Path, default=comparison.MFU / "report/h64k_a1_warm")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path)
    args = parser.parse_args()
    build_report(args.official_report, args.warm_report, args.output_dir, args.publish_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
