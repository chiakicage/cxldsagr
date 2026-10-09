"""Report accepted Q1 bridge, packing, promotion and score-layout measurements."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_echo_official.src import report_q1_followup as followup
from experiments.deepseek_v32_echo_official.src import report_q1_optimization as common

EXPERIMENT = common.EXPERIMENT
ROOT = common.ROOT
PROMOTION_KIND = "deepseek-official-q1-promotion-candidate-v1"


def bind_record(evidence, record):
    path = Path(record["path"])
    common.require(evidence.bind(path) == record["sha256"], f"Artifact changed: {path}")
    if "bytes" in record:
        common.require(path.stat().st_size == record["bytes"], f"Artifact size changed: {path}")


def bind_native(evidence, native):
    bind_record(evidence, {"path": native["artifact_path"], "sha256": native["artifact_sha256"]})


def promotion_data(evidence, directory):
    directory = directory.resolve(strict=True)
    bench = evidence.read("promotion_bench", directory / "result.json")
    summary = evidence.read("promotion_inventory", directory / "summary.json")
    identity = bench["identity"]
    common.require(bench["passed"] is True and bench["mode"] == "bench", "Unaccepted promotion")
    common.require(bench["run_id"] == directory.name == summary["run_id"], "Promotion run differs")
    common.require(
        evidence.bind(directory / "result.json") == summary["benchmark_result_sha256"],
        "Promotion result changed",
    )
    common.require(
        summary["source_sha256"] == identity["source_sha256"], "Promotion source differs"
    )
    check_directory = Path(summary["independent_check"]["path"]).resolve(strict=True)
    graph_directory = Path(summary["graph_stream_check"]["path"]).resolve(strict=True)
    receipts = []
    for name, path, kind, expected in (
        ("promotion", check_directory, PROMOTION_KIND, summary["independent_check"]),
        (
            "promotion_graph",
            graph_directory,
            PROMOTION_KIND + "-graph-stream",
            summary["graph_stream_check"],
        ),
    ):
        receipt_path = path / "receipt.json"
        verified = require_receipt(receipt_path, kind=kind, identity=identity)
        common.require(verified["receipt_sha256"] == expected["receipt_sha256"], "Receipt changed")
        evidence.read(name + "_receipt", receipt_path)
        for label, artifact in verified["artifact_paths"].items():
            artifact = Path(artifact)
            if artifact.suffix == ".json":
                evidence.read(name + "_" + Path(label).stem, artifact)
            else:
                evidence.bind(artifact)
        receipts.append(verified)
    check, graph_check = (receipt["checks"] for receipt in receipts)
    common.require(
        check["custom_case_count"] == len(check["custom_cases"]) == 84
        and check["adapter_tests"]
        == {"module_only_substitution": True, "passed": 22, "skipped": 0},
        "Promotion check coverage differs",
    )
    invalid = check["invalid_inputs"]
    common.require(
        {row["case"] for row in invalid}
        == {"duplicate_slot", "invalid_host", "owner_mismatch", "occupied_journal"}
        and len(invalid) == 4
        and all(row["device_assert"] is True and row["returncode"] != 0 for row in invalid),
        "Promotion negative checks differ",
    )
    common.require(
        graph_check["graph_replays"] == 72 and graph_check["nondefault_stream_calls"] == 24,
        "Promotion graph/stream coverage differs",
    )
    native = evidence.read("promotion_native", directory / "native.json")
    checked_native = json.loads(Path(receipts[0]["artifact_paths"]["native.json"]).read_text())
    common.require(native == checked_native, "Promotion native differs from acceptance")
    common.require(set(native) == {"baseline", "candidate"}, "Missing promotion native arm")
    for arm, record in native.items():
        bind_native(evidence, record)
        common.require(
            identity["native"][arm]
            == {
                "path": record["artifact_path"],
                "sha256": record["artifact_sha256"],
                "build_identity_sha256": identity_digest(record["build_identity"]),
            },
            "Promotion native identity differs",
        )
    # These runs deliberately use archived baseline/candidate sources. The
    # current production adapter may have advanced since the measured run.
    for path in (directory, check_directory):
        for filename, expected in identity["source_sha256"].items():
            relative = Path(filename).relative_to(ROOT)
            common.require(
                evidence.bind(path / "source" / relative) == expected,
                "Promotion source snapshot changed",
            )
    source_input = EXPERIMENT / "output/data/q1_inputs_20261008_01/kernel_inputs_layer_2.pt"
    common.require(
        evidence.bind(source_input) == identity["input_sha256"], "Promotion input changed"
    )
    common.require(
        identity["device_capability"] == [9, 0]
        and bench["cuda_visible_devices"] == "1"
        and bench["cpu_affinity"] == list(range(8, 16))
        and bench["warmups"] == 20
        and bench["repeats"] == 100,
        "Promotion measurement conditions differ",
    )
    hardware = directory / "hardware.csv"
    common.require(
        evidence.bind(hardware) == summary["hardware_csv_sha256"], "Hardware record changed"
    )
    rows, samples, keys = [], [], set()
    for case in bench["cases"]:
        key = (case["layout"], case["occupied_selected_slots"])
        common.require(key not in keys, "Duplicate promotion condition")
        keys.add(key)
        common.require(
            (
                case["N"],
                case["P"],
                case["stage_records"],
                case["record_width"],
                case["record_dtype"],
                case["offset_bytes"],
            )
            == (65537, 65600, 64, 576, "bfloat16", 0),
            "Promotion shape differs",
        )
        values = case["samples_us"]
        common.require(set(values) == {"baseline", "candidate"}, "Missing promotion samples")
        wins = sum(c < b for c, b in zip(values["candidate"], values["baseline"], strict=True))
        common.require(wins == case["candidate_paired_wins"], "Promotion paired wins differ")
        for arm, gpu in values.items():
            common.require(
                len(gpu) == 100 and all(math.isfinite(value) and value > 0 for value in gpu),
                "Incomplete or invalid promotion timing",
            )
            common.require(
                statistics.median(gpu) == case["median_us"][arm], "Promotion median differs"
            )
            base = {
                "operator": "promotion",
                "run_id": bench["run_id"],
                "layout": key[0],
                "occupied_selected_slots": key[1],
                "method": arm,
                "execution": "graph",
                "samples": 100,
                "calls_per_sample": 1,
                "warmups": 20,
            }
            rows.append(
                {
                    **base,
                    "gpu_us_median": statistics.median(gpu),
                    "gpu_us_min": min(gpu),
                    "gpu_us_max": max(gpu),
                    "candidate_paired_wins": wins,
                }
            )
            samples.extend(
                {**base, "pair": index, "gpu_us": value} for index, value in enumerate(gpu)
            )
    common.require(
        keys
        == {(layout, occupied) for layout in ("consecutive", "random") for occupied in (0, 32, 64)},
        "Incomplete promotion matrix",
    )
    evidence.identities["promotion"] = {
        "check_run_id": check_directory.name,
        "graph_check_run_id": graph_directory.name,
        "bench_run_id": bench["run_id"],
        "original_identity": identity,
    }
    return {
        "bench": bench,
        "check": check,
        "graph_check": graph_check,
        "native": native,
        "rows": rows,
        "samples": samples,
    }


def score_layout_data(evidence, directory):
    directory = directory.resolve(strict=True)
    bench = evidence.read("score_layout_bench", directory / "result.json")
    check_path = Path(bench["receipt"]["path"]).resolve(strict=True)
    check = evidence.read("score_layout_check", check_path)
    common.require(
        evidence.bind(check_path) == bench["receipt"]["sha256"], "Score-layout receipt changed"
    )
    common.require(
        check["accepted"] is True
        and check["mode"] == "check"
        and bench["accepted"] is True
        and bench["mode"] == "bench"
        and check["identity"] == bench["identity"]
        and check["native"] == bench["native"],
        "Score-layout acceptance, identity or native differs",
    )
    identity = bench["identity"]
    for path in (directory, check_path.parent):
        for filename, expected in identity["harness_sha256"].items():
            common.require(
                evidence.bind(path / "source" / filename) == expected,
                "Score-layout harness changed",
            )
    # This harness archives local Python sources; upstream C++ is fingerprinted
    # in its identity and still present in the pinned dependency checkout.
    for field in ("source_sha256", "inputs_sha256"):
        for filename, expected in identity[field].items():
            common.require(
                evidence.bind(ROOT / filename) == expected, "Score-layout source/input changed"
            )
    common.require(
        identity["physical_device"] == 2
        and identity["capability"] == [9, 0]
        and identity["cpu_affinity"] == list(range(16, 24)),
        "Score-layout measurement conditions differ",
    )
    native = bench["native"]
    bind_native(evidence, native["official"])
    common.require(
        any(
            item["name"] == "topk" and item["loaded_in_this_process"]
            for item in native["flashinfer"]
        ),
        "No loaded top-k artifact",
    )
    for artifact in native["flashinfer"]:
        for record in [artifact["library"], artifact["build_metadata"], *artifact["sources"]]:
            bind_record(evidence, record)
    common.require(bool(native["mask"]), "Missing nonfinite-mask compiled identity")
    expected_cases = {f"kernel_inputs_layer_{layer}" for layer in range(3)}
    common.require(
        len(check["results"]) == 3
        and {case["case"] for case in check["results"]} == expected_cases
        and all(
            case["bitwise_variants"] == ["actual", "ties", "changed", "causal"]
            and case["replays"] == 4
            for case in check["results"]
        ),
        "Score-layout exact-check coverage differs",
    )
    rows, samples, cases = [], [], set()
    for case in bench["results"]:
        common.require(
            case["case"] in expected_cases and case["case"] not in cases,
            "Unexpected score-layout case",
        )
        cases.add(case["case"])
        common.require(case["calls_per_sample"] == 1, "Score-layout call count differs")
        detail = case["samples"]
        expected_order = [
            (pair, arm)
            for pair in range(30)
            for arm in (("logical", "padded") if pair % 2 == 0 else ("padded", "logical"))
        ]
        common.require(
            [(row["pair"], row["arm"]) for row in detail] == expected_order,
            "Incomplete alternating score-layout pairs",
        )
        common.require(
            all(math.isfinite(row["gpu_us"]) and row["gpu_us"] > 0 for row in detail),
            "Invalid score-layout timing",
        )
        values = {
            arm: [row["gpu_us"] for row in detail if row["arm"] == arm]
            for arm in ("logical", "padded")
        }
        wins = sum(
            p < logical for p, logical in zip(values["padded"], values["logical"], strict=True)
        )
        for arm, gpu in values.items():
            common.require(
                statistics.median(gpu) == case["median_gpu_us"][arm], "Score-layout median differs"
            )
            base = {
                "operator": "score_layout",
                "run_id": directory.name,
                "layer": int(case["case"][-1]),
                "method": arm,
                "execution": "graph",
                "samples": 30,
                "calls_per_sample": 1,
                "warmups": 10,
            }
            rows.append(
                {
                    **base,
                    "gpu_us_median": statistics.median(gpu),
                    "gpu_us_min": min(gpu),
                    "gpu_us_max": max(gpu),
                    "candidate_paired_wins": wins,
                }
            )
            samples.extend(
                {**base, "pair": index, "gpu_us": value} for index, value in enumerate(gpu)
            )
    common.require(cases == expected_cases, "Incomplete score-layout matrix")
    evidence.identities["score_layout"] = {
        "check_run_id": check_path.parent.name,
        "bench_run_id": directory.name,
        "original_identity": identity,
    }
    return {"bench": bench, "check": check, "rows": rows, "samples": samples}


def pair(evidence, name, directory):
    bench_path = directory.resolve(strict=True) / "result.json"
    bench = json.loads(bench_path.read_text())
    check_path = Path(bench["receipt"]["path"]).resolve(strict=True)
    check, bench = evidence.pair(name, check_path, bench_path)
    common.require(check["mode"] == "check", "Expected an independent check")
    for path, data in ((check_path, check), (bench_path, bench)):
        for field in ("harness_sha256",):
            for relative, expected in data["identity"].get(field, {}).items():
                snapshot = path.parent / "source" / relative
                common.require(evidence.bind(snapshot) == expected, "Harness snapshot changed")
    for filename, expected in bench["identity"]["inputs_sha256"].items():
        common.require(evidence.bind(ROOT / filename) == expected, "Measured input changed")
    if name == "official":
        common.require(check["identity"] == bench["identity"], "Official identities differ")
        common.require(check["native"] == bench["native"], "Official native bytes differ")
        native = bench["native"]
        common.require(
            evidence.bind(Path(native["artifact_path"])) == native["artifact_sha256"],
            "Official immutable artifact changed",
        )
    else:
        accepted = {item["hash"]: item for item in check["compiled_kernels"]}
        common.require(
            bool(bench["compiled_kernels"])
            and all(accepted.get(item["hash"]) == item for item in bench["compiled_kernels"]),
            "Packing compiled identity differs from acceptance",
        )
    return check, bench


def timing(bench, operator):
    summaries, samples, keys = [], [], set()
    for row in bench["rows" if operator == "official" else "samples"]:
        match = re.fullmatch(r"kernel_inputs_layer_([0-2])(?:\.pt)?", row["case"])
        common.require(match is not None, "Expected real L0-L2 inputs")
        layer = int(match[1])
        condition = row["policy"] if operator == "official" else "resident"
        method = "official" if operator == "official" else row["method"]
        key = (layer, condition, row["boundary"], method, row["execution"])
        common.require(key not in keys, "Duplicate timing condition")
        keys.add(key)
        gpu, wall = row["gpu_us_samples"], row["wall_us_samples"]
        count = 30 if operator == "official" else 7
        calls = row["calls_per_sample"] if operator == "official" else row["iterations"]
        common.require(len(gpu) == len(wall) == count, "Incomplete timing samples")
        common.require(calls == (1 if operator == "official" else 100), "Call count changed")
        common.require(
            row["warmups"] == (10 if operator == "official" else 20), "Warmup count changed"
        )
        common.require(
            all(math.isfinite(value) and value > 0 for value in [*gpu, *wall]),
            "Invalid timing value",
        )
        common.require(statistics.median(gpu) == row["gpu_us_median"], "Median differs")
        attempts = row.get("reservation_attempts_samples", [None] * count)
        records = row.get("prefetched_records_samples", [None] * count)
        h2d = row.get("h2d_bytes_samples", [None] * count)
        common.require(len(attempts) == len(records) == len(h2d) == count, "Traffic is incomplete")
        if operator == "official":
            common.require(row["repeats"] == count, "Official repeat count differs")
            for reserved, staged, transferred in zip(attempts, records, h2d, strict=True):
                common.require(
                    isinstance(reserved, int)
                    and reserved >= 0
                    and staged == min(reserved, 64)
                    and transferred == staged * 1152,
                    "Official per-sample traffic differs from the cap and record width",
                )
                common.require(staged == (64 if condition == "zero" else 0), "Policy differs")
        else:
            common.require(row["tokens"] == 65537, "Packing context differs")
        base = {
            "operator": operator,
            "run_id": bench["run_id"],
            "layer": layer,
            "condition": condition,
            "boundary": row["boundary"],
            "method": method,
            "execution": row["execution"],
            "samples": count,
            "calls_per_sample": calls,
            "warmups": row["warmups"],
        }
        summaries.append(
            {
                **base,
                "gpu_us_median": statistics.median(gpu),
                "gpu_us_min": min(gpu),
                "gpu_us_max": max(gpu),
                "wall_us_median": statistics.median(wall),
            }
        )
        samples.extend(
            {
                **base,
                "sample": index,
                "gpu_us": gpu[index],
                "wall_us": wall[index],
                "reservation_attempts": attempts[index],
                "prefetched_records": records[index],
                "h2d_bytes": h2d[index],
            }
            for index in range(count)
        )
    conditions = ("zero", "warm") if operator == "official" else ("resident",)
    boundaries = ("prepared", "full") if operator == "official" else ("packing", "full_mqa")
    methods = ("official",) if operator == "official" else ("copies", "fused")
    common.require(
        keys
        == {
            (layer, condition, boundary, method, execution)
            for layer in range(3)
            for condition in conditions
            for boundary in boundaries
            for method in methods
            for execution in ("eager", "graph")
        },
        "Incomplete timing matrix",
    )
    return summaries, samples


def figures(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": output.name})
    configurations = (
        (
            "packing_latency",
            "Page64 packing and resident paged MQA · Q1, N = 65,537",
            ("packing", "full_mqa"),
            ("Packing only", "Resident MQA, including packing"),
            "boundary",
            "packing",
            "method",
            ("copies", "fused"),
            ("Tensor copies", "Fused packing"),
            "GPU0 · median of 7 batches × 100 calls; whiskers: full batch range. Exact top-k excluded.",
        ),
        (
            "official_prefetch_latency",
            "Official ECHO Q1 source bridge · N = 65,537, cap = 64",
            ("zero", "warm"),
            (
                "Cold map, threshold 0: 64 records prefetched",
                "Resident map: zero records prefetched",
            ),
            "condition",
            "official",
            "boundary",
            ("prepared", "full"),
            ("Prepared inputs", "Including packing + metadata"),
            (
                "GPU1 · log scale · median of 30 single replays; whiskers: full sample range, including outliers.\n"
                "Exact top-k, staging promotion and residual recall excluded; resident map is a raw-kernel condition."
            ),
        ),
    )
    for (
        name,
        title,
        values,
        titles,
        field,
        operator,
        series,
        methods,
        labels,
        caption,
    ) in configurations:
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
        for axis, value, subtitle in zip(axes, values, titles, strict=True):
            selected = [
                {**row, "case": row["layer"], "method": row[series]}
                for row in rows
                if row["operator"] == operator
                and row["execution"] == "graph"
                and row[field] == value
            ]
            if operator == "packing":
                common.plot_bars(axis, selected, methods, labels, list(range(3)))
                axis.set_ylim(0, max(row["gpu_us_max"] for row in selected) * 1.30)
                axis.legend(frameon=False, fontsize=9, loc="upper right")
            else:
                for index, (method, label) in enumerate(zip(methods, labels, strict=True)):
                    points = [row for row in selected if row["method"] == method]
                    x = [row["layer"] + (index - 0.5) * 0.34 for row in points]
                    y = [row["gpu_us_median"] for row in points]
                    axis.errorbar(
                        x,
                        y,
                        yerr=[
                            [row["gpu_us_median"] - row["gpu_us_min"] for row in points],
                            [row["gpu_us_max"] - row["gpu_us_median"] for row in points],
                        ],
                        fmt="o" if index == 0 else "s",
                        color="#555555" if index == 0 else "#0072B2",
                        markerfacecolor="white" if index == 0 else "#0072B2",
                        elinewidth=1,
                        capsize=3,
                        markersize=6,
                        label=label,
                    )
                    for left, height in zip(x, y, strict=True):
                        axis.annotate(
                            f"{height:.1f}",
                            (left, height),
                            xytext=(-7 if index == 0 else 7, 0),
                            textcoords="offset points",
                            ha="right" if index == 0 else "left",
                            va="center",
                            fontsize=9,
                        )
                axis.set_yscale("log")
                axis.set_ylim(
                    7,
                    max(
                        row["gpu_us_max"]
                        for row in rows
                        if row["operator"] == "official" and row["execution"] == "graph"
                    )
                    * 2,
                )
                axis.set_xticks(range(3))
                axis.set_xlim(-0.5, 2.5)
                axis.grid(axis="y", color="#dddddd", linewidth=0.6)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
                axis.legend(frameon=False, fontsize=9, loc="upper right")
            axis.set_xticklabels([f"Layer {layer}" for layer in range(3)])
            axis.set_ylabel("CUDA Graph GPU time (µs)")
            axis.set_title(subtitle, fontsize=10)
        figure.suptitle(title, fontsize=12)
        figure.supxlabel(caption, fontsize=9)
        for extension in ("svg", "pdf", "png"):
            figure.savefig(output / f"{name}.{extension}", dpi=180)
        plt.close(figure)


def component_figures(output, components):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": output.name})

    def bars(axis, rows, cases, methods, labels, case_field):
        for index, (method, label) in enumerate(zip(methods, labels, strict=True)):
            selected = [
                next(row for row in rows if row["method"] == method and row[case_field] == case)
                for case in cases
            ]
            x = [position + (index - 0.5) * 0.36 for position in range(len(cases))]
            median = [row["gpu_us_median"] for row in selected]
            color = "#666666" if index == 0 else "#0072B2"
            axis.bar(
                x,
                median,
                width=0.34,
                facecolor="white" if index == 0 else color,
                edgecolor=color,
                linewidth=1.2,
                label=label,
            )
            axis.errorbar(
                x,
                median,
                yerr=[
                    [row["gpu_us_median"] - row["gpu_us_min"] for row in selected],
                    [row["gpu_us_max"] - row["gpu_us_median"] for row in selected],
                ],
                fmt="none",
                ecolor=color,
                capsize=3,
                linewidth=0.9,
            )
            for position, row in zip(x, selected, strict=True):
                axis.annotate(
                    f"{row['gpu_us_median']:.1f}",
                    (position, row["gpu_us_max"]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
        axis.set_xticks(range(len(cases)), [str(case) for case in cases])
        axis.set_ylim(0, max(row["gpu_us_max"] for row in rows) * 1.28)
        axis.set_ylabel("GPU event time (µs)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=9)

    def save(figure, name):
        for extension in ("svg", "pdf", "png"):
            figure.savefig(output / f"{name}.{extension}", dpi=180)
        plt.close(figure)

    for name in ("promotion", "validation"):
        if name not in components:
            continue
        rows = components[name]["rows"]
        figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained", sharey=True)
        for axis, layout in zip(axes, ("consecutive", "random"), strict=True):
            selected = [row for row in rows if row["layout"] == layout]
            bars(
                axis,
                selected,
                (0, 32, 64),
                ("baseline", "candidate"),
                ("Serial copy", "Parallel record copy")
                if name == "promotion"
                else ("64-thread validation", "256-thread validation"),
                "occupied_selected_slots",
            )
            axis.set_xlabel("Occupied selected slots")
            axis.set_title(f"{layout.capitalize()} destination slots", fontsize=10)
        figure.suptitle("Staging promotion · 64 BF16 records × 576 elements", fontsize=11)
        save(figure, name + "_latency")
    if "topk" in components:
        figure, axis = plt.subplots(figsize=(5.1, 3.6), layout="constrained")
        bars(
            axis,
            [row for row in components["topk"]["rows"] if row["calls_per_sample"] == 1],
            (0, 1, 2),
            ("baseline", "candidate"),
            ("FlashInfer sorted + mask", "FlashInfer + CUB sort/mask"),
            "layer",
        )
        axis.set_xlabel("Layer")
        axis.set_title("Complete exact top-k · Q1, k = 2,048", fontsize=10)
        save(figure, "topk_latency")
    if "score_layout" in components:
        figure, axis = plt.subplots(figsize=(5.1, 3.6), layout="constrained")
        bars(
            axis,
            components["score_layout"]["rows"],
            (0, 1, 2),
            ("logical", "padded"),
            ("Logical score view", "Padded score view"),
            "layer",
        )
        axis.set_xlabel("Layer")
        axis.set_title("Complete exact top-k · Q1, N = 65,537, k = 2,048", fontsize=10)
        save(figure, "score_layout_latency")


def component_report(components, identities):
    text = []
    if "promotion" in components:
        data = components["promotion"]
        rows = data["rows"]
        baseline = [row["gpu_us_median"] for row in rows if row["method"] == "baseline"]
        candidate = [row["gpu_us_median"] for row in rows if row["method"] == "candidate"]
        wins = sum(row["candidate_paired_wins"] for row in rows if row["method"] == "candidate")
        source = identities["promotion"]
        text.append(
            "\n## Stage promotion\n\n"
            f"并行 record copy 将六种槽位条件下的完整 promotion 调用中位数从 {min(baseline):.3f}–{max(baseline):.3f} µs "
            f"降至 {min(candidate):.3f}–{max(candidate):.3f} µs，{wins}/600 组配对更快。\n\n"
            "![两种槽位布局下的完整 stage promotion](promotion_latency.svg)\n\n"
            "输入为真实 layer 2 的 64 条 BF16 record，每条 576 个元素，N=65,537、P=65,600。"
            "候选先用一个 CTA 验证元数据，再用 64 个 CTA 分别搬运 record 并发布映射；两个 kernel 都计时，新增 scratch 为 0。"
            "图中分别比较连续与随机目标槽，已有 owner 的目标槽数为 0、32、64；柱高为中位数，误差线覆盖全部样本的最小值至最大值。\n\n"
            "GPU1（PyTorch 识别为 H200 / SM90）、CPU 8–15，每个条件预热 20 次，再交替执行 100 组 baseline/candidate 配对。"
            "GPU event 节点捕获在完整 promote 调用的前后，metadata restore 与主机 replay dispatch 在计时外，不清空 cache。"
            "这里不包含官方打分、stage H2D、exact top-k 或 residual recall。\n\n"
            f"独立验收 `{source['check_run_id']}` 覆盖 84 个逐位状态案例、22 项原 adapter 测试和 4 类 device assert；"
            f"`{source['graph_check_run_id']}` 另覆盖 72 次 graph replay 与 24 次非默认 stream 调用。"
            "报告核验两份 receipt 的签名、artifact、候选私有源码快照与 immutable native；这些逐位结果来自运行时检查，"
            "报告生成没有重新运行 GPU。\n\n"
            f"正式计时：`{source['bench_run_id']}`。数据见 [promotion 汇总](promotion_latency.csv)和[全部配对样本](promotion_samples.csv)。\n"
        )
    if "score_layout" in components:
        rows = components["score_layout"]["rows"]
        logical = [row["gpu_us_median"] for row in rows if row["method"] == "logical"]
        padded = [row["gpu_us_median"] for row in rows if row["method"] == "padded"]
        wins = sum(row["candidate_paired_wins"] for row in rows if row["method"] == "padded")
        source = identities["score_layout"]
        text.append(
            "\n## Score padding 与 exact top-k\n\n"
            f"复用官方 logits 已初始化为 `-inf` 的物理尾部后，三层完整 exact top-k 的调用中位数从 {min(logical):.3f}–{max(logical):.3f} µs "
            f"降至 {min(padded):.3f}–{max(padded):.3f} µs，{wins}/90 组配对更快。\n\n"
            "![逻辑 score view 与 padded score view 的 exact top-k](score_layout_latency.svg)\n\n"
            "两组使用同一份官方 FP32 logits，Q=1、N=65,537、k=2,048。Padded view 只暴露已有物理尾部，"
            "不复制分数，也不增加 kernel；打分与 view 准备在两组计时之外。测量包含完整 exact top-k、排序和 nonfinite ID mask。\n\n"
            "GPU2（H200 / SM90）、CPU 16–23，每组预热 10 次，随后交替执行 30 组配对，每个样本仅 replay 一次。"
            "外部 CUDA event 包围这一次 graph replay；柱高为中位数，误差线覆盖全部样本的最小值至最大值。\n\n"
            f"验收 `{source['check_run_id']}` 在三层分别检查实际分数、ties、变化输入和 causal mask，"
            "每种输入的两种 graph 各 replay 4 次，top-k 分数和 ID 均逐位相等。"
            "计时与验收的完整 identity、official native、实际加载的 FlashInfer top-k 及 Triton mask 编译身份一致。"
            "原始 top-k 输出未保存；报告核对的是验收记录及绑定，不重新计算数值结果。\n\n"
            f"正式计时：`{source['bench_run_id']}`。数据见 [score layout 汇总](score_layout_latency.csv)和[全部配对样本](score_layout_samples.csv)。\n"
        )
    if components:
        text.append(
            "\n上述组件使用各自的独立计时边界和 GPU；promotion 的 event 节点位于 graph 内，score layout 的 event 位于 graph 外。"
            "这些时间不能相加推算完整模型延迟，也不能据此判断计算与 IO 是否重叠。完整模型的数值、流量、graph 预算和性能由对应模型运行单独验收。\n"
        )
    return "".join(text)


def generate(args):
    output = args.output_dir.resolve()
    common.require(
        output.is_relative_to(EXPERIMENT / "output/data"), "Use this experiment's output/data"
    )
    common.require(not output.exists(), "Use a new report run ID")
    if args.publish_dir:
        target = args.publish_dir.resolve()
        common.require(target.is_relative_to(EXPERIMENT / "report"), "Use this experiment's report")
        common.require(not target.exists(), "Use a new publication directory")
    evidence = common.Evidence(output)
    official_check, official = pair(evidence, "official", args.official_run)
    packing_check, packing = pair(evidence, "packing", args.packing_run)
    components = {}
    if args.promotion_run:
        components["promotion"] = promotion_data(evidence, args.promotion_run)
    if args.score_layout_run:
        components["score_layout"] = score_layout_data(evidence, args.score_layout_run)
    if getattr(args, "validation_run", None):
        common.require(args.validation_check is not None, "Validation requires its check receipt")
        components["validation"] = followup.validation_data(
            evidence, args.validation_run, args.validation_check
        )
    if getattr(args, "topk_run", None):
        components["topk"] = followup.topk_data(evidence, args.topk_run)
    for run, device, cpus in ((official, 1, list(range(8, 16))), (packing, 0, list(range(8)))):
        common.require(
            run["identity"]["physical_device"] == device
            and run["identity"]["cpu_affinity"] == cpus
            and run["identity"]["capability"] == [9, 0],
            "This report layout requires the declared GPU/CPU experiment conditions",
        )
    inputs = [
        {
            str((ROOT / name).resolve()): value
            for name, value in run["identity"]["inputs_sha256"].items()
        }
        for run in (official, packing)
    ]
    common.require(
        inputs[0] == inputs[1] and len(inputs[0]) == 3, "Expected the same three saved inputs"
    )
    if "topk" in components:
        common.require(
            {
                str((ROOT / name).resolve()): value
                for name, value in components["topk"]["inputs_sha256"].items()
            }
            == inputs[0],
            "CUB top-k inputs differ from shared L0-L2 inputs",
        )
    if "promotion" in components:
        common.require(
            components["promotion"]["bench"]["identity"]["input_sha256"] in inputs[0].values(),
            "Promotion input not in shared L0-L2 inputs",
        )
    if "score_layout" in components:
        score_inputs = {
            str((ROOT / name).resolve()): value
            for name, value in components["score_layout"]["bench"]["identity"][
                "inputs_sha256"
            ].items()
        }
        common.require(
            score_inputs == inputs[0], "Score-layout inputs differ from shared L0-L2 inputs"
        )
    rows, samples = [], []
    for name, run in (("official", official), ("packing", packing)):
        summary, detail = timing(run, name)
        rows.extend(summary)
        samples.extend(detail)
    sources = [
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(followup.__file__).resolve(),
        ROOT / "evaluation/validation.py",
    ]
    for source in sources:
        evidence.bind(source)
        target = output / "source_snapshot" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    common.write_csv(output / "latency.csv", rows)
    common.write_csv(output / "latency_samples.csv", samples)
    figures(output, rows)
    for name, data in components.items():
        common.write_csv(output / f"{name}_latency.csv", data["rows"])
        common.write_csv(output / f"{name}_samples.csv", data["samples"])
    component_figures(output, components)
    common.write_json(
        output / "summary.json",
        {
            "schema": "official-q1-path-component-report-v2",
            "run_id": output.name,
            "sources": evidence.identities,
            "official_native": official["native"],
            "packing_compiled_kernels": packing["compiled_kernels"],
            "additional_components": {
                name: {
                    "rows": data["rows"],
                    "native": data["native"] if "native" in data else data["bench"]["native"],
                    "validation": {
                        key: value
                        for key, value in data["check"].items()
                        if key not in ("identity", "native", "custom_cases")
                    },
                    "graph_stream_validation": data.get("graph_check"),
                }
                for name, data in components.items()
            },
            "validation": {
                "official_check_rows": len(official_check["rows"]),
                "packing_bytes": packing_check["bytes"],
                "packing_scores_and_selection": packing_check["scores_and_selection"],
            },
            "boundaries": {
                "official": official["identity"]["boundaries"],
                "packing": packing["identity"]["boundary"],
                "resident_map": "The raw bridge marks every historical mapping resident to suppress prefetch. It does not consume the device-pool KV payload and is not a full-model warm request.",
                "model": "Independent component timings; not a complete model request or an official SGLang versus local model comparison.",
                "profiling": "No GPU execution or profile collection occurs in this report generator.",
                "promotion": components["promotion"]["bench"]["timing"]
                if "promotion" in components
                else "Not included",
                "score_layout": components["score_layout"]["bench"]["identity"]["boundaries"]
                if "score_layout" in components
                else "Not included",
            },
        },
    )
    (output / "report.md").write_text(
        "# 官方 ECHO Q1 组件计时\n\n"
        "单 kernel 打包减少了本地 resident paged MQA 的准备开销；官方 ECHO 的 raw Q1 路径已在独立验收后完成计时。"
        "两项实验使用同一组 L0–L2 输入，Q=1、N=65,537；输入来自真实 checkpoint 的额外 eager 诊断 forward。\n\n"
        "![page64 打包及包含打包的 resident MQA](packing_latency.svg)\n\n"
        "图中完整 resident MQA 包含每次 K/scales 打包、页表、调度 metadata 和 causal mask，排除 exact top-k。"
        "GPU0、CPU 0–7，20 次预热，7 组样本，每组 100 次调用。柱高为 CUDA Graph 的 GPU event 时间中位数，误差线为全部样本的最小值至最大值。\n\n"
        "![官方 ECHO Q1 raw 路径](official_prefetch_latency.svg)\n\n"
        "官方路径直接编译固定版本 ECHO 的 paged fused decode kernel。Prepared inputs 预先准备打包、页表和 metadata；"
        "另一组将这些准备计入调用。两组均包含官方 fused kernel、causal clean 和实际最多 64 条记录的 stage 预取，"
        "排除 exact top-k、stage promotion 与 residual recall。GPU1、CPU 8–15，10 次预热，30 个单次调用样本；"
        "每次调用前在计时外恢复映射、stage 与计数。Cold map 的 threshold 为 0，每个样本预取 64 条记录，共 73,728 B。\n\n"
        "官方图使用对数纵轴；圆点与方点为中位数，误差线覆盖全部样本，包括离群值，未删除样本。\n\n"
        "Resident map 将全部 history 标为已驻留，用于测量 raw kernel 不预取时的开销。该 kernel 不消费 device pool 的 KV，"
        "这组条件不代表完整模型的 warm 请求。两张图也不能相加推算完整模型延迟，或与官方 SGLang 自然驻留的 decode 窗口计算等负载加速比。\n\n"
        f"官方 check / bench：`{official_check['run_id']}` / `{official['run_id']}`。"
        f"打包 check / bench：`{packing_check['run_id']}` / `{packing['run_id']}`。"
        f"本次 CPU 报告 run ID：`{output.name}`。\n\n"
        "[汇总](summary.json)保存环境、边界与验收身份；[计时表](latency.csv)和[全部样本](latency_samples.csv)保留 eager/graph、"
        "两种计时边界及各次实际 reservation、prefetch、H2D 计数。[来源记录](provenance.json)绑定原结果、源码快照、输入和官方 native 字节，"
        "[发布清单](publication_manifest.json)绑定本报告素材。独立验收中的逐位检查是运行时证据，报告生成不会重新执行 GPU 数值比较。\n"
        + component_report(
            {
                name: data
                for name, data in components.items()
                if name in ("promotion", "score_layout")
            },
            evidence.identities,
        )
        + followup.report(components, evidence.identities)
    )
    for filename, record in evidence.files.items():
        common.require(
            common.digest(ROOT / filename) == record["sha256"], "Evidence changed during generation"
        )
    common.write_json(
        output / "provenance.json",
        {
            "run_id": output.name,
            "raw_report_directory": str(output),
            "argv": sys.argv,
            "inputs_sha256": evidence.files,
            "source_snapshot_sha256": {
                path.name: common.digest(output / "source_snapshot" / path.name) for path in sources
            },
        },
    )
    selected = [path for path in output.iterdir() if path.is_file()]
    common.write_json(
        output / "publication_manifest.json",
        {
            "run_id": output.name,
            "files_sha256": {path.name: common.digest(path) for path in selected},
        },
    )
    if args.publish_dir:
        args.publish_dir.mkdir(parents=True)
        for path in [*selected, output / "publication_manifest.json"]:
            shutil.copyfile(path, args.publish_dir / path.name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--packing-run", type=Path, required=True)
    parser.add_argument("--promotion-run", type=Path)
    parser.add_argument("--score-layout-run", type=Path)
    parser.add_argument("--validation-run", type=Path)
    parser.add_argument("--validation-check", type=Path)
    parser.add_argument("--topk-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path)
    print(generate(parser.parse_args()))


if __name__ == "__main__":
    main()
