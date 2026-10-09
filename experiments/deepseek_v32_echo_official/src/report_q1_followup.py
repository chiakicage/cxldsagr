"""Verify follow-up promotion and exact-top-k evidence for the Q1 path report."""

import math
import statistics
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_echo_official.src import report_q1_optimization as common


def receipt(evidence, name, path, kind, identity):
    checked = require_receipt(path, kind=kind, identity=identity)
    evidence.read(name + "_receipt", Path(path))
    for artifact in checked["artifact_paths"].values():
        evidence.bind(Path(artifact))
    return checked


def snapshots(evidence, directories, identity):
    for filename, expected in identity["source_sha256"].items():
        relative = Path(filename)
        if relative.is_absolute():
            relative = relative.relative_to(common.ROOT)
        for directory in directories:
            common.require(
                evidence.bind(directory / "source" / relative) == expected,
                f"Follow-up source snapshot changed: {relative}",
            )


def summary(values, expected_count):
    common.require(
        len(values) == expected_count
        and all(math.isfinite(value) and value > 0 for value in values),
        "Incomplete or invalid follow-up timing samples",
    )
    return {
        "gpu_us_median": statistics.median(values),
        "gpu_us_min": min(values),
        "gpu_us_max": max(values),
    }


def validation_data(evidence, directory, check_path):
    directory = directory.resolve(strict=True)
    bench = evidence.read("validation_bench", directory / "result.json")
    common.require(bench["passed"] and bench["mode"] == "bench", "Unaccepted validation bench")
    common.require(bench["run_id"] == directory.name, "Validation run ID differs")
    checked = receipt(
        evidence,
        "validation",
        check_path,
        "deepseek-official-q1-validation-candidate-v1",
        bench["identity"],
    )
    common.require(
        evidence.bind(
            common.EXPERIMENT / "output/data/q1_inputs_20261008_01/kernel_inputs_layer_2.pt"
        )
        == bench["identity"]["input_sha256"],
        "Validation record input changed",
    )
    snapshots(evidence, (directory, check_path.parent), bench["identity"])
    native = evidence.read("validation_native", directory / "native.json")
    common.require(set(native) == {"baseline", "candidate"}, "Missing validation native arm")
    checked_native = evidence.read(
        "validation_checked_native", Path(checked["artifact_paths"]["native.json"])
    )
    common.require(native == checked_native, "Validation native differs from check")
    for arm, record in native.items():
        common.require(
            evidence.bind(Path(record["artifact_path"])) == record["artifact_sha256"],
            "Validation native bytes changed",
        )
        common.require(
            bench["identity"]["native"][arm]
            == {
                "path": record["artifact_path"],
                "sha256": record["artifact_sha256"],
                "build_identity_sha256": identity_digest(record["build_identity"]),
            },
            "Validation build identity changed",
        )
    checks = checked["checks"]
    common.require(
        checks["custom_case_count"] == len(checks["custom_cases"]) == 84
        and checks["changed_graph_replays"] == 72
        and checks["nondefault_stream_calls"] == 24
        and checks["adapter_tests"]["passed"] == 22
        and checks["adapter_tests"]["skipped"] == 0,
        "Validation check coverage differs",
    )
    invalid = checks["invalid_inputs"] + checks["additional_invalid_inputs"]
    common.require(
        len(invalid) == 8 and all(row["device_assert"] and row["returncode"] for row in invalid),
        "Validation negative checks incomplete",
    )
    common.require(
        bench["cuda_visible_devices"] == "3"
        and bench["identity"]["device_capability"] == [9, 0]
        and bench["cpu_affinity"] == list(range(24, 32))
        and bench["warmups"] == 20
        and bench["repeats"] == 100,
        "Validation measurement conditions differ",
    )
    rows, samples, keys = [], [], set()
    for case in bench["cases"]:
        key = (case["layout"], case["occupied_selected_slots"])
        common.require(key not in keys, "Duplicate validation condition")
        keys.add(key)
        common.require(
            (
                case["N"],
                case["P"],
                case["stage_records"],
                case["record_width"],
                case["record_dtype"],
            )
            == (65537, 65600, 64, 576, "bfloat16"),
            "Validation shape differs",
        )
        common.require(case["offset_bytes"] == 0, "Validation alignment differs")
        values = case["samples_us"]
        common.require(set(values) == {"baseline", "candidate"}, "Missing validation arm")
        wins = sum(c < b for c, b in zip(values["candidate"], values["baseline"], strict=True))
        common.require(wins == case["candidate_paired_wins"], "Validation wins differ")
        for arm, timings in values.items():
            stats = summary(timings, 100)
            common.require(stats["gpu_us_median"] == case["median_us"][arm], "Median differs")
            base = {
                "operator": "validation",
                "run_id": bench["run_id"],
                "layout": key[0],
                "occupied_selected_slots": key[1],
                "method": arm,
                "samples": 100,
                "calls_per_sample": 1,
                "warmups": 20,
            }
            rows.append({**base, **stats, "candidate_paired_wins": wins})
            samples.extend({**base, "pair": i, "gpu_us": t} for i, t in enumerate(timings))
    common.require(
        keys
        == {(layout, occupied) for layout in ("consecutive", "random") for occupied in (0, 32, 64)},
        "Incomplete validation matrix",
    )
    evidence.identities["validation"] = {
        "check_run_id": check_path.parent.name,
        "bench_run_id": bench["run_id"],
        "original_identity": bench["identity"],
    }
    return {"bench": bench, "check": checks, "native": native, "rows": rows, "samples": samples}


def topk_data(evidence, directory):
    directory = directory.resolve(strict=True)
    bench = evidence.read("topk_bench", directory / "result.json")
    common.require(bench["accepted"] and bench["mode"] == "bench", "Unaccepted top-k bench")
    common.require(bench["run_id"] == directory.name, "Top-k run ID differs")
    check_path = Path(bench["receipt"]["path"])
    common.require(evidence.bind(check_path) == bench["receipt"]["sha256"], "Top-k receipt changed")
    checked = receipt(
        evidence, "topk", check_path, "deepseek-q1-topk-fused-sort-mask-v2", bench["identity"]
    )
    check = evidence.read("topk_check", Path(checked["artifact_paths"]["result"]))
    common.require(check["accepted"] and check["native"] == bench["native"], "Top-k native differs")
    common.require(
        bench["identity"]["candidate"] == "cub"
        and bench["native"]["sort_mask"]["build_identity"]["source_identity"]
        == bench["identity"]["candidate_build"],
        "Top-k candidate or CUB build differs",
    )
    common.require(
        checked["checks"]["comparisons"] == 115
        and checked["checks"]["changed_graph_replays"] == 20,
        "Top-k check coverage differs",
    )
    snapshots(evidence, (directory, check_path.parent), bench["identity"])
    # The receipt binds the actually loaded DSOs, copied before the mutable
    # FlashInfer build directory could change. Never substitute its current DSO.
    for name, expected in (
        ("native/topk.so", bench["native"]["topk"]["library"]["sha256"]),
        ("native/cub_sort_mask.so", bench["native"]["sort_mask"]["artifact_sha256"]),
    ):
        common.require(
            evidence.bind(Path(checked["artifact_paths"][name])) == expected, "Top-k DSO differs"
        )
    for filename, expected in bench["identity"]["candidate_build"]["source_sha256"].items():
        path = Path(filename)
        # Mutable local source is covered by both archived snapshots above.
        if not path.is_relative_to(common.ROOT):
            common.require(evidence.bind(path) == expected, "Installed CUB header changed")
    common.require(
        bench["identity"]["physical_device"] == 1
        and bench["identity"]["cpu_affinity"] == list(range(8, 16)),
        "Top-k device or affinity differs",
    )
    rows, samples, layers = [], [], set()
    for case in bench["payload"]:
        common.require(
            len(case["samples"]) == 200 and {row["replays"] for row in case["samples"]} == {1, 20},
            "Unexpected top-k sample count or replay multiplicity",
        )
        layer = int(case["case"].removeprefix("layer_"))
        common.require(layer not in layers, "Duplicate top-k layer")
        layers.add(layer)
        for replays in (1, 20):
            selected = [row for row in case["samples"] if row["replays"] == replays]
            pairs = {}
            for row in selected:
                common.require(
                    row["order"] == ("AB" if row["pair"] % 2 == 0 else "BA"), "Pair order differs"
                )
                pair = pairs.setdefault(row["pair"], {})
                common.require(row["arm"] not in pair, "Duplicate top-k sample")
                pair[row["arm"]] = row["gpu_us_per_api"]
            common.require(
                set(pairs) == set(range(50))
                and all(set(p) == {"baseline", "candidate"} for p in pairs.values()),
                "Incomplete top-k pairs",
            )
            wins = sum(p["candidate"] < p["baseline"] for p in pairs.values())
            reported = case["summary"][str(replays)]
            common.require(wins == reported["wins"], "Top-k wins differ")
            deltas = [pairs[i]["candidate"] - pairs[i]["baseline"] for i in range(50)]
            common.require(
                deltas == reported["paired_delta_us"]
                and statistics.median(deltas) == reported["paired_median_delta_us"]
                and {
                    order: statistics.median(deltas[parity::2])
                    for parity, order in enumerate(("AB", "BA"))
                }
                == reported["order_median_delta_us"],
                "Top-k paired statistics differ",
            )
            for arm in ("baseline", "candidate"):
                timings = [pairs[i][arm] for i in range(50)]
                stats = summary(timings, 50)
                common.require(
                    stats["gpu_us_median"] == reported["median_us"][arm], "Top-k median differs"
                )
                base = {
                    "operator": "topk",
                    "run_id": bench["run_id"],
                    "layer": layer,
                    "method": arm,
                    "samples": 50,
                    "calls_per_sample": replays,
                    "warmups": 10,
                }
                rows.append({**base, **stats, "candidate_paired_wins": wins})
                samples.extend({**base, "pair": i, "gpu_us": t} for i, t in enumerate(timings))
    common.require(layers == {0, 1, 2}, "Top-k requires L0-L2")
    evidence.identities["topk"] = {
        "check_run_id": check_path.parent.name,
        "bench_run_id": bench["run_id"],
        "original_identity": bench["identity"],
    }
    return {
        "bench": bench,
        "check": checked["checks"],
        "inputs_sha256": check["payload"]["preparation"]["inputs_sha256"],
        "native": bench["native"],
        "rows": rows,
        "samples": samples,
    }


def report(components, identities):
    text = []
    for name in ("validation", "topk"):
        if name not in components:
            continue
        rows = [row for row in components[name]["rows"] if row["calls_per_sample"] == 1]
        medians = {
            arm: [r["gpu_us_median"] for r in rows if r["method"] == arm]
            for arm in ("baseline", "candidate")
        }
        wins = sum(r["candidate_paired_wins"] for r in rows if r["method"] == "candidate")
        total = sum(r["samples"] for r in rows if r["method"] == "candidate")
        text.append(
            f"\n## {'Promotion 并行校验' if name == 'validation' else 'CUB exact top-k 后处理'}\n\n"
        )
        text.append(
            f"完整 API 中位数从 {min(medians['baseline']):.3f}–{max(medians['baseline']):.3f} µs 降至 {min(medians['candidate']):.3f}–{max(medians['candidate']):.3f} µs，{wins}/{total} 组配对更快。\n\n"
        )
        text.append(f"![完整 API 的配对计时]({name}_latency.svg)\n\n")
        if name == "validation":
            text.append(
                "校验 CTA 从 64 线程改为 256 线程，并行检查重复记录；保留范围、owner、journal、映射和唯一性检查，官方 ECHO 融合内核及后续复制 kernel 不变。两组都使用既有并行 record copy。输入为 N=65,537、P=65,600、64 条 BF16 record，每条 576 个元素；六组条件覆盖连续/随机槽及 0/32/64 个已占槽。\n\nGPU3 / SM90、CPU 24–31，每组预热 20 次，交替测量 100 对。图内 event 包围完整 promotion 的两个 kernel，metadata 恢复和主机 dispatch 在计时外。84 个逐位案例、22 项 adapter 测试、72 次变化输入 replay、24 次非默认流调用及 8 个非法状态均通过验收。\n\n"
            )
        else:
            text.append(
                "保留 FlashInfer SMALL 的精确选择核心，使用官方 CUB BlockRadixSort 合并排序和 nonfinite ID mask，输出值与 ID 逐位不变。输入为真实 L0–L2 的 Q1 分数及已有 padding，k=2,048。图中每个样本只 replay 一次；表格另保存 20 次 replay 的确认测量。\n\nGPU1 / SM90、CPU 8–15，每组预热 10 次，再测量 50 组 AB/BA 配对，外部 CUDA event 包围完整 graph API。115 项 oracle/逐位比较、20 次变化输入 replay 通过；验收保存的 DSOs 与本次计时 native 身份一致。\n\n"
            )
        source = identities[name]
        text.append(
            f"独立验收 `{source['check_run_id']}`；计时 `{source['bench_run_id']}`。柱高为中位数，误差线覆盖所有样本的最小值至最大值，不删除离群值。数据见[汇总]({name}_latency.csv)和[全部配对样本]({name}_samples.csv)。这些组件时间不代表完整模型或官方 SGLang 的请求延迟。\n"
        )
    return "".join(text)
