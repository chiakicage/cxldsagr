"""Report separately accepted, labeled production exact-recall benchmark runs."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from evaluation.provenance import snapshot_report_helpers
from evaluation.validation import identity_digest, require_receipt
from experiments.cache_manager_performance.src.report import read_observer
from experiments.deepseek_v32_mfu.src.kernel_profile import file_sha256, load_inputs

ROOT = Path(__file__).resolve().parents[3]
KIND = "deepseek-state-exact-recall-v1"
STATES = ("certified_resident", "restored_all_hit", "cold_sparse_miss")
CONFIG = {"append": 128, "chunk": 1024, "history": 65536, "layers": 3}
BOUNDARY = (
    "prefix/reset/begin/append/drain/stats reset/commit/oracles excluded; private recall and "
    "device fence included; wall minus enqueue is not pure GPU compute; no IO subtraction "
    "or model claim"
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def summarize(result, label, expected_counts, *, repeats=31):
    """Reject incomplete runs and counter drift before calculating any statistic."""
    expected = {(state, layer) for state in STATES for layer in range(3)}
    groups = {key: [] for key in expected}
    for sample in result["samples"]:
        key = (sample["state"], sample["layer"])
        if key not in groups or sample["variant"] != "baseline":
            raise ValueError("report requires one production variant and all nine cases")
        wall, enqueue = sample["wall_ms"], sample["enqueue_ms"]
        if not all(math.isfinite(value) for value in (wall, enqueue)) or not 0 < enqueue <= wall:
            raise ValueError("invalid enqueue/wall timing")
        selected, cold_misses = expected_counts[sample["layer"]]
        misses = cold_misses if sample["state"] == "cold_sparse_miss" else 0
        counters = {
            "selection_records": selected,
            "resident_selection_records": selected - misses,
            "max_working_set": selected,
            "recalled_records": misses,
            "host_to_device_bytes": misses * 1152,
            "device_to_host_bytes": 0,
            "written_records": 0,
            "host_written_records": 0,
            "transient_written_records": 0,
            "evicted_records": 0,
            "prefetched_records": 0,
            "prefetch_capacity_failures": 0,
            "capacity_splits": 0,
            "record_bytes": 1152,
            "device_slots": 65664,
            "host_token_capacity": 65664,
            "session_host_tokens": 65664,
            "candidate_slots": 0,
            "padding_slots": 1,
        }
        if any(sample["metrics"].get(name) != value for name, value in counters.items()):
            raise ValueError("recall counters differ from the captured selection and reset state")
        groups[key].append(sample)
    rows = []
    for state in STATES:
        for layer in range(3):
            samples = groups[state, layer]
            if sorted(sample["sample"] for sample in samples) != list(range(repeats)):
                raise ValueError("missing, duplicate or unexpected sample index")
            row = {"label": label, "state": state, "layer": layer, "samples": repeats}
            for metric in ("wall", "enqueue"):
                values = [sample[f"{metric}_ms"] for sample in samples]
                q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
                row.update(
                    {
                        f"{metric}_median_ms": statistics.median(values),
                        f"{metric}_q1_ms": q1,
                        f"{metric}_q3_ms": q3,
                        f"{metric}_iqr_ms": q3 - q1,
                    }
                )
            row.update(
                recalled_records=samples[0]["metrics"]["recalled_records"],
                host_to_device_bytes=samples[0]["metrics"]["host_to_device_bytes"],
            )
            rows.append(row)
    return rows


def _matched_warmup(result, check_result):
    warmup = result.get("warmup")
    check_warmup = check_result.get("warmup")
    if (
        type(warmup) is not int
        or type(check_warmup) is not int
        or warmup < 1
        or warmup != check_warmup
    ):
        raise ValueError("check and benchmark require matching positive integer warmups")
    return warmup


def load_run(label, benchmark, check, observer):
    benchmark, check, observer = (
        Path(path).resolve(strict=True) for path in (benchmark, check, observer)
    )
    result, identity = (read(benchmark / name) for name in ("result.json", "identity.json"))
    accepted = require_receipt(check / "receipt.json", kind=KIND, identity=identity)
    check_result = read(check / "result.json")
    warmup = _matched_warmup(result, check_result)
    if (
        result.get("mode") != "bench"
        or check_result.get("mode") != "check"
        or result.get("passed") is not True
        or check_result.get("passed") is not True
        or result.get("identity_sha256") != identity_digest(identity)
        or check_result.get("identity_sha256") != result["identity_sha256"]
        or result.get("validation_receipt") != accepted["receipt_sha256"]
        or read(check / "identity.json") != identity
        or Path(accepted["artifact_paths"]["result"]) != check / "result.json"
        or result.get("checks") != []
        or result.get("boundary") != BOUNDARY
        or check_result.get("boundary") != BOUNDARY
        or identity.get("config") != CONFIG
        or identity.get("states") != list(STATES)
    ):
        raise ValueError("benchmark, accepted check or scope binding differs")
    cases = accepted["checks"].get("cases", [])
    expected_cases = {
        f"{state}_layer{layer}_baseline_{sample}"
        for state in STATES
        for layer in range(3)
        for sample in range(2)
    }
    if (
        len(cases) != 18
        or {case["case"] for case in cases} != expected_cases
        or not all(case.get("passed") is True for case in cases)
        or cases != check_result.get("checks")
    ):
        raise ValueError("acceptance must cover eighteen successful independent checks")
    for directory in (benchmark, check):
        for name, expected in identity["sources"].items():
            if file_sha256(directory / "source" / name) != expected:
                raise ValueError("execution source archive differs from acceptance")
    inputs = {}
    if sorted(entry["layer"] for entry in identity["captures"]) != [0, 1, 2]:
        raise ValueError("capture inventory must contain each layer exactly once")
    for entry in identity["captures"]:
        path = Path(entry["path"])
        if (
            file_sha256(path) != entry["sha256"]
            or file_sha256(path.parent / "result.json") != entry["source_result_sha256"]
        ):
            raise ValueError("captured inputs or source acceptance metadata changed")
        data = load_inputs(path)
        ids = data["indices"]
        if (
            tuple(ids.shape) != (128, 2048)
            or tuple(data["kv"].shape) != (65664, 576)
            or data["query_start"] != 65536
            or data["layer"] != entry["layer"]
            or data["source_run_id"] != entry["source_run_id"]
        ):
            raise ValueError("capture geometry differs from the declared recall workload")
        unique = ids.unique()
        unique = unique[unique >= 0]
        inputs[entry["layer"]] = (len(unique), int((unique < 65536).sum()))
    summarize(check_result, label, inputs, repeats=2)
    rows = summarize(result, label, inputs)
    observation = read_observer(observer, result["run_id"], identity["runtime"]["gpu_uuid"])
    command = observation["run"]["command"]
    if (
        "--validation-receipt" not in command
        or (ROOT / command[command.index("--validation-receipt") + 1]).resolve()
        != check / "receipt.json"
    ):
        raise ValueError("observer command used a different acceptance receipt")
    binding = {
        "label": label,
        "warmup": warmup,
        "benchmark_directory": str(benchmark),
        "check_directory": str(check),
        "run_id": result["run_id"],
        "check_run_id": check_result["run_id"],
        "identity_sha256": result["identity_sha256"],
        "receipt_sha256": accepted["receipt_sha256"],
        "identity": identity,
        "observer": observation,
        "files": {
            str(path): file_sha256(path)
            for path in (
                benchmark / "result.json",
                benchmark / "identity.json",
                check / "result.json",
                check / "identity.json",
                check / "receipt.json",
            )
        },
    }
    return binding, rows


def plot(rows, labels, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#555555")
    markers = ("o", "s", "D", "^", "v", "X")
    titles = ("Certified resident", "Restored all-hit", "Cold sparse miss")
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharey="row")
        for column, (state, title) in enumerate(zip(STATES, titles, strict=True)):
            for row_index, metric in enumerate(("wall", "enqueue")):
                axis = axes[row_index, column]
                for index, label in enumerate(labels):
                    selected = [
                        next(
                            r
                            for r in rows
                            if (r["label"], r["state"], r["layer"]) == (label, state, layer)
                        )
                        for layer in range(3)
                    ]
                    offset = (index - (len(labels) - 1) / 2) * 0.6 / len(labels)
                    median = [r[f"{metric}_median_ms"] for r in selected]
                    error = [
                        [r[f"{metric}_median_ms"] - r[f"{metric}_q1_ms"] for r in selected],
                        [r[f"{metric}_q3_ms"] - r[f"{metric}_median_ms"] for r in selected],
                    ]
                    axis.errorbar(
                        [layer + offset for layer in range(3)],
                        median,
                        yerr=error,
                        fmt=markers[index],
                        color=colors[index],
                        capsize=4,
                        markersize=5,
                        label=label,
                    )
                axis.set_xticks(range(3), ["0", "1", "2"])
                axis.set_xlabel("Layer")
                axis.spines[["top", "right"]].set_visible(False)
                axis.grid(axis="y", color="0.9", linewidth=0.6)
                if row_index == 0:
                    axis.set_title(title)
                if column == 0:
                    axis.set_ylabel(
                        "Synchronized wall (ms)" if metric == "wall" else "Enqueue (ms)"
                    )
        for row_index, metric in enumerate(("wall", "enqueue")):
            upper = max(row[f"{metric}_q3_ms"] for row in rows)
            axes[row_index, 0].set_ylim(0, upper * 1.12)
        handles, names = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, names, loc="upper center", ncol=min(len(labels), 3), frameon=False)
        figure.text(
            0.5,
            0.015,
            "Median and interquartile range; 31 independent resets per layer/state. Each layer has its own fence.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.94))
        figure.savefig(output / "latency.svg")
        figure.savefig(output / "latency.png", dpi=200)
        plt.close(figure)
    return matplotlib.__version__


def generate(runs, output, source_archive):
    labels = [run[0] for run in runs]
    if not labels or len(set(labels)) != len(labels) or len(labels) > 6:
        raise ValueError("provide one to six unique explicit run labels")
    if any(not label.strip() or any(c in label for c in "\n\r|`") for label in labels):
        raise ValueError("labels must be nonempty single-line table-safe text")
    bindings, rows = [], []
    comparison = None
    for run in runs:
        binding, selected = load_run(*run)
        identity = binding["identity"]
        comparable = {
            "config": identity["config"],
            "captures": identity["captures"],
            "runtime": {
                key: value
                for key, value in identity["runtime"].items()
                if key
                not in {
                    "echo_build",
                    "recall_dispatch_build",
                    "native_library_sha256",
                    "local_native_jit",
                }
            },
        }
        if comparison is not None and comparable != comparison:
            raise ValueError("labeled runs differ in inputs, geometry or runtime environment")
        comparison = comparable
        bindings.append(binding)
        rows.extend(selected)
    output, source_archive = Path(output).resolve(), Path(source_archive).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive.mkdir(parents=True, exist_ok=False)
    helpers = snapshot_report_helpers(source_archive)
    for value in helpers["files"].values():
        value["snapshot_path"] = str(source_archive / value["snapshot_path"])
    write(output / "report_helper_sources.json", helpers)
    with (output / "latency.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matplotlib_version = plot(rows, labels, output)
    table = [
        "| Run label | State | Layer | Wall median [Q1, Q3] ms | Enqueue median [Q1, Q3] ms |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    table.extend(
        f"| {r['label']} | {r['state']} | {r['layer']} | {r['wall_median_ms']:.6f} [{r['wall_q1_ms']:.6f}, {r['wall_q3_ms']:.6f}] | {r['enqueue_median_ms']:.6f} [{r['enqueue_q1_ms']:.6f}, {r['enqueue_q3_ms']:.6f}] |"
        for r in rows
    )
    run_text = "\n".join(
        f"- {b['label']}: bench `{b['run_id']}`, check `{b['check_run_id']}`, identity `{b['identity_sha256']}`; {b['warmup']} warmups per case."
        for b in bindings
    )
    runtime = bindings[0]["identity"]["runtime"]
    text = (
        "# Independent exact recall\n\n"
        "Each labeled run completed 31 independent reset samples for each of three states and three layers, after the run-specific warmup count listed below. "
        "The matched independent acceptance covers 18 checks. Each input is captured int32 `[128,2048]` exact top-k IDs; "
        "H=65,536, A=128, chunk=1,024, P=NH=65,664, BF16 records of width 576 (1,152 B).\n\n"
        f"{run_text}\n\n"
        f"Hardware: {runtime['gpu_name']}, GPU UUID `{runtime['gpu_uuid']}`; CPU affinity {runtime['cpu_affinity']}. "
        f"PyTorch {runtime['torch']}, CUDA {runtime['cuda']}, TVM FFI {runtime['tvm_ffi']}.\n\n"
        "![Per-layer exact recall medians and interquartile ranges](latency.svg)\n\n"
        "Points are medians; error bars span Q1–Q3, calculated by linear interpolation over all 31 retained samples "
        '(`statistics.quantiles(method="inclusive")`). IQR is Q3−Q1, not a confidence interval. '
        "Separate labeled runs are not paired samples. Full values and recall counters are in [latency.csv](latency.csv).\n\n"
        + "\n".join(table)
        + "\n\nCertified resident rebuilds history by production append and retains its CPU proof. Restored all-hit restores a production snapshot; "
        "GPU mappings remain resident but the CPU proof is uncertified. Cold sparse miss additionally releases history GPU IDs. "
        "Every sample resets independently.\n\n"
        "Timing includes production `_ensure_from_topk` and device synchronization. Prefix construction, reset/snapshot, begin/append, "
        "host drain, counter reset/read, commit and numerical checks are outside the timer. The report does not add separately fenced layers "
        "into a grouped request latency. Wall minus enqueue is not pure GPU compute, and no IO time is subtracted. "
        "Transfer bytes come from successful-record counters, not a hardware transfer profile.\n\n"
        "The entry consumes captured selections without executing or changing indexer arithmetic/top-k. It covers one session per layer with "
        "P=H+A: allocation ties concern free slots only, with no live eviction. It is not a full cache/indexer or model measurement and has no model gate.\n\n"
        "[Provenance](provenance.json) binds each check, benchmark, input, measured source archive, native identity and observer. "
        "[Report helper hashes](report_helper_sources.json) bind separately archived reporting sources. Observer samples are discrete and do not "
        "establish continuous GPU isolation or exclude unrelated CPU work.\n"
    )
    (output / "results.md").write_text(text)
    write(
        output / "provenance.json",
        {
            "schema": "cache-recall-report-v1",
            "runs": bindings,
            "matplotlib": matplotlib_version,
            "boundary": BOUNDARY,
            "statistics": "median; inclusive linearly interpolated quartiles; IQR=Q3-Q1",
            "report_files": {
                path.name: file_sha256(path) for path in output.iterdir() if path.is_file()
            },
        },
    )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        required=True,
        metavar=("LABEL", "BENCH_DIR", "CHECK_DIR", "OBSERVER_DIR"),
    )
    parser.add_argument("--output", type=Path, required=True, help="new report bundle directory")
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="new separate reporting-source archive directory",
    )
    args = parser.parse_args(argv)
    generate(args.run, args.output, args.source_archive)
    print(args.output)


if __name__ == "__main__":
    main()
