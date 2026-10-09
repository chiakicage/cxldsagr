"""Verify native graph-node ownership and publish the official H64K decode comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import statistics
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import decode_display
from experiments.deepseek_v32_echo_official.src import engine_timeline as timeline


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compatibility(identity):
    core = json.loads(json.dumps(identity))
    required = (
        "src/decode_run.py",
        "src/engine_run.py",
        "src/preflight.py",
        "src/capacity.py",
        "src/workload_client.py",
    )
    core["sources"] = {
        name: digest for name, digest in core["sources"].items() if name.endswith(required)
    }
    return core


def read_run(path, mode):
    record = json.loads((path / "run.json").read_text())
    require(record["status"] == "completed" and record["mode"] == mode, "Incomplete run")
    require(record["numerical_acceptance"] is False, "Unsupported numerical claim")
    expected_workers = record["arguments"]["repeats"] if mode == "bench" else 1
    require(
        expected_workers > 0
        and [worker["name"] for worker in record["workers"]]
        == [f"sample_{index:02d}" for index in range(expected_workers)],
        "Worker samples are missing, duplicated, or out of order",
    )
    for name, digest in record["sources"].items():
        require(
            timeline.sha256(path / "sources" / name) == digest, f"Changed source archive: {name}"
        )
    for worker in record["workers"]:
        directory = path / worker["name"]
        require(
            json.loads((directory / "engine/complete.json").read_text()) == worker["completion"],
            "Worker completion changed",
        )
        require(
            timeline.sha256(directory / "gpu_release_drain.json") == worker["gpu_release_sha256"],
            "GPU release evidence changed",
        )
        require(worker["capacity"]["passed"] is True, "Capacity did not pass")
        verify_capacity_binding(directory, worker["capacity"])
        require(
            timeline.sha256(directory / "engine/warmup.json")
            == worker["completion"]["warmup_sha256"],
            "Warmup evidence changed",
        )
        require(
            timeline.sha256(worker["jit"]["path"]) == worker["jit"]["sha256"],
            "JIT manifest changed",
        )
        if mode == "check":
            artifact = worker["completion"]["row"]["output_artifact"]
            require(
                artifact["finite"] and timeline.sha256(artifact["path"]) == artifact["sha256"],
                "Check output artifact changed",
            )
        else:
            require(
                timeline.sha256(directory / "engine/response.json")
                == worker["completion"]["row"]["response_sha256"],
                "Response changed",
            )
    return record


def verify_capacity_binding(directory, capacity):
    require(
        json.loads((directory / "capacity.json").read_text()) == capacity,
        "Capacity artifact changed",
    )
    server = json.loads((directory / "engine/server_info.json").read_text())
    require(
        hashlib.sha256(
            json.dumps(server, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == capacity["server_info_sha256"],
        "Capacity server evidence changed",
    )
    for name, digest in capacity["log_sha256"].items():
        # The reused capacity helper hashes read_text(), including universal newline conversion.
        require(
            hashlib.sha256(Path(name).read_text().encode()).hexdigest() == digest,
            "Capacity log text changed",
        )
    for rows in capacity["evidence"].values():
        for row in rows:
            lines = Path(row["source"]).read_text().splitlines()
            require(lines[row["line_number"] - 1] == row["line"], "Capacity line changed")


def classify_decode(row):
    timeline.classify(row)
    name = row["name"].lower()
    if row["kind"] == "kernel" and row.get("stage") == "indexer_prefetch" and "mqa" in name:
        row.update(
            lane="Compute",
            category="Indexer / top-k",
            potential_io=True,
            io_evidence="Official paged fused indexer may read mapped host memory; actual bytes unknown",
        )
    if row["kind"] == "kernel" and "recall" in name:
        row.update(
            lane="GPU control",
            category="ECHO recall (IO unknown)",
            potential_io=True,
            io_evidence="Official decode recall has dynamic guards; actual bytes unknown",
        )


def select(path, hooks, case):
    capture = timeline.attribute(timeline.read_capture(path))
    forwards = json.loads((hooks / "forwards.json").read_text())
    require(
        len(forwards) == 65 and [row["phase"] for row in forwards] == ["prefill"] * 64 + ["decode"],
        "Expected a full H64K prefill and exactly one real decode",
    )
    require(
        all(
            row["start"] == i * 1024 and row["end"] == (i + 1) * 1024
            for i, row in enumerate(forwards[:64])
        ),
        "Prefill coverage is incomplete",
    )
    decode = forwards[-1]
    require(
        decode["q"] == 1
        and decode["start"] == 65536
        and decode["end"] == 65537
        and decode["mode"] == "DECODE"
        and decode["cuda_graph"],
        "Not the expected normal graph decode",
    )
    formal = sorted(
        (scope for scope in capture["scopes"] if scope["fields"]["kind"] == "forward"),
        key=lambda scope: scope["fields"]["id"],
    )
    require(
        len(formal) == 65 and [scope["fields"]["id"] for scope in formal] == list(range(65)),
        "Formal NVTX does not contain exactly 65 unique forward ranges",
    )
    for index, (scope, metadata) in enumerate(zip(formal, forwards, strict=True)):
        require(
            all(scope["fields"][key] == metadata[key] for key in ("phase", "q", "start", "end"))
            and scope["fields"]["case"] == case,
            "Formal NVTX geometry differs from the worker ledger",
        )
        if index < 64:
            layers = [
                row["fields"]["layer"]
                for row in capture["scopes"]
                if row["fields"]["kind"] == "layer" and row["fields"].get("id") == index
            ]
            require(sorted(layers) == [0, 1, 2], "Formal prefill is missing a layer range")
    scopes = [
        scope
        for scope in capture["scopes"]
        if scope["fields"]["kind"] == "forward" and scope["fields"].get("phase") == "decode"
    ]
    require(len(scopes) == 1 and scopes[0]["fields"]["id"] == decode["id"], "Decode NVTX mismatch")
    process = scopes[0]["thread"] & timeline.PROCESS_MASK
    hook_manifest = json.loads((hooks / "hook_manifest.json").read_text())
    require(
        hook_manifest["pid"] == (process >> 24) & 0xFFFFFF, "Graph ledger and decode process differ"
    )
    require(
        hook_manifest["case"] == case and scopes[0]["fields"]["case"] == case,
        "Hook or NVTX case differs from its comparison label",
    )
    launches = [
        api
        for api in capture["apis"]
        if "GraphLaunch" in api["name"] and api["context"].get("id") == decode["id"]
    ]
    require(len(launches) == 1, "Expected exactly one graph launch in the real decode")
    gpu = [row for row in capture["activities"] if row["forward_id"] == decode["id"]]
    require(gpu, "No decode GPU activity")
    require(
        all(row["process"] == process for row in gpu),
        "Decode GPU activity belongs to another process",
    )
    graph_rows = [row for row in gpu if row["graph_node_id"]]
    graph_ids = {row["graph_id"] for row in graph_rows if row["graph_id"]}
    require(len(graph_ids) == 1, "Expected one executable graph")
    graphs = json.loads((hooks / "decode_graphs.json").read_text())["graphs"]
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        lineage = {}
        diagnostics = []
        if db.execute("SELECT name FROM sqlite_master WHERE name='DIAGNOSTIC_EVENT'").fetchone():
            diagnostics = [
                text
                for (text,) in db.execute(
                    "SELECT text FROM DIAGNOSTIC_EVENT WHERE globalPid=? AND severity>=2",
                    (process,),
                )
            ]
        for node, parent, thread in db.execute(
            "SELECT graphNodeId,originalGraphNodeId,globalTid FROM CUDA_GRAPH_NODE_EVENTS"
        ):
            if (
                parent is not None
                and thread is not None
                and thread & timeline.PROCESS_MASK == process
            ):
                require(node not in lineage or lineage[node] == parent, "Ambiguous node lineage")
                lineage[node] = parent
    seen = []
    for row in graph_rows:
        node, visited = row["graph_node_id"], set()
        while node in lineage:
            require(node not in visited, "Cyclic graph-node lineage")
            visited.add(node)
            node = lineage[node]
        seen.append(str(node))
        row["graph_capture_node_id"] = node
    require(len(seen) == len(set(seen)), "Repeated graph GPU node in one decode")
    matches = [graph for graph in graphs if graph["owners"].keys() == set(seen)]
    require(len(matches) == 1, "Replay does not exactly match one captured native GPU node set")
    graph = matches[0]
    for row in graph_rows:
        node = row["graph_capture_node_id"]
        owner = graph["owners"].get(str(node))
        require(owner is not None, f"Unknown graph GPU node: {node}")
        row.update(owner, graph_capture_node_id=node, graph_ownership="capture-time native node ID")
        classify_decode(row)
    require(
        len(seen) == len(set(seen)) and set(seen) == graph["owners"].keys(),
        "Missing or duplicated replay graph GPU nodes",
    )
    boundaries = []
    for layer in range(3):
        rows = [row for row in gpu if row["layer"] == layer and row["lane"] == "Compute"]
        require(rows, f"Missing L{layer} compute")
        boundaries.append(
            {
                "layer": layer,
                "start_ns": min(row["start_ns"] for row in rows),
                "end_ns": max(row["end_ns"] for row in rows),
            }
        )
    start, end = boundaries[0]["start_ns"], boundaries[2]["end_ns"]
    devices = {row["device"] for row in gpu}
    require(len(devices) == 1, "Expected one GPU")
    all_gpu = [row for row in capture["activities"] if row["device"] in devices]
    metrics = timeline.summarize_window(all_gpu, start, end)
    selected = [row for row in all_gpu if row["start_ns"] < end and row["end_ns"] > start]
    require(
        all(row["forward_id"] == decode["id"] for row in selected),
        "Foreign activity in decode window",
    )
    require(metrics["unattributed_activity_count"] == 0, "Unattributed decode activity")
    metrics.update(
        case=case,
        phase="decode",
        query_tokens=1,
        history_tokens=65536,
        graph_launches=1,
        graph_gpu_nodes=len(seen),
        observed_executable_graph_id=next(iter(graph_ids)),
        capture_graph_id=graph["capture_graph_id"],
        every_graph_node_verified=True,
        layer_boundaries=boundaries,
        full_forward_gpu_ms=(
            max(row["end_ns"] for row in gpu) - min(row["start_ns"] for row in gpu)
        )
        / 1e6,
        cpu_forward_ms=(scopes[0]["end_ns"] - scopes[0]["start_ns"]) / 1e6,
        source_sqlite_sha256=capture["sha256"],
        scheduler_collector_warnings=diagnostics,
        integrity_scope="All 65 formal forward scopes and every GPU node of the single decode graph are verified. Setup/warmup trace completeness is not claimed; collector warnings remain reported.",
    )
    return metrics, selected


def plot(panels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "official-h64k-decode",
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, constrained_layout=True)
    lanes = ["Compute", "H2D", "D2H", "GPU control", "GPU idle"]
    maximum = max(metrics["window_ms"] for metrics, _ in panels.values()) * 1.03
    for axis, (case, (metrics, activities)) in zip(axes, panels.items(), strict=True):
        origin = metrics["start_ns"]
        for row in activities:
            lane, category, hatched = decode_display.activity_style(row)
            if lane not in lanes:
                continue
            left = (max(origin, row["start_ns"]) - origin) / 1e6
            right = (min(metrics["end_ns"], row["end_ns"]) - origin) / 1e6
            color = timeline.COLORS.get(category, "#777777")
            axis.broken_barh(
                [(left, right - left)],
                (lanes.index(lane) - 0.30, 0.6),
                facecolors=color,
                edgecolors="#333333" if hatched else "none",
                hatch="///" if hatched else None,
                linewidth=0.25,
            )
        for left, right in metrics["gpu_idle_intervals_ns"]:
            axis.broken_barh(
                [((left - origin) / 1e6, (right - left) / 1e6)],
                (3.7, 0.6),
                facecolors=timeline.COLORS["GPU idle"],
            )
        for layer in metrics["layer_boundaries"]:
            x = (layer["start_ns"] - origin) / 1e6
            axis.axvline(x, color="#666666", linewidth=0.5, linestyle=":")
            axis.text(x, -0.48, f"L{layer['layer']}", va="bottom", fontsize=9)
        axis.set_yticks(range(len(lanes)), lanes)
        axis.set_ylim(4.6, -0.9)
        axis.set_xlim(0, maximum)
        axis.grid(axis="x", alpha=0.18)
        axis.set_title(
            f"{timeline.CASES[case]} | {metrics['window_ms']:.3f} ms | GPU idle {metrics['gpu_idle_ms']:.3f} ms",
            loc="left",
        )
    axes[-1].set_xlabel("Elapsed GPU time from first L0 compute kernel (ms)")
    legend = [
        Patch(color=timeline.COLORS[name], label=name)
        for name in [
            "Projection / RoPE",
            "Indexer / top-k",
            "Attention",
            "Output / MLP",
            "H2D",
            "D2H",
            "GPU idle",
        ]
    ]
    legend += [Patch(facecolor="#777777", hatch="///", label="Fused / potential mapped-host IO")]
    figure.legend(handles=legend, loc="outside upper center", ncol=4, frameon=False)
    figure.savefig(path, metadata={"Date": None})
    figure.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(figure)


def verify_run_group(runs, case, check_path):
    expected_case = "resident_reference" if case == "hbm" else "echo"
    require(
        all(
            run["case"] == expected_case
            and run["identity"]["case"] == expected_case
            and all(worker["completion"]["case"] == expected_case for worker in run["workers"])
            for run in runs.values()
        ),
        "Run case differs from its comparison label",
    )
    require(
        compatibility(runs["check"]["identity"])
        == compatibility(runs["bench"]["identity"])
        == compatibility(runs["profile"]["identity"]),
        "Check, timing and profile execution identities differ",
    )
    expected_check = {
        "run_id": runs["check"]["run_id"],
        "sha256": timeline.sha256(check_path / "run.json"),
    }
    prefixes = {
        worker["completion"]["prefix_sha256"] for run in runs.values() for worker in run["workers"]
    }
    require(len(prefixes) == 1, "Independent check and measured input tokens differ")
    prefix = next(iter(prefixes))
    require(
        all(
            worker["completion"]["row"]["input_sha256"] == prefix
            for run in runs.values()
            for worker in run["workers"]
        ),
        "Input hash differs",
    )
    for mode in ("bench", "profile"):
        require(
            all(runs[mode]["check"][key] == value for key, value in expected_check.items()),
            "Measured run references a different check",
        )
        require(
            Path(runs[mode]["check"]["path"]).resolve() == (check_path / "run.json").resolve(),
            "Check path differs",
        )
    return prefix


def verify_profile_binding(profile_worker, hooks, sqlite):
    require(
        {path.name: timeline.sha256(path) for path in hooks.glob("*.json")}
        == profile_worker["hooks"],
        "Profile hook artifacts changed",
    )
    nsys = profile_worker["nsys"]
    require(timeline.sha256(nsys["path"]) == nsys["sha256"], "Profile NSYS report changed")
    export = json.loads(sqlite.with_suffix(".export.json").read_text())
    require(
        export["exitcode"] == 0
        and export["input"] == nsys
        and export["output"]["sha256"] == timeline.sha256(sqlite)
        and Path(export["output"]["path"]).resolve() == sqlite.resolve(),
        "SQLite is not the verified export of the selected profile",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for case in ("hbm", "echo"):
        for mode in ("check", "bench", "profile"):
            parser.add_argument(f"--{case}-{mode}", type=Path, required=True)
        parser.add_argument(f"--{case}-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    panels, sources, samples = {}, {}, []
    for case in ("hbm", "echo"):
        runs = {
            mode: read_run(getattr(args, f"{case}_{mode}"), mode)
            for mode in ("check", "bench", "profile")
        }
        prefix = verify_run_group(runs, case, getattr(args, f"{case}_check"))
        for mode, record in runs.items():
            sources[f"{case}_{mode}"] = {
                "run_id": record["run_id"],
                "run_sha256": timeline.sha256(getattr(args, f"{case}_{mode}") / "run.json"),
                "raw_log_sha256": {
                    worker["name"]: {
                        name: timeline.sha256(name) for name in worker["capacity"]["log_sha256"]
                    }
                    for worker in record["workers"]
                },
                "capacity_log_digest_boundary": "Original capacity evidence hashes UTF-8 text after Python universal-newline conversion. raw_log_sha256 independently binds the original file bytes at publication.",
            }
        hooks = getattr(args, f"{case}_profile") / "sample_00/engine/hooks"
        profile_worker = runs["profile"]["workers"][0]
        verify_profile_binding(profile_worker, hooks, getattr(args, f"{case}_sqlite"))
        panels[case] = select(getattr(args, f"{case}_sqlite"), hooks, case)
        for index, worker in enumerate(runs["bench"]["workers"]):
            samples.append({"case": case, "sample": index, **worker["completion"]["row"]})
        residency = json.loads((hooks / "residency.json").read_text())
        require(
            residency["decode_input_ids"] == profile_worker["completion"]["row"]["output_ids"][:1],
            "Decode input is not the first sampled output token",
        )
        sources[case] = {
            "residency": residency,
            "prefix_sha256": prefix,
            "runtime": runs["profile"]["identity"]["preflight"]["runtime"],
            "capacity": profile_worker["capacity"],
            "engine_arguments": runs["profile"]["identity"]["engine_arguments"],
            "hook_files": {path.name: timeline.sha256(path) for path in hooks.glob("*.json")},
            "sqlite_sha256": timeline.sha256(getattr(args, f"{case}_sqlite")),
        }
    require(
        sources["hbm"]["prefix_sha256"] == sources["echo"]["prefix_sha256"],
        "HBM and ECHO requests have different histories",
    )
    require(
        sources["hbm"]["runtime"]["devices"] == sources["echo"]["runtime"]["devices"],
        "HBM and ECHO profiles use different hardware",
    )
    plot(panels, args.output / "timeline_decode.svg")
    metrics = [row[0] for row in panels.values()]
    (args.output / "windows.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "activities.json").write_text(
        json.dumps({case: row[1] for case, row in panels.items()}, indent=2) + "\n"
    )
    fields = [
        "case",
        "sample",
        "wall_ms",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "input_sha256",
    ]
    with (args.output / "timing_samples.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    summary = {
        case: {
            "samples": sum(row["case"] == case for row in samples),
            "full_request_median_ms": statistics.median(
                row["wall_ms"] for row in samples if row["case"] == case
            ),
        }
        for case in panels
    }
    report = {
        "schema": "echo-normal-decode-timeline-report-v1",
        "numerical_acceptance": False,
        "sources": sources,
        "analyzer_sha256": timeline.sha256(__file__),
        "timeline_helper_sha256": timeline.sha256(timeline.__file__),
        "windows": metrics,
        "request_timing": summary,
        "measurement": "H65536 greedy generation of two tokens: 64 normal prefill chunks, then one autoregressive decode at position 65536. Independent timing covers the complete Engine.generate. Timeline is intrusive NSYS L0-L2 decode GPU work with official CUDA Graph enabled; natural cache residency is preserved.",
        "io_boundary": decode_display.RECALL_DISPLAY["boundary"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), Path(timeline.__file__), Path(decode_display.__file__)):
        shutil.copyfile(source, snapshot / source.name)
    manifest = {
        "schema": "echo-decode-publication-v1",
        "source_run": args.output.name,
        "files": {
            path.name: timeline.sha256(path)
            for path in sorted(args.output.iterdir())
            if path.is_file()
        },
        "analysis_source_snapshot": {
            path.name: timeline.sha256(path) for path in snapshot.iterdir()
        },
    }
    (args.output / "publication.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        shutil.copytree(args.output, args.publish, ignore=shutil.ignore_patterns("source_snapshot"))
    print(
        json.dumps(
            {
                "request_timing": summary,
                "windows_ms": {case: row[0]["window_ms"] for case, row in panels.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
