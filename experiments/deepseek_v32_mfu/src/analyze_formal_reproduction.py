"""Compare two complete formal profiles using native HBM graph ownership."""

import argparse
import collections
import json
import shutil
import sqlite3
from pathlib import Path

from experiments.deepseek_v32_mfu.src.analyze_capture_boundary import PROCESS_MASK, digest, summary
from experiments.deepseek_v32_mfu.src.analyze_event_boundary import signature
from experiments.deepseek_v32_mfu.src.analyze_replay_timer import metadata
from experiments.deepseek_v32_motivation.src.graph_attribution import original_node, read_lineage


def control_sources(result, path, variant):
    if variant is None:
        assert "diagnostic_control" not in result
        return {}
    control = result["diagnostic_control"]
    kinds = {
        "inspector": "initial-compute-graph-inspector-omitted-v1",
        "collection": "initial-compute-bank-outside-collection-v1",
    }
    assert control["kind"] == kinds[variant]
    assert control["compute_preparation_calls"] == 1
    assert control["prefill_operator_attribution_available"] is False
    assert control["formal_gap_gate_available"] is False
    assert not (path / "graph_templates.json").exists()
    assert control["measurement_source"] == (
        f"experiments/deepseek_v32_mfu/src/q1_compute_{variant}.py"
    )
    return {control["measurement_source"]: control["measurement_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--control",
        choices=("identical", "compute-inspector", "compute-collection"),
        default="identical",
    )
    args = parser.parse_args()
    paths = [args.reference.resolve(), args.reproduction.resolve()]
    assert paths[0] != paths[1]
    assert not args.output_dir.exists()
    reference_result = json.loads((paths[0] / "result.json").read_text())
    variants = {
        "identical": (None, None),
        "compute-inspector": (None, "inspector"),
        "compute-collection": ("inspector", "collection"),
    }[args.control]
    reference_added = control_sources(reference_result, paths[0], variants[0])
    common_measurement_sources = {
        key: value
        for key, value in reference_result["measurement_identity"]["source_sha256"].items()
        if key not in reference_added
    }
    out = {
        "runs": {},
        "inputs_sha256": {},
        "analysis_sha256": digest(__file__),
        "control": args.control,
    }
    prior_sigs = None
    prior_owners = None
    prior_meta = None
    for p, variant in zip(paths, variants, strict=True):
        rid = p.name
        result = json.loads((p / "result.json").read_text())
        assert result["accepted"] is True and result["mode"] == "profile"
        assert result["run_id"] == rid
        for field in (
            "execution_identity",
            "execution_environment",
            "validation_receipt",
            "benchmark",
        ):
            assert result[field] == reference_result[field], field
        added = control_sources(result, p, variant)
        expected_measurement = {
            **reference_result["measurement_identity"],
            "source_sha256": {**common_measurement_sources, **added},
        }
        assert result["measurement_identity"] == expected_measurement
        for name, subdir in (
            ("sources.json", "source"),
            ("measurement_sources.json", "measurement_source"),
        ):
            manifest = json.loads((p / name).read_text())
            expected = json.loads((paths[0] / name).read_text())
            if name == "measurement_sources.json":
                expected = {**common_measurement_sources, **added}
            assert manifest == expected, name
            for relative, expected in manifest.items():
                assert digest(p / subdir / relative) == expected, relative
        assert digest(p / "request.json") == result["request_sha256"]
        for name in ("result.json", "sources.json", "measurement_sources.json", "request.json"):
            out["inputs_sha256"][str(p / name)] = digest(p / name)
        with sqlite3.connect((p / "capture_1.sqlite").as_uri() + "?mode=ro", uri=True) as setup:
            setup_names = dict(setup.execute("select id,value from StringIds"))
            setup_counts = {
                setup_names[name]: count
                for name, count in setup.execute(
                    "select nameId,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId"
                )
            }
            for name, count in (
                ("cudaStreamBeginCapture_v10000", 0 if variant == "collection" else 12),
                ("cudaStreamEndCapture_v10000", 0 if variant == "collection" else 12),
                ("cudaGraphLaunch_v10000", 0 if variant == "collection" else 6),
                (
                    "cudaGraphGetNodes_v10000",
                    {None: 306, "inspector": 12, "collection": 0}[variant],
                ),
                ("cudaGraphNodeGetType_v10000", 0 if added else 2301),
            ):
                assert setup_counts.get(name, 0) == count, (rid, name)
            if variant == "collection":
                assert (
                    setup.execute(
                        "select count(*) from NVTX_EVENTS where text='compute_bank_already_prepared'"
                    ).fetchone()[0]
                    == 1
                )
                tables = {row[0] for row in setup.execute("select name from sqlite_master")}
                for kind in ("KERNEL", "MEMCPY", "MEMSET"):
                    table = "CUPTI_ACTIVITY_KIND_" + kind
                    assert (
                        table not in tables
                        or setup.execute(f"select count(*) from {table}").fetchone()[0] == 0
                    )
            out["inputs_sha256"][str(p / "capture_1.sqlite")] = digest(p / "capture_1.sqlite")
        t = next(
            x
            for x in json.loads((p / "full_graph_templates.json").read_text())
            if x["method"] == "hbm"
        )
        parents = read_lineage([p / "capture_3.sqlite"])
        c = sqlite3.connect((p / "capture_4.sqlite").as_uri() + "?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        names = dict(c.execute("select id,value from StringIds"))
        scopes = c.execute(
            "select * from NVTX_EVENTS where text like 'echo/hbm/%/shared/extend_graph_replay_q_1/call_3' order by start"
        ).fetchall()
        assert len(scopes) == 2
        met = metadata(c)
        if prior_meta is not None:
            assert met == prior_meta
        prior_meta = met
        runs = []
        for s in scopes:
            launches = [
                dict(x)
                for x in c.execute(
                    "select * from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=? and globalTid=?",
                    (s["start"], s["end"], s["globalTid"]),
                )
                if "cudaGraphLaunch" in names[x["nameId"]]
            ]
            assert len(launches) == 1
            l = launches[0]
            assert s["globalTid"] & PROCESS_MASK == parents.process
            nodes = []
            for kind in ("kernel", "memcpy", "memset"):
                for r in c.execute(
                    f"select * from CUPTI_ACTIVITY_KIND_{kind.upper()} where correlationId=? and globalPid=?",
                    (l["correlationId"], parents.process),
                ):
                    r = dict(r)
                    r.update(
                        kind=kind,
                        name=names[r["demangledName"]]
                        if kind == "kernel"
                        else "Device-to-Device"
                        if kind == "memcpy" and r["copyKind"] == 8
                        else kind,
                    )
                    assert kind != "kernel" or r["graphId"] == t["executable_graph_id"]
                    r["capture_node"] = original_node(r["graphNodeId"], parents)
                    r["owner"] = t["node_owners"][str(r["capture_node"])]
                    nodes.append(r)
            nodes.sort(key=lambda r: r["capture_node"])
            assert len(nodes) == 197 and {r["capture_node"] for r in nodes} == set(
                t["gpu_node_ids"]
            )
            assert dict(collections.Counter(t["node_types"].values())) == {0: 181, 1: 15, 2: 1}
            sigs = [signature(r) for r in nodes]
            owners = [r["owner"] for r in nodes]
            if prior_sigs is not None:
                assert sigs == prior_sigs and owners == prior_owners
            prior_sigs = sigs
            prior_owners = owners
            layer = [r for r in nodes if r["owner"]["layer"] in ("layer_0", "layer_1", "layer_2")]
            assert len(layer) == 192
            stats = summary(layer)
            clipped = []
            for kind in ("kernel", "memcpy", "memset"):
                for r in c.execute(
                    f"select * from CUPTI_ACTIVITY_KIND_{kind.upper()} where end>? and start<?",
                    (stats["start_ns"], stats["end_ns"]),
                ):
                    r = dict(r)
                    r.update(
                        kind=kind,
                        name=names[r["demangledName"]]
                        if kind == "kernel"
                        else "Device-to-Device"
                        if kind == "memcpy" and r["copyKind"] == 8
                        else kind,
                    )
                    r["unclipped_start"] = r["start"]
                    r["unclipped_end"] = r["end"]
                    r["start"] = max(r["start"], stats["start_ns"])
                    r["end"] = min(r["end"], stats["end_ns"])
                    clipped.append(r)
            allstats = summary(clipped)
            assert (allstats["span_us"], allstats["busy_us"], allstats["idle_us"]) == (
                stats["span_us"],
                stats["busy_us"],
                stats["idle_us"],
            )
            runs.append(
                {
                    "scope": s["text"],
                    "launch_duration_us": (l["end"] - l["start"]) / 1000,
                    "statistics": stats,
                    "all_process_clipped_union": allstats,
                    "clipped_activities": clipped,
                    "nodes": nodes,
                    "signatures": sigs,
                }
            )
            print(
                rid,
                s["text"],
                {
                    k: stats[k]
                    for k in ("span_us", "busy_us", "idle_us", "gap_median_ns", "gap_max_ns")
                },
                "launch_us",
                runs[-1]["launch_duration_us"],
            )
        out["runs"][rid] = {
            "metadata": met,
            "compute_setup_runtime_counts": setup_counts,
            "node_signature_and_ownership_identical": True,
            "replays": runs,
        }
        for f in ("capture_3.sqlite", "capture_4.sqlite", "full_graph_templates.json"):
            out["inputs_sha256"][str(p / f)] = digest(p / f)
        c.close()
    out["boundary"] = (
        "Complete formal pipelines with the declared preparation control only. Exactly 197 native HBM GPU nodes and 192 L0-L2 nodes, identical ordered names/all 17 kernel launch fields/copy bytes and native ownership. Every process activity intersecting the selected window is clipped; its interval union must match. Execution source archives, execution identities and relevant raw NSYS settings match; the compute-inspector control adds one explicitly bound measurement source and omits the compute ledger. Receipt/native/runtime bytes and saved outputs require the separate independent audit. This invasive profile is not clean model latency or a hardware-cause proof."
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "windows.json"
    output.write_text(json.dumps(out, indent=2) + "\n")
    shutil.copy2(__file__, args.output_dir / Path(__file__).name)
    print(output)


if __name__ == "__main__":
    main()
