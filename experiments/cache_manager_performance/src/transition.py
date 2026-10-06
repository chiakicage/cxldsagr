"""Measure exact top-k completion to real MLA execution in NSYS captures.

This is read-only profile analysis, not a new GPU run. Complete-model captures
and independent replay captures retain their separate execution provenance.
"""

import argparse
import hashlib
import json
import os
import re
from itertools import pairwise
from pathlib import Path

from evaluation.validation import identity_digest
from experiments.cache_manager_performance.src.analyze import (
    SCOPE as MANAGER_SCOPE,
)
from experiments.cache_manager_performance.src.analyze import (
    annotate_diagnostic_io,
    diagnostic_lane,
)
from experiments.cache_manager_performance.src.transition_metrics import analyze_transition
from experiments.deepseek_v32_mfu.src.analyze_nsys import (
    _assign_scopes,
    _attribute,
    _read_capture,
)
from experiments.deepseek_v32_mfu.src.launch_gap import annotate_actual_io
from experiments.deepseek_v32_mfu.src.operator_report import _SCOPE as MODEL_SCOPE
from experiments.deepseek_v32_mfu.src.timeline import activity_lane, select_forward_activities

ROOT = Path(__file__).resolve().parents[3]
SCHEMES = ("echo", "serial_sparse")
STAGES = {
    "indexer",
    "indexer_prefetch",
    "indexer_qk",
    "indexer_fused",
    "offload_finalize",
    "exact_topk",
    "prefetch_hint",
    "cache_write",
    "offload_exact_recall",
    "sparse_mla",
    "attention_output",
}
EXECUTION_SOURCES = (
    "models/deepseek_v32/attention.py",
    "cache/sparse_token_cache.py",
    "cache/sparse_token_pool.py",
    "operators/deepseek_v32/indexer/selection.py",
    "operators/deepseek_v32/indexer/prefetch_hint.py",
    "operators/deepseek_v32/indexer/csrc/echo_cache.cuh",
    "operators/common/csrc/kv_transfer.cu",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"source artifact hash mismatch: {path}")
    return {"path": str(path), "sha256": digest}


def execution_sources(profile, sources, extra=()):
    return {
        relative: {
            **binding(profile / "source" / relative, sources[relative]),
            "matches_current_source": sha(ROOT / relative) == sources[relative],
        }
        for relative in (*EXECUTION_SOURCES, *extra)
    }


def capture(path, pattern):
    scopes, apis, activities, tables = _read_capture(path.resolve(), scope_pattern=pattern)
    _assign_scopes(apis, scopes)
    _attribute(apis, activities)
    return scopes, apis, activities, tables


def _post_topk_detail(detail):
    """Keep boundary evidence without publishing pre-top-k timing metrics."""
    return {
        **{key: value for key, value in detail.items() if key != "gpu_indexer_to_topk"},
        "boundaries": {
            key: value
            for key, value in detail["boundaries"].items()
            if not key.startswith("indexer_")
        },
        "stages": [
            stage
            for stage in detail["stages"]
            if stage["stage"]
            in {"prefetch_hint", "cache_write", "offload_exact_recall", "sparse_mla"}
        ],
        "notes": [note for note in detail["notes"] if "manager_ready" not in note],
    }


def model_transitions(profile):
    profile = Path(profile).resolve()
    result, audit = read(profile / "result.json"), read(profile / "gap_audit.json")
    if result.get("accepted") is not True or result.get("mode") != "profile":
        raise ValueError("model transition requires an accepted profile run")
    result_binding = binding(profile / "result.json", audit["input_result_sha256"])
    if audit["run_id"] != result["run_id"]:
        raise ValueError("model audit belongs to a different run")
    candidates = [row for row in audit["methods"] if row["method"] in SCHEMES]
    if len(candidates) != 2 or {row["method"] for row in candidates} != set(SCHEMES):
        raise ValueError("model audit must identify one capture per sparse method")
    rows, captures = [], []
    for expected in candidates:
        scheme = expected["method"]
        path = profile / expected["sqlite"]
        source = binding(path, expected["sqlite_sha256"])
        scopes, apis, activities, tables = capture(path, MODEL_SCOPE)
        if {(s["mode"], s["phase"]) for s in scopes} != {(scheme, "extend_annotated")}:
            raise ValueError("model capture method or phase differs from audit")
        activities, boundary = select_forward_activities(scopes, activities)
        if boundary != expected["capture_boundary"]:
            raise ValueError("model measured forward boundary differs from audit")
        counters = result["measurements"][scheme]["extend_cache_per_layer"]
        annotate_actual_io(activities, counters, scheme, "extend_annotated")
        for layer in range(result["num_layers"]):
            selected = [
                s
                for s in scopes
                if s["layer"] == f"layer_{layer}"
                and s["stage"] in STAGES
                and boundary["measured_start_ns"] <= s["start"] < boundary["measured_end_ns"]
                and s["stage"] != "attention_output"
            ]
            detail = analyze_transition(
                selected,
                apis,
                activities,
                indexer_stage="indexer_fused" if scheme == "echo" else "indexer_qk",
                topk_stage="exact_topk",
                consumer_stage="sparse_mla",
                endpoint_kind="attention_kernel",
                lane_classifier=activity_lane,
            )
            # Compute islands are outside this transition. Do not silently use
            # a graph classifier without its separate node-lineage evidence.
            for span in ("gpu_indexer_to_topk", "gpu_topk_to_consumer"):
                left, right = detail[span]["start_ns"], detail[span]["end_ns"]
                if any(
                    a.get("graph_id") and a["start"] < right and a["end"] > left for a in activities
                ):
                    raise ValueError("transition overlaps a graph; node lineage is required")
            rows.append(
                {
                    "source_kind": "complete_model",
                    "run_id": result["run_id"],
                    "scheme": scheme,
                    "phase": f"extend_{result['extend_residency']}",
                    "sample": 0,
                    "layer": layer,
                    "counters": counters[layer],
                    **_post_topk_detail(detail),
                }
            )
        captures.append({**source, "scheme": scheme, "tables": tables, "boundary": boundary})
    provenance = {
        "kind": "complete_model",
        "run_id": result["run_id"],
        "result": result_binding,
        "audit": binding(profile / "gap_audit.json"),
        "captures": captures,
        "sources_manifest": binding(profile / "sources.json"),
        "execution_sources": execution_sources(
            profile,
            read(profile / "sources.json"),
            (
                "operators/deepseek_v32/attention/offload/mla.py",
                "operators/deepseek_v32/attention/device_only/mla.py",
            ),
        ),
        "workload": {
            key: result[key]
            for key in (
                "prefix_tokens",
                "extend_tokens",
                "chunk_size",
                "slots",
                "num_layers",
                "extend_residency",
                "compute_graphs",
                "scope",
            )
        },
        "hardware": result["hardware"],
        "validation_receipt": result["validation_receipt"],
        "runtime_identity_limit": "Inherits V10: resident mask live JIT/CUBIN was not separately archived.",
    }
    return rows, provenance


def _manager_samples(result, scopes):
    """Pair repeated labels through ordered windows and unique scope IDs."""
    repeats = result.get("repeats")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("manager profile requires a positive repeat count")
    expected = {(scheme, phase) for scheme in SCHEMES for phase in ("extend_cold", "extend_warm")}
    samples = result["samples"]
    if len(samples) != repeats * len(expected):
        raise ValueError("manager profile sample count differs from sparse extend repeats")
    sample_groups = {key: [] for key in expected}
    for sample in samples:
        key = sample["scheme"], sample["phase"]
        if key not in expected or type(sample.get("sample")) is not int:
            raise ValueError("manager profile contains an unexpected sample identity")
        sample_groups[key].append(sample["sample"])
    if any(indices != list(range(repeats)) for indices in sample_groups.values()):
        raise ValueError("manager samples must have ordered indices from zero for every repeat")

    scope_ids = set()
    windows = []
    for scope in scopes:
        scope_id, start, end = scope.get("id"), scope.get("start"), scope.get("end")
        if scope_id is None or scope_id in scope_ids:
            raise ValueError("manager profile scope IDs must be unique")
        scope_ids.add(scope_id)
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("manager profile contains invalid scope timestamps")
        if (scope["scheme"], scope["phase"]) not in expected:
            raise ValueError("manager capture contains an unexpected scheme or phase")
        if scope["stage"] == "window":
            if start == end:
                raise ValueError("manager synchronized windows must be nonempty")
            windows.append(scope)
        elif not scope["stage"].startswith("window/"):
            raise ValueError("manager scope is outside a measured window")
    windows.sort(key=lambda scope: (scope["start"], scope["id"]))
    if any(left["end"] > right["start"] for left, right in pairwise(windows)):
        raise ValueError("manager synchronized windows must not overlap")
    window_groups = {key: [] for key in expected}
    for window in windows:
        window_groups[window["scheme"], window["phase"]].append(window)
    if any(len(group) != repeats for group in window_groups.values()):
        raise ValueError("manager capture window count differs from sparse extend repeats")

    groups = {window["id"]: [] for window in windows}
    for scope in scopes:
        containing = [
            window
            for window in window_groups[scope["scheme"], scope["phase"]]
            if window["thread"] == scope["thread"]
            and window["start"] <= scope["start"]
            and scope["end"] <= window["end"]
        ]
        if len(containing) != 1:
            raise ValueError("manager scope does not belong to one synchronized window")
        groups[containing[0]["id"]].append(scope)
    return [
        (
            sample_index,
            sample,
            window_groups[sample["scheme"], sample["phase"]][sample["sample"]],
            groups[window_groups[sample["scheme"], sample["phase"]][sample["sample"]]["id"]],
        )
        for sample_index, sample in enumerate(samples)
    ]


def manager_transitions(profile):
    profile = Path(profile).resolve()
    result_source = binding(profile / "result.json")
    identity_source = binding(profile / "identity.json")
    capture_source = binding(profile / "trace.sqlite")
    result, identity = read(profile / "result.json"), read(profile / "identity.json")
    if result.get("schema") != "deepseek-cache-manager-transition-v2":
        raise ValueError(
            "manager transition requires the real-MLA v2 schema; no-op replays are excluded"
        )
    if result.get("passed") is not True or result.get("mode") != "profile":
        raise ValueError("manager transition requires a checked profile run")
    if identity_digest(identity) != result["identity_sha256"]:
        raise ValueError("manager execution identity differs from result")
    scopes, apis, activities, tables = capture(profile / "trace.sqlite", MANAGER_SCOPE)
    layers = identity["config"]["layers"]
    if type(layers) is not int or layers < 1:
        raise ValueError("manager execution identity requires a positive layer count")
    rows = []
    for sample_index, sample, window, group in _manager_samples(result, scopes):
        scheme, phase = sample["scheme"], sample["phase"]
        if len(sample["metrics"]) != layers or len(sample["checks"]) != layers:
            raise ValueError("manager sample layer count differs from its execution identity")
        if any(
            check.get("attention_finite") is not True
            or check.get("attention_matches_resident") is not True
            for check in sample["checks"]
        ):
            raise ValueError("manager transition requires checked real MLA outputs")
        group_ids = {scope["id"] for scope in group}
        active = [
            activity
            for activity in activities
            if (activity["scope"] or {}).get("id") in group_ids
            or activity["start"] < window["end"]
            and activity["end"] > window["start"]
        ]
        for activity in active:
            s = activity["scope"]
            if (
                s is None
                or s["id"] not in group_ids
                or activity["start"] < window["start"]
                or activity["end"] > window["end"]
            ):
                raise ValueError("manager activity is unattributed or escaped its window")
        annotate_diagnostic_io(active, sample, sample_index)
        # Normalize stage names only after the per-layer transport counters
        # have been bound through the original full production NVTX labels.
        normalized = {s["id"]: {**s, "stage": s["stage"].split("/")[-1]} for s in group}
        normalized_apis = [
            {**a, "scope": normalized.get((a["scope"] or {}).get("id"), a["scope"])} for a in apis
        ]
        normalized_active = [{**a, "scope": normalized[a["scope"]["id"]]} for a in active]
        for layer in range(layers):
            selected = [
                normalized[s["id"]]
                for s in group
                if s["stage"].startswith(f"window/layer_{layer}/")
                and normalized[s["id"]]["stage"] in STAGES
            ]
            detail = analyze_transition(
                selected,
                normalized_apis,
                normalized_active,
                indexer_stage="indexer_prefetch" if scheme == "echo" else "indexer",
                topk_stage="exact_topk",
                consumer_stage="sparse_mla",
                endpoint_kind="attention_kernel",
                lane_classifier=diagnostic_lane,
            )
            rows.append(
                {
                    "source_kind": "standalone_replay",
                    "run_id": result["run_id"],
                    "scheme": scheme,
                    "phase": phase,
                    "sample": sample["sample"],
                    "layer": layer,
                    "counters": sample["metrics"][layer],
                    **_post_topk_detail(detail),
                }
            )
    provenance = {
        "kind": "standalone_replay",
        "run_id": result["run_id"],
        "result": result_source,
        "identity": identity_source,
        "capture": capture_source,
        "tables": tables,
        "execution_identity_sha256": result["identity_sha256"],
        "execution_sources": execution_sources(
            profile,
            identity["sources"],
            (
                "experiments/cache_manager_performance/src/workload.py",
                "operators/deepseek_v32/attention/offload/mla.py",
                "operators/deepseek_v32/attention/device_only/mla.py",
            ),
        ),
        "workload": identity["config"],
        "environment": identity["environment"],
        "validation_receipt_sha256": result["validation_receipt"],
        "repeats": result["repeats"],
        "sample_window_binding": "Per scheme/phase, result sample indices bind chronological synchronized NVTX windows; child scopes retain unique capture IDs.",
        "runtime_identity_limit": "Execution sources and native identity are bound by the run; individual live JIT/CUBIN artifacts are not separately proven by this analyzer.",
    }
    return rows, provenance


def write_analysis(output, run_id, *, manager_run=None, model_run=None):
    output = Path(output)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("analysis ID must be a plain nonempty name")
    rows, provenance = [], []
    for path, loader in ((manager_run, manager_transitions), (model_run, model_transitions)):
        if path is not None:
            samples, source = loader(path)
            rows.extend(samples)
            provenance.append(source)
    if not rows:
        raise ValueError("provide a manager profile, model profile, or both")
    output.mkdir(parents=True, exist_ok=False)
    helpers = (
        "experiments/cache_manager_performance/src/transition.py",
        "experiments/cache_manager_performance/src/transition_metrics.py",
        "experiments/cache_manager_performance/src/transition_report.py",
        "experiments/cache_manager_performance/src/analyze.py",
        "experiments/deepseek_v32_mfu/src/analyze_nsys.py",
        "experiments/deepseek_v32_mfu/src/launch_gap.py",
        "experiments/deepseek_v32_mfu/src/timeline.py",
        "experiments/deepseek_v32_mfu/src/operator_report.py",
        "evaluation/validation.py",
    )
    sources = {}
    for relative in helpers:
        destination = output / "analysis_sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
        sources[relative] = {
            "archive": str(destination.relative_to(output)),
            "sha256": sha(destination),
        }
    report = {
        "schema": "cache-manager-attention-transition-v2",
        "analysis_id": run_id,
        "new_gpu_run": False,
        "sources": provenance,
        "analysis_sources": sources,
        "artifact_path_base": "analysis output directory",
        "samples": rows,
        "notes": [
            "Read-only analysis of intrusive NSYS samples; profiler overhead remains.",
            "CPU submission, GPU service, actual IO and idle are separate overlapping views; do not add them.",
            "The measured interval starts at GPU exact top-k completion and ends at the first real MLA kernel start; indexer, top-k and MLA computation are outside it.",
            "Independent replay executes the real MLA wrapper and validates its output against resident replay; complete-model captures remain a separate source kind.",
            "GPU idle is observed inactivity, not proven Python, operating-system, or hardware latency.",
            "CPU API scopes preserve launch attribution; later CPU work overlapping a GPU gap is not its cause.",
            "Only ordinary persistent extend is covered; no GR transient candidate or split-query consumer.",
        ],
    }

    # Relative evidence paths survive profile.sh's successful staging-directory
    # move. Recheck sources after parsing rather than trusting a mutable input.
    def relativize(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                binding(value["path"], value["sha256"])
                value["path"] = os.path.relpath(value["path"], output.resolve())
            for child in value.values():
                relativize(child)
        elif isinstance(value, list):
            for child in value:
                relativize(child)

    relativize(report["sources"])
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    from experiments.cache_manager_performance.src.transition_report import render

    render(report, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager-run", type=Path)
    parser.add_argument("--model-run", type=Path)
    parser.add_argument("--run-id", required=True, help="new analysis ID; not a new GPU run")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = write_analysis(
        args.output_dir, args.run_id, manager_run=args.manager_run, model_run=args.model_run
    )
    print(json.dumps({"analysis_id": report["analysis_id"], "transitions": len(report["samples"])}))


if __name__ == "__main__":
    main()
