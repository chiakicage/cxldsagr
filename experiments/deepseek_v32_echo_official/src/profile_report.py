"""Audit actual Nsight activities for official cold/revisit diagnostic captures."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_echo_official.src import measure
from experiments.deepseek_v32_echo_official.src.measure import digest, write_json
from experiments.deepseek_v32_echo_official.src.profile import SCHEMA
from experiments.deepseek_v32_echo_official.src.report import contained, require
from experiments.deepseek_v32_echo_prefill.src.analyze_nsys import (
    _api_summary,
    _assign_scopes,
    _attribute,
    _device_summary,
    _group_summary,
    _kernel_summaries,
    _read_capture,
)
from experiments.deepseek_v32_motivation.src.analyze_pipeline import PATTERN
from experiments.deepseek_v32_motivation.src.report import write_csv
from experiments.nosa_motivation.src.cpu_environment import validate_cpu_environment_record
from experiments.nosa_motivation.src.validation import base_identity


def audit_evidence(directory, metadata, *, receipt_override=None):
    config = metadata["config"]
    schemes = [case["scheme"] for case in metadata["cases"]]
    require(
        bool(schemes)
        and len(set(schemes)) == len(schemes)
        and set(schemes) <= set(measure.SCHEMES),
        "profile cases must be nonempty, unique official schemes",
    )
    count = config["num_users"] + 1
    require(
        all(case["requests"] == count for case in metadata["cases"]),
        "incomplete profile trajectory",
    )
    expected = [
        (scheme, phase, request)
        for scheme in schemes
        for phase, request in (("cold", 0), ("revisit", config["num_users"]))
    ]
    require(
        [(item["scheme"], item["phase"], item["request_id"]) for item in metadata["captures"]]
        == expected,
        "profile must capture cold and revisit at the declared request IDs",
    )
    artifacts = metadata["profile_artifacts"]
    require(
        {
            "calls.json",
            "correctness.json",
            "source_manifest.json",
            "workload/requests.jsonl",
            "workload/workload.json",
        }
        <= artifacts.keys(),
        "profile evidence inventory is incomplete",
    )
    for relative, sha in artifacts.items():
        require(
            digest(contained(directory, relative)) == sha, f"profile evidence changed: {relative}"
        )
    source = json.loads((directory / "source_manifest.json").read_text())
    source_id = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    require(
        bool(source) and source_id == metadata["source_sha256"], "profile source identity differs"
    )
    for name, sha in source.items():
        relative = "source/" + name
        require(artifacts.get(relative) == sha, f"profile source not bound: {name}")
        require(digest(contained(directory, relative)) == sha, f"profile source changed: {name}")
    receipt_evidence = metadata["correctness_receipt"]
    receipt_path = receipt_override or receipt_evidence["path"]
    require(
        digest(receipt_path) == receipt_evidence["sha256"] == metadata["reference_receipt_sha256"],
        "profile reference receipt changed",
    )
    receipt = require_receipt(
        receipt_path, kind=measure.RECEIPT_KIND, identity=receipt_evidence["identity"]
    )
    checked = json.loads(Path(receipt["artifact_paths"]["metadata"]).read_text())
    require(
        checked.get("schema") == measure.CHECK_SCHEMA and checked.get("status") == "accepted",
        "profile reference is not an accepted independent check",
    )
    actual = metadata["validation_identity"]
    require(
        actual["base"] == base_identity(metadata, directory) == receipt["identity"]["base"],
        "profile execution identity differs from check",
    )
    require(
        set(actual["methods"]) == set(schemes)
        and all(
            actual["methods"][scheme] == receipt["identity"]["methods"][scheme]
            for scheme in schemes
        ),
        "profile runtime methods differ from check",
    )
    require(config == checked["config"], "profile configuration differs from check")
    for name in ("workload/requests.jsonl", "workload/workload.json"):
        reference_path = Path(receipt["artifact_paths"]["metadata"]).parent / name
        require(
            digest(directory / name) == digest(reference_path),
            "profile workload differs from check",
        )
    cpu = validate_cpu_environment_record(metadata)
    require(
        cpu["status"] == "recorded_and_equal", "profile lacks CPU environment start/end evidence"
    )
    require(
        metadata["execution_environment"]
        == {
            key: value
            for key, value in metadata["profiling_environment"].items()
            if key != "CUDA_INJECTION64_PATH"
        },
        "profile model environment differs from observed instrumentation environment",
    )
    comparisons = json.loads((directory / "correctness.json").read_text())
    require(
        [(row["scheme"], row["request_id"]) for row in comparisons]
        == [(scheme, request) for scheme in schemes for request in range(count)],
        "profile numerical evidence does not cover the complete preparation and capture trajectory",
    )
    for row in comparisons:
        require(set(row["numerical"]) == {"hidden", "logits"}, "profile numerical outputs missing")
        measure.require_numerical(row)
    calls = json.loads((directory / "calls.json").read_text())
    require(bool(calls), "profile host calls are empty")
    for index, (scheme, phase, request) in enumerate(expected, 1):
        roots = [
            row for row in calls if row["capture_index"] == index and row["stage"] == "request"
        ]
        require(
            len(roots) == 1
            and roots[0]["scheme"] == scheme
            and roots[0]["phase"] == phase
            and roots[0]["request_id"] == request,
            "profile host capture identity differs",
        )
    result = json.loads((directory / "result.json").read_text())
    require(
        result
        == {
            "accepted": True,
            "captures": len(expected),
            "checked_requests": len(comparisons),
            "source_sha256": source_id,
            "metadata_sha256": digest(directory / "metadata.json"),
            "reference_receipt_sha256": metadata["reference_receipt_sha256"],
        },
        "profile completion record differs from saved evidence",
    )
    return {
        "source_sha256": source_id,
        "reference_receipt_sha256": metadata["reference_receipt_sha256"],
        "checked_requests": len(comparisons),
        "cpu_environment": cpu,
        "artifacts_checked": len(artifacts),
    }


def analyze(directory, output, *, receipt_override=None):
    directory, output = Path(directory).resolve(strict=True), Path(output)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("status") != "accepted":
        raise ValueError("official profile is incomplete or has a different schema")
    evidence = audit_evidence(directory, metadata, receipt_override=receipt_override)
    expected = [
        (case["scheme"], phase) for case in metadata["cases"] for phase in ("cold", "revisit")
    ]
    if [(item["scheme"], item["phase"]) for item in metadata["captures"]] != expected:
        raise ValueError("profile does not contain both cold and revisit for every scheme")
    if output.exists():
        raise FileExistsError(output)
    captures, flat_kernels, flat_stages = [], [], []
    for index, item in enumerate(metadata["captures"], 1):
        if item["capture_index"] != index or item["sqlite"] != f"capture_{index}.sqlite":
            raise ValueError("capture index or path differs from the declared sequence")
        path = directory / item["sqlite"]
        scopes, apis, activities, tables = _read_capture(path, scope_pattern=PATTERN)
        if not scopes or not activities:
            raise ValueError("empty NVTX or GPU capture")
        if any(
            (scope["scheme"], scope["phase"]) != (item["scheme"], item["phase"]) for scope in scopes
        ):
            raise ValueError("Nsight capture scheme/phase differs from metadata")
        _assign_scopes(apis, scopes)
        _attribute(apis, activities)
        stages = _group_summary(
            lambda scope: (scope["layer"], scope["segment"] + "/" + scope["stage"]),
            scopes,
            apis,
            activities,
        )
        _, kernels = _kernel_summaries(activities)
        device, envelope = _device_summary(activities)
        kernel_activity = [activity for activity in activities if activity["kind"] == "kernel"]
        if not kernel_activity or sum(row["count"] for row in kernels) != len(kernel_activity):
            raise ValueError("kernel count conservation failed")
        kernel_ns = sum(row["end"] - row["start"] for row in kernel_activity)
        if not math.isclose(sum(row["total_ms"] for row in kernels), kernel_ns / 1e6):
            raise ValueError("kernel duration conservation failed")
        missing = [row for row in activities if row["scope"] is None]
        capture = {
            **item,
            "sqlite_sha256": digest(path),
            "tables": tables,
            "kernel_count": len(kernel_activity),
            "kernel_duration_sum_ms": kernel_ns / 1e6,
            "devices": device,
            "gpu_envelope": envelope,
            "api": _api_summary(apis),
            "unattributed_activity_count": len(missing),
            "unattributed_activity_duration_sum_ms": sum(
                row["end"] - row["start"] for row in missing
            )
            / 1e6,
            "stage_count": len(stages),
            "kernel_count_conserved": True,
            "kernel_duration_conserved": True,
        }
        captures.append(capture)
        common = {"capture_index": index, "scheme": item["scheme"], "phase": item["phase"]}
        flat_kernels.extend({**common, **row} for row in kernels)
        flat_stages.extend({**common, **row} for row in stages)
    result = {
        "schema": "deepseek-v32-echo-official-profile-audit-v1",
        "status": "passed",
        "evidence": evidence,
        "captures": captures,
        "metadata_sha256": digest(directory / "metadata.json"),
        "boundary": "intrusive diagnostic only; duration sums can overlap and are not request latency",
        "attribution": "runtime/driver launch correlation to innermost host scope; unmatched graph activities remain explicit",
        "mapped_host_bytes": "not inferred from memcpy activity; use checked cache counters",
        "analyzer_sources": {
            str(path.relative_to(measure.ROOT)): digest(path)
            for path in sorted(
                {
                    Path(__file__).resolve(),
                    measure.ROOT / "experiments/deepseek_v32_motivation/src/analyze_pipeline.py",
                    *(
                        Path(inspect.getsourcefile(function)).resolve()
                        for function in (
                            require_receipt,
                            digest,
                            write_json,
                            contained,
                            require,
                            _read_capture,
                            _assign_scopes,
                            _attribute,
                            _group_summary,
                            _kernel_summaries,
                            _device_summary,
                            _api_summary,
                            write_csv,
                            validate_cpu_environment_record,
                            base_identity,
                        )
                    ),
                }
            )
        },
    }
    output.mkdir(parents=True)
    write_json(output / "analysis.json", result)
    write_csv(output / "kernels.csv", flat_kernels)
    write_csv(output / "stages.csv", flat_stages)
    return result


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--receipt-override", type=Path)
    args = command.parse_args(argv)
    analyze(args.run_dir, args.output_dir, receipt_override=args.receipt_override)


if __name__ == "__main__":
    main()
