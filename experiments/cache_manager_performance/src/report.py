"""Publish independently checked top-k-to-MLA profile intervals."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from evaluation.validation import identity_digest, require_receipt
from experiments.cache_manager_performance.src.measure import (
    KIND,
    MEASURED_PHASES,
    MEASURED_SCHEMES,
)
from experiments.cache_manager_performance.src.workload import file_sha256

OBSERVER_FILES = (
    "monitor_audit.json",
    "run.json",
    "gpu_monitor.jsonl",
    "driver.py",
    "stdout.log",
    "stderr.log",
)


def _validate_observer_reconciliation(reconciliation, directory, run_id, audit, result_path):
    races = audit.get("exit_races")
    if (
        not isinstance(reconciliation, dict)
        or reconciliation.get("accepted") is not True
        or reconciliation.get("classification") != "owned_process_exit_race"
        or reconciliation.get("run_id") != run_id
        or not isinstance(races, list)
        or not races
        or json.dumps(reconciliation.get("reconciled_exit_races"), sort_keys=True)
        != json.dumps(races, sort_keys=True)
    ):
        raise ValueError("observer reconciliation does not exactly cover the original exit races")
    recorded_directory = reconciliation.get("observer_directory")
    recorded_result = reconciliation.get("result_path")
    if (
        not isinstance(recorded_directory, str)
        or not Path(recorded_directory).is_absolute()
        or Path(recorded_directory).resolve(strict=True) != directory
        or result_path is None
        or not isinstance(recorded_result, str)
        or not Path(recorded_result).is_absolute()
    ):
        raise ValueError("observer reconciliation does not bind the measured paths")
    result_path = Path(result_path).resolve(strict=True)
    if Path(recorded_result).resolve(strict=True) != result_path:
        raise ValueError("observer reconciliation identifies a different result")
    files = reconciliation.get("observer_files")
    if (
        not isinstance(files, dict)
        or set(files) != set(OBSERVER_FILES)
        or any(file_sha256(directory / name) != digest for name, digest in files.items())
    ):
        raise ValueError("observer reconciliation source file hashes differ")
    result = json.loads(result_path.read_text())
    if (
        reconciliation.get("result_sha256") != file_sha256(result_path)
        or result.get("passed") is not True
        or result.get("run_id") != run_id
        or not result.get("identity_sha256")
        or reconciliation.get("identity_sha256") != result["identity_sha256"]
    ):
        raise ValueError("observer reconciliation result or execution identity differs")


def read_observer(directory, run_id, device_uuid, *, reconciliation=None, result_path=None):
    directory = Path(directory).resolve(strict=True)
    audit = json.loads((directory / "monitor_audit.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    command = run["command"]
    if "--run-id" not in command or command[command.index("--run-id") + 1] != run_id:
        raise ValueError("observer command does not identify the measured run")
    if (
        run.get("exit_code") != 0
        or audit.get("exit_code") != 0
        or audit.get("samples", 0) < 2
        or any(audit.get(key) for key in ("foreign_processes", "monitor_errors"))
    ):
        raise ValueError("observer did not accept the complete measured process")
    if reconciliation is not None:
        _validate_observer_reconciliation(reconciliation, directory, run_id, audit, result_path)
    elif audit.get("exit_races"):
        raise ValueError("observer exit races require an explicit independent reconciliation")
    monitor = directory / "gpu_monitor.jsonl"
    samples = [json.loads(line) for line in monitor.read_text().splitlines()]
    if len(samples) != audit["samples"] or any(
        row.get("returncode") != 0 or row.get("gpu_utilization_returncode") != 0 for row in samples
    ):
        raise ValueError("observer raw samples are incomplete or contain monitor failures")
    if file_sha256(directory / "driver.py") != run["driver_sha256"]:
        raise ValueError("observer driver differs from the executed source")
    indices = set()
    for sample in samples:
        matching = [
            line.split(",")[0].strip()
            for line in sample["gpu_utilization"].splitlines()
            if line.split(",")[1].strip().removeprefix("GPU-") == device_uuid.removeprefix("GPU-")
        ]
        if len(matching) != 1:
            raise ValueError("observer samples do not cover the execution GPU UUID")
        indices.update(matching)
    if len(indices) != 1:
        raise ValueError("observer GPU index changed during the measured process")
    report = {
        "run_id": run_id,
        "audit": audit,
        "run": run,
        "source_directory": str(directory),
        "gpu_uuid": device_uuid,
        "physical_gpu_index": int(indices.pop()),
        "files": {name: file_sha256(directory / name) for name in OBSERVER_FILES},
    }
    if reconciliation is not None:
        report["exit_race_reconciliation"] = reconciliation
    return report


def _read_observer_reconciliation(path):
    document = json.loads(Path(path).read_text())
    if (
        not isinstance(document, dict)
        or document.get("schema") != "cache-manager-observer-reconciliation-v1"
        or document.get("accepted") is not True
        or not isinstance(document.get("runs"), dict)
        or not document["runs"]
        or set(document["runs"]) - {"check", "bench", "profile"}
    ):
        raise ValueError("report requires an accepted observer reconciliation document")
    return document


def _verify_saved_consumers(profile, result, check_path, check, receipt, config):
    """Re-read saved MLA outputs against the receipt-bound resident evidence."""
    layers, queries = config["layers"], config["append"]
    selected = min(2048, config["history"] + queries)
    if type(layers) is not int or layers < 1 or type(queries) is not int or queries < 1:
        raise ValueError("consumer evidence requires positive layer and query counts")

    def load_evidence(directory, sample):
        record = sample["evidence"]
        name = record["path"]
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("consumer evidence must name a file inside its run directory")
        directory = Path(directory).resolve(strict=True)
        path = (directory / name).resolve(strict=True)
        if path.parent != directory or file_sha256(path) != record["sha256"]:
            raise ValueError("consumer evidence path or hash differs from its saved sample")
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(tensors, dict)
            or set(tensors) != {"indices", "attention"}
            or any(
                not isinstance(values, list) or len(values) != layers for values in tensors.values()
            )
        ):
            raise ValueError("consumer evidence omits or duplicates selection/attention layers")
        for indices, attention in zip(tensors["indices"], tensors["attention"], strict=True):
            if (
                not isinstance(indices, torch.Tensor)
                or indices.device.type != "cpu"
                or indices.dtype != torch.int32
                or indices.shape != (queries, selected)
                or not isinstance(attention, torch.Tensor)
                or attention.device.type != "cpu"
                or attention.dtype != torch.bfloat16
                or attention.ndim != 3
                or attention.shape[0] != queries
                or attention.shape[1] not in (64, 128)
                or attention.shape[2] != 512
            ):
                raise ValueError("saved selection or MLA output has an invalid shape or dtype")
            if not bool(torch.isfinite(attention).all()):
                raise ValueError("saved MLA output contains non-finite values")
        return path, tensors

    references = {}
    accepted_paths = {
        Path(path).resolve(strict=True) for path in receipt["artifact_paths"].values()
    }
    for sample in check["samples"]:
        if sample["scheme"] != "hbm":
            continue
        phase = sample["phase"]
        if phase not in MEASURED_PHASES or phase in references or sample["sample"] != 0:
            raise ValueError("check evidence requires one resident reference per measured phase")
        path, tensors = load_evidence(Path(check_path).parent, sample)
        if path not in accepted_paths:
            raise ValueError("resident consumer evidence is not bound by the acceptance receipt")
        references[phase] = tensors
    if set(references) != set(MEASURED_PHASES):
        raise ValueError("check evidence is missing a measured phase's resident reference")

    repeats = result["repeats"]
    if type(repeats) is not int or repeats < 1:
        raise ValueError("profile consumer evidence requires positive repeats")
    expected = {
        (scheme, phase, sample)
        for scheme in MEASURED_SCHEMES
        for phase in MEASURED_PHASES
        for sample in range(repeats)
    }
    seen, evidence_paths = set(), set()
    for sample in result["samples"]:
        key = sample["scheme"], sample["phase"], sample["sample"]
        if key not in expected or key in seen:
            raise ValueError("profile consumer evidence has an unexpected or duplicate sample")
        seen.add(key)
        path, tensors = load_evidence(profile, sample)
        if path in evidence_paths:
            raise ValueError(
                "different profile samples must retain separate consumer evidence files"
            )
        evidence_paths.add(path)
        reference = references[sample["phase"]]
        for field in ("indices", "attention"):
            for actual, expected_tensor in zip(tensors[field], reference[field], strict=True):
                if (
                    actual.shape != expected_tensor.shape
                    or actual.dtype != expected_tensor.dtype
                    or not torch.equal(actual, expected_tensor)
                ):
                    raise ValueError(
                        f"saved profile {field} differs from the resident check evidence"
                    )
    if seen != expected:
        raise ValueError("profile consumer evidence is missing measured samples")
    return {
        "resident_reference_files": len(references),
        "profile_evidence_files": len(evidence_paths),
        "profile_layer_comparisons": len(evidence_paths) * layers,
        "exact_selection_match": True,
        "exact_attention_match": True,
        "all_attention_finite": True,
    }


def generate(
    profile, receipt_path, output, *, observers, analysis=None, observer_reconciliation=None
):
    profile, output = (Path(path).resolve() for path in (profile, output))
    analysis = Path(analysis).resolve() if analysis else profile / "analysis"
    result = json.loads((profile / "result.json").read_text())
    identity = json.loads((profile / "identity.json").read_text())
    receipt = require_receipt(receipt_path, kind=KIND, identity=identity)
    if (
        result.get("schema") != KIND
        or result.get("mode") != "profile"
        or result.get("passed") is not True
        or result["identity_sha256"] != identity_digest(identity)
        or result["validation_receipt"] != receipt["receipt_sha256"]
    ):
        raise ValueError("report requires an accepted actual-MLA profile and matching receipt")
    check_path = Path(receipt["artifact_paths"]["result"])
    check = json.loads(check_path.read_text())
    if (
        check.get("mode") != "check"
        or check.get("passed") is not True
        or check["identity_sha256"] != result["identity_sha256"]
    ):
        raise ValueError("check/profile identities differ")
    for directory in (profile, check_path.parent):
        for relative, digest in identity["sources"].items():
            if file_sha256(directory / "source" / relative) != digest:
                raise ValueError("execution source snapshot differs from acceptance")
    diagnostics = json.loads((analysis / "summary.json").read_text())
    if diagnostics.get("schema") not in {
        "cache-manager-attention-transition-v2",
        "cache-manager-attention-transition-v3",
    }:
        raise ValueError("report requires actual top-k-to-MLA interval analysis")
    sources = diagnostics["sources"]
    if len(sources) != 1 or sources[0]["kind"] != "standalone_replay":
        raise ValueError("independent report requires this replay's own profile only")
    source = sources[0]
    for name, path in (
        ("result", profile / "result.json"),
        ("identity", profile / "identity.json"),
        ("capture", profile / "trace.sqlite"),
    ):
        if (analysis / source[name]["path"]).resolve() != path or file_sha256(path) != source[name][
            "sha256"
        ]:
            raise ValueError("analysis evidence does not bind this profile")
    expected = {
        (s["scheme"], s["phase"], s["sample"], layer)
        for s in result["samples"]
        for layer in range(identity["config"]["layers"])
    }
    actual = [(s["scheme"], s["phase"], s["sample"], s["layer"]) for s in diagnostics["samples"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("profile analysis omitted or duplicated a measured layer")
    for sample in diagnostics["samples"]:
        if sample["boundaries"]["endpoint_kind"] != "attention_kernel":
            raise ValueError("readiness is not an actual MLA endpoint")
        window = sample["gpu_topk_to_consumer"]
        if window["compute_union_ms"] != 0 or window["fused_union_ms"] != 0:
            raise ValueError("post-top-k window contains model computation")
        if (
            abs(
                window["io_union_ms"]
                + window["exposed_control_ms"]
                + window["gpu_idle_ms"]
                - window["window_ms"]
            )
            > 1e-12
        ):
            raise ValueError("post-top-k partition does not reconcile")
    for record in diagnostics["analysis_sources"].values():
        if file_sha256(analysis / record["archive"]) != record["sha256"]:
            raise ValueError("analysis helper archive hash differs")
    consumer_evidence = _verify_saved_consumers(
        profile, result, check_path, check, receipt, identity["config"]
    )
    reconciliations = (
        _read_observer_reconciliation(observer_reconciliation)["runs"]
        if observer_reconciliation
        else {}
    )
    observer_reports = {
        mode: read_observer(
            observers[mode],
            run["run_id"],
            identity["environment"]["uuid"],
            reconciliation=reconciliations.get(mode),
            result_path=path,
        )
        for mode, run, path in (
            ("check", check, check_path),
            ("profile", result, profile / "result.json"),
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in (
        "windows.csv",
        "stages.csv",
        "latency.csv",
        "results.md",
        "transition.svg",
        "transition.png",
    ):
        shutil.copyfile(analysis / name, output / name)
    shutil.copyfile(receipt_path, output / "acceptance_receipt.json")
    for mode, observer in observer_reports.items():
        target = output / "observers" / mode
        target.mkdir(parents=True)
        for name in observer["files"]:
            shutil.copyfile(Path(observers[mode]) / name, target / name)
    (output / "observer_summary.json").write_text(
        json.dumps(observer_reports, indent=2, sort_keys=True) + "\n"
    )
    audit = {
        "schema": "post-topk-profile-report-v2",
        "run_id": result["run_id"],
        "checks": {
            "matching_receipt": True,
            "source_snapshots_match": True,
            "raw_capture_bound": True,
            "all_layer_samples_present": True,
            "actual_mla_endpoints": True,
            "partitions_reconcile": True,
            "saved_consumer_outputs_verified": True,
            "observers_accepted": True,
        },
        "profile_run": str(profile),
        "analysis_directory": str(analysis),
        "analysis_sha256": file_sha256(analysis / "summary.json"),
        "profile_samples": len(result["samples"]),
        "layer_windows": len(actual),
        "consumer_evidence": consumer_evidence,
        "measurement": "intrusive NSYS top-k GPU completion to MLA GPU start; no indexer/top-k/MLA computation or whole-replay wall timing",
        "report_source_sha256": file_sha256(Path(__file__)),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", required=True, type=Path)
    parser.add_argument("--validation-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--check-observer", required=True, type=Path)
    parser.add_argument("--profile-observer", required=True, type=Path)
    parser.add_argument("--observer-reconciliation", type=Path)
    args = parser.parse_args(argv)
    result = generate(
        args.profile_run,
        args.validation_receipt,
        args.output_dir,
        observers={"check": args.check_observer, "profile": args.profile_observer},
        analysis=args.analysis_dir,
        observer_reconciliation=args.observer_reconciliation,
    )
    print(json.dumps(result["checks"], sort_keys=True))


if __name__ == "__main__":
    main()
