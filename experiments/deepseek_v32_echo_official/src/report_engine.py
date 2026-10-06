"""Audit and export selected Engine benchmark/timeline evidence without a GPU.

Run from the repository root with python -B -m
experiments.deepseek_v32_echo_official.src.report_engine --help.
Original HTTP publications and raw captures are never modified or copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
REPRO_ROOT = REPO_ROOT / "3rdparty/ECHO/reproduction/cxldsagr"
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPRO_ROOT))

from src.capacity import audit_capacity
from src.workload_client import canonical, load_workload, token_hash

from experiments.deepseek_v32_echo_official.scripts.source_identity import (
    acceptance_identity,
    source_differences,
    verify_source_archive,
)

CASES = {"hbm": "resident_reference", "echo": "echo"}
PHASES = ("prefill", "extend")
ANNOTATIONS = {"sample_index", "measured", "pair_name"}
CASE_ENV = {
    "NSA_KV_OFFLOAD",
    "NSA_DEV_CACHE_SIZE",
    "SGLANG_NSA_FUSE_LOGITS_RECALL_EXTEND",
    "SGLANG_NSA_FUSE_LOGITS_RECALL_DECODE",
}
WARMUP_PROTOCOL = {
    "id": "two_disjoint_prefix_pairs_v1",
    "pair_count": 2,
    "first_token_offsets": [1, 2],
    "requests_per_pair": ["prefill", "extend"],
    "timed": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_json(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Evidence:
    """Hash inspected evidence once and retain its exact local identity."""

    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        key = str(path)
        if key not in self.files:
            before = path.stat()
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            after = path.stat()
            require(
                (before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_ino, after.st_size, after.st_mtime_ns),
                f"Evidence changed while reading: {path}",
            )
            self.files[key] = {"sha256": digest, "bytes": after.st_size}
        item = {"path": key, **self.files[key]}
        if expected is not None:
            require(item["sha256"] == expected, f"Evidence SHA256 differs: {path}")
        return item

    def json(self, path, expected=None):
        self.bind(path, expected)
        return json.loads(Path(path).read_text())

    def jsonl(self, path, expected=None):
        self.bind(path, expected)
        return [json.loads(line) for line in Path(path).read_text().splitlines()]


def ownership_audit(registry, events, observations, release):
    """Validate recorded PID incarnations and ancestry, without querying live processes."""

    def key(identity):
        require(
            type(identity.get("pid")) is int
            and identity["pid"] > 0
            and type(identity.get("start_time_ticks")) is int
            and identity["start_time_ticks"] >= 0,
            "Invalid recorded process incarnation",
        )
        return identity["pid"], identity["start_time_ticks"]

    require(
        registry.get("schema") == "echo-engine-nsys-descendants-v1"
        and release.get("schema") == "echo-engine-nsys-gpu-release-v1",
        "NSYS ownership/release schema differs",
    )
    root = registry["root"]
    root_key = key(root)
    started = [event for event in events if event.get("event") == "started"]
    require(
        len(started) == 1
        and started[0].get("pid") == root["pid"]
        and started[0].get("pgid") == root.get("pgid")
        and release.get("root") == root,
        "NSYS ownership root differs from the launched process/release incarnation",
    )
    for event in events:
        if event.get("event") in {"exit", "closed"}:
            require(event.get("pid") == root["pid"], "Process exit belongs to another root PID")
    proofs = {}
    for proof in registry["descendants"]:
        identity = proof["identity"]
        process_key = key(identity)
        require(process_key not in proofs, "Duplicate descendant incarnation")
        chain = proof["chain"]
        require(
            chain and chain[0] == root and chain[-1] == identity,
            "Descendant chain does not connect its exact identity to the NSYS root",
        )
        chain_keys = [key(item) for item in chain]
        require(len({item[0] for item in chain_keys}) == len(chain), "Cyclic ancestry proof")
        for parent, child in pairwise(chain):
            require(
                child.get("ppid") == parent["pid"]
                and child["start_time_ticks"] >= parent["start_time_ticks"],
                "Recorded ancestry has an invalid parent PID or process incarnation order",
            )
        proofs[process_key] = proof
    require(root_key in proofs, "NSYS root is absent from its own ownership registry")
    for proof in proofs.values():
        for ancestor in proof["chain"]:
            require(
                key(ancestor) in proofs and proofs[key(ancestor)]["identity"] == ancestor,
                "Ancestry contains an unregistered or inconsistent process incarnation",
            )
    seen = {}
    proved_observations = stale_observations = 0
    for observation in observations:
        for process in observation["processes"]:
            pid = process["pid"]
            if process.get("exited_previously_verified_owned_pid") is True:
                require(pid in seen, "NVML exited-owner entry has no preceding ancestry proof")
                stale_observations += 1
                continue
            process_key = key(process)
            require(
                process_key in proofs
                and process.get("ownership_chain") == proofs[process_key]["chain"]
                and (pid not in seen or seen[pid] == process_key[1]),
                "GPU observation lacks matching ancestry or reuses a previously owned PID",
            )
            seen[pid] = process_key[1]
            proved_observations += 1
    recorded_owned = release["owned_process_start_time_ticks"]
    require(
        {str(pid): ticks for pid, ticks in seen.items()} == recorded_owned,
        "Release owner incarnations differ from proved GPU observations",
    )
    for observation in release["observations"]:
        for process in observation["processes"]:
            require(
                process.get("verified_exited_owner") is True and process["pid"] in seen,
                "GPU release contains an owner without preceding ancestry evidence",
            )
    for action in registry["cleanup_actions"]:
        if "signal" in action:
            require(
                key(action["identity"]) in proofs and key(action["identity"]) != root_key,
                "NSYS descendant cleanup signalled an unproved process incarnation",
            )
    return {
        "root": root,
        "registered_incarnations": len(proofs),
        "proved_gpu_observations": proved_observations,
        "exited_previously_proved_nvml_observations": stale_observations,
        "scope": "Stored parent chains and PID start times anchor GPU owners to the launched NSYS process. Later process-group changes/reparenting are allowed. Exited NVML entries reuse preceding ownership evidence; they have no fresh procfs identity.",
    }


def process_audit(path, evidence, expected_release=None):
    release = evidence.json(path / "gpu_release_drain.json", expected_release)
    require(
        release.get("status") == "complete"
        and release.get("result") == "gpu_observed_empty"
        and release["observations"][-1]["processes"] == [],
        f"GPU release was not confirmed: {path}",
    )
    events = evidence.jsonl(path / "process_events.jsonl")
    require(
        any(row.get("event") == "exit" and row.get("returncode") == 0 for row in events)
        and any(row.get("event") == "closed" and row.get("returncode") == 0 for row in events),
        f"Worker did not exit and close successfully: {path}",
    )
    observations = evidence.jsonl(path / "gpu_processes.jsonl")
    require(bool(observations), f"GPU observations are missing: {path}")
    require(
        all(
            row.get("query_status") == "complete" and row.get("foreign") == []
            for row in observations
        ),
        f"GPU observations contain a query failure or foreign process: {path}",
    )
    ownership_path = path / "process_ownership.json"
    ownership = None
    if ownership_path.exists() or release.get("schema") == "echo-engine-nsys-gpu-release-v1":
        registry = evidence.json(ownership_path)
        ownership = ownership_audit(registry, events, observations, release)
    memory = [
        int(process["used_memory_mib"])
        for row in observations
        for process in row["processes"]
        if str(process["used_memory_mib"]).isdigit()
    ]
    return {
        "observations": len(observations),
        "max_observed_process_memory_mib": max(memory, default=0),
        "release_confirmed": True,
        "descendant_ownership": ownership,
        "scope": "Discrete observations, including initialization; not a continuous memory peak or allocator budget audit.",
    }


def audit_in_engine_warmup(engine, metadata, complete, expected_inputs, evidence):
    """Verify four untimed shape-warmup requests in the measured Engine."""
    require(
        metadata.get("warmup_protocol") == WARMUP_PROTOCOL
        and complete.get("warmup_protocol") == WARMUP_PROTOCOL,
        "Pair does not declare the required same-Engine warmup protocol",
    )
    receipt = evidence.json(engine / "in_engine_warmup.json", complete["in_engine_warmup_sha256"])
    original = expected_inputs["extend"]
    require(
        receipt.get("schema") == "echo-engine-in-process-warmup-v1"
        and receipt.get("status") == "completed"
        and receipt.get("source_input_sha256") == token_hash(original)
        and receipt.get("source_first_token") == original[0]
        and receipt.get("warmup_protocol") == WARMUP_PROTOCOL
        and len(receipt.get("rows", [])) == 4,
        "Same-Engine warmup receipt has an incomplete or different workload",
    )
    require(
        type(receipt.get("started_ns")) is int
        and type(receipt.get("finished_ns")) is int
        and metadata["started_ns"]
        <= receipt["started_ns"]
        < receipt["finished_ns"]
        <= complete["finished_ns"],
        "Same-Engine warmup timestamps are outside the pair lifetime",
    )
    for row, (offset, phase) in zip(
        receipt["rows"], [(offset, phase) for offset in (1, 2) for phase in PHASES], strict=True
    ):
        ids = list(expected_inputs[phase])
        ids[0] = (ids[0] + offset) % 129280
        prompt, cached = (65536, 0) if phase == "prefill" else (65664, 65536)
        require(
            row.get("phase") == phase
            and row.get("first_token_offset") == offset
            and row.get("synthetic_first_token") == ids[0]
            and row.get("input_sha256") == token_hash(ids)
            and row.get("prompt_tokens") == prompt
            and row.get("cached_tokens") == cached
            and row.get("computed_prompt_tokens") == prompt - cached
            and row.get("completion_tokens") == 1
            and row.get("finish_reason") == {"type": "length", "length": 1}
            and len(row.get("output_ids", [])) == 1
            and type(row["output_ids"][0]) is int
            and 0 <= row["output_ids"][0] < 129280
            and row.get("raw_output_ids") == ids[-5:] + row["output_ids"]
            and type(row.get("wall_ms")) in (int, float)
            and math.isfinite(row["wall_ms"])
            and row["wall_ms"] > 0,
            "Warmup input mutation, cache boundary, output or diagnostic timing differs",
        )
        response = evidence.json(
            engine / f"warmup_{offset}_{phase}.response.json", row["response_sha256"]
        )
        require(
            response["output_ids"] == row["raw_output_ids"]
            and response["meta_info"]["prompt_tokens"] == prompt
            and response["meta_info"]["cached_tokens"] == cached
            and response["meta_info"]["completion_tokens"] == 1
            and response["meta_info"]["finish_reason"] == row["finish_reason"],
            "Saved warmup provider response differs from the receipt",
        )
    return {
        "protocol": WARMUP_PROTOCOL,
        "receipt": evidence.bind(engine / "in_engine_warmup.json"),
        "started_ns": receipt["started_ns"],
        "finished_ns": receipt["finished_ns"],
        "completed_requests": 4,
        "included_in_benchmark_samples": False,
    }


def audit_pair(run_path, record, pair, mode, expected_inputs, evidence):
    path = run_path / pair["name"]
    engine = path / "engine"
    require(pair["mode"] == mode, f"Pair mode differs: {path}")
    complete = evidence.json(engine / "complete.json", pair["completion"]["complete_sha256"])
    metadata = evidence.json(engine / "metadata.json")
    rows = evidence.jsonl(engine / "requests.jsonl", pair["completion"]["requests_sha256"])
    require(
        complete.get("schema") == "echo-sglang-engine-pair-v1"
        and complete.get("status") == "completed"
        and complete.get("completed") == 2
        and complete.get("mode") == mode
        and complete.get("case") == record["case"]
        and complete.get("numerical_acceptance") is False
        and complete.get("requests_sha256") == pair["completion"]["requests_sha256"]
        and [row.get("phase") for row in rows] == list(PHASES),
        f"Incomplete or inconsistent pair: {path}",
    )
    require(
        all(complete.get(key) == value for key, value in metadata.items()),
        f"Pair metadata changed: {path}",
    )
    require(
        metadata["engine_arguments"] == record["execution_identity"]["engine_arguments"],
        f"Engine arguments differ from the qualified identity: {path}",
    )
    warmup = audit_in_engine_warmup(engine, metadata, complete, expected_inputs, evidence)
    require(
        warmup["finished_ns"]
        <= rows[0]["started_ns"]
        < rows[0]["finished_ns"]
        <= rows[1]["started_ns"]
        < rows[1]["finished_ns"]
        <= complete["finished_ns"],
        "Formal request timestamps overlap warmup or disagree with pair order",
    )
    stored = [
        {key: value for key, value in row.items() if key not in ANNOTATIONS}
        for row in pair["completion"]["rows"]
    ]
    require(stored == rows, f"Pair summary differs from raw requests: {path}")
    for phase, row in zip(PHASES, rows, strict=True):
        prompt, cached, hidden_q = (65536, 0, 1024) if phase == "prefill" else (65664, 65536, 128)
        ids = expected_inputs[phase]
        require(
            row.get("prompt_tokens") == prompt
            and row.get("cached_tokens") == cached
            and row.get("computed_prompt_tokens") == prompt - cached
            and row.get("completion_tokens") == 1
            and row.get("finish_reason") == {"type": "length", "length": 1}
            and row.get("input_sha256") == token_hash(ids)
            and row.get("source_request_id") == 0
            and len(row.get("output_ids", [])) == 1
            and type(row["output_ids"][0]) is int
            and 0 <= row["output_ids"][0] < 129280
            and row.get("raw_output_ids") == ids[-5:] + row["output_ids"],
            f"Request/cache/input/output boundary differs: {path}/{phase}",
        )
        if mode == "check":
            artifact = row["output_artifact"]
            require(
                artifact.get("finite") is True
                and artifact.get("hidden_shape") == [hidden_q, 7168]
                and artifact.get("logprobs_shape") == [129280],
                f"Independent output shape/finiteness receipt differs: {path}/{phase}",
            )
            require(
                Path(artifact["path"]).resolve() == (engine / f"{phase}.npz").resolve(),
                "Check tensor path differs",
            )
            evidence.bind(artifact["path"], artifact["sha256"])
        else:
            require(
                type(row.get("wall_ms")) in (float, int)
                and math.isfinite(row["wall_ms"])
                and row["wall_ms"] > 0,
                f"Invalid raw timing sample: {path}/{phase}",
            )
            response = evidence.json(engine / f"{phase}.response.json", row["response_sha256"])
            require(
                response["output_ids"] == row["raw_output_ids"]
                and response["meta_info"]["cached_tokens"] == cached
                and response["meta_info"]["prompt_tokens"] == prompt,
                f"Saved provider response differs: {path}/{phase}",
            )
    capacity = evidence.json(path / "capacity.json")
    require(
        capacity == pair["capacity"] and capacity.get("passed") is True,
        f"Capacity receipt differs: {path}",
    )
    logs = {}
    for filename, digest in capacity["log_sha256"].items():
        evidence.bind(filename)
        logs[filename] = Path(filename).read_text()
        require(
            hashlib.sha256(logs[filename].encode()).hexdigest() == digest,
            f"Allocation log changed: {filename}",
        )
    info = evidence.json(engine / "server_info.json")
    require(
        audit_capacity(info, logs, case=record["case"]) == capacity,
        f"Capacity audit differs: {path}",
    )
    evidence.json(path / "jit_artifacts.json", pair["jit_manifest"]["sha256"])
    monitoring = process_audit(path, evidence, pair["gpu_release_sha256"])
    monitoring["in_engine_warmup"] = warmup
    return rows, monitoring


def audit_run(path, case, mode, evidence):
    path = path.resolve(strict=True)
    record = evidence.json(path / "run.json")
    require(
        record.get("schema") == "echo-sglang-engine-run-v1"
        and record.get("run_id") == path.name
        and record.get("mode") == mode
        and record.get("case") == CASES[case]
        and record.get("status") == "completed"
        and record.get("exitcode") == 0
        and record.get("numerical_acceptance") is False
        and record.get("user_requested_performance_only") is True
        and record.get("structure_check_passed") is True,
        f"Run is not a completed performance-only {case}/{mode}: {path}",
    )
    identity = record["execution_identity"]
    require(
        digest_json(identity) == record["execution_identity_sha256"],
        f"Execution identity signature differs: {path}",
    )
    require(
        record.get("warmup_protocol") == WARMUP_PROTOCOL
        and identity.get("warmup_protocol") == WARMUP_PROTOCOL,
        f"Run/source identity lacks the required same-Engine warmup protocol: {path}",
    )
    preflight = evidence.json(path / "preflight.json")
    require(
        preflight == identity["preflight"] and preflight.get("runtime_validated") is True,
        f"Preflight differs: {path}",
    )
    require(
        record["reproduction_source_sha256"] == identity["reproduction_source_sha256"],
        "Recorded source lists differ",
    )
    archive = verify_source_archive(path, record)
    evidence.bind(archive["manifest"]["path"], archive["manifest"]["sha256"])
    for source in archive["sources"].values():
        evidence.bind(source["absolute_path"], source["sha256"])
    compatibility = acceptance_identity(identity)
    if "acceptance_identity" in record or "acceptance_identity_sha256" in record:
        require(
            record.get("acceptance_identity") == compatibility
            and record.get("acceptance_identity_sha256") == digest_json(compatibility),
            f"Recorded check compatibility key/rule differs: {path}",
        )
    diagnostic = evidence.json(
        record["numerical_diagnostic"]["path"], record["numerical_diagnostic"]["sha256"]
    )
    require(
        diagnostic.get("passed") is False
        and diagnostic.get("performance_receipt") is False
        and diagnostic["signature"]
        == digest_json({key: value for key, value in diagnostic.items() if key != "signature"}),
        "Retained failed numerical diagnostic changed",
    )
    manifest, inputs = load_workload(Path(record["arguments"]["workload"]))
    require(manifest["workload_sha256"] == record["workload_sha256"], "Workload identity differs")
    expected_inputs = {"prefill": inputs[0]["input_ids"][:65536], "extend": inputs[0]["input_ids"]}
    expected_names = [f"pair_{index:02d}" for index in range(5)] if mode == "bench" else [mode]
    require(
        [pair["name"] for pair in record["pairs"]] == expected_names,
        "Missing, reordered or duplicate pairs",
    )
    monitoring = {"preflight": process_audit(path / "preflight", evidence)}
    raw_pairs, samples = {}, []
    pairs = [(pair, mode) for pair in record["pairs"]]
    if mode != "check":
        require(
            record.get("warmup", {}).get("name") == "warmup", "Independent warmup pair is missing"
        )
        pairs.insert(0, (record["warmup"], "warmup"))
    for pair, pair_mode in pairs:
        rows, monitoring[pair["name"]] = audit_pair(
            path, record, pair, pair_mode, expected_inputs, evidence
        )
        raw_pairs[pair["name"]] = rows
        if mode == "bench":
            index = -1 if pair_mode == "warmup" else int(pair["name"].removeprefix("pair_"))
            for row in rows:
                measured = index >= 0 and (row["phase"] == "extend" or index < 3)
                samples.append(
                    {
                        "case": case,
                        "run_id": record["run_id"],
                        "pair": pair["name"],
                        "sample_index": index,
                        "phase": row["phase"],
                        "measured": measured,
                        "role": "measured" if measured else "warmup" if index < 0 else "setup",
                        "wall_ms": row["wall_ms"],
                        "prompt_tokens": row["prompt_tokens"],
                        "cached_tokens": row["cached_tokens"],
                        "computed_prompt_tokens": row["computed_prompt_tokens"],
                        "output_token": row["output_ids"][0],
                        "input_sha256": row["input_sha256"],
                        "requests_sha256": pair["completion"]["requests_sha256"],
                    }
                )
    if mode == "bench":
        counts = {
            phase: sum(row["measured"] and row["phase"] == phase for row in samples)
            for phase in PHASES
        }
        require(
            counts == {"prefill": 3, "extend": 5} == record["sample_counts"],
            "Benchmark sample counts differ",
        )
        saved = evidence.json(path / "timing_samples.json", record["timing_samples_sha256"])
        expected = [
            {**row, "sample_index": index, "measured": True, "pair_name": f"pair_{index:02d}"}
            for index in range(5)
            for row in raw_pairs[f"pair_{index:02d}"]
            if row["phase"] == "extend" or index < 3
        ]
        require(saved == expected, "Benchmark summary does not match recomputed raw samples")
    if mode == "profile":
        capture = record["pairs"][0]["nsys_report"]
        actual = evidence.bind(capture["path"], capture["sha256"])
        require(actual["bytes"] == capture["bytes"] > 0, "Profile capture size differs")
    return {
        "path": path,
        "record": record,
        "samples": samples,
        "monitoring": monitoring,
        "source_archive": archive,
        "acceptance_identity": compatibility,
    }


def audit_triplet(runs, case, evidence):
    check = runs["check"]
    identity = check["record"]["execution_identity"]
    compatibility = check["acceptance_identity"]
    for mode in ("bench", "profile"):
        record = runs[mode]["record"]
        require(
            runs[mode]["acceptance_identity"] == compatibility,
            f"{case}/{mode} worker/runtime compatibility key differs from check",
        )
        audit = record["check_audit"]
        require(
            Path(audit["path"]).resolve() == check["path"] / "run.json"
            and audit["run_id"] == check["record"]["run_id"]
            and audit["structure_check_passed"] is True
            and audit["numerical_acceptance"] is False,
            f"{case}/{mode} did not bind this independent check",
        )
        evidence.bind(audit["path"], audit["sha256"])
        if record["execution_identity"] != identity or "acceptance_identity_sha256" in audit:
            differences = {
                name: {"check": values["check"], "run": values["current"]}
                for name, values in source_differences(
                    identity, record["execution_identity"]
                ).items()
            }
            require(
                audit.get("acceptance_identity_sha256") == digest_json(compatibility)
                and audit.get("comparison_rule") == compatibility["comparison_rule"]
                and audit.get("check_execution_identity_sha256")
                == check["record"]["execution_identity_sha256"]
                and audit.get("orchestration_source_differences") == differences
                and audit.get("source_archive") == check["source_archive"],
                f"{case}/{mode} did not record the full check/source-compatibility comparison",
            )


def profile_snapshots(run, evidence):
    path = run["path"] / "profile/profile_hooks"
    case = "hbm" if run["record"]["case"] == "resident_reference" else "echo"
    manifest = evidence.json(path / "hook_manifest.json")
    source = run["record"]["reproduction_source_sha256"]
    hook_key = "experiments/deepseek_v32_echo_official/src/engine_profile_hooks.py"
    require(
        manifest.get("schema") == "echo-engine-nvtx-hooks-v1"
        and manifest.get("case") == case
        and manifest.get("numerical_acceptance") is False
        and manifest.get("snapshots") == list(PHASES)
        and manifest["hook_source"]["sha256"] == source[hook_key],
        "Profile hooks/phase snapshots do not match the execution identity",
    )
    begin = evidence.json(path / "formal_begin.json")
    require(
        begin.get("schema") == "echo-engine-formal-profile-begin-v1"
        and begin.get("record_count_before") == 0
        and begin.get("first_forward_id") == 0
        and begin.get("cache_mutated") is False
        and manifest.get("recording_enabled") is True
        and manifest.get("formal_begin") == begin,
        "Profile recording did not begin with empty bookkeeping after same-Engine warmup",
    )
    forwards = evidence.json(path / "forwards.json")
    require(
        len(forwards) == manifest["forward_count"] == 65,
        "Formal profile must contain exactly 64 history chunks and one extend forward",
    )
    for index, forward in enumerate(forwards):
        phase, q, start, end = (
            ("prefill", 1024, index * 1024, (index + 1) * 1024)
            if index < 64
            else ("extend", 128, 65536, 65664)
        )
        require(
            forward.get("id") == index
            and forward.get("completed") is True
            and forward.get("batch_size") == 1
            and forward.get("mode") == "EXTEND"
            and forward.get("phase") == phase
            and forward.get("layers") == [0, 1, 2]
            and (forward.get("q"), forward.get("start"), forward.get("end")) == (q, start, end)
            and forward.get("query_lens") == [q]
            and forward.get("prefix_lens") == [start]
            and forward.get("seq_lens") == [end],
            f"Formal profile forward {index} has incomplete or noncontiguous geometry",
        )
    result = {
        "formal_profile": {
            "begin_receipt": evidence.bind(path / "formal_begin.json"),
            "forward_receipt": evidence.bind(path / "forwards.json"),
            "completed_forwards": 65,
            "prefill_chunks": 64,
            "extend_forwards": 1,
            "synthetic_warmup_recorded": False,
        }
    }
    for phase in PHASES:
        snapshot = evidence.json(path / f"snapshot_{phase}.json")
        require(
            snapshot.get("schema") == "echo-engine-post-request-residency-v1"
            and snapshot["phase"] == phase
            and snapshot["forward_id"] == (63 if phase == "prefill" else 64)
            and snapshot["history_tokens"] == 65536
            and snapshot["sequence_tokens"] == (65536 if phase == "prefill" else 65664)
            and [row["layer"] for row in snapshot["layers"]] == [0, 1, 2],
            "Profile residency snapshot has the wrong phase/shape/layers",
        )
        candidate_tokens = 0 if phase == "prefill" else 128
        for layer in snapshot["layers"]:
            history = layer["history_resident_tokens"]
            candidate = layer["candidate_resident_tokens"]
            require(
                type(history) is int
                and 0 <= history <= 65536
                and type(candidate) is int
                and 0 <= candidate <= candidate_tokens
                and layer["all_history_resident"] == (history == 65536)
                and layer["all_sequence_resident"]
                == (history == 65536 and candidate == candidate_tokens),
                "Profile residency counts and flags disagree",
            )
        require(
            snapshot["device_pool_tokens"] == (131072 if case == "hbm" else 65664)
            and (case != "echo" or snapshot["host_pool_tokens"] == 16777216),
            "Profile residency pool capacity differs",
        )
        matching = [row for row in forwards if row["id"] == snapshot["forward_id"]]
        require(
            len(matching) == 1
            and matching[0]["completed"]
            and matching[0]["phase"] == phase
            and matching[0]["layers"] == [0, 1, 2],
            "Profile snapshot lacks a completed matching forward",
        )
        result[phase] = snapshot
    return result


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path, evidence):
    evidence.bind(path)
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def audit_timeline(path, runs, evidence):
    summary = evidence.json(path / "timeline_summary.json")
    require(summary.get("schema") == "echo-sglang-engine-timeline-v1", "Unexpected timeline schema")
    require(
        summary["selection"]["history"] == 65536
        and summary["selection"]["candidate"] == 128
        and summary["selection"]["chunk"] == 1024,
        "Timeline input selection differs",
    )
    evidence.bind(summary["generator"]["path"], summary["generator"]["sha256"])
    for case in CASES:
        source = summary["sources"][case]
        bench = runs[case]["bench"]
        require(
            Path(source["benchmark"]["path"]).resolve() == bench["path"] / "run.json",
            "Timeline benchmark path differs",
        )
        require(
            source["benchmark"]["metadata"] == bench["record"],
            "Timeline benchmark metadata differs",
        )
        evidence.bind(source["benchmark"]["path"], source["benchmark"]["sha256"])
        sqlite = Path(source["capture"]["path"]).resolve(strict=True)
        evidence.bind(sqlite, source["capture"]["sha256"])
        report = Path(runs[case]["profile"]["record"]["pairs"][0]["nsys_report"]["path"]).resolve()
        require(
            sqlite == report.with_suffix(".sqlite"),
            "Timeline SQLite is not beside the supplied profile capture",
        )
        receipt_path = sqlite.with_suffix(".export.json")
        receipt = evidence.json(receipt_path)
        require(
            receipt.get("schema") == "echo-engine-nsys-export-v1" and receipt.get("exitcode") == 0,
            "NSYS SQLite export did not succeed",
        )
        for field, expected_path in (("input", report), ("output", sqlite)):
            require(
                Path(receipt[field]["path"]).resolve() == expected_path,
                "NSYS export provenance path differs",
            )
            evidence.bind(expected_path, receipt[field]["sha256"])
        require("export" in receipt.get("command", []), "NSYS export command is missing")
        evidence.bind(ROOT / "scripts/export_nsys.py", receipt["exporter_sha256"])
        evidence.bind(sqlite.with_suffix(".export.stdout.log"), receipt["stdout_sha256"])
        evidence.bind(sqlite.with_suffix(".export.stderr.log"), receipt["stderr_sha256"])
    windows = read_csv(path / "windows.csv", evidence)
    require(
        {(row["case"], row["phase"]) for row in windows}
        == {(case, phase) for case in CASES for phase in PHASES}
        and len(windows) == 4,
        "Timeline windows are incomplete",
    )
    for row in windows:
        panel = summary["panels"][row["case"]][row["phase"]]
        require(
            [layer["layer"] for layer in panel["layers"]] == [0, 1, 2],
            "Timeline layer coverage differs",
        )
        window = panel["window"]
        require(
            int(row["start_ns"]) == window["start_ns"] == panel["layers"][0]["start_ns"]
            and int(row["end_ns"]) == window["end_ns"] == panel["layers"][2]["end_ns"]
            and math.isclose(
                float(row["window_ms"]),
                (window["end_ns"] - window["start_ns"]) / 1e6,
                abs_tol=1e-12,
            )
            and math.isclose(
                window["gpu_busy_ms"] + window["gpu_idle_ms"], window["window_ms"], abs_tol=1e-9
            ),
            "Timeline window boundaries/occupancy do not conserve",
        )
        query = panel["query"]
        expected = (1024, 64512, 65536) if row["phase"] == "prefill" else (128, 65536, 65664)
        require(
            (query["q"], query["start"], query["end"]) == expected,
            "Timeline query selection differs",
        )
        require(
            panel["prefill_chunks"] == (64 if row["phase"] == "prefill" else None),
            "Timeline history chunk coverage differs",
        )
    selected = []
    for activity in read_csv(path / "activities.csv", evidence):
        for phase in PHASES:
            panel = summary["panels"][activity["case"]][phase]
            if str(panel["device"]) != activity["device"]:
                continue
            left = max(int(activity["start_ns"]), panel["window"]["start_ns"])
            right = min(int(activity["end_ns"]), panel["window"]["end_ns"])
            if left < right:
                selected.append(
                    {
                        **activity,
                        "selected_phase": phase,
                        "clipped_start_ns": left,
                        "clipped_end_ns": right,
                        "clipped_duration_ns": right - left,
                    }
                )
    require(bool(selected), "Timeline selected activity table is empty")
    for case in CASES:
        for phase in PHASES:
            require(
                sum(row["case"] == case and row["selected_phase"] == phase for row in selected)
                == summary["panels"][case][phase]["window"]["activity_count"],
                "Selected activity count differs from timeline window",
            )
    return summary, windows, selected


def export(args):
    output = args.output_dir.resolve()
    require(
        output.is_relative_to((ROOT / "output").resolve()),
        "Initial report export must be inside experiment output",
    )
    require(not output.exists(), f"Output directory already exists: {output}")
    evidence = Evidence()
    runs = {
        case: {
            mode: audit_run(getattr(args, f"{case}_{mode}"), case, mode, evidence)
            for mode in ("check", "bench", "profile")
        }
        for case in CASES
    }
    for case in CASES:
        audit_triplet(runs[case], case, evidence)
    left, right = (
        runs[case]["check"]["acceptance_identity"]["execution_identity"] for case in CASES
    )
    for field in (
        "preflight",
        "reproduction_source_sha256",
        "workload_sha256",
        "source_input_sha256",
        "numerical_diagnostic_sha256",
    ):
        require(left[field] == right[field], f"HBM/ECHO common identity differs: {field}")
    require(
        {key: value for key, value in left["engine_arguments"].items() if key != "max_total_tokens"}
        == {
            key: value
            for key, value in right["engine_arguments"].items()
            if key != "max_total_tokens"
        },
        "HBM/ECHO non-capacity Engine settings differ",
    )
    require(
        {
            key: value
            for key, value in left["execution_environment_values"].items()
            if key not in CASE_ENV
        }
        == {
            key: value
            for key, value in right["execution_environment_values"].items()
            if key not in CASE_ENV
        },
        "HBM/ECHO non-policy environment differs",
    )
    timeline, windows, selected = audit_timeline(
        args.timeline_dir.resolve(strict=True), runs, evidence
    )
    snapshots = {case: profile_snapshots(runs[case]["profile"], evidence) for case in CASES}
    for case in CASES:
        for phase in PHASES:
            require(
                timeline["panels"][case][phase]["forward_id"]
                == snapshots[case][phase]["forward_id"],
                "Timeline and post-request snapshot refer to different forwards",
            )
    samples = [row for case in CASES for row in runs[case]["bench"]["samples"]]
    statistics_rows = []
    for case in CASES:
        for phase in PHASES:
            values = [
                row["wall_ms"]
                for row in samples
                if row["case"] == case and row["phase"] == phase and row["measured"]
            ]
            statistics_rows.append(
                {
                    "case": case,
                    "phase": phase,
                    "samples": len(values),
                    "median_ms": statistics.median(values),
                    "mean_ms": statistics.mean(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
            )
    runtime = left["preflight"]["runtime"]
    report = {
        "schema": "echo-sglang-engine-publication-v1",
        "numerical_acceptance": False,
        "structure_and_artifact_audit_passed": True,
        "numerical_scope": "Independent finite-output/shape and cache/capacity checks pass; retained official cross-run numerical acceptance remains failed.",
        "benchmark_statistics": statistics_rows,
        "benchmark_sample_source": "Recomputed from raw per-pair requests.jsonl: three prefill and five extend samples per case. Excludes every synthetic same-Engine warmup request, the separate outer warmup pair, and two setup-only prefills per case.",
        "workload": {
            "history": 65536,
            "candidate": 128,
            "history_chunk": 1024,
            "checkpoint_layers": [0, 1, 2],
            "source_request_id": 0,
            "workload_sha256": left["workload_sha256"],
            "input_sha256": left["source_input_sha256"],
            "warmup_protocol": WARMUP_PROTOCOL,
            "formal_cache_state": "Warmed Engine, logical prefix miss with occupied allocator and normal official eviction. Formal H has cached_tokens=0; original H+A reuses exactly H with cached_tokens=65536. No cache flush or manual reset.",
        },
        "hardware_runtime": {
            key: runtime[key]
            for key in (
                "devices",
                "placement",
                "python",
                "python_version",
                "torch_cuda",
                "versions",
                "native",
                "native_build_receipt",
            )
        },
        "source": {
            "official": left["preflight"]["source"],
            "acceptance_core_files": left["reproduction_source_sha256"],
            "acceptance_comparison_rule": runs["hbm"]["check"]["acceptance_identity"][
                "comparison_rule"
            ],
            "archive_verification": "Each full source set is verified against its per-run byte archive and original run.json, without comparing historical launcher bytes to live files.",
            "generator": evidence.bind(__file__),
        },
        "runs": {
            case: {
                mode: {
                    "run_id": run["record"]["run_id"],
                    "run_file": evidence.bind(run["path"] / "run.json"),
                    "execution_identity_sha256": run["record"]["execution_identity_sha256"],
                    "acceptance_identity_sha256": digest_json(run["acceptance_identity"]),
                    "recorded_source_sha256": run["record"]["reproduction_source_sha256"],
                    "source_archive": run["source_archive"],
                    "monitoring": run["monitoring"],
                }
                for mode, run in phases.items()
            }
            for case, phases in runs.items()
        },
        "capacities": {
            case: runs[case]["bench"]["record"]["pairs"][0]["capacity"] for case in CASES
        },
        "timeline": {
            "selection": timeline["selection"],
            "notes": timeline["notes"],
            "panels": {
                case: {
                    phase: {
                        key: (
                            {
                                name: item
                                for name, item in value.items()
                                if name != "gpu_idle_intervals_ns"
                            }
                            if isinstance(value, dict)
                            else value
                        )
                        for key, value in panel.items()
                        if key != "forwards"
                    }
                    for phase, panel in phases.items()
                }
                for case, phases in timeline["panels"].items()
            },
            "summary_source": evidence.bind(args.timeline_dir / "timeline_summary.json"),
        },
        "residency_snapshots": snapshots,
        "limitations": [
            "Engine.generate excludes HTTP but includes scheduler/IPC, model execution, sampling and detokenization; differing HTTP and Engine workloads cannot isolate HTTP overhead by subtraction.",
            "Each fresh Engine first runs two ordinary synthetic H->H+A warmup pairs, changing only token 0 by +1 and +2 modulo 129280. These four calls are excluded from formal timing and profile. A separate outer warmup Engine is also retained. Formal prefill starts with an occupied allocator and a logical prefix miss; normal official eviction remains in the measured request.",
            "Extend reuses exactly H with post-prefill residency retained. No cold reset or cache flush; this differs from local MFU cold offload measurements.",
            "HBM-only has 131072 device tokens; ECHO has 65664 device and 16777216 host tokens. These are not equal-capacity or equal-byte-budget measurements.",
            "Three original checkpoint layers only. No full-model or multiuser capacity claim; official radix retains candidates.",
            "NSYS three-layer windows are intrusive profile observations and exclude work outside L0 first compute through L2 last compute. Full Engine benchmarks are separate.",
            "Mapped-host fused prefetch/recall kernels may transfer zero records. Snapshot residency and final residual-recall counters do not determine total per-kernel H2D bytes.",
            "The exporter verifies check tensor hashes and their finite-output receipts; it does not perform a new cross-run numerical equivalence test.",
            "Check compatibility compares the exact worker, execution helpers, upstream/native/checkpoint identities, inputs, configuration, hardware and recorded environment. Launcher, observer and profile-hook sources remain fully archived provenance but are excluded by an explicit rule from this compatibility key; their versions and timing boundaries may differ.",
        ],
        "evidence_files": evidence.files,
    }
    output.mkdir(parents=True)
    for phase in PHASES:
        for suffix in ("svg", "png"):
            name = f"timeline_{phase}.{suffix}"
            evidence.bind(args.timeline_dir / name)
            shutil.copyfile(args.timeline_dir / name, output / name)
    write_csv(output / "windows.csv", windows)
    write_csv(output / "benchmark_samples.csv", samples)
    write_csv(output / "benchmark_statistics.csv", statistics_rows)
    write_csv(output / "selected_activities.csv", selected)
    with (output / "report.json").open("x") as target:
        json.dump(report, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
    publication = {
        "schema": "echo-sglang-engine-publication-manifest-v1",
        "numerical_acceptance": False,
        "generator": evidence.bind(__file__),
        "files": {
            path.name: {key: value for key, value in evidence.bind(path).items() if key != "path"}
            for path in sorted(output.iterdir())
        },
    }
    publication["signature"] = digest_json(publication)
    with (output / "publication.json").open("x") as target:
        json.dump(publication, target, indent=2, sort_keys=True)
        target.write("\n")
    print(
        json.dumps(
            {"output": str(output), "benchmarks": statistics_rows, "numerical_acceptance": False},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for case in CASES:
        for mode in ("check", "bench", "profile"):
            parser.add_argument(f"--{case}-{mode}", type=Path, required=True)
    parser.add_argument("--timeline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
