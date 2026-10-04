"""Build independent HBM outputs for a completed fixed-capacity ECHO probe.

Users execute one at a time, each from an empty resident cache. Visits retain
their order within each user. This reordered schedule validates numerical
outputs only; it is not a serving latency or resident-user capacity comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import torch

from experiments.deepseek_v32_echo_cache.src.capacity_probe import (
    REFERENCE_SCHEMA,
    check_lifecycle,
    reference_binding,
    tensor_summary,
    write_json,
)
from GR.workload import token_sha256


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_recorded_source(run_dir, state):
    from evaluation.provenance import verify_source_snapshot

    manifest = json.loads((run_dir / "source_manifest.json").read_text())
    aggregate = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if aggregate != state["source_sha256"]:
        raise AssertionError("ECHO source manifest aggregate differs")
    for name, expected in manifest.items():
        if digest(run_dir / "source" / name) != expected:
            raise AssertionError(f"ECHO archived source differs: {name}")
    # This checks every recorded file against the current tree. New unrelated
    # files are deliberately outside the original run's source identity.
    verify_source_snapshot(run_dir)


def load_run(run_dir):
    state = json.loads((run_dir / "status.json").read_text())
    if state.get("status") != "complete" or not state.get("capacity_passed"):
        raise ValueError("reference generation requires a completed ECHO capacity run")
    config = state["config"]
    if config["scheme"] != "echo" or config["num_layers"] != 10 or config["chunk_size"] != 1024:
        raise ValueError("reference scope requires the ten-block ECHO probe with C=1024")
    verify_recorded_source(run_dir, state)
    manifest = json.loads((run_dir / "numerical/manifest.json").read_text())
    if (
        manifest.get("schema") != REFERENCE_SCHEMA
        or manifest.get("scheme") != "echo"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("ECHO numerical manifest is incomplete")
    workload = json.loads((run_dir / "workload/workload.json").read_text())
    identity = {
        key: workload[key] for key in ("config", "heat_sha256", "tokenizer_sha256", "requests")
    }
    workload_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if (
        workload_digest != workload["workload_sha256"]
        or workload_digest != state["workload_sha256"]
    ):
        raise AssertionError("ECHO workload manifest identity differs")
    binding = reference_binding(
        config, state["source_sha256"], state["checkpoint"], workload["tokenizer_sha256"]
    )
    if manifest["binding"] != binding:
        raise AssertionError("ECHO numerical binding differs from its workload and source")
    requests = [
        json.loads(line) for line in (run_dir / "workload/requests.jsonl").read_text().splitlines()
    ]
    results = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
    if len(requests) != config["requests"] or len(results) != len(requests):
        raise AssertionError("ECHO workload or numerical output count is incomplete")
    if len(workload["requests"]) != len(requests):
        raise AssertionError("ECHO request identity count differs")
    groups = {}
    prefixes = {}
    history, candidate = config["history_tokens"], config["candidate_tokens"]
    for index, (request, recorded, result) in enumerate(
        zip(requests, workload["requests"], results, strict=True)
    ):
        if any(request.get(key) != value for key, value in recorded.items()):
            raise AssertionError(f"request {index}: differs from the workload identity manifest")
        check_lifecycle(request, result["metrics"], config, index)
        ids = request["input_ids"]
        if len(ids) != history + candidate:
            raise AssertionError(f"request {index}: token count differs")
        for name, tokens in (
            ("prefix_sha256", ids[:history]),
            ("candidate_sha256", ids[history:]),
            ("input_sha256", ids),
        ):
            if request[name] != token_sha256(tokens):
                raise AssertionError(f"request {index}: {name} differs")
        if result["request_id"] != index or result["input_sha256"] != request["input_sha256"]:
            raise AssertionError(f"request {index}: ECHO output input identity differs")
        user = request["user_id"]
        prefix = tuple(ids[:history])
        if prefixes.setdefault(user, prefix) != prefix:
            raise AssertionError(f"user {user}: prefix changed across visits")
        visits = groups.setdefault(user, [])
        if request["visit_index"] != len(visits):
            raise AssertionError(f"user {user}: visit order differs")
        visits.append(request)
    if len(groups) != config["users"] or any(
        len(rows) != config["rounds"] for rows in groups.values()
    ):
        raise AssertionError("workload does not contain all complete user traversals")
    return state, binding, groups, results


def compare_echo(run_dir, result, payload):
    index = payload["request_id"]
    if result["output_file"] != f"numerical/{index:06d}.pt":
        raise AssertionError(f"request {index}: ECHO output path differs")
    path = run_dir / result["output_file"]
    if digest(path) != result["output_file_sha256"]:
        raise AssertionError(f"request {index}: ECHO output file digest differs")
    expected = torch.load(path, map_location="cpu", weights_only=True)
    if expected["request_id"] != index or expected["input_sha256"] != payload["input_sha256"]:
        raise AssertionError(f"request {index}: ECHO numerical input identity differs")
    comparison = {"status": "passed", "atol": 0, "rtol": 0, "echo_output_sha256": digest(path)}
    for name in ("hidden", "logits"):
        reference, reference_summary = tensor_summary(payload[name])
        actual, actual_summary = tensor_summary(expected[name])
        if actual_summary != result[name]:
            raise AssertionError(f"request {index}: ECHO {name} summary differs")
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        if actual_summary != reference_summary:
            raise AssertionError(f"request {index}: {name} is not bitwise identical")
        comparison[name] = {"exact": True, "bitwise_equal": True, "max_abs": 0.0}
    return comparison


def run_users(backend, run_dir, output, state, groups, results, metadata):
    config = state["config"]
    history, candidate = config["history_tokens"], config["candidate_tokens"]
    with (output / "requests.jsonl").open("x") as stream:
        for user, requests in groups.items():
            metadata.update(stage="create_empty_session", active_user_id=user)
            write_json(output / "status.json", metadata)
            session = backend.create_session(history + candidate)
            try:
                if session.length != 0:
                    raise AssertionError("HBM reference session must start empty")
                metadata["stage"] = "history_prefill"
                write_json(output / "status.json", metadata)
                backend.prefill(session, requests[0]["input_ids"][:history])
                if session.length != history:
                    raise AssertionError("HBM reference history was not fully committed")
                for request in requests:
                    index = request["request_id"]
                    metadata.update(stage="candidate_extend", active_request_id=index)
                    write_json(output / "status.json", metadata)
                    hidden, hidden_summary = tensor_summary(
                        backend.extend(session, request["input_ids"][history:])
                    )
                    logits, logits_summary = tensor_summary(backend.last_logits)
                    if hidden.shape != (candidate, backend.cfg.dim):
                        raise AssertionError("HBM output must include all candidate hidden states")
                    if logits.shape != (1, backend.cfg.vocab_size):
                        raise AssertionError("HBM output must include complete last-token logits")
                    if session.length != history + candidate:
                        raise AssertionError("HBM candidate suffix was not fully committed")
                    payload = {
                        "request_id": index,
                        "input_sha256": request["input_sha256"],
                        "hidden": hidden,
                        "logits": logits,
                    }
                    path = output / f"{index:06d}.pt"
                    torch.save(payload, path)
                    metadata["stage"] = "numerical_comparison"
                    comparison = compare_echo(run_dir, results[index], payload)
                    backend.truncate(session, history)
                    if session.length != history:
                        raise AssertionError("HBM session was not restored to its fixed history")
                    stream.write(
                        json.dumps(
                            {
                                "request_id": index,
                                "user_id": user,
                                "visit_index": request["visit_index"],
                                "execution_index": metadata["completed_requests"],
                                "input_sha256": request["input_sha256"],
                                "hidden": hidden_summary,
                                "logits": logits_summary,
                                "comparison": comparison,
                                "output_file": path.name,
                                "output_file_sha256": digest(path),
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                    metadata["completed_requests"] += 1
                    print(
                        f"HBM request {index}: user={user} visit={request['visit_index']} "
                        f"bitwise_equal=True completed={metadata['completed_requests']}/{config['requests']}",
                        flush=True,
                    )
            finally:
                backend.release_session(session)
            metadata["completed_users"] += 1
    if metadata["completed_requests"] != config["requests"]:
        raise AssertionError("HBM reference did not execute every saved request")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", help="default: device recorded by the ECHO run")
    result.add_argument("--model-path", type=Path, help="default: recorded checkpoint path")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    output = Path(tempfile.mkdtemp(prefix="echo-capacity-hbm-reference-"))
    print(f"temporary HBM reference: {output}", flush=True)
    entry = Path(__file__).resolve()
    entry_digest = digest(entry)
    shutil.copyfile(entry, output / "reference_source.py")
    metadata = {
        "schema": "echo-capacity-reference-generation-v1",
        "status": "running",
        "stage": "load_echo_evidence",
        "echo_run_dir": str(args.run_dir.resolve()),
        "entry_source_sha256": entry_digest,
        "started_unix": time.time(),
        "completed_users": 0,
        "completed_requests": 0,
        "reference_candidate_persistence": "resident_extend_then_truncate",
        "schedule": "group by user; each user starts from empty HBM cache, builds the complete real history, then executes visits in order with truncate between visits; release before next user",
        "measurement_boundary": "independent full-output numerical reference only; reordered requests are not a latency or resident-user capacity comparison",
    }
    write_json(output / "status.json", metadata)
    backend = None
    binding = None
    try:
        from evaluation.provenance import backend_provenance

        state, binding, groups, results = load_run(args.run_dir)
        metadata.update(echo_run_id=state["run_id"], binding=binding, config=state["config"])
        metadata["stage"] = "checkpoint_and_backend_identity"
        model_path = (args.model_path or Path(state["checkpoint"]["path"])).resolve()
        inventory = {
            path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(model_path.glob("*.safetensors"))
        }
        if (
            str(model_path) != state["checkpoint"]["path"]
            or inventory != state["checkpoint"]["files"]
        ):
            raise AssertionError("checkpoint path or shard inventory differs from the ECHO run")
        metadata["backend_provenance"] = backend_provenance()
        if json.dumps(metadata["backend_provenance"], sort_keys=True) != json.dumps(
            state["backend_provenance"], sort_keys=True
        ):
            raise AssertionError("installed backend provenance differs from the ECHO run")
        metadata["stage"] = "hardware_initialization"
        device = torch.device(args.device or state["hardware"]["device"])
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("HBM reference requires one SM90/Hopper CUDA device")
        torch.cuda.set_device(device)
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
        props = torch.cuda.get_device_properties(device)
        metadata["hardware"] = {
            "device": str(device),
            "name": props.name,
            "uuid": str(props.uuid),
            "total_memory": props.total_memory,
            "sm_count": props.multi_processor_count,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        metadata["stage"] = "model_loading"
        write_json(output / "status.json", metadata)
        from models.deepseek_v32.serving_backend import DeepSeekServingBackend

        backend = DeepSeekServingBackend(
            model_path,
            scheme="hbm",
            device=str(device),
            num_layers=10,
            chunk_size=1024,
            sparse_pool_tokens=state["config"]["sparse_pool_tokens"],
            workspace_query_tokens=max(1024, state["config"]["candidate_tokens"]),
            linear_backend=state["backend"]["linear_backend"],
        )
        metadata["backend"] = backend.describe()
        run_users(backend, args.run_dir, output, state, groups, results, metadata)
        metadata["stage"] = "source_verification"
        verify_recorded_source(args.run_dir, state)
        if digest(entry) != entry_digest:
            raise AssertionError("reference generation entry changed while running")
        metadata["status"] = "complete"
    except BaseException as error:  # noqa: BLE001 - Preserve failures and interruptions for inspection.
        metadata["status"] = "failed"
        metadata["failure"] = {
            "stage": metadata["stage"],
            "type": type(error).__name__,
            "message": str(error),
        }
        (output / "error.txt").write_text(traceback.format_exc())
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:  # noqa: BLE001 - Failed cleanup invalidates publication.
                metadata["status"] = "failed"
                metadata["cleanup_error"] = repr(error)
        metadata["finished_unix"] = time.time()
        write_json(output / "status.json", metadata)
    if metadata["status"] != "complete":
        print(f"FAILED; HBM reference artifacts retained at {output}", flush=True)
        return 1
    write_json(
        output / "manifest.json",
        {
            "schema": REFERENCE_SCHEMA,
            "scheme": "hbm",
            "status": "complete",
            "binding": binding,
            "reference_candidate_persistence": "resident_extend_then_truncate",
            "echo_run_id": metadata["echo_run_id"],
            "entry_source_sha256": entry_digest,
            "requests": metadata["completed_requests"],
            "schedule": metadata["schedule"],
            "measurement_boundary": metadata["measurement_boundary"],
            "comparison": "all candidate hidden and complete last-token logits bitwise identical to saved ECHO outputs for every request",
        },
    )
    try:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        shutil.move(str(output), str(args.output_dir))
    except OSError as error:
        metadata["status"] = "failed"
        metadata["failure"] = {
            "stage": "publication",
            "type": type(error).__name__,
            "message": str(error),
        }
        write_json(output / "status.json", metadata)
        print(f"FAILED to publish; HBM reference retained at {output}: {error}", flush=True)
        return 1
    print(f"COMPLETE; all outputs bitwise identical; HBM reference={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
