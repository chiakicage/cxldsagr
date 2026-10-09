"""Reject mislabeled runs and incomplete native CUDA Graph evidence."""

import copy
import hashlib
import json
import sqlite3

import pytest

from experiments.deepseek_v32_echo_official.src import decode_timeline as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("mode", ["check", "bench", "profile"])
def test_reject_incomplete_completed_run(tmp_path, mode):
    write_json(
        tmp_path / "run.json",
        {
            "status": "completed",
            "mode": mode,
            "numerical_acceptance": False,
            "arguments": {"repeats": 3},
            "workers": [{"name": "sample_00"}, {"name": "sample_01"}],
        },
    )
    with pytest.raises(ValueError, match="Worker samples"):
        report.read_run(tmp_path, mode)


@pytest.fixture
def graph_evidence(tmp_path, monkeypatch):
    pid = 123
    process = (1 << 48) + (pid << 24)
    thread = process + 124
    hooks = tmp_path / "hooks"
    write_json(hooks / "hook_manifest.json", {"pid": pid, "case": "hbm"})
    forwards = [
        {"id": i, "phase": "prefill", "q": 1024, "start": i * 1024, "end": (i + 1) * 1024}
        for i in range(64)
    ]
    forwards.append(
        {
            "phase": "decode",
            "id": 64,
            "q": 1,
            "start": 65536,
            "end": 65537,
            "mode": "DECODE",
            "cuda_graph": True,
        }
    )
    write_json(hooks / "forwards.json", forwards)
    owners = {str(i + 1): {"stage": "projection", "layer": i} for i in range(3)}
    owners["4"] = {"stage": "cache_control", "layer": 1}
    write_json(
        hooks / "decode_graphs.json", {"graphs": [{"capture_graph_id": 1, "owners": owners}]}
    )
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId,originalGraphNodeId,globalTid)")
        db.executemany(
            "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(?,?,?)",
            [(101 + i, 1 + i, thread) for i in range(4)],
        )
    gpu = []
    for i in range(4):
        gpu.append(
            {
                "kind": "kernel" if i < 3 else "memcpy",
                "name": "compute" if i < 3 else "Device-to-host",
                "start_ns": 100 + i * 100 if i < 3 else 220,
                "end_ns": 150 + i * 100 if i < 3 else 230,
                "forward_id": 64,
                "graph_node_id": 101 + i,
                "graph_id": 2 if i < 3 else None,
                "process": process,
                "device": 0,
                "scope_id": 1,
                "bytes": None if i < 3 else 64,
            }
        )
    capture = {
        "sha256": "synthetic",
        "activities": gpu,
        "scopes": [
            {
                "thread": thread,
                "start_ns": 50,
                "end_ns": 80,
                "fields": {"kind": "forward", "case": "hbm", **forward},
            }
            for forward in forwards
        ]
        + [
            {
                "thread": thread,
                "start_ns": 50,
                "end_ns": 80,
                "fields": {"kind": "layer", "id": i, "layer": layer},
            }
            for i in range(64)
            for layer in range(3)
        ],
        "apis": [{"name": "cudaGraphLaunch", "context": {"id": 64}}],
    }
    monkeypatch.setattr(report.timeline, "read_capture", lambda path: capture)
    monkeypatch.setattr(report.timeline, "attribute", lambda value: value)
    return path, hooks, capture, thread


def test_exact_node_binding_includes_copy_without_graph_id(graph_evidence):
    path, hooks, _, _ = graph_evidence
    metrics, activities = report.select(path, hooks, "hbm")
    assert metrics["graph_gpu_nodes"] == 4
    assert metrics["every_graph_node_verified"]
    assert {row["graph_capture_node_id"] for row in activities} == {1, 2, 3, 4}


@pytest.mark.parametrize(
    "damage",
    [
        "missing_lineage",
        "foreign_lineage",
        "duplicate_node",
        "missing_node",
        "wrong_case",
        "missing_prefill",
        "missing_layer",
        "changed_geometry",
    ],
)
def test_reject_incomplete_graph_binding(graph_evidence, damage):
    path, hooks, capture, thread = graph_evidence
    if damage == "missing_lineage":
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM CUDA_GRAPH_NODE_EVENTS WHERE graphNodeId=102")
    elif damage == "foreign_lineage":
        with sqlite3.connect(path) as db:
            db.execute("UPDATE CUDA_GRAPH_NODE_EVENTS SET globalTid=?", (thread + (1 << 24),))
    elif damage == "duplicate_node":
        capture["activities"].append(copy.deepcopy(capture["activities"][0]))
    elif damage == "missing_node":
        capture["activities"].pop()
    elif damage == "wrong_case":
        write_json(hooks / "hook_manifest.json", {"pid": 123, "case": "echo"})
    elif damage == "missing_prefill":
        capture["scopes"].pop(0)
    elif damage == "missing_layer":
        capture["scopes"].pop()
    elif damage == "changed_geometry":
        capture["scopes"][0]["fields"]["q"] = 1
    with pytest.raises(ValueError):
        report.select(path, hooks, "hbm")


@pytest.fixture
def run_group(tmp_path):
    check_path = tmp_path / "check"
    write_json(check_path / "run.json", {"run_id": "independent_check"})
    check = {
        "run_id": "independent_check",
        "path": str(check_path / "run.json"),
        "sha256": report.timeline.sha256(check_path / "run.json"),
    }
    runs = {
        mode: {
            "case": "resident_reference",
            "run_id": "independent_check" if mode == "check" else mode,
            "identity": {"case": "resident_reference", "sources": {}},
            "check": copy.deepcopy(check),
            "workers": [
                {
                    "completion": {
                        "case": "resident_reference",
                        "prefix_sha256": "prefix",
                        "row": {"input_sha256": "prefix"},
                    }
                }
            ],
        }
        for mode in ("check", "bench", "profile")
    }
    return runs, check_path


@pytest.mark.parametrize(
    "damage", ["wrong_case", "changed_check", "wrong_check_path", "different_input"]
)
def test_reject_mismatched_run_group(run_group, damage):
    runs, path = run_group
    assert report.verify_run_group(runs, "hbm", path) == "prefix"
    if damage == "wrong_case":
        runs["bench"]["case"] = "echo"
    elif damage == "changed_check":
        (path / "run.json").write_text("changed bytes")
    elif damage == "wrong_check_path":
        runs["profile"]["check"]["path"] = str(path / "another.json")
    elif damage == "different_input":
        runs["bench"]["workers"][0]["completion"]["row"]["input_sha256"] = "another prefix"
    with pytest.raises(ValueError):
        report.verify_run_group(runs, "hbm", path)


@pytest.mark.parametrize("damage", ["hook", "nsys", "sqlite", "export_input"])
def test_reject_changed_profile_artifacts(tmp_path, damage):
    hooks = tmp_path / "hooks"
    write_json(hooks / "forwards.json", {"observed": "forward"})
    nsys = tmp_path / "trace.nsys-rep"
    nsys.write_bytes(b"original nsys bytes")
    sqlite = tmp_path / "trace.sqlite"
    sqlite.write_bytes(b"original export bytes")
    worker = {
        "hooks": {"forwards.json": report.timeline.sha256(hooks / "forwards.json")},
        "nsys": {"path": str(nsys), "sha256": report.timeline.sha256(nsys)},
    }
    receipt = {
        "exitcode": 0,
        "input": worker["nsys"],
        "output": {"path": str(sqlite), "sha256": report.timeline.sha256(sqlite)},
    }
    write_json(sqlite.with_suffix(".export.json"), receipt)
    report.verify_profile_binding(worker, hooks, sqlite)
    if damage == "hook":
        (hooks / "forwards.json").write_text("changed hook bytes")
    elif damage == "nsys":
        nsys.write_bytes(b"changed nsys bytes")
    elif damage == "sqlite":
        sqlite.write_bytes(b"changed export bytes")
    elif damage == "export_input":
        receipt["input"] = {"path": str(nsys), "sha256": "unrelated capture"}
        write_json(sqlite.with_suffix(".export.json"), receipt)
    with pytest.raises(ValueError):
        report.verify_profile_binding(worker, hooks, sqlite)


@pytest.mark.parametrize("damage", [None, "log", "server", "capacity", "line"])
def test_capacity_text_and_raw_bytes_boundaries(tmp_path, damage):
    log = tmp_path / "worker.log"
    log.write_bytes(b"progress\rallocated 64 tokens\n")
    server = {"pool": 64}
    capacity = {
        "server_info_sha256": hashlib.sha256(
            json.dumps(server, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "log_sha256": {str(log): hashlib.sha256(log.read_text().encode()).hexdigest()},
        "evidence": {
            "device": [{"source": str(log), "line_number": 2, "line": "allocated 64 tokens"}]
        },
    }
    write_json(tmp_path / "engine/server_info.json", server)
    write_json(tmp_path / "capacity.json", capacity)
    assert report.timeline.sha256(log) != capacity["log_sha256"][str(log)]
    report.verify_capacity_binding(tmp_path, capacity)
    if damage is None:
        return
    if damage == "log":
        log.write_bytes(b"allocated 128 tokens\n")
    elif damage == "server":
        write_json(tmp_path / "engine/server_info.json", {"pool": 128})
    elif damage == "capacity":
        write_json(tmp_path / "capacity.json", {})
    elif damage == "line":
        capacity["evidence"]["device"][0]["line"] = "allocated 128 tokens"
        write_json(tmp_path / "capacity.json", capacity)
    with pytest.raises(ValueError):
        report.verify_capacity_binding(tmp_path, capacity)
