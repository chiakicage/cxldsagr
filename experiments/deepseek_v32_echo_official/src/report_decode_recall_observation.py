"""Audit and publish input-derived official normal-decode recall byte counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from experiments.deepseek_v32_echo_official.src.decode_recall_observer import classify_recall
from experiments.deepseek_v32_echo_official.src.decode_run import PROTOCOL
from experiments.deepseek_v32_echo_official.src.engine_run import file_sha256, write_json


def audit(run):
    record = json.loads((run / "run.json").read_text())
    if (
        record["status"] != "completed"
        or record["mode"] != "observe"
        or record["case"] != "echo"
        or record["protocol"] != PROTOCOL
        or len(record["workers"]) != 1
        or record["numerical_acceptance"] is not False
    ):
        raise ValueError("Expected one completed independent official recall observation")
    for name, digest in record["sources"].items():
        if file_sha256(run / "sources" / name) != digest:
            raise ValueError(f"Archived observation source changed: {name}")
    worker = record["workers"][0]
    engine = run / worker["name"] / "engine"
    completion = json.loads((engine / "complete.json").read_text())
    if completion != worker["completion"] or "wall_ms" in completion["row"]:
        raise ValueError("Observation completion differs or claims formal latency")
    if (
        completion["status"] != "completed"
        or completion["mode"] != "observe"
        or completion["protocol"] != PROTOCOL
        or file_sha256(engine / "warmup.json") != completion["warmup_sha256"]
        or file_sha256(engine / "response.json") != completion["row"]["response_sha256"]
    ):
        raise ValueError("Completed ordinary request or warmup artifacts changed")
    capacity = json.loads((run / worker["name"] / "capacity.json").read_text())
    if capacity != worker["capacity"] or not capacity["passed"]:
        raise ValueError("Official token capacity audit changed or did not pass")
    server_info = json.loads((engine / "server_info.json").read_text())
    canonical = json.dumps(server_info, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != capacity["server_info_sha256"]:
        raise ValueError("Official capacity source changed")
    for name, digest in capacity["log_sha256"].items():
        normalized = Path(name).read_text().encode()
        if hashlib.sha256(normalized).hexdigest() != digest:
            raise ValueError("Official capacity log changed")
    hooks = engine / "hooks"
    if {path.name: file_sha256(path) for path in hooks.glob("*.json")} != worker["hooks"]:
        raise ValueError("Observation hook artifacts changed")
    observed = json.loads((hooks / "recall_observation.json").read_text())
    if (
        observed != worker["recall_observation"]
        or observed["observed_replays"] != 1
        or observed["decode_input_ids"] != completion["row"]["output_ids"][:1]
        or observed["numerical_acceptance"] is not False
    ):
        raise ValueError("Observation is not bound to the completed normal decode")
    if (
        file_sha256(hooks / "recall_observation.json")
        != completion["row"]["recall_observation"]["sha256"]
    ):
        raise ValueError("Observed recall digest differs from worker completion")
    forwards = json.loads((hooks / "forwards.json").read_text())
    if (
        len(forwards) != 65
        or any(not row["completed"] for row in forwards)
        or forwards[-1]["mode"] != "DECODE"
        or forwards[-1]["q"] != 1
        or forwards[-1]["seq_lens"] != [65537]
        or not forwards[-1]["cuda_graph"]
        or any(
            row["start"] != index * 1024
            or row["end"] != (index + 1) * 1024
            or row["phase"] != "prefill"
            for index, row in enumerate(forwards[:64])
        )
    ):
        raise ValueError("Official forward geometry does not cover H64K plus one normal decode")
    graphs = json.loads((hooks / "decode_graphs.json").read_text())["graphs"]
    graph = graphs[observed["capture_index"]]
    if graph["capture_id"] != observed["capture_id"]:
        raise ValueError("Recall snapshot does not bind its captured native graph")
    if sorted(item["layer"] for item in observed["layers"]) != [0, 1, 2]:
        raise ValueError("Recall observation does not cover exactly three layers")
    for layer in observed["layers"]:
        raw = layer["input_snapshot"]
        expected = classify_recall(
            raw["host_ids"],
            raw["old_slots"],
            raw["stage_ids"],
            layer["prefetch_device_start"],
            raw["counter"][0],
        )
        if any(layer[key] != value for key, value in expected.items()):
            raise ValueError("Saved recall classification differs from original input arrays")
    for key, value in observed["totals"].items():
        if sum(layer[key] for layer in observed["layers"]) != value:
            raise ValueError("Recall byte total differs from individual layers")
    release = run / worker["name"] / "gpu_release_drain.json"
    if (
        file_sha256(release) != worker["gpu_release_sha256"]
        or json.loads(release.read_text())["result"] != "gpu_observed_empty"
    ):
        raise ValueError("Owned GPU release has not been verified")
    return record, observed


def publish(run, output, destination=None):
    record, observed = audit(run)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "recall_observation.json", observed)
    write_json(
        output / "provenance.json",
        {
            "schema": "echo-official-recall-observation-report-v1",
            "run_id": record["run_id"],
            "run_path": str(run.resolve()),
            "run_sha256": file_sha256(run / "run.json"),
            "generator_sha256": file_sha256(Path(__file__)),
            "original_hook_sha256": record["workers"][0]["hooks"],
            "numerical_acceptance": False,
            "saved_input_arrays_reclassified": True,
            "formal_timing": False,
        },
    )
    columns = [
        "layer",
        "dram_h2d_records",
        "prefetch_d2d_records",
        "total_recall_records",
        "dram_h2d_bytes",
        "prefetch_d2d_bytes",
        "total_recall_copy_bytes",
        "official_recall_counter",
    ]
    with (output / "layers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(observed["layers"])
    if destination:
        shutil.copytree(output, destination)
    return {"run_id": record["run_id"], "totals": observed["totals"], "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.run, args.output, args.publish)))


if __name__ == "__main__":
    main()
