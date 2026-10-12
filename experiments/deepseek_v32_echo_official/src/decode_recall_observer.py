"""Snapshot official Q1 recall inputs inside an independent observation graph.

The original recall function runs once. Copies read its inputs before it mutates
the page table and its device counter after it returns. Host reads and branch
classification happen only after the ordinary request has completed.
"""

from __future__ import annotations

import inspect
import json
import os

from experiments.deepseek_v32_echo_official.src import engine_profile_hooks as base
from experiments.deepseek_v32_echo_official.src.decode_profile_hooks import DecodeProfiler

I32_MAX = 2**31 - 1
RECORD_BYTES = 576 * 2


def classify_recall(host_ids, old_slots, stage_ids, prefetch_device_start, counter):
    """Reproduce the fixed official recall kernel's copy-source branch on CPU."""
    if len(host_ids) != 1 or len(old_slots) != 1 or len(stage_ids) != 1:
        raise ValueError("Recall observation requires query=1")
    hosts, slots, staged = host_ids[0], old_slots[0], stage_ids[0]
    if len(hosts) != len(slots) or not staged or prefetch_device_start <= 0:
        raise ValueError("Inconsistent recall snapshot geometry")
    h2d, d2d, invalid = [], [], []
    for column, (host_id, old_slot) in enumerate(zip(hosts, slots)):
        if host_id < 0:
            invalid.append(column)
            continue
        is_prefetched = prefetch_device_start <= old_slot < I32_MAX
        slot = old_slot - prefetch_device_start
        if is_prefetched:
            # The official kernel only validates the host ID when slot < cap.
            # A larger tag would still enter its D2D branch with an invalid read.
            if slot >= len(staged):
                raise ValueError("Official staging tag exceeds the prefetch buffer")
            is_prefetched = staged[slot] == host_id
        (d2d if is_prefetched else h2d).append(column)
    total = len(h2d) + len(d2d)
    if type(counter) is not int or counter != total:
        raise ValueError("Input-derived recall count differs from official device counter")
    return {
        "query_tokens": 1,
        "selection_columns": len(hosts),
        "invalid_columns": invalid,
        "h2d_columns": h2d,
        "d2d_columns": d2d,
        "dram_h2d_records": len(h2d),
        "prefetch_d2d_records": len(d2d),
        "total_recall_records": total,
        "official_recall_counter": counter,
        "record_bytes": RECORD_BYTES,
        "dram_h2d_bytes": len(h2d) * RECORD_BYTES,
        "prefetch_d2d_bytes": len(d2d) * RECORD_BYTES,
        "total_recall_copy_bytes": total * RECORD_BYTES,
        "counter_matches_input": True,
    }


class RecallObserver(DecodeProfiler):
    def __init__(self, *args):
        super().__init__(*args)
        self.capture_snapshots = []
        self.current_snapshots = None
        self.selected_snapshots = None
        self.observed_replays = 0

    def capture_wrapper(self, original):
        def call(runner, graph, pool, stream, run_once):
            from experiments.deepseek_v32_echo_official.src.decode_graph_inspector import (
                NativeCaptureInspector,
            )

            if self.inspector is None:
                self.inspector = NativeCaptureInspector()
            if self.capture is not None:
                raise ValueError("Nested recall-observation graph capture")
            capture, snapshots = {"owners": {}}, []

            def observed():
                previous = self.current
                self.local.current = {"id": -1, "stage_calls": {}, "layers": []}
                self.capture, self.current_snapshots = capture, snapshots
                try:
                    with self.stage("graph_body"):
                        return run_once()
                finally:
                    self.capture, self.current_snapshots = None, None
                    self.local.current = previous

            result = original(runner, graph, pool, stream, observed)
            if sorted(item["layer"] for item in snapshots) != [0, 1, 2]:
                raise ValueError("Observation graph must capture each of three recall layers once")
            gpu_nodes = {
                str(node) for node, kind in capture["node_types"].items() if kind in (0, 1, 2)
            }
            if gpu_nodes != capture["owners"].keys():
                raise ValueError("Observation graph GPU ownership is incomplete")
            capture.update(provider=self.inspector.provenance, capture_index=len(self.graphs))
            self.capture_snapshots.append((graph, capture, snapshots))
            self.graphs.append(capture)
            self.persist()
            return result

        return call

    def recall_wrapper(self, original):
        signature = inspect.signature(original)

        def call(*args, **kwargs):
            if self.capture is None:
                return original(*args, **kwargs)
            if not self.torch.cuda.is_current_stream_capturing():
                raise ValueError("Recall snapshot called outside native graph capture")
            values = signature.bind(*args, **kwargs).arguments
            hosts = values["recall_host_indices"]
            slots = values["device_page_table_1"]
            staged = values["prefetch_host_loc"]
            if (
                hosts.ndim != 2
                or hosts.shape[0] != 1
                or hosts.shape != slots.shape
                or hosts.stride(0) != hosts.shape[1]
                or slots.stride(0) != slots.shape[1]
                or hosts.stride(1) != 1
                or slots.stride(1) != 1
                or staged.ndim != 2
                or staged.shape[0] != 1
                or values["kv_lora_rank"] != 512
                or values["qk_rope_head_dim"] != 64
                or values["layer_device_kv"].element_size() != 2
            ):
                raise ValueError("Unexpected official Q1 recall input layout")
            item = {
                "layer": self.current["layer"],
                "prefetch_device_start": int(values["prefetch_device_start"]),
                "prefetch_capacity": staged.shape[1],
            }
            with self.stage("recall_observation_input"):
                item["host_ids"] = hosts.clone()
                item["old_slots"] = slots.clone()
                item["stage_ids"] = staged.clone()
            with self.stage("cache_recall_transfer"):
                result = original(*args, **kwargs)
            with self.stage("recall_observation_counter"):
                item["counter"] = values["recall_counter"].clone()
            self.current_snapshots.append(item)
            return result

        return call

    def replay_wrapper(self, original):
        def call(runner, batch, *args, **kwargs):
            result = original(runner, batch, *args, **kwargs)
            if self.recording_enabled and batch.forward_mode.is_decode():
                graph = runner.graphs[runner.bs]
                matches = [item for item in self.capture_snapshots if item[0] is graph]
                if len(matches) != 1 or self.observed_replays:
                    raise ValueError("Expected exactly one observed decode graph replay")
                _, self.selected_capture, self.selected_snapshots = matches[0]
                self.observed_replays += 1
            return result

        return call

    def persist(self):
        super().persist()
        graphs = self.output / "decode_graphs.json"
        graph_record = json.loads(graphs.read_text())
        graph_record["schema"] = "echo-official-recall-observation-graphs-v1"
        for graph in graph_record["graphs"]:
            graph["capture_id"] = graph.pop("capture_graph_id")
        graph_record["scope"] = (
            "Independent observation graphs add input/counter copy nodes. Native node ownership "
            "is enumerated during capture; the selected actual graph object is bound at replay. "
            "These graphs do not supply the existing official decode timeline or timing results."
        )
        base._write_json(graphs, graph_record)
        manifest = self.output / "hook_manifest.json"
        hook_record = json.loads(manifest.read_text())
        hook_record["schema"] = "echo-engine-recall-observation-hooks-v1"
        hook_record["boundary"] = (
            "Independent observation only. Original callables run once with unchanged inputs. "
            "Added graph copies snapshot recall inputs/counter; no device readback or "
            "synchronization occurs inside forward. This run supplies recall bytes, not latency."
        )
        base._write_json(manifest, hook_record)
        base._write_json(
            self.output / "recall_observation_manifest.json",
            {
                "schema": "echo-official-recall-observer-v1",
                "pid": os.getpid(),
                "source_sha256": base._sha(__file__),
                "graph_count": len(self.graphs),
                "observed_replays": self.observed_replays,
                "numerical_acceptance": False,
                "boundary": "Independent observation only. Added graph copies snapshot actual recall inputs and the original device counter; original functions run once. No host readback or synchronization occurs in forward. Observation latency must not be used as formal timing or attributed to existing profile windows.",
            },
        )

    def finish(self, parameters):
        super().finish(parameters)
        if self.observed_replays != 1 or self.selected_snapshots is None:
            raise ValueError("No unique normal decode replay observation")
        self.torch.cuda.synchronize()
        layers = []
        for snapshot in sorted(self.selected_snapshots, key=lambda item: item["layer"]):
            raw = {
                key: snapshot[key].detach().cpu().tolist()
                for key in ("host_ids", "old_slots", "stage_ids", "counter")
            }
            if len(raw["counter"]) != 1:
                raise ValueError("Official recall counter must be a single device scalar")
            classified = classify_recall(
                raw["host_ids"],
                raw["old_slots"],
                raw["stage_ids"],
                snapshot["prefetch_device_start"],
                raw["counter"][0],
            )
            layers.append(
                {
                    **classified,
                    "layer": snapshot["layer"],
                    "prefetch_device_start": snapshot["prefetch_device_start"],
                    "prefetch_capacity": snapshot["prefetch_capacity"],
                    "input_snapshot": raw,
                }
            )
        base._write_json(
            self.output / "recall_observation.json",
            {
                "schema": "echo-official-normal-decode-recall-v1",
                "numerical_acceptance": False,
                "decode_input_ids": self.last_batch.input_ids.detach().cpu().tolist(),
                "capture_index": self.selected_capture["capture_index"],
                "capture_id": self.selected_capture["capture_graph_id"],
                "observed_replays": self.observed_replays,
                "layers": layers,
                "totals": {
                    key: sum(layer[key] for layer in layers)
                    for key in (
                        "dram_h2d_records",
                        "prefetch_d2d_records",
                        "total_recall_records",
                        "dram_h2d_bytes",
                        "prefetch_d2d_bytes",
                        "total_recall_copy_bytes",
                    )
                },
                "boundary": "Inputs copied before each layer's original recall_update_with_prefetch in the actual graph replay; counter copied after that function. CPU readback and branch reconstruction follow generate completion. H2D counts only this residual recall's mapped-host copies, D2D counts staging promotion copies; fused indexer prefetch traffic is excluded.",
            },
        )
        self.persist()


def scheduler(*args, **kwargs):
    # isort: off
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
    from sglang.srt.mem_cache import memory_pool_host
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    # isort: on

    base._Profiler = RecallObserver
    observer = base.install_hooks()
    observer.patch(CudaGraphRunner, "_capture_graph", observer.capture_wrapper)
    observer.patch(CudaGraphRunner, "replay", observer.replay_wrapper)
    observer.patch(memory_pool_host, "recall_update_with_prefetch", observer.recall_wrapper)
    Scheduler.echo_decode_finish = lambda scheduler, parameters: observer.finish(parameters)
    primary = None
    try:
        return run_scheduler_process(*args, **kwargs)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            observer.persist()
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup(
                    "Scheduler and recall-observation persistence failed", [primary, error]
                )
            raise
