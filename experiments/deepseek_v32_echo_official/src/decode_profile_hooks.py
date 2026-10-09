"""Observe official decode CUDA Graphs using captured native GPU node identities."""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager

from experiments.deepseek_v32_echo_official.src import engine_profile_hooks as base


class DecodeProfiler(base._Profiler):
    def __init__(self, *args):
        super().__init__(*args)
        self.graphs = []
        self.capture = None
        self.inspector = None
        self.collection_started = False

    @contextmanager
    def stage(self, stage, **fields):
        capture = self.capture
        if capture is None or not self.torch.cuda.is_current_stream_capturing():
            with super().stage(stage, **fields):
                yield
            return
        stream = self.torch.cuda.current_stream().cuda_stream
        graph_id, before = self.inspector.snapshot(stream)
        with super().stage(stage, **fields):
            yield
        after_id, after = self.inspector.snapshot(stream)
        if after_id != graph_id or not before.keys() <= after.keys():
            raise ValueError("Observed graph changed identity or removed nodes")
        for node in after.keys() - before.keys():
            if after[node] in (0, 1, 2) and str(node) not in capture["owners"]:
                capture["owners"][str(node)] = {
                    "stage": stage,
                    "layer": self.current.get("layer", -1),
                    **fields,
                }
        capture.update(capture_graph_id=graph_id, node_types=after)

    def capture_wrapper(self, original):
        def call(runner, graph, pool, stream, run_once):
            from experiments.deepseek_v32_echo_official.src.decode_graph_inspector import (
                GraphInspector,
            )

            if not self.collection_started:
                self.torch.cuda.profiler.start()
                # NSYS must observe graph creation/clone events before replay.
                # This delay is in Engine setup, before all ordinary requests.
                time.sleep(0.5)
                self.collection_started = True
            if self.inspector is None:
                self.inspector = GraphInspector()
            if self.capture is not None:
                raise ValueError("Nested official graph capture")
            capture = {"owners": {}}

            def observed():
                previous = self.current
                self.local.current = {"id": -1, "stage_calls": {}, "layers": []}
                self.capture = capture
                try:
                    with self.stage("graph_body"):
                        return run_once()
                finally:
                    self.capture = None
                    self.local.current = previous

            result = original(runner, graph, pool, stream, observed)
            gpu_nodes = {
                str(node) for node, kind in capture["node_types"].items() if kind in (0, 1, 2)
            }
            if gpu_nodes != capture["owners"].keys():
                raise ValueError("Graph GPU ownership is incomplete")
            capture.update(
                executable_graph_id=None,
                executable_binding="Torch 2.8 exposes no executable handle; the NSYS replay must match this complete native node set and process through graph lineage.",
                provider=self.inspector.provenance,
            )
            self.graphs.append(capture)
            self.persist()
            return result

        return call

    def layer_wrapper(self, original):
        parent = super().layer_wrapper(original)

        def call(layer, *args, **kwargs):
            # Parent sets the layer inside its wrapper; the final outer ownership
            # is added by inner stages or by the separate layer fallback below.
            if self.current is None:
                return parent(layer, *args, **kwargs)
            previous = self.current.get("layer", -1)
            self.current["layer"] = int(layer.layer_id)
            try:
                with self.stage("cache_control"):
                    return parent(layer, *args, **kwargs)
            finally:
                self.current["layer"] = previous

        return call

    def forward_wrapper(self, original):
        def call(runner, batch, *args, **kwargs):
            if not self.recording_enabled:
                return original(runner, batch, *args, **kwargs)
            if batch.forward_mode.is_decode():
                lengths = base._cpu_ints(batch.seq_lens_cpu, "seq_lens_cpu")
                if lengths != [65537] or batch.batch_size != 1 or batch.input_ids.numel() != 1:
                    raise ValueError("Expected one normal decode at H=65536")
                metadata = {
                    "phase": "decode",
                    "mode": batch.forward_mode.name,
                    "q": 1,
                    "start": 65536,
                    "end": 65537,
                    "batch_size": 1,
                    "seq_lens": lengths,
                    "request_pool_indices": base._cpu_ints(
                        batch.req_pool_indices_cpu, "req_pool_indices_cpu"
                    ),
                }
            else:
                metadata = base.describe_forward(batch)
            metadata.update(id=len(self.records), stage_calls={}, layers=[], completed=False)
            self.records.append(metadata)
            previous = self.current
            self.local.current = metadata
            try:
                with base.nvtx_range(
                    self.torch.cuda.nvtx,
                    self.label(
                        "forward",
                        phase=metadata["phase"],
                        q=metadata["q"],
                        start=metadata["start"],
                        end=metadata["end"],
                        case=self.case,
                    ),
                ):
                    result = original(runner, batch, *args, **kwargs)
                metadata.update(completed=True, cuda_graph=bool(result[1]))
                if metadata["phase"] == "decode" and not metadata["cuda_graph"]:
                    raise ValueError("Normal decode did not use its enabled official CUDA Graph")
                self.last_batch, self.last_metadata = batch, metadata
                return result
            finally:
                self.local.current = previous

        return call

    def persist(self):
        super().persist()
        base._write_json(
            self.output / "decode_graphs.json",
            {
                "schema": "echo-official-decode-graph-ownership-v1",
                "graphs": self.graphs,
                "hook_source_sha256": base._sha(__file__),
                "collection_started_before_first_capture": self.collection_started,
                "scope": "Read-only CUDA capture node enumeration and CUPTI IDs. Original callables run once. No graph nodes, cache state, or computation bodies are changed.",
            },
        )

    def finish(self, parameters):
        if parameters or len(self.records) != 65:
            raise ValueError("Expected exactly 64 prefill chunks plus one decode")
        for index, row in enumerate(self.records[:64]):
            if (
                row["phase"] != "prefill"
                or row["start"] != index * 1024
                or row["end"] != (index + 1) * 1024
            ):
                raise ValueError("Incomplete normal prefill")
        if self.records[-1]["phase"] != "decode" or not all(
            row["completed"] for row in self.records
        ):
            raise ValueError("Incomplete normal decode")
        batch = self.last_batch
        with base.nvtx_range(
            self.torch.cuda.nvtx, self.label("diagnostic", purpose="post_request_residency")
        ):
            self.torch.cuda.synchronize()
            request = self.last_metadata["request_pool_indices"]
            if request is None or len(request) != 1:
                raise ValueError("Missing completed request pool index")
            logical = batch.req_to_token_pool.req_to_token[request[0], :65537].detach().cpu()
            if logical.unique().numel() != 65537:
                raise ValueError("Duplicate request token slots")
            pool = batch.token_to_kv_pool
            layers = []
            if hasattr(pool, "host_token_to_device"):
                ids = logical.to(device=pool.host_token_to_device.device, dtype=self.torch.long)
                mapped = pool.host_token_to_device[:, ids]
                reverse = pool.device_token_to_host
                for layer in range(3):
                    slots = mapped[layer]
                    valid = (slots > 0) & (slots <= pool.device_pool.size)
                    safe = slots.clamp(0, pool.device_pool.size).long()
                    valid &= reverse[layer, safe] == ids
                    values = valid.cpu().tolist()
                    layers.append(
                        {
                            "layer": layer,
                            "history_resident_tokens": sum(values[:65536]),
                            "decode_input_resident_tokens": int(values[-1]),
                            "bidirectional_mapping_checked": True,
                        }
                    )
            else:
                if not bool(((logical >= 0) & (logical < pool.size + pool.page_size)).all()):
                    raise ValueError("Resident token slot exceeds its pool")
                layers = [
                    {
                        "layer": layer,
                        "history_resident_tokens": 65536,
                        "decode_input_resident_tokens": 1,
                        "bidirectional_mapping_checked": False,
                    }
                    for layer in range(3)
                ]
            snapshot = {
                "schema": "echo-normal-decode-residency-v1",
                "decode_input_ids": batch.input_ids.detach().cpu().tolist(),
                "layers": layers,
                "logical_ids_sha256": hashlib.sha256(logical.numpy().tobytes()).hexdigest(),
                "boundary": "Synchronized observation after generate returned, outside formal request and forward windows; no cache reset or flush.",
            }
        base._write_json(self.output / "residency.json", snapshot)
        self.persist()


def scheduler(*args, **kwargs):
    # isort: off
    # Preserve the official spawn import order; Indexer-first causes a circular import.
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
    import deep_gemm
    from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
    from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
    from sglang.srt.mem_cache import memory_pool_host
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    # isort: on

    base._Profiler = DecodeProfiler
    profiler = base.install_hooks()
    profiler.patch(CudaGraphRunner, "_capture_graph", profiler.capture_wrapper)
    for owner, name, stage in [
        (deep_gemm, "fp8_paged_mqa_logits_fused_v2", "indexer_prefetch"),
        (deep_gemm, "fp8_paged_mqa_logits", "indexer"),
        (NativeSparseAttnBackend, "forward_decode", "cache_control"),
        (NativeSparseAttnBackend, "_forward_fa3", "attention"),
        (NativeSparseAttnBackend, "_forward_flashmla_decode", "attention"),
        (Indexer, "_get_topk_paged", "indexer"),
        (
            memory_pool_host.NSATokenToKVPoolHost,
            "recall_miss_tokens_decode_cuda_graph_with_prefetch",
            "cache_recall",
        ),
        (
            memory_pool_host.NSATokenToKVPoolHost,
            "recall_miss_tokens_decode_cuda_graph",
            "cache_recall",
        ),
    ]:
        profiler.patch(owner, name, profiler.stage_wrapper(stage))
    Scheduler.echo_decode_finish = lambda scheduler, parameters: profiler.finish(parameters)
    primary = None
    try:
        return run_scheduler_process(*args, **kwargs)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            profiler.persist()
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup(
                    "Scheduler and observation persistence failed", [primary, error]
                )
            raise
