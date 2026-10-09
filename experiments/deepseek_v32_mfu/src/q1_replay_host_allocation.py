"""Exercise nine pinned-host allocation requests before the unchanged HBM timer."""

from __future__ import annotations

import weakref
from pathlib import Path

from experiments.deepseek_v32_mfu.src import q1_replay_matched as matched

KIND = "deepseek-q1-replay-host-allocation-v1"
HELPER = "cache/host_allocation.py"
SHAPE = (65600, 576)
LOGICAL_BYTES = 75_571_200
STORAGE_BYTES = 134_217_728
EVIDENCE = "host_allocation_observations.json"


def main():
    source_path = Path(__file__).resolve()
    source_relative = str(source_path.relative_to(matched.ROOT))
    source_sha256 = matched.digest(source_path)
    helper_path = matched.ROOT / HELPER
    helper_sha256 = matched.digest(helper_path)
    original_apply = matched.apply_environment

    def apply_environment(expected):
        original_apply(expected)
        # Preserve the timer's existing provider import order after environment setup.
        from experiments.deepseek_v32_mfu.src import q1_replay_timer as timer

        original_source, original_runtime = timer.source_identity, timer.runtime
        original_save, original_receipt = timer.save, timer.write_receipt

        import torch

        from cache.host_allocation import allocate_host_tensor

        matched.require(
            Path(allocate_host_tensor.__code__.co_filename).resolve() == helper_path,
            "Pinned allocation helper came from a different source",
        )
        run = None
        started = False
        completed = None
        observations = []
        runtime_calls = 0
        diagnostic_artifacts = {}

        def source_identity(args):
            nonlocal run
            result = original_source(args)
            matched.require(matched.digest(source_path) == source_sha256, "Driver source changed")
            matched.require(
                matched.digest(helper_path) == helper_sha256
                and result["sources"].get(HELPER) == helper_sha256,
                "Pinned allocation helper source changed or is not in the source snapshot",
            )
            current = {"run_id": args.run_id, "mode": args.mode}
            if run is None:
                run = current
            else:
                matched.require(run == current, "Allocation diagnostic run changed")
            result["sources"][source_relative] = source_sha256
            result["contract"]["host_allocation"] = {
                "kind": KIND,
                "helper": {"path": HELPER, "sha256": helper_sha256},
                "shape": list(SHAPE),
                "dtype": "bfloat16",
                "pin_memory": True,
                "lifetimes": 3,
                "simultaneous_backings": 3,
                "allocation_requests": 9,
                "logical_bytes_per_request": LOGICAL_BYTES,
                "storage_bytes_per_request": STORAGE_BYTES,
                "position": "after the one compute bank; before the original reduced HBM prefix",
                "cache_transitions": 0,
                "collection": "stopped by the unchanged timer; no added profiler ranges",
                "diagnostic_evidence": EVIDENCE,
                "boundary": "Allocate and drop three simultaneous pinned backings in each of "
                "three lifetimes. No pool, cache switch, offload kernel, extra stream or flush "
                "is added. The normal graph-entry host-cache flush remains. Nine requests "
                "do not imply nine CUDA allocations; absence of an explicit mapping call "
                "does not imply that CUDA pinned memory is inaccessible from the GPU.",
            }
            return result

        def observe(phase, **details):
            stats = dict(torch.cuda.host_memory_stats())
            matched.require(
                {"allocated_bytes.current", "num_host_alloc", "num_host_free"} <= stats.keys(),
                "Pinned host allocator statistics are unavailable",
            )
            observations.append({"phase": phase, **details, "reported_host_allocator_stats": stats})

        class AllocationModel(timer.DeepSeekEchoModel):
            def prepare_compute_graphs(self, query_sizes):
                nonlocal started, completed
                matched.require(not started and run is not None, "Allocation control must run once")
                matched.require(query_sizes == [1024, 1], "Unexpected compute bank query sizes")
                started = True
                result = super().prepare_compute_graphs(query_sizes)
                bank = self._compute_graphs
                generation = self._cache_generation
                cache_ids = tuple(id(block.cache) for block in self.blocks)
                matched.require(bank is not None and self.length == 0, "Unexpected initial bank")
                observe("before_allocation_lifetimes")
                for lifetime in range(3):
                    backings = [
                        allocate_host_tensor(SHAPE, dtype=torch.bfloat16, pin_memory=True)
                        for _ in range(3)
                    ]
                    logical = [tensor.numel() * tensor.element_size() for tensor in backings]
                    storage = [tensor.untyped_storage().nbytes() for tensor in backings]
                    matched.require(
                        logical == [LOGICAL_BYTES] * 3 and storage == [STORAGE_BYTES] * 3,
                        "Pinned logical or backing storage size differs",
                    )
                    view_refs = [weakref.ref(tensor) for tensor in backings]
                    backing_refs = [
                        weakref.ref(tensor if tensor._base is None else tensor._base)
                        for tensor in backings
                    ]
                    observe(
                        "three_backings_live",
                        lifetime=lifetime,
                        logical_bytes=logical,
                        storage_bytes=storage,
                    )
                    del backings
                    matched.require(
                        all(reference() is None for reference in (*view_refs, *backing_refs)),
                        "Pinned tensor or backing owner survived its declared lifetime",
                    )
                    observe("view_and_backing_owners_dropped", lifetime=lifetime)
                matched.require(
                    self.length == 0
                    and not self.offload
                    and self.cache_method == "hbm"
                    and not self._shared_pools
                    and not self._shared_sessions
                    and not self._pool_prefetch
                    and not self._extend_graphs
                    and self._extend_graph is None
                    and self._compute_graphs is bank
                    and self._cache_generation == generation
                    and tuple(id(block.cache) for block in self.blocks) == cache_ids,
                    "Allocation control changed the original HBM cache or compute bank",
                )
                completed = {
                    "lifetimes": 3,
                    "simultaneous_backings": 3,
                    "allocation_requests": 9,
                    "logical_bytes_per_request": LOGICAL_BYTES,
                    "storage_bytes_per_request": STORAGE_BYTES,
                    "view_and_backing_owners_dropped": True,
                    "compute_bank_preparations": 1,
                    "same_compute_bank": True,
                    "cache_generation_increase": 0,
                    "original_empty_hbm_cache_unchanged": True,
                }
                return result

        def runtime():
            nonlocal runtime_calls
            matched.require(completed is not None and runtime_calls < 2, "Invalid runtime boundary")
            result = original_runtime()
            runtime_calls += 1
            observe(
                "runtime_after_graph_construction_and_five_warmups"
                if runtime_calls == 1
                else "runtime_after_check_or_samples",
                runtime_call=runtime_calls,
            )
            result["host_allocation"] = completed
            return result

        def save(path, value):
            matched.require(
                Path(path).name == "result.json" and runtime_calls == 2 and len(observations) == 9,
                "Unexpected final allocation diagnostic boundary",
            )
            diagnostic_path = Path(path).with_name(EVIDENCE)
            original_save(
                diagnostic_path,
                {
                    "schema": "q1-host-allocation-observations-v1",
                    **run,
                    "driver_sha256": source_sha256,
                    "helper_sha256": helper_sha256,
                    "observations": observations,
                    "boundary": "Reported process host-allocator counters outside timed/profile "
                    "ranges. Runtime observations follow construction and five warmups, then "
                    "the check or samples; neither is immediately after a host-cache flush. "
                    "allocated_bytes counts rounded owned blocks, active plus cached. Active "
                    "counters are not used to derive exact cached bytes. These statistics do "
                    "not identify individual cudaHostAlloc/cudaHostRegister calls.",
                },
            )
            diagnostic_artifacts["host_allocation_observations"] = diagnostic_path
            return original_save(
                path,
                {
                    **value,
                    "host_allocation_observations": {
                        "path": EVIDENCE,
                        "sha256": matched.digest(diagnostic_path),
                        "bytes": diagnostic_path.stat().st_size,
                    },
                },
            )

        def write_receipt(path, **kwargs):
            matched.require(len(diagnostic_artifacts) == 1, "Missing allocation observations")
            kwargs["artifacts"].update(diagnostic_artifacts)
            return original_receipt(path, **kwargs)

        timer.DeepSeekEchoModel = AllocationModel
        timer.source_identity = source_identity
        timer.runtime = runtime
        timer.save = save
        timer.write_receipt = write_receipt

    matched.KIND = KIND
    matched.apply_environment = apply_environment
    matched.main()


if __name__ == "__main__":
    main()
