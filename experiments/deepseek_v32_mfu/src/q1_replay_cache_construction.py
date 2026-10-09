"""Construct four cache methods before the unchanged matched HBM timer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.deepseek_v32_mfu.src import q1_replay_matched as matched

KIND = "deepseek-q1-replay-cache-construction-v1"
METHODS = ("hbm", "echo", "serial_sparse", "dense_prefetch")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--formal-profile",
        type=Path,
        default=matched.EXPERIMENT / "output/data" / matched.FORMAL_ID,
    )
    options, _ = parser.parse_known_args()
    formal = json.loads((options.formal_profile / "result.json").read_text())
    matched.require(
        formal["methods"] == list(METHODS)
        and formal["warmups"] == 1
        and formal["extend_residency"] == "cold"
        and formal["compute_graphs"] is True
        and formal["extend_graph"] is True,
        "Unexpected formal cache-construction reference",
    )
    source_path = Path(__file__).resolve()
    source_relative = str(source_path.relative_to(matched.ROOT))
    source_sha256 = matched.digest(source_path)
    original_apply = matched.apply_environment

    def apply_environment(expected):
        original_apply(expected)
        from experiments.deepseek_v32_mfu.src import q1_replay_timer as timer

        original_source, original_runtime = timer.source_identity, timer.runtime
        from experiments.deepseek_v32_mfu.src import profile_layers

        matched.require(tuple(profile_layers.METHODS) == METHODS, "Formal method order changed")
        source_ready = False
        started = False
        completed = None

        def source_identity(args):
            nonlocal source_ready
            result = original_source(args)
            matched.require(
                matched.digest(source_path) == source_sha256, "Cache-construction source changed"
            )
            source_ready = True
            result["sources"][source_relative] = source_sha256
            result["contract"]["cache_construction"] = {
                "kind": KIND,
                "methods": list(METHODS),
                "final_method": "hbm",
                "position": "after the one compute bank; before the original reduced HBM prefix",
                "model_forwards": 0,
                "prefix_snapshots_or_restores": 0,
                "extend_graph_preparations": 0,
                "collection": "stopped by the unchanged timer; no added profiler ranges",
                "boundary": "Construct and release the four fresh cache methods, then return "
                "to HBM without prelude prefix or forward execution. Cache/native/GPU-buffer/"
                "stream initialization is one package, not an isolated allocation cause. "
                "The measured reduced prefix, capture and replay remain unchanged. "
                "No added flush, statistics query or statistics reset.",
            }
            return result

        class CacheConstructionModel(timer.DeepSeekEchoModel):
            def prepare_compute_graphs(self, query_sizes):
                nonlocal started, completed
                matched.require(
                    not started and source_ready, "Cache construction must run once after identity"
                )
                matched.require(query_sizes == [1024, 1], "Unexpected compute bank query sizes")
                started = True
                result = super().prepare_compute_graphs(query_sizes)
                bank = self._compute_graphs
                generation = self._cache_generation
                matched.require(bank is not None and self.length == 0, "Unexpected initial bank")
                selected = []
                for method in (*METHODS, "hbm"):
                    profile_layers.select_cache_method(self, method)
                    selected.append(method)
                    matched.require(
                        self.length == 0
                        and self.cache_method == method
                        and self.offload == (method != "hbm")
                        and self._compute_graphs is bank
                        and self._cache_generation == generation + len(selected)
                        and not self._extend_graphs
                        and self._extend_graph is None,
                        "Cache selection changed the empty-cache or compute-bank contract",
                    )
                matched.require(
                    not self._shared_pools
                    and not self._shared_sessions
                    and not self._pool_prefetch,
                    "Final HBM cache retains shared offload resources",
                )
                completed = {
                    "selected_methods": selected,
                    "compute_bank_preparations": 1,
                    "cache_generation_increase": 5,
                    "same_compute_bank": True,
                    "all_selected_caches_empty": True,
                    "fresh_empty_hbm_cache": True,
                    "shared_pools_sessions_helpers_empty": True,
                    "extend_graphs_empty_after_each_selection": True,
                }
                return result

        def runtime():
            matched.require(completed is not None, "Cache construction did not complete")
            result = original_runtime()
            result["cache_construction"] = completed
            return result

        timer.DeepSeekEchoModel = CacheConstructionModel
        timer.source_identity = source_identity
        timer.runtime = runtime

    matched.KIND = KIND
    matched.apply_environment = apply_environment
    matched.main()


if __name__ == "__main__":
    main()
