"""Run one selected formal warmup before the unchanged matched HBM timer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from experiments.deepseek_v32_mfu.src import q1_replay_matched as matched

KIND = "deepseek-q1-replay-single-method-prelude-v1"
METHODS = ("hbm", "echo", "serial_sparse", "dense_prefetch")


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--prelude-method", choices=METHODS, default="hbm")
    selected, remaining = selector.parse_known_args()
    method = selected.prelude_method
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--formal-profile",
        type=Path,
        default=matched.EXPERIMENT / "output/data" / matched.FORMAL_ID,
    )
    options, _ = parser.parse_known_args(remaining)
    formal = json.loads((options.formal_profile / "result.json").read_text())
    matched.require(
        formal["methods"] == list(METHODS)
        and formal["warmups"] == 1
        and formal["extend_residency"] == "cold"
        and formal["compute_graphs"] is True
        and formal["extend_graph"] is True,
        "Unexpected formal single-method prelude reference",
    )
    prelude_args = SimpleNamespace(extend_residency="cold", extend_graph=True)
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
        token_ids = None
        started = False
        completed = None

        def source_identity(args):
            nonlocal token_ids
            result = original_source(args)
            matched.require(
                matched.digest(source_path) == source_sha256, "Single-method prelude source changed"
            )
            ids = json.loads(args.request.read_text())["input_ids"]
            matched.require(len(ids) == 65537 and ids[-1] == 111090, "Unexpected prelude request")
            if token_ids is None:
                token_ids = ids
            else:
                matched.require(token_ids == ids, "Prelude request changed")
            result["sources"][source_relative] = source_sha256
            result["contract"]["single_method_prelude"] = {
                "kind": KIND,
                "method": method,
                "warmup_count": 1,
                "extend_residency": "cold",
                "position": "after the one compute bank; before the original reduced HBM prefix",
                "final_transition": "select_cache_method(hbm) creates an empty resident cache",
                "collection": "stopped by the unchanged timer; no added profiler ranges",
                "boundary": "Run one selected method's complete formal warmup, then return "
                "to HBM. This varies a method-specific initialization/execution package, "
                "not an isolated DMA, allocation, stream or fence cause. The measured "
                "reduced prefix, capture and replay remain unchanged. No added flush or statistics.",
            }
            return result

        class SingleMethodPreludeModel(timer.DeepSeekEchoModel):
            def prepare_compute_graphs(self, query_sizes):
                nonlocal started, completed
                matched.require(not started and token_ids is not None, "Prelude must run once")
                matched.require(query_sizes == [1024, 1], "Unexpected compute bank query sizes")
                started = True
                result = super().prepare_compute_graphs(query_sizes)
                bank = self._compute_graphs
                generation = self._cache_generation
                matched.require(bank is not None and self.length == 0, "Unexpected initial bank")
                profile_layers.select_cache_method(self, method)
                self.forward(token_ids[:65536])
                snapshot = self.snapshot_prefix()
                profile_layers.restore_extend_prefix(self, snapshot, prelude_args)
                profile_layers.prepare_extend_graph(self, token_ids[65536:], prelude_args)
                self.forward(token_ids[65536:])
                del snapshot
                matched.require(self.length == 65537, "Prelude append did not commit")
                matched.require(self._compute_graphs is bank, "Prelude replaced compute bank")
                profile_layers.select_cache_method(self, "hbm")
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
                    and self._cache_generation == generation + 2,
                    "Prelude did not return to a fresh HBM cache with the same compute bank",
                )
                completed = {
                    "method": method,
                    "warmup_count": 1,
                    "compute_bank_preparations": 1,
                    "cache_generation_increase": 2,
                    "same_compute_bank": True,
                    "fresh_empty_hbm_cache": True,
                    "shared_pools_sessions_helpers_empty": True,
                    "prelude_extend_graphs_closed": True,
                }
                return result

        def runtime():
            matched.require(completed is not None, "Single-method prelude did not complete")
            result = original_runtime()
            result["single_method_prelude"] = completed
            return result

        timer.DeepSeekEchoModel = SingleMethodPreludeModel
        timer.source_identity = source_identity
        timer.runtime = runtime

    matched.KIND = KIND
    matched.apply_environment = apply_environment
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        matched.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
