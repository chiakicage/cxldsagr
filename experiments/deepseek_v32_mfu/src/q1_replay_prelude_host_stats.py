"""Observe host allocator totals after the unchanged four-method prelude."""

from __future__ import annotations

from pathlib import Path

from experiments.deepseek_v32_mfu.src import q1_replay_prelude as prelude

KIND = "deepseek-q1-replay-prelude-host-stats-v1"
PARENT_KIND = "deepseek-q1-replay-prelude-v1"
EVIDENCE = "prelude_host_allocator_observations.json"
PHASES = (
    "runtime_after_graph_construction_and_five_warmups",
    "runtime_after_check_or_samples",
)


def main():
    matched = prelude.matched
    matched.require(prelude.KIND == PARENT_KIND, "Expected the unchanged four-method driver")
    paths = (
        Path(__file__).resolve(),
        Path(prelude.__file__).resolve(),
        Path(matched.__file__).resolve(),
        Path(__file__).with_name("q1_replay_timer.py").resolve(),
    )
    sources = {str(path.relative_to(matched.ROOT)): matched.digest(path) for path in paths}
    original_apply = matched.apply_environment

    def verify_sources():
        matched.require(
            all(matched.digest(matched.ROOT / name) == value for name, value in sources.items()),
            "Observer or frozen prelude source changed",
        )

    def apply_environment(expected):
        original_apply(expected)
        # The parent installs its unchanged model/source/runtime wrappers after this hook.
        from experiments.deepseek_v32_mfu.src import q1_replay_timer as timer

        original_source, original_runtime = timer.source_identity, timer.runtime
        original_save, original_receipt = timer.save, timer.write_receipt

        import torch

        run = None
        observations = []
        diagnostic_artifacts = {}

        def source_identity(args):
            nonlocal run
            result = original_source(args)
            verify_sources()
            current = {"run_id": args.run_id, "mode": args.mode}
            matched.require(run is None or run == current, "Host statistics run changed")
            run = current
            result["sources"].update(sources)
            result["contract"]["prelude_host_statistics"] = {
                "kind": KIND,
                "parent_kind": PARENT_KIND,
                "prelude_source": str(paths[1].relative_to(matched.ROOT)),
                "callbacks": list(PHASES),
                "observation_count": 2,
                "diagnostic_evidence": EVIDENCE,
                "boundary": "The four-method prelude, compute bank, cache/graph lifecycle and "
                "timer remain unchanged. Read host_memory_stats only at the two existing "
                "runtime callbacks outside collection. No added allocation, stream or flush. "
                "Variable allocator observations are artifacts, not execution identity.",
            }
            return result

        def runtime():
            matched.require(run is not None and len(observations) < 2, "Invalid observer callback")
            result = original_runtime()
            stats = dict(torch.cuda.host_memory_stats())
            matched.require(
                {"allocated_bytes.current", "num_host_alloc", "num_host_free"} <= stats.keys(),
                "Pinned host allocator statistics are unavailable",
            )
            observations.append(
                {
                    "phase": PHASES[len(observations)],
                    "runtime_call": len(observations) + 1,
                    "reported_host_allocator_stats": stats,
                }
            )
            return result

        def save(path, value):
            verify_sources()
            matched.require(
                Path(path).name == "result.json"
                and len(observations) == 2
                and not diagnostic_artifacts,
                "Unexpected final host statistics boundary",
            )
            diagnostic_path = Path(path).with_name(EVIDENCE)
            original_save(
                diagnostic_path,
                {
                    "schema": "q1-prelude-host-allocator-observations-v1",
                    "kind": KIND,
                    **run,
                    "source_sha256": sources,
                    "observations": observations,
                    "boundary": "Reported process host-allocator totals after construction "
                    "and five warmups, then after check or samples. Neither observation is "
                    "immediately after a flush. allocated_bytes includes rounded active and "
                    "cached blocks; active counters are not used to infer exact cached bytes. "
                    "These totals do not identify CUDA allocation APIs, allocation owners, "
                    "or memory outside PyTorch's host allocator, and do not prove a fence cause.",
                },
            )
            diagnostic_artifacts["prelude_host_allocator_observations"] = diagnostic_path
            return original_save(
                path,
                {
                    **value,
                    "prelude_host_allocator_observations": {
                        "path": EVIDENCE,
                        "sha256": matched.digest(diagnostic_path),
                        "bytes": diagnostic_path.stat().st_size,
                    },
                },
            )

        def write_receipt(path, **kwargs):
            matched.require(len(diagnostic_artifacts) == 1, "Missing host allocator observations")
            kwargs["artifacts"].update(diagnostic_artifacts)
            return original_receipt(path, **kwargs)

        timer.source_identity = source_identity
        timer.runtime = runtime
        timer.save = save
        timer.write_receipt = write_receipt

    prelude.KIND = KIND
    matched.apply_environment = apply_environment
    prelude.main()


if __name__ == "__main__":
    main()
