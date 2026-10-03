import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.gr_cache_serving.src.fixed_budget import (
    PROFILE,
    lru_outcomes,
    validate_configuration,
    validate_outcomes,
    validate_trace,
)
from experiments.gr_cache_serving.src.host_memory import plan_host_cache
from experiments.gr_cache_serving.src.replay import MODES, summarize
from experiments.gr_cache_serving.src.report_capacity import load_capacity_runs
from experiments.gr_cache_serving.src.report_capacity import main as report_main
from serving.echo_budget import plan_echo_budget


def trace_metadata(population):
    return {
        "artifact_type": "prepared_gr_workload",
        "stable_prefix_tokens": 65536,
        "candidate_suffix_tokens": 1024,
        "stats": {"population_users": population, "requests": 512},
        "requests_sha256": f"fixture-{population}",
        "generator": {
            "model": "deepseek_v32",
            "curve_dataset": "beauty",
            "schedule_config": {"seed": 42, "sampling": "weighted", "arrival": "poisson"},
        },
    }


def budget(mode):
    return plan_echo_budget(
        mode=mode,
        num_layers=5,
        users=128,
        prefix_tokens=65536,
        suffix_tokens=1024,
        checkpoint_stored_bytes=26700455104,
    )


def request_rows(population, capacity):
    ids = [i % population for i in range(512)]
    seen, rows = set(), []
    for ordinal, (uid, (hit, evicted, retained)) in enumerate(
        zip(ids, lru_outcomes(ids, capacity), strict=True)
    ):
        rows.append(
            {
                "ordinal": ordinal,
                "user_id": uid,
                "first_visit": uid not in seen,
                "prefix_reused": hit,
                "evicted_user_ids": evicted,
                "retained_prefix_tokens": retained * 65536,
                "prefill_ms": 0 if hit else 90,
                "candidate_ms": 10,
                "service_ms": 10 if hit else 100,
                "d2h_mla_payload_bytes": 0,
            }
        )
        seen.add(uid)
    return rows


class FixedBudgetTests(unittest.TestCase):
    def test_workload_population_does_not_change_cache_sizes(self):
        for population in (64, 128, 256, 384, 512):
            for mode in MODES:
                args = SimpleNamespace(
                    layers=5,
                    host_users=128,
                    device_cache_tokens=66624,
                    hbm_budget_gib=72,
                    host_cache_budget_gib=66,
                    workspace_reserve_gib=8,
                    non_torch_reserve_gib=2,
                    collect_transfers=False,
                    repetition=1,
                    mode=mode,
                )
                plan = budget(mode)
                validate_configuration(args, trace_metadata(population), plan)
                self.assertEqual(plan.retained_users_capacity, 93 if mode == "resident" else 128)
                args.host_users = population + 1
                with self.assertRaises(ValueError):
                    validate_configuration(args, trace_metadata(population), plan)

    def test_fixed_request_count_and_generator_contract(self):
        for population in (64, 128, 256, 384, 512):
            metadata = trace_metadata(population)
            validate_trace(metadata)
            metadata["stats"]["requests"] = 511
            with self.assertRaises(ValueError):
                validate_trace(metadata)
        metadata = trace_metadata(128)
        metadata["generator"]["schedule_config"]["seed"] = 1
        with self.assertRaises(ValueError):
            validate_trace(metadata)

    def test_lru_counts_evicted_revisits_and_checks_every_outcome(self):
        rows = request_rows(129, 128)
        validate_outcomes(rows, 128)
        self.assertGreater(summarize(rows)["revisit_reprefill"]["count"], 0)
        rows[129]["prefix_reused"] = True
        with self.assertRaisesRegex(ValueError, "request 129"):
            validate_outcomes(rows, 128)

    def make_run(self, root, population, mode):
        plan = budget(mode)
        host = plan_host_cache(
            users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=66 * 2**30
        )
        rows = request_rows(population, plan.retained_users_capacity)
        sha = hashlib.sha256(b"fixture").hexdigest()
        summary = summarize(rows) | {
            "artifact_type": "synchronous_gr_prototype_replay",
            "mode": mode,
            "experiment_profile": PROFILE,
            "topk_order": "native; no correctness-test sorting hooks",
            "runner_provenance": {"num_layers": 5, "checkpoint_metadata_sha256": "weights"},
            "trace_metadata": trace_metadata(population),
            "transfer_instrumentation": {"enabled": False},
            "budget_plan": plan.metadata(),
            "host_budget_plan": host,
            "host_users_capacity": 128,
            "host_cache_capacity_users": 0 if mode == "resident" else 128,
            "retained_users_capacity": plan.retained_users_capacity,
            "workload_population_users": population,
            "host_availability_preflight": {"checked": True},
            "host_memory_observation": {
                "peak_process_rss_bytes": 70 * 2**30,
                "samples": 513,
                "peak_pinned_reserved_bytes": 0 if mode == "resident" else 64 * 2**30,
                "charged_cpu_metadata_bytes": 1024,
            },
            "pool_accounting": {
                "host_mla_buffers_bytes": 0 if mode == "resident" else host["main_kv_tensor_bytes"]
            },
            "memory_guard": {
                "peak_sampled_device_bytes": 69 * 2**30,
                "common_total_hbm_limit_bytes": 72 * 2**30,
            },
            "source_sha256": {"code": sha},
            "peak_torch_allocated_bytes": 60 * 2**30,
            "peak_torch_reserved_bytes": 69 * 2**30,
            "process_max_rss_kib": 70 * 2**20,
            "measurement": "same",
            "gpu": "same GPU",
        }
        path = root / f"u{population}_{mode}"
        path.mkdir()
        (path / "summary.json").write_text(json.dumps(summary))
        (path / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        (path / "source_snapshot.json").write_text(json.dumps({"code": {"text": "fixture"}}))
        return path

    def test_report_groups_by_population_not_the_fixed_host_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [self.make_run(Path(temp), pop, mode) for pop in (64, 256) for mode in MODES]
            self.assertEqual(len(load_capacity_runs(paths)), 8)
            file = paths[-1] / "summary.json"
            summary = json.loads(file.read_text())
            summary["host_cache_capacity_users"] = 256
            file.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "fixed-cache budget"):
                load_capacity_runs(paths)

    def test_report_rejects_falsified_lru_even_when_summary_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_run(Path(temp), 256, "sparse_sync")
            rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
            rows[256]["prefix_reused"] = True
            rows[256]["prefill_ms"] = 0
            (path / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            file = path / "summary.json"
            file.write_text(json.dumps(json.loads(file.read_text()) | summarize(rows)))
            with self.assertRaisesRegex(ValueError, "LRU"):
                load_capacity_runs([path], allow_partial=True)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "requires the report environment")
    def test_report_renders_fixed_budget_figures_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [self.make_run(root, 256, mode) for mode in MODES]
            with patch("builtins.print"):
                report_main([*(str(path) for path in paths), "--output-dir", str(root / "report")])
            for name in ("overview.png", "u256_latency.png", "summary.csv", "manifest.json"):
                self.assertGreater((root / "report" / name).stat().st_size, 0)
            manifest = json.loads((root / "report" / "manifest.json").read_text())
            self.assertEqual(manifest["experiment_profile"], PROFILE)


if __name__ == "__main__":
    unittest.main()
