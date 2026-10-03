import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments.gr_cache_serving.src.replay import MODES, summarize
from experiments.gr_cache_serving.src.report_capacity import load_capacity_runs


class CapacityReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.paths = []
        source = "measured source"
        sha = hashlib.sha256(source.encode()).hexdigest()
        for mode in MODES:
            path = root / mode
            path.mkdir()
            self.paths.append(path)
            rows = [
                {
                    "ordinal": i,
                    "user_id": 7,
                    "first_visit": i == 0,
                    "prefix_reused": i > 0 and mode != "resident",
                    "service_ms": 100,
                    "prefill_ms": 90 if i == 0 or mode == "resident" else 0,
                    "candidate_ms": 10,
                    "evicted_user_ids": [],
                }
                for i in range(3)
            ]
            summary = summarize(rows) | {
                "artifact_type": "synchronous_gr_prototype_replay",
                "mode": mode,
                "topk_order": "native; no correctness-test sorting hooks",
                "runner_provenance": {"num_layers": 5, "checkpoint_metadata_sha256": "weights"},
                "trace_metadata": {
                    "stable_prefix_tokens": 65536,
                    "candidate_suffix_tokens": 1024,
                    "stats": {"requests": 3},
                    "requests_sha256": "input",
                },
                "transfer_instrumentation": {"enabled": True},
                "budget_plan": {
                    "total_hbm_bytes": 100,
                    "non_torch_reserve_bytes": 10,
                    "workspace_reserve_bytes": 20,
                },
                "memory_guard": {
                    "peak_sampled_device_bytes": 80,
                    "common_total_hbm_limit_bytes": 100,
                },
                "source_sha256": {"code": sha},
                "host_users_capacity": 128,
                "measurement": "same",
                "gpu": "same GPU",
            }
            (path / "summary.json").write_text(json.dumps(summary))
            (path / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            (path / "source_snapshot.json").write_text(json.dumps({"code": {"text": source}}))

    def test_capacity_outcomes_may_differ_but_all_revisits_are_counted(self):
        runs = load_capacity_runs(self.paths)
        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[0][1]["revisit_reprefill"]["count"], 2)
        self.assertEqual(runs[1][1]["revisit_reprefill"]["count"], 0)

    def test_incomplete_mode_matrix_rejected(self):
        with self.assertRaisesRegex(ValueError, "all four"):
            load_capacity_runs(self.paths[:-1])

    def test_mixed_dense_transports_rejected_even_at_different_populations(self):
        dense = next(p for p in self.paths if p.name == "dense_prefetch")
        second = dense.parent / "dense_direct"
        shutil.copytree(dense, second)
        path = second / "summary.json"
        summary = json.loads(path.read_text())
        summary["trace_metadata"]["stats"]["population_users"] = 256
        summary["runner_provenance"]["dense_prefetch"] = {"transport": "gpu_direct"}
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "different dense transports"):
            load_capacity_runs([dense, second], allow_partial=True)

    def test_partial_audit_does_not_relax_integrity_or_default_report(self):
        self.assertEqual(len(load_capacity_runs(self.paths[:1], allow_partial=True)), 1)
        summary_path = self.paths[0] / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["revisit"]["mean_ms"] = 0
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "statistics"):
            load_capacity_runs(self.paths[:1], allow_partial=True)

    def test_changed_budget_trace_and_statistics_rejected(self):
        path = self.paths[-1] / "summary.json"
        original = json.loads(path.read_text())
        for key, changed in (
            ("budget_plan", original["budget_plan"] | {"total_hbm_bytes": 90}),
            ("trace_metadata", original["trace_metadata"] | {"requests_sha256": "different"}),
            ("revisit", original["revisit"] | {"mean_ms": 1}),
        ):
            with self.subTest(key=key):
                path.write_text(json.dumps(original | {key: changed}))
                with self.assertRaises(ValueError):
                    load_capacity_runs(self.paths)
        path.write_text(json.dumps(original))


if __name__ == "__main__":
    unittest.main()
