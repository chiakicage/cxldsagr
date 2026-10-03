import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.gr_cache_serving.src.replay import (
    distribution,
    load_trace,
    summarize,
    timed_phases,
)
from experiments.gr_cache_serving.src.report_replay import load_runs


class ReplayTests(unittest.TestCase):
    def write_run(self, directory, mode="echo_gr_adapted"):
        directory.mkdir()
        summary = {
            "artifact_type": "synchronous_gr_prototype_replay",
            "mode": mode,
            "requests": 1,
            "trace_metadata": {"requests_sha256": "same trace", "stats": {"requests": 1}},
            "topk_order": "native; no correctness-test sorting hooks",
            "source_sha256": {"source": "same version"},
            "measurement": {"timer": "same boundary"},
            "host_users_capacity": 128,
            "configured_device_cache_tokens": 66624,
        }
        rows = [{"ordinal": 0, "user_id": 7, "first_visit": True, "prefix_reused": False}]
        (directory / "summary.json").write_text(json.dumps(summary))
        (directory / "requests.jsonl").write_text(json.dumps(rows[0]) + "\n")
        return summary, rows

    def test_report_accepts_identical_workload_and_timing_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_run(root / "echo")
            self.write_run(root / "sparse", "sparse_sync")
            self.assertEqual(len(load_runs([root / "echo", root / "sparse"])), 2)

    def test_report_rejects_diagnostics_incomplete_and_controlled_timings(self):
        changes = (
            ("artifact_type", "diagnostic_only"),
            ("requests", 2),
            ("topk_order", "logical"),
        )
        for key, value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run"
                summary, _ = self.write_run(path)
                summary[key] = value
                (path / "summary.json").write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    load_runs([path])

    def test_report_rejects_changed_source_or_cache_outcomes(self):
        for changed in ("source", "outcome"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_run(root / "a")
                summary, rows = self.write_run(root / "b")
                if changed == "source":
                    summary["source_sha256"] = {"source": "another version"}
                    (root / "b/summary.json").write_text(json.dumps(summary))
                else:
                    rows[0]["prefix_reused"] = True
                    (root / "b/requests.jsonl").write_text(json.dumps(rows[0]) + "\n")
                with self.assertRaises(ValueError):
                    load_runs([root / "a", root / "b"])

    def test_linear_percentiles_and_empty_groups(self):
        self.assertEqual(distribution([10, 20, 30])["p50_ms"], 20)
        self.assertEqual(distribution([10, 20, 30])["p95_ms"], 29)
        self.assertIsNone(distribution([])["p99_ms"])

    def test_first_visit_is_not_equivalent_to_cache_miss(self):
        rows = [
            {
                "user_id": 0,
                "first_visit": True,
                "prefix_reused": False,
                "service_ms": 100,
                "prefill_ms": 90,
                "candidate_ms": 5,
                "evicted_user_ids": [],
            },
            {
                "user_id": 0,
                "first_visit": False,
                "prefix_reused": True,
                "service_ms": 10,
                "prefill_ms": 0,
                "candidate_ms": 5,
                "evicted_user_ids": [],
            },
            {
                "user_id": 0,
                "first_visit": False,
                "prefix_reused": False,
                "service_ms": 100,
                "prefill_ms": 90,
                "candidate_ms": 5,
                "evicted_user_ids": [1],
            },
        ]
        stats = summarize(rows)
        self.assertEqual(stats["first_visit"]["count"], 1)
        self.assertEqual(stats["prefix_hit"]["count"], 1)
        self.assertEqual(stats["revisit_reprefill"]["count"], 1)
        self.assertEqual(stats["revisit"]["count"], 2)
        self.assertEqual(stats["revisit"]["mean_ms"], 55)
        self.assertEqual(stats["revisit_prefix_hit_fraction"], 0.5)
        self.assertEqual(stats["prefill_count"], 2)
        self.assertEqual(stats["prefill_total_ms"], 180)
        self.assertEqual(stats["prefill"]["count"], 2)
        self.assertEqual(stats["host_evictions"], 1)
        self.assertAlmostEqual(stats["inverse_mean_service_requests_per_s"], 3000 / 210)

    def test_phase_wrapper_restores_after_exception(self):
        prefill = lambda value: value + 1
        extend = lambda value: value + 2
        runner = SimpleNamespace(prefill=prefill, extend=extend)
        with self.assertRaisesRegex(RuntimeError, "stop"), timed_phases(runner) as state:
            self.assertEqual(runner.prefill(1), 2)
            self.assertEqual(runner.extend(1), 3)
            self.assertGreaterEqual(state["prefill_ms"], 0)
            raise RuntimeError("stop")
        self.assertIs(runner.prefill, prefill)
        self.assertIs(runner.extend, extend)

    def test_corrupt_trace_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "stable_prefix_tokens": 2,
                        "candidate_suffix_tokens": 1,
                        "requests_sha256": "bad",
                        "stats": {"requests": 1},
                    }
                )
            )
            path = root / "requests.jsonl"
            path.write_text(
                json.dumps({"user_id": 0, "stable_prefix_tokens": 2, "input_ids": [1, 2, 3]}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                load_trace(path)


if __name__ == "__main__":
    unittest.main()
