import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.gr_cache_serving.src import sweep_status


class SweepStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.trace = self.base / "group_input_u64"
        self.trace.mkdir()
        payload = b"test fixture; only the prepared checksum is inspected here\n"
        sha = hashlib.sha256(payload).hexdigest()
        self.metadata = {
            "artifact_type": "prepared_gr_workload",
            "stable_prefix_tokens": 65536,
            "candidate_suffix_tokens": 1024,
            "stats": {"population_users": 64, "requests": 256},
            "requests_sha256": sha,
            "users_sha256": sha,
        }
        for name in ("requests", "users"):
            (self.trace / f"{name}.jsonl").write_bytes(payload)
        (self.trace / "metadata.json").write_text(json.dumps(self.metadata))
        self.source = patch.object(sweep_status, "source_fingerprints", return_value={"a": "new"})
        self.source.start()
        self.addCleanup(self.source.stop)

    def audit(self, modes=("resident", "sparse_sync")):
        return sweep_status.audit_sweep(self.base, "group", (64,), modes)

    def test_pending_resident_and_blocked_offload_need_no_replay(self):
        with patch.object(
            sweep_status, "inspect_pinned_capacity", side_effect=MemoryError("too large")
        ) as inspect:
            result = self.audit()
        inspect.assert_called_once_with((64 * 65536 + 1088) * 5 * 1152)
        self.assertEqual(result["counts"], {"complete": 0, "pending": 1, "blocked_host": 1})
        self.assertEqual(result["runs"][1]["reason"], "too large")

    def test_viable_host_check_is_not_a_gpu_readiness_claim(self):
        with patch.object(sweep_status, "inspect_pinned_capacity", return_value={"checked": True}):
            result = self.audit()
        self.assertEqual(result["counts"]["pending"], 2)
        self.assertIn("no GPU readiness claim", result["scope"])

    def test_fixed_profile_uses_512_requests_and_128_user_host_pool(self):
        from experiments.gr_cache_serving.tests.test_fixed_budget import trace_metadata

        metadata = self.metadata | trace_metadata(64)
        metadata["requests_sha256"] = self.metadata["requests_sha256"]
        (self.trace / "metadata.json").write_text(json.dumps(metadata))
        with patch.object(
            sweep_status, "inspect_host_availability", return_value={"checked": True}
        ) as inspect:
            result = sweep_status.audit_sweep(
                self.base, "group", (64,), ("resident", "sparse_sync"), "fixed_budget_512"
            )
        self.assertEqual(result["counts"]["pending"], 2)
        self.assertEqual(inspect.call_args.args[0]["history_kv_bytes"], 45 * 2**30)

    def test_trace_corruption_and_invalid_matrix_rejected(self):
        (self.trace / "users.jsonl").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.audit()
        for group, users, modes in (
            ("../group", (64,), ("resident",)),
            ("group", (64, 64), ("resident",)),
            ("group", (64,), ("resident", "resident")),
            ("group", (65,), ("resident",)),
        ):
            with self.subTest(group=group, users=users, modes=modes), self.assertRaises(ValueError):
                sweep_status.audit_sweep(self.base, group, users, modes)

    def test_partial_directory_is_never_treated_as_completed(self):
        (self.base / "group_u64_resident_r1").mkdir()
        with self.assertRaises(FileNotFoundError):
            self.audit(("resident",))

    def test_completed_runs_use_integrity_validator_and_report_source_changes(self):
        path = self.base / "group_u64_resident_r1"
        path.mkdir()
        summary = {
            "mode": "resident",
            "host_users_capacity": 64,
            "trace_metadata": self.metadata,
            "budget_plan": {"total_hbm_bytes": 72 * 2**30},
            "source_sha256": {"a": "old"},
            "requests": 256,
        }
        with patch.object(
            sweep_status, "load_capacity_runs", return_value=[(path, summary, [])]
        ) as validate:
            result = self.audit(("resident",))
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(result["counts"]["complete"], 1)
        self.assertEqual(result["runs"][0]["source_changes_since_run"], ["a"])

    def test_different_measured_versions_are_audited_separately(self):
        paths = {}
        for mode in ("resident", "sparse_sync"):
            path = self.base / f"group_u64_{mode}_r1"
            path.mkdir()
            paths[path] = {
                "mode": mode,
                "host_users_capacity": 64,
                "trace_metadata": self.metadata,
                "budget_plan": {"total_hbm_bytes": 72 * 2**30},
                "source_sha256": {"a": mode},
                "requests": 256,
            }

        def validate(requested, *, allow_partial):
            self.assertTrue(allow_partial)
            self.assertEqual(len(requested), 1)
            return [(path, paths[path], []) for path in requested]

        with patch.object(sweep_status, "load_capacity_runs", side_effect=validate) as check:
            result = self.audit()
        self.assertEqual(check.call_count, 4)
        self.assertEqual(len(result["measured_source_versions"]), 2)
        self.assertEqual(result["counts"]["complete"], 2)


if __name__ == "__main__":
    unittest.main()
