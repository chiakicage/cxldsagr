import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.gr_cache_serving.src.host_memory import (
    GIB,
    HostCacheGuard,
    available_non_movable_bytes,
    cgroup_memory_limits,
    check_pinned_capacity,
    inspect_host_availability,
    non_movable_bytes,
    parse_nodes,
    plan_host_cache,
)


class HostMemoryTests(unittest.TestCase):
    def test_availability_checks_current_capacity_and_ancestor_quota(self):
        plan = plan_host_cache(
            users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=66 * GIB
        )

        def read(path):
            if path.name == "status":
                return "Mems_allowed_list:\t0,3\n"
            if path.name == "zoneinfo":
                return f"Node 0, zone Normal\nmanaged {128 * GIB // 4096}\npages free {100 * GIB // 4096}\nNode 3, zone Movable\nmanaged {256 * GIB // 4096}\npages free {256 * GIB // 4096}"
            if path.name == "meminfo":
                return f"MemAvailable: {300 * GIB // 1024} kB\n"
            raise AssertionError(path)

        with patch.object(Path, "read_text", read), patch("os.sysconf", return_value=4096):
            with patch(
                "experiments.gr_cache_serving.src.host_memory.cgroup_memory_limits", return_value=[]
            ):
                result = inspect_host_availability(plan)
                self.assertEqual(result["non_movable_free_clean_inactive_bytes"], 100 * GIB)
            with (
                patch(
                    "experiments.gr_cache_serving.src.host_memory.cgroup_memory_limits",
                    return_value=[{"headroom_bytes": 100 * GIB}],
                ),
                self.assertRaisesRegex(MemoryError, "ancestor cgroup"),
            ):
                inspect_host_availability(plan)

    def test_budget_charges_rounding_and_dense_staging_not_only_payload(self):
        plan = plan_host_cache(
            users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=66 * GIB
        )
        self.assertEqual(plan["history_kv_bytes"], 45 * GIB)
        self.assertEqual(plan["main_kv_tensor_bytes"], 48324648960)
        self.assertEqual(plan["main_kv_rounded_pinned_bytes"], 64 * GIB)
        self.assertEqual(plan["dense_staging_rounded_pinned_bytes"], GIB // 4)
        self.assertEqual(
            sum(
                plan[key]
                for key in (
                    "max_pinned_reserved_bytes",
                    "metadata_reserve_bytes",
                    "safety_reserve_bytes",
                )
            ),
            66 * GIB,
        )
        with self.assertRaises(MemoryError):
            plan_host_cache(users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=65 * GIB)

    def test_current_zone_headroom_excludes_movable_dirty_and_high_watermark(self):
        source = """Node 0, zone Normal
pages free 100
high 10
nr_zone_inactive_file 30
nr_zone_write_pending 5
Node 3, zone Movable
pages free 10000
"""
        self.assertEqual(available_non_movable_bytes(source, {0, 3}, 4096), 115 * 4096)

    def test_cgroup_v2_checks_finite_parent_and_high_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = root / "child"
            child.mkdir()
            (child / "memory.max").write_text("max")
            (child / "memory.high").write_text("80")
            (child / "memory.current").write_text("30")
            (root / "memory.max").write_text("100")
            (root / "memory.current").write_text("60")
            result = cgroup_memory_limits("0::/child", f"1 0 0:1 / {root} rw - cgroup2 cgroup rw")
            self.assertEqual([item["headroom_bytes"] for item in result], [50, 40])

    def test_cgroup_v1_and_missing_mount_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "memory.limit_in_bytes").write_text("100")
            (root / "memory.usage_in_bytes").write_text("40")
            result = cgroup_memory_limits(
                "1:memory:/", f"1 0 0:1 / {root} rw - cgroup cgroup rw,memory"
            )
            self.assertEqual(result[0]["headroom_bytes"], 60)
        with self.assertRaises(RuntimeError):
            cgroup_memory_limits("0::/", "")

    def test_runtime_guard_rejects_actual_reserved_peak(self):
        stats = {"allocated_bytes.peak": 64 * GIB, "reserved_bytes.peak": 64 * GIB}
        torch = SimpleNamespace(
            cuda=SimpleNamespace(memory=SimpleNamespace(host_memory_stats=lambda: stats))
        )
        plan = plan_host_cache(
            users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=66 * GIB
        )
        guard = HostCacheGuard(torch, plan)
        runner = SimpleNamespace(runner=SimpleNamespace(token_to_kv_pool=SimpleNamespace()))
        guard.sample(runner)
        self.assertEqual(guard.metadata()["peak_pinned_reserved_bytes"], 64 * GIB)
        stats["reserved_bytes.peak"] = plan["max_pinned_reserved_bytes"] + 16
        guard.sample(runner)
        self.assertEqual(guard.metadata()["pinned_above_main_and_staging_bytes"], 16)
        stats["reserved_bytes.peak"] = 65 * GIB
        with self.assertRaises(MemoryError):
            guard.sample(runner)
        stats["reserved_bytes.peak"] = 64 * GIB
        guard = HostCacheGuard(torch, plan)
        with (
            patch.object(Path, "read_text", return_value=f"VmHWM: {100 * GIB // 1024} kB\n"),
            self.assertRaisesRegex(MemoryError, "RSS"),
        ):
            guard.sample(runner)

    def test_nodes(self):
        self.assertEqual(parse_nodes("0-1,3"), {0, 1, 3})
        with self.assertRaises(ValueError):
            parse_nodes("3-1")

    def test_excludes_movable_device_and_disallowed_nodes(self):
        source = """Node 0, zone Normal
  managed 100
Node 1, zone Normal
  managed 200
Node 3, zone Movable
  managed 1000
Node 3, zone Device
  managed 1000
"""
        self.assertEqual(non_movable_bytes(source, {0, 3}, 4096), 100 * 4096)

    def test_power_of_two_rounding_can_prevent_an_otherwise_fitting_payload(self):
        result = check_pinned_capacity(45 * GIB, 125 * GIB)
        self.assertEqual(result["torch_pinned_allocation_rounded_bytes"], 64 * GIB)
        with self.assertRaisesRegex(MemoryError, "rounds to 128"):
            check_pinned_capacity(90 * GIB, 125 * GIB)
        with self.assertRaisesRegex(MemoryError, "rounds to 256"):
            check_pinned_capacity(180 * GIB, 125 * GIB)

    def test_invalid_or_missing_zone_data_fails_closed(self):
        with self.assertRaises(ValueError):
            non_movable_bytes("", {0}, 4096)
        with self.assertRaises(ValueError):
            non_movable_bytes("Node 0, zone Normal\nmanaged 2\nmanaged 3", {0}, 4096)


if __name__ == "__main__":
    unittest.main()
