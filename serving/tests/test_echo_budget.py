import unittest

from serving.echo_budget import GIB, plan_echo_budget


class EchoBudgetTests(unittest.TestCase):
    def plan(self, **updates):
        options = {
            "mode": "resident",
            "num_layers": 5,
            "users": 128,
            "prefix_tokens": 65536,
            "suffix_tokens": 1024,
            "checkpoint_stored_bytes": 26700455104,
        }
        return plan_echo_budget(**(options | updates))

    def test_resident_keeps_small_population_and_bounds_large_population(self):
        small, large = self.plan(users=64), self.plan(users=512)
        self.assertEqual(small.retained_users_capacity, 64)
        self.assertLess(large.retained_users_capacity, 128)
        self.assertGreater(large.retained_users_capacity, 64)
        self.assertEqual(large.logical_pool_tokens % 64, 0)
        self.assertEqual(large.device_cache_tokens, large.logical_pool_tokens)

    def test_offload_index_scales_with_all_host_users_and_dense_is_extra(self):
        small = self.plan(mode="sparse_sync")
        large = self.plan(mode="sparse_sync", users=512)
        dense = self.plan(mode="dense_prefetch", users=512)
        self.assertEqual(large.retained_users_capacity, 512)
        self.assertEqual(large.device_cache_tokens, 66624)
        self.assertGreater(
            large.estimated_persistent_cache_bytes, small.estimated_persistent_cache_bytes
        )
        self.assertGreater(
            dense.estimated_persistent_cache_bytes, large.estimated_persistent_cache_bytes
        )
        self.assertEqual(dense.total_hbm_bytes, self.plan().total_hbm_bytes)
        self.assertEqual(dense.torch_limit_bytes, 70 * GIB)

    def test_impossible_limits_rejected_without_model_import(self):
        for updates in (
            {"total_hbm_bytes": 20 * GIB},
            {"prefix_tokens": 65535},
            {"mode": "bad"},
            {"users": 0},
            {"num_layers": 6},
            {"mode": "sparse_sync", "device_cache_tokens": 1024},
            {"mode": "sparse_sync", "users": 4096},
        ):
            with self.subTest(updates=updates), self.assertRaises((ValueError, MemoryError)):
                self.plan(**updates)


if __name__ == "__main__":
    unittest.main()
