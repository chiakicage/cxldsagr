"""CPU lifecycle/policy checks, not ECHO execution or performance evidence."""

import hashlib
import json
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from serving.echo_cache_manager import EchoCacheManager


@dataclass(eq=False)
class FakePrefix:
    token_ids: tuple[int, ...]
    released: bool = False


class FakeRunner:
    def __init__(self, *, capacity=512, device_capacity=None, resident=False):
        pool = SimpleNamespace()
        if not resident:
            pool.free_req_device_pool = object()
        if device_capacity is not None:
            pool.device_pool = SimpleNamespace(size=device_capacity)
        self.runner = SimpleNamespace(
            page_size=64,
            model_config=SimpleNamespace(
                context_len=2048, hf_config=SimpleNamespace(vocab_size=100)
            ),
            token_to_kv_pool=pool,
            token_to_kv_pool_allocator=SimpleNamespace(size=capacity),
        )
        self.model_instance_id = "test-instance"
        self._prefixes = set()
        self._failed = False
        self._closed = False
        self.calls = []
        self.fail_on = None

    def _ensure_alive(self):
        if self._failed or self._closed:
            raise RuntimeError("runner is failed or closed")

    def _call(self, operation, *args):
        self._ensure_alive()
        self.calls.append((operation, *args))
        if self.fail_on == operation:
            raise RuntimeError(f"{operation} failed with possibly unfinished work")

    def prefill(self, token_ids):
        self._call("prefill", tuple(token_ids))
        prefix = FakePrefix(tuple(token_ids))
        self._prefixes.add(prefix)
        return prefix

    def extend(self, prefix, candidate_ids):
        self._call("extend", prefix, tuple(candidate_ids))
        if prefix not in self._prefixes or prefix.released:
            raise ValueError("invalid prefix")
        return prefix.token_ids + tuple(candidate_ids)

    def release(self, prefix):
        self._call("release", prefix)
        self._prefixes.remove(prefix)
        prefix.released = True

    def evict_hbm(self, prefix):
        self._call("evict_hbm", prefix)

    def count(self, operation):
        return sum(call[0] == operation for call in self.calls)


def request(uid=0, *, token=1, candidate=(2, 3), prefix_size=64, **metadata):
    row = {
        "model": "deepseek_v32",
        "user_id": uid,
        "input_ids": [token] * prefix_size + list(candidate),
        "stable_prefix_tokens": prefix_size,
    }
    row.update(metadata)
    return row


class EchoCacheManagerTests(unittest.TestCase):
    def make_manager(self, *, capacity=512, **kwargs):
        runner = FakeRunner(capacity=capacity)
        manager = EchoCacheManager(runner, host_capacity_tokens=capacity, **kwargs)
        self.addCleanup(manager.close)
        return manager, runner

    def test_revisit_keeps_candidate_private_and_does_not_deduplicate_users(self):
        manager, runner = self.make_manager()
        first = manager.execute(request())
        other = manager.execute(request(1))
        again = manager.execute(request(candidate=(4, 5, 6)))
        self.assertFalse(first.prefix_reused)
        self.assertFalse(other.prefix_reused)
        self.assertTrue(again.prefix_reused)
        self.assertEqual(runner.count("prefill"), 2)
        self.assertEqual(len(runner._prefixes), 2)
        self.assertEqual(again.hidden_states, (1,) * 64 + (4, 5, 6))
        self.assertEqual(again.retained_tokens, 128)
        self.assertEqual(manager.cached_user_ids, (1, 0))
        self.assertEqual(manager.observed_frequency, {0: 2.0, 1: 1.0})
        self.assertEqual(again.observed_frequency, 2.0)
        self.assertEqual(first.prefix_key, other.prefix_key)
        self.assertEqual(
            first.prefix_key.stable_prefix_sha256,
            hashlib.sha256(json.dumps([1] * 64, separators=(",", ":")).encode()).hexdigest(),
        )
        self.assertEqual(runner.count("evict_hbm"), 0)

    def test_full_prefix_and_model_identity_invalidate_same_user(self):
        manager, runner = self.make_manager()
        first = manager.execute(request(history_sha256="unchanged"))
        changed = request(history_sha256="unchanged")
        changed["input_ids"][0] = 7
        second = manager.execute(changed)
        self.assertFalse(second.prefix_reused)
        self.assertNotEqual(first.prefix_key, second.prefix_key)
        self.assertEqual(second.evicted_user_ids, (0,))
        runner.model_instance_id = "new-model-precision-or-rope"
        third = manager.execute(changed)
        self.assertFalse(third.prefix_reused)
        self.assertNotEqual(second.prefix_key, third.prefix_key)
        self.assertEqual(runner.count("release"), 2)
        self.assertEqual(runner.count("prefill"), 3)

    def test_gr_boundaries_are_validated_without_using_common_candidate_prefix(self):
        manager, runner = self.make_manager()
        row = request(
            instruction_tokens=3,
            user_tokens=61,
            item_tokens=5,
            candidate_suffix_tokens=2,
            total_input_tokens=66,
            history_token_span=[3, 64],
            candidate_token_span=[64, 66],
            attention_mask=[1] * 66,
            common_prefix_tokens=65,
        )
        manager.execute(row)
        manager.execute(row)
        self.assertEqual(runner.calls[-1][2], (2, 3))
        self.assertEqual(manager.retained_tokens, 64)

    def test_host_lru_reserves_suffix_and_guard_and_reprefills_evicted_user(self):
        manager, runner = self.make_manager(capacity=256)
        manager.execute(request(0))
        manager.execute(request(1))
        manager.execute(request(0, candidate=(4,)))
        result = manager.execute(request(2))
        self.assertEqual(result.evicted_user_ids, (1,))
        self.assertEqual(manager.cached_user_ids, (0, 2))
        # 65 suffix tokens require two pages plus one guard, evicting user 2.
        result = manager.execute(request(0, candidate=(4,) * 65))
        self.assertTrue(result.prefix_reused)
        self.assertEqual(result.evicted_user_ids, (2,))
        self.assertEqual(result.retained_tokens, 64)
        result = manager.execute(request(1))
        self.assertFalse(result.prefix_reused)
        self.assertEqual(runner.count("prefill"), 4)

    def test_observed_heat_decays_without_oracle_metadata_access(self):
        class OracleGuard(dict):
            def __getitem__(self, key):
                if key == "user_heat_weight":
                    raise AssertionError("oracle heat was inspected")
                return super().__getitem__(key)

            def get(self, key, *default):
                if key == "user_heat_weight":
                    raise AssertionError("oracle heat was inspected")
                return super().get(key, *default)

        manager, _ = self.make_manager(heat_decay_interval=4)
        for uid in (0, 0, 1, 2):
            manager.execute(OracleGuard(request(uid, user_heat_weight=object())))
        self.assertEqual(manager.observed_frequency, {0: 1.0, 1: 0.5, 2: 1.0})
        snapshot = manager.observed_frequency
        snapshot[0] = 999
        self.assertEqual(manager.observed_frequency[0], 1.0)

    def test_hbm_retention_is_observed_frequency_then_recency_not_promotion(self):
        manager, runner = self.make_manager(hbm_retained_users=1)
        manager.execute(request(0))
        manager.execute(request(0))
        manager.execute(request(1))
        user_one = runner.calls[-1][1]
        self.assertEqual(runner.calls[-1][0], "evict_hbm")
        self.assertEqual(runner.count("evict_hbm"), 1)
        manager.execute(request(1))
        self.assertEqual(runner.count("evict_hbm"), 2)
        self.assertIsNot(runner.calls[-1][1], user_one)
        # User 0 loses the recency tie. Its next request raises its frequency,
        # but the policy only drops user 1; it issues no prefetch/promotion call.
        manager.execute(request(0))
        self.assertEqual(runner.count("evict_hbm"), 3)
        self.assertIs(runner.calls[-1][1], user_one)

    def test_zero_hbm_retention_drops_each_completed_branch(self):
        manager, runner = self.make_manager(hbm_retained_users=0)
        manager.execute(request())
        manager.execute(request(candidate=(4,)))
        self.assertEqual(runner.count("evict_hbm"), 2)
        self.assertEqual(runner.count("prefill"), 1)

    def test_three_user_host_lru_is_independent_of_online_hbm_retention(self):
        manager, runner = self.make_manager(capacity=256, hbm_retained_users=1)
        manager.execute(request(0))
        manager.execute(request(0))
        manager.execute(request(1))
        self.assertEqual(runner.count("evict_hbm"), 1)
        result = manager.execute(request(2))
        # Host uses recency even though user 0 has the highest observed count.
        self.assertEqual(result.evicted_user_ids, (0,))
        self.assertEqual(manager.cached_user_ids, (1, 2))
        self.assertEqual(manager.observed_frequency, {0: 2.0, 1: 1.0, 2: 1.0})
        result = manager.execute(request(0))
        self.assertFalse(result.prefix_reused)
        self.assertEqual(result.evicted_user_ids, (1,))
        self.assertEqual(result.observed_frequency, 3.0)
        self.assertEqual(manager.cached_user_ids, (2, 0))
        self.assertEqual(runner.count("evict_hbm"), 2)
        self.assertEqual(runner.count("prefill"), 4)

    def test_constructor_rejects_impossible_capacity_and_policy(self):
        for kwargs in (
            {"host_capacity_tokens": 513},
            {"host_capacity_tokens": 576},
            {"host_capacity_tokens": True},
            {"host_capacity_tokens": 512, "heat_decay_interval": 0},
            {"host_capacity_tokens": 512, "hbm_retained_users": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EchoCacheManager(FakeRunner(), **kwargs)
        with self.assertRaisesRegex(ValueError, "authoritative host"):
            EchoCacheManager(
                FakeRunner(resident=True), host_capacity_tokens=512, hbm_retained_users=1
            )
        runner = FakeRunner()
        runner.prefill((1,) * 64)
        with self.assertRaisesRegex(ValueError, "existing prefixes"):
            EchoCacheManager(runner, host_capacity_tokens=512)

    def test_invalid_request_does_not_evict_or_allocate_and_poison_is_local(self):
        invalid = (
            {"model": "nosa"},
            {"user_id": True},
            {"stable_prefix_tokens": 65},
            {"input_ids": [1] * 64},
            {"input_ids": [True] * 66},
            {"input_ids": [100] * 66},
            {"total_input_tokens": 65},
            {"candidate_suffix_tokens": 1},
            {"candidate_token_span": [64, 65]},
            {"candidate_token_span": bytes([64, 66])},
            {"attention_mask": [1] * 65},
            {"attention_mask": bytes([1] * 66)},
            {"user_tokens": 1},
            {"history_token_span": [0, 64]},
            {"instruction_tokens": 65},
            {"instruction_tokens": 1, "user_tokens": 64},
            {"instruction_tokens": 1, "item_tokens": 2},
            {"instruction_tokens": 1, "history_token_span": [0, 64]},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                manager, runner = self.make_manager(capacity=192)
                manager.execute(request())
                before = list(runner.calls)
                with self.assertRaises(ValueError):
                    manager.execute(request(1, **changes))
                self.assertEqual(runner.calls, before)
                self.assertEqual(manager.cached_user_ids, (0,))
                self.assertFalse(runner._failed)
                with self.assertRaisesRegex(RuntimeError, "closed or failed"):
                    manager.execute(request())
                manager.close()
                self.assertEqual(runner.count("release"), 1)

    def test_limits_are_checked_before_existing_prefix_is_evicted(self):
        for limit in ("host", "device", "context"):
            with self.subTest(limit=limit):
                runner = FakeRunner(capacity=256)
                manager = EchoCacheManager(runner, host_capacity_tokens=256)
                self.addCleanup(manager.close)
                manager.execute(request())
                if limit == "device":
                    runner.runner.token_to_kv_pool.device_pool = SimpleNamespace(size=64)
                if limit == "context":
                    runner.runner.model_config.context_len = 64
                before = list(runner.calls)
                candidate = (2,) * 129 if limit == "host" else (2,)
                with self.assertRaises((MemoryError, ValueError)):
                    manager.execute(request(1, candidate=candidate))
                self.assertEqual(runner.calls, before)

    def test_failed_first_request_is_never_published_or_reclaimed_unsafely(self):
        for operation in ("prefill", "extend", "evict_hbm"):
            with self.subTest(operation=operation):
                manager, runner = self.make_manager(hbm_retained_users=0)
                runner.fail_on = operation
                with self.assertRaisesRegex(RuntimeError, f"{operation} failed"):
                    manager.execute(request())
                self.assertEqual(manager.cached_user_ids, ())
                self.assertEqual(manager.retained_tokens, 0)
                self.assertTrue(runner._failed)
                with self.assertRaisesRegex(RuntimeError, "closed or failed"):
                    manager.execute(request())
                manager.close()
                self.assertEqual(runner.count("release"), 0)

    def test_eviction_failure_stops_before_new_prefill_and_close_never_retries(self):
        manager, runner = self.make_manager(capacity=192)
        manager.execute(request())
        runner.fail_on = "release"
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            manager.execute(request(1))
        self.assertEqual(runner.count("prefill"), 1)
        self.assertEqual(manager.cached_user_ids, (0,))
        self.assertTrue(runner._failed)
        manager.close()
        self.assertEqual(runner.count("release"), 1)

    def test_failed_revisit_never_reuses_or_frees_prefix_again(self):
        manager, runner = self.make_manager()
        manager.execute(request())
        runner.fail_on = "extend"
        with self.assertRaisesRegex(RuntimeError, "extend failed"):
            manager.execute(request())
        self.assertEqual(runner.count("prefill"), 1)
        self.assertEqual(manager.cached_user_ids, (0,))
        self.assertTrue(runner._failed)
        with self.assertRaisesRegex(RuntimeError, "closed or failed"):
            manager.execute(request())
        manager.close()
        self.assertEqual(runner.count("release"), 0)

    def test_close_is_idempotent_and_leaves_healthy_runner_usable(self):
        manager, runner = self.make_manager()
        manager.execute(request(0))
        manager.execute(request(1))
        manager.close()
        manager.close()
        self.assertEqual(runner.count("release"), 2)
        self.assertEqual(runner._prefixes, set())
        self.assertEqual(manager.retained_tokens, 0)
        self.assertFalse(runner._closed)
        prefix = runner.prefill((3,) * 64)
        self.assertEqual(runner.extend(prefix, [4]), (3,) * 64 + (4,))
        runner.release(prefix)
        with self.assertRaisesRegex(RuntimeError, "closed or failed"):
            manager.execute(request())

    def test_close_failure_poisoning_does_not_retry_other_handles(self):
        manager, runner = self.make_manager()
        manager.execute(request(0))
        manager.execute(request(1))
        runner.fail_on = "release"
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            manager.close()
        manager.close()
        self.assertTrue(runner._failed)
        self.assertEqual(runner.count("release"), 1)
        self.assertEqual(manager.cached_user_ids, ())


if __name__ == "__main__":
    unittest.main()
