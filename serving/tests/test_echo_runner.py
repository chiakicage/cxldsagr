"""CPU ownership checks; fake pools do not represent offload execution."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from serving.echo_runner import EchoPrefix, EchoPrefixRunner, open_echo_runner


class EchoRunnerTests(unittest.TestCase):
    def make_runner(self, *, free=8192, device_capacity=None):
        pool = SimpleNamespace()
        if device_capacity is not None:
            pool.device_pool = SimpleNamespace(size=device_capacity)
        backend = SimpleNamespace(
            page_size=64,
            model_config=SimpleNamespace(
                context_len=8192, hf_config=SimpleNamespace(vocab_size=100)
            ),
            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: free, free=Mock()),
            token_to_kv_pool=pool,
        )
        runner = EchoPrefixRunner(backend, model_instance_id="test-instance", prefill_chunk=1024)
        return runner

    def make_prefix(self, runner):
        prefix = EchoPrefix((1,) * 64, object(), (object(),), runner)
        runner._prefixes.add(prefix)
        return prefix

    def test_capacity_rejects_before_gpu_work(self):
        runner = self.make_runner(free=128, device_capacity=256)
        self.assertEqual(runner._check_capacity(128, 64), 64)
        with self.assertRaises(MemoryError):
            runner._check_capacity(129, 64)
        with self.assertRaises(MemoryError):
            runner._check_capacity(320, 256)
        with self.assertRaises(ValueError):
            runner._check_capacity(64, 64)
        with self.assertRaises(ValueError):
            self.make_runner(free=16384)._check_capacity(8193, 0)

    def test_invalid_backend_options_need_no_gpu_imports(self):
        for kwargs in (
            {"mode": "echo"},
            {"mode": "dense_attention"},
            {"mode": "resident", "dense_prefetch_schedule": "layer_end"},
            {"mode": "dense_prefetch", "dense_prefetch_schedule": "invalid"},
            {"mode": "resident", "dense_prefetch_transport": "gpu_direct"},
            {"mode": "dense_prefetch", "dense_prefetch_transport": "invalid"},
            {"prefill_chunk": 65},
            {"max_total_tokens": 1024},
            {"mode": "sparse_sync", "max_total_tokens": 8192},
            {"mode": "echo_gr_adapted", "device_cache_tokens": 64},
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises(ValueError),
                open_echo_runner(model_path=Path("unused"), echo_path=Path("unused"), **kwargs),
            ):
                self.fail("invalid configuration was accepted")

    def test_release_rejects_other_owner_and_repeated_release(self):
        runner = self.make_runner()
        prefix = self.make_prefix(runner)
        with self.assertRaises(ValueError):
            self.make_runner().release(prefix)
        locations = prefix.locations
        runner.release(prefix)
        runner.runner.token_to_kv_pool_allocator.free.assert_called_once_with(locations)
        self.assertTrue(prefix.released)
        self.assertIsNone(prefix.locations)
        with self.assertRaises(ValueError):
            runner.release(prefix)

    def test_candidate_restores_predictor_and_frees_only_suffix(self):
        runner = self.make_runner()
        prefix = self.make_prefix(runner)
        destination = Mock()
        runner._predictors = lambda: (destination,)
        suffix_locations, hidden = object(), object()
        runner._forward = Mock(return_value=(hidden, suffix_locations))
        for candidate in ([2, 3], [4], [2, 3]):
            self.assertIs(runner.extend(prefix, candidate), hidden)
            runner._forward.assert_called_with(
                prefix.token_ids + tuple(candidate), prefix.locations
            )
        self.assertEqual(destination.copy_.call_count, 3)
        self.assertEqual(runner.runner.token_to_kv_pool_allocator.free.call_count, 3)
        runner.runner.token_to_kv_pool_allocator.free.assert_called_with(suffix_locations)
        self.assertFalse(prefix.released)

    def test_reclamation_failure_poisoning_prevents_double_free(self):
        runner = self.make_runner()
        prefix = self.make_prefix(runner)
        free = runner.runner.token_to_kv_pool_allocator.free
        free.side_effect = RuntimeError("partial free")
        with self.assertRaisesRegex(RuntimeError, "partial free"):
            runner.release(prefix)
        with self.assertRaisesRegex(RuntimeError, "closed or failed"):
            runner.extend(prefix, [2])
        runner.close()
        runner.close()
        free.assert_called_once()
        self.assertTrue(prefix.released)
        self.assertIsNone(runner.runner)

    def test_predictor_failure_poisoning(self):
        runner = self.make_runner()
        prefix = self.make_prefix(runner)
        destination = Mock()
        destination.copy_.side_effect = RuntimeError("copy failed")
        runner._predictors = lambda: (destination,)
        runner._forward = Mock()
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            runner.extend(prefix, [2])
        runner._forward.assert_not_called()
        self.assertTrue(runner._failed)
        runner.close()

    def test_close_failure_still_invalidates_handles(self):
        runner = self.make_runner()
        runner.dense_controller = object()
        prefixes = [self.make_prefix(runner), self.make_prefix(runner)]
        runner.runner.token_to_kv_pool_allocator.free.side_effect = RuntimeError("free failed")
        with self.assertRaisesRegex(RuntimeError, "free failed"):
            runner.close()
        self.assertTrue(runner._closed)
        self.assertTrue(all(prefix.released for prefix in prefixes))
        self.assertIsNone(runner.runner)
        self.assertIsNone(runner.dense_controller)

    def test_healthy_close_detaches_dense_controller(self):
        runner = self.make_runner()
        runner.dense_controller = object()
        runner.close()
        self.assertIsNone(runner.dense_controller)
        runner.close()


if __name__ == "__main__":
    unittest.main()
