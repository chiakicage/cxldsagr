"""CPU fakes exercise GR extend recall allocation and method restoration."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models.deepseek_v32.echo_cache import (
    make_extend_recall_with_free_slots,
    scoped_extend_recall_fix,
)


class Scalar(int):
    def to(self, dtype):
        return self


class Flags(list):
    def fill_(self, value):
        self[:] = [value] * len(self)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Flags(result) if isinstance(key, slice) else result

    def sum(self):
        return Scalar(sum(self))


def require_true(condition, message):
    if not condition:
        raise RuntimeError(message)


class EchoCacheTests(unittest.TestCase):
    def make_fixture(self, available, capacity=8):
        calls = []

        class Allocator:
            def __init__(self):
                self.free_slots = available
                self.size_in_tensor = Scalar(capacity)

            def available_size(self):
                return Scalar(self.free_slots)

            def alloc(self, count, indices):
                calls.append(("alloc", int(count)))
                require_true(count <= self.free_slots, "allocation exceeds available slots")
                self.free_slots -= count

        allocator = Allocator()

        def mark(flags, indices):
            calls.append(("mark", tuple(indices)))
            for index in indices:
                flags[index] = 1

        def free(count, layer, protected):
            calls.append(("free", int(count)))
            allocator.free_slots += count
            require_true(allocator.free_slots <= capacity, "freed unoccupied slots")

        host_module = SimpleNamespace(
            mark_host_need_recall=mark,
            recall_update_extend=lambda *args: calls.append(("recall_update", None)),
        )
        torch_module = SimpleNamespace(
            int32="int32",
            _assert_async=require_true,
            clamp_min=lambda value, low: Scalar(max(value, low)),
        )
        pool = SimpleNamespace(
            host_need_recall=Flags([0] * 32),
            start_layer=0,
            device_pool_allocator=[allocator],
            free_device_pool_cuda_graph=free,
            device_pool_loc_alloc_buf=object(),
            update_priority_when_use=lambda *args: calls.append(("priority", None)),
            host_token_to_device=[object()],
            device_token_to_host=[object()],
            device_pool=SimpleNamespace(get_key_buffer=lambda layer: object()),
            kv_buffer=[object()],
            recall_counter=object(),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        return pool, host_module, torch_module, calls

    def test_free_slots_zero_misses_and_eviction_preserve_operation_order(self):
        for available, misses, expected_free, expected_alloc in (
            (3, [1, 2, 2, -1], 0, 2),
            (3, [-1, -1], 0, 0),
            (1, [1, 2, 3, 4], 3, 4),
            (0, [1, 2, 3], 3, 3),
        ):
            with (
                self.subTest(available=available, misses=misses),
                patch.dict(os.environ, {"EXTEND_OFFLOAD_DEBUG": "1"}),
            ):
                pool, host, torch, calls = self.make_fixture(available)
                make_extend_recall_with_free_slots(host, torch)(pool, misses, object(), 0)
                self.assertEqual(
                    [call[0] for call in calls],
                    ["mark", "free", "alloc", "priority", "recall_update"],
                )
                self.assertEqual(calls[1], ("free", expected_free))
                self.assertEqual(calls[2], ("alloc", expected_alloc))

    def test_misses_beyond_total_pool_fail_before_eviction_or_allocation(self):
        pool, host, torch, calls = self.make_fixture(3)
        with self.assertRaisesRegex(RuntimeError, "device-pool capacity"):
            make_extend_recall_with_free_slots(host, torch)(pool, list(range(1, 10)), object(), 0)
        self.assertEqual([call[0] for call in calls], ["mark"])

    def test_invalid_available_count_fails_before_allocator(self):
        for available in (-1, 9):
            with self.subTest(available=available):
                pool, host, torch, calls = self.make_fixture(available)
                with self.assertRaisesRegex(RuntimeError, "device-pool capacity"):
                    make_extend_recall_with_free_slots(host, torch)(pool, [1], object(), 0)
                self.assertEqual([call[0] for call in calls], ["mark"])

    def test_scope_restores_original_method_after_exception(self):
        class Pool:
            def recall_miss_tokens_extend_cuda_graph(self, *args):
                return "original"

        original = Pool.recall_miss_tokens_extend_cuda_graph
        host_module = SimpleNamespace(NSATokenToKVPoolHost=Pool)
        with (
            self.assertRaisesRegex(RuntimeError, "caller failed"),
            scoped_extend_recall_fix(host_module, None) as adaptation,
        ):
            self.assertEqual(adaptation, "gr_extend_recall_free_slots_v1")
            self.assertIsNot(Pool.recall_miss_tokens_extend_cuda_graph, original)
            raise RuntimeError("caller failed")
        self.assertIs(Pool.recall_miss_tokens_extend_cuda_graph, original)


if __name__ == "__main__":
    unittest.main()
