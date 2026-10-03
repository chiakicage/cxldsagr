import threading
import unittest
from contextlib import ExitStack
from types import SimpleNamespace

from experiments.gr_cache_serving.src.profile_parts import PartRanges


class PartRangeTests(unittest.TestCase):
    def make_ranges(self):
        ranges = PartRanges.__new__(PartRanges)
        self.events = []
        ranges.torch = SimpleNamespace(
            cuda=SimpleNamespace(
                nvtx=SimpleNamespace(
                    range_push=lambda label: self.events.append(label),
                    range_pop=lambda: self.events.append("pop"),
                )
            )
        )
        ranges.local = threading.local()
        ranges.stack = ExitStack()
        self.addCleanup(ranges.stack.close)
        ranges.enabled = True
        ranges.ordinal, ranges.phase, ranges.chunk = 8, "candidate", 0
        return ranges

    def test_nested_scope_restores_layer_and_original_method(self):
        ranges = self.make_ranges()
        obj = SimpleNamespace(call=lambda: 42)
        original = obj.call
        ranges.annotate(obj, "call", "ffn_moe", layer=4)
        self.assertEqual(obj.call(), 42)
        self.assertIn("/l=4/s=ffn_moe", self.events[0])
        self.assertEqual(ranges.local.layer, -1)
        ranges.stack.close()
        self.assertIs(obj.call, original)

    def test_exception_balances_nvtx_and_disabled_path_does_not_mark(self):
        ranges = self.make_ranges()
        with self.assertRaises(ValueError), ranges.span("failure"):
            raise ValueError("failure")
        self.assertEqual(self.events[-1], "pop")
        ranges.enabled = False
        count = len(self.events)
        with ranges.span("disabled"):
            pass
        self.assertEqual(len(self.events), count)

    def test_worker_layer_does_not_change_main_thread(self):
        ranges = self.make_ranges()
        ranges.local.layer = 0
        obj = SimpleNamespace(call=lambda layer: layer)
        ranges.annotate(obj, "call", "dense_producer", layer_arg=True)
        thread = threading.Thread(target=obj.call, args=(3,))
        thread.start()
        thread.join()
        self.assertEqual(ranges.local.layer, 0)
        self.assertIn("/l=3/s=dense_producer", self.events[0])
