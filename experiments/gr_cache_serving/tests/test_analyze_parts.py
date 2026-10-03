import unittest

from experiments.gr_cache_serving.src.analyze_parts import (
    attribute_apis,
    complement,
    duration,
    intersection,
    parse_label,
    union,
)


class TimelineTests(unittest.TestCase):
    def test_union_does_not_double_count_overlap(self):
        self.assertEqual(union([(0, 8), (2, 4), (7, 10), (20, 20)]), [(0, 10)])
        self.assertEqual(intersection([(0, 8), (7, 10)], [(5, 7), (9, 12)]), [(5, 7), (9, 10)])
        self.assertEqual(duration([(0, 2_000_000), (1_000_000, 3_000_000)]), 3)
        self.assertEqual(complement([(-1, 2), (4, 6), (5, 8), (10, 12)], 0, 11), [(2, 4), (8, 10)])

    def test_ownership_uses_launch_thread_and_nested_scope(self):
        scopes = [
            {"globalTid": 1, "start": 0, "end": 100, "label": "phase"},
            {"globalTid": 1, "start": 10, "end": 30, "label": "indexer"},
            {"globalTid": 1, "start": 15, "end": 20, "label": "nested"},
            {"globalTid": 2, "start": 5, "end": 90, "label": "worker"},
        ]
        apis = [
            {"globalTid": t, "start": s, "correlationId": c}
            for t, s, c in ((1, 16, 1), (1, 22, 2), (1, 50, 3), (2, 16, 4), (1, 100, 5))
        ]
        self.assertEqual(
            attribute_apis(scopes, apis), {1: "nested", 2: "indexer", 3: "phase", 4: "worker"}
        )

    def test_label_fields_preserve_prefill_chunk_and_layer(self):
        self.assertEqual(
            parse_label("gr/r=8/p=prefill/c=63/l=4/s=ffn_moe"),
            {"r": 8, "p": "prefill", "c": 63, "l": 4, "s": "ffn_moe"},
        )
        self.assertIsNone(parse_label("unrelated"))
