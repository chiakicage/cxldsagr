import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.gr_cache_serving.src.transfer import TransferMeter


class Tensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Tensor(self.value)

    def reshape(self, shape):
        return self

    def __gt__(self, limit):
        return Tensor([value > limit for value in self.value])

    def sum(self):
        return Tensor(sum(self.value))

    def to(self, dtype):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class TransferTests(unittest.TestCase):
    def runner(self, mode):
        pool = SimpleNamespace(recall_counter=Tensor(0))
        pool.recall_miss_tokens_extend_cuda_graph = lambda: setattr(
            pool, "recall_counter", Tensor(3)
        )
        allocator = SimpleNamespace(post_alloc=lambda locations: None)
        pool.device_pool_allocator = [allocator]

        def forward(ids, locations):
            if mode == "echo_gr_adapted":
                # Deliberately oversized raw counter: only two valid allocations.
                pool.recall_counter = Tensor(9999)
                allocator.post_alloc(Tensor([4, 9, 0, 0]))
            if mode in ("sparse_sync", "echo_gr_adapted"):
                pool.recall_miss_tokens_extend_cuda_graph()
            return "hidden"

        return SimpleNamespace(
            runner=SimpleNamespace(token_to_kv_pool=pool),
            provenance={"mode": mode, "num_layers": 5},
            dense_controller=SimpleNamespace(last_batch_stats={"h2d_payload_bytes": 100}),
            _forward=forward,
        )

    def test_payload_counts_reset_and_hooks_restore(self):
        torch = SimpleNamespace(
            int64="int64", stack=lambda values: Tensor([x.value for x in values])
        )
        for mode, h2d in (
            ("resident", 0),
            ("sparse_sync", 3 * 1152),
            ("echo_gr_adapted", 5 * 1152),
            ("dense_prefetch", 100),
        ):
            runner = self.runner(mode)
            original = runner._forward
            with (
                self.subTest(mode=mode),
                patch.dict(sys.modules, {"torch": torch}),
                TransferMeter(runner) as meter,
            ):
                self.assertEqual(runner._forward([1, 2, 3], [1]), "hidden")
                stats = meter.finish_request()
                self.assertEqual(stats["h2d_mla_payload_bytes"], h2d)
                self.assertEqual(
                    stats["d2h_mla_payload_bytes"], 0 if mode == "resident" else 2 * 5 * 1152
                )
                meter.begin_request()
                self.assertEqual(meter.finish_request()["h2d_mla_payload_bytes"], 0)
            self.assertIs(runner._forward, original)


if __name__ == "__main__":
    unittest.main()
