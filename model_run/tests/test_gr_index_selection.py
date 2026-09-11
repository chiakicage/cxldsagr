"""Check query batching against the unsplit causal indexer on CUDA."""

import unittest

import torch

from model_run.deepseek_v32_decode import deep_gemm
from model_run.deepseek_v32_extend import V32ExtendRunner
from model_run.gr_index_selection import select_batched


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class IndexSelectionTests(unittest.TestCase):
    def test_batched_topk_and_sampled_logits_match_unsplit(self):
        torch.manual_seed(42)
        q = torch.randn(257, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        keys = torch.randn(4096, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(4096, device="cuda")
        weights = torch.randn(257, 64, device="cuda")
        ends = torch.linspace(1, 4096, 257, device="cuda").int()
        logits = deep_gemm.fp8_fp4_mqa_logits(
            (q, None),
            (keys, scales),
            weights,
            torch.zeros_like(ends),
            ends,
            False,
            0,
            torch.float32,
        )
        expected = V32ExtendRunner.select_indices(None, logits, ends, 2048)
        actual, samples = select_batched(q, keys, scales, weights, ends)
        torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0)
        rows = torch.tensor([0, 128, 256], device="cuda")
        mask = torch.arange(4096, device="cuda")[None, :] < ends[rows, None]
        torch.testing.assert_close(samples[mask], logits[rows][mask], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
