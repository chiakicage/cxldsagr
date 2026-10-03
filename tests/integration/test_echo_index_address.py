"""Opt-in correctness across 2 GiB and 4 GiB index-buffer byte offsets."""

import os
import unittest
from types import SimpleNamespace


@unittest.skipUnless(
    os.environ.get("SPARSEGR_RUN_ECHO_GPU_TESTS") == "1",
    "requires isolated ECHO GPU environment and >4 GiB free HBM",
)
class EchoIndexAddressTests(unittest.TestCase):
    def test_real_index_write_and_reads_across_large_byte_offsets(self):
        import torch
        from sglang.srt.layers.attention.nsa import index_buf_accessor as accessor

        from models.deepseek_v32.echo_index import scoped_index_address_fix

        stride = 64 * 132
        pages = [(2**31 - 1) // stride + 2, (2**32 - 1) // stride + 2]
        buf = torch.zeros((pages[-1] + 7, stride), dtype=torch.uint8, device="cuda")
        pool = SimpleNamespace(page_size=64, index_head_dim=128, quant_block_size=128)
        locations = torch.cat(
            [torch.arange(p * 64, (p + 1) * 64, device="cuda", dtype=torch.int64) for p in pages]
        )
        keys = (
            torch.arange(128 * 128, device="cuda", dtype=torch.float32)
            .reshape(128, 128)
            .remainder(7)
            .to(torch.float8_e4m3fn)
        )
        scales = torch.full((128, 1), 1.25, device="cuda", dtype=torch.float32)
        accessor.SetKAndS.execute(
            pool=pool, buf=buf, loc=locations, index_k=keys, index_k_scale=scales
        )
        page_ids = torch.tensor(pages, dtype=torch.int32, device="cuda")
        self.assertLess(int((page_ids[:1] * stride).item()), 0)
        with scoped_index_address_fix(accessor, torch):
            actual_k = accessor.GetK.execute(pool, buf, 128, page_ids)
            actual_s = accessor.GetS.execute(pool, buf, 128, page_ids)
            torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual_k, keys.view(torch.uint8)))
        self.assertTrue(torch.equal(actual_s.view(torch.float32), scales))


if __name__ == "__main__":
    unittest.main()
