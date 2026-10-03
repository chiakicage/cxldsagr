"""Direct pinned reads, including shuffled Host slots beyond 4 GiB."""

import os
import unittest


@unittest.skipUnless(os.environ.get("SPARSEGR_RUN_ECHO_GPU_TESTS") == "1", "GPU opt-in required")
class DirectGatherTests(unittest.TestCase):
    def test_pinned_reads_addressing_and_slot_reuse(self):
        import torch

        from operators.sm90.pinned_gather import gather_pinned_rows

        self.assertTrue(torch.cuda.is_available())
        high = (2**32 // 1152) + 17
        host = torch.empty((high + 8, 1, 576), dtype=torch.bfloat16, pin_memory=True)
        locations = torch.tensor([high + 2, 3, high, 19, high + 5], dtype=torch.int64)
        values = torch.arange(5 * 576, dtype=torch.float32).reshape(5, 1, 576).to(torch.bfloat16)
        host[locations] = values
        indices = locations.to("cuda", dtype=torch.int32)
        output = torch.full((7, 1, 576), -1, dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gather_pinned_rows(host, indices, output[1:6])
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(output[1:6].cpu(), values, rtol=0, atol=0)
        self.assertTrue((output[[0, 6]] == -1).all().item())
        host[locations] = -values
        gather_pinned_rows(host, indices, output[1:6])
        torch.testing.assert_close(output[1:6].cpu(), -values, rtol=0, atol=0)
        gather_pinned_rows(host, indices[:0], output[:0])
        with self.assertRaises(ValueError):
            gather_pinned_rows(host, locations, output[1:6])
