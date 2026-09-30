"""Opt-in GPU regression across a 4 GiB pinned-host MLA address boundary."""

import os
import unittest


@unittest.skipUnless(
    os.environ.get("SPARSEGR_RUN_ECHO_GPU_TESTS") == "1",
    "requires isolated ECHO GPU environment and >4 GiB pinned RAM",
)
class EchoRecallLargeAddressTests(unittest.TestCase):
    def test_sparse_recall_above_int32_element_offset(self):
        import torch
        from sglang.srt.mem_cache import recall_ops

        from models.deepseek_v32.echo_recall import scoped_extend_recall_address_fix

        high = (2**31 - 1) // 576 + 1
        size = high + 2
        host = torch.empty((size, 1, 576), dtype=torch.bfloat16, pin_memory=True)
        pattern = torch.arange(576).to(torch.bfloat16)
        host[1, 0] = pattern
        host[high, 0] = -pattern
        flags = torch.zeros(size, dtype=torch.bool, device="cuda")
        flags[[1, high]] = True
        slots = torch.tensor([1, 2, 0, 0], dtype=torch.int32, device="cuda")
        host_to_device = torch.full((size,), 2**31 - 1, dtype=torch.int32, device="cuda")
        device_to_host = torch.full((4,), 2**31 - 1, dtype=torch.int64, device="cuda")
        device = torch.zeros((4, 1, 576), dtype=torch.bfloat16, device="cuda")
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        with scoped_extend_recall_address_fix(recall_ops):
            recall_ops.recall_update_extend(
                flags, slots, host_to_device, device_to_host, device, host, counter, 512, 64
            )
            torch.cuda.synchronize()
        self.assertEqual(int(counter.item()), 2)
        for host_index in (1, high):
            device_index = int(host_to_device[host_index].item())
            self.assertIn(device_index, (1, 2))
            self.assertEqual(int(device_to_host[device_index].item()), host_index)
            self.assertTrue(torch.equal(device[device_index].cpu(), host[host_index]))
        self.assertEqual(int(host_to_device[0].item()), 2**31 - 1)


if __name__ == "__main__":
    unittest.main()
