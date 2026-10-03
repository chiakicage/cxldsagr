import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.gr_cache_serving.src.memory_guard import MemoryGuard


class MemoryGuardTests(unittest.TestCase):
    def setup_fake(self, *, memory=20, pid=None):
        self.calls = []
        nvml = SimpleNamespace(
            NVMLError=OSError,
            nvmlInit=lambda: self.calls.append("init"),
            nvmlShutdown=lambda: self.calls.append("shutdown"),
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda index: "GPU",
            nvmlDeviceGetComputeRunningProcesses=lambda handle: [
                SimpleNamespace(pid=os.getpid() if pid is None else pid, usedGpuMemory=memory)
            ],
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=memory),
        )
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda index: SimpleNamespace(total_memory=100),
                set_per_process_memory_fraction=lambda fraction, index: self.calls.append(fraction),
            )
        )
        return nvml, torch

    def test_success_records_peak_and_hard_allocator_limit(self):
        nvml, torch = self.setup_fake()
        with (
            patch.dict(sys.modules, {"pynvml": nvml}),
            MemoryGuard(torch, total_bytes=80, non_torch_reserve_bytes=10) as guard,
        ):
            guard.check()
        self.assertEqual(self.calls, ["init", 0.7, "shutdown"])
        self.assertEqual(guard.metadata["peak_sampled_device_bytes"], 20)
        self.assertGreater(guard.metadata["samples"], 0)

    def test_over_budget_and_foreign_processes_reject_results(self):
        for options in ({"memory": 81}, {"pid": -1}):
            nvml, torch = self.setup_fake(**options)
            with patch.dict(sys.modules, {"pynvml": nvml}):
                guard = MemoryGuard(torch, total_bytes=80, non_torch_reserve_bytes=10)
                guard.thread.join(timeout=1)
                with self.assertRaisesRegex(RuntimeError, "guard failed"):
                    guard.close()
            self.assertEqual(self.calls[-1], "shutdown")

    def test_exception_teardown_stops_sampler(self):
        nvml, torch = self.setup_fake()
        with (
            patch.dict(sys.modules, {"pynvml": nvml}),
            self.assertRaisesRegex(ValueError, "caller"),
            MemoryGuard(torch, total_bytes=80, non_torch_reserve_bytes=10) as guard,
        ):
            raise ValueError("caller failed")
        self.assertFalse(guard.thread.is_alive())
        self.assertEqual(self.calls[-1], "shutdown")


if __name__ == "__main__":
    unittest.main()
