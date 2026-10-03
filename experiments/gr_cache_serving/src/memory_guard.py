"""Allocator cap plus sampled NVML rejection for exclusive single-GPU runs."""

import os
import threading


class MemoryGuard:
    def __init__(self, torch, *, total_bytes, non_torch_reserve_bytes, interval_s=0.02):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() != 1:
            pynvml.nvmlShutdown()
            raise RuntimeError("the current memory guard requires exactly one physical GPU")
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total_bytes = total_bytes
        self.limit = total_bytes - non_torch_reserve_bytes
        physical = torch.cuda.get_device_properties(0).total_memory
        if not 0 < self.limit < total_bytes <= physical:
            pynvml.nvmlShutdown()
            raise ValueError("GPU budget and non-Torch reserve do not fit the physical device")
        torch.cuda.set_per_process_memory_fraction(self.limit / physical, 0)
        self.interval_s = interval_s
        self.samples = 0
        self.peak_device_bytes = 0
        self.peak_process_bytes = 0
        self.error = None
        self.metadata = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gr-memory-guard", daemon=True)
        self.thread.start()

    def _sample(self):
        nvml = self.nvml
        processes = nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        foreign = [process.pid for process in processes if process.pid != os.getpid()]
        if foreign:
            raise RuntimeError(f"foreign GPU processes appeared during replay: {foreign}")
        device = nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        process = max((p.usedGpuMemory for p in processes), default=0)
        self.peak_device_bytes = max(self.peak_device_bytes, device)
        self.peak_process_bytes = max(self.peak_process_bytes, process)
        self.samples += 1
        if device > self.total_bytes:
            raise MemoryError(f"sampled total GPU usage {device} exceeds budget {self.total_bytes}")

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self._sample()
                self.stop_event.wait(self.interval_s)
        except (self.nvml.NVMLError, RuntimeError, MemoryError) as exc:
            self.error = exc

    def check(self):
        if self.error is not None:
            raise RuntimeError("GPU memory/exclusivity guard failed") from self.error

    def close(self):
        self.stop_event.set()
        self.thread.join()
        try:
            self.check()
            self._sample()
            self.metadata = {
                "common_total_hbm_limit_bytes": self.total_bytes,
                "torch_allocator_hard_limit_bytes": self.limit,
                "peak_sampled_device_bytes": self.peak_device_bytes,
                "peak_sampled_process_bytes": self.peak_process_bytes,
                "sample_interval_seconds": self.interval_s,
                "samples": self.samples,
                "scope": "hard Torch allocator cap plus reserved external headroom and sampled NVML total guard; sub-sample external peaks not observed",
            }
            return self.metadata
        finally:
            self.nvml.nvmlShutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.stop_event.set()
            self.thread.join()
            self.nvml.nvmlShutdown()
        else:
            self.close()
