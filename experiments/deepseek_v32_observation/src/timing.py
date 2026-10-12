"""GPU API spans from isolated CUDA Graph replays; no profiler in clean timing."""

from __future__ import annotations

import time

import torch

from experiments.deepseek_v32_observation.src.contract import statistics


def benchmark(call, *, warmup=5, repeats=31, inner=20, before_call=None):
    if min(warmup, repeats, inner) < 1 or repeats < 3:
        raise ValueError("Positive warmup/inner and at least three repeats are required")
    if before_call is not None and inner != 1:
        raise ValueError("Mutable input reset requires one measured call per graph")
    device = torch.cuda.current_device()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            if before_call is not None:
                before_call()
            output = call()
    stream.synchronize()
    if before_call is not None:
        before_call()
        torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    # External events remain actual graph nodes. They delimit every API call,
    # including conversions and multi-kernel helpers, but exclude Python launch.
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(inner)]
    ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(inner)]
    with torch.cuda.graph(graph, stream=stream):
        for start, end in zip(starts, ends, strict=True):
            start.record(stream)
            output = call()
            end.record(stream)
    stream.synchronize()
    samples, wall_samples = [], []
    for _ in range(repeats):
        if before_call is not None:
            before_call()
            torch.cuda.synchronize(device)
        wall_start = time.perf_counter_ns()
        graph.replay()
        torch.cuda.synchronize(device)
        wall_samples.append((time.perf_counter_ns() - wall_start) / 1000 / inner)
        samples.append(
            sum(start.elapsed_time(end) for start, end in zip(starts, ends)) * 1000 / inner
        )
    result = {
        **statistics(samples),
        "samples_us": samples,
        "graph_wall_samples_us_per_call": wall_samples,
        "warmup": warmup,
        "repeats": repeats,
        "inner": inner,
        "timing_boundary": "external CUDA events around complete API; CUDA Graph replay; mean of inner spans per sample",
        "profiler_enabled": False,
        "input_reset_boundary": "outside timing graph and event spans"
        if before_call is not None
        else None,
    }
    del graph, output, starts, ends
    stream.synchronize()
    return result
