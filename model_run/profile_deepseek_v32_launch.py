import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from deepseek_v32_decode import CONFIG_PATH, load_config
from deepseek_v32_extend import V32ExtendRunner

p = Path("docs/extend_step_profile")
results = []
torch.manual_seed(0)
r = V32ExtendRunner(load_config(CONFIG_PATH), chunk_size=512)
for h in (4096, 65536):
    c = r.make_case(h, 4096)
    for _ in range(3):
        r.run_once(c)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = r.run_once(c)
    vals = {}
    for name, fn in [("eager", lambda case=c: r.run_once(case)), ("graph", g.replay)]:
        samples = []
        host = []
        for _ in range(12):
            torch.cuda.synchronize()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            t = time.perf_counter()
            fn()
            host.append((time.perf_counter() - t) * 1000)
            b.record()
            b.synchronize()
            samples.append(a.elapsed_time(b))
        vals[name] = {
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
            "host_submit_median_ms": statistics.median(host),
            "samples_ms": samples,
        }
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        r.run_once(c)
        torch.cuda.synchronize()
    path = p / f"h{h}_launch.trace.json"
    prof.export_chrome_trace(str(path))
    es = json.loads(path.read_text())["traceEvents"]
    gpu = sorted(
        [e for e in es if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
        key=lambda e: e["ts"],
    )
    end = gpu[0]["ts"]
    busy = 0
    gaps = []
    for e in gpu:
        gaps.append(max(0, e["ts"] - end))
        busy += max(0, e["ts"] + e["dur"] - max(end, e["ts"]))
        end = max(end, e["ts"] + e["dur"])
    span = end - gpu[0]["ts"]
    vals.update(
        history=h,
        gpu_events=len(gpu),
        trace_span_ms=span / 1000,
        trace_busy_ms=busy / 1000,
        trace_busy_pct=100 * busy / span,
        trace_gap_ms=sum(gaps) / 1000,
        max_gap_us=max(gaps),
        gaps_over_10us=sum(x > 10 for x in gaps),
    )
    results.append(vals)
    print(json.dumps(vals), flush=True)
    del g, out, c
(p / "launch_audit.json").write_text(json.dumps(results, indent=2) + "\n")
