# sparse_mla_sm120 V3.2 RTX 5080 benchmark results

Date: 2026-07-01

Device: NVIDIA GeForce RTX 5080, SM 120, 84 SMs

Environment:

- Python: `.venv/bin/python`
- Package: `flash_mla_sm120==0.1.0`, editable install from `file:///home/cage/dsa/sparse_mla_sm120`
- Reinstall command:

```bash
env MAX_JOBS=1 CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
  uv pip install --python .venv/bin/python --no-build-isolation --reinstall -e sparse_mla_sm120
```

Benchmark command:

```bash
.venv/bin/python sparse_mla_sm120/benchmarks/benchmark_decode_fp8.py --warmup 50 --reps 300
```

## V3.2 Decode FP8

Config:

- `d_qk=576`
- `d_v=512`
- `topk=2048`
- packed KV cache: `656 B/token`
- cache tokens: `65536`
- timing: CUDA graph replay

| topk | heads | batch | p50 us | p95 us | min us | GFLOP | logical MB | repeated MB | TFLOP/s | logical GB/s | repeated GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 16 | 1 | 40.6 | 41.7 | 40.1 | 0.071 | 1.378 | 1.378 | 1.75 | 33.9 | 33.9 |
| 2048 | 16 | 4 | 46.8 | 48.6 | 46.4 | 0.285 | 5.513 | 5.513 | 6.10 | 117.8 | 117.8 |
| 2048 | 16 | 8 | 61.1 | 61.3 | 60.5 | 0.570 | 11.026 | 11.026 | 9.33 | 180.4 | 180.4 |
| 2048 | 64 | 1 | 40.6 | 40.8 | 38.2 | 0.285 | 1.483 | 5.513 | 7.02 | 36.5 | 135.8 |
| 2048 | 64 | 4 | 81.6 | 82.6 | 81.2 | 1.141 | 5.931 | 22.053 | 13.98 | 72.7 | 270.3 |
| 2048 | 64 | 8 | 134.8 | 135.1 | 132.8 | 2.282 | 11.862 | 44.106 | 16.92 | 88.0 | 327.1 |
| 2048 | 128 | 1 | 52.9 | 53.1 | 50.8 | 0.570 | 1.622 | 11.026 | 10.78 | 30.6 | 208.3 |
| 2048 | 128 | 4 | 132.8 | 134.8 | 131.7 | 2.282 | 6.488 | 44.106 | 17.18 | 48.9 | 332.1 |
| 2048 | 128 | 8 | 239.3 | 240.3 | 237.2 | 4.563 | 12.976 | 88.211 | 19.07 | 54.2 | 368.6 |

Notes:

- FLOPs are counted as `2 * batch * heads * topk * (d_qk + d_v)`.
- `logical GB/s` counts one packed KV read per decode token plus Q and output.
- `repeated GB/s` approximates packed KV rereads by head groups: `ceil(heads / 16)`.
- The measured path is `flash_mla_sm120.sparse_mla_decode_fwd`, including split-KV and combine through the Python wrapper.
