# Exact prefetch hint component checkpoint

Status: integrated after component exactness/timing gates. Combined C7a + hint
full-model validation and the complete C7 formal trace passed independent
audits. C7 matrix API profiling and final publication remain pending.

## Implementation

`operators/deepseek_v32/indexer/prefetch_hint.py` fuses finite classification,
positive-zero replacement and exact integer partial counts. It retains the
same contiguous masked shape and PyTorch FP32 sum as the checked expression.
A final kernel sums counts and publishes only offset[0] using FP32 div.rn.
The remaining fifteen offset values and every input score byte are unchanged.
CPU, gradient-enabled execution, other devices and nonunit inner strides use
the original expression. All scratch is temporary on the calling stream.

`EchoAttentionRunner` invokes this helper at the same point after top-k,
including certified all-hit ECHO calls. A later evicted revisit still receives
the retained threshold. No selection, hint formula, maps, FIFO priorities,
record traffic, precision policy or transaction boundary changes.

Production identities:

- `operators/deepseek_v32/indexer/prefetch_hint.py`:
  `0ba70bdc2daf4be2c385b60af69309974394ab8d237f9aaf5519fb9350b6a07c`.
- `models/deepseek_v32/echo_attention.py`:
  `cddf8aef94f2c3f3520bc8152aea6eaed4411ad9b5d6ad26c0303d2facd1766e`.
- `operators/deepseek_v32/indexer/tests/test_prefetch_hint.py`:
  `c1c8cd9e63f0c86f1a5539ead8cea2874f5d688f1e06e23dea1164a5f6c8d267`.

## Exactness

The temporary prototype passed 186 byte-exact cases and non-default-stream
predecessor/consumer ordering. It covers every history N=1024..65536 in steps
of 1024, short rows, Q128 and actual Q1024 tails, padded strides, NaN/Inf,
signed zeros, raw subnormals, finite overflow producing Inf/NaN sums, and
one/two finite entries among invalids. Masked scratch, all offset bytes and
unchanged source bytes were checked. The final prototype SHA-256 is
`aedf29a4dcad0cdbb63f791d9b35fd6d987a955e94000122cd0ef442a7f9aa8d`.

All 43 production helper tests passed, including CPU/CUDA bit checks, padded
and nonzero storage offsets, nonunit-stride fallback and stream ordering.
The unchanged top-k suite passed all 18 checks after fixing the command's PATH.
The first combined invocation had 47 passes and 14 selection setup failures
because `.venv/bin` was absent from PATH and Ninja could not launch; this was
not a successful full suite. Only the affected selection suite was rerun.
Ruff passed. Test evidence and component probes remain engineering material
under `/tmp/deepseek_prefetch_hint_probe_20261004/`.

## Timing and profiling

Exclusive GPU0 timing on H200, PyTorch 2.12.1+cu130/CUDA13.0. Complete production
API timing includes validation, allocations, mask/count, unchanged sum,
publication and final device synchronization. Ten warmups and 100 alternating
samples per mode; all samples retained. Input generation is outside timing.
Source identities were rechecked after execution.

| Consumed/input geometry | Checked median ms | Production median ms |
|---|---:|---:|
| Q4/N1024 | 0.084920 | 0.063408 |
| Q4/N32768 | 0.092297 | 0.068525 |
| Q4/N65536 | 0.093168 | 0.067551 |
| Q128/N65664, last four rows | 0.093832 | 0.067211 |
| Q4/N65665, row padding 127 | 0.092887 | 0.067111 |

The matching arithmetic prototype's three-call traces reduce kernels from
11 to 3 for N65536 and N65664. Saved kernel-time sums reproduce directly from
all trace events. Those instrumented kernel sums exclude CPU overhead and
must not replace the complete production API times above. The profiler was
run separately from uninstrumented timings. `production_bench_01.json` records
the production API numbers; `screen_01.json`, `bench_01.json`, `profile_01.json`
and their executable sources distinguish each boundary.

Combined validation exec 52237 / pytest PID1804770 completed under cache_c3;
artifacts are `/tmp/deepseek_c7_hint_combined_validation_20261004_01/`.
All 176 checks passed, including the complete four-scheme checkpoint workload
at H=P65536, C1024, A121/128 and the separate ownership/failure cases. All 109
combined source hashes remained unchanged. Global CPU regression passed 2764
checks, 947 explicit optional/hardware skips and 58 subtests. The quiet trace
`motivation_c7_hint_20261004_u16_r2_01` completed from frozen source and passed
all 128 output checks, 96 offload/HBM byte comparisons and independent
saved-data arithmetic. ECHO first-visit mean is 2423.584 ms, 19.415 ms below C6;
the combined trace does not isolate this helper's contribution. See
[C7 integration](../../system/deepseek_motivation_c7_hint_integration.md).
Matching attribution remains required before publication.
