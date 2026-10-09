# Single-method warmup controls

One complete `dense_prefetch` warmup is sufficient to reproduce the long-gap
regime in this control. A `serial_sparse` warmup or an `echo` warmup does not
reproduce it in their first processes. **Every window and wall time below
measures HBM after returning to a fresh HBM cache**, not the selected offload
method's own execution latency.

| Warmup package before HBM | Median gap in all six scopes, ns | Measured plain spans, us | Measured plain idle, us |
| --- | ---: | ---: | ---: |
| `dense_prefetch` | 416 | 1057.795 / 1067.267 | 68.928 / 67.584 |
| `serial_sparse` | 96 | 1014.465 / 1008.641 | 20.481 / 16.770 |
| `echo` | 96 | 1018.660 / 1017.860 | 18.753 / 19.170 |

The separate clean benchmark retains every sample from 50 balanced AB/BA
plain/timed pairs per treatment. Plain uses no outer CUDA event pair; timed
adds the original timer's outer event pair. The paired deltas measure that
within-process timer difference, not the causal cost of the warmup package.

| Warmup package | Plain wall median, ms | Timed wall median, ms | Paired timed-minus-plain median, us | Timed wins |
| --- | ---: | ---: | ---: | ---: |
| `dense_prefetch` | 1.9567355 | 1.9597825 | +6.1620 | 21 / 50 |
| `serial_sparse` | 1.8925885 | 1.9065445 | +14.8485 | 12 / 50 |
| `echo` | 1.9046380 | 1.9204930 | +18.2550 | 12 / 50 |

Clean wall includes the complete synchronized forward; prefix restoration
and archival are outside timing. NSYS windows cover L0–L2 and include profiler
overhead. Each treatment has one independent benchmark process and one profile
process, plus its separate check. Neither 50 pairs nor six scopes increases
the number of independent treatment replications. Cross-treatment wall
differences are descriptive; this is not a replicated speedup measurement.

After one unchanged compute-bank preparation, each driver selects its method,
runs H=65,536 prefill, snapshots the prefix, restores the cold state, prepares
the Q1 full-extend graph and executes token 111090. It then deletes the snapshot
and selects fresh HBM. The recorded completion requires the same compute bank,
two cache-generation advances, zero valid length, closed prelude extend graphs
and no shared pools, sessions or prefetch helpers. The measured reduced HBM
preparation/replay remains unchanged, on GPU0/CPU0–7 with the matched FREE
formal environment. No flush, host-stat query or profiler range was added.

All three checks passed independent CPU rereads of six finite, bitwise saved
plain/timed-versus-eager outputs for tokens 111090/111091/111092. Dense and
serial receipts each bind 1,435 artifacts; ECHO binds 1,439. Every execution
archives the same 1,375 source files. Dense and serial each archive 57 runtime
files and leave two official offload DSOs plus one decode-hint specialization
unobserved. ECHO archives 61 runtime files and observes all formal runtime
entries. The actually observed entries match the formal identities exactly.

Independent raw SQLite reviews passed for all 18 scopes: 197 native GPU nodes,
192 layer owners, all 17 recorded kernel launch fields, copy bytes and native
ownership match the formal/A1/B1/C1/cache-construction references. All-process
and all-device rereads find no foreign activity in the windows. Integer unions
include any final-norm activity crossing the window end. Node signatures and
ownership do not prove equal complete graph dependency edges; the absent
per-launch DeepGEMM ledger and unavailable retained CuTe MLIR files remain
collector limits.

The shared driver is `q1_replay_single_method_prelude.py`, SHA256
`6368e0a8470a39c3a42a2eb9553988a36afa152740bbb446b8429641862b2fb3`.
The analyzer is `analyze_replay_single_method_prelude.py`, SHA256
`e5fb3e79e724acc14b7e237791d36d9cc0b1c1e472ecb4d4918aa74bf134be8c`.
Both are in `experiments/deepseek_v32_mfu/src/`. Accepted child run IDs are
`q1_replay_single_{dense,serial,echo}_{bench,profile,audit}_20261008_01`;
their checks use the corresponding `_check_` IDs under `/tmp/cxldsagr-checks/`.

The CPU evidence bundle is
`experiments/deepseek_v32_mfu/output/data/q1_replay_single_methods_20261008_01/`.
Its archived builder reads original check/bench/profile/audit artifacts,
rehashes 14,385 distinct source/input paths, directly rereads all 18 raw windows
and recomputes all 300 clean samples. It retains original receipt/audit/source
identities, all samples, interval unions and gaps, and six existing independent
reviews with their helpers. All original audit files remain unchanged. Bundle
`result.json` SHA256 is
`592a25b96fa82f5dd75f2d199aa27a4a62a13d00172a87e0b9a9e842de44d372`.
Root's independent review passed: it rehashed all 87 exports and 14,385 input
paths and recomputed the exported integer unions/gaps and original clean
samples. The copied review is indexed in `root_review/index.json`; external
`acceptance.json` records acceptance without changing the original bundle
result or its hash. The READMEs now describe this diagnostic result; published
figures and formal performance values retain their original measurement.

Together with the negative construction-only, HBM-repeat and direct pinned
allocation controls, the result narrows the reproducing package to a complete
dense-prefetch warmup in this environment. It does not identify DMA, allocator,
stream history, a fence or any other hardware cause, and does not prove dense
is the only possible trigger. The proposed
[method-isolated formal measurement](method_isolation_plan.md) remains a
design only. Its purpose is to remove another method's same-process warmup
influence; its effect still requires fresh formal acceptance and measurement.
