# Allocation-only prelude: first result

The allocation-only control did not reproduce the four-method prelude's long
recorded gaps. All six profile scopes have 96 ns median gaps, versus 416 ns
in both four-method ABBA profile processes. The two measured plain windows
have 18.625/18.657 us idle. This factor has one independent benchmark process
and one independent profile process; the 50 within-process plain/timed pairs
and six profile scopes do not provide independent treatment replications.

| Scope | Span us | Busy us | Idle us | Median recorded gap ns |
| --- | ---: | ---: | ---: | ---: |
| Warmup plain | 1019.969 | 999.393 | 20.576 | 96 |
| Warmup timed | 1013.890 | 995.329 | 18.561 | 96 |
| Pair 0 plain | 1017.026 | 998.401 | 18.625 | 96 |
| Pair 0 timed | 1014.850 | 996.770 | 18.080 | 96 |
| Pair 1 timed | 1014.306 | 994.850 | 19.456 | 96 |
| Pair 1 plain | 1012.866 | 994.209 | 18.657 | 96 |

The separate clean benchmark retains 50 balanced plain/timed AB/BA pairs.
Complete synchronized forward wall medians are 1.885065/1.8980505 ms; the
paired timed-minus-plain median is +17.0565 us, with 11/50 timed wins.
These pairs measure extra external-event overhead within this process.
Clean wall measurements and intrusive profile windows remain separate.

After the unchanged compute-bank construction, the driver creates three
lifetimes of three simultaneous direct pinned tensors, each requested through
`allocate_host_tensor((65600, 576), dtype=bfloat16, pin_memory=True)`.
Each tensor has 75,571,200 logical bytes and 134,217,728 storage bytes.
View and backing-owner weak references must expire between lifetimes.
The control adds no pool construction, cache transition, offload kernel,
stream or flush, and preserves the initial HBM cache objects and compute bank.
The subsequent reduced timer and normal graph-entry host-cache flush remain
unchanged.

All three modes' nine saved observations agree on the following selected
allocator counters. The middle two rows occur once per lifetime.

| Observation | Allocator-owned bytes | `num_host_alloc` | `num_host_free` |
| --- | ---: | ---: | ---: |
| Before the three lifetimes | 0 | 0 | 0 |
| Three backing owners live | 402,653,184 | 3 | 0 |
| View and backing owners dropped | 402,653,184 | 3 | 0 |
| After graph construction and five warmups | 0 | 3 | 3 |
| After check or samples | 0 | 3 | 3 |

The nine tensor requests therefore reused three allocator blocks, totaling
384 MiB while owned. By the first runtime callback, those blocks had been
freed. The observations occur at the recorded boundaries, not immediately
after a flush. `allocated_bytes.current` includes rounded active and cached
blocks. No subtraction of uncertain active counters is used to infer cached
bytes. Growth/free counters do not identify CUDA allocation APIs, and the
observations do not cover memory outside PyTorch's host allocator. This
single-process negative result does not exclude all allocation, registration,
visibility, graph-history or offload-lifecycle mechanisms.

The driver is `experiments/deepseek_v32_mfu/src/q1_replay_host_allocation.py`,
SHA256 `ec1397acaaa07334a247b2894752e6ccab1712b0e66a5362cf1bb7ce629477c2`.
The unchanged helper is `cache/host_allocation.py`, SHA256
`6c6c65e166fc6a2289547482a9bc51c1da136190b9105423edd23f9732567150`.
The check uses the distinct `deepseek-q1-replay-host-allocation-v1` receipt at
`/tmp/cxldsagr-checks/q1-replay-timer/q1_replay_host_allocation_check_20261008_01`.
Independent CPU rereading verified six saved bitwise comparisons for tokens
111090, 111091 and 111092 and rehashed 1,433 receipt artifacts.

Runs are `q1_replay_host_allocation_{bench,profile,audit}_20261008_01` under
`experiments/deepseek_v32_mfu/output/data/`. Check, benchmark and profile share
the same static execution identity; variable host-stat evidence is separately
hashed outside that identity. The analyzer binds accepted matched A1,
four-method B1 and HBM×4 C1 controls, preserves their recorded runtime match
boundaries, and delegates the unchanged raw timer audit. All six scopes retain
197 native GPU nodes and 192 layer owners with matched raw launch signatures.
Root accepted the raw-window audit. These checks do not establish equality
of full graph dependency edges or unrecorded per-launch JIT identities.

The analyzer is `experiments/deepseek_v32_mfu/src/analyze_replay_host_allocation.py`,
SHA256 `6446705af7575f27559512e077856061e0b2ac1704b3743a27ab06e4cf2509a8`.
Its independent review passed 23 actual-evidence and negative-mutation checks,
including all three references, source/runtime/receipt bindings, the nine
observation boundaries and variable statistics outside identity. The review
is `/tmp/cxldsagr-checks/q1_replay_host_allocation_analyzer_independent_review.json`,
SHA256 `f2930af078eb300d3170b49495628f17a1b4743df3e1a83dd4118ad32403438c`.
That review used CPU evidence only; root owns GPU execution and raw-profile
acceptance. No production code or published performance values change here.

The next diagnostic keeps the four-method prelude unchanged and reads host
statistics at the timer's two existing runtime callbacks. It will establish
whether the long-gap process still has allocator-owned pinned bytes at those
boundaries; either outcome alone would not establish a hardware mechanism.
