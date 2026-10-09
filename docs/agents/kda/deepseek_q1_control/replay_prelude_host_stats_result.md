# Four-method prelude with host allocator observations

The accepted observer control reproduces the long recorded gaps while reporting
zero PyTorch allocator-owned pinned bytes at both existing runtime callbacks.
All six profile scopes have 416 ns median gaps. The measured plain windows have
67.872/68.288 us idle, consistent with the preceding four-method B1/B2 controls.
This adds one independent benchmark process and one profile process; the
50 within-process pairs and six scopes are not independent treatment repeats.

| Scope | Span us | Busy us | Idle us | Median recorded gap ns |
| --- | ---: | ---: | ---: | ---: |
| Warmup plain | 1068.483 | 1000.067 | 68.416 | 416 |
| Warmup timed | 1058.593 | 991.041 | 67.552 | 416 |
| Pair 0 plain | 1068.163 | 1000.291 | 67.872 | 416 |
| Pair 0 timed | 1065.602 | 997.090 | 68.512 | 416 |
| Pair 1 timed | 1062.690 | 993.794 | 68.896 | 416 |
| Pair 1 plain | 1071.715 | 1003.427 | 68.288 | 416 |

The separate clean benchmark retains 50 balanced plain/timed AB/BA pairs.
Complete synchronized forward wall medians are 1.957072/1.9699895 ms; the
paired timed-minus-plain median is +12.946 us, with 13/50 timed wins.
Prefix restoration and archival remain outside timing. These measurements
are separate from intrusive profile windows and do not establish a speedup.

Check, benchmark and profile each record two `torch.cuda.host_memory_stats()`
observations: after graph construction and five warmups, then after check or
samples. All six observations report `allocated_bytes.current=0`,
`num_host_alloc=6` and `num_host_free=6`. The callbacks are outside collection,
not immediately after a host-cache flush. The unchanged four-method prelude
still prepares one compute bank, warms HBM/ECHO/serial/dense in order, releases
its offload resources and graphs, then returns to a fresh empty HBM cache.

The zero value covers rounded active and cached bytes owned by PyTorch's host
allocator at those boundaries. It does not measure allocations outside that
allocator, establish continuous absence of pinned memory, identify CUDA
allocation APIs, or exclude persistent effects of earlier resource use.
No active/cached subtraction is used. The observed long-gap state therefore
does not require nonzero PyTorch allocator-owned pinned bytes at these two
callbacks; its lower-level cause remains unresolved.

Runs are `q1_replay_prelude_host_stats_{bench,profile,audit}_20261008_01` under
`experiments/deepseek_v32_mfu/output/data/`. The distinct receipt is at
`/tmp/cxldsagr-checks/q1-replay-timer/q1_replay_prelude_host_stats_check_20261008_01/`.
The observer driver SHA256 is
`6fee25e67ff8e8077aad6478483fa6247a41e4acf2ab8dd8e12ea8654109960c`;
the analyzer SHA256 is
`b523902a56246d69058d6c00ca69156e53f7d8701a0c1850a16fa13c1ed84422`.
Only the observer source differs from accepted B1. Runtime and full-prelude
completion identities remain equal; variable observations are separately hashed.

Root's independent check reread rehashed 1,441 receipt artifacts and verified
six saved bitwise comparisons for tokens 111090, 111091 and 111092. The raw
SQLite reread checked all 197 correlated nodes in each scope, 192 layer owners,
every process/device activity intersecting each layer window, and exact integer
interval unions, including final-norm overlap at the window boundary. It found
no foreign activity and reproduced all recorded gaps. A1/B1 native signature
and owner comparisons also passed. These checks do not prove identical full
graph dependency edges or identify a scheduling mechanism.

Existing author CPU cases (8), independent observer CPU cases (22), the brief
independent analyzer source review, actual-check reread and raw-activity reread
are copied and hashed in the audit's `independent_review/index.json`. The raw
review SHA256 is
`6688d3f8d00ccdbf17375a254adcbcec7fe9b901c33b3bd5c1496b46cfc0d878`.
All 19 original audit files remain unchanged. This note adds no GPU execution
or tests and does not replace published production measurements.
