# Remaining Q1 ECHO adapter work

Read-only diagnosis against `deepseek_h64k_a1_cub_20261008_01` and the current
`experiments/deepseek_v32_echo_official/report/decode_gap/kernel_details.csv`.
No production change or new GPU measurement is part of this diagnosis.
The pinned official score/prefetch kernels remain unchanged.

## Measured remaining work

The following are sums of GPU activity durations over L0-L2 in the published
intrusive timeline, not complete API or synchronized model timings.

| Local ECHO stage | Activities | Activity sum, us | Sum of per-layer spans, us |
| --- | ---: | ---: | ---: |
| FIFO preparation | 30 | 70.944 | 112.224 |
| Prefill mean and decode EMA | 12 | 35.904 | 39.680 |
| Prefetch finalization | 15 | 19.904 | 24.864 |

FIFO preparation contains 18 kernels and 12 memsets. Key preparation costs
10.176 us, CUB histogram 6.816 us, exclusive sum 3.456 us, six OneSweep kernels
32.192 us, slot/journal preparation 4.832 us, and memsets 13.472 us. Its native
source is `operators/deepseek_v32/indexer/csrc/echo_sparse_recall.cuh:178-269`.
The radix call already sorts only significant age bits, starting at bit 32;
it does not sort all 64 encoded key bits. The rejected DeviceTopK candidate
must not be retried under a new name; see `prepare64_plan.md`.

The hint stage comprises mask/count 4.736 us, the original PyTorch FP32 sum
24.800 us, publication 3.680 us, and decode EMA 2.688 us. Each sum uses one CTA
of 512 threads in the recorded trace. This identifies low grid parallelism;
there is no current hint NCU evidence to attribute its time to memory bandwidth,
latency, or a specific stall. Existing promotion NCU must not be reused as hint
or FIFO evidence.

## First candidate: a proved free-slot preparation

Eligibility must require all of the following before choosing this normal
dispatch: the official Q1 path, exactly one active session, an exclusive lease,
ordinary nontransient append, initialized host history equal to the query start,
and `P - H >= 64`. At preparation, pending main KV is not written yet.

Every live host-backed pool record belongs to one of the session's H initialized
tokens. The cache invariant therefore gives `L <= H`, so
`free = P - L >= P - H >= 64`. This is a capacity argument, not a residency
certificate. It applies to arbitrary valid partial residency under these
conditions. Multi-session pools, insufficient slack, transient requests and
other indexer shapes keep their existing normal dispatch.

The saved current check at
`/tmp/cxldsagr-checks/deepseek_v32_mfu/data/deepseek_h64k_a1_cub_20261008_01_h65536_a1_check/echo_default_graph_prefetch_evidence.pt`
records H=65,536, P=65,600, one session, all 65,600 slots free, and clock 256
for each layer. Its first 64 prepared slots are 1 through 64. This is evidence
for the measured cold case; the eligibility proof does not assume that all
history is cold.

There is already a suitable selection algorithm in
`operators/deepseek_v32/indexer/csrc/echo_sparse_append_free.cuh:11-88`:
scan every priority to validate its range and emit warp free masks, then select
ascending free slots while checking their free bitmap and reverse mapping.
It traps when fewer slots exist than required. Reusing this algorithm for
preparation would preserve the original stable FIFO prefix exactly: free age
-1 sorts before all occupied ages, and selected free slots remain ascending.
It must not call the existing append API, which also copies and publishes KV.

A proposed prepare entry must still reset the entire allocation journal,
counter and statistics, initialize the same request metadata, and modify no
record, mapping, priority or clock. It also needs an explicit prepared bound
of 64 carried through the one-use lease: the current `_PreparedPrefetch` has
no bound and the general consumer can request up to 8,192 slots. Unprepared
tail entries cannot silently remain available. The unchanged official adapter
uses only the first `min(counter, 64)` slots. Finalization consumes the full
allocation journal, not the full sorted slot list
(`cache/sparse_token_pool.py:681-695`).

Correctness must cover cold and partial residency, the tight P-H=64 case,
free slots only at the end of the pool, malformed priorities, inconsistent
free bitmap/reverse map, failed capacity verification, and all prediction
counts from 0 to 64. Check bytes for unchanged storage and the full journal
reset; then run existing promotion/lifecycle and changed-input graph checks.
Never fall back after a device verification failure.

The proposed next measurement is current-baseline NCU on the saved actual
priority/clock tensors through `cache_ops.prepare_prefetch`, preceded by an
independent check against its existing full stable-sort contract. No candidate
implementation should precede that measurement. A small baseline harness is
needed because the rejected DeviceTopK prototype was removed.

That baseline check and independent NCU collection subsequently completed;
the measurements, source/native/input binding and limitations are recorded in
`baseline_profile_plan.md`. No bounded-free candidate has been implemented.

## Second candidate: preserve and fuse the exact Q1 mean

`models/deepseek_v32/attention.py:373-382` maintains both hints. Official decode
only updates the kth-score EMA (`nsa_indexer.py:413-416`); official prefill uses
a separate mean (`nsa_indexer.py:551-555`). Local `offset[0]` remains observable
and is used by later Q>1 calls, so omitting its Q1 update is not equivalent.
The existing hint contract also fixes the PyTorch FP32 reduction order.

A narrow candidate could fuse finite masking, integer counting and publication
while preserving the current sum's per-thread four-accumulator input schedule,
sequential accumulator combination, shared-memory halving tree and descending
warp shuffle. For N=65,537 the aligned vector body and one-element tail both
matter. Relevant installed PyTorch definitions are
`ATen/native/cuda/Reduce.cuh:500-559`, `:635-672`, and `:1041-1187`.
Masking must replace nonfinite values with positive zero, preserve finite
subnormal/signed-zero bits, use integer clamp-to-one count, and retain FP32
round-to-nearest division. EMA fusion additionally retains separate multiply
rounding (`enable_fp_fusion=False`) and writes only offset[1].

This removes launches and scratch traffic but does not automatically solve
the one-CTA reduction bottleneck: the fixed tree limits repartitioning. A
different generic Triton reduction or torch.compile result is not accepted
without exact-bit evidence. First obtain current hint NCU, then choose the
smallest source-bound candidate and validate adversarial values as well as
real L0-L2 scores. The unchanged general hint path remains available for
uncovered layouts and shapes.

## Official semantics constrain larger changes

The official decode consumer promotes only prefetched records that occur in
the actual recall set, then clears unused temporary mappings
(`mem_cache/recall_ops.py:385-551`, `memory_pool_host.py:1379-1429`). The local
adapter deliberately promotes every successful prediction before exact top-k,
including false positives; this behavior is in the model rules and its state
transition acceptance. Copying the official consumer scheduling wholesale
would change local residency, FIFO priority, future hits and traffic. It is not
an overhead-only optimization.

Current promotion validation is already the accepted 256-thread version.
The separate `q1_validation_ncu_20261008_01` shows validation alone improving
from 11.168 to 7.808 us under kernel replay, cache flush and base clocks;
the published CUB trace now spends 14.624 us across three validations. Those
NCU values exclude record copy and must not replace complete-call timing.
Further promotion changes rank behind the two candidates above unless fresh
profiling provides a stronger reason.
