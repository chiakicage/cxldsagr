# C7b draft: bounded record transport

Status: root reviewed this design and authorized isolated CPU prototype work.
The draft preceded implementation and does not modify C7a's current plan.

## Accepted evidence

C6 formal `motivation_c6_20261004_u16_r2_01` and dense-only profile
`motivation_c6_dense_profile_20261004_01` use matching production dependencies.
Twenty saved profile outputs match formal HBM byte-for-byte. Independent
attribution and a second overlap/event review verify all 63,688 physical GPU
activities and twenty event edges. Details are in the
[profile checkpoint](../../system/deepseek_motivation_graph_profile_checkpoint.md).

In that single profiled revisit, ten mapped-host gather kernels take
14.708310 ms and account for 720 MiB total. None overlaps any matrix primary or
helper activity; 0.457412 ms overlaps other main-stream metadata/copy work.
For target layers 1-9, the preceding projection graph API finishes
0.555525-1.036319 ms before gather completion, but its first matrix helper starts
24.992-44.992 us afterward. No captured main/context synchronization API
intersects the examined submission windows. The copy stream is non-blocking,
the main stream is null, and both have priority 0.

Every gather has 65,536 CTAs, 128 threads, twenty registers/thread and zero
static/dynamic shared memory in the raw trace. A 1,152-byte aligned record has
72 uint4 units, so only 72 threads perform one vector load/store for that record.
These facts do not establish concurrent residency, issue order or the cause
of the delayed main-stream work. Event GPU timestamps are zero, driver APIs
are absent, and one unmatched context sync occurs after the analyzed windows.

The primary evidence directory is
`experiments/deepseek_v32_motivation/output/data/motivation_c6_dense_profile_20261004_01/analysis/`.
Use `dense_overlap_by_layer.csv`, `stream_dependency_by_layer.csv`,
`independent_overlap_review.json` and their saved independent sources. Do not
reuse C6 API time as C7b measurement. C7a's metadata timing is a separate result;
its accepted component run is `deepseek_dense_prefetch_c7a_20261004_01`, saved
at `/tmp/deepseek_dense_prefetch_c7a_20261004_01.json`. The helper source SHA-256
is `f27260feaa6b21d302232e29325a461a31d158d6ef8671f7dfa164c2c3a62a7b`
and native build fingerprint is
`c2b3905b211ef59fe56032de605818aa877acc403431dcd2cbc6c45703ed23bf`.
Its full-displaced helper median is 2.775675 ms checked and 1.929521 ms native;
gather is unchanged at 1,472.559/1,471.843 us. This is complete helper timing,
with no lookahead overlap measurement. Root accepted C7a for combined
validation; C7b must not enter that locked source.

## Hypothesis and alternatives

A bounded grid lets each CTA process several records and may change the
gather's scheduling interaction with concurrent matrix work. It may also reduce
memory-level parallelism and lengthen copy time. Either effect can dominate
the complete pair. No SM partitioning, contention diagnosis or speedup is
assumed. The initial sweep changes only CTA count/record iteration:
unrestricted, ceil(SMs/4), ceil(SMs/2), SMs and 2xSMs. These are
65,536/33/66/132/264 CTAs for the recorded full-miss shape on 132 SMs.

Ranked steps are: establish unrestricted serial/concurrent controls; implement
cap-only transport; reject incorrect or slower pair candidates; evaluate any
survivor in complete lookahead. Warp width/vector mapping, stream priority,
ordinary-load cache policy, copy engines and scheduling topology are later
questions requiring separate candidates. They must not change within this
comparison. A compute-first concurrency control may expose launch-order
sensitivity, but cannot establish a hardware cause or replace the production
gather-first comparison.

## Correctness and measurement risks

- Reducing the grid without a row loop silently drops records. Tail counts,
  M below/equal/above the cap and many iterations per CTA need explicit checks.
- Offset contiguous tensors can be unaligned. Both addresses and width must
  choose the copy path for every row; dtype/width assumptions cannot come from
  DeepSeek alone. Copy arbitrary bytes and compare guards, sentinel/candidate
  tails and untouched rows, not just requested BF16 values.
- Duplicate destination IDs have the existing race semantics and cannot be
  used to claim uniqueness. Dense ticket IDs are unique; repeated generic host
  IDs with distinct destinations are legal and must remain separate work rows.
- Current-layer computation must read independent resident records, not the
  next-layer destination that is still being filled. Otherwise concurrency is
  a data race or a hidden serial dependency rather than lookahead.
- Inputs/IDs/selection, indexer data and checkpoint-derived activations are
  prepared before timing. API-owned quantization, layout conversion, allocation
  and GPU helpers remain inside complete API boundaries. Do not time a large
  fixture copy or reset and label it transport/compute overlap.
- Serial and concurrent controls must differ only in the required dependency.
  A CPU synchronize between submissions, replay input copy, hidden allocation
  or synthetic repeated GEMM chain can distort the comparison. Preserve an
  explicit record of both launch order and completion joins.
- Ordinary loads retain possible cache reuse. A 72 MiB source set and rotating
  disjoint source slabs can reduce repeated-address reuse, but size alone proves
  no cache property. Label address-reuse regimes; keep load instructions fixed.
- Kernel intervals show concurrent launches/execution, not necessarily useful
  copy/compute progress. The primary gate is complete latency. Any claim about
  internal useful overlap requires diagnostic actual-work intervals; diagnostic
  counters/timestamps cannot be used as the uninstrumented timing result.

## Source boundary and review

At planning time, the generic wrapper SHA-256 is
`bc65f850a908dfc4642b7f90170513881a0a564561d9902c736d8e7d7f16b919`
and native source SHA-256 is
`e082e3b2ef47bad7224f14213c4fe57a6d0bda7399a556230a814d145466bf1b`.
The prototype needs its own FFI name and build identity; it must not shadow the
live module or alter C7a's timing source. Each new run hashes both versions and
actual loaded libraries before and after execution.

The read-only cache review confirms positive-int opt-in, per-row bounds and
alignment, private ticket lifetime, no required cross-row barrier for disjoint
destinations, and separate diagnostic coverage counters. KDA's workflow and
task templates were read. Kernel-wiki was searched for Hopper concurrency;
its returned pages did not directly support this scheduling question, so no
external performance or causal claim is used. Repository source and C6 raw
evidence determine this plan.
