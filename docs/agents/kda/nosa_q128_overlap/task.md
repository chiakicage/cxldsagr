# NOSA Q128 fused offload task contract

Status on 2026-10-04: Group4, original Group8 V-first, shared-union row halves,
two-stripe fetch ILP, earliest-pair priority, frontier fanout and independent-warp
fetch are rejected for promotion. The last candidate passed partial20 and
representative correctness, then failed its first sample at 0.800550 for both
overlap metrics. See [checkpoint.md](checkpoint.md). The independent-warp contract
below remains historical until a new diagnostic/implementation plan is selected;
no candidate is currently accepted for integration.

## Objective and scope

Make the complete asynchronous sparse-offload API faster than the complete serial sparse-union-fetch plus FA3 API for the actual NOSA serving candidate workload. The target is H=65,536, A=128, two KV heads, 32 query heads, D=128, BF16, full NOSA selection/CIS, on the observed NVIDIA M403 device (SM90, 132 SMs, 143 GB-class reported memory). Preserve full-batch execution and the model's exact selection, causal mask, CIS bias, numerical repair, and output contract.

CUDA device properties and the capture's CUDA hardware metadata report NVIDIA
H200. NVML/`nvidia-smi` report NVIDIA M403 for the same UUID. Preserve both
observations with their tool provenance. The H200 SXM BF16 dense peak of
989.5 TFLOP/s is a declared reference denominator; it is not a measured peak.

The active candidate retains the 32 original eight-query selection unions and
their ordered page lists. It creates 64 real M64 work items by assigning the two
original contiguous row halves to separate CTAs. Selection grouping remains 8;
this does not execute the rejected four-query-union candidate. The explicit
`q128_compute_halves=2` option applies only to actual Q128 with two KV heads.
Other geometries and the separately addressable control retain original Group8.
Every launched CTA must own attention work; dedicated or idle fetch CTAs remain
outside this contract.

The successor starts from the frozen pair-priority stage, retaining its queue
and preparation order. Each of the three spare producer warps independently
claims one stripe and uses 32 lanes with stride32, full-warp synchronization and
the existing writer-fence/acq_rel-ready/TMA-acquire chain. Fetch still uses96
threads per CTA and24/240 registers. Original halves1, serialized fetch, compute
geometry and all math remain unchanged. No extra HBM workspace or launch is
introduced. Copy-marker meaning is preserved with the new warp synchronization;
actual selected instructions and fresh numerical/runtime evidence are required.

## Inputs and evidence

- Accepted serving run: `nosa_motivation_sm90_20261004_02`, source identity `6e3dfd17a86dd87be4ec89f0bfccc9bb25a52c5af773f5a5f2ed0753ba5c3e0c`. All 128 complete candidate hidden outputs exactly matched HBM in its four-method, 32-request trace.
- Accepted diagnostic: `nosa_motivation_current_diagnostic_20261004_02`, request 16, sample 0. Its saved per-layer selection, pre-call ownership tags, and independently audited miss sets define the initial offload state.
- Actual Q/K/V/CIS and output hashes come from the completed, audited attention-reference capture corresponding to that accepted run. Do not consume partially written capture data. The replay manifest must bind the capture identity, request, layer, call, selection, ownership evidence, and file hashes.
- Initial representative layers: 0 (204 miss pages, representative low union), 16 (449 pages; head split 335/114; lowest observed overlap), and 23 (563 pages, largest observed miss union). This selection is grounded in the current diagnostic, not synthetic inputs.

## Required invariants

1. Exactly one `.cv` host load per historical K/V vector in the unique selected miss union. No host-cache/L2 reuse assumption and no repeated loads per query group.
2. All participating CTAs execute real attention work. Fetch uses spare producer warps within those CTAs. Preserve cooperative residency and per-CTA register-pool safety.
3. Preserve eight disjoint 8-token stripes per 64-token page. Writer fences, fetch synchronization (96 threads on the original path; a full warp per independent stripe on the successor), acq_rel page completion, and TMA acquire/async-proxy ordering remain correct.
4. Page envelopes remain the min(start)/max(end) of the page's nonempty stripe intervals. Report page-envelope and stripe-copy overlap independently, using actual instrumented work. Never extend windows to improve ratios.
5. The row-half specialization has 256 threads, one consumer warpgroup and a 160-participant query-empty handshake. The queue-priority successor retains `128*24 + 128*240 = 33792` registers of demand, checked against `ceil(numRegs*32/256)*256*(threads/32)`. Original halves1 retains 24/240 for its two consumers. The matching serial-half path has its own 32/160 limits and 24,576-register demand. Preserve occupancy checks and do not reuse another specialization's compiled attributes.
6. Shared workspace planning, capacity checks, trace rows, output stores, fallback flags, and native build fingerprints cover the selected geometry. No unbudgeted allocations, silent fallback, selection truncation, or omitted helper time.
7. Both halves consume the same original Group8 page order, 128-token pairing and eight-bit membership masks. Preserve the `0xff` shortcut's original selection-group start, physical KV/CIS heads, both original fallback bins and disjoint output ownership. Pages, members and counts remain union-indexed; math trace rows use the 64 compute work items. Report duplicated HBM-to-shared K/V and CIS work separately from unchanged unique host bytes.

## Promotion criteria

The successor first has a separate, representative early-rejection screen:
compile/resource safety, layer16 exact/FP32/unique-byte checks, then up to three
internal samples with immediate rejection on either ratio below 0.90. Passing
that screen only permits the full gates below; it cannot promote the candidate.

- Verify the original Group8 control against every saved attention-output hash, then require exact saved-hash equality for every same-input row-half output across all 32 layers in both serial and async modes. Stop at the first mismatch. Independent FP32 `atol=rtol=0.016` is an additional check and cannot excuse a nonexact result.
- Only after that exact operator gate, run the independent full 32-layer fixture with two users/two visits and independently built empty-prefix controls. Preserve the existing full-hidden `atol=rtol=0.016`, report exactness separately and keep retained-output lifetime checks exact. No threshold changes after a failure.
- Independent unique-page/byte audit passes for every profiled layer and sample.
- Both page-envelope and nonempty-stripe-copy overlap ratios are at least 0.90 for **every** applicable layer/sample, not only their median.
- After correctness acceptance, measure 31 uninstrumented complete API samples, including initialization, planning, compaction, prepare, repair and launch gaps. Async must beat original serial Group8; also report the matching serial-half control. Three internal samples must each satisfy both overlap gates.
- Complete serving remeasurement has a new run ID, source/dependency identity, full numerical audit, memory audit, and trace integrity audit. Operator replay and NCU timing do not establish serving performance.

Baseline NCU has been collected and reviewed. Its evidence remains valid within
the recorded implementation and profiling boundary. The coordinator owns GPU
scheduling. Candidate runtime and harness changes stay outside the repository;
existing valid reports and run artifacts remain until an accepted replacement
is published. Group4 and V-first rejection evidence is summarized in
[checkpoint.md](checkpoint.md) and [vfirst_plan.md](vfirst_plan.md).

## Validation and evaluation entry points

Current gates and historical baseline commands are indexed in
[implementation_plan.md](implementation_plan.md). The active candidate's concrete
contract is [independent_warps_plan.md](independent_warps_plan.md). Planned validation is
distinct from completed baseline and rejected-candidate measurements.
