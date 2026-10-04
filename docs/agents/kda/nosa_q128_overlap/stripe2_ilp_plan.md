# Two-stripe host-load ILP on the validated Group8 row halves

Status: one staged successor authorized, 2026-10-04. No integration or GPU
launch is authorized before the corresponding gates below. The previous row-half
candidate is rejected for promotion; its exact operator/full-model evidence is
parent evidence, not acceptance of this changed implementation.

Stage:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_stripe2_ilp_candidate_20261004/`.
Parent:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_union_halves_candidate_20261004/`.
Keep both the parent source and its results frozen. Copy only required source and
helper context into the successor and create fresh manifests. Do not copy parent
results into successor result directories or modify repository runtime files.

## Evidence and hypothesis

The parent passed all 64 saved Group8 output hashes and the independent full32
fixture, but failed all three formal overlap samples: 0.710582, 0.728228, 0.695460.
Its complete API CUDA/wall medians were 0.632576/0.669274 ms versus original serial
Group8's 0.644640/0.669534 ms. The wall difference does not establish a robust gain.

The selected parent sm_90a SASS issues a K/V pair of
`LDG.E.128.STRONG.SYS` loads followed by a dependent store before the next loop
iteration. Baseline original-Group8 source NCU attributed 1583 long-scoreboard
samples to that store, with substantially larger TMA/consumer-wait counts.
Batching independent stripe loads could increase outstanding host requests and
reduce these dependencies. This is a bounded hypothesis; it does not predict a
90% result or remove the observed per-head queue dependencies.

## Fixed implementation scope

- Change only the Q128/two-KV-head halves2 specialization. Keep the original
  halves1 spare-fetch body, shared layout, 24/240 register split, Q1024 dispatch
  and other geometry paths unchanged.
- Preserve every original Group8 union, member bit, page order/pair, compute
  half, mask, reduction, mathematical traversal and repair decision. Keep the
  same physical unique queue and its ordering. No Group4 union, V-first change,
  dedicated fetch CTA, selection change or host-memory L2 assumption.
- One leader `atomicAdd(cursor, 2)` claims an even/odd adjacent stripe pair.
  With eight stripes/page and a zero cursor, both tasks belong to the same
  physical page. Broadcast one claimed base; derive the second task. Do not use
  two separately interleavable claims. Retain one unique owner per vector.
- Fully unroll two stripes and both vector chunks: issue all active K/V `.cv`
  loads into registers before any dependent payload store. Keep 96 fetch threads
  and the existing per-stripe writer fence, 96-thread barrier, acq_rel ready RMW,
  once-per-page byte count and post-publication barrier. Every page still gets
  eight increments; empty stripes perform no host load but complete normally.
- Use 64 producer registers for all four producer warps of halves2, including
  its TMA warp, and keep 240 consumer registers. The parent pool was 61,440
  registers; 128×64+128×240=38,912 would fit that pool, but the rebuilt selected
  kernel must independently pass the actual allocation and occupancy guard.
  Applying 64/240 to original halves1 would exceed its pool and is prohibited.
- No new public selector is needed. This successor changes only the existing
  halves2 implementation in an isolated stage. Native `build_info` must expose
  actual stripe batch size, producer/consumer budgets and trace-start mode;
  source fingerprints and actual selected binaries remain authoritative.

## Copy interval semantics

Never give both stripes a shared batch-start timestamp or add a 96-thread barrier
between their load phases. The former would widen stripe2's interval; the latter
would wait for prior loads and defeat the intended cross-stripe ILP.

Use shared first-load timestamps indexed by stripe and fetch warp (`[2][3]`).
Each warp's lane0 clears its own entries every batch and records a timestamp
immediately before its first actual host load for each nonempty stripe. Empty
warp entries stay zero. Any active warp contains lane0 for this contiguous vector
mapping. After each stripe's writer fence/barrier, the leader reduces positive
starts to their minimum and records that stripe's existing completion boundary.
Capture the end timestamp immediately after the writer fence/barrier and before
the new minimum reduction or trace/page atomics; those bookkeeping operations
must not lengthen the copy window.
Preserve unique stripe rows, page start/end atomics and the post-publication
barrier before shared-state reuse. Page envelopes must exactly equal the min/max
of their nonempty stripe rows, regardless of completion order. Preserve both
overlap definitions and the existing softmax-only math windows.

The selected SASS must confirm marker/load/store ordering; source placement is
insufficient. Describe these as instrumented copy windows, not precise memory
transaction issue/completion times. No WGMMA issue-to-delayed-wait envelope may
replace actual softmax windows or count an entire kernel as attention work.

## Ownership and gates

Experiment agent owns staged native edits and compilation. Cache agent owns a
separate early-rejection helper and its CPU checks. HBM-audit agent independently
reviews native code, trace semantics, selected SASS/resources and helper binding.
Root owns gate decisions, GPU scheduling and any eventual integration.

1. Before compilation, audit unique ownership for token counts 0..64, both empty
   and partial stripes, all 96 participants, shared timestamp reuse and exact
   ready publication. Check that halves1 body/layout and all math are unchanged.
   Bind the successor inspector's register demand to actual selected metadata;
   do not inherit the parent's hardcoded 24/240 assumption.
2. Compile both original and successor specializations. Inspect the selected
   sm_90a image, actual CTA register pool, shared storage, occupancy and spill
   sites. Require all intended payload loads before dependent stores and no
   payload spilling that defeats ILP. Distinguish consumer spills from payload
   staging. Freeze source, helper, header, build and binary identities.
3. Under root's GPU allocation, first cover actual Q128/two-head partial-history
   lengths 1, 7, 8, 9, 63 against original Group8 and the reference, with poisoned
   misses, unique payload, physical stripe/page audit and cleanup. This checks
   the new empty/partial-stripe branch absent from the full-page capture.
   After root allocates GPU0/NUMA0, run an explicitly labeled early-rejection
   screen on accepted request16/layer16. Verify original saved Group8 exactness
   first, then successor saved-hash equality, fixed FP32 tolerance, poisoned
   misses, unique bytes and cleanup. Only after that correctness check may the
   helper collect up to three separate internal samples, checking exact output,
   payload, page/stripe identities/envelopes and cleanup for every sample.
   Stop and reject on the first numerical/identity failure or either overlap
   ratio below 0.90. This screen cannot promote a candidate or claim full-model
   correctness. It does not reuse old full32 evidence as new-source acceptance.
4. Only a passing screen advances to all 32 layers × serial/async exact saved
   hashes, then the independent full32 fixture with its unchanged numerical and
   lifecycle contracts. Stop on the first nonexact operator output.
5. After those gates pass, run final 31-sample complete API controls and three
   internal samples on the frozen source. Original serial Group8 remains the
   latency gate; matching serial halves and the parent async halves are separate
   controls. Both candidate medians must beat original serial, and every page
   and stripe ratio must reach 0.90. A median ratio cannot hide a failed sample.
6. Only then consider integration, integrated correctness, final formal serving,
   independent API comparison and affected experiment remeasurement. Preserve
   accepted reports until validated replacements are published.

All development runs stay outside `experiments/`. Keep register8, GC enabled,
allocator/foreign-pool checks, cold-admission timing and original score dispatch.
No metric, numerical tolerance, request shape or capacity may be changed to pass
these gates. If the hypothesis fails, record rejection before choosing another
implementation direction.
