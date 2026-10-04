# Independent stripe workers in the three spare producer warps

Status: one isolated candidate selected on 2026-10-04 after independent feasibility
review. Frontier fanout is rejected at its first 0.8307389668 sample. This plan
authorizes staging and static preparation; root allocates each later GPU gate.
No parent correctness or performance result is new-source acceptance.

Stage:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_independent_warps_candidate_20261004/`.
Parent:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_pair_priority_candidate_20261004/`.
Preserve all rejected stages and accepted experiment reports. Copy only required
source/helper context, never parent results into the successor output tree.

## Evidence, hypothesis and limits

Earliest-pair priority reached 0.8395512653 in its first representative sample;
the fanout refinement reached 0.8307389668. Both preserve exact outputs and unique
bytes, but neither satisfies 0.90. The latter has 19.744 us startup and 43.584 us
internal uncovered copy, with no copy tail. These single samples do not establish
a statistical performance ranking or identify ready/TMA stall causes.

The baseline NCU source report identifies host-load-dependent stores, TMA waits
and fetch synchronization sites. It describes the older 32-CTA implementation;
it does not measure the current 64-CTA barrier cost. The parent fetch body gives
all three spare warps one stripe and synchronizes all 96 threads. A full stripe
has 128 uint4 K/V pairs: 32 lanes execute two iterations and 64 execute one.

The selected hypothesis removes cross-warp waiting by letting each spare warp
claim and finish its own stripe. There are still 96 fetch threads per CTA and
192 physical fetch warps across 64 CTAs. Independent in-flight stripes rise from
one to three per CTA; hardware warp count does not increase. Each warp instead
executes four vector iterations per lane, potentially increasing stripe latency
and atomic contention. This is a falsifiable scheduling experiment, with no
predicted overlap or API-latency pass.

Padded two-query quarters would require M64 tiles with only M32 useful rows,
128 compute works, more executed matrix/TMA/CIS work and new loading/masking/output
contracts. That broader decomposition is not selected.

## Fixed implementation scope

- Start from earliest-pair priority, without fanout or two-stripe ILP changes.
  Change only Q128/two-KV-head halves2 spare fetch scheduling and its metadata.
  Retain all 32 Group8 selections, 64 M64 compute works, page pairing, math,
  masks, repair, compute scheduler, minimum-pair-rank queue and prepare order.
  Keep original halves1, serialized fetch and other geometries unchanged.
- Each of producer warps 1,2,3 has one lane0 atomic queue claim. Broadcast the
  encoded task using full-mask shuffle. Use lane index `threadIdx.x & 31` and
  vector stride 32. Queue claims uniquely own stripe data/trace writes; no
  stripe or host vector is copied twice. No additional launch or HBM allocation.
- Remove every 96-thread fetch barrier and shared-slot access from the new path.
  Keep the unused `fetch_slot` storage initially to preserve SharedStorage layout.
  Uniform whole-warp termination is required. Preserve warp0 TMA and the common
  producer warpgroup register deallocation before role divergence.
- Preserve trace meaning: converge after decoding, lane0 records the start,
  synchronize all 32 lanes before any load, then execute all predicated copies.
  Every writer executes `__threadfence()`; a full-mask `__syncwarp()` joins the
  writes before lane0 records the end and publishes the existing acq_rel ready
  increment. Synchronize before claiming the next stripe. Shuffle alone supplies
  no memory-order guarantee. Keep the page min/max atomics and trace row layout.
- Empty stripes issue no host load or copy marker but still increment readiness
  exactly once. The eight acq_rel stripe publications form the existing chain;
  TMA still acquires ready=8 and executes the async-proxy fence. The last stripe
  completion attributes the full page bytes exactly once. Copy end remains
  distinct from readiness and physical memory-transaction completion.
- Preserve 24/240 dynamic register limits initially. Fresh compiled register,
  spill, actual CTA pool, SharedStorage and cooperative occupancy checks must
  pass before GPU execution. No silent register-budget changes.
- Valid claim and readiness atomic counts stay unchanged. Terminal claims become
  three per participating CTA for this path: cursor `8*pages + 3*fetch_ctas`.
  Target maximum is 192 terminal claims; the existing `max(grid,256)` overflow
  slack covers grid64. Do not extend the new worker policy to broader geometry.
- Expose worker count, threads per stripe, trace synchronization and cursor policy
  in build metadata. Bind source/includes, toolchain/header identity, actual
  selected images and helpers under the canonical one-prefix venv PATH. Halves1
  fused-main SASS must still match its actual parent. Halves2 whole-body identity
  is no longer a valid requirement because fetch is inlined: audit unchanged
  math source and actual fetch barriers/markers, then use fresh numerical gates.

## Ownership and gates

Experiment owns native staging, CPU ownership/cursor tests, exact-loader
compilation and the expanded partial fixture. Root owns the sibling screen
adaptation, decisions and GPU allocation. HBM-audit reviews synchronization,
resources, identity and result evidence. Cache separately handles unaffected
resident experiment work on GPU5; do not change integrated runtime during those
frozen runs. Compile on CPUs0–7/NUMA0 while GPU5 timing uses CPUs48–55/NUMA1.

1. CPU review must prove unique vector ownership for every partial length 0–64,
   full-warp participation/termination, empty/all-hit queues, exact terminal
   cursor and overflow bound, one page-byte attribution and unchanged queue.
   Review source before compilation; inspect actual selected SASS, resources,
   register pools and marker/fence ordering. Fresh freeze binds all artifacts.
2. Root allocates GPU4/NUMA1, CPUs72–79, for 20 partial checks: H=1,7,8,9,63,
   all_blocks/complete_blocks, original/successor. Preserve exact output pairs,
   fixed FP32 tolerance, poisoned payload, unique bytes, actual queue, prepared
   counts/pages, raw softmax and page/stripe/ready checks, and complete cleanup.
   Complete mode requires all 32 unions active and fallback0. Mixed mode retains
   its real mixed behavior. Partial correctness has no overlap threshold.
3. After partial acceptance, root allocates GPU0/NUMA0 with full NUMA0 affinity
   for accepted request16/layer16 input. Check original Group8 and successor
   correctness, then at most three independent internal samples. Stop at the
   first failure or either ratio below0.90. Preserve the fixed page-envelope and
   nonempty-stripe-copy definitions with actual softmax-only math windows.
4. Only a passing screen advances to fresh exact saved hashes for all32 layers
   in serial and async, then independent full32 empty-prefix/multi-user checks.
   After correctness, run 31 complete-API samples including all helpers and gaps:
   original serial Group8, pair-priority parent async, matching serial halves and
   candidate async. Both candidate CUDA/wall medians must beat original serial;
   three new internal samples must each pass both overlap gates.
5. Only after all gates consider integration, integrated-source validation,
   formal four-method serving/profiles/API reference and affected reruns. Publish
   audited replacements before removing superseded valid experiment artifacts.

All GPU gates retain GC, allocator/foreign-pool guards, canonical PATH and
register8 configuration, eight intra-op/OMP/MKL/OpenBLAS threads, original score
dispatch and tolerances. Record the unchanged inter-op default separately. No
artificial delay, metric/window widening, workload change or omitted helper time.
