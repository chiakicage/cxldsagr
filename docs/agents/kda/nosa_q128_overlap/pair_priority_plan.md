# Earliest-consumption priority for the existing unique fetch queue

Status: one isolated candidate authorized on 2026-10-04. No integration is
authorized by this plan. The prior two-stripe candidate is rejected; its first
page/stripe overlap ratio was 0.6207038995373431 despite exact representative
outputs. Its source and results remain frozen outside experiments.

Stage:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_pair_priority_candidate_20261004/`.
Start from the frozen row-half parent
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_union_halves_candidate_20261004/`,
not the rejected two-stripe implementation. Keep one-stripe claims, original
copy markers, 24/240 registers and the parent fused mainloop unchanged.

## Hypothesis and limits

Q128 has 32 original Group8 unions and uses a block-major fetch queue: page 0,
then descending block numbers. Each union consumes its own ascending prepared
page list in descending pairs. Block number is therefore only a proxy for when
that union needs a page. On the captured layer 16/request 16 inputs, the 94 miss
pages needed through pair rank 15 require a 128-page prefix of the current queue;
the 190 needed through rank 23 require a 329-page prefix. Earlier rank priority
could remove these unrelated pages from the dependency frontier.

This is a queue-position observation, not a latency prediction. Some head 0
dependencies move later, and equal rank buckets may bunch consumer work. The
producer issues K(next) before V(current), so completed copies of a mathematical
pair alone do not prove that its attention is ready. Added sorting cost must be
included in complete API timing. The evidence neither establishes nor rules out
the 0.90 gate. A CPU-only union-priority alternative may inform diagnosis, but
only the minimum-pair-rank implementation below is selected for this iteration.

## Fixed implementation scope

- Change only the existing Q128/two-KV-head halves2 path. No new public selector.
  Original halves1, Q1024, serialized sparse, math, group membership, selection,
  page pairing, repair, compute scheduling and output ownership stay unchanged.
- On the selected path, run the existing `original::prepare` before the new
  compactor on the same stream. Preserve the launch count. Keep the original
  launch order on other paths; do not duplicate preparation.
- For each original union with `0 < count <= 128`, each real prepared index
  `i < count` contributes rank `ceil(count / 2) - 1 - floor(i / 2)` to its
  physical historical page/head. Minimize ranks across all original unions.
  Count includes suffix positions; filter historical misses only after deriving
  this rank. Do not re-pair after filtering. The fused loader maps padded slots
  beyond count to OOB, so prepare's dummy page 0 contributes no dependency.
- Selected misses are exactly `first_use != INT_MAX && ready != 8`. Do not
  overwrite `first_use`, which still determines transfer statistics and cache
  publication. Zero counts or invalid counts read no stale prepared pages. Every
  selected miss with no valid rank remains in the queue at trailing rank 64.
- For up to 4096 physical history slots, use a single 256-thread CTA with
  4096 shared uint32 keys (16 KiB), no additional HBM workspace. Pack rank above
  the 12-bit original block-major ordinal. Unique ordinals give deterministic
  ties. Nonselected and padded sort entries are UINT_MAX. Sort the power-of-two
  extent covering the physical slot count, decode ordinals back to physical
  slots, and publish the same unique queue count and zero cursor. For larger
  slot ranges use the unchanged original compactor.
- In each bitonic stage, only the owner of `i < (i xor distance)` reads/writes
  both members of that pair; the full CTA synchronizes before the next stage.
  Explicitly cover empty queues, boundary slot counts and count publication.
- Preserve `fetch_spare_body`, fused mainloop, stripe claims and unique vector
  ownership, barriers/fences/acq_rel publication, byte attribution and trace
  layout. Both overlap definitions and actual softmax-only windows are unchanged.
- Expose actual queue-priority scope and bounded fallback in build metadata.
  Bind actual source/local includes, compiler/header identity and selected
  binaries. The canonical PATH contains exactly one venv-bin prefix for compile,
  metadata freezing and GPU execution, retaining production allocator identity.

## Ownership and gates

Experiment owns staged native changes, CPU ownership/order tests, compilation
and the adapted partial fixture. Cache owns the fresh early-screen helper and
freeze. HBM-audit independently reviews source/dataflow, sort ownership,
resources and bindings. Root owns GPU allocation and gate decisions.

1. Compare against a CPU reference permutation, covering empty/all-hit queues,
   ties, fallback counts, odd counts, suffix filtering, slot bounds, unique queue
   membership and the captured layer 16 order. Inspect selected compactor resources
   and verify parent fused-main SASS remains identical; retain actual register,
   shared-storage and occupancy guards. No numerical claim follows from this gate.
2. Freeze source/helper/header/build/binary identities. Under root allocation,
   run fresh Q128/two-head H=1,7,8,9,63 partial correctness on GPU4/NUMA1,
   CPUs 72-79, eight intra-op/OMP/MKL/OpenBLAS/registration threads. Preserve
   poisoned payload, original exactness, fixed FP32 tolerance, physical trace,
   readiness and cleanup checks; additionally verify actual queue permutation.
   Each H has two independent original/successor pairs: select all logical blocks
   to exercise the existing partial-tail fallback, then select only the complete
   logical blocks to exercise active fused consumers of partial host history.
   The second mode must have nonzero prepared counts, nonempty softmax intervals
   and no fallback flags. These 20 calls use synthetic correctness inputs only.
3. After partial acceptance, root allocates GPU0/NUMA0 with full NUMA0 affinity
   and the same explicit thread settings. On accepted layer 16/request 16 inputs,
   check original Group8, then successor exact saved hash, FP32 tolerance,
   unique bytes, actual payload/queue order and cleanup. Only then collect up to
   three internal samples, each from restored state. Stop immediately on any
   numerical/lifecycle/identity failure or either overlap ratio below 0.90.
4. Only a passing screen permits checks of all 32 layers in both serial/async modes,
   the independent full32 fixture, then 31 complete-API samples with original
   serial Group8, parent async and matching serial controls and three internal
   samples. Both candidate median latencies must beat original serial; every
   page and stripe ratio must pass 0.90. Parent correctness is not acceptance of
   this changed implementation.
5. Only after acceptance consider integration, integrated-source correctness,
   formal four-method serving, independent API reference and affected experiment
   reruns. Keep accepted reports until audited replacements are published.

All development artifacts stay outside experiments. Preserve GC and allocator
guards, register8, cold admission and original score dispatch. Do not widen
intervals, add idle compute delays, change tolerances or alter the workload to
pass. Record a failed hypothesis before selecting another implementation.
