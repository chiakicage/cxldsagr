# Shared dependency count within earliest-pair priority ties

Status: one isolated successor authorized on 2026-10-04. The earliest-pair
candidate is rejected at its first 0.8395512653 overlap sample. This plan does
not authorize integration or inherit its correctness as new-source acceptance.

Stage:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_frontier_fanout_candidate_20261004/`.
Parent:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_pair_priority_candidate_20261004/`.
Keep that parent and earlier rejected stages frozen. Copy only required source
and helper context; do not copy old results into successor output directories.

## Evidence and hypothesis

The parent preserves exact representative outputs and the complete 449-page miss
queue. Its uncovered copy is 17.344 us startup plus 41.696 us across 103 internal
gaps, with no final copy tail. Only 7.840 us of internal uncovered copy occurs
after head 1 finishes. A tail-only policy therefore does not address most residual
time. Copy-end observations are not ready/TMA or precise stall-cause evidence.

At equal earliest consumption rank, the existing tie uses only block-major
ordinal. During the largest 3.680 us gap, head 1 block 996 remains copy-pending for
24 compute works (12 original unions); its earliest-rank fanout is 8. A fixed
fanout tie moves it from queue position 60 to 57. On this captured input the policy
moves 235 of 449 pages, with maximum advance 16 and delay 4 positions. These are
dependency and queue-position observations, not a prediction of shorter gaps or
API latency. Singleton dependencies may be delayed. The fixed parent union
lengths cap even perfectly aligned coverage at 0.8847725889, so success requires
changing actual copy/consumer timing, not reinterpreting windows.

The selected hypothesis is lexicographic priority:
1. minimum real original-union consumption-pair rank, ascending;
2. number of original unions using that page at exactly that minimum rank,
   descending;
3. the existing block-major ordinal, ascending.

There are no tunable weights or normalized-progress scores. Duplicate compute
halves do not count as additional original-union dependencies.

## Fixed implementation scope

- Change only the already bounded Q128/two-KV-head halves2 priority compactor
  and corresponding metadata. Keep preparation/launch order and launch count,
  all math, selection, page pairing, compute scheduling, repair and output
  ownership unchanged. Preserve the one-stripe fetch body, 24/240 registers,
  barriers/fences/acq_rel publication, trace markers/layout and both metrics.
- Retain the 4096-slot limit, 256-thread compactor, 16 KiB shared key array and
  original compactor fallback above that limit. No added HBM workspace.
- Use 7 rank bits, 6 tie bits and 12 ordinal bits. Initialize selected misses with
  unranked priority 64 and tie 32. The first reference pass installs minimum rank
  by shared atomicMin with tie 32 unchanged. After a full CTA barrier, a second
  reference pass decrements the tie field once for each original-union reference
  whose real pair rank equals the stored minimum. This may use shared atomicSub
  by `1 << 12`, provided review proves no carry/borrow can affect other fields.
- Each prepared union contains each logical page once. At Q128/two heads only
  16 original unions can reference a given physical page/head; 32 is a conservative
  tie-field initial bound. Explicitly test distinct-union counting and field
  isolation. Do not count compute halves or references at later ranks.
- Preserve rank derivation from the full prepared count, including suffix
  positions, then filter historical misses. No dummy padded page 0 dependency.
  Zero counts or invalid counts read no stale pages; their selected misses remain
  unranked with fanout 0. Preserve first_use, queue membership/cursor, byte
  attribution and all-hit/empty behavior.
- After the second pass and barrier, retain the existing bitonic sort with exclusive
  pair writers and CTA stage barriers. Decode the unchanged low 12 ordinal bits.
- Expose the actual tie policy and packed-field bounds in build metadata.
  Bind source/includes, canonical one-prefix venv PATH, compiler/header identity,
  selected binary and helpers. Verify both fused-main bodies remain identical
  to their actual parent images; added compactor work is included in API timing.

## Ownership and gates

Experiment owns native edits, CPU reference/field/ownership tests, compilation
and the expanded partial fixture. Cache owns the new sibling early-screen helper,
independent queue expectation and freeze. HBM-audit reviews field concurrency,
resources, parent main identity and binding. Root owns GPU allocation and gates.

1. Verify CPU permutations for empty/all-hit/full and bounded-fallback queues,
   rank ties, fanout 0/1/maximum, duplicate input selection versus unique prepared
   membership, odd counts/suffix filtering, invalid counts and packed field
   isolation. Verify the captured 449-page order against an independent oracle.
   Compile only after source review; inspect actual compactor resources and
   unchanged fused-main SASS. Keep every intermediate artifact outside experiments.
2. Freeze fresh identities. Under root allocation, run 20 partial GPU4/NUMA1
   checks: H=1,7,8,9,63, all_blocks/complete_blocks, original/successor. CPUs 72-79,
   intra-op/OMP/MKL/OpenBLAS/registration 8, GC and allocator guards remain enabled.
   Preserve exact outputs, fixed FP32 tolerance, poison/payload/unique bytes,
   actual queue/permutation, prepared counts/pages, raw physical softmax rows,
   page/stripe envelopes/readiness and cleanup. Complete mode requires all 32
   unions active and fallback 0; mixed mode must retain its real mixed behavior.
3. After partial acceptance, root allocates GPU0/NUMA0, full NUMA0 affinity and
   the same explicit thread settings. Check original Group8 and successor on
   accepted layer 16/request 16 inputs, then at most 3 internal samples with fresh
   restored workspaces. Verify actual new queue order. Stop at the first
   numerical/lifecycle/identity failure or either ratio below 0.90. A passing
   screen is not promotion or full-model acceptance.
4. Only a passing screen advances to fresh all 32 exact saved hashes in serial
   and async, the independent full32 fixture, then 31 complete API samples with
   original serial Group8, parent async and matching serial controls, plus 3
   internal samples. Both candidate medians must beat original serial, and every
   page/stripe ratio must meet 0.90. Include preparation/sorting/repair/launch gaps.
5. Only after those gates consider integration, integrated-source correctness,
   formal four-method serving, matching profiles, independent API reference and
   affected experiment reruns. Retain accepted reports until audited replacements
   are published. Record rejection before choosing another implementation.

No artificial delays, metric/window changes, tolerance changes or workload
substitution are permitted. This is a bounded dependency-order experiment with
an uncertain result; it supplies no promise that 0.90 is achievable.
