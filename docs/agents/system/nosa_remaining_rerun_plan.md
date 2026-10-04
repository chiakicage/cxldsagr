# Remaining NOSA reruns at integrated source 02b5d0

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

Current status on 2026-10-04: the A1024 offload replacement and all three
GR budget formal/profile families are complete, independently audited and
published. GR's exact superseded outputs were cleaned after installed hashes
and links passed; see the [GR receipt](nosa_gr_selected_source_execution.md).
No Q128 native candidate was selected.

The fixed selected-source trio at `a73ad32b...` is accepted and published, with
exact superseded-output cleanup complete. The
[fixed report](../../../experiments/nosa_motivation/README.md) and
[fixed publication record](nosa_motivation_publication.md) preserve the results
and receipts. Candidate MFU, tails and compute/IO dominance remain
unresolved, and the fixed async latency/90% overlap gates failed. The overall
efficiency goal is not complete.

The `02b5d0` source, NCU activity, GPU hold and launch restrictions in the
planning sections below describe the earlier checkpoint. They are not current
holds or instructions to rerun the completed work.

## Frozen scope and isolation

- Current motivation source inventory exactly matches the accepted integrated
  digest `02b5d0f9589c5e49257b40b73a5ddce4b2811f0614cc79d4afe242a2aa09f08b`.
  The offload harness additionally freezes its own 103 source files, not just
  that differently scoped digest. Its source map and six retained input files
  are in `freeze.json` under the external planning directory below.
- Current NCU run `nosa_q128_independent_warps_ncu_gpu0_20261004_01` uses GPU0,
  PCI `0000:18:00.0`, NUMA0, and the full NUMA0 CPU set. It has no concluded
  diagnostic evidence yet. Its prepared manifest SHA is
  `f9fdada62fe8943f21615272c1429275402791cf5eb1d9e1cdd738d36f5f8914`.
  The native runtime manifest has 114 entries, all unchanged at this audit;
  92 paths overlap the offload source map. Keep those files and all selected
  plan/helper/header/library dependencies byte-stable until NCU collection ends.
- Use physical GPU5, UUID `GPU-a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`,
  PCI `0000:ab:00.0`, NUMA1. Bind CPUs48–55 and memory node1, with eight
  OMP/MKL/OpenBLAS/PyTorch intra-op threads and register8. Keep the inter-op
  default separately recorded; do not change GC, clocks, allocator policy or
  backend between the three measurements. GPU5 showed 1 MiB used and no work
  during the read-only check; root must recheck availability at launch.
- The repository fused build key is `d34d3d8f463d5b26`; its existing binary SHA
  is `43f624f9d25a0fe7f0299e42a4e6d7ba840a2fb356fbd1fa14a66cb6b08ec05d`.
  Resident FA3 key `a020a4e1d3210e11` also already exists, SHA
  `f8fea3b6e044d45299b771ea653fad39477653a478241097ced44e59d0cc4f43`.
  Both were identified without loading an extension or initializing CUDA.
  They differ from NCU's staged fused keys `218ecd89ad4b0696` and
  `e7f67e9263e250c1`. Reuse valid artifacts; never delete, replace or force-build
  a selected cache entry. Other shared mapped libraries and the allocator
  adapter remain protected. If an expected artifact is absent or corrupt,
  stop for root review rather than rebuilding it during NCU.
- The runtime test and three measurements must execute serially on GPU5.
  Offload source and input hashes, native build identity, toolchain, libraries,
  NUMA binding and allocator selection must match across all three measurements.
  Capture external launch receipts because the offload metadata does not record
  CPU affinity, NUMA policy or all thread/allocator environment settings.
  Do not claim that the metadata's environment subset supplies those facts.

Planning evidence is outside experimental deliverables:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_dependent_rerun_20261004_01/`.
It contains `freeze.json`, `native_build_readiness.json` and
`workload_reopen_explained.json`. Source reads, CLI help and `bash -n` passed;
these checks are not model or performance acceptance.

## A1024 sequence and commands

The earlier ordinary H64K/A1024 gate was at source `6c75e3…`; the integrated
`02b5d0…` gate covered fixed-serving A128. Do not transfer either label to a
fresh ordinary A1024 result. Root requires the specific full-checkpoint test
below before publication. It uses the byte-identical request copy in the new
resident native run, builds resident/offload sparse prefixes independently,
and compares all 1,024 normalized hidden states after all 32 layers.

Run from the repository root on the allocated GPU5. The shell scripts add
the venv PATH prefix themselves; the direct pytest command adds it once.
Record the selected base PATH with each launch. None of these commands changes
CUDA device clocks, clears caches or writes source files.

```bash
nosa_rerun_tmp=/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_dependent_rerun_20261004_01/staging
mkdir -p "$nosa_rerun_tmp"
nosa_base_path="$(python3 -c 'import os; from pathlib import Path; v=str(Path.cwd()/".venv/bin"); print(":".join(x for x in os.environ["PATH"].split(":") if x != v))')"
nosa_run_env=(env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_HIP_ALLOC_CONF
  -u PYTORCH_NO_CUDA_MEMORY_CACHING -u PYTORCH_NO_HIP_MEMORY_CACHING
  -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES
  PATH="$nosa_base_path" CUDA_VISIBLE_DEVICES=5
  CXLDSAGR_SM90_BACKEND=native PYTHONDONTWRITEBYTECODE=1
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  PYTORCH_ALLOC_CONF=backend:native,pinned_use_cuda_host_register:True,pinned_num_register_threads:8
  TMPDIR="$nosa_rerun_tmp"
  numactl --physcpubind=48-55 --membind=1)

"${nosa_run_env[@]}" env PATH="$PWD/.venv/bin:$nosa_base_path" \
  NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  NOSA_OFFLOAD_REQUEST=experiments/indexer_block_sparse_profile/output/data/sparse_flags_native_20261004_01/request.json \
  .venv/bin/python -m pytest \
  models/nosa/tests/test_offload_checkpoint.py::test_cuda_offloaded_checkpoint_matches_independent_resident_64k_1k \
  -q -s --tb=short -p no:cacheprovider

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh \
  nosa_cached_fetch_20261004_01 --device cuda:0 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 20 --reference-all

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh \
  nosa_cached_fetch_confirm40_20261004_01 --device cuda:0 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 40 --reference-all

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh \
  nosa_cached_fetch_profile_20261004_01 --profile --device cuda:0 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 3 --reference-all
```

All three proposed run IDs were unused at planning time. Verify all data/log/
profile destinations again before each launch. Retain existing reports and
`nosa_fused_stripe8_head1*_20260930_01` outputs until replacement acceptance and
publication. Failed or partial runs remain in the scripts' external staging.

Minimum acceptance:

1. The fresh checkpoint test passes without skips. Record its source identity,
   request identity and numerical scope separately; it supplies no model timing.
   The old README's 62-test/global-test counts remain historical and are not
   attributed to this single new check.
2. Each operator run checks every query in L0/L15/L31 against independent FP32,
   serial versus fused output equality, and CPU unique-byte/first-use counts
   after validation, warmup and every measured call. All 20/40/3 samples remain.
3. Reopen all raw timing samples and profile windows independently. Check exact
   nonempty stripe ownership/bytes, page envelopes as min(start)/max(end), all
   nine samples' two overlap ratios, one fused main per fused range, and zero
   serial fetch/attention kernel overlap. A valid sample below 90% is retained
   as a failed method claim, not discarded as an invalid run. Compare complete
   API latency with the whole-query serial union; profile duration is not timing.
4. Rebuild both reports into new data directories. Recheck selected source,
   input, hardware, library and report identities. A good median alone does not
   establish the per-sample overlap claim. Preserve the frozen historical-input
   qualification; these runs do not recapture a current full-model trajectory.

After the three runs finish, use CPU-only reporting on NUMA1:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=48-55 --membind=1 .venv/bin/python \
  -m experiments.nosa_offload_overlap.src.report \
  --measurement-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_20261004_01 \
  --profile-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_profile_20261004_01 \
  --output-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_20261004_01/report
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=48-55 --membind=1 .venv/bin/python \
  -m experiments.nosa_offload_overlap.src.report \
  --measurement-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_confirm40_20261004_01 \
  --profile-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_profile_20261004_01 \
  --output-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_confirm40_20261004_01/report
```

## GR budget work: completed execution and retained CPU trace proof

The three NOSA-only formal runs and matching layer-31 diagnostics completed
as the `gr_nosa_flags_h4k/h16k/h64k_20261004_02` families, each with its own
matching profile ID. They preserved U=1/8/32/64/128/256/512, seed42, weighted
Beauty interaction-count heat, max-revisits8, A128/C1024 and 4/16 GiB admission
caps. H4K/H16K kept up to32 draws; H64K kept6, its context override and empty
revisit groups. No fixed-P/NH, uncapped IID or complete-loop substitution occurred.

CPU regeneration has checked all 21 populations: all 444 complete request
records match the originals exactly, implying 1,776 four-scheme executions.
Old and new aggregate workload hashes differ because the current config adds
`sampling=weighted` and `heat_field=interaction_count`. Adding only those fields
to each old identity reproduces every new hash. Preserve both hashes and this
mapping; never overwrite the old identities. The initial raw-hash assertion
failed after request comparisons completed; the explained audit records the
schema difference rather than treating it as changed request content.

Execution froze all 280 GR source files at
`8dc00e23469a67ebee67eb603859c9de9987e9151520a607d1bb20a498e547f9`,
including both model trees, runtime tests, `gr_serving` and
`deepseek_v32_echo_cache` Python/shell files. Each formal run preceded its
profile and independent audit. H4K/H16K used population8 and H64K population1
for three real revisits. All 1776 requests and 1332 non-HBM complete-hidden
comparisons passed; all nine valid overlap samples remained below90%.
Publication installed 57 report assets, then cleaned only the exact old six
NOSA run families. The postpublication source check passed; documentation and
report edits did not alter runtime/test/harness identity.

The current GR harness reports active-allocation bounds and separate allocator
allocated/reserved peaks. Those bounds exclude inactive cached blocks and
direct-library/driver overhead; passing the ledger does not establish a full
physical HBM bound. Its static `host_memory_placement` string is not measured
NUMA placement, and it does not retain the allocator adapter's selected provider.
Require external launch receipts for actual placement/configuration. If a report
claims `private_cpp` with no fallback, obtain explicit evidence from the target
process; a source digest or a separate preflight is insufficient. Preserve
physical-budget and peak limitations unless separate measurement closes them.

## A1024 completion and publication — 2026-10-04

The ordinary full32 H65536/A1024 checkpoint gate and all three offload runs
completed on GPU5/CPUs48–55/NUMA1 at integrated runtime digest
`02b5d0f9589c5e49257b40b73a5ddce4b2811f0614cc79d4afe242a2aa09f08b`.
The checkpoint test passed without skips: independent empty resident/offload
prefixes produced bitwise-equal normalized hidden for all 1,024 extend tokens,
with `max_abs=0`. This supplies numerical acceptance, not model timing.

Accepted runs are `nosa_cached_fetch_20261004_01` (20 unprofiled repeats),
`nosa_cached_fetch_confirm40_20261004_01` (40 independent unprofiled repeats),
and `nosa_cached_fetch_profile_20261004_01` (three profile repeats).
All use the retained frozen L0/L15/L31 inputs and `--reference-all`.
Independent raw-data review passed 657 timing checks and all 27 measured profile
ranges. Every fused profile sample passed both 90% overlap thresholds, with
exact nonempty stripe identities, bytes and page envelopes; serial fetch and
attention ranges were disjoint. The fresh SASS/resource/read-once receipt
records the selected source/build/binary identity and its evidence limits.

Root accepted and published the README, two JSON/CSV report pairs and current
kernel receipt. The tracked
[publication record](../../../experiments/nosa_offload_overlap/report/publication.json)
records the accepted run IDs and published hashes. Only after acceptance and
publication, root removed the exact inventory of nine superseded 20260930
data/log/profile directories containing 294 files. The durable completion
receipt is also saved at
`experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_20261004_01/publication/published.json`.
New runs, source snapshots, audit provenance and the retained historical input
capture remain available. Earlier offload launch and retention instructions in
this plan describe the completed sequence; they no longer denote pending work.

These results cover the default cold-prefix whole-union A1024 operator API.
They do not recapture a current model trajectory or measure finite P/NH cache
hits, fixed-serving performance or budget-serving performance. Those separate
serving results now have their own accepted evidence and publication records;
the earlier GR hold is closed. This completed A1024 publication does not itself
authorize new GPU work, runtime edits or additional cleanup.
