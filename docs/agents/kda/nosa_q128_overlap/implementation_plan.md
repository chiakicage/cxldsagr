# NOSA Q128 execution index

Status on 2026-10-04: all tested Q128 successors are rejected for promotion,
including independent producer-warp stripes. That candidate passed partial20 and
representative correctness, then stopped at its first internal sample with both
ratios0.800550. Runtime is unchanged. Current64-CTA profiling under
[independent_warps_ncu_plan.md](independent_warps_ncu_plan.md) is complete;
[findings](independent_warps_ncu_findings.md) identify host-data dependencies,
consumer K/V waits and 12.5% host-sector amplification. The completed
[controlled alignment result](aligned_host_diagnostic_findings.md) shows18→16
host sectors per warp with exact outputs and unchanged payload. The subsequent
aligned-buffer screen stopped at its first internal sample: both ratios were
0.8408366534, below 0.90. The subsequent [four-pair ILP candidate](four_pair_ilp_plan.md)
passed static review, partial20 and both representative exact controls, then
rejected its first internal sample at 0.7027860967 for both metrics. Original
unpadded host buffers had residue16; numerical, payload, identity and cleanup
checks passed. No later sample, API latency, all32 or full32 candidate gate ran.
No tested Q128 native successor is selected or integrated this cycle.

The latest completed implementation contract is
[four_pair_ilp_plan.md](four_pair_ilp_plan.md), preceded by
[independent_warps_plan.md](independent_warps_plan.md),
[frontier_fanout_plan.md](frontier_fanout_plan.md),
[pair_priority_plan.md](pair_priority_plan.md),
[stripe2_ilp_plan.md](stripe2_ilp_plan.md) and
[union_halves_plan.md](union_halves_plan.md). Decisions and evidence are in
[checkpoint.md](checkpoint.md).

## Executed row-half validation order

1. Finish staged source review, CPU resource/trace checks and compilation of the
   original and split specializations. Check selected register allocation,
   occupancy and the 160-participant handshake before any split GPU launch.
2. Under the coordinator's GPU allocation, verify the original Group8 saved hash
   first and require exact saved-hash equality for every split output across all
   32 captured layers, in both serial-half and async-half modes. Stop at the first
   mismatch; independent FP32 0.016 tolerance cannot excuse it. Restore ownership,
   poison misses and check unique bytes/stripe identities.
3. Only after the exact operator gate, run the independent full 32-layer fixture
   with two users/two visits, empty-prefix controls and the split option effective
   before planning/allocation. Preserve full-hidden and retained-output contracts.
4. Only after correctness acceptance, measure 31 complete-API samples against
   original serial Group8 and matching serial-half controls, followed by three
   internal profiles. Every page/stripe ratio must reach 0.90 and async must beat
   original serial Group8. Only then may the coordinator integrate the candidate.
   Recheck correctness on the integrated source, then rerun formal serving and
   affected experiments before publishing or replacing valid reports. Staged
   evidence does not substitute for these integrated-source checks.

The completed row-half stage is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_union_halves_candidate_20261004/`.
No additional candidate or performance run is implied by this index. GPU scheduling
remains with the coordinator. Current source/compile progress does not constitute
numerical or performance acceptance.

The baseline procedures below are retained for reproducibility. They describe
original Group8 diagnostics; they are not pending candidate directions or evidence
that additional layers were measured. The accepted full/source report is run
`nosa_q128_ncu_baseline_20261004_02`, documented in [checkpoint.md](checkpoint.md).

## Retained baseline procedure: immutable actual-input replay

Stage source at `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_overlap_stage_20261004/`. Use the completed attention-reference `capture_000016` and the accepted diagnostic directory `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_motivation_current_profile/data/nosa_motivation_current_diagnostic_20261004_02`.

The replay CLI must accept `--capture-dir`, `--diagnostic-dir`, `--layer`, `--method async_sparse|sync_sparse`, and `--mode audit|validate|profile`. CPU audit verifies accepted identities, file hashes, actual operand shapes/dtypes/strides, selection equality, ownership tags, and a separately recomputed unique miss set. It must reject a partial capture or any mismatch rather than silently substituting generated inputs.

GPU replay uses `NosaFetchWorkspace.run` and the current native implementation, with captured Q layout, actual pinned host history, actual candidate suffix and CIS, and original cache tags. It restores all tags and initial hit values, poisons nonowned history records, clears writable metadata/scratch, and synchronizes before every invocation. Warmup occurs outside NVTX and is followed by a fresh reset. Profile mode executes exactly one API call inside `nosa_q128_baseline`; output and byte verification occur after this range. Validation mode additionally exports true internal intervals and checks their existing independent auditors. The same captured state is restored on every application replay.

The profile driver records the exact build flags, source identities, mapped native library hashes, hardware, input hashes, and expected/observed transfer bytes. No CUDA graph capture is used. Existing native builds already contain `-lineinfo`, so the original API is the profiling binary; an independently rewritten CUDA launcher is unnecessary.

## Retained baseline command templates

The original command templates below are preserved. For a separately authorized rerun, set `Q128_CAPTURE` to the finalized, audited `capture_000016` directory and choose a fresh `Q128_RUN`; do not overwrite existing runs or use a partially written capture. The template ID is not the accepted report ID cited above.

```bash
Q128_STAGE=/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_overlap_stage_20261004
Q128_DIAGNOSTIC=/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_motivation_current_profile/data/nosa_motivation_current_diagnostic_20261004_02
Q128_RUN=nosa_q128_ncu_baseline_20261004_01
Q128_OUT=/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_ncu_runs/$Q128_RUN
.venv/bin/python "$Q128_STAGE/replay.py" --capture-dir "$Q128_CAPTURE" --diagnostic-dir "$Q128_DIAGNOSTIC" --layer 16 --method async_sparse --mode audit
.venv/bin/python "$Q128_STAGE/replay.py" --capture-dir "$Q128_CAPTURE" --diagnostic-dir "$Q128_DIAGNOSTIC" --layer 16 --method async_sparse --mode validate --warmup 2
.venv/bin/python "$Q128_STAGE/replay.py" --capture-dir "$Q128_CAPTURE" --diagnostic-dir "$Q128_DIAGNOSTIC" --layer 16 --method sync_sparse --mode validate --warmup 2
```

The staged `collect.sh` creates a fresh SSD temporary run directory with `data/`, `log/`, and `profile/`; it rejects existing runs and preserves child failures. It copies the staged harness and exact arguments into `data/`. No GPU collection writes into `experiments/`. After numerical validation, NCU integrity checks, and coordinator review, a successful collection may be published under the experiment's `output/data`, `output/log`, and `output/profile` with its run ID. Failed or incomplete collections stay outside the experiment deliverables and must not be published.

NCU 2026.1.1 is installed. Its CLI lists `PmSampling`, `PmSampling_WarpStates`, and `SourceCounters`, but no `source` set. Use these actual supported commands after validation:

```bash
ncu --replay-mode application --app-replay-mode strict --app-replay-match grid --nvtx --nvtx-include 'nosa_q128_baseline/' --set full --section PmSampling --section PmSampling_WarpStates --import-source yes --kernel-name-base demangled -k 'regex:nosa_offload_fused::fused_main' -c 1 -o "$Q128_OUT/profile/full_async_l16" .venv/bin/python "$Q128_STAGE/replay.py" --capture-dir "$Q128_CAPTURE" --diagnostic-dir "$Q128_DIAGNOSTIC" --layer 16 --method async_sparse --mode profile --warmup 2
ncu --replay-mode application --app-replay-mode strict --app-replay-match grid --nvtx --nvtx-include 'nosa_q128_baseline/' --set basic --section SourceCounters --import-source yes --kernel-name-base demangled -k 'regex:nosa_offload_fused::fused_main' -c 1 -o "$Q128_OUT/profile/source_async_l16" .venv/bin/python "$Q128_STAGE/replay.py" --capture-dir "$Q128_CAPTURE" --diagnostic-dir "$Q128_DIAGNOSTIC" --layer 16 --method async_sparse --mode profile --warmup 2
```

Repeat for layers 0 and 23. Serial control uses `--method sync_sparse` and kernel filter `regex:nosa_offload_fused::serialized_fetch`; collect the original FA3 kernel separately if counters are needed to explain consumer behavior. Do not use `--kill`: the driver must finish its output/byte checks. Retain separate stdout/stderr files for each command.

Application replay is deliberate: per-kernel replay must not inherit mutated ready flags, fetch cursors, ownership tags, or already populated miss records. Keep `.cv` host loads. NCU is a bottleneck probe; its replay times, cache flushes, and clock policy are not the serving latency measurement. Any unavailable SM90 PM or source metric is reported as unavailable, not zero or silently replaced by an SM100 metric.

## Completed baseline diagnosis

The coordinator reviewed actual metrics across launch/occupancy, work balance, stalls, tensor-core use, time-varying behavior and memory traffic. Raw values, units, source PCs and replay-pass boundaries are retained with the reports. Direct-host payload is distinct from HBM counters; stall samples do not measure the full duration of the gaps between softmax intervals. See [checkpoint.md](checkpoint.md) for the supported findings and limits.

## Rejected candidates

Group4 failed full-model numerical acceptance and all representative overlap
samples despite a faster representative API. Original Group8 V-first preserved
layer-16 exactness but failed complete-API latency and all three overlap samples.
Neither remains an active implementation direction. Their scoped evidence is in
[checkpoint.md](checkpoint.md), [investigation_log.md](investigation_log.md) and
[vfirst_plan.md](vfirst_plan.md). Their stages remain outside experiments for the
recorded engineering boundary; valid baseline reports have not been replaced.

## CPU/NUMA environment boundary

The staged harness records actual CPU affinity, allowed CPU/memory-node sets, PyTorch intra-/inter-op thread counts, calling-thread NUMA memory policy, and relevant threading/allocator environment variables before and after replay. The collection script uses the project's OMP/MKL default of 8 unless explicitly overridden, inherits CPU/NUMA binding, and records it. It does not claim that the old `_02` metadata recorded these fields: it did not. New provenance records include these fields and require matching profile environments. Old pairs without these fields remain explicitly unrecorded; a new recorded environment cannot be equated to an old unrecorded one. Default NUMA policy is not proof of each pinned allocation's physical placement.
