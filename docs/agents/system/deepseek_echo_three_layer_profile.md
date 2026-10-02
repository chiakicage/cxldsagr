# DeepSeek ECHO first-three-layer profile

User request: recheck the ECHO prefill experiment, execute only checkpoint layers
0–2, rerun profiling, and report each operator's MFU.

## Measurement contract

- Real consecutive checkpoint layers 0–2, no repeated layer surrogate and no MoE.
- Preserve the experiment's 65,536-token prefix, 1,024-token extend, 1,024-token
  execution chunks, and 16,384 offload slots. Resident and ECHO use independent
  empty caches; each extend restores the same prefix cache residency.
- One idle SM90 device. Record UUID, PCI identity, clock state, software and all
  implementation sources. Do not change clocks or interfere with other jobs.
- Measure uninstrumented prefix and extend separately, then collect independent
  Nsight Systems captures for both phases and modes. Weight loading, compilation,
  state restoration and numerical comparison are outside measured ranges.
- Operator instrumentation wraps existing calls without replacing computation.
  Launch correlation, not CPU range containment of GPU timestamps, determines
  operator kernel time. Export every kernel and audit attribution conservation.
- Useful matrix FLOPs use actual dimensions, causal indexer pairs and valid
  sparse selections, with FMA = 2 FLOPs. MFU uses precision-specific dense peak;
  elementwise, sorting and data movement have MFU N/A, with latency still reported.
- Compare all extend normalized hidden and last-token logits, including profiled
  versus uninstrumented output. Save real layer inputs for separate NCU replay.
- NCU replay diagnoses selected kernels and does not replace in-context operator
  timing. Preserve the cold-pool replay boundary in its report.

## Publication

Use new run IDs and the experiment's standard output/data, output/log and
output/profile directories. Stage unsuccessful runs outside experiments. Publish
selected tables under report/layers3 and update README with scope and provenance.
The earlier complete-61-layer report is a different scope; three layers cannot
replace its outstanding post-fix full-model validation.

Status as of 2026-10-02: collection and original publication completed; baseline
acceptance is pending correction and rerun. The researcher subsequently identified
problems with the current ECHO implementation's MFU and cache strategy, and with
the current DeepSeek implementation's MFU. The existing MFU, performance
attribution, and ECHO comparison must not support an accepted baseline conclusion.
The root cause has not been established; this correction does not assert that a
particular FLOP formula is wrong. No kernel/cache fix or new GPU run was performed
as part of this status update.

Keep the original run IDs, measurements, tracked report assets and raw outputs
until corrected runs have passed numerical and measurement checks and have been
published. Then replace the affected report and remove superseded artifacts in
the same publication update. Regenerating the original report must preserve this
qualification rather than restore its previous baseline acceptance language.

## Evidence and outcome

- Main run: `20261002_echo_layers3_mfu_01`; physical GPU 1,
  `GPU-a2226185-cb05-a411-80da-f365154128fe`. PCI 2335 identifies H200 SXM.
- Prefix median: resident 2668.638 ms, offload 3540.019 ms (3 samples each).
  Extend median: resident 46.830 ms, offload 65.675 ms (5 samples each).
  Each mode/phase additionally has one independent annotated Nsight capture.
- All 8 numerical comparisons are bitwise equal, including every element of the
  1024x7168 extend normalized hidden. Prefix validation checks last-token logits.
- Independent CPU audit recalculated 5496 matrix-call FLOP counts and attributed
  all 136887 kernels independently. Per-layer compute rows total 172. No missing
  or duplicate query intervals; all metadata/capture counts and durations agree.
  These checks establish internal arithmetic and attribution consistency under
  the recorded contract, not adequate kernel efficiency or a sound cache policy.
  Historical `accepted` / `verified_mfu` fields remain evidence of those checks;
  they are not acceptance of the latest MFU/cache concerns.
- NCU runs: `20261002_echo_layers3_ncu_indexer_resident_01`,
  `20261002_echo_layers3_ncu_indexer_offload_01`,
  `20261002_echo_layers3_ncu_mla_01`. Each has full/SourceCounters reports using
  real layer-2 inputs. These are isolated replays with explicitly different cache
  state, not replacements for in-context operator timing.
- Relevant CPU regression: 163 passed. The model extension, instrumentation,
  FLOPs, attribution, hardware identity and lifecycle checks are covered.
- The publisher reuses verified raw measurements; formatting regeneration did
  not rerun or alter GPU results. All four MFU plots use a common x-axis scale.

Publication: [experiment report](../../../experiments/deepseek_v32_echo_prefill/README.md),
with tracked tables and plots in `report/layers3/`. Raw data, SQLite, source
snapshots, saved tensors and independent audit source remain under
`experiments/deepseek_v32_echo_prefill/output/data/20261002_echo_layers3_mfu_01/`.
The audit includes relocated SQLite path mappings with unchanged SHA256.

## Research meaning for Supervisor

The researcher's correction supersedes the previous interpretation that the
reported MFU values already established the implementation's efficiency limits.
MLA's approximately 8% and fused/resident indexer 13.79% / 46.14% remain original
report values awaiting review. Recorded synchronization counts, GPU gaps and NCU
samples are investigation leads; they do not identify the cause of the MFU/cache
problems or justify a fixed optimization order.

ECHO has implemented bounded HBM slots, pinned host backing, exact recall and
query splitting that preserves selection. Those functional capabilities and the
saved numerical comparisons do not establish a reasonable cache policy or fair
performance baseline. The cache currently stages each appended chunk through host
memory, reserves up to 8192 prefetch slots, and uses recursive exact-union capacity
checks. These are concrete behaviors to audit, not newly proven defects.

Supervisor should mark DeepSeek/ECHO baseline validity as unresolved in status
entry 2.5 and schedule review of implementation efficiency, MFU interpretation
and cache policy before comparative performance claims. Corrected implementations
require fresh numerical and performance measurements with new run IDs. The
complete 61-layer post-gather-fix rerun remains separately outstanding; this
three-layer diagnostic cannot satisfy it. Existing evidence does not establish
NOSA-versus-ECHO design superiority or GR serving representativeness.
