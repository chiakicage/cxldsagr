# Q1 official ECHO adapter follow-up

## Contract and baseline

This record covers the accepted promotion optimization on the local H64K/A1
ECHO cold path while using the unchanged pinned official fused score/prefetch
kernel. The current published implementation is
`deepseek_h64k_a1_free_20261008_01`, including the later bounded preparation.
The component investigation below retains its original independent identities. This follow-up
does not change official sources, prediction thresholds, exact top-k, map/tag
encoding, or cache lifecycle. Production edits remain with the parent agent.

The accepted node table is
`experiments/deepseek_v32_echo_official/report/decode_gap/kernel_details.csv`.
For current local ECHO, the official fused kernel takes 10.752/23.680/48.704 us across
L0/L1/L2. Its official natural-residency comparison is not the same workload.
Local adapter validation takes 4.672/4.576/4.576 us. The later bounded free-slot
preparation takes 18.080 us over three layers and contains nine kernels.
These are intrusive profile node sums, not clean invocation latency.

The retained NCU run `q1_promotion_ncu_20261008_02` identifies a one-block,
two-warp validation kernel with 3.0% achieved occupancy and one issued
instruction per 13.5 scheduler cycles. Sparse source sampling records one
short-scoreboard and one wait sample in the serial duplicate-check loop, and
three long-scoreboard samples in dependent map reads / the serial eviction
count. Sampling is too sparse to assign a percentage of runtime to a line.

## Component design

1. Isolate promotion validation first. Expand validation to 256 threads,
   distribute all triangular host/slot uniqueness checks across the block,
   and reduce the eviction count with a block vote. Keep every bounds, tag,
   old-owner, journal, uniqueness and count assertion. The subsequent copy
   kernel and all official code remain byte-identical.
2. Investigate the FIFO preparation separately. A bounded free-slot selection
   is possible only under a proved enough-free-slots condition; `H <= P` alone
   is not a residency proof and must not skip priority/map checks.
3. Preserve the prefill mean update. Omitting it for Q1 would change observable
   hint state and future Q>1 behavior, despite official decode consuming only
   `offset[1]`. No such omission is accepted here.

Risks: increased block size may cost more than the shorter loop; malformed
states must still assert before copy/publication; partial counters must not
read uninitialized shared values. Benefits require clean A/B evidence and
later complete-model remeasurement.

## Executable plan

GPU 3 and CPUs 24-31 are assigned by the parent. Use the existing immutable
native cache loader, current official adapter as private baseline, and a new
private candidate snapshot under
`experiments/deepseek_v32_echo_official/output/data/q1_validation_candidate_20261008_01/source/`.
Write independent check output under `/tmp/cxldsagr-checks/q1-validation/`.

1. Run `python -m experiments.deepseek_v32_echo_official.src.q1_promotion
   --mode check --source-dir <private source> --output-dir <new check>`.
   This checks 84 cases against byte-exact CPU reference, all 22 adapter
   lifecycle checks, and four malformed-input child processes that must
   terminate with a device assertion. Add graph/replay and cross-warp
   duplicate coverage if the candidate passes.
2. Run the same entry with `--mode bench --receipt <check/receipt.json>` into
   a new experiment data directory. Restore all input metadata outside every
   timed call; use six formal slot/occupancy cases, 20 warmups and 100 AB/BA
   pairs each. Include validation and copy in the timed API.
3. If clean timing improves, collect separate NCU full/source evidence of
   both validation versions, with source/native/input identities. Compare
   kernel duration and issue/stall evidence under identical NCU settings.
4. Send the patch and evidence to the parent for integration only after all
   checks pass. Keep published results until full-model acceptance, clean
   timing, profiles and publication succeed.

## Component acceptance

The private candidate passed; production is unchanged at this checkpoint.
The proposed patch is `validation_candidate.patch` in this directory.
Its only changes are the 256-thread validation launch, distributed pair
checks, and block-vote eviction count. All device assertions are retained;
the unchanged second kernel performs record copy and publication only after
validation completes.

`q1_validation.py` reuses the existing promotion harness and adds 72 graph
replays with changed slot/host/record inputs, 24 nondefault-stream calls and
four cross-warp duplicate-slot failures. All 84 base state/byte comparisons,
22 adapter tests, 72 replays, 24 stream calls and eight negative cases passed.
The accepted receipt is
`/tmp/cxldsagr-checks/q1-validation/q1_validation_check_20261008_02/receipt.json`.

The independent clean run is `q1_validation_bench_20261008_02`. Every sample
restores metadata outside the graph event interval. Both validation and copy
are included, with 20 warmups and 100 alternating AB/BA pairs per case on
GPU3 / CPUs 24-31 (H200 SM90, Torch 2.12.1+cu130).

| Slot layout | Occupied selected slots | Baseline median us | Candidate median us | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| consecutive | 0 | 13.088 | 10.720 | 100/100 |
| consecutive | 32 | 13.392 | 11.072 | 100/100 |
| consecutive | 64 | 13.312 | 11.168 | 100/100 |
| random | 0 | 13.632 | 11.200 | 100/100 |
| random | 32 | 13.536 | 11.296 | 100/100 |
| random | 64 | 13.696 | 11.408 | 100/100 |

Separate full/source NCU run `q1_validation_ncu_20261008_01`, with kernel
replay, cache flush and base clocks, measures validation alone at 11.168 us
for baseline and 7.808 us for candidate. Achieved occupancy rises from 2.99%
to 12.29%; scheduler issue activity rises from 0.0742 to 0.1326. The remaining
sparse source samples mainly identify barriers and a shared-load dependency.
The copy kernel is excluded from these NCU durations. This evidence supports
parallelizing the validation work; it does not establish a full-model gain.

NCU reports were read through `ncu_report` and stored with metrics, sampling,
rules and per-source-line stalls under this run's `analysis/` directory.
`component_acceptance.json` binds the check, benchmark, profiles, private
sources and actual DSOs; all six run identities are equal. At component
acceptance, production still matched the private baseline byte for byte. The preliminary
benchmark using the older harness was removed after the expanded independent
check and benchmark passed.

Root applied the candidate and passed the 22 actual adapter tests. The current
complete implementation also includes CUB top-k and the later bounded free-slot
preparation. It was rechecked and remeasured by
`deepseek_h64k_a1_free_20261008_01` and operator profile
`deepseek_h64k_a1_free_mfu_profile_20261008_01`; both profile output sets match
the independent check bitwise. The published ECHO synchronized step is
2.654501 ms and its cold L0-L2 profile window is 1.507332 ms. Those formal
numbers do not isolate promotion's effect. The separate paired preparation
model gate supports its narrow gain; ECHO remains slower than serial sparse
under the current cold condition.

Superseded CUB selected reports and operator-profile artifacts were removed
after this replacement passed publication review. The original CUB stage and
bench remain only because active graph diagnostics bind them. The old operator
profile's signed request remains at its original path. Cleanup evidence is
`experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_free_20261008_01/publication_review/cleanup.json`.
