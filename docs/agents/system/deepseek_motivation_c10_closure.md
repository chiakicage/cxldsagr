# C10 closure and coordinated publication

Status: complete. C10 is the final implementation selected by the user. Both
reports were published after independent acceptance, and scoped superseded
outputs were removed. No further optimization is part of this closure.

The user's 2026-10-04 instruction, “C10 优化差不多了，收尾。”, ends candidate
development for this round. Finalize C10 and report its measured remaining MFU
gap. Do not promote C11 cleanup/T3 or R1 as part of closure.

## Accepted implementation and existing evidence

Frozen root: `/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0`.
The 503-file map SHA-256 is
`6d9718b6177800112683e721650e6ae99081c71948a75555580006d0e457ffbe`.
Formal `motivation_c10_20261004_u16_r2_01` and FLOPs
`motivation_c10_flops_20261004_01` have passed independent audits. Current
official C10 checkpoint validation passed all three GPU cases; it does not
replace the required new serving trajectory.

Pending C11 files are archived under
`/tmp/deepseek_topk_order_t3_integration_20261004/deferred_source/`,
`/tmp/deepseek_cleanup_production_20261004/deferred_source/`, and
`/tmp/deepseek-c11-experiment-cleanup-deferred-20261004-jvhlon6o/`.
Each owner verifies byte-exact restoration against frozen C10. This restoration
does not revert unrelated workspace changes.

## Remaining execution

1. Complete matching profile `motivation_c10_profile_20261004_01`, with twelve
   setup/request captures and eighty complete payloads compared to accepted
   C10 formal HBM outputs. Root runs GPU0 exclusively; both PTXAS paths are
   pinned before imports, OMP/MKL threads are eight, and other GPU/timing work
   is held. Discrete process observations are under
   `/tmp/motivation_c10_profile_20261004_01_monitor/`.
2. Execute `echo_official_c10_20261004_u16_r2_01` from the same frozen source,
   with common compute graphs, original official selector/dispatch, two
   schemes each sixteen users/two rounds, and its independent full HBM repeat.
   Keep the existing numerical policy and unchanged workload/capacity.
3. Run matching analysis and independent numerical, source/runtime,
   graph-attribution, arithmetic, lifecycle and memory audits. Report matrix
   API MFU and formal E2E MFU using their respective denominators.
4. Stage both report trees and READMEs from accepted data, compare current
   local/official results with each experiment's own HBM control, inspect
   figures and verify source/input hashes and links.
5. Publish both replacements together. Then remove only superseded affected
   report/output artifacts, retaining the final raw data, logs, profiles and
   source snapshots. Historical IDs in internal notes are evidence history,
   not current experimental results.

No further kernel, model or serving implementation change is planned. Report
generation corrections, if needed, must preserve measured data and provenance.

## Execution checkpoint

Matching profile exited successfully in session 69491. Its production checks
accepted all 80 complete payloads and twelve captures. Executed profile source
SHA-256 is `c7c2f40915ce41897bd9eabd1dde1efb0dfcfcc528afaea039ef9ac03e57623c`;
the distinct formal source SHA is retained as its reference. Relocation verified
all 1,559 data files, 28 log files and twelve raw profiles byte-for-byte.
Independent source/runtime, numerical and attribution audits all passed:
45,933 matrix calls, 267,562 GPU activities, 6,560 graph replays, 180,320 graph
activities and 4,360 clone edges. See the
[C10 profile note](deepseek_motivation_c10_profile.md).
Monitor session 11557 exited successfully with no terminal output; per-sample
nvidia-smi return codes and stderr are in its original JSONL, rather than a
separate monitor stderr file.

Official `echo_official_c10_20261004_u16_r2_01` and monitor 33855 exited 0.
The GPU0 run used the same frozen tree with `--compute-graphs`; executed source
SHA is `fef9fef45d923343967198552709f7a7549da9145ce0927b535539d86f8cc60d`.
Relocation verified all 2,462 data files and two logs. Independent GR audit
accepted all 96 outputs, original numerical thresholds, source/runtime identity,
request lifecycle and graph totals of 41,600 HBM, 21,120 ECHO and 41,600
independent HBM-repeat replays, with zero fallback. The monitor source and
discrete observations are retained with the run. Post-run hardware inspection
records nvidia-smi M403 for the same UUID reported as H200 by PyTorch.

Separate workspace-only report/entrypoint corrections remove stale comparison
prose, set the default profile reference to C10, and render the real nested
operator-call ledger path. They do not change frozen inference or measurement;
final publication must record their generator hashes and focused test results.

## Publication result

The joint publication helper exited 0. Motivation now contains fifty selected
report files, official nine, with both READMEs updated. Fifty-two relative
Markdown links and all copied file hashes passed. Thirty-nine superseded
output directories were removed only after the replacement reports passed
their final checks. The three final motivation data runs and the final
official run, including original logs, profiles, source snapshots, numerical
payloads and independent audits, remain intact.

The retained publication receipt is
[deepseek_motivation_c10_publication.json](deepseek_motivation_c10_publication.json).
Its executable helper and staged manifests are also retained in the final
formal run's `analysis/c10_publication/`. The workspace's 331 checked code and
script files match the frozen C10 execution except four documented
report/default-reference/test corrections, recorded in
[the equivalence audit](deepseek_motivation_c10_execution_equivalence.json).
The focused motivation report/CLI suite passed 67 tests; official comparison
and report tests passed 78. These checks supplement the prior C10 complete
GPU validation and new full formal/profile/official acceptance.

Final first-visit E2E MFU is 42.37%–44.44%, while sampled matrix API MFU is
53.38%–56.18%. This remaining difference is reported explicitly. Completion
means the user's selected C10 scope is finalized; it does not claim that the
earlier aspiration of matching matrix API MFU was reached.
