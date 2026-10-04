# C10 frozen-source preflight and independent audit

The final CPU preflight passed on 2026-10-04. It used
`/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0`, with 503 frozen
files and freeze-map SHA-256
`6d9718b6177800112683e721650e6ae99081c71948a75555580006d0e457ffbe`.
The experiment's own snapshot collector retained 1,270 source files with
executed-source SHA-256
`11fc11b18b2e70baf450a82ab4fad66f2f8d4e22cda2e9f36bf74453a400df8f`.
Evidence is `/tmp/deepseek_c10_v2_preflight_20261004/precheck.json`, with its
fresh snapshot in the adjacent `executed_source/` directory. CUDA remained
uninitialized throughout this preflight.

## Profile correction and regression

The first freeze exposed a profile attribution defect: native runner setup
binds the selected validator on the instance, so later wrappers on class
methods did not observe its calls. A CPU reproduction recorded no native
validation scope, while the default validator produced one.

The final freeze wraps the actual runner instance's `_validate` once. Relative
to the first freeze, only `src/profile.py` changed and
`tests/test_profile_validation.py` was added under the motivation experiment;
no files were removed. The four new cases construct real runners and cover
default/native selection, success/error attribution, exactly one validation
scope, and restoration of the original methods. All four passed from the
final frozen tree. Together with the three existing graph instrumentation
cases, seven tests passed in the working tree; Ruff reported no issues.
Both frozen `measure --help` and `profile --help` checks exited successfully.

The preflight constructed an actual native CPU runner and confirmed its
selected callable. The helper's `.cpp` source and Python loader appear in the
experiment snapshot. Native binary SHA-256 was
`308a507e16c6871010f09d37cef7d2012f81cf25328f73103b59bf4a042e0867`.
The build identity includes the declared PATH, so a different launch PATH can
produce a different cache key even when the resolved compiler, dependency
bytes and binary bytes are identical. Formal acceptance checks these bytes
and each identity's own fingerprint; it does not require equal cache keys.

## Independent combined-gate audit

The independent audit accepted all 221 prepared tests and all 663 pytest
phase outcomes, with no skips and an exact match to the prepared node set.
Both required checkpoint graph and packed-input lifecycle cases ran. The
before/after gate source maps were identical; all 124 files shared with the
measurement freeze matched. Six additional gate files comprised its driver
and five integration tests omitted from the measurement freeze.

Native selection counts were HBM 8, ECHO 10, serial sparse 1 and dense
prefetch 1. All 309 native source/header/compiler closure files were checked.
Twenty-four live Triton quantizer specializations had loaded handles and
matching retained/artifact CUBIN hashes. The three bounded top-k payloads
matched their recorded hashes without changing original output hashes.
Evidence is `/tmp/deepseek_c10_preflight_20261004/combined_independent_audit.json`;
the gate details remain in [the combined validation record](deepseek_motivation_c10_validation.md).

## Formal audit boundary

The CPU preflight and combined correctness gate do not establish C10 formal
latency, complete request equivalence to frozen C9, or measured profile node
counts and attribution. Those require separate acceptance. The completed
formal audit is recorded below; matching profile attribution remains pending.
Existing published reports remain attached to their original run IDs until
the coordinated replacement is accepted.

The independent formal audit is prepared at
`/tmp/deepseek_motivation_c10_independent_formal_audit.py`. It imports no
production checking or reporting code. Its required checks cover all 128
saved payloads, 96 offload/HBM byte comparisons, 32 C10/C9 HBM byte comparisons,
identical input/configuration/checkpoint identities, LRU admission, warmup,
transient candidate writes, stage conservation, replay counters, and distinct
allocated/reserved/device memory observations. It also checks native selection
in every scheme and verifies actual native source/compiler/header/binary
identities against the final preflight. The expected observed graph-private
reservation is 5,603,590,144 bytes per scheme; this is a run-specific
observation, not a new fixed capacity deduction. Profile attribution remains
pending until its own trace audit passes.

## Completed formal and arithmetic audit

`motivation_c10_20261004_u16_r2_01` completed successfully. Its data and logs
were relocated with identical file hashes by the parent task. Independent
CPU audit results are retained in
`experiments/deepseek_v32_motivation/output/data/motivation_c10_20261004_u16_r2_01/analysis/independent_formal_audit/`.
The reference is `motivation_c9_triton_20261004_u16_r2_01`.

- `audit.json`: all 128 payloads passed; all 96 offload/HBM and 32 C10/C9
  HBM comparisons are byte-exact. Identical workload/configuration/checkpoint
  inventory, LRU admission, 12 warmup records, candidate discard, stage sums,
  104,960 graph replays and both memory snapshots for every request passed.
  All four native case identities passed the dependency and binary checks.
- `runtime_graph_audit.json`: eight observed Triton quantizer specializations
  passed loaded-handle, PTX and CUBIN checks; 202 dependency/artifact files were
  rehashed. FlashInfer rope, SiLU and top-k libraries were loaded and match C9
  binary bytes. Graph plans, allocation limits and all 256 request memory
  observations passed. The observed private reservation is 5,603,590,144 bytes
  in every graph state. Captured CuTe MLIR hashes remain metadata because the
  corresponding in-memory bytecode is not available after process exit.
- `report_arithmetic_crosscheck.json`: all 10,928 call-ledger formulas, 120
  operator totals, 384 request MFU rows, eight report groups and 24 stage
  groups agree. Source and input hashes match. Modeled matrix work and
  precision-specific peaks are identical to C9.
- `gpu_observation_audit.json`: 127 discrete all-device observations span the
  run, including empty first/final observations. All 81 process observations
  identify only PID 2069546 on the expected GPU. Every query succeeded. Four
  samples after recorded completion retain this PID but have unavailable
  command text; all available commands match the formal run, and no command
  is missing during its recorded execution interval. The maximum sampling
  gap is 27.541 s. This is not continuous monitoring. Original observations,
  monitor source, launch identity and hardware inventory were copied into the
  run directory with matching hashes and a separate boundary record.

The formal audit source is
`27597d0505aef785f69431c4000586e148116d05ff4ca0797bdec88e1d6fbeb2`;
the runtime audit source is
`f85f5f5f0e47bdce0f93f9640272cc181ad9edc7314cdd66a301ddf00980df52`.
Another agent reviewed the native, memory and graph additions without finding
a blocker; that review did not execute an additional test or measurement.

The following are whole-request means over all 16 requests in each group.
Deltas are C10 minus C9. E2E MFU uses useful matrix work and the recorded
precision-specific reference peaks; it is not GPU busy time or matrix API MFU.
HBM misses on every revisit in this workload; all three offload schemes hit
their retained histories on revisit. These are separate fixed-workload runs,
so the deltas do not isolate either selected component's contribution.

| Scheme | Visit | C10 E2E ms | Delta ms | C10 E2E MFU % | Delta pp |
| --- | --- | ---: | ---: | ---: | ---: |
| HBM | First | 2176.886 | -26.973 | 44.440 | +0.544 |
| HBM | Revisit | 2170.244 | -24.117 | 44.576 | +0.490 |
| ECHO | First | 2283.433 | -22.575 | 42.366 | +0.415 |
| ECHO | Revisit | 24.947 | -1.926 | 9.005 | +0.645 |
| Serial sparse | First | 2217.737 | -28.223 | 43.621 | +0.548 |
| Serial sparse | Revisit | 18.456 | -2.098 | 12.172 | +1.243 |
| Dense prefetch | First | 2233.838 | -16.267 | 43.307 | +0.313 |
| Dense prefetch | Revisit | 31.889 | -2.133 | 7.044 | +0.442 |

The exact values are in `c10_c9_e2e_comparison.csv` in the audit directory.
The FLOPs analysis is `motivation_c10_flops_20261004_01`. Formal results do not
complete the pending profile acceptance or authorize replacing published
report assets before the coordinated publication gate.

## Completed C10 profile acceptance and publication staging

The matching `motivation_c10_profile_20261004_01` analysis is accepted. The
production pipeline and seven independent audits completed with exit code 0;
all 11 analysis stderr files are empty. The receipt is
`experiments/deepseek_v32_motivation/output/data/motivation_c10_profile_20261004_01/analysis/independent_profile_audit/completion_receipt.json`,
SHA-256 `cc30286b462b21f1ac542155e871cd2f3843131e6a324914b67d9c89b313a829`.
It verifies 80 byte-exact outputs, 45,933 matrix calls and primary kernels,
267,562 GPU activities, 195 operator groups, 21 aggregate groups, 369 kernel
inventory rows, 6,560 graph replays, 180,320 graph GPU activities and 4,360 clone
edges. All eight native request-validation scopes were verified. These are
profile attribution and saved-artifact checks, not additional formal samples.

The separate profile monitor audit passed with 162 query samples containing
158 process entries, all identifying PID 2119270 on the designated GPU.
The first and final samples are empty and recorded execution is covered.
The maximum gap is 28.836934328079224 seconds. Four unavailable command
strings occurred after recorded completion. Monitor/profile exit codes are
zero. Original monitor stdout/stderr was returned through the exec tool,
not separate redirected files. The source, observations and completion/window
records were copied into the accepted profile run. The audit SHA-256 is
`fd5b33ad41aed9eeec69405638eceacc5afd1a8148436f8449866477f1307bf3`.
Discrete samples do not prove continuous exclusivity.

Canonical motivation publication files are staged under
`/tmp/deepseek_motivation_c10_publication_20261004/` as `README.md`, `report/`
and `manifest.json`. Root owns coordinated publication and cleanup. The
manifest includes accepted source hashes, report-only operational corrections,
frozen latency-plot generator identity, the current corrected operator-report
generator identity, and exact selected-file hashes. The four selected profile
figures are HBM cold and the three offload revisits, all visually reviewed by
root. No new GPU execution was used to build this report.

The operational changes are limited to the retained C10 profile reference
and correct operator call-ledger navigation. Existing motivation tests passed
67/67; Ruff lint/format and CLI help checks passed. These checks do not replace
the formal/profile numerical and performance acceptance above. C11/R1 remains
deferred at the user's C10 closeout; no C11 performance claim is included.
