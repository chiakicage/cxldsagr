# C10 integration and measurement plan

Parent: accepted C9 formal/profile pair on 2026-10-04. Objective remains the
four-scheme first-visit latency and E2E MFU goal, followed by the official ECHO
comparison. C10 formal execution and independent output/runtime/arithmetic
audits have passed. Its matching profile has not run.

## Selected components and scope

- CPU token predicate: production component accepted. Explicit
  `native_token_validation=True` selects the CPython predicate at runner
  construction; default and setup-fallback `_validate` remain the frozen C9
  implementation. No build/load work enters requests. Motivation formal,
  profile and official formal/repeat explicitly opt in and save runtime
  identity. Ordinary GR/NOSA runners keep the default. Source/CPU evidence:
  [validation checkpoint](deepseek_request_validation_checkpoint.md).
- Main MLA rotary layout: production component accepted.
  The same FlashInfer kernel reads aligned strided Q/K and writes Q directly
  into the final latent-plus-RoPE output tail. No math, indexer rotation,
  checkpoint weight or KV ownership changes. Complete projection and primitive
  tests are under `/tmp/deepseek_rotary_layout_20261004/`; see the
  [layout plan](deepseek_rotary_layout_plan.md). Unsupported layouts preserve
  the existing path and copy. Integrated source must receive its own gate.
- T1 exact top-k finalization: rejected after correct outputs but slower
  complete API times in all six cases. No top-k production changes enter C10.
  T2 finite tie repair after the official value sort also passed GPU correctness
  but regressed three complete API timing cases; it is rejected. Neither
  candidate changes production.

All candidates preserve H65536/A128/chunk1024/P65536/NH16777216, ten independent
checkpoint block copies using source 0–2 inputs, 16 users x 2 rounds, exact
selection/output contracts, FP32 residual/norm arithmetic, and transient
candidate/session/LRU semantics. Production `torch.compile` remains forbidden.

## Integration gate

`/tmp/deepseek_c10_combined_validation_20261004_01/run.py` will derive from the
accepted C9 gate. Add `.cpp` to the exact source closure and record the actual
selected CPU extension path/binary/source/compiler/ABI/header identities.
Inject explicit native opt-in into the selected runner integration cases;
do not change tests of default/fallback selection. Require the actual H64K
four-scheme eager/Graph tests to execute, with no skips, and include focused
production rotary tests. Any later selected top-k candidate needs separate public API/lifetime
coverage and actual Triton source/PTX/CUBIN/runtime identity.

C10 correctness must not inherit C9 results for changed code. Unchanged C9
quantizer/MLP component validation remains valid; the new complete gate checks
its changed surrounding graph and request context. Required checks include
alternate users, A121 fallback/A128 replay, all hidden/logits, independent
weight/cache ownership, packed request buffers, cleanup/failure handling and
allocated/reserved/device-used separation. Root froze source after candidate implementations/tests stabilized.
The final root is `/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0`,
503 files, map SHA `6d9718b6177800112683e721650e6ae99081c71948a75555580006d0e457ffbe`.
Executed-source SHA is
`11fc11b18b2e70baf450a82ab4fad66f2f8d4e22cda2e9f36bf74453a400df8f`.
The earlier 502-file freeze is superseded: it lacked the actual-instance
validation profiler wrapper and its CPU regression. Validation and final measurements
must identify their exact executed source, including local dependencies.

## Formal/profile and publication

Reuse the C9 command/measurement boundaries with new run IDs and frozen root:
`motivation_c10_20261004_u16_r2_01`, matching profile and FLOPs IDs. Keep both
PTXAS paths pinned before imports, eight OMP/MKL threads and one root-assigned
GPU. All other GPUs must be idle during timing. Record discrete device/process
observations spanning each execution; do not call them continuous monitoring.

Require 128 formal payloads, 96 byte-exact offload/HBM comparisons, and 32
cross-version HBM comparisons against C9. Audit complete warmup/session/replay
coverage and all request memory snapshots. Every native-opted case must record
the actually selected CPU backend; a fallback cannot be presented as a native
performance measurement. Source/compiler/library hashes must remain unchanged.

Obtain matching matrix/nonmatrix profiling with unchanged interval definitions.
Wrap the actual runner instance `_validate` once for each captured request.
Constructor-selected native methods bypass later class wrappers; four CPU
regressions now verify the instance wrapper, errors and restoration. New graph node counts/lineage must be measured and audited rather
than copied from C9. API active sums and formal wall time remain separate.

After accepted current-source motivation results, run fresh official H64K
Graph-enabled two-scheme formal measurements and independent HBM repeat under
the existing numerical policy. Publish coordinated motivation/official reports
only after both gates pass. Preserve existing report assets/backing runs until
then; clean only superseded artifacts using the reviewed publication manifest.
Shared model changes also leave other dependent experiment reports tied to their
recorded source versions until their corresponding reruns; no new figures are
claimed for those paths from this validation.

Formal run `motivation_c10_20261004_u16_r2_01` completed in exec session
21984 with exit 0. Discrete GPU monitor session 26372 was stopped after
completion. Verified relocation retained all 1,594 data files and two logs.
Independent audits under the run's `analysis/independent_formal_audit/`
checked all 128 payloads, 96 offload/HBM pairs and 32 C10/C9 HBM pairs,
exact workload and modeled work, native selection, runtime artifacts, lifecycle,
104,960 graph replays and all 256 memory observations. No current-source
matrix API MFU is claimed before a matching profile is accepted.
