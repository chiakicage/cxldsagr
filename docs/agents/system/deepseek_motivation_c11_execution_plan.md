# C11 integration plan: exact top-k and certified cleanup audit reuse

Status: deferred on 2026-10-04. The user selected C10 as the final optimization
revision and requested closure. Pending T3 and cleanup V2 production edits were
archived outside experiments and restored to the accepted frozen C10 source.
The prepared C11 combined driver was not run. Component evidence remains
separate from full-model performance; no C11 formal result is claimed.

## Selected changes

- T3 retains official SMALL selection and value sorting, resolves finite
  bit-identical tie runs of length at most four in a sort-free Triton kernel,
  and uses a flagged-row exact general fallback for longer runs. Signed zeros,
  nonfinite masking, fresh storage and unsupported uint32-packing fallback
  preserve the C10 public API. All six prototype complete-API comparisons
  improved; final production identity-observation overhead is being measured
  separately before acceptance.
- Cleanup V2 retains the full first pool audit, exact existing transient no-op
  predicate and synchronization. Explicit constructor selection supplies a
  canonical unbound hook; real bound-method identity is checked before and
  after invocation. Only an exact receipt for the fresh ticket permits reuse.
  Unknown/mutating/default paths retain audit two. The corrected prototype
  passed independent review and paired CPU timing. Production helper extraction,
  all experiment factories and actual profiler behavior receive their own gate.

R1 resident timestamp election remains a separate temporary candidate. No R1
code is selected for C11 from its CPU proof or initial GPU comparisons.

## Component and combined acceptance

Require public production top-k tests and exact all-value-bit/index comparisons
against frozen C10, complete API timings including both launches, flags and
runtime identity observation, and actual live specialization/artifact identity.
The original C5 fallback stays available for unsupported packing.

For cleanup, require production default/selected/native/NOSA tests, frozen-C10
AST/output/all-metric parity, adverse wrapper/storage/lifecycle cases, actual
bound production cleanup timing, and profiler parity on one/two audit calls.
The profile must wrap the runner's actual selected `_cleanup`, leaving guarded
backend methods untouched. Constructor policy metadata is not a per-request
claim that reuse occurred.

The combined GPU driver is prepared at
`/tmp/deepseek_c11_combined_validation_20261004_01/run.py`. It extends the C10
gate with current public selection tests, top-k build/live identity, and explicit
canonical cleanup selection for real DeepSeek runner instances. Default/negative
constructor tests run separately without this injection. Require all collected
tests and phases to pass, the H64K four-scheme eager/graph cases, A121/A128,
packed admission/ownership/failure coverage, and all four schemes' actual selected
runner identities. Do not reuse C10 GPU outcomes for changed paths.

## Full measurement and publication

After component and combined gates, freeze the selected source and execute a
new H65536/A128/chunk1024/P65536/NH16777216, sixteen-user/two-round trajectory
with the same checkpoint copies, precision, warmup, input and complete-request
timing boundary. Root assigns quiet measurement windows and records discrete
process observations. Require 128 full outputs, 96 exact offload/HBM pairs and
32 cross-version C11/C10 HBM pairs; verify lifecycle, all memory snapshots,
compute-island replays and actual native/Triton runtime identities.

Capture matching matrix/nonmatrix profiling and recompute current useful work,
API attribution and aggregate MFU. Report formal and profiled intervals separately.
Keep allocated, reserved and device-used observations distinct. Then rerun the
official formal comparison and independent resident repeat with unchanged
numerical policy. Coordinate public replacement only after both experiments'
final evidence is accepted; retain current valid reports and backing outputs
until then. C10 official correctness refresh is recorded separately in
[its validation note](deepseek_echo_official_c10_validation.md).
