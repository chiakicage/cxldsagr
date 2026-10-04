# Cleanup audit V2 checkpoint

Status: deferred at the user's requested C10 closure, 2026-10-04. C11 cleanup
is not promoted. Core code is restored to accepted frozen C10; the implementation
and new tests are retained with hashes under
`/tmp/deepseek_cleanup_production_20261004/deferred_source/`. The selection and
measurements below are historical engineering evidence, not current production
behavior or C10 performance evidence.

2026-10-04. Root selected the corrected V2 prototype for production
integration. The integrated code remains subject to its own CPU validation,
timing and root-owned GPU/formal gates. No V1 performance number is used as
V2 evidence. The integration contract is in
[the production plan](deepseek_cleanup_audit_integration_plan.md).

V2 requires an explicit canonical unbound cleanup function. Caller checks
require a real Python bound method with that function and backend self before
and after invocation. Backend checks also cover the outer hook and all inner
cleanup/validation methods. Unknown hooks run normally but always take audit
two. Profiling wraps the runner's complete cleanup helper, leaving guarded
backend methods unchanged. This avoids a trusted-wrapper registry.

The final prototype passes 48 focused tests and 393 unchanged CPU tests
(one optional NOSA checkpoint test skipped because its path was unset, five
CUDA tests deselected). The focused tests preserve full outputs and all
metric fields and cover forwarded/fabricated current receipts, hook
restoration/replacement, callable attribute spoofing, shared/inactive-session
storage growth, and runner-level profiling. CUDA remained uninitialized.
Results are `/tmp/deepseek_cleanup_audit_v2_20261004/parity_run.json` and
`regression.json`, with separate logs and before/after source hashes.

Independent review reproduced the V1 outer-hook hole and an intermediate V2
callable-metadata hole. The final exact MethodType guard makes the latter
reproduction perform two audits and report the storage overrun as required:
`/tmp/deepseek_cleanup_v2_callable_identity_fixed.json`. Failing engineering
reproductions remain outside experiments and are not acceptance results.

## Final V2 CPU timing

Both processes used the same actual-storage protocol described in the
[V1 geometry section](deepseek_cleanup_audit_candidate_checkpoint.md):
H=65,536, ten layers, one/sixteen sessions, fixed/byte-budget modes, full
production storage accounting, 88 scaled shared tensors, a mocked CUDA
barrier, ten warmups, and 41 randomized paired blocks of ten calls. Both
processes completed all sixteen cases with identical before/after source
hashes, matching storage results and correct audit counts. There are 1,312
retained paired blocks and 26,240 measured complete cleanup calls.

| Users | Mode | Run 1 baseline → candidate / paired saving (µs) | Run 2 baseline → candidate / paired saving (µs) |
|---:|---|---|---|
| 1 | Fixed | 256.752 → 133.630 / 122.961 | 281.446 → 144.678 / 136.892 |
| 1 | Bytes | 260.403 → 134.607 / 125.834 | 284.199 → 147.283 / 136.789 |
| 16 | Fixed | 1612.407 → 813.169 / 800.478 | 1762.671 → 888.417 / 873.768 |
| 16 | Bytes | 1629.319 → 821.864 / 807.659 | 1638.997 → 826.058 / 811.020 |

The certified sixteen-session paired speedups are 1.982–1.984. Absolute CPU
baselines vary across cases and processes; the paired saving is consistent.
All absent-hook, unknown-inner-method and default-helper cases still execute
two audits. Their paired changes are below 0.8% in magnitude and include both
signs; they do not establish zero overhead. All per-case medians, paired
deltas and speedups are in `benchmark_summary.tsv` in the V2 temporary
directory. Raw samples are `benchmark_h64k_01.json` and
`benchmark_h64k_02.json`, each with separate logs. Peak RSS is 542,224 and
542,212 KiB; actual storage capacity is not physical/pinned DRAM usage.

These are CPU cleanup-component results, not serving/MFU results. No model
or checkpoint executes, and the host arena is scaled. The timing window has
been returned to root. Final measured source bytes are retained under the
V2 `source_snapshot/`, with `source_snapshot_manifest.json`.

| Artifact | SHA256 |
|---|---|
| V2 `candidate.py` | `1e0e4f8b4241cc3207e7eb01b6c9e09b67408d4a16f3c166c30a7a9404400a1d` |
| `fixtures.py` | `6221df7bf87d4178108ae3ebc8f0703dfa3661eb89631006d62fe4cd9ca9062c` |
| `test_candidate.py` | `133245f5a221168a9bb45aa9c44277fdcd9d1f61837de49cf43b97b0370d75b4` |
| `benchmark.py` | `09e78212459e964ff00c6b8db1da6e9910e692425455d04173da9d297885347a` |
| `parity_run.json` | `68d6ae4ea7bb5e18023da3cd3226803e59a216f4c1b3cee92d62ff83cb828d5d` |
| `regression.json` | `7c27fe3bed83e430497d9af2e0eba28b841e77462c1477f87a3cf00014c81796` |
| `benchmark_h64k_01.json` | `1ba4371fbb10c3978ae25580843aae6e3c8dab1da193834e457b7e953376c272` |
| `benchmark_h64k_02.json` | `d056a09e4a805b58524449bb90032461626a1d59390dc59596ba746aaad5356c` |
