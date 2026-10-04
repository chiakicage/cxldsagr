# Independent cleanup-audit candidate review

Status at C10 closeout on 2026-10-04: cleanup audit reuse is deferred. The
user ended this optimization round at C10. Production core and experiment
wiring were restored to the accepted C10 frozen source, and the additional
CPU tests were archived outside the repository. The experiment restoration
manifest is
`/tmp/deepseek-c11-experiment-cleanup-deferred-20261004-jvhlon6o/restoration_manifest.json`.
The reviews and measurements below describe the deferred candidate, not the
current implementation or published serving performance. No C11 formal or
matching profile result was published.

The measured v1 prototype is not ready for production integration. Independent
CPU reproduction found that an unknown outer cleanup hook can invalidate a
genuine current receipt after the trusted backend operation returns. The
runner nevertheless reuses its earlier footprint instead of performing the
required second audit. This violates the conservative fallback contract in
[the candidate plan](deepseek_cleanup_audit_candidate_plan.md).

## Reproduced v1 correctness gap

Measured source:
`/tmp/deepseek_cleanup_audit_20261004/candidate.py`, SHA-256
`966071d754063f35ecdab51e59b99f9e0b250dd632ae68a3a673cd2b7d410993`.

The independent reproduction uses the existing real-CPU-storage fixtures and
the production pool audit. An outer hook calls the genuine backend hook,
resizes either shared storage or an inactive session's storage by 64 bytes,
then returns the unchanged authentic receipt. In both cases, selected cleanup
performs one audit and returns the same footprint object for before and after.
An explicit second audit raises the expected reservation-overrun exception.
No forged receipt, stale ticket or changed inner method is needed.

Reproduction: `/tmp/deepseek_cleanup_audit_outer_hook_repro.py`; result:
`/tmp/deepseek_cleanup_audit_outer_hook_repro.json`. CUDA remained uninitialized;
the CUDA barrier was replaced with a no-op. This was a correctness check, not
a timing run. The candidate owner confirmed the gap and retained the measured
v1 source without modification.

## Independent benchmark arithmetic

Both `benchmark_h64k_01.json` and `benchmark_h64k_02.json` passed independent
saved-data checks. The audit verified before/after/current source hashes,
the complete 32-case matrix, seeded case and paired execution order, all
1,312 paired blocks, per-call timing arithmetic, quantiles and paired medians.
All 2,624 timing blocks remain included. Recorded storage geometry and audit
counts agree with the real CPU tensor fixtures.

For 16 sessions, the certified v1 segment's paired median speedup is
1.981–1.985x, with paired median savings of 810–831 microseconds across the
two modes and two trials. This is the complete CPU
audit/truncate/mocked-barrier/audit segment. Shared buffers are small CPU
fixtures, and indexer allocations are not fully touched. The result is not
GPU or full-serving performance and cannot be subtracted from formal request
latency. Correct arithmetic does not remove the correctness blocker.

Audit source: `/tmp/deepseek_cleanup_benchmark_independent_audit.py`; result:
`/tmp/deepseek_cleanup_review_20261004/benchmark_independent_audit.json`.

## Required v2 contract

The caller accepting the receipt must check the actual bound backend hook
against the selected canonical function before invoking it and again after
it returns. Both checks and the exact fresh receipt are necessary for reuse.
Unknown hooks still execute normally and always cause audit two, even when
they forward a genuine receipt, fabricate a current receipt, or restore the
canonical hook before returning. Checking identity only inside the genuine
backend hook is insufficient because an unknown hook can bypass that body.

Profile instrumentation should wrap the runner's cleanup entrypoint and leave
guarded backend operations unchanged. It needs no exception for arbitrary
outer wrappers or `__wrapped__` metadata. The required first full audit,
fallback behavior, synchronization, exception handling and owner retention
remain unchanged. V2 must receive fresh correctness and complete-segment
timing evidence before any promotion; v1 timings do not validate its added
caller checks. Production and affected GPU/serving acceptance remain separate.

## Corrected v2 guard review

The first v2 revision added caller checks but matched method identity by
reading `__self__` and `__func__` from arbitrary callables. An independent
fixture copied those attributes onto an unknown function wrapper, which then
forwarded an authentic receipt after enlarging shared storage. The second
audit was again skipped. The failing result is
`/tmp/deepseek_cleanup_v2_callable_identity_repro.json`, tied to candidate SHA
`f09f1df1d45edc2e305a6f7008ac2cca3dd37b6079d6e06600dc71b5fe85c0b6`.

The corrected v2 requires exact `types.MethodType` in both caller and backend
guards before comparing bound self/function. Independent reproduction against
SHA `1e0e4f8b4241cc3207e7eb01b6c9e09b67408d4a16f3c166c30a7a9404400a1d`
now performs audit two and raises the shared reservation overrun. Evidence is
`/tmp/deepseek_cleanup_v2_callable_identity_fixed.json`. CUDA remained
uninitialized. The candidate owner separately reported 48 focused passes and
393 passes in unchanged CPU suites; the latter retain one optional NOSA
checkpoint skip and five deselected CUDA cases.

No further code-review blocker was found within the serial execution and
trusted-constructor contract. The configured canonical function is trusted
code; this is not protection against arbitrary mutation of Python module
objects. The [production integration plan](deepseek_cleanup_audit_integration_plan.md)
keeps explicit constructor selection, ordinary default behavior, and separate
runner-level profiling. The official DeepSeek backend inherits the guarded
base methods, so its planned canonical wiring is consistent. Fresh v2 timing
and integrated-source acceptance remain required.

## Corrected v2 timing acceptance

The two corrected-v2 benchmarks subsequently passed the independent source,
fixture geometry, seeded ordering and arithmetic audit: 32 cases, 1,312 paired
blocks and all 2,624 timing blocks retained. For fixed pools with 16 sessions,
paired median savings were 800.478 and 873.768 microseconds, with paired
speedups of 1.9833 and 1.9844. These measurements include the final exact
`MethodType` guards. Audit source:
`/tmp/deepseek_cleanup_v2_benchmark_independent_audit.py`; accepted result:
`/tmp/deepseek_cleanup_v2_review_20261004/benchmark_independent_audit.json`.
The result remains a CPU tensor-storage segment with a mocked CUDA barrier;
it does not establish a full-serving speedup. Root authorized production
integration after this corrected-v2 review.

## Production experiment and profiler wiring

The motivation measurement/profile and official-comparison measurement now
select the canonical hook and record `cleanup_audit` identity beside native
token-validation identity. This includes profiling warmup and graph setup,
and the official comparison's independent HBM repeat. The source snapshot
selection already includes these execution files and all three core modules.

For selected runners, the actual `InstrumentServing` context wraps
`runner._cleanup` as `request_cleanup`, retaining real pool-audit scopes and
leaving guarded backend methods intact. Default runners retain their existing
truncate/synchronize wrappers. Eight new CPU regression cases cover one-audit
reuse, two-audit default and mutating fallback, measurement of every retained
session, real shared-storage overrun, synchronization failure attribution and
restoration of all wrapped methods. The four prior native-validator profiling
cases remain covered.

Validation command:
`.venv/bin/python -m pytest --import-mode=importlib -q experiments/deepseek_v32_motivation/tests experiments/deepseek_v32_echo_official/tests`.
All 182 cases passed in 3.61 seconds; output was terminal-only. Ruff lint and
format checks passed for the four touched source/test files. The profiler
fixtures assert that CUDA remains uninitialized. This gate validates actual
profiling behavior on real CPU storage; combined GPU numerical/lifetime
acceptance and fresh formal/profile publication remain separate root-owned
gates. Existing public reports and backing runs were preserved.
