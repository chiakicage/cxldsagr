# C11 production cleanup audit checkpoint

Status: deferred at the user's requested C10 closure, 2026-10-04. C11 cleanup
is not promoted. `executor/serving_backend.py`, the DeepSeek serving backend,
and `serving/persistent.py` are restored byte-for-byte from
`/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0`. The C11 versions and
the two new cleanup test/fixture files were hash-verified in
`/tmp/deepseek_cleanup_production_20261004/deferred_source/` before restoration
and removal. `core_manifest.json` records archived and restored identities.
The experiment wiring was separately archived/restored by its owner.
The measurements and integration description below concern the archived C11
attempt only; they are not accepted C10 results or the current implementation.

2026-10-04. The production cleanup path is implemented and passes its core
CPU gate and two independent CPU timing runs. Motivation, official comparison
and selected profiling are wired and pass their experiment CPU suites. GPU correctness
and formal performance remain root-owned and have not run for this change.
Old public run IDs, report data and figures remain intact; both affected
READMEs mark the cleanup change as awaiting GPU/full-trace validation.

The implementation follows
[the integration plan](deepseek_cleanup_audit_integration_plan.md), with
selection supported by [separate V2 evidence](deepseek_cleanup_audit_v2_checkpoint.md).
No V1 or temporary V2 timing is presented as production performance here.

## Implemented contract

`executor/serving_backend.py` provides the typed per-call ticket/receipt and
exact Python bound-method identity matcher. The DeepSeek backend preserves
public truncate's default return and existing predicate/mutating path. Only
the existing successful-transient no-op can issue a receipt, and its optional
cleanup hook exposes that receipt after the original barrier succeeds and
all canonical methods/owner checks remain valid.

`PersistentGRRunner(..., cleanup_audit_hook=None)` keeps two complete audits
by default. Explicit selection takes an unbound Python function and binds the
selected cleanup helper once; invalid option types fail before ownership or
resource allocation. One `execute` body is retained. The first audit is always
the complete unchanged storage/quota audit. Reuse requires matching canonical
bound self/function before and after the call, an exact receipt type, and the
fresh ticket object. Unknown hooks, copied callable metadata, mutations and
uncertified returns always retain audit two. Cleanup failures still discard
the current session, and failed releases retain owner/session state.

All original metric fields and both native/default validation methods remain
unchanged. `cleanup_audit_identity` describes constructor selection and the
canonical function; it does not claim that every request reused an audit.
Default NOSA never selects the optional hook.

## CPU correctness

`/tmp/deepseek_cleanup_production_20261004/regression.json` records 480 passing
tests, one optional NOSA checkpoint skip, and five CUDA deselections, with
1,443 phase reports and 122 unchanged before/after source identities. CUDA
remained uninitialized. Separate stdout/stderr logs are retained.

The gate includes 57 production cleanup tests, 25 native-token-loader tests,
the earlier 393 applicable ownership/resource/storage/DeepSeek/NOSA tests,
and five frozen-C10 checks. The frozen checks restore the new `execute` AST
to C10 by replacing only the helper call with its original four statements;
they restore `truncate` by removing only the private ticket argument and its
optional no-op receipt. Both restored ASTs match the saved C10 source.
Default and native validation ASTs also match. Four fixed/budget × native
selection cases compare complete outputs and every request metric against
the frozen C10 execute/truncate across first visits, revisits, eviction and
candidate-size changes. This is CPU fixture parity, not full-checkpoint
or GPU acceptance.

The production tests cover default selection, ordinary truncate returning
`None`, invalid constructor options, forged/current/stale receipts,
unknown hooks restoring/replacing themselves, copied method attributes,
storage growth behind shared and inactive-session tensors, every first-audit
quota, cleanup failures and failed-release ownership.

The complete motivation and official experiment CPU suites passed separately:
182 tests in 3.61 seconds, including eight actual `InstrumentServing` cleanup
regressions. They cover selected/default one/two audit counts, mutating
fallback, traversal of all sessions, real shared-storage growth, synchronization
failure attribution and scope/method restoration. The profiler's matrix
instrumentation is replaced by a no-op in these CPU fixtures; the actual
serving wrapper and audit scopes execute. Command:
`.venv/bin/python -m pytest --import-mode=importlib -q experiments/deepseek_v32_motivation/tests experiments/deepseek_v32_echo_official/tests`.
This result is terminal-only, not a persisted experimental artifact. Ruff
and scoped diff checks pass for the changed code. An unrestricted diff check
also encountered pre-existing trailing whitespace in unrelated report SVGs;
those assets were not edited.

All selected motivation/official runner factories retain native token
validation and add the explicit canonical cleanup hook. Official resident
repeat, profile warmup and graph setup are included. Metadata records cleanup
selection beside token validation. Selected profiling wraps `runner._cleanup`
and retains actual pool-audit scopes; guarded backend methods are untouched.
Default profiling retains its previous wrappers. Source snapshot selection
covers the modified execution files and core modules.

## Actual production cleanup timing

The baseline uses frozen C10 truncate and its original inline cleanup body.
The candidate calls the real production `runner._cleanup`, including the
extracted bound helper call, receipt construction, all caller/backend guards
and the first complete audit. This is different from timing the earlier
prototype function. H=65,536, ten layers, one/sixteen actual CPU-storage
sessions, fixed/byte-budget modes and all fixture/storage boundaries match
the previous protocol: 86,521,648 bytes per session, 5,632 scaled shared bytes,
no filled NH arena, no checkpoint, no host transfer, and a mocked CUDA
barrier. Allocated tensor storage is not pinned or process resident DRAM.

Two independent processes used seeds 20261004/20261005, ten warmups, and
41 randomized paired blocks of ten calls. All 32 cases, 1,312 paired blocks
and 26,240 complete measured calls passed storage parity, audit counts and
source checks. Peak RSS was 528,392/526,860 KiB. No samples were removed.

| Users | Mode | Run 1 baseline → production / paired saving (µs) | Run 2 baseline → production / paired saving (µs) |
|---:|---|---|---|
| 1 | Fixed | 256.633 → 134.898 / 122.104 | 255.635 → 134.277 / 121.196 |
| 1 | Bytes | 261.960 → 137.339 / 124.753 | 260.002 → 135.574 / 124.781 |
| 16 | Fixed | 1622.418 → 819.370 / 803.139 | 1620.672 → 820.691 / 798.642 |
| 16 | Bytes | 1643.708 → 828.293 / 814.569 | 1659.675 → 838.319 / 821.903 |

Certified sixteen-user cleanup saves 0.799–0.822 ms in these CPU fixtures,
with paired speedups of 1.972–1.984. One-user paired speedups are 1.903–1.921.
Default helper extraction's largest paired overhead was 1.230 µs; some
cases favored production slightly. The explicitly selected unknown-inner
fallback added up to 5.373 µs at one user (about 2.1%) and 8.513 µs at sixteen
(about 0.52%). An absent-hook selected fallback added at most 7.960 µs. These
paths all kept audit two. They are not zero-cost paths, and these component
figures cannot be subtracted from or extrapolated to full serving latency.

All sixteen paths per process, including default extraction, absent hooks
and unknown inner methods, are in `benchmark_summary.tsv`. The raw JSONs,
separate logs, verification record and measured source snapshot are under
`/tmp/deepseek_cleanup_production_20261004/`. `driver_prepare.json` checks all
sixteen production combinations at H=128 without timing. The timing window
has been returned to root.

## Source and artifact identities

| File | SHA256 |
|---|---|
| `executor/serving_backend.py` | `4f5baca2b1f06963e613a827546b0850168370346797145d70de72bfcdbf6e0e` |
| `models/deepseek_v32/serving_backend.py` | `aa97911c687d98ba0b04f4f5d7749e98025d080feb3a585395218656c3c0da03` |
| `serving/persistent.py` | `c5cffaa70434eac9a09af3207138b9b768609ba6efe85caa15148f331624f734` |
| Motivation `src/measure.py` | `c8d701875b463be7114da67e362c126dcfa3e689568ef6af3dce03f6277feaa2` |
| Motivation `src/profile.py` | `77cc7d2d2f2a1f1cad37ff80a0d7ac84db94f044545be8e60b7e17bb96cce003` |
| Official `src/measure.py` | `3a9c0b26d3ec0cfed2568dfbbfa7a58f4460357951bb3f65bb64ca2399e04daf` |
| Motivation `tests/test_profile_cleanup.py` | `64083eb05b1bb3c0c05d85f5118283135beb56779f3cbeabd5407edbcbd0b009` |
| `regression.json` | `cb915559c9bc01bee75b1fe401f2e079b8d323e796441e289970b7374142c298` |
| `benchmark.py` | `47c18b9cc3b68a3d047e1014d80d130e1ef5178cdca0d2e6111c185140ce95ae` |
| `benchmark_h64k_01.json` | `16c2a1f904b4a451ada6b4e8400dedde2103bb17d2e1788813a1a3d799ddf7bc` |
| `benchmark_h64k_02.json` | `609784782aed05e88299339d7b744a372b23a6d9aeb336cd078ebdec3ea9f94a` |
| `benchmark_verified.json` | `0b382da3ac89c9f4a8c425fd688a9825536b6475879867c434fe136f75debeb2` |

The measured source bytes and complete mapping are retained in
`source_snapshot/` and `source_snapshot_manifest.json`; frozen C10 source is
also preserved in the V2 snapshot. These measured hashes identify the archived
C11 code. The worktree was subsequently restored to accepted C10 at the user's
request; exact restored hashes are in the deferral manifest. Any future reuse
or implementation change requires fresh affected validation and timing.
