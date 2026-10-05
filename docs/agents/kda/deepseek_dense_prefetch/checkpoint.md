# C7a component checkpoint

Status: independent source review, GPU correctness and exclusive component
timing are complete. Combined full-model validation with the exact ECHO hint
also passed, as recorded below. Root first granted GPU 2 for correctness while a norm screen
could run on GPU 0; no performance data was collected in that window. Timing
used a later exclusive GPU 2 grant. All C7 GPU processes have exited and CUDA
was explicitly released to root.

Implemented dense history classification without consumer-selection counters;
full-P flags/prefix and temporary sort remain within the caller lease. Positive
miss tickets own separate int64 M-sized host and physical ID allocations, with
binding checks rejecting shared/oversized/offset storage. The dedicated reserve
kernel uses dense maps-before-copy semantics on the caller stream. The existing
private-copy readiness/wait/drain sequence is retained. Metadata, event creation
and copy submission now share a failure guard that disables helper reuse.
Generic gather source, CTA geometry and caching policy are unchanged.

CPU validation: 28 passed, 26 CUDA skips in 1.97 seconds. The five new CPU cases
cover post-reservation failure cleanup and rejected borrowed ticket storage.
Ruff check and format check passed. GPU tests are prepared for full-state checked
comparison, fragmented pages, partial-free/full miss, delayed-copy scratch reuse,
H=0/all-hit/rollover, candidate tails and failures after reservation/copy enqueue.
The GPU outcomes are recorded below.

GPU validation command executed:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q models/deepseek_v32/tests/test_pool_prefetch.py models/deepseek_v32/tests/test_pool_prefetch_native.py cache/tests operators/common/tests/test_kv_transfer.py operators/deepseek_v32/indexer/tests
```

Component probe: `/tmp/deepseek_c7_dense_probe.py`; completed run ID
`deepseek_dense_prefetch_c7a_20261004_01`. It compares complete prefetch/wait/drain
on identical restored snapshots, five warmups, 25 alternating samples and three
single-call traces, with full state/ticket/metrics comparison before timing.
The separate timing window was granted after C6 profiling released all GPUs.

Source held stable during independent review and GPU validation:

- `models/deepseek_v32/pool_prefetch.py`: `f27260feaa6b21d302232e29325a461a31d158d6ef8671f7dfa164c2c3a62a7b`
- `operators/deepseek_v32/indexer/cache_ops.py`: `9283837189aeb5b8a0e55e2e6e6a25e24e08a1bb2f545edf2bf11fec2b93175c`
- `operators/deepseek_v32/indexer/csrc/echo_indexer.cu`: `86793e7375d8204a1fe445f42ecb4efd35d7b6083fa82ee2bf4e3e0a12da3eab`
- `operators/deepseek_v32/indexer/csrc/echo_dense_prefetch.cuh`: `30374a97873ecd7b7d87730ba40c400e111d0e3ba32feee3982efc8c508bb59d`
- `models/deepseek_v32/tests/test_pool_prefetch_native.py`: `8682408068e722a6973e2c262856d5224d72f5b54f34a750cd04a952ad5e37db`


GPU acceptance details:

- Initial full cache/indexer/prefetch/generic-transfer invocation collected 313
  cases: 311 passed, with two test-fixture failures and no implementation state
  or numerical mismatch. All four explicit event fault cases already passed.
- The old CPU host-unchanged test compared unused float storage containing NaNs
  with ordinary float equality. It now compares the entire backing bytes exactly.
- One delayed-copy test refused acceptance because the delay ended before shared
  scratch overwrite. Complete helper/scan/sort/transport paths now warm up on a
  restored snapshot before installing a longer, explicitly checked delay. This
  avoids accepting a test that never exercised the intended concurrent lifetime.
- Both changed helper test files then passed all 38 cases in 9.04 seconds. Every
  delayed-copy case verifies the delay is still pending when scratch is overwritten.
  Metadata-ready and copy-ready event constructor/record failure cases preserve
  the required copy-input lifetime until drain and recover the other user's data.
- Production source was unchanged between the first full invocation and the
  targeted fixture rerun. The test counts above describe their actual scopes;
  this is not presented as a second complete 313-case run.

The backing-byte test correction also changes
`models/deepseek_v32/tests/test_pool_prefetch.py`:
`5876f011dfb2553f7ee68df250550eca9800faa9383829e02f41d585e6b311b0`.

Read-only resource inspection after correctness found module
`cxldsagr_echo_indexer_513a53c54269f9bc`. Dense classify uses 22 registers/thread,
reserve uses 32, each with 2048 static shared bytes and zero stack/local bytes.
Both launch 256-thread CTAs with no cooperative launch or inter-CTA wait. Counter
clear uses one thread (6 registers, 1024 reported static shared bytes). Resource
output is `/tmp/deepseek_c7_resource_usage.txt`; it is not a performance result.

CPU timing-harness audit confirms the same committed snapshot and candidate data
are restored outside timing for checked and native paths. Five cases separately
cover certified all-hit, uncertified all-hit, half displacement, full displacement
and partial free slots. The certified fixture records its no-competitor cold-build
proof, then reapplies it only after restoring that exact snapshot and writing the
independent candidate tail. Other fixtures remain uncertified after restoration.
Both methods reuse a drained helper and the unchanged generic transport.

The timed call performs prefetch, wait, drain and final device synchronization;
snapshot restoration, candidate append/discard and attention are excluded. No
next-layer lookahead or full-model overlap is measured. Ticket/storage inspection
is restricted to untimed comparison and allocation samples. Modes alternate after
their warmups. All ticket references are dropped by drain before the next restore.
The profiler reports summed kernel duration as `kernel_work_us`; it excludes
memcpy/memset and is neither a union nor an overlap measure.

An independent second review accepted the revised harness. It reserves a fresh
output before CUDA initialization, refuses existing trace names, and records the
harness, shared cache, generic transport and native source identity, including
the shared-header build fingerprint. Identity is rechecked after measurements.
Harness SHA-256: `04413e1ec4da529fbc03ab6aa46b7671f69f07be10e162f5a290adeee57d076b`.
CPU compile passed. No host-cache/bandwidth claim is inferred from snapshot
restoration.

## Exclusive component timing

Run `deepseek_dense_prefetch_c7a_20261004_01`, exec session 43751, exited 0 on
2026-10-04. GPU: NVIDIA H200, SM90; PyTorch 2.12.1+cu130, CUDA 13.0. The timed
helper boundary and exclusions above apply to every row. No concurrent CUDA
work was permitted by root during this run.

| Fixture | P / H / A | Checked median ms | C7a median ms | Speedup | Checked / C7a kernel launches |
|---|---|---:|---:|---:|---:|
| Certified all-hit | 65536 / 65536 / 128 | 0.082902 | 0.083091 | 0.998x | 1 / 1 |
| Uncertified all-hit | 65536 / 65536 / 128 | 0.454114 | 0.184507 | 2.461x | 17 / 3 |
| 32768 displaced | 65536 / 65536 / 128 | 2.052901 | 1.203246 | 1.706x | 72 / 18 |
| 65536 displaced | 65536 / 65536 / 128 | 2.775675 | 1.929521 | 1.439x | 70 / 18 |
| Partial free slots | 193 / 137 / 35 | 1.112178 | 0.357596 | 3.110x | 58 / 9 |

The last fixture writes a competing 150-token history then truncates it to 80.
Both paths report 43 hits, 94 misses and exactly 24 evictions. All five fixtures
pass byte-exact complete pool state, clocks, metrics and ticket comparisons.
Candidate writes remain temporary, with no host writes or selection-counter
changes from whole-history prefetch.

The certified case uses the same existing C3 kernel in both modes, so the
0.189 microsecond median difference is not evidence of a changed fast path.
For full displacement, unchanged gather takes 1472.559 / 1471.843 microseconds
checked / C7a; total summed kernel work is 1648.936 / 1536.642 microseconds.
This attributes the measured improvement to metadata and launch/host overhead,
without asserting overlap or a transport improvement.

Temporary PyTorch allocated-byte deltas are not uniformly smaller: half
displacement is 3,236,864 / 3,521,024 bytes, while full displacement is
6,208,512 / 3,521,024 bytes. Private ticket arrays occupy exactly 16 bytes per
miss (two owned int64 arrays) in both compared helper paths. Every fixture has
zero retained allocated delta after drain. These samples are neither reserved
memory nor process peak/device-usage capacity measurements.

Post-run checks verified complete status, all 250 wall-time samples, all 30
readable traces and unchanged hashes for all 16 recorded sources. The native
build fingerprint, including shared headers, was checked after each case and
at completion: `c2b3905b211ef59fe56032de605818aa877acc403431dcd2cbc6c45703ed23bf`.
CUTLASS: `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`.

Generic transport remains unchanged:

- `operators/common/kv_transfer.py`: `bc65f850a908dfc4642b7f90170513881a0a564561d9902c736d8e7d7f16b919`
- `operators/common/csrc/kv_transfer.cu`: `e082e3b2ef47bad7224f14213c4fe57a6d0bda7399a556230a814d145466bf1b`

Evidence: `/tmp/deepseek_dense_prefetch_c7a_20261004_01.json`, the 30 trace
paths recorded there, and `/tmp/deepseek_c7_dense_probe.stdout` / `.stderr`.
Stderr contains profiler cycle/USDT notices, with no execution error.

Executed command:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 \
.venv/bin/python /tmp/deepseek_c7_dense_probe.py --profile \
  --output /tmp/deepseek_dense_prefetch_c7a_20261004_01.json \
  > /tmp/deepseek_c7_dense_probe.stdout 2> /tmp/deepseek_c7_dense_probe.stderr
```

C7a is accepted at the component gate. Combined full-workload correctness and
formal serving acceptance are recorded below. Actual lookahead overlap remains
unprofiled for C7; no C7b transport implementation is included in these results.

## Combined checkpoint/lifecycle gate

After the exact ECHO hint was integrated, root granted a new exclusive GPU 2
window for `deepseek_c7_hint_combined_validation_20261004_01`. Exec session
52237 exited 0: 176 tests passed, no failures or skips, in 93.97 seconds. All
four schemes matched complete candidate hidden/logit values at H=P65536,
NH131072, C1024, A121 eager fallback and A128 captured execution. The command
also passed the smaller real-checkpoint runner lifecycle/failure cases, C7a
private-ticket tests and all 43 hint tests. All 109 recorded source/test/config
hashes and the full native/shared-header build identity remained unchanged.
CUDA was released after verification.

The combined validation checkpoint（Git `934485b:docs/agents/system/deepseek_motivation_c7_hint_validation.md`）
contains the exact command, source identities, coverage boundaries and temporary
evidence paths. This correctness run supplies no latency/MFU result.

## Combined formal serving acceptance

Root accepted `motivation_c7_hint_20261004_u16_r2_01` using the combined C7a
helper and exact ECHO hint. The independent C7b capped transport is absent.
The combined integration record（Git `934485b:docs/agents/system/deepseek_motivation_c7_hint_integration.md`）
records frozen source identity, full numerical checks, arithmetic audit and
publication/replacement scope. Dense-prefetch first-request latency is
2375.510656 ms; revisit latency is 34.536917 ms, compared with 40.261614 ms in
the accepted C6 observation. These are complete-request measurements of the
combined implementation, not an isolated causal attribution to C7a.

No matching C7 operator/API profile or internal lookahead-overlap measurement
has been accepted. The C6 zero-overlap diagnostic cannot establish C7 overlap
or C7 API MFU. C7b's separate generic correctness and remaining pair work are
tracked in [its checkpoint](c7b_checkpoint.md).
