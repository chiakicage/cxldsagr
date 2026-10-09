# Integration checkpoint

Candidate: `official-q1-predictive-staging-promotion-v1`.
Parent: unchanged pinned official Q1 decode kernel with local pool adaptation.
Current production integration is accepted by the final
`deepseek_h64k_a1_official_20261008_03` real-three-layer matrix; see
[the current checkpoint](checkpoint.md) for the independent final audit and
separately accepted per-operator profile. The source identities and first
22-test result below describe the earlier adapter before immutable loading and
multi-CTA promotion; they remain attached only to that prior version.

The command in `integration_task.md` passed **22 tests in 46.42 s**. Coverage:
zero/partial/full/over-cap attempts, free and occupied FIFO victims, exact BF16
record bits, forward/reverse ownership, allocation journal, final priority/free
state, host ID zero, safe padded token tables, idempotent partial/full cleanup,
prepared-token consumption, headroom rejection, and injected post-stage failure.
An actual Q1/N257 official bridge invocation staged 64 unique history records,
promoted exact host bytes and matched a no-miss official score control bitwise.
Ruff passed. An initial command lacked `.venv/bin` on PATH and failed to find
`ninja`; adding the existing environment bin directory resolved setup. No
dependency was installed and no failed result was published as an experiment.

Source identities and the artifact inspected after that test run:

- `official_prefetch.py`: `ad0c3e926502b647be514740a8a817f14c01bdba83f88384cf1ac34fc0cbb111`
- `csrc/official_prefetch.cu`: `fcc2bf553ede3838ecaebd35383cc1c60102858958ade63352680b798c4cbe32`
- TVM FFI binary: `/root/.cache/tvm-ffi/cxldsagr_official_prefetch_562305c732186b09_ec74e6f155c7ef98/cxldsagr_official_prefetch_562305c732186b09.so`
- Binary SHA256: `d8d4b9409169d832f70d107cbf9be7a20d364dbee8dbdc3a4bf47f6d572a7096`

The binary identity above was inspected after the tests rather than emitted by
a numerical receipt. A later investigation found TVM FFI/Ninja could rebuild
unchanged sources into different ELF bytes between processes. The parent is
replacing that loader with a source/toolchain-bound immutable artifact cache;
formal acceptance must bind the actual loaded artifact after that change.

The callback key is `_pending_prefetch_cleanup`; retained staging is exposed as
`_official_prefetch`. The parent invokes cleanup inside the owning cache lease
before finalization and does not release state on failure. Successful promotion
leaves no temporary tags. The parent releases its local prefetch dictionary
after finalization to avoid retaining staging through attention.

Additional live adapter storage is `4*N + 64*(1152+4)` B, or **336,132 B** at
N=65,537, reusing the existing uint32 counter. Promotion adds `copied*1152` B of
D2D traffic, at most 73,728 B. H2D remains `(copied+residual_recall)*1152` B and
includes predictive false positives. Packed indexer/schedule/output allocations
and graph private reserved memory are accounted separately by the parent.

Full-model proof requirements: bind source/native/input/residency and both hint
slots; identify this official policy and effective cap64 separately from the
prepared local limit; certify strict FP32 `score > initial_hint[1]` eligibility,
host-backed history-only access, actual stage IDs/order/count, unique legal FIFO
publication and journal, no leftover tags, all stage clock/free/priority/counter
transitions, unchanged exact selection, residual recall and final resident KV.
The final resident set includes predictive false positives plus append and exact
selection. Unsaturated execution must consume every eligible history miss;
saturated execution may choose any eligible64. Official attempts may exceed64;
successful copies are `min(attempts,64)` and rejected attempts are the remainder.

These tests do not establish full-model correctness, H64K cold performance,
nondefault-stream or complete-graph behavior. The official bridge owner is
checking the latter two after adding explicit torch-stream scopes. The parent
owns independent full-model acceptance, complete timing/profile, budget audit
and experiment publication.

The complete-model observation and CPU proof now support this policy explicitly.
The existing coarse branch remains unchanged. New official receipts distinguish
effective cap64 from the prepared local limit and select hint slot1. Diagnostic
buffers capture actual stage IDs, prepared FIFO order, allocation journal and KV;
runtime checks compare staged bytes with host backing. Compact evidence retains
stage IDs/order/journal and checked KV identities, without claiming to retain or
recompute omitted score/KV payloads.

The dependent graph and matrix comparison gates permit differing official
false-positive resident sets and free counts only after both executions pass
their independently bound transition proof. Every execution still proves
`resident = predicted + appended + exact-selected`, legal ownership, free count,
clocks and actual transport statistics. Host KV/indexer bytes, exact scores,
top-k IDs and full retained hints remain strict across matching executions.
Coarse receipts retain their stricter residency comparison and prior schema.

The following final CPU command passed **94 tests in 2.65 s**, covering strict
FP32 threshold equality, unsaturated/full official stages, predictive false
positives, padding, stage/journal/KV/counter corruption, compact reload, observer
copy ordering, and both downstream receipt gates. Ruff and `git diff --check`
also passed. These eight source/test files were then frozen for parent runs.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  taskset -c 24-31 .venv/bin/python -B -m pytest -q -x --tb=short -p no:cacheprovider \
  experiments/deepseek_v32_mfu/tests/test_prefetch_transition_audit.py \
  experiments/deepseek_v32_mfu/tests/test_prefetch_validation.py \
  experiments/deepseek_v32_mfu/tests/test_extend_graph_validation.py \
  experiments/deepseek_v32_mfu/tests/test_audit_shape_matrix.py
```

## Read-only full-model binding review

After the source freeze, the parent reported a successful immutable-loader raw
check `_04` and another 22-test adapter GPU run. Those are parent-reported until
their records are independently inspected; the identities above remain attached
only to the earlier adapter candidate.

The current observation path is `measure.main -> profile_layers.run -> run_check
-> _check_extend_graph`. Only the cold ECHO independent check installs the
observer. Each prepare saves the actual returned lease; the finalize wrapper
receives that same dictionary and copies official stage IDs/KV, sorted slots and
journal before the cache calls its cleanup hook. The wrapper then snapshots
post-finalize maps, counters, free state, priority and clock. These copies precede
append/recall reuse of the shared counter and scratch. The model releases its
ordinary prefetch reference after finalize; only the diagnostic observer retains
its lease until the after-recall snapshot. Clean timing and profile never install
the observer.

Six separately bound observations cover baseline, default graph, two graph
replays, changed-input baseline and changed-input graph. Graph copy destinations
are allocated before capture and owned until the graphs close. The runtime local
native collector reads actual process mappings and includes all `cxldsagr_*.so`
artifacts, so both immutable bridge and promotion adapter paths/hashes enter the
before/after execution identity without scanning unrelated cache files.

The execution reservation includes adapter storage before taking the maximum of
the normal and Q1 indexer peaks. It is added once per device. Independent CPU
recomputation found N=65,537/Q1 indexer 12,289,584 B and total execution HBM
12,291,888 B; Q1024 remains 1,212,157,952 B and 1,214,517,248 B respectively.
At N=65,664 the Q1 indexer floor becomes 12,298,544 B. Published small-Q floor
claims therefore need updating even where complete Q1024 capacity plans remain
unchanged. Full graphs separately check all private-pool segments and static
input blocks, and report allocated, reserved and device-used values. These are
static binding findings; formal complete-request artifacts remain to be reviewed.


## Independent reread of the integration check

The CPU-only audit of `deepseek_official_q1_integration_20261008_01` passed. Its
receipt binds 30 artifacts; all 1,368 archived source files and 685 runtime or
native dependency files were rehashed. Before/after backend provenance is equal.
The receipt signature is
`1ecdcd659ca1223447829e1788d3db6365276bedbac00b793d36db44c5480496`.
This is the check under
`/tmp/cxldsagr-checks/deepseek_v32_mfu/data/deepseek_official_q1_integration_20261008_01/`.

All six official compact proofs were reconstructed, covering 18 layer states.
Every layer stages 64 records, clears temporary tags before normal publication,
publishes according to the saved FIFO allocation journal, preserves all exact
selection, and satisfies `H2D=(prefetched+recalled)*1152`. The baseline layers
have respectively 0/57/46 predictive false positives, 1,983/2,040/2,029 residual
recalls, and H2D 2,358,144/2,423,808/2,411,136 B. Each layer writes 1,152 B D2H
and promotes 73,728 B D2D. Different saturated schedules select different legal
false positives, which the six proofs account for separately.

The 36 saved numerical tensors passed all 31 independently reread comparisons
bitwise, with maximum absolute error zero; the changed input changes both
hidden and logits. The 13 top-level runtime numerical comparisons also report
bitwise equality. Default/repeat outputs not saved as tensors retain their
runtime comparison evidence. Compact proofs retain score/KV identities and
eligibility bitmaps, not raw score or KV payloads; eligibility derivation and
actual staged/final KV equality therefore remain the bound runtime evidence.

All four diagnostic full graphs report 56,623,104 B private reserved storage,
512 B static storage and a 2,147,483,648 B chosen private limit. Their total
reservation is 2,147,484,160 B. Allocated, reserved and device-used snapshots are
recorded separately; these diagnostic observer graphs are not clean benchmark
memory measurements. Failure cleanup and poisoned-owner behavior remain
covered by the source review and dedicated tests, not by this successful run.

The mapped immutable official decode library hashes to
`e0b87399b27f1150f0de6e39fd401afe19212f8fff6ebfa20eda25a0afc85586`;
the promotion adapter hashes to
`49be640870652188a14ee39858d801f1689fa7135993ead3deb92b3ab7d1d73b`.
Both on-disk libraries still match the execution records. The complete reread
and hash inventory are in `/tmp/deepseek_official_q1_integration_independent_audit.json`,
with its CPU helper at `/tmp/deepseek_official_q1_integration_independent_audit.py`.

This run contains no performance measurements. At reread time, current
`models/deepseek_v32/projections.py` and
`models/deepseek_v32/execution/extend_graph.py` differ from the accepted source
snapshot; the subsequent projection scheduling candidate needs its own check,
formal timing and profile. This integration evidence does not validate that
later implementation.


## Capacity publication after official staging

`cache_deepseek_q1_impact_20261008_02` independently recomputed the five retained
capacity plans using the current reservation source, without CUDA initialization.
All 1,222 scalar values and every selected/useful-P/next-infeasible ledger are
identical to the original plans. At N=65,664, official staging contributes
336,640 B and the Q1 indexer floor is 12,298,544 B. The complete formula is
`max(1181696*Q + 2101248, 12298544, 4816896) + 2304*Q` B. Q>=9 retains the
prior linear result, including Q1024 total execution HBM of 1,214,517,248 B.

The new static report binds 18 source files and its publication binds 30
artifacts. Published impact SHA256 is
`59cd33c500f8141d2226d589e83a90aed4f521bda68514c9df84fb054f202674`.
`experiments/cache_management/README.md` and the two Q1 report JSON files now
reference `_02`; the superseded `_01` Q1 impact output was removed after
publication. The five original capacity plans, DMA provenance and NOSA results
were preserved. No source-bound experiment implementation was edited, and no
request-memory, graph-pool, capacity-fill or physical-peak claim was added.
