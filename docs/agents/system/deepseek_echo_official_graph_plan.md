# Common compute graphs for the official ECHO comparison

The final updated comparison should use the same common checkpoint computation
policy where validated. The official indexer, top-k, allocator, prefetch, recall
and writeback remain pinned and eager. No change to upstream source is needed.

## Implementation and acceptance

1. Add the existing optional `project_callback` and `output_callback` contract
   to `OfficialAttentionRunner._forward`. Default eager calls stay unchanged.
   The callbacks replace only projection and attention-output computation;
   all official indexer/top-k/cache work remains between them.
2. CPU dispatch checks must exercise resident and offload runners, exact callback
   inputs, original eager behavior, cache-operation ordering, official fused
   dispatch on every offload call, and propagation of callback failures.
3. Keep the experiment's graph CLI rejection until hardware validation passes.
   The existing backend graph planner and owner/lifetime checks are reused.
4. Validate actual checkpoint graph/eager projection and finish results with
   matched inputs. Then run independent official H64K histories and candidate
   requests, preserving the existing official numerical policy and selected-KV
   mirror checks. Cover nondefault streams, another user, A121 eager fallback,
   A128 replay, changed-input replay and failure cleanup. Check actual official
   indexer/top-k calls and graph replay counts; CPU dispatch tests alone do not
   establish GPU execution identity.
5. After acceptance, expose the common compute graph option in official
   configuration/provenance, retain explicit eager selection, and recheck saved
   configuration compatibility and resource accounting. Use fresh official and
   HBM control trajectories with all current source identities.

This is an engineering preparation for the user-requested final comparison.
No new official numerical or performance result exists at this step. Existing
published reports and backing outputs remain until accepted replacement runs.

## Prepared gates and current boundary

CPU preparation on 2026-10-04 added
`models/deepseek_v32/tests/test_official_graphs.py`. Its bounded checkpoint
geometry is H=P=2,304, NH=4,608 and chunk=256. HBM and official ECHO each use
independent eager and captured backends with two independently built histories.
The five phases are Alice's prefix, A128, Bob's prefix on an alternate stream,
Alice's A121 fallback after displacement, and changed-input A128 on the
alternate stream. The three candidate calls retain earlier hidden/logit
outputs through later work. The test checks history/indexer preservation,
the declared candidate persistence contract, actual recall on the ECHO revisit,
and graph identity, pool capacity and per-pair replay/fallback counts.

The ordinary `EchoAttentionRunner._consume` remains bound to every runner.
Observers call the real official resident/fused logits and top-k entries and
`OfficialSparseTokenCache.ensure`, then compare the complete per-phase call
sequence. Output comparisons use the existing official FP64 numerical helper
against the same-scheme independent eager backend. The thresholds are inherited
unchanged; this test does not recalibrate them or assert bitwise equality
between independent unordered top-k runs. It is not a formal HBM/offload
comparison, H64K validation or serving performance measurement.

The existing `test_official_checkpoint.py` mirror now has explicit `eager`
and `compute_graphs` parameters. Selecting that file runs both independently;
the original H64K/P64K/NH16M histories, three A128 requests and exact mirror
comparisons are unchanged. The captured case additionally requires all twenty
`(layer, queries)` pairs: each Q1024 pair has 128 projection and 128 finish
replays, and each Q128 pair has three of each. Neither gate has run with this
new source yet. Common graph same-input projection/finish exactness and
failure/ownership tests must also pass under the final integrated Triton
quantizer; these unchanged mechanisms are not duplicated in the bounded test.

Root independently reviewed the bounded test after the public retained-capacity,
reported-persistence, history-slice and explicit stream-wait changes and found
no remaining source blocker. Ruff check and format passed. CPU validation of
the callback tests, both checkpoint test files and existing official measure
tests reported **27 passed and 3 explicit checkpoint GPU skips in 1.86 s**.
Those skips are not GPU passes.

| Prepared test | SHA-256 |
|---|---|
| `test_official_graphs.py` | `c6d8f39abd2bab2704828d55f76cc13d83f00f3b52df0611fd078d35c783cb49` |
| `test_official_checkpoint.py` | `e1103fd0671e519c9241730bc634c505cb6cd8d3b247dd176ad786c5a0201988` |

After the final implementation is frozen and root assigns a GPU,
set `DEEPSEEK_OFFICIAL_GATE_GPU` to that physical device and run this exact
command from the repository root. The required variable prevents accidentally
using the default GPU before assignment.

```bash
PATH="$PWD/.venv/bin:$PATH" \
CUDA_VISIBLE_DEVICES="${DEEPSEEK_OFFICIAL_GATE_GPU:?set the GPU assigned by root}" \
OMP_NUM_THREADS=8 \
DEEPSEEK_OFFICIAL_CHECKPOINT=/preset-models \
.venv/bin/python -m pytest -s -q -p no:cacheprovider \
  models/deepseek_v32/tests/test_official_checkpoint.py \
  models/deepseek_v32/tests/test_official_graphs.py
```

The experiment CLI still rejects compute graphs. Source review, Ruff and CPU
collection cannot replace these GPU gates or the fresh formal numerical run.

## Current verification

The callback adapter and CPU dispatch checks are implemented. The focused
official model/configuration suite passed 27 tests; both opt-in full-checkpoint
variants were explicitly skipped because no checkpoint GPU window was requested
by this CPU command. The latter now checks selected KV and same-order attention
under either eager computation or common compute graphs, and validates all
projection/finish replay counts. GPU validation and CLI enablement remain open.
