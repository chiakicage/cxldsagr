# Official ECHO compute-graph enablement

The official model callback path passed its three prepared checkpoint gates on
physical GPU 2. The experiment CLI and saved-report checks now expose this path
without changing the official numerical policy. No new formal official latency
measurement has been run. The published eager report and its older local
comparison remain in place until coordinated replacement.

## GPU correctness evidence

Execution record: `/tmp/deepseek_official_triton_validation_20261004_01`.
The wrapper is `/tmp/deepseek_official_triton_gate_20261004/run.py`, SHA-256
`a0d05242247ed86f1231d724808afc2460895a7f9af664e09e91080397922b00`.
Exec session 22768 / child PID 2021178 exited 0: **3 passed, no skips**
(one dependency warning). This is a correctness gate, not a timing experiment.

```bash
PATH="$PWD/.venv/bin:$PATH" \
TRITON_PTXAS_PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas \
TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
CUDA_VISIBLE_DEVICES=2 DEEPSEEK_OFFICIAL_CHECKPOINT=/preset-models OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_official_triton_gate_20261004/run.py \
  --output /tmp/deepseek_official_triton_validation_20261004_01
```

- `test_official_checkpoint.py` ran eager and graph variants at
  H=P=65,536, NH=16,777,216, chunk=1024, A=128, ten independent copied-input
  checkpoint blocks. Each variant built two histories and served three
  candidates, including a host-recall revisit. Each checked 1,310 attention
  calls and 2,650,296,320 selected-record occurrences against independently
  projected KV mirrors. Selected KV and same-query/same-ordered-selection
  FlashMLA outputs were byte-exact. The graph variant verified complete
  projection/finish replay counts and zero eager fallbacks.
- `test_official_graphs.py` retained the ordinary consumer at H=P=2304,
  NH=4608, chunk=256. It checked HBM and ECHO, two users, A128/A121/changed-A128,
  alternate streams, independent sessions, retained output storage and graph
  ownership. The ten scheme/phase comparisons passed the unchanged
  `official-resident-calibrated-fp64-v1` policy. Maximum hidden/logits relative
  L2 was 0.0009825049534584896 / 0.002458706656868961; every element met the
  existing elementwise tolerance. The intentional A121 eager fallback belongs
  to this bounded correctness case and is not allowed in formal graph runs.
- The wrapper forbade `torch.compile` before production imports. It required
  the exact three passing node IDs, all passing setup/call/teardown outcomes,
  stable source/compiler hashes, and exact callback scheme/phase coverage.
  Sixteen live SM90 Triton quantizer specializations had PTX/CUBIN hashes,
  matching retained CUBIN hashes and loaded module/function handles. Runtime
  build identity matched current build identity and the shared collector.

Evidence hashes:

| File | SHA-256 |
|---|---|
| `result.json` | `6a851d5c207ad90ac8a41732dc85f3f86c032633233d0a63868bc93cadf76e2f` |
| `runtime.json` | `3209f1d19146756abb2cbb42f70921b35cdd78b1783a0b34e129b5c2dbad4526` |
| `numerical_summaries.json` | `0f5cd5d4320898a2e5122e1c68235b76a5a37e74cb8e7e32c325f67a68d2f953` |
| `source_before.json` and `source_after.json` | `d560e45f8131f2f6b34ebd50ae556bcd62e6b700cbf68e5013ecca53ad683a11` |
| `test_outcomes.json` | `d102cef8a61af16e95f0bdd2b52fde2781e203bd1d02ff836dbefdec77de27dc` |

## Experiment changes and acceptance boundary

The GPU gate ran before these experiment-harness/report edits; model, cache and
operator code stayed unchanged. Subsequent CPU checks cover the new configuration
and persisted-evidence contracts. A fresh official formal run must validate the
complete new harness, its resident repeat and final report before publishing
performance.

- The CLI accepts `--compute-graphs`. New configs save canonical Boolean
  `enable_compute_graphs`; both accepted aliases require exact bool values and
  reject conflicts. Legacy saved configs with neither field remain eager and
  preserve their exact absent-field form during re-audit.
- Measured rows and resident-repeat rows record before/after graph states and
  use the shared replay validator. Repeat metadata also saves the final backend
  state and shared reservation. Graph preparation remains outside request
  latency. Warmup retains its existing lifecycle checks; it does not claim a
  separate saved replay-count audit.
- The report checks graph policy, query/layer coverage, plan and bank bounds,
  per-request replay deltas, zero initial counters, continuity, zero formal
  fallback and agreement with final bank state. Eager/legacy input does not
  require graph records, but contradictory enabled/allocated metadata fails.
- The local/official comparison requires equal normalized graph mode. Where
  both reports record effective precision flags, common flags must match;
  graph comparisons require the standard effective flags. Older eager reports
  without this evidence retain the explicit uncertainty boundary. Experiment
  policy labels and unmatched fields are not treated as proven full equality.

`NUMERICAL_POLICY` and `NUMERICAL_POLICY_BASIS` are unchanged. Do not use the
passing same-order mirror test to claim independent upstream top-k calls are
bitwise deterministic, and do not fit new tolerances to a later formal run.

## CPU validation

The complete official experiment test directory passed **107 tests in 2.83 s**
with no skips, using `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2` and both
compiler paths from the GPU command above:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider \
  experiments/deepseek_v32_echo_official/tests
```

Ruff check and format check passed for the three changed source modules and
their three test modules. The final report edit only joined a formatter-selected
expression and did not change behavior. AST comparison against the frozen C9
official source confirmed unchanged `NUMERICAL_POLICY` and
`NUMERICAL_POLICY_BASIS`; the README's local navigation links also resolved.
These checks do not constitute a formal official timing run.
