# C7 executable plan

Root granted implementation and then GPU correctness after freezing C6. C7a
passed independent review and the validation described in checkpoint.md.
Exclusive component timing also passed, and root accepted C7a for combined
full-model validation. The combined C7a/hint command then passed all 176 tests;
formal performance and report replacement remain pending.

- [x] Read current helper, backend lookahead order, generic gather and tests.
- [x] Cross-check ticket/scratch lifetime with independent CPU reviewer.
- [x] Record C4 request baseline and older C3 transport observation separately.
- [x] After root grants implementation, add C7a dense-range classification and
  private-M-ID compaction/reservation, preserving both FIFO events and metrics.
- [x] Add strict full-state checked differentials and metadata failure injections.
- [x] Add a delayed-copy test: enqueue sleep on the copy stream, prefetch next
  layer, overwrite every shared scratch buffer during current-layer work, then
  wait and verify exact target records/IDs. Check own storage and retained bytes.
- [ ] Cover same/changed caller streams, foreign/expired/unconsumed tickets,
  stale map_generation, speculative unused copy, failure after reservation and
  copy submission, drain-before-rollback/release, and poisoned completion.
- [x] Run CPU helper/cache tests, then targeted GPU correctness only in root's
  granted quiet window. Broaden to cache/indexer/prefetch/generic-transfer suites.
- [x] Benchmark C7a complete prefetch/wait/drain with restoration outside timing;
  record source hashes, run ID, shapes, warmups/repeats, kernel traces and memory.
- [x] Freeze/promote C7a on exactness plus latency evidence, or reject it. Do not
  mix transport tuning with this comparison.
- [ ] If root authorizes C7b, prepare a generic optional CTA cap with a grid-stride
  record loop. Inspect resource usage; verify every record and unaligned fallback.
- [ ] Compare cap choices on current full GEMM/attention workloads and serial
  control; measure actual work intervals, transport bytes and complete latency.
- [ ] Rerun full-model and affected experiment trajectories through root before
  report replacement. Preserve prior valid results until new acceptance.

Planned CPU entry:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q models/deepseek_v32/tests/test_pool_prefetch.py cache/tests/test_sparse_token_pool.py
```

Planned GPU entry after a window grant (extend with new tests):

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q models/deepseek_v32/tests/test_pool_prefetch.py cache/tests operators/common/tests/test_kv_transfer.py operators/deepseek_v32/indexer/tests
```

Component probe `/tmp/deepseek_c7_dense_probe.py` completed run
`deepseek_dense_prefetch_c7a_20261004_01`; its exactness, timing, source and
measurement checks are recorded in checkpoint.md. Full-model command and
coverage are in [validation_plan.md](validation_plan.md); the completed result
is in the combined checkpoint（Git `934485b:docs/agents/system/deepseek_motivation_c7_hint_validation.md`）.
Final performance/publication remains root-owned.
