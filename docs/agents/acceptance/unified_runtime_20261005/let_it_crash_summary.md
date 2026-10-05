# Error-propagation acceptance summary

Existing evidence supports propagating execution errors without automatic request
retry or provider fallback, while retaining necessary transaction/drain cleanup.
The covered injected failures include paired body/cleanup errors, independent
stream drain failures, poisoned resource ownership, graph record/close failures,
and CLI request/backend-close failures. A failed drain preserves ownership and
storage until safe teardown; it does not report success or silently resume.

At the earlier error-handling freeze, the global CPU log ended with
**3765 passed, 1274 skipped, 58 subtests
passed** in 84.81 s (one warning). Original log: `/tmp/cxldsagr-refactor-crash-final-cpu.stdout`;
SHA-256 `e4d55b7305018a030a3cc8ecf3ae4f61192f92fda6267631f308f949383163e9`. This was the error-handling freeze before the later pure
layout-cache optimization. Skipped hardware cases are not counted as validation.
The existing [layout-cache CPU receipt](nosa_layout_cache_cpu_01/evidence.json)
records 264 passing tests, including the serving owner/resource-drain suites,
with the changed production adapter hash; receipt-file SHA-256 `3758b3c98c1743c24576e93d0e4095ef3ac9b5e406ba63e5f70cdc997c68b4d9`.

Representative retained cases are in `cache/tests/test_lifecycle.py`,
`cache/tests/test_staging.py`, `cache/tests/test_sparse_token_pool.py`,
`models/nosa/tests/test_failure_propagation.py`,
`models/nosa/tests/test_resource_drain.py`, and `serving/tests/test_runner.py`.
The final NOSA numerical receipt independently validates ordinary successful
execution, but does not replace injected failure tests. The preceding historical
summary reused existing evidence without another test run or new fault injection.
It did not claim an exhaustive reread or new CUDA fault-path acceptance.

## CPU and entrypoint validation before final style normalization, 2026-10-05

The recorded complete CPU regression exited 0 with **3834 passed, 1274 skipped,
58 subtests passed**, one profiler-cycle warning, in 82.07 s. It ran with CUDA
hidden, CPU affinity 0–7, one OpenMP/MKL/OpenBLAS thread, disabled bytecode writes
and `PYTEST_ADDOPTS=--tb=short`:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS=--tb=short taskset -c 0-7 bash scripts/run_tests.sh cpu > /tmp/unified_global_cpu_final_20261005.log 2>&1
```

The log SHA256 is
`825269dc7b248d19ebe285a8fc43da7e4acb56b095c12e0ff742723da576267b`.
Hardware and opt-in skips are not GPU validation. These regression results do
not replace independently accepted numerical experiments or measured performance.
This run and the CLI checks below preceded the final import-order/formatting
normalization; they do not certify the later source bytes.

The preceding diagnostic rerun reproduced two stale published-source expectations
in the native/Triton cases of `test_checkpoint_source_graph.py`. They indexed new
metadata using a historical graph containing the retired experiment capture path.
The test alone was corrected to require every anchor of the accepted
failure-preserving graph; unknown and missing-source rejection tests remain.
That file then passed 508 tests before the complete run above. This was a
test expectation correction, not a runtime/kernel repair.

All 12 recorded CLI help checks exited 0: both model module and direct-script
entrypoints, both serving entrypoints, cache and NOSA publishers, the NOSA summary
renderer, the global test script, cache run script and real-three-layer run script.
`/tmp/unified_final_cli_check.json` preserves each command and result; its SHA256 is
`357379eb44a83dd90cd87635c6087c9e78581192a0a14db7c702ed04fd0aaa23`.
These checks validate parsing/import entrypoints, not model execution.

## Report-publication failure contracts

The focused publication/helper suite passed 29 tests with importlib collection;
the complete CPU regression above also covers these cases:

- Multi-directory publication attempts all owned rollbacks in reverse order.
  Pre-existing destinations are preserved. Original publication and cleanup
  exceptions retain their identities, including interruption cases.
- NOSA report staging cleanup no longer suppresses failures. A complete committed
  report remains intact if later staging removal fails; paired body/cleanup errors
  are propagated together. The outer rename-based publisher preserves the same
  commit boundary when removing its empty staging root fails.
- Report-helper snapshots retain exact saved bytes and hashes, enforce repository
  source exclusions and propagate source/write failures without retry or fallback.

The adjacent cases are in
`experiments/nosa_motivation/tests/test_provenance.py`,
`experiments/nosa_motivation/tests/test_report_publication.py`,
`experiments/nosa_motivation/tests/test_publish.py` and
`evaluation/tests/test_provenance.py`. No new GPU fault-path injection was performed
for these report-only changes.

## Source and publication boundary

Original GPU execution/native identities and independent numerical receipts remain
attached to their captured runs. Later report-helper snapshots save the current
disk files corresponding to already-loaded repository Python modules, plus
`pyproject.toml` and `uv.lock`; they do not identify loaded bytecode, all possible
sources or the earlier GPU/native execution. In particular, DeepSeek's final
`source_bindings.json` records that its helper collection occurred after the
completed analyses and that the newly added snapshot function was not called by
those earlier commands. Existing-function AST equality and the original source
hash remain separate records.

Published report manifests and their verified source boundaries are indexed in
[evidence.json](evidence.json). Successful failure propagation, CPU regression and
publication integrity do not imply serving speedup, overlap success, physical
capacity validation or exhaustive fault coverage.

## Final source and release validation, 2026-10-05

After the final import sorting and whitespace changes, the complete CPU regression
again exited 0: **3834 passed, 1274 skipped, 58 subtests passed**, one warning,
in 81.70 s. All 12 CLI checks were repeated and passed. Ruff check and format
check passed on all 236 changed/new Python files. The source/document diff check
passed with generated experiment report trees excluded; original CSV/SVG bytes
and accepted hashes were preserved instead of normalizing generated whitespace.

The [final validation record](final_validation.json) preserves commands, log
hashes, static checks and remaining limits. The [five-file source delta](static_style_source_delta.json)
records exact before/after bytes, identical non-import ASTs and identical imported
module/alias multisets. This CPU/static validation does not change the original
GPU source identity or constitute a new GPU timing/fault-path run.

All eight replacement reports are published. The completed refactor process
directory was deleted; T-007's four necessary documents and six original numerical
reference files remain. Final navigation found no missing local links or runtime
file-reading references to the deleted process documents. Residual timing costs
and failed serving performance targets remain explicit in the reports.
