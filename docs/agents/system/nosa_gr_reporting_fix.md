# NOSA GR optional host-flag accounting fix

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

Root reviewed and accepted this reporting-only correction on 2026-10-04.
The HBM session budget reserves one pinned host-flag byte, but the cache creates
that flag lazily. H4K and H16K requests do not reach the native indexer's checked
branch, so zero retained host-flag bytes is valid. The reservation remains
necessary for admission even when the flag has not been allocated.

`experiments/gr_serving/src/report.py` and `audit.py` now require equal actual
DRAM values before and after cleanup. For HBM these values must be nonnegative
integers no greater than the number of retained users. Fixed, temporary, shared
and total reservations remain exact. Offload schemes still require their exact
pinned K/V allocation bins. No model, cache allocator or execution path changed.

The focused regression run passed all 167 CPU tests:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  experiments/gr_serving/tests/test_nosa_shared_accounting.py \
  experiments/gr_serving/tests/test_gr_serving_report.py \
  experiments/gr_serving/tests/test_gr_serving_audit.py \
  experiments/gr_serving/tests/test_gr_serving_audit_sources.py
```

Ruff formatting/checks and `git diff --check` passed. The added regression cases
cover absent, newly created and retained flags, mixed two-user flag states,
invalid actual values, unequal cleanup boundaries, incorrect reservations and
undercounted offload allocations.

Review artifacts are outside the experiment directories at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_reporting_fix_20261004_01/`:

| File | SHA256 |
| --- | --- |
| `reporting_fix.patch` | `6bb6ec872c26af7ba1120afe012a34e529e0ad87d3b440cef220f4252278729e` |
| `review_receipt.json` | `bfb736eaa7ef8cdd52465aaf1b8696cc2426ae0b97d502a0ed4b260a0138a500` |
| `failed_rows_diagnostic.json` | `d061ffbd233c69577e0322196597cbb647d3c0651e6ca8ad458bd0fd0ee76a80` |

The read-only diagnostic checked all 804 rows across 28 cases from failed run
`gr_nosa_flags_h4k_20261004_01`: 201 HBM rows had zero actual DRAM bytes, and all
603 offload rows matched the exact pinned bins. It reused saved request records
and the existing workload proof without reading tensors. Failed-run metadata,
measurements and correctness records were unchanged. This diagnostic does not
accept or publish that failed run and supplies no performance result.

The three-file correction changes the 280-file GR source digest from
`a37e87ee66ba7699e79d05b88d17cd96a5ce01360e6e334a7087ce90db5bc933` to
`8dc00e23469a67ebee67eb603859c9de9987e9151520a607d1bb20a498e547f9`.
The 114-file fixed-serving/runtime digest remains
`a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`.
Root authorized a fresh freeze and unused `_02` GR run IDs. Each new family
still requires its own completed measurement and profile acceptance; the failed
`_01` data cannot substitute for those runs.
