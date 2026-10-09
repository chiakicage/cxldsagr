# Official Q1 staging integration contract

Owner: `review_a1`; parent task: H64K/A1 official ECHO integration.

Implement `operators/deepseek_v32/indexer/official_prefetch.py` and
`csrc/official_prefetch.cu`. Reuse the freshly compiled pinned official decode
kernel through `official_decode.py`; do not modify upstream source, model
dispatch, shared cache, or packing in this subtask.

Inputs are Q1 FP8 H64/D128 queries, current packed indexer pages, FP32 weights,
schedule metadata, and an exclusive prepared local sparse-pool lease. The one
pending suffix token has no initialized host KV. At least 64 legal prefetch
slots must remain after reserving ordinary append space. Host records are
pinned BF16 D576; pool slot zero is a sentinel, while host ID zero is valid.

The adapter expands logical page IDs, invokes the official strict predictive
threshold policy using `offset[1]`, and promotes all successfully staged records
to the prepared legal pool slots. Existing finalize stamps priority/free state;
existing exact top-k, append and residual recall consume every selected KV.
The adapter preserves every score bit returned by the official bridge. KV copy
and metadata validation are exact. Predictive false positives are legal and
their real H2D bytes must be counted.

Register `_pending_prefetch_cleanup` on the lease before the official launch.
The hook retains its stage and clears only still-matching temporary h2d tags,
including host ID zero. The parent adds hook execution before existing finalize.
Failure propagates; uncertain CUDA completion retains storage/owner and disables
reuse. No automatic retry, backend switch, or request recovery.

Run GPU adapter tests on GPU3 / CPU24-31 only:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=3 \
  MAX_JOBS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  taskset -c 24-31 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  operators/deepseek_v32/indexer/tests/test_official_prefetch.py
```

Promotion requires exact KV/map/journal/statistics checks for empty, partial,
saturated and over-cap attempts; free and occupied victims; host ID zero;
noncontiguous host pages; pending suffix and padding exclusion; idempotent
cleanup and preservation of published mappings. A raw official integration
check must pass after that bridge compiles. These checks do not establish
full-model correctness, cold timing, or performance improvement; the parent
owns those independent gates and the final experiment publication.
