# C4 final checkpoint

C4a is accepted as a component optimization and frozen for combined/full-model
validation. All C4 GPU tests and probes have exited; GPU 2 returned to 1 MiB.
The complete latency/MFU objective and experiment publication remain with root.

## Final implementation and choice

`echo.logits` retains the official DeepGEMM call and its FP32 result tensor.
When invalid tail positions exist, the existing PyTorch mask scans only columns
[query_start,N). At query_start=0 it uses the original result directly. When
query_start+1>=N, it skips masking: the valid input geometry implies Q=1 and
all logical columns are visible. No native/Triton helper or third-party change
is introduced. Exact sorted deterministic FlashInfer SMALL selection is unchanged.

For row i, columns >=query_start+i+1 are invalid. The mask retains one universally
valid column at query_start to preserve aligned widths in the target chunks.
The rectangle contains Q*(N-query_start) positions and marks exactly
Q*(N-query_start)-Q*(Q+1)/2 positions invalid. Earlier columns are never visited.

The first candidate started at query_start+1. It was exact and improved long
contexts, but changed the 1024-wide first chunk to an unaligned1023-wide view,
losing PyTorch's vectorized masked-fill path. A temporary aligned-slice candidate
restored first-chunk GPU cost (10.49→8.52 us) and API median (0.07405→0.06919 ms),
with exact outputs and similar long-context cost. The final source uses that
aligned variant; the initial candidate is not the delivered implementation.

## Correctness and source identity

33 GPU indexer/selection tests passed on GPU 2, including existing actual
DeepGEMM/ECHO numerical comparisons and nine added coverage cases. Seven dirty
strided-output cases compare every backing-storage bit, including nonzero offset,
row padding and adjacent rows, then compare tied-data exact-top-k values/indices.
Two actual DeepGEMM cases place queries before the end of N and compare against
an independent FP32 oracle. Edge cases include Q1/N1, empty invalid tail,
query_start0, arbitrary nonterminal starts, and ragged N. Complete old/new
resident logits comparisons at four benchmark shapes were bit-exact.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q --tb=short operators/deepseek_v32/indexer/tests/test_echo_indexer.py operators/deepseek_v32/indexer/tests/test_selection.py
```

Ruff check and format check passed. There is one unrelated upstream FlashInfer
Arch deprecation warning. No CUDA correctness case failed. A preliminary final
probe overlapped the last tests briefly; it was discarded and rerun only after
the test session exited. The following numbers are from the isolated rerun.

- `operators/deepseek_v32/indexer/echo.py`: `30492794f10c4bd34d730c0bb926c79827095feb59e7cad35765ff0bced4c219`.
- `operators/deepseek_v32/indexer/tests/test_echo_indexer.py`: `99a39417e240e95ad4617801b6c872d095150215825faef0f712afe71c828f0d`.
- `echo.build_info()` includes echo.py in the native source identity.

## Same-GPU measurements

Run ID: `deepseek_indexer_tail_c4a_20261004_01`. GPU 2 reports `NVIDIA H200`,
capability `[9, 0]`. PyTorch `2.12.1+cu130`, CUDA `13.0`,
FlashInfer `0.6.18`, official DeepGEMM `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`. TF32 disabled.
The comparison is the complete official resident `logits` API, with only the
old/new causal-mask implementation differing. Five warmups and twenty synchronized
wall samples per API case; compilation is outside timing. Both implementations
run on the same GPU and use identical Q/K/weights/scales.

| [Q,N,query_start] | Full median ms | C4a median ms | Full mean ms | C4a mean ms |
|---|---:|---:|---:|---:|
| [1024, 1024, 0] | 0.070504 | 0.069019 | 0.071712 | 0.071395 |
| [1024, 32768, 31744] | 0.772248 | 0.582342 | 0.773238 | 0.583247 |
| [1024, 65536, 64512] | 1.481421 | 1.095868 | 1.482935 | 1.099067 |
| [128, 65664, 65536] | 0.395763 | 0.338273 | 0.396555 | 0.335072 |

The 64K case improves by about26% and the candidate case by about15% in API
median. These are operator API results, not full-model latency or MFU claims.

Separate mask-only timing uses padded physical row stride N+13, five warmups
and thirty synchronized samples. Q1024/N65536 median is0.58500→0.03435 ms;
Q128/N65664 is0.08887→0.03095 ms. Q1024/N65536/start1234, where nearly all N
remains invalid, is0.59606→0.58880 ms; this correctly shows little benefit.
Q1/N17/start16 skips the mask entirely (harness-only median0.00487 ms).

CUDA activity measurement uses contiguous physical stride N, five warmups and
ten additional profiled samples per path. It sums actual kernel durations,
separately from uninstrumented wall time:

| [Q,N,start] | Full GPU us | C4a GPU us | Kernel launches (both) |
|---|---:|---:|---:|
| [1024,1024,0] | 8.5252 | 8.5472 | 3 |
| [1024,65536,64512] | 403.9909 | 10.5984 | 3 |
| [128,65664,65536] | 51.1684 | 4.9506 | 3 |

This candidate removes unnecessary scanned data rather than combining launches.
A custom one-launch helper is deferred; complete-model evidence should determine
its priority against the remaining top-k and other launch costs.

## Reproduction and remaining acceptance

Temporary engineering probe: `/tmp/deepseek_c4a_tail_probe.py`; data:
`/tmp/deepseek_indexer_tail_c4a_20261004_01.json`. CUDA activity probe:
`/tmp/deepseek_c4a_tail_profile.py`; summary:
`/tmp/deepseek_indexer_tail_c4a_20261004_01_profile.json`; raw traces:
`/tmp/deepseek_c4a_tail_<Q>_<N>_<full|slice>_trace.json`.
The initial-slice and aligned-candidate screening records stay under /tmp only.

Root's combined validator confirmed it captured the final30492794... source
hash. No subsequent production/test changes are planned. Full H+A and16×2
trajectory, current-source profiles and publication remain required before
making end-to-end claims. No existing experiment report was replaced here.
