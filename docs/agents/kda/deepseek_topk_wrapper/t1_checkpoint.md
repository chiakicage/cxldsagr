# T1 exact ordering: correct, rejected on API latency

Status: rejected, 2026-10-04. No production source changed. GPU0 was released
after both successful processes exited. The C5 implementation remains selected.
This is a component engineering screen, outside experiment deliverables.

`topk_order_t1` kept the installed official SMALL selector and replaced output
index sorting, stable ordered-value sorting and invalid masking with one
four-warp Triton uint64 bitonic sort. The exact selected set and final ordering
passed the bounded GPU gate; the complete API was slower in all six cases.

## Correctness and runtime identity

CPU proof passed 6,144 cases covering all output widths 1..2048, full raw FP32
bit patterns, signed zero, nonfinite masks and power-of-two padding. The GPU
gate passed 41 shape/pattern cases, 27 representative K values including CUB
dispatch/power-of-two boundaries, ties exceeding K, all invalid, K>N,
noncontiguous input, input preservation, fresh output retention, nondefault
stream and four changed-input graph replays. Three full NaN stress cases also
matched; arbitrary NaN logits remain outside the production input contract.
Six isolated finalizer widths preserved NaN payload bits. BF16/FP16 rejection
was checked on CPU against the unchanged FP32 API.

`gpu_correctness_v2.json` retains the actual loaded FlashInfer top-k library and
ten live Triton specializations, each with loaded module/function handles,
retained CUBIN, PTX and metadata hashes. `gpu_screen.json` retains two live
specializations from the timing shapes. An independent rehash checked 133
compiler/source dependencies per process, 50/10 runtime artifacts, the loaded
vendor binary/build/source records and screen source files. Runtime identity
collection ran after the measured loops. No profiler was used.

The 2048-wide finalizer compiled to 122 registers, zero stack/local memory,
16,384 bytes dynamic shared plus 1,024 bytes static shared. Its PTX has no FP
arithmetic or FTZ instruction. These resource observations do not establish
which part of the sorting kernel causes the slowdown.

## Complete API screen

Device: NVIDIA H200, SM90; PyTorch 2.12.1+cu130, CUDA 13.0, Triton 3.7.1.
Each case first compared complete values/indices, then ran ten warmups and
forty repetitions per implementation in alternating order. The table reports
means in ms. API wall includes completion and all API work; CUDA events include
the API's device interval and submission gaps. Neither metric is summed kernel
activity. Compilation, warmup and post-run identity collection are excluded.
Cache-C3 CPU timing had exited before this screen began; root held other GPU work.

| Q / N | Pattern | C5 API wall | T1 API wall | C5 event | T1 event |
|---|---|---:|---:|---:|---:|
| 1024 / 1024 | Causal | 0.103204 | 0.148600 | 0.089469 | 0.135413 |
| 1024 / 32768 | Causal | 0.332736 | 0.447637 | 0.318781 | 0.434211 |
| 1024 / 65536 | Causal | 0.437071 | 0.549478 | 0.423269 | 0.536310 |
| 128 / 65664 | Causal | 0.093932 | 0.107410 | 0.080736 | 0.094432 |
| 128 / 65664 | Ties | 0.117112 | 0.132154 | 0.104017 | 0.119337 |
| 128 / 65664 | All invalid | 0.149426 | 0.164566 | 0.136328 | 0.151668 |

The unchanged selector plus this Triton finalizer loses to the C5 complete API.
Source-level launch reduction is insufficient to select it. No broad tuning,
new profiling, allocator acceptance or whole-model run is justified for this
candidate. A different exact sorting algorithm remains a separate proposal;
the resident-selection path was not modified or measured by T1.

## Evidence

All files are under `/tmp/deepseek_topk_order_t1_20261004/`:

| Artifact | SHA-256 |
|---|---|
| `candidate.py` | `7cb5c3cf0e54e44d6254aad11084d422f4998a312d5b75e59ff97368fb0732ec` |
| `screen.py` | `0d740bac803129fa2f37f63bbc6673d8f731a23c045a7e50cdabc5d632e6b4af` |
| `gpu_correctness_v2.json` | `7be052768801873d8f4d8e65aea482c7390d6f59ab9173a212b07f44c1698403` |
| `gpu_screen.json` | `74da6b2f79247bd3ddca0d9544ec08abea64781a8eacb6b05323d72aae3c3c2f` |

`runtime_screen.py` adds actual post-run provenance without changing the timed
candidate. `identity_audit.json` records rehash counts. Separate stdout/stderr
files use the same result stems. The first startup attempt failed because ninja
was absent from PATH; it was retried after prepending `.venv/bin`. That failure
did not execute a selection API and is not correctness/performance evidence.
Exact compiler paths, environment and source identities are saved in each JSON.

Existing C9 and earlier published reports remain unchanged; T1 provides no new
formal latency or matrix API MFU result for the motivation experiment.
