# DeepSeek FP8 linear MFU investigation

Status: the local tuned-kernel candidate below was superseded on 2026-10-03 by
an adapter around official DeepGEMM `main`, commit
`057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`. The installed module reports version
`2.8.1`. Production source is frozen at `linear/fp8.py` SHA256
`180a2fd35c1a1b61aec00382c415f7e04b738139c29d5cc099394cfb01b9d63b`.
No local GPU quantization or GEMM kernel remains, and the temporary local
`linear/config.py` was removed.

The current model benchmark uses real checkpoint layers 0–2 sequentially with
embedding, final norm, and last-token LM head, preserving the existing request
input generator. Complete-model measurements and publication are owned by the
main MFU task. Operator diagnostics here do not replace the three-layer model
correctness and performance run. Existing experiment publications remain until
replacement model runs pass and are published.

## Current adapter

- Dense: official `deep_gemm.fp8_gemm_nt` with recipe `(1, 128, 128)`.
- Grouped: official `deep_gemm.m_grouped_fp8_gemm_nt_contiguous` with the same
  recipe. GPU routing pads each expert to the official contiguous alignment,
  marks unused rows with group `-1`, and restores original route order. A static
  upper bound avoids reading expert counts back to the CPU. Reusable routing,
  token-shared input, route-specific input, invalid routes, and empty input are
  retained.
- Quantization: `torch.compile` compiles a closure calling upstream
  `per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128,
  use_packed_ue8m0=False)`. It contains no copied arithmetic. Compile options are
  `fullgraph=True`, `dynamic=True`, and `options={"triton.cudagraphs": False}`.
  Literal format arguments keep `gran_k` constant while tensor shapes are
  dynamic. Compiling the helper directly with runtime keyword arguments made
  `gran_k` symbolic and produced 67.9 us quantization for M1024/K7168; the fixed
  wrapper takes 5.95 us with bitwise-equal data and scales. `functools.partial`
  did not remove the symbolic block size. This isolation is recorded under
  `/tmp/deepseek-linear-quant-isolate-20261003/`.
- Adapter padding, route packing, scale layout conversion, and inverse mapping
  are inside the public operator call. Actual checkpoint dense weights already
  satisfy alignment; small tests exercise partial N/K padding. FP8 padding
  preserves raw bits through a uint8 view.
- Original checkpoint FP32 weight scales are preserved. The independent CPU
  quantization/dequantization reference does not import DeepGEMM.
- Useful GEMM FLOPs remain exact. Native executed/padded FLOPs are unknown until
  read from the upstream launch; the removed local launch geometry must not be
  used for official kernels. The profiler identifies upstream
  `sm90_fp8_gemm_1d2d_impl`; projection attribution remains in NVTX.

## Current verification and operator diagnostics

Physical GPU 3: NVIDIA H200 / SM90; PyTorch 2.12.1+cu130, Triton 3.7.1.

`CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest -q
operators/deepseek_v32/linear/tests/test_deepseek_linear.py`: **25 passed** in
13.30 s. The suite covers exact quantization, independent dense/grouped oracles,
noncontiguous input, partial M/N/K, real index_k geometry, multi-tile per-expert
padding, reused routing, invalid/all-invalid routes, and empty input.
Ruff and `git diff --check` pass. Test stdout is
`/tmp/deepseek-linear-main-fixed-pytest.stdout`; no test results were added to
`experiments/` or test source directories.

The final A/B compares the complete `fp8_linear` API against original revision
`1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c`, using equal reused tensors: random
BF16 activations / FP8 weights at actual checkpoint shapes, seed 9031. Each CUDA
graph contains 10 calls; 11 event samples follow three warmups. Graph replay
includes compiled official quantization, layout adaptation, and GEMM; host
submission and allocator work are excluded. Clocks were not locked. These are
operator diagnostics and make no model latency/MFU claim.

| Projection | Original us | Official DeepGEMM us | Speedup |
| --- | ---: | ---: | ---: |
| index_k | 49.53 | 16.72 | 2.96x |
| kv_a | 50.74 | 24.32 | 2.09x |
| q_a | 68.26 | 26.55 | 2.57x |
| index_q | 54.63 | 25.85 | 2.11x |
| q_b | 166.52 | 65.93 | 2.53x |
| o | 564.51 | 210.34 | 2.68x |
| dense gate | 540.27 | 232.15 | 2.33x |
| dense down | 622.50 | 236.63 | 2.63x |

All eight outputs are bitwise equal to the original implementation. Independent
FP32-oracle relative L2 is 0.0006785–0.0006904 and passes the numerical tolerance.
The frozen source above was used for this A/B without subsequent source edits.

Evidence: `/tmp/deepseek-linear-main-fixed-20261003/compare.py`,
`full_operator_ab.json`, `stdout.log`, `stderr.log`, and
`inductor_source_manifest.json` in the same directory. The JSON records every
event sample, input shape, compile options, library versions, and SHA256 of the
adapter, upstream math helper, and original source. The generated-source
manifest hashes the two live TorchInductor Python artifacts. Formal model runs
must collect their own generated artifacts and source snapshot.

## Superseded local-kernel investigation

The remaining material describes a removed intermediate candidate. Its numbers,
launch geometry, split-K, and old 61-layer task framing do not describe the
current adapter or current three-layer benchmark. It is retained only as an
internal diagnostic history; it is not an experiment publication.

### Superseded candidate: evidence and diagnosis

Baseline revision: `1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c`.
Baseline `operators/deepseek_v32/linear/fp8.py` SHA256:
`2ad05824f2cc3110524b8b34e2cf25174180ea614dee5dca96037427efb6e61b`.

Physical GPU 2 was exclusively used by this subtask: NVIDIA M403, SM90, 132 SMs.
PyTorch 2.12.1+cu130, Triton 3.7.1, Nsight Compute 2026.1.1; clocks were not locked.
Original generated PTX already uses WGMMA and `cp.async`. Lack of Tensor Core
instructions is therefore not the problem.

NCU full + PM sampling and separate SourceCounters show:

| Shape M/N/K | Baseline grid | Baseline registers/thread | Baseline duration | Tensor active |
| --- | ---: | ---: | ---: | ---: |
| 1024 / 128 / 7168 (index_k) | 16 | 224 | 61.568 us | 1.778% |
| 1024 / 18432 / 7168 (dense gate) | 2304 | 224 | 529.536 us | 26.597% |

The narrow output leaves most SMs idle. The gate has enough blocks, but its
original two-stage pipeline waits on activation loads: the original source line
62 has 25,014 long-scoreboard samples, and the scale-load line 70 has 10,414
barrier samples. NCU reports 45.59% of average issue interval as long scoreboard
for gate. These are diagnostics of the sampled kernels, not end-to-end cost
components or predicted speedups.

### Superseded candidate: change and boundaries

- A pure Python `linear/config.py` exposes `fp8_linear_config` for both execution
  and FLOPs accounting. No CUDA or Triton import is required to read geometry.
- Ordinary dense GEMMs retain increasing-K FP32 accumulation. The measured
  tile/stage choices use 3 or 4 stages and N=64 for intermediate-width outputs.
- N <= 128 with K >= 1024 uses up to 16 K splits. M1024/N128/K7168 uses 8 splits,
  256 CTAs, FP32 partial output and a separate FP32 reduction. This changes the
  FP32 summation order before the single final BF16 conversion.
- Activation quantization processes 32 independent 128-channel blocks per CTA,
  previously 4. Quantized values and scales are bitwise unchanged.
- Grouped GEMM geometry and routing are unchanged; grouped experts share the
  faster activation quantization and therefore require affected model retesting.
- Checkpoint scaling, UE8M0 rounding, and CPU reference arithmetic are unchanged.
  No model projection fusion or new dependency is introduced.

For split-K, executed K work is
`ceil(ceil(K / 128) / split_k) * split_k * 128`. Useful FLOPs remain `2*M*N*K`.
Quantization and split reduction are scalar work excluded from FLOPs, included
in full operator duration. MFU instrumentation was coordinated with its owner.

### Superseded candidate: verification and measurements

`CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q
operators/deepseek_v32/linear/tests/test_deepseek_linear.py`: **24 passed**.
Coverage includes CPU oracles, exact quantization, ordinary and split dense
GEMM, partial M/N/K, actual checkpoint index_k geometry, grouped token/route
inputs, reused routing, invalid/empty expert routes and empty token inputs.
Ruff check passed for the changed operator files.

The final independent A/B uses the complete `fp8_linear` API, including
quantization, GEMM, and any split reduction. Each CUDA graph contains 10 calls;
11 event samples are collected after three warmups. Inputs are reused random
BF16 activations / FP8 weights with actual projection shapes, seed 9031. Host
submission and allocator work are outside graph replay, so this is not a model
latency measurement. Both implementations use the same tensors in each A/B.

| Projection | Original us | Candidate us | Speedup |
| --- | ---: | ---: | ---: |
| index_k | 49.56 | 13.00 | 3.81x |
| kv_a | 50.42 | 32.35 | 1.56x |
| q_a | 68.39 | 45.58 | 1.50x |
| index_q | 54.81 | 46.57 | 1.18x |
| q_b | 166.88 | 133.15 | 1.25x |
| o | 564.57 | 391.58 | 1.44x |
| dense gate | 539.74 | 402.41 | 1.34x |
| dense down | 620.04 | 457.57 | 1.36x |

All ordinary projections are bitwise equal to the original implementation.
Index_k old/new NRMSE is `1.09583e-5`; all candidate/independent FP32-oracle
NRMSE values are 0.000678–0.000691 and pass the existing tolerance.

The optimized NCU replay gives index_k 256 CTAs / 121 registers / 8.096 us,
and gate 402.624 us / 35.536% Tensor active. NCU duration covers only the
selected GEMM, excluding quantization and the split reduction. Baseline and
candidate NCU use separate random draws at identical shapes; the full A/B
above, not those profiler timings, compares the complete operator on equal
inputs. Reports retain their source text and metric names via `ncu_report`.

Temporary evidence is in `/tmp/deepseek-linear-20261002/`: `tune.py`,
`tile_sweep.json`, `split_sweep.py`, `split_sweep.json`, `quant_sweep.py`,
`quant_sweep.json`, `compare.py`, `baseline_fp8.py`, `full_operator_ab.json`,
`baseline_{full,source}.ncu-rep`, `optimized_{full,source}.ncu-rep`,
`ncu_analysis.json`, `optimized_ncu_analysis.json`, and stdout/stderr.
The A/B JSON records source SHA256 and each event sample. The final source
adds explanatory comments after this measurement; executable code is unchanged.
Failed or discarded tuning artifacts were never added to `experiments/`.
