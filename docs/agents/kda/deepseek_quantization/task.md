# DeepSeek quantization: KDA task contract

Owner: `echo_audit`; integration owner: parent agent. Created 2026-10-03.

Optimize non-matrix quantization overhead on SM90, prioritizing the indexer
Q/K D128 path. Compare activation quantization with the current compiled
official DeepGEMM helper before deciding whether a replacement is warranted.
This is an operator development task; temporary diagnostics under `/tmp`
are correctness and optimization evidence, not paper experiment results.

## Contract

- Inputs: index Q BF16 `[tokens, 64, 128]` and K BF16 `[tokens, 128]`;
  support the existing FP16/FP32 reference semantics and unsupported layouts
  through an explicit unchanged reference fallback if they are not optimized.
- Outputs: contiguous E4M3FN data with the input shape, and contiguous FP32
  scales with the last dimension replaced by one.
- Preserve FP32 amax, floor `1e-4`, division by 448, the existing
  `exp2(ceil(log2(scale)))` UE8M0 rule, and `scale_fmt=None`.
  Quantized bytes and FP32 scale bits must match the independent reference.
  No relaxed numerical threshold or changed selection/cache semantics.
- CPU/reference import must not load Triton or native extensions. CUDA
  optimization is SM90 only. GPU validation and timing use physical GPU 2
  after checking availability; no concurrent work on that GPU.
- Primary token counts: 1, 17, 128, 1024, 2048, 4096; complete Q/K call timing
  at 1024 is the primary admission gate. Include zero/tiny/large values,
  exponent and FP8 rounding boundaries, non-contiguous input and empty rows.
- Activation assessment uses actual checkpoint input widths and the existing
  `operators/deepseek_v32/linear/fp8.py` compiled official helper. Do not compare
  against a historical handwritten eager baseline.

## Boundaries and ownership

Implementation belongs in `operators/deepseek_v32/indexer/quantization.py`
and nearby tests. Parent owns `models/deepseek_v32/echo_model.py` integration.
Coordinate changes to `operators/deepseek_v32/linear/fp8.py` before editing.
Hadamard/FHT integration was canceled by the user and is outside this task.
Do not mix shared-cache changes into integrated performance comparisons.

The accepted official-compute baseline is run
`20261003_echo_layers3_official_01`; its frozen implementation is
`/tmp/deepseek-official-mfu-20261003`. Never modify that baseline tree.
Existing experiment results remain until parent publishes accepted new runs.

## Admission

One candidate at a time: independent exact correctness, complete API event
and host-wall measurements, then kernel/resource profiling when available.
Record source/dependency identities, device, input seed/shape, warmups,
repetitions and run ID. A kernel-only speedup does not imply a whole-model win.
