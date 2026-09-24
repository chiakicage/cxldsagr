# Extend step profile

RTX 5080, 2026-09-11. No-Hadamard, V3.2 FP8 QK synthetic attention, new tokens=1024, chunk=1024.

GPU times below are exclusive and summed across chunks. Event times in summary.json are inclusive (parents include children); do not sum them. E2E is measured separately without instrumentation. CUPTI kernel times are from one warmed iteration.

## History 65536

E2E: 17.472 ms; attributed GPU: 17.306 ms.

| Step | Calls | GPU ms (exclusive) |
|---|---:|---:|
| wq_a/quantize | 1 | 0.0183 |
| wq_a/gemm | 1 | 0.1013 |
| wq_a | 1 | 0.0000 |
| q_norm | 1 | 0.0032 |
| wq_b/quantize | 1 | 0.0028 |
| wq_b/gemm | 1 | 0.2240 |
| wq_b | 1 | 0.0000 |
| wkv_a/quantize | 1 | 0.0190 |
| wkv_a/gemm | 1 | 0.0386 |
| wkv_a | 1 | 0.0000 |
| kv_norm | 1 | 0.0041 |
| mla/rope | 1 | 0.0524 |
| wk_b/quantize | 1 | 0.0235 |
| wk_b/gemm | 1 | 0.1988 |
| wk_b | 1 | 0.0007 |
| index_wqi/quantize | 1 | 0.0054 |
| index_wqi/gemm | 1 | 0.0857 |
| index_wqi | 1 | 0.0000 |
| index_wki/quantize | 1 | 0.0203 |
| index_wki/gemm | 1 | 0.0189 |
| index_wki | 1 | 0.0000 |
| index_norm | 1 | 0.0016 |
| index/rope | 1 | 0.0189 |
| index_weights | 1 | 0.0131 |
| projection/layout | 1 | 0.9332 |
| cache/append | 1 | 0.0044 |
| index/q_quantize | 1 | 0.2314 |
| index/mqa_logits | 1 | 5.4191 |
| index/weights_bounds | 1 | 0.0060 |
| index/topk | 1 | 0.7434 |
| mla/prefill | 1 | 7.7205 |
| wv_b/quantize | 1 | 0.2207 |
| wv_b/gemm | 1 | 0.1152 |
| wv_b | 1 | 0.0008 |
| wv_b/layout | 1 | 0.3901 |
| wo/quantize | 1 | 0.0222 |
| wo/gemm | 1 | 0.6378 |
| wo | 1 | 0.0000 |
| wo/layout | 1 | 0.0000 |
| loop/output_copy | 1 | 0.0110 |

## Reproduce

Run from the repository root. The configuration is taken from the revision used by the existing experiment; it has been removed from the current working tree.

```bash
mkdir -p docs/extend_step_profile/no_hadamard_64k_1k
git show d2fab677c88de4995b5dfcda2b02eca38eca12a6:docs/config.json > docs/extend_step_profile/no_hadamard_64k_1k/config.json
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m model_run.deepseek_v32.profile_deepseek_v32_extend \
  --config docs/extend_step_profile/no_hadamard_64k_1k/config.json \
  --history-lens 65536 --new-tokens 1024 --chunk-size 1024 --iters 5 \
  --output-dir docs/extend_step_profile/no_hadamard_64k_1k
.venv/bin/python -m model_run.deepseek_v32.render_extend_timeline \
  docs/extend_step_profile/no_hadamard_64k_1k/summary.json \
  docs/extend_step_profile/extend_timeline_64k_1k.svg
```

The output finite-value assertion passed. No new numerical accuracy comparison was performed. Raw JSON/trace/configuration files are local and Git-ignored.
