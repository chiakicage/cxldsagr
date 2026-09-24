# SM120 算子与基准

| 路径 | 用途 | Python 导入名 |
| --- | --- | --- |
| [sparse_mla_sm120/](sparse_mla_sm120/README.md) | DeepSeek-style sparse MLA prefill / decode，FP8 packed KV | `flash_mla_sm120` |
| [benchmarks/deepgemm_v32_benchmark.py](benchmarks/deepgemm_v32_benchmark.py) | DeepSeek V3.2 张量形状下的 DeepGEMM 基准 | 直接运行脚本 |

DeepGEMM 位于 [`3rdparty/DeepGEMM/`](../../3rdparty/DeepGEMM/)，使用上游 `nv_dev`
子模块和共享的 CUTLASS / DeepJIT 依赖，准备与构建见 [第三方说明](../../3rdparty/README.md)。
DeepGEMM 2.8.0 包含 FP4 ties-to-even 舍入变化，本项目尚未 GPU 复测，历史 FP4 数值
结果需按新版本复核。`sparse_mla_sm120` 构建目标为 `sm_120a` / `sm_120f`。

从仓库根目录安装和运行：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync --group sm120
.venv/bin/python operators/sm120/benchmarks/deepgemm_v32_benchmark.py --quick \
  --output docs/deepgemm_v32_5080_results.md
.venv/bin/python operators/sm120/sparse_mla_sm120/benchmarks/benchmark_decode_fp8.py
.venv/bin/python -m pytest operators/sm120/sparse_mla_sm120/tests/test_decode.py \
  operators/sm120/sparse_mla_sm120/tests/test_prefill.py -v
```

DeepGEMM 基准需要本地 `docs/config.json` 中的 DeepSeek V3.2 attention / indexer 配置，
默认使用根目录 `.deep_gemm_cache/`，可通过 `DG_JIT_CACHE_DIR` 覆盖。
历史 RTX 5080 测量见 [`docs/`](../../docs/)。
