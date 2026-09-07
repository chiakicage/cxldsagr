# DSA

这个目录是一个围绕 **KV cache offload + sparse attention** 的实验项目。

研究目标是把 attention 路径里的 KV cache 尽量 offload 到 CPU pinned memory / host memory 上，GPU 只按 sparse indices 读取当前 token 真正需要的 KV records。这样做的核心动机不是单纯追求带宽，而是解决长上下文和高并发下 KV cache 占用 GPU 显存的问题。

项目里同时需要两类高性能算子：

- attention 前后的 projection、indexer、输出投影以及其他 GEMM 操作，所以引入并 benchmark 了 `DeepGEMM`。
- sparse attention / sparse MLA 本体，所以维护了面向 SM120 的 `sparse_mla_sm120` CUDA 扩展。

## 目录结构

```text
.
├── DeepGEMM/              # DeepSeek-AI DeepGEMM，本地用于 projection / GEMM / MQA logits 实验
├── sparse_mla_sm120/      # SM120 sparse MLA CUDA extension
├── docs/                  # 实验笔记、benchmark 结果、模型 shape 和 offload 分析
└── .venv/                 # 当前实验环境
```

重点文档：

- `docs/kv_cache_offload.md`：KV cache offload 的背景、mapped pinned memory bandwidth 和下一步实验方向。
- `docs/deepgemm_v32_5080_results.md`：RTX 5080 上 DeepGEMM V3.2 相关 GEMM / indexer benchmark。
- `docs/sparse_mla_sm120_v32_5080_results.md`：RTX 5080 上 sparse MLA decode benchmark。
- `DSA.md`：本仓库 sparse MLA prefill / decode 实现说明。

## DeepGEMM 子项目

`DeepGEMM/` 是高性能 tensor-core kernel 库，覆盖 FP8、FP4、BF16 GEMM、grouped GEMM、MoE、MQA logits 等算子。这里主要把它当作 attention 模块周边 projection 和 indexer scoring 的性能基线/候选实现：

- decode / prefill 阶段的 `WQ`、`WKV`、`WO` projection GEMM；
- head-grouped GEMM；
- MQA / paged MQA logits；
- FP8 与 FP8 x FP4 weight 路径对比。

常用结构：

```text
DeepGEMM/
├── deep_gemm/             # Python package 和 kernel headers
├── csrc/                  # Python binding、API、JIT runtime
├── tests/                 # correctness / benchmark-like tests
├── scripts/               # helper scripts
├── build.sh
├── develop.sh
└── install.sh
```

安装/开发：

```bash
cd DeepGEMM
./develop.sh
# 或
./install.sh
```

运行本项目里的 V3.2 shape benchmark：

```bash
.venv/bin/python docs/deepgemm_v32_benchmark.py
```

结果记录在 `docs/deepgemm_v32_5080_results.md`。

## sparse_mla_sm120 子项目

`sparse_mla_sm120/` 是面向 NVIDIA SM120 的 sparse MLA CUDA 扩展，当前 Python 包名是 `flash_mla_sm120`。它实现了 DeepSeek-style sparse MLA 的 prefill 和 decode forward，输入是 BF16 query、FP8 packed KV cache 和 sparse top-k indices。

主要能力：

- V3.2 KV layout：每 token 656B，包含 FP8 NoPE KV、4 个 FP32 scale、BF16 RoPE K。
- MODEL1/V4-like layout：支持 footer scales、paged block layout、extra cache、`topk_length` 等路径。
- prefill：单 kernel 扫完整 top-k，直接输出 BF16 result 和 LSE。
- decode：scheduler-driven split-KV，先写 partial output / partial LSE，再由 combine kernel 合并。
- 支持 CUDA tensor，也允许 pinned CPU tensor 作为 KV cache / extra KV cache，用于 offload 实验。

常用结构：

```text
sparse_mla_sm120/
├── flash_mla_sm120/       # 当前 Python API wrapper
├── csrc/
│   ├── binding.cpp
│   ├── model/             # KV layout traits
│   └── kernel/
│       ├── prefill/       # sparse_mla_prefill_fwd
│       ├── decode/        # split-KV decode
│       ├── combine/       # split result combine
│       └── sched/         # decode scheduler metadata
├── benchmarks/
└── tests/
```

安装：

```bash
env MAX_JOBS=1 CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
  uv pip install --python .venv/bin/python --no-build-isolation --reinstall -e sparse_mla_sm120
```

也可以在子目录里 build wheel：

```bash
cd sparse_mla_sm120
python3 setup.py bdist_wheel
pip install dist/flash_mla_sm120-*.whl
```

Python 调用示例：

```python
import torch
import flash_mla_sm120

q = torch.randn(batch, heads, 576, device="cuda", dtype=torch.bfloat16).contiguous()
kv_cache = torch.empty(num_blocks, block_size, 1, 656, device="cuda", dtype=torch.uint8)
indices = torch.randint(0, num_blocks * block_size, (batch, topk), device="cuda", dtype=torch.int32)

out, lse = flash_mla_sm120.sparse_mla_decode_fwd(q, kv_cache, indices, sm_scale=576**-0.5, d_v=512)
```

Benchmark：

```bash
.venv/bin/python sparse_mla_sm120/benchmarks/benchmark_decode_fp8.py --warmup 50 --reps 300
```

测试：

```bash
cd sparse_mla_sm120
pytest tests -v -s
```

## KV Cache Offload 实验脉络

当前 offload 方向把 V3.2 decode 的 KV cache 看成 token-granular records。V3.2 每条 record 是 656B：

```text
[0:512)    FP8 E4M3 NoPE payload
[512:528)  4 x FP32 scales
[528:656)  BF16 RoPE payload
```

GPU 可以通过 mapped pinned memory 直接读取 CPU 上的 records：

```cpp
cudaHostAlloc(&hptr, bytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&dptr, hptr, 0);
```

已有 RTX 5080 测量显示，656B record 的 mapped pinned memory 读取大约是 10-13 GB/s，远低于 HBM，但比“完全随机 4B load”的心智模型更接近可用的 sparse record streaming。更详细的分析见 `docs/kv_cache_offload.md`。

一个重要实现问题是 decode kernel 目前按 head group 处理，每组 16 个 heads。128 heads 时，同一批 selected KV records 可能被 8 个 head groups 重复 gather。若 KV 在 host memory，这个重复读取会成为 offload 路径的主导成本。

## 生成式推荐实验输入与权重

`GR/` 独立配置数据集用户热度、可读文本和请求到达时间，默认生成精确长度的固定历史与变化候选；文本可采用数据集商品标题或纯规则素材。运行方式及输出格式见 [GR/README.md](GR/README.md)。Embedding 与前两层 dense 层的本地权重选择见 [实验权重说明](model_run/deepseek_v32_two_dense.md)。

## Python 格式化

使用 Ruff，配置在 `pyproject.toml`：Python 3.12、100 字符行宽、双引号、空格缩进、LF 换行。`DeepGEMM/`、`sparse_mla_sm120/` 子项目与生成数据、模型文件排除在外。

```bash
uv run --no-sync ruff format .
uv run --no-sync ruff format --check .
```

Ruff 已加入 `dev` 依赖组；新环境先安装开发依赖。本地 VS Code 工作区已配置 Python 保存时格式化，需启用 `charliermarsh.ruff` 扩展。
