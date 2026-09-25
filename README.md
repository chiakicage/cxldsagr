# cxldsagr

通用 **sparse attention offloading** 实验项目：让 KV cache 驻留在 CPU / host
memory 等显存之外的位置，GPU 根据稀疏访问需求读取或搬入所需 KV，研究显存占用、
数据传输、计算与调度的关系。

当前选择 DeepSeek V3.2 和 NOSA，重点是 **NOSA + SM90 / Hopper**。
已有 DeepSeek V3.2 / SM120 的 attention decode、extend 和部分 checkpoint 层实验；
NOSA 已提供基于 FlashInfer Full Attention 的单 GPU 模型推理，以及直接消费 GR 输入的
本地串行执行框架。模型层、执行器与缓存管理已分离，当前 KV 全部驻留 HBM；
NOSA indexer、sparse attention 和 local DRAM offloading 仅预留接口。DeepGEMM 使用上游 `nv_dev`
子模块，其依赖更新尚未经过 GPU 构建和模型验证。

```text
operators/sm90/                  Hopper sparse attention + fetch 接口（未实现）
operators/sm120/                 sparse MLA 扩展与现有算子基准
layers/                         普通层与 indexer / main attention 契约
models/nosa/                     NOSA 模型结构、权重、位置编码与 KV 布局适配
models/deepseek_v32/             DeepSeek V3.2 decode/extend 与稀疏索引
executor/                       通用 prefill / extend 分块执行
cache/                          请求级 KV cache 管理，当前仅 resident 后端
serving/                        GR 输入驱动的本地串行执行
experiments/                     每个实验独立目录，含 src/scripts/tests/output
experiments/legacy/              旧 DeepSeek / SM120 等历史归档
3rdparty/DeepGEMM/               DeepGEMM nv_dev 子模块
3rdparty/cutlass/                共享 CUTLASS 子模块
3rdparty/DeepJIT/                共享 DeepJIT 子模块
GR/                             请求输入、用户热度与到达时间生成
docs/                           跨实验设计文档
weights/                         本地权重与 tokenizer
```

## 运行

先按 [第三方依赖说明](3rdparty/README.md) 准备源码。以下命令均从仓库根目录执行，
依赖版本由 `uv.lock` 固定，使用原有 CUDA 13 环境配置。

```bash
python3 scripts/prepare_3rdparty.py --init

# 基础环境
uv sync

# NOSA Full Attention（激活环境以便 FlashInfer 找到 ninja）
source .venv/bin/activate
python -m models.nosa.infer --prompt "请解释 KV cache 的作用。" --disable-thinking

# 本地 GR 请求：prefix prefill + candidate extend，输出完成摘要
python -m serving.run_gr --count 1

# DeepSeek V3.2 / SM120 实验及报告依赖
uv sync --group sm120 --group analysis

.venv/bin/python models/deepseek_v32/deepseek_v32_decode.py --quick
.venv/bin/python models/deepseek_v32/deepseek_v32_extend.py
.venv/bin/python -m experiments.legacy.deepseek_v32.sweep_gr_content_matrix
.venv/bin/python experiments/legacy/deepseek_v32/deepgemm_v32_benchmark.py --quick
```

模型实验需要对应配置、权重或 tokenizer，见 [NOSA](models/nosa/README.md) 和
[DeepSeek V3.2](models/deepseek_v32/README.md)。
构建与测试命令见 [SM120 算子](operators/sm120/README.md)。

## 提交检查

```bash
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

提交时自动执行 Ruff 检查、可安全自动修复及格式化，沿用 `pyproject.toml` 的排除目录。
若 hook 修改了文件，重新暂存后再次提交。

## 文档

- [模型索引](models/README.md)、[实验索引](experiments/README.md)、[算子索引](operators/README.md)
- [本地 serving](serving/README.md)、[模型执行器](executor/README.md)、[缓存管理](cache/README.md)、[共享层](layers/README.md)
- [GR 输入生成](GR/README.md)、[DeepSeek 实验权重](experiments/legacy/deepseek_v32/deepseek_v32_two_dense.md)
- [KV cache offload 分析](docs/kv_cache_offload.md)、[SM120 sparse MLA 实现](DSA.md)
- [RTX 5080 DeepGEMM 结果](experiments/legacy/deepseek_v32/docs/deepgemm_v32_5080_results.md)、[DeepSeek extend 分析](experiments/legacy/deepseek_v32/docs/extend_step_profile/extend_compute_zh.md)
