# 已撤回：NOSA 自回归生成试测

## 目的、内容与结论

该试测曾将 GR 输入送入固定 64-token 自回归生成循环，用于观察 NOSA 文本生成吞吐；
它不符合本项目 GR 的历史+候选前向定义，结果已撤回为 GR 性能依据。下面保留历史记录，
正式 GR 结果见 [64K+1K 实验](../../nosa_gr_65536_1024/README.md)。

## 当前运行方式与调用模块

仅用于复查旧生成试测，不作为 GR 默认入口：

```bash
bash experiments/legacy/nosa_generation/scripts/run.sh --help
bash experiments/legacy/nosa_generation/scripts/run.sh replay_001
```

脚本调用 `experiments.legacy.nosa_generation.src.measure`，使用 `GR.input_generator` 和
`models.nosa.model`。`src/` 保存该试测的测量代码；`tests/` 预留，本归档没有独立单元测试。
当时的原始结果、校正记录与源码快照保存在 `output/data/20260925/`；运行日志归
`output/log/<run_id>/`，`output/profile/` 预留且本试测没有 profiler 产物。

## 历史结果与原命令（保持原样）

> 已撤回为 GR 性能口径：本文实际测量了人为追加的 64-token 自回归生成，
> Decode tok/s、TTFT、TPOT 和生成请求耗时均不适用于本项目 GR workload。
> 仅保留原始实验记录。GR 重测与模块 MFU 见 [当前报告](../../nosa_gr_65536_1024/README.md)。

# NOSA-8B：GR 单请求性能（2026-09-25）

当前 `models/nosa` 的 Full Attention 基线在单张 SM90 GPU 上，GR 默认 12 组长度的
预热后 prefill 中位数为 **145.8–806.7 ms**，decode 为 **70.53–71.82 tokens/s**
（**13.92–14.18 ms/token**）。本次完整执行 12 次预热和 60 次正式测量。

这是 NOSA 权重的 dense causal GQA 推理；`A` / `delta` 未启用，未测量 NOSA sparse、
KV offloading、跨请求 prefix cache 或批量 serving。

## 环境与输入

- GPU 0：`nvidia-smi` 名称 NVIDIA M403；PyTorch/CUDA 名称 NVIDIA H200，SM90，
  驱动 570.124.06，CUDA 13.2。测量开始时两张 GPU 均无其他计算进程，本实验只使用 GPU 0。
- Python 3.12.12、PyTorch 2.10.0+cu132、FlashInfer 0.6.18、tokenizers 0.23.2、
  safetensors 0.8.0。沿用已有环境，**没有验证锁文件中的 PyTorch 2.12.1+cu130 环境**。
- checkpoint：`/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`；32 层，hidden 4096，32 Q heads / 2 KV heads，
  head dim 128，BF16。实际模型参数占 15.246 GiB，本次权重加载耗时 4.80 s。
- 模型代码基于 `dc8e85d2d79c7cbdd179eea3a80c52f33b71505a`，新增测量脚本；运行时源码
  与 checkpoint 配置/tokenizer 的 SHA-256 已保存到 `metadata.json`。
- 使用 `GR.input_generator.create_input_generator(model="nosa")`；Beauty 热度曲线、
  1000 个合成用户、synthetic 商品文本、weighted 抽样、seed 42、默认关闭 thinking 的模板。
- 每种长度分别创建固定长度的 GR 流，各取 6 条请求，首条用于预热。直接消费完整 `input_ids`，
  不再次套模板。`item_tokens` 包括候选之外的固定指令与模板开销；输入总长为 history + item。
- 记录 GR 的 Poisson 到达时间戳（配置 QPS 100），但顺序执行请求，不按时间戳回放。
  因此此处不测队列延迟、热度带来的缓存收益或服务可承载 QPS。

## 测量口径

batch size 1，prefill chunk size 1024，每条请求新建 KV cache；每种长度预热 1 次、正式测量
5 次，表中使用中位数。缓存覆盖整个输入及 64-token 输出预算，全部保存在 GPU。

计时采用 CUDA 同步后的墙钟时间，包含 Python 调度、模型前向和 greedy 采样；不含 GR 文本生成、
tokenization、输入 H2D、cache 分配和权重加载。prefill 为所有 chunk 的模型前向；TTFT 额外
包含首 token 的 argmax 和取回；TPOT 为其后 63 次 decode 的总耗时除以 63。
表中的请求时间从 prefill 开始到第 64 个 token 生成结束，不是完整服务端到端延迟。

固定输出 64 tokens，**忽略 EOS**，以避免不同推荐文本长度影响 decode 比较。60 条正式请求中
59 条在 64 tokens 内出现 EOS；真实应用遇 EOS 会提前停止，因此不能把固定 64-token
请求耗时当成实际推荐请求耗时，也不评价推荐质量。所有生成 token IDs 与 EOS 前文本均保留。

首次预热 prefill 为 1739.7 ms；正式测量已排除该初始化开销。所有长度在同一进程内测量，
权重只加载一次。12 组共 72 条请求的 GR 初始化/生成总耗时 17.81 s，单独记录，不计入模型性能。

## 结果

| History | Item 预算 | 输入 tokens | Prefill ms | TTFT ms | Decode tok/s | TPOT ms | 请求 ms | 峰值 allocated GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 128 | 4224 | 146.1 | 146.1 | 71.16 | 14.05 | 1031.5 | 15.529 |
| 4096 | 256 | 4352 | 145.8 | 145.9 | 71.32 | 14.02 | 1028.9 | 15.533 |
| 4096 | 512 | 4608 | 152.9 | 153.0 | 70.82 | 14.12 | 1043.8 | 15.540 |
| 4096 | 1024 | 5120 | 167.3 | 167.4 | 71.27 | 14.03 | 1049.9 | 15.556 |
| 4096 | 2048 | 6144 | 204.2 | 204.3 | 71.26 | 14.03 | 1088.9 | 15.587 |
| 4096 | 4096 | 8192 | 280.8 | 280.8 | 70.53 | 14.18 | 1174.0 | 15.650 |
| 16384 | 128 | 16512 | 634.6 | 634.6 | 71.37 | 14.01 | 1516.4 | 15.904 |
| 16384 | 256 | 16640 | 636.8 | 636.8 | 71.59 | 13.97 | 1516.9 | 15.908 |
| 16384 | 512 | 16896 | 644.2 | 644.3 | 71.82 | 13.92 | 1521.1 | 15.915 |
| 16384 | 1024 | 17408 | 664.0 | 664.0 | 71.65 | 13.96 | 1542.3 | 15.931 |
| 16384 | 2048 | 18432 | 712.1 | 712.2 | 71.29 | 14.03 | 1595.9 | 15.962 |
| 16384 | 4096 | 20480 | 806.7 | 806.8 | 71.31 | 14.02 | 1690.4 | 16.025 |

峰值 allocated 包含权重、cache、激活和 PyTorch 跟踪的 workspace；不是进程占用的全部 GPU
内存。allocator reserved 及每次测量的原始值另存，reserved 会受同一进程此前分配影响。
BF16 KV 每 token 为 32 KiB，本次 cache 容量从 134 MiB 到 642 MiB（包含输出预算）。

从 4224 到 20480 tokens，prefill 明显增长，decode 的中位数变化较小。这个结果仅适用于
当前 eager Python + FlashInfer 单请求实现；没有通过 profiler 归因瓶颈，也没有测试 CUDA Graph、
其他 chunk size、FP16、并发请求或 32768-token 上限。每组仅 5 次正式测量，不估计尾延迟。

## 复现与产物

在仓库根目录执行，输出目录须不存在：

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m experiments.measure_nosa_gr \
  --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B --device cuda:0 \
  --user-lengths 4096 16384 --item-lengths 128 256 512 1024 2048 4096 \
  --prefill-chunk-size 1024 --output-tokens 64 --warmup 1 --repeats 5 --seed 42 \
  --output-dir GR/generated/nosa_perf_run
```

脚本：[measure_nosa_gr.py](src/measure.py)。本次实际输出目录为
`GR/generated/nosa_perf_20260925/`（Git 忽略）：

- `requests.jsonl`：72 条完整 GR 请求、调度字段、输入 token IDs 和预热标记。
- `measurements.jsonl`：逐请求计时、显存、输出 IDs、首个 EOS 位置及 EOS 前文本。
- `summary.json`：12 组的 median / mean / min / max。
- `metadata.json`：环境、命令参数、源码摘要、开始/结束时间及测量语义。
- `benchmark_source.py`、`measurements.raw.jsonl`：运行时脚本和原始结果存档。
  首轮脚本在计时结束后的 EOS 元数据处理遗漏了 tuple 类型；已从保存的输出 IDs 修正
  `first_eos_index` 和 `text_before_eos`，正式脚本也已修正。所有计时、输出 IDs 与性能汇总未改变，
  修正来源记录在 metadata 的 `postprocessing` 字段。

## 验证

性能测量结束后运行现有测试，避免 GPU 测试干扰计时：

```bash
PATH="$PWD/.venv/bin:$PATH" NOSA_MODEL_PATH=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  .venv/bin/python -m pytest models/nosa/tests GR/tests -q -rs
```

结果：**41 passed、1 skipped、34 subtests passed**；跳过的是本地缺少 tokenizer 的
DeepSeek GR 测试，NOSA GPU 数值参考测试实际执行并通过。
新增脚本的 Ruff 检查、格式检查和模块 `--help` 通过。72 条原始请求与测量记录逐条核对，
输入长度、64-token 输出、63 次 decode、计时分段以及 12 组中位数汇总一致。
