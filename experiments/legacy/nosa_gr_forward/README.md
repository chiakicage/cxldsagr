# 已撤回：NOSA 多长度 GR 前向性能试测

## 目的、归档原因与当前状态

该试测曾比较 GR 默认 12 组长度的 full prefill / prefix prefill / candidate extend 与模块 MFU。
主工作区已经撤回此多长度实验；本次合并沿用该决定，仅保留 framework worktree 的历史源码与报告。
当前有效 GR 性能入口为 [instruction + 历史 65536，候选 1024](../../nosa_gr_65536_1024/README.md)。
下面的多长度结果不恢复为当前有效结果，也不能代替精确 65536 + 1024 的测量。

## 当前运行方式与调用模块

仅用于复查归档，不作为 GR 默认入口：

```bash
bash experiments/legacy/nosa_gr_forward/scripts/run.sh --help
bash experiments/legacy/nosa_gr_forward/scripts/run.sh replay_001
.venv/bin/python -m pytest experiments/legacy/nosa_gr_forward/tests -q
```

脚本先调用 `src/measure.py`，再调用 `src/profile.py`；两者显式复用
`executor.model_executor.run_chunks` 和当前 64K+1K 实验的 `src/sources.py`，profile 另复用其
`src/instrumentation.py`。测量调用 `GR.input_generator` 与 `models.nosa.model`，测试覆盖
Chrome trace 活动唯一归因、未归因报错和 MFU 聚合。新运行拒绝覆盖 run ID；stdout/stderr
分别存入 `output/log/<run_id>/`，计时/汇总放 `output/data/<run_id>/`，Chrome trace 放
`output/profile/<run_id>/`。GPU 重放尚未在本次合并后的代码上运行。

历史 run ID 为 `20260925`。保留的报告、CSV 与源码快照在 `output/data/20260925/`，迁移记录为
`output/data/migration.json`。framework worktree 没有历史 GR/generated 原始请求或 trace，
因此没有补造这些数据；下面的历史路径仅记录当时运行位置。
错误地执行 65564 prefix + 996 suffix 的旧 `profile_nosa_gr_nsys.py` 只作为该目录下的
源码快照保留，不提供运行入口。生成旧报告副本也仅保存为复现数据，现有
[生成归档](../nosa_generation/README.md) 保持原有历史内容。

## 历史报告（测量含义与原命令保留）

# NOSA：GR 前向性能与模块 MFU（2026-09-25）

按 GR workload 重测完成：**固定历史 + 当前候选区块的 backbone 前向，不做自回归生成**。
12 组长度的完整 prefill 为 **148.4–828.0 ms**、整段 MFU **46.8–51.4%**；
历史 KV 已就绪时，候选 extend 为 **15.3–189.6 ms**、整段 MFU **11.5–54.3%**。

此前把 GR 输入接到文本生成循环，人为追加 64 tokens，口径错误；Decode tok/s、TTFT、TPOT
和生成请求耗时均不能代表这里的 GR 性能。原始记录保存在
[已撤回的生成实验](../nosa_generation/README.md)，不用于下列数据。

## 平台与执行语义

- GPU 0：`nvidia-smi` 显示 NVIDIA M403，CUDA 显示 NVIDIA H200，132 SM，SM90，
  700 W power limit，最大 SM 时钟 1980 MHz；驱动 570.124.06。
- Python 3.12.12、PyTorch 2.10.0+cu132 / CUDA 13.2、FlashInfer 0.6.18、
  tokenizers 0.23.2、safetensors 0.8.0。使用已有环境，未验证锁定的 PyTorch 2.12.1+cu130。
- checkpoint：`/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`，全部 32 层，BF16；hidden 4096，
  MLP intermediate 16384，32 Q heads / 2 KV heads，head dimension 128。
- 当前 cxldsagr NOSA 仍是 FlashInfer **Full Attention**，未启用 `A` / `delta`、
  sparse selection 或 offloading，不代表原 NOSA sparse kernel 性能。
- GR Beauty 热度曲线、1000 合成用户、synthetic 商品文本、weighted 抽样、seed 42；
  每种长度取 6 条请求，第 1 条预热，其余 5 条正式测量，完整输入 IDs 原样保存。
- 使用新增的 `return_hidden=True`：返回输入 token 的最终 normalized hidden states，
  **不执行 LM head、采样、生成循环或推荐任务的后处理**。完整权重仍加载，LM head 权重只占内存。

KV 分界与 `cxl-recsys` 的 GR adapter 一致：固定指令占 28 tokens，
`prefix = user_tokens + 28`，`candidate = item_tokens - 28`。
item 是“除用户历史外的总预算”，不是候选实际长度。

测量三条路径：full prefill 从空 KV 处理完整输入；prefix prefill 从空 KV 构建固定前缀；
candidate extend 在该 prefix KV 已就绪时处理整个候选后缀。所有路径都按 1024 tokens 分块。
本次没有跨请求缓存调度、命中率测试、CXL 传输或 Poisson 到达回放；不是服务 QPS 测试。

## 前向延迟与整段 MFU

同步墙钟计时，包含 Python 调度、算子启动和 GPU 执行；不含 GR 生成/tokenization、输入 H2D、
cache 分配、模型加载与输出验证。下表均为 **5 次测量的中位数**，没有 profiler 开销。
三个阶段分别直接计时；prefix 和 full 的最后一个 chunk 形状不同，因此不要用二者耗时相减估算 extend。

MFU 分母统一采用 **989 TFLOPS：H200 SXM 标称 BF16 dense Tensor Core 参考值**。
本机名称存在 M403/H200 差异，该峰值不是本机实测，也未按 1980 MHz 最大时钟放大；
不使用 2:4 sparse 的 1979 TFLOPS。所有 MFU 都是相对此明确参考值的折算，脚本可用
`--peak-tflops` 更换分母；有效 TFLOPS 保存在 CSV 中，不依赖分母选择。

| 用户历史 | Item 预算 | 候选实际 tokens | Full ms | Full MFU | Prefix ms | Extend ms | Extend MFU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 128 | 100 | 148.7 | 46.8% | 149.4 | 15.3 | 11.5% |
| 4096 | 256 | 228 | 148.4 | 48.4% | 149.7 | 15.4 | 26.0% |
| 4096 | 512 | 484 | 154.4 | 49.4% | 150.5 | 20.2 | 42.2% |
| 4096 | 1024 | 996 | 171.3 | 49.9% | 152.3 | 35.1 | 50.5% |
| 4096 | 2048 | 2020 | 209.4 | 49.8% | 151.4 | 71.4 | 51.1% |
| 4096 | 4096 | 4068 | 287.4 | 49.9% | 153.1 | 147.7 | 51.2% |
| 16384 | 128 | 100 | 639.3 | 50.9% | 647.4 | 16.7 | 14.4% |
| 16384 | 256 | 228 | 643.6 | 51.1% | 647.1 | 18.8 | 29.2% |
| 16384 | 512 | 484 | 654.2 | 51.2% | 648.4 | 24.9 | 46.9% |
| 16384 | 1024 | 996 | 675.7 | 51.4% | 648.6 | 44.5 | 54.3% |
| 16384 | 2048 | 2020 | 725.9 | 51.3% | 649.4 | 92.2 | 53.8% |
| 16384 | 4096 | 4068 | 828.0 | 51.4% | 649.1 | 189.6 | 53.9% |

## 各模块 MFU

下列模块利用率使用 **模块 GPU kernel 时间之和**，不是 CPU scope 时长。
每组回放正式测量的第 1 条请求，完整预热后采集一次 profile，汇总全部 32 层和全部 chunk；
不是 5 次 profiler 平均。整段 MFU 使用上表独立测得的墙钟时间，包含启动间隙等开销，
因此与各模块 MFU 不同，不能取模块百分比的算术平均得到整段 MFU。

完整 12 组 × 3 阶段 × 18 模块的 FLOPs、GPU 时间、TFLOPS、MFU 和活动数量见
`output/data/20260925/nosa_gr_module_mfu.csv`。这里先对比最短与最长请求：

| 模块 | Full 4224 MFU | Extend 100 / prefix 4124 MFU | Full 20480 MFU | Extend 4068 / prefix 16412 MFU |
| --- | ---: | ---: | ---: | ---: |
| Q projection | 75.56% | 25.44% | 73.85% | 74.44% |
| K projection | 19.11% | 2.85% | 21.04% | 20.84% |
| V projection | 19.75% | 3.10% | 21.94% | 21.74% |
| Attention QK + softmax + AV | 46.07% | 13.28% | 62.89% | 66.82% |
| O projection | 74.47% | 26.89% | 73.38% | 74.50% |
| MLP gate | 75.30% | 35.50% | 74.74% | 75.66% |
| MLP up | 78.96% | 36.72% | 77.28% | 77.65% |
| MLP down | 81.58% | 31.81% | 80.51% | 80.92% |

Attention core 的分子只计有效 QK/AV 矩阵 FLOPs，分母包括融合 softmax 的全部 kernel 时间。
KV heads 少使 K/V 投影较窄，短候选下这两项折算利用率尤其低。

以下给出最大请求（prefix 16412 + candidate 4068）的全部模块时间。非矩阵模块以 “—” 表示
**BF16 Tensor Core MFU 不适用**，不是漏测或 0%；其 GPU 时间仍计入整段执行开销。

| 模块 | Full GPU ms | Full MFU | Candidate extend GPU ms | Extend MFU |
| --- | ---: | ---: | ---: | ---: |
| embedding | 0.088 | — | 0.017 | — |
| rope_prepare | 0.493 | — | 0.098 | — |
| input_layernorm | 47.112 | — | 9.335 | — |
| q_proj | 30.106 | 73.85% | 5.933 | 74.44% |
| k_proj | 6.605 | 21.04% | 1.324 | 20.84% |
| v_proj | 6.335 | 21.94% | 1.270 | 21.74% |
| rope_apply | 43.902 | — | 8.643 | — |
| kv_cache_and_layout | 1.908 | — | 0.379 | — |
| attention_core | 176.776 | 62.89% | 59.537 | 66.82% |
| o_proj | 30.300 | 73.38% | 5.928 | 74.50% |
| residual | 7.797 | — | 1.536 | — |
| post_attention_layernorm | 47.389 | — | 9.357 | — |
| gate_proj | 118.994 | 74.74% | 23.348 | 75.66% |
| up_proj | 115.080 | 77.28% | 22.751 | 77.65% |
| swiglu_elementwise | 26.918 | — | 5.313 | — |
| down_proj | 110.473 | 80.51% | 21.831 | 80.92% |
| final_norm | 1.472 | — | 0.293 | — |
| model_misc | 0.016 | — | 0.003 | — |

`kv_cache_and_layout` 是 attention 内投影、RoPE、attention core、O projection 之外的剩余
GPU 活动，主要为 K/V 写入；`residual` 是 decoder layer 中子模块之外的残差加法；
`swiglu_elementwise` 为 SiLU 与逐元素乘法。上述行互斥，不重复计父/子模块。

## FLOPs 与归属校验

FMA 计 2 FLOPs，按实际调用形状计算，不按 2 × 全部 checkpoint 参数简单估算：

- 线性层 `X[T,K] × W[K,N]`：`2*T*K*N`。包括 Q/K/V/O 与 gate/up/down，排除未执行的 LM head。
- 某 chunk 有 `q` 个 query，之前有 `p` 个 KV token：因果可见对数
  `q*p + q*(q+1)/2`；attention QK + AV 为 `4*Hq*D*可见对数`。
  GQA 按 Q head 数 32 计 attention，K/V 投影按 KV head 数 2 计。
- 全段有效 FLOPs 为所有线性层与 attention 矩阵 FLOPs 之和。每组分别用独立闭式公式核对，
  并检查 `FLOPs(full) = FLOPs(prefix) + FLOPs(extend)`。
- 忽略 norm、RoPE、SiLU、softmax、残差等少量非矩阵 FLOPs；不计 causal mask 外的无效区域，
  不计 kernel 内 tile padding/recompute。所以这是有效模型 FLOPs 折算，不是实际执行指令计数。
- `TFLOPS = FLOPs / GPU微秒 / 1e6`；模块 `MFU = TFLOPS / 989`。
  整段 `MFU = FLOPs / 墙钟秒 / (989*1e12)`。
- 用 CUPTI correlation ID 将每个 kernel/memcpy/memset 归到启动所在的最内层模块 scope，
  GPU 活动各计一次。PyTorch 2.10 的 `FunctionEvent.kernels` 可能在父子节点重复出现同一 kernel，
  因而直接解析 Chrome trace；未归属活动会报错，不静默丢弃。

## 数值与测试

全部 72 条输入/测量记录的长度、KV 分界、有限输出和中位数汇总通过校验；全部 12 份 profile
的 FLOPs、GPU 活动归属与“不执行 LM head”通过核对。真实权重下完整 prefill 与拆分 prefix/extend
的最后 token hidden state 因 BF16 和 chunk 边界不同并非逐位一致：
余弦相似度最低 **0.999521**，最大绝对差最高 **12.0**。
这些记录用于量化差异，不作为推荐质量或整模型严格数值等价证明。
新增 CPU 测试检查 features 经 LM head 后与原 logits 一致、GR 路径不调用 LM head、KV extend 一致；
GPU 测试将 candidate hidden states 与独立 dense attention 参考对照。
本轮相关测试结果：**45 passed、1 skipped、34 subtests passed**；仅跳过本地缺少 tokenizer
的 DeepSeek GR 测试。GPU 数值测试实际执行通过；Ruff 与格式检查通过。

## 复现与产物

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m experiments.measure_nosa_gr \
  --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B --device cuda:0 \
  --user-lengths 4096 16384 --item-lengths 128 256 512 1024 2048 4096 \
  --prefill-chunk-size 1024 --warmup 1 --repeats 5 --seed 42 \
  --output-dir GR/generated/nosa_gr_forward_run
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m experiments.profile_nosa_gr \
  --input-dir GR/generated/nosa_gr_forward_run --peak-tflops 989 \
  --output-dir GR/generated/nosa_gr_mfu_run
```

脚本：[前向测量](src/measure.py)、[模块 profile](src/profile.py)。输出目录须不存在。
本次原始产物（Git 忽略）保存在：

- `GR/generated/nosa_gr_forward_20260925/`：完整请求、逐次计时、summary、环境/源码 SHA-256；
  `benchmark_source.py` 和 `model_source.py` 保存实际测量代码（模型源码仅比当前少 docstring）。
- `GR/generated/nosa_gr_mfu_20260925/`：12 个 Chrome trace 与含模块明细/逐层投影明细的 `summary.json`。
  每个 kernel 的名字、次数、时间归属均可追溯。
- `GR/generated/nosa_perf_20260925/`：早先错误套用生成循环的原始实验，已排除。

未测量 sparse attention、offloading、推荐 head、批量并发、跨请求缓存策略或其他 chunk size。
不从单次 profile 估计尾延迟，也不把标称峰值折算当作硬件计数器的利用率。
