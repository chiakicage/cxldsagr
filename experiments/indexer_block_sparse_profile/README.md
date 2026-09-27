# NOSA indexer + block sparse：64K + 1K 端到端性能

## 实验目的与测量边界

测量完整 NOSA sparse 模型的单请求端到端前向延迟，并定位 CIS、indexer 与 block sparse
attention 的开销。主指标来自未启用 profiler 的独立进程；模块分解和 Nsight Systems
时间线由第二个进程采集。两者使用同一请求、checkpoint、模型参数、源码和设备。

| 阶段 | 起始缓存 | 模型输入 | 执行方式 | 主指标 |
| --- | --- | --- | --- | --- |
| `full_prefill` | 空 | 66,560 tokens | 65 个 1024-token chunk，全部 32 层 | 整段墙钟延迟、66,560 / 秒 |
| `extend` | 65,536-token prefix 已完成 | 1,024 candidate tokens | 单次 1024-token forward，全部 32 层 | 整段墙钟延迟、1,024 / 秒 |

64K 是 instruction + history 的合计，1K 是完整 candidate suffix，不把历史尾部移入 extend。
当前模板 instruction 为 28 tokens，因此 GR 输入预算为 history 65508、item budget 1052；
实际执行边界为 `[0,65536)` 与 `[65536,66560)`。`request.json` 和 `execution.json`
保留真实边界，逐层 audit 检查 `Q=[1024,32,128]`、`K=V=[66560,2,128]`、
selection=`[1024,2,64]` 且全部选择有效。

使用 NOSA-8B 的 BF16 权重、32 layers、32 Q heads、2 KV heads、D128。完整 sparse 策略为
64-token block、17 inclusive local、含 sink/local 的 query-aware 阶段保留 33 块，
query-agnostic CIS 补满 64 块；attention 加入原始 CIS 分数。模型及算子均调用
[现有 NOSA 实现](../../models/nosa/README.md)，不在实验代码复制计算逻辑。

端到端计时包含 embedding、全部 decoder、CIS 投影、K/V/CIS 写入、压缩、query-aware
打分与选块、block sparse attention、输出投影、FFN、final norm、Python 提交和 CUDA 完成等待。
输出为最后一个 chunk 的 normalized hidden states，不执行 LM head、采样或自回归生成。
checkpoint 加载、tokenizer/GR 生成、输入 H2D、缓存分配/reset 以及 extend 的 prefix 构建
均在计时外；这属于单请求模型前向延迟，不是网络服务延迟。

完整 K/V/CIS 驻留 HBM，无 DRAM backing、offload fetch、缓存淘汰或 fetch/compute overlap。
indexer 仍按当前实现重新压缩 resident 前缀并执行输入检查，这些开销纳入主计时。
仅在实验进程把上下文上限扩大为 66560，保留 checkpoint 的 LongRoPE factors；不改磁盘
配置，不对超出原 32768 上下文后的模型质量作结论。

## 运行

从仓库根目录执行，脚本也能从其他工作目录启动：

```bash
bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_e2e_001
bash experiments/indexer_block_sparse_profile/scripts/run.sh --help
```

默认设备 `cuda:0`，checkpoint `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`；每阶段预热 2 次，
独立端到端计时 5 次取中位数，模块 profile 每阶段 1 次。第二个进程同样先完整预热，
JIT 编译、CUDA 模块加载及 allocator 初始化不计入正式采集。

可明确使用已有实验的同一 GR 请求：

```bash
bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_e2e_same_request \
  --request-file experiments/nosa_gr_65536_1024/output/data/flashinfer_merged_gemm_65536_1024_20260925_02/request.json \
  --device cuda:0 --warmup 2 --repeats 5 --profile-repeats 1
```

不传 `--request-file` 时用共享 GR 生成器、seed=42 生成固定长度请求；profile 总是读取
benchmark 已保存的同一请求。脚本固定 64K+1K 和 1024 chunk。当前依赖 SM90/Hopper、
PyTorch、Triton、FlashInfer、safetensors、tokenizer 与 Nsight Systems，沿用项目环境。
`--without-nsys` 保留独立端到端计时及 CUDA event 模块分解，省略原始 nsys / SQLite。
分模块 kernel MFU 依赖 SQLite，`--without-nsys` 时仅生成端到端 MFU。

Python 入口：

```bash
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.capture --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.analyze --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.mfu --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.module_mfu --help
```

脚本依次执行独立 benchmark → nsys profile → SQLite export → analyze → 离线 MFU。所有步骤在系统
临时目录 staging，子进程失败保留退出码，失败产物留在实验目录之外。只有全链成功才发布
至新的 run ID，已有 ID 拒绝覆盖。
启用 nsys 时，最后还从 SQLite 生成分模块 kernel MFU。

## 指标定义与产物

- `wall_ms`：开始提交到设备全部完成的墙钟毫秒数，主端到端指标。
- `host_submit_ms`：模型前向的 host 提交时间，含模型内部 CUDA 等待；它与设备执行重叠，
  不能与 CUDA 时间相加，也不表示纯 CPU 计算时间。
- `cuda_span_ms`：整段 CUDA events 之间的时间，含 host launch 造成的空档，不是 kernel active 时间。
- 模块 `cuda_elapsed_ms`：单独 profile 下的 inclusive CUDA event 区间，可能含 launch gaps
  和 instrumentation 开销；先汇总一次运行内全部层/chunks，再对 profile repeats 取中位数。
  `cis_projection`、`indexer_total`、`block_sparse_attention` 为独立顶层区间。
  `compression_k`、`compression_cis`、`compressed_scores`、`select_from_scores` 是
  indexer 的嵌套子项，不能再次加到 indexer 总时间上，也不从主 wall 中减出“其他模块”。
- `mfu_pct`：有效矩阵 FLOPs /（未插桩墙钟中位数 × GPU 峰值 FLOP/s）× 100。
  默认采用 H200 SXM 标称 BF16 dense Tensor Core 峰值 **989 TFLOPS**，不是 2:4 稀疏峰值，
  也不是实测可持续峰值；其他 GPU/SKU 可用 `--peak-tflops` 显式指定。

完整前向与 prefix-ready extend 使用相同分块边界，测量外检查全部 1K candidate hidden
有限且精确一致；profile 还检查 instrumentation 不改变这些输出。正确性检查是数据验收，
不作为性能结果。每次 full-prefill 在计时前 reset 独立 cache；重复 extend 回退独立 prefix
cache 的长度，只重写 suffix。若模型未来引入 opaque layer state，会明确拒绝这种回放。

产物默认忽略，不提交 Git：

```text
output/log/<run_id>/             各子进程分开的 stdout/stderr
output/data/<run_id>/            request.json、execution.json、metadata.json
                                measurements.json、profile_metadata.json、attention_audit.json
                                summary.json、report.md、timings.csv、stages.csv、layers.csv、mfu.json
                                module_mfu.json、module_mfu.csv、module_mfu.md（需要 nsys）
                                sources/ 源码快照、nsys.sqlite
output/profile/<run_id>/         indexer_block_sparse_profile.nsys-rep
```

metadata 记录实际硬件/UUID、依赖版本、输入与精度、原始/实验上下文、checkpoint 文件
属性、源码指纹和测量边界。Nsight 关闭 CPU sampling/context-switch tracing，仅采集
CUDA/NVTX/OS runtime；模块 NVTX 标签为 `NOSA/<phase>/layer_<id>/<stage>/q<start>+<length>`。

调用模块：`src/capture.py` 复用共享 GR、`executor.model_executor.run_chunks`、
`models.nosa.model`、旧 dense 实验的
`experiments.nosa_gr_65536_1024.src.capture.execution_split` 和 `src.sources.source_hashes`；
`src/instrumentation.py` 临时包装已有 scoring/indexer/attention，退出时恢复；
`src/analyze.py` 校验完整阶段、层与 chunk 覆盖并聚合测量。实验 CPU 单元测试纳入
全局 `scripts/run_tests.sh`，结果仅输出终端。`src/mfu.py` 复用
`experiments.nosa_gr_65536_1024.src.mfu.matrix_flops` 的投影/FFN 计数，将 dense attention
替换为实际 sparse 因果工作量，并加入 CIS 与 indexer QK；只读取已完成的运行数据。
`src/module_mfu.py` 复用同一 FLOPs 口径，从 Nsight SQLite 关联 CUDA launch 与 kernel，
验证逐层执行顺序后归因模块。

## 结果与结论

已于 2026-09-26 完成测量，run ID：`nosa_sparse_e2e_65536_1024_20260926_03`。
硬件为 NVIDIA H200 / SM90（nvidia-smi 名称 M403，132 SM，GPU 0），UUID
`2522820c-89d9-aa17-f79c-ca8cc767fb77`；驱动 570.124.06，PyTorch `2.10.0+cu132`、
CUDA 13.2、Triton `3.6.0`、FlashInfer `0.6.18`、Nsight Systems `2025.6.3`。
实际 torch/triton 版本与仓库锁定版本不同，不能据此声称锁定版本组合已完成本实验。
测量期间没有并行 GPU 测试。

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/indexer_block_sparse_profile/scripts/run.sh \
  nosa_sparse_e2e_65536_1024_20260926_03 \
  --request-file experiments/nosa_gr_65536_1024/output/data/flashinfer_merged_gemm_65536_1024_20260925_02/request.json
```

使用上述默认 BF16、64K+1K、1024 chunk、2 warmup / 5 repeats / 1 profile repeat。
request 内容与所引用 dense 实验完全一致；完整前向与 prefix-ready extend 的全部
`[1024,4096]` candidate hidden 均有限，最大绝对差为 **0**，插桩后输出同样一致。

### 独立端到端计时

| 阶段 | 墙钟中位数 ms | min–max ms | Host 提交中位数 ms | CUDA span 中位数 ms | tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_prefill，66560 tokens | **10580.463** | 10571.731–10814.865 | 10579.554 | 10580.437 | 6290.84 |
| prefix-ready extend，1024 tokens | **194.837** | 194.584–194.928 | 193.916 | 194.821 | 5255.68 |

这是完整 sparse 模型前向的稳态延迟，包含 indexer 和 CIS；不是只测 attention kernel。
不包含 KV 分配、GR/tokenizer、prefix 构建（extend）或 LM head。Host 提交几乎覆盖整段
墙钟，但其中含 indexer 内部 CUDA 同步等待，不能据此将整段归因为 CPU 计算。

### 端到端 MFU

2026-09-27 根据上述同一 run 的配置、源码快照及无 profiler 墙钟中位数离线推导，
没有新增 GPU 测量。分子包含 QKV/O、gate/up/down、CIS 投影、选中块的 QK/AV，
以及 indexer 对有效压缩 K 的一次逻辑 QK；每次乘加计 2 FLOPs，不包含 LM head。

| 阶段 | 有效矩阵 FLOPs（10¹²） | 墙钟中位数 ms | 有效 TFLOPS | 端到端 MFU |
| --- | ---: | ---: | ---: | ---: |
| full_prefill，66560 tokens | 1183.130732 | 10580.463 | 111.82 | **11.31%** |
| prefix-ready extend，1024 tokens | 18.820463 | 194.837 | 96.60 | **9.77%** |

具体计数口径：

- 对零起始 query 位置 `p`，当前块必选，其他选中块均为完整历史块，因此每个 Q head
  可见的 KV token 数精确为 `64 * min(p // 64, 63) + p % 64 + 1`，与具体 top-k ID 无关。
  full/extend 的因果 token pairs 分别为 262275584 / 4162048；QK+AV 乘以
  `4 * 32 layers * 32 Q heads * 128 head_dim`。
- indexer 每个 query 的完整因果压缩窗口数为 `max(0, (p - 31) // 16 + 1)`。
  前四个 1024-token chunk 的 KV 总量不超过 64 块，跳过 QK 打分；压缩本身仍执行。
  full/extend 有效 query-window pairs 为 137830720 / 4225600，乘以
  `2 * 32 layers * 32 Q heads * 128 head_dim`，逻辑 QK 只计一次。
- CIS 的 delta 投影每层为 `2 * query_tokens * (2 * 128) * 2` FLOPs。
  压缩、softmax/softplus、GQA 分数归约、pooling/top-k、CIS 加性 bias、norm、RoPE、
  激活及内存操作不进入矩阵 FLOPs 分子，耗时均保留在分母。
- 两遍 indexer QK 的重算、未来位置遮罩和 tile padding 不属于有效模型 FLOPs，
  不进入分子；这与 dense 实验按有效因果矩阵工作量计数的约定一致。
  此处 MFU 不表示实际执行指令利用率或 SM occupancy，也不使用插桩模块区间计算主 MFU。

复算命令（默认峰值 989 TFLOPS）：

```bash
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.mfu \
  experiments/indexer_block_sparse_profile/output/data/nosa_sparse_e2e_65536_1024_20260926_03
```

`output/data/nosa_sparse_e2e_65536_1024_20260926_03/mfu.json` 保留逐项 FLOPs、有效 pair
数量、分母、峰值假设、输入及分析代码 SHA256。原始计时、metadata 和当时的源码快照保持不变。

### 分模块 MFU：实际 kernel 耗时口径

同一 run 的 Nsight Systems trace 于 2026-09-27 离线分析，无新增 GPU 测量。
下表分母为模块内 **GPU kernel duration 之和**，分子沿用上节有效矩阵 FLOPs，
峰值仍为 H200 BF16 dense **989 TFLOPS**。每个阶段只有一次 profile，跨全部层/chunks
求和；保留 scope 内非矩阵 kernel 的耗时，排除 CPU 提交空档、CUDA API、memcpy/memset
活动和 event instrumentation 的等待。此口径与端到端墙钟 MFU、下节 CUDA event 区间不同。

| 模块 | full kernel ms | full MFU | extend kernel ms | extend MFU |
| --- | ---: | ---: | ---: | ---: |
| QKV projection | 103.914 | **78.23%** | 1.594 | **78.46%** |
| O projection | 90.728 | **79.65%** | 1.395 | **79.71%** |
| FFN gate + up | 716.177 | **80.72%** | 11.014 | **80.75%** |
| FFN down | 343.922 | **84.05%** | 5.288 | **84.09%** |
| Query-agnostic CIS，含 softplus 等 | 25.357 | **0.00870%** | 0.388 | **0.00873%** |
| Indexer total，含全部子项及校验 | 2946.886 | **1.24%** | 86.910 | **1.29%** |
| Indexer compressed QK scores（子项） | 1977.463 | **1.85%** | 59.791 | **1.87%** |
| Block sparse attention | 1196.500 | **11.62%** | 18.490 | **11.93%** |

indexer total 与 compressed QK 行使用相同的有效 QK FLOPs，但前者分母还包括压缩、
选块及校验 kernel；两者是父子关系，不能相加。压缩和选块的矩阵 MFU 为 N/A：

| Indexer 非矩阵子项 | full kernel ms | extend kernel ms | 矩阵 MFU |
| --- | ---: | ---: | --- |
| K compression | 23.285 | 0.592 | N/A |
| CIS compression | 8.664 | 0.146 | N/A |
| Block selection | 565.477 | 19.908 | N/A |

四个 indexer 子项的核时之和不覆盖其全部校验、分配和拷贝 kernel。
norm、RoPE、SwiGLU 的逐元素激活同样不定义本次矩阵 MFU。CIS 的 delta 输出只有
2 个 KV-head 分数，矩阵为 `[1024,256] × [256,2]`，分子很小；该行还包含 softplus
等 kernel，不能和大型 projection GEMM 的利用率直接类比。

归因方法及检查：

- 使用 kernel 的 CUDA correlation ID 找到 host launch，再按同一 host thread 的
  NVTX 包含关系归入模块；不使用 GPU 执行时间是否落在 host NVTX 区间内作为依据。
- CIS、indexer 及其子项、block attention 有直接 NVTX 标签。四个普通投影没有单独标签，
  根据该 run 保存的模型源码顺序重建：QKV → RoPE → CIS → indexer → attention →
  O → norm → gate/up → SiLU → down。检查每层每 chunk 的五个 GEMM（含 CIS）、
  CIS/attention NVTX 锚点及 RoPE/norm/SiLU kernel 锚点；O 和 down 即使 kernel 同名也不混用。
  全部 2080 次 full 层调用及 32 次 extend 层调用通过检查，无未归属的 projection GEMM。
  离线分析校验保存的源码 SHA256；推理图变更后须重新核对顺序归因，或增加显式模块标签。
- trace 为单进程、单设备、单 stream；SQLite 的 GPU UUID、SM90 和 132 SM 与 metadata 一致。
  核时取 duration sum；Nsight 时间戳的最大边界重叠为 928 ns，因此此值不称为区间并集
  active time。它仍受 profiler 的运行环境影响，
  不替代无 profiler 的端到端测量，也不表示硬件指令利用率或 SM occupancy。

普通投影 GEMM 的 MFU 约 78%–84%，indexer 的有效矩阵 MFU 明显更低。
结合下节包含提交空档的模块区间，后续应优先优化 indexer 的计算、选块及提交/同步开销。

复算并生成逐项 JSON/CSV/Markdown：

```bash
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.module_mfu \
  experiments/indexer_block_sparse_profile/output/data/nosa_sparse_e2e_65536_1024_20260926_03
```

产物为同一运行数据目录内的 `module_mfu.json`、`module_mfu.csv` 和 `module_mfu.md`。

### 单独插桩运行的模块区间

插桩运行的整段墙钟为 full-prefill **16053.869 ms**、extend **268.304 ms**。
下表是该运行内跨全部层/chunks 的 inclusive CUDA event 区间合计；采集开销明显，
不能将表中模块时间当作上表无 profiler 墙钟的直接分解，也不能称为纯 kernel active 时间。

| 阶段 | full_prefill ms | extend ms | 归属 |
| --- | ---: | ---: | --- |
| CIS projection | 46.434 | 1.009 | 顶层 |
| indexer total | **13079.259** | **220.438** | 顶层 |
| block sparse attention | 1437.931 | 22.134 | 顶层 |
| K compression | 89.564 | 1.462 | indexer 子项 |
| CIS compression | 68.041 | 1.027 | indexer 子项 |
| compressed query-aware scores | 3988.300 | 87.412 | indexer 子项 |
| block selection | 3268.818 | 50.507 | indexer 子项 |

full-prefill 的三个顶层模块各调用 2080 次（65 chunks × 32 layers），extend 各 32 次。
extend 的 query chunk size 为现有 indexer 默认 64，因此压缩打分与选块各调用
512 次（16 query chunks × 32 layers）；相关调用和逐层结果已写入 CSV。

当前插桩结果中 indexer 区间明显大于 block sparse attention，后续性能工作应优先检查
indexer 的打分、选块及调用/同步开销。压缩与选块仍消费全 HBM resident 前缀；这些数字
不代表 offloading 收益，也不评价 sparse 模型质量。此实验没有重新测量 dense 对照，
不据此给出同条件 dense/sparse 加速比。

主结果、原始样本、逐层分解、源码及 SQLite 位于
`output/data/nosa_sparse_e2e_65536_1024_20260926_03/`，原始 trace 为
`output/profile/nosa_sparse_e2e_65536_1024_20260926_03/indexer_block_sparse_profile.nsys-rep`，
stdout/stderr 位于 `output/log/nosa_sparse_e2e_65536_1024_20260926_03/`。
