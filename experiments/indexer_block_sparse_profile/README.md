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
当前 indexer 通过 IndexerCache 复用压缩前缀，只计算新增 K/CIS 窗口和稳定 pool；
检查当前 Q 及尚未校验的 K/CIS 后缀。这些开销纳入主计时。Triton 一次处理完整 query batch，
workload 的 `indexer_query_chunk_size=null` 表示不分块，
`indexer_execution=cached_flashinfer_v1` 标识增量缓存与 FlashInfer 两阶段选择。
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
  --request-file experiments/indexer_block_sparse_profile/output/data/nosa_cached_indexer_20260928_01/request.json \
  --device cuda:0 --warmup 2 --repeats 5 --profile-repeats 1
```

不传 `--request-file` 时用共享 GR 生成器、seed=42 生成固定长度请求；profile 总是读取
benchmark 已保存的同一请求。脚本固定 64K+1K 和 1024 chunk。当前依赖 SM90/Hopper、
PyTorch、Triton、FlashInfer、safetensors、tokenizer 与 Nsight Systems，沿用项目环境。
`--without-nsys` 保留独立端到端计时及 CUDA event 模块分解，省略原始 nsys / SQLite。
分模块 kernel MFU 依赖 SQLite，`--without-nsys` 时仅生成端到端 MFU。

检查整体提交瓶颈时使用 `--timeline-only`：仍先运行独立、无 profiler 的 benchmark，
第二进程只记录每次完整 forward 的根 NVTX 与 CUDA 活动，不安装 `SparseScopes`、
逐层 hook 或模块 CUDA events；只保留整段计时的起止 events。该模式生成
`launch_analysis.json`、`launch_summary.json/csv/md`，不生成模块分解或 MFU。
它需要 nsys，不能与 `--without-nsys` 同用；可用 `--profile-repeats 3` 重复时间线采集。

Python 入口：

```bash
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.capture --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.analyze --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.mfu --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.module_mfu --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.launch_report --help
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
  当前 indexer 子项为 `indexer_validate`、`indexer_cache_update`、`pooled_scores`、
  `topk_qa`、`prepare_cis`、`topk_cis`、`finish_selection`。
  子项不能再次加到 indexer 总时间上，也不从主 wall
  中减出“其他模块”。没有新完整窗口时，cache update 可以不发起 kernel。
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
`models.nosa.model`、dense 实验的
`experiments.nosa_gr_65536_1024.src.capture.execution_split` 和 `src.sources.source_hashes`；
`src/instrumentation.py` 临时包装已有 scoring/indexer/attention，退出时恢复；
`src/analyze.py` 校验完整阶段、层与 chunk 覆盖并聚合测量。实验 CPU 单元测试纳入
全局 `scripts/run_tests.sh`，结果仅输出终端。`src/mfu.py` 复用
`experiments.nosa_gr_65536_1024.src.mfu.matrix_flops` 的投影/FFN 计数，将 dense attention
替换为实际 sparse 因果工作量，并加入 CIS 与 indexer QK；只读取已完成的运行数据。
`src/module_mfu.py` 复用同一 FLOPs 口径，从 Nsight SQLite 关联 CUDA launch 与 kernel，
验证逐层执行顺序后归因模块。

## 结果与结论

### 当前实现：增量压缩缓存与 FlashInfer Top-K（2026-09-28）

run ID：`nosa_cached_indexer_20260928_01`。模型 indexer / SM90 indexer 的源码 SHA256
分别以 `e5f636a626ca` / `980c667fc87d` 开头，完整指纹和源码快照保存在该 run 中。
Triton 一次处理完整 query batch；QK、GQA 归约、舍入和五窗口 pooling 融合，
两阶段 Top-33 / Top-64 调用 FlashInfer。`IndexerCache` 保存压缩 K/CIS 和稳定 pool，
extend 只更新新增窗口，scratch 跨阶段、层复用。

硬件为 H200、SM90、132 SM，GPU UUID `2522820c-89d9-aa17-f79c-ca8cc767fb77`。
环境为 PyTorch `2.10.0+cu132`、CUDA 13.2、Triton `3.6.0`、FlashInfer `0.6.18`、
Nsight Systems `2025.6.3`。使用 NOSA-8B BF16、32 layers、32 Q heads / 2 KV heads / D128，
P=65536、Q=1024，模型 chunk=1024，64-block sparse 策略。

无 profiler 的独立 benchmark 每阶段预热 2 次、测量 5 次；模块 profile 在另一进程
同样预热 2 次，每阶段采集 1 次。两进程的源码、checkpoint、请求、设备与参数经
metadata 校验一致。完整前向与 prefix-ready extend 的 candidate hidden 全部有限，
最大绝对差为 **0**；插桩前后输出也精确一致。GPU 测量串行执行。

| 阶段 | 无 profiler wall ms，中位数 [min,max] | 端到端 MFU | 模块插桩进程 wall ms |
| --- | ---: | ---: | ---: |
| full_prefill | **3698.391 [3692.369,3708.610]** | **32.35%** | 4354.819 |
| extend | **62.034 [61.978,62.172]** | **30.68%** | 70.196 |

端到端 MFU 分母使用未插桩 wall 与 H200 BF16 dense 989 TFLOPS。
主计时对应吞吐为 full-prefill **17997 tokens/s**、extend **16507 tokens/s**。
插桩进程仅用于归因，不能作为主延迟，也不用于替换端到端 MFU 分母。
当前请求每个 cache session 的容量统计为 **2,262,624,000 bytes**，包含 K/V/CIS、
派生压缩缓存和 indexer scratch，不包含模型权重、其他激活和库 workspace。

### 模块 kernel 耗时与 MFU

下表从当前模块 trace 的 SQLite 关联 CUDA launch 与 kernel，按阶段汇总全部层/chunks。
分母是对应 kernel duration 之和，排除 host 空档、API、memcpy 和 memset。
模块 MFU 与端到端 MFU 使用不同分母；它们均不表示 SM occupancy。

| 模块 | full kernel ms | full MFU | extend kernel ms | extend MFU |
| --- | ---: | ---: | ---: | ---: |
| QKV projection | 104.356 | 77.90% | 1.602 | 78.09% |
| O projection | 91.187 | 79.25% | 1.404 | 79.20% |
| FFN gate + up | 716.786 | 80.65% | 11.019 | 80.71% |
| FFN down | 345.975 | 83.55% | 5.313 | 83.70% |
| Query-agnostic CIS | 25.871 | 0.00852% | 0.396 | 0.00856% |
| **Indexer total** | **677.822** | **5.39%** | **15.367** | **7.29%** |
| Block sparse attention | 1195.239 | 11.63% | 18.416 | 11.98% |

Indexer 内部如下；这些子项已包含在 total 中，不能再次相加到顶层。

| Indexer 子项 | full kernel ms | extend kernel ms | full / extend kernel 数 |
| --- | ---: | ---: | ---: |
| 有限值检查 `indexer_validate` | 10.474 | 0.162 | 4160 / 64 |
| 增量压缩与稳定 pool `indexer_cache_update` | 12.955 | 0.200 | 2080 / 32 |
| QK、softmax、GQA 与 pooling `pooled_scores` | **293.354** | **8.819** | 1952 / 32 |
| FlashInfer Top-33 `topk_qa` | 152.645 | 2.493 | 3904 / 64 |
| 构造 CIS 候选分数 `prepare_cis` | 19.520 | 0.563 | 1952 / 32 |
| FlashInfer Top-64 `topk_cis` | 181.473 | 3.013 | 3904 / 64 |
| ID 排序与 validity 写出 `finish_selection` | 7.130 | 0.117 | 1952 / 32 |

Indexer total 的 kernel 数为 **20032 / 320**。full 的前四个短上下文 chunk 直接输出
全部因果块；其输出 kernel 计入 total，不归入上表的长上下文 `finish_selection`。
当前 trace 中每次 FlashInfer Top-K 调用对应两个 kernel。

extend 的 QK/pooling 占 indexer kernel 时间约 **57.39%**，两次 Top-K 合计约 **35.83%**；
有限值检查与增量缓存更新合计约 **2.35%**。当前主要开销在打分和选块。
`pooled_scores` 的有效矩阵 MFU 为 full **12.45%**、extend **12.70%**：只给逻辑 QK
计一次有效 FLOPs，而两遍 QK、softmax、归约与 pooling 的全部时间均计入分母。
两次 Top-K 不计矩阵 FLOPs，因此 indexer total 的 MFU 更低。

模块 CUDA event 区间仍含 host 提交空档和插桩开销。例如 extend 的 cache update
区间为 **4.550 ms**，实际 kernel 合计仅 **0.200 ms**；indexer total 的 event 区间为
**27.368 ms**，kernel 合计为 **15.367 ms**。这些 inclusive 区间不能与子项重复相加，
也不能用来直接分解独立 benchmark 的 62.034 ms。

### 复现与报告材料

从仓库根目录用新 run ID 重跑同一请求：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/indexer_block_sparse_profile/scripts/run.sh \
  nosa_cached_indexer_rerun \
  --request-file experiments/indexer_block_sparse_profile/output/data/nosa_cached_indexer_20260928_01/request.json \
  --warmup 2 --repeats 5 --profile-repeats 1
```

本次原始启动参数保存在 `metadata.json`；新 run 自身保存完整 `request.json`，
上述复现命令使用该副本。完整数据、源码快照、SQLite 位于
`output/data/nosa_cached_indexer_20260928_01/`，日志及 `.nsys-rep` 位于同 run ID 的
`output/log/`、`output/profile/`。

报告材料由 `src.analyze`、`src.mfu`、`src.module_mfu` 从该 run 生成后直接复制：

| 报告材料 | 该 run 内的来源 |
| --- | --- |
| [端到端 MFU 与计数](report/cached_indexer_mfu.json) | `mfu.json` |
| [模块 kernel MFU](report/cached_indexer_module_mfu.csv) | `module_mfu.csv` |
| [端到端计时](report/cached_indexer_timings.csv) | `timings.csv` |
| [模块 event 区间](report/cached_indexer_stages.csv) | `stages.csv` |

轻量整体时间线（`--timeline-only`）和 CIS 独立微测量（`scripts/bottleneck.sh`）
**改动后未运行**；当前报告不对 GPU active/idle 比例或 CIS eager/graph 加速比作结论。
这两条采集入口仍可使用新 run ID 执行。

### 有效矩阵 FLOPs 计数


分子包含 QKV/O、gate/up/down、CIS 投影、选中块的 QK/AV，以及 indexer 对有效压缩
K 的一次逻辑 QK；每次乘加计 2 FLOPs，不包含 LM head。

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
