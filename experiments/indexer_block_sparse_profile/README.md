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
检查当前 Q 及尚未校验的 K/CIS 后缀。这些开销纳入主计时。CUDA indexer 一次处理完整 query batch，
workload 的 `indexer_query_chunk_size=null` 表示不分块，
`indexer_execution=cached_flashinfer_v1` 标识增量缓存与 FlashInfer 两阶段选择。
仅在实验进程把上下文上限扩大为 66560，保留 checkpoint 的 LongRoPE factors；不改磁盘
配置，不对超出原 32768 上下文后的模型质量作结论。

本次比较使用同一代码、请求、checkpoint 和测量参数，仅切换 `--kernel-backend`：

- `native`：QK score 使用本地 CUDA/CuTe WGMMA：第一 kernel 计算分段 softmax normalizer，
  第二 kernel 重算 QK、合并 normalizer，融合 GQA 舍入和五窗口 pooling。
  本实验的 1024-query batch 在压缩 K≥2047（上下文≥32768 tokens）时使用该路径，
  更短的已打分 chunk 保留单 kernel Triton；
  block sparse attention 由 TMA/WGMMA 处理四个相邻 query 的选块并集，各 query 保持独立的
  selection membership、CIS、因果遮罩和在线 softmax；先启动 grouped kernel，再启动按掩码
  执行的 per-query fallback（repair）kernel。两次 launch 的全部耗时均计入 attention。
- `triton`：保留的两遍 fused QK/pooling 与 block sparse attention 实现，用新 run 重新测量。

这里的 `native` 表示上述混合 QK 调度与原生 attention 组合。
两组均使用增量压缩缓存与 FlashInfer Top-33/Top-64；native 不引入 EzKernelKit 运行时，
只通过 TVM FFI 编译 `operators/sm90/csrc/`，复用顶层共享 CUTLASS。
`workload.kernel_backend` 区分 `cuda_tvm_ffi` / `triton`；旧接口字段 `backend="triton"`
仍表示 SM90 dispatcher。构建元数据记录编译器、编译参数、TVM FFI、CUTLASS 与 CUDA 源码指纹。
64K+1K 的 native normalizer 临时缓冲为 0.25 MiB，其分配与计算计入计时；
另有 2 KiB 的 group fallback 掩码；临时缓冲均不属于请求 resident cache 的容量统计。
两组独立构建各自的 sparse prefix；浮点舍入可能改变近似并列的选块，不能据此声称输出逐位相同。

## 运行

从仓库根目录执行，脚本也能从其他工作目录启动：

```bash
bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_native_001 --kernel-backend native --peak-tflops 989
bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_triton_001 --kernel-backend triton --peak-tflops 989
bash experiments/indexer_block_sparse_profile/scripts/run.sh --help
```

默认设备 `cuda:0`，checkpoint `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`；每阶段预热 2 次，
独立端到端计时 5 次取中位数，模块 profile 每阶段 1 次。第二个进程同样先完整预热，
JIT 编译、CUDA 模块加载及 allocator 初始化不计入正式采集。

可明确使用已有实验的同一 GR 请求：

```bash
bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_e2e_same_request \
  --request-file experiments/indexer_block_sparse_profile/output/data/sparse_native_h200_gpu1_20260928_02/request.json \
  --device cuda:0 --warmup 2 --repeats 5 --profile-repeats 1 --peak-tflops 989
```

不传 `--request-file` 时用共享 GR 生成器、seed=42 生成固定长度请求；profile 总是读取
benchmark 已保存的同一请求。脚本固定 64K+1K 和 1024 chunk。当前依赖 SM90/Hopper、
PyTorch、Triton、FlashInfer、TVM FFI、nvcc、共享 CUTLASS、safetensors、tokenizer 与 Nsight Systems，沿用项目环境。
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

本节基于当前 grouped attention 与修正后的 score 实现重新完成的两组有效运行：
native 为 `sparse_native_h200_gpu1_20260928_02`，Triton control 为 `sparse_triton_h200_gpu1_20260928_01`。
两组的 58 个采集源码指纹、请求字节、checkpoint 文件记录、设备、依赖和预热/重复参数一致；
构建信息只允许 `selected_backend` 不同。`src/compare.py` 已校验这些条件及派生报告的输入哈希。
checkpoint 核验使用配置 SHA256 与权重文件路径、大小和 mtime。
输入为 synthetic GR、65536+1024 tokens，使用上述 NOSA-8B BF16 resident HBM 路径。
解析后的完整请求内容与本次 dense/pattern 实验相同。sparse capture 将请求重新保存为
带缩进的 JSON，因此其文件字节 SHA256 与 dense/pattern 不同；本节两组 sparse 文件彼此字节一致。

### 实测环境

native / Triton metadata 时间分别为 `2026-09-28T07:50:52.685583+00:00` / `2026-09-28T07:55:30.457797+00:00`。
两组均在 GPU 1 串行测量，原始设备名 `NVIDIA H20Z`，按本机硬件确认为 H200；
SM90、132 SM、设备显存 139.812 GiB。
CUDA 与 Nsight UUID 均为 `a2e4882b-3248-6018-14ab-90e45b5a5b04`，Nsight PCI 地址为 `0000:67:00.0`。
`CUDA_VISIBLE_DEVICES=1` 将本进程的 `cuda:0` 映射到物理 GPU 1；分析用 Nsight 的
`PROCESSES` 和 `TARGET_INFO_CUDA_DEVICE` 核验进程、CUDA ordinal 与物理 GPU，再验证 UUID、SM、架构及显存。
驱动为 `570.124.06`。MFU 显式使用 `--peak-tflops 989`，对应 H200 BF16 dense 标称参考值，
不使用 2:4 稀疏峰值，也不将其视为实测可持续峰值。

| 依赖 | 实际运行版本 | 同次源码快照中的 uv.lock |
| --- | --- | --- |
| PyTorch | 2.12.1+cu130 | 2.12.1+cu130 |
| Triton | 3.7.1 | 3.7.1 |
| FlashInfer | 0.6.18 | 0.6.18 |
| TVM FFI | 0.1.13.post3 | 0.1.13.post3 |

PyTorch CUDA 构建版本为 `13.0`（`torch.version.cuda`）；nvcc 路径为 `/usr/local/cuda-13.2/bin/nvcc`，版本
`Cuda compilation tools, release 13.2, V13.2.86`。
CUTLASS 为 `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`；Nsight 为 `NVIDIA Nsight Systems version 2025.6.3.541-256337736014v0`。
native 编译采用 `-O3`、C++20、`sm_90a` 与 `-lineinfo`，完整 flags、源码/头文件指纹保存于 metadata。
实际依赖以运行元数据为准，锁文件随源码快照保留。

### 独立端到端计时

每阶段预热 2 次、独立计时 5 次；下面为未插桩墙钟中位数和 `[min, max]`，单位 ms。
加速比为 Triton 中位数 / native 中位数，MFU 分母为各自独立进程的未插桩墙钟中位数。
本次 [dense 报告](../nosa_gr_65536_1024/README.md)的墙钟来自 nsys 进程内关闭采集的阶段，
与本节计时边界不同；下表加速比仅比较两个 sparse 后端。

| 阶段 | Triton wall ms [min, max] | native wall ms [min, max] | Triton MFU | native MFU | 加速比 |
| --- | --- | --- | --- | --- | --- |
| full_prefill | 3827.189 [3817.400, 3843.540] | 3304.487 [3240.375, 3364.833] | 31.26% | 36.20% | 1.158× |
| extend | 63.933 [63.754, 64.257] | 54.172 [54.088, 54.847] | 29.77% | 35.13% | 1.180× |

两组有效矩阵工作量相同：full 为 `1,183,130,731,937,792` FLOPs，
extend 为 `18,820,462,804,992` FLOPs。
QK 只计一次逻辑矩阵乘，attention 只计各 query 实际选择的因果 token；normalizer/score 两遍重算、
pooling halo、选块并集中不属于该 query 的工作、fallback 重算、future mask 与 padding 不计入分子，
所有已发起 kernel 的耗时均保留在分母。每组内部 full/extend 的全部 candidate hidden 均有限且
最大绝对差为 0，profile 也保持输出不变；该验收不表示 native 和 Triton 两组输出逐位相同。

### 算子 kernel 时间对照

每组在独立 nsys 进程采集 1 次完整模块 profile；表中为 correlated GPU kernel duration 总和，单位 ms。
它不是未插桩墙钟的可加分段，也不能与 CUDA event 区间相加。

| 模块 | full Triton | full native | 加速比 | extend Triton | extend native | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| `pooled_scores` | 293.170 | 255.735 | 1.146× | 9.156 | 7.169 | 1.277× |
| `block_sparse_attention` | 1197.041 | 614.079 | 1.949× | 18.480 | 9.932 | 1.861× |
| `indexer_total` | 678.745 | 641.800 | 1.058× | 15.743 | 13.754 | 1.145× |

native score 维持混合调度：1024-query batch 在完整压缩 K≥2047 时使用
`normalizer_kernel → scores_kernel`；更短的已打分 chunk 使用单个 Triton `_scores` kernel。
full 前四个 chunk 直接选择全部因果块并绕过 score；其后实际捕获
864 个 Triton score scopes 和 1088 个 native score scopes，
合计 3040 个 score kernels；extend 的 32 个 native scopes
对应 64 个 kernels。Triton control 的 full / extend score kernels 为
1952 / 32。

native attention 每层/chunk 发起 `grouped_attention_kernel` 和 `attention_kernel` 两个 launch，
后者根据 group fallback 掩码执行 per-query repair；两者完整耗时构成上表的 attention 分母。
其 full / extend kernel 数为 4160 / 64，
Triton control 为 2080 / 32。
下面从同一 SQLite 按准确 CUDA symbol 拆分，并核验时间和 launch 数与 module MFU 完全守恒。

| 阶段 | grouped ms | grouped launches | fallback/repair ms | fallback launches | attention total ms |
| --- | --- | --- | --- | --- | --- |
| full_prefill | 607.934699 | 2080 | 6.144527 | 2080 | 614.079226 |
| extend | 9.836671 | 32 | 0.095424 | 32 | 9.932095 |

fallback launch 次数不表示实际回退的 group/query 数量；本次没有采集 fallback 掩码统计。
每层/query 的 useful QK/PV FLOPs 只计一次，既不按两个 launch 加倍，也不由 launch 数推导层数。
单次模块 profile 的比值描述本次归属结果，不用于隔离纯 launch 成本或声称 HBM 带宽已饱和。

### native 模块 MFU 与 indexer 子项

模块 MFU 为 100 × useful matrix FLOPs /（correlated kernel 总秒数 × 989 TFLOPS），不表示 SM occupancy。
CIS 与 indexer 的 inclusive 分母包含其非矩阵操作。

| native 模块 | full kernel ms | full MFU | extend kernel ms | extend MFU |
| --- | --- | --- | --- | --- |
| `qkv_proj` | 103.927 | 78.22% | 1.597 | 78.32% |
| `o_proj` | 91.202 | 79.23% | 1.405 | 79.11% |
| `gate_up_proj` | 712.033 | 81.19% | 10.963 | 81.12% |
| `down_proj` | 343.827 | 84.07% | 5.281 | 84.21% |
| `block_sparse_attention` | 614.079 | 22.64% | 9.932 | 22.21% |
| `indexer_total` | 641.800 | 5.69% | 13.754 | 8.14% |
| `cis_projection` | 25.541 | 0.0086% | 0.394 | 0.0086% |

| native indexer 子项 | full kernel ms | extend kernel ms | full / extend kernel 数 |
| --- | --- | --- | --- |
| `indexer_validate` | 11.283 | 0.174 | 4160 / 64 |
| `indexer_cache_update` | 12.963 | 0.200 | 2080 / 32 |
| `pooled_scores` | 255.735 | 7.169 | 3040 / 64 |
| `topk_qa` | 153.419 | 2.512 | 3904 / 64 |
| `prepare_cis` | 19.471 | 0.563 | 1952 / 32 |
| `topk_cis` | 181.519 | 3.020 | 3904 / 64 |
| `finish_selection` | 7.139 | 0.117 | 1952 / 32 |

`pooled_scores` 的有效矩阵 MFU 为 14.29% /
15.62%（full / extend）；其余 indexer 子项没有矩阵 FLOPs，
MFU 不适用。子项已包含在 `indexer_total` 中，不能重复相加；full 的短上下文全块选择直接记入 parent。
同次 native profile 的 indexer CUDA event 区间为
2491.561 / 33.983 ms，
kernel 总时间为 641.800 /
13.754 ms；前者包含提交空档及插桩影响，二者不能相加。

当前输入下，native 组合的 full / extend 端到端加速为 1.158× / 1.180×，
attention kernel 加速为 1.949× / 1.861×。
这是完整 NOSA resident HBM 同源对照；本次没有 DRAM/CXL 搬运、fetch/compute overlap 或网络服务测量，
不用于 offload 性能或模型质量结论。独立 synthetic kernel 测量见 [nosa_kernel_mfu](../nosa_kernel_mfu/README.md)。

### 报告数据与复现

下面文件从这两组成功 run 直接复制。`stages.csv` 是 inclusive CUDA event 数据，
`module_mfu.csv` 是独立 nsys kernel/MFU 数据；完整按层/chunk 记录、源码快照和 SQLite
保存在对应 `output/data/<run_id>/`，原始 nsys 保存在 `output/profile/<run_id>/`。

| 来源 | 墙钟汇总 | event 阶段 | 端到端 MFU | 模块 kernel / MFU | 采集元数据 |
| --- | --- | --- | --- | --- | --- |
| `sparse_native_h200_gpu1_20260928_02` | [timings.csv](report/native/timings.csv) | [stages.csv](report/native/stages.csv) | [mfu.json](report/native/mfu.json) | [module_mfu.csv](report/native/module_mfu.csv) | [metadata.json](report/native/metadata.json) |
| `sparse_triton_h200_gpu1_20260928_01` | [timings.csv](report/triton/timings.csv) | [stages.csv](report/triton/stages.csv) | [mfu.json](report/triton/mfu.json) | [module_mfu.csv](report/triton/module_mfu.csv) | [metadata.json](report/triton/metadata.json) |

两组对照为 [comparison.csv](report/comparison.csv) 与含输入指纹/定义的
[comparison.json](report/comparison.json)。attention 双 launch 明细为
[attention_launches.csv](report/native/attention_launches.csv)，SQLite 和模块报告来源指纹保存在
[attention_launches.json](report/native/attention_launches.json)。这些派生文件来自 native run 的 `comparison/`。
README 表格由上述 JSON 生成，生成脚本快照为
`output/data/sparse_native_h200_gpu1_20260928_02/comparison/report_rebuild.py`。

从仓库根目录复现新的测量（使用新的 run ID），随后生成比较材料：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_native_rerun \
  --kernel-backend native --peak-tflops 989 \
  --request-file experiments/indexer_block_sparse_profile/output/data/sparse_native_h200_gpu1_20260928_02/request.json
CUDA_VISIBLE_DEVICES=1 bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_triton_rerun \
  --kernel-backend triton --peak-tflops 989 \
  --request-file experiments/indexer_block_sparse_profile/output/data/sparse_native_rerun/request.json
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.compare \
  --native-data-dir experiments/indexer_block_sparse_profile/output/data/sparse_native_rerun \
  --triton-data-dir experiments/indexer_block_sparse_profile/output/data/sparse_triton_rerun \
  --output-dir experiments/indexer_block_sparse_profile/output/data/sparse_native_rerun/comparison
```

默认仍为 warmup=2、repeats=5、profile-repeats=1；两次测量间必须保持采集源码、依赖、请求和设备一致。
只重建已有 run 的比较时，将路径替换为本节两组 run，使用尚不存在的 `comparison_rebuilt` 目录。
复制表中五类 run 文件及 comparison 文件即可重建选定 report 数据。原两组 run 的 grouped/fallback
明细和 README 可从仓库根目录用保存的脚本重新生成：

```bash
.venv/bin/python experiments/indexer_block_sparse_profile/output/data/sparse_native_h200_gpu1_20260928_02/comparison/report_rebuild.py \
  --native sparse_native_h200_gpu1_20260928_02 --triton sparse_triton_h200_gpu1_20260928_01
```

轻量 timeline-only 和 CIS 独立 bottleneck 沿用可运行入口，本次没有新增这两项测量或结论。
