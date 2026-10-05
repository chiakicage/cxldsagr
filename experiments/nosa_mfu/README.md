# NOSA 算子与模型 MFU 报告

本报告集中展示 NOSA 算子、完整 indexer / attention 模块和完整 resident 模型的效率，
用于检查 baseline 的实现性能。算子与完整模块是主要入口；完整模型结果用于核对这些
局部收益在 32 层前向中的表现。各层级分别报告 API、kernel 和端到端时间。

2026-10-05 已完成统一执行框架重构后的真实输入算子、native/Triton 完整模块与完整
sparse 与 dense 模型补测。各组先独立运行数值 check，再分别运行 bench 和 profile；
本页将无 profiler 的主延迟与独立诊断中的 kernel 时间分开报告。

Synthetic 算子保留 `kernel_flags_synthetic_20261004_01` 的有效结果，输入、GPU 与
源码身份仍属于该历史运行。本轮没有重测 synthetic，也不将它与新运行的源码摘要合并。

## 范围与结果入口

| 层级 | 输入与实际路径 | 计时边界 | 已有结果 |
| --- | --- | --- | --- |
| 算子 | Synthetic 4K/16K/32K/64K；冻结的真实 L0/L15/L31 输入；resident attention / pooled score | Graph 每次调用、eager API；输入生成、压缩和选块在计时外 | [算子结果](#算子结果) |
| 完整模块 | 同一冻结 64K+1K 输入；`NosaIndexer` / `NosaSparseAttention` 的 native、Triton 对照 | 完整 API 与独立 profile kernel 时间；不含 prefix 构建、K/V/CIS 写入、cache 分配或事务 begin/abort | [完整模块结果](#完整模块结果) |
| 完整 sparse 模型 | NOSA-8B 全部 32 层，H64K+A1024，ordinary owned resident K/V/CIS | 无 profiler 的 full-prefill / extend 墙钟；独立 nsys 模块归因 | [完整 sparse 模型结果](#完整-sparse-模型结果) |
| 完整 dense 模型 | 相同模型规模与 token 边界，FlashInfer dense full attention | 独立无 profiler 进程的墙钟；另行采集 detailed/light 区间 | [完整 dense 模型结果](#完整-dense-模型结果) |

本轮 native 真实输入的有效 MFU 如下，覆盖 L0/L15/L31。算子以 Graph 每次调用为
分母，完整模块以独立 profile 的 kernel duration 为分母。

| 对象 | 有效 MFU | 来源 |
| --- | --- | --- |
| `block_sparse_attention` 算子 | 40.485%–41.748% | `refactor_mfu_real_bench_20261005_01` |
| `pooled_scores` 算子 | 20.430%–20.544% | `refactor_mfu_real_bench_20261005_01` |
| 完整 `NosaIndexer` | 24.420%–25.430% | `refactor_mfu_modules_native_profile_20261005_01` |
| 完整 `NosaSparseAttention` | 39.630%–40.646% | `refactor_mfu_modules_native_profile_20261005_01` |

这些 resident 结果不覆盖固定 P/NH、用户 LRU 或 DRAM backing，也不评价模型质量。
A128 固定容量 serving 的效率见 [motivation](../nosa_motivation/README.md)，
搬运与 attention 融合的单层对照见 [offload overlap](../nosa_offload_overlap/README.md)。
H64K+A1024 的本报告结果不能代替这两种执行路径的验收。

## 当前运行入口

所有命令从仓库根目录执行。脚本默认 `--mode bench`，要求提供匹配的
`--validation-receipt`；数值检查、无 profiler 基准和诊断分别运行：

- `check`：生成独立数值验收依据。算子检查全部 query 的 FP32 reference 及 graph/eager
  一致性；完整模块另检查增量压缩、选择、缓存回滚和重复调用；完整模型检查 full/split
  的全部 candidate hidden，sparse 另核对插桩前后输出。
- `bench`：核验 receipt 后预热、计时；样本间不重跑数值参考，不自动启动 profiler。
- `profile`：单独采集 trace 或模块分解。Dense/sparse 模型脚本会分别启动 benchmark
  和诊断进程，诊断时间不能充当正式 API 或端到端延迟。

Receipt 绑定源码、native 构建、设备与软件环境、输入内容、形状、stride、dtype、
后端及 cache/graph 路径。改变覆盖条件须重做验收；调整重复次数或 MFU 峰值不触发
完整数值检查，算子 graph 的 `graph_calls` 仍属于验收配置。真实输入继续使用冻结
capture，不代表当前 motivation 的实际轨迹。

`check` 的数据、日志和 receipt 留在系统临时目录，成功后打印位置；模型检查可用
`--check-dir` 指定实验目录之外的位置。正式结果保存 receipt 副本并注明
`independent_check`，不把复用依据写成本轮重新验收。运行时 finite 检查、numerical
repair、cache 事务和异步等待仍保留在实际执行路径内。

### 算子与完整模块

```bash
# Synthetic resident operators: independent acceptance, then clean measurement.
bash experiments/nosa_mfu/scripts/run.sh kernel_check \
  --mode check --peak-tflops 989 --validation-receipt /tmp/nosa-kernel-check.json
bash experiments/nosa_mfu/scripts/run.sh kernel_bench \
  --peak-tflops 989 --validation-receipt /tmp/nosa-kernel-check.json
bash experiments/nosa_mfu/scripts/run.sh kernel_profile \
  --mode profile --peak-tflops 989 --validation-receipt /tmp/nosa-kernel-check.json

# Complete modules use a separate receipt for each backend and input configuration.
nosa_inputs=experiments/nosa_mfu/output/data/kda_inputs_baseline_20260928_1345
bash experiments/nosa_mfu/scripts/modules.sh modules_check \
  --mode check --input-dir "$nosa_inputs" --kernel-backend native --peak-tflops 989 \
  --validation-receipt /tmp/nosa-modules-native-check.json
bash experiments/nosa_mfu/scripts/modules.sh modules_bench \
  --input-dir "$nosa_inputs" --kernel-backend native --peak-tflops 989 \
  --validation-receipt /tmp/nosa-modules-native-check.json
bash experiments/nosa_mfu/scripts/modules.sh modules_profile \
  --mode profile --input-dir "$nosa_inputs" --kernel-backend native --peak-tflops 989 \
  --validation-receipt /tmp/nosa-modules-native-check.json
```

算子、模块脚本将 `check` 产物留在系统临时目录，可用 `--validation-receipt` 指定收据
位置。固定真实输入的算子测量在 `scripts/run.sh` 后增加同一 `--input-dir`；
`scripts/capture.sh` 可从独立空 cache 重新采集模型输入；下方结果沿用冻结输入，没有重新采集。

### 完整模型

模型 check 还执行 attention 形状审计。Bench/profile 通过
`--validation-receipt` 复用检查证据，核对请求、模型配置、checkpoint 身份、运行源码、
设备/依赖、矩阵乘法精度设置、dispatch 环境及预热后实际映射的 Torch/BLAS/FlashInfer/
项目 native `.so` 内容；改变重复次数不要求重新数值检查。二进制指纹在预热后、计时前
采集，运行结束再核对映射与文件字节；不把包版本相同当作二进制相同。该指纹不包含
driver 生成的设备代码。源码快照、metadata 写入和资源释放全部成功后才发布
check receipt。Receipt 不替代模型内 finite 决策、numerical repair、cache 事务或
异步等待，这些运行逻辑仍在实际路径中计时。

```bash
bash experiments/nosa_mfu/scripts/dense.sh dense_check --mode check \
  --check-dir /tmp/nosa_dense_check
bash experiments/nosa_mfu/scripts/dense.sh dense_bench \
  --validation-receipt /tmp/nosa_dense_check/receipt.json
bash experiments/nosa_mfu/scripts/dense.sh dense_profile --mode profile \
  --validation-receipt /tmp/nosa_dense_check/receipt.json

bash experiments/nosa_mfu/scripts/sparse.sh sparse_check --mode check \
  --kernel-backend native --check-dir /tmp/nosa_native_check
bash experiments/nosa_mfu/scripts/sparse.sh sparse_bench --kernel-backend native \
  --validation-receipt /tmp/nosa_native_check/receipt.json
bash experiments/nosa_mfu/scripts/sparse.sh sparse_profile --mode profile \
  --kernel-backend native --validation-receipt /tmp/nosa_native_check/receipt.json
```

默认 checkpoint `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`、设备 `cuda:0`、每阶段 warmup=2、
repeats=5，sparse profile-repeats=1。比较两后端时使用相同 `--request-file`，分别
执行对应 `--kernel-backend native|triton` 的 check；同一 receipt 不跨后端复用。
可显式设置 `CUDA_VISIBLE_DEVICES`、CPU/NUMA 绑定和线程数，实际配置记录在 metadata。

所有脚本支持 `--help`。Dense 和 sparse 使用 `scripts/dense.sh`、`scripts/sparse.sh`；
sparse 的 `--without-nsys` 与 `--timeline-only` 只在 `--mode profile` 下有效。
CIS 的 ncu 专项诊断留在 `scripts/sparse_bottleneck.sh`，不会随普通 benchmark 自动执行。
运行依赖 SM90/Hopper、PyTorch、FlashInfer、Triton、TVM FFI、nvcc、共享 CUTLASS、
safetensors 和 tokenizer；profile 另需 nsys，CIS 诊断另需 ncu。

主指标为同步墙钟中位数及范围；host submit 与 CUDA 区间重叠，不能相加。Module MFU
以独立 profile 的 correlated kernel duration 为分母；event 区间含 launch gaps，
不等于纯 kernel 时间。Useful FLOPs 只计逻辑矩阵工作，所有实际 prepare、repair、
recompute 和 padding 的执行成本保留在分母。MFU 默认采用 H200 BF16 dense 参考峰值
989 TFLOPS，其他硬件需显式指定 `--peak-tflops`；不是 2:4 稀疏峰值或持续实测峰值。

## 算子与完整模块 MFU

独立测量 `block_sparse_attention` 和 `pooled_scores` 的 API 延迟及有效矩阵 MFU，
并测量完整 `NosaIndexer` 和 `NosaSparseAttention` 的 API 成本。保留原测量使用的
30% MFU 参照，不将其作为独立研究目标。输入的 K/V/CIS 全部驻留 HBM。

### 输入与 MFU 口径

形状为 BF16、1024 queries、32 Q heads / 2 KV heads、D128。Q 保持模型
`[1024,4608]` QKV packed view 的 stride；K/V 包含 prefix 和当前 queries。
Synthetic prefix 为 4096、16384、32768、65536，K/V 使用标准正态分布，CIS 为
标准正态乘 0.1，seed 为 `42 + prefix`。压缩采用 32-token / stride-16 窗口，
选择采用完整 NOSA 33/64 policy。输入生成、压缩和选择在单算子计时外。

真实输入来自 `kda_inputs_baseline_20260928_1345`，保留当时 sparse 轨迹的
Q/K/V/CIS、压缩 K 和实际 selection；本轮没有重新采集模型轨迹。
每组 native / Triton 对照使用相同输入和选择。Synthetic 选择分布不能代替真实模型的
跨 query 重合度，两组输入的性能分别报告。

单算子均预热 10 次、重复 30 次，报告中位数及完整样本：

- `graph`：一个 CUDA Graph 连续执行 10 次同输入算子，CUDA event 总时间除以 10。
  包含全部算子 kernels 与 graph 间隙，重用捕获时的输入和存储。
- `eager`：一次公开 Python 算子调用的 CUDA event 区间，包含 host 提交空档。

Attention 输出和 native score 临时空间在 API 内分配。编译、数值检查、预热在计时外。
本轮使用 `--reference-all` 检查全部 1024 个 query；每种后端的 Graph 与 eager 输出
须精确相同。Attention 按现有 BF16/CIS 契约使用 `rtol=atol=0.016`，同时记录
`atol=0.001, rtol=0.016` 的超限元素数、绝对误差和相对 L2。数值检查只覆盖测量输入，
不替代模块数值回归或不同模型轨迹的等价性验收。

MFU = 有效矩阵 FLOPs /（时间 × 标称 dense BF16 Tensor Core 峰值）。本轮参考峰值为
989 TFLOP/s；attention 仅计选中因果 token 的 QK+AV，score 的完整因果压缩窗口 QK
只计一次。重算、padding、masked future、softmax、GQA 求和、pooling 与访存不增加
分子，耗时仍计入分母。64K+1K 每层 attention / score 分别为 68,190,994,432 /
34,616,115,200 FLOPs；达到 30% 分别要求时间不超过 229.831 / 116.670 µs。

### 已发布环境与源码

本轮真实输入算子、模块和完整模型使用物理 GPU 1、逻辑 `cuda:0`，CPU 16–23、
内存 NUMA 0，OMP/MKL/OpenBLAS 线程均为 8。GPU UUID 为
`a2226185-cb05-a411-80da-f365154128fe`，CUDA 名称 NVIDIA H200，SM90、132 SM。
MFU 分母为显式指定的 989 TFLOP/s BF16 dense 参考峰值。完整设备、软件、编译环境、
启动命令、独立 check receipt 和源码快照随各 run 保存，见[发布来源](report/publication.json)。

算子与完整模块使用同一冻结真实输入，采集覆盖的 123 份源码相同。完整模型使用各自
check 绑定的源码和运行配置；源码相等按实验路径检查，不要求不同实验的收集范围一致。
每个 timing/profile 阶段按独占窗口调度，并离散采样全部 GPU 的计算进程；采样未见
外部进程。离散观测不能排除采样间的短时活动，实际窗口还需由独立 UTC 区间审计核对。

保留 synthetic 的原运行使用 GPU 5、CPU 48–55、NUMA 1，其 UUID 为
`a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`。它的 metadata、命令和源码快照保持原样，
不能与本轮 GPU 1 的测量合成同一批结果。

### 算子结果

Synthetic run 为 `kernel_flags_synthetic_20261004_01`。下表是 Graph 中每次算子的
中位延迟；加速比为 Triton / native 配置。4K / 16K score 的 native 配置实际仍调用
Triton，不计算 native kernel 加速比。

| Prefix | 算子 | native 配置实际后端 | native Graph µs | Triton Graph µs | 加速比 | native 配置 MFU |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | attention | CUDA/CuTe | 154.485 | 567.603 | 3.67× | 44.632% |
| 4096 | pooled scores | Triton | 27.362 | 27.334 | — | 8.883% |
| 16384 | attention | CUDA/CuTe | 905.872 | 577.890 | 0.64× | 7.611% |
| 16384 | pooled scores | Triton | 80.349 | 80.176 | — | 11.132% |
| 32768 | attention | CUDA/CuTe | 828.885 | 585.093 | 0.71× | 8.318% |
| 32768 | pooled scores | CUDA/CuTe | 95.664 | 148.850 | 1.56× | 18.429% |
| 65536 | attention | CUDA/CuTe | 830.096 | 586.266 | 0.71× | 8.306% |
| 65536 | pooled scores | CUDA/CuTe | 169.797 | 285.605 | 1.68× | 20.614% |

Synthetic attention 只在 4K prefix 达到 30% MFU；16K、32K、64K 的 native 路径
均慢于 Triton。32K / 64K native score 快于 Triton，但仍未达到 30%。

本轮固定真实输入 bench 为 `refactor_mfu_real_bench_20261005_01`，独立 profile 为
`refactor_mfu_real_profile_20261005_01`，三层均为 H64K+A1024：

| Layer | 算子 | native 实际后端 | native Graph µs | Triton Graph µs | 加速比 | native MFU |
| --- | --- | --- | --- | --- | --- | --- |
| L00 | attention | CUDA/CuTe | 170.309 | 574.534 | 3.37× | 40.485% |
| L00 | pooled scores | CUDA/CuTe | 170.731 | 284.787 | 1.67× | 20.501% |
| L15 | attention | CUDA/CuTe | 168.728 | 580.355 | 3.44× | 40.864% |
| L15 | pooled scores | CUDA/CuTe | 171.322 | 282.440 | 1.65× | 20.430% |
| L31 | attention | CUDA/CuTe | 165.157 | 579.712 | 3.51× | 41.748% |
| L31 | pooled scores | CUDA/CuTe | 170.373 | 282.045 | 1.66× | 20.544% |

三层 native attention 的 Graph MFU 均超过 40%，pooled score 约为 20.4%–20.5%，
仍低于 30% 参照。12 个结果行复用独立 check 对全部 1024 个 query 的 FP32-reference
验收，以及 Graph/eager 精确一致性检查；bench 每种计时方式均保留 30 个样本。
独立 profile 的 12 份 trace 单独保存，不用于替换主 Graph/eager 延迟。

[算子对照 CSV](report/operator_comparison.csv)、[真实输入全部样本](report/captured/results.json)、
[真实输入元数据](report/captured/metadata.json)和[独立 profile](report/captured/profile/results.json)
记录本轮结果；[synthetic 全部样本](report/synthetic/results.json)继续保留历史身份。

### 完整模块结果

Native/Triton 各自运行独立 `check`、`bench` 和 `profile`。Bench ID 为
`refactor_mfu_modules_native_bench_20261005_01` 和
`refactor_mfu_modules_triton_bench_20261005_01`；profile 使用对应的 `_profile_` ID。
每次调用从相同的已提交 64K prefix 开始，调用后回滚 candidate 派生状态。
计时包含 Indexer 的有限值校验、增量压缩、排名和选块；prefix 构建、K/V/CIS 写入、
cache 分配及事务 begin/abort 在计时外。各组预热 10 次，bench 重复 30 次。

| Layer | 模块 | native event µs | Triton event µs | event 加速比 | native wall µs | Triton wall µs |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | indexer | 273.856 | 734.288 | 2.68× | 279.665 | 740.515 |
| 0 | attention | 243.296 | 635.744 | 2.61× | 249.022 | 641.708 |
| 15 | indexer | 265.536 | 722.608 | 2.72× | 271.266 | 728.646 |
| 15 | attention | 243.728 | 636.048 | 2.61× | 249.760 | 642.027 |
| 31 | indexer | 264.576 | 723.152 | 2.73× | 270.673 | 729.480 |
| 31 | attention | 239.360 | 638.752 | 2.67× | 245.244 | 644.763 |

独立 profile 每个模块采集 3 次。下表为 kernel duration 之和的中位数，包含 native
attention 的 prepare、sort 和 repair，不含 Python 空档与 D2H copy。

| Layer | 模块 | native kernel µs | native MFU | Triton kernel µs | Triton MFU |
| --- | --- | --- | --- | --- | --- |
| 0 | indexer | 143.329 | 24.420% | 499.134 | 7.012% |
| 0 | attention | 173.983 | 39.630% | 575.042 | 11.990% |
| 15 | indexer | 137.855 | 25.390% | 488.413 | 7.166% |
| 15 | attention | 172.450 | 39.982% | 576.257 | 11.965% |
| 31 | indexer | 137.638 | 25.430% | 490.882 | 7.130% |
| 31 | attention | 169.632 | 40.646% | 577.857 | 11.932% |

Native 完整 indexer 的 kernel MFU 为 24.42%–25.43%，仍低于 30%；attention 为
39.63%–40.65%。完整 API 的成本另见上表，不能用 kernel 时间替代。

独立 check 核验增量压缩、完整 indexer 选择、cache 回滚与重复输出；相对冻结 capture
的 selection 变化数为零。每个后端保留 18 份独立 trace。该测量不包含 CIS 投影，
也不代表模型端到端或 offload 性能。

[模块对照 CSV](report/module_comparison.csv)、[native bench](report/modules/native/results.json)、
[native profile](report/modules/native/profile/results.json)、
[Triton bench](report/modules/triton/results.json)和
[Triton profile](report/modules/triton/profile/results.json)分别保留原始样本。

## 完整模型 resident MFU

完整模型结果包括原 `nosa_gr_65536_1024` 的 dense 前向和
`indexer_block_sparse_profile` 的 native/Triton sparse 前向。

### 输入、模型与执行边界

使用 NOSA-8B 全部 32 层，BF16，32 Q heads / 2 KV heads，D128。输入由共享 GR
生成器产生，instruction 28 + history 65508 组成 65536-token prefix，candidate
suffix 为 1024 tokens，总长度 66560。prefix 与 full-prefill 按 1024-token chunk
执行，candidate 整批执行。GR 的 item budget=1052 包含 instruction；实际执行
边界为 `[0,65536)` 与 `[65536,66560)`。

| 路径 | Attention | 输出 | Cache |
| --- | --- | --- | --- |
| dense | FlashInfer dense full attention | normalized hidden | ordinary owned resident KV |
| sparse native | 原 NOSA indexer + native FA3 | normalized hidden | ordinary owned resident K/V/CIS |
| sparse Triton | 相同 NOSA policy 的 Triton control | normalized hidden | ordinary owned resident K/V/CIS |

两阶段分别测从空 cache 处理完整 66560 tokens 的 `full_prefill`，以及 prefix 已完成后
处理 1024 candidate tokens 的 `extend`。计时包含模型计算、cache 写入、host 提交和
CUDA 完成等待；加载权重、输入生成/H2D、cache 分配、reset 与 extend prefix 构建
在计时外。输出不含 LM head、自回归采样。上下文仅在实验进程扩为 66560，保留
checkpoint 的 LongRoPE factors，不由此评价超出原 32768 上下文后的模型质量。

这里没有固定 P/NH、有限历史槽位、DRAM backing、offload、用户 LRU 或 motivation
的 compute graph。H64K+A1024 resident 结果不能证明其他 candidate 长度或
固定 serving 路径已经高效。模型与权重加载调用 [NOSA 实现](../../models/nosa/README.md)。

### Sparse 后端与执行路径

两组均通过 `IndexerCache` 复用已提交的压缩前缀，只追加完整的 32-token / stride-16
K/CIS 窗口和稳定 CIS pool，校验当前 Q 与尚未校验的 K/CIS 后缀。这些开销均纳入计时。
CUDA indexer 一次处理完整 query batch，`indexer_query_chunk_size=null`。

| 项目 | native | Triton control |
| --- | --- | --- |
| `kernel_backend` | `cuda_tvm_ffi` | `triton` |
| `indexer_execution` | `cached_native_v5` | `cached_flashinfer_v1` |
| `indexer_preparation` | `native_guarded_ranked_checked_v1` | `triton_v1` |
| `selection_backend` | `cuda_tvm_ffi` | `flashinfer` |
| `attention_execution` | `native_fa3_v3` | `triton_v1` |

Native 大形状 indexer 使用 CUDA/CuTe WGMMA，两遍 QK 中融合归一化、GQA、BF16 舍入、
五窗口 pooling 和完整选块；checked C++ 入口合并有限值校验、增量压缩、共享 CIS 排名
及后续提交。64K prefix + 1K query 的目标形状还以向上舍入的 FP16 上界精确剪枝第二遍
QK，仅跳过可证明低于精确 cutoff 的 tile，并使用 BF16 成对转换。更短的已打分 chunk
保留 Triton scoring 或其他已验收路径；短上下文可直接选择全部因果块。实际调度通过
逐层/chunk kernel 序列核验，不能把整段 full-prefill 当成单一 score kernel。

Native attention 使用 FlashInfer 0.6.18 的 FA3 Hopper 模板，八个相邻 query 组成选块
并集，KV128、双 stage；Q 通过 TMA 读取原始 stride，epilogue 直接写最终输出，每个
query 独立应用 membership、CIS 和因果遮罩。本实验每次调用的完整路径为：并集准备 →
按并集 tile 数排序工作 → FA3 → native per-query repair，四个 kernel 的全部时间均计入
attention。Repair launch 数量不代表实际修复的 query 数量。

Triton control 使用现有两遍 fused QK/pooling、FlashInfer Top-33/Top-64 和 Triton
block sparse attention。Native 的 score 与 selection 已融合，不能以 `pooled_scores`
对子项做 score-only 加速比；有效对照是完整 `indexer_total`。

Native 通过 TVM FFI 编译 `operators/nosa/` 下各功能的 `csrc/`，复用共享 CUTLASS 和已安装的
FlashInfer headers，不加载 EzKernelKit。BF16 pooled-score scratch 为 4.0625 MiB，QA/CIS
及模型各层复用；kernel 临时空间不计入请求 resident KV cache 容量。旧模型接口
`sparse_backend="triton"` 仍指 SM90 dispatcher，实际后端以上表和 metadata 为准。

### 完整 sparse 模型结果

主 bench 为 `refactor_mfu_sparse_native_bench_20261005_01` 和
`refactor_mfu_sparse_triton_bench_20261005_01`。两后端读取相同请求，SHA256 为
`65d62f9ee9662fbe4b1d95d8c5440f16e70336a10b1eb2c860c55d8af3176b6b`。
两份独立 check 均核验 full/split 的全部 1024×4096 candidate hidden，输出有限且
最大绝对差为 0，并核对插桩前后输出。对应 profile 使用各自的 `_profile_` ID。

每阶段 warmup=2、repeats=5。下表来自独立无 profiler 的主 bench，单位 ms：

| 阶段 | Triton wall ms [min, max] | native wall ms [min, max] | Triton MFU | native MFU | 加速比 |
| --- | --- | --- | --- | --- | --- |
| full_prefill | 3663.113 [3659.398, 3665.236] | 2374.539 [2369.990, 2379.862] | 32.66% | 50.38% | 1.543× |
| extend | 61.776 [61.683, 61.809] | 36.208 [36.127, 36.612] | 30.80% | 52.56% | 1.706× |

有效矩阵工作量为 full-prefill `1,183,130,731,937,792` FLOPs、extend
`18,820,462,804,992` FLOPs。加速比只比较本轮相同输入的 native/Triton 配置；
不同日期、GPU 和接口整理之间的变化不能单独归因为某项代码优化。

每后端每阶段独立采集一次 nsys profile。下表是 correlated kernel duration 的总和；
一次采集不估计方差，也不能与主墙钟相加或相减。

| 模块 | full Triton ms | full native ms | 加速比 | extend Triton ms | extend native ms | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| `indexer_total` | 676.637 | 290.297 | 2.331× | 15.797 | 4.858 | 3.252× |
| `block_sparse_attention` | 1192.128 | 365.083 | 3.265× | 18.564 | 5.802 | 3.200× |

Native 各模块的 kernel 时间与有效 MFU 如下。Indexer 的完整分母包含 prepare、
有限值检查、压缩、评分和选块；attention 包含 prepare、sort、FA3 与 repair。

| 模块 | full kernel ms | full MFU | extend kernel ms | extend MFU |
| --- | --- | --- | --- | --- |
| `qkv_proj` | 104.894 | 77.5031% | 1.602 | 78.0896% |
| `o_proj` | 93.705 | 77.1180% | 1.406 | 79.0901% |
| `gate_up_proj` | 719.294 | 80.3712% | 10.984 | 80.9709% |
| `down_proj` | 351.218 | 82.3000% | 5.315 | 83.6615% |
| `block_sparse_attention` | 365.083 | 38.0837% | 5.802 | 38.0283% |
| `indexer_total` | 290.297 | 12.5848% | 4.858 | 23.0548% |
| `cis_projection` | 26.206 | 0.0084% | 0.398 | 0.0085% |

Extend 完整 indexer/attention 的 MFU 分别为 23.05%/38.03%，两者未同时达到 40%。
本结论覆盖普通 owned resident HBM 前向；固定 P/NH、DRAM backing 与 A128 serving
属于其他执行路径，分别见对应实验。

[后端比较](report/sparse/comparison.json)使用主 bench 墙钟与独立 profile 模块时间。
[Native timings](report/sparse/native/timings.csv)、[Triton timings](report/sparse/triton/timings.csv)、
[native 模块](report/sparse/native/module_mfu.csv)和[Triton 模块](report/sparse/triton/module_mfu.csv)
保留完整精度；profile wrapper 内额外 bench 的分母不替代本页主 bench。
完整层/chunk 记录、SQLite、原始 nsys、receipt 和源码快照保存在各 run 的 `output/` 中。

### 完整 dense 模型结果

本轮使用 `refactor_mfu_dense_bench_20261005_02` 与
`refactor_mfu_dense_profile_20261005_02`，复用独立
`refactor_mfu_dense_check_20261005_02` 的数值依据。全部 1024×4096 candidate hidden
有限，full/split 输出最大绝对差为 0；32 层 attention 形状均核对通过。
Dense 请求 SHA256 为
`0bbabf07bc72f9804dbc2e9644cf708e21eafe237631f4a56c64b20f9cda7f2f`；
它与 sparse 请求的 token 字节不同，不能将两者视为同请求的 dense/sparse 加速比。

| 阶段 | 无 profiler wall（ms） | Useful FLOPs | MFU |
| --- | --- | --- | --- |
| full_prefill | 3506.117727 | 2170865718394880 | 62.61% |
| extend | 81.806672 | 50990120173568 | 63.02% |

每阶段 warmup=2、repeats=5，wall 取五个独立样本的中位数。下表来自另一次进程的
Nsight detailed capture，分母为模块关联 kernel 时间之和；它不替代上表的端到端时间。

| 模块 | full-prefill kernel（ms） | MFU | extend kernel（ms） | MFU |
| --- | --- | --- | --- | --- |
| qkv_proj | 120.805365 | 67.30% | 1.847874 | 67.68% |
| o_proj | 110.512476 | 65.39% | 1.689538 | 65.80% |
| gate_up_proj | 817.348343 | 70.73% | 12.496844 | 71.17% |
| down_proj | 401.797615 | 71.94% | 6.141542 | 72.41% |
| attention_core | 1921.111988 | 61.13% | 57.653321 | 62.19% |

Profile 包含一次 light full-prefill、三次 light extend，以及 full-prefill/extend
各一次 detailed，共六个区间；完整分析保存在 [analysis.json](report/dense/analysis.json)。
[组合 MFU](report/dense/mfu.json)显式关联独立 bench 与 profile，
[原 bench MFU](report/dense/benchmark_mfu.json)、[原 profile MFU](report/dense/profile_mfu.json)、
[五样本分布](report/dense/timings.csv)和[模块表](report/dense/module_mfu.csv)分别保留。

首轮 dense profile 在采集前因 EOS tuple/list 的 Python 比较失败而停止。修复仅将
完整 execution identity 的比较改为已有 canonical digest，并保留 bench-mode 检查；
新 check、bench、profile 均在修复后独立运行。先前完成的 sparse、算子与模块采集仍保留
各自的源码快照，不因这处 gate 修复改写身份。交付源码差异见
[源码与导航变更](report/source_delta.json)，失败诊断不作为实验结果。

## 代码、产物与复现

| 入口 | 作用与调用模块 |
| --- | --- |
| `scripts/run.sh` / `src/measure.py` | 算子验收、计时与 profile；调用 `models.nosa.indexer` 的压缩/FP32 reference、`operators.nosa.indexer.api` 和 `operators.nosa.attention.device_only.api` |
| `scripts/modules.sh` / `src/measure_modules.py` | 完整 `NosaIndexer` / `NosaSparseAttention` 验收、计时与 profile |
| `scripts/capture.sh` / `src/capture_inputs.py` | 从独立空 cache 采集真实 sparse 模型输入 |
| `scripts/dense.sh` / `src/dense/` | 完整 dense 模型 capture、NVTX、SQLite 分析和 FLOPs 计数 |
| `scripts/sparse.sh` / `src/sparse/` | 完整 sparse 模型 capture、模块归因、MFU 和后端比较 |
| `scripts/sparse_bottleneck.sh` | CIS 的独立 ncu 诊断 |

模型前向调用 `executor.model_executor.run_chunks` 及 `models.nosa.model`。
数值证据绑定由 `src/acceptance.py`、`src/phases.py` 和 `evaluation/validation.py` 提供；
算子与模块 FLOPs 复用 `src/sparse/mfu.py` 的 `work_counts`，来源记录共用
`src/dense/sources.py` 的 `source_hashes`。其他实验还显式复用 `src/dense/capture.py`
的 `execution_split` 和 `src/dense/mfu.py` 的 `matrix_flops`。

Sparse 模块归因验证已登记的源码图，并在本轮 profile 中核对实际活动；未知源码身份
仍会拒绝。普通 resident GEMM 顺序及 useful FLOP 口径保持各自实现定义。

算子与模块脚本先写系统临时目录，全链成功后发布到新的
`output/{data,log,profile}/<run_id>/`；源码中途变化或 run ID 已存在会失败。
stdout/stderr 分开保存，module Chrome trace 和 nsys 原始产物放在 `output/profile/`，
样本、源码快照、SQLite 与环境记录放在 `output/data/`。

算子与模块报告位于 `report/` 根目录及 `synthetic/`、`captured/`、`modules/`；
完整模型报告位于 `report/dense/` 和 `report/sparse/`。每份报告的 selected-file 清单、
独立数值依据与采集源码身份见[发布来源](report/publication.json)。运行 metadata 中的
历史输入路径、命令与源码快照保持原样；当前导航单独指向保留的 byte-identical 请求。

完整产物位于 `output/{data,log,profile}/<run_id>/`。CPU 重建先写新的系统临时目录，
不得覆盖已发布 run。报告生成与复核脚本保存在本轮主 run 的 `report_generation/`，
具体调用与输入身份由 publication 记录。

保留 synthetic 的原测量命令如下，记录当时的旧目录与无独立 receipt 入口；当前
复现使用前文 `experiments/nosa_mfu/` 的分阶段命令。

```bash
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=48-55 --membind=1 \
  bash experiments/nosa_kernel_mfu/scripts/run.sh kernel_synthetic_rerun \
  --device cuda:0 --peak-tflops 989 --reference-all
```
