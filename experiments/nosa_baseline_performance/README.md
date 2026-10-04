# NOSA baseline 性能检查

本实验检查完整 NOSA checkpoint 的 ordinary resident baseline 是否高效，将原
`nosa_gr_65536_1024` 的 dense 前向和 `indexer_block_sparse_profile` 的 native/Triton
sparse 前向合并维护。它属于 baseline 性能检查，不承担固定 P/NH motivation 或
自有 offload kernel microbench 的任务。

2026-10-05 仅整理入口、拆分数值检查与性能测量，没有运行新实验。下方保留的是
`dense_wrapper_20261004_01`、`sparse_flags_native_20261004_01` 和
`sparse_flags_triton_20261004_01` 的原结果。它们对应各自保存的实现快照；新的入口、
独立 check receipt 和 dense 无 profiler 基准路径均为改动后未运行，不能把旧数字
当作新入口的性能或数值验收。

## 输入、模型与适用范围

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
的 compute graph。旧 H64K+A1024 resident 结果也不能证明其他 candidate 长度或
固定 serving 路径已经高效。模型与权重加载调用 [NOSA 实现](../../models/nosa/README.md)。

## 独立数值检查、基准与诊断

脚本默认 `--mode bench`：只做充分预热和无 profiler 的性能测量，再做 CPU 汇总。
`--mode profile` 显式运行独立 benchmark 进程与第二个诊断进程，生成 nsys/SQLite
或 CUDA-event 模块分解；默认 benchmark 不再自动启动这条链。

`--mode check` 单独执行原有 full/split 数值对照和 attention 形状审计。Dense 和 sparse 均比较全部 candidate hidden；sparse 另验证 instrumentation
不改变输出。检查产物和 receipt 放在实验目录之外。Bench/profile 通过
`--validation-receipt` 复用检查证据，核对请求、模型配置、checkpoint 身份、运行源码、
设备/依赖、矩阵乘法精度设置、dispatch 环境及预热后实际映射的 Torch/BLAS/FlashInfer/
项目 native `.so` 内容；改变重复次数不要求重新数值检查。二进制指纹在预热后、计时前
采集，运行结束再核对映射与文件字节；不把包版本相同当作二进制相同。该指纹不包含
driver 生成的设备代码。源码快照、metadata 写入和资源释放全部成功后才发布 check receipt。Receipt 不替代模型内 finite
决策、numerical repair、cache 事务或异步等待。这些运行逻辑仍在实际路径中计时。

从仓库根目录运行以下入口；这些是后续使用方式，本轮未执行：

```bash
bash experiments/nosa_baseline_performance/scripts/dense.sh dense_check --mode check \
  --check-dir /tmp/nosa_dense_check
bash experiments/nosa_baseline_performance/scripts/dense.sh dense_bench \
  --validation-receipt /tmp/nosa_dense_check/receipt.json
bash experiments/nosa_baseline_performance/scripts/dense.sh dense_profile --mode profile \
  --validation-receipt /tmp/nosa_dense_check/receipt.json

bash experiments/nosa_baseline_performance/scripts/sparse.sh sparse_check --mode check \
  --kernel-backend native --check-dir /tmp/nosa_native_check
bash experiments/nosa_baseline_performance/scripts/sparse.sh sparse_bench --kernel-backend native \
  --validation-receipt /tmp/nosa_native_check/receipt.json
bash experiments/nosa_baseline_performance/scripts/sparse.sh sparse_profile --mode profile \
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
recompute 和 padding 的执行成本保留在分母。默认采用 H200 BF16 dense 参考峰值
989 TFLOPS，其他硬件需显式指定 `--peak-tflops`；不是 2:4 稀疏峰值或持续实测峰值。

## 代码与产物

`src/dense/` 维护 dense capture、NVTX、SQLite 分析和 FLOPs 计数；`src/sparse/`
维护 sparse capture、模块归因、MFU 和后端比较。二者调用 `executor.model_executor.run_chunks`
及 `models.nosa.model`，共享数值证据绑定位于 `src/acceptance.py` 和
`evaluation/validation.py`。其他实验可显式复用 `src/dense/sources.py`、
`src/dense/capture.py` 的 `execution_split` 及 `src/dense/mfu.py` 的 `matrix_flops`。

原报告字节分别保留在 `report/dense/`、`report/sparse/`，完整运行产物迁至本目录
`output/{data,log,profile}/<原 run ID>/`。原 metadata、source snapshots、JSON 路径和
run ID 不改写；其中旧路径是历史记录，当前路径按本段映射。迁移逐文件 SHA256 清单在
`output/data/layout_migration_20261005_01/migration.json`，它是目录整理记录，不是实验运行。
下方历史命令同样原样保留，其旧入口已删除；当前执行使用前节命令。

Sparse 模块归因仍保留历史 source graph 白名单；新增当前迁移路径及 reserved-workspace
adapter 修订的显式登记。普通 resident GEMM 顺序没有因此改变；未知源码身份继续拒绝。
这一静态登记不表示已运行新的 GPU profile。

## Sparse 后端与计时边界

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

## 已有 dense 结果（2026-10-04）

本次 run 为 `dense_wrapper_20261004_01`，已完成全部 32 层前向、nsys 捕获、SQLite 导出和分析。
实际命令使用物理 GPU 5、CPU 48–55、内存 NUMA 1，OMP/MKL/OpenBLAS 与 PyTorch intra-op 均为 8：

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONDONTWRITEBYTECODE=1 numactl --physcpubind=48-55 --membind=1 \
  bash experiments/nosa_gr_65536_1024/scripts/run.sh dense_wrapper_20261004_01
```

PyTorch 设备名为 `NVIDIA H200`，NVML 为 `NVIDIA M403`；对应 UUID 均为
`a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`，SM90、132 SM，PCI 地址 `0000:ab:00.0`。
驱动 570.124.06，PyTorch 2.12.1+cu130 / CUDA build 13.0，FlashInfer 0.6.18；
Nsight Systems 2025.6.3.541，MFU 采用 989 TFLOPS 的 BF16 dense 参考分母。
原始设备名、全部依赖和启动环境保存在 metadata 与 launch_environment。

输入为 65536-token prefix 与 1024-token suffix，BF16、chunk=1024。每阶段预热 2 次、
基准重复 5 次。全部 32 层 extend 形状与前述边界一致；full 与 prefix+extend 的末 token
hidden 均有限，最大绝对差为 0、余弦相似度为 1。请求原始字节与前次 dense 完全一致。
该检查覆盖本次 dense 拆分执行，不扩展为全部 hidden 或模型质量验收。

### 采集关闭区间的延迟与整段 MFU

时间来自 nsys 进程内、CUDA profiler capture 尚未启动的基准阶段。nsys 注入仍可能
影响计时，本次没有独立的无 profiler 进程。墙钟包括提交与完成等待；host submit
可能包含 CUDA API 内部等待，不能与 GPU 时间相加。

| 阶段 | 墙钟中位数 ms [min, max] | Host submit 中位数 ms | 整段 MFU |
| --- | --- | --- | --- |
| full_prefill | 3540.284 [3530.417, 3540.812] | 3323.058 | 62.00% |
| extend | 82.793 [81.950, 83.260] | 8.873 | 62.27% |

有效矩阵工作量为 full-prefill `2,170,865,718,394,880` FLOPs、extend `50,990,120,173,568` FLOPs。
整段 MFU 用上述墙钟；模块 MFU 用另一个 detailed 区间，两类时间不能相减来估计 launch 开销。

### 模块 GPU 时间与 MFU

每阶段只有一次 detailed 捕获。时间按 CUDA launch 关联到模块的 GPU 活动 duration 求和，
包含 kernel 与该模块的 MEMSET；非矩阵模块 MFU 不适用。

| 模块 | Full GPU ms | Full MFU | Extend kernel 数 | Extend GPU ms | Extend MFU |
| --- | --- | --- | --- | --- | --- |
| `embedding` | 0.303 | 不适用 | 1 | 0.005 | 不适用 |
| `input_layernorm` | 0.319 | 不适用 | 1 | 0.005 | 不适用 |
| `qkv_proj` | 122.093 | 66.59% | 32 | 1.855 | 67.43% |
| `rope_apply` | 15.854 | 不适用 | 32 | 0.244 | 不适用 |
| `kv_cache_and_layout` | 11.301 | 不适用 | 64 | 0.173 | 不适用 |
| `attention_core` | 1936.515 | 60.64% | 32 | 57.788 | 62.04% |
| `o_proj` | 111.503 | 64.81% | 32 | 1.690 | 65.78% |
| `post_attention_layernorm_add_residual` | 16.293 | 不适用 | 32 | 0.248 | 不适用 |
| `gate_up_proj` | 827.388 | 69.87% | 32 | 12.535 | 70.95% |
| `swiglu_elementwise` | 50.218 | 不适用 | 32 | 0.772 | 不适用 |
| `down_proj` | 405.931 | 71.21% | 32 | 6.144 | 72.38% |
| `input_layernorm_add_residual` | 15.235 | 不适用 | 31 | 0.233 | 不适用 |
| `final_norm_add_residual` | 0.496 | 不适用 | 1 | 0.007 | 不适用 |

每层 QKV、O、gate/up、down 各一次 GEMM，extend 共 128 个投影 kernel；gate/up
另有 32 次 MEMSET，其时间已计入。K/V 写入为每层两个 kernel，extend 共 64 个。
Detailed extend 共 354 个 kernel，attention core 为 57.788 ms，占模块活动
时间的 70.73%，是本次 dense 路径最大的单项开销。

### GPU 活动与提交排队

下表为 light 捕获。active 是 kernel/memcpy/memset 区间的并集；“API 前 gap”仅统计
GPU 已空闲且下一关联 CUDA API 尚未进入的部分。Launch-to-kernel 为 API 结束到 kernel
开始的等待中位数。

| Light 区间 | Kernel 数 | GPU span ms | GPU active ms | Gap ms | Active 占比 | API 前 gap ms | Launch-to-kernel 中位数 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_prefill/0 | 23010 | 3506.190 | 3477.056 | 29.135 | 99.17% | 0.476 | 141.310 |
| extend/0 | 354 | 80.053 | 79.299 | 0.754 | 99.06% | 0.280 | 34.195 |
| extend/1 | 354 | 80.952 | 80.329 | 0.623 | 99.23% | 0.158 | 34.696 |
| extend/2 | 354 | 82.856 | 82.214 | 0.642 | 99.23% | 0.182 | 35.647 |

三次 extend 的 GPU gap 为 0.623–0.754 ms，API 前 gap 为
0.158–0.280 ms；GPU active 约为 99.06%–99.23%。
已提交工作存在排队，支持本次主要时间花在 GPU 执行上的判断。不能把所有 gap 归因于
CPU launch，也不能将 active 占比解释为 SM occupancy。本实验未测 CUDA Graph 或 offload。

### 报告数据与复现

[metadata.json](report/dense/metadata.json)、[analysis.json](report/dense/analysis.json)、
[mfu.json](report/dense/mfu.json) 均从 `dense_wrapper_20261004_01` 原样复制。
[重算审计](report/dense/result_integrity.json)核验全部六个 NVTX 区间、kernel 与 MEMSET 活动守恒、
墙钟中位数及 MFU；95 份源码快照与 metadata 及审计时源码一致。

| 报告数据 | 生成方式 |
| --- | --- |
| [timings.csv](report/dense/timings.csv) | 从五次 timings、medians 和 MFU 生成，保留完整精度 |
| [module_mfu.csv](report/dense/module_mfu.csv) | detailed 模块活动按名字关联 FLOPs/MFU，非矩阵项留空 |
| [launch.csv](report/dense/launch.csv) | 全部六个 NVTX 区间的活动与提交指标，等待单位为微秒 |
| [provenance.json](report/dense/provenance.json) | 原始输入、选用文件与生成器哈希 |

完整请求、32 层形状、源码与 SQLite 保存在 `output/data/dense_wrapper_20261004_01/`，
原始 nsys 与日志分别在对应 `output/profile/`、`output/log/`。请求 SHA256 为
`0bbabf07bc72f9804dbc2e9644cf708e21eafe237631f4a56c64b20f9cda7f2f`。报告生成器与审计脚本保存在本 run 的 `report_generation/`。

```bash
.venv/bin/python -m experiments.nosa_baseline_performance.src.dense.analyze \
  experiments/nosa_baseline_performance/output/data/dense_wrapper_20261004_01/nsys.sqlite \
  --output /tmp/nosa-dense-analysis-rebuilt.json
```

MFU 入口会写入传入的数据目录；需要重建时先复制 metadata 与 analysis 到新的系统临时目录，
再运行 `python -m experiments.nosa_baseline_performance.src.dense.mfu <copied-data-dir> --peak-tflops 989`。

## 已有 sparse 结果（2026-10-04）

本次 native / Triton run 分别为 `sparse_flags_native_20261004_01` / `sparse_flags_triton_20261004_01`。
两组的 109 份源码、请求字节、checkpoint 文件记录、设备、依赖和预热/重复参数一致；
构建元数据仅允许 selected_backend 不同。请求 SHA256 为
`65d62f9ee9662fbe4b1d95d8c5440f16e70336a10b1eb2c860c55d8af3176b6b`。
Checkpoint 采用配置 SHA256 及权重文件路径、大小、mtime 核验。

### 实测环境

两组在物理 GPU 5 串行运行，进程内为 `cuda:0`。PyTorch 设备名为 `NVIDIA H200`，
NVML/Nsight 为 `NVIDIA M403`，UUID 均为 `a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`；
SM90、132 SM，CUDA 可见显存 150121545728 B。MFU 采用 989 TFLOPS 的参考分母。
CPU 固定 48–55、内存 NUMA 1，OMP/MKL/OpenBLAS 线程均为 8，PyTorch intra-op 为 8。
驱动 570.124.06；PyTorch 2.12.1+cu130 / CUDA build 13.0；Triton 3.7.1、FlashInfer 0.6.18、
TVM FFI 0.1.13.post3、nvcc V13.2.78，CUTLASS 为 `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。
实际启动环境保存在每组的 `launch_environment.json`，完整依赖与构建记录见 metadata。

### 独立端到端计时

每阶段 warmup=2、repeats=5；下表为未插桩独立进程的墙钟中位数与 `[min, max]`，单位 ms。

| 阶段 | Triton wall ms [min, max] | native wall ms [min, max] | Triton MFU | native MFU | 加速比 |
| --- | --- | --- | --- | --- | --- |
| full_prefill | 3678.514 [3676.505, 3680.259] | 2394.455 [2391.498, 2401.130] | 32.52% | 49.96% | 1.536× |
| extend | 62.075 [62.029, 62.094] | 36.752 [36.092, 37.126] | 30.66% | 51.78% | 1.689× |

有效矩阵工作量为 full-prefill `1,183,130,731,937,792` FLOPs、extend `18,820,462,804,992` FLOPs。
两组 full/extend 的全部 candidate hidden 有限且最大绝对差为 0，profile 不改变输出。
逐层 extend audit、完整层/chunk/kernel 序列和 GPU 映射通过；完整模块报告由原始 SQLite
重新计算，除派生时间戳和 staging 路径外与保存结果完全一致。相关 CPU 实验测试 735 项通过。
这些检查用于正确性和测量完整性，不用于模型质量结论。

### 完整模块 kernel 时间

每组每阶段仅采集 1 次独立 nsys profile。以下为 correlated kernel duration 总和，单位 ms；
它不估计多次采集的方差，也不能与主墙钟相加或相减。

| 模块 | full Triton | full native | 加速比 | extend Triton | extend native | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| `indexer_total` | 676.283 | 292.677 | 2.311× | 15.777 | 4.861 | 3.245× |
| `block_sparse_attention` | 1190.253 | 367.710 | 3.237× | 18.496 | 5.788 | 3.195× |

Native full-prefill 的 2080 个 layer/chunk 调用中，128 个不计算 score、864 个使用
Triton `pooled_scores`、1088 个使用融合 `score_selection`。其中 1056 个为常规融合，
32 个为精确剪枝；extend 的 32 个调用均为精确剪枝。融合 score/selection 没有与
Triton score-only 等价的边界，因此只比较完整 indexer。
Attention 的 full / extend 均为 FA3 v3，共 2080 / 32 次调用；四族全部计时如下。

| kernel 族 | full ms | full launches | extend ms | extend launches |
| --- | --- | --- | --- | --- |
| `prepare` | 6.609984 | 2080 | 0.102081 | 32 |
| `sort_work_by_union_size` | 3.425390 | 2080 | 0.050976 | 32 |
| `PrefillWithKVCacheKernel` | 349.682529 | 2080 | 5.540260 | 32 |
| `attention_kernel` | 7.992497 | 2080 | 0.094946 | 32 |

每层调用、根区间与完整 attention 模块的数量和时间完全守恒。未采集实际 repair query 数量，
repair launch 数不能解释为回退比例；四次 launch 也不使有效 QK/PV FLOPs 乘四。

### Native 模块 MFU 与 indexer 分解

| 模块 | full kernel ms | full MFU | extend kernel ms | extend MFU |
| --- | --- | --- | --- | --- |
| `qkv_proj` | 106.878 | 76.0644% | 1.628 | 76.8304% |
| `o_proj` | 95.056 | 76.0219% | 1.429 | 77.8125% |
| `gate_up_proj` | 726.513 | 79.5726% | 11.054 | 80.4578% |
| `down_proj` | 354.875 | 81.4520% | 5.365 | 82.8950% |
| `block_sparse_attention` | 367.710 | 37.8116% | 5.788 | 38.1182% |
| `indexer_total` | 292.677 | 12.4825% | 4.861 | 23.0392% |
| `cis_projection` | 26.175 | 0.0084% | 0.391 | 0.0087% |

| Native indexer 子项 | full kernel ms | extend kernel ms | full / extend kernel 数 |
| --- | --- | --- | --- |
| `compression_k` | 0.000000 | 0.000000 | 0 / 0 |
| `compression_cis` | 0.000000 | 0.000000 | 0 / 0 |
| `compressed_scores` | 0.000000 | 0.000000 | 0 / 0 |
| `select_from_scores` | 0.000000 | 0.000000 | 0 / 0 |
| `indexer_validate` | 0.000000 | 0.000000 | 0 / 0 |
| `indexer_cache_update` | 0.000000 | 0.000000 | 0 / 0 |
| `pooled_scores` | 69.255345 | 0.000000 | 864 / 0 |
| `topk_qa` | 0.000000 | 0.000000 | 0 / 0 |
| `prepare_cis` | 0.000000 | 0.000000 | 0 / 0 |
| `topk_cis` | 0.000000 | 0.000000 | 0 / 0 |
| `finish_selection` | 0.000000 | 0.000000 | 0 / 0 |
| `native_selection` | 15.448113 | 0.000000 | 864 / 0 |
| `native_prepare` | 2.024306 | 0.000000 | 256 / 0 |
| `native_prepare_ranked` | 14.031819 | 0.000000 | 1728 / 0 |
| `score_selection` | 171.266040 | 4.262748 | 1088 / 32 |
| `native_finite_check` | 11.307383 | 0.319331 | 1088 / 32 |
| `native_ranked_compression` | 9.115826 | 0.279362 | 1088 / 32 |

Native indexer 的 full / extend CUDA-event 区间为 672.511 / 6.647 ms；
它包含提交空档与插桩影响，不能与上表 kernel 总时间相加。

当前输入下，native 的 full / extend 端到端加速为 1.536× / 1.689×。
Extend 的完整 indexer / attention MFU 为 23.04% / 38.12%，
两个完整模块仍未同时达到 40%。这些结果只覆盖 resident HBM 前向，不能用于 offload、
网络服务或模型质量结论；[算子与三层模块回放](../nosa_kernel_mfu/README.md)采用不同计时边界。

### 报告数据

每组的 `report/sparse/native/`、`report/sparse/triton/` 选自上述新 run：

| 后端 | 墙钟汇总 | event 阶段 | 端到端 MFU | 模块 kernel / MFU | 采集元数据 |
| --- | --- | --- | --- | --- | --- |
| native | [timings.csv](report/sparse/native/timings.csv) | [stages.csv](report/sparse/native/stages.csv) | [mfu.json](report/sparse/native/mfu.json) | [module_mfu.csv](report/sparse/native/module_mfu.csv) | [metadata.json](report/sparse/native/metadata.json) |
| triton | [timings.csv](report/sparse/triton/timings.csv) | [stages.csv](report/sparse/triton/stages.csv) | [mfu.json](report/sparse/triton/mfu.json) | [module_mfu.csv](report/sparse/triton/module_mfu.csv) | [metadata.json](report/sparse/triton/metadata.json) |

[后端比较](report/sparse/comparison.json)、[attention 四族](report/sparse/native/attention_launches.json)、
[实际调度](report/sparse/native/kernel_dispatches.json)和[融合 score 核验](report/sparse/native/score_kernels.json)
均从这两组原始数据派生；[重算审计](report/sparse/result_integrity.json)记录验证项。
[报告来源清单](report/sparse/report_manifest.json)保存选用文件与生成器哈希。
完整源码、全部 layer/chunk 记录、SQLite 与 nsys 保留在对应 run 的 `output/` 分类目录。
报告生成器与审计脚本保存在 native run 的 `report_generation/`；从仓库根目录重建到系统临时目录。
