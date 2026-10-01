# NOSA indexer + block sparse：64K + 1K 端到端性能

## 实验目的与测量边界

测量完整 NOSA sparse 模型的单请求前向延迟，比较当前 native 与 Triton 后端，并定位
indexer、block sparse attention 和 CIS 的开销。主指标来自未启用 profiler 的独立进程；
模块分解和 Nsight Systems 时间线由第二个进程采集。两组使用同一请求、checkpoint、
源码、设备和测量参数，仅切换 `--kernel-backend`。

本报告已发布测量对应 `94bf521` 的 BF16-pair indexer / FA3 v3 attention，见
[实现检查点](../../docs/nosa_sm90_checkpoint.md)和 [SM90 算子](../../operators/sm90/README.md)。
实际实现身份以每次 metadata 的源码 SHA256、构建信息和 `sources/` 快照为准。

2026-09-30 新增 NOSA offload 分支后，模型与 cache 依赖的源码图已扩展。下文数字、
run ID 和保存的源码快照仍属于当时测量的 resident 执行图，不能作为新增 offload
执行图的结果；当前分支尚未在本实验完成改动后补测与验收。原报告与产物保留至受影响
范围补测验收后再更新。新增 sparse fetch 回放的状态与测量边界见
[独立 offload 实验](../nosa_offload_overlap/README.md)，不以其替代本实验的全模型测量。

| 阶段 | 起始缓存 | 模型输入 | 执行方式 | 主指标 |
| --- | --- | --- | --- | --- |
| `full_prefill` | 空 | 66,560 tokens | 65 个 1024-token chunk，全部 32 层 | 整段墙钟延迟、66,560 / 秒 |
| `extend` | 65,536-token prefix 已完成 | 1,024 candidate tokens | 单次 1024-token forward，全部 32 层 | 整段墙钟延迟、1,024 / 秒 |

64K 为 instruction + history 合计，1K 为完整 candidate suffix。模板 instruction 为
28 tokens，GR 输入预算为 history 65508、item budget 1052；实际执行边界为
`[0,65536)` 与 `[65536,66560)`，保存在 `request.json` 和 `execution.json`。
模型为 NOSA-8B，BF16、32 layers、32 Q heads、2 KV heads、D128；extend 的逐层 audit
检查 `Q=[1024,32,128]`、`K=V=[66560,2,128]`、selection=`[1024,2,64]`，全部选择有效。

使用完整 NOSA sparse policy：64-token block、17 inclusive local，含 sink/local 的
query-aware 阶段保留 33 块，query-agnostic CIS 补满 64 块；attention 加入原始 CIS
分数。模型与算子复用[现有 NOSA 实现](../../models/nosa/README.md)。两组分别从空 cache
构建自己的 sparse prefix；浮点舍入可能改变近似并列的选块，不要求跨后端输出逐位一致。

端到端计时包含 embedding、全部 decoder、CIS 投影、K/V/CIS 写入、压缩、query-aware
打分与选块、block sparse attention、输出投影、FFN、final norm、Python 提交和 CUDA
完成等待。输出为最后一个 chunk 的 normalized hidden states，不执行 LM head、采样
或自回归生成。checkpoint 加载、tokenizer/GR 生成、输入 H2D、cache 分配/reset 和 extend
的 prefix 构建均在计时外，因此这里测量单请求模型前向延迟。

完整 K/V/CIS 驻留 HBM；没有 DRAM backing、offload fetch、缓存淘汰或 fetch/compute
overlap，本实验不能作为 offload 验证。仅实验进程将上下文上限扩大为 66560，保留
checkpoint 的 LongRoPE factors，不改磁盘配置；不评价超出原 32768 上下文后的模型质量。

## 当前后端

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

Native 通过 TVM FFI 编译自有 `operators/sm90/csrc/`，复用共享 CUTLASS 和已安装的
FlashInfer headers，不加载 EzKernelKit。BF16 pooled-score scratch 为 4.0625 MiB，QA/CIS
及模型各层复用；kernel 临时空间不计入请求 resident KV cache 容量。旧模型接口
`sparse_backend="triton"` 仍指 SM90 dispatcher，实际后端以上表和 metadata 为准。

## 运行与复现

从仓库根目录运行；脚本也能自行定位仓库根目录。下面使用新的 run ID，并将同一请求
传给两组。`CUDA_VISIBLE_DEVICES=1` 将物理 GPU 1 映射为进程的 `cuda:0`。

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_native_rerun \
  --kernel-backend native --peak-tflops 989 \
  --request-file experiments/indexer_block_sparse_profile/output/data/sparse_native_h200_gpu1_20260929_01/request.json
CUDA_VISIBLE_DEVICES=1 bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_triton_rerun \
  --kernel-backend triton --peak-tflops 989 \
  --request-file experiments/indexer_block_sparse_profile/output/data/sparse_native_rerun/request.json
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.compare \
  --native-data-dir experiments/indexer_block_sparse_profile/output/data/sparse_native_rerun \
  --triton-data-dir experiments/indexer_block_sparse_profile/output/data/sparse_triton_rerun \
  --output-dir experiments/indexer_block_sparse_profile/output/data/sparse_native_rerun/comparison
```

默认 checkpoint 为 `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`、设备 `cuda:0`、prefix=65536、
new=1024、chunk=1024。每阶段预热 2 次，独立端到端计时 5 次取中位数；第二个进程同样
完整预热，每阶段采集 1 次模块 profile，JIT 编译、CUDA 模块加载及 allocator 初始化不计
入正式采集。不传 `--request-file` 时用共享 GR 生成器、seed=42 生成请求；profile 总是
读取 benchmark 保存的同一请求。两组须串行运行并保持源码、依赖、请求和设备一致。

脚本依次执行独立 benchmark → nsys profile → SQLite export → analyze → 端到端/模块
MFU。过程在系统临时目录 staging，保留子进程失败状态，只有全链成功才发布新的 run ID；
已有 ID 拒绝覆盖。失败产物留在实验目录之外。

```bash
bash experiments/indexer_block_sparse_profile/scripts/run.sh --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.capture --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.analyze --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.mfu --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.module_mfu --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.compare --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.launch_report --help
```

依赖 SM90/Hopper、PyTorch、Triton、FlashInfer、TVM FFI、nvcc、共享 CUTLASS、
safetensors、tokenizer 和 Nsight Systems，沿用项目环境。`--without-nsys` 仍测独立端到端
延迟与 CUDA-event 模块区间，但没有原始 nsys/SQLite 或模块 kernel MFU。
`--timeline-only` 在第二进程只记录整段 forward 根 NVTX、CUDA 活动和整段起止 events，
生成 `launch_analysis.json`、`launch_summary.json/csv/md`，不安装模块 wrappers/hooks，
也不生成模块分解或 MFU；该模式需要 nsys，不能与 `--without-nsys` 同用。
本节没有新增 timeline-only 或 CIS 独立 bottleneck 测量。

## 指标、正确性验收与产物

| 指标 | 定义与限制 |
| --- | --- |
| `wall_ms` | 提交开始至设备全部完成的墙钟毫秒数；未插桩独立进程的主指标。 |
| `host_submit_ms` | 模型前向 host 提交时间，含模型内部 CUDA 等待，与设备执行重叠，不能和 CUDA 时间相加。 |
| `cuda_span_ms` | 整段 CUDA events 的间隔，包含 host launch 造成的空档。 |
| 模块 `cuda_elapsed_ms` | profile 内 inclusive CUDA-event 区间，包含提交空档和插桩影响；先汇总全部层/chunks，再对 repeats 取中位数。 |
| 模块 `kernel_ms` | 通过 NVTX/CUDA launch correlation 归属的 GPU kernel duration 总和；不是主墙钟的可加分段。 |
| 端到端 `mfu_pct` | 100 × useful matrix FLOPs /（未插桩墙钟中位数 × 峰值 FLOP/s）。 |
| 模块 `mfu_pct` | 100 × useful matrix FLOPs /（对应 kernel 总秒数 × 峰值 FLOP/s），不表示 occupancy。 |

MFU 使用 `--peak-tflops 989`，即 **H200 SXM BF16 dense Tensor Core 标称参考峰值
989 TFLOPS**，不是 2:4 稀疏峰值或实测可持续峰值。设备原始名称保留为 `NVIDIA H20Z`；
不能根据所选峰值分母反推硬件 SKU。其他硬件应显式指定适当分母。

Useful FLOPs 按逻辑工作计数：indexer QK 一次，attention 按各 query 实际选择的因果
token 计算 QK/PV。重算、精确剪枝的准备、并集中其他 query 的工作、repair、future mask
和 padding 不增加有效分子，所有已发起 kernel 的耗时均保留在分母。
融合 `score_selection` 持有一次完整 indexer QK 及其整个融合 kernel 时间；full-prefill
按每层/chunk 的实际归属计数后汇总。CIS、indexer 总分母包含非矩阵操作，单独的非矩阵
子项 MFU 为不适用。Indexer 子项均已包含在 `indexer_total`，不能重复相加；联合入口中
拆出的 kernel 子项没有独立 CUDA-event 区间，不据此构造伪造的 event MFU。

计时外检查每组 full-prefill 与 prefix-ready extend 的全部 1K candidate hidden 有限且
精确一致，并检查 profile instrumentation 不改变输出；这些验收用于正确性，不作为
性能或模型质量结果。每次 full-prefill 在计时前 reset 独立 cache；重复 extend 回退
独立 prefix cache 长度并重写 suffix。含 opaque layer state 的模型会拒绝这种回放。

```text
output/log/<run_id>/       各子进程独立 stdout/stderr
output/data/<run_id>/      request.json、execution.json、metadata.json、profile_metadata.json
                          measurements.json、attention_audit.json、summary.json、report.md
                          timings.csv、stages.csv、layers.csv、mfu.json
                          module_mfu.json/csv/md、nsys.sqlite、sources/ 源码快照
output/profile/<run_id>/   indexer_block_sparse_profile.nsys-rep
```

以上完整产物默认不提交；报告选用文件放 `report/` 并随 Git 保存。Metadata 记录设备名称、
UUID、依赖、输入形状/精度、上下文设置、checkpoint 配置哈希及权重文件属性、源码与构建
指纹。Nsight 关闭 CPU sampling/context-switch tracing，采集 CUDA/NVTX/OS runtime；
逐层标签为 `NOSA/<phase>/layer_<id>/<stage>/q<start>+<length>`。

调用模块：`src/capture.py` 复用共享 GR、`executor.model_executor.run_chunks`、
`models.nosa.model`、`experiments.nosa_gr_65536_1024.src.capture.execution_split` 和
该实验的 `src.sources.source_hashes`；`src/instrumentation.py` 临时包装已有模型/算子，
退出时恢复；`src/analyze.py` 检查阶段、层、chunk 覆盖并聚合结果；`src/mfu.py` 复用
`experiments.nosa_gr_65536_1024.src.mfu.matrix_flops` 的投影/FFN 计数，将 dense attention
替换为 sparse 工作量并加入 CIS/indexer QK；`src/module_mfu.py` 从 SQLite 归属完整
kernel 序列；`src/compare.py` 校验两组输入身份和派生报告哈希后计算比值。实验 CPU 单元
测试纳入全局 `scripts/run_tests.sh`，测试结果仅输出终端。

## 结果与结论

本次发布的 native run 为 `sparse_native_h200_gpu1_20260929_01`，Triton run 为
`sparse_triton_h200_gpu1_20260929_01`。两组已完成独立计时、模块 profile 和对照验收。
74 个采集源码指纹、请求字节、checkpoint 文件记录、设备、依赖和预热/重复参数一致；
构建元数据仅允许 selected_backend 不同，src.compare 已检查这些条件及派生文件哈希。
Checkpoint 核验采用配置 SHA256 和权重文件路径、大小、mtime。

### 实测环境

两组在物理 GPU 1 串行运行，原始设备名 `NVIDIA H20Z`，SM90、132 SM，显存
139.812 GiB。CUDA/Nsight UUID 为 `a2e4882b-3248-6018-14ab-90e45b5a5b04`，PCI 地址为 `0000:67:00.0`，
驱动为 `570.124.06`。Nsight 通过进程、CUDA ordinal、UUID、架构/SM 和显存核验
测量设备。MFU 使用上述 H200 标称 989 TFLOPS 参考分母。

| 项目 | 实际测量记录 |
| --- | --- |
| native / Triton metadata 时间（UTC） | 2026-09-29T15:49:29.613004+00:00 / 2026-09-29T15:52:15.578463+00:00 |
| PyTorch / CUDA build | 2.12.1+cu130 / 13.0 |
| Triton / FlashInfer / TVM FFI | 3.7.1 / 0.6.18 / 0.1.13.post3 |
| nvcc | `/usr/local/cuda-13.2/bin/nvcc`，13.2 / V13.2.86 |
| CUTLASS commit | f3fde58372d33e9a5650ba7b80fc48b3b49d40c8 |
| Nsight Systems | NVIDIA Nsight Systems version 2025.6.3.541-256337736014v0 |

Native 编译参数、FA3 headers 与 CUDA 源码指纹均保存于 metadata；实际依赖以测量记录
为准，`uv.lock` 随源码快照保留。

### 独立端到端计时

每阶段 warmup=2、repeats=5。下表为未插桩墙钟中位数与 `[min, max]`，单位 ms；
加速比为 Triton 中位数 / native 中位数。只比较本节边界一致的两个 sparse 后端。

| 阶段 | Triton wall ms [min, max] | native wall ms [min, max] | Triton MFU | native MFU | 加速比 |
| --- | --- | --- | --- | --- | --- |
| full_prefill | 3799.321 [3775.707, 3808.557] | 2430.978 [2425.055, 2442.510] | 31.49% | 49.21% | 1.563× |
| extend | 63.729 [63.676, 63.787] | 36.520 [36.454, 37.619] | 29.86% | 52.11% | 1.745× |

两组有效矩阵工作量相同：full-prefill 为 `1,183,130,731,937,792` FLOPs，extend 为
`18,820,462,804,992` FLOPs。两组内部 full/extend 的全部 candidate hidden 均有限，最大绝对差为 0，
profile 不改变输出；各组 32 层 extend audit 及完整层/chunk/kernel 归因通过。
本次实验工具 CPU 测试 553 项通过，全局 GPU 回归 361 项通过。测试仅用于正确性验收。

### 完整模块 kernel 时间

每组每阶段采集 **1 次** 独立 nsys 模块 profile。下表是 correlated kernel duration 总和，
单位 ms；其比值只描述本次采集，不估计多次采集的方差，不能与主墙钟相加或相减。

| 模块 | full Triton | full native | 加速比 | extend Triton | extend native | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| `indexer_total` | 678.140 | 279.321 | 2.428× | 15.736 | 4.840 | 3.251× |
| `block_sparse_attention` | 1195.549 | 353.703 | 3.380× | 18.459 | 5.768 | 3.201× |

`score_selection` 融合后两组不存在等价的 score-only 边界，因此不报告 `pooled_scores`
加速比。保留完整 indexer 比较以及各组实际子项数据，融合 kernel 的全部时间归入 indexer。

Native full-prefill 共 2080 个 layer/chunk 调用：前四个 chunk 的 128 个调用不计算 score，
864 个调用使用 Triton `pooled_scores`，1088 个调用使用融合 `score_selection`。
其中 1056 次为常规融合 kernel，最后 32 次命中精确剪枝 kernel；
Extend 的 32 次均命中精确剪枝 `score_selection`。Attention 的 full / extend 分别为
2080 / 32 个 FA3 v3 调用，四族 kernel 合计 8320 / 128 个 launch。

Native attention 的四族 launch 明细来自同一 SQLite，逐层、逐阶段的数量与耗时已核验与模块报告完全守恒：

| kernel 族 | full ms | full launches | extend ms | extend launches |
| --- | --- | --- | --- | --- |
| 并集准备 `nosa_fa3::prepare` | 6.310646 | 2080 | 0.101632 | 32 |
| 工作排序 `nosa_fa3::sort_work_by_union_size` | 3.591695 | 2080 | 0.055072 | 32 |
| FA3 `flashinfer::PrefillWithKVCacheKernel<nosa_fa3::...>` | 335.995863 | 2080 | 5.515615 | 32 |
| Repair `nosa_attention::attention_kernel` | 7.804705 | 2080 | 0.095200 | 32 |
| **完整 attention** | **353.702909** | **8320** | **5.767519** | **128** |

未采集实际 repair query 数量；不能把 repair launch 次数作为回退比例，也不能按四个
launch 将 useful QK/PV FLOPs 乘四。

### Native 模块 MFU 与 indexer 分解

| 模块 | full kernel ms | full MFU | extend kernel ms | extend MFU |
| --- | --- | --- | --- | --- |
| `qkv_proj` | 104.671 | 77.67% | 1.609 | 77.73% |
| `o_proj` | 91.663 | 78.84% | 1.411 | 78.81% |
| `gate_up_proj` | 715.091 | 80.84% | 11.000 | 80.85% |
| `down_proj` | 347.383 | 83.21% | 5.335 | 83.35% |
| `block_sparse_attention` | 353.703 | 39.31% | 5.768 | 38.26% |
| `indexer_total` | 279.321 | 13.08% | 4.840 | 23.14% |
| `cis_projection` | 26.003 | 0.0085% | 0.400 | 0.0085% |

| Native indexer 子项 | full kernel ms | extend kernel ms | full / extend kernel 数 |
| --- | --- | --- | --- |
| `native_prepare` | 2.015663 | 0.000000 | 256 / 0 |
| `native_prepare_ranked` | 14.027866 | 0.000000 | 1728 / 0 |
| `pooled_scores` | 69.475660 | 0.000000 | 864 / 0 |
| `native_selection` | 14.845017 | 0.000000 | 864 / 0 |
| `native_finite_check` | 10.952200 | 0.322049 | 1088 / 32 |
| `native_ranked_compression` | 8.620735 | 0.274144 | 1088 / 32 |
| `score_selection` | 159.156980 | 4.243575 | 1088 / 32 |
| 短上下文全块选择（parent 内） | 0.227362 | 0.000000 | 128 / 0 |

Native indexer 的 full / extend CUDA-event 区间为
1264.197 / 13.047 ms，
kernel 总时间为 279.321 / 4.840 ms；
前者包含提交空档和插桩影响，二者不能相加。

当前输入下，native 的 full / extend 端到端加速为 1.563× / 1.745×；完整 indexer kernel 加速为 2.428× / 3.251×，
完整 attention kernel 加速为 3.380× / 3.201×。
Extend 的完整 indexer / attention MFU 为 23.14% / 38.26%，
仍未达到两个完整模块均为 40% 的目标；端到端 MFU 不能替代完整模块 MFU。

这些结果仅覆盖当前输入下的完整 NOSA resident HBM 前向，不能用于 offload 性能、网络
服务延迟或模型质量结论。独立 synthetic operator 实验见
[nosa_kernel_mfu](../nosa_kernel_mfu/README.md)，其计时边界与本实验不同。

### 报告数据

下列文件来自已验收的新 run，已替换原 020961b 报告；对应旧运行产物已清理。 `stages.csv` 为 inclusive CUDA-event 数据，`module_mfu.csv`
为 nsys kernel/MFU 数据；完整层/chunk 记录、源码快照、SQLite 和原始 nsys 留在本节
两个 run 的 `output/` 分类目录。

| 来源 run ID | 墙钟汇总 | event 阶段 | 端到端 MFU | 模块 kernel / MFU | 采集元数据 |
| --- | --- | --- | --- | --- | --- |
| `sparse_native_h200_gpu1_20260929_01` | [timings.csv](report/native/timings.csv) | [stages.csv](report/native/stages.csv) | [mfu.json](report/native/mfu.json) | [module_mfu.csv](report/native/module_mfu.csv) | [metadata.json](report/native/metadata.json) |
| `sparse_triton_h200_gpu1_20260929_01` | [timings.csv](report/triton/timings.csv) | [stages.csv](report/triton/stages.csv) | [mfu.json](report/triton/mfu.json) | [module_mfu.csv](report/triton/module_mfu.csv) | [metadata.json](report/triton/metadata.json) |

后端比较为 [comparison.csv](report/comparison.csv) 与含定义及输入指纹的
[comparison.json](report/comparison.json)；native attention 四族明细为
[attention_launches.csv](report/native/attention_launches.csv) 和含来源指纹的
[attention_launches.json](report/native/attention_launches.json)，实际调度汇总为
[kernel_dispatches.json](report/native/kernel_dispatches.json)。
精确剪枝 kernel 的实际调用计数与融合时间守恒证据见
[score_kernels.json](report/native/score_kernels.json)。报告文件哈希与生成器指纹见
[report_manifest.json](report/report_manifest.json)。

只重建两组已有 run 的比较时，运行上节 `src.compare` 命令并替换两组数据目录，输出到
尚不存在的目录。表中每组文件直接取自对应 run；comparison、attention 四族及 dispatch
明细由同一组原始数据派生。
本节 README 的模板、生成器和四族核验脚本快照保存在 native run 的 comparison/。
从仓库根目录重建本节报告到新的系统临时目录（不修改采集数据）：

```bash
.venv/bin/python experiments/indexer_block_sparse_profile/output/data/sparse_native_h200_gpu1_20260929_01/comparison/nosa-profile-report-rebuild-20260929.py \
  --template experiments/indexer_block_sparse_profile/output/data/sparse_native_h200_gpu1_20260929_01/comparison/readme_template.md \
  --output-dir /tmp/nosa-sparse-report-rebuilt
```
