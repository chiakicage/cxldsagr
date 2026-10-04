# NOSA attention / indexer score 的有效 MFU

本目录归入 baseline 实现性能检查，负责 resident 算子和完整模块的效率诊断。
整理后的入口将数值验收、性能计时和 profile 分开；本次整理没有运行新实验。下文保留
2026-10-04 已完成的四组 resident 测量：synthetic 算子、固定真实输入算子，
以及 native / Triton 完整模块。所有测量使用物理 GPU 5，源码和构建身份一致。
本页数字只描述这些输入与调用边界，不代表 offload、完整模型前向或模型质量。

## 实验目的与边界

独立测量 `block_sparse_attention` 和 `pooled_scores` 的 API 延迟及有效矩阵 MFU，
并检查完整 `NosaIndexer` 和 `NosaSparseAttention` 的 API 成本，判断 baseline 是否存在
明显的实现瓶颈。下文保留原测量使用的 30% MFU 参照，不将其作为独立研究目标。
算子测试使用固定 seed 的 synthetic 输入或已保存的 L0/L15/L31 真实输入，K/V/CIS
全部驻留 HBM。完整模型测量见[完整 sparse profile](../nosa_baseline_performance/README.md)。

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

MFU = 有效矩阵 FLOPs /（时间 × 标称 dense BF16 Tensor Core 峰值）。本轮分母为
989 TFLOP/s；attention 仅计选中因果 token 的 QK+AV，score 的完整因果压缩窗口 QK
只计一次。重算、padding、masked future、softmax、GQA 求和、pooling 与访存不增加
分子，耗时仍计入分母。64K+1K 每层 attention / score 分别为 68,190,994,432 /
34,616,115,200 FLOPs；达到 30% 分别要求时间不超过 229.831 / 116.670 µs。

## 独立验收与性能入口

新入口默认 `--mode bench`，要求提供匹配的 `--validation-receipt`。三个阶段独立调用：

- `check`：算子检查全部 query 的 FP32 参考输出及 graph/eager 一致性；完整模块另检查
  增量压缩、选择、缓存回滚和重复调用。只生成验收依据，不给出延迟或 MFU。
- `bench`：核验已有 receipt 后预热、计时；样本之间不比较输出，也不启动 profiler。
- `profile`：使用相同 receipt，单独采集算子 trace 或完整模块的 kernel 时间。
  profile 数字用于定位瓶颈，不充当正式 API 延迟。

Receipt 绑定源码、native 构建信息、软件环境、设备、后端、cache 路径、输入内容、
形状、stride 和 dtype。修改这些条件须重做对应验收；调整预热、重复次数或 MFU 峰值
不触发完整数值检查。算子 graph 的 `graph_calls` 仍属于验收配置。真实模型输入继续
使用冻结的 capture，并不代表当前 motivation 的实际运行轨迹。

脚本将 `check` 的数据、日志和 receipt 留在系统临时目录，成功后打印位置；可显式
指定 receipt 路径。`bench/profile` 仍按实验的 `output/data|log|profile/<run_id>/` 保存。
正式结果保存 receipt 副本并注明 `independent_check`，不把复用依据写成本轮重新验收。
模型或算子 API 内已有的 finite 检查、repair 和同步保留在原执行路径内。

```bash
# Synthetic resident operators: independent acceptance, then clean measurement.
bash experiments/nosa_kernel_mfu/scripts/run.sh kernel_check \
  --mode check --peak-tflops 989 --validation-receipt /tmp/nosa-kernel-check.json
bash experiments/nosa_kernel_mfu/scripts/run.sh kernel_bench \
  --peak-tflops 989 --validation-receipt /tmp/nosa-kernel-check.json
bash experiments/nosa_kernel_mfu/scripts/run.sh kernel_profile \
  --mode profile --peak-tflops 989 --validation-receipt /tmp/nosa-kernel-check.json

# Complete modules use a separate receipt for each backend and input configuration.
nosa_inputs=experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345
bash experiments/nosa_kernel_mfu/scripts/modules.sh modules_check \
  --mode check --input-dir "$nosa_inputs" --kernel-backend native --peak-tflops 989 \
  --validation-receipt /tmp/nosa-modules-native-check.json
bash experiments/nosa_kernel_mfu/scripts/modules.sh modules_bench \
  --input-dir "$nosa_inputs" --kernel-backend native --peak-tflops 989 \
  --validation-receipt /tmp/nosa-modules-native-check.json
bash experiments/nosa_kernel_mfu/scripts/modules.sh modules_profile \
  --mode profile --input-dir "$nosa_inputs" --kernel-backend native --peak-tflops 989 \
  --validation-receipt /tmp/nosa-modules-native-check.json
```

以下结果与图表仍来自原 run ID。本轮只完成入口整理和 CPU 检查，新的分阶段路径尚未
运行 CUDA 数值验收或性能测量，不能据此更新原性能结论。

## 运行与调用模块

实际命令均从仓库根目录执行。四次测量串行使用物理 GPU 5、NUMA1、CPU 48–55，
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=8`，PyTorch intra-op 为 8。
以下保留该轮旧入口的命令形式；它们未使用独立 receipt。新运行使用上一节入口，
完整历史启动环境保存在各 run 的 `launch_environment.json`。

```bash
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=48-55 --membind=1 \
  bash experiments/nosa_kernel_mfu/scripts/run.sh kernel_synthetic_rerun \
  --device cuda:0 --peak-tflops 989 --reference-all
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=48-55 --membind=1 \
  bash experiments/nosa_kernel_mfu/scripts/run.sh kernel_captured_rerun \
  --device cuda:0 --peak-tflops 989 --reference-all \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=48-55 --membind=1 \
  bash experiments/nosa_kernel_mfu/scripts/modules.sh modules_native_rerun \
  --device cuda:0 --kernel-backend native --peak-tflops 989 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=48-55 --membind=1 \
  bash experiments/nosa_kernel_mfu/scripts/modules.sh modules_triton_rerun \
  --device cuda:0 --kernel-backend triton --peak-tflops 989 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345
```

`src/measure.py` 调用 `models.nosa.indexer` 的压缩与 FP32 reference、
`operators.nosa.indexer.api` 和 `operators.nosa.attention.device_only.api`；
FLOP 口径复用 `experiments.nosa_baseline_performance.src.sparse.mfu.work_counts`。
`src/measure_modules.py` 调用现有 `NosaIndexer` / `NosaSparseAttention`。
两者复用 `experiments.nosa_baseline_performance.src.dense.sources.source_hashes` 记录源码。
`scripts/capture.sh` / `src/capture_inputs.py` 可从独立空 cache 重新采集实际 sparse 输入，
本轮只读取上述保留的输入 capture。

脚本先写系统临时目录，全链成功后发布到新的 `output/{data,log,profile}/<run_id>/`；
源码中途变化或 run ID 已存在会失败。stdout/stderr 分开保存，原始 module Chrome trace
放在 `output/profile/`，全部样本、源码快照和环境记录放在 `output/data/`。

本轮物理 GPU 5 的 CUDA 名称为 `NVIDIA H200`，NVML 名称为 `NVIDIA M403`；
两者对应同一 UUID `a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`。两种工具的原始名称均保存在 metadata，
设备为 SM90、132 SM；989 TFLOP/s 是本实验显式指定的 MFU 分母。
依赖：PyTorch 2.12.1+cu130、CUDA build 13.0、nvcc 13.2.78、Triton 3.7.1、
FlashInfer 0.6.18、TVM FFI 0.1.13.post3，CUTLASS 提交
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。完整设备、编译参数和环境以 metadata 为准。
四个 run 的 105 个源码指纹完全一致，其排序 JSON 摘要为
`9c03515108129af50633b8b9b21d73b4c2f05ccf7523bb272375e65120963564`；摘要算法与来源文件见[发布来源](report/publication.json)。

## 完整模块检查点

Run ID 为 `modules_flags_native_20261004_01` 和 `modules_flags_triton_20261004_01`。
每次调用都从相同的已提交 64K prefix 开始，调用后回滚候选派生状态；完整 indexer
的有限值校验、增量压缩、排名和选块在计时内。Prefix 构建、K/V/CIS 写入、cache
分配、事务 begin/abort 在计时外。两个模块均预热 10 次、重复 30 次，另各采集
3 次 torch.profiler。下表是完整 API 中位延迟：

| Layer | 模块 | native event µs | Triton event µs | event 加速比 | native wall µs | Triton wall µs |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | indexer | 288.096 | 761.488 | 2.64× | 293.747 | 767.352 |
| 0 | attention | 266.672 | 650.320 | 2.44× | 272.183 | 656.257 |
| 15 | indexer | 300.528 | 771.792 | 2.57× | 306.058 | 776.722 |
| 15 | attention | 262.064 | 650.896 | 2.48× | 267.572 | 656.699 |
| 31 | indexer | 296.192 | 762.208 | 2.57× | 301.919 | 768.111 |
| 31 | attention | 260.096 | 652.640 | 2.51× | 265.402 | 657.930 |

下表单独列出 profiler 的 GPU kernel 时间和。它不包含 Python 空档与 D2H copy，
不能解释为完整 API 延迟。Native indexer 每次有 3 个 kernel，attention 有 4 个；
排序、prepare 和 repair 均包含在 attention 分母中。

| Layer | 模块 | native kernel µs | native MFU | Triton kernel µs | Triton MFU |
| --- | --- | --- | --- | --- | --- |
| 0 | indexer | 141.826 | 24.679% | 496.989 | 7.043% |
| 0 | attention | 176.287 | 39.112% | 577.511 | 11.939% |
| 15 | indexer | 138.301 | 25.308% | 489.667 | 7.148% |
| 15 | attention | 175.010 | 39.397% | 577.860 | 11.932% |
| 31 | indexer | 136.575 | 25.628% | 491.427 | 7.122% |
| 31 | attention | 171.355 | 40.238% | 580.203 | 11.884% |

Native 完整 indexer 的 kernel MFU 约为 24.7%–25.6%，仍低于 30%；native attention
约为 39.1%–40.2%，只有 L31 超过 40%。所有完整 API 的 MFU 都低于各自 kernel
时间口径，不能用后者代替公开接口的成本。

增量压缩与独立完整压缩一致，完整 indexer 与 standalone 选择一致，所有重复输出
精确相同；相对于保留 capture 的 selection 变化数为零，native / Triton 的 selection
输出哈希相同。36 个 trace 的 162 个 kernel 均通过时间和核验，未执行 offload kernel。
这组模块结果不包含 CIS 投影，也不代表模型端到端加速。

数据：[模块对照 CSV](report/module_comparison.csv)、
[native 全部样本](report/modules/native/results.json)、
[native 元数据](report/modules/native/metadata.json)、
[Triton 全部样本](report/modules/triton/results.json)、
[Triton 元数据](report/modules/triton/metadata.json)。

## 结果与结论

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

固定真实输入 run 为 `kernel_flags_real_20261004_01`，三层均为 64K+1K：

| Layer | 算子 | native 配置实际后端 | native Graph µs | Triton Graph µs | 加速比 | native 配置 MFU |
| --- | --- | --- | --- | --- | --- | --- |
| L00 | attention | CUDA/CuTe | 169.491 | 573.266 | 3.38× | 40.680% |
| L00 | pooled scores | CUDA/CuTe | 172.403 | 282.701 | 1.64× | 20.302% |
| L15 | attention | CUDA/CuTe | 167.952 | 575.234 | 3.42× | 41.053% |
| L15 | pooled scores | CUDA/CuTe | 170.880 | 284.499 | 1.66× | 20.483% |
| L31 | attention | CUDA/CuTe | 164.902 | 575.163 | 3.49× | 41.812% |
| L31 | pooled scores | CUDA/CuTe | 169.410 | 284.080 | 1.68× | 20.661% |

固定真实输入的 native attention Graph MFU 均超过 40%，其 Graph 延迟约为
164.9–169.5 µs；native pooled score 约为 169.4–172.4 µs，MFU 约为 20.3%–20.7%，
尚未达到 30%。同一实现对 synthetic 与真实 selection 的表现不同，本实验未单独
测量其中各项开销的成因。

全部 28 个算子结果行均检查完整 1024-query FP32 reference，Graph/eager 输出精确
一致；12 个模块结果行均通过完整重复与组合检查。共核对 2,436 个计时间隔、源码
快照和 profile 内容哈希，见[产物完整性记录](report/result_integrity.json)。

数据：[算子对照 CSV](report/operator_comparison.csv)、
[synthetic 全部样本](report/synthetic/results.json)、
[synthetic 元数据](report/synthetic/metadata.json)、
[固定真实输入全部样本](report/captured/results.json)、
[固定真实输入元数据](report/captured/metadata.json)。报告中的原始 JSON/CSV 直接复制自
四个 run 的同名文件；对照表从这些样本的中位数生成，来源与哈希见
[发布来源](report/publication.json)。
