# NOSA attention / indexer score 的有效 MFU

2026-10-01 目录整理：算子已迁至 `operators/nosa/`、`operators/deepseek_v32/` 和
`operators/common/`。本页性能仍对应下文原 run ID 与源码快照；目录迁移后的性能未重新测量。

## 实验目的与边界

独立测量 `block_sparse_attention` 和 `pooled_scores` 的有效矩阵 MFU，检验两个算子
能否超过 30%。此实验使用固定 seed 的 synthetic Q/K/V/CIS，测 resident HBM 算子，
不执行模型前向、checkpoint 加载、offloading 或模型质量评估。
实际模型的逐层 kernel 与端到端测量见 [完整 sparse profile](../indexer_block_sparse_profile/README.md)。

默认 NOSA-8B 形状：BF16，1024 queries，32 Q heads / 2 KV heads，D128，Q 来自
`[1024,4608]` 的 QKV packed view，保持模型的 query stride。prefix 分别为
4096、16384、32768、65536 tokens；K/V 包含 prefix 和新增 queries。
K/V 为标准正态分布，CIS 为标准正态乘 0.1；seed 为 `42 + prefix`。
压缩调用模型的 32-token / stride-16 实现，选择调用完整 NOSA 33/64 indexer。
native 与 Triton 使用相同输入和相同选择；生成输入、压缩和选择在计时外。
synthetic 选择分布不能替代真实模型的跨 query 相似度或 cache 行为。

两个计时口径均预热 10 次、重复 30 次，报告中位数及 min/max：

- `graph`：一个 CUDA Graph 连续执行 10 次同输入算子，CUDA events 总时间除以 10；
  包含算子全部 kernels 和 graph 间隙，消除逐 kernel 的 Python 提交空档。重复使用输入。
- `eager`：单次普通 Python 调用的 CUDA event 区间，包含 host launch 空档。

attention 输出及 native score 临时缓冲在算子调用内分配，graph 重用捕获时的存储。
编译、数值验收和预热在计时外；运行保存实际依赖版本、设备 UUID、源码与输入哈希。
每种后端从实际完整 batch 输出抽样检查 FP32 reference，`--reference-all` 检查全部行；
graph replay 输出还须与同后端 eager 输出精确一致。attention 使用既有 BF16/CIS 算子
契约的 `rtol=atol=0.016`，同时记录更严格 `atol=0.001, rtol=0.016` 的超限元素数及相对 L2
误差；score 记录绝对、相对 L2 和最大相对误差。完整数值回归仍属于
`operators/nosa/` 下各功能的 `tests/`，验收记录不能作为模型等价性证明。

MFU = 有效矩阵 FLOPs /（时间 × 标称 dense BF16 Tensor Core 峰值）。H200 默认峰值
989 TFLOPS；其他 SKU 必须显式指定。attention 仅计选中因果 token 的 QK+AV；score
每个完整因果压缩窗口的 QK 只计一次。重算、padding、masked future、softmax、GQA 求和、
pooling 与访存不增加分子，其耗时保留在分母。此指标不是 occupancy 或硬件指令利用率。
64K+1K 每层 attention 为 68,190,994,432 FLOPs，score 为 34,616,115,200 FLOPs；
超过 30% 分别要求低于 229.831 µs 与 116.670 µs。

## 运行与调用模块

```bash
bash experiments/nosa_kernel_mfu/scripts/run.sh kernel_mfu_001 --device cuda:0 --peak-tflops 989
bash experiments/nosa_kernel_mfu/scripts/capture.sh kernel_inputs_001 --kernel-backend triton
bash experiments/nosa_kernel_mfu/scripts/run.sh kernel_real_mfu_001 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kernel_inputs_001 --reference-all --peak-tflops 989
bash experiments/nosa_kernel_mfu/scripts/run.sh --help
bash experiments/nosa_kernel_mfu/scripts/modules.sh module_mfu_001 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kernel_inputs_001 \
  --kernel-backend native --peak-tflops 989
```

`modules.sh` 调用 `src/measure_modules.py` 测完整 `NosaIndexer` 和
`NosaSparseAttention`。每次调用均从相同已提交 prefix 开始，回滚 suffix 的派生缓存；
indexer 校验、增量压缩及选块全部在计时内。prefix 构建、K/V/CIS 写入、cache
begin/abort 在计时外。独立报告完整 API 的 eager CUDA event 与墙钟延迟，以及另一轮
torch.profiler 采集的完整 kernel 时间和。后者不包含 Python 空档与 D2H copy，不代表
API 延迟；不把局部 score 的 Graph 时间代入完整 indexer 分母。原始 Chrome traces
放在 `output/profile/<run_id>/`。本入口的三层检查点数据见下方完整模块检查点。

需要 Hopper/SM90、nvcc、共享 CUTLASS、TVM FFI、PyTorch、Triton 和 FlashInfer，沿用项目
环境。以上命令按本机已确认的 H200 配置显式指定 989 TFLOPS；设备上报 H20Z 时，
名称检测不会自动识别为 H200，必须保留该参数；其他型号应填写相应的 dense BF16 峰值。
脚本从任意工作目录定位仓库根目录，先在系统临时目录测量，成功后发布到新 run ID；
失败输出留在实验目录外。测量过程中源码变化会拒绝发布，已有 run ID 拒绝覆盖。

`src/measure.py` 调用 `models.nosa.indexer` 的压缩与 FP32 score reference、
`operators.nosa.indexer.api` 和 `operators.nosa.attention.device_only.api` 的现有算子；
复用 `experiments.indexer_block_sparse_profile.src.mfu.work_counts` 的 FLOP 口径，
以及 `experiments.nosa_gr_65536_1024.src.sources.source_hashes` 的源码记录。
不在实验中实现模型计算或优化 kernel。

`src/capture_inputs.py` 从独立空 cache 构建 64K sparse prefix，采集 1K suffix 的第
0、15、31 层实际 attention 输入（可用 `--layers` 改变）。保留 Q/K/V/CIS、实际消费的
selection 和请求 cache 中的压缩 K，不重算选择或压缩。采集后的无包装 suffix 重放须
与采集时 hidden 精确相同。保存 checkpoint 内容哈希、请求字节、源码快照和输入 stride。
此入口生成性能实验输入，不包含计时；不能将重放一致解释为不同后端或模型质量等价。

完整 `metadata.json`、`results.json`、`summary.csv` 及源码快照放在
`output/data/<run_id>/`，stdout/stderr 分开放在 `output/log/<run_id>/`。
报告选定数据放在 `report/`，本文注明来源 run ID。

## 完整模块检查点

已验收 run `kda_main_bf16_pair_v3_development` 于 2026-09-29 测量，现整理发布；
此次整理未重新计时。输入来自 `kda_inputs_baseline_20260928_1345` 的 L0/L15/L31
真实激活和选择，使用 H20Z/SM90、132 SM、物理 GPU 0；64K prefix + 1K queries，
BF16、32 Q heads / 2 KV heads / D128、完整 NOSA 33/64 与 CIS，峰值分母 989 TFLOPS。
依赖为 PyTorch 2.12.1+cu130、Triton 3.7.1、FlashInfer 0.6.18、TVM FFI 0.1.13.post3；
完整设备身份、编译参数、依赖和源码哈希保存在下方 metadata。

每个模块预热 10 次，eager 重复 30 次，另采集 3 次 torch.profiler。
GPU kernel 时间总和覆盖完整 indexer 的三个 kernel，以及完整 attention 的四个 kernel。
它包含校验、增量派生缓存准备、排序和选块；不包含 host 空档和 D2H copy，不能解释为
完整 API 延迟。prefix 构建、普通 K/V/CIS append 写入、cache 分配和事务 begin/abort
在该 append-module 测量外。

| Layer | Indexer kernel µs | Indexer MFU | Attention kernel µs | Attention MFU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 141.471 | 24.741% | 175.391 | 39.312% |
| 15 | 137.854 | 25.390% | 173.247 | 39.798% |
| 31 | 137.504 | 25.455% | 170.879 | 40.350% |

| Layer | Indexer event API µs | Indexer wall API µs | Attention event API µs | Attention wall API µs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 330.304 | 338.864 | 288.080 | 293.984 |
| 15 | 345.344 | 351.797 | 289.632 | 295.696 |
| 31 | 335.040 | 341.541 | 281.824 | 287.761 |

仅 L31 attention 达到 40%；完整 indexer 与三层 attention 的共同目标尚未完成。
独立压缩和选块检查通过，captured selection ID 变化数为零；18 个 trace 均通过
完整 kernel 序列与时间守恒检查。该记录提供模块性能证据，不表示端到端加速、模型质量
或 offload 效果，也不替代下节 synthetic 同输入 native/Triton 对照。

报告数据从该 run 原始文件直接复制：
[摘要](report/module_checkpoint/compact_summary.json)、
[完整样本](report/module_checkpoint/results.json)、
[元数据与源码指纹](report/module_checkpoint/metadata.json)、
[归因检查](report/module_checkpoint/attribution_check.json)、
[发布来源](report/module_checkpoint/publication.json)。
原始数据、源码快照、trace 和独立日志分别保存于
`output/data/kda_main_bf16_pair_v3_development/`、
`output/profile/kda_main_bf16_pair_v3_development/` 和
`output/log/kda_main_bf16_pair_v3_development/`。

发布保留原测量 run ID 和 source SHA256。metadata 中的 `git_commit=020961b` 是测量时
未提交工作树的基线；实际测量实现以 `source_sha256` 和源码快照为准。本次提交相对于
该测量快照，仅对 attention Python adapter 做 AST 相同的格式调整，并补充精确推理图
注册；CUDA 计算源码未改变。原始测量命令保存在
上述 data 目录的 `original_invocation.py`，命令中的临时路径保留原样；当前复现入口为：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_kernel_mfu/scripts/modules.sh module_checkpoint_rerun \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --kernel-backend native --peak-tflops 989 --layers 0 15 31 \
  --warmup 10 --repeats 30 --profile-repeats 3
```

## 结果与结论

补测状态（2026-09-29）：当前已接入 KDA score/attention 检查点，**改动后未运行本节完整实验**。
以下已发布数字仍对应 `020961b` 及 run `kernel_mfu_h200_gpu1_20260928_071840`，
不代表候选性能。原报告与运行产物保留至新实现完成验收和补测。

正确性边界更新（2026-09-29）：原 `020961b` 与优化中的 native attention 均在
大 QK/CIS 抵消、或 CIS 带大公共偏移的有限输入上超出既有误差容限。原因是
缩放/加法融合的舍入差异，以及减去最大值前转换到 base-2 造成的小差值丢失。
当前已修复为先按 FP32 分别舍入 QK 缩放与 CIS 加法，再在原单位中减去最大值。
旧结果不能证明这些边界输入上的正确性，也不能作为修复后实现的性能结果；
本次确认并未表明旧运行的实际输入触发该问题。原产物保留至修复后的完整补测验收。

同日还确认 native score 在有限 QK logits 带大公共偏移时存在同类精度问题：
BF16 的独立构造用例相对 FP32 reference 偏差为 2.54%，超出既有 0.8% 相对容限。
当前 score 已改为自然单位最大值及差值后 base-2 转换，并已通过数值回归；旧 score
结果同样不能证明该输入边界的正确性，原运行是否受影响须按其实际输入另行核验。

另确认原 native attention 的未归一化 PV 累加在极大但有限的 BF16 V 上可能溢出。
当前检查点已接入按实际 token 数缩放 V、归一化后恢复尺度及最终 dtype 转换保护，
并完成 361 项 GPU 回归和 18 组完整模块归因检查。旧报告不能证明上述边界的正确性；
尚无证据表明原实验输入触发这些问题，原产物保留至受影响实验补测验收后再替换。

当前已验收实现与未完成事项见 [实现检查点](../../docs/kda/README.md#resident-联合验收与报告状态)；
完整模块 run `kda_main_bf16_pair_v3_development` 的原始摘要和源码指纹已在
[模块检查点报告](../nosa_kernel_mfu/README.md#完整模块检查点) 发布。
该三层模块测量不替代本节原实验，原表格数字及 run ID 保持原测量含义。
另有原实现的真实输入基线 `kda_real_baseline_20260928_1352`，输入来自
`kda_inputs_baseline_20260928_1345`，尚未摘入下方 synthetic 报告。

已运行 synthetic resident 测量，run ID 为
`kernel_mfu_h200_gpu1_20260928_071840`。当前 native attention 在四种 prefix 下均快于
Triton；32K / 64K 的 native score 也快于 Triton，但本次各项有效 MFU 均未达到 30%。
下方表格仅汇总 synthetic 输入，不能据此推断模型端到端或 offload 性能。

硬件为 GPU 1，NVIDIA H20Z（本机按 H200 规格，132 SM，700 W 功率上限），PCI 地址
`00000000:67:00.0`；运行前显存占用 4 MiB、GPU 利用率 0%。显式指定标称 dense BF16
峰值 989 TFLOPS。使用 Python 3.12.13、PyTorch 2.12.1+cu130、PyTorch CUDA 构建版本 13.0、
nvcc 13.2.86、Triton 3.7.1、FlashInfer 0.6.18、TVM FFI 0.1.13.post3，
CUTLASS 提交为 `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。
输入形状、分布和 FLOP 口径沿用上文默认值；所有 case 均检查全部 1024 个 query。
本机 CUDA runtime 和 NVML 返回的 UUID 不同，均按原值保留；运行前核对了 CUDA
设备的 PCI bus 为 `0x67`，与 GPU 1 一致。运行后再次核对的
[设备映射记录](report/gpu_identity.json)保存两种 UUID 和相同的 PCI 地址。

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/nosa_kernel_mfu/scripts/run.sh \
  kernel_mfu_h200_gpu1_20260928_071840 \
  --device cuda:0 --peak-tflops 989 --reference-all
```

以下为 CUDA Graph 中每次算子的中位延迟（µs）；预热 10 次、重复 30 次，每次 graph
连续调用 10 次。加速比为 Triton 延迟 / native 配置延迟。4K / 16K score 在 native
配置下仍由 dispatcher 调用 Triton，因此不报告 native kernel 加速比。

| Prefix | 算子 | native 配置实际后端 | native 配置（µs） | Triton（µs） | 加速比 | native 配置 MFU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | attention | CUDA/CuTe | 289.30 | 567.48 | 1.96× | 23.83% |
| 16384 | attention | CUDA/CuTe | 438.25 | 581.28 | 1.33× | 15.73% |
| 32768 | attention | CUDA/CuTe | 459.63 | 584.32 | 1.27× | 15.00% |
| 65536 | attention | CUDA/CuTe | 468.68 | 586.57 | 1.25× | 14.71% |
| 4096 | pooled scores | Triton | 27.42 | 27.28 | — | 8.86% |
| 16384 | pooled scores | Triton | 80.36 | 79.81 | — | 11.13% |
| 32768 | pooled scores | CUDA/CuTe | 123.14 | 147.90 | 1.20× | 14.32% |
| 65536 | pooled scores | CUDA/CuTe | 216.17 | 282.67 | 1.31× | 16.19% |

native attention 的计时覆盖 grouped kernel 和后续 per-query repair kernel，未单独
测量 grouped kernel 或统计 fallback 比例。synthetic 选块的跨 query 重合度会影响
此路径的表现，本表不能外推为真实模型输入上的加速比。

普通 eager 调用在 64K prefix 下，native attention / score 分别为 537.23 / 263.78 µs，
Triton 分别为 651.22 / 313.33 µs。完整 graph/eager 中位数见
[汇总 CSV](report/summary.csv)，全部 30 次样本及 min/max 见
[测量与数值验收数据](report/results.json)。
所有 case 的 graph replay 输出均与同后端 eager 输出精确一致，FP32 reference 验收通过。
attention 最大绝对误差为 0.0009765625，较严格的
`atol=0.001, rtol=0.016` 检查亦无超限元素；score 最大绝对误差不超过 0.000244140625。

[运行元数据](report/metadata.json)记录设备、依赖、源码和输入哈希。
以上三个 `report/` 文件直接复制自该 run 的 `output/data/` 同名文件，表格由
`results.json` 的中位数及其比值整理，未混用旧实现数据。源码快照、按锁文件导出的
安装清单和每秒 GPU 状态分别保存在
`output/data/kernel_mfu_h200_gpu1_20260928_071840/sources/`、
`output/data/kernel_mfu_h200_gpu1_20260928_071840/environment_requirements.txt` 和
`output/data/kernel_mfu_h200_gpu1_20260928_071840/gpu_telemetry.csv`。
stdout/stderr 位于对应的 `output/log/` run 目录，stderr 为空。
