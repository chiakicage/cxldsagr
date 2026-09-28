# NOSA attention / indexer score 的有效 MFU

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
`operators/sm90/tests/`，验收记录不能作为模型等价性证明。

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
```

需要 Hopper/SM90、nvcc、共享 CUTLASS、TVM FFI、PyTorch、Triton 和 FlashInfer，沿用项目
环境。以上命令按本机已确认的 H200 配置显式指定 989 TFLOPS；设备上报 H20Z 时，
名称检测不会自动识别为 H200，必须保留该参数；其他型号应填写相应的 dense BF16 峰值。
脚本从任意工作目录定位仓库根目录，先在系统临时目录测量，成功后发布到新 run ID；
失败输出留在实验目录外。测量过程中源码变化会拒绝发布，已有 run ID 拒绝覆盖。

`src/measure.py` 调用 `models.nosa.indexer` 的压缩与 FP32 score reference、
`operators.sm90.nosa_indexer` 和 `operators.sm90.nosa_attention` 的现有算子；
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

## 结果与结论

已运行 synthetic resident 测量，run ID 为
`kernel_mfu_h200_gpu1_20260928_071840`。当前 native attention 在四种 prefix 下均快于
Triton；32K / 64K 的 native score 也快于 Triton，但本次各项有效 MFU 均未达到 30%。
真实模型输入的测量入口尚未运行，不能据此推断模型端到端或 offload 性能。

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
