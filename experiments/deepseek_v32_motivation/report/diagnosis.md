# C10 请求与单层执行诊断

当前 profile 为 `deepseek_mfu_c10_profile_20261006_02`，对应正式计时 `deepseek_mfu_c10_bench_20261006_01` 和独立数值检查 `deepseek_mfu_c10_check_20261006_01`。三者均为 H=P=65,536、A=128、history chunk=1,024、NH=16,777,216，使用同一 C10 checkpoint 工作负载替身。正式请求延迟见[计时报告](results.md)；这里仅解释带 instrumentation 的调度与 GPU activity。

## 当前图表

- [Candidate 单层实测时间线](single_layer/results.md)：request 16、L1，保留全部 stream 的活动和空隙。
- [Dense L1 compute 与完整 L2 gather](single_layer/dense_prefetch_actual.svg)：重叠 1.097090 / 1.837602 ms，即 59.702264%；分母包含 L2 gather 超过 L1 finish 的部分。
- [History prefill 单层时间线](prefill_layer/results.md)：request 0、chunk 63、L1，显示三条 offload 路径各 1.125 MiB 主 KV 写回；选定窗口内没有写回与 compute kernel 的重叠。

| 方案 | Cold request | Revisit request |
| --- | --- | --- |
| HBM-only | [完整请求](diagnosis/pipeline_hbm_cold.svg) | [完整请求](diagnosis/pipeline_hbm_revisit.svg) |
| ECHO | [完整请求](diagnosis/pipeline_echo_cold.svg) | [完整请求](diagnosis/pipeline_echo_revisit.svg) |
| Sparse fetch | [完整请求](diagnosis/pipeline_serial_sparse_cold.svg) | [完整请求](diagnosis/pipeline_serial_sparse_revisit.svg) |
| Dense prefetch | [完整请求](diagnosis/pipeline_dense_prefetch_cold.svg) | [完整请求](diagnosis/pipeline_dense_prefetch_revisit.svg) |

完整请求图与 [stage_costs.csv](diagnosis/stage_costs.csv)用于定位阶段，不能把不同 capture 的时间轴拼接，也不能将侵入式 CUDA API 或 GPU window 当作正式 benchmark 延迟。算子与真实前三层完整执行的 MFU 统一见 [DeepSeek V3.2 MFU](../../deepseek_v32_mfu/README.md)，本目录不维护第二套算子 MFU 结果。

## 数值、来源与图形核验

本次 profile 保存 12 次 capture：4 次 graph 准备、4 次 cold request 和 4 次 revisit request。独立 CPU 审查逐字节比较了全部 80 份 candidate hidden/logits，与独立 HBM check 一致；其中包含 12 份 warmup、60 份复访准备以及 8 份 cold/revisit 输出。审查不导入项目 validation 代码，也未初始化 CUDA。见[数值审查](diagnosis/independent_numerical_audit.json)。

独立来源审查重新计算了 1,297 个 profile 源码快照文件和 703 个运行时源码/native 文件的哈希；1,278 项已接受的执行源码身份与 bench 相同。输入、checkpoint 身份、native 后端、精度、四方案 resource plan 和 token validator 身份均匹配，graph eager fallback 为零。完整 profile 源码摘要为 `8a75ecbaa0c8c5262cbc78022d7582b8a7e82ac61927ae3f558797edeb703227`，见[来源审查](diagnosis/independent_provenance_audit.json)。

图形审查直接读取原始 SQLite，将 candidate 和 prefill 两组窗口中的每条 kernel、memcpy、memset 及 kernel 名称逐项对回图表数据；所有可见活动均保留。另行核对全部 memcpy 的字节数和方向，见[搬运审查](diagnosis/independent_transfer_audit.json)。Dense L2 完整 gather 通过 launch correlation 和最内层 host-gather NVTX 归属复核。图表源码的身份及原始数据哈希保存在各自 `timeline.json` 和[发布清单](publication_manifest.json)中。

Candidate 窗口里仍有明显提交空隙：HBM-only、ECHO、sparse fetch 的最大空隙分别为 114.305、175.136、133.088 µs。HBM 的主要间隔覆盖 node-traced graph launch；ECHO 的最大间隔位于 prefetch 准备到 fused indexer 启动之间；sparse fetch 则位于 exact recall 的映射完成到 MLA 启动之间。详细相邻 kernel、CUDA API 与 host NVTX 见[间隔上下文](diagnosis/gap_context.json)。没有 Python 栈或 CPU 调度记录，不能将这些间隔全部归为可消除开销。ECHO fused kernel 内部的计算与 host-read 时间也未被分解。

## 环境与重建

Profile 使用 H200 / SM90、物理 GPU 3、CPU24–31 / NUMA0。Nsight Systems 启用 `cuda,nvtx,osrt` 和 node-level graph tracing，关闭 CPU sampling 与 context-switch tracing；没有采集 Nsight Compute 硬件计数。305 次离散观测的最大间隔为 31.677917 秒，未观察到所选 GPU 上的其他进程。同期 GPU 1 有三个其他 PID，采样利用率最高为 73%；本轮不属于全机独占运行，也不能据离散采样声称连续隔离。见[观测摘要](diagnosis/observer_summary.json)。

完整数据在 `output/data/deepseek_mfu_c10_profile_20261006_02/`，原始 profiler 文件在 `output/profile/deepseek_mfu_c10_profile_20261006_02/`。`src.analyze_pipeline` 读取 capture 2/3、5/6、8/9、11/12；`src.plot_single_layer` 选 revisit/L1，`src.plot_prefill_timeline` 选 cold/chunk63/L1。分析复用 `experiments.deepseek_v32_mfu.src.analyze_nsys`。全部命令从仓库根目录运行，来源快照与独立审查 helper 随对应 output 保存。
