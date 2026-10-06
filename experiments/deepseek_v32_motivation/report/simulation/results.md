# 单层计算与 I/O simulation

Simulation run ID：`deepseek_compute_io_simulation_20261006_04`。测量输入来自 `deepseek_mfu_dma_a128_profile_20261006_01`，平台为 H200 / SM90。
源实验的环境、依赖与独立验收见 [MFU 说明](../../../deepseek_v32_mfu/README.md)，选定原始区间见 [MFU timeline CSV](../../../deepseek_v32_mfu/report/four_methods/timeline_activities.csv)。

按 motivation 的 H=65,536、A=128、history chunk=1,024 设置，选从 0 编号的 L1。Prefill 只取最后一个 chunk（63），此前已有 64,512 tokens；extend 使用完整 128-token batch。
计时输入来自 MFU 实验中真实 checkpoint 前三层连续执行的 L1，同一形状不表示与 C10 的某个 GR 请求有相同 token 或激活。Motivation 的十 block、P=65,536、NH=16,777,216 作为目标背景；本次只模拟一层，不运行准入或容量轨迹，也不推算完整十层或全部 64 个 chunk。

## 实测成本与排除范围

解析方案共用 HBM 路径的计算 kernel 时长；ECHO 行单独替换其已测 fused indexer、causal mask 和 recall，其他阶段仍用公共成本。计算按阶段累加 kernel duration，保留 norm、RoPE、量化、top-k 等计算辅助操作。去掉 CPU launch、CPU scope、独立的 cache 管理、映射、D2D、memset 和原时间线的空隙；融合 kernel 内部操作随整体时长保留。因 KV 尚未到齐产生的依赖等待仍保留。

| 成本 | Prefill L1 / chunk 63 ms | Extend L1 ms |
| --- | ---: | ---: |
| Norm + projection | 0.378111 | 0.143040 |
| Indexer | 1.156542 | 0.270816 |
| Exact top-k | 0.394816 | 0.083392 |
| Sparse MLA | 0.969374 | 0.120672 |
| Output + norm + MLP | 1.014174 | 0.203200 |
| 计算合计 C | 3.913017 | 0.821120 |
| 主 KV D2H W | 0.026720 | 0（candidate discard） |
| 稀疏 H2D S | 0 | 0.071968 |
| 当前层完整历史 H2D D₁ | 0 | 1.374494 |

Prefill 的 D2H 为 1,179,648 B（1.125 MiB）；sparse extend 为 3,312,000 B，dense 为 75,497,472 B（72 MiB），本层流量比为 22.80×。Sparse 使用已测 mapped-host gather 时长，dense 使用已测 `cudaMemcpyAsync` DMA 时长；两者均不是按名义带宽估计。主 KV 为 BF16 512 latent + 64 RoPE，1,152 B/token；线性层采用 FP8，indexer K/scales 留在 HBM。

MFU 的实际 pool=65,664，extend 是持久 append，每层另有 147,456 B candidate D2H。这里按 motivation 的 candidate discard 语义排除该写回，只保留历史 H2D；也不把源 pool 的分配或命中行为移植成 P=65,536 的容量结论。每种方法的每个阶段只有一次侵入式 profile，时长没有重复采样置信区间。

## 计算 MFU 与通信带宽

FLOPs 来自与图中 L1/chunk 对应的调用账本，每个 prefill/extend 样本匹配 14 个矩阵 API，FMA 计 2 FLOPs。Graph API 通过 replay 归属匹配；prefill 使用 chunk 63 的实际 causal pair 数，没有把完整 prefill 工作量除以 64。
混合精度先分别归一化：T_ideal=Σ(FLOPs_d/Peak_d)，纯计算 MFU=T_ideal/计算 kernel 时长之和。FP8、BF16、FP32 的 H200 dense peak 分别为 1,979、989.5、67 TFLOP/s，来源为 [NVIDIA H200 规格](https://www.nvidia.com/en-us/data-center/h200/)。不把混合 FLOPs 全部除以 FP8 峰值，也不对逐算子 MFU 求算术平均。

| 计算阶段 | Prefill GFLOPs | Prefill 计算 MFU | Extend GFLOPs | Extend 计算 MFU |
| --- | ---: | ---: | ---: | ---: |
| Norm + projection | 154.081952 | 26.47% | 19.260244 | 8.75% |
| Indexer | 1090.930082 | 47.66% | 137.574220 | 25.67% |
| Exact top-k | N/A | N/A | N/A | N/A |
| Sparse MLA | 584.115552 | 60.90% | 73.014444 | 61.15% |
| Output + norm + MLP | 1069.446857 | 54.14% | 133.680857 | 33.78% |
| 全部计算 | 2898.574442 | 45.76% | 363.529765 | 27.33% |

Top-k 没有矩阵 FLOPs，MFU 为 N/A；它的时间仍计入全部计算分母。Norm、RoPE、量化、softmax 等非矩阵工作也不增加 useful matrix FLOPs，但其 kernel 时间保留。这里的 MFU 不等于 NCU 的硬件利用率。

通信量统计各层完整搬运的主 KV payload；indexer 保持驻留，控制传输、D2D 和 candidate D2H 不计入。有效带宽=字节数/完整搬运区间，GB/s 使用十进制，MiB 使用 2²⁰ B。即使搬运被隐藏或在图中截断，带宽分母仍是完整搬运时长，不能改用暴露的等待时间或可见片段。它不是链路含协议开销的物理吞吐。

| 搬运 | Records | Payload B | MiB | 完整区间 ms | 有效带宽 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill L1 D2H | 1,024 | 1,179,648 | 1.125000 | 0.026720 | 44.149 |
| Extend sparse L1 H2D | 2,875 | 3,312,000 | 3.158569 | 0.071968 | 46.020 |
| Extend dense L1 H2D | 65,536 | 75,497,472 | 72.000000 | 1.374494 | 54.927 |
| Dense L2 H2D（相邻层上下文） | 65,536 | 75,497,472 | 72.000000 | 1.374430 | 54.930 |
| ECHO residual recall H2D | 269 | 309,888 | 0.295532 | 0.011008 | 28.151 |
| ECHO fused prefetch（整个融合区间） | 2,606 | 3,002,112 | 2.863037 | 0.775775 | N/A |

图中另外标出调度 MFU=T_ideal/模拟完成时间，包含必要的 I/O 等待，与纯计算 MFU 分开。Dense 的分子和完成时间均对应 L1；L2 搬运仅用于说明相邻层的衔接，不计入 L1 的通信量或 MFU。截断片段不按时间比例估算字节数。完整精度分量和数值见 [计算指标](compute_metrics.csv)及[通信指标](io_metrics.csv)。

## Prefill

令 p、i、k、a、f 分别为 projection、indexer、top-k、attention、finish 的时长，C=p+i+k+a+f。该 chunk 的历史已驻留，没有历史 H2D。串行写回为 C+W；理想异步写回在 projection 产出主 KV 后启动，结束时间为 max(C,p+W)。假设写回源能保持有效、传输与计算互不减速，不计管理它们的开销。

| 模拟方案 | L1 输出及必要写回完成 ms | 调度 MFU |
| --- | ---: | ---: |
| HBM resident | 3.913017 | 45.76% |
| Offload: serial writeback | 3.939737 | 45.45% |
| Offload: overlapped writeback | 3.913017 | 45.76% |

![Prefill 模拟时间线](simulation_prefill.svg)

## Extend

主表从 L1 P 开始计到 L1 输出就绪。HBM 行假设主 KV 已驻留；sparse 与 ECHO 行从本层尚未搬入历史开始，dense 的历史搬运则接在上一层 fetch 后面，部分发生在零点之前。

- 串行 sparse：精确 top-k 后搬入选择集，再计算 attention，T=C+S。
- 理想 sparse overlap：假设 indexer 开始时已提前获知需要搬运的精确集合，H2D 与 indexer 重叠，T=p+max(i,S)+k+a+f。这是显式的重叠收益上界假设；当前 indexer 完成前通常没有完整精确集合，因此不能当作 ECHO 的已实现时序或预测性能。
- Dense 跨层预取：本层完整历史 H2D 接在上一层 fetch 后，P 仍从 t=0 开始，attention 等待本层 H2D 与 top-k 完成。令 fetch 提前量为 Δ，则 T=max(p+i+k,D₁−Δ)+a+f。提前量由上一层实测成本确定，详见下文。

| 模拟方案 | L1 输出就绪 ms | 调度 MFU |
| --- | ---: | ---: |
| HBM resident | 0.821120 | 27.33% |
| Serial sparse fetch | 0.893088 | 25.13% |
| Ideal sparse overlap (oracle) | 0.821120 | 27.33% |
| ECHO: measured fused + recall | 1.350686 | 16.62% |
| Dense prefetch: chained across layers | 1.370143 | 16.38% |

本层 S<i，理想情况下可隐藏全部稀疏搬运时间，较串行 sparse 的收益上限为 1.088×。这组成本下，完整历史传输仍长于可用于隐藏它的计算窗口。资源竞争和调度可行性不在模型内，不能据此宣布实际 ECHO 或 dense 实现的加速比。

![Extend 模拟时间线](simulation_extend.svg)

## Dense 跨层预取的时间基准

横轴零点始终是本层 L1 的 P 开始。Dense 在前一层 fetch 结束后立即启动本层 fetch；L1 fetch 结束后又立即启动 L2 fetch。图只显示 L1 P 开始到 L1 输出就绪的窗口，因此左侧为 L1 fetch 的剩余部分，右侧为 L2 fetch 的开头。

使用同一 MFU profile 的 L0 实测成本确定零点前的进度：P₀+I₀+K₀=0.493728 ms，A₀+F₀=0.328223 ms，完整 DMA D₀=1.373758 ms。令 L0 P 与 L0 DMA 同时开始，则 L0 完成于 max(P₀+I₀+K₀,D₀)+A₀+F₀；L1 fetch 比 L1 P 提前 Δ=max(P₀+I₀+K₀−D₀,0)+A₀+F₀=0.328223 ms 启动。L0 的逐算子独占时长、调用数和 kernel 汇总已交叉核对，选定源行随 inputs.json 保存。

L1 的计算仍按 P→I→K→A→F 执行，A 等待本层 fetch。因此 L1 完成时间 T=max(p+i+k,D₁−Δ)+a+f，等待时间为 max(D₁−Δ−p−i−k,0)。这里 Δ 来自相邻实测层，没有假定各层计算时间相同。

L1 完整 fetch 区间为 [-0.328223, 1.046271] ms，图中可见 [0.000000, 1.046271] ms；KV 等待为 0.549023 ms，L1 输出在 1.370143 ms 就绪。L2 fetch 从 1.046271 ms 开始，图中仅保留至 L1 输出的 0.323872 ms；其完整结束时刻 2.420701 ms 不延长 L1 窗口。

L1 通信量仍为完整 72 MiB，带宽仍用完整 D₁=1.374494 ms 计算。本层 fetch 提前启动，减少了本层等待。图中从零点开始显示 L1 fetch 的剩余部分，末尾接上 L2 fetch 的开头。L0 和 L1 的 A+F 时长略有差异，因此该窗口不等于假定同构层的稳态周期，也不代表完整模型吞吐。

## 已测 ECHO fused prefetch 与 recall

图中的 ECHO 行保留本次 profile 的融合 indexer/prefetch、causal mask 和剩余精确 recall 时长，其他 P/K/A/F 阶段沿用公共 HBM 成本。图外的 cache 管理及 CPU launch 已排除；融合 kernel 内部的管理和同步无法从现有计时中剥离。因此整行是按已测组件重排的 simulation，不是 ECHO 完整层延迟。这里的“已测”指上述 run ID；不替代后续代码版本的补测。

融合 kernel 耗时 0.775775 ms，预取 2,606 records（3,002,112 B），覆盖本层 90.64% 的历史 miss；后续 recall 补齐 269 records（309,888 B），耗时 0.011008 ms。两部分合计 3,312,000 B，与串行 sparse 的历史 H2D 一致。Candidate 的 128 records 已在 GPU，不计入历史预取覆盖率。

融合区间的 MFU 为 8.96%，分母包含融合计算与搬运。其平均载荷率为 3.870 GB/s，分母同样是完整 fused kernel；内部 IO 时间不可分解，故不能称为独立通信带宽。图上使用一个跨计算/I/O 两条 lane 的区间，不虚构内部 overlap 比例。

融合 kernel 后的 causal score mask 是必要计算，单独保留 13.599 µs，不增加矩阵 FLOPs。此行模拟完成时间为 1.350686 ms，调度 MFU 为 16.62%；纯计算 MFU 因融合区间无法拆分而记为 N/A。Prefetch 减少了后续 recall 的通信量，但不能仅凭覆盖率推出整层加速。

## 核验与复现

模拟使用整数 ns，核验每个方案的计算阶段顺序、P 的零点、成本守恒、依赖等待和结束边界。Dense 的完整 DMA 元数据保留负起点及窗口外终点，图示区间和 timeline.csv 仅保存与 L1 窗口的交集；核验二者一致，并将下一层通信与本层记账分开。模型计算与 I/O 时长固定，不建模带宽竞争、SM 争用、融合 kernel 的内部时序或未测输入。原始 fused ECHO kernel 未被拆成虚构的实测分量。

[输入及来源哈希](inputs.json)、[模拟区间](timeline.csv)、[数值汇总](summary.csv)、[模拟与核验](simulation.json)、[发布清单](publication_manifest.json)随报告保存。输入提取读取已发布数据及与之绑定的原始 FLOPs 调用账本，不启动 GPU；选定调用随 inputs.json 保存。本次结果是 CPU simulation，不是新的硬件计时或正确性验收。

从仓库根目录运行（使用新的 run ID 与输出目录）：

```bash
.venv/bin/python -m experiments.deepseek_v32_motivation.src.simulation_inputs \
  --run-id simulation_new \
  --output experiments/deepseek_v32_motivation/output/data/simulation_new/inputs.json
.venv/bin/python -m experiments.deepseek_v32_motivation.src.simulate \
  --inputs experiments/deepseek_v32_motivation/output/data/simulation_new/inputs.json \
  --output-dir experiments/deepseek_v32_motivation/output/data/simulation_new/report
```

本次完整产物保留在 `experiments/deepseek_v32_motivation/output/data/deepseek_compute_io_simulation_20261006_04/`。未替换原 motivation 硬件实测报告。
