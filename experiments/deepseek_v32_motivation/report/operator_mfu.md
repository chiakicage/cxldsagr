# 算子 MFU

诊断采集为 `motivation_c10_profile_20261004_01`，对应正式运行 `motivation_c10_20261004_u16_r2_01`。
本报告处理 8 个请求 capture、45,933 次矩阵调用。
算子时间取每次 API 所属 GPU 活动的时间并集，包括内部量化、转置、归约和搬运，再按阶段累计。
MFU 为累计理论计算时间除以累计实测时间；理论时间按每次调用的有效 FLOPs 和对应精度峰值计算。
CPU 提交、API 外的 cache 管理和 GPU 发射间隙仍保留在端到端时间中，不计入此处的算子分母。

其中 6,560 次 CUDA Graph 重放共记录 180,320 个 GPU 节点活动。
图内 API 用捕获时记录的节点集合归属；实例化克隆关系取自 Nsight 的 originalGraphNodeId。
分母使用本次重放的实际节点时间，未借用 eager 运行时间。图内 API 行不声明独立 CPU 范围。

参考 dense 峰值（TFLOPS）：FP8=1979, BF16=989.5, FP32=67。精度设置和峰值来源保存在汇总 JSON。

## cold / candidate

| 算子 | 精度 | hbm | echo | serial_sparse | dense_prefetch |
|---|---|---:|---:|---:|---:|
| index_k_proj | FP8 | 0.76% | 0.82% | 0.81% | 0.82% |
| index_q_proj | FP8 | 12.73% | 13.43% | 13.15% | 13.31% |
| index_weights_proj | FP32 | 10.57% | 11.43% | 11.34% | 11.51% |
| indexer | FP8 | 21.81% | 23.78% | 23.61% | 23.92% |
| kv_a_proj | FP8 | 3.39% | 3.64% | 3.63% | 3.67% |
| lm_head | BF16 | 0.44% | 0.44% | 0.44% | 0.44% |
| mla_qk_pv | BF16 | 53.92% | 57.30% | 56.95% | 58.03% |
| mlp_down | FP8 | 27.23% | 29.19% | 28.88% | 29.44% |
| mlp_gate | FP8 | 38.45% | 39.45% | 39.00% | 39.30% |
| mlp_up | FP8 | 39.72% | 40.66% | 40.82% | 40.95% |
| o_proj | FP8 | 26.61% | 28.58% | 28.35% | 28.85% |
| q_a_proj | FP8 | 8.78% | 9.34% | 9.32% | 9.44% |
| q_absorb | BF16 | 18.76% | 19.53% | 19.43% | 19.82% |
| q_b_proj | FP8 | 22.82% | 23.53% | 23.45% | 23.56% |
| v_expand | BF16 | 20.29% | 20.34% | 19.91% | 20.53% |

## cold / history

| 算子 | 精度 | hbm | echo | serial_sparse | dense_prefetch |
|---|---|---:|---:|---:|---:|
| index_k_proj | FP8 | 4.92% | 5.16% | 4.99% | 4.98% |
| index_q_proj | FP8 | 43.89% | 46.12% | 44.41% | 44.44% |
| index_weights_proj | FP32 | 33.88% | 35.18% | 34.26% | 34.23% |
| indexer | FP8 | 46.20% | 49.07% | 46.71% | 46.72% |
| kv_a_proj | FP8 | 15.11% | 15.75% | 15.22% | 15.23% |
| lm_head | BF16 | 0.44% | 0.44% | 0.44% | 0.44% |
| mla_qk_pv | BF16 | 60.05% | 64.07% | 60.57% | 60.56% |
| mlp_down | FP8 | 56.01% | 58.18% | 56.52% | 56.54% |
| mlp_gate | FP8 | 58.51% | 60.91% | 59.09% | 59.10% |
| mlp_up | FP8 | 60.08% | 61.90% | 60.57% | 60.61% |
| o_proj | FP8 | 55.67% | 57.98% | 56.19% | 56.21% |
| q_a_proj | FP8 | 40.73% | 42.96% | 41.45% | 41.44% |
| q_absorb | BF16 | 29.63% | 30.05% | 29.68% | 29.69% |
| q_b_proj | FP8 | 57.98% | 60.48% | 58.52% | 58.52% |
| v_expand | BF16 | 32.94% | 33.31% | 33.07% | 33.10% |

## revisit / candidate

| 算子 | 精度 | hbm | echo | serial_sparse | dense_prefetch |
|---|---|---:|---:|---:|---:|
| index_k_proj | FP8 | 0.79% | 0.87% | 0.87% | 0.87% |
| index_q_proj | FP8 | 13.14% | 13.91% | 13.97% | 13.97% |
| index_weights_proj | FP32 | 10.99% | 12.13% | 12.04% | 12.07% |
| indexer | FP8 | 22.58% | 8.46% | 25.31% | 25.62% |
| kv_a_proj | FP8 | 3.51% | 3.85% | 3.81% | 3.84% |
| lm_head | BF16 | 0.44% | 0.44% | 0.43% | 0.44% |
| mla_qk_pv | BF16 | 55.95% | 60.62% | 60.57% | 60.93% |
| mlp_down | FP8 | 28.17% | 30.83% | 30.74% | 31.09% |
| mlp_gate | FP8 | 39.50% | 39.87% | 39.82% | 40.33% |
| mlp_up | FP8 | 41.44% | 42.26% | 41.93% | 42.34% |
| o_proj | FP8 | 27.45% | 30.33% | 30.30% | 30.69% |
| q_a_proj | FP8 | 9.01% | 10.02% | 9.99% | 9.71% |
| q_absorb | BF16 | 19.01% | 20.32% | 20.38% | 19.66% |
| q_b_proj | FP8 | 23.07% | 24.56% | 24.65% | 24.05% |
| v_expand | BF16 | 20.68% | 20.28% | 19.96% | 20.75% |

## revisit / history

| 算子 | 精度 | hbm | echo | serial_sparse | dense_prefetch |
|---|---|---:|---:|---:|---:|
| index_k_proj | FP8 | 4.89% | — | — | — |
| index_q_proj | FP8 | 43.48% | — | — | — |
| index_weights_proj | FP32 | 33.69% | — | — | — |
| indexer | FP8 | 45.59% | — | — | — |
| kv_a_proj | FP8 | 14.98% | — | — | — |
| lm_head | BF16 | 0.44% | — | — | — |
| mla_qk_pv | BF16 | 59.16% | — | — | — |
| mlp_down | FP8 | 55.35% | — | — | — |
| mlp_gate | FP8 | 57.78% | — | — | — |
| mlp_up | FP8 | 59.23% | — | — | — |
| o_proj | FP8 | 55.02% | — | — | — |
| q_a_proj | FP8 | 40.48% | — | — | — |
| q_absorb | BF16 | 29.62% | — | — | — |
| q_b_proj | FP8 | 57.49% | — | — | — |
| v_expand | BF16 | 32.93% | — | — | — |

Indexer 一行按实际 resident/fused 调用合并，使用总理论时间除以总 GPU 活动时间。
融合 indexer 的矩阵 kernel 同时包含预取和标量工作，当前 trace 不能把这些工作拆开。
主 kernel 仅计矩阵入口；split-K reduction、scale 转置等辅助节点保留在完整 API 时间中。

每方案采集一个首访和一个复访请求。不同 history chunk 的上下文长度不同，不能当作同形状算子的独立重复测量。
本报告反映诊断进程中的算子表现；正式端到端 MFU 使用对应未插桩运行的请求延迟。

[完整数据](operator_mfu/operator_mfu.csv)、[逐层数据](operator_mfu/operator_mfu_by_layer.csv)、
[kernel 清单](operator_mfu/operator_kernel_inventory.csv)、[汇总与定义](operator_mfu/operator_mfu_summary.json)、
[独立复核](operator_mfu/crosscheck.json)和[发布来源](operator_mfu/publication.json)保留核验信息。

分析 ID：`operator_mfu`。逐调用数据位于 `/mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_motivation/output/data/motivation_c10_profile_20261004_01/analysis/operator_mfu/operator_mfu_calls.jsonl`。
