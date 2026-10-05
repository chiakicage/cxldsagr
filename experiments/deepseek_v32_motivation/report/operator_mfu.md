# 算子 MFU

诊断采集为 `refactor_final_deepseek_profile_20261005_01`，对应正式运行 `refactor_final_deepseek_bench_20261005_01`。
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
| index_k_proj | FP8 | 0.81% | 0.82% | 0.79% | 0.79% |
| index_q_proj | FP8 | 13.25% | 13.36% | 13.18% | 13.05% |
| index_weights_proj | FP32 | 11.25% | 11.61% | 11.13% | 11.02% |
| indexer | FP8 | 23.21% | 24.27% | 23.08% | 22.85% |
| kv_a_proj | FP8 | 3.55% | 3.69% | 3.56% | 3.52% |
| lm_head | BF16 | 0.44% | 0.44% | 0.44% | 0.44% |
| mla_qk_pv | BF16 | 57.09% | 58.19% | 55.91% | 55.45% |
| mlp_down | FP8 | 28.76% | 29.68% | 28.35% | 28.25% |
| mlp_gate | FP8 | 39.47% | 39.62% | 38.78% | 38.57% |
| mlp_up | FP8 | 41.13% | 41.34% | 40.81% | 40.27% |
| o_proj | FP8 | 28.09% | 29.06% | 27.72% | 27.58% |
| q_a_proj | FP8 | 9.23% | 9.46% | 9.08% | 9.01% |
| q_absorb | BF16 | 19.22% | 19.75% | 19.39% | 19.33% |
| q_b_proj | FP8 | 23.28% | 23.59% | 23.52% | 23.56% |
| v_expand | BF16 | 20.72% | 20.45% | 20.18% | 20.26% |

## cold / history

| 算子 | 精度 | hbm | echo | serial_sparse | dense_prefetch |
|---|---|---:|---:|---:|---:|
| index_k_proj | FP8 | 4.89% | 5.13% | 4.95% | 4.96% |
| index_q_proj | FP8 | 43.51% | 45.66% | 44.05% | 44.08% |
| index_weights_proj | FP32 | 33.73% | 35.06% | 34.09% | 34.10% |
| indexer | FP8 | 45.74% | 48.90% | 46.39% | 46.37% |
| kv_a_proj | FP8 | 15.02% | 15.68% | 15.12% | 15.15% |
| lm_head | BF16 | 0.44% | 0.44% | 0.44% | 0.44% |
| mla_qk_pv | BF16 | 59.35% | 63.57% | 60.19% | 60.24% |
| mlp_down | FP8 | 55.38% | 57.53% | 56.01% | 56.02% |
| mlp_gate | FP8 | 57.87% | 60.29% | 58.53% | 58.54% |
| mlp_up | FP8 | 59.35% | 60.83% | 59.93% | 59.91% |
| o_proj | FP8 | 55.06% | 57.37% | 55.70% | 55.73% |
| q_a_proj | FP8 | 40.33% | 42.42% | 41.07% | 41.05% |
| q_absorb | BF16 | 29.60% | 30.10% | 29.68% | 29.69% |
| q_b_proj | FP8 | 57.35% | 59.69% | 57.93% | 57.94% |
| v_expand | BF16 | 33.05% | 33.37% | 33.10% | 33.14% |

## revisit / candidate

| 算子 | 精度 | hbm | echo | serial_sparse | dense_prefetch |
|---|---|---:|---:|---:|---:|
| index_k_proj | FP8 | 0.80% | 0.86% | 0.86% | 0.87% |
| index_q_proj | FP8 | 13.03% | 13.77% | 13.68% | 13.91% |
| index_weights_proj | FP32 | 11.05% | 12.11% | 12.03% | 12.08% |
| indexer | FP8 | 22.71% | 8.39% | 25.51% | 25.63% |
| kv_a_proj | FP8 | 3.53% | 3.85% | 3.82% | 3.84% |
| lm_head | BF16 | 0.44% | 0.44% | 0.43% | 0.44% |
| mla_qk_pv | BF16 | 56.00% | 60.32% | 60.52% | 60.82% |
| mlp_down | FP8 | 28.27% | 30.69% | 30.95% | 31.14% |
| mlp_gate | FP8 | 39.47% | 39.36% | 39.50% | 40.10% |
| mlp_up | FP8 | 41.03% | 41.95% | 41.42% | 42.08% |
| o_proj | FP8 | 27.66% | 30.33% | 30.29% | 30.50% |
| q_a_proj | FP8 | 9.08% | 9.86% | 9.88% | 9.68% |
| q_absorb | BF16 | 19.20% | 20.58% | 20.46% | 19.61% |
| q_b_proj | FP8 | 23.38% | 24.17% | 24.17% | 23.83% |
| v_expand | BF16 | 20.31% | 19.98% | 19.78% | 20.79% |

## revisit / history

| 算子 | 精度 | hbm | echo | serial_sparse | dense_prefetch |
|---|---|---:|---:|---:|---:|
| index_k_proj | FP8 | 4.87% | — | — | — |
| index_q_proj | FP8 | 43.28% | — | — | — |
| index_weights_proj | FP32 | 33.60% | — | — | — |
| indexer | FP8 | 45.41% | — | — | — |
| kv_a_proj | FP8 | 14.96% | — | — | — |
| lm_head | BF16 | 0.44% | — | — | — |
| mla_qk_pv | BF16 | 58.97% | — | — | — |
| mlp_down | FP8 | 55.04% | — | — | — |
| mlp_gate | FP8 | 57.49% | — | — | — |
| mlp_up | FP8 | 58.94% | — | — | — |
| o_proj | FP8 | 54.75% | — | — | — |
| q_a_proj | FP8 | 40.23% | — | — | — |
| q_absorb | BF16 | 29.63% | — | — | — |
| q_b_proj | FP8 | 57.08% | — | — | — |
| v_expand | BF16 | 32.99% | — | — | — |

Indexer 一行按实际 resident/fused 调用合并，使用总理论时间除以总 GPU 活动时间。
融合 indexer 的矩阵 kernel 同时包含预取和标量工作，当前 trace 不能把这些工作拆开。
主 kernel 仅计矩阵入口；split-K reduction、scale 转置等辅助节点保留在完整 API 时间中。

每方案采集一个首访和一个复访请求。不同 history chunk 的上下文长度不同，不能当作同形状算子的独立重复测量。
本报告反映诊断进程中的算子表现；正式端到端 MFU 使用对应未插桩运行的请求延迟。

[完整数据](operator_mfu/operator_mfu.csv)、[逐层数据](operator_mfu/operator_mfu_by_layer.csv)、
[kernel 清单](operator_mfu/operator_kernel_inventory.csv)、[汇总与定义](operator_mfu/operator_mfu_summary.json)、
[独立复核](operator_mfu/crosscheck.json)和[发布来源](operator_mfu/publication.json)保留核验信息。

分析 ID：`operator_mfu`。逐调用数据位于 `/mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_motivation/output/data/refactor_final_deepseek_profile_20261005_01/analysis/operator_mfu/operator_mfu_calls.jsonl`。
