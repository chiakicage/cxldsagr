## 前三层重新测量：`refactor_three_layers_profile_20261005_01`

完整 checkpoint 的第 0–2 层依次传播 hidden/residual，包含 embedding、final norm 和最后 token LM head。Prefix=65,536，extend=1,024，chunk=1,024，offload pool=16,384 tokens/层。仅代表前三层。

延迟与利用率分母来自独立 bench `refactor_three_layers_bench_20261005_01`；其 result SHA256 为 `a625e71e78bf54734b6a6335b35f54e1e3c904f854cc75c0797438acbdcacce6`，源码清单见 summary.json。

| 阶段 | Resident 中位延迟 (ms) | Offload 中位延迟 (ms) | 每种模式重复次数 |
| --- | ---: | ---: | ---: |
| prefix | 642.340 | 1190.485 | 3 |
| extend | 13.954 | 27.632 | 5 |

![前三层无插桩延迟](latency.svg)

端到端精度归一化利用率 = 100 × Σ精度（useful matrix FLOPs / 对应精度 dense peak）/ 无插桩同步 wall time。分子使用同一 run、同一阶段完整 annotated 调用账本中的逻辑矩阵工作量，按 FP8/BF16/FP32 分别换算理想计算时间；新 schema 的分母来自匹配的独立 bench，包含整个请求阶段的CPU 调度、非矩阵计算、搬运、等待和 launch gap。它是当前 absorbed-MLA 实现的三层工作负载指标，不外推完整 61 层，也不等于单一峰值 MFU 或 Tensor pipe active。

| 阶段 | 理想矩阵计算时间 (ms) | Resident 端到端利用率 (%) | Offload 端到端利用率 (%) |
| --- | ---: | ---: | ---: |
| prefix | 289.549 | 45.08 | 24.32 |
| extend | 5.400 | 38.70 | 19.54 |

表中利用率由中位 wall time 计算；每次重复的比率、分精度 FLOPs 与理想计算时间见 [summary.json](summary.json) 的 `end_to_end_utilization`。

下面保留 `operator_mfu` 数据字段名；其含义是算子 kernel 利用率 = useful matrix FLOPs /（同算子实际 GPU kernel duration 之和 × 对应精度 dense peak）。FMA 计 2 FLOPs；FP8/BF16/FP32 分母分别为 1979.0/989.5/67.0 TFLOPS；TF32 关闭。前缀汇总全部 chunk 与三层，extend 汇总三层的一次完整 query batch（含实际 offload leaf 拆分）。LM head 每个阶段仅执行最后一个 token。非矩阵算子的 MFU 为 N/A。

每种模式、每个阶段各采集一次独立 annotated profile；算子 MFU 没有重复采样置信区间。FP8Linear 的分母包含量化和 GEMM，indexer 包含 logits API 的辅助 kernel；这些是按完整算子口径计算的 MFU，与单 GEMM 或 Tensor pipe active 指标不同。算术与归因检查不证明计算实现或 cache 策略合理，MFU 及性能解释须独立验收。

![四组逐算子 MFU](mfu.svg)

### Prefix 矩阵算子

| 算子 | 精度 | Resident kernel ms | MFU (%) | Offload kernel ms | MFU (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| q_a_proj | FP8 | 5.155 | 42.44 | 5.130 | 42.65 |
| q_b_proj | FP8 | 12.429 | 60.35 | 12.433 | 60.33 |
| q_absorb | BF16 | 10.871 | 30.66 | 10.698 | 31.16 |
| kv_a_proj | FP8 | 5.276 | 15.55 | 5.228 | 15.69 |
| index_q_proj | FP8 | 5.472 | 45.69 | 5.422 | 46.11 |
| index_k_proj | FP8 | 3.606 | 5.06 | 3.565 | 5.11 |
| index_weights_proj | FP32 | 7.624 | 35.31 | 7.579 | 35.53 |
| indexer_qk | FP8 | 106.862 | 49.91 | 8.270 | 45.51 |
| indexer_fused | FP8 | N/A | N/A | 280.531 | 17.67 |
| mla_qk_pv | BF16 | 171.130 | 65.20 | 170.502 | 65.44 |
| v_expand | BF16 | 9.839 | 33.88 | 9.833 | 33.90 |
| o_proj | FP8 | 40.168 | 58.09 | 39.907 | 58.47 |
| mlp_gate | FP8 | 43.163 | 60.82 | 42.931 | 61.15 |
| mlp_up | FP8 | 42.405 | 61.91 | 42.140 | 62.30 |
| mlp_down | FP8 | 45.167 | 58.12 | 44.878 | 58.50 |
| lm_head | BF16 | 0.421 | 0.44 | 0.420 | 0.45 |

indexer_qk 记录 resident logits 调用，indexer_fused 包含融合 prefetch。同一 offload 阶段可包含两种调用，分别列出；MLA 行共同计入 QK/PV。

下表按最内层 scope 归因；attention_projection、dense_mlp 等父 scope 仅保留未归入子算子的剩余 kernel。这里只列 kernel 时间，cache_write 的 memcpy、CPU API 等另见完整数据，0 ms kernel 不表示没有搬运或同步开销。

| 非矩阵 scope（MFU N/A） | Resident kernel ms | Offload kernel ms |
| --- | ---: | ---: |
| apply_rope_pair | 2.003 | 2.050 |
| attention_output | 0.000 | 0.000 |
| attention_projection | 9.324 | 9.299 |
| cache_write | 0.462 | 21.785 |
| dense_mlp | 0.000 | 0.000 |
| embedding | 0.403 | 0.404 |
| exact_topk | 54.313 | 55.278 |
| final_norm_lm_head | 0.013 | 0.012 |
| forward_misc | 0.018 | 0.022 |
| hidden_transfer | 0.000 | 0.000 |
| index_cache_write | 0.000 | 0.000 |
| index_layer_norm | 0.727 | 0.721 |
| indexer_aux | 0.000 | 0.000 |
| indexer_prefetch_aux | N/A | 0.000 |
| input_residual_norm | 0.000 | 0.000 |
| layer_misc | 0.000 | 1.990 |
| offload_exact_recall | N/A | 93.127 |
| offload_finalize | N/A | 0.615 |
| offload_prepare | 0.000 | 8.109 |
| offload_source_reservation | 0.000 | 0.000 |
| post_attention_residual_norm | 0.000 | 0.000 |
| prepare_rotary_cache | 1.188 | 1.183 |
| quantize_index | 1.261 | 1.278 |
| residual_rms_norm | 5.095 | 5.092 |
| rms_norm | 1.384 | 1.374 |
| silu_mul | 5.532 | 5.523 |
| sparse_mla_aux | 0.000 | 0.000 |

### Extend 矩阵算子

| 算子 | 精度 | Resident kernel ms | MFU (%) | Offload kernel ms | MFU (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| q_a_proj | FP8 | 0.081 | 42.42 | 0.080 | 42.62 |
| q_b_proj | FP8 | 0.193 | 60.81 | 0.195 | 60.09 |
| q_absorb | BF16 | 0.168 | 31.00 | 0.168 | 31.00 |
| kv_a_proj | FP8 | 0.082 | 15.68 | 0.081 | 15.73 |
| index_q_proj | FP8 | 0.085 | 46.10 | 0.084 | 46.44 |
| index_k_proj | FP8 | 0.056 | 5.11 | 0.056 | 5.10 |
| index_weights_proj | FP32 | 0.117 | 35.85 | 0.119 | 35.47 |
| indexer_qk | FP8 | 3.214 | 52.27 | N/A | N/A |
| indexer_fused | FP8 | N/A | N/A | 9.369 | 17.93 |
| mla_qk_pv | BF16 | 2.657 | 66.66 | 2.690 | 65.83 |
| v_expand | BF16 | 0.153 | 34.01 | 0.153 | 34.05 |
| o_proj | FP8 | 0.622 | 58.58 | 0.623 | 58.54 |
| mlp_gate | FP8 | 0.669 | 61.28 | 0.670 | 61.22 |
| mlp_up | FP8 | 0.658 | 62.32 | 0.658 | 62.36 |
| mlp_down | FP8 | 0.701 | 58.47 | 0.701 | 58.51 |
| lm_head | BF16 | 0.422 | 0.44 | 0.421 | 0.45 |

indexer_qk 记录 resident logits 调用，indexer_fused 包含融合 prefetch。同一 offload 阶段可包含两种调用，分别列出；MLA 行共同计入 QK/PV。

下表按最内层 scope 归因；attention_projection、dense_mlp 等父 scope 仅保留未归入子算子的剩余 kernel。这里只列 kernel 时间，cache_write 的 memcpy、CPU API 等另见完整数据，0 ms kernel 不表示没有搬运或同步开销。

| 非矩阵 scope（MFU N/A） | Resident kernel ms | Offload kernel ms |
| --- | ---: | ---: |
| apply_rope_pair | 0.032 | 0.032 |
| attention_output | 0.000 | 0.000 |
| attention_projection | 0.144 | 0.144 |
| cache_write | 0.007 | 0.446 |
| dense_mlp | 0.000 | 0.000 |
| embedding | 0.006 | 0.006 |
| exact_topk | 1.133 | 1.149 |
| final_norm_lm_head | 0.012 | 0.012 |
| forward_misc | 0.004 | 0.004 |
| hidden_transfer | 0.000 | 0.000 |
| index_cache_write | 0.000 | 0.000 |
| index_layer_norm | 0.011 | 0.011 |
| indexer_aux | 0.000 | N/A |
| indexer_prefetch_aux | N/A | 0.000 |
| input_residual_norm | 0.000 | 0.000 |
| layer_misc | 0.000 | 0.067 |
| offload_exact_recall | N/A | 2.082 |
| offload_finalize | N/A | 0.013 |
| offload_prepare | 0.000 | 0.170 |
| offload_source_reservation | 0.000 | 0.000 |
| post_attention_residual_norm | 0.000 | 0.000 |
| prepare_rotary_cache | 0.019 | 0.018 |
| quantize_index | 0.019 | 0.020 |
| residual_rms_norm | 0.080 | 0.080 |
| rms_norm | 0.024 | 0.024 |
| silu_mul | 0.086 | 0.085 |
| sparse_mla_aux | 0.000 | 0.000 |

GPU kernel 时间、CPU API 时间、NVTX host 区间、无插桩 wall time 分别保存，不相加为端到端分解。Padding 工作另列 `executed_matmul_flops`，不计入 useful MFU；cuBLAS 内部 padding 未知，保留空值。

正确性与覆盖验收、硬件 identity、源码及输入 SHA256 见 [summary.json](summary.json)。逐层及 pooled 数据见 [operator_mfu.csv](operator_mfu.csv)、[operator_mfu_by_layer.csv](operator_mfu_by_layer.csv)、[nonmatrix.csv](nonmatrix.csv)。

本次未采集 NCU replay；报告不包含 NCU full/source 指标或验证结论。

生成命令：

```bash
python -m experiments.deepseek_v32_echo_prefill.src.publish_layers --run-id refactor_three_layers_profile_20261005_01  --publish
```
