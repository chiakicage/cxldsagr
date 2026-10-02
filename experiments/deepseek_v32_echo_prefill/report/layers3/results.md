## 前三层重新测量：`20261002_echo_layers3_mfu_01`

**2026-10-02 状态修正：旧实现记录，baseline 待修正复测。** 研究者指出 ECHO
实现的 MFU/cache 策略及 DeepSeek 当前实现的 MFU 有问题；下列相关 MFU、性能归因
及 ECHO 比较暂不可作为有效 baseline 结论。数值一致性、计数与归因审计不覆盖这些
问题，具体根因尚未定位。本轮没有修复实现或重跑 GPU 实验，原数字、run ID 及产物
保留到新 run 完成修正后的验收与发布，再统一替换和清理。

完整 checkpoint 的第 0–2 层依次传播 hidden/residual，包含 embedding、final norm 和最后 token LM head。Prefix=65,536，extend=1,024，chunk=1,024，offload pool=16,384 tokens/层。仅代表前三层。

| 阶段 | Resident 中位延迟 (ms) | Offload 中位延迟 (ms) | 每种模式重复次数 |
| --- | ---: | ---: | ---: |
| prefix | 2668.638 | 3540.019 | 3 |
| extend | 46.830 | 65.675 | 5 |

![前三层无插桩延迟](latency.svg)

原报告的计算口径为 MFU = useful matrix FLOPs /（同算子实际 GPU kernel duration 之和 × 对应精度 dense peak），其有效性待复核。FMA 计 2 FLOPs；FP8/BF16/FP32 分母分别为 1979.0/989.5/67.0 TFLOPS；TF32 关闭。前缀汇总全部 chunk 与三层，extend 汇总三层的一次完整 query batch（含实际 offload leaf 拆分）。LM head 每个阶段仅执行最后一个 token。非矩阵算子的 MFU 为 N/A。

每种模式、每个阶段各采集一次独立 annotated profile；算子 MFU 没有重复采样置信区间。FP8Linear 的分母包含量化和 GEMM，indexer 包含 logits API 的辅助 kernel；这些是原报告按完整算子计算的数值，不能视为已通过最新复核的 MFU，也不同于单 GEMM 或 Tensor pipe active 指标。

![四组逐算子 MFU](mfu.svg)

### Prefix 矩阵算子

| 算子 | 精度 | Resident kernel ms | MFU (%) | Offload kernel ms | MFU (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| q_a_proj | FP8 | 13.740 | 15.92 | 13.724 | 15.94 |
| q_b_proj | FP8 | 32.205 | 23.29 | 32.178 | 23.31 |
| q_absorb | BF16 | 9.894 | 33.69 | 9.887 | 33.72 |
| kv_a_proj | FP8 | 13.666 | 6.00 | 13.648 | 6.01 |
| index_q_proj | FP8 | 11.522 | 21.70 | 11.491 | 21.76 |
| index_k_proj | FP8 | 13.677 | 1.33 | 13.658 | 1.33 |
| index_weights_proj | FP32 | 7.574 | 35.55 | 7.586 | 35.49 |
| indexer | FP8 | 116.057 | 45.96 | 395.046 | 13.50 |
| mla_qk_pv | BF16 | 1387.094 | 8.04 | 1380.362 | 8.08 |
| v_expand | BF16 | 9.420 | 35.39 | 9.491 | 35.12 |
| o_proj | FP8 | 108.385 | 21.53 | 108.365 | 21.53 |
| mlp_gate | FP8 | 103.953 | 25.25 | 103.988 | 25.24 |
| mlp_up | FP8 | 103.496 | 25.36 | 103.519 | 25.36 |
| mlp_down | FP8 | 119.162 | 22.03 | 119.068 | 22.05 |
| lm_head | BF16 | 0.421 | 0.44 | 0.422 | 0.44 |

Indexer 行在 offload 中包含融合 prefetch；MLA 行共同计入 QK/PV。

下表按最内层 scope 归因；attention_projection、dense_mlp 等父 scope 仅保留未归入子算子的剩余 kernel。这里只列 kernel 时间，cache_write 的 memcpy、CPU API 等另见完整数据，0 ms kernel 不表示没有搬运或同步开销。

| 非矩阵 scope（MFU N/A） | Resident kernel ms | Offload kernel ms |
| --- | ---: | ---: |
| apply_rope | 64.693 | 64.548 |
| attention_output | 8.329 | 8.282 |
| attention_projection | 74.743 | 74.740 |
| cache_write | 0.463 | 0.000 |
| dense_mlp | 44.366 | 44.386 |
| embedding | 0.408 | 0.409 |
| exact_topk | 149.456 | 149.440 |
| final_norm_lm_head | 0.012 | 0.012 |
| forward_misc | 0.022 | 0.021 |
| hidden_transfer | 0.000 | 0.000 |
| index_layer_norm | 0.719 | 0.718 |
| indexer_aux | 0.000 | N/A |
| indexer_prefetch_aux | N/A | 0.000 |
| input_residual_norm | 12.300 | 12.280 |
| layer_misc | 0.000 | 1.984 |
| normalized_hadamard | 97.522 | 97.555 |
| offload_exact_recall | N/A | 84.405 |
| offload_prepare | 0.000 | 61.349 |
| post_attention_residual_norm | 16.927 | 16.943 |
| quantize_index | 31.296 | 31.299 |
| rms_norm | 43.372 | 43.283 |
| sparse_mla_aux | 0.000 | 0.000 |

### Extend 矩阵算子

| 算子 | 精度 | Resident kernel ms | MFU (%) | Offload kernel ms | MFU (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| q_a_proj | FP8 | 0.214 | 15.97 | 0.215 | 15.93 |
| q_b_proj | FP8 | 0.506 | 23.18 | 0.508 | 23.07 |
| q_absorb | BF16 | 0.155 | 33.58 | 0.154 | 33.73 |
| kv_a_proj | FP8 | 0.213 | 6.01 | 0.215 | 5.96 |
| index_q_proj | FP8 | 0.180 | 21.69 | 0.180 | 21.68 |
| index_k_proj | FP8 | 0.213 | 1.33 | 0.214 | 1.33 |
| index_weights_proj | FP32 | 0.118 | 35.56 | 0.118 | 35.74 |
| indexer | FP8 | 3.641 | 46.14 | 12.181 | 13.79 |
| mla_qk_pv | BF16 | 22.066 | 8.03 | 21.935 | 8.07 |
| v_expand | BF16 | 0.148 | 35.11 | 0.149 | 34.98 |
| o_proj | FP8 | 1.697 | 21.48 | 1.707 | 21.36 |
| mlp_gate | FP8 | 1.629 | 25.18 | 1.634 | 25.11 |
| mlp_up | FP8 | 1.625 | 25.25 | 1.627 | 25.21 |
| mlp_down | FP8 | 1.872 | 21.91 | 1.873 | 21.90 |
| lm_head | BF16 | 0.422 | 0.44 | 0.421 | 0.44 |

Indexer 行在 offload 中包含融合 prefetch；MLA 行共同计入 QK/PV。

下表按最内层 scope 归因；attention_projection、dense_mlp 等父 scope 仅保留未归入子算子的剩余 kernel。这里只列 kernel 时间，cache_write 的 memcpy、CPU API 等另见完整数据，0 ms kernel 不表示没有搬运或同步开销。

| 非矩阵 scope（MFU N/A） | Resident kernel ms | Offload kernel ms |
| --- | ---: | ---: |
| apply_rope | 1.013 | 1.011 |
| attention_output | 0.130 | 0.129 |
| attention_projection | 1.171 | 1.169 |
| cache_write | 0.008 | 0.000 |
| dense_mlp | 0.692 | 0.694 |
| embedding | 0.006 | 0.006 |
| exact_topk | 4.005 | 4.006 |
| final_norm_lm_head | 0.011 | 0.011 |
| forward_misc | 0.004 | 0.004 |
| hidden_transfer | 0.000 | 0.000 |
| index_layer_norm | 0.011 | 0.011 |
| indexer_aux | 0.000 | N/A |
| indexer_prefetch_aux | N/A | 0.000 |
| input_residual_norm | 0.194 | 0.193 |
| layer_misc | 0.000 | 0.064 |
| normalized_hadamard | 1.526 | 1.523 |
| offload_exact_recall | N/A | 1.671 |
| offload_prepare | 0.000 | 0.972 |
| post_attention_residual_norm | 0.264 | 0.266 |
| quantize_index | 0.487 | 0.490 |
| rms_norm | 0.684 | 0.685 |
| sparse_mla_aux | 0.000 | 0.000 |

GPU kernel 时间、CPU API 时间、NVTX host 区间、无插桩 wall time 分别保存，不相加为端到端分解。Padding 工作另列 `executed_matmul_flops`，不计入 useful MFU；cuBLAS 内部 padding 未知，保留空值。

原 run 的正确性与覆盖验收、硬件 identity、源码及输入 SHA256、完整 NCU full/source 的验证记录见 [summary.json](summary.json)。其中及 [postrun_audit.json](postrun_audit.json) 的验收字段保留当时的检查状态，不代表已解决最新 MFU/cache 策略问题。逐层及 pooled 数据见 [operator_mfu.csv](operator_mfu.csv)、[operator_mfu_by_layer.csv](operator_mfu_by_layer.csv)、[nonmatrix.csv](nonmatrix.csv)。

NCU 是同 run 对应真实层激活的独立 replay，不替代 NSYS 实际调用的 MFU 或正式 wall 延迟；其 cache 状态、排除项和报告 SHA256 单独保存。NCU run IDs：`20261002_echo_layers3_ncu_indexer_resident_01`, `20261002_echo_layers3_ncu_indexer_offload_01`, `20261002_echo_layers3_ncu_mla_01`。

生成命令：

```bash
python -m experiments.deepseek_v32_echo_prefill.src.publish_layers --run-id 20261002_echo_layers3_mfu_01 --ncu-run-id 20261002_echo_layers3_ncu_indexer_resident_01 --ncu-run-id 20261002_echo_layers3_ncu_indexer_offload_01 --ncu-run-id 20261002_echo_layers3_ncu_mla_01 --publish
```
