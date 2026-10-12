# DeepSeek V3.2 device-resident operator observation

来源 run ID：`observation_20261012_01`。全部 66 个工作负载完成匹配的独立数值验收、
clean 计时与独立 NCU profile。公共算子与 MoE 各覆盖 7 个 S，attention 覆盖 52 个 B/Q/H 点。
本报告测量具体 GEMM、非 GEMM 操作与另外列出的完整 API 聚合，不代表完整模型或 serving 延迟。

Decode 使用 Q=1、B=[1,32,64,128,256,512,1024]；Extend 使用 B=1、
Q=[32,64,128,256,512,1024]。H=[4096,16384,32768,65536]，S=B×Q。
输入与 cache 均驻留设备，不执行 offload；权重来自 checkpoint，activation/KV 使用声明的合成输入。

公共算子与 MoE 按 S 测量一次。`expanded.csv` 映射到对应 B/Q/H，`measurement_kind`
与 `source_case_id` 保留原测量来源；映射不新增测量。

## 指标与测量边界

Clean 时间使用完整调用两端的 CUDA events，通过独立 CUDA Graph replay 测量。
它包含调用内部 GPU 操作与间隙，不含 Python 提交、编译、数据生成和最终主机同步。
MFU 使用对应具体矩阵工作的 useful FLOPs 与这一 clean 时间；FP8/BF16/FP32 的
H200 SXM dense 峰值分别为 1979/989.5/67 TFLOP/s。非 GEMM 的矩阵 MFU 标为 N/A。
混合精度 API 聚合先求各精度的理想矩阵时间之和，再除以完整 API 时间。

HBM 指标来自另一次 NCU range/graph profile，`cache-control=none`。读取与写入
字节均使用 DRAM 硬件计数器；每调用字节由 profile range 总字节除以该 range 的调用数得到。
HBM bandwidth = 实测总字节 / profile range GPU 时间，utilization = bandwidth / NCU
对应 HBM 峰值。它的时间分母与 clean 时间不同，不能把 NCU 时间替换进 clean MFU，
也不能用 tensor 大小或 FLOPs 估算 HBM 实测流量。保留 runtime 的 cache 状态，
不代表冷缓存带宽；cache 命中的重复调用可能使 HBM 字节少于逻辑 tensor 字节。

NCU profile 是侵入式测量，clean 样本与其独立运行。profile range 的时间、inner、
metric 名称、cache control 和原始 NCU 文件路径/哈希逐行保留。矩阵 MFU 和 HBM utilization
都不是 Tensor Core 活跃率；完整 API 与其组件有重叠，不相加为模型时间或总流量。

矩阵 dense 峰值来源：[NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/)。
官方名义 HBM 带宽为 4.8 TB/s；本次利用率使用逐 profile 的 NCU sustained peak，
约 4.811–4.812 TB/s。两种峰值分别保留，不混用。

小操作受到明显的 profile 扰动：attention 的 NCU 每调用时间与 clean 时间之比，
中位数为 3.3154，p95 为 13.0107，最大为 39.5850。该比值只比较两种测量边界，
不能用作正常执行的减速比，也不能把 NCU 时间当作原始开销。
warm-cache 的低 HBM 流量可能来自 L2 命中，不能直接判断访存效率。

Clean 计时边界：

- `external CUDA events around complete API; CUDA Graph replay; mean of inner spans per sample`

NCU 计时边界与 replay 模式：

- `NCU entire CUDA Graph GPU duration, no external event nodes; intrusive profile, separate from clean API timing`
- replay：`kernel-graph`

| 表格 | 记录条数 | 含义 |
| --- | ---: | --- |
| [common.csv](common.csv) | 506 | 公共 GEMM、非 GEMM 和 API 聚合 |
| [moe.csv](moe.csv) | 938 | MoE GEMM、非 GEMM 和 API 聚合 |
| [attention.csv](attention.csv) | 1172 | 每个 B/Q/H 的 attention 操作 |
| [expanded.csv](expanded.csv) | 11916 | 公共映射与 attention 直接观测 |

其中具体操作为 2073 行，另列 API 聚合 543 行；每行均有独立 HBM profile。

## Projection

输入 projection、输出 `o_proj`、`q_absorb` 与 `v_expand` 均按实际 GEMM/BMM 叶操作单列。
吸收 MLA 不额外执行普通 `kv_b_proj` API：对应 checkpoint 权重用于 Q absorb 与 value expand。
完整 `attention_projection` 是输入准备聚合，包含输入投影、Q absorb、norm、casts、RoPE、
量化、scale 与 KV concat；它不含 Indexer score、MLA、value expand 或输出 `o_proj`。
聚合保留生产执行顺序与共享准备，不能用独立叶操作之和替代。

## 优先 Decode 配置的逐算子数值

以下列出 S=128 的公共与 MoE 具体操作及 API 聚合，以及 H=65536 的 Q1/B1、Q1/B128 attention。
其他 S、H 与 Extend 配置见各 CSV 和后续图表。表中 MFU 使用 clean 时间，HBM 使用独立 profile 时间。

### 公共操作：S=128

| 算子 | 类型 | 时间（µs） | MFU（%） | HBM（GB/s） | HBM 利用率（%） |
| --- | --- | ---: | ---: | ---: | ---: |
| `index_weights_input_cast` | non_gemm | 7.331 | N/A | 0.251 | 0.005 |
| `lm_head_output_cast` | non_gemm | 55.541 | N/A | 1783.860 | 37.072 |
| `mlp_down_gemm` | gemm | 51.546 | 33.157 | 2564.770 | 53.301 |
| `mlp_gate_gemm` | gemm | 41.814 | 40.873 | 3136.516 | 65.185 |
| `mlp_silu_mul` | non_gemm | 7.541 | N/A | 0.308 | 0.006 |
| `mlp_up_gemm` | gemm | 41.853 | 40.836 | 3176.023 | 66.003 |
| `embedding` | non_gemm | 5.640 | N/A | 0.001 | 0.000 |
| `index_k_proj_gemm` | gemm | 12.226 | 0.971 | 0.025 | 0.001 |
| `index_q_proj_gemm` | gemm | 9.240 | 17.616 | 0.014 | 0.000 |
| `index_weights_gemm` | gemm | 17.899 | 9.793 | 0.058 | 0.001 |
| `kv_a_proj_gemm` | gemm | 12.458 | 4.287 | 0.023 | 0.000 |
| `q_a_proj_gemm` | gemm | 13.432 | 10.603 | 0.032 | 0.001 |
| `q_b_proj_gemm` | gemm | 13.814 | 35.348 | 464.323 | 9.650 |
| `main_kv_concat` | non_gemm | 5.219 | N/A | 0.009 | 0.000 |
| `rotary_positions` | non_gemm | 4.664 | N/A | 0.005 | 0.000 |
| `rotary_trig_concat` | non_gemm | 5.035 | N/A | 0.001 | 0.000 |
| `lm_head_gemm` | gemm | 492.856 | 48.644 | 3926.739 | 81.602 |
| `final_residual_norm` | non_gemm | 6.333 | N/A | 0.000 | 0.000 |
| `index_layer_norm` | non_gemm | 6.418 | N/A | 0.024 | 0.000 |
| `index_layer_norm_input_cast` | non_gemm | 6.037 | N/A | 0.011 | 0.000 |
| `index_layer_norm_output_cast` | non_gemm | 4.891 | N/A | 0.032 | 0.001 |
| `input_rms_norm` | non_gemm | 6.035 | N/A | 0.062 | 0.001 |
| `kv_rms_norm` | non_gemm | 7.091 | N/A | 0.002 | 0.000 |
| `post_attention_residual_norm` | non_gemm | 6.475 | N/A | 0.010 | 0.000 |
| `query_rms_norm` | non_gemm | 5.557 | N/A | 1.044 | 0.022 |
| `o_proj_gemm` | gemm | 46.448 | 32.707 | 2517.754 | 52.325 |
| `q_absorb` | gemm | 12.419 | 17.475 | 203.717 | 4.234 |
| `v_expand` | gemm | 10.811 | 20.074 | 48.968 | 1.018 |
| `index_k_proj_activation_quantize` | non_gemm | 5.640 | N/A | 0.047 | 0.001 |
| `index_key_quantize` | non_gemm | 4.917 | N/A | 0.001 | 0.000 |
| `index_q_proj_activation_quantize` | non_gemm | 5.586 | N/A | 0.002 | 0.000 |
| `index_query_quantize` | non_gemm | 5.501 | N/A | 0.047 | 0.001 |
| `kv_a_proj_activation_quantize` | non_gemm | 5.720 | N/A | 0.044 | 0.001 |
| `linear_activation_quantize_k1536` | non_gemm | 5.578 | N/A | 0.022 | 0.000 |
| `linear_activation_quantize_k16384` | non_gemm | 6.096 | N/A | 0.184 | 0.004 |
| `linear_activation_quantize_k18432` | non_gemm | 6.202 | N/A | 0.077 | 0.002 |
| `linear_activation_quantize_k7168` | non_gemm | 5.754 | N/A | 0.052 | 0.001 |
| `mlp_down_activation_quantize` | non_gemm | 6.118 | N/A | 0.009 | 0.000 |
| `mlp_gate_activation_quantize` | non_gemm | 5.629 | N/A | 0.019 | 0.000 |
| `o_proj_activation_quantize` | non_gemm | 5.966 | N/A | 0.032 | 0.001 |
| `q_a_proj_activation_quantize` | non_gemm | 5.749 | N/A | 0.007 | 0.000 |
| `q_b_proj_activation_quantize` | non_gemm | 5.549 | N/A | 0.001 | 0.000 |
| `index_rope` | non_gemm | 6.160 | N/A | 0.044 | 0.001 |
| `main_rope` | non_gemm | 6.046 | N/A | 0.009 | 0.000 |
| `rotary_angles` | non_gemm | 4.886 | N/A | 0.001 | 0.000 |
| `rotary_cos` | non_gemm | 5.080 | N/A | 0.015 | 0.000 |
| `rotary_sin` | non_gemm | 5.104 | N/A | 0.030 | 0.001 |
| `index_weights_dim_scale` | non_gemm | 4.803 | N/A | 0.013 | 0.000 |
| `index_weights_head_scale` | non_gemm | 4.771 | N/A | 0.013 | 0.000 |
| `index_weights_query_scale` | non_gemm | 4.872 | N/A | 0.004 | 0.000 |
| `index_k_proj_scale_layout` | non_gemm | 5.525 | N/A | 0.011 | 0.000 |
| `index_q_proj_scale_layout` | non_gemm | 5.125 | N/A | 0.001 | 0.000 |
| `kv_a_proj_scale_layout` | non_gemm | 5.670 | N/A | 0.010 | 0.000 |
| `mlp_down_scale_layout` | non_gemm | 7.368 | N/A | 0.121 | 0.003 |
| `mlp_gate_scale_layout` | non_gemm | 5.589 | N/A | 0.004 | 0.000 |
| `mlp_up_scale_layout` | non_gemm | 5.595 | N/A | 0.002 | 0.000 |
| `o_proj_scale_layout` | non_gemm | 7.195 | N/A | 0.002 | 0.000 |
| `q_a_proj_scale_layout` | non_gemm | 5.677 | N/A | 0.003 | 0.000 |
| `q_b_proj_scale_layout` | non_gemm | 5.109 | N/A | 0.005 | 0.000 |
| `dense_mlp` | api_aggregate | 151.992 | 33.734 | 2759.730 | 57.351 |
| `mlp_down` | api_aggregate | 60.914 | 28.058 | 2272.109 | 47.219 |
| `mlp_gate` | api_aggregate | 47.654 | 35.864 | 2794.047 | 58.064 |
| `mlp_up` | api_aggregate | 48.283 | 35.397 | 2836.123 | 58.940 |
| `attention_projection` | api_aggregate | 154.560 | 8.095 | 256.579 | 5.332 |
| `index_k_proj` | api_aggregate | 16.760 | 0.708 | 0.068 | 0.001 |
| `index_q_proj` | api_aggregate | 13.230 | 12.303 | 0.162 | 0.003 |
| `index_weights_proj` | api_aggregate | 22.037 | 7.954 | 0.000 | 0.000 |
| `kv_a_proj` | api_aggregate | 16.936 | 3.154 | 0.038 | 0.001 |
| `q_a_proj` | api_aggregate | 17.827 | 7.989 | 0.054 | 0.001 |
| `q_b_proj` | api_aggregate | 19.117 | 25.544 | 343.981 | 7.149 |
| `lm_head` | api_aggregate | 539.051 | 44.476 | 3709.011 | 77.078 |
| `o_proj` | api_aggregate | 54.389 | 27.932 | 2250.935 | 46.779 |
| `rotary_cache` | api_aggregate | 9.424 | N/A | 0.001 | 0.000 |

### MoE 操作：S=128

| 算子 | 类型 | 时间（µs） | MFU（%） | HBM（GB/s） | HBM 利用率（%） |
| --- | --- | ---: | ---: | ---: | ---: |
| `moe_grouped_down_data_gather` | non_gemm | 26.411 | N/A | 2563.493 | 53.274 |
| `moe_grouped_down_data_mask` | non_gemm | 146.746 | N/A | 480.685 | 9.989 |
| `moe_grouped_down_output_gather` | non_gemm | 10.614 | N/A | 844.234 | 17.546 |
| `moe_grouped_down_output_mask` | non_gemm | 21.858 | N/A | 309.311 | 6.428 |
| `moe_grouped_down_packed_valid` | non_gemm | 4.642 | N/A | 0.012 | 0.000 |
| `moe_grouped_down_scale_gather` | non_gemm | 25.563 | N/A | 0.488 | 0.010 |
| `moe_grouped_down_scale_mask` | non_gemm | 7.573 | N/A | 0.020 | 0.000 |
| `moe_grouped_gate_data_gather` | non_gemm | 77.838 | N/A | 3070.541 | 63.811 |
| `moe_grouped_gate_data_mask` | non_gemm | 496.214 | N/A | 500.925 | 10.410 |
| `moe_grouped_gate_input_rows` | non_gemm | 5.878 | N/A | 0.000 | 0.000 |
| `moe_grouped_gate_output_gather` | non_gemm | 6.347 | N/A | 0.152 | 0.003 |
| `moe_grouped_gate_output_mask` | non_gemm | 10.826 | N/A | 0.073 | 0.002 |
| `moe_grouped_gate_packed_valid` | non_gemm | 4.862 | N/A | 0.000 | 0.000 |
| `moe_grouped_gate_scale_gather` | non_gemm | 25.595 | N/A | 0.118 | 0.002 |
| `moe_grouped_gate_scale_mask` | non_gemm | 10.752 | N/A | 0.146 | 0.003 |
| `moe_grouped_up_data_gather` | non_gemm | 77.781 | N/A | 3081.076 | 64.030 |
| `moe_grouped_up_data_mask` | non_gemm | 495.997 | N/A | 500.244 | 10.396 |
| `moe_grouped_up_input_rows` | non_gemm | 5.707 | N/A | 0.346 | 0.007 |
| `moe_grouped_up_output_gather` | non_gemm | 6.363 | N/A | 0.239 | 0.005 |
| `moe_grouped_up_output_mask` | non_gemm | 10.856 | N/A | 0.015 | 0.000 |
| `moe_grouped_up_packed_valid` | non_gemm | 4.704 | N/A | 0.000 | 0.000 |
| `moe_grouped_up_scale_gather` | non_gemm | 25.475 | N/A | 0.320 | 0.007 |
| `moe_grouped_up_scale_mask` | non_gemm | 10.549 | N/A | 0.071 | 0.001 |
| `moe_metadata_argsort` | non_gemm | 34.397 | N/A | 0.000 | 0.000 |
| `moe_metadata_counts_scatter_add` | non_gemm | 7.936 | N/A | 0.000 | 0.000 |
| `moe_metadata_counts_zero` | non_gemm | 4.696 | N/A | 0.001 | 0.000 |
| `moe_metadata_expert_clamp` | non_gemm | 4.971 | N/A | 0.000 | 0.000 |
| `moe_metadata_expert_count_gather` | non_gemm | 6.565 | N/A | 0.000 | 0.000 |
| `moe_metadata_expert_index_cast` | non_gemm | 6.094 | N/A | 0.000 | 0.000 |
| `moe_metadata_grouped_layout` | non_gemm | 6.602 | N/A | 0.000 | 0.000 |
| `moe_metadata_local_row_lt` | non_gemm | 4.843 | N/A | 0.003 | 0.000 |
| `moe_metadata_local_row_sub` | non_gemm | 4.962 | N/A | 0.000 | 0.000 |
| `moe_metadata_offset_zero` | non_gemm | 4.627 | N/A | 0.000 | 0.000 |
| `moe_metadata_output_slots` | non_gemm | 6.037 | N/A | 0.000 | 0.000 |
| `moe_metadata_packed_expert_lt` | non_gemm | 4.896 | N/A | 0.007 | 0.000 |
| `moe_metadata_packed_experts` | non_gemm | 6.485 | N/A | 0.000 | 0.000 |
| `moe_metadata_packed_positions` | non_gemm | 4.880 | N/A | 0.000 | 0.000 |
| `moe_metadata_packed_routes` | non_gemm | 6.837 | N/A | 0.002 | 0.000 |
| `moe_metadata_packed_valid_and` | non_gemm | 4.795 | N/A | 0.004 | 0.000 |
| `moe_metadata_rank_arange` | non_gemm | 4.472 | N/A | 0.000 | 0.000 |
| `moe_metadata_rank_scatter` | non_gemm | 8.043 | N/A | 0.000 | 0.000 |
| `moe_metadata_route_ge_zero` | non_gemm | 4.872 | N/A | 0.000 | 0.000 |
| `moe_metadata_route_lt_experts` | non_gemm | 4.910 | N/A | 0.000 | 0.000 |
| `moe_metadata_route_ones` | non_gemm | 4.597 | N/A | 0.000 | 0.000 |
| `moe_metadata_route_valid_and` | non_gemm | 4.720 | N/A | 0.000 | 0.000 |
| `moe_metadata_row_cumsum` | non_gemm | 6.362 | N/A | 0.000 | 0.000 |
| `moe_metadata_row_offset_cat` | non_gemm | 4.837 | N/A | 0.000 | 0.000 |
| `moe_metadata_row_offset_gather` | non_gemm | 6.600 | N/A | 0.003 | 0.000 |
| `moe_metadata_safe_keys` | non_gemm | 4.680 | N/A | 0.000 | 0.000 |
| `moe_metadata_slot_base_mul` | non_gemm | 4.632 | N/A | 0.000 | 0.000 |
| `moe_metadata_slot_rank_add` | non_gemm | 6.227 | N/A | 0.000 | 0.000 |
| `moe_metadata_slot_row_gather` | non_gemm | 6.360 | N/A | 0.000 | 0.000 |
| `moe_metadata_slot_row_sub` | non_gemm | 6.114 | N/A | 0.009 | 0.000 |
| `moe_metadata_slot_tile_gather` | non_gemm | 6.374 | N/A | 0.000 | 0.000 |
| `moe_metadata_sort_keys` | non_gemm | 6.142 | N/A | 0.000 | 0.000 |
| `moe_metadata_sorted_index_cast` | non_gemm | 5.902 | N/A | 0.001 | 0.000 |
| `moe_metadata_sorted_position_add` | non_gemm | 4.798 | N/A | 0.004 | 0.000 |
| `moe_metadata_sorted_position_clamp` | non_gemm | 4.762 | N/A | 0.001 | 0.000 |
| `moe_metadata_tile_count_add` | non_gemm | 4.810 | N/A | 0.002 | 0.000 |
| `moe_metadata_tile_count_div` | non_gemm | 5.224 | N/A | 0.002 | 0.000 |
| `moe_metadata_tile_count_sub_one` | non_gemm | 4.893 | N/A | 0.000 | 0.000 |
| `moe_metadata_tile_cumsum` | non_gemm | 5.899 | N/A | 0.003 | 0.000 |
| `moe_metadata_tile_ends` | non_gemm | 4.550 | N/A | 0.000 | 0.000 |
| `moe_metadata_tile_offset_cat` | non_gemm | 4.813 | N/A | 0.000 | 0.000 |
| `moe_metadata_tile_offset_gather` | non_gemm | 6.638 | N/A | 0.001 | 0.000 |
| `moe_metadata_tile_start_mul` | non_gemm | 4.944 | N/A | 0.002 | 0.000 |
| `moe_routed_silu_pack` | non_gemm | 7.950 | N/A | 1.514 | 0.031 |
| `moe_shared_silu_pack` | non_gemm | 5.315 | N/A | 0.003 | 0.000 |
| `moe_routed_silu_kernel` | non_gemm | 6.901 | N/A | 0.135 | 0.003 |
| `moe_shared_silu_kernel` | non_gemm | 5.192 | N/A | 0.002 | 0.000 |
| `moe_output_cast` | non_gemm | 5.520 | N/A | 0.005 | 0.000 |
| `moe_routed_output_cast` | non_gemm | 24.814 | N/A | 1487.861 | 30.921 |
| `moe_routing_hidden_cast` | non_gemm | 6.976 | N/A | 0.041 | 0.001 |
| `moe_shared_output_cast` | non_gemm | 7.165 | N/A | 0.049 | 0.001 |
| `moe_grouped_down_gemm` | gemm | 1024.549 | 1.483 | 3743.517 | 77.795 |
| `moe_grouped_gate_gemm` | gemm | 1079.408 | 1.407 | 3791.500 | 78.792 |
| `moe_grouped_up_gemm` | gemm | 1074.576 | 1.414 | 3755.331 | 78.040 |
| `moe_shared_down_gemm` | gemm | 10.998 | 17.266 | 0.032 | 0.001 |
| `moe_shared_gate_gemm` | gemm | 13.563 | 14.001 | 0.000 | 0.000 |
| `moe_shared_up_gemm` | gemm | 13.490 | 14.077 | 0.001 | 0.000 |
| `moe_route_reduce` | non_gemm | 11.317 | N/A | 237.816 | 4.942 |
| `moe_shared_add` | non_gemm | 5.920 | N/A | 0.006 | 0.000 |
| `moe_weighted_multiply` | non_gemm | 27.046 | N/A | 2131.253 | 44.294 |
| `moe_routing_correction` | non_gemm | 5.360 | N/A | 0.000 | 0.000 |
| `moe_routing_denominator_clamp` | non_gemm | 4.781 | N/A | 0.000 | 0.000 |
| `moe_routing_expert_topk` | non_gemm | 16.213 | N/A | 0.000 | 0.000 |
| `moe_routing_gate_gemm` | gemm | 27.686 | 25.324 | 0.000 | 0.000 |
| `moe_routing_group_exclude_mask` | non_gemm | 4.654 | N/A | 0.001 | 0.000 |
| `moe_routing_group_mask` | non_gemm | 6.701 | N/A | 0.000 | 0.000 |
| `moe_routing_group_mask_scatter` | non_gemm | 5.725 | N/A | 0.000 | 0.000 |
| `moe_routing_group_mask_zero` | non_gemm | 4.661 | N/A | 0.000 | 0.000 |
| `moe_routing_group_score_reduce` | non_gemm | 5.525 | N/A | 0.003 | 0.000 |
| `moe_routing_group_top2` | non_gemm | 16.474 | N/A | 0.000 | 0.000 |
| `moe_routing_group_topk` | non_gemm | 14.853 | N/A | 0.000 | 0.000 |
| `moe_routing_sigmoid` | non_gemm | 5.160 | N/A | 0.037 | 0.001 |
| `moe_routing_weight_gather` | non_gemm | 8.062 | N/A | 0.001 | 0.000 |
| `moe_routing_weight_normalize` | non_gemm | 5.083 | N/A | 0.000 | 0.000 |
| `moe_routing_weight_scale` | non_gemm | 4.680 | N/A | 0.001 | 0.000 |
| `moe_routing_weight_sum` | non_gemm | 5.579 | N/A | 0.000 | 0.000 |
| `moe_grouped_down_activation_quantize` | non_gemm | 5.824 | N/A | 0.094 | 0.002 |
| `moe_grouped_gate_activation_quantize` | non_gemm | 5.618 | N/A | 0.000 | 0.000 |
| `moe_grouped_up_activation_quantize` | non_gemm | 5.469 | N/A | 0.015 | 0.000 |
| `moe_shared_down_activation_quantize` | non_gemm | 5.622 | N/A | 0.004 | 0.000 |
| `moe_shared_gate_activation_quantize` | non_gemm | 5.693 | N/A | 0.000 | 0.000 |
| `moe_shared_up_activation_quantize` | non_gemm | 5.701 | N/A | 0.003 | 0.000 |
| `moe_grouped_down_scale_layout` | non_gemm | 6.080 | N/A | 0.000 | 0.000 |
| `moe_grouped_gate_scale_layout` | non_gemm | 7.723 | N/A | 0.739 | 0.015 |
| `moe_grouped_up_scale_layout` | non_gemm | 7.682 | N/A | 0.289 | 0.006 |
| `moe_shared_down_scale_layout` | non_gemm | 5.114 | N/A | 0.000 | 0.000 |
| `moe_shared_gate_scale_layout` | non_gemm | 5.642 | N/A | 0.001 | 0.000 |
| `moe_shared_up_scale_layout` | non_gemm | 5.624 | N/A | 0.000 | 0.000 |
| `moe_metadata_expert_counts` | api_aggregate | 10.232 | N/A | 0.000 | 0.000 |
| `moe_metadata_inverse_rank` | api_aggregate | 9.427 | N/A | 0.000 | 0.000 |
| `moe_metadata_local_rows` | api_aggregate | 9.646 | N/A | 0.000 | 0.000 |
| `moe_metadata_output_slot_arithmetic` | api_aggregate | 15.522 | N/A | 0.159 | 0.003 |
| `moe_metadata_packed_valid` | api_aggregate | 11.040 | N/A | 0.001 | 0.000 |
| `moe_metadata_route_valid` | api_aggregate | 7.792 | N/A | 0.297 | 0.006 |
| `moe_metadata_row_offsets` | api_aggregate | 5.848 | N/A | 0.000 | 0.000 |
| `moe_metadata_safe_experts` | api_aggregate | 7.674 | N/A | 0.268 | 0.006 |
| `moe_metadata_sorted_positions` | api_aggregate | 12.554 | N/A | 0.164 | 0.003 |
| `moe_metadata_tile_counts` | api_aggregate | 7.333 | N/A | 0.000 | 0.000 |
| `moe_metadata_tile_offsets` | api_aggregate | 5.674 | N/A | 0.000 | 0.000 |
| `moe_prepare_routing` | api_aggregate | 119.256 | N/A | 0.018 | 0.000 |
| `moe_routed_silu` | api_aggregate | 11.461 | N/A | 27.621 | 0.574 |
| `moe_shared_silu` | api_aggregate | 7.067 | N/A | 0.004 | 0.000 |
| `moe` | api_aggregate | 4944.382 | 1.179 | 2650.375 | 55.078 |
| `moe_grouped_down` | api_aggregate | 1251.904 | 1.214 | 3207.919 | 66.664 |
| `moe_grouped_gate` | api_aggregate | 1685.832 | 0.901 | 2673.286 | 55.554 |
| `moe_grouped_up` | api_aggregate | 1682.784 | 0.903 | 2682.528 | 55.746 |
| `moe_shared_down` | api_aggregate | 15.267 | 12.438 | 0.019 | 0.000 |
| `moe_shared_expert` | api_aggregate | 50.664 | 11.245 | 316.298 | 6.573 |
| `moe_shared_gate` | api_aggregate | 18.035 | 10.529 | 0.000 | 0.000 |
| `moe_shared_up` | api_aggregate | 17.958 | 10.574 | 0.014 | 0.000 |
| `moe_routing` | api_aggregate | 89.589 | 7.826 | 0.300 | 0.006 |

### Attention：B=1，Q=1，H=65536

| 算子 | 类型 | 时间（µs） | MFU（%） | HBM（GB/s） | HBM 利用率（%） |
| --- | --- | ---: | ---: | ---: | ---: |
| `indexer_block_table` | non_gemm | 4.413 | N/A | 0.428 | 0.009 |
| `indexer_metadata` | non_gemm | 5.750 | N/A | 0.000 | 0.000 |
| `page64_pack` | non_gemm | 7.104 | N/A | 1.079 | 0.022 |
| `indexer` | gemm | 7.723 | 7.025 | 0.014 | 0.000 |
| `indexer_causal_mask` | non_gemm | 4.506 | N/A | 0.001 | 0.000 |
| `mla_lse_combine` | non_gemm | 6.632 | N/A | 0.001 | 0.000 |
| `mla_query_repeat` | non_gemm | 7.162 | N/A | 0.122 | 0.003 |
| `sparse_mla_core` | gemm | 11.406 | 5.054 | 2.655 | 0.055 |
| `exact_topk_radix` | non_gemm | 34.613 | N/A | 0.014 | 0.000 |
| `topk_cub_sort_mask` | non_gemm | 22.272 | N/A | 0.000 | 0.000 |
| `topk_ids_clone` | non_gemm | 4.549 | N/A | 0.005 | 0.000 |
| `topk_values_clone` | non_gemm | 4.427 | N/A | 0.000 | 0.000 |
| `indexer_prepare` | api_aggregate | 8.192 | N/A | 17.181 | 0.357 |
| `sparse_mla` | api_aggregate | 18.621 | 3.096 | 0.027 | 0.001 |
| `exact_topk` | api_aggregate | 53.550 | N/A | 0.002 | 0.000 |
| `topk_output_clone` | api_aggregate | 5.781 | N/A | 0.000 | 0.000 |
| `topk_postprocess` | api_aggregate | 24.656 | N/A | 0.001 | 0.000 |

### Attention：B=128，Q=1，H=65536

| 算子 | 类型 | 时间（µs） | MFU（%） | HBM（GB/s） | HBM 利用率（%） |
| --- | --- | ---: | ---: | ---: | ---: |
| `indexer_block_table` | non_gemm | 5.768 | N/A | 0.008 | 0.000 |
| `indexer_metadata` | non_gemm | 7.162 | N/A | 0.005 | 0.000 |
| `page64_pack` | non_gemm | 544.096 | N/A | 3139.159 | 65.239 |
| `indexer` | gemm | 278.059 | 24.977 | 4076.460 | 84.714 |
| `indexer_causal_mask` | non_gemm | 5.104 | N/A | 0.085 | 0.002 |
| `mla_id_nonnegative` | non_gemm | 5.062 | N/A | 0.003 | 0.000 |
| `mla_id_offset_add` | non_gemm | 5.979 | N/A | 0.050 | 0.001 |
| `mla_id_to_int32` | non_gemm | 5.698 | N/A | 0.000 | 0.000 |
| `mla_id_to_int64` | non_gemm | 6.246 | N/A | 0.052 | 0.001 |
| `mla_id_upper_bound` | non_gemm | 5.008 | N/A | 0.016 | 0.000 |
| `mla_id_valid_and` | non_gemm | 5.051 | N/A | 0.001 | 0.000 |
| `mla_id_where` | non_gemm | 7.051 | N/A | 0.036 | 0.001 |
| `mla_sequence_arange` | non_gemm | 4.453 | N/A | 0.019 | 0.000 |
| `mla_sequence_offset_mul` | non_gemm | 4.757 | N/A | 0.002 | 0.000 |
| `sparse_mla_core` | gemm | 148.480 | 49.696 | 2542.561 | 52.838 |
| `exact_topk_radix` | non_gemm | 67.848 | N/A | 7.057 | 0.147 |
| `topk_ids_clone` | non_gemm | 4.926 | N/A | 0.003 | 0.000 |
| `topk_nonfinite_mask` | non_gemm | 5.088 | N/A | 0.000 | 0.000 |
| `topk_values_clone` | non_gemm | 4.955 | N/A | 0.020 | 0.000 |
| `indexer_prepare` | api_aggregate | 545.632 | N/A | 3053.413 | 63.458 |
| `mla_global_indices` | api_aggregate | 20.522 | N/A | 0.008 | 0.000 |
| `sparse_mla` | api_aggregate | 165.822 | 44.499 | 1731.974 | 35.993 |
| `exact_topk` | api_aggregate | 72.336 | N/A | 6.524 | 0.136 |
| `topk_output_clone` | api_aggregate | 6.582 | N/A | 0.018 | 0.000 |
| `topk_postprocess` | api_aggregate | 8.243 | N/A | 0.026 | 0.001 |

## 具体算子清单

下表列出直接测量的算子名称、矩阵精度与所属 API；GEMM 包括融合的矩阵核心，
`non_gemm` 的矩阵 MFU 为 N/A。支持条件和实际形状仍以每行 metadata 为准。

| 分组 | 实际算子名称 | 类型 | 矩阵精度 | 所属 API | 直接测量覆盖 |
| --- | --- | --- | --- | --- | --- |
| attention/topk | `exact_topk_radix` | non_gemm | N/A | `exact_topk` | 52 B/Q/H points |
| attention/indexer | `indexer` | gemm | fp8 | `indexer` | 52 B/Q/H points |
| attention/cache_preparation | `indexer_block_table` | non_gemm | N/A | `indexer_prepare` | 28 B/Q/H points |
| attention/indexer | `indexer_causal_mask` | non_gemm | N/A | `indexer_causal_mask` | 52 B/Q/H points |
| attention/cache_preparation | `indexer_metadata` | non_gemm | N/A | `indexer_metadata` | 28 B/Q/H points |
| attention/sparse_attention | `mla_id_nonnegative` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/sparse_attention | `mla_id_offset_add` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/sparse_attention | `mla_id_to_int32` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/sparse_attention | `mla_id_to_int64` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/sparse_attention | `mla_id_upper_bound` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/sparse_attention | `mla_id_valid_and` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/sparse_attention | `mla_id_where` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/sparse_attention | `mla_lse_combine` | non_gemm | N/A | `sparse_mla` | 4 B/Q/H points |
| attention/sparse_attention | `mla_query_repeat` | non_gemm | N/A | `sparse_mla` | 4 B/Q/H points |
| attention/sparse_attention | `mla_sequence_arange` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/sparse_attention | `mla_sequence_offset_mul` | non_gemm | N/A | `mla_global_indices` | 48 B/Q/H points |
| attention/cache_preparation | `page64_pack` | non_gemm | N/A | `indexer_prepare` | 28 B/Q/H points |
| attention/sparse_attention | `sparse_mla_core` | gemm | bf16 | `sparse_mla` | 52 B/Q/H points |
| attention/topk | `topk_cub_sort_mask` | non_gemm | N/A | `exact_topk` | 2 B/Q/H points |
| attention/topk | `topk_ids_clone` | non_gemm | N/A | `topk_output_clone` | 52 B/Q/H points |
| attention/topk | `topk_nonfinite_mask` | non_gemm | N/A | `exact_topk` | 50 B/Q/H points |
| attention/topk | `topk_values_clone` | non_gemm | N/A | `topk_output_clone` | 52 B/Q/H points |
| common/embedding | `embedding` | non_gemm | N/A | `embedding` | S=1,32,64,128,256,512,1024 |
| common/norm | `final_residual_norm` | non_gemm | N/A | `final_residual_norm` | S=1,32,64,128,256,512,1024 |
| common/quantization | `index_k_proj_activation_quantize` | non_gemm | N/A | `index_k_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `index_k_proj_gemm` | gemm | fp8 | `index_k_proj` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `index_k_proj_scale_layout` | non_gemm | N/A | `index_k_proj` | S=32,64,128,256,512,1024; S > 1 |
| common/quantization | `index_key_quantize` | non_gemm | N/A | `index_key_quantize` | S=1,32,64,128,256,512,1024 |
| common/norm | `index_layer_norm` | non_gemm | N/A | `index_layer_norm` | S=1,32,64,128,256,512,1024 |
| common/norm | `index_layer_norm_input_cast` | non_gemm | N/A | `index_layer_norm` | S=1,32,64,128,256,512,1024 |
| common/norm | `index_layer_norm_output_cast` | non_gemm | N/A | `index_layer_norm` | S=1,32,64,128,256,512,1024 |
| common/quantization | `index_q_proj_activation_quantize` | non_gemm | N/A | `index_q_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `index_q_proj_gemm` | gemm | fp8 | `index_q_proj` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `index_q_proj_scale_layout` | non_gemm | N/A | `index_q_proj` | S=32,64,128,256,512,1024; S > 1 |
| common/quantization | `index_query_quantize` | non_gemm | N/A | `index_query_quantize` | S=1,32,64,128,256,512,1024 |
| common/rope | `index_rope` | non_gemm | N/A | `index_rope` | S=1,32,64,128,256,512,1024 |
| common/scale | `index_weights_dim_scale` | non_gemm | N/A | `index_weights_dim_scale` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `index_weights_gemm` | gemm | fp32 | `index_weights_proj` | S=1,32,64,128,256,512,1024 |
| common/scale | `index_weights_head_scale` | non_gemm | N/A | `index_weights_head_scale` | S=1,32,64,128,256,512,1024 |
| common/cast | `index_weights_input_cast` | non_gemm | N/A | `index_weights_proj` | S=1,32,64,128,256,512,1024 |
| common/scale | `index_weights_query_scale` | non_gemm | N/A | `index_weights_query_scale` | S=1,32,64,128,256,512,1024 |
| common/norm | `input_rms_norm` | non_gemm | N/A | `input_rms_norm` | S=1,32,64,128,256,512,1024 |
| common/quantization | `kv_a_proj_activation_quantize` | non_gemm | N/A | `kv_a_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `kv_a_proj_gemm` | gemm | fp8 | `kv_a_proj` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `kv_a_proj_scale_layout` | non_gemm | N/A | `kv_a_proj` | S=32,64,128,256,512,1024; S > 1 |
| common/norm | `kv_rms_norm` | non_gemm | N/A | `kv_rms_norm` | S=1,32,64,128,256,512,1024 |
| common/quantization | `linear_activation_quantize_k1536` | non_gemm | N/A | `linear_activation_quantize_k1536` | S=1,32,64,128,256,512,1024; all requested S |
| common/quantization | `linear_activation_quantize_k1536_consumer_scales` | non_gemm | N/A | `linear_activation_quantize_k1536_consumer_scales` | S=1; S == 1 |
| common/quantization | `linear_activation_quantize_k16384` | non_gemm | N/A | `linear_activation_quantize_k16384` | S=1,32,64,128,256,512,1024; all requested S |
| common/quantization | `linear_activation_quantize_k16384_consumer_scales` | non_gemm | N/A | `linear_activation_quantize_k16384_consumer_scales` | S=1; S == 1 |
| common/quantization | `linear_activation_quantize_k18432` | non_gemm | N/A | `linear_activation_quantize_k18432` | S=1,32,64,128,256,512,1024; all requested S |
| common/quantization | `linear_activation_quantize_k18432_consumer_scales` | non_gemm | N/A | `linear_activation_quantize_k18432_consumer_scales` | S=1; S == 1 |
| common/quantization | `linear_activation_quantize_k7168` | non_gemm | N/A | `linear_activation_quantize_k7168` | S=1,32,64,128,256,512,1024; all requested S |
| common/quantization | `linear_activation_quantize_k7168_consumer_scales` | non_gemm | N/A | `linear_activation_quantize_k7168_consumer_scales` | S=1; S == 1 |
| common/lm_head | `lm_head_gemm` | gemm | bf16 | `lm_head` | S=1,32,64,128,256,512,1024 |
| common/cast | `lm_head_output_cast` | non_gemm | N/A | `lm_head` | S=1,32,64,128,256,512,1024 |
| common/layout | `main_kv_concat` | non_gemm | N/A | `main_kv_concat` | S=1,32,64,128,256,512,1024 |
| common/rope | `main_rope` | non_gemm | N/A | `main_rope` | S=1,32,64,128,256,512,1024 |
| common/quantization | `mlp_down_activation_quantize` | non_gemm | N/A | `mlp_down` | S=1,32,64,128,256,512,1024 |
| common/dense_mlp | `mlp_down_gemm` | gemm | fp8 | `mlp_down` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `mlp_down_scale_layout` | non_gemm | N/A | `mlp_down` | S=32,64,128,256,512,1024; S > 1 |
| common/quantization | `mlp_gate_activation_quantize` | non_gemm | N/A | `mlp_gate` | S=1,32,64,128,256,512,1024 |
| common/dense_mlp | `mlp_gate_gemm` | gemm | fp8 | `mlp_gate` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `mlp_gate_scale_layout` | non_gemm | N/A | `mlp_gate` | S=32,64,128,256,512,1024; S > 1 |
| common/dense_mlp | `mlp_silu_mul` | non_gemm | N/A | `mlp_silu_mul` | S=1,32,64,128,256,512,1024 |
| common/dense_mlp | `mlp_up_gemm` | gemm | fp8 | `mlp_up` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `mlp_up_scale_layout` | non_gemm | N/A | `mlp_up` | S=32,64,128,256,512,1024; S > 1 |
| common/quantization | `o_proj_activation_quantize` | non_gemm | N/A | `o_proj` | S=1,32,64,128,256,512,1024 |
| common/output_projection | `o_proj_gemm` | gemm | fp8 | `o_proj` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `o_proj_scale_layout` | non_gemm | N/A | `o_proj` | S=32,64,128,256,512,1024; S > 1 |
| common/norm | `post_attention_residual_norm` | non_gemm | N/A | `post_attention_residual_norm` | S=1,32,64,128,256,512,1024 |
| common/quantization | `q_a_proj_activation_quantize` | non_gemm | N/A | `q_a_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `q_a_proj_gemm` | gemm | fp8 | `q_a_proj` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `q_a_proj_scale_layout` | non_gemm | N/A | `q_a_proj` | S=32,64,128,256,512,1024; S > 1 |
| common/projection_transform | `q_absorb` | gemm | bf16 | `q_absorb` | S=1,32,64,128,256,512,1024 |
| common/quantization | `q_b_proj_activation_quantize` | non_gemm | N/A | `q_b_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `q_b_proj_gemm` | gemm | fp8 | `q_b_proj` | S=1,32,64,128,256,512,1024 |
| common/scale_layout | `q_b_proj_scale_layout` | non_gemm | N/A | `q_b_proj` | S=32,64,128,256,512,1024; S > 1 |
| common/norm | `query_rms_norm` | non_gemm | N/A | `query_rms_norm` | S=1,32,64,128,256,512,1024 |
| common/rope | `rotary_angles` | non_gemm | N/A | `rotary_angles` | S=1,32,64,128,256,512,1024 |
| common/rope | `rotary_cos` | non_gemm | N/A | `rotary_cache` | S=1,32,64,128,256,512,1024 |
| common/layout | `rotary_positions` | non_gemm | N/A | `rotary_cache` | S=1,32,64,128,256,512,1024 |
| common/rope | `rotary_sin` | non_gemm | N/A | `rotary_cache` | S=1,32,64,128,256,512,1024 |
| common/layout | `rotary_trig_concat` | non_gemm | N/A | `rotary_cache` | S=1,32,64,128,256,512,1024 |
| common/projection_transform | `v_expand` | gemm | bf16 | `v_expand` | S=1,32,64,128,256,512,1024 |
| moe/quantization | `moe_grouped_down_activation_quantize` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_down_data_gather` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_down_data_mask` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_grouped_down_gemm` | gemm | fp8 | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_down_output_gather` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_down_output_mask` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_down_packed_valid` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_down_scale_gather` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/scale_layout | `moe_grouped_down_scale_layout` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_down_scale_mask` | non_gemm | N/A | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/quantization | `moe_grouped_gate_activation_quantize` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_gate_data_gather` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_gate_data_mask` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_grouped_gate_gemm` | gemm | fp8 | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_gate_input_rows` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_gate_output_gather` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_gate_output_mask` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_gate_packed_valid` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_gate_scale_gather` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/scale_layout | `moe_grouped_gate_scale_layout` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_gate_scale_mask` | non_gemm | N/A | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/quantization | `moe_grouped_up_activation_quantize` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_up_data_gather` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_up_data_mask` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_grouped_up_gemm` | gemm | fp8 | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_up_input_rows` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_up_output_gather` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_up_output_mask` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_up_packed_valid` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_up_scale_gather` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/scale_layout | `moe_grouped_up_scale_layout` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_grouped_up_scale_mask` | non_gemm | N/A | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_argsort` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_counts_scatter_add` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_counts_zero` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_expert_clamp` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_expert_count_gather` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_expert_index_cast` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_grouped_layout` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_local_row_lt` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_local_row_sub` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_offset_zero` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_output_slots` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_packed_expert_lt` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_packed_experts` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_packed_positions` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_packed_routes` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_packed_valid_and` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_rank_arange` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_rank_scatter` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_route_ge_zero` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_route_lt_experts` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_route_ones` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_route_valid_and` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_row_cumsum` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_row_offset_cat` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_row_offset_gather` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_safe_keys` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_slot_base_mul` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_slot_rank_add` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_slot_row_gather` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_slot_row_sub` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_slot_tile_gather` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_sort_keys` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_sorted_index_cast` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_sorted_position_add` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_sorted_position_clamp` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_count_add` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_count_div` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_count_sub_one` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_cumsum` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_ends` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_offset_cat` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_offset_gather` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_start_mul` | non_gemm | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_cast | `moe_output_cast` | non_gemm | N/A | `moe` | S=1,32,64,128,256,512,1024 |
| moe/moe_reduce | `moe_route_reduce` | non_gemm | N/A | `moe` | S=1,32,64,128,256,512,1024 |
| moe/moe_cast | `moe_routed_output_cast` | non_gemm | N/A | `moe` | S=1,32,64,128,256,512,1024 |
| moe/moe_activation | `moe_routed_silu_kernel` | non_gemm | N/A | `moe_routed_silu` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_routed_silu_pack` | non_gemm | N/A | `moe_routed_silu` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_correction` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_denominator_clamp` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_expert_topk` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_gate_gemm` | gemm | fp32 | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_group_exclude_mask` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_group_mask` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_group_mask_scatter` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_group_mask_zero` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_group_score_reduce` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_group_top2` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_group_topk` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_cast | `moe_routing_hidden_cast` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_sigmoid` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_weight_gather` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_weight_normalize` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_weight_scale` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing_weight_sum` | non_gemm | N/A | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_reduce | `moe_shared_add` | non_gemm | N/A | `moe` | S=1,32,64,128,256,512,1024 |
| moe/quantization | `moe_shared_down_activation_quantize` | non_gemm | N/A | `moe_shared_down` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_shared_down_gemm` | gemm | fp8 | `moe_shared_down` | S=1,32,64,128,256,512,1024 |
| moe/scale_layout | `moe_shared_down_scale_layout` | non_gemm | N/A | `moe_shared_down` | S=1,32,64,128,256,512,1024 |
| moe/quantization | `moe_shared_gate_activation_quantize` | non_gemm | N/A | `moe_shared_gate` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_shared_gate_gemm` | gemm | fp8 | `moe_shared_gate` | S=1,32,64,128,256,512,1024 |
| moe/scale_layout | `moe_shared_gate_scale_layout` | non_gemm | N/A | `moe_shared_gate` | S=1,32,64,128,256,512,1024 |
| moe/moe_cast | `moe_shared_output_cast` | non_gemm | N/A | `moe` | S=1,32,64,128,256,512,1024 |
| moe/moe_activation | `moe_shared_silu_kernel` | non_gemm | N/A | `moe_shared_silu` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_shared_silu_pack` | non_gemm | N/A | `moe_shared_silu` | S=1,32,64,128,256,512,1024 |
| moe/quantization | `moe_shared_up_activation_quantize` | non_gemm | N/A | `moe_shared_up` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_shared_up_gemm` | gemm | fp8 | `moe_shared_up` | S=1,32,64,128,256,512,1024 |
| moe/scale_layout | `moe_shared_up_scale_layout` | non_gemm | N/A | `moe_shared_up` | S=1,32,64,128,256,512,1024 |
| moe/moe_reduce | `moe_weighted_multiply` | non_gemm | N/A | `moe` | S=1,32,64,128,256,512,1024 |

## 单独测量的 API 聚合

这些聚合保留生产调用的实际准备和执行顺序，与具体操作分别测量，不能重复相加。

| 分组 | 实际算子名称 | 类型 | 矩阵精度 | 所属 API | 直接测量覆盖 |
| --- | --- | --- | --- | --- | --- |
| attention/topk | `exact_topk` | api_aggregate | N/A | `exact_topk` | 52 B/Q/H points |
| attention/cache_preparation | `indexer_prepare` | api_aggregate | N/A | `indexer_prepare` | 28 B/Q/H points |
| attention/sparse_attention | `mla_global_indices` | api_aggregate | N/A | `sparse_mla` | 48 B/Q/H points |
| attention/sparse_attention | `sparse_mla` | api_aggregate | bf16 | `sparse_mla` | 52 B/Q/H points |
| attention/topk | `topk_output_clone` | api_aggregate | N/A | `exact_topk` | 52 B/Q/H points |
| attention/topk | `topk_postprocess` | api_aggregate | N/A | `exact_topk` | 52 B/Q/H points |
| common/input_projection | `attention_projection` | api_aggregate | mixed | `attention_projection` | S=1,32,64,128,256,512,1024 |
| common/dense_mlp | `dense_mlp` | api_aggregate | fp8 | `dense_mlp` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `index_k_proj` | api_aggregate | fp8 | `index_k_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `index_q_proj` | api_aggregate | fp8 | `index_q_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `index_weights_proj` | api_aggregate | fp32 | `index_weights_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `kv_a_proj` | api_aggregate | fp8 | `kv_a_proj` | S=1,32,64,128,256,512,1024 |
| common/lm_head | `lm_head` | api_aggregate | bf16 | `lm_head` | S=1,32,64,128,256,512,1024 |
| common/dense_mlp | `mlp_down` | api_aggregate | fp8 | `mlp_down` | S=1,32,64,128,256,512,1024 |
| common/dense_mlp | `mlp_gate` | api_aggregate | fp8 | `mlp_gate` | S=1,32,64,128,256,512,1024 |
| common/dense_mlp | `mlp_up` | api_aggregate | fp8 | `mlp_up` | S=1,32,64,128,256,512,1024 |
| common/output_projection | `o_proj` | api_aggregate | fp8 | `o_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `q_a_proj` | api_aggregate | fp8 | `q_a_proj` | S=1,32,64,128,256,512,1024 |
| common/input_projection | `q_b_proj` | api_aggregate | fp8 | `q_b_proj` | S=1,32,64,128,256,512,1024 |
| common/rope | `rotary_cache` | api_aggregate | N/A | `rotary_cache` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe` | api_aggregate | mixed | `moe` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_grouped_down` | api_aggregate | fp8 | `moe_grouped_down` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_grouped_gate` | api_aggregate | fp8 | `moe_grouped_gate` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_grouped_up` | api_aggregate | fp8 | `moe_grouped_up` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_expert_counts` | api_aggregate | N/A | `moe_metadata_expert_counts` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_inverse_rank` | api_aggregate | N/A | `moe_metadata_inverse_rank` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_local_rows` | api_aggregate | N/A | `moe_metadata_local_rows` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_output_slot_arithmetic` | api_aggregate | N/A | `moe_metadata_output_slot_arithmetic` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_packed_valid` | api_aggregate | N/A | `moe_metadata_packed_valid` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_route_valid` | api_aggregate | N/A | `moe_metadata_route_valid` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_row_offsets` | api_aggregate | N/A | `moe_metadata_row_offsets` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_safe_experts` | api_aggregate | N/A | `moe_metadata_safe_experts` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_sorted_positions` | api_aggregate | N/A | `moe_metadata_sorted_positions` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_counts` | api_aggregate | N/A | `moe_metadata_tile_counts` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_metadata_tile_offsets` | api_aggregate | N/A | `moe_metadata_tile_offsets` | S=1,32,64,128,256,512,1024 |
| moe/layout | `moe_prepare_routing` | api_aggregate | N/A | `moe_prepare_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_activation | `moe_routed_silu` | api_aggregate | N/A | `moe_routed_silu` | S=1,32,64,128,256,512,1024 |
| moe/moe_routing | `moe_routing` | api_aggregate | fp32 | `moe_routing` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_shared_down` | api_aggregate | fp8 | `moe_shared_down` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_shared_expert` | api_aggregate | fp8 | `moe_shared_expert` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_shared_gate` | api_aggregate | fp8 | `moe_shared_gate` | S=1,32,64,128,256,512,1024 |
| moe/moe_activation | `moe_shared_silu` | api_aggregate | N/A | `moe_shared_silu` | S=1,32,64,128,256,512,1024 |
| moe/moe_gemm | `moe_shared_up` | api_aggregate | fp8 | `moe_shared_up` | S=1,32,64,128,256,512,1024 |

## Projection 图表

![Projection](figures/common_projection_1.png)

![Projection](figures/common_projection_2.png)

## 其他具体操作与 API 聚合

每行四列依次为 clean 时间、clean MFU、NCU HBM bandwidth 与 NCU HBM utilization。
图中的 N/A 表示非 GEMM，不以零值代替；标题含 non-additive 的图是独立 API 聚合。

![common_norm_1](figures/common_norm_1.png)

![common_norm_2](figures/common_norm_2.png)

![common_rope_1](figures/common_rope_1.png)

![common_quantization_1](figures/common_quantization_1.png)

![common_quantization_2](figures/common_quantization_2.png)

![common_quantization_3](figures/common_quantization_3.png)

![common_scale_layout_1](figures/common_scale_layout_1.png)

![common_scale_layout_2](figures/common_scale_layout_2.png)

![common_cast_1](figures/common_cast_1.png)

![common_scale_1](figures/common_scale_1.png)

![common_layout_1](figures/common_layout_1.png)

![common_aggregate_dense_mlp_1](figures/common_aggregate_dense_mlp_1.png)

![common_aggregate_lm_head_1](figures/common_aggregate_lm_head_1.png)

![common_aggregate_projection_1](figures/common_aggregate_projection_1.png)

![common_aggregate_projection_2](figures/common_aggregate_projection_2.png)

![common_aggregate_rope_1](figures/common_aggregate_rope_1.png)

![common_dense_mlp_1](figures/common_dense_mlp_1.png)

![common_embedding_1](figures/common_embedding_1.png)

![common_lm_head_1](figures/common_lm_head_1.png)

![moe_quantization_1](figures/moe_quantization_1.png)

![moe_scale_layout_1](figures/moe_scale_layout_1.png)

![moe_layout_1](figures/moe_layout_1.png)

![moe_layout_2](figures/moe_layout_2.png)

![moe_layout_3](figures/moe_layout_3.png)

![moe_layout_4](figures/moe_layout_4.png)

![moe_layout_5](figures/moe_layout_5.png)

![moe_layout_6](figures/moe_layout_6.png)

![moe_layout_7](figures/moe_layout_7.png)

![moe_layout_8](figures/moe_layout_8.png)

![moe_layout_9](figures/moe_layout_9.png)

![moe_layout_10](figures/moe_layout_10.png)

![moe_layout_11](figures/moe_layout_11.png)

![moe_layout_12](figures/moe_layout_12.png)

![moe_moe_routing_1](figures/moe_moe_routing_1.png)

![moe_moe_routing_2](figures/moe_moe_routing_2.png)

![moe_moe_routing_3](figures/moe_moe_routing_3.png)

![moe_moe_gemm_1](figures/moe_moe_gemm_1.png)

![moe_moe_activation_1](figures/moe_moe_activation_1.png)

![moe_moe_cast_1](figures/moe_moe_cast_1.png)

![moe_moe_reduce_1](figures/moe_moe_reduce_1.png)

![moe_aggregate_layout_1](figures/moe_aggregate_layout_1.png)

![moe_aggregate_layout_2](figures/moe_aggregate_layout_2.png)

![moe_aggregate_moe_activation_1](figures/moe_aggregate_moe_activation_1.png)

![moe_aggregate_moe_gemm_1](figures/moe_aggregate_moe_gemm_1.png)

![moe_aggregate_moe_gemm_2](figures/moe_aggregate_moe_gemm_2.png)

![moe_aggregate_moe_routing_1](figures/moe_aggregate_moe_routing_1.png)

![attention_b1_q1_aggregate_cache_preparation_1](figures/attention_b1_q1_aggregate_cache_preparation_1.png)

![attention_b1_q1_aggregate_sparse_attention_1](figures/attention_b1_q1_aggregate_sparse_attention_1.png)

![attention_b1_q1_aggregate_topk_1](figures/attention_b1_q1_aggregate_topk_1.png)

![attention_b1_q1_cache_preparation_1](figures/attention_b1_q1_cache_preparation_1.png)

![attention_b1_q1_indexer_1](figures/attention_b1_q1_indexer_1.png)

![attention_b1_q1_sparse_attention_1](figures/attention_b1_q1_sparse_attention_1.png)

![attention_b1_q1_topk_1](figures/attention_b1_q1_topk_1.png)

![attention_b32_q1_aggregate_cache_preparation_1](figures/attention_b32_q1_aggregate_cache_preparation_1.png)

![attention_b32_q1_aggregate_sparse_attention_1](figures/attention_b32_q1_aggregate_sparse_attention_1.png)

![attention_b32_q1_aggregate_topk_1](figures/attention_b32_q1_aggregate_topk_1.png)

![attention_b32_q1_cache_preparation_1](figures/attention_b32_q1_cache_preparation_1.png)

![attention_b32_q1_indexer_1](figures/attention_b32_q1_indexer_1.png)

![attention_b32_q1_sparse_attention_1](figures/attention_b32_q1_sparse_attention_1.png)

![attention_b32_q1_sparse_attention_2](figures/attention_b32_q1_sparse_attention_2.png)

![attention_b32_q1_topk_1](figures/attention_b32_q1_topk_1.png)

![attention_b64_q1_aggregate_cache_preparation_1](figures/attention_b64_q1_aggregate_cache_preparation_1.png)

![attention_b64_q1_aggregate_sparse_attention_1](figures/attention_b64_q1_aggregate_sparse_attention_1.png)

![attention_b64_q1_aggregate_topk_1](figures/attention_b64_q1_aggregate_topk_1.png)

![attention_b64_q1_cache_preparation_1](figures/attention_b64_q1_cache_preparation_1.png)

![attention_b64_q1_indexer_1](figures/attention_b64_q1_indexer_1.png)

![attention_b64_q1_sparse_attention_1](figures/attention_b64_q1_sparse_attention_1.png)

![attention_b64_q1_sparse_attention_2](figures/attention_b64_q1_sparse_attention_2.png)

![attention_b64_q1_topk_1](figures/attention_b64_q1_topk_1.png)

![attention_b128_q1_aggregate_cache_preparation_1](figures/attention_b128_q1_aggregate_cache_preparation_1.png)

![attention_b128_q1_aggregate_sparse_attention_1](figures/attention_b128_q1_aggregate_sparse_attention_1.png)

![attention_b128_q1_aggregate_topk_1](figures/attention_b128_q1_aggregate_topk_1.png)

![attention_b128_q1_cache_preparation_1](figures/attention_b128_q1_cache_preparation_1.png)

![attention_b128_q1_indexer_1](figures/attention_b128_q1_indexer_1.png)

![attention_b128_q1_sparse_attention_1](figures/attention_b128_q1_sparse_attention_1.png)

![attention_b128_q1_sparse_attention_2](figures/attention_b128_q1_sparse_attention_2.png)

![attention_b128_q1_topk_1](figures/attention_b128_q1_topk_1.png)

![attention_b256_q1_aggregate_cache_preparation_1](figures/attention_b256_q1_aggregate_cache_preparation_1.png)

![attention_b256_q1_aggregate_sparse_attention_1](figures/attention_b256_q1_aggregate_sparse_attention_1.png)

![attention_b256_q1_aggregate_topk_1](figures/attention_b256_q1_aggregate_topk_1.png)

![attention_b256_q1_cache_preparation_1](figures/attention_b256_q1_cache_preparation_1.png)

![attention_b256_q1_indexer_1](figures/attention_b256_q1_indexer_1.png)

![attention_b256_q1_sparse_attention_1](figures/attention_b256_q1_sparse_attention_1.png)

![attention_b256_q1_sparse_attention_2](figures/attention_b256_q1_sparse_attention_2.png)

![attention_b256_q1_topk_1](figures/attention_b256_q1_topk_1.png)

![attention_b512_q1_aggregate_cache_preparation_1](figures/attention_b512_q1_aggregate_cache_preparation_1.png)

![attention_b512_q1_aggregate_sparse_attention_1](figures/attention_b512_q1_aggregate_sparse_attention_1.png)

![attention_b512_q1_aggregate_topk_1](figures/attention_b512_q1_aggregate_topk_1.png)

![attention_b512_q1_cache_preparation_1](figures/attention_b512_q1_cache_preparation_1.png)

![attention_b512_q1_indexer_1](figures/attention_b512_q1_indexer_1.png)

![attention_b512_q1_sparse_attention_1](figures/attention_b512_q1_sparse_attention_1.png)

![attention_b512_q1_sparse_attention_2](figures/attention_b512_q1_sparse_attention_2.png)

![attention_b512_q1_topk_1](figures/attention_b512_q1_topk_1.png)

![attention_b1024_q1_aggregate_cache_preparation_1](figures/attention_b1024_q1_aggregate_cache_preparation_1.png)

![attention_b1024_q1_aggregate_sparse_attention_1](figures/attention_b1024_q1_aggregate_sparse_attention_1.png)

![attention_b1024_q1_aggregate_topk_1](figures/attention_b1024_q1_aggregate_topk_1.png)

![attention_b1024_q1_cache_preparation_1](figures/attention_b1024_q1_cache_preparation_1.png)

![attention_b1024_q1_indexer_1](figures/attention_b1024_q1_indexer_1.png)

![attention_b1024_q1_sparse_attention_1](figures/attention_b1024_q1_sparse_attention_1.png)

![attention_b1024_q1_sparse_attention_2](figures/attention_b1024_q1_sparse_attention_2.png)

![attention_b1024_q1_topk_1](figures/attention_b1024_q1_topk_1.png)

![attention_b1_q32_aggregate_sparse_attention_1](figures/attention_b1_q32_aggregate_sparse_attention_1.png)

![attention_b1_q32_aggregate_topk_1](figures/attention_b1_q32_aggregate_topk_1.png)

![attention_b1_q32_indexer_1](figures/attention_b1_q32_indexer_1.png)

![attention_b1_q32_sparse_attention_1](figures/attention_b1_q32_sparse_attention_1.png)

![attention_b1_q32_sparse_attention_2](figures/attention_b1_q32_sparse_attention_2.png)

![attention_b1_q32_topk_1](figures/attention_b1_q32_topk_1.png)

![attention_b1_q64_aggregate_sparse_attention_1](figures/attention_b1_q64_aggregate_sparse_attention_1.png)

![attention_b1_q64_aggregate_topk_1](figures/attention_b1_q64_aggregate_topk_1.png)

![attention_b1_q64_indexer_1](figures/attention_b1_q64_indexer_1.png)

![attention_b1_q64_sparse_attention_1](figures/attention_b1_q64_sparse_attention_1.png)

![attention_b1_q64_sparse_attention_2](figures/attention_b1_q64_sparse_attention_2.png)

![attention_b1_q64_topk_1](figures/attention_b1_q64_topk_1.png)

![attention_b1_q128_aggregate_sparse_attention_1](figures/attention_b1_q128_aggregate_sparse_attention_1.png)

![attention_b1_q128_aggregate_topk_1](figures/attention_b1_q128_aggregate_topk_1.png)

![attention_b1_q128_indexer_1](figures/attention_b1_q128_indexer_1.png)

![attention_b1_q128_sparse_attention_1](figures/attention_b1_q128_sparse_attention_1.png)

![attention_b1_q128_sparse_attention_2](figures/attention_b1_q128_sparse_attention_2.png)

![attention_b1_q128_topk_1](figures/attention_b1_q128_topk_1.png)

![attention_b1_q256_aggregate_sparse_attention_1](figures/attention_b1_q256_aggregate_sparse_attention_1.png)

![attention_b1_q256_aggregate_topk_1](figures/attention_b1_q256_aggregate_topk_1.png)

![attention_b1_q256_indexer_1](figures/attention_b1_q256_indexer_1.png)

![attention_b1_q256_sparse_attention_1](figures/attention_b1_q256_sparse_attention_1.png)

![attention_b1_q256_sparse_attention_2](figures/attention_b1_q256_sparse_attention_2.png)

![attention_b1_q256_topk_1](figures/attention_b1_q256_topk_1.png)

![attention_b1_q512_aggregate_sparse_attention_1](figures/attention_b1_q512_aggregate_sparse_attention_1.png)

![attention_b1_q512_aggregate_topk_1](figures/attention_b1_q512_aggregate_topk_1.png)

![attention_b1_q512_indexer_1](figures/attention_b1_q512_indexer_1.png)

![attention_b1_q512_sparse_attention_1](figures/attention_b1_q512_sparse_attention_1.png)

![attention_b1_q512_sparse_attention_2](figures/attention_b1_q512_sparse_attention_2.png)

![attention_b1_q512_topk_1](figures/attention_b1_q512_topk_1.png)

![attention_b1_q1024_aggregate_sparse_attention_1](figures/attention_b1_q1024_aggregate_sparse_attention_1.png)

![attention_b1_q1024_aggregate_topk_1](figures/attention_b1_q1024_aggregate_topk_1.png)

![attention_b1_q1024_indexer_1](figures/attention_b1_q1024_indexer_1.png)

![attention_b1_q1024_sparse_attention_1](figures/attention_b1_q1024_sparse_attention_1.png)

![attention_b1_q1024_sparse_attention_2](figures/attention_b1_q1024_sparse_attention_2.png)

![attention_b1_q1024_topk_1](figures/attention_b1_q1024_topk_1.png)

MoE grouped GEMM 叶操作与 packing、量化、mapping、activation、加权归约分别展示。
完整 MoE 聚合另列；useful FLOPs 只计实际 selected routes，padding capacity 不等于执行 FLOPs。

来源、results SHA-256、具体工作负载的输入身份、数值验收 receipt、clean result、
NCU profile result 与原始 profiler 文件哈希见 [provenance.json](provenance.json)。
独立算子验收不代替完整模型或实际 serving 验证。

## 独立审计证据

[完整性核查](completion.json)保存各组覆盖、样本和源码/native 绑定的核查结果；
[attention 验收审计](attention_audit.json)保存全部 52 个配置的参考覆盖与检查。
[最大形状 Indexer 数值诊断](indexer_numerical_diagnostic.json)保留官方 normalized
criterion 与旧逐元素标准的诊断值；旧标准不决定本次验收。

[实际 NCU kernel 清单](attention_actual_ncu_kernel_inventory.csv)记录每项 attention
操作的真实 launch 名称及每调用数量。有些 non-GEMM callable 执行多个 kernel；
本报告保留完整 callable 边界，没有给内部 kernel 虚构独立的 clean 时间。
[profile/clean 时间比值](attention_profile_clean_time_ratios.csv)、
[profile 审计汇总](attention_profile_audit_summary.json)与
[H64K Decode 选定数值](attention_selected_decode_h65536.csv)供逐项回查。
[硬件与官方规格证据](hardware.json)保留原查询日期，不证明整个运行期间设备持续独占。
