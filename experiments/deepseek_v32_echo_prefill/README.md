# DeepSeek V3.2 ECHO prefill/extend

**2026-10-02 状态修正：ECHO baseline 尚未成立。** 研究者指出当前 ECHO 实现的
MFU 和 cache 策略有问题，DeepSeek 当前实现的 MFU 也不太正确。因此，本页相关
MFU、性能归因及 ECHO baseline 比较暂不可用于研究结论；此前数值一致性和采样归因
审计通过，不等于 cache 策略合理或 baseline 实现效率已经验收。具体根因尚待核查，
不能据此断言某个 FLOPs 公式错误，也不能将低 MFU 数字直接视为 ECHO 方法的局限。

前三层 run `20261002_echo_layers3_mfu_01` 已采集并发布，以下保留该版本的原始
数字、报告素材和对应运行产物以便复核，待修正后以新 run ID 补测验收，再统一替换
和清理。本轮只修正状态与结论边界，未修复实现、未重新运行 GPU 性能实验。
该 run 仅执行 checkpoint 第 0–2 层，包含真实顺序的 hidden/residual 传播，
没有复制层、没有 MoE；其单卡 resident/ECHO 记录不替代完整 61 层或 GR serving
的性能验证。下文完整 61 层旧报告保留其原测量边界，并同样受 baseline 待修正限制。

本轮硬件为物理 GPU 1，UUID `GPU-a2226185-cb05-a411-80da-f365154128fe`；
驱动名称为 NVIDIA M403，PCI `10de:2335` 与本地 PCI 数据库对应 H200 SXM 141GB，
SM90 / 132 SM。使用默认频率与本机 pinned CPU DRAM；GPU 0 有其他任务，
GPU 1 采集前空闲。时钟记录是采前快照，不是连续 DVFS 采样。
PyTorch 2.12.1+cu130、Triton 3.7.1、Nsight Systems 2025.6.3、
Nsight Compute 2026.1.1；完整环境和官网 dense peak 换算证据见
[summary.json](report/layers3/summary.json)。

使用原 run 保存的同一 GR 请求 token IDs，边界严格为 65,536 + 1,024。
两种模式各自从空 cache 构建 prefix，每次 extend 恢复相同的 prefix HBM residency。
各阶段预热 1 次；加载、编译、状态恢复和数值比较在计时外。正式延迟不安装诊断包装，
prefix 与 extend 的四份独立 nsys capture 才开启逐算子 NVTX 包装。

八项数值检查均逐元素一致（max_abs=0）：包括两模式的全部 1,024×7,168 extend
normalized hidden 和最后 token logits，以及插桩/未插桩对照；prefix 核验最后
token logits。独立审计重算 5,496 次矩阵调用 FLOPs，复核 136,887 个 kernel，
归因覆盖率为 100%，计数和时间守恒。三个层的 query 拆分无遗漏或重复，
resident/offload 的 useful FLOPs 按当时口径相同。详见[独立审计](report/layers3/postrun_audit.json)。
审计文件中的 `accepted` / `verified_mfu` 记录当时的计数、归因和算术检查状态，
不构成对最新 MFU 问题或 cache 策略的复核结论。
Checkpoint 记录配置、tokenizer 与权重索引 hash，未对全部权重 shard 内容另做 hash。

<!-- BEGIN LAYERS3 PROFILE RESULTS -->
## 前三层重新测量：`20261002_echo_layers3_mfu_01`

**旧实现记录，baseline 待修正复测。** 下列延迟与 MFU 数字保留原 run 的测量口径，
不作为已验收 baseline 或方法优劣证据。ECHO 的 MFU/cache 策略及 DeepSeek 实现
MFU 问题尚未完成定位和修正；已有数值、计数与归因检查不覆盖这些问题。

完整 checkpoint 的第 0–2 层依次传播 hidden/residual，包含 embedding、final norm 和最后 token LM head。Prefix=65,536，extend=1,024，chunk=1,024，offload pool=16,384 tokens/层。仅代表前三层。

| 阶段 | Resident 中位延迟 (ms) | Offload 中位延迟 (ms) | 每种模式重复次数 |
| --- | ---: | ---: | ---: |
| prefix | 2668.638 | 3540.019 | 3 |
| extend | 46.830 | 65.675 | 5 |

![前三层无插桩延迟](report/layers3/latency.svg)

原报告的计算口径为 MFU = useful matrix FLOPs /（同算子实际 GPU kernel duration 之和 × 对应精度 dense peak），其有效性待复核。FMA 计 2 FLOPs；FP8/BF16/FP32 分母分别为 1979.0/989.5/67.0 TFLOPS；TF32 关闭。前缀汇总全部 chunk 与三层，extend 汇总三层的一次完整 query batch（含实际 offload leaf 拆分）。LM head 每个阶段仅执行最后一个 token。非矩阵算子的 MFU 为 N/A。

每种模式、每个阶段各采集一次独立 annotated profile；算子 MFU 没有重复采样置信区间。FP8Linear 的分母包含量化和 GEMM，indexer 包含 logits API 的辅助 kernel；这些是原报告按完整算子计算的数值，不能视为已通过最新复核的 MFU，也不同于单 GEMM 或 Tensor pipe active 指标。

![四组逐算子 MFU](report/layers3/mfu.svg)

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

正确性与覆盖验收、硬件 identity、源码及输入 SHA256、完整 NCU full/source 的验证记录见 [summary.json](report/layers3/summary.json)。逐层及 pooled 数据见 [operator_mfu.csv](report/layers3/operator_mfu.csv)、[operator_mfu_by_layer.csv](report/layers3/operator_mfu_by_layer.csv)、[nonmatrix.csv](report/layers3/nonmatrix.csv)。

NCU 是同 run 对应真实层激活的独立 replay，不替代 NSYS 实际调用的 MFU 或正式 wall 延迟；其 cache 状态、排除项和报告 SHA256 单独保存。NCU run IDs：`20261002_echo_layers3_ncu_indexer_resident_01`, `20261002_echo_layers3_ncu_indexer_offload_01`, `20261002_echo_layers3_ncu_mla_01`。

生成命令：

```bash
python -m experiments.deepseek_v32_echo_prefill.src.publish_layers --run-id 20261002_echo_layers3_mfu_01 --ncu-run-id 20261002_echo_layers3_ncu_indexer_resident_01 --ncu-run-id 20261002_echo_layers3_ncu_indexer_offload_01 --ncu-run-id 20261002_echo_layers3_ncu_mla_01 --publish
```
<!-- END LAYERS3 PROFILE RESULTS -->

## 原 run 的诊断记录与待核查项

原 run 中 ECHO 的 prefix 比 resident 慢 32.65%，extend 慢 40.24%。这些差值只记录
当时三层实现的执行结果，不能作为有效 ECHO baseline 比较，不能外推全部 61 层。
下列 kernel/API 观察保留为复核线索；MFU 解释、性能归因与优化次序须待实现及 cache
策略核查后重建。

- Extend 的 resident MLA 为 22.066 ms，占该 capture 全部 kernel 时间的 49.14%；
  offload 为 21.935 ms。原报告 MFU 为 8.03% / 8.07%，现列为待复核数字。
  该项同时计入 QK 与 PV，原归因没有把同一 kernel 时间重复分给两者。
- 三层 extend 的完整 logits API GPU 时间由 3.641 ms 增至 12.181 ms；原报告
  MFU 为 46.14% / 13.79%，现列为待复核数字。融合预取、histogram 和辅助 kernel
  的时间均包含在原分母，不能把两项差值全称为 KV 传输时间。
- 同一组 annotated extend 中，
  `cudaStreamSynchronize` 从 3 次增至 124 次，其 API 时间由 0.020 ms 增至
  35.554 ms；GPU activity envelope 内 gap 从 3.226 ms 增至 13.129 ms。
  API 时间包含等待 GPU，不是能从 wall time 直接扣掉的独立 CPU 成本；这些记录
  尚未解释 cache 策略问题的具体根因。
- index_k_proj 的三层 extend 约 0.214 ms，最后单 token 的 LM head 约 0.421 ms。
  原报告分别给出约 1.33% / 0.44% MFU；本轮不再据此排列优化优先级。

### 新 NCU：真实第 2 层输入

三个 kernel 分别采集 full + PmSampling + PmSampling_WarpStates，以及独立
SourceCounters；每次预热 2 次、采一个目标 launch，kernel replay、cache-control all、
clock-control none。NCU 2026.1.1 无 source set，因此使用 SourceCounters section。
这六份新报告的来源 run ID、native report hash、source/PM 摘要见
[summary.json](report/layers3/summary.json)，精确指标见[ncu_metrics.csv](report/layers3/ncu_metrics.csv)。

| NCU 指标 | Resident indexer | Fused indexer/prefetch | Sparse MLA |
| --- | ---: | ---: | ---: |
| Replay duration (ms) | 1.119 | 2.423 | 7.413 |
| Achieved occupancy (%) | 14.06 | 26.56 | 12.49 |
| Tensor pipe active (% elapsed) | 51.64 | 23.10 | 12.16 |
| HBM throughput (% peak) | 5.38 | 3.50 | 1.27 |
| Registers/thread | 112 | 96 | 163 |
| Local spilling requests | 0 | 21,450,007 | 0 |

融合 indexer 的 source sampling 在 CUTLASS `barrier.h:424` 的等待位置记录
80,394 个 long-scoreboard samples，伴随 21.45M spilling requests；这些是待核查的
寄存器压力与同步等待线索，尚未建立 baseline 问题的根因。MLA 的 L1/TEX throughput
为 67.45%，occupancy 仅 12.49%；
`mla.py:75–76` 的 KV load 合计 174,621 个 long-scoreboard samples，`:79` QK dot
有 113,651 个 short-scoreboard samples。两者的 HBM throughput 都很低，现有数据
不支持 HBM 带宽饱和的解释。

NCU fused replay 使用冷历史 pool、offset=0、最多预取 8,192 records；未恢复模型
执行时的 cache 命中与 histogram 状态。MLA replay 使用完整 resident KV 与逻辑选择，
不是 offload 物理 pool。NCU duration 不替代前面来自真实调用的算子 MFU/延迟，
也不能据此计算完整模型 offload 收益。

## 前三层复现入口

从仓库根目录执行，选择空闲的同型号 GPU；`--physical-device` 必须与可见设备 UUID
一致。旧请求文件只作为固定输入复用，不复用旧性能数字：

```bash
CUDA_VISIBLE_DEVICES=1 \
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-profile \
ECHO_RUN_ID=<NEW_RUN_ID> \
bash experiments/deepseek_v32_echo_prefill/scripts/profile_layers.sh \
  --physical-device 1 --model /preset-models \
  --request experiments/deepseek_v32_echo_prefill/output/data/20260930T0115Z_echo_full61_fp8/request.json

CUDA_VISIBLE_DEVICES=1 \
bash experiments/deepseek_v32_echo_prefill/scripts/ncu.sh \
  --input experiments/deepseek_v32_echo_prefill/output/data/<NEW_RUN_ID>/kernel_inputs_layer_2.pt \
  --kernel indexer-offload --run-id <NEW_NCU_RUN_ID>
```

`--kernel` 分别选择 `indexer-resident`、`indexer-offload`、`mla`，使用不同 run ID。
NCU 提取入口沿用下文的 `src.analyze_ncu`；报告生成命令见前面的结果区段。
脚本拒绝覆盖已有 run。原始 nsys 在
`output/profile/20261002_echo_layers3_mfu_01/`，SQLite、调用形状与 FLOPs ledger、
源码快照、数值张量、独立审计源码在对应 `output/data/`；原始 NCU 在各自 run 的
`output/profile/`。完整 kernel 名称清单在该主 run 的 `analysis/kernel_inventory.csv`。
原分析中的 staging 路径到当前路径的映射及 SHA 验证保存在独立审计内。

新增入口复用 `DeepSeekEchoModel(num_layers=3)`、原 indexer/MLA/linear 与
`SparseTokenCache`；诊断包装仅存在于实验中，不替换模型计算。
调用模块还包括本实验的 `operator_instrumentation`、`operator_flops`、
`operator_report`、`profile_hardware`、`publish_layers`；NCU 摘要复用
`profile_summary.collect_ncu`。相关 CPU 回归为 163 passed，正式 GPU profile 与
数值验收另按上述 run 记录；CPU 测试本身不是实验结果。

## 完整 61 层原报告（修复后性能仍待补测）

以下保留原完整模型 run 的结果与测量边界。前三层诊断与其范围不同，未将旧数据
改写为新实现结果，也未用三层结果替代完整 checkpoint 验证。

2026-10-01 目录整理：算子已迁至 `operators/nosa/`、`operators/deepseek_v32/` 和
`operators/common/`。本页性能仍对应下文原 run ID 与源码快照；目录迁移后的性能未重新测量。

本实验复刻 ECHO 的 extend indexer 内融合 KV prefetch，比较完整 DeepSeek V3.2
的 65,536-token prefix prefill + 1,024-token extend，在 resident 与 local DRAM
offload 两种模式下的端到端延迟、attention 时间和实际 KV 搬运量。

**现有报告对应修复前版本；2026-10-01 的 KV gather 对齐修复后，完整模型性能尚未补测。**
旧版完整 61 层、64K + 1K 的 resident/offload 记录曾完成数值与测量检查，最终
logits 逐位一致；这不构成当前 MFU/cache 策略问题下的 baseline 验收。
Run ID：`20260930T0115Z_echo_full61_fp8`。本次 offload 降低主 KV 的 HBM 分配，
但增加端到端延迟；具体数值与测量边界见下文。旧 SM120 实验不受新增路径影响。

旧 run 的 `kv_transfer.cu` SHA256 为
`b776b40a1ceb83be864cc210ef26e73cf4bacec45c3f4f2b3f0e8af355a39742`，完整源码身份保留在
[summary.json](report/summary.json)。该运行的 cache 自行分配对齐 record buffers，
没有触发本次发现的非对齐 storage-offset 问题；其结果只说明旧实现和原测量边界，
不能用于证明修复后延迟。修复后的 gather 会检查源、目标行地址，非对齐视图使用逐字节复制。
旧报告及对应运行产物保留到补测验收并发布完成；NOSA overlap 不调用此 gather，结果不受影响。
2026-10-01 已恢复项目锁定环境；目录迁移后的 gather 对齐/非对齐检查及缓存回归
已通过。完整模型性能仍未补测；这些代码正确性检查不构成新一轮性能实验。

ECHO 参考提交为 `bc1b75c1000010d0ac6f032ebaac283255c050b1`，上游
`DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh` 的 extend 分支。
本地 kernel 只复用共享 CUTLASS 头文件，不导入 SGLang 或 ECHO Python runtime。
源码中保留所提取 MIT 代码的许可与出处。

## 测量边界

- 完整 checkpoint 的 61 层，包含 embedding、三个 dense MLP、58 个 MoE、final norm
  和末 token LM head；不把单层结果外推成整模型。
- 真实 GR 可读输入，稳定历史与候选后缀的边界严格为 65,536 + 1,024 tokens。
- 原始 FP8 checkpoint 权重在分配到的 GPU 上常驻；普通投影使用 block FP8 GEMM，
  吸收的 MLA K/V 投影使用 checkpoint 反量化 BF16。
- Indexer 使用 RoPE、normalized Hadamard、FP8 Q/K、64 heads、top-2048。
  主 KV 按 ECHO 使用 512 latent + 64 RoPE BF16，共 1152 B/token。
- Offload 使用 pinned local DRAM backing、有限 HBM token pool、融合 coarse histogram
  预取、精确 top-k 的 residual recall 和物理 ID remap。过大的 query union 分批消费，
  每 query 的精确选择不变。该实现不包含 CXL/RDMA。
- Prefix 从空 cache 构建。Extend 每次恢复同一 prefix 的 HBM residency，防止测成
  重复请求的热缓存。计时包含 CPU 调度、GPU 计算、KV 写入、搬运和等待；不包含加载
  权重、编译和恢复基准状态。带事件/NVTX 的拆分采样与无插桩延迟分开记录。
- 使用串行层放置，每个 1024-token chunk 遍历全部 61 层，然后处理下一个 chunk；
  层边界传递 hidden/residual。该结果对应单请求执行器，不代表 tensor parallel 或
  continuous batching serving 吞吐。分词、请求生成和 logits 的 CPU 验收在计时外。

## 本次运行条件

5 张 NVIDIA H20Z（SM90、每张 132 SM，PyTorch 可见显存 143167 MiB），物理 GPU
`0,1,2,6,7`；层放置依次为 `0–13 / 14–24 / 25–36 / 37–48 / 49–60`。权重在这些
设备常驻，主机 pinned DRAM 只承载主 MLA KV。主机还有其他 GPU 任务，未锁定 GPU 频率。

依赖：PyTorch `2.12.1+cu130`、Triton `3.7.1`、safetensors `0.8.0`、TVM FFI
`0.1.13.post3`；共享 CUTLASS `f3fde583`；Nsight Systems `2025.6.3`、Nsight Compute
`2026.1.1`。权重来自 `/mnt/user-ssd/chenkaiqi/DeepSeek-V3.2/`，checkpoint metadata、
请求及 14 个运行源码文件的 SHA256 随本次数据保存。

每种模式各做 1 次 prefix 预热、1 次正式 prefix、1 次独立 annotated prefix；从所得
prefix 状态做 1 次 extend 预热、5 次正式 extend 和 1 次 annotated extend。正式计时
不创建事件/NVTX scope，Nsight 注入库仍加载、capture 关闭。Nsight trace 仅捕获两种
模式各自的 annotated extend；prefix 分项来自 CUDA events。额外保存第 0/30/60 层
kernel 输入的运行不参与计时。仅一个 GR 请求、一个 seed，prefix 无重复方差估计。

<!-- BEGIN ECHO GENERATED RESULTS -->
## 完整模型旧记录（baseline 待修正复测）

以下保留原 run 的数字与来源，不用于当前 ECHO baseline 的性能结论。
除 gather 修复后的完整模型补测外，还须处理研究者指出的 MFU/cache 策略问题。

来源 run ID：`20260930T0115Z_echo_full61_fp8`。完整 61 层 checkpoint，65,536-token prefix 从空 cache 构建，随后执行 1,024-token extend；只输出最后 token 的完整词表 logits。
设备：cuda:0, cuda:1, cuda:2, cuda:6, cuda:7；chunk=1024；每层 offload pool=16384 tokens；预热 1 次。硬件详情、依赖版本、源码 SHA256 与逐阶段数据见 [summary.json](report/summary.json)。

| 阶段 | Resident 中位延迟 (ms) | Offload 中位延迟 (ms) | Offload / resident | 重复次数 |
| --- | ---: | ---: | ---: | ---: |
| Prefix prefill | 66342.000 | 202897.830 | 3.0584× | 1 |
| Extend | 1171.554 | 2220.250 | 1.8951× | 5 |

主 MLA KV 的 HBM record allocation 总计从 4.3561 GiB 降至 1.0723 GiB；该数字不含权重、indexer cache、映射、scratch 或激活。

prefill 的独立 annotated 采样合计 H2D 235.9967 GiB、D2H 4.2891 GiB，来自 61 层实际 record 计数，每 record 为 1152 B。
extend 的独立 annotated 采样合计 H2D 2.5065 GiB、D2H 0.0670 GiB，来自 61 层实际 record 计数，每 record 为 1152 B。

H2D 统计包含融合预取与全部 gather；`recalled_records` 同时包含 `offload_prepare` 对当前 chunk 的 staging 与后续 exact recall，不能全部解释为 residual recall。D2H 统计为新 KV 写入 host backing。

Resident/offload 保存的末 token logits 已逐元素核验 bitwise 相同（max_abs=0，NRMSE=0），next token 相同。

![完整模型无插桩延迟](report/latency.png)

![独立 annotated CUDA scope](report/annotated_scopes.png)

延迟表来自无插桩完整 forward。Scope 图来自独立带事件采样，各类别分别对照，CUDA-event elapsed time 仅扣除嵌套子 scope；仍可能包含 CPU 提交空隙、stream 依赖等待与跨 GPU 重叠。这些区间不等价于 nsys 的实际 kernel duration，不能相加作端到端分解或加回无插桩 wall time。`hidden_transfer` 包含等待源 GPU 计算完成的时间，不能当成独立 NVLink copy 耗时或据此推算带宽。

`indexer_prefetch` 同时包含 indexer 计算与融合预取，不能解读为独立搬运耗时；`offload_prepare` 与 `offload_exact_recall` 分别呈现。Prefix 与 extend 分别测量；输入准备、权重加载、编译及每次恢复 prefix residency 均在计时外。

图表、[CSV](report/summary.csv) 与 JSON 由 `python -m experiments.deepseek_v32_echo_prefill.src.report --result experiments/deepseek_v32_echo_prefill/output/data/20260930T0115Z_echo_full61_fp8/result.json --publish` 生成。完整原始产物保留在 `experiments/deepseek_v32_echo_prefill/output/data/20260930T0115Z_echo_full61_fp8/`；原始 trace 位于对应 `output/profile/20260930T0115Z_echo_full61_fp8/`。
<!-- END ECHO GENERATED RESULTS -->

## Attention / offload 拆分

下表来自同一 run 的两份 Nsight Systems **annotated extend**，是 61 层实际 GPU
kernel duration 之和，单位 ms。与上方 CUDA-event 区间和无插桩端到端时间分别呈现；
各设备计算、CPU API 及传输可以重叠，不将这些数字相加解释端到端 wall time。

| GPU 工作 | Resident | Offload | 范围 |
| --- | ---: | ---: | --- |
| Attention/indexer 投影 | 120.975 | 121.081 | Q/K/V、RoPE、Hadamard、量化等 |
| Sparse MLA | 453.055 | 441.590 | Attention 核心，61 / 484 次 launch |
| Attention 输出投影 | 40.093 | 40.182 | MLA 输出恢复与输出投影 |
| Native indexer / fused indexer+prefetch | 67.438 | 140.961 | 61 次 native kernel；融合预取无法独立计时 |
| Exact top-k scope | 81.821 | 81.849 | 精确选择及该 scope 内辅助 kernel |
| Offload preparation | 0 | 19.986 | 当前 chunk staging、LRU/映射与预留 slots |
| Exact recall scope | 0 | 209.811 | 去重、容量判断、驱逐、补取与 remap |
| 其中：mapped-host KV gather | 0 | 43.658 | 已包含在前两行 offload scope 中，不能再相加 |
| 整模型全部 kernel | 1135.785 | 1437.772 | 包含 MoE、norm 等及少量范围外输入/输出检查 |

Offload 的 preparation / recall CUDA-event 区间分别为 **152.403 / 976.584 ms**。
这些区间包含提交间隙和依赖等待，不能把 976.584 ms 全部归因于 KV 传输。实际 gather
为 preparation 的 1.589 ms 加 exact recall 的 42.069 ms。全部 H2D KV 还包含融合
kernel 内预取，实际搬运量仍使用 cache counter，不使用 CUDA Memcpy 事件估算。

Offload trace 中有 75,893 次 kernel（resident 为 19,205 次），907 次 exact recall
尝试对应 484 个可装入 HBM pool 的 MLA 批次，满足 `907 = 2 × 484 − 61`。完整
1K query 的选择并集超过 16K slots 时才拆分；本次最小叶批次为 16 query，没有
裁剪单 query 的 top-2048。额外 423 个拼接 kernel 合计 8.813 ms。

`cudaStreamSynchronize` 从 resident 的 4 次 / 0.028 ms 增至 offload 的
8,907 次 / 713.430 ms。该 API 时间包含等待 GPU 的时间，并与 GPU 工作重叠；
它记录缓存控制相关的等待，不能直接确定当前 baseline 问题的根因，也不代表可从
wall time 扣除的独立 CPU 时间。当前 chunk 先写 host 再回读 HBM、每次预取前
预留至多 8192 slots，以及递归失败节点重复执行 GPU unique/sort，是 cache 策略
复核时应检查的实现行为；本轮未验证哪一项构成问题或其修正效果。

两份 trace 各有 12 个范围外活动，合计仅 0.035 / 0.036 ms，均有 API correlation，
属于输入检查、输出检查或小拷贝，已计入 capture 总量。Annotated wall 为
1186.285 / 2550.058 ms；offload 插桩开销约 14.9%，原报告的无插桩延迟为
1171.554 / 2220.250 ms，现不用于 baseline 性能结论。逐阶段表和指标来源见 [nsys_stages.csv](report/nsys_stages.csv)
及 [profile_summary.json](report/profile_summary.json)。

## 真实第 30 层的 NCU 诊断

三个独立 run 分别为 `20260930T0151Z_ncu_indexer_resident`、
`20260930T0149Z_ncu_indexer_offload`、`20260930T0153Z_ncu_mla`。它们读取完整模型
run 捕获的第 30 层输入，在物理 GPU 6 上预热 2 次，然后对一次目标 launch 收集
`full + PmSampling + PmSampling_WarpStates` 和独立 `SourceCounters`；NCU 2026.1.1
没有 `source` set。全部 kernel 保留行号，使用 kernel replay、`cache-control all`、
`clock-control none`；表中 duration 是 profiler replay 数据，不能替代完整模型计时。

Indexer 输入为 FP8 Q `[1024,64,128]`、K `[66560,128]`；MLA 为 BF16 Q
`[1024,128,576]`、KV `[66560,576]`、int32 selection `[1024,2048]`。Offload replay
采用**冷历史 HBM pool**和零 histogram offset，实际预取 8192 records / 9 MiB；
它没有重建原模型 cache hit 状态。MLA replay 使用完整 resident KV 和逻辑选择，
不代表原 offload 物理 pool 布局。独立 `recall` replay 入口本次未采集。

| NCU 指标 | Resident indexer | Fused indexer/prefetch | Sparse MLA |
| --- | ---: | ---: | ---: |
| Duration (ms) | 1.123 | 1.854 | 7.439 |
| SM throughput (% peak) | 51.81 | 38.06 | 20.50 |
| DRAM throughput (% peak) | 5.37 | 3.83 | 1.56 |
| Tensor pipe active (% elapsed) | 51.81 | 30.57 | 12.11 |
| Achieved / theoretical occupancy (%) | 14.06 / 18.75 | 26.56 / 31.25 | 12.49 / 12.50 |
| Registers/thread | 112 | 96 | 163 |
| Grid blocks / threads per block | 132 / 384 | 132 / 640 | 8192 / 256 |
| Waves/SM | 1.00 | 1.00 | 62.06 |
| L1 / L2 hit rate (%) | 63.83 / 88.56 | 90.41 / 96.67 | 5.47 / 96.58 |
| Local spilling requests | 0 | 14,507,576 | 0 |

精确 metric 名、单位、native report SHA、PC 热点和 PM 采样摘要保存在
[ncu_metrics.csv](report/ncu_metrics.csv) 与 [profile_summary.json](report/profile_summary.json)。
这些数字来自 `ncu_report` API，未将不存在的 metric 当作零。

融合 indexer 的 long-scoreboard 占平均发射间隔约 51.5%，同时 DRAM 利用率仅
3.83%，不能称为 HBM 带宽饱和。SourceCounters 中，CUTLASS `barrier.h:424` 的
`mbarrier.try_wait` 对应 62,127 个 long-scoreboard samples；
`echo_logits.cuh:906` 的候选 logits 读取为 5,494 samples。加上 14.51M 次 spilling
requests，作为融合后的同步、寄存器压力与访存延迟的核查线索，尚不确定 baseline
问题的根因。NCU 对 local-memory
开销给出的 kernel 层估计改善空间为 20.34%，不是已实现收益或端到端加速预测。

MLA 的 grid 足够大，62.06 waves/SM；每 SM 仅一个 block，约 149.5 KB 动态 shared
memory 和 163 registers/thread 将 occupancy 限制为 12.5%。L1/TEX throughput 为
67.85%，DRAM 仅 1.56%；源码 `deepseek_mla.py:75–76` 的 KV load 合计 177,245 个
long-scoreboard samples，`:79` 的 QK dot 对应 113,971 个 short-scoreboard samples。
应先评估稀疏 KV staging、shared-memory 布局及 pipeline，而非直接认定是 HBM 带宽瓶颈。

负载与时间序列也不同：两个 indexer 都是 132 个 persistent CTA，resident 的
每 SM active cycles 最小/平均比为 0.771，fused 为 0.819，末段存在不均衡；MLA
该比值为 0.992。PM tensor 活跃率的中段采样中，resident 约 54%，MLA 约 12.2%；
fused 有明显中段下降后恢复。PM 来自多次 replay、不同 metric 采样序列，保留原始
correlation IDs 和样本顺序，不用这些曲线推导单次执行的精确 prefetch overlap。

下一步应先复核 DeepSeek/ECHO 的实现效率、MFU 口径和 cache 策略，明确 baseline
验收标准，再决定具体优化次序。cache ensure/unique、同步、fused indexer 的
spilling/barrier 和 MLA 资源占用保留为候选核查项；本轮未定位根因、未修复或复测。

## 运行

从仓库根目录，在已准备好基础环境及共享 CUTLASS 的 Hopper 上执行：

```bash
bash experiments/deepseek_v32_echo_prefill/scripts/run.sh --help
bash experiments/deepseek_v32_echo_prefill/scripts/run.sh \
  --model /mnt/user-ssd/chenkaiqi/DeepSeek-V3.2 --devices 0,1,2,6,7 \
  --prefix 65536 --extend 1024 --chunk-size 1024 --slots 16384 \
  --warmups 1 --prefill-repeats 1 --repeats 5 --chrome-trace

# Nsight Systems：分别捕获 resident/offload 的 annotated extend
ECHO_NSYS=1 bash experiments/deepseek_v32_echo_prefill/scripts/run.sh \
  --devices 0,1,2,6,7 --save-kernel-inputs

# 真实模型第 30 层激活上的单 kernel Nsight Compute 诊断
CUDA_VISIBLE_DEVICES=6 bash experiments/deepseek_v32_echo_prefill/scripts/ncu.sh \
  --input experiments/deepseek_v32_echo_prefill/output/data/<run_id>/kernel_inputs_layer_30.pt \
  --kernel indexer-offload --run-id <ncu_run_id>
```

调用模块：`models.deepseek_v32.echo_infer`、`echo_block`、`echo_attention`、
`echo_model`；`cache.sparse_token_cache`；`operators.deepseek_v32.indexer.echo`、
`operators.deepseek_v32.attention.device_only.mla`、`operators.deepseek_v32.attention.offload.mla`、
`operators.deepseek_v32.linear.fp8`、`operators.common.kv_transfer`；共享 `GR.input_generator`。

运行参数、输入、完整结果、源码 SHA256 在 `output/data/<run_id>/`，stdout/stderr
分别在 `output/log/<run_id>/`，Chrome/Nsight 原始 trace 在 `output/profile/<run_id>/`。
运行脚本先在 `${TMPDIR:-/tmp}` 暂存，全部成功后才发布 data/log/profile；失败退出码
原样保留，诊断目录打印到 stderr，正式 `output/` 不留下失败 run。运行脚本拒绝已有
run ID，避免覆盖数据与日志；完整模型 run ID 用 `ECHO_RUN_ID` 指定。
下方命令从有效 run 生成 `report/`
数据与图表，并更新本页的结果区段。

`src/report.py --result output/data/<run_id>/result.json` 在同一 data 目录生成汇总和图表，
`--publish --report-dir experiments/deepseek_v32_echo_prefill/report` 发布精选素材；实际
命令须使用从仓库根目录起的完整相对路径。图表生成使用已有 `analysis` 依赖组。
Nsight 导出使用 `nsys export --type sqlite --output <data_path>/<mode>.sqlite
<profile_path>/extend.<1或2>.nsys-rep`，随后调用
`python -m experiments.deepseek_v32_echo_prefill.src.analyze_nsys --sqlite <sqlite>
--output <analysis.json> --result <result.json> --mode <resident或offload>`。

NCU 原始报告用 `python -m experiments.deepseek_v32_echo_prefill.src.analyze_ncu
--report <full.ncu-rep> --report <source.ncu-rep> --run-id <ncu_run_id>` 提取，默认写入
对应 `output/data/<ncu_run_id>/ncu_analysis.json`；再次分析须用 `--output` 选择新的
JSON 路径。API 的逐实例读取经过 PC sample 总和与 aggregate 对照，保留零值及原始
correlation IDs。原精选 profile 数据由以下命令生成，输入来自上述旧 run；这些记录
保留原检查状态，不构成当前 baseline 验收：

```bash
python -m experiments.deepseek_v32_echo_prefill.src.profile_summary \
  --full-run-id 20260930T0115Z_echo_full61_fp8 \
  --ncu-run-id 20260930T0151Z_ncu_indexer_resident \
  --ncu-run-id 20260930T0149Z_ncu_indexer_offload \
  --ncu-run-id 20260930T0153Z_ncu_mla --publish
```

NCU 的 `indexer-resident`、`indexer-offload`、`mla`、`recall` 入口均读取捕获激活；
其中 offload 使用冷 HBM pool，recall 为独立 cold-union 传输诊断，不等同于整模型的
残余搬运时间。整模型搬运量来自 cache 计数，mapped host 读取不能通过 CUDA Memcpy
事件缺失解释为零传输。融合 indexer/prefetch 时间作为整体呈现，不拆成可相加的两项。
