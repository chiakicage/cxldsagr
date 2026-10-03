# NOSA sparse fetch 与 attention overlap

2026-10-01 目录整理：算子已迁至 `operators/nosa/`、`operators/deepseek_v32/` 和
`operators/common/`。本页性能仍对应下文原 run ID 与源码快照；目录迁移后的性能未重新测量。

本实验在真实 NOSA 单层输入上，比较 resident attention、一次取齐稀疏并集后执行完整
attention 的串行对照，以及在同一 CUDA 主 kernel 中执行 sparse fetch 与 attention
的融合实现。目标形状为 65536-token prefix + 1024-token suffix，BF16，32 Q heads /
2 KV heads / D128，64-token block / 64-block budget。

**状态：2026-09-30 已完成 stripe8 融合优化及 KDA 独立验收。** L0/L15/L31 的
copy/softmax overlap 中位数为 **95.09% / 95.34% / 94.94%**，全部 9 个 profile
样本均超过 90%，最低 **92.20%**。20 次无 profiler 主测中，融合总调用延迟较重新
测量的整批稀疏并集串行对照下降 **23.80% / 23.30% / 19.87%**；40 次独立复测
确认收益。完整 32 层的独立 resident/offload 64K + 1K 检查逐位一致。

## 实现与读取语义

实现位于 [SM90 offload 算子](../../operators/nosa/attention/offload/api.py)和
[融合 CUDA kernel](../../operators/nosa/attention/offload/csrc/nosa_offload_fused.cu)。GPU first-use planner
对 `(KV head, logical block)` 去重，一个 256-thread compactor 生成精确选页队列。
每个 64-token 页分为 **8 个互不重叠的 8-token stripe**。主 kernel 中所有 CTA 都
执行 persistent FA3；默认最多 96 个 CTA 同时使用 producer warpgroup 中的
warp 1–3（96 个线程）原子领取 `(page, stripe)`。warp 0 保留 TMA 工作，两个
consumer warpgroups 保留 attention 计算。

仅在 2 KV heads、256 个八-query/head batch 的几何下，fetch 和 compute 都按
head 1 → head 0 执行；head 内 compute 仍使用原有 cost 降序与 tie 顺序，fetch 则
按 page 0 优先、其余 block 降序。各 batch 内的 attention 运算与页顺序不变。
其他几何保留原 block-major fetch 与 native scheduler。
页内并行降低首次可消费页的等待时间，head 顺序对齐让已取回数据更快进入计算。

每个 stripe 只被领取一次，每个历史 16-byte K/V 向量由唯一线程执行一次
`ld.global.cv.v4.u32`，写入 HBM staging。各 query / head 对 KV 的后续复用均来自
staging，不依赖 host memory 的 L2 缓存。每个 writer 完成 store 和 fence 后，
通过 96-thread barrier 汇合，由 leader 执行 acq_rel 页完成计数加一；TMA warp
acquire 等待计数为 8，再经 async-proxy fence 读取 HBM。空尾 stripe 仍参与完成，
不执行 host load，也不计 trace 或 payload。主 kernel 完成后执行原有数值 repair。

native init 将 metadata reset、历史 page 0 padding 与 strided suffix staging
合并为一次 kernel，first-use planner 仍独立执行。串行对照同样使用该初始化优化。

producer / consumer 的动态寄存器上限分别为 24 / 240；启动前检查编译后的 CTA pool
能容纳 `128 × 24 + 256 × 240 = 64512` 个寄存器，并验证 cooperative grid 可同时驻留。
启动属性按 CUDA context ID 与 device 缓存。最终 SM90a main 的资源为 REG168 /
STACK32 / LOCAL0，SASS 仍有 21 处静态 LDL/STL，不能将 LOCAL0 表述为无 spill。
`fetch_ctas` 是参与 fetch 的 CTA 数量上限，不扣除
attention CTA；`query_tile_size` 只定义 first-use 字节直方图，不拆分 attention。
静态读取与同步证据见 [kernel 审计](report/fused/kernel_audit.json)。审计中的 SASS、
源码与构建指纹对应本轮测量；逻辑 read-once 不等于物理 PCIe 字节恰好等于 KV payload。

## 测量环境与结果

硬件为 GPU 1，NVIDIA H20Z，132 SM，SM90；CUDA device UUID
`a2e4882b-3248-6018-14ab-90e45b5a5b04`。Torch 2.12.1+cu130、CUDA runtime 13.0、
NVCC 13.2.86、FlashInfer 0.6.18、Triton 3.7.1、TVM FFI 0.1.13.post3。
最终 fused build key 为 `484e32532ba74fb6`；准确实现身份以报告的源码 SHA256、
构建信息及各 run 的完整源码快照为准。

输入来自历史真实 sparse 模型轨迹
`experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345` 的 L0/L15/L31。
本实验回放这些固定 Q/K/V/CIS/selection，没有把它们描述为新实现重新采集的模型轨迹。
所有 run 都预热 5 次。主性能测量为无 profiler 的 20 次重复；另一个独立进程运行
40 次复测，用于确认收益。Nsight 使用独立进程的 3 次重复，只用于并发分析。
模式顺序逐轮轮换，不剔除慢样本，也不跨层平均。

| 层 | prefix K+V（MiB） | resident（ms） | 串行（ms） | 融合（ms） | 延迟下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L0 | 8.28125 | 0.268240 | 0.472336 | 0.359904 | 23.80% |
| L15 | 12.43750 | 0.256752 | 0.544144 | 0.417376 | 23.30% |
| L31 | 12.46875 | 0.262048 | 0.542784 | 0.434944 | 19.87% |

40 次独立复测：

| 层 | 串行（ms） | 融合（ms） | 延迟下降 | 同轮融合更快的次数 |
| --- | ---: | ---: | ---: | ---: |
| L0 | 0.460464 | 0.351344 | 23.70% | 40/40 |
| L15 | 0.542016 | 0.415440 | 23.35% | 40/40 |
| L31 | 0.550112 | 0.431776 | 21.51% | 40/40 |

主测和复测中，各层的 CUDA-event 与 wall-time 配对样本分别为 20/20 和 40/40
融合更快；三种模式轮转顺序分组的中位数也均支持改进。以上为本次固定输入的实测，
不保证任意调用或其他形状都取得相同收益。Nsight 会明显扰动总调用时间，不用
profiled latency 计算上述提速。

| 层 | stripe-copy union（μs） | softmax union（μs） | 交集（μs） | overlap 中位数 | 有效 copy-window GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| L0 | 191.104 | 213.088 | 181.728 | 95.09% | 45.44 |
| L15 | 268.928 | 275.040 | 256.384 | 95.34% | 48.50 |
| L31 | 274.048 | 283.680 | 260.192 | 94.94% | 47.71 |

表中为每层 3 个 Nsight 样本的中位数，比例先逐样本计算再汇总。各层最小 overlap
分别为 93.16% / 92.20% / 94.17%；全部 9 个样本的 page-envelope 与 stripe-copy
两套指标都超过 90%。两种窗口的 union 在每个样本中完全相等，没有仅由页 envelope
覆盖的间隙。每个融合样本恰有一个 fused main；9 个串行样本的 fetch/attention
kernel 交集全为零。该指标只统计 copy window 与 softmax 的相交比例，不能称为
完整传输隐藏率。

| 用途 | run ID | 报告素材 |
| --- | --- | --- |
| 主测量，20 次 | `nosa_fused_stripe8_head1_20260930_01` | [JSON](report/fused/report.json) / [CSV](report/fused/comparison.csv) |
| 独立确认，40 次 | `nosa_fused_stripe8_head1_confirm40_20260930_01` | [JSON](report/fused_confirmation/report.json) / [CSV](report/fused_confirmation/comparison.csv) |
| Nsight，3 次 | `nosa_fused_stripe8_head1_nsys_20260930_01` | 上述两份报告共用此独立时间线 |

完整产物位于 `output/data/<run_id>/`、`output/profile/<run_id>/` 和
`output/log/<run_id>/`。JSON/CSV 由 `src/report.py` 从对应 timing/profile run 生成，
核验源码快照、输入、硬件、参考检查及 SQLite 指纹。独立脚本不导入分析/报告代码，
从原 SQLite 重提取全部 27 个 measured ranges，核对 kernel 分类、时间交集和
compactor 的准备工作分类；159,432 条 page→softmax 依赖均在 copy 完成后消费。
三次运行的源码、构建、输入指纹一致，报告重建逐字节相同。

正式审计、复算脚本、SASS 与实际 shared object 保存在
`output/data/nosa_fused_stripe8_head1_20260930_01/audit/`。其中
`archived_artifacts.json` 将审计中原 `/tmp` / build-cache 路径映射至持久副本并记录
SHA256；原审计 JSON 保持不变。已验收的这三次运行替换本实验上一版性能结果。

## 正确性验收

三个 run 都使用 `--reference-all`：三个层的全部 1024 queries 对独立 FP32
QK + CIS / causal softmax / AV 参考通过 BF16 `atol=rtol=0.016` 验收；serialized 与
overlap 输出逐位一致，与 resident 的比较通过相同 BF16 容差验收。
每次 validation、warmup 和 timed iteration 后，GPU 总传输字节与
逐 first-use group 字节都必须等于 CPU 独立去重结果。L0/L15/L31 的精确 prefix K+V
payload 分别为 8,683,520 / 13,041,664 / 13,074,432 B。

最终全局 CPU 回归：1339 passed、674 skipped、34 subtests passed；GPU 环境下的
SM90 offload 算子及 cache/model 专项共 62 passed，覆盖尾块、stripe trace/ownership、
repair、跨 stream 复用、输入生命周期、scratch 边界及缓存事务，其中包含 CPU 检查。
实验工具的 126 项 CPU 测试已包含在全局回归中，不重复计数；跳过不计作通过。

完整 checkpoint 正确性由
[test_offload_checkpoint.py](../../models/nosa/tests/test_offload_checkpoint.py) 单独检查：
完整 32 层分别从独立空 resident/offload cache 构建 64K prefix，再执行 1K extend。
最终 normalized hidden states 逐位一致，`max_abs=0`。提交后的 cache 分配为 resident
HBM 2,262,627,840 B、offload HBM 150,825,168 B、offload host 2,181,038,080 B；这不包含
模型权重，也不是进程峰值显存。
测试只输出终端，不作为完整模型性能测量，也不将包含加载/编译的测试耗时当作推理延迟。

```bash
CUDA_VISIBLE_DEVICES=1 NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  .venv/bin/python -m pytest models/nosa/tests/test_offload_checkpoint.py \
  -s -q -p no:cacheprovider
```

## 计时边界

| 模式 | prefix K/V | 执行方式 |
| --- | --- | --- |
| `resident` | 已在 HBM | 原始完整 1024-query native FA3 |
| `serialized` | pinned local DRAM | 优化的 128-thread CTAs 一次读取精确稀疏并集，再执行原完整 attention |
| `overlap` | 同一 pinned local DRAM | 同一主 kernel 内由空闲 producer warps fetch、所有 CTA compute |

串行与融合路径使用相同 `.cv` host-load 策略和相同精确 KV payload；复制调度各自优化。
每次调用都重置 prefix fetch 状态。suffix K/V 是 GPU 新生成的 1024 tokens，不计入
prefix host 流量，但 suffix staging 计入调用时间。工作区和 scratch 可跨调用复用，
每次返回独立输出；不保留已取回 prefix 作为下一次测量的暖缓存。

计时包括状态重置、first-use planning、队列 compaction（融合路径）、suffix staging、
host fetch、完整-query FA3 prepare/sort/main/repair、输出分配调用与完成依赖。
CUDA end event 位于 caller stream 并同步等待，event 区间可能包含主机提交间隙，
不是纯 device-kernel 时间；wall time 还覆盖 completion synchronization，
submit time 单独记录。排除输入读取/初始化、host pinning/registration、KV 写回主存、
indexer/CIS 计算、编译、预热、正确性检查与 trace 导出。全部原始样本和 median/min/max
保存在 run 中，报告保留相应分布。

这是单层 attention 输入回放，不是完整 32 层或 serving 性能测量。HBM staging 仍按
完整逻辑 NHD 长度预留一层地址空间，没有实现有限 slots / eviction。传输速率为本机
映射 pinned DRAM 的实际速率，没有强制限速到此前分析假设的 50 GB/s；不等同于已验证
CXL/RDMA 设备路径。

## Overlap 定义

Nsight 按 NVTX measured range 及 CUDA launch correlation 验证：每个融合样本恰好
一个 fused main、没有独立 host-fetch kernel；队列 compactor 属于准备工作。
它不以单个融合 kernel 的持续时间推断内部重叠。

带 `--profile` 的运行额外记录 device `%globaltimer`：每个非空 stripe 的 copy
window 从领取/解码完成后的 fetch-thread 同步前开始，到 stores、per-thread fences
和 fetch-thread barrier 完成为止，不包括后续 counter/ready publication。每页另记录
其所有非空 stripe 的最早 start / 最晚 end envelope；envelope 可能包含 stripe
间隙，因此必须独立核验 stripe 窗口。softmax window 包围 consumer 的 softmax
update，排除 ready polling 和 QK/PV MMA。二者使用同一 device clock，不与 Nsight
绝对时钟直接混合。

```text
F = union(nonempty stripe-copy windows)
P = union(per-page min(start)/max(end) envelopes)
M = union(instrumented softmax-update windows)
fetch_stripe_math_overlap_fraction = duration(F ∩ M) / duration(F)
fetch_math_overlap_fraction = duration(P ∩ M) / duration(P)
```

schema 3 按每个样本核验完整 stripe 身份、字节数、时间戳和 page envelope，两套
ratio 均须 ≥0.9；中位数通过不能替代全部样本通过。每个预期的非空 `(page, stripe)`
必须恰好出现一次。page ratio 减 stripe ratio 可以为正或负，不能假定一方单向界定
另一方。本次全部样本二者相等。

该比例只衡量 copy window 与 softmax 的相交，不是完整 attention 隐藏率、PCIe
占用率或端到端提速比例。报告的有效 fetch-window GB/s 使用逻辑选中 K+V 字节
除以 P 的长度，本次每个样本 P = F；它不证明物理链路字节数。
真正的性能收益单独由无 profiler 的总调用延迟
比较决定。

## 复现

从仓库根目录运行，要求项目 `.venv`、SM90/Hopper、native FA3 依赖及 mapped pinned
host memory 支持。脚本创建独立 run ID，stdout/stderr 分开，保留失败退出码；失败诊断
留在 `/tmp`，已有 run 不覆盖。

```bash
bash experiments/nosa_offload_overlap/scripts/run.sh --help

CUDA_VISIBLE_DEVICES=1 bash experiments/nosa_offload_overlap/scripts/run.sh nosa_fused_stripe8_head1_20260930_01 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 20 --reference-all

CUDA_VISIBLE_DEVICES=1 bash experiments/nosa_offload_overlap/scripts/run.sh nosa_fused_stripe8_head1_confirm40_20260930_01 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 40 --reference-all

CUDA_VISIBLE_DEVICES=1 bash experiments/nosa_offload_overlap/scripts/run.sh nosa_fused_stripe8_head1_nsys_20260930_01 --profile \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 3 --reference-all
```

复现时改用新的 run ID；生成已有报告的命令如下，复查时也需选择新的输出目录。

```bash
python -m experiments.nosa_offload_overlap.src.report \
  --measurement-dir experiments/nosa_offload_overlap/output/data/nosa_fused_stripe8_head1_20260930_01 \
  --profile-dir experiments/nosa_offload_overlap/output/data/nosa_fused_stripe8_head1_nsys_20260930_01 \
  --output-dir experiments/nosa_offload_overlap/report/fused

python -m experiments.nosa_offload_overlap.src.report \
  --measurement-dir experiments/nosa_offload_overlap/output/data/nosa_fused_stripe8_head1_confirm40_20260930_01 \
  --profile-dir experiments/nosa_offload_overlap/output/data/nosa_fused_stripe8_head1_nsys_20260930_01 \
  --output-dir experiments/nosa_offload_overlap/report/fused_confirmation
```

调用模块：`operators.nosa.attention.offload.api.NosaFetchWorkspace`、
`operators.nosa.attention.offload._fused`、`operators.nosa.attention.device_only.api`、
`layers.attention.BlockSelection`。输入加载及 fingerprint 显式复用
`experiments.nosa_kernel_mfu.src.capture_inputs` / `src.measure`，源码快照复用
`experiments.nosa_gr_65536_1024.src.sources`。`--synthetic` 是合成激活和选择，不能作为真实
请求结果。此前 resident 实验保留原测量身份，没有改写为本 offload 实验的结果。
