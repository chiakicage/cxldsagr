# NOSA sparse fetch 与 attention overlap

本目录归入自有设计的 microbench，回放真实 NOSA 单层输入，比较 resident attention、
一次取齐稀疏并集后执行完整 attention 的串行对照，以及在同一 CUDA 主 kernel 中执行 sparse fetch 与 attention
的融合实现。形状为 65536-token prefix + 1024-token suffix，BF16，32 Q heads /
2 KV heads / D128，64-token block / 64-block budget。

下文保留 2026-10-04 的有效测量。整理后的独立 check / bench / profile 入口尚未运行
CUDA 验收或性能实验，本次没有新增实验结果。该轮 L0/L15/L31 的融合完整 API 延迟较
同轮串行对照
下降 **29.34% / 24.53% / 22.71%**，40 次独立复测确认收益。全部 9 个 profile 样本的
page-envelope 与 stripe-copy 两套 overlap ratio 都达到 90%，最低为 **90.62%**。
该轮源码的完整 32 层 resident/offload 64K + 1K 数值检查也已通过，全部 extend
normalized hidden 逐位一致。

这组测量使用默认无 cache tag 调用，每次重取完整稀疏并集；不测量固定 P/NH 缓存命中
或 serving 性能。输入仍为历史冻结的真实算子输入，没有重新采集当前模型轨迹。
该轮结论来自同轮 run 内的串行/融合配对比较；不同 GPU、工具链和日期之间的差异不能
单独归因为代码提速。

## 独立验收与性能入口

默认 `--mode bench` 只预热、计时，要求提供独立 `check` 生成的匹配 receipt。
`check` 比较全部 query 的 FP32 参考输出、resident/serialized/overlap 输出，检查
串行与融合输出逐位一致，并在重复冷态调用中核对 CPU 稀疏并集与 GPU 总量、分 tile
计数。它不生成性能样本。`--profile` 单独采集 nsys 与 kernel 内部区间，并保留诊断
流量检查；正式 bench 的样本之间不再读取诊断计数或比较输出。

Receipt 绑定源码、native 构建信息、软件环境、设备、后端、输入内容及布局、dtype、
fetch CTA 数、tile 统计宽度和冷态 cache 路径。性能结果保存 receipt 副本，报告生成器
核对其配置及各 case 的数值依据。改变覆盖条件后重做验收，改变重复次数不需要重做。
算子 API 内的 finite 检查、repair、ready 协议和异步完成等待仍属于实际执行成本。
本 receipt 只验收所测单层输入；完整 32 层模型数值检查仍为独立验收。

```bash
nosa_inputs=experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345
bash experiments/nosa_offload_overlap/scripts/run.sh overlap_check \
  --mode check --input-dir "$nosa_inputs" --validation-receipt /tmp/nosa-overlap-check.json
bash experiments/nosa_offload_overlap/scripts/run.sh overlap_bench \
  --input-dir "$nosa_inputs" --validation-receipt /tmp/nosa-overlap-check.json
bash experiments/nosa_offload_overlap/scripts/run.sh overlap_profile \
  --profile --input-dir "$nosa_inputs" --validation-receipt /tmp/nosa-overlap-check.json
```

`check` 的日志和数值证据留在系统临时目录，脚本完成后打印位置；以上显式 receipt
位于 `/tmp/`。`bench/profile` 保持 `output/data|log|profile/<run_id>/` 布局。报告同时
支持原有内置检查记录和新 receipt，不将旧报告当作新入口的实测结果。下文旧运行命令
保留原样，新运行应采用本节的分阶段入口。

## 实现与读取语义

实现位于 [SM90 offload 算子](../../operators/nosa/attention/offload/api.py)和
[融合 CUDA kernel](../../operators/nosa/attention/offload/csrc/nosa_offload_fused.cu)。GPU first-use
planner 对 `(KV head, logical block)` 去重，256-thread compactor 生成精确选页队列。
每个 64-token 页分为 8 个不重叠的 8-token stripe。主 kernel 的所有 CTA 都执行
persistent FA3；最多 96 个 CTA 使用 producer warpgroup 的 warp 1–3（96 个线程）
原子领取 `(page, stripe)`。warp 0 保留 TMA，两个 consumer warpgroups 保留 attention。

仅在 2 KV heads、256 个八-query/head batch 的几何下，fetch 和 compute 都按
head 1 → head 0 执行；head 内 compute 保留 cost 降序与 tie 顺序，fetch 按 page 0
优先、其余 block 降序。各 batch 内的 attention 运算与页顺序不变；其他几何保留
block-major fetch 与原 native scheduler。

每个 stripe 只领取一次，每个历史 16-byte K/V 向量由唯一线程执行一次
`ld.global.cv.v4.u32`，写入 HBM staging。后续 query/head 复用来自 staging。
writer 完成 store 和 fence 后通过 96-thread barrier 汇合，leader 以 acq_rel RMW
累计页完成计数；TMA acquire 等待计数为 8，再经 async-proxy fence 读取 HBM。
空尾 stripe 仍参与完成，不读 host，也不计 trace 或 payload。最后完成的 stripe
只累计一次整页字节。主 kernel 后保留原数值 repair。

native init 合并 metadata reset、历史 page 0 padding 和 strided suffix staging；
first-use planner 独立执行。串行对照也使用该初始化。`fetch_ctas` 只限制参与 fetch
的 CTA 数量，所有 CTA 继续计算；`query_tile_size` 仅定义 first-use 字节直方图。

当前 SM90a 主 kernel 的 cuobjdump 资源为 REG168 / STACK32 / SHARED1024 / LOCAL0，
新 SASS 有 21 处静态 LDL/STL，不能据 LOCAL0 声称无 spill。producer/consumer 动态
寄存器上限为 24/240，`128 × 24 + 256 × 240 = 64512` 恰好等于编译后的 CTA pool。
启动前核验此容量及 cooperative grid 可同时驻留。实际 Nsight/CUPTI 记录为
132 CTAs × 384 threads，dynamic shared memory 197,728 B，static shared memory 0 B。
cuobjdump 的 SHARED 与 CUPTI 的 static 字段分别保留，不合并或视作相同字段。

当前二进制、源码、SASS PC、读取和同步证据见 [kernel 审计](report/fused/kernel_audit.json)。
审计将构建 key 与运行前后未变的缓存二进制 SHA256/inode/mtime 对应；本轮没有采集
进程 `/proc/maps`。静态 ownership 枚举与实际 profile 唯一 stripe 验收分别记录。
逻辑 read-once 不等于物理 PCIe 字节恰好等于 KV payload。

## 测量环境与结果

使用物理 GPU 5，UUID `GPU-a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`，SM90、132 SM，
总显存 150,121,545,728 B。CUDA 设备名称为 NVIDIA H200，NVML 名称为 NVIDIA M403；
以 UUID 标识本次设备。CPU 绑定 48–55，host memory 绑定 NUMA node 1。
Torch 2.12.1+cu130、CUDA runtime 13.0、NVCC 13.2.78、FlashInfer 0.6.18、
Triton 3.7.1、TVM FFI 0.1.13.post3。

fused build key 为 `d34d3d8f463d5b26`，二进制 SHA256 为
`43f624f9d25a0fe7f0299e42a4e6d7ba840a2fb356fbd1fa14a66cb6b08ec05d`。
运行前后冻结的 runtime 源码 digest 为
`02b5d0f9589c5e49257b40b73a5ddce4b2811f0614cc79d4afe242a2aa09f08b`；
完整源码快照、文件哈希和构建信息随各 run 保存。

输入来自 `experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345`
的 L0/L15/L31，固定回放 Q/K/V/CIS/selection。所有 run 预热 5 次；主测为无 profiler
的 20 次重复，另一独立进程复测 40 次，Nsight 独立进程重复 3 次。模式顺序逐轮轮换，
不剔除慢样本，不跨层平均。下表为完整 CUDA-event API 时间的中位数。

| 层 | prefix K+V（MiB） | resident（ms） | 串行（ms） | 融合（ms） | 延迟下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L0 | 8.28125 | 0.239424 | 0.476944 | 0.336992 | 29.34% |
| L15 | 12.43750 | 0.238208 | 0.571936 | 0.431664 | 24.53% |
| L31 | 12.46875 | 0.235264 | 0.571680 | 0.441840 | 22.71% |

40 次独立复测：

| 层 | 串行（ms） | 融合（ms） | 延迟下降 | 同轮融合更快的次数 |
| --- | ---: | ---: | ---: | ---: |
| L0 | 0.476992 | 0.338320 | 29.07% | 40/40 |
| L15 | 0.573456 | 0.435920 | 23.98% | 40/40 |
| L31 | 0.571680 | 0.441120 | 22.84% | 40/40 |

主测和复测中，各层 CUDA-event 与 wall-time 配对样本分别为 20/20、40/40 融合更快。
该结论限于本次输入、设备和调用语义；Nsight 的总调用时间不用于计算提速。

| 层 | stripe-copy union（μs） | softmax union（μs） | 交集（μs） | overlap 中位数 | 最小 overlap | 有效 copy-window GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 215.200 | 218.176 | 199.872 | 93.70% | 92.88% | 40.35 |
| L15 | 314.112 | 302.944 | 286.720 | 91.28% | 90.81% | 41.52 |
| L31 | 314.720 | 309.088 | 287.264 | 91.28% | 90.62% | 41.54 |

profile 表中各项分别取 3 个样本的中位数，比例先逐样本计算再汇总；最小值逐层单列。
全部 9 个样本的 page-envelope 与 stripe-copy 两套 ratio 均 ≥0.9，二者 union 在
每个样本中相等，envelope-only union 为 0。每个融合样本只有一个 fused main；
9 个串行样本的 fetch/attention kernel 交集均为 0。该比例衡量 copy window 与
softmax 的相交，不能作为完整传输隐藏率。

| 用途 | run ID | 报告素材 |
| --- | --- | --- |
| 主测量，20 次 | `nosa_cached_fetch_20261004_01` | [JSON](report/fused/report.json) / [CSV](report/fused/comparison.csv) |
| 独立确认，40 次 | `nosa_cached_fetch_confirm40_20261004_01` | [JSON](report/fused_confirmation/report.json) / [CSV](report/fused_confirmation/comparison.csv) |
| Nsight，3 次 | `nosa_cached_fetch_profile_20261004_01` | 上述两份报告共用此独立时间线 |

完整产物位于 `output/data/<run_id>/`、`output/profile/<run_id>/` 和
`output/log/<run_id>/`。JSON/CSV 由 `src/report.py` 从新 timing/profile run 重建，
核验源码、输入、硬件、参考检查和 SQLite 指纹。独立原始数据审计完成 657 项 timing
检查及全部 27 个 measured ranges 的 profile 核验，覆盖 stripe 身份、字节、页 envelope、
内部区间和串行 launch 分类。审计收据、脚本、启动环境及二进制副本保存在
`output/data/nosa_cached_fetch_20261004_01/audit/`。

## 正确性验收

三次运行均使用 `--reference-all`。每层全部 1024 queries 对独立 FP32
QK + CIS / causal softmax / AV 参考通过 BF16 `atol=rtol=0.016` 验收；serialized
与 overlap 输出逐位一致，与 resident 的比较通过相同容差验收。
每次 validation、warmup、timed iteration 后，GPU 总传输字节和逐 first-use group
字节都须等于 CPU 独立去重结果。L0/L15/L31 的 prefix K+V payload 分别为
8,683,520 / 13,041,664 / 13,074,432 B。

2026-10-04 在同一冻结 runtime 源码下单独运行
[test_offload_checkpoint.py](../../models/nosa/tests/test_offload_checkpoint.py) 的
`test_cuda_offloaded_checkpoint_matches_independent_resident_64k_1k`，1 项通过。
完整 32 层分别从独立空 resident/offload cache 构建 64K prefix，再执行 1K extend；
全部 normalized hidden 逐位一致，`max_abs=0`。请求取自
`experiments/indexer_block_sparse_profile/output/data/sparse_flags_native_20261004_01/request.json`，
SHA256 为 `65d62f9ee9662fbe4b1d95d8c5440f16e70336a10b1eb2c860c55d8af3176b6b`。

提交后的 cache 分配为 resident HBM 2,262,627,840 B、offload HBM 150,825,168 B、
offload host 2,181,038,080 B。这些分配不含模型权重，也不是进程峰值显存。
该检查使用普通 owned-cache A1024 路径；它不替代固定 P/NH serving 的验收，测试耗时
也不作为模型推理延迟。测试结果只输出终端，不保存为实验报告。

## 计时边界

| 模式 | prefix K/V | 执行方式 |
| --- | --- | --- |
| `resident` | 已在 HBM | 原完整 1024-query native FA3 |
| `serialized` | pinned local DRAM | 128-thread CTAs 一次读取精确稀疏并集，再执行完整 attention |
| `overlap` | 同一 pinned local DRAM | 同一主 kernel 中 producer warps fetch，全部 CTA compute |

串行和融合使用相同 `.cv` host-load 策略及精确 KV payload，复制调度各自优化。
`NosaFetchWorkspace.run` 未传 `cache_tags/cache_owner`，默认 prepare 每次重置 prefix
fetch 状态；不跨调用保留暖 prefix。suffix K/V 是 GPU 上的 1024 个新 token，不计入
host payload，但 suffix staging 计时。工作区和 scratch 可复用，每次返回独立输出。

计时包括状态重置、first-use planning、融合队列 compaction、suffix staging、host fetch、
完整-query FA3 prepare/sort/main/repair、输出分配调用和完成依赖。CUDA end event
位于 caller stream 并同步等待，event 区间可能含主机提交间隙；wall time 还覆盖
completion synchronization，submit time 单独记录。排除输入读取/初始化、host
pinning/registration、KV 写回主存、indexer/CIS 计算、编译、预热、正确性检查和 trace
导出。全部原始样本及分布随 run 保存。

本实验是一层算子回放，不测完整 32 层或 serving 延迟。本调用按完整逻辑 NHD 长度
预留一层 HBM staging，没有评估有限 P/NH 的命中、准入或淘汰。传输来自本机 mapped
pinned DRAM，没有强制限速到 50 GB/s，也没有验证 CXL/RDMA 路径。

## Overlap 定义

Nsight 按 NVTX measured range 与 CUDA launch correlation 核验每个融合样本恰好
一个 fused main、没有独立 host-fetch kernel；队列 compactor 计为准备工作。
融合主 kernel 的完整执行窗口不能代替内部工作区间。

`--profile` 额外记录 device `%globaltimer`。每个非空 stripe 的 copy window 在
任务领取和解码后的 barrier 完成后，由 leader 记录 start，再经一次 fetch-thread
barrier 开始 copy；end 位于 stores、per-thread fences 和完成 barrier 之后，
不包括后续 counter/ready publication。每页 envelope 为全部非空 stripe 的
min(start)/max(end)。softmax window 包围 consumer 的 softmax update，排除
ready polling 与 QK/PV MMA。二者使用同一 device clock，不与 Nsight 绝对时钟混合。

```text
F = union(nonempty stripe-copy windows)
P = union(per-page min(start)/max(end) envelopes)
M = union(instrumented softmax-update windows)
fetch_stripe_math_overlap_fraction = duration(F ∩ M) / duration(F)
fetch_math_overlap_fraction = duration(P ∩ M) / duration(P)
```

schema 3 逐样本核验完整 stripe 身份、字节、时间戳和 page envelope，每个预期非空
`(page, stripe)` 恰好出现一次；两套 ratio 均须 ≥0.9。page ratio 与 stripe ratio
之差可以为正或负；本次各样本均相等。有效 copy-window GB/s 是逻辑 K+V payload
除以 P 的长度，本次 P = F；该数值不是物理链路字节或占用率。
总调用延迟收益由独立无 profiler 测量决定。

## 复现

从仓库根目录运行，要求项目 `.venv`、SM90/Hopper、native FA3 和 mapped pinned
host memory。脚本创建独立 run ID，stdout/stderr 分开，保留失败退出码；失败诊断
保存在实验目录之外，已有 run 不覆盖。下面使用新的 run ID；CPU/NUMA 绑定需与目标
设备拓扑对应。已发布运行的原命令及运行前后环境收据保存在主 run 的 `audit/`。

```bash
bash experiments/nosa_offload_overlap/scripts/run.sh --help

base_path="$(python3 -c 'import os; from pathlib import Path; v=str(Path.cwd()/".venv/bin"); print(":".join(x for x in os.environ["PATH"].split(":") if x != v))')"
nosa_run_env=(env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_HIP_ALLOC_CONF
  -u PYTORCH_NO_CUDA_MEMORY_CACHING -u PYTORCH_NO_HIP_MEMORY_CACHING
  -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES
  PATH="$base_path" CUDA_VISIBLE_DEVICES=5 CXLDSAGR_SM90_BACKEND=native
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  PYTORCH_ALLOC_CONF=backend:native,pinned_use_cuda_host_register:True,pinned_num_register_threads:8
  numactl --physcpubind=48-55 --membind=1)

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh nosa_offload_new_main \
  --device cuda:0 --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 20 --reference-all

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh nosa_offload_new_confirm40 \
  --device cuda:0 --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 40 --reference-all

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh nosa_offload_new_profile --profile \
  --device cuda:0 --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 3 --reference-all
```

本次两份 JSON/CSV 在新 run 的 `report/` 子目录生成，再选入本实验的 `report/fused/`
和 `report/fused_confirmation/`。重建时选用新的输出目录：

```bash
.venv/bin/python -m experiments.nosa_offload_overlap.src.report \
  --measurement-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_20261004_01 \
  --profile-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_profile_20261004_01 \
  --output-dir /tmp/nosa_offload_report_rebuild_main

.venv/bin/python -m experiments.nosa_offload_overlap.src.report \
  --measurement-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_confirm40_20261004_01 \
  --profile-dir experiments/nosa_offload_overlap/output/data/nosa_cached_fetch_profile_20261004_01 \
  --output-dir /tmp/nosa_offload_report_rebuild_confirm40
```

调用模块：`operators.nosa.attention.offload.api.NosaFetchWorkspace`、
`operators.nosa.attention.offload._fused`、`operators.nosa.attention.device_only.api`、
`layers.attention.BlockSelection`。输入加载及 fingerprint 显式复用
`experiments.nosa_kernel_mfu.src.capture_inputs` / `src.measure`，源码快照复用
`experiments.nosa_baseline_performance.src.dense.sources`。`--synthetic` 生成合成激活和选择，不能
作为真实请求结果；resident 实验保留各自测量身份。
