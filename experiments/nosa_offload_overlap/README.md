# NOSA sparse fetch 与 attention overlap

2026-10-05 已完成重构后的独立 check、20 次主测、40 次确认和 3 次 profile。
本页新结果使用各自的 `refactor_nosa_overlap_*_20261005_01` run ID；数值参考与传输
计数验收来自独立 check，正式计时不在样本之间重跑参考计算。

本目录归入自有设计的 microbench，回放真实 NOSA 单层输入，比较 resident attention、
一次取齐稀疏并集后执行完整 attention 的串行对照，以及在同一 CUDA 主 kernel 中
执行 sparse fetch 与 attention 的融合实现。形状为 65536-token prefix + 1024-token suffix，BF16，32 Q heads /
2 KV heads / D128，64-token block / 64-block budget。

本轮 L0/L15/L31 的融合完整 API 延迟比同轮串行对照下降
**29.47% / 24.33% / 23.02%**，40 次独立确认得到同方向结果。全部 9 个 profile
样本的 page-envelope 与 stripe-copy ratio 均达到 90%，最低为 **91.08%**。
同轮另行完成完整 32 层 resident/offload H64K+A1024 数值检查，全部 candidate
normalized hidden 逐位一致。这个单层 A1024 overlap 结果不替代 A128 固定 serving
路径的 overlap 目标；后者见 [motivation](../nosa_motivation/README.md)。

这组测量使用默认无 cache tag 调用，每次重取完整稀疏并集；不测量固定 P/NH 缓存命中
或 serving 性能。输入仍为历史冻结的真实算子输入，没有重新采集当前模型轨迹。
本轮结论来自各 run 内的串行/融合配对比较；不同 GPU、工具链和日期之间的差异不能
单独归因为代码提速。

## 独立验收与性能入口

默认 `--mode bench` 只预热、计时，要求提供独立 `check` 生成的匹配 receipt。
`check` 比较全部 query 的 FP32 参考输出、resident/serialized/overlap 输出，检查
串行与融合输出逐位一致，并在重复冷态调用中核对 CPU 稀疏并集与 GPU 总传输量、分 tile
计数。它不生成性能样本。`--profile` 单独采集 nsys 与 kernel 内部区间，并保留诊断
流量检查；正式 bench 的样本之间不再读取诊断计数或比较输出。

Receipt 绑定源码、native 构建信息、软件环境、设备、后端、输入内容及布局、dtype、
fetch CTA 数、tile 统计宽度和冷态 cache 路径。性能结果保存 receipt 副本，报告生成器
核对其配置及各 case 的数值依据。改变覆盖条件后重做验收，改变重复次数不需要重做。
算子 API 内的 finite 检查、repair、ready 协议和异步完成等待仍属于实际执行成本。
本 receipt 只验收所测单层输入；完整 32 层模型数值检查仍为独立验收。

```bash
nosa_inputs=experiments/nosa_mfu/output/data/kda_inputs_baseline_20260928_1345
bash experiments/nosa_offload_overlap/scripts/run.sh overlap_check \
  --mode check --input-dir "$nosa_inputs" --validation-receipt /tmp/nosa-overlap-check.json
bash experiments/nosa_offload_overlap/scripts/run.sh overlap_bench \
  --input-dir "$nosa_inputs" --validation-receipt /tmp/nosa-overlap-check.json
bash experiments/nosa_offload_overlap/scripts/run.sh overlap_profile \
  --profile --input-dir "$nosa_inputs" --validation-receipt /tmp/nosa-overlap-check.json
```

`check` 的日志和数值证据留在系统临时目录，脚本完成后打印位置；以上显式 receipt
位于 `/tmp/`。`bench/profile` 保持 `output/data|log|profile/<run_id>/` 布局。本轮报告
使用独立 receipt；历史 kernel 审计保留其原始来源，不将历史验收写成本轮重跑。

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

保留的 2026-10-04 SM90a kernel 审计中，cuobjdump 资源为 REG168 / STACK32 / SHARED1024 / LOCAL0，
该轮 SASS 有 21 处静态 LDL/STL，不能据 LOCAL0 声称无 spill。producer/consumer 动态
寄存器上限为 24/240，`128 × 24 + 256 × 240 = 64512` 恰好等于编译后的 CTA pool。
启动前核验此容量及 cooperative grid 可同时驻留。该历史 Nsight/CUPTI 记录为
132 CTAs × 384 threads，dynamic shared memory 197,728 B，static shared memory 0 B。
cuobjdump 的 SHARED 与 CUPTI 的 static 字段分别保留，不合并或视作相同字段。

该历史二进制、源码、SASS PC、读取和同步证据见 [kernel 审计](report/fused/kernel_audit.json)。
审计将构建 key 与原运行前后未变的缓存二进制 SHA256/inode/mtime 对应；该轮没有采集
进程 `/proc/maps`。本次只选用该审计的静态源码、二进制、SASS 与资源字段，保留 12 个原路径文件；
选用范围和逐文件哈希见[静态证据清单](report/fused/kernel_audit_selection.json)。
原审计中的 runtime-binding、动态 profile 和归档字段不用于本轮结论；保留原生成器
只为记录生成来源，不承诺清理后可重跑整个历史审计。当前唯一 stripe 验收由本轮 profile 提供。
逻辑 read-once 不等于物理 PCIe 字节恰好等于 KV payload。

## 测量环境与结果

本轮使用物理 GPU 1，UUID `GPU-a2226185-cb05-a411-80da-f365154128fe`，SM90、132 SM，
总显存 150,121,545,728 B。CUDA 设备名称为 NVIDIA H200，NVML 名称为 NVIDIA M403；
以 UUID 标识本次设备。CPU 绑定 16–23，host memory 绑定 NUMA node 0。
Torch 2.12.1+cu130、CUDA runtime 13.0、NVCC 13.2.78、FlashInfer 0.6.18、
Triton 3.7.1、TVM FFI 0.1.13.post3。

本轮构建信息、native 二进制身份和运行源码快照随各 run 的 metadata 保存。
历史 kernel 审计的 build key `d34d3d8f463d5b26`、二进制 SHA256
`43f624f9d25a0fe7f0299e42a4e6d7ba840a2fb356fbd1fa14a66cb6b08ec05d` 仍属于
2026-10-04 的记录，不作为本轮新测的二进制身份。

输入来自 `experiments/nosa_mfu/output/data/kda_inputs_baseline_20260928_1345`
的 L0/L15/L31，固定回放 Q/K/V/CIS/selection。测量 run 均预热 5 次；主测为无 profiler
的 20 次重复，另一独立进程复测 40 次，Nsight 独立进程重复 3 次。模式顺序逐轮轮换，
不剔除慢样本，不跨层平均。下表为完整 CUDA-event API 时间的中位数。

| 层 | prefix K+V（MiB） | resident（ms） | 串行（ms） | 融合（ms） | 延迟下降 |
| --- | --- | --- | --- | --- | --- |
| L00 | 8.28125 | 0.239152 | 0.476224 | 0.335888 | 29.47% |
| L15 | 12.43750 | 0.235536 | 0.571616 | 0.432560 | 24.33% |
| L31 | 12.46875 | 0.232688 | 0.569168 | 0.438128 | 23.02% |

40 次独立复测：

| 层 | 串行（ms） | 融合（ms） | 延迟下降 | 同轮融合更快的次数 |
| --- | --- | --- | --- | --- |
| L00 | 0.477296 | 0.335920 | 29.62% | 40/40 |
| L15 | 0.572688 | 0.434400 | 24.15% | 40/40 |
| L31 | 0.572464 | 0.441376 | 22.90% | 40/40 |

主测和复测中，各层的 CUDA-event 和 wall-time 两种计时均显示：融合分别在
20/20、40/40 个同轮配对样本中更快。
该结论限于本次输入、设备和调用语义；Nsight 的总调用时间不用于计算提速。

独立 profile 的内部工作区间如下，时间均为三个样本的中位数。每个样本先分别合并
page envelope、非空 stripe-copy 与 softmax-update 区间，再计算交集；表中 ratio
单独取各样本比例的中位数，不用两个中位时间相除。

| 层 | Copy union P=F（µs） | Softmax union（µs） | 交集（µs） | Page / stripe ratio 中位数 | 两种 ratio 最小值 | 逻辑 payload / copy union（GB/s） |
| --- | --- | --- | --- | --- | --- | --- |
| layer_00 | 212.576 | 220.000 | 200.480 | 94.60% | 93.89% | 40.849 |
| layer_15 | 312.320 | 306.880 | 290.496 | 93.13% | 92.44% | 41.757 |
| layer_31 | 313.056 | 308.288 | 287.648 | 91.88% | 91.08% | 41.764 |

[当前 profile 独立审计](report/current_profile_integrity.json)记录全部九个样本的唯一页、
stripe、payload、P=F、交集和 launch correlation。所有样本各有一个 cooperative
fused main，均为 132 CTAs × 384 threads，dynamic shared 197,728 B、static shared 0 B；
这些字段来自本轮 CUPTI。逻辑 payload 换算的 GB/s 不代表物理链路利用率。

| 用途 | run ID | 报告素材 |
| --- | --- | --- |
| 主测量，20 次 | `refactor_nosa_overlap_bench_20261005_01` | [JSON](report/fused/report.json) / [CSV](report/fused/comparison.csv) |
| 独立确认，40 次 | `refactor_nosa_overlap_confirm40_20261005_01` | [JSON](report/fused_confirmation/report.json) / [CSV](report/fused_confirmation/comparison.csv) |
| Nsight，3 次 | `refactor_nosa_overlap_profile_20261005_01` | 上述两份报告共用此独立时间线 |

完整产物位于 `output/data/<run_id>/`、`output/profile/<run_id>/` 和
`output/log/<run_id>/`。JSON/CSV 由 `src/report.py` 分别关联独立的主测/确认与 profile 数据，核验
源码快照、输入、设备、独立 receipt 与 SQLite 指纹。[发布来源](report/publication.json)记录选用文件、运行监督
和报告生成身份；历史 kernel 审计单列保存，不与本轮 timing 完整性检查合并。

## 正确性验收

独立 `refactor_nosa_overlap_check_20261005_01` 检查每层全部 1024 queries，
FP32 QK+CIS / causal softmax / AV 参考通过 BF16 `atol=rtol=0.016` 验收。
Serialized 与 overlap 输出逐位一致，与 resident 通过相同容差。独立 check 还在
重复冷态调用中将 GPU 总传输字节、分 first-use group 字节与 CPU 去重结果核对。
L0/L15/L31 的 prefix K+V payload 分别为 8,683,520 / 13,041,664 / 13,074,432 B。
Bench/profile 复用此 receipt；bench 样本之间不读取诊断计数或比较输出。

同轮独立运行 [test_offload_checkpoint.py](../../models/nosa/tests/test_offload_checkpoint.py)
的 `test_cuda_offloaded_checkpoint_matches_independent_resident_64k_1k`，1 项通过。
完整 32 层分别从独立空 resident/offload cache 构建 64K prefix，再执行 1K extend；
全部 normalized hidden 逐位一致，`max_abs=0`。请求 SHA256 为
`65d62f9ee9662fbe4b1d95d8c5440f16e70336a10b1eb2c860c55d8af3176b6b`，
当前 byte-identical 请求保存在
`experiments/nosa_mfu/output/data/refactor_mfu_sparse_native_bench_20261005_01/request.json`。
该检查覆盖普通 owned-cache A1024 路径，不替代固定 P/NH serving 的验收；测试耗时
不作为模型推理延迟。测试结果只输出终端，独立验收记录不作为实验性能报告。

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
设备拓扑对应。已发布运行的原命令和环境保存在对应监督记录及 run metadata 中。

```bash
bash experiments/nosa_offload_overlap/scripts/run.sh --help

base_path="$(python3 -c 'import os; from pathlib import Path; v=str(Path.cwd()/".venv/bin"); print(":".join(x for x in os.environ["PATH"].split(":") if x != v))')"
nosa_run_env=(env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_HIP_ALLOC_CONF
  -u PYTORCH_NO_CUDA_MEMORY_CACHING -u PYTORCH_NO_HIP_MEMORY_CACHING
  -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES
  PATH="$base_path" CUDA_VISIBLE_DEVICES=1 CXLDSAGR_SM90_BACKEND=native
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  PYTORCH_ALLOC_CONF=backend:native,pinned_use_cuda_host_register:True,pinned_num_register_threads:8
  numactl --physcpubind=16-23 --membind=0)

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh nosa_offload_new_check \
  --mode check --device cuda:0 \
  --input-dir experiments/nosa_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --validation-receipt /tmp/nosa-offload-new-check.json

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh nosa_offload_new_main \
  --device cuda:0 --input-dir experiments/nosa_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 20 --validation-receipt /tmp/nosa-offload-new-check.json

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh nosa_offload_new_confirm40 \
  --device cuda:0 --input-dir experiments/nosa_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 40 --validation-receipt /tmp/nosa-offload-new-check.json

"${nosa_run_env[@]}" bash experiments/nosa_offload_overlap/scripts/run.sh nosa_offload_new_profile --profile \
  --device cuda:0 --input-dir experiments/nosa_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 3 --validation-receipt /tmp/nosa-offload-new-check.json
```

本次两份 JSON/CSV 先在系统临时目录独立生成，再选入本实验的 `report/fused/`
和 `report/fused_confirmation/`；生成器与复核依据保存在新主测 run 的 `report_generation/`。
重建时选用新的输出目录：

```bash
.venv/bin/python -m experiments.nosa_offload_overlap.src.report \
  --measurement-dir experiments/nosa_offload_overlap/output/data/refactor_nosa_overlap_bench_20261005_01 \
  --profile-dir experiments/nosa_offload_overlap/output/data/refactor_nosa_overlap_profile_20261005_01 \
  --output-dir /tmp/nosa_offload_report_rebuild_main

.venv/bin/python -m experiments.nosa_offload_overlap.src.report \
  --measurement-dir experiments/nosa_offload_overlap/output/data/refactor_nosa_overlap_confirm40_20261005_01 \
  --profile-dir experiments/nosa_offload_overlap/output/data/refactor_nosa_overlap_profile_20261005_01 \
  --output-dir /tmp/nosa_offload_report_rebuild_confirm40
```

调用模块：`operators.nosa.attention.offload.api.NosaFetchWorkspace`、
`operators.nosa.attention.offload._fused`、`operators.nosa.attention.device_only.api`、
`models.attention_contracts.BlockSelection`。输入加载及 fingerprint 显式复用
`experiments.nosa_mfu.src.capture_inputs` / `src.measure`，源码快照复用
`experiments.nosa_mfu.src.dense.sources`。`--synthetic` 生成合成激活和选择，不能
作为真实请求结果；resident 实验保留各自测量身份。

本次导航随公共契约迁移更新；已发布 run 的原源码清单和命令保留采集时路径，
目录迁移不构成新的 GPU 测量。
