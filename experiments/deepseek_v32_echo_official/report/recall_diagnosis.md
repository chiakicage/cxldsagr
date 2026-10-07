# 官方 ECHO recall 耗时诊断

官方 `_recall_update_extend_kernel` 的主要开销来自扫描整个 host pool 的标记。
NH=16,777,216 时，即使剩余 miss 为零，也会启动 65,536 个 CTA，各自循环检查
256 个标记。原 NSYS 中的 recall 几乎固定为 1.533 ms；独立零 miss 实验复现了
1.530 ms，NCU 将瓶颈指向循环中的整数与逻辑 ALU 指令。这一长耗时不需要实际
KV 传输就能出现。

## 源码与 launch 证据

官方提交为 `bc1b75c1000010d0ac6f032ebaac283255c050b1`，本次核对的文件与该提交一致。
`3rdparty/ECHO/sglang/python/sglang/srt/mem_cache/memory_pool_host.py`
第 917–919 行按 `self.size + 1` 分配 HBM 上的布尔标记，
第 1303–1337 行先清零、标记 miss、归约计数，随后调用 `recall_update_extend`，
没有按零 miss 跳过该调用。

`3rdparty/ECHO/sglang/python/sglang/srt/mem_cache/recall_ops.py`
第 611–638 行用完整标记数组长度决定 `grid = ceil(host_size / 256)`；
第 593–608 行让每个 CTA 依次读取 256 个 host ID 的标记，只有标记为真时才搬运 KV。
因此扫描范围由全局 NH 决定，不随本请求的 H、A 或实际 miss 数缩小。
`BLOCK=256` 表示每个 CTA 检查的标记数，实际线程数为 128。

原始 SQLite 中 195 次 recall 全部采用 `grid=(65536,1,1)`、`block=(128,1,1)`，
每线程 32 个寄存器。最小、平均、最大耗时分别为
**1.532445 / 1.532828 / 1.535710 ms**。其中 64 个 prefill chunk 的三层共调用
192 次，累计 294.302649 ms；完整 extend 的三层共调用 3 次，累计 4.598743 ms。
对照图中的本地 serial sparse gather 使用 `grid=(128,1,1)`、`block=(128,1,1)`。

官方窗口内的六次 recall 可在[活动表](sglang_engine/selected_activities.csv)核对：
prefill L0/L1/L2 位于第 464/579/694 行，extend 位于第 809/924/1039 行。
全阶段汇总见[诊断数据](recall_diagnosis/summary.json)。195 次 launch 的原始来源为
`experiments/deepseek_v32_echo_official/output/profile/first3_engine_warm_echo_profile_20261006_01/engine_pair.sqlite`
中的 `CUPTI_ACTIVITY_KIND_KERNEL`，按 `StringIds` 中的 kernel 名称筛选。

## 独立零 miss 验证

诊断入口 [recall_zero_miss.py](../src/recall_zero_miss.py)直接导入未修改的官方 kernel，
将 HBM bitmap 全部设为 false，传入不会进入访问分支的单条 dummy KV。
独立 check 检查 counter=0、映射与 KV 未变；它只验证零 miss 分支，不构成新的模型
或跨框架数值验收。源码 SHA256 为
`acccb812e162141a7c483a304aa82e14d4dd6b808c9e3d82d8bbd6dc693fbfee`。

16M 诊断沿用原 profile 已记录的 cubin，SHA256 为
`54389c69a31618c937e6a1f666a2486403c4b5d561ca683d8ec6b07089f7ea44`。
64K、1M 变体的 PTX 与它只差 host index 上界常量，dummy buffer 没有改变扫描实现。

运行环境为 GPU 7 的 H200 / SM90、Torch 2.8.0+cu128、Triton 3.4.0。
每张 CUDA Graph 包含 20 个原 kernel 节点，预热后测量 20 次 replay，使用 CUDA event
时长除以 20，报告每次 kernel 的中位数。bitmap 清零、归约、allocator 与 Engine
均不计入。check、计时与 NCU 使用独立运行。

| NH | CTA 数 | 零 miss kernel 中位数 ms |
| ---: | ---: | ---: |
| 65,536 | 256 | 0.022296 |
| 1,048,576 | 4,096 | 0.106495 |
| 16,777,216 | 65,536 | 1.530142 |

数据见[零 miss 表](recall_diagnosis/zero_miss.csv)，独立运行 ID 分别为
`recall_zero_miss_check_20261007_01`、`recall_zero_miss_bench_20261007_01`；
原始 JSON 位于本实验 `output/data/<run_id>/check.json` 或 `bench.json`。
零 miss 下仅改变 NH 就大幅改变耗时，确认了全 NH 扫描的固定成本。

NH=16,777,216 的独立 NCU full 采集耗时为 1.531616 ms，ALU 利用率为
83.9899%，SM throughput 为 84.2376%，achieved occupancy 为 98.3706%，
L1 hit rate 为 99.21875%，HBM 读吞吐仅为峰值的 0.2277%。
NCU 的 `HighPipeUtilization` 规则指出，整数、逻辑指令使 ALU 成为主要瓶颈，
热点对应源码第 594 行循环、第 596 行边界判断和第 597 行标记读取。
高 occupancy 与低 HBM 吞吐不支持“occupancy 不足”或“HBM 带宽耗尽”的解释。
指标及来源见[NCU 表](recall_diagnosis/ncu_metrics.csv)和
[来源记录](recall_diagnosis/provenance.json)，原始报告位于
`experiments/deepseek_v32_echo_official/output/profile/recall_zero_miss_ncu_20261007_01/`。

## Extend 与本地搬运对照

下表均为已有 NSYS 中的 GPU 活动时长，单位 ms，三层分别求和。
官方来源为 `first3_engine_warm_echo_profile_20261006_01`，本地来源为
`deepseek_dense_late_wait_profile_20261006_01`。

| 路径 | L0 | L1 | L2 | 合计 | 已确认的 host KV 读取量 |
| --- | ---: | ---: | ---: | ---: | --- |
| 官方 ECHO recall | 1.532925 | 1.533021 | 1.532797 | 4.598743 | 逐 kernel 字节数未知 |
| 本地 serial sparse gather | 0.215616 | 0.071488 | 0.194367 | 0.481471 | 8,909 / 2,875 / 8,007 records |
| 本地 ECHO residual recall | 0.062368 | 0.010208 | 0.037856 | 0.110432 | 2,464 / 269 / 1,458 records |
| 本地 dense 完整 history H2D | 1.374237 | 1.373885 | 1.373373 | 4.121495 | 每层 65,536 records |

本地时长、实际搬运计数及关联信息来自
[IO 核对表](../../deepseek_v32_mfu/report/full_extend_graph/audit/timeline_actual_io.csv)：
ECHO 位于第 11–16 行，serial 位于第 20–22 行，dense 位于第 26/27/29 行。
serial 的 launch 参数来自
`experiments/deepseek_v32_mfu/output/data/deepseek_dense_late_wait_profile_20261006_01/capture_10.sqlite`。

这里 serial sparse 的“全量 fetch”覆盖完整稀疏选择集合，不是全部 history。
三层实际读取共 19,791 条 BF16 record，每条 1,152 B，合计 22,799,232 B。
dense 则每层读取 75,497,472 B。官方 recall 的活动总时长是本地 serial gather 的
9.55 倍，也略长于本地 dense 的三层完整 history H2D。

本地 ECHO 的融合预取分别搬入 6,445 / 2,606 / 6,549 条 record，
与剩余 recall 相加，恰好等于该次 serial 的读取量。它的 residual recall 已随
剩余 miss 减少，三层合计仅 0.110432 ms。这组数据没有出现官方固定 1.533 ms 的症状。

## 驻留、图示与结论边界

[官方报告](sglang_engine/report.json)的 `residency_snapshots` 显示，
prefill 和 extend 结束后三层 history 全部驻留，extend 的 128 个 candidate tokens
也全部驻留。两个阶段保存的 `last_residual_recall_counter` 均为 0。
该 counter 在每次 recall 内清零，快照只保留最后一层最终 residual recall 的计数，
不包含先前层或融合预取的流量。阶段末驻留也不能证明整个执行期间一直驻留。

合图将 recall 画成 H2D 行的橙色条，条宽是 kernel 的执行时间；原始分类仍为
`ECHO recall (IO unknown)`。这一展示规则见
[合图来源记录](combined_extend/provenance.json)。mapped-host 的逐 kernel KV 字节数
没有测得，不能把橙色条当作已确认的纯传输时间，也不能因缺少 CUPTI memcpy 就认定
没有 H2D。标记扫描与真正的 host KV 读取是两种不同工作。

官方 Engine 保留 prefill 后的自然驻留，本地 offload 每次 extend 前清除主 KV 的
HBM 驻留；两者容量、软件环境、Hadamard 路径及执行组织也不同，官方数值验收仍为
`false`。这些 NSYS 活动时长不能代替独立请求计时，表中比值也不是同条件实现的
加速倍数。新增零 miss 实验隔离了原 kernel 的标记扫描成本，NCU 进一步定位到
整数与逻辑 ALU；它没有测量真实 miss 下的传输性能，也没有改变原 SGLang 的数值状态。

后续若优化，应先让召回使用压紧后的 miss 列表，并避免零 miss 时扫描全池；仅缩小
NH 会同时降低服务容量，只能作为诊断，不能冒充等容量优化。本次没有实施这些改动。

## 复现

分析发布 ID 为 `recall_diagnosis_20261007_03`，原始指标、逐行 stall 计数、PM sampling
和源码副本位于本实验的 `output/data/recall_diagnosis_20261007_03/`。
NCU source 采集的 kernel 时间为 1.531648 ms；未发射的 math-pipe-throttle 样本
共 16,717 个，全部位于第 594、596、597 行，barrier 与 membar 样本均为零。
这一结果与 full 采集的 ALU 诊断一致，不能归因为 miss 分支中的原子操作或 barrier。

以下从仓库根目录运行，替换输出 ID 以免覆盖已有数据。入口会在导入 GPU 依赖前
将可写 cache、临时路径限定到本实验 `output/`。

```bash
source 3rdparty/ECHO/reproduction/cxldsagr/env/activate.sh
export CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
recall_diag_dir=experiments/deepseek_v32_echo_official/output/data/recall_zero_miss_new
numactl --physcpubind=72-79 --membind=1 python -B \
  -m experiments.deepseek_v32_echo_official.src.recall_zero_miss \
  --mode check --output "$recall_diag_dir/check.json"
numactl --physcpubind=72-79 --membind=1 python -B \
  -m experiments.deepseek_v32_echo_official.src.recall_zero_miss \
  --mode bench --check "$recall_diag_dir/check.json" --output "$recall_diag_dir/bench.json"
```

NCU 使用同一入口的 `--mode profile --host-sizes 16777216`，传入匹配的 `--check`
和新的 `--output`。两次采集分别使用 `--set full --section PmSampling
--section PmSampling_WarpStates` 与 `--set source --section SourceCounters`，均用
`-k regex:_recall_update_extend_kernel -c 1`；完整参数与文件哈希见来源记录。
