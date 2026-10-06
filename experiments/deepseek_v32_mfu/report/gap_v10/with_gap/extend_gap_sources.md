# Extend gap 来源

Serial sparse 的 gap 最大，其中约 75% 是 trace 中没有 GPU 活动的空闲；ECHO 约为
60%。以下统计覆盖 L0 首个计算 kernel 开始到 L2 最后一个计算 kernel 结束，沿用
当前 timeline 的 cold extend、128 tokens 和三层范围。融合计算/IO 整段算有效工作，
不计入 gap，也不从比例分母扣除。单位均为 µs。

| 来源 | HBM | ECHO | Serial sparse | Dense prefetch |
| --- | ---: | ---: | ---: | ---: |
| 暴露的 GPU 控制操作 | 105.856 | 292.736 | 194.720 | 52.768 |
| 下一活动对应 API 尚未开始时的空闲 | 0.000 | 148.640 | 330.780 | 0.000 |
| 下一活动对应 API 执行期间的空闲 | 0.000 | 28.442 | 60.174 | 0.000 |
| 下一活动对应 API 已返回后的空闲 | 150.368 | 264.228 | 192.086 | 57.152 |
| **Gap 合计** | **256.224** | **734.046** | **777.760** | **109.920** |

表中三类空闲合计为 GPU 空闲，分别占各方案 gap 的 58.69%、60.12%、74.96% 和 51.99%。
每段空闲按结束它的下一个 GPU 活动所对应 API 的开始、返回时间划分。
图内节点可能共用一个 `cudaGraphLaunch`。这只说明提交与执行的时间关系，不能将
“API 已返回后的空闲”直接归因为某种硬件停顿，也不能把“尚未开始”全部归为 Python。

## 暴露的 GPU 控制操作

这里只统计没有被计算或实际 IO 覆盖的部分，各行可以相加得到上表第一行。

| 操作 | HBM | ECHO | Serial sparse | Dense prefetch |
| --- | ---: | ---: | ---: | ---: |
| DeepGEMM scale 转置 | 59.776 | 59.776 | 59.552 | 19.808 |
| Projection 图内元数据与布局整理 | 26.208 | 26.048 | 25.920 | 8.704 |
| KV/index cache 写入与独立写回源拷贝 | 19.872 | 33.088 | 33.312 | 7.072 |
| 精确 recall / 驻留选择的元数据处理 | 0.000 | 65.792 | 75.936 | 7.104 |
| ECHO prefetch prepare | 0.000 | 65.536 | 0.000 | 0.000 |
| ECHO prefetch finalize | 0.000 | 15.712 | 0.000 | 0.000 |
| Prefetch hint 更新 | 0.000 | 26.784 | 0.000 | 0.000 |
| Dense 映射清除与发布 | 0.000 | 0.000 | 0.000 | 10.080 |

Scale 转置对应 `deep_gemm::transpose_fp32`；projection 布局整理包括图内的
arange、位置/边界元数据、cat 和 copy。Cache 写入项包含空槽分配、record/映射发布、
index K/scales 的 D2D，以及持久 D2H 写回所需的独立源拷贝；实际 H2D/D2H 已排除。
Recall 项包含 union、classify、空槽分配和映射发布，搬运本身不计入。

Dense 的 GPU 控制操作并集实际为 147.935 µs，其中 95.167 µs 与计算或 IO 重叠，
只有 52.768 µs 暴露为 gap。对应被覆盖的控制时间，HBM 为 0、ECHO 为 13.632、
serial sparse 为 4.128 µs。因此不能从较小的暴露值推断 dense 没有这些管理操作。

## 空闲主要出现在哪里

以下按空闲结束后执行的 GPU 活动定位，不作为该操作的独占耗时或因果归因。

- ECHO：融合 indexer 启动前的空闲合计 152.800 µs，其中 L0 单段为 146.112 µs；
  这段中有 138.426 µs 发生在对应 launch API 尚未开始时。Prefetch prepare 各活动前的
  空闲另有 77.152 µs，精确 recall 各活动前有 39.360 µs。
- Serial sparse：精确 recall 各活动前的空闲合计 392.288 µs；最长一段为 L0 的
  union clear 前 101.216 µs。该合计还包括三层空槽计数/分配前 77.472、57.472、72.256 µs
  的长空隙。MLA 前合计 40.256 µs，其中 L1 为 37.920 µs。
- HBM：空闲主要分散在计算图节点和 cache 写入之间，最大单段为 4.832 µs。
  每段空闲的下一活动 API 都已返回。
- Dense prefetch：最大单段为 4.672 µs；每段空闲的下一活动 API 也都已返回。
  与 H2D 重叠的控制操作及等待时间已经从 gap 中排除。

## 数据与验证

分析 ID 为 `deepseek_gap_v10_extend_sources_20261006_01`，输入为
`deepseek_gap_v10_three_layer_window_20261006_03/window_rows.json`，与当前最终图
使用相同的活动和窗口。原 profile run ID
为 `deepseek_gap_v10_a128_minimal_20261006_01`。这是已有侵入式 profile 的离线统计，
没有新增 GPU 运行，也没有修改模型、cache 或算子。

[汇总 CSV](extend_gap_sources.csv)、[控制操作 CSV](extend_control_sources.csv)和
[来源及核验记录](extend_gap_sources.json)保存未舍入数据、来源哈希和独立核对结果。
完整空闲区间、下一 GPU 活动及其 API 记录保存在
`experiments/deepseek_v32_mfu/output/data/deepseek_gap_v10_extend_sources_20261006_01/sources.json`。
两种独立区间算法逐项核对了四方案 gap、控制操作和空闲总量；所有空闲段均有唯一的
下一 GPU 活动及对应 API，三段提交时间划分逐段守恒。原始硬件、输入和验证边界见
[实验 README](../../../README.md)。
