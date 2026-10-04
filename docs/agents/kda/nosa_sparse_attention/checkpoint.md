# NOSA resident block sparse attention 实现检查点

本页记录 2026-09-29 已验收的 FA3-v3 resident 实现，保留完整 NOSA 33/64 selection、CIS
bias、因果遮罩和数值修复。它是 resident 算子检查点；融合 host fetch 的实现见
[offload 检查点](../nosa_offload_attention/checkpoint.md)。

## 实现与派发

满足 native D128、GQA16 和地址/stride 对齐条件，且 BF16 的 K/V stride 匹配时，
使用固定版本 FlashInfer 0.6.18 的 FA3 Hopper 模板。八个相邻 query 组成逻辑块并集，
KV128、双 stage，通过 TMA 直接读取原始 stride 的 Q，epilogue 直接写最终输出。
每个 query 仍使用自己的 membership、CIS 和因果遮罩。

当工作项数为 256 时，按实际 paired-page tile 数降序调度，以逻辑 ID 升序处理并列。
只有八个 query 都选择了两个完整因果页时，common-page 快路才省去对应 element mask。
目标形状的完整调用包含四个 kernel：并集准备、工作排序、FA3、native per-query repair。
其他正长度 FA3 形状保留三个 kernel，空 query 不 launch；FP16 或 K/V stride 不同保留
native 路径。异常值检测和 repair 的耗时始终计入完整 attention。

QK 缩放和 CIS 加法保持独立 FP32 RN 舍入，先在自然单位中减去最大值，再转换为 exp2
输入，修复大公共偏移及 QK/CIS 抵消导致的精度问题。Native BF16 PV 按实际 token 数
缩放 V，归一化后恢复尺度，并保护最终 dtype 转换，修复极大有限 V 导致的中间溢出。
既有误差容限未放宽。

## 已知结果与未完成工作

原 run `kda_main_bf16_pair_v3_development` 的完整 attention kernel 总时间在 L0/L15/L31
分别为 175.391 / 173.247 / 170.879 µs，对应 39.312% / 39.798% / 40.350% useful MFU。
只有 L31 达到 40%；目标要求三层均低于 172.374 µs。该数据包含全部准备、排序和修复，
与 CUDA-event API、wall completion 和历史 private graph API 区间分开报告。

该联合检查点及其来源、测试和实验补测状态见[KDA 索引](../README.md#resident-联合验收与报告状态)。
模块数据见[实验报告](../../../../experiments/nosa_kernel_mfu/README.md#完整模块检查点)，
完整模型对照见[端到端与模块报告](../../../../experiments/nosa_baseline_performance/README.md)。
未通过性能验收或尚未完成 GPU gate 的候选保存在[调查记录](investigation_log.md)，均未升级为当前实现。

入口：[SM90 算子](../../../../operators/nosa/README.md)、
[NOSA 模型](../../../../models/nosa/README.md)、[后续计划](implementation_plan.md)。
