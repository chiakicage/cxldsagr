# NOSA resident indexer 实现检查点

本页记录 2026-09-29 已验收的 BF16-pair 实现。完整 indexer 尚未达到三层均为 40% useful MFU
的目标；原结果与联合验收信息见[KDA 索引](../README.md#resident-联合验收与报告状态)。

## 实现与语义

模型负责压缩和选择语义，`NosaKVCache` / `IndexerCache` 保存请求级派生记录与共享
scratch。32-token、stride-16 压缩只追加新窗口，稳定 CIS pool 随 KV 一起提交、回滚
和截短。checked native 入口合并有限值校验、增量准备与选块的主机提交，先检查可写
缓冲别名；非法 Q、新 K 或新 CIS 不修改派生记录。同步 checked 入口拒绝 CUDA Graph
capture，异步准备接口保持可捕获。

SM90 大形状路径采用 CUDA/CuTe WGMMA。第一遍 QK 计算逐 Q head softmax normalizer，
第二遍融合归一化、GQA、BF16 舍入、五窗口 pooling 和完整选择，减少中间张量与 launch。
ranked preparation 同时准备 CIS 排序，最终选择复用精确 rank-33 cutoff，并只排序保留的
64 个逻辑 ID。Q16/N128 第一遍以四个 consumer warpgroup 和 producer 交错执行。

对 `queries=1024, KV heads=2, compressed count=4159, blocks=1040,
query_start=65536`，额外使用向上舍入的 FP16 上界进行精确剪枝：只有上界严格低于当前
精确 cutoff 才跳过第二遍计算，不确定情况仍计算。Q8 独立尾段列表避免另一组 query
不需要的 tile；描述符生存期、固定几何特化及 BF16 成对转换降低额外开销。
其他形状和资源条件不满足时保留已有路径；公开 score-only 调用不使用该剪枝。

QK 缩放保持独立 FP32 RN 舍入，在自然单位中求最大值并做减法，再转换为 exp2 输入。
相关大公共偏移精度问题已按原容限修复，未通过放宽误差容限验收。

## 已知结果与未完成工作

原 run `kda_main_bf16_pair_v3_development` 的完整 indexer kernel 总时间在 L0/L15/L31
分别为 141.471 / 137.854 / 137.504 µs，对应 24.741% / 25.390% / 25.455% useful MFU。
达到 40% 需要降到 87.503 µs。该区间包含校验、增量准备、selection 与 helper，不能由
score-only 或 private fused-kernel 时间替代。详细源码身份、paired 候选验证以及不同
计时区间保留在[调查记录](investigation_log.md)。

完整模型 native/Triton 已在 `94bf521` 补测；2026-09-30 模型/cache 源码图扩展后的
当前分支尚未补测。Synthetic operator 对照和受影响的 full-NOSA pattern 也仍待补测，
原 run 和适用边界见[共用报告状态](../README.md#resident-联合验收与报告状态)。
本页没有把路径整理当作新测量，也没有把历史候选升级为已集成实现。

入口：[SM90 算子](../../../operators/nosa/README.md)、
[NOSA 模型](../../../models/nosa/README.md)、[后续计划](implementation_plan.md)。
