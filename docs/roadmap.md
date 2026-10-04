# 下一步任务

> **2026-10-05 最新安排：**按用户继续推进目标的要求，恢复 H64K NOSA 性能优化。
> 继续比较相同配置的 HBM-only、dense prefetch、sync sparse 和 async sparse，
> 验证 candidate 效率、cache 正确性与完整请求收益。已完成的报告整理不再列为待办。

> **当前测量范围：**按研究者要求，后续先只测 64K history，暂停追加 4K、16K。
> 本轮已完成并验收的两档结果继续保留，可完成报告整理。

> **2026-10-04 实测更新：** [deepseek_v32_motivation](../experiments/deepseek_v32_motivation/README.md)
> 的固定 P=65,536 / NH=16,777,216、16 用户两轮四方案测量已完成，结果已回写状态表。
> NOSA 同参数的 `poolscan` 四方案及匹配 profile/API 已验收；通用 budget 三档短轨迹已于 2026-10-05 结束独立实验维护。其他 loop 规模、candidate 效率与 T-007 的归因仍未完成。
> 后续依据使用本次固定容量结果，不恢复已撤回的 4 GiB / W / chunk 排名。

> **2026-10-03 用户修正：** 原 DeepSeek 4 GiB / W / chunk 结果已撤回。
> T-006、T-007 中依赖旧控制点与旧排名的任务依据受影响，原任务保留待重新核对；
> [固定 P/NH 容量分析](../experiments/deepseek_v32_echo_cache/README.md)已按 GPU 临时
> candidate 更新：NH 只计 history，512 GiB DRAM 对应的静态边界为 455 个 64K 用户。
> 按用户要求不跑满容量，长跑不列为继续执行的任务。数值检查不等于容量或性能实测，
> 新估算不能恢复旧性能排名。其他研究者修正与 NOSA 待办保留。

更新：2026-10-05。两模型各有 16 用户两轮固定 P/NH 控制点。NOSA 元数据复用的测量与独立审计已完成，局部改善尚未证明完整 serving 收益，暂不接入。下一步准备 indexer 执行 workspace 复用原型，以完整 candidate 的实测收益决定是否采用；已有四方案正式结果保持原实现边界。

| 任务 | 下一步 | 主研究条目 |
|---|---|---|
| T-006 loop 数据与容量压力 | 在 64K history 范围内，基于两模型已测控制点选择其他有区分力的 U/R、P/NH 范围，保持完整用户覆盖；区分 token 额度与字节预算 | 1.4 |
| T-003 共享资源接入与可比范围 | 明确五类 baseline 的共同模型适配和实际执行路径；固定 token 额度与物理容量分别解释，保持 allocated、reserved 与设备已用量分开 | 2.1 |
| T-007 loop motivation 测量与归因 | 在 H64K 验证 indexer 执行 workspace 复用能否降低完整 candidate 延迟，保持 cache 正确性；有效后补齐四方案验收。保留 ECHO 预测/预取与历史保留的归因 | 4.1（关联 2.1、4.2） |
| T-002 场景与模型数据 | 继续明确 sparse attention GR 的具体任务与模型适配，评估 loop 能解释的系统问题及其真实场景边界，不以热度数据作为本轮 motivation 前置条件 | 1.2 |
