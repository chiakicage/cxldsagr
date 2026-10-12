# 实验索引

实验按要回答的问题分为四类。各实验明确数值验收状态，独立验收不作为性能结果。
官方 SGLang 复现按用户要求仅测量性能，保留数值未通过的说明。
各报告分别记录独立验收、正式计时和 profile 的 run ID、源码身份及测量边界。
DeepSeek C10 已完成三档历史、
四档候选的 12 点四方案独立验收和正式计时，本轮没有新采集 profile。
数值检查不替代正式计时和 profile。

| 类别 | 实验 | 目的与现有证据 |
|---|---|---|
| Motivation | [DeepSeek 固定容量](deepseek_v32_motivation/README.md) | C10 checkpoint 工作负载替身，固定 P/NH 的 12 点四方案矩阵，含末 token LM head；单轮轨迹不证明稳定排序或跨模型排名 |
| Motivation 容量依据 | [DeepSeek cache 管理](cache_management/README.md) | 准入和计费契约、全局 arena；静态规划和请求观测分开 |
| Baseline 性能合理性 | [官方 SGLang 复现](deepseek_v32_echo_official/README.md) | 原始前三层固定历史负载的 ECHO / HBM-only 独立运行；仅报告性能，容量差异单列 |
| Baseline 性能合理性 | [DeepSeek MFU](deepseek_v32_mfu/README.md) | 真实 checkpoint 第 0–2 层，四方案 prefill/extend 的逐算子与阶段 MFU、单层计算/控制/IO 时间线；非完整 61 层 |
| Baseline 性能合理性 | [DeepSeek observation](deepseek_v32_observation/README.md) | 无 offload 的 52 点 B/Q/H 单层矩阵，公共算子与 MoE 各测 7 档 S；分列具体 GEMM、non-GEMM 与 API 聚合的时间/MFU、独立实际 HBM 带宽及利用率，图已嵌入报告 |

## 独立验收、计时和诊断

DeepSeek motivation 与三层实验提供独立的
验收、计时和 profile 入口，具体参数及完成状态见各 README 和脚本 `--help`。
官方 SGLang 的复现代码和原始产物位于本地 `3rdparty/ECHO/reproduction/cxldsagr/`，
实验目录只保存报告及选定数据；原本接入本地框架的官方 ECHO 适配及旧记录已删除。

`check` 的证据放系统临时目录或指定工程目录；`bench` 在正式采样前核对匹配的验收
记录，不在样本之间反复跑完整参考或保存全部输出；`profile` 单独采集诊断，插桩总
时间不替代正式延迟。记录复用不跳过模型运行时的 finite/repair、cache 事务、
allocator 校验或必要同步。CPU/GPU 回归使用 `bash scripts/run_tests.sh [cpu|gpu|all]`。

Baseline 检查须覆盖实际采用的实现和配置。旧 A1024 resident 对照不能单独证明
A128、compute graph、固定 cache 路径高效；缺少的测量仍为未测。

## 范围调整与历史资料

旧 `gr_serving` 的 capped-popularity 短轨迹已结束独立维护，四方案和三档历史作为
同一范围退出。共享请求构造迁至 `GR/workload.py`，来源与内存审计迁至 `evaluation/`；
通用 budget serving 实现及必要回归保留，当前接口见 [serving](../serving/README.md)。

旧 DeepSeek / SM120 与 CPU DRAM 带宽实验已移出 Git，分别保存在
`local/experiments/legacy/` 和 `local/experiments/cpu_dram_bandwidth/`。
这些本地副本保留原报告和产物，不随仓库分发，也不作为当前实验的依赖。
性能运行按 `output/{log,data,profile}/<run_id>/` 分类；报告素材放 `report/`。
环境准备见[项目 README](../README.md)，模型实现见[models](../models/README.md)。
