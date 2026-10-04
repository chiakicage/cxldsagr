# 实验索引

实验按要回答的问题分为四类。数值正确性是共同前提，独立验收不作为性能结果。
本轮只整理代码、入口和已有产物，没有运行新实验。已有报告保留原 run ID、源码身份
及计时边界；新的 check/bench/profile 入口尚未完成 GPU 验收或性能测量。

| 类别 | 实验 | 目的与现有证据 |
|---|---|---|
| Motivation | [NOSA 固定容量](nosa_motivation/README.md) | 完整 32 层、固定 P/NH 四方案；历史保留与稀疏搬运有收益，A128 async 未快于 sync |
| Motivation | [DeepSeek 固定容量](deepseek_v32_motivation/README.md) | C10 checkpoint 工作负载替身四方案，含末 token LM head；不能直接拼接跨模型排名 |
| Motivation 容量依据 | [DeepSeek ECHO 容量](deepseek_v32_echo_cache/README.md) | history-only NH 与 GPU candidate 的静态规划；离线估算和实测分开，未跑满容量 |
| Baseline 性能合理性 | [NOSA resident baseline](nosa_baseline_performance/README.md) | 汇集完整 dense 与 native/Triton sparse 前向、模块耗时和 MFU，各自保留测量边界 |
| Baseline 性能合理性 | [NOSA 算子与模块效率](nosa_kernel_mfu/README.md) | 固定输入的 attention、score、完整 indexer/attention API；单个 MFU 不替代 serving 效率 |
| Baseline 性能合理性 | [官方 ECHO 适配](deepseek_v32_echo_official/README.md) | 同模型与 P/NH 的官方适配和本地 C10 对照 |
| Baseline 性能合理性 | [DeepSeek prefill/extend](deepseek_v32_echo_prefill/README.md) | 真实 checkpoint 第 0–2 层的计算后端与非矩阵适配效率，非完整 61 层 |
| Baseline 平台依据 | [CPU DRAM 带宽](cpu_dram_bandwidth/README.md) | NUMA、线程与 IMC 带宽，不是 offload 延迟实测 |
| Sparse pattern | [NOSA 选择模式](nosa_indexer_pattern_65536_1024/README.md) | sparse 传播及 dense 激活上的 QA/full-NOSA 选择、并集和分解；服务后续设计，暂不在论文主线 |
| 自有设计 microbenchmark | [NOSA offload overlap](nosa_offload_overlap/README.md) | 相同冷态稀疏并集的 serial/fused 延迟、唯一读取和内部工作区间；A1024 不替代 A128 serving |

## 独立验收、计时和诊断

本轮整理的 NOSA baseline、算子、offload 与两模型 motivation 提供独立的 `check`、
`bench`、`profile` 模式，参数见各 README 和脚本 `--help`。其他入口保留其文档声明的
阶段与测量边界，不因归类而声称已经完成相同的拆分。

`check` 的证据放系统临时目录或指定工程目录；`bench` 在正式采样前核对匹配的验收
记录，不在样本之间反复跑完整参考或保存全部输出；`profile` 单独采集诊断，插桩总
时间不替代正式延迟。记录复用不跳过模型运行时的 finite/repair、cache 事务、
allocator 校验或必要同步。CPU/GPU 回归使用 `bash scripts/run_tests.sh [cpu|gpu|all]`。

Baseline 检查须覆盖实际采用的实现和配置。旧 A1024 resident 对照不能单独证明
A128、compute graph、固定 cache 路径高效；缺少的测量仍为未测。

## 范围调整与历史资料

旧 `gr_serving` 的 capped-popularity 短轨迹已结束独立维护，四方案和三档历史作为
同一范围退出。共享请求构造迁至 `GR/workload.py`，来源与内存审计迁至 `evaluation/`；
通用 budget serving 实现及必要回归保留。详见[整理记录](../docs/agents/system/experiment_organization.md)。

Dense 与 sparse resident 的原报告和数据迁入 `nosa_baseline_performance`，保持
原 run ID 和字节身份。Pattern 的假定带宽/MFU 下理想时间、overlap 与阈值支线已
结束；实际 QA/full-NOSA 观测保留。整理不产生新的性能结论。

[旧 DeepSeek / SM120](legacy/deepseek_v32/README.md)保留有效历史报告与 CPU 重建工具。
性能运行按 `output/{log,data,profile}/<run_id>/` 分类；报告素材放 `report/`。
环境准备见[项目 README](../README.md)，模型实现见[models](../models/README.md)。
