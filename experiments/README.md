# 实验索引

实验按要回答的问题分为四类。数值正确性是共同前提，独立验收不作为性能结果。
各报告分别记录独立验收、正式计时和 profile 的 run ID、源码身份及测量边界。
两模型 motivation 给出原已发布结果、同机 P0 与重构后结果的对照，保留阶段开销增加和
未达到的性能目标。数值检查不替代正式计时和 profile。

| 类别 | 实验 | 目的与现有证据 |
|---|---|---|
| Motivation | [NOSA 固定容量](nosa_motivation/README.md) | 完整 32 层、固定 P/NH 四方案；历史保留与稀疏搬运有收益，A128 async 未快于 sync |
| Motivation | [DeepSeek 固定容量](deepseek_v32_motivation/README.md) | C10 checkpoint 工作负载替身四方案，含末 token LM head；不能据此直接给出跨模型排名 |
| Motivation 容量依据 | [NOSA / DeepSeek cache 管理](cache_management/README.md) | 共同准入和计费契约；分别说明 NOSA 懒分配与 DeepSeek 全局 arena，静态规划和请求观测分开 |
| Baseline 性能合理性 | [NOSA MFU](nosa_mfu/README.md) | 统一汇总算子、完整模块的效率，以及完整 dense/native/Triton 模型的已有测量；各自保留输入与计时边界 |
| Baseline 性能合理性 | [官方 ECHO 适配](deepseek_v32_echo_official/README.md) | 同模型与 P/NH 的官方适配和本地 C10 对照 |
| Baseline 性能合理性 | [DeepSeek prefill/extend](deepseek_v32_echo_prefill/README.md) | 真实 checkpoint 第 0–2 层的计算后端与非矩阵适配效率，非完整 61 层 |
| Sparse pattern | [NOSA 选择模式](nosa_indexer_pattern_65536_1024/README.md) | sparse 传播及 dense 激活上的 QA/full-NOSA 选择、并集和分解；服务后续设计，暂不在论文主线 |
| 自有设计 microbenchmark | [NOSA offload overlap](nosa_offload_overlap/README.md) | 相同冷态稀疏并集的 serial/fused 延迟、唯一读取和内部工作区间；A1024 不替代 A128 serving |

## 独立验收、计时和诊断

NOSA MFU、offload、两模型 motivation、官方 ECHO 与 DeepSeek 三层实验提供独立的
验收、计时和 profile 入口，具体参数及完成状态见各 README 和脚本 `--help`。
Pattern 按自己的数值验收与选择采集阶段运行，不把采集过程作为性能计时。

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

NOSA resident baseline 与算子 MFU 已合并为 `nosa_mfu`，作为本组唯一的 MFU 报告入口；
受重构影响的真实输入算子、模块和完整模型已有独立补测；未受影响的 synthetic 与
kernel 专项仍保留原 run ID 和字节身份。Pattern 在假定带宽/MFU 下的理想时间、overlap
与阈值支线已结束；实际 QA/full-NOSA 观测保留。目录合并本身不产生性能结论。

旧 DeepSeek / SM120 与 CPU DRAM 带宽实验已移出 Git，分别保存在
`local/experiments/legacy/` 和 `local/experiments/cpu_dram_bandwidth/`。
这些本地副本保留原报告和产物，不随仓库分发，也不作为当前实验的依赖。
性能运行按 `output/{log,data,profile}/<run_id>/` 分类；报告素材放 `report/`。
环境准备见[项目 README](../README.md)，模型实现见[models](../models/README.md)。
