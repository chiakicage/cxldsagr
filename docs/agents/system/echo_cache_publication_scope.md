# ECHO cache 补测的发布与清理范围记录

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

> 迁移定位：本轮实验已独立到 [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md)。
> 下文执行历史中的原命令、源码路径和哈希保持原样；产物现位置通过
> [迁移记录](echo_cache_experiment_migration.md)回查，目录迁移不构成新的测量。

2026-10-03。**P0–P6 已完成**：七组正式 GR 运行、默认配置 C1024/W1024、最终报告与
README 已发布；随后完成 prepared03 的限定范围清理。实际发布与安装证据见
[发布 ledger](echo_cache_publication_ledger.json)，新结果见
正式报告（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/results.md`）。本次文档收口
不执行清理或修改 NOSA 研究状态；以下记录实际保留和替换范围。

## 保留独立的最新非矩阵与 MFU 交付

以下两 run 已完成 shared-cache 迁移，并在相同 cache 源码、输入与预算下比较非矩阵实现：

- `20261003_echo_layers3_nonmatrix_control_01`
- `20261003_echo_layers3_nonmatrix_candidate_02`

范围为真实前三层、64K+1K、C=1024、P=16384、NH=66560、5 GiB HBM / 64 GiB DRAM。
当前 P=32768、64K+128 的 chunk/GR 实验不替代这一非矩阵实验，保留：

| 位置 | 来源与用途 |
| --- | --- |
| `report/layers3/` | candidate_02 延迟、逐算子 MFU、分解、守恒审计 |
| `report/backend_comparison/` | 两 run 对照、控制组审计、相同 cache 源码证明 |
| `report/nonmatrix_comparison/` | 非矩阵互斥 scope 对照及运行时来源审计 |
| `report/mla_reference/` | candidate_02 实际 MLA 输入的独立 FP32 数值检查 |
| 两 run 的 `output/{data,log,profile}/<run_id>/` | 原始测量、源码、数值张量及 profiler 证据 |
| 三个 NCU run 的同类目录 | `20261003_echo_ncu_nonmatrix_{mla,indexer_resident,indexer_offload}_02` |

核对依据为上述实验 README、`backend_comparison/cache_source_identity.json`、
`layers3/summary.json` 与 `nonmatrix_comparison/source_runtime_audit.json`。
两 run 的共享池 hash 均为 `5b578d06d94e67e0d8131951d364346dbbb008b14f3233ff8dea15d74a7d5193`，
已非旧 per-session cache；resource helper hash 为
`bd8cabb0ba72fc4647fddaf14a14bdd4ed6897edd01dab75a7c394e96dd924a0`。
candidate 保存的模型/算子/cache 运行源与本次修正后相比，既有文件仅
`echo_infer.py`、`sparse_token_pool.py`、`cache_resources.py` 不同，另新增
`host_allocation.py`。非矩阵与算子源码保持其候选身份。

这两个 run 早于 pinned-bin 计费与 40 B CPU execution scratch 修正。原资源账本
记录的是当时的逻辑 tensor 容量，低估实际 pinned allocation，不能作为当前完整硬预算
验收。固定 NH/P 下，旧 allocator 本已分配相同的物理 bins；规划与 backing 分配不在
其正式计时内。因此保留原单用户计算性能、MFU 和数值证据，同时明确旧源码与预算边界，
不把旧数字重新标为新实现结果。若未来更改实际计算或传输路径，按影响范围另行补测。

## 已替换的旧 GR DeepSeek 结果

本轮替换前发布的是以下 mixed 短轨迹，并不存在已验收的 sequential 报告。
表中保留原 run ID 和整体源码身份供溯源；这些目录现只保留 NOSA 子集及其原证据：

| 报告 | 正式 run ID | 整体源码 SHA256 |
| --- | --- | --- |
| `report/h4k/` | `gr_serving_h200_20261002_h4k_01` | `67f91c34ead43d4056500661caa1c67e8ef8a50bc25327e081a0561bbab43060` |
| `report/h16k/` | `gr_serving_h200_20261002_h16k_01` | `45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c` |
| `report/h64k/` | `gr_serving_h200_20261002_h64k_01` | `45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c` |

三者均用旧 per-session `SparseTokenCache`（hash
`646066a9bdf70c58a272f4838b7dbe484fdccfd2e51202f7f97e5e44416399a6`），
源码 manifest 中没有 shared pool。全部 DeepSeek HBM/ECHO/serial/dense 的旧延迟、流量、
预算容量、LRU 解释和派生图表均受本轮模型调度/cache 改造影响。新交付验收发布后，
相应旧 DeepSeek 子集已替换并清理，不再作为“优化前 baseline”保留。

新 64K 顺序轨迹不等于重测旧 4K/16K 热度轨迹。发布已明确本轮任务覆盖与结束的旧范围，
不能把新 64K 数字填入旧长度，也不能将旧数据继续作为当前容量/规模结论。

这些正式 run 的 README、CSV/JSON、图表及 `output/` 原先都混有 NOSA，清理按模型
子集处理混合素材及其溯源，未整目录删除；保留尚需由 NOSA 任务独立处理的有效证据。
以下 NOSA-only layer-31 profile 也不属于 ECHO 改动的清理范围：
`gr_serving_h200_20261002_h4k_profile_02`、`gr_serving_h200_20261002_h16k_profile_01`、
`gr_serving_h200_20261002_h64k_profile_01`，其共同 profile 源码 SHA 为
`cd72caf5edbac19a54d162e26d3339b05dd0671048162d970729669af5d0c48b`。
NOSA 自身后续改造的验收与替换由对应任务决定，本文不为其重新背书。

`gr_serving_h200_20261002_sequential_u16_t32_h64k_01` 因冻结检查失败，仅为系统临时目录
中的诊断；当前 `report/` 和分类 `output/` 都没有这一 run。它不是需要保留的旧有效对照，
也不能成为新报告的性能 baseline。发布脚本仅清理明确列出的受替换 run/subset，保留
上述独立非矩阵交付及未受本次改动影响的 NOSA 证据。

## 已完成的发布与清理

新 GR 报告共 7 轮、896 请求，默认 review 已通过；C1024/W1024 绑定四组已接受内存
证据，保留独立 observer 身份以及完整请求内容相同、generator context-limit 元数据
不同的边界，见[workload 绑定记录](echo_memory_formal_workload_binding.json)。
报告仍明确 C2048 冷请求/p95 更低、C1024 ECHO 比同 C `serial_sparse` 慢 15.08%，
选值仅适用于固定 16 用户两轮 64K+128 轨迹。

prepared03 在新报告与 README 发布后由 **root session 35272 exit 0** 实际 apply，
**93 个文件写入、486 个 DeepSeek-only 文件删除、1,590 个 retained NOSA 文件核验通过**。
ledger 的 `prepared_cleanup.status=applied`，并记录外部
`/mnt/ssd-wlcb/chenkaiqi/.cache/echo-nosa-subset-publication-03/receipt.json` 的位置与身份。
原 NOSA 子集共 84 cases、1,776 请求/数值行、1,332 次非 HBM 比较，保留原运行身份和
预算限制，不升级为 NOSA shared-workspace 硬预算 serving 验收。

外部预览、保留 NOSA 子集的 hash guard、frozen run 导入、37/20 项报告层定向测试和
111 项 corrected-auditor CPU 回归记录见[发布执行记录](echo_cache_publication_execution.md)。
历史及复现命令见[命令记录](echo_cache_publication_commands.md)。报告层身份修复没有
新增 runtime 变化；本次仅更新文档并检查链接和空白，未重跑 GPU、raw audit 或 hash。
