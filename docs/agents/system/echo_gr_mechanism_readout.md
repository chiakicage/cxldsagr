# ECHO GR 采样机制解读草稿

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

2026-10-03。仅根据已有小型摘要
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/diagnostic_mechanism_preview.json`
和[已接受诊断记录](echo_gr_diagnostics_checkpoint.md)整理，没有重跑分析或增加验收。
来源为 `20261003_echo_gr_diagnostics_c1024_04`、
`20261003_echo_gr_diagnostics_c2048_04`。以下段落待与正式 GR 轨迹、预算和重复结果
结合后使用；诊断本身不支持延迟、overlap 收益或默认 C 结论。

## 可用于报告的事实段落

本实验分别观察用户历史是否仍可复用，以及当前稀疏选择中的历史记录是否已在 HBM。
对 ECHO 和 serial sparse，session/prefix 命中表示用户的固定历史及相关状态仍保留，
请求可以直接执行 candidate；历史主 KV 可在 host arena 中，所选记录是否保留在
有限 HBM pool 是另一项事实。历史 HBM token hit 在每个实际 layer/batch 的融合预取
与当前 append 之前测量，分母只包含精确选择中的历史记录，不含本批次新写入的记录。
汇总比例为各 layer/batch 的 resident 历史记录数之和除以历史选择记录数之和，
不是 batch 比例的均值，也不是跨全部层和批次全局去重后的 token 比例。

C1024/W1024 下，ECHO 与 serial sparse 的采样复访请求 16、31 均命中用户 session，
因此没有再次执行 prefix。但两者在 candidate extend 前的历史 HBM token hit 都是
0/135,860，即这些采样访问需要的历史记录在预取前均不在 HBM。ECHO 将其中
107,450 条记录通过融合预取搬入，其余 28,410 条由精确 recall 补齐；serial sparse
的 135,860 条均由 recall 搬入。两者在这两个采样请求的十层 extend 中，H2D 总量
同为 156,510,720 B。这说明 session 命中可以免去历史重建，同时仍需从 host 搬运
本次选择；这里观察到的 ECHO 差异是搬运分布在预取与 recall 两个阶段，不能据此
声称它减少了这些请求的 H2D 总量。

C2048/W2048 下，ECHO 与 serial sparse 的采样复访请求 16、31 都重建了 prefix，
复访身份保持不变。ECHO 随后的 extend 历史 HBM token hit 为
131,937/135,854（97.12%），serial sparse 为 131,898/135,854（97.09%）。
这个高比例对应刚完成本次历史重建后的 HBM 状态，不能作为跨请求保留了历史 token
的证据。以 ECHO 为例，两次采样复访的 prefix 重建产生 197,224,704 B H2D，随后
extend 另有 4,593,024 B；完整采样请求包含两部分。C1024/W1024 与 C2048/W2048
同时改变了执行 chunk 和预留 workspace，最终比较须使用对应部署配置的完整请求
时间及实际 session 容量，不能只比较 extend 阶段的 token hit。

上述分层计数仅覆盖预先指定的请求 0、15、16、31，分别是第一轮首尾首访和第二轮
首尾复访；每个采样请求包含实际执行的全部 prefix/extend layer/batch。两个诊断 run
都执行了完整的 32 请求、四种方案，并保存和独立核对了全部 128 份方案/请求的
candidate hidden 与末 token logits；其余 112 个方案/请求没有分层插桩记录。数值
输出覆盖全部请求，分层流量和 token hit 只描述四个采样请求，未采样请求不记为零，
这些计数也不外推为整个 trace 的流量。正式延迟使用独立、未插桩的性能运行。

dense prefetch 的历史 selection residency 为 100%，表示它已经为当前 layer/batch
搬入完整历史。这个量按 dense staging 单独报告；搬入完成后的可访问状态不能计成
预先保留在 HBM 的 token hit。在两个配置中，HBM 与 dense prefetch 的采样复访
16、31 也都重建了 prefix。session 命中、token 驻留和 dense staging 完成状态应
分别标注，避免将三个语义不同的量汇成同一种“缓存命中率”。

## 对应的最小数值表

下表只统计采样复访请求 16、31 的 extend；每行是 20 个 layer/batch 的合计。
H2D 字节不含同请求可能执行的 prefix 重建；该部分已在上文单独说明。

| C / W | 方案 | 采样复访的 session 状态 | 预取前 resident 历史记录 / 历史选择 | 历史 token hit | Prefetch records | Recall records | Extend H2D / B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 / 1024 | ECHO | 命中 | 0 / 135,860 | 0.00% | 107,450 | 28,410 | 156,510,720 |
| 1024 / 1024 | Serial sparse | 命中 | 0 / 135,860 | 0.00% | 0 | 135,860 | 156,510,720 |
| 2048 / 2048 | ECHO | 重建 | 131,937 / 135,854 | 97.12% | 781 | 3,206 | 4,593,024 |
| 2048 / 2048 | Serial sparse | 重建 | 131,898 / 135,854 | 97.09% | 0 | 3,971 | 4,574,592 |

预取前 resident 与最终搬运数不要求构成简单的静态互补等式：预取、新记录 append
及其后的资源管理会继续改变驻留状态。最终 bytes 使用已接受诊断中的实际搬运计数。
这 30 个已聚合采样 phase/visit 行的 capacity splits 与跨 consumer-group reread
均为 0；这个零值仅说明被观察的访问，不能升级成所有输入或整个 trace 的保证。

正式报告生成后使用其 `diagnostic_counts.csv`、`diagnostic_phases.csv` 和原 run
身份作最终引用；本临时预览不是新的实验产物，不复制到 `report/` 冒充正式汇总。
