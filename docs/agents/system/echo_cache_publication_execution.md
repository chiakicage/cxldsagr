# ECHO cache P6 发布执行记录

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

2026-10-03。**P0–P6 已完成**：正式 GR 结果、默认配置 review 与 README 已发布，
受影响旧 DeepSeek 子集已按 prepared03 清理。默认配置为 **C=1024/W=1024**，范围沿用
[发布清理范围](echo_cache_publication_scope.md) 和[实施计划](echo_cache_implementation_plan.md)。
实际 run、证据身份、配置及清理收据统一见[发布 ledger](echo_cache_publication_ledger.json)；
读者入口为正式报告（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/results.md`）和
[实验 README](../../../experiments/deepseek_v32_echo_cache/README.md)。下文保留已执行流程的验收边界与复现方式。

## 本次独立检查

`echo_chunks_report.py` 已区分 fixed workspace 与 deployment workspace，不将两协议
池化；C=2048 同一配置参加两协议但不当成独立重复。表格包括完整轨迹时间、首访、复访
命中、复访重建、cold prefix、p95、实际 retained session 数及 shared/session/hard-cap
字节。诊断复核完整 32 请求身份及采样范围；token ratio 使用分子/分母计数之和，dense
staging 不称为 retention hit。最终 default-selection gate 已为 `reviewed`；
选值记录（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/default_review.json`）结合重复波动、
冷构建/复访取舍和独立预算检查选择 C1024/W1024。

standalone sweep `20261003_echo_shared_chunks_02` 已验收并发布，接受 C=256/1024/2048；
C512 未通过原跨 C extend 门槛，C4096 超出预算，均不进入 GR 矩阵。首轮 formal run
prefix 为 `20261003_echo_gr_chunks_01`。首配置 fixed C256 已完整采集，后置 auditor
的 40 B CPU scratch 分类错误修复后恢复发布；其余配置沿用 frozen04 的 GPU 0 顺序
执行并接受。五组首轮配置加 C1024/C2048 独立重复共 **7 轮、896 请求**已发布，全部
candidate hidden 与末 token logits 已通过各自 HBM 对照审计。原 wrapper 的 exit 1
不改写为成功；C2048 fixed/deployment 复用同一配置，不重复计算独立样本。
standalone 的 cold prefill 排名不代替 GR 默认值选择。
memory collector 的运行是工程预算验收，不是新的论文性能实验；CPU/数值/回归测试也不
因此获得实验结果身份。不得把诊断插桩耗时放入正式延迟表。

本轮真实 run IDs、历史编排命令及已完成的 GR README 替换范围见
[发布命令记录](echo_cache_publication_commands.md)。最终报告由 session 25659 exit 0
生成，prepared03 由 root session 35272 exit 0 安装并清理。C1024 两次完整 ECHO 轨迹为
76.982–78.437 秒，C2048 为 128.349–129.513 秒；选择只适用于本次固定两轮轨迹。
C2048 的冷请求和 p95 更低，C1024 ECHO 比同 C `serial_sparse` 慢 15.08%，这些取舍
保留在正式报告中，不据此声称 ECHO 为最快方案或融合预取带来加速。

## 验收与报告层身份绑定

正式 source identity 为
`47585a615f0bb017f531118e8c8c180cb33cb14d63de447cd83c84966210a211`。
最终默认配置固定 NH=1,050,624、每层 P=32,768、4 GiB HBM / 64 GiB CPU DRAM，
H200 SM90 GPU 0 默认时钟，16 用户两轮顺序请求、64K history + 128 candidate。
GR 仍是十个独立 checkpoint dense-block source-input replay；真实前三层 64K+1K
数值门禁单独验收，不外推为训练得到的 DeepSeek 8B 或完整 61 层验证。

[内存 workload 绑定记录](echo_memory_formal_workload_binding.json)与正式报告绑定四组
已接受工程内存证据：`20261003_echo_memory_c1024_03`、
`20261003_echo_memory_c2048_fixed_03`、`20261003_echo_memory_c512_hbm_03`、
`20261003_echo_memory_c256_dense_05`。原 memory/formal workload 聚合身份分别保留；
唯一允许的配置差异是 generator `context_limit` 及其两个派生字段，两个上限均覆盖
实际 65,664 tokens。完整请求文件、32 条有序身份、tokenizer、访问轨迹及其他配置相同，
runtime/backend、固定 session 长度、资源预留和硬预算检查保持原约束。

按执行 agent/root 交接记录，报告层 workload 身份绑定修复的两组定向测试分别
**37 项、20 项通过**；这不是本次文档收口重跑的测试。独立 auditor 的 40 B CPU
execution scratch 分类修复完成 **111 项 CPU 回归及 Ruff**，完整 CPU 重审与原
wrapper 失败的恢复边界见[恢复记录](echo_gr_auditor_recovery.md)。这些修复及最终
报告生成没有新增 runtime 变化，不改写原始 measurement、source snapshot 或计时。

## 正式 GR 的外部审计与直接发布

首配置的 collector 及其 128 条请求完整有效，失败发生在测量完成后的独立 CPU auditor。
恢复证据、源码身份和命令见[恢复记录](echo_gr_auditor_recovery.md)。冻结树和原 stage
保持不变；新版 auditor 独立保存源码与 hash，仍以 `--repo frozen04` 核对原始 runtime。
外部 helper 对首配置采用已经接受的修正版审计，保留 collector 原始字节，在发布副本
添加 accepted audit、成功 auditor 日志及 `external_validation.json`。原失败 auditor
日志保留在外部 stage，只在 repair provenance 中引用其 hash。

后续 formal 配置使用同一 helper，在 frozen04 cwd 调用原 `-m measure`，从新外部
staging 完成新版 auditor 后直接发布到主仓库。log/profile 先安装，data 最后以
NOREPLACE 原子提交；失败回滚本次已移动目录。收据和实际 auditor/helper 源码随每个
run 的 `acceptance_tools/` 保存，不属于原 runtime snapshot，不重签测量 source。
这些 run 不再经过普通 import helper，也不填写虚构的原 wrapper exit 0。

## 从 frozen tree 导入接受的 sweep / diagnostics

当前冻结树为 `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-gr-tools-04`。
该树没有 `output` 符号链接；使用原 wrapper 的 sweep / diagnostics 成功运行先落到
冻结树的各实验 `output/`。它们保留以下导入步骤及既有 receipt；正式 GR 使用上节路径。

1. 从 run wrapper 的独立审计输出确认 status、完整请求/方案矩阵、数值张量和源码身份。
   同时核对 freeze/import guards。失败或不完整运行留在外部诊断区，不导入主仓库。
2. 对每个已接受 run 的 `output/data`、`output/log`、`output/profile` 生成逐文件 SHA256
   inventory。使用 `cp -a` 或等价复制到主仓库相同实验/分类的新 run ID；目标必须不存在。
   复制前后 inventory 必须完全相同。部分复制失败时只清理本次创建的目标，不改其他 run。
3. 保留 metadata 的原 frozen cwd、参数、run ID、时间及 source identity；路径迁移不
   重新签成主仓库当前源码、不改测量时间。源码 snapshot 仍由原 manifest 验证。
4. 导入后的数值依据为原 wrapper 的独立原始张量审计与逐文件一致的 copy receipt。
   从接受的 imported data 生成新报告，执行报告器自身必需的来源/数值检查；没有新疑虑
   时不再追加一轮完整张量重开。source enumeration 与诊断要匹配同一 runtime scope。
   intrusive diagnostics 的 C/workspace、workload、每请求 visit/cache state 必须与
   formal run 匹配。主仓库有无关 NOSA 开发不影响保存的原 source identity。
5. 写入 artifact import receipt，记录 source/destination、run ID、原/新 inventory
   hash 及审核入口；这是来源核验，不是新测量。receipt 放对应 run 的复现数据目录；
   不修改原始被审计文件。原运行 inventory 不包括新增 receipt，明确这个附加项。

正式 GR report 的复现入口为
`python -m experiments.gr_serving.src.echo_chunks_report --screen ... --runs ... --diagnostics ... --candidate-chunk ... --output-dir ...`。
本次已保留 C=1024、C=2048 deployment 的完整轨迹重复及匹配诊断。C256 既非
领先者也非拟选值，未触发额外重复/诊断；报告保留其有效首轮结果。每个样本的 run ID
和 actual workspace 均已写入报告。

### 最终默认配置的独立 review

本次独立 review 已通过，选定 C1024/W1024，并绑定全部七个 formal run、screen 与
四组 memory 身份。以下保留从 draft 到 reviewed 的流程；已发布的 review 不再待选值。
`echo_chunks_report` 不根据最快中位数自动选 C：完整矩阵、重复、诊断和内存验收齐备后，
先生成无 review 的报告，再从该报告生成尚未选值的记录：

```bash
.venv/bin/python -m experiments.gr_serving.src.echo_default_review \
  --summary <GENERATED_REPORT>/summary.json \
  --memory-summary docs/agents/system/memory_acceptance_summary.json \
  --output docs/agents/system/echo_cache_default_review.json
```

生成器只读取小型 JSON，初始 `status=draft`、`selected_chunk=null`。执行者审阅全部
样本后填写选定 C，并在 `assessments` 中分别写清完整轨迹差异与重复波动、cold prefix
和复访取舍、workspace 与 session 容量、适用范围和限制；最后改为 `status=reviewed`。
小于重复波动的差异不能被写成最优证明。配置受限于当前工作负载、硬件、NH/P、预算及
运行时 identity。此记录是工程选值依据，不是用户审批流程或统计全局最优声明。

用原报告的完整输入列表重建到新目录，增加 `--default-review`：

```bash
.venv/bin/python -m experiments.gr_serving.src.echo_chunks_report \
  --screen <ACCEPTED_SWEEP_DATA> --runs <ALL_FORMAL_RUN_DATA> \
  --diagnostics <MATCHING_DIAGNOSTIC_DATA> \
  --default-review docs/agents/system/echo_cache_default_review.json \
  --output-dir <NEW_REVIEWED_REPORT>
```

Review 绑定正式 source identity、所有 run IDs、screen run/hash 和完整测量事实 hash；
新增重复或改变任何测量事实后须重新生成 draft。它要求选定配置、1024/2048 对照和
descriptive fastest 的 deployment trace 至少重复两次，选定配置与对照的诊断齐备。
`--candidate-chunk` 可省略；若同时提供，必须等于 review 的选值。

内存引用绑定 `memory_acceptance_summary.json` 的实际 hash，并复核其各 run 的原
manifest/source identity、接受状态、workload/backend identity 和配置预算行。
报告层按上节已验证的请求内容绑定处理 context-limit 元数据差异，保留不同的原聚合身份。
比较正式
run 保存的非测试 runtime source 字典，不要求 memory 03/05 与 formal 04 的整体源码
hash 相同，不重签原 observer 身份。该步骤不重新解析大体积 raw traces；它引用之前已
完成的独立内存验收，并在输出中明确这一边界。

通过后 `summary.json` 的 gate 为 `reviewed` 并包含 selected C、review hash 和内存
核对依据；`results.md` 渲染同一选值与文字，避免 README 和机器结果状态矛盾。
`default_review.json` 原字节复制到报告，`report_provenance.json` 保存生成命令及四个
报告/验证工具的 hash 和来源路径，区别于正式 runtime source。发布时将这些 JSON 与
summary/results 一并选入报告；对应报告工具源码应随发布复现材料保存在 `output/data/`。
没有 `--default-review` 时仍保持 pending，不改变正式延迟或默认 C。

## 已执行的旧 NOSA 子集提取工具

工具：[旧 GR 实验（已结束）](experiment_organization.md#retired-gr-serving)。
测试：[旧 GR 实验（已结束）](experiment_organization.md#retired-gr-serving)。
默认行为为只读预检；不会请求或运行 GPU，也不改有效实验产物。

```bash
.venv/bin/python -m experiments.gr_serving.src.retain_nosa_subset
.venv/bin/python -m experiments.gr_serving.src.retain_nosa_subset \
  --prepare /mnt/ssd-wlcb/chenkaiqi/.cache/echo-nosa-subset-publication-03
.venv/bin/python -m experiments.gr_serving.src.retain_nosa_subset \
  --verify /mnt/ssd-wlcb/chenkaiqi/.cache/echo-nosa-subset-publication-03
```

`--prepare` 必须使用新的仓库外目录。其 `files/` 是待安装的替换文件，`plan.json`
记录全部旧文件 inventory、新文件 hash 和逐文件删除名单；可以先审图和审计再发布。
工具或原产物变化后必须重新 prepare，不能直接编辑 staging 来绕过 hash guard。

| 原 run | 保留测量/数值行 | Cases | Summary groups | HBM references |
|---|---:|---:|---:|---:|
| `gr_serving_h200_20261002_h4k_01` | 804 / 804 | 28 | 84 | 201 |
| `gr_serving_h200_20261002_h16k_01` | 804 / 804 | 28 | 84 | 201 |
| `gr_serving_h200_20261002_h64k_01` | 168 / 168 | 28 | 84 | 42 |

读取时核对原 audit 的全部 evidence hash、原完整 CPU reference hash inventory、
每个 source hash 和聚合 source identity。NOSA JSONL 逐字节选取原行，不重序列化。
用原冻结 `report.py` 重算后，与原 summary 的全部 NOSA groups 精确相等；64K 的
16 个空复访组仍为 count=0/null。读取原 report 源使用内存编译，不在 source snapshot
生成 `__pycache__`。新 summary/per-request 图只画 NOSA，使用 categorical population
和真实整数 request ID，空组保留图中断点，不补成零。

工具只允许显式列出的三个 mixed run 及其 NOSA profile 的两项 formal-input 元数据。
遇到未知旧报告/analysis 文件、非空混合 stderr、原证据变化或 symlink 时拒绝执行，
要求先核对归属。它不触及以下内容：

- 完整原 `source/`、source manifest、NOSA workloads 和 HBM references。
- NOSA profile 的原 metadata、source、samples、实际 work intervals 与张量。
- DeepSeek 独立 MFU/nonmatrix 两 run、三个 NCU run 及四个报告子目录。

它提取旧 `measure.stdout.log` 的 NOSA 原行，去除 DeepSeek 行和混合矩阵末尾的
`accepted` event，新增明确的 `derived_subset_retained` event。新 `audit.json` 为
`retained_subset`，记录原/新 hash 与原审核边界；不继续沿用整矩阵 accepted 断言。

原 profile metadata 本体保持不变。`formal_inputs/latency_metadata.json` 替换成带明确
derivation 标记的 NOSA metadata 副本，同目录新增 `subset_retention.json`，记录原/新
metadata hash、原 profile hash 和 run ID。这不改变 profile 的 source identity，不是
重跑或重签 profile。其报告中的原 profile metadata/analysis 也保持逐字节一致。

## 最后安装与 README 更新

新运行全部验收、正式报告与 README 可核验发布后，实际完成了 prepared03 安装。
[发布 ledger](echo_cache_publication_ledger.json)的 `status=published`，六项
`accepted_gates` 均已接受，`prepared_cleanup.status=applied`。ledger 保存七个正式
run、各 gate 证据、最终 C/NH/P/workload/hardware、选值取舍，以及发布文件与审计身份。
`published_files_sha256` 覆盖当前 README 和九个正式报告文件；正式 run 的 metadata、
audit 与 external validation 身份另列于 `formal_evidence_sha256`。
旧 4K/16K DeepSeek 范围已结束，新 64K 数字只进入新报告。

以下为已执行的安装命令，保留复现边界，不能向已安装目录重复 apply：

```bash
.venv/bin/python -m experiments.gr_serving.src.retain_nosa_subset \
  --apply /mnt/ssd-wlcb/chenkaiqi/.cache/echo-nosa-subset-publication-03 \
  --publication docs/agents/system/echo_cache_publication_ledger.json
```

Apply 再次核对全部原文件和 staging，先在外部创建短期 rollback 副本，逐文件安装；
异常恢复已改文件，成功后删除 rollback 副本。仅删除明确的 DeepSeek reference/workload
文件；保留 NOSA raw 和 source 的最终 hash 必须不变。外部 `receipt.json` 记录已安装
新文件、旧文件删除名单及保留核验；将 receipt 的相关证据索引加入工程发布记录。

README 与报告已在同次发布中更新为新 ECHO 64K 受控轨迹及原 NOSA 子集：移除旧
DeepSeek 的延迟表/排名/总计/容量结论，按模型重算原矩阵描述与比较数量；旧 NOSA 仍标
原 run、历史实现及预算边界，不称为共享 workspace 验收。保留原 NOSA profile 的限制和
run ID。不要删除 NOSA 正在补测说明。独立非矩阵/MFU README 只需保持现有计费边界说明。

## 本工具验证状态

12 项 CPU 检查通过：字节保持、未知模型/路径拒绝、历史代码读取不写 bytecode、
metadata 明示提取身份、原产物或 staging 变化拒绝、保护文件/独立 MFU 范围、
已发布证据 hash 约束、安装失败回滚及预存临时文件不被误删。真实三 run 已完成外部
准备和原证据核对。最终外部 staging 为上述 `echo-nosa-subset-publication-03`，
检查 2,162 个原文件，准备 93 个写入文件及 486 个仅属于 DeepSeek 的 reference/workload
删除项；原 measurements/correctness 的 NOSA 原行另做独立字节比对一致。再次 `--verify`
通过。工具 SHA256 为
`46795300ebf3fa60ef5b6f3abe1ff7dda38f5d660436024e2a1ec3df3f7b22a4`，
外部 plan SHA256 为
`25a4f75e9bea1343ee9c73739a90fb6fb3f1b97047024055ee1785ca5430e999`。

已目视检查三个长度的 summary 和全部 21 个逐请求面板：模型标签只有 NOSA，图例与
坐标可读，64K 无复访档位为 n=0/断点，未补零。`summary` 与 `summary_readable` 是同图
别名；逐请求同理。最终发布前完成外部预览，执行者已记录发布 ledger。
root session **35272 exit 0** 完成真实 `--apply`：**93 个文件写入、486 个 DeepSeek-only
文件删除、1,590 个 retained NOSA 文件核验通过**。外部 receipt 为
`/mnt/ssd-wlcb/chenkaiqi/.cache/echo-nosa-subset-publication-03/receipt.json`，其位置和
SHA256 已写入 ledger；独立非矩阵/MFU/NCU 产物保持原范围。

本次收口只读取现存小型 ledger、review、绑定记录、报告与 receipt，更新文档并检查
本地链接和空白；未重跑 GPU、原始张量/分配审计、全量 hash 或报告生成。
