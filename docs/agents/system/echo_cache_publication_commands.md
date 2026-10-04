# ECHO cache P6 发布与复现命令记录

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

> 迁移定位：本轮实验已独立到 [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md)。
> 下文执行历史中的原命令、源码路径和哈希保持原样；产物现位置通过
> [迁移记录](echo_cache_experiment_migration.md)回查，目录迁移不构成新的测量。

2026-10-03。**P0–P6 已完成**，standalone sweep、七组正式 GR 运行、默认配置 review、
最终报告及 README 已发布，prepared03 清理已应用。默认 **C1024/W1024**；实际执行、
run ID 和 receipt 见[发布 ledger](echo_cache_publication_ledger.json)，结论见
正式报告（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/results.md`）。已存在的 run
不得重跑或覆盖。首个 GR fixed C256 用修正后的独立 auditor 恢复发布，详见
[恢复记录](echo_gr_auditor_recovery.md)。其余首轮配置与重复沿用 frozen04 collector，
由外部 helper 在审计通过后直接发布到主仓库。沿用[发布执行记录](echo_cache_publication_execution.md)和
[清理范围](echo_cache_publication_scope.md)。权重固定为 `/preset-models`；非 GR
数值验收是 checkpoint 第 0–2 层，GR 是十个独立 dense block 的 source-input replay。

下文保留原编排步骤和复现命令模板，不表示草稿中的每一条中间命令都实际运行。
实际报告命令与工具身份以
report provenance（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/report_provenance.json`）
为准，发布复现目录为 `output/data/20261003_echo_gr_chunks_publication_01/` 的
`all_traces/`、`reviewed/` 和 `reviewed_report_tools/`。重新测量或重建应使用新的 run ID
或输出目录，不能原样覆盖下列已接受来源。

## 1. 首轮配置与核对（已完成）

历史命令从主仓库同一 Bash 会话执行。以下 run ID 和最终发布目录现已存在；它们记录
本轮来源，不再是预留任务。正式 GPU 测量期间未并发运行报告、全量 hash 或张量重审。

```bash
set -euo pipefail
cd /mnt/ssd-wlcb/chenkaiqi/cxldsagr
echo_frozen=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-gr-tools-04
echo_output=experiments/gr_serving/output
echo_screen=experiments/deepseek_v32_echo_prefill/output/data/20261003_echo_shared_chunks_02
echo_publication=$echo_output/data/20261003_echo_gr_chunks_publication_01
echo_tools=$PWD/$echo_output/data/20261003_echo_gr_chunks_01_fixed_c256/acceptance_tools
echo_first_ids=(
  20261003_echo_gr_chunks_01_fixed_c256
  20261003_echo_gr_chunks_01_fixed_c1024
  20261003_echo_gr_chunks_01_fixed_c2048
  20261003_echo_gr_chunks_01_deployment_c256
  20261003_echo_gr_chunks_01_deployment_c1024
)
echo_diagnostics=(
  "$echo_output/data/20261003_echo_gr_diagnostics_c1024_04"
  "$echo_output/data/20261003_echo_gr_diagnostics_c2048_04"
)
```

以下函数保留每次原始 collector 的 cwd、参数、源码与运行时间。auditor 从已归档的
修正版独立调用，仍用 `--repo frozen04` 核对原源码；未修改冻结树的旧 auditor。
helper 自带失败留外部 staging、无覆盖发布和 data 最后原子提交。首配置已经恢复，
这里保留其余四个已完成配置的命令。

```bash
echo_formal_case() {
  local echo_run_id="$1" echo_chunk="$2" echo_workspace="$3"
  .venv/bin/python "$echo_tools/formal_runner.py" \
    --run-id "$echo_run_id" --frozen-repo "$echo_frozen" \
    --output-root "$PWD/$echo_output" \
    --staging-root /mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs \
    --auditor "$echo_tools/audit.py" \
    --source-sha256 47585a615f0bb017f531118e8c8c180cb33cb14d63de447cd83c84966210a211 \
    --auditor-sha256 775e0d4b4b15ece3dd1fdbe5f6b8b612cf43a7684b7f3a6a006e597ed79a4b54 \
    --freeze-verification-record 'Root frozen04 verification session 66704 exit 0; each collector/auditor verifies frozen sources.' \
    --chunk-size "$echo_chunk" --workspace-query-tokens "$echo_workspace" --gpu 0
}
echo_formal_case 20261003_echo_gr_chunks_01_fixed_c1024 1024 2048
echo_formal_case 20261003_echo_gr_chunks_01_fixed_c2048 2048 2048
echo_formal_case 20261003_echo_gr_chunks_01_deployment_c256 256 256
echo_formal_case 20261003_echo_gr_chunks_01_deployment_c1024 1024 1024
```

矩阵全部接受后已核对每个 run 的 `external_validation.json` 与工具执行 exit；以下保留
首轮汇总模板。首配置 receipt 保留原 wrapper exit 1；其他配置记录外部 collector /
auditor 的真实退出状态。它们不使用普通 import helper，也不声称原 wrapper exit 0。

```bash
mkdir "$echo_publication"
echo_first_runs=()
for echo_run_id in "${echo_first_ids[@]}"; do
  test -f "$echo_output/data/$echo_run_id/independent_audit.json"
  test -f "$echo_output/data/$echo_run_id/external_validation.json"
  echo_first_runs+=("$echo_output/data/$echo_run_id")
done
test "${#echo_first_runs[@]}" -eq 5
```

外部 helper 保留 collector 的 metadata、source snapshot、测量及完整张量。恢复 run
有原库存与副本字节核对，续跑 run 从独立 staging 审计并原子提交。直接执行报告器
必需的检查，不额外重复全量张量重开。已接受 sweep/diagnostics 的原 import receipt 不变。
任何步骤失败均先解决失败；不得继续汇总不完整矩阵。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.gr_serving.src.echo_chunks_report \
  --screen "$echo_screen" --runs "${echo_first_runs[@]}" \
  --diagnostics "${echo_diagnostics[@]}" \
  --output-dir "$echo_publication/first_pass"
```

使用主仓库当前报告工具。frozen04 原工具没有新 `measurement_identity` / review
支持；不为生成报告改动 frozen04。无 review 的中间汇总按工具契约保持 default gate
pending；已发布的最终报告为 reviewed。

## 2. 独立完整轨迹重复（已完成）

首轮描述性领先者为 C1024，随后顺序完成 C1024/W1024 与 C2048/W2048 独立重复。
两轮退出状态见 ledger 的 `required_repeats=96754 exit 0`。C256 未领先且未作为拟选值，
原流程中的条件分支未触发，不是待补任务。每个实际 run 都运行四方案、两次预热及完整
32 请求，不只重复 ECHO。以下保留两条已执行命令与原条件分支。

实际运行沿用上面的外部 helper，由 owner 顺序调度。

```bash
echo_formal_case 20261003_echo_gr_chunks_repeat_01_deployment_c1024 1024 1024
echo_formal_case 20261003_echo_gr_chunks_repeat_01_deployment_c2048 2048 2048
# Only if C256 is the leading or proposed selected configuration:
# echo_formal_case 20261003_echo_gr_chunks_repeat_01_deployment_c256 256 256
```

每个外部 helper 的退出状态和 `external_validation` 收据已分别记录，核对后将路径加入
`echo_all_runs`。全部接受的重复均保留，不因偏慢而删样本。C2048 的两协议复用
是同一个物理 run，不能增加独立重复数。

```bash
echo_all_runs=(
  "${echo_first_runs[@]}"
  "$echo_output/data/20261003_echo_gr_chunks_repeat_01_deployment_c1024"
  "$echo_output/data/20261003_echo_gr_chunks_repeat_01_deployment_c2048"
)
# Append only after the conditional C256 repeat was accepted/published:
# echo_all_runs+=("$echo_output/data/20261003_echo_gr_chunks_repeat_01_deployment_c256")
```

下列 C256 诊断与导入是未触发的历史条件分支：仅当它进入最终候选时，在正式计时全部
结束后执行。本次最终选 C1024，C1024/C2048 诊断已齐备，无需补跑这段命令。

```bash
(
  cd "$echo_frozen"
  TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs CUDA_VISIBLE_DEVICES=0 \
    bash experiments/gr_serving/scripts/echo_diagnostics.sh \
    20261003_echo_gr_diagnostics_c256_04 --chunk-size 256 --workspace-query-tokens 256
)
```

若实际执行了 C256，记录 wrapper/source guard 后按 `--kind diagnostic` 导入：

```bash
: "${ECHO_C256_DIAGNOSTIC_EXIT_RECORD:?Fill the observed C256 diagnostic exit-zero record}"
: "${ECHO_DIAGNOSTIC_SOURCE_SHA:?Fill the verified diagnostic source SHA256}"
: "${ECHO_POSTRUN_FREEZE_RECORD:?Fill the observed diagnostic postrun frozen-source verification}"
echo_c256_diagnostic=20261003_echo_gr_diagnostics_c256_04
.venv/bin/python /tmp/cxldsagr_import_accepted_run.py \
  --kind diagnostic --run-id "$echo_c256_diagnostic" \
  --source-output-root "$echo_frozen/experiments/gr_serving/output" \
  --destination-output-root "$echo_output" \
  --expected-source-sha256 "$ECHO_DIAGNOSTIC_SOURCE_SHA" \
  --wrapper-exit-zero-confirmed-by "$ECHO_C256_DIAGNOSTIC_EXIT_RECORD" \
  --postrun-source-verification "$ECHO_POSTRUN_FREEZE_RECORD" --copy
echo_diagnostics+=("$echo_output/data/$echo_c256_diagnostic")
```

其 wrapper 保存全部 128 方案/请求输出，但只观察请求 0/15/16/31；不要将诊断时间
加入正式延迟。实际 batch 数以 audit 为准，原 `audit.json` 不覆写。
如新增重复改变领先配置或波动判断，先补齐必要的重复/诊断，再生成最终事实集。

## 3. 汇总、默认 review 与正式报告（已发布）

最终报告使用下列相同的七个正式 run 与两个 diagnostic 输入；生成 session 25659
exit 0。默认 review 已完成四项取舍评估，选定 C1024/W1024。下列流程保留供复现，
重建时须改用尚不存在的输出目录。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.gr_serving.src.echo_chunks_report \
  --screen "$echo_screen" --runs "${echo_all_runs[@]}" \
  --diagnostics "${echo_diagnostics[@]}" \
  --output-dir "$echo_publication/all_traces"
.venv/bin/python -m experiments.gr_serving.src.echo_default_review \
  --summary "$echo_publication/all_traces/summary.json" \
  --memory-summary docs/agents/system/memory_acceptance_summary.json \
  --output docs/agents/system/echo_cache_default_review.json
```

执行者已审阅全部轨迹，填写 review 的 `selected_chunk` 和四项 `assessments`，设置
`status=reviewed`。这属于证据判断，不另设用户审批。review 已说明：重复波动是否需要更多
样本；冷构建与复访的取舍；C/workspace 对实际 session 容量的影响；结果只适用于本
硬件、固定 NH/P/预算和受控合成轨迹。差异落在波动内时按计划保留较小 workspace，
不宣称全局最优。C1024、C2048、选值和 descriptive fastest 都需至少两条完整轨迹。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.gr_serving.src.echo_chunks_report \
  --screen "$echo_screen" --runs "${echo_all_runs[@]}" \
  --diagnostics "${echo_diagnostics[@]}" \
  --default-review docs/agents/system/echo_cache_default_review.json \
  --output-dir "$echo_publication/reviewed"
```

已核对 reviewed gate、精确 run IDs、source/GPU/workload/backend 身份、每协议的全部
样本及两套 latency/capacity 表。内存 review 绑定四组已接受工程证据和保存的非测试
runtime source；报告层只允许已核实的 generator context-limit 元数据差异，并验证
完整请求内容身份，保留原 memory/formal 聚合 hash。详见
[workload 绑定记录](echo_memory_formal_workload_binding.json)与
[验收记录](echo_cache_publication_execution.md#验收与报告层身份绑定)。修复的两组定向测试
按执行 agent/root 交接记录分别为 37 项、20 项通过；corrected auditor 的 111 项 CPU
回归见恢复记录。本次只引用这些结果，没有新增 runtime 变化。若事实集更新，
原 review 作废，保留依据并重新生成，不手改绑定的 hash。

生成的 PNG/SVG 已审阅，九个报告文件已复制到 `report/echo_chunks/`，实际工具快照保存在
`$echo_publication/reviewed_report_tools/`，并有 `reviewed_report_tools_snapshot.json`。
以下原复制模板保留为操作说明；最终发布身份以 provenance 和 ledger 为准。

```bash
mkdir "$echo_publication/report_tools"
cp --parents \
  experiments/gr_serving/src/echo_chunks_report.py \
  experiments/gr_serving/src/echo_default_review.py \
  experiments/gr_serving/src/echo_diagnostics.py \
  experiments/deepseek_v32_echo_prefill/src/chunk_sweep_report.py \
  "$echo_publication/report_tools/"
mkdir "$echo_publication/report_tools/external_validation"
cp "$echo_tools/audit.py" "$echo_tools/formal_runner.py" \
  "$echo_publication/report_tools/external_validation/"
mkdir experiments/gr_serving/report/echo_chunks
for echo_asset in summary.json traces.csv diagnostic_counts.csv diagnostic_phases.csv \
  results.md default_review.json report_provenance.json chunk_latency.svg chunk_latency.png; do
  cp "$echo_publication/reviewed/$echo_asset" experiments/gr_serving/report/echo_chunks/
done
```

核对 copied tools 与 `report_provenance.json` 的四个 hash，保留完整 publication
复现目录。另核对 external-validation 工具与每个 run 的 `independent_audit.json` /
`external_validation.json` 所绑定的 hash，保存实际修正后的 auditor；原 runtime
snapshot 中的旧 auditor 只记录当时的文件身份。README 使用新 `report/echo_chunks/` 的相对链接；忽略的 `output/` 路径
只写普通代码，不建链接。report 生成没有新增性能样本。

## 4. GR README 替换范围与 scoped cleanup（已完成）

执行 agent 已按下表完成 `experiments/gr_serving/README.md` 的正式发布更新：

| 原段落 | 发布后的内容 |
| --- | --- |
| 开头 ECHO pending、2026-10-02 baseline 审计、失败 sequential 补测说明 | 新受控 64K 矩阵的 run IDs、source identity、默认值及取舍；短述旧 DeepSeek 范围已结束。移除“问题来源尚未确认”等过时判断。NOSA 的独立补测/预算限制原样保留。 |
| 原双模型 168 cases / 3552 请求 / 2664 非 HBM 比较总计 | 明确旧 NOSA retained subset 为 84 cases、1776 请求/数值行、1332 非 HBM 比较；4K/16K/64K 分别 804/804/168 请求，不能与新 DeepSeek 计数混合。 |
| 模型范围及旧 per-session DeepSeek slots、清空/腾位的解释 | 当前共享 host arena、每层 P、workspace、精确 recall 与候选 truncate；十个独立 source-input-replay block 和 LM head 边界保持。非 GR 三层数值验收另列，不称 61 层通过。 |
| 主运行入口、参数和调用模块 | 当前 `/preset-models`、`echo_chunks.sh`、固定/部署协议、实际 run 及重建命令；旧 run 的历史命令原样保留且标历史，不能作为当前 CLI 示例。 |
| 4K/16K/64K 的 DeepSeek 延迟表、排名及复访解释 | 完整移除 DeepSeek 子表及对应文字；保留 NOSA 原数字、来源和局限，新 DeepSeek 64K 数字只进入新章节。 |
| 三组 cache 表、旧 DeepSeek 容量和权重/activation 总计、16K HBM 淘汰表 | 提取 NOSA 行和列；移除旧 DeepSeek 容量、淘汰总计、排名及开销归因。NOSA 申报容量不升级成硬预算验收。 |
| 各长度 metadata/audit/source、summary 和 request 图 | 指向 prepared03 的 NOSA 子集文件；说明 `retained_subset` 是从原 run 提取、原 source 未重签、没有新测量。每长度 28 cases，非 HBM 比较为 603/603/126。 |
| NOSA layer-31 profile | 保留全部三个 run IDs、原 metadata/source、page/stripe window 定义、未达 90% 和实际采样边界；只说明 copied formal metadata 的 NOSA 提取身份。 |
| 最后 ECHO pending 入口及旧 pinned-budget 修正状态 | 新 fixed/deployment 表、完整轨迹与首访/复访 hit/rebuild/p95、实际 retained sessions、shared/session/hard-cap 字节、诊断覆盖、review 和工程内存证据；注明未观察请求不是零流量。 |

NOSA 三组旧进程总历时属于原混合双模型运行；不得改称 NOSA-only 执行时间。64K 的
16 个空复访 summary groups 继续保留 null/n=0。原 workload/热度局限、长上下文
override 和输出边界不能因移除 DeepSeek 而消失。独立 MFU/nonmatrix 的四个报告目录
及五个 run 完全保留。

新报告与 README 可核验发布后，已按
[ledger 字段和检查](echo_cache_publication_execution.md#最后安装与-readme-更新)
写入[发布 ledger](echo_cache_publication_ledger.json)，记录发布文件、正式审计、gate
证据与最终配置，再执行既有 prepared03。以下是已完成的命令，不重复 apply：

```bash
.venv/bin/python -m experiments.gr_serving.src.retain_nosa_subset \
  --verify /mnt/ssd-wlcb/chenkaiqi/.cache/echo-nosa-subset-publication-03
.venv/bin/python -m experiments.gr_serving.src.retain_nosa_subset \
  --apply /mnt/ssd-wlcb/chenkaiqi/.cache/echo-nosa-subset-publication-03 \
  --publication docs/agents/system/echo_cache_publication_ledger.json
```

prepared03 的实际安装为 **root session 35272 exit 0**，写入 **93 个文件**，删除
**486 个 DeepSeek-only 文件**，核验 **1,590 个 retained NOSA 文件**。
`receipt.json` 的外部路径与身份已加入 ledger，清理状态为 `applied`；独立非矩阵/MFU/NCU
交付和原 NOSA source/profile 边界保留。README、正式 review 及报告已发布，不再待安装。

收口只更新本 publication 文档，核对本地链接和空白；不重跑 GPU、raw audit、hash 或
报告生成。研究状态表仍由 Supervisor 维护。
