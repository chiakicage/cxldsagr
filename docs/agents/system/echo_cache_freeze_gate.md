# ECHO 新源码冻结与验收入口

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

> 迁移定位：本轮实验已独立到 [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md)。
> 下文执行历史中的原命令、源码路径和哈希保持原样；产物现位置通过
> [迁移记录](echo_cache_experiment_migration.md)回查，目录迁移不构成新的测量。

2026-10-03；执行准备材料。旧冻结的数值结果见
[chunk 数值诊断](echo_chunk_numerical_screen.md)，不能替代当前代码验收。

## 本轮已完成验收

新冻结 `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-shared-chunks-02` 已创建：
真实 detached worktree，HEAD `1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c`，745 个主树实际
文件在复制结束时逐字节一致，overlay SHA256
`ab721f90a3d09bb07e8ec9624552be12a77f4791390274573ec5ceb33b5c5b0b`。
26 个预加载仓库模块全部导入自冻结目录；这只是预加载集合，不宣称涵盖全部运行期动态导入。
第一次数值执行前的创建尝试因主树并发修改失败，未启动 GPU；该未验收副本已删除，
上述第二次创建完成时没有源文件变化。主树其他正在开发的路径不因此获得运行验收。

GPU 0 上的真实第 0–2 层、独立空 resident/offload cache、64K+1K gate：
**1 passed，28.60 s**。全部 `[1024,7168]` extend hidden、prefix/extend 末 logits
`[1,129280]` 逐位一致，max absolute error 与 relative L2 均为零。
1213 文件的执行前后 gate SHA256 均为
`428240214945b379f0c3016b845b906d6b5924f236abe5c9b1b05fa21b84ab25`，`changed=[]`。
执行后 `--verify` 再次通过，745 个复制文件、官方依赖、实际导入与已安装 backend 身份均未变化。
独立 sweep 的 1228-source SHA256 为
`b727c720864641bc94a7a2cb4cfa69c1c1ddae2831e370feed882a043a716b6d`。

上述 gate 自身不推导跨 C 数值可行集合或性能结果；完整 sweep 已在后续 freeze 04
以 `20261003_echo_shared_chunks_02` 重新执行并验收，见下节正式 sweep 记录。
发布审计已加入全部 requested C 的独立 CPU 资源规划复算；45 项 sweep/provenance CPU 检查
通过，包括删除可行 C 的全部证据并伪造预算排除、账本少记以及规划维度篡改等负例。

Root 在同一冻结树、GPU 1 上完成所选 DeepSeek 模型/算子与共享 cache 回归：
**279 passed、1 skipped**（session `63696`，exit 0）。唯一 skip 是三层 checkpoint opt-in，
已由上述独立 GPU 0 gate 实际通过。参数化 linear 测试采用测试 fixture 隔离 Dynamo 编译
缓存，避免不同测试形状累计触发 per-code recompile limit；没有为通过测试修改 runtime。
此回归与三层 gate 都是正确性检查，不计作论文性能测量。

Root 后续全局 CPU 回归在 **main** 完成：**2338 passed、834 GPU/opt-in skips、
58 subtests，63.88 s**（session `31915`，exit 0）。该 main 的 DeepSeek runtime 与 02
冻结版本一致，但这不是冻结树上的全局 CPU 运行。前一轮全 CPU 的唯一失败来自并行 NOSA
开发中新加入的 Frozen GC 检查误拒，已由该开发者修复；Root 没有代改 NOSA，最终结果以上述
新运行替换。GPU 1 回归与三层 checkpoint gate 的源码身份仍是 02 freeze。

freeze 02 gate 完成时，正式 sweep 仍等待 C=1024 与固定 workspace C=2048 内存审计，
以及所有 GPU 检查结束后的计时调度。后续内存验收以
[预算审计记录](echo_cache_memory_audit.md) 为准，不以本段旧时点描述替代其状态。
后续正式 sweep 的实际验收状态见下节，不由 gate 或 memory observer 的用时替代。

## freeze 04 与 GR 诊断验收

后续仅同步 GR 诊断、报告和内存 observer 工具，创建独立冻结
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-gr-tools-04`，仍为上述 detached HEAD，
包含 749 个复制源文件，overlay SHA256 为
`d5c7de6e8dc7b577ac69322cbb3d219510ec116bf71b18edb04dd8b347d77b0a`。
gate 的 1213-file SHA256、sweep 的 1228-file SHA256 及已安装 backend 与 freeze 02
相同，不因纯 observer 修复重复宣称新的 runtime 验收。freeze 03 的 observer 只包装
backend 方法，漏掉 persistent pool 预先捕获的 callback，首请求的阶段覆盖检查失败；
该次数据未发布。freeze 04 包装并恢复 pool 实际 allocation/release callback，新增真实
`PersistentGRRunner` 的生命周期回归。

C1024/W1024 诊断 `20261003_echo_gr_diagnostics_c1024_04` 在 H200 GPU 1 完成：四方案
各执行全部 32 请求，**128 个完整 candidate hidden / 末 token logits 均与 HBM 逐位
一致**；仅采样 `[0,15,16,31]`，共 16 个方案/请求、7,840 个实际逐层批次。
采集进程和独立 CPU 重开产物审计均通过，wrapper exit 0。该诊断的完整 source manifest
为 1390 文件，SHA256
`60fc2e1cbd26155a6d347d3a817910c10ed11d8e0b3af21b3dde06ed93025662`。
结束后 `--verify` 再次通过：749 个复制文件、26 个预加载导入、gate/sweep source 与
backend 身份不变。

C2048/W2048 诊断 `20261003_echo_gr_diagnostics_c2048_04` 同样完成，root 确认
session `82252` exit 0，采集及独立 CPU 审计均通过。四方案各 32 请求的完整输出均与
该配置 HBM 逐位一致；16 个采样方案/请求均实际重建 prefix，每个 330 批次，共
5,280 个逐层批次。C2048 的 ECHO/serial_sparse 采样复访属于 miss，不能改记为首访。
两配置的 1390-file 诊断源码身份与完整工作负载身份相同，验收不外推为跨 C 数值相等。
C2048 完成后再次执行 freeze 04 `--verify`，749 个复制文件、26 个导入及上述
gate/sweep 源码身份仍全部通过。两诊断的原 data/log/profile 已逐字节导入 main，分别
核对 9,369 / 6,808 个原文件；独立 receipt 保存前后 inventory，不改原 metadata/audit。

采集时有外部 NOSA 进程也使用 GPU 1，已记录到该 run 的
`output/log/<run_id>/execution_context.json`。该执行仅验收完整输出及采样机制证据，
不用于正式延迟、overlap 或峰值内存结论；两诊断 GPU 已释放。原产物从 freeze 04 导入 main
时逐文件核对 hash，不改原 metadata/audit，另存 import receipt。具体 batch 覆盖、
边界及失败尝试见 [GR 诊断记录](echo_gr_diagnostics_checkpoint.md)。

## GR fixed C256 的测量完成与审计器失败

五配置 GR matrix 的首个 run `20261003_echo_gr_chunks_01_fixed_c256` 已完成 C256 /
workspace 2048 的四方案、各 32 请求；matrix wrapper session `44142` 随后 **exit 1**，
后四配置当时没有开始，不能把该 wrapper 表述为已经成功。旧轻量 monitor 已停止。
原数据仍在外部 staging：
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/gr-serving-20261003_echo_gr_chunks_01_fixed_c256.XsCqEw`，
原产物及 freeze 04 未修改。后续经修正版 external auditor 验收并发布恢复副本，见本节末。

采集器写入 `accepted` 之前，已执行完整 case 检查、测量前后源码核对和 installed
backend 身份守卫。其独立 audit 随后在第一条 HBM 请求的固定 host backing 断言失败：
`('deepseek_v32', 'hbm', 16)/0 fixed host backing capacity: got 0, expected 40`。
失败发生在 `audit_case_rows`：旧条件强制 DRAM retained allocation 等于总 reservation，
没有扣除明确预留、请求边界已释放的 40 B CPU execution scratch。各方案 metadata
明确记录 `workspace_cpu_bytes=40`，由 indexer 8 B、metrics 24 B、scalar 8 B 组成。

只读完整性复核得到以下结果；它不替代修复后的完整独立验收：

- `measurements.jsonl` 和 `correctness.jsonl` 均为完整 128 行，四方案各 request ID
  0–31，无缺失或重复。128 个在线完整 hidden/logits 比较均记录 exact；128 个独立
  tensor-evidence 路径与 32 个 HBM reference 文件均存在且非空。此次复核未再次解码
  张量，因此**修复前尚无独立 CPU 原始张量 pass**。
- 全部 128 行的 DRAM 差额严格为：total reserved − actual = 40、shared reserved −
  shared actual = 40、request actual − retained actual = 0。HBM 与 DRAM 原有
  `0 <= actual <= request_boundary <= reserve <= cap` 全部成立。修复应验证明确的
  40 B 字段及精确容量等式，不能把固定容量验收放宽成任意 `<=`。
- 测量 source manifest 有 **1389 个文件**，其聚合 SHA256 与 metadata 均为
  `47585a615f0bb017f531118e8c8c180cb33cb14d63de447cd83c84966210a211`，
  保存的 source 文件数量一致。由失败调用路径可确认，旧独立 auditor 已先完成
  `audit_sources` 的 frozen/current 与 snapshot 全文件、覆盖集合、官方 pins、
  installed backend 和 FlashInfer JIT artifact 检查，再进入失败的 case 断言。
  Root 后续 freeze 04 `--verify` 也通过（session `66704` exit 0）：749 个复制文件、
  26 imports、gate 1213 / sweep 1228 身份及 backend 不变；没有额外重复该全量核验。
- 旧 auditor 在失败前也已核对完整有序 measurement/correctness 矩阵及工作负载重建。
  本次只读复核确认全部请求的 workload SHA256 相同，且等于已有诊断的
  `4885ab864bed148f9f4a54230c3dced1c9a3749ac29ca92e13faf1edb019b9d1`。
  每方案仍为 16 次首访、16 次复访；本配置全部 prefix miss，复访没有改记首访。

原小文件 SHA256 为：

| 原文件 | SHA256 |
|---|---|
| metadata.json | `72ab9d63bb376fd5d75e2bc50481a403585de8cb1d0136ed47401b9e73f4a9f2` |
| source_manifest.json | `6ae9a7149965846bcee994e2027a16ee60c7f1913fa5f62d6e076c8953ea3555` |
| measurements.jsonl | `c25375452f6dd4ca2c799d1efee6944cc0082b089185b05d011555fc844cf684` |
| correctness.jsonl | `c55beb237352f78d4650c727919b2e2661eb4500214c032f555497d0f5aa6d46` |

**修复前的有条件恢复判断：无需因这个审计器断言错误重跑 GPU 计时，但当时仍未接受。**
后续接受必须以 admission 修正后的完整 CPU audit 为准，完成所有原始 hidden/logits、
LRU、预算、summary 和来源守卫；遇到其他实际不一致仍须重新判断。新 auditor 是独立
stdlib/torch 工具，可使用 `--repo=frozen04` 核对原测量源码，并单独保存新 auditor
代码/哈希、命令和 external audit 结果。不得修改或重签原 metadata、source snapshot、
时间或 run ID。原 wrapper exit 1 与失败 stderr 保留；恢复 receipt 必须区分原失败和
新 external validator 的成功，不能伪装原 wrapper 已通过。原通用 import helper 的
`kind=gr` 要求源目录已有独立通过结果和 wrapper exit 0，不能直接用于此例而冒填参数；
恢复导入须显式支持 external audit 与新增材料的独立 inventory。

上述条件随后满足：admission 的修正版 auditor SHA256 为
`775e0d4b4b15ece3dd1fdbe5f6b8b612cf43a7684b7f3a6a006e597ed79a4b54`，
111 项 CPU 回归及 Ruff 通过；session `24529` 的完整 external CPU audit exit 0，
核对 4 cases、128 request/correctness、32 份 HBM reference 与 128 份完整 hidden/logits，
96 次非 HBM 比较均 exact，原 1389-file source 身份保持不变。Accepted audit SHA256 为
`c315b5758ad2ccbad4c784e8a0aaaf6bbc9e5cc472444528cbd76725a3f3a4da`。

独立修复 review 对照 freeze 04 核对了 helper 提取及 row 验收的精确扣除，没有修改
runtime 或弱化到任意容量上界。正例覆盖四方案及多用户共享一次预留；负例覆盖缺少
host 字节、shared 字节错误归属、丢失 session 页表、错误/缺失 CPU 分量、错误总和、
未预留及仅凭不明差额推断 40 B。未发现阻塞问题；没有重复运行上述测试或全量张量
audit。只读核验 external audit 小文件、实际 auditor SHA 和 execution exit 0 后，
本次有条件恢复获独立 review 支持。

恢复副本已由单独的外部 helper 发布到 main，原 collector 数据、源码及时间未改写，
原失败 wrapper 记录也未冒充成功。新 auditor/helper 身份、原字节 inventory、原失败
与新成功 audit 的区分见 [恢复记录](echo_gr_auditor_recovery.md)。当前仅恢复这个
完整 C256/W2048 run；不意味着五配置 matrix 已完成，也不选择 serving 默认 C。

## 冻结边界

本轮从主树实际文件创建新的 detached worktree；包含 tracked 与未忽略的 untracked 文件，
保留主树的删除和执行权限，不复用旧 freeze overlay。临时工具
`/tmp/cxldsagr_echo_freeze.py` 默认只做 dry run，`--create` 才创建；工具自身不运行 GPU。
复制前后核对主树 manifest、每个文件的冻结副本，以及官方依赖 HEAD/dirty 状态。

`.venv` 链接原环境，四个官方顶层依赖和已有 `3rdparty/ECHO` 链接主树。
这不是依赖物理副本；使用冻结目录实际导入、官方源文件、安装库和编译参数的身份记录，
并由 gate/sweep 前后守源。FlashInfer 已安装源码身份与执行后 JIT 实物身份分别记录。
禁止在守源运行期间修改冻结源码、依赖或环境。

```bash
python3 /tmp/cxldsagr_echo_freeze.py \
  --source /mnt/ssd-wlcb/chenkaiqi/cxldsagr \
  --destination /mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-shared-chunks-02
```

相关 CPU 边界检查完成后，同命令加 `--create`。已有目标目录会明确失败。
工具生成冻结目录内的 `docs/agents/system/echo_cache_frozen_source.json` 和
`echo_cache_frozen_imports.json`；后者保存实际仓库导入路径、gate/sweep source manifest
及 backend provenance。新分配模块 `cache/host_allocation.py` 必须进入两套 manifest。
`--verify` 核对复制源码、依赖链接，以及首次保存的导入/源码/安装库身份，不刷新基线。
GR observer 仍在修改时，本轮首先只验收 DeepSeek gate；若后续同步纯 GR 工具，须在守源运行
结束后进行，另存新的全源码身份，并核对 gate/sweep/GR 的 runtime 交集完全一致。

## 实际 checkpoint gate

从新冻结目录运行，输出直接进终端，测试不创建实验结果目录：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 DEEPSEEK_ECHO_CHECKPOINT=/preset-models \
.venv/bin/python -m pytest models/deepseek_v32/tests/test_echo_checkpoint.py \
  -q -s --tb=short -p no:cacheprovider
```

必须实际执行 checkpoint 第 0–2 层，resident/offload 各从独立空 cache 构建 64K prefix，
比较完整 1K extend hidden 和末 logits，检查源码前后不变。此 gate 不扩展到 61 层，
也不能替代独立的 10-block GR surrogate 验收。

## 已接受的正式 sweep

在相关预算门禁结束并确认整机无其他 GPU 任务后，root 从 freeze 04 在 GPU 0 执行
下列命令，wrapper session `94098` exit 0。独立 CPU raw-artifact audit 明确接受该
run；source/runtime/backend 前后守卫通过。该入口对**所有预算可行 C**重新做完整
数值筛选；没有沿用旧 C=512 排除，也没有预先重复保存同一轮大体积数值材料。

```bash
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs \
CUDA_VISIBLE_DEVICES=0 ECHO_RUN_ID=20261003_echo_shared_chunks_02 \
bash experiments/deepseek_v32_echo_prefill/scripts/chunk_sweep.sh \
  --model /preset-models --prefix 65536 --extend 128 \
  --chunks 256 512 1024 2048 4096 --sparse-pool-tokens 32768 \
  --hbm-cache-budget-gib 4 --dram-cache-budget-gib 64 \
  --cold-warmups 2 --cold-repeats 5 --extend-warmups 5 --extend-repeats 20
```

SSD 上的 TMPDIR 必须预先存在；完整 hidden/selection 及源码快照体积较大，不能使用空间有限
的 `/tmp` overlay。只有数值通过的 C 才进入 warmup/timing；失败 C 保留原始数值证据与原因。
本轮已接受结果先进入 freeze 04 的实验 `output/`，再经逐文件 byte hash 导入 main，
不修改原 metadata/result；独立 import receipt 不计入原文件 inventory。
导入已完成，1272 个原文件三次 inventory 一致，SHA256 为
`ace72203581e029a657c03bad7a4bf7ac4d6001ef2c2e1af3fa5edcc37b73f19`。

本轮预算保留 `[256,512,1024,2048]`；C4096 因需要 5,108,643,392 B cache 而超过
4 GiB。八个可行 C/mode 的完整数值证据全部保存并被独立重开：所有 within-C 完整输出
与精确选集逐位相同。新 C512 的 cross-C extend hidden 有 1 个元素（`[106,587]`）
超过原 `rtol=0.01, atol=0.02`，其余 C 的完整 extend hidden/logits 通过；因此本轮
计时集合为 `[256,1024,2048]`。每个模式/C 的 cold 2/5 与 fixed-prefix 5/20 覆盖齐全，
共 42 warmup、180 正式样本和 6 个单独诊断条目。预算账本、排除理由、完整采样矩阵、
fresh generation、snapshot restore 及交错顺序均经独立轻量复核。

source 仍为上述 1228-file SHA256；`result.json` SHA256 为
`0933f9ea9b5368c9e79060a29b57f9e0b0217f5d612c0eb92919ada2b11603a2`。
完整数值边界见 [chunk 数值诊断](echo_chunk_numerical_screen.md)。这是实际前三层
standalone 测量，不从其 cold 最低延迟选择 GR serving 默认 C。旧有效实验报告仍保留，
后续由独立的 10-block/16-user 正式 trace 和发布审计决定替换范围。

三层 gate、相关预算门禁及本轮 standalone 筛选/计时审计已经完成；剩余正式交付重点为
10-block/16-user GR 完整 trace、匹配诊断和发布前独立审计。预算各配置的具体门禁状态
仍以预算审计记录为准。前三层 sweep 不选定 serving 默认 C。

## 正式 GR matrix 与重复的轻量监控记录

在上述恢复之后，root continuation session `31401` 完成首轮剩余四个 run：
`20261003_echo_gr_chunks_01_fixed_c1024`、`20261003_echo_gr_chunks_01_fixed_c2048`、
`20261003_echo_gr_chunks_01_deployment_c256` 与
`20261003_echo_gr_chunks_01_deployment_c1024`。它们均已由外部 helper 发布到主工作区，
监控 session `91906` 对四组已接受的小文件逐项核对后 exit 0。
首轮五配置由这四组加上前述单独恢复的 `fixed_c256` 构成；恢复过程原 wrapper exit 1
的事实仍保留，不能被 continuation 的成功覆盖。

root 随后通过 session `96754` 发起独立重复
`20261003_echo_gr_chunks_repeat_01_deployment_c1024` 与
`20261003_echo_gr_chunks_repeat_01_deployment_c2048`。两组均已发布，重复监控
session `59065` 核对通过后 exit 0。重复沿用 freeze 04、GPU 0，以及首轮已归档的
外部 helper / corrected CPU auditor；没有修改冻结 runtime 或重新包装旧测量。

以上六个新执行 run 的小文件核对均满足：四方案各 32 请求，共 128 条 measurement
和 correctness；每方案首次访问 16 次、复访 16 次，复访 miss 仍归复访；全部
`[128,7168]` candidate hidden 与 `[1,129280]` logits 的 exact 标记为 true。
外部 CPU audit 接受 128 份完整输出、32 份 HBM reference 与 96 次非 HBM 精确比较，
collector / audit exit code 均为 0，`runtime_snapshot_rewritten=false`。
监控仅阅读小型 JSON / JSONL 与进程身份，不解码或 hash 原始 tensor，不重做 raw audit，
也不运行报告生成或 GPU 计算。完整数值验证由每个 run 的独立外部 CPU audit 提供。

六组使用相同 measurement source、workload、auditor 和 helper 身份：

- measurement source，1389 files：
  `47585a615f0bb017f531118e8c8c180cb33cb14d63de447cd83c84966210a211`。
- workload：`4885ab864bed148f9f4a54230c3dced1c9a3749ac29ca92e13faf1edb019b9d1`。
- corrected auditor：
  `775e0d4b4b15ece3dd1fdbe5f6b8b612cf43a7684b7f3a6a006e597ed79a4b54`。
- archived helper：
  `01628f9ef5fd2f3a3724a38084dba565ed712296c92cc576558bd33524c66ef7`。

以下为已接受小文件的 ECHO 描述性预览，单位均为 ms。total 使用全部 32 请求的
`math.fsum(latency_ms)`；first / revisit mean 各使用 16 请求，按 `is_revisit`
区分，不按 hit / miss 区分。capacity 取 `independent_audit.json` 的
`case_audit[].admitted_session_capacity`；final users 取末请求 `cached_users`。

| Run ID | C / workspace | Total | First mean | Revisit mean | Hits | Capacity / final users |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20261003_echo_gr_chunks_01_deployment_c256` | 256 / 256 | 227801.668596 | 14163.687762 | 73.916525 | 16 | 16 / 16 |
| `20261003_echo_gr_chunks_01_deployment_c1024` | 1024 / 1024 | 78436.862621 | 4827.457615 | 74.846299 | 16 | 16 / 16 |
| `20261003_echo_gr_chunks_01_fixed_c2048` | 2048 / 2048 | 129513.492560 | 4002.318200 | 4092.275085 | 0 | 15 / 15 |
| `20261003_echo_gr_chunks_repeat_01_deployment_c1024` | 1024 / 1024 | 76981.564481 | 4738.079581 | 73.268199 | 16 | 16 / 16 |
| `20261003_echo_gr_chunks_repeat_01_deployment_c2048` | 2048 / 2048 | 128348.682994 | 3970.365545 | 4051.427142 | 0 | 15 / 15 |

首轮 C2048 的 fixed / deployment 是同一配置、同一 run 的共享控制，不能重复计作
独立样本。这是工程预览，不代替正式报告；默认配置的选择、最终比较和报告发布由 root
统一完成。上述 run 的原始产物位于 `experiments/gr_serving/output/data/<run_id>/`。

### 监控白名单边界说明

session `91906` 末尾曾输出 PID `1006405` 的 `external_gpu_activity`，
并在 `external_gpu_events` 中保留该条目。原始输出不删行、不改写。随后读取该 PID
的实际 `ps` command line，确认它是 root session `96754` 刚启动的
`20261003_echo_gr_chunks_repeat_01_deployment_c1024` collector，执行文件为
freeze 04 的 `.venv/bin/python -m experiments.gr_serving.src.measure`，GPU 为
`GPU-1bdee8b4-22ac-536c-208b-bfb4ed38b878`。

该事件由旧 monitor 白名单只包含首轮四个 run ID 导致，属于新旧矩阵交接时的身份
分类误报，不是真实外部 GPU 干扰，也不影响正式 latency 的有效性。重复 monitor
`59065` 将实际重复 run ID 纳入白名单，PID `1006405` 与后续 C2048 collector
PID `1009237` 均识别为本次正式任务；其 `external_gpu_events=[]`。本次轻量
监控未发现真实外部 GPU 活动，不能把旧误报解释成共享 GPU 干扰证据。
