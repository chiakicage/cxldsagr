# GR fixed C256 的后置 auditor 修复与恢复

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

> 迁移定位：本轮实验已独立到 [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md)。
> 下文执行历史中的原命令、源码路径和哈希保持原样；产物现位置通过
> [迁移记录](echo_cache_experiment_migration.md)回查，目录迁移不构成新的测量。

2026-10-03。`20261003_echo_gr_chunks_01_fixed_c256` 已在原 frozen04 runtime 完成
四方案、128 条请求；collector 保存 `status=accepted` 和完整结束事件，并通过原
源码/后端身份检查。首轮 `echo_chunks.sh` 随后因独立 CPU auditor 退出 1 而停止。
错误为 HBM case 的 `fixed host backing capacity: got 0, expected 40`：审计把已计入
shared reservation 的 40 B CPU execution scratch 误归为常驻 host backing。
这次修复仅改变独立 auditor，不改变模型、cache、算子、测量逻辑或接受阈值。

## 接受依据和身份

- 原测量 run ID、时间、参数及 source SHA 保持：
  `47585a615f0bb017f531118e8c8c180cb33cb14d63de447cd83c84966210a211`。
- 原冻结树为 `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-gr-tools-04`；
  root 的 session 66704 以 exit 0 核对 749 个 overlay 文件未变。
- 原 stage
  `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/gr-serving-20261003_echo_gr_chunks_01_fixed_c256.XsCqEw`
  保持原样。原 wrapper 仍记录 exit 1；没有将失败 wrapper 描述为成功。
- 修正版 auditor SHA 为
  `775e0d4b4b15ece3dd1fdbe5f6b8b612cf43a7684b7f3a6a006e597ed79a4b54`。
  Admission 完成 111 项 CPU 回归及 Ruff，并在 session 24529 完成一次完整 CPU 重审，
  exit 0：4 cases、128 requests/correctness、96 次非 HBM exact comparisons，
  32 份 resident reference 与 128 份完整 hidden/logits 均被核对。
- Accepted audit SHA 为
  `c315b5758ad2ccbad4c784e8a0aaaf6bbc9e5cc472444528cbd76725a3f3a4da`。
  原审核 bundle 位于
  `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/auditor-repair-20261003-fixed-c256-01/`，
  含实际 auditor、JSON、stdout/stderr、command/cwd/exit 和 repair provenance。

## 恢复发布

外部 helper `/tmp/cxldsagr_echo_formal_runner.py` 的 SHA 为
`01628f9ef5fd2f3a3724a38084dba565ed712296c92cc576558bd33524c66ef7`。
它在新外部 staging 创建副本，采用上述已经完成的 accepted audit，没有重新执行 GPU
或再次重开全部张量。核对原 stage 的 1,564 个文件，逐字节保留除两项失败 auditor
日志外的 1,562 个文件；复制前后原 inventory 相等。保留文件 inventory SHA 为
`1fecb9b54e15b45d2e155a17cc3c9b415cdf13259d8e45c4596beb9507f96997`。

发布副本添加修正版 `independent_audit.json`、实际成功 auditor stdout/stderr 和
`acceptance_tools/`，原 collector stdout/stderr 也逐字节保留。失败 auditor 日志
没有作为成功运行日志复制，其原 hash 和路径保存在 receipt；原 stage 本身未修改。
Linux `renameat2(RENAME_NOREPLACE)` 先安装 log/profile，再原子提交 data；中途失败
只回滚本次移动，不覆盖已有目标。helper session 24023 以 exit 0 完成恢复发布。

主仓库的 accepted data 为
`experiments/gr_serving/output/data/20261003_echo_gr_chunks_01_fixed_c256/`：
`external_validation.json` 记录原 wrapper exit 1、新审计 exit 0、完整文件 inventory、
独立 auditor/helper 身份及 `collector_execution=null`，表明恢复时没有重新运行
collector；helper 的结束事件也明确为 `collector_executed=false`。
原 `metadata.json`、source manifest/snapshot、测量、reference 和数值
张量没有重签或改写。

10 项临时生命周期检查在实际 SSD staging_root 所在挂载上通过，耗时 0.88 s，覆盖
NOREPLACE、data 最后提交、中途失败回滚、旧目标保护、原字节保持、接受审计复用、
stale evidence / auditor pin / symlink 拒绝，以及 collector/auditor 失败不发布。
临时测试代码为 `/tmp/test_cxldsagr_echo_formal_runner.py`；这些属于工具验证，
不是论文实验或性能样本。

## 继续剩余配置

不修改 frozen04 的旧 `run.sh` / `audit.py`，也不使用关闭审计的旧 wrapper 提前发布。
续跑使用已归档的
`experiments/gr_serving/output/data/20261003_echo_gr_chunks_01_fixed_c256/acceptance_tools/formal_runner.py`
及同目录 `audit.py`：前者从 frozen04 cwd 调用原 `-m measure` 到新外部 stage，
后者用 `--repo frozen04` 验证全部保存证据和原源码；二者实际退出成功后才原子发布。
每条新 trace 仍保存同一 formal source SHA，外部工具身份另存 `acceptance_tools/`
和 `external_validation.json`，直接发布到主仓库，避免重复复制一份冻结树 output。

剩余首轮四配置是 fixed C1024/W2048、fixed C2048/W2048、deployment C256/W256、
deployment C1024/W1024；首个 C256/W2048 不重测。完整命令和后续独立重复见
[发布命令清单](echo_cache_publication_commands.md)。原 sweep/diagnostics 导入收据
保持原边界；本 formal 恢复不冒用它们要求的 wrapper exit 0。

最终 publication 的 `report_tools/` 除四个报告模块外，还须保存实际修正 auditor 与
外部 helper，并核对每条 `independent_audit.json` / `external_validation.json`
绑定的 hash。原 runtime snapshot 中保留旧 auditor 是原测量文件身份的一部分，
不能拿它代替此次真正执行的修正版审核工具。
