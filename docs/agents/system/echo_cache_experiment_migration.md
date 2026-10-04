# ECHO cache 独立实验迁移

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

日期：2026-10-03。按用户要求，将已完成的 ECHO cache 与 chunk 配置对照独立为
[`experiments/deepseek_v32_echo_cache/`](../../../experiments/deepseek_v32_echo_cache/README.md)。
这是目录迁移，没有新增 GPU 测量，没有改变既有 run ID、数值结论或性能数据。

## 实验范围与入口

新实验包含真实 checkpoint 第 0–2 层顺序传播的 64K+128 chunk 扫描，以及十个独立
dense block 输入回放替身的 16 用户两轮 GR 对照。真实前三层独立空 cache 的 64K+1K
正确性验收仍是工程门禁，与性能扫描和 GR 替身分别说明。权重取自 `/preset-models`。

| 原位置 | 当前归属 |
| --- | --- |
| `deepseek_v32_echo_prefill/src/chunk_sweep.py`、`chunk_sweep_report.py`，对应脚本和测试 | 新实验的 `src/`、`scripts/chunk_sweep.sh`、`tests/` |
| `gr_serving/src/echo_chunks_report.py`、`echo_default_review.py`、`echo_diagnostics.py`，对应脚本和测试 | 新实验的 `src/`、`scripts/run.sh`、`scripts/echo_diagnostics.sh`、`tests/` |
| `deepseek_v32_echo_prefill/report/chunk_sweep/` | 新实验的 `report/chunk_sweep/` |
| `gr_serving/report/echo_chunks/` | 新实验的 `report/echo_chunks/` |
| 上述扫描、七轮正式 GR、两轮诊断、四组内存门禁及报告生成的运行目录 | 新实验的 `output/{data,log,profile}/`，原 run ID 不变 |

`gr_serving` 的通用测量、请求生成、审计和内存检查工具继续由新实验显式调用，不复制模型
计算逻辑。旧 NOSA 实验与独立 DeepSeek MFU、非矩阵算子及 NCU 报告保留在原实验。
原目录不保留已迁专用入口的兼容壳，不新增 `__init__.py` 或根包。

## 迁移与原始证据

机器可读的位置映射（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/artifact_locations.json`）
按原目录精确记录当时位置；迁移记录（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/directory_migration.json`）
记录每个目录的文件数量、字节数及额外校验的原 SHA256。

共迁移 **45 个目录、34,753 个文件、50,539,345,137 bytes**。所有目录采用同文件系统
`rename`，逐项比较迁移前后的相对路径、文件类型、大小、mtime、inode、device 和
symlink target，均一致。55 份报告素材及关键验收文件另外核对 SHA256，也全部一致。
未复制大体积张量或 profiler 数据，没有使用 `/tmp` 暂存这些产物。

完整 stat 清单与迁移复现材料保存在
`experiments/deepseek_v32_echo_cache/output/data/20261003_echo_cache_directory_migration/`。
其中 `original_documentation/` 保留迁移前两个实验 README 和发布 ledger；原 GR README
的 SHA256 已与发布 ledger 核对，原值为
`154f7192d1a4d60c51699a30f320048a6b03e0bd774a7eccafedfc13d5f71014`。

原 metadata、source snapshot、命令 invocation、memory summary、default review、
发布 ledger 和清理 receipt 不改写。它们的旧目录名描述当时执行位置；当前文件位置通过
上述映射解析，源码身份仍按原快照验证。报告工具通过 `--locations` 接收位置映射，
不因此放宽原文件哈希、运行身份、请求内容或内存门禁检查。

当前可运行命令见新实验 README；执行材料中保留的原命令用于解释既有测量，不能直接
将路径迁移或 CLI 检查当成新的性能结果。原 C1024/W1024 选择只适用于已声明的
H200/SM90、64K+128、16 用户两轮、4 GiB HBM / 64 GiB DRAM 控制点；同配置 ECHO
比 serial sparse 慢 15.08% 的结论保持不变。

## 验证

目录迁移和验证已完成：

- 迁移相关测试 127 项通过；通用 GR 测试 367 项通过、4 项 GPU opt-in 跳过；
  补充入口路由及源码指纹检查 19 项通过，后者包含前述套件中的定向复查，不合并计数。
- 9 项 CLI 帮助、脚本语法、Ruff 及代码/文档的定向 `git diff --check` 通过；
  活跃 Python/shell 代码中无已迁入口的旧导入或旧脚本调用。
- 已发布 summary 与原 default review 的真实 `apply_review` 通过，四组内存证据
  经位置映射读取，保持原哈希与 C1024 的 reviewed 状态。
- README 导航检查通过；15 份内部文档的 178 个本地文件链接、2 个标题锚点全部有效。
- 最终复核 45 个原目录已迁出、当前目录在位，55 个报告/关键验收文件与 3 份原文档
  快照的 SHA256 均保持一致。`/tmp` 所在文件系统仍有 36 GiB 可用。

此次没有重跑 GPU 测量。其他并行实现改动在新 README 中保留其“改动后未运行”状态，
不能将本次迁移验证用于证明这些运行时改动的数值或性能。
