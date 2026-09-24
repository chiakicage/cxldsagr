# 项目约定

本项目研究通用 sparse attention offloading，选取 DeepSeek V3.2 和 NOSA 验证方案。
后续开发优先 NOSA，主要平台为 SM90 / Hopper；保留现有 DeepSeek V3.2 / SM120 实验。

## 目录与职责

- 自有硬件算子放在 `operators/sm90/`、`operators/sm120/`；共享第三方库放在 `3rdparty/`。
- 模型结构、稀疏选择语义、KV 表示和运行适配放在 `model_run/deepseek_v32/`、
  `model_run/nosa/`。接入另一架构时复用对应模型目录，不按架构复制模型树。
- `GR/` 保存共享请求内容、用户热度和调度工具。DeepSeek 专用的 GR checkpoint /
  indexer 实验保留在 DeepSeek 模型目录。
- 不将 DeepSeek V3.2 的 656 B packed MLA record、indexer 或 tokenizer 作为通用
  offloading / NOSA 的固定假设；新增模型时显式适配其稀疏访问语义和 KV 布局。
- 修改目录时同步更新 Python 导入、`-m` 子进程入口、构建路径、配置和文档链接。
  保留 `models/`、`GR/generated/`、`docs/` 的现有数据和报告路径；不删除历史实验文档。

## 脚本与环境

- 暂时不把 cxldsagr 做成 Python 包：保留 `[tool.uv] package = false`，不新增根包、
  安装入口或打包配置。`model_run/` 与 `operators/` 保持源码目录，不为目录整理添加
  `__init__.py`；算子子项目保留自己的扩展包及导入名称。
- 实验命令默认从仓库根目录运行。保留已支持的直接脚本入口；跨目录脚本使用
  `python -m model_run.<模型>.<脚本>`，无需安装仓库。
- 用 `pyproject.toml` 和 `uv.lock` 管理环境，依赖改动同步维护两者。当前 SM120 扩展
  放入显式 `sm120` 依赖组，基础环境不默认安装；未接入后端前不虚设 SM90 安装组。

## 第三方依赖

- `3rdparty/DeepGEMM` 使用上游 `nv_dev` 分支的 Git 子模块，固定提交由父仓库记录。
  当前为 `b64107f`（2.8.0）；不再把整个 DeepGEMM 源码复制进架构算子目录。
- 共享依赖为 `3rdparty/cutlass/`（`f3fde583`）与 `3rdparty/DeepJIT/`（`e5bdee2`），
  与 DeepGEMM 一起作为顶层三个子模块维护；不要新增嵌套的重复源码副本。
- 使用 `python3 scripts/prepare_3rdparty.py --init` 准备依赖，再执行 `uv sync`。
  该脚本只初始化顶层子模块，在 DeepGEMM 的 `third-party/cutlass/` 和
  `third-party/deep_jit/` 下将 `include` 链接到顶层共享源码，并关闭嵌套子模块初始化，
  不修改上游版本化源码。避免 `git clone --recursive` 或递归更新子模块；已递归初始化
  的 checkout 会被准备脚本拒绝，需先按诊断处理，不能与共享链接布局混用。
- `3rdparty/EzKernelKit/` 仅保留为本地未跟踪参考，不纳入当前依赖。其 CUTLASS
  提交与 DeepGEMM 不同，未经适配验证不能强行合并依赖。
- 第三方依赖的初始化、链接和构建遵循 `3rdparty/README.md` 及现有配置；修改版本、
  分支或布局时同步更新子模块和相关引用。不要将可选本地参考 checkout 的存在
  视为已经接入项目的后端。

## 文档与验证

- 给 agent 的目录维护、实现职责、依赖管理和后续开发约定写在 `AGENTS.md`。
  README 保留项目介绍、实际状态、入口索引和可用命令，避免重复维护约定。
- 保留历史报告的平台、依赖和测量含义；目录迁移或上游源码支持某架构不等于
  已完成 GPU 验证。明确区分已运行的验证、静态检查和未验证路径。
- 历史测量报告里的命令行和依赖路径（例如已移出仓库的 `gpu-benches/`）保留原样，
  它们记录当时的运行方式；只修正导航性链接和「当前文件在哪」的指向。
  被 `.gitignore` 排除的运行产物（如 `docs/model_extend_v32_*.json`、
  `docs/extend_step_profile/summary.json`）在文档里写成普通代码路径，不做链接。
- 按改动选择验证：路径迁移检查导入、子进程入口和文档链接，并运行可用的 CLI /
  现有测试；算子或数值改动运行相应硬件和正确性测试。缺少依赖或 GPU 时如实记录。
- Python 格式遵循根 `pyproject.toml` 的 Ruff 配置；第三方和独立算子子项目遵循
  自己的格式与构建设置，避免无关的批量格式化。
