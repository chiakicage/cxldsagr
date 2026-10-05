# 第三方依赖规则

本文件适用于依赖版本、布局、构建、准备脚本和环境依赖的修改。
代码路径均相对仓库根目录；同时遵守[根规则](../AGENTS.md)，操作入口见
[第三方 README](README.md)。

- `3rdparty/DeepGEMM` 使用上游 `main` 分支的 Git 子模块，固定提交由父仓库记录。
  当前为 `057ca596`（2.8.1）；不使用 `nv_dev`，不把源码复制进自有算子目录。
  resident indexer 与可匹配的 FP8 GEMM 优先调用官方 DeepGEMM。
- `3rdparty/FlashMLA` 固定 `ba89a346`，这是上游为 Hopper / V3.2 明确保留的提交；
  最新主线已移除这些支持。DeepSeek MLA 调用其官方 sparse prefill API，offload 复用
  相同计算路径，不保留自研 Triton MLA 作为替代 baseline。
- 共享依赖为 `3rdparty/cutlass/`（`f3fde583`）与 `3rdparty/DeepJIT/`（`2efdab4`），
  与 DeepGEMM、FlashMLA 一起作为顶层四个子模块维护；不要新增嵌套的重复源码副本。
- 使用 `python3 scripts/prepare_3rdparty.py --init` 准备依赖，再执行 `uv sync`。
  该脚本只初始化顶层子模块，在 DeepGEMM 的 `third-party/cutlass/` 和
  `third-party/deep_jit/` 下将 `include` 链接到顶层共享源码；FlashMLA 的
  `csrc/cutlass/{include,tools/util/include}` 同样链接到共享源码。关闭嵌套子模块初始化，
  不修改上游版本化源码。避免 `git clone --recursive` 或递归更新子模块；已递归初始化
  的 checkout 会被准备脚本拒绝，需先按诊断处理，不能与共享链接布局混用。
- `3rdparty/EzKernelKit/` 仅保留为本地未跟踪参考，不纳入当前依赖。其 CUTLASS
  提交与 DeepGEMM 不同，未经适配验证不能强行合并依赖。
- 第三方依赖的初始化、链接和构建遵循 `3rdparty/README.md` 及现有配置；修改版本、
  分支或布局时同步更新子模块和相关引用。不要将可选本地参考 checkout 的存在
  视为已经接入项目的后端。
