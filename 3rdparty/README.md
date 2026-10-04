# 第三方依赖

| 子模块 | 版本来源 | 用途 |
| --- | --- | --- |
| `DeepGEMM/` | `main`，`057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7` | DeepSeek 官方 FP8 GEMM / resident indexer 及独立基准 |
| `cutlass/` | 与 DeepGEMM 的 gitlink 一致 | NOSA / DeepSeek SM90 及 DeepGEMM 共用的 CUTLASS / CuTe headers |
| `DeepJIT/` | 与 DeepGEMM 的 gitlink 一致 | DeepGEMM 的 JIT runtime headers |
| `FlashMLA/` | `ba89a3466e9470ad08ab39738d4e7bb66989e1e7` | 官方 Hopper / DeepSeek V3.2 sparse MLA prefill |

从仓库根目录初始化并连接构建所需的头文件路径：

```bash
python scripts/prepare_3rdparty.py --init
uv sync
```

源码只在顶层检出一份。准备脚本将 DeepGEMM 原有 `third-party/*/include` 与
FlashMLA 的 `csrc/cutlass/{include,tools/util/include}` 连接到共享目录。
上游 `.gitmodules` 保持原样，嵌套依赖不初始化；FlashMLA setup.py 的子模块更新由
本地 `submodule.csrc/cutlass.update=none` 阻止，未修改上游源码。
已有源码时可省略 `--init`，重复运行不会覆盖已有文件。切换 DeepGEMM 提交后，
先移除旧的生成目录 `3rdparty/DeepGEMM/build/`，再运行
`uv sync --reinstall-package deep-gemm`，避免上游增量打包夹带上一版本已删除的 headers。
本次已 clean rebuild，并逐文件核对安装后的 859 个 JIT headers 与固定源码一致。

FlashMLA 2026-09-30 主线移除了 Hopper 与 V3.2；这里固定到上游 README 指定的
兼容提交。该提交原 CUTLASS pin 为 `147f5673`，本项目连接现有 `f3fde583` 共享
headers；准备脚本检查消费者、原 pin 和共享 pin 的确切组合，升级须重新验证。
该组合已在 H200 / SM90 上通过 FlashMLA 的 29 项 GPU 测试，以及真实第 2 层输入
与独立 FP32 参考的数值检查；这不是完整 benchmark 的性能验收。
`uv sync` 在基础环境安装两个官方库；FlashMLA 通过 `pyproject.toml` 的 build
variables 只生成 SM90a 代码（`FLASH_MLA_DISABLE_SM100=1`），不引入 SM120 后端。

DeepGEMM 2.8.1 使用 DeepJIT 和 C++20，不再依赖 fmt。需要运行保留的
[legacy DeepGEMM 基准](../experiments/legacy/deepseek_v32/README.md)时，显式执行
`uv sync --group legacy`（保留组名，DeepGEMM 已进入基础环境）。DeepGEMM main 已通过 Hopper FP8 linear / grouped GEMM 与 resident indexer 数值检查；
旧性能记录仍对应旧版本。FP4 路径不在本次验证范围内。
项目自有 SM120 扩展及其安装组已删除；复现其历史结果须使用归档 README 指定的旧 revision。

`EzKernelKit/` 是由 Git 忽略的本地参考 checkout，不是子模块或自动构建依赖。

被 Git 忽略的本地 `ECHO/` checkout 固定为
`bc1b75c1000010d0ac6f032ebaac283255c050b1`，供 DeepSeek GPU policy 差分测试和
可选的 `OfficialDeepSeekServingBackend` 使用。该 backend 运行官方 ECHO 的原始
logits、top-k、allocator 和 recall 实现，实验入口与适配边界见
[官方 ECHO 实验](../experiments/deepseek_v32_echo_official/README.md)。
普通模型入口不加载这套可选依赖。运行上述测试或 backend 时，独立项目 checkout
也须提供这个目录；缺少依赖导致的失败不能计为通过。

ECHO 不加入四个顶层子模块，也不初始化其嵌套依赖。目录尚不存在时，从仓库根目录执行：

```bash
git -c submodule.recurse=false clone --no-recurse-submodules \
  https://github.com/sjtu-zhao-lab/ECHO.git 3rdparty/ECHO
git -C 3rdparty/ECHO -c submodule.recurse=false checkout --detach \
  bc1b75c1000010d0ac6f032ebaac283255c050b1
```

已有目录应先检查 `git -C 3rdparty/ECHO status --short` 与
`git -C 3rdparty/ECHO rev-parse HEAD`，保留上游已跟踪文件原样。
[独立加载器](../operators/deepseek_v32/indexer/official.py)校验 ECHO 和共享 CUTLASS
的提交及已跟踪文件状态；CUTLASS 须为 `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。
加载器为官方源码构建独立命名的绑定，复用顶层 CUTLASS，不安装 ECHO 自带的 DeepGEMM，
也不替换项目环境中的 mainline `deep_gemm`。

首次构建会下载固定为 `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28` 的 fmt 归档，
校验 SHA-256 后提取头文件。fmt、绑定和 JIT 产物保存在
`~/.cache/cxldsagr/echo-official/`；设置 `XDG_CACHE_HOME` 时，缓存根目录随之改变。
加载器记录实际源码、头文件及 native/JIT 产物的身份，实验在计时前完成构建。

项目自有的 SM90 NOSA 与 DeepSeek ECHO CUDA 内核复用顶层 `cutlass/include`，通过项目环境中的
TVM FFI 按需编译；使用 `uv sync`，不要求安装 `legacy` 组或 EzKernelKit。
