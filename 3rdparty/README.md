# 第三方依赖

| 子模块 | 版本来源 | 用途 |
| --- | --- | --- |
| `DeepGEMM/` | `nv_dev`，父仓库固定提交 | legacy 独立 GEMM / MQA logits 基准 |
| `cutlass/` | 与 DeepGEMM 的 gitlink 一致 | NOSA / DeepSeek SM90 及 DeepGEMM 共用的 CUTLASS / CuTe headers |
| `DeepJIT/` | 与 DeepGEMM 的 gitlink 一致 | DeepGEMM 的 JIT runtime headers |

从仓库根目录初始化并连接构建所需的头文件路径：

```bash
python scripts/prepare_3rdparty.py --init
uv sync
```

源码只在顶层检出一份。准备脚本将 DeepGEMM 原有 `third-party/*/include`
连接到共享目录，并核对上游锁定版本。上游 `.gitmodules` 保持原样，嵌套依赖不初始化。
已有源码时可省略 `--init`，重复运行不会覆盖已有文件。

DeepGEMM 2.8.0 改用 DeepJIT 和 C++20，不再依赖 fmt。需要运行保留的
[legacy DeepGEMM 基准](../experiments/legacy/deepseek_v32/README.md)时，显式执行
`uv sync --group legacy`。本项目尚未完成此次依赖更新后的 GPU 验证，旧性能记录
仍对应旧版本。新版 FP4 量化采用 ties-to-even 舍入，重新比较数值时需要考虑该变化。
项目自有 SM120 扩展及其安装组已删除；复现其历史结果须使用归档 README 指定的旧 revision。

`EzKernelKit/` 是由 Git 忽略的本地参考 checkout，不是子模块或自动构建依赖。

SM90 的 NOSA 与 DeepSeek ECHO CUDA 内核复用顶层 `cutlass/include`，通过项目环境中的
TVM FFI 按需编译；只运行这些后端时使用 `uv sync`，不要求安装 `legacy` 组或 EzKernelKit。
