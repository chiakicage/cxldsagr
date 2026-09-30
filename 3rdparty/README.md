# 第三方依赖

| 子模块 | 版本来源 | 用途 |
| --- | --- | --- |
| `DeepGEMM/` | `nv_dev`，父仓库固定提交 | GEMM / MQA logits |
| `cutlass/` | 与 DeepGEMM 的 gitlink 一致 | CUTLASS / CuTe headers |
| `DeepJIT/` | 与 DeepGEMM 的 gitlink 一致 | JIT runtime headers |

从仓库根目录初始化并连接构建所需的头文件路径：

```bash
python scripts/prepare_3rdparty.py --init
uv sync --group sm120
```

源码只在顶层检出一份。准备脚本将 DeepGEMM 原有 `third-party/*/include`
连接到共享目录，并核对上游锁定版本。上游 `.gitmodules` 保持原样，嵌套依赖不初始化。
已有源码时可省略 `--init`，重复运行不会覆盖已有文件。

DeepGEMM 2.8.0 改用 DeepJIT 和 C++20，不再依赖 fmt。SM120 构建需要支持
`sm_120f` 的 CUDA 工具链；上游的 SM120 验证使用 CUDA 13.2。本项目尚未完成此次
更新后的 GPU 验证，旧性能记录仍对应旧版本。新版 FP4 量化采用 ties-to-even 舍入，
重新比较数值时需要考虑该变化。

`EzKernelKit/` 是由 Git 忽略的本地参考 checkout，不是子模块或自动构建依赖。

`ECHO/` 是 [GR serving cache 实验](../experiments/gr_cache_serving/README.md) 的独立、
被忽略的上游 checkout，固定为 `sjtu-zhao-lab/ECHO@bc1b75c`。
通过 `bash experiments/gr_cache_serving/scripts/prepare_echo.sh` 准备源码，默认跳过
论文 LFS 数据及嵌套子模块；这不代表运行环境或 CUDA 扩展已准备完成。
ECHO 的修改版 DeepGEMM 与本项目顶层共享依赖隔离，不用它替换通用后端。

64K 检查发现原版 SM90 fused prefetch 的共享 phase flag 竞态。研究侧
[echo_kernel.py](../models/deepseek_v32/echo_kernel.py) 提供显式派生修复
`echo_sm90_prefetch_phase_snapshot_v1`：在独立 header overlay 中替换一个经过 SHA256
校验的 `.cuh`，其余 include 链接原安装包；不写回 ECHO checkout、安装包或顶层 DeepGEMM。
默认 overlay 缓存在 `~/.cache/cxldsagr/echo-deepgemm/`，记录原/修复 header SHA、
完整 installed include manifest SHA、C++ 扩展及初始化器源码 SHA，保留独立 JIT identity。
使用期间不可更新被链接的原包。

Python `open_echo_runner(..., kernel_patch=None)` 默认原版；GPU 正确性脚本仅对
`echo_gr_adapted` 默认选 `phase_snapshot`，`SPARSEGR_ECHO_TEST_KERNEL_PATCH=native`
可禁用。选择必须早于 DeepGEMM/SGLang 导入，编译器配置每进程仅一次、不可恢复，
切换 kernel 须新进程。永久 overlay 接入后，修复版已通过 4K + 1K、64K + 1K
三层 GR 子序列的四路径正确性检查及 resident 文件交叉参考；不是完整请求流或性能回放。
原版 ECHO 64K 仍有挂起，不能把派生修复结果归于 native kernel。
另有 test-only logical top-k 次序控制，仅重排原 selected multiset、不解决边界 tie，
也不是上述 kernel 修复；它包含同步，禁止进入性能测量路径。

SM90 的本地 NOSA CUDA 内核复用顶层 `cutlass/include`，通过项目环境中的 TVM FFI
按需编译；只运行该后端时使用 `uv sync`，不要求安装 `sm120` 组或 EzKernelKit。
