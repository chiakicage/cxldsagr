# NOSA kernels

NOSA 算子按功能组织，当前 GPU 实现面向 SM90 / Hopper。模型 policy、CIS 投影、
KV 布局和 cache 事务适配保留在 [models/nosa](../../models/nosa/README.md)。

| 目录 | 功能与入口 |
| --- | --- |
| [indexer](indexer/api.py) | 压缩后的 query-aware scoring、CIS pooling、精确选块，以及 checked preparation |
| [attention/reference](attention/reference/torch.py) | PyTorch FP32 数学参考，支持 CPU 和 CUDA 正确性对照 |
| [attention/device_only](attention/device_only/api.py) | 输入 K/V 已在设备上，直接消费逻辑块选择；native FA3/CUDA 与 Triton dispatch |
| [attention/offload](attention/offload/api.py) | pinned host 稀疏 KV fetch 与 persistent FA3 attention，以及整批稀疏并集串行对照 |

`indexer/compression.py`、`offload_compression.py` 和 `validation.py` 负责派生记录
准备；offload compression 仍属于 indexer。每个功能的 CUDA 源码与头文件放在自身
`csrc/`，单元测试放在对应 `tests/`。reference 导入不会加载 Triton、TVM FFI 或编译扩展。

device-only 的 BF16 / D128 / GQA16 主路径使用 FA3，其他支持布局保留 native CUDA
或 Triton。offload 复用同一 FA3 源码和数值 repair，通过 include 依赖共享实现；
当前 staging 覆盖一层完整逻辑地址范围，尚无有限 HBM slots / eviction。

[attention/workspace.py](attention/workspace.py) 提供可选的有界 FA3 scratch，
device-only API 可通过 `workspace=` 注入。offload 的 `bounded=True` 配合显式
`max_queries`、`trace_capacity` 预分配所有执行空间，供 NOSA backend 跨用户串行借用。
默认 standalone 仍按原方式惰性分配。共享路径不回退到未预留的 Triton/FP16/layout
分支；CPU 构造仅用于容量与状态测试，attention 需先转数学参考。输出 hidden/attention
结果独立分配，不与下一次请求复用的 scratch 混同。

[_native.py](_native.py) 保留后端选择与 native JIT 构建入口，按组件实际本地 include
依赖生成缓存键。FA3 和 offload loader 另外记录 FlashInfer header 与共享源码指纹。
编译缓存位于源码树外，依赖顶层共享 CUTLASS 和已安装的 FlashInfer 0.6.18。
`CXLDSAGR_SM90_BACKEND=triton` 仍可选择已有 device-only/indexer Triton 对照。

从仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest operators/nosa -q
.venv/bin/python -m pytest operators/nosa -q
```

第一条只验证 CPU 可运行检查；GPU 检查需要 Hopper 和编译依赖。全局强制 GPU
环境检查使用 `bash scripts/run_tests.sh gpu`。测试不代表完整模型性能测量。

算子、模块与完整 resident 模型的效率统一见 [NOSA MFU](../../experiments/nosa_mfu/README.md)，
搬运与重叠性能见
[offload overlap](../../experiments/nosa_offload_overlap/README.md)。目录迁移不产生新的性能结果。

2026-10-05 已完成冻结真实输入 L0/L15/L31 的独立算子与完整模块验收，随后分别运行
干净 bench 和独立 profile。算子覆盖 native/Triton 的 attention 与 pooled score，
每阶段 12 组结果；完整模块在两种后端各覆盖 6 组结果，profile 每组 3 个样本。
新运行使用 `refactor_mfu_real_{bench,profile}_20261005_01` 和
`refactor_mfu_modules_{native,triton}_{bench,profile}_20261005_01`；三个验收回执均为
`*_check_20261005_02`。产物、回执、实际源码和 trace 已重新核验并发布。
独立数值验收不进入每个计时样本；这些结果不代表完整模型或 serving 延迟。

本轮完整 H64K/A1024 resident/owned-offload 检查和 overlap 数值检查已通过，见
[独立验收记录](../../docs/agents/acceptance/unified_runtime_20261005/nosa_non_timing_gpu1_ledger.json)。
完整模型 native/Triton sparse 与 dense 的独立 bench/profile 已发布。冷态 A1024
单层 overlap 在 L0/L15/L31 比串行完整 API 延迟下降 29.47% / 24.33% / 23.02%；
九个 profile 样本的 page/stripe ratio 均达到 90%，最低 91.08%。这组算子结果与
固定 A128 serving 分开：后者 96 个有搬运层样本的 ratio 为 62.34%–80.29%，均未达到
90% 目标，见[固定容量实验](../../experiments/nosa_motivation/README.md)。
