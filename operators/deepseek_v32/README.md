# DeepSeek V3.2 kernels

独立 Hopper / SM90 实现，服务完整 checkpoint 的 ECHO prefill/extend。模型投影语义、
精确选块、缓存调度和事务位于 [模型目录](../../models/deepseek_v32/README.md)。

| 功能 | 入口 | 当前实现 |
| --- | --- | --- |
| Indexer | [indexer/echo.py](indexer/echo.py) | FP8 causal logits；可在同一 kernel 内预取主 MLA records，仍需精确 top-k 和剩余 miss recall |
| Indexer selection | [indexer/selection.py](indexer/selection.py) | FlashInfer deterministic exact top-k，保留 causal padding 与 int32 无效 ID |
| Indexer quantization | [indexer/quantization.py](indexer/quantization.py) | KDA 验证的 SM90 BF16/D128 单 kernel，FP8 bytes 与 FP32 scales 对齐独立 reference |
| Attention reference | [attention/reference/torch.py](attention/reference/torch.py) | 独立 FP32 PyTorch oracle，可单独在 CPU 导入运行，不加载 Triton 或 native 扩展 |
| Attention device_only | [attention/device_only/mla.py](attention/device_only/mla.py) | 官方 FlashMLA SM90 sparse prefill，BF16 / H64 或 H128 / D576 / V512 |
| Attention offload | [attention/offload/mla.py](attention/offload/mla.py) | 消费已精确 recall 的 HBM pool 和物理 ID，复用同一个 device-only MLA kernel |
| Linear / MoE | [linear/fp8.py](linear/fp8.py) | 官方 DeepGEMM block FP8 dense / grouped GEMM，激活量化使用手写 Triton |

ECHO 的融合发生在 indexer 与 prefetch 之间；当前没有将 host fetch 融入 attention
kernel。每个 offload query batch 必须先补齐其精确选择，模型在选中并集超过 pool
容量时拆分 query 消费，保持每个 query 的选择不变。通用 pinned-host record gather
位于 [operators/common/kv_transfer.py](../common/kv_transfer.py)，有限 slots、映射和
事务位于 [cache/sparse_token_cache.py](../../cache/sparse_token_cache.py)。

主 KV record 是 512 个 BF16 latent 加 64 个 BF16 RoPE key，1152 B/token；indexer
FP8 K 与 FP32 scale 常驻 GPU。MLA 的 selection 在所有 attention heads 间共享；
重复 ID 逐 slot 参与计算，负值和越界 ID 是 padding，全 padding 行输出零。
FlashMLA 通过顶层 [子模块](../../3rdparty/FlashMLA) 接入；adapter 仅处理 KV head
维度、连续布局、int64 ID 安全转换及 selection 补齐到 128 的倍数。必要的复制与
padding 都属于算子时间。其他 dtype 或布局明确失败，不使用 Triton attention fallback。
CPU reference 仍可独立导入，不加载 FlashMLA。

FP8 投影使用顶层 [DeepGEMM 子模块](../../3rdparty/DeepGEMM) 的 `main` 分支固定版本。
激活量化由 [本地接口](linear/quantization.py) 直接启动 Triton kernel，不调用
`torch.compile`。它保留 128-channel / UE8M0 格式，返回独立、连续的 FP8 数据和
FP32 scales；编译后的上游 `per_token_cast_to_fp8` 仅用于验证和性能对照。
dense / grouped 路径调用官方公共 API。适配层的 padding、
route packing、scale layout 转换和 inverse mapping 均计入算子时间。
CPU FP32 oracle 独立实现，不依赖 DeepGEMM。
新量化路径已通过生产接口、实际 checkpoint MLP 和四方案 H64K 数值验收；
motivation 的 C9 端到端补测与配套 profile 已验收。其他实验的旧数字不代表当前实现。

单元测试随 indexer、linear 和三种 attention 实现存放；通用搬运测试在
`operators/common/tests/`。从仓库根目录运行全局 CPU 或 Hopper GPU 回归：

```bash
bash scripts/run_tests.sh cpu
bash scripts/run_tests.sh gpu
```

当前非 GR 三层 benchmark、历史完整模型测量及新实现补测状态见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。原 run ID 和源码
快照继续标识原测量实现；切换后端不构成新的测量结果。
