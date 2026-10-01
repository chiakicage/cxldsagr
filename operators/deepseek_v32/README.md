# DeepSeek V3.2 kernels

独立 Hopper / SM90 实现，服务完整 checkpoint 的 ECHO prefill/extend。模型投影语义、
精确选块、缓存调度和事务位于 [模型目录](../../models/deepseek_v32/README.md)。

| 功能 | 入口 | 当前实现 |
| --- | --- | --- |
| Indexer | [indexer/echo.py](indexer/echo.py) | FP8 causal logits；可在同一 kernel 内预取主 MLA records，仍需精确 top-k 和剩余 miss recall |
| Attention reference | [attention/reference/torch.py](attention/reference/torch.py) | 独立 FP32 PyTorch oracle，可单独在 CPU 导入运行，不加载 Triton 或 native 扩展 |
| Attention device_only | [attention/device_only/mla.py](attention/device_only/mla.py) | BF16/FP16 Triton sparse MLA，消费设备内的 latent/RoPE records |
| Attention offload | [attention/offload/mla.py](attention/offload/mla.py) | 消费已精确 recall 的 HBM pool 和物理 ID，复用同一个 device-only MLA kernel |
| Linear / MoE | [linear/fp8.py](linear/fp8.py) | block FP8 linear、激活量化及 grouped expert GEMM |

ECHO 的融合发生在 indexer 与 prefetch 之间；当前没有将 host fetch 融入 attention
kernel。每个 offload query batch 必须先补齐其精确选择，模型在选中并集超过 pool
容量时拆分 query 消费，保持每个 query 的选择不变。通用 pinned-host record gather
位于 [operators/common/kv_transfer.py](../common/kv_transfer.py)，有限 slots、映射和
事务位于 [cache/sparse_token_cache.py](../../cache/sparse_token_cache.py)。

主 KV record 是 512 个 BF16 latent 加 64 个 BF16 RoPE key，1152 B/token；indexer
FP8 K 与 FP32 scale 常驻 GPU。MLA 的 selection 在所有 attention heads 间共享；
重复 ID 逐 slot 参与计算，负值和越界 ID 是 padding，全 padding 行输出零。

单元测试随 indexer、linear 和三种 attention 实现存放；通用搬运测试在
`operators/common/tests/`。从仓库根目录运行全局 CPU 或 Hopper GPU 回归：

```bash
bash scripts/run_tests.sh cpu
bash scripts/run_tests.sh gpu
```

历史完整模型测量和 KV gather 修复后待补测的边界见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。本次目录迁移不构成新的
完整模型性能测量，原 run ID 和源码快照继续标识原测量实现。
