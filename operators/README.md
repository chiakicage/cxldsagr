# 算子索引

| 平台 | 当前内容 |
| --- | --- |
| [SM90 / Hopper](sm90/README.md) | NOSA block sparse / indexer / sparse fetch overlap；DeepSeek ECHO fused prefetch、sparse MLA、FP8 linear / grouped MoE、pinned-host KV transfer |
| [SM120](sm120/README.md) | sparse MLA prefill/decode 扩展与 DeepSeek V3.2 shape 基准 |

[flashinfer.py](flashinfer.py) 封装 Full Attention、RMSNorm、residual-add + RMSNorm、
SiLU × up 与 Q/K RoPE 的 FlashInfer 调用，供 dense main attention、普通层及模型使用；
不包含新的 CUDA/Triton 算子。RoPE 使用 `apply_rope_with_cos_sin_cache_inplace`，接收
模型准备的 FP32 cos/sin cache，不解析 NOSA 配置或生成 LongRoPE factors。
层级接口见 [共享层](../layers/README.md)，位置编码见 [NOSA](../models/nosa/README.md)。

DeepGEMM `nv_dev` 源码位于 [`3rdparty/DeepGEMM/`](../3rdparty/DeepGEMM/)，依赖准备见
[第三方说明](../3rdparty/README.md)。

新增 DeepSeek SM90 算子不依赖 SGLang、DeepGEMM 或 SM120 扩展。DeepSeek 独立
token-cache 路径的完整 61 层、64K + 1K
resident/offload 测量已验收，测量边界与结果见
[ECHO 实验](../experiments/deepseek_v32_echo_prefill/README.md)。

NOSA 的 [nosa_offload.py](sm90/nosa_offload.py) 已实现 native SM90 BF16/D128/GQA16
按需 fetch 与 attention 融合主 kernel：唯一页拆为 8 个不交叠 token stripe，
每个历史向量只读一次 host；完成链到 ready=8 后，FA3 通过 TMA 复用 HBM staging。
20 次主测和独立 40 次确认的单层回放均快于整批稀疏串行对照；每个 profile 样本的
page-envelope 与实际 stripe-copy 两种 softmax overlap 均达到 90%；
通用 `SM90SparseAttention` 仍是显式失败的占位接口，
见 [NOSA offload 实验](../experiments/nosa_offload_overlap/README.md)。
