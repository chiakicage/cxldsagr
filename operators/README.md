# 算子索引

| 平台 | 当前内容 |
| --- | --- |
| [SM90 / Hopper](sm90/README.md) | 主开发平台，sparse attention + fetch 接口尚未实现 |
| [SM120](sm120/README.md) | sparse MLA prefill/decode 扩展与 DeepSeek V3.2 shape 基准 |

[flashinfer.py](flashinfer.py) 封装 Full Attention、RMSNorm、residual-add + RMSNorm、
SiLU × up 与 Q/K RoPE 的 FlashInfer 调用，供 dense main attention、普通层及模型使用；
不包含新的 CUDA/Triton 算子。RoPE 使用 `apply_rope_with_cos_sin_cache_inplace`，接收
模型准备的 FP32 cos/sin cache，不解析 NOSA 配置或生成 LongRoPE factors。
层级接口见 [共享层](../layers/README.md)，位置编码见 [NOSA](../models/nosa/README.md)。

DeepGEMM `nv_dev` 源码位于 [`3rdparty/DeepGEMM/`](../3rdparty/DeepGEMM/)，依赖准备见
[第三方说明](../3rdparty/README.md)。
