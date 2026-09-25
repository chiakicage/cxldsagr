# 算子索引

| 平台 | 当前内容 |
| --- | --- |
| [SM90 / Hopper](sm90/README.md) | 主开发平台，sparse attention + fetch 接口尚未实现 |
| [SM120](sm120/README.md) | sparse MLA prefill/decode 扩展与 DeepSeek V3.2 shape 基准 |

[flashinfer.py](flashinfer.py) 封装当前已使用的 Full Attention 调用，供 dense main attention
适配层使用；不包含新的 CUDA/Triton 算子。层级接口见 [共享层](../layers/README.md)。

DeepGEMM `nv_dev` 源码位于 [`3rdparty/DeepGEMM/`](../3rdparty/DeepGEMM/)，依赖准备见
[第三方说明](../3rdparty/README.md)。
