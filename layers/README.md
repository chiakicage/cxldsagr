# 共享模型层与 attention 契约

[normalization.py](normalization.py) 和 [feed_forward.py](feed_forward.py) 提供现有数学实现的
RMSNorm 与 SwiGLU，接收显式维度、eps、bias；NOSA 配置解析、LongRoPE、Q/K/V projection
以及 decoder 组合位于 [models/nosa](../models/nosa/README.md)。

[attention.py](attention.py) 定义 `Indexer`、`MainAttention`、`BlockSelection` 和
`AttentionContext`。indexer 输出逐 query/head 的逻辑块索引；main attention 接收 Q、
selection、cache access 和 layer/query 位置上下文。实际 dense adapter 从 resident view
取 K/V，调用现有 [FlashInfer 后端](../operators/flashinfer.py)。

未来 cache access 可以描述 HBM 驻留块和 host 来源，不要求提前 gather 所有选中 KV。
[SM90 入口](../operators/sm90/README.md) 将负责 attention 与 fetch 的重叠执行。
NOSA indexer 与 SM90 sparse attention 目前都明确报未实现；dense 路径不伪造块选择。
