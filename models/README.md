# 模型推理

模型结构、权重、稀疏选择语义、KV 布局适配与推理代码，只保留模型推理相关代码及其测试。
NOSA 复用 [共享层](../layers/README.md)、[执行器](../executor/README.md) 和
[缓存管理](../cache/README.md)，通过 [本地 serving](../serving/README.md) 执行 GR 请求。
实验脚本、运行记录与实验文档见 [`experiments/`](../experiments/README.md)，
本地权重与 tokenizer 见 `weights/`。

| 模型 | 当前内容 |
| --- | --- |
| [NOSA](nosa/README.md) | FlashInfer Full Attention / SM90 block sparse、CIS 与 indexer、resident cache、GR 前向及文本生成 |
| [DeepSeek V3.2](deepseek_v32/README.md) | SM120 synthetic decode/extend、packed MLA cache、indexer 稀疏索引选择 |

共享请求生成工具见 [`GR/`](../GR/README.md)，算子与构建说明见 [`operators/`](../operators/README.md)。
