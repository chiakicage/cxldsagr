# 模型推理

模型结构、稀疏选择语义、KV 表示与推理运行代码，只保留模型推理相关代码及其测试。
实验脚本、运行记录与实验文档见 [`experiments/`](../experiments/README.md)，
本地权重与 tokenizer 见 `weights/`。

| 模型 | 当前内容 |
| --- | --- |
| [NOSA](nosa/README.md) | FlashInfer Full Attention 单 GPU 推理、分块 prefill、KV cache 与文本生成 |
| [DeepSeek V3.2](deepseek_v32/README.md) | SM120 synthetic decode/extend、packed MLA cache、indexer 稀疏索引选择 |

共享请求生成工具见 [`GR/`](../GR/README.md)，算子与构建说明见 [`operators/`](../operators/README.md)。
