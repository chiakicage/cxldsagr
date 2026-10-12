# 模型推理

模型结构、权重、稀疏选择语义、KV 布局适配与推理代码，只保留模型推理相关代码及其测试。
各模型保留自己的普通层与 attention 适配，复用 [执行器](../executor/README.md) 和
[缓存管理](../cache/README.md)，通过 [本地 serving](../serving/README.md) 执行 GR 请求。
实验脚本、运行记录与实验文档见 [`experiments/`](../experiments/README.md)，
本地权重与 tokenizer 见 `weights/`。

[attention_contracts.py](attention_contracts.py) 定义轻量的 `Indexer`、`MainAttention`、
`BlockSelection`、`TokenSelection` 和 `AttentionContext`，不导入具体模型。
DeepSeek 使用逻辑 token IDs，
padding 语义由调用模型定义。这些结构不展开、复制或搬运 KV。
main attention 接收 Q、selection、cache access
及 layer/query 位置上下文，cache access 可提供 resident view 或 host 来源与
device append，不要求预先 gather 全部选中 KV。
| 模型 | 当前内容 |
| --- | --- |
| [DeepSeek V3.2](deepseek_v32/README.md) | 独立 SM90 完整 61 层 ECHO prefill/extend、indexer 融合 prefetch、bounded HBM / pinned DRAM KV |

DeepSeek SM90 不依赖 SGLang。当前四方法 MFU 使用真实 checkpoint 第 0–2 层依次
传播，H=65,536、A=128、P=65,664，已完成独立数值验收、正式计时和 profile。
完整 61 层实现仍保留，本轮结果只覆盖真实前三层，不能作为完整模型验收。
测量边界与版本见 [DeepSeek 四方法 MFU](../experiments/deepseek_v32_mfu/README.md)。
十 block C10 GR 工作负载另见 [本地四方案对照](../experiments/deepseek_v32_motivation/README.md)，
其固定历史和临时候选语义与前三层 MFU 的持久追加分别报告。
共享请求生成工具见 [`GR/`](../GR/README.md)，算子与构建说明见 [`operators/`](../operators/README.md)。
