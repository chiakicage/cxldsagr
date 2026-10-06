# 模型推理

模型结构、权重、稀疏选择语义、KV 布局适配与推理代码，只保留模型推理相关代码及其测试。
各模型保留自己的普通层与 attention 适配，复用 [执行器](../executor/README.md) 和
[缓存管理](../cache/README.md)，通过 [本地 serving](../serving/README.md) 执行 GR 请求。
实验脚本、运行记录与实验文档见 [`experiments/`](../experiments/README.md)，
本地权重与 tokenizer 见 `weights/`。

[attention_contracts.py](attention_contracts.py) 定义轻量的 `Indexer`、`MainAttention`、
`BlockSelection`、`TokenSelection` 和 `AttentionContext`，不导入具体模型。
NOSA 使用逐 query/head 的逻辑块选择与 validity mask；DeepSeek 使用逻辑 token IDs，
padding 语义由调用模型定义。这些结构不展开、复制或搬运 KV。
main attention 接收 Q、selection、cache access
及 layer/query 位置上下文，cache access 可提供 resident view 或 host 来源与
device append，不要求预先 gather 全部选中 KV。
NOSA 的 RMSNorm / SwiGLU、`DenseMainAttention` 与 `ResidentLayerView` 由
[NOSA 模型](nosa/README.md)实现；模型布局、位置编码和稀疏策略仍由各模型管理。

| 模型 | 当前内容 |
| --- | --- |
| [NOSA](nosa/README.md) | FlashInfer Full Attention / SM90 block sparse、CIS 与 indexer、resident / pinned DRAM offload cache、GR 前向及文本生成 |
| [DeepSeek V3.2](deepseek_v32/README.md) | 独立 SM90 完整 61 层 ECHO prefill/extend、indexer 融合 prefetch、bounded HBM / pinned DRAM KV |

DeepSeek SM90 不依赖 SGLang。当前四方法 MFU 使用真实 checkpoint 第 0–2 层依次
传播，H=65,536、A=128、P=65,664，已完成独立数值验收、正式计时和 profile。
完整 61 层实现仍保留，本轮结果只覆盖真实前三层，不能作为完整模型验收。
测量边界与版本见 [DeepSeek 四方法 MFU](../experiments/deepseek_v32_mfu/README.md)。
十 block C10 GR 工作负载另见 [本地四方案对照](../experiments/deepseek_v32_motivation/README.md)，
其固定历史和临时候选语义与前三层 MFU 的持久追加分别报告。
NOSA 默认 resident，显式 offload 使用 native SM90 sparse fetch / attention overlap；
唯一页内的 8 个 token stripe 各只读取一次 host，完成后 attention 复用 HBM。
完整 32 层从独立空 cache 构建 64K sparse prefix + 1K extend，数值逐位一致。
融合主 kernel 在 20 次主测和独立 40 次确认的单层回放中均快于整批稀疏并集串行
fetch + 完整 attention；每个 profile 样本的 page-envelope / stripe-copy 两项
softmax overlap 均达到 90%。完整模型 offload 性能尚未测量，见
[NOSA offload 实验](../experiments/nosa_offload_overlap/README.md)。

共享请求生成工具见 [`GR/`](../GR/README.md)，算子与构建说明见 [`operators/`](../operators/README.md)。
