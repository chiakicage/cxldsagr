# DeepSeek V3.2 embedding + 两层 dense 实验权重

本地分片位于 `models/DeepSeek-V3.2/model-00001-of-000163.safetensors`，从 `/mnt/nfs/share/models/DeepSeek-V3.2/` 复制，源文件流与本地文件进行了 SHA-256 比对。该目录已加入 `.gitignore`。

本次选择：

| 模块 | 原始权重前缀 | 说明 |
| --- | --- | --- |
| Embedding | `model.embed_tokens.` | 完整词表 embedding，形状 `[129280, 7168]` |
| 第一层 | `model.layers.0.` | attention、indexer、dense MLP、layer norms |
| 第二层 | `model.layers.1.` | attention、indexer、dense MLP、layer norms |

原模型 `first_k_dense_replace=3`，第 0、1、2 层为 dense；实验选择第 0、1 层。Embedding 与这两层的 **55 个 tensor 全部位于第一分片**，包括 FP8 weight 对应的 `weight_scale_inv`。无需为了这次选择复制其他分片。

[manifest](deepseek_v32_two_dense_manifest.json) 记录了本地路径、文件大小、校验和、所选 tensor 的原始名称/形状/dtype。第一分片原样保留，包含的其他 tensor 不在本次实验选择中。

同目录还复制了 `config.json`、`tokenizer.json` 和 `tokenizer_config.json`。配置保留原始 61 层模型参数，不能把这个部分 checkpoint 当成完整模型直接 `from_pretrained`。所选权重用于 embedding → 第 0 层 → 第 1 层的 hidden states 实验；没有 LM head，不能据此进行完整推荐答案生成。

生成匹配本地 tokenizer 的真实输入：

```bash
.venv/bin/python -m GR.input_generator \
  --tokenizer models/DeepSeek-V3.2 \
  --heat-source beauty --text-dataset beauty --count 1000 \
  --output GR/generated/beauty.jsonl
```

`GR` 输出的 token IDs 可以用于 embedding lookup。现有 `deepseek_v32_decode.py` 仍是随机张量 benchmark；此次准备了真实输入和所需 checkpoint，尚未实现这三个模块的真实权重 forward。
