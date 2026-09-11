# GR 64K 前缀 + 4K extend：MLA cache 去重统计

本次使用 GR InputGenerator 生成一份输入，读取本地 DeepSeek V3.2 checkpoint 的真实 embedding、层前 RMSNorm 和第 0 层 indexer 权重。保留当前实现的 NeoX RoPE、MXFP8 投影和 indexer 量化，已移除 Hadamard。4096 个新增 query 一次性计算 logits 和 top-k。

## 结果

| 统计项 | 数量 |
|---|---:|
| Query 数 | 4,096 |
| 每个 query 的 top-k | 2,048 |
| 未去重引用总数 | 8,388,608 |
| **去重 MLA cache token 总数** | **16,018** |
| 其中：64K 历史 token | 12,427 |
| 其中：4K 新增 token | 3,591 |
| 历史 cache 覆盖比例 | 18.96% |
| 全部 cache 覆盖比例 | 23.00% |
| 平均每个入选 token 被引用次数 | 523.70 |
| 单个 token 最多被引用次数 | 3,381 |
| 去重 page 数（每页 64 token） | 1,088 / 1,088 |
| 历史 page / 新增 page | 1024 / 64 |

按每条 MLA cache 记录 656 字节计算，去重 token 对应 **10,507,808 字节（10.021 MiB）**。其中历史记录为 7.774 MiB。如果只能整页搬运，则需要全部 1,088 页，共 43.562 MiB；其中历史部分覆盖全部 1,024 页。

这些数值表示逻辑需求集合，不是实测 HBM 流量或 PCIe 搬运量。同一 token 的 MLA latent cache 由 128 个 heads 共享，计数不再乘 head 数；也不把 RoPE 部分拆成另一条记录。

## 输入与统计边界

生成器使用单用户 0、seed=42、规则商品描述和 20 个候选商品。固定前缀恰好为 65,536 token，候选后缀恰好为 4,096 token，共 69,632 token。GR 原有 `user_lengths` 只计用户历史正文，固定指令另有 23 token，因此本次配置正文 65,513 token、`item_lengths=4,119`，保证实际 cache/extend 边界准确为 64K/4K。

第 0 层的 index K 只依赖当前 token 的 embedding、层前 RMSNorm、K 投影、LayerNorm 和位置旋转，不需要先计算该层 attention 输出。历史 K 按 4096 token 批次准备，4K extend 的 Q、logits 与 top-k 均整批计算。统计的是全部 top-k 索引的并集，过滤边界由因果 ends 限定。已检查 logits 有限、索引不越过各 query 的因果边界、每行 2048 个索引互不重复。

本次只测一份输入的第 0 层，不能代表其他层、完整 61 层或其他文本分布。后续层需要前面层的真实输出才能继续做同样统计。

## 复现与产物

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m model_run.measure_gr_mla_cache_union
```

脚本：[measure_gr_mla_cache_union.py](../../model_run/measure_gr_mla_cache_union.py)。模型来自本地 `models/DeepSeek-V3.2`，依赖 `safetensors` 已加入项目配置。

输入和原始统计保留在本地 `GR/generated/cache_union_64k_4k/`，该目录受 Git 忽略规则保护：

- `input.jsonl`：完整文本、token IDs 和 GR 元数据。
- `prompt.txt`：可读输入。
- `selected_indices.pt`：`[4096,2048]` 的 top-k 索引。
- `unique_token_ids.json`：排序后的去重 token 索引。
- `result.json`：汇总结果。
