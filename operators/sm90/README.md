# SM90 / Hopper

Hopper 是项目主开发平台，优先面向 [NOSA](../../models/nosa/README.md)。
[nosa_attention.py](nosa_attention.py) 提供 resident NOSA block sparse attention，以及
独立 FP32 数学参考；Triton kernel 位于 [_nosa_attention_triton.py](_nosa_attention_triton.py)。
输入 Q 为 `[query, query_head, D]`，K/V 为 `[token, KV_head, D]`，选择为逻辑 64-token
block ID 与 validity mask，CIS 为 `[token, KV_head]` 的加性 logits bias。
每 program 处理一个 query / KV head 的 GQA 组，按块直接读取 K/V，融合 QK、CIS、
因果 mask、在线 softmax 和 AV；不物化逐 query 展开的选中 KV。

[nosa_indexer.py](nosa_indexer.py) 将两遍 tiled query-aware scoring 与五窗口 max pooling
融合，使用 FlashInfer `top_k(..., tie_break=SMALL)` 完成 Top-33 / Top-64，最后只排序
64 个 ID 并写出 validity。模型层负责 CIS 投影、选择策略与 cache 布局，见
[NOSA 实现](../../models/nosa/README.md)。CUDA 后端要求 SM90、FP16/BF16、D=64/128、
GQA group≤32，FP32 累积；AV 概率舍入为输入 dtype。reference 用于正确性对照。

完整 NOSA 的 Triton indexer 一次处理完整 query batch；`query_chunk_size` 仅控制
reference（默认 64）。连续入口以标量 `query_start + row` 计算位置，不创建 Torch
下标张量。外部位置接口仍校验范围。上限为 262144 tokens / 4096 blocks。
64K+1K 的 BF16 pooled-score scratch 为 4.0625 MiB，QA 与 CIS 两阶段复用；模型请求
还在各层之间复用该 scratch。GQA 求和后先舍入到模型 dtype，再做 pooling。

[nosa_compression.py](nosa_compression.py) 增量写入完整的 32-token / stride-16 K/CIS
压缩窗口及稳定 CIS pool，与 [IndexerCache](../../cache/indexer_cache.py) 配合。
末尾最多两个未稳定 block 在选择阶段计算。已提交的压缩前缀不重算；融合有限值检查
只扫描当前 Q 和尚未校验的自有 K/CIS 后缀，外部 tensor view 仍扫描完整输入。

[sparse_attention.py](sparse_attention.py) 仍预留 offloaded cache fetch 与 sparse compute
重叠执行入口，调用抛出 `NotImplementedError`。resident kernel 不实现 DRAM 读取、
搬运、缓存淘汰或 overlap，不能将其作为 offloading 验证。

NOSA 默认 dense 路径使用共享 [FlashInfer Full Attention](../flashinfer.py) 适配。
本地 EzKernelKit 可作为 block sparse attention 参考，见
[第三方说明](../../3rdparty/README.md)；它不是当前运行后端。
