# SM90 / Hopper

Hopper 是项目主开发平台，优先面向 [NOSA](../../models/nosa/README.md)。
[nosa_attention.py](nosa_attention.py) 提供 resident NOSA block sparse attention，以及
独立 FP32 数学参考；Triton kernel 位于 [_nosa_attention_triton.py](_nosa_attention_triton.py)。
输入 Q 为 `[query, query_head, D]`，K/V 为 `[token, KV_head, D]`，选择为逻辑 64-token
block ID 与 validity mask，CIS 为 `[token, KV_head]` 的加性 logits bias。
每 program 处理一个 query / KV head 的 GQA 组，按块直接读取 K/V，融合 QK、CIS、
因果 mask、在线 softmax 和 AV；不物化逐 query 展开的选中 KV。

[nosa_indexer.py](nosa_indexer.py) 提供两遍 tiled query-aware scoring、五窗口 max pooling
及确定性两阶段 Top-K。模型层负责 CIS 投影、压缩、选择策略与 cache 布局，见
[NOSA 实现](../../models/nosa/README.md)。CUDA 后端要求 SM90、FP16/BF16、D=64/128、
GQA group≤32，FP32 累积；AV 概率舍入为输入 dtype。reference 用于正确性对照。

[sparse_attention.py](sparse_attention.py) 仍预留 offloaded cache fetch 与 sparse compute
重叠执行入口，调用抛出 `NotImplementedError`。resident kernel 不实现 DRAM 读取、
搬运、缓存淘汰或 overlap，不能将其作为 offloading 验证。

NOSA 默认 dense 路径使用共享 [FlashInfer Full Attention](../flashinfer.py) 适配。
本地 EzKernelKit 可作为 block sparse attention 参考，见
[第三方说明](../../3rdparty/README.md)；它不是当前运行后端。
