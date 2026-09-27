# SM90 / Hopper

Hopper 是项目主开发平台，优先面向 [NOSA](../../models/nosa/README.md)。
[nosa_attention.py](nosa_attention.py) 提供 resident NOSA block sparse attention，以及
独立 FP32 数学参考。默认 CUDA 后端的 NOSA-8B specialization 位于
[csrc/nosa_attention.cu](csrc/nosa_attention.cu)，使用 CuTe WGMMA 与 TMA；
其他已支持形状使用 [_nosa_attention_triton.py](_nosa_attention_triton.py)。
输入 Q 为 `[query, query_head, D]`，K/V 为 `[token, KV_head, D]`，选择为逻辑 64-token
block ID 与 validity mask，CIS 为 `[token, KV_head]` 的加性 logits bias。
每 CTA 处理一个 query / KV head 的 GQA 组，producer 用 TMA 加载 K/V，
consumer 流水执行 WGMMA QK/PV，融合 CIS、
因果 mask、在线 softmax 和 AV；不物化逐 query 展开的选中 KV。

[nosa_indexer.py](nosa_indexer.py) 将两遍 tiled query-aware scoring 与五窗口 max pooling
融合；[csrc/nosa_scores.cu](csrc/nosa_scores.cu) 分两次 launch：第一遍使用 TMA 与 WGMMA 双缓冲流水计算 softmax normalizer
（query 数不足 128 时最多分 4 段；否则 1 段），第二遍重算 QK，
合并 normalizer 并完成 GQA 舍入与 pooling。两遍不物化完整 score 矩阵。压缩 K 少于 2047 个（query 数≥1024）或
511 个（query 数<1024）时，dispatcher 使用单 kernel Triton scoring。使用 FlashInfer `top_k(..., tie_break=SMALL)` 完成 Top-33 / Top-64，最后只排序
64 个 ID 并写出 validity。模型层负责 CIS 投影、选择策略与 cache 布局，见
[NOSA 实现](../../models/nosa/README.md)。CUDA 后端要求 SM90、FP16/BF16、D=64/128、
GQA group≤32，FP32 累积；AV 概率舍入为输入 dtype。reference 用于正确性对照。

完整 NOSA 的 CUDA indexer 一次处理完整 query batch；`query_chunk_size` 仅控制
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
本地实现按 EzKernelKit 的 Hopper 调度模式适配 NOSA 布局、CIS 与五窗口 pooling；
具体参考文件及 commit 见 CUDA 源码头。编译只使用顶层共享 `3rdparty/cutlass/include`，
通过 TVM FFI 懒加载，要求 `nvcc` 与 `ninja` 在 PATH 上，构建缓存位于源码树外。
不加载 EzKernelKit Python 包或头文件，见[第三方说明](../../3rdparty/README.md)。

原生 specialization 支持 FP16/BF16、D128、GQA=16，要求 Q/K/V 满足 16-byte
地址和外层 stride 对齐；非此形状/布局走既有 Triton 路径。
`CXLDSAGR_SM90_BACKEND=triton` 可显式选择 Triton 对照；默认 `native`。
模型已有的 `sparse_backend="triton"` 接口保留，当前指向这个 SM90 dispatcher。
64K+1K 的新旧后端同源测量见
[indexer_block_sparse_profile](../../experiments/indexer_block_sparse_profile/README.md)。
