# 共享模型层与 attention 契约

[normalization.py](normalization.py) 和 [feed_forward.py](feed_forward.py) 提供
RMSNorm 与 SwiGLU，接收显式维度、eps、bias；NOSA 配置解析、LongRoPE、Q/K/V projection
以及 decoder 组合位于 [models/nosa](../models/nosa/README.md)。

CUDA BF16/FP16 且关闭 autograd 时调用 FlashInfer `rmsnorm`、`fused_add_rmsnorm` 和
`silu_and_mul`；CPU、FP32、开启 autograd 或不满足 kernel 布局要求时保留 PyTorch 数学路径。
`RMSNorm(x, residual)` 返回 `(normalized, sum)`，FlashInfer 路径会原地覆盖两个输入，
调用者须持有独立且可覆盖的 activation buffer；不传 residual 时只返回 normalized。
SwiGLU 用 `gate_up_proj` 一次 GEMM 生成 `[gate | up]`，直接交给融合激活；
`down_proj` 执行输出投影。合并权重是实际参数，不在 forward 拼接 activation 或维护权重副本。
原 checkpoint 的 gate/up 权重由模型加载器合并；共享层仅接收显式维度与 bias 配置。
融合减少中间低精度舍入，与逐算子路径不保证逐位一致；性能测量见
[64K+1K 实验](../experiments/nosa_gr_65536_1024/README.md)。

NOSA 的 LongRoPE 由模型复用 FP32 cos/sin cache，并通过 FlashInfer
`apply_rope_with_cos_sin_cache_inplace` 在一次调用中旋转 Q/K；位置编码及其缓存属于
[模型实现](../models/nosa/rotary.py)，不放入共享普通层。

[attention.py](attention.py) 定义 `Indexer`、`MainAttention`、`BlockSelection` 和
`AttentionContext`。indexer 输出逐 query/head 的逻辑块索引；main attention 接收 Q、
selection、cache access 和 layer/query 位置上下文。实际 dense adapter 从 resident view
取 K/V，调用现有 [FlashInfer 后端](../operators/flashinfer.py)。

未来 cache access 可以描述 HBM 驻留块和 host 来源，不要求提前 gather 所有选中 KV。
[SM90 入口](../operators/sm90/README.md) 将负责 attention 与 fetch 的重叠执行。
NOSA indexer 已实现 resident K 上的 query-aware 参考选块，见
[pattern 实验](../experiments/nosa_indexer_pattern_65536_1024/README.md)。实验只旁路记录选择，
dense adapter 仍拒绝非空 selection。完整 NOSA 由模型的
[sparse adapter](../models/nosa/attention.py) 读取 resident K/V/CIS，调用 SM90 Triton
block sparse 算子；query-agnostic 分数和选块策略仍归模型。offload 路径明确报未实现。
