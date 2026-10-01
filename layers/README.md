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
融合减少中间低精度舍入，与逐算子路径不保证逐位一致；dense 性能测量入口见
[64K+1K 实验](../experiments/nosa_gr_65536_1024/README.md)，已于 2026-09-28 在 H200 上补测。

NOSA 的 LongRoPE 由模型复用 FP32 cos/sin cache，并通过 FlashInfer
`apply_rope_with_cos_sin_cache_inplace` 在一次调用中旋转 Q/K；位置编码及其缓存属于
[模型实现](../models/nosa/rotary.py)，不放入共享普通层。

[attention.py](attention.py) 定义 `Indexer`、`MainAttention`、`BlockSelection` 和
`AttentionContext`。indexer 输出逐 query/head 的逻辑块索引；main attention 接收 Q、
selection、cache access 和 layer/query 位置上下文。实际 dense adapter 从 resident view
取 K/V，调用现有 [FlashInfer 后端](../operators/flashinfer.py)。

cache access 可描述 resident view 或 host 来源与 device append，不要求提前 gather
所有选中 KV；[NOSA SM90 实现](../operators/nosa/attention/offload/api.py) 在单个 cooperative
主 kernel 中调度稀疏 fetch 与 persistent attention。所有 CTA 保留 attention，
启用 fetch 的 CTA 用 producer warpgroup 的三个空闲 warp，动态领取唯一页中
互不重叠的 8-token stripe。每个历史向量只读一次；全部 8 个 stripe 经 acq_rel
完成链后，TMA acquire-ready 并复用 HBM。两 KV heads / 256 work batches 时，
fetch 与 compute 都优先 head 1 再 head 0，其他几何保留原调度。
NOSA indexer 已实现 resident K 上的 query-aware 参考选块，见
[pattern 实验](../experiments/nosa_indexer_pattern_65536_1024/README.md)。QA-only 模式旁路记录选择，
dense adapter 仍拒绝非空 selection；完整 NOSA 对照还记录实际 sparse 传播中的选择。
完整 NOSA 由模型的
[sparse adapter](../models/nosa/attention.py) 在 resident 模式读取 K/V/CIS，调用 SM90
CUDA/CuTe 或 Triton block sparse 算子；显式 offload 模式使用 pinned 历史 K/V、resident
CIS 与 native SM90 fetch workspace。query-agnostic 分数和选块策略仍归模型。
NOSA adapter 调用模型专用 offload 算子；共享层仍通过 cache access 契约访问 KV。
完整 32 层 checkpoint 检查 1 passed：resident/offload 分别从独立空 cache
构建 64K sparse prefix，再执行 1K extend，全部 normalized hidden 逐位相同，max_abs=0。
[单层回放](../experiments/nosa_offload_overlap/README.md) 独立验证完整调用延迟与
stripe / page-envelope 两种 overlap，不作为完整模型性能测量。
