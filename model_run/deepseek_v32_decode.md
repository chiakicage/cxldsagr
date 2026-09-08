# V3.2 decode 算子实验

`deepseek_v32_decode.py` 使用随机权重和随机历史 cache，执行一次 attention 子层的 decode。
它尚未加载 `deepseek_v32_two_dense_manifest.json` 中的真实权重，也不包含 embedding、
残差相加、MLP 或 LM head。输入 `case.x` 视为 attention 子层的输入。

## Norm 和 RoPE

`deepseek_v32_ops.py` 中的 `FlashInferV32Ops` 使用 FlashInfer 0.6.18：

- `norm.rmsnorm`：Q 的低秩投影和 512 维 KV latent；实验 affine weight 为 1。
- `norm.layernorm`：indexer K；FP32 affine weight/bias 分别为 1/0。
- `rope.apply_rope_with_cos_sin_cache`：主 MLA 的 64 维位置分量用 interleaved
  配对（`is_neox=False`）；indexer 的前 64 维用 split-half 配对（`is_neox=True`）。

位置来自 `CaseTensors.position_ids`，当前单步实验为每个序列的 `history_len`。
FP32 cos/sin 表在 runner 初始化时构建，不进入 decode 计时。
默认采用 V3.2 的 theta=10000、YaRN factor=40、original_seq_len=4096、
beta_fast=32、beta_slow=1、max_seq_len=163840；这些字段可以在实验配置中覆盖。
YaRN 是否启用由配置的最大长度决定，不随当前请求长度切换。
Attention scale 按原始 192 维 QK 计算，并应用 YaRN mscale 的平方修正。

Indexer 的 Hadamard 旋转目前采用 PyTorch FP32 butterfly 运算，最后转回 BF16；
FP8 量化和 packed cache 写入也采用 PyTorch。这些操作计入端到端时间。
Indexer 的 `weights_proj` 使用 BF16 权重和计算；Q 动态量化后的 scale 与
head_dim 的缩放合入 FP32 head weights，logits 输出为 FP32。

## Cache

当前 token 的两份 cache 都由本次投影结果更新，不再写入预生成的随机记录。

- MLA：每 token 656 B，512 个 FP8 latent、4 个 FP32 scale、64 个 BF16 RoPE key。
- Indexer：每页先存 64×128 个 FP8 key，再存 64 个 FP32 scale；虽然对外 tensor
  形状是 `[pages, 64, 1, 132]`，不能用该形状的 token 维直接写入记录。
  `index_cache_views()` 提供正确的 key/scale 视图。

## 运行

依赖已加入根目录 `pyproject.toml` 和 `uv.lock`，使用 CUDA 13 的 FlashInfer extra。
从项目根目录运行，并激活虚拟环境，让 JIT 能找到其中的 `ninja`：

```bash
uv sync --frozen
source .venv/bin/activate
python -m pytest model_run/tests/test_deepseek_v32_ops.py -q
python model_run/deepseek_v32_decode.py --batch-sizes 1 --history-lens 4096 --warmups 1 --iters 2
```

测试覆盖 YaRN 频率、Hadamard 点积保持、FP8 scale、真实 cache 更新，以及 GPU 上的
RMSNorm/LayerNorm、两种 RoPE 与复数旋转参考的比较（包括位置 163839）、DeepGEMM
indexer logits 与 PyTorch 参考比较和 decode 集成。GPU 测试要求进程能访问 NVIDIA 驱动。

新增操作改变了计时范围，旧 benchmark 数字不能直接作为当前实现的性能结果。
字节数/FLOP 估算仍以主要 GEMM 和 cache 操作为主，并非全部辅助运算的精确统计。

参考：[DeepSeek 参考实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)、
[FlashInfer RoPE](https://docs.flashinfer.ai/generated/flashinfer.rope.apply_rope_with_cos_sin_cache.html)。
