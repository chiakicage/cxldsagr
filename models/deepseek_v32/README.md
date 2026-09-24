# DeepSeek V3.2

SM120 synthetic decode/extend 的模型运行代码：DeepGEMM 投影、packed MLA KV cache、
indexer 稀疏索引选择。张量随机初始化并驻留 GPU，不含 embedding、MLP、LM head 或
KV offload 路径。DeepGEMM 切换 `nv_dev` 后尚未重新验证模型运行。

| 模块 | 用途 |
| --- | --- |
| [deepseek_v32_decode.py](deepseek_v32_decode.py) | `V32DecodeRunner`：投影、GroupedLinear、输出投影、decode cache 更新 |
| [deepseek_v32_extend.py](deepseek_v32_extend.py) | `V32ExtendRunner`：分块 extend、因果 sparse prefill、top-k |
| [deepseek_v32_extend_kernels.py](deepseek_v32_extend_kernels.py) | 融合 FP8 量化与 cache 追加的 Triton kernel |
| [deepseek_v32_ops.py](deepseek_v32_ops.py) | FlashInfer Norm / RoPE 与 indexer 量化适配 |
| [gr_index_selection.py](gr_index_selection.py) | 限制 logits 显存的分批因果 top-k 选择 |
| [tests/](tests/) | ops 与 extend 的数学、cache / indexer 验证 |

## 运行

从仓库根目录执行；第三方源码准备见 [依赖说明](../../3rdparty/README.md)：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync --group sm120
.venv/bin/python models/deepseek_v32/deepseek_v32_decode.py --quick
.venv/bin/python models/deepseek_v32/deepseek_v32_extend.py
.venv/bin/python -m pytest models/deepseek_v32/tests -q
```

`deepseek_v32_decode.py` 与 `deepseek_v32_extend.py` 同时支持 `-m` 入口和直接脚本入口。
两者默认读取 `docs/config.json` 的 attention / indexer 配置。

## KV 与 RoPE 语义

- MLA cache 每 token 656 B：512 个 FP8 latent、4 个 FP32 scale、64 个 BF16 RoPE key。
- indexer cache 每页先存 64×128 个 FP8 key，再存 64 个 FP32 scale；对外 tensor 形状
  `[pages, 64, 1, 132]` 不能按 token 维写入，须用 `index_cache_views()` 取 key/scale 视图。
- 主 MLA 的 64 维位置分量用 interleaved 配对（`is_neox=False`），indexer 前 64 维用
  split-half 配对（`is_neox=True`）。Attention scale 按原始 192 维 QK 计算并应用 YaRN mscale。
- extend 融合路径同时量化 MLA latent 与 indexer key 并直接写最终 cache；历史 cache 不重新打包。

## 相关实验

benchmark、profile 与 GR checkpoint 层实验见 [`experiments/`](../../experiments/README.md)：
[decode 实验](../../experiments/deepseek_v32_decode.md)、
[extend 实验](../../experiments/deepseek_v32_extend.md)。

共享 GR 请求生成使用 [request_format.py](request_format.py)：DeepSeek 历史请求模板与长上下文预算适配。
用法见 [GR 生成器](../../GR/README.md)。
