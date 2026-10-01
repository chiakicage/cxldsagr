# DeepSeek V3.2

同一模型目录保留两条实现：独立 SM90 完整 checkpoint 的 ECHO prefill/extend，及既有
SM120 synthetic attention decode/extend。SM90 路径不依赖 SGLang、DeepGEMM 或 SM120
扩展；包含全部 61 层、embedding、3 个 dense MLP、58 个 MoE、final norm 和 LM head。
修复前完整 61 层的 64K + 1K resident/offload 测量已验收，末 token logits bitwise 相同；
KV gather 对齐修复后的完整模型性能待补测，版本与结果见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。

| 模块 | 用途 |
| --- | --- |
| [echo_infer.py](echo_infer.py) | 完整模型：跨 GPU 放置常驻权重、按 token chunk 执行全部层、统一缓存事务及 LM head |
| [echo_block.py](echo_block.py) | 双路 hidden/residual、RMSNorm、dense / grouped MoE 与 checkpoint 权重加载 |
| [echo_attention.py](echo_attention.py) | 分块投影、indexer 融合 prefetch、精确 top-k / recall、物理 ID remap 与 sparse MLA |
| [echo_model.py](echo_model.py) | checkpoint 配置与读取、FP8 投影、RoPE / Hadamard、MLA 吸收投影 |
| [deepseek_v32_decode.py](deepseek_v32_decode.py) | `V32DecodeRunner`：投影、GroupedLinear、输出投影、decode cache 更新 |
| [deepseek_v32_extend.py](deepseek_v32_extend.py) | `V32ExtendRunner`：分块 extend、因果 sparse prefill、top-k |
| [deepseek_v32_extend_kernels.py](deepseek_v32_extend_kernels.py) | 融合 FP8 量化与 cache 追加的 Triton kernel |
| [deepseek_v32_ops.py](deepseek_v32_ops.py) | FlashInfer Norm / RoPE 与 indexer 量化适配 |
| [gr_index_selection.py](gr_index_selection.py) | 限制 logits 显存的分批因果 top-k 选择 |
| [tests/](tests) | ECHO checkpoint / block / 全模型调度事务参考测试，以及既有 SM120 ops / extend 测试 |

## 运行

从仓库根目录执行；第三方源码准备见 [依赖说明](../../3rdparty/README.md)。SM90 使用基础环境：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync
source .venv/bin/activate
python -m models.deepseek_v32.echo_infer --help
python -m models.deepseek_v32.echo_infer \
  --model /mnt/user-ssd/chenkaiqi/DeepSeek-V3.2 --devices 0,1,2,6,7 \
  --input-ids /path/to/input_ids.json --history 65536 --chunk-size 1024 --offload --slots 16384
```

`input_ids.json` 是单个请求的 token ID 列表，64K + 1K 场景共 66,560 项；省略
`--offload` 使用 resident 主 KV。权重常驻各 GPU，hidden/residual 在层放置边界传输。
每个 token chunk 顺序经过全部层；所有 chunk、输出与 GPU 同步成功后才提交请求长度。
默认只计算最后 token 的 LM head。完整 GR 请求生成与可复现 profile 命令见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。

无需 SM120 依赖的模型 CPU 回归：

```bash
.venv/bin/python -m pytest \
  models/deepseek_v32/tests/test_echo_model.py \
  models/deepseek_v32/tests/test_echo_block.py \
  models/deepseek_v32/tests/test_echo_infer.py -q
```

原 SM120 路径保留 DeepGEMM 投影、packed MLA cache 与 indexer 稀疏索引；张量随机
初始化且驻留 GPU，不含 embedding、MLP、LM head 或 KV offload。DeepGEMM 切换
`nv_dev` 后尚未重新验证该模型路径，运行仍使用显式 SM120 依赖组：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync --group sm120
.venv/bin/python models/deepseek_v32/deepseek_v32_decode.py --quick
.venv/bin/python models/deepseek_v32/deepseek_v32_extend.py
.venv/bin/python -m pytest models/deepseek_v32/tests/test_deepseek_v32_ops.py \
  models/deepseek_v32/tests/test_deepseek_v32_extend.py -q
```

`deepseek_v32_decode.py` 与 `deepseek_v32_extend.py` 同时支持 `-m` 入口和直接脚本入口。
两者默认读取 `experiments/legacy/deepseek_v32/docs/config.json` 的 attention / indexer 配置。

## KV 与 RoPE 语义

- SM90 ECHO 主 KV 每 token 1152 B：512 个 BF16 latent 加 64 个 BF16 RoPE key。
  Indexer FP8 K 与 FP32 scale 常驻 GPU；主 KV 使用 resident 存储，或
  [SparseTokenCache](../../cache/sparse_token_cache.py) 的 pinned DRAM backing 与有限 HBM slots。
  Indexer 内预取后执行精确 top-k，补齐剩余 miss；过大的 query 选中并集拆分消费，
  不截短 query 的精确选择。此路径使用本地 DRAM，不包含 CXL/RDMA。
- 原 SM120 MLA cache 每 token 656 B：512 个 FP8 latent、4 个 FP32 scale、64 个 BF16 RoPE key。
- SM120 indexer cache 每页先存 64×128 个 FP8 key，再存 64 个 FP32 scale；对外 tensor 形状
  `[pages, 64, 1, 132]` 不能按 token 维写入，须用 `index_cache_views()` 取 key/scale 视图。
- 主 MLA 的 64 维位置分量用 interleaved 配对（`is_neox=False`），indexer 前 64 维用
  split-half 配对（`is_neox=True`）。Attention scale 按原始 192 维 QK 计算并应用 YaRN mscale。
- SM120 extend 融合路径同时量化 MLA latent 与 indexer key 并直接写最终 cache；历史 cache 不重新打包。

## 相关实验

benchmark、profile 与 GR checkpoint 层实验见 [`experiments/`](../../experiments/README.md)：
[SM90 完整模型 ECHO prefill/extend](../../experiments/deepseek_v32_echo_prefill/README.md)、
[decode 实验](../../experiments/legacy/deepseek_v32/deepseek_v32_decode.md)、
[extend 实验](../../experiments/legacy/deepseek_v32/deepseek_v32_extend.md)。

共享 GR 请求生成使用 [request_format.py](request_format.py)：DeepSeek 历史请求模板与长上下文预算适配。
用法见 [GR 生成器](../../GR/README.md)。
