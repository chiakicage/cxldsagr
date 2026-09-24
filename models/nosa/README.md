# NOSA

此目录提供 NOSA checkpoint 的单 GPU 文本推理，目标平台为 SM90 / Hopper。
当前 attention 使用 FlashInfer **Full Attention**，支持分块 prefill、KV cache 和逐 token decode；
尚未实现 sparse block selection、KV offloading 或批量调度。

- [model.py](model.py)：模型配置、权重加载、LongRoPE、模型前向与 KV cache。
- [infer.py](infer.py)：本地 tokenizer、chat template、采样与命令行入口。
- 现有 DeepSeek 实验见 [DeepSeek V3.2](../deepseek_v32/README.md)。

## 模型与 attention 语义

默认 checkpoint 为 `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`。该配置包含 32 层、4096 hidden size、
16384 FFN intermediate size、32 个 query heads、2 个 KV heads、128 head dimension，
使用独立的 Q/K/V 投影、SwiGLU、RMSNorm 和 LongRoPE。KV cache 保存 RoPE 后的 K 与原始 V，
采用此模型的 GQA 布局。BF16 权重约 16.37 GB，还需要 KV cache、激活和 kernel workspace 显存。

当前实现计算普通 causal GQA attention，保留 checkpoint 的 LongRoPE 缩放向量。
NOSA 的 `self_attn.A` 与 `self_attn.delta.weight` 在加载时明确跳过。
原 sparse 分支中这两个参数既用于块选择，也用于 CIS attention 加权；当前两者均不启用，
语义对应上游模型的普通 dense `eager` / `flash_attention_2` 分支，而非仅移除 sparse mask。
因此当前结果是使用 NOSA 权重的 Full Attention 基线，不表示原 NOSA sparse 推理结果。

## 运行

在仓库根目录使用既有环境，不需要安装仓库或加载 checkpoint 中的 Python 实现：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync
source .venv/bin/activate  # FlashInfer 首次 JIT 编译需从 PATH 找到 ninja
.venv/bin/python -m models.nosa.infer \
  --prompt "请用简洁的中文解释 KV cache 的作用。" \
  --disable-thinking --max-new-tokens 128
```

也支持直接脚本入口与自定义 checkpoint：

```bash
.venv/bin/python models/nosa/infer.py \
  --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --prompt-file /tmp/nosa_prompt.txt \
  --system-prompt "你是一位简洁的中文助手。" \
  --device cuda:0 --dtype bfloat16 \
  --prefill-chunk-size 1024 --max-new-tokens 256
```

默认读取 `tokenizer_config.json` 的 chat template，将输入作为单轮 user message，
并添加 assistant 生成前缀；模板结果编码时不再添加 BOS 等特殊 token。
`--disable-thinking` 将 `enable_thinking=False` 传给模板。
使用 `--raw-prompt` 可直接补全文本，此时保留 tokenizer 自带的 BOS 处理，
不接受 `--system-prompt` 或 `--disable-thinking`。

默认 greedy decoding；设置正数 temperature 可启用采样：

```bash
.venv/bin/python -m models.nosa.infer \
  --prompt "写一段关于秋天的短文。" \
  --temperature 0.8 --top-p 0.8 --seed 42 --max-new-tokens 128
```

遇到配置中的 EOS token（此 checkpoint 为 `2`、`73440`）或达到 `--max-new-tokens` 时停止。
prompt token 数与请求生成上限之和不得超过 `max_position_embeddings`（此 checkpoint 为 32768）；
超长输入会报错，不自动截断。prefill 默认每次处理 1024 tokens，可通过 `--prefill-chunk-size` 调整。
模型与 cache 均驻留指定 CUDA GPU，dtype 支持 `bfloat16` 与 `float16`。
cache 的 K、V 分别为 `[层数, 容量, KV heads, head_dim]`，每层传入 FlashInfer 的布局为 NHD；
此 checkpoint 使用 BF16 时，每个 token 的全部层 K/V 共 32 KiB，32768 tokens 约 1 GiB。
当前仅支持默认 RoPE 或此 NOSA checkpoint 的静态 LongRoPE（short/long factors 相同），
不支持需要在运行中切换频率的配置。

stdout 仅输出生成文本，stderr 输出 JSON 统计，可分别重定向。
统计中的 `generated_tokens` 包含终止 EOS（输出文本会移除特殊 token）；
`decode_steps` 是实际执行的单 token 模型前向次数，首个 token 从 prefill logits 采样。
`prefill_seconds` 与 `decode_seconds` 包含首次 kernel 编译、采样和 Python 调度开销，
仅用于本次请求的耗时观察，不作为经过预热的吞吐基准。

## 验证状态

2026-09-24 已在本机 NVIDIA M403（SM90）上完成 BF16 验证，使用现有环境的
PyTorch `2.10.0+cu132`、FlashInfer `0.6.18`、safetensors `0.8.0`、tokenizers `0.23.2`。
该 PyTorch 版本与仓库锁定的 `2.12.1+cu130` 不同；本次没有重建完整锁定环境，也没有验证
FP16、32768-token 上限、sparse attention、offloading 或 DeepGEMM。

- CPU 独立 FP64 数学参考验证 GQA、RMSNorm、SwiGLU、LongRoPE 和模型 logits；
  同时检查分块 cache、重置、溢出、单文件/分片权重、模板、EOS 和生成长度。
- GPU FlashInfer prefill、追加 prefill、decode 与独立 dense attention 参考对照通过。
- 本地 NOSA-8B 全部 dense 权重加载与文本生成通过：27-token chat prompt，
  `--prefill-chunk-size 8`，生成 59 tokens（含 EOS），执行 58 次缓存 decode 并正常停止。
- Ruff、模块入口与直接脚本 `--help` 检查通过。

本次完整测试集 **24 项通过**。运行方式如下（没有 CUDA 时 GPU 项跳过；未设置模型路径时
本地 metadata 项跳过）：

```bash
source .venv/bin/activate
NOSA_MODEL_PATH=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  python -m pytest models/nosa/tests -q
```

复现本次真实权重生成检查：

```bash
python -m models.nosa.infer \
  --prompt "请用一句话解释 KV cache 的作用。" \
  --disable-thinking --max-new-tokens 64 --prefill-chunk-size 8
```
