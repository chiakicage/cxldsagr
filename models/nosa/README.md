# NOSA

此目录提供 NOSA checkpoint 的单 GPU 文本推理，目标平台为 SM90 / Hopper。
默认 attention 使用 FlashInfer **Full Attention**，支持分块 prefill、KV cache 和逐 token decode；
支持 [GR 本地串行执行](../../serving/README.md)，默认推理不启用 sparse selection。
显式 sparse 模式提供 query-aware / query-agnostic 选块及 resident block sparse attention。
KV offloading 与批量调度尚未实现。

- [model.py](model.py)：模型参数树、权重加载与前向；保留原有导入接口。
- [config.py](config.py)、[rotary.py](rotary.py)：NOSA 配置与 LongRoPE。
- [layers.py](layers.py)：NOSA projection、attention 与 decoder 组合。
- [cache.py](cache.py)：NOSA KV 布局及 resident session 适配。
- [indexer.py](indexer.py)：64-token block，默认 1 sink + 16 local + 47 query-aware top-k；
  支持 32-block 分析预算，以及显式完整 NOSA 两阶段选块。
- [scoring.py](scoring.py)：query-agnostic CIS 打分与 32-token / stride-16 压缩。
- [attention.py](attention.py)：resident sparse adapter，调用
  [SM90 算子](../../operators/sm90/README.md) 或 CPU 数学参考。
- [infer.py](infer.py)：本地 tokenizer、chat template、采样与命令行入口。
- 现有 DeepSeek 实验见 [DeepSeek V3.2](../deepseek_v32/README.md)。

普通 RMSNorm / SwiGLU 与 attention 契约见 [共享层](../../layers/README.md)，FlashInfer
调用见 [算子](../../operators/README.md)。[执行器](../../executor/README.md) 统一分块前向，
[缓存管理器](../../cache/README.md) 管理逐层写入、有效长度提交及请求释放。
main attention 接收逻辑块选择与 cache access，为后续 SM90 算子内部 fetch/compute overlap
保留边界；当前 dense 路径不执行 indexer，也没有异步搬运或 DRAM backing。

## 模型与 attention 语义

默认 checkpoint 为 `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`。该配置包含 32 层、4096 hidden size、
16384 FFN intermediate size、32 个 query heads、2 个 KV heads、128 head dimension，
使用 Q/K/V 投影、SwiGLU、RMSNorm 和 LongRoPE。Q/K/V 合并为一次 `qkv_proj` GEMM，
gate/up 合并为一次 `gate_up_proj` GEMM。KV cache 保存 RoPE 后的 K 与原始 V，
采用此模型的 GQA 布局。BF16 权重约 16.37 GB，还需要 KV cache、激活和 kernel workspace 显存。

CUDA BF16/FP16 推理的 RMSNorm、residual-add + RMSNorm、SwiGLU 激活已接入 FlashInfer。
Decoder 返回待相加的 MLP 输出和 residual，下一层 input norm 合并相加；最后一次相加
由 final norm 完成。输出选择会同步截取这两个分量。
融合的数值舍入与逐算子路径不同，正确性通过模型单元测试与跨模块回归检查。

`from_pretrained` 接受原 NOSA checkpoint 的独立 Q/K/V、gate/up 权重与 bias，
在加载时直接复制进合并参数的对应切片，支持跨 safetensors 分片；前向没有权重拼接或副本缓存。
运行时 `state_dict` 使用 `self_attn.qkv_proj` 和 `mlp.gate_up_proj` 参数名，加载器也支持这种格式。
重复混入同一投影的独立与合并权重、缺失参数或 shape 不符均报错。

LongRoPE 使用模型持有的 FP32 cos/sin cache，复用静态频率及位置对应的旋转值。
CUDA 推理通过 FlashInfer `apply_rope_with_cos_sin_cache_inplace` 一次融合 Q/K 旋转，
直接使用合并 GEMM 输出的行跨距视图，保留 V 并避免 Q/K 复制。
保留 NOSA 的 split-half 布局和 LongRoPE scaling。dense 性能测量入口见
[64K+1K 实验](../../experiments/nosa_gr_65536_1024/README.md)，共享 cache 改动后未运行。

默认 dense 模式计算普通 causal GQA attention，保留 checkpoint 的 LongRoPE 缩放向量。
此模式将 NOSA 的 `self_attn.A` 与 `self_attn.delta.weight` 在加载时明确跳过。
原 sparse 分支中这两个参数既用于块选择，也用于 CIS attention 加权；dense 两者均不启用，
语义对应上游模型的普通 dense `eager` / `flash_attention_2` 分支，而非仅移除 sparse mask。
dense 性能实验测量使用 NOSA 权重的 Full Attention。

`NosaIndexer()` 接受现有 `q/cache_access/context` 契约，在 resident K 上用 FP32
计算 query-aware 评分，返回 `[query, KV head, block_budget]` 的逻辑 block IDs 与 validity mask。
默认 `block_budget=64`；使用 `NosaIndexer(block_budget=32)` 可改为 32 块。
Q/K 均为 RoPE 后张量；K mean compression 为窗口 32、stride 16，各 Q head 先独立
causal softmax，再按 GQA 分组求和、五窗口 max pooling，最后排除 sink/local 后选择
剩余的 query-aware top-k：64-block 预算选 47 块，32-block 预算选 15 块。
local 明确包含当前块及之前 15 块，选择与相同分数的排序均确定。此分支不需要 A/delta。
非 resident access 明确报错，默认 dense adapter 仍拒绝非空 selection。
[64K+1K pattern 实验](../../experiments/nosa_indexer_pattern_65536_1024/README.md)
对比同一 dense 激活上的 QA-only / 完整 NOSA 与实际 sparse 传播，统计每层各 KV head 对 1K queries 的选块并集、K+V 容量及覆盖率。

## 完整 NOSA block sparse

`NosaForCausalLM(..., attention_mode="sparse", sparse_backend="auto")` 以及同参数的
`from_pretrained` 启用完整路径；默认值仍为 `attention_mode="dense"`。
实现参考 cxl-recsys commit `6e20e7df07518be4a0669dd94bdbb304e619fe31` 中
`cxl_recsys/models/nosa_ops.py`、`nosa_indexer.py`、`nosa_attention.py`，使用本项目的
模型、LongRoPE 和 resident cache 接口，无需导入 cxl-recsys 或 HiSparse。

每层加载 `A=[KV heads]` 与 `delta.weight=[KV heads, KV heads * head_dim]`；
配置启用 attention bias 时也加载 `delta.bias`。缺失、shape 错误和非浮点权重均报错。
对原始 V 计算 `CIS = softplus(delta(V.flatten(1)).float()) * A.float()`，再转回模型 dtype。
attention 使用 `softmax(QK / sqrt(D) + CIS) V`，CIS 直接作为加性偏置。
即使短上下文所有块均选中，输出也可能与无 CIS 的 dense 基线不同。

`NosaIndexer(mode="nosa", backend="reference" | "triton" | "auto")` 返回
`[query, KV head, 64]`，有效 ID 升序排列，缺位为 `-1` 并附 validity mask：

| 模式 | local（含当前块） | query-aware | query-agnostic |
| --- | --- | --- | --- |
| 默认分析 `query_aware` | 16 | sink/local 外选 47（32预算选15） | 不启用 |
| 完整 `nosa` | 17 | 含 sink/local 共保留 33 | 排除已选块后补满 64 |

完整模式复刻 cxl-recsys 的 inclusive local 边界（当前块和前 16 块），仅支持 64-block
预算。K 与 CIS 均按 32-token、stride-16 对完整窗口取 mean，再对每个 64-token block
重叠的五窗口取 max。query-aware 分数先逐 Q head causal softmax，再按 GQA 求和；
query-agnostic 使用压缩后的 CIS。相同分数优先较小 block ID。完整模式保留模型 dtype
的压缩/分数舍入；原 query-aware-only FP32 分析保持原语义。

`auto` 在 CPU 使用 reference、CUDA 使用 SM90 dispatcher；历史接口名 `triton` 保留。
NOSA-8B 的 D128/GQA=16 默认启用迁移后的 CUDA/CuTe QK score 和 block sparse attention；
QK 在短上下文按[算子阈值](../../operators/sm90/README.md)使用单 kernel Triton。
其余已支持形状走 Triton。`CXLDSAGR_SM90_BACKEND=triton` 强制对照后端。
原生构建要求 nvcc、TVM FFI 与共享 CUTLASS，见[算子说明](../../operators/sm90/README.md)。
CUDA 不支持的设备或 shape 明确失败。
Triton 支持 SM90、FP16/BF16、head_dim 64/128、GQA group 1–32，使用 FP32 在线 softmax
累积，AV 概率转回输入 dtype；数值不保证与 FP32 reference 逐位相等，indexer 的舍入
也可能改变近似并列分数的排名。算子按逻辑块直接读取 resident NHD K/V，逐 token
应用 causal mask，不生成 `[queries, selected_tokens, K/V]` 展开缓冲区。

完整 NOSA 的 CUDA indexer 一次处理完整 query batch，reference 默认 64；
`NosaIndexer(query_chunk_size=...)` 仅控制 reference。`effective_query_chunk_size(device)`
对 CUDA 返回 `None` 表示不分块，传入 query_length 可查询本次实际大小。CUDA 使用连续整数 query
区间，省去位置张量的同步检查；Q/K/CIS 有限性检查仍保留，合并为一次 GPU 状态读取。

sparse cache 将 CIS 作为 `[layer, capacity, KV head]` 的命名 record，按新增 token
计算一次，与 K/V 共同提交。完整 NOSA 的 SM90 路径使用请求持有的
[IndexerCache](../../cache/indexer_cache.py)：按层惰性缓存压缩 K/CIS 和稳定 CIS pool，
prefill 完成后仅追加新窗口。五窗口 pool 在块结束后再到来 16 tokens 才稳定，末尾最多
两个块现场计算。有限值检查复用已校验前缀；外部 CIS 覆盖和 reference 仍独立检查与计算。
Top-33 / Top-64 使用 FlashInfer；query scratch 跨阶段和层复用。上下文上限为 256K tokens。
`cache.truncate(length)` 同步回退原始和派生记录，失败 append 随模型事务一起回滚。
offload fetch 与 overlap 尚未实现。原 QA-only pattern 不启用此模式；
新增完整 NOSA pattern 对照从独立空 cache 构建 sparse prefix，记录 attention 实际消费的选块。
完整 sparse 路径的全模型 prefill/extend 测量见
[64K+1K 端到端 profile](../../experiments/indexer_block_sparse_profile/README.md)。

```bash
source .venv/bin/activate
python -m models.nosa.infer --attention-mode sparse --sparse-backend triton \
  --prompt "请用简洁的中文解释 KV cache。" --disable-thinking --max-new-tokens 32
python -m serving.run_gr --attention-mode sparse --count 1 \
  --user-lengths 4096 --item-lengths 128 --prefill-chunk-size 1024
```

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

## 正确性检查

模型单元测试覆盖独立数学参考、分块 cache、权重加载、模板与生成停止条件。
没有 CUDA 时 GPU 项跳过；未设置模型路径时本地 metadata 项跳过。

2026-09-26 在 NVIDIA H200（SM90）上验证了 sparse 路径：CIS/两阶段选择的独立参考、
BF16/FP16 Triton attention 与 indexer、严格 A/delta 加载、缓存回滚、生成/GR CLI，
以及 4097-token 场景的 65→64 块裁剪。另以本地 NOSA-8B 完成一次
`--user-lengths 4096 --item-lengths 128` 的 sparse GR 前向，总长 4224，
stable prefix 4124、candidate suffix 100。此检查不构成吞吐测量或模型质量评估。
当时环境为 PyTorch `2.10.0+cu132`、Triton `3.6.0`、FlashInfer `0.6.18`；
与当前 `pyproject.toml` / `uv.lock` 中的 torch/triton 版本不同，锁定版本组合尚未在本次验证。

直接加载 cxl-recsys 的 `nosa_ops.py` 作只读对照，CPU/CUDA × FP32/BF16、6147 tokens、
4 Q heads / 2 KV heads / D64 下，CIS 与 reference 完整选块 ID 均精确一致。
Triton scoring 使用 Tensor Core 舍入，不能将 reference 对齐表述为所有后端逐位一致。

```bash
source .venv/bin/activate
NOSA_MODEL_PATH=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  python -m pytest models/nosa/tests -q
```

跨模块回归使用 `bash scripts/run_tests.sh [cpu|gpu|all]`，默认 CPU；GPU 模式要求
CUDA 与 FlashInfer 可用。测试命令和环境准备见[项目 README](../../README.md)。

## GR 前向与性能测量

共享 GR 请求生成使用 [request_format.py](request_format.py)：NOSA 聊天模板、tokenizer 与请求预算适配。
用法见 [GR 生成器](../../GR/README.md)。
dense 单请求性能测量入口见 [GR 实验](../../experiments/nosa_gr_65536_1024/README.md)，
共享 cache 改动后未运行。通过 `bash experiments/nosa_gr_65536_1024/scripts/run.sh <run_id>`
可重新测量延迟和模块 MFU。

GR 前向使用 `model(input_ids, cache, return_hidden=True)` 返回本次调用所有输入 token 的最终
normalized hidden states，跳过 LM head；不与 `logits_to_keep` 同时使用。
生成 CLI 仍默认返回 logits。64K+1K 实验测量完整 prefill 与 prefix 已就绪后的候选 extend，
不执行自回归生成。

分块执行 `executor.model_executor.run_chunks` 返回最后一个 chunk 的输出；serving 从中
取末 token hidden，不累积整条请求的全部 hidden。每请求独立创建和释放 cache；同一请求
的 candidate extend 使用其已完成的 stable prefix，当前没有跨请求的用户前缀缓存。
