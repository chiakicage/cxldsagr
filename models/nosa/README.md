# NOSA

此目录提供 NOSA checkpoint 的单 GPU 文本推理，目标平台为 SM90 / Hopper。
默认 attention 使用 FlashInfer **Full Attention**，支持分块 prefill、KV cache 和逐 token decode；
支持 [GR 本地串行执行](../../serving/README.md)，默认推理不启用 sparse selection。
显式 sparse 模式提供 query-aware / query-agnostic 选块及 block sparse attention。
默认 cache 为 resident；新增 `cache_backend="offload"` 支持 pinned local DRAM 历史
K/V 与 SM90 sparse fetch / attention overlap。当前在同一个 cooperative CUDA
主 kernel 中融合稀疏 fetch 与整批 attention，每个唯一页拆成 8 个不交叠 token
stripe，跨 CTA 只读一次 host K/V。完整 32 层 checkpoint 检查 1 passed：resident/offload 分别从独立空 cache
构建 64K sparse prefix，再执行 1K extend，全部 normalized hidden 逐位相同，max_abs=0。
单层性能和 stripe / page-envelope overlap 见下文。新增跨请求保留 prefix 的
串行 serving 适配见下节；共享 workspace 的 4K / 16K / 64K 短轨迹已用
原源码完成测量与验收；该短轨迹范围现已结束，保留通用模型实现和回归。
独立的固定 P/NH 入口及其
正式测量、匹配 profile 和 API 对照见[固定容量实验](../../experiments/nosa_motivation/README.md)。

- [model.py](model.py)：模型参数树、权重加载与前向；保留原有导入接口。
- [config.py](config.py)、[rotary.py](rotary.py)：NOSA 配置与 LongRoPE。
- [layers.py](layers.py)：NOSA projection、attention 与 decoder 组合。
- [normalization.py](normalization.py)、[feed_forward.py](feed_forward.py)：RMSNorm 与 SwiGLU。
- [cache/resident.py](cache/resident.py)：NOSA KV 布局及 resident session 适配。
- [cache/offload.py](cache/offload.py)：pinned 历史 K/V、resident CIS/压缩记录及共享 staging。
- [execution/adapter.py](execution/adapter.py)：完整 NOSA 的跨请求 prefix session、四种缓存方案及预算适配。
- [execution/resources.py](execution/resources.py)：backend 持有的有界 staging/scratch、
  C/A/Q 预留、串行执行 lease 与 session 所有权检查。
- [execution/fixed.py](execution/fixed.py)：固定 P/NH 的四方案入口，候选整批 GPU 执行后 discard。
- [execution/fixed_resources.py](execution/fixed_resources.py)：逐层有限 P 槽、共享执行资源与固定容量计费。
- [cache/fixed.py](cache/fixed.py)：独立用户 history、直接映射的 HBM residency 与候选事务。
- [indexer.py](indexer.py)：64-token block，默认 1 sink + 16 local + 47 query-aware top-k；
  支持 32-block 分析预算，以及显式完整 NOSA 两阶段选块。
- [scoring.py](scoring.py)：query-agnostic CIS 打分与 32-token / stride-16 压缩。
- [attention.py](attention.py)：`DenseMainAttention`、`ResidentLayerView` 及 resident / offload sparse adapter，调用
  [SM90 算子](../../operators/nosa/README.md) 或 CPU 数学参考。
- [infer.py](infer.py)：本地 tokenizer、chat template、采样与命令行入口。
- 现有 DeepSeek 实验见 [DeepSeek V3.2](../deepseek_v32/README.md)。

普通 RMSNorm / SwiGLU 保留在本模型目录；公共 indexer / attention 协议见
[attention_contracts.py](../attention_contracts.py)，FlashInfer 调用见
[算子](../../operators/README.md)。[执行器](../../executor/README.md) 统一分块前向，
[缓存管理器](../../cache/README.md) 管理逐层写入、有效长度提交及请求释放。
main attention 接收逻辑块选择与 cache access；显式 offload 由 SM90 算子在单个
cooperative 主 kernel 内调度 fetch/compute。`DenseMainAttention` 通过 cache access
读取 resident K/V 并调用 FlashInfer Full Attention；无持久 cache 的前向由
`ResidentLayerView` 提供视图。dense 路径不执行 indexer。

## 跨请求 GR serving

[NosaServingBackend](execution/adapter.py) 提供按 HBM / CPU DRAM 字节预算准入的普通入口，
连接 [PersistentGRRunner](../../serving/persistent.py)，
支持 `hbm`、`serial_sparse`、`dense_prefetch` 与 `overlap`。四种方案均执行完整
32 层 NOSA sparse 模型，保留相同的 CIS、selection 和 causal mask；`dense_prefetch`
表示搬运完整历史 K/V，attention 本身仍为 sparse。

persistent 入口通过 [TokenRuntime](../../executor/runtime.py) 调用 NOSA 适配器。
NOSA 提供具名的 `ResourcePlan` / `SessionPlan`，保留自己的 cache 布局和 attention
执行流程；共享 [ResourceLifecycle](../../cache/lifecycle.py) 管理 owner、generation
和执行占用。会话计划区分实际 storage 与计费预留，分配时沿用该计划的维度。
backend 最多复用一套不可变分配声明及其 footprint；布局参数、provider 或 generation
变化时重新构建。history/resource 身份和配额仍逐请求生成并核验，allocator 实时审计
不缓存。布局复用已通过 [CPU 验收](../../docs/agents/acceptance/unified_runtime_20261005/nosa_layout_cache_cpu_01/evidence.json)
和 336 组规划配置对照；规划微基准不能替代完整 serving 计时。

每个用户持有自己的稳定 prefix session。命中时直接复用，candidate 执行后统一截短回
prefix；全局 LRU 按相同 HBM / CPU DRAM cache 上限准入，淘汰时释放该用户的两级缓存。
`hbm` 不使用 CPU KV backing。旧 `serving.run_gr` / `GRRunner` 仍逐请求创建、释放 session，
跨请求复用由新的 persistent 入口提供。

`dense_prefetch` 使用 pinned 历史 K/V、两个完整逻辑层的 HBM staging 及独立 copy stream。
下一层拷贝与当前层计算流水执行，event 保护 staging 复用与读取。两种 sparse offload
复用现有原生算子：串行方案先取齐完整 query batch 的稀疏并集，overlap 使用 cooperative
FA3。该普通 budget 路径没有有限 token slots 或 token 淘汰，LRU 的粒度是整个用户 session。
预算包括 K/V/CIS、派生 indexer records、staging、拥有的 pending append 和 scratch，
模型权重及临时计算内存单独统计。

2026-10-03 的共享资源改造已接入 NOSA backend：sparse staging 与 dense 双缓冲由
backend 持有，hbm/dense 使用同一种显式有界 FA3 scratch；用户仍分别持有自己的
历史、CIS、派生记录、indexer scratch 与 pending append。直接调用 backend 时必须
先 `plan_resources` / `allocate_shared`，session 的 `prefill` / `extend` 自动取得执行
lease。释放用户不会释放共享存储；全部 session 释放后由最外层 `backend.close()` 关闭。
普通 `model.new_cache()` 保留原有 owned 默认行为。

共享 GPU 路径限定 native SM90/BF16/D128/GQA16，C/A/Q 或 trace 越界在执行前失败。
其中 C 为最大 session 容量，A 为最大 candidate 长度，Q 为 `max(min(chunk_size,C),A)`；
未提供 A 时按 C 保守预留。公共 runner 已接入唯一准入 owner 和构造失败回滚，
CLI 与正式入口显式传入 C/A。公共集成后的完整 32 层、两用户交错复访的
64K+1K 与 64K+128 均逐位一致，普通 owned 64K+1K 路径也通过检查。
冻结集成版本的全仓库回归为 CPU 2672 passed、GPU 1639 passed；另行验收两种几何、
四方案的全部 48 个分配阶段，32 组 candidate hidden 对照逐位一致。
以上为 2026-10-03 的集成 checkpoint；按当时要求只验收正确性，未运行共享版本的
性能或完整用户容量轨迹，证据见
[集成验收索引](../../docs/agents/acceptance/unified_runtime_20261005/shared_cache_integration_evidence.json)。

完整分配审计随后发现原公式遗漏 allocator 尾部占用及 cache helper 临时张量。
`nosa_backend_workspace_v2` 将逻辑 storage 与 allocator 预留分开，按每个独立张量
计算上界，并覆盖 CIS、选块、有限性检查、边界拼接和 pending K/V。
这一修正已通过独立 backend 的完整分配复验：32 层、两种几何、四方案共 48 个
阶段通过独立预算上界检查，32 次 candidate hidden 比较逐位一致。审计按完整
allocator 回调顺序匹配分配代次，并保留至 `free_completed` 或 capture 结束。
公共 runner 的准入、所有权和失败清理已通过集成回归；该分配 checkpoint 当时未运行
完整用户容量轨迹及正式性能。后续共享版本曾完成 H4K / H16K / H64K 短轨迹补测，
该短轨迹实验现已退出；这些结果没有证明填满用户容量时的进程物理 HBM 上界。
Pinned DRAM 按每次分配的实际 power-of-two bin 预留和报告，不能只累加 K/V
逻辑字节；`session_storage_bytes()` 另提供逻辑 storage 账本。
上述普通 budget 入口的 CUDA 预算限定 PyTorch native 默认 allocator，配置在资源存活期间固定，显式拒绝
可观察的非默认 pool、CUDA Graph 及不支持的 allocator 设置；进程 reserved 内存和普通
activation 另报。当前规划与实测边界见
[cache 管理实验](../../experiments/cache_management/README.md)。

[pool_referrers.py](../../cache/allocator/pool_referrers.py) 为 Python MemPool 全代检查提供
共享 native provider，启动时认证 CPython/header ABI。认证、编译、加载和调用失败直接
传播，不自动切换 provider。它与 allocator snapshot 是两个独立 provider。
此前接入版本已通过实际路径的空 MemPool、生命周期和完整 checkpoint 检查；通用 budget
短轨迹已退出。固定容量结果及其源码版本见
[固定容量实验](../../experiments/nosa_motivation/README.md)。

独立的 [execution/fixed.py](execution/fixed.py) 使用固定 P/NH：HBM-only 按 P 个 history
token 做 session LRU；offload 按 NH 准入，host backing 随 session 分配，逐层有限
P 槽以逻辑页偏移和 session tag 直接映射。当前要求 H<=P，history 与 prefill chunk
按 64 token 对齐，不提供任意容量下的 token LRU。dense 预取下一层全部历史 miss，
sync/async sparse 只取完整 query batch 的稀疏并集 miss，均复用逐层 pool。
候选整批在 GPU 执行后 discard，主 K/V 不写回 host history；CIS/indexer 候选尾部
仍按实际 session storage 计费。该入口可显式启用纯计算 CUDA Graph，cache、indexer、
attention 与 IO 留在图外，graph static allocated 和 private reserved 分别计入。
2026-10-05 的独立验收、正式计时、补充 P0/当前配对、主 profile 与独立 attention
reference 均已完成，run ID、请求延迟及归因边界见
[实验报告](../../experiments/nosa_motivation/README.md)。
[独立验收回执](../../docs/agents/acceptance/unified_runtime_20261005/nosa_final/receipt.json)
覆盖 H65536/A128 的全部 128 个候选输出；96 次 offload/HBM 对照逐位一致。
主 profile `refactor_final_nosa_profile_20261005_02` 的诊断证据通过复核，但 async 的
96 个有搬运层样本均未达到 90% page/stripe 重叠目标，比例为 62.34%–80.29%；
另外 96 个无搬运层样本单列。独立 attention reference
`refactor_final_nosa_attention_reference_20261005_01` 已通过保存操作数、身份与计时算术复核。
完整请求延迟仍取独立 benchmark；API 组合时间不等同于请求延迟。后续测量先限于 64K history。

2026-10-02，CPU 专项为 10 passed；单卡 SM90 完整 checkpoint 分别通过 8192+128 与
65536+1024 两组检查。每种方案从独立空 cache 构建 sparse prefix，再处理两次不同候选，
全部候选 hidden 均与 HBM 逐位相同，max_abs=0。以下为完整 64K+1K 正确性入口，
不构成 serving 性能结果：

```bash
CUDA_VISIBLE_DEVICES=1 PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH" \
  NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=1024 \
  .venv/bin/python -m pytest -s -q -p no:cacheprovider \
  models/nosa/tests/test_serving.py::test_cuda_serving_checkpoint_independent_prefixes_and_revisits
```

当前多用户固定顺序负载、逐请求及复访延迟见
[固定容量 motivation](../../experiments/nosa_motivation/README.md)。

## 模型与 attention 语义

默认 checkpoint 为 `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`。该配置包含 32 层、4096 hidden size、
16384 FFN intermediate size、32 个 query heads、2 个 KV heads、128 head dimension，
使用 Q/K/V 投影、SwiGLU、RMSNorm 和 LongRoPE。Q/K/V 合并为一次 `qkv_proj` GEMM，
gate/up 合并为一次 `gate_up_proj` GEMM。KV cache 保存 RoPE 后的 K 与原始 V，
采用此模型的 GQA 布局。BF16 权重约 16.37 GB，还需要 KV cache、激活和 kernel workspace 显存。

[normalization.py](normalization.py) 和 [feed_forward.py](feed_forward.py) 分别实现
RMSNorm 与 SwiGLU，接收显式维度、eps 和 bias。CUDA BF16/FP16 且关闭 autograd 时，
归一化和激活调用 FlashInfer `rmsnorm`、`fused_add_rmsnorm` 与 `silu_and_mul`；
CPU、FP32、开启 autograd 或不满足 kernel 布局要求时使用 PyTorch 数学路径。
`RMSNorm(x, residual)` 返回 `(normalized, sum)`，FlashInfer 路径会原地覆盖两个输入，
调用者须持有独立且可覆盖的 activation buffer；不传 residual 时只返回 normalized。
SwiGLU 的 `gate_up_proj` 一次生成 `[gate | up]` 并直接交给融合激活，随后由
`down_proj` 执行输出投影，前向不拼接独立的 gate/up activation。
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
[64K+1K 实验](../../experiments/nosa_mfu/README.md)，已于 2026-09-28 在 H200 上补测。

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
QK 在短上下文按[算子阈值](../../operators/nosa/README.md)使用单 kernel Triton。
其余已支持形状走 Triton。`CXLDSAGR_SM90_BACKEND=triton` 强制对照后端。
原生构建要求 nvcc、TVM FFI、共享 CUTLASS 和已安装的 FlashInfer 0.6.18，见[算子说明](../../operators/nosa/README.md)。
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
默认原生后端使用精确的 Top-33 / Top-64 选择，并在满足连续 query 与稳定 pool 条件时
共享每 KV head 的 CIS prefix 排名；Triton 对照使用 FlashInfer。query scratch 跨阶段和层
复用。原生后端对已验证布局合并有限值检查和增量压缩；检查失败时不写任何派生记录，
由模型撤销预约并抛出 `ValueError`。满足连续追加条件时，准备阶段同时生成 CIS 排名，
放在与评分 workspace 不重叠的临时区域；选块直接消费该排名。大 batch 的原生
specialization 将评分、pooling 和稳定选块合并，其他已支持形状保留分离路径。
支持的自有连续追加使用 checked C++ 入口合并准备与选块；请求持有可复用的 pinned
host flag，模型在校验成功后完成派生预约。验证 scratch 与下游输出不重叠；原始输入
非有限或输出别名非法时，在写入前拒绝。异步准备接口仍支持 CUDA Graph，带 host
有限值检查的完整入口不支持捕获。
上下文上限为 256K tokens。
`cache.truncate(length)` 同步回退原始和派生记录，失败 append 随模型事务一起回滚。
原 QA-only pattern 不启用此模式；
新增完整 NOSA pattern 对照从独立空 cache 构建 sparse prefix，记录 attention 实际消费的选块。
完整 sparse 路径的全模型 prefill/extend 测量见
[64K+1K 端到端 profile](../../experiments/nosa_mfu/README.md)。
2026-10-05 已完成当前完整模型的 native/Triton 独立验收、bench 与 profile，
run ID、实际 kernel 调度及测量边界见报告。2026-10-05 的 full-NOSA pattern 已完成
独立 observer 检查和三组 capture/analysis，运行分别为
`refactor_nosa_pattern_check_20261005_01`、`refactor_nosa_pattern_capture_20261005_01`；
这组结果不测量延迟。

```bash
source .venv/bin/activate
python -m models.nosa.infer --attention-mode sparse --sparse-backend triton \
  --prompt "请用简洁的中文解释 KV cache。" --disable-thinking --max-new-tokens 32
python -m serving.run_gr --attention-mode sparse --count 1 \
  --user-lengths 4096 --item-lengths 128 --prefill-chunk-size 1024
```

## Sparse KV offload

`NosaForCausalLM(..., attention_mode="sparse", cache_backend="offload",
offload_query_tile_size=128, offload_fetch_ctas=96, offload_overlap=True)`
以及同参数的 `from_pretrained` 启用普通 owned offload 后端。该 CUDA 路径要求 native SM90/Hopper、
BF16、D128、GQA16；dense mode、
CUDA reference attention、强制 Triton attention 与 CUDA Graph capture 不支持该路径，
调用明确失败。CPU offload adapter 只用于独立数值与事务检查。

历史 K/V 按原始 NHD 布局保存在 pinned local DRAM。本次 append 保留独立 device
副本，并异步写回 backing；全部层和 GPU 成功后统一提交。CIS、压缩 K/CIS 及稳定
CIS pool 常驻 GPU，新增压缩窗口只读 device append 和最多 31 个历史边界 token。
该 indexer 不为选块把整个 K prefix 搬回 HBM。

请求共享一层完整逻辑地址范围的 K/V staging。
GPU planner 对 `(KV head, block)` 去重，compactor 生成唯一页队列；每个
64-token 页拆为 8 个不交叠的 8-token stripe，由 fetch leader 原子领取
`(page, stripe)`，经 shared slot 和 96-thread barrier 广播。每个历史 K/V
16-byte 向量有唯一线程，以一次 `ld.global.cv.v4.u32` 读取并写入 HBM staging。
不同 CTA 可并行读取同页的不同 stripe，跨 query group 的复用只读取 HBM。

每个 stripe 的全部 writer 完成 stores / thread fences / 96-thread barrier 后，
leader 对 page ready 执行 `atom.acq_rel.gpu.global.add.u32`。跨 CTA 的 RMW 链
累计到 ready=8，TMA warp acquire 观察到 8 后执行 async-proxy fence，再读取完整
页；仅最后完成者累计一次整页字节。尾页空 stripe 不读取 host、没有 trace/payload，
但仍参与完成计数，保证尾页也完成同一协议。

所有 CTA 保留 persistent FA3；最多 96 个 CTA 使用 producer warpgroup 的 warp
1–3 fetch，warp 0 保留 TMA，两个 consumer warpgroup 保留 attention。只有
`KV_heads == 2 && ceil(queries / 8) * KV_heads == 256` 时，fetch 与 compute
都优先 head 1、再 head 0。每个 head 内 fetch 为 block 0 优先、其余 block 降序，
compute 保留原 cost 排序与 logical-batch ties。其他几何回到 block-major fetch
队列和原 FA3 调度。24 / 240 动态寄存器满足本 CTA 的 64512-register pool，
cooperative occupancy 检查保证全部 CTA 同时驻留。

native initialization 合并全容量 metadata 清零、历史 page-0 padding 和 strided
suffix staging；first-use planner 仍在后续有 stream 依赖的独立 launch。强串行
对照也使用同一初始化，再一次读取完整稀疏并集并运行原整批 FA3。初始化、planning、
compaction、prepare / sort / main / repair、输出/完成依赖与 launch gaps 全部计入
完整调用延迟。selection、CIS、causal mask 和每个 query 的算术次序保持原语义。
下一层等待当前操作完成后复用这份 staging。上述 owned 路径尚无有限 HBM slots、eviction
或跨请求块复用，CXL/RDMA 路径未验证。

生成 CLI 与 GR CLI 均提供 `--cache-backend offload`、`--offload-fetch-ctas`
（参与 fetch 的 attention CTA 数上限，默认 96）、`--offload-query-tile-size`（默认 128，仅控制
首次读取流量的 query 分组）及 `--no-fetch-overlap`。最后一项一次 fetch 完整稀疏
并集，再运行原 FA3 整批 attention，不按 query tiles 拆分计算。命令例如：

```bash
source .venv/bin/activate
CXLDSAGR_SM90_BACKEND=native python -m models.nosa.infer \
  --attention-mode sparse --cache-backend offload --offload-fetch-ctas 96 \
  --dtype bfloat16 --prompt "请解释 KV cache 的作用。" --disable-thinking
CXLDSAGR_SM90_BACKEND=native python -m serving.run_gr \
  --attention-mode sparse --cache-backend offload --offload-fetch-ctas 96 \
  --count 1 --user-lengths 4096 --item-lengths 128 --prefill-chunk-size 1024
```

2026-10-05 的当前实现补测已发布：主测为
`refactor_nosa_overlap_bench_20261005_01`，独立确认和 profile 分别为
`refactor_nosa_overlap_confirm40_20261005_01`、
`refactor_nosa_overlap_profile_20261005_01`。单层回放的全部 query 通过 FP32 参考，
串行/融合输出逐位一致；普通完整 32 层 resident/offload 从独立空 cache 构建
64K prefix 后，全部 1K extend hidden 也逐位一致。两类验收分别报告。

L0/L15/L31 的 20 次主测中，融合完整 API 延迟较同轮串行对照下降
29.47% / 24.33% / 23.02%，40 次独立复测确认收益。每层独立 profile 3 次，
全部 9 个样本的 page-envelope 与非空 stripe-copy 两套 ratio 均达到 90%，
最低为 91.0812%。两类全局窗口 union 在每个样本中相等；这不表示每页 envelope
都没有 stripe 间隙。输入沿用冻结的历史算子轨迹，默认调用每次重取完整稀疏并集；
本次不测 fixed P/NH 命中或 serving 延迟，也不作跨 GPU/日期的代码提速归因。
schema 3 分别记录非空 stripe-copy window、每页的 min(start)/max(end)
envelope 和 softmax update。stripe 起点在任务领取/解码与同步之后、host load
之前；终点在 stores / fences / barrier 之后、ready RMW 与计数之前。page
envelope 可能包含 stripe 间隙，因此分别将两类窗口取并集，与同一 softmax union
求交，再除以各自 union 持续时间。并行区间只计一次，不能用 envelope 间隙充当真实
copy。90% 验收要求每个 profiled sample 的两种 ratio 都 >= 0.9，中位数不能替代
全部样本通过。二者都只覆盖 attention 的 softmax 部分，不代表完整 attention
隐藏率、PCIe 线上占用或整体加速比；逻辑 payload 也不证明物理链路字节数。
以上为算子级结果，来源与测量边界见
[offload overlap 实验](../../experiments/nosa_offload_overlap/README.md)；完整模型的固定容量
hidden-output 延迟另见 [motivation](../../experiments/nosa_motivation/README.md)。本机传输
没有强制限速到 50 GB/s；此前分析与 resident profile 不属于本实现实测结果。

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
默认 resident 模式的模型与 cache 均驻留指定 CUDA GPU，dtype 支持 `bfloat16` 与 `float16`。
resident cache 的 K、V 分别为 `[层数, 容量, KV heads, head_dim]`，每层传入 FlashInfer 的布局为 NHD；
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

2026-10-05 的 [七阶段独立验收](../../docs/agents/acceptance/unified_runtime_20261005/nosa_non_timing_gpu1_ledger.json)
已通过：完整 native sparse、Triton sparse、dense 的 H64K/A1024 检查，普通 owned
resident/offload 完整 checkpoint 对照，pattern observer 检查与三组 capture/analysis，
以及 offload overlap 数值检查。完整 checkpoint 对照实际执行 1 项、无跳过；
这些结果只证明各自的数值与执行契约，性能由独立 bench/profile 报告。

2026-09-26 在 NVIDIA H200（SM90）上验证了 sparse 路径：CIS/两阶段选择的独立参考、
BF16/FP16 Triton attention 与 indexer、严格 A/delta 加载、缓存回滚、生成/GR CLI，
以及 4097-token 场景的 65→64 块裁剪。另以本地 NOSA-8B 完成一次
`--user-lengths 4096 --item-lengths 128` 的 sparse GR 前向，总长 4224，
stable prefix 4124、candidate suffix 100。此检查不构成吞吐测量或模型质量评估。
当时环境为 PyTorch `2.10.0+cu132`、Triton `3.6.0`、FlashInfer `0.6.18`；
与当前 `pyproject.toml` / `uv.lock` 中的 torch/triton 版本不同；上述验证记录对应 2026-09-26 的环境。
2026-10-05 已使用 PyTorch `2.12.1+cu130` / Triton `3.7.1` 完成
[dense 与 sparse native/Triton](../../experiments/nosa_mfu/README.md) 的独立数值验收、
正式计时与 profile，具体覆盖范围见报告。

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

新增 [完整 checkpoint offload 检查](tests/test_offload_checkpoint.py) 仅在设置
`NOSA_OFFLOAD_CHECKPOINT` 后运行：严格加载一次全部 32 层 BF16 权重，将运行时上下文
上限扩至 66560，resident/offload 分别从空 cache 构建 64K prefix，再比较全部 1K
extend hidden 与缓存长度、内存统计。默认每 chunk 1024，offload 的首次读取流量
按 128 queries 分组；它是正确性检查，不计时、不生成报告。
默认读取本轮 native sparse bench 中保留的 request JSON，
可用 `NOSA_OFFLOAD_REQUEST` 指定另一个相同长度的 NOSA GR 请求。

完整 32 层 checkpoint 检查 1 passed：resident/offload 分别从独立空 cache
构建 64K sparse prefix，再执行 1K extend，全部 normalized hidden 逐位相同，max_abs=0。
提交后的 cache 分配为 resident HBM `2262627840 B`、offload HBM
`150825168 B`、offload pinned host `2181038080 B`；不包含模型权重，也不是进程峰值显存。
测试只验收正确性；包含加载或编译的测试总耗时不是模型性能或 overlap 测量。

```bash
NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  .venv/bin/python -m pytest models/nosa/tests/test_offload_checkpoint.py \
  -s -q -p no:cacheprovider
```

## GR 前向与性能测量

共享 GR 请求生成使用 [request_format.py](request_format.py)：NOSA 聊天模板、tokenizer 与请求预算适配。
用法见 [GR 生成器](../../GR/README.md)。
dense 单请求性能测量入口见 [GR 实验](../../experiments/nosa_mfu/README.md)，
2026-10-05 的 dense full-prefill/extend wall 中位数为 3506.12 / 81.81 ms，
MFU 为 62.61% / 63.02%；测量边界与 run ID 见报告。
通过 `bash experiments/nosa_mfu/scripts/dense.sh --help` 查看独立验收、计时和 profile 的参数。
本轮 dense、native sparse、Triton sparse 的完整 H64K/A1024 独立验收均已通过；
各自的正式计时和 profile 已按独立阶段完成并发布，结果见实验报告。

GR 前向使用 `model(input_ids, cache, return_hidden=True)` 返回本次调用所有输入 token 的最终
normalized hidden states，跳过 LM head；不与 `logits_to_keep` 同时使用。
生成 CLI 仍默认返回 logits。64K+1K 实验测量完整 prefill 与 prefix 已就绪后的候选 extend，
不执行自回归生成。

分块执行 `executor.model_executor.run_chunks` 返回最后一个 chunk 的输出；旧
`serving.runner` / `run_gr` 从中取末 token hidden，每请求独立创建和释放 cache。
`serving.persistent` / `run_multi_user` 则通过 NOSA serving adapter 返回完整 candidate
hidden，并跨请求保留固定用户历史；两条路径的输出范围和 cache 生命周期不同。
