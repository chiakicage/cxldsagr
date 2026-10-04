# 本地 GR serving

直接消费 [GR input generator](../GR/README.md)，在单进程、单 GPU 上串行执行请求。
`serving.run_multi_user` 跨请求保留用户的固定 prefix，支持 DeepSeek V3.2 约 8B
输入复制代理负载和完整 NOSA-8B。预算模式按 HBM/DRAM 字节配额做 session LRU；
DeepSeek ECHO 的固定容量模式只按共享 host 页容量准入，P 和 NH 由调用者指定。
DeepSeek `echo/serial_sparse` 只为固定 history 保留用户 session；候选主 KV 和
indexer 使用 backend 的共享 GPU 临时空间，候选主 KV 不写回 DRAM。
已有 `serving.run_gr` 继续提供逐请求创建并释放 cache 的 NOSA 入口。两个入口均无网络服务、
批调度或模拟到达时间等待。

## 持久多用户入口

从仓库根目录运行；默认首先使用 DeepSeek V3.2 的 HBM 方案：

```bash
python -m serving.run_multi_user --help
python -m serving.run_multi_user \
  --model deepseek_v32 --scheme hbm --model-path /preset-models \
  --num-users 8 --count 16 --history-tokens 16384 --candidate-tokens 1024 \
  --hbm-budget-gib 1 --dram-budget-gib 16 --device cuda:0
python -m serving.run_multi_user \
  --model deepseek_v32 --scheme echo --model-path /preset-models \
  --resource-mode fixed-pools --sparse-pool-tokens 32768 --host-arena-tokens 1048576 \
  --num-users 16 --count 32 --history-tokens 65536 --candidate-tokens 128 \
  --context-limit 65664 --chunk-size 1024 --device cuda:0
python -m serving.run_multi_user \
  --model nosa --scheme overlap --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --num-users 8 --count 16 --history-tokens 16384 --candidate-tokens 1024 \
  --hbm-budget-gib 1 --dram-budget-gib 16 --device cuda:0
```

DeepSeek 支持 `hbm`、`echo`、`serial_sparse`、`dense_prefetch`；NOSA 支持 `hbm`、
`serial_sparse`、`dense_prefetch`、`overlap`。DeepSeek 使用 checkpoint 前三层的独立
权重副本和对应层输入构成约 8B 负载，不含 MoE，不是训练得到的 8B 模型。两种模型各方案
保持对应模型的稀疏 attention 语义；`dense_prefetch` 表示逐层读取全部历史 KV 的搬运方式。

用户访问遵循 GR 中的 Beauty 热度曲线，加权有放回采样，默认 seed 42。
默认不设复访上限，按固定热度概率独立有放回抽取 `--count` 次。实际人数和复访次数由
采样结果统计，`--num-users` 是概率向量的用户池大小。旧 `--max-revisits` 仅在显式设置
时限制复访，达到上限的用户退出抽样，此时总数为上限。不强制覆盖用户或补齐次数。
`--history-tokens` 是固定指令加 history 的完整 prefix 长度；`--candidate-tokens` 是完整
变化 suffix 长度。复访时验证 prefix token identity，只保留固定 prefix。
DeepSeek `echo/serial_sparse` 通过 `retained_session_capacity` 按 H 准入，调用
`extend_candidate` 一次整批执行 GPU 临时候选，不沿用 history prefill 的 chunk 划分；
正常结束后 discard 候选，不推进 history 长度或覆盖其 indexer 状态。候选失败直接
报错终止，runner 释放该用户 session，不恢复或重试。在已规划的执行上限内改变
候选长度 A 不会导致相同 history 重建。`hbm/dense_prefetch` 和其他后端仍使用原持久
extend 后 truncate 的生命周期；
直接调用 `backend.extend` 也继续持久提交新增 token。
`is_revisit` 与实际 `prefix_cache_hit` 分别报告；被淘汰后的复访仍计入复访。
默认每层 query chunk 为 1024。DeepSeek 使用 `--sparse-pool-tokens` 配置逐层共享
token pool（P），`--host-arena-tokens` 配置共享 host arena（NH）。`--resource-mode fixed-pools`
要求显式 NH，不接受 HBM/DRAM 字节子预算和独立 W；workspace 根据实际 chunk 与 candidate
大小自动规划。indexer 和临时空间仍实际占用显存，但其预留量不再触发 session 淘汰。
NH 按 `padded(H)=ceil(H/64)×64` 为每个用户分配页；同长度 history 可容纳的用户数为
`floor(NH/padded(H))`。候选 KV 和一层 `[H+A,d]` 合并 indexer K/scales workspace
只计入共享 GPU 资源，不乘以用户数。P 只表示每层可淘汰的历史 HBM 槽数，候选 GPU
尾部另按 A 上限预留。
该模式支持 DeepSeek `echo/serial_sparse`，不保证任意 P/NH 都能装入机器，运行失败须明确报告。
预算模式仍可用 `--workspace-query-tokens` 声明最大执行 workspace；原 `--deepseek-slots`
已移除，显式使用会报错。
`deepseek_v32_motivation` 通过 backend/runner 接口运行固定 P/NH 的四方案对照。
该入口的 HBM-only 使用独立 HBM history token 配额 P；三个 offload 方案使用 NH
host 页配额和相同的 P 槽历史 cache，dense 仅预取完整历史中的 miss。四方案候选
均整批在 GPU 临时执行。独立 token 配额不借用 host pages，也不换算成字节子预算。

NOSA 默认 GR 生成长度上限为 32768；超过该长度需显式传入 `--context-limit`，例如
64K history 加 128-token candidate 使用 `--context-limit 65664`。该选项只声明请求
构造与后端执行边界，不验证长上下文任务质量；单个 64K resident NOSA session 需要超过
2 GiB 的 cache 预留，预算须相应配置。

预算包含缓存持有的 KV、派生记录、metadata、staging 和 cache scratch；模型权重及一般
计算临时内存不在 cache budget 内。HBM 方案不使用 DRAM 历史 backing，其 CPU
验证和统计 scratch 仍计费。NOSA 的逻辑地址
staging 与整个用户 session 的 LRU 淘汰，不代表已有有限 token slots / eviction。

runner 从构造到关闭独占 backend 的准入所有权，构造失败回滚本次新资源，保留原有
有效 plan。CLI 分别声明 H 的保留上限、A 的执行上限和 H+A 的上下文上限，越界请求在
准入前拒绝。history-only session 不缩小完整请求的执行范围。共享候选空间只能在
执行 lease 内借用，复用前须等待相关 GPU 操作完成；关闭 runner 释放用户，再由外层
关闭共享资源。已有实验仍保留原 run 和测量边界，cache 统计不能代替进程峰值。
DeepSeek GPU candidate 路径已通过短 GPU 正确性检查；完整多用户容量与实际峰值尚未
实测，旧 ECHO 容量运行不作为新实现的容量证明，当前状态见
[ECHO cache 实验](../experiments/deepseek_v32_echo_cache/README.md)。

stdout 输出启动配置、逐请求 JSON 统计及 hidden shape/dtype/device，最后输出请求数、
复访数、命中数、淘汰数和平均延迟；不输出 hidden 向量或输入 token 数组。
该入口没有独立预热，逐请求时间包含首次使用时触发的 JIT 编译，排除权重加载和 GR 文本生成。
正式方案对照使用[独立实验入口](../experiments/gr_serving/README.md)完成预热、空 cache
重置、逐请求数值比较和报告，不能把本 CLI 的首次运行时间直接当作该实验结果。
成功或失败退出时均关闭持有的用户 sessions 和 backend 共享资源。

## 逐请求 NOSA 入口

从仓库根目录、已准备的 Python 环境运行：

```bash
python -m serving.run_gr --count 1
python -m serving.run_gr --attention-mode sparse --count 1
CXLDSAGR_SM90_BACKEND=native python -m serving.run_gr \
  --attention-mode sparse --cache-backend offload --offload-fetch-ctas 96 \
  --dtype bfloat16 --count 1 --user-lengths 4096 --item-lengths 128
python -m serving.run_gr \
  --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --device cuda:0 --dtype bfloat16 --count 2 \
  --user-lengths 4096 --item-lengths 128 --prefill-chunk-size 1024
```

默认 count 为 1、用户数 1000、seed 42、chunk size 1024，使用 NOSA 的现有长度分布、
Beauty 热度曲线和 synthetic 文本。tokenizer 来自同一 checkpoint。
`--cache-backend offload` 要求 sparse 模式及 native SM90 BF16/D128/GQA16，历史 K/V
保存在 pinned local DRAM，CIS/压缩记录常驻 GPU，并共享一层完整逻辑地址 staging。
融合主 kernel 中，所有 CTA 保留 attention；启用 fetch 的 CTA 用三个空闲
producer warp 动态领取唯一页中的 8-token stripe。每个历史向量只读一次，acq_rel
完成链累计到 ready=8 后 TMA acquire 并复用 HBM。两 KV heads / 256 work batches
时 fetch 与 compute 都优先 head 1 再 head 0，其他形状保留原调度。
`--offload-fetch-ctas` 默认 96，为参与 fetch 的 CTA 数上限；
`--offload-query-tile-size` 默认 128，仅分组统计首次读取字节。
`--no-fetch-overlap` 共用新 native initialization，一次 fetch 整批稀疏并集，再
运行原完整 attention。有限 HBM slots / eviction 尚未实现，CXL/RDMA 路径未验证。
stdout 每条请求输出一行 JSON 完成摘要，包含请求信息、长度和 feature 的 shape/dtype/device，
不输出 hidden 向量。timestamp 仅为模拟到达时间，不能由此推算服务 QPS。

## Python 接口

```python
import torch

from executor.model_executor import ModelExecutor
from GR.input_generator import create_input_generator
from models.nosa.infer import DEFAULT_MODEL_PATH
from models.nosa.model import NosaForCausalLM
from serving.runner import GRRunner

model = NosaForCausalLM.from_pretrained(DEFAULT_MODEL_PATH, device="cuda:0", dtype=torch.bfloat16)
executor = ModelExecutor(model, chunk_size=1024)
runner = GRRunner(executor, device="cuda:0")
generator = create_input_generator(model="nosa", tokenizer=DEFAULT_MODEL_PATH)
for result in runner.run(generator.iter_generate(2)):
    print(result.metadata, result.last_hidden.shape)
```

`run` 惰性读取请求，直接使用完整 `input_ids` 和 `stable_prefix_tokens`，不重新 tokenize。
请求成功或失败均释放当前 session；错误向调用者传播，不自动重试。生成器的
`common_prefix_tokens` 表示文本共同前缀，不代表已命中 KV cache。

框架流程为 GR → serving → [executor](../executor/README.md) →
[NOSA layers](../models/nosa/README.md) → [operators](../operators/README.md)，
由 [cache manager](../cache/README.md) 统一管理 KV。
默认使用 Full Attention；`--attention-mode sparse` 启用 NOSA 完整选块与 CIS attention，
`--sparse-backend auto` 在 CUDA 上使用 SM90 dispatcher，NOSA-8B attention 默认
native FA3；indexer 按形状调度 CUDA/CuTe 或 Triton。resident 模式可显式用
`CXLDSAGR_SM90_BACKEND=triton` 作对照。
完整 32 层 checkpoint 检查 1 passed：resident/offload 分别从独立空 cache
构建 64K sparse prefix，再执行 1K extend，全部 normalized hidden 逐位相同，max_abs=0。
单层完整调用性能与 stripe / page-envelope overlap 独立测量，不作为 serving 性能，
见 [NOSA offload 实验](../experiments/nosa_offload_overlap/README.md)。
