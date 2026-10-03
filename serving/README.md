# 本地 GR serving

直接消费 [GR input generator](../GR/README.md)，在单进程、单 GPU 上串行执行请求。
`serving.run_multi_user` 跨请求保留用户的固定 prefix，支持 DeepSeek V3.2 约 8B
输入复制代理负载和完整 NOSA-8B，并在统一 HBM/DRAM 字节预算下按用户 session 做 LRU 淘汰。
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
变化 suffix 长度。复访时验证 prefix token identity，只保留固定 prefix，candidate 执行后
截短。`is_revisit` 与实际 `prefix_cache_hit` 分别报告；被淘汰后的复访仍计入复访。
默认每层 query chunk 为 1024，DeepSeek 稀疏 pool 为 4096 slots。
NOSA 默认 GR 生成长度上限为 32768；超过该长度需显式传入 `--context-limit`，例如
64K history 加 128-token candidate 使用 `--context-limit 65664`。该选项只声明请求
构造与后端执行边界，不验证长上下文任务质量；单个 64K resident NOSA session 需要超过
2 GiB 的 cache 预留，预算须相应配置。

预算包含缓存持有的 KV、派生记录、metadata、staging 和 cache scratch；模型权重及一般
计算临时内存不在 cache budget 内。HBM 方案不使用其获准的 DRAM 容量。NOSA 的逻辑地址
staging 与整个用户 session 的 LRU 淘汰，不代表已有有限 token slots / eviction。

ECHO 的 MFU/cache 策略与 DeepSeek MFU 存在问题，当前模型内 baseline 比较尚未成立。
DeepSeek 临时 cache scratch 的完整预算覆盖也待审计；上述接口预留及边界统计不等于
全过程峰值验证。当前判断与补测范围见[系统状态](../docs/agents/system/implementation-status.md)。

stdout 输出启动配置、逐请求 JSON 统计及 hidden shape/dtype/device，最后输出请求数、
复访数、命中数、淘汰数和平均延迟；不输出 hidden 向量或输入 token 数组。
该入口没有独立预热，逐请求时间包含首次使用时触发的 JIT 编译，排除权重加载和 GR 文本生成。
正式方案对照使用[独立实验入口](../experiments/gr_serving/README.md)完成预热、空 cache
重置、逐请求数值比较和报告，不能把本 CLI 的首次运行时间直接当作该实验结果。
成功或失败退出时均关闭持有的用户 sessions。

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

## 截断 ECHO 适配

独立的 [echo_runner.py](echo_runner.py) 接入研究用 ECHO 环境，不改变上面的 NOSA 入口。
它配合 [前 1-5 层模型适配器](../models/deepseek_v32/echo_adapter.py)，显式保留 prefix handle，
逐候选分支恢复预取阈值、返回完整 candidate hidden、等待传输完成后释放私有 suffix。
prefix 必须 64-token 对齐；暂时要求整个活动 context 能放进每层 device pool，容量不足明确失败。
runner 本身只管理显式 handle；自动复用与淘汰由下述 manager 管理。
已通过真实三层模型 4K + 1K、64K + 1K 的受控 GPU 缓存正确性检查；不等于完成论文性能测量。
五层（包括两层完整 MoE）也已通过上述两种长度的四路径交叉检查。默认仍为三层，
通过 `num_layers=5` / `SPARSEGR_ECHO_TEST_LAYERS=5` 选择新配置。

支持选择 `resident`、`sparse_sync`、`echo_gr_adapted` 和 `dense_prefetch`，
不同模式必须在新进程中启动。后三者使用 ECHO host pool；
`echo_gr_adapted` 修正 extend recall 对空闲槽的处理，并管理 GR 分支的预测状态，
不称为原封不动的 ECHO server。Python API `open_echo_runner(..., kernel_patch=None)`
默认仍用原 fused prefetch kernel；显式传入
`models.deepseek_v32.echo_kernel.PREFETCH_PHASE_PATCH_ID` 才启用
`echo_sm90_prefetch_phase_snapshot_v1` header overlay。
该修复处理 64K 原 kernel 的共享 phase flag 竞态，不改上游 checkout 或安装包；
source/patched header SHA、include manifest、扩展及初始化器 SHA 随 provenance 记录。
编译器选择先于 DeepGEMM/SGLang 导入，每进程仅一次，退出后不恢复为 native。
每次 forward 后的全设备同步是初期正确性边界；保留该边界的完整回放只能作为同步原型测量，
不能冒充优化后在线 serving 性能。入口见 [GR 回放实验](../experiments/gr_cache_serving/README.md)。
offload 模式还应用 `echo_extend_recall_int64_address_v1`，修复大 Host pool 的
extend recall 地址乘法溢出；记录 `provenance["recall_address_fix"]`，不修改上游文件或 policy。
`dense_prefetch` 另加两个逐层预取 buffer，保留原 write pool，相关额外容量记录在
`provenance["dense_memory_accounting"]`。新 [echo_budget.py](echo_budget.py) 为容量实验规划
共同 HBM 上限下的有限 resident 用户数，offload 保留显式指定的 device pool；
实验入口另外执行 Torch allocator cap 与 NVML 采样保护，不将不同实际分配量说成相同。
`mem_fraction_static` 只供上游容量规划，不能单独当作全进程显存硬上限。
实际模型保留原 checkpoint 的 FP8 量化与 scales；BF16 指 activation 和 MLA KV，
不是全部权重精度。ECHO 的重叠对象是 fetch/indexer；同层 fetch/main-attention 流水未实现，
dense 跨层传输是否实际重叠仍须 timeline 证明。

[echo_cache_manager.py](echo_cache_manager.py) 提供串行 `EchoCacheManager.execute(request)`：
以完整 prefix token 摘要和当前模型实例校验同用户复用，候选始终私有；
host 按整用户 LRU 淘汰，预留候选页和 guard page。预算单位是 allocator token slot，
不是含全部 index、激活和元数据的总字节预算。manager 独占其 runner 的 prefix 生命周期，
不能与其他 manager 或手动分配交错使用；健康 `close()` 释放 prefix 但不关闭 runner。

默认 HBM 使用 ECHO 原生策略。可选 `hbm_retained_users=N` 根据已观察到的衰减频次、
recency 决定哪些用户允许在请求后保留副本，并驱逐其他用户副本；不主动预取或锁定热用户。
它是用户级保留消融，不是精确的 block 热度策略、HBM 字节预算或 HBM 命中统计。
代码不读取生成器的 `user_heat_weight`。真实到达时间回放仍未实现。

独立 GPU 正确性入口读取真实 checkpoint，并使用现有 GR 生成器产生 4K + 1K 请求：

```bash
bash scripts/run_echo_tests.sh resident
bash scripts/run_echo_tests.sh sparse_sync
bash scripts/run_echo_tests.sh echo_gr_adapted
bash scripts/run_echo_tests.sh dense_prefetch
bash scripts/run_echo_tests.sh all
```

`all` 还将三种 offload 输出与临时 resident 参考比较；
`SPARSEGR_ECHO_TEST_TRACE` 可指定准备好的 GR JSONL，保留其完整输入长度。
脚本默认仅给 `echo_gr_adapted` 启用 `SPARSEGR_ECHO_TEST_KERNEL_PATCH=phase_snapshot`；
设置为 `native` 可禁用。该脚本默认与 Python API 的 `kernel_patch=None` 不同。

正确性测试还默认用 `test_only_logical_topk_order_v1` 对原 indexer 输出做逻辑次序重排，
仅改变排列，保留 selected multiset、重复项和 `-1`。原子输出次序及边界 tie 都可能导致
原生结果不确定；控制不解决 membership tie，额外同步也不能用于性能实验。
`SPARSEGR_ECHO_TEST_TOPK_ORDER=native` 可关闭。参考校验输入、checkpoint 元数据、
ECHO revision/patch SHA 和控制 ID/源码 SHA，并检查 prefix KV/index 快照不被候选污染。
当前 4K + 1K 与 64K + 1K 的四路径 `all` 均已通过，包括修复版 ECHO 与同批 resident
文件交叉参考，使用上述 logical 控制及原容差 `rtol=0.02, atol=0.02`。
64K 检查取 V2 Beauty 128 用户池、512 请求中的三用户子序列，不是完整回放。
4K native 次序四路径也曾通过；64K native 次序不宣称通过，原版 ECHO kernel 仍有挂起。
这些检查没有性能测量或实验报告产物。

环境构建和实际验证状态以 [实验说明](../experiments/gr_cache_serving/README.md) 为准。
