# 本地 GR serving

直接消费 [GR input generator](../GR/README.md)，在单进程、单 GPU 上串行执行 NOSA。
每条请求独立分配 resident KV cache，执行 stable prefix prefill 和 candidate extend，
返回末 token 的 normalized hidden。当前不运行 LM head 或文本生成，没有跨请求缓存复用、
批调度、网络服务或到达时间回放。

## 命令行

从仓库根目录、已准备的 Python 环境运行：

```bash
python -m serving.run_gr --count 1
python -m serving.run_gr --attention-mode sparse --count 1
python -m serving.run_gr \
  --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --device cuda:0 --dtype bfloat16 --count 2 \
  --user-lengths 4096 --item-lengths 128 --prefill-chunk-size 1024
```

默认 count 为 1、用户数 1000、seed 42、chunk size 1024，使用 NOSA 的现有长度分布、
Beauty 热度曲线和 synthetic 文本。tokenizer 来自同一 checkpoint。
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
`--sparse-backend auto` 在 CUDA 上使用 SM90 Triton。两者均为全 HBM resident cache；
local DRAM offloading 尚未实现。

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
