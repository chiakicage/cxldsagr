# 本地 GR serving

直接消费 [GR input generator](../GR/README.md)，在单进程、单 GPU 上串行执行 NOSA。
每条请求独立分配 KV cache（默认 resident，可显式选择 offload），执行 stable prefix
prefill 和 candidate extend，
返回末 token 的 normalized hidden。当前不运行 LM head 或文本生成，没有跨请求缓存复用、
批调度、网络服务或到达时间回放。

## 命令行

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
