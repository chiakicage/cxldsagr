# NOSA query-aware pattern：GR 65536 + 1024

## 实验目的与测量边界

检查 NOSA-8B 在 GR 输入上的 query-aware 稀疏选块：每个 candidate token、每层、
每 KV head 选择 64 或 32 个逻辑块，再沿 1024 个 candidate queries 合并去重，求需要访问的
K+V 块容量以及它占完整 66560-token KV cache 的比例。

已完成两种预算的同请求采集：64-block 并集为 **788.34375 MiB（37.90%）**，
32-block 并集为 **526.65625 MiB（25.32%）**。逐层/head 结果和时间估算见下文。
在 attention MFU 减半、50 GB/s 的 overlap 模型下，阈值扫描建议32-block用 **21.5%**、
64-block用 **37%**；相较30%，未隐藏fetch分别减少 **21.94% / 24.36%**，详见末节。

复用 [dense 性能实验](../nosa_gr_65536_1024/README.md) 的精确输入，源 run ID 为
`flashinfer_merged_gemm_65536_1024_20260925_02`，源文件为
`experiments/nosa_gr_65536_1024/output/data/flashinfer_merged_gemm_65536_1024_20260925_02/request.json`。
输入为 synthetic GR 内容，user 623、visit 0、seed 42；instruction 28 + history 65508
= 已缓存 prefix 65536，candidate suffix 1024，总计 66560。不是 1K 个推荐商品。

使用原 NOSA-8B BF16 权重和 **dense Full Attention 前向产生的 RoPE 后 Q/K**。
每层候选阶段旁路 indexer，记录选择后仍执行原 dense attention；选择不改变后续层激活。
因此结果描述 dense 激活上的稀疏 pattern，不等于逐层 sparse attention 传播后的 NOSA 轨迹。
不执行 LM head、自回归生成、query-agnostic/CIS 或 sparse attention/offloading。

本实验统计每个选中块仅计一次的 **unique K+V payload**；不测延迟、吞吐、带宽或总线流量，
不计 indexer 读取、KV 写入、元数据、重复 fetch 或缓存命中。一次采集，无计时预热/重复。
模型上下文仅在实验进程内从 32768 覆盖到 66560，保留原 LongRoPE factors；不评价超长上下文质量。
另使用已有 dense MFU 与固定 50 GB/s 带宽，推算 sparse attention、逐层 GPU 计算及 KV fetch
时间，见末节。这部分是基于已有数据的估算，没有新增 sparse kernel 或 PCIe 性能测量。

## 选择算法与官方依据

参考 [NOSA 论文 v2 §3.1](https://arxiv.org/html/2510.13602v2#S3.SS1)、
[Appendix B.2.2](https://arxiv.org/html/2510.13602v2#A2.SS2.SSS2)，以及官方固定提交
`1cbee77d607f9051b206a09c862bea28becb9e67` 的
[模型代码](https://github.com/thunlp/NOSA/blob/1cbee77d607f9051b206a09c862bea28becb9e67/modeling_llama_nosa.py)、
[stage1 数学参考](https://github.com/thunlp/NOSA/blob/1cbee77d607f9051b206a09c862bea28becb9e67/dependencies/infllmv2_cuda_impl/tests/test_stage1.py)
和 [pooling kernel](https://github.com/thunlp/NOSA/blob/1cbee77d607f9051b206a09c862bea28becb9e67/dependencies/infllmv2_cuda_impl/csrc/max_pooling_1d.cuh)。

索引从 0 开始。对绝对 query 位置 `p`、KV head `h`：

1. `Kc[j,h] = mean(K[16*j:16*j+32,h])`，仅形成完整的 32-token 窗口。
2. 每个 Q head 单独计算 `softmax(Q @ Kc.T / sqrt(head_dim))`；softmax 前屏蔽
   `16*j+31 > p` 的窗口。NOSA-8B 每 KV head 对应连续的 16 个 Q heads，概率按组求和。
3. 64-token block `b` 的评分为压缩位置 `[4*b-1, 4*b+3]` 的最大概率和，两端截断。
   这是五个重叠窗口的 max，不是 sum、mean 或 overlap 加权。
4. 固定选择 sink block 0，local 取 `[p//64-15, p//64]` 共 16 块（含当前块）。
   64-block 配置从其余 causal blocks 选 query-aware top47，32-block 配置选 top15。
   sink/local 仅在选块时排除，仍参与评分归一化。

官方原始 47 个 dynamic blocks 分为 16 query-aware + 31 query-agnostic；本实验全部改成
query-aware。纯 query-aware 不需要 checkpoint 的 A/delta/CIS，现有 dense 加载器继续忽略它们。
官方部分 kernel 的闭区间将 `local_blocks=16` 实际展开为 17 块；本实验严格采用用户指定及
论文预算的 **1 + 16 + 47**，以及新增的 **1 + 16 + 15**。
当前块仍需 token causal mask，但容量统计按完整块计。

[模型 indexer](../../models/nosa/indexer.py) 使用 PyTorch，均值、点积、softmax 和概率聚合均为
FP32，CUDA 点积禁用 TF32，按 64 个 queries 分批评分。分数相同优先较小 block ID，输出按
block ID 升序；不承诺与上游低精度 kernel 在临界 top-k 上逐位相同。不引入自定义高性能 kernel。

## 指标与产物

对每层每 head 独立计算 `U[layer,head] = union(selection[query,head])`，只沿 query 去重。
不同 KV heads 或不同层的同号 block 仍是不同 KV 数据，层总量/模型总量通过字节数相加。

| 容量项 | BF16 K+V 容量 |
| --- | ---: |
| 每 KV head 的一个 64-token block | `64 × 128 × 2 × 2 = 32768 B`，32 KiB |
| 每层每 KV head 完整 cache，1040 blocks | 32.5 MiB |
| 每层两个 KV heads 完整 cache | 65 MiB |
| 全部 32 层完整 cache | 2080 MiB |

每 head 比例为 `len(U)/1040`。层比例分母为 65 MiB，模型比例分母为 2080 MiB。
分别报告 prefix 与 candidate 分项，其各自比例的分母对应 65536 和 1024 tokens。
1K candidates 覆盖 16 个新块，local 并集为 `[1009,1039]` 共 31 块；加入 sink 与首个 query
的额外块，每个 head 的总 union 至少为 `block_budget + 15`：64-block 为 79 块，
32-block 为 47 块；上界均为 1040 块。

成功运行的产物位于 `output/data/<run_id>/`：

- `block_ids.npy` / `valid_mask.npy`：`[32,1024,2,block_budget]`，逐 query 选择及有效性；ID 为 int32。
- `union_mask.npy`：`[32,2,1040]`；`head_stats.csv` 共 64 行，`layer_stats.csv` 共 32 行。
- `summary.json` / `report.md`：统计口径、每 head/层/模型数据和 prefix/candidate 分项。
- `capacity_by_layer.png` / `.svg`：各层 KV 容量和比例；`union_heatmap.png` / `.svg`：两 head 的并集。
- `request.json`、`execution.json`、`attention_shapes.json`、`metadata.json` 与 `source/`：
  精确请求、运行边界、实际形状、硬件与依赖、权重/输入/源码哈希及源码快照。

stdout/stderr 分别存 `output/log/<run_id>/`；未运行 profiler，`output/profile/<run_id>/` 为空。
产物默认不进入 Git；README 保留必要结果，完整数组和图使用上述普通路径。

## 运行方式与调用模块

从仓库根目录运行，环境使用现有基础依赖和 `analysis` 组：

```bash
uv sync --group analysis
bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh --help
bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh <run_id>
```

默认预算 64 块、权重 `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`、设备 `cuda:0` 和上述源请求。
使用 `--block-budget 32` 选择 1 sink + 16 local + 15 query-aware。可显式指定：

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh <run_id> \
  --request-file /path/to/request.json --model-path /path/to/NOSA-8B --device cuda:0
```

脚本定位仓库根目录、拒绝覆盖旧 run ID，在系统临时目录完成 capture → analyze 后发布成功产物。
源请求缺失会报错，不重新生成另一份输入。失败日志留在终端提示的系统临时目录，不进入实验产物。
`src/` 不实现 shell 调度；也可查看独立入口或重新分析已有数组：

```bash
.venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.capture --help
.venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.analyze --help
.venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.analyze \
  experiments/nosa_indexer_pattern_65536_1024/output/data/<run_id>
```

调用模块为 [capture.py](src/capture.py)、[analyze.py](src/analyze.py)、
[NOSA model/indexer](../../models/nosa/README.md) 和
[共享 executor](../../executor/model_executor.py)。显式复用原性能实验
`experiments.nosa_gr_65536_1024.src.capture.execution_split` 检查请求边界，
`experiments.nosa_gr_65536_1024.src.sources.source_hashes` 记录源码。

正确性检查通过 `bash scripts/run_tests.sh [cpu|gpu|all]` 运行；其结果输出终端，不作为实验产物。

## 64-block 实验结果与结论

已完成 run `query_aware_fp32_65536_1024_20260925_01`（2026-09-25 15:25 UTC），运行命令：

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh query_aware_fp32_65536_1024_20260925_01
```

硬件为 `cuda:0`，SM90、132 SM；`nvidia-smi` 名称为 NVIDIA M403（143771 MiB），
PyTorch 名称为 NVIDIA H200（143167 MiB），驱动 570.124.06。实际环境为
PyTorch **2.10.0+cu132**、CUDA 13.2、FlashInfer 0.6.18、NumPy 2.5.0、
Matplotlib 3.11.1、safetensors 0.8.0、tokenizers 0.23.2。模型 BF16，indexer FP32；
32 层均记录 `Q=[1024,32,128]`、`K=V=[66560,2,128]`。无计时预热，单次采集。

本次沿用已有基础运行环境，其 PyTorch/CUDA 版本与当前 lockfile 的 2.12.1+cu130 不同；
没有将其宣称为该锁定基础环境的验证。缺失的 analysis 组按 `uv export --frozen --only-group analysis`
导出后安装；完整实际依赖、输入与权重 SHA256 记录在本 run 的 `metadata.json`。

**全部层合计：826638336 B = 788.34375 MiB，占完整 2080 MiB 的 37.90%。**
其中 prefix 为 756.34375 MiB / 2048 MiB（36.93%）；candidate 为 32 MiB / 32 MiB（100%）。
在每个选中块只读取一次的口径下，相较完整 cache，所需 KV 容量减少 62.10%。

下面层号和 KV head 号均从 0 开始。括号为占对应完整 cache 的比例：head 分母 32.5 MiB，
层分母 65 MiB。完整精确 bytes 和 prefix/candidate 分项见 CSV/JSON。

| 层 | Head 0 块数 | Head 0 MiB（%） | Head 1 块数 | Head 1 MiB（%） | 层合计 MiB（%） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 209 | 6.531（20.10%） | 147 | 4.594（14.13%） | 11.125（17.12%） |
| 1 | 233 | 7.281（22.40%） | 276 | 8.625（26.54%） | 15.906（24.47%） |
| 2 | 314 | 9.812（30.19%） | 221 | 6.906（21.25%） | 16.719（25.72%） |
| 3 | 397 | 12.406（38.17%） | 302 | 9.438（29.04%） | 21.844（33.61%） |
| 4 | 327 | 10.219（31.44%） | 269 | 8.406（25.87%） | 18.625（28.65%） |
| 5 | 276 | 8.625（26.54%） | 219 | 6.844（21.06%） | 15.469（23.80%） |
| 6 | 279 | 8.719（26.83%） | 481 | 15.031（46.25%） | 23.750（36.54%） |
| 7 | 308 | 9.625（29.62%） | 277 | 8.656（26.63%） | 18.281（28.12%） |
| 8 | 284 | 8.875（27.31%） | 259 | 8.094（24.90%） | 16.969（26.11%） |
| 9 | 855 | 26.719（82.21%） | 343 | 10.719（32.98%） | 37.438（57.60%） |
| 10 | 158 | 4.938（15.19%） | 301 | 9.406（28.94%） | 14.344（22.07%） |
| 11 | 439 | 13.719（42.21%） | 209 | 6.531（20.10%） | 20.250（31.15%） |
| 12 | 255 | 7.969（24.52%） | 309 | 9.656（29.71%） | 17.625（27.12%） |
| 13 | 572 | 17.875（55.00%） | 328 | 10.250（31.54%） | 28.125（43.27%） |
| 14 | 289 | 9.031（27.79%） | 264 | 8.250（25.38%） | 17.281（26.59%） |
| 15 | 334 | 10.438（32.12%） | 168 | 5.250（16.15%） | 15.688（24.13%） |
| 16 | 979 | 30.594（94.13%） | 289 | 9.031（27.79%） | 39.625（60.96%） |
| 17 | 342 | 10.688（32.88%） | 288 | 9.000（27.69%） | 19.688（30.29%） |
| 18 | 688 | 21.500（66.15%） | 852 | 26.625（81.92%） | 48.125（74.04%） |
| 19 | 517 | 16.156（49.71%） | 271 | 8.469（26.06%） | 24.625（37.88%） |
| 20 | 301 | 9.406（28.94%） | 340 | 10.625（32.69%） | 20.031（30.82%） |
| 21 | 272 | 8.500（26.15%） | 296 | 9.250（28.46%） | 17.750（27.31%） |
| 22 | 978 | 30.562（94.04%） | 290 | 9.062（27.88%） | 39.625（60.96%） |
| 23 | 952 | 29.750（91.54%） | 746 | 23.312（71.73%） | 53.062（81.63%） |
| 24 | 304 | 9.500（29.23%） | 625 | 19.531（60.10%） | 29.031（44.66%） |
| 25 | 611 | 19.094（58.75%） | 366 | 11.438（35.19%） | 30.531（46.97%） |
| 26 | 716 | 22.375（68.85%） | 655 | 20.469（62.98%） | 42.844（65.91%） |
| 27 | 305 | 9.531（29.33%） | 315 | 9.844（30.29%） | 19.375（29.81%） |
| 28 | 266 | 8.312（25.58%） | 467 | 14.594（44.90%） | 22.906（35.24%） |
| 29 | 227 | 7.094（21.83%） | 709 | 22.156（68.17%） | 29.250（45.00%） |
| 30 | 445 | 13.906（42.79%） | 320 | 10.000（30.77%） | 23.906（36.78%） |
| 31 | 273 | 8.531（26.25%） | 320 | 10.000（30.77%） | 18.531（28.51%） |

观察与结论：

- 单个 query/head 的固定 64 块在 1K queries 合并后扩大为 **147–979 块**，对应
  **14.13%–94.13%** 的完整 head cache。单 query 的 6.15% 预算不能代表整批的访问量。
- 层内 head 差异显著，例如第 16 层：head 0 为 979 块（94.13%），head 1 为 289 块
  （27.79%）。两个 head 必须分别去重、分别计费；仅报告平均 head 会隐藏这类差异。
- 层合计从第 0 层的 11.125 MiB（17.12%）到第 23 层的 53.0625 MiB（81.63%）。
  图中部分 head 的覆盖接近全 cache，其他 head 保持较集中的选择，访存收益依赖层和 head。
- 全部 candidate 块都会进入并集，符合 local 窗口语义。该单请求显示 query 合并后仍有
  KV 容量缩减空间；不能据此推导实际带宽收益、延迟加速或其他 GR 请求的分布。

数组审计确认所有 65536 个 query/head/layer 选择均含 64 个唯一 causal blocks，
并满足 sink/local 约束；union 范围符合 79–1040 的结构下界/上界。CPU 全局回归
251 passed（43 项按 CPU 环境或缺少可选 fixture 跳过，另有 34 subtests），GPU 回归
41 passed；参考 indexer 数学 oracle 与旁路前后 dense hidden/cache 精确一致检查均通过。
这些回归用于验证代码，不作为额外实验测量。

## 固定 MFU 的计算时间与 50 GB/s fetch 估算

本节先给出上述 64-block pattern run 的估算；32-block 结果见下一节。
计算使用该 pattern run，以及原 dense run
`flashinfer_merged_gemm_65536_1024_20260925_02` 的 `mfu.json`、`analysis.json` 与 `nsys.sqlite`。
两份 run 的请求和模型配置必须一致。此节处理 prefix 已就绪后的 **1024-token extend**。

### 计算口径

保持各层原 dense attention 的 MFU，其他层内模块的 GPU 时间保持原值。算力分母沿用
H200 BF16 dense 标称 **989 TFLOPS**；attention 全32层聚合 MFU 为 **62.8489702%**。
线性投影 FLOPs 不变；norm、RoPE、SwiGLU、KV 写入等也保留该层原测量时间。

原 MFU 使用 causal **有效矩阵 FLOPs**，因此本次也按 token causal mask 计数：

```text
P = 65536, T = 1024, Hq = 32, Hkv = 2, d = 128
F_dense_attention/layer = 4 * Hq * d * (T*P + T*(T+1)/2)
                        = 1,108,109,950,976 FLOPs

每个 query/head 的 sparse 有效 KV 数 = 63*64 + (p % 64 + 1)
1K queries 每个 head 的有效 query-key 对数 = 4,162,048
F_sparse_attention/layer = 4 * Hq * d * 4,162,048
                         = 68,190,994,432 FLOPs
F_sparse / F_dense       = 6.153811214%

t_sparse_attention[layer] = t_dense_attention[layer] * F_sparse / F_dense
t_sparse_layer[layer]     = t_other_modules[layer] + t_sparse_attention[layer]
```

代码从逐 query / KV head 的实际选择数组计算有效对数，再乘对应 GQA 的 Q head 数。
**计算量按每个 query 的选择计；37.90% 的跨 query 并集比例只用于 fetch，不能用来缩放 FLOPs。**
当前块的未可见 token 不计有效 FLOPs，fetch 则仍按完整块计费。

逐层时间通过原 trace 的 CUDA launch correlation ID 找到包围 launch 的 decoder NVTX 区间，
再将各模块 GPU 活动归到该层；不能用异步 GPU kernel 的开始时间定位 CPU NVTX 层区间。
层计算时间使用与原 MFU 一致的 **模块活动 duration 之和**，包括 gate/up 的 MEMSET。
原 detailed extend 活动时长总和为 80.549862 ms，因部分活动重叠，区别于 GPU active 并集
80.534502 ms。Embedding/final norm 不摊进 decoder 层。

这些计算时间不含 indexer、host gap、fetch 或调度开销；不假设 fetch/compute overlap。
它们是固定 MFU 条件下的 GPU 工作量估算，不是端到端墙钟预测。

### Fetch 口径

按 **50 GB/s = 50,000,000,000 B/s**，两个 KV heads 共用这一个带宽：

```text
t_sparse_fetch[layer] (ms) = sum_head(union_blocks[layer,head]) * 32768 / 50,000,000
t_full_fetch[layer]   (ms) = 66560 * 2 KV heads * 128 * 2(K,V) * 2(BF16) / 50,000,000
                          = 1.363148800 ms
```

主口径包括 candidate KV，与完整 64K+1K cache 分母一致。如果后续只 fetch 历史 prefix、
candidate 已在 HBM，每层两种方案均应减去 1 MiB 对应的 **0.020971520 ms**；完整 JSON/CSV
另外保存 prefix-only 的数值。带宽视为有效 payload 带宽，不再额外扣协议效率，不计传输启动时间。

### 运行与产物

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/estimate.sh --help
bash experiments/nosa_indexer_pattern_65536_1024/scripts/estimate.sh estimate_mfu_pcie50_20260925_01
```

脚本仅运行 `python -m experiments.nosa_indexer_pattern_65536_1024.src.estimate`，不加载模型、
不执行 GPU 前向；默认读取上面的两个源 run，支持 `--dense-data-dir`、`--pattern-data-dir`、
`--bandwidth-gbps`。调用 [estimate.py](src/estimate.py)、[dense_timing.py](src/dense_timing.py)、
已有 [analyze.py](src/analyze.py) 以及原 dense 实验的 `src.mfu.matrix_flops`。

每次计算使用独立 run ID，成功后保存至 `output/data/<run_id>/`：`estimates.json`、
`layer_estimates.csv`、`head_fetch_estimates.csv`、`dense_layer_timing.json`、`estimate.md`、
`time_estimates.png` / `.svg` 及 `source/`。JSON 保存假设、源 run、输入与源码 SHA256；
stdout/stderr 在 `output/log/<run_id>/`，没有新的 profiler 产物。

### 估算结果

已完成离线计算 run `estimate_mfu_pcie50_20260925_01`。这是基于既有测量的估算，
没有重新测量 GPU 或 PCIe。每层保持对应的 dense attention MFU，故 attention 时间略有层间差异。

| 项目 | 每层平均 ms | 32 层合计 ms |
| --- | ---: | ---: |
| Sparse attention | 0.109707 | 3.510610 |
| 其余层内 GPU 模块 | 0.734066 | 23.490099 |
| Sparse decoder 层 GPU 工作 | 0.843772 | 27.000709 |
| Fetch sparse KV 并集 | 0.516649 | 16.532767 |
| Fetch full KV | 1.363149 | 43.620762 |

Embedding 与 final norm 合计另 **0.012032 ms**，计入后全模型 GPU 模块时间估算为
**27.012741 ms**。此处计算与 fetch 分别列出，没有将二者相加或假设重叠。

每层投影仍为 485,331,304,448 FLOPs，attention 降至 68,190,994,432 FLOPs；
全32层有效矩阵工作量为 17,712,713,564,160 FLOPs。其余非矩阵操作保留原时间。

以下所有时间单位均为 **ms**，层号从0开始：

| 层 | Dense attention MFU | Sparse attention | 其他模块 | Sparse 层 GPU | Fetch sparse KV | Fetch full KV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 61.66% | 0.111818 | 0.745153 | 0.856971 | 0.233308 | 1.363149 |
| 1 | 62.22% | 0.110808 | 0.741120 | 0.851928 | 0.333578 | 1.363149 |
| 2 | 62.41% | 0.110485 | 0.740320 | 0.850805 | 0.350618 | 1.363149 |
| 3 | 62.35% | 0.110590 | 0.741409 | 0.851999 | 0.458097 | 1.363149 |
| 4 | 62.37% | 0.110548 | 0.740929 | 0.851477 | 0.390595 | 1.363149 |
| 5 | 62.22% | 0.110818 | 0.744288 | 0.855106 | 0.324403 | 1.363149 |
| 6 | 62.26% | 0.110753 | 0.742080 | 0.852833 | 0.498074 | 1.363149 |
| 7 | 62.81% | 0.109778 | 0.734625 | 0.844403 | 0.383386 | 1.363149 |
| 8 | 62.99% | 0.109453 | 0.734145 | 0.843598 | 0.355860 | 1.363149 |
| 9 | 63.04% | 0.109378 | 0.732416 | 0.841794 | 0.785121 | 1.363149 |
| 10 | 63.06% | 0.109343 | 0.731904 | 0.841247 | 0.300810 | 1.363149 |
| 11 | 63.10% | 0.109264 | 0.730657 | 0.839921 | 0.424673 | 1.363149 |
| 12 | 63.02% | 0.109404 | 0.731329 | 0.840733 | 0.369623 | 1.363149 |
| 13 | 63.11% | 0.109256 | 0.731041 | 0.840297 | 0.589824 | 1.363149 |
| 14 | 63.07% | 0.109321 | 0.731008 | 0.840329 | 0.362414 | 1.363149 |
| 15 | 63.04% | 0.109369 | 0.730177 | 0.839546 | 0.328991 | 1.363149 |
| 16 | 63.07% | 0.109323 | 0.732513 | 0.841836 | 0.830996 | 1.363149 |
| 17 | 63.00% | 0.109447 | 0.731905 | 0.841352 | 0.412877 | 1.363149 |
| 18 | 63.02% | 0.109402 | 0.731456 | 0.840858 | 1.009254 | 1.363149 |
| 19 | 62.96% | 0.109506 | 0.732512 | 0.842018 | 0.516424 | 1.363149 |
| 20 | 63.01% | 0.109424 | 0.732162 | 0.841586 | 0.420086 | 1.363149 |
| 21 | 63.03% | 0.109388 | 0.730881 | 0.840269 | 0.372244 | 1.363149 |
| 22 | 63.05% | 0.109361 | 0.730560 | 0.839921 | 0.830996 | 1.363149 |
| 23 | 63.07% | 0.109329 | 0.730752 | 0.840081 | 1.112801 | 1.363149 |
| 24 | 63.03% | 0.109398 | 0.732738 | 0.842136 | 0.608829 | 1.363149 |
| 25 | 63.01% | 0.109430 | 0.731041 | 0.840471 | 0.640287 | 1.363149 |
| 26 | 63.02% | 0.109410 | 0.730528 | 0.839938 | 0.898499 | 1.363149 |
| 27 | 63.05% | 0.109361 | 0.731808 | 0.841169 | 0.406323 | 1.363149 |
| 28 | 63.06% | 0.109345 | 0.732929 | 0.842274 | 0.480379 | 1.363149 |
| 29 | 63.04% | 0.109374 | 0.731745 | 0.841119 | 0.613417 | 1.363149 |
| 30 | 63.07% | 0.109317 | 0.732448 | 0.841765 | 0.501350 | 1.363149 |
| 31 | 63.02% | 0.109406 | 0.731520 | 0.840926 | 0.388628 | 1.363149 |

Fetch sparse KV 从层0的 **0.233308 ms** 到层23的 **1.112801 ms**；full KV 固定为
**1.363149 ms/层**。Sparse fetch 合计比 full 少 **27.087995 ms**，对应相同的 62.10% payload 缩减。
两个 KV heads 共用链路，各 head 的 fetch 时间在 `head_fetch_estimates.csv` 中分别记录，
相加即为层 fetch 时间，不能将每个 head 都按独享50 GB/s并行计费。

计算正确性：新增30项测试覆盖 causal partial block、GQA FLOPs、union与FLOPs分离、
逐层固定MFU、十进制带宽换算，以及异步trace的launch归属、MEMSET与shared操作边界。
全局CPU回归 **281 passed、43 skipped、34 subtests passed**；本次未修改GPU数值路径，也未新增GPU测量。

## 32-block 新结果：1 sink + 16 local + 15 query-aware

已完成新采集 run `query_aware32_fp32_65536_1024_20260926_01`（2026-09-26 00:12 北京时间，
即 2026-09-25 16:12 UTC），以及离线估算 run `estimate32_mfu_pcie50_20260926_01`。
沿用同一份完整请求、NOSA-8B 权重、dense 激活口径、BF16/FP32、LongRoPE 与 66560 上下文。
实际硬件仍为 `cuda:0` 的 H200（nvidia-smi 名称 M403），SM90、132 SM；依赖版本与上述64-block运行相同。
无计时预热，每请求只采集一次；固定 MFU 与 50 GB/s 的时间仍是估算。

复现命令：

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh query_aware32_fp32_65536_1024_20260926_01 \
  --block-budget 32
bash experiments/nosa_indexer_pattern_65536_1024/scripts/estimate.sh estimate32_mfu_pcie50_20260926_01 \
  --pattern-data-dir experiments/nosa_indexer_pattern_65536_1024/output/data/query_aware32_fp32_65536_1024_20260926_01
```

采集产物在 `output/data/query_aware32_fp32_65536_1024_20260926_01/`，
包含 `[32,1024,2,32]` 逐 query 选择、union、64 行 head 统计、32 行层统计、
容量图和 block union 热图；估算表和时间图在
`output/data/estimate32_mfu_pcie50_20260926_01/`。两次运行均记录独立的输入与源码哈希，保留原64-block产物。

### 与 64-block 对比

下表容量和时间均为32层合计；decoder GPU时间不含另列的 embedding/final norm。

| 指标 | 64 blocks：top47 | 32 blocks：top15 |
| --- | ---: | ---: |
| 去重 KV 并集 | 788.34375 MiB | 526.65625 MiB |
| 占完整 2080 MiB cache | 37.90% | 25.32% |
| 每 head 并集范围 | 147–979 blocks | 114–788 blocks |
| Sparse attention，固定 MFU | 3.510610 ms | 1.741701 ms |
| Decoder GPU 工作 | 27.000709 ms | 25.231800 ms |
| Fetch sparse KV，50 GB/s | 16.532767 ms | 11.044782 ms |
| Fetch full KV，50 GB/s | 43.620762 ms | 43.620762 ms |

32-block 的并集为 **552239104 B = 526.65625 MiB（25.32%）**，
其中 prefix **494.65625 MiB / 2048 MiB（24.15%）**，candidate **32 MiB / 32 MiB（100%）**。
相较64-block，并集容量与 fetch 时间减少 **33.19%**；
attention 计算量和固定MFU时间减少 **50.39%**。
其余模块时间保持不变，decoder 层总计算时间减少 **6.55%**。
选块预算减半后，并集仍保留64-block容量的66.81%；跨 query 的选择差异使访存缩减不与单query预算成正比。

### 每层、每 KV head 容量

head 比例分母为32.5 MiB，层比例分母为65 MiB。所有head均包含16个candidate blocks，
因此每head的prefix容量为表中容量减0.5 MiB，每层prefix容量为层合计减1 MiB。

| 层 | Head 0 块数 | Head 0 MiB（%） | Head 1 块数 | Head 1 MiB（%） | 层合计 MiB（%） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 120 | 3.750（11.54%） | 114 | 3.562（10.96%） | 7.312（11.25%） |
| 1 | 205 | 6.406（19.71%） | 167 | 5.219（16.06%） | 11.625（17.88%） |
| 2 | 193 | 6.031（18.56%） | 152 | 4.750（14.62%） | 10.781（16.59%） |
| 3 | 274 | 8.562（26.35%） | 189 | 5.906（18.17%） | 14.469（22.26%） |
| 4 | 177 | 5.531（17.02%） | 168 | 5.250（16.15%） | 10.781（16.59%） |
| 5 | 147 | 4.594（14.13%） | 158 | 4.938（15.19%） | 9.531（14.66%） |
| 6 | 174 | 5.438（16.73%） | 276 | 8.625（26.54%） | 14.062（21.63%） |
| 7 | 154 | 4.812（14.81%） | 206 | 6.438（19.81%） | 11.250（17.31%） |
| 8 | 150 | 4.688（14.42%） | 169 | 5.281（16.25%） | 9.969（15.34%） |
| 9 | 526 | 16.438（50.58%） | 218 | 6.812（20.96%） | 23.250（35.77%） |
| 10 | 140 | 4.375（13.46%） | 166 | 5.188（15.96%） | 9.562（14.71%） |
| 11 | 296 | 9.250（28.46%） | 133 | 4.156（12.79%） | 13.406（20.62%） |
| 12 | 161 | 5.031（15.48%） | 151 | 4.719（14.52%） | 9.750（15.00%） |
| 13 | 380 | 11.875（36.54%） | 219 | 6.844（21.06%） | 18.719（28.80%） |
| 14 | 167 | 5.219（16.06%） | 172 | 5.375（16.54%） | 10.594（16.30%） |
| 15 | 228 | 7.125（21.92%） | 144 | 4.500（13.85%） | 11.625（17.88%） |
| 16 | 762 | 23.812（73.27%） | 178 | 5.562（17.12%） | 29.375（45.19%） |
| 17 | 172 | 5.375（16.54%） | 167 | 5.219（16.06%） | 10.594（16.30%） |
| 18 | 470 | 14.688（45.19%） | 626 | 19.562（60.19%） | 34.250（52.69%） |
| 19 | 352 | 11.000（33.85%） | 137 | 4.281（13.17%） | 15.281（23.51%） |
| 20 | 186 | 5.812（17.88%） | 229 | 7.156（22.02%） | 12.969（19.95%） |
| 21 | 191 | 5.969（18.37%） | 198 | 6.188（19.04%） | 12.156（18.70%） |
| 22 | 782 | 24.438（75.19%） | 179 | 5.594（17.21%） | 30.031（46.20%） |
| 23 | 788 | 24.625（75.77%） | 567 | 17.719（54.52%） | 42.344（65.14%） |
| 24 | 208 | 6.500（20.00%） | 418 | 13.062（40.19%） | 19.562（30.10%） |
| 25 | 417 | 13.031（40.10%） | 216 | 6.750（20.77%） | 19.781（30.43%） |
| 26 | 551 | 17.219（52.98%） | 450 | 14.062（43.27%） | 31.281（48.12%） |
| 27 | 162 | 5.062（15.58%） | 167 | 5.219（16.06%） | 10.281（15.82%） |
| 28 | 185 | 5.781（17.79%） | 310 | 9.688（29.81%） | 15.469（23.80%） |
| 29 | 150 | 4.688（14.42%） | 464 | 14.500（44.62%） | 19.188（29.52%） |
| 30 | 314 | 9.812（30.19%） | 199 | 6.219（19.13%） | 16.031（24.66%） |
| 31 | 171 | 5.344（16.44%） | 193 | 6.031（18.56%） | 11.375（17.50%） |

单个query/head选32块，占完整head cache的3.08%；1K queries合并后为
**114–788块（10.96%–75.77%）**。层总容量从层0的 **7.3125 MiB（11.25%）**
到层23的 **42.34375 MiB（65.14%）**，层/head差异仍明显。

### 固定 MFU 的计算与 fetch 时间

沿用上一节逐层 dense MFU、其他模块时间以及50 GB/s有效payload带宽，仅代入新的选择：

```text
每个 query/head 的有效 KV 数 = 31*64 + (p % 64 + 1)
1K queries 每个 head 的有效 query-key 对数 = 2,064,896
F_sparse_attention/layer = 4 * 32 * 128 * 2,064,896
                         = 33,831,256,064 FLOPs
F_sparse / F_dense       = 3.053059494%
```

每层投影仍为485,331,304,448 FLOPs；全32层有效矩阵计算量为16,613,201,936,384 FLOPs。
Sparse attention 每层平均 **0.054428 ms**，
decoder 每层平均 **0.788494 ms**。
Embedding/final norm另计0.012032 ms后，全模型GPU模块时间估算为 **25.243832 ms**。

以下时间单位均为ms。MFU及其他模块的逐层数值与64-block表相同；计算与fetch分别列出。

| 层 | Sparse attention | Sparse 层 GPU | Fetch sparse KV | Fetch full KV |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.055476 | 0.800629 | 0.153354 | 1.363149 |
| 1 | 0.054975 | 0.796095 | 0.243794 | 1.363149 |
| 2 | 0.054814 | 0.795134 | 0.226099 | 1.363149 |
| 3 | 0.054866 | 0.796275 | 0.303432 | 1.363149 |
| 4 | 0.054846 | 0.795775 | 0.226099 | 1.363149 |
| 5 | 0.054980 | 0.799268 | 0.199885 | 1.363149 |
| 6 | 0.054947 | 0.797027 | 0.294912 | 1.363149 |
| 7 | 0.054464 | 0.789089 | 0.235930 | 1.363149 |
| 8 | 0.054302 | 0.788447 | 0.209060 | 1.363149 |
| 9 | 0.054265 | 0.786681 | 0.487588 | 1.363149 |
| 10 | 0.054248 | 0.786152 | 0.200540 | 1.363149 |
| 11 | 0.054209 | 0.784866 | 0.281149 | 1.363149 |
| 12 | 0.054278 | 0.785607 | 0.204472 | 1.363149 |
| 13 | 0.054205 | 0.785246 | 0.392561 | 1.363149 |
| 14 | 0.054237 | 0.785245 | 0.222167 | 1.363149 |
| 15 | 0.054260 | 0.784437 | 0.243794 | 1.363149 |
| 16 | 0.054238 | 0.786751 | 0.616038 | 1.363149 |
| 17 | 0.054300 | 0.786205 | 0.222167 | 1.363149 |
| 18 | 0.054277 | 0.785733 | 0.718275 | 1.363149 |
| 19 | 0.054329 | 0.786841 | 0.320471 | 1.363149 |
| 20 | 0.054288 | 0.786450 | 0.271974 | 1.363149 |
| 21 | 0.054270 | 0.785151 | 0.254935 | 1.363149 |
| 22 | 0.054257 | 0.784817 | 0.629801 | 1.363149 |
| 23 | 0.054241 | 0.784993 | 0.888013 | 1.363149 |
| 24 | 0.054275 | 0.787013 | 0.410255 | 1.363149 |
| 25 | 0.054291 | 0.785332 | 0.414843 | 1.363149 |
| 26 | 0.054281 | 0.784809 | 0.656015 | 1.363149 |
| 27 | 0.054257 | 0.786065 | 0.215613 | 1.363149 |
| 28 | 0.054249 | 0.787178 | 0.324403 | 1.363149 |
| 29 | 0.054263 | 0.786008 | 0.402391 | 1.363149 |
| 30 | 0.054235 | 0.786683 | 0.336200 | 1.363149 |
| 31 | 0.054279 | 0.785799 | 0.238551 | 1.363149 |

Sparse fetch 每层平均 **0.345149 ms**，范围 **0.153354–0.888013 ms**；
full fetch仍为 **1.363149 ms/层**。相较full，32层共少fetch **32.575980 ms（74.68%）**。
若candidate已在HBM、只fetch prefix，每层两列均减0.020971520 ms，
模型合计为 **10.373693 ms sparse / 42.949673 ms full**。每head明细在 `head_fetch_estimates.csv`。

时间估算不含indexer、host gap、传输启动开销或fetch/compute overlap；尚无32-block sparse数值前向，
因此本实验不评价精度或端到端加速。

完整数组审计确认65536条query/head/layer记录均满足1 sink + 16 local + 15其他causal块；
每条32-block选择和每个head的union均为原64-block结果的子集，union符合47–1040的结构边界。
新旧run的请求逐字节、输入token、模型配置和checkpoint哈希一致；新run的37份源码快照
与记录哈希及采集时源码一致。32/64预算的独立数学参考、因果窗口、同分排序和
旁路dense hidden/cache一致性检查通过。
全局CPU回归 **314 passed、45 skipped、34 subtests passed**；全局GPU回归 **43 passed**。
CPU跳过项为GPU检查及不可用的可选本地fixture；GPU回归没有跳过。

## 将 sink/local 与 query-aware 分开统计

对每个 query 先分开固定选择 `D_p = sink_p ∪ local_p` 与 query-aware 选择
`A_p = S_p \ D_p`，再分别沿 1024 个 queries 求并集：

```text
D = union_p(D_p)
A = union_p(A_p)
O = A ∩ D                   # 两类并集的重叠部分
E = A \ D                   # 固定部分已覆盖后，query-aware 额外需要的块
U = D ∪ A = D ∪ E
bytes(U) = bytes(D) + bytes(E)
```

虽然每个 query 的 `D_p` 与 `A_p` 互斥，**跨 query 合并后的 D 与 A 可以重叠**。
某个块可能在较早 query 属于 local，在较晚 query 离开 local 窗口后又被 query-aware 选中。
因此分别报告 QA 自身并集 `A` 和额外并集 `E`；直接相加 `bytes(D) + bytes(A)` 会重复计费 `O`。
这只拆分原选块集合，不改变选择策略、attention FLOPs 或前述计算时间。

### 固定部分：两种预算完全相同

对 `p=65536..66559`，当前块号从1024移到1039，因此：

```text
sink union  = {0}                 # 1 block/head
local union = {1009, ..., 1039}   # 31 blocks/head
D           = {0, 1009, ..., 1039}  # 32 blocks/head
```

每个 query 的16个local在整个1K批次内变为31个，固定部分并集共32块/head。
下表 fetch 仍按50 GB/s十进制有效带宽、两个KV heads共享链路，完整块计费：

| 固定部分 | 每 head 块数 | 每 head MiB | 每层 MiB | 每层 fetch ms | 全32层 MiB | 全32层 fetch ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sink | 1 | 0.03125 | 0.0625 | 0.00131072 | 2 | 0.04194304 |
| Local union | 31 | 0.96875 | 1.9375 | 0.04063232 | 62 | 1.30023424 |
| 固定并集 D | 32 | 1 | 2 | 0.04194304 | 64 | 1.34217728 |

固定并集占各head/层/模型完整cache的 **3.076923%**。其中prefix为sink加15个历史local块，
每层 **1 MiB / 0.02097152 ms**；另外16个candidate块也是每层1 MiB。
如果candidate已经在HBM，只fetch历史prefix，则固定部分按前一个数值计。

### 分项结果与 fetch 时间

复用原32/64-block数组完成离线分解，分别保存为 `components32_pcie50_20260926_01` 和
`components64_pcie50_20260926_01`。没有重新运行模型；请求、硬件、精度和dense激活来源沿用各自采集记录。
下面容量与时间均为32层合计，两个head分别去重后相加；时间单位ms、带宽50 GB/s。

| 集合 | 32-block MiB | 32-block fetch ms | 64-block MiB | 64-block fetch ms |
| --- | ---: | ---: | ---: | ---: |
| 固定并集 D | 64.00000 | 1.34217728 | 64.00000 | 1.34217728 |
| QA 自身并集 A | 492.21875 | 10.32257536 | 754.34375 | 15.81973504 |
| 重叠 O（不能重复计费） | 29.56250 | 0.61997056 | 30.00000 | 0.62914560 |
| QA 额外并集 E = A\D | 462.65625 | 9.70260480 | 724.34375 | 15.19058944 |
| 合计 U = D∪E | 526.65625 | 11.04478208 | 788.34375 | 16.53276672 |

32-block 的额外QA并集占完整cache的 **22.243089%**，加上固定部分 **3.076923%**，
恢复原来的 **25.320012%**；64-block 对应 **34.824219% + 3.076923% = 37.901142%**。
如固定部分已驻留HBM，QA新增fetch量应采用 **E**；若固定部分也需要传输，则按 **D + E** 计费。
两个分项是同一链路上的payload预算，此处没有据此假设传输与计算重叠。

两组QA自身并集都只含prefix块：1024–1039这些candidate块在整个候选阶段始终属于local。
QA与固定部分的重叠只能来自历史local块1009–1023。64-block每head都重叠15块；
32-block为9–15块，其中只有以下5个head少于15块，其余59个head均为15块：

| 层 | KV head | 32-block 重叠块数 |
| ---: | ---: | ---: |
| 1 | 1 | 12 |
| 4 | 0 | 14 |
| 5 | 1 | 9 |
| 9 | 0 | 12 |
| 27 | 0 | 14 |

每head的额外QA容量可由前述head总容量减 **1 MiB** 得到；但QA自身并集还需要加回该head的重叠量。
因此不能把全部head都按固定15个重叠块修正32-block的QA自身并集。

以下按层列出两组QA自身并集、额外并集及额外fetch时间。每一层的固定部分均为
**2 MiB / 0.04194304 ms**，合计fetch等于表中额外QA时间加0.04194304 ms。

| 层 | 32 QA MiB | 32 重叠 MiB | 32 额外QA MiB | 32 额外fetch ms | 64 QA MiB | 64 额外QA MiB | 64 额外fetch ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6.25000 | 0.93750 | 5.31250 | 0.11141120 | 10.06250 | 9.12500 | 0.19136512 |
| 1 | 10.46875 | 0.84375 | 9.62500 | 0.20185088 | 14.84375 | 13.90625 | 0.29163520 |
| 2 | 9.71875 | 0.93750 | 8.78125 | 0.18415616 | 15.65625 | 14.71875 | 0.30867456 |
| 3 | 13.40625 | 0.93750 | 12.46875 | 0.26148864 | 20.78125 | 19.84375 | 0.41615360 |
| 4 | 9.68750 | 0.90625 | 8.78125 | 0.18415616 | 17.56250 | 16.62500 | 0.34865152 |
| 5 | 8.28125 | 0.75000 | 7.53125 | 0.15794176 | 14.40625 | 13.46875 | 0.28246016 |
| 6 | 13.00000 | 0.93750 | 12.06250 | 0.25296896 | 22.68750 | 21.75000 | 0.45613056 |
| 7 | 10.18750 | 0.93750 | 9.25000 | 0.19398656 | 17.21875 | 16.28125 | 0.34144256 |
| 8 | 8.90625 | 0.93750 | 7.96875 | 0.16711680 | 15.90625 | 14.96875 | 0.31391744 |
| 9 | 22.09375 | 0.84375 | 21.25000 | 0.44564480 | 36.37500 | 35.43750 | 0.74317824 |
| 10 | 8.50000 | 0.93750 | 7.56250 | 0.15859712 | 13.28125 | 12.34375 | 0.25886720 |
| 11 | 12.34375 | 0.93750 | 11.40625 | 0.23920640 | 19.18750 | 18.25000 | 0.38273024 |
| 12 | 8.68750 | 0.93750 | 7.75000 | 0.16252928 | 16.56250 | 15.62500 | 0.32768000 |
| 13 | 17.65625 | 0.93750 | 16.71875 | 0.35061760 | 27.06250 | 26.12500 | 0.54788096 |
| 14 | 9.53125 | 0.93750 | 8.59375 | 0.18022400 | 16.21875 | 15.28125 | 0.32047104 |
| 15 | 10.56250 | 0.93750 | 9.62500 | 0.20185088 | 14.62500 | 13.68750 | 0.28704768 |
| 16 | 28.31250 | 0.93750 | 27.37500 | 0.57409536 | 38.56250 | 37.62500 | 0.78905344 |
| 17 | 9.53125 | 0.93750 | 8.59375 | 0.18022400 | 18.62500 | 17.68750 | 0.37093376 |
| 18 | 33.18750 | 0.93750 | 32.25000 | 0.67633152 | 47.06250 | 46.12500 | 0.96731136 |
| 19 | 14.21875 | 0.93750 | 13.28125 | 0.27852800 | 23.56250 | 22.62500 | 0.47448064 |
| 20 | 11.90625 | 0.93750 | 10.96875 | 0.23003136 | 18.96875 | 18.03125 | 0.37814272 |
| 21 | 11.09375 | 0.93750 | 10.15625 | 0.21299200 | 16.68750 | 15.75000 | 0.33030144 |
| 22 | 28.96875 | 0.93750 | 28.03125 | 0.58785792 | 38.56250 | 37.62500 | 0.78905344 |
| 23 | 41.28125 | 0.93750 | 40.34375 | 0.84606976 | 52.00000 | 51.06250 | 1.07085824 |
| 24 | 18.50000 | 0.93750 | 17.56250 | 0.36831232 | 27.96875 | 27.03125 | 0.56688640 |
| 25 | 18.71875 | 0.93750 | 17.78125 | 0.37289984 | 29.46875 | 28.53125 | 0.59834368 |
| 26 | 30.21875 | 0.93750 | 29.28125 | 0.61407232 | 41.78125 | 40.84375 | 0.85655552 |
| 27 | 9.18750 | 0.90625 | 8.28125 | 0.17367040 | 18.31250 | 17.37500 | 0.36438016 |
| 28 | 14.40625 | 0.93750 | 13.46875 | 0.28246016 | 21.84375 | 20.90625 | 0.43843584 |
| 29 | 18.12500 | 0.93750 | 17.18750 | 0.36044800 | 28.18750 | 27.25000 | 0.57147392 |
| 30 | 14.96875 | 0.93750 | 14.03125 | 0.29425664 | 22.84375 | 21.90625 | 0.45940736 |
| 31 | 10.31250 | 0.93750 | 9.37500 | 0.19660800 | 17.46875 | 16.53125 | 0.34668544 |

64-block的重叠固定为每层0.9375 MiB，已从其额外QA列扣除。
32/64-block每层平均额外QA fetch分别为 **0.30320640 / 0.47470592 ms**。
加固定部分后恢复原平均总fetch **0.34514944 / 0.51664896 ms**。
仅fetch prefix时，额外QA时间保持不变，固定部分改为每层0.02097152 ms。

### 复现与产物

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/decompose.sh --help
bash experiments/nosa_indexer_pattern_65536_1024/scripts/decompose.sh components32_pcie50_20260926_01 \
  --pattern-data-dir experiments/nosa_indexer_pattern_65536_1024/output/data/query_aware32_fp32_65536_1024_20260926_01
bash experiments/nosa_indexer_pattern_65536_1024/scripts/decompose.sh components64_pcie50_20260926_01 \
  --pattern-data-dir experiments/nosa_indexer_pattern_65536_1024/output/data/query_aware_fp32_65536_1024_20260925_01
```

脚本默认读取32-block run，支持 `--bandwidth-gbps`；复现时使用新的run ID。
调用 [decompose.py](src/decompose.py) 与纯NumPy分解函数
[selection_parts.py](src/selection_parts.py)，复用 [analyze.py](src/analyze.py) 的输入校验和容量口径。
成功后写入对应 `output/data/components{32|64}_pcie50_20260926_01/`：

- `component_union_masks.npz`：sink、local、fixed、query_aware、overlap、query_aware_additional、combined共7类union。
- `decomposition.json` / `.md`：定义、参数、逐head/层/模型分项、fetch及prefix/candidate数据。
- `head_components.csv`、`layer_components.csv`、`model_components.csv`：按component展开的448/224/7行统计。
- `component_capacity_fetch.png` / `.svg`：各层fixed与额外QA的容量和fetch堆叠图。
- `source/`和JSON中的SHA256：分析源码快照、原metadata/请求/选择数组/union输入哈希。

日志位于对应 `output/log/<run_id>/`；失败仅保留系统临时目录，旧采集与估算产物均保留。
独立重算确认所有64个head的fixed均为32块，`D∪A`与旧union数组逐项相同，
且 `bytes(D)+bytes(E)=bytes(U)`；源码快照与记录哈希一致。
新增22项单测覆盖跨query重叠、独立head计费、短序列sink/local重叠、padding、固定项完整性和带宽换算。
全局CPU回归 **336 passed、45 skipped、34 subtests passed**。本次仅增加离线分析，没有修改GPU数值路径或新增GPU测量。

## 全部 KV heads 的并集占比分布

已完成离线分析 run `head_coverage_distribution_32_64_20260926_01`，对比上述32/64-block原始采集。
每种配置各有 **32层 × 2 KV heads = 64个样本**，每个样本是对应head的完整选块并集占比：

```text
coverage[layer, head] = |union over 1024 queries| / 1040
                     = 该head选中KV并集容量 / 32.5 MiB
```

这里的并集包含sink、local和query-aware，按原始总集合去重。不同层、不同head分别作为样本，
每个样本权重相同；由于完整head容量均相同，样本均值也等于全模型选中KV占总KV的比例。
**占比越低表示越稀疏**；如果将稀疏度定义为未选中KV的比例，则为 `1 - coverage`。

生成的 `head_coverage_distribution.png` / `.svg` 包含两幅图：

- 左：并列直方图，横轴是head的KV并集占比，每10个百分点一个分箱，纵轴是head数量。
  区间左闭右开，最后一个区间包含100%；每种配置的所有分箱合计64个heads。
- 右：经验累计分布，纵轴表示并集占比不超过横轴值的heads比例；使用原始64个样本阶梯图。

完整图位于
`output/data/head_coverage_distribution_32_64_20260926_01/head_coverage_distribution.png`，
同目录保存SVG。以下分位数使用NumPy的 `method="linear"`，数值均为并集占比：

| 预算 | 均值 | P25 | 中位数 | P75 | P90 | P95 | 最小 | 最大 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 blocks | 25.32% | 16.03% | 18.27% | 28.80% | 48.96% | 59.34% | 10.96% | 75.77% |
| 64 blocks | 37.90% | 26.23% | 29.47% | 43.32% | 68.64% | 82.17% | 14.13% | 94.13% |

32-block有 **39/64个heads集中在10%–20%**，64-block有 **31/64个heads集中在20%–30%**。
32-block有45个head的占比不超过25%，64-block只有11个；占比超过75%的head分别为2个和5个。
两组均值均高于中位数，说明少量覆盖较广的heads提高了总体访存量。
按 `1 - coverage` 计，平均稀疏度分别为 **74.68% / 62.10%**。
这是同一GR请求内的层/head分布，各配置独立统计，样本不代表不同请求。

复现命令：

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/distribution.sh --help
bash experiments/nosa_indexer_pattern_65536_1024/scripts/distribution.sh head_coverage_distribution_32_64_20260926_01
```

复现时使用新run ID；默认读取已记录的32/64-block两组，可重复传入 `--pattern-data-dir` 指定对照组。
脚本调用 [distribution.py](src/distribution.py)，复用 [analyze.py](src/analyze.py) 重算并校验原始union，
并检查两组请求、模型配置、执行边界、精度与checkpoint一致。没有重新运行GPU或改变之前的采集产物。
成功结果发布到对应 `output/data/<run_id>/`，失败保留系统临时目录：

- `head_coverage_distribution.png` / `.svg`：直方图与经验累计分布。
- `head_coverage.csv`：两组共128行，保留layer、KV head、并集块数、容量、占比及未选中比例。
- `histogram.csv`：两组共20个分箱的边界、head数量和比例。
- `distribution.json` / `.md`：统计口径、完整精度分位数、输入与源码SHA256；`source/`保存源码快照。

独立校核确认128个head样本与原union数组逐项一致，分箱数量各为64，均值恢复原模型占比，
所有统计量与独立NumPy计算一致；CLI、Ruff及源码快照哈希检查通过。

## 30% 分界下的 fetch overlap 效率估算

按完整KV并集占比将每个层/head分类：**小于30%为sparse头，大于等于30%为dense头**。
本次两组数据均没有恰好30%的head。按确认的口径，sparse头传输其选中KV并集，
dense头预取完整KV；**两类头的attention仍执行原32/64-block稀疏计算**，不增加dense attention FLOPs。
仍使用50 GB/s共享链路、原逐层固定MFU计算时间，以及完整64K+1K cache的payload口径。

### 时间窗口与效率定义

记 `A[l]` 为第l层原sparse attention时间，`O[l]` 为该层其他GPU模块时间。
sparse fetch使用本层的attention窗口；dense prefetch使用前一层的非attention窗口，
即当前层非attention计算用于预取下一层。这样不会用本层attention之后的MLP倒过来隐藏本层fetch。
层0没有前驱窗口，其dense预取隐藏时间设为0；本次两组的层0都没有dense头。

```text
S[l] = sum(sparse头的KV并集bytes) / 50,000,000       # ms
D[l] = sum(dense头的完整KV bytes) / 50,000,000       # ms
H_sparse[l] = min(S[l], A[l])
H_dense[l]  = min(D[l], O[l-1])                     # l=0时窗口为0
H[l]       = H_sparse[l] + H_dense[l]
R[l]       = S[l] + D[l] - H[l]

fetch overlap效率 = sum_l(H[l]) / sum_l(S[l] + D[l])
```

每层先合计同类heads的传输量，再限制时间窗口；两个head不能分别独占50 GB/s或重复使用整段计算时间。
attention与非attention窗口分别供两类传输使用。混合层中，已预取的dense类头仍执行原稀疏attention，
其计算也包含在本层可用的attention预算中。模型效率按总传输时间加权，不对各层/head百分比直接平均。
本估算只将前一层窗口分配给下一层，不将未用完的窗口在任意多层间累积。

这是**理想时间窗口预算估算**：将头分类、待传输KV和缓冲空间视为预先就绪，允许fetch/compute流式重叠；
不计indexer、首块等待、传输启动、kernel资源争用或host gap。实际DMA起止和tile依赖未实现，
因此这些数值不代表已经测得的流水线效率或端到端延迟。

### 计算结果

已完成离线run `overlap30_full_prefetch_32_64_20260926_01`。下面时间均为32层合计，单位ms：

| 预算 | Sparse / dense heads | 总fetch | 可隐藏fetch | 未隐藏fetch | Overlap效率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 blocks | 49 / 15 | 16.112681 | 9.906578 | 6.206103 | **61.48%** |
| 64 blocks | 34 / 30 | 26.276004 | 18.773740 | 7.502264 | **71.45%** |

| 预算 | 类别 | Fetch | 可隐藏 | 未隐藏 | 类别内效率 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | Sparse union，attention overlap | 5.889065 | 1.578902 | 4.310163 | 26.81% |
| 32 | Dense full KV，层间prefetch | 10.223616 | 8.327676 | 1.895940 | 81.46% |
| 64 | Sparse union，attention overlap | 5.828772 | 2.745087 | 3.083685 | 47.10% |
| 64 | Dense full KV，层间prefetch | 20.447232 | 16.028653 | 4.418579 | 78.39% |

32-block包含20个全sparse层、9个混合层、3个全dense层；64-block分别为9、16、7层。
单个dense头完整KV的fetch为 **0.6815744 ms**，低于上一层约0.73–0.75 ms的非attention时间，
因此有前驱窗口的单dense头层可以完全隐藏其预取；两个dense头共需 **1.3631488 ms**，
会超出一个非attention窗口。sparse头则主要受到较短attention窗口的限制。

64-block的隐藏比例更高，但其总fetch和未隐藏fetch仍大于32-block，不能仅由效率百分比判断最终耗时。
由于dense类头改为完整KV传输，本策略总fetch分别为16.112681 / 26.276004 ms，
高于全体heads都只传输选中并集时的11.044782 / 16.532767 ms；这里的效率分母采用前者。

### 复现与明细

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/overlap.sh --help
bash experiments/nosa_indexer_pattern_65536_1024/scripts/overlap.sh overlap30_full_prefetch_32_64_20260926_01 \
  --threshold-pct 30 --bandwidth-gbps 50
```

默认读取 `estimate32_mfu_pcie50_20260926_01` 与 `estimate_mfu_pcie50_20260925_01`，
也可重复传入 `--estimate-data-dir`。复现时使用新run ID。脚本调用 [overlap.py](src/overlap.py)，
其数学函数由实验单测覆盖，并由现有全局CPU入口自动收集；本次10项针对性测试通过。

结果位于 `output/data/overlap30_full_prefetch_32_64_20260926_01/`：
`overlap.json` / `.md` 保存公式口径与汇总；`layer_overlap.csv` 保存两组共64行逐层预算、
隐藏和未隐藏时间；`head_classes.csv` 保存128个head的分类及传输bytes。
JSON记录源估算run和输入/源码SHA256，`source/`保存源码快照，日志位于对应 `output/log/`。
独立计算核对全部汇总、前驱窗口与分类计数；CLI、Ruff、源码哈希检查通过，没有新增GPU测量。

### Attention MFU 降至原来的一半

已完成离线run `overlap30_half_mfu_32_64_20260926_01`。保持30%分类、50 GB/s、
dense头完整KV预取、全部fetch bytes和非attention时间不变，将**所有层、两类头的attention MFU乘0.5**。
原聚合attention MFU约62.848970%，减半后约31.424485%；FLOPs不变，attention时间因此翻倍：

```text
A_half[l]       = A_original[l] / 0.5 = 2 * A_original[l]
H_sparse_half[l] = min(S[l], 2 * A_original[l])
H_dense_half[l]  = min(D[l], O[l-1])             # 不变
```

延长attention窗口只增加sparse fetch的可隐藏时间，不增加由非attention时间提供的dense预取预算。
隐藏量仍受本层fetch总量上限约束，不能直接将原隐藏时间全部乘2。以下时间均为32层合计，单位ms：

| 预算 | 总fetch（不变） | 原MFU效率 | MFU减半效率 | 减半后可隐藏fetch | 未隐藏fetch：原→减半 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 blocks | 16.112681 | 61.48% | **71.10%** | 11.456384 | 6.206103 → **4.656297** |
| 64 blocks | 26.276004 | 71.45% | **79.17%** | 20.802062 | 7.502264 → **5.473942** |

Sparse类隐藏时间从1.578902 / 2.745087 ms提高到 **3.128708 / 4.773409 ms**，
类别内效率为 **53.13% / 81.89%**。Dense类隐藏时间仍为 **8.327676 / 16.028653 ms**，
类别内效率仍为81.46% / 78.39%。

更高的隐藏率来自计算变慢，不能单独解释为更好的延迟。将计算与未隐藏fetch一起核算：

| 预算 | Attention：原→减半MFU | Decoder计算：原→减半MFU | 计算+未隐藏fetch：原→减半MFU |
| ---: | ---: | ---: | ---: |
| 32 blocks | 1.741701 → 3.483402 | 25.231800 → 26.973501 | 31.437903 → **31.629799** |
| 64 blocks | 3.510610 → 7.021219 | 27.000709 → 30.511318 | 34.502972 → **35.985260** |

非attention计算合计仍为23.490099 ms。上表最后一列是同一理想窗口模型的代数时间预算，
分别增加 **0.191895 / 1.482287 ms**；不包含embedding/final norm（若计入，两边均加0.012032 ms）、
indexer、首块等待与调度开销，不是实际端到端延迟测量。降低MFU最多用增加的计算时间抵消等量fetch等待，
在此固定策略下不会让这个时间预算变短。

复现命令：

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/overlap.sh overlap30_half_mfu_32_64_20260926_01 \
  --threshold-pct 30 --bandwidth-gbps 50 --attention-mfu-scale 0.5
```

新增 `--attention-mfu-scale` 为相对于源估算MFU的比例，默认1保留之前口径；复现使用新run ID。
产物位于 `output/data/overlap30_half_mfu_32_64_20260926_01/`，布局同上一节，
逐层CSV/JSON另外记录原attention时间、缩放后attention窗口、非attention时间、decoder计算及
计算加未隐藏fetch的预算。原MFU结果保留。
全部19项overlap单测通过，覆盖MFU反比缩放、fetch隐藏上限、dense窗口不变和总时间预算的单调性；
独立计算核对两预算结果，并确认默认scale1复现之前全部统计。CLI、Ruff与源码快照检查通过。

## 调整 sparse/dense 分界线

沿用上一节 **attention MFU为原来的0.5、共享50 GB/s链路**，在同一请求上穷尽阈值 `t ∈ [0,100]%`。
每head依然按完整选中并集占比分类：`coverage < t` 为sparse，等号归dense；sparse传并集，
dense传完整KV。两类的attention FLOPs不变，层0的dense fetch无前驱预取窗口。
计算和传输窗口沿用 [overlap.py](src/overlap.py)，没有重新采集激活或执行GPU/DMA测量。

只有跨过某个head的coverage时分类才变化，因此按所有不同coverage构造**左开右闭的分类区间**
（包含0的首区间左闭），在各区间内部求值；重复coverage同时切换。
32/64-block分别有58/59个不同coverage，共 **59/60个分类区间**。
另保存0–100%的整数采样；仅按整数扫描会漏掉32-block的连续最优区间。

阈值不改变计算时间，所以最小化 `sum(R[l])` 与最小化 `sum(A[l]+O[l]+R[l])` 得到相同最优区间。
同时单独求 `sum(H[l])/sum(S[l]+D[l])` 的最大值；阈值改变传输量和效率分母，两个目标可能不同。

### 最短时间预算与最高隐藏比例

已完成离线run `threshold_sweep_half_mfu_32_64_20260926_01`。下表时间均为32层合计，单位ms；
“计算+未隐藏”仍是理想窗口模型的代数预算，不是实际端到端延迟：

| 预算 | 阈值与目标 | Sparse / dense heads | 总fetch | 未隐藏fetch | 总overlap效率 | 计算+未隐藏fetch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 原30% | 49 / 15 | 16.112681 | 4.656297 | 71.10% | 31.629799 |
| 32 | **21.5%，时间与效率均最优** | 43 / 21 | 19.145032 | **3.634605** | **81.02%** | **30.608107** |
| 32 | 22%，整数阈值最优 | 44 / 20 | 18.612879 | 3.769878 | 79.75% | 30.743380 |
| 64 | 原30% | 34 / 30 | 26.276004 | 5.473942 | 79.17% | 35.985260 |
| 64 | **37%，时间最优** | 45 / 19 | 21.170094 | **4.140375** | 80.44% | **34.651694** |
| 64 | 29.5%，效率最优 | 32 / 32 | 27.234796 | 5.159297 | **81.06%** | 35.670616 |

最优是区间，不是某个唯一小数点。下列百分比端点为近似值，精确端点使用整数block比例：

| 预算与目标 | 最优阈值区间 | 精确条件 | 可选阈值 |
| --- | --- | --- | --- |
| 32，最短预算/最高效率 | **(21.057692%, 21.923077%]** | `219/1040 < t/100 <= 228/1040` | **21.5%** |
| 64，最短预算 | **(35.192308%, 38.173077%]** | `366/1040 < t/100 <= 397/1040` | **36%、37%、38%均相同** |
| 64，最高效率 | **(29.326923%, 29.615385%]** | `305/1040 < t/100 <= 308/1040` | 29.5% |

按最短预算选取21.5% / 37%，相对30%的未隐藏fetch分别减少 **1.021692 / 1.333566 ms**，
即 **21.94% / 24.36%**；计算项保持26.973501 / 30.511318 ms，因此计算加未隐藏fetch的预算
分别降低 **3.23% / 3.71%**。这组请求下，两个选块预算需要不同分界线。

### 两类传输分别核算

| 预算与阈值 | 类别 | Fetch ms | 可隐藏 ms | 未隐藏 ms | 类别内效率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 32，21.5% | Sparse union | 4.831969 | 3.093304 | 1.738665 | **64.02%** |
| 32，21.5% | Dense full KV prefetch | 14.313062 | 12.417122 | 1.895940 | **86.75%** |
| 64，37% | Sparse union | 8.220180 | 5.975746 | 2.244435 | **72.70%** |
| 64，37% | Dense full KV prefetch | 12.949914 | 11.053973 | 1.895940 | **85.36%** |

32-block降低阈值后，将6个head从sparse改为dense；全sparse/混合/全dense层由20/9/3变为
14/15/3。每个新增dense头都利用原本空闲的非attention预取窗口，虽然总fetch增加，
未隐藏fetch反而减少。继续降低阈值会让更多层的两个dense头争用同一个预取窗口。

64-block提高阈值后，将11个head从dense改为sparse；全sparse/混合/全dense层由9/16/7变为
16/13/3。完整KV传输减少，dense未隐藏量下降2.522638 ms，代价是sparse未隐藏量增加
1.189072 ms，合计仍减少1.333566 ms。若只追求效率百分比，29.5%可达81.06%，
但其未隐藏fetch为5.159297 ms，高于37%的4.140375 ms；因此建议以未隐藏时间作为调参目标。

这些最优区间来自**单请求的离线扫描**，仅对当前pattern、MFU和带宽假设成立；未证明能泛化到
其他请求。分类预先已知、不计indexer、首块等待、传输启动及资源争用的边界与上一节相同。

### 复现、曲线与明细

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/threshold_sweep.sh --help
bash experiments/nosa_indexer_pattern_65536_1024/scripts/threshold_sweep.sh threshold_sweep_half_mfu_32_64_20260926_01 \
  --baseline-threshold-pct 30 --bandwidth-gbps 50 --attention-mfu-scale 0.5
```

复现使用新run ID；默认读取两组原始 `estimates.json`，支持重复 `--estimate-data-dir`。
[threshold_sweep.py](src/threshold_sweep.py) 复用现有overlap计算，没有另写一套窗口模型。
与原 `overlap.sh` 不同，扫描脚本默认MFU scale为0.5，与本节场景一致。

产物位于 `output/data/threshold_sweep_half_mfu_32_64_20260926_01/`：
`threshold_sweep.png` / `.svg` 分别画两预算的阈值—未隐藏时间和阈值—隐藏比例阶梯曲线，
标注30%基线及两个目标的最优区间；`threshold_regions.csv` 保存119个分类区间，
`threshold_samples.csv` 保存202个整数阈值结果；`selected_layers.csv` / `selected_heads.csv`
保存基线及最优配置的逐层预算和head分类。JSON/Markdown保存全部汇总与定义，
`inputs/`保存输入估算快照，`source/`保存源码，provenance记录输入/源码SHA256、实际分析主机、
Python和Matplotlib版本，并继承原GPU采集与计时来源；本次无预热或重复计时。

独立计算核对全部最优区间、两类效率及相对30%的变化；原MFU与减半MFU的30%结果均复现。
新增数学测试覆盖区间枚举、相同coverage、0/100与等号边界、不同优化目标和参数传递，
由现有全局CPU测试入口自动收集。扫描与overlap共 **48项针对性测试通过**，Ruff、CLI、
源码/输入哈希和CSV行数检查通过；未新增GPU验证。
