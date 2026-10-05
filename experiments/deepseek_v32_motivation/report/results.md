# DeepSeek V3.2 有限缓存对照（refactor_final_deepseek_bench_20261005_01）

同一条顺序访问轨迹包含 16 个用户、2 轮；H=65,536，A=128，C=1,024，P=65,536，NH=16,777,216。只执行这 16 个用户，未填满 NH。

四种方案使用相同的有限 P 历史主 KV 容量。HBM-only 按 P 保留用户，容量不足时淘汰并在复访时重建；ECHO、serial_sparse 和 dense_prefetch 在 NH 内保留历史，共享有限 P 历史槽。候选整批执行。未设置 cache 字节子预算或额外 headroom。

硬件为 NVIDIA H200，PyTorch 2.12.1+cu130 / CUDA 13.0。十个独立 dense block 使用真实 checkpoint 前三层及对应 source hidden/residual 输入；这不是训练得到的十层模型或完整 DeepSeek。GR 输入由共享生成器产生，属于合成内容。

每种方案依次预热首位用户首访、第二位用户首访、首位用户复访，共三次请求。三种 offload 方案的最后一次预热均核验 prefix hit、candidate H2D>0 和 recalled_records>0，确认通用 host gather 路径已执行。随后释放全部用户和共享缓存，从空 cache 执行完整轨迹。端到端 wall latency 包含准入、淘汰、miss 时的历史构建、全部候选 hidden、末 token LM head 和历史清理。输入生成、模型加载、预热、输出复制、诊断统计和数值比较不计时；每请求仅测一次，p95 使用线性插值。

## 首次访问与复访

| 方案 | 访问 | 请求数 | prefix hit | E2E 均值 ms | 中位数 ms | p95 ms | 均值相对 HBM 加速 |
|---|---|---:|---:|---:|---:|---:|---:|
| hbm | 首次 | 16 | 0/16 | 2192.406 | 2192.204 | 2199.826 | 1.000× |
| hbm | 复访 | 16 | 0/16 | 2187.398 | 2187.548 | 2191.141 | 1.000× |
| echo | 首次 | 16 | 0/16 | 2294.994 | 2295.142 | 2299.413 | 0.955× |
| echo | 复访 | 16 | 16/16 | 24.530 | 24.346 | 25.607 | 89.172× |
| serial_sparse | 首次 | 16 | 0/16 | 2228.258 | 2228.076 | 2237.858 | 0.984× |
| serial_sparse | 复访 | 16 | 16/16 | 18.084 | 18.045 | 18.406 | 120.959× |
| dense_prefetch | 首次 | 16 | 0/16 | 2239.190 | 2238.555 | 2246.349 | 0.979× |
| dense_prefetch | 复访 | 16 | 16/16 | 33.197 | 32.528 | 35.543 | 65.892× |

被淘汰后的请求仍计为复访。prefix hit 表示用户固定历史被保留，不表示选中的全部主 KV 已驻留 HBM。

## 候选阶段搬运与缓存

下表的传输与 token 计数只覆盖 candidate forward。后端在该阶段开始时重置计数，因此它们不包含首次构建或重建 history 的流量，不能当作整请求传输总量。

| 方案 | 访问 | candidate H2D 总 GiB | candidate D2H 总 GiB | 精确消费并集驻留率 | dense 历史命中率 |
|---|---|---:|---:|---:|---:|
| hbm | 首次 | 0.000000 | 0.000000 | — | — |
| hbm | 复访 | 0.000000 | 0.000000 | — | — |
| echo | 首次 | 0.000000 | 0.000000 | 100.000% | — |
| echo | 复访 | 1.161186 | 0.000000 | 78.083% | — |
| serial_sparse | 首次 | 0.000000 | 0.000000 | 100.000% | — |
| serial_sparse | 复访 | 1.161186 | 0.000000 | 2.587% | — |
| dense_prefetch | 首次 | 0.000000 | 0.000000 | 100.000% | 100.000% |
| dense_prefetch | 复访 | 11.250000 | 0.000000 | 100.000% | 0.000% |

精确消费并集驻留率按 resident_selection_records / selection_records 汇总，采样位于融合 prefetch 和 append 之后、精确 recall 之前；不表示请求开始时的命中率。dense 历史命中率单独使用 dense_resident_records / dense_requested_records。各原始计数与 hit_ratio_stage 保存在逐请求表中。

## 内存与数值验收

| 方案 | CUDA allocated 峰值 GiB | reserved 峰值 GiB |
|---|---:|---:|
| hbm | 14.517269 | 23.634766 |
| echo | 16.357162 | 24.296875 |
| serial_sparse | 16.358688 | 24.296875 |
| dense_prefetch | 16.358688 | 24.304688 |

每种方案在释放预热资源后、重新分配共享缓存前重置 CUDA allocator 峰值；峰值覆盖已加载权重、共享分配和完整请求轨迹。allocated 为活跃分配，reserved 还含 allocator 保留的空闲缓存，两者不能相加。device free-memory 只在边界采样，不等于连续进程峰值。实际 cache 拥有的 HBM/DRAM 字节与这些峰值分开保存。

数值验收来自独立 check：`/mnt/ssd-wlcb/chenkaiqi/cxldsagr/docs/agents/acceptance/unified_runtime_20261005/deepseek_final/receipt.json`。本次 bench 未逐请求复制、比较或保存完整输出。

逐请求数据见 [per_request.csv](per_request.csv)，分组数据见 [summary.csv](summary.csv)，完整汇总与边界见 [summary.json](summary.json)，验收与来源见 [report_provenance.json](report_provenance.json)。本次运行的完整源数据保留在 `output/data/refactor_final_deepseek_bench_20261005_01/`。
