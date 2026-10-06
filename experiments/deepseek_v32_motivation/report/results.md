# DeepSeek V3.2 有限缓存对照（deepseek_mfu_c10_bench_20261006_01）

同一条顺序访问轨迹包含 16 个用户、2 轮；H=65,536，A=128，C=1,024，P=65,536，NH=16,777,216。只执行这 16 个用户，未填满 NH。

四种方案使用相同的有限 P 历史主 KV 容量。HBM-only 按 P 保留用户，容量不足时淘汰并在复访时重建；ECHO、serial_sparse 和 dense_prefetch 在 NH 内保留历史，共享有限 P 历史槽。候选整批执行。未设置 cache 字节子预算或额外 headroom。

硬件为 NVIDIA H200，PyTorch 2.12.1+cu130 / CUDA 13.0。十个独立 dense block 使用真实 checkpoint 前三层及对应 source hidden/residual 输入；这不是训练得到的十层模型或完整 DeepSeek。GR 输入由共享生成器产生，属于合成内容。

每种方案依次预热首位用户首访、第二位用户首访、首位用户复访，共三次请求。三种 offload 方案的最后一次预热均核验 prefix hit、candidate H2D>0 和 recalled_records>0，确认通用 host gather 路径已执行。随后释放全部用户和共享缓存，从空 cache 执行完整轨迹。端到端 wall latency 包含准入、淘汰、miss 时的历史构建、全部候选 hidden、末 token LM head 和历史清理。输入生成、模型加载、预热、输出复制、诊断统计和数值比较不计时；每请求仅测一次，p95 使用线性插值。

## 首次访问与复访

| 方案 | 访问 | 请求数 | prefix hit | E2E 均值 ms | 中位数 ms | p95 ms | 均值相对 HBM 加速 |
|---|---|---:|---:|---:|---:|---:|---:|
| hbm | 首次 | 16 | 0/16 | 2182.316 | 2179.182 | 2192.458 | 1.000× |
| hbm | 复访 | 16 | 0/16 | 2178.992 | 2179.084 | 2185.062 | 1.000× |
| echo | 首次 | 16 | 0/16 | 2288.305 | 2286.530 | 2298.243 | 0.954× |
| echo | 复访 | 16 | 16/16 | 23.520 | 23.304 | 24.151 | 92.644× |
| serial_sparse | 首次 | 16 | 0/16 | 2225.483 | 2222.269 | 2237.459 | 0.981× |
| serial_sparse | 复访 | 16 | 16/16 | 17.188 | 17.135 | 17.488 | 126.776× |
| dense_prefetch | 首次 | 16 | 0/16 | 2237.748 | 2238.938 | 2244.715 | 0.975× |
| dense_prefetch | 复访 | 16 | 16/16 | 25.462 | 25.408 | 25.918 | 85.579× |

用户被淘汰后再次访问，仍计为复访。prefix hit 表示用户固定历史被保留，不表示选中的全部主 KV 已驻留 HBM。这里的复访端到端收益包含省去 history 重建的收益，不能直接解释为算子加速或搬运重叠收益。

本轮只有一次完整计时轨迹。表中的均值、中位数和 p95 描述这 16 个首访或复访请求，不提供重复运行的置信区间。

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

| 方案 | PyTorch allocated 峰值 GiB | reserved 峰值 GiB | 请求结束时设备已用量最大 GiB | cache HBM 记账最大 GiB | cache DRAM 记账最大 GiB |
|---|---:|---:|---:|---:|---:|
| hbm | 14.517354 | 23.634766 | 24.376038 | 6.624072 | 0.000000 |
| echo | 16.357248 | 24.296875 | 25.665100 | 8.462284 | 320.001038 |
| serial_sparse | 16.358774 | 24.296875 | 25.665100 | 8.462284 | 320.001038 |
| dense_prefetch | 16.358774 | 24.304688 | 25.674866 | 8.462284 | 320.001038 |

每种方案在释放预热资源后、重新分配共享缓存前重置 CUDA allocator 峰值；峰值覆盖已加载权重、共享分配和完整请求轨迹。allocated 为活跃分配，reserved 还含 allocator 保留的空闲缓存，两者不能相加。device free-memory 只在边界采样，不等于连续进程峰值。实际 cache 拥有的 HBM/DRAM 字节与这些峰值分开保存。

Dense 的 HBM 预留为 17.436591 GiB，ECHO 与 sparse fetch 为 17.426820 GiB；差额为 10,491,392 B 的在途预取 ticket workspace。预留包含执行期间可能同时存活的空间，不等于请求结束时的实际 cache 分配，也不能与 PyTorch 峰值相加。完整字节数、graph storage 与 private reservation 见 [memory.csv](memory.csv)。普通 activation 没有独立测峰。

数值验收来自独立 check：`/tmp/cxldsagr-checks/deepseek_v32_motivation/data/deepseek_mfu_c10_check_20261006_01/receipt.json`。本次 bench 未逐请求复制、比较或保存完整输出。

报告发布前重新核验了收据中的 132 项证据文件，并在 CPU 上逐字节比较全部 128 份 candidate hidden/logits；其中 96 组 offload/HBM 对照全部一致。check 与 bench 的 128 条请求中，cache 计数、访问分类和历史命中一致；270 条内存采样与请求记录逐项对应。审查过程没有初始化 CUDA，详见 [独立核验](independent_bench_audit.json)。

## 运行环境观测

计时使用物理 GPU 3、CPU24–31 和 NUMA0。观察器保存 148 次离散采样，最大采样间隔为 30.617866 秒；采样时未发现所选 GPU 上有其他进程。同期 GPU 1 上观察到 PID 3486827，共 24 次，GPU 1 的利用率采样最高为 25%。本轮不属于全机独占运行。离散采样不能排除间隙中的活动，也未监测无关 CPU 工作；这些记录不确定其他 GPU 工作是否影响了本轮延迟。见 [观测摘要](observer_summary.json)。

逐请求数据见 [per_request.csv](per_request.csv)，分组数据见 [summary.csv](summary.csv)，完整汇总与边界见 [summary.json](summary.json)，验收与来源见 [report_provenance.json](report_provenance.json)。本次运行的完整源数据保留在 `output/data/deepseek_mfu_c10_bench_20261006_01/`。
