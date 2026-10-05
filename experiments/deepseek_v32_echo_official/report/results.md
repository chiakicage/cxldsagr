# DeepSeek V3.2 官方 ECHO 适配实验（refactor_final_official_bench_20261005_01）

本轮正式计时复用匹配的独立验收收据。数值表来自该验收；计时进程不逐请求保存输出、比较参考或重跑 HBM 数值轨迹。

本轮使用 NVIDIA H200，PyTorch 2.12.1+cu130 / CUDA 13.0。16 个用户顺序访问 2 轮，H=65,536、A=128、chunk=1,024、P=65,536、NH=16,777,216。

官方 ECHO 的 allocator、融合 indexer/prefetch 和精确 recall 接入本地 GR 生命周期；HBM 对照使用官方 resident logits，两方案采用官方原始 top-k 语义。模型权重、投影和 FlashMLA 沿用 motivation，indexer 不加 Hadamard。HBM 对照也已替换 motivation 的 mainline DeepGEMM logits 和 FlashInfer top-k。结果覆盖这条适配路径，不能表述为未经修改的上游 serving。十个独立 dense block 复用真实 checkpoint 前三层及对应 source hidden/residual 输入，共 7,827,793,408 个参数；这是工作负载替身。

两方案各预热三条请求，包含两位用户首访及第一位用户复访；ECHO 最后一次预热验证实际 host recall。释放全部预热 cache 后，从空缓存执行正式轨迹。每条请求只测一次，端到端同步墙钟包含输入搬入、准入/淘汰、miss 时 history 构建、全部 candidate hidden、末 token LM head 与清理。forward 内的 GPU 计数累积和归约也计入延迟；加载、编译、预热、计数的 host 读取、输出保存和比较不计时。

| 方案 | 访问 | 请求数 | history hit | 均值 ms | 中位数 ms | p95 ms | 相对 HBM 加速 |
|---|---|---:|---:|---:|---:|---:|---:|
| HBM-only | 首次 | 16 | 0/16 | 2126.499 | 2126.137 | 2132.862 | 1.000× |
| HBM-only | 复访 | 16 | 0/16 | 2123.192 | 2124.119 | 2126.623 | 1.000× |
| Official ECHO (adapted offload pipeline) | 首次 | 16 | 0/16 | 4596.177 | 4593.201 | 4609.691 | 0.463× |
| Official ECHO (adapted offload pipeline) | 复访 | 16 | 16/16 | 44.862 | 44.516 | 46.391 | 47.327× |

完整轨迹的请求时间之和：HBM-only 67.995 s，官方 ECHO 适配路径 74.257 s，加速 0.916×。被淘汰后的请求仍计为复访；history hit 只表示用户历史被保留，不代表所选 KV 已在 HBM。

| 方案 | 访问 | candidate H2D GiB | candidate D2H GiB |
|---|---|---:|---:|
| HBM-only | 首次 | 0.000000 | 0.000000 |
| HBM-only | 复访 | 0.000000 | 0.000000 |
| Official ECHO (adapted offload pipeline) | 首次 | 0.000000 | 0.000000 |
| Official ECHO (adapted offload pipeline) | 复访 | 1.160774 | 0.000000 |

搬运计数只覆盖 candidate forward，不包含 history prefill。GPU 计数累积和归约包含在 forward 计时中，host 读取在计时外。候选整批在 GPU 执行后丢弃，NH 只保存历史。P/NH 是 token 容量，不是总 HBM/DRAM 字节数；不扣除经验性 headroom。

| 方案 | CUDA allocated 峰值 GiB | reserved 峰值 GiB | cache HBM 最大 GiB | cache DRAM 最大 GiB |
|---|---:|---:|---:|---:|
| HBM-only | 14.549007 | 23.875000 | 6.625494 | 0.000000 |
| Official ECHO (adapted offload pipeline) | 16.409619 | 24.521484 | 8.485923 | 320.001038 |

allocator 峰值在释放预热 cache 后重置，包含已加载权重与完整正式轨迹。allocated 是活跃分配，reserved 还含 allocator 缓存，二者不能相加。cache 数字来自请求边界；设备 free memory 也只在边界采样，不等于连续进程峰值。

官方 top-k 的输出顺序会随执行变化，HBM 重复运行也可能产生不同的输出。独立验收从空缓存执行完整 HBM 和 ECHO 轨迹，并额外重复一次 HBM；全部候选 hidden/logits 通过预先固定的数值门槛。

| 比较 | 输出 | 最大绝对误差 | 最大 relative L2 | 最低元素通过比例 | 逐位相等请求 | allclose 请求 | 数值通过请求 |
|---|---|---:|---:|---:|---:|---:|---:|
| 官方 ECHO / HBM | hidden | 0.1796875 | 0.0025282611 | 99.99291556% | 0/32 | 31/32 | 32/32 |
| 官方 ECHO / HBM | logits | 0.0625 | 0.0025014025 | 100.00000000% | 0/32 | 32/32 | 32/32 |
| HBM 重复 / HBM | hidden | 0.046875 | 0.0012273704 | 100.00000000% | 0/32 | 32/32 | 32/32 |
| HBM 重复 / HBM | logits | 0.0625 | 0.0040260684 | 99.99845297% | 0/32 | 31/32 | 32/32 |

误差在 CPU 上用 FP64 重新计算。每条请求的 hidden / logits 须分别满足 relative L2≤0.005 / 0.01，且至少 99.9% 元素满足 |actual−reference|≤1/32+(1/64)×|reference|。全元素 allclose、最大绝对误差与逐位相等仅作为观测项，不单独决定验收。这些规则依据独立 64K resident 校准，在正式 64K offload 输出测量前固定；完整依据及 hash 保存在 summary.json 的 numerical_policy / numerical_policy_basis。top-k 的顺序和并列值可能影响结果，不能把全部差异都归因为已证实的舍入误差。报告也检查所有保存输出、每层 P/NH、候选生命周期，以及源码和 native 二进制快照。这些结果只适用于本轮合成 GR 输入和工作负载替身。

逐请求数据见 [per_request.csv](per_request.csv)，分组数据见 [summary.csv](summary.csv)，完整配置与验收见 [summary.json](summary.json)，来源见 [report_provenance.json](report_provenance.json)。完整产物在 `output/data/refactor_final_official_bench_20261005_01/`。
