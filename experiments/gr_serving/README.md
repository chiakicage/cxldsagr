# GR 多用户 Serving 缓存对照

本实验比较固定用户 history、每次变化 candidate 的串行请求延迟，以及排除第一次访问后
的复访延迟。先测试 DeepSeek V3.2 前三层及输入复制得到的约 8B 代理负载，再测试完整
NOSA-8B。仅使用一张 GPU，不包含网络服务、并发排队或模拟到达时间等待。

**2026-10-02 当前判断：ECHO / DeepSeek baseline 尚未成立，需审计、修正并复测。**
研究者指出 ECHO 实现的 MFU 与 cache 策略有问题，DeepSeek 当前实现的 MFU 也不太正确。
现有输出一致、申报预算与边界采样、数据可追溯的检查不能证明实现效率、MFU 口径或
cache 策略合理，也不证明临时 cache scratch 的全过程峰值已受预算约束。
因此下文涉及 DeepSeek HBM / ECHO / serial sparse / dense prefetch 的旧排名，暂不能用于
判断 baseline 优劣、offload 代价或 ECHO 设计价值；MFU 数字和 cache 行为需独立审计。
问题的具体来源及修正方案尚未确认，本次没有修改算子或完成受影响部分的 GPU 复测。
旧 run ID、源码身份、数字、图表和原始数据保留至修正后的结果通过验收并发布，
不视为当前 baseline 已验收。合理的慢对照仍应保留并按一致要求复测。

**最近的补测尝试：**用户改为先运行 **64K history + 128 candidate** 的小组：
16 个用户按 ID `0..15` 顺序访问两遍，共 32 条请求，不使用热度分布。
每方案首访和复访各 16 次；两模型各四方案共 8 cases / 256 条测量。
统一预算为 **HBM 4 GiB / DRAM 64 GiB**，DeepSeek sparse slots 为 32768。
此配置是受控循环访问，不代表工业热度或真实到达序列。
`gr_serving_h200_20261002_sequential_u16_t32_h64k_01` 在最终源码冻结检查失败，
不作为实验结果；诊断保留在系统临时目录。原执行记录计划以隔离源码的 run 02 完整重跑，
尚无已验收的替换结果，且仅更换请求轨迹不能解决上述 baseline 实现问题。
先前 industrial 10M、1024 用户、4096 请求的大组已在输入生成阶段停止，没有形成
测量结果；4K、16K 暂不启动。下文旧报告及原始数据保留至新结果通过验收并发布。

工作负载状态：**旧短轨迹的数值与计量核验已完成；用户规模和缓存容量压力实验未成立，需重构负载后补测。**
研究者指出，固定很少的请求却扩大抽样用户池，不能形成有意义的用户规模对照；没有复访
miss 的档位也没有检验缓存容量收益。下文保留原 run 的实际数据和测量边界，不能据其
配置人数横轴推断规模扩展能力。16K 的 13 次 HBM 复访 miss 是该条短轨迹的现象，
不能补足其他规模的覆盖。此前工作负载修订选择按热度概率独立有放回抽取总计 T 次，取消默认
8 次上限，实际人数和复访分布由结果统计；不再强制配额或覆盖。入口与统计已调整，
**IID 热度轨迹尚未完成 GPU 补测**，后续以实际人数、复用距离和缓存预算判断实验覆盖。
补测完成并验收后再替换受影响的规模图与结论，旧数字不代表新构造的结果。

原 4K、16K、64K 三个矩阵及其 NOSA layer 31 profile 均通过各自的原始数值/计量检查。
共 168 组、3552 个测量请求，2664 条非 HBM 完整 candidate hidden 比较逐元素一致。
七档用户为 1/8/32/64/128/256/512；每用户最多复访 8 次，按 Beauty 热度抽样，达到上限
才移出采样，不强制首访或补齐用户次数。candidate 固定为 128 tokens，统一 cache 配额
为 HBM 4 GiB / DRAM 16 GiB。

| 固定历史 | 正式 run ID | 每档请求上限 | 全矩阵请求数 | NOSA HBM 复访 miss / 每方案复访数 | 报告 |
|---|---|---:|---:|---:|---|
| 4K | `gr_serving_h200_20261002_h4k_01` | 32 | 1608 | 0 / 63 | [4K](#4k-结果与结论) |
| 16K | `gr_serving_h200_20261002_h16k_01` | 32 | 1608 | 13 / 63 | [16K](#16k-结果与结论) |
| 64K | `gr_serving_h200_20261002_h64k_01` | 6 | 336 | 0 / 8 | [64K](#64k-结果与结论) |

16K NOSA 在统一预算下观察到复访容量收益：HBM 有 13 次历史重建，三个 offload 方案都
没有复访 miss。但 NOSA 三个长度各档的全部请求均值仍以 HBM 最低，不能将复访子集收益
改写为完整 trace 的收益。4K、64K 的已观测复访全部命中，尚未观察到此类容量收益。
64K 只有 6 请求窗口，四档没有复访；其 null 统计和实际覆盖明确展示，不与 32 请求窗口
的全部请求均值直接计算跨历史加速比。

NOSA overlap 相对 serial sparse 的全部请求均值在 21 个长度/用户组合中有 20 个更低，
但复访均值没有形成稳定改善；全请求结果包含冷 prefix 构建成本，二者分别报告。九个独立
layer 31 样本的 page/stripe ratio 均未达到 90%，这不会推翻已测得的请求延迟差异。机制
记录只覆盖实际 softmax 区间，整体判断使用未插桩请求时间。原算子边界另见
[原算子实验](../nosa_offload_overlap/README.md)。

## 内容与测量边界

- DeepSeek：纯 HBM、ECHO offload、串行 sparse fetch、dense 逐层预取。
- NOSA：纯 HBM、串行 sparse fetch、dense 逐层预取、attention 与 sparse fetch overlap。
- 每组对照保持相同的允许 HBM cache / CPU DRAM cache 字节上限；同时报告实际占用。
  纯 HBM 方案不使用 CPU DRAM cache。模型权重和临时计算内存另行披露，不与 cache 预算混同。
  HBM budget 是人为设置的统一 cache 配额，远小于本机 H200 的物理显存容量；该配置用于
  观察用户工作集越过配额后的行为，不证明这些用户数会耗尽 H200 的全部显存。
- 跨请求保存完整用户 prefix session；按全局 LRU 淘汰，候选 suffix 不进入下一次访问的
  固定 prefix。命中与否以实际 session 状态和精确 token identity 判定。
- `latency_ms` 为同步后的逐请求墙钟时间，包含 miss 时的 prefix 构建、candidate 执行、
  截短和清理；`prefix_ms`、`extend_ms`、`cleanup_ms` 若有单独测量则同时报告。
  权重加载、编译、独立预热和工作负载生成不计入请求延迟；每种方案测量前重置为空 cache。
- `visit_index > 0` 表示复访，无论此前 prefix 是否已被淘汰。报告所有请求、首访、复访
  三组的数量、均值、中位数、p95 和 p99；分位数使用 `(n - 1) * p` 位置线性插值。
  单次 trace 的 p95/p99 是描述统计，不代表置信区间；无复访时结果为 null，不写为零。

## 模型与实现范围

DeepSeek 使用 `/preset-models` 的真实 embedding、norm、LM head 及源层 0/1/2 权重。
10 个独立物理 dense block 按 `[0,1,2,0,1,2,0,1,2,0]` 复制，共 7,827,793,408 参数；
其中 dense block 共 5,974,428,160 参数，端点共 1,853,365,248 参数。复制层同时接收
对应源层的 hidden 与 residual 输入副本，避免将未训练的重复层当作新的深层模型传播。
这是 checkpoint 支持的执行负载替身，不是训练得到的 8B 推荐模型；没有执行 MoE。
每条请求计算全部 candidate normalized hidden 和最后 token 的 LM head。

NOSA 严格加载完整 NOSA-8B checkpoint 的 32 层，所有方案均执行完整 NOSA sparse
selection、CIS 与 causal attention，返回全部 candidate normalized hidden，不执行
LM head/decode。因此延迟排名只在各模型内部比较，不将两模型的绝对时间当作同一任务比较。
这里的 dense prefetch 指搬运整层历史 K/V，attention 仍使用相同的稀疏选择。

DeepSeek ECHO / serial sparse 每层为每用户保留显式配置的 main-KV 槽位；4K/16K/64K
分别使用 4096/8192/32768 slots，两种稀疏 offload 使用相同设置。候选选择并集
超出槽位时，沿用原实现拆分 query 消费，保留每 query 精确 top-k；缓存管理、重复 gather
和增加的 attention launch 全部计入请求时间。ECHO 每个 scoring chunk 会腾出历史槽位
用于融合预取，用户 session 命中不意味着上次选中的历史记录全部继续命中 HBM。
这些记录描述旧实现的行为。研究者已指出 ECHO cache 策略与 MFU 问题，当前不能以此
接受这两个 baseline 的合理性；还需审计各方案的计算效率、槽位保留/清空、精确 recall、
重复搬运和调度开销，并在修正后重新对照。不能将全部额外开销归因于 DRAM 带宽或融合机制。

NOSA offload 保留完整逻辑地址的一层 staging，dense prefetch 使用两层交替 staging。
跨用户的有限预算与淘汰由 session LRU 提供；NOSA 内部仍没有 token/page slots 或页级
eviction。准入为每个用户保留 cache 的峰值上界，包含 scratch 和 pending append，
因此不是利用所有空闲字节的最优装填策略。报告分别列 cache 分配、准入预留和 CUDA
allocator 峰值，三者不混用。

## 工作负载

复用 [GR 请求生成器](../../GR/README.md) 的 Beauty `interaction_count` 累计热度曲线，
通过分段线性插值生成指定数量的用户并作加权有放回抽样。文本为合成的可读历史与候选，
不包含真实推荐标签，热度曲线不证明此负载代表真实 GR 服务。

每用户 history 固定，candidate 随 `visit_index` 改变。默认 seed 为 42。
默认按固定热度概率独立有放回抽样，无逐用户复访上限。`--requests` 是总访问次数 T，
`--users` 是概率向量的用户池大小；实际出现的人数及复访用户数由结果统计，不能以用户池
大小代替实际人数。旧 `--max-revisits` 只在显式使用时启用，此时 T 为可提前耗尽的上限。
不分配逐用户配额、不强制首访；manifest 保存实际请求数、访问及复访用户数、逐用户
访问/复访次数和最大复访数。
`history_tokens` 精确等于指令与历史合起来的完整 KV prefix 长度；`candidate_tokens`
精确等于变化 suffix 长度，不能直接用 GR 的 `user_tokens` / `item_tokens` 代替。
用户规模应覆盖小于、接近和超出公共 HBM budget 可容纳用户数的情况，具体网格由运行参数记录。

每个用户规模只生成一次 workload，所有方案共享相同请求顺序及内容；manifest 保存逐请求
用户、visit、完整 input hash、prefix hash、candidate hash、热度资源 hash、tokenizer hash，
以及整体 `workload_sha256`。不强制每用户至少访问一次，报告实际覆盖用户数和复访数量。

## 运行

从仓库根目录执行，测量入口以 DeepSeek、NOSA 为默认模型顺序。以下保留原有界矩阵的
参数示例，含显式 8 次上限；它不代表当前无 cap 请求流的测量，也不能作为新规模结论：

```bash
bash experiments/gr_serving/scripts/run.sh --help
bash experiments/gr_serving/scripts/run.sh <NEW_RUN_ID> \
  --users 1 8 32 64 128 256 512 --history-tokens 16384 --candidate-tokens 128 \
  --requests 32 --max-revisits 8 --deepseek-slots 8192 \
  --hbm-budget-gib 4 --dram-budget-gib 16
```

完整矩阵会保存逐请求 HBM 参考 hidden，临时目录需有足够空间。本次运行设置
`TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving`，避免使用容量仅 20 GiB
的根文件系统；该设置只改变中间产物位置。

单独生成 workload（不加载模型，也不构成性能实验）：

```bash
.venv/bin/python -m experiments.gr_serving.src.workload \
  --model nosa --num-users 8 --requests 128 \
  --history-tokens 16384 --candidate-tokens 1024 --seed 42 \
  --tokenizer /mnt/ssd-wlcb/chenkaiqi/NOSA-8B/tokenizer.json \
  --output-dir /tmp/gr-serving-workload
```

报告重建入口（`measurements.jsonl` 为测量入口交付的逐请求数据）：

```bash
.venv/bin/python -m experiments.gr_serving.src.report \
  --input experiments/gr_serving/output/data/<RUN_ID>/measurements.jsonl \
  --output-dir experiments/gr_serving/output/data/<RUN_ID>/analysis
```

报告器拒绝缺失请求、同组方案输入不一致、预算不一致、warmup 混入结果和超预算数据。
输出 `per_request.csv`、`per_request.svg`、`summary.csv`、`summary.json` 和 `summary.svg`，
包含实际命中和淘汰信息。逐请求 SVG 中空心点为首访，实心点为复访；汇总 SVG 按模型分别
比较用户数增加时的全部请求和复访延迟，实线为均值、虚线为 p95，保留淘汰后 miss 的影响。

调用模块：`GR.input_generator`、`GR.heat`、`GR.scheduling`、`serving.persistent`、
`cache.prefix_pool`、`models.deepseek_v32.serving_backend`、`models.nosa.serving`。
模型适配复用原模型、indexer、attention、fetch、linear 实现，测量入口不复制模型计算。
shell 只编排 Python 入口并保留失败状态。
成功运行分别写入 `output/data/<run_id>/`、`output/log/<run_id>/` 和
`output/profile/<run_id>/`；失败诊断留在系统临时目录，不作为实验结果保存。

## 4K 结果与结论

本节为旧 run 的实测记录；DeepSeek / ECHO 对照受上述 MFU 与 cache 策略问题影响，
当前不能据其排名作 baseline 或设计结论。数值一致性与原始数据核验的边界保持不变。

来源为 `gr_serving_h200_20261002_h4k_01`，输入 **4096 + 128**，prefix chunk 为 1024，
DeepSeek ECHO / serial sparse 每层每用户均为 4096 slots。每种方案先执行 2 条不计入结果的
预热请求，再对每档用户规模从独立空 cache 开始执行同一 trace；每条请求只测量一次。
硬件为 NVIDIA H200（SM90，132 SM，150,121,545,728 B 物理显存），pinned backing 使用
默认进程 NUMA 策略下的本机 CPU DRAM。软件为 Python 3.12.13、PyTorch 2.12.1+cu130、
CUDA 13.0、Triton 3.7.1、FlashInfer 0.6.18、TVM FFI 0.1.13.post3、
Safetensors 0.8.0、Tokenizers 0.23.2。DeepSeek 使用 FP8 linear 和 BF16 main KV；
NOSA 使用 BF16。更完整的配置与依赖在[运行 metadata](report/h4k/metadata.json)。

[独立审计](report/h4k/audit.json)确认 56 组完整覆盖、1608 条测量/数值记录，1206 条非 HBM
输出与相同输入的 HBM 参考逐元素完全一致（`atol=rtol=0`）；所有预算及逐请求 LRU 行为通过检查。
这是已保存数值记录的审计，未在发布阶段重新执行模型。源码身份固定为
`67f91c34ead43d4056500661caa1c67e8ef8a50bc25327e081a0561bbab43060`，
153 个文件的身份见[源码 manifest](report/h4k/source_manifest.json)；基准 Git revision 为
`16dc058014ee0474a0fa0893ad89271b890aeb92`，测量包含 manifest 记录的工作区修改。

### 实际访问覆盖

两模型和各方案的用户访问顺序一致；token 内容按模型 tokenizer 分别生成。下表每档的数量
是每个方案的实测数量，不是各方案求和。用户总体规模不等于这条短 trace 已经访问的用户数。

| 配置用户数 | 请求数 | 实际用户数 / 首访数 | 复访数 | 实际最大复访数 |
|---:|---:|---:|---:|---:|
| 1 | 9 | 1 | 8 | 8 |
| 8 | 32 | 8 | 24 | 8 |
| 32 | 32 | 21 | 11 | 2 |
| 64 | 32 | 21 | 11 | 3 |
| 128 | 32 | 28 | 4 | 2 |
| 256 | 32 | 30 | 2 | 1 |
| 512 | 32 | 29 | 3 | 1 |

完整覆盖数据见[population_coverage.csv](report/h4k/population_coverage.csv)。N=128/256/512
只有 4/2/3 次复访，其 p95 是这些小样本的插值统计，不能用于推断稳定的服务尾延迟。

### 请求延迟

表中每格为 **均值 / p95，单位 ms**；“全部”保留首次访问的 prefix 构建成本，
“复访”按用户访问次数排除首访。各模型仅在自身四种方案之间比较。

**DeepSeek checkpoint 工作负载替身**

| 用户数 | 请求范围 | HBM | ECHO | Serial sparse | Dense prefetch |
|---:|---|---:|---:|---:|---:|
| 1 | 全部 | 85.23 / 315.98 | 112.90 / 380.58 | 97.38 / 346.30 | 85.18 / 315.08 |
| 1 | 复访 | 32.86 / 33.40 | 52.20 / 53.21 | 40.82 / 40.97 | 32.94 / 33.09 |
| 8 | 全部 | 150.81 / 505.65 | 195.63 / 628.41 | 169.66 / 552.10 | 151.32 / 504.54 |
| 8 | 复访 | 32.94 / 33.09 | 52.74 / 52.90 | 42.08 / 42.54 | 33.69 / 33.96 |
| 32 | 全部 | 342.36 / 506.21 | 423.93 / 630.97 | 377.79 / 554.96 | 343.71 / 507.64 |
| 32 | 复访 | 33.59 / 34.34 | 53.75 / 54.93 | 42.98 / 43.47 | 34.39 / 35.02 |
| 64 | 全部 | 342.20 / 505.12 | 411.93 / 601.49 | 377.48 / 554.68 | 343.52 / 506.91 |
| 64 | 复访 | 33.59 / 34.39 | 53.12 / 54.57 | 42.77 / 43.72 | 34.42 / 35.52 |
| 128 | 全部 | 445.59 / 506.71 | 539.62 / 634.00 | 493.06 / 555.35 | 449.38 / 510.74 |
| 128 | 复访 | 33.86 / 34.49 | 54.20 / 55.52 | 42.87 / 43.96 | 35.30 / 36.37 |
| 256 | 全部 | 474.86 / 506.57 | 568.82 / 617.24 | 521.52 / 556.11 | 479.25 / 510.92 |
| 256 | 复访 | 34.44 / 34.72 | 54.49 / 55.21 | 43.90 / 44.40 | 36.14 / 36.72 |
| 512 | 全部 | 460.36 / 506.65 | 550.13 / 603.48 | 505.07 / 555.35 | 464.50 / 511.56 |
| 512 | 复访 | 34.28 / 35.04 | 53.92 / 54.87 | 43.08 / 43.97 | 35.94 / 37.05 |

**NOSA-8B，完整 32 层 hidden 输出**

| 用户数 | 请求范围 | HBM | Serial sparse | Dense prefetch | Overlap |
|---:|---|---:|---:|---:|---:|
| 1 | 全部 | 42.41 / 104.33 | 57.91 / 142.74 | 58.16 / 139.20 | 58.07 / 142.71 |
| 1 | 复访 | 28.34 / 28.42 | 38.72 / 39.37 | 39.77 / 40.06 | 38.87 / 39.15 |
| 8 | 全部 | 60.66 / 157.22 | 101.49 / 300.31 | 83.28 / 206.99 | 82.84 / 213.60 |
| 8 | 复访 | 28.69 / 29.57 | 39.24 / 39.68 | 42.08 / 51.20 | 39.34 / 39.59 |
| 32 | 全部 | 113.20 / 158.54 | 188.43 / 300.43 | 150.88 / 211.06 | 154.32 / 217.72 |
| 32 | 复访 | 29.09 / 29.44 | 39.58 / 39.92 | 40.95 / 42.13 | 39.47 / 40.28 |
| 64 | 全部 | 112.99 / 157.95 | 153.41 / 213.90 | 149.97 / 208.96 | 153.24 / 213.63 |
| 64 | 复访 | 29.10 / 29.52 | 39.62 / 40.01 | 40.70 / 41.18 | 39.62 / 40.16 |
| 128 | 全部 | 142.20 / 160.27 | 210.67 / 301.91 | 189.39 / 227.86 | 191.24 / 213.92 |
| 128 | 复访 | 29.37 / 29.59 | 39.75 / 40.13 | 40.76 / 41.15 | 39.99 / 40.57 |
| 256 | 全部 | 150.07 / 160.09 | 207.95 / 253.67 | 197.43 / 209.66 | 202.26 / 214.70 |
| 256 | 复访 | 29.86 / 30.40 | 39.97 / 40.04 | 41.26 / 41.45 | 40.55 / 40.85 |
| 512 | 全部 | 146.01 / 159.55 | 197.37 / 216.17 | 192.45 / 209.86 | 196.91 / 214.44 |
| 512 | 复访 | 29.16 / 29.52 | 39.81 / 40.92 | 41.11 / 41.62 | 40.19 / 40.82 |

![4K serving 全部请求与复访延迟](report/h4k/summary_readable.svg)

图中用户规模按类别等距排列，各面板采用独立纵轴；实线为均值，虚线为 p95。
[PNG 预览](report/h4k/summary_readable.png)、[原始汇总 SVG](report/h4k/summary.svg)、
[逐请求图 SVG](report/h4k/per_request.svg)、[逐请求图 PNG](report/h4k/per_request.png)均可查看。
逐请求图以空心点标首访、实心点标复访。完整数值保存于
[summary.csv](report/h4k/summary.csv)、[summary.json](report/h4k/summary.json)和
[per_request.csv](report/h4k/per_request.csv)，包含首访、median/p99、阶段时间、命中和淘汰。

DeepSeek 各档复访均值为 HBM 32.86–34.44 ms、dense prefetch 32.94–36.14 ms、
serial sparse 40.82–43.90 ms、ECHO 52.20–54.49 ms。NOSA HBM 为 28.34–29.86 ms，
serial sparse 为 38.72–39.97 ms，overlap 为 38.87–40.55 ms；此次 overlap 未展示
稳定的复访延迟改善。上述数字保留原实现的算子、同步与缓存管理总成本；DeepSeek 排名
受 MFU 与 ECHO cache 策略问题影响，不能用于接受 baseline 或归因于 DRAM 带宽/融合机制。
各档全部请求均值的上升主要伴随首访占比上升；它不是容量收益的证据。

### Cache 预算、实际分配与 CUDA allocator

所有方案的允许 cache 配额相同：**HBM 4 GiB / DRAM 16 GiB**。下表“容量”为保守的单
session 预留量在两个预算下允许准入的 session 数；“边界峰值”为七档请求中观测到的最大
cache 自有分配，已排除模型权重和普通 activation。两个字节量并不相同。

| 模型 | 方案 | 单 session 预留 HBM / DRAM (MiB) | 准入容量 | 边界 cache 峰值 HBM / DRAM (MiB) | CUDA allocator 峰值 allocated / reserved (GiB) |
|---|---|---:|---:|---:|---:|
| DeepSeek | HBM | 52.53 / 0.00 | 77 | 1575.90 / 0.00 | 11.96 / 12.24 |
| DeepSeek | ECHO | 51.10 / 46.41 | 80 | 1533.12 / 1392.19 | 11.93 / 12.24 |
| DeepSeek | Serial sparse | 51.10 / 46.41 | 80 | 1533.12 / 1392.19 | 11.93 / 12.24 |
| DeepSeek | Dense prefetch | 15.40 / 46.41 | 265 | 462.15 / 1392.19 | 10.87 / 12.24 |
| NOSA | HBM | 137.19 / 0.00 | 29 | 3964.23 / 0.00 | 24.47 / 24.89 |
| NOSA | Serial sparse | 43.32 / 132.00 | 94 | 294.84 / 3960.00 | 15.73 / 24.98 |
| NOSA | Dense prefetch | 45.44 / 132.00 | 90 | 388.42 / 3960.00 | 15.82 / 24.98 |
| NOSA | Overlap | 43.32 / 132.00 | 94 | 294.84 / 3960.00 | 15.73 / 24.98 |

[cache_and_memory.csv](report/h4k/cache_and_memory.csv)保留精确字节数，分别列出 truncate 后
分配、请求边界分配、准入预留及 CUDA allocator 峰值。NOSA HBM 容量为 29 sessions，
N=256 的 trace 访问 30 个用户并发生 1 次淘汰，但被淘汰用户未再次访问；其他所有组未发生
淘汰。所有复访均命中完整用户 prefix，因此本次 4K 结果未证明 offload 的容量收益。

CUDA allocator 峰值覆盖每个 case 中进程当时的全部存活分配，包括模型切换时尚未释放的
旧对象；reserved 也包含此前保留的 allocator 内存块。第一组 NOSA HBM 的 allocated
峰值为 26,273,759,232 B，包含模型切换保留，不应解释为 NOSA 自身的干净峰值。
这些统计不等于物理进程显存峰值，也不能相减得到普通 activation 的独立峰值。
DeepSeek 单模型权重记录为 9,864,625,792 B；NOSA 的 8,185,270,336 个 BF16 参数对应
16,370,540,672 B 逻辑载荷，这是按 dtype 计算的参数载荷，不是独立测得的 allocator 峰值。
普通 activation 与 cache 分开计量：每请求完整 candidate hidden 输出，DeepSeek 为
`[128,7168]` BF16，即 1,835,008 B；NOSA 为 `[128,4096]` BF16，即 1,048,576 B。
这些是数值验收记录覆盖的输出张量载荷，不包括中间 activation 和算子临时空间。
普通 activation 没有单独的峰值测量；其全部开销包含在上述进程 allocator 统计内。

### NOSA layer 31 内部 overlap

独立 profile run 为 `gr_serving_h200_20261002_h4k_profile_02`，使用上述 8 用户 workload
的前三条复访（request ID 2/3/7），仅观察 layer 31。每条请求对 serial sparse / overlap
各采集 1 个样本、各执行 1 次未插桩数值对照，profile warmup 为 0；所有模式从独立空
cache 构建 sparse prefix。六个样本的全部 candidate hidden 均与未插桩对照及正式 HBM
参考逐元素一致，实际逻辑选择、唯一读取字节及 page/stripe 结构检查通过。

下表 ratio 的分子为 copy 区间并集与实际 consumer softmax 区间并集的交集时长，分母为
相应 copy 区间并集时长。page envelope 必须等于该页全部非空 stripe 的最早 start / 最晚
end；非空 stripe 指标独立计算，不把 envelope 内空隙当作 copy。此次两个全局区间并集
恰好相等，因此两种 ratio 数值相同。

| 复访 request ID | 唯一历史 K+V 逻辑字节 | Page-envelope ratio | 非空 stripe-copy ratio | 两者均 ≥90% |
|---:|---:|---:|---:|---|
| 2 | 4,161,536 | 72.87% | 72.87% | 否 |
| 3 | 4,161,536 | 74.34% | 74.34% | 否 |
| 7 | 4,161,536 | 75.24% | 75.24% | 否 |

每样本为 127 个历史 page、1016 个非空 stripe，logical payload 均为 4,161,536 B。
三个 overlap 样本均未满足 90% 门槛，仍保留为有效测量。此处 math 覆盖仅为 **softmax**，
不是完整 attention 计算覆盖，也不是 PCIe 线速占用或整体延迟隐藏率。serial 样本没有
softmax 区间插桩，其 overlap ratio 保留 null，不推定为零。整体性能判断仍使用前述未
插桩完整 serving 延迟，不能以内部区间相交代替请求延迟收益。

该诊断额外保留 trace 与 selection，没有执行多用户 LRU 准入，不能充当正式 cache 预算
占用测试。分析定义及六条样本见[profile_analysis.json](report/h4k/profile_analysis.json)，
数值、存储与源码边界见[profile_metadata.json](report/h4k/profile_metadata.json)。
profile 的独立源码 SHA 为 `cd72caf5edbac19a54d162e26d3339b05dd0671048162d970729669af5d0c48b`；
与正式延迟源码相比，仅诊断入口 `src/profile.py` 及其 CPU 测试有显式允许的修改，
旧/新文件 SHA 与原始源码副本验证记录在 `formal_source_comparison` 中。运行时性能代码未改，
正式延迟的源码身份仍为前述 `67f91c34…43060`。

复现入口须在正式 GPU 测量结束后运行，使用新的 run ID：

```bash
bash experiments/gr_serving/scripts/profile.sh <NEW_PROFILE_RUN_ID> \
  --latency-data experiments/gr_serving/output/data/gr_serving_h200_20261002_h4k_01 \
  --num-users 8
```

### 复现与素材来源

本次正式命令（从仓库根目录执行）：

```bash
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving \
  bash experiments/gr_serving/scripts/run.sh gr_serving_h200_20261002_h4k_01 \
  --users 1 8 32 64 128 256 512 --history-tokens 4096 --candidate-tokens 128 \
  --requests 32 --max-revisits 8 --deepseek-slots 4096 \
  --hbm-budget-gib 4 --dram-budget-gib 16
```

原始有效数据仍保留在 `output/data/gr_serving_h200_20261002_h4k_01/`，日志在对应
`output/log/`。`report/h4k/` 的五个原始 CSV/JSON/SVG 与该 run 的 `analysis/` 内容逐字节
一致，发布前对照审计中的 SHA-256；完整对应关系见[provenance.json](report/h4k/provenance.json)。
原始 SVG 的 PNG 使用 librsvg/cairo 按原始尺寸渲染；所有逐请求面板已检查。

为避免原始汇总图 1/8 用户标签重叠，并展示复访差异，`summary_readable.svg/png` 直接从
已核验的 `summary.json` 重绘，只改变横轴类别间距与面板纵轴。生成脚本保存为复现材料
`output/data/gr_serving_h200_20261002_h4k_01/render_report.py`，源码 SHA-256 为
`71704ada5c64e417f4af703f21043eca5d3232bb5e26934cc6dc0ef93bec166d`，也记录在 provenance。
它可用于其他已验收 run，只需指定对应 summary、metadata 和输出目录：

```bash
.venv/bin/python experiments/gr_serving/output/data/gr_serving_h200_20261002_h4k_01/render_report.py \
  --summary experiments/gr_serving/report/h4k/summary.json \
  --metadata experiments/gr_serving/report/h4k/metadata.json \
  --output-dir experiments/gr_serving/report/h4k
```

`population_coverage.csv` 选取 metadata 的各档一致访问计数；`cache_and_memory.csv` 从
metadata 和 audit 逐方案取七档最大值，预留容量保留审计结果。二者的精确字段及来源在
provenance 中登记。图表检查与文档发布不构成新的性能测量，也没有改变这次 run 的源码身份。

## 16K 结果与结论

本节为旧 run 的实测记录；DeepSeek / ECHO 排名不作为已成立的 baseline 结论。

正式 run `gr_serving_h200_20261002_h16k_01` 通过原始数值/计量审计，记录的测量进程历时 **31.983 分钟**（包括
生成、加载、预热、数值比较和报告生成；该耗时不是请求延迟指标，也不包含独立 profile）。输入为 **16384 + 128**，
DeepSeek ECHO / serial sparse 使用 8192 slots，prefix chunk 为 1024。统一预算仍为
HBM 4 GiB / DRAM 16 GiB；七档用户、请求上限 32、每用户最多 8 次复访、seed 42、每方案
2 条独立预热及其余数值/计时边界与 4K 相同。使用同一张 H200，依赖版本与 4K 一致。

[metadata](report/h16k/metadata.json)与[独立审计](report/h16k/audit.json)确认 56 组、
1608 条请求/数值记录，以及 1206 条非 HBM 输出对完整 HBM hidden 的逐元素完全一致
（`atol=rtol=0`）；预算与逐请求 LRU 行为通过。正式源码 SHA 为
`45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c`，
对应 153 文件的[源码 manifest](report/h16k/source_manifest.json)。4K 原始产物与身份保持不变。

### 实际访问与复访 miss

每方案实际请求和访问用户数如下；完整数字在[覆盖表](report/h16k/population_coverage.csv)。
这是热度抽样后的短 trace，不是全部配置用户各访问一次。

| 配置用户数 | 请求数 | 实际用户数 / 首访数 | 复访数 | 实际最大复访数 |
|---:|---:|---:|---:|---:|
| 1 | 9 | 1 | 8 | 8 |
| 8 | 32 | 8 | 24 | 8 |
| 32 | 32 | 21 | 11 | 2 |
| 64 | 32 | 21 | 11 | 3 |
| 128 | 32 | 28 | 4 | 2 |
| 256 | 32 | 30 | 2 | 1 |
| 512 | 32 | 29 | 3 | 1 |

下表统计每档全部 session 淘汰和其中复访 miss；被淘汰后再来的请求仍计为复访。
所有 offload 方案在各档均无淘汰、无复访 miss。

| 配置用户数 | DeepSeek HBM 淘汰 | DeepSeek HBM 复访 miss / 复访数 | NOSA HBM 淘汰 | NOSA HBM 复访 miss / 复访数 |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 / 8 | 0 | 0 / 8 |
| 8 | 0 | 0 / 24 | 2 | 1 / 24 |
| 32 | 2 | 0 / 11 | 17 | 3 / 11 |
| 64 | 2 | 0 / 11 | 17 | 3 / 11 |
| 128 | 9 | 0 / 4 | 24 | 3 / 4 |
| 256 | 11 | 0 / 2 | 25 | 2 / 2 |
| 512 | 10 | 0 / 3 | 23 | 1 / 3 |

NOSA HBM 七档合计发生 108 次 session 淘汰，63 次复访中 13 次需要重建历史；三个 offload
方案的 63 次复访全部命中。这里观察到了统一 cache 配额下的复访容量收益。DeepSeek HBM
虽有 34 次淘汰，但其 63 次复访全部命中，不能仅凭淘汰数声称已有复访性能收益。

### 全部请求与复访延迟

每格为 **均值 / p95，单位 ms**，定义与 4K 相同。

**DeepSeek checkpoint 工作负载替身**

| 用户数 | 请求范围 | HBM | ECHO | Serial sparse | Dense prefetch |
|---:|---|---:|---:|---:|---:|
| 1 | 全部 | 264.52 / 1268.06 | 348.15 / 1622.50 | 305.00 / 1449.74 | 264.46 / 1268.94 |
| 1 | 复访 | 36.51 / 37.01 | 58.62 / 59.30 | 44.84 / 44.97 | 36.19 / 36.36 |
| 8 | 全部 | 547.63 / 2081.46 | 737.54 / 2786.55 | 633.03 / 2397.41 | 550.73 / 2092.32 |
| 8 | 复访 | 36.61 / 36.91 | 59.46 / 59.96 | 45.63 / 45.84 | 37.04 / 37.41 |
| 32 | 全部 | 1378.89 / 2082.91 | 1822.00 / 2792.91 | 1588.99 / 2401.71 | 1386.75 / 2095.42 |
| 32 | 复访 | 37.34 / 37.98 | 61.54 / 67.45 | 46.43 / 47.67 | 38.16 / 39.34 |
| 64 | 全部 | 1382.22 / 2091.50 | 1772.97 / 2674.92 | 1588.71 / 2401.23 | 1386.45 / 2094.74 |
| 64 | 复访 | 37.40 / 38.27 | 60.82 / 62.11 | 45.98 / 46.59 | 37.74 / 38.78 |
| 128 | 全部 | 1834.09 / 2092.61 | 2367.84 / 2785.34 | 2107.17 / 2412.76 | 1837.08 / 2097.04 |
| 128 | 复访 | 37.52 / 38.06 | 61.42 / 62.83 | 46.43 / 47.41 | 38.68 / 40.50 |
| 256 | 全部 | 1959.13 / 2091.09 | 2519.48 / 2725.24 | 2258.49 / 2410.49 | 1966.16 / 2097.48 |
| 256 | 复访 | 37.65 / 37.70 | 61.84 / 62.75 | 46.91 / 47.56 | 39.80 / 40.26 |
| 512 | 全部 | 1889.36 / 2082.44 | 2425.90 / 2680.46 | 2185.19 / 2410.56 | 1901.52 / 2096.69 |
| 512 | 复访 | 37.17 / 37.51 | 61.53 / 62.62 | 46.44 / 47.50 | 39.22 / 40.13 |

**NOSA-8B，完整 32 层 hidden 输出**

| 用户数 | 请求范围 | HBM | Serial sparse | Dense prefetch | Overlap |
|---:|---|---:|---:|---:|---:|
| 1 | 全部 | 94.08 / 376.65 | 130.71 / 529.89 | 123.56 / 485.11 | 127.01 / 509.32 |
| 1 | 复访 | 29.96 / 31.07 | 40.00 / 40.13 | 41.41 / 41.55 | 40.15 / 40.42 |
| 8 | 全部 | 192.88 / 612.90 | 319.02 / 1200.57 | 227.35 / 783.87 | 236.45 / 823.32 |
| 8 | 复访 | 53.53 / 30.18 | 40.05 / 40.50 | 42.24 / 43.47 | 41.00 / 41.65 |
| 32 | 全部 | 463.98 / 611.36 | 716.56 / 1201.19 | 529.34 / 786.44 | 554.32 / 826.34 |
| 32 | 复访 | 187.27 / 609.43 | 40.49 / 40.87 | 42.74 / 43.13 | 41.69 / 42.81 |
| 64 | 全部 | 464.69 / 611.40 | 581.17 / 878.57 | 529.42 / 786.18 | 554.31 / 825.03 |
| 64 | 复访 | 187.88 / 609.93 | 41.15 / 41.69 | 42.98 / 44.47 | 41.73 / 42.04 |
| 128 | 全部 | 591.17 / 610.84 | 834.14 / 1207.46 | 692.86 / 792.05 | 724.60 / 827.15 |
| 128 | 复访 | 465.31 / 611.46 | 41.41 / 42.34 | 43.19 / 43.61 | 42.44 / 43.24 |
| 256 | 全部 | 610.88 / 613.79 | 832.87 / 1022.20 | 738.16 / 790.26 | 773.72 / 828.77 |
| 256 | 复访 | 611.78 / 612.37 | 42.61 / 42.71 | 43.02 / 43.22 | 42.38 / 42.40 |
| 512 | 全部 | 574.89 / 612.48 | 785.62 / 866.31 | 714.82 / 787.62 | 749.24 / 826.39 |
| 512 | 复访 | 223.27 / 552.54 | 41.72 / 42.40 | 43.14 / 43.77 | 42.13 / 42.56 |

![16K serving 全部请求与复访延迟](report/h16k/summary_readable.svg)

[PNG 预览](report/h16k/summary_readable.png)、[逐请求 SVG](report/h16k/per_request.svg)、
[逐请求 PNG](report/h16k/per_request.png)和[原始汇总 SVG](report/h16k/summary.svg)保留全部
首访和复访。原始数值为[summary.csv](report/h16k/summary.csv)、
[summary.json](report/h16k/summary.json)及[per_request.csv](report/h16k/per_request.csv)。

NOSA 的容量收益体现在复访子集：N≥8 时，HBM 复访均值为 53.53–611.78 ms，三个 offload
方案约为 40–43 ms；单用户档则 HBM 更快。**七档全部请求均值仍均为 HBM 最低**，因为
本 trace 首访占比较高，offload 首次构建历史的额外成本仍占重要部分。不能把复访收益直接
替换为这条完整请求 trace 的收益，也不能视为并发服务吞吐或真实推荐任务质量的结果。

N=8 的 NOSA HBM 24 次复访中仅 1 次 miss，其均值 53.53 ms 高于 p95 30.18 ms，
这是分位数插值与单个大值共同作用的实际统计，并非漏计 miss。N=128/256/512 只有
4/2/3 次复访，尾分位数仍是小样本描述。NOSA overlap 的复访均值没有稳定优于 serial sparse，
容量收益来自保留更多用户历史，不能归因于 overlap 本身。DeepSeek HBM 复访均值
36.51–37.65 ms；dense prefetch 为 36.19–39.80 ms，serial sparse 为 44.84–46.91 ms，
ECHO 为 58.62–61.84 ms，各方案额外算子和管理开销均包含在内；这些旧 DeepSeek 数字
不解决 MFU 与 ECHO cache 策略问题，不能用于判断 baseline 或设计优劣。

### 16K Cache 与进程分配

单位和采样边界与 4K 相同：预留用于准入，边界峰值是 cache 自有分配；CUDA allocator
统计包含全部进程存活分配与模型切换保留，不是独立模型、activation 或物理进程显存峰值。

| 模型 | 方案 | 单 session 预留 HBM / DRAM (MiB) | 准入容量 | 边界 cache 峰值 HBM / DRAM (MiB) | CUDA allocator 峰值 allocated / reserved (GiB) |
|---|---|---:|---:|---:|---:|
| DeepSeek | HBM | 205.34 / 0.00 | 19 | 3901.51 / 0.00 | 16.00 / 16.75 |
| DeepSeek | ECHO | 112.67 / 181.41 | 36 | 3380.00 / 5442.19 | 15.50 / 16.76 |
| DeepSeek | Serial sparse | 112.67 / 181.41 | 36 | 3380.00 / 5442.19 | 15.50 / 16.76 |
| DeepSeek | Dense prefetch | 60.22 / 181.41 | 68 | 1806.52 / 5442.19 | 13.95 / 16.79 |
| NOSA | HBM | 536.31 / 0.00 | 7 | 3746.98 / 0.00 | 24.48 / 25.29 |
| NOSA | Serial sparse | 70.44 / 516.00 | 31 | 1092.54 / 15480.00 | 16.51 / 25.92 |
| NOSA | Dense prefetch | 84.56 / 516.00 | 31 | 1545.97 / 15480.00 | 16.95 / 26.04 |
| NOSA | Overlap | 70.44 / 516.00 | 31 | 1092.54 / 15480.00 | 16.51 / 26.04 |

精确字节数据见[cache_and_memory.csv](report/h16k/cache_and_memory.csv)。NOSA HBM 的
准入容量为 7，三个 offload 方案均为 31，后者受 DRAM 预算约束；本 trace 最多实际访问
30 个不同用户，因而 offload 都能保留这些 session。DeepSeek HBM 容量为 19，ECHO / serial
为 36，dense prefetch 为 68。模型权重载荷与 4K 相同；普通 activation 仍无独立峰值测量。

### 16K NOSA layer 31 内部 overlap

独立 run `gr_serving_h200_20261002_h16k_profile_01` 使用 8 用户 trace 的 request ID 2/3/7，
只观察 layer 31；每模式每请求 1 个样本、1 次未插桩对照，profile warmup 为 0，各自从独立
空 cache 构建 sparse prefix。六条记录的 18 项完整 `[128,4096]` hidden 比较均完全一致，
实际选择、唯一读取、page/stripe 对应关系及来源身份通过 CPU 审计。

ratio 定义与 4K 相同，math 覆盖仅为 softmax；各页 envelope 由非空 stripe 的 min/max
确定。三个样本的全局 envelope 和非空 stripe 并集恰好相同，但仍分别验证和报告。

| request ID | 唯一历史 K+V 字节 | Pages / 非空 stripes | Page-envelope ratio | 非空 stripe-copy ratio | 两者均 ≥90% |
|---:|---:|---:|---:|---:|---|
| 2 | 6,881,280 | 210 / 1680 | 78.37% | 78.37% | 否 |
| 3 | 7,176,192 | 219 / 1752 | 79.83% | 79.83% | 否 |
| 7 | 7,274,496 | 222 / 1776 | 81.21% | 81.21% | 否 |

三个样本均未达到 90% 门槛，仍是有效测量；serial 的未测 math overlap 保留 null。
结果仅支持该层、该输入、该插桩定义下的机制观察，不替代未插桩完整请求延迟。
每样本额外 trace HBM 为 212,992 B，此诊断没有多用户 LRU 准入，不用于正式预算占用结论。
完整记录见[profile_analysis.json](report/h16k/profile_analysis.json)与
[profile_metadata.json](report/h16k/profile_metadata.json)。profile 独立源码 SHA 为
`cd72caf5edbac19a54d162e26d3339b05dd0671048162d970729669af5d0c48b`；采集来源集合与正式
测量不同，但对正式源码集合的验证无任何文件漂移，`allowed_diagnostic_source_changes=[]`。

### 16K 复现与素材来源

正式运行命令：

```bash
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving \
  bash experiments/gr_serving/scripts/run.sh gr_serving_h200_20261002_h16k_01 \
  --users 1 8 32 64 128 256 512 --history-tokens 16384 --candidate-tokens 128 \
  --requests 32 --max-revisits 8 --deepseek-slots 8192 \
  --hbm-budget-gib 4 --dram-budget-gib 16
```

独立 profile 须在正式 GPU 测量结束后运行：

```bash
bash experiments/gr_serving/scripts/profile.sh <NEW_PROFILE_RUN_ID> \
  --latency-data experiments/gr_serving/output/data/gr_serving_h200_20261002_h16k_01 \
  --num-users 8
```

原始有效产物保留在 `output/data/gr_serving_h200_20261002_h16k_01/` 和独立 profile run
目录。`report/h16k/` 中原始数据/图与 audit 的 SHA-256 对应，
[provenance.json](report/h16k/provenance.json)记录复制、派生表、渲染器和 profile 的完整来源。
原始 SVG 已精确渲染；十四个逐请求面板和四个可读汇总面板均已人工检查。
可读图复用原 4K 渲染器，在本 run 的忽略目录保存同一份源码：

```bash
.venv/bin/python experiments/gr_serving/output/data/gr_serving_h200_20261002_h16k_01/render_report.py \
  --summary experiments/gr_serving/report/h16k/summary.json \
  --metadata experiments/gr_serving/report/h16k/metadata.json \
  --output-dir experiments/gr_serving/report/h16k
```

渲染器 SHA-256 仍为 `71704ada5c64e417f4af703f21043eca5d3232bb5e26934cc6dc0ef93bec166d`；
只改图表布局，不重新测量或改变已有统计。两张派生 CSV 沿用 4K 的字段选择和最大值规则。

## 64K 结果与结论

本节仍为原六请求轨迹的实测记录，没有被失败的 sequential run 替换；DeepSeek / ECHO
对照需在 MFU 与 cache 策略审计修正后复测，旧排名不作为 baseline 结论。

正式 run `gr_serving_h200_20261002_h64k_01` 通过原始数值/计量审计，记录的测量进程历时 **32.797 分钟**，
包含生成、加载、预热、数值比较和报告生成，不是请求延迟，也不包含独立 profile。
输入为 **65536 + 128**，DeepSeek ECHO / serial sparse 使用 32768 slots，prefix chunk
为 1024。使用同一 H200 与前述依赖，HBM 4 GiB / DRAM 16 GiB 配额、seed 42、每用户最多
8 次复访及每方案 2 条独立预热保持一致。为控制时长，**每档请求上限改为 6**，显式允许
没有复访的 trace，不补造访问；56 组共 336 个测量请求。

[metadata](report/h64k/metadata.json)和[独立审计](report/h64k/audit.json)确认完整矩阵、
336 条请求/数值记录，以及 252 条非 HBM 完整 candidate hidden 对 HBM 参考的逐元素完全
一致（`atol=rtol=0`）；预算与逐请求 LRU 行为通过。正式源码与 16K 相同：
`45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c`，
对应[源码 manifest](report/h64k/source_manifest.json)。

NOSA 这里显式使用 `context_limit=65664` 与 backend `max_seq_len=65664`，超过其默认
32768-token 文本生成边界。验收证明此执行配置下 resident/offload 数值一致，不证明
checkpoint 的长上下文推荐质量；该 override 及原默认值保留在 workload/provenance 中。

### 64K 实际覆盖与缺失复访

| 配置用户数 | 请求数 | 实际用户数 / 首访数 | 复访数 | 实际最大复访数 |
|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 5 | 5 |
| 8 | 6 | 4 | 2 | 2 |
| 32 | 6 | 6 | 0 | 0 |
| 64 | 6 | 5 | 1 | 1 |
| 128 | 6 | 6 | 0 | 0 |
| 256 | 6 | 6 | 0 | 0 |
| 512 | 6 | 6 | 0 | 0 |

完整覆盖表见[population_coverage.csv](report/h64k/population_coverage.csv)。32/128/256/512
用户档没有复访，其复访 mean/median/p95/p99 均为 **null**，表中显示“— (n=0)”；不能用
零延迟或相邻档插值替代。其余三档也只有 5/2/1 次复访，尤其单样本 p95 等于该样本自身，
不代表稳定尾延迟。配置为 512 用户的总体在本次窗口中实际只访问了 6 个不同用户。

### 64K 全部请求与复访延迟

每格为 **均值 / p95，单位 ms**，缺失统计明确保留。

**DeepSeek checkpoint 工作负载替身**

| 用户数 | 请求范围 | HBM | ECHO | Serial sparse | Dense prefetch |
|---:|---|---:|---:|---:|---:|
| 1 | 全部 | 1531.41 / 6745.39 | 1948.81 / 8525.02 | 1670.00 / 7332.53 | 1532.39 / 6746.64 |
| 1 | 复访 | 41.81 / 43.00 | 69.96 / 70.83 | 52.16 / 52.46 | 42.61 / 42.65 |
| 8 | 全部 | 6020.75 / 9013.37 | 7828.65 / 11828.79 | 6500.93 / 9726.77 | 6001.79 / 8984.26 |
| 8 | 复访 | 41.66 / 41.71 | 70.76 / 70.77 | 52.23 / 52.33 | 42.73 / 42.75 |
| 32 | 全部 | 9015.05 / 9016.60 | 11542.57 / 11832.66 | 9683.41 / 9711.17 | 8980.07 / 8981.39 |
| 32 | 复访 | — (n=0) | — (n=0) | — (n=0) | — (n=0) |
| 64 | 全部 | 7518.98 / 9015.74 | 9510.42 / 11411.85 | 8069.78 / 9679.72 | 7490.16 / 8983.00 |
| 64 | 复访 | 41.89 / 41.89 | 69.85 / 69.85 | 52.61 / 52.61 | 43.46 / 43.46 |
| 128 | 全部 | 9015.45 / 9016.89 | 11388.76 / 11395.32 | 9675.85 / 9686.03 | 8977.52 / 8978.91 |
| 128 | 复访 | — (n=0) | — (n=0) | — (n=0) | — (n=0) |
| 256 | 全部 | 9015.20 / 9016.82 | 11394.25 / 11402.95 | 9690.90 / 9706.35 | 8978.64 / 8980.05 |
| 256 | 复访 | — (n=0) | — (n=0) | — (n=0) | — (n=0) |
| 512 | 全部 | 9014.59 / 9016.44 | 11483.56 / 11573.34 | 9669.45 / 9673.04 | 8981.26 / 8982.72 |
| 512 | 复访 | — (n=0) | — (n=0) | — (n=0) | — (n=0) |

**NOSA-8B，完整 32 层 hidden 输出**

| 用户数 | 请求范围 | HBM | Serial sparse | Dense prefetch | Overlap |
|---:|---|---:|---:|---:|---:|
| 1 | 全部 | 419.27 / 1782.26 | 619.08 / 2636.40 | 546.80 / 2289.76 | 584.40 / 2471.66 |
| 1 | 复访 | 29.85 / 29.92 | 42.72 / 42.96 | 48.85 / 49.20 | 45.22 / 45.68 |
| 8 | 全部 | 1597.41 / 2385.03 | 3033.73 / 4864.40 | 2039.72 / 3039.58 | 2206.35 / 3293.31 |
| 8 | 复访 | 30.32 / 30.46 | 42.94 / 42.97 | 48.86 / 48.93 | 45.65 / 45.65 |
| 32 | 全部 | 2385.96 / 2390.58 | 3966.58 / 4870.78 | 3038.73 / 3044.02 | 3299.24 / 3303.07 |
| 32 | 复访 | — (n=0) | — (n=0) | — (n=0) | — (n=0) |
| 64 | 全部 | 1986.03 / 2385.90 | 2938.88 / 3527.79 | 2544.63 / 3055.82 | 2765.40 / 3357.89 |
| 64 | 复访 | 30.39 / 30.39 | 44.37 / 44.37 | 49.25 / 49.25 | 45.98 / 45.98 |
| 128 | 全部 | 2390.79 / 2396.45 | 3519.01 / 3523.76 | 3038.28 / 3043.05 | 3283.68 / 3291.55 |
| 128 | 复访 | — (n=0) | — (n=0) | — (n=0) | — (n=0) |
| 256 | 全部 | 2391.58 / 2393.14 | 3517.26 / 3522.49 | 3038.81 / 3043.68 | 3284.89 / 3289.31 |
| 256 | 复访 | — (n=0) | — (n=0) | — (n=0) | — (n=0) |
| 512 | 全部 | 2392.95 / 2396.88 | 3516.99 / 3523.44 | 3038.39 / 3043.23 | 3282.61 / 3288.41 |
| 512 | 复访 | — (n=0) | — (n=0) | — (n=0) | — (n=0) |

![64K serving：显式保留无复访组](report/h64k/summary_readable.svg)

[PNG 预览](report/h64k/summary_readable.png)中的灰色列表示没有观测到复访，曲线在这里
断开；下方 n 为复访样本数。[逐请求可读 SVG](report/h64k/per_request_readable.svg)与
[PNG](report/h64k/per_request_readable.png)按真实 request ID 0–5 标注，并保留每个首访/
复访点。原始分析文件仍保存为[summary.svg](report/h64k/summary.svg)和
[per_request.svg](report/h64k/per_request.svg)。完整统计与数据为
[summary.csv](report/h64k/summary.csv)、[summary.json](report/h64k/summary.json)、
[per_request.csv](report/h64k/per_request.csv)。

每方案七档总计只有 8 次复访，全部命中。DeepSeek HBM 复访均值为 41.66–41.89 ms，
dense prefetch 为 42.61–43.46 ms、serial sparse 为 52.16–52.61 ms、ECHO 为
69.85–70.76 ms。NOSA HBM 为 29.85–30.39 ms、serial sparse 为 42.72–44.37 ms、
overlap 为 45.22–45.98 ms、dense prefetch 为 48.85–49.25 ms；这是三个非空复访档的
描述统计。DeepSeek 数字不用于当前 baseline 排名。此短 trace 没有因历史被淘汰而发生的
复访重建，不能据此声称 offload 容量收益。

### 64K Cache 与进程分配

预留、边界分配及进程 allocator 的定义与前两档一致；普通 activation 没有独立峰值，
模型权重载荷与前述相同，allocator 统计仍包含模型切换期间的存活分配。

| 模型 | 方案 | 单 session 预留 HBM / DRAM (MiB) | 准入容量 | 边界 cache 峰值 HBM / DRAM (MiB) | CUDA allocator 峰值 allocated / reserved (GiB) |
|---|---|---:|---:|---:|---:|
| DeepSeek | HBM | 816.59 / 0.00 | 5 | 4082.96 / 0.00 | 24.58 / 39.24 |
| DeepSeek | ECHO | 450.17 / 721.41 | 9 | 2701.00 / 4328.44 | 23.23 / 39.24 |
| DeepSeek | Serial sparse | 450.17 / 721.41 | 9 | 2701.00 / 4328.44 | 23.23 / 39.24 |
| DeepSeek | Dense prefetch | 239.47 / 721.41 | 17 | 1436.80 / 4328.44 | 22.00 / 39.26 |
| NOSA | HBM | 2132.77 / 0.00 | 1 | 2128.75 / 0.00 | 24.50 / 26.77 |
| NOSA | Serial sparse | 178.93 / 2052.00 | 7 | 851.44 / 12312.00 | 16.33 / 26.80 |
| NOSA | Dense prefetch | 241.02 / 2052.00 | 7 | 1230.01 / 12312.00 | 16.69 / 26.80 |
| NOSA | Overlap | 178.93 / 2052.00 | 7 | 851.44 / 12312.00 | 16.33 / 26.80 |

精确字节数据见[cache_and_memory.csv](report/h64k/cache_and_memory.csv)。DeepSeek HBM
准入容量为 5，七档合计 4 次淘汰；NOSA HBM 容量为 1，七档合计 27 次淘汰。但所有发生
淘汰的历史都没有在淘汰后再被访问，因此两者复访 miss 均为零。所有 offload 方案无淘汰，
其容量为 DeepSeek 9/9/17（ECHO/serial/dense）、NOSA 三种方案均为 7，后者受 DRAM
预算约束。容量数值不同不等于已经观察到请求延迟上的容量收益。

### 64K NOSA layer 31 内部 overlap

独立 run `gr_serving_h200_20261002_h64k_profile_01` 使用**单用户** trace 的前三条复访
（request ID 1/2/3），只观察 layer 31。8 用户 trace 仅有两条复访，因此没有补造第三条。
每模式每请求 1 个样本、1 次未插桩对照，profile warmup 为 0，各自从独立空 cache 构建
sparse prefix。六条记录的 18 项完整 `[128,4096]` hidden 比较全部一致，唯一读取和
page/stripe 对应关系、workload/GPU/源码身份通过独立 CPU 验证。

| request ID | 唯一历史 K+V 字节 | Pages / 非空 stripes | Page-envelope ratio | 非空 stripe-copy ratio | 两者均 ≥90% |
|---:|---:|---:|---:|---:|---|
| 1 | 7,536,640 | 230 / 1840 | 76.87% | 76.87% | 否 |
| 2 | 7,274,496 | 222 / 1776 | 73.18% | 73.18% | 否 |
| 3 | 7,536,640 | 230 / 1840 | 79.27% | 79.27% | 否 |

三个样本均未达到 90% 门槛，仍保留为有效结果。定义与前两档相同，math 只覆盖 softmax；
每页 envelope 由其非空 stripes 的 min/max 定义。本次全局 page 与 stripe 并集相等，
envelope-only union 均为 0，但两个指标独立核验。serial 未测 math overlap 仍为 null。
profile 每样本额外 trace HBM 为 655,360 B，没有多用户 LRU 准入，不能用于正式 cache
占用结论，也不以插桩时长替代完整 serving 延迟。

完整定义/记录为[profile_analysis.json](report/h64k/profile_analysis.json)及
[profile_metadata.json](report/h64k/profile_metadata.json)。独立 profile 源码 SHA 为
`cd72caf5edbac19a54d162e26d3339b05dd0671048162d970729669af5d0c48b`；正式源码集合核验
无漂移，`allowed_diagnostic_source_changes=[]`。

### 64K 复现与素材来源

正式运行命令：

```bash
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving \
  bash experiments/gr_serving/scripts/run.sh gr_serving_h200_20261002_h64k_01 \
  --users 1 8 32 64 128 256 512 --history-tokens 65536 --candidate-tokens 128 \
  --requests 6 --max-revisits 8 --allow-empty-revisits --deepseek-slots 32768 \
  --hbm-budget-gib 4 --dram-budget-gib 16
```

独立 profile 须在正式 GPU 测量结束后运行：

```bash
bash experiments/gr_serving/scripts/profile.sh <NEW_PROFILE_RUN_ID> \
  --latency-data experiments/gr_serving/output/data/gr_serving_h200_20261002_h64k_01 \
  --num-users 1
```

原始产物保留在 `output/data/gr_serving_h200_20261002_h64k_01/` 与独立 profile run 目录。
`report/h64k/` 的复制文件对照 audit SHA-256，完整来源、派生字段与 override 在
[provenance.json](report/h64k/provenance.json)。原始图逐字节保留；可读汇总图明确显示空
复访组，可读逐请求图修正了原 SVG 对短 trace 的小数刻度取整显示，不改测量数据。
所有汇总和逐请求面板均已渲染检查。

64K 使用独立的报告渲染器版本，保留 4K/16K 渲染器及素材不变：

```bash
.venv/bin/python experiments/gr_serving/output/data/gr_serving_h200_20261002_h64k_01/render_report.py \
  --summary experiments/gr_serving/report/h64k/summary.json \
  --metadata experiments/gr_serving/report/h64k/metadata.json \
  --output-dir experiments/gr_serving/report/h64k
```

该渲染器 SHA-256 为 `eeec4852dbcc439b930caeb9c6ab254653c160a563f593ed7ebdfb33af2b2c2f`，
同时读取同目录下的 `per_request.csv`。它保留 null、给空组留白，使用真实请求编号，
不会把图表重建当作新的性能运行。两张派生 CSV 沿用前述字段选择和最大值规则。

CPU 验证入口：

```bash
.venv/bin/python -m pytest experiments/gr_serving/tests -q
```
