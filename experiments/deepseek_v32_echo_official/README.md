# DeepSeek V3.2 官方 ECHO 适配实验


本实验验证官方 ECHO 适配路径的性能，沿用 [motivation](../deepseek_v32_motivation/README.md)
的模型、输入及固定 P/NH 容量，比较官方 ECHO offload 与同一实验重新执行的
HBM-only。结果覆盖官方算子接入本地单卡固定历史 GR 生命周期后的路径，不代表
上游完整 TP=8 AWQ SGLang serving。

独立数值验收为 `refactor_final_official_check_20261005_01`，正式计时为 `refactor_final_official_bench_20261005_01`，
诊断为 `refactor_final_official_profile_20261005_01`。正式运行与四段 profile 均已通过各自验收。

## 结果

| 方案 | 访问 | 请求数 | history hit | 均值 ms | 中位数 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|
| HBM-only | 首次 | 16 | 0/16 | 2126.499 | 2126.137 | 2132.862 |
| HBM-only | 复访 | 16 | 0/16 | 2123.192 | 2124.119 | 2126.623 |
| 官方 ECHO 适配路径 | 首次 | 16 | 0/16 | 4596.177 | 4593.201 | 4609.691 |
| 官方 ECHO 适配路径 | 复访 | 16 | 16/16 | 44.862 | 44.516 | 46.391 |

HBM/ECHO 的 32 请求总耗时分别为 67.995 / 74.257 s。
ECHO 的 16 次复访全部命中历史；HBM-only 的复访均重建历史。被淘汰后的访问仍算复访。复访请求
时间比包含 history 重建差异，不能解释为 attention 算子加速。单条正式轨迹的尾
分位数只描述本轮样本，不能证明差异稳定。

详细结果、逐请求数据、配置、数值摘要和来源见 [results.md](report/results.md)、[逐请求数据](report/per_request.csv)和[summary.json](report/summary.json)。

## 与本地实现及旧结果对照

本地 motivation 和官方实验分别保留自己的 HBM 基线。下表引用两边的独立正式
bench；输入、请求顺序、模型、P/NH、计算图及实际执行环境须经对照工具核验。

| 实现 | 首访均值 ms | 复访均值 ms | 复访 p95 ms | 32 请求总耗时 s |
|---|---:|---:|---:|---:|
| HBM-only（本地对照） | 2192.406 | 2187.398 | 2191.141 | 70.077 |
| 本地 ECHO | 2294.994 | 24.530 | 25.607 | 37.112 |
| HBM-only（官方对照） | 2126.499 | 2123.192 | 2126.623 | 67.995 |
| 官方 ECHO 适配路径 | 4596.177 | 44.862 | 46.391 | 74.257 |

本地 run ID 为 `refactor_final_deepseek_bench_20261005_01`，官方 run ID 为
`refactor_final_official_bench_20261005_01`。以官方 ECHO 为分母，本地 ECHO 的首访、复访与完整轨迹耗时分别低 50.067%、45.322%、50.021%；这是独立运行间的观测差异，未证明稳定性。
本地 HBM 使用 mainline DeepGEMM resident logits，本地 ECHO 使用项目融合
indexer/prefetch，两者均使用 FlashInfer top-k。官方组使用 ECHO 原始
resident/fused logits 和 top-k。选择路径与 HBM 基线有差异，观测差额不能全部
归因于缓存或搬运。两边各自在独立 check 内验收，不声称已完成新的跨运行数值比较。

旧发布 `echo_official_c10_20261004_u16_r2_01` 使用合并验收与计时流程，作为历史
对照单独列在 [旧/新选定对照](report/old_new/results.md)。旧运行的物理 GPU 与当前不同，计时契约也不同；所列差值仅作描述。旧完整产物清理后，选定样本与身份不能支持完整历史源码、native 或数值重审。
本地实现的旧发布/P0/新实现三版本对照见 motivation；不得把本地 P0 当作官方路径
的 P0，也不得由单条官方新轨迹声称稳定回归或优化收益。

完整来源、未四舍五入数值及报告生成器身份见 [实现对照](report/existing_implementation_comparison.md)与[JSON](report/existing_implementation_comparison.json)。

## 设置与测量边界

H=65,536、A=128、history chunk=1,024，16 用户顺序访问两轮，seed=42。
P=65,536 个逐层 history HBM 槽，NH=16,777,216 个全局 host token。
P/NH 是容量配额，不另设 cache 字节子预算，也不扣除经验性 headroom。

模型将 `/preset-models` checkpoint 前三层独立复制为十个 dense block，共
7,827,793,408 参数，含 embedding、final norm 和 LM head。每个副本重放对应
source block 的 hidden/residual 输入，权重、KV 和 indexer 独立。该工作负载
替身不代表经过训练的十层模型、完整 DeepSeek V3.2 或真实 GR 任务质量。

普通 linear 使用 FP8，主 KV 为 BF16 的 576 元素 record，indexer K/scales
resident，RoPE 后不加 Hadamard。每请求计算全部 candidate hidden `[128, 7168]`
和末 token logits `[1, 129280]`；candidate 在 GPU 临时执行后丢弃，不写入 DRAM。

两方案均使用官方原始 top-k：HBM-only 调用官方 resident logits，ECHO 调用官方
融合 indexer/prefetch、allocator、精确 recall 和释放 helper。offload 每次
indexer 调用均执行官方融合 kernel，策略为 `official_fused_every_offload_call_v1`。
模型投影与 FlashMLA 沿用 motivation。当前实验未单独测量内部 overlap。

两方案各预热两位用户首访及第一位用户复访，ECHO 复访预热须实际读取 host KV。
释放预热资源后，从空缓存执行正式轨迹。check 保存两方案全部输出及独立 HBM
重跑，签发数值 receipt；bench 匹配 receipt 后计时，不在样本间保存或比较完整输出。

同步墙钟包含输入验证与搬入、准入/淘汰、miss 时 history 构建、candidate 和清理。
forward 内原有计数归约计时；加载、编译、计算图准备、预热、计数 host 读取和
数值保存/比较不计时。传输计数只覆盖 candidate forward。

## 计算图、精度与来源

十个独立层在 Q=1,024 和 Q=128 两种形状下分别使用 projection/finish 图，
共 40 张。正式轨迹与独立 check 的重放覆盖、fallback 和图内精度见
[summary.json](report/summary.json) 的 `audit.compute_graphs` 与逐方案 `backend.compute_graphs`。图 bank 静态 storage、实际 allocator 占用与规划
reservation 上限分别记录，不能将规划值当成已分配 storage。

实际精度与执行环境为 FP8 普通 linear、BF16 主 KV，TF32 关闭，intraop=8、interop=96；硬件、GPU UUID
和 CPU/NUMA 绑定为 H200 / SM90 / 132 SM，GPU UUID `80ff95c3-176e-fd8a-728f-9c5577c4a779`，CPU24–31、NUMA0。官方 ECHO 固定提交为
`bc1b75c1000010d0ac6f032ebaac283255c050b1`。依赖、源码、native/JIT 产物及其身份
随运行保存。checkpoint 身份基于路径、shard 大小和 mtime，未 hash 全部权重。

独立 check 的完整源码快照为
`a9d45df3f2ccca5b3ed2a5e36bbb6c0d54a768c87fe7d1863555398ff76c47a1`，共 1,303 个文件。
receipt 文件 SHA-256 为
`665f934d09aee400c452aba3af10e9becf4bebeafc3591d79e20973ea10b39d5`；其独立的内部
canonical signature 为 `1ac1fc23f0a3fea378e610fc9bf555d7c10d7fb27720799163d847337e66b0b4`。
本轮 bench/profile 均为同一 `a9d45df3f2ccca5b3ed2a5e36bbb6c0d54a768c87fe7d1863555398ff76c47a1` 完整源码快照；工作负载 SHA256 为 `7e4c737a86464c12238191933e707344426231bf671a29658837e676ff5284ae`。
receipt 的执行源码集合不包含仅用于 profile/报告的辅助文件；bench 仍逐项核验
输入、模型、native、backend/cache/graph、精度和执行环境。Nsight 注入库及完整
诊断环境单独记录，不能把 profiler 环境差异隐藏成相同运行。

运行窗口的全部 GPU 进程观测与退出记录见 [运行验收](report/profile/run_acceptance.json)。
观测次数和最大间隔分别为 bench 58 次、profile 75 次、bench 28.315 秒、profile 29.443 秒；
离散观测不能排除采样间隙中的工作。

## 内存与搬运

| 方案 | allocated 峰值 GiB | reserved 峰值 GiB | 设备已用量采样最大 GiB | cache HBM 最大 GiB | cache DRAM 最大 GiB |
|---|---:|---:|---:|---:|---:|
| HBM-only | 14.549 | 23.875 | 24.616 | 6.625 | 0.000 |
| 官方 ECHO | 16.410 | 24.521 | 25.892 | 8.486 | 320.001 |

PyTorch 峰值在各方案释放预热资源后重置，包含模型和正式执行。reserved 包含
allocator 缓存，不能与 allocated 相加。cache 在请求边界计量，包含 shared
与全部 session；计算图相关占用不能重复加算。设备已用量由 total−free 计算，
仅为边界采样最大值，不是连续峰值。

NH 对应 180 GiB 逻辑主 KV 容量；包含实际 pinned backing 与元数据的 DRAM cache 计费最大值为 320.001038 GiB。轨迹仅保留 16 个 history，未填满 NH；相同 P/NH
不等于相同总 HBM/DRAM 字节数。pinned allocator 的 active-byte 统计可能受缓存块
复用影响，不能单独用于推断物理 backing。

candidate 的首访/复访 H2D 分别为 0 /
1,246,371,840 B，D2H 为 0 B。
上述 H2D 是各 16 条请求的合计。这些计数不包含 history 构建阶段。

## 独立数值验收与诊断

数值 check 保存 HBM、独立 HBM 重跑和官方 ECHO 共 96 份输出。按预先固定的门槛
核验 finite、shape/dtype、hidden/logits relative L2（分别不超过 0.005/0.01），
并要求至少 99.9% 元素满足 `|actual-reference| <= 1/32 + (1/64)*|reference|`。
门槛来自独立 resident 校准，不随本轮输出调整。当前 check 的 64 条 HBM/ECHO
请求、32 条独立 HBM 重跑及 6 条预热均通过验收，2499 项带签名证据已另行复核。
ECHO 预热确认经过 host recall。check 轨迹中的 HBM/ECHO 重放次数为
41,600/21,120，独立 HBM 重跑为 41,600；未保存预热重放明细，因此不把轨迹
覆盖证明扩大到预热的每次重放。

| 比较 | 输出 | 最大 relative L2 | 最低元素通过比例 | 最大绝对误差 |
|---|---|---:|---:|---:|
| HBM 重跑 / HBM | hidden | 0.001227370399 | 100.00000000% | 0.046875 |
| HBM 重跑 / HBM | logits | 0.004026068375 | 99.99845297% | 0.0625 |
| 官方 ECHO / HBM | hidden | 0.002528261085 | 99.99291556% | 0.1796875 |
| 官方 ECHO / HBM | logits | 0.002501402473 | 100.00000000% | 0.0625 |

HBM 重跑的 hidden/logits 分别有 32/31 条满足逐元素 allclose，ECHO 的
hidden/logits 分别为 31/32 条；四组均没有逐位相等的请求。全部请求通过的是
上述预先固定的数值门槛，不能写成与本地 motivation 相同的逐位一致结论。
完整数值证据在
`docs/agents/acceptance/unified_runtime_20261005/official_final/driver/check_summary.json`，
数值摘要见 [numerical_summary.json](report/numerical_summary.json)。
官方 top-k 的顺序及并列值可能变化，不能把全部差异归为已证实的舍入误差。
独立 HBM 重跑只用于数值验证，不发布延迟。模型及算子正确性范围见[模型说明](../../models/deepseek_v32/README.md)；正确性检查不代替完整 serving 计时。

check 的 driver 以退出码 0 完成；71 次全 GPU 观测覆盖 387.035 秒，最大间隔
32.855 秒，所选 GPU 未观察到外来进程。其他设备上保留了获准并行的 NOSA
正确性检查记录。该监测只支持离散采样下所选 GPU 的归属核验，不是未来正式
bench/profile 的独占窗口证明。

profile 在两方案各采集首访和完整首轮之后的第一次复访，共四次 capture。
数值比较在 capture 之外；其他请求用于恢复缓存状态。34 条请求通过固定数值门槛；wrapper 独立核验 2,401 项证据，四段共 207,988 个 kernel 的计数与时长守恒。
分析核对 kernel 数量与时长，按实际 launch 关联到最内层 host scope，保留无法
归属的 graph 活动，不把阶段时长解释为独立算子时间、MFU 或正式请求延迟。

| 方案 / 访问 | kernel 数量 | kernel duration sum ms | 未归属活动数量 | 未归属活动 duration sum ms |
|---|---:|---:|---:|---:|
| HBM / 首访 | 42019 | 2057.074 | 0 | 0.000 |
| HBM / 复访 | 42019 | 2063.195 | 0 | 0.000 |
| ECHO / 首访 | 122290 | 4139.862 | 0 | 0.000 |
| ECHO / 复访 | 1660 | 34.460 | 0 | 0.000 |

duration sum 可能包含重叠区间；未归属活动可能包含 kernel 及 copy，不能当作纯
kernel 子集相减。mapped-host 访问流量使用验收后的 cache 计数，不由 memcpy
活动推算。本次四段 capture 均无未归属活动；HBM 复访仍执行 history 构建。证据见 [analysis.json](report/profile/analysis.json)、[stages.csv](report/profile/stages.csv)与[kernels.csv](report/profile/kernels.csv)。

## 复现与模块

按[第三方依赖说明](../../3rdparty/README.md)准备基础环境及固定提交的只读 ECHO
checkout，从仓库根目录使用新 run ID 运行。先设置 motivation 运行方式中列出的
环境，再执行以下入口。完整环境和 placement 仍以实际运行记录为准。

```bash
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=24-31 --membind=0 \
  .venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.measure \
  --mode check --run-id official_check_new --compute-graphs \
  --output-dir /tmp/cxldsagr-checks/deepseek_v32_echo_official/official_check_new

CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=24-31 --membind=0 \
  .venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.measure \
  --mode bench --run-id official_bench_new --compute-graphs \
  --validation-receipt /tmp/cxldsagr-checks/deepseek_v32_echo_official/official_check_new/receipt.json

CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=24-31 --membind=0 \
  env PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  bash experiments/deepseek_v32_echo_official/scripts/profile.sh \
  --run-id official_profile_new \
  --reference-run experiments/deepseek_v32_echo_official/output/data/official_bench_new
```

模型装配位于 `models/deepseek_v32/execution/official.py`，cache 适配位于
`models/deepseek_v32/cache/official.py`，官方绑定位于
`operators/deepseek_v32/indexer/official.py`。measure 复用 motivation 的配置、
预热和请求检查，通过共享 `GR/` 与 `serving/persistent.py` 执行。

check 证据位于 `docs/agents/acceptance/unified_runtime_20261005/official_final`，bench 数据与源码位于
`output/data/<run_id>/`，日志位于 `output/log/<run_id>/`，原始 Nsight 文件位于
`output/profile/<profile_run_id>/`。report 或 profile 的 `--receipt-override`
允许证据重定位，前提是原 receipt 字节和全部文件哈希不变；不改写原 bench metadata。
正式报告由 `src.report` 生成，诊断由 `src.profile_report` 生成。本地/官方对照
使用发布时已安装并记录身份的 `src.compare_existing`，完整来源见
[发布清单](report/publication_manifest.json)。

测量时源码、原分析生成器与发布时辅助文件分别绑定，见 [source_bindings.json](report/source_bindings.json)和[helper 清单](report/report_helper_sources.json)。发布时的辅助快照只记录当时磁盘上的已加载模块；它不替代原测量或早先分析的源码身份。
