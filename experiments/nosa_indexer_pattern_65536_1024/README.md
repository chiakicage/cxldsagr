# NOSA indexer pattern：GR 65536 + 1024

本实验属于 sparse pattern 设计探索，用于理解实际访问集合并指导后续设计，目前不在论文
性能主线。2026-10-05 仅整理代码和范围，没有重新采集或生成实验结果。保留完整 NOSA
三组、有效 QA32/QA64 选择与并集分解/分布；退出依赖假定 MFU 的时间、理想 overlap
和阈值策略支线。新三组采集入口把 observer 重放改为独立 `--mode check`，改动后未运行。
下方选择数字仍来自原 run ID 和源码快照。

## 实验目的与测量边界

测量 NOSA-8B 在同一 GR 请求上的稀疏选块并集：每个 candidate query、每层、每 KV head
先选块，再沿 1024 个 candidate queries 去重，统计 unique K+V payload。
三组分别为 dense QA-only、dense full NOSA 与实际 sparse full NOSA。
输入为 synthetic GR 内容，user 623、visit 0、seed 42：instruction 28 + history 65508
组成 65536-token prefix，candidate suffix 为 1024 tokens，总计 66560。
新采集、保留 QA32/64 与 dense 测量的请求字节相同，SHA256 为
`0bbabf07bc72f9804dbc2e9644cf708e21eafe237631f4a56c64b20f9cda7f2f`。

容量只计每个 `(layer, KV head, logical block)` 一次的 K+V；不同层/head 的同号块分别计费。
不计 CIS、indexer 读取、KV 写入、元数据、重复 fetch 或缓存命中。
本实验不测模型质量、sparse 延迟、吞吐、总线流量或 offloading；所有模型采集不执行
LM head 或自回归生成。上下文只在实验进程中从 32768 覆盖为 66560，保留 LongRoPE factors。

## 配置、硬件与来源

NOSA-8B：32 layers、32 Q heads / 2 KV heads、D128、BF16、64-token block；prefix 按
1024 tokens 分块，candidate 一次执行全部 1024 tokens。每个 head 完整 KV 为
1040 blocks / 32.5 MiB，每层 65 MiB，模型合计 2080 MiB。每块 K+V 为 32768 B。

新三组采集与新 dense 测量使用物理 GPU 5、进程内 `cuda:0`。PyTorch 名称为
`NVIDIA H200`，NVML 为 `NVIDIA M403`，UUID 均为 `a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`；
SM90、132 SM、驱动 570.124.06。CPU 48–55、内存 NUMA 1，OMP/MKL/OpenBLAS 与
PyTorch intra-op 均为 8。PyTorch 2.12.1+cu130 / CUDA build 13.0、Triton 3.7.1、
FlashInfer 0.6.18、NumPy 2.5.0、Matplotlib 3.11.1、safetensors 0.8.0、tokenizers 0.23.2。
checkpoint/tokenizer 为 `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`，完整身份和源码哈希见 metadata。

| 内容 | Run ID | 性质 |
| --- | --- | --- |
| 完整 NOSA 三组 | `nosa_pattern_flags_20261004_01` | 2026-10-04 GPU 采集 |
| QA32 | `qa32_h200_gpu1_20260928_01` | 保留有效 GPU 采集 |
| QA64 原始选择 | `query_aware_fp32_65536_1024_20260925_01` | 保留有效 GPU 采集 |
| QA64 图表与汇总 | `qa64_reanalysis_20260928_01` | 保留数组的 CPU 分析 |
| QA64 环境对照 | `qa64_environment_compare_flags_20261004_01` | 新 dense QA64 与保留数组逐元素比较 |
| 32 / 64 分解 | `components32_pcie50_20260928_01 / components64_pcie50_20260928_01` | 保留有效 CPU 分析 |
| 32 / 64 分布 | `distribution32_64_20260928_01` | 保留有效 CPU 分析 |

原三组运行无计时预热，每条轨迹采集一次，各额外重放一次 suffix 检查 observer 输出；
重放不作为性能重复。保留离线分析没有 GPU 前向或计时重复。

保留 QA32 来自 2026-09-28 GPU1，CUDA 设备名 `NVIDIA H20Z`、SM90/132 SM，UUID
`a2e4882b-3248-6018-14ab-90e45b5a5b04`，PyTorch 2.12.1+cu130 / CUDA build 13.0。
保留 QA64 来自 2026-09-25，H200/SM90、132 SM，UUID
`2522820c-89d9-aa17-f79c-ca8cc767fb77`，PyTorch 2.10.0+cu132 / CUDA 13.2。
新三组的 `dense_qa64` 与保留 QA64 的 `block_ids.npy`、`valid_mask.npy`、`union_mask.npy`
逐元素相同，不一致元素均为 0；请求、checkpoint 哈希和模型配置也相同。
[环境对照](report/qa64/environment_comparison.json)只支持本请求的复现，不扩展为跨环境
的一般逐位一致保证。[QA64 重分析来源](report/qa64/analysis_provenance.json)继续保留。

## 完整 NOSA 三组

| 组别 | Prefix / candidate 传播 | Indexer 与选择来源 |
| --- | --- | --- |
| `dense_qa64` | 全程 dense，不加 CIS bias | FP32 QA-only：1 sink + 16 inclusive local + 47 QA；旁路记录 |
| `dense_nosa64` | 与上一组同一次 dense 前向、同一 Q/K/CIS | 完整 NOSA：17 inclusive local，QA 阶段含 mandatory 共 33 块，再由 CIS 补满 64；旁路记录 |
| `sparse_nosa64` | 全程 resident block sparse，含 CIS bias | 同一完整 indexer；记录 attention 实际收到的 selection |

两条传播轨迹从独立空 cache 构建完整 prefix，严格加载 A/delta。Dense 轨迹禁用模型内
indexer，CIS 供旁路打分但不加入 dense attention；sparse 轨迹使用独立 cache 与原 indexer/attention。
QA-only reference 按 64 queries 分批，完整 NOSA 使用 1024-query batch。

本次显式设置 `CXLDSAGR_SM90_BACKEND=native`，继承环境、源码和 checkpoint 身份均已保存。
模型接口的 `indexer_backend="triton"` 进入 SM90 dispatcher；受支持路径调用 native，
短 score 等分支仍可回退 Triton。本实验没有逐 kernel profile，不能据此断言每次调用
都使用 native。[执行来源](report/sparse_compare/execution_provenance.json)记录这一边界。
图中的完整 NOSA 后端标签据此写为 SM90 dispatcher；只更正图注，不修改选择数组或原 metadata。
[算子与完整模块测量](../nosa_kernel_mfu/README.md)另行报告时间，不替代本节 pattern 观测。

| 组别 | Unique K+V MiB | 完整 KV 覆盖率 | Prefix MiB | Candidate MiB |
| --- | --- | --- | --- | --- |
| Dense QA-only64 | 788.34375 | 37.9011% | 756.34375 | 32 |
| Dense full NOSA64 | 583.46875 | 28.0514% | 551.46875 | 32 |
| Actual sparse full NOSA64 | 492.28125 | 23.6674% | 460.28125 | 32 |

从 dense QA-only 到 dense full NOSA，并集减少 204.87500 MiB、覆盖率减少
9.8498 个百分点。这同时改变 local 边界、QA/CIS 分配和评分算术，不能单独归因为 CIS。
从 dense full NOSA 到实际 sparse full NOSA，并集再减少 91.18750 MiB、
4.3840 个百分点，反映 block mask、CIS bias 和后续激活变化组成的传播差异。
三组均覆盖全部 candidate 块；容量缩减不直接等于延迟或带宽收益。
两条轨迹 observer 重放的 hidden 最大绝对差均为 0、输出有限；第 0 层两组 full-NOSA 选择相同。

![完整 NOSA 三组](report/sparse_compare/comparison.png)

![完整 NOSA 逐层并集](report/sparse_compare/union_heatmap.png)

[JSON](report/sparse_compare/comparison.json)、[逐层对照](report/sparse_compare/layer_comparison.csv)、
[逐 head 对照](report/sparse_compare/head_comparison.csv)均取自新 run；图表来源及标签修改见
[图表来源](report/sparse_compare/figure_provenance.json)。

## QA-only 的 32 / 64 blocks

该 policy 在 dense 激活的 RoPE 后 Q/K 上旁路运行，不改变 dense attention。
K 采用 32-token / stride-16 mean compression；逐 Q head FP32 softmax，再做 GQA 求和与五窗口
max pooling。TF32 关闭；相同分数优先较小 block ID。固定 1 sink + 16 local（含当前块），
其余分别选 15 / 47 个 query-aware blocks，不启用 query-agnostic/CIS。
它与完整 NOSA 的 17-local、QA 阶段保留 33 块再用 CIS 补足 64 块的 policy 分开分析。

| 预算 | Unique K+V MiB | 完整 KV 覆盖率 | Prefix MiB | Candidate MiB | 单 head 并集块数范围 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 526.65625 | 25.3200% | 494.65625 | 32 | 114–788 |
| 64 | 788.34375 | 37.9011% | 756.34375 | 32 | 147–979 |

32-block 观测并集比 64-block 少 261.6875 MiB；完整输入、模型与 QA-only policy 之外的
预算固定项相同，当前环境的 QA64 选择复现证据见前节。
两组的 65536 个 `(layer, query, KV head)` 选择均满足预算数、唯一性与因果约束。
单 query 的预算比例不能代表 1024-query 并集：部分 head 的选择覆盖了大部分历史。

![QA32 逐层容量](report/qa32/capacity_by_layer.png)

![QA64 逐层容量](report/qa64/capacity_by_layer.png)

数据：[QA32 汇总](report/qa32/summary.json)、[QA32 逐 head](report/qa32/head_stats.csv)、
[QA64 汇总](report/qa64/summary.json)、[QA64 逐 head](report/qa64/head_stats.csv)。
对应 `report/qa32/` 和 `report/qa64/` 还保存逐层 CSV、union heatmap 与 SVG。

### Sink/local 与 query-aware 分解

先分别跨 queries 合并固定部分与 QA 部分，再求二者交集。单个 query 的两部分不重叠，
但某块可在较早 query 中属于 local，滑出 local 窗口后在较晚 query 中被 QA 选中，
因此两种 union 不能直接相加。

| 预算 | Sink MiB | Local MiB | Fixed union MiB | QA union MiB | 交集 MiB | 额外 QA MiB | Combined MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 62 | 64 | 492.21875 | 29.56250 | 462.65625 | 526.65625 |
| 64 | 2 | 62 | 64 | 754.34375 | 30.00000 | 724.34375 | 788.34375 |

按 50 GB/s 计算，固定部分为 1.342177 ms，额外 QA 部分分别为 9.702605 / 15.190589 ms。
这些是 payload / 假定带宽，不是传输计时。重建 combined union 与原选择逐元素一致。
[32 分解](report/components32/decomposition.json)、[64 分解](report/components64/decomposition.json)
以及各目录的逐层/head CSV、`component_capacity_fetch.png` / `.svg` 来自上述两组分解 run。

### 所有 layer / KV heads 的分布

每个 `(layer, KV head)` 等权提供一个样本，共 64 个；分母为该 head 的完整 KV。
分位数使用线性插值。32 / 64 的均值分别为 25.32% / 37.90%，中位数为 18.27% / 29.47%，
P90 为 48.96% / 68.64%，最大值为 75.77% / 94.13%。平均值不能描述高覆盖 head 的开销。

![32/64 head 覆盖率分布](report/distribution/head_coverage_distribution.png)

来自 `distribution32_64_20260928_01`；
[完整统计](report/distribution/distribution.json)、[样本表](report/distribution/head_coverage.csv)、
[直方图](report/distribution/histogram.csv)。

## 运行与调用模块

从仓库根目录运行，分析依赖使用 `uv sync --group analysis`。新三组的实际采集设置如下；
复现时使用新的 run ID。原采集读取前次 dense 请求，新 dense 中的请求字节完全相同，
所以下方复现入口指向保留的新 dense 请求。

```bash
CUDA_VISIBLE_DEVICES=5 CXLDSAGR_SM90_BACKEND=native \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  numactl --physcpubind=48-55 --membind=1 \
  bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh new_pattern \
  --request-file experiments/nosa_baseline_performance/output/data/dense_wrapper_20261004_01/request.json
```

保留 QA32 的原始命令如下，记录当时请求来源；重跑可使用上述字节相同的新请求。

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh \
  qa32_h200_gpu1_20260928_01 --block-budget 32 \
  --request-file experiments/nosa_gr_65536_1024/output/data/dense_h200_gpu1_20260928_01/request.json
```

当前三组脚本默认 `--mode capture`，只采集选择和执行完整性信息，不重放模型做数值对照。
独立 observer 数值检查使用同一脚本的 `--mode check`，从独立空 cache 构建 dense/sparse
prefix，比较开/关 observer 的 suffix hidden；产物留在系统临时目录，不发布为 pattern
或性能结果。Capture metadata 用 `null` 表示本次没有执行的 hidden 数值检查，不能
把它解释为通过。选块预算、因果约束、shape 和第 0 层相等关系仍作为采集完整性检查。

```bash
bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh pattern_check --mode check
bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh --help
```

保留 QA32/QA64 的分解和分布入口为 `scripts/decompose.sh`、`scripts/distribution.sh`。
其中既有分解报告的 payload / 50 GB/s 字段只是公式换算，不是搬运测量，也不再接入
已退出的 dense-MFU 时间、理想 overlap 或阈值策略链。保留这些文件是为了维持原
selection 分解证据的完整性，不将旧公式数值充当设计的实测收益。

脚本支持 `--help`，在系统临时目录完成后发布成功产物，拒绝覆盖 run ID；失败诊断留在
实验目录之外。数组、JSON/CSV、图表、源码在 `output/data/<run_id>/`，日志在
`output/log/<run_id>/`；本实验无 profiler。报告图表与数据在 `report/`，
[本轮范围清单](report/scope_manifest.json)只记录保留素材和退出路径，没有新增数值验收。
原 [Pattern 重算](report/pattern_integrity.json)、报告来源和 publication 仍描述
2026-10-04 的采集/发布；其中源码相等检查和旧 README 哈希只适用于当时的快照，
不代表本轮入口已经验收。原采集 metadata、选择数组和源码保持原样。

调用模块：三组为 [sparse_capture.py](src/sparse_capture.py)、[sparse_compare.py](src/sparse_compare.py)；
单组为 [capture.py](src/capture.py)、[analyze.py](src/analyze.py)；分解/分布为
[decompose.py](src/decompose.py)、[selection_parts.py](src/selection_parts.py)、[distribution.py](src/distribution.py)。
模型调用 `models.nosa.model`、`models.nosa.indexer`、`executor.model_executor.run_chunks`；
请求校验、执行边界和源码指纹显式复用
[baseline 性能实验](../nosa_baseline_performance/README.md) 的 `src/sparse/capture.py`、
`src/dense/capture.py` 与 `src/dense/sources.py`。
