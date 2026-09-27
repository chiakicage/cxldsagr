# NOSA indexer pattern：GR 65536 + 1024

## 实验目的与测量边界

检查 NOSA-8B 在 GR 输入上的稀疏选块，包括 dense 激活上的 query-aware 旁路采集，
以及完整 NOSA 在 dense / 实际 sparse 传播下的对照。每个 candidate token、每层、
每 KV head 选择 64 个逻辑块，再沿 1024 个 candidate queries 合并去重，计算需要访问的
K+V 块容量及其占完整 KV cache 的比例。

**完整 NOSA 三组对照：改动后未运行。** indexer 已改用增量压缩缓存、融合 QK/pooling
和 FlashInfer 两阶段 top-k；受影响的原三组对照结果、图表和运行产物已删除。
需要用当前实现重新采集后，才能报告完整 NOSA 的覆盖率、并集容量和组间差异。

输入为 synthetic GR 内容，user 623、visit 0、seed 42；instruction 28 + history 65508
组成 65536-token prefix，candidate suffix 为 1024 tokens，总计 66560。
默认请求读取
`experiments/indexer_block_sparse_profile/output/data/nosa_cached_indexer_20260928_01/request.json`。
这是独立保存的请求输入，不从其他实验的性能数字推导本实验结果。

本实验统计每个选中块仅计一次的 **unique K+V payload**，不测延迟、吞吐、带宽或总线流量。
不计 CIS、indexer 读取、KV 写入、元数据、重复 fetch 或缓存命中。
所有组均不执行 LM head、自回归生成或 offloading。模型上下文仅在实验进程内从
32768 覆盖到 66560，保留原 LongRoPE factors；不评价超长上下文质量。

## 完整 NOSA 三组对照

配置为 NOSA-8B BF16、32 layers、32 Q heads / 2 KV heads、D128、64-token block、
64 blocks/query/head。prefix 按 1024 tokens 分块；每条传播轨迹从独立空 cache 构建
完整 prefix，再用一次 forward 采集 candidate 的全部层。
无计时预热，每条轨迹采集一次；另各重放一次 suffix 检查 recorder 不改变输出，
该检查不作为性能重复测量。

| 组别 | Prefix / candidate 传播 | Indexer 与选择来源 |
| --- | --- | --- |
| `dense_qa64` | 全程 dense，不加 CIS bias | FP32 QA-only；1 sink + 16 local + 47 QA；旁路记录 |
| `dense_nosa64` | 与上一组同一次 dense 前向、同一 Q/K/CIS | 完整 NOSA Triton；17 inclusive local，QA 阶段含 mandatory 共 33 块，再由 CIS 补满 64；旁路记录 |
| `sparse_nosa64` | 全程实际 block sparse，含 CIS bias | 同一完整 indexer；记录 attention 实际收到的 selection，不重新选块 |

两条轨迹统一加载 A/delta。dense 轨迹仅替换 main attention 为 dense adapter，禁用
模型内 indexer；CIS 可供旁路打分使用，但不加入 dense attention。
sparse 轨迹使用原 indexer/attention 和独立 cache。
QA-only reference 的 query chunk 为 64；完整 NOSA Triton 使用完整 query batch，
实际值记录在各 arm 的 metadata 中。

第一对比较同时改变 local 边界、QA/CIS 分配和评分算术，不能把差异单独归因 CIS。
第二对保持完整 NOSA indexer、参数与后端一致，比较完整 attention 传播路径对 pattern
的影响，包括 block mask、CIS bias 和后续激活的变化。

当前结果状态为 **改动后未运行**，不提供完整 NOSA 的容量结论或图表。

## 独立 QA-only 采集

保留有效 run `query_aware_fp32_65536_1024_20260925_01` 的原始数据和复现记录。
其采集时间为 2026-09-25 15:25 UTC，H200 / SM90、132 SM，GPU UUID
`2522820c-89d9-aa17-f79c-ca8cc767fb77`；PyTorch `2.10.0+cu132`、CUDA 13.2、
FlashInfer `0.6.18`、NumPy `2.5.0`。BF16 模型权重、FP32 indexer，输入与上文相同，
无预热、采集一次，不测性能。

该路径在 dense Full Attention 前向产生的 RoPE 后 Q/K 上旁路运行
`mode="query_aware", backend="reference"`，记录选择后继续执行 dense attention。
它不使用本次改动的完整 NOSA Triton indexer、CIS 或压缩缓存；其 FP32 选块数学和
所用 dense 计算路径未变，因此独立原始采集仍有效。它不能代替当前完整 NOSA 三组对照。

数据保存在 `output/data/query_aware_fp32_65536_1024_20260925_01/`，包括
`block_ids.npy`、`valid_mask.npy`、`union_mask.npy`、请求、metadata、执行边界和源码快照。
对应日志保存在同 run ID 的 `output/log/`；没有保留独立 QA-only 报告图表。
保留记录中的输入与源码指纹，不将本次依赖审查表述为重新测量。

## 运行与调用模块

从仓库根目录执行，使用新的 run ID：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh \
  nosa_sparse_pattern_cached_rerun

CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh \
  query_aware_fp32_rerun
```

脚本支持 `--help`、`--request-file`、`--model-path` 和 `--device`；需要 Hopper / SM90、
PyTorch、Triton、FlashInfer、checkpoint/tokenizer 及 NumPy/Matplotlib。
capture → analyze 在系统临时目录完成后才发布成功产物，不覆盖已有有效 run。
默认只分析本次采集的三组；`--baseline-data-dir` 可显式指定独立 QA-only 采集核对。
这种核对还要求 `--request-file` 使用该采集保存的原始请求字节，以满足输入指纹校验。

完整对照调用 [sparse_capture.py](src/sparse_capture.py)、[sparse_compare.py](src/sparse_compare.py)；
独立 QA-only 调用 [capture.py](src/capture.py)、[analyze.py](src/analyze.py)。
复用 `models.nosa.model` / `models.nosa.indexer`、`executor.model_executor.run_chunks`、
`experiments.indexer_block_sparse_profile.src.capture.validate_request`，以及 dense 实验的
`execution_split` / `source_hashes`。实验代码不复制 attention 或打分数学实现。

成功运行的产物布局：

```text
output/data/<run_id>/       metadata、request、execution、源码快照和分析结果
                           arms/<arm>/block_ids.npy、valid_mask.npy、union_mask.npy
                           逐层/逐 head 表格、comparison 与 union_heatmap 图表
output/log/<run_id>/        独立 stdout/stderr
output/profile/<run_id>/    本实验默认不启用 profiler
```

完整对照的选择形状为 `[32,1024,2,64]`，并集 mask 为 `[32,2,1040]`。
分析后选定的有效报告图表可复制至 `report/`，并记录新 run ID 和生成方式。

`distribution.sh` 与 `decompose.sh` 可离线分析保留的 QA-only 原始选择。
`estimate.sh`、`overlap.sh`、`threshold_sweep.sh` 的延迟/传输推算依赖有效的性能输入，
当前为 **改动后未运行**：`estimate.sh` 必须显式传入新有效 dense run 的
`--dense-data-dir`，后两者必须显式传入新生成的 `--estimate-data-dir`。
这些工具是计算模型，不是实测 sparse 延迟或 offload 验证。
