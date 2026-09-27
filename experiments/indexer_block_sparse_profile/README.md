# NOSA indexer + block sparse：64K + 1K 端到端性能

## 实验目的与测量边界

测量完整 NOSA sparse 模型的单请求端到端前向延迟，并定位 CIS、indexer 与 block sparse
attention 的开销。主指标来自未启用 profiler 的独立进程；模块分解和 Nsight Systems
时间线由第二个进程采集。两者使用同一请求、checkpoint、模型参数、源码和设备。

| 阶段 | 起始缓存 | 模型输入 | 执行方式 | 主指标 |
| --- | --- | --- | --- | --- |
| `full_prefill` | 空 | 66,560 tokens | 65 个 1024-token chunk，全部 32 层 | 整段墙钟延迟、66,560 / 秒 |
| `extend` | 65,536-token prefix 已完成 | 1,024 candidate tokens | 单次 1024-token forward，全部 32 层 | 整段墙钟延迟、1,024 / 秒 |

64K 是 instruction + history 的合计，1K 是完整 candidate suffix，不把历史尾部移入 extend。
当前模板 instruction 为 28 tokens，因此 GR 输入预算为 history 65508、item budget 1052；
实际执行边界为 `[0,65536)` 与 `[65536,66560)`。`request.json` 和 `execution.json`
保留真实边界，逐层 audit 检查 `Q=[1024,32,128]`、`K=V=[66560,2,128]`、
selection=`[1024,2,64]` 且全部选择有效。

使用 NOSA-8B 的 BF16 权重、32 layers、32 Q heads、2 KV heads、D128。完整 sparse 策略为
64-token block、17 inclusive local、含 sink/local 的 query-aware 阶段保留 33 块，
query-agnostic CIS 补满 64 块；attention 加入原始 CIS 分数。模型及算子均调用
[现有 NOSA 实现](../../models/nosa/README.md)，不在实验代码复制计算逻辑。

端到端计时包含 embedding、全部 decoder、CIS 投影、K/V/CIS 写入、压缩、query-aware
打分与选块、block sparse attention、输出投影、FFN、final norm、Python 提交和 CUDA 完成等待。
输出为最后一个 chunk 的 normalized hidden states，不执行 LM head、采样或自回归生成。
checkpoint 加载、tokenizer/GR 生成、输入 H2D、缓存分配/reset 以及 extend 的 prefix 构建
均在计时外；这属于单请求模型前向延迟，不是网络服务延迟。

完整 K/V/CIS 驻留 HBM，无 DRAM backing、offload fetch、缓存淘汰或 fetch/compute overlap。
当前 indexer 通过 IndexerCache 复用压缩前缀，只计算新增 K/CIS 窗口和稳定 pool；
检查当前 Q 及尚未校验的 K/CIS 后缀。这些开销纳入主计时。CUDA indexer 一次处理完整 query batch，
workload 的 `indexer_query_chunk_size=null` 表示不分块，
`indexer_execution=cached_flashinfer_v1` 标识增量缓存与 FlashInfer 两阶段选择。
仅在实验进程把上下文上限扩大为 66560，保留 checkpoint 的 LongRoPE factors；不改磁盘
配置，不对超出原 32768 上下文后的模型质量作结论。

本次比较使用同一代码、请求、checkpoint 和测量参数，仅切换 `--kernel-backend`：

- `native`：QK score 使用本地 CUDA/CuTe WGMMA：第一 kernel 计算分段 softmax normalizer，
  第二 kernel 重算 QK、合并 normalizer，融合 GQA 舍入和五窗口 pooling。
  本实验的 1024-query batch 在压缩 K≥2047（上下文≥32768 tokens）时使用该路径，
  更短的已打分 chunk 保留单 kernel Triton；
  block sparse attention 使用 TMA producer 与 WGMMA QK/PV consumer，融合 CIS、因果遮罩和在线 softmax。
- `triton`：保留的两遍 fused QK/pooling 与 block sparse attention 实现，用新 run 重新测量。

这里的 `native` 表示上述混合 QK 调度与原生 attention 组合。
两组均使用增量压缩缓存与 FlashInfer Top-33/Top-64；native 不引入 EzKernelKit 运行时，
只通过 TVM FFI 编译 `operators/sm90/csrc/`，复用顶层共享 CUTLASS。
`workload.kernel_backend` 区分 `cuda_tvm_ffi` / `triton`；旧接口字段 `backend="triton"`
仍表示 SM90 dispatcher。构建元数据记录编译器、编译参数、TVM FFI、CUTLASS 与 CUDA 源码指纹。
64K+1K 的 native normalizer 临时缓冲为 0.25 MiB，其分配与计算计入计时；
该临时缓冲不属于请求 resident cache 的容量统计。
两组独立构建各自的 sparse prefix；浮点舍入可能改变近似并列的选块，不能据此声称输出逐位相同。

## 运行

从仓库根目录执行，脚本也能从其他工作目录启动：

```bash
bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_native_001 --kernel-backend native
bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_triton_001 --kernel-backend triton
bash experiments/indexer_block_sparse_profile/scripts/run.sh --help
```

默认设备 `cuda:0`，checkpoint `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`；每阶段预热 2 次，
独立端到端计时 5 次取中位数，模块 profile 每阶段 1 次。第二个进程同样先完整预热，
JIT 编译、CUDA 模块加载及 allocator 初始化不计入正式采集。

可明确使用已有实验的同一 GR 请求：

```bash
bash experiments/indexer_block_sparse_profile/scripts/run.sh sparse_e2e_same_request \
  --request-file experiments/indexer_block_sparse_profile/output/data/nosa_native_wgmma_20260928_02/request.json \
  --device cuda:0 --warmup 2 --repeats 5 --profile-repeats 1
```

不传 `--request-file` 时用共享 GR 生成器、seed=42 生成固定长度请求；profile 总是读取
benchmark 已保存的同一请求。脚本固定 64K+1K 和 1024 chunk。当前依赖 SM90/Hopper、
PyTorch、Triton、FlashInfer、TVM FFI、nvcc、共享 CUTLASS、safetensors、tokenizer 与 Nsight Systems，沿用项目环境。
`--without-nsys` 保留独立端到端计时及 CUDA event 模块分解，省略原始 nsys / SQLite。
分模块 kernel MFU 依赖 SQLite，`--without-nsys` 时仅生成端到端 MFU。

检查整体提交瓶颈时使用 `--timeline-only`：仍先运行独立、无 profiler 的 benchmark，
第二进程只记录每次完整 forward 的根 NVTX 与 CUDA 活动，不安装 `SparseScopes`、
逐层 hook 或模块 CUDA events；只保留整段计时的起止 events。该模式生成
`launch_analysis.json`、`launch_summary.json/csv/md`，不生成模块分解或 MFU。
它需要 nsys，不能与 `--without-nsys` 同用；可用 `--profile-repeats 3` 重复时间线采集。

Python 入口：

```bash
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.capture --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.analyze --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.mfu --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.module_mfu --help
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.launch_report --help
```

脚本依次执行独立 benchmark → nsys profile → SQLite export → analyze → 离线 MFU。所有步骤在系统
临时目录 staging，子进程失败保留退出码，失败产物留在实验目录之外。只有全链成功才发布
至新的 run ID，已有 ID 拒绝覆盖。
启用 nsys 时，最后还从 SQLite 生成分模块 kernel MFU。

## 指标定义与产物

- `wall_ms`：开始提交到设备全部完成的墙钟毫秒数，主端到端指标。
- `host_submit_ms`：模型前向的 host 提交时间，含模型内部 CUDA 等待；它与设备执行重叠，
  不能与 CUDA 时间相加，也不表示纯 CPU 计算时间。
- `cuda_span_ms`：整段 CUDA events 之间的时间，含 host launch 造成的空档，不是 kernel active 时间。
- 模块 `cuda_elapsed_ms`：单独 profile 下的 inclusive CUDA event 区间，可能含 launch gaps
  和 instrumentation 开销；先汇总一次运行内全部层/chunks，再对 profile repeats 取中位数。
  `cis_projection`、`indexer_total`、`block_sparse_attention` 为独立顶层区间。
  当前 indexer 子项为 `indexer_validate`、`indexer_cache_update`、`pooled_scores`、
  `topk_qa`、`prepare_cis`、`topk_cis`、`finish_selection`。
  子项不能再次加到 indexer 总时间上，也不从主 wall
  中减出“其他模块”。没有新完整窗口时，cache update 可以不发起 kernel。
- `mfu_pct`：有效矩阵 FLOPs /（未插桩墙钟中位数 × GPU 峰值 FLOP/s）× 100。
  默认采用 H200 SXM 标称 BF16 dense Tensor Core 峰值 **989 TFLOPS**，不是 2:4 稀疏峰值，
  也不是实测可持续峰值；其他 GPU/SKU 可用 `--peak-tflops` 显式指定。

完整前向与 prefix-ready extend 使用相同分块边界，测量外检查全部 1K candidate hidden
有限且精确一致；profile 还检查 instrumentation 不改变这些输出。正确性检查是数据验收，
不作为性能结果。每次 full-prefill 在计时前 reset 独立 cache；重复 extend 回退独立 prefix
cache 的长度，只重写 suffix。若模型未来引入 opaque layer state，会明确拒绝这种回放。

产物默认忽略，不提交 Git：

```text
output/log/<run_id>/             各子进程分开的 stdout/stderr
output/data/<run_id>/            request.json、execution.json、metadata.json
                                measurements.json、profile_metadata.json、attention_audit.json
                                summary.json、report.md、timings.csv、stages.csv、layers.csv、mfu.json
                                module_mfu.json、module_mfu.csv、module_mfu.md（需要 nsys）
                                sources/ 源码快照、nsys.sqlite
output/profile/<run_id>/         indexer_block_sparse_profile.nsys-rep
```

metadata 记录实际硬件/UUID、依赖版本、输入与精度、原始/实验上下文、checkpoint 文件
属性、源码指纹和测量边界。Nsight 关闭 CPU sampling/context-switch tracing，仅采集
CUDA/NVTX/OS runtime；模块 NVTX 标签为 `NOSA/<phase>/layer_<id>/<stage>/q<start>+<length>`。

调用模块：`src/capture.py` 复用共享 GR、`executor.model_executor.run_chunks`、
`models.nosa.model`、dense 实验的
`experiments.nosa_gr_65536_1024.src.capture.execution_split` 和 `src.sources.source_hashes`；
`src/instrumentation.py` 临时包装已有 scoring/indexer/attention，退出时恢复；
`src/analyze.py` 校验完整阶段、层与 chunk 覆盖并聚合测量。实验 CPU 单元测试纳入
全局 `scripts/run_tests.sh`，结果仅输出终端。`src/mfu.py` 复用
`experiments.nosa_gr_65536_1024.src.mfu.matrix_flops` 的投影/FFN 计数，将 dense attention
替换为实际 sparse 因果工作量，并加入 CIS 与 indexer QK；只读取已完成的运行数据。
`src/module_mfu.py` 复用同一 FLOPs 口径，从 Nsight SQLite 关联 CUDA launch 与 kernel，
验证逐层执行顺序后归因模块。

## 结果与结论

本节只使用迁移后重新完成的两组有效运行：native 为 `nosa_native_wgmma_20260928_02`，
Triton control 为 `nosa_triton_control_20260928_01`。两组的 57 个采集源码指纹、请求字节、
checkpoint、设备、依赖和预热/重复参数一致；构建信息只允许 `selected_backend` 不同。
每组独立完成 benchmark 和模块 profile，`compare.py` 已校验派生报告的输入哈希及上述一致性。
请求为 synthetic GR、user 623 / visit 0 / seed 42，输入与精度沿用上文的 65536+1024、BF16。

### 实测环境

两组 metadata 分别记录于 2026-09-27 18:36 / 18:39 UTC（北京时间 9 月 28 日 02:36 / 02:39）。
PyTorch 返回设备名 `NVIDIA H200`，Nsight 和 `nvidia-smi` 的原始名称为 `NVIDIA M403`；
设备为 SM90、132 SM，三个来源的 UUID 均为 `2522820c-89d9-aa17-f79c-ca8cc767fb77`。
驱动为 `570.124.06`、功率上限 700 W。MFU 统一采用前述 **989 TFLOPS 标称参考值**。

| 依赖 | 实际运行版本 | 同次源码快照中的 `uv.lock` |
| --- | --- | --- |
| PyTorch | `2.10.0+cu132` | `2.12.1+cu130` |
| Triton | `3.6.0` | `3.7.1` |
| FlashInfer | `0.6.18` | `0.6.18` |
| TVM FFI | `0.1.14.post0` | `0.1.13.post3` |

实际 CUDA 为 13.2，nvcc 为 `13.2.78`（`/usr/local/cuda-13.2/bin/nvcc`）；
CUTLASS 为 `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`，
Nsight Systems 为 `2025.6.3.541-256337736014v0`。native 编译使用 `-O3`、C++20、
`sm_90a` 与 `-lineinfo`，完整 flags、版本和指纹见两组 metadata。
本次使用现有实测环境，未将它表述为按当前锁文件重新安装后的验证；
`pyproject.toml` / `uv.lock` 已随源码快照保存，运行版本以 metadata 为准。

### 独立端到端计时

每阶段预热 2 次、计时 5 次；表中为墙钟中位数及 `[min, max]`，单位 ms。
加速比统一为 Triton 中位数 / native 中位数，MFU 分母为各自未插桩的墙钟中位数。

| 阶段 | Triton wall ms `[min, max]` | native wall ms `[min, max]` | Triton MFU | native MFU | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_prefill | 3726.672 [3707.632, 3742.651] | 3429.516 [3422.823, 3431.758] | 32.10% | 34.88% | **1.087×** |
| extend | 63.283 [62.978, 64.920] | 58.134 [58.091, 58.205] | 30.07% | 32.73% | **1.089×** |

当前输入下，native 组合使 full_prefill / extend 墙钟分别降低 **7.97% / 8.14%**。
两组有效矩阵工作量相同：full 为 `1,183,130,731,937,792` FLOPs，extend 为
`18,820,462,804,992` FLOPs。QK 只计一次逻辑矩阵乘，不把 normalizer/score 两遍重算、
pooling halo 重叠、future mask 或 tile padding 计入分子；非矩阵操作的时间仍留在分母。
每组内部的 full/extend candidate hidden 检查均有限且最大绝对差为 0，profile 也保持输出不变；
这些验收不表示 native 与 Triton 两组输出逐位相同。

### 算子 kernel 时间对照

以下为每组独立 nsys 进程中一次完整 profile 的 correlated kernel duration 总和，单位 ms。
它不是前表的墙钟分段；不能从未插桩 wall 中减去这些时间，也不能与 CUDA event 区间相加。

| 模块 | full Triton | full native | 加速比 | extend Triton | extend native | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pooled_scores` | 292.927 | 285.270 | 1.027× | 8.815 | 8.290 | 1.063× |
| `block_sparse_attention` | 1193.840 | 874.409 | 1.365× | 18.450 | 13.931 | 1.324× |
| `indexer_total` | 677.388 | 667.225 | 1.015× | 15.365 | 14.750 | 1.042× |

attention 的 kernel 时间改善最明显，full / extend 分别为 **1.365× / 1.324×**；
pooled_scores 为 **1.027× / 1.063×**，整个 indexer 为 **1.015× / 1.042×**。
每组只有一次模块 profile，这些比值描述本次归属结果；不把 profile 的时间差直接当成
未插桩墙钟差的可加分解，也不据此声称已隔离纯 launch 开销或 HBM 带宽瓶颈。

native 的 QK 采用模型 dispatcher 中的混合调度：1024-query batch 在完整压缩
K 数量达到 2047 时才使用 `normalizer_kernel → scores_kernel`，此前使用一个 Triton
`_scores` kernel。full 的前 4 个 chunk（4096 tokens）直接选全块、绕过 QK；
随后 27 个 chunk × 32 层 = 864 个 Triton score scopes，最后 34 个 chunk × 32 层
= 1088 个 native score scopes。因此 full `pooled_scores` 为 **864 + 2×1088 = 3040 kernels**；
extend 的 32 个 scopes 全走 native，对应 **64 kernels**。Triton control 分别为
**1952 / 32 kernels**。两组 attention 都是每层/chunk 一次 launch，即 **2080 / 32 kernels**。
增加 launch 数没有被当作额外 useful FLOPs；分模块归属按捕获的 query 范围严格验证分支和顺序。

### native 模块 MFU 与 indexer 子项

模块 MFU 百分比为 100 × 有效矩阵 FLOPs /（correlated kernel 总秒数 × 989 TFLOPS）。
这是有效矩阵 MFU，不是 SM occupancy；CIS 与 indexer 的 inclusive 分母也包含非矩阵工作。

| native 模块 | full kernel ms | full MFU | extend kernel ms | extend MFU |
| --- | ---: | ---: | ---: | ---: |
| `qkv_proj` | 104.333 | 77.92% | 1.604 | 77.99% |
| `o_proj` | 90.824 | 79.56% | 1.396 | 79.64% |
| `gate_up_proj` | 714.265 | 80.94% | 10.983 | 80.98% |
| `down_proj` | 346.598 | 83.40% | 5.315 | 83.67% |
| `block_sparse_attention` | 874.409 | 15.90% | 13.931 | 15.84% |
| `indexer_total` | 667.225 | 5.48% | 14.750 | 7.59% |
| `cis_projection` | 25.849 | 0.0085% | 0.396 | 0.0086% |

| native indexer 子项 | full kernel ms | extend kernel ms | full / extend kernel 数 |
| --- | ---: | ---: | ---: |
| `indexer_validate` | 10.496 | 0.160 | 4160 / 64 |
| `indexer_cache_update` | 12.883 | 0.199 | 2080 / 32 |
| `pooled_scores` | 285.270 | 8.290 | 3040 / 64 |
| `topk_qa` | 153.038 | 2.500 | 3904 / 64 |
| `prepare_cis` | 19.477 | 0.561 | 1952 / 32 |
| `topk_cis` | 178.672 | 2.922 | 3904 / 64 |
| `finish_selection` | 7.119 | 0.116 | 1952 / 32 |

`pooled_scores` 的有效矩阵 MFU 为 **12.81% / 13.51%**（full / extend）；
其他 indexer 子项没有计入矩阵 FLOPs，MFU 为不适用，不能记为 0%。
这些子项已包含在 `indexer_total` 中，不能与 parent 再相加。full 的短上下文全块选择
直接记入 parent，因此子项表不覆盖 parent 内的全部 kernel。
同次 native profile 的 indexer CUDA event 区间为 **1523.893 / 27.311 ms**，
而 kernel 总时间为 **667.225 / 14.750 ms**；前者含提交空档及插桩影响，二者不是可相加的组成项。

本次完成的是 resident HBM 路径的同源实现对照：当前工作量下端到端约 **1.09×**，
attention kernel 的收益大于 QK/indexer 的收益。没有 DRAM/CXL 搬运、fetch/compute overlap
或网络服务测量，结果不能用作 offload 性能或模型质量结论。

### 报告数据与复现

以下文件从上述成功 run 直接复制，保留原始数值。`stages.csv` 是 inclusive event 数据，
`module_mfu.csv` 是独立 nsys kernel 数据，二者用途不同。完整按层/按 chunk 记录、
源码快照与 SQLite 仍在各 run 的 `output/data/`，原始 nsys 在对应 `output/profile/`。

| 来源 | 墙钟汇总 | event 阶段 | 端到端 MFU | 模块 kernel / MFU | 采集元数据 |
| --- | --- | --- | --- | --- | --- |
| `nosa_native_wgmma_20260928_02` | [timings.csv](report/native/timings.csv) | [stages.csv](report/native/stages.csv) | [mfu.json](report/native/mfu.json) | [module_mfu.csv](report/native/module_mfu.csv) | [metadata.json](report/native/metadata.json) |
| `nosa_triton_control_20260928_01` | [timings.csv](report/triton/timings.csv) | [stages.csv](report/triton/stages.csv) | [mfu.json](report/triton/mfu.json) | [module_mfu.csv](report/triton/module_mfu.csv) | [metadata.json](report/triton/metadata.json) |

两组比较表为 [comparison.csv](report/comparison.csv)，完整输入指纹与比较定义为
[comparison.json](report/comparison.json)。它们来自 native run 的 `comparison/`，
由现有 `src/compare.py` 生成；该工具校验同一请求内容、checkpoint、设备、运行参数、
源码和 build，以及 MFU/模块报告的输入哈希，再计算中位数之比。

重建比较材料时从仓库根目录执行；`--output-dir` 必须使用尚不存在的新目录：

```bash
native_data=experiments/indexer_block_sparse_profile/output/data/nosa_native_wgmma_20260928_02
triton_data=experiments/indexer_block_sparse_profile/output/data/nosa_triton_control_20260928_01
report_dir=experiments/indexer_block_sparse_profile/report
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.compare \
  --native-data-dir "$native_data" --triton-data-dir "$triton_data" \
  --output-dir "$native_data/comparison_rebuilt"
mkdir -p "$report_dir/native" "$report_dir/triton"
for name in timings.csv stages.csv mfu.json module_mfu.csv metadata.json; do
  cp "$native_data/$name" "$report_dir/native/$name"
  cp "$triton_data/$name" "$report_dir/triton/$name"
done
cp "$native_data/comparison_rebuilt/comparison.csv" "$report_dir/comparison.csv"
cp "$native_data/comparison_rebuilt/comparison.json" "$report_dir/comparison.json"
```
