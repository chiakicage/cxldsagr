# Q1 融合准备的 private 对照

该候选将 page64 打包、identity block table 和 host-token/staging 准备合并，保留官方
ECHO score/prefetch 核心、阈值、64 条 record 上限、精确 top-k、EMA 及后续发布和清理。
它目前只在实验私有入口中运行。本报告保留接入 production 前的有效优化对照；独立验收
和测量完整性检查通过，不等于生产版本已获得相同收益。

组件与完整模型使用不同计时边界。组件的冷缓存三层配对收益合计为
4.944 µs；完整模型的配对中位数变化为
-34.6505 µs。这两个数字不能相减或用来证明全部差值来自准备融合。

## 组件计时

输入取真实前三层 Q1 张量，H=65,536、N=65,537。每个 L0/L1/L2 ×
zero/small/partial/warm/empty 组合预热 20 次，再运行 100 对交替 AB/BA，共 3,000 个
arm 样本。每次在计时外恢复该策略的驻留状态。CUDA Graph 内的 GPU event 覆盖完整
`echo.logits` 和原 staging cleanup，包括打包、边界和页表准备、官方调度、stage reset、
融合分数、causal clean 和 promotion；caller FIFO 准备、精确 top-k 与 recall 在窗口外。

zero 从无驻留历史、阈值 0 开始；small 同样无驻留，阈值取 history 分数的第 17 大值。
partial 预先驻留隔一个 token 的历史，warm 驻留全部历史，两者阈值均为 0。
empty 无驻留且阈值为正无穷，因而不触发预测搬运。这些是明确定义的组件控制条件。

| 冷缓存 | baseline µs | candidate µs | 配对中位数变化 µs |
| --- | ---: | ---: | ---: |
| L0 | 37.168 | 35.936 | -0.976 |
| L1 | 62.336 | 60.976 | -1.744 |
| L2 | 75.744 | 73.536 | -2.224 |

15 个组合的配对中位数均下降，范围为 0.976–2.800 µs；每个组合的 AB、BA 中位数也都
下降。完整组合数据见 [component_groups.csv](component_groups.csv)。原始 3,000 个
样本保留在组件 run 的 `summary.json` 及本报告 run 的输入副本中。

## 完整前三层模型计时

使用原始 FP8 权重、BF16 KV，真实 checkpoint L0–L2，H=65,536、A=1，history chunk
为 1,024，P=NH=65,600。baseline 与 candidate 是同一进程中依次准备的两个固定模型
实例，分别预热 5 次。正式计时运行预先规定的 500 对 AB/BA，各 250 对；每次恢复 cold
prefix/hint。窗口为完整 `forward(return_hidden=True)` 加 synchronize，恢复、绑定、
计数读取和诊断在窗口外。本表不使用 profile 的时长。

| arm | 中位数 ms | 均值 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.6468310 | 2.7294701 | 3.4974475 | 3.5892731 |
| candidate | 2.6119945 | 2.6728263 | 3.3105815 | 3.6582937 |

candidate 在 413 / 500 对中较快，配对中位数下降
34.6505 µs。下表保留全部时间区段；block 0–4 分别对应连续
100 对。AB 表示 baseline 先执行，BA 表示 candidate 先执行。差值均为 candidate 减
baseline，负值表示本次观察中 candidate 较快。

| block | 顺序 | 对数 | 配对中位数变化 µs | 配对均值变化 µs | candidate 较快对数 |
| --- | --- | ---: | ---: | ---: | ---: |
| all | all | 500 | -34.650 | -56.644 | 413 |
| all | AB | 250 | -29.900 | 3.139 | 197 |
| all | BA | 250 | -38.483 | -116.427 | 216 |
| 0 | all | 100 | -31.555 | 15.135 | 76 |
| 1 | all | 100 | -38.435 | -67.487 | 81 |
| 2 | all | 100 | -35.290 | -156.598 | 87 |
| 3 | all | 100 | -35.605 | -92.234 | 87 |
| 4 | all | 100 | -33.285 | 17.965 | 82 |

两种顺序及五个区段的中位数都改善；每个区段内单独的 AB、BA 中位数也都改善。
均值和尾部并不一致：AB 均值上升 3.139 µs，block 0 和 4 均值上升；candidate p99
从 3.589273 增至 3.658294 ms。81 对的绝对差值超过
0.5 ms，其中 35 对 candidate 较慢、46 对较快。没有删除异常值、裁剪早晚区段、
替换样本或重加权。[model_pairs.csv](model_pairs.csv)保留全部 500 对及各对实际 H2D，
[model_statistics.csv](model_statistics.csv)包括每个 block 的两种顺序。
五区段重采样只描述本次观察对区段变化的敏感性，不是独立运行的置信区间。

每层实际预取均为 64 条，每条 1,152 B；预测与精确 top-k 的重合会随调度变化，
剩余选择仍完整 recall。三层合计 H2D 的配对中位数增加 2,304 B，
均值增加 5,603.328 B，范围
-79,488 至 73,728 B。
这些字节来自每次执行的实际 cache 计数，不能由 memcpy 行数推断，也不能用 profile
样本的流量替代。H2D 增加说明两条轨迹的工作量并非逐项相同。

两 arm 的 graph private reserved 均为 62,914,560 B。保存的 `graph.describe` 计数和
reservation 算术已核验，但没有保留 allocator 原始 segments，无法独立重建 private
segment 归属。allocated、reserved 和设备已用量分别保留在 [result.json](result.json)；
它们是依次准备模型时的进程快照，不能相减得到单个模型占用，也不是连续峰值或容量验收。

## 验收、来源与复现

两组运行都使用 GPU 1（NVIDIA H200，SM90），CPU 8–15，线程数 8，GPU UUID 为
`a2226185-cb05-a411-80da-f365154128fe`；Torch 2.12.1+cu130、CUDA 13.0、
Triton 3.7.1、FlashInfer 0.6.18。组件 receipt 覆盖 100 个 byte/layout case、210 个
完整 case，另有 6 个 malformed 子进程检查。模型 receipt 覆盖两个独立准备的 prefix、
三个变化 suffix token（111090、111091、111092）、eager、诊断 graph 和 clean graph，
包括输出、所有 offset 位、精确 scores/top-k 与实际 saturated-prefetch 转移。

独立 CPU 复核重新读取了 36 组 raw layer scores/top-k，重算 12 组 compact transition
proof，并检查映射、priority、free bit、clock 和计数。compact 证据没有完整 staged、
host、最终 KV payload；这些内容的逐字节比较仍依赖签名运行时验收。CPU 复核没有执行
GPU、重测性能、完整重哈希权重或重建 allocator segments。

来源 run ID：组件 `q1_fused_prepare_bench_20261008_02`，模型
`q1_fused_prepare_model_bench_20261008_02`，模型分析
`q1_fused_prepare_model_analysis_20261009_01`。receipt 签名分别为
`d7d9a19db115369a9d9bd025f71e5090896f21aa7adb1eb7d695f29b1a25549e` 和
`d60a9ee4d574f4c78846d12926d35fd1fa7b454805eb8f1cde420524f949e714`。完整输入、源码及 native 身份在原运行与
分析清单中；[provenance.json](provenance.json)给出来源文件哈希、关键源码/ELF、
三份不可变 FlashInfer ELF 和独立复核范围。FlashInfer 的源码、spec、include、toolchain
与编译器 `.d` 依赖闭包绑定到实际加载文件，check 和 bench 使用相同 ELF。官方 ECHO
headers 未修改，固定提交为 `bc1b75c1000010d0ac6f032ebaac283255c050b1`。
checkpoint 身份使用 metadata 哈希和全部 shard 的文件系统身份，未完整哈希权重内容。

报告生成器核验已接受分析及复核文件的来源哈希，重新统计全部组件和模型样本、顺序、
区段与流量算术，再逐字节核验选定副本。它不重复运行 GPU 验收，也不重新读取全部
tensor/native/source 依赖；该范围由已归档的独立检查和分析负责。

从仓库根目录运行下列命令；输出目录必须使用新的 run ID。完整报告原件和输入副本在
`experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_report_20261009_02/`，选定文件的哈希见
[publication.json](publication.json)。

```bash
python -m experiments.deepseek_v32_echo_official.src.report_q1_fused_prepare \
  --component-dir experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_bench_20261008_02 \
  --model-analysis-dir experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_model_analysis_20261009_01 \
  --model-check-review /tmp/cxldsagr-checks/q1-fused-prepare-model/check_20261008_02_independent_review.json \
  --output-dir experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_report_YYYYMMDD_NN
```

本次结论只覆盖 private 的真实前三层、固定输入和单进程配对计时。独立 source-bound
profile/work-equivalence 解释及 production 接入后的正式 cohort 另行验收；它不建立
完整 61 层、serving、session 容量或 SGLang 与本地模型数值等价的结论。
