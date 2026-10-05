# DeepSeek V3.2 前三层：执行效率与 Offload 开销

本实验用于检查 baseline 实现的性能合理性。在同一 H200 上运行真实 checkpoint 的
第 0–2 层，比较 resident 与 ECHO offload 的完整阶段延迟、逐算子利用率和非矩阵
开销。两种模式使用相同输入和矩阵后端，hidden/residual 依次穿过三层，执行
embedding、三个 dense MLP、final norm 和末 token LM head。

这是一段真实模型计算，结果仅覆盖所测三层。它不代表完整 61 层、GR serving
或推荐任务质量，也不能用来验收多用户 cache 的实际硬预算。

## 工作负载与测量边界

Prefix=65,536，extend=1,024，prefill chunk=1,024，每层可用 HBM pool=16,384 tokens，
host arena=66,560 tokens。主 KV 为 BF16 512 latent + 64 RoPE，indexer FP8
K/scales 留在 HBM。权重和普通 activation 单独统计，不计入 cache 容量。
每种模式从独立空 cache 构建 prefix；extend 重复之间恢复同一 prefix residency。
选择超过工作集容量时保留完整逻辑选择，按 query 拆分消费。

本次使用 GPU1 H200 SXM / SM90，CPU16–23，NUMA0。每模式 prefix 预热 1 次、
正式测量 3 次，extend 预热 1 次、正式测量 5 次。check、bench 和 profile 在独立
进程中运行。bench 先核验匹配源码、输入、依赖与执行路径的数值验收记录，正式
样本之间不重跑完整参考比较或保存输出；加载、编译、快照恢复和数值诊断不计入
同步 wall time。运行时必须的事务处理与同步仍保留在各自执行契约内。

`profile_layers.sh` 采集 resident/offload 的 prefix/extend 四段独立 annotated
profile。逐算子数据按最内层 scope 归因，父 scope 仅保留未归入子算子的活动。
Kernel、memcpy、CPU API、NVTX host 区间及正式 wall time 各自报告，不能相加
作为端到端分解。每组 profile 只有一次抓取，算子利用率没有重复采样置信区间。

运行依赖为 PyTorch 2.12.1+cu130、Triton 3.7.1、safetensors 0.8.0、apache-tvm-ffi 0.1.13.post3。DeepGEMM、FlashMLA、FlashInfer 与实际 JIT/native 身份见 [summary.json](report/layers3/summary.json)。checkpoint 身份包含配置及 shard stat 清单，未 hash 全部权重。本实验记录指定 pool/arena 的执行，未运行多用户 NH 填满轨迹；这些容量参数不能当作完整 serving 的物理硬预算验收。

## 当前验收与结果

| 用途 | Run ID |
| --- | --- |
| 独立正确性 | `refactor_three_layers_check_20261005_02` |
| 正式计时 | `refactor_three_layers_bench_20261005_01` |
| 独立 profile | `refactor_three_layers_profile_20261005_01` |

独立 check 的五组完整 hidden/logit 对照全部逐位一致，最大绝对误差与相对 L2
均为零。验收记录位于 `/tmp/cxldsagr_three_layer_refactor_20261005/check_02/receipt.json`，
canonical SHA256 为
`9ac8ab2d3b695d32e9fd81b269ca4f54130023fab4f63ce89df6ee077c18f9a5`。
这是三层真实传播验收，不是 C10 请求回放或完整 61 层验收。

bench 与 profile 分别以退出码 0 完成，并通过来源、数值记录与阶段覆盖验收。bench result SHA256 为
`a625e71e78bf54734b6a6335b35f54e1e3c904f854cc75c0797438acbdcacce6`，profile result SHA256 为
`9113543b627e5eb4be414c5bff722a24e166066a1c0f25418ac93b4fae0efb3b`。这两个哈希标识 result 文件，源码集合另列于 summary。

bench/profile 各有 11/16 次全 GPU 进程观测，最大采样间隔分别为 5.121/5.490 秒，未观察到外来进程或监测错误。离散观测不排除采样间隙中的工作；完整命令、时间区间及原始证据哈希见 [运行验收](report/layers3/run_acceptance.json)。

在当前 H64K+A1K、每层 16K slots 设置下，offload 的 prefix 和 extend 中位耗时均高于 resident。当前 profile 将 offload prefix 中的 resident logits 与融合 indexer/prefetch 分行报告：两类调用均实际发生，覆盖检查确认每个 query 恰好处理一次。不将融合调用时间全部解释为矩阵计算，也不由单次 profile 推断改动前后的因果差异。

本次未采集 NCU replay；当前报告不包含 NCU full/source 指标或验证结论。
逐算子 FLOPs 使用实际调用账本，FMA 计 2 FLOPs；精度分别对应 FP8/BF16/FP32
dense peak。非矩阵算子的 FLOPs 与利用率为 N/A。FP8Linear 的 kernel 分母包含
量化和 GEMM，indexer 包含其 API 的辅助 kernel；offload indexer 还包含融合
prefetch，因此这些数据不能解释为单 GEMM 效率。

端到端精度归一化利用率用完整三层 annotated 账本中的 useful matrix FLOPs，按
精度换算理想计算时间，再除以匹配独立 bench 的同步 wall time。分母包含 CPU
调度、非矩阵计算、搬运、等待和 launch gap。它描述本实验所测阶段，与单一
峰值 MFU 或 Tensor pipe active 含义不同。

<!-- BEGIN LAYERS3 PROFILE RESULTS -->

| 阶段 | Resident 中位延迟 ms | Offload 中位延迟 ms | Resident 端到端利用率 | Offload 端到端利用率 |
| --- | ---: | ---: | ---: | ---: |
| prefix | 642.340 | 1190.485 | 45.08% | 24.32% |
| extend | 13.954 | 27.632 | 38.70% | 19.54% |

![三层阶段延迟](report/layers3/latency.svg)

![三层逐算子利用率](report/layers3/mfu.svg)

逐算子表、非矩阵活动及完整定义见 [结果](report/layers3/results.md)。61 条矩阵汇总与 101 条非矩阵汇总分别保存在 [operator_mfu.csv](report/layers3/operator_mfu.csv)和[nonmatrix.csv](report/layers3/nonmatrix.csv)。

<!-- END LAYERS3 PROFILE RESULTS -->

原始运行资料位于
`experiments/deepseek_v32_echo_prefill/output/{data,log,profile}/<run_id>/`。
选出的图表、CSV、结果与来源记录保存在 [report/layers3](report/layers3/results.md)。
其中 [postrun_audit.json](report/layers3/postrun_audit.json) 重算保存的 extend
hidden/logit、账本及汇总守恒；它不独立解析原生 trace，也不代替源码与 native
身份核验。Prefix 数值比较在实际运行中完成，未保存的 prefix control tensor
不属于该离线审计范围。

## 运行方式与调用模块

从仓库根目录运行，依赖准备见 [3rdparty](../../3rdparty/README.md)。从同一源码、
checkpoint、GPU、CPU 亲和性和运行环境依次运行，例如：

```bash
CUDA_VISIBLE_DEVICES=5 ECHO_RUN_ID=my_layers3_check \
  bash experiments/deepseek_v32_echo_prefill/scripts/run.sh --mode check \
  --physical-device 5 --model /preset-models

CUDA_VISIBLE_DEVICES=5 ECHO_RUN_ID=my_layers3_bench \
  bash experiments/deepseek_v32_echo_prefill/scripts/run.sh --mode bench \
  --physical-device 5 --model /preset-models \
  --validation-receipt /tmp/cxldsagr-checks/deepseek_v32_echo_prefill/data/my_layers3_check/receipt.json

CUDA_VISIBLE_DEVICES=5 ECHO_RUN_ID=my_layers3_profile \
  bash experiments/deepseek_v32_echo_prefill/scripts/profile_layers.sh \
  --physical-device 5 --model /preset-models \
  --validation-receipt /tmp/cxldsagr-checks/deepseek_v32_echo_prefill/data/my_layers3_check/receipt.json \
  --benchmark-run experiments/deepseek_v32_echo_prefill/output/data/my_layers3_bench
```

check 默认保存在 `${TMPDIR:-/tmp}/cxldsagr-checks/deepseek_v32_echo_prefill/`。
bench/profile 使用独立的新 run ID；正式报告必须绑定匹配的 bench，不能用
profile wall time 代替正式计时。测量身份、样本、完整命令和依赖以各 run 的
保存记录为准。

`src.profile_layers` 调用 `models/deepseek_v32/` 的 model、layers、projections、
rotary 与 attention，以及共享 cache。算子入口包括 `operators/flashinfer.py`、
`operators/deepseek_v32/indexer/{echo,selection,quantization}.py`、
`attention/device_only/mla.py` 和 `linear/fp8.py`。模型的具体执行与支持条件见
[模型说明](../../models/deepseek_v32/README.md)。

`src.operator_instrumentation` / `src.operator_report` 生成调用账本和 NSYS 分解，
`src.backend_provenance` 记录依赖及实际 FlashInfer JIT 身份。
`src.postrun_audit` 核对保存输出和报告算术，`src.publish_layers` 验证来源并生成
图表与报告。独立计时、数值与利用率的绑定方式见 `src.run_contract`。

十 block GR 工作负载的固定 P/NH 容量实验见
[统一 cache management 实验](../cache_management/README.md)。它使用不同工作负载和容量边界，
结果不能与本页三层固定 chunk 测量互换。

当前选定资料及 README 的哈希见 [发布清单](report/layers3/publication_manifest.json)。测量时源码、实际报告生成器及稍后采集的已加载辅助模块分别记录在 [来源绑定](report/layers3/source_bindings.json)。辅助源码快照保留在对应 `output/data/` 目录，不替代原运行身份。

新结果验收后，已清理受本次实现改动影响的旧三层计时、profile、NCU 与报告素材。本轮不沿用旧 NCU 或完整模型性能结论。
