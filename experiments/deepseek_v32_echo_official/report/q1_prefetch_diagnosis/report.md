# ECHO Q1 冷态预取的尾部开销

相同 L2 输入和正零阈值下，完全驻留对照消除了冷态融合核的大部分耗时与最慢 SM 的长尾。
额外开销位于 miss、额度预约和预取搬运这条路径；本次对照同时去掉预约与搬运，不能
单独归因于原子操作，也不能把差值称为可获得的优化收益。

| 指标 | 冷态 | 完全驻留 |
| --- | ---: | ---: |
| NCU core duration (us) | 55.008 | 10.880 |
| Mean SM active cycles | 11,937.689 | 11,508.523 |
| Maximum SM active cycles | 78,125.000 | 12,448.000 |
| Atomic ALU instructions | 1,048.000 | 0.000 |
| L1 global atomic requests | 1,048.000 | 0.000 |
| TEX sysmem read miss sectors | 2,304.000 | 0.000 |

冷态在应用可见的完成状态中尝试预约 33,464 条记录，实际暂存 64 条，共 73,728 B；
完全驻留时三者均为 0。两侧 DRAM 读取量分别为 8,996,352 / 9,251,072 B，
寄存器 spill 的 local-memory 请求均为 9,884。平均 SM 活跃周期接近，而最大值相差明显，
支持预取额外工作存在长尾的判断。表中 NCU 聚合指标均明确返回 `has_value=true`。
完整数值与指标名称见 [metrics.csv](metrics.csv)，身份、来源哈希和可用性见
[evidence.json](evidence.json) 与 [provenance.json](provenance.json)。

## 官方源码与可观测行为

固定版本 ECHO 已在 warp 内聚合 miss mask，再由一个线程调用 `atomicAdd`。
预约的记录数不能当成原子指令数。成功取得额度后，每个 warp 在循环中逐条搬运其记录。
单 pass pilot 的实际 ID 为 14944–14975 和 14912–14943；冷态 full 的最终可见 ID 为
28736–28767 和 28672–28703。两次都是同一 256-token task 的两个完整 32-record
组，结合源码可知每个获胜 warp 串行处理 32 条记录。这是集中搬运工作的证据，
尚不足以证明它独自造成全部尾部延迟。

源码固定为 `bc1b75c1000010d0ac6f032ebaac283255c050b1`：
[warp 预约与搬运循环](https://github.com/sjtu-zhao-lab/ECHO/blob/bc1b75c1000010d0ac6f032ebaac283255c050b1/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh#L788)，
[1152 B record 搬运](https://github.com/sjtu-zhao-lab/ECHO/blob/bc1b75c1000010d0ac6f032ebaac283255c050b1/DeepGEMM/deep_gemm/include/deep_gemm/common/utils.cuh#L335)。
本次没有修改官方 kernel。

## 测量与验证边界

Run ID：`q1_official_core_ncu_20261008_01`。GPU 为 NVIDIA H200 / SM90，物理 GPU1，
CPU affinity 为 `[8, 9, 10, 11, 12, 13, 14, 15]`；Torch `2.12.1+cu130`、
CUDA `13.0`、Triton `3.7.1`。使用真实 checkpoint L2 的
Q1/H64/D128 FP8 Q/K、FP32 scales/weights 和 BF16 D576 host records，H=65,536，
N=65,537。两侧 query、key、scale、weight 的保存文件身份相同，阈值 bits 均为 0。
输入属于独立组件数据，不等同于官方 SGLang 的实际 decode 输入与自然驻留状态。

NCU 2026.1.1 使用 kernel replay、cache flush 和 base clock control。先运行一个
冷态单硬件 pass pilot，再分别采集冷态和完全驻留的 full/source，每项 45 passes。
未采集 L0/L1。该时长仅覆盖官方融合 core，是侵入式 profile；打包、promotion、
top-k、精确 recall 和完整模型延迟均不在此时长内。

实际官方 ELF SHA256 为 `f67b4dffe705a90803ebdfb61ed15fb449aebe030dd49409d5ab8c15a53f9998`。
独立组件验收签名为 `d7d9a19db115369a9d9bd025f71e5090896f21aa7adb1eb7d695f29b1a25549e`，验收路径为
`/tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02/receipt.json`。
采集委托给已验收 baseline，核验原输入、源码/native、NVTX、唯一 kernel 和 launch
geometry；读回在 ProfilerStop 后完成。NCU 在 replay 间恢复设备写入，但合法预约
次序仍可变化，最终读回不能代表每个内部 pass 的实际预取集合。

原始 per-PC 与 PM 数组虽有非零数值，其实例可用性标志为 false；因此未用这些数值
推算源码行的耗时比例或绘制有效 PM 时间线。warm 的 1.5 µs 采样间隔还触发了超过
workload 时长 10% 的警告。六项 CTC 指标不可用，不能记为 0；全部警告保留在原日志。
原始数据、日志和 profiler 文件分别位于本实验 `output/data/`、`output/log/`、
`output/profile/` 下的同名 run ID 目录。

采集入口为 `scripts/profile_q1_official_core.sh`，解析入口为
`src/analyze_q1_official_core.py` 和 `src/report_q1_official_core.py`。
本报告由 `src/publish_q1_prefetch_diagnosis.py --run-dir <原 run 目录>
--output-dir <新报告目录>` 生成；按实验约定用 `python -m` 运行。
