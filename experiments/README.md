# 实验索引

本目录保存为论文服务的实验，实验目的、测量边界、运行方式、结果和结论见各实验 README。
正确性回归通过 `bash scripts/run_tests.sh` 运行，使用方式见[项目 README](../README.md)。

| 实验 | 内容与状态 | 仓库根目录运行入口 |
| --- | --- | --- |
| [GR serving cache](gr_cache_serving/README.md) | 真实五层 DSA 固定预算、每规模 512 请求的四路径对照；与下列 dense 层替身实验独立 | `bash experiments/gr_cache_serving/scripts/run_replay.sh --help` |
| [GR 多用户 serving](gr_serving/README.md) | DeepSeek 约 8B dense 层替身与 NOSA-8B；ECHO MFU/cache 策略与 DeepSeek MFU 待修正，相关 baseline 比较暂停采用；用户规模与复访压力也待补测 | `bash experiments/gr_serving/scripts/run.sh <run_id>` |
| [CPU DRAM 带宽](cpu_dram_bandwidth/README.md) | 双路 Xeon 8558P / DDR5-4400；理论 563.2 GB/s，当前容器 IMC 顺序读约 428 GB/s、应用约 435 GB/s，含 NUMA 与线程扫描 | `bash experiments/cpu_dram_bandwidth/scripts/run.sh <run_id> --imc` |
| [NOSA indexer + block sparse profile](indexer_block_sparse_profile/README.md) | H200 BF16 64K+1K；BF16-pair / FA3 v3 的 native/Triton 端到端与模块 kernel MFU 已于 2026-09-29 补测 | `bash experiments/indexer_block_sparse_profile/scripts/run.sh <run_id>` |
| [NOSA sparse fetch + attention overlap](nosa_offload_overlap/README.md) | 单主 kernel 内 stripe fetch 只读一次 host 并与 attention 重叠；两次 64K + 1K 单层回放均快于整批稀疏串行对照；全部 profile 样本的 page/stripe 两项 softmax overlap ≥90%；完整模型数值检查通过 | `bash experiments/nosa_offload_overlap/scripts/run.sh <run_id>` |
| [NOSA kernel MFU](nosa_kernel_mfu/README.md) | H200 synthetic resident attention / pooled score 已测量；本次均未达到 30% 有效 MFU | `bash experiments/nosa_kernel_mfu/scripts/run.sh <run_id> --peak-tflops 989` |
| [NOSA GR 65536 + 1024](nosa_gr_65536_1024/README.md) | H200 dense 前向、模块 MFU 与 Nsight 活动分析已补测；计时边界见报告 | `bash experiments/nosa_gr_65536_1024/scripts/run.sh <run_id>` |
| [NOSA indexer pattern 65536 + 1024](nosa_indexer_pattern_65536_1024/README.md) | H200 完整 NOSA 三组对照与 QA32 已补测，QA64 已复核；容量、传输与 overlap 离线分析已更新 | `bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh <run_id>` |
| [DeepSeek V3.2 ECHO prefill/extend](deepseek_v32_echo_prefill/README.md) | 保留真实前 3 层 64K + 1K 的 nsys/NCU 记录；MFU、cache 策略及性能归因待修正复核，完整 61 层性能仍待补测 | `bash experiments/deepseek_v32_echo_prefill/scripts/profile_layers.sh --help`；完整模型用 `scripts/run.sh` |
| [旧 DeepSeek / SM120](legacy/deepseek_v32/README.md) | 有效历史报告、CPU 重建工具与独立 DeepGEMM 基准；SM120 执行入口已移除 | 见归档 README |

```bash
bash experiments/nosa_gr_65536_1024/scripts/run.sh --help
```

新运行的 stdout/stderr 在 `output/log/<run_id>/`，原始与整理后的数据在
`output/data/<run_id>/`，nsys / Chrome trace 在 `output/profile/<run_id>/`。
各实验的 `report/` 保存报告引用的图片、表格和数据，随 Git 维护。

环境准备见 [项目 README](../README.md)。模型推理见 [models](../models/README.md)，
共享请求和热度资源见 [GR](../GR/README.md)，目录维护规则见 [AGENTS.md](../AGENTS.md)。
