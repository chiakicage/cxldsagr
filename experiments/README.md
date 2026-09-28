# 实验索引

本目录保存为论文服务的实验，实验目的、测量边界、运行方式、结果和结论见各实验 README。
正确性回归通过 `bash scripts/run_tests.sh` 运行，使用方式见[项目 README](../README.md)。

| 实验 | 内容与状态 | 仓库根目录运行入口 |
| --- | --- | --- |
| [CPU DRAM 带宽](cpu_dram_bandwidth/README.md) | 双路 Xeon 8558P / DDR5-4400；理论 563.2 GB/s，当前容器 IMC 顺序读约 428 GB/s、应用约 435 GB/s，含 NUMA 与线程扫描 | `bash experiments/cpu_dram_bandwidth/scripts/run.sh <run_id> --imc` |
| [NOSA indexer + block sparse profile](indexer_block_sparse_profile/README.md) | H200 BF16 64K+1K native/Triton 端到端与模块 kernel MFU 已补测 | `bash experiments/indexer_block_sparse_profile/scripts/run.sh <run_id>` |
| [NOSA kernel MFU](nosa_kernel_mfu/README.md) | H200 synthetic resident attention / pooled score 已测量；本次均未达到 30% 有效 MFU | `bash experiments/nosa_kernel_mfu/scripts/run.sh <run_id> --peak-tflops 989` |
| [NOSA GR 65536 + 1024](nosa_gr_65536_1024/README.md) | H200 dense 前向、模块 MFU 与 Nsight 活动分析已补测；计时边界见报告 | `bash experiments/nosa_gr_65536_1024/scripts/run.sh <run_id>` |
| [NOSA indexer pattern 65536 + 1024](nosa_indexer_pattern_65536_1024/README.md) | H200 完整 NOSA 三组对照与 QA32 已补测，QA64 已复核；容量、传输与 overlap 离线分析已更新 | `bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh <run_id>` |
| [旧 DeepSeek / SM120](legacy/deepseek_v32/README.md) | 旧脚本、测试和历史报告整体归档 | 见归档 README |

```bash
bash experiments/nosa_gr_65536_1024/scripts/run.sh --help
```

新运行的 stdout/stderr 在 `output/log/<run_id>/`，原始与整理后的数据在
`output/data/<run_id>/`，nsys / Chrome trace 在 `output/profile/<run_id>/`。
各实验的 `report/` 保存报告引用的图片、表格和数据，随 Git 维护。

环境准备见 [项目 README](../README.md)。模型推理见 [models](../models/README.md)，
共享请求和热度资源见 [GR](../GR/README.md)，目录维护规则见 [AGENTS.md](../AGENTS.md)。
