# 实验索引

本目录保存为论文服务的实验，实验目的、测量边界、运行方式、结果和结论见各实验 README。
正确性回归通过 `bash scripts/run_tests.sh` 运行，使用方式见[项目 README](../README.md)。

| 实验 | 内容与状态 | 仓库根目录运行入口 |
| --- | --- | --- |
| [CPU DRAM 带宽](cpu_dram_bandwidth/README.md) | 双路 Xeon 8558P / DDR5-4400；理论 563.2 GB/s，当前容器 IMC 顺序读约 428 GB/s、应用约 435 GB/s，含 NUMA 与线程扫描 | `bash experiments/cpu_dram_bandwidth/scripts/run.sh <run_id> --imc` |
| [NOSA indexer + block sparse profile](indexer_block_sparse_profile/README.md) | 增量压缩缓存 + FlashInfer Top-K；H200 BF16 64K+1K 无插桩 full-prefill 3.698 s、extend 62.034 ms，含模块 kernel MFU | `bash experiments/indexer_block_sparse_profile/scripts/run.sh <run_id>` |
| [NOSA GR 65536 + 1024](nosa_gr_65536_1024/README.md) | dense 单请求前向与模块 MFU；共享 cache 改动后未运行 | `bash experiments/nosa_gr_65536_1024/scripts/run.sh <run_id>` |
| [NOSA indexer pattern 65536 + 1024](nosa_indexer_pattern_65536_1024/README.md) | 完整 NOSA 三组对照改动后未运行；保留独立 QA-only FP32 选块原始数据 | `bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh <run_id>` |
| [旧 DeepSeek / SM120](legacy/deepseek_v32/README.md) | 旧脚本、测试和历史报告整体归档 | 见归档 README |

```bash
bash experiments/nosa_gr_65536_1024/scripts/run.sh --help
```

新运行的 stdout/stderr 在 `output/log/<run_id>/`，原始与整理后的数据在
`output/data/<run_id>/`，nsys / Chrome trace 在 `output/profile/<run_id>/`。
各实验的 `report/` 保存报告引用的图片、表格和数据，随 Git 维护。

环境准备见 [项目 README](../README.md)。模型推理见 [models](../models/README.md)，
共享请求和热度资源见 [GR](../GR/README.md)，目录维护规则见 [AGENTS.md](../AGENTS.md)。
