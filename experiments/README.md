# 实验索引

本目录保存为论文服务的实验，实验目的、测量边界、运行方式、结果和结论见各实验 README。
正确性回归通过 `bash scripts/run_tests.sh` 运行，使用方式见[项目 README](../README.md)。

| 实验 | 内容与状态 | 仓库根目录运行入口 |
| --- | --- | --- |
| [CPU DRAM 带宽](cpu_dram_bandwidth/README.md) | 双路 Xeon 8558P / DDR5-4400；理论 563.2 GB/s，当前容器 IMC 顺序读约 428 GB/s、应用约 435 GB/s，含 NUMA 与线程扫描 | `bash experiments/cpu_dram_bandwidth/scripts/run.sh <run_id> --imc` |
| [NOSA indexer + block sparse profile](indexer_block_sparse_profile/README.md) | 完整 NOSA sparse 的 64K+1K 端到端：H200 BF16 full-prefill 10.580 s、prefix-ready extend 194.837 ms；含 CIS/indexer/attention 分解与 nsys | `bash experiments/indexer_block_sparse_profile/scripts/run.sh <run_id>` |
| [NOSA GR 65536 + 1024](nosa_gr_65536_1024/README.md) | instruction + 历史 65536、候选 1024；QKV/gate-up GEMM 合并与 FlashInfer 融合，已完成 GPU 时间、模块 MFU 与 launch 开销测量 | `bash experiments/nosa_gr_65536_1024/scripts/run.sh <run_id>` |
| [NOSA query-aware pattern 65536 + 1024](nosa_indexer_pattern_65536_1024/README.md) | dense 激活上比较 64/32 blocks（1 sink + 16 local + 47/15 query-aware）；并集占完整 KV 的 37.90%/25.32%，含逐层/head 结果、全部head占比分布图、固定 MFU、50 GB/s fetch、sink/local 与 QA 分项及30%分类overlap估算 | `bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh <run_id> --block-budget 32`（默认64）；离线计算用同目录 `scripts/estimate.sh` / `scripts/decompose.sh` |
| [旧 DeepSeek / SM120](legacy/deepseek_v32/README.md) | 旧脚本、测试和历史报告整体归档 | 见归档 README |

```bash
bash experiments/nosa_gr_65536_1024/scripts/run.sh --help
```

新运行的 stdout/stderr 在 `output/log/<run_id>/`，原始与整理后的数据在
`output/data/<run_id>/`，nsys / Chrome trace 在 `output/profile/<run_id>/`。
各实验的 `report/` 保存报告引用的图片、表格和数据，随 Git 维护。

环境准备见 [项目 README](../README.md)。模型推理见 [models](../models/README.md)，
共享请求和热度资源见 [GR](../GR/README.md)，目录维护规则见 [AGENTS.md](../AGENTS.md)。
