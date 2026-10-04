# 实验索引

本目录保存为论文服务的实验，实验目的、测量边界、运行方式、结果和结论见各实验 README。
正确性回归通过 `bash scripts/run_tests.sh` 运行，使用方式见[项目 README](../README.md)。

| 实验 | 内容与状态 | 仓库根目录运行入口 |
| --- | --- | --- |
| [GR 多用户 serving](gr_serving/README.md) | 保留 NOSA 原 4K/16K/64K 短轨迹、run ID 与测量边界；旧 DeepSeek 子集已清理 | `bash experiments/gr_serving/scripts/run.sh <run_id>` |
| [DeepSeek V3.2 Motivation](deepseek_v32_motivation/README.md) | C10：16 用户两轮，P=65,536、NH=16,777,216；四方案端到端测量、全部输出逐位检查及匹配 profile 已验收 | `bash experiments/deepseek_v32_motivation/scripts/run.sh --help` |
| [DeepSeek V3.2 官方 ECHO 适配](deepseek_v32_echo_official/README.md) | C10：相同模型与 P/NH 的官方适配路径；复访 45.561 ms，两轮 74.930 s；数值验收通过，并附本地 C10 对照 | `bash experiments/deepseek_v32_echo_official/scripts/run.sh --help` |
| [DeepSeek V3.2 ECHO 固定 P/NH 容量](deepseek_v32_echo_cache/README.md) | NH 只计 history；原源码静态规划给出 512 GiB DRAM 下的 455 用户边界，未跑满容量；当前 metadata 改动后尚未重新规划 | `bash experiments/deepseek_v32_echo_cache/scripts/run.sh --help` |
| [CPU DRAM 带宽](cpu_dram_bandwidth/README.md) | 双路 Xeon 8558P / DDR5-4400；理论 563.2 GB/s，当前容器 IMC 顺序读约 428 GB/s、应用约 435 GB/s，含 NUMA 与线程扫描 | `bash experiments/cpu_dram_bandwidth/scripts/run.sh <run_id> --imc` |
| [NOSA indexer + block sparse profile](indexer_block_sparse_profile/README.md) | H200 BF16 64K+1K；BF16-pair / FA3 v3 的 native/Triton 端到端与模块 kernel MFU 已于 2026-09-29 补测 | `bash experiments/indexer_block_sparse_profile/scripts/run.sh <run_id>` |
| [NOSA sparse fetch + attention overlap](nosa_offload_overlap/README.md) | 单主 kernel 内 stripe fetch 只读一次 host 并与 attention 重叠；两次 64K + 1K 单层回放均快于整批稀疏串行对照；全部 profile 样本的 page/stripe 两项 softmax overlap ≥90%；完整模型数值检查通过 | `bash experiments/nosa_offload_overlap/scripts/run.sh <run_id>` |
| [NOSA kernel MFU](nosa_kernel_mfu/README.md) | H200 synthetic resident attention / pooled score 已测量；本次均未达到 30% 有效 MFU | `bash experiments/nosa_kernel_mfu/scripts/run.sh <run_id> --peak-tflops 989` |
| [NOSA GR 65536 + 1024](nosa_gr_65536_1024/README.md) | H200 dense 前向、模块 MFU 与 Nsight 活动分析已补测；计时边界见报告 | `bash experiments/nosa_gr_65536_1024/scripts/run.sh <run_id>` |
| [NOSA indexer pattern 65536 + 1024](nosa_indexer_pattern_65536_1024/README.md) | H200 完整 NOSA 三组对照与 QA32 已补测，QA64 已复核；容量、传输与 overlap 离线分析已更新 | `bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh <run_id>` |
| [DeepSeek V3.2 ECHO prefill/extend](deepseek_v32_echo_prefill/README.md) | 真实 checkpoint 第 0–2 层顺序传播；64K+1K 计算优化对照、MFU 与 NCU，保留跨版本数值差异 | `bash experiments/deepseek_v32_echo_prefill/scripts/run.sh --help`；详细 profile 用 `scripts/profile_layers.sh` |
| [旧 DeepSeek / SM120](legacy/deepseek_v32/README.md) | 有效历史报告、CPU 重建工具与独立 DeepGEMM 基准；SM120 执行入口已移除 | 见归档 README |

```bash
bash experiments/nosa_gr_65536_1024/scripts/run.sh --help
```

新运行的 stdout/stderr 在 `output/log/<run_id>/`，原始与整理后的数据在
`output/data/<run_id>/`，nsys / Chrome trace 在 `output/profile/<run_id>/`。
各实验的 `report/` 保存报告引用的图片、表格和数据，随 Git 维护。

环境准备见 [项目 README](../README.md)。模型推理见 [models](../models/README.md)，
共享请求和热度资源见 [GR](../GR/README.md)，目录维护规则见 [AGENTS.md](../AGENTS.md)。
