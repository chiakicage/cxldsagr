# 实验索引

每个实验的目的、内容、调用模块、运行方式、结果和结论均在其 README 中。
源码在 `src/`，运行脚本在 `scripts/`，单元测试在 `tests/`；运行产物统一归 `output/`，默认不进 Git。

| 实验 | 内容与状态 | 仓库根目录运行入口 |
| --- | --- | --- |
| [NOSA GR 65536 + 1024](nosa_gr_65536_1024/README.md) | instruction + 历史 65536、候选 1024；nsys / 模块 MFU | `bash experiments/nosa_gr_65536_1024/scripts/run.sh <run_id>` |
| [NOSA framework 重构验证](nosa_framework_refactor/README.md) | serving / executor / cache 分层与 profiler 接线；保留原 worktree 验证记录 | `bash experiments/nosa_framework_refactor/scripts/run.sh <run_id> cpu` |
| [旧 DeepSeek / SM120](legacy/deepseek_v32/README.md) | 旧脚本、测试和历史报告整体归档 | 见归档 README |
| [已撤回的 NOSA 多长度前向试测](legacy/nosa_gr_forward/README.md) | 保留 framework worktree 历史源码/报告，不恢复为当前有效结果 | 见归档 README |
| [已撤回的 NOSA 生成试测](legacy/nosa_generation/README.md) | 固定 64-token 生成不适用 GR，仅供追溯 | 见归档 README |

```bash
bash experiments/nosa_gr_65536_1024/scripts/run.sh --help
.venv/bin/python -m pytest experiments/nosa_gr_65536_1024/tests -q
bash experiments/nosa_framework_refactor/scripts/run.sh --help
```

新运行的 stdout/stderr 在 `output/log/<run_id>/`，原始与整理后的数据在
`output/data/<run_id>/`，nsys / Chrome trace 在 `output/profile/<run_id>/`。
旧运行使用 `20260925` 标识，修正语义的 64K+1K 运行使用 `semantic_65536_1024_20260925`；历史文件原内容未改写，旧路径到新路径的对应关系在各实验的
`output/data/migration.json`。未保存过的历史 stdout 不补造。

环境准备见 [项目 README](../README.md)。模型推理见 [models](../models/README.md)，
共享请求和热度资源见 [GR](../GR/README.md)，目录维护规则见 [AGENTS.md](../AGENTS.md)。
