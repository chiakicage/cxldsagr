# 实验

实验脚本、运行记录与实验文档。模型推理代码见 [`models/`](../models/README.md)，
历史测量报告与性能记录仍在 [`docs/`](../docs/)。

| 入口 | 用途 |
| --- | --- |
| [deepseek_v32_decode.md](deepseek_v32_decode.md)、[deepseek_v32_extend.md](deepseek_v32_extend.md) | synthetic attention benchmark、执行流程与测量结果 |
| `profile_deepseek_v32_*.py`、`render_extend_timeline.py` | profile 与时间线渲染 |
| `measure_gr_*.py`、`sweep_gr_*.py` | checkpoint 层与稀疏索引实验 |
| `report_gr_*.py`、`export_gr_kv_hits.py`、`analyze_gr_kv_coverage.py` | 实验汇总、KV 命中分布与回放输入导出 |
| `validate_gr_numerics.py`、[tests/](tests/) | GR 数值参考与 checkpoint 数学验证 |
| [deepseek_v32_two_dense.md](deepseek_v32_two_dense.md) | embedding + 两层 dense 实验权重说明 |

## 运行

从仓库根目录执行。跨目录脚本使用 `-m experiments.<脚本>`，无需安装仓库：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync --group sm120 --group analysis

.venv/bin/python -m experiments.sweep_gr_content_matrix
.venv/bin/python -m experiments.report_gr_content_matrix
.venv/bin/python -m experiments.measure_gr_mla_cache_union
.venv/bin/python -m pytest experiments/tests -q
```

逐步骤 profile 与时间线：

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m experiments.profile_deepseek_v32_extend \
  --history-lens 4096,65536 --new-tokens 4096 --chunk-size 512 --iters 5
.venv/bin/python -m experiments.render_extend_timeline
```

GR 实验默认从 `weights/DeepSeek-V3.2/` 读取权重与 tokenizer，见
[权重说明](deepseek_v32_two_dense.md)。实验数据输出到 `GR/generated/`（Git 忽略），
汇总报告输出到 `docs/`。
