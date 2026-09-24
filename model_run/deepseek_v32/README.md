# DeepSeek V3.2

现有实现以 SM120 实验为基础，使用 DeepGEMM、`flash_mla_sm120` 和 FlashInfer。
Synthetic decode/extend 的张量驻留 GPU；GR 实验运行部分 checkpoint 层，提取稀疏索引，
分析 KV 命中与理想传输成本。DeepGEMM 切换 `nv_dev` 后尚未重新验证模型运行。

| 入口 | 用途 |
| --- | --- |
| [decode](deepseek_v32_decode.md)、[extend](deepseek_v32_extend.md) | attention benchmark 与执行流程 |
| `profile_deepseek_v32_*.py`、`render_extend_timeline.py` | profile 与时间线 |
| `measure_gr_*.py`、`sweep_gr_*.py` | checkpoint 层与稀疏索引实验 |
| `report_gr_*.py`、`export_gr_kv_hits.py`、`analyze_gr_kv_coverage.py` | 实验汇总、KV 命中分布与回放输入 |
| [tests/](tests/) | 模型数学、cache/indexer 与 extend 验证 |

从仓库根目录运行；第三方源码准备见 [依赖说明](../../3rdparty/README.md)：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync --group sm120 --group analysis
.venv/bin/python model_run/deepseek_v32/deepseek_v32_decode.py --quick
.venv/bin/python model_run/deepseek_v32/deepseek_v32_extend.py
.venv/bin/python -m model_run.deepseek_v32.sweep_gr_content_matrix
.venv/bin/python -m model_run.deepseek_v32.report_gr_content_matrix
.venv/bin/python -m pytest model_run/deepseek_v32/tests -q
```

Synthetic benchmark 默认读取 `docs/config.json`，需准备 DeepSeek V3.2 的 attention /
indexer 配置；GR 实验默认从 `models/DeepSeek-V3.2/` 读取权重与 tokenizer，见
[权重说明](deepseek_v32_two_dense.md)。实验数据输出到 `GR/generated/`，报告输出到 `docs/`。
