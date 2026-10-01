# DeepSeek V3.2 / SM120 历史实验归档

## 实验目的与内容

保留旧 DeepSeek V3.2 attention decode/extend、DeepGEMM、GR checkpoint/indexer、
KV 命中及 offloading 预算实验，以及 SM120/RTX 5080 的有效历史报告、图表和数据。
报告内容与原始测量命令保持原样。项目自有 SM120 算子、旧模型实现、测量调度脚本
及其测试已删除；本目录只保留无需旧模型的 CPU 数据分析工具和独立 DeepGEMM 基准。

## 历史结果复现

旧 SM120 模型、扩展及完整实验入口保存在整理前 revision
`397e64530c976a64f39afb742ead0e8913b0736e`（原 SM120 代码基线为 `639cc96`）。
复现历史 GPU 测量须在该 revision 的独立 checkout 中，按照当时 README 准备依赖、
对应 GPU、配置、checkpoint 和输入数据。报告中的旧路径及命令只说明当时运行方式，
不能作为当前工作树的入口。此次目录清理没有重新测量，也不改变历史结果的实现版本。

## 当前保留工具与调用模块

从仓库根目录运行，CPU 报告工具读取原有 `GR/generated/` 或归档数据，
不执行 SM120 模型。安装绘图依赖后可检查入口：

```bash
uv sync --group analysis
.venv/bin/python -m experiments.legacy.deepseek_v32.report_gr_mla_cache --help
.venv/bin/python -m experiments.legacy.deepseek_v32.report_gr_content_matrix --help
.venv/bin/python -m experiments.legacy.deepseek_v32.export_gr_kv_hits --help
.venv/bin/python -m experiments.legacy.deepseek_v32.render_extend_timeline --help
```

- `report_gr_*` / `export_gr_kv_hits` / `analyze_gr_kv_coverage` 读取旧 GR 结果，
  整理表格与可视化；`render_extend_timeline` 绘制已有 profile 数据。
- `report_gr_mla_cache` 保留原汇总计算，已移除 GPU 测量选项；其常量供其他报告工具复用。
- `deepgemm_v32_benchmark.py` 独立调用共享 DeepGEMM，不依赖已删除的 SM120 扩展。
  该工具需要本地 `docs/config.json` 及对应 GPU/工具链；先执行
  `python3 scripts/prepare_3rdparty.py --init` 和 `uv sync --group legacy`，再运行
  `.venv/bin/python -m experiments.legacy.deepseek_v32.deepgemm_v32_benchmark --quick`。
  当前 DeepGEMM 更新尚未完成 GPU 复测，不能把历史报告视为当前版本结果。

保留工具沿用原数据布局及报告生成语义；本次仅检查导入与 CLI，不生成或覆盖历史报告。
没有恢复历史输入数据时，`--help` 成功也不代表能够重建完整报告。

## 实验结果与结论索引

- [decode](deepseek_v32_decode.md)、[extend](deepseek_v32_extend.md)：synthetic attention 测量和执行路径。
- [checkpoint 说明](deepseek_v32_two_dense.md)：embedding + 两层 dense 权重准备。
- [DeepGEMM 5080 结果](docs/deepgemm_v32_5080_results.md)、[指令微基准](docs/5080_microbench.md)。
- [extend 计算与精度](docs/extend_step_profile/extend_compute_zh.md)：逐模块 FLOPs、精度、GPU 时间与利用率。
- [GR 多层 KV 命中](docs/extend_step_profile/gr_multilayer_kv_hits.md)、
  [offload 数值验证](docs/extend_step_profile/gr_multilayer_offload_validation.md)：内容、层间访问和缓存预算。
- [CPU 带宽](docs/cpu_memory_bandwidth_results.md)、[PCIe 根因](docs/gpu_cpu_pcie_bandwidth_root_cause.md)、
  [GPU 直接读 host](docs/gpu_host_direct_ld_results.md)：当时硬件的数据访问结果。

各报告的平台、精度、结果和结论保持原义；它们不证明当前 NOSA/SM90 的稀疏或 offloading 路径已实现。
