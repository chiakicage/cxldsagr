# DeepSeek V3.2 / SM120 历史实验归档

## 实验目的与内容

整体归档旧 DeepSeek V3.2 attention decode/extend、DeepGEMM、GR checkpoint/indexer、
KV 命中及 offloading 预算实验，以及其 SM120/RTX 5080 的配套硬件测量报告。
旧目录内部不再按新模板拆分：根目录保留脚本与实验说明，`tests/` 保留测试，`docs/` 保留历史报告/图表。
模型实现仍在 `models/deepseek_v32/`，独立算子工程仍在 `operators/sm120/`。

## 当前运行方式与调用模块

从仓库根目录运行，新模块前缀为 `experiments.legacy.deepseek_v32`：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync --group sm120 --group analysis
.venv/bin/python -m experiments.legacy.deepseek_v32.sweep_gr_content_matrix --help
.venv/bin/python -m experiments.legacy.deepseek_v32.measure_gr_multilayer_hits --help
.venv/bin/python -m experiments.legacy.deepseek_v32.deepgemm_v32_benchmark --quick
.venv/bin/python -m pytest experiments/legacy/deepseek_v32/tests -q
```

- `measure_gr_*` / `validate_gr_numerics` 调用 `models.deepseek_v32` 的模型、算子适配和数学参考，
  读取 `weights/DeepSeek-V3.2/`；生成请求时显式选择 GR 的 `deepseek_v32` 格式。
- `sweep_gr_*` 用新的 `-m experiments.legacy.deepseek_v32.<module>` 子进程入口调度。
- `profile_deepseek_v32_*` 调用 DeepSeek runner、DeepGEMM 与 SM120 sparse MLA 扩展。
- `report_gr_*` / `export_gr_kv_hits` / `analyze_gr_kv_coverage` 读取旧 GR 结果，整理表格与可视化。
- `deepgemm_v32_benchmark.py` 是从 `operators/sm120/benchmarks/` 移入的历史实验脚本。

归档脚本仍保留原有数据布局；报告写入本归档 `docs/`，共享历史 GR 产物路径不重新组织。
原始历史报告内的命令记录当时的位置，不是当前入口；当前入口以上方示例为准。
缺少 SM120 扩展、对应 GPU、配置或 checkpoint 时不能运行 GPU 验证；迁移未重新测量。
本次环境没有安装 `deep_gemm`，相关 CLI 和两份 legacy 测试在导入依赖时受阻；
已静态检查新的模块导入、子进程入口和报告导航，不将其记作 GPU 验证通过。

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
