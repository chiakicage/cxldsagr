# V10 timeline：计算、IO、GPU 空闲与 ECHO 操作

最终两张图统一展示在[实验 README](../../../README.md)。Prefill 只取最后一个
chunk（第 64/64 个、1,024 tokens）的 L0–L2；extend 取 128-token batch 的 L0–L2。
Timeline 与 gap 均从 L0 首个计算 kernel 开始，到 L2 最后一个计算 kernel 结束。

计算集中在一行，只用颜色表示投影/RoPE、indexer/top-k、attention 和输出/MLP，
阶段名称放在图例中。H2D（DRAM→GPU）与 D2H（GPU→DRAM）分别标为橙色和紫色，
各占一行；融合计算与 IO 保留斜线。红色只表示没有任何 GPU 活动的空闲。
ECHO 另设一行，棕色、深灰色和绿色分别表示 prepare、finalize、hint 的 GPU 活动；
其他控制操作不着色。图标题报告 GPU idle，完整 gap 数值保留在实验 README 和窗口表中。
Prefill 最后一个 chunk 的 finalize 没有 GPU 活动，不画外层 CPU scope。

Gap 是上述窗口中计算与实际 IO 区间并集之外的时间。层内和层间未被覆盖的
metadata、D2D/layout、空 gather 与空闲均计入 gap；窗口外的活动不计入。
融合计算/IO 段整体视为有效工作，不计入 gap，并完整保留在比例的分母中。
分母只扣除独立 IO 覆盖且没有计算的时间，各方案均报告确定值。
同一张图中四种方法共用时间尺度，kernel 间的实际空隙保持可见。

本次重读原始活动并裁剪窗口，没有重跑 GPU 或修改模型、cache、算子。
原 profile run ID 为 `deepseek_gap_v10_a128_minimal_20261006_01`，
当前绘图 ID 为 `deepseek_gap_v10_idle_echo_timeline_20261006_01`。
每种方法和阶段均只有一次侵入式 NSYS profile，图中时间不等于独立 benchmark 时延；
验证边界沿用 [V10 原报告](../results.md)。

[窗口表](windows.csv)保存纳秒端点和未舍入数值；
[绘图记录](split_receipt.json)绑定输入、源码、空闲及 ECHO 操作区间和图片，
[标注数据](annotations.csv)以 ms 保存八个面板的时长，
[标注核对](annotation_audit.json)保存独立重建的纳秒区间；
[独立核对](window_audit.json)核对八个窗口的区间并集与确定口径的 gap 比例，
[文件清单](publication.json)记录发布文件哈希。

从仓库根目录重画，输出目录须尚不存在：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m experiments.deepseek_v32_mfu.src.compact_timeline \
  --prefill-dir experiments/deepseek_v32_mfu/output/data/deepseek_gap_v10_full_prefill_timeline_20261006_01 \
  --extend-dir /tmp/deepseek_gap_v10_independent_audit_20261006_01/timelines \
  --output-dir experiments/deepseek_v32_mfu/output/data/deepseek_gap_v10_three_layer_window_new \
  --layout separate --window three-layers --io-layout directions --annotations idle-echo
```

当前输出保存在
`experiments/deepseek_v32_mfu/output/data/deepseek_gap_v10_idle_echo_timeline_20261006_01/`，
包含裁剪后的原始活动、窗口表、图片、绘图记录和绘图源码快照。
