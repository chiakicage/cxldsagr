# 已有实现结果对照

本表只整理已发布且通过验收的结果，没有新增 GPU 运行或计时。本地列对应 `refactor_final_deepseek_bench_20261005_01`，官方列对应 `refactor_final_official_bench_20261005_01`。结果仅代表这两次运行记录的实现。

两组均使用 10 个独立 checkpoint dense block 副本（7,827,793,408 参数），采用 FP8 linear；各副本复用对应源层的 hidden/residual 输入。该负载是 checkpoint 工作负载替身。P=65,536，NH=16,777,216，H=65,536，A=128，history chunk=1,024；16 用户按顺序访问 2 轮，seed=42。两组均启用公共计算 CUDA Graph。工作负载 SHA256 相同，P/NH 不扣除字节子预算或经验预留。

| 实现 | 首访均值 ms | 首访 p95 ms | 复访均值 ms | 复访 p95 ms | 总耗时 s | allocated 峰值 GiB | reserved 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| HBM-only（本地对照） | 2192.406 | 2199.826 | 2187.398 | 2191.141 | 70.077 | 14.517 | 23.635 |
| 我们的 ECHO（已有验收版本） | 2294.994 | 2299.413 | 24.530 | 25.607 | 37.112 | 16.357 | 24.297 |
| HBM-only（官方对照） | 2126.499 | 2132.862 | 2123.192 | 2126.623 | 67.995 | 14.549 | 23.875 |
| 官方 ECHO 适配路径 | 4596.177 | 4609.691 | 44.862 | 46.391 | 74.257 | 16.410 | 24.521 |

首访、复访分别含 16、16 个请求；p95 直接引用各组已发布的线性插值分位数。总耗时为全部请求延迟之和，包含 miss 时的 history 重建和全部 candidate hidden、末 token LM head。allocated/reserved 是各方案完整正式轨迹的 PyTorch 峰值；两者分别报告，不作为设备已用 HBM 的峰值。

以官方 ECHO 为分母，本地 ECHO 的首访均值差为 -50.067%，复访均值差为 -45.322%，总耗时差为 -50.021%。正值表示本地耗时较高，负值表示本地耗时较低。这些是两次独立轨迹的观测差异，不足以判断优势是否稳定。

两次运行使用同一物理 GPU，PyTorch 记录的设备名称均为 NVIDIA H200（132 SM、139.812 GiB HBM）。本地 HBM 使用 mainline DeepGEMM logits，本地 ECHO 使用项目融合 prefetch 路径，两者使用 FlashInfer top-k；官方组使用 ECHO 原始 resident/fused logits 和原始 top-k。两组选择路径和 HBM 基线有差异，不能把差额全部归因于缓存或搬运实现。

两组记录的 FP8 linear 设置一致，checkpoint 身份核对依赖路径和 shard stat 清单，没有权重内容哈希。两组共同记录的有效精度设置一致；仅一组记录的字段及各实验的 policy ID 不能用于确认完整精度策略相同。两组数值验收均来自独立 check，正式计时不保存或比较完整输出。本地输出与其 HBM 对照逐位一致；官方输出及独立 resident 重跑均通过该实验预先固定的数值门槛，官方 top-k 顺序及并列值可能变化。本次整理未新增跨运行数值比较。

## 来源

- motivation：`refactor_final_deepseek_bench_20261005_01`；源码 SHA256 `1319a540b7f78da9c363dd09c0f1568b450df3df5e4d3e8b60155a58c348b3dc`。
  输入 `experiments/deepseek_v32_motivation/output/data/refactor_final_deepseek_bench_20261005_01/report/summary.json`，SHA256 `b14dc62ae72ba1bbb97915f68c5153d3d99fb4204ef92fdbde1ff44c8093d143`。
  GPU UUID `80ff95c3-176e-fd8a-728f-9c5577c4a779`。
- official：`refactor_final_official_bench_20261005_01`；源码 SHA256 `a9d45df3f2ccca5b3ed2a5e36bbb6c0d54a768c87fe7d1863555398ff76c47a1`。
  输入 `experiments/deepseek_v32_echo_official/output/data/refactor_final_official_bench_20261005_01/report/summary.json`，SHA256 `c4ca38532520f8fb29982d8015d7b2205deb28869d77db7d68b57882b8c0d3d8`。
  GPU UUID `80ff95c3-176e-fd8a-728f-9c5577c4a779`。

共同 workload SHA256：`7e4c737a86464c12238191933e707344426231bf671a29658837e676ff5284ae`。

CSV 保存未四舍五入的数据；JSON 保存输入及生成器哈希、配置核对、原验收状态、完整 run ID 和测量边界。生成入口：

```bash
python -m experiments.deepseek_v32_echo_official.src.compare_existing \
  --motivation-report experiments/deepseek_v32_motivation/output/data/refactor_final_deepseek_bench_20261005_01/report/summary.json \
  --official-report experiments/deepseek_v32_echo_official/output/data/refactor_final_official_bench_20261005_01/report/summary.json \
  --motivation-receipt docs/agents/acceptance/unified_runtime_20261005/deepseek_final/receipt.json \
  --official-receipt docs/agents/acceptance/unified_runtime_20261005/official_final/receipt.json \
  --output-dir experiments/deepseek_v32_echo_official/output/data/refactor_final_official_comparison_20261005_01
```
