# 已有实现结果对照

本表只整理已发布且通过验收的结果，没有新增 GPU 运行或计时。本地列对应 `motivation_c10_20261004_u16_r2_01`，官方列对应 `echo_official_c10_20261004_u16_r2_01`。结果仅代表这两次运行记录的实现。

两组均使用 10 个独立 checkpoint dense block 副本（7,827,793,408 参数），采用 FP8 linear；各副本复用对应源层的 hidden/residual 输入。该负载是 checkpoint 工作负载替身。P=65,536，NH=16,777,216，H=65,536，A=128，history chunk=1,024；16 用户按顺序访问 2 轮，seed=42。两组均启用公共计算 CUDA Graph。工作负载 SHA256 相同，P/NH 不扣除字节子预算或经验预留。

| 实现 | 首访均值 ms | 首访 p95 ms | 复访均值 ms | 复访 p95 ms | 总耗时 s | allocated 峰值 GiB | reserved 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| HBM-only（本地对照） | 2176.886 | 2187.259 | 2170.244 | 2178.394 | 69.554 | 14.517 | 23.635 |
| 我们的 ECHO（已有验收版本） | 2283.433 | 2293.095 | 24.947 | 25.509 | 36.934 | 16.357 | 24.297 |
| HBM-only（官方对照） | 2114.286 | 2121.285 | 2106.515 | 2117.721 | 67.533 | 14.549 | 23.875 |
| 官方 ECHO 适配路径 | 4637.566 | 4651.222 | 45.561 | 46.138 | 74.930 | 16.410 | 24.521 |

首访、复访分别含 16、16 个请求；p95 直接引用各组已发布的线性插值分位数。总耗时为全部请求延迟之和，包含 miss 时的 history 重建和全部 candidate hidden、末 token LM head。allocated/reserved 是各方案完整正式轨迹的 PyTorch 峰值；两者分别报告，不作为设备已用 HBM 的峰值。

以官方 ECHO 为分母，本地 ECHO 的首访均值差为 -50.762%，复访均值差为 -45.244%，总耗时差为 -50.709%。正值表示本地耗时较高，负值表示本地耗时较低。这些是两次独立轨迹的观测差异，不足以判断优势是否稳定。

两次运行使用同一物理 GPU，PyTorch 记录的设备名称均为 NVIDIA H200（132 SM、139.812 GiB HBM）。本地 HBM 使用 mainline DeepGEMM logits，本地 ECHO 使用项目融合 prefetch 路径，两者使用 FlashInfer top-k；官方组使用 ECHO 原始 resident/fused logits 和原始 top-k。两组选择路径和 HBM 基线有差异，不能把差额全部归因于缓存或搬运实现。

两组记录的 FP8 linear 设置一致，checkpoint 身份核对依赖路径和 shard stat 清单，没有权重内容哈希。两组共同记录的有效精度设置一致；仅一组记录的字段及各实验的 policy ID 不能用于确认完整精度策略相同。本地输出在来源运行内与 HBM 对照逐位一致；官方输出及独立 resident 重跑均通过该实验预先固定的数值门槛，官方 top-k 顺序及并列值可能变化。本次整理未新增跨运行数值比较。

## 来源

- motivation：`motivation_c10_20261004_u16_r2_01`；源码 SHA256 `11fc11b18b2e70baf450a82ab4fad66f2f8d4e22cda2e9f36bf74453a400df8f`。
  输入 `experiments/deepseek_v32_motivation/output/data/motivation_c10_20261004_u16_r2_01/report/summary.json`，SHA256 `a00fb1dba99cb33ca0bed7ff34e192206829689974224e65ebc6892c6899a4db`。
  GPU UUID `1bdee8b4-22ac-536c-208b-bfb4ed38b878`。
- official：`echo_official_c10_20261004_u16_r2_01`；源码 SHA256 `fef9fef45d923343967198552709f7a7549da9145ce0927b535539d86f8cc60d`。
  输入 `experiments/deepseek_v32_echo_official/output/data/echo_official_c10_20261004_u16_r2_01/report/summary.json`，SHA256 `a7dd7655b97c6bc8ed3fd579f8ff64fdfe75f1eae494a63461b519872c1c7b16`。
  GPU UUID `1bdee8b4-22ac-536c-208b-bfb4ed38b878`。

共同 workload SHA256：`7e4c737a86464c12238191933e707344426231bf671a29658837e676ff5284ae`。

CSV 保存未四舍五入的数据；JSON 保存输入及生成器哈希、配置核对、原验收状态、完整 run ID 和测量边界。生成入口：

```bash
python -m experiments.deepseek_v32_echo_official.src.compare_existing \
  --motivation-report experiments/deepseek_v32_motivation/output/data/motivation_c10_20261004_u16_r2_01/report/summary.json \
  --official-report experiments/deepseek_v32_echo_official/output/data/echo_official_c10_20261004_u16_r2_01/report/summary.json \
  --output-dir /tmp/deepseek_official_c10_publication_20261004/comparison_final
```
