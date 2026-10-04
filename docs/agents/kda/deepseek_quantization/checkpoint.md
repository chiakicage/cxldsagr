# DeepSeek 量化：已验收算子检查点

2026-10-03 接受 q2，已完成 indexer 集成及真实 checkpoint 第 0–2 层重新测量。
新的 control01 / candidate02 和三组 NCU 报告已验收发布，旧结果在发布后清理。
本检查点区分量化算子的逐位精确性与整组非矩阵改动后的模型数值边界；前三层
验收不代表完整 61 层或任务质量验证。正式结果见
[实验报告](../../../../experiments/deepseek_v32_echo_prefill/README.md)。

## 实现与精度边界

实现为 `operators/deepseek_v32/indexer/quantization.py`，SHA256：
`083b9671efe7f5dee7e71752dda3255b2d41ee45f906e48876771db5804411ea`。
公开 `quantize_index(x, scale_fmt="ue8m0")` 与旧模型函数签名一致。
contiguous BF16 D128 在 SM90 使用一个 Triton kernel；其他 dtype、布局、维度、
CPU 或非 SM90 使用原始 eager reference。模块导入不加载 Triton/native 扩展。
返回原输入形状的 contiguous FP8 E4M3FN，以及末维为 1 的 contiguous FP32 scale。

保留 FP32 amax、`1e-4` floor 与 PyTorch scalar divide 的舍入。BF16 fast path
的 UE8M0 用指数位取整；在全部 BF16 最大值与 floor 的离散定义域上，逐位等于旧
`exp2(ceil(log2(scale)))`。随后按行计算 power-of-two scale 的 reciprocal 再乘。
`scale_fmt=None` 保留逐元素 `div_rn`。显式传播 NaN，并保留无穷值尺度，避免
Triton 默认 min/max 丢弃 NaN。量化组件本身没有修改 Hadamard、RoPE、selection、
cache 或线性层；整组集成另行移除了 indexer Hadamard，并替换普通非矩阵算子。

30 项 CPU/SM90 检查全部通过，涵盖全部 65,536 个 BF16 编码作为行最大值、
NaN payload、Inf、混合非有限行、正负零、FP8 中点及邻点、随机 Q/K、空行、
FP16/FP32/reference fallback 和非连续布局。FP8 使用原始 uint8 字节比较，
scale 使用原始 int32 位比较。Ruff lint/format 均通过。测试证据位于
`/tmp/deepseek_quantization_q2_20261003/pytest.stdout`。

## 配对计时

Run ID：`deepseek_quantization_q2_20261003`。物理 GPU 2，CUDA 属性为
NVIDIA H200 / SM90 / 132 SM / 150,121,545,728 bytes；PyTorch 2.12.1+cu130，
Triton 3.7.1，CUDA runtime 13.0。seed 72891，20 warmups，单 API 60 次交替采样；
100-call 批次交替测 9 次；graph 每次含 32 次调用，共 12 次。
输出分配、shape 适配和 Python 发射属于 API 时间，编译排除。
graph 单独报告 GPU service time，不代表完整 API 或模型支持 CUDA Graph。

单位均为 microseconds，中位数：

| 输入 / 实现 | 单 API event | 单 API wall | 批 API event/调用 | graph GPU/调用 |
| --- | ---: | ---: | ---: | ---: |
| Q `[1024,64,128]` / 旧 eager | 163.696 | 176.633 | 160.411 | 143.416 |
| Q / compiled official DeepGEMM | 100.144 | 113.776 | 66.232 | 6.232 |
| Q / q2 | 47.712 | 61.047 | 23.904 | 4.635 |
| K `[1024,128]` / 旧 eager | 101.472 | 114.319 | 71.912 | 17.456 |
| K / compiled official DeepGEMM | 99.488 | 112.909 | 65.440 | 1.419 |
| K / q2 | 47.520 | 60.414 | 23.613 | 1.286 |

官方比较调用当前 `linear/fp8.py` 的 fixed-argument compiled upstream helper，
Q 映射为 `[65536,128]`，并恢复原输出形状；不使用历史慢 baseline。
每个计时 shape 先通过 exact 检查。token sweep 为 1/17/128/1024/2048/4096，
每种分别覆盖 K 与 64-head Q。`None` 的 Q1024/K1024 单 API event 为
39.264 / 38.464，graph 为 6.823 / 3.536，也通过原始 eager 的逐位对照。

原始样本、硬件/依赖、源码哈希与可复现脚本位于
`/tmp/deepseek_quantization_q2_20261003/{benchmark.json,compare.py}`。
这些是合成输入的算子诊断，不能替代真实 checkpoint 的完整模型验收。

## NCU 证据与 activation 评估

独立 run `deepseek_quantization_q2_profile_20261003` 保存 full+PM sampling
与 SourceCounters。Q1024 为一个 kernel，128 threads/CTA、4096 CTAs、
34 registers/thread，NCU replay 7.968 microseconds；achieved occupancy 60.15%，
SM throughput 29.67%，DRAM read throughput 44.02%，L2 throughput 66.86%。
NCU 默认 cache replay 与 graph 的热输入不同，7.968 不能替代上表 4.635。
原始 `.ncu-rep`、Python API 提取的全部 metrics 和源码分析在该 `/tmp` 目录。
profile 表明后续上限主要受内存延迟/流量影响，没有矩阵计算或 tensor-core MFU。

activation 对照使用实际 K=1536/7168/16384/18432，另含 MoE K=2048。
官方 helper 已是一个 kernel；K7168 在 graph 下 5.587 microseconds，q2 按
128-group 重新解释的诊断 wrapper 为 4.108，完整单 API 为 79.376 对 50.480。
较大 K=16384/18432 的 GPU 收益只有约 3% / 2%。这些诊断没有改动线性层。
官方 UE8M0 helper 对 Inf/NaN 使用 exponent clamp，与 indexer 原式不同，
因此不能将当前 indexer kernel 无条件当作 activation API 替换；若后续接入，
需单独保存其异常值、padding、dtype 和 layout 契约并重新验证。

## 已完成集成与正式证据

`echo_model.py` 已调用上述公开函数，保持 q_scale/k_scale 的原始下游使用方式。
量化源码维持上述 SHA256；线性 activation 量化仍使用 compiled official
DeepGEMM helper。正式对照在相同当前共享 cache、预算、请求和矩阵后端下完成：

| 用途 | Run ID |
| --- | --- |
| 原非矩阵实现控制组 | `20261003_echo_layers3_nonmatrix_control_01` |
| q2 与 FlashInfer 集成 | `20261003_echo_layers3_nonmatrix_candidate_02` |
| MLA NCU | `20261003_echo_ncu_nonmatrix_mla_02` |
| Resident indexer NCU | `20261003_echo_ncu_nonmatrix_indexer_resident_02` |
| Offload indexer NCU | `20261003_echo_ncu_nonmatrix_indexer_offload_02` |

两实现在同一物理 GPU 5 H200 / SM90 上顺序测量真实 checkpoint 第 0–2 层，
hidden/residual 依次传播，包含 embedding、三个 dense MLP 和末 token 的
final norm / LM head。Prefix=65,536、extend=1,024、chunk=1024、每层 sparse
pool=16,384。每模式预热 1 次，prefix 测 3 次、extend 测 5 次；编译、加载、
prefix residency 恢复与数值诊断不计入 wall time。独立空 cache 构建的
resident/offload 与普通/插桩执行，各 run 的八项检查全部逐位一致；候选三层
MLA 的全部输出亦通过独立 FP32 QK/softmax/PV 检查，沿用原有阈值。

Resident prefix / extend 中位数为 1176.2025 / 24.2997 → 850.6642 / 18.3442 ms；
offload 为 2025.6879 / 51.5112 → 1773.9631 / 36.2451 ms。控制组 offload extend
五个样本为 51.5112、51.7155、52.2868、43.9949、41.4298 ms，存在明显波动；
候选范围为 36.1651–36.3524 ms。中位数比不构成跨负载稳定收益或置信区间。
Offload extend 的 indexer 量化互斥 kernel 成本由 72 个 / 0.486659 ms 降至
6 个 / 0.020320 ms。这是整组改动的真实执行 profile，不是 q2 单独的因果收益。
正式三组 NCU 针对 MLA 与两个 indexer 矩阵 kernel，区别于上文临时 q2 NCU；
其 replay 时间和 tensor pipe 指标不能当作模型 wall time 或量化算子 MFU。

跨版本并不等价：全部 extend hidden 的 relative L2=0.01505245，
50,532 / 7,340,032 个元素超过原 `atol=0.02, rtol=0.01`；logits 的
relative L2=0.00792472，9,129 / 129,280 个元素超差。相同 argmax 不能证明
任务质量。移除 Hadamard 和融合浮点运算使模型输入、传播、选择及 cache 访问
发生变化；相同输入下量化逐位精确不等于整组改动逐位等价，不将漂移全部归于
某一个操作，也不将 offload 的全部 wall 收益归于 q2 或 cache 优化。

Candidate01 的 annotated hidden 重建在 `forward()` 返回后离开了 inference
mode，使 grad-enabled 分支使用 eager norm；结果虽过原容差但未逐位一致。
为 `profile_layers.annotate` 添加 `@torch.inference_mode()` 后以 candidate02
完整重跑，模型/算子源码未改变。重建位于 `cudaProfilerStop()` 之后，不计入
profile，验收覆盖全部 `[1024,7168]` hidden。Candidate01 未作为有效实验发布。

两个 run 各自 1,218 项源码记录均已核验；candidate02 的实际 FlashInfer/CuTe
runtime identity 亦已核验。各 run 四 capture 的原始 SQLite、kernel inventory
与互斥账本守恒，零遗漏和调用数不匹配。发布时当前 checkout 的 32 个生产源码中
有 31 个与候选快照相同；
唯一后续差异是 `echo_infer.py` 构造阶段增加
`max(query reservation, chunk, extend_chunk) <= slots` 检查，不在计时区间，
且本轮 1024 < 16,384 不触发。报告的源码身份仍以保存的快照为准。

集成验证另有 40 项 model/quantization 检查通过；最终比较、backend provenance
与 profile_layers 检查 16 passed，comparison audit 24 passed。完整验证矩阵、
互斥归因边界及来源索引见
[系统检查点](../../system/deepseek_nonmatrix_optimization.md)和实验报告。
