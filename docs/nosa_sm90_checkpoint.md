# NOSA SM90 实现检查点（2026-09-29）

本次发布当前已验收的 resident indexer 与 block sparse attention 实现，继续保留
完整 NOSA 33/64、CIS bias、因果遮罩和稳定并列排序。40% useful MFU 目标尚未全部达到；
当前实现不包含 DRAM/CXL offloading。

## Indexer

模型负责压缩和选择语义，`NosaKVCache` / `IndexerCache` 保存请求级派生记录与共享
scratch。32-token、stride-16 压缩只追加新窗口，稳定 CIS pool 随 KV 一起提交、回滚
和截短。checked native 入口合并有限值校验、增量准备与选块的主机提交，先检查可写
缓冲别名；非法 Q、新 K 或新 CIS 不修改派生记录。同步 checked 入口拒绝 CUDA Graph
capture，异步准备接口保持可捕获。

SM90 大形状路径采用 CUDA/CuTe WGMMA。第一遍 QK 计算逐 Q head softmax normalizer，
第二遍融合归一化、GQA、BF16 舍入、五窗口 pooling 和完整选择，减少中间张量与 launch。
ranked preparation 同时准备 CIS 排序，最终选择复用精确 rank-33 cutoff，并只排序保留的
64 个逻辑 ID。Q16/N128 第一遍以四个 consumer warpgroup 和 producer 交错执行。

对 `queries=1024, KV heads=2, compressed count=4159, blocks=1040,
query_start=65536`，额外使用向上舍入的 FP16 上界进行精确剪枝：只有上界严格低于当前
精确 cutoff 才跳过第二遍计算，不确定情况仍计算。Q8 独立尾段列表避免另一组 query
不需要的 tile；描述符生存期、固定几何特化及 BF16 成对转换降低额外开销。
其他形状和资源条件不满足时保留已有路径；公开 score-only 调用不使用该剪枝。

## Block sparse attention

满足 native D128、GQA16 和地址/stride 对齐条件，且 BF16 的 K/V stride 匹配时，
使用固定版本 FlashInfer 0.6.18 的 FA3 Hopper 模板，
八个相邻 query 组成逻辑块并集，KV128、双 stage，通过 TMA 直接读取原始 stride 的 Q，
epilogue 直接写最终输出。每个 query 仍使用自己的 membership、CIS 和因果遮罩。
当工作项数为 256 时，按实际 paired-page tile 数降序调度，并以逻辑 ID 升序处理并列。

目标形状的完整调用包含四个 kernel：并集准备、工作排序、FA3、native per-query repair。
其他正长度 FA3 形状保留三个 kernel，空 query 不 launch；FP16 或 K/V stride 不同保留
native 路径。异常值检测和 repair 的耗时始终计入完整 attention。

## 数值与测量

QK 缩放和 CIS 加法保持独立 FP32 RN 舍入，先在自然单位中减去最大值，再转换为 exp2
输入，修复大公共偏移及 QK/CIS 抵消导致的精度问题。Native BF16 PV 按实际 token 数
缩放 V，归一化后恢复尺度，并保护最终 dtype 转换，修复极大有限 V 导致的中间溢出。
既有误差容限未放宽。

以下来自原测量 `kda_main_bf16_pair_v3_development`：H20Z/SM90、132 SM，64K prefix
+ 1K queries，BF16、32 Q heads / 2 KV heads / D128，峰值分母 989 TFLOPS。
表中时间为完整模块的 CUDA kernel 时间总和，包含校验、准备、选择和辅助 kernel；
CUDA-event API 与墙钟延迟另行报告。

| Layer | Indexer µs | Indexer MFU | Attention µs | Attention MFU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 141.471 | 24.741% | 175.391 | 39.312% |
| 15 | 137.854 | 25.390% | 173.247 | 39.798% |
| 31 | 137.504 | 25.455% | 170.879 | 40.350% |

只有 L31 attention 达到 40%。Indexer 还需从约 138–141 µs 降到 87.503 µs；attention
需三层均低于 172.374 µs。完整数据、原始源码指纹与计时边界见
[模块检查点报告](../experiments/nosa_kernel_mfu/README.md#完整模块检查点)。
发布时仅对 attention Python adapter 做等价格式化，原测量 metadata 保留原哈希。
metadata 的 `git_commit` 指测量时未提交工作树的基线；实际实现身份由源码哈希和快照确定。

原实现验收包含七个 native 组件重建、361 项 GPU 回归、18 组完整 kernel 归因检查，
以及独立压缩和选块一致性检查；三个 captured layer 的选择 ID 均未改变。测试用于代码
正确性验收，不作为模型质量或论文实验结论。

本次提交前重新运行 GPU 全局入口，361 项通过；测量与归因工具的 CPU 测试 560 项
通过，Ruff 检查和格式检查通过。全局 CPU 入口两次运行均在既有 pattern 分析测试的
NumPy 数组排序阶段长时间未完成，限制线程数后仍未消除，已停止；不将其计为通过。
该未完成检查仍需后续排查，当前结果不能表述为本次全局 CPU/GPU 回归全部通过。

完整模型 native/Triton 对照已于 2026-09-29 按本检查点 `94bf521` 补测，run ID 为
`sparse_native_h200_gpu1_20260929_01` 与 `sparse_triton_h200_gpu1_20260929_01`。
两组同源和输出验收通过，已发布[端到端与模块报告](../experiments/indexer_block_sparse_profile/README.md)
并替换该实验的旧报告及运行产物；这不表示两个完整模块的 40% MFU 目标已达成。
Synthetic operator 对照及受影响的 full-NOSA pattern 仍待补测，对应 README 继续保留
原 run ID、原报告和适用边界。未通过性能验收的私有候选不属于本次实现。

入口：[SM90 算子](../operators/sm90/README.md)、[NOSA 模型](../models/nosa/README.md)、
[后续优化计划](plan.md)。
