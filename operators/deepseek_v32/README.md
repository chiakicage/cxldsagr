# DeepSeek V3.2 kernels


独立 Hopper / SM90 实现，为 checkpoint ECHO prefill/extend 提供算子。模型投影语义、
精确选块、缓存调度和事务位于 [模型目录](../../models/deepseek_v32/README.md)。

H64K+A1 的真实前三层四方案已在 `deepseek_h64k_a1_fused_prepare_20261009_01` 完成
独立正确性验收、正式计时及 graph node profile。当前版本包含官方 ECHO paged Q1
decode、合并 KV 打包、Q1 scale 布局、多 CTA 暂存发布、精确 top-k 的对齐 score
视图与 CUB 后处理、官方 Q1 专用的 64 空槽准备、精确 hint 均值融合及 page64／页表／暂存区准备融合。
check/profile 保存的 44 个输出 tensor 的独立复读比较全部逐位一致，六次 cold
ECHO 状态证明及每个正式样本的流量均通过核验。每方法的 check、bench、阶段
profile 和逐算子 profile 均使用独立进程并只预热本方法；逐算子 profile 的 8 个
保存输出与相应验收逐位一致。此项测量流程修正没有改变算子计算。结果见
[四方案 MFU](../../experiments/deepseek_v32_mfu/README.md)和
[官方对照与 Q1 算子实验](../../experiments/deepseek_v32_echo_official/README.md)。
准备融合每次重建当前 K/scales、页表和暂存状态，不新增持久 storage；
三层准备从 9 个节点减到 3 个，完整 ECHO graph 从 263 个节点减到 257 个。
151 项 GPU 检查和四方法独立进程补测通过。私有 500 对计时的中位数下降约
1.32%，部分均值与 p99 变差，不外推尾延迟或完整 serving 收益。
已有报告仍按各自 run ID、源码和测量边界解读。

Q1 hint 均值融合的 70 项算子检查、生产入口验收及正式三层补测均已完成。
在 SM90 推理、FP32 `[1,65537]`、内层 stride 为 1 且输入按 16 B 对齐时，
[均值入口](indexer/q1_hint_exact.py)用一个 kernel 保留已验收的 Torch FP32 归约树，
合并有限值掩码、计数与发布；只更新 `offset[0]`，decode EMA 仍单独更新。
其他输入保留原分派。生产验收覆盖全部 offset 位、变化输入的 graph replay、非默认
stream 和 Q1 后直接执行 A2 的实际消费。通用执行预留不变；本轮四方法的完整图
private reserved 仍均为 62,914,560 B，临时张量减少未带来该观测值下降。

| 功能 | 入口 | 当前实现 |
| --- | --- | --- |
| Indexer | [indexer/echo.py](indexer/echo.py) | FP8 causal logits；可在同一 kernel 内预取主 MLA records，仍需精确 top-k 和剩余 miss recall |
| Indexer selection | [indexer/selection.py](indexer/selection.py) | FlashInfer SMALL 精确选择；Q1/k2048/至少 32K 列使用官方 CUB 合并排序与 mask，其他形状保留原 sorted 路径 |
| Indexer quantization | [indexer/quantization.py](indexer/quantization.py) | KDA 验证的 SM90 BF16/D128 单 kernel，FP8 bytes 与 FP32 scales 对齐独立 reference |
| Attention reference | [attention/reference/torch.py](attention/reference/torch.py) | 独立 FP32 PyTorch oracle，可单独在 CPU 导入运行，不加载 Triton 或 native 扩展 |
| Attention device_only | [attention/device_only/mla.py](attention/device_only/mla.py) | 官方 FlashMLA sparse prefill；显式 Q1 decode 使用 16 分片与 LSE 合并，均为 SM90 / BF16 / H64 或 H128 / D576 / V512 |
| Attention offload | [attention/offload/mla.py](attention/offload/mla.py) | 消费已精确 recall 的 HBM pool 和物理 ID，复用同一个 device-only MLA kernel |
| Linear / MoE | [linear/fp8.py](linear/fp8.py) | 官方 DeepGEMM block FP8 dense / grouped GEMM，激活量化使用手写 Triton |

ECHO 的融合发生在 indexer 与 prefetch 之间；当前没有将 host fetch 融入 attention
kernel。每个 offload query batch 必须先补齐其精确选择，模型在选中并集超过 pool
容量时拆分 query 消费，保持每个 query 的选择不变。通用 pinned-host record gather
位于 [operators/common/kv_transfer.py](../common/kv_transfer.py)，有限 slots、映射和
事务位于 [cache/sparse_token_cache.py](../../cache/sparse_token_cache.py)。

融合 indexer 在依赖 scale 的 logits 写出后归还共享 KV stage，防止 TMA 提前覆盖
仍在读取的 scale。原 Q1024/N66560 的间歇分数差异已通过该释放顺序修正；固定输入
连续 64 轮完整分数逐位验收通过，生产回归也覆盖了同一复用过程。修正后的 C10
模型级正式计时见下文；这些数值检查不替代 kernel 的独立性能测量。

主 KV record 是 512 个 BF16 latent 加 64 个 BF16 RoPE key，1152 B/token；indexer
FP8 K 与 FP32 scale 常驻 GPU。MLA 的 selection 在所有 attention heads 间共享；
重复 ID 逐 slot 参与计算，负值和越界 ID 是 padding，全 padding 行输出零。
FlashMLA 通过顶层 [子模块](../../3rdparty/FlashMLA) 接入；默认 `sparse_mla` 入口处理
KV head 维度、连续布局、int64 ID 安全转换及 selection 补齐到 128 的倍数。必要的复制与
padding 都属于算子时间。其他 dtype 或布局明确失败，不使用 Triton attention fallback。
CPU reference 仍可独立导入，不加载 FlashMLA。

Resident indexer 在 Q1 且有效 causal context `query_start + 1 >= 32768` 时调用
官方 paged MQA。每次调用用一个 kernel 按 64-token 页打包 FP8 K 和
FP32 scales，N=65,537 时 packed storage 为 8,659,200 B（约 8.66 MB），另有页表和
调度 metadata。这些临时空间已纳入执行预算，最大 query 配置也覆盖可达的 Q1 尾块。

有效 causal context 至少为 32,768，且满足完整 history 和 64-slot 预取空间条件的 Q1 offload 优先调用
[官方源码桥接](indexer/official_decode.py)，再由[pool 适配](indexer/official_prefetch.py)
发布暂存记录。官方最多预取 64 条阈值预测记录，余下精确选择继续召回；Q>1 保留
原 prefill 路径。官方 score storage 的物理行宽按 256 个 FP32 元素对齐，默认返回
逻辑长度 N；模型通过私有选项将已有的 `-inf` 尾部一并暴露给精确 top-k，避免复制
分数或增加 kernel。选择数量和 hint 归约仍按逻辑 N 处理，保留原始分数、预测阈值和精确选择。

暂存发布先用一个 256-thread CTA 并行核验 ID、slot、旧 owner、journal 和重复记录，再由最多 64 个 CTA
逐 record 搬运并发布映射。两阶段均在原 stream 中执行，保留任意连续 BF16 对齐，
无需新增 scratch；清理及原有异常传播不变。所有实际预测预取都计入 H2D，
包括未进入精确 top-k 的记录。暂存后的 device-to-device 搬运单列为 GPU control，
不重复计为 host IO。独立正确性、完整两阶段计时及 NCU 依据见上述 Q1 算子实验；
该版本也已通过上述真实前三层的完整路径验收。

官方 Q1 的准备在单 session、独占 pool 操作、普通持久 append、已初始化
`H=query_start` 且 `P-H>=64` 时，按槽位升序选出前 64 个空槽。三个 kernel
完成全池 priority 核验、空槽选择及完整 journal、计数和统计重置；选中槽仍须通过
free bitmap 与反向映射检查。`L<=H` 只证明空槽足够，不证明 history 驻留。
独立 bounded token 绑定 storage、stream 和 64 槽上限，通用 prefill 拒绝消费；
其他形状与容量条件保留原完整准备。三层完整调用的 100 组成对计时见
[模型说明](../../models/deepseek_v32/README.md)，逐次状态证明见上述 MFU 实验。

显式 `sparse_mla_decode` 将 selection 分成 16 份，重复 Q 后调用一次官方 sparse
prefill，再由 [decode.py](attention/device_only/decode.py) 用自然对数 LSE 以 FP32 合并 BF16
partial 输出。重复 ID、无效 ID 和空分片语义保持不变；新增舍入按原数值容差验收。
模型仅对原始 Q1 且 selection 宽度为 2,048 的 batch 使用该入口；prefill 因容量拆分
产生的 Q1 仍调用 `sparse_mla`，保持切片逐位契约。Query 复制和合并空间计入预算，
完整图保留的 storage 另由 graph private pool 核验。
独立完整 Graph API 计时为 54.92–54.99 → 15.37–15.55 µs，包含 Q repeat、
官方主 kernel 与合并；eager 约 55 µs，基本持平。该组件结果与完整模型补测分别报告，
原始数据和数值容差见[attention 报告](../../experiments/deepseek_v32_echo_official/report/q1_optimization/report.md)。

FP8 投影使用顶层 [DeepGEMM 子模块](../../3rdparty/DeepGEMM) 的 `main` 分支固定版本。
激活量化由 [本地接口](linear/quantization.py) 直接启动 Triton kernel，不调用
`torch.compile`。它保留 128-channel / UE8M0 格式，返回独立、连续的 FP8 数据和
FP32 scales；编译后的上游 `per_token_cast_to_fp8` 仅用于验证和性能对照。
dense / grouped 路径调用官方公共 API。适配层的 padding、
route packing、scale layout 转换和 inverse mapping 均计入算子时间。
CPU FP32 oracle 独立实现，不依赖 DeepGEMM。
量化路径已通过生产接口、实际 checkpoint MLP 和四方案 H64K 数值验收。
当前 C10 批次 `deepseek_motivation_matrix_20261007_01` 已完成三档 H、四档 A 的
12 点独立验收，1,152 组 offload/HBM 输出逐字节一致。修正后的 ECHO、dense DMA
与 v4 局部计算图已完成每点四方案各 32 请求的正式计时；结果及测量边界见
[motivation](../../experiments/deepseek_v32_motivation/README.md)。本轮没有新采集
profile，保留的 profile 只匹配旧 H65536/A128 单点。这些是模型级路径证据，
不能替代某个 kernel 的独立性能测量或证明完整 61 层性能。

单元测试随 indexer、linear 和三种 attention 实现存放；通用搬运测试在
`operators/common/tests/`。从仓库根目录运行全局 CPU 或 Hopper GPU 回归：

```bash
bash scripts/run_tests.sh cpu
bash scripts/run_tests.sh gpu
```

真实 checkpoint 第 0–2 层顺序传播的验收为
`refactor_three_layers_check_20261005_02`，与 C10 输入重放是不同工作负载。
三层 benchmark/profile 的已发布结果见
[ECHO 实验](../../experiments/deepseek_v32_mfu/README.md)。原 run ID、
源码与测量边界继续标识原结果；切换后端或通过回归测试不构成新的性能结果。
