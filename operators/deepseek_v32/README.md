# DeepSeek V3.2 kernels


独立 Hopper / SM90 实现，为 checkpoint ECHO prefill/extend 提供算子。模型投影语义、
精确选块、缓存调度和事务位于 [模型目录](../../models/deepseek_v32/README.md)。

Q1 offload 当前使用 `official-q1-exact-selected-staging-v2`：官方融合内核最多
暂存 64 条预测，适配器在精确 top-k 后只驻留选中的历史记录，清除其余暂存 tag。
所有实际暂存 H2D 都计费，未选中的预测也包含在内；剩余精确选择继续完整 recall。
当前 promotion 仍在 append 前完成，与官方 recall 内发布的时序有差异。

H64K+A1 的真实前三层四方案已完成当前策略的独立验收、正式计时及两类 profile，
cold run ID 为 `deepseek_h64k_a1_exact_selected_cold_20261011_02`。相同官方 history
和 decode token 57841 的独立 warm 对照保留 prefill 驻留，正式计时与 profile 中
三层 prefetch、residual recall、H2D 均为 0，append 的 D2H 单独记录；实际仍执行
融合 Q1 内核。具体结果与 source/native/receipt 边界见
[四方案 MFU](../../experiments/deepseek_v32_mfu/README.md)。这些条件不证明完整官方
cache 调度或跨实现数值等价。

合并 KV 打包、scale 布局、精确 top-k 的对齐 score 视图与 CUB 后处理、64 空槽
准备、hint 均值融合及 page64／页表／暂存区准备融合继续保留。准备每次重建当前
K/scales、页表和暂存状态，不新增持久 storage。独立算子结果见
[官方对照与 Q1 算子实验](../../experiments/deepseek_v32_echo_official/README.md)，
不沿用旧 promotion 策略的全模型配对收益。

Q1 hint 均值融合的 70 项算子检查、生产入口验收及正式三层补测均已完成。
在 SM90 推理、FP32 `[1,65537]`、内层 stride 为 1 且输入按 16 B 对齐时，
[均值入口](indexer/q1_hint_exact.py)用一个 kernel 保留已验收的 Torch FP32 归约树，
合并有限值掩码、计数与发布；只更新 `offset[0]`，decode EMA 仍单独更新。
其他输入保留原分派。生产验收覆盖全部 offset 位、变化输入的 graph replay、非默认
stream 和 Q1 后直接执行 A2 的实际消费。通用执行预留与实际 graph private reserved
分别核验，临时张量减少不直接证明 reserved 下降。

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

无 offload 算子观测使用独立序列适配器：`indexer/resident_batch.py` 为每个序列保留
独立的 K/scales，Q1 显式调用官方 paged API，B1/Q>1 调用官方 nonpaged API。
`attention/device_only/batch.py` 将各序列的 BF16 主 KV 展平为互不相交的范围，并
转换本地 selection ID 后调用 sparse prefill；B1/Q1/k2048 继续使用 split16 decode。
B1024/Q1/H65536 的 page64 准备通过有界分批调用原 kernel，避免 int32 地址计算溢出。
全部 52 个 B/Q/H 配置已完成独立算子验收、静态 Graph replay 检查和正式计时，结果见
[驻留算子观测](../../experiments/deepseek_v32_observation/report/results.md)。该范围不执行
Cache Manager 事务或 offload，不证明完整模型容量或 serving 收益。

有效 causal context 至少为 32,768，且满足完整 history 和 64-slot 预取空间条件的 Q1 offload 优先调用
[官方源码桥接](indexer/official_decode.py)，再由[pool 适配](indexer/official_prefetch.py)
发布精确选中的暂存记录。官方最多预取 64 条阈值预测记录；适配器在精确 top-k
之后只驻留其中选中的历史记录，清除其余暂存 tag，再完整召回剩余选择。策略为
`official-q1-exact-selected-staging-v2`；Q>1 保留
原 prefill 路径。官方 score storage 的物理行宽按 256 个 FP32 元素对齐，默认返回
逻辑长度 N；模型通过私有选项将已有的 `-inf` 尾部一并暴露给精确 top-k，避免复制
分数或增加 kernel。选择数量和 hint 归约仍按逻辑 N 处理，保留原始分数、预测阈值和精确选择。

暂存发布先用一个 256-thread CTA 核验精确选择与预测的交集，保护已驻留的精确选择，
检查 ID、slot、旧 owner、journal 和重复记录，再由最多 64 个 CTA 搬运选中的
record 并发布映射。两阶段均在原 stream 中执行，保留任意连续 BF16 对齐，
无需新增 scratch；清理及原有异常传播不变。所有实际预测预取都计入 H2D，
包括未进入精确 top-k 的记录。暂存后的 device-to-device 搬运单列为 GPU control，
不重复计为 host IO。Promotion 在 append 前执行，尚未完整复现官方 recall 内
发布的时序。新策略的完整路径结果见上述 MFU 实验；原打包、精确 top-k
与 attention 的独立算子结果继续保留各自的测量边界。

官方 Q1 的准备在单 session、独占 pool 操作、普通持久 append、已初始化
`H=query_start` 且 `P-H>=64` 时，按槽位升序选出前 64 个空槽。三个 kernel
完成全池 priority 核验、空槽选择及完整 journal、计数和统计重置；选中槽仍须通过
free bitmap 与反向映射检查。`L<=H` 只证明空槽足够，不证明 history 驻留。
独立 bounded token 绑定 storage、stream 和 64 槽上限，通用 prefill 拒绝消费；
其他形状与容量条件保留原完整准备。当前策略的逐次状态证明与完整路径计时见
上述 MFU 实验。

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
