# DeepSeek V3.2


独立 SM90 checkpoint 推理实现，支持 ECHO prefill/extend。完整结构包含 61 层、
embedding、3 个 dense MLP、58 个 MoE、final norm 和 LM head。硬件算子按功能位于
[operators/deepseek_v32](../../operators/deepseek_v32/README.md)。
当前非 GR benchmark 仅执行真实 checkpoint 第 0–2 层，包含 embedding、final norm
和末 token LM head；GR 对照另用十个独立 dense block 的输入重放工作负载。
完整层数入口仍保留，本轮验收不等于完整 61 层验证。两条工作负载分别报告，
真实三层的数值、性能和 profile 边界见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。

| 模块 | 用途 |
| --- | --- |
| [model.py](model.py)、[infer.py](infer.py) | 完整模型与 CLI：跨 GPU 放置权重、按 token chunk 执行全部层、统一缓存事务及 LM head |
| [layers.py](layers.py) | 双路 hidden/residual、RMSNorm、dense / grouped MoE 与 checkpoint 权重加载 |
| [attention.py](attention.py) | 执行外层 query batch 的投影、融合 prefetch、精确 top-k / recall、物理 ID remap 与 sparse MLA |
| [config.py](config.py)、[checkpoint.py](checkpoint.py) | checkpoint 配置、tensor 读取与 BF16/FP8 线性权重 |
| [projections.py](projections.py)、[rotary.py](rotary.py) | MLA/indexer 投影、YaRN 与 RoPE 配对 |
| [nonmatrix.py](nonmatrix.py) | FlashInfer RMSNorm、残差归一化与 SiLU 适配，保留 FP32 checkpoint norm 权重 |
| [adapter.py](execution/adapter.py) | 单卡 GR serving 的 10 个 dense block / 输入复制工作负载与可复用用户 cache session |
| [replay.py](replay.py) | checkpoint 副本的独立加载、source 层映射及 hidden/residual 输入克隆；物理层数由入口显式指定 |
| [session.py](cache/session.py) | 用户 session 状态与借用双缓冲的 dense cache view |
| [planning.py](execution/planning.py) | 根据模型维度和执行上限计算固定 P/NH 或字节预算的资源计划 |
| [pipeline.py](execution/pipeline.py)、[official.py](execution/official.py) | 本地与官方 attention/cache/resource factory；官方入口为 `build_official_backend` |
| [tests/](tests) | ECHO checkpoint / block / 全模型调度事务参考测试 |

## 运行

从仓库根目录执行；第三方源码准备见 [依赖说明](../../3rdparty/README.md)。SM90 使用基础环境：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync
source .venv/bin/activate
python -m models.deepseek_v32.infer --help
python -m models.deepseek_v32.infer \
  --model /preset-models --num-layers 3 --devices 0 \
  --input-ids /path/to/input_ids.json --history 65536 --chunk-size 1024 --offload --slots 16384
```

`input_ids.json` 是单个请求的 token ID 列表，64K + 1K 场景共 66,560 项；省略
`--offload` 使用 resident 主 KV。权重常驻各 GPU，hidden/residual 在层放置边界传输。
每个 token chunk 顺序经过全部层；所有 chunk、输出与 GPU 同步成功后才提交请求长度。
默认只计算最后 token 的 LM head。前三层 benchmark 的输入生成与可复现命令见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。

显式 `--num-layers 3 --devices 0` 只加载并顺序执行 checkpoint 第 0–2 层；实验入口
固定使用这一范围。模型 CLI 省略 `--num-layers` 仍执行完整模型。Python 接口为
`DeepSeekEchoModel(..., num_layers=3)`；`forward(..., return_hidden=True)` 另返回
全部输入 token 经 final norm 后的 hidden，供数值比较。前三层顺序传播是 checkpoint
工作负载，不能称为独立训练的三层模型或完整 61 层输出。

模型 CPU 回归：

```bash
.venv/bin/python -m pytest \
  models/deepseek_v32/tests/test_echo_model.py \
  models/deepseek_v32/tests/test_echo_block.py \
  models/deepseek_v32/tests/test_echo_infer.py -q
```

## KV 与 RoPE 语义

- SM90 ECHO 主 KV 每 token 1152 B：512 个 BF16 latent 加 64 个 BF16 RoPE key。
  Indexer FP8 K 与 FP32 scale 常驻 GPU；主 KV 使用 resident 存储，或
  [共享 token pool](../../cache/sparse_token_pool.py) 的 pinned DRAM backing 与逐层有限 HBM slots。
  模型/backend 持有 pool，session 持有独立 host page table 与 indexer 状态。
  持久 append 在 indexer 历史预取后执行精确 top-k，再直接写入当前主 KV 并异步写回
  host，补齐剩余 miss；GR 临时候选另按下节使用共享 GPU 尾部，不写回 host。
  过大的 query 选中并集拆分消费，
  不截短 query 的精确选择。此路径使用本地 DRAM，不包含 CXL/RDMA。
- Offload attention 入口消费已召回的 HBM records 与物理 ID，复用 device-only MLA。
  融合 prefetch 属于 indexer；当前 attention kernel 本身不读取 host backing。
- 主 MLA 的 64 维位置分量用 interleaved 配对（`is_neox=False`），indexer 前 64 维用
  split-half 配对（`is_neox=True`）。Attention scale 按原始 192 维 QK 计算并应用 YaRN mscale。
- Indexer RoPE 后直接进行 FP8 量化，不执行 Hadamard；精确 top-k 选择当前量化 logits
  中的最大值。该变更可能改变量化值与选择，不能视为与旧 Hadamard 输出逐位等价。
  RoPE、norm、SiLU 与 top-k 复用 FlashInfer；量化按 KDA 流程独立验证和测量。

## 相关实验

当前前三层 benchmark 与 profile 见
[SM90 ECHO prefill/extend](../../experiments/deepseek_v32_echo_prefill/README.md)。
十 block GR 工作负载的固定 P/NH 容量检查见
[统一 cache management 实验](../../experiments/cache_management/README.md)。
原有 DeepSeek / SM120 实验的有效历史结果保存在
`local/experiments/legacy/deepseek_v32/`，不进 Git，仍按原 run ID 和测量环境解读。

共享 GR 请求生成使用 [request_format.py](request_format.py)：DeepSeek 历史请求模板与长上下文预算适配。
用法见 [GR 生成器](../../GR/README.md)。

## 单卡 GR Serving 工作负载

共享 ECHO cache 与模型级 chunk 调度已接入。固定 P/NH 的静态规划、实际分配和
完整请求验收分别报告。十 block 本地四方案已完成 H65,536/A128、16 用户两轮的
独立数值验收及三次正式计时；官方路径另按固定容差完成独立验收。这些运行没有
填满 NH，也不能代表任意用户数或 H/A 配置。容量范围与可核验结果见
[统一 cache management 实验](../../experiments/cache_management/README.md)。
旧 4 GiB / W / chunk 对照已撤回，不再据此指定默认 chunk 或给出性能排名。

[adapter.py](execution/adapter.py) 提供本次 GR serving 对照使用的单卡工作负载：
从真实 checkpoint 前三层重复加载 10 个独立的 dense block，source 顺序为
`[0,1,2,0,1,2,0,1,2,0]`，不执行 MoE。每个副本复制其 source block 的 hidden 与
residual 输入，并使用独立权重、主 KV 和 indexer cache。加上真实 embedding、final
norm 和 LM head 共 7,827,793,408 参数；其中 dense backbone 为 5,974,428,160 参数。
这个工作负载用于控制计算和缓存大小，不能作为经过训练的 DeepSeek 8B 模型；它与
上述非 GR 前三层 benchmark 分别报告。

`DeepSeekServingBackend.runtime_driver(policy)` 接入公共 token runtime 契约，每个用户 session 在多次请求
间保留固定 prefix。策略包括 `hbm`、`echo`、`serial_sparse` 和 `dense_prefetch`：
前两种复用 resident / ECHO 路径，串行 sparse
在 indexer 与精确 top-k 完成后召回缺失 records。通用预算模式的 dense 使用两块
完整 layer staging 和独立 CUDA stream，在当前 block 执行时预取下一层的全部
历史主 KV；固定 P/NH 的 dense 直接使用逐层 pool，见下文。四者均执行相同的
稀疏选择与 MLA。所有用户 cache 的主 KV、indexer、映射、ECHO counter 及 dense staging
在 budget 模式下都计入 cache budget；权重和临时激活另计。固定 P/NH 模式按指定
pool/arena 容量执行，并记录物理显存和 DRAM 占用；cache 统计仍不等于进程峰值。

模型装配已拆分为 replay、session、资源规划和 pipeline。资源与 session 计划列出具名
分配项，创建 session 时消费同一份不可变计划；共享生命周期管理 owner、generation、
执行 lease 与失败后的资源保留。逻辑 token ID 通过 `TokenSelection` 传给 attention，
不新增 tensor 搬运或改变 indexer/prefetch 融合顺序。`num_layers` 由调用入口显式
传入；GR 入口与当前实验使用十个物理 block。执行时把 session 的 attention runner 和
chunk 大小作为参数传给 block，模型不再临时挂载用户 attention/cache。官方 ECHO 由
`build_official_backend` 装配同一 backend，直接构造官方 runner 和 cache view，并单独
预留、核验和释放官方额外资源。数值验收、正式计时和诊断 profile 使用独立入口；
报告中的 run ID、源码和测量边界共同标识结果。当前正式发布状态为
本地三次 bench、独立 profile 和官方 bench/profile 已验收发布；P0/当前对照仍保留部分阶段延迟增加，当前 profile 未单独确定其原因；结果入口见
[motivation](../../experiments/deepseek_v32_motivation/README.md)和
[官方 ECHO](../../experiments/deepseek_v32_echo_official/README.md)。

GR 的 `echo/serial_sparse` 在 session 计划中将保留容量设为 H，
host pages 与私有 history indexer 都按 H 保留；`extend_candidate` 使用 backend
共享的 GPU 临时空间处理候选。每层主 KV storage 为 `[P+1+Amax,576]`，前 P+1 行
包括 P 个历史槽和一个 sentinel，保留原映射；尾部只存本次候选，不分配 host ID 或淘汰元数据。
candidate 一次整批进入各层。当前层的候选 indexer 与该用户 history 直接拼入一层
共享的 `[H+A,d]` K 和对应 scales workspace，容量按执行上限预留；不保留逐层候选
indexer。候选相关共享空间不乘以用户数；各层和各用户的历史数值仍独立。

候选正常结束后 discard 临时 KV/indexer 状态，恢复 history 的有效长度和 indexer
hint；相关异步使用结束后才能复用共享空间。候选失败直接报错终止，runner 释放该
用户 session，不恢复或重试。NH 按 `ceil(H/64)×64` 准入，
因此在已规划的 A 和 H+A 上限内改变 A 不会重建相同 history。请求仍执行全部候选
hidden 和末 token logits，history-only 只改变保留容量。
通用 budget 模式的 `hbm/dense_prefetch` 保留原 session 容量及 extend 后 truncate 的接口；
直接 `backend.extend` 与上述非 GR `DeepSeekEchoModel` 继续持久提交新增 token，并
保留原有 chunk 执行语义。

固定 P/NH 模式已接入四方案：HBM-only 按 H 扣除独立 HBM token 配额 P，
`echo/serial_sparse/dense_prefetch` 按 H 扣除 NH 的 host 页配额，并复用相同的逐层
P 槽历史 cache。固定模式 dense 要求 H<=P，使用 [cache/prefetch.py](cache/prefetch.py) 在独立 stream
提前读取下一层完整历史中的 miss，命中直接复用；每层 pool 本身提供独立拷贝目标，
不另分配完整 layer staging。HBM-only 的 candidate 紧接 resident history，offload
的 candidate 使用 pool 尾部；四方案都使用共享的合并 indexer workspace，结束后
丢弃候选。端到端对照见 [motivation 实验](../../experiments/deepseek_v32_motivation/README.md)。

通用 budget 模式的 dense 将双缓冲及其 stream/events 归于 backend，每个用户仍独立持有历史
host records、映射和 indexer 状态。一次完整 prefill/extend 借用双缓冲，槽位覆盖前
等待前一个 consumer，归还前等待包括未消费预取在内的全部异步操作。
runner 从构造到 close 绑定唯一准入 owner；其关闭只释放 session，最外层再关闭
backend。真实三层验收 `refactor_three_layers_check_20261005_02` 在 H65,536/A1,024
下比较完整 extend hidden、末 token logits、默认输出与 resident/offload prefix
logits，五项比较均逐位相等；它沿真实第 0–2 层顺序传播，不使用 C10 输入重放。
对应 bench/profile 的发布结果见
[三层实验](../../experiments/deepseek_v32_echo_prefill/README.md)，当前状态为
独立 bench/profile 已验收发布，当前报告覆盖真实前三层，未采集 NCU replay。

C10 本地验收 `refactor_final_deepseek_check_20261005_01` 保存 128 份完整候选输出，
96 组 offload/HBM 对照逐位一致。官方验收
`refactor_final_official_check_20261005_01` 保存 HBM、独立 HBM 重跑和官方 ECHO
共 96 份输出，全部通过预先固定的数值门槛，但比较请求没有逐位相等。
官方门槛与本地 exact 对照不能互换。模型测试与这些数值 check 不替代固定 P/NH
容量实验、正式计时或未运行的配置。

`sparse_pool_tokens` 配置 backend 每层共享容量，`host_arena_tokens` 配置全局 host
容量；原 per-session `slots` 参数已移除。固定 P/NH 模式按实际 prefill chunk 与
candidate 大小执行，不使用独立 `workspace_query_tokens` 准入。通用 budget 模式仍
可显式指定该预留上限。prefill 的每个 chunk 依次执行十个 block，一次请求只提交一次；
GR candidate 固定整批执行，不随 prefill chunk 再拆分投影、MLP 或 indexer。资源计划
分别保留 H、A 和 H+A 的上限；候选临时执行不提交持久 history。

输出包含全部 candidate normalized hidden，同时执行最后 token 的 LM head；当前 NOSA
serving backend 只输出 hidden，其执行边界不同。跨方案数值对照使用同一模型的相同边界。

单元测试及可选的真实 checkpoint 正确性检查：

```bash
source .venv/bin/activate
python -m pytest models/deepseek_v32/tests/test_serving_backend.py -q
DEEPSEEK_SERVING_CHECKPOINT=/preset-models \
  python -m pytest models/deepseek_v32/tests/test_serving_checkpoint.py -q
```

真实 checkpoint 检查使用独立空 cache 构建 2,304-token prefix，并比较 16 / 23-token
candidate 与截短后的复访；所有策略的全部 candidate hidden，以及每个副本相对其 source
的 hidden / residual 均通过 bitwise 对照。这是正确性验证，serving 性能结果由对应实验
单独报告。GPU candidate 的短检查同样比较 2,304-token history 与 16 / 23-token
候选的全部 hidden/logits，确认与 HBM 逐位一致、候选 D2H 为零、history KV 与 indexer
不变；该工程检查不提供完整多用户容量或显存峰值结果。实现契约见
[模型规则](AGENTS.md)，完整请求与测量边界见上述对应实验。
