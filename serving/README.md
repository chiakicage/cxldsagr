# 本地 GR serving

通过 [GR 请求生成器](../GR/README.md)在单进程、单 GPU 上串行执行 DeepSeek V3.2 请求。
`serving.run_multi_user` 跨请求保留用户固定 prefix。预算模式按 HBM/DRAM 字节额度做
session LRU；固定容量模式按显式 P/NH 准入。该入口没有网络服务、批调度或模拟到达等待。

## 持久多用户入口

从仓库根目录运行；默认使用 DeepSeek V3.2 的 HBM 方案：

```bash
python -m serving.run_multi_user --help
python -m serving.run_multi_user \
  --model deepseek_v32 --scheme hbm --model-path /preset-models \
  --num-users 8 --count 16 --history-tokens 16384 --candidate-tokens 1024 \
  --hbm-budget-gib 1 --dram-budget-gib 16 --device cuda:0
python -m serving.run_multi_user \
  --model deepseek_v32 --scheme echo --model-path /preset-models \
  --resource-mode fixed-pools --sparse-pool-tokens 32768 --host-arena-tokens 1048576 \
  --num-users 16 --count 32 --history-tokens 65536 --candidate-tokens 128 \
  --context-limit 65664 --chunk-size 1024 --device cuda:0
```

DeepSeek 支持 `hbm`、`echo`、`serial_sparse`、`dense_prefetch`。这里使用 checkpoint
前三层的独立权重副本和对应层输入构成约 8B 的 C10 代理负载，不含 MoE；具体模型
与输出范围见 [DeepSeek 模型](../models/deepseek_v32/README.md)。

用户访问遵循 GR 中的 Beauty 热度曲线，加权有放回采样，默认 seed 42。
默认不设复访上限，按固定热度概率独立有放回抽取 `--count` 次。实际人数和复访次数由
采样结果统计，`--num-users` 是概率向量的用户池大小。旧 `--max-revisits` 仅在显式设置
时限制复访，达到上限的用户退出抽样，此时总数为上限。不强制覆盖用户或补齐次数。
`--history-tokens` 是固定指令加 history 的完整 prefix 长度；`--candidate-tokens` 是完整
变化 suffix 长度。复访时验证 prefix token identity，只保留固定 prefix。
DeepSeek `echo/serial_sparse` 通过 `retained_session_capacity` 按 H 准入，调用
`extend_candidate` 一次整批执行 GPU 临时候选，不沿用 history prefill 的 chunk 划分；
正常结束后 discard 候选，不推进 history 长度或覆盖其 indexer 状态。候选失败直接
报错终止，runner 释放该用户 session，不恢复或重试。在已规划的执行上限内改变
候选长度 A 不会导致相同 history 重建。本公共 CLI 的 DeepSeek `hbm/dense_prefetch`
后端仍使用原有的持久 extend 后 truncate 生命周期；
直接调用 `backend.extend` 也继续持久提交新增 token。
`is_revisit` 与实际 `prefix_cache_hit` 分别报告；被淘汰后的复访仍计入复访。
默认每层 query chunk 为 1024。DeepSeek 使用 `--sparse-pool-tokens` 配置逐层共享
token pool（P），`--host-arena-tokens` 配置共享 host arena（NH）。`--resource-mode fixed-pools`
要求显式 NH，不接受 HBM/DRAM 字节子预算和独立 W；workspace 根据实际 chunk 与 candidate
大小自动规划。indexer 和临时空间仍实际占用显存，但其预留量不再触发 session 淘汰。
NH 按 `padded(H)=ceil(H/64)×64` 为每个用户分配页；同长度 history 可容纳的用户数为
`floor(NH/padded(H))`。候选 KV 和一层 `[H+A,d]` 合并 indexer K/scales workspace
只计入共享 GPU 资源，不乘以用户数。P 只表示每层可淘汰的历史 HBM 槽数，候选 GPU
尾部另按 A 上限预留。
该模式支持 DeepSeek `echo/serial_sparse`，不保证任意 P/NH 都能装入机器，运行失败须明确报告。
预算模式仍可用 `--workspace-query-tokens` 声明最大执行 workspace；原 `--deepseek-slots`
已移除，显式使用会报错。
`deepseek_v32_motivation` 通过 backend/runner 接口运行固定 P/NH 的四方案对照。
该入口的 HBM-only 使用独立 HBM history token 配额 P；三个 offload 方案使用 NH
host 页配额和相同的 P 槽历史 cache，dense 仅预取完整历史中的 miss。四方案候选
均整批在 GPU 临时执行。独立 token 配额不借用 host pages，也不换算成字节子预算。

预算包含缓存持有的 KV、派生记录、metadata、staging 和 cache scratch；模型权重及一般
计算临时内存不在 cache budget 内。HBM 方案不使用 DRAM 历史 backing，其 CPU
验证和统计 scratch 仍计费。

runner 从构造到关闭独占 backend 的准入所有权，构造失败回滚本次新资源，保留原有
有效 plan。CLI 分别声明 H 的保留上限、A 的执行上限和 H+A 的上下文上限，越界请求在
准入前拒绝。history-only session 不缩小完整请求的执行范围。共享候选空间只能在
执行 lease 内借用，复用前须等待相关 GPU 操作完成；关闭 runner 释放用户，再由外层
关闭共享资源。
DeepSeek GPU candidate 路径已完成 C10、16 用户两轮的独立数值验收和正式测量，
请求内存观测与静态容量计划见[DeepSeek cache 管理报告](../experiments/cache_management/README.md)。
该轨迹未填满 NH，也不验证离线最大 P/NH 的物理容量。

stdout 输出启动配置、逐请求 JSON 统计及 hidden shape/dtype/device，最后输出请求数、
复访数、命中数、淘汰数和平均延迟；不输出 hidden 向量或输入 token 数组。
该入口没有独立预热，逐请求时间包含首次使用时触发的 JIT 编译，排除权重加载和 GR 文本生成。
正式固定容量对照使用[motivation 入口](../experiments/deepseek_v32_motivation/README.md)，
数值验收、计时和 profile 分开。不能把本 CLI 首次运行时间当作该实验结果。
成功或失败退出时均关闭持有的用户 sessions 和 backend 共享资源。
