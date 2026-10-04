# Loop motivation 的研究边界与交接

更新：2026-10-03。主条目 1.4，关联 1.3、2.1–2.5、4.1–4.3；任务 T-006/T-007，
依赖 T-008/T-003 的系统验收。本轮为计划更新与只读核对，没有生成数据或运行实验。

## 研究者已选择

- 固定 U 个 user，按同一顺序完整遍历 R 轮，不依赖热度或随机访问概率。
- 用这份 loop 数据做 motivation，对照 HBM-only、ECHO、full prefetch、
  sync sparse loading、async sparse loading。
- cache 系统仍在实现，相关新实验数据尚未产出；局部工程验证不视为性能结果。

沿用当前固定 history、变化 candidate 的 prefill/extend 生命周期。用户访问顺序可写成
`[0, 1, ..., U-1]` 重复 R 次。R 至少为 2 才能包含复访；具体 U/R、history/candidate
长度、chunk 与预算尚未冻结。此前的热度 IID 方案不再是本轮执行要求，旧文档中的
16 用户、两轮、64K+128 只是此前尝试的参数，不自动成为本次选定配置。

## 可以直接推导的量

完整执行时总请求为 UR，首访 U，复访 U(R−1)，首访比例 1/R。相邻同用户请求的
序号间隔为 U，中间有 U−1 个不同用户。首访/复访按访问次数判定；session 被淘汰后
再次访问仍是复访，必须计入重建成本。每种方案从独立空 cache 开始，单独预热不能
消除正式首轮；不同 baseline 使用同一份请求及 token 身份。

等大小 session、严格 LRU、固定容量 C、history 身份不变且没有其他失效时，U≤C
会在首轮后全部命中，U>C 会形成循环扫描的全复访 miss。这是调度推理，尚非实测。
它描述整个用户 session 的保留，不等同于 ECHO token pool 的命中率或 NOSA 页缓存。

建议根据每种方案实际可准入的容量 C_b，选择共同 U 点覆盖容量以内、HBM miss 而
offload hit、双方均 miss 的区间；只有实际容量允许时才存在中间区间。各方案共享
staging、索引、映射、scratch、pending append 与 session 开销不同，C_b 不能只用
历史 KV 字节计算，也不能沿用旧实现的 7/31 等数字。U 控制容量压力，R 控制首访占比；
多规模比较宜保持相同 R，完整运行每一点，不截断用户覆盖或最后一轮。

## baseline 含义与现有映射

| 目标 baseline | 本轮要比较的语义 | 当前测量入口 |
|---|---|---|
| HBM-only | 相同 HBM 硬预算下保留用户历史，超出容量后的重建计时；不使用 DRAM cache | 两模型的 `hbm` |
| ECHO | indexer 内预取、共享有限 pool 与精确 recall 的具体方法；不默认归为 sync 或 async 类 | DeepSeek 的 `echo`，新系统验收未完成 |
| full prefetch | 逐层提前搬运完整历史主 KV，双缓冲；attention 保持相同稀疏语义 | 两模型的 `dense_prefetch`，共享适配进度分别核对 |
| sync sparse loading | 完整 query batch 的稀疏并集取齐再计算，作为强串行对照 | 两模型的 `serial_sparse`；超 pool 拆分及重复搬运要披露 |
| async sparse loading | 稀疏取数与 attention 重叠，和 sync 保持选择/数值及可比缓存条件 | NOSA 的 `overlap` |

五类是目标集合，当前每个模型只有四类入口。ECHO 的 NOSA 适配、或 DeepSeek 的
对应 async 路径尚未由本轮确定；不能把两模型的绝对延迟拼成五类排名。NOSA 只输出
candidate hidden，DeepSeek 工作负载替身另执行末 token LM head。共同模型、输出和
选择语义的可比范围由 T-003 明确；full prefetch 不改成 dense attention。

## 当前代码支持与待办

`experiments/gr_serving/src/workload.py` 的 sequential 分支使用显式合成 ID，不读
热度；`GR/scheduling.py` 按 `order % U` 循环。现有实验入口用
`--sampling sequential --users U --requests T` 且 T=U×R 可产生完整 loop。
工作负载逐请求检查固定 prefix、相邻访问 candidate 变化及 visit index 连续，并保存
token hash。已有双遍测试代码，本轮未运行。

目前没有显式 rounds 参数或完整轮数校验；多个 users 档共用同一个 requests 值，
不能自动保证相同 R。T-006 应补齐或在编排中明确这些约束，并记录 round index、
每轮实际用户覆盖与请求身份。普通 `serving/run_multi_user.py` 仍固定 weighted，
实验入口的支持不能表述为所有入口都支持 loop。

## 实验要回答的问题与修正条件

- 容量：增加 U 后，HBM-only 的复访重建是否带来可量化开销？若各方案实际容量
  相近，或 offload 的保留成本抵消收益，应缩小或修正容量 motivation。
- 搬运：full prefetch 与 sparse loading 的实际历史搬运量、取数等待和总延迟如何
  变化？较少字节不自动意味着较低延迟。
- 重叠：async 相对完整并集 sync 是否改善完整复访延迟？若只有内部区间相交而
  没有整体收益，不把 overlap 本身作为系统加速结论。ECHO 的差异需在可比域解释。

建议同时报告完整 loop、每轮、首访、复访命中/重建的数量与延迟，保留 miss 构建、
管理、搬运及计算成本。HBM/DRAM 预算与实际使用分开，权重、普通 activation 与
进程峰值另报；共享资源不能在多用户间重复计费或遗漏。性能以未插桩完整请求为准，
唯一读取和内部 overlap 另采样核验，遵循项目两类区间与逐样本验收要求。

先完成公共 serving 所有权、完整硬预算和稳定源码验收，再运行正式 loop 与对应
profile。以新 run ID 发布受影响实验，通过验收前保留旧报告的原 run/源码/测量边界，
不改写旧数字。本计划只支持受控系统 motivation；真实推荐任务、热度、到达时间、
并发吞吐和质量仍需独立证据。工程进度依据见 [S-020](sources.md)。
