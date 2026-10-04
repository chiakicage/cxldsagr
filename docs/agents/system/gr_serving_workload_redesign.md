# GR serving workload: heat-weighted independent sampling

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

2026-10-02；按用户最新明确的规则收敛：以已有用户访问次数构造概率分布，给定总访问次数
T 后有放回随机抽样。取消每用户最多 8 次的默认限制。此前 4N、最少一次复访、最大余数
配额和全局打散方案不再作为当前协议；相关代码已撤回，其 CPU 数字不用于当前请求流。

## 当前协议

已有逐用户次数 c_i 时，使用 p_i=c_i/sum(c)。现存 Beauty 累计热度曲线可通过
`HeatPopulation.from_curve` 近似生成对应权重；这是合成身份与热度近似，不是恢复原始
逐用户记录。原始日志不是构造该概率分布的必要条件。

从固定概率向量独立、有放回抽取 T 次，随机种子固定。用户 i 的实测次数 n_i 决定：

- 实际访问用户数：N_actual = count(n_i > 0)。
- 复访用户数：count(n_i > 1)。
- 用户复访次数：r_i = max(n_i - 1, 0)。
- 总复访数：T - N_actual。

不为每个用户增加首访票，不分配复访配额，不保证所有用户出现或复访，也不因零 miss
重新挑种子。每个用户的访问次数边际服从 Binomial(T,p_i)，联合计数服从
Multinomial(T,p)；概率决定的是期望频率，不是每一条短 trace 的精确配额。

用户池大小是概率向量的维度，实际人数是输出。仅有归一化曲线不能唯一确定池大小；
若采用原 Beauty 统计中的 22,332 人，则重建同样数量的合成热度权重。改变用户池大小
是在改变概率模型，不能把该参数直接称为实测活跃人数。N_actual 的期望为
sum_i[1-(1-p_i)^T]。

## 现有代码与边界

`ScheduleConfig(sampling="weighted", max_revisits=None)` 已支持上述抽样；constant 或
poisson 只控制模拟时间戳，用户抽样随机流与时钟随机流独立。timeslot 带冷却约束，
不属于当前 IID 协议。固定 history / 变化 candidate、首访和复访按访问次数识别、
逐用户 LRU 与双预算管理继续使用现有实现。

测量和单模型入口取消默认 8 次上限，旧 cap 仅保留为显式兼容选项。manifest 除
unique_users / first_visits / revisits 外，记录 returning_users 和逐用户 revisits；
独立审计支持无 cap，并核对实际覆盖与复访统计。原始日志时序的回放不再是当前前置条件。

固定 history 4K/16K/64K 分开运行，模型内各方案共享完整请求流。总请求预算约束执行量，
并不自动保证整轮能在 30 分钟内结束；首访和复访 miss 的历史构建成本仍需先估计。
后续采用不同 T 时可使用同一固定概率模型、同一种子的流前缀。原先七档人数是用户池
参数，不能继续当成实际人数的实验横轴。

## CPU 可行性检查

以下不是新 GPU 结果。现有 Beauty 曲线近似为 22,332 人，population seed=42；访问
seed=42/43。T=1024 时实际用户为 981/986，复访为 43/38；T=4096 时实际用户为
3532/3521，复访为 564/575。用户人数均由抽样得到，没有强制填充。

按 NOSA 16K 旧配额对应的 HBM/offload 容量 7/31 人做 LRU 模拟，T=4096 时 HBM
复访 miss 为 562/573，offload 为 554/566；两者都接近全 miss。T=128、seed42 没有
复访，复访 miss 率为 N/A。这说明 IID 热度构造可用于研究，但给定 T、概率模型和
缓存预算不自动产生有区分力的容量对照。需检查完整 miss 曲线，不能通过强制配额或
换有利 seed 掩盖不足。尚未基于这条新协议重新运行 GPU 或发布性能结论。

数据与时序边界：现有曲线足以构造热度驱动的合成请求流；它不提供跨用户真实时间顺序，
也不证明生成的历史/候选内容有真实推荐标签。[原始行为轨迹考察](gr_serving_dataset_trace.md)
保留为替代路线说明，不作为当前阻塞条件。
