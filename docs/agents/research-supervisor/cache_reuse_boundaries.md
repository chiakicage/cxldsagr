# ECHO、NOSA sparse loading 与 dense prefetch 的 cache 复用边界

日期：2026-10-03。主研究条目 3.2，关联 2.1、2.4、2.5、4.1、4.2。
研究者说明另一 agent 正在修改 ECHO cache，本轮仅核对工作区并讨论其他方案如何复用。
以下为设计建议，未选定或实施 NOSA 改造，没有新增 GPU 验证或性能结果。
后续已按研究者要求制定[修改计划](../system/nosa_shared_cache_implementation_plan.md)，
包括文件分工、资源上限、异步验收和补测范围；当前仍未实施。

## 判断与当前依据

可以共享资源规划、预算准入、所有权与异步生命周期；各模型保留记录布局，
各方案保留取数与驻留策略。当前 ECHO 的具体 token pool 不能直接作为 NOSA 的完整后端。
若统一资源所有权后改变 session 容量或方案排名，应重新判断原结果中缓存管理与取数
调度各自的影响，不能将变化全部归于 overlap。

| 当前材料 | 本轮核实的边界 |
|---|---|
| [ECHO 实施计划](../system/echo_cache_implementation_plan.md) | backend/device 拥有 host arena 和每层有限 HBM pool；session 保留页表、独立 indexer 与事务。实施中，完整验收以执行 agent 的交付为准 |
| [共享 serving 契约](../../../executor/serving_backend.py)与 [PrefixSessionPool](../../../cache/prefix_pool.py) | 工作区已出现 `SharedCachePlan`、先规划后分配、shared/session 分开计费和 host-page 准入；NOSA 尚未接入这一可选接口 |
| [NOSA offload cache](../../../models/nosa/offload_cache.py)与 [fetch workspace](../../../operators/nosa/attention/offload/api.py) | 每 session 一份跨层复用的完整逻辑地址 staging；串行与 overlap 使用同一 workspace 类型。不保留跨调用的 HBM 热块，没有有限 slots/eviction |
| [NOSA serving](../../../models/nosa/serving.py) | dense prefetch 每 session 分配两份完整层 K/V staging，copy stream 提前搬下一层；仍执行完整 NOSA sparse attention |
| [ECHO 共享池](../../../cache/sparse_token_pool.py) | width/dtype 参数化，但现有预取接口、token 映射和事件 priority 仍带 ECHO 策略语义，不能只改 width 就用于 NOSA |

## 建议先复用的部分

1. 复用 `plan_resources → allocate_shared → create_session` 和 shared/session 计费契约。
   backend 持有与模型/device/布局兼容的执行资源，session 持有独立历史、长度和派生记录。
   不跨用户共享历史内容或 indexer 数值；跨方案对照也不共享会污染测量的运行状态。
2. NOSA 串行 sparse 与 overlap 使用相同缓存实现、布局、预算和初始状态，只切换调度。
   当前可先将一份 `NosaFetchWorkspace` 从每个 session 移到 backend，串行请求借用。
   先保留现有连续 host K/V 与逻辑寻址，避免同时引入 paged host 和 finite HBM 两种变化。
3. dense prefetch 同样将双层 staging 移到 backend；当前层消费与下一层搬运重叠，
   slot 再用前等待旧 consumer，读取前等待 copy 完成。它按整层历史搬运，
   不继承 ECHO speculative prefetch、token priority 或 sparse miss 路径。
4. 共用事务原则：所有模型层成功后统一提交；abort/truncate/release 等待相关异步使用；
   KV、CIS 和压缩派生记录统一推进/回退。共享 buffer 释放归 backend，session release
   仅归还租借资源并释放自身状态。可共享的 indexer scratch 与 session 派生数据分开处理。

对于当前串行 serving，仅 K/V staging 的占用可从“各 session 容量之和”变为一份
预先规划的最大活动容量；dense 对应两份。CIS、压缩记录和待提交 append 仍须计费，
这不使 idle NOSA session 成为零 HBM。跨请求共享 staging 也不等于保留跨请求热块。
ECHO 的每层持久 pool 与 NOSA 的跨层 staging 有不同生命周期，不强制二者使用相同层数倍数。

## 有限 HBM 页缓存属于后续独立改造

若要让 NOSA 保留历史热块，应显式建立有限页池；可复用 allocator/lease/映射管理的基础，
但需要 NOSA 专用的模型与算子适配：

- 访问身份为用户/历史版本、层、KV head、64-token logical block；K/V 成对管理。
  ECHO 的 64-token host 分配页和 NOSA 的每 KV head attention block 语义不同。
- attention 保留逻辑选择以计算 causal/CIS，实际 K/V 访问使用物理 slot 映射。
  当前 TMA、padding、suffix 和 ready 都依赖逻辑 staging，必须同步适配。
  若还接入非连续 host arena，现有连续 host tensor 契约也要改，不能只共享页表。
- loading/ready/consumer 使用状态与映射分开管理；copy 发布 ready 后才能读取，
  consumer 完成前不能驱逐或复用 slot。HBM 淘汰只移除副本，DRAM 有效历史仍保留；
  回收 host 身份前等待旧引用结束，防止跨用户误用。
- 保持按 `(KV head, block)` 去重和当前唯一历史读取要求。当前整批 attention 所需并集
  若超过池容量，不能靠循环驱逐尚未消费的页或裁剪 selection 解决；需明确拒绝该几何，
  或另行设计并验收受限容量的执行调度，披露重读与拆分成本并增加匹配串行对照。

通用接口不能要求“返回前全部 KV 已在 HBM”，否则 NOSA attention 内的 loading overlap
会被串行化。共享部分管理身份、容量与使用权；ECHO 和 NOSA 各自的算子负责实际搬运及
与消费者之间的发布协议。ECHO 的推测预取规则不作为 NOSA 的默认淘汰策略。

## 预算与比较要求

预算按 `shared 固定资源 + 所有 session 持久状态 + 活动执行峰值预留` 核算，每份实际
分配只计一次。shared 预分配的 host arena 按物理容量计费，session 另受占用页数限制；
继续独立分配 host backing 时则按 session 计费，两者不能重复或漏计。
metadata、CIS/派生记录、cache scratch、pending append、in-flight copy source 都包含，
权重和普通 activation 单列。

比较调度时固定同模型的 cache 策略、容量、精确选择和已验证的起始状态；串行 sparse
先搬完完整 query batch 的唯一并集，再执行 attention。比较 serving 时固定预算与完整
轨迹，允许各方案产生自然 residency，分别报告首次构建、复访重建、历史命中及实际流量。
若 dense 双层完整 staging 放不进预算，明确该配置不可行；分块 dense streaming 应另报。

建议实施顺序是先等待 ECHO 共享接口稳定，再接 NOSA 的 shared workspace 与 dense 双
staging，最后单独评估有限 page cache。建议归 T-003/T-007，未启动这些实现任务。
任何执行路径改动均须按项目规则重新验收并补测受影响实验，本次讨论不替换旧报告。
