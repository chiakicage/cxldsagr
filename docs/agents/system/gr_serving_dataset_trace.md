# 从行为数据采样 GR 复访

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

2026-10-02；对应研究条目 1.3、1.4、2.2、4.3。用户询问复访能否从已有数据集采样。
本次核对数据来源和构造协议，未运行新 GPU 实验，未实现新的生产调度器。

**后续用户选择：**当前采用已有访问热度构造概率分布，给定总访问次数后独立有放回抽样，
取消 8 次上限和强制复访配额。本文是未采用的真实时序替代路线；取得原始日志不再是
当前工作的前置条件。当前规则见[热度抽样协议](gr_serving_workload_redesign.md)。

## 已核实的本地条件

- `GR/analysis/README.md` 明确说明 Amazon Beauty/Games/Books/Clothing 的
  `timestep_map.json` 使用每用户交互时间的排名，不是全局秒级时间。
  `GR/heat.py::HeatPopulation.load` 累加用户交互次数，丢弃 timestep 键；
  `from_curve` 从累计曲线生成合成用户权重。这两条路径都不保留跨用户访问顺序。
- industrial CSV 来自归档中的合成数据生成流程，不能称为真实工业请求日志。
- `GR/dataset.py` 只读取 `dataset.pkl` 的 `meta` 商品标题；目前不能据此断言
  原 pickle 是否还保留真实时间戳，因为源文件未取得。
- provenance 中的 `/mnt/nfs/share/archive/260808_old_nfs/share/wsh/Bi-KV/Bi-KV/data`
  当前不可访问（`/mnt/nfs` 不存在）。限定检查本仓库及相邻的
  `cxldsagr-nosa-framework`、`cxl-recsys` 只找到相同旧路径，没有找到可用原始行为流。
  这不是对整个共享存储的数据盘点。

因此已有曲线可继续提供边际热度，但不能恢复真实复访顺序。按每用户 ordinal 对齐所有
人的第 1/2/3 次访问，会人为制造 waves；即使能产生 miss，也不属于真实时间回放。

## 候选采样协议

需要包含稳定 user ID 和全局 timestamp 的行为数据。优先考察 Taobao UserBehavior 的
浏览事件；原始 Amazon 数据若仍有真实时间戳也可使用，但评论事件是较弱的服务请求代理。
具体数据集尚未由用户选定，原始数据尚未导入。

候选官方入口：[阿里云天池 UserBehavior](https://tianchi.aliyun.com/dataset/649)。
2026-10-02 页面 HTTP 200，标题为“淘宝用户购物行为数据集”，页面元信息说明其用于
隐式反馈推荐研究。正文依赖 JavaScript，元数据接口在当前环境返回 403；本次没有从
官方正文或原始文件核验字段、时间范围和下载可用性。下述按 user/timestamp/事件类型
处理的规则是待检验的接入协议，不能称为已完成的数据适配。

1. 明确定义事件到请求的映射。浏览事件可作为推荐请求的代理，不直接宣称每个行为记录
   都是一次线上推理。只取一种事件类型，例如 `pv`。同一用户相同时间戳的记录可合并成
   一次候选集合；不同用户的 timestamp tie 使用独立固定 hash 排序，不能声称恢复了同秒
   内的真实顺序。这里的复访指同一用户的后续请求，不自动等于跨 session 回访。
2. 固定时间窗、数据清理规则与 seed，再从窗内至少有两次映射后请求的用户中均匀抽取
   N 人。N 表示真实出现且复访的用户数。七档可采用同一固定用户排列的嵌套子集；不按
   命中率选择用户。需披露这是条件于复访用户的子集，未涵盖只访问一次的用户混合。
3. 每人保留前 `min(k_i, 9)` 次请求，合并后按原时间排序，所有 cache 方案共享同一条
   trace。复访数是 `min(k_i - 1, 8)`，由数据产生；总请求数允许为 2N–9N，首访比例
   也随数据变化。原候选 4N、25% 首访是合成控制条件，不再强加给数据采样。
4. 固定 history 与访问轨迹分开。三档 4K/16K/64K 复用相同访问序列；每个用户的
   history 在观察窗前冻结。若数据不能提供对应长度的模型输入，继续明确使用合成固定
   history，不能把事件轨迹的真实来源扩展成请求文本或推荐任务质量的真实性。
   Candidate 的商品/类别信息与文本生成方式单独记录；不能用观察窗后的行为补历史。
5. 先做 CPU 复用距离分析。等大小 session 的 LRU 在两次访问间不同其他用户数
   `D >= C` 时 miss，C 是包含全部 cache 附属开销后的用户容量。报告全部预定采样的
   实际人数、复访次数分布、相邻复访比例、D 分布，以及各预算下的复访 hit/miss。
   主容量比较须覆盖 HBM miss/offload hit 和两者均 miss；若数据缺少某区间，说明
   覆盖不足并独立安排预算敏感性或合成压力对照，不按 miss 重抽有利窗口或 seed。
6. 为控时可按顺序连续执行并省略日志中的空闲等待。此时测量是串行请求执行和缓存行为，
   不是原始 QPS、排队延迟或并发吞吐。先用完整 trace 的首次与预测重建次数估算成本，
   再分批运行完整实验点；不截短到未覆盖声明的 N 名用户。

用户子集会删除未选用户的请求，8 次复访封顶还会删除热用户后续请求；两者都会改变并
通常减小复用距离。因此结果应称为“由真实行为日志采样的受限用户轨迹”，不能声称回放了
完整线上流量。保留原始文件 checksum、窗口、过滤/合并规则、seed、逐用户次数、完整
trace 及 hash，报告过滤前后的分布变化。

## 与已有设计的关系

[4N 热度协议](gr_serving_workload_redesign.md)保留为受控合成对照，其 CPU miss 和时间
推算不适用于新数据轨迹。后续优先取得真实全局时间数据、冻结采样规则并做 CPU 预检，
再决定约 30 分钟内能完成哪些完整 GPU 点。现有模型成本不因换用真实数据而消失；不承诺
两模型、七档实际用户及三档历史的完整矩阵可在 30 分钟内结束。
