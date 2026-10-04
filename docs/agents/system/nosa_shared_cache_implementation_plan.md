# NOSA 共享执行缓存与 dense prefetch 修改计划

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

日期：2026-10-03。状态：**两侧共享缓存、公共集成及正确性验收已完成；性能补测已取消**。
主研究条目 3.2；关联 2.1、2.4、2.5、4.1、4.2。依据见
[复用边界讨论](../research-supervisor/cache_reuse_boundaries.md)。
执行进展、源码身份和验收边界见[工程 checkpoint](nosa_shared_cache_checkpoint.md)。
DeepSeek 接口已交接，公共协议和 runner 由 root 顺序集成。
用户随后明确要求“保证正确性就行了，你不用再跑性能测试了”。因此当前交付只要求
实现、数值、生命周期和缓存预算正确性；第 8 节及原发布范围中的性能测量、性能
profile、默认性能选值和结果替换不再执行，也不作为本任务未完成项。旧结果保留
原始身份，新实现明确标注性能未测。

## 1. 目标与实施范围

在当前单 GPU、串行 GR serving 中，由 backend 持有执行缓冲，让不同用户 session
依次借用。保留每个用户独立的历史 K/V、CIS、压缩派生记录和事务；将共享字节与
session 字节分别纳入分配前预算。比较 serial sparse 与 overlap 时只改变取数调度。

分两个可独立验收的批次：

- **A：NOSA 主线。** 迁移 `serial_sparse` / `overlap` 的一层 sparse staging、队列与
  attention scratch；迁移 `dense_prefetch` 的双层 staging、stream/events，并补齐
  device-only FA3 scratch 的预算与复用。`hbm` 对照使用相同的 device-only scratch
  注入机制，完整 KV 仍按用户常驻。
- **B：DeepSeek dense prefetch。** 在 ECHO 共享接口冻结后，将其双层 staging 和
  异步执行状态迁至 backend，复用 A 的通用双缓冲生命周期。与当前 ECHO 复现分开验收。

本计划保持连续的 session host backing。indexer byte scratch 和 pending append
先保留现有 session 所有权与峰值预留；这些预留仍会限制用户容量，不声称已消除所有
执行态的重复预留。host arena 分页、NOSA 有限 HBM slots/淘汰、跨请求热块保留、
candidate 免 D2H、resident 冷构建、CUDA Graph 和并发服务均留给后续独立任务。
CUDA attention/indexer 数学、selection、CIS、causal mask 和 numerical repair 保持原语义。

预期得到的是更清楚的资源所有权和减少重复 staging 分配；延迟是否改善须由补测判断。
共享大容量 workspace 的初始化开销可能高于原小 session，不能预设一定加速。

## 2. 与正在进行的 ECHO 改造协作

当前工作区已有 `SharedCachePlan`、shared/session 分账及 ECHO shared pool。
ECHO 执行者已完成发布并由用户确认交接，见
[发布记录](echo_cache_publication_execution.md)。以下契约仍由本批集成验收。
[ECHO 实施计划](echo_cache_implementation_plan.md)、
[core checkpoint](echo_cache_core_checkpoint.md)与
[native checkpoint](echo_cache_native_checkpoint.md)不等于完整模型或性能验收完成。

P0 先取得其执行者交付的接口版本和源码身份，冻结以下契约：

1. `plan_resources(budgets, limits)` 纯估算，`allocate_shared(plan)` 先于 session 分配。
2. `shared_bytes()` 与 `session_bytes()` 不重复计费；没有共享 host arena 的后端
   `host_pages=0`，host backing 继续按 session 计费。
3. `PersistentGRRunner.close()` 只关闭 session pool，backend owner 负责最终 `close()`。
4. 相同 plan 可幂等复用；存在 live session/lease 时拒绝换 plan 或增长资源。
5. 一个 backend 同时只绑定一个准入 owner，范围为 runner 构造至 close，即使它暂时
   没有 session 也不能绑定第二个 runner。执行 lease 不能防止两个独立 PrefixPool
   各自按完整预算保留用户；P0 须补齐绑定/解绑接口及构造失败清理。

接口冻结不要求等待全部 ECHO chunk sweep，但冻结前不改 ECHO owner 正在维护的
`cache/sparse_token_{cache,pool}.py`、DeepSeek ECHO 模型/native 文件及 shared serving 接口。
公共文件由一个集成执行者顺序修改；不为复用强行重构 ECHO 的 token/FIFO policy。
保留工作区已有修改，不 reset/stash 他人工作。执行材料保存实际文件 hash 和 diff，
仅记录 HEAD 不足以标识当前实现。

## 3. 资源所有权和预算

### 3.1 所有权

| 资源 | 修改后的所有者 | 释放/复用边界 |
|---|---|---|
| NOSA sparse K/V staging、first-use/ready、fetch queue、FA3 scratch | backend 一份 `NosaFetchWorkspace`，跨用户及层复用 | 当前执行 lease 完成后才能借给其他 session |
| NOSA/DeepSeek dense 双层 staging、copy stream、ready/consumer events | backend 一组双缓冲 | 槽位按 layer 交替；覆盖前等待前 consumer |
| NOSA hbm/dense 的 device-only FA3 scratch | 各 backend 一份有界 workspace | 与当前完整 prefill/extend lease 绑定 |
| 历史 host K/V、resident KV、CIS、compressed K/CIS、stable pool、有效长度 | 每 session 独立 | commit/abort/truncate/reset/release 维持现有模型语义 |
| indexer byte scratch、各层 pending append 与 D2H source | 本批仍为 session | 继续按该 session 峰值预留，不混入共享 slab |
| 诊断 trace | 独立诊断 plan 显式预留 | 正式 plan 不开启，不允许开关触发未预留的增长 |

只共享执行存储，不跨用户共享数值状态。ECHO 的逐层持久 pool 也不复制成 NOSA 的
默认分配拓扑：NOSA sparse 继续只需要一层完整逻辑地址 staging。

### 3.2 上限与预留

固定负载入口显式传：

```text
C = max_session_capacity = history_tokens + candidate_tokens
A = max_candidate_tokens = candidate_tokens
Q = max(min(prefix_chunk_size, C), A)
```

`C` 不超过模型上限；显式 `A` 为正且不超过 `C`。未提供 A 的通用调用者按 C 保守
预留，不能默认 A 等于 prefill chunk。runner 仍要求非空固定历史。backend 在
`create_session/prefill/extend` 入口再次检查实际容量和 query 上限，在分配、begin_step
或 LRU 变更前拒绝不可能请求。精确固定负载应传 A，避免不必要的超大保守预留。

以层数 L、KV heads H、head dim D、dtype 字节 b、单 session 容量 c 记。
以下 HBM 式子描述逻辑存储组成；CUDA 预算须对每个独立 allocation 应用已核验的
allocator 上界。令 P(n) 为单次 pinned host 分配向上取整到 2 的幂后的容量，P(0)=0：

```text
NOSA sparse shared HBM = 2*C*H*D*b + sparse metadata + bounded FA3 scratch
NOSA dense  shared HBM = 4*C*H*D*b + bounded device-only FA3 scratch
NOSA hbm    shared HBM = bounded device-only FA3 scratch
NOSA offload session K/V payload = 2*L*c*H*D*b
NOSA offload session DRAM reserve = 2*P(L*c*H*D*b) + host temporary peak
NOSA session HBM = CIS/派生记录 + indexer scratch 峰值 + pending append 峰值
                  + 其他实际 session metadata
NOSA hbm session HBM 另含完整 resident K/V

DeepSeek dense shared HBM = 2*C*record_bytes + 有界执行 scratch/源存储预留
DeepSeek dense session host payload = L*c*record_bytes
```

NOSA 的 K、V 各自是一份跨层连续 host allocation，必须分别取 pinned bin，不能将
tensor 的逻辑 storage bytes 直接当作 DRAM 容量。当前 CUDA/BF16、L32/H2/D128 下，
c=66560 或 65664 时每个 K/V payload 略大于 1 GiB，各占 2 GiB bin；每 session
host K/V 容量因此为 4 GiB，另预留 sparse 的 16 B 或 dense 的 8 B 临时峰值。
CPU reference 不使用 pinned allocator，继续保留 payload 角色账本。默认 pinned
allocator 配置与不可观察的外部路由边界见
[分配验收 checkpoint](nosa_cache_allocation_checkpoint.md)。

DeepSeek `record_bytes` 由模型布局声明，不能写成通用 MLA 常量。其既有 indexer/copy
source 活动预留保持原有覆盖范围，迁移 staging 时不减掉仍然存在的临时分配；其
host payload 也不等于 allocator 容量，实际预留由交接后的对应实现核验。

所有实际分配只计一次。`shared reservation + sum(session reservation)` 同时不超过
HBM/DRAM 预算；实际字节不超过各自预留。会增长的 session indexer scratch 继续覆盖
旧/新 buffer 同时存活的峰值。权重、普通 activation 与 allocator allocated/reserved
峰值分别报告；执行后 `stats()` 不能代替瞬时分配审计。

共享 workspace 初始化时按 C/Q 预分配，正式运行不隐式扩容。纯估算与实际 storage
核验使用同一布局契约，并独立枚举真实 storage 去重验证。CPU 仅验证状态与分配容量，
不能将 CPU 参考账本当作实际 HBM 测量。

## 4. 接口与文件修改

| 文件 | 具体修改 |
|---|---|
| 新增 `models/nosa/serving_resources.py` | `NosaExecutionResources`：纯 plan/estimate、按 scheme 分配、lease、owner/generation、shared_bytes、close；只读模型显式传入的维度 |
| 新增 `cache/staging.py` | 用于两个模型 dense 路径的双层缓冲和串行借用机制；输入命名 record shape、capacity、dtype、device。只管理存储/events/使用权，不选择预取内容、不导入模型配置 |
| `models/nosa/offload_cache.py` | 增可选 resources 注入；区分 owned/borrowed；borrowed 模式禁止再次分配 workspace，stats/release 不重复统计或释放；begin_step 检查有效 lease |
| `models/nosa/cache.py` | shared hbm session 也持有可选 owner/lease guard，阻止绕过 backend 直接 forward 后退回未计费的默认 scratch 分配；owned 默认行为保持 |
| `models/nosa/serving.py` | 实现已有 SharedServingBackend 接口；create_session 注入资源；prefill/extend 获取 lease；dense 使用共享双缓冲；明确 backend.close 和 session owner 校验 |
| `operators/nosa/attention/offload/api.py` | 增纯字节估计及显式 query/trace reserve、有界模式；shared 路径越界在 launch 前失败。默认 standalone owned 路径保留原 lazy 行为 |
| 新增 `operators/nosa/attention/workspace.py` | 只提供 NOSA FA3 执行 scratch 的有界分配/切片接口，接收显式 query/head/device/dtype；不实现 attention 数学 |
| `operators/nosa/attention/device_only/{api,_cuda,_fa3}.py` | 透传可选 workspace；hbm/dense serving 使用其 fallback/pages/members/counts，默认调用分支保持兼容 |
| `models/nosa/attention.py` | resident adapter 可显式接收上述 scratch provider；由 serving 在 lease 内注入，reference/default model 路径不依赖它 |
| `models/deepseek_v32/serving_backend.py`（批次 B） | dense shared plan、双缓冲借用、session 分账与关闭；保持 ECHO/serial_sparse 已冻结行为 |
| `serving/run_multi_user.py` | C/A limits 同时传入；外层 finally 关闭 backend。DeepSeek CLI 的参数兼容由 ECHO owner 版本统一，避免恢复旧 per-session slots 含义 |
| `executor/serving_backend.py`、`serving/persistent.py` | 由公共文件 owner 增补单一准入 owner 的 bind/unbind；第二个 live runner 在任何分配前拒绝，构造失败释放本次绑定，close 不销毁 backend shared buffers |
| `experiments/gr_serving/src/{measure,profile,audit,report}.py` | 生命周期、limits、分账/版本元数据和报告校验；报告器按实际字段需求修改，不复制模型逻辑 |
| 对应模块/tests、`tests/integration/test_gr_persistent.py` | 迁移直接调用方式，增加下节的实际资源/状态/异步验收；集成预算改成 shared 加 session |
| 模块 README、AGENTS.md、实验 README | 实施验收后更新所有权与已验证边界；保留尚无 finite NOSA pool 的说明 |

不新增 `__init__.py`、根包或依赖。优先复用 ECHO 已有 `SharedCachePlan` / PrefixPool；
只有 P0 发现接口缺口时才由公共文件 owner 做最小扩展，不另建通用调度框架。

### 4.1 NOSA sparse 的 owned/borrowed 契约

`NosaOffloadCache(..., execution_resources=None)` 保持直接模型调用的 owned 默认模式。
serving 注入 resources 后，仅在本 session 有效 lease 内返回共享 workspace；校验
device、dtype、H/D、容量以及 overlap/fetch 参数。一个 backend 固定一个 scheme，
serial/overlap 使用相同布局与预留公式，各自独立运行状态。

共享 C 可以大于当前 session 的 c；host backing 保持 c，算子仍只访问有效 prefix。
当前 prepare 可能清零完整 C 的 metadata，此成本照常计时，不能按 c 扣除未实际消除的
开销。跨用户和大小交替必须清除旧 ready/queue 状态，不依赖残留内容。

### 4.2 Dense attention scratch

现有 `_fa3.py` 会临时分配 fallback/pages/members/counts；只迁移双 staging 会漏掉
这部分 cache scratch。本批改为显式可选 workspace 注入，dense 与 hbm 对照都使用。
为每次实际 q/head 提供形状正确的视图，不能直接把 Q 上限形状传给要求实际 shape 的 FFI。
覆盖 `ceil(q/8)*H == 256` 时的双份 counts，以及 q/head 的边界变化。
counts 容量固定预留 `2*ceil(Q/8)*H`，再按实际 batch 数切片：只按最大 Q 的当前
分支预留会漏掉“最大 batch 为 258，但较小 q 的 batch 为 256 需要 512 项”的情况。
专项覆盖 `H=2、Q>1024、实际 q=1024`。

共享 GPU 路径先限定已验收的 SM90/BF16/D128/GQA16 native 配置。显式关闭 native、
不匹配布局或 workspace 不支持的后端，在执行前清楚失败，不能静默转到未计费的分配
路径。native 判定读取 `native_enabled()` / `CXLDSAGR_SM90_BACKEND`；当前 adapter
的 `backend="triton"` 名称也可经 API 路由到 native，不能仅凭该字符串拒绝。
CPU 在获取 CUDA workspace/provider 前转 reference，新 workspace 模块顶层不导入
`_fa3`、`tvm_ffi` 或 Triton。未注入 workspace 的既有 API 保留
原本支持范围。当前 numerical repair 和调度顺序均包含在执行与计时内。

## 5. 生命周期和失败处理

1. runner 先绑定唯一准入 owner，再初始化资源与 session pool；构造失败清理本次
   绑定/分配，不破坏已有有效 plan。第二个 live runner 即使尚无请求也必须拒绝。
   不同 backend 不能接管 foreign session。直接 profile/test 的资源使用范围须显式
   声明，不能绕过 runner 后仍声称已完成多用户硬预算准入。
2. `prefill` 在所有 token chunks 外取得一个执行 lease，各 chunk 仍按现有方式提交；
   `extend` 的 lease 覆盖 begin_step、所有层、输出、commit 或 abort。
   `model.main_attention` 的临时替换也在此范围内，finally 恢复。
3. NOSA commit 仍可能补齐派生记录，因此不能在最后一个 attention 返回时提前归还。
   本批保留现有保守同步，不同时做 stream 同步优化。绕过 backend 直接执行 borrowed
   cache 时，如果没有 lease，应在 begin_step 拒绝。
4. dense slot 身份包括 session、generation、layer。写入前等待旧 consumer；消费前
   等待本次 copy ready；新 lease 重置 layer ownership。异常时下一层 copy 可能已
   入队但 caller 尚未 wait ready，必须 join 所有已提交 copy/compute 再归还。
   CUDA 错误使完成状态不可确认时，将资源标为不可再用，不能清 owner 后继续运行。
5. truncate/reset/release 在无冲突 lease 且相关操作完成后修改 session；释放 session
   清除借用 aliases，不释放共享 buffers。计数或 trace 视图不能被 session 长期保存
   后再被其他用户覆盖；诊断在该次执行结束、下一次借用前导出。
6. `runner.close` 关闭 sessions；最外层 finally 调用 `backend.close`。
   backend.close 幂等，但 live session/lease 未结束时拒绝释放；成功关闭不销毁模型
   权重，之后可重新 plan。measure 切 scheme 前关闭旧 NOSA backend，不能只覆盖变量；
   warmup 与同 scheme 正式测量之间保留同一共享 plan，只释放 warmup sessions。

## 6. 执行阶段与依赖

| 阶段 | 工作与交付 | 完成条件 |
|---|---|---|
| P0 接口及账本冻结 | 记录 ECHO 接口身份、C/A/Q 和全部 allocation；检查直接调用入口 | owner 边界明确，资源公式覆盖 staging/metadata/scratch/pending，旧结果依赖清单齐全 |
| P1 资源基础 | 共享 resource、dense buffer 生命周期、纯估算与限额；就近 CPU 测试 | 一份 shared 可支持两个独立 session；相同 plan 复用；非法请求无状态损失 |
| P2 NOSA sparse | owned/borrowed 注入、有界 workspace、serial/overlap 接入 | A→B→A、不同 c/q、跨 stream、abort/release 正确；无共享字节重复计费 |
| P3 NOSA dense/hbm | 双缓冲迁移，device-only scratch 注入与 hbm 对照统一 | dense 完整历史流量不变，slot fence 正确，实际 attention scratch 被预留 |
| P4 入口与计量 | CLI、warmup、measure/profile、direct tests 全部适配，报告版本字段 | 正式和诊断入口都使用明确 plan；无旧 backend 遗留；source gate/预算审计不放松 |
| P5 NOSA 正确性验收 | CPU/GPU、完整 checkpoint、分配与异步生命周期检查 | 独立 prefix 的全部 candidate hidden 一致；共享/session 实际分配符合预留 |
| P6 DeepSeek dense（B） | 在冻结接口上复用双缓冲基础，独立改 session 账本/别名/关闭 | 不影响 ECHO policy；全部 candidate hidden 与末 token head、共享 staging 及分配预算验收 |

P1 的通用双缓冲与 NOSA resource 可分文件并行；P2/P3 修改 `serving.py` 时由一人
顺序集成。P6 不要求在 NOSA 发布前完成，但只在 ECHO 公共接口及该文件改动交接后进入。
有限 page cache 不作为这些阶段的隐含完成条件。

## 7. 验收矩阵

| 检查 | 必须覆盖 |
|---|---|
| 预算 | 纯 plan 无 cache 分配；真实 shared storage 只计一次；恰容纳一个/两个用户；比最低可行容量少 1 B；host DRAM 独立不足；allocation 峰值不只检查 forward 后 stats |
| 上限 | C/A/Q 精确边界、q 大于 prefill chunk、非法 q 在写入前拒绝；共享模式不增长；Q 预留后改变 plan 被拒绝 |
| 数据隔离 | A→B→A 且历史内容不同；同历史长度不同 token 身份；大小 session 交替；64-token 尾块；不同 suffix 后 truncate，CIS/派生长度一致 |
| 资源生命周期 | release A 不破坏 B；runner 关闭后 plan 可复用；两个空 runner 也不能同时绑定；构造失败无残留 owner；backend close 无遗留 aliases；foreign/stale session；禁止并发借用 |
| 异步 | 延迟 copy、非默认 caller stream、跨用户接力、中间层抛错；归还包含未被 caller 等待的预取；返回 hidden 不被下一请求覆盖 |
| 模型正确性 | 完整 NOSA 32 层，各方案独立空 sparse cache 构建 prefix，比较所有 candidate normalized hidden；两个不同用户交错复访；禁止只比较末 token |
| 稀疏语义 | serial 整批唯一并集一次 fetch；overlap 相同 selection/CIS/mask；每个历史向量唯一读取；full-address staging 并未变成有限 slots |
| Dense 语义 | 每层仍搬完整历史，当前 suffix 直接写 stage；流量按模型 K/V/record 大小计算；无 token hit/eviction 快捷路径 |
| 诊断 | trace 在独立 plan 中预留；实际 trace 不超上限；page envelope 为所有非空 stripe min/max；分别报告 envelope 与 stripe-copy 两套 ratio |

现有测试承载：`models/nosa/tests/{test_serving,test_offload_cache,test_offload_model,
test_offload_checkpoint}.py`、`serving/tests/`、`cache/tests/` 和
`tests/integration/test_gr_persistent.py`。新 resource/staging/scratch 的单元测试分别
放就近 tests；不将运行结果放入 tests。集成测试只编排跨模块契约，避免重复数值检查。

修改后的 checkpoint serving 测试必须实际走 shared 模式。64K+1K 验收和短 candidate
64K+128 都保留；短输入 CPU 或默认 8K checkpoint 检查不能代替完整长上下文验收。
若实现不改变数值路径，要求与独立 resident 输出逐位一致；任何差异先定位，不能为了
通过而放宽已有阈值。CPU/FP32 reference 继续使用原有合理容差。

以下为执行阶段命令模板，本次没有运行。GPU 0 仅作示例，执行前选空闲 SM90：

```bash
bash scripts/run_tests.sh cpu

CUDA_VISIBLE_DEVICES=0 \
NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=1024 \
bash scripts/run_tests.sh gpu

CUDA_VISIBLE_DEVICES=0 \
NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
.venv/bin/python -m pytest models/nosa/tests/test_offload_checkpoint.py -s -q -p no:cacheprovider

CUDA_VISIBLE_DEVICES=0 \
NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=128 \
.venv/bin/python -m pytest \
  models/nosa/tests/test_serving.py::test_cuda_serving_checkpoint_independent_prefixes_and_revisits \
  -s -q -p no:cacheprovider
```

显式 GPU 验收缺少硬件/依赖/checkpoint 要失败，不能将 skip 计为通过。
standalone offload 测试若缺默认请求文件，使用有效 65536+1024 请求设置
`NOSA_OFFLOAD_REQUEST`。新增 GPU staging 测试需加入全局 GPU 编排，已收集的数值测试
不重复添加。P6 另显式运行 `DEEPSEEK_SERVING_CHECKPOINT` 对应模型测试；其当前短 prefix
测试不能代替正式 GR trace。

## 8. 原性能补测与替换方案（用户已取消执行）

以下保留原设计和测量要求，供将来明确重启时参考。本轮未启动正式计时，按用户最新
要求不执行本节补测、性能 profile 或报告替换。第 8.1 节的入口适配代码仍属于已完成
实现；其可用性和预算检查由正确性测试验证，不代表获得了性能结果。

### 8.1 必须适配的计量入口

- `measure.py` 的 warmup/正式 runner 均传 C/A；每个 scheme 和异常出口正确关闭
  backend。同模型所有方案使用同输入、预算和输出范围。
- `profile.py` 当前直接 create_session 并取得 `attention_workspace`，须显式
  plan/allocate/lease/release/close；计费拆成 shared/session/trace，不能沿用只看
  session 的统计。先声明 trace 上限，再分配，不再以首次懒分配增长作为验收条件。
- 正式 plan 的 trace 预留为零。独立诊断 plan 将 trace 实际计入自己的预算，另报相对
  正式 plan 的额外字节；保留其无多用户 LRU、不能代表正式容量表现的边界。
- 报告至少保存 policy revision、workspace scope、C/A/Q、stage 数量、shared 与
  session 的预留/实际字节、实际 H2D/D2H、缓存命中与复访 miss、总延迟及分项。
  `cache_pool_scope=backend_workspace` 不表示用户历史共享；host scope 仍是 session。

### 8.2 补测范围

| 实验 | 要求 |
|---|---|
| `experiments/gr_serving` 的 NOSA | 必须以新 run 重测四方案，hbm 作为同次独立对照；NOSA layer-31 profile 另用新 run ID |
| `experiments/gr_serving` 的 DeepSeek | P6 后重测 dense 和同条件对照，与 ECHO 执行者协调合并发布，保持 checkpoint 工作负载替身及 LM-head 边界 |
| `experiments/nosa_offload_overlap` | 若改动公用 run/prepare/sync/统计或默认执行路径，重新做主测、确认和内部 profile；仅增未启用的 bounded/optional API 且默认路径保持原样时，可保留原有效报告并记录判定依据 |
| 其他 NOSA resident/indexer 实验 | 检查本次 device-only 可选 workspace 的默认调用是否受影响；只有确实改变其执行/计量才扩展补测，不按目录名称全部重跑或删除 |
| `deepseek_v32_echo_prefill` | P6 仅改 serving dense 时不直接受影响；若扩到公用 ECHO 模型/cache 路径，重新核定范围 |

先用已定义的 64K+128、16 用户两轮顺序请求作受控完整执行检查，不把它称作规模主实验。
新 run ID 必须唯一，示例命令中的 ID 运行前替换：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/gr_serving/scripts/run.sh NEW_NOSA_SHARED_RUN_ID \
  --models nosa --sampling sequential --users 16 --requests 32 \
  --history-tokens 65536 --candidate-tokens 128 \
  --chunk-size 1024 --hbm-budget-gib 4 --dram-budget-gib 64

.venv/bin/python -m experiments.gr_serving.src.audit \
  experiments/gr_serving/output/data/NEW_NOSA_SHARED_RUN_ID \
  --expected-run-id NEW_NOSA_SHARED_RUN_ID \
  --json experiments/gr_serving/output/data/NEW_NOSA_SHARED_RUN_ID/audit.json

CUDA_VISIBLE_DEVICES=0 bash experiments/gr_serving/scripts/profile.sh NEW_NOSA_SHARED_PROFILE_ID \
  --latency-data experiments/gr_serving/output/data/NEW_NOSA_SHARED_RUN_ID --num-users 16
```

容量范围遵循研究者后续选择的[完整 loop 规则](../research-supervisor/loop_motivation.md)，
此前热度 IID 不再是本轮要求。依据实际 shared/session 容量选择共同 U 点，完整执行
相同轮数，覆盖容量以内、HBM miss/offload hit（若存在）及双方均 miss；16 用户两轮
仍保留为本计划受控验收点。只报名义人数或全命中的检查不完成容量验收。每组记录硬件、
依赖、形状、精度、预热与完整请求数，按轮次、首访、复访命中和复访重建分别报告。

NOSA overlap 的延迟门槛仍是同条件完整 query batch 的串行并集 fetch 对照；不能只看
内部区间相交。90% 声明要求每个 profiled sample 的两种 ratio 都达标，低于门槛的合理
测量应如实报告。无收益不是删除合理对照或修改任务定义的理由。

### 8.3 发布顺序

1. 实施后立即在受影响实验 README 标注“共享 cache 改动后未运行”，保留旧 run ID、
   实现版本和测量边界。确认旧预算有遗漏时，注明不能用于硬预算结论的范围。
2. 性能测量前冻结实际源码；GR source snapshot 覆盖两模型及公共模块，另一 agent
   同时改 ECHO 也会导致 NOSA-only run 身份校验失败。使用冻结集成版本或隔离工作树，
   不缩小/放松身份校验以迁就并行修改。
3. 新 run 保存实际 Git/diff/source hash、依赖/native key、输入和参数，完成数值、
   allocation/预算、唯一读取及相应内部 overlap 审计。所有 initialization、planning、
   prepare/repair 和 launch gaps 计入对应执行窗口。
4. 新结果可核验发布后，同次更新 README 数字/结论和 report 素材，再删除已替换的
   旧产物。旧 `h4k/h16k/h64k` report/output 混有两模型，不整目录提前清空。
   不过滤改写旧 raw run 的 measurements 或 source manifest；若原始混合 run 仍承载
   尚未替换的另一模型结果，暂保留该容器并列入 ECHO/NOSA 联合清理清单，替换完成即删。

## 9. 最终交付条件

- A/B 各自的代码、模块说明、CLI、tests 与预算统计一致；已完成批次才更新状态。
- shared/owned 两种模式均保持正确，正式 serving 不隐式退回每用户分配。
- 工程记录区分已验证的分配上界、静态准入推导与未运行的完整容量/延迟轨迹。
- 工程 checkpoint 记录源码身份、验收命令、结果与尚未覆盖路径；不生成本轮性能报告。
- 没有将本计划或 CPU/算子回归写作完整模型性能验证，没有将 NOSA staging 写作有限
  HBM cache。后续 finite page cache 另立任务，重新设计物理寻址、驻留和消费者协议。
