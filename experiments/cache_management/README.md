# NOSA 与 DeepSeek cache 管理

DeepSeek Q1 indexer、attention 与官方预取 staging 的执行预留已纳入静态规划。重算后，
五项完整容量计划及下一不可行配置均与原结果一致。原有 dense DMA 请求内存观测和
NOSA 结果保留各自来源；本次只补充静态审计，未重新测量请求内存或运行容量填满轨迹。

本实验为 motivation 提供容量依据，说明两模型如何准入用户、保留 history、执行
candidate，以及怎样核对 HBM / CPU DRAM 占用。公共框架统一容量计划和生命周期；
模型分别声明 KV 布局、派生状态与搬运方式。静态规划、完整请求中的内存观测和物理
容量边界分别报告。

## 公共接口与模型差异

`ResourcePlan` 声明共享分配，`SessionPlan` 同时用于用户准入和实际 session 创建。
分配项包含名称、owner、dtype、shape、设备、生命周期和计费；别名不重复收费，
互斥阶段按峰值计费，同时存活的分配相加。模型控制原有事务和 CUDA 完成边界，
`TokenRuntime` 不重复 commit。固定 history 请求的 candidate 成功后保留 history，
普通 append 持久追加。
执行失败直接传播；必要清理也失败时保留全部异常。无法确认异步完成时保留资源，
拒绝复用，不自动恢复、重试或切换后端。

| 项目 | NOSA | DeepSeek V3.2 |
| --- | --- | --- |
| 本文工作负载 | 完整 32 层 checkpoint，全部 candidate hidden，无 LM head | 前三层独立复制为 C10 source-input replay，全部 candidate hidden 与末 token LM head |
| 逻辑主 KV | BF16 K/V，2 KV heads、D128；1,024 B/token/层 | BF16 512 latent + 64 RoPE；1,152 B/token/层 |
| 稀疏选择 | query/KV-head 的 64-token block IDs 与 validity，保留 CIS 语义 | token top-k 与 causal padding，保留 indexer/prefetch 融合 |
| 固定 P | offload 逐层历史槽直接映射；当前要求 H≤P | offload 逐层有限历史 pool；精确工作集超过 P 时拆分 query 消费 |
| 固定 NH | host history 页配额，随 session 懒分配 pinned backing | 一次分配全局 pinned arena，另计各 session 的 indexer 与页表 |
| 固定模式 candidate | 主 K/V 留在 GPU 临时尾部；CIS/派生尾部按 session 计费 | 主 KV 留在 GPU 临时尾部；当前层借用一份共享合并 indexer workspace |
| HBM-only 准入 | 独立的 P history-token 配额，按整个用户 session LRU | 独立的 P history-token 配额，按整个用户 session LRU |

通用 budget 模式按 HBM / DRAM 字节硬预算做 session LRU；固定 P/NH 模式按 token
或页配额准入。相同 P/NH 不表示相同物理分配，也不构成等字节预算的两模型比较。
PyTorch allocated、reserved 和设备已用量分别保留。观测差额、规划时额外扣减的
额度与实现预分配的 storage 各有来源，不能混为同一项开销。

## NOSA 固定配额的容量含义

离线计划 `cache_nosa_fixed_20261005_02` 使用 H=65,536、A=128、C=1024、P=65,536、
NH=16,777,216，并分析 16 个用户。执行上下文显式设为 H+A=65,664，原 checkpoint
声明的 32,768 上下文单独保留；本分析不提供长上下文质量验证。

每用户的 32 层主 K/V 为 2 GiB，16 个历史的逻辑主 K/V 为 32 GiB。固定路径的
host payload 按 H 分配，session 准入预留按 H+A 的 pinned 档位保守计费，每用户为
4 GiB；这不表示 host allocator 实际占用 4 GiB。以下为生产分配声明给出的上界，
尚未加入 graph storage：

| 方案 | history 用户配额 | 共享 HBM 预留 GiB | 每 session HBM 预留 GiB | 每 session host payload GiB | 每 session host 预留 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| HBM-only | 1 | 0.000980377 | 2.120296478 | 0 | 0 |
| Dense prefetch | 256 | 2.007329464 | 0.167774677 | 2 | 4 |
| Sync sparse | 256 | 2.007358074 | 0.167774677 | 2 | 4 |
| Async sparse | 256 | 2.007358074 | 0.167774677 | 2 | 4 |

两个 sparse offload 方案保留 16 个用户时，HBM 预留上界约为 4.69175 GiB；填满
256 用户配额时约为 44.95768 GiB，均不含 graph。完整配额的逻辑主 K/V 为 512 GiB，
host 预留为 1,024 GiB，因此 NH/H=256 不能证明 512 GiB DRAM 能容纳全部用户。
16 个历史只占 NH 配额的 6.25%。P=H 时 HBM-only 只能保留一个历史，也是策略配额
的结果，不是 H200 的物理容量上限。

[离线入口](src/nosa_plan.py) 复用生产代码的具名分配与 allocator 公式，不加载模型
或初始化 CUDA；假设为 SM90、BF16 和原生 allocator。graph static allocation 与
private reservation 从源码、配置均匹配的 motivation 记录读取，已计入共享预留
的项目不再相加。

## DeepSeek 的容量账本

五项静态计划对应十个独立 dense block，source 顺序为 `[0,1,2,0,1,2,0,1,2,0]`。
在对应的执行路径中，每个副本使用各自 source 的 hidden/residual 输入及独立权重、KV、indexer；包含
embedding、final norm、全部 candidate hidden 和末 token LM head。这是 checkpoint
工作负载替身，不是训练得到的十层模型，也不代表完整 61 层 DeepSeek。

这些离线边界针对 ECHO／serial sparse 的 history-only session 与 GPU transient
candidate。固定模式 dense prefetch 另要求 H≤P，不能把 P=32,768 的计划视作该
dense 路径的可执行配置。普通 append 及 budget 对照仍各自保留原有事务语义。

H=65,536、A=128、C=1024，session 保留容量为 H，执行范围为 N=H+A=65,664，
共享 query 容量 Q=max(C,A)=1024。主 KV 为 BF16 512 latent + 64 RoPE；indexer
为 128 维 FP8 K 与 FP32 scale。H 按 64-token page 对齐，U=floor(NH/65,536)。
在规划的 A 与 H+A 上限内改变 candidate 长度，不因候选容量变化重建相同 history。

| 项目 | 十层合计或完整公式 | 计费分母 |
| --- | ---: | --- |
| BF16 主 KV | 11,520 B | 每个 NH token 的 DRAM 逻辑数据；每个 P 槽的 HBM 数据 |
| FP8 indexer K 与 FP32 scale | 1,320 B | 每个保留的 history token，常驻 HBM |
| host→device 映射 | 40 B | 每个 NH token，常驻 HBM |
| device→host、priority、free bitmap | 170 B | 每个 P 槽，另计 sentinel |
| append-order metadata | 80 B | 每个 P 槽 |
| session GPU page table | 4,096 B | 每 session |
| session hints、prefetch/native counters | 2,560 B | 每 session |
| 共享持久 scratch | 20P+36 B | 整个 backend 一份 |
| resident selection bitmap | 4×ceil((P+129)/32) B | 整个 backend 一份 |
| 各层 clock 与共享 selection count | 84 B | 整个 backend |

每个 session 的私有 HBM 预留为 `10×65,536×132+2,560+4,096=86,514,176 B`。
主 KV 命中 HBM 后仍保留 DRAM 副本，增加 P 不会减少 DRAM 容量。候选 KV 尾部为
`10×128×1,152=1,474,560 B`；单层合并 indexer 为 `(65,536+128)×132=8,667,648 B`，
两项合计 10,142,208 B，均跨用户复用。合并 indexer 只有一份，不乘层数。

在本节 N=65,664、top-k=2,048、record 宽度为 576 的条件下，当前
[容量 planner](../../models/deepseek_v32/execution/capacity.py) 的基础账本为：

```text
E(Q) = max(1,181,696Q + 2,101,248, 12,298,544, 4,816,896)
       + 2,304Q B
E(1024) = 1,214,517,248 B
bitmap = 4 × ceil((P+129)/32)
HBM_base = E(1024) + 11,854P + 104NH + 11,874
           + 86,514,176U + 10,142,208 + bitmap B
U = floor(NH / 65,536)
DRAM = 10×next_power_of_two(1,152NH) + NH/16 + 4,096U + 40 B
```

E 的 max 三项分别覆盖最大 query 的 indexer/selection、可达 Q1 尾块的 paged
indexer（含官方预取 staging）和 Q1 attention 活跃空间；互斥阶段取较大峰值。
官方暂存包括逐 token host ID 表、64 条 BF16 record 和 64 个 stage host ID，
共 `4N + 64×(1,152+4)` B；在 N=65,664 时为 336,640 B，使 Q1 indexer
预留从 11,961,904 B 增至 12,298,544 B。`2,304Q` 计入两份在途
append source，覆盖 history prefill 写回，candidate D2H 仍为零。
Q≥9 时，该式与原来的 `1,184,000Q + 2,101,248 B` 相同；较小 Q 须保留上述 max。
HBM_base 另含 `64(P+1)+64NH` 的 metadata 执行预留；它不含模型、普通 activation
或 allocator allowance。当前账本包括
append-order、native counter 与 bitmap。每层新增的 int64 recall counter 使十层
session 的逻辑计数 slab 从 560 B 增至 640 B，每 session 增加 80 B；两者落在
同一 allocator 档位，逻辑预留的增加与 allowance 的减少抵消，五项静态边界和
舍入后总额均未改变。此前 DMA 改动的源码与公式核验保留在
[DMA 源码影响审计](report/unified/deepseek_static_impact.json)和
[原发布审计](report/unified/audit.json)，其中的源码身份仍对应当时版本。

新审计 `cache_deepseek_q1_impact_20261008_02` 复算了包括官方预取 staging 在内的 Q1 预留。
五项计划均使用 Q=1024，indexer 预留为 1,212,157,952 B，已覆盖 Q1 indexer 和 attention 的活跃峰值，
attention 增量为 0；两份 append source 为 2,359,296 B，因此 E(1024) 不变。
[Q1 静态影响记录](report/unified/deepseek_q1_static_impact.json)绑定当前源码、原计划及
checkpoint 配置哈希，逐项比较五个完整计划，包括所选配置、有效 P 配置及下一不可行
配置；共 1,222 个标量全部相等。[Q1 发布清单](report/unified/deepseek_q1_static_publication.json)记录
报告与复现文件的哈希。脚本、当前源码和原计划副本位于
`output/data/cache_deepseek_q1_impact_20261008_02/`。此次复算不更新后文的请求内存
观测，也不验收 Graph 私有池或物理容量。

上述静态边界针对 ECHO／serial sparse。完整请求中的 dense DMA ticket 借用已有
连续 host/HBM storage，不再分配私有 GPU ID/count，预留为 0。这比前一实现减少
10,491,392 B 的执行预留，不等于实际 allocated 同量下降。graph static allocation
与实际 private reservation 仍在请求账本中单列并计费。

## DeepSeek 静态边界

沿用历史测量的总 HBM 与模型加载占用作为显式规划输入：

```text
HBM 额度 = floor(150,121,545,728 × 0.9) − 10,032,775,168
         = 125,076,615,987 B = 116.486676 GiB
DRAM 额度 = 512 GiB
```

0.9 作用于总 HBM，历史模型加载占用 9.34375 GiB 单独扣除；这不是对本次运行模型
占用的重新测量。planner 还计入 CUDA allocator 舍入、尾部和 64 MiB scratch /
fragmentation allowance。14.5 GiB 是显式选择的额外扣减假设，来源于旧候选持久化
路径的占用差额；实现没有对应的预分配，也未证明当前路径需要这些空间。
链接的统一结果表将该假设计入 HBM 规划总额；14.5 GiB 单列为 extra headroom，
不能把包含它的总额解释为 cache 实际分配或实现必需的 storage。

| Plan ID | 额外扣减 GiB | P | NH | U | 有效 P `min(P,NH)` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cache_deepseek_nh_20261006_01` | 0 | 32,768 | 29,818,880 | 455 | 32,768 |
| `cache_deepseek_p_base_20261006_01` | 0 | 10,296,447 | 1,048,576 | 16 | 1,048,576 |
| `cache_deepseek_p_base_headroom_20261006_01` | 14.5 | 8,983,043 | 1,048,576 | 16 | 1,048,576 |
| `cache_deepseek_p_nhmax_20261006_01` | 0 | 6,451,364 | 29,818,880 | 455 | 6,451,364 |
| `cache_deepseek_p_nhmax_headroom_20261006_01` | 14.5 | 5,137,960 | 29,818,880 | 455 | 5,137,960 |

P 与 NH 共同消耗 HBM，不能同时取各自独立最大值。P>NH 只增加分配，不增加独立
历史 token 的可驻留数量。固定 P=32,768 时，NH 上界由 pinned DRAM 档位决定：
455 个用户的十层 pinned arena 为 320 GiB，第 456 个用户使其跨入 640 GiB 档位，
超过 512 GiB 规划预算。这是静态边界，没有填满配额的请求实测或 OOM 证据。

## 完整请求内存观测

统一报告 `cache_unified_dma_20261006_01` 汇入两个模型各一条正式 trace：
`refactor_final_nosa_bench_20261005_01` 与
`deepseek_dma_c10_bench_20261006_01`。每条 trace 含四方案各 32 个请求，
使用 H=65,536、A=128、C=1024、P=65,536、NH=16,777,216、16 用户两轮访问。
两条 trace 各自在 GPU3、CPU24–31、NUMA0 上计时，使用对应的独立数值验收收据。
DeepSeek 的测量与 observer 均以 0 退出；147 次离散采样的最大间隔为 30.353405 秒，
未发现所选 GPU 上的未归属进程或其他 GPU 进程。这些记录不证明全机独占、连续
隔离或 CPU 独占。完整原始观测、归属及归档核验见
[observer 审计](report/unified/deepseek_observer_audit.json)和
[运行归属记录](report/unified/deepseek_observer_reconciliation.json)；NOSA 仍沿用原
trace 的观测证据。性能比较见各模型 motivation 报告；本节只汇总这两条正式 trace
的内存。表中来源、逐请求最大值和 NOSA 数据保留情况均已独立复核。

下表单位为 GiB。allocated／reserved 为正式请求轨迹中的 PyTorch 峰值；设备已用量
取每个请求结束后的边界采样最大值，不是连续采样的进程峰值。各项均包含模型与执行
临时空间，不能相加，也不能用 allocated 一项判断物理 HBM 是否超额。

| 模型 | 方案 | 最多保留用户 | Allocated 峰值 | Reserved 峰值 | 设备已用量采样最大值 |
| --- | --- | ---: | ---: | ---: | ---: |
| nosa | hbm | 1 | 18.839923 | 24.078125 | 24.831116 |
| nosa | dense_prefetch | 16 | 19.989842 | 25.330078 | 26.147522 |
| nosa | serial_sparse | 16 | 19.989294 | 25.318359 | 26.137756 |
| nosa | overlap | 16 | 19.989294 | 25.318359 | 26.137756 |
| deepseek | hbm | 1 | 14.517354 | 23.634766 | 24.376038 |
| deepseek | echo | 16 | 16.357248 | 24.296875 | 25.665100 |
| deepseek | serial_sparse | 16 | 16.358774 | 24.296875 | 25.665100 |
| deepseek | dense_prefetch | 16 | 16.358774 | 24.294922 | 25.665100 |

cache 记账与计划预留单独列在下表。HBM 记账包含共享 cache/workspace、session
storage、graph static allocated 与观测到的 graph private reserved；预留还包含所声明
的保守上界。它们不是两笔可相加的实际分配，也不包含模型权重和普通 activation。

| 模型 | 方案 | HBM 记账 GiB | HBM 预留 GiB | DRAM 记账 GiB | DRAM 预留 GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| nosa | hbm | 8.602579 | 8.644013 | 0.000000 | 0.000000 |
| nosa | dense_prefetch | 9.727347 | 11.214461 | 32.000000 | 64.000000 |
| nosa | serial_sparse | 9.727373 | 11.214489 | 32.000000 | 64.000000 |
| nosa | overlap | 9.727373 | 11.214489 | 32.000000 | 64.000000 |
| deepseek | hbm | 6.624072 | 14.585029 | 0.000000 | 0.000000 |
| deepseek | echo | 8.462284 | 17.426820 | 320.001038 | 320.001038 |
| deepseek | serial_sparse | 8.462284 | 17.426820 | 320.001038 | 320.001038 |
| deepseek | dense_prefetch | 8.462284 | 17.426820 | 320.001038 | 320.001038 |

NOSA 的 16 份历史实际主 KV payload 为 32 GiB，准入保守预留为 64 GiB。DeepSeek
在此 NH 下的全局 arena 逻辑容量为 180 GiB，每层 18 GiB pinned 分配进入 32 GiB
档位，十层 backing 为 320 GiB；本次 16 个历史只写入 11.25 GiB 主 KV。相同 NH
因此对应不同物理分配，不能用这两列直接排列模型的内存效率。

模型加载后的 allocated／reserved／设备已用量采样分别为 NOSA
15.246635／15.248047／15.762756 GiB、DeepSeek
9.188932／9.378906／9.983459 GiB。这是加载边界观测，不是逐 tensor 权重清单；
DeepSeek 静态 planner 仍使用前节声明的历史 9.34375 GiB 输入。普通 activation
没有独立测峰，记录为 null，不从进程峰值减去 cache 预留反推。

HBM-only 的复访历史命中均为 0/16，offload 均为 16/16。本次轨迹没有填满 NH，
也没有验收静态最大 P/NH 或 455 用户容量。明细见[统一结果](report/unified/results.md)、
[内存表](report/unified/memory.csv)与[来源和计划](report/unified/summary.json)。
报告使用的源码、生成命令及选定文件 hash 记录在[发布来源](report/unified/publication.json)。

## 复现入口

从仓库根目录执行。NOSA 离线规划使用 production allocation declarations；DeepSeek
两条命令分别固定 P 与 NH：

```bash
.venv/bin/python -m experiments.cache_management.src.nosa_plan \
  --run-id NEW_NOSA_PLAN_ID --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B
.venv/bin/python -m experiments.cache_management.src.capacity_plan \
  --run-id NEW_NH_PLAN_ID --fixed-p 32768 \
  --total-hbm-gib 139.81158447265625 --model-hbm-gib 9.34375 \
  --hbm-fraction 0.9 --dram-budget-gib 512
.venv/bin/python -m experiments.cache_management.src.capacity_plan \
  --run-id NEW_P_PLAN_ID --fixed-nh 29818880 \
  --total-hbm-gib 139.81158447265625 --model-hbm-gib 9.34375 \
  --hbm-fraction 0.9 --dram-budget-gib 512
.venv/bin/python -m experiments.cache_management.src.report --help
```

DeepSeek 省略 `--total-hbm-gib` 和 `--model-hbm-gib` 时会加载模型重新测量，本文五项
计划均使用给定输入。`src.report` 接收 NOSA plan、全部 DeepSeek plans、两个独立
验收后的正式 motivation bench 与可选 receipt 路径，核对来源后生成统一报告。
源计划、源码快照及完整结果保存在各自 `output/data/<run_id>/`，选定的报告数据保存在
`report/`。上述命令用于生成新规划和报告。重做某次运行的完整来源或数值审计仍需
该次原始依赖；清理旧运行后，不能仅凭选定表格和摘要重做这类审计。

完整容量轨迹入口为 `bash experiments/cache_management/scripts/run.sh --help`，
通过 `--sparse-pool-tokens` 与 `--host-arena-tokens` 指定 P/NH。它调用
`src.capacity_probe`、DeepSeek adapter、persistent runner 和共享 token pool；
请求构造复用 `GR.workload`，来源记录复用 `evaluation.provenance`。独立空 cache
参考由 `src.capacity_reference` 生成，再通过
`src.capacity_probe --audit-existing ... --reference-dir ...` 核对输出。
本轮没有运行容量填满轨迹；该入口的计时含首次 JIT，不用于性能结论。
