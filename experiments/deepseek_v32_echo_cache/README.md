# DeepSeek V3.2 ECHO 容量分析

2026-10-04 的 motivation 优化正在调整共享 pool 的 metadata 存储。下文静态规划
仍对应所记录的源码与容量公式，新实现尚未重新规划；旧数据保留至更新验收发布。
详见[优化记录](../../docs/agents/system/deepseek_motivation_optimization_plan.md)。

本实验按指定的 P（每层 HBM 历史主 KV 槽数）和 NH（所有用户共享的 DRAM 历史
主 KV token 容量），分析每 token 的内存成本与联合容量边界。本次发布为静态规划，
没有新实现的完整多用户容量实测，也不提供延迟或吞吐结论。

GR `echo/serial_sparse` 只为固定 history 保留 session。候选一次整批执行，主 KV
留在 GPU 临时尾部，不申请 host pages，也不写回 DRAM；当前层使用一份 backend
共享的合并 indexer workspace。成功后丢弃候选，保留 history；失败直接报错终止并
释放 session，不恢复或重试。直接 `backend.extend`、非 GR 模型与 `hbm/dense_prefetch`
对照仍保留各自原有接口，本文的 H-only 容量结论仅适用于该 GR 路径。

## 工作负载与分析范围

- 真实 checkpoint 前三层复制为十个独立 dense block，source 顺序为
  `[0,1,2,0,1,2,0,1,2,0]`。每个副本使用对应 source 的 hidden/residual 输入和
  独立权重、KV、indexer；包含 embedding、final norm、全部 candidate hidden 与
  末 token LM head。这是 checkpoint 工作负载替身，不是训练得到的十层模型或完整 DeepSeek。
- `L=10`、history `H=65,536`、candidate `A=128`、history prefill chunk `C=1024`。
  保留容量为 H，执行范围为 `N=H+A=65,664`，共享 query 容量 `Q=max(C,A)=1024`。
  在已规划的 A 和 H+A 上限内改变候选长度，不因候选容量变化重建相同 history。
- 主 KV 为 BF16 512 latent + 64 RoPE；indexer 为 128 维 FP8 K 和 FP32 scale。
  H 按 64-token page 对齐，当前用户容量 `U=floor(NH/65,536)`。
- 预算输入来自 NVIDIA H200（SM90 / Hopper，132 SM），总 HBM 为 139.811584473 GiB；
  Python 3.12.13、PyTorch 2.12.1+cu130、CUDA 13.0，checkpoint 位于 `/preset-models`。
  源码身份及规划输入见报告 provenance，依赖由仓库 `pyproject.toml` 与 `uv.lock` 固定。
- 完整运行入口使用共享 GR 生成器、seed=42、`content_is_synthetic=True`，依次访问
  全部用户两轮，比较首次构建与复访；合成输入不提供真实 GR 质量或场景代表性的证据。

新路径的短 GPU 正确性检查已通过：从独立空 cache 构建 2,304-token history，执行
16 / 23-token candidate；全部 hidden 和末 token logits 与 HBM 逐位一致，candidate
D2H 为零，history KV、index K 与 scales 不变。这是工程正确性证据，不是本节
64K history、完整用户集合或峰值显存的实验结果。

## 每个 token 的 HBM 与 DRAM 占用

B 表示字节，MiB=`2^20` B，GiB=`2^30` B。历史 token、HBM 槽位和执行 query 的
分母不同：保留更多用户增加 NH，增加主 KV 驻留空间增加 P，增大执行 batch 增加 Q。

| 项目 | 单层字节数 | 十层合计或完整公式 | 位置与计费分母 |
|---|---:|---:|---|
| BF16 主 KV：`(512+64)×2` | 1,152 | 11,520 | 每个 NH token 的 DRAM 逻辑数据；每增加一个 P 槽的 HBM 数据 |
| FP8 indexer K | 128 | 1,280 | 每个保留的 history token，常驻 HBM |
| FP32 indexer scale | 4 | 40 | 每个保留的 history token，常驻 HBM |
| host→device 映射 | 4 | 40 | 每个 NH token，常驻 HBM |
| device→host、priority、free bitmap | 8+8+1 | 170 | 每个 P 槽，另计每层 sentinel |
| session GPU page table | 不逐层复制 | `4/64=0.0625` | 每个 history token；每 session 为 4,096 B |
| session hints 与 counters | 每 session 216 | 每 session 2,160 | HBM 预留：三份 64 B hints 加 24 B counters，其中一份 hint 用于事务备份 |
| 共享持久 scratch | 不逐层复制 | `20P+36` | 整个 backend 一份 |

一个 session 的私有 HBM 预留为：

```text
10 × 65,536 × 132 + 2,160 + 4,096 = 86,513,776 B
```

即 **82.505966 MiB/session、1,320.095459 B/history token**。这里包含事务备份
额度，不等于每时刻精确驻留的 tensor 字节数；共享 pool、全局映射、执行空间和
allocator 余量另计。主 KV 命中 HBM 后仍保留 DRAM 副本，增大 P 不会减少 DRAM 容量。

每层主 KV storage 为 `[P+1+A,576]`，前 P+1 行是历史槽和 sentinel，尾部为候选。
候选没有 host ID、页表项或淘汰元数据。共享候选 KV 为 `10×128×1152=1,474,560 B`；
单层合并 indexer K/scales 为 `(65,536+128)×132=8,667,648 B`，合计 **10,142,208 B**。
合并 indexer 只有一份，不乘以 L；两项均跨用户复用，不乘以 U。

上下文按 128 对齐后 `T=65,664`，top-k `K=2048`。
[执行空间预留](../../models/deepseek_v32/cache_resources.py) 为：

```text
E(Q) = 16QT + 64Q×min(K,N) + 32T + 2Q×576×2
     = 1,184,000Q + 2,101,248 B
E(1024) = 1,214,517,248 B = 1.131107 GiB
```

其中 indexer 的 logits、mask、top-k、remap、union/sort 等预留 1,212,157,952 B；
两份在途 append source 预留 2,359,296 B，覆盖 history prefill 写回，候选 D2H 仍为零。
另有 `64(P+1)+64NH` B 的 pool metadata 执行预留。这些空间在执行期间占用 HBM，
必须与历史 cache 同时装入显存；串行层和用户共享额度，不再乘以 L 或 U。
Q 表示执行 query 容量，增大 Q 会增大 workspace，不直接增加 P 或 NH。当前固定
P/NH 模式没有独立 W 参数，也不按 workspace 字节预留淘汰 session。

[容量 planner](../../models/deepseek_v32/capacity.py) 的基础 HBM 账本可化为：

```text
HBM_base = E(1024) + 11,774P + 104NH + 11,870
           + 86,513,776U + 10,142,208 B
U = floor(NH / 65,536)
```

11,774 B/P 包括主 KV、槽元数据、共享 scratch 和 metadata 执行预留；104 B/NH
包括 40 B 常驻映射和 64 B 执行预留。NH 恰好装满完整用户时，摊销为：

```text
HBM_base / NH = 1,424.095459 + 11,774×P/NH
                + (E(1024)+10,154,078)/NH B/history token
```

例如 P=32,768、NH=1,048,576 时，基础预留为 **2.890596 GiB，即 2,959.971 B/history
token**；若 P=NH，线性项合计 13,198.095 B/history token，共享常数项另计。
这些数均不含模型、普通 activation 和 allocator allowance，不能当作进程峰值。

CPU 主 KV 由 [pinned allocator](../../cache/host_allocation.py) 按二次幂档位分配，
每层一个 arena；DRAM 账本为：

```text
DRAM = 10×next_power_of_two(1,152NH) + NH/16 + 4,096U + 40 B
```

依次为 pinned storage、共享 free-page stack、session CPU 页表和 CPU 执行 scratch。
16-user 的 NH=1,048,576，pinned 主 KV 为 20 GiB，摊销为 **20,480 B/NH token**，
高于逻辑主 KV 的 11,520 B/token。16 人来自 `1,048,576/65,536`，不是实现固定上限。

## 预算输入与静态结果

沿用同机总 HBM 和模型加载占用作为规划输入：

```text
HBM 额度 = floor(150,121,545,728 × 0.9) − 10,032,775,168
         = 125,076,615,987 B = 116.486676 GiB
DRAM 额度 = 512 GiB
```

0.9 作用于总 HBM，模型加载占用 9.34375 GiB 单独扣除。planner 还计入持久 storage
的 CUDA allocator 舍入与尾部，以及 64 MiB scratch / fragmentation allowance。
14.5 GiB 来自旧候选持久化路径中观测到的 allocator 缓存与设备占用差额。下表将
额外扣除该值作为一项规划假设；实现没有对应的 14.5 GiB 预分配，也未证明新路径
需要这些空间。该假设不作为默认最大 P 的依据。若显式采用它，扣除后，
cache、workspace 和原有 allocator allowance 可用 101.986676 GiB。
PyTorch allocated 统计活跃分配，reserved 还包含 allocator 保留的空闲缓存段，
二者不能相加；设备已用量还可能包含非 PyTorch 分配。新路径未测量这些峰值，
静态账本不能证明实际设备占用始终低于额度。

本次发布 `20261003_echo_gpu_candidate_capacity_report_01` 包含 **0 项完整运行、5 项静态规划**。
详细分项见[容量结果](report/capacity/results.md)、[summary.json](report/capacity/summary.json)
和 [provenance](report/capacity/report_provenance.json)。

| Plan ID | 假设额外扣减 GiB | 固定条件 | 所选 P | 所选 NH | U | 有效 P `min(P,NH)` |
|---|---:|---|---:|---:|---:|---:|
| `20261003_echo_gpu_candidate_nh_plan_01` | 0 | P=32,768 | 32,768 | 29,818,880 | 455 | 32,768 |
| `20261003_echo_gpu_candidate_p_plan_base_01` | 0 | NH=1,048,576 | 10,367,498 | 1,048,576 | 16 | 1,048,576 |
| `20261003_echo_gpu_candidate_p_plan_nhmax_01` | 0 | NH=29,818,880 | 6,496,179 | 29,818,880 | 455 | 6,496,179 |
| `20261003_echo_gpu_candidate_p_plan_base_headroom_01` | 14.5 | NH=1,048,576 | 9,045,156 | 1,048,576 | 16 | 1,048,576 |
| `20261003_echo_gpu_candidate_p_plan_nhmax_headroom_01` | 14.5 | NH=29,818,880 | 5,173,835 | 29,818,880 | 455 | 5,173,835 |

P 与 NH 共同消耗 HBM，不能同时取各自独立最大值。P>NH 只增加分配，不增加独立
历史 token 的可驻留数量。固定 P=32,768 时，NH 上界由 pinned DRAM 档位决定；
假设额外扣除 14.5 GiB 也不改变该上界：

| 用户数 | NH | 每层 pinned 主 KV | 十层 pinned 主 KV | 总 DRAM B |
|---|---:|---:|---:|---:|
| 16 | 1,048,576 | 2 GiB | 20 GiB | 21,474,967,592 |
| 455 | 29,818,880 | 32 GiB | 320 GiB | 343,601,111,080 |
| 456 | 29,884,416 | 64 GiB | 640 GiB | 687,198,502,952 |

第 456 个用户使每层 arena 跨入下一档，规划 DRAM 超过 512 GiB；这不是一次实测 OOM。
**不额外扣除旧观测差额时，静态账本得到 P=6,496,179、NH=29,818,880（455 users）。**
P=5,173,835 是再假设扣除 14.5 GiB 后的条件结果。这两项都没有完整执行或物理
峰值证明。旧候选持久化路径的运行结果不用于新实现的容量结论。

## 复现入口

从仓库根目录以 CPU 使用上述预算输入规划，以下两项分别固定 P 和 NH：

```bash
.venv/bin/python -m experiments.deepseek_v32_echo_cache.src.capacity_plan \
  --run-id NEW_NH_PLAN_ID --fixed-p 32768 \
  --total-hbm-gib 139.81158447265625 --model-hbm-gib 9.34375 \
  --hbm-fraction 0.9 --dram-budget-gib 512
.venv/bin/python -m experiments.deepseek_v32_echo_cache.src.capacity_plan \
  --run-id NEW_P_PLAN_ID --fixed-nh 29818880 \
  --total-hbm-gib 139.81158447265625 --model-hbm-gib 9.34375 \
  --hbm-fraction 0.9 --dram-budget-gib 512
```

省略 `--total-hbm-gib` 和 `--model-hbm-gib` 会加载模型重新测量，本文五项计划均使用
给定输入。`src.capacity_report --plans <plan目录>... --run-id <新ID> --output-dir <新目录>`
可重建选定计划的报告，静态发布省略 `--runs`。源计划和源码快照在各自
`output/data/<plan_id>/`，报告选用的数据保存在 `report/capacity/`。

完整请求入口为 `bash experiments/deepseek_v32_echo_cache/scripts/run.sh --help`，
通过 `--sparse-pool-tokens` 与 `--host-arena-tokens` 指定 P/NH；16-user 基础配置应
使用 NH=1,048,576。该入口调用 `src.capacity_probe`、
`models/deepseek_v32/serving_backend.py`、`serving/persistent.py` 和
`cache/sparse_token_pool.py`；请求构造及来源记录复用 `experiments.gr_serving.src`
工具。运行结果在 `output/data/<run_id>/`，stdout/stderr 分别在 `output/log/<run_id>/`。
独立空 cache 数值参考由 `src.capacity_reference` 生成，再通过
`src.capacity_probe --audit-existing ... --reference-dir ...` 核对全部输出。
本轮未执行完整容量轨迹；这些入口的计时含首次 JIT，不用于性能结论。
