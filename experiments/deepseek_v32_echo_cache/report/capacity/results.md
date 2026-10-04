# ECHO 固定容量结果（20261003_echo_gpu_candidate_capacity_report_01）

P 是每层 HBM history token pool 容量，NH 是全局 host arena 的逻辑 token 容量。gpu_transient 按 H 保留用户历史，U=floor(NH/padded(H))；候选整批执行，各层 KV 尾部保留 A 行，indexer 只使用当前层 H+A 合并 scratch。旧 host_backed 结果按 H+A 预留，不与新语义混用。

本次发布只包含静态规划，未包含完整容量运行，也没有新实现的 allocated/reserved 峰值观测。

## 固定一个容量后的规划上界

HBM 规划额度为 floor(设备总 HBM × fraction) − 模型加载占用；显式 noncache headroom 另扣一次。规划包含 cache 预留、allocator allowance 和 CPU DRAM 预算。每行只改变一个容量，独立求得的 P 与 NH 上界不能拼成已验证组合。

预算输入：设备总 HBM 139.812 GiB，fraction=0.9，模型加载占用 9.344 GiB；HBM 规划额度 116.487 GiB，另留 noncache headroom 0.000 GiB；CPU DRAM 预算 512.000 GiB。

预算输入：设备总 HBM 139.812 GiB，fraction=0.9，模型加载占用 9.344 GiB；HBM 规划额度 116.487 GiB，另留 noncache headroom 14.500 GiB；CPU DRAM 预算 512.000 GiB。

余量来源：14.5 GiB 沿用此前 allocator 观察作为估计余量；未对 GPU 临时 candidate 版本重新校准，不保证物理峰值。

| Plan ID | 候选语义 | 固定条件 | 规划 P | 规划 NH | U | 有用 P=min(P,NH) | 下一候选违反 | 所选点实测 Run ID | 所选点预算观察 |
|---|---|---|---:|---:|---:|---:|---|---|---|
| `20261003_echo_gpu_candidate_nh_plan_01` | gpu_transient | P=32,768 | 32,768 | 29,818,880 | 455 | 32,768 | dram_budget | 未实测 | 未测量 |
| `20261003_echo_gpu_candidate_p_plan_base_01` | gpu_transient | NH=1,048,576 | 10,367,498 | 1,048,576 | 16 | 1,048,576 | hbm_budget | 未实测 | 未测量 |
| `20261003_echo_gpu_candidate_p_plan_nhmax_01` | gpu_transient | NH=29,818,880 | 6,496,179 | 29,818,880 | 455 | 6,496,179 | hbm_budget | 未实测 | 未测量 |
| `20261003_echo_gpu_candidate_p_plan_base_headroom_01` | gpu_transient | NH=1,048,576 | 9,045,156 | 1,048,576 | 16 | 1,048,576 | hbm_budget | 未实测 | 未测量 |
| `20261003_echo_gpu_candidate_p_plan_nhmax_headroom_01` | gpu_transient | NH=29,818,880 | 5,173,835 | 29,818,880 | 455 | 5,173,835 | hbm_budget | 未实测 | 未测量 |

`20261003_echo_gpu_candidate_p_plan_base_01` 的原始 P 规划上界超过 NH，多出的 slots 不增加可驻留的独立 host token 数。有用容量点的实测为：未实测；预算观察为未测量。

`20261003_echo_gpu_candidate_p_plan_base_headroom_01` 的原始 P 规划上界超过 NH，多出的 slots 不增加可驻留的独立 host token 数。有用容量点的实测为：未实测；预算观察为未测量。

下一候选违反的是规划约束，不代表实测 OOM。静态规划未证明物理容量最大值。完整执行与数值验收也需和 fraction 预算的实际观测分别判断。

## 测量范围与来源

模型是 checkpoint 前三层独立复制出的 10 个 dense block 工作负载替身；副本使用对应源层的 hidden/residual 输入。结果不代表经过训练的 DeepSeek 8B 或完整 61 层模型。

本次使用给定的模型加载占用作为规划输入，未重新加载模型或执行用户轨迹。

完整字节数、源码身份、checkpoint 路径和测量边界见 [summary.json](summary.json)；表格见 [cases.csv](cases.csv) 与 [plans.csv](plans.csv)。输入文件、源码快照及本报告生成器的 hash 见 [report_provenance.json](report_provenance.json)。
