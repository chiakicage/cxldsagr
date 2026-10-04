# ECHO 固定容量报告：实现与显存账本

更新：2026-10-03。面向读者的容量结论和可复现命令统一见
[实验报告](../../../experiments/deepseek_v32_echo_cache/README.md)。本记录用于核对实现；
原先按 H+A 持久保存 candidate 的公式和容量表已由本版替换。

## 本轮执行范围

DeepSeek GR 的 `echo/serial_sparse` 只保留固定 history。`retained_session_capacity`
返回 H，NH 按 `padded(H)` 分配 host pages；私有 indexer 也只分配 H 行。
`extend_candidate` 一次执行完整候选，不按模型 chunk 拆分。每层主 KV storage 为
`[P+1+A,576]` BF16，前 P+1 行用于历史和 sentinel，尾部 A 行用于当前请求。
候选没有 host ID，不参与历史映射、priority 或 free bitmap，也不写回 DRAM。

indexer 使用一份 backend 共享的 `[H+A,128]` FP8 K 及 FP32 scales workspace。
当前层复制自己的 history indexer，然后直接写入候选；不额外保留逐层 candidate
indexer。候选执行完成并同步后 discard 临时 KV，恢复 history offset，session 长度
保持 H。失败直接抛出并释放 session，不保留失败请求用于重试。普通 `extend` 和
其他后端的存储策略不变。

## 64K history 的字节核对

配置为十个独立 dense block，H=65,536、A=128、prefill C=1024，执行空间按
`Q=max(C,A)=1024`、完整上下文 H+A 规划。P 是每层历史槽数，U 为完整用户数。

```text
U = floor(NH / 65,536)
私有 HBM/session = 10×65,536×132 + 2,160 + 4,096
                 = 86,513,776 B
共享候选主 KV    = 10×128×1,152 = 1,474,560 B
单层合并 indexer = (65,536+128)×132 = 8,667,648 B
E(1024)         = 1,214,517,248 B

HBM_base = E(1024) + 11,774P + 104NH + 11,870
         + 86,513,776U + 1,474,560 + 8,667,648
DRAM     = 10×next_power_of_two(1,152NH) + NH/16 + 4,096U + 40
```

HBM_base 包含保守执行预留，不是精确常驻 tensor 总量。模型、allocator allowance
及额外保护余量分别计费。主 KV 为每层 1,152 B/token，历史 indexer 为每层
132 B/token；NH 增长还增加 HBM 中的映射和全部用户的 indexer，所以 DRAM 不是
NH 的唯一约束。

512 GiB DRAM 在当前单 arena 分配方式下可规划 455 个完整 history，NH=29,818,880。
其十层 pinned 主 KV 共占 320 GiB。增加到 456 个用户时，每层 arena 从 32 GiB 档
升为 64 GiB，十层共 640 GiB；这是静态分配档位边界，没有执行满用户容量测试。

## 验证与来源

- 短 checkpoint 检查使用 H=2304、A=16/23、C=256、十个独立 dense block。ECHO、
  serial_sparse 和 dense_prefetch 的全部 candidate hidden/logits 与独立 HBM 逐位一致。
  ECHO/serial 的 candidate D2H=0，history host 内容和私有 indexer 保持不变。
- transient CUDA cache 检查通过。runner、backend、diagnostics 与实验元数据的最终
  定向回归共 64 项通过；另有缓存和 planner 的模块检查。它们是实现正确性检查，
  不作为容量或性能实验。
- 新容量报告只收入五份静态计划，不把原 candidate 写回 DRAM 的运行重新标为当前结果。
  P/NH 的边界由计划及相邻候选约束计算；没有宣称实测 OOM 或证明物理最大值。

实现入口：[serving_backend.py](../../../models/deepseek_v32/serving_backend.py)、
[sparse_token_pool.py](../../../cache/sparse_token_pool.py)、
[sparse_token_cache.py](../../../cache/sparse_token_cache.py)。
规划入口：[capacity.py](../../../models/deepseek_v32/capacity.py)。
