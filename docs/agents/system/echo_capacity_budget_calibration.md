# ECHO 旧显存观测与额外扣减假设

更新：2026-10-03。当前 candidate 在 GPU 临时执行，NH 只保留 history；容量数字以
[实验报告](../../../experiments/deepseek_v32_echo_cache/README.md)为准。原先 candidate
写回 DRAM 的运行与容量表已被替换，本记录仅说明继续用于静态规划的 14.5 GiB 假设，
不把旧观测当成当前实现的显存测量。

## 预算口径

本轮沿用已有模型加载测量作为规划输入：总 HBM 为 150,121,545,728 B，加载十个
block 和公共权重的 device free-memory 差值为 10,032,775,168 B。用户指定的额度为：

```text
floor(总 HBM×0.9)−模型加载占用 = 125,076,615,987 B
```

默认不再额外扣除旧运行观测差额。规划同时保留逐个持久 storage 的 CUDA allocator 舍入
与未拆分尾部 allowance，以及默认 64 MiB scratch/fragmentation 余量。

## 14.5 GiB 的来源与限度

先前两条 16-user 轨迹在 cache 尚未释放的请求边界出现以下最大差额：

```text
max(reserved−allocated) = 14,674,100,736 B
max(device_used−reserved) = 835,780,608 B
两者之和向上取 256 MiB 档 = 15,569,256,448 B = 14.5 GiB
```

allocated 是 PyTorch 活跃分配，reserved 还包含 allocator 保留的空闲块；二者不能
相加。device_used 为总量减 free 的边界采样，可能包括 CUDA 上下文和其他占用，
不等于连续采集的进程峰值。这里排除释放 cache 后的样本，避免把已经释放的 pool
再算成下一轮与活跃 cache 并存的开销。

其中两项条件计划通过 `--noncache-headroom-gib 14.5` 人为扣除这部分额度，剩余 cache 与
执行空间额度为 101.986676 GiB。14.5 GiB 是从旧轨迹继承的估计参数，不是对 GPU
临时 candidate 版本的重新校准，也不是所有 P/NH 或请求序列的 reserved 上界；它
还可能与已有执行预留部分重叠。因此报告同时列出额外扣减为 0 和 14.5 GiB 的条件计划，默认结论采用前者。
观测差额本身不是预留；实现没有额外预分配 14.5 GiB，也没有证明它是当前路径必需
的固定开销。

用户不要求跑满容量。本次只生成静态计划和短数值检查，没有发布新版本的
allocated/reserved 峰值、完整用户轨迹或实测物理最大 P/NH。
