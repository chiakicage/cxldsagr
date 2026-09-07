# Sparse MLA 实现说明

这份文档说明 `sparse_mla_sm120` 里的 sparse MLA forward 实现，重点比较 prefill 和 decode 的差异。

## 入口

当前主要 Python API 在 `sparse_mla_sm120/flash_mla_sm120/ops.py`：

- `sparse_mla_prefill_fwd(...) -> (output, max_logits, lse)`
- `sparse_mla_decode_fwd(...) -> (output, out_lse)`

C++/CUDA binding 在 `sparse_mla_sm120/csrc/binding.cpp`。按 `Q.shape[-1]` 推断 model type：

- `d_qk == 576`：V32 layout。
- `d_qk == 512`：MODEL1 layout。

KV cache 可以是 CUDA tensor，也可以是 pinned CPU tensor。这一点是 offload 实验的关键，因为 kernel 可以直接拿到 pinned CPU tensor 对应的 device pointer。

## 数据布局

公共参数：

- `D_V = 512`
- `D_ROPE = 64`
- `HPB = 16`，一个 head group 处理最多 16 个 heads。
- `BI = 64`，每个 sparse tile 处理 64 个 selected KV entries。
- 一个 CTA 使用 12 个 warps：8 个 math warps + 4 个 IO warps。

V32 KV record 是 inline layout，每 token 656B：

```text
[0:512)    FP8 E4M3 NoPE K/V
[512:528)  4 x FP32 scale
[528:656)  BF16 RoPE K
```

MODEL1 layout 是 paged/footer layout：

```text
per token data: [0:448) FP8 NoPE, [448:576) BF16 RoPE
scale footer:   page_block_size * 576 之后，每 token 8B scale
logical stride: 584B/token
IO stride:      576B/token
```

IO 路径在 `csrc/kernel/common/kv_cache_io.cuh`。IO warps 根据 `indices` gather 一个 `BI=64` tile 到 shared memory，并用双 buffer 加 mbarrier 和 math warps 做 pipeline。

## 共同计算流程

prefill 和 decode 的主体 math loop 很接近，都是按 sparse tile 做：

1. IO warps 根据 sparse indices gather KV tile。
2. Math warps 加载/量化 Q。
3. 计算 QK NoPE。
4. 计算 QK RoPE。
5. 对 invalid index、`topk_length` 越界和 extra cache 越界做 mask。
6. 用 online softmax 维护每个 head 的 running max 和 row sum。
7. 把 softmax weight 量化成 FP8。
8. 做 XV NoPE MMA，把 weighted value 累加到 output accumulator。
9. 如果模型的 V 带 RoPE 部分，再做 XV RoPE。
10. epilogue 里归一化 accumulator，写 output 和 LSE。

这里的关键点是 online softmax：kernel 不需要把所有 logits materialize 出来，而是在遍历 sparse KV tiles 时持续维护稳定的 softmax 状态。

## Prefill

prefill 入口：

```text
flash_mla_sm120.sparse_mla_prefill_fwd
  -> _C.sparse_mla_prefill_fwd
  -> sparse_mla_prefill_launch_v32 / sparse_mla_prefill_launch_model1
  -> sparse_mla_prefill_kernel 或 sparse_mla_prefill_mg_kernel
```

prefill 的特点是 **单 kernel、单 pass、直接输出**：

- grid 以 token 和 head group 展开。
- 对每个 token/head group，kernel 遍历该 token 的全部 top-k tiles。
- 不做 split-KV，所以没有 partial buffer，也没有 combine kernel。
- epilogue 直接写 BF16 `output`、FP32 `out_lse`，可选写 `out_max_logits`。

head 数较少时使用 SG kernel：

```text
SG: 16 heads / CTA
grid = num_tokens * (num_heads / 16)
```

head 数较多时使用 MG kernel：

```text
MG: 32 heads / CTA
grid = num_tokens * (num_heads / 32)
```

MG 的目的很直接：同一批 KV tile 被两个 16-head groups 复用，减少 KV gather 重复，提升 compute/load ratio。V32 prefill 当前支持 `num_heads = 16, 64, 128`。

## Decode

decode 入口：

```text
flash_mla_sm120.sparse_mla_decode_fwd
  -> get_decode_metadata
  -> sparse_mla_splitkv_v2_fwd
  -> sparse_mla_combine_v2_fwd
```

decode 的特点是 **小 batch / 小 token 数、延迟敏感、需要 split-KV 提高并行度**。

Python wrapper 先计算：

- `replicate_h = ceil(num_heads / 16)`
- `num_sm_parts = max(num_sms / replicate_h, 1)`
- `sched_meta`
- `num_splits`

`get_decode_metadata` 是一个单 warp GPU scheduler。它把每个 request 的 top-k blocks 分配给 `num_sm_parts` 个 partition，并记录：

- partition 从哪个 request / block 开始；
- 到哪个 request / block 结束；
- request 是否被切成多个 split；
- `num_splits` prefix sum，用于把 partial results 写入连续 workspace。

decode main kernel 的 grid 是：

```text
(head_group, s_q, num_sm_parts)
```

每个 CTA 不一定只处理一个 request。它会根据 scheduler metadata 在一个 partition 中顺序处理一个或多个 batch elements。对于每个 request，如果当前 partition 覆盖的是完整 top-k，则可以直接写最终 BF16 output；如果 request 被 split 了，则写：

```text
o_accum:   [total_splits, s_q, num_heads, D_V] float32
lse_accum: [total_splits, s_q, num_heads]      float32
```

之后 `sparse_mla_combine_v2_fwd` 合并同一个 request 的多个 split。combine 使用 LSE 做稳定归并：

```text
global_lse = logsumexp(split_lse)
output = sum(exp(split_lse - global_lse) * split_output)
```

如果有 `attn_sink`，decode 在 direct-output 或 combine 阶段把 sink 合进 LSE/denominator。

## Prefill 和 Decode 的核心差异

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 目标 | 吞吐优先，处理较多 tokens | 延迟优先，处理少量 decode tokens |
| 并行策略 | token x head group | head group x scheduler partition |
| KV 遍历 | 每个 CTA 扫完整 top-k | top-k 可以被拆成多个 split |
| 输出 | 直接写 BF16 output + LSE | 无 split 时直接写；有 split 时写 partial，再 combine |
| 额外 kernel | 无 | `get_decode_metadata` + main split-KV + combine |
| workspace | 基本不需要 partial workspace | 需要 `o_accum`、`lse_accum`、`num_splits`、`sched_meta` |
| head group | SG 16 heads/CTA，MG 32 heads/CTA | 主要按 16 heads/head group |
| KV 复用 | MG prefill 能让 32 heads 共享一次 KV gather | 多 head groups 会重复 gather 同一批 selected KV |
| offload 影响 | 更容易通过较大 token 数摊薄 host latency | host KV 重复读取更容易成为瓶颈 |

## 对 KV Cache Offload 的含义

当前 decode kernel 每个 head group 独立 gather selected KV。对 V32 128 heads 来说：

```text
head_groups = ceil(128 / 16) = 8
host_bytes ~= batch * topk * 656B * head_groups
```

如果 KV cache 在 HBM，这个重复读取还能依赖 L2/HBM 带宽缓解；如果 KV cache 在 CPU pinned mapped memory，host link 带宽会更紧张。因此 offload 方向上更值得探索的 decode 形态是：

```text
host_bytes ~= batch * topk * 656B
```

也就是让一次 host KV record load 尽量服务更多 heads，或者先把 selected KV staging 到 HBM workspace，再复用给后续 attention math。

prefill 的 MG kernel 已经体现了这个思路：一个 CTA 处理 32 heads，让同一份 KV tile 在更多 heads 间共享。decode 后续若要适配 host offload，也需要类似地减少 head-group 之间的重复 host gather。

## `cp.async.bulk.prefetch` 对 Host Offload 的结论

关注的指令是普通地址版：

```ptx
cp.async.bulk.prefetch.L2.global [srcMem], size;
```

它和 `prefetch.tensormap` / `cute::prefetch_tma_descriptor` 不是一回事：后者预取的是 TMA descriptor，不是数据本身。

在 RTX 5080 上测到的现象：

- 对 device memory，`cp.async.bulk.prefetch + wait_group.read` 后再普通 load，timed tile cycles 明显下降，说明它确实能作为 L2 data prefetch 使用。
- 对 CPU mapped pinned host memory，timed tile cycles 基本不下降，说明它没有把 host 数据稳定变成后续普通 load 的 L2-hit 路径。
- host streaming bandwidth 偶尔在 `prefetch+wait` 模式下变高一些，但 `prefetch-ahead` 基本无收益甚至变慢；这更像 transaction/scheduling 形态变化，不是可靠的 PCIe latency/bandwidth fix。

关键数字，512MiB buffer：

| Tile | Host no-prefetch | Host prefetch-ahead | Host prefetch+wait | Host timed cycles no-prefetch | Host timed cycles after prefetch+wait |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4KiB | 21.99 GB/s | 17.33 GB/s | 24.64 GB/s | 1850 | 1843 |
| 16KiB | 15.73 GB/s | 15.83 GB/s | 19.40 GB/s | 6832 | 6731 |
| 64KiB | 11.16 GB/s | 10.75 GB/s | 13.38 GB/s | 27395 | 26989 |

结论：`cp.async.bulk.prefetch` 不能解决 GPU 访问 CPU pinned host memory 低带宽/高延迟的问题。更可利用的方向仍然是减少 host 读取次数：让一次 host KV record load 服务更多 heads，或者先把 selected KV staging 到 HBM/shared memory，再被更多 math 复用。纯 streaming host read 靠 prefetch 本身不够。

## GPU-CPU PCIe Bandwidth 的修正结论

之前把 `cudaHostAlloc` 路径下 `~20-22 GB/s` 的 GPU-CPU 带宽主要归因到 CPU memory bandwidth，是不准确的。后续 CPU STREAM-like 测试显示这台 DDR4-3600 双通道机器能稳定跑到 `45-46 GB/s` DRAM traffic，所以 `~20 GB/s` 不是 CPU DRAM 的硬上限。

关键 A/B：

| Host buffer | H2D | D2H | GPU mapped read | GPU mapped write |
| --- | ---: | ---: | ---: | ---: |
| `cudaHostAllocMapped` | ~21 GB/s | ~20 GB/s | ~21 GB/s | ~15 GB/s |
| 4KiB-backed memory + `cudaHostRegisterMapped` | ~21 GB/s | ~20 GB/s | ~21 GB/s | ~15 GB/s |
| THP-backed memory + `cudaHostRegisterMapped` | ~44-46 GB/s | ~53 GB/s | ~36 GB/s | ~17 GB/s |

系统当前启用了 Intel DMAR/IOMMU，THP 是 `madvise` 模式。`/proc/self/smaps` 里普通 registered buffer 是 `AnonHugePages: 0 kB`，而 `MADV_HUGEPAGE` 后的 1GiB buffer 是 `AnonHugePages: 1048576 kB`。因此最可能的瓶颈是默认 4KiB pinned page 在 DMA/IOMMU/scatter-gather 路径上的开销；hugepage-backed registered memory 大幅减少 page/IOMMU mapping 数量，把 `cudaMemcpy` 带宽推到了接近这台平台的实际上限。

后续做 host offload benchmark 时，不要把 plain `cudaHostAlloc` 当作 PCIe5 x16 上限测试。高带宽路径应该用：

```cpp
posix_memalign(&ptr, 2 * 1024 * 1024, bytes);
madvise(ptr, bytes, MADV_HUGEPAGE);
touch_all_pages(ptr, bytes);
cudaHostRegister(ptr, bytes, cudaHostRegisterMapped);
```

详细数据见 `docs/gpu_cpu_pcie_bandwidth_root_cause.md`。

### GPU Direct `ld.global` Host Memory

专项测了 GPU kernel 直接 `ld.global` 读 CPU pinned mapped memory：

| Host buffer | C++ `uint4` load | PTX `ld.global.v4.u32` |
| --- | ---: | ---: |
| `cudaHostAllocMapped` / 4KiB pages | ~21-23 GB/s | ~21-23 GB/s |
| THP-backed `cudaHostRegisterMapped` | ~36 GB/s | ~36 GB/s |

C++ 普通 `uint4` load 和显式 PTX `ld.global.v4.u32` 基本一致，所以普通 mapped-read kernel 已经是在测 GPU direct load host memory。进一步优化后，THP + 每线程 `uint2` load 最好能到 `~39.4 GB/s`，比 `uint4` 的 `~36-37 GB/s` 更好，可能是 warp 256B request 形态比 512B 更适合这条 PCIe read completion 路径。

测过但没有超过 `uint2` direct load 的形态：

- `cp.async.bulk.prefetch` + wait / ahead：明显更慢。
- `cp.async.bulk.shared::cta.global` 读 host 到 shared：最好约 `34 GB/s`。
- 普通 `cp.async.shared.global` 4B/8B/16B copy 到 shared：最好约 `34 GB/s`。
- 普通 load size：4B/thread 约 `25 GB/s`，8B/thread 最好约 `39.4 GB/s`，16B/thread 约 `37 GB/s`，32B/thread 明显变慢。
- 每线程 ILP 2/4/8：没有提升，反而变慢。
- 临时把 GPU endpoint MRRS 从 256B 改到 512B/1024B：没有提升，已恢复 256B。

所以 direct GPU read 的当前最佳是 `~39.4 GB/s`，仍低于 copy engine 的 `cudaMemcpy` H2D `~44-46 GB/s` 和 D2H `~53 GB/s`。因此 dense bulk transfer 还是应优先 staging/copy；direct mapped load 更适合 sparse/on-demand 访问。

专项 benchmark 在 `gpu-benches/gpu-host-direct-ld/`，详细结果见 `docs/gpu_host_direct_ld_results.md`。
