# KV Cache Offload 笔记

日期：2026-07-01

## 背景

我们做 KV cache offload，主要目标是解决容量问题，而不只是带宽问题。

DSA / sparse attention 可以减少每个 token 计算时实际读取的 KV 数量，但它并不消除“完整 KV cache 必须存下来”这件事。长上下文或者高并发下，完整 KV cache 仍然会吃掉大量 GPU 显存。所以这里的优化目标是：

- 尽量把大部分甚至全部 KV cache 放到 GPU 显存之外；
- GPU 按 sparse indices 直接取需要的 KV records；
- 保留 sparse MLA 降低读取量的优势；
- 先搞清楚 GPU 直接读 host pinned memory 的真实性能。

现在 MLA kernel 本身离硬件峰值还有距离，所以一开始不应该直接把完整 attention 路径调到很复杂。更基础的问题是：如果 KV cache 放在 CPU pinned mapped memory 里，GPU kernel 直接读 KV-sized records，到底能跑出多少带宽？

## 当前假设

当前 V32 MLA decode 的 KV cache 是按 token 存的一条连续 record：

| 字段 | 字节数 | 格式 |
| --- | ---: | --- |
| noPE value/key payload | 512 | FP8 E4M3 |
| scales | 16 | 4 x FP32 |
| RoPE key payload | 128 | BF16 |
| 总计 | 656 | mixed |

所以 offload 场景里的访问模式不是“4B random load”。一个 sparse index 选中一个 token 后，有效读取粒度更接近一整条连续的 656B KV record。record 之间的顺序可以是 sparse/random，但被选中 record 内部的字节是连续的。

GPU 直接访问 host memory 的方式是 mapped pinned memory：

```cpp
cudaHostAlloc(&hptr, bytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&dptr, hptr, 0);
```

之后 GPU kernel 直接解引用 `dptr`。一阶模型里，可以先把这条路径看成 host-link limited，并且不假设有有效的 L2 cache 复用。这和先把 KV staging 到 HBM 再计算是不一样的。

## 基线

本机 RTX 5080 的硬件基线见 [benchmark_5080_results.md](benchmark_5080_results.md)：

| 测试 | 结果 |
| --- | ---: |
| BF16 tensor-core GEMM | 75.99 TFLOP/s |
| FP16 tensor-core GEMM | 76.03 TFLOP/s |
| STREAM-like HBM bandwidth | 797.9 GB/s |
| Device-to-device copy, logical bytes | 398.7 GB/s |
| Device-to-device copy, HBM read + write | 797.3 GB/s |

mapped-memory benchmark：

```bash
nvcc -O3 -std=c++17 -arch=sm_120a benchmarks/mapped_kv_record_kernel.cu \
  -o /tmp/mapped_kv_record_bench

/tmp/mapped_kv_record_bench --records 524288 --reps 20
```

RTX 5080 上的结果：

```text
 records  recB  location    pattern         ms       GB/s
--------------------------------------------------------------
  524288   528    mapped        seq     23.259      11.90
  524288   528    mapped     random     31.632       8.75
  524288   528    device        seq      0.809     342.33
  524288   528    device     random      0.805     344.02
  524288   656    mapped        seq     27.102      12.69
  524288   656    mapped     random     34.317      10.02
  524288   656    device        seq      0.884     388.87
  524288   656    device     random      0.897     383.36
```

这个 benchmark 里，每个 warp 负责读取一条连续 record。`random` case 随机的是 record 顺序，不是 record 内部的 byte 顺序。

## 初步结论

对最接近真实 KV layout 的 656B record：

| 来源 | 访问模式 | 带宽 |
| --- | --- | ---: |
| CPU pinned mapped memory | sequential records | 12.69 GB/s |
| CPU pinned mapped memory | random records | 10.02 GB/s |
| GPU device memory, same kernel | sequential records | 388.87 GB/s |
| GPU device memory, same kernel | random records | 383.36 GB/s |

也就是说，在这个访问粒度下，GPU 直接读 host pinned mapped memory 大概是 10-13 GB/s。它远慢于 HBM，但也不是 4B random load 那种极端悲观数字。更合适的心智模型是：通过 host link 做 sparse record streaming。

和 STREAM-like HBM 带宽相比：

- mapped random 656B：`10.02 / 797.9 = 1.26%` HBM bandwidth；
- mapped sequential 656B：`12.69 / 797.9 = 1.59%` HBM bandwidth。

和同一个 record-read kernel 读 device memory 相比：

- mapped random 656B：`10.02 / 383.36 = 2.61%`；
- mapped sequential 656B：`12.69 / 388.87 = 3.26%`。

## 对 Sparse MLA Decode 的影响

当前 decode kernel 是 MQA-like 的：所有 Q heads 共享同一份 KV cache。但是实现上按 16 个 head 一组处理，也就是 `HPB=16`。如果是 128 heads，就有 8 个 head groups；每个 head group 会重新 gather 同一批 selected KV records。

这个重复读取在 HBM 上已经很重要；如果 KV cache offload 到 host memory，它大概率会变成主导问题。

如果直接把当前 kernel 搬到 host KV 上，host 侧流量大概是：

```text
host_bytes ~= batch * topk * 656B * ceil(num_heads / 16)
```

对于 `num_heads = 128`，重复读取系数就是 8。这会抵消掉 MQA 共享 KV 的很多收益。更适合 offload 的 kernel 应该尽量让 host read 变成：

```text
host_bytes ~= batch * topk * 656B
```

然后在 GPU 上把同一条 KV record 复用给更多 heads。

可能的方向：

- 先把 selected KV records 从 host staging 到 HBM workspace，再用正常 sparse MLA math 计算；
- host read 和 compute 融合，但 CTA 组织方式要让一次 host record load 服务更多 head groups；
- 通过 batch 多个 decode requests 增加 in-flight host transactions；
- offload 前进一步压缩 KV record，比如更紧凑的 KV 格式；
- GPU 上保留一个小的 hot cache，host pinned memory 作为 backing store。

## 当前相关文件

- [mapped_kv_record_kernel.cu](../benchmarks/mapped_kv_record_kernel.cu)：standalone CUDA benchmark，用来测 GPU 直接读 CPU mapped pinned memory 的 KV-sized records。
- [benchmark_decode_fp8.py](../benchmarks/benchmark_decode_fp8.py)：当前 sparse MLA decode benchmark，里面有 logical KV traffic 和按 head-group 重复读取后的 traffic 估算。
- [benchmark_5080_results.md](benchmark_5080_results.md)：本机 RTX 5080 的 compute / HBM bandwidth 基线。

## 下一步要测什么

接下来应该从纯 record-read bandwidth 往真实 offload decode path 靠：

1. 写一个 minimal decode-like kernel，把 device KV cache 换成 mapped pinned KV cache，测端到端带宽。
2. sweep `topk`、batch size、head count，看看 host read 需要多少并发才能打满。
3. 比较“direct compute from host memory”和“先 staging selected KV 到 HBM 再 compute”的代价。
4. 定量测 `HPB=16` 的重复读取惩罚，对比一个“一次读取 KV record，复用给更多 heads”的 kernel。
5. 检查 NUMA / PCIe placement，因为 mapped pinned memory 带宽会对 CPU memory path 很敏感。
