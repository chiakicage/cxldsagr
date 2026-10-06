# DeepSeek V3.2 官方 SGLang 性能复现

本实验属于 baseline 性能合理性检查，使用 ECHO 发布的 SGLang，在固定 history、
变化 candidate 的负载上分别测量 offload 与 HBM-only。模型取真实 DeepSeek V3.2
checkpoint 的第 0–2 层，依次传播 hidden/residual，包含 embedding、final norm 和
末 token LM head；沿用原始 FP8 权重，不使用 AWQ，也不复制成十层。

按用户要求，本轮仅报告性能，数值验收未通过。原本将官方 ECHO 接入本地框架的
适配代码、专属测试及旧实验记录已删除。本目录只保存独立 SGLang 运行的报告和
选定数据；复现代码、环境、权重及原始运行产物留在本地
`3rdparty/ECHO/reproduction/cxldsagr/`，不进入父项目 Git。

## 结果

两组均在 GPU 7（PyTorch 识别为 NVIDIA H200 / SM90）上运行，CPU 绑定 72–79，
内存绑定 NUMA 1，OMP/MKL 线程数均为 8。每组独立预热 3 条请求，然后在新服务
进程中顺序执行 16 个用户的两轮访问，共 32 条正式请求。

| 方案 | 访问 | 请求数 | 完整 history 命中 | 平均延迟 ms | 中位延迟 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SGLang HBM-only | 首次 | 16 | 0/16 | 1238.838 | 1216.973 | 1311.721 |
| SGLang HBM-only | 复访 | 16 | 0/16 | 1222.878 | 1222.715 | 1229.755 |
| SGLang ECHO offload | 首次 | 16 | 0/16 | 1930.418 | 1906.806 | 2021.202 |
| SGLang ECHO offload | 复访 | 16 | 16/16 | 74.499 | 73.663 | 80.046 |

HBM-only 与 ECHO 的 32 条请求计时总和分别为 **39.387 s / 32.079 s**。
复访吞吐分别为 **0.818 / 13.423 请求/秒**，分母为该组 16 条复访请求的计时总和。
ECHO 每次复访均命中 65,536-token 历史；HBM-only 的复访均重建历史。
因此复访耗时差包含历史复用的影响，不能解释为 attention kernel 的加速。
分位数采用排序样本位置 `(n-1)*0.95` 的线性插值，只描述本轮轨迹。

| 报告 | run ID | 分组统计 | 逐请求数据 | 完整数据与来源 |
| --- | --- | --- | --- | --- |
| [HBM-only](report/sglang_fixed_history/hbm/report.md) | `first3_hbm_perfonly_20261006_01` | [CSV](report/sglang_fixed_history/hbm/summary.csv) | [CSV](report/sglang_fixed_history/hbm/requests.csv) | [JSON](report/sglang_fixed_history/hbm/report.json) |
| [ECHO offload](report/sglang_fixed_history/echo/report.md) | `first3_echo_perfonly_20261006_04` | [CSV](report/sglang_fixed_history/echo/summary.csv) | [CSV](report/sglang_fixed_history/echo/requests.csv) | [JSON](report/sglang_fixed_history/echo/report.json) |

## 负载与容量

H=65,536、A=128、history prefill chunk=1,024，16 用户按同一顺序访问两轮，seed=42。
TP=1、DP=1，串行请求，page size=64，context length=131,072。
每请求向 `/generate` 传入完整 H+A token IDs，计算全部 candidate hidden，再从
末 token 的完整词表 logits 采样一个 token；没有后续 decode forward。

| 方案 | device 可用 token 槽 | host token 槽 | allocator padding |
| --- | ---: | ---: | ---: |
| SGLang ECHO offload | 65,664，即 64K+128 | 16,777,216 | device 额外 1 槽 |
| SGLang HBM-only | 131,072 | 无 host KV pool | device 额外 64 槽 |

容量由实际 allocator 日志及 scheduler 状态核验。ECHO 的 device pool 由 history
与 candidate 共同使用，NH 是共享逻辑 host token 配额。HBM-only 调度器还会从总池
扣除 6 个 prompt 位置，65,664 槽无法接收本次完整请求，因此采用已跑通的 131,072
槽配置。**两组不是等容量或等 HBM 字节预算对照。** 本轮也未填满 ECHO 的 NH。

官方 radix cache 保留完整 H+A。ECHO 会将 candidate 写入 host，HBM-only 将缓存
保留在 HBM；这里不加入项目本地的 candidate discard 或 session LRU。
Indexer 沿用官方 Hadamard 和量化路径，主 KV 使用 BF16。上述语义与本地 C10
工作负载不同，不能将其数值结果或计时直接拼入本次比较。

## 计时与数值状态

每请求计时从 HTTP 序列化开始，覆盖服务器执行、完整响应接收和 JSON 解码。
输出检查、文件保存与请求间记录不计时；计时请求不返回完整 hidden/logprobs。
模型加载、环境核验和服务控制初始化在请求计时之外。独立预热服务关闭后，正式
服务从空 radix 和 allocator 开始；JIT 磁盘缓存可复用，正式服务中的模型与算子
首次使用开销仍包含在请求中。

服务就绪检查使用 `/get_model_info`，再通过一次无延迟 `/slow_down` 控制请求启动
正常的 tokenizer 接收循环，随后从 `/get_server_info` 核验容量。这些步骤不执行
推理，也不操作缓存。不使用会执行推理的 `/health` 或不能清除 ECHO 映射的
`/flush_cache`。

此前三个独立数值检查保存了 96 份输出：resident 重跑的 16/32 条比较、ECHO 的
19/32 条比较未通过预定门槛，全部 greedy token 一致。用户明确要求继续测量性能；
本轮不调整门槛、不签发数值通过记录，也不把差异归因于已经证实的单一根因。
检查原件留在本地 `3rdparty/ECHO/reproduction/cxldsagr/output/acceptance/diagnostic_first3_triplet_20261006_01.json`。
两份报告均记录 `numerical_acceptance=false`。

正式运行、请求完成状态、容量、报告签名和逐请求统计已核对。GPU 离散观测中未见
所选设备上的外来计算进程，关闭后已确认释放；离散观测不能证明采样间隙内没有其他
工作。主服务阶段观测到的本进程显存最大值为 HBM-only 13,532 MiB、ECHO 20,068 MiB，
包含初始化；它不是连续峰值、PyTorch allocated/reserved 或 cache 字节预算验收。

## 环境与来源

ECHO 固定提交为 `bc1b75c1000010d0ac6f032ebaac283255c050b1`，上游已跟踪源码保持
原样。独立环境使用 SGLang 0.5.3.post3、Torch 2.8.0+cu128、ECHO DeepGEMM 2.1.1、
sgl-kernel 0.3.16.post2、FlashInfer 0.4.0 和 Transformers 4.57.1；FlashMLA 固定到
ECHO CI 使用的 `1408756a88e52a25196b759eaf8db89d2b51b5a1`。

两组的模型字节、负载、官方源码、native 构建、GPU、CPU/NUMA、精度和客户端计时
边界一致。后补 HBM 运行增加了显式计时入口，因此 driver 与容量说明文件的源码
身份不同；模型计算、请求客户端和生命周期实现不变。各自的实际身份保存在报告
JSON 中，完整文件 SHA256 与发布签名见[发布清单](report/sglang_fixed_history/publication.json)。
选定报告由 ECHO 目录中的 `src.report` 生成后按原字节复制到本目录，未改写原报告签名。

原始数据分别位于本地
`3rdparty/ECHO/reproduction/cxldsagr/output/data/first3_echo_perfonly_20261006_04/` 和
`3rdparty/ECHO/reproduction/cxldsagr/output/data/first3_hbm_perfonly_20261006_01/`，
日志在同一复现目录的 `output/log/<run_id>/`。这些路径不随项目 Git 分发。

## 复现

以下从仓库根目录进入已有的独立复现目录，使用新的 run ID。代码和环境不复制到
本实验目录，也不通过项目的模型或 serving 入口执行。

```bash
cd 3rdparty/ECHO/reproduction/cxldsagr
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

numactl --physcpubind=72-79 --membind=1 \
  bash scripts/run.sh --run-id sglang_echo_new \
  --case echo --mode bench --gpu 7 --performance-only \
  --numerical-diagnostic output/acceptance/diagnostic_first3_triplet_20261006_01.json

numactl --physcpubind=72-79 --membind=1 \
  bash scripts/run.sh --run-id sglang_hbm_new \
  --case resident_reference --mode bench --gpu 7 --performance-only \
  --numerical-diagnostic output/acceptance/diagnostic_first3_triplet_20261006_01.json

source env/activate.sh
python -B -m src.report --run output/data/sglang_echo_new
python -B -m src.report --run output/data/sglang_hbm_new
```
