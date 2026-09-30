# GR serving cache：三层 DSA baseline 原型

面向初次阅读的独立报告：[实验流程与结果](report/serial_beauty_64k_20260930/实验流程与结果.md)，
用通俗名称解释四组方案、缓存生命周期、计时口径、结果及结论边界。

## 目的与状态

研究固定用户历史、变化候选和用户热度下的跨请求 KV 复用、HBM/host 放置与搬运。
目标比较 ECHO、逐层全量 KV prefetch、无 overlap 的 sparse fetch，三组都执行相同
sparse attention。模型取 DeepSeek-V3.2 embedding 和完整前三层，单卡 H100 PCIe，
输出 hidden，不运行 LM head、文本生成或完整 61 层模型。

**已有完整 512 请求的串行同步原型测量；不是统一 HBM 预算或在线排队 serving 结果。**
本目录提供容量/源码 preflight、GR 请求流、独立正确性检查、完整回放与报告入口。
真实 embedding + 三层模型已在 H100 通过 4K + 1K、64K + 1K GR 输入的四路径 GPU 正确性验证；
已有串行 host prefix LRU 和基于已发生请求的用户级 HBM 保留选项；
全量 prefix 双缓冲预取已经接入，精确 block 热度、完整字节预算管理、
完整串行回放已经实现；真实到达时间与排队回放尚未实现。
两种长度均在下述 test-only logical top-k 次序控制下完成 `all`，三种 offload 输出
与同批 resident 文件交叉参考通过，保持原容差 `rtol=0.02, atol=0.02`。
其中 ECHO 使用显式 phase snapshot overlay 修复；这里的 64K 正确性检查来自 V2 Beauty
请求流的三用户子序列，不是完整 512 请求回放。独立的完整回放见末节，也不表示原版 ECHO 已通过。
系统设计、预算、policy 和验收条件见 [设计说明](../../docs/gr_cache_serving.md)。
准备材料不构成模型运行结果，不分配性能 run ID，不画预设加速比。

## 来源和依赖

- ECHO：`sjtu-zhao-lab/ECHO@bc1b75c1000010d0ac6f032ebaac283255c050b1`，
  本地 `3rdparty/ECHO/`，用独立环境。它的修改版 DeepGEMM 与仓库通用依赖分开。
- checkpoint：`/mnt/nfs/share/models/DeepSeek-V3.2`；前三层和 embedding 位于第一分片，
  preflight 只读 safetensors header，不加载权重。参数容量不是 GPU 峰值显存。
- 实际模型保留 checkpoint 的 FP8 权重量化配置、128x128 block scales 和上游权重后处理，
  activation 与 MLA KV 使用 BF16，index K 为 FP8、scale 为 FP32。
  preflight 中“FP8 解量化到 BF16”的容量列是假设估算，不表示运行的是全 BF16 权重模型。
- workload：复用 `GR.input_generator.create_input_generator`、`TextConfig` 和
  `GR.scheduling.ScheduleConfig`；tokenizer 使用同一 checkpoint，默认 Beauty / weighted / Poisson。
- preflight 只需 Python 3.12 标准库；workload 需要仓库版本的 `tokenizers` 和 `ijson`。
  ECHO 环境是否可运行须另行验证，源码存在不代表扩展已构建。

2026-09-29 已恢复上游 Python 依赖：上游固定 `flashinfer_python==0.4.0`，其官方
PyPI 元数据依赖 `apache-tvm-ffi==0.1.0b15`，但该版本在官方 PyPI 返回 404，
FlashInfer 官方 wheel/nightly 索引也未提供。使用官方 PyPI、允许预发布进行依赖解析
仍失败。相关上游问题见 [FlashInfer #1962](https://github.com/flashinfer-ai/flashinfer/issues/1962)。
改从 [TVM FFI 官方版本提交](https://github.com/apache/tvm-ffi/commit/70927743bd9f9e24eba65a06eb7a695137c49522)
构建同版本，并将源码约束同时传给 FlashInfer 的隔离构建环境；未用 `--no-deps`
跳过解析。不声称源码构建与已下架的原 wheel 字节一致，也不声称完整重建了作者环境。

本机隔离环境为 `3rdparty/ECHO/.venv`，CPython 3.12.14、Torch 2.8.0 / CUDA 12.8、
FlashInfer 0.4.0、SGLang 0.5.3.post3。系统 Python 缺少开发头文件，因此使用
uv-managed Python；CUDA 12.8 工具链也位于该环境，不替换系统 CUDA。
已从固定 ECHO 源码构建 `sgl-kernel==0.3.16.post2` 和修改版
`deep-gemm==2.1.1+bc1b75c`，后者在安装 sgl-kernel 后重新安装，避免被其附带版本覆盖。
Hadamard 使用上游 Docker 固定的
`Dao-AILab/fast-hadamard-transform@7fd811c2b47f63b0b08d2582619f939e14dad77c`。
安装后的 `uv pip check` 有一个已知版本声明冲突：上游 SGLang 声明
`sgl-kernel==0.3.15`，实际使用上述 ECHO 自定义版本，不能称依赖检查全部通过。
构建命令、wheel 摘要和环境记录保存于忽略的 `3rdparty/ECHO/.environment/`。
FlashMLA 使用 Docker 固定的
`deepseek-ai/FlashMLA@1408756a88e52a25196b759eaf8db89d2b51b5a1`，仅构建 SM90。
它的 CUTLASS 版本与顶层共享版本不同，按其精确 pin 隔离；ECHO DeepGEMM 的同版本
CUTLASS include 则链接顶层共享源码。已安装的 173 个包及源码指纹见上述本地记录。

新建 Python 环境可用以下命令；已有环境不要重复执行 `uv venv`。约束文件纳入版本管理，
不依赖本机临时文件。此过程只安装 Python 依赖，不构建完整 ECHO CUDA 后端：

```bash
uv python install 3.12.14
uv venv --python 3.12.14 3rdparty/ECHO/.venv
uv pip install --python 3rdparty/ECHO/.venv/bin/python \
  -b experiments/gr_cache_serving/scripts/echo-build-constraints.txt \
  -e 3rdparty/ECHO/sglang/python \
  'apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@70927743bd9f9e24eba65a06eb7a695137c49522'
uv pip check --python 3rdparty/ECHO/.venv/bin/python
```

已有 Python 环境、CUDA 12.8（含 `nvcc`、`cuobjdump`）及顶层 CUTLASS 后：

```bash
bash experiments/gr_cache_serving/scripts/build_echo_backend.sh
bash experiments/gr_cache_serving/scripts/build_echo_backend.sh --check
```

构建脚本固定五个组件的来源/版本，跳过已安装且来源匹配的组件，不修改系统 CUDA
或根仓库依赖。`--check` 只检查导入和来源，不代替 GPU 数值检查。

运行时将该环境的 `bin` 加入 `PATH`，供 JIT 找到 ninja。从本仓库根目录运行，
不要从 `3rdparty/ECHO/` 根目录导入 `sglang`，避免同名外层目录遮蔽已安装包。

## 准备入口

从仓库根目录运行，已有有效目录拒绝覆盖：

```bash
bash experiments/gr_cache_serving/scripts/prepare_echo.sh
python3 -m experiments.gr_cache_serving.src.preflight \
  --model-path /mnt/nfs/share/models/DeepSeek-V3.2 --layers 3

# 本机已建立的独立输入工具环境，不是 ECHO 模型运行环境。
uv venv 3rdparty/ECHO/.venv-tools --python 3.12
uv pip install --python 3rdparty/ECHO/.venv-tools/bin/python \
  tokenizers==0.23.2 ijson==3.5.1
3rdparty/ECHO/.venv-tools/bin/python -m experiments.gr_cache_serving.src.workload \
  --tokenizer /mnt/nfs/share/models/DeepSeek-V3.2 \
  --prefix-tokens 65536 --candidate-tokens 1024 \
  --num-users 128 --count 512 --history-cache-users 128 --curve-dataset beauty \
  --sampling weighted --arrival poisson --qps 1 --seed 42 \
  --output-dir experiments/gr_cache_serving/output/data/input_beauty_128_512
```

`qps` 只定义输入时间戳，生成器不按墙钟发请求。缓存命中必须由实际 cache 测量，不能
用 `common_prefix_tokens` 或重复用户比例冒充。脚本从实际 tokenizer 计算指令长度，
保证 prefix 包含指令与历史，candidate suffix 恰好为指定长度。
`--history-cache-users` 仅控制 GR 生成器在 CPU 缓存多少份历史文本，不是 serving KV cache。

请求产物为 `requests.jsonl`、`users.jsonl`、`metadata.json`、`requests.sha256`，包含指纹、边界和复访统计。
最终目录发布要求 Linux `renameat2(RENAME_NOREPLACE)`，原子拒绝覆盖；缺少支持时明确失败。
它们位于忽略的 `output/data/`，不作为报告结果；正式测量的日志、数据和 profiler 将按
项目约定分别写入 `output/log/<run_id>/`、`output/data/<run_id>/`、`output/profile/<run_id>/`。

准备工具测试使用临时 fixture，不加载模型或生成论文测量：

```bash
python3 -m unittest discover -s experiments/gr_cache_serving/tests -v
```

真实权重正确性入口与论文测量分开：

```bash
bash scripts/run_echo_tests.sh all
# 使用已经生成的完整长度输入，不裁短历史或候选。
SPARSEGR_ECHO_TEST_TRACE=experiments/gr_cache_serving/output/data/input_beauty_128_512_20260929_v2/requests.jsonl \
  bash scripts/run_echo_tests.sh all
```

`all` 以独立进程分别运行 resident、阻塞 sparse fetch、`echo_gr_adapted` 与 dense prefetch，
用临时 resident hidden 参考比较三组 offload 输出；临时参考结束后删除。
检查包括 candidate A/B/A 分支、强制 HBM 淘汰、释放后重新 prefill、
三个用户交错复用与 host 容量淘汰；指定的 trace 必须包含三个用户及首用户的复访。
不包含延迟统计，也不将这些子序列检查称为完整 512 请求回放。

### 次序控制与原版 kernel 限制

原生 top-k 通过原子操作分配输出位置，同一 selected multiset 也可能以不同顺序返回，
继而改变 FlashMLA 浮点归约次序；边界同分还可能改变入选 membership，二者不能混为一谈。
正确性入口默认启用 `tests/integration/echo_topk_control.py` 的
`test_only_logical_topk_order_v1`：只将原 indexer 返回的条目按逻辑 token 次序重排，
保留重复项和 `-1`，并逐行检查重排前后的 multiset 完全相同。
它覆盖 cold-prefill chunks 和 candidate extends，不修改预测阈值或替换选块集合，
也不解决边界 tie 的 membership 差异。额外排序和检查包含同步，禁止用于性能测量；
受控数值比对通过不等于原生 serving 路径具有确定性。

`SPARSEGR_ECHO_TEST_TOPK_ORDER=native` 可禁用该测试控制。
临时 resident reference 校验所有参与比较的输入、checkpoint 元数据指纹、
ECHO revision/patch SHA，以及次序控制 ID/源码 SHA，禁止混用 logical 和 native 参考。
权重元数据指纹不是全部权重 payload 的内容哈希。

64K 检查还暴露了独立的原版 SM90 fused prefetch 共享 flag 竞态：
`s_prefetch_enabled` 的 phase 分支判断可能与另一 warp 的更新交错，造成参与线程
进入不同的同步路径。它不是 top-k 输出排序问题，logical 次序控制不能修复它。
[echo_kernel.py](../../models/deepseek_v32/echo_kernel.py) 提供显式派生修复
`echo_sm90_prefetch_phase_snapshot_v1`：先把 phase flag 读入线程局部快照，再同步后
按快照分支。它不修改 ECHO checkout 或已安装包，在独立 header overlay 中替换一个 `.cuh`，
其余 include 链接原安装树；原包在使用期间须保持不变。

Python `open_echo_runner(..., kernel_patch=None)` 默认仍用原版 kernel；显式传入
`PREFETCH_PHASE_PATCH_ID` 才选择 overlay。GPU 正确性脚本默认只给 `echo_gr_adapted`
选择 `SPARSEGR_ECHO_TEST_KERNEL_PATCH=phase_snapshot`，其他路径不应用此修复；
设置 `SPARSEGR_ECHO_TEST_KERNEL_PATCH=native` 可复查原版；原版 ECHO 64K 仍观察到挂起，
不作为已通过的路径。此前 4K native top-k 次序四路径检查也通过，但不外推 64K native 次序。
修复后仍叫派生版本，不包装成未修改的原版 ECHO 结果。

overlay provenance 包含原 header/修复后 header SHA256、完整 installed include manifest 摘要、
C++ 扩展 SHA、初始化器源码 SHA、overlay identity 与 JIT 配置；
DeepGEMM 以修改后的 `.cuh` 内容和 include 路径生成不同 JIT cache key。
选择必须早于 DeepGEMM/SGLang 导入且每进程仅一次，退出后不恢复或继续使用该编译器；
切回 native 必须新进程。默认 overlay 位于 `~/.cache/cxldsagr/echo-deepgemm/`，
不在源码或 installed package 内。

### Dense prefetch 的边界

`dense_prefetch` 调用 [echo_dense.py](../../models/deepseek_v32/echo_dense.py)，记录为
`dense_all_prefix_staging_v1`。完整已有 prefix 从 host gather 到 pinned staging，
再按层用两个 HBM buffer 交替传输；当前 suffix 的本层 KV 尚不存在，由该层在 GPU
生成后直接补入 buffer。仍使用原 indexer 和 FlashMLA sparse attention，不调用 sparse recall。
下一层传输在当前层 kernel 入队后提交，是否有实际 overlap 须由时间线测量，不能由 stream
数量推出。当前原型保留 ECHO 的逐层 write pool，必须和新增双缓冲、mapping、index
一起计入预算；不能与只限制原 pool 的其他组直接宣称“同 HBM 预算”。
这条路径不利用原 write pool 的历史命中省略 prefix 传输，属于全量搬运消融，
不是已经完成的统一 admission/公平字节预算 baseline。

## 结果与结论

### 完整串行回放入口（2026-09-30）

新增 `src/replay.py` 与 `scripts/run_replay.sh`，测量现有同步原型的完整请求服务时间，
不是在线排队/网络 serving，也不将正确性测试耗时当作性能结果。默认完整读取上述
512 请求，先预热第一个请求两次并释放全部 prefix/device 槽位，再从空逻辑缓存按原顺序回放。
不读取未来热度，不启用额外用户级 HBM 保留策略，不启用 test-only top-k 排序。
Host 默认容纳 128 个历史，HBM MLA pool 每层 66624 slots，候选结束释放；
offload 的 index 仍覆盖完整 Host token 容量并常驻 HBM。

计时包围 `EchoCacheManager.execute`，包括校验/摘要、缓存查找、首访 prefill、候选前向、
逐 forward 同步和候选释放；不含 JSON 读取、模型加载、预热、输出有限性检查和日志。
分别记录首访、prefix hit、Host 淘汰后的 re-prefill，以及 prefill/candidate/其余管理时间。
`inverse_mean_service_requests_per_s` 仅为平均服务时间的倒数，不是在线吞吐或满足 SLO 的容量。
记录明确的 MLA/index buffers、PyTorch allocated/reserved 峰值和源码指纹，
仍未强制三组总 HBM 字节数完全相同；dense 额外 staging、resident 全量 GPU pool 不能忽略。

```bash
bash experiments/gr_cache_serving/scripts/run_replay.sh \
  echo_gr_adapted serial_beauty_64k_echo_i64_r1_20260930 --repetition 1
bash experiments/gr_cache_serving/scripts/run_replay.sh \
  sparse_sync serial_beauty_64k_sparse_i64_r1_20260930 --repetition 1
```

只在完整成功后发布 `output/data/<run_id>/summary.json`、逐请求 `requests.jsonl`
和 `output/log/<run_id>/`；失败日志保留在系统临时目录，不进入实验结果。
`SPARSEGR_REPLAY_DIAGNOSTIC=1 CUDA_LAUNCH_BLOCKING=1` 可做同步定位，产物仅在临时目录，
不能用于性能报告。测量期间源码发生变化时拒绝发布。完整回放结果以下方有效运行记录为准。

本次扩大 Host pool 后发现原 `_recall_update_extend_kernel` 的 `host_idx * 576`
使用 signed int32，约 373 万 token 后元素地址溢出。增加显式派生修复
`echo_extend_recall_int64_address_v1`（`models/deepseek_v32/echo_recall.py`），
在乘法前提升 host index 到 int64；仅克隆 Triton JIT 对象并修改这一行，不改上游文件或缓存策略。
作用域结束恢复原对象，源码和修复后 SHA 写入 runner provenance。
跨 4 GiB pinned Host 地址的 GPU 回归、修复后的 64K 四路径 resident 交叉检查已通过；
这不替代完整流回放，也不把原失败运行保留为性能结果。

已完成输入准备 `input_beauty_128_512_20260929_v2`：128 个用户池，512 条请求中实际访问
115 个用户，115 次首访、397 次复访；每条请求为 65536-token prefix 和 1024-token suffix。
产物位于 `output/data/input_beauty_128_512_20260929_v2/`，metadata 明确记录
`serving_executed=false`。这是 Beauty 热度曲线驱动的合成 GR 输入，不是真实线上 trace；
复访次数不等于 cache hit 次数。

GPU 正确性使用真实 FP8 checkpoint 前三层、BF16 activation/MLA KV：4K + 1K 与上述
V2 trace 三用户子序列的 64K + 1K 均通过 resident、`sparse_sync`、修复版
`echo_gr_adapted`、`dense_prefetch` 的 `all` 检查。包含候选分支、KV/index 快照、
HBM 驱逐/重取、host LRU 淘汰和逐路径 resident 文件参考，不包含计时或完整 512 请求回放。
2026-09-30 共 107 项聚焦 CPU 测试通过：39 项实验工具（含 7 项回放/报告）、
40 项模型适配、24 项 serving、4 项次序控制；
未运行全仓库回归。这些是正确性检查，不生成论文性能 run ID。

### 首轮完整回放结果

日期 2026-09-30，H100 PCIe 80GB，Torch 2.8.0 / CUDA 12.8，真实 checkpoint FP8 权重、
BF16 activation/MLA KV。每个模式独立进程、同一 seed 42 trace、一次完整冷缓存回放，
各 512 请求，均为 115 首访 + 397 prefix hit，Host eviction 为 0。
预热及计时边界见上节；没有多次重复或多种子置信区间，不使用测试专用 top-k 重排。
所有输出做有限性检查；不把它当成推荐质量或 native 路径逐元素一致性证明。

| 路径 | 首访 p50 (ms) | 复访 p50 (ms) | 复访 p95 (ms) | 累计服务时间 (s) | 峰值 Torch allocated (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident，全量 GPU 容量参照 | 1180.66 | 29.19 | 30.05 | 147.46 | 34.31 |
| Blocking sparse fetch | 1455.90 | 34.79 | 35.58 | 181.21 | 7.62 |
| ECHO，GR + kernel 修复 | 1861.83 | 43.92 | 44.74 | 231.42 | 7.62 |
| Dense prefetch，另加 staging | 1756.93 | 53.26 | 54.61 | 223.29 | 7.80 |

累计服务时间为逐请求计时之和，不含初始化/预热/检查/日志，也不是 Poisson 时间轴的排空时间。
这里的 prefix hit 是历史可复用，不是 HBM KV hit ratio。各组实际峰值 reserved 和主要 buffer
字节另存 summary；Torch allocated 不等于 NVML 总显存，表格不是强制等总 HBM 预算比较。
三组 offload 具有相同 66624-token/layer 主 KV pool（共约 219.59 MiB）、
约 27.00 GiB Host MLA pool、约 3.094 GiB resident index；dense 另有约 178.79 MiB
持久 HBM staging/mapping。Resident 为全部 128 用户预留 GPU KV，不是等预算 LRU 对照。

![串行 GR 原型的延迟分布、请求序列与显存代价](report/serial_beauty_64k_20260930/overview.png)

图表/汇总：[summary.csv](report/serial_beauty_64k_20260930/summary.csv)、
[manifest.json](report/serial_beauty_64k_20260930/manifest.json)。图中的重复次数目前均为 1，
min/max whisker 不代表置信区间。有效 run ID 如下，原始数据与日志分别位于
`output/data/<run_id>/`、`output/log/<run_id>/`，源码快照另存各 data 目录的 `source_snapshot.json`：

- `serial_beauty_64k_echo_i64_r1_20260930`
- `serial_beauty_64k_sparse_i64_r1_20260930`
- `serial_beauty_64k_dense_i64_r1_20260930`
- `serial_beauty_64k_resident_i64_r1_20260930`

报告生成命令（新报告目录拒绝覆盖，图表不使用失败运行）：

```bash
uv run --no-project --python 3rdparty/ECHO/.venv/bin/python --with matplotlib==3.10.7 \
  python -m experiments.gr_cache_serving.src.report_replay \
  experiments/gr_cache_serving/output/data/serial_beauty_64k_echo_i64_r1_20260930 \
  experiments/gr_cache_serving/output/data/serial_beauty_64k_sparse_i64_r1_20260930 \
  experiments/gr_cache_serving/output/data/serial_beauty_64k_dense_i64_r1_20260930 \
  experiments/gr_cache_serving/output/data/serial_beauty_64k_resident_i64_r1_20260930 \
  --output-dir experiments/gr_cache_serving/report/serial_beauty_64k_20260930
```

本轮观察：ECHO 复访 p50 比 blocking sparse fetch 高 26.2%，比 dense prefetch 低 17.5%；
累计服务时间比 blocking 高 27.7%，也比 dense 高 3.6%。ECHO 的冷 prefill 占累计服务时间
约 90.1%，因此必须把首访和复访分开，不用一个整体平均值代替缓存复用分析。
ECHO 的峰值 Torch allocated 比全量 resident 低约 77.8%，代价是复访 p50 高约 50.4%。
这些是在当前配置中实测的取舍，不是 ECHO 在所有 GR 场景都更慢的结论。

尚不能从本轮推出主 attention overlap 的收益、ECHO 较慢的具体 kernel 原因、
HBM 命中率/搬运量、在线 QPS/SLO，或 sparse attention 相对 dense attention 的收益。
下一步应做同总预算的 HBM-only prefix LRU 对照、容量 sweep、多种子/重复和独立 Nsight
时间线。当前三组执行器保留逐 forward 同步，ECHO 也包含显式派生修复，不能冒充论文原版结果。
