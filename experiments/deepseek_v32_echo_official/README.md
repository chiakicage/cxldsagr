# DeepSeek V3.2 官方 ECHO 适配实验

本实验沿用 [motivation](../deepseek_v32_motivation/README.md) 的模型、输入和固定 P/NH 容量，比较官方 ECHO offload 路径与本轮重新执行的 HBM-only 基线。
C10 正式运行 `echo_official_c10_20261004_u16_r2_01` 已完成两方案各 32 条请求、32 条独立 resident 重跑及全部保存输出的验收。两方案均启用公共投影与收尾计算图。下文数字来自本轮冻结源码。

官方 ECHO 的复访均值为 45.561 ms，HBM-only 为 2106.515 ms；对应请求时间比为 46.235。32 条请求耗时合计为 74.930 s 和 67.533 s，ECHO 相对 HBM 增加 10.95%。

## 结果

| 方案 | 访问 | 请求数 | history hit | 均值 ms | 中位数 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|
| HBM-only | 首次 | 16 | 0/16 | 2114.286 | 2115.503 | 2121.285 |
| HBM-only | 复访 | 16 | 0/16 | 2106.515 | 2105.440 | 2117.721 |
| 官方 ECHO 适配路径 | 首次 | 16 | 0/16 | 4637.566 | 4638.341 | 4651.222 |
| 官方 ECHO 适配路径 | 复访 | 16 | 16/16 | 45.561 | 45.470 | 46.138 |

完整轨迹的 HBM/ECHO 时间比为 0.901。首访 history 构建均值分别为 2100.909 ms 和 4591.403 ms；复访 candidate 执行阶段均值分别为 10.112 ms 和 41.411 ms。ECHO 保留全部 16 位用户的历史；HBM 的 history 配额只容纳一位用户，两轮顺序访问均需重建。被淘汰后的访问仍计作复访。复访端到端时间比包含历史重建差异，不能解释为 attention 算子加速。

完整数字见[报告](report/results.md)、[分组数据](report/summary.csv)、[逐请求数据](report/per_request.csv)、[配置与验收](report/summary.json)和[报告来源](report/report_provenance.json)。六个正式报告文件从本轮冻结版 `src.report` 生成的 `output/data/<run_id>/report/` 原样复制。

## 与本地 C10 实现对照

以下引用本地与官方各自的新正式轨迹，保留两组 HBM 对照。两组使用相同输入 token、请求顺序、模型、P/NH 和计算图模式。表格整理本身不增加 GPU 测量。

| 实现 | 首访均值 ms | 复访均值 ms | 复访 p95 ms | 32 请求总耗时 s |
|---|---:|---:|---:|---:|
| HBM-only（本地对照） | 2176.886 | 2170.244 | 2178.394 | 69.554 |
| 我们的 ECHO（已有验收版本） | 2283.433 | 24.947 | 25.509 | 36.934 |
| HBM-only（官方对照） | 2114.286 | 2106.515 | 2117.721 | 67.533 |
| 官方 ECHO 适配路径 | 4637.566 | 45.561 | 46.138 | 74.930 |

本地 run ID 为 `motivation_c10_20261004_u16_r2_01`，官方 run ID 为 `echo_official_c10_20261004_u16_r2_01`。两次独立运行使用同一物理 GPU。两组共同记录的有效精度设置一致；仅一组记录的字段及各实验的 policy ID 不能用于确认完整精度策略相同。本地 HBM 使用 mainline DeepGEMM resident logits，本地 ECHO 使用项目融合 indexer/prefetch，二者均使用 FlashInfer top-k；官方组使用 ECHO 原始 resident/fused logits 和原始 top-k。两组选择路径和 HBM 基线有差异，不能把观测差额全部归因于缓存或搬运路径。单次轨迹不足以判断差异是否稳定。本地和官方输出分别在来源运行内验收，未新增跨运行数值比较。

完整来源、未四舍五入差值和生成器身份见[对照报告](report/existing_implementation_comparison.md)、[对照数据](report/existing_implementation_comparison.csv)和[来源清单](report/existing_implementation_comparison.json)。对照生成器是报告修正版本，其哈希单独记录；它不改变冻结推理源码。

## 设置与测量边界

- 固定 P=65,536 个逐层 history HBM 槽，NH=16,777,216 个全局 host token；H=65,536、A=128、history chunk=1,024，16 用户顺序访问两轮，seed=42。P/NH 是 token 容量，不另设 cache 字节子预算，也不扣除经验性 headroom。
- 使用 `/preset-models` 的真实 checkpoint 前三层，独立复制为十个 dense block，共 7,827,793,408 参数，含 embedding、final norm 和 LM head。每个副本重放对应 source block 的 hidden/residual 输入，独立权重、KV、indexer 不共享。这是 checkpoint 工作负载替身，不代表经过训练的十层模型或完整 DeepSeek V3.2。
- 普通 linear 使用 FP8，主 KV 为 BF16 576 元素 record，indexer K/scales resident，RoPE 后不加 Hadamard。计算全部 candidate hidden `[128, 7168]` 和末 token logits `[1, 129280]`。candidate 整批在 GPU 临时执行后丢弃，不写入 DRAM。
- 两方案各预热两位用户的首访及第一位用户的复访；ECHO 复访预热实际读取 host KV。随后释放预热 cache，从空缓存执行正式轨迹，每请求测量一次。
- 同步墙钟包含输入搬入、准入/淘汰、miss 时 history 构建、候选执行和清理。forward 内 GPU 计数累积与归约计入延迟；加载、编译、图准备、预热、计数 host 读取、输出保存与比较不计时。传输计数只覆盖 candidate forward。

两方案都使用官方 ECHO 原始 top-k：HBM-only 调用官方 resident logits，ECHO 调用官方融合 indexer/prefetch、allocator、精确 recall 和释放 helper。offload 每次 indexer 调用均执行官方融合 kernel，策略为 `official_fused_every_offload_call_v1`。模型投影与 FlashMLA 沿用 motivation。结果覆盖官方算子接入本地单卡固定历史 GR 生命周期后的适配路径，不是上游完整 TP=8 AWQ SGLang serving。本轮未测内部 overlap；合成输入及工作负载替身不证明实际 GR 任务质量或场景代表性。

## 计算图、精度与源码

十个独立层在 1,024-token history chunk 和 128-token candidate 两种形状下，各有 projection/finish 两类计算图，共 40 张。HBM、官方 ECHO 和独立 HBM 重跑的 replay 总数分别为 41,600、21,120、41,600；逐请求覆盖完整，eager fallback 为零。图 bank 的静态 storage、实际 allocator 分配和所选 reservation 上限分别核对，规划上限不等于已分配 storage。预热 replay 明细未保存，因此不把正式轨迹检查扩大为预热覆盖证明。

FP32 matmul 使用 `highest`，matmul TF32 关闭；BF16/FP16 reduced-precision reduction 分别为 `True` / `True`，cuDNN TF32 为 `True`。运行前后设置及图捕获精度均经核对。三个正式/验证 runner 均实际选择 C10 native token validator，其加载二进制和编译依赖哈希已核对。

PyTorch 将设备标识为 NVIDIA H200；运行后的 `nvidia-smi` 查询将同一 UUID 标识为 NVIDIA M403。GPU UUID 为 `1bdee8b4-22ac-536c-208b-bfb4ed38b878`，132 SM，可见 HBM 139.812 GiB；Python 3.12.13、PyTorch 2.12.1+cu130、CUDA 13.0。依赖版本为 DeepGEMM 2.8.1+057ca59、FlashMLA 1.0.0+ba89a34、FlashInfer 0.6.18。

运行窗口保存了 150 次全部设备的 compute-process 查询，仅观察到本轮指定进程及 GPU。最大采样间隔为 30.383 s；离散观测不能排除采样间隙中的其他工作。监测程序、原始观测和退出记录随运行数据保留。

源码身份为 `fef9fef45d923343967198552709f7a7549da9145ce0927b535539d86f8cc60d`；输入身份为 `7e4c737a86464c12238191933e707344426231bf671a29658837e676ff5284ae`。官方 ECHO 固定提交为 `bc1b75c1000010d0ac6f032ebaac283255c050b1`。完整源码、官方 native/JIT 产物和运行参数随原始数据保留；checkpoint 身份核对依赖路径、shard 大小和 mtime 清单，没有完整权重内容哈希。

## 内存与搬运

| 方案 | allocated 峰值 GiB | reserved 峰值 GiB | 边界采样设备已用最大 GiB | cache HBM 最大 GiB | cache DRAM 最大 GiB |
|---|---:|---:|---:|---:|---:|
| HBM-only | 14.549007 | 23.875000 | 24.616272 | 6.625494 | 0.000000 |
| 官方 ECHO 适配路径 | 16.409619 | 24.521484 | 25.891663 | 8.485923 | 320.001038 |

PyTorch 峰值在各方案释放预热 cache 后重置，包含权重和正式执行；reserved 含 allocator 缓存，不能与 allocated 相加。cache 为请求边界的实际分配计量，已经包含 shared 与全部 session；其中计算图计入观测到的静态 allocator blocks 和 private reserved segments，不能再次加上 shared 字段。规划 reservation 是分配上限，另行记录。设备已用量由 total−free 计算，只在边界采样，不是连续峰值。

NH 对应的 BF16 主 KV 逻辑 storage 为 180.000 GiB，本轮 pinned record backing 实际为 320.000 GiB。正式轨迹保留 1,048,576 个 history token，未填满 NH；相同 P/NH 不等于相同总 HBM/DRAM 字节数。pinned allocator 的 active-byte 统计可能受缓存块复用影响，不单独用于推断物理 backing。

| 访问 | candidate H2D B | candidate D2H B |
|---|---:|---:|
| 首次 | 0 | 0 |
| 复访 | 1,246,828,032 | 0 |

HBM-only 的 candidate H2D/D2H 均为零。这些计数不含 history 构建阶段。

## 数值验收

HBM、独立 HBM 重跑和官方 ECHO 共保存 96 份输出。CPU FP64 独立重算全部 hidden/logits，32 条 resident 重跑与 32 条 ECHO 请求均通过原有门槛：输出有限、shape/dtype 一致，hidden/logits relative L2 分别不超过 0.005/0.01，且至少 99.9% 元素满足 `|actual-reference| <= 1/32 + (1/64)*|reference|`。规则在正式 offload 测量前根据独立 resident 校准固定，没有按本轮输出调整。

| 比较 | 输出 | 最大 relative L2 | 最低元素通过比例 | 最大绝对误差 |
|---|---|---:|---:|---:|
| HBM 重跑 / HBM | hidden | 0.001140275 | 99.99989101% | 0.046875 |
| HBM 重跑 / HBM | logits | 0.003396577 | 100.00000000% | 0.0625 |
| 官方 ECHO / HBM | hidden | 0.002477424 | 99.99291556% | 0.1796875 |
| 官方 ECHO / HBM | logits | 0.003394204 | 100.00000000% | 0.0625 |

allclose、逐位相等和最大绝对误差分别报告，见[数值汇总](report/numerical_summary.json)。官方 top-k 顺序及并列值可能变化，不能把全部差异归为已经证实的舍入误差。独立 HBM 重跑只用于数值验证，不发布延迟。

C10 另已通过三项 checkpoint 数据路径检查：eager/graph 的完整 H64K 检查分别验证 1,310 次 attention、2,650,296,320 次选中 KV，以及相同有序选择下的 FlashMLA 输出；H2304 检查保留普通 consumer，覆盖两位用户、128/121-token candidate、非默认 stream 和输出所有权。这些是正确性测试，不能替代正式 serving 轨迹。来源见[C10 验证记录](../../docs/agents/system/deepseek_echo_official_c10_validation.md)。

## 复现与模块

按[第三方依赖说明](../../3rdparty/README.md)准备基础环境和固定提交的只读 ECHO checkout。从仓库根目录运行，使用新的 run ID；`--help` 给出全部参数，默认 checkpoint 为 `/preset-models`。

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TRITON_PTXAS_PATH="$PWD/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
CUDA_VISIBLE_DEVICES=0 bash experiments/deepseek_v32_echo_official/scripts/run.sh \
  --run-id deepseek_v32_echo_official_c10_new_run --compute-graphs
```

Blackwell 环境项只固定依赖导入配置，不表示本实验在 Blackwell 上运行。入口复用 `experiments.deepseek_v32_motivation.src.measure` 的配置、预热和请求检查，使用共享 `GR/` 生成器及 `serving/persistent.py` 的计时边界。模型适配位于 `models/deepseek_v32/official_serving.py` 和 `official_cache.py`；官方独立绑定位于 `operators/deepseek_v32/indexer/official.py`。

本轮实际命令从冻结目录 `/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0` 执行，使用本页 run ID 和 `--compute-graphs`；完整命令及环境保存在 `output/data/echo_official_c10_20261004_u16_r2_01/gpu_monitor/window.json`。

stdout/stderr 位于 `output/log/echo_official_c10_20261004_u16_r2_01/`；配置、输入、96 份输出 tensor、内存采样、源码、native 快照和独立审计位于 `output/data/echo_official_c10_20261004_u16_r2_01/`。只有验收通过的运行进入实验目录。报告重建须使用本轮冻结源码及尚不存在的输出目录：

```bash
python -m experiments.deepseek_v32_echo_official.src.report \
  --run-dir <run_dir> --output-dir <new_report_dir>

python -m experiments.deepseek_v32_echo_official.src.compare_existing \
  --motivation-report experiments/deepseek_v32_motivation/report \
  --official-report experiments/deepseek_v32_echo_official/report \
  --output-dir <new_comparison_dir>
```
