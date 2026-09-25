# NOSA framework 重构验证

## 目的与验证边界

验证 GR → serving → executor → NOSA layers → operators 的调用、请求级 cache 生命周期、
原生成接口和 profiler 挂钩。当前后端为 FlashInfer Full Attention + 全 HBM KV；不评价
sparse attention、offloading、吞吐或跨请求缓存收益。此实验与
[精确 64K+1K 性能实验](../nosa_gr_65536_1024/README.md) 分开记录。

## 当前运行方式与调用模块

从仓库根目录运行，选择新的 run ID：

```bash
bash experiments/nosa_framework_refactor/scripts/run.sh --help
bash experiments/nosa_framework_refactor/scripts/run.sh cpu_001 cpu
bash experiments/nosa_framework_refactor/scripts/run.sh gpu_001 gpu
```

默认模式为 `all`；CPU 回归覆盖 `models/nosa/tests`、`cache/tests`、`executor/tests`、
`serving/tests`、`GR/tests`、64K+1K 实验和归档 profiler 测试。GPU 模式需要 CUDA 与
NOSA-8B checkpoint；可用 `NOSA_MODEL_PATH` 指定权重。

- `src/gpu_smoke.py` 复用 `models.nosa.tests.test_model` 的小模型和 dense reference；
  显式调用 `experiments.legacy.nosa_gr_forward.src.measure/profile` 的三阶段测量与 Chrome
  trace 归因函数，用于接线验证。归档实验的历史性能结果不因此恢复。
- shell 脚本调用 `serving.run_gr` 和 `models.nosa.infer`，`src/analyze_cli.py` 只校验已保存
  stdout/stderr，不负责进程调度。源码指纹复用 64K+1K 实验的 `src/sources.py`。
- 日志放 `output/log/<run_id>/`，JSON 与源码快照放 `output/data/<run_id>/`，
  Chrome trace 放 `output/profile/<run_id>/`。脚本拒绝覆盖已有 run ID，并保留子进程失败状态。
  CPU 测试的临时文件通过 `--basetemp` 固定在 `output/data/<run_id>/pytest/`。
  `tests/` 不重复实现模型测试，相关回归位于上述被调用模块。

## 合并后验证（2026-09-25）

在 `cxldsagr` 合入 framework 分层，同时保留本工作区的实验目录规则与精确 65536 + 1024
实验边界。Run ID：`merge_cpu_20260925_02`，运行命令：

```bash
NOSA_MODEL_PATH=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  bash experiments/nosa_framework_refactor/scripts/run.sh merge_cpu_20260925_02 cpu
```

CPU 回归 **117 passed、2 skipped**，另有 **34 个 subtests** 通过。两项跳过分别为
FlashInfer CUDA 数值测试和本地缺少 DeepSeek tokenizer 的 GR 测试。合并检查还通过了
47 个 Python 文件的 Ruff lint/format、模型与实验 CLI 帮助、shell 语法及 run ID 防覆盖检查、
216 个文档相对链接和 `git diff --check`。新增回归覆盖实际模型经 executor/cache 的
attention 形状审计，以及移动到 `models.nosa.layers` 后的 RoPE profiler 挂钩与恢复。

日志位于 `output/log/merge_cpu_20260925_02/`，检查摘要、环境和源码指纹位于
`output/data/merge_cpu_20260925_02/validation.json`。17 个历史源码、报告、CSV、日志和 trace
副本与原文件逐字一致。另一次历史 CLI 日志重分析使用 run ID
`merge_historical_cli_reanalysis_20260925`，结果与原 JSON 一致，没有执行 GPU。

当前环境没有可用的 NVIDIA 驱动，未执行合并后的 GPU 数值或性能验证。尝试收集 DeepSeek
历史测试时因缺少 `deep_gemm` 失败，未改动依赖环境；该项不计入上述通过数。

## 历史结果与合并状态

下文记录原 worktree 在 2026-09-25 的验证，run ID 归档为 `20260925`。
原 `/tmp/nosa-framework-validation/` 中仍存在的 JSON、脚本快照、stdout/stderr、Chrome trace
已原样保存到上述分类 output；不复制 JIT cache。对应关系与 SHA-256 位于
`output/data/migration.json`，原报告在 `output/data/20260925/report_original.md`。
小模型为 2 层、hidden 256、4 Q heads / 2 KV heads、head dim 64、MLP 384、BF16；
输入 9 tokens、prefix 5、candidate 4、chunk size 3。计时与 profile 各执行一次，没有性能
统计用的预热/重复；记录时间仅用于确认三阶段执行。真实权重 serving 执行两条 384-token
请求，生成 CLI 执行一次 25-token prompt，最多生成 4 tokens。

本次合并只进行可用的 CPU/静态验证；当前 NVIDIA 驱动不可用，未重跑 GPU smoke 或性能测量。
以下数字保留原平台、依赖与测量含义，不能作为合并后 GPU 已验证的声明。

## 原 worktree 验证记录


2026-09-25，在独立 worktree 的 `refactor/nosa-local-framework` 分支完成分层重构。
起点为 `dc8e85d` 加启动时已有的 NOSA/GR 工作区改动副本；本次没有向原工作区回写文件。
原工作区同时进行的实验目录整理未合入此 worktree。

## 实现状态

GR generator → serving → executor → NOSA layers → operators 已接通。
cache manager 管理请求级 resident cache 和每次 forward 的写入/提交/释放。
每条 GR 请求独立执行 stable prefix prefill 与 candidate extend，返回末 token hidden，
不执行 LM head。原文本生成接口、checkpoint 参数路径及实验三阶段计时语义保留。

当前实际后端为 FlashInfer Full Attention + 全 HBM KV。NOSA indexer、64-token block 的
1 sink + 16 local + 47 top-k 策略，以及 SM90 sparse attention / fetch 仅预留接口；
没有 DRAM backing、数据搬运、淘汰、跨请求缓存复用或 overlap 实现。
本次不构成 sparse/offloading 的正确性或性能验证，也没有重跑历史性能矩阵。

## 环境与隔离

- 只读复用原工作区 Python 环境及 `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B` 权重。
- PyTorch `2.10.0+cu132`、FlashInfer `0.6.18`；没有重建锁定的 PyTorch `2.12.1+cu130` 环境。
- GPU compute capability 为 SM90；PyTorch 报告 NVIDIA H200，`nvidia-smi` 标识为 NVIDIA M403。
- 验证脚本、FlashInfer/Triton/扩展缓存及输出位于 `/tmp/nosa-framework-validation/`。
- DeepSeek、SM120、依赖配置和三个子模块版本未改动；既有 NOSA 历史报告与 CSV 保留原字节。

## 检查结果

- CPU 回归 **108 passed、2 skipped**，另有 34 个 subtests 通过。覆盖模型数学、权重、
  cache 事务和释放、executor 容量预检、请求隔离、GR 输入、生成及 profiler 接线。
  GPU 用例在该次 CPU 运行中跳过，随后单独验证通过；另一项因本地 DeepSeek tokenizer
  不可用而跳过。
- 原 GPU Full Attention 参考测试通过，覆盖 prefill、追加 prefill 与 decode。
- 小模型三阶段 profiler 验证通过：784 个 GPU activity 完成唯一归因，RoPE 与 attention
  挂钩均生效；full/split 末 token hidden 最大绝对误差 `0.0234375`，余弦相似度
  `0.99996996`。这些小模型计时仅用于接线验证，不作为性能结果。
- 真实 NOSA-8B serving：同用户两条 384-token 请求，均返回 `[4096]` BF16 特征；
  `--device cuda` 正确规范化为 `cuda:0`。
- 真实 NOSA-8B 直接生成入口：25-token prompt、chunk size 8，生成 4 tokens，执行
  3 次 decode；首次 JIT 开销计入请求时间，不作为吞吐测量。
- Ruff 检查和 34 个 Python 文件的格式检查通过；旧/新 CLI 帮助、文档链接及
  `git diff --check` 通过。新增目录保持 namespace 源码目录，没有添加 `__init__.py`。

CPU 回归入口：

```bash
NOSA_MODEL_PATH=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B CUDA_VISIBLE_DEVICES='' \
  python -m pytest models/nosa/tests cache/tests executor/tests serving/tests \
  GR/tests experiments/tests/test_nosa_gr_profile.py -q -p no:cacheprovider
```

真实权重 smoke 使用的入口：

```bash
python -m serving.run_gr --device cuda --count 2 --num-users 1 \
  --user-lengths 256 --item-lengths 128 --prefill-chunk-size 128
python models/nosa/infer.py --device cuda --prompt '请用一句话解释 KV cache。' \
  --disable-thinking --max-new-tokens 4 --prefill-chunk-size 8
```

实际验证从独立 worktree 运行，使用原环境解释器，并将其 `bin` 加入 PATH 供 Ninja 使用。
GPU 验证记录为 `/tmp/nosa-framework-validation/gpu_smoke.json` 和
`/tmp/nosa-framework-validation/real_cli_smoke.json`；原始 profiler trace 与 CLI 输出
也保存在该临时目录，不作为仓库中的历史性能报告。
