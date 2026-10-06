# cxldsagr

**sparse attention offloading** 研究项目，从 sparse KV fetching 与 attention 重叠
的设计出发，探索生成式推荐 serving 场景。当前候选方案之一是固定 user history、
每轮改变 candidate items 的 prefill 负载；模型、数据及场景代表性仍待完善。存储先关注
HBM 与 CPU DRAM，希望在有限 HBM 下复用更多历史 KV，并降低 offload 的服务开销。
当前验证模型为 NOSA 与 DeepSeek V3.2，优先 NOSA，主要开发平台为 SM90 / Hopper。

[研究状态](docs/status.md)按实验室四环节记录当前理解和缺口，
[下一步任务](docs/roadmap.md)列出待做事项；各环节可以交叉推进，研究者可直接修正。
`GR/` 支持固定 history 与候选变化；本地 serving 已增加按用户 LRU 保留历史的路径，
按统一 HBM / DRAM 准入账本管理首次访问与复访。旧热度短轨迹已结束独立实验维护，
通用 budget 实现和必要回归仍保留。当前系统对照使用完整 32 层、固定 P/NH、16 用户两轮的
[NOSA motivation 报告](experiments/nosa_motivation/README.md)：指定 token 配额下，
保留历史避免复访重建；稀疏搬运减少复访所需的数据量，async 仍未快于 sync。
报告给出重构前后同机三轮对照、独立 API 参考与内部重叠区间，分别说明剩余阶段开销、
性能门槛、物理容量和场景代表性的验证边界。优化进度以研究状态为准。
DeepSeek 真实前三层四方法的 64K+128 数值、独立计时和模块 profile 分别报告。
固定 P/NH、16 用户两轮的 [motivation 四方案对照](experiments/deepseek_v32_motivation/README.md)
单独报告十 block C10 工作负载；[官方 SGLang 复现](experiments/deepseek_v32_echo_official/README.md)
单独测量真实前三层的固定历史负载，按用户要求仅报告性能。官方 ECHO 不接入本地框架。
[两模型 cache 管理](experiments/cache_management/README.md)汇总静态 P/NH 规划、
各自的存储账本与完整请求观测，说明 NOSA 懒分配配额和 DeepSeek 全局 arena 的差异；
当前没有跑满容量。旧 4 GiB / W / chunk 对照已撤回；
静态规划与固定 16 用户的性能测量分别报告。
NOSA 共享执行资源、公共唯一准入 owner 与 DeepSeek dense 共享双缓冲已接入，
原共享资源实现的数值和缓存预算检查见
[独立工程验收](docs/agents/acceptance/unified_runtime_20261005/shared_cache_integration_evidence.json)。
上述工程检查不代表新容量实验已完成，也不提供新实现的性能排名。
原逐请求分配释放的 NOSA 入口继续保留；当前没有网络服务或到达队列吞吐测量。

NOSA 支持 dense、完整 sparse policy 与显式 pinned-DRAM offload。Offload 主 kernel
融合唯一页的 stripe fetch 和 persistent FA3 attention。通用 budget 路径的 HBM staging
覆盖一层完整逻辑地址；固定 P/NH 入口另用逐层有限 pool 与 session 标签，
尚不提供通用的热点淘汰策略。
已有完整模型数值验证和单层 overlap 性能报告，
完整模型的串行 GR serving 延迟另见上述实验；不包含 LM head 或并发服务测量。
算子结果见 [NOSA 实验](experiments/nosa_offload_overlap/README.md)。

DeepSeek V3.2 当前非 GR benchmark 使用真实 checkpoint 第 0–2 层依次传播，包含
embedding、final norm 和末 token LM head；这不是独立训练的三层模型。
Indexer 融合 KV prefetch，主 KV 使用 resident 存储或有限 HBM pool 与 pinned DRAM
backing。模型仍保留完整 61 层实现，当前数值、计时和 profile 覆盖真实前三层 64K + 128。
DeepGEMM main 与 FlashMLA 已接入，普通算子复用 FlashInfer，indexer 量化经 KDA 优化，
运行路径已移除 Hadamard。同 GPU 上四种 cache 方法的数值、阶段延迟与逐算子 MFU 见
[DeepSeek MFU](experiments/deepseek_v32_mfu/README.md)；不代表任务质量等价，GR 对照单独报告。
报告数字保留各自 run ID 与源码快照，目录迁移和回归检查不替代性能复测。

[实验索引](experiments/README.md)按 motivation、baseline 性能合理性、sparse pattern、
自有设计 microbenchmark 四类组织。数值验收、正式计时和 profile 分开运行，复用
覆盖相同执行路径的验收记录。两个模型共用容量计划、资源 owner/lease 和 token 执行契约，
保留各自计算与事务边界。错误直接传播；必要清理失败时一并保留全部异常，不自动重试
或切换 provider。各实验报告记录补测 run ID、源码、测量边界及仍未达到的性能目标。

```text
operators/nosa/                  indexer；attention/reference、device_only、offload
operators/deepseek_v32/          indexer；attention 三类；linear / grouped MoE
operators/common/               模型无关的 host record 搬运
models/attention_contracts.py   公共 indexer / attention 契约
models/nosa/                    NOSA 普通层、结构、权重、位置编码、选择语义与 KV 布局
models/deepseek_v32/             完整 ECHO checkpoint 推理及请求适配
executor/                       通用 prefill / extend 分块执行
cache/                          请求事务、resident / host backing、有限 token pool
serving/                        GR 驱动的本地串行执行
evaluation/                     共享来源记录、独立验收接口与内存审计
tests/integration/              跨模块正确性测试
experiments/                    各实验的源码、脚本、报告和原始产物
local/experiments/              本地 legacy 与 CPU DRAM 带宽实验，不进 Git
3rdparty/                       CUTLASS、DeepGEMM、DeepJIT、FlashMLA 子模块
GR/                             请求内容、热度和调度工具
docs/                           给人的研究状态与下一步任务；agents/ 保存内部执行文档
skills/research-supervisor/      项目内持续维护的 Research Supervisor
```

SM120 扩展、旧 synthetic 模型与相关可执行测量入口已清理。历史报告保留原测量含义，
本地副本及复现所需 Git revision 见 `local/experiments/legacy/deepseek_v32/README.md`。
CPU DRAM 带宽资料位于 `local/experiments/cpu_dram_bandwidth/`；两者均不随 Git 仓库分发。

## 运行

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync
source .venv/bin/activate

python -m models.nosa.infer --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --attention-mode sparse --prompt "请解释 KV cache 的作用。" --disable-thinking

CXLDSAGR_SM90_BACKEND=native python -m models.nosa.infer \
  --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B --attention-mode sparse \
  --cache-backend offload --offload-fetch-ctas 96 \
  --prompt "请解释 KV cache 的作用。" --disable-thinking

python -m serving.run_gr --help
python -m serving.run_multi_user --help
python -m models.deepseek_v32.infer --model /preset-models --help
bash experiments/deepseek_v32_mfu/scripts/run.sh --help
bash experiments/cache_management/scripts/run.sh --help
```

本机 NOSA checkpoint 位于 `/mnt/ssd-wlcb/chenkaiqi/NOSA-3B`、
`/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`，DeepSeek V3.2 位于 `/preset-models`。
原始权重保持外部存储，不提交仓库。模型形状支持范围见各模型与算子 README。

## 验证

```bash
bash scripts/run_tests.sh cpu
bash scripts/run_tests.sh gpu

# 额外启用 checkpoint 元数据与完整 NOSA-8B 64K + 1K 数值检查
NOSA_MODEL_PATH=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  bash scripts/run_tests.sh gpu
```

默认运行 CPU 回归。GPU 模式要求 Hopper、nvcc、CUTLASS、TVM FFI、Triton 和 FlashInfer，
缺少环境时失败；测试结果直接输出终端。普通 CLI 集成检查使用临时小模型权重，
不能代替完整 checkpoint 检查。性能实验另从 [实验索引](experiments/README.md) 进入。

提交检查使用 `uv tool install pre-commit` 和 `pre-commit install`；hook 自动执行 Ruff。
独立历史 DeepGEMM 基准使用 `uv sync --group legacy`，图表工具另加 `--group analysis`。

## 文档

- [系统架构](docs/architecture.md)：模块职责、请求与缓存生命周期、模型接入。
- [文档分工](docs/README.md)、[项目内 Research Supervisor](skills/research-supervisor/SKILL.md)
- [KDA 组件文档](docs/agents/kda/README.md)
- [模型](models/README.md)、[算子目录与类型](operators/README.md)、[实验](experiments/README.md)
- [Attention 契约](models/attention_contracts.py)、[缓存](cache/README.md)、[执行器](executor/README.md)、[serving](serving/README.md)
- [第三方依赖](3rdparty/README.md)、[GR](GR/README.md)

Supervisor 可直接使用项目文件，无需安装。在本项目对话中请求：

```text
请读取 skills/research-supervisor/SKILL.md 并运行 Supervisor，
维护 docs/status.md 和简短的 docs/roadmap.md 待办清单，
内部记录放到 docs/agents/research-supervisor/，保留我的修正并更新相关内容。
```
