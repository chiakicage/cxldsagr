# cxldsagr

**sparse attention offloading** 研究项目，从 sparse KV fetching 与 attention 重叠
的设计出发，探索生成式推荐 serving 场景。当前一种候选方案是固定 user history、
每轮变化 candidate items 的 prefill 负载；模型、数据及场景代表性仍待完善。存储先关注
HBM 与 CPU DRAM，希望在有限 HBM 下复用更多历史 KV，并降低 offload 的服务开销。
当前验证模型为 NOSA 与 DeepSeek V3.2，优先 NOSA，主要开发平台为 SM90 / Hopper。

[研究状态](docs/status.md)按实验室四环节记录当前理解和缺口，
[下一步任务](docs/roadmap.md)列出待做事项，允许交叉推进和人直接修正。
`GR/` 已支持固定 history 与候选变化；本地 `serving/` 仍逐请求分配和释放 cache，跨请求 history KV 复用与服务吞吐验证尚未完成。

NOSA 支持 dense、完整 sparse policy 与显式 pinned-DRAM offload。Offload 主 kernel
融合唯一页的 stripe fetch 和 persistent FA3 attention；HBM staging 仍覆盖一层完整
逻辑地址，尚无有限 slots / eviction。已有完整模型数值验证和单层 overlap 性能报告，
完整模型 offload 性能尚未测量，见 [NOSA 实验](experiments/nosa_offload_overlap/README.md)。

DeepSeek V3.2 支持独立完整 61 层 ECHO prefill/extend，不依赖 SGLang。Indexer 融合
KV prefetch，主 KV 使用 resident 存储或有限 HBM pool 与 pinned DRAM backing。
保留的完整 64K + 1K 报告显示 resident/offload 末 token logits 逐位一致；KV gather
对齐修复后的完整模型性能待补测，见 [ECHO 实验](experiments/deepseek_v32_echo_prefill/README.md)。
报告数字保留各自 run ID 与源码快照，目录迁移和回归检查不替代性能复测。

```text
operators/nosa/                  indexer；attention/reference、device_only、offload
operators/deepseek_v32/          indexer；attention 三类；linear / grouped MoE
operators/common/               模型无关的 host record 搬运
layers/                         普通层与 indexer / attention 契约
models/nosa/                    NOSA 结构、权重、位置编码、选择语义与 KV 布局
models/deepseek_v32/             完整 ECHO checkpoint 推理及请求适配
executor/                       通用 prefill / extend 分块执行
cache/                          请求事务、resident / host backing、有限 token pool
serving/                        GR 驱动的本地串行执行
tests/integration/              跨模块正确性测试
experiments/                    各实验的源码、脚本、报告和原始产物
experiments/legacy/deepseek_v32/ 有效历史报告、CPU 重建工具与独立 DeepGEMM 基准
3rdparty/                       共享 CUTLASS、DeepGEMM、DeepJIT 子模块
GR/                             请求内容、热度和调度工具
docs/                           给人的研究状态与下一步任务；agents/ 保存内部执行文档
skills/research-supervisor/      项目内持续维护的 Research Supervisor
```

SM120 扩展、旧 synthetic 模型与相关可执行测量入口已清理。历史报告保留原测量含义，
复现所需的 Git revision 见 [legacy README](experiments/legacy/deepseek_v32/README.md)。

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
python -m models.deepseek_v32.echo_infer --model /preset-models --help
bash experiments/deepseek_v32_echo_prefill/scripts/run.sh --help
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

- [文档分工](docs/README.md)、[项目内 Research Supervisor](skills/research-supervisor/SKILL.md)
- [KDA 组件文档](docs/agents/kda/README.md)
- [模型](models/README.md)、[算子目录与类型](operators/README.md)、[实验](experiments/README.md)
- [共享层](layers/README.md)、[缓存](cache/README.md)、[执行器](executor/README.md)、[serving](serving/README.md)
- [第三方依赖](3rdparty/README.md)、[GR](GR/README.md)、[DeepSeek / SM120 历史资料](experiments/legacy/deepseek_v32/README.md)

Supervisor 可直接使用项目文件，无需安装。在本项目对话中请求：

```text
请读取 skills/research-supervisor/SKILL.md 并运行 Supervisor，
维护 docs/status.md 和简短的 docs/roadmap.md 待办清单，
内部记录放到 docs/agents/research-supervisor/，保留我的修正并更新相关内容。
```
