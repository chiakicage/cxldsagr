# 项目约定

本项目研究通用 sparse attention offloading，选取 DeepSeek V3.2 和 NOSA 验证方案。
后续开发优先 NOSA，主要平台为 SM90 / Hopper；保留现有 DeepSeek V3.2 / SM120 实验。

## 目录与职责

- 自有硬件算子放在 `operators/sm90/`、`operators/sm120/`；共享第三方库放在 `3rdparty/`。
- `operators/` 的共享适配只封装现有后端调用；`layers/` 保存共享普通层及 indexer /
  main attention 契约。共享层接收显式维度和参数，不反向导入模型配置。
- 模型结构、权重加载、位置编码、稀疏选择语义、KV 布局适配放在
  `models/deepseek_v32/`、`models/nosa/`。`models/` 只保留模型推理相关代码及其测试；
  接入另一架构时复用对应模型目录，不按架构复制模型树。此次分层仅接入 NOSA，
  DeepSeek V3.2 / SM120 保留既有组织。
- `executor/` 负责通用模型分块执行和输出选择，不读取 GR 请求；`serving/` 负责 GR
  请求适配及串行请求生命周期，不包含网络服务或 CXL/RDMA 依赖。
- `cache/` 管理请求级缓存分配、逐层写入、提交、重置和释放。模型提供 KV 布局和
  兼容信息；所有模型层成功执行后统一推进有效长度。当前后端仅为模型设备上的
  resident cache，CPU 用于参考测试；local DRAM backing 与 HBM caching 尚未实现。
  `cache/indexer_cache.py` 管理请求级派生 record 与共享 scratch；压缩和稳定 pool 的
  语义由模型声明。派生缓存随 KV 统一提交、回滚和截短，不占用通用 opaque layer state。
- main attention 接收逻辑块选择、cache access 与执行上下文，不能把「全部 KV
  已完成搬入 HBM」作为通用前置条件。未来 fetch/compute overlap 由 SM90 算子实现；
  resident NOSA block sparse attention 已接入 SM90 Triton；offload 入口仍只预留接口，
  调用未实现路径须明确失败，不将 resident 验证表述为 offload 验证。
- NOSA query-aware indexer 已有 resident K 上的 PyTorch FP32 参考实现：64-token block、
  默认 `block_budget=64`，1 sink + 16 causal local（含当前块）+ 47 query-aware top-k；
  支持 `block_budget=32`，保持 1 sink + 16 local，query-aware top-k 改为 15。
  返回逻辑选择形状为 `[query, KV head, block_budget]`，短上下文的不足位置使用 validity mask。
  采用 RoPE 后 Q/K、32-token / stride-16 mean compression、逐 Q head softmax、GQA
  求和与五窗口 max pooling；此默认分析模式不启用 query-agnostic / CIS。
  显式 `attention_mode="sparse"` 复刻 cxl-recsys 的完整 NOSA：64-token / 64-block，
  inclusive local 为当前块加前 16 块，query-aware 阶段含 sink/local 共保留 33 块，
  再按 query-agnostic CIS 补满 64 块。A/delta 从 checkpoint 严格加载；
  `softplus(delta(V)) * A` 同时用于压缩后选块及 attention 加性 bias。
  K/V/CIS 作为同一 resident cache step 提交，CPU reference 与 SM90 Triton 均可运行。
  两种 policy 不混用；原 query-aware pattern 实验在 dense 激活上旁路选块，不改变 dense 基线。
  完整 NOSA pattern 对照分别采集同一 dense 激活上的 QA-only/full NOSA 选择，以及真实
  sparse 传播中 attention 实际消费的选择；dense/sparse prefix 从独立空 cache 构建。
- `tests/` 只保存保证代码正确性的跨模块集成测试代码，`scripts/run_tests.sh`
  负责全局回归编排；模块单元测试仍留在对应模块的 `tests/`。测试与论文实验的目的不同，
  smoke、数值正确性和回归检查不能作为 `experiments/` 的实验或结果。
- `experiments/` 按实验独立目录维护，具体布局见下节。DeepSeek 专用 GR checkpoint /
  indexer 等旧实验整体归档在 `experiments/legacy/deepseek_v32/`，不混入 `models/`。
- `GR/` 保存共享请求内容、用户热度和调度工具。
- 本地权重与 tokenizer 放在 `weights/`（已被 `.gitignore` 排除），例如
  `weights/DeepSeek-V3.2/`；与代码目录 `models/` 分离。
- 不将 DeepSeek V3.2 的 656 B packed MLA record、indexer 或 tokenizer 作为通用
  offloading / NOSA 的固定假设；新增模型时显式适配其稀疏访问语义和 KV 布局。
- 修改目录时同步更新 Python 导入、`-m` 子进程入口、构建路径、配置和文档链接。
  保留 `weights/` 和 GR 共享资源；符合实验目的的有效运行产物迁入对应实验的 `output/`，
  报告需要呈现的图表和数据保存在该实验的 `report/`。
  `docs/` 保留跨实验设计文档，有效的旧 DeepSeek / SM120 报告随 legacy 归档。

## 实验目录与产物

- `experiments/` 是服务论文的实验交付产物，只保留符合明确实验目的、具有正确实现与
  测量语义的结果。实现修正后，被判定不合理的旧实现（如已替换的未融合基线）、错误语义、
  失败运行和已撤回的结果必须删除，不能以历史记录或 `legacy/` 归档为由保留在实验目录。
  不能仅因性能较差而删除合理的对照组；有效比较必须由实验目的明确支持。
  正确性检查只负责验证代码，不因使用 GPU、计时或 profiler 就成为论文实验。
- 凡涉及性能优化或正确性修正的改动，必须在同一改动中清理 `experiments/` 内受影响的
  旧报告内容，包括 README 中的结果、结论和性能数字、`report/` 图表与数据，以及
  `output/` 中对应的旧运行产物；不得以历史记录、优化前对照、标注过期或 `legacy/`
  归档为由保留。需要对照的实现应按当前正确性与测量要求重新运行，生成新的有效结果。
  新报告必须基于改动后的实现重新测量生成，注明新的 run ID；尚未重跑时明确写
  “改动后未运行”，删除旧结果及其引用，不沿用旧数字或仅修改文字包装为新结果。
  清理范围按改动影响确定，共享模块变更须检查所有依赖实验；未受影响的有效实验可保留。
- 每个实验使用一个有明确目的的目录 `experiments/<experiment>/`，禁止继续在
  `experiments/` 根目录堆放测量脚本、报告或 JSON/CSV。根 `README.md` 只维护实验入口索引。
- 每个实验必须有 `README.md`，说明实验目的、实验内容与测量边界、运行方式及调用模块、
  实验结果和结论；未运行的实验明确写未运行，不把路径迁移或静态检查当成新的测量结果。
  记录硬件、依赖、输入形状、精度、预热/重复次数、指标定义和已有数据的 run ID。
- 使用以下统一结构，不为目录整理新增 `__init__.py` 或根包：

  ```text
  experiments/<experiment>/
    README.md
    src/                 实验的 measure / profile / analyze / report 等 Python 代码
    scripts/             可复现的运行脚本，负责参数组合、环境和日志重定向
    tests/               本实验的单元测试
    report/              报告需要呈现的图片、表格及数据，随 Git 保存
    output/              默认全部不进 Git，由运行脚本自动创建
      log/<run_id>/      stdout 日志；stderr 使用独立文件记录
      data/<run_id>/     JSON/JSONL/CSV/NPY、整理后的表格、分析结果、导出的 SQLite 等
      profile/<run_id>/  nsys/nsys-rep、ncu、Chrome trace 等原始 profiler 产物
  ```

- 运行生成的完整数据和分析结果放 `output/data/`，包括汇总 CSV/JSON；不放回源码目录。
  源码快照、运行参数和迁移索引等复现数据也放 `output/data/`，原始日志和 profiler
  产物分别放 `output/log/`、`output/profile/`，默认不提交。
- `report/` 保留从有效实验结果中选出的、需要在报告里呈现的图片、表格和数据
  （例如 PNG/SVG、CSV/JSON），随 Git 保存。README 保留结果说明和结论，通过相对路径
  嵌入或链接 `report/` 文件；报告所需的图像应放 `report/`，不为它们放开 `output/` 的忽略规则。
  在 README 中注明这些素材的来源 run ID 和生成方式；可从对应 `output/` 复制选定素材，
  保留仍然有效的原运行产物以便复现；性能优化或正确性修正后按上述规则清理受影响的旧产物。
  仍被忽略的产物使用普通代码路径，不创建仓库文档链接。
- `src/` 不实现 shell 调度；`scripts/` 只编排 `python -m ...` / nsys 等入口，不复制模型
  计算逻辑。模型推理代码及其测试仍归 `models/`，共享 GR 生成器及热度曲线仍归 `GR/`。
  复用其他实验的工具时使用显式导入，并在 README 的调用模块中写明依赖。
- 默认从仓库根目录执行，Python 实验入口为
  `python -m experiments.<experiment>.src.<module>`；运行脚本应能定位仓库根目录，
  支持 `--help`，创建分类输出目录并保留子进程失败状态。不同运行用不同 run ID，
  不覆盖仍然有效的旧结果；受性能优化或正确性修正影响的旧结果按上述规则删除。
- 旧 DeepSeek 及配套 SM120 实验集中保存在 `experiments/legacy/deepseek_v32/`，
  作为整体归档，不强行套用新目录层级或拆分历史数据；仅修复导入、入口和导航链接。
  仅保留符合其目的的有效实验；这不豁免上述结果有效性要求。保留报告中的原命令与
  原测量含义，当前运行方式写在归档 README。

## 全局测试

- 跨模块集成检查放在 `tests/integration/`。`tests/` 及各模块测试目录只保存测试代码和
  必要的输入 fixture，禁止添加 README、文档、reports、运行结果、日志或 profiler 产物。
  已有的测试报告与运行结果直接删除，不迁移到其他目录伪装为实验或历史交付物。
- 全局入口为 `bash scripts/run_tests.sh [cpu|gpu|all]`，默认运行 CPU 回归。
  脚本定位仓库根目录、支持 `--help`、保留子进程失败状态，不复制模型计算逻辑。
  根 README 只写简洁运行方式；测试结果直接输出终端，不创建 run ID 或持久结果目录。
- 临时文件使用 pytest `tmp_path` 或系统临时目录。GPU 检查要求可用硬件和依赖；显式选择
  GPU 时缺少它们须失败，不能把跳过当成通过。不在集成入口重复调用已收集的数值测试。

## 脚本与环境

- 暂时不把 cxldsagr 做成 Python 包：保留 `[tool.uv] package = false`，不新增根包、
  安装入口或打包配置。模型、实验、算子及 `layers/`、`executor/`、`cache/`、
  `serving/`、`tests/` 保持源码目录，不为目录整理添加 `__init__.py`；算子子项目保留自己的
  扩展包及导入名称。
- 实验命令默认从仓库根目录运行。模型直接脚本入口保持可用；实验整理后统一使用上节的新
  模块入口，不在旧目录遗留兼容壳文件。旧命令只在历史运行记录中保留，无需安装仓库。
- 用 `pyproject.toml` 和 `uv.lock` 管理环境，依赖改动同步维护两者。当前 SM120 扩展
  放入显式 `sm120` 依赖组，基础环境不默认安装；未接入后端前不虚设 SM90 安装组。

## 第三方依赖

- `3rdparty/DeepGEMM` 使用上游 `nv_dev` 分支的 Git 子模块，固定提交由父仓库记录。
  当前为 `b64107f`（2.8.0）；不再把整个 DeepGEMM 源码复制进架构算子目录。
- 共享依赖为 `3rdparty/cutlass/`（`f3fde583`）与 `3rdparty/DeepJIT/`（`e5bdee2`），
  与 DeepGEMM 一起作为顶层三个子模块维护；不要新增嵌套的重复源码副本。
- 使用 `python3 scripts/prepare_3rdparty.py --init` 准备依赖，再执行 `uv sync`。
  该脚本只初始化顶层子模块，在 DeepGEMM 的 `third-party/cutlass/` 和
  `third-party/deep_jit/` 下将 `include` 链接到顶层共享源码，并关闭嵌套子模块初始化，
  不修改上游版本化源码。避免 `git clone --recursive` 或递归更新子模块；已递归初始化
  的 checkout 会被准备脚本拒绝，需先按诊断处理，不能与共享链接布局混用。
- `3rdparty/EzKernelKit/` 仅保留为本地未跟踪参考，不纳入当前依赖。其 CUTLASS
  提交与 DeepGEMM 不同，未经适配验证不能强行合并依赖。
- 第三方依赖的初始化、链接和构建遵循 `3rdparty/README.md` 及现有配置；修改版本、
  分支或布局时同步更新子模块和相关引用。不要将可选本地参考 checkout 的存在
  视为已经接入项目的后端。

## 文档与验证

- 给 agent 的目录维护、实现职责、依赖管理和后续开发约定写在 `AGENTS.md`。
  README 保留项目介绍、实际状态、入口索引和可用命令，避免重复维护约定。
- 保留有效实验报告的平台、依赖和测量含义；目录迁移或上游源码支持某架构不等于
  已完成 GPU 验证。明确区分已运行的验证、静态检查和未验证路径。
- 保留的有效测量报告里的命令行和依赖路径（例如已移出仓库的 `gpu-benches/`）保留原样，
  它们记录当时的运行方式；只修正导航性链接和「当前文件在哪」的指向。
  被 `.gitignore` 排除的运行产物（如实验 `output/data/`、`output/profile/` 下的文件）
  在文档里写成普通代码路径，不做链接；随 Git 保存的 `report/` 图表和数据使用相对链接或图片嵌入。
- 按改动选择验证：路径迁移检查导入、子进程入口和文档链接，并运行可用的 CLI /
  现有测试；算子或数值改动运行相应硬件和正确性测试。缺少依赖或 GPU 时如实记录。
- Python 格式遵循根 `pyproject.toml` 的 Ruff 配置；第三方和独立算子子项目遵循
  自己的格式与构建设置，避免无关的批量格式化。
