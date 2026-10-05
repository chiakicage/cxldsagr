# 项目约定

本项目研究通用 sparse attention offloading，选取 DeepSeek V3.2 和 NOSA 验证方案。
后续开发优先 NOSA，主要平台为 SM90 / Hopper；SM120 可执行代码已移除，保留有效历史报告。

## 按任务读取规则

以下子规则按任务涉及的模块读取，不能只依赖待修改文件所在目录的作用域。

- 涉及 NOSA 的模型、算子、cache、serving 或实验时，必须读取
  [NOSA 模型规则](models/nosa/AGENTS.md)和[NOSA 算子规则](operators/nosa/AGENTS.md)。
- 涉及 DeepSeek V3.2 的模型、算子、cache、serving 或实验时，必须读取
  [DeepSeek 模型与执行规则](models/deepseek_v32/AGENTS.md)。
- 修改 `cache/`、`executor/`、`models/attention_contracts.py`、`serving/` 或其他共享实现时，必须读取两个
  模型的规则，检查受影响的模型路径和全部依赖实验。
- 涉及实验、`evaluation/`、性能优化或正确性修正时，必须读取[实验规则](experiments/AGENTS.md)
  和受影响实验的 README；模型或算子目录内的改动也遵守此要求。
- 修改第三方版本、依赖布局、native 构建、依赖准备脚本或环境依赖时，必须读取
  [第三方规则](3rdparty/AGENTS.md)及 `3rdparty/README.md`。

子规则记录实现与验收约束；当前能力、运行状态和结果以模块或实验 README 为准。

## 错误处理

错误处理采用 let it crash：执行、初始化、编译、认证或 provider 调用失败时直接向上传播，
不自动重试请求、恢复执行或切换后端。保留事务结束、CUDA drain 和资源安全释放所需的
清理；清理也失败时保留原始异常对象和全部清理异常，不能用后一个异常覆盖前一个。
无法确认异步完成时保留 owner 与 storage，禁用复用。显式选择的 reference 路径和
按支持条件选择的正常分派仍可使用；DeepSeek 精确工作集超过 pool 后拆分 query 消费
属于保留完整选择的调度，不属于失败恢复。

## 容量、请求与缓存事务

当前探索的一种 GR serving 方案是固定 history、变化 candidate 的 prefill 负载，
其代表性及模型、数据适配仍待确定。`GR/` 的固定前缀语义不等于跨请求 KV 复用；
`serving/runner.py` 保留逐请求创建并释放 cache 的旧入口；
`serving/persistent.py` 使用 `cache/prefix_pool.py` 保留跨请求用户历史。
通用预算模式在相同 HBM / CPU DRAM 硬预算下按用户 session 做 LRU；固定 P/NH 模式
按对应的 host 页或 HBM history token 配额准入，两种模式的容量和命中结果不能混用。
DeepSeek `echo/serial_sparse` 的候选成功后 discard 临时状态，保留 history；
候选失败直接报错终止并释放 session，不恢复请求或自动重试。
未接入 transient candidate 入口的后端仍在候选执行后 truncate 到固定历史。
普通模型和直接 `backend.extend` 保留持久追加语义；显式 candidate 入口的行为由模型规则规定。
历史 token 身份变化或保留容量不足时重建。首次访问与复访由用户访问次数区分，
不能将缓存 miss 的复访计成首次访问。模型权重与普通 activation 单独报告，cache 预算
包含索引、映射、staging、cache scratch 与待提交 append，分配前预留并核验实际容量。

HBM 额度评估须区分 PyTorch allocated、reserved 和设备已用量；allocator 缓存仍占用
HBM，不能只因 allocated 未超额就声称满足实际预算。观测到的占用差额、规划时人为
扣除的额度与实现预分配的 storage 分别描述。离线估算与实际完整请求验收分别报告；
模型专用的规划公式和范围见对应子规则。

## 项目内 Research Supervisor

- 用户要求运行 Supervisor、讨论研究进展、修正研究理解或安排研究探索时，
  读取项目内 [skills/research-supervisor/SKILL.md](skills/research-supervisor/SKILL.md)，
  每次使用读取当前版本，不安装或同步到 Codex 用户技能目录。
- 研究状态维护在 [docs/status.md](docs/status.md)，下一步任务在 `docs/roadmap.md`；
  Supervisor 内部依据在 `docs/agents/research-supervisor/`。人可以直接修改状态表，
  后续运行保留修正并更新受影响的判断和任务。
- Supervisor 不生成论文或组会叙事，这些产物由其他 agent 完成。
  项目的探索历程不能充当呈现给读者的研究任务或问题定义。
- `docs/roadmap.md` 保持简短，只列当前待做或进行中的任务及对应研究条目；
  完成或取消后移出清单，研究发现回写状态表，详细计划与必要历史留在 `docs/agents/`。
- 当前研究先关注 HBM 与 CPU DRAM，GR serving 是待完善的候选场景；
  不从项目名称自动加入 CXL/RDMA，也不把已有设计当成场景和 motivation 已成立。
- 普通工程任务按下述约定执行，无需启动完整研究梳理；若结果改变已有研究判断，
  将研究含义同步到对应条目，详细实现和验证仍留在工程/实验材料中。

## 目录与职责

- 自有算子按模型放在 `operators/nosa/`、`operators/deepseek_v32/`，模型内部按
  `indexer/`、`attention/`、`linear/` 等实际功能组织。attention 分为 `reference/`、
  `device_only/`、`offload/`；reference 可独立导入，不加载 Triton 或 native 扩展。
  架构和后端是实现属性，不再作为算子的顶层目录；不为分类复制 kernel。
  ECHO 融合 indexer/prefetch 仍归 indexer，offload attention 复用 device-only MLA。
  真正通用的 record 搬运放 `operators/common/`，宽度和 dtype 由调用者提供；
  共享第三方库放 `3rdparty/`。各功能的单元测试放就近 `tests/`，不新增 `__init__.py`。
  JIT 路径和指纹必须覆盖实际源码及本地 include 依赖，不能跨模型扫描全部算子源码。
- `operators/` 的共享适配只封装现有后端调用。`models/attention_contracts.py` 保存
  `BlockSelection`、`TokenSelection`、`AttentionContext`、`Indexer` 和 `MainAttention`
  的轻量公共契约，不导入具体模型实现。普通层及 dense attention 适配放在对应模型目录；
  NOSA 的 RMSNorm / SwiGLU 接收显式维度、eps 和 bias，不反向导入模型配置。
- 模型结构、权重加载、位置编码、稀疏选择语义、KV 布局适配放在
  `models/deepseek_v32/`、`models/nosa/`。`models/` 只保留模型推理相关代码及其测试；
  接入另一架构时复用对应模型目录，不按架构复制模型树。
  standalone SM90 ECHO 放在 `models/deepseek_v32/`，不依赖 SGLang 或 SM120 扩展。
- `executor/` 负责通用模型分块执行和输出选择，不读取 GR 请求；`serving/` 负责 GR
  请求适配、串行请求生命周期及跨请求固定历史复用，不包含网络服务或 CXL/RDMA 依赖。

- `cache/` 管理请求级缓存分配、逐层写入、提交、重置和释放。模型提供 KV 布局和
  兼容信息；所有模型层成功执行后统一推进有效长度。`CacheManager` 接收模型 allocator。
  共享资源由 backend/model 持有，先 plan/allocate，再创建独立用户 session；borrowed
  cache 的写入、提交和回滚必须在执行 lease 内，释放 session 不释放 shared buffers。
  普通模型 owned 模式仍可独立调用。公共 runner 从构造到关闭绑定唯一准入 owner，
  构造失败仅回滚本次新资源；无法确认异步完成时保留 owner 并禁用复用。runner 关闭
  全部 session 后解绑，最外层负责 `backend.close()`。正确性和分配验收不能替代完整
  serving 性能或未运行的容量轨迹。模型专用 storage、容量和候选策略见对应子规则。
  `cache/indexer_cache.py` 管理请求级派生 record 与共享 scratch；压缩和稳定 pool 的
  语义由模型声明。派生缓存随 KV 统一提交、回滚和截短，不占用通用 opaque layer state。

- main attention 接收逻辑块选择、cache access 与执行上下文，不能把「全部 KV
  已完成搬入 HBM」作为通用前置条件。模型与算子的实现、支持条件和验收要求见子规则。

- `tests/` 只保存保证代码正确性的跨模块集成测试代码，`scripts/run_tests.sh`
  负责全局回归编排；模块单元测试仍留在对应模块的 `tests/`。测试与论文实验的目的不同，
  smoke、数值正确性和回归检查不能作为 `experiments/` 的实验或结果。
- `experiments/` 按实验独立目录维护，具体布局见[实验规则](experiments/AGENTS.md)。DeepSeek 专用 GR checkpoint /
  indexer 等旧实验保存在 `local/experiments/legacy/deepseek_v32/`，CPU DRAM 带宽实验
  保存在 `local/experiments/cpu_dram_bandwidth/`。整个 `local/` 不进 Git，当前代码不得依赖
  这些本地副本；不把旧实验移入 `models/`。
- `GR/` 保存共享请求内容、用户热度和调度工具。
- 本地权重与 tokenizer 放在 `weights/`（已被 `.gitignore` 排除），例如
  `weights/DeepSeek-V3.2/`；与代码目录 `models/` 分离。
- 不将 DeepSeek V3.2 的 656 B packed MLA record、indexer 或 tokenizer 作为通用
  offloading / NOSA 的固定假设；新增模型时显式适配其稀疏访问语义和 KV 布局。
- 修改目录时同步更新 Python 导入、`-m` 子进程入口、构建路径、配置和文档链接。
  保留 `weights/` 和 GR 共享资源；符合实验目的的有效运行产物迁入对应实验的 `output/`，
  报告需要呈现的图表和数据保存在该实验的 `report/`。
  `docs/` 保留跨实验设计文档，有效的旧 DeepSeek / SM120 报告随本地 legacy 副本保存。

## 实验与结果变更

实验目录、独立验收/计时/profile、产物和发布要求完整保存在
[实验规则](experiments/AGENTS.md)。测试和静态检查不能代替实验结果。
性能优化或正确性修正后，必须先完成受影响路径的正确性验收、正式补测和测量完整性
检查，以新 run ID 发布可核验结果，再替换 README 和清理受影响的旧报告及运行产物。
新结果发布前保留原结果并说明实现版本、测量边界和补测状态；受已知正确性问题影响的
范围须明确标注。共享模块变更须检查全部依赖实验，未受影响的有效结果可保留。
失败或已撤回结果的清理、合理对照的保留，以及所有具体发布细则仍按实验规则执行。

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
  安装入口或打包配置。模型、实验、算子及 `executor/`、`cache/`、
  `serving/`、`tests/` 保持源码目录，不为目录整理添加 `__init__.py`；算子子项目保留自己的
  扩展包及导入名称。
- 实验命令默认从仓库根目录运行。模型直接脚本入口保持可用；实验统一使用
  [实验规则](experiments/AGENTS.md)中的 `python -m experiments.<experiment>.src.<module>`，
  不在旧目录遗留兼容壳文件。
  旧命令只在历史运行记录中保留，无需安装仓库。
- 用 `pyproject.toml` 和 `uv.lock` 管理环境，依赖改动同步维护两者。基础环境用于
  Hopper；官方 DeepGEMM / FlashMLA 进入基础环境，`legacy` 组名保留给历史命令，
  不恢复 SM120 扩展或安装组。

## 第三方依赖

详细版本、上游分支、共享 include 布局和禁止事项见[第三方规则](3rdparty/AGENTS.md)。
使用 `python3 scripts/prepare_3rdparty.py --init` 准备顶层依赖，再执行 `uv sync`；
不要递归初始化子模块，不将可选本地参考 checkout 视为已接入的后端。

## 文档与验证

- 新写或修改的中文内容在交付前须使用 `humanizer-zh` 润色，包括文档、报告、说明和
  注释。润色须保留事实、数值、公式、术语、来源与验证边界，不改变结论的确定程度；
  代码、命令、路径、显式 ID、结构化数据及须原样保留的历史记录不作文字改写。
- `docs/` 中面向人类的文档应讲清研究理解、当前问题、计划及必要的技术分析。
  `docs/status.md`、`docs/roadmap.md` 由 Research Supervisor 维护，
  保留研究者直接修正；职责与阅读入口见 [docs/README.md](docs/README.md)。
- agent 的任务契约、实现计划、详细验收、交接、证据索引和内部状态统一放在 `docs/agents/`。
  KDA 文档按实现组件放在 `docs/agents/kda/<component>/`，使用 `task.md`、
  `implementation_plan.md`、`checkpoint.md`、`investigation_log.md` 等明确用途的文件名；
  当前计划与历史候选分开，保留 run ID、源码身份和验证边界，执行材料可使用英文。
  系统工程材料放在 `docs/agents/system/`，Supervisor 内部记录放在
  `docs/agents/research-supervisor/`；不再在 `docs/` 根目录放 `draft.md`、`plan.md` 等执行文档。
  执行 agent 维护自己的内部材料，将研究含义返回给 Supervisor，由其更新研究状态和下一步任务。
- `docs/agents/system/` 是在办工程任务的工作区。任务完成或被重构替代后，将仍有效的
  约束写入对应 `AGENTS.md`，当前设计写入模块 README，必要验收依据保留在相应正式
  报告或独立验收材料中；更新引用后删除旧计划、草稿、重复检查点、交接及失效 patch。
  不通过改放 `history/`、`legacy/` 或新建历史汇总继续保留这些过程文档，历史由 Git 追溯。
  未完成的独立任务只保留必要材料；没有在办任务时可以删除整个 `system/` 目录。
  重构计划本身在执行验收完成后同样删除。此规则不改变有效实验报告及证据的替换要求。
- 具体实验的报告、数据和源码仍按 `experiments/` 约定维护，模块说明留在模块 README。
  文档分层不复制或重命名实验运行产物；迁移文档时更新导航链接，保留历史命令和测量含义。
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
