# 项目约定

本项目研究通用 sparse attention offloading，选取 DeepSeek V3.2 和 NOSA 验证方案。
后续开发优先 NOSA，主要平台为 SM90 / Hopper；SM120 可执行代码已移除，保留有效历史报告。

当前探索的一种 GR serving 方案是固定 history、变化 candidate 的 prefill 负载，
其代表性及模型、数据适配仍待确定。`GR/` 的固定前缀语义不等于跨请求 KV 复用；
当前 `serving/` 仍逐请求创建并释放 cache。

## 项目内 Research Supervisor

- 用户要求运行 Supervisor、讨论研究进展、修正研究理解、安排研究探索或准备研究汇报时，
  读取项目内 [skills/research-supervisor/SKILL.md](skills/research-supervisor/SKILL.md)，
  每次使用读取当前版本，不安装或同步到 Codex 用户技能目录。
- 研究状态维护在 [docs/status.md](docs/status.md)，四环节叙事在 `docs/research.md`，
  研究任务与推进建议在 `docs/roadmap.md`；Supervisor 内部依据在 `docs/agents/research-supervisor/`。
  人可以直接修改状态表；后续运行保留修正并更新受影响的叙事和任务。
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
- `operators/` 的共享适配只封装现有后端调用；`layers/` 保存共享普通层及 indexer /
  main attention 契约。共享层接收显式维度和参数，不反向导入模型配置。
- 模型结构、权重加载、位置编码、稀疏选择语义、KV 布局适配放在
  `models/deepseek_v32/`、`models/nosa/`。`models/` 只保留模型推理相关代码及其测试；
  接入另一架构时复用对应模型目录，不按架构复制模型树。共享分层当前接入 NOSA；
  standalone SM90 ECHO 放在 `models/deepseek_v32/`，不依赖 SGLang 或 SM120 扩展。
- `executor/` 负责通用模型分块执行和输出选择，不读取 GR 请求；`serving/` 负责 GR
  请求适配及串行请求生命周期，不包含网络服务或 CXL/RDMA 依赖。
- `cache/` 管理请求级缓存分配、逐层写入、提交、重置和释放。模型提供 KV 布局和
  兼容信息；所有模型层成功执行后统一推进有效长度。`CacheManager` 接收模型 allocator，
  NOSA 默认 resident，显式 offload 使用 `cache/host_backing.py` 与
  `models/nosa/offload_cache.py`：pinned local DRAM 保存历史 K/V，CIS 和压缩派生记录
  resident；CPU 用于参考测试。NOSA 共享一层完整逻辑地址范围的 HBM staging，
  尚无有限 slots 或淘汰策略，不将该实现表述为通用 HBM caching 已完成。
  独立 `cache/sparse_token_cache.py` 为 DeepSeek SM90 ECHO
  提供 pinned local DRAM backing、有限 HBM slots、精确 recall 与缓存事务；record
  宽度和 dtype 由模型提供，不能将其有限 HBM pool 能力归于 NOSA。
  `cache/indexer_cache.py` 管理请求级派生 record 与共享 scratch；压缩和稳定 pool 的
  语义由模型声明。派生缓存随 KV 统一提交、回滚和截短，不占用通用 opaque layer state。
- main attention 接收逻辑块选择、cache access 与执行上下文，不能把「全部 KV
  已完成搬入 HBM」作为通用前置条件。resident NOSA block sparse attention 已接入
  SM90 CUDA/CuTe 与 Triton；显式 NOSA offload 通过 `operators/nosa/attention/offload/api.py`
  实现 BF16 / D128 / GQA16 native attention 与稀疏 fetch。当前融合版本在一个
  cooperative CUDA 主 kernel 内保留所有 CTA 的 persistent FA3 计算；默认最多
  96 个 CTA 使用 producer warpgroup 的 warp 1–3（96 线程）读取 host，warp 0
  保留 TMA，两个 consumer warpgroup 保留 attention。每次层调用按
  `(KV head, logical block)` 去重并压成唯一页队列，每个 64-token 页拆为 8 个
  不交叠的 8-token stripe；leader 原子领取 `(page, stripe)`，经 shared slot 和
  96-thread barrier 广播，每个历史向量只执行一次 `.cv` host load。每 stripe
  writer 完成 fence/barrier 后，leader 以 acq_rel RMW 累计 ready；跨 CTA 完成链
  达到 ready=8 后，TMA acquire 并执行 async-proxy fence 再读取 HBM。空尾 stripe
  不读 host、仍参与完成；最后完成者只累计一次整页字节。
  仅两 KV heads 且 `ceil(queries / 8) * KV_heads == 256` 时，fetch 与 compute
  都按 head 1 → head 0；head 内保留 block 0 优先/其余 block 降序的 fetch 顺序
  和原 compute cost/tie 顺序。其他几何保留 block-major fetch 与原 attention 调度。
  串行与融合共用新 native initialization，合并全容量 metadata reset、历史
  page-0 padding 和 strided suffix staging；first-use planning 保留独立依赖
  launch。初始化、planning、compaction、prepare / repair 与 launch gaps 全部计时。
  不依赖 host memory 的 L2 复用，不得按八-query group 重复搬运；保留原 selection、
  CIS、causal mask 与 numerical repair。prepare / repair helper 全部计入算子时间，
  cooperative launch 与 occupancy 检查须保证全部 CTA 可同时驻留；producer /
  consumer 的 24 / 240 动态寄存器预算须满足本 CTA 的 64512-register pool，
  不能只按整个 SM 的寄存器上限检查，避免 `setmaxnreg` 等待死锁。
  `query_tile_size` 仅分组统计首次读取流量，不再拆分 attention；`fetch_ctas` 控制
  参与 fetch 的 attention CTA 数上限，不划出专用 fetch CTA。`overlap=False`
  一次 fetch 完整稀疏并集，再执行原 FA3 整批 attention。
  事务提交及 staging 复用须等待相关异步操作完成。CUDA Graph capture、有限 HBM
  slots / eviction 与 CXL/RDMA 尚未支持或验证；不支持路径明确失败，不将 resident
  检查、算子回放或 CPU reference 表述为完整模型 offload 性能验证。
  NOSA 完整 checkpoint 数值验收须为 resident/offload 分别从独立空 cache 构建
  sparse prefix，比较全部 extend hidden；cache 分配统计不等于进程峰值显存。
  NOSA overlap 性能对照须包括完整 query batch 的稀疏并集一次 fetch 后计算，
  以此为整体延迟验收门槛；若候选拆分 query tiles，另加相同拆分的串行调度对照。
  单 kernel 的完整执行窗口不能同时充当 fetch 与 attention 的区间；须使用 kernel
  内部实际工作区间证明重叠，且执行窗口相交不能替代整体延迟收益判断。
  stripe 路径同时报告 page envelope 与非空 stripe-copy window 两套指标；envelope
  须等于本页全部非空 stripe 的 min(start)/max(end)，不能以其空隙充当真实 copy。
  90% 验收要求每个 profiled sample 的两种 ratio 都 >= 0.9，不能只检查中位数。
  每轮实现更新须重新验收正确性、唯一读取和内部 overlap，并以新 run ID 发布受
  影响的性能结果；旧结果按下述实验规则保留至替换完成，不以旧验证冒充新实现结果。
- DeepSeek SM90 ECHO 使用完整 checkpoint 的 61 层、embedding、dense / grouped MoE、
  final norm 与 LM head；按 token chunk 依次执行全部层，限制临时 hidden 显存。主 KV
  使用 BF16 512 latent + 64 RoPE record，indexer FP8 K/scales 仍 resident。融合
  indexer prefetch 后必须执行精确 top-k / residual recall；工作集超过 HBM pool 时
  拆分 query 消费，不裁剪每 query 的精确选择。全部层和 GPU 同步成功后统一提交；
  失败只回滚本次启动的事务。完整 resident/offload 对照从独立空 cache 构建 prefix，
  每次 extend 恢复相同 prefix HBM residency；权重加载、编译和状态恢复不计入执行时间。
  layer 0 / layer 3 或单算子正确性检查不替代完整 64K + 1K 的性能测量。
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
  K/V/CIS 作为同一 cache step 提交；resident 支持 CPU reference 与 SM90 CUDA/Triton，
  offload 另遵循上述 native SM90 限制。
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

- `experiments/` 是服务论文的实验交付产物，只有符合明确实验目的、具有正确实现与
  测量语义的结果才能作为有效交付。被判定不合理的实现（如已替换的未融合基线）、错误语义、
  失败运行和已撤回的结果不能作为有效结果或对照。失败运行和已撤回结果须从实验目录删除；
  因优化或正确性修正需要替换的旧报告按下一条先补测再清理，不能以历史记录或 `legacy/`
  归档为由长期保留。
  不能仅因性能较差而删除合理的对照组；有效比较必须由实验目的明确支持。
  正确性检查只负责验证代码，不因使用 GPU、计时或 profiler 就成为论文实验。
- 凡涉及性能优化或正确性修正的改动，必须先用改动后的实现补测受影响的实验，完成
  正确性验收与测量完整性检查，生成注明新 run ID、可核验的新报告，再替换和清理旧结果。
  新结果完成验收并发布前，保留原 README 中的报告、`report/` 图表与数据，以及
  `output/` 中对应的运行产物，不提前清空。README 须保留旧 run ID、实现版本与测量边界，
  并明确标注新实现“改动后未运行”或补测状态；已确认受正确性问题影响的旧结果还须说明
  问题及不可用于结论的范围。旧数字不得冒充新实现结果，也不能仅修改文字包装为新结果。
  新结果可核验发布后，在同次报告更新中替换 README 的结果、结论和性能数字，并清理
  受影响的旧 `report/` 素材及 `output/` 运行产物。需要对照的实现应按当前正确性与测量
  要求重新运行，生成新的有效结果，不以历史记录、优化前对照、标注过期或 `legacy/`
  归档为由长期保留已被替换的旧产物。
  补测与清理范围按改动影响确定，共享模块变更须检查所有依赖实验；未受影响的有效实验可保留。
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
  保留仍然有效的原运行产物以便复现；性能优化或正确性修正后，待新结果验收并发布，
  再按上述规则清理受影响的旧产物。
  仍被忽略的产物使用普通代码路径，不创建仓库文档链接。
- `src/` 不实现 shell 调度；`scripts/` 只编排 `python -m ...` / nsys 等入口，不复制模型
  计算逻辑。模型推理代码及其测试仍归 `models/`，共享 GR 生成器及热度曲线仍归 `GR/`。
  复用其他实验的工具时使用显式导入，并在 README 的调用模块中写明依赖。
- 默认从仓库根目录执行，Python 实验入口为
  `python -m experiments.<experiment>.src.<module>`；运行脚本应能定位仓库根目录，
  支持 `--help`，创建分类输出目录并保留子进程失败状态。不同运行用不同 run ID，
  不覆盖仍然有效的旧结果；受性能优化或正确性修正影响的旧结果保留至新结果验收并发布，
  再按上述规则替换和删除。
- 旧 DeepSeek / SM120 的有效报告集中保存在 `experiments/legacy/deepseek_v32/`。
  已删除依赖 SM120 的模型、算子及测量入口；历史原命令与测量含义保留，复现使用
  整理前 Git revision，见归档 README。独立 DeepGEMM 基准和 CPU 报告重建工具可保留，
  不得再导入已删除的模型/测量脚本；这不豁免上述结果有效性要求。

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
- 用 `pyproject.toml` 和 `uv.lock` 管理环境，依赖改动同步维护两者。基础环境用于
  Hopper；独立历史 DeepGEMM 基准使用可选 `legacy` 依赖组，不恢复 SM120 扩展或安装组。

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

- `docs/` 中面向人类的文档应讲清研究理解、当前问题、计划及必要的技术分析。
  `docs/research.md`、`docs/status.md`、`docs/roadmap.md` 由 Research Supervisor 维护，
  保留研究者直接修正；职责与阅读入口见 [docs/README.md](docs/README.md)。
- agent 的任务契约、实现计划、详细验收、交接、证据索引和内部状态统一放在 `docs/agents/`。
  KDA 文档按实现组件放在 `docs/agents/kda/<component>/`，使用 `task.md`、
  `implementation_plan.md`、`checkpoint.md`、`investigation_log.md` 等明确用途的文件名；
  当前计划与历史候选分开，保留 run ID、源码身份和验证边界，执行材料可使用英文。
  系统工程材料放在 `docs/agents/system/`，Supervisor 内部记录放在
  `docs/agents/research-supervisor/`；不再在 `docs/` 根目录放 `draft.md`、`plan.md` 等执行文档。
  执行 agent 维护自己的内部材料，将研究含义返回给 Supervisor，由其同步人类状态和叙事。
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
