# NOSA 实现与验收规则

本文件适用于 NOSA 模型及其关联算子、cache、serving 和实验任务。
代码路径均相对仓库根目录；同时遵守[根规则](../../AGENTS.md)、
[NOSA 算子规则](../../operators/nosa/AGENTS.md)和[实验规则](../../experiments/AGENTS.md)。
当前模块能力见 [README](README.md)，固定容量性能与 MFU 验收状态见
[motivation 实验](../../experiments/nosa_motivation/README.md)。

## 普通层与 attention 适配

`models/nosa/normalization.py`、`models/nosa/feed_forward.py` 保存 RMSNorm 与 SwiGLU，
接收显式维度、eps 和 bias，不导入模型配置。FlashInfer 的 residual RMSNorm 路径
会原地覆盖输入，调用者必须持有独立且可覆盖的 activation buffer。gate/up 合并权重
作为实际参数由模型加载器建立，不在 forward 拼接 activation 或维护权重副本。
位置编码及 cos/sin cache 仍归模型的 `rotary.py`。

`DenseMainAttention` 与 `ResidentLayerView` 放在 `models/nosa/attention.py`；dense
适配只读取 resident K/V，并拒绝非空 selection。公共选择与 indexer / main attention
协议放在 `models/attention_contracts.py`，保持轻量，不导入具体模型实现。
cache access 可以描述 resident 或 host 来源与 device append，不能把提前 gather
全部选中 KV 作为公共契约的要求。

## 固定容量、计算图与校验

`nosa_motivation` 使用与 `deepseek_v32_motivation` 相同的 H/A/chunk、用户访问顺序
和 P/NH 配置，调用完整 32 层
NOSA checkpoint。`models/nosa/execution/fixed.py` 提供独立的固定容量入口：HBM-only
按 P 个 history token 做 session LRU；offload 按 NH 准入，host backing 随 session
分配，逐层 P 槽以逻辑页偏移和 session tag 直接映射。当前要求 H<=P，history 与
prefill chunk 按 64 token 对齐，不代表任意容量下的 token LRU。dense 预取下一层
全部历史 miss，sync/async sparse 只取完整 query batch 的稀疏并集 miss，均复用
逐层 pool。候选整批在 GPU 执行并 discard，不写入 host history；CIS/indexer 的
候选尾部仍按实际 session storage 计费。可显式启用纯计算 CUDA Graph，cache、
indexer、attention 与 IO 留在图外；图的 static allocated 和 private reserved
分别计入。图模式可在保留原打分分派的前提下，将 finite 检查推迟到事务提交前：
`count >= 2047` 仍使用原 joint-native 路径；`count < 2047` 时，需要打分的短前缀
通过 guarded 入口执行原 standalone Triton/native 打分，不扩大 joint-native 范围
或改换后端。

逐层 GPU flag 与归约结果计入静态容量；全部检查成功后才提交或返回输出。flag
为 false 时，guarded 打分不得读取打分数据或写入打分输出，native selection 输出安全的
空选择，模型回滚整个 step；offload 只允许提前写未提交的派生尾部。

allocator 校验使用私有 C++ adapter；每次调用都获取新的全 pool snapshot，
保留原有配置、pool 所属关系和 expandable segment 检查，不缓存 snapshot 的检查结果。
adapter 初始化、编译或执行失败时直接报错，不切换至官方 snapshot API。
该固定容量入口与原 NOSA budget
模式分开，完整性能与 MFU 验收状态以实验 README 为准，不以正确性检查代替
性能结果。

NOSA 的 Python MemPool 全代检查可通过 `cache/allocator/pool_referrers.py` 加速；
该 provider 与 allocator snapshot provider 分别记录。私有 GC 遍历须认证实际
CPython 可执行文件、符号来源及完整 header 依赖，保留原动态 getter、subclass 和
GC 变化后的重新扫描。初始化、ABI 认证、审计或执行异常直接传播；不支持的环境也报错，
不自动切换至 Python provider。动态 getter 的正常分派仍保留，不把它当作失败恢复。
不修改 GC 阈值、禁用 GC 或用旧原型验收替代当前路径的 CUDA 与完整请求验收。

## 普通 resident / offload 与共享执行资源

NOSA 默认 resident，显式 offload 使用 `cache/host_backing.py` 与
`models/nosa/cache/offload.py`：pinned local DRAM 保存历史 K/V，CIS 和压缩派生记录
resident；CPU 用于参考测试。普通 owned / budget 路径共享一层完整逻辑地址范围的
HBM staging；serving 的 `models/nosa/execution/resources.py` 由 backend 持有 sparse
workspace、dense 双缓冲和 hbm/dense FA3 scratch，先按 C/A/Q plan/allocate，再创建
独立用户 session。borrowed cache、owner、lease 和关闭流程遵守根规则。
普通 owned / budget 路径尚无有限 HBM slots 或 token 淘汰策略，不能将其表述为通用
HBM caching 已完成；固定 P/NH 的直接映射历史 pool 遵守本文件上一节的独立规则。

## 选块、CIS 与 pattern 语义

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
  offload 另遵循 [NOSA 算子规则](../../operators/nosa/AGENTS.md)中的 native SM90 限制。
  两种 policy 不混用；原 query-aware pattern 实验在 dense 激活上旁路选块，不改变 dense 基线。
  完整 NOSA pattern 对照分别采集同一 dense 激活上的 QA-only/full NOSA 选择，以及真实
  sparse 传播中 attention 实际消费的选择；dense/sparse prefix 从独立空 cache 构建。
