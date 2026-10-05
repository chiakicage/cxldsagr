# 模型执行器

[contracts.py](contracts.py) 定义请求形状、执行上限、输出规格和模型 driver 接口。
[runtime.py](runtime.py) 的 `TokenRuntime` 校验资源身份、session 状态、输入和输出，
调用模型显式提供的 `prefill`、`candidate`、`append` 与释放操作。
DeepSeek 与 NOSA 在模型目录装配 driver；用户身份、热度、预算准入和 LRU 由
serving/cache 负责，执行器不读取 GR 请求。
模型内部的 indexer / main attention 协议见
[attention_contracts.py](../models/attention_contracts.py)，具体普通层和 attention 适配由模型持有。

[adapters.py](adapters.py) 保留模型原有计算循环、事务和同步边界。候选策略在创建
driver 时声明，报告 metadata 只反映该配置；runner 不再通过探测可选方法来决定行为。
同一份 `SessionPlan` 同时用于准入预留和实际分配，HBM/DRAM 字节、host pages 与
retained-HBM tokens 分别计费。容量类型见 [cache](../cache/capacity.py)。

`candidate` 成功后保留已提交 history；固定容量路径使用 GPU 临时状态并 discard，
普通 NOSA budget 与 DeepSeek budget 的 `hbm/dense_prefetch` 保留原
`extend + truncate` 行为，DeepSeek budget 的 `echo/serial_sparse` 使用 GPU transient
candidate。`append` 持久提交新增 token。
`OutputSpec` 明确 last/all hidden、logits、LM head 工作量和输出所有权；DeepSeek serving
继续执行末 token LM head，NOSA serving 继续只计算 hidden。候选阶段的 `executed` /
`cleaned` 观测点分别位于执行结束与清理结束，不改变正式计时边界。

共享资源扩展通过纯 `plan_resources` 预留，`allocate_shared` 分配一次。
runner 在分配前 `bind_owner`，关闭全部 session 后 `unbind_owner`，共享资源由外层
`backend.close()` 释放。构造回滚只释放本次绑定后新建的资源；已有相同 plan 保留。
无法确认异步完成时保留 owner 并禁用复用，避免第二个 pool 重复占用同一预算。
执行异常直接传播；必要清理也失败时保留原始异常对象及全部清理异常，不自动恢复、
重试请求或切换 provider。

[model_executor.py](model_executor.py) 提供与 GR 请求格式无关的 token tensor 执行接口。
`ModelExecutor(model, cache_manager=None, chunk_size=1024)` 负责分配/释放 session，
并用共享分块循环执行 `prefill` 和 `extend`。prefill 要求空 cache，extend 要求已有前缀；
单 token decode 也使用 extend。

分块前校验整段输入是否超过 cache 容量或模型上下文；这类错误不提交任何 chunk。
执行中某一 chunk 失败时，由模型回滚该次 forward，之前成功完成的 chunk 仍有效。

`run_chunks(model, ids, cache, chunk_size, output="hidden", logits_to_keep=0)` 默认跳过
LM head，返回最后一个 chunk 的全部 normalized hidden；不拼接整条序列的输出。
`output="logits"` 用于现有文本生成前向，采样和 EOS 处理保留在 NOSA `infer.py`。

模型负责参数树、逐层前向和一次 forward 的缓存提交；执行器不读取 GR 字典。
请求的 prefix/candidate 分界由 [serving](../serving/README.md) 提取，
性能计时和 full-vs-split 对照保留在 [experiments](../experiments/README.md)。
