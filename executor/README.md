# 模型执行器

[serving_backend.py](serving_backend.py) 定义跨请求 serving 的 token-only 模型适配契约：
session 容量估计、分配、prefix 构建、完整 candidate hidden 输出、truncate 和同步释放。
DeepSeek 与 NOSA 在模型目录实现该契约；用户身份、热度、预算准入和 LRU 由 serving/cache
负责，执行器不读取 GR 请求。

共享资源扩展通过纯 `plan_resources` 预留，`allocate_shared` 分配一次。
runner 在分配前 `bind_owner`，关闭全部 session 后 `unbind_owner`，共享资源由外层
`backend.close()` 释放。构造回滚只释放本次绑定后新建的资源；已有相同 plan 保留。
无法确认异步完成时保留 owner 并禁用复用，避免第二个 pool 重复占用同一预算。

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
