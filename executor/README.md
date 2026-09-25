# 模型执行器

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
