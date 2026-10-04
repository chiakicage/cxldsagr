# 通用 dense staging 实施与验收

日期：2026-10-03。范围：`cache/staging.py` 及其就近单元测试；未修改模型、算子或
公共 serving 接口。对应 [NOSA shared cache 计划](nosa_shared_cache_implementation_plan.md)
P1/P3 的双缓冲基础。模型集成、完整 checkpoint 数值验收和实验补测由集成执行者完成。

## 接口

```python
staging = DoubleBufferStaging(
    capacity,
    {"keys": (num_kv_heads, head_dim), "values": (num_kv_heads, head_dim)},
    dtype=dtype,
    device=device,
)
with staging.lease(session, generation=generation) as lease:
    lease.prefetch(layer_idx, {"keys": history_keys, "values": history_values})
    records = lease.wait_ready(layer_idx)  # 完整 capacity view，用于后续 suffix 写入
    records = lease.view(layer_idx, end=visible_length)
    # caller 在本 lease 捕获的 CUDA stream 上消费 records
    lease.reset()  # 同一 prefill lease 的下一 token chunk 开始前调用
staging.close()
```

- `lease()` 调用即取得串行使用权，可使用 context manager 或显式 `close()`。
- record 名称和尾维由调用者声明；可传 NOSA K/V，也支持单个 `records` 或标量 record。
  `prefetch` 搬完整传入 tensor，模块不选取 token 或预取内容。所有 records 的首维相同，
  不超过 capacity。CUDA 从 CPU 取非空内容时要求 pinned source，空 history 可用。
- `wait_ready` 先在本 lease 的 caller stream 等 copy-ready，再返回可写 capacity view；
  `view` 在 wait 之前、layer 已被覆盖、lease 已归还或 stream 不匹配时拒绝。
- 新 copy 前在 caller stream 记录 consumer event，copy stream 等待该 event。
  现有 `prefetch(0) -> write suffix -> prefetch(next) -> current attention` 顺序可直接使用。
- slot identity 含 session object identity、generation、layer、chunk cycle。lease 归还或
  `reset()` 同步 caller 与 copy stream，清除两 slot 身份和保留的 source 引用。
  不保留跨 session 的数值有效性；已有 borrowed tensor view 不能逃出其 lease。
- 异常退出同样 join 尚未被 caller 等待的 speculative copy。无法确认完成则 poison，
  保留 ownership、source 和 buffers，拒绝后续 lease。close 仅在重试 join 成功后可处置。
  live lease 存在时 staging.close 拒绝。已关闭的实例不再使用；backend 可创建新实例。
- 同一 lease 绑定获取时实际 caller stream；不同 lease 可以分别使用不同非默认 stream。
  不支持在同一 lease 内更换消费 stream，也不支持 CUDA Graph capture。

## 分配契约

`estimate_bytes(capacity, record_shapes, *, dtype)` 纯整数估算，无 tensor 分配或 CUDA
初始化。固定容量为 `2 * capacity * sum(prod(shape)) * dtype.itemsize`。
`storage_tensors()` 返回实际 owned tensor tuple，可独立枚举 storage 去重。
`shared_bytes()` 返回 `{"hbm": tensor_storage_bytes, "dram": 0}`；CPU reference
沿用 device-storage 角色账本，不能解释为 CPU 环境实际发生了 HBM 分配。
events 和 stream 无随 query/capacity 增长的 tensor 状态；正式执行不扩容。

## 已运行验证

- CPU：`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest cache/tests/test_staging.py -q -p no:cacheprovider`
  初版 13 passed、3 CUDA tests skipped；最终 CUDA 全集也执行这 13 个 CPU reference 检查。
- GPU：`CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest cache/tests/test_staging.py -q -p no:cacheprovider`
  最终 **17 passed**。设备由 PyTorch 报告 NVIDIA H200、SM90，PyTorch 2.12.1+cu130。
  nvidia-smi 列出 GPU 3 为 NVIDIA M403，运行前显存仅 4 MiB。
- Ruff：`.venv/bin/ruff check cache/staging.py cache/tests/test_staging.py` 通过。

覆盖：纯估算/真实 storage、不同 record 布局、零 history、固定容量上限、A→B→A、
generation/layer stale guard、禁止同时借用、live close、chunk reset、非法输入无状态损失、
partial copy 异常恢复、join 失败保留 ownership/poison。GPU 检查包含延迟 copy、延迟
consumer、非默认 caller stream、不同大小/64-token 尾块、异常中尚未 wait 的下一层 copy
以及 reset 前全部 copy 完成；返回输出在下一用户覆盖 staging 后仍保持正确。

这些为资源/异步正确性测试，不是模型性能实验。完整模型结果、预算准入及实验报告仍需
集成后的对应验收。全局 GPU 编排需由集成执行者收集 `cache/tests/test_staging.py`。

## 源码身份

最终验证时 SHA-256：

```text
0edf2e431dfcaa4781e729b7ac5176e0277d8f865460f339b17d7e22d864803e  cache/staging.py
e60e64fd640356b435b64e6777a84b5e794eaa0018a37a07b4dacec5589adb8b  cache/tests/test_staging.py
```

文件在当前工作区为新增、未提交；未改动 DeepSeek/ECHO 所有者文件。

## 小型 native shared serving 集成检查

在集成执行者接入共享 resource 后，新增
`models/nosa/tests/test_shared_serving_cuda.py`；本子任务没有修改模型实现。
此检查使用随机初始化的 4 层 NOSA（hidden=256、MLP=384、Q heads=32、KV heads=2、
D=128、BF16），不是 32 层 checkpoint 验收。权重先用 FP32 CPU 随机数初始化，再复制
到 BF16；稀疏 attention 实际使用 SM90 native 路径。

2026-10-03 在 GPU 3/H200 运行：

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest \
  models/nosa/tests/test_shared_serving_cuda.py -x -q -p no:cacheprovider
```

结果 **12 passed**（6.52 s），仅 FlashInfer 的 Arch deprecation warning。
Ruff check/format check 通过；隐藏 GPU 后该文件 12 skipped，表明 CPU 总回归不会隐式
启动 native GPU 检查，不能将这次 skip 计为 CUDA 验收。

四个 scheme 各覆盖三项：

1. 独立空 cache 的 hbm 对照和 A→B→A 交错。C=224、A=81、chunk=47；两个 session
   分别 capacity=210/prefix=129 和 capacity=160/prefix=79；query=81、17、33、65、1。
   每个用户使用独立非默认 stream，全部 candidate normalized hidden 与 hbm 逐位一致。
   truncate 后检查全部 prefix K/V、CIS 和派生记录；返回 hidden 留到后续请求后重新比较。
2. 在每层 forward hook 独立枚举 shared/session 实际 storage，验证共享只计一次、
   session 彼此独立且不超过预留，shared 指针/容量不变；release A 不破坏 B。
   枚举 shared storage 时不继续遍历 live staging lease 的借入 session/source 引用。
   此检查覆盖存活 tensor，不代替进程峰值或方法内部瞬时分配审计。
3. layer 2 异常后 prefix/派生状态保持，attention adapter 恢复，执行 lease 已归还；
   B 可执行后 A 能逐位重试。dense 路径给已入队的下一层 copy 加延迟，确认异常归还时
   copy event 已完成。非法 query 和临时关闭 native 均在 begin_step 前拒绝，plan、
   shared 存储和 prefix 无变化。

测试文件 SHA-256：

```text
8e59a4a33f5fd888c8d147e81b2995e695d7a384a30951ce6304b79b50732f02  models/nosa/tests/test_shared_serving_cuda.py
```

未将小型模型检查当作正式 checkpoint、64K 工作负载或性能实验的替代。

## 后续 payload 计量集成

完成上述初次检查后，`StagingLease.submitted_copy_bytes` 增加成功提交的 record payload
计数；异常也只计已经成功提交的 copy。NOSA shared serving 的计量按 prefix/candidate
区分主 KV H2D、D2H 和 indexer boundary H2D，稀疏读数使用 session 独立的两个 int64
累计器并计入预算。普通 owned 默认路径不启用这些 serving 计数。

GPU 3 的后续验证为 38 passed（21 项 native shared serving 与 17 项 staging），
另一个 dense 第二 record H2D 失败计量测试单独通过。新增项独立从 attention 实际消费
selection 计算唯一并集，并逐层对照累计字节；覆盖第二个 D2H copy 失败、跨 caller stream
初始化、prefix/复访归零以及失败 status。这里的字节是 KV payload，不是物理链路测量。
最终更广的 NOSA 回归、当前 source hash 与 diff 统一见
[集成 checkpoint](nosa_shared_cache_checkpoint.md)；前节 SHA 只对应前节的初次版本。
