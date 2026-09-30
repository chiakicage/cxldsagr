# DeepSeek V3.2

既有 SM120 synthetic decode/extend 模型运行代码包含 DeepGEMM 投影、packed MLA KV cache、
indexer 稀疏索引选择。张量随机初始化并驻留 GPU，不含 embedding、MLP、LM head 或
KV offload 路径。DeepGEMM 切换 `nv_dev` 后尚未重新验证模型运行。

| 模块 | 用途 |
| --- | --- |
| [echo_adapter.py](echo_adapter.py) | 固定 ECHO 版本的真实 checkpoint 前 1-3 层加载与 hidden-only 适配 |
| [echo_dense.py](echo_dense.py) | 全量已有 prefix 的双缓冲预取，保持 sparse attention；另计原 write pool 预算 |
| [echo_kernel.py](echo_kernel.py) | 显式选择原 kernel 或 SM90 fused prefetch phase flag 修复的独立 header overlay |
| [echo_recall.py](echo_recall.py) | 大 Host pool 的 extend recall 地址提升为 int64；单行、带 SHA 的 Triton JIT 派生修复 |
| [deepseek_v32_decode.py](deepseek_v32_decode.py) | `V32DecodeRunner`：投影、GroupedLinear、输出投影、decode cache 更新 |
| [deepseek_v32_extend.py](deepseek_v32_extend.py) | `V32ExtendRunner`：分块 extend、因果 sparse prefill、top-k |
| [deepseek_v32_extend_kernels.py](deepseek_v32_extend_kernels.py) | 融合 FP8 量化与 cache 追加的 Triton kernel |
| [deepseek_v32_ops.py](deepseek_v32_ops.py) | FlashInfer Norm / RoPE 与 indexer 量化适配 |
| [gr_index_selection.py](gr_index_selection.py) | 限制 logits 显存的分批因果 top-k 选择 |
| [tests/](tests) | ops 与 extend 的数学、cache / indexer 验证 |

## 运行

从仓库根目录执行；第三方源码准备见 [依赖说明](../../3rdparty/README.md)：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync --group sm120
.venv/bin/python models/deepseek_v32/deepseek_v32_decode.py --quick
.venv/bin/python models/deepseek_v32/deepseek_v32_extend.py
.venv/bin/python -m pytest models/deepseek_v32/tests -q
```

`deepseek_v32_decode.py` 与 `deepseek_v32_extend.py` 同时支持 `-m` 入口和直接脚本入口。
两者默认读取 `experiments/legacy/deepseek_v32/docs/config.json` 的 attention / indexer 配置。

## ECHO 真实权重适配

`echo_adapter.py` 的 `inspect_checkpoint(model_path, num_layers)` 仅用标准库读取 config、
index 和文件元数据。`scoped_echo_adapter(echo_path, model_path, num_layers)` 延迟导入固定
`bc1b75c1000010d0ac6f032ebaac283255c050b1` 的 ECHO；记录其 tracked patch 摘要，并在作用域内
替换、随后恢复 V3.2 及上游 V3 配置兼容入口的模型注册。它不启动服务器。

在作用域内将 `bindings.loader_class` 赋给 `server_args.load_format`，并用
`json_model_override_args='{"num_hidden_layers":3}'` 初始化本地 `ModelRunner`。
loader 在 `safe_open/get_tensor` 前按索引选择 shard/权重，只加载完整 embedding 和所选
各层的 attention、indexer、norm、dense MLP 与 FP8 scales；保留原 ECHO 权重后处理和模型数学。
实际要求 checkpoint 的 FP8 量化配置与 128x128 block scales，拒绝去掉量化配置的运行时。
activation 与 MLA KV 使用 BF16，index K 为 FP8、scale 为 FP32；容量 preflight 的
“假设 FP8 解量化为 BF16”一列不表示实际权重已全部改成 BF16。
模型输出所选末层的完整 residual hidden states，不执行 final norm 或 LM head。
`next_token_logits` 是形状 `[tokens, 0]` 的空占位，仅满足 SGLang DP 裁剪契约，禁止采样。

当前限定 TP=PP=1、真实 token embedding 与 prefill/extend，拒绝 dummy loader、
`FAKE_P_NODE`、`DS_DEBUG_LAYERS` 和越过 dense MLP 前缀的层选择。
每种后端配置使用独立进程，先设置 ECHO 环境变量再导入；作用域恢复注册表，不卸载已导入模块。
作用域必须覆盖整个 runner 生命周期：其中还应用 [echo_cache.py](echo_cache.py) 的
`gr_extend_recall_free_slots_v1` 修复，将 extend recall 的驱逐数改为
`max(misses - available_slots, 0)`，允许 GR candidate 释放后保留空 slot；退出时恢复原方法。
保留上游 recall/free/alloc 顺序，并用设备断言检查计数边界，
替换原 debug 检查中“存在 miss 时池必须已满”的假设。设备断言失败后须丢弃该 CUDA 进程。
调用侧须限制单请求完整可见上下文不超过 HBM pool，保证 protected hits 与 misses 的并集可容纳。
这属于 `echo_gr_adapted`，不是未修改的原版 server；运行记录应包含
`bindings.cache_adaptations` 和 `bindings.cache_adapter_sha256`。
serving 负责 prefix 与 candidate 生命周期；缓存身份必须包含完整 token 摘要、模型兼容信息及
`bindings.model_instance_id`。元数据指纹不是权重内容校验和。

### SM90 fused prefetch overlay

另在 2026-09-30 多用户完整回放中定位了独立的大地址问题：原 extend recall 在
`host_idx * 576` 中使用 signed int32，约 373 万个 Host token 后可能地址溢出。
`echo_recall.py` 以原 kernel source SHA 为前置条件，克隆全新的 Triton JIT 对象，
只将乘法前的 `host_idx` 提升为 int64，不改上游文件。`open_echo_runner` 的 offload
模式自动应用该 scope，并记录 `provenance["recall_address_fix"]`；退出恢复原对象。
大于 4 GiB 的 pinned Host 地址回归与改动后 64K 四模式交叉正确性已通过。

64K 检查暴露原版 fused prefetch 的 `s_prefetch_enabled` 共享 flag 竞态：
一个 warp 进入下一阶段更新 flag 时，另一个 warp 可能仍在判断前一阶段分支，
导致同步路径不一致。[echo_kernel.py](echo_kernel.py) 提供派生修复
`echo_sm90_prefetch_phase_snapshot_v1`，先读取线程局部 phase 快照，再同步并按快照分支。
它不更换 indexer/top-k 算法，不修改 checkout 或已安装包，只在独立 overlay 中实体化
一个修复后的 `.cuh`，其他 include 链接原安装树；使用期间原包须保持不变。

目标 header 为 `include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh`，固定摘要为：

- 原始 source SHA256：`38db9dfc0086f7d7b5d058df6e87465e3bdd51c68bc0b5d6bbb15db1723b7691`。
- 修复后 SHA256：`a7f2db638388150dae94664af5fca7b0277ca3c46759749db28a583da12269f4`。

overlay 还记录 installed include manifest SHA、C++ 扩展 SHA、初始化器源码 SHA、
overlay identity 和 JIT 配置。源文件不匹配或已有 overlay 被修改时拒绝使用，
不静默调整补丁。默认缓存位于 `~/.cache/cxldsagr/echo-deepgemm/`，源码及 include 路径
参与不同的 JIT cache key，不覆盖上游 native 编译产物。

`scoped_echo_kernel(echo_path, patch_id=None)` 默认 native，必须先于 DeepGEMM/SGLang
导入；overlay 传入 `PREFETCH_PHASE_PATCH_ID`。DeepGEMM 的 lazy compiler 会保存 include
配置，因此该作用域不可逆、每进程仅一次；退出后保留所选配置但禁止继续使用或重新配置，
切回 native 必须新进程。serving API 对应参数为 `open_echo_runner(..., kernel_patch=None)`；
GPU 脚本默认只给 `echo_gr_adapted` 选择 `phase_snapshot`，
`SPARSEGR_ECHO_TEST_KERNEL_PATCH=native` 可禁用。结果必须标记派生修复，不能称为原版。

### 正确性边界

原生 top-k 的原子输出位置分配可能打乱同一 selected multiset 的次序，影响 FlashMLA
浮点归约；边界同分还可能改变 membership。测试默认
`test_only_logical_topk_order_v1` 只按逻辑 token 重排原 indexer 输出，保留重复项和 `-1`，
逐行验证 multiset 不变；不解决边界 tie，也不改预测阈值。
该控制覆盖所有 cold-prefill chunks 和 candidate extends，有额外同步，仅用于数值检查，
禁止用于性能测量。`SPARSEGR_ECHO_TEST_TOPK_ORDER=native` 可禁用，
临时 reference 校验控制 ID/源码 SHA、ECHO revision/patch SHA 与全部比较输入。
该次序控制和上述 phase flag 修复解决不同问题，不能相互替代。

CPU 合约测试及真实 checkpoint 文件头检查已通过。三层模型在 H100 的 4K + 1K、64K + 1K
GR 输入上通过 resident、阻塞 fetch、修复 ECHO fetch 和全量预取的数值比对与缓存生命周期检查；
这不是完整模型或性能实验结果。长上下文验收和测试控制见
[实验说明](../../experiments/gr_cache_serving/README.md)。
两种长度的 `all` 均使用 logical 次序控制，phase snapshot 修复 ECHO 与同批 resident
参考文件比对通过，容差保持 `rtol=0.02, atol=0.02`。64K 来自 V2 Beauty 请求流的三用户
子序列，不是完整 512 请求回放。4K native 次序四路径也曾通过；不宣称 64K native 次序
通过，原版 ECHO 64K kernel 仍有挂起。完整串行同步回放结果见上述实验说明；统一 HBM 总预算尚未完成；
dense 仍额外保留双缓冲和原 write pool，不能据其当前分配做同预算比较。

```bash
python3 -m unittest models.deepseek_v32.tests.test_echo_adapter -v
python3 -m unittest models.deepseek_v32.tests.test_echo_cache -v
python3 -m unittest models.deepseek_v32.tests.test_echo_kernel -v
```

## 既有 SM120 KV 与 RoPE 语义

以下 packed KV 布局属于既有 SM120 路径；ECHO 初版使用 BF16 MLA KV，
每 token 每层 1152 B，另加 FP8 index K 与 scale，不能混用容量计算。

- MLA cache 每 token 656 B：512 个 FP8 latent、4 个 FP32 scale、64 个 BF16 RoPE key。
- indexer cache 每页先存 64×128 个 FP8 key，再存 64 个 FP32 scale；对外 tensor 形状
  `[pages, 64, 1, 132]` 不能按 token 维写入，须用 `index_cache_views()` 取 key/scale 视图。
- 主 MLA 的 64 维位置分量用 interleaved 配对（`is_neox=False`），indexer 前 64 维用
  split-half 配对（`is_neox=True`）。Attention scale 按原始 192 维 QK 计算并应用 YaRN mscale。
- extend 融合路径同时量化 MLA latent 与 indexer key 并直接写最终 cache；历史 cache 不重新打包。

## 相关实验

benchmark、profile 与 GR checkpoint 层实验见 [`experiments/`](../../experiments/README.md)：
[decode 实验](../../experiments/legacy/deepseek_v32/deepseek_v32_decode.md)、
[extend 实验](../../experiments/legacy/deepseek_v32/deepseek_v32_extend.md)。

共享 GR 请求生成使用 [request_format.py](request_format.py)：DeepSeek 历史请求模板与长上下文预算适配。
用法见 [GR 生成器](../../GR/README.md)。
