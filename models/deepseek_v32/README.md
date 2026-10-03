# DeepSeek V3.2

独立 SM90 完整 checkpoint 的 ECHO prefill/extend，包含全部 61 层、embedding、
3 个 dense MLP、58 个 MoE、final norm 和 LM head。硬件算子按功能位于
[operators/deepseek_v32](../../operators/deepseek_v32/README.md)。
修复前完整 61 层的 64K + 1K resident/offload 测量已验收，末 token logits bitwise 相同；
KV gather 对齐修复后的完整模型性能待补测，版本与结果见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。

| 模块 | 用途 |
| --- | --- |
| [echo_infer.py](echo_infer.py) | 完整模型：跨 GPU 放置常驻权重、按 token chunk 执行全部层、统一缓存事务及 LM head |
| [echo_block.py](echo_block.py) | 双路 hidden/residual、RMSNorm、dense / grouped MoE 与 checkpoint 权重加载 |
| [echo_attention.py](echo_attention.py) | 分块投影、indexer 融合 prefetch、精确 top-k / recall、物理 ID remap 与 sparse MLA |
| [echo_model.py](echo_model.py) | checkpoint 配置与读取、FP8 投影、RoPE / Hadamard、MLA 吸收投影 |
| [serving_backend.py](serving_backend.py) | 单卡 GR serving 的 10 个 dense block / 输入复制工作负载与可复用用户 cache session |
| [tests/](tests) | ECHO checkpoint / block / 全模型调度事务参考测试 |

以下为独立 ECHO 上游环境的 GR cache 实验适配，与上面的 standalone 实现分开验证。

| 模块 | 用途 |
| --- | --- |
| [echo_adapter.py](echo_adapter.py) | 固定 ECHO 版本的真实 checkpoint 前 1-5 层加载与 hidden-only 适配 |
| [echo_dense.py](echo_dense.py) | 全量已有 prefix 的双缓冲预取，保持 sparse attention；另计原 write pool 预算 |
| [echo_kernel.py](echo_kernel.py) | 显式选择原 kernel 或 SM90 fused prefetch phase flag 修复的独立 header overlay |
| [echo_recall.py](echo_recall.py) | 大 Host pool 的 extend recall 地址提升为 int64；单行、带 SHA 的 Triton JIT 派生修复 |

## 运行

从仓库根目录执行；第三方源码准备见 [依赖说明](../../3rdparty/README.md)。SM90 使用基础环境：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync
source .venv/bin/activate
python -m models.deepseek_v32.echo_infer --help
python -m models.deepseek_v32.echo_infer \
  --model /preset-models --devices 0,1,2,6,7 \
  --input-ids /path/to/input_ids.json --history 65536 --chunk-size 1024 --offload --slots 16384
```

`input_ids.json` 是单个请求的 token ID 列表，64K + 1K 场景共 66,560 项；省略
`--offload` 使用 resident 主 KV。权重常驻各 GPU，hidden/residual 在层放置边界传输。
每个 token chunk 顺序经过全部层；所有 chunk、输出与 GPU 同步成功后才提交请求长度。
默认只计算最后 token 的 LM head。完整 GR 请求生成与可复现 profile 命令见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。

显式 `--num-layers 3 --devices 0` 可只加载并顺序执行 checkpoint 第 0–2 层，用于
算子诊断；省略 `--num-layers` 仍执行完整模型。Python 接口为
`DeepSeekEchoModel(..., num_layers=3)`；`forward(..., return_hidden=True)` 另返回
全部输入 token 经 final norm 后的 hidden，供数值比较。截断层数后的输出仅用于
诊断，不能称为完整模型输出或完整 61 层验证。

模型 CPU 回归：

```bash
.venv/bin/python -m pytest \
  models/deepseek_v32/tests/test_echo_model.py \
  models/deepseek_v32/tests/test_echo_block.py \
  models/deepseek_v32/tests/test_echo_infer.py -q
```

## ECHO 真实权重适配

### Dense 预取调度

`scoped_dense_prefetch(..., schedule="attention_window", transport="gpu_direct")` 默认在
独立 CUDA stream 上由 GPU kernel 直接读取 pinned Host KV，写入连续 HBM scratch。
使用 `operators/common/pinned_gather.py`，Host 地址乘法为 int64；没有 CPU KV gather、
pinned staging 或 DMA copy。与 DSA recall 相同的是 GPU-issued Host load，
并非相同的完整 kernel：Dense 不执行按需筛选、分配和缓存映射更新。
在第 L 层 main attention 提交处发起 L+1 层任务，后续输出投影、FFN
及 L+1 层 Q/indexer 可继续提交；直到 L+1 层写入/读取 attention scratch 前才等待该层
copy-ready event。GPU direct 路径在主线程提交 kernel 并记录 event，不需要 Future。
首层在 batch 准备后启动，无法借用不存在的前层 attention 窗口。

双份 HBM scratch 的容量不变，覆盖等待旧层释放；请求边界及关闭时同步相关 GPU 工作。
新 suffix 仍在设备端生成，不搬运未初始化的 Host 数据。
这里只提供可用的依赖窗口，不保证覆盖完整窗口或已测得实际 overlap 比例。

`schedule="layer_end"` 选择层末提交；`transport="cpu_staging"` 显式保留旧传输作对照。
旧传输使用双份 pinned staging，提前调度时使用一个后台 worker，复用前等 DMA，
batch/异常路径排空 worker，关闭先 join 再同步 CUDA。
`open_echo_runner` 对应参数为 `dense_prefetch_schedule`，回放参数为
`--dense-prefetch-schedule attention_window|layer_end`，两者写入 provenance。
传输参数为 `dense_prefetch_transport` / `--dense-prefetch-transport gpu_direct|cpu_staging`，
传输版本、调度和源码均写入 provenance，不能混成同一条性能曲线。
新实现的正确性及配对性能状态见 [GR 实验报告](../../experiments/gr_cache_serving/README.md)。

### Checkpoint 加载

`echo_adapter.py` 的 `inspect_checkpoint(model_path, num_layers)` 仅用标准库读取 config、
index 和文件元数据。`scoped_echo_adapter(echo_path, model_path, num_layers)` 延迟导入固定
`bc1b75c1000010d0ac6f032ebaac283255c050b1` 的 ECHO；记录其 tracked patch 摘要，并在作用域内
替换、随后恢复 V3.2 及上游 V3 配置兼容入口的模型注册。它不启动服务器。

在作用域内将 `bindings.loader_class` 赋给 `server_args.load_format`，并用
`json_model_override_args='{"num_hidden_layers":3}'` 初始化本地 `ModelRunner`。
loader 在 `safe_open/get_tensor` 前按索引选择 shard/权重，只加载完整 embedding 和所选
各层的 attention、indexer、norm、MLP 与 FP8 scales；保留原 ECHO 权重后处理和模型数学。
默认三层，支持前 1-5 层；第四、五层加载完整 MoE 的所有 routed/shared experts、router
与 correction bias。五层共 3212 个张量、6 个 checkpoint 分片，不只加载命中的专家。
2026-09-30 五层 4K/64K 四路径受控正确性已通过；容量性能扫描状态见实验 README。
实际要求 checkpoint 的 FP8 量化配置与 128x128 block scales，拒绝去掉量化配置的运行时。
activation 与 MLA KV 使用 BF16，index K 为 FP8、scale 为 FP32；容量 preflight 的
“假设 FP8 解量化为 BF16”一列不表示实际权重已全部改成 BF16。
模型输出所选末层的完整 residual hidden states，不执行 final norm 或 LM head。
`next_token_logits` 是形状 `[tokens, 0]` 的空占位，仅满足 SGLang DP 裁剪契约，禁止采样。

当前限定 TP=PP=1、真实 token embedding 与 prefill/extend，拒绝 dummy loader、
`FAKE_P_NODE`、`DS_DEBUG_LAYERS` 和超过五层的选择。MoE routing 配置、专家数量/形状与
FP8 scale 均严格校验；仍然拒绝缺失参数、不同 checkpoint 语义或 dummy 模型。
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

五层容量扫描额外使用 [echo_index.py](echo_index.py) 的
`echo_index_read_int64_address_v1`：对单层超过 2 GiB 的 index buffer，在原 GetK/GetS
执行字节偏移乘法前将 page ID 提升到 int64。小 buffer 不安装作用域，写入路径已有 int64
location。原文件不修改，校验固定源码 SHA 并记录适配器 SHA；边界检查位于
`tests/integration/test_echo_index_address.py`。

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

## KV 与 RoPE 语义

- SM90 ECHO 主 KV 每 token 1152 B：512 个 BF16 latent 加 64 个 BF16 RoPE key。
  Indexer FP8 K 与 FP32 scale 常驻 GPU；主 KV 使用 resident 存储，或
  [SparseTokenCache](../../cache/sparse_token_cache.py) 的 pinned DRAM backing 与有限 HBM slots。
  Indexer 内预取后执行精确 top-k，补齐剩余 miss；过大的 query 选中并集拆分消费，
  不截短 query 的精确选择。此路径使用本地 DRAM，不包含 CXL/RDMA。
- Offload attention 入口消费已召回的 HBM records 与物理 ID，复用 device-only MLA。
  融合 prefetch 属于 indexer；当前 attention kernel 本身不读取 host backing。
- 主 MLA 的 64 维位置分量用 interleaved 配对（`is_neox=False`），indexer 前 64 维用
  split-half 配对（`is_neox=True`）。Attention scale 按原始 192 维 QK 计算并应用 YaRN mscale。

## 相关实验

完整模型测量与 profile 见
[SM90 ECHO prefill/extend](../../experiments/deepseek_v32_echo_prefill/README.md)。
原有 DeepSeek / SM120 实验的有效历史结果见
[归档入口](../../experiments/legacy/deepseek_v32/README.md)，按原 run ID 和测量环境解读。

共享 GR 请求生成使用 [request_format.py](request_format.py)：DeepSeek 历史请求模板与长上下文预算适配。
用法见 [GR 生成器](../../GR/README.md)。

## 单卡 GR Serving 工作负载

当前 ECHO 实现的 MFU/cache 策略及 DeepSeek MFU 存在问题；以下描述已实现接口和
数值验证范围，不证明 baseline 比较有效。相关效率、性能归因及排名需修正复测后再判断。

[serving_backend.py](serving_backend.py) 提供本次 GR serving 对照使用的单卡工作负载：
从真实 checkpoint 前三层重复加载 10 个独立的 dense block，source 顺序为
`[0,1,2,0,1,2,0,1,2,0]`，不执行 MoE。每个副本复制其 source block 的 hidden 与
residual 输入，并使用独立权重、主 KV 和 indexer cache。加上真实 embedding、final
norm 和 LM head 共 7,827,793,408 参数；其中 dense backbone 为 5,974,428,160 参数。
这个工作负载用于控制计算和缓存大小，不能作为经过训练的 DeepSeek 8B 模型，也不替代
上面的完整 61 层模型验证。

`DeepSeekServingBackend` 实现共享 serving backend 契约，每个用户 session 在多次请求
间保留固定 prefix，candidate 执行后截短回该 prefix。策略包括 `hbm`、`echo`、
`serial_sparse` 和 `dense_prefetch`：前两种复用现有 resident / ECHO 路径，串行 sparse
在 indexer 与精确 top-k 完成后召回缺失 records，dense 策略使用两块完整 layer staging
和独立 CUDA stream，在当前 block 执行时预取下一层的全部历史主 KV。四者均执行相同的
稀疏选择与 MLA。所有用户 cache 的主 KV、indexer、映射、ECHO counter 及 dense staging
都计入 cache budget；权重和临时激活是另外的执行显存，不能把 cache 统计当作进程峰值。

输出包含全部 candidate normalized hidden，同时执行最后 token 的 LM head；当前 NOSA
serving backend 只输出 hidden，其执行边界不同。跨方案数值对照使用同一模型的相同边界。

单元测试及可选的真实 checkpoint 正确性检查：

```bash
source .venv/bin/activate
python -m pytest models/deepseek_v32/tests/test_serving_backend.py -q
DEEPSEEK_SERVING_CHECKPOINT=/preset-models \
  python -m pytest models/deepseek_v32/tests/test_serving_checkpoint.py -q
```

真实 checkpoint 检查使用独立空 cache 构建 2,304-token prefix，并比较 16 / 23-token
candidate 与截短后的复访；所有策略的全部 candidate hidden，以及每个副本相对其 source
的 hidden / residual 均通过 bitwise 对照。这是正确性验证，serving 性能结果由对应实验
单独报告。实现契约与验证边界见
[执行材料](../../docs/agents/system/gr_serving_deepseek.md)。
