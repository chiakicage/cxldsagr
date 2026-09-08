# DeepSeek V3.2 extend：计算步骤、形状与精度

本文描述当前 `model_run/deepseek_v32_extend.py` 的默认 `mode="fp8"` 路径，按源码核对于 2026 年 9 月 8 日。文中的 FP8/MXFP8 沿用本项目约定：E4M3 数据配任意 FP32 scale 称为 FP8；E4M3 数据配 UE8M0 可表达的 scale 称为 MXFP8。代码参数 `fp8` 是已有接口名称，其投影实际使用 MXFP8 量化和 MMA。

当前 runner 处理单条序列的一个 attention 子层。输入、权重和历史 cache 都是随机构造的，Norm 使用单位权重，LayerNorm 偏置为零。它没有执行 embedding、attention 前的整层 RMSNorm、残差相加、MLP/MoE、跨层计算或 LM head。因此，本文的输出是 attention 输出投影，不能直接理解为完整 Transformer 层输出。

## 形状约定

用 `L` 表示 history 长度，`T` 表示新增 token 数；默认关注 `L=4096～65536`、`T=1024～4096`。一个 chunk 覆盖新增输入的 `[s,e)`，有效行数 `m=e-s`，当前可用 cache 长度为 `S=L+e`。默认 chunk 为 512。

投影阶段实际使用 `p=ceil(m/128)×128` 行。下文主流程表写有效形状 `m`；如果是尾块，投影内核会先计算 `p` 行，再截回 `m` 行。这里的 token 轴不代表多条请求的 batch。

| 参数 | 数值 | 含义 |
|---|---:|---|
| hidden dim | 7168 | attention 子层输入、输出宽度 |
| MLA heads | 128 | attention head 数 |
| Q LoRA rank | 1536 | Q 的中间低秩维度 |
| KV LoRA rank | 512 | cache 中的压缩 latent 维度，也是 MLA 输出宽度 |
| NoPE head dim | 128 | 吸收 WK_B 前的 Q NoPE 宽度 |
| RoPE head dim | 64 | MLA 和 indexer 各自旋转的宽度 |
| V head dim | 128 | WV_B 后每个 head 的宽度 |
| Index heads | 64 | indexer 的 Q head 数 |
| Index head dim | 128 | indexer 的点积宽度 |
| top-k | 2048 | 每个 query 选取的 cache token 数上限 |
| cache page size | 64 | 每页 token 数 |

## 精度要分三层看

张量的存储 dtype、量化 scale 的取值范围和硬件指令类型不一定同名。

例如，activation 量化返回的 scale tensor 是 FP32，但每个 scale 已经取整为 2 的整数次幂。按本文约定，这属于 MXFP8 量化。随后 DeepGEMM 把这些 scale 打包成 UE8M0 字节，交给 block-scaled MMA。

Indexer 的 Q/K 也使用这种 power-of-two scale，却没有把 scale 传入 MMA；它在指令外恢复缩放。其量化格式属于本文约定的 MXFP8，执行指令仍是普通 FP8 MMA。

| 路径 | 操作数的量化格式 | MMA/计算类型 | 输出存储 |
|---|---|---|---|
| 8 个 DeepGEMM 投影 | MXFP8 A × MXFP8 W | block-scaled E4M3，FP32 累加 | BF16 |
| Index weights 投影 | BF16 A × BF16 W | BF16 GEMM | BF16，随后转 FP32 |
| Index logits | MXFP8 Q/K，scale 在外部处理 | 普通 E4M3 MMA，FP32 累加 | FP32 |
| MLA QK，当前默认 | Q 为 BF16，latent K 从量化 cache 恢复 | BF16 MMA，FP32 累加 | FP32 中间分数 |
| MLA PV，当前默认 | 内核量化的 attention 权重 × E4M3 latent V | 普通 FP8 MMA，FP32 累加及缩放 | BF16 |

这里的 MXFP8 不要求每 32 个元素独立求 scale。当前投影 activation 按 128 元素分组，在连续四个 K=32 的 MMA 块上复用 scale。不要据此把本文格式等同于所有库默认的原生 MXFP8 量化布局。

## 每次投影前的动态量化

`QuantizedLinear` 和 `GroupedLinear` 接收 BF16 张量，在每次调用时先执行 `quantize_activation`。对每行沿 K 维的一组 128 个元素，计算：

\[
a=\max(\max_i|x_i|,10^{-4}),\qquad
s=2^{\lceil\log_2(a/448)\rceil},\qquad
x_8=\operatorname{E4M3}(x/s).
\]

Triton 实现用 FP32 求最大值、除法及 scale，并通过指数位向上取整，最后把数据写成 `torch.float8_e4m3fn`。返回的数据形状为 `[M,K]`，scale 形状为 `[M,ceil(K/128)]`，scale 的存储 dtype 为 FP32。

默认模式的权重在 runner 初始化时完成量化。普通线性层按 `[N,K]` 的 `128×128` 块共享一个 scale；权重数据仍为 `[N,K]`，scale 为 `[ceil(N/128),ceil(K/128)]`。GroupedLinear 每个 head 分别采用同样的分块规则。权重量化不计入每次 extend 的运行时间。

DeepGEMM 在 GEMM API 内调整 scale 布局，并打包成 UE8M0。已经被截成 2 的整数次幂的正常有限 scale 在这一步不需要再次近似。MMA 计算缩放后的乘积，在 FP32 寄存器里累加，输出转换成 BF16。这一步仍有输入量化误差和输出舍入误差。

下面给出全部投影的逻辑矩阵形状。权重统一按 `[输出维,输入维]` 表示，计算为 `A @ W.T`。

| 投影 | A 的 BF16 形状 | W 的逻辑形状 | 输出 BF16 形状 | A scale / W scale 形状 |
|---|---|---|---|---|
| WQ_A | `[p,7168]` | `[1536,7168]` | `[p,1536]` | `[p,56]` / `[12,56]` |
| WQ_B | `[p,1536]` | `[24576,1536]` | `[p,24576]` | `[p,12]` / `[192,12]` |
| WKV_A | `[p,7168]` | `[576,7168]` | `[p,576]` | `[p,56]` / `[5,56]` |
| WK_B，128 groups | `[128,p,128]` | `[128,512,128]` | `[128,p,512]` | `[128,p,1]` / `[128,4,1]` |
| Index WQI | `[p,1536]` | `[8192,1536]` | `[p,8192]` | `[p,12]` / `[64,12]` |
| Index WKI | `[p,7168]` | `[128,7168]` | `[p,128]` | `[p,56]` / `[1,56]` |
| WV_B，128 groups | `[128,p,512]` | `[128,128,512]` | `[128,p,128]` | `[128,p,4]` / `[128,1,4]` |
| WO | `[m,16384]` | `[7168,16384]` | `[m,7168]` | `[m,128]` / `[56,128]` |

WO 在 WV_B 截回有效行后执行，输入为 `m` 行。内核自己的 tile 边界处理与 runner 对 grouped GEMM 显式补齐到 `p` 行是两回事。Index weights 不在此表中，因为它保留 BF16 权重，直接计算 `[p,7168] @ [64,7168].T`。

## 输入与分块

整次 extend 的输入 `x` 为 `[T,7168]` BF16，`position_ids` 为 `[T]` INT64，内容是 `L,L+1,…,L+T-1`。输出预分配为 `[T,7168]` BF16。

每个 chunk 都依次完成投影、cache 追加、indexer 打分、top-k、sparse MLA 和输出投影，然后进入下一个 chunk。历史 cache 常驻 GPU，只追加当前 token，不重新投影历史 token，也不把整个历史 cache 拼接复制一遍。

尾块投影前补零，并为补出的行提供位置 0。投影结束后逐个截回有效行，补出的 token 不会写入 cache 或参与后续索引。WV_B 前也按 128 行补齐，以满足当前 SM120 masked grouped GEMM 的使用约束。

## MLA 的 Q/K 投影、Norm 与 RoPE

| 顺序 | 计算 | 有效输出形状 | 精度 |
|---|---|---|---|
| 1 | `qr = RMSNorm(WQ_A(x))` | `[m,1536]` | GEMM 和 Norm 输出 BF16 |
| 2 | `WQ_B(qr)`，reshape | `[m,128,192]` | BF16 |
| 3 | 拆成 `q_nope`、`q_pe` | `[m,128,128]`、`[m,128,64]` | BF16 视图 |
| 4 | `WKV_A(x)`，拆分 | latent `[m,512]`、`k_pe [m,64]` | BF16 |
| 5 | 对 latent 做 RMSNorm | `[m,512]` | BF16 输出 |
| 6 | 对 `q_pe` 和 `k_pe` 做 RoPE | 形状不变 | BF16 输出，FP32 cos/sin cache |
| 7 | 拼接 latent 与旋转后的 `k_pe` | `kv_current [m,576]` | BF16 |
| 8 | `q_nope` 转 head-major，执行 WK_B | `[128,m,512]` | MXFP8 GEMM，BF16 输出 |
| 9 | 转回 token-major，拼接 `q_pe` | `q_attn [m,128,576]` | BF16 |

Q 与 KV latent 的 RMSNorm 调用 FlashInfer，沿最后一维归一化，使用各自的 BF16 affine 权重和配置中的 `norm_eps`。归约以 FP32 进行，结果存为 BF16。这里的 Norm 作用于低秩投影结果。

MLA RoPE 只旋转独立的 64 维部分，采用 interleaved 配对，即相邻两维组成一对。`k_pe` 在 128 个 MLA heads 间共享。位置使用全序列的绝对位置，包含 history 偏移。YaRN cos/sin cache 在初始化阶段用 FP32 构造，运行时调用 FlashInfer 查表旋转。

WK_B 把每个 head 的 128 维 NoPE query 映射到 512 维 latent 空间。这是把原本用于展开 K 的线性变换吸收到 Q 一侧。每个候选 token 的 attention 分数在数学上写成：

\[
u_{t,h,j}=q^{latent}_{t,h}\cdot c_j
             +q^{rope}_{t,h}\cdot k^{rope}_j.
\]

两项相加后才进行一次 softmax。RoPE 和 NoPE 没有各自独立的 softmax。内核接收拼好的 576 维 Q，但会按 cache 格式分别处理 latent 与 RoPE 部分。

转置后的 `.contiguous()` 和 `cat` 会分配或复制数据。它们是 profile 中布局操作的主要来源之一，不应只按 GEMM 计算量估算这一阶段时间。

## Indexer 的投影、旋转与 head weights

Indexer 复用经过 RMSNorm 的 `qr`，同时从原始 chunk 输入计算共享 K。

| 顺序 | 计算 | 有效输出形状 | 精度 |
|---|---|---|---|
| 1 | `Index WQI(qr)`，reshape | `[m,1,64,128]` | MXFP8 GEMM，BF16 输出 |
| 2 | `Index WKI(x)` | `[m,128]` | MXFP8 GEMM，BF16 输出 |
| 3 | K 的 LayerNorm | `[m,128]` | BF16 输入输出；affine 为 FP32 |
| 4 | Q/K 前 64 维做 RoPE | Q `[m,64,64]`，K `[m,64]` | BF16 输出 |
| 5 | 与后 64 维拼回，再做 Hadamard | Q `[m,1,64,128]`，K `[m,128]` | FP32 蝶形加减，最后舍入 BF16 |
| 6 | `Index weights(x)` | `[m,64]` | BF16 GEMM 输出 |
| 7 | 转 FP32，乘 `64^(-1/2)` | `idx_weights [m,64]` | FP32 |

Indexer RoPE 采用 split-half/NeoX 配对：在前 64 维里，第 `i` 维与第 `i+32` 维组成一对。它与 MLA 的 interleaved 布局不同，后 64 维不做 RoPE。

Hadamard 作用于完整的 128 维向量，执行 7 轮蝶形加减，最终乘 `128^(-1/2)`。Q 和 K 采用同一个归一化正交变换，在精确算术下保持点积；当前 BF16 舍入及后续量化会引入误差。这部分目前由 PyTorch 实现，Norm 和 RoPE 才调用 FlashInfer。

## 追加两份 cache

一个融合 Triton kernel 同时写 MLA cache 和 indexer cache。写入位置是 `[L+s,L+e)`，只读取当前 chunk 的 `kv_current` 与 `idx_k`。

MLA cache 按 64 token 一页分配，物理张量为 `[ceil((L+T)/64),64,1,656]`，外层 dtype 为 UINT8。每个 token 的 656 字节布局如下：

| 字节区间，右端不含 | 内容 | 逻辑形状／精度 |
|---|---|---|
| `[0,512)` | 压缩 latent | 512 个 E4M3，按本文约定属于 MXFP8 |
| `[512,528)` | latent scales | 4 个 FP32 存储的 power-of-two scale，每组 128 维 |
| `[528,656)` | 已做 RoPE 的 K | 64 个 BF16 |

Indexer cache 分成两个连续张量：`index_keys [L+T,128]` 为 E4M3，`index_scales [L+T]` 为 FP32。每个 token 的 128 维共享一个 power-of-two scale，因此其数值量化格式也属于本文约定的 MXFP8。

这两份 cache 的 scale 都由 FP32 计算并向上取整到 2 的整数次幂，物理存储保留 FP32。MLA 的 64 维 RoPE K 不做 E4M3 量化。

当前 benchmark 的 `make_case` 直接随机生成历史 cache。真实前缀应来自先前计算，而这段准备工作不属于每次 extend 的计时范围。

## MQA logits：量化格式与指令精度分别记录

当前 chunk 的 index Q 去掉长度为 1 的轴，得到 `[m,64,128]`。`quantize_index` 对每个 token、每个 head 的 128 维向量求一个 power-of-two scale，得到：

- `q8 [m,64,128]`：E4M3；
- `q_scale [m,64,1]`：FP32 存储，数值满足 UE8M0 格式。

在调用 DeepGEMM 前，代码计算 FP32 的合并权重：

\[
\widetilde w_{t,h}=idx\_weights_{t,h}\,s_{Q,t,h}/\sqrt{128}.
\]

其中 `idx_weights` 已经包含上一节的 `1/sqrt(64)`。这两个缩放因子对应不同维度，不能漏掉其中一个。

`fp8_fp4_mqa_logits` 收到 `(q8,None)`、`(index_keys[:S],index_scales[:S])` 以及合并权重。这里 Q 的 `None` 表示 scale 已并入权重。每个位置的 logits 为：

\[
L_{t,j}=s_{K,j}\sum_{h=0}^{63}\widetilde w_{t,h}
\operatorname{ReLU}\left(\sum_{d=0}^{127}Q_{8,t,h,d}K_{8,j,d}\right).
\]

Q/K scale 为正，K 又在 heads 之间共享，所以这与在点积内恢复 Q/K scale 的数学表达一致。浮点运算的舍入顺序仍由具体实现决定。

这次 profile 对应的内核使用 `QMMA.16832.F32.E4M3.E4M3`，没有 `.SF`。点积以 FP32 累加，ReLU、head 加权、归约和 K scale 恢复也使用 FP32，输出 `logits [m,S]` 为 FP32。Q/K 的量化格式是 MXFP8，指令则是普通 FP8 MMA；当前没有保留恒等 scale 的实验改动。

## 因果边界与 top-k

chunk 内第 `r` 行的绝对位置为 `L+s+r`，其有效 key 范围是 `[0,L+s+r+1)`。`ends [m]` 为 INT32，内容就是这些右开边界，起点数组为全零 INT32。

整个 chunk 的 K 会先写入 cache，但每行只读取自己的有效范围，因此当前 query 可以看到自身，不能看到 chunk 内后续 token。MQA 内核可以按 tile 多做部分计算，但无效位置不参与 top-k。

DeepGEMM 返回的 logits 可能是带行 padding 的视图。当前代码通过 `as_strided` 暴露已分配的完整行宽，避免为了连续布局复制整个 logits；FlashInfer 使用 `ends` 限制有效列。没有请求清理无效 logits，因此未来列和 padding 列不能作为有效结果读取。

`top_k_ragged_transform(..., deterministic=True)` 输出 INT32 索引 `[m,2048]`，每个 query 的索引集合由所有 128 个 MLA heads 共享。常用的 history 至少为 4K，有效候选数足够；更短上下文的不足项用无效索引表示。这里的 top-k 是选择比较操作，不执行 GEMM，也没有把 logits 转成 BF16。

## Sparse MLA：BF16 QK、FP8 PV

当前 `flash_mla_sm120.sparse_mla_prefill_fwd` 默认 `bf16_qk=True`。runner 传入：

| 参数 | 形状／类型 |
|---|---|
| `q_attn` | `[m,128,576]` BF16 |
| MLA cache | `[pages,64,1,656]` UINT8 容器，内部布局见前文 |
| `indices` | `[m,2048]` INT32 |
| `d_v` | 512 |
| `sm_scale` | Python 浮点数，由配置计算 |

每个 query 只访问选出的 cache token。latent K 从 E4M3 和 scale 恢复为 BF16，RoPE K 本来就是 BF16；QK 的 latent 和 RoPE 两部分都调用 BF16 MMA，以 FP32 累加。

分数缩放使用原始 Q head 宽度 `128+64=192`，不是吸收 WK_B 后的 576：

\[
\alpha=192^{-1/2}.
\]

启用当前 YaRN 长上下文修正条件时，再乘 `(1+0.1×mscale×log(rope_factor))²`。softmax 在每个 query/head 的选中集合内进行，内核维护 FP32 的最大值、指数权重、归一化量和输出累加器，不生成完整 `[m,128,2048]` attention 矩阵。

PV 阶段直接使用 cache 中的 E4M3 latent V。内核把 attention 权重与对应的 V scale 结合，动态求缩放并量化成 E4M3，再用普通 FP8 MMA 做 PV，以 FP32 累加并恢复缩放。这里内部 scale 由 FP32 最大值除以 448 得到，没有使用前面 activation 量化的 power-of-two 取整，因此不要把全部 PV 中间量也统称为 MXFP8。BF16 QK 模式并不代表 PV 也是 BF16。

主输出 `attn_out [m,128,512]` 为 BF16。接口还返回 FP32 的 `max_logits [m,128]` 和 `lse [m,128]`，runner 只保留第一个返回值继续计算。

## WV_B、WO 与输出写回

`attn_out` 转置并复制成 `[128,p,512]`，按 head 调用 WV_B。每次仍先把 BF16 activation 量化成 MXFP8，然后执行 MXFP8 MMA，结果为 `[128,p,128]` BF16。

结果转回 `[p,128,128]`，截去尾块补出的行并整理连续布局，得到 `[m,128,128]`。展平 heads 后是 `[m,16384]`，再次动态量化成 MXFP8，与 WO 的 `[7168,16384]` 权重做 GEMM，输出 `[m,7168]` BF16。

最后将它复制到 `output[s:e]`。全部 chunk 完成后返回 `[T,7168]`。下一次 extend 若继续同一序列，需要沿用已更新的 cache，并提供正确的 history 长度；当前 benchmark 的 `make_case` 本身负责另建随机 case。

## 初始化、临时操作与可选模式

预先执行的工作包括权重构造和量化、Norm affine 构造、YaRN cos/sin cache 构造、历史 cache 和输入准备。每次 extend 内则包括 activation 量化、scale 布局转换、所有投影、Norm/RoPE/Hadamard、cache 追加、indexer、top-k、MLA、转置拼接、临时张量分配及输出复制。GroupedLinear 还构造 `[128]` INT32 的有效行数数组，供 masked GEMM 使用。

`--cuda-graph` 捕获同一条计算路径，不改变张量精度或计算公式。以上形状也不依赖是否启用步骤计时；计时用的 CUDA events 不是模型计算。

可选 `mode="fp8_fp4w"` 会把 8 个投影的权重改成 E2M1 FP4，按每行 32 元素共享 UE8M0 scale，两个 FP4 数据打包进一个字节。Activation 仍使用本文的 128 元素 MXFP8 量化，GEMM 改走混合 FP8×FP4 路径。Index weights、indexer Q/K cache 和 MLA 的上述配置不随这个开关改变。本文的 SASS 核查结果和主表针对默认 `fp8` 模式。

## 源码与核查记录

- [extend 主循环、cache 追加、logits 与 top-k](../../model_run/deepseek_v32_extend.py)
- [投影、GroupedLinear 和输出投影](../../model_run/deepseek_v32_decode.py)
- [Norm、RoPE、Hadamard 和 indexer 量化](../../model_run/deepseek_v32_ops.py)
- [融合 activation 量化和 cache 写入](../../model_run/deepseek_v32_extend_kernels.py)
- [DeepGEMM scale 布局转换](../../DeepGEMM/csrc/apis/layout.hpp)
- [SM120 MMA 指令封装](../../DeepGEMM/deep_gemm/include/deep_gemm/mma/sm120.cuh)
- [MQA logits 内核](../../DeepGEMM/deep_gemm/include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh)
- [Sparse MLA prefill 内核](../../sparse_mla_sm120/csrc/kernel/prefill/prefill_kernel.cuh)
- [各投影和 logits 的模板、缓存路径、SASS 记录](gemm_precision.json)

文中精度描述以当前源码、默认参数和已有 SASS 核查为依据。后续微基准已确认 MXFP8 MMA 的吞吐约为普通 FP8 MMA 的两倍；本项目采用 BF16 123、FP8 247、MXFP8 487 TFLOPS 作为对应精度的实测峰值参考，见 [5080 微基准](../5080_microbench.md)。

上面链接的 profile JSON、trace 和缓存产物仅保留在本地，不随 Git 提交；仓库保留计算说明、Markdown 报告和复跑脚本。
