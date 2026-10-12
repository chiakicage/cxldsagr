# DeepSeek V3.2 observation

本实验观察 device-resident 条件下的具体 GEMM、非 GEMM 操作和完整 API 聚合，属于
baseline 性能合理性实验。指标包括独立 clean GPU 时间、有效矩阵 MFU，以及另一进程
采集的 NCU HBM 带宽与利用率。范围包括公共算子、MoE 和独立序列的 Indexer/MLA。
输入与 cache 均驻留设备，各操作与 API 聚合分别计时，不能相加为完整模型或 serving 延迟。

当前正式结果为 `observation_20261012_01`。全部 66 个负载完成匹配的独立数值验收、
三次静态 Graph replay、clean 计时和独立 NCU profile；共 2,616 行直接观测，其中公共
算子 506 行、MoE 938 行、attention 1,172 行。具体操作为 2,073 行，其中 GEMM 244 行、
非 GEMM 1,829 行；API 聚合另列 543 行。全部观测保留 81,096 个 clean 时间样本，
完成 7,848 次静态 Graph replay 比较，2,616 行均有独立 NCU HBM 指标。报告与完整来源见
[结果](report/results.md)、[完整性核查](report/completion.json)和
[来源索引](report/provenance.json)。具体算子及支持条件见
[算子清单](report/operator_inventory.csv)。

## 负载与输入

`B` 是独立序列数，`Q` 是每个序列本步追加的 token 数，`H` 是此前的历史长度。
Decode 使用 `Q=1`、`B=[1,32,64,128,256,512,1024]`；Extend 使用 `B=1`、
`Q=[32,64,128,256,512,1024]`。两类均测 `H=[4096,16384,32768,65536]`，共 52 个
attention 配置。长度中的 K 表示 1,024 tokens，Indexer 精确 top-k 固定为 **2,048**。
各序列拥有独立 Indexer Cache 和 Main KV；单层能执行某个 batch，不证明完整 61 层
模型能容纳同样的 batch。

公共算子与 MoE 按 `S=BQ=[1,32,64,128,256,512,1024]` 各测一次。Indexer score、
MLA 与精确 top-k 逐 B/Q/H 配置测量。物理 score shape 为
`[BQ, round_up(H+Q,256)]`，超出每行 causal endpoint 的位置掩为 `-inf`；top-k 的
输入宽度随 H 变化，不能只按 S 映射。[expanded.csv](report/expanded.csv)将公共结果
映射到对应 B/Q/H，共 11,916 行；`measurement_kind` 与 `source_case_id` 保留原来源，
映射不新增实测。

公共算子读取 checkpoint 第 0 层投影、norm 与 dense MLP，以及 embedding、final norm
和 LM head；MoE 读取首个 MoE 层，当前为第 3 层，包含 256 个 routed experts 和一个
shared expert。合成 activation/KV 使用固定 seed；公共位置为 `65536 + arange(S)`。
这些输入用于观察算子形状与执行效率，不代表真实请求分布或完整模型的传播轨迹。
LM head 测全部 S 行，不等同于 serving 只输出最后一个 token 的路径。

## 具体操作与 API 聚合

| 范围 | 具体矩阵操作 | 单独测量的准备及非矩阵操作 |
| --- | --- | --- |
| Projection | FP8 query/KV/indexer/output GEMM，BF16 `q_absorb`/`v_expand`，FP32 indexer head weights | activation quantization、scale layout、norm、cast、scale、RoPE 与 KV concat |
| Dense MLP 与输出 | FP8 gate/up/down GEMM，BF16 LM head GEMM | SwiGLU、embedding、final residual norm、LM head output cast |
| MoE routing | FP32 gate GEMM，TF32 关闭 | hidden cast、sigmoid、correction、group/expert top-k、weights gather/sum/normalize/scale；metadata 的比较、排序、初始化、scatter、prefix sum、gather、算术、cast 与 mask |
| MoE experts | prepared grouped gate/up/down FP8 GEMM，shared gate/up/down FP8 GEMM | quantization、data/scale gather 与 mask、scale layout、inverse output gather/mask、SiLU packing/kernel、casts、weighting、reduce 与 add |
| Attention | prepared FP8 Indexer，BF16 sparse MLA；融合矩阵核心按实际输出与 useful FLOPs 验收 | page64 K/scales packing、scheduler metadata、causal cleanup、exact top-k、ID/layout 准备及必要的输出合并 |

完整 routing、metadata、linear、SiLU、projection、dense MLP、shared expert、MoE、
Indexer/MLA API 聚合另列为 `api_aggregate`，保留生产执行顺序与内部准备。具体操作为
`gemm` 或 `non_gemm`，后者的矩阵 MFU 为 N/A。API 与具体操作各自独立测量，不能
把 API 减去 GEMM 当成开销，也不能把独立操作时间之和当成生产执行时间。

Projection 明确区分输入投影和输出 `o_proj`。吸收 MLA 使用 checkpoint 的 `kv_b_proj`
权重执行 Q absorb 与 value expand，不另测普通 `kv_b_proj` 调用。
`attention_projection` 聚合包括输入投影、Q absorb、norm、cast、RoPE、quantization、
scale 和 KV concat，不含 Indexer scores、MLA、value expand 或 `o_proj`。
生产 dense MLP 的 packed gate/up 共用一次 activation quantization；prepared gate/up
GEMM 保留这组生产操作数与输出 stride，独立 linear API 各有自己的量化。它们分别
报告，不能重复计入生产开销。

全部 Q1 Indexer 显式选择官方 DeepGEMM paged API，并单列 page64 packing 与 scheduler
metadata；低 H 配置不表示模型默认分派，resident Q1 paged 默认仍要求有效 causal
context 至少 32,768。B1/Q>1 使用官方 nonpaged API。B1/Q1 MLA 使用 split16 decode；
其余配置将独立序列 IDs 转为互不重叠的 spans，调用官方 BF16 sparse prefill。
条件化 copies/scale layouts 及无 kernel 的 view/identity 操作由各行 metadata 说明。

MoE 每档 S 均为 134 行：7 个 GEMM、104 个非 GEMM 操作、23 个 API 聚合。
useful FLOPs 只计实际 selected routes，route histogram、block alignment 和 packed
capacity 单独保存。静态 padding capacity 不等于 DeepGEMM 实际执行 FLOPs。

## 计时、MFU 与 HBM

Clean 时间使用调用两端的 CUDA events，在独立 CUDA Graph replay 中测量。默认
预热 5 次、31 个样本，每个样本平均 20 次调用，报告样本中位数；page64 packing
和需要在区间外 reset 的 mutable 输入使用 `inner=1`。实际 inner 与支持条件逐行保留。
时间包含调用内部 GPU 操作和间隙，不含 Python 提交、编译、数据生成或最终主机同步。

MFU 使用这一 clean 时间和 useful matrix FLOPs；H200 SXM dense 峰值为 FP8 1979、
BF16 989.5、FP32 67 TFLOP/s，来源为
[NVIDIA H200 规格](https://www.nvidia.com/en-us/data-center/h200/)。单精度公式为
`100 × useful_flops / (time_us × 10^6 × peak_tflops)`。混合精度聚合先求各精度
`flops/peak` 之和，再除以完整 API 时间；没有单一 peak。非矩阵操作为 N/A。

HBM 指标来自独立、侵入式 NCU whole-Graph profile，使用实际
`dram__bytes_read.sum + dram__bytes_write.sum`，不根据 tensor 大小或 FLOPs 估算。
带宽为 range 总字节除以 NCU range GPU 时间；每调用字节再除以该 range 的 inner，
利用率为带宽除以该 profile 的 NCU sustained peak。NCU 使用 `cache-control=none`、
`clock-control=none`、单次采集 pass，保留静态重复调用的 warm-cache 状态。
cache 命中可能使实测 HBM 流量远少于逻辑 tensor 字节，因此接近零的 HBM 利用率
不能直接解释为访存低效。NCU 与 clean 的时间分母不同，不能用 NCU 时间替换 MFU，
也不能用 clean 时间重新计算实测带宽。短操作的 NCU Graph 时间可能明显高于 clean
时间，例如数微秒的 clean 操作在 profile 中可超过 100 µs；profile 用于观测本次采集
的 HBM 流量与带宽，不能用于估计原始执行开销。H200 官方名义 HBM 带宽为 4.8 TB/s，
NCU 报告的 sustained peak 随本次采集略有变化，两者分别保留。MFU 和 HBM 利用率均
不等于 Tensor Core 活跃率。

## 当前结果与图表

[完整报告](report/results.md)列出 S=128 公共与 MoE 的逐操作数值，以及 H64K 下
Q1/B1 和 Q1/B128 attention 的时间、MFU、HBM 带宽和利用率。图中每行四列依次为
clean 时间、clean MFU、NCU HBM 带宽、NCU HBM 利用率；横轴为 S 或 H。
下列 PNG 由本 run 的直接观测生成。完整报告嵌入全部 139 张 PNG，对应 SVG 同时保存。

![Projection 第一页](report/figures/common_projection_1.png)

![Projection 第二页](report/figures/common_projection_2.png)

![Normalization 第一页](report/figures/common_norm_1.png)

![Normalization 第二页](report/figures/common_norm_2.png)

![RoPE](report/figures/common_rope_1.png)

![Quantization 第一页](report/figures/common_quantization_1.png)

![Quantization 第二页](report/figures/common_quantization_2.png)

![Quantization 第三页](report/figures/common_quantization_3.png)

![Scale layout 第一页](report/figures/common_scale_layout_1.png)

![Scale layout 第二页](report/figures/common_scale_layout_2.png)

![Layout](report/figures/common_layout_1.png)

![MoE grouped and shared GEMM](report/figures/moe_moe_gemm_1.png)

![MoE routing](report/figures/moe_moe_routing_1.png)

![MoE activation](report/figures/moe_moe_activation_1.png)

![B128 Decode Indexer](report/figures/attention_b128_q1_indexer_1.png)

![B128 Decode main attention](report/figures/attention_b128_q1_sparse_attention_2.png)

以上图均为具体操作；non-additive API 聚合图与完整 inventory 见报告。独立算子结果
用于检查 resident baseline 的执行效率，尚不能说明 offload 能覆盖多少时间。Indexer
Cache offload、Main Cache offload、Recall、Cache Manager 事务和计算/IO 重叠均未在
本实验执行，也未验证真实 agentic/GR 请求、完整 61 层容量或 serving 收益。

## 验收与来源

每个 common/MoE S 和 attention B/Q/H 配置分别启动 check、bench 与 NCU profile
进程。Check 使用独立数值参考并检查三次静态 Graph replay；bench/profile 先核对绑定
源码、输入、后端、精度、shape 和已导出 native/JIT artifact 的 receipt，正式样本和
profile range 内不运行参考计算或逐元素比较。

公共/MoE 矩阵参考采样 token rows 与对齐的 output blocks，覆盖完整输入通道和
选中路由；多数非矩阵操作及量化检查完整逻辑输出。Attention 对每个独立序列采样
query，参考覆盖全部可见 Indexer keys 与完整选中 MLA slots，并全量检查 causal mask
和选择语义。实际 coverage、误差、容差、graph 检查见各 checks.json 和
[attention 验收审计](report/attention_audit.json)。独立算子验收不等于完整模型验收。

来源核查保存每行 clean result、HBM result、NCU 原件 SHA-256、输入身份、receipt、
源码与 native/JIT specialization。公共/MoE 输入 hash 为最多 4,096 个均匀抽样字节；attention 每个序列抽取最多 32 个
均匀位置，并保留各位置完整元素字节。这些均为抽样 hash，不是多 GiB KV 的完整 hash。
Checkpoint 保存 config hash 和 shard size/mtime，不声称权重全量 hash。
已安装 DeepGEMM API 与官方 headers 已绑定，当前未导出其实际 JIT CUBIN，不能声称
每个 vendor JIT 二进制均已逐项核对。allocated/reserved 与设备已用量分别保留；它们
是进程观测，不构成完整 61 层容量准入验证。

## 运行与产物

从仓库根目录运行，环境准备见[第三方说明](../../3rdparty/README.md)：

```bash
bash experiments/deepseek_v32_observation/scripts/run.sh --run-id observation_new --gpu 2

CUDA_VISIBLE_DEVICES= .venv/bin/python -m experiments.deepseek_v32_observation.src.publish \
  --run-id observation_new \
  --report-dir experiments/deepseek_v32_observation/output/data/observation_new/published_report
```

脚本分别执行 check、clean bench、NCU capture 与 parse。可按 group 和 attention shard
在不同设备运行同一 run ID；完整矩阵结束后再 aggregate 和 publish。独立验收放
`/tmp/cxldsagr-checks/deepseek_v32_observation/<run_id>/`，正式数据为
`output/data/<run_id>/`，stdout/stderr 分别为 `output/log/<run_id>/`，NCU 原件为
`output/profile/<run_id>/`。表格、PNG/SVG 和选定来源数据保存到 report/；完整 samples、
输入与源码清单、日志及 profiler 原件保留在 output/。

本轮使用 H200 SXM / SM90；依赖为 DeepGEMM `2.8.1+057ca59`、FlashMLA
`1.0.0+ba89a34`、FlashInfer `0.6.18`、Torch `2.12.1+cu130` 和 Triton `3.7.1`。
实际进程 UUID、后端、native/JIT 绑定与内存观测见结果及来源核查。CPU 测试、静态检查
与独立 pilots 不替代正式性能结果。
