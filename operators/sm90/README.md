# SM90 / Hopper

Hopper 是项目主开发平台，优先面向 [NOSA](../../models/nosa/README.md)。
本目录另有 [DeepSeek V3.2 ECHO](../../models/deepseek_v32/README.md) 的独立
prefill/extend 算子，见文末。两模型分别提供自己的 offload 布局与调度适配。
[nosa_attention.py](nosa_attention.py) 提供 resident NOSA block sparse attention，以及
独立 FP32 数学参考。默认 NOSA-8B specialization 使用本项目的
[csrc/nosa_attention_fa3.cu](csrc/nosa_attention_fa3.cu)，复用已安装的 FlashInfer 0.6.18
FA3 mainloop 和顶层共享 CUTLASS；未复制或修改第三方源码。
输入 Q 为 `[query, query_head, D]`，K/V 为 `[token, KV_head, D]`，选择为逻辑 64-token
block ID 与 validity mask，CIS 为 `[token, KV_head]` 的加性 logits bias。
BF16、D128、GQA=16 且 K/V strides 相同时，每 CTA 合并八个 query 的选块并集，
用两次 64×64 TMA 搬入每个物理 64-token page。准备阶段保存 page membership，
主循环预计算 query/page 的因果边界，并按各自选择和 CIS 计算 attention。
一个 KV tile 的两页均由八个 query 选择、且最早 query 已能看到两页全部 token 时，
跳过该 tile 的逐元素遮罩判断；QK 缩放、CIS 加法和 softmax 舍入次序保持不变。
Q 直接从原 stride 通过 TMA 读取，epilogue 直接写最终输出，并标记非有限输出以触发修复。
当 `ceil(query / 8) × KV_heads == 256` 时，准备后按实际并集的 KV tile 数降序
排列 CTA 工作，同长度按逻辑 batch ID 升序；调度仍保留原 query、输出与 repair 位置。
完整路径包括选块准备、工作排序、FA3 和 per-query repair 四个 kernel，全部计入 attention 时间；
其他非空 FA3 几何省去排序，保留三个 kernel。
不物化逐 query 展开的选中 KV。非空 BF16 输入从一个 query 起使用该路径。
FP16 或 K/V strides 不同的输入使用现有
[csrc/nosa_attention.cu](csrc/nosa_attention.cu)；其四-query grouped 双缓冲与 repair
实现保留。超出 FA3 并集容量、选中尾部物理块或产生非有限输出的组由 repair 处理。
其他已支持形状使用 [_nosa_attention_triton.py](_nosa_attention_triton.py)。
所有 native attention 路径先按 FP32 分别舍入 QK 缩放与 CIS 加法，在原单位中
减去最大值，再把差值转换到 base-2，保留大数抵消和公共偏移下的小 logits 差异。
native BF16 PV 按实际选中 token 数对 V 做二次幂缩放，归一化后恢复尺度，防止
极大有限 V 的未归一化累加溢出；整数转换保留 subnormal、NaN 和 Inf。最终转换前
仅将有限结果限制在输出 dtype 的有限范围内，FP16 同样使用这一转换保护。

[nosa_indexer.py](nosa_indexer.py) 将两遍 tiled query-aware scoring 与五窗口 max pooling
融合；[csrc/nosa_scores.cu](csrc/nosa_scores.cu) 在大 batch 上使用
[nosa_scores_fused.cuh](csrc/nosa_scores_fused.cuh)：16 queries 的四个 consumer warpgroup
共享 K 三级缓冲，第一遍的两组 consumer 错峰执行 QK 与归一化；在单 kernel 内完成
两遍 QK，normalizer 保留在寄存器中。score 同样在原单位中计算最大值和差值。动态寄存器分配
通过运行时资源检查后才使用；其他 native 路径分两次 launch：第一遍使用 TMA 与 WGMMA 双缓冲流水计算 softmax normalizer
（query 数不足 128 时最多分 4 段；否则 1 段），第二遍重算 QK，
合并 normalizer 并完成 GQA 舍入与 pooling。两遍不物化完整 score 矩阵。压缩 K 少于 2047 个（query 数≥1024）或
511 个（query 数<1024）时，dispatcher 使用单 kernel Triton scoring。native selection
在 [nosa_selection.cu](csrc/nosa_selection.cu) 完成精确 Top-33 / Top-64、smaller-ID ties
及有序 ID/validity 输出。满足连续 query 覆盖条件时，每 KV head 预选共享 CIS prefix
候选；通用形状保留逐行路径。符合布局条件的大 batch 通过一个 C++ 入口提交 score
与 selection kernels。自有 cache 的连续追加路径用 checked C++ 入口提交有限值检查、
增量压缩、共享 CIS 排名和选块；模型仍持有预约/提交/回滚，校验 scratch 与所有输出
分离，失败时不写派生记录、排名或选择。显式 Triton 对照继续使用 FlashInfer `top_k(..., tie_break=SMALL)`。
模型层负责 CIS 投影、选择策略与 cache 布局，见
[NOSA 实现](../../models/nosa/README.md)。CUDA 后端要求 SM90、FP16/BF16、D=64/128、
GQA group≤32，FP32 累积；AV 概率舍入为输入 dtype。reference 用于正确性对照。

64K prefix + 1K query、2 KV heads 的完整 NOSA 选择还使用
[nosa_scores_pruned.cuh](csrc/nosa_scores_pruned.cuh)：两遍 QK 保持相同的对齐 tile，
第一遍保存向上舍入的 FP16 最大值及边界列摘要。精确的部分 Top-33 阈值只剔除
上界严格更低的第二遍 tile；最终概率、GQA 舍入、五窗口 pooling 和稳定 ties 保持原语义。
摘要非有限或无法证明上界时继续计算。其他几何及资源检查失败时保留原融合路径，
独立 pooled-score API 不启用剪枝；完整 indexer 时间包含构造上界和求阈值的开销。
第一次精确阈值搜索只枚举四个 seed、边界 halo 和 mandatory 块的去重支持集；
未计算位置仍按负无穷处理，保留其对 rank33 的有效重数。支持集前提不成立时回到全行搜索。
最终选块复用 seed 的精确 Top-33 阈值作为搜索下界，遇到 NaN 键时保留完整范围搜索；
最终升序排序只处理已选出的 64 个逻辑 ID，保持原稳定选块结果。
共同的首遍、四个 seed tile 和精确阈值之后，两个八-query consumer 组使用独立
的尾段 tile 列表与 K buffer，避免短组计算仅由另一组保留的 tile。
该剪枝入口已检查的 1024-query / 4159-record / 1040-block 几何使用编译期常量，
Q/K 数据与 stride 仍在运行时传入；其他输入继续使用原 dispatch。

完整 NOSA 的 CUDA indexer 一次处理完整 query batch；`query_chunk_size` 仅控制
reference（默认 64）。连续入口以标量 `query_start + row` 计算位置，不创建 Torch
下标张量。外部位置接口仍校验范围。上限为 262144 tokens / 4096 blocks。
64K+1K 的 BF16 pooled-score scratch 为 4.0625 MiB，QA 与 CIS 两阶段复用；模型请求
还在各层之间复用该 scratch。GQA 求和后先舍入到模型 dtype，再做 pooling。

[nosa_compression.py](nosa_compression.py) 增量写入完整的 32-token / stride-16 K/CIS
压缩窗口及稳定 CIS pool，与 [IndexerCache](../../cache/indexer_cache.py) 配合。
末尾最多两个未稳定 block 在选择阶段计算。已提交的压缩前缀不重算；融合有限值检查
只扫描当前 Q 和尚未校验的自有 K/CIS 后缀，外部 tensor view 仍扫描完整输入。

[nosa_offload.py](nosa_offload.py) 的 `NosaFetchWorkspace` 通过
[_nosa_offload_fused.py](_nosa_offload_fused.py) 加载
[csrc/nosa_offload_fused.cu](csrc/nosa_offload_fused.cu)，在一个 cooperative CUDA
主 kernel 内融合 NOSA sparse host fetch 与 persistent FA3 attention。
当前仅支持 native SM90、BF16、D128、GQA16，保留原 attention 的分组、算术与
numerical repair；首次使用计划仍由 [nosa_offload.cu](csrc/nosa_offload.cu) 生成。
GPU planner 对 `(KV head, block)` 去重，compactor 生成唯一页队列；每个
64-token 页拆为 8 个不交叠的 8-token stripe，由 fetch leader 原子领取
`(page, stripe)`，经 shared slot 和 96-thread barrier 广播。每个历史 K/V
16-byte 向量有唯一线程，以一次 `ld.global.cv.v4.u32` 读取并写入 HBM staging。
不同 CTA 可并行读取同页的不同 stripe，跨 query group 的复用只读取 HBM。

每个 stripe 的全部 writer 完成 stores / thread fences / 96-thread barrier 后，
leader 对 page ready 执行 `atom.acq_rel.gpu.global.add.u32`。跨 CTA 的 RMW 链
累计到 ready=8，TMA warp acquire 观察到 8 后执行 async-proxy fence，再读取完整
页；仅最后完成者累计一次整页字节。尾页空 stripe 不读取 host、没有 trace/payload，
但仍参与完成计数，保证尾页也完成同一协议。

所有 CTA 保留 persistent FA3；最多 96 个 CTA 使用 producer warpgroup 的 warp
1–3 fetch，warp 0 保留 TMA，两个 consumer warpgroup 保留 attention。只有
`KV_heads == 2 && ceil(queries / 8) * KV_heads == 256` 时，fetch 与 compute
都优先 head 1、再 head 0。每个 head 内 fetch 为 block 0 优先、其余 block 降序，
compute 保留原 cost 排序与 logical-batch ties。其他几何回到 block-major fetch
队列和原 FA3 调度。24 / 240 动态寄存器满足本 CTA 的 64512-register pool，
cooperative occupancy 检查保证全部 CTA 同时驻留。

native initialization 合并全容量 metadata 清零、历史 page-0 padding 和 strided
suffix staging；first-use planner 仍在后续有 stream 依赖的独立 launch。强串行
对照也使用同一初始化，再一次读取完整稀疏并集并运行原整批 FA3。初始化、planning、
compaction、prepare / sort / main / repair、输出/完成依赖与 launch gaps 全部计入
完整调用延迟。selection、CIS、causal mask 和每个 query 的算术次序保持原语义。

schema 3 分别记录非空 stripe-copy window、每页的 min(start)/max(end)
envelope 和 softmax update。stripe 起点在任务领取/解码与同步之后、host load
之前；终点在 stores / fences / barrier 之后、ready RMW 与计数之前。page
envelope 可能包含 stripe 间隙，因此分别将两类窗口取并集，与同一 softmax union
求交，再除以各自 union 持续时间。并行区间只计一次，不能用 envelope 间隙充当真实
copy。90% 验收要求每个 profiled sample 的两种 ratio 都 >= 0.9，中位数不能替代
全部样本通过。二者都只覆盖 attention 的 softmax 部分，不代表完整 attention
隐藏率、PCIe 线上占用或整体加速比；逻辑 payload 也不证明物理链路字节数。

`fetch_ctas=96` 是参与 fetch 的 attention CTA 数上限，实际不超过 grid 大小。
`query_tile_size=128` 只分组统计首次读取字节，attention 始终处理完整 query batch。
`overlap=False` 使用相同 native initialization 和 `.cv` 读取策略，一次取齐精确
稀疏并集后运行原 FA3 / repair。返回值在 caller stream 就绪；输入/backing 须
保持有效直至完成，staging/queue/scratch 等待前次操作完成后才能复用。
全局 CPU 回归为 1339 passed、674 skipped、34 subtests passed；SM90 GPU
环境专项为 62 passed，另重复通过的 3 个增强 trace 场景不重复计数。
完整 32 层 checkpoint 检查 1 passed：resident/offload 分别从独立空 cache
构建 64K sparse prefix，再执行 1K extend，全部 normalized hidden 逐位相同，max_abs=0。
L0/L15/L31 的 20 次主测中，完整调用中位延迟相对重新测量的整批稀疏并集串行
对照下降 23.80% / 23.30% / 19.87%；独立 40 次确认下降 23.70% / 23.35% / 21.51%。
三个层的 page-envelope 与 stripe-copy 两套 overlap 中位数均为
95.0938% / 95.3356% / 94.9440%；9 个 profiled sample 的最小值为 92.2007%，全部满足两项 >=90%。

结果来自固定 L0/L15/L31 单层算子回放，完整模型 offload 性能尚未测量。staging
仍为一层完整逻辑地址，无有限 slots / eviction。CUDA Graph capture、并发 Python
调用、跨请求 residency 和 CXL/RDMA 不支持或未验证，主存读取未限速到 50 GB/s。
正式来源见 [offload 实验](../../experiments/nosa_offload_overlap/README.md)。
通用 [sparse_attention.py](sparse_attention.py) 仍是明确失败的接口占位；NOSA 使用专用路径。

NOSA 默认 dense 路径使用共享 [FlashInfer Full Attention](../flashinfer.py) 适配。
本地实现按 EzKernelKit 的 Hopper 调度模式适配 NOSA 布局、CIS 与五窗口 pooling；
具体参考文件及 commit 见 CUDA 源码头。编译使用顶层共享 CUTLASS 和已安装的
FlashInfer Hopper headers；FA3 的独立 flags、版本和 header 哈希记录在构建信息的
`attention_fa3` 字段。通过 TVM FFI 懒加载，要求 `nvcc` 与 `ninja` 在 PATH 上，构建缓存位于源码树外。
不加载 EzKernelKit Python 包或头文件，见[第三方说明](../../3rdparty/README.md)。

原生 specialization 支持 FP16/BF16、D128、GQA=16，要求 Q/K/V 满足 16-byte
地址和外层 stride 对齐；非此形状/布局走既有 Triton 路径。
`CXLDSAGR_SM90_BACKEND=triton` 可显式选择 Triton 对照；默认 `native`。
模型已有的 `sparse_backend="triton"` 接口保留，当前指向这个 SM90 dispatcher。
当前已验收的 BF16-pair / FA3 v3 实现见 [实现检查点](../../docs/nosa_sm90_checkpoint.md)，
完整模块数据已发布，两个完整模块均达 40% 的目标尚未完成。
`94bf521` 的完整模型 native/Triton 同源对照已于 2026-09-29 补测，见
[indexer_block_sparse_profile](../../experiments/indexer_block_sparse_profile/README.md)；
synthetic operator 对照及受影响的 full-NOSA pattern 仍待补测。

## DeepSeek V3.2 ECHO prefill/extend

[echo_indexer.py](echo_indexer.py) 与 [csrc/echo_indexer.cu](csrc/echo_indexer.cu) 复刻
ECHO 的 SM90 extend 分支，在 indexer 计算中融合 pinned-host KV prefetch。
输入为 64 个 128 维 FP8 index heads；coarse histogram 只预取精确 top-2048 的子集，
随后由模型完成精确 top-k 与 residual recall。复用顶层共享 CUTLASS 头文件，保留上游
MIT 许可与提交来源，不加载 SGLang、DeepGEMM 或 SM120 扩展。

[deepseek_mla.py](deepseek_mla.py) 用 Triton tensor-core online softmax 消费每 query
的逻辑或物理 token ID，支持 BF16 576 维 latent/RoPE key 与 512 维 latent value。
[deepseek_linear.py](deepseek_linear.py) 直接消费 checkpoint FP8 权重与 128×128 FP32
block scales，activation 按 128 维量化并使用 UE8M0 scales；grouped MoE 通过 GPU
routing plan 按 expert tile 执行，gate/up/down 共用路由，无逐 token Python GEMM。
CPU reference 用于小规模数学测试，CUDA 路径要求 SM90。

[kv_transfer.py](kv_transfer.py) 与 [csrc/kv_transfer.cu](csrc/kv_transfer.cu) 在当前
CUDA stream 上直接读取 mapped pinned host records；与
[SparseTokenCache](../../cache/sparse_token_cache.py) 配合管理有限 HBM slots、淘汰、
精确补取和 ID remap。主 MLA KV 为 BF16 1152 B/token，indexer K/scales 保持 resident。

修复前算子数值检查及真实 checkpoint layer 0 / layer 3 检查已通过，完整 61 层的
64K + 1K resident/offload 测量已验收，末 token logits bitwise 相同。
KV gather 现支持非对齐连续视图的逐字节复制；修复后的完整模型性能待补测，见
[ECHO 实验](../../experiments/deepseek_v32_echo_prefill/README.md)。全局 GPU 回归入口为
`bash scripts/run_tests.sh gpu`，需要可用 Hopper、nvcc、共享 CUTLASS、TVM FFI 与 Triton；
该入口也运行既有 NOSA 检查，因此同时需要 FlashInfer。
