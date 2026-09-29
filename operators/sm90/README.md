# SM90 / Hopper

Hopper 是项目主开发平台，优先面向 [NOSA](../../models/nosa/README.md)。
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

[sparse_attention.py](sparse_attention.py) 仍预留 offloaded cache fetch 与 sparse compute
重叠执行入口，调用抛出 `NotImplementedError`。resident kernel 不实现 DRAM 读取、
搬运、缓存淘汰或 overlap，不能将其作为 offloading 验证。

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
