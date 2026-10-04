# GR prefill/extend 的 offload 策略候选

日期：2026-10-03。主研究条目 2.5；关联 1.1、2.2–2.4、3.1–3.2、4.1–4.3。
本轮只核对论文与源码、讨论候选策略，没有修改执行路径或运行性能实验。

## 本轮问题与判断边界

研究者猜测 ECHO 的 prefill fetch 用于 chunked prefill，并询问当前只有 history
prefill 与 candidate extend 的 GR 应采用何种 offload 策略。要澄清的是：fetch 的必要
条件、ECHO 的适用范围，以及历史保留与请求内取数应否使用同一种策略。
若没有历史 HBM miss，prefill fetch 没有需要隐藏的读回；若整批选择的并集接近完整
历史，则“每 query 稀疏”不足以支持稀疏搬运优势；若同预算下没有减少历史重建，也不能
用容量收益解释 offload 的价值。这些是后续可能改变当前候选解释的判别条件。

## ECHO 的证据

原论文：[USENIX 最终版](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf)。
原仓库：`https://github.com/sjtu-zhao-lab/ECHO`，本地参考 `3rdparty/ECHO/`，
commit `bc1b75c1000010d0ac6f032ebaac283255c050b1`。本地目录不纳入父项目 Git。

- §3，PDF 第 7–8 页：支持 PD disaggregation 与 mixed execution；专用 P 实例禁用
  offload 是 PD 部署策略。不能把它解释为 ECHO 所有 prefill 路径均禁用 offload。
- §5.2，PDF 第 10 页：Q block i 的 KV 预取与 Q block i+1 的 index-score 计算重叠。
  这是同次调用内部的 Q-block 流水线；不要求 serving 将 candidate 拆成多个调用。
  该节用前一个 prefill chunk 尾部 scores 的 EMA 调整分桶；§6.1 明确 chunk size=2048。
- §6.4.3，PDF 第 15 页：inter-query 微基准至多 1.1×，未做独立端到端吞吐消融，
  因为 §6.2 的 PD 实验使用预计算 KV。PD 主吞吐结果不能证明 GR extend 的收益。

以下源码位置均相对上述原始 ECHO checkout：

| 位置 | 支持的判断 |
|---|---|
| `sglang/python/sglang/srt/mem_cache/memory_pool_host.py:1126–1212` | 当前新 KV 分配 device slots、写入 HBM，同时写入 host backing；不是先只存 host 再读回当前 chunk |
| `DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh:1298–1347,1426–1429` | 预取只领取不在 device 的记录；排除当前 extend 的扫描范围在首个无 prefix chunk 中为空。`kv_len` 是逐 query causal 长度，不能将该范围说成精确覆盖所有历史 |
| `sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py:502,514` | 识别首次 chunk 并重置 offset；fused extend 入口检查 host pool 与开关，无必须使用 serving chunked prefill 的条件 |
| `sglang/python/sglang/srt/layers/attention/nsa_backend.py:638–646` | 精确 top-k 后补齐 remaining misses，预取子集不替代完整选择 |

因此：chunked prefill 是典型用途；第一个 chunk 无旧历史可 fetch，后续 chunk
只需读回选中且已离开 HBM 的历史。已有 offloaded prefix 的单次 extend 也适用，
即使 candidate 在 serving 层不分 chunk。GR 的固定 history 加变化 candidate 属于后一种
访问结构；这不证明真实推荐任务已成立，也不证明 ECHO 在此结构上一定有收益。

## 当前 GR 实现与候选策略

当前 `serving/persistent.py` 在 session miss 时构建历史，命中时直接 extend，
成功后 truncate 到固定历史。`models/nosa/serving.py` 分块 prefill、整批 extend。
当前 scheme 固定，`cache/prefix_pool.py` 超预算时释放整个用户 session 的两级资源；
尚无“HBM 淘汰但 DRAM 历史保留”的独立分级淘汰。

下列是 agent 建议，未记录为研究者已选择的实现或实测结论：

| 生命周期 | 候选策略 | 前提与成本 |
|---|---|---|
| 首访、DRAM 淘汰后的复访或历史失效 | 预算允许时 resident 构建 history，再将需保留的历史写入 DRAM；可以仍按 token chunk 控制 activation | 必须满足现存 session 加构建峰值的同一硬预算，包含索引、派生状态、scratch、待提交 append 与双份 cache；不能借物理空闲显存绕过配额，不能为构建盲目挤出所有热历史 |
| 活跃构建本身不能在预算内 resident | offloaded/chunked build，逐步将历史写入 DRAM，后续 chunk 按需读取 | 分 chunk 可减小 activation、待提交 append 与部分 scratch 的峰值，但不自动消除累计 KV 容量；写回、历史重读与同步均计入构建延迟 |
| HBM 命中或 DRAM 命中的复访 | HBM 保留热 history 或热块；其他有效历史保留在 DRAM，按层加载 candidate 所需工作集 | HBM miss 与 DRAM miss 分开；只有后者或兼容/身份失效才重建。需比较完整热 session 与页级保留的成本，不预设跨请求选块高度相似 |
| candidate 执行期间与结束 | candidate KV/派生状态作为请求内临时状态，结束后丢弃，固定历史保持不变 | 在预算允许的候选范围内可不做持久 D2H；本请求后续 candidate chunk 仍须访问前面 candidate，成功/失败及派生状态须一致处理 |

历史复用依赖当前固定 token 身份、模型/位置编码兼容和 causal prefix 语义。真实 GR
若有不同的 attention mask 或更新历史方式，需要重新核对，不能直接套用缓存。
首次构建与 DRAM 写回在当前在线请求计时中不能免除；离线预构建只有在服务场景明确
支持时才能另设测量范围。无 autoregressive decode 时，历史保留的成本由后续访问摊销。

推荐先把“DRAM 保留历史、HBM 服务活跃工作集”作为候选总体结构，再单独比较热点
保留策略。对当前串行 serving，全局共享执行 staging 是可探索的方向，不是现有能力。
NOSA 现在每个 offload session 都保留 CIS/压缩派生记录及跨层复用的一层完整逻辑地址
staging，idle session 并非零 HBM；DeepSeek 的有限 slots 也不能算作 NOSA 已支持。
分级迁移还必须明确 indexer/派生状态如何随 session 管理。

`cache/host_backing.py:105–123` 目前对候选也执行 D2H，随后再 truncate。
消除不需要持久化的候选写回是 GR 适配候选，需修改事务和派生缓存生命周期；本轮未实现。

## DRAM-hit extend 的策略比较

保持同一模型、精确 selection、CIS、causal mask 和输出范围，保留以下合理对照：

1. **整层历史提前预取**：不依赖当前层 Q，可提前且连续搬运。双 staging、stream
   同步和 HBM 占用计入预算；不能因为传输字节更多就判定它更慢。
2. **整批稀疏并集串行 fetch**：按层、KV head 与逻辑块去重，排除已在 HBM 的记录，
   完成一次并集搬运后执行整批 attention。这是 overlap 的必要门槛。
3. **稀疏 fetch 与计算重叠**：ECHO 隐藏在其他 Q block 的 indexer 计算后面，
   NOSA 当前隐藏在 attention 计算后面；二者窗口不同。短 candidate 的 indexer
   窗口可能有限，不能只由长度断言收益；ECHO 风格需要模型专门适配，不能直接移植语义。

先比较冷构建策略与已有前缀上的取数策略，再看固定完整访问轨迹上的综合收益，避免
将频繁 cold build 的代价误当成 DRAM-hit fetch 代价。全 HBM 保留及冷热分级保留同样
是重要容量对照；研究容量策略时应共享正确的取数后端，研究 kernel 时则固定缓存策略。

判别量包括：每层整批选择的唯一历史字节比例、HBM 命中、实际 H2D/D2H、选集产生时刻、
fetch/indexer/attention 时间及总延迟；另分首次构建、复访重建、DRAM hit 与 HBM hit。
若并集接近完整历史，整层预取可能更合适；并集小时再检验稀疏节流是否抵消选块等待与
管理成本。收益必须在相同 HBM/DRAM 预算和同一访问轨迹下成立。

下一步由 T-003 比较生命周期策略，T-007 解释候选长度/并集/时间窗口，T-008 继续修正
已有 ECHO/DeepSeek baseline。此次源码阅读不恢复旧 baseline 的有效性，也不改变已有
实验结果的保留与补测要求。
