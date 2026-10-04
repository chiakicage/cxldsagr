# ECHO chunk 数值筛选与计算形状诊断

> 迁移定位：本轮实验已独立到 [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md)。
> 下文执行历史中的原命令、源码路径和哈希保持原样；产物现位置通过
> [迁移记录](echo_cache_experiment_migration.md)回查，目录迁移不构成新的测量。

日期：2026-10-03。本文记录冻结实现上的数值筛选及计算形状诊断；正式计时摘要随运行产物保存。
对应[实现计划](echo_cache_implementation_plan.md) §5.4 / §6。

## 当前正式 run 的重新验收

`20261003_echo_shared_chunks_02` 已在 H200 GPU 0、freeze 04 完成，root 确认 wrapper
session `94098` exit 0。其 runtime 包含当前 pinned host 实际容量与 CPU scratch 预算
修正，完整 sweep source 为 1228 文件，SHA256
`b727c720864641bc94a7a2cb4cfa69c1c1ddae2831e370feed882a043a716b6d`；冻结身份见
[冻结验收记录](echo_cache_freeze_gate.md)。实际 checkpoint 范围仍为第 0–2 层，不是 GR
10-block 替身，也不由此选择 serving 默认 C。

本轮重新请求 C=`[256,512,1024,2048,4096]`，64K prefix + 128 extend，P=32768、
NH=65664，统一 workspace=2048，HBM 4 GiB / DRAM 64 GiB。独立 CPU 预算复算保留
256/512/1024/2048；C4096 需要 5,108,643,392 B HBM cache（含执行预留），超过
4,294,967,296 B，预算排除原因和全部资源账本复算一致。

全部 **8 个预算可行 C/mode** 保存完整 prefix/extend hidden、末 logits 和三层各
65,664 行精确 selection；每个 C 内 resident/offload 的完整输出及精确选集均逐位相同。
独立 wrapper CPU auditor 重开全部原始证据，包括失败 C，没有只审计参与计时的 C。
它使用原 tensor dtype 的 `torch.isclose(rtol=0.01, atol=0.02)`，重算通过/失败计数、
first mismatch、bytewise equality、完整误差及逐层选集差异。audit stdout 明确为该 run
`accepted=true`，stderr 为空。

| C | 本轮跨 C extend hidden 超阈元素 | Extend 末 logits 超阈元素 | 本轮处置 |
|---:|---:|---:|---|
| 256 | 0 | 0 | 保留 |
| 512 | 1，位置 `[106,587]` | 0 | 数值排除，保留全部原始证据 |
| 1024 | 0 | 0 | 参考点，保留 |
| 2048 | 0 | 0 | 保留 |

C512 的本轮 extend hidden 最大绝对差为 0.03125、相对 L2 为
0.0017323080036618747。其失败结论由本轮重新采集及原始材料审计产生；没有沿用 freeze 01
的旧排除。完整跨 C prefix 指标和选集差异仍是附加诊断，不提升为额外 gating 条件。

独立轻量 protocol review 再次复算全部 requested C 的预算、从本轮数值字段推导排除并
核对完整计时矩阵；对 auditor 的完整 raw 覆盖和代码进行了检查，没有无新疑点地第三遍
解码大张量。保留 `[256,1024,2048]` 的两模式均完成 cold 2 次 warmup / 5 次测量及
固定 prefix 5 次 warmup / 20 次测量：42 个 warmup、180 个正式样本、6 个单独诊断
条目。失败 C 没有 warmup、timing 或性能排名。fresh generation、natural extend 配对、
固定 snapshot restore 及交错顺序检查通过。

本轮 `result.json` SHA256 为
`0933f9ea9b5368c9e79060a29b57f9e0b0217f5d612c0eb92919ada2b11603a2`。
运行先发布于 freeze 04 的
`experiments/deepseek_v32_echo_prefill/output/data/20261003_echo_shared_chunks_02`；
已导入 main，1272 个原 data/log/profile 文件的 source-before、source-after 与
destination-after inventory SHA256 均为
`ace72203581e029a657c03bad7a4bf7ac4d6001ef2c2e1af3fa5edcc37b73f19`。
导入另存 receipt，不重签 source 或修改原结果，也没有第三遍解码完整数值张量。

## 早期 freeze 01 诊断边界

下文记录早期 freeze 01 的定位实验，不包含后续 pinned host 与 CPU scratch 修正；
其因果对照不能替代上述当前 run 的重新验收。

## 范围与源码身份

- 冻结目录：`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-shared-chunks-01`。
  真实 detached worktree，HEAD `1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c` 加当时实际 dirty 源码；
  710 个仓库文件复制时逐字节一致，overlay SHA256
  `b5fa16a13aa2d6be63543beb5c3beb481517957486dce840ed0864c06389cfbc`。
  35 个实际导入的仓库模块均来自冻结目录；环境及第三方使用原 pinned 依赖链接。
- checkpoint：`/preset-models`，实际第 0–2 层依次传播 hidden/residual，含 embedding、
  三个 dense MLP、final norm 和末 token LM head；物理 GPU 0，SM90。
- 独立空 resident/offload cache 的 64K+1K gate：1 passed，30.01 s，全部
  `[1024,7168]` extend hidden、prefix/extend 末 logits 逐位一致。1212 文件的前后 SHA256 均为
  `4530d282a22b30e7bfcc3d0a31a301756c8b61082eb5ecc2817298796526c89c`，`changed=[]`。
- 64K+128 chunk 诊断：P=32768，NH=65664，统一 query workspace=2048，
  cache 硬预算 HBM 4 GiB / DRAM 64 GiB。1221 文件的前后 sweep/runtime SHA256 均为
  `1e9fa4e10ad6beb4534d14997112dfad0060f94eeb566fddc1a96e6a350141a6`，`changed=[]`。
  DeepGEMM `057ca596`、DeepJIT `2efdab42`、FlashMLA `ba89a346`、CUTLASS `f3fde583`；
  FlashInfer 0.6.18。完整依赖、导入与复制证据在冻结目录的
  `docs/agents/system/echo_cache_frozen_source.json` 和 `echo_cache_frozen_imports.json`。

正式 attempt `20261003_echo_shared_chunks_01` 在计时前的额外跨 C 全 prefix 门槛停止，
没有性能样本，也没有发布到 `experiments/`。其临时 staging 原为
`/tmp/echo-chunk-sweep-20261003_echo_shared_chunks_01.6kAVpa`，已按用户要求清理。

## 完整数值结果

后续诊断保存每个 C、每种模式的完整 prefix/extend hidden、末 logits 与逐 query 精确集合。
以原 tensor dtype 执行 `torch.isclose(rtol=0.01, atol=0.02)`，没有改阈值。
不能用先转 FP32 后手写容差公式的计数替代这个 predicate；本表使用独立 canonical audit。

**C=256、512、1024、2048 各自的 resident/offload 完整 prefix/extend hidden 与末 logits
全部逐位一致；三个层的 hidden/residual 行哈希和保存的排序后精确集合也逐位一致。**
跨 C 参考固定为 C=1024。

| C | Prefix hidden 失败元素 / 469762048 | 最大绝对差 | 相对 L2 |
|---:|---:|---:|---:|
| 256 | 1622 | 0.11279296875 | 0.0014829804628627048 |
| 512 | 2314 | 0.125 | 0.0016780129093655457 |
| 1024 | 0 | 0 | 0 |
| 2048 | 1264 | 0.11279296875 | 0.0013880859439595727 |

| C | Extend hidden 失败元素 / 917504 | 最大绝对差 | 相对 L2 | Extend 末 logits 失败元素 / 129280 | Logits 最大绝对差 | Logits 相对 L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0 | 0.03125 | 0.0014806478905717646 | 0 | 0.0625 | 0.002601082669777554 |
| 512 | 1 | 0.03125 | 0.0017323080036618747 | 0 | 0.0625 | 0.0026797318642156507 |
| 1024 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2048 | 0 | 0.03125 | 0.00150326380280864 | 0 | 0.0625 | 0.0026893682857854167 |

跨 C prefix 末 logits 均通过；C=256/512/2048 最大绝对差均为 0.0625，相对 L2 分别为
0.0025355554605125713、0.0024438883889242105、0.002568877112523291。
C=512 的 extend 唯一失败位置是 `[106,587]`：`-0.197265625` 对 `-0.220703125`。
在此冻结版本上，C=256/1024/2048 通过最终 extend 输出门槛；C=512 必须按数值失败排除。
C=4096 的预分配规划需要 5108643392 B HBM cache（含执行预留），超过 4294967296 B，按预算排除。
这些排除原因不构成失败 C 的有效性能 baseline。

## 固定输入的因果对照

C=256 相对 C=1024：layer 0 residual 首差在 token 2262、hidden 首差在 3382；
layer 2 在 token 413 已出现 residual/hidden 差异，而该位置的前两层输入仍逐位相同。
因此不能把所有差异都归于前层传播或 cache 访问。

取 C=1024 的真实 layer 2 前 2048 行输入，分别按 C=256/512/1024/2048 执行固定输入阶段：
norm、DeepGEMM 投影及 MLP、量化派生字段、两处 BMM、固定 projection 的 indexer 与 ordered top-k、
固定 Q/K/indices 的 FlashMLA、output projection 和 post norm 都逐位一致。
唯独 index-head 的 FP32 `F.linear` 随批形状变化，最大差约 8.94e-8。
实际 flags：`fp32_precision="ieee"`、`allow_bf16_reduced_precision_reduction=True`、
`allow_fp16_reduced_precision_reduction=True`、preferred BLAS 为 Cublas；两处 BMM 的
`allow_bf16_reduced_precision_reduction=False` 对照也逐位不变。

第二个对照固定所有 Q/K、index-Q/K/scales，并将 indexer 与 MLA 的批形状固定为 1024，
只替换按 C=256 计算的 `index_weights`：其最大差为 4.656612873077393e-10，logits 最大差
为 1.9073486328125e-6。token 413 的 rank 295/296 首次互换，选集完全相同：

| Token ID | C=1024 score | C=256 score |
|---:|---:|---:|
| 120 | 3.2172956466674805 | 3.2172951698303223 |
| 343 | 3.2172956466674805 | 3.2172954082489014 |

原来的完全并列变为近并列，官方精确 top-k 因而改变顺序。仅这个权重替换就重现
MLA、attention output 与 block residual 在 token 413 的首差，与完整传播检查吻合。
将双方相同选集按逻辑 ID 排序的**诊断对照**使 MLA 恢复逐位一致。
没有把此排序加到 runtime，也没有修改官方 top-k 消费顺序、计算默认 flags 或验收阈值。

## 可复核材料与下一轮门槛

所有新增诊断材料在
`/mnt/ssd-wlcb/chenkaiqi/.cache/echo-chunk-diagnosis-20261003-01`，不作为性能实验发布：

- `diagnose.py`、`data02/numeric.json`、`data02/numeric/<C>/<mode>/outputs.pt`、
  对应 `selection_layer_*.npy`、`data02/layer_diagnostics.json`：完整逐 C/模式数值与层输出证据。
- `canonical_audit.py`、`data02/canonical_audit.json`：独立重算的原 dtype 完整阈值、误差和哈希；
  完整诊断及 canonical audit 均 exit 0。
- `stages.py`、`stages_l2_t2048/results.json`：固定输入逐阶段与 BMM flag 对照，exit 0。
- `weight_cause.py`、`head_weight_cause02/results.json`、`head_weight_cause02/reference.pt`、
  `head_weight_cause02/weight_c*.pt`：
  只替换 index weights 的因果与逐行近并列分数证据，exit 0。

下一正式 run 保存全部预算可行 C 的原始数值证据。只有同 C 完整 prefix/extend hidden、
末 logits 和精确选集逐位一致，且跨 C 全部 extend hidden/末 logits 通过原阈值的 C，
才进入预热、计时与性能排名。完整跨 C prefix 指标保留为额外诊断，不宣称 prefix 无差异。
审计须从保存的原始张量和选集重算门槛及排除原因；失败 C 不参与性能比较。
预算排除也必须复核：结果保存实际规划维度，独立 auditor 对全部 requested C 调用纯 CPU
资源规划，核对预算可行集合、预算排除原因、每 C 与固定 workspace 资源账本及单独 GR
可行性。不能通过把 C 改标为预算失败来删除本应保存的数值证据。
