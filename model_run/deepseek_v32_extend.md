# V3.2 extend benchmark

运行单个序列的一次 extend：保留已有 history cache，一次输入 N 个新 token，输出
`[N, 7168]` attention 子层结果。权重与历史 cache 仍为随机数据，不是完整模型生成。

```bash
source .venv/bin/activate
python model_run/deepseek_v32_extend.py
```

默认 sweep：history = 4096/8192/16384/32768/65536；new tokens = 1024/2048/4096。
结果写入 `docs/model_extend_v32_results.json`，包括端到端时间、新 token 吞吐、
峰值已分配显存和各阶段时间。每个 case 在计时前完成初始化及 JIT warmup；历史
cache 生成、权重量化、RoPE 表构建不计时，新 token 的全部计算及 cache 写入计时。

```bash
python model_run/deepseek_v32_extend.py \
  --history-lens 65536 --new-tokens 4096 --chunk-size 256 \
  --warmups 1 --iters 3 --output /tmp/extend_64k_4k.json
```

`--chunk-size` 控制投影、indexer、attention 和输出投影共同的处理块大小，默认 512，
可调整为 128/512/1024 来比较吞吐与显存。每块仍是多 token GEMM + sparse prefill，
不会转成逐 token decode。输出始终包含所有新 token，切块不改变因果可见范围。
SM120 masked grouped GEMM 的投影输入和 W_VB 输入按 128 行补齐，计算后裁掉
补齐行；默认 512-token 块无需额外行，任意长度的尾块也采用这一规则。

实现路径：

1. 用 decode runner 的投影及 FlashInfer Norm/YaRN RoPE；position 为 history + 新 token 偏移。
2. 把本块新 KV 写入 MLA 的 656 B/token cache，并更新 indexer 的 FP8 key 与 FP32 scale。
3. DeepGEMM `fp8_fp4_mqa_logits` 读取连续 indexer cache，每行的 exclusive end 是
   `history + token_offset + 1`。该 extend cache 使用独立连续 key/scale，避免 decode
   paged cache 到连续格式的反复转换。
4. 因果 top-k 显式屏蔽未来位置，无效位置填 -1，再调用 SM120 sparse MLA **prefill**。
5. W_VB 和 W_O，写入完整输出中当前块的切片。

不分配全量 `[new_tokens, total_tokens]` logits：默认 chunk=256 时，64K+4K 场景的
单块 FP32 logits 约 68 MiB（另有内核对齐）。所有历史 token 的 KV 保留；稀疏 top-k
减少 attention 读取量，不意味着只保留 top-k 个 cache token。

`stages_ms` 是独立一次 profile 中，各块 CUDA event 测量的同名阶段之和；可能包含
CPU 发射间隙，不等于纯 kernel 时间，也不要求与另外测量的平均 E2E 完全相加相等。
峰值显存包含 runner 权重、历史 cache、输入输出和本次临时分配；不包含驱动的所有占用。

后端兼容 `sparse_mla_sm120` 和当前安装的 `flash_mla_sm120`。后者按自身默认参数
选择 QK 精度，本脚本不强制覆写该库的精度选项。权重 `--mode` 默认 fp8，也支持
fp8_fp4w。未包含 MLP、残差连接、多请求调度和真实历史 prefill。

测试：

```bash
python -m pytest model_run/tests/test_deepseek_v32_extend.py -q
```

覆盖显式未来 token 屏蔽、DeepGEMM logits 与 PyTorch 参考比较、分块边界、
分块与整块输出比较、历史 KV 不被覆盖。真实最大规模由 benchmark sweep 验证。

## RTX 5080 基线

已运行全部 15 组默认配置（FP8 权重、chunk=256、1 次 warmup、3 次测量），
输出全部为有限值；3 项 extend 测试通过。以下是单个 synthetic attention 子层的时间，
不是完整模型的 token 生成速度：

| History | 新 token | E2E ms | 新 token/s |
| ---: | ---: | ---: | ---: |
| 4096 | 1024 | 18.20 | 56270 |
| 4096 | 4096 | 68.94 | 59412 |
| 65536 | 1024 | 26.91 | 38054 |
| 65536 | 4096 | 104.63 | 39148 |

完整数据见 [results](../docs/model_extend_v32_results.json)。本次使用已安装的
`flash_mla_sm120` prefill 后端（`bf16_qk=True` 默认值）。64K+4K 的独立分段测量中，
index logits 约 26.8 ms、top-k 13.2 ms、sparse prefill 33.1 ms、输出投影 14.6 ms。

## Cache 与 top-k 优化

extend 仅保留融合路径；PyTorch 参考计算仅用于测试。

- `deepseek_v32_extend_kernels.py` 每个新 token 一个 Triton program，同时量化 MLA
  latent 与 indexer key，直接写最终 cache、scale 和 BF16 RoPE，省去临时 packed
  tensor、多轮逐 tile 运算和复制。历史 cache 不重新打包或复制。
- FlashInfer `top_k_ragged_transform` 直接接收每行长度，输出 int32 索引。优化路径不再
  创建完整因果 mask，也不要求 DeepGEMM 另发一个 kernel 清理不可见 logits。
- 整块已对齐时不再调用 padding 复制输入。尾块仍保留 128 行对齐以保证 grouped GEMM 正确。
- `--cuda-graph` 可在 warmup 后捕获整次 extend 并测 replay。捕获使用固定输入地址和
  固定形状；时间排除捕获本身，但包含所有新 token 计算和 cache 更新。分段时间仍来自
  另一次 eager profile，不能当作 graph 的分段计时。

```bash
python model_run/deepseek_v32_extend.py --history-lens 4096,65536 \
  --new-tokens 4096 --cuda-graph --output /tmp/extend_graph.json
```

对历史长度的性能预期应区分 sparse MLA 与 indexer：前者固定读取每个 query 的 top-2048，
后者仍需要扫描可见的全部 indexer key。历史变长会增加 indexer 和 top-k 工作量，
即使 MLA 本身耗时基本不变，整个 attention 子层也不会保持恒定耗时。

### 复测结果（RTX 5080）

进一步将 GEMM 输入的 1×128 FP8/UE8M0 activation 量化融合成 Triton kernel，避免
通用 PyTorch 工具的多次 padding、FP32 临时张量及逐元素 kernel。decode 默认继续使用
原量化函数，extend 通过 `activation_quantizer` 注入融合量化实现。

| History + New | 原版 chunk=256 | 优化版 chunk=512（eager） |
| --- | ---: | ---: |
| 4K + 1K | 18.20 ms | 12.45 ms |
| 4K + 4K | 68.94 ms | 50.10 ms |
| 64K + 1K | 26.91 ms | 19.33 ms |
| 64K + 4K | 104.63 ms | 77.36 ms |

这是辅助操作优化加分块调整的总收益。单独保持 chunk=256 时，优化后两个 4K-new
端点约为 52.68/80.17 ms；调整 chunk=512 后约为 50.12/77.65 ms（独立 5 次测量）。
Graph replay 对应约 49.93/77.16 ms，因此 CPU 发射和调度已不是主要差距来源。
所有这些数字都是 synthetic attention 子层的测量，不是完整模型生成速度。

64K+4K 的 eager 分段中，cache 写入约 0.043 ms，top-k 约 3.57 ms；原版约为
2.12/13.25 ms。Sparse MLA 自身在 4K/64K history 下约 31.24/31.94 ms，历史长度
造成的主要差异仍是 indexer 扫描（约 3.19/26.68 ms）。

Kernel 级 profile 测得设备事件总计 76.06 ms，其中 MLA 与 DeepGEMM 内核约
62.42 ms（82%），融合 cache kernel 合计约 0.022 ms。仍有布局转换、Hadamard 等
辅助开销；这些数据说明瓶颈主要在 GPU 算子，不能据此声称每个内核都达到硬件算力上限
或排除了 GPU 显存带宽瓶颈。

15 项 GPU 测试通过：包括融合 cache 逐字节一致、activation 量化和 scale 逐值一致、
未来位置设为极大值仍不会被 top-k 选中、尾块与因果验证，以及 decode 回归。
全部 15 组 history/new 长度组合已复测，输出均为有限值。

- [优化后完整 sweep](../docs/model_extend_v32_optimized_sweep.json)
- [分块与 graph 对比](../docs/model_extend_v32_optimized.json)（process peak 显存为进程累计峰值）
- [GPU kernel 耗时](../docs/model_extend_v32_kernel_profile.json)

逐步骤 profile（量化、各投影 GEMM、Norm、RoPE、Hadamard、cache、indexer、top-k、MLA）：

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python model_run/profile_deepseek_v32_extend.py \
  --history-lens 4096,65536 --new-tokens 4096 --chunk-size 512 --iters 5
```

结果见 [逐步骤耗时](../docs/extend_step_profile/summary.md) 与
[原始统计](../docs/extend_step_profile/summary.json)，同目录的 `*.trace.json` 可用 Perfetto 打开。
E2E 为预热后无插桩的 5 次均值；GPU 表为单次预热后 CUPTI 采样，按 launch correlation
归因到最内层步骤，覆盖直接 CUDA driver 启动的内核。表中时间累加了全部 chunk，
不重复计入子步骤。JSON 中另存的 CUDA event 时间包含子步骤，不能直接相加。
# 中文计算说明

逐步计算、张量形状、量化格式、MMA 累加精度与 cache 布局见
[《DeepSeek V3.2 extend：计算步骤、形状与精度》](../docs/extend_step_profile/extend_compute_zh.md)。
