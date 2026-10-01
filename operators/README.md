# 算子索引

算子按模型和功能组织，当前 native 平台为 SM90 / Hopper。Attention 的三种类型为：
`reference` 提供独立数学参考，`device_only` 消费 GPU 上的 K/V，`offload` 接入主存搬运或已完成精确 recall 的 HBM pool。

```text
operators/
  nosa/
    indexer/                  scoring、selection、compression、validation
    attention/
      reference/              PyTorch FP32 参考
      device_only/            CUDA/CuTe FA3、native fallback、Triton
      offload/                sparse host fetch + persistent FA3
  deepseek_v32/
    indexer/                  ECHO FP8 logits + fused prefetch
    attention/
      reference/              PyTorch MLA 参考
      device_only/            Triton sparse MLA
      offload/                精确 recall 后消费 pool，复用同一 MLA kernel
    linear/                   FP8 linear / grouped MoE
  common/                     模型无关的 pinned-host record 搬运
  flashinfer.py               Full Attention、Norm、SiLU、RoPE 的共享适配
```

各功能的 CUDA 源码位于对应 `csrc/`，单元测试位于对应 `tests/`。
Python reference 可独立导入，不触发 Triton 或 native extension 加载。

| 入口 | 当前能力 |
| --- | --- |
| [NOSA](nosa/README.md) | resident indexer / attention，及 BF16 / D128 / GQA16 offload 融合算子 |
| [DeepSeek V3.2](deepseek_v32/README.md) | ECHO indexer、MLA、FP8 linear / grouped MoE |
| [共享 KV 搬运](common/kv_transfer.py) | record 宽度和 dtype 由调用者传入，支持非 16-byte 对齐的 contiguous view |
| [FlashInfer 适配](flashinfer.py) | 封装现有后端，不解析模型配置 |

NOSA offload 的单主 kernel 融合 fetch 和 attention；HBM staging 仍覆盖一层完整逻辑地址，
尚无有限 slots / eviction。ECHO 将预取融合在 indexer 中；其 attention/offload 入口
复用 device-only MLA，有限 HBM slots 和精确补取由 [token cache](../cache/sparse_token_cache.py)
及模型调度管理。两种路径的能力分别记录，不混用。

本次目录迁移保持 CUDA 计算源码和 Triton 计算语义，更新导入、JIT 依赖、源码采集及测试入口。
已有性能报告仍对应各自 run ID 与源码快照；迁移和回归测试不产生新的性能结果。
NOSA 完整模型 offload 性能尚未测量；DeepSeek KV gather 对齐修复后的完整模型性能仍待补测。
见 [NOSA overlap](../experiments/nosa_offload_overlap/README.md) 和
[ECHO 实验](../experiments/deepseek_v32_echo_prefill/README.md)。

SM120 扩展、旧 synthetic 模型及依赖它们的测量入口已移除。有效历史报告与可独立使用的
CPU 重建工具见 [legacy](../experiments/legacy/deepseek_v32/README.md)，历史执行环境使用
该页记录的整理前 Git revision。共享第三方源码保留，见 [依赖说明](../3rdparty/README.md)。

从仓库根目录执行 `bash scripts/run_tests.sh [cpu|gpu|all]`。GPU 模式检查 Hopper 和
构建依赖，并收集全部相关功能测试；完整 checkpoint 检查需显式设置模型路径，见
[项目 README](../README.md)。
