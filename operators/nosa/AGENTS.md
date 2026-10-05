# NOSA 算子协议与验收规则

本文件适用于 NOSA 算子及会改变其调用、cache 或测量语义的任务。
代码路径均相对仓库根目录；同时遵守[根规则](../../AGENTS.md)、
[NOSA 模型规则](../../models/nosa/AGENTS.md)和[实验规则](../../experiments/AGENTS.md)。
当前算子入口见 [README](README.md)，运行结果见
[offload overlap 实验](../../experiments/nosa_offload_overlap/README.md)。

## Resident 与 offload 执行协议

resident NOSA block sparse attention 已接入 SM90 CUDA/CuTe 与 Triton；显式 NOSA
offload 通过 `operators/nosa/attention/offload/api.py`
实现 BF16 / D128 / GQA16 native attention 与稀疏 fetch。当前融合版本在一个
cooperative CUDA 主 kernel 内保留所有 CTA 的 persistent FA3 计算；默认最多
96 个 CTA 使用 producer warpgroup 的 warp 1–3（96 线程）读取 host，warp 0
保留 TMA，两个 consumer warpgroup 保留 attention。每次层调用按
`(KV head, logical block)` 去重并压成唯一页队列，每个 64-token 页拆为 8 个
不交叠的 8-token stripe；leader 原子领取 `(page, stripe)`，经 shared slot 和
96-thread barrier 广播，每个历史向量只执行一次 `.cv` host load。每 stripe
writer 完成 fence/barrier 后，leader 以 acq_rel RMW 累计 ready；跨 CTA 完成链
达到 ready=8 后，TMA acquire 并执行 async-proxy fence 再读取 HBM。空尾 stripe
不读 host、仍参与完成；最后完成者只累计一次整页字节。

仅两 KV heads 且 `ceil(queries / 8) * KV_heads == 256` 时，fetch 与 compute
都按 head 1 → head 0；head 内保留 block 0 优先/其余 block 降序的 fetch 顺序
和原 compute cost/tie 顺序。其他几何保留 block-major fetch 与原 attention 调度。

串行与融合共用新 native initialization，合并全容量 metadata reset、历史
page-0 padding 和 strided suffix staging；first-use planning 保留独立依赖
launch。初始化、planning、compaction、prepare / repair 与 launch gaps 全部计时。
不依赖 host memory 的 L2 复用，不得按八-query group 重复搬运；保留原 selection、
CIS、causal mask 与 numerical repair。prepare / repair helper 全部计入算子时间，
cooperative launch 与 occupancy 检查须保证全部 CTA 可同时驻留；producer /
consumer 的 24 / 240 动态寄存器预算须满足本 CTA 的 64512-register pool，
不能只按整个 SM 的寄存器上限检查，避免 `setmaxnreg` 等待死锁。

`query_tile_size` 仅分组统计首次读取流量，不再拆分 attention；`fetch_ctas` 控制
参与 fetch 的 attention CTA 数上限，不划出专用 fetch CTA。`overlap=False`
一次 fetch 完整稀疏并集，再执行原 FA3 整批 attention。
事务提交及 staging 复用须等待相关异步操作完成。完整 offload 调用的 CUDA Graph
capture 与 CXL/RDMA 尚未支持或验证；普通 owned / budget staging 路径尚无有限 HBM
slots / eviction。固定 P/NH 的直接映射历史 pool 与纯计算 CUDA Graph 另遵循 NOSA
模型规则，不把这些能力扩展为完整 offload capture 或任意容量下的 token LRU。
不支持路径明确失败，不将 resident 检查、算子回放或 CPU reference 表述为完整模型
offload 性能验证。

## 数值与性能验收

NOSA 完整 checkpoint 数值验收须为 resident/offload 分别从独立空 cache 构建
sparse prefix，比较全部 extend hidden；cache 分配统计不等于进程峰值显存。
NOSA overlap 性能对照须包括完整 query batch 的稀疏并集一次 fetch 后计算，
以此为整体延迟验收门槛；若候选拆分 query tiles，另加相同拆分的串行调度对照。
单 kernel 的完整执行窗口不能同时充当 fetch 与 attention 的区间；须使用 kernel
内部实际工作区间证明重叠，且执行窗口相交不能替代整体延迟收益判断。
stripe 路径同时报告 page envelope 与非空 stripe-copy window 两套指标；envelope
须等于本页全部非空 stripe 的 min(start)/max(end)，不能以其空隙充当真实 copy。
90% 验收要求每个 profiled sample 的两种 ratio 都 >= 0.9，不能只检查中位数。
每轮实现更新须重新验收正确性、唯一读取和内部 overlap，并以新 run ID 发布受
影响的性能结果；旧结果按[实验规则](../../experiments/AGENTS.md)保留至替换完成，
不以旧验证冒充新实现结果。
