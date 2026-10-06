# 官方 SGLang 实验约定

- 按用户指定的范围，本实验新增或修改的代码、脚本、文档和运行产物都放在
  `experiments/deepseek_v32_echo_official/`。Engine 执行、profile hooks 和绘图源码放
  `src/`，进程编排放 `scripts/`，不改项目模型、cache、serving 或第三方源码。
- 直接 Engine 实验复用 `3rdparty/ECHO/reproduction/cxldsagr/` 中已有的独立环境、
  权重、输入、构建凭据和必要工具，只读访问这些文件。新的日志、数据、NSYS 文件、
  JIT cache 和临时目录均写入本实验的 `output/`；禁止向原复现目录写入字节码或新产物。
  此处取代此前“本实验只保存报告、复现代码全部放在第三方目录”的布局约定。
- 从仓库根目录通过 `python -m experiments.deepseek_v32_echo_official.src.<module>`
  运行 Python 实验入口，不新增 `__init__.py` 或项目安装步骤。
- Profile hooks 只增加 NVTX 和显式观测，不改写官方计算函数；驻留快照的同步和读回
  放在正式请求及 forward 窗口之外。按 CUDA correlation
  归属 GPU 活动，不以 CPU scope 时长代替 GPU 计算或 IO 时长；动态条件下可能不搬数据
  的 fused prefetch/recall kernel 不能直接计为实际 H2D。
- 正式计时须区分同进程预热和仅共享磁盘 JIT cache 的独立进程预热；不得将新 shape
  的首次使用空隙删掉后称为稳态。预热使用正常请求，不手工清除 radix 或 ECHO 映射。
  NSYS 子进程按 PID、启动时间和已观测的祖先链确认归属，不能只依赖进程名或 PGID；
  进程清理接口必须在实际独立 Python 环境中验证。
- 独立检查、正式计时与侵入式 profile 分开运行。保留官方数值验收未通过的状态，
  结构、容量、输出有限性检查不等于跨实现数值验收。性能边界和来源以 README 与报告为准。
- 有效 HTTP 报告保留原签名。Engine 请求、GPU 三层窗口和本地 MFU 阶段计时分别说明；
  不通过相减不同负载的结果估计 HTTP 开销。
