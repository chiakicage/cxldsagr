# 共享实验与验收工具

这里保存实验间复用的工具，不执行模型计算，也不保存性能报告。

- `validation.py`：独立数值验收记录的写入、身份匹配与证据文件校验。正式计时可复用
  覆盖相同实现和配置的记录；源码、native、输入及执行路径由调用者显式提供。
- `provenance.py`：运行源码快照、后端来源及完整输出比较。数值比较供独立验收使用。
- `pool_scan_provenance.py`：实际 pool-referrer provider 的来源与调用路径，
  与 allocator snapshot provider 分别解释。
- `cache_memory_audit.py`：侵入式 allocator 活动分析，用于工程验收和诊断。

请求构造位于 `GR/workload.py`。验收文件由调用者存入系统临时目录或指定工程目录；
测试使用 `tmp_path`。这里不缓存运行时 allocator snapshot，也不替代 finite、
事务、修复或异步完成检查。CPU 回归随 `bash scripts/run_tests.sh cpu` 收集。
