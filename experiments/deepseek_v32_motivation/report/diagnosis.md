# H64K/A128 单点 profile 证据

本页只保留 `deepseek_motivation_rerun_profile_20261006_01` 的已验收机制证据。它匹配旧正式计时 `deepseek_motivation_rerun_bench_20261006_01` 与独立 check `deepseek_motivation_rerun_check_20261006_01`，配置为 H=P=65,536、A=128、chunk=1,024、NH=16,777,216。原预热策略为 `two_users_then_first_user_host_recall_v1`，每方案执行两个用户首访，再复访第一个用户；新矩阵使用按容量选择用户数的 v2 策略。当前 12 点矩阵另有逐点 check/bench，没有新采集 profile；本页不为它提供匹配诊断。

C10 使用 `deepseek-compute-islands-v4-bound-inputs`，只捕获 projection/finish。Profile 的窗口带有 instrumentation，不能替代正式请求或层延迟，也不能称为完整 extend 图。原单层、prefill 和完整请求时间线展示已撤下，原始 SQLite、运行参数、数值和区间审查继续保留。

[独立数值审查](diagnosis/independent_numerical_audit.json)确认 80 份完整 hidden/logits 与原 check 逐字节一致。[来源审查](diagnosis/independent_provenance_audit.json)重核 1,305 项源码，其中旧 bench 的 1,302 项执行源码相同，并核对 987 项 runtime/native 文件。Profile 源码摘要为 `b0eddd27596c64058d512004f837d5c5eb2f30615dd9771865d0d4afd7cc1043`，匹配的旧 bench 摘要为 `15b38f1df5eee926c3aba0f2b063acff090903c0e31a64758e74b6b233bad1c7`。

[原始活动审查](diagnosis/independent_raw_audit.json)核对 6,560 次 graph replay 与 186,880 个 GPU 节点。Dense cold candidate 的历史主 KV H2D 为 0 B；复访 request 16 共 10 次 H2D DMA，合计 754,974,720 B。实际字节、时间和 API correlation 来自 CUPTI，地址连续性与 pinned 属性来自运行时参数记录。

所选 L1 compute 与完整 L2 DMA 的交集为 0.805312/1.371746 ms，即 58.707078%。分母保留 L2 DMA 超过 L1 finish 的尾部，分子排除 L1 自身搬运尾部及 D2D/control。该局部交集不证明消除全部等待或获得完整请求收益；ECHO 融合 kernel 的计算与 host read 未从 trace 内部分离。

保留的结构化材料包括 [pipeline.json](diagnosis/pipeline.json)、[stage_costs.csv](diagnosis/stage_costs.csv) 和 [cuda_apis.csv](diagnosis/cuda_apis.csv)。不同 capture 的时间轴不能拼接，嵌套 CPU duration 不能直接相加，GPU activity 并集也不是 SM 利用率。

原始 data 位于 `experiments/deepseek_v32_motivation/output/data/deepseek_motivation_rerun_profile_20261006_01/`，Nsight 产物位于 `experiments/deepseek_v32_motivation/output/profile/deepseek_motivation_rerun_profile_20261006_01/`。单层区间、原始检查和 overlap 数值的原始 JSON/CSV 位于 `experiments/deepseek_v32_motivation/output/data/deepseek_motivation_rerun_analysis_20261006_01/`；当时发布索引与 helper 源码的来源为 `experiments/deepseek_v32_motivation/output/data/deepseek_motivation_rerun_report_20261006_01/`。这些 output 路径不随 Git 分发。旧发布索引只记录当时素材，不表示已撤下的图片仍是当前交付，也不要求保留整套旧展示副本。当前发布清单选择上述无图证据，没有重新运行或扩大原 profile 的验收范围。
