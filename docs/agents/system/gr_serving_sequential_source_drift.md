# Sequential serving run01: source drift and device identity

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

2026-10-02 只读核查。结论：run01 确实触发源码冻结检查失败；两个变化文件不在
serving 的静态调用依赖中。同期 layers3 profiler 使用物理 GPU 1，不能将其中的
逻辑 `cuda:0` 解释为 serving 使用的物理 GPU 0。run01 不作为验收或性能结果发布，
后续在隔离源码树中以新 run ID 完整重测，保留严格 source guard。

## Run01 的直接证据

Run ID 为 `gr_serving_h200_20261002_sequential_u16_t32_h64k_01`；诊断目录为
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving/gr-serving-gr_serving_h200_20261002_sequential_u16_t32_h64k_01.aGAeuj/`。
以下均为该目录中的原始文件，未修改：

- `data/metadata.json` 记录开始时间 `1790952446.5538576`，即北京时间
  `2026-10-02 22:47:26.553858`；状态仍为 `running`，无 completed 时间。这是失败后
  留下的未完成 metadata，不能据此推断进程仍存活或运行已被接受。
- `data/measurements.jsonl` 和 `data/correctness.jsonl` 各有 256 条记录，检查到的
  `correctness_exact` / `exact` 均为 true；metadata 包含八个 case。这里只核查记录数量
  和布尔字段，未重新执行完整数值审计，也未运行模型。
- `log/measure.stderr.log` 明确报错：
  `RuntimeError: implementation changed while measuring: ['models/deepseek_v32/echo_infer.py', 'models/deepseek_v32/tests/test_echo_infer.py']`。
  失败点为 `measure.py` 的 `verify_source_snapshot(output)`，在最后请求之后、接受与
  发布之前。stdout/stderr 最后 mtime 为北京时间 `23:12:36`；mtime 仅作时间界限证据，
  不冒充精确进程结束时间。
- `data/source_manifest.json` 含 155 项；逐文件重算当前 SHA-256 后，恰好只有报错中的
  两项不同。run 的总源码身份为
  `75909d12fb212c7a80277bae5e52acc340136fd88a9848da7bc51cb8e5750e58`。

| 文件 | 保存源码 SHA-256 | 核查时源码 SHA-256 |
| --- | --- | --- |
| `models/deepseek_v32/echo_infer.py` | `1e623f1692084454ecf0f4a2b14e160bad9c9ee5a4c26cc434a7587ddbadf9d2` | `bda86a7aa2ca402ac1c02f35481587b5ec5b03d53fcb455b9baaef8044d90701` |
| `models/deepseek_v32/tests/test_echo_infer.py` | `31423031739e0be58ca8b76e9b1ddafe7b74502746e80565b75b16f1d56c4b08` | `aa766ef578e662f2c0fe34584aa003657b6371180062403a0d038db9eed62c1b` |

## 改动与 serving 调用路径

与 run01 保存源码直接比较，`echo_infer.py` 增加 `DeepSeekEchoModel.num_layers`
诊断前缀选择、`return_hidden` 返回值及相应 CLI 参数；测试文件增加这些行为的检查。
改动均在 standalone `DeepSeekEchoModel` 或其测试中。

[旧 GR 实验（已结束）](experiment_organization.md#retired-gr-serving) 的 `_backend` 直接构造
[DeepSeekServingBackend](../../../models/deepseek_v32/serving_backend.py)，后者使用
`CheckpointBlock`、`EchoAttentionRunner`、`Config`、`CheckpointReader` 和 `rms_norm`，
由自身的执行代码调用 block、final norm 和 LM head，并未构造 `DeepSeekEchoModel`。

使用 Python AST 只解析源码、不导入或运行模块，从 serving 测量入口递归收集本地
`import` / `from` 依赖：闭包共 62 个本地模块，不包含 `echo_infer` 或它的测试。
其中 DeepSeek 模块仅有 `serving_backend`、`echo_attention`、`echo_block`、
`echo_model`、`request_format`；此闭包没有发现 `__import__`、`importlib.import_module`、
`exec`、`eval` 调用。源清单本身覆盖整个 `models/`，因此依赖外文件变化仍会正确触发
既定的冻结规则。静态调用路径结论不用于绕过该规则。

## 同期 layers3 实验的 GPU 和时间

[layers3 summary](../../../experiments/deepseek_v32_echo_prefill/report/layers3/summary.json)
和 [postrun audit](../../../experiments/deepseek_v32_echo_prefill/report/layers3/postrun_audit.json)
对应 `20261002_echo_layers3_mfu_01`。其原始 `hardware.json` 记录
`physical_device_selector="1"`、`CUDA_VISIBLE_DEVICES="1"`，NVIDIA-SMI index 为 1。

| 运行 | 设备信息 | GPU UUID |
| --- | --- | --- |
| Serving run01 | `device=cuda:0`，`visible_devices=null` | `1bdee8b4-22ac-536c-208b-bfb4ed38b878` |
| Layers3 profiles | 物理 GPU 1 映射到逻辑 CUDA 0 | `a2226185-cb05-a411-80da-f365154128fe` |

进一步以 SQLite 只读模式核查
`experiments/deepseek_v32_echo_prefill/output/data/20261002_echo_layers3_mfu_01/capture_1.sqlite`
至 `capture_4.sqlite`：每份 `TARGET_INFO_CUDA_DEVICE` 都记录
`gpuId=1, cudaId=0, pid=536504`；`TARGET_INFO_GPU` 对应物理设备 PCI
`0000:2a:00.0` 和相同 `a222...` UUID。

按 `TARGET_INFO_SESSION_START_TIME.utcEpochNs` 加实际 kernel start/end 得到的北京时间：

| Capture | 首个至最后 kernel 时间 | Kernel 数量 |
| --- | --- | ---: |
| 1 | `23:02:46.063030` – `23:02:48.745541` | 47502 |
| 2 | `23:02:52.098190` – `23:02:52.146767` | 756 |
| 3 | `23:03:08.884282` – `23:03:12.594218` | 87174 |
| 4 | `23:03:17.488713` – `23:03:17.558758` | 1455 |

因此这些 profile 确实处于 serving 的运行时间范围内，但使用的是另一块物理 GPU。
三个 `20261002_echo_layers3_ncu_*_01` 的 `full.json` 和 `source.json` 也记录
`CUDA_VISIBLE_DEVICES="1"` 和同一个 `a222...` UUID。现有证据不能支持“该 layers3
实验与 serving 重叠使用物理 GPU 0”的说法；本核查也没有测量共享 CPU/DRAM 的影响。

本次仅读取源码、JSON/JSONL、日志和 SQLite，并写入本说明；未修改数值源码、guard、
原始产物，未执行测试或 GPU 工作。
