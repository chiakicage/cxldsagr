# SM90 / Hopper

Hopper 是项目主开发平台，优先面向 [NOSA](../../models/nosa/README.md)。
[sparse_attention.py](sparse_attention.py) 预留 main sparse attention + offloaded cache fetch
的算子入口，目前调用会抛出 `NotImplementedError`。算子接收逻辑 block selection 与
cache access，未来在内部安排计算与搬运；尚未实现 sparse attention、DRAM 读取或 overlap。

当前 NOSA 可运行路径使用共享 [FlashInfer Full Attention](../flashinfer.py) 适配。
本地 EzKernelKit 可作为 block sparse attention 参考，见
[第三方说明](../../3rdparty/README.md)；它不是当前运行后端。
