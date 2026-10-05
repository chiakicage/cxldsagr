# 官方 ECHO 旧发布与当前结果

下表按各自原始计时边界保留观测值。旧运行合并数值验收与计时，当前运行使用独立 check；两次运行的物理 GPU 不同，不能将差值解释为同条件回归或稳定收益。

| 版本 | 方案 | 访问 | 均值 ms | 中位数 ms | p95 ms |
|---|---|---|---:|---:|---:|
| historical | hbm | first | 2114.286 | 2115.503 | 2121.285 |
| historical | hbm | revisit | 2106.515 | 2105.440 | 2117.721 |
| historical | echo | first | 4637.566 | 4638.341 | 4651.222 |
| historical | echo | revisit | 45.561 | 45.470 | 46.138 |
| current | hbm | first | 2126.499 | 2126.137 | 2132.862 |
| current | hbm | revisit | 2123.192 | 2124.119 | 2126.623 |
| current | echo | first | 4596.177 | 4593.201 | 4609.691 |
| current | echo | revisit | 44.862 | 44.516 | 46.391 |

保留的逐请求时间可重算表中均值与总量，原 run ID、源码身份、硬件、环境及计时边界见 [comparison.json](comparison.json)，选定样本见 [request_samples.csv](request_samples.csv)。受影响的旧完整运行在新报告验收后清理，保留资料不支持完整历史源码、native 或数值重审。
