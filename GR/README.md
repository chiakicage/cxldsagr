# GR serving 请求生成器

生成用于 KV cache 实验的可读请求：**用户历史固定、候选每次更新、token 长度精确控制**。用户热度、文本素材、到达时间分别配置。默认从仓库内的 Beauty 热度曲线生成合成用户，商品名也由规则生成，无需原始数据集。默认使用本地 NOSA-8B tokenizer 和关闭 thinking 的聊天模板，保留显式 DeepSeek V3.2 适配。完整文本编码后，可直接通过 Python 接口获取，也可用命令行输出 JSONL，不加载模型或运行 serving。

## Python 接口

```python
from GR.input_generator import create_input_generator, TextConfig
from GR.scheduling import ScheduleConfig

generator = create_input_generator(
    model="nosa",  # 默认模型；DeepSeek 使用 "deepseek_v32"
    heat_source="curve",  # 默认入口
    curve_dataset="beauty",
    num_users=1000,
    text_material="synthetic",  # 默认不读取商品目录
    text_config=TextConfig(
        user_lengths=(4096, 16384),
        user_probabilities=(0.5, 0.5),
        item_lengths=(1024, 4096),
        item_probabilities=(0.5, 0.5),
    ),
    schedule_config=ScheduleConfig(seed=42, qps=100, arrival="poisson"),
)

# 按需逐条生成；request 是包含文本、token IDs 和调度信息的 dict。
requests = generator.iter_generate(1000)
request = next(requests)
input_ids = request["input_ids"]
prompt = request["prompt"]
timestamp = request["timestamp"]
# 继续消费同一个 requests 迭代器，保留这条流的访问计数和模拟时间。
for request in requests:
    pass  # 在这里交给你的 serving / KV cache 实验代码

# 也可以自行指定已有用户和访问序号；此接口不包含调度字段。
uid = generator.users[0]
first = generator.for_user(uid, visit_index=0)
revisit = generator.for_user(uid, visit_index=1)
```

初始化函数读取所选热度资源、可选标题和 tokenizer，返回可复用的 `InputGenerator`，不写文件。路径参数 `data_root`、`heat_path`、`text_catalog_path`、`tokenizer` 接受字符串或 `Path`；`tokenizer` 也接受已加载的 `tokenizers.Tokenizer` 对象（会关闭其 padding 和 truncation）。默认 `heat_source="curve"`、`text_material="synthetic"`；使用真实商品标题需显式设置 `text_material="catalog"`。

`iter_generate(count)` 返回惰性迭代器，不会一次保存全部请求。每次调用都会重新开始一条可复现的流；需要连续消费时保留同一个迭代器。`for_user(uid, item_variant=i)` 可指定独立于 user 的候选内容流；同一 seed、item 长度、visit 与 `item_variant` 下，不同 user 得到相同候选区块。省略时保留原来的 per-user 候选流。`for_user` 的 `visit_index` 由调用方维护，同一用户和访问序号会得到相同内容。已有内存热度和标题时，可直接构造 `InputGenerator(population, tokenizer, titles=..., text_config=..., schedule_config=...)`，其中 `population` 为 `HeatPopulation`。直接构造时也默认 `model="nosa"`。

## 直接运行

```bash
.venv/bin/python -m GR.input_generator \
  --heat-source curve --curve-dataset beauty \
  --num-users 1000 \
  --text-material synthetic \
  --user-lengths 4096 16384 \
  --item-lengths 128 256 512 1024 2048 4096 \
  --count 1000 --qps 100 --arrival poisson \
  --output GR/generated/serving_requests.jsonl
```

请求文本由规则组织。`--text-material catalog` 从已有数据集提取商品标题，使用模板生成浏览、比较、收藏、购买等记录和商品描述；这些记录、属性及候选选择是合成的，不代表真实用户行为，也不提供真实下一商品标签。标题过长时按完整单词缩短到最多 24 tokens，无法保留单词时使用商品编号标签。

默认的 `--text-material synthetic` 不需要任何数据集标题，商品名称也由代码规则生成。两种素材模式均保留英文可读文本。

## 1. 请求内容和长度

默认 NOSA 布局为（固定指令也在 user message 内）：

```text
<|im_start|>user
固定指令
User history:
User profile U42.
Record 1: Viewed P123, Blue travel bag. Preference: easy to clean.
...
Candidate pool for visit 0:
(A) P456: Silver water bottle.
...
Details for (A): designed for travel.
...
<|im_end|>
<|im_start|>assistant
<think>

</think>
```

- `user-lengths`：历史区块长度，包含 `User profile` 和所有历史记录；NOSA 默认 4K/16K（等概率），DeepSeek 默认 4K/16K/64K/256K/1024K，1K = 1024 tokens。
- `item-lengths`：**整个请求除历史区块之外的总预算**，包括固定指令、历史标题、候选、分隔符和 chat 特殊 token；NOSA 默认 128/256/512/1K/2K/4K（等概率），DeepSeek 还包含 64。
- 因此 `total_input_tokens = user_tokens + item_tokens`，没有漏算模板开销。
- 不指定概率时，各长度档位等概率。例如 NOSA 可指定 `--user-probabilities 0.3 0.7`、`--item-probabilities 0.1 0.1 0.1 0.2 0.2 0.3`；数量须匹配，概率之和须为 1。
- 两个长度均在用户初始化时抽取一次，独立于热度；复访只更新候选内容，保持长度不变。同一个 seed 和用户 ID 可以重新生成同一份历史。
- `--candidate-count` 默认 20，是候选商品数量上限；短预算下自动减少完整候选条目，为收尾预留 NOSA 8 tokens / DeepSeek 2 tokens，最少保留 1 个。若单个 catalog 标题仍超预算，按完整单词缩短，必要时使用简短商品标签。长预算由更多完整描述句填充，不靠增加候选数凑长度。
- 历史用不同编号和商品的完整记录填充；最后的小额 token 余量用简短完整句子补齐。不会用随机 token ID 或截断半个 Unicode 字符填充预算。
- 每条完整文本都重新编码，并检查完整编码等于前缀、历史、候选各块编码的拼接。NOSA 的 SentencePiece 会给独立文本块加 dummy prefix，因此历史和候选按实际换行上下文编码；完整 prompt 仍按原始 tokenizer 编码，不修改 tokenizer。边界发生合并时明确报错。NOSA 的解码文本可能多出特殊 token 附近的空格，因此校验原 prompt 的编码，不要求 decode 文本逐字相等。
- `--max-input-tokens` 对 NOSA 默认 32768，对 DeepSeek 默认 1052672（1024K history + 4K item）。NOSA 输入不能超过当前支持的 32768-token context；实际推理还需为输出预留空间。所有档位组合必须在输入预算内；mandatory 候选文本过长时明确报错，不临时裁剪历史。NOSA 的模板开销较大，默认去掉了无法容纳完整候选的 64-token item 档。

生成的 `prompt` 已含完整聊天模板，`input_ids` 可直接交给模型，避免再次套用模板或添加 BOS。

为便于观察跨用户复用，历史开头带稳定用户编号；不同用户仍可能共享指令及一小段模板前缀，不能把 `user_id` 当成 KV 命中的充分条件。

## 2. 用户热度

`--heat-source curve|beauty|games|books|clothing|industrial`，默认 `curve`。

曲线模式仅读取 [heat_curves.csv](analysis/heat_curves.csv)，不读取 provenance 中的原始文件路径：

- `--curve-dataset` 默认 `beauty`；仓库曲线还包含 `games`、`books`、`clothing`、`industrial_100k` 和 `industrial_10M`。
- `--curve-field` 默认对 Amazon 使用 `interaction_count`，对 `industrial_*` 使用 `pv_share`；工业曲线也可选择 `pv_int` 或 `pv_scaled_1_100`。此参数独立于原始 CSV 模式的 `--industrial-heat-field`。
- `--heat-path PATH` 可覆盖曲线 CSV；默认路径相对于 `GR/heat.py` 定位，不受 `--data-root` 影响。自定义文件须有 `dataset`、`field`、`user_fraction`、`traffic_fraction` 列；所选曲线须按用户比例递增、累计流量严格递增，以 `(1, 1)` 结束，且符合热度降序的凹曲线。起点 `(0, 0)` 可省略。
- `--num-users N` 是生成的用户数量，默认 1000，必须大于 0，可以超过原始人口数。对补上原点的累计曲线 `C(x)` 分段线性插值，以 `C(i/N) - C((i-1)/N)` 作为访问权重；用 seed 将这些权重随机分配给合成 ID `0..N-1`。
- 同一曲线、用户数和 seed 可复现；改变 seed 只改变热度到 ID 的分配，不改变热度分布。降采样曲线和人口规模变化会带来近似误差，不能精确恢复所有分位数、Gini、有效用户数或原始用户身份。生成的是总体曲线的近似，不是对原始用户抽取子集。
- Python 可直接调用 `HeatPopulation.from_curve(dataset="beauty", field="interaction_count", num_users=1000, seed=42)`；工厂函数对应参数为 `curve_dataset`、`curve_field`。

例如，使用工业曲线生成合成热度和文本：

```bash
.venv/bin/python -m GR.input_generator \
  --curve-dataset industrial_100k --curve-field pv_share \
  --num-users 1000 --user-lengths 4096 --item-lengths 1024 \
  --count 100 --output GR/generated/curve_requests.jsonl
```

原始文件入口保留，显式指定 `--heat-source beauty|games|books|clothing|industrial` 即可。来源是用户身份和热度，完全不需要该用户的真实历史、候选或文本映射：

- Amazon 数据集：全量累加 `timestep_map.json` 中用户的交互次数，作为活跃度代理。
- industrial：直接使用 CSV 中的用户 ID 和 `--industrial-heat-field`，支持 `pv_share`（默认）、`pv_int`、`pv_scaled_1_100`。不使用其 `tokens` 列。
- 原始文件模式下，`--num-users 1000`（默认）从全文件均匀、无放回选用户，再在子集内归一化权重。不会直接截取按热度排序的 CSV 前 N 行。仅此模式支持 `--num-users 0` 使用全部用户；超过源用户数时报错。
- `--heat-path PATH`：覆盖所选数据集的时间映射或 industrial CSV 路径。工业数据支持流式蓄水池抽样；选取全部 10M 用户时仍需为其采样权重和调度状态预留内存。
- `--sampling weighted`（默认）：按归一化权重有放回抽样。也可用 `uniform` 或 `sequential` 做对照；热度概率仅在 weighted 模式实际使用。

不保证每个用户访问固定次数；需要复访时，应设置足够多请求或减少活跃用户数量。热度分布与 industrial 字段差异见 [分析记录](analysis/README.md)。

## 3. 时间分布

- `--arrival poisson`（默认）：指数分布到达间隔，平均 QPS 由 `--qps` 决定。
- `--arrival constant`：固定间隔 `1 / qps` 秒，第一个请求在 0 秒。
- `--arrival timeslot`：每秒最多 qps 条，并使用参考脚本的 5–65 秒用户冷却间隔分布。无用户可用时跳到下一可用时刻；该约束会改变实际用户占比，不能把它与独立 weighted 抽样等同。

这里生成的是模拟时间戳，不会按墙钟时间睡眠或发送请求。`iter_generate` 每次调用重置随机状态、时钟和复访计数。

## 输出和复现

每行含 `prompt`、`input_ids`、`attention_mask`，以及：

| 字段 | 含义 |
|---|---|
| `model` | 请求格式：`nosa` 或 `deepseek_v32` |
| `user_id`, `user_heat_weight` | 曲线模式为合成 ID 和曲线流量增量；原始模式为源用户 ID 和所选字段的权重 |
| `task_id`, `timestamp`, `visit_index` | 请求序号、模拟秒数、从 0 开始的用户访问序号 |
| `previous_request_id`, `revisit_interval_s` | 上一次访问及间隔，首访为 null |
| `user_tokens`, `item_tokens`, `total_input_tokens` | 精确长度预算及实际完整输入长度 |
| `instruction_tokens`, `candidate_suffix_tokens` | item 预算的前后两部分 |
| `history_token_span`, `candidate_token_span` | `[start, end)`；候选区间包含标题、描述和 assistant 提示 |
| `stable_prefix_tokens`, `history_sha256` | 指令 + 固定历史的长度，以及历史文本摘要 |
| `common_prefix_tokens` | 与该用户上一请求比较得到的实际共同 token 前缀长度；首访为 0，并非缓存实际命中量 |
| `candidate_item_ids` | 本次候选；catalog 模式来自素材库，synthetic 模式为合成编号 |
| `content_is_synthetic`, `target_item_id` | true、null：合成请求，无真实目标标签 |

`.users.jsonl` 保存每个用户的热度、归一化概率及固定长度；`.meta.json`（schema version 5）保存模型、热度来源、文本来源、tokenizer、长度概率和时间配置。曲线模式的 `heat` 还记录曲线文件路径和 SHA-256、dataset/field、插值方法、seed 与合成身份说明。使用同一源文件、tokenizer、参数和随机种子可复现。默认只缓存最近 8 个用户的历史文本，用 `--history-cache-users` 调整，避免为所有用户常驻最长 1024K 文本；淘汰后可确定性重建。

默认资源：

```text
热度曲线：GR/analysis/heat_curves.csv（相对于仓库，运行时按模块路径定位）
原始数据根目录（仅原始热度或 catalog 模式使用）：/mnt/nfs/share/archive/260808_old_nfs/share/wsh/Bi-KV/Bi-KV/data
NOSA tokenizer：/mnt/ssd-wlcb/chenkaiqi/NOSA-8B/tokenizer.json
DeepSeek tokenizer：/mnt/nfs/share/models/DeepSeek-V3.2/tokenizer.json
```

可用 `--data-root`、`--tokenizer`、`--text-catalog-path` 覆盖。`--tokenizer` 接受 JSON 文件或目录，须与 `--model` 匹配。DeepSeek 可用 `--model deepseek_v32 --tokenizer weights/DeepSeek-V3.2`。`TextConfig()` 本身采用 NOSA 默认值；显式配置 DeepSeek 长上下文时应同时设置 `max_input_tokens`。未传 `text_config` 时，工厂和构造函数自动采用所选模型的预算。Python API 为 `GR.input_generator.InputGenerator`、`TextConfig`、`GR.scheduling.ScheduleConfig` 和 `GR.heat.HeatPopulation`。

```bash
.venv/bin/python -m unittest discover -s GR/tests -v
```

测试包含默认长度组合、精确编码、复访稳定前缀和候选变化、共同前缀计算、热度载入，以及曲线插值、复现、输入校验、summary 集中度对照和 API/CLI 默认资源加载。已在本地 NOSA tokenizer 上验证默认 12 种长度组合、模板与 checkpoint 一致、精确编码和复访；DeepSeek 适配保留原模板与 35 种长度组合测试，本机缺少该 tokenizer，当前跳过。

## 三层内容交叉实验

这些历史实验显式选择 `deepseek_v32`，保留原有模板与长度档位。`experiments.sweep_gr_content_matrix` 使用 5 个 history 长度 × 7 个 new 长度 × 3 份 history × 3 份 item，共 315 份输入，各测第 0/1/2 层，共 945 组层级结果。每个 history 长度下的三份内容固定；同一 item 长度的三份候选内容在不同 history 间复用，通过 SHA-256 校验独立组合。

测量边界沿用 KV 实验口径：history 包含 23-token 固定指令，new 是完整候选后缀。因此测量配置将 generator 的 user 预算设为 `history - instruction_tokens`，item 预算设为 `new + instruction_tokens`，总 token 数和实际 KV 边界精确匹配。

```bash
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m experiments.sweep_gr_content_matrix
.venv/bin/python -m experiments.report_gr_content_matrix
```

结果和回放索引在 `GR/generated/content_matrix/`，报告见 [三层 KV 命中](../docs/extend_step_profile/gr_multilayer_kv_hits.md)。
