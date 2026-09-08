# GR serving 请求生成器

生成用于 KV cache 实验的可读请求：**用户历史固定、候选每次更新、token 长度精确控制**。用户热度、文本素材、到达时间分别配置。完整文本经本地 DeepSeek V3.2 tokenizer 编码，可直接通过 Python 接口获取，也可用命令行输出 JSONL，不加载模型或运行 serving。

## Python 接口

```python
from GR.input_generator import create_input_generator, TextConfig
from GR.scheduling import ScheduleConfig

generator = create_input_generator(
    heat_source="industrial",
    industrial_heat_field="pv_share",
    num_users=1000,
    text_material="catalog",
    text_dataset="beauty",
    text_config=TextConfig(
        user_lengths=(4096, 16384, 65536),
        user_probabilities=(0.3, 0.4, 0.3),
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

初始化函数读取热度、标题和 tokenizer，返回可复用的 `InputGenerator`，不写文件。路径参数 `data_root`、`heat_path`、`text_catalog_path`、`tokenizer` 接受字符串或 `Path`；`tokenizer` 也接受已加载的 `tokenizers.Tokenizer` 对象（会关闭其 padding 和 truncation）。纯规则商品名使用 `text_material="synthetic"`。

`iter_generate(count)` 返回惰性迭代器，不会一次保存全部请求。每次调用都会重新开始一条可复现的流；需要连续消费时保留同一个迭代器。`for_user` 的 `visit_index` 由调用方维护，同一用户和访问序号会得到相同内容。已有内存热度和标题时，可直接构造 `InputGenerator(population, tokenizer, titles=..., text_config=..., schedule_config=...)`，其中 `population` 为 `HeatPopulation`。

## 直接运行

```bash
.venv/bin/python -m GR.input_generator \
  --heat-source industrial --industrial-heat-field pv_share \
  --num-users 1000 \
  --text-material catalog --text-dataset beauty \
  --user-lengths 4096 8192 16384 32768 65536 \
  --item-lengths 1024 2048 4096 \
  --count 1000 --qps 100 --arrival poisson \
  --output GR/generated/serving_requests.jsonl
```

请求文本由规则组织。`--text-material catalog`（默认）从已有数据集提取商品标题，使用模板生成浏览、比较、收藏、购买等记录和商品描述；这些记录、属性及候选选择是合成的，不代表真实用户行为，也不提供真实下一商品标签。标题过长时按完整单词缩短到最多 24 tokens，无法保留单词时使用商品编号标签。

若不需要任何数据集标题，使用 `--text-material synthetic`，商品名称也由代码规则生成。两种素材模式均保留英文可读文本。

## 1. 请求内容和长度

实际布局为：

```text
<bos>固定指令<User>User history:
User profile U42.
Record 1: Viewed P123, Blue travel bag. Preference: easy to clean.
...
Candidate pool for visit 0:
(A) P456: Silver water bottle.
...
Details for (A): designed for travel.
...
<Assistant></think>
```

- `user-lengths`：历史区块长度，包含 `User profile` 和所有历史记录；默认 4K/8K/16K/32K/64K，1K = 1024 tokens。
- `item-lengths`：**整个请求除历史区块之外的总预算**，包括固定指令、历史标题、候选、分隔符和 chat 特殊 token；默认 1K/2K/4K。
- 因此 `total_input_tokens = user_tokens + item_tokens`，没有漏算模板开销。
- 不指定概率时，各长度档位等概率。可指定 `--user-probabilities 0.1 0.2 0.3 0.3 0.1`、`--item-probabilities 0.2 0.5 0.3`；数量须匹配，概率之和须为 1。
- 两个长度均在用户初始化时抽取一次，独立于热度；复访只更新候选内容，保持长度不变。同一个 seed 和用户 ID 可以重新生成同一份历史。
- `--candidate-count` 默认 20，决定候选商品数量。长预算由更多完整描述句填充，不靠增加候选数凑长度。
- 历史用不同编号和商品的完整记录填充；最后的小额 token 余量用简短完整句子补齐。不会用随机 token ID 或截断半个 Unicode 字符填充预算。
- 每条完整文本都重新编码，并检查完整编码等于指令、历史、候选各块编码的拼接。若 tokenizer 在区块边界发生合并则报错，不宣称虚假的精确长度或稳定前缀。已针对本地 DeepSeek tokenizer 验证全部默认档位。
- `--max-input-tokens` 默认 131072。所有配置档位组合必须在预算内；mandatory 候选文本过长时也明确报错。不会根据候选长度临时裁剪用户历史。

为便于观察跨用户复用，历史开头带稳定用户编号；不同用户仍可能共享指令及一小段模板前缀，不能把 `user_id` 当成 KV 命中的充分条件。

## 2. 用户热度

`--heat-source beauty|games|books|clothing|industrial`，默认 Beauty。来源是用户身份和热度，完全不需要该用户的真实历史、候选或文本映射。

- Amazon 数据集：全量累加 `timestep_map.json` 中用户的交互次数，作为活跃度代理。
- industrial：直接使用 CSV 中的用户 ID 和 `--industrial-heat-field`，支持 `pv_share`（默认）、`pv_int`、`pv_scaled_1_100`。不使用其 `tokens` 列。
- `--num-users 1000`（默认）：从全文件均匀、无放回选用户，再在子集内归一化权重。不会直接截取按热度排序的 CSV 前 N 行。`--num-users 0` 使用全部用户；超过源用户数时报错。
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
| `user_id`, `user_heat_weight` | 热度源原始用户 ID 和所选字段的权重 |
| `task_id`, `timestamp`, `visit_index` | 请求序号、模拟秒数、从 0 开始的用户访问序号 |
| `previous_request_id`, `revisit_interval_s` | 上一次访问及间隔，首访为 null |
| `user_tokens`, `item_tokens`, `total_input_tokens` | 精确长度预算及实际完整输入长度 |
| `instruction_tokens`, `candidate_suffix_tokens` | item 预算的前后两部分 |
| `history_token_span`, `candidate_token_span` | `[start, end)`；候选区间包含标题、描述和 assistant 提示 |
| `stable_prefix_tokens`, `history_sha256` | 指令 + 固定历史的长度，以及历史文本摘要 |
| `common_prefix_tokens` | 与该用户上一请求比较得到的实际共同 token 前缀长度；首访为 0，并非缓存实际命中量 |
| `candidate_item_ids` | 本次候选；catalog 模式来自素材库，synthetic 模式为合成编号 |
| `content_is_synthetic`, `target_item_id` | true、null：合成请求，无真实目标标签 |

`.users.jsonl` 保存每个用户的热度、归一化概率及固定长度；`.meta.json` 保存热度来源、文本来源、tokenizer、长度概率和时间配置。使用同一源文件、tokenizer、参数和随机种子可复现。默认只缓存最近 8 个用户的历史文本，用 `--history-cache-users` 调整，避免为所有用户常驻 64K 文本；淘汰后可确定性重建。

默认资源：

```text
数据根目录：/mnt/nfs/share/archive/260808_old_nfs/share/wsh/Bi-KV/Bi-KV/data
Tokenizer：/mnt/nfs/share/models/DeepSeek-V3.2/tokenizer.json
```

可用 `--data-root`、`--tokenizer`、`--text-catalog-path` 覆盖。本地 tokenizer 副本也可用 `--tokenizer models/DeepSeek-V3.2`。Python API 为 `GR.input_generator.InputGenerator`、`TextConfig`、`GR.scheduling.ScheduleConfig` 和 `GR.heat.HeatPopulation`。

```bash
.venv/bin/python -m unittest discover -s GR/tests -v
```

测试包含全套 15 种默认长度组合、精确编码、复访稳定前缀和候选变化、共同前缀计算、热度载入。本地 DeepSeek tokenizer 不存在时跳过真实 tokenizer 测试。
