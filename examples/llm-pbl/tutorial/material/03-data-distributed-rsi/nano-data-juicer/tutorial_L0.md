# nano-data-juicer · L0 教程：OP 抽象 + 配置驱动 pipeline

> **本节目标（L0）**：用 ~150 行纯 Python 抓住 Data-Juicer 的两个核心机制——
> ① 把数据处理抽象成**可组合算子（OP）**；② 用**配置**驱动 pipeline。
> **前置**：无。**本节 K+1**：从「写死的数据清洗脚本」到「接口统一 + 配置驱动」。

---

## 1. 问题：数据清洗脚本为什么会烂掉

几乎每个做 LLM 数据的人都写过这样的脚本：

```python
data = load("raw.jsonl")
tmp = []
for d in data:
    d["text"] = d["text"].strip()          # 清洗 1
    if len(d["text"]) < 100: continue      # 过滤 2
    if d["text"] in seen: continue         # 去重 3
    seen.add(d["text"])
    tmp.append(d)
data = tmp
# ……三周后再加一个「过滤非中文」，只能继续往循环里塞 if
```

它一开始能跑，但很快会烂掉：

- **耦合**：所有逻辑挤在同一个循环里，改一处怕坏另一处；
- **不可复用**：下一个项目需要同样的去重逻辑，只能复制粘贴；
- **不可配置**：换一个阈值、调换两步顺序，都要改代码；
- **不可观测**：每一步剩多少条、哪一步砍得最狠，全靠加 print。

Data-Juicer 对这个问题的回答是两条抽象：**OP（算子）**与**配置驱动**。
本节用最少的代码把这两条抽象的内核做出来。

---

## 2. 运行

文件：`L0_toy_ops.py`，纯标准库，零外部依赖，CPU 即跑。

```bash
$ python3 L0_toy_ops.py
```

真实输出：

```
[input] 6 samples
[lowercase_mapper] -> 6 samples
[length_filter] -> 4 samples
[deduplicator] -> 2 samples

[final]
   {'text': 'hello world', 'id': 1}
   {'text': 'data-juicer is a data system', 'id': 4}
```

**L0 基线指标（toy metric）**：漏斗 `6 → 6 → 4 → 2`——
mapper 不改条数，filter 砍掉 2 条短文本，dedup 再砍掉 2 条重复。
后续级别都以这个漏斗为对照（L1 是 `10 → 10 → 7 → 7`）。

先盯着输出里每步的条数变化，下面逐个拆开。

---

## 3. 核心抽象一：OP = list[Sample] → list[Sample]

一个样本就是一个 dict；一个 OP 就是一个「吃样本列表、吐样本列表」的 callable：

```python
Sample = Dict[str, Any]
OP = Callable[[List[Sample]], List[Sample]]
```

就这么一行类型签名，是整个体系的地基。三种最典型的 OP：

### 3.1 Filter：按谓词决定去留（条数可变）

```python
def make_length_filter(min_chars: int) -> OP:
    def op(samples: List[Sample]) -> List[Sample]:
        return [s for s in samples if len(s.get("text", "")) >= min_chars]
    return op
```

注意 `make_length_filter(min_chars)` 返回的是一个 OP——**参数通过闭包注入，
OP 本身保持无参的 `list → list` 签名**。这样 pipeline 串联时才不需要关心每个 OP 的参数。

### 3.2 Mapper：逐样本变换（条数不变）

```python
def make_lowercase_mapper() -> OP:
    def op(samples: List[Sample]) -> List[Sample]:
        out = []
        for s in samples:
            s = dict(s)  # 不修改原样本，避免副作用
            s["text"] = s.get("text", "").strip().lower()
            out.append(s)
        return out
    return op
```

`s = dict(s)` 这个浅拷贝很重要：**OP 不应该原地修改输入**（思考题 2）。

### 3.3 Deduplicator：按键去重（条数减少）

```python
def make_deduplicator(key: str = "text") -> OP:
    def op(samples: List[Sample]) -> List[Sample]:
        seen = set()
        out = []
        for s in samples:
            k = s.get(key)
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out
    return op
```

这三个 OP 对应真实 Data-Juicer 三大类算子的最小内核。
真实系统里还有 Selector（按分数/排名挑选）、Grouper/Aggregator（分组聚合，
用于多轮对话等场景）、Formatter（数据加载）——但它们的共同点仍是同一个接口思想。

---

## 4. 核心抽象二：配置驱动 pipeline

有了统一接口，pipeline 就只是「按顺序复合」：

```python
@dataclass
class Pipeline:
    steps: List[tuple]  # [(name, OP), ...]

    def run(self, samples: List[Sample]) -> List[Sample]:
        cur = samples
        print(f"[input] {len(cur)} samples")
        for name, op in self.steps:
            cur = op(cur)
            print(f"[{name}] -> {len(cur)} samples")
        return cur
```

`Pipeline.run` 不知道也不关心每个 OP 内部做什么——它只负责串联和观测（每步打印条数，
这就是「可观测」的起点）。**主流程从此稳定**：以后加任何新 OP，这行代码都不用改。

那 OP 从哪来？从**配置**来：

```python
def build_pipeline_from_config(config: List[Dict[str, Any]]) -> Pipeline:
    registry = {
        "length_filter": lambda cfg: make_length_filter(cfg.get("min_chars", 1)),
        "lowercase_mapper": lambda cfg: make_lowercase_mapper(),
        "deduplicator": lambda cfg: make_deduplicator(cfg.get("key", "text")),
    }
    steps = []
    for item in config:
        name, cfg = item["op"], item.get("params", {})
        if name not in registry:
            raise KeyError(f"unknown op: {name} (available: {list(registry)})")
        steps.append((name, registry[name](cfg)))
    return Pipeline(steps)
```

两个要点：

- **registry（注册表）**：op 名字 → 构造函数。用户写配置，代码按名字实例化。
  未注册的名字直接报错，而不是静默跳过——配置驱动系统的第一坑就是
  「拼错 op 名字，pipeline 照样跑完，结果悄悄错了」。
- **配置即声明**：用户只描述「要哪些 OP、什么参数、什么顺序」，不写流程代码。

于是整个数据清洗流程变成一份纯数据：

```python
config = [
    {"op": "lowercase_mapper"},
    {"op": "length_filter", "params": {"min_chars": 6}},
    {"op": "deduplicator", "params": {"key": "text"}},
]
pipeline = build_pipeline_from_config(config)
result = pipeline.run(data)
```

这份 list 可以原样换成 yaml/json，可以从实验配置管理系统读进来，
可以做版本管理、diff、code review——**流程从代码变成了数据**。

---

## 5. 顺序是语义的一部分：一次真实的翻车

配置驱动意味着「顺序由配置决定」，而顺序不是中性的。
toy 数据里 id=1 是 `"Hello World"`，id=3 是 `"hello world"`——
大小写不同、规范化后相同。我们把 dedup 挪到 lowercase 前面跑一次（真实运行）：

```python
bad_config = [
    {"op": "deduplicator", "params": {"key": "text"}},
    {"op": "lowercase_mapper"},
    {"op": "length_filter", "params": {"min_chars": 6}},
]
```

```
=== 反例：先 dedup 再 lowercase ===
[input] 6 samples
[deduplicator] -> 5 samples
[lowercase_mapper] -> 5 samples
[length_filter] -> 3 samples
final: [1, 3, 4]
```

对比正确顺序的 `final: [1, 4]`：id=3 漏网了。
原因：dedup 在规范化之前执行，`"Hello World"` 和 `"hello world"` 当时还不是同一个 key。

教训：**OP 的组合顺序是 pipeline 语义的一部分**——哪个 OP 会改变后续 OP 读取的字段，
谁就必须先想清楚。这也是真实数据管线要做「OP 顺序审查」的原因，
不是代码 bug，是配置 bug，而配置 bug 往往悄无声息。

---

## 6. 与真实 Data-Juicer 的对应（概念层）

以下路径均来自本地 checkout `${DATA_JUICER_REPO}`
（行号以当前本地 checkout 为准，升级版本后可能漂移）：

| nano 实现 | Data-Juicer 对应 | 说明 |
|-----------|-----------------|------|
| `OP = list[Sample] → list[Sample]` | `data_juicer/ops/base_op.py` 的 `class OP`（L289）及其子类 `Mapper`(L555) / `Filter`(L666) / `Deduplicator`(L816) / `Selector`(L882) / `Grouper`(L924) / `Aggregator`(L969) | DJ 的 OP 是类而非闭包，带 schema 校验、tracer、异常容忍等装饰；接口思想相同 |
| `build_pipeline_from_config` 的 registry | `data_juicer/ops/load.py` 的 `load_ops(process_list, ...)` | DJ 按配置列表实例化全部 OP |
| `config = [{"op": ..., "params": ...}]` | `data_juicer/config/config_all.yaml` 的 `process:` 段（L103 起），形如 `- op_name: {参数...}` | nano 版是 dict list，DJ 是 yaml，表达力等价 |
| `Pipeline.run` 顺序执行 + 打印条数 | `data_juicer/core/executor/default_executor.py`（入口）+ `factory.py`（工厂） | DJ 另有 `ray_executor.py` 做分布式——那是 L2 的事 |

L0 抓住了「OP 接口统一 + 配置驱动」这个骨架。真实的 200+ 算子
（跨模态总数，text-only 精确子集待核 `[TODO: verify text-subset counts]`）
都只是同一套接口的不同实现——读懂本节，就读懂了它们的骨架。

---

## 7. 费曼：讲给外行听

**类比：乐高积木。**

每个 OP 是一块乐高积木：不管这块积木是「轮子」还是「窗户」
（filter 还是 mapper），它的凸起和凹槽（输入输出接口）形状完全一样——
都是「一盒样本进，一盒样本出」。所以你可以随意增减积木、调换顺序，
不需要重新设计整条拼装线（`Pipeline.run` 永远不用改）。

而配置清单就是拼装说明书：「先装这块、再装那块、那块要拧两圈
（`params: {"min_chars": 6}`）」。换一份说明书，同一条流水线就变成
另一套清洗流程——流水线本身一行不用动。

一句话版本：**OP 把「怎么处理」封装成积木，配置把「处理什么顺序」变成说明书。**

---

## 8. 思考题

1. 现在要加一个「按语言过滤」的 `language_filter`。你需要修改 `Pipeline.run` 吗？
   需要动哪几处？（答案：不改主流程；只写 `make_language_filter` 并注册进 registry。
   这正是可组合性的价值——README 费曼自检里也问了这一题。）

2. `make_lowercase_mapper` 里为什么先 `s = dict(s)` 再改？如果直接 `s["text"] = ...`
   原地修改，会出什么事？（提示：原始数据被污染；同一份 data 想换一套配置重跑时，
   第二次的输入已经不是原始数据了；「可复现实验」悄悄失效。）

3. `make_deduplicator` 保留**首次**出现。什么场景下「保留最后一次」才是对的？
   （提示：同一题目后来被人工订正过、或高分版本应覆盖低分版本时。
   去重策略不是中性的，它隐含了「哪个版本更可信」的判断。）

---

## 9. 反例：OP 化不是免费的

**「把一切都拆成 OP，pipeline 越长越好」——错。**

每多一个 OP 都多一份遍历与拷贝开销，更重要的是每个 filter 都在做
「质量 vs 数量」的取舍：阈值拧得太紧会误杀好数据，拧得太松等于没洗。
L0 的 `min_chars: 6` 在 toy 数据上显然合理，但换一个数据集，
没有任何先验能保证同样的阈值还成立。**OP 化让每一步可审计、可观测——
但每一步的阈值是否合理，仍要靠人看数据分布来定**（这也是真实
Data-Juicer 提供 tracer / 统计画像的原因）。

---

## 10. 下一步 L1

L1 在两个方向加码（见 `tutorial_L1.md`）：

1. **真实数据**：从 6 条内联 dict 换成 10 条真实医学 SFT 样本（嵌套 messages 格式），
   引入 Formatter（文件加载）与「展平」Mapper；
2. **LLM-based OP**：规则抓不住「推理质量」，引入调 LLM API 的打分 OP——
   注意它的接口与 rule-based OP 完全一样，仍是 `list[Sample] → list[Sample]`。

核心不变量在 L1 会被再次验证：**不管 OP 内部是正则还是 LLM API，
pipeline 看到的都是同一个接口**。

---

## 11. 溯源

- 运行输出来自本机真实执行：`python3 L0_toy_ops.py`
  （漏斗 6→6→4→2）；反例输出为同一模块真实运行（`final: [1, 3, 4]`）。
- Data-Juicer 源码引用均来自本地 checkout `${DATA_JUICER_REPO}`：
  `data_juicer/ops/base_op.py`、`data_juicer/ops/load.py`、
  `data_juicer/config/config_all.yaml`、`data_juicer/core/executor/`
  （`default_executor.py` / `factory.py` / `ray_executor.py`）。
  行号为当前 checkout 实测；源码级接口细节（schema / tracer / batch）留到 L3 对照。
- 仓库：<https://github.com/modelscope/data-juicer>。
