# nano-data-juicer · L1 教程：真实数据 + LLM-based OP

> **前置**：先跑通 `L0_toy_ops.py`，理解「OP = list[Sample] → list[Sample]」和「配置驱动 pipeline」。
> **本节 K+1**：从 toy 数据升级到真实 SFT 数据，从纯规则 OP 升级到 LLM-based OP。

---

## 1. 问题：L0 的 toy 数据太简单了

L0 用 6 条内联 dict 演示了 OP 可组合性。但真实训练数据长这样：

```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "请直接回答以下单项选择题目..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "<think>...\n\\boxed{C}"}]}]}
```

这是一份真实的医学 SFT 样本（10 条，本地文件 `Data-Training-Router/ProjectZ/examples/medical/sft_data/train.jsonl`，仅作真实小样本使用）。
结构是嵌套的 messages 格式，不是扁平的 `{"text": "..."}`。

L1 要解决三个新问题：
1. 怎么从文件加载真实数据（Formatter）
2. 怎么处理嵌套结构（extract_fields_mapper）
3. 怎么引入 LLM 做质量评估（llm-based OP）

---

## 2. 加载真实数据：Formatter

L0 的数据是代码里写死的 list。L1 从 JSONL 文件读：

```python
def load_jsonl(path: Path) -> List[Sample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_id"] = i  # 保留行号，方便溯源
            samples.append(obj)
    return samples
```

注意 `_line_id`：真实 pipeline 里你需要知道「哪条数据被过滤了」，行号是最简单的溯源手段。
Data-Juicer 真实实现里用 Arrow 的 row index + metadata 做这件事，原理相同。

---

## 3. 处理嵌套结构：extract_fields_mapper

真实 SFT 数据的文本藏在 `messages[i].content[j].text` 里。后续 OP 不想每次都解析这层嵌套，
所以第一个 OP 负责「展平」：

```python
def make_extract_fields_mapper() -> OP:
    def op(samples: List[Sample]) -> List[Sample]:
        out = []
        for s in samples:
            s = dict(s)
            messages = s.get("messages", [])
            # ... 提取 question_text, answer_text, thinking_text ...
            think_match = re.search(r"<think>(.*?)</think>", assistant_msg, re.DOTALL)
            s["thinking_text"] = think_match.group(1).strip() if think_match else ""
            s["thinking_chars"] = len(s["thinking_text"])
            out.append(s)
        return out
    return op
```

这是一个 **Mapper**：不改变条数，只给每个样本附加新字段。
设计原则：「展平」OP 放 pipeline 最前面，后续 OP 只读扁平字段，不碰原始嵌套结构。

---

## 4. Rule-based 质量过滤

两个便宜的规则 OP：

**answer_format_filter**：最终回答必须含 `\boxed{A}` 这类格式。如果 SFT 数据的回答格式不对，
说明生成过程有 bug，不应进入训练集。

```python
pattern = re.compile(r"\\boxed\{[A-E]\}")
def op(samples):
    return [s for s in samples if pattern.search(s.get("final_answer", ""))]
```

**thinking_length_filter**：推理链太短（< 4500 字符）的样本，推理可能不充分。

```python
def op(samples):
    return [s for s in samples if s.get("thinking_chars", 0) >= min_chars]
```

这两个 OP 都是 CPU 微秒级操作。10 条数据瞬间完成。

---

## 5. LLM-based OP：质量打分（L1 核心）

规则能抓格式问题，但抓不了「推理质量」。比如：
- 推理链很长但逻辑混乱
- 引用了错误的医学知识
- 推理和最终答案矛盾

这些需要「理解」能力——正是 LLM 擅长的。

### 5.1 接口设计

LLM-based OP 的接口和 rule-based OP **完全一样**：`list[Sample] → list[Sample]`。
区别只在内部实现：对每条数据调用一次 LLM API。

```python
def make_llm_quality_scorer(api_key, base_url, model, threshold=3) -> OP:
    def op(samples: List[Sample]) -> List[Sample]:
        out = []
        for s in samples:
            prompt = PROMPT_TEMPLATE.format(
                question=s["question_text"][:200],
                thinking=s["thinking_text"][:500],
                answer=s["final_answer"],
            )
            response = call_llm(prompt, api_key, base_url, model)
            score = int(re.search(r"[1-5]", response).group())
            s = dict(s)
            s["llm_score"] = score
            out.append(s)
        return [s for s in out if s["llm_score"] >= threshold]
    return op
```

关键设计决策：
- **截取前 500 字**：thinking 可能几千字，全发给 LLM 太贵。截取足够评估的窗口。
- **temperature=0**：打分要确定性，不要创造性。
- **rate limit sleep**：真实 API 有 QPS 限制，每条之间等 0.5s。
- **错误容忍**：单条调用失败不中断整个 pipeline，给 score=0 跳过。

### 5.2 成本意识：为什么 rule-based 在前、llm-based 在后

```
config = [
    {"op": "extract_fields_mapper"},        # CPU, 免费
    {"op": "answer_format_filter"},         # CPU, 免费
    {"op": "thinking_length_filter", ...},  # CPU, 免费
    {"op": "llm_quality_scorer", ...},      # 网络, 花钱, 慢
]
```

先用便宜 OP 过滤掉明显不合格的，减少 LLM 调用次数。
本例中 thinking_length_filter 把 10 条砍到 7 条，LLM 只需调 7 次而非 10 次。
数据量大时，前置 rule filter 能显著减少 LLM 调用次数，但具体节省比例取决于数据分布与 filter 命中率，`[TODO: verify]` 需实测或引用内部成本报告。

这与 Data-Juicer 的工程实践一致：在配置中把确定性高、成本低的 rule filter 放在前面，把昂贵的 LLM OP 放在后面（可参见 `data-juicer/core/executor/default_executor.py` 的执行顺序保证）。

---

## 6. 运行与输出

### Mock 模式（无需 API key，演示 pipeline 流程）

```bash
python L1_real_data.py --mock
```

真实输出（注意：mock 警告先打印，然后才加载数据，这样无 API key 时不会白读大文件）：

```
⚠️  MOCK MODE: LLM 打分将由启发式规则代替，非真实 LLM 输出。
    去掉 --mock 并设置 DASHSCOPE_API_KEY 环境变量可启用真实 LLM 打分。

[load] /path/to/train.jsonl
[load] 10 samples loaded

============================================================
[pipeline] input: 10 samples
============================================================
[extract_fields_mapper] -> 10 samples
[answer_format_filter] -> 10 samples
[thinking_length_filter] -> 7 samples
  [mock_scorer] sample 0: thinking_chars=5585, score=5
  [mock_scorer] sample 1: thinking_chars=5086, score=5
  [mock_scorer] sample 2: thinking_chars=6889, score=5
  [mock_scorer] sample 3: thinking_chars=5979, score=5
  [mock_scorer] sample 4: thinking_chars=5172, score=5
  [mock_scorer] sample 5: thinking_chars=6392, score=5
  [mock_scorer] sample 6: thinking_chars=6720, score=5
  [mock_scorer] 7 scored, 7 kept (threshold=3)
  ⚠️  MOCK MODE: 分数由启发式规则生成，非真实 LLM 输出。
[llm_quality_scorer] -> 7 samples
============================================================
[pipeline] output: 7 samples
============================================================

────────────────────────────────────────────────────────────
最终保留 7 条样本：
────────────────────────────────────────────────────────────
  [line 2] score=5(mock_heuristic) thinking=5585chars | 请直接回答以下单项选择题目...
  [line 3] score=5(mock_heuristic) thinking=5086chars | 请直接回答以下单项选择题目...
  [line 4] score=5(mock_heuristic) thinking=6889chars | 请直接回答以下单项选择题目...
  [line 6] score=5(mock_heuristic) thinking=5979chars | 请直接回答以下单项选择题目...
  [line 7] score=5(mock_heuristic) thinking=5172chars | 请直接回答以下单项选择题目...
  [line 8] score=5(mock_heuristic) thinking=6392chars | 请直接回答以下单项选择题目...
  [line 9] score=5(mock_heuristic) thinking=6720chars | 请直接回答以下单项选择题目...

[stats] pipeline 漏斗：
  input: 10
  extract_fields_mapper: 10
  answer_format_filter: 10
  thinking_length_filter: 7
  llm_quality_scorer: 7
```

解读：
- 10 条全部通过格式检查（这批数据格式都正确）
- thinking_length_filter 过滤了 3 条（line 0/1/5，thinking < 4500 chars）
- mock 打分全部 5 分（因为启发式只看长度，> 1000 就给 5）
- 最终保留 7 条

### 真实 LLM 模式

```bash
export DASHSCOPE_API_KEY="CHANGEME"
python L1_real_data.py
```

真实模式下，每条数据会调用 qwen-plus 进行质量评估。LLM 可能给出 3-5 分的差异化评分，
threshold=3 会过滤掉推理质量差的样本。

---

## 7. L0 → L1 对比

| 维度 | L0 | L1 |
|------|----|----|
| 数据 | 6 条内联 dict | 10 条真实 JSONL（医学 SFT 样本） |
| 数据结构 | 扁平 `{"text": ...}` | 嵌套 messages 格式 |
| OP 种类 | 3 个 rule-based | 3 rule-based + 1 llm-based |
| 外部依赖 | 零 | 零（urllib 调 API）|
| 新抽象 | — | Formatter（文件加载）、LLM-based OP |
| pipeline 效果 | 6→4→2 | 10→10→7→7 |

核心不变量：**OP 接口始终是 `list[Sample] → list[Sample]`**。
不管你内部是正则匹配还是调 LLM API，对 pipeline 来说都是同一个接口。
这就是 Data-Juicer 200+ 算子能无缝组合的原因。

---

## 8. 费曼自检

**讲给外行听**：

想象一个工厂流水线。原料（原始数据）进来后，经过一道道工序（OP）：
- 第一道：把原料从包装里拆出来（extract_fields_mapper）
- 第二道：目测检查外观（answer_format_filter）
- 第三道：量尺寸，太小的扔掉（thinking_length_filter）
- 第四道：请老师傅仔细鉴定质量（llm_quality_scorer）

每道工序只管自己的事，不关心前后是谁。你要加一道新工序？
不用改流水线本身，只要在配置里插一步。

**思考题**：

1. 如果把 llm_quality_scorer 放到 thinking_length_filter 前面，功能上对不对？
   经济上呢？（提示：10 次 API 调用 vs 7 次）

2. mock 模式的启发式打分（只看长度）和真实 LLM 打分，本质区别是什么？
   什么情况下「长度」是质量的合理 proxy？什么情况下不是？

3. 如果数据量从 10 条变成 10 万条，当前 L1 的 llm_quality_scorer 有什么问题？
   （提示：串行、单线程、无 batch。L2 会用 Ray 解决这个。）

**反例**：

「LLM-based OP 就是比 rule-based OP 好」——错。
rule-based 确定性高、零成本、可审计。LLM-based 有幻觉风险、有 API 成本、结果不确定。
正确做法：能用规则解决的不用 LLM；LLM 只用在规则抓不了的「需要理解」的维度。

---

## 9. 与真实 Data-Juicer 的对应

| nano 实现 | Data-Juicer 对应 | 差异 |
|-----------|-----------------|------|
| `load_jsonl()` | `data_juicer.format.formatter` | DJ 支持 csv/parquet/arrow 等多格式 |
| `extract_fields_mapper` | `data_juicer.ops.mapper` 系列 | DJ 的 mapper 有统一 schema 校验 |
| `thinking_length_filter` | `data_juicer.ops.filter.text_length_filter` | DJ 支持多字段、多统计量 |
| `llm_quality_scorer` | `data_juicer.ops.selector` + 外部模型 | DJ 有 batch 调用 + 异步 + 缓存 |
| `Pipeline` | `data_juicer/core/executor/`（入口 `default_executor.py`，工厂 `factory.py`） | DJ 有 checkpoint、断点续跑、分布式 |

L1 抓住了「OP 接口统一 + 配置驱动」这个骨架。
L2/L3 会补上分布式（Ray）和完整接口对齐。

---

## 10. 权威实现与延伸

- **对标 Data-Juicer**：`llm_quality_scorer` 对应 Data-Juicer 的 selector / 质量打分算子
  （`data_juicer/ops/selector`、`data_juicer/ops/filter` 系列，本地：`${DATA_JUICER_REPO}`）。
  权威实现额外提供 batch 调用、异步、缓存、多字段统计——这正是 L2/L3 要补的工程差距。

- **概念延伸（数据打分 → 路由）**：给数据打分、再按分数筛选/加权，是「数据路由 / 数据配比」的最简形态，
  也是 data-model co-development（RSI）闭环里「筛选高质量数据回流训练」的核心动作（见轨道 03 主线）。

- **延伸思考**：当数据量从 10 条到 10 万条，打分 OP 的吞吐成为瓶颈——这是轨道 03 `nano-ray`（分布式）
  与 `nano-vllm-sglang`（高吞吐 llm-based 采样）要解决的问题。
