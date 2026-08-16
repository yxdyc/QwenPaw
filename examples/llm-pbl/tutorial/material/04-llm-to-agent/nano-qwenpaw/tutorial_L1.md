# nano-qwenpaw L1 — 记忆与上下文管理：窗口是 cache，store 才是 memory

> L0 给**单个任务**套上了 harness：system prompt + 输出自检 + critique 重试，跨任务无状态。
> L1 让 harness **在有限上下文窗口下活过多轮**：状态无限增长，窗口不会——总得有什么让路。
> 本节在**同一段真实对话、同一预算**下实测三种记忆政策（append-only / summarize / write-through+evict-index）的损失谱：
> 40–40% / 10–20% / 100–100%，并逐项对照 qwenpaw scroll 源码（`manager.py` / `eviction_index.py` / `history.py` / `cap_middleware.py`）。

---

## 1. 先跑为敬

文件：`L1_real_memory_loop.py`，零外部依赖（纯标准库：`sqlite3` / `hashlib` / `re` / `math` / `random`），CPU 即跑，约 1 秒。

```bash
$ python3 L1_real_memory_loop.py
```

真实输出（2026-08-14 复验，逐字节粘贴）：

```text
====================================================================
nano-qwenpaw L1 — memory & context management, measured
====================================================================
python 3.9.6
declarations: WindowModel = declared mock (hard window cut +
  extractive answer); recall agency declared (index map, FTS
  fallback); corpus/store/tokens/summary = real (files+sha256,
  sqlite3+FTS5, declared estimator, TF-IDF). Real hosted model
  behind the loop: [TODO: needs key]

[0] corpus: real sources -> facts + padding
    eviction_index.py    sha256[:8]=aa8c30a1  mode=live
    cap_middleware.py    sha256[:8]=5ea09476  mode=live
    manager.py           sha256[:8]=6260c313  mode=live
    history.py           sha256[:8]=48b71b62  mode=live
    SOUL.md              sha256[:8]=e143a057  mode=live
    facts=10 (detail+gist Q each) | padding paragraphs=14
    conversation: 24 turns | transcript ~2514 est-tokens | model window=1000 | harness budget=1000

[1] append-only (none): recall of fact#1 as the conversation grows
    recall curve (probe fact#1 after each turn once it entered): 11111111100000000000000
    fact#1 falls out of the window after turn 11 (window=1000 est-tokens)

[2] three policies, same 24-turn conversation, 20 probe questions
    policy        detail    gist  ctx_tok  extras
    none           40.0%   40.0%     2514  transcript overflows window=1000 -> head dropped
    summarize      10.0%   20.0%     1148  compressions=17 kept_turns=['-', 6, '-']
    evict-index   100.0%  100.0%      985  recalls map/fts/both=0/9/11 index_lines=8 store=24rows/9905B

[3] where the losses sit (per-fact detail recall under each policy)
    fact   none  summarize  evict-index   source
    F1   MISSED     MISSED          hit   eviction_index.py
    F2   MISSED     MISSED          hit   eviction_index.py
    F3   MISSED     MISSED          hit   cap_middleware.py
    F4   MISSED     MISSED          hit   cap_middleware.py
    F5   MISSED     MISSED          hit   manager.py
    F6   MISSED        hit          hit   manager.py
    F7      hit     MISSED          hit   history.py
    F8      hit     MISSED          hit   history.py
    F9      hit     MISSED          hit   SOUL.md
    F10     hit     MISSED          hit   SOUL.md
    summarize: summary 1041 tok <- folded 2417 tok (ratio 0.43, lossy & irreversible)

[4] the qwenpaw invariant, checked on real storage
    write-through: store rows == turns entered: 24 == 24 -> True
    FTS5 recall: MATCH 'tier cap blocks' -> seq [2] (fact#1 seq = 2)

[5] self-check (structural assertions)
    PASS  append-only: early facts lost, late facts visible
    PASS  append-only: per-fact recall monotone in recency
    PASS  summarize: lossy — detail recall below evict-index
    PASS  summarize: losses NOT recency-monotone (unpredictable, unlike none)
    PASS  summarize: salience kept padding over facts (failure mode measured)
    PASS  evict-index: detail recall == 100% (nothing lost)
    PASS  evict-index: gist recall == 100%
    PASS  evict-index: final context within budget
    PASS  evict-index: recall agency actually fired
    PASS  evict-index: store complete (write-through)
    PASS  summarize: compression is lossy (summary < folded)
    ✅ self-check passed

====================================================================
takeaway: the window is a cache, not the memory. Append-only makes
  the cache the whole truth and silently loses the head; summary
  trades detail for space and cannot be undone; write-through +
  eviction index + recall keeps context bounded while the store
  stays complete — memory reliability is a storage design, not a
  model property. (qwenpaw scroll: manager.py / eviction_index.py
  / history.py; real hosted model [TODO: needs key])
====================================================================
```

**声明（课程可运行性契约）**——输出开头也打印了同样的声明：

- **WindowModel 是声明的 mock**，只带两条性质：(a) 硬窗口裁切——它只看得到最后 W 个估计 token；(b) 抽取式回答——从可见内容里挑 TF-IDF 余弦最高的一句作答。它不是语言模型，不产生任何「智能」。
- **recall agency 也是声明的**，由 harness 中介：问题与某条索引行重叠 ≥2 个内容词元 → 召回该 seq span；无索引命中时 fallback 到对持久 store 的 FTS5 查询。真实系统里「要不要召回、召回哪段」的判断坐在 LLM 那边（qwenpaw 的 REPL / `ms.sql_query`），这里用确定性规则替代：`[TODO: needs key]`。
- **其余全部是真**：真实源文件运行时读取（sha256 记录，漂移可检测，读不到时用 pinned snapshot fallback 并打印 mode）；真实 sqlite3 write-through store + FTS5 召回索引（与 qwenpaw `history.db` 同构的存储技术）；真实 token 记账（声明的估计器 `ceil(chars/4)`）；真实 TF-IDF 抽取式摘要。

确定性声明：seed=42、无计时行——**同一源码快照与 Python 环境上重复运行，输出逐字节一致**。源码会演进，因此教程把来源哈希与运行日一起记录；读者应以自己运行得到的结果为准。

---

## 2. K+1：L0 留下什么，L1 接住什么

L0 的 harness 是**单轮任务**的：接一个任务、注入 system prompt、检查输出、critique 重试、交付。任务之间没有状态——上一个任务发生了什么，下一个任务一概不知。

L0 教程 §8 曾预告「L1 把 mock 换成真实 API」。实际落地的 K+1 步骤做了调整，这里如实说明：L1 接住的是 L0 预告三件事里**最本质的一件——多轮记忆管理**（README 对 L1 的原定义：「加记忆与上下文管理：跨轮状态、何时压缩/检索」）。模型侧仍是声明的 mock，但换上了一个 L0 没有的、对记忆至关重要的性质：**有限窗口**。真实托管模型路径保留为 `[TODO: needs key]`，留给后续级别。

为什么这是正确的 K+1？因为 agent 的记忆问题**不取决于模型多聪明，而取决于窗口装不下时怎么办**。任何生产 agent（无论背后是 7B 还是旗舰模型）都面对同一个物理约束：上下文窗口有限，而对话、工具结果、轨迹无限增长。窗口装不下时的策略选择，是 harness 层的存储设计问题——这正是本节要测的东西。

于是要回答的问题变成：**状态无限增长、窗口有限时，什么留、什么压、什么检索？**三种政策对应三种答案：

| 政策 | 答案 | 真实系统里的化身 |
|------|------|------------------|
| `none` | 全留（append-only），装不下时模型窗口自己切掉头 | 裸拼 messages 的 agent |
| `summarize` | 超预算就把旧轮折叠成摘要 | 各类 conversation summary / compaction |
| `evict-index` | 每轮**进窗口即写穿**到持久 store；超预算时把旧轮逐出窗口，但留一行索引（`[seq N] headline`），按需召回 | qwenpaw scroll（`manager.py` + `eviction_index.py` + `history.py`） |

---

## 3. 实验设计：把变量控制到「只差政策」

三个政策跑在**完全相同的输入**上，预算也相同——这样测出的差异只能来自政策本身：

- **同一段 24 轮对话**：10 条 fact 轮（study note）与 14 段 padding 轮（reading note）交错，再加 padding 尾巴。fact 是 grounding 在真实源码行上的学习笔记，padding 是真实散文（LLM-PBL README/学习总导航 + qwenpaw README 的段落）。
- **同一个预算**：`WINDOW = BUDGET = 1000`（估计 token）。模型硬窗口与 harness 预算**刻意相等**——如果 harness 预算比窗口小，省预算的功劳就分不清是政策的还是预算的；相等时，三种政策在「同一个物理约束」下竞争。
- **同一组 20 个探针问题**：每个 fact 两问——detail 问（答案必须包含精确子串，如 `_TIER_CAP = 10`）与 gist 问（答案必须包含关键短语，如 `carries up`）。判分是机械的：抽取出的句子包含目标子串/短语即 hit。
- **transcript 构造为必须溢出**：全部 24 轮约 2514 est-tokens > 1000（代码里有构造断言）——不溢出就没有可比的损失。

### 3.1 facts 从哪来：live 源码 + sha256 + 正则提取

10 条 fact 不是手写的，而是**运行时从五个 live 源文件正则提取**的：

```python
n_principles = len(re.findall(r"(?m)^## \d+\.", t["SOUL.md"])) or 7
m = re.search(r"_TIER_CAP = (\d+)", t["eviction_index.py"])
m = re.search(r"token_cap: int = (\d+)", t["cap_middleware.py"])
m = re.search(r"pinned: int = (\d+)", t["manager.py"])
```

于是教程里的数字**跟着权威源码走**：如果哪天 `eviction_index.py` 把 `_TIER_CAP` 改成 12，fact#1 的文本与答案会自动跟着变（思考题 5），而不是材料里写死一个过时的 10。每个源文件读取时记录 sha256 前 8 位（本次输出：`aa8c30a1 / 5ea09476 / 6260c313 / 48b71b62 / e143a057`，全部 `mode=live`），源文件不可读时退回代码内 pinned snapshot 并把 mode 打成 `PINNED`——漂移与降级都是**可见的**。

每条 fact 还带 ground-truth 守卫：`detail_ans` 与 `gist_kw` 必须是 fact 文本的子串（`assert` 在位）——答案不可能「在对话之外」。

### 3.2 padding 的纯度：压力必须是体积，不是泄题

padding 段落构成「窗口压力」，但压力必须只是体积：任何 padding 段落若逐字包含某 fact 的答案短语，就被过滤掉，且入对话前有断言复查：

```python
for t in turns:
    if "fact" not in t:
        assert not any(f["detail_ans"].lower() in low
                       or f["gist_kw"].lower() in low for f in facts), \
            "padding leaked a fact phrase"
```

（token 级重叠不过滤——那是 TF-IDF 被设计来处理的东西。）另有一个真实陷阱的防御：代码块与目录树（含 `├└│▼` 的行）**不算散文段落**——它们的稀有 ASCII 行会劫持 salience 排序，这是写摘要政策时真的会踩的坑。

### 3.3 WindowModel 为什么够用

要测的是「政策让模型**看到**什么」，不是「模型多会回答」。WindowModel 的两条性质恰好是这个测量所需的最小模型：

- **硬窗口裁切**（`prompt[-budget:]`）：忠实模拟「窗口外的内容对模型不存在」；
- **抽取式回答**（可见内容句子里 TF-IDF 余弦最高者，低于阈值答 UNKNOWN）：答案只能来自可见内容——于是 recall 命中率**恰好等于**「答案所在句子是否在窗口内」，没有模型能力变量混进来。

结构行（`[seq `、`evicted turns (recall`、`summary so far`、`system:` 等）不算内容句——索引行与 section header 是 harness 的结构，不是对话内容。

---

## 4. 政策一 `none`：静默的、按新近度单调的损失

`none` 就是裸拼：`SYSTEM + 全部 24 轮`，prompt 长 2514 est-tokens，模型窗口 1000——**头部 1514 est-tokens（约 60% 的对话）对模型不存在**。

[1] 板块把损失过程拍了下来：从 fact#1 入场（turn 2）起，每加一轮就探测一次「fact#1 还能不能被召回」：

```text
recall curve: 11111111100000000000000
fact#1 falls out of the window after turn 11
```

9 个 1 之后断崖式归零——turn 11 之后，fact#1 所在的句子被挤出窗口，召回从 100% 瞬间变 0%。没有警告、没有日志、模型也不会说「我看不到了」：**append-only 的损失是静默的**。

[2]/[3] 给出全貌：detail/gist 都是 40.0%（F1–F6 全丢、F7–F10 全中）。损失谱是**按新近度单调**的——丢的永远是最早的，留的永远是最近的（self-check 第 2 条断言）。这在某些场景甚至算「可预测」，但注意它的含义：**对话的开头——通常正是任务定义、用户约束、系统原则——恰恰是最先被丢的**。

---

## 5. 政策二 `summarize`：salience ≠ importance，损失不可预测

`summarize` 的策略：上下文超预算时，把**较旧的一半**折叠掉，从「已有摘要 + 被折叠轮」的池子里按 salience 保留 top-3 整轮（`SUM_KEEP_TURNS = 3`）。salience = 一轮内容词元的**平均 IDF**（IDF 在候选池上算）——即「用词越稀有越显眼」。

24 轮跑完：17 次压缩，最终摘要 prompt 为 1148 est-tokens，detail/gist 为 **10.0% / 20.0%**——比什么都不做的 append-only 还低一大截。这里的 `summarize` 是对照政策，不保证自己总能压回 1000；这恰好暴露了固定 top-k 摘要在长轮次下可能同时发生**信息损失与预算失守**。损失在哪？

```text
kept_turns=['-', 6, '-']
```

保留的 3 个槽位里，**两个被 padding 段落（'-'）占了**，只留下 fact#6。这不是偶然噪声，而是 salience 公式在这段语料上的结果：10 条 fact 共享一套词汇（seq / span / turns / token / store…），互相之间 IDF 被摊低；padding 散文主题各异，稀有词多，salience 反而高。**token 稀有度与任务重要性无关**——摘要政策保住了「最显眼的」，丢掉了「最该留的」。self-check 第 5 条就是把这个失败模式测出来（「salience kept padding over facts」）。

再看 [3] 表：summarize 的损失**不是新近度单调**的——F6 活着，F7–F10（都比 F6 更新）却死了（self-check 第 4 条断言「损失非单调」）。这是比 append-only 更糟的性质：**你无法用「旧的不重要」来安慰自己，丢什么取决于词频统计，不可预测**。

最后是不可逆的算术：

```text
summarize: summary 1041 tok <- folded 2417 tok (ratio 0.43, lossy & irreversible)
```

2417 tok 的内容被压成 1041 tok 的保留集，ratio 0.43。被丢的 1376 tok 没有任何副本——summarize 政策不写 store，**压缩即销毁**。

---

## 6. 政策三 `evict-index`：write-through + 逐出索引 + 按需召回

这是 qwenpaw scroll 的方式，三个部件各管一件事：

**1）write-through（进窗口即落盘）**。每一轮**进入窗口的那一刻**就写进 sqlite store（`store.put` 返回全局唯一 seq）——不是逐出时才写，因为逐出发生时再写就晚了（崩溃、截断都会丢）。代价是零：写穿不改变窗口内容。

**2）eviction index（逐出但留地图）**。超预算时，最旧的 tail 轮被逐出窗口，但索引里多一行：

```text
[seq 7] manager.py: `pinned: int = 1` — 1 turn stays pinned raw at the hea...
```

索引行 = seq 地址 + headline。索引本身也有界：超过 `IDX_CAP = 8` 行时，**最旧两行折叠成一条 span 行**（只显示端点 headline）——于是索引永远 ≤8 行，对话再长，索引的上下文开销也是 O(1)。最终 `index_lines=8`、`ctx=985 ≤ 1000`：窗口贴近预算但没超。

**3）recall（按需展开）**。问题来了：与索引行重叠 ≥2 个内容词元 → 取该 span（map 候选）；同时 FTS5 查 store（fts 候选，`OR` 查询）；任一命中就从 store 拉出原文拼到 prompt 尾部再作答。20 个探针问题全部触发召回：

```text
recalls map/fts/both=0/9/11
```

一个值得盯着看的数字：**map 从不单独命中（0）**。原因在包含关系里：headline 是 turn 文本的前缀（`text[:120]`），所以「与 headline 重叠 ≥2 词元」的问题，其词元必然也落在 store 里的完整内容上 → FTS 一定同时命中 → 记 both。headline 证据永远是内容证据的子集——map 的价值不在「独立命中」，而在提供 **span 地址**（一次取回一整段，不用逐行查）。9 个 fts-only 是索引行没覆盖到的问题（tail 内与 pinned 的 fact：还没被逐出、没有索引行，但内容在 store 里）。

（声明的边界：这个规则 agency 不知道 tail 里已经有什么，所以 tail 内 fact 的问题也付一次召回——这是测出来的成本的一部分，真实系统的 LLM 判断可以省掉这部分。见反例 §12.1。）

结果：detail **100.0%**、gist **100.0%**，且「recall agency actually fired」断言在位——100% 不是靠 transcript 没溢出混来的（ctx=985 接近预算，20 问全走召回），是靠存储设计换来的。

---

## 7. 不变量：在真实存储上检查

[4] 板块把 qwenpaw 的核心不变量直接跑在真实 sqlite 上：

```text
write-through: store rows == turns entered: 24 == 24 -> True
FTS5 recall: MATCH 'tier cap blocks' -> seq [2] (fact#1 seq = 2)
```

- **write-through 不变量**：store 行数 == 入场轮数。24 == 24——**nothing is lost** 不是一句口号，是一个可以 fail 的断言。
- **可寻址性**：FTS5 `MATCH 'tier cap blocks'` 命中 seq [2]，正是 fact#1 的 seq。任何一行历史都有一个全局唯一地址，一条 SQL 就能展开——qwenpaw 源码的原话：「Nothing is lost — every line carries a `seq` span and the full turns stay in `conversation_history`」（`eviction_index.py:L20-22`，2026-08-06 核验）。

store 的构造与 qwenpaw `history.py` 逐字同构（外部表 + fts5 影子表）：

```python
CREATE TABLE conversation_history (seq INTEGER PRIMARY KEY, role TEXT, content TEXT)
CREATE VIRTUAL TABLE conversation_history_fts USING fts5(
    content, content='conversation_history', content_rowid='seq',
    tokenize='porter unicode61')
```

对照 `history.py:L148`（`CREATE TABLE IF NOT EXISTS conversation_history`）与 `L212`（`fts5(content, content='conversation_history', content_rowid='seq', tokenize='porter unicode61')`）——**召回用的存储技术不是教学简化，就是真实实现本身**（本次 sqlite 文件为 `store=24rows/9905B`；字节数会随语料与 SQLite 版本变化，关键不变量是 24 轮全部可寻址）。

---

## 8. 与 qwenpaw scroll 源码对照（2026-08-14 核验）

权威实现就在本仓库：`qwenpaw_coach/src/qwenpaw/agents/context/scroll/`。以下锚点全部当日现场核验（sha256 见 §14）：

| nano L1 | qwenpaw scroll | 核验锚点 | 差异与原因 |
|---------|----------------|----------|-----------|
| 入场即 `store.put`（write-through） | HistoryStore write-through | `history.py:L57`（「Every event the agent appends is write-through-persisted」） | 思想相同；真实系统还把 write-through 失败做成 durability 降级健康态（`history.py:L480`、`L489`「history write-through FAILED; durability degraded」）——落盘失败不是异常日志，是系统健康状态 |
| 单层索引，`IDX_CAP=8` 时最旧两行折叠成 span 行 | 分层 odometer `EvictionIndex`：每层最多 `_TIER_CAP = 10` 块，满了 carry 上层、级联进位；`compact()` 是压力阀 | `eviction_index.py:L31`、`L177-197`（carry）、`L199-226`（compact） | nano 只演示「地图 + span 可寻址」的最小形态；qwenpaw 要应对**任意长**对话——span-of-spans 与 span 用同一方式折叠（自相似），索引可以一直向单层收敛 |
| `[seq N] headline` 索引行 | `Line(seq_lo, seq_hi, head, tail)`，span 行只显示端点 | `eviction_index.py:L57-73` | 两边一致：折叠只丢「显示粒度」，不丢地址 |
| recall = 索引重叠 ≥2 词元 + FTS5 fallback（harness 规则） | 模型经 recall tool / `ms.expand` 自主展开 | `eviction_index.py:L20-22`、`L296`（「Re-expand a span … ms.expand(lo, hi)」） | 真实系统里 agency 坐在 LLM 侧；nano 用确定性规则保证实验可复现 `[TODO: needs key]` |
| sqlite3 + fts5（porter unicode61，`content_rowid='seq'`） | 同 | `history.py:L98`、`L148`、`L212` | 存储技术逐字同构 |
| pin 第 1 轮 | manager 的 `pinned` 配置 | 运行时由 `manager.py` 的 context configuration 解析 | 默认值由当前配置提供；关键性质是头部（任务/原则）不参与中段逐出 |
| 超阈值 → recent tail，中间逐出成索引 | 同 | `manager.py:L10`、`L282`（持久化失败时禁止逐出） | 结构同构；当前实现还把 durability 作为 eviction 的前置条件 |
| `est_tokens = ceil(chars/4)`（声明估计器） | 模型自己的 count_tokens | `cap_middleware.py:L24`（「the model's own estimator」） | 估计器是声明的：绝对值不可与真实系统比，政策排序与损失谱不受影响 |
| 未建模工具结果封顶 | 单条工具结果超 `token_cap: int = 3000` → 全文写穿、上下文留 preview + 按 `tool_call_id` 的 recall pointer | `cap_middleware.py:L38`、`L46`、`L97`（「the only capping path and it never loses data」，`L27-28`） | 同一个「写穿 + 留指针」模式在工具结果维度的实例，留 L2 |

一个值得单独点的工程细节：qwenpaw 的索引在 prompt 里是**一个占位消息**，`render()` 的 docstring 明确写了 KV-cache 安全——稳定前缀与逐步追加的低层块尽量保留 prefix cache，carry 重排才打破更高层前缀（`eviction_index.py:L274-297`）。**索引的数据结构选择直接影响推理成本**——这与 03 轨 nano-vllm-sglang L1 讲的 prefix caching 是同一个机制的两端。

---

## 9. 机制深潜：三种损失谱，一个设计决策

把三张表并排：

| 政策 | detail / gist | 损失是什么 | 损失可预测吗 | 可逆吗 |
|------|---------------|-----------|--------------|--------|
| none | 40% / 40% | 头部（最旧）内容 | 可预测：按新近度单调 | 不可逆（没有副本） |
| summarize | 10% / 20% | salience 低的轮——**与重要性无关**；且本次未压回预算 | 不可预测：非单调 | 不可逆（1041←2417，ratio 0.43） |
| evict-index | 100% / 100% | **无内容损失**；付出的是召回开销（20 问 20 次召回）与索引的上下文租金（≤8 行） | — | 完全可逆（store 24==24） |

三个结论：

1. **窗口是 cache，store 才是 memory**。append-only 把 cache 当成了全部真相——窗口一切，记忆就没了；write-through 把真相放在 store 里，窗口只是「当前工作集」。前者在赌「重要的都在最近」，后者不赌。
2. **摘要的损失谱比不管理更糟**。这反直觉（「压缩总比丢了好」），但本次测得 detail 10% < 40%，而且损失不可预测。根因是 salience 信号（词元稀有度）与任务信号（这条信息后面会不会被问）正交。**摘要不是无损压缩，是基于重要性代理的有损丢弃**；固定 top-k 还可能压不回硬预算。这里证明的是这个具体摘要器的失败模式，不是“所有摘要必然失败”。
3. **记忆可靠性是存储设计，不是模型属性**。同一个 WindowModel、同一段对话，换个政策，detail recall 从 40% 到 100%。模型没变，变的是「什么在窗口里」——而这完全由 harness 的存储策略决定。这就是 04 轨的核心命题之一：**agent 的可靠性工程，很大一部分是上下文工程**。

---

## 10. 费曼：讲给外行听

**类比：阅览桌、书库与目录卡。**

你（模型）在图书馆查资料，面前有一张**阅览桌**（上下文窗口），桌面只能摊开 1000 页。图书馆有两层：桌面（快，但小）和**书库**（慢一点，但装得下全部）。

- **append-only**：所有借来的书全堆在桌上，新书不断往上摞，最早的书被挤出桌沿掉进碎纸机。你要找第一本书的内容？它已经不存在了——而且没人告诉你它什么时候掉的。
- **summarize**：桌面快满时，管理员把旧书**抄成一页摘要**然后扔掉原书。听起来合理——但管理员判断「哪页值得抄」的标准是「用词越生僻越重要」。结果你的任务定义被扔了，两页无关的诗歌介绍留在了桌上（kept_turns=['-', '-', 6]）。而且原书已扔，抄错了也找不回来。
- **evict-index**：每本书一进阅览室就**先登记入库**（write-through，书库里永远有全本）。桌面满了，旧书送回书库，但桌上留一张**目录卡**：「[seq 7] manager.py：pinned 头部设计」。你要细节？报卡号，管理员从书库把原书取回来摊开（recall）。桌面始终 ≤1000 页，书库 24 本一本不少（24==24）。

类比的边界：真实书库取书有延迟（召回开销），nano 里召回是零成本的内存操作——真实系统的代价模型更复杂（这正是 recalls 计数存在的意义）；另外 nano 的「管理员」（recall agency）是死板的规则，真实系统里是 LLM 自己判断「我该去书库查哪一段」。

---

## 11. 思考题

1. **窗口算术**：transcript 2514 est-tokens、窗口 1000——`none` 政策下模型看不到的比例是多少？（1514/2514 ≈ 60%。）如果把对话加长到 40 轮，recall curve 的断崖点会不会移动？改 `turns` 构造跑一遍验证；不要只凭“轮数更多”猜答案，因为断崖由前缀与单轮长度共同决定。
2. **salience 实验**：`SUM_KEEP_TURNS = 3` 时 kept_turns=['-', '-', 6]。把它改成 1，detail recall 会变成多少？用 salience 公式（平均 IDF）解释为什么 fact 之间互相「卷」不过 padding。（提示：facts 共享 seq/span/turns 词汇。）
3. **map 为什么不单独命中**：recalls map/fts/both = 0/9/11。从「headline 是内容前缀」出发证明 map 命中必然伴随 fts 命中；然后想：map 提供的 span 地址在什么场景下才显出价值？（提示：一次 `ms.expand(lo, hi)` 取回整段 vs 逐行查询。）
4. **pinned 的由头**：recall 展开会把总 context 推过 WINDOW，WindowModel 的头部裁切会连 SYSTEM prompt 一起切掉（prompt_head 985 接近预算，展开后可能超）——这是声明过的 mock 性质。真实系统为什么要把头部 pin 住、让它永远不参与逐出？
5. **数字跟着源码走**：本节 facts 是运行时从源码正则提取的。如果有人把 `eviction_index.py:L31` 的 `_TIER_CAP` 从 10 改成 12，本教程的哪些数字会变、哪些不变？这与「材料里写死数字」相比，是什么反幻觉姿势？

---

## 12. 反例与边界

1. **recall agency 是规则，不是判断**。真实系统里「要不要召回、召回哪段」是 LLM 的决策（qwenpaw 经 REPL / `ms.sql_query`）；nano 用确定性规则替代，换来可复现，代价是测出的召回成本（20 问 → 20 次召回）偏「笨」——聪明的 agency 对 tail 内 fact 可以不召回。真模型接入是 `[TODO: needs key]`。
2. **WindowModel 的头部裁切连 SYSTEM 都会切**（见思考题 4）——这是声明的 mock 性质，不是对真实系统的忠实刻画；真实 qwenpaw 用 pinned head 规避。用它推导「真实模型会怎样」时，这条性质要剔除。
3. **估计器的绝对值不可外推**。`ceil(chars/4)` 是声明估计器（真实系统用模型自己的 count_tokens，`cap_middleware.py:L24`）——2514/1148/985 这些绝对值只在声明口径内成立；政策排序与损失谱的结构不依赖估计器选择。
4. **padding 压力是纯体积的**。纯度断言保证 padding 不逐字泄题，但真实对话的压力还有**语义干扰**（主题相似但内容错误的轮次）——抽取式回答在语义干扰下的行为是另一回事，本节不覆盖。
5. **卫生项**：`tempfile.mkdtemp` 未 rmtree，history.db 残留在系统 TMPDIR（macOS 会自清）。生产里 store 的生命周期（持久 vs 清理、durability 降级，见 `history.py:L464`）是真实工程问题。

---

## 13. 阶梯预告 + 交叉引用

- **L2（注入方法论）**：把 K+1 / 费曼 / 对抗自检变成 harness 的内置流程——本节的 self-check 断言块已经是「对抗自检」的雏形，L2 让它成为 agent 运行时的常规动作；同时补上工具结果维度的写穿（`cap_middleware.py` 的 token_cap + recall pointer 模式）。
- **L3（对照 SOUL.md + skills 架构）**：对照 qwenpaw coach 的 `coach/profile/SOUL.md`（7 条编号原则，2026-08-06 核验）与 `coach/profile/skills/`，复现一个「有原则的 agent」。
- 交叉引用：agent loop 的可靠性代数见 [nano-agentscope L1](../nano-agentscope/tutorial_L1.md)（违规率 p 与重试算术）；prefix 稳定性为什么值推理成本见 [nano-vllm-sglang L1](../../03-data-distributed-rsi/nano-vllm-sglang/tutorial_L1.md)（KV cache / prefix caching）；agent 轨迹回流成训练数据是 03 轨 RSI 闭环的入口。

---

## 14. 溯源与校准

**五源 sha256（2026-08-14 现场核验，`shasum -a 256`，全部 `mode=live`）**：

| 源文件 | sha256 |
|--------|--------|
| `src/qwenpaw/agents/context/scroll/eviction_index.py` | `aa8c30a1f368ddb78b48988634e0b7156d0f6b600a9e38e7c941e8eb5b2443f1` |
| `src/qwenpaw/agents/context/scroll/cap_middleware.py` | `5ea09476ab7c778dfc67faec01d23c2627e616cf4c2f442df60a462fb581aef1` |
| `src/qwenpaw/agents/context/scroll/manager.py` | `6260c313140c499147fe14786e829363b313ee6f676be3d8d7a6bbdbd1e0706e` |
| `src/qwenpaw/agents/context/scroll/history.py` | `48b71b622ba55b4a06eb6dbb1da5f9f5cb111391ca0c8f102dfb984e42ce2f23` |
| `coach/profile/SOUL.md` | `e143a057ba1183f7c18e303ade00f9043b17e850628de4c2827a4594c9b9a72a` |

**引用锚点**（同上快照，行号级核验）：`eviction_index.py:L20-22`（nothing is lost）、`L31`（`_TIER_CAP = 10`）、`L57-73`（Leaf/Line）、`L177-197`（carry）、`L199-226`（compact 压力阀）、`L274-297`（render / KV-cache 友好布局与召回说明）；`cap_middleware.py:L24`（模型自己的估计器）、`L27-28`（按 `tool_call_id` 保留全文）、`L38`（`token_cap: int = 3000`）、`L46`/`L97`（键控）；`manager.py:L10`（阈值后保留 recent tail）、`L227`（write-through 失败）、`L282`（durability 降级时禁止逐出）；`history.py:L57`（write-through）、`L98`（sqlite3.connect）、`L148`（conversation_history）、`L212`（fts5 porter unicode61）、`L480`/`L489`（durability 降级）。

**SOUL.md 原则数**：7 条编号原则（`## 1`–`## 7`；principle 5 = Anti-Hallucination: Zero Tolerance）。L1 代码在运行时解析，不依赖教程中的硬编码计数。

**确定性记录**：同一源码快照与 Python 环境重复运行，输出逐字节一致；跨版本比较应先检查上述五个来源哈希。

**复验环境**：Python 3.9.6（`python3`）/ 纯标准库 / CPU / seed=42。真实托管模型：`[TODO: needs key]`。
