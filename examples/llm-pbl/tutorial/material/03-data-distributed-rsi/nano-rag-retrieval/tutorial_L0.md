# nano-rag-retrieval L0 — embedding 索引 + 向量相似度检索 + 检索评估（纯 Python 本质模拟）

> **前置**：无。Python 3.10+，纯标准库，CPU 秒级。
> **运行**：`python3 L0_vector_index_and_recall.py`（任意目录可跑，输出确定，复跑逐字节一致）。
> **本文件是 notebook-style 教程**：叙述 + 代码摘录 + 真实运行输出 + 思考题交替推进。

---

## §1 为什么检索是 LLM 系统的一等公民问题

LLM 有两个绕不开的知识边界：**训练截止**（它不知道你上周的文档）与**私有数据**（它从没见过的内部知识）。检索增强生成（Retrieval-Augmented Generation, RAG）是对这两个边界最直接的工程回答——先检索相关文档，再让生成器**基于**检索结果作答。RAG 的原始论文（Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*，arXiv:[2005.11401]，v1 2020-05-22）把「参数化记忆（模型权重）+ 非参数化记忆（可更新的外部文档库）」组合成一个端到端架构——本模块只造其中的 **R**：检索器。生成侧怎么用检索结果是 01/04 轨的事。

在 LLM-PBL 的闭环里检索的位置：04 轨 agent 运行时查私有知识库，检索质量直接决定生成质量（检索回来的垃圾会直接变成幻觉的燃料，§6）；而 agent 轨迹与新文档不断回流成新的语料——语料必须是**可重建、可钉住版本**的（nano-data-platform L0 的快照语义，`../nano-data-platform/tutorial_L0.md` §6），索引才谈得上可复现。

本模块抓的核心机制链条（ROADMAP §七）：

```
embedding 索引 → 向量检索 →（混合检索 → 重排序）→ 检索评估
```

L0 裸出前三个里的核心：**索引 + 精确检索 + 评估**；混合检索与重排序是 L2 主题（README 阶梯表）。

## §2 L0 模拟真实系统的哪四面

L0 的验收标准是「能口头讲清它在模拟真实系统的哪一面」。本实现模拟四面，刻意不模拟其余（§9 列边界）：

| # | 机制面 | nano 实现 | 真实系统对应 |
|---|--------|-----------|--------------|
| [1] | embedding：文本 → 几何空间，相似度 = 距离近 | `embed`（词哈希 + 字符 trigram） | 神经 embedding 模型（L1 接真实模型） |
| [2] | flat index：暴力精确 kNN = 一切 ANN 的基线 | `FlatIndex.search` | Milvus FLAT（官方文档直接称 Brute-Force） |
| [3] | 检索评估：recall@k / precision@k / MRR | `evaluate` | 标注评估集 + IR 指标（TREC 传统的现代沿用） |
| [4] | 治理：成本账本 + 分数阈值（会说「不知道」） | `search(tau=…)` + §7 账本 | 索引成本 / 延迟预算 / 阈值校准 |

先跑一遍，建立全局印象（完整输出；以下各节的输出块均从此同一次运行中截取）：

```bash
$ python3 L0_vector_index_and_recall.py
== nano-rag-retrieval L0: embedding 索引 + 向量相似度检索 + 检索评估（纯 Python 本质模拟）==
...
self-check: 13/13 PASS
```

demo 的剧本：14 篇迷你知识库文档建索引 → 四点「光谱」探针展示 embedding 几何 → 词序盲演示 → top-k 检索 → 8 条标注查询评估（5 条 rank-1 命中、1 条被噪声压到 rank 2、2 条纯同义词 MISS）→ 域外查询被阈值挡住 → 成本账本外推 → 确定性 digest 收尾。

> **fixture 声明**：`CORPUS` 是内嵌的 14 篇演示文档（LLM 系统世界的自指内容），`EVAL` 是 8 条标注查询。这是实验设计而非假数据冒充：L0 的机制对象就是「lexical embedding 的天花板长什么样、指标如何把它量化出来」——d14（汽车保养文档）刻意与 LLM 语料格格不入，好让同义词查询有一个明确的 gold；两条 MISS 查询（`automobile fuel economy` / `bonuses for correct behavior`）是刻意设计的纯同义词鸿沟（零词形桥）。L1 会换真实 embedding 模型在同一个评估集上量化突破。

---

## §3 机制面 [1]：embedding —— 把文本映射进几何空间

检索的第一性问题：**「相关」怎么变成可计算的量？** 答案是找一个函数，把文本映到一个几何空间里，使得「语义相近」≈「距离相近」。L0 用最朴素的 lexical embedding：词哈希 + 字符 trigram，feature hashing（Weinberger et al., arXiv:[0902.2206]，v1 2009-02-12）的玩具版：

```python
D = 256  # embedding 维度：toy 取值——大到哈希碰撞少，小到手算可追

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def _h(s):  # 必须 md5，不能用内建 hash()：后者按进程随机盐（PYTHONHASHSEED），输出不可复现
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

# ---- [1] lexical embedding：词(权重 1.0) + 字符 trigram(权重 0.25) → L2 归一化（feature hashing [0902.2206]）----
def embed(text):
    v = [0.0] * D
    for tok in tokenize(text):
        v[_h("w:" + tok) % D] += 1.0
        pad = f"#{tok}#"
        for i in range(len(pad) - 2):
            v[_h("g:" + pad[i:i + 3]) % D] += 0.25
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v

def cosine(a, b):  # 两向量均已归一化 → 点积即 cosine
    return sum(x * y for x, y in zip(a, b))
```

三个设计决策，逐个拆开：

**（a）为什么是哈希而不是词表？** 经典向量化（one-hot / 计数向量）需要维护词表：新词要重训、词表要落盘、分布式场景要同步。feature hashing 把每个特征（词 / n-gram）直接用哈希函数打到固定 D 维的某一维——**无词表、单遍扫描、流式可用**。代价是哈希碰撞：两个无关特征可能打到同一维（下面 [2] 探针会亲眼看到它的后果）。真实系统里这是显式的工程权衡：维度开多大，碰撞噪声就压多低。

**（b）为什么加字符 trigram？** 纯词向量里 `retrieve` 与 `retrieval` 是两个完全无关的维度——词形变化全盲。把每个词再拆成字符 trigram（`#re, ret, etr, tri, rie, iev, eve, ve#` 之类，权重 0.25），词形相近的词就获得部分重叠。这是「subword 特征」思想的最小形态（真实 tokenizer 的 BPE 分片是它的远亲，nano 侧不展开）。

**（c）为什么 L2 归一化？** 不归一化时 cosine 会被文本长度主导——长文档赢在「词多」，不是「更像」。归一化后点积即 cosine，分数只反映方向（内容构成），不反映模长（篇幅）。

**（d）为什么用 md5 而不是 Python 内建 `hash()`？** 内建 `hash()` 对字符串按进程随机盐（PYTHONHASHSEED），同一份代码两次运行得到不同的向量——检索结果不可复现。**可复现性是检索系统的一等公民工程问题**（「为什么这次查出来的和上次不一样」是真实事故），所以 L0 就把这个坑填死：全部哈希走 md5，输出逐字节确定（§8）。

现在看 embedding 几何的实测「光谱」：

```text
[2] embedding 几何：相似度 = 距离近。lexical embedding 有一条可见的『光谱』
  完全一致   cosine = 1.0000
  词形变化   cosine = 0.2449   (retrieve vs retrieval：trigram 部分救回)
  同义词     cosine = 0.0000    (car vs automobile：lexical 空间全盲)
  哈希碰撞   cosine = 0.2052   (gpu vs vram：无关词撞到同一维，假相似)
  [check 02] PASS  完全一致 = 1.0
  [check 03] PASS  词形变化部分相似 (0.1 < cos < 0.5)
  [check 04] PASS  同义词恰好正交 (cos == 0.0)
  [check 05] PASS  哈希碰撞产生假相似 (0.1 < cos < 0.4)——噪声与盲点是 lexical embedding 的一体两面
```

四个点读出四层信息：

1. **完全一致 = 1.0**——几何的自反性，sanity check。
2. **词形变化 = 0.2449**——trigram 把 `retrieve`/`retrieval` 部分救回。lexical 方法能跨过**词形桥**。
3. **同义词 = 0.0000**——`car` 与 `automobile` 在 lexical 空间里**恰好正交**（零共享词、零共享 trigram）。这就是 lexical embedding 的天花板：语义等价但字面不同的词，距离和两个随机词没有区别。**神经 embedding 存在的理由就是把这个 0.0 变成 0.7+**——L1 的主题。
4. **哈希碰撞 = 0.2052**——`gpu` 与 `vram` 语义无关（一个是芯片、一个是显存），却拿到 0.2 的「相似度」。这不是语义，是事故现场：可手算验证 `md5("g:gpu") % 256 == md5("w:vram") % 256 == 169`——`gpu` 的一个 trigram 维度与 `vram` 的词维度撞到了同一维。**盲点（该近不远）与噪声（不该近偏近）是 lexical embedding 的一体两面**，都源于同一个事实：这个几何不是「学」出来的，是「规定」出来的。

**思考题 3.1**：把 D 从 256 开到 2^20，碰撞噪声会怎么变？词形桥（0.2449）会变吗？同义词盲（0.0）会消失吗？（参考方向：碰撞概率随 D 线性下降，词形桥不变——trigram 重叠是真实特征；同义词盲**不会**消失——再大的 D 也只是把词表摊得更开，学不出语义等价。这正是「换更大的哈希」救不了 lexical、必须换**函数类别**（神经模型）的论证。）

---

## §4 机制面 [2]：flat index —— 暴力精确 kNN 为什么是基线

有了向量，检索 = 找最近的 k 个。L0 用最笨的方法：全扫。

```python
# ---- [2] flat index：暴力精确 kNN，O(N·D)/query——精确的代价就是 ANN 的动机（L2）----
class FlatIndex:
    def __init__(self):
        self.docs = []                        # (doc_id, text, vector)
    def add(self, doc_id, text):
        self.docs.append((doc_id, text, embed(text)))
    def search(self, query, k=3, tau=0.0):    # tau = 分数阈值：低于阈值宁可空手而归（§7）
        qv = embed(query)
        ranked = sorted(((cosine(qv, v), did) for did, _, v in self.docs), reverse=True)
        hits = [(did, round(s, 4)) for s, did in ranked[:k] if s >= tau]
        return hits, len(self.docs) * D       # 返回扫描成本：N×D 次乘加（toy 口径，tutorial §7）
```

**为什么先教最笨的方法？因为精确 kNN 是一切近似方法的基线与裁判。** 任何 ANN（approximate nearest neighbor）索引的正确性只能用「相对精确解的 recall」来度量——没有精确基线，「快」就没有参照系。Milvus 官方文档（v2.6.x *Index Explained*，2026-08-13 抓取）对 FLAT 索引的定位：

> "If the filter ratio is over 98%, use Brute-Force (FLAT) for the most accurate search results."
> —— milvus-io/web-content `v2.6.x/site/en/userGuide/indexes/index-explained.md`（官方站 https://milvus.io/docs/index.md 当日 bot 拦截，经 GitHub 文档源核验）

注意这句话的两层信息：FLAT = Brute-Force = **most accurate**（精确是它的卖点）；同时它给了一个使用条件（filter ratio > 98%，即候选集被过滤到很小时）——**暴力不是永远错，要看规模与候选集**。同页对「索引」本身的定义更值得背下来：

> "An index is an additional structure built on top of data. …… using an index typically lowers the recall rate (though the effect is negligible, it still matters)."
> —— 同上

**索引 = 用额外结构换速度，代价是 recall**——这一句就是 §7 成本账本与 L2 全部 ANN 内容的总纲。

两个实现细节：

**（a）排序的确定性。** `sorted(..., reverse=True)` 按 `(score, doc_id)` 元组排序——同分时按 doc_id 倒序打破平局。平局在哈希向量里很常见（一堆 0 分文档），不显式定序，top-k 就会在不同运行间漂移。

**（b）`tau` 参数。** 低于阈值的命中宁可丢弃（§6 的「不知道」语义）。阈值默认 0 = 不过滤，显式传参才启用——策略与机制分离。

**思考题 4.1**：N = 10^6、D = 256 时，单查询暴力扫描 = 2.56×10^8 次乘加。如果业务要求 p99 延迟 10 ms，暴力法还成立吗？哪些场景下它反而成立？（参考方向：批量离线评估、被强过滤后候选极小的在线查询——Milvus 文档的 filter ratio 判据正是此意；「精确但慢」与「近似但快」的选择是场景函数，不是绝对优劣。）

---

## §5 机制面 [3]：检索评估 —— 没有标注，就不知道检索有没有在工作

检索器「感觉挺好」不算数。IR（information retrieval）几十年沉淀出的最小指标集，L0 全部实现——定义就是自含数学，不需要任何库：

- **recall@k** = 平均每条查询，它的 gold 文档落在 top-k 内的比例（本 fixture 每条查询恰 1 个 gold，recall@k = 命中率）；
- **precision@k** = top-k 里有多少是 gold（单 relevant 口径下上限 = 1/k，所以它在这里主要作对照，判断以 recall 与 MRR 为主）；
- **MRR**（mean reciprocal rank）= 1/gold排名 的均值（rank 1 记 1，rank 2 记 1/2，MISS 记 0）——它同时奖励「命中」与「排得靠前」。

```python
# ---- [3] 检索评估：recall@k / precision@k / MRR（定义 = 自含数学，单 relevant 口径）----
def evaluate(index, eval_set, k):
    hits_n, prec, rr, per_q = 0, 0.0, 0.0, []
    for q, gold in eval_set:
        ids = [did for did, _ in index.search(q, k=k)[0]]
        r = ids.index(gold) + 1 if gold in ids else 0
        hits_n += 1 if r else 0
        prec += (1.0 / k) if r else 0.0       # 单 relevant：precision@k = 命中/k
        rr += 1.0 / r if r else 0.0
        top = index.search(q, k=1)[0]
        per_q.append((q, gold, r, top[0] if top else ("-", 0.0)))
    n = len(eval_set)
    return dict(recall=hits_n / n, precision=prec / n, mrr=rr / n, per_q=per_q)
```

8 条标注查询的实测结果——这就是本模块的 **toy 基线**（L1 的真实 embedding 必须用它证明进步）：

```text
[5] 检索评估：recall@k / precision@k / MRR（8 条标注查询，每条 1 个 gold）
  recall@1 = 0.625   recall@3 = 0.750   MRR = 0.688
    'reset my password please'                   gold=d01  rank 1
    'i was charged two times'                    gold=d02  rank 2
    'nearest neighbor query over embeddings'     gold=d03  rank 1
    'how to resume training after a crash'       gold=d09  rank 1
    'reduce hallucination with fetched documents' gold=d11  rank 1
    'remove near duplicate documents before training' gold=d12  rank 1
    'automobile fuel economy'                    gold=d14  MISS (top1 = d01@0.1925)
    'bonuses for correct behavior'               gold=d10  MISS (top1 = d04@0.3298)
  [check 08] PASS  lexical 基线: recall@1 = 0.625 / recall@3 = 0.750 / MRR = 0.6875
  [check 09] PASS  弱信号查询: gold(d02) 被碰撞噪声压到 rank 2，与 top1 差距 < 0.02
  [check 10] PASS  两条 MISS 恰是刻意设计的纯同义词查询
```

逐行解读三种典型命运：

**（a）5 条 rank-1 命中**：查询与 gold 有强词面重叠（`reset/password`、`nearest neighbor/embeddings`……）。词面重叠在，lexical 检索就很强——**永远记住这一点**，它是 §10 反例与 L2 混合检索的伏笔。

**（b）1 条被噪声压到 rank 2**：`i was charged two times` 的 gold 是 d02（含 `charged twice`），但 top-1 是 d01@0.2410。实测核验：查询与 d01 **零共享词**（`{i, was, charged, two, times}` ∩ d01 词集 = ∅），0.2410 全部来自哈希碰撞噪声；真信号 d02 只有 0.2224，差距 0.0186——**弱真信号被碰撞噪声淹没**。这就是 §3 第 4 点的代价在评估指标上的显形：碰撞不只制造假相似，还会压死真命中。

**（c）2 条纯同义词 MISS**：`automobile fuel economy`（gold = d14 的汽车文档）——automobile↔car 零词形桥，gold 连 top-4 都进不了，top-1（d01@0.1925）全是碰撞噪声；`bonuses for correct behavior`（gold = d10 的 reward 文档）——唯一共享词是停用词 `for`（实测：从查询里删掉 `for`，与 d04 的分数从 0.3298 掉到 0.2589，`for` 贡献约 0.07，其余全是碰撞），top-1 = d04@0.3298 是垃圾。**lexical 跨不过语义鸿沟——这两条 MISS 就是 L1 存在的理由。**

一个诚实的旁注：把 `automobile fuel economy` 换成 `automobile yearly upkeep`，d14 会回到 rank 1（0.2557）——`yearly`↔`year` 的 trigram 桥救的场。lexical 能跨词形桥、跨不了语义桥，这条边界在 toy 尺度上画得清清楚楚。

**思考题 5.1**：recall@3（0.750）比 recall@1（0.625）高，能不能得出「k 越大越好」？（参考方向：不能。k 增大时进来的是**噪声层**的文档——本例 rank 2/3 里混着 d01 这类碰撞 top；RAG 场景把 top-3 全塞进 context，等于把 0.16 分的垃圾和 0.69 分的真命中等量齐观。k 与阈值必须一起调，见 §6。）

---

## §6 失败模式总账与阈值的边界

把 L0 亲眼看到的失败模式记一笔总账——这不是缺陷清单，是**机制地图**（每一条都指向更高阶梯的解法）：

| 失败模式 | 实测证据 | 根因 | 解法在哪一级 |
|----------|----------|------|--------------|
| 词序盲 | `dog bites man` vs `man bites dog` cosine = 1.0000 | BoW 向量只依赖词的多重集合 | 神经 embedding（L1） |
| 同义词盲 | `car` vs `automobile` cosine = 0.0000 | 语义等价没有字面重叠 | 神经 embedding（L1） |
| 碰撞假相似 | `gpu` vs `vram` cosine = 0.2052（dim 169 相撞） | D 有限，哈希碰撞 | 加大 D / 学习式 embedding（L1） |
| 弱信号被淹 | d02 真信号 0.2224 < 噪声 0.2410 | 无 IDF 加权，停用词/碰撞与真信号同台竞技 | BM25/IDF 混合检索（L2） |
| 域外垃圾 | 语料无火山，`volcano eruption lava` 照样给出 top-1 | 检索器永远会排序 | 分数阈值（本节） |

```text
[6] 失败模式二：域外查询——阈值让检索器会说『不知道』
  'volcano eruption lava' 无阈值 top1 = d07@0.1448（噪声级分数，语料里根本没有火山）
  tau=0.15 → 空手而归（toy 尺度策略；噪声底随查询漂移，见 tutorial §6）
  [check 11] PASS  阈值之下宁可空手：不把垃圾喂给生成器
```

**阈值是检索器的「不知道」按钮。** 无阈值时，任何查询都会返回 k 篇文档——哪怕语料里根本没有相关内容（check 11：0.1448 的 top-1 纯属碰撞噪声）。在 RAG 链路里这尤其致命：低分文档照样被塞进 context，生成器对垃圾上下文没有免疫力，幻觉就是这么被喂出来的。**宁可空手而归，让生成器走「我不知道」分支，也不喂垃圾。**

但必须诚实交代阈值的边界：噪声底**随查询漂移**。实测反例——`quantum entanglement entropy`（长词、trigram 多、碰撞机会多）的 top-1 = 0.2503，直接越过 tau=0.15。固定阈值不是尺度不变的：查询越长，噪声底越高。真实系统的对策是**按分数分布校准阈值**（per-query 归一化 / 在标注集上扫 precision-recall 曲线选点）或引入重排序模型——L2 主题。L0 只要求把这个坑**看见**。

**思考题 6.1**：如果把 tau 从 0.15 提到 0.3，§5 的 recall@1 与 recall@3 各会变成多少？（参考方向：实测 recall@1 维持 0.625 不变——5 条 rank-1 真命中全是强词面匹配（分数 0.4850–0.7342，全部高于 0.3），阈值动不了它们；被杀的是 rank-2 的弱真信号 d02@0.2224，recall@3 从 0.750 掉到 0.625。再注意 `bonuses for correct behavior` 的垃圾 top-1（d04@0.3298）安然越过 0.3——固定阈值杀不死它不知道的噪声，这正是正文说噪声底要按查询校准的原因。阈值每上调一格，都在同时杀垃圾与杀弱真信号；选 tau = 在 precision/recall 曲线上选工作点，这是业务决策不是技术决策。）

---

## §7 机制面 [4]：治理 first-class —— 成本账本与 ANN 的动机

ROADMAP §七 的硬性写作原则：**安全与成本不是附录，是机制的一部分**。检索侧的成本有两本账：

```text
[1] 索引构建：14 篇文档 → 每篇一个 256 维归一化向量（lexical embedding：词 + 字符 trigram）
  index bytes = N×D×8(float64) = 28672 B（另需原文 1218 B 供展示/重排）
  [check 01] PASS  索引尺寸公式: 14×256×8 = 28672 B
```

```text
[7] 成本账本（toy 口径：乘加 op 数，非真实 benchmark）：暴力不 scale → ANN 的动机
  N = 14:  3584 ops/query；N = 10^6: 2.56×10^8 ops/query
  ANN（HNSW 等）= 用可控 recall 损失换延迟 [1603.09320]——recall/latency 权衡是 L2 主题
  [check 12] PASS  扫描成本随 N 线性增长
```

**（a）存储账**：28672 B 的向量 vs 1218 B 的原文——**向量比原文贵 23.5 倍**（256 维 float64 的代价）。真实系统因此把向量索引与原文存储分开：索引常驻内存吃延迟，原文落盘按需取（展示 / 重排时才读）。D 与量化位宽（float32/float16/int8）是直接的存储-精度旋钮——L2 对照 Milvus 源码时展开 `[TODO: verify L2 源码锚点]`。

**（b）计算账**：每查询 N×D 次乘加，随 N **线性**增长（check 12）。N = 14 时 3584 次微不足道；N = 10^6 时 2.56×10^8 次/查询（算术外推，非实测 benchmark）——暴力法在线服务直接不成立。这就是 ANN 的全部动机：HNSW（Malkov & Yashunin, arXiv:[1603.09320]，v1 2016-03-30）用分层可导航小世界图把搜索复杂度压到对数级，代价是 §4 引文里那句「typically lowers the recall rate」。**recall/latency 权衡是向量检索的第一权衡**——L2 的主题不是「HNSW 怎么用」，而是这条权衡曲线怎么读。

**（c）安全面**：本模块的语料是公开的 toy 文档，但真实检索系统必须回答「**谁能检索到什么**」——多租户向量库的文档级 ACL。L0 不实现（语料无敏感面），但位置要摆对：检索侧权限与 nano-data-platform 的消费侧 default-deny（`../nano-data-platform/tutorial_L0.md` §7）是同一治理问题在两层的投影，L2 展开。

**思考题 7.1**：向量 28672 B、原文 1218 B。如果预算只够常驻一份，留哪个？（参考方向：留向量——查询路径只需要向量，原文可以慢存储按需取；反过来留原文则每次查询要现场重算 embedding，计算账爆炸。这个不对称就是「索引常驻内存、payload 落盘」架构的理由。）

---

## §8 确定性锚点

```text
retrieval digest: 603eda71b56457f2  两次独立构建逐位一致: True
  [check 13] PASS  确定性：两次独立构建索引 + 评估，digest 逐位一致

self-check: 13/13 PASS
```

脚本整体输出确定性：两次独立 CWD（`/tmp/rag_cwd1` / `/tmp/rag_cwd2`）、`python3 -B` 双跑，全 EXIT=0、stderr 0 B，stdout 55 行、md5 `dc5a77c3b697ed25e780253a67d76b0a`，逐字节一致（RUN1==RUN2 BYTE-IDENTICAL）。确定性的来源在 §3（d）讲过：全部哈希走 md5、无随机、无 wall-clock——**「检索结果可复现」从 L0 第一天就是设计约束，不是事后补丁**。

---

## §9 它模拟了什么、刻意没模拟什么（L0 边界 → L1/L2）

**模拟了**（本教程的验收内容）：lexical embedding（词 + trigram，feature hashing）与它的四类失败模式实测；flat 暴力精确 kNN = ANN 基线；recall@k / precision@k / MRR 标注评估；成本账本（索引字节 / 扫描 op）；分数阈值「不知道」语义；全链路确定性。

**刻意没模拟**（每一面都是更高阶梯的课题，不是遗漏）：

| 没模拟 | 为什么 L0 不做 | 哪一级做 |
|--------|----------------|----------|
| 真实神经 embedding | 需要模型依赖；L0 先把「embedding 是什么、lexical 天花板在哪」裸出来 | L1（真实小模型，同评估集量化突破） |
| 索引持久化 / 增量更新 | L0 内存重建以突出「索引 = 派生物」语义 | L1（文件级持久化 + 增量 add） |
| ANN 索引（HNSW/IVF） | 先有精确基线才谈得上近似 | L2 对照 Milvus/OpenSearch/Weaviate 源码 |
| 混合检索（BM25 + dense） | §5（b）的停用词/IDF 教训是它的动机，机制到 L2 才完整 | L2 |
| 重排序（reranker） | 需要 cross-encoder 模型 | L2 |
| 检索侧 ACL / 多租户 | toy 语料无敏感面 | L2（与 nano-data-platform 治理合流） |

## §10 费曼自检

**讲给外行听**：把向量检索想象成一座图书馆。每本书入库时，馆员按内容给它一个**坐标**（embedding——§3）；读者来了，馆员把读者的需求也换算成坐标，然后找出坐标最近的几本书（top-k——§4）。这座小图书馆的馆员很笨：她只数书里出现过的词，所以《汽车保养》和《automobile 手册》在她眼里是两个世界（同义词盲——§5），「狗咬人」和「人咬狗」在她眼里是同一本书（词序盲——§3），偶尔还会把两本不相干的书误标成邻居（哈希碰撞——§3）。她有个优点：宁可说「没有这本书」，也不拿一本不相干的书糊弄你（阈值——§6）。馆里每添一本书，所有查询都要多比对一个坐标（暴力扫描的成本——§7）——大图书馆因此发明了「楼层导览图」（ANN 索引），代价是偶尔漏掉真正最近的那本书（recall 损失——§4 引文）。而判断馆员干得好不好，靠的是一份带标准答案的查询清单（recall@k / MRR——§5），不是「感觉挺准」。

**思考题汇总**（正文内另有 3.1 / 4.1 / 5.1 / 6.1 / 7.1）：

1. 一句话说清：「embedding 的质量决定检索的天花板」——本教程里哪个数字是「天花板」的直接证据？（0.0000：car vs automobile 的正交。索引结构再好，也救不回 embedding 空间里根本不存在的邻近关系。）
2. 本实现里哪两个东西分别对应 Milvus 的「FLAT 索引」与「评估集」？（`FlatIndex` 的暴力全扫 / `EVAL` + `evaluate`。）
3. 为什么 L0 坚持用注定失败的 lexical embedding，而不是直接上真实模型？（先看见天花板，才有资格谈突破——L1 的每一个 recall 增量都要落在 §5 那张逐查询表上，否则「模型更好」就是不可证的口号。这是 PBL 的「做出来」标准在材料设计上的投影。）

**反例（一个常见错误直觉）**：「向量检索就是关键词搜索换个时髦外壳——都是查词，BM25 时代早就解决了。」——错在三点，但诚实的对半分也要说清。错的部分：其一，几何泛化是关键词搜索没有的能力类别——神经 embedding 能把「汽车」与「automobile」放进同一邻域（本教程的 0.0000 正是 lexical 做不到、而学习式 embedding 能做到的那个量，L1 量化）；其二，关键词搜索的匹配是离散的（命中/不命中），向量检索的分数是连续的几何量，可阈值、可加权、可融合——§6 的阈值语义在 BM25 里没有直接对应物；其三，本教程把评估指标当一等公民（§5），而「感觉差不多」的关键词搜索调优没有这个闭环。诚实的部分：词面重叠依然是极强的信号（§5（a）的 5 条 rank-1 全靠它），所以生产系统的答案不是二选一，而是**混合检索**（BM25 + dense 融合）——L2 主题。反例的价值正在于此：它逼你把「何时向量赢、何时词面赢」说清楚。

## §11 溯源

| 声明 | 类型 | 来源 |
|------|------|------|
| RAG = 参数化 + 非参数化记忆组合（§1） | 文献已有 | Lewis et al., arXiv:[2005.11401]，v1 2020-05-22 / v4 2021-04-12，abs 页 2026-08-13 重抓（标题/日期核验） |
| feature hashing（词/n-gram 哈希进固定维，§3） | 文献已有 | Weinberger et al., arXiv:[0902.2206]，v1 2009-02-12 / v5 2010-02-27，abs 页 2026-08-13 重抓 |
| FLAT = Brute-Force = most accurate + filter ratio 判据；「index typically lowers the recall rate」两句引文（§4） | 文献已有（逐字引文） | Milvus 官方文档 *Index Explained*：milvus-io/web-content `v2.6.x/site/en/userGuide/indexes/index-explained.md`（官方站 milvus.io/docs/index.md 当日 bot 拦截，经 GitHub 文档源抓取核验，2026-08-13） |
| HNSW = 分层可导航小世界图 ANN，对数复杂度（§4/§7） | 文献已有 | Malkov & Yashunin, arXiv:[1603.09320]，v1 2016-03-30 / v4 2018-08-14，abs 页 2026-08-13 重抓（摘要核验） |
| Milvus / OpenSearch / Weaviate 为权威参照实现 | 纲领已有 | ROADMAP §五/§七 参照表；三 repo 页面 2026-08-13 重抓坐实（milvus-io/milvus / opensearch-project/OpenSearch / weaviate/weaviate） |
| BM25/IDF 下权高频词、混合检索（§5（b）/§9/§10） | 机制类别的概念性提及 | 不作数字声明；L2 展开时补一手锚点 `[TODO: verify L2 锚点]` |
| 全部实测数字（0.2449 / 0.0000 / 0.2052 / 0.6880 / recall 0.625·0.750 / MRR 0.6875 / 0.2410 / 0.2224 / 0.3298→0.2589 / 0.1925 / 0.1448 / 0.2503 / 28672 B / 3584 ops / digest 等） | 本实现实测（toy 设定） | `L0_vector_index_and_recall.py` 本次运行输出与同代码探针实测，非真实系统 benchmark、不可外推 |
| 「向量索引与 payload 分离存储」（§7（a）） | 合理推断（机制同类） | 概念性提及；行号级源码锚点 `[TODO: verify L2 源码锚点]` |

下一站：**L1**——真实小 embedding 模型（CPU 可跑）接上同一套索引与评估集，量化 lexical → 语义的 recall 突破（§5 的两条 MISS 是验收靶点）+ 索引文件级持久化与增量更新；**L2**——对照 Milvus/OpenSearch/Weaviate 源码的 ANN（HNSW/IVF）取舍分析 + 混合检索（BM25+dense）+ 重排序 + recall/latency 权衡实测（见 README 阶梯表）。
