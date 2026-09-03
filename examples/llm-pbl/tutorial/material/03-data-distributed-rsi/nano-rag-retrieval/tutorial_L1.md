# nano-rag-retrieval L1 — 真实神经 embedding + 索引持久化 + 增量 add（自 L0 的 K+1）

> **前置**：`tutorial_L0.md`（L0 的 lexical 天花板、flat kNN、评估指标与阈值语义是本级的 K 层）。Python 3.10+。
> **运行**：`python3 L1_semantic_embedding_and_persistence.py`（默认离线：模型快照已钉住，零网络）。
> **双路径（可运行性契约）**：环境中有 `sentence-transformers` 且 HF 缓存包含 `all-MiniLM-L6-v2` 快照 → **real 路径**（真模型）；否则自动降级 **fallback 路径**（零依赖显式 mock：手写同义词簇 + bigram，dim 512）——同一套机制检查，数值口径各归各（§10）。首次取模型快照须显式运行：`NANO_RAG_ALLOW_DOWNLOAD=1 python3 L1_semantic_embedding_and_persistence.py`。
> **确定性**：同一快照 + CPU fp32 推理逐位确定；elapsed 行随运行时长变化，掩码口径承 L0 家族：`sed '/^[[:space:]]*elapsed/d'`。
> **产物卫生**：默认在临时目录完成 save/load 并自动清理；需要检查索引文件时，显式设置 `NANO_RAG_INDEX_DIR=/path/to/index`。
> **本文件是 notebook-style 教程**：叙述 + 代码摘录 + 真实运行输出 + 思考题交替推进。

---

## §1 K+1：L0 留下了哪两笔债

L0 用注定失败的 lexical embedding 把天花板裸了出来（tutorial_L0 §3/§5），又用纯内存索引把「索引 = 派生物」的语义简化到极致。这留下两笔债，正是生产检索系统 day-1 就会撞上的：

1. **语义鸿沟跨不过去。** car vs automobile 的 cosine = 0.0000（L0 §3 光谱第 3 点），两条刻意设计的纯同义词查询 MISS（L0 §5）。再大的哈希维度也救不了——必须换**函数类别**：从「规定出来的几何」（哈希）换成「学出来的几何」（神经模型）。
2. **进程一重启，索引蒸发。** L0 的 `FlatIndex` 全活在内存里。真实检索服务不可能每次启动都重新 embed 全部文档——embedding 是**昂贵的模型推理**（L0 的哈希是免费的，这是 L1 才暴露的成本结构变化），索引必须落盘、可重载、可增量追加。

L1 用四个机制面还清这两笔债（L1 验收标准，README 阶梯表）：

| # | 机制面 | nano 实现 | 真实系统对应 |
|---|--------|-----------|--------------|
| [1] | 真实神经 embedding：同评估集量化突破 | `load_real_embedder`（all-MiniLM-L6-v2，384 维） | 生产 embedding 模型服务（模型选型 = 检索天花板） |
| [2] | 索引文件级持久化 + 身份契约 | `PersistentFlatIndex.save/load`（meta/vec 两件分离） | 向量库的索引文件 + collection schema/元数据 |
| [3] | 增量 add：O(new) 模型调用 + 幂等 | `PersistentFlatIndex.add` | Milvus insert/upsert 语义（primary key 锚定） |
| [4] | 双路径 fallback（可运行性契约） | `embed_fallback`（显式 mock，零依赖） | 真实依赖不可用时的降级路径（机制同一套） |

刻意不模拟的（§12 列边界）：ANN 索引（HNSW/IVF）、混合检索（BM25+dense）、重排序、量化压缩（int8/PQ）、多租户 ACL——全部是 L2 课题。

---

## §2 先跑一遍：完整输出

```bash
$ python3 L1_semantic_embedding_and_persistence.py
```

```text
== nano-rag-retrieval L1: 真实神经 embedding + 索引持久化 + 增量 add（自 L0 的 K+1）==

[0] embedding 路径 = real（sentence-transformers/all-MiniLM-L6-v2；snapshot 1110a243fdf4…，dim 384，离线快照钉住）

[1] lexical 基线复刻（L0 算法逐字同一）——突破的参照系
  recall@1 = 0.625   recall@3 = 0.750   MRR = 0.6875
  MISS = ['d10', 'd14']（L0 刻意设计的两条纯同义词查询 = L1 的验收靶点）
  [check 01] PASS  lexical 基线与 L0 逐位一致: recall@1 0.625 / recall@3 0.750 / MRR 0.6875
  [check 02] PASS  lexical MISS 集合 == [d10, d14]

[2] embedding 几何修复验证（括号为 L0 lexical 录值，对照读）
  同义词 car/automobile      cos = 0.8645   (L0: 0.0000 全盲)
  同义词 bonuses/rewards     cos = 0.6854   (L0: 0.0000 全盲)
  词序 dog-man/man-bites-dog cos = 0.9072   (L0: 1.0000 全盲)
  词形 retrieve/retrieval    cos = 0.5978   (L0: 0.2449 trigram 桥)
  [check 03] PASS  同义词修复 car/automobile: cos ∈ (0.7, 1.01)
  [check 04] PASS  同义词修复 bonuses/rewards: cos ∈ (0.5, 1.01)
  [check 05] PASS  词序可见: 0.5 < cos < 1.0（严格小于 1 = 向量不再相同）
  [check 06] PASS  词形桥仍在: cos(retrieve, retrieval) ∈ (0.3, 1.01)

[3] 检索评估（同一 14 篇语料 + 同一 8 条标注查询，路径 = real）
  建索引: 14 篇 embed（模型调用 14 次），跳过 0
  recall@1 = 0.875（lexical 0.625）  recall@3 = 1.000（lexical 0.750）  MRR = 0.9375（lexical 0.6875）
    'reset my password please'                   gold=d01  rank 1
    'i was charged two times'                    gold=d02  rank 1
    'nearest neighbor query over embeddings'     gold=d03  rank 1
    'how to resume training after a crash'       gold=d09  rank 1
    'reduce hallucination with fetched documents' gold=d11  rank 1
    'remove near duplicate documents before training' gold=d12  rank 1
    'automobile fuel economy'                    gold=d14  rank 2
    'bonuses for correct behavior'               gold=d10  rank 1
  [check 07] PASS  recall@1 == 0.875
  [check 08] PASS  验收靶点: recall@3 == 1.0（L0 两条 MISS 全部进入 top-3）
  [check 09] PASS  MRR == 0.9375
  [check 10] PASS  MISS 修复一: 'bonuses for correct behavior' rank == 1
  [check 11] PASS  MISS 修复二: 'automobile fuel economy' rank == 2（进入 top-2）
  [check 12] PASS  'i was charged two times' rank == 1（L0: 被碰撞噪声压到 rank 2）

[4] 索引持久化（meta = 身份契约，vec = float32 向量段）
  nano_index.meta.json: 1970 B（model/snapshot/dim/文档清单）
  nano_index.vec.f32: 21504 B = 14×384×4(float32)（L0 的 float64 256 维 = 28672 B）
  [check 13] PASS  向量段落盘字节 == 14×384×4 = 21504
  [check 14] PASS  重载后向量逐字节一致（float32 往返无损）
  [check 15] PASS  重载后检索结果逐位一致（8 条查询 top-3 全同）
  [check 16] PASS  meta 钉住身份: model/dim/n_docs 与当前 embedder 一致
  身份契约: model=sentence-transformers/all-MiniLM-L6-v2  snapshot=1110a243fdf4…  dim=384

[5] embedding drift 治理：索引与模型是绑定契约
  篡改 meta.model 后加载 → IndexModelError: model mismatch: index built by 'some-other-model-v9', current embedder 'sentence-transformers/all-MiniLM-L6-v2'
  [check 17] PASS  模型身份不符 → 拒绝加载（宁可报错，不用旧坐标查新空间）

[6] 增量 add（d15/d16：L0 语料之外的新主题）
  add(NEW_DOCS): 新增 2 / 跳过 0；重复 add: 新增 0 / 跳过 2（幂等）
  模型调用增量 = 2 次（全量重建 16 篇需 16 次——增量省 14 次）
  [check 18] PASS  增量 add 只 embed 新文档: 模型调用增量 == 2
  [check 19] PASS  幂等: 重复 add 零新增
  [check 20] PASS  旧 14 篇向量逐字节不变（前缀比对）
    'how does an LRU cache decide what to evict'       gold=d15  rank 1
    'detecting conflicts between replicas with vector clocks' gold=d16  rank 1
  [check 21] PASS  新文档可检索: d15/d16 均 rank 1（16 篇索引）
  [check 22] PASS  增量不扰旧查询: 原 8 条 recall@3 仍 == 1.0

[7] 阈值 revisited（承 L0 §6：检索器要会说『不知道』）
  域外 'volcano eruption lava' top1 = d06@0.0898 → tau=0.25: 空手而归
  [check 23] PASS  域外查询 top1 分数 < 0.25 且被阈值挡住
  灰区 'automobile fuel economy'（tau=0.25）: 连同 gold 一起被滤掉——阈值工作点是业务决策（tutorial §8）

[8] 成本账本（toy 口径，承 L0 §7）
  存储: 16 篇向量 = 16×384×4 = 24576 B + meta/原文（payload 分离，L2 对照 Milvus segment）
  扫描: N×D = 6144 次乘加/query（L0: 14×256 = 3584）——暴力不 scale，ANN 是 L2 主题
  embed: 一次性建索引成本（全量 16 次模型调用）vs 每查询只 embed 查询一句——索引是摊销结构
  [check 24] PASS  索引字节公式: 16×384×4 == 24576

retrieval digest (real): 4c8c4347e418b602  两次独立构建逐位一致: True
  [check 25] PASS  确定性：两次独立构建索引 + 评估，digest 逐位一致

self-check: 25/25 PASS
```

demo 剧本：路径判定 → lexical 基线复刻（参照系）→ 几何修复验证 → 同集 recall 突破 → 持久化与重载 → 身份契约的牙齿（篡改 meta → 拒载）→ 增量 add（幂等 + 旧向量不动）→ 阈值 revisited → 成本账本 → 确定性 digest。

> **fixture 声明（承 L0，跨级锚设计）**：`CORPUS`（14 篇）与 `EVAL`（8 条标注查询）与 L0 **逐字同一**——突破必须有参照系，「同一语料、同一评估集」是 L1 全部数字可与 L0 直接相减的前提（check 01/02 把 L0 基线复刻成 self-check，参照系冻结进代码）。`NEW_DOCS`（d15 LRU 缓存 / d16 向量时钟）是 L0 语料之外的两个新主题，专供增量 add 验收：新主题保证「新增即可检索」有牙齿（若只 add 已有主题，幂等跳过后什么都验不到）。

---

## §3 机制面 [1]：真实神经 embedding —— 把「规定出来的几何」换成「学出来的几何」

L0 §3 的论证是：哈希维度开到多大也学不出语义等价，必须换函数类别。L1 换上真实小模型：

```python
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
def load_real_embedder():
    from sentence_transformers import SentenceTransformer  # 延迟 import：fallback 路径零依赖
    model = SentenceTransformer(MODEL_ID)
    dim = int(model.get_embedding_dimension())
    snapshot = "unknown"
    try:  # 溯源用：从 HF 缓存读快照 commit（离线可读，失败不致命）
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(MODEL_ID, "config.json")
        if isinstance(p, str):
            snapshot = os.path.basename(os.path.dirname(p))
    except Exception:
        pass
    def embed(text):
        return _f32(model.encode([text])[0].tolist())  # 单条编码：与批组成无关，逐位确定
    return embed, dim, snapshot
```

**（a）为什么选 all-MiniLM-L6-v2？** 它是「真实神经 embedding」的最小可用样本：6 层 MiniLM、384 维、约 22.7M 参数（由钉住快照中的张量文件字节账估算），CPU 可运行。
[模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)将它定位为把句子或段落映射到 384 维稠密空间、用于聚类和语义检索的 sentence-transformers 模型。

时效性定位：它是经典小模型——生产检索可能选更大的 instruct-embedding 模型，但「学习式 embedding 把语义等价映进几何邻近」这一机制不随模型大小改变。经典 ≠ 前沿，也不代表这里的 toy 指标可以外推。

**（b）快照钉住：可复现性是索引的一等公民（承 L0 §3(d)）。** 脚本在 import 前就设 `HF_HUB_OFFLINE=1`——默认离线，只用缓存里的快照；本次 real 路径使用 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，并由 [0] 段打印短锚。绝不隐式拉新快照：模型上游一更新，全部历史向量可能与旧索引失配（这正是 §6 的 drift 事故）。首次取快照必须显式 `NANO_RAG_ALLOW_DOWNLOAD=1`——**网络副作用是显式开关，不是默认行为**。dim 384 同时由模型配置与实际输出维度交叉确认。

**（c）单条编码的确定性。** `model.encode([text])[0]` 每次只编码一条——与批组成无关（批内 padding 长度会影响数值，这是真实 embedding 服务的确定性陷阱之一），CPU fp32 推理逐位确定。于是「同一快照 → 同一向量」成为可依赖的契约，§10 的 digest 靠它成立。

几何修复的实测光谱（§2 [2] 段，括号为 L0 录值）：

```text
  同义词 car/automobile      cos = 0.8645   (L0: 0.0000 全盲)
  同义词 bonuses/rewards     cos = 0.6854   (L0: 0.0000 全盲)
  词序 dog-man/man-bites-dog cos = 0.9072   (L0: 1.0000 全盲)
  词形 retrieve/retrieval    cos = 0.5978   (L0: 0.2449 trigram 桥)
```

四个点逐一读：**同义词从 0.0000 到 0.8645/0.6854**——语义等价第一次有了几何邻近性，这是 L1 存在理由的兑现；**词序从 1.0000 到 0.9072**——两句话的向量不再相同（严格 <1，check 05），但注意 0.9072 依然很高：Transformer 池化向量对词序的敏感度有限（同词袋、不同语序的句子本来语义就接近），L1 只要求「可见」（不再全盲），不夸大修复程度；**词形从 0.2449 到 0.5978**——subword tokenizer + 学习式表示把词形桥加宽了。

**思考题 3.1**：为什么 cos(car, automobile) = 0.8645 而不是 1.0？如果把它「修」到 1.0（比如对所有同义词对做对齐训练），检索会付出什么代价？（参考方向：1.0 意味着两个词在任何上下文里都不可区分——car 还有「火车车厢」「车厢级部署」等义项，automobile 没有；0.86 是「高度相关但保留语境差异」的几何表达。把同义词强行压到同一点 = 用多义性换同义性，下游查询的区分度会受损。embedding 的质量不是「同义词越近越好」，而是「该近的近、该远的远」——几何的保真度。）

---

## §4 同一语料 + 同一评估集：lexical → 语义的 recall 突破

L1 最硬的验收不是「跑通了」，而是**在同一参照系上的可量化进步**。§2 [3] 段的逐查询表读出三种命运：

**（a）6 条 rank-1 命中（L0 是 5 条）。** 新晋的一条是 `i was charged two times`：L0 里它的真信号（d02@0.2224）被哈希碰撞噪声（d01@0.2410）压到 rank 2（L0 §5(b)）；神经空间里没有哈希碰撞这回事——语义信号直接占主导，rank 1（check 12）。**换函数类别不只修同义词盲，连噪声面也一起修了**：L0 失败模式总账（L0 §6）里「碰撞假相似」与「弱信号被淹」两条同源于哈希，在 L1 同时消失。

**（b）验收靶点一：`bonuses for correct behavior` 从 MISS 到 rank 1。** 查询里没有任何 d10 的词面（L0：唯一共享词是停用词 `for`），全靠 bonuses↔rewards 的语义桥（cos 0.6854）。check 10。

**（c）验收靶点二：`automobile fuel economy` 从 MISS 到 rank 2——注意，只是 rank 2。** automobile↔car 的桥搭上了（cos 0.8645），但 d14 正文讲的是保养（oil change / tire rotation / brake inspection），**没有 fuel economy 的内容**——embedding 能桥接同义词，不能虚构语料里不存在的相关性。rank 2 是诚实的几何结果：语义桥把 d14 从「连 top-4 都进不了」拉进 top-2，但没把它拉到 rank 1，因为查询的另一半信号（fuel economy）在 d14 里确实弱。check 11 的验收线定在「进入 top-2」而非 rank 1，正是为了不夸大突破。

总账：recall@1 0.625 → **0.875**，recall@3 0.750 → **1.000**，MRR 0.6875 → **0.9375**（= (7×1 + 1/2)/8，一条 rank-2 贡献 1/2）。三个增量全部落在 L0 逐查询表的具体行上——「模型更好」从口号变成了可证伪的断言。

```python
def evaluate(index, eval_set, k):  # 承 L0 逐字同一：参照系必须是同一把尺子
    ...
```

`evaluate` 与 lexical 基线函数 `embed_lexical`（L0 `embed` 的逐字搬运，含 md5 哈希与 256 维设定）都原封不动——**同集对照的基线必须是同一个函数**，否则「突破」里混进实现差异，不可归因。

**思考题 4.1**：如果把 d14 的正文改成同时覆盖保养与油耗（加一句 "fuel economy depends on tire pressure and regular engine checks"），预测 `automobile fuel economy` 的 rank 变化；再预测：把查询加长为 `automobile fuel economy and yearly maintenance schedule`，rank 又会怎么变？（参考方向：前者 rank 2 → 1——语料侧补上了缺失的信号面；后者 gold 不变但查询里 maintenance 信号增强，d14 分数进一步拉开与 top-1 竞争者的差距。两问合起来说明：检索质量是「查询 × 语料 × embedding」三者的乘积，embedding 只负责其中一项——语料没写的东西，换再大的模型也检索不出来。这是 RAG 系统「先治语料再调模型」优先级的几何依据。）

---

## §5 机制面 [2]a：索引持久化 —— 从「内存派生物」到「落盘契约」

L0 的索引是进程内的派生物；L1 把它拆成两件落盘：

```python
META_FILE = "nano_index.meta.json"   # 身份契约 + 文档清单（人类可读，控制面）
VEC_FILE = "nano_index.vec.f32"      # float32 向量段（机器热路径，数据面）
FORMAT = "nano-rag-index/v1"

def save(self):
    meta = {"format": FORMAT, "model": self.model_name, "snapshot": self.snapshot,
            "dim": self.dim, "dtype": "float32", "n_docs": len(self.docs),
            "docs": [{"id": did, "text": text} for did, text in self.docs]}
    with open(META_FILE, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
    with open(VEC_FILE, "wb") as f:
        f.write(self.vec_bytes())
```

三个设计决策，逐个拆开：

**（a）为什么 meta 与 vec 两件分离？** 两件东西的读者不同：**meta 是控制面**——人可读（出事故时 `cat` 一眼看到索引用哪个模型建的、多少篇、什么快照）、加载时先验证再读数据；**vec 是数据面**——纯二进制热路径，`array.array("f").frombytes` 一次读尽，零解析开销。真实向量库同构：Milvus 的索引/数据文件（segment）与元数据分层管理，manifest 类结构描述「有哪些文件、各属于哪个版本」，本模块的 meta/vec 是它的最小形态（行号级源码锚点 `[TODO: verify L2 源码锚点]`，L2 展开）。混成一件的代价：每次想知道「这索引是谁建的」都得解析整个向量段。

**（b）为什么 float32 而不是沿用 L0 的 float64？** 量化契约：`_f32` 在**向量产出时**统一量化（`array.array("f", v)`），于是内存 == 落盘 == 重载三者逐位同一——check 14（重载后向量逐字节一致）与 check 15（重载后 8 条查询 top-3 全同）靠它成立。若内存用 float64、落盘截断成 float32，重载就是**有损往返**，「索引可重建」退化为「索引近似可重建」。存储账随之变化：14×384×4 = **21,504 B**，比 L0 的 14×256×8 = 28,672 B 还小——**维度升了 50%，字节反而降了 25%**，全靠 64→32 位宽（思考题 9.1）。float32 不是终点：int8/PQ 量化是生产向量库的存储-精度旋钮，L2 对照 Milvus 量化实现。

**（c）`del idx` 是模拟的一部分。** [4] 段在 save 之后显式 `del idx`（内存态蒸发），再从盘上 `load` 出 `idx2`——重载后的 check 15/16 证明「进程退出后世界还在」。L0 做不到这个演示：内存派生物没有「之后」。

```text
  nano_index.meta.json: 1970 B（model/snapshot/dim/文档清单）
  nano_index.vec.f32: 21504 B = 14×384×4(float32)（L0 的 float64 256 维 = 28672 B）
  [check 14] PASS  重载后向量逐字节一致（float32 往返无损）
  [check 15] PASS  重载后检索结果逐位一致（8 条查询 top-3 全同）
```

**思考题 5.1**：meta 里为什么连原文（`docs[].text`）也存？向量不是已经「包含」文档内容了吗？（参考方向：向量是**有损压缩**——384 维浮点承载不了原文的逐字信息。展示（给用户看命中了哪段）、重排（cross-encoder 需要原文）、调试（为什么这篇排第一）都要原文。meta 存原文让索引自含（self-contained），代价是 meta 变大；真实系统的选择是 payload 单独存储（对象存储/文档库），meta 只存 id——§9 的 payload 分离与 L2 的 Milvus segment 结构正是这个权衡的工业形态。）

---

## §6 机制面 [2]b：身份契约 —— 模型身份是索引的一部分

meta 里最容易被省略、却最不能省的三个字段：`model` / `snapshot` / `dim`。原因一句话：**向量只有相对产出它的模型才有意义。** 384 维空间里的一个点，换了模型就是另一个坐标系里的另一组数——直接拿旧坐标查新空间，得到的相似度是几何上无意义的噪声，而且**不会报错**：检索照常返回 top-k，只是全是垃圾。这比崩溃更糟——静默错误。

这类事故有名字：**embedding drift**（模型升级/漂移导致存量索引与新模型不匹配）。L1 的对策是给契约装上牙齿——加载时比对身份，不符即拒：

```python
@classmethod
def load(cls, embed, dim, model_name):
    with open(META_FILE) as f:
        meta = json.load(f)
    if meta.get("format") != FORMAT:
        raise IndexModelError(f"format mismatch: {meta.get('format')!r} != {FORMAT!r}")
    if meta.get("model") != model_name:
        raise IndexModelError(  # 换模型不重建索引 = 用旧坐标查新空间，静默垃圾
            f"model mismatch: index built by {meta.get('model')!r}, current embedder {model_name!r}")
    if meta.get("dim") != dim:
        raise IndexModelError(f"dim mismatch: index dim {meta.get('dim')}, embedder dim {dim}")
    ...
```

§2 [5] 段把牙齿亲跑给你看：篡改 `meta.model` 为 `some-other-model-v9` 再加载 → `IndexModelError` 拒载（check 17），然后恢复真身。**宁可报错，不用旧坐标查新空间。**

权威系统在同一问题上的选择可以对照。Milvus 的 embedding function 机制把「写入与查询必须用同一个模型」做成了平台级保证——collection 绑定 embedding function，查询时平台代你生成查询向量（Milvus 官方文档 *Embedding Function Overview*，milvus-io/web-content `v2.6.x/site/en/userGuide/embeddings-reranking/embedding-function/embedding-function-overview.md`，2026-08-15 抓取 21,506 B，逐字引文）：

> "Milvus generates the query vector with the same model you used for ingestion, compares it to the stored vectors, and returns the most relevant results."

注意这句的潜台词：真实系统不是在文档里提醒「请自行保证模型一致」，而是**把一致性做成机制**——你根本拿不到「用错模型」的机会。nano 版的 meta 比对是同一思想的最小形态：平台用绑定消灭错误类别，nano 用加载时校验拦截错误实例。OpenSearch 的 k-NN 索引则把维度一致性钉在建索引时（官方文档 *Creating a vector index*，https://docs.opensearch.org/latest/vector-search/creating-vector-index/ ，2026-08-15 抓取 439,413 B，标题 "Creating a vector index | OpenSearch Documentation"，去标记口径逐字引文）：

> "Specify the dimension: Set the dimension property to match the size of the vectors used."

dim 在 mapping 里一次定死——之后维度不符的向量根本写不进来。nano 的 `dim mismatch` 拒载是它的加载时对应物。

安全面（§七 first-class）：身份契约也是检索器的**完整性边界**。meta 是盘上的明文 JSON，任何能写盘的人都能改——篡改 meta 而不改 vec，就是一次「供应链投毒」的最小形态（让检索器误信索引的身份）。L1 的校验至少让「投毒后静默生效」变成「投毒后加载被拒/行为可检测」；真实系统的对策是索引文件的 content hash + 签名 + 存储层 ACL——与 nano-data-platform 的消费侧 default-deny（`../nano-data-platform/tutorial_L0.md` §7）是同一治理问题在检索层的投影，L2 展开。

**思考题 6.1**：业务真的要升级 embedding 模型（比如从 all-MiniLM-L6-v2 换更大的模型），正确姿势是什么？为什么「原地把 meta.model 改成新模型」是最坏选项？（参考方向：正确姿势 = 用新模型对全部原文重新 embed，建新索引，双跑比对评估集（§4 的同集方法），然后切换——蓝绿替换。原地改 meta = 把旧坐标谎报成新坐标系，收获 §6 的静默垃圾。这也解释了为什么「模型升级」在向量检索里是**批处理运维操作**而不是热更新：成本账本（§9）里 embed 是一次性建索引成本，升级就是再付一次。）

---

## §7 机制面 [3]：增量 add —— 只 embed 新文档，旧向量逐字节不动

索引落盘之后，下一个 day-1 问题：新文档来了怎么办？全量重建 = 把 14 篇重新 embed 一遍，只为了加 2 篇——embed 是模型推理（§9 的成本账），重建成本随存量线性上涨。L1 的 `add` 是增量的：

```python
def add(self, pairs):
    """幂等增量：已收录 id 跳过，只 embed 新文档。返回 (新增数, 跳过数)。"""
    known = {did for did, _ in self.docs}
    added = skipped = 0
    for did, text in pairs:
        if did in known:
            skipped += 1
            continue
        self.docs.append((did, text))
        self.vecs.append(self.embed(text))
        self.embed_calls += 1
        added += 1
    return added, skipped
```

§2 [6] 段验证三个性质，每个都有牙齿：

**（a）O(new) 模型调用。** add(d15, d16)：模型调用增量 == 2（check 18），不是 16。`embed_calls` 是显式的成本计数器——增量省下的不是抽象的「计算」，是**具体的 14 次模型推理**。这与 nano-data-platform L1 的水位线增量同步（`../nano-data-platform/tutorial_L1.md` §4：游标之后才拉）是同一治理思想在两层投影：**只处理新增，不重扫世界**。

**（b）幂等：已收录 id 跳过。** 同一批 NEW_DOCS 再 add 一次：新增 0 / 跳过 2（check 19）。id 是幂等锚——投递可以是 at-least-once（重试安全），存储语义仍然是 exactly-once（与 platform L1 §4(b) 的 PK 去重同款组合）。

**（c）旧向量逐字节不变。** check 20 做的是**前缀字节比对**：增量后的 `vec_bytes()` 前 21,504 B 与增量前逐位相同。追加式写入（append-only）不触碰存量——存量查询的可复现性不依赖「相信 add 没改旧数据」，而是字节级可验证（check 22：原 8 条 recall@3 仍 == 1.0）。

对照 Milvus 的写入语义（官方文档 *Upsert Entities*，milvus-io/web-content `v2.6.x/site/en/userGuide/insert-and-delete/upsert-entities.md`，2026-08-15 抓取 31,958 B，逐字引文）：

> "You can use `upsert` to either insert a new entity or update an existing one, depending on whether the primary key provided in the upsert request exists in the collection. If the primary key is not found, an insert operation occurs. Otherwise, an update operation will be performed."

同与不同要分清：**同**在 primary key（nano 的 doc id）是幂等锚，两边都用它决定「新增还是别的」；**不同**在「别的」是什么——Milvus upsert 对已存在的 PK 执行**覆写**（insert + delete 原行），nano 的 add 对已存在的 id 执行**跳过**。nano 选跳过是因为本模块的文档是**不可变文本**（语料版本化，承 nano-data-platform L0 的快照语义）：同一 id 的文本永不改变，「更新」语义不存在。若文档会改（比如 FAQ 改写），跳过就是错的——需要 delete-then-add 或 upsert 覆写 + 重新 embed，L2 对照 Milvus upsert 实现展开。

**思考题 7.1**：如果 d15 的文本被修订了（同一 id、新内容），用本实现的 add 重投会发生什么？给出最小修复（不改数据结构），并说明修复后哪个 check 会替你把关。（参考方向：重投被幂等跳过——旧向量留着，新文本永远进不了索引，静默陈旧。最小修复 = add 前比对已知文本，文本不一致时先移除旧条目再 embed 新文本；check 20 的前缀不变断言会立刻变红提醒你「存量被动了」，check 21/22 负责验证修订后仍可检索且不扰旧查询。）

---

## §8 阈值工作点是业务决策

代码 [7] 段的输出把 L0 §6 的阈值话题搬到神经空间重审：

```text
  域外 'volcano eruption lava' top1 = d06@0.0898 → tau=0.25: 空手而归
  [check 23] PASS  域外查询 top1 分数 < 0.25 且被阈值挡住
  灰区 'automobile fuel economy'（tau=0.25）: 连同 gold 一起被滤掉——阈值工作点是业务决策（tutorial §8）
```

先看好消息：神经空间的噪声底干净得多。域外查询 `volcano eruption lava` 的 top-1 分数 0.0898（L0 lexical 是 0.1448）——学习式几何里，无关内容与语料的「偶然相似」更低，阈值更容易选。check 23：tau=0.25 挡住域外查询，「不知道」语义在 L1 依然成立（承 L0 §6：宁可空手，不喂垃圾给生成器）。

再看灰区，这是本节的主角：`automobile fuel economy` 的真命中 d14 分数在 0.25 附近——tau=0.25 时**连同 gold 一起被滤掉**。§4 说过它只是 rank 2（信号本就不强），现在看到后果：阈值每上调一格，都在同时杀垃圾与杀弱真信号。**tau 不是一个有「正确答案」的超参数，而是 precision/recall 曲线上的工作点选择，选择依据在模型之外——在业务里**：

- 下游是 RAG 生成（检索结果直接进 context）：喂垃圾的代价 = 幻觉燃料（L0 §1 的链路位置），宁可多漏、不可多喂 → tau 偏高；
- 下游是人工复核列表（检索结果给人看）：漏掉的代价 = 用户查无结果而库里有，多给几条人来筛 → tau 偏低；
- 域外查询的比例、弱真信号的业务重要性，都只有业务方知道。

所以代码输出里那句「阈值工作点是业务决策」不是推卸——是机制的边界声明：检索器能给出**分数的几何**（0.0898 的噪声底、0.25 附近的灰区），工作点选在哪由「漏检 vs 误报」的代价不对称决定。真实系统的完整形态是：在标注集上扫 precision-recall 曲线（L0 §6 已提）+ 按业务代价选点 + 上线后监控灰区比例漂移——L2 的重排序（cross-encoder 复打分）则是把「分数校准」本身升级的路线。

**思考题 8.1**：给一个可操作的 tau 校准方案：你手上只有 §2 的 8 条标注查询和 [7] 段的域外查询，如何选 tau？如果明天语料从 16 篇涨到 16 万篇，这个方案还成立吗？（参考方向：在标注集上扫 tau（0 到 1），画 recall@k-tau 与域外拦截率两条曲线，选业务代价最小的交点。语料规模涨四个量级后不成立：噪声底随语料变化（更多文档 = 域外查询的「最近邻」分数统计上会抬高——极值效应），tau 必须随索引规模重新校准。固定 tau 不是尺度不变量——这正是 L0 §6「噪声底随查询漂移」在规模维度的孪生兄弟。）

---

## §9 成本账本（toy 口径，承 L0 §7）

成本是机制的一部分，不是附录。L1 的成本结构相对 L0 发生了质变——**embedding 从免费哈希变成模型推理**，三本账都要重记：

```text
  存储: 16 篇向量 = 16×384×4 = 24576 B + meta/原文（payload 分离，L2 对照 Milvus segment）
  扫描: N×D = 6144 次乘加/query（L0: 14×256 = 3584）——暴力不 scale，ANN 是 L2 主题
  embed: 一次性建索引成本（全量 16 次模型调用）vs 每查询只 embed 查询一句——索引是摊销结构
```

**（a）存储账**：16×384×4 = 24,576 B（增量后）。对照 L0 的 28,672 B（14×256×8）：维度 +50%、篇数 +2，字节反而更少——float64→float32 的位宽减半赢过了维度上涨（思考题 5.1(b)/9.1）。meta 另计 1,970 B（含原文，§5 的自含设计）。真实系统的存储大头与量化位宽直接挂钩——这是 Milvus 量化选项（SQ/PQ）存在的理由 `[TODO: verify L2 源码锚点]`。

**（b）计算账（查询侧）**：暴力扫描 N×D = 16×384 = 6,144 次乘加/query，随 N 线性（承 L0 §7(b) 的外推：N=10^6 时暴力法不成立）。L1 没有改善这一项——flat 精确 kNN 依然是基线（Milvus 文档的 FLAT = Brute-Force 判据承 L0 §4，本轮重抓坐实），改善它是 L2 的 ANN 主题。

**（c）计算账（写入侧）——L1 新增的一本账**：embed 是一次性建索引成本。全量建 16 篇 = 16 次模型调用；之后每查询只需 embed 查询一句（1 次调用），文档向量复用——**索引是摊销结构**：把昂贵推理从查询路径挪到建索引路径，摊薄到所有未来查询。这本账直接推出三个工程结论：① 增量 add（§7）省的是真金白银的推理，不只是 CPU；② 模型升级 = 重付一次全量 embed（§6 思考题 6.1），所以是批处理运维；③ 查询侧的 embedding 调用也有成本——高频查询值得缓存查询向量（本实现未做，真实系统的 query embedding cache）。

**思考题 9.1**：把三本账合起来算一个场景：N = 10^6 篇、每天新增 10^4 篇、每天 10^6 次查询。全量重建一次 vs 增量 add 一天，embed 调用各是多少次？查询路径每天 embed 多少次？哪本账最大？（参考方向：全量重建 = 10^6 次调用/次；增量 = 10^4 次/天，差两个量级；查询侧 = 10^6 次/天（每查询 embed 一句）——与增量同量级，查询向量缓存因此值钱；最大的账是扫描：10^6×384 次乘加/查询 × 10^6 查询/天，暴力法直接不成立——ANN（L2）的动机在 L1 的成本账里已经写好了。）

---

## §10 确定性锚点：双路径口径分离

```text
retrieval digest (real): 4c8c4347e418b602  两次独立构建逐位一致: True
  [check 25] PASS  确定性：两次独立构建索引 + 评估，digest 逐位一致
```

确定性的来源承 L0 §3(d)/§8（md5 而非内建 hash、无随机、无 wall-clock 进 digest），L1 新增两条纪律：

**（a）快照钉住是确定性的前提。** 真实路径的逐位可复现依赖「同一模型快照」——HF_HUB_OFFLINE=1 默认离线（§3(b)）。没有快照钉住，「确定性」只在上游不更新的窗口内成立。

**（b）双路径 digest 分离，不混口径（反幻觉纪律）。** real 与 fallback 是两个不同的函数，digest 天然不同：real `4c8c4347e418b602`，fallback `008632d5402a86b7`——各自内部两次独立构建逐位一致，但**两者之间不可比**。代码里两条路径的期望值表（`EXP`）分开声明；把「fallback 的 1.0 recall」与「real 的 0.875 recall」放在一起比大小是口径错误：fallback 是显式 mock，其 1.0 来自手写同义词簇表对本评估集的**定向覆盖**（「开卷」），不是学到的泛化。fallback 只演示**机制管线**，不是模型能力。

**运行锚点**（2026-08-31，`-B`）：代码 md5 `a24ad13c8e1c206c650a436d2783be9c`/423 行/25,017 B。real 路径在两个新建空 CWD 运行，均 EXIT=0、stderr 0 B；掩码输出 md5 `9872b395c1a20d3b5cfafd982e37868a`/77 行/5,138 B，逐字节一致，25/25 PASS。fallback 路径用两个隔离的空 HF cache 强制触发，掩码输出 md5 `d5fc765dccb42fa22d057a139d58c620`/78 行/5,216 B，同样逐字节一致，25/25 PASS。四个 CWD 均未残留 `nano_index.*`；§2 paste 块与 real 掩码输出逐字节一致。

---

## §11 费曼自检

**讲给外行听**：L0 的小图书馆来了位只数词频的馆员，把《汽车保养》和《automobile 手册》分到了两个世界。L1 换了一位**真读过书的绘图员**（神经模型）：她给每本书一个按内容画的坐标（§3），于是「汽车」和「automobile」终于成了邻居（0.0000 → 0.8645），「狗咬人」和「人咬狗」也不再是同一本书（1.0000 → 0.9072）。但有三条新规矩。其一，**坐标和绘图员是绑定的**：坐标册（vec）旁边必须放一张身份卡（meta），写明坐标是哪位绘图员、哪个版本画的；换绘图员不重画坐标册，等于拿旧北京地图在上海找路——图书馆宁可拒绝开馆也不给你错的坐标（§6）。其二，**坐标册存档了**：闭馆（进程退出）后账本不蒸发，开馆时从档案重载，逐字节核对无误（§5）。其三，**新书只需画新坐标**：不用把全馆的书重画一遍，而且旧坐标一个字节都不动；同一本书送两次，第二次直接签收跳过（§7）。她还保留了一个美德：你问「火山喷发」，馆里没有，她宁可说没有，也不拿一本 0.0898 分的书糊弄你（§8）——至于「多低的分数算没有」，得由图书馆的业务来定，不是她一个人说了算。

**思考题汇总**（正文内另有 3.1 / 4.1 / 5.1 / 6.1 / 7.1 / 8.1 / 9.1）：

1. 一句话说清：「向量索引和模型是绑定契约」——哪个 check 是这句话的牙齿？没有牙齿的契约（比如只在文档里写「请勿混用模型」）与有牙齿的契约，事故形态差在哪？（check 17 的拒载；差别 = 静默垃圾 vs 响亮失败——前者是检索照常返回但全是噪声，可能几天后才被业务指标发现；后者在加载时就停摆。）
2. 本实现里哪三个东西分别对应 Milvus 的「embedding function 绑定」「upsert 的 primary key 语义」「索引与元数据分层」？（meta 的 model/snapshot/dim 校验 + load 拒载 / `add` 的 id 幂等锚 / meta.json 与 vec.f32 两件分离——L2 对照源码展开。）
3. fallback 路径跑出 recall 1.0，能不能写进汇报说「我们的检索器达到 1.0 recall」？（不能。fallback 是显式 mock，1.0 是手写簇表对评估集开卷覆盖的结果（§10(b)）——汇报口径必须声明路径；混口径本身就是反幻觉纪律要拦截的事故。）

**反例（一个常见错误直觉）**：「换了更强的 embedding 模型，索引不用重建——把 meta 里的模型名改改就行，向量是通用的。」——错在把向量当成了与模型无关的「内容指纹」。向量是**特定模型这个坐标系下的坐标**：all-MiniLM-L6-v2 的 384 维空间与任何另一个模型的空间之间没有逐维对应关系，旧坐标在新坐标系里的「相似度」是两组无关联数字的点积——几何上无意义，而且检索照常返回结果（静默垃圾，比崩溃危险）。§6 的 check 17 就是为这个错误直觉准备的牙齿。诚实的另一半也要说清：真实系统确实存在「不重建」的路径——但那是用新模型对原文**重新 embed**（蓝绿替换，思考题 6.1），不是复用旧向量；「重新 embed 要付全量推理成本」（§9(c)）正是模型升级必须当批处理运维来做、而不能热更的原因。

---

## §12 边界与下一站

**模拟了**（本教程验收内容）：真实神经 embedding（all-MiniLM-L6-v2，快照钉住、离线可复现）；同一语料 + 同一评估集上 lexical → 语义的量化突破（recall@1 0.625→0.875 / recall@3 0.750→1.000 / MRR 0.6875→0.9375，L0 两条 MISS = 验收靶点）；索引文件级持久化（meta/vec 分离 + float32 往返无损 + 重载逐位复验）；身份契约与 embedding drift 治理（篡改 meta → 拒载）；增量 add（O(new) 模型调用 + 幂等 + 旧向量字节不变）；阈值工作点的业务决策语义（灰区演示）；成本账本三本（存储/扫描/embed 摊销）；双路径确定性（digest 分离口径）。

**刻意没模拟**（每行都是更高阶梯的课题）：

| 没模拟 | 为什么 L1 不做 | 哪一级做 |
|--------|----------------|----------|
| ANN 索引（HNSW/IVF） | flat 精确 kNN 是 recall 裁判，先有基线才谈近似（承 L0 §4 Milvus FLAT 判据） | L2 对照 Milvus/OpenSearch/Weaviate 源码 |
| 混合检索（BM25 + dense 融合） | L0 §5(a) 的词面强信号是它的动机，机制到 L2 才完整 | L2 |
| 重排序（cross-encoder reranker） | 需要第二段模型与两阶段管线 | L2 |
| 量化压缩（int8/SQ/PQ） | float32 契约先立住（§5(b)），再谈有损压缩的 recall 代价 | L2 对照 Milvus 量化实现 |
| 更新语义（文档修订 → 向量更新） | 本模块文档不可变（§7 对照 Milvus upsert 已讲清边界） | L2 |
| 多租户 ACL / 索引签名 | toy 语料无敏感面；完整性机制已给最小形态（§6 安全面） | L2（与 nano-data-platform 治理合流） |

**L2 预告**（README 阶梯表 L2 行）：对照 Milvus（FLAT/HNSW/IVF 索引族、segment 与 payload 分离）+ OpenSearch（k-NN + BM25 混合）+ Weaviate（HNSW 实现）做取舍分析；混合检索 + 重排序 + recall/latency 权衡实测，并明确标注 toy 与真实集群边界。本级的 flat 索引、meta/vec 分离与身份契约，都会在 L2 找到工业级对应物。

---

## §13 溯源

| 声明 | 类型 | 来源 |
|------|------|------|
| all-MiniLM-L6-v2 = 384 维 sentence-transformers 模型（§3 引文） | 文献已有（逐字引文） | Hugging Face 模型卡 `sentence-transformers/all-MiniLM-L6-v2` README.md raw，2026-08-15 抓取（10,502 B）；「384 dimensional dense vector space」句逐字 |
| dim 384 = config.json hidden_size；约 22.7M 参数来自钉住快照的张量字节账；snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`（§3/§10） | 当前运行环境实测 | 快照配置、张量文件字节账与代码 [0]/[4] 输出交叉确认；属于该 revision 的实测，不是模型族永久常数 |
| Milvus 查询向量与写入同模型（§6 引文） | 文献已有（逐字引文） | Milvus 官方文档 *Embedding Function Overview*：milvus-io/web-content `v2.6.x/site/en/userGuide/embeddings-reranking/embedding-function/embedding-function-overview.md`，2026-08-15 抓取（21,506 B） |
| Milvus upsert 的 primary key 语义（§7 引文） | 文献已有（逐字引文） | Milvus 官方文档 *Upsert Entities*：milvus-io/web-content `v2.6.x/site/en/userGuide/insert-and-delete/upsert-entities.md`，2026-08-15 抓取（31,958 B） |
| OpenSearch k-NN 索引 dimension 建索引时钉死（§6 引文） | 文献已有（去标记口径逐字引文） | https://docs.opensearch.org/latest/vector-search/creating-vector-index/ ，2026-08-15 抓取（439,413 B，标题 "Creating a vector index \| OpenSearch Documentation"；源页 `<strong>`/`<code>` 标记已去） |
| Milvus FLAT = Brute-Force = most accurate + filter ratio 判据（§9(b)，承 L0 §4） | 文献已有（L0 逐字引文，本轮重抓坐实） | milvus-io/web-content `v2.6.x/site/en/userGuide/indexes/index-explained.md`，2026-08-15 重抓（17,894 B，与 L0 2026-08-13 录值零漂移） |
| Weaviate 为权威参照实现（§12/L2 预告） | 参照范围 | 本级只列为后续源码对照对象；没有用未核验页面支撑具体事实 |
| RAG [2005.11401] / feature hashing [0902.2206] / HNSW [1603.09320]（承 L0 §1/§3/§4） | 文献已有（L0 已溯源） | `tutorial_L0.md` §11 溯源表（arXiv abs 页 2026-08-13 重抓核验） |
| embedding drift = 模型升级导致存量索引失配的事故类别（§6） | 机制类别归纳（合理推断） | 未引具体事故报告；机制由 §6 篡改实验（check 17）自证 |
| 全部实测数字（0.8645 / 0.6854 / 0.9072 / 0.5978 / recall 0.875·1.000 / MRR 0.9375 / 0.0898 / 21,504 B / 24,576 B / 1,970 B / 6,144 ops / digest `4c8c4347…`·`008632d5…` 等） | 本实现实测（toy 设定） | `L1_semantic_embedding_and_persistence.py` 当前运行输出（§2 paste 块与 real 掩码输出 BYTE-IDENTICAL），非真实系统 benchmark、不可外推 |
| 「索引/元数据分层」「payload 分离」对应真实向量库结构（§5(a)/§9(a)） | 合理推断（机制同类） | 概念性提及；行号级源码锚点 `[TODO: verify L2 源码锚点]` |

下一站：**L2**——对照 Milvus/OpenSearch/Weaviate 源码的 ANN（HNSW/IVF）取舍分析 + 混合检索（BM25+dense）+ 重排序 + 量化 + recall/latency 权衡实测（见 README 阶梯表）。
