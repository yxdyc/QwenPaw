# nano-rag-retrieval L2 — 混合检索 + HNSW + 两阶段重排 + 检索评估升级（自 L1 的 K+1）

> **定位**：L1 用真实神经 embedding 把 recall 推过了同义词鸿沟，但索引仍是 flat 暴力扫描（O(N·D)/query），且只有单路信号、没有重排、评估只问「找没找到」。L2 一次还清 L1 §12 债表的前三笔债 + 评估升级：**ANN 索引（HNSW 从零实现）/ 混合检索（BM25+dense 融合）/ 重排序（两阶段）/ 评估升级（nDCG 分级）**，全部对照权威实现源码（Lucene BM25Similarity / hnswlib / OpenSearch / Elastic / Milvus）做取舍分析。
> **可运行性契约**：本级 = **可运行的本质模拟 + 显式注明**——BM25/HNSW/融合/分片/重排/nDCG 全部是从零写的实现，数字来自脚本运行；reranker 是显式 mock cross-encoder（只验证管线机制，不冒充语义质量），真实集群行为不在本级证据范围内。
> **双路径契约**（承 L1 §10(b)）：real 路径 = sentence-transformers + all-MiniLM-L6-v2 本地快照（离线钉住）；无依赖/无快照自动走 fallback（零依赖显式 mock）。**两路径期望值分开声明，数字不混比**。

**运行锚点**（2026-08-31 重新复验）：

| 锚 | 值 |
|----|----|
| 代码 | `L2_hybrid_search_hnsw_and_rerank.py` md5 `3d0f75ffd50951c562820db492ba072d` / 777 行 / 47,570 B |
| real 路径掩码输出 | md5 `79ae241041f5f69609ed17ac90544202` / 86 行 / 7,651 B（2×新建空独立 CWD BYTE-IDENTICAL） |
| fallback 路径掩码输出 | md5 `82bac3cae504a3afe0ff5c9aaf28e63e` / 87 行 / 7,724 B（2×新建空独立 CWD BYTE-IDENTICAL） |
| retrieval digest（路径无关核） | `7823a73786681b38`（双路径吻合，两次独立构建逐位一致） |
| 跨级锚（L1 digest 复现） | real `4c8c4347e418b602` / fallback `008632d5402a86b7`（与 tutorial_L1 §10 录值逐位吻合） |
| self-check | 34/34 PASS（双路径）；掩码口径 `sed '/^[[:space:]]*elapsed/d'`（承 L0 家族） |

---

## §1 K+1：L1 留下了哪四笔债

L1 教程 §12「边界与下一站」的债表里，有三行直接指向本级：**ANN 索引（HNSW/IVF）/ 混合检索（BM25 + dense 融合）/ 重排序（cross-encoder reranker）**。加上评估侧的第四笔债（recall/MRR 只问「找没找到」，不问「排得好不好」——L1 §8 阈值工作点讨论时已露头的 nDCG 需求），就是 L2 的全部议程：

| 债 | L1 的现状 | L2 的回答 |
|----|-----------|-----------|
| ① flat 扫描 O(N·D)/query | 14 篇语料无所谓，N 上去就是线性墙 | [4] HNSW 从零实现：recall/距离计算量权衡曲面实测（§6） |
| ② 单路信号 | 只有 dense 一路；L0 §5(a) 的词面强信号（'charged' → d02 rank 1）被浪费 | [2][3] BM25（Lucene 公式）+ 两种工业融合术（§4–§5） |
| ③ 无重排 | top-k 直接出自索引 | [6] 两阶段：粗召回定天花板、精排定位次（§8） |
| ④ 评估只有 recall/MRR | 二元相关性 | [7] nDCG@k 分级相关性（§9） |

L1 §12 债表的另外三行——量化压缩 / 更新语义 / 多租户 ACL——**本级刻意不做**（边界声明见 §13 与 §18），其中量化机制以「粗召回+精排」同构在 [6] 触及。

---

## §2 先跑一遍

```bash
python3 L2_hybrid_search_hnsw_and_rerank.py          # real 路径：需 sentence-transformers + 本地快照（承 L1 钉住口径）
HOME=$(mktemp -d) python3 L2_hybrid_search_hnsw_and_rerank.py   # fallback 路径探针：空 HOME 逼离线失败 → 零依赖 mock
```

核心机制路径零依赖（BM25/HNSW/融合/分片/重排/nDCG 纯标准库，CPU ~10–20s）；模型只进 dense 一路。确定性：md5 驱动一切「随机」（层级分配 / 合成语料），无内建 `hash()`、无未钉种子 RNG、无 wall-clock 进 digest；elapsed 行可掩码。

以下是 real 路径（snapshot `1110a243fdf4…`）在新建空独立 CWD 的完整掩码输出（`sed '/^[[:space:]]*elapsed/d'`，与 RUN1 逐字节同一，md5 `79ae241041f5f69609ed17ac90544202`/86 行/7,651 B）：

```text
== nano-rag-retrieval L2: 混合检索 + HNSW + 两阶段重排 + 检索评估升级（自 L1 的 K+1）==

[0] embedding 路径 = real（sentence-transformers/all-MiniLM-L6-v2；snapshot 1110a243fdf4…，dim 384，离线钉住）

[1] 跨级锚：L1 的 lexical 基线与 digest 必须逐字复现（同函数同 fixture）
  lexical 基线: recall@1 = 0.625 / recall@3 = 0.750 / MRR = 0.6875，MISS = ['d10', 'd14']
  [check 01] PASS  跨级锚: lexical 基线与 L0/L1 逐位一致 (0.625 / 0.750 / 0.6875)
  [check 02] PASS  跨级锚: MISS 集合 == [d10, d14]（L0 刻意设计的两条纯同义词查询）
  L1 digest (real): 4c8c4347e418b602  期望: 4c8c4347e418b602
  [check 03] PASS  跨级锚: L1 retrieval digest (real) 逐字复现

[2] BM25 稀疏打分器（Lucene BM25Similarity 公式：k1=1.2 / b=0.75）
  [check 04] PASS  BM25 手算锚: score(A) == ln(1.2)·1/(1+0.9)（闭式独立重算）
  [check 05] PASS  BM25 手算锚: score(B) == ln(1.2)·2/(2+1.5)（tf=2 但长度 4）
  [check 06] PASS  BM25 tf 饱和: 词频 2 的得分 < 2× 词频 1（亚线性，k1 饱和）
  [check 07] PASS  BM25 idf 恒正: log(1+·) 形式使任何 df 都有正权重（Lucene 口径）
  BM25 单路: recall@1 = 0.750 / recall@3 = 0.750 / MRR = 0.7500
  纯同义词查询得分: 'automobile fuel economy'→d14 = 0.0000，'bonuses for correct behavior'→d10 = 0.0000（词面零重叠 = 恒 0，稀疏侧盲点）
  [check 08] PASS  BM25 盲点机器证明: 两条纯同义词查询对 gold 得分恰为 0
  [check 09] PASS  BM25 单路 recall@3 < 1.0（两条同义词查询注定 MISS——L0 鸿沟在稀疏侧原样存在）
  [check 10] PASS  BM25 词面强信号: 'i was charged two times' 的 gold(d02) rank 1（idf 加权 'charged'）

[3] 混合检索（dense=real 路径 + BM25 稀疏，14 篇语料 × 8 条标注查询）
  方法                       recall@1  recall@3  MRR
  dense only               0.875     1.000     0.9375
  BM25 only                0.750     0.750     0.7500
  minmax+weighted .3/.7    0.875     1.000     0.9375
  RRF k=60                 0.750     1.000     0.8750
  [check 11] PASS  dense only 复刻 L1: recall@1 0.875 / recall@3 1.0 / MRR 0.9375
  [check 12] PASS  混合 RRF（实测派生）: recall@1 == 0.75 且 recall@3 == 1.0
  [check 13] PASS  混合 ≥ 两路短板: RRF recall@3 ≥ BM25 recall@3（融合不丢稀疏侧能力）
  [check 14] PASS  'automobile fuel economy' 混合后 rank == 2（BM25 得 0 分，dense 侧救回——融合的并集能力）
  反例探针（全列含零分档）: RRF recall@1 = 0.750 / recall@3 = 0.875（vs score>0 语义 0.750 / 1.000）；'automobile' gold rank 2 → 8
  尺度探针（BM25 分数 ×1000）: 裸加权和(无归一) top-1 == BM25 top-1（7 条非零命中查询）: True（量纲侧赢者通吃）；minmax 归一 top-3 位序不变: True；RRF top-10 位序逐位不变: True
  [check 15] PASS  尺度探针: 裸加权和（无归一）退化为大量纲侧（top-1 全同 BM25，7 条非零命中查询）
  [check 16] PASS  尺度探针: RRF 对分数正缩放不变（只看名次，top-10 逐位一致）

[4] HNSW（从零实现，对照 hnswlib）：合成语料上的 recall/成本权衡
  [check 17] PASS  N=200: layer-0 度数 ≤ 2M=16 且上层 ≤ M=8（hnswlib L113 maxM0 口径）
  N=  200  ef=  4: recall@10 = 0.994   距离计算 =   79.7/query（brute = 200/query）
  N=  200  ef= 16: recall@10 = 0.994   距离计算 =  105.8/query（brute = 200/query）
  N=  200  ef= 64: recall@10 = 1.000   距离计算 =  201.3/query（brute = 200/query）
  [check 18] PASS  N=800: layer-0 度数 ≤ 2M=16 且上层 ≤ M=8（hnswlib L113 maxM0 口径）
  [check 19] PASS  N=800: 层级 ≥1 占比 0.115 ≈ 1/M = 0.125（指数衰减，±0.05 容差）
  N=  800  ef=  4: recall@10 = 1.000   距离计算 =   98.6/query（brute = 800/query）
  N=  800  ef= 16: recall@10 = 1.000   距离计算 =  107.1/query（brute = 800/query）
  N=  800  ef= 64: recall@10 = 1.000   距离计算 =  233.6/query（brute = 800/query）
  [check 20] PASS  N=1600: layer-0 度数 ≤ 2M=16 且上层 ≤ M=8（hnswlib L113 maxM0 口径）
  N= 1600  ef=  4: recall@10 = 0.997   距离计算 =  114.4/query（brute = 1600/query）
  N= 1600  ef= 16: recall@10 = 1.000   距离计算 =  128.3/query（brute = 1600/query）
  N= 1600  ef= 64: recall@10 = 1.000   距离计算 =  175.5/query（brute = 1600/query）
  [check 21] PASS  HNSW 省距离: N=1600/ef=16 的每查询距离计算 < brute-force 的 1600
  [check 22] PASS  recall 随 ef 单调不降: N=1600 时 ef 4→16→64（ef = 召回宽度旋钮）
  [check 23] PASS  高 ef 高召回: N=1600/ef=64 recall@10 ≥ 0.95（recall/成本权衡的可控端）
  邻居选择消融（N=800, ef=16，同层级同插入序）: heuristic2 = 1.000 vs naive top-M = 0.887
  [check 24] PASS  heuristic2 多样性选择 recall ≥ naive top-M（hnswlib L443 的价值，论文摘要同款声称）

[5] 分片 scatter-gather（N=1600 → 4 片）：top-k 合并的精确性与 ANN 代价
  分片 brute 合并 == 全局 brute top-10: True（32 查询逐位，含分数舍入）
  分片 HNSW recall@10 = 1.000 vs 全局 HNSW = 1.000（ef=16）
  [check 25] PASS  不变量: 每片返回各自 top-k → 归并 == 全局 top-k（精确，无 ANN 近似——可证明）
  [check 26] PASS  分片 HNSW recall 与全局 HNSW 同量级（差距 ≤ 0.05；ANN 误差不随分片数爆炸）

[6] 两阶段重排（N=800：HNSW ef=4 粗召回 top-8 → OverlapReranker 精排 top-3，Milvus refiner 同构）
  第一阶段 top-3 命中 gold: 0.750 → 重排后 top-3: 0.969（天花板 = 候选召回 0.969，其中 7 条被重排从候选 4-8 位提进 top-3）
  reranker 调用 = 256 次（32 查询 × 8 候选）——不是 32×800 全量精排
  [check 27] PASS  两阶段: 重排后 gold 命中率 ≥ 第一阶段（精排只升不降——rerank top-3 ⊆ 候选集，gold ∈ 候选 ⇒ 可提位）
  [check 28] PASS  两阶段（实测派生）: 第一阶段 0.750 → 重排后 0.969 == 天花板 0.969（粗召回定天花板，精排定位次；粗召回漏掉的 gold 精排救不回）
  [check 29] PASS  成本边界: reranker 总调用 == 32×8 = 256（两阶段 = 便宜宽召回 + 昂贵窄精排）

[7] nDCG@k 分级评估（grade 2 = 正解 / grade 1 = 相关；recall 看不见的排名质量）
  'checkpointing after a crash': nDCG@3 BM25 = 1.000 / lexical = 0.964 / RRF 混合 = 1.000（gold2=d09，BM25 top3=['d09', 'd04', 'd03']，RRF top3=['d09', 'd04', 'd05']）
  'vector index similarity search': nDCG@3 BM25 = 0.826 / lexical = 0.826 / RRF 混合 = 0.826（gold2=d03，BM25 top3=['d03']，RRF top3=['d03', 'd06', 'd14']）
  [check 30] PASS  nDCG 有区分度: 两条查询的 RRF nDCG@3 不全相等（排名质量可测）
  [check 31] PASS  nDCG 归一: 完美排序 nDCG@3 == 1.0（构造判据自校验）

[8] 成本账本（toy 口径；对照 Milvus Index Explained 的 1M×128d HNSW 内存账）
  N=800: 向量段 = 800×128×4 = 409600 B；图 = 13832 条边×4 = 55328 B（平均每点 17.3 条边）
  Milvus 口径参照: 1M×128d HNSW = 图 128 MB + 向量 512 MB = 640 MB（度数 32 口径，文档算例）
  [check 32] PASS  图字节公式: 边数×4 == 图字节
  [check 33] PASS  图非空且稀疏: 0 < 平均每点边数 ≤ 2M+上层（图是稀疏结构，不是全连接）

retrieval digest (path-independent core): 7823a73786681b38  两次独立构建逐位一致: True
  [check 34] PASS  确定性: digest 两次独立构建逐位一致（md5 层级 + 确定插入序 + 无 RNG 状态）

self-check: 34/34 PASS
```

fallback 路径（空 HOME 探针）结构相同，差异只在 [0] 路径声明、[1] L1 digest 期望值（`008632d5402a86b7`）与 [3] dense 侧数字（开卷 mock，recall@1 = 1.0——**不与 real 比大小**，口径承 L1 §10(b)）；掩码锚 `82bac3cae504a3afe0ff5c9aaf28e63e`/87 行/7,724 B，digest 与 real 路径逐位同一（`7823a73786681b38`——digest 只覆盖纯 Python 确定性部分，模型相关数字由 L1 digest 锚与各路径 EXP 表分别钉住）。

---

## §3 机制面 [1]：跨级锚——为什么 L1 的函数要逐字搬运

[1] 节把 L1 的 `embed_lexical` / `embed_fallback` / `load_real_embedder` / `FlatL1` / `evaluate_l1` **逐字搬进 L2**（连注释都不改），然后在同一 fixture 上复现 L0/L1 的基线：lexical `0.625 / 0.750 / 0.6875`、MISS `[d10, d14]`、L1 digest `4c8c4347e418b602`（real）。

这不是复制粘贴的惰性，而是**同集对照的方法论前提**：L2 要声称「混合检索比单路强」「HNSW 比 flat 省」，这些声称的参照系必须与 L1 是**同一个函数、同一份语料、同一套评估集**——否则任何数字差异都可以在「实现变了 / 数据变了 / 评估变了」三个解释之间漂移，实验就失去了证伪能力。digest（per_q 全表 + mrr 的 sha256 前 16 位）把「同一」钉成机器可验的锚：L2 跑出的 digest 与 tutorial_L1 §10 录值逐位吻合 = L1 的全部检索行为在 L2 进程内逐字复现。

> 思考：如果 L2 顺手把 `tokenize` 「优化」了一下（比如加词干化），跨级锚会发生什么？——digest 必变，L0/L1 的所有录值失去对照资格。这就是「锚比方便重要」的代价意识。

---

## §4 机制面 [2]：BM25——Lucene 公式、手算锚与稀疏侧的盲点

### 4.1 公式与 Lucene 行锚

BM25 是稀疏侧的打分器（Okapi BM25 的 Lucene 实现口径）。对查询 q 与文档 d：

```
score(q, d) = Σ_{t∈q} idf(t) · tf(t,d) / (tf(t,d) + k1·(1 − b + b·dl/avgdl))
idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))        默认 k1=1.2, b=0.75
```

行锚（抓取件 `lucene_BM25Similarity.java`，2026-08-16 抓取）：默认参数在 **L109/L122** 双构造器 `this(1.2f, 0.75f, ...)`；idf 公式 **L139-140**（L183 解释串同式）；tf 项的归一分母在 **L219** 做成 norm 缓存（`1f / (k1 * ((1 - b) + b * LENGTH_TABLE[i] / avgdl))`，每文档一次）；**L257-267** 的 doScore 重写注释逐字说明它与经典 Okapi 形式 `tf·(k1+1)/(tf+norm)` **只差每词常数 (k1+1)，不影响排序**——代数恒等，不是近似。nano 版按 Lucene 口径实现（无 (k1+1) 因子），每查询重算 norm（14 篇语料不需要缓存，这是 toy 与生产的取舍，§12）。

### 4.2 手算锚：不信任实现的独立闭式

checks 04–05 用 N=2 的微型语料 `[("A","x y"), ("B","x x z w")]` 做闭式重算：df(x)=2 → `idf = ln(1 + 0.5/2.5) = ln(1.2)`；avgdl=3；A（tf=1, dl=2）分母 `1 + 1.2·(0.25+0.75·2/3) = 1.9`；B（tf=2, dl=4）分母 `2 + 1.2·(0.25+0.75·4/3) = 3.5`。实现值与闭式值 1e-12 内吻合。手算锚的意义：**公式抄错、norm 写反、avgdl 算错，都会在这里当场现形**——它独立于语料与评估集。

### 4.3 三个机制性质 + 一个机器证明的盲点

- **tf 饱和**（check 06）：`sB < 2·sA`——词频 2 的得分不到词频 1 的两倍。k1 是饱和旋钮：tf 越大边际贡献越小，防止长文档靠词频堆赢。
- **idf 恒正**（check 07）：`log(1+·)` 形式使 df=N 的极常见词也有正权重（经典 Okapi 的 idf 在 df>N/2 时为负——Lucene 的这个改动避免了「常见词反向扣分」的怪异行为）。
- **词面强信号**（check 10）：'i was charged two times' 的 gold d02 rank 1——'charged' 的高 idf 加权直接锁定。这正是 L0 §5(a) 记录、L1 债表②要回收的稀疏侧优势。
- **盲点机器证明**（checks 08–09）：两条纯同义词查询（'automobile fuel economy'→d14、'bonuses for correct behavior'→d10）对 gold 得分**恰为 0**——词面零重叠时 idf 加权毫无作用。L0 的同义词鸿沟在稀疏侧原样存在：**BM25 是词表上的对角算子，看不见语义**。这是混合检索存在理由的一半（另一半 = dense 也有自己的盲点，见 §5.4 反例与 §17）。

---

## §5 机制面 [3]：融合术——min-max 加权与 RRF，以及「RRF 不是免费午餐」

### 5.1 融合的前提：只融合各路「返回的结果列表」

真实引擎的混合检索有一个容易被忽略的前提：**每一路只返回它检索到的结果列表，融合器只对入列的文档计分**。Elastic RRF 文档给出的公式以 `if d in result(q)` 为前提（抓取件 `elastic_rrf.txt` 在盘）；OpenSearch normalization-processor 的 min_max 归一也以「该路返回的分数集合」为归一基线。

nano 版把这个前提实现为 `BM25.rank()` 的 **score>0 过滤**：零分文档不入列。为什么这是机制而非实现细节，§5.4 的反例探针给出机器证明。

### 5.2 两种工业融合术

**min-max 归一 + 加权算术平均**（OpenSearch 口径，抓取件 `opensearch_hybrid-search` 示例 `technique=min_max + arithmetic_mean + weights` 在盘）：每路先做 `(s−lo)/(hi−lo)` 归一到 [0,1]，再加权平均（本跑 .3/.7）。它的前提是**分数可比**——归一只能抹平线性尺度差，抹不平分布形状差。某路对该查询空列（纯同义词查询的 BM25 侧）则贡献 0：没有归一基线可言。

**RRF**（Cormack et al. 2009）：`RRFscore(d) = Σ_r 1/(k + r(d))`，k=60。论文原文「where k = 60 was fixed during a pilot investigation」（抓取件 `cormack2009_rrf.pdf` 逐字在盘）——k 在试点调查中钉死后就没再调过。Elastic rrf retriever 的 `rank_constant` 默认 60 且文档明言「RRF requires no tuning」（抓取件逐字在盘）。**「requires no tuning」的机制根源 = 只看名次、不看分数**：分数尺度、分布形状统统无关，两路质量参差时也不需要调权重。

各家 API 形态差异（反幻觉声明）：当前
[PyMilvus ORM `hybrid_search`](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/hybrid_search.md)
签名要求显式传入 `rerank: BaseRanker`，可选 `WeightedRanker` 或 `RRFRanker`；因此不能把某个
示例选择 RRFRanker 改写成“系统默认 reranker=RRF”。这里描述的是该版本 API，不外推到其他 SDK。

### 5.3 实测方法表（real 路径）与诚实声明

| 方法 | recall@1 | recall@3 | MRR |
|------|----------|----------|-----|
| dense only | 0.875 | 1.000 | 0.9375 |
| BM25 only | 0.750 | 0.750 | 0.7500 |
| minmax+weighted .3/.7 | 0.875 | 1.000 | 0.9375 |
| RRF k=60 | **0.750** | 1.000 | 0.8750 |

**诚实声明（融合使强路径退化的事实）**：RRF 的 recall@1 = 0.750 **低于** dense only 的 0.875——名次融合丢掉了 dense 分数的幅度信息，在这个 toy 评估集上把 recall@1 拉低了。融合不是免费午餐，它买的是**并集能力与免调参**（check 14：'automobile fuel economy' BM25 得 0 分、dense 侧救回 rank 2——单路 BM25 在此查询上 rank 0，单路 dense 也非处处 rank 1），以及 recall@3 守住 1.0（check 13：≥ 两路短板）。何时该用加权融合、何时该用 RRF、何时干脆只用强路径——这是评估集说了算的工程决策，不是教条。

### 5.4 反例探针：零分全列注入 = RRF 稀释的机器证明

如果 `BM25.rank()` 不过滤零分、返回**全语料**（零分档 tie-break by doc_id），会发生什么？代码里的反例探针实测：RRF recall@3 从 1.000 跌到 **0.875**，'automobile' 的 gold rank 从 2 掉到 **8**。

机制：零分档 12 篇文档按 doc_id 升序入列 = 向融合器注入一组**伪名次信号**（它们的名次只由 doc_id 决定，与查询毫无关系）。纯同义词查询的 gold（d10/d14，大 doc_id）被钉在列尾，RRF 项 1/(60+74) vs 1/(60+61) 的差值不足以翻盘 dense 侧的优势——**零分全列稀释了强路径**。这就是「只融合各路返回的结果列表」这个前提的分量：它不是 API 文档的脚注，是融合正确性的一部分。

### 5.5 尺度探针：三种融合术对「BM25 分数 ×1000」的反应

故意把 BM25 分数放大 1000 倍（check 15–16 + print 录值）：

- **裸加权和（无归一，`raw_weighted_fuse`）**：7 条非零命中查询的 top-1 **全部**变成 BM25 的 top-1——大量纲侧赢者通吃。这是生产不用裸加权和的原因：BM25 无上界 vs cosine ∈ [−1,1]，尺度本就不可比。
- **minmax 归一**：本跑 top-3 位序不变（print 录值 True）。注意口径：min-max 对**单路列表的正缩放**在数学上不变（缩放被 (s−lo)/(hi−lo) 吸收），所以本 fixture 上位序保持；但这**不是稳健的工程保证**——浮点末位差异在 sort key `(−score, doc_id)` 上可以翻转近 tie 文档的次序（另一个近 tie 构造已实测出现翻转）。因此代码只把 **RRF 的位序不变性**做成 check（check 16），minmax 不变性只录值、不声称。
- **RRF**：top-10 逐位不变（check 16）——只看名次的直接推论，也是它「requires no tuning」的另一面。

---

## §6 机制面 [4]：HNSW 从零实现——层级、束搜索与 heuristic2 消融

### 6.1 结构：指数衰减层级 + 双层搜索

HNSW（Malkov & Yashunin, arXiv:[1603.09320]）的核心是**多层跳表式图**：每个点按指数衰减概率分到层级 `level = floor(−ln(u)·mL)`，`mL = 1/ln(M)`（hnswlib `hnswalg.h` **L142** `mult_ = 1 / log(1.0 * M_)`；**L207-209** getRandomLevel `-log(distribution(...)) * reverse_size`；**L1186** 调用点）。实测 N=800 时层级 ≥1 占比 **0.115 ≈ 1/M = 0.125**（check 19，±0.05 容差）——指数衰减的直接验证。nano 版用 md5 从点 id 派生 u（无 RNG 状态，确定性可复现——hnswlib 用随机数，结构同构、随机源不同）。

搜索分两段（hnswlib **L1277-1302** 上层贪心下降 + **L1307-1308** 底层 `searchBaseLayerST<true>(…, std::max(ef_, k), …)`）：上层只持一个入口点做贪心下降（快、粗），到底层后展开 **ef 束搜索**（beam：同时维护 ef 个候选）。底层停止条件（**L353** 逐字对应）：**最近候选比最远结果还远、且结果已满 ef**——再扩也不会更好。度数上限 layer-0 = 2M、上层 = M（**L113** `maxM0_ = M_ * 2`；checks 17/18/20 三个 N 全验）；另有 `ef_construction ≥ M`（**L114**）与默认 `ef_ = 10`（**L115**）两条 hnswlib 口径，录值备查。

### 6.2 recall / 距离计算量权衡曲面

brute-force 是 recall 裁判（承 L0 §4 Milvus FLAT 判据：ANN 的 recall 定义 = 与暴力 top-k 的重合率）。合成世界（16 主题 × 词面信号，md5 驱动）上扫 N ∈ {200, 800, 1600} × ef ∈ {4, 16, 64}：

- **省距离**（check 21）：N=1600/ef=16 每查询 **128.3** 次距离计算 vs brute 1600——图搜索的本质收益。距离计数器 `dist_comps` 同构 hnswlib `metric_distance_computations`（**L1286-1287**）。
- **ef 是召回宽度旋钮**（check 22）：N=1600 时 recall@10 随 ef 4→16→64 单调不降（0.997→1.000→1.000）。
- **高 ef 高召回**（check 23）：N=1600/ef=64 ≥ 0.95。
- 注意 N=200/ef=64 距离计算 201.3 > 200：**小 N 下 HNSW 不比 brute 省**——图搜索有常数开销，ANN 的收益随 N 增大才显现。这是「何时该上 ANN」的第一性答案。

### 6.3 heuristic2 消融：同一层级、同一插入序，只换邻居选择规则

`getNeighborsByHeuristic2`（hnswlib **L443-480**）的多样性规则：**已选邻居若比「候选到查询」更靠近候选，则候选落选**——避免邻居们挤在同一个方向（角度多样性）。nano 版消融（checks 24）：同一层级序列、同一插入序、N=800/ef=16，只切 `heuristic` 开关——**heuristic2 = 1.000 vs naive top-M = 0.887**。0.113 的 recall 差就是这 30 行代码的价值，与论文摘要的同款声称互为印证（论文给的是大数据集统计，这里是单合成世界的机器证明——口径声明见 §13）。

---

## §7 机制面 [5]：分片 scatter-gather——一个可证明的精确不变量

N=1600 切 4 片。**不变量（check 25，32 查询逐位 True，含分数舍入）**：每片 brute 返回各自 top-10 → 归并取全局 top-10 == 全局 brute top-10。这是**可证明的**：任何进入全局 top-10 的点必然属于某一片，且在该片内不差于全局第 10 名（否则全局第 10 名之外的点不可能挤掉它），故必入该片的 top-10——归并不丢点。tie-break 两侧同按 doc_id，打平时也严格成立。

**ANN 侧没有这个保证**：每片的图连通性弱于全局图（入口点少、邻居池小），分片 HNSW 的 recall 可能低于全局 HNSW。实测（check 26）：本跑 1.000 vs 1.000（ef=16，差距 ≤ 0.05 断言）——合成世界的几何友好，但**不变量只在 brute 侧成立**这一事实不变。分片的工程收益（并行构建 / 内存分摊 / 增量加片）要以「ANN 误差可能随分片数累积」为代价来换——这是 Milvus segment、Weaviate shard 背后同一笔账。

---

## §8 机制面 [6]：两阶段重排——粗召回定天花板，精排定位次

### 8.1 与 Milvus refiner 同构

Milvus 的两阶段检索（抓取件 `milvus_index-explained.md`：**L100-104** Refiner 节 + **L288-291**「10 (topK) x 5 (expansion rate) = 50 candidates」逐字在盘——refiner 对照的真实出处是 index-explained 页，不是 reranking 页 [后者抓取 404，见 §19]）：先用索引取 topK×expansion rate 个候选，再用更高精度重算、返回最终 topK。nano 版同构：HNSW ef=4 粗召回 top-8（ef<k 被抬到 max(ef,k)=8，hnswlib **L1307-1308** 口径）→ `OverlapReranker` 精排 top-3。

**显式 mock 声明**：`OverlapReranker` = idf 加权词重叠打分器，是 cross-encoder 的**机制 mock**——真实 cross-encoder 是 BERT 类模型对 (q, d) 联合编码出相关性分，机制同类（成对细粒度交互，比双塔独立编码贵但更准），代价与精度都高一个量级 `[TODO: verify on real system]`。本节机器证明的是**两阶段的机制不变量**，不是 reranker 的语义正确性。

### 8.2 实测：0.750 → 0.969，天花板 = 候选兜住率

第一阶段 top-3 命中 gold **0.750** → 重排后 top-3 **0.969** == 候选兜住率 **0.969**（32 查询中 31 条的 gold 在 top-8 候选里），其中 **7** 条被重排从候选 4–8 位提进 top-3；reranker 调用恰 **256 = 32×8** 次（check 29：不是 32×800 全量精排——两阶段 = 便宜宽召回 + 昂贵窄精排的成本结构）。

两条教学声称都从实测反推（check 27–28）：

- **精排只升不降**（机制不变量）：rerank top-3 ⊆ 候选集，gold ∈ 候选 ⇒ 重排后命中率 ≥ 第一阶段。这是集合包含关系，不依赖 reranker 质量——但**等号何时成立**依赖 reranker 与 gold 判据的同源性（本探针的 gold = 最大词面重叠并列全收，reranker = idf 加权重叠，判据同源，故重排后 == 天花板）。
- **粗召回定天花板**：1/32 的 miss 是 gold 根本没进候选（recall 侧边界）——精排救不回粗召回漏掉的东西。生产里调 `expansion rate` / `ef` 就是在调这个天花板，调 reranker 只是在天花板下定位次。

### 8.3 gold 口径教训（为什么「并列全收」）

本段第一版用「单 gold + 任意 tie-break」口径，实测第一阶段 0.375 → 重排后 0.000——原声称被自己的探针证伪（§11 第四坑）。根因：合成世界里多条文档与查询的词面重叠**并列最大**，任意选一条当唯一 gold，tie-break 噪声就淹没了机制信号。改为「最大词面重叠**并列全收**」后，gold 是「最佳答案集合」，探针才量出机制不变量。**小评估集上 gold 的构造方式本身就是实验设计**——这是从失败实验里长出来的方法论。

---

## §9 机制面 [7]：nDCG@k——recall 看不见的排名质量

recall 只问「找没找到」，nDCG 问「排得好不好」。分级相关性（grade 2 = 正解 / grade 1 = 相关但非正解）+ 位置折扣：

```
DCG@k = Σ_i (2^g_i − 1) / log2(i + 2)        nDCG@k = DCG@k / IDCG@k（理想排序归一）
```

实测（[7] 节，路径无关的 lexical dense + BM25 口径）：'checkpointing after a crash' 三方法 nDCG@3 = 1.000 / 0.964 / 1.000——lexical 把 grade-1 的 d04 排到了 d09 之前？不，是 d09 仍 rank 1 但 grade-1 文档的相对位置吃了折扣（0.964）；'vector index similarity search' 三方法全 0.826（BM25 侧只有 ['d03'] 一条入列——短列表的 nDCG 形态）。checks 30–31：两条查询的 nDCG 不全相等（**有区分度**）+ 完美排序 nDCG == 1.0（**归一自校验**——构造判据的闭合检查）。

---

## §10 机制面 [8]：成本账本——图字节与 Milvus 1M×128d 参照

N=800（D_SYN=128，float32）：向量段 800×128×4 = **409,600 B**；图 = **13,832** 条边×4 = **55,328 B**（平均每点 **17.3** 条边，check 33：0 < 平均度数 ≤ 2M+上层——图是稀疏结构，不是全连接）。对照 Milvus Index Explained 的文档算例（抓取件 **L297-315** 逐字在盘）：1M×128d、度数 32 口径 HNSW = **128 MB（图）+ 512 MB（向量）= 640 MB**——向量段是大头，图约 1/4。toy 数字不可外推（§13），但**账本结构**（向量段 + 图段分开记、度数决定图段）与生产口径同构。

---

## §11 反例教材：第一版代码的六个坑——「期望值表必须实测派生」

本节保留一次失败版本的教训：它的 28 个绿色 check 全是真的，同时有 6 处声称从未被对应探针覆盖——**输出全绿 ≠ 机制正确**。六个坑逐一教材化：

**坑 1（EXP 表先验声明）**：第一版期望值表先验写了 hybrid recall 1.0/1.0，实测 real 路径 RRF = 0.750/0.875（当时还是全列语义）、fallback = 0.750/0.750——双路径全灭。机制诊断：当时 `BM25.rank()` 返回全语料（含零分档），零分档按 doc_id 序注入伪信号（§5.4 反例探针即此机制的机器证明）。**教训：期望值表必须是实跑产物，不是设计意图的誊写**——修复后的 EXP 表每个数字都标注「实测派生」，且双路径分开声明。

**坑 2（fail-loud 走错通道）**：第一版用 `ranks.index("d14") + 1` 断言 rank，gold 不在 top-3 时抛**失控 ValueError**——fail-loud 应该是**受控的 check FAIL**（带断言名、进 self-check 账本），不是让解释器 traceback 在半路。修复：gold 不在列 → r=0 → check 受控 FAIL。

**坑 3（探针自相矛盾）**：第一版「未归一加权」尺度探针实际调用的是 `minmax_weighted_fuse`——恒做归一，声称的「量纲退化」机制上不可能出现，print 文案与实测直接矛盾。修复：另写 `raw_weighted_fuse`（真正无归一的裸加权和，生产不用、探针专用），check 15 才量到真实现象（赢者通吃）。**教训：探针的每一个字都要与它的实现匹配——声称的量纲行为必须发生在确实没有归一的代码路径上。**

**坑 4（元组位置）**：第一版 `qtext_of = {qid: qt for qid, qt, _ in qvecs8}` 取出的第二元素是 **embedding 向量**（`qvecs8` 存的是 (qid, vec, topic)）——`tokenize(list)` 直接 AttributeError，整个 [6] 段从未执行过。同族：`(score, did)` 元组取错位会得到恒 0 的假数字。**教训：解包位置是契约，探针要先跑通再谈结论。**

**坑 5（设计被实测证伪）**：第一版重排段按「单 gold」口径声称「精排只升不降 / ≥0.9」，修复坑 4 后实测 0.375 → 0.000——双声称皆假。重构 = §8.3 的 gold 口径（并列全收）+ 声称从实跑反推（0.750 → 0.969 == 天花板）。**教训：教学声称必须从实跑反推，不能先写结论再凑实验。**

**坑 6（返回值元数）**：第一版 `build_digest` 对 `synth_world` 的 3 元返回值做 5 元解包 → ValueError 必崩——digest 确定性 check 从未执行过。**教训：digest 段是确定性的最后一道闸，它自己必须先能跑。**

**补充坑（无溯源声称）**：第一版 docstring 声称「Milvus hybrid search 默认 reranker=rrf」，却没有可靠来源。§5.2 当前版本 API 显示 rerank 是**必选参数、无默认值**——「默认」之说被证伪。**教训：API 行为声称必须链接到固定版本文档或源码，不能凭印象。**

元教训（与 nano-data-orchestration L2 的 zombie 反例教材同族）：**每一个坑的共同签名都是「声明先于测量」**。senior 与半成品的分界线就在这里——senior 的每一个数字都是跑出来的。

---

## §12 权威实现取舍表：nano 版 vs Lucene / hnswlib / OpenSearch / Elastic / Milvus

| 机制面 | nano L2 做法 | 权威实现（行锚，抓取日 2026-08-16 除注明外） | 为什么 nano 不那样做 |
|--------|--------------|----------------------------------------------|----------------------|
| BM25 | 每查询 O(N) 重算 score，无倒排 | Lucene `BM25Similarity.java`：倒排 + norm 缓存（L219）+ doScore 单调重写（L257-267）+ 默认 k1/b（L109/L122） | 14 篇语料，倒排的构建/维护成本 > 收益；公式与默认参数逐字对齐 |
| 融合 | 进程内 dict，score>0 过滤 | OpenSearch normalization-processor（min_max+arithmetic_mean+weights）；Elastic rrf retriever（rank_constant 默认 60，requires no tuning）；pymilvus hybrid_search（rerank 必选参数，master orm/collection.py L896-899，**2026-08-18 live 抓取**） | 无网络/分片/一致性面；融合语义（只融合入列结果）与真实引擎同口径 |
| HNSW | 纯 Python，md5 派生层级，单线程构建 | hnswlib `hnswalg.h`：mL=1/log(M)（L142）/ getRandomLevel（L207-209, L1186）/ maxM0=2M（L113）/ ef_construction≥M（L114）/ 默认 ef=10（L115）/ 停止条件（L353）/ heuristic2（L443-480）/ 双向连接修剪（L603/L1052）/ 上层下降（L1277-1302）/ max(ef,k)（L1307-1308）/ 距离计数（L1286-1287） | 无 SIMD / 并行构建 / mmap 序列化——结构不变量（层级分布 / 度数上限 / 停止条件 / 消融）逐条对照，性能面不冒充 |
| 重排 | OverlapReranker（idf 加权重叠，显式 mock） | Milvus refiner（topK×expansion rate，index-explained L100-104 / L288-291）；真实 cross-encoder（BERT 联合编码） | mock 声明在 docstring；机制同构（候选集上重算），语义精度不冒充 [TODO: verify on real system] |
| 分片 | 进程内 4 片，strided 切分 | Milvus segment / Weaviate shard（分布式 scatter-gather） | 无网络/尾延迟/部分失败面；精确不变量（brute 侧）与 ANN 代价的账本结构同构 |
| 评估 | recall/MRR/nDCG，8+2 条标注 | TREC 系评估（大标注集 + 统计显著性） | toy 规模；nDCG 公式与归一自校验同口径 |
| 持久化 | L2 图不落盘（承 L1 思路，未实现） | Milvus segment 落盘 / hnswlib 序列化 | 声明为边界（§13）：图字节 = 序列化对象，身份契约思路承 L1 §6 |

---

## §13 toy vs 生产：差距的诚实声明

- **HNSW 性能**：纯 Python 逐点距离 vs hnswlib C++（SIMD / 并行构建 / 预取）——本教程的距离**计算次数**是结构量（可对照），**墙钟时间**不可外推。
- **合成世界**：16 主题 × 词面信号的几何，recall 数字（0.994–1.000）只对该世界有效；真实语料的聚类结构、维度灾难、分布偏斜都不在 toy 射程内。
- **reranker 是 mock**：idf 加权词重叠与真实 cross-encoder 的语义精度差一个量级以上；两阶段**成本结构**（256 vs 25,600 次调用）与**机制不变量**（天花板/定位次）是真的，语义质量声明是假的（已在 docstring 显式注明）。`[TODO: verify on real system]`
- **真实引擎混合检索**：真 Milvus hybrid_search / OpenSearch hybrid query 的网络往返、segment 合并、一致性语义未触及。`[TODO: verify on real system]`
- **分片**：进程内切分，无网络延迟 / 尾延迟 / 部分失败 / 再平衡。
- **评估集**：8 条标注查询 + 2 条分级查询，toy 基线（承 L0 §6 声明）；生产评估需要量级与统计检验。
- **图持久化未实现**：L2 的 HNSW 图是内存结构；序列化 / 重载 / 身份契约是 L1 思路的自然延伸但本级不做（边界，非遗漏）。
- **确定性契约**：md5 驱动层级与合成语料、无 RNG 状态、无 wall-clock 进 digest——双路径 × 双 CWD 掩码输出 BYTE-IDENTICAL 是它的机器证明；elapsed 行掩码（口径承 L0 家族）。

---

## §14 HNSW / RRF / BM25 的时效性定位

- **HNSW**（arXiv:[1603.09320]，2016）= **A 层经典机制**：当今主流向量库的第一等索引（Milvus / OpenSearch / Weaviate / pgvector 均原生支持，抓取件 2026-08-16 在盘）；图索引 + 多层跳表结构至今仍是 ANN 检索的主力形态，无时效问题。前沿演进（量化 / 磁盘驻留 / 流式更新）是在这个骨架上做文章，本模块止于 L2 不展开。
- **RRF**（Cormack et al. 2009）= **A 层经典机制**：Elastic / OpenSearch / Milvus 内建融合器，「requires no tuning」在 2026 年的文档里仍是核心卖点（抓取件在盘）。学习型融合（learning-to-rank）是另一条路线，不在本模块射程。
- **BM25**（Robertson 等 1994 谱系，Lucene 实现为事实标准）= 稀疏侧检索的 A 层经典；learned sparse（SPLADE 族）是近年新路线——本模块只对照 Lucene 经典口径（抓取件行锚在盘），不展开。
- **两阶段（检索+重排）** = 当今 RAG / 搜索管线的主流标准配置（Milvus refiner / 各厂 reranker API 均此形态，抓取件在盘）。

---

## §15 费曼自检

**讲给外行听的版本**：你去图书馆找资料。HNSW 是「问楼层引导员」——引导员（上层节点）先把你指到大致楼层（贪心下降），再到书架间逐排细找（ef 束搜索），不用走遍每个书架；混合检索是「同时问两个馆员」——一个按关键词卡片查（BM25），一个按内容相似度查（dense），然后把两份书单合并（融合）；两阶段重排是「先抱回一摞候选（top-8），再在桌上逐本精读挑三本（top-3）」——抱书便宜、精读贵；nDCG 是「不只问找没找到，还问推荐顺序好不好」。

自检问：

1. 为什么「同时问两个馆员」不一定比「只问最懂的那个馆员」好？（§5.3：RRF recall@1 0.750 < dense 0.875——名次融合丢幅度信息。）
2. 引导员为什么不用把每层楼都走一遍？（停止条件：最近候选比最远结果还远且结果已满——§6.1。）
3. 「抱回一摞再精读」为什么救不了「根本没抱回来的那本」？（粗召回定天花板：重排 ⊆ 候选——§8.2。）
4. 如果两个馆员一个用「米」报距离、一个用「英里」报距离，直接加权平均会怎样？（§5.5：裸加权和赢者通吃——归一或 RRF 是回答。）

---

## §16 思考题（×5）

1. RRF 只看名次、丢掉分数幅度——在什么场景下这是**优点**（两路分布不可比时），什么场景下是**损失**（一路远强于另一路时无法加权偏向）？本跑 recall@1 退化 0.875→0.750 是哪种？
2. 零分全列注入为什么对 RRF 的伤害比「零分文档不入列」看起来大得多？从 1/(60+r) 的函数形状（r 小时导数大）解释为什么列尾伪名次能翻盘 dense 优势。
3. 分片 brute 合并的精确不变量为什么对 ANN 不成立？如果每片 HNSW 的 ef 翻倍，分片 recall 与全局 recall 的差距会如何变化？设计一个实验验证。
4. gold「并列全收」消除了 tie-break 噪声，但引入了另一个问题：gold 集合可能很大（多条文档并列最大重叠）。这对 recall 的定义（命中 = 交集非空）意味着什么？换成「恰好一条」的严格口径，数字会怎样漂移？
5. L1 的身份契约（模型身份是索引的一部分）如果推广到 HNSW 图，契约里应该钉住哪些字段？（提示：M / ef_construction / 距离度量 / 层级分配源——图结构本身是「模型」。）

---

## §17 反例与边界

- **RRF 稀释反例**（§5.4）：零分全列注入 → recall@3 1.000→0.875、gold rank 2→8。边界声明：score>0 语义下 RRF 仍可能使强路径 recall@1 退化（本跑 0.750 < 0.875）——融合保的是并集与免调参，不是处处变强。
- **EXP 表教训**（§11 坑 1）：期望值表不能来自设计意图。现行 EXP 表每个 hybrid 值都是双路径实测派生，注释保留了证伪过程。
- **minmax 缩放不变性不是 check**（§5.5）：本跑 top-3 位序 True 是 run-specific 录值；浮点末位可翻近 tie 次序，另一个构造已实测 False。绿名单只含 RRF 位序不变（check 16）。
- **mock reranker 边界**：gold 判据与 reranker 判据同源（词面重叠）——本探针证明的是两阶段**机制不变量**（天花板/定位次/成本边界），不是 OverlapReranker 的语义正确性。
- **合成世界边界**：HNSW recall 曲面数字只对该世界有效；层级分布 0.115≈0.125 是分布律验证，与语料无关。
- **fallback 路径**：开卷覆盖评估集，recall 1.0 不与 real 比大小（承 L1 §10(b) 反幻觉纪律）。

---

## §18 阶梯预告：本模块止于 L2

nano-rag-retrieval 的阶梯到 L2 封顶。L1 §12 债表中未回收的三行——**量化压缩（int8/SQ/PQ）/ 更新语义（upsert → 图修复）/ 多租户 ACL**——是刻意保留的边界：量化在 [6] 以「粗召回+精排」同构触及机制、不做真实量化误差模型；更新与 ACL 与 [nano-data-platform](../nano-data-platform/) 的治理合流（语料 = 版本化快照，索引才可复现；索引重建 = [nano-data-orchestration](../nano-data-orchestration/) 的一条 DAG 任务）。检索栈的 SOTA 深挖需要在独立专题中继续验证。

---

## §19 溯源与口径声明

**一手来源边界**：源码行号会随上游变动，因此正文保留机制与稳定入口，不把本地抓取行号当作永久事实。

- [Lucene `BM25Similarity`](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java)：BM25 公式与默认参数的实现参照。
- [hnswlib `hnswalg.h`](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h)：层级、束搜索、邻居裁剪与度数上限的源码参照。
- [PyMilvus `hybrid_search`](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/hybrid_search.md)：显式 reranker 参数与两种 ranker。
- HNSW arXiv:[1603.09320]；RRF Cormack et al. 2009（SIGIR）；RAG arXiv:[2005.11401]。

**运行锚点**（2026-08-31，`-B`）：代码 md5 `3d0f75ffd50951c562820db492ba072d`/777 行/47,570 B；real 掩码 `79ae241041f5f69609ed17ac90544202`/86 行/7,651 B，fallback 掩码 `82bac3cae504a3afe0ff5c9aaf28e63e`/87 行/7,724 B。两条路径均在两个新建空 CWD 中 BYTE-IDENTICAL、EXIT=0、stderr 0 B、34/34 PASS；digest `7823a73786681b38`（双路径），跨级锚 real `4c8c4347e418b602` / fallback `008632d5402a86b7`。real 与 fallback 数字不混比。

**口径分离声明**：real / fallback 两路径 EXP 表分开（代码 L447-452），数字不混比；digest 只覆盖纯 Python 确定性核（BM25/RRF/HNSW/分片/重排/nDCG-lexical），模型相关数字由 L1 digest 锚 + 各路径 EXP 表分钉；toy 数字一律不外推生产（§13）。
