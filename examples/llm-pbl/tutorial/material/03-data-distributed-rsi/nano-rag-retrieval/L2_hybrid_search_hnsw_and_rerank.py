#!/usr/bin/env python3
"""nano-rag-retrieval L2 — 混合检索 + HNSW + 两阶段重排 + 检索评估升级（自 L1 的 K+1）。

L1 换真实神经 embedding 把 recall 推过同义词鸿沟，但索引仍是 flat 暴力扫描（O(N·D)/query），
且只有一路信号。L2 回答 L1 §12 债表里的四笔债（ANN 索引 / 混合检索 / 重排序 / 评估升级），
全部从零实现、全部由脚本运行得到数字：

  [1] 跨级锚：L1 的 lexical 基线与 digest 逐字复现（真实/ fallback 双路径口径承 L1 §10）；
  [2] BM25 稀疏打分器（Lucene BM25Similarity 公式：idf = ln(1+(N-n+0.5)/(n+0.5))，
      tf 饱和 k1=1.2，长度归一 b=0.75）——并机器证明它的盲点：纯同义词查询得分恒 0；
  [3] 混合检索：dense + BM25 两路融合。两种工业融合术——min-max 归一 + 加权算术平均
      （OpenSearch normalization-processor 口径）与 RRF（Cormack et al.，k=60；Elastic rrf
      retriever 默认 rank_constant=60；pymilvus hybrid_search 的 rerank 为必选参数、无默认，
      官方示例选 RRFRanker）；RRF 尺度不变性探针：加权融合以「分数可比」为前提，RRF 只看名次、
      无此前提；零分全列注入反例探针（真实引擎只融合各路返回的结果列表）；
  [4] HNSW 从零实现（对照 hnswlib/hnswalg.h）：指数衰减层级（-ln(u)·mL，mL=1/ln(M)）、
      上层贪心下降 + 底层 ef 束搜索、getNeighborsByHeuristic2 多样性邻居选择（消融实验
      机器证明其价值）、layer-0 度数上限 2M；在合成语料上实测 recall@10 vs 距离计算量的
      N×ef 权衡曲面（brute-force 是 recall 裁判，承 L0 §4 Milvus FLAT 判据）；
  [5] 分片 scatter-gather：top-k 分片合并的精确性不变量（机器证明）+ 分片 HNSW 的 recall 代价；
  [6] 两阶段重排：HNSW 粗召回 top-N → reranker 精排 top-k（Milvus refiner/expansion rate
      同构）；reranker = idf 加权词重叠打分器（显式 mock cross-encoder：机制 = 查询与文档
      成对细粒度交互，真实 cross-encoder = BERT 类模型对 (q, d) 联合编码 [TODO: verify on real system]）；
  [7] 评估升级：nDCG@k 分级相关性（recall 只问「找没找到」，nDCG 问「排得好不好」）；
  [8] 成本账本：HNSW 图字节 + 向量字节（对照 Milvus Index Explained 的 1M×128d 内存账）。

刻意不模拟（L1 §12 债表中声明、本模块止于 L2 的边界）：量化压缩（int8/SQ/PQ——Milvus
quantization+refiner 机制已在 [6] 以「粗召回+精排」同构触及，真实量化误差模型不做）、
文档更新语义（upsert → 图修复）、多租户 ACL（与 nano-data-platform 治理合流，见 tutorial §11）。

运行：python3 L2_hybrid_search_hnsw_and_rerank.py
依赖：核心机制路径零依赖（纯标准库，CPU，~10-20s）；跨级锚真实路径 = sentence-transformers
      + all-MiniLM-L6-v2 本地快照（承 L1 钉住口径，HF_HUB_OFFLINE=1；无依赖/无快照自动走
      fallback，两路径期望值分开声明，数字不混比——承 L1 §10(b) 反幻觉纪律）。
确定性：md5 驱动一切「随机」（层级分配 / 合成语料），无内建 hash()、无未钉种子 RNG、
      无 wall-clock 进 digest；elapsed 行可掩码（口径承 L0 家族：sed '/^[[:space:]]*elapsed/d'）。
digest 口径声明：digest 只覆盖纯 Python 确定性部分（BM25/RRF/HNSW/分片/重排/nDCG-lexical），
      模型相关数字（跨级锚 + 语义 dense 混合表）由 L1 digest 锚与各路径 EXP 表分别钉住。
"""
import array, hashlib, heapq, json, math, os, re, sys, time
from collections import Counter

# 环境钉死在 import 前（承 L1：默认离线 = 快照钉住，索引构建绝不隐式拉新快照）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if os.environ.get("NANO_RAG_ALLOW_DOWNLOAD"):
    os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

CHECKS = []
def check(name, cond):
    CHECKS.append(bool(cond))
    print(f"  [check {len(CHECKS):02d}] {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: raise SystemExit("self-check failed: " + name)

def tokenize(text):  # 与 L0/L1 逐字同款
    return re.findall(r"[a-z0-9]+", text.lower())

def _h(s):  # md5 而非内建 hash()（PYTHONHASHSEED 随机盐会毁掉可复现性，承 L0 §3(d)）
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

def _f32(v):  # 统一量化到 float32（承 L1：内存 == 落盘 == 重载逐位同一）
    return list(array.array("f", v))

def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))

# ---- fixture：与 L0/L1 逐字同一（同一语料、同一评估集——跨级锚的前提）----
CORPUS = [
    ("d01", "how to reset your password: open settings, choose security, click reset password and confirm by email"),
    ("d02", "billing faq: if you are charged twice, we refund the duplicate charge within five business days"),
    ("d03", "a vector index stores document embeddings and answers nearest neighbor queries by similarity"),
    ("d04", "gpu memory grows with batch size; gradient checkpointing trades compute for memory"),
    ("d05", "a data lake keeps raw immutable batches; curated snapshots are derived and versioned"),
    ("d06", "the scheduler runs tasks in topological order and retries transient failures with backoff"),
    ("d07", "embedding models map sentences to vectors so that similar meanings stay close in the space"),
    ("d08", "the tokenizer splits text into subword pieces before the model reads it"),
    ("d09", "checkpointing saves weights and optimizer state so training can resume after a crash"),
    ("d10", "reinforcement learning rewards the policy when answers pass the rule based verifier"),
    ("d11", "retrieval augmented generation grounds the model on fetched documents to reduce hallucination"),
    ("d12", "deduplication removes near duplicate documents before training to avoid memorization"),
    ("d13", "api rate limit: requests above the quota receive 429 until the next window opens"),
    ("d14", "a car needs regular maintenance: oil change, tire rotation and brake inspection every year"),
]
EVAL = [("reset my password please", "d01"),
        ("i was charged two times", "d02"),
        ("nearest neighbor query over embeddings", "d03"),
        ("how to resume training after a crash", "d09"),
        ("reduce hallucination with fetched documents", "d11"),
        ("remove near duplicate documents before training", "d12"),
        ("automobile fuel economy", "d14"),
        ("bonuses for correct behavior", "d10")]
# nDCG 分级判据（graded relevance）：grade 2 = 正解，grade 1 = 相关但非正解
# recall 只问「找没找到」，nDCG 问「排得好不好」——分级判据让两者的分歧可测（§7）
GRADED = [("checkpointing after a crash", {"d09": 2, "d04": 1}),
          ("vector index similarity search", {"d03": 2, "d07": 1})]

# ================= [1] 跨级锚：L1 函数逐字搬运（同集对照的基线必须是同一个函数） =================
D_LEX = 256
def embed_lexical(text):  # L1 embed_lexical 逐字同款（= L0 embed）
    v = [0.0] * D_LEX
    for tok in tokenize(text):
        v[_h("w:" + tok) % D_LEX] += 1.0
        pad = f"#{tok}#"
        for i in range(len(pad) - 2):
            v[_h("g:" + pad[i:i + 3]) % D_LEX] += 0.25
    n = math.sqrt(sum(x * x for x in v))
    return _f32([x / n for x in v] if n else v)

# L1 fallback embedder 逐字同款（SYN_CLUSTERS + bigram，dim 512）——诚实声明承 L1：
# 簇表定向覆盖本评估集的同义词对，其 recall 是「开卷」，不是学到的泛化，数字不与真实模型比。
SYN_CLUSTERS = [
    frozenset({"car", "automobile", "vehicle"}),
    frozenset({"bonus", "bonuses", "reward", "rewards", "incentive"}),
    frozenset({"buy", "purchase", "purchased", "bought"}),
    frozenset({"quick", "fast", "rapid"}),
    frozenset({"cheap", "inexpensive", "affordable"}),
    frozenset({"begin", "start", "commence"}),
    frozenset({"end", "finish", "terminate"}),
    frozenset({"help", "assist", "support"}),
    frozenset({"big", "large", "huge"}),
    frozenset({"small", "tiny", "little"}),
    frozenset({"evict", "evicts", "eviction"}),
    frozenset({"conflict", "conflicts"}),
]
_CLUSTER_OF = {w: i for i, cl in enumerate(SYN_CLUSTERS) for w in cl}
D_FB = 512
def embed_fallback(text):
    v = [0.0] * D_FB
    toks = tokenize(text)
    for tok in toks:
        v[_h("w:" + tok) % D_FB] += 1.0
        c = _CLUSTER_OF.get(tok)
        if c is not None:
            v[_h(f"c:{c}") % D_FB] += 2.0
        pad = f"#{tok}#"
        for i in range(len(pad) - 2):
            v[_h("g:" + pad[i:i + 3]) % D_FB] += 0.25
    for a, b in zip(toks, toks[1:]):
        v[_h("b:" + a + " " + b) % D_FB] += 0.5
    n = math.sqrt(sum(x * x for x in v))
    return _f32([x / n for x in v] if n else v)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
def load_real_embedder():  # L1 逐字同款
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID)
    dim = int(model.get_embedding_dimension())
    snapshot = "unknown"
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(MODEL_ID, "config.json")
        if isinstance(p, str):
            snapshot = os.path.basename(os.path.dirname(p))
    except Exception:
        pass
    def embed(text):
        return _f32(model.encode([text])[0].tolist())
    return embed, dim, snapshot

class FlatL1:  # L1 PersistentFlatIndex 的检索核逐字同款（跨级 digest 锚的前提）
    def __init__(self, embed, dim):
        self.embed, self.dim = embed, dim
        self.docs, self.vecs = [], []
    def add(self, pairs):
        for did, text in pairs:
            self.docs.append((did, text)); self.vecs.append(self.embed(text))
    def search(self, query, k=3, tau=0.0):
        qv = self.embed(query)
        ranked = sorted(((cosine(qv, v), did) for (did, _), v in zip(self.docs, self.vecs)), reverse=True)
        return [(did, round(s, 4)) for s, did in ranked[:k] if s >= tau], len(self.docs) * self.dim

def evaluate_l1(index, eval_set, k):  # L1 evaluate 逐字同款
    hits_n, prec, rr, per_q = 0, 0.0, 0.0, []
    for q, gold in eval_set:
        ids = [did for did, _ in index.search(q, k=k)[0]]
        r = ids.index(gold) + 1 if gold in ids else 0
        hits_n += 1 if r else 0
        prec += (1.0 / k) if r else 0.0
        rr += 1.0 / r if r else 0.0
        top = index.search(q, k=1)[0]
        per_q.append((q, gold, r, top[0] if top else ("-", 0.0)))
    n = len(eval_set)
    return dict(recall=hits_n / n, precision=prec / n, mrr=rr / n, per_q=per_q)

# ================= [2] BM25：Lucene BM25Similarity 公式（稀疏侧打分器） =================
# 锚点（抓取件 lucene_BM25Similarity.java，2026-08-16 抓取，行号以抓取件为准）：
#   默认参数 k1=1.2 / b=0.75（L109/L122 构造器 this(1.2f, 0.75f, ...)）
#   idf = ln(1 + (N - n + 0.5)/(n + 0.5))（L139-140；L183 解释串同式）
#   tf 项 = tf/(tf + k1·(1-b+b·dl/avgdl))（L219 norm 缓存 + L257-267 doScore 重写；
#   与经典 Okapi 形式 tf·(k1+1)/(...) 只差每词常数 (k1+1)，不影响排序——代数恒等）
class BM25:
    K1, B = 1.2, 0.75
    def __init__(self, pairs):
        self.toks = {did: tokenize(text) for did, text in pairs}
        self.N = len(pairs)
        self.dl = {did: len(ts) for did, ts in self.toks.items()}
        self.avgdl = sum(self.dl.values()) / self.N
        df = Counter()
        for ts in self.toks.values():
            df.update(set(ts))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = {did: Counter(ts) for did, ts in self.toks.items()}
    def score(self, qtext, did):
        s, dl = 0.0, self.dl[did]
        norm = self.K1 * ((1 - self.B) + self.B * dl / self.avgdl)
        for t in tokenize(qtext):
            f = self.tf[did].get(t, 0)
            if f:
                s += self.idf.get(t, 0.0) * f / (f + norm)
        return s
    def rank(self, qtext):
        # score>0 过滤：真实引擎的检索只返回有匹配的结果列表（Elastic RRF 公式以
        # 「if d in result(q)」为前提——未匹配文档不入融合列表）。若返回全语料含零分档，
        # 零分档会按 doc_id 序注入伪信号、稀释强路径优势——见 [3] 反例探针。
        scored = ((did, self.score(qtext, did)) for did in self.toks)
        return sorted(((did, s) for did, s in scored if s > 0.0), key=lambda p: (-p[1], p[0]))

# ================= [3] 融合术：min-max 加权（OpenSearch 口径）与 RRF（Cormack k=60） =================
def minmax_weighted_fuse(score_lists, weights):
    """OpenSearch normalization-processor 同构：每路 min-max 归一 → 加权算术平均。
    锚点：opensearch_hybrid-search 文档示例 technique=min_max + arithmetic_mean + weights。
    某路对该查询无结果（空列表）则贡献 0——没有归一基线可言（真实引擎该路不返回即不参与）。"""
    out = {}
    for lst, w in zip(score_lists, weights):
        if not lst:
            continue
        vals = [s for _, s in lst]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        for did, s in lst:
            out[did] = out.get(did, 0.0) + w * (s - lo) / rng
    return sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))

def raw_weighted_fuse(score_lists, weights):
    """裸加权和 Σ w·s：不做任何归一（[3] 尺度探针专用，生产不用——两路分数的尺度本就不可比：
    BM25 无上界 vs cosine ∈ [-1,1]；谁乘上放大系数谁赢者通吃）。"""
    out = {}
    for lst, w in zip(score_lists, weights):
        for did, s in lst:
            out[did] = out.get(did, 0.0) + w * s
    return sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))

def rrf_fuse(rankings, k=60):
    """RRFscore(d) = Σ_r 1/(k + r(d))（Cormack et al. 2009，k=60 在 pilot 中钉死后未再调；
    Elastic rrf retriever rank_constant 默认 60、「requires no tuning」）。只看名次、不看分数
    ——这是它「requires no tuning」的机制根源。Milvus 侧（2026-08-18 live 核验）：pymilvus
    Collection.hybrid_search 的 rerank 是必选参数、无默认值（pymilvus master orm/collection.py
    L896-899），官方 multi-vector-search 示例显式选用 RRFRanker（抓取件 L823）——「默认 rrf」
    之说没有来源支持。融合前提：只融合各路返回的结果列表（见 BM25.rank）。"""
    score = {}
    for ranking in rankings:
        for pos, (did, _) in enumerate(ranking, start=1):
            score[did] = score.get(did, 0.0) + 1.0 / (k + pos)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))

def evaluate_ranked(ranked, eval_set, k):
    """对任意融合产物算 recall@k / MRR（ranked = [(did, score)] 降序）。"""
    hits, rr = 0, 0.0
    for q, gold in eval_set:
        ids = [did for did, _ in ranked.get(q, [])[:k]]
        r = ids.index(gold) + 1 if gold in ids else 0
        hits += 1 if r else 0
        rr += 1.0 / r if r else 0.0
    n = len(eval_set)
    return dict(recall=hits / n, mrr=rr / n)

# ================= [4] HNSW：从零实现（对照 hnswlib/hnswalg.h） =================
# 锚点（抓取件 hnswlib_hnswalg.h，2026-08-16 抓取）：
#   mL = 1/ln(M)（L142 mult_ = 1 / log(1.0 * M_)）；层级 = floor(-ln(u)·mL)（L207-209
#   getRandomLevel：-log(distribution(...)) * reverse_size；L1186 调用 getRandomLevel(mult_)）
#   layer-0 度数上限 2M（L113 maxM0_ = M_ * 2）；ef_construction ≥ M（L114）；默认 ef=10（L115）
#   searchKnn = 上层贪心下降（L1277-1302）+ 底层束搜索 max(ef, k)（L1307-1308）
#   停止条件：最近候选比最远结果还远且结果已满 ef（L353）
#   getNeighborsByHeuristic2（L443-480）：已选邻居若比候选更靠近候选，则候选落选——保角度多样性
#   距离计算计数：hnswlib metric_distance_computations（L1286-1287）——本实现的 dist_comps 同构
class HNSW:
    def __init__(self, M=8, ef_construction=32, heuristic=True, tag=""):
        self.M, self.maxM0, self.efC = M, 2 * M, ef_construction
        self.mL = 1.0 / math.log(M)
        self.heuristic, self.tag = heuristic, tag
        self.ids, self.vecs, self.levels, self.graph = [], [], [], []
        self.entry, self.maxlevel = -1, -1
        self.dist_comps = 0
    def _dist(self, a, b):  # 向量均已 L2 归一 → 1 - 点积 = cosine 距离
        self.dist_comps += 1
        return 1.0 - sum(x * y for x, y in zip(a, b))
    def level_for(self, i):  # md5 派生均匀分布 u ∈ (0,1) → 指数衰减层级（hnswlib L207-209 同构）
        u = _h(f"hnsw-level:{self.tag}:{i}") / 2 ** 128
        return int(-math.log(max(u, 2 ** -128)) * self.mL)
    def _search_layer(self, q, eps, ef, level):
        visited, cand, res = set(eps), [], []
        for ep in eps:
            d = self._dist(q, self.vecs[ep])
            heapq.heappush(cand, (d, ep)); heapq.heappush(res, (-d, ep))
        while cand:
            d_c, c = heapq.heappop(cand)
            if d_c > -res[0][0] and len(res) >= ef:
                break  # hnswlib L353 停止条件
            for nb in self.graph[c][level]:
                if nb in visited: continue
                visited.add(nb)
                d_n = self._dist(q, self.vecs[nb])
                if d_n < -res[0][0] or len(res) < ef:
                    heapq.heappush(cand, (d_n, nb)); heapq.heappush(res, (-d_n, nb))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted((-d, i) for d, i in res)
    def _select(self, cands, Mlv):  # heuristic2（hnswlib L443-480）或 naive top-M
        cands = sorted(cands)
        if not self.heuristic:
            return [i for _, i in cands[:Mlv]]
        chosen = []
        for d_q, c in cands:
            if len(chosen) >= Mlv: break
            if all(self._dist(self.vecs[c], self.vecs[r]) >= d_q for r in chosen):
                chosen.append(c)  # 没有已选邻居比「候选到查询」更靠近候选 → 保留（多样性）
        for _, c in cands:  # 兜底补齐（hnswlib 同款第二遍）
            if len(chosen) >= Mlv: break
            if c not in chosen: chosen.append(c)
        return chosen
    def add(self, doc_id, vec, level=None):
        i = len(self.vecs)
        self.ids.append(doc_id); self.vecs.append(vec)
        lv = self.level_for(i) if level is None else level
        self.levels.append(lv)
        self.graph.append([[] for _ in range(lv + 1)])
        if self.entry < 0:
            self.entry, self.maxlevel = i, lv
            return
        ep = self.entry
        for l in range(self.maxlevel, lv, -1):  # 上层贪心下降（hnswlib L1277-1302 同构）
            changed = True
            while changed:
                changed = False
                best = self._dist(vec, self.vecs[ep])
                for nb in self.graph[ep][l]:
                    d = self._dist(vec, self.vecs[nb])
                    if d < best:
                        best, ep, changed = d, nb, True
        eps = [ep]
        for l in range(min(lv, self.maxlevel), -1, -1):
            W = self._search_layer(vec, eps, self.efC, l)
            Mlv = self.maxM0 if l == 0 else self.M
            self.graph[i][l] = self._select(W, Mlv)
            for nb in self.graph[i][l]:  # 双向连接 + 溢出按同一规则修剪（hnswlib L603/L1052 同构）
                gl = self.graph[nb][l]; gl.append(i)
                if len(gl) > Mlv:
                    self.graph[nb][l] = self._select(
                        [(self._dist(self.vecs[nb], self.vecs[x]), x) for x in gl], Mlv)
            eps = [x for _, x in W]
        if lv > self.maxlevel:
            self.entry, self.maxlevel = i, lv
    def search(self, q, k, ef):
        ep = self.entry
        for l in range(self.maxlevel, 0, -1):
            changed = True
            while changed:
                changed = False
                best = self._dist(q, self.vecs[ep])
                for nb in self.graph[ep][l]:
                    d = self._dist(q, self.vecs[nb])
                    if d < best:
                        best, ep, changed = d, nb, True
        W = self._search_layer(q, [ep], max(ef, k), 0)  # hnswlib L1307-1308：max(ef, k)
        return [(self.ids[i], d) for d, i in W[:k]]

def brute_knn(q, vecs, ids, k):
    # tie-break 统一按 doc_id（与分片合并同口径——[5] 的精确不变量在距离打平时也严格成立）
    scored = sorted((1.0 - sum(x * y for x, y in zip(q, v)), ids[i]) for i, v in enumerate(vecs))
    return [(did, d) for d, did in scored[:k]]

# ---- 合成语料：主题词 + 填充词的确定性世界（md5 驱动，无 RNG 状态）----
D_SYN = 128
def embed_syn(text):  # 与 L0 embed 同算法、维度 128（toy 口径：合成世界的几何只需词面信号）
    v = [0.0] * D_SYN
    for tok in tokenize(text):
        v[_h("w:" + tok) % D_SYN] += 1.0
        pad = f"#{tok}#"
        for i in range(len(pad) - 2):
            v[_h("g:" + pad[i:i + 3]) % D_SYN] += 0.25
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v

def _ri(key, lo, hi):
    return lo + _h(key) % (hi - lo + 1)

def synth_world(n_docs, n_topics=16, seed="synth-v1"):
    topics = [[f"t{t:02d}w{j}" for j in range(10)] for t in range(n_topics)]
    filler = [f"fill{j:02d}" for j in range(40)]
    docs = []
    for i in range(n_docs):
        t1 = _ri(f"{seed}:t1:{i}", 0, n_topics - 1)
        t2 = _ri(f"{seed}:t2:{i}", 0, n_topics - 1) if _ri(f"{seed}:mix:{i}", 0, 9) < 3 else None
        words = []
        for j in range(_ri(f"{seed}:nt:{i}", 7, 10)):
            pool = topics[t1] if (t2 is None or j % 3 != 2) else topics[t2]
            words.append(pool[_ri(f"{seed}:w:{i}:{j}", 0, 9)])
        for j in range(_ri(f"{seed}:nf:{i}", 3, 5)):
            words.append(filler[_ri(f"{seed}:f:{i}:{j}", 0, 39)])
        words.sort(key=lambda w: _h(f"{seed}:pos:{i}:{w}"))
        docs.append((f"s{i:05d}", " ".join(words), t1))
    queries = []
    for qi in range(32):
        t = _ri(f"{seed}:qt:{qi}", 0, n_topics - 1)
        qs = " ".join(topics[t][_ri(f"{seed}:qw:{qi}:{j}", 0, 9)] for j in range(6))
        queries.append((f"q{qi:02d}", qs, t))
    return docs, queries, topics

def ann_recall(ann_hits, truth_hits, k):  # ANN recall 定义：与 brute-force top-k 的重合率
    a = {did for did, _ in ann_hits[:k]}; b = {did for did, _ in truth_hits[:k]}
    return len(a & b) / k

# ================= [6] 两阶段重排：HNSW 粗召回 top-N → reranker 精排 top-k =================
class OverlapReranker:
    """显式 mock cross-encoder：idf 加权词重叠（查询与文档成对打分）。
    真实 cross-encoder = BERT 类模型对 (q, d) 联合编码出相关性分——机制同类（成对细粒度
    交互，比双塔独立编码贵但更准），代价与精度都高出一个量级 [TODO: verify on real system]。
    对照：Milvus refiner「取 topK×expansion rate 候选，用更高精度重算，返回最终 topK」。"""
    def __init__(self, bm25):
        self.idf = bm25.idf
    def score(self, qtext, text):
        qt, dt = set(tokenize(qtext)), set(tokenize(text))
        common = qt & dt
        if not common:
            return 0.0
        num = sum(self.idf.get(t, 0.0) for t in common)
        den = math.sqrt(sum(self.idf.get(t, 0.0) ** 2 for t in qt) *
                        sum(self.idf.get(t, 0.0) ** 2 for t in dt))
        return num / den if den else 0.0

# ================= [7] nDCG@k：分级相关性（recall 的排名敏感版） =================
def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

def ndcg_at_k(ranked_ids, graded, k):
    grades = [graded.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(grades) / idcg if idcg else 0.0

# ================= 期望值表（双路径分开声明，数字不混比——承 L1 §10(b)） =================
# real 路径 = all-MiniLM-L6-v2 快照钉住；fallback = L1 显式 mock（开卷覆盖，口径不与 real 比）
# hybrid 各值由双路径实测派生（score>0 结果列表语义，非先验声明）：旧表曾先验写成
# hybrid 1.0/1.0，随后被实测证伪——「期望值表必须实测派生」，教训见 tutorial §10。
EXP_REAL = dict(l1_digest="4c8c4347e418b602", hybrid_r1=0.750, hybrid_r3=1.0,
                dense_r1=0.875, dense_r3=1.0, dense_mrr=0.9375,
                auto_rank_hybrid=2)
EXP_FB = dict(l1_digest="008632d5402a86b7", hybrid_r1=0.875, hybrid_r3=1.0,
              dense_r1=1.0, dense_r3=1.0, dense_mrr=1.0,
              auto_rank_hybrid=1)

def main():
    t0 = time.time()
    print("== nano-rag-retrieval L2: 混合检索 + HNSW + 两阶段重排 + 检索评估升级（自 L1 的 K+1）==")

    # ---- [0] 路径判定（承 L1 双路径契约）----
    try:
        embed, dim, snapshot = load_real_embedder()
        path, model_name = "real", MODEL_ID
        print(f"\n[0] embedding 路径 = real（{MODEL_ID}；snapshot {str(snapshot)[:12]}…，dim {dim}，离线钉住）")
    except Exception as e:
        embed, dim, snapshot = embed_fallback, D_FB, "n/a"
        path, model_name = "fallback", "fallback-synonym-cluster-v1"
        print(f"\n[0] embedding 路径 = fallback（显式 mock：手写同义词簇 + bigram，dim {dim}，零依赖）")
        print(f"    原因：{type(e).__name__}（真实模型需 sentence-transformers + 本地快照，承 L1）")
    EXP = EXP_REAL if path == "real" else EXP_FB

    # ---- [1] 跨级锚：L1 基线与 digest 逐字复现 ----
    print("\n[1] 跨级锚：L1 的 lexical 基线与 digest 必须逐字复现（同函数同 fixture）")
    lex = FlatL1(embed_lexical, D_LEX)
    lex.add(CORPUS)
    ev1, ev3 = evaluate_l1(lex, EVAL, k=1), evaluate_l1(lex, EVAL, k=3)
    miss = sorted(g for _, g, r, _ in ev3["per_q"] if r == 0)
    print(f"  lexical 基线: recall@1 = {ev1['recall']:.3f} / recall@3 = {ev3['recall']:.3f} / "
          f"MRR = {ev3['mrr']:.4f}，MISS = {miss}")
    check("跨级锚: lexical 基线与 L0/L1 逐位一致 (0.625 / 0.750 / 0.6875)",
          abs(ev1["recall"] - 0.625) < 1e-9 and abs(ev3["recall"] - 0.75) < 1e-9
          and abs(ev3["mrr"] - 0.6875) < 1e-9)
    check("跨级锚: MISS 集合 == [d10, d14]（L0 刻意设计的两条纯同义词查询）", miss == ["d10", "d14"])
    sem = FlatL1(embed, dim)
    sem.add(CORPUS)
    ev3p = evaluate_l1(sem, EVAL, k=3)
    body = json.dumps({"path": path, "per_q": ev3p["per_q"], "mrr": round(ev3p["mrr"], 6)},
                      sort_keys=True, ensure_ascii=False).encode()
    dg_l1 = hashlib.sha256(body).hexdigest()[:16]
    print(f"  L1 digest ({path}): {dg_l1}  期望: {EXP['l1_digest']}")
    check(f"跨级锚: L1 retrieval digest ({path}) 逐字复现", dg_l1 == EXP["l1_digest"])

    # ---- [2] BM25：Lucene 公式 + 盲点机器证明 ----
    print("\n[2] BM25 稀疏打分器（Lucene BM25Similarity 公式：k1=1.2 / b=0.75）")
    micro = [("A", "x y"), ("B", "x x z w")]  # 手算锚：N=2, df(x)=2, avgdl=3
    bm_micro = BM25(micro)
    idf_x = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))          # ln(1.2)，闭式
    sA = idf_x * 1 / (1 + 1.2 * (0.25 + 0.75 * 2 / 3))       # tf=1, dl=2
    sB = idf_x * 2 / (2 + 1.2 * (0.25 + 0.75 * 4 / 3))       # tf=2, dl=4
    check("BM25 手算锚: score(A) == ln(1.2)·1/(1+0.9)（闭式独立重算）",
          abs(bm_micro.score("x", "A") - sA) < 1e-12)
    check("BM25 手算锚: score(B) == ln(1.2)·2/(2+1.5)（tf=2 但长度 4）",
          abs(bm_micro.score("x", "B") - sB) < 1e-12)
    check("BM25 tf 饱和: 词频 2 的得分 < 2× 词频 1（亚线性，k1 饱和）", sB < 2 * sA)
    check("BM25 idf 恒正: log(1+·) 形式使任何 df 都有正权重（Lucene 口径）",
          all(v > 0 for v in bm_micro.idf.values()))
    bm = BM25(CORPUS)
    bm_ranked = {q: bm.rank(q) for q, _ in EVAL}
    ev_bm1 = evaluate_ranked(bm_ranked, EVAL, k=1)
    ev_bm3 = evaluate_ranked(bm_ranked, EVAL, k=3)
    s_auto = bm.score("automobile fuel economy", "d14")
    s_bonus = bm.score("bonuses for correct behavior", "d10")
    print(f"  BM25 单路: recall@1 = {ev_bm1['recall']:.3f} / recall@3 = {ev_bm3['recall']:.3f} / "
          f"MRR = {ev_bm3['mrr']:.4f}")
    print(f"  纯同义词查询得分: 'automobile fuel economy'→d14 = {s_auto:.4f}，"
          f"'bonuses for correct behavior'→d10 = {s_bonus:.4f}（词面零重叠 = 恒 0，稀疏侧盲点）")
    check("BM25 盲点机器证明: 两条纯同义词查询对 gold 得分恰为 0", s_auto == 0.0 and s_bonus == 0.0)
    check("BM25 单路 recall@3 < 1.0（两条同义词查询注定 MISS——L0 鸿沟在稀疏侧原样存在）",
          ev_bm3["recall"] < 1.0)
    r_ch = [did for did, _ in bm_ranked["i was charged two times"][:1]]
    check("BM25 词面强信号: 'i was charged two times' 的 gold(d02) rank 1（idf 加权 'charged'）",
          r_ch == ["d02"])

    # ---- [3] 混合检索：dense + BM25 融合 ----
    print(f"\n[3] 混合检索（dense={path} 路径 + BM25 稀疏，14 篇语料 × 8 条标注查询）")
    dense_ranked = {q: sorted(((did, cosine(sem.embed(q), v))
                               for (did, _), v in zip(sem.docs, sem.vecs)),
                              key=lambda p: (-p[1], p[0]))
                    for q, _ in EVAL}
    ev_d1 = evaluate_ranked(dense_ranked, EVAL, k=1)
    ev_d3 = evaluate_ranked(dense_ranked, EVAL, k=3)
    w_ranked = {q: minmax_weighted_fuse([dense_ranked[q], bm_ranked[q]], [0.3, 0.7]) for q, _ in EVAL}
    rrf_ranked = {q: rrf_fuse([dense_ranked[q], bm_ranked[q]], k=60) for q, _ in EVAL}
    ev_w1, ev_w3 = evaluate_ranked(w_ranked, EVAL, k=1), evaluate_ranked(w_ranked, EVAL, k=3)
    ev_rrf1, ev_rrf3 = evaluate_ranked(rrf_ranked, EVAL, k=1), evaluate_ranked(rrf_ranked, EVAL, k=3)
    print(f"  {'方法':<24s} recall@1  recall@3  MRR")
    for name, e1, e3 in [("dense only", ev_d1, ev_d3), ("BM25 only", ev_bm1, ev_bm3),
                         ("minmax+weighted .3/.7", ev_w1, ev_w3),
                         ("RRF k=60", ev_rrf1, ev_rrf3)]:
        print(f"  {name:<24s} {e1['recall']:.3f}     {e3['recall']:.3f}     {e3['mrr']:.4f}")
    check(f"dense only 复刻 L1: recall@1 {EXP['dense_r1']} / recall@3 {EXP['dense_r3']} / "
          f"MRR {EXP['dense_mrr']}",
          abs(ev_d1["recall"] - EXP["dense_r1"]) < 1e-9 and abs(ev_d3["recall"] - EXP["dense_r3"]) < 1e-9
          and abs(ev_d3["mrr"] - EXP["dense_mrr"]) < 1e-9)
    check(f"混合 RRF（实测派生）: recall@1 == {EXP['hybrid_r1']} 且 recall@3 == {EXP['hybrid_r3']}",
          abs(ev_rrf1["recall"] - EXP["hybrid_r1"]) < 1e-9 and abs(ev_rrf3["recall"] - EXP["hybrid_r3"]) < 1e-9)
    check("混合 ≥ 两路短板: RRF recall@3 ≥ BM25 recall@3（融合不丢稀疏侧能力）",
          ev_rrf3["recall"] >= ev_bm3["recall"])
    ranks_rrf = {q: [did for did, _ in rrf_ranked[q]][:3] for q, _ in EVAL}
    # 受控 rank 计算：gold 不在 top-3 → r_auto = 0 → check 受控 FAIL（fail-loud 走 check 通道，
    # 不是 .index() 失控 ValueError——这是边界输入必须显式处理的修复点）
    r_auto = ranks_rrf["automobile fuel economy"].index("d14") + 1 if "d14" in ranks_rrf["automobile fuel economy"] else 0
    check(f"'automobile fuel economy' 混合后 rank == {EXP['auto_rank_hybrid']}"
          f"（BM25 得 0 分，dense 侧救回——融合的并集能力）",
          r_auto == EXP["auto_rank_hybrid"])
    # 反例探针（bug 教材化，tutorial §10）：若 BM25.rank() 返回全语料（含零分档、tie-break by doc_id），
    # 零分档按 doc_id 序注入伪信号——同义词查询 gold 被钉列尾，RRF 反而稀释 dense 优势。
    # 真实引擎只融合各路返回的结果列表（Elastic RRF 公式「if d in result(q)」前提），score>0 语义即此。
    bm_full = {q: sorted(((did, bm.score(q, did)) for did, _ in CORPUS), key=lambda p: (-p[1], p[0]))
               for q, _ in EVAL}
    rrf_full = {q: rrf_fuse([dense_ranked[q], bm_full[q]], k=60) for q, _ in EVAL}
    ev_full1, ev_full3 = evaluate_ranked(rrf_full, EVAL, k=1), evaluate_ranked(rrf_full, EVAL, k=3)
    r_auto_full = [did for did, _ in rrf_full["automobile fuel economy"]].index("d14") + 1
    print(f"  反例探针（全列含零分档）: RRF recall@1 = {ev_full1['recall']:.3f} / recall@3 = {ev_full3['recall']:.3f}"
          f"（vs score>0 语义 {ev_rrf1['recall']:.3f} / {ev_rrf3['recall']:.3f}）；'automobile' gold rank {r_auto} → {r_auto_full}")
    # 尺度探针：加权融合以「分数可比」为前提。故意把 BM25 分数 ×1000，三种融合术现出原形：
    # 裸加权和（无归一）→ 大量纲侧赢者通吃；minmax 归一 → 位序不变；RRF → 只看名次、天然不变。
    bm_scaled = {q: [(did, s * 1000.0) for did, s in bm_ranked[q]] for q, _ in EVAL}
    raw_fused = {q: raw_weighted_fuse([dense_ranked[q], bm_scaled[q]], [0.3, 0.7]) for q, _ in EVAL}
    w_scaled = {q: minmax_weighted_fuse([dense_ranked[q], bm_scaled[q]], [0.3, 0.7]) for q, _ in EVAL}
    rrf_raw = {q: rrf_fuse([dense_ranked[q], bm_scaled[q]], k=60) for q, _ in EVAL}
    has_bm = [q for q, _ in EVAL if bm_ranked[q]]  # 纯同义词查询 BM25 无非零命中 → 不参与尺度对照
    top1_raw = [raw_fused[q][0][0] for q in has_bm]
    top1_bm = [bm_ranked[q][0][0] for q in has_bm]
    print(f"  尺度探针（BM25 分数 ×1000）: 裸加权和(无归一) top-1 == BM25 top-1（{len(has_bm)} 条非零命中查询）: "
          f"{top1_raw == top1_bm}（量纲侧赢者通吃）；minmax 归一 top-3 位序不变: "
          f"{all([d for d, _ in w_scaled[q][:3]] == [d for d, _ in w_ranked[q][:3]] for q, _ in EVAL)}；"
          f"RRF top-10 位序逐位不变: {all(rrf_raw[q][:10] == rrf_ranked[q][:10] for q, _ in EVAL)}")
    check(f"尺度探针: 裸加权和（无归一）退化为大量纲侧（top-1 全同 BM25，{len(has_bm)} 条非零命中查询）",
          top1_raw == top1_bm)
    check("尺度探针: RRF 对分数正缩放不变（只看名次，top-10 逐位一致）",
          all(rrf_raw[q][:10] == rrf_ranked[q][:10] for q, _ in EVAL))

    # ---- [4] HNSW：结构不变量 + recall/cost 权衡曲面 ----
    print("\n[4] HNSW（从零实现，对照 hnswlib）：合成语料上的 recall/成本权衡")
    sweep = []
    truths = {}
    for N in (200, 800, 1600):
        docs, queries, _ = synth_world(N)
        vecs = [embed_syn(t) for _, t, _ in docs]
        ids = [did for did, _, _ in docs]
        qvecs = [(qid, embed_syn(qt), t) for qid, qt, t in queries]
        truth = {qid: brute_knn(qv, vecs, ids, 10) for qid, qv, _ in qvecs}
        truths[N] = (docs, vecs, ids, qvecs, truth)
        g = HNSW(M=8, ef_construction=32, tag=f"n{N}")
        for (did, _, _), v in zip(docs, vecs):
            g.add(did, v)
        lv_hist = Counter(g.levels)
        frac_ge1 = sum(n for l, n in lv_hist.items() if l >= 1) / N
        deg0 = max(len(g.graph[i][0]) for i in range(N))
        deg_up = max((len(g.graph[i][l]) for i in range(N) for l in range(1, len(g.graph[i]))), default=0)
        check(f"N={N}: layer-0 度数 ≤ 2M=16 且上层 ≤ M=8（hnswlib L113 maxM0 口径）",
              deg0 <= 16 and deg_up <= 8)
        if N == 800:  # 层级分布断言只在一个 N 上打（指数衰减：P(l≥1) = 1/M = 0.125）
            check(f"N={N}: 层级 ≥1 占比 {frac_ge1:.3f} ≈ 1/M = 0.125（指数衰减，±0.05 容差）",
                  abs(frac_ge1 - 1 / 8) < 0.05)
        for ef in (4, 16, 64):
            g.dist_comps = 0
            recs = []
            for qid, qv, _ in qvecs:
                hits = g.search(qv, 10, ef)
                recs.append(ann_recall(hits, truth[qid], 10))
            dc = g.dist_comps / len(qvecs)
            sweep.append((N, ef, sum(recs) / len(recs), dc))
            print(f"  N={N:>5d}  ef={ef:>3d}: recall@10 = {sweep[-1][2]:.3f}   "
                  f"距离计算 = {dc:6.1f}/query（brute = {N}/query）")
    check("HNSW 省距离: N=1600/ef=16 的每查询距离计算 < brute-force 的 1600",
          [dc for N, ef, r, dc in sweep if (N, ef) == (1600, 16)][0] < 1600)
    r_1600 = {ef: r for N, ef, r, dc in sweep if N == 1600}
    check("recall 随 ef 单调不降: N=1600 时 ef 4→16→64（ef = 召回宽度旋钮）",
          r_1600[4] <= r_1600[16] <= r_1600[64])
    check("高 ef 高召回: N=1600/ef=64 recall@10 ≥ 0.95（recall/成本权衡的可控端）",
          r_1600[64] >= 0.95)
    # heuristic2 消融：同一层级序列、同一插入序，只换邻居选择规则
    docs8, vecs8, ids8, qvecs8, truth8 = truths[800]
    levels8 = [HNSW(tag="abl").level_for(i) for i in range(800)]
    g_h, g_n = HNSW(tag="abl-h", heuristic=True), HNSW(tag="abl-n", heuristic=False)
    for (did, _, _), v, lv in zip(docs8, vecs8, levels8):
        g_h.add(did, v, level=lv); g_n.add(did, v, level=lv)
    rec_h = sum(ann_recall(g_h.search(qv, 10, 16), truth8[qid], 10) for qid, qv, _ in qvecs8) / 32
    rec_n = sum(ann_recall(g_n.search(qv, 10, 16), truth8[qid], 10) for qid, qv, _ in qvecs8) / 32
    print(f"  邻居选择消融（N=800, ef=16，同层级同插入序）: heuristic2 = {rec_h:.3f} vs naive top-M = {rec_n:.3f}")
    check("heuristic2 多样性选择 recall ≥ naive top-M（hnswlib L443 的价值，论文摘要同款声称）",
          rec_h >= rec_n)

    # ---- [5] 分片 scatter-gather：合并精确性不变量 + 分片 HNSW ----
    print("\n[5] 分片 scatter-gather（N=1600 → 4 片）：top-k 合并的精确性与 ANN 代价")
    docs16, vecs16, ids16, qvecs16, truth16 = truths[1600]
    S = 4
    shards = [(vecs16[i::S], ids16[i::S]) for i in range(S)]
    g16 = HNSW(M=8, ef_construction=32, tag="n1600")
    for (did, _, _), v in zip(docs16, vecs16):
        g16.add(did, v)
    shard_gs = []
    for si, (sv, sid) in enumerate(shards):
        sg = HNSW(M=8, ef_construction=32, tag=f"shard{si}")
        for did, v in zip(sid, sv):
            sg.add(did, v)
        shard_gs.append(sg)
    ok_exact, rec_shard, rec_global = True, [], []
    for qid, qv, _ in qvecs16:
        local = [brute_knn(qv, sv, sid, 10) for sv, sid in shards]
        merged = sorted(((d, did) for lst in local for did, d in lst))[:10]
        if [(did, round(d, 6)) for d, did in merged] != \
           [(did, round(d, 6)) for did, d in truth16[qid]]:
            ok_exact = False
        loc_h = [sg.search(qv, 10, 16) for sg in shard_gs]
        m_h = sorted(((d, did) for lst in loc_h for did, d in lst))[:10]
        rec_shard.append(ann_recall([(did, d) for d, did in m_h], truth16[qid], 10))
        rec_global.append(ann_recall(g16.search(qv, 10, 16), truth16[qid], 10))
    rs, rg = sum(rec_shard) / 32, sum(rec_global) / 32
    print(f"  分片 brute 合并 == 全局 brute top-10: {ok_exact}（32 查询逐位，含分数舍入）")
    print(f"  分片 HNSW recall@10 = {rs:.3f} vs 全局 HNSW = {rg:.3f}（ef=16）")
    check("不变量: 每片返回各自 top-k → 归并 == 全局 top-k（精确，无 ANN 近似——可证明）", ok_exact)
    check("分片 HNSW recall 与全局 HNSW 同量级（差距 ≤ 0.05；ANN 误差不随分片数爆炸）",
          abs(rs - rg) <= 0.05)

    # ---- [6] 两阶段重排：HNSW 粗召回 top-N → reranker 精排 top-k ----
    print("\n[6] 两阶段重排（N=800：HNSW ef=4 粗召回 top-8 → OverlapReranker 精排 top-3，Milvus refiner 同构）")
    docs8, vecs8, ids8, qvecs8, truth8 = truths[800]
    text_of = {did: t for did, t, _ in docs8}
    _, queries8, _ = synth_world(800)  # qvecs8 只存向量；查询文本从确定性世界重取（把向量当文本会触发 AttributeError）
    qtext_of = {qid: qt for qid, qt, _ in queries8}
    # gold = 与查询词面重叠最大的所有文档（并列全收——gold 是「最佳答案集合」，消除任意 tie-break
    # 噪声）；reranker 判据（idf 加权重叠）与 gold 判据同源，对应 refiner「更高精度重算」语义。
    # 本探针机器证明的是两阶段的机制不变量，不是 reranker 的语义正确性（mock 声明见类 docstring）。
    gold_of = {}
    for qid, _, _ in qvecs8:
        qt = set(tokenize(qtext_of[qid]))
        ov = [(len(set(tokenize(text_of[did])) & qt), did) for did, _, _ in docs8]
        mx = max(c for c, _ in ov)
        gold_of[qid] = {did for c, did in ov if c == mx}
    rr = OverlapReranker(BM25([(did, t) for did, t, _ in docs8]))
    r1_hits, rc_hits, re_hits, rescued, calls = [], [], [], 0, 0
    for qid, qv, _ in qvecs8:
        qtext = qtext_of[qid]
        first = g_h.search(qv, 8, 4)  # 低 ef 粗召回：故意让第一阶段不完美（ef<k 被抬到 max(ef,k)=8，hnswlib L1307-1308）
        calls += len(first)
        cands = [did for did, _ in first]
        top3_first = cands[:3]
        reranked = sorted(((rr.score(qtext, text_of[did]), did) for did in cands),
                          key=lambda p: (-p[0], p[1]))
        top3_re = [did for _, did in reranked[:3]]  # 元组是 (score, did)；取错位会得到恒 0 的假数字
        gset = gold_of[qid]
        r1_hits.append(1 if gset & set(top3_first) else 0)
        rc_hits.append(1 if gset & set(cands) else 0)
        re_hits.append(1 if gset & set(top3_re) else 0)
        rescued += 1 if (gset & set(cands)) and not (gset & set(top3_first)) else 0
    rec_first, rec_cand, rec_re = sum(r1_hits) / 32, sum(rc_hits) / 32, sum(re_hits) / 32
    print(f"  第一阶段 top-3 命中 gold: {rec_first:.3f} → 重排后 top-3: {rec_re:.3f}"
          f"（天花板 = 候选召回 {rec_cand:.3f}，其中 {rescued} 条被重排从候选 4-8 位提进 top-3）")
    print(f"  reranker 调用 = {calls} 次（32 查询 × 8 候选）——不是 32×800 全量精排")
    check("两阶段: 重排后 gold 命中率 ≥ 第一阶段（精排只升不降——rerank top-3 ⊆ 候选集，gold ∈ 候选 ⇒ 可提位）",
          rec_re >= rec_first)
    check(f"两阶段（实测派生）: 第一阶段 {rec_first:.3f} → 重排后 {rec_re:.3f} == 天花板 {rec_cand:.3f}"
          f"（粗召回定天花板，精排定位次；粗召回漏掉的 gold 精排救不回）",
          abs(rec_first - 0.750) < 1e-9 and abs(rec_re - 31 / 32) < 1e-9 and abs(rec_cand - 31 / 32) < 1e-9)
    check("成本边界: reranker 总调用 == 32×8 = 256（两阶段 = 便宜宽召回 + 昂贵窄精排）", calls == 256)

    # ---- [7] nDCG@k：分级相关性（lexical dense + BM25 口径，路径无关） ----
    print("\n[7] nDCG@k 分级评估（grade 2 = 正解 / grade 1 = 相关；recall 看不见的排名质量）")
    lex_ranked = {q: sorted(((did, cosine(embed_lexical(q), embed_lexical(t)))
                             for did, t in CORPUS), key=lambda p: (-p[1], p[0]))
                  for q, _ in GRADED}
    rrf_lex = {q: rrf_fuse([lex_ranked[q], bm.rank(q)], k=60) for q, _ in GRADED}
    for q, graded in GRADED:
        n_bm = ndcg_at_k([did for did, _ in bm.rank(q)], graded, 3)
        n_lex = ndcg_at_k([did for did, _ in lex_ranked[q]], graded, 3)
        n_rrf = ndcg_at_k([did for did, _ in rrf_lex[q]], graded, 3)
        r_bm = [did for did, _ in bm.rank(q)][:3]
        r_rrf = [did for did, _ in rrf_lex[q]][:3]
        g2 = [did for did, g in graded.items() if g == 2][0]
        print(f"  {q!r}: nDCG@3 BM25 = {n_bm:.3f} / lexical = {n_lex:.3f} / RRF 混合 = {n_rrf:.3f}"
              f"（gold2={g2}，BM25 top3={r_bm}，RRF top3={r_rrf}）")
    ndcg_rrf = [ndcg_at_k([did for did, _ in rrf_lex[q]], graded, 3) for q, graded in GRADED]
    ndcg_bm = [ndcg_at_k([did for did, _ in bm.rank(q)], graded, 3) for q, graded in GRADED]
    check("nDCG 有区分度: 两条查询的 RRF nDCG@3 不全相等（排名质量可测）",
          abs(ndcg_rrf[0] - ndcg_rrf[1]) > 1e-9 or ndcg_rrf[0] < 1.0)
    check("nDCG 归一: 完美排序 nDCG@3 == 1.0（构造判据自校验）",
          ndcg_at_k(sorted(GRADED[0][1], key=lambda d: -GRADED[0][1][d]), GRADED[0][1], 3) == 1.0)

    # ---- [8] 成本账本 ----
    print("\n[8] 成本账本（toy 口径；对照 Milvus Index Explained 的 1M×128d HNSW 内存账）")
    g = HNSW(M=8, ef_construction=32, tag="cost")
    docs_c, vecs_c, ids_c, _, _ = truths[800]
    for (did, _, _), v in zip(docs_c, vecs_c):
        g.add(did, v)
    links = sum(len(g.graph[i][l]) for i in range(800) for l in range(len(g.graph[i])))
    graph_bytes = links * 4
    vec_bytes = 800 * D_SYN * 4
    print(f"  N=800: 向量段 = 800×{D_SYN}×4 = {vec_bytes} B；图 = {links} 条边×4 = {graph_bytes} B"
          f"（平均每点 {links / 800:.1f} 条边）")
    print(f"  Milvus 口径参照: 1M×128d HNSW = 图 128 MB + 向量 512 MB = 640 MB（度数 32 口径，文档算例）")
    check("图字节公式: 边数×4 == 图字节", graph_bytes == links * 4)
    check("图非空且稀疏: 0 < 平均每点边数 ≤ 2M+上层（图是稀疏结构，不是全连接）",
          0 < links / 800 <= 16 + 8)

    # ---- [9] digest：纯 Python 确定性部分，两次独立构建逐位一致 ----
    def build_digest():
        docs2, queries2, _ = synth_world(800)  # 3 元返回 (docs, queries, topics)；错误的 5 元解包会直接失败
        vecs2 = [embed_syn(t) for _, t, _ in docs2]
        ids2 = [did for did, _, _ in docs2]
        qvecs2 = [(qid, embed_syn(qt), t) for qid, qt, t in queries2]
        g2 = HNSW(M=8, ef_construction=32, tag="digest")
        for (did, _, _), v in zip(docs2, vecs2):
            g2.add(did, v)
        truth2 = {qid: brute_knn(qv, vecs2, ids2, 10) for qid, qv, _ in qvecs2}
        rec2 = sum(ann_recall(g2.search(qv, 10, 16), truth2[qid], 10) for qid, qv, _ in qvecs2) / 32
        bm2 = BM25(CORPUS)
        bm2_r = {q: bm2.rank(q) for q, _ in EVAL}
        payload = {
            "bm25_recall3": round(evaluate_ranked(bm2_r, EVAL, k=3)["recall"], 6),
            "hnsw_n800_ef16_recall10": round(rec2, 6),
            "sweep": [(N, ef, round(r, 6), round(dc, 3)) for N, ef, r, dc in sweep],
            "ablation": [round(rec_h, 6), round(rec_n, 6)],
            "shard": [round(rs, 6), round(rg, 6)],
            "rerank": [round(rec_first, 6), round(rec_cand, 6), round(rec_re, 6)],
            "ndcg_rrf": [round(x, 6) for x in ndcg_rrf],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    dg1, dg2 = build_digest(), build_digest()
    print(f"\nretrieval digest (path-independent core): {dg1}  两次独立构建逐位一致: {dg1 == dg2}")
    check("确定性: digest 两次独立构建逐位一致（md5 层级 + 确定插入序 + 无 RNG 状态）", dg1 == dg2)

    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")
    print(f"  elapsed: {time.time() - t0:.1f}s（含模型加载当且仅当 real 路径；掩码口径承 L0 家族）")

if __name__ == "__main__":
    main()
