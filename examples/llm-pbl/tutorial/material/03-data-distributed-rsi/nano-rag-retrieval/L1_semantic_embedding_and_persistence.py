#!/usr/bin/env python3
"""nano-rag-retrieval L1 — 真实神经 embedding + 索引持久化 + 增量 add（自 L0 的 K+1）。

L0 裸出了 lexical embedding 的天花板（同义词盲 cos=0.0000 / 词序盲 cos=1.0000 / 碰撞噪声）
与两条刻意设计的 MISS 查询。L1 换真实小神经模型，在**同一语料、同一评估集**上量化突破，
并把索引从「内存派生物」升级为「落盘契约」：

  [1] embedding 换真实模型：sentence-transformers/all-MiniLM-L6-v2（384 维、22.7M 参数、CPU 秒级）。
      验收靶点 = L0 的两条纯同义词 MISS（automobile↔car / bonuses↔rewards）被检索回来；
  [2] 索引文件级持久化：meta（模型身份 + dim + 文档清单）与 vec（float32 向量段）两件分离。
      模型身份是索引的一部分——「换模型不重建索引」是生产事故（embedding drift），加载时拒绝；
  [3] 增量 add：只 embed 新文档（O(new) 模型调用，不是 O(N) 重建）、幂等（已收录 id 跳过）、
      旧向量逐字节不变——与 nano-data-platform L1 的水位线增量同步是同一治理思想；
  [4] 双路径（可运行性契约）：无依赖/无模型快照 → 确定性 fallback——
      手写同义词簇 + 词 bigram 的「最小语义机制」模拟，显式 mock（其 1.0 recall 来自手写表
      对本评估集的定向覆盖，不是学到的泛化，数字不与真实模型比高低）。fallback 零依赖。

运行：python3 L1_semantic_embedding_and_persistence.py          （默认离线：快照已钉住，零网络）
首次取模型快照（一次性）：NANO_RAG_ALLOW_DOWNLOAD=1 python3 L1_semantic_embedding_and_persistence.py
保留索引：NANO_RAG_INDEX_DIR=/path/to/index python3 L1_semantic_embedding_and_persistence.py
          （默认在临时目录演示持久化并自动清理，不污染当前工作目录。）
依赖：真实路径 = sentence-transformers + torch（CPU）；
      fallback 路径零依赖（纯标准库）。
确定性：同一模型快照 + CPU fp32 推理逐位确定 + md5 无随机；elapsed 行可掩码（口径承 L0 家族：
      sed '/^[[:space:]]*elapsed/d'）。真实路径 digest 与 fallback 路径 digest 不同（函数不同），
      各自内部两次独立构建逐位一致。
"""
import array, hashlib, json, math, os, re, sys, tempfile, time

# 环境钉死在 import 前：默认离线 = 快照钉住（索引构建绝不隐式拉新快照——可复现性是索引的一等公民，
# 承 L0 §3(d)）。首次取快照走 NANO_RAG_ALLOW_DOWNLOAD=1（见 docstring）。
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

# ---- fixture：与 L0 逐字同一（同一语料、同一评估集——突破必须有参照系）----
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
# 增量 add 的新文档（L0 语料之外的主题：缓存淘汰 / 向量时钟）
NEW_DOCS = [
    ("d15", "the cache evicts least recently used entries first when memory reaches the high watermark"),
    ("d16", "vector clocks stamp replica events so conflicts can be detected and merged at read time"),
]
NEW_EVAL = [("how does an LRU cache decide what to evict", "d15"),
            ("detecting conflicts between replicas with vector clocks", "d16")]

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def _h(s):  # 承 L0：md5 而非内建 hash()（PYTHONHASHSEED 随机盐会毁掉可复现性）
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

def _f32(v):  # 统一量化到 float32：内存 == 落盘 == 重载逐位同一（两条路径同口径）
    return list(array.array("f", v))

# ---- [0a] lexical embedding：L0 实现逐字搬运——同集对照的基线必须是同一个函数 ----
D_LEX = 256
def embed_lexical(text):
    v = [0.0] * D_LEX
    for tok in tokenize(text):
        v[_h("w:" + tok) % D_LEX] += 1.0
        pad = f"#{tok}#"
        for i in range(len(pad) - 2):
            v[_h("g:" + pad[i:i + 3]) % D_LEX] += 0.25
    n = math.sqrt(sum(x * x for x in v))
    return _f32([x / n for x in v] if n else v)

# ---- [0b] fallback：显式 mock——「语义 = 共享潜在特征」的最小手工机制（零依赖一键跑通用）----
# 同义词簇表 = 手写的微型知识：簇内词共享一个额外特征维（权重 2.0），这就是神经 embedding
# 「不同表面形式 → 相近坐标」的最小机制。词 bigram（权重 0.5）给出词序敏感性。
# 诚实声明：簇表定向覆盖了本评估集的同义词对——它的 1.0 recall 是「开卷」，不是泛化。
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
            v[_h(f"c:{c}") % D_FB] += 2.0          # 共享潜在特征 = 最小「语义」机制
        pad = f"#{tok}#"
        for i in range(len(pad) - 2):
            v[_h("g:" + pad[i:i + 3]) % D_FB] += 0.25
    for a, b in zip(toks, toks[1:]):
        v[_h("b:" + a + " " + b) % D_FB] += 0.5     # bigram = 词序敏感
    n = math.sqrt(sum(x * x for x in v))
    return _f32([x / n for x in v] if n else v)

# ---- [0c] 真实路径：sentence-transformers/all-MiniLM-L6-v2 ----
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

def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))

# ---- [2] 持久化索引：meta（身份契约）+ vec（float32 向量段）两件分离 ----
META_FILE = "nano_index.meta.json"
VEC_FILE = "nano_index.vec.f32"
FORMAT = "nano-rag-index/v1"

class IndexModelError(Exception):
    pass

class PersistentFlatIndex:
    """flat 暴力精确 kNN（承 L0）+ 文件级持久化 + 幂等增量 add。

    身份契约：向量只有相对产出它的模型才有意义。meta 钉住 model/snapshot/dim，
    加载时与当前 embedder 比对——不符即拒（embedding drift 治理，§6）。
    """
    def __init__(self, embed, dim, model_name, snapshot):
        self.embed, self.dim, self.model_name, self.snapshot = embed, dim, model_name, snapshot
        self.docs = []            # [(doc_id, text)]
        self.vecs = []            # [float32 向量]，与 docs 同序
        self.embed_calls = 0      # 模型调用计数：增量成本账（§7）

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

    def search(self, query, k=3, tau=0.0):
        qv = self.embed(query)
        ranked = sorted(((cosine(qv, v), did) for (did, _), v in zip(self.docs, self.vecs)), reverse=True)
        hits = [(did, round(s, 4)) for s, did in ranked[:k] if s >= tau]
        return hits, len(self.docs) * self.dim   # 扫描成本 N×D（toy 口径，承 L0 §7）

    def vec_bytes(self):
        out = array.array("f")
        for v in self.vecs: out.extend(v)
        return out.tobytes()

    def save(self):
        meta = {"format": FORMAT, "model": self.model_name, "snapshot": self.snapshot,
                "dim": self.dim, "dtype": "float32", "n_docs": len(self.docs),
                "docs": [{"id": did, "text": text} for did, text in self.docs]}
        with open(META_FILE, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
        with open(VEC_FILE, "wb") as f:
            f.write(self.vec_bytes())

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
        raw = open(VEC_FILE, "rb").read()
        n = meta["n_docs"]
        if len(raw) != n * dim * 4:
            raise IndexModelError(f"vec size mismatch: {len(raw)} B != {n}×{dim}×4 = {n * dim * 4} B")
        idx = cls(embed, dim, model_name, meta.get("snapshot"))
        idx.docs = [(d["id"], d["text"]) for d in meta["docs"]]
        vals = array.array("f"); vals.frombytes(raw)
        idx.vecs = [list(vals[i * dim:(i + 1) * dim]) for i in range(n)]
        return idx

# ---- [3] 检索评估：承 L0（recall@k / precision@k / MRR，单 relevant 口径）----
def evaluate(index, eval_set, k):
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

def run_demo(index_dir):
    global META_FILE, VEC_FILE
    os.makedirs(index_dir, exist_ok=True)
    META_FILE = os.path.join(index_dir, "nano_index.meta.json")
    VEC_FILE = os.path.join(index_dir, "nano_index.vec.f32")
    t0 = time.time()
    print("== nano-rag-retrieval L1: 真实神经 embedding + 索引持久化 + 增量 add（自 L0 的 K+1）==")

    # ---- [0] 路径判定：真实模型优先，缺依赖/缺快照 → 显式 fallback ----
    try:
        embed, dim, snapshot = load_real_embedder()
        path, model_name = "real", MODEL_ID
        print(f"\n[0] embedding 路径 = real（{MODEL_ID}；snapshot {snapshot[:12]}…，dim {dim}，离线快照钉住）")
    except Exception as e:
        embed, dim, snapshot = embed_fallback, D_FB, "deterministic-handcoded"
        path, model_name = "fallback", "fallback-synonym-cluster-v1"
        print(f"\n[0] embedding 路径 = fallback（显式 mock：手写同义词簇 + bigram，dim {dim}，零依赖）")
        print(f"    原因：{type(e).__name__}（真实模型需 sentence-transformers + 本地快照；"
              f"首次运行见 docstring 的 NANO_RAG_ALLOW_DOWNLOAD=1）")
    # 两条路径的期望值分开声明——机制检查同一套，数值口径各归各（反幻觉：不混口径）
    if path == "real":
        EXP = dict(cos_car=(0.7, 1.01), cos_bonus=(0.5, 1.01), cos_order=(0.5, 1.0 - 1e-9),
                   cos_morph=(0.3, 1.01), recall1=0.875, recall3=1.0, mrr=0.9375,
                   rank_auto=2, ood_max=0.25, index_bytes=14 * dim * 4)
    else:
        EXP = dict(cos_car=(0.5, 1.01), cos_bonus=(0.5, 1.01), cos_order=(0.5, 1.0 - 1e-9),
                   cos_morph=(0.1, 0.5), recall1=1.0, recall3=1.0, mrr=1.0,
                   rank_auto=1, ood_max=0.25, index_bytes=14 * dim * 4)

    # ---- [1] lexical 基线复刻（L0 算法逐字搬运）：同集对照的锚 ----
    print("\n[1] lexical 基线复刻（L0 算法逐字同一）——突破的参照系")
    lex = PersistentFlatIndex(embed_lexical, D_LEX, "lexical-l0-copy", "n/a")
    lex.add(CORPUS)
    ev1, ev3 = evaluate(lex, EVAL, k=1), evaluate(lex, EVAL, k=3)
    print(f"  recall@1 = {ev1['recall']:.3f}   recall@3 = {ev3['recall']:.3f}   MRR = {ev3['mrr']:.4f}")
    miss = sorted(g for _, g, r, _ in ev3["per_q"] if r == 0)
    print(f"  MISS = {miss}（L0 刻意设计的两条纯同义词查询 = L1 的验收靶点）")
    check("lexical 基线与 L0 逐位一致: recall@1 0.625 / recall@3 0.750 / MRR 0.6875",
          abs(ev1["recall"] - 0.625) < 1e-9 and abs(ev3["recall"] - 0.75) < 1e-9 and abs(ev3["mrr"] - 0.6875) < 1e-9)
    check("lexical MISS 集合 == [d10, d14]", miss == ["d10", "d14"])

    # ---- [2] embedding 几何：L0 的四个失败点，现在逐一复测 ----
    print("\n[2] embedding 几何修复验证（括号为 L0 lexical 录值，对照读）")
    s_car = cosine(embed("car"), embed("automobile"))
    s_bonus = cosine(embed("bonuses"), embed("rewards"))
    s_order = cosine(embed("dog bites man"), embed("man bites dog"))
    s_morph = cosine(embed("retrieve"), embed("retrieval"))
    print(f"  同义词 car/automobile      cos = {s_car:.4f}   (L0: 0.0000 全盲)")
    print(f"  同义词 bonuses/rewards     cos = {s_bonus:.4f}   (L0: 0.0000 全盲)")
    print(f"  词序 dog-man/man-bites-dog cos = {s_order:.4f}   (L0: 1.0000 全盲)")
    print(f"  词形 retrieve/retrieval    cos = {s_morph:.4f}   (L0: 0.2449 trigram 桥)")
    check(f"同义词修复 car/automobile: cos ∈ {EXP['cos_car']}", EXP["cos_car"][0] < s_car < EXP["cos_car"][1])
    check(f"同义词修复 bonuses/rewards: cos ∈ {EXP['cos_bonus']}", EXP["cos_bonus"][0] < s_bonus < EXP["cos_bonus"][1])
    check("词序可见: 0.5 < cos < 1.0（严格小于 1 = 向量不再相同）", EXP["cos_order"][0] < s_order < EXP["cos_order"][1])
    check(f"词形桥仍在: cos(retrieve, retrieval) ∈ {EXP['cos_morph']}", EXP["cos_morph"][0] < s_morph < EXP["cos_morph"][1])

    # ---- [3] 同一语料 + 同一评估集：lexical → 语义的 recall 突破 ----
    print(f"\n[3] 检索评估（同一 14 篇语料 + 同一 8 条标注查询，路径 = {path}）")
    idx = PersistentFlatIndex(embed, dim, model_name, snapshot)
    added, skipped = idx.add(CORPUS)
    print(f"  建索引: {added} 篇 embed（模型调用 {idx.embed_calls} 次），跳过 {skipped}")
    ev1p, ev3p = evaluate(idx, EVAL, k=1), evaluate(idx, EVAL, k=3)
    print(f"  recall@1 = {ev1p['recall']:.3f}（lexical 0.625）  recall@3 = {ev3p['recall']:.3f}（lexical 0.750）  "
          f"MRR = {ev3p['mrr']:.4f}（lexical 0.6875）")
    ranks = {q: r for q, _, r, _ in ev3p["per_q"]}
    for q, gold, r, top in ev3p["per_q"]:
        flag = f"rank {r}" if r else f"MISS (top1 = {top[0]}@{top[1]:.4f})"
        print(f"    {q!r:44s} gold={gold}  {flag}")
    check(f"recall@1 == {EXP['recall1']}", abs(ev1p["recall"] - EXP["recall1"]) < 1e-9)
    check("验收靶点: recall@3 == 1.0（L0 两条 MISS 全部进入 top-3）", abs(ev3p["recall"] - 1.0) < 1e-9)
    check(f"MRR == {EXP['mrr']}", abs(ev3p["mrr"] - EXP["mrr"]) < 1e-9)
    check("MISS 修复一: 'bonuses for correct behavior' rank == 1", ranks["bonuses for correct behavior"] == 1)
    check(f"MISS 修复二: 'automobile fuel economy' rank == {EXP['rank_auto']}（进入 top-2）",
          ranks["automobile fuel economy"] == EXP["rank_auto"])
    r_ch = ranks["i was charged two times"]
    check("'i was charged two times' rank == 1（L0: 被碰撞噪声压到 rank 2）", r_ch == 1)

    # ---- [4] 持久化：meta + vec 两件落盘 → 清空内存 → 重载逐位复验 ----
    print("\n[4] 索引持久化（meta = 身份契约，vec = float32 向量段）")
    idx.save()
    meta_size, vec_size = os.path.getsize(META_FILE), os.path.getsize(VEC_FILE)
    print(f"  {os.path.basename(META_FILE)}: {meta_size} B（model/snapshot/dim/文档清单）")
    print(f"  {os.path.basename(VEC_FILE)}: {vec_size} B = 14×{dim}×4(float32)（L0 的 float64 256 维 = 28672 B）")
    check(f"向量段落盘字节 == 14×{dim}×4 = {EXP['index_bytes']}", vec_size == EXP["index_bytes"])
    vecs_before = idx.vec_bytes()
    hits_before = [idx.search(q, k=3)[0] for q, _ in EVAL]
    del idx  # 模拟进程退出：内存态蒸发，只剩盘上两件
    idx2 = PersistentFlatIndex.load(embed, dim, model_name)
    check("重载后向量逐字节一致（float32 往返无损）", idx2.vec_bytes() == vecs_before)
    hits_after = [idx2.search(q, k=3)[0] for q, _ in EVAL]
    check("重载后检索结果逐位一致（8 条查询 top-3 全同）", hits_after == hits_before)
    meta = json.load(open(META_FILE))
    check("meta 钉住身份: model/dim/n_docs 与当前 embedder 一致",
          meta["model"] == model_name and meta["dim"] == dim and meta["n_docs"] == 14)
    print(f"  身份契约: model={meta['model']}  snapshot={str(meta['snapshot'])[:12]}…  dim={meta['dim']}")

    # ---- [5] 身份契约的牙齿：换模型不重建索引 → 加载拒绝 ----
    print("\n[5] embedding drift 治理：索引与模型是绑定契约")
    original_meta = open(META_FILE).read()
    tampered = json.loads(original_meta)
    tampered["model"] = "some-other-model-v9"
    with open(META_FILE, "w") as f:
        json.dump(tampered, f)
    try:
        PersistentFlatIndex.load(embed, dim, model_name)
        rejected = False
    except IndexModelError as e:
        rejected = True
        print(f"  篡改 meta.model 后加载 → IndexModelError: {e}")
    finally:
        with open(META_FILE, "w") as f:
            f.write(original_meta)  # 恢复真身
    check("模型身份不符 → 拒绝加载（宁可报错，不用旧坐标查新空间）", rejected)

    # ---- [6] 增量 add：只 embed 新文档，旧向量逐字节不动 ----
    print("\n[6] 增量 add（d15/d16：L0 语料之外的新主题）")
    calls_before = idx2.embed_calls  # load 出的索引从 0 计——增量成本按本次 add 的调用数算
    added2, skipped2 = idx2.add(NEW_DOCS)
    added3, skipped3 = idx2.add(NEW_DOCS)  # 幂等复投：同一批再 add 一次
    print(f"  add(NEW_DOCS): 新增 {added2} / 跳过 {skipped2}；重复 add: 新增 {added3} / 跳过 {skipped3}（幂等）")
    delta = idx2.embed_calls - calls_before
    full = len(idx2.docs)
    print(f"  模型调用增量 = {delta} 次（全量重建 {full} 篇需 {full} 次——增量省 {full - delta} 次）")
    check("增量 add 只 embed 新文档: 模型调用增量 == 2", delta == 2)
    check("幂等: 重复 add 零新增", added3 == 0 and skipped3 == 2)
    vb2 = idx2.vec_bytes()
    check("旧 14 篇向量逐字节不变（前缀比对）", vb2[:len(vecs_before)] == vecs_before)
    idx2.save()
    ev_new = evaluate(idx2, NEW_EVAL, k=3)
    for q, gold, r, top in ev_new["per_q"]:
        print(f"    {q!r:50s} gold={gold}  rank {r}")
    check("新文档可检索: d15/d16 均 rank 1（16 篇索引）",
          all(r == 1 for _, _, r, _ in ev_new["per_q"]))
    ev3_after = evaluate(idx2, EVAL, k=3)
    check("增量不扰旧查询: 原 8 条 recall@3 仍 == 1.0", abs(ev3_after["recall"] - 1.0) < 1e-9)

    # ---- [7] 阈值 revisited：神经空间的噪声底更干净，但灰区仍在 ----
    print("\n[7] 阈值 revisited（承 L0 §6：检索器要会说『不知道』）")
    ood_raw, _ = idx2.search("volcano eruption lava", k=3)
    ood, _ = idx2.search("volcano eruption lava", k=3, tau=0.25)
    print(f"  域外 'volcano eruption lava' top1 = {ood_raw[0][0]}@{ood_raw[0][1]:.4f} → tau=0.25: "
          f"{ood if ood else '空手而归'}")
    check("域外查询 top1 分数 < 0.25 且被阈值挡住", ood_raw[0][1] < EXP["ood_max"] and ood == [])
    gray, _ = idx2.search("automobile fuel economy", k=2, tau=0.25)
    print(f"  灰区 'automobile fuel economy'（tau=0.25）: {gray if gray else '连同 gold 一起被滤掉'}"
          f"——阈值工作点是业务决策（tutorial §8）")

    # ---- [8] 成本账本 ----
    print("\n[8] 成本账本（toy 口径，承 L0 §7）")
    n, d = len(idx2.docs), idx2.dim
    print(f"  存储: {n} 篇向量 = {n}×{d}×4 = {n * d * 4} B + meta/原文（payload 分离，L2 对照 Milvus segment）")
    print(f"  扫描: N×D = {n * d} 次乘加/query（L0: 14×256 = 3584）——暴力不 scale，ANN 是 L2 主题")
    print(f"  embed: 一次性建索引成本（全量 {n} 次模型调用）vs 每查询只 embed 查询一句——索引是摊销结构")
    check(f"索引字节公式: {n}×{d}×4 == {n * d * 4}", os.path.getsize(VEC_FILE) == n * d * 4)

    # ---- [9] 确定性 digest：两次独立构建（重新 encode 全部文档）逐位一致 ----
    idx_b = PersistentFlatIndex(embed, dim, model_name, snapshot)
    idx_b.add(CORPUS)
    ev_b = evaluate(idx_b, EVAL, k=3)
    body = lambda ev: json.dumps({"path": path, "per_q": ev["per_q"], "mrr": round(ev["mrr"], 6)},
                                 sort_keys=True, ensure_ascii=False).encode()
    dg, dg2 = hashlib.sha256(body(ev3p)).hexdigest()[:16], hashlib.sha256(body(ev_b)).hexdigest()[:16]
    print(f"\nretrieval digest ({path}): {dg}  两次独立构建逐位一致: {dg == dg2}")
    check("确定性：两次独立构建索引 + 评估，digest 逐位一致", dg == dg2)

    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")
    print(f"  elapsed: {time.time() - t0:.1f}s（含模型加载；掩码口径承 L0 家族）")

def main():
    requested = os.environ.get("NANO_RAG_INDEX_DIR")
    if requested:
        run_demo(os.path.abspath(requested))
    else:
        with tempfile.TemporaryDirectory(prefix="nano-rag-index-") as index_dir:
            run_demo(index_dir)


if __name__ == "__main__":
    main()
