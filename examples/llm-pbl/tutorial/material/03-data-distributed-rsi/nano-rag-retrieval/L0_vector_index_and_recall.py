#!/usr/bin/env python3
"""nano-rag-retrieval L0 — embedding 索引 + 向量相似度检索 + 检索评估，纯 Python 本质模拟。

它在模拟真实系统的哪一面（L0 验收标准，ROADMAP §二）：
  [1] embedding = 把文本映射进几何空间的函数：相似度 ≈ 距离近；函数的质量决定检索的天花板。
      L0 用 lexical embedding（词哈希 + 字符 trigram，feature hashing [0902.2206]），并让它的失败模式
      （同义词盲 / 词序盲 / 哈希碰撞假相似）可见——这正是神经语义 embedding 存在的动机（L1 接真实模型）；
  [2] flat index = 暴力精确 kNN：每次查询扫描全部向量，归一化后 cosine = 点积。它是所有 ANN 索引
      近似的精确基线（Milvus 文档直接称 FLAT 为 Brute-Force；HNSW [1603.09320] 用可控的 recall 换速度）；
  [3] 检索评估：recall@k / precision@k / MRR——没有标注评估，就不知道检索到底有没有在工作；
  [4] 治理 first-class：成本账本（索引字节 / 每查询扫描 op）+ 分数阈值（「不知道」好过把垃圾喂给生成器）。
刻意不模拟：真实神经 embedding（L1）、ANN 索引 / 混合检索 / 重排序（L2 对照 Milvus/OpenSearch/Weaviate 源码）。
零依赖（纯标准库），CPU 秒级；输出确定（md5 哈希、无随机 / 无 wall-clock），复跑逐字节一致。
"""
import hashlib, json, math, re

CHECKS = []
def check(name, cond):
    CHECKS.append(bool(cond))
    print(f"  [check {len(CHECKS):02d}] {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: raise SystemExit("self-check failed: " + name)

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

# ---- demo fixture：迷你知识库（LLM 系统世界的自指文档）。实验设计显式声明见 tutorial §2 ----
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
# 标注评估集：(query, gold_doc_id)。两条 MISS 是刻意设计的纯同义词鸿沟（automobile↔car / bonuses↔rewards，零词形桥）
EVAL = [("reset my password please", "d01"),
        ("i was charged two times", "d02"),
        ("nearest neighbor query over embeddings", "d03"),
        ("how to resume training after a crash", "d09"),
        ("reduce hallucination with fetched documents", "d11"),
        ("remove near duplicate documents before training", "d12"),
        ("automobile fuel economy", "d14"),
        ("bonuses for correct behavior", "d10")]

def main():
    print("== nano-rag-retrieval L0: embedding 索引 + 向量相似度检索 + 检索评估（纯 Python 本质模拟）==")
    idx = FlatIndex()
    for did, text in CORPUS: idx.add(did, text)
    raw_b = sum(len(t) for _, t in CORPUS)
    print(f"\n[1] 索引构建：{len(idx.docs)} 篇文档 → 每篇一个 {D} 维归一化向量（lexical embedding：词 + 字符 trigram）")
    print(f"  index bytes = N×D×8(float64) = {len(idx.docs) * D * 8} B（另需原文 {raw_b} B 供展示/重排）")
    check("索引尺寸公式: 14×256×8 = 28672 B", len(idx.docs) * D * 8 == 28672)
    print("\n[2] embedding 几何：相似度 = 距离近。lexical embedding 有一条可见的『光谱』")
    s_exact = cosine(embed("vector index nearest neighbor"), embed("vector index nearest neighbor"))
    s_morph = cosine(embed("retrieve"), embed("retrieval"))
    s_syn = cosine(embed("car"), embed("automobile"))
    s_coll = cosine(embed("gpu"), embed("vram"))
    print(f"  完全一致   cosine = {s_exact:.4f}")
    print(f"  词形变化   cosine = {s_morph:.4f}   (retrieve vs retrieval：trigram 部分救回)")
    print(f"  同义词     cosine = {s_syn:.4f}    (car vs automobile：lexical 空间全盲)")
    print(f"  哈希碰撞   cosine = {s_coll:.4f}   (gpu vs vram：无关词撞到同一维，假相似)")
    check("完全一致 = 1.0", abs(s_exact - 1.0) < 1e-9)
    check("词形变化部分相似 (0.1 < cos < 0.5)", 0.1 < s_morph < 0.5)
    check("同义词恰好正交 (cos == 0.0)", s_syn == 0.0)
    check("哈希碰撞产生假相似 (0.1 < cos < 0.4)——噪声与盲点是 lexical embedding 的一体两面", 0.1 < s_coll < 0.4)
    print("\n[3] 失败模式一：词序盲——BoW 向量只依赖词的多重集合，与顺序无关")
    s_order = cosine(embed("dog bites man"), embed("man bites dog"))
    print(f"  'dog bites man' vs 'man bites dog'  cosine = {s_order:.4f}")
    check("词序盲: cos == 1.0（两句话向量完全相同）", abs(s_order - 1.0) < 1e-9)
    print("\n[4] top-k 检索：暴力精确 kNN，每次查询扫描全部向量")
    hits, ops = idx.search("how to reset password", k=3)
    for did, s in hits: print(f"  {did}  {s:.4f}")
    check("精确词查询命中 d01 且 rank 1", hits[0][0] == "d01")
    print(f"  扫描成本 = N×D = {ops} 次乘加（精确不是免费的）")
    print("\n[5] 检索评估：recall@k / precision@k / MRR（8 条标注查询，每条 1 个 gold）")
    ev1, ev3 = evaluate(idx, EVAL, k=1), evaluate(idx, EVAL, k=3)
    print(f"  recall@1 = {ev1['recall']:.3f}   recall@3 = {ev3['recall']:.3f}   MRR = {ev3['mrr']:.3f}")
    for q, gold, r, top in ev3["per_q"]:
        flag = f"rank {r}" if r else f"MISS (top1 = {top[0]}@{top[1]:.4f})"
        print(f"    {q!r:44s} gold={gold}  {flag}")
    check("lexical 基线: recall@1 = 0.625 / recall@3 = 0.750 / MRR = 0.6875",
          abs(ev1["recall"] - 0.625) < 1e-9 and abs(ev3["recall"] - 0.75) < 1e-9 and abs(ev3["mrr"] - 0.6875) < 1e-9)
    b_hits, _ = idx.search("i was charged two times", k=2)
    check("弱信号查询: gold(d02) 被碰撞噪声压到 rank 2，与 top1 差距 < 0.02",
          b_hits[0][0] == "d01" and b_hits[1][0] == "d02" and b_hits[0][1] - b_hits[1][1] < 0.02)
    miss = [(q, g) for q, g, r, _ in ev3["per_q"] if r == 0]
    check("两条 MISS 恰是刻意设计的纯同义词查询", sorted(g for _, g in miss) == ["d10", "d14"])
    print("\n[6] 失败模式二：域外查询——阈值让检索器会说『不知道』")
    ood_raw, _ = idx.search("volcano eruption lava", k=3)
    ood, _ = idx.search("volcano eruption lava", k=3, tau=0.15)
    print(f"  'volcano eruption lava' 无阈值 top1 = {ood_raw[0][0]}@{ood_raw[0][1]:.4f}（噪声级分数，语料里根本没有火山）")
    print(f"  tau=0.15 → {ood if ood else '空手而归'}（toy 尺度策略；噪声底随查询漂移，见 tutorial §6）")
    check("阈值之下宁可空手：不把垃圾喂给生成器", ood == [] and ood_raw[0][1] < 0.15)
    print("\n[7] 成本账本（toy 口径：乘加 op 数，非真实 benchmark）：暴力不 scale → ANN 的动机")
    print(f"  N = {len(idx.docs):>2}: {ops:>5} ops/query；N = 10^6: {10**6 * D / 1e8:.2f}×10^8 ops/query")
    print("  ANN（HNSW 等）= 用可控 recall 损失换延迟 [1603.09320]——recall/latency 权衡是 L2 主题")
    check("扫描成本随 N 线性增长", ops == len(idx.docs) * D)
    idx2 = FlatIndex()
    for did, text in CORPUS: idx2.add(did, text)
    ev3b = evaluate(idx2, EVAL, k=3)
    body = lambda ev: json.dumps({"per_q": ev["per_q"], "mrr": round(ev["mrr"], 6)}, sort_keys=True, ensure_ascii=False).encode()
    dg, dg2 = hashlib.sha256(body(ev3)).hexdigest()[:16], hashlib.sha256(body(ev3b)).hexdigest()[:16]
    print(f"\nretrieval digest: {dg}  两次独立构建逐位一致: {dg == dg2}")
    check("确定性：两次独立构建索引 + 评估，digest 逐位一致", dg == dg2)
    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")

if __name__ == "__main__":
    main()
