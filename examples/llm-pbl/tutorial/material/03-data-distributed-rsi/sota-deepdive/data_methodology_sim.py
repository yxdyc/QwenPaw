#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_methodology_sim.py — 03 轨 sota-deepdive（LLM 数据方法论）的可运行本质模拟。

这是什么
========
把四个数据方法论机制面在 toy 尺度下「跑出来」，每个机制面演示一件本质的事：

  [A] 去重（exact + MinHash near-dup + LSH 分带 + 传递聚类）
      —— 为什么精确哈希抓不到「改了几个字的近重复」，MinHash 如何用签名估计 Jaccard、
         LSH 分带如何在 O(1) 候选里命中「≥75% 相似」的文档对（FineWeb 口径）。
  [B] 质量过滤（rule-based 启发式 + 质量分数阈值）
      —— 为什么「以标点结尾的行占比 <=0.12」这类廉价启发式能把低质文档分布切开
         （FineWeb Fig 8 的分布判别），以及阈值如何决定保留率（DCLM fastText 选子集）。
  [C] 数据配比 / 域重加权（DoReMi 的 Group DRO 本质）
      —— 为什么「用一个小 proxy 模型按域损失做乘性权重更新（minimax）」能产出一组
         域权重：收敛时各域损失近似拉平（minimax 不动点），最难域拿到最多预算。
         minimax 优化的是**最坏域**而非平均损失：toy 的逐域独立损失曲线没有跨域迁移，
         论文 8B 尺度的 2.6x fewer steps 是经验结果，toy 只演示机制、不外推（见 [C3b]）。
         nano 侧无实测锚，本机制面以本 sim 为可运行锚点；真实 280M→8B 见 DoReMi 论文。
  [D] 去污染 / 污染检测（n-gram 重叠）
      —— 为什么「训练文档与 benchmark 的 n-gram 重叠率」能把「背题」的文档挑出来，
         而干净文档重叠率趋 0（Lee et al. / DCLM 去污染工具的本质）。

可运行性契约（ROADMAP §三）
==========================
- 本文件是**本质模拟**：[C] 数据配比 与 [D] 去污染 在 nano 侧没有现成实测锚，
  本 sim 用 toy 尺度 + 真实算法语义（真实 MinHash/LSH、真实乘性权重重加权、真实 n-gram 重叠）
  演示机制；[A]/[B] 的真实系统实测锚由 nano-data-juicer / nano-ray 提供（deepdive §1/§2 交叉引用）。
- 纯标准库（hashlib / math / random），零外部依赖，CPU 秒级。
- seed 固定，跨运行逐字节一致；无计时行。
- 真实生产规模（15T token / 280M-8B 模型 / H100 小时）标 [TODO: verify on real system]。

运行
====
    python3 -B data_methodology_sim.py
"""

import hashlib
import math
import random

SEED = 3
DIGEST_PARTS = []          # 收集关键指标，末尾做 md5 digest（跨运行不变性锚）
CHECKS = []                # (name, ok) 自检清单


def record(key, val):
    DIGEST_PARTS.append(f"{key}={val}")


def check(name, ok):
    CHECKS.append((name, bool(ok)))


def hbytes(s: str) -> bytes:
    return hashlib.md5(s.encode("utf-8")).digest()


# ----------------------------------------------------------------------------
# [A] 去重：exact + MinHash near-dup + LSH 分带 + 传递聚类
# ----------------------------------------------------------------------------

def shingles(text: str, k: int = 3):
    """词级 k-shingle 集合（小写、去首尾空白）。"""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash_sig(sh, num_hashes: int):
    """确定性 MinHash 签名：第 i 个哈希 = min over shingle of md5(i:shingle)。
    用 hashlib 而非内置 hash()，规避 PYTHONHASHSEED 随机化，保证跨运行一致。"""
    sig = []
    for i in range(num_hashes):
        m = min(int.from_bytes(hbytes(f"{i}:{s}")[:8], "big") for s in sh)
        sig.append(m)
    return sig


def jac_true(a, b):
    ia, ib = set(a), set(b)
    if not ia and not ib:
        return 1.0
    return len(ia & ib) / len(ia | ib)


def jac_minhash(sa, sb):
    return sum(1 for x, y in zip(sa, sb) if x == y) / len(sa)


def run_A():
    print("=" * 72)
    print("[A] Deduplication: exact hash + MinHash/LSH near-dup (FineWeb 口径)")
    print("=" * 72)

    base = ("the quick brown fox jumps over the lazy dog near the river bank "
            "while the sun sets behind the hills and the wind blows gently")
    exact_dup = base                                        # 完全重复
    near_dup = base.replace("quick brown", "fast brown").replace("river", "creek")  # 改几个字
    unrelated = ("a completely different document about distributed systems "
                 "and consensus protocols such as raft and paxos in clusters")

    docs = {"d0_base": base, "d1_exact": exact_dup, "d2_near": near_dup, "d3_unrel": unrelated}
    sh = {k: shingles(v) for k, v in docs.items()}

    # --- 精确哈希去重：只能抓逐字节相同 ---
    seen, kept_exact, removed_exact = {}, [], []
    for k in docs:
        h = hashlib.sha1(docs[k].encode()).hexdigest()
        if h in seen:
            removed_exact.append((k, seen[h]))
        else:
            seen[h] = k
            kept_exact.append(k)
    print(f"[A1] exact-hash dedup: 4 docs -> keep {len(kept_exact)} {kept_exact}")
    print(f"     removed (逐字节重复): {removed_exact}")
    print(f"     near-dup '{'d2_near' if 'd2_near' in kept_exact else ''}' 仍在? "
          f"{'是 ← 精确哈希抓不到改字近重复' if 'd2_near' in kept_exact else '否'}")
    check("A1 exact dedup 抓 d1_exact", ("d1_exact", "d0_base") in removed_exact)
    check("A1 exact dedup 漏 d2_near", "d2_near" in kept_exact)

    # --- MinHash 签名估计 Jaccard ---
    NUM_HASHES = 64
    sig = {k: minhash_sig(sh[k], NUM_HASHES) for k in docs}
    pairs = [("d0_base", "d1_exact"), ("d0_base", "d2_near"), ("d0_base", "d3_unrel")]
    print(f"\n[A2] MinHash 估计 Jaccard (num_hashes={NUM_HASHES}) vs 真值:")
    est_near = None
    for a, b in pairs:
        jt = jac_true(sh[a], sh[b])
        jm = jac_minhash(sig[a], sig[b])
        tag = ""
        if (a, b) == ("d0_base", "d2_near"):
            est_near = jm
            tag = "  <- near-dup"
        print(f"     {a:>10} vs {b:<10}: true={jt:.3f}  minhash={jm:.3f}  |Δ|={abs(jt-jm):.3f}{tag}")
    check("A2 exact 对 Jaccard=1", abs(jac_true(sh['d0_base'], sh['d1_exact']) - 1.0) < 1e-9)
    check("A2 MinHash 估计误差 <0.15 (near 对)", est_near is not None and
          abs(est_near - jac_true(sh['d0_base'], sh['d2_near'])) < 0.15)
    check("A2 unrelated 对 Jaccard 低 (<0.2)", jac_true(sh['d0_base'], sh['d3_unrel']) < 0.2)
    record("A_near_true_jac", round(jac_true(sh['d0_base'], sh['d2_near']), 4))
    record("A_near_minhash_jac", round(est_near, 4))

    # --- LSH 分带的 S 曲线：把「相似度」放大成「是否同桶」的软阈值（FineWeb: ≥75% 相似）---
    # 构造受控 Jaccard 的文档对：base 集 + 添加 m 个两两不相交的新 shingle => J = n/(n+m)。
    # 对每个 J 用 40 个独立变体统计「成为候选」的命中率，展示 S 曲线在拐点
    # s* = (1/b)^(1/r) ≈ 0.77 附近陡升——这正是 FineWeb「targeting ≥75% similar」的机制。
    n_base = 60
    base_sh = {f"w{i}" for i in range(n_base)}
    BANDS, ROWS = 8, 8                       # 64 hashes = 8 bands × 8 rows
    s_star = (1.0 / BANDS) ** (1.0 / ROWS)   # S 曲线拐点 ≈0.77
    print(f"\n[A3] LSH S 曲线 ({BANDS} bands × {ROWS} rows, 拐点 s*≈{s_star:.2f}, 对应 FineWeb『≥75% 相似』):")
    hit_by_J = {}
    sig_base = minhash_sig(base_sh, NUM_HASHES)
    for J in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
        m = round(n_base * (1 - J) / J)      # 使 Jaccard = n/(n+m) ≈ J
        hits, trials = 0, 40
        for t in range(trials):
            variant = base_sh | {f"n{t}_{i}" for i in range(m)}
            sig_v = minhash_sig(variant, NUM_HASHES)
            cand = any(sig_base[b * ROWS:(b + 1) * ROWS] == sig_v[b * ROWS:(b + 1) * ROWS]
                       for b in range(BANDS))
            hits += 1 if cand else 0
        hit_by_J[J] = hits / trials
        print(f"     J≈{J:.2f} (added {m:>2} new shingles): candidate hit {hits}/{trials} = {hits/trials:.2f}")
    low, high = hit_by_J[0.5], hit_by_J[0.9]
    print(f"     低相似(0.5) 命中率 {low:.2f} → 高相似(0.9) 命中率 {high:.2f}：S 曲线把『~75% 相似』变成软阈值")
    check("A3 LSH S 曲线分离 (0.9 命中≫0.5)", high >= 0.8 and low <= 0.2)
    check("A3 拐点附近(0.75)命中率居中", 0.2 <= hit_by_J[0.75] <= 0.9)
    record("A_lsh_hit_0.5", round(low, 3))
    record("A_lsh_hit_0.75", round(hit_by_J[0.75], 3))
    record("A_lsh_hit_0.9", round(high, 3))

    # --- 传递聚类：A~B, B~C => {A,B,C} 同簇（FineWeb transitive clustering）---
    # 构造链：x~y（近重复）, y~z（近重复）, 但 x~z 相似度被推到阈值边缘
    x = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    y = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda nu"   # 末词改
    z = "alpha beta gamma delta epsilon zeta eta theta iota kappa nu xi"       # 再改两词
    chain = {"x": shingles(x), "y": shingles(y), "z": shingles(z)}
    edge = {(a, b): jac_true(chain[a], chain[b]) for a, b in [("x", "y"), ("y", "z"), ("x", "z")]}
    TH = 0.5  # toy 聚类阈值（真实 FineWeb 用 ≥75%，toy shingle 少故调低，见局限声明）
    adj = {k: set() for k in chain}
    for (a, b), v in edge.items():
        if v >= TH:
            adj[a].add(b); adj[b].add(a)
    # BFS 连通分量
    comp, visited = [], set()
    for k in chain:
        if k in visited:
            continue
        stack, c = [k], set()
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n); c.add(n)
            stack.extend(adj[n] - visited)
        comp.append(sorted(c))
    print(f"\n[A4] 传递聚类: 边 Jaccard xy={edge[('x','y')]:.2f} yz={edge[('y','z')]:.2f} "
          f"xz={edge[('x','z')]:.2f} (阈值 {TH})")
    print(f"     连通分量: {comp}  <- x~z 即便低于阈值也经 y 同簇")
    check("A4 传递聚类 x,y,z 同簇", sorted(["x", "y", "z"]) in comp)
    record("A_cluster", str(comp))

    print("\n[A 小结] exact-hash 只抓逐字节重复；MinHash 签名把『改字近重复』估计成高 Jaccard；"
          "LSH 分带用 S 曲线把『~75% 相似』放大成软阈值（之上高概率同桶、之下几乎不漏）；"
          "传递聚类把相似链收进同一簇——这就是 FineWeb "
          "『112 hashes / 14 buckets×8 / ≥75% / transitive clustering』的 toy 展开。")


# ----------------------------------------------------------------------------
# [B] 质量过滤：rule-based 启发式（分布判别）+ 质量分数阈值（保留率曲线）
# ----------------------------------------------------------------------------

def frac_lines_end_punct(doc):
    lines = [l for l in doc.split("\n") if l.strip()]
    if not lines:
        return 0.0
    return sum(1 for l in lines if l.rstrip()[-1] in ".!?;:") / len(lines)


def frac_short_lines(doc, min_chars=30):
    lines = [l for l in doc.split("\n") if l.strip()]
    if not lines:
        return 1.0
    return sum(1 for l in lines if len(l.strip()) < min_chars) / len(lines)


def run_B():
    print("\n" + "=" * 72)
    print("[B] Quality filtering: 启发式分布判别 + 分数阈值保留率")
    print("=" * 72)

    # 构造两组文档：高质量（完整句子、以标点收尾）vs 低质量（导航/样板、短行、无标点）
    good = [
        "The experiment confirmed the hypothesis clearly.\nReplication across seeds was stable.\n"
        "These results support the broader theory.",
        "Machine learning models generalize when the data is representative.\n"
        "Careful evaluation prevents overfitting to benchmarks.\nThis is a core principle.",
        "Scientific writing favors complete sentences.\nEach claim is backed by evidence.\n"
        "The argument proceeds step by step.",
    ]
    bad = [
        "home login cart\nsearch menu\ncookie policy terms of use\nclick here now",
        "buy cheap fast\nlimited offer act now\nfree shipping today\nsign up",
        "error 404\nnot found\nretry\nreload page",
    ]
    docs = [("good", d) for d in good] + [("bad", d) for d in bad]

    # --- 启发式分布判别（FineWeb Fig 8 的 <0.12 密度差异）---
    print("[B1] 启发式特征分布（FineWeb 口径：punct-line-frac / short-line-frac）:")
    feats = []
    for label, d in docs:
        fp = frac_lines_end_punct(d)
        fs = frac_short_lines(d)
        feats.append((label, fp, fs))
        print(f"     {label:>4}: punct_line_frac={fp:.3f}  short_line_frac={fs:.3f}")
    good_fp = [fp for l, fp, fs in feats if l == "good"]
    bad_fp = [fp for l, fp, fs in feats if l == "bad"]
    sep = min(good_fp) > max(bad_fp)
    print(f"     分布判别: min(good punct)={min(good_fp):.3f} > max(bad punct)={max(bad_fp):.3f} "
          f"=> 可分: {sep}")
    check("B1 启发式把 good/bad 分开", sep)
    record("B_min_good_punct", round(min(good_fp), 4))
    record("B_max_bad_punct", round(max(bad_fp), 4))

    # --- 应用 FineWeb 阈值（<=0.12 punct / >=0.67 short 即丢）---
    TH_PUNCT, TH_SHORT = 0.12, 0.67
    kept, dropped = [], []
    for label, d in docs:
        fp, fs = frac_lines_end_punct(d), frac_short_lines(d)
        drop = (fp <= TH_PUNCT) or (fs >= TH_SHORT)
        (dropped if drop else kept).append(label)
    print(f"\n[B2] 应用 FineWeb 阈值 (punct<= {TH_PUNCT} 或 short>= {TH_SHORT} 即丢):")
    print(f"     kept={kept}  dropped={dropped}")
    check("B2 丢全部 bad", all(l == "bad" for l in dropped) and len(dropped) == 3)
    check("B2 留全部 good", all(l == "good" for l in kept) and len(kept) == 3)

    # --- 质量分数阈值 → 保留率曲线（DCLM fastText 选子集的本质）---
    # mock 一个分类器分数：真实场景是 fastText 在 OH-2.5+ELI5 上训出的打分器
    rng = random.Random(SEED)
    n = 200
    scores = sorted([rng.betavariate(2, 5) for _ in range(n)], reverse=True)  # 右偏质量分布
    print(f"\n[B3] 质量分数阈值 → 保留率 (n={n}, mock 分类器分数, 右偏):")
    curve = []
    for th in (0.1, 0.2, 0.3, 0.4, 0.5):
        keep = sum(1 for s in scores if s >= th)
        curve.append((th, keep, keep / n))
        print(f"     threshold={th:.1f}: keep={keep:>3}  keep_ratio={keep/n:.3f}")
    mono = all(curve[i][1] >= curve[i + 1][1] for i in range(len(curve) - 1))
    print(f"     保留率随阈值单调下降: {mono}  <- 阈值=『质量 vs 数据量』的唯一旋钮")
    check("B3 保留率随阈值单调降", mono)
    record("B_keep_at_0.3", curve[2][1])

    # --- 静默污染机制（呼应 nano-data-juicer L3 typo 锚）---
    # 阈值配置 typo 掉回默认 => 全保留 => 分布被污染而条数「看起来没变」
    intended_keep = sum(1 for s in scores if s >= 0.3)
    typo_default_keep = n  # 默认阈值≈0，全保留
    print(f"\n[B4] 静默污染: 正确阈值保留 {intended_keep}/{n}, 配置 typo 掉回默认保留 "
          f"{typo_default_keep}/{n} —— 条数『没变』但分布已污染")
    check("B4 typo 导致全保留(≠预期)", typo_default_keep != intended_keep)

    print("\n[B 小结] 廉价启发式之所以有效，是因为低质文档在『标点收尾行占比』等特征上"
          "分布显著不同（可分）；分数阈值把『质量 vs 数据量』压成一个旋钮；"
          "而阈值配错会静默污染分布——条数不撒谎，分布才撒谎。")


# ----------------------------------------------------------------------------
# [C] 数据配比 / 域重加权：DoReMi 的 Group DRO 乘性权重本质（nano 无实测锚 → 本 sim 为锚）
# ----------------------------------------------------------------------------

def run_C():
    print("\n" + "=" * 72)
    print("[C] Mixture reweighting: DoReMi Group DRO 本质 (toy, 真实乘性权重更新)")
    print("=" * 72)

    # toy 设定：K 个域，每域损失随分配 token 指数下降 L_d(t)=floor+(init-floor)*exp(-rate*t)。
    # 域 1 起点最高、下降最慢 => 长期是「最坏域」且改进空间（headroom）最大。
    K = 4
    init  = [1.5, 3.5, 2.2, 1.2]
    floor = [0.5, 0.6, 0.5, 0.4]
    rate  = [1.2, 0.5, 0.8, 1.5]
    # 「The Pile 式」默认配比（非均匀、偏经验）：偏爱易域 0，亏待难域 1
    default_mix = [0.40, 0.10, 0.30, 0.20]

    def loss_vec(t):
        return [floor[d] + (init[d] - floor[d]) * math.exp(-rate[d] * t[d]) for d in range(K)]

    def avg_loss(t):
        return sum(loss_vec(t)) / K

    def worst_loss(t):
        return max(loss_vec(t))

    # --- 阶段 1：proxy 模型用 Group DRO（乘性权重更新，minimax）产出域权重 ---
    w = default_mix[:]
    tokens = [0.0] * K
    ETA, BATCH, STEPS = 0.35, 1.0, 80
    for _ in range(STEPS):
        for d in range(K):
            tokens[d] += BATCH * w[d]
        losses = loss_vec(tokens)
        w = [w[d] * math.exp(ETA * losses[d]) for d in range(K)]
        z = sum(w)
        w = [x / z for x in w]

    final_losses = loss_vec(tokens)
    print("[C1] Group DRO proxy 收敛后的域权重 (vs 默认配比):")
    for d in range(K):
        arrow = "↑" if w[d] > default_mix[d] else "↓"
        print(f"     domain {d}: init_loss={init[d]:.1f} rate={rate[d]:.1f} "
              f"default={default_mix[d]:.2f} -> DRO={w[d]:.3f} {arrow}  (收敛损失={final_losses[d]:.3f})")
    hardest = max(range(K), key=lambda d: init[d])
    print(f"     最难域 domain {hardest} 被显著上调: {default_mix[hardest]:.2f} -> {w[hardest]:.3f}")
    spread = max(final_losses) - min(final_losses)
    print(f"     minimax 不动点签名: 收敛时各域损失近似拉平 "
          f"(max={max(final_losses):.3f} min={min(final_losses):.3f} spread={spread:.3f})")
    check("C1 最难域被上调权重", w[hardest] > default_mix[hardest])
    check("C1 权重严格为正 (乘法更新不丢弃任何域)", min(w) > 0)
    check("C1 收敛时各域损失近似拉平 (minimax 不动点)", spread <= 0.35)
    record("C_default_mix", ",".join(f"{x:.2f}" for x in default_mix))
    record("C_doremi_weights", ",".join(f"{x:.3f}" for x in w))

    # --- 阶段 2：minimax 性质——同预算下，自适应 Group DRO 比固定默认更低压低 worst-domain ---
    def run_budget(mix_adaptive: bool, budget: int):
        t = [0.0] * K
        ww = default_mix[:]
        for _ in range(budget):
            mix = ww if mix_adaptive else default_mix
            for d in range(K):
                t[d] += 1.0 * mix[d]
            if mix_adaptive:
                losses = loss_vec(t)
                ww = [ww[d] * math.exp(ETA * losses[d]) for d in range(K)]
                z = sum(ww)
                ww = [x / z for x in ww]
        return worst_loss(t), avg_loss(t)

    BUDGET = 80
    worst_gd, avg_gd = run_budget(True, BUDGET)
    worst_df, avg_df = run_budget(False, BUDGET)
    print(f"\n[C2] minimax 性质: 同预算 {BUDGET} token 下的 worst-domain 损失")
    print(f"     Group DRO 自适应: worst={worst_gd:.4f}  avg={avg_gd:.4f}")
    print(f"     固定默认配比    : worst={worst_df:.4f}  avg={avg_df:.4f}")
    print(f"     Group DRO 的 worst 更低: {worst_gd < worst_df}  <- 把最坏域拉上来（minimax）")
    check("C2 Group DRO worst-loss 更低", worst_gd < worst_df)
    record("C_worst_gd", round(worst_gd, 4))
    record("C_worst_default", round(worst_df, 4))

    # --- 阶段 3：minimax 加速——更少 token 把所有域拉过质量底线（worst-domain 目标）---
    # Group DRO 优化的是最坏域，所以加速的正确度量是「worst-domain 损失降到质量底线
    # 所需的总 token」，而不是平均损失（平均被易域主导，见 [C3b] 的 toy 边界观察）。
    TARGET_WORST = 0.75

    def tokens_to_worst_target(adaptive: bool, cap=600.0):
        t = [0.0] * K
        ww = default_mix[:]
        total = 0.0
        while total < cap:
            mix = ww if adaptive else default_mix
            for d in range(K):
                t[d] += 1.0 * mix[d]
            total += 1.0
            if adaptive:
                ls = loss_vec(t)
                ww = [ww[d] * math.exp(ETA * ls[d]) for d in range(K)]
                z = sum(ww)
                ww = [x / z for x in ww]
            if worst_loss(t) <= TARGET_WORST:
                return total
        return None

    tt_w_dro = tokens_to_worst_target(True)
    tt_w_default = tokens_to_worst_target(False)
    worst_speedup = (tt_w_default / tt_w_dro) if (tt_w_dro and tt_w_default) else float("nan")
    print(f"\n[C3] 把所有域拉过质量底线 (worst-domain 损失 <= {TARGET_WORST}) 所需总 token:")
    print(f"     自适应 Group DRO: {tt_w_dro} 步")
    print(f"     固定默认配比    : {tt_w_default} 步")
    print(f"     speedup = {worst_speedup:.2f}x  <- 固定配比的瓶颈 = 最难域拿不到预算")
    check("C3 worst-domain 目标 DRO 更快达标", tt_w_dro is not None and tt_w_default is not None and
          tt_w_dro < tt_w_default)
    record("C_tt_worst_dro", tt_w_dro)
    record("C_tt_worst_default", tt_w_default)
    record("C_worst_speedup", round(worst_speedup, 4))

    # --- [C3b] toy 边界观察：静态 proxy 权重两阶段 + 平均损失目标 ---
    # DoReMi 的真实部署 = 主模型用 proxy 权重**静态**重采样。在 toy 的逐域独立损失曲线上，
    # 静态权重把预算压到难域、易域被饿着，平均损失目标反而比默认配比更慢达标——
    # 因为平均损失由易域主导，而 toy 没有跨域迁移（真实 LM 中域间知识会迁移）。
    # 论文 8B 尺度「2.6x fewer steps」是经验结果（最坏域覆盖改善 → 下游全面改善），
    # toy 数字只演示机制方向，**不外推**。
    TARGET_AVG = 0.85

    def tokens_to_avg_target(mix, cap=600.0):
        t = [0.0] * K
        total = 0.0
        while total < cap:
            for d in range(K):
                t[d] += 1.0 * mix[d]
            total += 1.0
            if avg_loss(t) <= TARGET_AVG:
                return total
        return None

    tt_fw_dro = tokens_to_avg_target(w)
    tt_fw_default = tokens_to_avg_target(default_mix)
    fw_ratio = (tt_fw_dro / tt_fw_default) if (tt_fw_dro and tt_fw_default) else float("nan")
    print(f"\n[C3b] toy 边界观察: 静态 proxy 权重两阶段 + 平均损失目标 {TARGET_AVG}:")
    print(f"     DoReMi 静态权重: {tt_fw_dro} 步 | 默认配比: {tt_fw_default} 步 (比值 {fw_ratio:.2f})")
    print(f"     toy 上静态权重反而更慢——平均损失由易域主导、toy 无跨域迁移；"
          f"论文 2.6x 为 8B 尺度经验结果，不外推")
    record("C_fw_avg_dro", tt_fw_dro)
    record("C_fw_avg_default", tt_fw_default)

    # --- DoReMi 反直觉性质：被降权的域最终也改善（论文『even when it downweights』）---
    downweighted = [d for d in range(K) if w[d] < default_mix[d]]
    improved = all(loss_vec(tokens)[d] < init[d] for d in downweighted) if downweighted else True
    print(f"\n[C4] 被降权域 {downweighted} 在 proxy 阶段损失仍从 init 下降: {improved} "
          f"<- 『improves perplexity across all domains, even when it downweights a domain』")
    check("C4 被降权域也改善", improved)

    print("\n[C 小结] Group DRO 的本质是『用乘性权重把训练预算往当前最坏域倾斜』的 minimax 控制：")
    print("        收敛时各域损失近似拉平（不动点签名），worst-domain 质量底线用更少 token 就能守住。")
    print("        注意它优化的是最坏域而非平均——toy 逐域独立曲线无跨域迁移，平均损失口径的加速")
    print("        （论文 8B 实测 2.6x fewer steps / 6.5pp）是大规模经验结果，toy 只演示机制、不外推。")


# ----------------------------------------------------------------------------
# [D] 去污染 / 污染检测：n-gram 重叠（Lee et al. / DCLM 工具的本质）
# ----------------------------------------------------------------------------

def word_ngrams(text, n):
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def contamination_ratio(doc, test_ngrams, n):
    dg = word_ngrams(doc, n)
    if not dg:
        return 0.0
    return len(dg & test_ngrams) / len(dg)


def run_D():
    print("\n" + "=" * 72)
    print("[D] Decontamination: n-gram 重叠检测 (Lee et al. / DCLM 工具本质)")
    print("=" * 72)

    N = 4
    benchmark = [
        "what is the capital of france and why is it known for the eiffel tower",
        "explain the process of photosynthesis and how plants convert sunlight",
        "describe the causes of the french revolution and its major outcomes",
    ]
    test_ngrams = set()
    for b in benchmark:
        test_ngrams |= word_ngrams(b, N)

    # 训练语料：干净文档 + 植入污染（exact 拷贝 / 近拷贝 / 改写）
    clean1 = ("distributed systems use consensus protocols like raft and paxos to "
              "replicate state machines across faulty nodes reliably")
    clean2 = ("the history of typography shows how movable type changed printing "
              "and accelerated the spread of literacy across europe")
    exact_copy = benchmark[0]                                            # 背题：逐字拷贝
    near_copy = benchmark[1].replace("photosynthesis", "photosynthesis in leaves")  # 轻微改写
    paraphrase = ("plants turn sunlight into chemical energy through a process "
                  "called photosynthesis which is fundamental to life")   # 改写（重叠下降）

    train = {"clean1": clean1, "clean2": clean2, "exact_copy": exact_copy,
             "near_copy": near_copy, "paraphrase": paraphrase}

    print(f"[D1] 每个训练文档与 benchmark 的 {N}-gram 重叠率:")
    ratios = {}
    for k, d in train.items():
        r = contamination_ratio(d, test_ngrams, N)
        ratios[k] = r
        print(f"     {k:>11}: overlap={r:.3f}")
    check("D1 exact_copy 重叠率=1.0", abs(ratios["exact_copy"] - 1.0) < 1e-9)
    check("D1 clean 文档重叠率≈0", ratios["clean1"] < 0.05 and ratios["clean2"] < 0.05)
    check("D1 near_copy 重叠显著高于 clean", ratios["near_copy"] > 0.2 and
          ratios["near_copy"] > max(ratios["clean1"], ratios["clean2"]))
    check("D1 改写(paraphrase)逃过 n-gram 检测 (≈clean)", ratios["paraphrase"] <= 0.2)
    print(f"     ⚠ 改写(paraphrase) 重叠率仅 {ratios['paraphrase']:.3f} —— n-gram 探针抓得住拷贝、"
          f"抓不住改写（已知盲区，需语义/嵌入级去污染补位）")
    record("D_ratios", ",".join(f"{k}:{ratios[k]:.3f}" for k in sorted(train)))

    # --- 阈值过滤：重叠率超阈值即判污染并移除 ---
    TH = 0.3
    flagged = {k for k, r in ratios.items() if r >= TH}
    print(f"\n[D2] 阈值 {TH} 判污染: flagged={sorted(flagged)}")
    print(f"     干净文档误报: {sorted(flagged & {'clean1','clean2'})}  "
          f"植入污染漏报: {sorted({'exact_copy','near_copy'} - flagged)}")
    check("D2 exact+near 被标记", {"exact_copy", "near_copy"} <= flagged)
    check("D2 clean 不误报", not (flagged & {"clean1", "clean2"}))
    record("D_flagged", ",".join(sorted(flagged)))

    # --- 去污染前后：移除污染文档，保留干净 + 改写（低重叠）---
    kept = [k for k in train if k not in flagged]
    print(f"\n[D3] 去污染后保留: {sorted(kept)}  ({len(kept)}/{len(train)})")
    print(f"     条数下降 {len(train)-len(kept)} 条 = 背题文档被剔除，评测才可信")
    check("D3 至少剔除 2 条污染", len(train) - len(kept) >= 2)

    print("\n[D 小结] n-gram 重叠率是『背题』的廉价而有效的探针：逐字拷贝=1.0、近拷贝高、干净≈0；"
          "但改写级污染重叠率也≈0，会逃过检测——这是 n-gram 法的已知盲区。DCLM 选择"
          "『发去污染工具 + 要求披露报告』而非直接清洗整个池，正是因为污染对下游的影响"
          "『remains largely unclear』——先可测量，再谈清洗。")


# ----------------------------------------------------------------------------
# 自检 + digest
# ----------------------------------------------------------------------------

def main():
    print("LLM-PBL 03 轨 sota-deepdive — data methodology essence sim")
    print(f"seed={SEED} | 纯标准库 (hashlib/math/random) | toy 尺度，机制演示，不外推生产数字")
    print()
    run_A()
    run_B()
    run_C()
    run_D()

    print("\n" + "=" * 72)
    print("[E] self-check")
    print("=" * 72)
    passed = sum(1 for _, ok in CHECKS if ok)
    for name, ok in CHECKS:
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    digest = hashlib.md5("|".join(DIGEST_PARTS).encode()).hexdigest()
    print(f"\n    self-check {'passed' if passed == len(CHECKS) else 'FAILED'} "
          f"({passed}/{len(CHECKS)})")
    print(f"    digest(md5 of metrics) = {digest}")
    if passed != len(CHECKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
