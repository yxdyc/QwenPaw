#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_training_evolution_sim.py — 01 轨 sota-deepdive（post-training-algorithm-evolution.md）
的可运行本质模拟：PPO → GRPO/RLVR → OPD 演进主线的四个机制面。

这是什么
========
deepdive 正文的机制论断此前全部锚在 相关训练模块 nano 模块（nano-verl / nano-opd /
nano-llamafactory，交叉引用）与一手论文/博客引文上；GRPO 族变体修复的缺陷
（ratio 粒度、长度偏置）在正文里只有摘要/博客层引文。本文件把其中四件「可以用
算术和 toy 训练直接跑出来」的机制变成本主题自己的实测锚：

  [A] PPO 地基：importance sampling ratio 的无偏性（MC vs 精确求和双通道）
      + clipping 在「旧批次多轮复用」下把策略拉出采样分布的速度限住（§2）。
  [B] GRPO：组内均值基线对 prompt 级 reward 常数偏移的精确不变性（机器证明）
      + 方差缩减实测 + 基线自身方差 ~1/sqrt(G)（「采样换参数」的统计账，§3）。
  [C] ratio 的粒度：token 级 vs 序列级（几何平均、长度归一，GSPO 定义）——
      单 token logprob 扰动在两种粒度下的传导系数 1 vs 1/T（机器证明）、
      同一批同一 eps 两种粒度的 clip 比例实测、sum/mean 聚合的长度偏置
      （Dr. GRPO 机制类别，§4）。
  [D] OPD 合流：reward 来源换成教师 logprob 的 on-policy 训练——教师是**自回归**
      双模分布（序列级双峰），学生是分解式（逐位置 softmax，无跨位置关联）。
      三臂对照：opd_seq（自采样 + 序列级 advantage = 负 reverse KL，MiniLLM
      配方 + GRPO 组基线）锁模；sft（教师采样 + NLL）掉谷；opd_tok（TM 逐 token
      配方）在无上下文学生上收敛到 mode-covering 不动点——配方×容量共轭的实测
      发现；[D2] 在可全支撑枚举的缩小版上用精确求和复现 nano-opd L0 的驻点
      算术：锁模点上正确估计器≈0、教师采样估计器系统性指向谷（§6）。

可运行性契约（课程可运行性契约）
==========================
- 本文件是**本质模拟**：toy 尺度（V=6 词表、T=8 位、分解式策略——每位置一个
  softmax，即「仍有 token 级 logprob 的最小语言模型」），演示机制、不外推量级。
  真实系统行为见 deepdive §2–§7 的一手来源（arXiv 2402.03300 / 2507.18071 /
  2503.20783 / TM OPD 博客 2025-10-27 等）与 相关训练模块 nano-verl / nano-opd 实测锚。
- 纯标准库（math / random / hashlib），零外部依赖，CPU 秒级。
- seed=3 固定，无计时行，跨运行逐字节一致（digest 见输出末行）。
- 方法学继承 nano-opd L0：关键算术用**精确求和**——分解式策略的 KL/期望有闭式
  （O(T·V)），[D2] 缩小版全支撑枚举（3^6=729 条序列逐条精确）——「偏差不是
  噪声」的同源示范。

运行
====
    python3 -B post_training_evolution_sim.py
"""

import hashlib
import math
import random
import sys

sys.dont_write_bytecode = True

SEED = 3
V, T = 6, 8                 # 主动场：词表 6、序列长 8
EPS_CLIP = 0.2               # PPO/GRPO 族通用裁剪界（deepdive §2.1）
F_TOKEN = [0.0, 0.5, 1.0, 0.2, 0.8, 0.3]   # [A] 的逐 token 可加 reward

MODE_A = [0, 1, 0, 1, 0, 1, 0, 1]          # [D] 教师双模：两个交替 pattern
MODE_B = [1, 0, 1, 0, 1, 0, 1, 0]

rng = random.Random(SEED)
CHECKS = []
DIGEST = {}


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"    {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def softmax(logits):
    m = max(logits)
    es = [math.exp(x - m) for x in logits]
    s = sum(es)
    return [e / s for e in es]


def logsoftmax(logits):
    m = max(logits)
    lse = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - lse for x in logits]


def policy_tables(theta):
    """theta: [T][V] logits -> (pi, logpi) 两套表。"""
    pi, lpi = [], []
    for t in range(len(theta)):
        pi.append(softmax(theta[t]))
        lpi.append(logsoftmax(theta[t]))
    return pi, lpi


def seq_logprob(lpi, y):
    return sum(lpi[t][y[t]] for t in range(len(y)))


def sample_seq(pi):
    return [rng.choices(range(len(pi[t])), weights=pi[t])[0] for t in range(len(pi))]


def kl_factored(theta_p, theta_q):
    """KL(P||Q) 精确值：分解式分布 = 逐位置 KL 之和（闭式，零采样）。"""
    kl = 0.0
    for t in range(T):
        p, lp = softmax(theta_p[t]), logsoftmax(theta_p[t])
        lq = logsoftmax(theta_q[t])
        for v in range(V):
            if p[v] > 0.0:
                kl += p[v] * (lp[v] - lq[v])
    return kl


def exact_additive_expectation(theta):
    """E_pi[sum_t F_TOKEN[y_t]]：逐位置精确求和。"""
    _, lpi = policy_tables(theta)
    return sum(math.exp(lpi[t][v]) * F_TOKEN[v] for t in range(T) for v in range(V))


def parity_reward(y):
    return 1 if (sum(y) % 2 == 0) else 0


def score_seq(lpi, y):
    """d log pi(y) / d theta：每位置 one-hot − pi，展平向量。"""
    n = len(lpi)
    voc = len(lpi[0])
    g = [0.0] * (n * voc)
    for t in range(n):
        pi_t = [math.exp(x) for x in lpi[t]]
        for v in range(voc):
            g[t * voc + v] = (1.0 if v == y[t] else 0.0) - pi_t[v]
    return g


def flat_norm(g):
    return math.sqrt(sum(x * x for x in g))


def pctl(sorted_xs, q):
    """已排序列表的 q 分位（最近邻法，确定性）。"""
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


print("=" * 72)
print("post-training evolution sim — IS+clip / GRPO baseline / ratio 粒度 / OPD")
print("=" * 72)
print(f"toy: V={V}, T={T}, 分解式策略(每位置 softmax) | seed={SEED} | 纯标准库")
print("机制回声对象: deepdive §2(PPO) §3(GRPO) §4(GRPO族) §6(OPD)；")
print("全部数字为本 sim 实测，论文侧只作机制对照、不外推量级。")

# ======================================================================
# [A] PPO 地基：IS ratio 无偏 + clipping 限住信任域（deepdive §2）
# ======================================================================
print()
print("[A] IS ratio + clipping：旧数据复用的算术地基")

# theta_old：初始策略；theta_new = theta_old + 0.6*(F - mean F)：偏向高 F token
theta_old = [[0.5 * (rng.random() - 0.5) for _ in range(V)] for _ in range(T)]
mean_f = sum(F_TOKEN) / V
theta_new = [[theta_old[t][v] + 0.6 * (F_TOKEN[v] - mean_f) for v in range(V)]
             for t in range(T)]

e_old = exact_additive_expectation(theta_old)
e_new = exact_additive_expectation(theta_new)
print(f"    精确期望（闭式）: E_old[f] = {e_old:.6f}  E_new[f] = {e_new:.6f}")

# [A0] IS 无偏：从 pi_old 采样，ratio 加权重估 E_new[f]；ratio 均值应为 1
N_A0 = 4000
pi_old, lpi_old = policy_tables(theta_old)
_, lpi_new = policy_tables(theta_new)
batch_a0 = [sample_seq(pi_old) for _ in range(N_A0)]
ratios_a0 = [math.exp(seq_logprob(lpi_new, y) - seq_logprob(lpi_old, y)) for y in batch_a0]
f_a0 = [sum(F_TOKEN[y[t]] for t in range(T)) for y in batch_a0]
mean_ratio = sum(ratios_a0) / N_A0
is_est = sum(r * f for r, f in zip(ratios_a0, f_a0)) / N_A0
print(f"    [A0] N={N_A0}: mean(ratio) = {mean_ratio:.6f} (应≈1)")
print(f"         IS 估计 E_new[f] = {is_est:.6f} vs 精确 {e_new:.6f}"
      f"  (相对误差 {abs(is_est - e_new) / e_new * 100:.2f}%)")
DIGEST["a0_mean_ratio"] = round(mean_ratio, 6)
DIGEST["a0_is_est"] = round(is_est, 6)
check("A0a E_old[ratio]≈1（旧数据复用的算术恒等式）", abs(mean_ratio - 1.0) < 0.03,
      f"mean_ratio={mean_ratio:.6f}")
check("A0b IS 重加权估计命中精确期望（<5%）", abs(is_est - e_new) / e_new < 0.05,
      f"est={is_est:.4f} exact={e_new:.4f}")

# [A1] 同一旧批次多轮更新：clipped vs unclipped 的信任域漂移
N_A1, LR_A1, K_A1 = 500, 1.5, 12
batch_a1 = [sample_seq(pi_old) for _ in range(N_A1)]
adv_a1 = [sum(F_TOKEN[y[t]] for t in range(T)) - e_old for y in batch_a1]
lp_old_a1 = [seq_logprob(lpi_old, y) for y in batch_a1]


def surrogate_train(clip, n_epochs=K_A1):
    theta = [row[:] for row in theta_old]
    kl_traj, obj_traj, rmax_traj = [], [], []
    for _ in range(n_epochs):
        pi, lpi = policy_tables(theta)
        grad = [[0.0] * V for _ in range(T)]
        rmax = 0.0
        for y, a, lpo in zip(batch_a1, adv_a1, lp_old_a1):
            r = math.exp(seq_logprob(lpi, y) - lpo)
            rmax = max(rmax, r)
            if clip:
                active = (a > 0 and r < 1 + EPS_CLIP) or (a < 0 and r > 1 - EPS_CLIP)
                w = a * r if active else 0.0
            else:
                w = a * r
            if w == 0.0:
                continue
            for t in range(T):
                yt = y[t]
                for v in range(V):
                    grad[t][v] += w * ((1.0 if v == yt else 0.0) - pi[t][v])
        for t in range(T):
            for v in range(V):
                theta[t][v] += LR_A1 * grad[t][v] / N_A1
        kl_traj.append(kl_factored(theta_old, theta))
        obj_traj.append(exact_additive_expectation(theta))
        rmax_traj.append(rmax)
    return theta, kl_traj, obj_traj, rmax_traj


theta_clip, kl_clip_traj, obj_clip_traj, rmax_clip = surrogate_train(clip=True)
theta_noclip, kl_noclip_traj, obj_noclip_traj, rmax_noclip = surrogate_train(clip=False)
kl_clip, kl_noclip = kl_clip_traj[-1], kl_noclip_traj[-1]
df_clip = obj_clip_traj[-1] - e_old
df_noclip = obj_noclip_traj[-1] - e_old
print(f"    [A1] 同一旧批次 {N_A1} 条 × {K_A1} 轮（lr={LR_A1}, eps={EPS_CLIP}）:")
print(f"         KL(old||cur) 第1/6/12轮  clipped   = "
      f"{kl_clip_traj[0]:.4f} / {kl_clip_traj[5]:.4f} / {kl_clip:.4f}"
      f"  | 批内最大 ratio {rmax_clip[-1]:.2f}")
print(f"         KL(old||cur) 第1/6/12轮  unclipped = "
      f"{kl_noclip_traj[0]:.4f} / {kl_noclip_traj[5]:.4f} / {kl_noclip:.4f}"
      f"  | 批内最大 ratio {rmax_noclip[-1]:.2f}")
print(f"         真实目标 E_cur[f]: clipped {e_old:.4f}→{obj_clip_traj[-1]:.4f} (+{df_clip:.4f})"
      f" | unclipped →{obj_noclip_traj[-1]:.4f} (+{df_noclip:.4f})")
DIGEST["a1_kl_clip"] = round(kl_clip, 6)
DIGEST["a1_kl_noclip"] = round(kl_noclip, 6)
DIGEST["a1_df_clip"] = round(df_clip, 6)
DIGEST["a1_df_noclip"] = round(df_noclip, 6)
DIGEST["a1_rmax_clip"] = round(rmax_clip[-1], 4)
DIGEST["a1_rmax_noclip"] = round(rmax_noclip[-1], 4)
check("A1a clipping 把信任域漂移压在 0.5 nats 内", kl_clip < 0.5, f"kl_clip={kl_clip:.4f}")
check("A1b unclipped 漂移 > 2× clipped（限幅器在做事）", kl_noclip > 2 * kl_clip,
      f"{kl_noclip:.4f} vs {kl_clip:.4f}")
check("A1c unclipped 批内最大 ratio 失控（> 2× clipped 的最大 ratio）",
      rmax_noclip[-1] > 2 * rmax_clip[-1],
      f"{rmax_noclip[-1]:.2f} vs {rmax_clip[-1]:.2f}")
check("A1d clipped 复用仍能提升真实目标", df_clip > 0.3, f"+{df_clip:.4f}")

# ======================================================================
# [B] GRPO：组内均值基线 = 穷人版 critic（deepdive §3）
# ======================================================================
print()
print("[B] GRPO group baseline：偏移不变性（机器证明）+ 方差缩减（实测）")

# 用 [A] 的 theta_new 作「训练中策略」，parity reward R∈{0,1}
theta_b = [row[:] for row in theta_new]
pi_b, lpi_b = policy_tables(theta_b)

# [B0] prompt 级常数偏移 c：A_i = R_i − mean(R) 精确不变
G_B, M_B = 16, 300
max_diff = 0.0
for g in range(M_B):
    ys = [sample_seq(pi_b) for _ in range(G_B)]
    rs = [parity_reward(y) for y in ys]
    c = 0.3 * (g % 5)                      # 每个「prompt」一个不同的难度偏移
    mu = sum(rs) / G_B
    mu_c = sum(r + c for r in rs) / G_B
    for r in rs:
        max_diff = max(max_diff, abs((r - mu) - ((r + c) - mu_c)))
print(f"    [B0] {M_B} 组 × G={G_B}，reward 加 prompt 级常数 c：max|A − A'| = {max_diff:.2e}")
DIGEST["b0_max_diff"] = float(f"{max_diff:.2e}")
check("B0 组基线对 reward 常数偏移精确不变（critic 要学的量，组均值免费消掉）",
      max_diff < 1e-12, f"max_diff={max_diff:.2e}")

# [B1] 方差缩减：同一批样本，b=0 vs b=组均值，梯度估计的组间离散
def group_grad_var():
    msdev0, msdevb, gsum0, gsumb = [], [], [0.0] * (T * V), [0.0] * (T * V)
    grads0, gradsb = [], []
    for _ in range(M_B):
        ys = [sample_seq(pi_b) for _ in range(G_B)]
        rs = [parity_reward(y) for y in ys]
        mu = sum(rs) / G_B
        g0 = [0.0] * (T * V)
        gb = [0.0] * (T * V)
        for y, r in zip(ys, rs):
            s = score_seq(lpi_b, y)
            for k in range(T * V):
                g0[k] += s[k] * r / G_B
                gb[k] += s[k] * (r - mu) / G_B
        grads0.append(g0)
        gradsb.append(gb)
        for k in range(T * V):
            gsum0[k] += g0[k]
            gsumb[k] += gb[k]
    mean0 = [x / M_B for x in gsum0]
    meanb = [x / M_B for x in gsumb]
    for g0, gb in zip(grads0, gradsb):
        msdev0.append(sum((a - m) ** 2 for a, m in zip(g0, mean0)))
        msdevb.append(sum((a - m) ** 2 for a, m in zip(gb, meanb)))
    return sum(msdev0) / M_B, sum(msdevb) / M_B


msdev0, msdevb = group_grad_var()
var_ratio = msdev0 / msdevb
print(f"    [B1] 梯度估计组间离散 MSDEV: b=0 → {msdev0:.6f} | b=组均值 → {msdevb:.6f}"
      f"  (方差缩减 {var_ratio:.2f}×)")
DIGEST["b1_var_ratio"] = round(var_ratio, 4)
check("B1 组均值基线降低梯度估计方差（>1.3×）", var_ratio > 1.3, f"ratio={var_ratio:.2f}")

# [B2] 「采样换参数」的统计账：基线自身方差 ~ 1/G
def baseline_std(gsize, n_groups=400):
    bs = []
    for _ in range(n_groups):
        rs = [parity_reward(sample_seq(pi_b)) for _ in range(gsize)]
        bs.append(sum(rs) / gsize)
    mu = sum(bs) / len(bs)
    return math.sqrt(sum((b - mu) ** 2 for b in bs) / len(bs))


std4, std16, std64 = baseline_std(4), baseline_std(16), baseline_std(64)
decay = std64 / std4
print(f"    [B2] 组均值基线的标准差: G=4 → {std4:.4f} | G=16 → {std16:.4f} | G=64 → {std64:.4f}"
      f"  (std64/std4 = {decay:.3f}, 理论 1/sqrt(16) = 0.250)")
DIGEST["b2_decay"] = round(decay, 4)
check("B2 基线精度随 G 按 1/sqrt(G) 提升（采样预算换统计质量）", 0.15 < decay < 0.38,
      f"decay={decay:.3f}")
print("    declared 算术（非实测，口径 = nano-verl tutorial_L3 §5 COST 模型，相关训练模块锚）:")
print("    PPO 训练态 ≈ policy P + critic P + 两套 Adam m/v + 梯度 ≈ 8P；")
print("    GRPO 去掉 critic 及其优化器态 → 4P 口径（nano-verl L3: 4P/N，7B@N=4 ≈ 34 GB/rank）。")

# ======================================================================
# [C] ratio 的粒度：token 级 vs 序列级（deepdive §4，GSPO/Dr.GRPO 机制类别）
# ======================================================================
print()
print("[C] token 级 vs 序列级 ratio：粒度决定噪声与 clip 的对象")

# 制造真实「陈旧度」：把 [A1] 的 clipped 训练在同一旧批次上续到第 32 轮——
# 策略持续训练而 rollout 批次不刷新，正是 GRPO 族每天面对的日常陈旧度
K_C = 32
theta_cur, kl_c_traj, _, _ = surrogate_train(clip=True, n_epochs=K_C)
print(f"    陈旧度来源: [A1] 同款 clipped 训练续到 {K_C} 轮"
      f"（KL(old||cur) = {kl_c_traj[-1]:.4f} nats）")

pi_oldc, lpi_oldc = policy_tables(theta_old)    # 旧策略 = 采样分布
pi_newc, lpi_newc = policy_tables(theta_cur)    # 新策略 = 当前训练策略
N_C = 2000
batch_c = [sample_seq(pi_oldc) for _ in range(N_C)]
tok_logr, seq_logr = [], []
for y in batch_c:
    lrs = [lpi_newc[t][y[t]] - lpi_oldc[t][y[t]] for t in range(T)]
    tok_logr.extend(lrs)
    seq_logr.append(sum(lrs) / T)               # 几何平均、长度归一 = GSPO 序列级 ratio 定义
abs_tok = sorted(abs(x) for x in tok_logr)
abs_seq = sorted(abs(x) for x in seq_logr)
p95_tok, p95_seq = pctl(abs_tok, 0.95), pctl(abs_seq, 0.95)
granularity_ratio = p95_tok / p95_seq
print(f"    [C0] 陈旧批次 N={N_C}（从旧策略采样，ratio = 当前策略/旧策略）:")
print(f"         p95|log token ratio| = {p95_tok:.4f} | p95|log seq ratio| = {p95_seq:.4f}"
      f"  (token 级噪声宽 {granularity_ratio:.2f}×)")

# 敏感性机器证明：单 token logprob 扰动 Δ，token ratio 传导 Δ，序列 ratio 只传导 Δ/T
DELTA = 1.0
tok_change = DELTA                            # 该位置 log ratio 的变化
seq_change = DELTA / T                        # 序列级 = 逐 token 平均
transmit = tok_change / seq_change
print(f"    [C0b] 单 token logprob 扰动 Δ={DELTA}: token ratio ×e^{DELTA:.0f}={math.exp(DELTA):.3f}"
      f" | seq ratio ×e^{{Δ/T}}={math.exp(DELTA / T):.3f}  (传导系数比 = {transmit:.1f} = T)")
DIGEST["c0_p95_tok"] = round(p95_tok, 6)
DIGEST["c0_p95_seq"] = round(p95_seq, 6)
DIGEST["c0_granularity_ratio"] = round(granularity_ratio, 4)
DIGEST["c0_transmit"] = round(transmit, 6)
check("C0a token 级 ratio 噪声宽度 > 2× 序列级（几何平均吃掉单 token 噪声）",
      granularity_ratio > 2.0, f"{granularity_ratio:.2f}x")
check("C0b 序列级对单 token 扰动的敏感性恰为 1/T（机器证明）", abs(transmit - T) < 1e-12,
      f"transmit={transmit:.1f}")

# [C1] 同一批、同一 ε：两种粒度 clip 的对象不同
tok_clip = sum(1 for x in tok_logr if math.exp(x) > 1 + EPS_CLIP or math.exp(x) < 1 - EPS_CLIP)
seq_clip = sum(1 for x in seq_logr if math.exp(x) > 1 + EPS_CLIP or math.exp(x) < 1 - EPS_CLIP)
tok_frac, seq_frac = tok_clip / (N_C * T), seq_clip / N_C
print(f"    [C1] eps={EPS_CLIP}: token 级 clip 比例 = {tok_frac * 100:.2f}% ({tok_clip}/{N_C * T})"
      f" | 序列级 clip 比例 = {seq_frac * 100:.2f}% ({seq_clip}/{N_C})")
DIGEST["c1_tok_frac"] = round(tok_frac, 6)
DIGEST["c1_seq_frac"] = round(seq_frac, 6)
check("C1 两种粒度 clip 的是不同对象（token 极端值 vs 序列整体漂移）",
      tok_frac > 0.05 and tok_frac > 2 * seq_frac,
      f"tok={tok_frac * 100:.2f}% seq={seq_frac * 100:.2f}%")

# [C2] 长度偏置（Dr. GRPO 机制类别）：聚合归一决定长度是否进入梯度
A_PER_TOK = 0.5
push_sum_long = A_PER_TOK * T          # sum 聚合：总推力 = Σ_t a_t = T·a
push_sum_short = A_PER_TOK * (T // 2)  # 同每 token 优势、半长响应
push_mean_long = push_sum_long / T     # mean 聚合：总推力 = ā，与长度无关
push_mean_short = push_sum_short / (T // 2)
ratio_sum = push_sum_long / push_sum_short
ratio_mean = push_mean_long / push_mean_short
print(f"    [C2] 每 token 优势同为 {A_PER_TOK}: sum 聚合总推力 长/短 = "
      f"{push_sum_long:.1f}/{push_sum_short:.1f} = {ratio_sum:.1f}（∝长度）"
      f" | mean 聚合 = {push_mean_long:.2f}/{push_mean_short:.2f} = {ratio_mean:.1f}")
DIGEST["c2_ratio_sum"] = round(ratio_sum, 6)
DIGEST["c2_ratio_mean"] = round(ratio_mean, 6)
check("C2a sum 聚合使更新幅度 ∝ 响应长度（机器证明）", abs(ratio_sum - 2.0) < 1e-12,
      f"ratio={ratio_sum:.1f}")
check("C2b mean 聚合消除长度依赖（机器证明）", abs(ratio_mean - 1.0) < 1e-12,
      f"ratio={ratio_mean:.1f}")

# ======================================================================
# [D] OPD：同一个 IS loss，reward = 负 reverse KL（deepdive §6，TM 配方）
# ======================================================================
print()
print("[D] OPD 合流：自采样+教师背书 → 锁模（opd_seq）；教师采样 → 掉谷（sft）；")
print("    逐 token 配方的锁模需要学生有序列容量（opd_tok，实测对照）")


# 教师：**自回归**双模分布——先隐变量选模 z∈{A,B}，再逐位置以 0.9 概率沿模走。
# 等价实现：条件概率依赖前缀的模一致性差 d = (#匹配A − #匹配B)：
#   d=0（未定模）: p(A_t)=0.45, p(B_t)=0.45, 其余 4 token 各 0.025
#   d>0（偏 A）  : p(A_t)=0.9,  p(B_t)=0.02, 其余各 0.02      d<0 镜像
# 序列级双峰：纯模序列 p ≈ 0.45×0.9^7 ≈ 0.215，混合序列概率指数级小。
def teacher_cond_logprobs(y):
    """逐位置条件 logprob 列表 + 序列 logprob（自回归链式）。"""
    lps = []
    d = 0
    for t in range(T):
        if d > 0:
            pa, pb, pr = 0.9, 0.02, 0.02
        elif d < 0:
            pa, pb, pr = 0.02, 0.9, 0.02
        else:
            pa, pb, pr = 0.45, 0.45, 0.025
        yt = y[t]
        if yt == MODE_A[t]:
            lps.append(math.log(pa))
        elif yt == MODE_B[t]:
            lps.append(math.log(pb))
        else:
            lps.append(math.log(pr))
        if yt == MODE_A[t]:
            d += 1
        elif yt == MODE_B[t]:
            d -= 1
    return lps, sum(lps)


def sample_teacher():
    y = []
    d = 0
    for t in range(T):
        if d > 0:
            pa, pb, pr = 0.9, 0.02, 0.02
        elif d < 0:
            pa, pb, pr = 0.02, 0.9, 0.02
        else:
            pa, pb, pr = 0.45, 0.45, 0.025
        weights = [pr] * V
        weights[MODE_A[t]] = pa
        weights[MODE_B[t]] = pb
        yt = rng.choices(range(V), weights=weights)[0]
        y.append(yt)
        if yt == MODE_A[t]:
            d += 1
        elif yt == MODE_B[t]:
            d -= 1
    return y


def mode_mass(theta):
    _, lpi = policy_tables(theta)
    ma = sum(lpi[t][MODE_A[t]] for t in range(T))
    mb = sum(lpi[t][MODE_B[t]] for t in range(T))
    return max(math.exp(ma), math.exp(mb))


def teacher_endorsement(theta, n=2000):
    """E_{y~pi}[ mean_t log p_teacher(y_t|y_{<t}) ]：教师对学生采样的平均背书度。"""
    pi, _ = policy_tables(theta)
    tot = 0.0
    for _ in range(n):
        y = sample_seq(pi)
        lps, lp = teacher_cond_logprobs(y)
        tot += lp / T
    return tot / n


def dwell_rate(theta, n=2000):
    """采样序列与两个模 pattern 的 Hamming 距离 ≤1 的比例。"""
    pi, _ = policy_tables(theta)
    hit = 0
    for _ in range(n):
        y = sample_seq(pi)
        da = sum(1 for t in range(T) if y[t] != MODE_A[t])
        db = sum(1 for t in range(T) if y[t] != MODE_B[t])
        if min(da, db) <= 1:
            hit += 1
    return hit / n


# 同一初始化分叉两条路线（受控对照：样本从谁采是唯一变量）。
# 初始化带 +0.2 的 mode-A 先验偏置（显式声明：真实学生经 SFT 预热后总带先验；
# 机制主张 = OPD 把小偏置放大成整模锁定，SFT 把先验冲掉、停在谷里）。
theta_init = [[0.2 * (rng.random() - 0.5) for _ in range(V)] for _ in range(T)]
for t in range(T):
    theta_init[t][MODE_A[t]] += 0.2
LR_D, ITERS_D, G_D = 0.1, 400, 24
print(f"    对照设置: 同一初始化（噪声±0.1 + mode-A 先验偏置 +0.2）、同 {ITERS_D} 步、"
      f"同 lr={LR_D}——信号配方 × 样本从谁采是变量")
endorse0 = teacher_endorsement(theta_init)
mass0 = mode_mass(theta_init)

# [D0] 三臂对照（同初始化，唯一变量 = 信号配方 × 样本从谁采）：
#   opd_seq: 学生自采样 + 序列级 advantage A(y) = logp_teacher(y) − logq(y)
#            （= 负序列级 reverse KL，MiniLLM 2306.08543 的 REINFORCE 配方）+
#            组均值基线——机制上就是「GRPO 的采样/基线机制 + reward 来源换成教师 logprob」
#   opd_tok: 学生自采样 + 逐 token advantage = logp(y_t|y_{<t}) − logq_t(y_t)
#            （TM 2025-10-27 配方的逐字机制）——在**无上下文学生**上的实测行为见输出
#   sft:     教师采样 + 学生 NLL（off-policy 蒸馏）
def train_opd(seq_level):
    theta = [row[:] for row in theta_init]
    for it in range(ITERS_D):
        pi, lpi = policy_tables(theta)
        batch = [sample_seq(pi) for _ in range(G_D)]
        weights = []                       # 每条序列的逐位置更新权重
        if seq_level:
            advs = []
            for y in batch:
                _, lp_teach = teacher_cond_logprobs(y)
                advs.append(lp_teach - seq_logprob(lpi, y))
            b = sum(advs) / len(advs)      # 组均值基线（GRPO 机制）
            weights = [[(a - b)] * T for a in advs]
        else:
            for y in batch:
                lps_t, _ = teacher_cond_logprobs(y)
                weights.append([lps_t[t] - lpi[t][y[t]] for t in range(T)])
        grad = [[0.0] * V for _ in range(T)]
        for y, ws in zip(batch, weights):
            for t in range(T):
                w = ws[t]
                yt = y[t]
                for v in range(V):
                    grad[t][v] += w * ((1.0 if v == yt else 0.0) - pi[t][v])
        for t in range(T):
            for v in range(V):
                theta[t][v] += LR_D * grad[t][v] / G_D
        if seq_level and (it + 1) % 100 == 0:
            print(f"    [D0] opd_seq iter {it + 1:3d}: 教师背书 = "
                  f"{teacher_endorsement(theta, 1000):.4f} nats/token"
                  f" | 模质量 = {mode_mass(theta):.4f}")
    return theta


theta_opd = train_opd(seq_level=True)
theta_tok = train_opd(seq_level=False)

endorse_opd = teacher_endorsement(theta_opd)
mass_opd = mode_mass(theta_opd)
dwell_opd = dwell_rate(theta_opd)
mass_tok = mode_mass(theta_tok)
dwell_tok = dwell_rate(theta_tok)

# [D1] off-policy SFT（教师采样、学生 NLL）：同初始化、同步数、同 lr
theta_sft = [row[:] for row in theta_init]
for _ in range(ITERS_D):
    batch_s = [sample_teacher() for _ in range(G_D)]
    pi_s, _ = policy_tables(theta_sft)
    grad = [[0.0] * V for _ in range(T)]
    for y in batch_s:
        for t in range(T):
            yt = y[t]
            for v in range(V):
                grad[t][v] += (1.0 if v == yt else 0.0) - pi_s[t][v]
    for t in range(T):
        for v in range(V):
            theta_sft[t][v] += LR_D * grad[t][v] / G_D

mass_sft = mode_mass(theta_sft)
dwell_sft = dwell_rate(theta_sft)
mass_ratio = mass_opd / mass_sft
print(f"    [D1] 终局对照（同初始化/同 {ITERS_D} 步/同 lr）:")
print(f"         opd_seq(自采样+序列级adv): 背书 {endorse0:.4f}→{endorse_opd:.4f} nats/token"
      f" | 模质量 {mass0:.6f}→{mass_opd:.4f} | 驻留率 = {dwell_opd:.3f}")
print(f"         opd_tok(自采样+逐token adv): 模质量 →{mass_tok:.6f} | 驻留率 = {dwell_tok:.3f}")
print(f"         sft(教师采样+NLL):          模质量 →{mass_sft:.6f} | 驻留率 = {dwell_sft:.3f}")
print(f"         模质量比 opd_seq/sft = {mass_ratio:.0f}×")
print(f"    实测发现（配方×容量共轭）: 逐 token 配方在无上下文学生上收敛到 mode-covering")
print(f"    不动点（模质量与 sft 同量级）——TM 配方的锁模预设学生本身是序列模型")
print(f"    （nano-opd L1 的真实 Transformer 学生即如此，token 级 OPD 锁 codebook）。")
DIGEST["d0_endorse0"] = round(endorse0, 6)
DIGEST["d0_endorse"] = round(endorse_opd, 6)
DIGEST["d0_mass"] = round(mass_opd, 6)
DIGEST["d0_dwell"] = round(dwell_opd, 6)
DIGEST["d0_mass_tok"] = round(mass_tok, 6)
DIGEST["d1_mass_sft"] = round(mass_sft, 6)
check("D0a opd_seq 锁定教师的一个模（模质量 > 0.2）", mass_opd > 0.2, f"mass={mass_opd:.4f}")
check("D0b opd_seq 采样驻留在模上（驻留率 > 0.5）", dwell_opd > 0.5, f"dwell={dwell_opd:.3f}")
check("D0c opd_seq 教师背书度大幅抬升（> +1.5 nats/token）", endorse_opd > endorse0 + 1.5,
      f"{endorse0:.4f}→{endorse_opd:.4f}")
check("D1a off-policy SFT 掉谷：opd_seq/sft 模质量比 > 30×", mass_ratio > 30,
      f"{mass_ratio:.0f}x")
check("D1b 逐 token 变体在无上下文学生上不锁模（实测，模质量 < 0.05）", mass_tok < 0.05,
      f"mass_tok={mass_tok:.6f}")

# [D2] 锁模点上的估计器算术——nano-opd L0 驻点算术（+0.002 vs +136.149）的序列级
#      同源对照。缩小到 V=3、T=6（全支撑 729 条可逐条精确求和，零采样）：
V2, T2 = 3, 6
MA2 = [0, 1, 0, 1, 0, 1]
MB2 = [1, 0, 1, 0, 1, 0]


def teacher2_logprob(y):
    """缩小版自回归双模教师的精确序列概率（同 [D] 条件规则）。"""
    lp = 0.0
    d = 0
    for t in range(T2):
        if d > 0:
            pa, pb, pr = 0.9, 0.02, 0.08
        elif d < 0:
            pa, pb, pr = 0.02, 0.9, 0.08
        else:
            pa, pb, pr = 0.45, 0.45, 0.10
        yt = y[t]
        lp += math.log(pa if yt == MA2[t] else (pb if yt == MB2[t] else pr))
        if yt == MA2[t]:
            d += 1
        elif yt == MB2[t]:
            d -= 1
    return lp


# 全支撑枚举（3^6 = 729 条）：每条序列的精确 p 与 token 路径
SUPPORT = []
for code in range(V2 ** T2):
    y, c = [], code
    for _ in range(T2):
        y.append(c % V2)
        c //= V2
    SUPPORT.append((y, teacher2_logprob(y)))


def exact_kl_and_grads(theta2):
    """全支撑精确：KL(q||p)、真梯度（学生采样期望）、错估梯度（教师采样期望）。"""
    _, lpi2 = policy_tables(theta2)
    kl = 0.0
    g_right = [0.0] * (T2 * V2)
    g_wrong = [0.0] * (T2 * V2)
    for y, lp in SUPPORT:
        lq = sum(lpi2[t][y[t]] for t in range(T2))
        q = math.exp(lq)
        p = math.exp(lp)
        w = lq - lp
        kl += q * w
        for t in range(T2):
            qt = [math.exp(x) for x in lpi2[t]]
            for v in range(V2):
                s = (1.0 if v == y[t] else 0.0) - qt[v]
                g_right[t * V2 + v] += q * s * w
                g_wrong[t * V2 + v] += p * s * w
    return kl, g_right, g_wrong


# 精确梯度下降找 reverse KL 在分解式学生族里的最优点（噪声初始化破缺对称）
theta2 = [[0.3 * (rng.random() - 0.5) for _ in range(V2)] for _ in range(T2)]
for _ in range(1200):
    _, g_right2, _ = exact_kl_and_grads(theta2)
    for t in range(T2):
        for v in range(V2):
            theta2[t][v] -= 0.1 * g_right2[t * V2 + v]

kl_lock, g_right_lock, g_wrong_lock = exact_kl_and_grads(theta2)
n_right, n_wrong = flat_norm(g_right_lock), flat_norm(g_wrong_lock)
bias_ratio = n_wrong / n_right
_, lpi2_lock = policy_tables(theta2)
locked_a2 = sum(lpi2_lock[t][MA2[t]] for t in range(T2)) > sum(lpi2_lock[t][MB2[t]] for t in range(T2))
valley_tok2 = MB2 if locked_a2 else MA2
valley_grad2 = sum(g_wrong_lock[t * V2 + valley_tok2[t]] for t in range(T2))
mass_lock = max(math.exp(sum(lpi2_lock[t][MA2[t]] for t in range(T2))),
                math.exp(sum(lpi2_lock[t][MB2[t]] for t in range(T2))))

# 对照点：逐位置边缘匹配（mode covering 的分解式解）= SFT 的极限
marg2 = [[0.0] * V2 for _ in range(T2)]
for y, lp in SUPPORT:
    p = math.exp(lp)
    for t in range(T2):
        marg2[t][y[t]] += p
theta_marg = [[math.log(max(marg2[t][v], 1e-12)) for v in range(V2)] for t in range(T2)]
kl_marg, _, _ = exact_kl_and_grads(theta_marg)
kl_ratio = kl_marg / kl_lock

print(f"    [D2] 缩小版（V={V2}, T={T2}，全支撑 {V2 ** T2} 条精确求和）:")
print(f"         reverse KL 最优点 q*（锁{'A' if locked_a2 else 'B'}）: 模质量 = {mass_lock:.4f}"
      f" | KL(q*||p) = {kl_lock:.4f}")
print(f"         对照：边缘匹配解（SFT 极限）KL = {kl_marg:.4f}"
      f"  (reverse KL 偏好锁模解 {kl_ratio:.1f}×)")
print(f"         锁模点上: |g_right(学生采样)| = {n_right:.6f}（近驻点，模待得住）")
print(f"                   |g_wrong(教师采样)| = {n_wrong:.4f}（偏差比 = {bias_ratio:.0f}×）")
print(f"         g_wrong 在谷方向({('B' if locked_a2 else 'A')}模 token)分量和 = {valley_grad2:.3f}"
      f"（负 = 梯度下降会把质量推向另一个模 = mode covering → 掉谷）")
DIGEST["d2_kl_lock"] = round(kl_lock, 6)
DIGEST["d2_kl_marg"] = round(kl_marg, 6)
DIGEST["d2_n_right"] = round(n_right, 6)
DIGEST["d2_n_wrong"] = round(n_wrong, 6)
DIGEST["d2_valley_grad"] = round(valley_grad2, 4)
check("D2a reverse KL 在受限学生族里的最优解是锁模（模质量 > 0.5）", mass_lock > 0.5,
      f"mass={mass_lock:.4f}")
check("D2b reverse KL 偏好锁模解而非边缘匹配（KL 比 > 3×）", kl_ratio > 3,
      f"{kl_marg:.3f} vs {kl_lock:.3f}")
check("D2c 正确估计器在锁模点近驻点（|g| < 0.01）", n_right < 0.01, f"|g|={n_right:.6f}")
check("D2d 教师采样估计器有系统性大偏差（>10× 真梯度）", bias_ratio > 10,
      f"{bias_ratio:.0f}x")
check("D2e 偏差方向指向谷（mode covering，分量和 < −0.1）", valley_grad2 < -0.1,
      f"valley_grad={valley_grad2:.3f}")

# ======================================================================
# [E] self-check 汇总 + digest
# ======================================================================
print()
print("[E] self-check")
n_pass = sum(1 for _, ok in CHECKS if ok)
for name, ok in CHECKS:
    if not ok:
        print(f"    FAIL  {name}")
ok_all = n_pass == len(CHECKS)
print(f"    {'✅' if ok_all else '❌'} self-check {'passed' if ok_all else 'FAILED'}"
      f" ({n_pass}/{len(CHECKS)})")

DIGEST["n_pass"] = n_pass
DIGEST["n_checks"] = len(CHECKS)
digest = hashlib.md5(repr(sorted(DIGEST.items())).encode()).hexdigest()
print(f"\ndigest(md5 of metrics) = {digest}")
if not ok_all:
    sys.exit(1)
