#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kimi_k3_agentic_rl_sim.py — Kimi K3 agentic RL 规模化的四机制面本质模拟
=======================================================================

定位（可运行性契约）：
  这是 01 轨 sota-deepdive「Kimi-K3 agentic RL 规模化」的 native sim，
  对齐 02/03 轨 deepdive 形态（deepseek_v3_mechanisms_sim.py /
  data_methodology_sim.py）与 01 轨既有 post_training_evolution_sim.py。
  toy 尺度、纯标准库（math/random/hashlib）、seed 固定、无计时行、
  跨运行逐字节一致。全部数字为本 sim 实测或显式声明的 declared 折算，
  论文侧数字只作机制对照、不外推量级。

机制回声对象（全部一手来源现场核验，2026-08-15，详见 deepdive §8 溯源表）：
  [A] partial rollout 与 straggler-staleness 权衡
      —— Kimi K3 [2607.24653] §4.1.2「Algorithm」：N×K 轨迹池，完成比例达
         λ∈(0,1) 即暂停生成、推进训练；暂停轨迹下一迭代优先续跑。
         「an individual long-horizon trajectory naturally spans multiple
         iterations, introducing data staleness」（逐字）。
  [B] per-token 正则为什么能容忍极端 off-policy（三小面）
      —— K3 的 policy optimization「follows the algorithm in Kimi K2.5
         [2602.02276]」（逐字）；K2.5 Eq.1 = token 级 ratio 窗口 Clip(r,α,β)
         （log-ratio 出窗即梯度掩码，**与 advantage 符号无关**，区别于 PPO
         clip 的符号依赖）+ 每 token 平方 log-ratio 正则 −τ(log r)²。
         K3 称之为「per-token regularization ... constraining policy
         updates within a localized neighborhood」（逐字）。
         [B1] 梯度通图 / [B2] 最坏情形梯度界 / [B3] advantage 符号翻转
         压力测试（陈旧到符号不可信的极端 off-policy 形态）。
  [C] 预算控制与 verbosity：reward hacking → 硬预算掉坑 → Toggle 交替
      —— K3 §4.1.2「Reasoning Effort RL」：per-problem budget b_0(x)，
         T(y) > τ·b_0(x) 时 task reward 覆写为 −1；agentic 任务 T(y) 计
         「cumulative output tokens, including both reasoning traces and
         tool-call arguments」（逐字）。GRM 侧「output length exceeds
         σ·ℓ_0 automatically loses the binary comparison」（逐字）。
         Toggle 来自 K2.5 §4.4.2「Token Efficient RL」：Phase0/Phase1 交替，
         修复「length-overfitting phenomenon」（逐字）。
  [D] sandbox pause/resume 经济学（declared 折算，非实测）
      —— K3 §5.3.2 AgentENV：增量 checkpoint/resume 低至 133 ms / 49 ms；
         等待模型推理「can account for as much as 98% of the sandbox
         lifetime」；内存超卖「up to 6.5×」（均逐字，declared 参数入算）。

运行：python3 -B kimi_k3_agentic_rl_sim.py
      （Python 3.10+；任意空 CWD 可跑，零外部依赖，<10 s）
"""

import math
import random
import hashlib

SEED = 20260815
V = 5          # [B] toy 动作词表
T_LEN = 6      # [B] toy 序列长度（分解式策略，每位置一个 softmax）


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def kl_cat(p, q):
    """分类分布 KL(p||q)，逐位置求和（分解式序列 = 各位置独立）。"""
    tot = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            tot += pi * math.log(pi / qi)
    return tot


# =====================================================================
# [A] partial rollout：straggler 等待 vs 数据陈旧度
# =====================================================================
#
# 模型（显式声明的理想化）：
#   - N 个 prompt × K 条补全 = N*K 条轨迹，活跃工作负载恒定 N*K
#     （K3 逐字：「maintaining an active workload of N×K trajectories」）。
#   - 每条轨迹的持续时长 D（工具调用步数）服从重尾分布——agentic 任务
#     长尾是 K3「long-tail latency that intensifies in long-horizon tasks」
#     （逐字）的 toy 形态：D = Geom(p) 截断，seed 固定。
#   - 时间以 slot 计：每 slot 所有未完成轨迹各推进一步（并行生成）。
#     **生成总时长与 λ 无关**（= max D）——partial rollout 不加速生成，
#     它解耦的是「训练何时开始」：完成数达 ⌈λNK⌉ 即触发训练（迭代边界），
#     其余轨迹跨边界续跑（K3：「Paused rollouts are enqueued and
#     prioritized for resumption at the start of the next iteration」）。
#   - 陈旧度（迭代单位，token 加权）：token 的生成迭代号与所属轨迹被
#     训练消费的迭代号之差。K3 消费规则「Once all K responses for a
#     prompt complete, they are immediately dispatched」；toy 简化为按
#     轨迹粒度消费（prompt 聚合只改常数、不改机制方向，显式声明）。
#   - 尾部 flush：最后不足 ⌈λNK⌉ 的已完成轨迹触发一次收尾训练。
#
# 不建模：真实 KV-cache 命中/抢占、权重更新耗时、sandbox 启停延迟
# （[D] 单独以 declared 折算处理）。

def sample_durations(rng, n_traj, p_done=0.22, cap=24):
    """重尾持续时长：每步以 p_done 结束，否则继续，截断于 cap。"""
    ds = []
    for _ in range(n_traj):
        d = 1
        while d < cap and rng.random() > p_done:
            d += 1
        ds.append(d)
    return ds


def simulate_partial_rollout(n_prompts, k_per_prompt, lam, durations):
    """
    返回 dict：total_slots / first_train_slot / train_steps /
    mean_stale（token 加权，迭代单位）/ span2（跨≥2迭代轨迹占比）。
    """
    n_traj = n_prompts * k_per_prompt
    assert len(durations) == n_traj
    need = max(1, math.ceil(lam * n_traj))
    remain = list(durations)
    gen_iter = [[] for _ in range(n_traj)]   # 每 token 的生成迭代号
    done_at = [None] * n_traj                # 轨迹完成时刻（slot）
    consume_iter = [None] * n_traj           # 被训练消费的迭代号
    it = 0
    slots = 0
    completed_unconsumed = 0
    first_train_slot = None
    train_steps = 0
    while any(r > 0 for r in remain):
        for i in range(n_traj):
            if remain[i] > 0:
                gen_iter[i].append(it)
                remain[i] -= 1
                if remain[i] == 0:
                    done_at[i] = slots
                    completed_unconsumed += 1
        slots += 1
        if completed_unconsumed >= need:
            if first_train_slot is None:
                first_train_slot = slots
            train_steps += 1
            for i in range(n_traj):
                if done_at[i] is not None and consume_iter[i] is None:
                    consume_iter[i] = it
            completed_unconsumed = 0
            it += 1
    if any(c is None for c in consume_iter):   # 尾部 flush（显式声明）
        if first_train_slot is None:
            first_train_slot = slots
        train_steps += 1
        for i in range(n_traj):
            if consume_iter[i] is None:
                consume_iter[i] = it
        it += 1
    gaps = []
    span2 = 0
    for i in range(n_traj):
        if len(set(gen_iter[i])) >= 2:
            span2 += 1
        for g in gen_iter[i]:
            gaps.append(consume_iter[i] - g)
    return {"total_slots": slots,
            "first_train_slot": first_train_slot,
            "train_steps": train_steps,
            "mean_stale": sum(gaps) / len(gaps),
            "span2": span2 / n_traj}


def run_A():
    print("[A] partial rollout：straggler 等待 vs 数据陈旧度")
    print("    toy: N=24 prompt × K=4 = 96 轨迹，重尾时长（Geom p=0.22 截断 24）")
    print("    机制回声: K3 §4.1.2 λ 暂停-续跑；陈旧度 = 消费迭代 − 生成迭代")
    rng = random.Random(SEED + 1)
    durations = sample_durations(rng, 96)
    d_sorted = sorted(durations)
    print(f"    时长分布: min={d_sorted[0]} median={d_sorted[48]} "
          f"p90={d_sorted[86]} max={d_sorted[-1]}"
          f"（重尾：max/median = {d_sorted[-1]/d_sorted[48]:.1f}×）")
    rows = []
    for lam in (1.0, 0.75, 0.5, 0.25):
        r = simulate_partial_rollout(24, 4, lam, durations)
        rows.append((lam, r))
        print(f"    λ={lam:<4}: 首次训练@slot {r['first_train_slot']:<3} "
              f"训练步数={r['train_steps']:<2} 平均陈旧度={r['mean_stale']:.3f} "
              f"迭代 跨≥2迭代轨迹占比={r['span2']:.3f}")
    sync, p75, p50, p25 = rows[0][1], rows[1][1], rows[2][1], rows[3][1]
    print(f"    生成总时长 = max D = {sync['total_slots']} slot，与 λ 无关"
          f"（四 λ 逐位同一={sync['total_slots'] == p75['total_slots'] == p50['total_slots'] == p25['total_slots']}）")
    print(f"    λ=0.25 vs 同步: 首次训练 {sync['first_train_slot']}→"
          f"{p25['first_train_slot']} slot，训练步数 {sync['train_steps']}→"
          f"{p25['train_steps']}，平均陈旧度 {sync['mean_stale']:.3f}→"
          f"{p25['mean_stale']:.3f}")
    checks = []
    checks.append(("A1 同步 λ=1 陈旧度恰为 0（无轨迹跨迭代）",
                   abs(sync["mean_stale"]) < 1e-12 and sync["span2"] == 0.0,
                   f"stale={sync['mean_stale']}, span2={sync['span2']}"))
    checks.append(("A2 λ 越小首次训练越早（训练不再等 straggler）",
                   sync["first_train_slot"] > p75["first_train_slot"] >
                   p50["first_train_slot"] > p25["first_train_slot"],
                   f"first_train={[r['first_train_slot'] for _, r in rows]}"))
    checks.append(("A3 λ 越小同窗内训练步数越多（训练频率↑）",
                   sync["train_steps"] < p75["train_steps"] <=
                   p50["train_steps"] <= p25["train_steps"],
                   f"steps={[r['train_steps'] for _, r in rows]}"))
    checks.append(("A4 λ 越小平均陈旧度越高（权衡成立）",
                   sync["mean_stale"] < p75["mean_stale"] <
                   p50["mean_stale"] < p25["mean_stale"],
                   f"stale={[round(r['mean_stale'], 3) for _, r in rows]}"))
    checks.append(("A5 λ=0.25 时多数轨迹跨迭代（占比 >0.5）",
                   p25["span2"] > 0.5, f"span2={p25['span2']:.3f}"))
    checks.append(("A6 生成总时长与 λ 无关（partial rollout 解耦训练而非加速生成）",
                   sync["total_slots"] == p75["total_slots"] ==
                   p50["total_slots"] == p25["total_slots"],
                   f"total_slots={sync['total_slots']}"))
    return checks, {"A_first_train": [r["first_train_slot"] for _, r in rows],
                    "A_steps": [r["train_steps"] for _, r in rows],
                    "A_stale": [round(r["mean_stale"], 4) for _, r in rows]}


# =====================================================================
# [B] per-token 正则：log-ratio 窗口裁剪的符号无关性
# =====================================================================
#
# K2.5 Eq.1（K3 沿用）：
#   L = E[ (1/N) Σ_t Clip(r_t, α, β)·A_t − τ·(log r_t)² ]
#   r_t = π_θ(y_t|...)/π_old(y_t|...)，Clip 为 ratio 窗口 [α,β] 截断，
#   出窗 token 的 policy gradient 被掩码（梯度置零）——报告原文：
#   「policy gradients are computed normally for tokens with log-ratios
#   within the interval [α,β], while gradients for tokens falling outside
#   this range are zeroed out ... regardless of the sign of the
#   advantages」（K2.5 §4.4.2 逐字）。
#
# PPO clip 的符号依赖（对照）：
#   L = min(r·A, clip(r,1−ε,1+ε)·A)
#   A>0 时只封 r>1+ε 一侧（r<1−ε 侧梯度照传）；A<0 时只封 r<1−ε 一侧。
#   即 PPO 只封「对 advantage 不利方向出窗」的那半边。
#
# [B1] 梯度通图：dL/d(log r) 在 log-ratio 网格 × {A=+1, A=−1} 上的形态。
# [B2] 最坏情形梯度界：对 A 符号取 max 后的 |dL/dlogr|——符号无关的
#      定量形态 = 最坏情形界相同（PPO 有 ∝r 的无界边，K2.5 全网格有界）。
# [B3] 陈旧批次多步训练 + advantage 符号翻转压力测试：KL(π_θ||π_old)
#      轨迹——per-token 正则把策略锁在邻域内，且界不随 A 符号翻转。

ALPHA, BETA, TAU_REG = 0.75, 1.35, 0.08    # k25 toy 超参（显式声明）
EPS_PPO = 0.2


def loss_and_dlogr(logp_theta, logp_old, A, loss):
    """
    单 token 的 surrogate 损失与对 log-ratio 的梯度 dL/d(log r)。
    （分解式 softmax 下 logit 梯度 = dL/d(logr) · (1_{a} − p_θ)，
    [B1] 通图只看 dL/d(logr) 的量级与符号结构。）
    loss: 'pg' 裸策略梯度 | 'ppo' | 'k25'（窗口掩码 + 平方正则）
    约定：对 θ 做梯度下降，L 中策略梯度项取负号（最大化 A·r）。
    """
    logr = logp_theta - logp_old
    r = math.exp(logr)
    if loss == "pg":
        return -A * r, -A * r
    if loss == "ppo":
        rc = min(max(r, 1 - EPS_PPO), 1 + EPS_PPO)
        surr = min(r * A, rc * A)
        if r * A <= rc * A:      # 未截断支路被 min 选中 → 梯度照传
            return -surr, -A * r
        return -surr, 0.0
    # k25：窗口内传 A 项，窗口外掩码；平方正则恒传（锚向 behavior）
    g = (-A * r) if (ALPHA <= r <= BETA) else 0.0
    g += 2 * TAU_REG * logr
    surr = (-A * r if ALPHA <= r <= BETA else 0.0) + TAU_REG * logr * logr
    return surr, g


def run_B1():
    """梯度通图：log-ratio 网格上三种 loss 的 dL/d(logr)。"""
    grid = [round(x * 0.25, 4) for x in range(-8, 9)]   # −2 … +2，17 点
    signed = {}
    for loss in ("pg", "ppo", "k25"):
        for A in (+1.0, -1.0):
            signed[(loss, A)] = [loss_and_dlogr(lr, 0.0, A, loss)[1]
                                 for lr in grid]
    print("[B] per-token 正则：log-ratio 窗口裁剪为什么符号无关")
    print("    [B1] 梯度通图（|dL/dlogr|，log-ratio 网格 −2…+2，步长 0.25）")
    for loss in ("pg", "ppo", "k25"):
        line_p = " ".join(f"{abs(g):6.2f}" for g in signed[(loss, +1.0)][::2])
        line_n = " ".join(f"{abs(g):6.2f}" for g in signed[(loss, -1.0)][::2])
        print(f"    {loss:>3} A=+1: {line_p}")
        print(f"    {loss:>3} A=−1: {line_n}")
    checks = []
    ppo_p_left = abs(signed[("ppo", +1.0)][0])    # logr=−2, r≈0.135
    ppo_p_right = abs(signed[("ppo", +1.0)][-1])  # logr=+2, r≈7.39
    checks.append(("B1a PPO A=+1 只封右尾（左尾梯度照传、右尾掩码）",
                   ppo_p_left > 0.1 and ppo_p_right == 0.0,
                   f"left={ppo_p_left:.3f}, right={ppo_p_right:.3f}"))
    ppo_n_left = abs(signed[("ppo", -1.0)][0])
    ppo_n_right = abs(signed[("ppo", -1.0)][-1])
    checks.append(("B1b PPO A=−1 只封左尾（封的半边随 A 符号翻转）",
                   ppo_n_left == 0.0 and ppo_n_right > 0.1,
                   f"left={ppo_n_left:.3f}, right={ppo_n_right:.3f}"))
    k25_p = [abs(g) for g in signed[("k25", +1.0)]]
    k25_n = [abs(g) for g in signed[("k25", -1.0)]]
    checks.append(("B1c K2.5 两端出窗后只剩平方正则（两端对称 = 2τ|logr|）",
                   abs(k25_p[0] - k25_p[-1]) < 1e-9 and
                   abs(k25_p[0] - 2 * TAU_REG * 2.0) < 1e-9,
                   f"ends={k25_p[0]:.3f}/{k25_p[-1]:.3f}, "
                   f"2τ|logr|={2 * TAU_REG * 2.0:.3f}"))
    sg_p = signed[("k25", +1.0)][8]    # logr=0（窗口内）
    sg_n = signed[("k25", -1.0)][8]
    checks.append(("B1d K2.5 窗口内 A 项照传（logr=0 处 signed = −A）",
                   abs(sg_p + 1.0) < 1e-9 and abs(sg_n - 1.0) < 1e-9,
                   f"A=+1:{sg_p:.3f} / A=−1:{sg_n:.3f}"))
    asym_ppo = abs((abs(signed[('ppo', +1.0)][0]) +
                    abs(signed[('ppo', +1.0)][-1])) -
                   (abs(signed[('ppo', -1.0)][0]) +
                    abs(signed[('ppo', -1.0)][-1])))
    sym_k25 = abs((k25_p[0] + k25_p[-1]) - (k25_n[0] + k25_n[-1]))
    checks.append(("B1e 符号无关性总账：K2.5 两符号通图逐位重合、PPO 不重合",
                   sym_k25 < 1e-9 and asym_ppo > 1.0,
                   f"k25 两符号差={sym_k25:.2e}, ppo 两符号差={asym_ppo:.2f}"))
    return checks, {"B1_ppo_asym": round(asym_ppo, 4),
                    "B1_k25_sym": round(sym_k25, 9)}


def worst_grad_map():
    """G_worst(logr) = max over A∈{+1,−1} of |dL/dlogr|——对 advantage
    符号取最坏情形后的单 token 梯度幅度（「regardless of the sign of
    the advantages」的定量形态：符号无关 = 最坏情形界相同）。"""
    grid = [round(x * 0.25, 4) for x in range(-8, 9)]
    maps = {}
    for loss in ("pg", "ppo", "k25"):
        maps[loss] = [max(abs(loss_and_dlogr(lr, 0.0, +1.0, loss)[1]),
                          abs(loss_and_dlogr(lr, 0.0, -1.0, loss)[1]))
                      for lr in grid]
    return grid, maps


def run_B2():
    print("    [B2] 最坏情形梯度界（对 A 符号取 max 的 |dL/dlogr|）")
    grid, maps = worst_grad_map()
    for loss in ("pg", "ppo", "k25"):
        line = " ".join(f"{g:6.2f}" for g in maps[loss][::2])
        print(f"    {loss:>3}: {line}")
    checks = []
    ppo_tail = max(maps["ppo"])
    k25_bound = max(maps["k25"])
    pg_tail = max(maps["pg"])
    print(f"    网格最大值: pg={pg_tail:.2f}（∝r 无界） ppo={ppo_tail:.2f}"
          f"（无界边） k25={k25_bound:.2f}（有界）")
    checks.append(("B2a PPO 最坏梯度随 |logr| 无界增长（= r，符号依赖漏边）",
                   ppo_tail > 5.0 and abs(ppo_tail - math.exp(2.0)) < 1e-6,
                   f"max={ppo_tail:.3f} = e^2={math.exp(2.0):.3f}"))
    checks.append(("B2b K2.5 最坏梯度全网格有界（≤ 窗口内 A 项 + 正则）",
                   k25_bound < 1.6,
                   f"max={k25_bound:.3f}（界内上界 β+2τ|logr_max|="
                   f"{BETA + 2 * TAU_REG * 2.0:.3f}）"))
    checks.append(("B2c 有界性差距 >4×（同一网格、同一最坏口径）",
                   ppo_tail / k25_bound > 4.0,
                   f"ratio={ppo_tail / k25_bound:.2f}"))
    return checks, {"B2_worst": {"pg": round(pg_tail, 4),
                                  "ppo": round(ppo_tail, 4),
                                  "k25": round(k25_bound, 4)}}


def train_on_stale_batch(loss, n_steps=60, lr=0.06, n_traj=400,
                         flip_adv=False, seed=SEED + 2):
    """
    固定 behavior 策略 π_b 采的旧批次，反复训练 n_steps 步。
    toy 策略 = 每位置独立 softmax（V=5，T_LEN=6），reward 为逐 token
    可加的固定函数 f（ground truth），advantage A_t = f_t − mean(f)。
    flip_adv=True：A → −A，模拟「陈旧到 advantage 符号都不可信」的
    极端 off-policy 压力测试（partial rollout 跨多迭代续跑的极端形态）。
    度量 KL(π_θ||π_old)——离开采样分布的速度。
    """
    rng = random.Random(seed)
    f = [[rng.uniform(-1, 1) for _ in range(V)] for _ in range(T_LEN)]
    logits_b = [[rng.uniform(-0.3, 0.3) for _ in range(V)]
                for _ in range(T_LEN)]
    probs_b = [softmax(row) for row in logits_b]
    batch = []
    for _ in range(n_traj):
        seq, logps = [], []
        for t in range(T_LEN):
            u = rng.random()
            cum = 0.0
            act = V - 1
            for a in range(V):
                cum += probs_b[t][a]
                if u <= cum:
                    act = a
                    break
            seq.append(act)
            logps.append(math.log(probs_b[t][act]))
        batch.append((seq, logps))
    logits = [row[:] for row in logits_b]
    kl_traj = []
    for step in range(n_steps):
        grad = [[0.0] * V for _ in range(T_LEN)]
        for seq, logps_old in batch:
            fs = [f[t][seq[t]] for t in range(T_LEN)]
            mean_f = sum(fs) / T_LEN
            for t in range(T_LEN):
                A = fs[t] - mean_f
                if flip_adv:
                    A = -A
                p_now = softmax(logits[t])
                logp_now = math.log(p_now[seq[t]])
                _, dlogr = loss_and_dlogr(logp_now, logps_old[t], A, loss)
                for a in range(V):
                    ind = 1.0 if a == seq[t] else 0.0
                    grad[t][a] += dlogr * (ind - p_now[a]) / len(batch)
        for t in range(T_LEN):
            for a in range(V):
                logits[t][a] -= lr * grad[t][a]
        if step % 10 == 0 or step == n_steps - 1:
            p_now = [softmax(row) for row in logits]
            kl = sum(kl_cat(p_now[t], probs_b[t]) for t in range(T_LEN))
            kl_traj.append((step, kl))
    return kl_traj


def run_B3():
    print("    [B3] 陈旧批次多步训练（60 步，lr=0.06，400 条×6 token）")
    print("         压力测试：advantage 符号翻转（A→−A）= 陈旧到符号不可信")
    kl = {}
    for loss in ("pg", "ppo", "k25"):
        for flip in (False, True):
            kl[(loss, flip)] = train_on_stale_batch(loss, flip_adv=flip)
    for loss in ("pg", "ppo", "k25"):
        ln = " ".join(f"step{s}:{v:.3f}" for s, v in kl[(loss, False)][::3])
        print(f"    {loss:>3} 正常   KL: {ln}")
    for loss in ("pg", "ppo", "k25"):
        lf = " ".join(f"step{s}:{v:.3f}" for s, v in kl[(loss, True)][::3])
        print(f"    {loss:>3} 翻转   KL: {lf}")
    end = {(l, fl): kl[(l, fl)][-1][1] for l in ("pg", "ppo", "k25")
           for fl in (False, True)}
    ppo_amp = end[("ppo", True)] / max(end[("ppo", False)], 1e-12)
    k25_amp = end[("k25", True)] / max(end[("k25", False)], 1e-12)
    print(f"    翻转放大系数: pg={end[('pg', True)]/end[('pg', False)]:.2f} "
          f"ppo={ppo_amp:.2f} k25={k25_amp:.2f}")
    print("    口径说明: 本 toy 的正常漂移方向恰与 PPO 掩码边同向（掩码随 A "
          "符号旋转），故 ppo 翻转不放大——K2.5 的符号无关性是结构性保证"
          "（B1/B2 机器证明），在 advantage 符号相对漂移方向不可信的 regime"
          "（train-inference mismatch + 跨迭代陈旧）才成为必需；k25 正常漂移"
          f" {end[('k25', False)]:.3f} > ppo {end[('ppo', False)]:.3f} 系窗口"
          "更宽（[0.75,1.35] vs [0.8,1.2]）的超参权衡——窗口越宽传过的梯度"
          "越多、界越松，宽度是旋钮。")
    checks = []
    checks.append(("B3a 正常陈旧批次：裸 PG 漂移最大（无任何界）",
                   end[("pg", False)] > end[("ppo", False)] and
                   end[("pg", False)] > end[("k25", False)],
                   f"KL_end pg={end[('pg', False)]:.3f}"))
    checks.append(("B3b 裁剪族把陈旧漂移压在 0.25 nats 内（四种情形全满足）",
                   max(end[("ppo", False)], end[("k25", False)],
                       end[("ppo", True)], end[("k25", True)]) < 0.25,
                   f"max={max(end[('ppo', False)], end[('k25', False)], end[('ppo', True)], end[('k25', True)]):.3f}"
                   f" vs pg={end[('pg', False)]:.3f}"))
    checks.append(("B3c K2.5 翻转近似不变（界不随 A 符号翻，结构性保证）",
                   abs(k25_amp - 1.0) < 0.35, f"amp={k25_amp:.2f}"))
    checks.append(("B3d 极端 off-policy 下 k25 漂移 ≪ 裸 PG",
                   end[("k25", True)] < 0.6 * end[("pg", True)],
                   f"k25_flip={end[('k25', True)]:.3f} "
                   f"vs pg_flip={end[('pg', True)]:.3f}"))
    return checks, {"B3_kl_end": {f"{l}_flip{int(fl)}": round(v, 4)
                                   for (l, fl), v in end.items()},
                    "B3_amp": {"ppo": round(ppo_amp, 4),
                               "k25": round(k25_amp, 4)}}


# =====================================================================
# [C] 预算控制与 verbosity：reward hacking → 硬预算掉坑 → Toggle 交替
# =====================================================================
#
# toy 任务族：训练难度 d∈{easy, hard}。模型输出「思考长度」ℓ（token
# 数的 toy 形态：离散步数），成功概率随 ℓ 上升但饱和——easy 饱和快、
# hard 饱和慢（机制内核 = 边际收益递减；K3「fine-tune reasoning effort
# while maximizing token efficiency」的 toy 对应）：
#   P(success | ℓ, d) = 1 − exp(−ℓ / s_d)，s_easy=3, s_hard=12
# 策略参数 = 每难度一个目标长度 ℓ_π(d)，RL 用 reward 加权有限差分更新。
#
# 三臂对照（同初始化 ℓ=6、同 lr=1.2、同 30 轮、同 seed）：
#   free   : reward = success，无预算 → 长度膨胀（verbosity hacking）
#   budget : K3 硬预算——ℓ > τ_b·b_0 时 reward 覆写为 −1（无条件，
#            K3 §4.1.2 逐字「override the task reward with −1」）
#   toggle : K2.5 Toggle——Phase0（预算；组均 reward < λ 时豁免——
#            「模型还不会做时不施加预算」，K2.5 逐字条件项）与 Phase1
#            （自由）每 m=4 轮交替；λ=0.6（toy 超参，显式声明）
#
# 终局评测两口径：
#   - 训练难度放量评测（cap=40）
#   - **未见更难难度** s_harder=24 泛化评测——检验「length-overfitting」
#     （K2.5 逐字：硬预算训练的模型「fail to generalize to higher
#     compute scales ... defaulting to truncated reasoning patterns」）：
#     硬预算臂学会了「早停」，遇到需要更长的新题时放量也救不回来。

def p_success(ell, diff):
    s = {"easy": 3.0, "hard": 12.0, "harder": 24.0}[diff]
    return 1.0 - math.exp(-ell / s)


def rl_update_budget(arm, n_rounds=30, lr=1.2, tau_b=2.0, lam_tg=0.6,
                     m_toggle=4, cap=40.0, seed=SEED + 3):
    """arm ∈ {'free','budget','toggle'}。返回 (ell_easy, ell_hard) 终值。"""
    rng = random.Random(seed)
    ell = {"easy": 6.0, "hard": 6.0}
    b0 = {"easy": 6.0, "hard": 6.0}      # 预算基线 = cold-start 估计
    for rnd in range(n_rounds):
        phase0 = (arm == "budget") or (
            arm == "toggle" and (rnd // m_toggle) % 2 == 0)
        for diff in ("easy", "hard"):
            G = 32
            samples = []
            for _ in range(G):
                ell_i = max(1.0, min(cap, ell[diff] + rng.gauss(0, 2.0)))
                r = 1.0 if rng.random() < p_success(ell_i, diff) else 0.0
                samples.append((ell_i, r))
            mean_r = sum(r for _, r in samples) / G
            gated = []
            for ell_i, r in samples:
                # K3 覆写：超预算 → −1；K2.5 Phase0 豁免条件：组均 < λ
                if phase0 and ell_i > tau_b * b0[diff]:
                    if not (arm == "toggle" and mean_r < lam_tg):
                        r = -1.0
                gated.append((ell_i, r))
            base = sum(r for _, r in gated) / G
            ws = [r - base for _, r in gated]
            num = sum(w * (e - ell[diff]) for (e, _), w in zip(gated, ws))
            den = sum(abs(w) for w in ws) + 1e-9
            ell[diff] = max(1.0, min(cap, ell[diff] + lr * num / den))
    return ell["easy"], ell["hard"]


def run_C():
    print("[C] 预算控制与 verbosity：reward hacking → 硬预算掉坑 → Toggle")
    print("    toy: P(success|ℓ,d) = 1−exp(−ℓ/s_d)，s_easy=3 / s_hard=12")
    print("    三臂同初始化 ℓ=6、同 lr=1.2、同 30 轮；预算 τ_b=2.0×b_0=12")
    arms = {}
    for arm in ("free", "budget", "toggle"):
        arms[arm] = rl_update_budget(arm)
        pe = p_success(arms[arm][0], "easy")
        ph = p_success(arms[arm][1], "hard")
        print(f"    {arm:>6}: 终局 ℓ_easy={arms[arm][0]:6.2f} "
              f"ℓ_hard={arms[arm][1]:6.2f} | 放量成功率 P(easy)={pe:.3f} "
              f"P(hard)={ph:.3f}")
    ev = {a: (p_success(arms[a][0], "easy"), p_success(arms[a][1], "hard"))
          for a in arms}
    # 未见更难难度泛化评测（length-overfitting 探针）
    evh = {a: p_success(arms[a][1], "harder") for a in arms}
    line = " ".join(f"{a}={evh[a]:.3f}" for a in arms)
    print(f"    未见更难题（s=24）泛化: {line}（硬预算臂早停习惯的代价）")
    checks = []
    checks.append(("C1 free 臂 easy 题长度膨胀（verbosity hacking）",
                   arms["free"][0] > 1.8 * 6.0,
                   f"ell_easy={arms['free'][0]:.2f} vs 初始 6"))
    checks.append(("C2 硬预算臂 easy 题长度被压在预算附近（≤2.2×b_0）",
                   arms["budget"][0] <= 2.2 * 6.0,
                   f"ell_easy={arms['budget'][0]:.2f}（预算 12）"))
    checks.append(("C3 硬预算臂 hard 题掉坑（length-overfitting）",
                   ev["budget"][1] < ev["free"][1] - 0.05,
                   f"P(hard) budget={ev['budget'][1]:.3f} "
                   f"vs free={ev['free'][1]:.3f}"))
    checks.append(("C4 Toggle 两全：easy 比 free 省、hard 比 budget 强",
                   arms["toggle"][0] < arms["free"][0] and
                   ev["toggle"][1] > ev["budget"][1] + 0.05,
                   f"ell_easy={arms['toggle'][0]:.2f}"
                   f"<{arms['free'][0]:.2f}, P(hard)="
                   f"{ev['toggle'][1]:.3f}>{ev['budget'][1]:.3f}"))
    checks.append(("C5 未见更难题：Toggle 泛化 > 硬预算（早停代价）",
                   evh["toggle"] > evh["budget"] + 0.05,
                   f"harder: toggle={evh['toggle']:.3f} "
                   f"vs budget={evh['budget']:.3f}"))
    # GRM verbosity 自动判负（K3 §4.1.2 逐字机制的算术形态）
    sigma, ell0 = 1.5, 100
    thr = sigma * ell0
    q_a, ell_a, q_b = 0.82, 170, 0.80   # A 质量胜但超长
    winner_plain = "A" if q_a > q_b else "B"
    auto_lose = ell_a > thr
    winner_k3 = "B" if (auto_lose and winner_plain == "A") else winner_plain
    print(f"    GRM 判负探针: ℓ_0={ell0}, σ={sigma} → 阈值 {thr:.0f}；"
          f"候选 A(q={q_a},ℓ={ell_a}) 质量胜但超长 → 自动判负={auto_lose}，"
          f"胜者 {winner_plain}→{winner_k3}")
    checks.append(("C6 GRM verbosity 控制：超长候选质量胜也自动判负",
                   auto_lose and winner_k3 == "B",
                   f"threshold={thr:.0f}, ell_A={ell_a}"))
    return checks, {"C_ell": {a: [round(x, 3) for x in arms[a]]
                              for a in arms},
                    "C_Phard": {a: round(ev[a][1], 4) for a in arms},
                    "C_Pharder": {a: round(evh[a], 4) for a in arms}}


# =====================================================================
# [D] sandbox pause/resume 经济学（declared 折算，非实测）
# =====================================================================
#
# K3 §5.3.2 逐字数字作 declared 参数（AgentENV 开源仓库 README 独立
# 在盘，resume <50 ms / pause <100 ms / 增量快照 <100 ms 口径与报告
# 133/49 ms 同族）：
#   - 等待模型推理占 sandbox 寿命至多 98%（报告原文「can account for
#     as much as 98% of the sandbox lifetime」——取上界入算并声明）
#   - 暂停的 sandbox「consumes no memory or CPU resources」（逐字）
#   - 增量 checkpoint 133 ms / resume 49 ms（「as low as」口径入算）
# 问题：一条寿命 L 秒的 agentic 轨迹，sandbox 实际占用多少「资源·秒」？
#   无 pause: 占用 = L（全程驻留）
#   有 pause: 占用 = 活跃段 + (checkpoint+resume) × 切换次数
# 推论量：同一物理池的超卖比。与报告实测 6.5× 的差异显式解释。

def run_D():
    print("[D] sandbox pause/resume 经济学（declared 折算，K3 §5.3.2 参数）")
    wait_frac = 0.98                   # K3 逐字上界
    ckpt_s, resume_s = 0.133, 0.049    # K3「as low as」口径
    L = 3600.0                         # 一条长时程轨迹寿命 1h（toy 口径）
    n_switch = 40                      # pause/resume 切换次数
    active = L * (1 - wait_frac)
    overhead = n_switch * (ckpt_s + resume_s)
    hold_no_pause = L
    hold_with_pause = active + overhead
    overcommit = hold_no_pause / hold_with_pause
    print(f"    轨迹寿命 L={L:.0f}s，等待占比 {wait_frac:.0%}，"
          f"切换 {n_switch} 次 × (133+49)ms")
    print(f"    无 pause: 占用 {hold_no_pause:.0f} 资源·s | "
          f"有 pause: 活跃 {active:.1f} + 切换开销 {overhead:.2f} = "
          f"{hold_with_pause:.2f} 资源·s")
    print(f"    超卖比 = {overcommit:.1f}×（98% 上界口径；K3 报告实测 up to "
          f"6.5×——真实负载等待占比低于上界，故实测更低，方向一致）")
    checks = []
    checks.append(("D1 pause/resume 切换税 <1%（开销占比可忽略）",
                   overhead / L < 0.01,
                   f"overhead={overhead:.2f}s / {L:.0f}s"))
    checks.append(("D2 超卖比 >10×（98% 上界口径，机制方向坐实）",
                   overcommit > 10.0, f"overcommit={overcommit:.1f}"))
    checks.append(("D3 开销 = 切换次数 × 0.182s（线性，机器证明）",
                   abs(overhead - n_switch * 0.182) < 1e-9,
                   f"overhead={overhead:.3f}"))
    return checks, {"D_overcommit": round(overcommit, 3),
                    "D_overhead_s": round(overhead, 3)}


# =====================================================================
# [E] self-check + digest
# =====================================================================

def main():
    print("=" * 72)
    print("kimi k3 agentic rl sim — partial rollout / per-token 正则 / 预算 / sandbox")
    print("=" * 72)
    print(f"toy: 纯标准库 | seed={SEED} | 机制回声: K3 [2607.24653] "
          f"§4.1.2/§4.1.3/§5.3 + K2.5 [2602.02276] §4.4.2")
    print("全部数字为本 sim 实测或 declared 折算，论文侧只作机制对照、不外推量级。")
    print()
    all_checks, metrics = [], {}
    for fn in (run_A, run_B1, run_B2, run_B3, run_C, run_D):
        chs, ms = fn()
        all_checks.extend(chs)
        metrics.update(ms)
        print()
    print("[E] self-check")
    n_pass = 0
    for name, ok, detail in all_checks:
        tag = "PASS" if ok else "FAIL"
        n_pass += 1 if ok else 0
        print(f"    {tag}  {name}  [{detail}]")
    print(f"    {'✅' if n_pass == len(all_checks) else '❌'} "
          f"self-check {'passed' if n_pass == len(all_checks) else 'FAILED'} "
          f"({n_pass}/{len(all_checks)})")
    ser = ";".join(f"{k}={v}" for k, v in sorted(metrics.items()))
    print()
    print(f"digest(md5 of metrics) = {hashlib.md5(ser.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
