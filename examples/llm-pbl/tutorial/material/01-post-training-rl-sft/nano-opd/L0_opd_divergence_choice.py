#!/usr/bin/env python3
"""nano-opd L0 — reverse KL vs forward KL vs SFT 蒸馏：on-policy distillation 的最小机制

OPD（on-policy distillation）= 学生自己采样轨迹 + 教师给（token 级）分布监督。
本 toy 抓住它的最小机制：同一个受限学生、同一个教师，只换「优化什么 + 信号从哪来」：

  [SFT 蒸馏]  教师样本 + 硬标签（MLE）        —— 信号=教师生成的数据，离线
  [forward KL] KL(p*||q)，期望在教师分布下     —— 信号=教师全分布，离线、无需学生采样
  [reverse KL] KL(q||p*)，期望在学生分布下     —— 必须从学生自己采样（on-policy），
              用 score-function（REINFORCE）梯度 + 移动平均 baseline 估计
              —— 这正是 MiniLLM [arXiv:2306.08543] 的路线；GKD [arXiv:2306.13649]
              把「学生自采样 + 任意散度」推广成通用框架

toy 口径声明（重要）：
- 「token 空间」用一维离散网格代替；教师是双峰分布（两个「模」）；
- 学生被故意做成容量受限：只有一个可动参数 mu（单峰，峰宽固定）。
  容量受限不是缺陷而是设置——mode-covering vs mode-seeking 的差异只有在
  学生装不下两个模时才显形（真实蒸馏里学生永远比教师小，同一处境）。
- 全部数字是本脚本现场计算的 toy 值，不是任何真实模型的 benchmark。

零依赖（math/random），固定 seed 确定性，CPU 即跑。
"""
import math
import random

# ---------------- 分布设置 ----------------
GRID = list(range(-6, 7))            # 13 个位置，充当「token 表」
SIGMA_T = 1.0                        # 教师两个模的宽度
SIGMA_S = 0.8                        # 学生单峰的宽度（容量受限：只有 mu 可动）
MODE_C, MODE_R = 3.0, 1.5            # 教师高概率区：|x ∓ 3| <= 1.5


def _gauss(x, mu, sigma):
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


def _normalize(ws):
    z = sum(ws)
    return [w / z for w in ws]


# 教师：0.5*N(-3,1) + 0.5*N(+3,1)，离散化后归一
TEACHER = _normalize(
    [0.5 * _gauss(x, -MODE_C, SIGMA_T) + 0.5 * _gauss(x, MODE_C, SIGMA_T) for x in GRID]
)
LOG_TEACHER = [math.log(p) for p in TEACHER]


def student_dist(mu):
    """学生 q_mu：单峰，位置 mu 是唯一可学参数。"""
    return _normalize([_gauss(x, mu, SIGMA_S) for x in GRID])


def kl(p, q):
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0.0)


def sample(dist, rng):
    r, acc = rng.random(), 0.0
    for x, p in zip(GRID, dist):
        acc += p
        if r <= acc:
            return x
    return GRID[-1]


def mode_mass(q):
    """学生样本落在教师高概率区 |x∓3|<=1.5 的质量占比——本节的 toy 指标。"""
    return sum(p for x, p in zip(GRID, q) if abs(abs(x) - MODE_C) <= MODE_R)


def mean_teacher_prob(q):
    """学生分布下教师的平均概率 E_q[p*(x)]：学生生成的东西教师认不认账。"""
    return sum(p * tp for p, tp in zip(q, TEACHER))


# ---------------- 三种配方 ----------------

def train_sft(seed=7, n_data=256, steps=120, lr=0.5, mu0=0.5):
    """SFT 蒸馏（离线）：从教师采 n_data 个样本，硬标签 MLE。
    只需教师「生成的数据」，不需教师分布。∇ E[log q] = E[(x-mu)/σ²]，
    最优解 mu = 样本均值——双峰数据的均值在两峰中间。"""
    rng = random.Random(seed)
    data = [sample(TEACHER, rng) for _ in range(n_data)]
    mu = mu0
    for _ in range(steps):
        mu += lr * sum(x - mu for x in data) / len(data) / (SIGMA_S * SIGMA_S)
    return mu


def train_forward(steps=120, lr=0.5, mu0=0.5):
    """forward KL KL(p*||q)（离线 KD）：期望在教师分布下，精确求和即可，
    完全不需要学生采样。∇ = -E_{p*}[(x-mu)/σ²] → 最优 mu = E_{p*}[x]。"""
    mu = mu0
    for _ in range(steps):
        g = -sum(p * (x - mu) for x, p in zip(GRID, TEACHER)) / (SIGMA_S * SIGMA_S)
        mu -= lr * g
    return mu


def train_reverse(seed=7, mu0=0.5, steps=300, batch=32, lr=0.15, use_baseline=True):
    """reverse KL KL(q||p*)（MiniLLM 式 OPD）：期望在学生分布下——
    所以必须从学生自己采样（on-policy 的由来）。score-function 梯度：
        ∇ KL(q||p*) = E_{x~q}[(log q(x) - log p*(x) - b) · (x-mu)/σ²]
    其中 log p*(x) = 教师对学生每个样本的「打分」（真实 LLM 里 = teacher logprob）。
    b 是移动平均 baseline（方差缩减，policy gradient 的标准件）。"""
    rng = random.Random(seed)
    mu, b = mu0, 0.0
    for _ in range(steps):
        q = student_dist(mu)
        log_q = [math.log(v) for v in q]
        g, f_mean = 0.0, 0.0
        for _ in range(batch):
            x = sample(q, rng)                       # ← 学生自采样（on-policy）
            i = x - GRID[0]
            f = log_q[i] - LOG_TEACHER[i]            # log q - log p*
            f_mean += f
            g += (f - b) * (x - mu) / (SIGMA_S * SIGMA_S)
        g /= batch
        if use_baseline:
            b = 0.9 * b + 0.1 * f_mean / batch
        mu -= lr * g
    return mu


def grad_expected(mu, sampling_dist):
    """score-function 更新量 g 在给定采样分布下的精确期望（无采样噪声）。
    正确估计器要求 sampling_dist == q_mu（学生自己）；换成教师分布即错误估计器。"""
    q = student_dist(mu)
    log_q = [math.log(v) for v in q]
    return sum(
        sp * (lq - lt) * (x - mu) / (SIGMA_S * SIGMA_S)
        for x, sp, lq, lt in zip(GRID, sampling_dist, log_q, LOG_TEACHER)
    )


def train_reverse_offpolicy(seed=7, mu0=0.5, steps=300, batch=32, lr=0.02):
    """反例：把 reverse KL 的期望错用教师样本来估（off-policy 错误估计器）。
    期望分布换成 p* 后梯度有偏——教师把学生从模上往谷里拽。
    小 lr + 边界钳制，防止有偏大步长冲出网格（那本身也是症状之一）。"""
    rng = random.Random(seed)
    mu = mu0
    for _ in range(steps):
        q = student_dist(mu)
        log_q = [math.log(v) for v in q]
        g = 0.0
        for _ in range(batch):
            x = sample(TEACHER, rng)                 # ← 错：从教师采，不是从学生采
            i = x - GRID[0]
            g += (log_q[i] - LOG_TEACHER[i]) * (x - mu) / (SIGMA_S * SIGMA_S)
        mu = max(-5.5, min(5.5, mu - lr * g / batch))
    return mu


# ---------------- 实验 ----------------

def report(tag, mu):
    q = student_dist(mu)
    print(f"{tag:34s} mu={mu:+.3f} | 模区驻留率={mode_mass(q):.3f} "
          f"| E_q[p*]={mean_teacher_prob(q):.4f} | KL(q||p*)={kl(q, TEACHER):.3f}")


def main():
    print("=" * 72)
    print("nano-opd L0 — reverse KL vs forward KL vs SFT 蒸馏")
    print("=" * 72)
    print("toy 口径: 网格双峰教师 + 单峰受限学生(只有 mu 可动)；数字为现场算术，非 benchmark。")

    print("\n[1] 教师分布：两个模，中间是谷")
    i0, i3 = 0 - GRID[0], 3 - GRID[0]
    print(f"    p*(x=0)={TEACHER[i0]:.4f}  p*(x=3)={TEACHER[i3]:.4f}  "
          f"(模/谷 ≈ {TEACHER[i3] / TEACHER[i0]:.0f}x)")

    print("\n[2] 三种配方，同一起点 mu0=+0.5：")
    mu_sft = train_sft(seed=7)
    mu_fwd = train_forward()
    mu_rev = train_reverse(seed=7, mu0=0.5)
    report("sft  (教师样本+硬标签, 离线)", mu_sft)
    report("fwd  (KL(p*||q), 精确求和, 离线)", mu_fwd)
    report("rev  (KL(q||p*), 学生自采样, OPD)", mu_rev)
    print("    => SFT/fwd 把唯一的峰摆在两模中间的谷里(mode-covering, 生成教师不认账的样本)；")
    print("       rev 锁定一个模(mode-seeking, 生成的样本教师给高分)。")

    print("\n[3] 对称破缺：rev 锁哪个模由起点/早期样本决定，绝不骑墙")
    mu_pos = train_reverse(seed=7, mu0=+0.5)
    mu_neg = train_reverse(seed=7, mu0=-0.5)
    print(f"    mu0=+0.5 → mu={mu_pos:+.3f} | mu0=-0.5 → mu={mu_neg:+.3f}")

    print("\n[4] 反例：reverse KL 为什么必须 on-policy")
    q_at_mode = student_dist(3.0)
    g_true = grad_expected(3.0, q_at_mode)      # 正确：期望在学生分布下
    g_wrong = grad_expected(3.0, TEACHER)       # 错误：期望换到教师分布下
    print(f"    a) 在锁模点 mu=+3 处看更新量的精确期望（无采样噪声）:")
    print(f"       正确估计器(从学生采): E[g]={g_true:+.3f}  ≈0，模是驻点")
    print(f"       错误估计器(从教师采): E[g]={g_wrong:+.3f}  ← mu -= lr*g，被拽向更小的 mu（离模进谷）")
    mu_off = train_reverse_offpolicy(seed=7, mu0=0.5)
    q_off = student_dist(mu_off)
    print(f"    b) 真跑起来(教师采样, 300 步): mu={mu_off:+.3f} "
          f"| 模区驻留率={mode_mass(q_off):.3f}  ← 锁不住模，掉回谷里")
    print(f"       （若用主实验的 lr=0.15，有偏的大步长直接冲出网格——偏差不是噪声，是系统性的）")

    print("\n[5] self-check")
    q_sft, q_fwd, q_rev = student_dist(mu_sft), student_dist(mu_fwd), student_dist(mu_rev)
    assert abs(mu_fwd) < 0.1, "forward KL 应收敛到 E_{p*}[x]=0"
    assert abs(mu_sft) < 0.5, "SFT MLE 应收敛到教师样本均值(两峰之间)"
    assert mu_rev > 2.5 and mu_pos > 2.5 and mu_neg < -2.5, "reverse KL 应锁定一个模"
    assert mode_mass(q_rev) > 0.9 > mode_mass(q_fwd) + 0.5, "模区驻留率应拉开"
    assert mean_teacher_prob(q_rev) > 3 * mean_teacher_prob(q_fwd), "教师给模上样本的分数应远高于谷上"
    assert abs(g_true) < 0.5 and g_wrong > 5.0, "正确估计器在模上近似驻点，错误估计器显著拽离"
    assert mode_mass(q_off) < mode_mass(q_rev) - 0.3, "off-policy 错误估计器应锁不住模"
    print("✅ self-check passed: 三配方收敛形态 / 对称破缺 / 模区驻留率 / "
          "教师认账度 / 估计器偏差分析")

    print("\ntakeaway: reverse KL 的期望在学生分布下 → 必须学生自采样(on-policy)；")
    print("          教师只需给每个样本打分(logprob)。这一「学生写、教师批」的接口")
    print("          就是 OPD；L1 把它搬进真实序列模型，L2 加 multi-teacher。")


if __name__ == "__main__":
    main()
