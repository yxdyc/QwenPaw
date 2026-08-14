#!/usr/bin/env python3
"""nano-trinity-rft L0 — 统一 SFT+RL 循环：三组件协同 + 配置驱动

Trinity-RFT（arXiv:2505.17826）把 RFT 解耦成三个协同组件（README L21–25，
agentscope-ai/Trinity-RFT main 分支 2026-08-05 快照）：
  Explorer — 通过 agent-环境交互产生经验数据
  Trainer  — 在数据上最小化 loss 以更新模型权重
  Buffer   — 贯穿 RFT 全生命周期的数据管线
而 SFT 在其中只是一个 `algorithm_type: sft` 配置项（README L121），与
PPO/GRPO/DPO 并列。本 L0 抓的最小机制：**统一数据流 + 配置驱动的阶段切换**
——同一份样本记录、同一个 Buffer、同一个 Trainer，只改 config 字段，训练配方
就在 sft_only / rl_only / sft_then_rl / mix 之间切换。

Toy 口径（显式声明：无真实模型/数据，只演机制）：
  - 表格 softmax 策略：6 context × 4 action，初始权重全 0（= 均匀分布）
  - 环境：每个 context 有唯一正确 action，reward ∈ {0, 1}
  - teacher：只覆盖 ctx 0–2（SFT 数据源）；ctx 3–5 是「数据空洞」，
    只能靠 RL 探索填上——这是 SFT 与 RL 分工的最小可演示形态
  - SFT loss = −log π(target|c)；RL = REINFORCE，baseline 取**每个 context
    组内 reward 均值**——这正是 GRPO 的 group-relative advantage 核心思想
    （arXiv:2402.03300）
零外部依赖（stdlib random/math），CPU 即跑，固定 seed 可复现。

L1 预告：真实小模型上配置驱动跑通 SFT→RL 两阶段；L2：reward 信号来源
（rule-based vs model-based）；L3：对照 Trinity-RFT 真实配置 schema 与
explorer/trainer/buffer 源码。
"""

import math
import random

# ----------------------------- toy 世界 -----------------------------
N_CTX, N_ACT = 6, 4
BEST = [2, 0, 3, 1, 2, 0]        # 每个 context 的唯一正确 action
TEACHER_CTX = [0, 1, 2]          # teacher（SFT 数据源）只覆盖前 3 个 context
EVAL_SEED, EVAL_M = 20260805, 1024  # 采样式评测的固定 rng 与每 context 采样数


def env_reward(ctx, act):
    return 1.0 if act == BEST[ctx] else 0.0


def softmax(w):
    mx = max(w)
    es = [math.exp(x - mx) for x in w]
    z = sum(es)
    return [e / z for e in es]


class Policy:
    """表格 softmax 策略：W[c][a]。初始全 0 = 均匀分布（未训练）。"""

    def __init__(self):
        self.W = [[0.0] * N_ACT for _ in range(N_CTX)]

    def probs(self, c):
        return softmax(self.W[c])

    def sample(self, c, rng):
        p = self.probs(c)
        x, acc = rng.random(), 0.0
        for a, pa in enumerate(p):
            acc += pa
            if x < acc:
                return a
        return N_ACT - 1

    def eval_reward(self):
        """采样式评测：固定 rng，每 context 采 EVAL_M 个 action 取平均 reward。
        未训练策略 = 均匀分布 → 期望恰为 0.25；训练后趋向 1.0。"""
        rng = random.Random(EVAL_SEED)
        tot = 0.0
        for c in range(N_CTX):
            tot += sum(env_reward(c, self.sample(c, rng)) for _ in range(EVAL_M))
        return tot / (N_CTX * EVAL_M)


class Sample:
    """统一数据协议：SFT 与 RL 样本走同一种记录。
    SFT 样本的 reward 只是登记值（teacher 给的 target 必对，恒 1.0），不参与 loss；
    version = -1 表示 teacher 数据（与策略版本无关，天然无 staleness 问题）。"""
    __slots__ = ("kind", "ctx", "act", "reward", "version", "trains")

    def __init__(self, kind, ctx, act, reward, version):
        self.kind, self.ctx, self.act, self.reward, self.version = kind, ctx, act, reward, version
        self.trains = 0    # 生命周期字段：已被训练几步（buffer 管）


class Explorer:
    """产生数据：RL 模式用当前策略与环境交互；SFT 模式从 teacher 取监督对。"""

    def __init__(self, policy, rng):
        self.policy, self.rng = policy, rng

    def rl_rollout(self, version, k):
        return [Sample("rl", c, (a := self.policy.sample(c, self.rng)),
                       env_reward(c, a), version)
                for c in range(N_CTX) for _ in range(k)]

    def sft_data(self, k):
        return [Sample("sft", c, BEST[c], 1.0, -1)
                for c in TEACHER_CTX for _ in range(k)]


class Buffer:
    """统一数据流上的生命周期管线：add（FIFO 容量）→ select → train_count →
    retire（达到 max_reuse 即退役）。对应 Trinity README 的
    'Active data management ... throughout the RFT lifecycle'（L102–105）。"""

    def __init__(self, capacity, max_reuse):
        self.capacity, self.max_reuse = capacity, max_reuse
        self.items = []
        self.n_added = self.n_trainings = self.n_retired = 0

    def add(self, samples):
        for s in samples:
            self.items.append(s)
            self.n_added += 1
        while len(self.items) > self.capacity:   # FIFO 驱逐最旧
            self.items.pop(0)
            self.n_retired += 1

    def select(self, batch_size):
        # trainer 从 buffer 采一个 batch（FIFO 取最旧），不是把整池都训一遍
        return self.items[:batch_size]

    def mark_trained(self, batch):
        for s in batch:
            s.trains += 1
            self.n_trainings += 1

    def prune(self):
        keep = [s for s in self.items if s.trains < self.max_reuse]
        self.n_retired += len(self.items) - len(keep)
        self.items = keep


class Trainer:
    """同一个优化步里消化两种 loss：SFT 交叉熵 + RL REINFORCE（组内 baseline）。"""

    def __init__(self, policy, lr):
        self.policy, self.lr, self.version = policy, lr, 0

    def step(self, batch):
        g = [[0.0] * N_ACT for _ in range(N_CTX)]
        cnt = [0] * N_CTX
        sft = [s for s in batch if s.kind == "sft"]
        rl = [s for s in batch if s.kind == "rl"]
        for s in sft:                                     # ∇ −log π(t|c)
            p = self.policy.probs(s.ctx)
            for a in range(N_ACT):
                g[s.ctx][a] += p[a] - (1.0 if a == s.act else 0.0)
            cnt[s.ctx] += 1
        if rl:                                            # 每 context 组内 baseline
            mean_r = {}
            for c in {s.ctx for s in rl}:
                rs = [s.reward for s in rl if s.ctx == c]
                mean_r[c] = sum(rs) / len(rs)
            for s in rl:
                adv = s.reward - mean_r[s.ctx]
                p = self.policy.probs(s.ctx)
                for a in range(N_ACT):                    # adv=0 的组贡献天然为 0
                    g[s.ctx][a] += -adv * ((1.0 if a == s.act else 0.0) - p[a])
                cnt[s.ctx] += 1
        for c in range(N_CTX):                            # 组内平均，一步更新
            if cnt[c]:
                for a in range(N_ACT):
                    self.policy.W[c][a] -= self.lr * g[c][a] / cnt[c]
        self.version += 1
        return len(sft), len(rl)


def run(cfg, seed=42, verbose=True):
    """按 config 跑完整个三组件循环，返回每轮记录。"""
    rng = random.Random(seed)
    policy = Policy()
    explorer, buffer = Explorer(policy, rng), Buffer(cfg["capacity"], cfg["max_reuse"])
    trainer = Trainer(policy, cfg["lr"])
    hist = []
    total_rounds = cfg["rounds"] if cfg["mix"] else cfg["sft_rounds"] + cfg["rl_rounds"]
    for r in range(total_rounds):
        if cfg["mix"]:
            mode = "mix"
        elif r < cfg["sft_rounds"]:
            mode = "sft"
        else:
            mode = "rl"
        fresh = []
        if mode in ("sft", "mix"):
            fresh += explorer.sft_data(cfg["sft_k"])
        if mode in ("rl", "mix"):
            fresh += explorer.rl_rollout(trainer.version, cfg["rl_k"])
        buffer.add(fresh)
        batch = buffer.select(cfg["batch_size"])
        n_sft, n_rl = trainer.step(batch) if batch else (0, 0)
        buffer.mark_trained(batch)
        buffer.prune()
        ev = policy.eval_reward()
        hist.append(dict(round=r + 1, mode=mode, new_sft=len([s for s in fresh if s.kind == "sft"]),
                         new_rl=len([s for s in fresh if s.kind == "rl"]),
                         b_sft=n_sft, b_rl=n_rl, eval=ev, version=trainer.version))
        if verbose:
            h = hist[-1]
            print(f"  {h['round']:>2}   {h['mode']:<4}   {h['new_sft']:>2}/{h['new_rl']:<2}"
                  f"        {h['b_sft']:>2}/{h['b_rl']:<2}        {h['eval']:.4f}      v{h['version']}")
    return policy, buffer, hist


def per_ctx_eval(policy):
    rng = random.Random(EVAL_SEED)
    return [sum(env_reward(c, policy.sample(c, rng)) for _ in range(EVAL_M)) / EVAL_M
            for c in range(N_CTX)]


def fmt_ctx(vals):
    return "[" + " ".join(f"{v:.2f}" for v in vals) + "]"


# ----------------------------- 实验 -----------------------------
BASE = dict(lr=4.0, capacity=256, max_reuse=2, sft_k=8, rl_k=6, batch_size=48)

print("=" * 72)
print("nano-trinity-rft L0 — 统一 SFT+RL：三组件协同 + 配置驱动")
print("=" * 72)
print(f"toy 口径: 表格 softmax（{N_CTX} ctx × {N_ACT} act），reward∈{{0,1}}；"
      f"teacher 只覆盖 ctx {TEACHER_CTX}，ctx 3–5 是数据空洞。")
print(f"config 基底: lr={BASE['lr']} capacity={BASE['capacity']} "
      f"max_reuse={BASE['max_reuse']} sft_k={BASE['sft_k']} rl_k={BASE['rl_k']}")

# ---- [1] 配置驱动两阶段：sft_then_rl ----
print("\n[1] 配置驱动两阶段：sft_then_rl（sft_rounds=3 → rl_rounds=6）")
print("round  mode  new(sft/rl)  batch(sft/rl)  eval_reward  version")
CFG1 = dict(BASE, sft_rounds=3, rl_rounds=6, mix=False)
p1, b1, h1 = run(CFG1)
assert h1[2]["eval"] > 0.55, "SFT 阶段应在 teacher 覆盖的 ctx 上快速爬升"
assert h1[-1]["eval"] >= 0.95, "RL 阶段应把数据空洞填上"
assert any(h["b_sft"] > 0 for h in h1[3:]), "RL 阶段 batch 里应仍有存量 SFT 样本（统一数据流跨阶段）"
print("=> SFT 阶段在 teacher 覆盖处快爬；RL 阶段把其余 ctx 拉到 ~1.0；")
print("   注意 r4：RL 阶段第一轮的 batch 里仍混着存量 SFT 样本——数据流跨阶段统一。")

# ---- [2] 配置消融阶梯：同 9 轮，只改 config 字段 ----
print("\n[2] 配置消融阶梯（同样 9 轮，只改 config 字段，循环代码一字不动）")
CFG_S = dict(BASE, sft_rounds=9, rl_rounds=0, mix=False)
CFG_R = dict(BASE, sft_rounds=0, rl_rounds=9, mix=False)
pS, _, hS = run(CFG_S, verbose=False)
pR, _, hR = run(CFG_R, verbose=False)
print("recipe        eval@r1   eval@r2   eval@r3   eval@r9   per-context final [c0..c5]")
print(f"sft_only      {hS[0]['eval']:.4f}    {hS[1]['eval']:.4f}    {hS[2]['eval']:.4f}    {hS[-1]['eval']:.4f}    {fmt_ctx(per_ctx_eval(pS))}")
print(f"rl_only       {hR[0]['eval']:.4f}    {hR[1]['eval']:.4f}    {hR[2]['eval']:.4f}    {hR[-1]['eval']:.4f}    {fmt_ctx(per_ctx_eval(pR))}")
print(f"sft_then_rl   {h1[0]['eval']:.4f}    {h1[1]['eval']:.4f}    {h1[2]['eval']:.4f}    {h1[-1]['eval']:.4f}    {fmt_ctx(per_ctx_eval(p1))}")
cov1 = sum(per_ctx_eval(pS)[:3]) / 3
assert hS[0]["eval"] > hR[0]["eval"], "第 1 轮：SFT 模仿 teacher 应快于 RL 冷启探索"
assert cov1 >= 0.9, "sft_only：覆盖处应近 1.0"
assert sum(per_ctx_eval(pS)[3:]) / 3 <= 0.4, "sft_only：空洞处应停在 ~0.25"
assert hR[-1]["eval"] >= 0.9, "rl_only 给足轮数也能填洞，但要自己探索出来"
print("=> 同一套 Explorer/Trainer/Buffer，config 字段一换就是不同配方可直接对比；")
print("   SFT 模仿快但被 teacher 覆盖画死，RL 起步慢但能越过 teacher。")

# ---- [3] mix 模式：一个 batch 同时训两种 loss ----
print("\n[3] mix 模式（每轮同时产出 sft+rl 样本，一个 batch、一步更新）")
print("round  mode  new(sft/rl)  batch(sft/rl)  eval_reward  version")
CFG_M = dict(BASE, rounds=8, mix=True)
pM, bM, hM = run(CFG_M)
assert all(h["b_sft"] > 0 and h["b_rl"] > 0 for h in hM), "mix：每 batch 都该有两种样本"
assert hM[-1]["eval"] >= 0.95, "mix 应收敛到高 reward"
print("=> 两种信号在同一 batch 里叠加——Trinity 的 mix 类算法（如 CHORD）即此形态。")

# ---- [4] 反例：SFT 填不上数据空洞 ----
print("\n[4] 反例：把 sft_only 加到 15 轮——更多 SFT 轮数填不上空洞")
CFG_S15 = dict(BASE, sft_rounds=15, rl_rounds=0, mix=False)
pS15, _, hS15 = run(CFG_S15, verbose=False)
fin = per_ctx_eval(pS15)
mean15 = hS15[-1]["eval"]
print(f"sft_only@15r: eval={mean15:.4f} per-context={fmt_ctx(fin)}")
print(f"算术预测: 覆盖 3 ctx→1.0，空洞 3 ctx→0.25 → mean=(3×1.0+3×0.25)/6=0.625")
assert abs(mean15 - 0.625) < 0.07, "天花板应由数据覆盖决定，而非训练轮数"
assert sum(fin[3:]) / 3 <= 0.4, "空洞 ctx 不随 SFT 轮数改善"
print("=> SFT 的天花板由 teacher 数据覆盖画死；要突破只能靠 RL 探索（或补数据）。")

# ---- buffer 生命周期账本 ----
print(f"\n[账本] [1] 的 buffer: 进入 {b1.n_added} 条 | 训练 {b1.n_trainings} 次"
      f"（max_reuse={CFG1['max_reuse']}）| 退役 {b1.n_retired} 条 | 现存 {len(b1.items)} 条")
assert b1.n_trainings <= b1.n_added * CFG1["max_reuse"], "每条样本最多被训 max_reuse 次"

print("\n✅ self-check passed: 两阶段收敛 / warm start 更快 / sft_only 天花板≈0.625"
      " / mix 双信号共存 / 样本复用不超 max_reuse")
