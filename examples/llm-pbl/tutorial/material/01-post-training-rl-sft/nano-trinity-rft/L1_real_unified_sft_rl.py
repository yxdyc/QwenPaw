#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nano-trinity-rft L1 — 真实小模型上的统一 SFT+RL：配置驱动两阶段 + checkpoint 衔接

Trinity-RFT（arXiv:2505.17826）把 RFT 解耦成三个协同组件（README L21–25，
agentscope-ai/Trinity-RFT main 分支 2026-08-06 快照，sha256 d513f140…b73982）：
  Explorer — 通过 agent-环境交互产生经验数据
  Trainer  — 在数据上最小化 loss 以更新模型权重
  Buffer   — 贯穿 RFT 全生命周期的数据管线
SFT 在其中只是一个 `algorithm_type: sft` 配置项（README L121，
trinity/algorithm/policy_loss_fn/sft_loss.py），与 PPO/GRPO/DPO/mix_chord 并列。

L0（L0_unified_sft_rl_loop.py）用表格 softmax 演了这个机制；L1 把同一套架构搬到
真实字符级 GPT（~0.8M 参数，torch）上：
  - 同一种统一 Sample 协议（kind/ctx/resp/reward/adv/version/trains）
  - 同一组 config 字段（sft_rounds / rl_rounds / mix）驱动四配方
  - SFT = 真实交叉熵（含 EOS 格式监督）；RL = 组内相对 policy gradient
    （baseline = 组内 reward 均值，GRPO arXiv:2402.03300 的核心；未除组内标准差）
  - 新增：checkpoint save/load——SFT 阶段产 ckpt、RL 阶段从 ckpt 续训，
    权重/版本号/评测衔接逐项验证（scaffold 原「几步优化」目标并入本级）

任务口径（显式声明：合成任务，只演机制）：6 个 context（prompt "k0:"–"k5:"），
每个 context 有唯一 target 响应（长 4，字母表 a–d）；teacher 只覆盖 ctx 0–2
（SFT 数据源），ctx 3–5 是数据空洞——L0 的覆盖不对称原样搬来。
reward = 逐位字符匹配率（dense、rule-based；稀疏 0/1 与 reward model 是 L2 主题）。

依赖：仅 torch。CPU 单文件，任意 CWD 可跑。固定 seed → 指标行逐字节确定，
累计耗时行随机器负载浮动。
"""

import sys
import time
import tempfile
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
SEED = 20260806

# ----------------------------- 任务与词表 -----------------------------
ALPHABET = "abcd"              # 响应字母表（4 字符，与 L0 的 4 action 分支因子对称）
RESP_LEN = 4                   # 响应定长 4
N_CTX = 6
TEACHER_CTX = [0, 1, 2]        # teacher（SFT 数据源）只覆盖前 3 个 context


def make_targets(seed):
    """确定性生成 6 个 target 响应（任务实例由 seed 完全决定）。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(ALPHABET), (N_CTX, RESP_LEN), generator=g)
    return ["".join(ALPHABET[int(i)] for i in row) for row in idx]


TARGETS = make_targets(SEED)

# 词表：pad/bos/eos + 提示符（'k'、数字 0–5、':'）+ 响应字母表 = 15
VOCAB = ["<pad>", "<bos>", "<eos>", ":", "k"] + [str(d) for d in range(N_CTX)] + list(ALPHABET)
ID = {c: i for i, c in enumerate(VOCAB)}
PAD, BOS, EOS = ID["<pad>"], ID["<bos>"], ID["<eos>"]
ALPHABET_IDS = [ID[c] for c in ALPHABET]


def prompt_ids(c):
    return [ID["k"], ID[str(c)], ID[":"]]


def target_ids(c):
    return [ID[ch] for ch in TARGETS[c]]


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


def reward_of(c, resp_ids):
    """rule-based dense reward：与 target 逐位匹配率 ∈ {0, .25, .5, .75, 1}。"""
    t = TARGETS[c]
    return sum(1 for i, tok in enumerate(resp_ids) if VOCAB[tok] == t[i]) / RESP_LEN


# ----------------------------- 模型 -----------------------------
class Block(nn.Module):
    def __init__(self, d, nhead, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    """字符级 GPT：tok/pos embedding + 4 层 pre-norm attn + 权重绑定 lm_head。"""

    def __init__(self, vocab=len(VOCAB), d=128, nhead=4, nlayers=4, ff=512, maxpos=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxpos, d)
        self.blocks = nn.ModuleList([Block(d, nhead, ff) for _ in range(nlayers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # 权重绑定

    def forward(self, x):
        n = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(n))
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


@torch.no_grad()
def generate(model, c, greedy=False, temp=1.0):
    """从 prompt 解码 4 个响应 token（定长，不看 EOS；贪心不消耗 RNG）。"""
    ids = [BOS] + prompt_ids(c)
    out = []
    for _ in range(RESP_LEN):
        logits = model(torch.tensor([ids]))[0, -1]
        if greedy:
            t = int(logits.argmax())
        else:
            t = int(torch.multinomial((logits / temp).softmax(-1), 1))
        ids.append(t)
        out.append(t)
    return out


def evaluate(model):
    """确定性评测：每 ctx 贪心解码，exact（全对 0/1）与 characc（匹配率）。"""
    ex, ca = [], []
    for c in range(N_CTX):
        resp = generate(model, c, greedy=True)
        ex.append(1.0 if reward_of(c, resp) == 1.0 else 0.0)
        ca.append(reward_of(c, resp))
    return sum(ex) / N_CTX, sum(ca) / N_CTX, ex, ca


# ----------------------------- 统一数据协议与三组件 -----------------------------
class Sample:
    """与 L0 同构的统一样本记录：SFT/RL 走同一种记录。
    SFT 样本 version=-1（teacher 数据，与策略版本无关，天然无 staleness）；
    RL 样本 version=产生时的 trainer.version，adv = 逐 token 的组内相对优势
    （reward 逐位可分解 → baseline 也逐位取组内均值，token 级 credit assignment）。
    trains = 已被训练次数（buffer 管）。
    注：ε-探索样本不做重要性比率校正（prototype 实测 ρ=π/q 会把探索发现
    的低概率正确 token 降权，策略被锁回自身偏好——见 tutorial §6），
    样本又都在同轮附近消费（staleness 账本见 [5]），朴素 REINFORCE 即可。"""
    __slots__ = ("kind", "ctx", "resp", "reward", "adv", "version", "trains")

    def __init__(self, kind, ctx, resp, reward, adv, version):
        self.kind, self.ctx, self.resp = kind, ctx, resp
        self.reward, self.adv, self.version = reward, adv, version
        self.trains = 0


class Explorer:
    """产生经验数据。ε-探索是本级的关键设计：SFT 之后模型在空洞 ctx 上坍缩成
    近确定的错误吸引子（prototype 实测：组内 unique≤5/24、位置熵≈0、|adv|≈0，
    RL 完全停摆）——toy 世界里探索免费（均匀初始化），真实模型里探索是必须
    显式维护的机制。每个响应位置以概率 ε 从字母表均匀采样，保证组内方差。"""

    def __init__(self, model, eps=0.3, temp=1.0):
        self.model, self.eps, self.temp = model, eps, temp

    def sft_data(self, k):
        """从 teacher 取监督对：target 必对，reward 只是登记值（不参与 RL loss）。
        到达顺序跨 context 交错（c0,c1,c2,c0,…）——严格 FIFO select 下若按 context
        成块到达，batch 会偏食单一 context（prototype 实测教训，见 tutorial §7）。"""
        return [Sample("sft", c, target_ids(c), 1.0, 0.0, -1)
                for _ in range(k) for c in TEACHER_CTX]

    def _sample_resp(self, c):
        """ε-探索采样一条响应：每个位置以概率 ε 从字母表均匀采样。"""
        ids = [BOS] + prompt_ids(c)
        out = []
        with torch.no_grad():
            for _ in range(RESP_LEN):
                logits = self.model(torch.tensor([ids]))[0, -1]
                p = (logits / self.temp).softmax(-1)
                if torch.rand(1).item() < self.eps:
                    t = ALPHABET_IDS[int(torch.randint(len(ALPHABET_IDS), (1,)))]
                else:
                    t = int(torch.multinomial(p, 1))
                ids.append(t)
                out.append(t)
        return out

    def rl_rollout(self, version, group):
        """当前策略 × 环境交互：每 ctx 采 group 条；reward 逐位分解，
        baseline 逐位取组内均值 → 每 token 一个组内相对优势。
        组内算完 adv 后跨 context 交错入池（同 sft_data 的理由）。"""
        groups = []
        for c in range(N_CTX):
            resps = [self._sample_resp(c) for _ in range(group)]
            t = TARGETS[c]
            matches = [[1.0 if VOCAB[tok] == t[j] else 0.0 for j, tok in enumerate(r)]
                       for r in resps]
            mean_j = [sum(m[j] for m in matches) / group for j in range(RESP_LEN)]
            groups.append([Sample("rl", c, r, sum(m) / RESP_LEN,
                                  [m[j] - mean_j[j] for j in range(RESP_LEN)], version)
                           for r, m in zip(resps, matches)])
        return [groups[c][i] for i in range(group) for c in range(N_CTX)]


class Buffer:
    """与 L0 同构：容量 → select → mark_trained → prune（max_reuse 退役）。
    两处 L1 升级（均有 prototype 实测依据，见 tutorial §7）：
    ① select 从严格 FIFO 改为随机采样（真实 buffer 的行为；FIFO 在成块到达下
       会偏食单一 context，且跨阶段回放只集中在切换头一两轮）；
    ② 账本字段：rl/sft 各被训多少次、RL 样本训练时的版本差（staleness）、
       容量驱逐损失的潜在训练次数（供 [5] 恒等式对账）。"""

    def __init__(self, capacity, max_reuse):
        self.capacity, self.max_reuse = capacity, max_reuse
        self.items = []
        self.n_added = self.n_trainings = self.n_retired = 0
        self.rl_trainings = self.sft_trainings = 0
        self.rl_gap_sum = self.rl_gap_max = 0
        self.lost_potential = 0

    def add(self, samples):
        for s in samples:
            self.items.append(s)
            self.n_added += 1
        while len(self.items) > self.capacity:
            ev = self.items.pop(0)
            self.lost_potential += self.max_reuse - ev.trains
            self.n_retired += 1

    def select(self, batch_size):
        # 随机采样（消耗全局 RNG，固定 seed 下确定）——不按到达顺序取
        n = min(batch_size, len(self.items))
        idx = torch.randperm(len(self.items))[:n].tolist()
        return [self.items[i] for i in idx]

    def mark_trained(self, batch, version):
        for s in batch:
            s.trains += 1
            self.n_trainings += 1
            if s.kind == "rl":
                gap = version - s.version      # 训练时版本 − 产生时版本（≥1）
                self.rl_gap_sum += gap
                self.rl_gap_max = max(self.rl_gap_max, gap)
                self.rl_trainings += 1
            else:
                self.sft_trainings += 1

    def prune(self):
        keep = [s for s in self.items if s.trains < self.max_reuse]
        self.n_retired += len(self.items) - len(keep)
        self.items = keep


class Trainer:
    """同一个优化步消化两种 loss（与 L0 Trainer.step 同构）：
    SFT = 交叉熵（响应 4 token + EOS，prompt 区不参与）；
    RL  = −adv × token logprob（+ 可选 β·KL(π‖π_ref)）。
      adv 逐 token 组内中心化（explorer 算好）；ε-探索样本不做比率校正
      （见 Sample 注释）；KL 为 GRPO 目标函数中的正则项（arXiv:2402.03300），
      本节默认 β=0——实测 β>0 时空洞处的平衡点被拉向错误 ref 一侧。"""

    def __init__(self, model, lr, beta=0.0):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.beta = beta
        self.version = 0

    def step(self, batch, ref=None):
        ids = torch.tensor([[BOS] + prompt_ids(s.ctx) + s.resp + [EOS] for s in batch])
        logits = self.model(ids)                      # (B, 9, V)
        logp = F.log_softmax(logits, dim=-1)
        sft = [i for i, s in enumerate(batch) if s.kind == "sft"]
        rl = [i for i, s in enumerate(batch) if s.kind == "rl"]
        loss_sft = loss_rl = torch.tensor(0.0)
        kl_val = 0.0
        if sft:
            # 位置 3..7 预测 ids[:, 4..8] = 4 个响应 token + EOS
            loss_sft = F.nll_loss(logp[sft, 3:8].reshape(-1, len(VOCAB)),
                                  ids[sft, 4:9].reshape(-1))
        if rl:
            # 响应 token 在 ids 的位置 4..7，预测它们的 logits 在位置 3..6
            tok_logp = logp[rl, 3:7].gather(2, ids[rl, 4:8].unsqueeze(-1)).squeeze(-1)
            adv = torch.tensor([batch[i].adv for i in rl])          # (n,4) 逐 token
            loss_rl = -(adv * tok_logp).sum(dim=1).mean()
            if ref is not None and self.beta > 0:
                with torch.no_grad():
                    ref_logp = F.log_softmax(ref(ids), dim=-1)
                # 响应 4 个位置上的全词表 KL(π‖π_ref)，锚定格式与分布
                kl = (logp[rl, 3:7].exp() * (logp[rl, 3:7] - ref_logp[rl, 3:7])).sum(-1)
                loss_rl = loss_rl + self.beta * kl.mean()
                kl_val = kl.mean().item()
        loss = loss_sft + loss_rl
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.version += 1
        return len(sft), len(rl), loss_sft.item(), loss_rl.item(), kl_val


# ----------------------------- checkpoint -----------------------------
def save_ckpt(path, model, trainer, cfg):
    """权重 + 优化器状态 + 版本号 + config + 全局 RNG 状态：换配方只需换 cfg 字段。"""
    torch.save({"model": model.state_dict(), "opt": trainer.opt.state_dict(),
                "version": trainer.version, "cfg": cfg,
                "torch_rng": torch.get_rng_state()}, path)


def load_ckpt(path, model, trainer):
    ck = torch.load(path, weights_only=False)
    model.load_state_dict(ck["model"])
    trainer.opt.load_state_dict(ck["opt"])
    trainer.version = ck["version"]
    torch.set_rng_state(ck["torch_rng"])
    return ck


# ----------------------------- 统一循环（与 L0 run(cfg) 同构） -----------------------------
def run(cfg, model=None, trainer=None, buffer=None, rounds_override=None):
    """按 config 跑三组件循环。mode 由 sft_rounds/rl_rounds/mix 决定，循环代码一字不动。
    全新启动才播种；从 ckpt 续训（传入 model/trainer）时沿用 ckpt 恢复的 RNG 流。"""
    if model is None:
        torch.manual_seed(cfg["seed"])
        model = TinyGPT()
    if trainer is None:
        trainer = Trainer(model, cfg["lr"], cfg.get("beta", 0.0))
    if buffer is None:
        buffer = Buffer(cfg["capacity"], cfg["max_reuse"])
    explorer = Explorer(model, eps=cfg.get("eps", 0.3), temp=cfg.get("temp", 1.0))
    total = rounds_override if rounds_override is not None else \
        (cfg["rounds"] if cfg["mix"] else cfg["sft_rounds"] + cfg["rl_rounds"])
    hist = []
    ref = None  # π_ref：RL 起点策略的冻结副本（GRPO 语义），首次进 RL 模式时创建
    for r in range(total):
        mode = "mix" if cfg["mix"] else ("sft" if r < cfg["sft_rounds"] else "rl")
        if mode in ("rl", "mix") and ref is None:
            ref = deepcopy(model)
            for p in ref.parameters():
                p.requires_grad_(False)
        # phase-aware lr：RL/mix 阶段用更小的 lr（生产配方常见做法——
        # RL 梯度噪声大、且会干扰 SFT 已得行为，低 lr 换稳定性）
        scale = 1.0 if mode == "sft" else cfg.get("rl_lr_scale", 0.1)
        for pg in trainer.opt.param_groups:
            pg["lr"] = cfg["lr"] * scale
        if mode in ("rl", "mix"):
            # 探索退火：RL 后期线性降 ε（0.3→~0.1），让策略在末期定稿
            rl_start = 0 if cfg["mix"] else cfg["sft_rounds"]
            frac = min(1.0, (r - rl_start) / max(1, total - rl_start - 1))
            explorer.eps = cfg.get("eps", 0.3) * (1 - 0.66 * frac)
        fresh = []
        if mode in ("sft", "mix"):
            fresh += explorer.sft_data(cfg["sft_k"])
        if mode in ("rl", "mix"):
            fresh += explorer.rl_rollout(trainer.version, cfg["group"])
        buffer.add(fresh)
        b_sft = b_rl = 0
        for _ in range(cfg["steps_per_round"]):
            batch = buffer.select(cfg["batch_size"])
            if not batch:
                break
            n_sft, n_rl, _, _, _ = trainer.step(batch, ref=ref)
            buffer.mark_trained(batch, trainer.version)
            buffer.prune()
            b_sft, b_rl = b_sft + n_sft, b_rl + n_rl
        exact, characc, per_ex, per_ca = evaluate(model)
        hist.append(dict(round=r + 1, mode=mode,
                         new_sft=len([s for s in fresh if s.kind == "sft"]),
                         new_rl=len([s for s in fresh if s.kind == "rl"]),
                         b_sft=b_sft, b_rl=b_rl, exact=exact, characc=characc,
                         per_ex=per_ex, per_ca=per_ca, version=trainer.version))
        if cfg.get("verbose", True):
            h = hist[-1]
            print(f"  {h['round']:>2}   {h['mode']:<4}  {h['new_sft']:>3}/{h['new_rl']:<3}"
                  f"     {h['b_sft']:>3}/{h['b_rl']:<3}    {h['exact']:.3f}   "
                  f"{h['characc']:.3f}      v{h['version']}")
    return model, trainer, buffer, hist


def fmt(v):
    return "[" + " ".join(f"{x:.0f}" for x in v) + "]"


def fmtc(v):
    return "[" + " ".join(f"{x:.2f}" for x in v) + "]"




def main():
    # ----------------------------- 实验 -----------------------------
    BASE = dict(seed=SEED, lr=3e-3, capacity=1024, max_reuse=2, sft_k=40, group=64,
                batch_size=32, steps_per_round=24, temp=1.0, eps=0.3, beta=0.0,
                rl_lr_scale=0.15)

    print("=" * 76)
    print("nano-trinity-rft L1 — 真实小模型统一 SFT+RL：配置驱动两阶段 + checkpoint 衔接")
    print("=" * 76)
    print(f"env: python {sys.version.split()[0]} | torch {torch.__version__} | seed {SEED}")

    # ---- [0] 任务 + 模型 + 确定性探针 ----
    torch.manual_seed(SEED)  # m0 初始化消耗全局 RNG，显式播种保证 [0] 探针确定
    m0 = TinyGPT()
    n_params = sum(p.numel() for p in m0.parameters())
    print(f"\n[0] 任务与模型")
    print(f"targets: {TARGETS}  （teacher 只覆盖 ctx {TEACHER_CTX}，ctx 3–5 是数据空洞）")
    print(f"model: char-GPT vocab={len(VOCAB)} d=128 layers=4 heads=4 | params={n_params:,}")
    x = torch.tensor([[BOS] + prompt_ids(0) + target_ids(0) + [EOS]])
    lg1, lg2 = m0(x), m0(x)
    assert torch.equal(lg1, lg2), "同一输入两次前向必须逐位相同（CPU 确定性）"
    ex0, ca0, _, per_ca0 = evaluate(m0)
    print(f"确定性探针: 同输入两次前向逐位一致 ✓ | 未训练贪心评测 exact={ex0:.3f} characc={ca0:.3f}")
    print(f"未训练 per-ctx characc: {fmtc(per_ca0)}")

    # ---- [1] 配置驱动两阶段 sft_then_rl（主干）----
    print(f"\n[1] 配置驱动两阶段：sft_then_rl（sft_rounds=3 → rl_rounds=20，连续 buffer）")
    CFG1 = dict(BASE, sft_rounds=3, rl_rounds=20, mix=False)
    print("round  mode  new(sft/rl)  batch(sft/rl)  exact  characc  version")
    m1, t1, b1, h1 = run(CFG1)
    print("per-ctx characc 轨迹（c0–c2 覆盖 / c3–c5 空洞）:")
    for r in (1, 2, 3, 4, 6, 9, 12, 16, 23):
        print(f"  r{r:>2}: {fmtc(h1[r - 1]['per_ca'])}")
    assert h1[2]["characc"] >= 0.45, "SFT 阶段应把 teacher 覆盖处快速拉高（空洞贪心基线=0，聚合上限 0.5）"
    assert all(h1[2]["per_ca"][c] >= 0.95 for c in TEACHER_CTX), "覆盖处 SFT 后应近 1.0"
    assert h1[-1]["exact"] >= 5 / 6, "RL 阶段应把数据空洞填上"
    assert all(h1[-1]["per_ca"][c] >= 0.95 for c in range(N_CTX)), "终态全 ctx 应近 1.0"
    assert all(h["b_sft"] == 0 for h in h1[3:]), \
        "本配置 SFT 阶段供给>需求，无存量跨阶段（存量 regime 见 [5] 探针）"
    print("=> SFT 阶段在覆盖处快爬；RL 阶段逐位填洞，终态全 1.0。")
    print("   本配置 SFT 阶段供给(768)>需求(240)，样本当轮训完、无存量跨阶段——")
    print("   存量是否跨阶段由供给/需求决定，反向探针见 [5]。")

    # ---- [2] checkpoint 衔接：SFT 产 ckpt → 新实例 load → RL 续训 ----
    print(f"\n[2] checkpoint 衔接（SFT 阶段产 ckpt，RL 阶段从 ckpt 续训）")
    tmpdir = tempfile.mkdtemp(prefix="trinity_l1_ckpt_")
    ck_path = os.path.join(tmpdir, "sft_phase.pt")
    CFG_A = dict(BASE, sft_rounds=3, rl_rounds=0, mix=False, verbose=False)
    mA, tA, _, hA = run(CFG_A)
    save_ckpt(ck_path, mA, tA, CFG_A)
    size_kb = os.path.getsize(ck_path) / 1024
    # 衔接验证 a：确定性交叉——Phase A 轨迹必须与 [1] 前 3 轮逐位一致（同 cfg 同 seed）
    assert all(hA[i]["exact"] == h1[i]["exact"] and hA[i]["characc"] == h1[i]["characc"]
               and hA[i]["version"] == h1[i]["version"] for i in range(3)), \
        "同 cfg 同 seed：Phase A 与连续跑前 3 轮必须逐位一致"
    # 衔接验证 b：load 进全新模型实例，贪心输出/评测/版本号逐项衔接
    mB = TinyGPT()
    tB = Trainer(mB, CFG_A["lr"])
    outs_before = [decode(generate(mA, c, greedy=True)) for c in range(N_CTX)]
    load_ckpt(ck_path, mB, tB)
    outs_after = [decode(generate(mB, c, greedy=True)) for c in range(N_CTX)]
    exB, caB, _, _ = evaluate(mB)
    assert outs_before == outs_after, "load 后贪心输出必须与保存前逐字一致"
    assert exB == hA[-1]["exact"] and caB == hA[-1]["characc"], "load 后评测必须与保存前一致"
    assert tB.version == tA.version == h1[2]["version"], "版本号必须衔接（以实跑账本为准）"
    print(f"ckpt: {size_kb:.0f} KB（model+opt+version+cfg+rng）| version={tB.version} 衔接 ✓")
    print(f"load 进全新实例：贪心输出逐字一致 ✓ | exact={exB:.3f} characc={caB:.3f} 与保存前一致 ✓")
    print(f"greedy@ckpt: {outs_after}")
    # 衔接验证 c：从 ckpt 续 RL，轨迹必须与 [1] 的 r4–r23 逐轮逐位一致——
    # 本配置 r3 末 buffer 恰为空（供给>需求），于是「ckpt 续训」与「不间断跑」
    # 的唯一差别就是有没有走 save/load：权重+RNG+version+cfg 全衔接 ⇒ 逐位相等
    CFG_B = dict(BASE, sft_rounds=0, rl_rounds=20, mix=False, verbose=False)
    _, _, _, hB = run(CFG_B, model=mB, trainer=tB, rounds_override=20)
    assert all(hB[i]["exact"] == h1[3 + i]["exact"] and hB[i]["characc"] == h1[3 + i]["characc"]
               and hB[i]["version"] == h1[3 + i]["version"] for i in range(20)), \
        "ckpt 续训轨迹必须与连续跑的 r4–r23 逐位一致"
    print(f"从 ckpt 续 RL 20 轮: exact {exB:.3f} → {hB[-1]['exact']:.3f} | "
          f"characc {caB:.3f} → {hB[-1]['characc']:.3f}")
    print(f"与 [1] 的 r4–r23 逐轮逐位一致 ✓（20/20 轮 exact/characc/version 全等）"
          f" —— checkpoint 衔接是精确的，不是「差不多」")
    os.remove(ck_path)
    os.rmdir(tmpdir)

    # ---- [3] 配置消融阶梯：同 20 轮，只改 config 字段 ----
    print(f"\n[3] 配置消融阶梯（同样 20 轮，只改 config 字段，循环代码一字不动）")
    CFG_S = dict(BASE, sft_rounds=20, rl_rounds=0, mix=False, verbose=False)
    CFG_R = dict(BASE, sft_rounds=0, rl_rounds=20, mix=False, verbose=False)
    CFG_SR = dict(BASE, sft_rounds=3, rl_rounds=17, mix=False, verbose=False)
    CFG_M = dict(BASE, rounds=20, mix=True, verbose=False)
    mS, _, bS, hS = run(CFG_S)
    mR, _, bR, hR = run(CFG_R)
    _, _, _, hSR = run(CFG_SR)
    mM, _, bM, hM = run(CFG_M)
    print("recipe        exact@r1  exact@r3  exact@r6  exact@r12  exact@r20  per-ctx exact 末态")
    for name, h in (("sft_only", hS), ("rl_only", hR), ("sft_then_rl", hSR), ("mix", hM)):
        print(f"{name:<13} {h[0]['exact']:.3f}     {h[2]['exact']:.3f}     "
              f"{h[5]['exact']:.3f}     {h[11]['exact']:.3f}     {h[-1]['exact']:.3f}"
              f"      {fmt(h[-1]['per_ex'])}")
    assert hS[0]["exact"] > hR[0]["exact"], "第 1 轮：SFT 模仿应快于 RL 冷启探索"
    assert hSR[2]["exact"] > hR[2]["exact"], "warm start：前 3 轮 SFT 打底应快于纯 RL"
    assert hS[-1]["exact"] <= 0.51, "sft_only 天花板：覆盖 3/6 ctx → exact ≤ 0.5"
    assert all(hS[-1]["per_ex"][c] == 0.0 for c in range(3, N_CTX)), "空洞 ctx 贪心不应碰对"
    assert hR[-1]["exact"] >= 5 / 6, "rl_only 给足轮数应能填洞"
    assert all(h["b_sft"] > 0 and h["b_rl"] > 0 for h in hM), "mix：每轮 batch 都该有两种样本"
    assert hM[-1]["exact"] >= 5 / 6, "mix 应收敛"
    print("=> 同一套 Explorer/Trainer/Buffer，config 一换就是不同配方；SFT 快但被覆盖画死，")
    print("   RL 慢但能越过 teacher，mix 双信号同一步叠加。")

    # ---- [4] 反例：sft_only 20 轮填不上空洞（复用 [3] 的 hS，不另跑） ----
    print(f"\n[4] 反例：sft_only 跑满 20 轮——更多 SFT 轮数填不上数据空洞")
    fin_ex, fin_ca = hS[-1]["exact"], hS[-1]["characc"]
    hole_ca = sum(hS[-1]["per_ca"][3:]) / 3
    cov_ca = sum(hS[-1]["per_ca"][:3]) / 3
    print(f"sft_only@20r: exact={fin_ex:.3f} characc={fin_ca:.3f}")
    print(f"per-ctx exact 末态: {fmt(hS[-1]['per_ex'])} | "
          f"覆盖处 characc={cov_ca:.3f} | 空洞处 characc={hole_ca:.3f}")
    print(f"算术: 覆盖 3 ctx→exact 1.0，空洞 3 ctx 贪心 exact→0 → mean=(3×1+3×0)/6=0.500")
    assert abs(fin_ex - 0.5) < 0.02, "天花板由数据覆盖决定，不随 SFT 轮数改变"
    assert hS[-1]["per_ex"][0] == 1.0 and all(p == 0.0 for p in hS[-1]["per_ex"][3:]), \
        "覆盖处全对、空洞处全错的结构应稳定"
    print("=> SFT 的天花板由 teacher 数据覆盖画死；突破只能靠 RL 探索（或补数据）。")

    # ---- [5] buffer 与 staleness 账本 ----
    print(f"\n[5] 账本（[1] 的 buffer）")
    residual = sum(CFG1["max_reuse"] - s.trains for s in b1.items)
    print(f"进入 {b1.n_added} 条 | 训练 {b1.n_trainings} 次（sft {b1.sft_trainings} + "
          f"rl {b1.rl_trainings}，max_reuse={CFG1['max_reuse']}）| 退役 {b1.n_retired} 条 | "
          f"现存 {len(b1.items)} 条")
    print(f"恒等式: 进入×max_reuse = {b1.n_added * CFG1['max_reuse']} = 训练 {b1.n_trainings} + "
          f"现存潜在 {residual} + 容量驱逐损失 {b1.lost_potential} ✓"
          if b1.n_added * CFG1["max_reuse"] == b1.n_trainings + residual + b1.lost_potential
          else "恒等式不平！")
    assert b1.n_added * CFG1["max_reuse"] == b1.n_trainings + residual + b1.lost_potential
    assert b1.n_trainings <= b1.n_added * CFG1["max_reuse"], "每条样本最多被训 max_reuse 次"
    if b1.rl_trainings:
        print(f"staleness（RL 样本训练时版本差）: mean={b1.rl_gap_sum / b1.rl_trainings:.2f} "
              f"max={b1.rl_gap_max} —— 有界但非零：均值≈半轮版本数，max≈2 轮")
        print("   （随机采样下个别样本被拖到下一轮才训；真实系统用 importance ratio")
        print("     校正陈旧样本，见 nano-verl L1 的 ratio 与 clip）")
    print(f"供给/需求: 每轮新产 sft {CFG1['sft_k'] * len(TEACHER_CTX)} + rl "
          f"{CFG1['group'] * N_CTX} = {CFG1['sft_k'] * len(TEACHER_CTX) + CFG1['group'] * N_CTX} 条"
          f"（×max_reuse 2 = 需求 {(CFG1['sft_k'] * len(TEACHER_CTX) + CFG1['group'] * N_CTX) * 2}"
          f" 槽位/轮）vs 供给 batch {CFG1['batch_size']}×{CFG1['steps_per_round']} 步 = "
          f"{CFG1['batch_size'] * CFG1['steps_per_round']} 槽位/轮")
    # 反向探针：需求>供给时，存量 SFT 会跨阶段流入 RL batch（L0 r4 24/24 的真实版）
    CFG_FLOW = dict(BASE, sft_rounds=3, rl_rounds=2, mix=False, sft_k=140, verbose=False)
    _, _, _, hF = run(CFG_FLOW)
    print(f"跨阶段存量探针（sft_k=140：需求 {140 * len(TEACHER_CTX) * 2} > 供给 768）: "
          f"r4（RL 第一轮）batch = sft {hF[3]['b_sft']} / rl {hF[3]['b_rl']}"
          f" —— 上一阶段的 SFT 存量流进了 RL 阶段")
    assert hF[3]["b_sft"] > 0, "需求>供给时应有存量 SFT 跨阶段"

    print(f"\n✅ self-check passed: 确定性前向 / SFT 快爬 / RL 填洞 / ckpt 衔接逐字一致 / "
          f"消融阶梯 / sft_only 天花板≈0.5 / mix 双信号 / 账本恒等式")
    print(f"elapsed: {time.time() - T0:.1f}s（计时行随机器负载浮动，指标行逐字节确定）")



if __name__ == "__main__":
    main()
