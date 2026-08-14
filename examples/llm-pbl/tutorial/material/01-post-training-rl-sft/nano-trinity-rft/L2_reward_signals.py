#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nano-trinity-rft L2 — RL 的信号来源：rule reward vs learned reward model

L0/L1 讲了统一数据流与配置驱动（SFT/RL 同一条 Sample 记录、同一个循环）；L1 的
reward 是「逐位匹配率」，当时声明它是 dense、rule-based，把「稀疏 0/1 与 reward
model」留给本级。L2 就回答这个问题：**RL 阶段的 reward 信号从哪里来，选哪种，
各自付出什么代价。**

对标权威实现（2026-08-12 现场核验，agentscope-ai/Trinity-RFT main 分支）：
  - reward 的家在 `trinity/common/rewards/`：基类 RewardFn 返回
    Dict[str, float] 多组件（reward_fn.py:L7-16），注册表 REWARD_FUNCTIONS
    （__init__.py:L7-18）里 7 个已注册 reward 全部是 rule/parse 型
    （math/format/countdown/accuracy/dapo/rlcr）；
  - 组合方式：workflow 里 `reward = sum(reward_dict.values())`
    （math_rm_workflow.py:L43，customized_math_workflows.py 同构），各组件同时
    记进 response.metrics（L40-42）——proxy 与 gold 分开记账；
  - 稀疏/同分组的信号问题：GRPOGroupedAdvantage 的 std_threshold 参数
    （grpo_advantage.py:L97、L106-107 docstring、L160-163 实现）——组内 reward
    标准差低于阈值的组直接清空跳过（DAPO Dynamic Sampling 同机制，
    arXiv:2503.14476 §3.2）；
  - 非可验证域：README:L73 指向 RULER（LLM-as-judge 排序）/ rubric-as-reward
    示例（RaR，arXiv:2507.17746）；RULER 示例 README 明确同时跟踪
    reward(judge) / gold_reward(rule) / judge_success / eval_accuracy，并提醒
    「rewards can be noisy → lr 调小」。
其余引用：GRPO arXiv:2402.03300；reward model 过优化与 KL 缓解
arXiv:2210.10760（gold/proxy 合成实验 + Goodhart）；learned reward 源头
arXiv:1706.03741（Bradley-Terry 成对偏好）；RLHF 管线 arXiv:2203.02155；
DeepSeek-R1 rule-based RLVR arXiv:2501.12948（math_rm_workflow.py:L12 自述
「as introduced in DeepSeek-R1」）。Trinity 论文 arXiv:2505.17826。

Toy 口径（显式声明：合成任务，只演机制）：沿用 L1 的字符级 GPT（~0.8M 参数）
与同一张任务表（SEED 同源）——6 个 context、4 字符响应、teacher 只覆盖 ctx 0–2。
本级新增四件事：
  [1] 信号算术：reward 粒度 × 策略水平 → 组内方差存在性（dead group 率，
      实测 vs 解析式 p^G+(1-p)^G）；
  [2] 学习后果：稀疏 reward 下 group size 与 dynamic sampling（std_threshold
      的 nano 版）对填洞的影响；
  [3] learned reward model：Bradley-Terry 成对偏好训练，含注入的标注者偏置
      （同分对偏好含 'a' 多的响应），测准确率 / 相关性 / 偏置；
  [4] Goodhart：只优化 RM（proxy）时策略钻偏置空子——proxy 涨、gold 掉；
      KL 正则的缓解；rule reward 对照组。
依赖：仅 torch。CPU 单文件，任意 CWD 可跑（python3 -B）。固定 seed → 指标行
逐字节确定，elapsed 计时行随机器负载浮动（掩码口径见 tutorial §2）。
"""

import sys
import time
import json
import math
import hashlib
import itertools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
SEED = 20260806          # 与 L1 同源：同一张任务表、同一条初始化 RNG 流

# ----------------------------- 任务与词表（与 L1 同构） -----------------------------
ALPHABET = "abcd"
RESP_LEN = 4
N_CTX = 6
TEACHER_CTX = [0, 1, 2]


def make_targets(seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(ALPHABET), (N_CTX, RESP_LEN), generator=g)
    return ["".join(ALPHABET[int(i)] for i in row) for row in idx]


TARGETS = make_targets(SEED)

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


# ----------------------------- reward 函数（本级主角） -----------------------------
def reward_dense(c, resp_ids):
    """rule-based dense：逐位匹配率 ∈ {0,.25,.5,.75,1}（L1 的 reward，逐位可分解）。"""
    t = TARGETS[c]
    return sum(1 for i, tok in enumerate(resp_ids) if VOCAB[tok] == t[i]) / RESP_LEN


def reward_sparse(c, resp_ids):
    """rule-based sparse：exact match {0,1}（Trinity AccuracyReward 的形态：
    'Reward 1 if the content is the same as the ground truth, 0 otherwise'，
    accuracy_reward.py:L61-67）。"""
    return 1.0 if decode(resp_ids) == TARGETS[c] else 0.0


# ----------------------------- 模型（与 L1 同构） -----------------------------
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
    def __init__(self, vocab=len(VOCAB), d=128, nhead=4, nlayers=4, ff=512, maxpos=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxpos, d)
        self.blocks = nn.ModuleList([Block(d, nhead, ff) for _ in range(nlayers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        n = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(n))
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


@torch.no_grad()
def generate(model, c, greedy=False, temp=1.0):
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
    ex, ca = [], []
    for c in range(N_CTX):
        resp = generate(model, c, greedy=True)
        ex.append(1.0 if reward_sparse(c, resp) == 1.0 else 0.0)
        ca.append(reward_dense(c, resp))
    return sum(ex) / N_CTX, sum(ca) / N_CTX, ex, ca


# ----------------------------- 统一数据协议与三组件 -----------------------------
class Sample:
    """与 L0/L1 同构的统一样本记录，新增 rsrc = reward 来源标签。
    Trinity 的对应物：Experience.metrics 里分组件记录 reward_dict
    （math_rm_workflow.py:L40-42）——信号来源可追溯，proxy/gold 分开记账。"""
    __slots__ = ("kind", "ctx", "resp", "reward", "adv", "version", "trains", "rsrc")

    def __init__(self, kind, ctx, resp, reward, adv, version, rsrc):
        self.kind, self.ctx, self.resp = kind, ctx, resp
        self.reward, self.adv, self.version, self.rsrc = reward, adv, version, rsrc
        self.trains = 0


class Explorer:
    """与 L1 同构的 ε-探索 explorer；唯一变化：reward 成为 config 字段 reward_fn，
    循环代码不关心信号来自 rule 还是 RM。组内 advantage = r − mean(r)
    （outcome 级，GRPO 语义 arXiv:2402.03300；未除组内 std，与 L1 口径一致——
    Trinity 生产实现除 std+ε，grpo_advantage.py:L169）。
    同时统计 dead group：组内 reward 全同（std=0）→ advantage 全 0 → 零梯度。"""

    def __init__(self, model, reward_fn, rsrc, eps=0.3, temp=1.0, per_token=False):
        self.model, self.reward_fn, self.rsrc = model, reward_fn, rsrc
        self.eps, self.temp, self.per_token = eps, temp, per_token

    def sft_data(self, k):
        return [Sample("sft", c, target_ids(c), 1.0, [0.0] * RESP_LEN, -1, "teacher")
                for _ in range(k) for c in TEACHER_CTX]

    def _sample_resp(self, c):
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

    def group(self, c, version, g):
        """采一个 context 的一组 g 条响应，返回 (samples, is_dead, rs, resps)。
        dense reward 逐位可分解 → 逐 token 组内相对优势（L1 的 credit assignment，
        也是 dense rule reward 的第一重红利）；sparse/RM 只有标量 → outcome 级
        优势广播到 4 个 token（GRPO 形态）。Sample.adv 统一为长 4 的列表。"""
        resps = [self._sample_resp(c) for _ in range(g)]
        rs = [self.reward_fn(c, r) for r in resps]
        mean_r = sum(rs) / g
        dead = (max(rs) - min(rs)) < 1e-12
        if self.per_token and not dead:
            # dense：逐位匹配向量 → 逐位组内均值 → 每 token 一个优势
            t = TARGETS[c]
            matches = [[1.0 if VOCAB[tok] == t[j] else 0.0 for j, tok in enumerate(r)]
                       for r in resps]
            mean_j = [sum(m[j] for m in matches) / g for j in range(RESP_LEN)]
            advs = [[m[j] - mean_j[j] for j in range(RESP_LEN)] for m in matches]
        else:
            advs = [[r - mean_r] * RESP_LEN for r in rs]
        samps = [Sample("rl", c, r, r_, a, version, self.rsrc)
                 for r, r_, a in zip(resps, rs, advs)]
        return samps, dead, rs, resps

    def rl_rollout(self, version, g):
        """每 ctx 一组 g 条；组内算完 adv 后跨 context 交错入池（L1 §7 教训：
        随机 select 下顺序影响已小，仍保持 L1 的入池形态）。"""
        per_ctx, n_dead, rewards, resps = [], 0, [], []
        for c in range(N_CTX):
            samps, dead, rs, rr = self.group(c, version, g)
            per_ctx.append(samps)
            n_dead += int(dead)
            rewards.extend(rs)
            resps.extend(rr)
        all_s = [per_ctx[c][i] for i in range(g) for c in range(N_CTX)]
        return all_s, n_dead, rewards, resps


class Buffer:
    """与 L1 同构：容量 → 随机 select → mark_trained → prune。"""

    def __init__(self, capacity, max_reuse):
        self.capacity, self.max_reuse = capacity, max_reuse
        self.items = []
        self.n_added = self.n_trainings = self.n_retired = 0

    def add(self, samples):
        for s in samples:
            self.items.append(s)
            self.n_added += 1
        while len(self.items) > self.capacity:
            self.items.pop(0)
            self.n_retired += 1

    def select(self, batch_size):
        n = min(batch_size, len(self.items))
        idx = torch.randperm(len(self.items))[:n].tolist()
        return [self.items[i] for i in idx]

    def mark_trained(self, batch, version):
        for s in batch:
            s.trains += 1
            self.n_trainings += 1

    def prune(self):
        keep = [s for s in self.items if s.trains < self.max_reuse]
        self.n_retired += len(self.items) - len(keep)
        self.items = keep


class Trainer:
    """与 L1 同构：SFT 交叉熵 + RL −adv·logp（+ 可选 β·KL）。reward 来源的变化
    完全不进入这里——它只消费 Sample.adv。"""

    def __init__(self, model, lr, beta=0.0):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.beta = beta
        self.version = 0

    def step(self, batch, ref=None):
        ids = torch.tensor([[BOS] + prompt_ids(s.ctx) + s.resp + [EOS] for s in batch])
        logits = self.model(ids)
        logp = F.log_softmax(logits, dim=-1)
        sft = [i for i, s in enumerate(batch) if s.kind == "sft"]
        rl = [i for i, s in enumerate(batch) if s.kind == "rl"]
        loss_sft = loss_rl = torch.tensor(0.0)
        if sft:
            loss_sft = F.nll_loss(logp[sft, 3:8].reshape(-1, len(VOCAB)),
                                  ids[sft, 4:9].reshape(-1))
        if rl:
            tok_logp = logp[rl, 3:7].gather(2, ids[rl, 4:8].unsqueeze(-1)).squeeze(-1)
            # adv 统一为 (n,4)：dense 逐位 / sparse、RM outcome 级广播（explorer 产）
            adv = torch.tensor([batch[i].adv for i in rl])
            loss_rl = -(adv * tok_logp).sum(dim=1).mean()
            if ref is not None and self.beta > 0:
                with torch.no_grad():
                    ref_logp = F.log_softmax(ref(ids), dim=-1)
                kl = (logp[rl, 3:7].exp() * (logp[rl, 3:7] - ref_logp[rl, 3:7])).sum(-1)
                loss_rl = loss_rl + self.beta * kl.mean()
        loss = loss_sft + loss_rl
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.version += 1
        return len(sft), len(rl)


# ----------------------------- checkpoint（内存级，L2 不需要落盘） -----------------------------
def snapshot(model, trainer):
    return dict(model={k: v.clone() for k, v in model.state_dict().items()},
                opt=deepcopy(trainer.opt.state_dict()),
                rng=torch.get_rng_state().clone())


def restore(snap, lr, beta=0.0):
    model = TinyGPT()
    model.load_state_dict({k: v.clone() for k, v in snap["model"].items()})
    trainer = Trainer(model, lr, beta)
    trainer.opt.load_state_dict(deepcopy(snap["opt"]))
    torch.set_rng_state(snap["rng"].clone())
    return model, trainer


# ----------------------------- 统一 RL 循环（reward = config 字段） -----------------------------
def rl_run(cfg, snap, reward_fn, rsrc, rounds, beta=0.0, dynamic=None, log_every=1):
    """与 L1 run() 同构的循环；reward_fn 是唯一随 config 变的东西。
    dynamic: None 或 dict(target_live, max_extra) —— nano 版 std_threshold 过滤 +
    补采（DAPO Dynamic Sampling / Trinity grpo_advantage.py:L160-163 的语义）：
     nominal G 采一组，dead 组丢弃，live 组不足 target_live 就继续补采，
    补采批数上限 max_extra（rollout 预算封顶）。
    返回 (model, hist)；hist 每行含 proxy 均值 / dead 率 / gold 指标 / 'a' 含量。"""
    model, trainer = restore(snap, cfg["lr"], beta)
    buffer = Buffer(cfg["capacity"], cfg["max_reuse"])
    explorer = Explorer(model, reward_fn, rsrc, eps=cfg["eps"], temp=cfg.get("temp", 1.0),
                        per_token=cfg.get("per_token", False))
    ref = None
    if beta > 0:
        ref = deepcopy(model)
        for p in ref.parameters():
            p.requires_grad_(False)
    hist = []
    total_rollouts = 0
    for r in range(rounds):
        # 探索退火（L1 同口径）
        frac = min(1.0, r / max(1, rounds - 1))
        explorer.eps = cfg["eps"] * (1 - 0.66 * frac)
        for pg in trainer.opt.param_groups:
            pg["lr"] = cfg["lr"] * cfg.get("rl_lr_scale", 0.15)
        fresh, n_dead, rewards, resps = explorer.rl_rollout(trainer.version, cfg["group"])
        n_groups = N_CTX
        n_roll = cfg["group"] * N_CTX
        if dynamic is not None:
            # nano 版 std_threshold 过滤：dead 组（组内 reward 全同）整组丢弃，
            # live 组不足 target_live 就补采，补采批数上限 max_extra（预算封顶）。
            def live_of(samps):
                by_ctx = {}
                for s in samps:
                    by_ctx.setdefault(s.ctx, []).append(s)
                kept = []
                for c in sorted(by_ctx):
                    gs = by_ctx[c]
                    rs_c = [s.reward for s in gs]
                    if max(rs_c) - min(rs_c) >= 1e-12:
                        kept.extend(gs)
                return kept

            kept_samps = live_of(fresh)
            extra = 0
            while len(kept_samps) // cfg["group"] < dynamic["target_live"] \
                    and extra < dynamic["max_extra"]:
                more_s, more_dead, more_r, more_resp = explorer.rl_rollout(
                    trainer.version, cfg["group"])
                rewards.extend(more_r)
                resps.extend(more_resp)
                n_dead += more_dead
                n_groups += N_CTX
                n_roll += cfg["group"] * N_CTX
                kept_samps.extend(live_of(more_s))
                extra += 1
            fresh = kept_samps
        total_rollouts += n_roll
        buffer.add(fresh)
        for _ in range(cfg["steps_per_round"]):
            batch = buffer.select(cfg["batch_size"])
            if not batch:
                break
            trainer.step(batch, ref=ref)
            buffer.mark_trained(batch, trainer.version)
            buffer.prune()
        exact, characc, per_ex, per_ca = evaluate(model)
        a_cnt = sum(1 for rr in resps for tok in rr if VOCAB[tok] == "a") / max(1, len(resps))
        hist.append(dict(round=r + 1,
                         proxy=sum(rewards) / max(1, len(rewards)),
                         dead=n_dead / n_groups,
                         exact=exact, characc=characc,
                         per_ex=per_ex, per_ca=per_ca,
                         a_per_resp=a_cnt, n_roll=n_roll,
                         version=trainer.version))
    return model, hist, total_rollouts


def fmtc(v):
    return "[" + " ".join(f"{x:.2f}" for x in v) + "]"


# ----------------------------- reward model -----------------------------
def rm_features(c, resp_ids):
    """RM 输入 = context one-hot(6) ⊕ 逐位字符 one-hot(4×5，含 other 桶) = 26 维。
    RM 看得到 context（真实 RM 看 prompt+response），但看不到 target。
    响应可能含字母表外 token（策略采样全词表）→ 归入 other 桶。"""
    f = torch.zeros(6 + RESP_LEN * 5)
    f[c] = 1.0
    for j, tok in enumerate(resp_ids):
        ch = VOCAB[tok]
        k = ALPHABET.index(ch) if ch in ALPHABET else 4
        f[6 + j * 5 + k] = 1.0
    return f


class TinyRM(nn.Module):
    """26 → 32(tanh) → 1 的小打分器。真实 RM 是同架构族的 LLM
    （InstructGPT arXiv:2203.02155），这里用最小 MLP 演同一学习过程。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(26, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def collect_preference_data(snap, cfg, n_per_ctx=400, eps=0.5):
    """从 warm 策略采响应池，构造成对偏好。标注者模型（显式声明）：
    - |q1−q2| ≥ 0.25：以 1−noise 概率标对（q 高者胜）；
    - q1 == q2（同分对）：标注者退回启发式——以 bias_p 偏好含 'a' 更多的一条；
      'a' 数相同则抛硬币。这是注入的系统性偏置（真实对应物：标注者的表面
      偏好渗进偏好数据，RM 把它学走）。"""
    model, _ = restore(snap, cfg["lr"])
    explorer = Explorer(model, reward_dense, "probe", eps=eps, temp=1.0)
    pool = {c: [] for c in range(N_CTX)}
    for c in range(N_CTX):
        for _ in range(n_per_ctx):
            resp = explorer._sample_resp(c)
            pool[c].append((resp, reward_dense(c, resp)))
    pairs, test_resps = [], {c: [] for c in range(N_CTX)}
    for c in range(N_CTX):
        items = pool[c]
        n_test = n_per_ctx // 5
        test_resps[c] = items[:n_test]
        train_items = items[n_test:]
        idx = torch.randperm(len(train_items)).tolist()
        for i in range(0, len(idx) - 1, 2):
            (r1, q1), (r2, q2) = train_items[idx[i]], train_items[idx[i + 1]]
            if abs(q1 - q2) >= 0.25:
                winner = 0 if q1 > q2 else 1
                if torch.rand(1).item() < cfg["ann_noise"]:
                    winner = 1 - winner
                kind = "clean"
            else:
                a1 = sum(1 for tok in r1 if VOCAB[tok] == "a")
                a2 = sum(1 for tok in r2 if VOCAB[tok] == "a")
                if a1 != a2:
                    pref_first = (a1 > a2) if (torch.rand(1).item() < cfg["ann_bias_p"]) \
                        else (a1 < a2)
                    winner = 0 if pref_first else 1
                else:
                    winner = int(torch.rand(1).item() < 0.5)
                kind = "tie"
            pairs.append((c, r1, r2, winner, kind))
    return pairs, test_resps


def train_rm(pairs, cfg):
    """Bradley-Terry 成对偏好损失 −log σ(r_w − r_l)（arXiv:1706.03741 的 RM 形态）。"""
    rm = TinyRM()
    opt = torch.optim.Adam(rm.parameters(), lr=cfg["rm_lr"])
    feats_w, feats_l = [], []
    for c, r1, r2, winner, _ in pairs:
        if winner == 0:
            feats_w.append(rm_features(c, r1))
            feats_l.append(rm_features(c, r2))
        else:
            feats_w.append(rm_features(c, r2))
            feats_l.append(rm_features(c, r1))
    Xw = torch.stack(feats_w)
    Xl = torch.stack(feats_l)
    for _ in range(cfg["rm_steps"]):
        loss = -F.logsigmoid(rm(Xw) - rm(Xl)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rm, loss.item()


@torch.no_grad()
def rm_score(rm, c, resp_ids):
    return float(torch.sigmoid(rm(rm_features(c, resp_ids).unsqueeze(0)))[0])


def bias_probe(rm):
    """枚举式偏置探针：对每个 ctx、每个匹配数 k（q=k/4），列出所有恰匹配 k 位
    的响应（C(4,k)·3^k 条），按含 'a' 数分上下两半，比 RM 均分差。
    q 固定 → 差值只能是 'a' 偏置（或同 q 内部的其它表面特征）。"""
    deltas = []
    for c in range(N_CTX):
        t = TARGETS[c]
        for k in (1, 2, 3):
            resps = []
            positions = [0, 1, 2, 3]
            for match_pos in itertools.combinations(positions, k):
                others = [p for p in positions if p not in match_pos]
                choices = [[ch for ch in ALPHABET if ch != t[p]] for p in others]
                for combo in itertools.product(*choices):
                    resp = list("....")
                    for p in match_pos:
                        resp[p] = t[p]
                    for p, ch in zip(others, combo):
                        resp[p] = ch
                    resps.append([ID[ch] for ch in resp])
            scored = [(rm_score(rm, c, r),
                       sum(1 for tok in r if VOCAB[tok] == "a")) for r in resps]
            scored.sort(key=lambda x: x[1])
            half = len(scored) // 2
            lo = scored[:half]
            hi = scored[-half:]
            if hi and lo and hi[-1][1] > lo[0][1]:
                deltas.append(sum(s for s, _ in hi) / len(hi) - sum(s for s, _ in lo) / len(lo))
    return sum(deltas) / len(deltas), len(deltas)


def rm_argmax_scan(rm):
    """对每个 ctx 暴力扫全部 4^4=256 条响应，找 RM 的最爱——与 target 对照。"""
    out = []
    for c in range(N_CTX):
        best_s, best_r = -1.0, None
        for combo in itertools.product(ALPHABET, repeat=RESP_LEN):
            r = [ID[ch] for ch in combo]
            s = rm_score(rm, c, r)
            if s > best_s:
                best_s, best_r = s, "".join(combo)
        out.append((c, best_r, TARGETS[c], best_s, rm_score(rm, c, target_ids(c))))
    return out


# ----------------------------- 主实验 -----------------------------
def main():
    BASE = dict(lr=3e-3, capacity=2048, max_reuse=2, batch_size=32,
                eps=0.3, rl_lr_scale=0.15, ann_noise=0.05, ann_bias_p=0.95,
                rm_lr=3e-2, rm_steps=500)

    print("=" * 76)
    print("nano-trinity-rft L2 — RL 的信号来源：rule reward vs learned reward model")
    print("=" * 76)
    print(f"env: python {sys.version.split()[0]} | torch {torch.__version__} | seed {SEED}")

    # ---- [0] 任务 + 确定性探针 + SFT warm start + warm RL ----
    torch.manual_seed(SEED)
    m0 = TinyGPT()
    n_params = sum(p.numel() for p in m0.parameters())
    x = torch.tensor([[BOS] + prompt_ids(0) + target_ids(0) + [EOS]])
    lg1, lg2 = m0(x), m0(x)
    assert torch.equal(lg1, lg2), "同一输入两次前向必须逐位相同（CPU 确定性）"
    print(f"\n[0] 任务与起点（与 L1 同一张任务表）")
    print(f"targets: {TARGETS}  （teacher 覆盖 ctx {TEACHER_CTX}，ctx 3–5 数据空洞）")
    print(f"model: char-GPT | params={n_params:,} | 确定性探针: 两次前向逐位一致 ✓")

    # SFT 阶段（L1 同配方：3 轮 × sft_k=40 × 24 步）
    torch.manual_seed(SEED)  # 重置到与 L1 完全同源的 RNG 流起点
    model = TinyGPT()
    trainer = Trainer(model, BASE["lr"])
    buffer = Buffer(BASE["capacity"], BASE["max_reuse"])
    sft_exp = Explorer(model, reward_dense, "teacher", eps=BASE["eps"])
    for r in range(3):
        buffer.add(sft_exp.sft_data(40))
        for _ in range(24):
            batch = buffer.select(BASE["batch_size"])
            if not batch:
                break
            trainer.step(batch)
            buffer.mark_trained(batch, trainer.version)
            buffer.prune()
    ex_sft, ca_sft, per_ex_sft, per_ca_sft = evaluate(model)
    print(f"SFT@3r:  exact={ex_sft:.3f} characc={ca_sft:.3f} "
          f"per-ctx characc={fmtc(per_ca_sft)}")

    # warm RL：dense rule reward 填洞到「偶尔会对」水平（为 [1] 的稀疏信号算术
    # 制造真实的 p 窗口——稀疏 RLVR 只在策略有一定胜率时才有信号，数学题同理）
    snap_sft = snapshot(model, trainer)
    CFG_W = dict(BASE, group=64, steps_per_round=24, per_token=True)
    model_w, hist_w, _ = rl_run(CFG_W, snap_sft, reward_dense, "rule_dense", rounds=10)
    # 各 RL 臂的统一起点 = warm 末态权重 + 全新 Adam（opt 状态不带入，臂间公平）
    snap_warm = snapshot(model_w, Trainer(model_w, BASE["lr"]))
    ex_w, ca_w, per_ex_w, per_ca_w = evaluate(model_w)
    print(f"warm RL@10r（dense rule reward, G=64，L1 同口径）: exact={ex_w:.3f} characc={ca_w:.3f}")
    print(f"per-ctx characc: {fmtc(per_ca_w)}  （空洞 ctx 3–5 爬到中途——『偶尔会对』窗口）")

    # 每 ctx 采样估计 exact 胜率 p̂（M=1536/ctx，ε=0.3 与训练同口径）
    M_PROBE = 1536
    probe_resps = {c: [] for c in range(N_CTX)}
    model_p, _ = restore(snap_warm, BASE["lr"])
    exp_p = Explorer(model_p, reward_sparse, "probe", eps=BASE["eps"])
    for c in range(N_CTX):
        for _ in range(M_PROBE):
            resp = exp_p._sample_resp(c)
            probe_resps[c].append((resp, reward_sparse(c, resp), reward_dense(c, resp)))
    p_hat = {c: sum(r for _, r, _ in probe_resps[c]) / M_PROBE for c in range(N_CTX)}
    print(f"每 ctx exact 胜率 p̂（M={M_PROBE}）: " +
          " ".join(f"c{c}:{p_hat[c]:.3f}" for c in range(N_CTX)))

    # ---- [1] 信号算术：dead group 率 = p^G + (1-p)^G ----
    print(f"\n[1] 信号算术：组内 reward 全同（std=0）→ advantage 全 0 → 零梯度")
    print(f"    解析式 P(dead | p, G) = p^G + (1-p)^G（全对组与全错组都无信号）")
    print("G     sparse 实测(dead率, 空洞ctx均值)   sparse 解析    dense 实测(G 同值)")
    hole = [3, 4, 5]
    dead_by_G = {}
    for G in (4, 8, 16, 32, 64):
        emp = []
        for c in hole:
            rs = [r for _, r, _ in probe_resps[c]]
            n_g = M_PROBE // G
            dead = sum(1 for i in range(n_g)
                       if max(rs[i * G:(i + 1) * G]) == min(rs[i * G:(i + 1) * G]))
            emp.append(dead / n_g)
        emp_v = sum(emp) / len(emp)
        ana = sum(p_hat[c] ** G + (1 - p_hat[c]) ** G for c in hole) / len(hole)
        # dense 实测：逐位匹配率全同的组占比（用 dense 列）
        emp_d = []
        for c in hole:
            qs = [q for _, _, q in probe_resps[c]]
            n_g = M_PROBE // G
            dead = sum(1 for i in range(n_g)
                       if max(qs[i * G:(i + 1) * G]) - min(qs[i * G:(i + 1) * G]) < 1e-12)
            emp_d.append(dead / n_g)
        emp_dv = sum(emp_d) / len(emp_d)
        dead_by_G[G] = emp_v
        print(f"{G:<5} {emp_v:>8.3f}                        {ana:>8.3f}      {emp_dv:>8.4f}")

    # ---- [2] 学习后果：稀疏 reward 下 G=8 vs G=64 vs dynamic sampling ----
    print(f"\n[2] 学习后果（同一起点 snap_warm，稀疏 exact reward，12 轮）")
    rounds = 12  # 三臂共用的 RL 轮数；[2] 表「rollouts/轮」列 = rollout 总量 // rounds
    CFG2 = dict(BASE, group=8, steps_per_round=4, batch_size=24)
    print("arm               rollouts/轮   dead率@r1   exact@r12  characc@r12  空洞ctx末态")
    m_g8, h_g8, roll_g8 = rl_run(CFG2, snap_warm, reward_sparse, "rule_sparse", rounds=rounds)
    CFG2_64 = dict(CFG2, group=64)
    m_g64, h_g64, roll_g64 = rl_run(CFG2_64, snap_warm, reward_sparse, "rule_sparse", rounds=rounds)
    CFG2d = dict(CFG2)
    m_dyn, h_dyn, roll_dyn = rl_run(CFG2d, snap_warm, reward_sparse, "rule_sparse", rounds=rounds,
                                    dynamic=dict(target_live=24, max_extra=7))
    for name, h, roll in (("sparse G=8", h_g8, roll_g8),
                          ("sparse G=64", h_g64, roll_g64),
                          ("sparse G=8+dyn", h_dyn, roll_dyn)):
        hole_ex = [h[-1]["per_ex"][c] for c in hole]
        print(f"{name:<16} {roll // rounds:>6}        {h[0]['dead']:.3f}      "
              f"{h[-1]['exact']:.3f}     {h[-1]['characc']:.3f}      {fmtc(hole_ex)}")
    print(f"    dyn = nano 版 std_threshold 过滤 + 补采（rollout 预算封顶 = G64 等价）；")
    print(f"    Trinity 对应物: grpo_advantage.py:L160-163（std≤阈值的组 exps.clear() 跳过）")
    print(f"    + duplicate_experiences 补位（L178-194）；DAPO Dynamic Sampling [2503.14476 §3.2]")

    # ---- [3] learned reward model：BT 偏好训练 + 偏置测量 ----
    print(f"\n[3] learned reward model（从 snap_warm 采偏好数据，Bradley-Terry 训练）")
    pairs, test_resps = collect_preference_data(snap_warm, BASE, n_per_ctx=400)
    n_clean = sum(1 for p in pairs if p[4] == "clean")
    n_tie = len(pairs) - n_clean
    rm, rm_loss = train_rm(pairs, BASE)
    print(f"偏好对: {len(pairs)}（分差≥0.25 的干净对 {n_clean} + 同分对 {n_tie}"
          f"；同分对标注者以 p={BASE['ann_bias_p']} 偏好含 'a' 多者——注入偏置）")
    print(f"RM: 26→32(tanh)→1 | BT loss −log σ(r_w−r_l) | 末步 loss={rm_loss:.4f}")
    # held-out pair accuracy（用真实 q 重新打标，不含偏置）
    n_ok_all = n_ok_clean = n_ok_tie = n_all = n_clean_t = n_tie_t = 0
    corr_s, corr_q = [], []
    for c in range(N_CTX):
        items = test_resps[c]
        for i in range(0, len(items) - 1, 2):
            (r1, q1), (r2, q2) = items[i], items[i + 1]
            if q1 == q2:
                continue
            pred = 0 if rm_score(rm, c, r1) > rm_score(rm, c, r2) else 1
            truth = 0 if q1 > q2 else 1
            n_all += 1
            n_ok_all += int(pred == truth)
            if abs(q1 - q2) >= 0.25:
                n_clean_t += 1
                n_ok_clean += int(pred == truth)
            else:
                n_tie_t += 1
                n_ok_tie += int(pred == truth)
        for r, q in items:
            corr_s.append(rm_score(rm, c, r))
            corr_q.append(q)
    acc_all = n_ok_all / max(1, n_all)
    acc_clean = n_ok_clean / max(1, n_clean_t)
    ms = sum(corr_s) / len(corr_s)
    mq = sum(corr_q) / len(corr_q)
    cov = sum((s - ms) * (q - mq) for s, q in zip(corr_s, corr_q)) / len(corr_s)
    corr = cov / (math.sqrt(sum((s - ms) ** 2 for s in corr_s) / len(corr_s)) *
                  math.sqrt(sum((q - mq) ** 2 for q in corr_q) / len(corr_q)))
    print(f"held-out 对准确率: 全部 {acc_all:.3f}（{n_ok_all}/{n_all}）| "
          f"分差≥0.25 对 {acc_clean:.3f}（{n_ok_clean}/{n_clean_t}）")
    print(f"RM 分与真实匹配率 Pearson r = {corr:.3f}（N={len(corr_q)} held-out 响应）")
    bias_delta, n_probe = bias_probe(rm)
    print(f"偏置探针（固定 q，比含 'a' 多/少两半的 RM 均分差）: Δ = {bias_delta:+.4f}"
          f"（{n_probe} 个 (ctx,k) 探针均值；>0 = RM 学到了 'a' 偏置）")
    scan = rm_argmax_scan(rm)
    fav = [s[1] for s in scan]
    n_fav_wrong = sum(1 for s in scan if s[1] != s[2])
    print(f"RM 的最爱（每 ctx 暴力扫 256 条）: {fav} vs targets {TARGETS}")
    print(f"    {n_fav_wrong}/6 个 ctx 的 RM 最爱不是正确答案——proxy 与 gold 从训练完那天起就不是一回事")

    def reward_rm(c, resp_ids):
        return rm_score(rm, c, resp_ids)

    # ---- [4] Goodhart：只优化 RM 时策略钻偏置空子 ----
    print(f"\n[4] Goodhart 三臂（同一起点 snap_warm，10 轮，G=24；gold 只监测不训练）")
    CFG4 = dict(BASE, group=24, steps_per_round=8)
    print("round  arm          proxy(RM均分)  gold exact  gold characc  'a'/resp")
    m_rule, h_rule, _ = rl_run(dict(CFG4, per_token=True), snap_warm, reward_dense,
                               "rule_dense", rounds=10)
    m_rm, h_rm, _ = rl_run(CFG4, snap_warm, reward_rm, "rm", rounds=10)
    m_kl, h_kl, _ = rl_run(CFG4, snap_warm, reward_rm, "rm", rounds=10, beta=0.2)
    for r_idx in (0, 2, 5, 9):
        for name, h in (("rule_dense", h_rule), ("rm(β=0)", h_rm), ("rm(β=0.2)", h_kl)):
            hrow = h[r_idx]
            print(f"{hrow['round']:>3}    {name:<11}  {hrow['proxy']:.4f}        "
                  f"{hrow['exact']:.3f}      {hrow['characc']:.3f}       {hrow['a_per_resp']:.3f}")
        print()
    print(f"末态 greedy 输出:  rm(β=0) = {[decode(generate(m_rm, c, greedy=True)) for c in range(N_CTX)]}")
    print(f"                   targets = {TARGETS}")
    print(f"    β=0 臂: proxy 持续涨而 gold 反降/停滞——策略找到的是 RM 的错误面而非任务解")
    print(f"    （arXiv:2210.10760 的 gold/proxy 分离在 toy 尺度的复现）；KL 臂把策略锚在")
    print(f"    ref 附近: proxy 照样能涨（RM 与质量相关），但 gold 不坍缩；rule 臂 gold 一路升。")

    # ---- [5] 账本与取舍 ----
    print(f"\n[5] 账本与取舍")
    print(f"成本: rule reward = 纯函数（0 次模型调用）；RM = 偏好对 {len(pairs)} 条 + "
          f"{BASE['rm_steps']} 步训练 + 每条响应 1 次 RM 前向")
    print(f"      且 gold 评测依然在循环里——只是从训练信号变成监测信号"
          f"（Trinity RULER 示例同时记 reward/gold_reward/judge_success/eval_accuracy）")
    print("取舍表:")
    print("  维度        rule(verifiable)              learned RM")
    print("  忠实度      对任务精确（就是任务本身）     proxy：拟合偏好数据，含偏置")
    print("  覆盖面      只有 checker 存在的域          任意域（非可验证域的唯一路径）")
    print("  粒度        可稀可密（exact/逐位/格式）    天然 dense")
    print("  可被 hack   不能（答对就是答对）           能（[4] 实测：钻偏置空子）")
    print("  成本        ~0                             标注/偏好数据 + 训练 + 推理")
    print("  Trinity     注册表 7 个 reward 全 rule 型  非可验证域走 RULER/rubric 示例")

    # ---- self-check ----
    print(f"\nself-check:")
    checks = []
    checks.append((abs(ca_sft - 0.5) < 0.15 and all(per_ca_sft[c] >= 0.9 for c in TEACHER_CTX),
                   f"SFT@3r 覆盖处 characc≥0.9（实测 {fmtc(per_ca_sft[:3])}），聚合≈0.5（空洞贪心基线）"))
    checks.append((hist_w[-1]["characc"] >= 0.7,
                   f"warm RL@10r characc≥0.7（实测 {hist_w[-1]['characc']:.3f}）"))
    checks.append((0.005 < p_hat[3] < 0.45 and 0.005 < p_hat[4] < 0.45 and 0.005 < p_hat[5] < 0.45,
                   "空洞 ctx 胜率 p̂ 落在稀疏信号窗口 (0.005, 0.45)"))
    ana8 = sum(p_hat[c] ** 8 + (1 - p_hat[c]) ** 8 for c in hole) / 3
    checks.append((abs(dead_by_G[8] - ana8) <= 0.08,
                   f"dead 率实测 vs 解析@G=8: |{dead_by_G[8]:.3f} − {ana8:.3f}| ≤ 0.08"))
    ana64 = sum(p_hat[c] ** 64 + (1 - p_hat[c]) ** 64 for c in hole) / 3
    checks.append((abs(dead_by_G[64] - ana64) <= 0.08,
                   f"dead 率实测 vs 解析@G=64: |{dead_by_G[64]:.3f} − {ana64:.3f}| ≤ 0.08"))
    checks.append((dead_by_G[8] >= dead_by_G[64] + 0.15,
                   f"dead 率随 G 单调降: G=8 {dead_by_G[8]:.3f} ≥ G=64 {dead_by_G[64]:.3f} + 0.15"))
    checks.append((dead_by_G[4] >= 0.3,
                   f"稀疏 reward 在小 G 下信号大面积缺失: dead@G=4 = {dead_by_G[4]:.3f} ≥ 0.3"))
    hole_ca_g8 = sum(h_g8[-1]["per_ca"][c] for c in hole) / 3
    hole_ca_g64 = sum(h_g64[-1]["per_ca"][c] for c in hole) / 3
    checks.append((h_g64[-1]["characc"] >= h_g8[-1]["characc"] + 0.05,
                   f"稀疏学习后果: G=64 characc {h_g64[-1]['characc']:.3f} ≥ G=8 {h_g8[-1]['characc']:.3f} + 0.05"))
    checks.append((hole_ca_g64 >= hole_ca_g8 + 0.3,
                   f"空洞 ctx 末态: G=64 {hole_ca_g64:.3f} ≥ G=8 {hole_ca_g8:.3f} + 0.3"
                   f"（小 G 稀疏 RL 净破坏，大 G 保住/填上）"))
    checks.append((h_dyn[-1]["characc"] >= h_g8[-1]["characc"] - 0.05,
                   f"dynamic sampling 不差于固定 G=8（characc）: {h_dyn[-1]['characc']:.3f} vs {h_g8[-1]['characc']:.3f}"))
    checks.append((roll_dyn <= roll_g64,
                   f"dyn rollout 预算封顶生效: {roll_dyn} ≤ G64 的 {roll_g64}"))
    checks.append((acc_clean >= 0.72,
                   f"RM held-out 干净对准确率 ≥ 0.72（实测 {acc_clean:.3f}）"))
    checks.append((corr >= 0.3,
                   f"RM 分与真实匹配率相关 ≥ 0.3（实测 r={corr:.3f}；warm 态 q 值域窄，r 受值域限制）"))
    checks.append((bias_delta > 0.0,
                   f"RM 学到注入的 'a' 偏置: 固定 q 下 Δ={bias_delta:+.4f} > 0"))
    checks.append((n_fav_wrong >= 4,
                   f"RM 的最爱多数不是正确答案: {n_fav_wrong}/6"))
    checks.append((h_rm[-1]["proxy"] >= h_rm[0]["proxy"] + 0.01,
                   f"只优化 RM: proxy 上升 {h_rm[0]['proxy']:.3f} → {h_rm[-1]['proxy']:.3f}（起点已近饱和）"))
    checks.append((h_rm[-1]["characc"] <= h_rm[0]["characc"] + 0.02 and
                   h_rule[-1]["characc"] >= h_rm[-1]["characc"] + 0.1,
                   f"Goodhart 分离: rm 臂 gold 不涨且低于 rule 臂 "
                   f"({h_rm[-1]['characc']:.3f} vs {h_rule[-1]['characc']:.3f})"))
    checks.append((h_rm[-1]["exact"] <= 1 / 3,
                   f"钻空子末态: rm 臂多数 greedy 输出已非 target（exact={h_rm[-1]['exact']:.3f} ≤ 1/3）"))
    checks.append((h_kl[-1]["characc"] >= h_rm[-1]["characc"],
                   f"KL 缓解: β=0.2 gold {h_kl[-1]['characc']:.3f} ≥ β=0 {h_rm[-1]['characc']:.3f}"))
    checks.append((h_kl[-1]["characc"] >= h_kl[0]["characc"] - 0.05,
                   f"KL 锚定: β=0.2 臂 gold 不坍缩（{h_kl[0]['characc']:.3f} → {h_kl[-1]['characc']:.3f}）"))
    checks.append((h_rule[-1]["exact"] >= ex_w,
                   f"rule 对照组 gold 不回退: {ex_w:.3f} → {h_rule[-1]['exact']:.3f}"))
    checks.append((all(0.0 < h["proxy"] < 1.0 for h in h_rm),
                   "RM reward 经 sigmoid 有界于 (0,1)（rm 臂全部轮次 proxy 在界内）"))
    n_pass = sum(ok for ok, _ in checks)
    for ok, msg in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {msg}")
    assert n_pass == len(checks), f"self-check failed: {len(checks) - n_pass} item(s)"
    print(f"    ✅ self-check passed ({n_pass}/{len(checks)})")

    digest_src = {
        "targets": TARGETS,
        "sft": {"exact": round(ex_sft, 6), "characc": round(ca_sft, 6),
                "per_ca": [round(v, 6) for v in per_ca_sft]},
        "warm": {"exact": round(ex_w, 6), "characc": round(ca_w, 6),
                 "per_ca": [round(v, 6) for v in per_ca_w]},
        "p_hat": {str(c): round(p_hat[c], 6) for c in range(N_CTX)},
        "dead_by_G": {str(g): round(v, 6) for g, v in dead_by_G.items()},
        "arms2": {
            "g8": {"exact": round(h_g8[-1]["exact"], 6), "characc": round(h_g8[-1]["characc"], 6),
                   "roll": roll_g8},
            "g64": {"exact": round(h_g64[-1]["exact"], 6), "characc": round(h_g64[-1]["characc"], 6),
                    "roll": roll_g64},
            "dyn": {"exact": round(h_dyn[-1]["exact"], 6), "characc": round(h_dyn[-1]["characc"], 6),
                    "roll": roll_dyn}},
        "rm": {"loss": round(rm_loss, 6), "acc_all": round(acc_all, 6),
               "acc_clean": round(acc_clean, 6), "corr": round(corr, 6),
               "bias_delta": round(bias_delta, 6), "n_fav_wrong": n_fav_wrong,
               "fav": fav},
        "arms4": {
            "rule": [{"round": h["round"], "proxy": round(h["proxy"], 6),
                      "exact": round(h["exact"], 6), "characc": round(h["characc"], 6),
                      "a": round(h["a_per_resp"], 6)} for h in h_rule],
            "rm": [{"round": h["round"], "proxy": round(h["proxy"], 6),
                    "exact": round(h["exact"], 6), "characc": round(h["characc"], 6),
                    "a": round(h["a_per_resp"], 6)} for h in h_rm],
            "kl": [{"round": h["round"], "proxy": round(h["proxy"], 6),
                    "exact": round(h["exact"], 6), "characc": round(h["characc"], 6),
                    "a": round(h["a_per_resp"], 6)} for h in h_kl]},
    }
    digest = hashlib.md5(json.dumps(digest_src, sort_keys=True).encode()).hexdigest()
    print(f"\ndigest(md5 of metrics) = {digest}")
    print(f"elapsed: {time.time() - T0:.1f}s（计时行随机器负载浮动，指标行逐字节确定）")


if __name__ == "__main__":
    main()
