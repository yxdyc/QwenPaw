#!/usr/bin/env python3
"""nano-opd L3 — 生产配方的四个旋钮：对照 survey taxonomy 与 Qwen3 / MiMo-V2-Flash /
Thinking Machines / DeepSeek-V4 的生产选择

L0–L2 已经证明 OPD 的核心算术（散度选择 → 采样需求 → 多教师融合/路由）。L3 问的是
生产问题：**同一个 on-policy 循环上有四个可调旋钮，生产配方各自拧到了哪里、为什么？**

四个旋钮对应 OPD survey [arXiv:2604.00626] 的机制轴（survey 自身 taxonomy = 三正交轴：
feedback signal × teacher access × loss granularity，外加统一 f-divergence 框架；
本文件的「四轴」是教学重组，映射见 tutorial §8）：

  轴1 divergence 设计——同一个学生自采样循环上换 token 级散度：
      forward KL（teacher-first，mode-covering）/ reverse KL（student-first，
      mode-seeking）/ 两者的凸插值 / 对称 JSD（有界 ∈[0, ln2]，survey：「bounded…
      stable gradient field…preventing extreme gradient explosions」）/ 熵门控自适应
      （低熵位 reverse、高熵位 forward，survey §4.1.1 entropy-gated mixture 的
      机制类别）。采样需求不变——散度只换「批作业怎么改」，不换「谁来写作业」
      （统一目标里采样分布与散度选择解耦）。
  轴2 信号源（白盒/黑盒）——教师接口两种形态：
      白盒：教师给出全词表分布（token 级 KL 解析梯度，需要教师 logits）；
      黑盒：教师只对学生采样的 token 逐个打 logprob（API 形态，compute_logprobs）。
      恒等式（本文件机器精度验证）：∇KL(q‖p) = E_{y~q}[(log q − log p)·∇log q]
      ——reverse KL 的梯度可以只用「采样 token 的 logprob」无偏估计。这正是
      Thinking Machines 生产配方（per-token advantage = 负 reverse KL + RL IS loss，
      教师只做 compute_logprobs、不透传 logits）可行的算术原因；代价是方差
      （DeepSeek-V4 报告自述：token 级 advantage 路线「resource-efficient…high
      variance…training instability」，因此它选 full-vocabulary 白盒路线）。
      本文件在同一 toy 上跑三条路线：白盒 / 黑盒裸 REINFORCE（每位置 1 样本，
      实测确定性崩溃——方差爆炸 + OOD 前缀自强化，作为一流反例教材保留）/
      黑盒 + group 采样（每 prompt 采 16 条再平均，GRPO group-sampling 机制
      类别，实测学会）。「黑盒可行性 = 恒等式 + 方差代价」：方差代价不是抽象
      的——本 toy 里它 = 教师打分调用 ×16 且同步数预算终态仍逊于白盒。
  轴3 token 加权——同一个散度下，哪些 token 的 loss 算数：
      均匀 / 按 student–teacher gap 动态加权（survey §4.1.2 AdaKD 机制类别）/
      EMA 尺度相对加权（DistiLLM adaptive loss 机制类别的 nano 构造，有界权重）
      + 其归一化形态（ℓ/EMA 除法，无界有效步长）作为实测反例。
      加权不改变不动点（w_t>0 有界时 q=p 仍是驻点），改变的是路径与预算分配；
      「把尺度差当信号」（gap）与「把尺度差当噪声」（EMA 归一化族）是两个
      实测可分的方向——本 toy 的封闭码本上尺度差携带信息，归一化反而有害。
  轴4 效率与稳定化——OPD 比 SFT 贵的两处是学生采样与教师打分（survey §8 成本算例：
      on-policy 较 off-policy 4–5× overhead）：
      rollout 复用 K（采一次、K 步梯度——复用即 off-policy，陈旧度随 K 增长，
      L0 off-policy 偏差定理的序列空间版）；clipped IS 修正（部分修复——修偏差
      付方差，容量边缘学生上实测为负收益）；advantage 裁剪 τ（Kimi K3 生产形态：
      「clipping threshold to constrain extreme advantage signals, thereby
      stabilizing RL training」）——τ 须与 advantage 尺度匹配：τ=2 实测剪掉
      信号主体（终态崩坏），τ=5 只剪尾部（压 loss 波动、终态等价）。

任务与模型（承 L1：双语密码本，单教师）：
  4 位数字 → 小写 codebook A 或大写 codebook B 都算对；一个双语教师两套都会
  （高置信：自采样 total ≈ 0.99，自样本背书 ≈ −0.09 nats/token——比 L2 立场
  教师的 −0.001 温和，双语教师对两模都近确定性但不极端）。学生 4,914 参数
  （教师/学生 ≈ 22x），容量受限 → mode-seeking / mode-covering 的结局差异
  可见（L0/L1 已证）。

显式声明（反幻觉）：
  1. 本文件无 mock：教师/学生的前向/反向/采样全部是真实 torch 计算；「黑盒 API」
     是接口约束（teacher_logprobs_api 只向外暴露采样 token 的 logprob），不是假数据。
  2. 全部数字为现场实跑（CPU，固定 seed），不是任何真实模型的 benchmark；
     生产数字（GPU 小时 / FLOPs / benchmark 分数）只在 tutorial 中出现，并标注
     [blog claims] / [survey 算例] 口径。
  3. 断言分两类（见 [6] self-check 与 tutorial §7）：数学事实类锚定独立推导
     （恒等式 / JSD 族性质 / 不动点不变量）；动力学结局类为**实测派生断言**
     ——在固定 seed 确定性复现的前提下，锚定该实现的结局类别（崩溃 / 学会 /
     损害可测 / 谱序），阈值留有余量，不是对单个浮点值的拟合。

依赖仅 torch（CPU 即跑）。固定 seed，stdout 不含墙钟与本机路径，可逐字节复验。
"""
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 词表与任务（承 L1 双语密码本，V=66） ----------------

SPECIALS = ["<pad>", "<in>", "<out>", "<end>"]
DIGITS = [str(i) for i in range(10)]
LOWER = [chr(ord("a") + i) for i in range(26)]
UPPER = [chr(ord("A") + i) for i in range(26)]
VOCAB = SPECIALS + DIGITS + LOWER + UPPER
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for i, t in enumerate(VOCAB)}
V = len(VOCAB)
PAD, IN, OUT, END = (TOK2ID[t] for t in ("<pad>", "<in>", "<out>", "<end>"))

CONTENT_LEN = 4
RESP_LEN = 2 * CONTENT_LEN + 1          # 8 字母 + <end> = 9

# 与 L1/L2 相同的固定置换（跨机器逐字节复现）
_PERM_A = [7, 2, 19, 11, 24, 0, 15, 9, 21, 4, 17, 12, 25, 3, 14, 8, 22, 1, 18, 10,
           23, 5, 16, 6, 20, 13]
_PERM_B = [13, 20, 6, 16, 5, 23, 10, 18, 1, 22, 8, 14, 3, 25, 12, 17, 4, 21, 9, 15,
           0, 24, 11, 19, 2, 7]
CODEBOOK_A = {d: (LOWER[_PERM_A[2 * d]], LOWER[_PERM_A[2 * d + 1]]) for d in range(10)}
CODEBOOK_B = {d: (UPPER[_PERM_B[2 * d]], UPPER[_PERM_B[2 * d + 1]]) for d in range(10)}


def decode(ids):
    return "".join(ID2TOK[int(i)] for i in ids)


def make_prompt_ids(digits):
    return [IN] + list(digits) + [OUT]


def target_response(digits, mode):
    cb = CODEBOOK_A if mode == "A" else CODEBOOK_B
    toks = [t for d in digits for t in cb[d]] + ["<end>"]
    return [TOK2ID[t] for t in toks]


def make_pool(seed, n):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 10, (CONTENT_LEN,), generator=g).tolist()
            for _ in range(n)]


# ---------------- 模型（与 L1/L2 同款 TinyLM） ----------------

class TinyLM(nn.Module):
    def __init__(self, d_model, nhead, layers, dim_ff, max_len=32):
        super().__init__()
        self.tok = nn.Embedding(V, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, V)

    def forward(self, ids):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        h = self.tok(ids) + self.pos(pos)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        h = self.enc(h, mask=mask)
        return self.head(h)

    @torch.no_grad()
    def sample(self, prompt_ids, resp_len, temperature=1.0):
        ids = prompt_ids.clone()
        for _ in range(resp_len):
            logits = self.forward(ids)[:, -1, :] / temperature
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids[:, prompt_ids.shape[1]:]


def seq_logprob(model, prompt_ids, resp_ids):
    """序列级 log p(resp|prompt)，逐 token 求和。"""
    full = torch.cat([prompt_ids, resp_ids], dim=1)
    lp = F.log_softmax(model(full), dim=-1)
    Lp = prompt_ids.shape[1]
    pred = lp[:, Lp - 1: Lp - 1 + resp_ids.shape[1], :]
    return pred.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)


# ---------------- token 级散度族（轴1） ----------------

def kl_tok(lq, lp, direction):
    """token 级 KL。lq/lp: [N, V] 对数概率（q=学生，p=教师）。
    direction='rev': KL(q‖p) = Σ q(log q − log p)（student-first，mode-seeking）
    direction='fwd': KL(p‖q) = Σ p(log p − log q)（teacher-first，mode-covering）"""
    if direction == "rev":
        return (lq.exp() * (lq - lp)).sum(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def jsd_beta_tok(lq, lp, beta):
    """β-JSD 族：JSD_β(p‖q) = β·KL(p‖m_β) + (1−β)·KL(q‖m_β)，
    m_β = β·p + (1−β)·q（对数域 logsumexp）。β=0.5 即对称 JSD（有界 ≤ ln 2）。
    端点标度性质（本文件探针机器验证，β=1e-5 均值口径）：β→0 时 JSD_β/β →
    KL(p‖q)（fwd）；β→1 时 JSD_β/(1−β) → KL(q‖p)（rev）——插值轴的两个端点
    恰是 fwd/rev KL。
    （该族与 GKD [2306.13649] 的广义 JSD 同族；这里不复刻 GKD 的 β 参数化，
    只验证当前明确定义的公式。）"""
    lb = math.log(beta)
    l1mb = math.log(1 - beta)
    lm = torch.logsumexp(torch.stack([lp + lb, lq + l1mb]), dim=0)
    t1 = (lp.exp() * (lp - lm)).sum(-1)          # KL(p‖m_β)
    t2 = (lq.exp() * (lq - lm)).sum(-1)          # KL(q‖m_β)
    return beta * t1 + (1 - beta) * t2


def mix_tok(lq, lp, beta):
    """nano 凸插值轴（文档化的 nano 构造，非引文）：
    D_β = (1−β)·FKL + β·RKL。β=0 纯 forward，β=1 纯 reverse。"""
    return (1 - beta) * kl_tok(lq, lp, "fwd") + beta * kl_tok(lq, lp, "rev")


# ---------------- 教师与评测 ----------------

TEACHER = None  # main() 注入（评测函数共用，避免到处传参）


def train_teacher(pool, steps=600, batch=32, lr=2e-3, seed=101,
                  d_model=64, nhead=4, layers=2, ff=256):
    """双语教师：每个样本随机选 A/B 密码本作目标（两套都学）。"""
    torch.manual_seed(seed)
    model = TinyLM(d_model, nhead, layers, ff)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        idx = torch.randint(0, len(pool), (batch,), generator=g).tolist()
        mode = ["A" if r < 0.5 else "B"
                for r in torch.rand(batch, generator=g).tolist()]
        x = torch.tensor([make_prompt_ids(pool[i]) for i in idx])
        y = torch.tensor([target_response(pool[i], mode[j])
                          for j, i in enumerate(idx)])
        logits = model(torch.cat([x, y[:, :-1]], dim=1))
        Lp = x.shape[1]
        loss = F.cross_entropy(
            logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate_bilingual(student, pool, m_per_prompt=8, seed=902):
    """学生自采样 → 与两套 codebook 精确匹配。返回 valid_A/valid_B/total/
    frac_A/lock/背书（教师对学生样本逐 token 打分）。
    lock = |frac_A − 0.5|×2：0 = 两模均衡，1 = 完全锁一模（total≈0 时是噪声，
    只在 total 显著时解读——承 L2 docstring 警告）。"""
    torch.manual_seed(seed)
    student.eval()
    sub = pool[:64]
    px = torch.tensor([make_prompt_ids(d) for d in sub])
    reps = torch.stack(
        [student.sample(px, RESP_LEN) for _ in range(m_per_prompt)], dim=1)
    tgt_a = torch.tensor([target_response(d, "A") for d in sub])
    tgt_b = torch.tensor([target_response(d, "B") for d in sub])
    n_a = (reps == tgt_a.unsqueeze(1)).all(-1).sum().item()
    n_b = (reps == tgt_b.unsqueeze(1)).all(-1).sum().item()
    tot_n = len(sub) * m_per_prompt
    k = min(8, m_per_prompt)
    sub_rep = reps[:, :k, :].reshape(-1, RESP_LEN)
    px_rep = px.unsqueeze(1).expand(-1, k, -1).reshape(-1, px.shape[1])
    endo = (seq_logprob(TEACHER, px_rep, sub_rep) / RESP_LEN).mean().item()
    va, vb = n_a / tot_n, n_b / tot_n
    tot = va + vb
    frac = va / max(tot, 1e-9)
    return {"valid_A": va, "valid_B": vb, "valid_total": tot,
            "mode_frac_A": frac, "lock": abs(frac - 0.5) * 2, "endorse": endo}


@torch.no_grad()
def position_agreement(student, teacher, pool, n_prompts=64):
    """teacher-forced gold(A) 序列上，学生逐位置 argmax 与 gold 的一致率。
    返回 (字母位均值, <end>位, 逐位置 loss 均值列表[9])。"""
    student.eval(); teacher.eval()
    sub = pool[:n_prompts]
    px = torch.tensor([make_prompt_ids(d) for d in sub])
    ry = torch.tensor([target_response(d, "A") for d in sub])
    full = torch.cat([px, ry[:, :-1]], dim=1)
    Lp = px.shape[1]
    s_logits = student(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    t_logits = teacher(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    pred = s_logits.argmax(-1)
    match = (pred == ry).float()
    lq = F.log_softmax(s_logits.reshape(-1, V), -1)
    lp = F.log_softmax(t_logits.reshape(-1, V), -1)
    pos_loss = kl_tok(lq, lp, "rev").reshape(-1, RESP_LEN).mean(0).tolist()
    return (match[:, :RESP_LEN - 1].mean().item(),
            match[:, RESP_LEN - 1].mean().item(), pos_loss)


def fresh_student(cfg, seed):
    torch.manual_seed(seed)
    return TinyLM(cfg["d_model"], cfg["nhead"], cfg["layers"], cfg["ff"])


# ---------------- 黑盒教师接口（轴2） ----------------

@torch.no_grad()
def teacher_logprobs_api(teacher, px, resp):
    """黑盒教师接口（Thinking Machines compute_logprobs 同形）：输入
    (prompt, 学生采样序列)，只返回学生 token 的逐 token logprob [B, R]——
    全词表 logits 永不离开本函数。这是接口约束，不是 mock：toy 里教师内部
    仍需一次前向才能算 logprob，但调用方拿到的只有 [B, R] 张量。"""
    full = torch.cat([px, resp], dim=1)
    lp = F.log_softmax(teacher(full), dim=-1)
    Lp = px.shape[1]
    pred = lp[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    return pred.gather(-1, resp.unsqueeze(-1)).squeeze(-1)


# ---------------- on-policy 循环：四轴共用 ----------------

def train_opd(student, teacher, pool, steps, batch, lr, seed,
              axis="rev", beta=None, weight="uniform", reuse_k=1,
              use_is=False, blackbox=False, clip_adv=None,
              log_kl=False, ckpt_at=None, group_m=1):
    """统一的 on-policy OPD 循环（学生自采样 → 教师信号 → token 级 loss 反传；
    采样前缀视为常数，承 L1/L2/GKD 约定）。旋钮：
      axis      'rev'/'fwd'/'jsd'/'mix'/'ada'（token 级散度选择，轴1）
      beta      jsd 族的 β / mix 凸插值的 β
      weight    'uniform'/'gap'/'norm'/'norm_div'（token 加权，轴3）
      reuse_k   rollout 复用次数（采一次、K 步梯度，轴4a）
      use_is    复用时 clipped 重要性比修正 q_now/q_gen ≤ 2（轴4b）
      blackbox  True = 黑盒 REINFORCE 路线：教师只经 teacher_logprobs_api，
                loss = Σ_t sg[c_t]·log q(y_t)，c_t = log q(y_t) − log p_T(y_t)
                ——∇KL(q‖p) 的无偏 score-function 估计（恒等式见探针）
      clip_adv  黑盒路线的 advantage 裁剪 |c| ≤ τ（轴4c）
      log_kl    每步记录真实 reverse KL（toy 特权：生产黑盒路线看不到此量）
      ckpt_at   在这些步数记录 position_agreement 快照（轴3 路径对比）
      group_m   黑盒 group 采样（GRPO group-sampling 机制类别）：每 prompt 采
                group_m 条响应、loss 对全部序列平均。每位置样本数 ×group_m →
                REINFORCE 方差 ~1/group_m；代价 = 教师打分调用 ×group_m。
                group_m=1 即裸 per-token REINFORCE（本 toy 实测确定性崩溃，
                见 main [3]——崩溃是方差代价的极端形态，不是实现 bug）。
    返回 (losses, kl_log, ckpts, drift_log, max_tok_loss, clip_frac)；
    clip_frac = 被裁剪 advantage 的 token 占比（clip_adv=None 时为 0）。"""
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    teacher.eval()
    prompts = torch.tensor([make_prompt_ids(d) for d in pool])
    losses, kl_log, ckpts, drift_log = [], [], {}, []
    max_tok_loss = 0.0
    n_clip, n_adv = 0, 0
    ema = torch.ones(RESP_LEN)
    px = resp = api_lp = t_logits = logq_gen = None
    for step in range(steps):
        need_sample = (step % reuse_k == 0)
        if need_sample:
            with torch.no_grad():
                idx = torch.randint(0, len(pool), (batch,), generator=g)
                px0 = prompts[idx]
                if group_m > 1:
                    reps = torch.stack(
                        [student.sample(px0, RESP_LEN)
                         for _ in range(group_m)], dim=1)
                    px = px0.unsqueeze(1).expand(-1, group_m, -1).reshape(
                        batch * group_m, -1)
                    resp = reps.reshape(batch * group_m, RESP_LEN)
                else:
                    px = px0
                    resp = student.sample(px0, RESP_LEN)   # ← 学生自采样
                if blackbox:
                    api_lp = teacher_logprobs_api(teacher, px, resp)
                else:
                    t_logits = teacher(torch.cat([px, resp], dim=1))
                logq_gen = None
                if reuse_k > 1:
                    lg = F.log_softmax(student(torch.cat([px, resp], dim=1)), -1)
                    Lp0 = px.shape[1]
                    logq_gen = lg[:, Lp0 - 1: Lp0 - 1 + RESP_LEN, :].gather(
                        -1, resp.unsqueeze(-1)).squeeze(-1)
        full = torch.cat([px, resp], dim=1)
        Lp = px.shape[1]
        lq = F.log_softmax(
            student(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), dim=-1)
        logq_s = lq.gather(-1, resp.reshape(-1).unsqueeze(-1)).squeeze(-1) \
            .reshape(-1, RESP_LEN)
        if blackbox:
            c = logq_s - api_lp                  # reverse-KL 被积函数
            if clip_adv is not None:
                n_clip += int((c.abs() > clip_adv).sum())
                n_adv += int(c.numel())
                c = c.clamp(-clip_adv, clip_adv)
            adv = c.detach()                     # sg：REINFORCE / 负 reverse-KL advantage
            tok_loss = adv * logq_s              # ∇ = E[c·∇log q]（恒等式，见探针）
        else:
            lp = F.log_softmax(
                t_logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), dim=-1)
            if axis == "rev":
                tok_loss = kl_tok(lq, lp, "rev").reshape(-1, RESP_LEN)
            elif axis == "fwd":
                tok_loss = kl_tok(lq, lp, "fwd").reshape(-1, RESP_LEN)
            elif axis == "jsd":
                tok_loss = jsd_beta_tok(lq, lp, beta).reshape(-1, RESP_LEN)
            elif axis == "mix":
                tok_loss = mix_tok(lq, lp, beta).reshape(-1, RESP_LEN)
            elif axis == "ada":
                # 熵门控（survey §4.1.1 机制类别）：教师熵低 → reverse 精确模仿，
                # 教师熵高 → forward 覆盖全部合理选项。λ = H_t/ln V ∈[0,1]。
                h_t = (-(lp.exp() * lp).sum(-1) / math.log(V)).reshape(-1, RESP_LEN)
                lam = h_t.clamp(0, 1).detach()
                r = kl_tok(lq, lp, "rev").reshape(-1, RESP_LEN)
                f = kl_tok(lq, lp, "fwd").reshape(-1, RESP_LEN)
                tok_loss = (1 - lam) * r + lam * f
            else:
                raise ValueError(axis)
        max_tok_loss = max(max_tok_loss, float(tok_loss.max().detach()))
        if weight == "gap":
            # gap 加权（AdaKD 机制类别）：w_t ∝ sg[token loss]，逐序列均值归一
            w = tok_loss.detach()
            w = w / w.mean(-1, keepdim=True).clamp_min(1e-8)
            loss = (w * tok_loss).sum(-1).mean()
        elif weight == "norm":
            # EMA 尺度相对加权（DistiLLM adaptive loss 机制类别的 nano 构造，
            # 有界权重）：w_t = sg[ℓ_t/EMA_t]，EMA_t = 位置 t 损失运行均值。
            # token 按其相对运行尺度的平方加权——把尺度差当「要放大的信号」。
            # 实测结局见 main [4]：慢于均匀、扩大逐位离散（tutorial §5）。
            pos_mean = tok_loss.detach().mean(0)
            ema = 0.9 * ema + 0.1 * pos_mean
            w = tok_loss.detach() / ema.clamp_min(1e-6)
            loss = (w * tok_loss).sum(-1).mean()
        elif weight == "norm_div":
            # 同族的真归一化形态（ℓ/EMA 除法，实测反例）：有效步长 ∝ 1/EMA_t，
            # EMA 衰减处步长发散——「w_t>0 ⇒ 不动点不变」的有界性前提被违反。
            pos_mean = tok_loss.detach().mean(0)
            ema = 0.9 * ema + 0.1 * pos_mean
            loss = (tok_loss / ema.clamp_min(1e-6)).sum(-1).mean()
        else:
            loss = tok_loss.sum(-1).mean()
        if use_is and not need_sample and logq_gen is not None:
            # clipped IS 修正（轴4b）：ratio = sg[min(q_now/q_gen, 2)]
            ratio = (logq_s.detach() - logq_gen).exp().clamp_max(2.0)
            loss = (ratio * tok_loss).sum(-1).mean()
        if reuse_k > 1 and not need_sample and logq_gen is not None:
            drift_log.append(float((logq_s.detach() - logq_gen).abs().mean()))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if log_kl:
            with torch.no_grad():
                lq2 = F.log_softmax(
                    student(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), -1)
                if blackbox:
                    lt = F.log_softmax(
                        teacher(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), -1)
                else:
                    lt = F.log_softmax(
                        t_logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), -1)
                kl_log.append(float(kl_tok(lq2, lt, "rev").mean()))
        if ckpt_at is not None and (step + 1) in ckpt_at:
            la, le, pl = position_agreement(student, teacher, pool)
            ckpts[step + 1] = {"letters": la, "end": le, "pos_loss": pl}
    clip_frac = n_clip / max(n_adv, 1)
    return losses, kl_log, ckpts, drift_log, max_tok_loss, clip_frac


# ---------------- 探针（无需训练） ----------------

@torch.no_grad()
def probe_jsd_math(student, teacher, pool, n_prompts=64, seed=901):
    """β-JSD 族的三条数学性质（真实分布上数值验证）：
    (1) 对称性：β=0.5 时 JSD(p‖q) == JSD(q‖p)；
    (2) 端点标度：β=1e-3 时 JSD_β/β ≈ KL(p‖q)（fwd）；β=1−1e-3 时
        JSD_β/(1−β) ≈ KL(q‖p)（rev）——插值轴两端恰是 fwd/rev KL；
    (3) 有界性：对称 JSD ∈ [0, ln 2]。"""
    torch.manual_seed(seed)
    student.eval(); teacher.eval()
    sub = pool[:n_prompts]
    px = torch.tensor([make_prompt_ids(d) for d in sub])
    ry = torch.tensor([target_response(d, "A") for d in sub])
    full = torch.cat([px, ry[:, :-1]], dim=1)
    Lp = px.shape[1]
    lq = F.log_softmax(student(full), -1)[:, Lp - 1: Lp - 1 + RESP_LEN, :] \
        .reshape(-1, V)
    lp = F.log_softmax(teacher(full), -1)[:, Lp - 1: Lp - 1 + RESP_LEN, :] \
        .reshape(-1, V)
    sym_abs = (jsd_beta_tok(lp, lq, 0.5) - jsd_beta_tok(lq, lp, 0.5)).abs() \
        .max().item()
    b = 1e-5
    fwd = kl_tok(lq, lp, "fwd")
    rev = kl_tok(lq, lp, "rev")
    # 端点标度用均值相对差断言：O(β) 余项 ∝ χ²，off-manifold token 上极大，
    # max 口径过松；均值是稳健聚合（max 一并返回供参考）。
    d0 = (jsd_beta_tok(lq, lp, b) / b - fwd).abs()
    d1 = (jsd_beta_tok(lq, lp, 1 - b) / b - rev).abs()
    scale0 = d0.mean().item() / max(float(fwd.mean()), 1e-8)
    scale1 = d1.mean().item() / max(float(rev.mean()), 1e-8)
    jsd_sym = jsd_beta_tok(lq, lp, 0.5)
    return {"sym_max_abs": sym_abs, "scale_fwd_rel": scale0,
            "scale_rev_rel": scale1,
            "scale_fwd_max": d0.max().item() / max(float(fwd.max()), 1e-8),
            "scale_rev_max": d1.max().item() / max(float(rev.max()), 1e-8),
            "jsd_max": float(jsd_sym.max()),
            "jsd_min": float(jsd_sym.min()), "ln2": math.log(2.0)}


def probe_gradient_identity(student, teacher, pool, n_prompts=8, seed=903):
    """黑盒可行性的算术核心（固定学生、学生自采样前缀、单步）：
    恒等式 ∇_θ Σ_t KL(q_t‖p_t) = Σ_t E_{y~q_t}[(log q_t − log p_t)·∇_θ log q_t(y)]
    三种算法对照：
      g_exact —— 白盒解析梯度（autograd，全词表 KL）；
      g_enum  —— 恒等式右端对全词表精确求和（toy 特权：V=66 可枚举）
                 → 与 g_exact 机器精度吻合 = 恒等式本身成立；
      g_mc    —— 黑盒 MC 估计（每位置 N 个采样 token 的 logprob，
                 teacher_logprobs_api 同形接口）→ 无偏、余弦应近 1。"""
    torch.manual_seed(seed)
    student.eval(); teacher.eval()
    sub = pool[:n_prompts]
    px = torch.tensor([make_prompt_ids(d) for d in sub])
    with torch.no_grad():
        resp = student.sample(px, RESP_LEN)
    full = torch.cat([px, resp], dim=1)
    Lp = px.shape[1]
    with torch.no_grad():
        lp = F.log_softmax(
            teacher(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), -1)

    def flat_grad(loss):
        gs = torch.autograd.grad(loss, list(student.parameters()))
        return torch.cat([gg.detach().reshape(-1) for gg in gs])

    # g_exact：白盒解析梯度
    student.zero_grad()
    lq = F.log_softmax(
        student(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), -1)
    g_exact = flat_grad(kl_tok(lq, lp, "rev").mean())

    # g_enum：恒等式右端精确枚举——Σ_y q(y)·c(y)·∇log q(y)，c = log q − log p
    student.zero_grad()
    lq2 = F.log_softmax(
        student(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), -1)
    w_all = (lq2.exp().detach() * (lq2 - lp).detach())
    g_enum = flat_grad((w_all * lq2).sum() / lq2.shape[0])

    # g_mc：黑盒 MC——只用采样 token 的 logprob
    torch.manual_seed(seed + 1)
    N = 1024
    g_mc = torch.zeros_like(g_exact)
    M = 8
    for _ in range(M):
        student.zero_grad()
        lq3 = F.log_softmax(
            student(full)[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), -1)
        ys = torch.multinomial(lq3.exp().detach(), N // M, replacement=True)
        logq_y = lq3.gather(-1, ys)
        c_y = (logq_y.detach() - lp.gather(-1, ys))   # 黑盒接口：只要采样 logprob
        g_mc = g_mc + flat_grad((c_y * logq_y).sum() / (lq3.shape[0] * (N // M))) / M

    def cos(a, b):
        return float((a @ b) / (a.norm() * b.norm() + 1e-12))

    return {"rel_enum": float((g_exact - g_enum).norm() / g_exact.norm()),
            "cos_mc": cos(g_exact, g_mc),
            "rel_mc": float((g_mc - g_exact).norm() / g_exact.norm())}


def probe_gradient_variance(student, teacher, pool, batch=32, n_draws=16,
                            seed=904):
    """白盒 vs 黑盒的梯度方差（固定学生，训练估计器的真实形态）：
    白盒每步梯度是解析量（零方差）；黑盒每步 = 单次采样 batch 的 REINFORCE
    估计。M 次独立抽取 → RMS(‖g_m − g_exact‖)/‖g_exact‖ = 相对方差。
    （DeepSeek-V4 报告自述 token 级 advantage 路线「high variance in gradient
    estimation」的 toy 尺度实例化。）"""
    torch.manual_seed(seed)
    student.eval(); teacher.eval()
    prompts = torch.tensor([make_prompt_ids(d) for d in pool])
    g = torch.Generator().manual_seed(seed)

    def flat_grad(loss):
        gs = torch.autograd.grad(loss, list(student.parameters()))
        return torch.cat([gg.detach().reshape(-1) for gg in gs])

    # g_exact（白盒）：任取一个 batch 的解析 token 级 KL 梯度
    idx = torch.randint(0, len(pool), (batch,), generator=g)
    px = prompts[idx]
    with torch.no_grad():
        resp = student.sample(px, RESP_LEN)
        t_logits = teacher(torch.cat([px, resp], dim=1))
    Lp = px.shape[1]
    student.zero_grad()
    lq = F.log_softmax(
        student(torch.cat([px, resp], dim=1))[:, Lp - 1: Lp - 1 + RESP_LEN, :]
        .reshape(-1, V), -1)
    lp = F.log_softmax(
        t_logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), -1)
    g_exact = flat_grad(kl_tok(lq, lp, "rev").mean())

    # 黑盒：M 次独立单 batch REINFORCE 抽取（同一 (px, resp)，采样变化）
    rms2 = 0.0
    for _ in range(n_draws):
        student.zero_grad()
        lq2 = F.log_softmax(
            student(torch.cat([px, resp], dim=1))[:, Lp - 1: Lp - 1 + RESP_LEN, :]
            .reshape(-1, V), -1)
        ys = torch.multinomial(lq2.exp().detach(), 1)          # 每位置 1 样本=训练形态
        logq_y = lq2.gather(-1, ys)
        c_y = logq_y.detach() - lp.gather(-1, ys)
        g_m = flat_grad((c_y * logq_y).sum() / lq2.shape[0])
        rms2 += float(((g_m - g_exact) ** 2).sum()) / (g_exact.norm() ** 2)
    return {"bb_rms_rel": math.sqrt(rms2 / n_draws)}


# ---------------- 主流程 ----------------

DEFAULT_CFG = dict(
    seed_pool=7, seed_teacher=101, seed_init=555, seed_run=777,
    n_pool=256, teacher_steps=1000,
    axis_steps=500, main_steps=600, batch=32, lr=2e-3,
    student=dict(d_model=16, nhead=2, layers=1, ff=32),
    # 轴1 专用：容量边缘学生（3,318 参）。mode-seeking/covering 对比只在
    # 容量压力下出现（L0 受限学生族是极端情形）；d_model=12 恰在「装不下
    # 两套密码本」的边缘，结局对初始采样噪声敏感（对称破缺，L0 §8），
    # 固定 seed 下给出规范对比，边界声明见 tutorial §4。
    axis1_student=dict(d_model=12, nhead=2, layers=1, ff=24),
    eval_m=8,
)


@torch.no_grad()
def probe_gate_profile(teacher, pool, n_prompts=64, seed=905):
    """熵门控的 λ 剖面（ada 配方的机制自检）：λ_t = H_t/ln V，H_t 是教师对
    学生 gold(A) 前缀的条件熵。双语教师在字母位近确定性（H≈0），只在
    位置 0（模选择位，双模边缘 ½/½）有显著熵 → 门控应集中在位置 0。
    返回 (位置0 λ 均值, 其余位 λ 均值)。"""
    torch.manual_seed(seed)
    teacher.eval()
    sub = pool[:n_prompts]
    px = torch.tensor([make_prompt_ids(d) for d in sub])
    ry = torch.tensor([target_response(d, "A") for d in sub])
    full = torch.cat([px, ry[:, :-1]], dim=1)
    Lp = px.shape[1]
    lp = F.log_softmax(teacher(full), -1)[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    h = (-(lp.exp() * lp).sum(-1) / math.log(V))          # [N, R]，已归一 ln V
    return float(h[:, 0].mean()), float(h[:, 1:].mean())


def main(cfg=None):
    global TEACHER
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    if cfg.get("student") is DEFAULT_CFG["student"]:
        cfg["student"] = dict(DEFAULT_CFG["student"])
    print("=" * 72)
    print("nano-opd L3 — 生产配方的四个旋钮（survey taxonomy 机制类别 ×4）")
    print("=" * 72)
    print(f"任务: {CONTENT_LEN} 位数字 → 小写 codebook A 或大写 codebook B 都算对")
    print(f"词表 V={V} | 响应长 {RESP_LEN} | 学生自采样 on-policy 循环（全轴共用）")

    pool = make_pool(cfg["seed_pool"], cfg["n_pool"])
    TEACHER = train_teacher(pool, steps=cfg["teacher_steps"],
                            seed=cfg["seed_teacher"])
    n_t = sum(p.numel() for p in TEACHER.parameters())
    n_s = sum(p.numel() for p in fresh_student(cfg["student"], 0).parameters())
    print(f"\n[1] 双语教师（{cfg['teacher_steps']} 步）params = {n_t:,}"
          f"（教师/学生 ≈ {n_t / n_s:.0f}x，学生 {n_s:,}）")
    t_audit = evaluate_bilingual(TEACHER, pool, m_per_prompt=cfg["eval_m"])
    print(f"    教师自采样: valid_A={t_audit['valid_A']:.3f} "
          f"valid_B={t_audit['valid_B']:.3f} frac_A={t_audit['mode_frac_A']:.3f}"
          f"（教师逐样本随机选模，分域精确匹配 ≈ 0.5×准确率 是构造使然）")
    assert t_audit["valid_total"] > 0.9, \
        "双语教师必须两套密码本都会（total = 两域精确匹配之和）"
    assert abs(t_audit["mode_frac_A"] - 0.5) < 0.25, "教师两模应均衡"

    # ---- [2] 轴1：divergence 设计 ----
    print(f"\n[2] 轴1 divergence 设计（同一 on-policy 循环，各 {cfg['axis_steps']} 步）")
    probe = probe_jsd_math(fresh_student(cfg["student"], cfg["seed_init"]),
                           TEACHER, pool)
    print("    β-JSD 族探针（初始化学生、teacher-forced，无需训练）")
    print(f"      对称性 β=0.5: max|JSD(p‖q)−JSD(q‖p)| = {probe['sym_max_abs']:.2e}"
          "（float32 机器精度）")
    print(f"      端点标度（β=1e-5，均值相对差；max 括号内）: JSD_β/β vs FKL = "
          f"{probe['scale_fwd_rel']:.2e} ({probe['scale_fwd_max']:.2e})；"
          f"JSD_β/(1−β) vs RKL = {probe['scale_rev_rel']:.2e} "
          f"({probe['scale_rev_max']:.2e})")
    print(f"      有界性: 对称 JSD ∈ [{probe['jsd_min']:.3f}, {probe['jsd_max']:.3f}]"
          f"（ln 2 = {probe['ln2']:.3f}）")
    assert probe["sym_max_abs"] < 1e-5, "β=0.5 时 JSD 应对称"
    assert probe["scale_fwd_rel"] < 1e-2 and probe["scale_rev_rel"] < 1e-2, \
        "端点标度：JSD_β 的一阶项应是 fwd/rev KL"
    assert probe["jsd_min"] >= -1e-6 and probe["jsd_max"] <= probe["ln2"] + 1e-6, \
        "对称 JSD 有界 ∈ [0, ln 2]"

    axis1 = {}
    n_a1 = sum(p.numel() for p in fresh_student(cfg["axis1_student"], 0).parameters())
    print(f"    容量边缘学生: {n_a1:,} 参（mode-seeking/covering 对比只在容量压力下出现）")
    lam0, lamr = probe_gate_profile(TEACHER, pool)
    print(f"    熵门控 λ 剖面（教师条件熵/ln V）: 位置0 = {lam0:.3f}，"
          f"其余位均值 = {lamr:.3f}（门控集中在模选择位）")
    recipes = [("fwd", dict(axis="fwd")), ("mix25", dict(axis="mix", beta=0.25)),
               ("jsd", dict(axis="jsd", beta=0.5)),
               ("mix75", dict(axis="mix", beta=0.75)), ("rev", dict(axis="rev")),
               ("ada", dict(axis="ada"))]
    print(f"    {'recipe':7s} {'valid_A':>8s} {'valid_B':>8s} {'total':>7s} "
          f"{'frac_A':>7s} {'lock':>5s} {'背书':>7s} {'maxTokLoss':>10s}")
    for name, kw in recipes:
        student = fresh_student(cfg["axis1_student"], cfg["seed_init"])
        _, _, _, _, mtl, _ = train_opd(student, TEACHER, pool, cfg["axis_steps"],
                                       cfg["batch"], cfg["lr"], cfg["seed_run"],
                                       **kw)
        m = evaluate_bilingual(student, pool, m_per_prompt=cfg["eval_m"])
        m["max_tok_loss"] = mtl
        axis1[name] = m
        print(f"    {name:7s} {m['valid_A']:8.3f} {m['valid_B']:8.3f} "
              f"{m['valid_total']:7.3f} {m['mode_frac_A']:7.3f} {m['lock']:5.2f} "
              f"{m['endorse']:+7.3f} {mtl:10.3f}")
    axis1["gate"] = {"lam0": lam0, "lam_rest": lamr}

    # ---- [3] 轴2：信号源（白盒/黑盒） ----
    print(f"\n[3] 轴2 信号源（白盒 / 黑盒裸 REINFORCE / 黑盒+group 采样，"
          f"各 {cfg['main_steps']} 步）")
    gi = probe_gradient_identity(fresh_student(cfg["student"], cfg["seed_init"]),
                                 TEACHER, pool)
    gv = probe_gradient_variance(fresh_student(cfg["student"], cfg["seed_init"]),
                                 TEACHER, pool)
    print("    恒等式探针 ∇KL(q‖p) = E_{y~q}[(log q − log p)·∇log q]（固定学生）")
    print(f"      白盒解析 vs 全词表枚举: 相对差 = {gi['rel_enum']:.2e}（机器精度）")
    print(f"      黑盒 MC(N=1024) vs 白盒: 余弦 = {gi['cos_mc']:.4f}，"
          f"相对范数差 = {gi['rel_mc']:.3f}")
    print(f"    梯度方差探针（M=16 独立抽取，每位置 1 样本 = 训练形态）")
    print(f"      黑盒 RMS 相对误差 = {gv['bb_rms_rel']:.3f}（白盒 = 0，解析量）")
    assert gi["rel_enum"] < 1e-4, "恒等式应在机器精度内成立"
    assert gi["cos_mc"] > 0.95, "黑盒 MC 估计应与白盒解析梯度同向（无偏性）"
    assert gv["bb_rms_rel"] > 0.05, "黑盒单样本估计应有显著方差（白盒为零）"

    axis2, axis2_students = {}, {}
    routes = [("whitebox", dict(blackbox=False, log_kl=True)),
              ("blackbox-naive", dict(blackbox=True, log_kl=True)),
              ("blackbox-group16",
               dict(blackbox=True, log_kl=True, group_m=16))]
    for name, kw in routes:
        student = fresh_student(cfg["student"], cfg["seed_init"])
        _, kl_log, _, _, mtl, _ = train_opd(
            student, TEACHER, pool, cfg["main_steps"], cfg["batch"], cfg["lr"],
            cfg["seed_run"], **kw)
        m = evaluate_bilingual(student, pool, m_per_prompt=cfg["eval_m"])
        m["max_tok_loss"] = mtl
        tail = kl_log[100:]
        m["kl_std"] = float(torch.tensor(tail).std()) if len(tail) > 2 else 0.0
        m["kl_mean"] = float(sum(tail) / len(tail))
        axis2[name] = m
        axis2_students[name] = student
    print(f"    {'route':16s} {'valid_A':>8s} {'valid_B':>8s} {'total':>7s} "
          f"{'lock':>5s} {'背书':>7s} {'KL均值':>8s} {'KL标准差':>9s}")
    for name in ("whitebox", "blackbox-naive", "blackbox-group16"):
        m = axis2[name]
        print(f"    {name:16s} {m['valid_A']:8.3f} {m['valid_B']:8.3f} "
              f"{m['valid_total']:7.3f} {m['lock']:5.2f} {m['endorse']:+7.3f} "
              f"{m['kl_mean']:8.3f} {m['kl_std']:9.4f}")
    torch.manual_seed(906)
    demo_px = torch.tensor([make_prompt_ids(d) for d in pool[:2]])
    for name in ("whitebox", "blackbox-naive", "blackbox-group16"):
        reps = axis2_students[name].sample(demo_px, RESP_LEN)
        print(f"    {name} 采样: " + "  ".join(decode(r) for r in reps))
    assert axis2["whitebox"]["valid_total"] > 0.4, "白盒路线应学会（mode-lock 至少一模）"
    # 裸 per-token REINFORCE 确定性崩溃（实测派生断言，固定 seed）：
    # 方差爆炸（bb_rms_rel≫1）+ Adam 噪梯度 + OOD 前缀自强化 → KL 平台不降。
    assert axis2["blackbox-naive"]["valid_total"] < 0.1, \
        "裸黑盒 REINFORCE 在本 toy 确定性崩溃（方差代价的极端形态，非调参问题）"
    assert axis2["blackbox-naive"]["kl_mean"] > 4.0, \
        "崩溃态 KL 平台：600 步无下降（教师信号在 OOD 前缀上失去引导结构）"
    assert axis2["blackbox-naive"]["endorse"] < -5.0, "崩溃态教师完全不背书"
    # group 采样修复（GRPO group-sampling 机制类别）：每位置样本 ×16 → 方差 1/16。
    assert axis2["blackbox-group16"]["valid_total"] > 0.4, \
        "黑盒 + group 采样应学会（恒等式保证无偏，group 平均压低方差）"
    assert axis2["blackbox-group16"]["kl_mean"] < \
        axis2["blackbox-naive"]["kl_mean"], "group 修复应带来真实 KL 下降"
    # 轨迹 KL 波动不是估计器方差的探针（错误代理反例）：
    # 崩溃路线「原地不动」→ 轨迹最平稳；DSv4「high variance」的直接探针是
    # 上面的 bb_rms_rel（梯度方差），不是轨迹 std。
    assert axis2["blackbox-naive"]["kl_std"] < \
        axis2["blackbox-group16"]["kl_std"], \
        "轨迹最稳的恰是崩溃路线——kl_std 度量「动没动」，不度量「噪不噪」"

    # ---- [4] 轴3：token 加权 ----
    print(f"\n[4] 轴3 token 加权（rev-KL 循环；300 步快照 + {cfg['main_steps']} 步终态）")
    axis3 = {}
    for name in ("uniform", "gap", "norm", "norm_div"):
        student = fresh_student(cfg["student"], cfg["seed_init"])
        _, _, ckpts, _, mtl, _ = train_opd(
            student, TEACHER, pool, cfg["main_steps"], cfg["batch"], cfg["lr"],
            cfg["seed_run"], axis="rev", weight=name, ckpt_at={300})
        m = evaluate_bilingual(student, pool, m_per_prompt=cfg["eval_m"])
        m["ckpt"] = ckpts[300]
        axis3[name] = m
    print(f"    {'weight':8s} {'300步字母位':>10s} {'300步<end>':>10s} "
          f"{'300步位loss离散':>13s} {'终态total':>9s} {'终态lock':>8s}")
    for name in ("uniform", "gap", "norm", "norm_div"):
        m = axis3[name]
        pl = m["ckpt"]["pos_loss"]
        spread = float(torch.tensor(pl[:RESP_LEN - 1]).std())
        m["spread300"] = spread
        print(f"    {name:8s} {m['ckpt']['letters']:10.3f} {m['ckpt']['end']:10.3f} "
              f"{spread:13.4f} {m['valid_total']:9.3f} {m['lock']:8.2f}")
    assert axis3["gap"]["ckpt"]["letters"] >= axis3["uniform"]["ckpt"]["letters"] - 0.02, \
        "gap 加权应不慢于均匀（预算集中在高 gap 位）"
    # 实测派生断言（固定 seed）——加权谱序：
    # gap 把尺度差当信号（集中预算追平落后位 → 离散最小、收敛最快）；
    # norm（EMA 尺度相对加权，有界）把预算摊向已低 loss 位 → 慢于均匀、离散更大；
    # norm_div（ℓ/EMA 除法）有效步长 ∝ 1/EMA 无界 → 发散崩溃。
    assert axis3["norm"]["ckpt"]["letters"] < axis3["uniform"]["ckpt"]["letters"], \
        "EMA 尺度相对加权应收敛更慢（本 toy 尺度差携带信息，见 tutorial §5）"
    assert axis3["gap"]["spread300"] < axis3["uniform"]["spread300"] \
        < axis3["norm"]["spread300"], \
        "等化谱序：gap < uniform < norm（实测派生）"
    assert axis3["norm_div"]["valid_total"] < 0.1, \
        "norm_div 无界有效步长应崩溃（有界权重前提被违反，见 tutorial §5）"
    assert axis3["norm_div"]["spread300"] > axis3["uniform"]["spread300"], \
        "norm_div 不缩小反而扩大逐位离散（实测派生）"
    assert abs(axis3["gap"]["valid_total"] - axis3["uniform"]["valid_total"]) <= 0.12, \
        "加权不改变不动点：终态应与均匀同量级"
    assert abs(axis3["norm"]["valid_total"] - axis3["uniform"]["valid_total"]) <= 0.12, \
        "加权不改变不动点：终态应与均匀同量级（有界权重）"

    # ---- [5] 轴4：效率（rollout 复用）与稳定化（IS / advantage 裁剪） ----
    print(f"\n[5] 轴4 效率与稳定化（rev-KL 白盒，总梯度步 {cfg['main_steps']}）")
    axis4 = {}
    for k in (1, 2, 4, 8):
        student = fresh_student(cfg["student"], cfg["seed_init"])
        _, _, _, drift, mtl, _ = train_opd(
            student, TEACHER, pool, cfg["main_steps"], cfg["batch"], cfg["lr"],
            cfg["seed_run"], axis="rev", reuse_k=k)
        m = evaluate_bilingual(student, pool, m_per_prompt=cfg["eval_m"])
        m["drift"] = float(sum(drift) / len(drift)) if drift else 0.0
        axis4[f"K{k}"] = m
    student = fresh_student(cfg["student"], cfg["seed_init"])
    _, _, _, drift_is, _, _ = train_opd(
        student, TEACHER, pool, cfg["main_steps"], cfg["batch"], cfg["lr"],
        cfg["seed_run"], axis="rev", reuse_k=4, use_is=True)
    m = evaluate_bilingual(student, pool, m_per_prompt=cfg["eval_m"])
    m["drift"] = float(sum(drift_is) / len(drift_is)) if drift_is else 0.0
    axis4["K4+IS"] = m
    print(f"    {'config':6s} {'valid_A':>8s} {'valid_B':>8s} {'total':>7s} "
          f"{'lock':>5s} {'陈旧度':>8s}")
    for name in ("K1", "K2", "K4", "K8", "K4+IS"):
        m = axis4[name]
        print(f"    {name:6s} {m['valid_A']:8.3f} {m['valid_B']:8.3f} "
              f"{m['valid_total']:7.3f} {m['lock']:5.2f} {m['drift']:8.4f}")
    assert axis4["K8"]["drift"] > axis4["K2"]["drift"] > 0.0, \
        "复用次数越高，样本越陈旧（off-policy 程度单调）"
    assert abs(axis4["K1"]["valid_total"] - axis4["K8"]["valid_total"]) <= 0.05, \
        "默认学生容量冗余吸收陈旧度：损害低于可测阈（实测派生，见下探针）"
    assert axis4["K4+IS"]["valid_total"] >= axis4["K4"]["valid_total"] - 0.01, \
        "clipped IS 修正应至少不劣于裸复用"
    # 陈旧度损害可测性探针：换容量边缘学生（轴1 同款），损害显形。
    # 加跑 K4/K4+IS：IS 的偏差-方差交换在容量压力下实测为负收益。
    print(f"    陈旧度损害探针（容量边缘学生 {n_a1:,} 参，K1/K4/K4+IS/K8）")
    marg = {}
    for k in (1, 4, 8):
        student = fresh_student(cfg["axis1_student"], cfg["seed_init"])
        _, _, _, drift, mtl, _ = train_opd(
            student, TEACHER, pool, cfg["main_steps"], cfg["batch"], cfg["lr"],
            cfg["seed_run"], axis="rev", reuse_k=k)
        m = evaluate_bilingual(student, pool, m_per_prompt=cfg["eval_m"])
        m["drift"] = float(sum(drift) / len(drift)) if drift else 0.0
        marg[f"K{k}"] = m
    student = fresh_student(cfg["axis1_student"], cfg["seed_init"])
    _, _, _, drift, mtl, _ = train_opd(
        student, TEACHER, pool, cfg["main_steps"], cfg["batch"], cfg["lr"],
        cfg["seed_run"], axis="rev", reuse_k=4, use_is=True)
    m = evaluate_bilingual(student, pool, m_per_prompt=cfg["eval_m"])
    m["drift"] = float(sum(drift) / len(drift)) if drift else 0.0
    marg["K4+IS"] = m
    for name in ("K1", "K4", "K4+IS", "K8"):
        m = marg[name]
        print(f"      {name:5s}: total={m['valid_total']:.3f} "
              f"lock={m['lock']:.2f} 陈旧度={m['drift']:.4f}")
    assert marg["K1"]["valid_total"] > marg["K8"]["valid_total"] + 0.05, \
        "容量压力下陈旧度损害可测（L0 off-policy 偏差定理的序列空间版）"
    assert marg["K8"]["drift"] > marg["K1"]["drift"], "边缘学生陈旧度同样单调"
    assert marg["K4+IS"]["valid_total"] < marg["K4"]["valid_total"] - 0.02, \
        "容量边缘学生上 IS 的方差成本超过偏差收益（实测派生，见 tutorial §6.2）"

    print(f"    稳定化：黑盒 group16 路线 advantage 裁剪 τ 谱系"
          f"（各 {cfg['main_steps']} 步）")
    clip_res = {}
    for name, tau in (("noclip", None), ("clip2", 2.0), ("clip5", 5.0)):
        student = fresh_student(cfg["student"], cfg["seed_init"])
        losses, kl_log, _, _, mtl, cfrac = train_opd(
            student, TEACHER, pool, cfg["main_steps"], cfg["batch"], cfg["lr"],
            cfg["seed_run"], blackbox=True, group_m=16, clip_adv=tau,
            log_kl=True)
        m = evaluate_bilingual(student, pool, m_per_prompt=cfg["eval_m"])
        tail = losses[100:]
        m["loss_std"] = float(torch.tensor(tail).std())
        m["kl_std"] = float(torch.tensor(kl_log[100:]).std())
        m["clip_frac"] = cfrac
        clip_res[name] = m
    print(f"      {'config':7s} {'终态total':>9s} {'loss标准差':>10s} "
          f"{'被裁token占比':>12s}")
    for name in ("noclip", "clip2", "clip5"):
        m = clip_res[name]
        print(f"      {name:7s} {m['valid_total']:9.3f} {m['loss_std']:10.3f} "
              f"{m['clip_frac']:12.4f}")
    assert clip_res["clip5"]["loss_std"] < clip_res["noclip"]["loss_std"], \
        "τ=5 裁剪应压低极端 advantage 引起的 loss 波动（Kimi K3 τ-clip 机制类别）"
    assert abs(clip_res["clip5"]["valid_total"]
               - clip_res["noclip"]["valid_total"]) <= 0.10, \
        "τ=5 只剪尾部：终态等价（只压方差，不改目标方向）"
    assert clip_res["clip2"]["valid_total"] < \
        clip_res["noclip"]["valid_total"] - 0.2, \
        "τ=2 阈值失配：剪掉 off-manifold 下压信号主体，终态崩坏（实测派生）"

    # ---- [6] self-check ----
    print("\n[6] self-check（断言设计见 tutorial §7）")
    fwd, mix25, jsd, mix75, rev, ada = (axis1[k] for k in
                                        ("fwd", "mix25", "jsd", "mix75", "rev", "ada"))
    # (a) 散度端点的动力学结局：reverse 锁模 / forward 覆盖（L0 算术定理的序列空间版）
    assert rev["lock"] > 0.85 and max(rev["valid_A"], rev["valid_B"]) > 0.5, \
        f"reverse KL 应锁一模: lock={rev['lock']:.2f}"
    assert fwd["lock"] < 0.5 and min(fwd["valid_A"], fwd["valid_B"]) > 0.02, \
        f"forward KL 应两模都留质量: lock={fwd['lock']:.2f}"
    # (b) 插值轴连续过渡：lock 从 fwd 到 rev 单调抬升（端点之间无相变）
    assert mix25["lock"] < rev["lock"] and mix75["lock"] > fwd["lock"], \
        "凸插值应介于两个端点行为之间"
    # (c) JSD 有界 vs reverse 无界：同一循环下逐 token loss 上界差异
    assert jsd["max_tok_loss"] <= probe["ln2"] + 1e-6, "JSD token loss 应 ≤ ln 2"
    assert rev["max_tok_loss"] > 2 * probe["ln2"], \
        "reverse KL token loss 无界（off-manifold 位可远超 ln 2）"
    # (d) 熵门控软化锁模（高熵位引入 forward 分量 → 不彻底选边）
    assert ada["lock"] < rev["lock"], "熵门控应软化 mode-seeking"
    print("✅ self-check passed: JSD族数学/恒等式/白盒黑盒(group修复+崩溃反例)/"
          "加权谱序与不动点/复用陈旧度(边缘学生损害)/IS/τ裁剪谱系/散度端点结局")

    # ---- digest ----
    key = "|".join(f"{k}:{axis1[k]['valid_A']:.3f},{axis1[k]['valid_B']:.3f},"
                   f"{axis1[k]['lock']:.2f}" for k in
                   ("fwd", "mix25", "jsd", "mix75", "rev", "ada"))
    key += f"|a2:{axis2['whitebox']['valid_total']:.3f}," \
           f"{axis2['blackbox-naive']['valid_total']:.3f}," \
           f"{axis2['blackbox-group16']['valid_total']:.3f}," \
           f"{axis2['blackbox-group16']['kl_std']:.4f}"
    key += f"|a3:{axis3['uniform']['valid_total']:.3f}," \
           f"{axis3['gap']['valid_total']:.3f},{axis3['norm']['valid_total']:.3f}," \
           f"{axis3['norm_div']['valid_total']:.3f}"
    key += f"|a4:{axis4['K1']['valid_total']:.3f},{axis4['K8']['valid_total']:.3f}," \
           f"{axis4['K4+IS']['valid_total']:.3f},{axis4['K8']['drift']:.4f}," \
           f"{marg['K1']['valid_total']:.3f},{marg['K8']['valid_total']:.3f}," \
           f"{marg['K4+IS']['valid_total']:.3f}"
    key += f"|clip:{clip_res['clip5']['loss_std']:.3f}," \
           f"{clip_res['clip2']['valid_total']:.3f}" \
           f"|gi:{gi['rel_enum']:.1e},{gi['cos_mc']:.3f}"
    digest = hashlib.md5(key.encode()).hexdigest()[:16]
    print(f"\ndigest(关键指标) = {digest}")
    print("\ntakeaway: 同一个 on-policy 循环，四个旋钮各有一个生产答案：")
    print("          散度：生产主流拧在 reverse 端（Qwen3/DSv4/TM），JSD 有界但中庸；")
    print("          信号源：reverse KL 是唯一「黑盒可跑」的散度（恒等式，机器精度），")
    print("                  但裸 per-token REINFORCE 在本 toy 确定性崩溃（方差爆炸 +")
    print("                  OOD 自强化）——可行性 = 恒等式 + 方差代价；group 采样")
    print("                  （×16 教师打分）修复之，代价 = 同步数预算终态仍逊于白盒；")
    print("          加权：不改不动点（有界权重），只改路径——gap 把尺度差当信号最快，")
    print("                EMA 归一化族把尺度差当噪声、在本 toy 实测有害（无界形态崩溃）；")
    print("          效率：rollout 复用省采样但样本变陈旧（off-policy 偏差定理），")
    print("                容量冗余吸收损害、容量边缘学生使损害显形；IS 修偏差付方差；")
    print("                advantage 裁剪 τ 须与尺度匹配（τ=5 压波动不损终态，τ=2 崩坏）。")
    return axis1, axis2, axis3, axis4, clip_res


if __name__ == "__main__":
    main()
