#!/usr/bin/env python3
"""nano-opd L2 — multi-teacher OPD：融合位置与路由的最小机制

L0/L1 证明了单教师 OPD 的核心算术：reverse KL 的期望在学生分布下 → 学生必须
自采样（on-policy），教师只打分。L2 引入第二个教师，问生产配方真正遇到的问题：

  两个教师的知识怎么合进一个学生？

机制类别（对应 OPD survey [arXiv:2604.00626] taxonomy 的多教师分支；生产数据点：
MOPD [arXiv:2606.30406] / MiMo-V2-Flash [arXiv:2601.02780] 的 per-domain teachers
+ 学生自采样蒸馏。只教机制类别，不追未经核验的单源变体方法名）：

  一、融合位置（fuse-then-distill 的三个变体 + 一个恒等式）：
    mix_prob  概率空间算术混合  p_mix = ½(p_A + p_B)，再对 p_mix 做 OPD
    mix_poe   logit 空间几何混合 p_mix ∝ √(p_A·p_B)（product-of-experts），再 OPD
    loss_mix  loss 级加权和     L = ½KL(q||p_A) + ½KL(q||p_B)
    恒等式：loss_mix ≡ 对「未归一化几何混合」的 reverse KL（逐 token 机器精度可验）。
    「在哪写加号」不决定目标函数——数学决定。
    Jensen 间隙：loss_mix − KL(q||p_mix_prob) = E_q[log Σw p − Σw log p] ≥ 0，
    恰是「教师对学生自己样本的分歧度」——可逐位置测量。

  二、路由（distill-then-route）：
    route      按 prompt 的领域标记选教师（context routing）——生产形态：
               per-domain teachers，每条学生自采样样本由其领域教师打分
    route_self 按学生自己的样本选教师（output routing）——反例：路由信息来自
               学生输出时会自增强学生的既有偏向

任务（L1 双语言密码本的领域化扩展）：
  4 位数字 → codebook A（小写）或 codebook B（大写），prompt 携带领域标记：
    <in> 3 1 4 1 <a> <out>  → 只有 codebook A 正确（领域 A）
    <in> 3 1 4 1 <b> <out>  → 只有 codebook B 正确（领域 B）
  两个「立场教师」：教师 A 在任何上下文都坚持 codebook A（领域 A 内正确、
  领域 B 内自信地错），教师 B 镜像。刻意设置：近确定性教师（自己领域 gold
  ≈ −0.001 nats/token）把「教师冲突」推到极端，让融合失败模式的机制可见。
  真实领域教师的域外行为介于随机与自信地错之间，机制相同、程度较轻。

本节实测的三个关键现象（断言全部理论锚定，见 [5]）：
  1. loss_mix 与 mix_poe 是同一目标函数（恒等式）：token 级 ½/½ 独立混合，
     序列模式被销毁 → 精确匹配塌缩到 ≈ (½)^8 量级。
  2. mix_prob 的目标保留两个序列模式，但 on-policy 采样找不到它们：
     算术混合在首字母位给每个教师候选恰 ½ 票（−ln 2，前缀还是 prompt、
     两教师都在流形上），杂交前缀一旦开始，两教师同时 off-manifold、票衰减
     → 梯度无方向偏好，学生停在「token 级边缘匹配混合边缘」的杂交平衡。
  3. 唯一改变结局的是路由：route 两域全学会；route_self（按学生输出路由）
     自增强锁一模。路由信号必须来自上下文，不能来自学生自己。

显式声明（反幻觉）：
  1. 本节无 mock：教师/学生的前向/反向/采样全部是真实 torch 计算。
  2. 「立场教师」是刻意的机制放大设置（见上），不是对生产教师的写实。
  3. 全部数字为现场实跑（CPU，固定 seed），不是任何真实模型的 benchmark。
  4. 恒等式、Jensen 间隙、½ 票是数学事实（任意分布成立），实验只是把它跑出来。

依赖仅 torch（CPU 即跑）。固定 seed，stdout 不含墙钟与本机路径，可逐字节复验。
"""
import hashlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 词表与任务（L1 扩展：+ 领域标记 <a>/<b>） ----------------

SPECIALS = ["<pad>", "<in>", "<out>", "<end>", "<a>", "<b>"]
DIGITS = [str(i) for i in range(10)]
LOWER = [chr(ord("a") + i) for i in range(26)]
UPPER = [chr(ord("A") + i) for i in range(26)]
VOCAB = SPECIALS + DIGITS + LOWER + UPPER
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for i, t in enumerate(VOCAB)}
V = len(VOCAB)
PAD, IN, OUT, END = (TOK2ID[t] for t in ("<pad>", "<in>", "<out>", "<end>"))
MARK_A, MARK_B = TOK2ID["<a>"], TOK2ID["<b>"]
LOWER_IDS = torch.tensor([TOK2ID[c] for c in LOWER])
UPPER_IDS = torch.tensor([TOK2ID[c] for c in UPPER])

CONTENT_LEN = 4
RESP_LEN = 2 * CONTENT_LEN + 1          # 8 字母 + <end> = 9
MARK_POS = 1 + CONTENT_LEN              # prompt: [IN, d0..d3, MARK, OUT]

# 与 L1 相同的固定置换（跨机器逐字节复现）
_PERM_A = [7, 2, 19, 11, 24, 0, 15, 9, 21, 4, 17, 12, 25, 3, 14, 8, 22, 1, 18, 10,
           23, 5, 16, 6, 20, 13]
_PERM_B = [13, 20, 6, 16, 5, 23, 10, 18, 1, 22, 8, 14, 3, 25, 12, 17, 4, 21, 9, 15,
           0, 24, 11, 19, 2, 7]
CODEBOOK_A = {d: (LOWER[_PERM_A[2 * d]], LOWER[_PERM_A[2 * d + 1]]) for d in range(10)}
CODEBOOK_B = {d: (UPPER[_PERM_B[2 * d]], UPPER[_PERM_B[2 * d + 1]]) for d in range(10)}


def decode(ids):
    return "".join(ID2TOK[int(i)] for i in ids)


def make_prompt_ids(digits, domain):
    return [IN] + list(digits) + [MARK_A if domain == "A" else MARK_B] + [OUT]


def target_response(digits, domain):
    cb = CODEBOOK_A if domain == "A" else CODEBOOK_B
    toks = [t for d in digits for t in cb[d]] + ["<end>"]
    return [TOK2ID[t] for t in toks]


def hybrid_response(digits):
    """大小写交替杂交：偶数内容位用 codebook A 字母、奇数位用 codebook B。
    构造性杂交样本——两个教师各认一半字母位。"""
    a = [t for d in digits for t in CODEBOOK_A[d]]
    b = [t for d in digits for t in CODEBOOK_B[d]]
    toks = [a[j] if j % 2 == 0 else b[j] for j in range(2 * CONTENT_LEN)] + ["<end>"]
    return [TOK2ID[t] for t in toks]


def make_pool(seed, n_per_domain):
    """混合领域 prompt 池：n_per_domain 个领域 A + n_per_domain 个领域 B。"""
    g = torch.Generator().manual_seed(seed)
    pool = []
    for domain in ("A", "B"):
        for _ in range(n_per_domain):
            d = torch.randint(0, 10, (CONTENT_LEN,), generator=g).tolist()
            pool.append((d, domain))
    return pool


# ---------------- 模型（与 L1 同款 TinyLM） ----------------

class TinyLM(nn.Module):
    def __init__(self, d_model, nhead, layers, dim_ff, max_len=32):
        super().__init__()
        self.tok = nn.Embedding(V, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=0.0, batch_first=True
        )
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


# ---------------- 教师：两个「立场教师」 ----------------

def train_fundamentalist(pool, own_domain, steps=500, batch=32, lr=2e-3,
                         seed=201, d_model=64, nhead=4, layers=2, ff=256):
    """立场教师：在全部上下文（两个领域的 prompt）都坚持自己的 codebook。
    own_domain='A' → 所有 target 用 codebook A（领域 A 内正确、B 内自信地错）。"""
    torch.manual_seed(seed)
    model = TinyLM(d_model, nhead, layers, ff)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        idx = torch.randint(0, len(pool), (batch,), generator=g).tolist()
        x = torch.tensor([make_prompt_ids(pool[i][0], pool[i][1]) for i in idx])
        y = torch.tensor([target_response(pool[i][0], own_domain) for i in idx])
        logits = model(torch.cat([x, y[:, :-1]], dim=1))
        Lp = x.shape[1]
        loss = F.cross_entropy(
            logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), y.reshape(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def endorsement_table(teachers, pool, n_prompts=64):
    """2×2 背书表：每个教师对每个领域的 gold 序列逐 token 打分。
    对角线高、非对角线低 = 教师冲突可测。"""
    sub_a = [p for p in pool if p[1] == "A"][:n_prompts]
    sub_b = [p for p in pool if p[1] == "B"][:n_prompts]
    table = {}
    for tname, t in teachers.items():
        t.eval()
        row = {}
        for dname, sub in (("A", sub_a), ("B", sub_b)):
            px = torch.tensor([make_prompt_ids(d, dname) for d, _ in sub])
            ry = torch.tensor([target_response(d, dname) for d, _ in sub])
            row[dname] = (seq_logprob(t, px, ry) / RESP_LEN).mean().item()
        table[tname] = row
    return table


# ---------------- 探针：恒等式 + Jensen 间隙 + ½ 票（无需训练） ----------------

def rev_kl(lq, lp):
    """token 级 reverse KL：Σ q·(log q − log p)，q = softmax(lq)。
    lp 允许未归一化（此时量是 Σ q(log q − log p̃)，恒等式用）。"""
    return (lq.exp() * (lq - lp)).sum(-1)


@torch.no_grad()
def probe_fusion_math(student, tA, tB, pool, n_prompts=64, seed=901):
    """在真实分布上验证三条数学事实（初始化学生、teacher-forced 序列）：
    (1) 恒等式  ½KL(q||p_A)+½KL(q||p_B) == KL(q|| p_A^½·p_B^½ [未归一化])
    (2) Jensen  gap = loss_mix − KL(q||½(p_A+p_B)) = E_q[log Σw p − Σw log p] ≥ 0
        按位置分类聚合：字母位（教师冲突）vs <end> 位（教师一致）。
    (3) ½ 票   构造性杂交样本（大小写交替）在算术混合下：位置 0（前缀还是
        prompt，两教师都在流形上）每个教师候选恰 ½ 票 → logprob ≈ −ln 2；
        deeper 位置前缀已是杂交，两教师同时 off-manifold → 票显著衰减。"""
    torch.manual_seed(seed)
    student.eval(); tA.eval(); tB.eval()
    sub = [p for p in pool if p[1] == "A"][:n_prompts // 2] + \
          [p for p in pool if p[1] == "B"][:n_prompts // 2]
    px = torch.tensor([make_prompt_ids(d, dn) for d, dn in sub])
    ry = torch.tensor([target_response(d, dn) for d, dn in sub])
    full = torch.cat([px, ry[:, :-1]], dim=1)
    Lp = px.shape[1]
    lq = F.log_softmax(student(full), dim=-1)[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    lA = F.log_softmax(tA(full), dim=-1)[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    lB = F.log_softmax(tB(full), dim=-1)[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    lq, lA, lB = (t.reshape(-1, V) for t in (lq, lA, lB))

    loss_mix = 0.5 * rev_kl(lq, lA) + 0.5 * rev_kl(lq, lB)
    # 未归一化几何混合的 log：½log p_A + ½log p_B
    lp_geo_unnorm = 0.5 * lA + 0.5 * lB
    kl_geo_unnorm = rev_kl(lq, lp_geo_unnorm)
    ident_abs = (loss_mix - kl_geo_unnorm).abs()

    # 算术混合的 log：logsumexp(log ½ + lA, log ½ + lB)
    log_half = -torch.log(torch.tensor(2.0))
    lp_arith = torch.logsumexp(torch.stack([lA + log_half, lB + log_half]), dim=0)
    kl_arith = rev_kl(lq, lp_arith)
    gap = loss_mix - kl_arith                    # Jensen 间隙，逐 token

    # 位置分类：响应 8 个字母位（教师冲突）vs 末位 <end>（教师一致）
    gap = gap.reshape(-1, RESP_LEN)
    gap_letters = gap[:, :RESP_LEN - 1]
    gap_end = gap[:, RESP_LEN - 1]

    # ½ 票：构造性杂交样本在算术混合下的字母位 logprob。
    # 位置 0 的前缀还是 prompt（两个教师都在流形上）→ 各投 ½ 票；
    #  deeper 位置的前缀已是杂交（两个教师同时 off-manifold）→ 票衰减。
    hy = torch.tensor([hybrid_response(d) for d, _ in sub])
    full_h = torch.cat([px, hy[:, :-1]], dim=1)
    lhA = F.log_softmax(tA(full_h), dim=-1)[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    lhB = F.log_softmax(tB(full_h), dim=-1)[:, Lp - 1: Lp - 1 + RESP_LEN, :]
    lp_h = torch.logsumexp(torch.stack([lhA + log_half, lhB + log_half]), dim=0)
    hybrid_letter_lp = lp_h[:, :RESP_LEN - 1, :].gather(
        -1, hy[:, :RESP_LEN - 1].unsqueeze(-1)).squeeze(-1)
    return {
        "ident_max_abs": ident_abs.max().item(),
        "gap_min": gap.min().item(),
        "gap_letters_mean": gap_letters.mean().item(),
        "gap_end_mean": gap_end.mean().item(),
        "hybrid_pos0_lp": hybrid_letter_lp[:, 0].mean().item(),
        "hybrid_deep_lp": hybrid_letter_lp[:, 1:].mean().item(),
    }


# ---------------- 五个配方：同一 on-policy 循环，只换教师信号组合 ----------------

def fresh_student(cfg, seed):
    torch.manual_seed(seed)
    return TinyLM(cfg["d_model"], cfg["nhead"], cfg["layers"], cfg["ff"])


def train_multi_opd(student, tA, tB, pool, steps, batch, lr, seed, recipe):
    """OPD 循环（学生自采样 → 教师信号 → token 级 reverse KL 反传）。
    采样前缀视为常数（与 L1/GKD 一致）。recipe 决定教师信号怎么组合：
      mix_prob   目标 = ½(p_A+p_B)（概率空间算术混合）
      mix_poe    目标 = √(p_A·p_B)/Z（logit 空间几何混合，product-of-experts）
      loss_mix   loss = ½KL(q||p_A)+½KL(q||p_B)（loss 级加权和）
      route      逐样本按 prompt 领域标记选教师（context routing）
      route_self 逐样本按学生响应首字母的大小写选教师（output routing；
                 非字母 fallback 到算术混合——路由未决时退回融合）"""
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    tA.eval(); tB.eval()
    prompts = torch.tensor([make_prompt_ids(d, dn) for d, dn in pool])
    log_half = -torch.log(torch.tensor(2.0))
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(pool), (batch,), generator=g)
        px = prompts[idx]
        with torch.no_grad():
            resp = student.sample(px, RESP_LEN)              # ← 学生自采样
            full = torch.cat([px, resp], dim=1)
            lA = F.log_softmax(tA(full), dim=-1)             # ← 教师 A 打分
            lB = F.log_softmax(tB(full), dim=-1)             # ← 教师 B 打分
        s_logits = student(full)
        Lp = px.shape[1]
        lq = F.log_softmax(
            s_logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), dim=-1)
        lA_r = lA[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V)
        lB_r = lB[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V)

        if recipe == "mix_prob":
            lp_t = torch.logsumexp(
                torch.stack([lA_r + log_half, lB_r + log_half]), dim=0)
            loss = rev_kl(lq, lp_t).mean()
        elif recipe == "mix_poe":
            lp_unnorm = 0.5 * lA_r + 0.5 * lB_r
            lp_t = lp_unnorm - torch.logsumexp(lp_unnorm, dim=-1, keepdim=True)
            loss = rev_kl(lq, lp_t).mean()
        elif recipe == "loss_mix":
            loss = (0.5 * rev_kl(lq, lA_r) + 0.5 * rev_kl(lq, lB_r)).mean()
        elif recipe == "route":
            is_a = (px[:, MARK_POS] == MARK_A)               # prompt 领域标记
            sel = is_a.repeat_interleave(RESP_LEN)
            lp_t = torch.where(sel.unsqueeze(-1), lA_r, lB_r)
            loss = rev_kl(lq, lp_t).mean()
        elif recipe == "route_self":
            first = resp[:, 0]                               # 学生响应首 token
            # 词表布局保证 LOWER/UPPER 的 id 区间连续，区间判定即大小写判定
            is_lo = (first >= LOWER_IDS.min()) & (first <= LOWER_IDS.max())
            is_up = (first >= UPPER_IDS.min()) & (first <= UPPER_IDS.max())
            lp_mix = torch.logsumexp(
                torch.stack([lA_r + log_half, lB_r + log_half]), dim=0)
            lp_t = lp_mix.clone()
            lo_rows = is_lo.unsqueeze(-1).expand(-1, RESP_LEN).reshape(-1)
            up_rows = is_up.unsqueeze(-1).expand(-1, RESP_LEN).reshape(-1)
            lp_t[lo_rows] = lA_r[lo_rows]                    # 试小写 → 教师A批
            lp_t[up_rows] = lB_r[up_rows]                    # 试大写 → 教师B批
            loss = rev_kl(lq, lp_t).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


# ---------------- 评测：领域化精确匹配 + 背书 ----------------

@torch.no_grad()
def evaluate_domains(student, teachers, pool, m_per_prompt=32, seed=902):
    """逐领域评测：学生自采样 → 与该领域 codebook 精确匹配。
    返回 valid_A/valid_B（领域内有效率）、mode_frac_A（有效响应中 codebook A
    占比——锁模指示；total≈0 时该量是噪声，只在 total 显著时解读）、
    endorse（权威教师对学生样本的逐 token 背书）。"""
    torch.manual_seed(seed)
    student.eval()
    out = {}
    for domain in ("A", "B"):
        sub = [p for p in pool if p[1] == domain]
        px = torch.tensor([make_prompt_ids(d, domain) for d, _ in sub])
        reps = torch.stack(
            [student.sample(px, RESP_LEN) for _ in range(m_per_prompt)], dim=1)
        tgt_own = torch.tensor([target_response(d, domain) for d, _ in sub])
        tgt_other = torch.tensor(
            [target_response(d, "B" if domain == "A" else "A") for d, _ in sub])
        n_own = (reps == tgt_own.unsqueeze(1)).all(-1).sum().item()
        n_other = (reps == tgt_other.unsqueeze(1)).all(-1).sum().item()
        # 背书：权威教师（领域 A → 教师 A）对学生样本逐 token 打分
        t = teachers[domain]
        k = min(8, m_per_prompt)
        sub_rep = reps[:, :k, :].reshape(-1, RESP_LEN)
        px_rep = px.unsqueeze(1).expand(-1, k, -1).reshape(-1, px.shape[1])
        endo = (seq_logprob(t, px_rep, sub_rep) / RESP_LEN).mean().item()
        out[f"valid_{domain}"] = n_own / (len(sub) * m_per_prompt)
        out[f"match_other_{domain}"] = n_other / (len(sub) * m_per_prompt)
        out[f"endorse_{domain}"] = endo
    tot = out["valid_A"] + out["valid_B"]
    out["valid_total"] = tot / 2
    frac_a = out["valid_A"] / max(tot, 1e-9)
    out["mode_frac_A"] = frac_a
    out["lock"] = abs(frac_a - 0.5) * 2          # 0=两域均衡, 1=完全锁一模
    return out


def show_samples(student, pool, n=2, seed=903):
    """每领域抽 n 个 prompt，展示学生实际生成（确定性采样）。"""
    torch.manual_seed(seed)
    student.eval()
    lines = []
    for domain in ("A", "B"):
        sub = [p for p in pool if p[1] == domain][:n]
        for d, _ in sub:
            px = torch.tensor([make_prompt_ids(d, domain)])
            resp = student.sample(px, RESP_LEN)[0].tolist()
            gold = decode(target_response(d, domain)).replace("<end>", "")
            got = decode(resp[:resp.index(END)] if END in resp else resp)
            ok = "OK " if got == gold else "ERR"
            lines.append(f"    {ok} {''.join(map(str, d))} <{domain.lower()}> "
                         f"-> {got:<10s} (gold {gold})")
    return lines


# ---------------- 主流程 ----------------

DEFAULT_CFG = dict(
    seed_pool=7, seed_tA=201, seed_tB=301, seed_init=555, seed_run=777,
    n_per_domain=128, teacher_steps=500,
    student_steps=900, batch=32, lr=2e-3,
    student=dict(d_model=16, nhead=2, layers=1, ff=32),
    eval_m=32,
)

RECIPES = ["mix_prob", "mix_poe", "loss_mix", "route", "route_self"]


def main(cfg=None):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    if cfg.get("student") is DEFAULT_CFG["student"]:
        cfg["student"] = dict(DEFAULT_CFG["student"])
    print("=" * 72)
    print("nano-opd L2 — multi-teacher OPD：融合位置与路由的最小机制")
    print("=" * 72)
    print(f"任务: {CONTENT_LEN} 位数字 + 领域标记 → 该领域唯一正确的 codebook")
    print(f"词表 V={V} | prompt = <in> digits <a|b> <out> | 响应长 {RESP_LEN}")
    print(f"codebook A 例: 3141 -> {''.join(t for d in [3,1,4,1] for t in CODEBOOK_A[d])}   "
          f"codebook B 例: 3141 -> {''.join(t for d in [3,1,4,1] for t in CODEBOOK_B[d])}")

    pool = make_pool(cfg["seed_pool"], cfg["n_per_domain"])

    # ---- [1] 两个立场教师 + 2×2 背书表 ----
    print("\n[1] 两个立场教师（各 500 步；A 教师在任何上下文坚持 codebook A，B 镜像）")
    tA = train_fundamentalist(pool, "A", steps=cfg["teacher_steps"],
                              seed=cfg["seed_tA"])
    tB = train_fundamentalist(pool, "B", steps=cfg["teacher_steps"],
                              seed=cfg["seed_tB"])
    n_t = sum(p.numel() for p in tA.parameters())
    print(f"    teacher params = {n_t:,} × 2")
    tab = endorsement_table({"A": tA, "B": tB}, pool)
    print("    2×2 背书表（nats/token；对角线=自己领域，非对角线=对方领域）")
    print(f"      教师A:  领域A gold {tab['A']['A']:+.3f}   领域B gold {tab['A']['B']:+.3f}")
    print(f"      教师B:  领域A gold {tab['B']['A']:+.3f}   领域B gold {tab['B']['B']:+.3f}")
    assert tab["A"]["A"] - tab["A"]["B"] > 1.0, "教师A必须能区分两个领域"
    assert tab["B"]["B"] - tab["B"]["A"] > 1.0, "教师B必须能区分两个领域"

    # ---- [2] 探针：恒等式 + Jensen 间隙 + ½ 票 ----
    print("\n[2] 探针（初始化学生，teacher-forced，无需训练）")
    probe = probe_fusion_math(fresh_student(cfg["student"], cfg["seed_init"]),
                              tA, tB, pool)
    print(f"    恒等式 ½KL(q||p_A)+½KL(q||p_B) == KL(q||√(p_A·p_B)[未归一化])")
    print(f"      逐 token 最大绝对差 = {probe['ident_max_abs']:.2e}（float32 机器精度）")
    print(f"    Jensen 间隙 gap = loss_mix − KL(q||½(p_A+p_B)) = E_q[log Σw p − Σw log p]")
    print(f"      最小值 = {probe['gap_min']:.2e}（应 ≥ 0，log 凹性）")
    print(f"      字母位均值 = {probe['gap_letters_mean']:.3f} nats/token（教师冲突）")
    print(f"      <end>位均值 = {probe['gap_end_mean']:.3f} nats/token（教师一致）")
    print(f"    ½ 票: 构造性杂交（大小写交替）在算术混合下的字母位 logprob")
    print(f"      位置0 = {probe['hybrid_pos0_lp']:.3f} nats/token（预测 −ln 2 = {-math.log(2):.3f}，两教师都在流形上）")
    print(f"      deeper = {probe['hybrid_deep_lp']:.3f} nats/token（杂交前缀使两教师同时 off-manifold）")
    assert probe["ident_max_abs"] < 1e-5, "恒等式应在机器精度内成立"
    assert probe["gap_min"] > -1e-6, "Jensen 间隙非负（log 凹性）"
    assert probe["gap_letters_mean"] > 3 * max(probe["gap_end_mean"], 1e-6), \
        "教师分歧应集中在内容位（字母），而非结构位（<end>）"
    assert abs(probe["hybrid_pos0_lp"] + math.log(2)) < 0.05, \
        "位置0（prompt 前缀）算术混合给杂交字母恰 ½ 票（−ln 2）"
    assert probe["hybrid_deep_lp"] < -1.0, \
        f"deeper 位置杂交前缀下票应显著衰减: {probe['hybrid_deep_lp']:.3f}"

    # ---- [3] 五个配方：同一 on-policy 循环，只换教师信号组合 ----
    print(f"\n[3] 五配方 OPD（同一初始学生、各 {cfg['student_steps']} 步、"
          f"batch={cfg['batch']}、lr={cfg['lr']}）")
    results, loss_curves, students = {}, {}, {}
    for name in RECIPES:
        student = fresh_student(cfg["student"], cfg["seed_init"])
        losses = train_multi_opd(student, tA, tB, pool, cfg["student_steps"],
                                 cfg["batch"], cfg["lr"], cfg["seed_run"], name)
        results[name] = evaluate_domains(student, {"A": tA, "B": tB}, pool,
                                         m_per_prompt=cfg["eval_m"])
        loss_curves[name] = losses
        students[name] = student
    n_s = sum(p.numel() for p in fresh_student(cfg["student"], 0).parameters())
    print(f"    student params = {n_s:,}（教师/学生 ≈ {n_t/n_s:.0f}x）")
    print(f"    {'recipe':10s} {'valid_A':>8s} {'valid_B':>8s} {'total':>7s} "
          f"{'frac_A':>7s} {'lock':>5s} {'背书A':>7s} {'背书B':>7s}")
    for name in RECIPES:
        m = results[name]
        print(f"    {name:10s} {m['valid_A']:8.3f} {m['valid_B']:8.3f} "
              f"{m['valid_total']:7.3f} {m['mode_frac_A']:7.3f} {m['lock']:5.2f} "
              f"{m['endorse_A']:+7.3f} {m['endorse_B']:+7.3f}")

    # ---- [4] 样本展示 ----
    print("\n[4] 学生实际生成（每领域 2 例；OK=与该领域 gold 精确匹配）")
    for name in RECIPES:
        print(f"  {name}:")
        for line in show_samples(students[name], pool):
            print(line)

    # ---- [5] self-check ----
    print("\n[5] self-check（断言设计见 tutorial §6）")
    mp, poe, lm, rt, rs = (results[k] for k in RECIPES)
    # (a) 恒等式/Jensen/½票 已在 [2] 断言
    # (b) 几何/loss 混合塌缩：token 级 ½/½ 独立目标销毁序列模式，
    #     精确匹配掉到 ≈(½)^8=0.004 量级（理论预测，非经验猜测）
    assert poe["valid_total"] < 0.05, \
        f"几何混合应塌缩: total={poe['valid_total']:.3f}"
    assert lm["valid_total"] < 0.05, \
        f"loss 级混合（≡几何混合）应塌缩: total={lm['valid_total']:.3f}"
    # (c) loss_mix 与 mix_poe 同一目标函数 → 终态一致（恒等式的训练级后果）
    assert abs(poe["valid_total"] - lm["valid_total"]) <= 0.02, \
        f"同一目标函数的两个写法终态应一致: {poe['valid_total']:.3f} vs {lm['valid_total']:.3f}"
    # (d) 算术混合 = 杂交平衡：目标保留两个序列模式，但 on-policy 采样找不到
    #     （任何字母杂交每字母位恰 ½ 票 → 梯度无方向偏好）。不是锁模、不是学会，
    #     是停在 token 级边缘匹配。
    assert mp["valid_total"] < 0.05, \
        f"算术混合下 on-policy 应停在杂交平衡: total={mp['valid_total']:.3f}"
    # (e) 路由覆盖两域：每条样本只由其领域教师打分 = 逐域单教师 OPD
    assert min(rt["valid_A"], rt["valid_B"]) > 0.80, \
        f"route 应两个领域都学会: valid_A={rt['valid_A']:.3f} valid_B={rt['valid_B']:.3f}"
    assert rt["valid_total"] > mp["valid_total"] + 0.50, \
        f"route 应显著优于融合: {rt['valid_total']:.3f} vs {mp['valid_total']:.3f}"
    # (f) 按学生输出路由 = 自确认回路：学会一个领域、锁死另一个领域
    assert rs["valid_total"] > 0.30, \
        f"route_self 应靠自增强学会一个领域: total={rs['valid_total']:.3f}"
    assert (rs["lock"] > 0.90 and min(rs["valid_A"], rs["valid_B"]) < 0.05), (
        f"route_self 应锁一模: lock={rs['lock']:.2f} "
        f"valid_A={rs['valid_A']:.3f} valid_B={rs['valid_B']:.3f}")
    assert rt["valid_total"] > rs["valid_total"] + 0.30, \
        f"route 应显著优于 route_self: {rt['valid_total']:.3f} vs {rs['valid_total']:.3f}"
    print("✅ self-check passed: 恒等式/Jensen/½票/塌缩/杂交平衡/路由覆盖/自确认锁模")

    # ---- digest（关键指标 md5，跨运行收敛检查） ----
    key = "|".join(
        f"{k}:{results[k]['valid_A']:.3f},{results[k]['valid_B']:.3f},"
        f"{results[k]['mode_frac_A']:.3f}" for k in RECIPES)
    key += f"|gap:{probe['gap_letters_mean']:.3f}/{probe['gap_end_mean']:.3f}"
    key += f"|half:{probe['hybrid_pos0_lp']:.3f}/{probe['hybrid_deep_lp']:.3f}"
    digest = hashlib.md5(key.encode()).hexdigest()[:16]
    print(f"\ndigest(关键指标) = {digest}")
    print("\ntakeaway: 教师冲突时，三个融合位置全失败，且失败方式各有数学签名：")
    print("          loss 级/logit 级混合销毁序列模式（恒等式：两者是同一目标函数）→ 塌缩；")
    print("          概率级混合保留模式，但 on-policy 采样找不到模式（½ 票杂交平衡）→ 瘫痪；")
    print("          按学生输出路由 = 自确认回路 → 锁一模牺牲一个领域。")
    print("          唯一成功的是 context routing：冲突在融合之前解决——")
    print("          每条学生样本只由其领域教师打分。路由信号必须来自上下文。")
    return results


def probe_mixprob_patience(T=1.0, steps=2400, eval_every=600, lr=2e-3,
                           m_per_prompt=8):
    """边界探针（OPD_L2_PROBE=1 时运行）：mix_prob 的杂交平衡是「步数不够」
    还是「教师太锐」？对算术混合目标做长训练 + 周期性评测。
    T 对教师 logprob 施加温度（目标构造时，温度下重归一化）——token 级软化。
    注意：本探针训练 2400 步，耗时数倍于主流程，仅显式触发。"""
    cfg = DEFAULT_CFG
    pool = make_pool(cfg["seed_pool"], cfg["n_per_domain"])
    tA = train_fundamentalist(pool, "A", steps=cfg["teacher_steps"],
                              seed=cfg["seed_tA"])
    tB = train_fundamentalist(pool, "B", steps=cfg["teacher_steps"],
                              seed=cfg["seed_tB"])
    student = fresh_student(cfg["student"], cfg["seed_init"])
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = torch.Generator().manual_seed(cfg["seed_run"])
    prompts = torch.tensor([make_prompt_ids(d, dn) for d, dn in pool])
    log_half = -torch.log(torch.tensor(2.0))
    tA.eval(); tB.eval()
    print(f"--- probe: mix_prob T={T} steps={steps} lr={lr} "
          f"(eval m_per_prompt={m_per_prompt}) ---")
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(pool), (cfg["batch"],), generator=g)
        px = prompts[idx]
        with torch.no_grad():
            resp = student.sample(px, RESP_LEN)
            full = torch.cat([px, resp], dim=1)
            lA = F.log_softmax(tA(full), dim=-1) / T
            lB = F.log_softmax(tB(full), dim=-1) / T
        s_logits = student(full)
        Lp = px.shape[1]
        lq = F.log_softmax(
            s_logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), dim=-1)
        lA_r = lA[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V)
        lB_r = lB[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V)
        lp_t = torch.logsumexp(torch.stack([lA_r + log_half, lB_r + log_half]),
                               dim=0)
        lp_t = lp_t - torch.logsumexp(lp_t, dim=-1, keepdim=True)
        loss = rev_kl(lq, lp_t).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0:
            r = evaluate_domains(student, {"A": tA, "B": tB}, pool,
                                 m_per_prompt=m_per_prompt, seed=902)
            print(f"    step {step:4d}: valid_A={r['valid_A']:.3f} "
                  f"valid_B={r['valid_B']:.3f} lock={r['lock']:.2f}")


if __name__ == "__main__":
    if os.environ.get("OPD_L2_PROBE"):
        for _T in (1.0, 3.0):
            probe_mixprob_patience(T=_T)
    else:
        main()
