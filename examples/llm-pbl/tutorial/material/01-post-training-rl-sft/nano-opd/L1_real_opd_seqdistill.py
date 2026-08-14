#!/usr/bin/env python3
"""nano-opd L1 — 真实序列模型上的 OPD：学生自采样 + 教师 token 级 logprob 打分

L0 在玩具网格上证明了「散度选择决定采样需求」（reverse KL 的期望在学生分布下 →
必须学生自采样）。L1 把同一机制搬进真实序列模型：

  - 真实 tokenizer（字符级，确定性 char→id 表）
  - 两个真实小 Transformer：教师（大）与学生（小，容量受限）
  - 教师 logprob 由真实前向算出（白盒教师），学生用真实梯度下降训练

任务是「双语言密码本」：给一串数字，把它翻译成两种合法语言之一——
  模式 A：每个数字 → 2 个小写字母（codebook A）
  模式 B：每个数字 → 2 个大写字母（codebook B）
教师对两种语言都打满分（同一输入、两个模，镜像 L0 的双峰教师）；
学生容量受限，装不下两份密码本时，不同配方的收敛形态就显形（镜像 L0 的受限学生）。

2×2 因子设计（精确隔离「信号从哪来」与「优化什么」两个变量）：

  recipe      前缀/数据来源          监督信号
  ----------  ---------------------  ------------------------------
  sft         教师采样序列(离线)      硬标签 CE（MLE）
  kd          教师采样序列(teacher-forced)  token 级 forward KL（软标签，经典 KD）
  opd_off     教师采样序列(teacher-forced)  token 级 reverse KL  ← L0 的错误估计器同款
  opd         学生自采样(on-policy)   token 级 reverse KL        ← MiniLLM/GKD 路线

对照锚点：MiniLLM [arXiv:2306.08543]（reverse KL + on-policy 优化，防学生在教师
低概率区过度赋 mass）；GKD [arXiv:2306.13649]（在学生自生成序列上训练 + 可选散度，
「learning from self-generated mistakes」）；DistiLLM [arXiv:2402.03898]（skew KL +
自适应 off-policy 复用学生样本）。本文件的 token 级 reverse-KL-on-student-samples
即 GKD 框架的一个实例化；MiniLLM 原文用序列级 score-function 梯度（L0 已实现），
token 级版本方差更低，是生产配方的常见选择（survey [2604.00626] 的 token-level 信号）。

显式声明（反幻觉）：
  1. 本节无 mock：教师与学生的前向/反向/采样全部是真实 torch 计算。
  2. 「白盒教师」假设：能取教师在整个词表上的分布。黑盒 API 教师只能给采样 token
     的 logprob，配方要相应调整（序列级 REINFORCE / token 级 advantage 形式）——
     与 nano-verl L1 的 importance sampling 相接，留 L3。
  3. 学生比教师小是刻意的容量设置（同 L0）：mode-covering vs mode-seeking 的差异
     只在学生装不下教师全分布时显形。
  4. 任务可泛化（替换密码本是合成任务），但本 toy 的结论只对 toy 口径成立：
     数字为现场实跑，不是任何真实模型的 benchmark。

依赖仅 torch（CPU 即跑）。固定 seed 确定性；不同 seed 的鲁棒性见教程 §5。
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 词表与任务（真实字符级 tokenizer） ----------------

SPECIALS = ["<pad>", "<in>", "<out>", "<end>"]
DIGITS = [str(i) for i in range(10)]
LOWER = [chr(ord("a") + i) for i in range(26)]
UPPER = [chr(ord("A") + i) for i in range(26)]
VOCAB = SPECIALS + DIGITS + LOWER + UPPER
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for i, t in enumerate(VOCAB)}
V = len(VOCAB)
PAD, IN, OUT, END = TOK2ID["<pad>"], TOK2ID["<in>"], TOK2ID["<out>"], TOK2ID["<end>"]

CONTENT_LEN = 4                    # 每个 prompt 的数字个数
RESP_LEN = 2 * CONTENT_LEN + 1     # 响应 = 每数字 2 字母 + <end> = 9

# 两份密码本：digit -> 2 字母。小写/大写天然不相交，首字符即暴露模。
# 置换固定写死（开发期用 seed 生成后固化），保证任何机器逐字节复现。
_PERM_A = [7, 2, 19, 11, 24, 0, 15, 9, 21, 4, 17, 12, 25, 3, 14, 8, 22, 1, 18, 10,
           23, 5, 16, 6, 20, 13]
_PERM_B = [13, 20, 6, 16, 5, 23, 10, 18, 1, 22, 8, 14, 3, 25, 12, 17, 4, 21, 9, 15,
           0, 24, 11, 19, 2, 7]
CODEBOOK_A = {d: (LOWER[_PERM_A[2 * d]], LOWER[_PERM_A[2 * d + 1]]) for d in range(10)}
CODEBOOK_B = {d: (UPPER[_PERM_B[2 * d]], UPPER[_PERM_B[2 * d + 1]]) for d in range(10)}


def encode(text):
    return [TOK2ID[c] for c in text]


def decode(ids):
    return "".join(ID2TOK[int(i)] for i in ids)


def make_prompt_ids(digits):
    """'<in> 3 1 4 1 <out>' -> id 列表（响应的预测从这里开始）。"""
    return [IN] + digits + [OUT]


def target_response(digits, mode):
    cb = CODEBOOK_A if mode == "A" else CODEBOOK_B
    toks = [t for d in digits for t in cb[d]] + ["<end>"]
    return [TOK2ID[t] for t in toks]


def make_pool(seed, n):
    g = torch.Generator().manual_seed(seed)
    pool = []
    for _ in range(n):
        pool.append(torch.randint(0, 10, (CONTENT_LEN,), generator=g).tolist())
    return pool


# ---------------- 模型：真实小 Transformer（causal LM） ----------------

class TinyLM(nn.Module):
    """与 nano-llamafactory L1 同款结构：token embed + 学习位置编码 +
    nn.TransformerEncoderLayer(显式 causal mask) + LM head。dropout=0 保证确定性。"""

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
        return self.head(h)                      # logits [B, L, V]

    @torch.no_grad()
    def sample(self, prompt_ids, resp_len, temperature=1.0):
        """自回归多项式采样：prompt 批量对齐，固定生成 resp_len 个 token。"""
        ids = prompt_ids.clone()
        for _ in range(resp_len):
            logits = self.forward(ids)[:, -1, :] / temperature
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids[:, prompt_ids.shape[1]:]      # 只返回响应部分


def seq_logprob(model, prompt_ids, resp_ids):
    """序列级 log p(resp|prompt)（逐 token 求和），用于教师背书度指标。"""
    full = torch.cat([prompt_ids, resp_ids], dim=1)
    logits = model(full)
    lp = F.log_softmax(logits, dim=-1)
    Lp = prompt_ids.shape[1]
    # 位置 t 的 logits 预测 t+1；响应 token 在全序列的 [Lp, Lp+R)
    pred = lp[:, Lp - 1: Lp - 1 + resp_ids.shape[1], :]
    return pred.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)


# ---------------- 教师：把两种语言都学下来 ----------------

def build_batch(digits_list, mode, device):
    """一个 (prompt, response) batch：mode 可为 'A'/'B' 或逐样本指定列表。"""
    prompts, resps = [], []
    for i, digits in enumerate(digits_list):
        m = mode if isinstance(mode, str) else mode[i]
        prompts.append(make_prompt_ids(digits))
        resps.append(target_response(digits, m))
    x = torch.tensor([p for p in prompts], device=device)
    y = torch.tensor([r for r in resps], device=device)
    return x, y


def train_teacher(pool, d_model=64, nhead=4, layers=2, ff=256,
                  steps=600, batch=32, lr=2e-3, seed=101, device="cpu"):
    torch.manual_seed(seed)
    model = TinyLM(d_model, nhead, layers, ff).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    for step in range(steps):
        idx = torch.randint(0, len(pool), (batch,), generator=g).tolist()
        digits = [pool[i] for i in idx]
        mode = ["A" if r < 0.5 else "B" for r in torch.rand(batch, generator=g).tolist()]
        x, y = build_batch(digits, mode, device)
        logits = model(torch.cat([x, y[:, :-1]], dim=1))
        Lp = x.shape[1]
        # 响应 token 由位置 Lp-1 起的 logits 预测（shifted CE，同 nano-llamafactory L1）
        loss = F.cross_entropy(
            logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V), y.reshape(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def teacher_audit(teacher, pool, n_sample_prompts=64, m_per_prompt=32, device="cpu"):
    """审计教师：采样有效率 / 两个模的占比 / 对合法 vs 杂交序列的背书差。"""
    sub = pool[:n_sample_prompts]
    prompts = torch.tensor([make_prompt_ids(d) for d in sub], device=device)
    reps = torch.stack(
        [teacher.sample(prompts, RESP_LEN) for _ in range(m_per_prompt)], dim=1
    )                                                          # [N, M, R]
    tgt_a = torch.tensor([target_response(d, "A") for d in sub], device=device)
    tgt_b = torch.tensor([target_response(d, "B") for d in sub], device=device)
    n_valid_a = (reps == tgt_a.unsqueeze(1)).all(-1).sum().item()
    n_valid_b = (reps == tgt_b.unsqueeze(1)).all(-1).sum().item()
    n_total = n_sample_prompts * m_per_prompt
    # 背书差：同一批 prompt 上，合法序列 vs 大小写杂交序列的平均 token logprob
    # 杂交：A 的字母强行大小写交错（既非 A 也非 B 的序列）
    hyb = []
    for d in sub:
        toks = [t for dd in d for t in CODEBOOK_A[dd]] + ["<end>"]
        toks = [t.upper() if i % 2 == 0 else t.lower()
                for i, t in enumerate(toks[:-1])] + ["<end>"]
        hyb.append([TOK2ID[t] for t in toks])
    ry_h = torch.tensor(hyb, device=device)
    lp_valid = seq_logprob(teacher, prompts, tgt_a) / RESP_LEN
    lp_hyb = seq_logprob(teacher, prompts, ry_h) / RESP_LEN
    return {
        "valid_rate": (n_valid_a + n_valid_b) / n_total,
        "mode_a_frac": n_valid_a / max(n_valid_a + n_valid_b, 1),
        "lp_valid": lp_valid.mean().item(),
        "lp_hybrid": lp_hyb.mean().item(),
    }


# ---------------- 四个配方 ----------------

def fresh_student(cfg, seed):
    torch.manual_seed(seed)
    return TinyLM(cfg["d_model"], cfg["nhead"], cfg["layers"], cfg["ff"])


def train_sft(student, teacher_data, steps, batch, lr, seed, device="cpu"):
    """SFT 蒸馏：教师离线采样数据集 + 硬标签 MLE。信号=教师生成的数据。"""
    px, ry = teacher_data
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = px.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        logits = student(torch.cat([px[idx], ry[idx, :-1]], dim=1))
        Lp = px.shape[1]
        loss = F.cross_entropy(
            logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V),
            ry[idx].reshape(-1),
        )
        opt.zero_grad(); loss.backward(); opt.step()


def token_kl(student_logits, teacher_logits, reverse):
    """token 级 KL。logits: [N, V]（已对齐到同一批响应位置）。
    reverse=False: KL(p_T||q_S)（forward，经典 KD 软标签）
    reverse=True : KL(q_S||p_T)（reverse，OPD）"""
    lq = F.log_softmax(student_logits, dim=-1)
    lp = F.log_softmax(teacher_logits, dim=-1)
    if reverse:
        return (lq.exp() * (lq - lp)).sum(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def train_kd(student, teacher, teacher_data, steps, batch, lr, seed,
             reverse=False, device="cpu"):
    """kd：teacher-forced 教师序列 + token 级 forward KL（经典白盒 KD）。
    reverse=True 时变体：同样的教师序列上换 reverse KL —— 即 L0 错误估计器的
    真实模型版（期望该在学生分布下，却用了教师序列）。全程无学生采样。"""
    px, ry = teacher_data
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = px.shape[0]
    teacher.eval()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        full = torch.cat([px[idx], ry[idx, :-1]], dim=1)
        with torch.no_grad():
            t_logits = teacher(full)
        s_logits = student(full)
        Lp = px.shape[1]
        loss = token_kl(s_logits[:, Lp - 1:, :].reshape(-1, V),
                        t_logits[:, Lp - 1:, :].reshape(-1, V), reverse).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def train_opd(student, teacher, pool, steps, batch, lr, seed, device="cpu",
              log_every=0):
    """OPD（GKD 式 token 级 reverse KL on 学生自采样）：
    每步：学生就 batch 个 prompt 自采样完整响应（on-policy）→
    教师一次前向给全词表分布（只打分，不生成、不求梯度）→
    token 级 KL(q_S||p_T) 反传。采样前缀视为常数（梯度不穿过采样过程，
    与 GKD 一致；序列级 score-function 版见 L0）。"""
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    teacher.eval()
    curve = []
    prompts_all = torch.tensor([make_prompt_ids(d) for d in pool], device=device)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, len(pool), (batch,), generator=g)
        px = prompts_all[idx]
        with torch.no_grad():
            resp = student.sample(px, RESP_LEN)          # ← 学生自采样
            full = torch.cat([px, resp], dim=1)
            t_logits = teacher(full)                     # ← 教师打分（白盒）
        s_logits = student(full)
        Lp = px.shape[1]
        # 位置 Lp-1 起的 logits 依次预测响应的 RESP_LEN 个 token
        loss = token_kl(
            s_logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V),
            t_logits[:, Lp - 1: Lp - 1 + RESP_LEN, :].reshape(-1, V),
            reverse=True,
        ).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            m = evaluate(student, teacher, pool, m_per_prompt=16, device=device)
            curve.append((step, m["valid"], m["endorse"], time.time() - t0))
    return curve


# ---------------- 评测：同一协议评所有配方 ----------------

@torch.no_grad()
def evaluate(student, teacher, pool, m_per_prompt=64, device="cpu"):
    """学生采样 m_per_prompt 条响应/prompt：精确匹配两个模、统计杂交，
    并算教师对学生样本的平均 token logprob（背书度）。"""
    prompts = torch.tensor([make_prompt_ids(d) for d in pool], device=device)
    reps = []
    for _ in range(m_per_prompt):
        reps.append(student.sample(prompts, RESP_LEN))
    reps = torch.stack(reps, dim=1)                      # [N, M, R]
    tgt_a = torch.tensor([target_response(d, "A") for d in pool], device=device)
    tgt_b = torch.tensor([target_response(d, "B") for d in pool], device=device)
    n_valid_a = (reps == tgt_a.unsqueeze(1)).all(-1).sum().item()
    n_valid_b = (reps == tgt_b.unsqueeze(1)).all(-1).sum().item()
    # 背书度：逐 prompt 抽 8 条学生样本算教师 logprob（控制算力）
    k = min(8, m_per_prompt)
    sub = reps[:, :k, :].reshape(-1, RESP_LEN)
    px_rep = prompts.unsqueeze(1).expand(-1, k, -1).reshape(-1, prompts.shape[1])
    lp = seq_logprob(teacher, px_rep, sub)
    return {
        "valid": (n_valid_a + n_valid_b) / (len(pool) * m_per_prompt),
        "mode_a_frac": n_valid_a / max(n_valid_a + n_valid_b, 1),
        "endorse": (lp / RESP_LEN).mean().item(),
    }


def report(tag, m):
    print(f"  {tag:22s} 有效率={m['valid']:.3f} | A模占有效={m['mode_a_frac']:.3f} "
          f"| 教师背书={m['endorse']:+.3f} nats/token")


# ---------------- 主流程 ----------------

# 全部可调旋钮集中于此（开发期扫描用；定稿配置见下方默认值）。
DEFAULT_CFG = dict(
    seed_pool=7, seed_teacher=101, seed_init=555, seed_run=777,
    n_prompts=256, teacher_steps=600,
    student_steps=300, batch=32, lr=2e-3,
    student=dict(d_model=16, nhead=2, layers=1, ff=32),
    eval_m=64,
)


def main(cfg=None):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    if "student" in cfg and cfg["student"] is DEFAULT_CFG["student"]:
        cfg["student"] = dict(DEFAULT_CFG["student"])
    DEV = "cpu"
    SEED_POOL, SEED_TEACHER = cfg["seed_pool"], cfg["seed_teacher"]
    SEED_INIT, SEED_RUN = cfg["seed_init"], cfg["seed_run"]
    N_PROMPTS, TEACHER_STEPS = cfg["n_prompts"], cfg["teacher_steps"]
    STUDENT_STEPS, BATCH, LR = cfg["student_steps"], cfg["batch"], cfg["lr"]
    STUDENT_CFG = cfg["student"]

    print("=" * 72)
    print("nano-opd L1 — 真实序列模型上的 OPD（学生自采样 + 教师 logprob 打分）")
    print("=" * 72)
    print(f"任务: {CONTENT_LEN} 位数字 → 两种合法语言之一 "
          f"(A=小写密码本, B=大写密码本)；响应长 {RESP_LEN}")
    print(f"词表 V={V}（字符级 tokenizer）| prompt 池={N_PROMPTS}")
    print(f"codebook A 例: 3141 -> {''.join(t for d in [3,1,4,1] for t in CODEBOOK_A[d])}")
    print(f"codebook B 例: 3141 -> {''.join(t for d in [3,1,4,1] for t in CODEBOOK_B[d])}")

    pool = make_pool(SEED_POOL, N_PROMPTS)

    # ---- [1] 教师：真实训练 + 审计 ----
    print("\n[1] 教师训练（两种语言各半，masked SFT）与审计")
    teacher = train_teacher(pool, seed=SEED_TEACHER, steps=TEACHER_STEPS, device=DEV)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    audit = teacher_audit(teacher, pool, device=DEV)
    print(f"    teacher params = {n_teacher:,}")
    print(f"    采样有效率 = {audit['valid_rate']:.3f} | A模占有效 = {audit['mode_a_frac']:.3f}")
    print(f"    教师背书: 合法序列 {audit['lp_valid']:+.3f} vs 杂交序列 "
          f"{audit['lp_hybrid']:+.3f} nats/token（差 {audit['lp_valid']-audit['lp_hybrid']:+.3f}）")
    assert audit["valid_rate"] > 0.95, "教师必须近乎完美，否则对比失去基准"
    assert audit["lp_valid"] - audit["lp_hybrid"] > 1.0, "教师必须能区分合法与杂交"

    # ---- [2] 教师离线数据集（sft/kd/opd_off 共用，隔离数据源变量） ----
    torch.manual_seed(SEED_RUN + 1)
    tpx = torch.tensor([make_prompt_ids(d) for d in pool], device=DEV)
    treps = []
    with torch.no_grad():
        for _ in range(2):                       # 每 prompt 采 2 条，混模
            treps.append(teacher.sample(tpx, RESP_LEN))
    teacher_data = (tpx.repeat(2, 1), torch.cat(treps, dim=0))

    # ---- [3] 四个配方：同一学生初始化、同步数、同 lr ----
    print(f"\n[2] 四配方训练（同一初始权重、各 {STUDENT_STEPS} 步、batch={BATCH}、lr={LR}）")
    results = {}
    curves = {}
    recipes = ["sft", "kd", "opd_off", "opd"]
    for name in recipes:
        student = fresh_student(STUDENT_CFG, SEED_INIT).to(DEV)
        if name == "sft":
            train_sft(student, teacher_data, STUDENT_STEPS, BATCH, LR, SEED_RUN, DEV)
        elif name == "kd":
            train_kd(student, teacher, teacher_data, STUDENT_STEPS, BATCH, LR,
                     SEED_RUN, reverse=False, device=DEV)
        elif name == "opd_off":
            train_kd(student, teacher, teacher_data, STUDENT_STEPS, BATCH, LR,
                     SEED_RUN, reverse=True, device=DEV)
        else:
            curve = train_opd(student, teacher, pool, STUDENT_STEPS, BATCH, LR,
                              SEED_RUN, DEV, log_every=60)
            curves[name] = curve
        results[name] = evaluate(student, teacher, pool, m_per_prompt=cfg["eval_m"],
                                 device=DEV)

    n_student = sum(p.numel() for p in fresh_student(STUDENT_CFG, SEED_INIT).parameters())
    print(f"    student params = {n_student:,}（教师/学生 ≈ {n_teacher/n_student:.0f}x）")
    for name in recipes:
        report(name, results[name])

    print("\n[3] OPD 训练动力学（on-policy：数据随学生自己变化）")
    for step, v, e, dt in curves["opd"]:
        print(f"    step {step:4d}: 有效率={v:.3f} | 教师背书={e:+.3f} | 累计 {dt:.1f}s")

    # ---- [4] 对照 L0 的结论 ----
    print("\n[4] L0 结论在真实序列模型上的落点")
    sft, kd, off, opd = (results[k] for k in recipes)
    print(f"    a) 信号源固定(教师序列)，换散度: kd(fwd) 有效率={kd['valid']:.3f} vs "
          f"opd_off(rev)={off['valid']:.3f} —— reverse KL 用在教师前缀上不锁模")
    print(f"    b) 散度固定(rev)，换信号源: opd_off(教师前缀)={off['valid']:.3f} vs "
          f"opd(学生自采样)={opd['valid']:.3f} —— on-policy 是 reverse KL 的算术要求")
    print(f"    c) 教师背书: opd {opd['endorse']:+.3f} > sft {sft['endorse']:+.3f} "
          f"—— 学生写出来的东西，教师认不认账")

    print("\n[5] self-check")
    # 核心机制验证——三条断言分别测三个理论预测：
    # (a) 信号源效应：同一 reverse KL，on-policy 的教师背书必须最高
    #     理论依据：on-policy reverse KL = -E_{q_S}[log p_T] - H(q_S)，
    #     优化直接最大化教师对学生样本的背书度；off-policy 在教师前缀上
    #     优化同一散度，但前缀分布不匹配 → 背书度较低。
    #     用背书度（连续量）而非有效率（二值量）做比较——有效率在双方都
    #     收敛后差距消失，背书度在整个训练过程中保持区分度。
    assert opd["endorse"] > off["endorse"], (
        f"on-policy reverse KL 背书({opd['endorse']:+.3f}) "
        f"应高于 off-policy({off['endorse']:+.3f})")
    # (b) 背书度排序：OPD > SFT。SFT 用硬标签 MLE 在教师离线样本上训练，
    #     学生学到的分布偏向教师采样的频率分布；OPD 在学生自己采样的序列上
    #     直接优化教师背书 → 学生写出来的东西教师更认账。
    assert opd["endorse"] > sft["endorse"], (
        f"OPD 背书({opd['endorse']:+.3f}) 应高于 SFT({sft['endorse']:+.3f})")
    # (c) 锁模 vs 覆盖：reverse KL on-policy → mode-seeking（锁一模）；
    #     MLE → mode-covering（覆盖两模）。阈值 0.80 留裕量（300 步刚过
    #     相变，锁模尚在发展；理论方向正确即可，精确值依赖步数与 seed）。
    lock = max(opd["mode_a_frac"], 1 - opd["mode_a_frac"])
    assert lock > 0.80, (
        f"OPD 应锁定一个模（mode-seeking）：lock={lock:.3f}，"
        f"mode_a_frac={opd['mode_a_frac']:.3f}")
    cover = max(sft["mode_a_frac"], 1 - sft["mode_a_frac"])
    assert cover < 0.90, (
        f"SFT 蒸馏应两个模都覆盖（mode-covering）：cover={cover:.3f}")
    assert sft["valid"] > 0.05, "SFT 学生应学到一些东西（否则对比无意义）"
    print("✅ self-check passed: on-policy 背书优势 / 锁模 vs 覆盖")

    print("\ntakeaway: 同一个 reverse KL，前缀从教师换成学生自己，行为从「覆盖两模」")
    print("          变成「锁定一模 + 教师背书最高」——L0 的算术（期望在学生分布下）")
    print("          在真实序列模型上逐字成立。教师全程只打分、不生成、不求梯度。")
    return results


if __name__ == "__main__":
    main()
