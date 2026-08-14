#!/usr/bin/env python3
"""
L2_dpo_preference_pairs.py — nano-llamafactory L2

DPO (Direct Preference Optimization) 建立在 L0/L1 的数据侧三件套之上：
    偏好对用同一个 chat template 渲染、同一套 -100 遮罩、同一个双层遮罩 collator——
    一个偏好对就是两条普通 SFT 行，新的只是 loss 怎么消费它们：
        loss = -log sigmoid( beta * ( margin_policy - margin_ref ) )
        margin = logp(chosen) - logp(rejected)
    即 DPO 目标函数（arXiv 2305.18290 Eq. 7，ar5iv 2026-08-13 抓取逐字核验）。

实验设计（全部真实 torch 梯度下降，CPU，固定种子确定性输出）：
    [0] pairwise collate 机制：偏好对 = 两条普通 SFT 行（与 L1 collate 逐位相同的机器证明）
    [1] SFT 基线：干净数据 vs 含噪数据（正确答案与 off-by-one 错答都当正例喂）
        —— 含噪 SFT 会 mode-cover：概率质量在对错答案之间劈开，greedy 开始答错
    [2] DPO 从含噪 ref 修复（beta=0.1，LLaMA-Factory pref_beta 默认值）：
        只靠「哪个更好」的比较信号，不需要干净目标
    [3] beta 扫描 + 答案位分布探针：pair loss 只约束 chosen/rejected 的相对质量，
        对 pair 之外的概率质量完全失明——win 6/6 与 greedy 崩坏可以同时成立
    [4] 反例：颠倒的偏好对把模型教成「自信地答错」——比较信号本身没有方向感

与权威实现的对应（LLaMA-Factory main @0bbe481e，2026-08-13 codeload tarball 抓取）：
    data/collator.py:L564            PairwiseDataCollatorWithPadding（2n 行，前 n = chosen）
    data/processor/pairwise.py:L66   chosen_labels = [IGNORE_INDEX]*source_len + chosen_ids
    train/dpo/trainer.py:L219-253    concatenated_forward（单次前向，split(batch//2)）
    train/trainer_utils.py:L592      get_batch_logps（shifted gather、-100 遮罩、求和）
    hparams/finetuning_args.py:L171  pref_beta 默认 0.1；L183 pref_loss 默认 sigmoid

依赖：torch（CPU 即可，实测 ~4s）。
"""

import copy
import hashlib
import random
import re
import time

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise SystemExit(
        "[error] torch is required to run this L2 script.\n"
        "        Install it with: pip install torch\n"
        "        (CPU is enough; no transformers needed.)"
    ) from e


SEED = 42
PAD = "<pad>"
SYS, USR, ASST, EOT = "<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"
IGNORE_INDEX = -100  # LLaMA-Factory extras/constants.py:L50（2026-08-13 复验零漂移）

TOKEN_RE = re.compile(r"<\|[\w]+\|>|\n|\S+")

# ---------- 超参数：与 L1 同构（75K 参数 TinyLM），CPU 分钟级 ----------
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
MAX_LEN = 64
LR = 5e-3
SFT_EPOCHS = 300
DPO_STEPS = 200
BETA_MAIN = 0.1  # = LLaMA-Factory pref_beta 默认值（finetuning_args.py:L171-173）
BETA_SWEEP = [0.1, 0.5, 2.0]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text):
    return TOKEN_RE.findall(text)


def apply_chat_template(system, user, assistant=None):
    """与 L1 完全相同的 ChatML 风格 toy 模板。"""
    text = f"{SYS}\n{system}\n{EOT}\n{USR}\n{user}\n{EOT}\n{ASST}\n"
    if assistant is not None:
        text += f"{assistant}\n{EOT}"
    return text


def build_vocab(texts):
    specials = [PAD, SYS, USR, ASST, EOT]
    words = sorted({t for tx in texts for t in tokenize(tx)
                    if t not in specials and t != "\n"})
    return specials + words + ["\n"]


def build_labels(prompt_ids, response_ids):
    """与 L1 相同：prompt 全 -100，response（含 eot）进 loss。"""
    return [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)


def make_row(vocab, system, question, answer):
    """渲染一条 (prompt+answer) 的未 pad 行：(input_ids, attention_mask, labels)。
    与 L1 的构造逐位同构——这是 [0] 连续性证明的基准。"""
    tok2id = {t: i for i, t in enumerate(vocab)}
    full = apply_chat_template(system, question, answer)
    prompt = apply_chat_template(system, question)
    ids = [tok2id[t] for t in tokenize(full)]
    prompt_ids = [tok2id[t] for t in tokenize(prompt)]
    assert ids[:len(prompt_ids)] == prompt_ids, "推理 prompt 必须是训练串的真前缀"
    labels = build_labels(prompt_ids, ids[len(prompt_ids):])
    am = [1] * len(ids)
    return ids, am, labels


def collate_rows(rows, vocab):
    """与 L1 相同的右 pad 双层遮罩 collator。rows = [(ids, am, labels), ...]"""
    pid = vocab.index(PAD)
    L = max(len(ids) for ids, _, _ in rows)
    batch_ids, batch_am, batch_labels = [], [], []
    for ids, am, labels in rows:
        n = L - len(ids)
        batch_ids.append(ids + [pid] * n)
        batch_am.append(am + [0] * n)
        batch_labels.append(labels + [IGNORE_INDEX] * n)
    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(batch_am, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )


def pairwise_collate(pairs, vocab):
    """偏好对 → 2n 行批。

    对照 LLaMA-Factory PairwiseDataCollatorWithPadding
    （data/collator.py:L564，2026-08-13 抓取）：
        "We generate 2 * n examples where the first n examples represent
         chosen examples and the last n examples represent rejected examples."
    即：先排全部 chosen，再排全部 rejected；每行本身只是普通 SFT 行。
    """
    chosen_rows = [make_row(vocab, sys_, q, c) for (sys_, q, c, _r) in pairs]
    rejected_rows = [make_row(vocab, sys_, q, r) for (sys_, q, _c, r) in pairs]
    return collate_rows(chosen_rows + rejected_rows, vocab), chosen_rows, rejected_rows


class TinyLM(nn.Module):
    """与 L1 完全相同的极小 causal LM（约 75K 参数）。"""

    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 max_len=MAX_LEN):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        key_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(
            torch.ones((L, L), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        h = self.blocks(h, mask=causal_mask, src_key_padding_mask=key_mask)
        return self.head(self.norm(h))


def seq_logps(model, input_ids, attention_mask, labels):
    """每条序列的 label 对数概率之和（shifted gather + -100 遮罩）。

    对照 LLaMA-Factory get_batch_logps（train/trainer_utils.py:L592）：
        labels = labels[:, 1:]; logits = logits[:, :-1, :]
        loss_mask = labels != IGNORE_INDEX
        per_token_logps = gather(log_softmax(logits), labels)
        logps = (per_token_logps * mask).sum(-1)   # sigmoid 族用「和」
    （ipo/orpo/simpo 族再除以 valid_length——长度归一，见 trainer.py:L234-241。）
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != IGNORE_INDEX).float()
    safe = shift_labels.clamp(min=0)
    per_tok = torch.gather(
        shift_logits.log_softmax(-1), dim=2, index=safe.unsqueeze(2)
    ).squeeze(2)
    return (per_tok * mask).sum(-1)


def sft_loss(logits, labels):
    """与 L1 相同的 shifted CE（prompt/pad 全 -100）。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def train_sft(vocab, rows, epochs=SFT_EPOCHS):
    """全批 SFT（与 L1 同构）。rows 里每条都是「正例」——含噪 SFT 就是把
    rejected 行也混进来当正例喂。"""
    input_ids, attn_mask, labels = collate_rows(rows, vocab)
    model = TinyLM(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = sft_loss(model(input_ids, attention_mask=attn_mask), labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    """DPO 目标（arXiv 2305.18290 Eq. 7）：
        -log sigmoid( beta * ( (logp_w - logp_l)_policy - (logp_w - logp_l)_ref ) )
    LLaMA-Factory 经 compute_preference_loss（train/dpo/trainer.py:L187）
    委托给 trl DPOTrainer.dpo_loss（trainer.py:L27 import），默认 sigmoid 族。
    """
    margin_policy = policy_chosen_logps - policy_rejected_logps
    margin_ref = ref_chosen_logps - ref_rejected_logps
    logits = beta * (margin_policy - margin_ref)
    return -nn.functional.logsigmoid(logits).mean(), margin_policy.mean().item()


def train_dpo(vocab, pairs, ref_model, beta, steps=DPO_STEPS, log_every=None):
    """从 ref 深拷贝出 policy，用偏好对训练。

    对照 LLaMA-Factory concatenated_forward（train/dpo/trainer.py:L219-253）：
    2n 行一次前向，get_batch_logps 后 split(batch//2) 得 chosen/rejected。
    """
    (input_ids, attn_mask, labels), _, _ = pairwise_collate(pairs, vocab)
    n = len(pairs)

    ref = copy.deepcopy(ref_model)
    ref.eval()
    ref_logps = seq_logps(ref, input_ids, attn_mask, labels)
    ref_chosen, ref_rejected = ref_logps.split(n, dim=0)

    policy = copy.deepcopy(ref_model)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    curve = []
    for step in range(steps):
        policy.train()
        opt.zero_grad()
        # 与 L1 的 SFT 前向完全相同的批；区别只在 loss 怎么消费
        logits = policy(input_ids, attention_mask=attn_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = (shift_labels != IGNORE_INDEX).float()
        safe = shift_labels.clamp(min=0)
        per_tok = torch.gather(
            shift_logits.log_softmax(-1), dim=2, index=safe.unsqueeze(2)
        ).squeeze(2)
        logps = (per_tok * mask).sum(-1)
        pol_chosen, pol_rejected = logps.split(n, dim=0)

        loss, margin = dpo_loss(pol_chosen, pol_rejected,
                                ref_chosen, ref_rejected, beta)
        loss.backward()
        opt.step()

        if log_every is not None and (step % log_every == 0 or step == steps - 1):
            pair_acc = int(((pol_chosen - pol_rejected)
                            > (ref_chosen - ref_rejected)).sum().item())
            curve.append((step, loss.item(), margin, pair_acc))
    return policy, curve


@torch.no_grad()
def first_gen_token(model, vocab, system, question):
    """greedy 解码的第一个生成 token（答案位置）。"""
    tok2id = {t: i for i, t in enumerate(vocab)}
    prompt = apply_chat_template(system, question)
    ids = [tok2id[t] for t in tokenize(prompt)]
    model.eval()
    x = torch.tensor([ids], dtype=torch.long)
    am = torch.ones_like(x)
    logits = model(x, attention_mask=am)
    return int(logits[0, -1].argmax().item())


@torch.no_grad()
def answer_position_dist(model, vocab, system, question, topk=3):
    """答案决策位（prompt 末位）上的 top-k 分布。"""
    tok2id = {t: i for i, t in enumerate(vocab)}
    prompt = apply_chat_template(system, question)
    ids = torch.tensor([[tok2id[t] for t in tokenize(prompt)]], dtype=torch.long)
    am = torch.ones_like(ids)
    model.eval()
    p = model(ids, attention_mask=am)[0, -1].softmax(-1)
    top = torch.topk(p, topk)
    return [(vocab[int(i)], float(v)) for v, i in zip(top.values, top.indices)]


def evaluate_pairs(model, vocab, pairs):
    """逐对测量：p(chosen)、p(rejected)（序列概率）、win、greedy 答案。
    win/greedy 对应 LLaMA-Factory 训练指标 rewards/accuracies 与生成侧验收
    （train/dpo/trainer.py:L305-308）。"""
    tok2id = {t: i for i, t in enumerate(vocab)}
    (input_ids, attn_mask, labels), _, _ = pairwise_collate(pairs, vocab)
    n = len(pairs)
    logps = seq_logps(model, input_ids, attn_mask, labels)
    pc, pr = logps.split(n, dim=0)
    wins = int((pc > pr).sum().item())
    greedy_ok = 0
    first_tokens = []
    for (sys_, q, c, r) in pairs:
        g = first_gen_token(model, vocab, sys_, q)
        first_tokens.append(g)
        if g == tok2id[c]:
            greedy_ok += 1
    return {
        "win": wins, "n": n, "greedy": greedy_ok,
        "p_chosen": float(pc.exp().mean().item()),
        "p_rejected": float(pr.exp().mean().item()),
        "margin": float((pc - pr).mean().item()),
        "first_tokens": first_tokens,
    }


def drift_kl(policy, ref, vocab, pairs):
    """答案决策位（prompt 末位）上的 KL(policy || ref)，单位 nats。
    度量「策略离开 ref 有多远」——beta 的 KL 缰绳松紧的直接读数。"""
    tok2id = {t: i for i, t in enumerate(vocab)}
    total = 0.0
    for (sys_, q, _c, _r) in pairs:
        prompt = apply_chat_template(sys_, q)
        ids = torch.tensor([[tok2id[t] for t in tokenize(prompt)]], dtype=torch.long)
        am = torch.ones_like(ids)
        policy.eval(); ref.eval()
        with torch.no_grad():
            lp = policy(ids, attention_mask=am)[0, -1].log_softmax(-1)
            lr = ref(ids, attention_mask=am)[0, -1].log_softmax(-1)
        total += float((lp.exp() * (lp - lr)).sum().item())
    return total / len(pairs)


def main():
    t0 = time.perf_counter()
    set_seed(SEED)
    print("=" * 68)
    print("nano-llamafactory L2 — DPO: preference pairs on the same mask")
    print("=" * 68)

    system = "You are a helpful assistant."
    # 单位数加法：chosen = 正确答案，rejected = off-by-one 错答（同样短、同样流畅）。
    # 问题故意两种长度（"What is …?" / "Compute …"），让 pairwise 批真的走 pad。
    pairs = [
        (system, "What is 1+1?", "2", "3"),
        (system, "Compute 2+2", "4", "3"),
        (system, "What is 1+2?", "3", "4"),
        (system, "Compute 2+3", "5", "6"),
        (system, "What is 3+3?", "6", "5"),
        (system, "Compute 1+6", "7", "6"),
    ]
    texts = [apply_chat_template(sys_, q, a) for (sys_, q, a, _r) in pairs] + \
            [apply_chat_template(sys_, q, r) for (sys_, q, _c, r) in pairs]
    vocab = build_vocab(texts)
    tok2id = {t: i for i, t in enumerate(vocab)}
    print(f"vocab size = {len(vocab)}")
    print(f"model params = {sum(p.numel() for p in TinyLM(len(vocab)).parameters()):,}")
    print(f"pairs = {len(pairs)} (chosen = correct sum, rejected = off-by-one)")

    # ---------------- [0] pairwise collate = 两条普通 SFT 行 ----------------
    print("\n[0] pairwise collate: a preference pair = two ordinary SFT rows")
    (batch_ids, batch_am, batch_labels), chosen_rows, rejected_rows = \
        pairwise_collate(pairs, vocab)
    n = len(pairs)
    print(f"    batch rows = {batch_ids.shape[0]} = 2 x {n} "
          f"(first {n} = chosen, last {n} = rejected; collator.py:L564 顺序)")
    sup = [(batch_labels[i] != IGNORE_INDEX).sum().item() for i in range(2 * n)]
    print(f"    supervised tokens per row = {sorted(set(sup))}  "
          f"(answer + \\n + {EOT}; prompt 全 -100)")
    # 连续性机器证明：pairwise 批的每一行（去 pad 后）== L1 式单条 SFT 行
    cont = True
    for i in range(2 * n):
        row = (chosen_rows + rejected_rows)[i]
        ids, am, labs = row
        L = len(ids)
        same = (batch_ids[i, :L].tolist() == ids and
                batch_am[i, :L].tolist() == am and
                batch_labels[i, :L].tolist() == labs and
                batch_am[i, L:].sum().item() == 0 and
                (batch_labels[i, L:] == IGNORE_INDEX).all().item())
        cont = cont and same
    print(f"    every row identical to its L1-style SFT row (pre-pad): {cont}")

    # ---------------- [1] SFT 基线：干净 vs 含噪 ----------------
    print("\n[1] SFT baselines: clean data vs noisy data (rejected also fed as positive)")
    set_seed(SEED)
    model_clean, losses_clean = train_sft(vocab, chosen_rows)
    set_seed(SEED)
    model_noisy, losses_noisy = train_sft(vocab, chosen_rows + rejected_rows)
    print(f"    clean SFT: loss {losses_clean[0]:.4f} -> {losses_clean[-1]:.4f} "
          f"({len(chosen_rows)} rows x {SFT_EPOCHS} epochs)")
    print(f"    noisy SFT: loss {losses_noisy[0]:.4f} -> {losses_noisy[-1]:.4f} "
          f"({len(chosen_rows) + len(rejected_rows)} rows x {SFT_EPOCHS} epochs)")
    ev_clean = evaluate_pairs(model_clean, vocab, pairs)
    ev_noisy = evaluate_pairs(model_noisy, vocab, pairs)
    print("    model        win(p_c>p_r)  greedy  mean_p_chosen  mean_p_rejected")
    print(f"    clean SFT    {ev_clean['win']}/{ev_clean['n']}          "
          f"{ev_clean['greedy']}/{ev_clean['n']}     "
          f"{ev_clean['p_chosen']:.4f}         {ev_clean['p_rejected']:.4f}")
    print(f"    noisy SFT    {ev_noisy['win']}/{ev_noisy['n']}          "
          f"{ev_noisy['greedy']}/{ev_noisy['n']}     "
          f"{ev_noisy['p_chosen']:.4f}         {ev_noisy['p_rejected']:.4f}")

    # ---------------- [2] DPO 从含噪 ref 修复（LF 默认 beta） ----------------
    print(f"\n[2] DPO from noisy ref (beta={BETA_MAIN} = LLaMA-Factory pref_beta default):")
    print("    loss = -log sigmoid(beta * (margin_policy - margin_ref))")
    set_seed(SEED)
    policy, curve = train_dpo(vocab, pairs, model_noisy, BETA_MAIN, log_every=40)
    for (step, loss, margin, pair_acc) in curve:
        print(f"    step {step:3d}: loss={loss:.4f}  margin={margin:+.4f}  "
              f"pair_acc={pair_acc}/{n}")
    ev_dpo = evaluate_pairs(policy, vocab, pairs)
    dpo_drift = drift_kl(policy, model_noisy, vocab, pairs)
    gap = BETA_MAIN * (ev_dpo["margin"] - ev_noisy["margin"])
    print(f"    after DPO:  win={ev_dpo['win']}/{ev_dpo['n']}  "
          f"greedy={ev_dpo['greedy']}/{ev_dpo['n']}  margin={ev_dpo['margin']:+.4f}")
    print(f"    p_rejected: {ev_noisy['p_rejected']:.4f} (noisy ref) -> "
          f"{ev_dpo['p_rejected']:.3g} (DPO)")
    print(f"    drift KL(policy || ref) at answer position = {dpo_drift:.4f} nats")
    print(f"    implicit reward gap beta*(margin_policy-margin_ref) = {gap:+.4f}")

    # ---------------- [3] beta 扫描 + 答案位分布探针 ----------------
    print("\n[3] beta sweep: separation vs drift (each from a fresh copy of ref)")
    sweep = {}
    for beta in BETA_SWEEP:
        if beta == BETA_MAIN:
            ev_b, drift_b, pol_b = ev_dpo, dpo_drift, policy
        else:
            set_seed(SEED)
            pol_b, _ = train_dpo(vocab, pairs, model_noisy, beta)
            ev_b = evaluate_pairs(pol_b, vocab, pairs)
            drift_b = drift_kl(pol_b, model_noisy, vocab, pairs)
        sweep[beta] = (ev_b, drift_b, pol_b)
        print(f"    beta={beta:<4}: margin={ev_b['margin']:+.4f}  drift={drift_b:.4f} nats  "
              f"win={ev_b['win']}/{ev_b['n']}  greedy={ev_b['greedy']}/{ev_b['n']}")
    # win 6/6 与 greedy 崩坏可以同时成立——质量漏去了哪里？探针：
    probe_q = ("Compute 2+2", "4", "3")
    print(f"    answer-position dist for '{probe_q[0]}' "
          f"(chosen '{probe_q[1]}', rejected '{probe_q[2]}'):")
    for label, m in [("noisy ref", model_noisy)] + \
                    [(f"beta={b}", sweep[b][2]) for b in BETA_SWEEP]:
        top = answer_position_dist(m, vocab, system, probe_q[0])
        s = "  ".join(f"'{t if t != chr(10) else chr(92) + 'n'}'={p:.4f}"
                      for t, p in top)
        print(f"        {label:10s}: {s}")
    print("    pair loss 只约束 p(chosen) vs p(rejected)，pair 之外的质量完全自由。")

    # ---------------- [4] 反例：颠倒的偏好对 ----------------
    print("\n[4] counter-example: reversed pairs teach the model to be confidently wrong")
    reversed_pairs = [(sys_, q, r, c) for (sys_, q, c, r) in pairs]  # 字段互换
    set_seed(SEED)
    policy_rev, _ = train_dpo(vocab, reversed_pairs, model_noisy, BETA_MAIN)
    ev_rev = evaluate_pairs(policy_rev, vocab, pairs)  # 仍按正确标准评
    print(f"    reversed-DPO: win={ev_rev['win']}/{ev_rev['n']}  "
          f"greedy={ev_rev['greedy']}/{ev_rev['n']}  "
          f"p_rejected={ev_rev['p_rejected']:.4f}")
    for (sys_, q, c, r), g in zip(pairs[:2], ev_rev["first_tokens"]):
        print(f"    example: '{q}' -> '{vocab[g]}'  (chosen '{c}', rejected '{r}')")

    # ---------------- [5] self-check ----------------
    print("\n" + "=" * 68)
    ev_b05, drift_b05, _ = sweep[BETA_SWEEP[1]]
    ev_b20, drift_b20, _ = sweep[BETA_SWEEP[2]]
    checks = [
        ("pairwise batch = 2n rows, chosen-first", batch_ids.shape[0] == 2 * n),
        ("every pair row == its L1-style SFT row", cont),
        ("each row supervises exactly 3 tokens", sorted(set(sup)) == [3]),
        ("noisy SFT mode-covers: p_rejected substantial", ev_noisy["p_rejected"] >= 0.05),
        ("noisy SFT loses some prompts (win < 6 or greedy < 6)",
         ev_noisy["win"] < n or ev_noisy["greedy"] < n),
        ("clean SFT wins all", ev_clean["win"] == n and ev_clean["greedy"] == n),
        ("DPO(beta=0.1) repairs: win 6/6 and greedy 6/6",
         ev_dpo["win"] == n and ev_dpo["greedy"] == n),
        ("DPO crushes p_rejected (>=10x lower)",
         ev_noisy["p_rejected"] / max(ev_dpo["p_rejected"], 1e-30) >= 10.0),
        ("pair_acc reaches 6/6", curve[-1][3] == n),
        ("implicit reward gap positive", gap > 0.0),
        ("sigmoid saturation: margin(0.1) > margin(0.5) > margin(2.0)",
         ev_dpo["margin"] > ev_b05["margin"] > ev_b20["margin"]),
        ("gentle path drifts less: drift(0.1) < drift(0.5)", dpo_drift < drift_b05),
        ("mass-leak pathology: beta=0.5 wins 6/6 but greedy <= 3",
         ev_b05["win"] == n and ev_b05["greedy"] <= 3),
        ("reversed pairs invert preference (win <= 1)", ev_rev["win"] <= 1),
        ("reversed pairs invert generation (greedy <= 1)", ev_rev["greedy"] <= 1),
    ]
    passed = sum(1 for _name, ok in checks if ok)
    for name, ok in checks:
        assert ok, f"self-check failed: {name}"
    digest_src = (
        f"clean:{ev_clean['win']}/{ev_clean['greedy']}/{ev_clean['p_chosen']:.4f}/"
        f"{ev_clean['p_rejected']:.4f}|"
        f"noisy:{ev_noisy['win']}/{ev_noisy['greedy']}/{ev_noisy['p_chosen']:.4f}/"
        f"{ev_noisy['p_rejected']:.4f}/{ev_noisy['margin']:.4f}|"
        f"dpo:{ev_dpo['win']}/{ev_dpo['greedy']}/{ev_dpo['p_chosen']:.4f}/"
        f"{ev_dpo['p_rejected']:.3g}/{ev_dpo['margin']:.4f}/{dpo_drift:.4f}/{gap:.4f}|"
        + "|".join(f"b{b}:{sweep[b][0]['win']}/{sweep[b][0]['greedy']}/"
                   f"{sweep[b][0]['margin']:.4f}/{sweep[b][1]:.4f}" for b in BETA_SWEEP)
        + f"|rev:{ev_rev['win']}/{ev_rev['greedy']}/{ev_rev['p_rejected']:.4f}"
    )
    digest = hashlib.md5(digest_src.encode()).hexdigest()
    print(f"[self-check] {passed}/{len(checks)} PASS")
    print(f"digest: {digest}")
    print(f"    elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
