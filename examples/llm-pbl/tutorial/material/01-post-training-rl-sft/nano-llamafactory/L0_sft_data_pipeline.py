#!/usr/bin/env python3
"""
nano-llamafactory L0 — SFT 数据侧三件套：chat template / loss mask / collator

零外部依赖（纯标准库），CPU 即跑，输出完全确定。

预训练与 SFT 用的是同一个 next-token prediction loss，唯一的差别是
「哪些位置的 loss 计入」——这件事完全发生在数据侧：
  1. chat template：把 messages 展开成一条模型能读的字符串（训练/推理渲染不同）
  2. loss mask：prompt 位置的 label 置 IGNORE_INDEX(-100)，只在 response 上算 loss
  3. collator：变长样本 pad 成 batch，attention_mask 管注意力，labels 同样遮掉 pad

权威实现对应（github.com/hiyouga/LLaMA-Factory，main 分支行号 2026-08-04 初测、
2026-08-05 复测一致，详见 tutorial_L0.md §溯源；IGNORE_INDEX=-100 与 PyTorch
nn.CrossEntropyLoss 默认 ignore_index 相同：被忽略位置不进 loss、不进梯度）。

toy 口径：tokenizer / 模板 / 「coasting 模型」都是确定性玩具，只为量化 mask 的作用；
真实 tokenizer / Jinja 模板 / 梯度下降在 L1 接入。
"""

import math
import re

IGNORE_INDEX = -100          # PyTorch nn.CrossEntropyLoss 的默认 ignore_index
PAD = "<pad>"
SYS, USR, ASST, EOT = "<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"

# ---------------- [1] chat template ----------------

def apply_chat_template(system, user, assistant=None):
    """ChatML 风格的 toy 模板。assistant=None 即推理用的 generation prompt。"""
    text = f"{SYS}\n{system}\n{EOT}\n{USR}\n{user}\n{EOT}\n{ASST}\n"
    if assistant is not None:
        text += f"{assistant}\n{EOT}"
    return text

# ---------------- tokenizer（word 级玩具版） ----------------

TOKEN_RE = re.compile(r"<\|[\w]+\|>|\n|\S+")

def tokenize(text):
    return TOKEN_RE.findall(text)

def build_vocab(texts):
    vocab = [PAD, SYS, USR, ASST, EOT]
    words = sorted({t for tx in texts for t in tokenize(tx) if t not in vocab and t != "\n"})
    return vocab + words + ["\n"]

# ---------------- [2] labels 构造（loss mask 本体） ----------------

def build_labels(prompt_ids, response_ids):
    """HF/LlamaFactory 约定（对齐 supervised.py:L109 注释）：
    input_ids = X Y，labels = <ignore>...<ignore> Y，真实训练时 logits[i] 对 labels[i+1]。"""
    return [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)

# ---------------- [3] coasting 模型（确定性 bigram） ----------------

def fit_bigram(corpus, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    cnt = [[0] * len(vocab) for _ in vocab]
    toks = tokenize(corpus)
    for a, b in zip(toks, toks[1:]):
        cnt[idx[a]][idx[b]] += 1
    return cnt, idx

def nll(cnt, idx, ids):
    """每个位置的 next-token 负对数似然（bit），加性平滑。"""
    V, out = len(idx), []
    for a, b in zip(ids, ids[1:]):
        row, total = cnt[a], sum(cnt[a])
        out.append(-math.log2((row[b] + 1e-3) / (total + 1e-3 * V)))
    return out

def mean(xs):
    return sum(xs) / len(xs)

# ---------------- [4] collator ----------------

def collate(samples, vocab):
    """samples: (input_ids, labels)。右 pad；attention_mask 遮 pad 的注意力，labels 遮 pad 的 loss。"""
    pid = vocab.index(PAD)
    L = max(len(ids) for ids, _ in samples)
    batch = []
    for ids, labels in samples:
        n = L - len(ids)
        batch.append((ids + [pid] * n,
                      [1] * len(ids) + [0] * n,
                      labels + [IGNORE_INDEX] * n))
    return batch

def masked_mean_nll(per_token_nll, labels):
    """HF 语义：loss 只在非 ignore 的 label 上取平均（shift 一位对齐后）。"""
    vals = [v for v, lab in zip(per_token_nll, labels[1:]) if lab != IGNORE_INDEX]
    return sum(vals) / len(vals), len(vals)

# ---------------- [5] 反例 ----------------

def build_labels_off_by_one(prompt_ids, response_ids):
    """遮罩边界多退一格：logits 从「最后一个 response token」才开始监督。"""
    return [IGNORE_INDEX] * (len(prompt_ids) + 1) + list(response_ids[1:])

def build_labels_leak_pad(input_ids):
    """预训练式：labels = input_ids（pad 也进 loss）。"""
    return list(input_ids)

# ---------------- 实验 ----------------

def main():
    print("=" * 64)
    print("nano-llamafactory L0 — SFT 数据侧：template / loss mask / collator")
    print("=" * 64)

    SYSTEM = "You are a helpful assistant."
    raw = [("What is the capital of France?", "The capital of France is Paris."),
           ("How many hours are in a day?", "There are 24 hours in a day."),
           ("Who wrote Hamlet?", "Hamlet was written by Shakespeare.")]
    full_texts = [apply_chat_template(SYSTEM, q, a) for q, a in raw]
    vocab = build_vocab(full_texts)
    tok2id = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    # [1] template：训练渲染 vs 推理渲染（generation prompt）
    print("\n[1] chat template：同一份 messages，训练/推理各渲染一次")
    for (q, a), full in zip(raw, full_texts):
        prompt_text = apply_chat_template(SYSTEM, q)          # add_generation_prompt=True
        assert prompt_text == full[:len(prompt_text)]         # 推理 prompt 必须是训练串的真前缀
        print(f"    训练全文 {len(tokenize(full)):2d} tok | 推理 prompt {len(tokenize(prompt_text)):2d} tok | Q: {q}")
    print(f"    推理时模型只见到左前缀（到 {ASST}\\n 为止），response 由它自己续写")

    # [2] labels：shift 配对与 mask 边界
    print("\n[2] labels 构造：prompt 全遮（-100），response 进 loss（含结束符）")
    samples = []
    for (q, a), full in zip(raw, full_texts):
        ids = [tok2id[t] for t in tokenize(full)]
        prompt_ids = [tok2id[t] for t in tokenize(apply_chat_template(SYSTEM, q))]
        assert ids[:len(prompt_ids)] == prompt_ids            # 与 [1] 的前缀不变量同源
        samples.append((ids, build_labels(prompt_ids, ids[len(prompt_ids):])))
    ids0, labels0 = samples[0]
    P = sum(1 for l in labels0 if l == IGNORE_INDEX)
    R = len(labels0) - P
    k = P - 1  # 边界位置：logits[k] 预测 labels[k+1] = 第一个 response token
    print(f"    例 1: input_ids={len(ids0)} tok | IGNORE={P}（prompt）| 监督={R}（response+{EOT}）")
    print(f"    边界: logits[{k}]（token {vocab[ids0[k]]!r}）预测 labels[{k + 1}] = {vocab[labels0[k + 1]]!r}")
    print(f"          —— 第一个 response token 由「最后一个 prompt 位置」的 logits 监督")
    assert labels0[k] == IGNORE_INDEX and labels0[k + 1] != IGNORE_INDEX
    assert labels0[-1] != IGNORE_INDEX   # 结束符也进 loss：模型要学会「何时停」

    # [3] mask 的作用量化：coasting bigram（只见过 prompt，从没见过 response）
    print("\n[3] loss mask 的作用：coasting 模型下，unmasked 的低 loss 是假的")
    boilerplate = " ".join(apply_chat_template(SYSTEM, q) for q, _ in raw) * 10
    cnt, idx = fit_bigram(boilerplate, vocab)
    s_all, s_prompt, s_resp = [], [], []
    n_tok_prompt = n_tok_all = 0
    for ids, labels in samples:
        P_i = sum(1 for l in labels if l == IGNORE_INDEX)
        n_tok_prompt += P_i; n_tok_all += len(ids)
        for pos, v in enumerate(nll(cnt, idx, ids)):   # logits[pos] 预测 ids[pos+1]
            s_all.append(v)
            (s_prompt if pos + 1 < P_i else s_resp).append(v)
    assert len(s_all) == len(s_prompt) + len(s_resp)
    print(f"    unmasked（预训练式，{len(s_all)} 位置）平均 NLL = {mean(s_all):.4f} bit")
    print(f"    masked  （SFT，仅 {len(s_resp)} 个 response 位置）平均 NLL = {mean(s_resp):.4f} bit")
    print(f"    loss 总和构成: prompt 区 {sum(s_prompt):7.2f} bit ({sum(s_prompt) / sum(s_all) * 100:.0f}%)"
          f" | response 区 {sum(s_resp):7.2f} bit ({sum(s_resp) / sum(s_all) * 100:.0f}%)")
    print(f"    token 占比: prompt 区 {n_tok_prompt}/{n_tok_all} = {n_tok_prompt / n_tok_all * 100:.0f}%")
    assert mean(s_resp) > 2 * mean(s_all), "mask 后平均 NLL 应显著更高（低 loss 是模板刷出来的）"

    # [4] collator：pad + attention_mask + labels mask；batch loss 与逐样本 pooled 一致
    print("\n[4] collator：变长样本 pad 成 batch（pad 同时被两层遮罩）")
    batch = collate(samples, vocab)
    rows = ["input_ids".ljust(12), "attn_mask".ljust(12), "labels".ljust(12)]
    for ids, am, labs in batch:
        rows[0] += " " + " ".join(str(x).rjust(2) for x in ids)
        rows[1] += " " + " ".join(str(x).rjust(2) for x in am)
        rows[2] += " " + " ".join((".." if l == IGNORE_INDEX else str(l)).rjust(2) for l in labs)
    for r in rows:
        print("    " + r)
    per_sample = []
    for ids, _, labs in batch:
        m, n = masked_mean_nll(nll(cnt, idx, ids), labs)
        per_sample.append((m, n))
    batch_all = [v for ids, _, labs in batch for v, lab in zip(nll(cnt, idx, ids), labs[1:]) if lab != IGNORE_INDEX]
    pooled = sum(v for v in batch_all) / len(batch_all)
    weighted = sum(m * n for m, n in per_sample) / sum(n for _, n in per_sample)
    print(f"    batch loss（token 级 pooled）= {pooled:.4f} bit | 逐样本加权平均 = {weighted:.4f} bit | 有效 token = {len(batch_all)}")
    assert abs(pooled - weighted) < 1e-9, "正确 collate 不应改变 loss"

    # [5] 反例
    print("\n[5] 反例：两种常见遮罩错误")
    m_leak = masked_mean_nll(nll(cnt, idx, batch[0][0]), build_labels_leak_pad(batch[0][0]))[0]
    print(f"    a) pad 漏进 labels（labels=input_ids）: loss {pooled:.4f} -> {m_leak:.4f} bit"
          f"（loss 反而变低是假象：pad 可预测、稀释了均值，模型还在学续写 {PAD}）")
    assert m_leak != pooled
    labs_bad = build_labels_off_by_one([tok2id[t] for t in tokenize(apply_chat_template(SYSTEM, raw[0][0]))],
                                       [tok2id[t] for t in tokenize(raw[0][1] + f"\n{EOT}")])
    lost = [i for i, (a, b) in enumerate(zip(labels0, labs_bad)) if a != b]
    print(f"    b) mask 边界多退一格: labels 位置 {lost} 的监督丢失 —— 丢的正是第一个 response token"
          f" {vocab[labels0[lost[0]]]!r}（答案开头没人教）")
    assert lost == [k + 1]   # k 是 logits 空间（预测端），lost 是 labels 空间（被预测端），差 shift 的一位

    print("\n" + "=" * 64)
    print("✅ self-check passed: 模板前缀 / mask 边界 / 稀释方向 / batch loss 不变量 / 两种反例")
    print("=" * 64)
    print(f"\ntakeaway: SFT 与预训练共用同一个 next-token loss，差别只在 labels 的遮罩：\n"
          f"          模板定边界（{ASST}\\n 之后才算 response）、-100 定 loss、collator 定 batch。\n"
          f"          vocab={V}，全部数字由确定性 toy 模型现场算出。")

if __name__ == "__main__":
    main()
