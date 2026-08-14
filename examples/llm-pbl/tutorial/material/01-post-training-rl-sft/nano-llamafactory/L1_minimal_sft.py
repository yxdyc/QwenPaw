#!/usr/bin/env python3
"""
L1_minimal_sft.py — nano-llamafactory L1

真实小模型（TransformerEncoder，约 75K 参数）上的最小 SFT 循环：
    复用 L0 的 chat template / loss mask / collator，把 coasting bigram
    换成真实 torch 梯度下降，验证「labels 遮罩决定模型学到什么」。

依赖：torch（CPU/MPS/GPU 均可；本机用 CPU 即可复现）。
对应真实系统：LLaMA-Factory 的 SFT pipeline。
"""

import random
import re

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise SystemExit(
        "[error] torch is required to run this L1 script.\n"
        "        Install it with: pip install torch\n"
        "        (CPU/MPS/GPU all work; no transformers needed.)"
    ) from e


SEED = 42
PAD = "<pad>"
SYS, USR, ASST, EOT = "<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"
IGNORE_INDEX = -100

TOKEN_RE = re.compile(r"<\|[\w]+\|>|\n|\S+")

# ---------- 超参数：模型小到 CPU 秒开，但足够展示 SFT ----------
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
MAX_LEN = 64
LR = 5e-3
EPOCHS = 400


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text):
    return TOKEN_RE.findall(text)


def apply_chat_template(system, user, assistant=None):
    """ChatML 风格 toy 模板；assistant=None 即推理用的 generation prompt。"""
    text = f"{SYS}\n{system}\n{EOT}\n{USR}\n{user}\n{EOT}\n{ASST}\n"
    if assistant is not None:
        text += f"{assistant}\n{EOT}"
    return text


def build_vocab(texts):
    specials = [PAD, SYS, USR, ASST, EOT]
    words = sorted({t for tx in texts for t in tokenize(tx)
                    if t not in specials and t != "\n"})
    vocab = specials + words + ["\n"]
    return vocab


def build_labels(prompt_ids, response_ids):
    """HF/LlamaFactory 约定：prompt 全 -100，response（含 eos）进 loss。"""
    return [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)


def build_labels_off_by_one(prompt_ids, response_ids):
    """遮罩边界多退一格：第一个 response token 没人教（复现 L0 反例 b）。"""
    return [IGNORE_INDEX] * (len(prompt_ids) + 1) + list(response_ids[1:])


def collate(samples, vocab):
    """右 pad；attention_mask 管前向，labels 管反向。"""
    pid = vocab.index(PAD)
    L = max(len(ids) for ids, _ in samples)
    batch_ids, batch_am, batch_labels = [], [], []
    for ids, labels in samples:
        n = L - len(ids)
        batch_ids.append(ids + [pid] * n)
        batch_am.append([1] * len(ids) + [0] * n)
        batch_labels.append(labels + [IGNORE_INDEX] * n)
    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(batch_am, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )


class TinyLM(nn.Module):
    """极小 TransformerLM：embedding + 可学习位置编码 + causal TransformerEncoder + head。"""

    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 max_len=MAX_LEN):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        # key_padding_mask: True 表示该位置是 pad，需要忽略
        key_mask = None
        if attention_mask is not None:
            key_mask = attention_mask == 0
        # 手画因果 mask：True 表示该位置被遮住（只能看到 <= i）
        causal_mask = torch.triu(
            torch.ones((L, L), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        h = self.blocks(h, mask=causal_mask, src_key_padding_mask=key_mask)
        return self.head(self.norm(h))


def compute_loss(logits, labels):
    """shifted CE：logits[:, i, :] 预测 labels[:, i + 1]。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def generate(model, prompt_ids, eos_id, max_new_tokens=20):
    """greedy 解码，返回完整 token id 列表（含 prompt）。"""
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long)
        am = torch.ones_like(x)
        logits = model(x, attention_mask=am)
        nxt = int(logits[0, -1].argmax().item())
        ids.append(nxt)
        if nxt == eos_id:
            break
    return ids


def ids_to_text(ids, vocab):
    return "".join(vocab[i] for i in ids)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_model(vocab, samples, labels_mode="masked", epochs=EPOCHS):
    """
    labels_mode:
      - "masked": 正常 SFT labels（prompt -100，response 进 loss）
      - "off_by_one": 边界多退一格，第一个 response token 被遮掉
    """
    collated = collate(samples, vocab)
    input_ids, attn_mask, labels = collated

    if labels_mode == "off_by_one":
        # 逐个样本重造 off-by-one labels（保持 pad 仍 -100）
        L = max(len(ids) for ids, _ in samples)
        new_labels = []
        for (ids, labs) in samples:
            P_i = sum(1 for x in labs if x == IGNORE_INDEX)
            prompt_ids = ids[:P_i]
            response_ids = ids[P_i:]
            off = build_labels_off_by_one(prompt_ids, response_ids)
            new_labels.append(off + [IGNORE_INDEX] * (L - len(off)))
        labels = torch.tensor(new_labels, dtype=torch.long)

    model = TinyLM(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attn_mask)
        loss = compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses


def evaluate_generations(model, vocab, raw, system):
    tok2id = {t: i for i, t in enumerate(vocab)}
    eos_id = tok2id[EOT]
    results = []
    for q, a in raw:
        prompt_text = apply_chat_template(system, q)
        prompt_ids = [tok2id[t] for t in tokenize(prompt_text)]
        gen_ids = generate(model, prompt_ids, eos_id)
        gen_text = ids_to_text(gen_ids[len(prompt_ids):], vocab)
        # tokenizer 是无空格的 word 级切分，比较时也用无空格版本
        expected = ids_to_text([tok2id[t] for t in tokenize(a)], vocab)
        results.append((q, expected, gen_text.strip()))
    return results


def print_generations(label, results):
    print(f"    {label}:")
    for q, expected, gen in results:
        ok = expected in gen
        print(f"      Q: {q}\n         -> {gen!r}  {'✅' if ok else '❌'}")


def main():
    set_seed(SEED)
    print("=" * 64)
    print("nano-llamafactory L1 — real torch SFT on a tiny Transformer")
    print("=" * 64)

    system = "You are a helpful assistant."
    raw = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("How many hours are in a day?", "There are 24 hours in a day."),
        ("Who wrote Hamlet?", "Hamlet was written by Shakespeare."),
    ]
    full_texts = [apply_chat_template(system, q, a) for q, a in raw]
    vocab = build_vocab(full_texts)
    tok2id = {t: i for i, t in enumerate(vocab)}
    print(f"vocab size = {len(vocab)}")

    ref_model = TinyLM(len(vocab))
    print(f"model params = {count_params(ref_model):,}")

    # [1] 构造 (input_ids, labels)，与 L0 完全一致
    samples = []
    for (q, a), full in zip(raw, full_texts):
        ids = [tok2id[t] for t in tokenize(full)]
        prompt_ids = [tok2id[t] for t in tokenize(apply_chat_template(system, q))]
        assert ids[:len(prompt_ids)] == prompt_ids
        samples.append((ids, build_labels(prompt_ids, ids[len(prompt_ids):])))

    print(f"\n[1] data: per-sample token counts (prompt ignored / response+{EOT} supervised)")
    for i, (ids, labs) in enumerate(samples):
        P_i = sum(1 for l in labs if l == IGNORE_INDEX)
        R_i = len(labs) - P_i
        print(f"    sample {i}: prompt {P_i} tok / response+{EOT} {R_i} tok")

    # [2] masked SFT：训练前、后生成对比
    print("\n[2] masked SFT (prompt ignored, response supervised)")
    model_masked, losses_masked = train_model(vocab, samples, labels_mode="masked")
    print(f"    initial loss = {losses_masked[0]:.4f}  ->  final loss = {losses_masked[-1]:.4f}")

    before = evaluate_generations(ref_model, vocab, raw, system)
    after = evaluate_generations(model_masked, vocab, raw, system)
    print_generations("before training", before)
    print_generations("after training", after)

    # [3] 反例：mask 边界多退一格，第一个 response token 没人教
    print("\n[3] ablation: mask boundary off-by-one")
    model_oo, losses_oo = train_model(vocab, samples, labels_mode="off_by_one")
    print(f"    final loss = {losses_oo[-1]:.4f}  (masked final = {losses_masked[-1]:.4f})")
    off_by_one = evaluate_generations(model_oo, vocab, raw, system)
    print_generations("off-by-one generation", off_by_one)

    first_expected_texts = [tokenize(a)[0] for _, a in raw]
    print(f"    first expected response tokens = {first_expected_texts}")

    # [4] self-check
    print("\n" + "=" * 64)
    assert losses_masked[-1] < losses_masked[0] / 2, "masked loss should drop significantly"
    assert losses_masked[-1] < 0.5, "masked loss should converge"
    for q, expected, gen in after:
        assert expected in gen, f"masked model failed to learn: {q} -> {gen!r}"
    # off-by-one 模型至少丢失第一个 response token
    for (_, _, gen), first in zip(off_by_one, first_expected_texts):
        assert not gen.startswith(first), f"off-by-one should not start with {first!r}, got {gen!r}"
    print("✅ self-check passed: masked loss drops / answers emerge / off-by-one drops first token")


if __name__ == "__main__":
    main()
