#!/usr/bin/env python3
"""L1 — real single-device pretraining loop on top of the L0 lifecycle.

K+1 from L0: L0 proved in pure Python that "full state == same training run".
L1 re-states the same proposition on real PyTorch machinery:

  real: char tokenizer (encode/decode round-trip), block packing with
        document ids, attention-mask + ignore_index boundary policy,
        torch.optim.AdamW (decoupled weight decay), LambdaLR warmup+cosine,
        gradient accumulation + grad clipping, bf16 torch.autocast,
        torch.save/torch.load serialization, torch RNG-state checkpointing.
  toy : corpus scale and vocab, absolute positions (no per-document position
        reset), CPU single-thread baseline (threads=1 for determinism).

Determinism contract: fixed seed + threads=1 + no I/O nondeterminism ->
two runs in fresh empty CWDs must produce byte-identical masked output.

Run:  python3 L1_real_torch_lifecycle.py     (torch CPU, < ~30s)
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import math
import random
import sys
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    sys.exit("this L1 needs torch (CPU build is enough): pip install torch")

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
SEED = 20260818
SEQ_LEN = 16            # input positions per packed block (target = shift by 1)
TOTAL_STEPS = 20        # optimizer steps per full run
SPLIT_STEP = 8          # where the resume checkpoint is taken
ACCUM = 2               # micro-batches per optimizer step
MICRO_BATCH = 4         # packed blocks per micro-batch
WARMUP_STEPS = 3
PEAK_LR = 3e-3
D_MODEL, NHEAD, NLAYERS, FFN, DROPOUT = 32, 2, 2, 64, 0.1
MIXTURE = {"general": 1, "domain": 2}      # same policy as L0
SAMPLER_SEED = 20260813                    # same policy as L0
IGNORE = -100

# Measured 2026-08-18 (dual-channel, bit-identical): md5 of L0 stdout when run
# standalone (`python3 -B L0_pretraining_lifecycle.py`, 36 lines / 1,598 B) ==
# md5 captured in-process by l0_anchor_check() below via redirect_stdout.
# L1 re-runs L0 in-process and requires the same anchor: the cross-level
# invariant is machine-checked, not asserted by prose.
L0_ANCHOR = "b342b389739d1d3c04659b2349f24392"


# --------------------------------------------------------------------------
# tokenizer: char-level, real encode/decode round-trip (toy scale)
# --------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, texts: list[str]):
        chars = sorted({c for t in texts for c in t})
        self.special = ["<bos>", "<eos>"]
        self.itos = self.special + chars
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.bos, self.eos = 0, 1

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode_doc(self, text: str) -> list[int]:
        return [self.bos] + [self.stoi[c] for c in text] + [self.eos]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i >= 2)


def build_corpus(seed: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministic two-domain corpus: combinatorial 'general' grammar and a
    repetitive 'domain' (DNA-like) grammar, so both domains are learnable.
    Returns (train_docs, valid_docs); validation docs are held out."""
    rng = random.Random(seed)
    nouns, verbs = ["cat", "dog", "fox", "owl", "bee"], ["runs", "jumps", "flies", "sits"]
    docs: list[tuple[str, str]] = []
    for _ in range(16):
        parts = [f"the {rng.choice(nouns)} {rng.choice(verbs)}" for _ in range(rng.randint(3, 5))]
        docs.append(("general", " ".join(parts) + " ."))
    bases = ["acgt", "ttca", "ggta", "ccag"]
    for _ in range(8):
        parts = [rng.choice(bases) for _ in range(rng.randint(6, 10))]
        docs.append(("domain", "gene " + " ".join(parts) + " stop"))
    rng_valid = random.Random(seed + 777)
    valid_docs: list[tuple[str, str]] = []
    for _ in range(4):
        parts = [f"the {rng_valid.choice(nouns)} {rng_valid.choice(verbs)}" for _ in range(rng_valid.randint(3, 5))]
        valid_docs.append(("general", " ".join(parts) + " ."))
    for _ in range(2):
        parts = [rng_valid.choice(bases) for _ in range(rng_valid.randint(6, 10))]
        valid_docs.append(("domain", "gene " + " ".join(parts) + " stop"))
    return docs, valid_docs


# --------------------------------------------------------------------------
# sampler + packer: mixture-weighted, per-epoch seeded shuffle, cursor state
# (same state machine as L0: mixture + sampler_seed + epoch + cursor)
# --------------------------------------------------------------------------
def pack_documents(token_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate docs into one stream, cut into overlapping blocks of
    SEQ_LEN+1 tokens, and keep per-position doc ids (the boundary ledger)."""
    stream_ids: list[int] = []
    stream_doc: list[int] = []
    for doc_index, tokens in enumerate(token_lists):
        stream_ids.extend(tokens)
        stream_doc.extend([doc_index] * len(tokens))
    ids = torch.tensor(stream_ids, dtype=torch.long)
    doc = torch.tensor(stream_doc, dtype=torch.long)
    n_blocks = (len(ids) - 1) // SEQ_LEN
    keep = n_blocks * SEQ_LEN + 1
    # block k = stream[k*S : k*S + S + 1] (boundary tokens are shared);
    # doc ids are unfolded with the SAME shape so the label at position
    # S-1 (predicting token S) can still check doc[S-1] == doc[S].
    ids = ids[:keep].unfold(0, SEQ_LEN + 1, SEQ_LEN)
    docb = doc[:keep].unfold(0, SEQ_LEN + 1, SEQ_LEN)
    return ids, docb


class PackedSampler:
    def __init__(self, docs: list[tuple[str, str]], tok: CharTokenizer):
        self.tok = tok
        self.doc_tokens = {domain: [] for domain in MIXTURE}
        for domain, text in docs:
            self.doc_tokens[domain].append(tok.encode_doc(text))
        self.epoch = 0
        self.cursor = 0

    def _epoch_blocks(self) -> tuple[torch.Tensor, torch.Tensor]:
        weighted: list[list[int]] = []
        for domain, weight in MIXTURE.items():
            for tokens in self.doc_tokens[domain]:
                weighted.extend([tokens] * weight)
        random.Random(SAMPLER_SEED + self.epoch).shuffle(weighted)
        return pack_documents(weighted)

    def next_batch(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `count` packed blocks: ids [B, SEQ_LEN+1], doc [B, SEQ_LEN+1]."""
        ids_out, doc_out = [], []
        while len(ids_out) < count:
            ids, docb = self._epoch_blocks()
            remaining = ids.shape[0] - self.cursor
            take = min(count - len(ids_out), remaining)
            ids_out.append(ids[self.cursor : self.cursor + take])
            doc_out.append(docb[self.cursor : self.cursor + take])
            self.cursor += take
            if self.cursor == ids.shape[0]:
                self.epoch += 1
                self.cursor = 0
        return torch.cat(ids_out), torch.cat(doc_out)

    def state(self) -> dict:
        return {"sampler_epoch": self.epoch, "sampler_cursor": self.cursor}

    def load_state(self, state: dict) -> None:
        self.epoch, self.cursor = state["sampler_epoch"], state["sampler_cursor"]


def boundary_labels(ids: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Causal shifted labels with the L0 boundary policy: position t predicts
    t+1 only inside the same document; cross-document targets become IGNORE.
    ids: [B, S+1]; doc: [B, S+1] (doc ids for every token in the block)."""
    target = ids[:, 1:]
    same_doc = doc[:, 1:] == doc[:, :-1]
    labels = torch.where(same_doc, target, torch.full_like(target, IGNORE))
    return labels


def document_attn_mask(doc_input: torch.Tensor) -> torch.Tensor:
    """[B*NHEAD, S, S] bool mask for nn.MultiheadAttention.

    Two contract points of nn.MultiheadAttention (torch 2.13, CPU path):
      1. a 3-D attn_mask must be shaped (batch * num_heads, S, S) — MHA
         flattens batch and heads into one parallel dim before masking
         (torch/nn/functional.py: multi_head_attention_forward shape check,
         `correct_3d_size = (bsz * num_heads, tgt_len, src_len)`);
      2. for a bool mask, True means BLOCKED: _canonical_mask does
         `zeros_like(mask).masked_fill_(mask, float("-inf"))`, the exact
         opposite of F.scaled_dot_product_attention (True = allowed there).
    So we build allowed = causal & same-doc, invert it, then expand over heads.
    doc_input: [B, S] doc ids for the S input positions."""
    bsz, seq = doc_input.shape
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    same = doc_input.unsqueeze(2) == doc_input.unsqueeze(1)
    blocked = ~(causal.unsqueeze(0) & same)                    # [B, S, S]
    return blocked.unsqueeze(1).expand(bsz, NHEAD, seq, seq).reshape(bsz * NHEAD, seq, seq)


# --------------------------------------------------------------------------
# model: tiny GPT with explicit (causal & document) mask — mask IS the policy
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d: int, nhead: int, ffn: int, drop: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d), nn.Dropout(drop)
        )

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=attn_mask, need_weights=False)
        h = h + a
        return h + self.mlp(self.ln2(h))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList(Block(D_MODEL, NHEAD, FFN, DROPOUT) for _ in range(NLAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying (GPT-2 style)

    def forward(self, ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.drop(self.tok(ids) + self.pos(pos))
        for blk in self.blocks:
            h = blk(h, attn_mask)
        return self.head(self.ln_f(h))


# --------------------------------------------------------------------------
# training machinery: real AdamW + LambdaLR + accumulation + clipping
# --------------------------------------------------------------------------
def lr_factor(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_world(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    docs, valid_docs = build_corpus(seed)
    tok = CharTokenizer([t for _, t in docs] + [t for _, t in valid_docs])
    model = TinyGPT(tok.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    sampler = PackedSampler(docs, tok)
    # fixed held-out validation batch: packing it once keeps validation
    # stateless, so it never consumes the training cursor (L0's VALID_DOCS).
    val_ids, val_doc = pack_documents([tok.encode_doc(t) for _, t in valid_docs])
    return model, optimizer, scheduler, sampler, tok, (val_ids, val_doc)


def run_step(model, optimizer, scheduler, sampler, use_amp: bool) -> tuple[float, float, int]:
    model.train()
    loss_total, valid_tokens = 0.0, 0
    optimizer.zero_grad()
    for _ in range(ACCUM):
        ids, doc = sampler.next_batch(MICRO_BATCH)
        labels = boundary_labels(ids, doc)
        mask = document_attn_mask(doc[:, :-1])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids[:, :-1], mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE)
        (loss / ACCUM).backward()
        valid = int((labels != IGNORE).sum())
        loss_total += float(loss.detach()) * valid
        valid_tokens += valid
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    scheduler.step()
    return loss_total / max(1, valid_tokens), optimizer.param_groups[0]["lr"], grad_norm


@torch.no_grad()
def validation_loss(model, val_batch) -> float:
    model.eval()
    ids, doc = val_batch
    labels = boundary_labels(ids, doc)
    mask = document_attn_mask(doc[:, :-1])
    logits = model(ids[:, :-1], mask)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE)
    return float(loss)


# --------------------------------------------------------------------------
# checkpoint: the full state bundle, serialized with real torch.save
# --------------------------------------------------------------------------
def save_bundle(model, optimizer, scheduler, sampler, step: int, consumed_tokens: int) -> bytes:
    buf = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "py_rng": random.getstate(),
            "sampler": sampler.state(),
            "step": step,
            "consumed_tokens": consumed_tokens,
        },
        buf,
    )
    return buf.getvalue()


def load_bundle(raw: bytes, model, optimizer, scheduler, sampler, drop: str = "") -> dict:
    bundle = torch.load(io.BytesIO(raw), weights_only=False)
    model.load_state_dict(bundle["model"])
    if drop == "weights_only":
        # naive resume: keep weights, rebuild everything else from scratch
        sampler.load_state({"sampler_epoch": 0, "sampler_cursor": 0})
        return bundle
    if drop != "optimizer":
        optimizer.load_state_dict(bundle["optimizer"])
    if drop != "scheduler":
        scheduler.load_state_dict(bundle["scheduler"])
    if drop != "rng":
        torch.set_rng_state(bundle["torch_rng"])
        random.setstate(bundle["py_rng"])
    if drop == "cursor":
        sampler.load_state({"sampler_epoch": 0, "sampler_cursor": 0})
    else:
        sampler.load_state(bundle["sampler"])
    return bundle


def train_world(seed: int, until: int, use_amp: bool = False):
    model, optimizer, scheduler, sampler, tok, val_batch = make_world(seed)
    consumed = 0
    history: list[tuple[int, float, float, float]] = []
    for step in range(1, until + 1):
        train_loss, lr, _gnorm = run_step(model, optimizer, scheduler, sampler, use_amp)
        consumed += ACCUM * MICRO_BATCH * SEQ_LEN
        val = validation_loss(model, val_batch)
        history.append((step, train_loss, lr, val))
    return model, optimizer, scheduler, sampler, tok, history, consumed


def max_param_diff(a: nn.Module, b: nn.Module) -> float:
    return max(float((pa - pb).detach().abs().max()) for pa, pb in zip(a.parameters(), b.parameters()))


@torch.no_grad()
def probe_logits_diff(a: nn.Module, b: nn.Module, sampler: PackedSampler) -> float:
    a.eval(), b.eval()
    ids, doc = sampler.next_batch(MICRO_BATCH)
    mask = document_attn_mask(doc[:, :-1])
    la = a(ids[:, :-1], mask)
    lb = b(ids[:, :-1], mask)
    return float((la - lb).abs().max())


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def section_boundary(tok: CharTokenizer) -> tuple[int, int, float, float]:
    """Replay L0's exact 3-doc corpus through L1's mask + ignore_index policy."""
    l0 = load_l0()
    docs = [tokens for _domain, tokens in l0.TRAIN_DOCS]
    stream: list[int] = [t for d in docs for t in d]
    doc_id: list[int] = [i for i, d in enumerate(docs) for _ in d]
    valid = sum(1 for i in range(len(stream) - 1) if doc_id[i] == doc_id[i + 1])
    leaked = len(stream) - 1 - valid

    ids = torch.tensor(stream[:-1]).unsqueeze(0)
    labels = torch.tensor(stream[1:]).unsqueeze(0)
    doc_t = torch.tensor(doc_id[:-1]).unsqueeze(0)
    doc_t_full = torch.tensor(doc_id).unsqueeze(0)
    masked_labels = torch.where(
        doc_t_full[:, 1:] == doc_t_full[:, :-1], labels, torch.full_like(labels, IGNORE)
    )
    torch.manual_seed(SEED)
    model = TinyGPT(tok.vocab_size)
    model.eval()
    with torch.no_grad():
        logits_masked = model(ids, document_attn_mask(doc_t))
        loss_masked = float(
            F.cross_entropy(logits_masked.reshape(-1, logits_masked.size(-1)),
                            masked_labels.reshape(-1), ignore_index=IGNORE)
        )
        seq = ids.shape[1]
        # same MHA contract as document_attn_mask: (B*NHEAD, S, S), True=blocked
        causal_blocked = ~torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        causal_only = causal_blocked.unsqueeze(0).expand(1, NHEAD, seq, seq).reshape(NHEAD, seq, seq)
        logits_naive = model(ids, causal_only)
        loss_naive = float(
            F.cross_entropy(logits_naive.reshape(-1, logits_naive.size(-1)), labels.reshape(-1))
        )
    return valid, leaked, loss_masked, loss_naive


def load_l0():
    path = Path(__file__).resolve().parent / "L0_pretraining_lifecycle.py"
    spec = importlib.util.spec_from_file_location("l0_pretraining_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l0_anchor_check() -> tuple[str, bool]:
    module = load_l0()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        module.main()
    digest = hashlib.md5(captured.getvalue().encode()).hexdigest()
    return digest, digest == L0_ANCHOR


def main() -> None:
    started = time.time()
    torch.set_num_threads(1)  # determinism baseline (slime L1 precedent)
    print("=" * 78)
    print("Pretraining lifecycle L1 — real torch loop: packing, AdamW, AMP, resume")
    print("=" * 78)

    print("\n[0] environment & contract")
    print(f"    torch={torch.__version__} threads={torch.get_num_threads()} seed={SEED} device=cpu")
    print("    real: char tokenizer / block packing / document mask / AdamW / LambdaLR /")
    print("          bf16 autocast / torch.save serialization / RNG-state checkpoint")
    print("    toy : corpus scale, vocab, absolute positions (no per-doc position reset)")

    docs, valid_docs = build_corpus(SEED)
    tok = CharTokenizer([t for _, t in docs] + [t for _, t in valid_docs])
    roundtrip_ok = all(tok.decode(tok.encode_doc(t)) == t for _, t in docs)

    valid, leaked, loss_masked, loss_naive = section_boundary(tok)
    print("\n[1] document boundary: L0 pair filter == L1 mask + ignore_index")
    print(f"    L0 corpus replayed: 3 docs, within-doc targets={valid}, cross-doc leaked={leaked}")
    print(f"    loss(masked policy)={loss_masked:.4f}  loss(naive concat)={loss_naive:.4f}"
          f"  differ={abs(loss_masked - loss_naive) > 1e-9}")
    print(f"    tokenizer round-trip on all {len(docs)} docs: {roundtrip_ok}")

    model, optimizer, scheduler, sampler, tok, history, consumed = train_world(SEED, TOTAL_STEPS)
    initial_val = history[0][3]
    final_val = history[-1][3]
    best_step, best_val = min(history, key=lambda item: item[3])[0], min(history, key=lambda item: item[3])[3]
    print("\n[2] real training run (fp32, AdamW + LambdaLR, accum=2)")
    print("    step  train_loss   lr        val_loss")
    for step, train_loss, lr, val in history:
        if step in (1, 5, 10, 15, 20):
            print(f"    {step:>4}  {train_loss:>10.4f} {lr:>8.5f} {val:>9.4f}")
    print(f"    val: {initial_val:.4f} -> {final_val:.4f}; best checkpoint step={best_step} val={best_val:.4f}")
    print(f"    consumed_tokens={consumed}  sampler=(epoch={sampler.epoch}, cursor={sampler.cursor})")

    # checkpoint at SPLIT_STEP: rebuild an identical world, run to the split
    model_s, opt_s, sch_s, sam_s, tok_s, _vb = make_world(SEED)
    consumed_s = 0
    for step in range(1, SPLIT_STEP + 1):
        run_step(model_s, opt_s, sch_s, sam_s, use_amp=False)
        consumed_s += ACCUM * MICRO_BATCH * SEQ_LEN
    raw_ckpt = save_bundle(model_s, opt_s, sch_s, sam_s, SPLIT_STEP, consumed_s)
    fields = ",".join(sorted(torch.load(io.BytesIO(raw_ckpt), weights_only=False).keys()))

    # uninterrupted reference already trained above (model). Resume from bundle:
    model_r, opt_r, sch_r, sam_r, tok_r, _vb = make_world(SEED)
    load_bundle(raw_ckpt, model_r, opt_r, sch_r, sam_r)
    for step in range(SPLIT_STEP + 1, TOTAL_STEPS + 1):
        run_step(model_r, opt_r, sch_r, sam_r, use_amp=False)
    exact_diff = max_param_diff(model, model_r)
    probe_sampler = PackedSampler(docs, tok)
    probe_diff = probe_logits_diff(model, model_r, probe_sampler)

    print("\n[3] exact resume with the real state bundle (torch.save round-trip)")
    print(f"    checkpoint@step{SPLIT_STEP} fields: {fields}")
    print(f"    uninterrupted == resume: max param diff={exact_diff:.3e}  probe logits diff={probe_diff:.3e}")

    injections = {
        "drop torch RNG state": "rng",
        "drop scheduler state": "scheduler",
        "drop sampler cursor ": "cursor",
        "weights-only resume ": "weights_only",
    }
    inj_diffs: dict[str, float] = {}
    for name, drop in injections.items():
        model_i, opt_i, sch_i, sam_i, tok_i, _vb = make_world(SEED)
        load_bundle(raw_ckpt, model_i, opt_i, sch_i, sam_i, drop=drop)
        for step in range(SPLIT_STEP + 1, TOTAL_STEPS + 1):
            run_step(model_i, opt_i, sch_i, sam_i, use_amp=False)
        inj_diffs[name] = max_param_diff(model, model_i)
    print("    failure injections (same step-8 checkpoint, one component broken):")
    for name, diff in inj_diffs.items():
        print(f"      {name} -> max param diff={diff:.3e}")

    model_b, _o, _s, _sam, _t, history_b, _c = train_world(SEED, TOTAL_STEPS, use_amp=True)
    bf16_val = history_b[-1][3]
    amp_diff = max_param_diff(model, model_b)
    print("\n[4] AMP: bf16 autocast tracks fp32 but is not bit-identical")
    print(f"    fp32 final val={final_val:.4f}  bf16 final val={bf16_val:.4f}  |diff|={abs(final_val - bf16_val):.4f}")
    print(f"    max param diff fp32 vs bf16 = {amp_diff:.3e}  (>0: numerics changed; val still tracks)")
    big = torch.tensor([70000.0]).half()
    subnormal = torch.tensor([1e-6]).half()
    tiny = torch.tensor([1e-8]).half()
    scaled = torch.tensor([1e-6 * 2.0**16])
    print(f"    fp16 range demo: 70000.0 -> {float(big):g} (overflow); 1e-8 -> {float(tiny):g} (flushed to zero)")
    print(f"    subnormal band: 1e-6 -> {float(subnormal):.17g} (still stored, but subnormal: relative precision degrading)")
    print(f"    loss x 2^16 first: 1e-6*65536 = {float(scaled):.4f} finite in fp16 -> unscale in fp32 before step")
    print("    (CPU autocast defaults to bf16: fp32 exponent range, no GradScaler needed here)")

    l0_digest, l0_match = l0_anchor_check()
    print("\n[5] cross-level anchor: L0 invariant re-verified inside L1")
    print(f"    L0 stdout md5={l0_digest}  anchor match={l0_match}")

    lrs = [item[2] for item in history]
    checks = (
        (roundtrip_ok, "tokenizer encode/decode round-trips on every doc"),
        (valid == 9 and leaked == 2, "mask+ignore_index reproduces L0's 9 valid / 2 leaked targets"),
        (abs(loss_masked - loss_naive) > 1e-9, "boundary policy changes the actual objective"),
        (final_val < initial_val, "validation loss improves on this constructed corpus"),
        (lrs[1] > lrs[0], "warmup raises the learning rate"),
        (lrs[-1] < 0.1 * max(lrs), "cosine decay lowers the learning rate"),
        (best_val == min(item[3] for item in history), "best checkpoint selected by validation artifact"),
        (consumed == TOTAL_STEPS * ACCUM * MICRO_BATCH * SEQ_LEN, "consumed_tokens counter closes arithmetically"),
        (exact_diff == 0.0, "full-state resume is bit-for-bit equal (params)"),
        (probe_diff == 0.0, "full-state resume is bit-for-bit equal (probe logits)"),
        (inj_diffs["drop torch RNG state"] > 1e-4, "dropping RNG state (dropout masks) diverges"),
        (inj_diffs["drop scheduler state"] > 1e-4, "dropping scheduler state diverges"),
        (inj_diffs["drop sampler cursor "] > 1e-4, "dropping data cursor diverges"),
        (inj_diffs["weights-only resume "] > 1e-4, "weights-only resume diverges"),
        (abs(bf16_val - final_val) < 0.25, "bf16 autocast tracks fp32 validation loss"),
        (amp_diff > 0.0, "bf16 is not bit-identical to fp32"),
        (math.isinf(float(big)) and float(tiny) == 0.0 and float(subnormal) != 0.0,
         "fp16 overflow / flush-to-zero / subnormal demo holds"),
        (l0_match, "L0 output anchor unchanged (cross-level invariant)"),
    )
    print("\n[6] self-check")
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise AssertionError(f"self-check failed: {failed}")
    print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")

    digest_src = repr(
        (
            round(initial_val, 6), round(final_val, 6), round(best_val, 6), best_step,
            round(amp_diff, 9), round(bf16_val, 6), exact_diff, probe_diff,
            tuple(round(v, 9) for v in inj_diffs.values()), sampler.epoch, sampler.cursor,
        )
    )
    print(f"digest: {hashlib.md5(digest_src.encode()).hexdigest()}")
    print("takeaway: the machinery changed (real tokenizer/AdamW/autocast/torch.save),")
    print("          the invariant did not: same full state == same training run.")
    print(f"    elapsed {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
