#!/usr/bin/env python3
"""GPU empirical verification for nano-pretraining-loop L1.

Targets the [TODO: verify on real system] notes in tutorial_L1.md:
- real GPU bf16 autocast drift vs fp32,
- exact resume on a real GPU (single device) and why CUDA RNG state matters,
- DataLoader with num_workers changes determinism.

Run on a CUDA machine:
    python3 L1_gpu_verify.py
"""
from __future__ import annotations

import hashlib
import io
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import L1 module as a library.
L1_PATH = Path(__file__).resolve().parent / "L1_real_torch_lifecycle.py"
if not L1_PATH.exists():
    raise FileNotFoundError(f"L1 not found at {L1_PATH}")

import importlib.util
spec = importlib.util.spec_from_file_location("l1", L1_PATH)
l1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l1)

# Re-use L1 hyperparameters.
SEED = l1.SEED
SEQ_LEN = l1.SEQ_LEN
TOTAL_STEPS = l1.TOTAL_STEPS
SPLIT_STEP = l1.SPLIT_STEP
ACCUM = l1.ACCUM
MICRO_BATCH = l1.MICRO_BATCH
WARMUP_STEPS = l1.WARMUP_STEPS
PEAK_LR = l1.PEAK_LR
D_MODEL, NHEAD, NLAYERS, FFN, DROPOUT = l1.D_MODEL, l1.NHEAD, l1.NLAYERS, l1.FFN, l1.DROPOUT
IGNORE = l1.IGNORE

CharTokenizer = l1.CharTokenizer
build_corpus = l1.build_corpus
pack_documents = l1.pack_documents
PackedSampler = l1.PackedSampler
boundary_labels = l1.boundary_labels


def _document_attn_mask_gpu(doc_input: torch.Tensor) -> torch.Tensor:
    """GPU-aware wrapper for L1 document_attn_mask: causal follows doc_input device."""
    bsz, seq = doc_input.shape
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=doc_input.device))
    same = doc_input.unsqueeze(2) == doc_input.unsqueeze(1)
    blocked = ~(causal.unsqueeze(0) & same)
    return blocked.unsqueeze(1).expand(bsz, l1.NHEAD, seq, seq).reshape(bsz * l1.NHEAD, seq, seq)


l1.document_attn_mask = _document_attn_mask_gpu
document_attn_mask = l1.document_attn_mask
Block = l1.Block
TinyGPT = l1.TinyGPT
lr_factor = l1.lr_factor
make_world = l1.make_world
validation_loss = l1.validation_loss
max_param_diff = l1.max_param_diff
save_bundle = l1.save_bundle
load_bundle = l1.load_bundle


def probe_logits_diff_gpu(a: nn.Module, b: nn.Module, sampler: PackedSampler, device: torch.device) -> float:
    a.eval(), b.eval()
    ids, doc = sampler.next_batch(MICRO_BATCH)
    ids, doc = ids.to(device), doc.to(device)
    mask = document_attn_mask(doc[:, :-1])
    with torch.no_grad():
        la = a(ids[:, :-1], mask)
        lb = b(ids[:, :-1], mask)
    return float((la - lb).abs().max().detach())


def run_step_gpu(model, optimizer, scheduler, sampler, device: torch.device, use_amp: bool) -> tuple[float, float, int]:
    model.train()
    loss_total, valid_tokens = 0.0, 0
    optimizer.zero_grad()
    for _ in range(ACCUM):
        ids, doc = sampler.next_batch(MICRO_BATCH)
        ids, doc = ids.to(device), doc.to(device)
        labels = boundary_labels(ids, doc)
        mask = document_attn_mask(doc[:, :-1])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
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


def train_world_gpu(seed: int, until: int, device: torch.device, use_amp: bool = False):
    torch.manual_seed(seed)
    random.seed(seed)
    docs, valid_docs = build_corpus(seed)
    tok = CharTokenizer([t for _, t in docs] + [t for _, t in valid_docs])
    model = TinyGPT(tok.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    sampler = PackedSampler(docs, tok)
    val_ids, val_doc = pack_documents([tok.encode_doc(t) for _, t in valid_docs])
    val_batch = (val_ids.to(device), val_doc.to(device))

    consumed = 0
    history = []
    for step in range(1, until + 1):
        train_loss, lr, _ = run_step_gpu(model, optimizer, scheduler, sampler, device, use_amp)
        consumed += ACCUM * MICRO_BATCH * SEQ_LEN
        val = validation_loss(model, val_batch)
        history.append((step, train_loss, lr, val))
    return model, optimizer, scheduler, sampler, tok, history, consumed


def save_bundle_gpu(model, optimizer, scheduler, sampler, step: int, consumed_tokens: int) -> bytes:
    """Extend L1 bundle with CUDA RNG state — this is the extra field GPU training needs."""
    buf = io.BytesIO()
    bundle = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "py_rng": random.getstate(),
        "sampler": sampler.state(),
        "step": step,
        "consumed_tokens": consumed_tokens,
    }
    torch.save(bundle, buf)
    return buf.getvalue()


def load_bundle_gpu(raw: bytes, model, optimizer, scheduler, sampler, drop: str = "") -> dict:
    bundle = torch.load(io.BytesIO(raw), weights_only=False)
    model.load_state_dict(bundle["model"])
    if drop != "weights_only":
        if drop != "optimizer":
            optimizer.load_state_dict(bundle["optimizer"])
        if drop != "scheduler":
            scheduler.load_state_dict(bundle["scheduler"])
        if drop != "rng":
            torch.set_rng_state(bundle["torch_rng"])
            random.setstate(bundle["py_rng"])
            if "cuda_rng" in bundle and drop != "cuda_rng":
                torch.cuda.set_rng_state(bundle["cuda_rng"])
    if drop == "weights_only":
        sampler.load_state({"sampler_epoch": 0, "sampler_cursor": 0})
    elif drop == "cursor":
        sampler.load_state({"sampler_epoch": 0, "sampler_cursor": 0})
    else:
        sampler.load_state(bundle["sampler"])
    return bundle


def make_world_gpu(seed: int, device: torch.device):
    torch.manual_seed(seed)
    random.seed(seed)
    docs, valid_docs = build_corpus(seed)
    tok = CharTokenizer([t for _, t in docs] + [t for _, t in valid_docs])
    model = TinyGPT(tok.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    sampler = PackedSampler(docs, tok)
    val_ids, val_doc = pack_documents([tok.encode_doc(t) for _, t in valid_docs])
    return model, optimizer, scheduler, sampler, tok, (val_ids.to(device), val_doc.to(device))


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available; this verification must run on a GPU.")

    device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(SEED)
    print("=" * 78)
    print("nano-pretraining-loop L1 — GPU empirical verification")
    print("=" * 78)
    print(f"[0] environment: torch={torch.__version__} device={device} "
          f"name={torch.cuda.get_device_name(device)}")

    # Baseline: fp32 on GPU.
    model_fp32, _, _, _, _, history_fp32, _ = train_world_gpu(SEED, TOTAL_STEPS, device, use_amp=False)
    fp32_final_val = history_fp32[-1][3]
    print(f"\n[1] fp32 GPU final val_loss={fp32_final_val:.4f}")

    # bf16 autocast on GPU.
    model_bf16, _, _, _, _, history_bf16, _ = train_world_gpu(SEED, TOTAL_STEPS, device, use_amp=True)
    bf16_final_val = history_bf16[-1][3]

    amp_diff = max_param_diff(model_fp32, model_bf16)
    print(f"\n[2] bf16 autocast on GPU")
    print(f"    fp32 final val={fp32_final_val:.4f}  bf16 final val={bf16_final_val:.4f}  |diff|={abs(bf16_final_val - fp32_final_val):.4f}")
    print(f"    max param diff fp32 vs bf16 = {amp_diff:.3e}")

    # Exact resume on GPU: need CUDA RNG in bundle.
    model_s, opt_s, sch_s, sam_s, tok_s, _vb = make_world_gpu(SEED, device)
    consumed_s = 0
    for step in range(1, SPLIT_STEP + 1):
        run_step_gpu(model_s, opt_s, sch_s, sam_s, device, use_amp=False)
        consumed_s += ACCUM * MICRO_BATCH * SEQ_LEN
    raw_ckpt = save_bundle_gpu(model_s, opt_s, sch_s, sam_s, SPLIT_STEP, consumed_s)

    model_r, opt_r, sch_r, sam_r, tok_r, _vb = make_world_gpu(SEED, device)
    load_bundle_gpu(raw_ckpt, model_r, opt_r, sch_r, sam_r)
    for step in range(SPLIT_STEP + 1, TOTAL_STEPS + 1):
        run_step_gpu(model_r, opt_r, sch_r, sam_r, device, use_amp=False)
    resume_diff = max_param_diff(model_fp32, model_r)
    probe_sampler = PackedSampler(build_corpus(SEED)[0], tok_r)
    probe_diff_gpu = probe_logits_diff_gpu(model_fp32, model_r, probe_sampler, device)
    print(f"\n[3] single-GPU exact resume (bundle includes cuda_rng)")
    print(f"    max param diff={resume_diff:.3e}  probe logits diff={probe_diff_gpu:.3e}")
    print("    (GPU kernels are non-deterministic; diff==0 is not promised, magnitude is the signal.)")

    # Without cuda_rng: load bundle but skip CUDA RNG state.
    model_nr, opt_nr, sch_nr, sam_nr, tok_nr, _vb = make_world_gpu(SEED, device)
    load_bundle_gpu(raw_ckpt, model_nr, opt_nr, sch_nr, sam_nr, drop="cuda_rng")
    for step in range(SPLIT_STEP + 1, TOTAL_STEPS + 1):
        run_step_gpu(model_nr, opt_nr, sch_nr, sam_nr, device, use_amp=False)
    no_cuda_rng_diff = max_param_diff(model_fp32, model_nr)
    print(f"\n[4] drop cuda_rng state")
    print(f"    max param diff={no_cuda_rng_diff:.3e}")

    # DataLoader with num_workers: determinism contract changes.
    from torch.utils.data import Dataset, DataLoader
    class TinyDataset(Dataset):
        def __init__(self, n: int):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return idx * 2

    loader = DataLoader(TinyDataset(8), batch_size=2, num_workers=2, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    worker_order_a = [batch.tolist() for batch in loader]
    loader = DataLoader(TinyDataset(8), batch_size=2, num_workers=2, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    worker_order_b = [batch.tolist() for batch in loader]
    print(f"\n[5] DataLoader num_workers=2 reproducibility")
    print(f"    same seed yields identical batches: {worker_order_a == worker_order_b}")
    print("    (worker spawn order is OS-dependent; identical seed usually reproduces but is not guaranteed cross-run.)")

    checks = (
        (abs(bf16_final_val - fp32_final_val) < 0.25, "bf16 GPU tracks fp32 GPU val loss"),
        (amp_diff > 0.0, "bf16 GPU is not bit-identical to fp32 GPU"),
        (resume_diff < 1e-3, "single-GPU resume with cuda_rng stays within tolerance"),
        (no_cuda_rng_diff > resume_diff, "dropping cuda_rng state increases drift"),
        (worker_order_a == worker_order_b, "DataLoader with fixed generator seed reproduces batches"),
    )
    print("\n[6] self-check")
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise AssertionError(f"GPU verification failed: {failed}")
    print(f"\nGPU SELF-CHECK: {len(checks)}/{len(checks)} PASS")

    digest_src = repr((
        round(fp32_final_val, 6), round(bf16_final_val, 6), round(amp_diff, 9),
        round(resume_diff, 9), round(probe_diff_gpu, 9), round(no_cuda_rng_diff, 9),
        worker_order_a == worker_order_b,
    ))
    print(f"gpu_digest: {hashlib.md5(digest_src.encode()).hexdigest()}")
    print(f"    elapsed {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
