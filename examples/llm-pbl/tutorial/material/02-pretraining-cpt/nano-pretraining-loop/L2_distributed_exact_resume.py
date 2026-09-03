#!/usr/bin/env python3
"""L2 — distributed exact resume: data-parallel loop, rank-local state, recovery.

K+1 from L1: L1 proved "same full state == same training run" on a single device.
L2 extends the same proposition to a distributed data-parallel loop:

  real: torch.distributed (gloo, CPU), per-rank data partition, manual
        all-reduce gradient averaging, rank-local checkpoint files.
  toy : world_size=2, tiny model, loopback interconnect, no real GPU.

Run:  python3 -B L2_distributed_exact_resume.py     (torch CPU, < ~60s)
"""
from __future__ import annotations

import hashlib
import argparse
import importlib.util
import json
import os
import random
import socket
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

# Import L1 machinery as a library.
L1_PATH = Path(__file__).resolve().parent / "L1_real_torch_lifecycle.py"
if not L1_PATH.exists():  # pragma: no cover
    raise FileNotFoundError(f"L1 not found at {L1_PATH}")
spec = importlib.util.spec_from_file_location("l1", L1_PATH)
l1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l1)

SEED = l1.SEED
SEQ_LEN = l1.SEQ_LEN
TOTAL_STEPS = l1.TOTAL_STEPS
SPLIT_STEP = 8
ROLLBACK_STEP = 10
SPIKE_STEP = 12
ACCUM = l1.ACCUM
MICRO_BATCH = l1.MICRO_BATCH
WARMUP_STEPS = l1.WARMUP_STEPS
PEAK_LR = l1.PEAK_LR
D_MODEL, NHEAD, NLAYERS, FFN, DROPOUT = l1.D_MODEL, l1.NHEAD, l1.NLAYERS, l1.FFN, l1.DROPOUT
IGNORE = l1.IGNORE
MIXTURE = l1.MIXTURE
SAMPLER_SEED = l1.SAMPLER_SEED

CharTokenizer = l1.CharTokenizer
build_corpus = l1.build_corpus
pack_documents = l1.pack_documents
boundary_labels = l1.boundary_labels
document_attn_mask = l1.document_attn_mask
TinyGPT = l1.TinyGPT
lr_factor = l1.lr_factor
validation_loss = l1.validation_loss

WORLD_SIZE = 2


# --------------------------------------------------------------------------
# distributed sampler: each rank owns a deterministic partition of the epoch
# --------------------------------------------------------------------------
class DistributedPackedSampler:
    """Weighted-mixture packed sampler sharded across data-parallel ranks.

    Each epoch the *global* weighted doc list is shuffled with the same seed on
    every rank, then partitioned round-robin by rank.  Every rank therefore sees
    a disjoint, deterministic slice of the same global order.
    """

    def __init__(self, docs, tok, rank: int, world_size: int, shuffle_seed: int):
        self.rank = rank
        self.world_size = world_size
        self.shuffle_seed = shuffle_seed
        self.tok = tok
        self.doc_tokens: dict[str, list[list[int]]] = {domain: [] for domain in MIXTURE}
        for domain, text in docs:
            self.doc_tokens[domain].append(tok.encode_doc(text))
        self.epoch = 0
        self.cursor = 0

    def _epoch_blocks(self) -> tuple[torch.Tensor, torch.Tensor]:
        weighted: list[list[int]] = []
        for domain, weight in MIXTURE.items():
            for tokens in self.doc_tokens[domain]:
                weighted.extend([tokens] * weight)
        random.Random(self.shuffle_seed + self.epoch).shuffle(weighted)
        local = weighted[self.rank :: self.world_size]
        return pack_documents(local)

    def next_batch(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state(self, state: dict) -> None:
        self.epoch, self.cursor = state["epoch"], state["cursor"]


# --------------------------------------------------------------------------
# data-parallel training step (manual all-reduce, transparent to the learner)
# --------------------------------------------------------------------------
def run_step_dp(
    model,
    optimizer,
    scheduler,
    sampler,
    rank: int,
    world_size: int,
    use_amp: bool = False,
    spike: bool = False,
) -> tuple[float, float, float]:
    model.train()
    loss_total, valid_tokens = 0.0, 0
    optimizer.zero_grad()
    for micro in range(ACCUM):
        ids, doc = sampler.next_batch(MICRO_BATCH)
        labels = boundary_labels(ids, doc)
        mask = document_attn_mask(doc[:, :-1])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids[:, :-1], mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE)
        # Fault injection: every rank sees a blown-up loss at one micro-batch.
        if spike and micro == 0:
            loss = loss * 100.0
        (loss / ACCUM).backward()
        valid = int((labels != IGNORE).sum())
        loss_total += float(loss.detach()) * valid
        valid_tokens += valid

    # Average one flattened gradient bucket across ranks.  Real DDP uses
    # multiple overlap-aware buckets; one bucket keeps this CPU lesson fast
    # while preserving the same numerical contract.
    grad_params = [p for p in model.parameters() if p.grad is not None]
    flat = torch.cat([p.grad.detach().reshape(-1) for p in grad_params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    offset = 0
    for p in grad_params:
        count = p.grad.numel()
        p.grad.copy_(flat[offset : offset + count].view_as(p.grad))
        offset += count

    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    scheduler.step()
    local_loss = loss_total / max(1, valid_tokens)
    global_loss = _global_train_loss(local_loss, valid_tokens)
    return global_loss, optimizer.param_groups[0]["lr"], grad_norm


def _global_train_loss(local_loss: float, local_valid: int) -> float:
    t = torch.tensor([local_loss * local_valid, float(local_valid)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0] / max(1, t[1]))


@torch.no_grad()
def global_validation_loss(model, val_batch, rank: int) -> float:
    local = validation_loss(model, val_batch)
    t = torch.tensor(local, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t) / dist.get_world_size()


# --------------------------------------------------------------------------
# rank-local checkpoint set: replicated tensors plus rank-specific state
# --------------------------------------------------------------------------
def save_rank_local_checkpoint(
    ckpt_dir: Path,
    rank: int,
    world_size: int,
    model,
    optimizer,
    scheduler,
    sampler,
    step: int,
    consumed_tokens: int,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shard = {
        "rank": rank,
        "world_size": world_size,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
        "sampler": sampler.state(),
        "step": step,
        "consumed_tokens": consumed_tokens,
    }
    torch.save(shard, ckpt_dir / f"shard_{rank}.pt")
    dist.barrier()
    if rank == 0:
        meta = {
            "world_size": world_size,
            "step": step,
            "consumed_tokens": consumed_tokens,
            "ranks": list(range(world_size)),
            "shards": [f"shard_{r}.pt" for r in range(world_size)],
        }
        (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    dist.barrier()


def load_rank_local_checkpoint(
    ckpt_dir: Path,
    rank: int,
    world_size: int,
    model,
    optimizer,
    scheduler,
    sampler,
    force_shard_zero: bool = False,
):
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    if meta["world_size"] != world_size:
        raise RuntimeError(
            f"checkpoint world_size={meta['world_size']} != current {world_size}"
        )
    shard_path = ckpt_dir / ("shard_0.pt" if force_shard_zero else f"shard_{rank}.pt")
    shard = torch.load(shard_path, weights_only=False)
    if shard.get("world_size") != world_size:
        raise RuntimeError(
            f"rank-local shard world_size={shard.get('world_size')} != current {world_size}"
        )
    if shard.get("rank") != rank:
        raise RuntimeError(
            f"rank-local shard rank={shard.get('rank')} loaded by rank={rank}"
        )
    model.load_state_dict(shard["model"])
    optimizer.load_state_dict(shard["optimizer"])
    scheduler.load_state_dict(shard["scheduler"])
    torch.set_rng_state(shard["torch_rng"])
    random.setstate(shard["py_rng"])
    sampler.load_state(shard["sampler"])
    return shard["step"], shard["consumed_tokens"]


# --------------------------------------------------------------------------
# per-rank worker: builds an identical world and runs a distributed phase
# --------------------------------------------------------------------------
def make_world(rank: int, world_size: int):
    torch.manual_seed(SEED)
    random.seed(SEED)
    docs, valid_docs = build_corpus(SEED)
    tok = CharTokenizer([t for _, t in docs] + [t for _, t in valid_docs])
    model = TinyGPT(tok.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    sampler = DistributedPackedSampler(docs, tok, rank, world_size, SAMPLER_SEED)
    val_ids, val_doc = pack_documents([tok.encode_doc(t) for _, t in valid_docs])
    return model, optimizer, scheduler, sampler, tok, (val_ids, val_doc)


def worker(rank: int, world_size: int, port: int, temp_root: str, mode: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    interfaces = {name for _index, name in socket.if_nameindex()}
    loopback = "lo" if "lo" in interfaces else "lo0" if "lo0" in interfaces else ""
    if loopback:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.set_num_threads(1)
    # All ranks start from identical model and sampler state; divergence comes
    # only from the per-rank data partition, which is captured in sampler.state().
    model, optimizer, scheduler, sampler, tok, val_batch = make_world(rank, world_size)
    consumed = 0
    history: list[tuple[int, float, float, float]] = []

    def run_steps(start: int, end: int, spike_steps: set[int] | None = None):
        nonlocal consumed
        spike_steps = spike_steps or set()
        for step in range(start, end + 1):
            train_loss, lr, _ = run_step_dp(
                model, optimizer, scheduler, sampler, rank, world_size,
                use_amp=False, spike=(step in spike_steps),
            )
            consumed += ACCUM * MICRO_BATCH * SEQ_LEN
            val = global_validation_loss(model, val_batch, rank)
            history.append((step, train_loss, lr, val))

    if mode == "baseline":
        run_steps(1, SPLIT_STEP)
        save_rank_local_checkpoint(
            Path(temp_root) / "ckpt_split", rank, world_size,
            model, optimizer, scheduler, sampler, SPLIT_STEP, consumed,
        )
        run_steps(SPLIT_STEP + 1, ROLLBACK_STEP)
        save_rank_local_checkpoint(
            Path(temp_root) / "ckpt_rollback", rank, world_size,
            model, optimizer, scheduler, sampler, ROLLBACK_STEP, consumed,
        )
        run_steps(ROLLBACK_STEP + 1, TOTAL_STEPS)

    elif mode == "resume":
        # A new process group resumes the checkpoint emitted by baseline.
        model_r, optimizer_r, scheduler_r, sampler_r, _tok_r, val_batch_r = make_world(rank, world_size)
        step0, consumed0 = load_rank_local_checkpoint(
            Path(temp_root) / "ckpt_split", rank, world_size,
            model_r, optimizer_r, scheduler_r, sampler_r,
        )
        model, optimizer, scheduler, sampler, val_batch = model_r, optimizer_r, scheduler_r, sampler_r, val_batch_r
        consumed = consumed0
        run_steps(step0 + 1, TOTAL_STEPS)

    elif mode == "resume_mismatch":
        model_r, optimizer_r, scheduler_r, sampler_r, _tok_r, val_batch_r = make_world(rank, world_size)
        rejected = 0
        try:
            load_rank_local_checkpoint(
                Path(temp_root) / "ckpt_split", rank, world_size,
                model_r, optimizer_r, scheduler_r, sampler_r, force_shard_zero=True,
            )
        except RuntimeError as exc:
            rejected = int("rank-local shard rank=" in str(exc))
        rejection_count = torch.tensor(rejected, dtype=torch.int64)
        dist.all_reduce(rejection_count, op=dist.ReduceOp.SUM)
        torch.save(
            {"rank_mismatch_rejections": int(rejection_count)},
            Path(temp_root) / f"result_{mode}_{rank}.pt",
        )
        dist.destroy_process_group()
        return

    elif mode in {"spike", "rollback_clean"}:
        # Rollback anchor must already exist from baseline phase.
        model_r, optimizer_r, scheduler_r, sampler_r, _tok_r, val_batch_r = make_world(rank, world_size)
        step0, consumed0 = load_rank_local_checkpoint(
            Path(temp_root) / "ckpt_rollback", rank, world_size,
            model_r, optimizer_r, scheduler_r, sampler_r,
        )
        model, optimizer, scheduler, sampler, val_batch = model_r, optimizer_r, scheduler_r, sampler_r, val_batch_r
        consumed = consumed0
        injected = {SPIKE_STEP} if mode == "spike" else set()
        run_steps(step0 + 1, TOTAL_STEPS, spike_steps=injected)

    # Gather rank-0 state for the orchestrator.
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "history": history,
            "state_dict": state_dict,
            "consumed": consumed,
            "sampler": sampler.state(),
        },
        Path(temp_root) / f"result_{mode}_{rank}.pt",
    )

    dist.destroy_process_group()


def run_phase(temp_root: str, mode: str, port: int) -> dict:
    mp.spawn(
        worker,
        args=(WORLD_SIZE, port, temp_root, mode),
        nprocs=WORLD_SIZE,
        join=True,
    )
    return {
        rank: torch.load(
            Path(temp_root) / f"result_{mode}_{rank}.pt", weights_only=False
        )
        for rank in range(WORLD_SIZE)
    }


def max_state_diff(a: dict, b: dict) -> float:
    return max(float((a[k] - b[k]).abs().max()) for k in a)


# --------------------------------------------------------------------------
# main orchestrator: runs phases and compares distributed trajectories
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-rank CPU/gloo exact-resume teaching experiment."
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=int(os.environ.get("NANO_PRETRAINING_L2_BASE_PORT", "29500")),
        help="first of five consecutive localhost ports (default: 29500)",
    )
    args = parser.parse_args()
    if not 1024 <= args.base_port <= 65531:
        parser.error("--base-port must be between 1024 and 65531")
    return args


def main() -> None:
    args = parse_args()
    print("=" * 78)
    print("Pretraining lifecycle L2 — distributed exact resume & fault triage")
    print("=" * 78)
    print(f"[0] environment: torch={torch.__version__} world_size={WORLD_SIZE} backend=gloo device=cpu")
    print("    real: torch.distributed / manual all-reduce / rank-local checkpoint set")
    print("    toy : world_size=2, loopback interconnect, tiny model")
    print("    DP model/optimizer tensors are replicated; sampler/RNG identity remains rank-local")

    with tempfile.TemporaryDirectory(prefix="l2_dist_") as tmp:
        tmp_path = Path(tmp)

        # Phase 1: continuous distributed run is the reference.
        base = run_phase(str(tmp_path), "baseline", port=args.base_port)
        rank0_base = base[0]
        history_base = rank0_base["history"]
        final_val_base = history_base[-1][3]
        consumed_per_rank = rank0_base["consumed"]
        consumed_global = consumed_per_rank * WORLD_SIZE
        print(f"\n[1] continuous distributed run (world_size={WORLD_SIZE})")
        print("    step  global_train   lr        global_val")
        for step, train_loss, lr, val in history_base:
            if step in (1, 5, 10, 15, 20):
                print(f"    {step:>4}  {train_loss:>12.4f} {lr:>8.5f} {val:>10.4f}")
        print(f"    final val={final_val_base:.4f}  tokens/rank={consumed_per_rank} global_tokens={consumed_global}")

        # Phase 2: run to SPLIT_STEP, save the rank-local set, resume, continue.
        resumed = run_phase(str(tmp_path), "resume", port=args.base_port + 1)
        rank0_resumed = resumed[0]
        history_resumed = rank0_resumed["history"]
        final_val_resumed = history_resumed[-1][3]
        resume_diff = max_state_diff(rank0_base["state_dict"], rank0_resumed["state_dict"])
        print(f"\n[2] exact resume from rank-local checkpoint set@step{SPLIT_STEP}")
        print(f"    resumed final val={final_val_resumed:.4f}")
        print(f"    max param diff vs continuous = {resume_diff:.3e}")

        # Phase 3: rank metadata must reject shard_0 on every non-zero rank.
        mismatched = run_phase(str(tmp_path), "resume_mismatch", port=args.base_port + 2)
        mismatch_rejections = mismatched[0]["rank_mismatch_rejections"]
        print(f"\n[3] fault injection: load shard_0.pt on all ranks")
        print(f"    rejected non-owner loads={mismatch_rejections}/{WORLD_SIZE - 1}")
        print("    fail-closed rank binding prevents silent sampler/RNG identity corruption")

        # Phase 4: observe an anomalous branch from the rollback anchor.
        spiked = run_phase(str(tmp_path), "spike", port=args.base_port + 3)
        rank0_spike = spiked[0]
        history_spike = rank0_spike["history"]
        spike_idx = next(i for i, (s, *_rest) in enumerate(history_spike) if s == SPIKE_STEP)
        pre_spike_train = history_spike[spike_idx - 1][1]
        spike_train = history_spike[spike_idx][1]
        spike_ratio = spike_train / pre_spike_train
        print(f"\n[4] anomaly branch from checkpoint@step{ROLLBACK_STEP}: inject at step{SPIKE_STEP}")
        print(f"    pre-spike train={pre_spike_train:.4f} spike_train={spike_train:.4f} ratio={spike_ratio:.1f}x")

        # Phase 5: discard the anomaly branch and replay cleanly from the anchor.
        clean = run_phase(str(tmp_path), "rollback_clean", port=args.base_port + 4)
        rank0_clean = clean[0]
        clean_diff = max_state_diff(rank0_base["state_dict"], rank0_clean["state_dict"])
        clean_final_val = rank0_clean["history"][-1][3]
        print(f"\n[5] rollback: discard anomaly branch and replay step{ROLLBACK_STEP + 1}..{TOTAL_STEPS}")
        print(f"    clean replay final val={clean_final_val:.4f}")
        print(f"    max param diff vs continuous = {clean_diff:.3e}")

        # Self-check.
        lrs = [item[2] for item in history_base]
        checks = (
            (final_val_base < history_base[0][3], "distributed training improves validation loss"),
            (lrs[1] > lrs[0], "warmup raises the learning rate"),
            (lrs[-1] < 0.1 * max(lrs), "cosine decay lowers the learning rate"),
            (consumed_per_rank == TOTAL_STEPS * ACCUM * MICRO_BATCH * SEQ_LEN,
             "per-rank token ledger closes arithmetically"),
            (consumed_global == consumed_per_rank * WORLD_SIZE,
             "global token ledger includes every data-parallel rank"),
            (resume_diff == 0.0, "rank-local exact resume matches continuous run"),
            (final_val_resumed == final_val_base, "resumed final val equals continuous final val"),
            (mismatch_rejections == WORLD_SIZE - 1, "wrong-rank checkpoint loads fail closed"),
            (spike_ratio > 10.0, "injected loss spike is observable before promotion"),
            (clean_diff == 0.0, "clean rollback replay rejoins the continuous trajectory"),
        )
        print("\n[6] self-check")
        for ok, name in checks:
            print(f"    {'PASS' if ok else 'FAIL'} | {name}")
        failed = [name for ok, name in checks if not ok]
        if failed:
            raise AssertionError(f"self-check failed: {failed}")
        print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")

        metrics = {
            "clean_replay_param_diff": clean_diff,
            "final_validation_loss": round(final_val_base, 6),
            "global_tokens": consumed_global,
            "rank_mismatch_rejections": mismatch_rejections,
            "resume_param_diff": resume_diff,
            "spike_train_ratio": round(spike_ratio, 6),
            "tokens_per_rank": consumed_per_rank,
        }
        canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        print(f"digest(sha256 of metrics): {digest}")
        print("takeaway: data parallelism changes the bookkeeping, not the contract:")
        print("          same full distributed state == same distributed training run.")
        result = {
            "checks": {"passed": len(checks), "total": len(checks)},
            "digest": digest,
            "evidence_boundary": (
                "Real two-process torch.distributed/gloo on CPU; tiny model and loopback only. "
                "DP tensors are replicated, not FSDP-sharded, and timing is not benchmarked."
            ),
            "metrics": metrics,
            "module": "nano_pretraining_loop_l2",
            "schema_version": 1,
        }
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
