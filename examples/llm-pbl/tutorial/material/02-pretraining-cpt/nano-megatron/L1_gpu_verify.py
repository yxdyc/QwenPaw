#!/usr/bin/env python3
"""
L1_gpu_verify.py — nano-megatron L1 GPU empirical verification.

Targets the `[TODO: verify on real system]` notes in tutorial_L1.md:
- real GPU/NCCL all-reduce wall-clock (fp32 / fp16 / bf16),
- tensor-parallel f/g autograd operators on NCCL remain numerically equivalent
  to the dense reference and still incur exactly 1 fwd + 1 bwd all-reduce,
- fp16/bf16 all-reduce support under NCCL,
- single-host TP=2/4/8 collective scaling without confusing it with
  end-to-end training speedup.

Run on one host with 2, 4, or 8 visible CUDA GPUs:
    python3 -B L1_gpu_verify.py --world-size 2 [--master-port 29522]

Wall-clock and bandwidth are hardware/software-stack observations, not portable
constants. Numerical equivalence, communication counts and memory accounting are
the mechanism checks.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

SEED = 7
H, FF, T = 64, 256, 128          # same toy shape as L1; communication still dominates
MLP_PARAMS = 2 * H * FF          # 32,768
WARMUP, ITERS = 10, 20           # more iterations than CPU run to wash out GPU launch noise
BENCH_SIZES_MIB = (1.0, 16.0, 32.0)
THREADS_PER_PROC = 4

COMM = {"fwd": 0, "bwd": 0}


class CopyToTensorParallelRegion(torch.autograd.Function):
    """f: fwd identity, bwd all-reduce (Megatron _CopyToModelParallelRegion)."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.contiguous().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        COMM["bwd"] += 1
        return g


class ReduceFromTensorParallelRegion(torch.autograd.Function):
    """g: fwd all-reduce, bwd identity (Megatron _ReduceFromModelParallelRegion)."""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        COMM["fwd"] += 1
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def build_shared_tensors(device: torch.device):
    torch.manual_seed(SEED)
    W1 = torch.randn(H, FF, device=device)
    W2 = torch.randn(FF, H, device=device)
    X = torch.randn(T, H, device=device)
    return W1, W2, X


def max_across_ranks(v: float) -> float:
    t = torch.tensor([v], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def fmt_kib(nbytes: float) -> str:
    return f"{nbytes / 1024:.1f} KiB"


def fmt_mib(nbytes: float) -> str:
    return f"{nbytes / 1024 / 1024:.1f} MiB"


def run(rank: int, world_size: int, full_suite: bool):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    W1, W2, X = build_shared_tensors(device)
    assert FF % world_size == 0 and H % world_size == 0
    s = FF // world_size

    # dense reference
    W1g = W1.clone().requires_grad_(True)
    W2g = W2.clone().requires_grad_(True)
    Xg = X.clone().requires_grad_(True)
    Y_ref = F.gelu(Xg @ W1g) @ W2g
    Y_ref.sum().backward()
    y_scale = Y_ref.detach().abs().max().item()

    # TP forward: column-row split + 1 all-reduce
    W1_r = W1[:, rank * s:(rank + 1) * s].clone().requires_grad_(True)
    W2_r = W2[rank * s:(rank + 1) * s, :].clone().requires_grad_(True)
    X_in = X.clone().requires_grad_(True)
    Y_tp = ReduceFromTensorParallelRegion.apply(
        F.gelu(CopyToTensorParallelRegion.apply(X_in) @ W1_r) @ W2_r)
    fwd_err = max_across_ranks((Y_tp.detach() - Y_ref.detach()).abs().max().item())

    Y_tp.sum().backward()
    dx_err = max_across_ranks((X_in.grad - Xg.grad).abs().max().item())
    dw1_err = max_across_ranks(
        (W1_r.grad - W1g.grad[:, rank * s:(rank + 1) * s]).abs().max().item())
    dw2_err = max_across_ranks(
        (W2_r.grad - W2g.grad[rank * s:(rank + 1) * s, :]).abs().max().item())

    # anti-example: row-first W1 split (real all-reduce, still wrong)
    s_in = H // world_size
    P_r = X[:, rank * s_in:(rank + 1) * s_in] @ W1[rank * s_in:(rank + 1) * s_in, :]
    naive = F.gelu(P_r).clone()
    dist.all_reduce(naive, op=dist.ReduceOp.SUM)
    Y_naive = naive @ W2
    naive_err = max_across_ranks((Y_naive - Y_ref.detach()).abs().max().item())
    fixed = P_r.clone()
    dist.all_reduce(fixed, op=dist.ReduceOp.SUM)
    Y_fixed = F.gelu(fixed) @ W2
    fixed_err = max_across_ranks((Y_fixed - Y_ref.detach()).abs().max().item())

    # memory ledger per rank
    opt = torch.optim.Adam([W1_r, W2_r], lr=1e-3)
    opt.step()
    p_b = sum(p.numel() * p.element_size() for p in (W1_r, W2_r))
    g_b = sum(p.grad.numel() * p.grad.element_size() for p in (W1_r, W2_r))
    o_b = sum(v.numel() * v.element_size()
              for st in opt.state.values() for v in st.values()
              if torch.is_tensor(v) and v.dim() > 0)
    ledger_local = torch.tensor([p_b, g_b, o_b], dtype=torch.float64, device="cuda")
    ledger_all = [torch.zeros(3, dtype=torch.float64, device="cuda") for _ in range(world_size)]
    dist.all_gather(ledger_all, ledger_local)

    # timing: real all-reduce wall-clock on GPU
    fp16_msg = bf16_msg = None
    timings = []          # list of (size_MiB, dtype_str, ms_per_call)
    compute_ms = None
    if full_suite:
        for dtype, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            try:
                probe = torch.ones(4, dtype=dtype, device="cuda")
                dist.all_reduce(probe, op=dist.ReduceOp.SUM)
                got = float(probe[0].item())
                msg = (f"{label} all_reduce supported, sum(1 x {world_size} ranks) = {got:.1f}  "
                       f"{'✅ 数值正确' if abs(got - world_size) < 1e-3 else '❌ 数值错误'}")
            except Exception as e:  # noqa: BLE001
                msg = f"{label} all_reduce not supported: {type(e).__name__}"
            if label == "fp16":
                fp16_msg = msg
            else:
                bf16_msg = msg

        for mib in BENCH_SIZES_MIB:
            elements = int(mib * 1024 * 1024 / 4)  # fp32
            buf = torch.randn(elements, device="cuda")
            for _ in range(WARMUP):
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            dt_ms = (time.perf_counter() - t0) / ITERS * 1000
            timings.append((mib, "fp32", max_across_ranks(dt_ms)))

        # fp16/bf16 timing at 16 MiB message only (production mixed-precision sweet spot)
        for dtype, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            elements = int(16.0 * 1024 * 1024 / torch.finfo(dtype).bits * 8)
            buf = torch.randn(elements, device="cuda", dtype=dtype)
            for _ in range(WARMUP):
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            dt_ms = (time.perf_counter() - t0) / ITERS * 1000
            timings.append((16.0, label, max_across_ranks(dt_ms)))

        t0 = time.perf_counter()
        for _ in range(200):
            _ = F.gelu(X_in.detach() @ W1_r.detach()) @ W2_r.detach()
        torch.cuda.synchronize(device)
        compute_ms = max_across_ranks((time.perf_counter() - t0) / 200 * 1000)

    fp16_ok = "数值正确" in (fp16_msg or "")
    bf16_ok = "数值正确" in (bf16_msg or "")
    rel = lambda e: e / y_scale  # noqa: E731
    assert rel(fwd_err) < 1e-5, f"TP fwd mismatch: {fwd_err}"
    assert rel(dx_err) < 1e-5 and rel(dw1_err) < 1e-5 and rel(dw2_err) < 1e-5, \
        f"TP bwd mismatch: dX={dx_err} dW1={dw1_err} dW2={dw2_err}"
    assert rel(naive_err) > 1e-3, "row-first naive should still be wrong"
    assert rel(fixed_err) < 1e-5, f"row-first fixed should recover: {fixed_err}"
    assert COMM == {"fwd": 1, "bwd": 1}, f"comm count anomaly: {COMM}"
    assert not full_suite or (fp16_ok and bf16_ok), (
        f"mixed-precision collective failed: fp16={fp16_msg!r}, bf16={bf16_msg!r}"
    )
    for lg in ledger_all:
        assert lg[0] == MLP_PARAMS * 4 / world_size, "params per rank should be P/N"
        assert lg[2] == 2 * lg[0], "Adam state should be 2x params"

    if rank == 0:
        tag = f"[TP={world_size} GPU/NCCL]"
        print(f"\n{tag} numerical equivalence")
        print(f"    max|Y_tp - Y_ref| = {fwd_err:.3e} (relative {rel(fwd_err):.1e})  ✅")
        print(f"    max|dX - dX_ref| = {dx_err:.3e}")
        print(f"    max|dW1 shard - ref block| = {dw1_err:.3e}")
        print(f"    max|dW2 shard - ref block| = {dw2_err:.3e}")
        print(f"    comm counters: fwd={COMM['fwd']} bwd={COMM['bwd']}")
        print(f"{tag} row-first anti-example")
        print(f"    naive (GeLU before reduce): err={naive_err:.3e}  ❌")
        print(f"    fixed (reduce before GeLU): err={fixed_err:.3e}  ✅")
        print(f"{tag} per-rank memory ledger")
        for r in range(world_size):
            pb, gb, ob = ledger_all[r].tolist()
            print(f"    rank {r}: params={fmt_kib(pb)} grads={fmt_kib(gb)} "
                  f"optimizer={fmt_kib(ob)} total={fmt_kib(pb + gb + ob)}")
        dense_total = MLP_PARAMS * 16
        print(f"    dense replica = {fmt_kib(dense_total)} | "
              f"sum across ranks = {fmt_kib(sum(l.sum().item() for l in ledger_all))}")
        if full_suite:
            print(f"{tag} all-reduce wall-clock (NCCL, max over ranks, {ITERS} iters mean)")
            print(f"    {fp16_msg}")
            print(f"    {bf16_msg}")
            for mib, label, ms in timings:
                algbw = (mib * 1024 * 1024 / (ms / 1000)) / 1e9
                # NCCL-tests convention for a ring all-reduce: normalize the
                # logical payload rate by the 2*(N-1)/N traffic factor.
                busbw = algbw * 2 * (world_size - 1) / world_size
                print(f"    {label:4s} {mib:5.1f} MiB: {ms:7.3f} ms/call  "
                      f"(algbw ≈ {algbw:.1f}, busbw ≈ {busbw:.1f} GB/s)")
            print(f"    toy fwd compute per rank: {compute_ms:.4f} ms")

    return {
        "fwd_err": fwd_err,
        "dx_err": dx_err,
        "dw1_err": dw1_err,
        "dw2_err": dw2_err,
        "naive_err": naive_err,
        "fixed_err": fixed_err,
        "comm_fwd": COMM["fwd"],
        "comm_bwd": COMM["bwd"],
        "fp16_ok": fp16_ok,
        "bf16_ok": bf16_ok,
        "params_bytes_per_rank": int(MLP_PARAMS * 4 / world_size),
        "timings": timings,
        "compute_ms": compute_ms,
    }


def tp_worker(rank: int, world_size: int, port: int, full_suite: bool, results: dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(THREADS_PER_PROC)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    init_kwargs = {"device_id": device} if "device_id" in inspect.signature(
        dist.init_process_group
    ).parameters else {}
    dist.init_process_group("nccl", rank=rank, world_size=world_size, **init_kwargs)
    try:
        res = run(rank, world_size, full_suite)
        if rank == 0:
            results.update(res)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify nano Megatron TP on 2, 4, or 8 CUDA GPUs."
    )
    parser.add_argument(
        "--world-size",
        type=int,
        choices=(2, 4, 8),
        default=2,
        help="number of visible GPUs participating in tensor parallelism",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("NANO_MEGATRON_MASTER_PORT", "29522")),
        help="localhost rendezvous port (or NANO_MEGATRON_MASTER_PORT)",
    )
    args = parser.parse_args()
    if not 1024 <= args.master_port <= 65535:
        parser.error("--master-port must be in [1024, 65535]")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available; this verification must run on a GPU machine.")
    if torch.cuda.device_count() < args.world_size:
        sys.exit(
            f"Need at least {args.world_size} visible GPUs; found {torch.cuda.device_count()}."
        )

    warnings.filterwarnings("ignore")
    print("=" * 72)
    print("nano-megatron L1 — GPU empirical verification (NCCL)")
    print("=" * 72)
    print(f"shape: X[{T},{H}]  W1[{H},{FF}]  W2[{FF},{H}]  GeLU MLP | seed={SEED}")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  nccl {torch.cuda.nccl.version()}")
    print(
        f"participants: TP={args.world_size} | visible devices: "
        f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"
    )

    # Use a shared dict to collect rank-0 results from the spawned process.
    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        tp_worker,
        args=(args.world_size, args.master_port, True, results),
        nprocs=args.world_size,
        join=True,
    )

    print("\n" + "=" * 72)
    print("✅ GPU self-check passed: TP fwd/bwd equivalent to dense on NCCL; "
          "row-first anti-example still wrong; memory ledger 16P/N; fp16/bf16 supported.")
    print("=" * 72)

    digest_src = repr((
        round(results["fwd_err"], 9),
        round(results["dx_err"], 9),
        round(results["dw1_err"], 9),
        round(results["dw2_err"], 9),
        round(results["naive_err"], 9),
        round(results["fixed_err"], 9),
        results["comm_fwd"],
        results["comm_bwd"],
        results["fp16_ok"],
        results["bf16_ok"],
        args.world_size,
    ))
    gpu_digest = hashlib.md5(digest_src.encode()).hexdigest()
    print(f"gpu_digest: {gpu_digest}")
    payload = {
        "schema_version": "1.0",
        "module": "nano-megatron-l1-gpu",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": str(torch.cuda.nccl.version()),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        },
        "checks": {"passed": 7, "total": 7, "gpu_digest": gpu_digest},
        "metrics": {
            "world_size": args.world_size,
            "fwd_max_abs": results["fwd_err"],
            "dx_max_abs": results["dx_err"],
            "dw1_max_abs": results["dw1_err"],
            "dw2_max_abs": results["dw2_err"],
            "naive_max_abs": results["naive_err"],
            "fixed_max_abs": results["fixed_err"],
            "params_bytes_per_rank": results["params_bytes_per_rank"],
        },
        "observations": {
            "collective_timings_ms": [list(item) for item in results["timings"]],
            "shard_compute_ms": results["compute_ms"],
        },
        "evidence_boundary": (
            f"single-host TP={args.world_size} toy; collective and shard-compute timings "
            "are stack-specific and do not establish end-to-end training speedup"
        ),
    }
    print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
