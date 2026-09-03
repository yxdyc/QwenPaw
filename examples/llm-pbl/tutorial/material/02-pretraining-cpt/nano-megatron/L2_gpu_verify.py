#!/usr/bin/env python3
"""
L2_gpu_verify.py — nano-megatron L2 GPU empirical verification.

Targets the `[TODO: verify on real system]` notes in tutorial_L2.md:
- real GPU/NCCL P2P wall-clock and pipeline bubble,
- PP correctness (bit-identical to dense mirror on NCCL),
- P2P byte accounting and memory ledger on GPU.

Run on one host with at least two visible CUDA GPUs:
    python3 -B L2_gpu_verify.py [--master-port 29541]

Wall-clock and bubble measurements depend on the hardware/software stack. The
portable teaching claims are numerical equivalence, P2P byte accounting, live
microbatch bounds and the memory ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

SEED = 7
H, FF = 128, 512
N_BLOCKS = 4                    # 4 个 GeLU-MLP 块，按层切成 2 个 stage
WORLD_SIZE = 2
T = 512                         # 一个 batch 的 token 数
LR = 1e-3
MICROBATCHES = (1, 2, 4, 8)
REPEATS = 5                     # 每个 (m, schedule) 计时重复次数，取中位数
WARMUP_REPEATS = 2              # 额外热身轮，洗掉 CUDA launch 噪声
THREADS_PER_PROC = 4
FULL_BATCH_ATOL = 1e-5
FULL_BATCH_RTOL = 1e-4

P_BLOCK = 2 * H * FF
P_TOTAL = N_BLOCKS * P_BLOCK
BLOCKS_PER_STAGE = N_BLOCKS // WORLD_SIZE
FP32 = 4

# 每 rank 的 p2p 计数器 + 墙钟累计
P2P = {'send_fwd': 0, 'recv_fwd': 0, 'send_bwd': 0, 'recv_bwd': 0, 'bytes': 0, 'time': 0.0}


def build_blocks(device: torch.device):
    torch.manual_seed(SEED)
    blocks = []
    for _ in range(N_BLOCKS):
        W1 = (torch.randn(H, FF, device=device) / H ** 0.5).requires_grad_(True)
        W2 = (torch.randn(FF, H, device=device) / FF ** 0.5).requires_grad_(True)
        blocks.append([W1, W2])
    X = torch.randn(T, H, device=device)
    return blocks, X


def stage_forward(blocks, act):
    for W1, W2 in blocks:
        act = F.gelu(act @ W1) @ W2
    return act


def max_across_ranks(v: float, device: torch.device) -> float:
    t = torch.tensor([v], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def fmt_kib(nbytes: float) -> str:
    return f"{nbytes / 1024:.1f} KiB"


def _p2p_batch(ops_info, device):
    """非阻塞批量 P2P；统计调用次数、字节数与墙钟。"""
    if not ops_info:
        return []

    recv_tensors = []
    ops = []
    for op_type, data, peer, direction in ops_info:
        if op_type == 'send':
            t = data.contiguous()
            ops.append(dist.P2POp(dist.isend, t, peer))
        else:
            t = torch.empty(data, device=device)
            recv_tensors.append(t)
            ops.append(dist.P2POp(dist.irecv, t, peer))

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    P2P['time'] += dt

    for op_type, data, peer, direction in ops_info:
        if op_type == 'send':
            nbytes = data.numel() * data.element_size()
            if direction == 'fwd':
                P2P['send_fwd'] += 1
            else:
                P2P['send_bwd'] += 1
        else:
            nbytes = data[0] * data[1] * FP32
            if direction == 'fwd':
                P2P['recv_fwd'] += 1
            else:
                P2P['recv_bwd'] += 1
        P2P['bytes'] += nbytes

    return recv_tensors


def reference_run(m, per_mb, device):
    """单进程 GPU 参照。per_mb=True 与 PP 同形逐 mb；per_mb=False 为真 full-batch。"""
    blocks, X = build_blocks(device)
    Ws = [w for blk in blocks for w in blk]
    opt = torch.optim.Adam(Ws, lr=LR)
    if per_mb:
        mb = T // m
        outs = [stage_forward(blocks, X[i * mb:(i + 1) * mb]) for i in range(m)]
        losses = [o.sum() / T for o in outs]
        for l in losses:
            l.backward()
        losses = [l.item() for l in losses]
    else:
        Y = stage_forward(blocks, X)
        loss = Y.sum() / T
        losses = [loss.item()]
        loss.backward()
    opt.step()
    return losses, [w.detach().clone() for w in Ws]


def run_pipeline(schedule, m, rank, world, blocks_local, X, device):
    """跑一遍完整 forward+backward（梯度累加，不做 step）。"""
    global P2P
    P2P = {'send_fwd': 0, 'recv_fwd': 0, 'send_bwd': 0, 'recv_bwd': 0, 'bytes': 0, 'time': 0.0}
    mb = T // m
    is_first = rank == 0
    is_last = rank == world - 1
    live = []
    peak = 0
    busy = 0.0
    losses = []

    def mark_busy(t0):
        nonlocal busy
        torch.cuda.synchronize(device)
        busy += time.perf_counter() - t0

    def do_forward(i):
        nonlocal busy, peak
        if not is_first:
            result = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')], device)
            act = result[0].requires_grad_(True)
        else:
            act = X[i * mb:(i + 1) * mb]

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = stage_forward(blocks_local, act)
        mark_busy(t0)

        if is_last:
            loss = out.sum() / T
            losses.append(loss)
            live.append((act, loss))
        else:
            _p2p_batch([('send', out, rank + 1, 'fwd')], device)
            live.append((act, out))
        peak = max(peak, len(live))

    def do_backward(i):
        nonlocal busy
        act, out = live.pop(0)
        if is_last:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out.backward()
            mark_busy(t0)
            _p2p_batch([('send', act.grad, rank - 1, 'bwd')], device)
        else:
            result = _p2p_batch([('recv', (mb, H), rank + 1, 'bwd')], device)
            g = result[0]
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out.backward(g)
            mark_busy(t0)
            if not is_first:
                _p2p_batch([('send', act.grad, rank - 1, 'bwd')], device)

    dist.barrier()
    torch.cuda.synchronize(device)
    t_wall0 = time.perf_counter()
    if schedule == 'gpipe':
        for i in range(m):
            do_forward(i)
        for i in range(m):
            do_backward(i)
    else:
        # 1F1B：与 Megatron schedules.py:L2129 without_interleaving 同构
        warmup = min(world - rank - 1, m)
        remaining = m - warmup
        nf = nb = 0

        # warmup forward
        for _ in range(warmup):
            if is_first:
                act = X[nf * mb:(nf + 1) * mb]
            else:
                r = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')], device)
                act = r[0].requires_grad_(True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = stage_forward(blocks_local, act)
            mark_busy(t0)
            if is_last:
                loss = out.sum() / T
                losses.append(loss)
                live.append((act, loss))
            else:
                _p2p_batch([('send', out, rank + 1, 'fwd')], device)
                live.append((act, out))
            nf += 1
            peak = max(peak, len(live))

        # steady state
        for _ in range(remaining):
            if is_first:
                act = X[nf * mb:(nf + 1) * mb]
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out = stage_forward(blocks_local, act)
                mark_busy(t0)
                r = _p2p_batch([
                    ('send', out, rank + 1, 'fwd'),
                    ('recv', (mb, H), rank + 1, 'bwd'),
                ], device)
                bwd_grad = r[0]
                live.append((act, out))
                peak = max(peak, len(live))
                act_b, out_b = live.pop(0)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out_b.backward(bwd_grad)
                mark_busy(t0)
            elif is_last:
                r = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')], device)
                fwd_act = r[0].requires_grad_(True)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                fwd_out = stage_forward(blocks_local, fwd_act)
                mark_busy(t0)
                loss = fwd_out.sum() / T
                losses.append(loss)
                live.append((fwd_act, loss))
                peak = max(peak, len(live))
                act_b, out_b = live.pop(0)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out_b.backward()
                mark_busy(t0)
                _p2p_batch([('send', act_b.grad, rank - 1, 'bwd')], device)
            else:
                r = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')], device)
                fwd_act = r[0].requires_grad_(True)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out = stage_forward(blocks_local, fwd_act)
                mark_busy(t0)
                r = _p2p_batch([
                    ('send', out, rank + 1, 'fwd'),
                    ('recv', (mb, H), rank + 1, 'bwd'),
                ], device)
                bwd_grad_mid = r[0]
                live.append((fwd_act, out))
                peak = max(peak, len(live))
                act_b, out_b = live.pop(0)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out_b.backward(bwd_grad_mid)
                mark_busy(t0)
                _p2p_batch([('send', act_b.grad, rank - 1, 'bwd')], device)

            nf += 1
            nb += 1
            peak = max(peak, len(live))

        # cooldown backward
        for _ in range(warmup):
            act_b, out_b = live.pop(0)
            if is_last:
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out_b.backward()
                mark_busy(t0)
                _p2p_batch([('send', act_b.grad, rank - 1, 'bwd')], device)
            else:
                r = _p2p_batch([('recv', (mb, H), rank + 1, 'bwd')], device)
                g = r[0]
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out_b.backward(g)
                mark_busy(t0)
                if not is_first:
                    _p2p_batch([('send', act_b.grad, rank - 1, 'bwd')], device)
            nb += 1

    dist.barrier()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t_wall0
    return losses, peak, busy, wall, P2P['bytes'], dict(P2P)


def run(rank: int, world_size: int):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    lo = rank * BLOCKS_PER_STAGE
    is_first, is_last = rank == 0, rank == world_size - 1

    results = {}
    ref_losses_m4 = None
    for m in MICROBATCHES:
        ref_mb_losses, ref_mb_params = reference_run(m, per_mb=True, device=device)
        _, ref_full_params = reference_run(m, per_mb=False, device=device)
        ref_mb_slice = ref_mb_params[lo * 2:(lo + BLOCKS_PER_STAGE) * 2]
        ref_full_slice = ref_full_params[lo * 2:(lo + BLOCKS_PER_STAGE) * 2]
        if m == 4:
            ref_losses_m4 = ref_mb_losses

        for schedule in ('gpipe', '1f1b'):
            bubbles, peaks, walls, p2p_times = [], [], [], []
            losses_chk = None
            params_after = None
            comm_bytes = comm_counts = None

            for rep in range(WARMUP_REPEATS + REPEATS):
                blocks_all, X = build_blocks(device)
                blocks_local = [
                    [w.detach().clone().requires_grad_(True) for w in blocks_all[b]]
                    for b in range(lo, lo + BLOCKS_PER_STAGE)
                ]
                Ws = [w for blk in blocks_local for w in blk]
                opt = torch.optim.Adam(Ws, lr=LR)

                losses, peak, busy, wall, nbytes, counts = run_pipeline(
                    schedule, m, rank, world_size, blocks_local, X, device)
                opt.step()

                if rep >= WARMUP_REPEATS:
                    bubbles.append(1.0 - busy / wall if wall > 0 else 0.0)
                    peaks.append(peak)
                    walls.append(wall)
                    p2p_times.append(counts['time'])
                    if rep == WARMUP_REPEATS:
                        losses_chk = losses
                        params_after = [w.detach().clone() for w in Ws]
                        comm_bytes, comm_counts = nbytes, counts

            delta_mb_loss = 0.0
            if losses_chk:
                delta_mb_loss = max(abs(a - b) for a, b in
                                    zip([l.item() for l in losses_chk], ref_mb_losses))
            d_mirror = max((a - b).abs().max().item()
                           for a, b in zip(params_after, ref_mb_slice))
            d_full = max((a - b).abs().max().item()
                         for a, b in zip(params_after, ref_full_slice))
            full_scale = max(b.abs().max().item() for b in ref_full_slice)

            # 聚合通信计数：rank 0 发 fwd / 收 bwd，rank 1 收 fwd / 发 bwd
            counts_t = torch.tensor([
                comm_counts['send_fwd'], comm_counts['recv_fwd'],
                comm_counts['send_bwd'], comm_counts['recv_bwd'],
            ], dtype=torch.int64, device=device)
            dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)
            agg_counts = {
                'send_fwd': int(counts_t[0].item()),
                'recv_fwd': int(counts_t[1].item()),
                'send_bwd': int(counts_t[2].item()),
                'recv_bwd': int(counts_t[3].item()),
            }

            results[(m, schedule)] = {
                'delta_mb_loss': max_across_ranks(delta_mb_loss, device),
                'delta_vs_mirror': max_across_ranks(d_mirror, device),
                'delta_vs_fullbatch': max_across_ranks(d_full, device),
                'fullbatch_scale': max_across_ranks(full_scale, device),
                'bubble': sorted(bubbles)[len(bubbles) // 2],
                'peak': max_across_ranks(float(max(peaks)), device),
                'wall_ms': sorted(walls)[len(walls) // 2] * 1000,
                'p2p_ms': sorted(p2p_times)[len(p2p_times) // 2] * 1000,
                'comm_bytes': max_across_ranks(float(comm_bytes), device),
                'comm_counts': agg_counts,
                'params': params_after,
            }

    # ------------------------------------------------------------------
    # self-check
    # ------------------------------------------------------------------
    exp_bytes = 2 * (world_size - 1) * T * H * FP32
    exp_peak_gpipe = {m: m for m in MICROBATCHES}
    exp_peak_1f1b = {m: min(m, world_size) for m in MICROBATCHES}

    checks = []
    def ck(ok, msg):
        checks.append((bool(ok), msg))

    for m in MICROBATCHES:
        for schedule in ('gpipe', '1f1b'):
            r = results[(m, schedule)]
            full_tol = FULL_BATCH_ATOL + FULL_BATCH_RTOL * r['fullbatch_scale']
            ck(r['delta_vs_mirror'] == 0.0,
               f"m={m} {schedule}: params after step bit-identical to mirror reference")
            ck(r['delta_vs_fullbatch'] <= full_tol,
               f"m={m} {schedule}: params vs true full-batch ref within "
               f"atol+rtol*scale={full_tol:.2e} "
               f"(measured {r['delta_vs_fullbatch']:.2e}, fp32 归约形状差)")
            ck(r['comm_bytes'] == exp_bytes,
               f"m={m} {schedule}: p2p bytes/rank == 2(N-1)*T*H*4 = {exp_bytes}")
            exp_peak = exp_peak_gpipe[m] if schedule == 'gpipe' else exp_peak_1f1b[m]
            if schedule == 'gpipe':
                ck(int(round(r['peak'])) == exp_peak,
                   f"m={m} {schedule}: peak live mb == {exp_peak}")
            else:
                ck(int(round(r['peak'])) <= exp_peak,
                   f"m={m} {schedule}: peak live mb <= {exp_peak} (measured {int(round(r['peak']))})")

    # bubble 趋势
    bubble_m1_gpipe = results[(1, 'gpipe')]['bubble']
    bubble_m8_gpipe = results[(8, 'gpipe')]['bubble']
    ck(bubble_m1_gpipe > bubble_m8_gpipe + 0.05,
       f"bubble trend: m=1 ({bubble_m1_gpipe:.3f}) > m=8 ({bubble_m8_gpipe:.3f}) + 0.05")

    # 1f1b 与 gpipe 步后权重 bit 相同
    for m in MICROBATCHES:
        p_g = results[(m, 'gpipe')]['params']
        p_1 = results[(m, '1f1b')]['params']
        d = max((a - b).abs().max().item() for a, b in zip(p_g, p_1))
        ck(max_across_ranks(d, device) == 0.0,
           f"m={m}: params after step gpipe == 1f1b bit-identical")

    # 显存账本：跑一次 dummy forward/backward/step，让 grad 与 Adam state 就位
    blocks_all, X_ledger = build_blocks(device)
    blocks_local = [
        [w.detach().clone().requires_grad_(True) for w in blocks_all[b]]
        for b in range(lo, lo + BLOCKS_PER_STAGE)
    ]
    Ws = [w for blk in blocks_local for w in blk]
    opt = torch.optim.Adam(Ws, lr=LR)
    mb_ledger = T // 4
    out_ledger = stage_forward(blocks_local, X_ledger[:mb_ledger])
    out_ledger.sum().backward()
    opt.step()
    p_b = sum(p.numel() * p.element_size() for p in Ws)
    g_b = sum(p.grad.numel() * p.grad.element_size() for p in Ws)
    o_b = sum(v.numel() * v.element_size()
              for st in opt.state.values() for v in st.values()
              if torch.is_tensor(v) and v.dim() > 0)
    ledger_local = torch.tensor([p_b, g_b, o_b], dtype=torch.float64, device=device)
    ledger_all = [torch.zeros(3, dtype=torch.float64, device=device) for _ in range(world_size)]
    dist.all_gather(ledger_all, ledger_local)

    dense_total = P_TOTAL * 16
    sum_across = sum(l.sum().item() for l in ledger_all)
    ck(abs(sum_across - dense_total) < 1, "ledger: sum across ranks == 16P")
    for lg in ledger_all:
        ck(abs(lg[0].item() - P_TOTAL * 4 / world_size) < 1,
           "params per rank == P/N")
        ck(abs(lg[2].item() - 2 * lg[0].item()) < 1,
           "Adam state == 2x params")

    passed = sum(1 for ok, _ in checks if ok)
    total = len(checks)

    if rank == 0:
        tag = f"[PP={world_size} GPU/NCCL]"
        print("=" * 72)
        print("nano-megatron L2 — GPU empirical verification (NCCL)")
        print("=" * 72)
        print(f"model: {N_BLOCKS} x GeLU-MLP blocks (H={H}, FF={FF}) | P = {P_TOTAL} | fp32 | seed={SEED}")
        print(f"cluster: {world_size} ranks (nccl, GPU) | microbatches = {MICROBATCHES} | batch T = {T}")
        print(f"torch {torch.__version__}  cuda {torch.version.cuda}  nccl {torch.cuda.nccl.version()}")
        print(f"devices: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")

        r4 = results[(4, 'gpipe')]
        print(f"\n{tag} numerical equivalence (m=4, GPipe)")
        print(f"    per-mb losses vs mirror: max Δ = {r4['delta_mb_loss']:.1e}  {'✅' if r4['delta_mb_loss'] == 0.0 else '❌'}")
        print(f"    params after step vs mirror: max|Δ| = {r4['delta_vs_mirror']:.1e}  {'✅' if r4['delta_vs_mirror'] == 0.0 else '❌'}")
        r4_tol = FULL_BATCH_ATOL + FULL_BATCH_RTOL * r4['fullbatch_scale']
        print(f"    params vs true full-batch: max|Δ| = {r4['delta_vs_fullbatch']:.2e}  "
              f"{'✅' if r4['delta_vs_fullbatch'] <= r4_tol else '❌'} "
              f"(tol={r4_tol:.2e}, fp32 归约形状差)")
        print(f"    comm counters: "
              f"send_fwd={r4['comm_counts']['send_fwd']} recv_fwd={r4['comm_counts']['recv_fwd']} "
              f"send_bwd={r4['comm_counts']['send_bwd']} recv_bwd={r4['comm_counts']['recv_bwd']}")

        print(f"\n{tag} bubble vs formula (N-1)/(m+N-1)  [median over {REPEATS} repeats]")
        for m in MICROBATCHES:
            formula = (world_size - 1) / (m + world_size - 1) * 100
            bg = results[(m, 'gpipe')]['bubble'] * 100
            b1 = results[(m, '1f1b')]['bubble'] * 100
            print(f"    m={m}: gpipe={bg:5.1f}%  1f1b={b1:5.1f}%  (formula {formula:5.1f}%)")

        print(f"\n{tag} P2P wall-clock")
        print(f"    per-rank bytes/step = {exp_bytes} B = 2*(N-1)*T*H*4")
        for m in MICROBATCHES:
            r_g = results[(m, 'gpipe')]
            r_1 = results[(m, '1f1b')]
            print(f"    m={m}: wall gpipe={r_g['wall_ms']:.3f} ms  1f1b={r_1['wall_ms']:.3f} ms  "
                  f"p2p time gpipe={r_g['p2p_ms']:.3f} ms  1f1b={r_1['p2p_ms']:.3f} ms")

        print(f"\n{tag} peak live microbatches")
        for m in MICROBATCHES:
            print(f"    m={m}: GPipe={int(round(results[(m,'gpipe')]['peak']))}  "
                  f"1F1B={int(round(results[(m,'1f1b')]['peak']))} (≤N={world_size})")

        print(f"\n{tag} per-rank memory ledger")
        for r in range(world_size):
            pb, gb, ob = ledger_all[r].tolist()
            print(f"    rank {r}: params={fmt_kib(pb)} grads={fmt_kib(gb)} "
                  f"optimizer={fmt_kib(ob)} total={fmt_kib(pb + gb + ob)}")
        print(f"    dense replica = {fmt_kib(dense_total)} | "
              f"sum across ranks = {fmt_kib(sum_across)}")

        print(f"\n[5] self-check")
        for ok, msg in checks:
            print(f"    {'PASS' if ok else 'FAIL'}  {msg}")
        print(f"    {'✅ PASS' if passed == total else '❌ FAIL'} self-check ({passed}/{total})")

    digest_src = repr((
        round(results[(4, 'gpipe')]['delta_mb_loss'], 9),
        round(results[(4, 'gpipe')]['delta_vs_mirror'], 9),
        round(results[(4, 'gpipe')]['delta_vs_fullbatch'], 9),
        round(results[(4, 'gpipe')]['fullbatch_scale'], 9),
        results[(4, 'gpipe')]['comm_counts'],
        results[(4, '1f1b')]['comm_counts'],
        {m: int(round(results[(m, 'gpipe')]['peak'])) for m in MICROBATCHES},
        {m: int(round(results[(m, '1f1b')]['peak'])) for m in MICROBATCHES},
        int(round(results[(1, 'gpipe')]['comm_bytes'])),
    ))
    gpu_digest = hashlib.md5(digest_src.encode()).hexdigest()
    if rank == 0:
        print(f"\ngpu_digest = {gpu_digest}")

    return {
        'results': results,
        'checks': checks,
        'gpu_digest': gpu_digest,
        'ledger_all': [lg.tolist() for lg in ledger_all],
    }


def pp_worker(rank: int, world_size: int, port: int, results: dict):
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
        res = run(rank, world_size)
        if rank == 0:
            # 只放可序列化的标量/字符串，避免 CUDA tensor 进入 manager dict
            results['gpu_digest'] = res['gpu_digest']
            results['checks_ok'] = sum(1 for ok, _ in res['checks'] if ok)
            results['checks_total'] = len(res['checks'])
            results['ledger_sum_kib'] = sum(sum(x) for x in res['ledger_all']) / 1024
            r4 = res['results'][(4, 'gpipe')]
            results['mirror_max_abs'] = r4['delta_vs_mirror']
            results['fullbatch_max_abs'] = r4['delta_vs_fullbatch']
            results['fullbatch_tolerance'] = (
                FULL_BATCH_ATOL + FULL_BATCH_RTOL * r4['fullbatch_scale']
            )
            results['p2p_bytes_per_rank'] = int(round(r4['comm_bytes']))
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify nano Megatron PP on two CUDA GPUs.")
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("NANO_MEGATRON_MASTER_PORT", "29541")),
        help="localhost rendezvous port (or NANO_MEGATRON_MASTER_PORT)",
    )
    args = parser.parse_args()
    if not 1024 <= args.master_port <= 65535:
        parser.error("--master-port must be in [1024, 65535]")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available; this verification must run on a GPU machine.")
    if torch.cuda.device_count() < WORLD_SIZE:
        sys.exit(f"Need at least {WORLD_SIZE} GPUs; found {torch.cuda.device_count()}.")

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(pp_worker, args=(WORLD_SIZE, args.master_port, results), nprocs=WORLD_SIZE, join=True)

    success = results["checks_ok"] == results["checks_total"]
    print("\n" + "=" * 72)
    print(
        ("✅ GPU self-check passed: " if success else "❌ GPU self-check failed: ")
        + "PP matches the same-microbatch mirror; P2P bytes and memory ledger are checked; "
        + "measured bubble must only follow the ideal formula's direction."
    )
    print("=" * 72)
    print(f"gpu_digest: {results['gpu_digest']}")
    print(f"self-check: {results['checks_ok']}/{results['checks_total']}")
    payload = {
        "schema_version": "1.0",
        "module": "nano-megatron-l2-gpu",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": str(torch.cuda.nccl.version()),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        },
        "checks": {
            "passed": results["checks_ok"],
            "total": results["checks_total"],
            "gpu_digest": results["gpu_digest"],
        },
        "metrics": {
            "fullbatch_max_abs": results["fullbatch_max_abs"],
            "fullbatch_tolerance": results["fullbatch_tolerance"],
            "mirror_max_abs": results["mirror_max_abs"],
            "p2p_bytes_per_rank": results["p2p_bytes_per_rank"],
        },
        "evidence_boundary": (
            "single-host PP=2 toy; same-microbatch mirror is the primary correctness oracle; "
            "full-batch uses atol+rtol because FP32 reduction shape changes; timing and bubble "
            "magnitudes are stack-specific"
        ),
    }
    print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
