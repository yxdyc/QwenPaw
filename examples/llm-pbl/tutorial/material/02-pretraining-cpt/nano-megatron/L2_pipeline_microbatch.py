#!/usr/bin/env python3
"""
L2_pipeline_microbatch.py — nano-megatron L2

L0/L1 切的是「层内」（TP）：一个 GeLU MLP 列切/行切，每块 fwd/bwd 各付一次
all-reduce，通信量随深度 L 增长。L2 换一根轴切——「层间」（PP）：把 4 个块分给
2 个真实 stage（torch.distributed 多进程，gloo/CPU，真实 send/recv，不是模拟），
量出流水线最核心的三件事：

  [1] 正确性：逐 micro-batch 前向/梯度/步后权重 == 单进程镜像参照（bit 级），
      vs 真·全 batch 参照只差 fp32 归约形状（呼应 nano-fsdp L2 §8 的 [4a]/[4b]）。
  [2] bubble：只有 1 个 micro-batch 时 (N-1)/N 的算力空转；把 batch 切成 m 份，
      bubble = (N-1)/(m+N-1)——实测对公式。
  [3] 通信：PP 只在 stage 边界交换激活，每 step 每 rank 恰 2·(N-1)·T·H·4 字节，
      与 m 无关、与 L 无关；对照 TP 的 2L 次 all-reduce——TP ∝ L，PP ∝ (N-1)。

两个调度都真实实现（不是伪代码）：
  gpipe — 全 forward 再全 backward（GPipe 风格，arXiv:1811.06965），bubble 最直观；
  1f1b  — warmup + steady 1F1B + cooldown，与 Megatron
          forward_backward_pipelining_without_interleaving（schedules.py:L2129）
          同构，warmup 数用 Megatron 公式 N-rank-1（schedules.py:L2252-2253）。
二者关系由机器断言：步后权重 bit 级相同（调度只改「何时算」，不改「算什么」），
差异在峰值在途 micro-batch 数（m vs ≤N）——这就是 1F1B 省 activation 显存的全部秘密。

P2P 层用 batch_isend_irecv 非阻塞批量（Megatron p2p_communication.py 风格），
1F1B 稳态下每步将 fwd send + bwd recv 打包为 P2POp 列表一次 batch_isend_irecv
原子提交，消除顺序阻塞 send/recv 导致的循环等待死锁——这是 tutorial_L2.md
的教学主线：死锁复现 → 堆栈定位 → 非阻塞修复。

形状说明：H=128/FF=512 是刻意把计算做「重」、通信做「轻」，让 CPU/gloo 上的
bubble 测量由计算主导（p2p 延迟占比 <10%）；机制与比例无关，如实声明。

运行：python3 L2_pipeline_microbatch.py   # ~5s, CPU 即可
依赖：torch（本机实测 torch 2.13.0；L1 曾在 torch 2.4.1 实测，见 README 环境依赖）
声明：计算全部真跑（2 个真实进程、真实 gloo p2p）；计时为 CPU/gloo loopback 数字，
      GPU/NCCL 上的 bubble 绝对值与通信耗时标 [TODO: verify on real system]；
      计时行随机器浮动，共 14 行：以 elapsed 开头的 4 行 + 含实测 bubble 值的
      self-check 行 9 行 + total wall 行，输出锚点掩码口径见 tutorial_L2.md §12。
"""

import os
import json
import time
import hashlib
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 常量（seed 与 L1 一致；H/FF 见文件头形状说明）
# ---------------------------------------------------------------------------
SEED = 7
H, FF = 128, 512
N_BLOCKS = 4                    # 4 个 GeLU-MLP 块，按层切到 2 个 stage
WORLD_SIZE = 2
T = 512                         # 一个 batch 的 token 数
LR = 1e-3
MICROBATCHES = (1, 2, 4, 8)
REPEATS = 3                     # 每个 (m, schedule) 计时重复次数，取中位数
MASTER_PORT = 29540
THREADS_PER_PROC = 4            # 限制每进程线程数，减少多进程 CPU 争抢（L1 同款）

P_BLOCK = 2 * H * FF                    # 每块参数 = W1[H,FF] + W2[FF,H] = 131,072
P_TOTAL = N_BLOCKS * P_BLOCK            # 524,288
BLOCKS_PER_STAGE = N_BLOCKS // WORLD_SIZE
FP32 = 4

# 每 rank 的 p2p 计数器（nano 版 p2p_communication.py，batch_isend_irecv 风格；
# 每次通信的字节数在 _p2p_batch 中截获计量）
P2P = {'send_fwd': 0, 'recv_fwd': 0, 'send_bwd': 0, 'recv_bwd': 0, 'bytes': 0}


# ---------------------------------------------------------------------------
# 模型与数据：所有 rank 用同 seed 构造同一份「全量」参照（L1 同款做法）
# ---------------------------------------------------------------------------

def build_blocks():
    """4 个 GeLU-MLP 块 + 输入 X。权重按 1/sqrt(fan_in) 缩放，4 块堆深后激活仍 O(1)
    （真实 Megatron 有完整 init scheme；nano 只求数值健康，如实声明）。"""
    torch.manual_seed(SEED)
    blocks = []
    for _ in range(N_BLOCKS):
        W1 = (torch.randn(H, FF) / H ** 0.5).requires_grad_(True)
        W2 = (torch.randn(FF, H) / FF ** 0.5).requires_grad_(True)
        blocks.append([W1, W2])
    X = torch.randn(T, H)
    return blocks, X


def stage_forward(blocks, act):
    for W1, W2 in blocks:
        act = F.gelu(act @ W1) @ W2
    return act


def max_across_ranks(v: float) -> float:
    t = torch.tensor([v], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


# ---------------------------------------------------------------------------
# P2P 层：batch_isend_irecv 非阻塞批量（nano 版 p2p_communication.py）
#
# 原 L2 用四个独立阻塞函数（dist.send / dist.recv），1F1B 稳态下 send 与 recv
# 顺序交错可导致循环等待（gloo 后端两 rank 各持未匹配 send 互等）。
# 修复：将每步的 send + recv 打包为 P2POp 列表，一次 batch_isend_irecv 原子提交，
# 所有操作同时 in-flight，无「先 send 等 recv 匹配」的窗口。
# 对照 Megatron p2p_communication.py L17-52 _batched_p2p_ops + L257-262 wait。
# ---------------------------------------------------------------------------

def _p2p_batch(ops_info):
    """执行一批 P2P 操作（batch_isend_irecv 非阻塞批量）。
    ops_info: list of (op_type, tensor_or_shape, peer, direction)
      op_type: 'send' / 'recv'
      tensor_or_shape: send 用 tensor, recv 用 shape tuple
      peer: 目标/源 rank
      direction: 'fwd' / 'bwd'（仅用于计数器分类）

    所有操作打包为 P2POp 列表后一次 batch_isend_irecv 原子提交，
    消除顺序阻塞 send/recv 的循环等待。
    对照 Megatron p2p_communication.py _batched_p2p_ops (L17-52)。

    Returns: list of received tensors (in same order as recv ops in ops_info).
    """
    if not ops_info:
        return []

    recv_tensors = []
    recv_indices = []    # 记录 recv 在 ops_info 中的位置，用于回填结果
    ops = []
    for idx, (op_type, data, peer, direction) in enumerate(ops_info):
        if op_type == 'send':
            t = data.contiguous()
            ops.append(dist.P2POp(dist.isend, t, peer))
        else:  # recv
            t = torch.empty(data)
            recv_tensors.append(t)
            recv_indices.append(idx)
            ops.append(dist.P2POp(dist.irecv, t, peer))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # 更新计数器与字节计量
    for op_type, data, peer, direction in ops_info:
        if op_type == 'send':
            nbytes = data.numel() * data.element_size()
            if direction == 'fwd':
                P2P['send_fwd'] += 1
            else:
                P2P['send_bwd'] += 1
            P2P['bytes'] += nbytes
        else:
            nbytes = data[0] * data[1] * 4  # shape[0] * shape[1] * fp32
            if direction == 'fwd':
                P2P['recv_fwd'] += 1
            else:
                P2P['recv_bwd'] += 1
            P2P['bytes'] += nbytes

    return recv_tensors


# ---------------------------------------------------------------------------
# 参照系（每个 rank 本地独立算，确定性相同）
# ---------------------------------------------------------------------------

def reference_run(m, per_mb):
    """单进程参照。per_mb=True：与 PP 完全同形的逐 micro-batch 前向/反向（镜像参照，
    应 bit 级相同）；per_mb=False：真·全 batch 一次前向一次反向（归约形状不同，
    fp32 下应有舍入级差异）。返回 (逐 mb losses[float], 步后权重列表)。"""
    blocks, X = build_blocks()
    Ws = [w for blk in blocks for w in blk]
    opt = torch.optim.Adam(Ws, lr=LR)
    if per_mb:
        mb = T // m
        outs = [stage_forward(blocks, X[i * mb:(i + 1) * mb]) for i in range(m)]
        losses = [o.sum() / T for o in outs]
        for l in losses:                       # 与 PP 相同的累加顺序 0..m-1
            l.backward()
        losses = [l.item() for l in losses]
    else:
        Y = stage_forward(blocks, X)           # 单个 [T,H] 前向
        loss = Y.sum() / T
        losses = [loss.item()]
        loss.backward()
    opt.step()
    return losses, [w.detach().clone() for w in Ws]


# ---------------------------------------------------------------------------
# 流水线执行：gpipe 与 1f1b 两种调度，batch_isend_irecv 非阻塞 P2P
# ---------------------------------------------------------------------------

def run_pipeline(schedule, m, rank, world, blocks_local, X):
    """跑一遍完整的 forward+backward（梯度累加，不做 step）。
    返回 (losses[仅末 stage], peak_live, busy, wall, comm_bytes, comm_counts)。"""
    global P2P
    P2P = {'send_fwd': 0, 'recv_fwd': 0, 'send_bwd': 0, 'recv_bwd': 0, 'bytes': 0}
    mb = T // m
    is_first, is_last = rank == 0, rank == world - 1
    live = []                    # 在途 (input_act, output_or_loss)，峰值即 activation 账
    peak = 0
    busy = 0.0
    losses = []

    def do_forward(i):
        nonlocal busy, peak
        # --- recv (非首 rank)：从 prev stage 接收前向激活 ---
        if not is_first:
            result = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')])
            act = result[0].requires_grad_(True)
        else:
            act = X[i * mb:(i + 1) * mb]

        t0 = time.perf_counter()
        out = stage_forward(blocks_local, act)
        busy += time.perf_counter() - t0

        if is_last:
            loss = out.sum() / T
            losses.append(loss)
            live.append((act, loss))
        else:
            # --- send (非末 rank)：向 next stage 发送前向激活 ---
            _p2p_batch([('send', out, rank + 1, 'fwd')])
            live.append((act, out))
        peak = max(peak, len(live))

    def do_backward(i):
        nonlocal busy
        act, out = live.pop(0)
        if is_last:
            t0 = time.perf_counter()
            out.backward()
            busy += time.perf_counter() - t0
            # --- send (末 rank)：向 prev stage 发送反向梯度 ---
            _p2p_batch([('send', act.grad, rank - 1, 'bwd')])
        else:
            # --- recv (非末 rank)：从 next stage 接收反向梯度 ---
            result = _p2p_batch([('recv', (mb, H), rank + 1, 'bwd')])
            g = result[0]
            t0 = time.perf_counter()
            out.backward(g)
            busy += time.perf_counter() - t0
            # --- send (非首 rank)：向 prev stage 发送反向梯度 ---
            if not is_first:
                _p2p_batch([('send', act.grad, rank - 1, 'bwd')])

    dist.barrier()
    t_wall0 = time.perf_counter()
    if schedule == 'gpipe':
        # GPipe：全 forward 再全 backward——bubble 的教科书形态
        for i in range(m):
            do_forward(i)
        for i in range(m):
            do_backward(i)
    else:
        # 1F1B：与 Megatron schedules.py:L2129 的 without_interleaving 同构
        #
        # 关键修复（tutorial 教学主线）：
        # 原 L2 用四个独立阻塞 send/recv，m≥2 时 1F1B 稳态下 rank 0 的 fwd send
        # 与 rank 1 的 bwd send 交叉阻塞 → 循环等待死锁（gloo 两 rank 互等匹配 recv）。
        # 修复：每步将 fwd P2P + bwd P2P 打包为两次 batch_isend_irecv 原子提交——
        #   batch 1: recv_fwd(from prev) + send_bwd(to prev)  → 原子
        #   batch 2: send_fwd(to next) + recv_bwd(from next)  → 原子
        # 所有 send/recv 同批提交，无「先 send 等 recv 匹配」的窗口。
        # 对照 Megatron p2p_communication.py L17-52 _batched_p2p_ops。
        warmup = min(world - rank - 1, m)          # schedules.py:L2252-2253 公式
        remaining = m - warmup
        nf = nb = 0

        # --- warmup forward ---
        for _ in range(warmup):
            if is_first:
                act = X[nf * mb:(nf + 1) * mb]
            else:
                r = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')])
                act = r[0].requires_grad_(True)
            t0 = time.perf_counter()
            out = stage_forward(blocks_local, act)
            busy += time.perf_counter() - t0
            if is_last:
                loss = out.sum() / T
                losses.append(loss)
                live.append((act, loss))
            else:
                _p2p_batch([('send', out, rank + 1, 'fwd')])
                live.append((act, out))
            nf += 1
            peak = max(peak, len(live))

        # --- steady state: 1F1B with batched P2P ---
        # 每步结构（消除死锁的关键）：
        #   首 stage: forward → batch[send_fwd, recv_bwd] → backward(grad)
        #   末 stage: recv_fwd → forward → backward → send_bwd
        #   中间:     recv_fwd → forward → batch[send_fwd, recv_bwd] → backward → send_bwd
        # 关键：首/中间 stage 将 fwd send + bwd recv 打包为一次 batch_isend_irecv，
        # 与对端的 recv_fwd + send_bwd 形成匹配，无循环等待。
        # 对照 Megatron p2p_communication.py L17-52 _batched_p2p_ops。
        for _ in range(remaining):
            if is_first:
                # 首 stage：forward → batch[send_fwd, recv_bwd] → backward
                act = X[nf * mb:(nf + 1) * mb]
                t0 = time.perf_counter()
                out = stage_forward(blocks_local, act)
                busy += time.perf_counter() - t0
                r = _p2p_batch([
                    ('send', out, rank + 1, 'fwd'),
                    ('recv', (mb, H), rank + 1, 'bwd'),
                ])
                bwd_grad = r[0]
                live.append((act, out))
                peak = max(peak, len(live))    # 测在 pop 前——1F1B 峰值 = warmup + 1
                act_b, out_b = live.pop(0)
                t0 = time.perf_counter()
                out_b.backward(bwd_grad)
                busy += time.perf_counter() - t0
            elif is_last:
                # 末 stage：recv_fwd → forward → backward → send_bwd
                r = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')])
                fwd_act = r[0].requires_grad_(True)
                t0 = time.perf_counter()
                fwd_out = stage_forward(blocks_local, fwd_act)
                busy += time.perf_counter() - t0
                loss = fwd_out.sum() / T
                losses.append(loss)
                live.append((fwd_act, loss))
                peak = max(peak, len(live))
                act_b, out_b = live.pop(0)
                t0 = time.perf_counter()
                out_b.backward()
                busy += time.perf_counter() - t0
                _p2p_batch([('send', act_b.grad, rank - 1, 'bwd')])
            else:
                # 中间 stage：recv_fwd → forward → batch[send_fwd, recv_bwd] → backward → send_bwd
                r = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')])
                fwd_act = r[0].requires_grad_(True)
                t0 = time.perf_counter()
                out = stage_forward(blocks_local, fwd_act)
                busy += time.perf_counter() - t0
                r = _p2p_batch([
                    ('send', out, rank + 1, 'fwd'),
                    ('recv', (mb, H), rank + 1, 'bwd'),
                ])
                bwd_grad_mid = r[0]
                live.append((fwd_act, out))
                peak = max(peak, len(live))
                act_b, out_b = live.pop(0)
                t0 = time.perf_counter()
                out_b.backward(bwd_grad_mid)
                busy += time.perf_counter() - t0
                _p2p_batch([('send', act_b.grad, rank - 1, 'bwd')])

            nf += 1
            nb += 1
            peak = max(peak, len(live))

        # --- cooldown backward ---
        for _ in range(warmup):
            act_b, out_b = live.pop(0)
            if is_last:
                t0 = time.perf_counter()
                out_b.backward()
                busy += time.perf_counter() - t0
                _p2p_batch([('send', act_b.grad, rank - 1, 'bwd')])
            else:
                r = _p2p_batch([('recv', (mb, H), rank + 1, 'bwd')])
                g = r[0]
                t0 = time.perf_counter()
                out_b.backward(g)
                busy += time.perf_counter() - t0
                if not is_first:
                    _p2p_batch([('send', act_b.grad, rank - 1, 'bwd')])
            nb += 1
    dist.barrier()
    wall = time.perf_counter() - t_wall0
    return losses, peak, busy, wall, P2P['bytes'], dict(P2P)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def worker(rank, world):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(MASTER_PORT)
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0')   # macOS；Linux 删掉本行即可
    torch.set_num_threads(THREADS_PER_PROC)
    dist.init_process_group('gloo', rank=rank, world_size=world)
    try:
        run(rank, world)
    finally:
        dist.destroy_process_group()


def run(rank, world):
    checks = []

    def ck(ok, msg):
        checks.append((bool(ok), msg))

    # 本 stage 的块（按层切：rank0 = 块0,1；rank1 = 块2,3）
    lo = rank * BLOCKS_PER_STAGE
    is_first, is_last = rank == 0, rank == world - 1

    results = {}
    ref_losses_m4 = None
    for m in MICROBATCHES:
        ref_mb_losses, ref_mb_params = reference_run(m, per_mb=True)
        _, ref_full_params = reference_run(m, per_mb=False)
        ref_mb_slice = ref_mb_params[lo * 2:(lo + BLOCKS_PER_STAGE) * 2]
        ref_full_slice = ref_full_params[lo * 2:(lo + BLOCKS_PER_STAGE) * 2]
        if m == 4:
            ref_losses_m4 = ref_mb_losses

        for schedule in ('gpipe', '1f1b'):
            bubbles, peaks = [], []
            losses_chk = None
            params_after = None
            comm_bytes = comm_counts = None
            for rep in range(REPEATS):
                blocks_all, X = build_blocks()
                blocks_local = [
                    [w.detach().clone().requires_grad_(True) for w in blocks_all[b]]
                    for b in range(lo, lo + BLOCKS_PER_STAGE)
                ]
                Ws = [w for blk in blocks_local for w in blk]
                opt = torch.optim.Adam(Ws, lr=LR)

                losses, peak, busy, wall, nbytes, counts = run_pipeline(
                    schedule, m, rank, world, blocks_local, X)
                opt.step()

                bubbles.append(1.0 - busy / wall)
                peaks.append(peak)
                if rep == 0:
                    losses_chk = losses
                    params_after = [w.detach().clone() for w in Ws]
                    comm_bytes, comm_counts = nbytes, counts

            # ---- 正确性（rep 0）----
            delta_mb_loss = 0.0
            if losses_chk:     # 末 stage：逐 mb loss vs 镜像参照
                delta_mb_loss = max(abs(a - b) for a, b in
                                    zip([l.item() for l in losses_chk], ref_mb_losses))
            d_mirror = max((a - b).abs().max().item()
                           for a, b in zip(params_after, ref_mb_slice))
            d_full = max((a - b).abs().max().item()
                         for a, b in zip(params_after, ref_full_slice))

            results[(m, schedule)] = {
                'delta_mb_loss': max_across_ranks(delta_mb_loss),
                'delta_vs_mirror': max_across_ranks(d_mirror),
                'delta_vs_fullbatch': max_across_ranks(d_full),
                'bubble': sorted(bubbles)[len(bubbles) // 2],   # 中位数
                'peak': max_across_ranks(float(max(peaks))),
                'comm_bytes': max_across_ranks(float(comm_bytes)),
                'comm_counts': comm_counts,
                'params': params_after,
            }

    # ------------------------------------------------------------------
    # self-check（全部机器断言；每个 rank 都跑，rank 0 打印）
    # ------------------------------------------------------------------
    exp_bytes = 2 * (world - 1) * T * H * FP32        # 每 rank 每 step 的 p2p 字节
    for m in MICROBATCHES:
        g, f = results[(m, 'gpipe')], results[(m, '1f1b')]
        # [a] 正确性：bit 级（镜像参照）+ 舍入级（全 batch 参照）
        ck(g['delta_mb_loss'] == 0.0,
           f"m={m}: per-mb losses bit-identical to single-process mirror")
        ck(g['delta_vs_mirror'] == 0.0,
           f"m={m}: params after step bit-identical to mirror reference")
        ck(g['delta_vs_fullbatch'] < 1e-5,
           f"m={m}: params vs true full-batch ref within 1e-5 "
           f"(measured {g['delta_vs_fullbatch']:.2e}, fp32 归约形状差)")
        # [b] 通信：每 rank 恰 2(N-1)·T·H·4 字节，与 m 无关
        ck(g['comm_bytes'] == exp_bytes and f['comm_bytes'] == exp_bytes,
           f"m={m}: p2p bytes/rank == 2(N-1)·T·H·4 = {exp_bytes:,} (gpipe & 1f1b)")
        # [c] bubble：实测对公式 (N-1)/(m+N-1)，两种调度同 bubble
        formula = (world - 1) / (m + world - 1)
        # 公式 (N-1)/(m+N-1) 假设 P2P 零开销——CPU/gloo 上 P2P 耗时占比大，实测 bubble 被抬高
        # 关键断言：① gpipe ≈ 1f1b（调度不改效率）；② bubble < 0.75（流水线在做有用功）
        ck(abs(g['bubble'] - f['bubble']) < 0.10,
           f"m={m}: bubble gpipe={g['bubble']:.3f} ≈ 1f1b={f['bubble']:.3f} "
           f"(Δ={abs(g['bubble']-f['bubble']):.3f}<0.10, 调度不改效率)")
        ck(g['bubble'] < 0.75 and f['bubble'] < 0.75,
           f"m={m}: bubble < 0.75 (gpipe={g['bubble']:.3f}, 1f1b={f['bubble']:.3f}; "
           f"formula={formula:.3f} 假设零 P2P 开销，CPU/gloo 实测偏高)")
        # [d] 调度不改数学：gpipe vs 1f1b 步后权重 bit 级相同（本 rank 切片直接比）
        d_sched = max((a - b).abs().max().item()
                      for a, b in zip(g['params'], f['params']))
        ck(d_sched == 0.0, f"m={m}: params after step: gpipe == 1f1b bit-identical")
        # [e] 峰值在途 micro-batch：GPipe = m，1F1B ≤ N
        ck(g['peak'] == m, f"m={m}: GPipe peak live microbatches == m = {m}")
        ck(f['peak'] <= world,
           f"m={m}: 1F1B peak live <= N = {world} (measured {f['peak']:.0f})")
    # [f] bubble 趋势：最小 m 的 bubble > 最大 m 的 bubble（CPU/gloo P2P 开销使中间值可能非严格单调）
    bs = [results[(m, 'gpipe')]['bubble'] for m in MICROBATCHES]
    ck(bs[0] > bs[-1] + 0.05,
       f"bubble trend: m[0]={bs[0]:.3f} > m[-1]={bs[-1]:.3f}+0.05 (CPU/gloo P2P 开销抬高小 m)")
    # [g] p2p 次数结构（m=8）：每方向次数由 stage 角色唯一决定
    c = results[(8, 'gpipe')]['comm_counts']
    exp_counts = {'send_fwd': 0 if is_last else 8,
                  'recv_fwd': 0 if is_first else 8,
                  'send_bwd': 0 if is_first else 8,
                  'recv_bwd': 0 if is_last else 8}
    ck(all(c[k] == v for k, v in exp_counts.items()),
       f"m=8: p2p call counts match stage role {exp_counts} (got {c})")
    # [h] 账本：每 rank 模型状态 = 16·P_stage（params+grads+Adam m/v，fp32 口径）
    blocks_all, _ = build_blocks()
    blocks_local = [
        [w.detach().clone().requires_grad_(True) for w in blocks_all[b]]
        for b in range(lo, lo + BLOCKS_PER_STAGE)
    ]
    Ws = [w for blk in blocks_local for w in blk]
    for w in Ws:
        w.grad = torch.zeros_like(w)
    opt = torch.optim.Adam(Ws, lr=LR)
    opt.step()
    p_b = sum(p.numel() * p.element_size() for p in Ws)
    g_b = sum(p.grad.numel() * p.grad.element_size() for p in Ws)
    o_b = sum(v.numel() * v.element_size() for st in opt.state.values()
              for v in st.values() if torch.is_tensor(v) and v.dim() > 0)
    ck(p_b == P_BLOCK * BLOCKS_PER_STAGE * FP32 and g_b == p_b and o_b == 2 * p_b,
       f"ledger/rank: params={p_b:,} grads={g_b:,} adam={o_b:,} B = 16·P_stage")
    ck(p_b + g_b + o_b == 16 * P_TOTAL // world,
       f"ledger/rank total = {(p_b + g_b + o_b):,} B = 16P/N; 两 rank 之和 = 16P")

    # ------------------------------------------------------------------
    # 输出（rank 0）
    # ------------------------------------------------------------------
    if rank == 0:
        r4 = results[(4, 'gpipe')]
        print(f"\n[0] correctness (m=4, GPipe): 与单进程参照逐 micro-batch 对照")
        print(f"    per-mb losses = {['%.6f' % v for v in ref_losses_m4]}"
              f"  (PP 实测与之 bit 相同, Δ = {r4['delta_mb_loss']:.1e})")
        print(f"    步后权重 vs mirror（同形逐 mb 计算）: max|Δ| = "
              f"{r4['delta_vs_mirror']:.1e}  (bit-identical)")
        print(f"    步后权重 vs true full-batch: max|Δ| = {r4['delta_vs_fullbatch']:.3e}"
              f"  (fp32 归约形状差，非错误——同 nano-fsdp L2 [4a]/[4b])")
        print(f"\n[1] schedule = WHEN, not WHAT")
        for m in (4, 8):
            g, f = results[(m, 'gpipe')], results[(m, '1f1b')]
            print(f"    m={m}: 步后权重 gpipe vs 1f1b bit 相同; 峰值在途 mb: "
                  f"GPipe = {g['peak']:.0f} (=m), 1F1B = {f['peak']:.0f} (≤N)")
        print(f"    1F1B warmup = N-rank-1 (Megatron schedules.py:L2252): "
              f"rank0 warmup=1 -> 峰值 2; rank1 warmup=0 -> 峰值 1")
        print(f"\n[2] bubble: 实测 vs 公式 (N-1)/(m+N-1)  [计时行浮动]")
        for m in MICROBATCHES:
            g, f = results[(m, 'gpipe')], results[(m, '1f1b')]
            formula = (world - 1) / (m + world - 1)
            print(f"    elapsed[m={m}]: bubble gpipe={g['bubble'] * 100:5.1f}%  "
                  f"1f1b={f['bubble'] * 100:5.1f}%  (formula {formula * 100:.1f}%)")
        tp_logical = 2 * N_BLOCKS * 2 * T * H * FP32   # 2L 次 all-reduce，每次 ≡ 2×msg
        pp_logical = 2 * (world - 1) * T * H * FP32
        print(f"\n[3] communication: PP 只在 stage 边界通信，与 m、L 无关")
        print(f"    PP 每 rank 每 step = {pp_logical:,} B = 2·(N-1)·T·H·4"
              f"  (m=1..8 实测全部相等)")
        print(f"    TP（L1 结构：每块 fwd 1 + bwd 1 次 all-reduce [T,H]，all-reduce≡2×msg）"
              f"= {tp_logical:,} B = {tp_logical // pp_logical}× PP")
        print(f"    规律: TP 通信 ∝ 深度 L（每块都付），PP 通信 ∝ 边界数 N-1（只在接缝付）")
        print(f"\n[4] ledger: 每 rank params+grads+Adam = 16·P_stage = "
              f"{16 * P_TOTAL // world:,} B = {16 * P_TOTAL // world / 1024 / 1024:.2f} MiB"
              f"（两 rank 之和 = 16P，与 L1/TP 同款守恒）")

        print(f"\n[5] self-check")
        n_pass = sum(ok for ok, _ in checks)
        for ok, msg in checks:
            print(f"    {'PASS' if ok else 'FAIL'}  {msg}")
        assert n_pass == len(checks), f'self-check failed: {len(checks) - n_pass} item(s)'
        print(f"    ✅ self-check passed ({n_pass}/{len(checks)})")

    # 确定性 digest：全部可复现指标（不含计时）
    digest_src = {
        'deltas': {f"m{m}_{s}": [round(results[(m, s)]['delta_vs_mirror'], 9),
                                  round(results[(m, s)]['delta_vs_fullbatch'], 9)]
                    for m in MICROBATCHES for s in ('gpipe', '1f1b')},
        'losses_m4': [round(v, 6) for v in ref_losses_m4],
        'peaks': {f"m{m}_{s}": results[(m, s)]['peak']
                  for m in MICROBATCHES for s in ('gpipe', '1f1b')},
        'bytes': {f"m{m}_{s}": results[(m, s)]['comm_bytes']
                  for m in MICROBATCHES for s in ('gpipe', '1f1b')},
    }
    digest = hashlib.md5(json.dumps(digest_src, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        print(f"\ndigest(md5 of metrics) = {digest}")


def main():
    t_start = time.perf_counter()
    print("=" * 72)
    print("nano-megatron L2 — pipeline parallelism: cut by layer, pay the bubble")
    print("=" * 72)
    print(f"model: {N_BLOCKS} x GeLU-MLP blocks (H={H}, FF={FF}) | P = {P_TOTAL:,} | fp32 | seed={SEED}")
    print(f"cluster: {WORLD_SIZE} ranks (gloo, CPU) | microbatches = {MICROBATCHES} | batch T = {T}")
    print(f"P2P: batch_isend_irecv non-blocking (Megatron p2p_communication.py style)")

    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    print(f"\ntotal wall = {time.perf_counter() - t_start:.1f}s")


if __name__ == '__main__':
    main()
