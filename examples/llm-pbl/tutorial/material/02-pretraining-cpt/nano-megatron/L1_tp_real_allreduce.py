#!/usr/bin/env python3
"""
L1_tp_real_allreduce.py — nano-megatron L1

L0 用「矩阵逐元素和」模拟 all-reduce 的语义；L1 付出 all-reduce 的真实代价：
    用 torch.distributed（gloo backend，CPU 多进程）把同一个 GeLU MLP 真正切开，
    ① 前向：W1 列切 + W2 行切，1 次真实 dist.all_reduce，与 dense 数值等价；
    ② 反向：手写 f/g 两个 autograd 算子（Megatron mappings.py 的同构物），
       验证「前向 1 次 all-reduce（g）+ 反向 1 次 all-reduce（f）」，
       且每个 rank 的权重分片梯度无需额外通信就是完整的；
    ③ 反例搬上真实通信：行切 W1 的 naive 版照样错，fix 版对但消息量 4x；
    ④ 账本实测：每 rank 真实持有的 params/grads/Adam 状态字节 = 16P/N；
    ⑤ 计时：gloo loopback 上不同消息大小的 all-reduce 真实墙钟，对照 L0 账本。

运行要求：torch（CPU 即可）。本机命令：
    python L1_tp_real_allreduce.py

可运行性声明（课程可运行性契约）：本脚本是真实分布式通信（gloo 多进程），
不是模拟；GPU/NVLink 上的真实耗时属 真实 GPU/多机环境，
标 [TODO: verify on real system]，本脚本不冒充。
"""

import os
import time
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

SEED = 7
H, FF, T = 64, 256, 128        # hidden, ffn (=4h, 真实 Transformer 常见比例), tokens (=b*s)
MLP_PARAMS = 2 * H * FF        # W1[h,f] + W2[f,h] = 32,768
WARMUP, ITERS = 2, 5           # 计时用
THREADS_PER_PROC = 4           # 限制每进程线程数，减少多进程 CPU 争抢

# 每 rank 的集合通信计数器（只计 f/g 算子内部的 all-reduce，
# 与 Megatron「每块 fwd 1 + bwd 1」的核算口径一致）
COMM = {"fwd": 0, "bwd": 0}


# ================= f/g 算子：Megatron mappings.py 的同构最小版 =================
# 对照（NVIDIA/Megatron-LM main, 2026-08-05 抓取）：
#   f = _CopyToModelParallelRegion  (mappings.py:L201, fwd identity / bwd all-reduce)
#   g = _ReduceFromModelParallelRegion (mappings.py:L221, fwd all-reduce / bwd identity)

class CopyToTensorParallelRegion(torch.autograd.Function):
    """f 算子：前向 identity（输入本就各 rank 复制），反向 all-reduce 汇合 dX 部分和。"""

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
    """g 算子：前向 all-reduce 行并行的部分和，反向 identity（dY 各 rank 相同）。"""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        COMM["fwd"] += 1
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ================= 确定性数据：所有 rank 用同 seed 构造同一份「全量」参照 =================

def build_shared_tensors():
    torch.manual_seed(SEED)
    W1 = torch.randn(H, FF)
    W2 = torch.randn(FF, H)
    X = torch.randn(T, H)
    return W1, W2, X


def max_across_ranks(v: float) -> float:
    """把各 rank 的标量取 MAX 同步（误差类指标各 rank 应一致，取 MAX 更稳）。"""
    t = torch.tensor([v], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def fmt_kib(nbytes: float) -> str:
    return f"{nbytes / 1024:.1f} KiB"


# ================= 主流程 =================

def run(rank: int, world_size: int, full_suite: bool):
    W1, W2, X = build_shared_tensors()
    assert FF % world_size == 0 and H % world_size == 0
    s, s_in = FF // world_size, H // world_size

    # dense 参照（前向 + 反向）：所有 rank 独立算，结果相同，用作本地校验基准
    W1g = W1.clone().requires_grad_(True)
    W2g = W2.clone().requires_grad_(True)
    Xg = X.clone().requires_grad_(True)
    Y_ref = F.gelu(Xg @ W1g) @ W2g
    Y_ref.sum().backward()
    y_scale = Y_ref.detach().abs().max().item()

    # ---- [1] 前向：列并行 W1 + 行并行 W2 + 1 次真实 all-reduce ----
    W1_r = W1[:, rank * s:(rank + 1) * s].clone().requires_grad_(True)
    W2_r = W2[rank * s:(rank + 1) * s, :].clone().requires_grad_(True)
    X_in = X.clone().requires_grad_(True)
    Y_tp = ReduceFromTensorParallelRegion.apply(
        F.gelu(CopyToTensorParallelRegion.apply(X_in) @ W1_r) @ W2_r)
    fwd_err = max_across_ranks((Y_tp.detach() - Y_ref.detach()).abs().max().item())

    # ---- [2] 反向：f/g 算子路由梯度 ----
    Y_tp.sum().backward()
    dx_err = max_across_ranks((X_in.grad - Xg.grad).abs().max().item())
    dw1_err = max_across_ranks(
        (W1_r.grad - W1g.grad[:, rank * s:(rank + 1) * s]).abs().max().item())
    dw2_err = max_across_ranks(
        (W2_r.grad - W2g.grad[rank * s:(rank + 1) * s, :]).abs().max().item())

    # ---- [3] 反例：W1 按行切（输入维切），真实 all-reduce ----
    P_r = X[:, rank * s_in:(rank + 1) * s_in] @ W1[rank * s_in:(rank + 1) * s_in, :]
    naive = F.gelu(P_r).clone()
    dist.all_reduce(naive, op=dist.ReduceOp.SUM)          # naive：先 GeLU 再 reduce
    Y_naive = naive @ W2
    naive_err = max_across_ranks((Y_naive - Y_ref.detach()).abs().max().item())
    fixed = P_r.clone()
    dist.all_reduce(fixed, op=dist.ReduceOp.SUM)           # fixed：先 reduce 再 GeLU
    Y_fixed = F.gelu(fixed) @ W2
    fixed_err = max_across_ranks((Y_fixed - Y_ref.detach()).abs().max().item())

    # ---- [4] 账本实测：每 rank 真实持有的 params / grads / Adam 状态 ----
    opt = torch.optim.Adam([W1_r, W2_r], lr=1e-3)
    opt.step()   # 物化 Adam 状态（exp_avg / exp_avg_sq）
    p_b = sum(p.numel() * p.element_size() for p in (W1_r, W2_r))
    g_b = sum(p.grad.numel() * p.grad.element_size() for p in (W1_r, W2_r))
    o_b = sum(v.numel() * v.element_size()
              for st in opt.state.values() for v in st.values()
              if torch.is_tensor(v) and v.dim() > 0)  # 只计 m/v（与参数同形），不含 step 标量
    ledger_local = torch.tensor([p_b, g_b, o_b], dtype=torch.float64)
    ledger_all = [torch.zeros(3, dtype=torch.float64) for _ in range(world_size)]
    dist.all_gather(ledger_all, ledger_local)

    # ---- [5] 计时：真实 all-reduce 墙钟（只 full_suite 跑）----
    fp16_msg, timings, compute_ms = None, [], None
    if full_suite:
        try:
            probe = torch.ones(4, dtype=torch.float16)
            dist.all_reduce(probe, op=dist.ReduceOp.SUM)
            got = probe[0].item()
            fp16_msg = (f"supported, sum(1 x {world_size} ranks) = {got}"
                        f"{'  ✅ 数值正确' if got == float(world_size) else '  ❌ 数值错误'}"
                        "（本机 torch 2.4.1 + gloo 实测）")
        except Exception as e:  # noqa: BLE001 — 记录真实行为
            fp16_msg = f"not supported: {type(e).__name__}"
        for mib in (1.0, 16.0, 32.0):
            buf = torch.randn(int(mib * 1024 * 1024 / 4))   # fp32
            for _ in range(WARMUP):
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            dt_ms = (time.perf_counter() - t0) / ITERS * 1000
            timings.append((mib, max_across_ranks(dt_ms)))
        t0 = time.perf_counter()
        for _ in range(200):
            _ = F.gelu(X_in.detach() @ W1_r.detach()) @ W2_r.detach()
        compute_ms = max_across_ranks((time.perf_counter() - t0) / 200 * 1000)

    # ---- 输出（rank 0）+ 全 rank self-check ----
    rel = lambda e: e / y_scale                                    # noqa: E731
    assert rel(fwd_err) < 1e-5, f"TP fwd mismatch: {fwd_err}"
    assert rel(dx_err) < 1e-5 and rel(dw1_err) < 1e-5 and rel(dw2_err) < 1e-5, \
        f"TP bwd mismatch: dX={dx_err} dW1={dw1_err} dW2={dw2_err}"
    assert rel(naive_err) > 1e-3, "row-first naive 应显著出错"
    assert rel(fixed_err) < 1e-5, f"row-first fixed 应恢复一致: {fixed_err}"
    assert COMM == {"fwd": 1, "bwd": 1}, f"集合通信次数异常: {COMM}"
    for lg in ledger_all:
        assert lg[0] == MLP_PARAMS * 4 / world_size, "每 rank 参数应恰为 P/N"
        assert lg[2] == 2 * lg[0], "Adam 状态应恰为参数 2 倍字节 (m+v)"

    if rank == 0:
        tag = f"[TP={world_size}]"
        print(f"{tag} forward: W1 列切 + W2 行切 + 1 次真实 all-reduce")
        print(f"    max|Y_tp - Y_ref| = {fwd_err:.3e} (相对 {rel(fwd_err):.1e})  ✅ 舍入级等价")
        print(f"{tag} backward: f/g autograd 算子")
        print(f"    max|dX - dX_ref| = {dx_err:.3e}   (f 反向 all-reduce 汇合)")
        print(f"    max|dW1 分片 - dW1_ref 对应块| = {dw1_err:.3e}   (本地即完整，零通信)")
        print(f"    max|dW2 分片 - dW2_ref 对应块| = {dw2_err:.3e}   (本地即完整，零通信)")
        print(f"    集合通信计数: fwd all-reduce = {COMM['fwd']}, bwd all-reduce = {COMM['bwd']}")
        print(f"{tag} 反例: W1 行切（切输入维）")
        print(f"    naive（先 GeLU 再 all-reduce）: err = {naive_err:.3e}  ❌ 错")
        print(f"    fixed（先 all-reduce 再 GeLU）: err = {fixed_err:.3e}  ✅ 对")
        msg_cr = T * H * 4
        msg_rf = T * FF * 4
        print(f"    {'design':<26}{'fwd all-reduce':<18}{'message':<18}{'params/rank'}")
        print(f"    {'column-row (Megatron)':<26}{'1':<18}"
              f"{fmt_kib(msg_cr) + ' [T,H]':<18}{MLP_PARAMS // world_size:,}")
        print(f"    {'row-first naive':<26}{'1 (wrong result)':<18}"
              f"{fmt_kib(msg_rf) + ' [T,F]':<18}{H * FF // world_size + H * FF:,}")
        print(f"    {'row-first fixed':<26}{'1':<18}"
              f"{fmt_kib(msg_rf) + ' [T,F]':<18}{H * FF // world_size + H * FF:,}")
        print(f"{tag} 账本实测: 每 rank params + grads + Adam 状态")
        for r in range(world_size):
            pb, gb, ob = ledger_all[r].tolist()
            print(f"    rank {r}: params={fmt_kib(pb)}  grads={fmt_kib(gb)}  "
                  f"optimizer={fmt_kib(ob)}  total={fmt_kib(pb + gb + ob)}")
        dense_total = MLP_PARAMS * 16
        print(f"    dense 完整副本 = 16 x {MLP_PARAMS:,} = {fmt_kib(dense_total)}"
              f" | TP 各 rank 之和 = {fmt_kib(sum(l.sum().item() for l in ledger_all))}")
        if full_suite:
            print(f"{tag} 计时: 真实 all-reduce 墙钟（gloo loopback, fp32, "
                  f"max over ranks, {ITERS} 次均值）")
            print(f"    fp16 all_reduce 探测（gloo）: {fp16_msg}")
            for mib, ms in timings:
                print(f"    msg {mib:5.1f} MiB: {ms:8.2f} ms/call")
            print(f"    toy fwd 计算（每 rank, 本形状）: {compute_ms:8.3f} ms")


def tp_worker(rank: int, world_size: int, port: int, full_suite: bool):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    torch.set_num_threads(THREADS_PER_PROC)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        run(rank, world_size, full_suite)
    finally:
        dist.destroy_process_group()


def main():
    warnings.filterwarnings("ignore")
    print("=" * 68)
    print("nano-megatron L1 — real tensor parallel MLP (torch.distributed/gloo)")
    print("=" * 68)
    print(f"shape: X[{T},{H}]  W1[{H},{FF}]  W2[{FF},{H}]  (f = 4h, GeLU MLP) | fp32 | seed={SEED}")
    print(f"MLP params P = {MLP_PARAMS:,} | torch {torch.__version__}, gloo, CPU")

    # TP=2：完整套件（前向/反向/反例/账本/计时）
    mp.spawn(tp_worker, args=(2, 29520, True), nprocs=2, join=True)
    # TP=4：前向/反向/反例/账本（验证 P/N 与等价性随 degree 保持）
    mp.spawn(tp_worker, args=(4, 29521, False), nprocs=4, join=True)

    print("\n" + "=" * 68)
    print("✅ self-check passed: 前向/反向与 dense 舍入级等价 · 行切反例在真实通信上"
          "依旧错且可修复 · 每 rank 状态恰为 16P/N · fwd/bwd 各 1 次 all-reduce")
    print("=" * 68)


if __name__ == "__main__":
    main()
