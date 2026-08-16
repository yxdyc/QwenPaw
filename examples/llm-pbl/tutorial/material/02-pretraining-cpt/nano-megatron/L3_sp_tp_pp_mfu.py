#!/usr/bin/env python3
r"""
L3_sp_tp_pp_mfu.py — nano-megatron L3

L1 切了「层内参数」（TP：列切 W1 + 行切 W2，fwd/bwd 各一次 all-reduce），
L2 切了「层间深度」（PP：stage 边界 p2p + bubble）。L3 上最后一块拼图——
**序列并行（Sequence Parallelism, SP）**，并做出 Megatron 的真实组合
TP × PP × SP，最后用 MFU 把这整套并行化「值不值」量化出来。

SP 的问题（arXiv:2205.05198）：TP 只切了 MLP/Attention 内部的激活
（[T, FF/t]），但 LayerNorm / Dropout / 残差这些「TP 未切区域」的激活
在每个 TP rank 上都是**完整副本** [T, H]——t 个 rank 存 t 份一模一样的东西。
SP 的解法：这些区域沿**序列维**切到 t 个 rank（每 rank 存 [T/t, H]），
进 MLP 前 all-gather 拼回、出 MLP 后 reduce-scatter 散回。关键恒等式：
    all-reduce ≡ reduce-scatter + all-gather
这个分解本身通信量中性；但真实实现（Megatron layers.py）不保存 gathered 的
[T,H] 输入（存了就把省下的显存吐回去），而是在 backward 里**重放一次
all-gather** 算 wgrad（L609-618）——所以 SP 是**显存优化，代价是通信微增**
（本节实测：每块 wire 字节 2m → 2.5m，+25%，t=2）。

本节量出五件事：
  [0] 等价性：同一块 LN+MLP，TP vs TP+SP 的权重/γ/β/dX 梯度逐项对照；
  [1] 通信账：SP 把每块 2 次 all-reduce 换成 fwd(AG+RS) + bwd(AG+RS+重放AG)，
      ring 等价字节 2m → 2.5m——分解中性、重放加价，账要算在实现上；
  [2] 显存账：TP 未切区域激活字节 SP 恰为 TP 的 1/t（t=TP），
      TP 已切区域不变——SP 的收益只在未切区域；
  [3] 组合：4 rank = PP2 × TP2 真实多进程一步训练（GPipe m=2，
      batched p2p 沿用 L2），SP on/off 步后权重对照 + PP 接缝字节
      在 SP 下减半（stage 间传的本来就是 SP 分片形态 [T/t, H]，
      对照 Megatron schedules.py get_tensor_shapes 的 seq // tp_size）
      + SP 下 LN γ/β 梯度是序列分片部分和、须跨 TP 组 all-reduce
      （对照 finalize_model_grads.py 的 layernorm-grads 归约）；
  [4] MFU：先用 GEMM 标定本机 fp32 峰值，再按 Megatron 的 FLOPs 公式
      （training.py num_floating_point_operations：MLP fwd = 4·expansion·
      tokens·h²，fwd+bwd ×3）算出模型有用 FLOPs，MFU = 有用 / (峰值×墙钟)——
      dense（无通信）/ TP+PP / TP+PP+SP 三个 MFU 把「并行效率」拆成
      可溯源的三段。CPU/gloo 上 MFU 很低（通信主导），这正是 MFU 的价值：
      它把 bubble/通信/小算子低效全部暴露成一个数字。

可运行性声明（课程可运行性契约）：全部计算真跑（4 个真实进程、真实 gloo
集合通信 all-gather/reduce-scatter/all-reduce + 真实 p2p）；MFU 的峰值用
GEMM 现场标定（不引用厂商标称值）；GPU/NCCL 上的绝对 MFU 与 SP 的真机
显存收益标 [TODO: verify on real system]，本脚本不冒充。

运行：python3 L3_sp_tp_pp_mfu.py   # ~5-15s, CPU 即可
依赖：torch（本机实测 torch 2.13.0；gloo 后端 list 版 all_gather /
      reduce_scatter 实测可用）
声明：计时行随机器浮动，共 9 行：含 "elapsed[" 的 4 行（calib/dense/nosp/sp）+
      [5] 解读行 1 行（含实测 wall 比值）+ 含 "MFU sanity" 的 self-check 行 2 行 +
      含 "MFU SP/非SP" 的 self-check 行 1 行 + total wall 行。
      掩码口径：sed '/elapsed\[/d; /解读: MFU/d; /MFU sanity/d; /MFU SP\/非SP/d;
      /^total wall/d'（paste 块与运行输出两侧同施），见 tutorial_L3.md §12。
      其余全部确定性（seed=7）。
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
# 常量（seed/形状与 L2 一致，便于跨级别对照；新增 LN/Dropout 与 MFU 参数）
# ---------------------------------------------------------------------------
SEED = 7
H, FF = 128, 512                 # hidden / FF 宽（expansion = FF/H = 4）
T = 512                          # 一个 batch 的 token 数（b*s 折叠，同 L1/L2）
N_BLOCKS = 4                     # 4 个 LN+GeLU-MLP 块，按层切到 2 个 PP stage
TP, PP = 2, 2                    # 4 rank = PP2 × TP2（SP 与 TP 同组同 degree）
WORLD = TP * PP
DROPOUT_P = 0.1
LN_EPS = 1e-5
LR = 1e-3
M = 2                            # 组合实验的 micro-batch 数（GPipe；调度非本节重点）
MASTER_PORT = 29550
THREADS_PER_PROC = 4
FP32 = 4
MFU_ITERS = 10                   # MFU 计时迭代数（取中位数）
GEMM_N, GEMM_ITERS, GEMM_ROUNDS = 512, 10, 3

BLOCKS_PER_STAGE = N_BLOCKS // PP
P_BLOCK = 2 * H * FF + 2 * H     # W1 + W2 + γ + β = 131,328
P_TOTAL = N_BLOCKS * P_BLOCK     # 525,312
EXPANSION = FF // H              # 4

# 每 rank 的集合通信计数器（nano 版 mappings.py 的账本；
# wire = ring 等价字节：AR(m)=2m(t-1)/t，AG(m)=RS(m)=m(t-1)/t）
COMM = {'ar_fwd': 0, 'ar_bwd': 0, 'ag_fwd': 0, 'rs_fwd': 0,
        'ag_bwd': 0, 'rs_bwd': 0, 'wire': 0}

# 激活账本：记录「为 backward 保存」的张量字节（在保存点实测，见各 Function）
ACT_SAVED = []                   # 元素: (tag, nbytes)


def ring_wire(op: str, full_numel: int) -> int:
    """ring 拓扑等价通信字节（与 L1/L2 的 all-reduce≡2×msg 同族口径）。"""
    m = full_numel * FP32
    if op == 'ar':
        return 2 * m * (TP - 1) // TP
    return m * (TP - 1) // TP    # ag / rs


def max_across_ranks(v: float) -> float:
    t = torch.tensor([v], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


# ---------------------------------------------------------------------------
# 模型与数据：所有 rank 同 seed 构造同一份「全量」参照（L1/L2 同款做法）
# 块结构（pre-LN 残差）：x ← x + Dropout(MLP(LN(x)))
#   LN 的 γ/β 不被 TP 切（Megatron 同款：LN 在 TP 区域之外）；
#   Dropout 掩码按**位置**确定（dedicated generator），SP 下按序列分片切片。
# ---------------------------------------------------------------------------

def build_model():
    torch.manual_seed(SEED)
    blocks = []
    for _ in range(N_BLOCKS):
        W1 = (torch.randn(H, FF) / H ** 0.5).requires_grad_(True)
        W2 = (torch.randn(FF, H) / FF ** 0.5).requires_grad_(True)
        gamma = torch.ones(H).requires_grad_(True)
        beta = torch.zeros(H).requires_grad_(True)
        blocks.append([W1, W2, gamma, beta])
    X = torch.randn(T, H)
    return blocks, X


def dropout_mask_full(block_idx: int, n: int) -> torch.Tensor:
    """位置确定的 inverted dropout 掩码 [n]：专用 generator，与 rank/切法无关。
    同一 seed 的前缀性质保证 mask(i, mb) == mask(i, T)[:mb]——micro-batch
    切片与全 batch 参照逐位一致。SP 的正确用法 = 按序列分片**切**这份掩码
    （掩码是位置数据，随激活一起切）。"""
    g = torch.Generator().manual_seed(SEED * 1000 + block_idx)
    keep = torch.rand(n, generator=g) >= DROPOUT_P
    return keep.float() / (1.0 - DROPOUT_P)


def block_params(blocks, lo, n):
    return [p for blk in blocks[lo:lo + n] for p in blk]


def shard_blocks(blocks_all, lo, tp_rank):
    """组合实验的本地块：W1 列切 + W2 行切（L0/L1 切法），γ/β 不切（TP 区域外，
    Megatron 同款）。每 rank 持有 P_stage = 2·(H·FF/t) + 2·2H 参数。"""
    s0, s1 = tp_rank * (FF // TP), (tp_rank + 1) * (FF // TP)
    return [
        [blocks_all[b][0][:, s0:s1].detach().clone().requires_grad_(True),
         blocks_all[b][1][s0:s1, :].detach().clone().requires_grad_(True),
         blocks_all[b][2].detach().clone().requires_grad_(True),
         blocks_all[b][3].detach().clone().requires_grad_(True)]
        for b in range(lo, lo + BLOCKS_PER_STAGE)
    ]


def sharded_param_delta(local_params, ref_params, tp_rank):
    """本地 TP 分片权重 vs dense 参照对应切片的 max|Δ|（γ/β 全量对照）。"""
    s0, s1 = tp_rank * (FF // TP), (tp_rank + 1) * (FF // TP)
    d = 0.0
    for i in range(0, len(local_params), 4):
        W1l, W2l, gl, bl = local_params[i:i + 4]
        W1r, W2r, gr, br = ref_params[i:i + 4]
        d = max(d, (W1l - W1r[:, s0:s1]).abs().max().item(),
                (W2l - W2r[s0:s1, :]).abs().max().item(),
                (gl - gr).abs().max().item(), (bl - br).abs().max().item())
    return d


def param_delta(params_a, params_b):
    """同切分形态的两组参数逐件 max|Δ|（如 SP vs 非SP，两侧分片方式相同）。"""
    return max((a - b).abs().max().item() for a, b in zip(params_a, params_b))


# ---------------------------------------------------------------------------
# 通信算子：nano 版 mappings.py（f/g 来自 L1；AG/RS 为 SP 新增）
#
# 对照（NVIDIA/Megatron-LM main, 2026-08-10 抓取，行号以抓取日为准）：
#   f = _CopyToModelParallelRegion          (mappings.py:L201  fwd identity / bwd AR)
#   g = _ReduceFromModelParallelRegion      (mappings.py:L221  fwd AR / bwd identity)
#   _GatherFromSequenceParallelRegion       (mappings.py:L300  fwd AG / bwd RS)
#   _ReduceScatterToSequenceParallelRegion  (mappings.py:L355  fwd RS / bwd AG)
# ---------------------------------------------------------------------------

def _all_gather_seq(x_shard, group):
    """序列维 all-gather：[s/t, H] → [s, H]（对照 mappings.py:L118
    _gather_along_first_dim 的 list 版路径）。"""
    world = dist.get_world_size(group)
    buf = [torch.empty_like(x_shard) for _ in range(world)]
    dist.all_gather(buf, x_shard.contiguous(), group=group)
    return torch.cat(buf, dim=0)


def _reduce_scatter_seq(full, group):
    """序列维 reduce-scatter：[s, H] 部分和 → 本 rank 的 [s/t, H]
    （对照 mappings.py:L159 _reduce_scatter_along_first_dim 的 list 版路径）。"""
    world = dist.get_world_size(group)
    out = torch.empty(full.shape[0] // world, *full.shape[1:])
    dist.reduce_scatter(out, list(full.contiguous().chunk(world)), group=group)
    return out


class CopyToTP(torch.autograd.Function):
    """f 算子（非 SP）：fwd identity，bwd all-reduce 汇合 dX 部分和。"""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous().clone()
        dist.all_reduce(g, group=ctx.group)
        COMM['ar_bwd'] += 1
        COMM['wire'] += ring_wire('ar', g.numel())
        return g, None


class ReduceFromTP(torch.autograd.Function):
    """g 算子（非 SP）：fwd all-reduce 行并行部分和，bwd identity。"""

    @staticmethod
    def forward(ctx, x, group):
        x = x.contiguous().clone()
        dist.all_reduce(x, group=group)
        COMM['ar_fwd'] += 1
        COMM['wire'] += ring_wire('ar', x.numel())
        return x

    @staticmethod
    def backward(ctx, g):
        return g, None


class ReduceScatterToSP(torch.autograd.Function):
    """SP 出口算子：fwd reduce-scatter（行并行部分和 → 序列分片），bwd all-gather。
    对照 mappings.py:L355 _ReduceScatterToSequenceParallelRegion。
    这正是 Megatron RowParallelLinear 在 sequence_parallel=True 时的分支
    （layers.py:L1472-1478：reduce_scatter_to_sequence_parallel_region
    替代 reduce_from_tensor_model_parallel_region 的 all-reduce）。"""

    @staticmethod
    def forward(ctx, full, group):
        ctx.group = group
        COMM['rs_fwd'] += 1
        COMM['wire'] += ring_wire('rs', full.numel())
        return _reduce_scatter_seq(full, group)

    @staticmethod
    def backward(ctx, g):
        COMM['ag_bwd'] += 1
        COMM['wire'] += ring_wire('ag', g.numel() * dist.get_world_size(ctx.group))
        return _all_gather_seq(g, ctx.group), None


# ---------------------------------------------------------------------------
# 受控保存的 LN / Dropout / SP 列并行线性——激活账本在保存点实测
# ---------------------------------------------------------------------------

class NanoLayerNorm(torch.autograd.Function):
    """逐 token 的 LayerNorm（统计量只在 H 维上）——因此可以按序列维切片算，
    结果与全序列算逐位相同（SP 成立的机制前提）。保存集 = {xhat, rstd}，
    显式记账。γ/β 梯度 = 对本地 token 的部分和（SP 下须跨 TP 组再归约，
    对照 finalize_model_grads.py:L416/L451-453）。"""

    @staticmethod
    def forward(ctx, x, gamma, beta, tag):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        rstd = 1.0 / (var + LN_EPS).sqrt()
        xhat = (x - mean) * rstd
        y = xhat * gamma + beta
        ctx.save_for_backward(xhat, gamma, rstd)
        ACT_SAVED.append((tag + '.xhat', xhat.numel() * FP32))
        ACT_SAVED.append((tag + '.rstd', rstd.numel() * FP32))
        return y

    @staticmethod
    def backward(ctx, dy):
        xhat, gamma, rstd = ctx.saved_tensors
        dgamma = (dy * xhat).sum(0)          # SP 下 = 序列分片部分和
        dbeta = dy.sum(0)
        dxhat = dy * gamma
        dx = rstd * (dxhat
                     - dxhat.mean(-1, keepdim=True)
                     - (dxhat * xhat).mean(-1, keepdim=True) * xhat)
        return dx, dgamma, dbeta, None


class MaskedDropout(torch.autograd.Function):
    """位置掩码 dropout。只保存掩码（区域张量：非 SP = [s]，SP = [s/t] 切片）。"""

    @staticmethod
    def forward(ctx, x, mask, tag):
        ctx.save_for_backward(mask)
        ACT_SAVED.append((tag + '.mask', mask.numel() * FP32))
        return x * mask[:, None]

    @staticmethod
    def backward(ctx, dy):
        mask, = ctx.saved_tensors
        return dy * mask[:, None], None, None


class SPColumnLinear(torch.autograd.Function):
    """SP 列并行线性：fwd = all-gather(输入分片) → x@W1_shard；
    **只保存分片输入**，backward 里重新 all-gather 算 wgrad——
    对照 Megatron 的融合线性（layers.py:L565-573 fwd 全局 buffer gather；
    L609-618 bwd 重放 all-gather）。这是 SP 显存收益成立的关键：
    若把 gathered 的 [s,H] 存下来给 wgrad，省下的激活会原样吐回去；
    Megatron 选择反向**重放**（以一次额外 gather 的通信换激活显存）——
    这正是「SP 通信微增」的来源（本脚本 [1] 实测 +25% wire 字节）。"""

    @staticmethod
    def forward(ctx, x_shard, W1, group):
        ctx.group = group
        ctx.save_for_backward(x_shard, W1)
        ACT_SAVED.append(('spcol.x_shard', x_shard.numel() * FP32))
        COMM['ag_fwd'] += 1
        COMM['wire'] += ring_wire('ag', x_shard.numel() * dist.get_world_size(group))
        full = _all_gather_seq(x_shard, group)
        return full @ W1

    @staticmethod
    def backward(ctx, dout):
        x_shard, W1 = ctx.saved_tensors
        COMM['ag_bwd'] += 1                    # bwd 重放 gather（layers.py:L609-618）
        COMM['wire'] += ring_wire('ag', x_shard.numel() * dist.get_world_size(ctx.group))
        full = _all_gather_seq(x_shard, ctx.group)
        dW1 = full.T @ dout
        dfull = dout @ W1.T
        COMM['rs_bwd'] += 1                    # gather 的反向 = reduce-scatter
        COMM['wire'] += ring_wire('rs', dfull.numel())
        return _reduce_scatter_seq(dfull, ctx.group), dW1, None


# ---------------------------------------------------------------------------
# 块前向：非 SP（TP）与 SP 两条路径，区域激活账本在保存点实测
# ---------------------------------------------------------------------------

def block_forward(x, blk, mask, tp_group, sp: bool, tag: str):
    """x ← x + Dropout(MLP(LN(x)))。
    非 SP：x 为全序列 [s, H]（TP 组内复制）；SP：x 为分片 [s/t, H]。
    tp_group=None → 纯本地 dense 路径（参照系用，无任何通信算子）。"""
    W1, W2, gamma, beta = blk
    ln = NanoLayerNorm.apply(x, gamma, beta, tag)
    if tp_group is None:                       # dense 参照：纯本地
        h = ln @ W1
    elif sp:
        h = SPColumnLinear.apply(ln, W1, tp_group)   # AG 在算子内部
    else:
        h = CopyToTP.apply(ln, tp_group) @ W1        # f 算子 + 本地列切 matmul
        ACT_SAVED.append((tag + '.ln_for_matmul', ln.numel() * FP32))
    a = F.gelu(h)
    ACT_SAVED.append((tag + '.gelu_in', h.numel() * FP32))
    ACT_SAVED.append((tag + '.a', a.numel() * FP32))
    y_partial = a @ W2
    if tp_group is None:
        y = y_partial
    elif sp:
        y = ReduceScatterToSP.apply(y_partial, tp_group)   # RS 替代 AR
    else:
        y = ReduceFromTP.apply(y_partial, tp_group)        # g 算子 AR
    return x + MaskedDropout.apply(y, mask, tag)


def chunked_loss(out, tp_group, sp: bool):
    """loss = 全元素和 / T。两条路径用**同一归约分组**：先把序列对半求和再相加
    （fp32 下归约顺序是数值的一部分，须显式对齐才能隔离 SP 自身的影响）。
    非 SP：out = [mb, H] 全序列，本地两半和相加（复制，无需通信）；
    SP：out = [mb/t, H] 分片，本地分片和 + TP 组 all-reduce——t=2 时恰为
    「前半和 + 后半和」同一分组，两模式 loss bit 相同。"""
    if sp:
        local = out.sum()
        dist.all_reduce(local, group=tp_group)
        return local / T
    half = out.shape[0] // 2
    return (out[:half].sum() + out[half:].sum()) / T


# ---------------------------------------------------------------------------
# 参照系（每个 rank 本地独立算，确定性相同）
# ---------------------------------------------------------------------------

def dense_reference(per_mb: bool):
    """单进程 dense 参照：自建同一模型、同一掩码、同一归约分组（纯本地路径，
    无任何通信算子）。per_mb=True 逐 micro-batch（镜像参照），False 全 batch。"""
    blocks, X = build_model()
    masks = [dropout_mask_full(i, T) for i in range(N_BLOCKS)]
    params = block_params(blocks, 0, N_BLOCKS)
    opt = torch.optim.Adam(params, lr=LR)

    def run_mb(x_mb, t0):
        s = x_mb.shape[0]
        ms = [m[t0:t0 + s] for m in masks]
        out = x_mb
        for i, blk in enumerate(blocks):
            out = block_forward(out, blk, ms[i], None, None, tag=f'r{i}')
        half = s // 2
        return (out[:half].sum() + out[half:].sum()) / T

    if per_mb:
        mb = T // M
        losses = []
        for i in range(M):
            l = run_mb(X[i * mb:(i + 1) * mb], i * mb)
            losses.append(l)
        for l in losses:
            l.backward()
        losses = [l.item() for l in losses]
    else:
        l = run_mb(X, 0)
        losses = [l.item()]
        l.backward()
    opt.step()
    return losses, [p.detach().clone() for p in params]


# ---------------------------------------------------------------------------
# [0]-[2] SP 机制实验：单块 LN+MLP，TP vs TP+SP（各 TP 组内独立跑）
# ---------------------------------------------------------------------------

def mechanism_experiment(rank, tp_rank, tp_group):
    """单块前向+反向：等价性 / 通信账 / 激活账 / dropout 反例。
    返回本地指标 dict（rank 0 汇报）。"""
    global COMM, ACT_SAVED
    res = {}
    blocks, X = build_model()
    blk = blocks[0]
    mask_full = dropout_mask_full(0, T)
    mb = T // TP
    lo, hi = tp_rank * (FF // TP), (tp_rank + 1) * (FF // TP)
    sx0, sx1 = tp_rank * mb, (tp_rank + 1) * mb

    # ---- 参照：dense 单进程（本地算，无通信算子）----
    b = [w.detach().clone().requires_grad_(True) for w in blk]
    x = X.clone().requires_grad_(True)
    ln = NanoLayerNorm.apply(x, b[2], b[3], 'd')
    y = F.gelu(ln @ b[0]) @ b[1]
    out = x + MaskedDropout.apply(y, mask_full, 'd')
    out.sum().backward()
    dW1_ref, dW2_ref, dgamma_ref, dbeta_ref = [p.grad for p in b]
    dX_ref = x.grad

    # ---- 模式一：非 SP TP（f/g 算子，全序列复制）----
    COMM = {k: 0 for k in COMM}
    ACT_SAVED = []
    W1 = blk[0][:, lo:hi].detach().clone().requires_grad_(True)   # 列切 W1
    W2 = blk[1][lo:hi, :].detach().clone().requires_grad_(True)   # 行切 W2
    gamma = blk[2].detach().clone().requires_grad_(True)
    beta = blk[3].detach().clone().requires_grad_(True)
    x = X.clone().requires_grad_(True)
    ln = NanoLayerNorm.apply(x, gamma, beta, 'n')
    h = CopyToTP.apply(ln, tp_group) @ W1
    ACT_SAVED.append(('n.ln_for_matmul', ln.numel() * FP32))
    a = F.gelu(h)
    ACT_SAVED.append(('n.gelu_in', h.numel() * FP32))
    ACT_SAVED.append(('n.a', a.numel() * FP32))
    y = ReduceFromTP.apply(a @ W2, tp_group)
    out = x + MaskedDropout.apply(y, mask_full, 'n')
    out.sum().backward()
    comm_tp = dict(COMM)
    act_tp = dict(region=sum(bb for tt, bb in ACT_SAVED
                             if 'gelu_in' not in tt and '.a' not in tt),
                  sharded=sum(bb for tt, bb in ACT_SAVED
                              if 'gelu_in' in tt or '.a' in tt))
    res['dW1_tp'] = (W1.grad - dW1_ref[:, lo:hi]).abs().max().item()
    res['dW2_tp'] = (W2.grad - dW2_ref[lo:hi, :]).abs().max().item()
    res['dgamma_tp'] = (gamma.grad - dgamma_ref).abs().max().item()
    res['dbeta_tp'] = (beta.grad - dbeta_ref).abs().max().item()
    dX_tp_full = x.grad.detach().clone()

    # ---- 模式二：TP+SP（AG/RS 算子，序列分片）----
    COMM = {k: 0 for k in COMM}
    ACT_SAVED = []
    W1 = blk[0][:, lo:hi].detach().clone().requires_grad_(True)   # 列切 W1
    W2 = blk[1][lo:hi, :].detach().clone().requires_grad_(True)   # 行切 W2
    gamma = blk[2].detach().clone().requires_grad_(True)
    beta = blk[3].detach().clone().requires_grad_(True)
    x_shard = X[sx0:sx1].clone().requires_grad_(True)
    mask_shard = mask_full[sx0:sx1]
    ln = NanoLayerNorm.apply(x_shard, gamma, beta, 's')
    h = SPColumnLinear.apply(ln, W1, tp_group)
    a = F.gelu(h)
    ACT_SAVED.append(('s.gelu_in', h.numel() * FP32))
    ACT_SAVED.append(('s.a', a.numel() * FP32))
    y = ReduceScatterToSP.apply(a @ W2, tp_group)
    out = x_shard + MaskedDropout.apply(y, mask_shard, 's')
    out.sum().backward()
    comm_sp = dict(COMM)
    act_sp = dict(region=sum(bb for tt, bb in ACT_SAVED
                             if 'gelu_in' not in tt and '.a' not in tt),
                  sharded=sum(bb for tt, bb in ACT_SAVED
                              if 'gelu_in' in tt or '.a' in tt))

    res['dW1_sp'] = (W1.grad - dW1_ref[:, lo:hi]).abs().max().item()
    res['dW2_sp'] = (W2.grad - dW2_ref[lo:hi, :]).abs().max().item()
    # SP 下 γ/β 梯度 = 序列分片部分和（本地只见过 T/t 个 token）——
    # 须跨 TP 组 all-reduce 才完整（nano 版 finalize_model_grads，
    # finalize_model_grads.py:L416/L451-453）。先录部分和与全量的差（教学用），
    # 再 all-reduce 后与 dense 对照。
    res['dgamma_sp_partial'] = (gamma.grad - dgamma_ref).abs().max().item()
    res['dbeta_sp_partial'] = (beta.grad - dbeta_ref).abs().max().item()
    dgamma_sp_full = gamma.grad.detach().clone()
    dbeta_sp_full = beta.grad.detach().clone()
    dist.all_reduce(dgamma_sp_full, group=tp_group)
    dist.all_reduce(dbeta_sp_full, group=tp_group)
    res['dgamma_sp'] = (dgamma_sp_full - dgamma_ref).abs().max().item()
    res['dbeta_sp'] = (dbeta_sp_full - dbeta_ref).abs().max().item()
    res['gamma_scale'] = dgamma_ref.abs().max().item()
    res['beta_scale'] = dbeta_ref.abs().max().item()
    res['dX_sp_vs_tp'] = (x_shard.grad - dX_tp_full[sx0:sx1]).abs().max().item()
    res['comm_tp'], res['comm_sp'] = comm_tp, comm_sp
    res['act'] = {'tp_region': act_tp['region'], 'tp_sharded': act_tp['sharded'],
                  'sp_region': act_sp['region'], 'sp_sharded': act_sp['sharded']}

    # ---- 反例：非 SP 却用「每 rank 独立掩码」（SP 的 RNG fork 误搬到非 SP）----
    # SP 下每个 rank 只持有序列的一部分，掩码必须各 rank 独立（Megatron 在 SP 区域
    # fork RNG，transformer_block.py:L595-598）；但非 SP 下激活是全序列**复制**，
    # 两 rank 必须用**同一份**掩码——否则复制流分叉，梯度从分叉的激活上算出来。
    forked = dropout_mask_full(100 + tp_rank, T)     # rank 相关 = 错误
    W1f = blk[0][:, lo:hi].detach().clone().requires_grad_(True)
    W2f = blk[1][lo:hi, :].detach().clone().requires_grad_(True)
    xf = X.clone().requires_grad_(True)
    lnf = NanoLayerNorm.apply(xf, blk[2].detach().clone().requires_grad_(True),
                              blk[3].detach().clone().requires_grad_(True), 'f')
    hf = CopyToTP.apply(lnf, tp_group) @ W1f
    yf = ReduceFromTP.apply(F.gelu(hf) @ W2f, tp_group)
    outf = xf + MaskedDropout.apply(yf, forked, 'f')
    outf.sum().backward()
    res['forked_dW1'] = (W1f.grad - dW1_ref[:, lo:hi]).abs().max().item()
    return res


# ---------------------------------------------------------------------------
# [3] 组合实验：PP2 × TP2，GPipe m=2，SP on/off，一步 Adam
# ---------------------------------------------------------------------------

def _p2p_batch(ops_info):
    """L2 同款：一批 send/recv 打包为一次 batch_isend_irecv 原子提交
    （Megatron p2p_communication.py _batched_p2p_ops 风格，消除循环等待）。"""
    if not ops_info:
        return []
    recv_tensors = []
    ops = []
    for op_type, data, peer in ops_info:
        if op_type == 'send':
            ops.append(dist.P2POp(dist.isend, data.contiguous(), peer))
        else:
            t = torch.empty(data)
            recv_tensors.append(t)
            ops.append(dist.P2POp(dist.irecv, t, peer))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return recv_tensors


def combined_step(rank, world, tp_rank, tp_group, sp: bool, blocks_local, X, lo: int):
    """PP2×TP2 一步（GPipe m=2，梯度累加 + Adam step）。
    SP on：stage 间传 [mb/t, H] 分片（对照 schedules.py:L2122-2123
    get_tensor_shapes 的 seq // tp_size）；LN γ/β 梯度为分片部分和，
    步前跨 TP 组 all-reduce（nano 版 finalize_model_grads，
    finalize_model_grads.py:L416/L451-453）。
    返回 (losses[仅末 stage], p2p_bytes, wall, busy)。"""
    global ACT_SAVED
    ACT_SAVED = []
    pp_rank = rank // TP
    peer_pp = (pp_rank ^ 1) * TP + tp_rank          # 另一 stage 的同 TP 位 rank
    is_first, is_last = pp_rank == 0, pp_rank == PP - 1
    mb = T // M
    s_local = mb // TP if sp else mb                # 本 rank 持有的序列长
    masks_full = [dropout_mask_full(lo + j, T) for j in range(BLOCKS_PER_STAGE)]
    p2p_bytes = 0
    busy = 0.0
    losses = []
    live = []

    params = [p for blk in blocks_local for p in blk]
    for p in params:
        p.grad = None

    t_wall0 = time.perf_counter()
    # ---- GPipe：全 forward 再全 backward（调度非本节重点，L2 已做透）----
    for i in range(M):
        if is_first:
            x_mb = X[i * mb:(i + 1) * mb]
            if sp:
                x_mb = x_mb[tp_rank * s_local:(tp_rank + 1) * s_local]
            x_in = x_mb.clone().requires_grad_(True)
        else:
            r = _p2p_batch([('recv', (s_local, H), peer_pp)])
            p2p_bytes += s_local * H * FP32
            x_in = r[0].requires_grad_(True)
        t0 = time.perf_counter()
        out = x_in
        for j, blk in enumerate(blocks_local):
            mask_mb = masks_full[j][i * mb:(i + 1) * mb]
            if sp:
                mask_mb = mask_mb[tp_rank * s_local:(tp_rank + 1) * s_local]
            out = block_forward(out, blk, mask_mb, tp_group, sp, tag=f'c{i}b{j}')
        busy += time.perf_counter() - t0
        if is_last:
            loss = chunked_loss(out, tp_group, sp)
            losses.append(loss)
            live.append((x_in, loss))
        else:
            _p2p_batch([('send', out, peer_pp)])
            p2p_bytes += out.numel() * FP32
            live.append((x_in, out))

    for i in range(M):
        x_in, out = live.pop(0)
        if is_last:
            t0 = time.perf_counter()
            out.backward()
            busy += time.perf_counter() - t0
            _p2p_batch([('send', x_in.grad, peer_pp)])
            p2p_bytes += x_in.grad.numel() * FP32
        else:
            r = _p2p_batch([('recv', (s_local, H), peer_pp)])
            p2p_bytes += s_local * H * FP32
            g = r[0]
            t0 = time.perf_counter()
            out.backward(g)
            busy += time.perf_counter() - t0
            if not is_first:
                _p2p_batch([('send', x_in.grad, peer_pp)])
                p2p_bytes += x_in.grad.numel() * FP32

    # ---- nano 版 finalize_model_grads：SP 下 LN γ/β 梯度 = 序列分片部分和，
    #      跨 TP 组 all-reduce 后才完整（非 SP 下本地即完整，无需通信）----
    if sp:
        for blk in blocks_local:
            for p in blk[2:]:                        # γ, β（非 TP 切参数）
                dist.all_reduce(p.grad, group=tp_group)

    opt = torch.optim.Adam(params, lr=LR)
    opt.step()
    dist.barrier()
    wall = time.perf_counter() - t_wall0
    return ([l.item() for l in losses] if is_last else []), p2p_bytes, wall, busy


# ---------------------------------------------------------------------------
# [4] MFU：GEMM 标定峰值 + Megatron FLOPs 公式 + 三段 MFU
# ---------------------------------------------------------------------------

def model_flops_per_step():
    """Megatron training.py:L391 num_floating_point_operations 的 nano 口径：
    MLP fwd = 4·expansion·tokens·h²（L428-431 mlp_layer_flops，只数 GEMM，
    LN/Dropout/残差 O(T·H) 忽略——与 Megatron 同口径）；fwd+bwd = ×3（L548）。
    nano 块无 attention（Megatron 的 attn 项 attn_layer_flops 在此不适用，声明）。"""
    fwd_mlp = 4 * EXPANSION * T * H * H              # 每块 fwd
    return 3 * N_BLOCKS * fwd_mlp


def gemm_peak_gflops():
    """本机 fp32 GEMM 峰值标定（不引用厂商标称值）：n³ GEMM × iters × rounds，
    取最快一轮。每 rank 都跑，取 MAX（最慢 rank 决定流水线速度，如实声明口径）。"""
    a = torch.randn(GEMM_N, GEMM_N)
    b = torch.randn(GEMM_N, GEMM_N)
    for _ in range(2):                               # warmup
        a @ b
    best = float('inf')
    for _ in range(GEMM_ROUNDS):
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(GEMM_ITERS):
            a @ b
        best = min(best, time.perf_counter() - t0)
    dt = max_across_ranks(best)
    return 2 * GEMM_N ** 3 * GEMM_ITERS / dt / 1e9


def mfu_loop(rank, world, tp_rank, tp_group, sp: bool, blocks_local, X, lo: int):
    """MFU 计时环：MFU_ITERS 遍 fwd+bwd+step，取中位数墙钟。"""
    walls = []
    for _ in range(MFU_ITERS):
        _, _, wall, _ = combined_step(rank, world, tp_rank, tp_group, sp,
                                      blocks_local, X, lo)
        walls.append(wall)
    return sorted(walls)[len(walls) // 2]


# ---------------------------------------------------------------------------
# worker / run
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

    # TP 组：{0,1} 与 {2,3}（new_group 须所有 rank 同序调用）
    tp_groups = [dist.new_group([pp * TP + i for i in range(TP)]) for pp in range(PP)]
    pp_rank, tp_rank = rank // TP, rank % TP
    tp_group = tp_groups[pp_rank]
    lo = pp_rank * BLOCKS_PER_STAGE

    # ================= [0]-[2] SP 机制（单块，各 TP 组独立） =================
    mech = mechanism_experiment(rank, tp_rank, tp_group)

    # SP / 非SP 各自 vs dense 参照（同一物理量：dW 分片 / dγ / dβ / dX 分片）
    d_vs_ref = {
        'dW1': max(mech['dW1_sp'], mech['dW1_tp']),
        'dW2': max(mech['dW2_sp'], mech['dW2_tp']),
        'dgamma': max(mech['dgamma_sp'], mech['dgamma_tp']),
        'dbeta': max(mech['dbeta_sp'], mech['dbeta_tp']),
    }
    d_vs_ref = {k: max_across_ranks(v) for k, v in d_vs_ref.items()}
    partial_delta = max_across_ranks(max(mech['dgamma_sp_partial'],
                                         mech['dbeta_sp_partial']))
    dX_sp_vs_tp = max_across_ranks(mech['dX_sp_vs_tp'])
    forked_dW1 = max_across_ranks(mech['forked_dW1'])

    # 通信账（各 rank 本地值即组内对称值，取 MAX 稳）
    comm_tp = {k: max_across_ranks(float(v)) for k, v in mech['comm_tp'].items()}
    comm_sp = {k: max_across_ranks(float(v)) for k, v in mech['comm_sp'].items()}
    act = {k: max_across_ranks(float(v)) for k, v in mech['act'].items()}

    # ================= [3] 组合：PP2 × TP2，SP on/off =================
    results = {}
    for sp in (False, True):
        blocks_all, X = build_model()
        blocks_local = shard_blocks(blocks_all, lo, tp_rank)
        losses, p2p_bytes, wall, busy = combined_step(
            rank, world, tp_rank, tp_group, sp, blocks_local, X, lo)
        results[sp] = {
            'losses': losses,
            'p2p_bytes': max_across_ranks(float(p2p_bytes)),
            'params': [w.detach().clone() for blk in blocks_local for w in blk],
            'wall': wall, 'busy': busy,
        }

    # dense 参照（镜像：逐 mb）与全 batch
    ref_mb_losses, ref_mb_params = dense_reference(per_mb=True)
    _, ref_full_params = dense_reference(per_mb=False)
    ref_mb_slice = ref_mb_params[lo * 4:(lo + BLOCKS_PER_STAGE) * 4]
    ref_full_slice = ref_full_params[lo * 4:(lo + BLOCKS_PER_STAGE) * 4]

    d_sp_nosp = max_across_ranks(param_delta(results[True]['params'],
                                             results[False]['params']))
    d_nosp_ref = max_across_ranks(sharded_param_delta(
        results[False]['params'], ref_mb_slice, tp_rank))
    d_sp_ref = max_across_ranks(sharded_param_delta(
        results[True]['params'], ref_mb_slice, tp_rank))
    d_nosp_full = max_across_ranks(sharded_param_delta(
        results[False]['params'], ref_full_slice, tp_rank))
    loss_delta = 0.0
    if results[True]['losses']:
        loss_delta = max(abs(a - b) for a, b in
                         zip(results[True]['losses'], results[False]['losses']))
    loss_delta = max_across_ranks(loss_delta)

    exp_bytes_nosp = 2 * (PP - 1) * T * H * FP32
    exp_bytes_sp = 2 * (PP - 1) * (T // TP) * H * FP32

    # ================= [4] MFU =================
    peak_gflops = gemm_peak_gflops()
    flops_step = model_flops_per_step()
    # per-rank 口径：每 rank 算 1/(TP·PP) 的模型 FLOPs（W 按 TP 切、层按 PP 切；
    # LN/Dropout 的复制计算 O(T·H) 忽略）——per-device MFU 是标准口径
    # （PaLM arXiv:2204.02311 的 MFU 即按单设备算）。
    flops_rank = flops_step // (TP * PP)

    mfu = {}
    for sp in (False, True):
        blocks_all, X = build_model()
        blocks_local = shard_blocks(blocks_all, lo, tp_rank)
        med_wall = mfu_loop(rank, world, tp_rank, tp_group, sp, blocks_local, X, lo)
        mfu[sp] = {'wall': med_wall,
                   'achieved': flops_rank / med_wall / 1e9,
                   'mfu': flops_rank / med_wall / (peak_gflops * 1e9)}

    # dense（无通信）MFU：并行效率的上界参照（所有 rank 同算，rank 0 汇报）
    blocks_all, X = build_model()
    masks = [dropout_mask_full(i, T) for i in range(N_BLOCKS)]
    dense_walls = []
    for _ in range(MFU_ITERS):
        for p in block_params(blocks_all, 0, N_BLOCKS):
            p.grad = None
        dist.barrier()
        t0 = time.perf_counter()
        out = X
        for i, blk in enumerate(blocks_all):
            out = block_forward(out, blk, masks[i], None, None, tag=f'm{i}')
        loss = (out[:T // 2].sum() + out[T // 2:].sum()) / T
        loss.backward()
        dense_walls.append(time.perf_counter() - t0)
    dense_wall = sorted(dense_walls)[len(dense_walls) // 2]
    mfu_dense = flops_step / dense_wall / (peak_gflops * 1e9)

    # ------------------------------------------------------------------
    # self-check（全部机器断言；每个 rank 都跑，rank 0 打印）
    # ------------------------------------------------------------------
    m_unit = T * H * FP32                            # 一次 [T,H] 全集体的消息字节
    # [a] 等价性：SP 与 TP 各自 vs dense 参照
    #     dW 分片实测 bit 级相同（1e-6 绝对阈值）；γ/β 梯度值 O(100)，
    #     fp32 归约顺序差为绝对 O(1e-5)、相对 ~1e-7（舍入级），用相对阈值。
    for name in ('dW1', 'dW2'):
        ck(d_vs_ref[name] <= 1e-6,
           f"single-block {name}: SP & TP both vs dense ref within 1e-6 "
           f"(max Δ = {d_vs_ref[name]:.2e})")
    for name, scale in (('dgamma', mech['gamma_scale']), ('dbeta', mech['beta_scale'])):
        ck(d_vs_ref[name] <= 1e-6 * scale,
           f"single-block {name}: SP & TP both vs dense ref within rel 1e-6 "
           f"(Δ = {d_vs_ref[name]:.2e}, |{name}| max = {scale:.1f}, "
           f"rel = {d_vs_ref[name] / scale:.2e}; fp32 归约顺序差)")
    ck(dX_sp_vs_tp <= 1e-6,
       f"single-block dX: SP shard vs TP full-slice Δ = {dX_sp_vs_tp:.2e} ≤ 1e-6")
    # [b] 通信账：次数结构与 ring 等价字节（分解中性 + 重放加价）
    ck(comm_tp['ar_fwd'] == 1 and comm_tp['ar_bwd'] == 1
       and comm_tp['ag_fwd'] == 0 and comm_tp['rs_fwd'] == 0,
       f"TP comm/block: 1 AR fwd + 1 AR bwd (got ar_fwd={comm_tp['ar_fwd']:.0f}, "
       f"ar_bwd={comm_tp['ar_bwd']:.0f})")
    ck(comm_sp['ag_fwd'] == 1 and comm_sp['rs_fwd'] == 1
       and comm_sp['ag_bwd'] == 2 and comm_sp['rs_bwd'] == 1
       and comm_sp['ar_fwd'] == 0 and comm_sp['ar_bwd'] == 0,
       f"SP comm/block: fwd AG+RS, bwd AG+RS + 1 重放AG(wgrad) "
       f"(got ag_fwd={comm_sp['ag_fwd']:.0f}, rs_fwd={comm_sp['rs_fwd']:.0f}, "
       f"ag_bwd={comm_sp['ag_bwd']:.0f}, rs_bwd={comm_sp['rs_bwd']:.0f})")
    ck(comm_tp['wire'] == 2 * m_unit,
       f"TP wire/block = 2m = {2 * m_unit:,} B (got {comm_tp['wire']:.0f})")
    ck(comm_sp['wire'] == 5 * m_unit // 2,
       f"SP wire/block = 2.5m = {5 * m_unit // 2:,} B = TP×1.25 "
       f"(got {comm_sp['wire']:.0f}; AR≡RS+AG 分解中性, 重放 gather 加价)")
    # [c] 激活账：区域字节恰 1/t，TP 已切字节不变
    ck(act['tp_region'] == act['sp_region'] * TP,
       f"region activations/block: TP = {act['tp_region']:.0f} B = "
       f"{TP} × SP {act['sp_region']:.0f} B (恰 1/t)")
    ck(act['tp_sharded'] == act['sp_sharded'],
       f"TP-sharded activations/block 不变: {act['tp_sharded']:.0f} B "
       f"(SP 的收益只在未切区域)")
    # [d] dropout 掩码必须随序列切（反例）
    ck(forked_dW1 > 1e-3,
       f"counterexample: 非 SP 用 per-rank forked 掩码 → dW1 Δ = {forked_dW1:.3e} "
       f"(复制流分叉, 显著错)")
    # [d2] SP γ/β 梯度 all-reduce 前确为部分和（差显著非零），all-reduce 后收敛
    ck(partial_delta > 1e-3,
       f"SP γ/β grads pre-allreduce = partial sums (Δ vs dense = {partial_delta:.3e} "
       f"> 1e-3, 本地只见过 T/{TP} 个 token)")
    # [e] 组合正确性
    ck(d_sp_nosp <= 1e-6,
       f"combined: 步后权重 SP vs 非SP max|Δ| = {d_sp_nosp:.2e} ≤ 1e-6")
    ck(d_nosp_ref < 1e-5,
       f"combined: 非SP vs dense 镜像参照 max|Δ| = {d_nosp_ref:.2e} < 1e-5")
    ck(d_sp_ref < 1e-5,
       f"combined: SP vs dense 镜像参照 max|Δ| = {d_sp_ref:.2e} < 1e-5")
    ck(d_nosp_full < 1e-5,
       f"combined: 非SP vs true full-batch max|Δ| = {d_nosp_full:.2e} < 1e-5 "
       f"(fp32 归约形状差, 同 L2 [0])")
    ck(loss_delta <= 1e-6,
       f"combined: per-mb losses SP vs 非SP Δ = {loss_delta:.2e} ≤ 1e-6 "
       f"(归约分组显式对齐)")
    # [f] PP 接缝字节：SP 下减半（stage 间传的是 SP 分片形态）
    ck(results[False]['p2p_bytes'] == exp_bytes_nosp,
       f"combined p2p bytes 非SP = 2(N-1)·T·H·4 = {exp_bytes_nosp:,} "
       f"(got {results[False]['p2p_bytes']:.0f})")
    ck(results[True]['p2p_bytes'] == exp_bytes_sp,
       f"combined p2p bytes SP = 2(N-1)·(T/t)·H·4 = {exp_bytes_sp:,} = 非SP/{TP} "
       f"(got {results[True]['p2p_bytes']:.0f}; schedules.py:L2122 seq//tp)")
    # [g] MFU sanity
    ck(flops_step == 3 * N_BLOCKS * 4 * EXPANSION * T * H * H,
       f"FLOPs/step = 3·L·4·exp·T·h² = {flops_step:,} (Megatron 公式口径)")
    for sp in (False, True):
        ck(0 < mfu[sp]['mfu'] < 1.0,
           f"MFU sanity (sp={sp}): 0 < {mfu[sp]['mfu'] * 100:.3f}% < 100% (CPU/gloo)")
    ck(0.25 < mfu[True]['mfu'] / mfu[False]['mfu'] < 4.0,
       f"MFU SP/非SP = {mfu[True]['mfu'] / mfu[False]['mfu']:.2f} ∈ (0.25, 4.0) "
       f"(CPU/gloo: list 版 AG/RS 单次开销大 + 集合次数 5 vs 2 → SP 实测更慢; "
       f"GPU/NCCL 上通信量仅 1.25×, 吞吐应近似 [TODO: verify on real system])")

    # ------------------------------------------------------------------
    # 输出（rank 0）
    # ------------------------------------------------------------------
    if rank == 0:
        print(f"\n[0] SP mechanism (single LN+MLP block, TP={TP}): 等价性")
        print(f"    dW1/dW2 shard: SP 与 TP 各自 vs dense 参照 max|Δ| = "
              f"{max(d_vs_ref['dW1'], d_vs_ref['dW2']):.1e} (bit 级)")
        print(f"    dγ/dβ: max|Δ| = {max(d_vs_ref['dgamma'], d_vs_ref['dbeta']):.1e}"
              f" (相对 {max(d_vs_ref['dgamma'] / mech['gamma_scale'], d_vs_ref['dbeta'] / mech['beta_scale']):.1e}"
              f", fp32 归约顺序差)")
        print(f"    dX: SP 分片 vs TP 全序列对应切片 max|Δ| = {dX_sp_vs_tp:.1e}")
        print(f"    SP γ/β 梯度 = 序列分片部分和: all-reduce 前 vs dense Δ = "
              f"{partial_delta:.3e} ❌, "
              f"all-reduce 后 Δ = {max(d_vs_ref['dgamma'], d_vs_ref['dbeta']):.1e} ✅")
        print(f"    结论: all-reduce ≡ reduce-scatter + all-gather 不只是通信恒等式——")
        print(f"          拆开后中间能插序列维切分的 LN/Dropout，数学不变")
        print(f"\n[1] communication: 分解中性，重放加价")
        print(f"    TP/block: 1 AR fwd + 1 AR bwd = wire 2m = {comm_tp['wire']:.0f} B")
        print(f"    SP/block: fwd(AG+RS) + bwd(AG+RS) + 1 重放AG(wgrad) = wire 2.5m "
              f"= {comm_sp['wire']:.0f} B = TP×1.25")
        print(f"    重放 = 不存 gathered 输入的代价（layers.py:L609-618）——"
              f"存了就把省下的显存吐回去")
        print(f"\n[2] activation ledger: SP 的收益只在 TP 未切区域")
        print(f"    区域激活/block/rank (LN xhat+rstd+输出+掩码): "
              f"TP = {act['tp_region']:.0f} B → SP = {act['sp_region']:.0f} B "
              f"(= 1/{TP})")
        print(f"    TP 已切激活/block/rank (gelu_in+a, [T, FF/t]): "
              f"{act['tp_sharded']:.0f} B, 两者相同")
        print(f"\n[3] dropout 掩码必须随序列切（反例）")
        print(f"    非 SP 误用 per-rank forked 掩码: dW1 Δ = {forked_dW1:.3e} ❌ "
              f"(复制流分叉)")
        print(f"    正确做法: 掩码是位置数据——全序列生成、按分片切片 "
              f"(SP 区域 fork RNG, transformer_block.py:L595)")
        print(f"\n[4] combined: PP{PP} × TP{TP} = {world} ranks, GPipe m={M}, "
              f"一步 Adam")
        print(f"    步后权重: SP vs 非SP max|Δ| = {d_sp_nosp:.1e}; "
              f"vs dense 镜像参照: 非SP {d_nosp_ref:.1e} / SP {d_sp_ref:.1e}")
        print(f"    per-mb losses = {['%.6f' % v for v in ref_mb_losses]}  "
              f"(SP vs 非SP Δ = {loss_delta:.1e})")
        print(f"    PP 接缝字节/step/rank: 非SP = {results[False]['p2p_bytes']:.0f} B "
              f"[mb,H], SP = {results[True]['p2p_bytes']:.0f} B [mb/t,H] = 1/{TP}")
        print(f"    LN γ/β 梯度: SP 下为序列分片部分和, 步前跨 TP 组 all-reduce "
              f"(finalize_model_grads.py:L416)")
        print(f"\n[5] MFU (GEMM-calibrated peak, Megatron FLOPs 公式, per-rank 口径)")
        print(f"    model FLOPs/step = {flops_step:,} = 3 × {N_BLOCKS} blocks × "
              f"4·{EXPANSION}·T·h² (training.py:L428/L548); per rank = /TP/PP "
              f"= {flops_rank:,}")
        print(f"    elapsed[calib]: GEMM peak = {peak_gflops:.1f} GFLOP/s "
              f"(fp32, {GEMM_N}³×{GEMM_ITERS}×{GEMM_ROUNDS}, "
              f"{THREADS_PER_PROC} threads)")
        print(f"    elapsed[mfu-dense]: {dense_wall * 1e3:.1f} ms/step → "
              f"MFU(dense, 无通信) = {mfu_dense * 100:.2f}%")
        for sp in (False, True):
            tag = 'sp' if sp else 'nosp'
            print(f"    elapsed[mfu-{tag}]: {mfu[sp]['wall'] * 1e3:.1f} ms/step → "
                  f"achieved = {mfu[sp]['achieved']:.2f} GFLOP/s, "
                  f"MFU = {mfu[sp]['mfu'] * 100:.2f}%")
        print(f"    解读: MFU(dense) = 计算效率上界; 分布式 MFU 再扣通信/调度; "
              f"SP wall/非SP wall = {mfu[True]['wall'] / mfu[False]['wall']:.2f}× "
              f"(CPU/gloo list 版 AG/RS 开销 + 集合 5 vs 2; GPU/NCCL 通信量仅 1.25×)")
        print(f"    CPU/gloo 绝对值低 = 通信主导; GPU/NCCL 真机 MFU 与 SP 显存收益 "
              f"[TODO: verify on real system]")

        print(f"\n[6] self-check")
        n_pass = sum(ok for ok, _ in checks)
        for ok, msg in checks:
            print(f"    {'PASS' if ok else 'FAIL'}  {msg}")
        assert n_pass == len(checks), f'self-check failed: {len(checks) - n_pass} item(s)'
        print(f"    ✅ self-check passed ({n_pass}/{len(checks)})")

    # 确定性 digest：全部可复现指标（不含计时/MFU）
    digest_src = {
        'mech': {k: round(v, 9) for k, v in d_vs_ref.items()},
        'partial_delta': round(partial_delta, 9),
        'dX_sp_vs_tp': round(dX_sp_vs_tp, 9),
        'forked': round(forked_dW1, 9),
        'comm': {'tp': comm_tp, 'sp': comm_sp},
        'act': act,
        'combined': {
            'd_sp_nosp': round(d_sp_nosp, 9),
            'd_nosp_ref': round(d_nosp_ref, 9),
            'd_sp_ref': round(d_sp_ref, 9),
            'd_nosp_full': round(d_nosp_full, 9),
            'loss_delta': round(loss_delta, 9),
            'p2p_nosp': results[False]['p2p_bytes'],
            'p2p_sp': results[True]['p2p_bytes'],
        },
        'losses': [round(v, 6) for v in ref_mb_losses],
        'flops_step': flops_step,
    }
    digest = hashlib.md5(json.dumps(digest_src, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        print(f"\ndigest(md5 of metrics) = {digest}")


def main():
    t_start = time.perf_counter()
    print("=" * 72)
    print("nano-megatron L3 — sequence parallelism: TP×PP×SP combined + MFU")
    print("=" * 72)
    print(f"model: {N_BLOCKS} x (LN + GeLU-MLP) blocks (H={H}, FF={FF}, "
          f"p_drop={DROPOUT_P}) | P = {P_TOTAL:,} | fp32 | seed={SEED}")
    print(f"cluster: {WORLD} ranks = PP{PP} x TP{TP} (gloo, CPU; SP 与 TP 同组同 degree) "
          f"| T = {T}")
    print(f"SP primitives: all-gather / reduce-scatter (Megatron mappings.py style); "
          f"AR ≡ RS+AG")

    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)

    print(f"\ntotal wall = {time.perf_counter() - t_start:.1f}s")


if __name__ == '__main__':
    main()
