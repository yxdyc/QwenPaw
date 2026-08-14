"""
nano-fsdp L3 — mixed precision + activation checkpointing: the other half of memory

L0–L2 把「模型状态」这本账算透了：16 bytes/param、ZeRO 各 stage 的切法与通信量。
但训练时显存里还有另一半——**激活（activations）**，它随 batch×seq×层数增长，
ZeRO 对它一个字节都动不了；而 mixed precision 改变的也不只是「算得快」，它重写了
账本里每一行的 dtype 构成。L3 把这两件事在真实 PyTorch 上量出来：

  实验块   内容                                                        权威对照
  -------  --------------------------------------------------------    ------------------------
  [0]      激活账本：fp32/bf16 × with/without checkpoint（真 kernel）   ZeRO §3.2 / 1604.06174
  [1]      为什么需要 master weights：bf16 Adam 吞掉小更新（真 Adam）    ZeRO §3.1 / DeepSpeed
  [2]      混合精度账本：16Ψ 的三种拆法（公式 + 实测 bytes/param）       ZeRO §3.1 (K=12)
  [3]      真 FSDP mixed precision：每 rank 模型状态（2 进程 gloo）      FSDP MixedPrecision
  [4]      真 FSDP mixed precision：每步通信字节（dtype 感知计量）        FSDP1 _flat_param.py
  [5]      FSDP + activation checkpointing：两笔账正交叠加（官方 AC API） checkpoint_wrapper
  [6]      FSDP2 fully_shard：DTensor 契约 + MP 存储 dtype（真跑）       FSDP2 文档

核心结论（先剧透，下面用机器断言验证）：
  1. mixed precision 不减少模型状态总量（fp32 16Ψ = 混合精度 16Ψ），它减少的是
     激活（bf16 减半）与 all-gather 通信（bf16 减半）——省显存靠的是 ZeRO 分片，
     两者正交叠加：16Ψ/W 的模型状态 + 减半的激活。
  2. fp32 master copy 不是可有可无的「精度保险」，而是数值上的硬需求：bf16 在 1.0
     附近的分辨率是 2^-7，lr=1e-3 的 Adam 更新被舍入直接吞掉（[1] 实测参数冻结）。
  3. activation checkpointing 用一次重算换激活显存（1604.06174 的 O(√n) 思想），
     与分片完全正交：FSDP 包裹下省出的激活字节与单进程逐字节相同（[5]）。

运行：python3 L3_mixed_precision_checkpointing.py  # ~3s, CPU
依赖：torch（本机实测 torch 2.13.0；分布式部分 2 个真实进程 + gloo，任意 CWD 可跑）
声明：计算全部真跑（真实 bf16/fp32 kernel、真实 gloo 集合通信、真实 FSDP1/FSDP2）；
      计时行（以 elapsed 开头的行）随机器浮动，输出锚点按 tutorial §2 口径掩码
      （整行删除：sed '/^[[:space:]]*elapsed/d'）。GPU 吞吐结论标 [TODO: verify on
      real system]。FSDP2 在 macOS 本机需显式 CPU device mesh（默认 mesh 自动探测
      走 torch.mps 路径在本机 build 不可用，见 tutorial §10）。
"""

import os
import sys
import gc
import json
import time
import hashlib
import warnings
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 常量（与 L1/L2 完全一致，数字可以直接对照 L2 的输出）
# ---------------------------------------------------------------------------
SEED = 7
WORLD_SIZE = 2
STEPS = 3          # 计量的训练步数（另有 1 步 warmup 不计量）
VOCAB, DIM, LAYERS = 128, 64, 2
MASTER_PORT = 29530
ACT_BATCH = (8, 16)   # 激活账本用的 batch（单进程 [0] 与 FSDP [5] 同形，便于逐字节对照）


class TinyLM(nn.Module):
    """与 L1/L2 逐字相同的 TinyLM：116,480 参数，CPU 秒开。"""

    def __init__(self, vocab_size: int = VOCAB, dim: int = DIM, n_layers: int = LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=4, dim_feedforward=dim * 4,
                batch_first=True, dropout=0.0,
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


P_TOTAL = count_params(TinyLM())          # 116,480


def mib(b: float) -> float:
    return b / (1024 * 1024)


def make_data(rank, shape=(4, 16)):
    """每个 rank 的固定数据：seed = SEED + rank。"""
    g = torch.Generator().manual_seed(SEED + rank)
    x = torch.randint(0, VOCAB, shape, generator=g)
    y = torch.randint(0, VOCAB, shape, generator=g)
    return x, y


def fsdp_policy():
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.TransformerEncoderLayer},
    )


# ---------------------------------------------------------------------------
# [0] 激活账本：saved_tensors_hooks 直接量 autograd 为 backward 保存的字节
#
# SaveMeter 与 10:00 轮 scratch（workspace 9d88b729/fsdpL3/exp1_local.py，
# 机器复现实验）同构：outer saved_tensors_hooks 记 live/peak。
# 本正式版增加两个计数器，把「重算到底发生没有、hook 为什么看不见」量出来：
#   body_calls  —— forward 体执行次数（在 forward 体内打点，重算必然计入）
#   hook_calls  —— nn.Module forward hook 触发次数
# 两者在 checkpoint 下的差就是教程 §5 的机制发现（_StopRecomputationError）。
# ---------------------------------------------------------------------------
class SaveMeter:
    def __init__(self):
        self.live = 0
        self.peak = 0
        self.after_fwd = 0
        self.n_pack = 0
        self.ids = {}

    def pack(self, t):
        if torch.is_tensor(t):
            b = t.numel() * t.element_size()
            self.n_pack += 1
            self.live += b
            self.peak = max(self.peak, self.live)
            self.ids[id(t)] = b
        return t

    def unpack(self, t):
        if torch.is_tensor(t):
            b = self.ids.pop(id(t), None)
            if b is not None:
                self.live -= b
        return t


class CountedBlock(nn.Module):
    """给 TransformerEncoderLayer 套一层 forward 体打点（重算也会经过这里）。"""

    def __init__(self, inner, body_calls):
        super().__init__()
        self.inner = inner
        self.body_calls = body_calls

    def forward(self, x):
        self.body_calls.append(1)
        return self.inner(x)


def run_activation_combo(dtype, use_ckpt):
    """单进程跑一个 (dtype, checkpoint) 组合，返回激活账本 + 两个计数器。"""
    torch.manual_seed(SEED)
    model = TinyLM().to(dtype)
    body_calls = []
    blocks = [CountedBlock(b, body_calls) for b in model.blocks]
    hook_calls = []
    for b in blocks:
        b.register_forward_hook(lambda *a: hook_calls.append(1))
    x, y = make_data(0, ACT_BATCH)
    meter = SaveMeter()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack):
        opt.zero_grad()
        h = model.embed(x)
        for b in blocks:
            h = checkpoint(b, h, use_reentrant=False) if use_ckpt else b(h)
        out = model.head(model.norm(h))
        loss = F.cross_entropy(out.view(-1, VOCAB), y.view(-1))
        meter.after_fwd = meter.live
        loss.backward()
    opt.step()
    return dict(dtype=str(dtype).split('.')[-1], ckpt=use_ckpt, loss=loss.item(),
                after_fwd=meter.after_fwd, npack=meter.n_pack,
                body=len(body_calls), hooks=len(hook_calls))


# ---------------------------------------------------------------------------
# [1] master weights：bf16 参数上跑真 Adam，看小更新被舍入吞掉
# ---------------------------------------------------------------------------
def run_master_weights(steps=5, lr=1e-3):
    out = {}
    for name, dt in (('bf16', torch.bfloat16), ('fp32', torch.float32)):
        p = torch.ones(8, dtype=dt, requires_grad=True)
        opt = torch.optim.Adam([p], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            p.grad = torch.full_like(p, 0.01)
            opt.step()
        st = opt.state[p]
        out[name] = dict(p0=p[0].item(), delta=p[0].item() - 1.0,
                         exp_avg_dtype=str(st['exp_avg'].dtype).split('.')[-1])
    # torch 2.x 强制梯度 dtype 与参数 grad_dtype 一致——混合精度 policy 的语义面
    p = torch.randn(8, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(8, dtype=torch.float32)
    try:
        p.grad = g
        out['grad_dtype_guard'] = 'NO ERROR (unexpected)'
    except RuntimeError as e:
        out['grad_dtype_guard'] = 'RuntimeError'
        out['grad_dtype_guard_msg_has_grad_dtype'] = 'grad_dtype' in str(e)
    out['eps'] = {n: torch.finfo(dt).eps for n, dt in
                  (('bf16', torch.bfloat16), ('fp16', torch.float16),
                   ('fp32', torch.float32))}
    return out


# ---------------------------------------------------------------------------
# [2] 混合精度账本：三种 dtype 体制下真跑 Adam，量每 param 的字节构成
# ---------------------------------------------------------------------------
def run_ledger(n=1024, lr=1e-3):
    """fp32 / bf16-naive / mixed 三体制，各跑一步真 Adam 后盘点状态字节。"""
    rows = {}

    def measure(params, opt, grad_bytes, master_bytes=0):
        pb = sum(p.numel() * p.element_size() for p in params)
        ob = sum(v.numel() * v.element_size()
                 for p in params for v in opt.state.get(p, {}).values()
                 if torch.is_tensor(v) and v.dtype in (torch.float32, torch.bfloat16))
        # 只计 m/v 两个与参数量成正比的状态张量（step 计数器是每参数组几 B 零头）
        return pb, grad_bytes, master_bytes, ob

    # fp32 纯精度：params/grads/m/v 全 fp32
    torch.manual_seed(SEED)
    p32 = torch.randn(n, requires_grad=True)
    o32 = torch.optim.Adam([p32], lr=lr)
    p32.grad = torch.randn(n)
    o32.step()
    rows['fp32'] = measure([p32], o32, n * 4)

    # bf16 naive：参数/梯度/Adam 状态全 bf16（torch Adam 状态跟随参数 dtype）
    torch.manual_seed(SEED)
    pb16 = torch.randn(n, dtype=torch.bfloat16, requires_grad=True)
    ob16 = torch.optim.Adam([pb16], lr=lr)
    pb16.grad = torch.randn(n, dtype=torch.bfloat16)
    ob16.step()
    rows['bf16-naive'] = measure([pb16], ob16, n * 2)

    # mixed（ZeRO §3.1 布局 / DeepSpeed single_partition_of_fp32_groups）：
    # bf16 参数与梯度只用于计算，fp32 master 分片 + fp32 m/v 由优化器持有
    torch.manual_seed(SEED)
    p_lp = torch.randn(n, dtype=torch.bfloat16)
    master = p_lp.detach().clone().to(torch.float32).requires_grad_(True)
    o_mixed = torch.optim.Adam([master], lr=lr)
    g_lp = torch.randn(n, dtype=torch.bfloat16)          # backward 产出 bf16 梯度
    master.grad = g_lp.to(torch.float32)                  # cast 回 fp32 再入优化器
    o_mixed.step()
    with torch.no_grad():
        p_lp.copy_(master.to(torch.bfloat16))             # 更新写回低精度参数
    # 优化器状态挂在 master 上（DeepSpeed 同款：param_group['params'] = fp32 分片）
    _, _, _, ob_mixed = measure([master], o_mixed, 0)
    rows['mixed'] = (n * 2, n * 2, n * 4, ob_mixed)   # lp params / lp grads / master / m+v
    return rows, n


# ---------------------------------------------------------------------------
# dtype 感知通信计量表（L2 CommMeter 升级版）：记 (kind, 完整张量元素数, dtype)
# 口径不变（ZeRO 论文 arXiv:1910.02054 §7）：all-reduce(P)=2P，
# reduce-scatter(P)/all-gather(->P) 各记 P；L3 再乘 dtype 字节数折算成通信字节。
# ---------------------------------------------------------------------------
class CommMeter:
    OPS = {
        'all_reduce':            'ar',
        'all_gather_single':     'gather',
        'all_gather':            'gather',
        '_all_gather_base':      'gather',
        'reduce_scatter_single': 'rs',
        'reduce_scatter':        'rs',
        '_reduce_scatter_base':  'rs',
    }

    def __init__(self):
        self.enabled = False
        self.calls = []          # (kind, elems_of_full_tensor, dtype)
        self._orig = {}

    def _elems_dtype(self, fn, args, kwargs):
        # FSDP1 用位置参数、FSDP2 用关键字参数（_fsdp_collectives.py 的
        # reduce_scatter_single(output=..., input=...)），两种形态都要能拦
        def arg(i, *names):
            if i < len(args):
                return args[i]
            for n in names:
                if n in kwargs:
                    return kwargs[n]
            raise TypeError(f'{fn}: missing argument')

        if fn == 'all_reduce':
            t = arg(0, 'tensor')
            return t.numel(), t.dtype
        if fn in ('all_gather_single', '_all_gather_base'):
            t = arg(0, 'output_tensor', 'output')
            return t.numel(), t.dtype
        if fn in ('reduce_scatter_single', '_reduce_scatter_base'):
            t = arg(1, 'input')
            return t.numel(), t.dtype
        if fn == 'all_gather':
            tl = arg(0, 'tensor_list')
            return sum(t.numel() for t in tl), tl[0].dtype
        if fn == 'reduce_scatter':
            ol = arg(0, 'output_list')
            return sum(t.numel() for t in ol), ol[0].dtype
        raise ValueError(fn)

    def install(self):
        for fn in self.OPS:
            orig = getattr(dist, fn)
            self._orig[fn] = orig

            def make_wrapped(fn_name, orig_fn):
                def wrapped(*a, **k):
                    if self.enabled:
                        e, dt = self._elems_dtype(fn_name, a, k)
                        self.calls.append((self.OPS[fn_name], e, dt))
                    return orig_fn(*a, **k)
                return wrapped

            setattr(dist, fn, make_wrapped(fn, orig))

    def uninstall(self):
        for fn, orig in self._orig.items():
            setattr(dist, fn, orig)
        self._orig = {}

    def reset(self):
        self.calls = []

    def bytes_by_kind(self):
        """{'gather': B, 'rs': B, 'ar': B} + 各 kind 的 dtype 集合。"""
        out = {}
        dts = {}
        for kind, elems, dt in self.calls:
            out[kind] = out.get(kind, 0) + elems * dt.itemsize
            dts.setdefault(kind, set()).add(str(dt).split('.')[-1])
        return out, {k: sorted(v) for k, v in dts.items()}


METER = CommMeter()


def state_bytes(params, optimizer):
    """模型状态字节（L2 口径）：params + grads + Adam 张量状态。
    DTensor（FSDP2）按本 rank 驻留的 _local_tensor 计。"""
    def t_bytes(t):
        loc = getattr(t, '_local_tensor', t)
        return loc.numel() * loc.element_size()

    params = list(params)
    pb = sum(t_bytes(p) for p in params)
    gb = sum(t_bytes(p.grad) for p in params if p.grad is not None)
    ob = 0
    for p in params:
        for v in optimizer.state.get(p, {}).values():
            if torch.is_tensor(v):
                ob += t_bytes(v)
    return pb, gb, ob


def train_loop_fsdp(model, optimizer, x, y, measure_losses):
    """L2 同款循环：1 warmup + STEPS 计量；loss 跨 rank 聚合平均（= 全 batch loss）。"""
    losses = []
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
    loss.backward()
    optimizer.step()
    METER.reset()
    METER.enabled = True
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        optimizer.step()
        lv = torch.tensor([loss.item()])
        gathered = [torch.zeros(1) for _ in range(dist.get_world_size())]
        METER.enabled = False
        dist.all_gather(gathered, lv)
        METER.enabled = True
        losses.append(sum(t.item() for t in gathered) / dist.get_world_size())
    METER.enabled = False
    if measure_losses is not None:
        measure_losses.extend(losses)


# ---------------------------------------------------------------------------
# 分布式模式（全部真实多进程 + 真实 gloo 通信）
# ---------------------------------------------------------------------------
def mode_fsdp_fp32(rank, x, y, m):
    """L2 fsdp3 的同款：FULL_SHARD fp32（use_orig_params=True，L3 统一口径）。"""
    torch.manual_seed(SEED)
    model = FSDP(TinyLM(), auto_wrap_policy=fsdp_policy(),
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 device_id=torch.device('cpu'), use_orig_params=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    train_loop_fsdp(model, optimizer, x, y, losses)
    m['losses'] = losses
    m['pb'], m['gb'], m['ob'] = state_bytes(model.parameters(), optimizer)
    m['comm_bytes'], m['comm_dtypes'] = METER.bytes_by_kind()
    m['param_dtype'] = str(next(model.parameters()).dtype).split('.')[-1]
    return model


def mode_fsdp_mp(rank, x, y, m):
    """真 FSDP1 mixed precision：param_dtype=bf16（计算+all-gather），
    reduce_dtype=fp32（梯度归约），存储保持 fp32（= master weights）。"""
    torch.manual_seed(SEED)
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16,
                               reduce_dtype=torch.float32,
                               buffer_dtype=torch.bfloat16)
    model = FSDP(TinyLM(), auto_wrap_policy=fsdp_policy(),
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=mp_policy,
                 device_id=torch.device('cpu'), use_orig_params=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    train_loop_fsdp(model, optimizer, x, y, losses)
    m['losses'] = losses
    m['pb'], m['gb'], m['ob'] = state_bytes(model.parameters(), optimizer)
    m['comm_bytes'], m['comm_dtypes'] = METER.bytes_by_kind()
    m['param_dtype'] = str(next(model.parameters()).dtype).split('.')[-1]
    return model


def mode_fsdp_ac(rank, x8, y8, m, use_ac):
    """FULL_SHARD fp32 ± 官方 apply_activation_checkpointing；batch 与 [0] 同形。
    外层 saved_tensors_hooks 量激活——与单进程 [0] 逐字节对照用。"""
    torch.manual_seed(SEED)
    model = FSDP(TinyLM(), auto_wrap_policy=fsdp_policy(),
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 device_id=torch.device('cpu'), use_orig_params=True)
    if use_ac:
        apply_activation_checkpointing(
            model, check_fn=lambda sub: isinstance(sub, nn.TransformerEncoderLayer))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    meter = SaveMeter()
    losses = []
    with torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x8).view(-1, VOCAB), y8.view(-1))
        meter.after_fwd = meter.live
        loss.backward()
        lv = torch.tensor([loss.item()])
        gathered = [torch.zeros(1) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, lv)
        losses.append(sum(t.item() for t in gathered) / dist.get_world_size())
        optimizer.step()
    m['losses'] = losses
    m['saved_after_fwd'] = meter.after_fwd
    return model


def mode_fsdp2(rank, x, y, m, use_mp):
    """真 FSDP2 fully_shard（composable API，DTensor 分片）。
    macOS 本机默认 mesh 自动探测不可用（走 torch.mps 路径），显式 CPU mesh。"""
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh('cpu', (dist.get_world_size(),))
    torch.manual_seed(SEED)
    model = TinyLM()
    kwargs = {}
    if use_mp:
        kwargs['mp_policy'] = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                   reduce_dtype=torch.float32)
    for b in model.blocks:
        fully_shard(b, mesh=mesh, **kwargs)
    fully_shard(model, mesh=mesh, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    train_loop_fsdp(model, optimizer, x, y, losses)
    m['losses'] = losses
    m['pb'], m['gb'], m['ob'] = state_bytes(model.parameters(), optimizer)
    p0 = next(model.parameters())
    m['param_type'] = type(p0).__name__
    m['param_dtype'] = str(p0.dtype).split('.')[-1]
    loc = getattr(p0, '_local_tensor', None)
    m['local_shard_dtype'] = str(loc.dtype).split('.')[-1] if loc is not None else 'n/a'
    m['local_shard_numel_total'] = sum(
        (getattr(p, '_local_tensor', p)).numel() for p in model.parameters())
    return model


def run_reference(dtype):
    """单进程参照（ground truth）：不分片，直接在全 batch（两 rank 数据拼接）上训练。"""
    torch.manual_seed(SEED)
    model = TinyLM().to(dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0, y0 = make_data(0)
    x1, y1 = make_data(1)
    x = torch.cat([x0, x1])
    y = torch.cat([y0, y1])
    losses = []
    for _ in range(1 + STEPS):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses[1:]


# ---------------------------------------------------------------------------
# worker：rank 0 汇总打印 + self-check（输出顺序确定）
# ---------------------------------------------------------------------------
def worker(rank, world_size, port, single_results):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0')
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    METER.install()
    x, y = make_data(rank)
    x8, y8 = make_data(rank, ACT_BATCH)
    ref_fp32 = run_reference(torch.float32)
    ref_bf16 = run_reference(torch.bfloat16)

    results = {}
    m = {}; mode_fsdp_fp32(rank, x, y, m);            results['fsdp3_f32'] = m; gc.collect()
    m = {}; mode_fsdp_mp(rank, x, y, m);               results['fsdp3_mp'] = m;  gc.collect()
    m = {}; mode_fsdp_ac(rank, x8, y8, m, False);      results['fsdp_ac0'] = m;  gc.collect()
    m = {}; mode_fsdp_ac(rank, x8, y8, m, True);       results['fsdp_ac1'] = m;  gc.collect()
    m = {}; mode_fsdp2(rank, x, y, m, False);          results['fsdp2_f32'] = m; gc.collect()
    m = {}; mode_fsdp2(rank, x, y, m, True);           results['fsdp2_mp'] = m;  gc.collect()

    if rank == 0:
        report(single_results, results, ref_fp32, ref_bf16)
    dist.destroy_process_group()


def report(single, results, ref_fp32, ref_bf16):
    P = P_TOTAL
    W = WORLD_SIZE

    # ---------------- [0] 激活账本 ----------------
    print(f"\n[0] ACTIVATION LEDGER（单进程，saved_tensors_hooks 实测；batch={ACT_BATCH}）")
    print(f"    {'dtype':<9} {'ckpt':<6} {'saved after fwd':>16} {'n_pack':>7} "
          f"{'body':>5} {'hook':>5} {'loss':>16}")
    print("    " + "-" * 72)
    act = single['act']
    for r in act:
        print(f"    {r['dtype']:<9} {str(r['ckpt']):<6} {r['after_fwd']:>16,} "
              f"{r['npack']:>7} {r['body']:>5} {r['hooks']:>5} {r['loss']:>16.13f}")
    f32 = next(r for r in act if r['dtype'] == 'float32' and not r['ckpt'])
    f32c = next(r for r in act if r['dtype'] == 'float32' and r['ckpt'])
    b16 = next(r for r in act if r['dtype'] == 'bfloat16' and not r['ckpt'])
    b16c = next(r for r in act if r['dtype'] == 'bfloat16' and r['ckpt'])
    print(f"    checkpoint 省激活：fp32 {f32['after_fwd']:,}→{f32c['after_fwd']:,} B"
          f"（-{100*(1-f32c['after_fwd']/f32['after_fwd']):.1f}%），"
          f"bf16 {b16['after_fwd']:,}→{b16c['after_fwd']:,} B")
    print(f"    重算计数：ckpt 下 forward 体执行 {f32c['body']} 次（2 块 × 首遍+重算），"
          f"而 forward hook 只触发 {f32c['hooks']} 次——重算被 _StopRecomputationError "
          f"提前终止，hook 永远等不到 forward 返回（机制见 tutorial §5）")

    # ---------------- [1] master weights ----------------
    mw = single['mw']
    print(f"\n[1] WHY MASTER WEIGHTS（bf16 参数上跑真 Adam，lr=1e-3，5 步，初值 1.0）")
    print(f"    bf16 params: p[0] = {mw['bf16']['p0']!r}（Δ = {mw['bf16']['delta']:+.1e}，"
          f"更新被舍入吞掉——参数冻结）| exp_avg dtype = {mw['bf16']['exp_avg_dtype']}")
    print(f"    fp32 params: p[0] = {mw['fp32']['p0']!r}（Δ = {mw['fp32']['delta']:+.4f}，"
          f"正常前进）| exp_avg dtype = {mw['fp32']['exp_avg_dtype']}")
    print(f"    分辨率：eps(bf16) = {mw['eps']['bf16']}，eps(fp16) = {mw['eps']['fp16']}，"
          f"eps(fp32) = {mw['eps']['fp32']:.3e}（torch.finfo 实测）")
    print(f"    torch 梯度 dtype 守卫：bf16 参数赋 fp32 梯度 → {mw['grad_dtype_guard']}"
          f"（消息含 'grad_dtype'：{mw['grad_dtype_guard_msg_has_grad_dtype']}）")

    # ---------------- [2] 混合精度账本 ----------------
    rows, n = single['ledger']
    print(f"\n[2] MIXED-PRECISION LEDGER（n = {n} 参数，真跑一步 Adam 后盘点；B/param）")
    print(f"    {'regime':<11} {'params':>7} {'grads':>7} {'master':>7} {'m+v':>7} | {'total':>7}")
    print("    " + "-" * 52)
    ledger_tot = {}
    for name in ('fp32', 'bf16-naive', 'mixed'):
        pb, gb, mb, ob = rows[name]
        tot = pb + gb + mb + ob
        ledger_tot[name] = tot / n
        print(f"    {name:<11} {pb/n:>7.1f} {gb/n:>7.1f} {mb/n:>7.1f} {ob/n:>7.1f} | "
              f"{tot/n:>7.1f}")
    print(f"    ZeRO 论文（arXiv:1910.02054 §3.1）口径：2Ψ+2Ψ+KΨ = 16Ψ，K=12——"
          f"mixed 一行就是它的实测展开")
    print(f"    注意 total：mixed({ledger_tot['mixed']:.0f}) == fp32({ledger_tot['fp32']:.0f})"
          f" ≠ 减半——混合精度不省模型状态，省的是激活（[0]）与通信（[4]）；"
          f"bf16-naive 的 8 B/param 以参数冻结为代价（[1]），不可用")

    # ---------------- [3] FSDP mixed precision：每 rank 模型状态 ----------------
    print(f"\n[3] per-rank MODEL STATE（真 FSDP FULL_SHARD，W={W}；params+grads+Adam）")
    print(f"    {'mode':<11} {'params':>9} {'grads':>9} {'opt':>9} | {'total':>9} "
          f"{'storage dtype':<14} {'formula':<14}")
    print("    " + "-" * 78)
    state_rows = {}
    for name, key in (('fsdp3_f32', 'fsdp3_f32'), ('fsdp3_mp', 'fsdp3_mp'),
                      ('fsdp2_f32', 'fsdp2_f32'), ('fsdp2_mp', 'fsdp2_mp')):
        r = results[key]
        tot = r['pb'] + r['gb'] + r['ob']
        state_rows[name] = tot
        print(f"    {name:<11} {mib(r['pb']):8.2f}M {mib(r['gb']):8.2f}M "
              f"{mib(r['ob']):8.2f}M | {mib(tot):8.2f}M {r['param_dtype']:<14} "
              f"{'16P/W':<14}")
    print(f"    （M = MiB；16P/W = {mib(16*P//W):.2f} MiB。**mp 两行存储仍是 fp32**——"
          f"低精度只发生在计算与 all-gather 通信上，fp32 shard 就是 master weights）")

    # ---------------- [4] FSDP mixed precision：每步通信字节 ----------------
    print(f"\n[4] per-step COMM BYTES（dtype 感知计量；口径同 L2：gather/rs 各记完整张量）")
    print(f"    {'mode':<11} {'gather':>12} {'reduce-scatter':>15} {'total':>12} "
          f"{'gather dtype':<14}")
    print("    " + "-" * 70)
    comm_tot = {}
    for name in ('fsdp3_f32', 'fsdp3_mp'):
        cb, cd = results[name]['comm_bytes'], results[name]['comm_dtypes']
        g, rs = cb.get('gather', 0), cb.get('rs', 0)
        comm_tot[name] = g + rs
        print(f"    {name:<11} {g/1e6:9.3f} MB {rs/1e6:12.3f} MB {(g+rs)/1e6:9.3f} MB "
              f"{','.join(cd.get('gather', ['-'])):<14}")
    ratio = comm_tot['fsdp3_mp'] / comm_tot['fsdp3_f32']
    print(f"    mixed precision 通信 = fp32 的 {100*ratio:.1f}%：gather 减半（bf16），"
          f"reduce-scatter 不变（reduce_dtype=fp32 保数值稳定）")

    # ---------------- [5] FSDP + activation checkpointing ----------------
    print(f"\n[5] FSDP + ACTIVATION CHECKPOINTING（官方 apply_activation_checkpointing）")
    a0, a1 = results['fsdp_ac0'], results['fsdp_ac1']
    print(f"    {'场景':<34} {'saved after fwd':>16}")
    print("    " + "-" * 54)
    print(f"    {'单进程 fp32（[0] 参照）':<34} {f32['after_fwd']:>16,}")
    print(f"    {'单进程 fp32 + checkpoint（[0] 参照）':<34} {f32c['after_fwd']:>16,}")
    print(f"    {'FSDP FULL_SHARD fp32':<34} {a0['saved_after_fwd']:>16,}")
    print(f"    {'FSDP FULL_SHARD fp32 + AC':<34} {a1['saved_after_fwd']:>16,}")
    print(f"    loss（rank 平均）：no-AC {a0['losses'][0]:.6f} vs AC {a1['losses'][0]:.6f}"
          f"——fp32 重算逐位一致")
    print(f"    激活账本在 FSDP 包裹下与单进程逐字节相同：ZeRO 切模型状态、checkpoint 切激活，"
          f"两笔账正交叠加")

    # ---------------- [6] FSDP2 ----------------
    print(f"\n[6] FSDP2 fully_shard（composable API，DTensor 分片，真跑）")
    f2, f2m = results['fsdp2_f32'], results['fsdp2_mp']
    print(f"    type(param) = {f2['param_type']}；本 rank local shard 元素总数 = "
          f"{f2['local_shard_numel_total']:,} = P/{W}（{P:,}/{W}）")
    print(f"    fsdp2_f32 每 rank 状态 = {mib(state_rows['fsdp2_f32']):.2f} MiB（= 16P/W）；"
          f"fsdp2_mp local shard dtype = {f2m['local_shard_dtype']}"
          f"（MP policy 下存储仍 fp32——sharded 高精度参数即 master weights，"
          f"FSDP2 文档原话见 tutorial §9）")

    # ---------------- [7] correctness ----------------
    print(f"\n[7] correctness：各模式 vs 单进程参照（loss 为 rank 平均 = 全 batch loss）")
    print(f"    {'mode':<11} {'step losses':<28} {'max|Δ| vs ref':>14}")
    print("    " + "-" * 56)
    deltas = {}
    for name, ref in (('fsdp3_f32', ref_fp32), ('fsdp3_mp', ref_bf16),
                      ('fsdp2_f32', ref_fp32), ('fsdp2_mp', ref_bf16)):
        ls = results[name]['losses']
        d = max(abs(a - b) for a, b in zip(ls, ref))
        deltas[name] = d
        print(f"    {name:<11} {' '.join(f'{l:.6f}' for l in ls):<28} {d:14.3e}")
    print(f"    {'ref_fp32':<11} {' '.join(f'{l:.6f}' for l in ref_fp32):<28} {'(ground truth)':>14}")
    print(f"    {'ref_bf16':<11} {' '.join(f'{l:.6f}' for l in ref_bf16):<28} {'(bf16 参照)':>14}")
    ulp_loss = torch.finfo(torch.bfloat16).eps * 4     # loss ∈ [4,8) → ULP = eps×2^2
    print(f"    fsdp3_mp/fsdp2_mp 的 Δ 是 bf16 量化噪声（恰 1 ULP：loss ∈ [4,8) 的 ULP = eps×4 = "
          f"{ulp_loss}）；fp32 模式的 Δ 是归约结构舍入（同 L2 §8）")

    # ---------------- [8] self-check ----------------
    checks = []

    def ck(ok, msg):
        checks.append((bool(ok), msg))

    # [0] 激活账本
    ck(f32c['after_fwd'] < f32['after_fwd'], 'ckpt saves less activation (fp32)')
    ck(b16c['after_fwd'] < b16['after_fwd'], 'ckpt saves less activation (bf16)')
    ck(0.45 < b16['after_fwd'] / f32['after_fwd'] < 0.55,
       f"bf16 activation ≈ fp32/2 (ratio {b16['after_fwd']/f32['after_fwd']:.4f})")
    ck(f32c['loss'] == f32['loss'], 'ckpt recompute bit-exact loss (fp32)')
    ck(b16c['loss'] == b16['loss'], 'ckpt recompute bit-exact loss (bf16)')
    ck(f32['body'] == LAYERS, f"no-ckpt forward body execs == {LAYERS}")
    ck(f32c['body'] == 2 * LAYERS, f"ckpt forward body execs == 2×{LAYERS} (recompute real)")
    ck(f32c['hooks'] == LAYERS, 'ckpt forward hooks fire only on first pass (early-stop abort)')
    ck(f32c['npack'] < f32['npack'], 'ckpt packs fewer saved tensors')
    # [1] master weights
    ck(mw['bf16']['delta'] == 0.0, 'bf16 Adam: param frozen (update swallowed by rounding)')
    ck(mw['fp32']['delta'] < -4e-3, 'fp32 Adam: param advances')
    ck(mw['grad_dtype_guard'] == 'RuntimeError'
       and mw['grad_dtype_guard_msg_has_grad_dtype'],
       'torch enforces grad dtype == param grad_dtype (RuntimeError)')
    ck(mw['bf16']['exp_avg_dtype'] == 'bfloat16', 'Adam state follows param dtype (bf16)')
    # [2] 账本
    ck(abs(ledger_tot['fp32'] - 16) < 0.01, 'fp32 regime = 16 B/param (measured)')
    ck(abs(ledger_tot['bf16-naive'] - 8) < 0.01, 'bf16-naive regime = 8 B/param (measured)')
    ck(abs(ledger_tot['mixed'] - 16) < 0.01, 'mixed regime = 16 B/param (measured)')
    ck(ledger_tot['mixed'] == ledger_tot['fp32'],
       'mixed total == fp32 total (MP 不省模型状态)')
    # [3] 模型状态
    exp16 = 16 * P // W
    for name in ('fsdp3_f32', 'fsdp3_mp', 'fsdp2_f32', 'fsdp2_mp'):
        ck(abs(state_rows[name] - exp16) / exp16 < 0.015,
           f"{name} per-rank state {state_rows[name]} within 1.5% of 16P/W={exp16}")
    ck(results['fsdp3_mp']['param_dtype'] == 'float32',
       'FSDP1 MP: storage stays fp32 (master weights)')
    # [4] 通信
    g32 = results['fsdp3_f32']['comm_bytes'].get('gather', 0)
    gmp = results['fsdp3_mp']['comm_bytes'].get('gather', 0)
    rs32 = results['fsdp3_f32']['comm_bytes'].get('rs', 0)
    rsmp = results['fsdp3_mp']['comm_bytes'].get('rs', 0)
    ck(gmp * 2 == g32, f'MP gather bytes exactly halved ({gmp} vs {g32})')
    ck(rsmp == rs32, f'reduce-scatter bytes unchanged (fp32): {rsmp}')
    ck(comm_tot['fsdp3_mp'] < comm_tot['fsdp3_f32'], 'MP total comm < fp32 total comm')
    ck(results['fsdp3_mp']['comm_dtypes'].get('gather') == ['bfloat16'],
       'MP gather collective runs in bf16')
    # [5] AC 正交性
    ck(a0['saved_after_fwd'] == f32['after_fwd'],
       'FSDP activation bytes == single-process (sharding 不动激活)')
    ck(a1['saved_after_fwd'] == f32c['after_fwd'],
       'FSDP+AC activation bytes == single-process ckpt (正交叠加)')
    ck(a1['losses'][0] == a0['losses'][0], 'AC loss bit-exact vs no-AC (fp32)')
    # [6] FSDP2
    ck(f2['param_type'] == 'DTensor', 'FSDP2 params are DTensor')
    ck(f2['local_shard_numel_total'] == P // W, 'FSDP2 local shards sum to P/W')
    ck(f2m['local_shard_dtype'] == 'float32', 'FSDP2 MP: storage stays fp32')
    # [7] 正确性
    for name in ('fsdp3_f32', 'fsdp2_f32'):
        ck(deltas[name] < 1e-5, f'{name} losses match fp32 reference (<1e-5)')
    for name in ('fsdp3_mp', 'fsdp2_mp'):
        ck(deltas[name] < 0.04, f'{name} losses within 1 bf16 ULP of bf16 reference')

    print(f"\n[8] self-check")
    n_pass = sum(ok for ok, _ in checks)
    for ok, msg in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {msg}")
    assert n_pass == len(checks), f'self-check failed: {len(checks) - n_pass} item(s)'
    print(f"    ✅ self-check passed ({n_pass}/{len(checks)})")

    digest_src = {
        'act': act,
        'mw': {k: (v if not isinstance(v, dict) else
                   {kk: (vv if not isinstance(vv, float) else round(vv, 12))
                    for kk, vv in v.items()}) for k, v in mw.items()},
        'ledger_b_per_param': ledger_tot,
        'state': state_rows,
        'comm': {k: dict(results[k]['comm_bytes']) for k in ('fsdp3_f32', 'fsdp3_mp')},
        'saved': {'ac0': a0['saved_after_fwd'], 'ac1': a1['saved_after_fwd']},
        'losses': {k: [round(v, 6) for v in results[k]['losses']]
                   for k in results},
        'ref_fp32': [round(v, 6) for v in ref_fp32],
        'ref_bf16': [round(v, 6) for v in ref_bf16],
        'deltas': {k: round(v, 9) for k, v in deltas.items()},
        'fsdp2_local_numel': f2['local_shard_numel_total'],
    }
    digest = hashlib.md5(json.dumps(digest_src, sort_keys=True).encode()).hexdigest()
    print(f"\ndigest(md5 of metrics) = {digest}")


def main():
    t_start = time.perf_counter()
    print("=" * 72)
    print("nano-fsdp L3 — mixed precision + activation checkpointing")
    print("=" * 72)
    print(f"TinyLM: vocab={VOCAB} dim={DIM} layers={LAYERS} | P = {P_TOTAL:,} params")
    print(f"cluster: W = {WORLD_SIZE} real processes, gloo backend, CPU "
          f"(真实多进程 + 真实集合通信；生产在 GPU，机制相同)")
    print(f"steps: 1 warmup + {STEPS} measured | seed = {SEED} (与 L1/L2 一致)")

    # ---- 单进程部分 [0][1][2] ----
    t0 = time.perf_counter()
    act = []
    for dtype in (torch.float32, torch.bfloat16):
        for ck_ in (False, True):
            act.append(run_activation_combo(dtype, ck_))
            gc.collect()
    mw = run_master_weights()
    ledger, n_ledger = run_ledger()
    single = {'act': act, 'mw': mw, 'ledger': (ledger, n_ledger)}
    print(f"\nelapsed[single-process]: {time.perf_counter() - t0:.1f}s "
          f"([0][1][2] 真 kernel; 计时行浮动)")

    # ---- 分布式部分 [3]–[7] ----
    t0 = time.perf_counter()
    mp.spawn(worker, args=(WORLD_SIZE, MASTER_PORT, single), nprocs=WORLD_SIZE, join=True)
    print(f"\nelapsed[distributed]: {time.perf_counter() - t0:.1f}s "
          f"(6 模式真跑; 计时行浮动)")
    print(f"\nelapsed: {time.perf_counter() - t_start:.1f}s total "
          f"(计算真跑; 计时行浮动, 锚点口径见 tutorial §2)")


if __name__ == '__main__':
    main()
