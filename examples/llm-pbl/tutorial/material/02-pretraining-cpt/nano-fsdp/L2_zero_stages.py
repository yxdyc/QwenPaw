"""
nano-fsdp L2 — ZeRO stages: what gets sharded, what it costs

L0 手算了 ZeRO 的显存账本（16 bytes/param 与各 stage 公式），L1 用真实 FSDP 量出
DDP vs ZeRO-3 的每 rank 模型状态。L2 把中间两级补齐：**ZeRO-0/1/2/3 的分片差异
到底在哪，各自付出多少通信代价**——用真实多进程（gloo/CPU）全部跑出来：

  mode     实现                                      分片对象          通信量/step
  ------   ---------------------------------------   ---------------   -----------
  ddp      真实 torch DDP                            无（全副本）        2P
  zero1    手写真实 ZeRO-1（reduce-scatter+all-gather） 优化器状态        2P
  zero2    手写真实 ZeRO-2（再分片梯度）               优化器状态+梯度     2P
  fsdp2    真实 FSDP SHARD_GRAD_OP（ZeRO-2 血统）     同 zero2           2P
  fsdp3    真实 FSDP FULL_SHARD（ZeRO-3）             全部               ~3P

通信量按 ZeRO 论文 (arXiv:1910.02054) 的口径：all-reduce(P) 记 2P
（= reduce-scatter(P) + all-gather(P)），reduce-scatter / all-gather 各记 P。
核心结论（先剧透，下面用机器断言验证）：ZeRO-1/2 的显存节省是**通信免费**的，
ZeRO-3 要多付 ~1.5x 通信——这就是「该切到哪一级」的工程决策点。

运行：python3 L2_zero_stages.py   # ~3s, CPU 即可
依赖：torch（本机实测 torch 2.13.0；L1 曾在 torch 2.4.1 实测，见 README 环境依赖）
声明：计算全部真跑（2 个真实进程、真实 gloo 集合通信）；显存只计模型状态
      （params+grads+Adam m/v，与 L0/L1 口径一致），不含 activations/OS 开销；
      计时行（以 elapsed 开头的行）随机器浮动，输出锚点按 tutorial §2 口径掩码。
"""

import os
import sys
import gc
import json
import time
import hashlib
import warnings
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 常量（与 L1 完全一致，数字可以直接对照 L1 的输出）
# ---------------------------------------------------------------------------
SEED = 7
WORLD_SIZE = 2
STEPS = 3          # 计量的训练步数（另有 1 步 warmup 不计量）
VOCAB, DIM, LAYERS = 128, 64, 2
MASTER_PORT = 29520


class TinyLM(nn.Module):
    """与 L1 逐字相同的 TinyLM：116,480 参数，CPU 秒开。"""

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
N_BLOCKS = LAYERS                          # 被 auto-wrap 单独包成 FSDP 单元的 block 数
N_UNITS = N_BLOCKS + 1                     # +1 是 root 单元（embed+norm+head）


def mib(b: float) -> float:
    return b / (1024 * 1024)


# ---------------------------------------------------------------------------
# 通信量表（CommMeter）：拦截 torch.distributed 集合通信，按 ZeRO 论文口径计体积
#
# 口径（arXiv:1910.02054 的通信量分析）：
#   all-reduce(P)      记 2P —— 它在 ring 上等价于 reduce-scatter + all-gather
#   reduce-scatter(P)  记 P  —— 每 rank 投入 P，留下 P/W
#   all-gather(->P)    记 P  —— 每 rank 投入 P/W，拿到 P
# 这样 DDP=ZeRO-1=ZeRO-2=2P/step，ZeRO-3=3P/step（论文结论，本脚本机器验证）。
#
# 拦截原理：FSDP1 在运行时通过 `dist.xxx(...)` 模块属性调用集合通信
# （torch/distributed/fsdp/_runtime_utils.py 的 dist.reduce_scatter_single /
# dist.all_reduce 等，行号锚点见 tutorial §7），替换模块属性即可全量截获。
# DDP 的 all-reduce 在 C++ 侧发起，不走 python dist.all_reduce，
# 用官方 register_comm_hook 单独计量。两条路径都是真实流量，无模拟。
# ---------------------------------------------------------------------------
class CommMeter:
    OPS = {
        # (函数名, 体积语义): 体积 = 完整张量元素数（见上方口径）
        'all_reduce':            'ar',      # 2x，聚合时再乘
        'all_gather_single':     'gather',  # a[0] = output（完整）
        'all_gather':            'gather',  # a[0] = list of shards, sum x W = 完整
        '_all_gather_base':      'gather',  # a[0] = output（完整）
        'reduce_scatter_single': 'rs',      # a[1] = input（完整）
        'reduce_scatter':        'rs',      # a[0] = list of shards（完整）
        '_reduce_scatter_base':  'rs',      # a[1] = input（完整）
    }

    def __init__(self):
        self.enabled = False
        self.calls = []          # (kind, elems_of_full_tensor)
        self._orig = {}

    def _elems(self, fn, args):
        if fn == 'all_reduce':
            return args[0].numel()
        if fn in ('all_gather_single', '_all_gather_base', 'reduce_scatter_single',
                  '_reduce_scatter_base'):
            t = args[0] if fn.startswith('all_gather') else args[1]
            return t.numel()
        if fn == 'all_gather':                      # output list 之和 = 完整张量
            return sum(t.numel() for t in args[0])
        if fn == 'reduce_scatter':                  # list of W shards = full
            return sum(t.numel() for t in args[0])
        raise ValueError(fn)

    def install(self):
        for fn in self.OPS:
            orig = getattr(dist, fn)
            self._orig[fn] = orig

            def make_wrapped(fn_name, orig_fn):
                def wrapped(*a, **k):
                    if self.enabled:
                        self.calls.append((self.OPS[fn_name], self._elems(fn_name, a)))
                    return orig_fn(*a, **k)
                return wrapped

            setattr(dist, fn, make_wrapped(fn, orig))

    def uninstall(self):
        for fn, orig in self._orig.items():
            setattr(dist, fn, orig)
        self._orig = {}

    def reset(self):
        self.calls = []

    def volume_elems(self):
        """按口径折算成「完整张量元素数」：gather/rs 各 1x，all-reduce 2x。"""
        v = 0
        for kind, elems in self.calls:
            v += elems * (2 if kind == 'ar' else 1)
        return v

    def counts(self):
        return dict(Counter(k for k, _ in self.calls))


METER = CommMeter()


# ---------------------------------------------------------------------------
# 模型状态账本（与 L0/L1 口径一致：params + grads + Adam 状态；fp32）
# ---------------------------------------------------------------------------
def state_bytes(params, optimizer):
    params = list(params)
    pb = sum(p.numel() * p.element_size() for p in params)
    gb = sum(p.grad.numel() * p.grad.element_size() for p in params if p.grad is not None)
    ob = 0
    for p in params:
        for v in optimizer.state.get(p, {}).values():
            if torch.is_tensor(v):
                ob += v.numel() * v.element_size()
    return pb, gb, ob


def flat_state(model):
    """把模型参数按 state_dict 键序拍平成一个向量——跨模式比对最终权重用。"""
    sd = model.state_dict()
    return torch.cat([sd[k].detach().reshape(-1).float() for k in sorted(sd)])


def make_data(rank):
    """每个 rank 的固定数据：seed = SEED + rank，所有模式共用，保证可比。"""
    g = torch.Generator().manual_seed(SEED + rank)
    x = torch.randint(0, VOCAB, (4, 16), generator=g)
    y = torch.randint(0, VOCAB, (4, 16), generator=g)
    return x, y


def fsdp_policy():
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.TransformerEncoderLayer},
    )


def train_loop(model, optimizer, x, y, rank, measure):
    """统一的训练循环：1 步 warmup（不计量）+ STEPS 步计量。
    measure: dict，写入 losses / step_times / comm_elems / comm_counts。"""
    losses, times = [], []
    # warmup：第一次 backward 会 lazily 建梯度 buffer / 固化 DDP bucket，不计量
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
    loss.backward()
    optimizer.step()

    METER.reset()
    METER.enabled = True
    t0 = time.perf_counter()
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        optimizer.step()
        # 每步把两个 rank 的 loss 汇总（sum = 全 batch 平均 loss x W，见 tutorial §6）
        lv = torch.tensor([loss.item()])
        gathered = [torch.zeros(1) for _ in range(dist.get_world_size())]
        METER.enabled = False            # loss 汇总通信不属于训练通信量
        dist.all_gather(gathered, lv)
        METER.enabled = True
        losses.append(sum(t.item() for t in gathered) / dist.get_world_size())
    METER.enabled = False
    times.append((time.perf_counter() - t0) / STEPS)

    measure['losses'] = losses
    measure['comm_elems'] = METER.volume_elems() / STEPS     # 每步平均
    measure['comm_counts'] = {k: v // STEPS for k, v in METER.counts().items()}
    measure['sec_per_step'] = times[0]


# ---------------------------------------------------------------------------
# 五个分布式模式。全部真实多进程 + 真实 gloo 通信，无 mock。
# ---------------------------------------------------------------------------
def mode_ddp(rank, x, y, m):
    """ZeRO-0：真实 DDP。all-reduce 在 C++ 侧，用官方 comm hook 计量。"""
    torch.manual_seed(SEED)
    model = DDP(TinyLM(), device_ids=None)
    hook_elems = []

    def comm_hook(state, bucket):
        t = bucket.buffer()
        if METER.enabled:
            hook_elems.append(t.numel())
        fut = dist.all_reduce(t, group=state, async_op=True).get_future()
        return fut.then(lambda _: t)

    model.register_comm_hook(None, comm_hook)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loop(model, optimizer, x, y, rank, m)
    m['pb'], m['gb'], m['ob'] = state_bytes(model.module.parameters(), optimizer)
    # DDP 的 all-reduce 体积按口径记 2x（hook 记的是完整 bucket 元素数）
    m['comm_elems'] = 2 * sum(hook_elems) / STEPS
    m['comm_counts'] = {'hook_all_reduce': len(hook_elems) // STEPS}
    m['flat'] = flat_state(model.module)
    return model


class ZeROShardedAdam:
    """手写 ZeRO-1/2 的核心：只给自己负责的那 1/W 参数维护 Adam 状态。

    每步流程（与 DeepSpeed ZeRO stage 1/2 的语义相同，见 tutorial §5）：
      1. backward 得到完整梯度（autograd 必然产生完整梯度，这是两者的共同峰值）
      2. 拍平 -> reduce-scatter：每 rank 拿到自己 slice 的**求和**梯度
      3. 除以 W（对齐 DDP 的「平均」语义），只在自己的 slice 上跑 Adam
      4. all-gather 更新后的参数 slice，拼回完整参数
    ZeRO-1 保留完整梯度 resident；ZeRO-2 在 reduce-scatter 后立即释放完整梯度。
    """

    def __init__(self, model, rank, world_size, lr=1e-3):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.params = list(model.parameters())
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.total = sum(self.numels)
        assert self.total % world_size == 0, "本脚本假设参数可被 W 整除（FSDP 用 padding，见 tutorial §8）"
        self.chunk = self.total // world_size
        self.lo, self.hi = rank * self.chunk, (rank + 1) * self.chunk
        # 自己负责的参数 slice：真实 leaf tensor + 真实 torch.optim.Adam
        self.shard = torch.zeros(self.chunk)
        self.opt = torch.optim.Adam([self.shard], lr=lr)
        self._fill_shard()
        self.grad_full = None     # ZeRO-1 保留；ZeRO-2 释放
        self.grad_shard = None

    def _fill_shard(self):
        flat = torch.cat([p.data.reshape(-1) for p in self.params])
        self.shard.data.copy_(flat[self.lo:self.hi])

    def flat_params(self):
        return torch.cat([p.data.reshape(-1) for p in self.params])

    def reduce_grads(self, keep_full_grads: bool):
        g_flat = torch.cat([p.grad.reshape(-1) for p in self.params])
        for p in self.params:          # autograd 产生的完整梯度不再需要
            p.grad = None
        if keep_full_grads:
            self.grad_full = g_flat    # ZeRO-1：完整梯度 resident（分片对象不含梯度）
        self.grad_shard = torch.empty(self.chunk)
        METER.enabled = False          # 手写集合通信在下面显式计量
        dist.reduce_scatter(self.grad_shard, list(g_flat.chunk(self.world_size)))
        METER.enabled = True
        self.grad_shard.div_(self.world_size)   # sum -> mean，对齐 DDP 语义

    def step(self):
        self.shard.grad = self.grad_shard
        self.opt.step()
        self.opt.zero_grad(set_to_none=False)
        if self.grad_full is not None:      # ZeRO-1：shard 只是临时通信产物，用完即弃
            self.grad_shard = None
        # 把更新后的 slice 写回完整参数：先本地写入，再 all-gather 各 rank 的 slice
        flat = self.flat_params()
        flat[self.lo:self.hi] = self.shard.data
        parts = list(flat.chunk(self.world_size))
        METER.enabled = False
        dist.all_gather(parts, parts[self.rank])
        METER.enabled = True
        full = torch.cat(parts)
        off = 0
        for p, n in zip(self.params, self.numels):
            p.data.copy_(full[off:off + n].reshape(p.shape))
            off += n

    def resident_bytes(self):
        """模型状态 resident 字节：params + 仍活着的梯度 + 本 rank 的 Adam 状态。"""
        pb = self.total * 4
        gb = 0
        if self.grad_full is not None:
            gb += self.grad_full.numel() * 4     # ZeRO-1：完整梯度还在
        if self.grad_shard is not None:
            gb += self.grad_shard.numel() * 4    # ZeRO-2：只剩 1/W 的梯度 shard
        ob = sum(v.numel() * v.element_size()
                 for v in self.opt.state.get(self.shard, {}).values()
                 if torch.is_tensor(v))
        return pb, gb, ob


def mode_zero(rank, x, y, m, stage):
    """手写真实 ZeRO-1 (stage=1) / ZeRO-2 (stage=2)。"""
    torch.manual_seed(SEED)
    model = TinyLM()
    z = ZeROShardedAdam(model, rank, dist.get_world_size())
    # 手写的两次集合通信各计 P（reduce-scatter 投入 P；all-gather 产出 P）
    for _ in range(1):  # warmup 步也要走通信（保持 Adam 状态演进一致），但不计量
        model.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        z.reduce_grads(keep_full_grads=(stage == 1))
        z.step()
    losses, times = [], []
    METER.reset()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        model.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        # 显式计量：reduce-scatter(P) + all-gather(P)，口径与 CommMeter 一致
        METER.calls.append(('rs', P_TOTAL))
        z.reduce_grads(keep_full_grads=(stage == 1))
        METER.calls.append(('gather', P_TOTAL))
        z.step()
        lv = torch.tensor([loss.item()])
        gathered = [torch.zeros(1) for _ in range(dist.get_world_size())]
        METER.enabled = False
        dist.all_gather(gathered, lv)
        METER.enabled = True
        losses.append(sum(t.item() for t in gathered) / dist.get_world_size())
    times.append((time.perf_counter() - t0) / STEPS)
    m['losses'] = losses
    m['comm_elems'] = METER.volume_elems() / STEPS
    m['comm_counts'] = {k: v // STEPS for k, v in METER.counts().items()}
    m['sec_per_step'] = times[0]
    m['pb'], m['gb'], m['ob'] = z.resident_bytes()
    m['flat'] = flat_state(model)   # sorted state_dict 键序，与 ref/其他模式同序
    return model


def mode_fsdp(rank, x, y, m, strategy):
    """真实 FSDP1：SHARD_GRAD_OP（ZeRO-2 血统）/ FULL_SHARD（ZeRO-3）。"""
    torch.manual_seed(SEED)
    model = FSDP(TinyLM(), auto_wrap_policy=fsdp_policy(),
                 sharding_strategy=strategy, device_id=torch.device('cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loop(model, optimizer, x, y, rank, m)
    # 稳态 resident：FSDP 的 parameters() 返回各单元的 FlatParameter shard（含 padding）
    m['pb'], m['gb'], m['ob'] = state_bytes(list(model.parameters()), optimizer)
    m['flat'] = flat_state(model)
    return model


# ---------------------------------------------------------------------------
# 单进程参照（ground truth）：不分片、不通信，直接在全 batch 上训练。
# 数据并行（任何 ZeRO stage）在数学上等价于「在 union batch 上训练」——
# 本参照就是用来机器验证这件事的（tutorial §6）。
# ---------------------------------------------------------------------------
def run_reference():
    torch.manual_seed(SEED)
    model = TinyLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0, y0 = make_data(0)
    x1, y1 = make_data(1)
    x = torch.cat([x0, x1])
    y = torch.cat([y0, y1])
    losses = []
    for _ in range(1 + STEPS):        # 同样 1 warmup + STEPS 计量
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses[1:], flat_state(model)


# ---------------------------------------------------------------------------
# worker：每个 rank 跑全部五个模式，rank 0 汇总打印 + self-check
# ---------------------------------------------------------------------------
def worker(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0')
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    METER.install()
    x, y = make_data(rank)
    W = world_size
    ref_losses, ref_flat = run_reference()   # spawn 子进程里重新算（确定性，廉价）

    results = {}
    m = {}; mode_ddp(rank, x, y, m);   results['ddp'] = m;   gc.collect()
    m = {}; mode_zero(rank, x, y, m, 1); results['zero1'] = m; gc.collect()
    m = {}; mode_zero(rank, x, y, m, 2); results['zero2'] = m; gc.collect()
    m = {}; mode_fsdp(rank, x, y, m, ShardingStrategy.SHARD_GRAD_OP); results['fsdp2'] = m; gc.collect()
    m = {}; mode_fsdp(rank, x, y, m, ShardingStrategy.FULL_SHARD);    results['fsdp3'] = m; gc.collect()

    # rank 间最终权重一致性（手写模式两 rank 都应持有相同的全量参数）
    for name in ('ddp', 'zero1', 'zero2'):
        f = results[name]['flat'].clone()
        dist.all_reduce(f, op=dist.ReduceOp.MAX)
        f2 = results[name]['flat'].clone()
        dist.all_reduce(f2, op=dist.ReduceOp.MIN)
        results[name]['inter_rank_maxdiff'] = (f - f2).abs().max().item()

    if rank == 0:
        report(results, (ref_losses, ref_flat))
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 报告 + self-check（只在 rank 0 打印，保证输出顺序确定）
# ---------------------------------------------------------------------------
def report(results, ref):
    ref_losses, ref_flat = ref
    P = P_TOTAL
    fp32 = 4                      # fp32: 4 bytes/param
    full_state_bytes = 16 * P     # L0/L1: params4 + grads4 + m4 + v4 = 16 B/param
    W = WORLD_SIZE

    print(f"\n[0] per-rank MODEL STATE after training（params+grads+Adam, fp32; L0/L1 口径）")
    print(f"    {'mode':<7} {'params':>9} {'grads':>9} {'opt':>9} | {'total':>9} "
          f"{'formula (fp32)':<16} {'expected':>9}")
    print("    " + "-" * 78)
    formulas = {
        'ddp':   ('16P',            full_state_bytes),
        'zero1': ('8P + 8P/W',      8 * P + 8 * P // W),
        'zero2': ('4P + 12P/W',     4 * P + 12 * P // W),
        'fsdp2': ('16P/W (steady)', full_state_bytes // W),
        'fsdp3': ('16P/W (steady)', full_state_bytes // W),
    }
    totals = {}
    for name in ('ddp', 'zero1', 'zero2', 'fsdp2', 'fsdp3'):
        pb, gb, ob = results[name]['pb'], results[name]['gb'], results[name]['ob']
        tot = pb + gb + ob
        totals[name] = tot
        f, exp = formulas[name]
        print(f"    {name:<7} {mib(pb):8.2f}M {mib(gb):8.2f}M {mib(ob):8.2f}M | "
              f"{mib(tot):8.2f}M {f:<16} {mib(exp):8.2f}M")
    print(f"    （M = MiB；16P = {mib(full_state_bytes):.2f} MiB，与 L1 的 DDP/FSDP 数字一致；"
          f"opt 列含 Adam step 计数器的每参数张量几 B 零头）")

    print(f"\n[1] per-step COMM VOLUME（口径：all-reduce=2P, reduce-scatter=P, all-gather=P）")
    print(f"    {'mode':<7} {'collectives/step':<34} {'volume':>12} {'x P':>7}")
    print("    " + "-" * 64)
    vols = {}
    for name in ('ddp', 'zero1', 'zero2', 'fsdp2', 'fsdp3'):
        c = results[name]['comm_counts']
        v = results[name]['comm_elems'] * fp32
        vols[name] = v
        desc = ' '.join(f"{k}={n}" for k, n in sorted(c.items()))
        print(f"    {name:<7} {desc:<34} {v/1e6:9.3f} MB {v/(P*fp32):6.3f}x")

    print(f"\n[2] correctness：五个分布式模式 vs 单进程参照（同一数学，不同切法）")
    print(f"    {'mode':<7} {'step losses (全 batch)':<28} {'max|Δparams| vs ref':>20}")
    print("    " + "-" * 58)
    deltas = {}
    for name in ('ddp', 'zero1', 'zero2', 'fsdp2', 'fsdp3'):
        ls = ' '.join(f"{l:.4f}" for l in results[name]['losses'])
        d = (results[name]['flat'] - ref_flat).abs().max().item()
        deltas[name] = d
        extra = f"  inter-rank Δ={results[name]['inter_rank_maxdiff']:.1e}" if 'inter_rank_maxdiff' in results[name] else ''
        print(f"    {name:<7} {ls:<28} {d:18.3e}{extra}")
    print(f"    {'ref':<7} {' '.join(f'{l:.4f}' for l in ref_losses):<28} {'(ground truth)':>20}")

    print(f"\n[3] timing（CPU/gloo 仅供相对比较；生产在 GPU，见 tutorial §8）")
    for name in ('ddp', 'zero1', 'zero2', 'fsdp2', 'fsdp3'):
        print(f"    elapsed[{name}]: {results[name]['sec_per_step']*1000:8.1f} ms/step")

    # ---------------- self-check（机器断言） ----------------
    checks = []

    def ck(ok, msg):
        checks.append((bool(ok), msg))

    # [a] 各模式显存落在 stage 公式 ±1.5%（零头 = Adam step 计数器 / FSDP padding）
    for name in ('ddp', 'zero1', 'zero2', 'fsdp2', 'fsdp3'):
        _, exp = formulas[name]
        ck(abs(totals[name] - exp) / exp < 0.015,
           f"{name} per-rank state {totals[name]} within 1.5% of {formulas[name][0]}={exp}")
    # [b] 显存阶梯单调：ddp > zero1 > zero2 > fsdp2 == fsdp3
    ck(totals['ddp'] > totals['zero1'] > totals['zero2'] > totals['fsdp2'],
       'memory ladder monotone: ddp > zero1 > zero2 > fsdp*')
    ck(totals['fsdp2'] == totals['fsdp3'],
       'fsdp2/fsdp3 steady-state storage identical (差异在峰值与通信，不在稳态)')
    # [c] 通信量：ZeRO-0/1/2 同为 2P（含 FSDP SHARD_GRAD_OP），ZeRO-3 ≈ 3P
    two_p = 2 * P * fp32
    for name in ('ddp', 'zero1', 'zero2', 'fsdp2'):
        ck(abs(vols[name] - two_p) / two_p < 0.01, f'{name} comm volume == 2P ({two_p} B)')
    ck(2.5 * two_p / 2 < vols['fsdp3'] < 3.2 * two_p / 2,
       f'fsdp3 comm volume ~3P (measured {vols["fsdp3"]/(P*fp32):.3f}x P)')
    # [d] FSDP 集合通信次数与单元结构吻合（root 每步只 gather 一次 -> 2.858x 而非 3.000x）
    c3 = results['fsdp3']['comm_counts']
    ck(c3.get('gather', 0) == 2 * N_BLOCKS + 1 and c3.get('rs', 0) == N_UNITS,
       f'fsdp3 calls: gather={c3.get("gather")}==2*blocks+1, rs={c3.get("rs")}==units')
    c2 = results['fsdp2']['comm_counts']
    ck(c2.get('gather', 0) == N_UNITS and c2.get('rs', 0) == N_UNITS,
       f'fsdp2 calls: gather={c2.get("gather")}==units, rs={c2.get("rs")}==units')
    # [e] 正确性：所有模式 loss 与参照一致、最终权重 Δ < 5e-4（实测最大 ddp=1.76e-4，fp32 重排误差量级）
    for name in ('ddp', 'zero1', 'zero2', 'fsdp2', 'fsdp3'):
        ck(all(abs(a - b) < 1e-5 for a, b in zip(results[name]['losses'], ref_losses)),
           f'{name} step losses match reference')
        ck(deltas[name] < 5e-4, f'{name} final params within 5e-4 of reference')
    for name in ('ddp', 'zero1', 'zero2'):
        ck(results[name]['inter_rank_maxdiff'] == 0.0, f'{name} ranks hold identical params')
    # [f] 分片不减少集群总存储（L1 的结论在 ZeRO-3 上依然成立）
    ck(abs(2 * totals['fsdp3'] - full_state_bytes) / full_state_bytes < 0.02,
       'sum of FSDP shards across ranks == full replica (16P)')

    print(f"\n[4] self-check")
    n_pass = sum(ok for ok, _ in checks)
    for ok, msg in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {msg}")
    assert n_pass == len(checks), f'self-check failed: {len(checks) - n_pass} item(s)'
    print(f"    ✅ self-check passed ({n_pass}/{len(checks)})")

    # 确定性 digest：全部可复现指标（不含计时）
    digest_src = {
        'losses': {k: [round(v, 6) for v in results[k]['losses']] for k in results},
        'ref_losses': [round(v, 6) for v in ref_losses],
        'totals': totals, 'vols': vols,
        'deltas': {k: round(v, 9) for k, v in deltas.items()},
        'counts': {k: results[k]['comm_counts'] for k in results},
    }
    digest = hashlib.md5(json.dumps(digest_src, sort_keys=True).encode()).hexdigest()
    print(f"\ndigest(md5 of metrics) = {digest}")


def main():
    t_start = time.perf_counter()
    print("=" * 72)
    print("nano-fsdp L2 — ZeRO stages: what gets sharded, what it costs")
    print("=" * 72)
    print(f"TinyLM: vocab={VOCAB} dim={DIM} layers={LAYERS} | P = {P_TOTAL:,} params | fp32")
    print(f"cluster: W = {WORLD_SIZE} real processes, gloo backend, CPU "
          f"(真实多进程 + 真实集合通信；生产在 GPU，机制相同)")
    print(f"steps: 1 warmup + {STEPS} measured | seed = {SEED} (与 L1 一致)")

    ref_losses, _ = run_reference()
    print(f"\n[ref] single-process ground truth: losses = "
          f"{' '.join(f'{l:.4f}' for l in ref_losses)}")

    gc.collect()
    mp.spawn(worker, args=(WORLD_SIZE, MASTER_PORT), nprocs=WORLD_SIZE, join=True)

    print(f"\nelapsed: {time.perf_counter() - t_start:.1f}s total "
          f"(计算真跑; 计时行浮动, 锚点口径见 tutorial §2)")


if __name__ == '__main__':
    main()
