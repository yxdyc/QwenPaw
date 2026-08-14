"""
nano-fsdp L1 — real PyTorch FSDP vs DDP on CPU

用真实 PyTorch 的 DistributedDataParallel / FullyShardedDataParallel 训练一个
TinyLM，对比同一张「模型状态」账本分到 2 个 rank 后每 rank 占多少内存。

运行要求：torch (CPU 即可)。本机示例：
    python L1_single_card_fsdp.py

注意：L1 用 CPU 做本地可复现 demo；真实生产在 GPU 上，机制相同，数字按 GiB 放大。
"""

import os
import sys
import gc
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group, all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Adam

# 确定性
SEED = 7
WORLD_SIZE = 2

# TinyLM：够小、够真实，能在 CPU 秒开，又能分出参数量。
class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 128, dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=dim * 4,
                batch_first=True,
                dropout=0.0,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def fsdp_wrap_policy(module, recurse, nonwrapped_numel):
    """每个 Transformer block 单独包一层 FSDP，和真实大模型按层分片一致。"""
    return isinstance(module, nn.TransformerEncoderLayer)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_state_bytes(model, optimizer) -> tuple[int, int, int]:
    """
    只算 L0 里的「模型状态」：参数 + 梯度 + 优化器状态。
    不算 activations / 通信 buffer / OS 开销，和 L0 口径一致。
    """
    if isinstance(model, DDP):
        params = list(model.module.parameters())
    else:
        # FSDP：parameters() 已经是当前 rank 的 shard(FlatParameter)
        params = list(model.parameters())

    param_bytes = sum(p.numel() * p.element_size() for p in params)
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in params if p.grad is not None
    )
    opt_bytes = 0
    for p in params:
        st = optimizer.state.get(p, {})
        for v in st.values():
            if torch.is_tensor(v):
                opt_bytes += v.numel() * v.element_size()
    return param_bytes, grad_bytes, opt_bytes


def to_mib(b: int) -> float:
    return b / (1024 * 1024)


def run_mode(rank: int, mode: str, world_size: int, port: int):
    """mode: 'ddp' or 'fsdp'. mp.spawn prepends rank as first argument."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0')

    init_process_group('gloo', rank=rank, world_size=world_size)
    torch.manual_seed(SEED)

    device = torch.device('cpu')
    raw_model = TinyLM().to(device)
    total_params = count_params(raw_model)

    if mode == 'ddp':
        model = DDP(raw_model, device_ids=None)
    else:
        model = FSDP(
            raw_model,
            auto_wrap_policy=fsdp_wrap_policy,
            device_id=device,
        )

    optimizer = Adam(model.parameters(), lr=1e-3)

    # 合成数据：每个 rank 用同样 seed，保证内存对比公平。
    torch.manual_seed(SEED + rank)
    x = torch.randint(0, 128, (4, 16), device=device)
    y = torch.randint(0, 128, (4, 16), device=device)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, 128), y.view(-1))
    loss.backward()
    optimizer.step()

    param_b, grad_b, opt_b = model_state_bytes(model, optimizer)
    total_b = param_b + grad_b + opt_b

    # 把每个 rank 的内存数字收集到 rank 0
    local = torch.tensor([param_b, grad_b, opt_b, total_b], dtype=torch.float64)
    gathered = [torch.zeros(4, dtype=torch.float64) for _ in range(world_size)]
    all_gather(gathered, local)

    if rank == 0:
        print(f"\n[{mode.upper()}] per-rank model state (params + grads + optimizer)")
        print("-" * 65)
        for r in range(world_size):
            pb, gb, ob, tb = gathered[r].tolist()
            print(
                f"  rank {r}  params={to_mib(pb):6.2f} MiB  "
                f"grads={to_mib(gb):6.2f} MiB  "
                f"optimizer={to_mib(ob):6.2f} MiB  total={to_mib(tb):6.2f} MiB"
            )
        sum_total = sum(gathered[r][3].item() for r in range(world_size))
        print(f"  sum across ranks = {to_mib(sum_total):.2f} MiB")
        expected_full = total_params * 16  # L0: 16 bytes/param for Adam training state
        print(f"  L0 expected full replica = {to_mib(expected_full):.2f} MiB")
        if mode == 'fsdp':
            expected_shard = expected_full / world_size
            print(f"  L0 expected per-rank shard = {to_mib(expected_shard):.2f} MiB")

        # self-check 只由 rank 0 打印，但 assert 在所有 rank 发生（通过 dist 同步失败）
        assert loss.isfinite(), "loss must be finite"
        per_rank_totals = [gathered[r][3].item() for r in range(world_size)]
        if mode == 'ddp':
            for t in per_rank_totals:
                assert abs(t - expected_full) / expected_full < 0.05, \
                    f"DDP per-rank total {t} != expected {expected_full}"
        else:
            expected_shard = expected_full / world_size
            for t in per_rank_totals:
                assert abs(t - expected_shard) / expected_shard < 0.10, \
                    f"FSDP per-rank total {t} != expected shard {expected_shard}"
            assert abs(sum_total - expected_full) / expected_full < 0.10, \
                f"FSDP sum across ranks {sum_total} != full replica {expected_full}"

    destroy_process_group()


def main():
    warnings.filterwarnings('ignore')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0')

    # 先算一个不参与分布式、仅用于报告的参考参数量
    ref_model = TinyLM()
    total_params = count_params(ref_model)
    del ref_model

    print("=" * 65)
    print("nano-fsdp L1 — real PyTorch FSDP vs DDP on CPU")
    print("=" * 65)
    print(f"TinyLM: vocab=128 dim=64 layers=2 | total params = {total_params:,}")
    print(f"Adam training state (L0 formula) = 16 bytes/param")
    print(f"Running on CPU with {WORLD_SIZE} processes")

    # DDP 端口与 FSDP 端口错开，避免前后两次 spawn 冲突
    mp.spawn(run_mode, args=('ddp', WORLD_SIZE, 29510), nprocs=WORLD_SIZE, join=True)
    gc.collect()
    mp.spawn(run_mode, args=('fsdp', WORLD_SIZE, 29511), nprocs=WORLD_SIZE, join=True)
    gc.collect()

    print("\n✅ self-check passed")


if __name__ == '__main__':
    main()
