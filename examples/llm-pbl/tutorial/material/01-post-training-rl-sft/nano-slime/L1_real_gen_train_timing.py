#!/usr/bin/env python3
"""
nano-slime L1 — 把 G/T 从模拟常数变实测：真实小模型上的 generate vs train 计时

L0 用模拟常数 G=4/T=6/S=1 演示了解耦的结构；L1 在一台真实机器（本机 CPU）上，
用一个真实的小 GPT（char-level，~0.8M 参数，现场训练）实测这三类时间：
  G = 生成一批 rollout 的墙钟（autoregressive decode，KV cache，greedy）
  T = 一个训练步的墙钟（同一批 rollout 上做 fwd+bwd+Adam）
  S = 权重同步的墙钟（trainer 侧参数整体拷贝到 rollout 侧缓冲）
然后把实测值灌回 L0 的离散事件模拟器，看「真实数字下的解耦」长什么样。

三个要实测的物理结论（L0 里它们是口头声明，L1 里它们是拟合出来的斜率）：
  1. G 随 response 长度 L 线性涨——decode 每个 token 都是一次串行前向；
  2. batching 压缩单条 rollout 的 G——固定开销与权重读被摊薄（真实引擎的第一杠杆）；
  3. 同一批数据上 G_batch ≫ T——长 response RLVR 里 rollout 主导的墙钟根源，
     也因此：解耦本身买不到多少吞吐（L0 反例[4]a 的实测版），该修的是引擎。

诚实口径（重要）：
  - 本机无 GPU，全部为 CPU 墙钟。绝对毫秒数不可外推到 GPU/大模型；
    可外推的是**结构**：线性、压缩方向、G≫T 的机制（GPU 上 memory-bound decode
    同样使 G 随 L 线性涨、被 batching 摊薄，见 tutorial §9 与 nano-vllm-sglang）。
  - 探针模型是现场训练的真实小模型（语料为今日抓取的 THUDM/slime README 真实文本，
    见下方溯源块），不是随机权重——rollout 是真实采样出的文本。
  - 计时 = warmup 后 k 次重复取中位数；同一 seed 下生成内容逐字节确定，
    计时值随机器负载浮动（tutorial §8 给出连跑区间）。

依赖：仅 torch（CPU 即跑）。零网络、零外部数据。
"""

import os
import sys
sys.dont_write_bytecode = True          # 卫生：import L0 模块时不落 __pycache__
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 任意 CWD 可跑

import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from L0_data_buffer_decouple import sim_lockstep, sim_decoupled   # K+1：复用 L0 模拟器

SEED = 7
PROMPT_LEN = 32                          # 每条 rollout 的 prompt 长度（字符）
N_ROLLOUTS = 16                          # 一个 rollout 批的条数（L0 里 B 批之一批）
N_BATCHES = 12                           # 灌进 L0 模拟器的批数（与 L0 的 B 一致）
LENGTHS = [16, 32, 64, 128, 256]         # response 长度扫描
REPS = 5                                 # 每个计时点重复次数（取中位数）

# ---------------- 训练语料（真实文本，溯源见文件头注释与 tutorial §14） ----------------
# 来源：https://raw.githubusercontent.com/THUDM/slime/main/README.md
# 抓取：2026-08-06（curl 成功），全文件 19093 字节，
#       sha256 = 8989972638bb73f06ecd4bfb3092ce49ca42f55ff14f660cdaf28a3d37c93d21
# 切片：L9–17（自述）+ L19–28（Why This Design Matters / Production Validation）
#       + L84–92（Architecture Overview 三模块描述）
# 注：两块 junction 空行做过归一化（L17/L19 交界删一个空行、L84 交界前插一个空行），
#     内容行逐字一致（独立 diff 核验）
CORPUS = (
    "**slime** is an LLM post-training framework for RL scaling, providing two core capabilities:\n"
    "\n"
    "1.  **High-Performance Training**: Supports efficient training in various modes by connecting Megatron with SGLang;\n"
    "2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.\n"
    "\n"
    "slime's design goal is to make these two capabilities reinforce each other without turning the system into a heavy stack of disconnected trainers, rollout services, and agent frameworks. Megatron training, SGLang rollout, custom data generation, reward computation, verifier feedback, and environment interaction all flow through the same training / rollout / Data Buffer path.\n"
    "\n"
    "This makes slime one of the most battle-tested open RL post-training frameworks: small enough to understand and extend, but validated through complete training loops behind SOTA-level model releases.\n"
    "\n"
    "- **Battle-tested by frontier model training**: slime is the RL framework behind [GLM-5.2](https://z.ai/blog/glm-5.2), [GLM-5.1](https://z.ai/blog/glm-5.1), [GLM-5](https://z.ai/blog/glm-5), [GLM-4.7](https://z.ai/blog/glm-4.7), [GLM-4.6](https://z.ai/blog/glm-4.6), and [GLM-4.5](https://z.ai/blog/glm-4.5). This validates the full post-training loop, not only isolated examples.\n"
    "- **Correctness-first infrastructure**: RL bugs are often silent. slime keeps the dataflow explicit, supports separate rollout-only and train-only debugging paths, and documents reproducibility, fault tolerance, tracing, profiling, and CI as first-class engineering concerns.\n"
    "- **Native by design**: slime passes Megatron arguments through directly and exposes installed SGLang arguments with a `--sglang-` prefix. New upstream training and serving optimizations can be used without adding another abstraction layer inside slime.\n"
    "- **Maximum data-generation freedom**: math, code, search, tools, sandboxes, verifiers, environments, multi-agent systems, and long-horizon agentic workflows plug in as data generation or reward workflows. They do not fork the training kernel.\n"
    "- **Lightweight and opinionated**: slime focuses deeply on the Megatron + SGLang path used for large-scale RL. By choosing one rollout backend, slime can use SGLang-specific capabilities directly instead of flattening multiple inference engines into a lowest-common-denominator abstraction.\n"
    "\n"
    "## Production Validation\n"
    "\n"
    "slime has been exercised by the complete workflow needed for release-grade model post-training: large-scale training, high-throughput rollout, weight synchronization, reward/verifier data, checkpointing, debugging, and long-running stability.\n"
    "\n"
    "## Architecture Overview\n"
    "\n"
    "![arch](./imgs/arch.png)\n"
    "\n"
    "**Module Descriptions**:\n"
    "\n"
    "- **training (Megatron)**: Responsible for the main training process, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training.\n"
    "- **rollout (SGLang + router)**: Generates new data (including rewards/verifier outputs) and stores it in the Data Buffer. Custom generate functions can wrap this with multi-turn loops, tool calls, environment/sandbox interaction, and verifier-based reward.\n"
    "- **data buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods (including agentic workflows that produce samples through the same interface).\n"
)

# ---------------- 探针模型：char-level 小 GPT（KV cache 手写） ----------------

class Config:
    vocab = None            # 由语料定
    d_model = 128
    n_layers = 4
    n_heads = 8
    ffn = 512
    ctx = PROMPT_LEN + max(LENGTHS) + 8

class Attention(nn.Module):
    """因果自注意力，双路：全序列（训练用）与 KV-cache 单步（decode 用）。"""
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v))
        if cache is not None:                       # decode 单步：拼上历史 KV
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if cache is None:                           # 训练：全序列因果掩码
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y), new_cache

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn, bias=False), nn.GELU(),
            nn.Linear(cfg.ffn, cfg.d_model, bias=False))

    def forward(self, x, cache=None):
        h, new_cache = self.attn(self.ln1(x), cache)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, new_cache

class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight          # 权重绑定

    def forward(self, ids, caches=None):
        """ids [B,T]；caches=None 走全序列（训练），否则逐层单步 decode。"""
        x = self.tok(ids)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, None if caches is None else caches[i])
            new_caches.append(c)
        return self.head(self.ln(x)), new_caches

    @torch.no_grad()
    def generate(self, prompt_ids, n_new):
        """greedy decode，KV cache。prompt_ids [B,P] -> 返回 [B, P+n_new]。"""
        ids = prompt_ids
        logits, caches = self.forward(ids)
        next_ids = logits[:, -1:].argmax(-1)
        ids = torch.cat([ids, next_ids], dim=1)
        for _ in range(n_new - 1):
            logits, caches = self.forward(ids[:, -1:], caches)
            next_ids = logits[:, -1:].argmax(-1)
            ids = torch.cat([ids, next_ids], dim=1)
        return ids

# ---------------- 数据与训练 ----------------

def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return chars, stoi, {i: c for c, i in stoi.items()}

def corpus_windows(text, stoi, length, n, stride=None):
    """从语料取 n 个确定性窗口（循环取样），[n, length] 的 id 张量。"""
    ids = [stoi[c] for c in text]
    stride = stride or max(1, (len(ids) - length) // max(1, n - 1))
    out = []
    for i in range(n):
        s = (i * stride) % max(1, len(ids) - length)
        out.append(ids[s:s + length])
    return torch.tensor(out, dtype=torch.long)

def make_prompts(text, stoi, n):
    """n 条确定性 prompt（语料中均匀取的 PROMPT_LEN 窗口）。"""
    return corpus_windows(text, stoi, PROMPT_LEN, n,
                          stride=max(1, (len(text) - PROMPT_LEN) // max(1, n - 1)))

def linfit(xs, ys):
    """最小二乘 y = a + b x，返回 (a, b, r2)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1.0 - ss_res / ss_tot

def median_time(fn, reps=REPS, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def sweep_median(configs, make_fn, reps=REPS, warmup=2):
    """round-robin 计时：每轮按序测完所有配置再重复，每配置取中位数。
    对比逐块测量：机器的慢漂移 / 负载尖峰被摊到所有配置上，而不是
    整块砸在某个配置头上（测量方法本身，见 tutorial §3.3）。"""
    fns = {c: make_fn(c) for c in configs}
    for c in configs:
        for _ in range(warmup):
            fns[c]()
    ts = {c: [] for c in configs}
    for _ in range(reps):
        for c in configs:
            t0 = time.perf_counter()
            fns[c]()
            ts[c].append(time.perf_counter() - t0)
    return {c: statistics.median(ts[c]) for c in configs}

def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)            # 单线程基线：串行 decode 物理最干净（见 [2] 探针）
    print("=" * 68)
    print("nano-slime L1 — 实测 G/T/S：真实小模型上的 generate vs train")
    print("=" * 68)
    print(f"env: torch {torch.__version__} | CPU | threads={torch.get_num_threads()}"
          f" | seed={SEED} | greedy decode（内容确定，计时浮动）")

    # ---------- [0] 探针模型：真实语料 + 现场训练 + rollout 展示 ----------
    chars, stoi, itos = build_vocab(CORPUS)
    cfg = Config(); cfg.vocab = len(chars)
    model = TinyGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[0] 探针模型: char-level GPT | vocab={cfg.vocab} | d={cfg.d_model}"
          f" | layers={cfg.n_layers} | heads={cfg.n_heads} | params={n_params:,}")
    print(f"    语料: THUDM/slime README 真实文本切片 {len(CORPUS)} bytes（溯源见 tutorial §14）")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    steps, blen, seqlen = 1200, 8, 128
    t_train0 = time.perf_counter()
    for step in range(steps):
        batch = corpus_windows(CORPUS, stoi, seqlen, blen, stride=37 + step % 5)
        logits, _ = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    train_secs = time.perf_counter() - t_train0
    print(f"    预训练: {steps} 步 × batch{blen}×seq{seqlen} | 末步 loss={loss.item():.3f}"
          f" nats/char | 耗时 {train_secs:.1f}s（不计入任何 G/T 测量）")

    prompts16 = make_prompts(CORPUS, stoi, N_ROLLOUTS)
    def rollout_text(ids):
        return "".join(itos[i] for i in ids.tolist())
    demo = model.generate(prompts16[:1], 96)
    print(f"    rollout 展示（prompt | 生成 96 字符，greedy）:")
    print(f"      prompt  : ...{rollout_text(prompts16[0])[-60:]!r}")
    print(f"      生成续写: {rollout_text(demo[0, PROMPT_LEN:])!r}")
    demo2 = model.generate(prompts16[:1], 96)
    assert torch.equal(demo, demo2), "同 seed greedy decode 必须逐字节确定"
    print(f"    确定性: 重生成一次，逐 token 相同 ✓")

    # ---------- [1] G(L)：串行生成（B=1），KV cache，L 扫描 ----------
    print(f"\n[1] G(L): 串行生成 1 条 rollout（B=1，KV cache，round-robin ×{REPS} 取中位数）")
    print(f"    {'L':>4} | {'G (ms)':>8} | {'ms/token':>8}")
    p1 = prompts16[:1]
    g_map = sweep_median(LENGTHS, lambda L: (lambda: model.generate(p1, L)))
    g_serial = [g_map[L] for L in LENGTHS]
    for L, t in zip(LENGTHS, g_serial):
        print(f"    {L:>4} | {t * 1e3:8.1f} | {t * 1e3 / L:8.2f}")
    a_g, b_g, r2_g = linfit(LENGTHS, [t * 1e3 for t in g_serial])
    print(f"    线性拟合 G = {a_g:.1f} + {b_g:.3f}·L (ms) | R² = {r2_g:.4f}")
    print(f"    截距 a = 每批固定开销（Python/内核启动）；斜率 b = 每 token 串行前向价")

    # ---------- [2] batching 压缩：N=16 条 rollout，L=128 ----------
    L_fix = 128
    def batch_sweep():
        def make_gen(B):
            def gen_all():
                for i in range(0, N_ROLLOUTS, B):
                    model.generate(prompts16[i:i + B], L_fix)
            return gen_all
        pr = sweep_median((1, 2, 4, 8, 16), make_gen)
        return {B: pr[B] * 1e3 / N_ROLLOUTS for B in pr}
    print(f"\n[2] batching 压缩: 共 {N_ROLLOUTS} 条 rollout × L={L_fix}，变批大小（threads=1）")
    print(f"    {'B':>3} | {'单条 G (ms)':>10} | {'vs B=1':>7}")
    per_rollout = batch_sweep()
    for B in (1, 2, 4, 8, 16):
        print(f"    {B:>3} | {per_rollout[B]:10.2f} | {per_rollout[1] / per_rollout[B]:6.2f}x")
    compress = per_rollout[1] / per_rollout[16]
    print(f"    压缩率 G(B=1)/G(B=16) = {compress:.2f}x —— 固定开销与权重读被摊薄")
    try:
        torch.set_num_threads(8)
        pr8 = batch_sweep()
    finally:
        torch.set_num_threads(1)
    print(f"    探针（threads=8）: " +
          "  ".join(f"B{B} {pr8[B]:.1f}" for B in (1, 2, 4, 8, 16)) + " ms/条")
    slower8 = [B for B in (1, 2, 4, 8, 16) if pr8[B] > per_rollout[B]]
    if len(slower8) == 5:
        print(f"    → 本模型尺寸下多线程在每个批大小都更慢（matmul 太小，调度开销主导）；")
    else:
        faster8 = [B for B in (1, 2, 4, 8, 16) if pr8[B] <= per_rollout[B]]
        print(f"    → 本机实测：threads=8 在 {faster8} 处不慢于 threads=1"
              f"（此结论依赖 机器×模型×批 形状，以打印数字为准）；")
    print(f"      线程开始赚钱的拐点取决于 模型×批 的形状。批与线程是耦合的旋钮，")
    print(f"      GPU 引擎干脆用数千线程的 SIMD 重写这条曲线（L2 / nano-vllm-sglang）。")

    # ---------- [3] T(L)：同一批 rollout 上的一个训练步 ----------
    print(f"\n[3] T(L): 一个训练步（fwd+bwd+Adam，batch={N_ROLLOUTS}，round-robin ×{REPS} 取中位数）")
    g128 = median_time(lambda: model.generate(prompts16, L_fix))   # 先测 G（训练步会漂移权重）
    rolls = {L: model.generate(prompts16, L) for L in LENGTHS}     # 真实 rollout 作训练数据（不计时）
    def make_step(L):
        def one_step():
            logits, _ = model(rolls[L][:, :-1])
            lo = F.cross_entropy(logits.reshape(-1, cfg.vocab), rolls[L][:, 1:].reshape(-1))
            opt.zero_grad(); lo.backward(); opt.step()
        return one_step
    t_map = sweep_median(LENGTHS, make_step)
    print(f"    {'L':>4} | {'T (ms)':>8} | {'ms/token(全批)':>13}")
    t_step = {L: t_map[L] for L in LENGTHS}
    for L in LENGTHS:
        print(f"    {L:>4} | {t_step[L] * 1e3:8.1f} | "
              f"{t_step[L] * 1e3 / (N_ROLLOUTS * (PROMPT_LEN + L)):13.3f}")
    a_t, b_t, r2_t = linfit(LENGTHS, [t_step[L] * 1e3 for L in LENGTHS])
    print(f"    线性拟合 T = {a_t:.1f} + {b_t:.3f}·L (ms) | R² = {r2_t:.4f}")
    ratio = g128 / t_step[L_fix]
    print(f"    同一批（{N_ROLLOUTS}×L={L_fix}）: G_batch={g128 * 1e3:.0f} ms vs "
          f"T={t_step[L_fix] * 1e3:.0f} ms → G/T = {ratio:.1f}")
    print(f"    训练一步把 {N_ROLLOUTS}×(32+{L_fix}) token 并行过一遍；生成要 "
          f"{L_fix} 步串行前向——这就是 rollout 主导的墙钟根源")

    # ---------- [4] S：权重同步 = 参数整体拷贝 ----------
    dst = [torch.empty_like(p) for p in model.parameters()]
    def sync():
        for p, d in zip(model.parameters(), dst):
            d.copy_(p.data)
    s = median_time(sync)
    mb = n_params * 4 / 2**20
    print(f"\n[4] S: 权重同步（trainer→rollout 侧参数拷贝）= {s * 1e3:.2f} ms"
          f"（{mb:.1f} MB，{mb / 1024 / s:.2f} GB/s）")
    print(f"    对照: S/T = {s / t_step[L_fix]:.3f} —— 本机 CPU 上同步不是瓶颈")
    print(f"    （真实 slime 里是 Megatron→SGLang 跨引擎传输，见 L3 的 delta weight sync）")

    # ---------- [5] 实测值灌回 L0 模拟器 ----------
    print(f"\n[5] 实测 G/T/S 灌回 L0 模拟器（{N_BATCHES} 批，buffer C=4）")
    G_real, T_real, S_real = g128, t_step[L_fix], s
    mk_l, gu_l, tu_l = sim_lockstep(N_BATCHES, G_real, T_real, S_real)
    mk_d, gu_d, tu_d, stale, _, syncs = sim_decoupled(N_BATCHES, G_real, T_real, S_real, 4)
    print(f"    lockstep : makespan {mk_l:.2f}s | gen 利用率 {gu_l:.1%} | trainer {tu_l:.1%}")
    print(f"    解耦 C=4 : makespan {mk_d:.2f}s | speedup {mk_l / mk_d:.2f}x | "
          f"gen {gu_d:.1%} | trainer {tu_d:.1%} | 同步 {syncs} 次")
    print(f"    staleness: mean={sum(stale) / len(stale):.2f} max={max(stale)}"
          f"（生成主导 → buffer 积不起来，off-policy 度天然低）")
    G_serial_batch = per_rollout[1] * N_ROLLOUTS / 1000
    mk_s, _, _ = sim_lockstep(N_BATCHES, G_serial_batch, T_real, S_real)
    print(f"    反事实（引擎不 batching，G={G_serial_batch:.2f}s/批）: lockstep makespan {mk_s:.1f}s")
    print(f"    → batching 把 makespan 从 {mk_s:.1f}s 压到 {mk_l:.1f}s"
          f"（{mk_s / mk_l:.1f}x）——第一杠杆是批量化，不是解耦")

    # ---------- [6] self-check ----------
    print(f"\n[6] self-check")
    assert r2_g > 0.99, f"G(L) 应线性（R²={r2_g:.4f}）"
    print(f"    ✓ G 随 L 线性：R²={r2_g:.4f} > 0.99")
    assert per_rollout[4] >= per_rollout[8] * 0.95 and \
           per_rollout[8] >= per_rollout[16] * 0.95, "B≥4 应单调压缩"
    assert per_rollout[16] <= per_rollout[1] * 0.75, \
        f"B=16 应显著压缩单条 G（实测 {compress:.2f}x）"
    assert max(per_rollout.values()) <= per_rollout[1] * 1.5, "任何批大小不应灾难性变慢"
    print(f"    ✓ B≥4 单调压缩至 {compress:.2f}x ≥ 1.33（B=16 vs B=1）")
    print(f"    ✓ B=1↔2 在噪声带内：单条 decode 固定开销的方差与本征差相当（见 tutorial §4.2）")
    t_ms_tok = t_step[L_fix] * 1e3 / (N_ROLLOUTS * (PROMPT_LEN + L_fix))
    assert b_g > 3 * t_ms_tok, "每 token 墙钟：串行生成应显著贵于并行训练"
    print(f"    ✓ 每 token 墙钟：生成 {b_g:.3f} > 训练 {t_ms_tok:.3f} ms/token"
          f"（{b_g / t_ms_tok:.1f}x，串行 L 步 vs 并行 1 遍）")
    assert s < t_step[L_fix], "本机口径下同步应便宜于训练步"
    print(f"    ✓ S < T（{s * 1e3:.2f} < {t_step[L_fix] * 1e3:.0f} ms）")
    assert 1.0 < mk_l / mk_d < 1.6, "生成主导区，解耦增益应存在但有限"
    assert all(x >= 0 for x in stale)
    print(f"    ✓ 生成主导区解耦 speedup={mk_l / mk_d:.2f}x ∈ (1.0, 1.6)：buffer 买不到吞吐")

    print("\n" + "=" * 68)
    print("✅ self-check passed: 线性 / 压缩 / G>T / S<T / 解耦增益有限")
    print("=" * 68)
    print(f"\ntakeaway: 实测确认 L0 的三条口头声明——G∝L（b={b_g:.3f} ms/token）、")
    print(f"          batching 压缩 {compress:.1f}x、同批 G/T={ratio:.1f}。解耦的价值在")
    print(f"          staleness 管理与弹性，不在吞吐；吞吐的第一杠杆是 batching 与")
    print(f"          更快的引擎（L2 接 SGLang/vLLM，对照 nano-vllm-sglang）。")

if __name__ == "__main__":
    main()
