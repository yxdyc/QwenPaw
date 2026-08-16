"""nano-vllm-sglang · L1 真实小模型实测
=========================================

目标：把 L0 代价模型里的三个机制换上**真实模型、真实张量、真实墙钟**——
    ① KV cache：真实 decoder 上「无 cache 每步重算前缀」vs「预分配 cache 只算新 token」，
       数值逐 token 一致 + 真实计时 + 真实 KV 字节账；
    ② batching：真实 batched forward 的 tokens/s 随 B 的曲线，拟合 L0 的仿射代价模型
       t(B) = a + b·B，量化「固定项」在 CPU 上的真实占比；
    ③ static vs continuous batching：同一组真实请求（L0 的 lengths=[6,2,9,3,14,5,1,8]）
       在两种真实调度器下跑——**迭代账本与 L0 模拟器逐位一致，生成 token 逐位一致，
       只有墙钟与浪费计算量不同**（跨级别一致性契约）；
    ④ 代价模型对照：用 [2] 拟合的 (a, b) 去**预测** [3] 的墙钟，
       检验 L0 模型形式（仿射）在真实硬件上的预测力与失效位置。

声明（课程可运行性契约）：
    - 模型为**随机初始化的小 GPT**（GQA 结构，固定 seed）——真实权重 + 真实引擎
      （vLLM/SGLang on GPU）留 L2 / 真实 GPU/多机环境 `[TODO: verify on real system]`。
      机制本身是真的：真实 forward、真实 KV 张量、真实 greedy 解码、真实墙钟。
    - 本机为 Apple Silicon CPU 执行（torch，fp32，threads=4）。
      **CPU decode 是 compute-bound，不是 GPU 的 memory-bandwidth-bound**——
      曲线「形状」可对照，物理成因不同，教程 §5 专门拆解；GPU 数字全部留真机验证。
    - 依赖仅 torch；greedy + 固定 seed，除计时外输出确定。

运行：
    python L1_real_kv_batching.py
"""

import math
import time

import torch
import torch.nn as nn

# ===== 配置（小 GPT，GQA 结构，随机初始化）=====
VOCAB, D, LAYERS = 512, 256, 4
HEADS, KV_HEADS, HEAD_DIM = 8, 2, 32        # GQA：8 个 Q 头共享 2 组 KV（4:1）
FFN, MAX_T, SLOTS = 1024, 512, 16
SEED, THREADS = 42, 4
PROMPT_LEN = 8                               # [3] 调度实验的 prompt 长度
REQ_LENGTHS = [6, 2, 9, 3, 14, 5, 1, 8]      # 与 L0 实验 [3] 完全相同的请求长度
BATCH = 4                                    # 与 L0 相同的并发上限
T_GEN = 48                                   # [2] 每档 B 的 decode 步数

torch.manual_seed(SEED)
torch.set_num_threads(THREADS)


# ===== 模型：最小 GQA decoder（RMSNorm + 绝对位置编码 + 2 层 MLP）=====

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class MiniGPT(nn.Module):
    """Qwen2 家族的最小同构体：pre-norm + GQA + 因果 attention。
    位置编码用可学习绝对嵌入（Qwen2 实际用 RoPE，rope_theta=1e6——PE 选型
    不影响 KV cache 机制：cache 存的都是「已经算好的历史 K/V」）。"""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(MAX_T, D)
        self.norm1 = nn.ModuleList([RMSNorm(D) for _ in range(LAYERS)])
        self.norm2 = nn.ModuleList([RMSNorm(D) for _ in range(LAYERS)])
        self.wq = nn.ModuleList([nn.Linear(D, HEADS * HEAD_DIM, bias=False) for _ in range(LAYERS)])
        self.wk = nn.ModuleList([nn.Linear(D, KV_HEADS * HEAD_DIM, bias=False) for _ in range(LAYERS)])
        self.wv = nn.ModuleList([nn.Linear(D, KV_HEADS * HEAD_DIM, bias=False) for _ in range(LAYERS)])
        self.wo = nn.ModuleList([nn.Linear(HEADS * HEAD_DIM, D, bias=False) for _ in range(LAYERS)])
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(D, FFN, bias=False), nn.GELU(),
            nn.Linear(FFN, D, bias=False)) for _ in range(LAYERS)])
        self.final = RMSNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        # KV 池：预分配，形状对齐真实引擎的「启动即划走整块显存」
        self.pool_k = [torch.zeros(SLOTS, KV_HEADS, MAX_T, HEAD_DIM) for _ in range(LAYERS)]
        self.pool_v = [torch.zeros(SLOTS, KV_HEADS, MAX_T, HEAD_DIM) for _ in range(LAYERS)]

    def forward(self, x, pos_vec, slots=None, use_cache=False):
        """x:[B,T]；pos_vec:[B] 每行已缓存前缀长度；绝对位置 = pos_vec + i。
        use_cache=True 时从 pool 读历史 K/V、把新 K/V 写回 slots。"""
        B, T = x.shape
        pos_vec = torch.as_tensor(pos_vec)
        max_pos = int(pos_vec.max())
        Klen = max_pos + T
        pos_id = pos_vec[:, None] + torch.arange(T)[None, :]
        h = self.tok(x) + self.pos(pos_id)
        # 允许矩阵：行 b 的查询 i 只能看 j <= pos_b + i（因果 + 各自前缀长度）
        j_idx = torch.arange(Klen)[None, None, :]
        allow = j_idx <= (pos_vec[:, None, None] + torch.arange(T)[None, :, None])
        for l in range(LAYERS):
            g = self.norm1[l](h)
            q = self.wq[l](g).view(B, T, HEADS, HEAD_DIM).transpose(1, 2)
            k_new = self.wk[l](g).view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)
            v_new = self.wv[l](g).view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)
            if use_cache:
                k_full = torch.zeros(B, KV_HEADS, Klen, HEAD_DIM)
                v_full = torch.zeros(B, KV_HEADS, Klen, HEAD_DIM)
                for b in range(B):                      # 逐行搬运：真实引擎由
                    pb = int(pos_vec[b])                # paged-attention kernel 完成
                    k_full[b, :, :pb] = self.pool_k[l][slots[b], :, :pb]
                    v_full[b, :, :pb] = self.pool_v[l][slots[b], :, :pb]
                    k_full[b, :, pb:pb + T] = k_new[b]
                    v_full[b, :, pb:pb + T] = v_new[b]
                    self.pool_k[l][slots[b], :, pb:pb + T] = k_new[b]
                    self.pool_v[l][slots[b], :, pb:pb + T] = v_new[b]
            else:
                k_full, v_full = k_new, v_new
            k_rep = k_full.repeat_interleave(HEADS // KV_HEADS, dim=1)   # GQA 展开
            v_rep = v_full.repeat_interleave(HEADS // KV_HEADS, dim=1)
            s = q @ k_rep.transpose(-1, -2) / math.sqrt(HEAD_DIM)
            s = s.masked_fill(~allow[:, None, :, :], float("-inf"))
            h = h + self.wo[l]((torch.softmax(s, -1) @ v_rep)
                               .transpose(1, 2).reshape(B, T, HEADS * HEAD_DIM))
            h = h + self.mlp[l](self.norm2[l](h))
        return self.head(self.final(h))


model = MiniGPT()
model.eval()
N_PARAMS = sum(p.numel() for p in model.parameters())


# ===== 解码原语 =====

@torch.no_grad()
def decode_no_cache(prompt, n_new):
    """每步把「prompt+已生成」整段重跑一遍——没有任何复用的反例基线。"""
    gen, per_step = list(prompt), []
    for _ in range(n_new):
        t0 = time.perf_counter()
        logits = model(torch.tensor([gen]), torch.tensor([0]))
        gen.append(int(logits[0, -1].argmax()))
        per_step.append(time.perf_counter() - t0)
    return gen[len(prompt):], per_step


@torch.no_grad()
def prefill(tokens, slot):
    """prompt 一次并行进模型（compute-bound 阶段），返回 token #1，KV 写入池。"""
    x = torch.tensor([tokens])
    logits = model(x, torch.tensor([0]), slots=[slot], use_cache=True)
    return int(logits[0, -1].argmax())


@torch.no_grad()
def decode_cached(prompt, n_new, slot, timings=False):
    """prefill 后每步只喂 1 个 token：历史 K/V 全部来自池子。
    返回 n_new 个新 token（gen 本身不含 prompt，勿再切片）。"""
    gen, per_step = [prefill(list(prompt), slot)], []
    pos = len(prompt)
    for _ in range(n_new - 1):
        t0 = time.perf_counter()
        logits = model(torch.tensor([[gen[-1]]]), torch.tensor([pos]),
                       slots=[slot], use_cache=True)
        gen.append(int(logits[0, -1].argmax()))
        pos += 1
        if timings:
            per_step.append(time.perf_counter() - t0)
    return gen, per_step


@torch.no_grad()
def batched_decode_step(last_tokens, slots, pos_vec):
    """一个 decode 迭代：batch 里每行各前进一步（pos 可参差——continuous
    batching 的常态）。返回各行新 token。"""
    x = torch.tensor(last_tokens).view(-1, 1)      # [B, 1]：每行一个 token
    logits = model(x, torch.tensor(pos_vec), slots=list(slots), use_cache=True)
    return [int(v) for v in logits[:, -1].argmax(-1).tolist()]


def make_prompts(n, seed=7):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (PROMPT_LEN,), generator=g).tolist()
            for _ in range(n)]


# ===== 调度器：static / continuous =====
# 迭代语义与 L0 模拟器严格同构：每个「迭代」每条活跃序列恰好产出 1 个 token；
# 序列被接纳的那一迭代由 prefill 产出 token #1。

def serve_static(prompts, lengths, batch=BATCH):
    """凑齐一批 → 整批跑到最长序列结束；完成的序列**留在 batch 里继续空转**
    （真实 static 引擎保持张量形状不变，finished 的输出丢弃但计算照付）。"""
    tokens_out = [[] for _ in lengths]
    iters, wasted = 0, 0
    lat_done = [0] * len(lengths)
    t0 = time.perf_counter()
    for g0 in range(0, len(lengths), batch):
        grp = lengths[g0:g0 + batch]
        slots = list(range(len(grp)))
        iters += 1                                   # 批内第 1 迭代 = prefill
        lasts = [prefill(prompts[g0 + k], s) for k, s in enumerate(slots)]
        gen = [1] * len(grp)
        pos = [PROMPT_LEN] * len(grp)
        for k, L in enumerate(grp):
            tokens_out[g0 + k].append(lasts[k])
            if gen[k] == L:
                lat_done[g0 + k] = iters
        while any(g < L for g, L in zip(gen, grp)):
            iters += 1
            new = batched_decode_step(lasts, slots, pos)
            for k, L in enumerate(grp):
                if gen[k] < L:
                    gen[k] += 1
                    tokens_out[g0 + k].append(new[k])
                    lasts[k] = new[k]
                    pos[k] += 1
                    if gen[k] == L:
                        lat_done[g0 + k] = iters
                else:
                    wasted += 1                      # 空转 token-step（真实计算！）
    wall = time.perf_counter() - t0
    return tokens_out, iters, wasted, lat_done, wall


def serve_continuous(prompts, lengths, batch=BATCH):
    """iteration 级调度：每迭代开始，完成的让位、等待的进场（进场即 prefill，
    本迭代产出它的 token #1），然后其余活跃序列前进一步。"""
    tokens_out = [[] for _ in lengths]
    lat_done = [0] * len(lengths)
    pending = list(range(len(lengths)))
    active = {}                                      # i -> (gen 数, 最后 token)
    iters, wasted = 0, 0
    t0 = time.perf_counter()
    while True:
        for i in [i for i, (g, _) in active.items() if g >= lengths[i]]:
            del active[i]                            # EOS 即让位
        if not pending and not active:
            break
        iters += 1
        admitted = set()
        while pending and len(active) < batch:
            i = pending.pop(0)
            first = prefill(prompts[i], i)           # 进场即 prefill
            tokens_out[i].append(first)
            active[i] = (1, first)
            admitted.add(i)
            if lengths[i] == 1:
                lat_done[i] = iters
        cont = [i for i, (g, _) in active.items()
                if g < lengths[i] and i not in admitted]
        if cont:
            lasts = [active[i][1] for i in cont]
            # gen 个已生成 token 占据位置 P..P+gen-1：最后 token 的位置是 P+gen-1
            pos = [PROMPT_LEN + active[i][0] - 1 for i in cont]
            new = batched_decode_step(lasts, cont, pos)
            for i, nv in zip(cont, new):
                g, _ = active[i]
                active[i] = (g + 1, nv)
                tokens_out[i].append(nv)
                if g + 1 == lengths[i]:
                    lat_done[i] = iters
    wall = time.perf_counter() - t0
    return tokens_out, iters, wasted, lat_done, wall


def ledger_continuous(reqs, batch=BATCH):
    """纯账本版（与 L0 sim_continuous 同构）：完成迭代号 + 每迭代 decode 活跃数。"""
    pending, active, done = list(range(len(reqs))), {}, {}
    hist, t = [], 0
    while True:
        for i in [i for i in active if active[i] >= reqs[i]]:
            done[i] = t                     # t = 已完成迭代数 = 它的完成迭代号
            del active[i]
        if not pending and not active:
            break
        t += 1
        admitted = set()
        while pending and len(active) < batch:
            i = pending.pop(0)
            active[i] = 1
            admitted.add(i)
        cont = [i for i in active if active[i] < reqs[i] and i not in admitted]
        hist.append(len(cont))
        for i in cont:
            active[i] += 1
    for i in range(len(reqs)):
        assert i in done
    return t, [done[i] for i in range(len(reqs))], hist


def ledger_static(reqs, batch=BATCH):
    t, done, wasted = 0, [0] * len(reqs), 0
    for g0 in range(0, len(reqs), batch):
        grp = reqs[g0:g0 + batch]
        m = max(grp)
        for k, L in enumerate(grp):
            done[g0 + k] = t + L
            wasted += m - L
        t += m
    return t, done, wasted


def fit_affine(curve):
    """最小二乘 t(B) = a + b·B（纯 Python，无额外依赖）。"""
    xs, ys = list(curve.keys()), [curve[x] for x in curve]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = sy / n - b * sx / n
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - sy / n) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot


def main():
    print("=" * 68)
    print("nano-vllm-sglang L1 — 真实小模型实测 L0 的三个机制")
    print("=" * 68)
    print(f"torch {torch.__version__} | CPU fp32 | threads={THREADS} | seed={SEED}")
    print("声明: 随机初始化小 GPT（真实 forward / 真实 KV 张量 / 真实墙钟）；")
    print("      真实权重 + 真实引擎 (vLLM/SGLang on GPU) 见 L2 / [TODO: verify on real system]")

    # ---- [0] 模型与 KV 账本 ----
    kv_per_tok = 2 * LAYERS * KV_HEADS * HEAD_DIM * 4          # fp32
    pool_bytes = sum(t.nbytes for t in model.pool_k + model.pool_v)
    print(f"\n[0] 模型: {N_PARAMS:,} params | {LAYERS} 层 | GQA {HEADS}Q/{KV_HEADS}KV"
          f" | head_dim={HEAD_DIM}")
    print(f"    KV 池: {SLOTS} slots × {MAX_T} tokens 预分配"
          f" = {pool_bytes / 2**20:.1f} MiB（启动即划走，对齐真实引擎做法）")
    print(f"    每 token KV = 2(K,V)×{LAYERS}层×{KV_HEADS}KV头×{HEAD_DIM}dim×4B"
          f" = {kv_per_tok} B；池字节数 == 公式 ✅")
    assert pool_bytes == SLOTS * MAX_T * kv_per_tok
    qwen = 2 * 24 * 2 * 64 * 2          # Qwen2.5-0.5B（config 2026-08-06 核验）bf16
    llama = 2 * 32 * 32 * 128 * 2       # L0 的 Llama-2-7B 账（fp16）
    print(f"    同式外推 Qwen2.5-0.5B(bf16): {qwen} B/token = {qwen / 1024:.1f} KiB"
          f" → 4096 token 占 {qwen * 4096 / 2**20:.0f} MiB")
    print(f"    对照 L0 的 Llama-2-7B: {llama / 2**20:.1f} MiB/token → GQA 把每 token KV"
          f" 缩小 {llama / qwen:.1f}×（2 KV 头 vs 32 + 更小 head_dim）")

    # ---- [1] KV cache：数值一致 + 真实计时 + 字节账 ----
    T = 256
    prompt = torch.randint(0, VOCAB, (1,)).tolist()
    gen_nc, steps_nc = decode_no_cache(prompt, T)
    gen_c, steps_c = decode_cached(prompt, T, slot=0, timings=True)
    wall_nc, wall_c = sum(steps_nc), sum(steps_c)
    mark = "✅" if gen_nc == gen_c else "❌"
    print(f"\n[1] KV cache（prompt=1 token，生成 T={T}，greedy）")
    print(f"    无 cache（每步重算整段前缀） vs 预分配 cache（每步只算 1 个新 token）")
    print(f"    生成结果逐 token 一致: {gen_nc == gen_c} {mark}（{T} 个 token 全同）")
    print(f"    墙钟: 无 cache {wall_nc * 1e3:7.0f} ms / 有 cache {wall_c * 1e3:6.0f} ms"
          f" → 实测加速 {wall_nc / wall_c:.1f}×")
    print(f"    算术上界 (T+1)/2 = {(T + 1) / 2:.1f}×（token-step 数 Σt = T(T+1)/2 vs T）")
    # 两参数拟合 steps_nc[t] ≈ ovh + c·t：把「eager 固定开销」与「每 token 计算」分开
    n = len(steps_nc)
    sx = sum(range(1, n + 1))
    sy = sum(steps_nc)
    sxx = sum(t * t for t in range(1, n + 1))
    sxy = sum(t * s for t, s in zip(range(1, n + 1), steps_nc))
    c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    ovh = sy / n - c * sx / n
    ovh_c = min(steps_c)                      # cache 单步下界（最短前缀处）
    print(f"    差距解释: 无 cache 单步 ≈ 固定开销 {ovh * 1e3:.2f} ms + 每 token 计算"
          f" {c * 1e3:.3f} ms（{n} 步最小二乘拟合）；cache 单步下界 {ovh_c * 1e3:.2f} ms")
    for mult, tag in ((1, "本 toy（实测区间）"), (100, "每 token 计算 ×100（≈真实大模型区间）")):
        w_nc = sum(ovh + c * mult * t for t in range(1, T + 1))
        w_c = T * (ovh_c + c * mult)
        print(f"    外推 {tag}: 重建加速 = {w_nc / w_c:6.1f}×"
              + ("（实测 {:.1f}×；重建未含 cache 侧 O(t) gather，故偏高）".format(wall_nc / wall_c) if mult == 1 else " → 逼近算术上界"))
    print(f"      结论: 算术上界 (T+1)/2 在「计算主导」区间成立；toy 模型每 token 计算太小，")
    print(f"      eager 开销两条路径都付，加速被压扁——这正是真实引擎要压开销的原因")
    print(f"    单步对照（无 cache 单步随前缀长度线性涨；cache 单步近恒定，")
    print(f"    微涨来自逐行 gather 拷贝 O(t)——真实引擎把这步融进 attention kernel）:")
    for t in (8, 32, 64, 128, 256):
        i = t - 1
        c_step = steps_c[min(i, len(steps_c) - 1)]
        print(f"      t={t:>3}: 无 cache {steps_nc[i] * 1e3:6.2f} ms"
              f" / cache {c_step * 1e3:5.2f} ms → {steps_nc[i] / c_step:5.1f}×"
              f"（计算比 ≈ {t}×）")
    assert gen_nc == gen_c, "KV cache 必须与重算前缀逐 token 一致"
    assert wall_nc > wall_c * 3, "cache 路径应显著更快"
    # 预分配 vs 追加（torch.cat 重分配）：纯张量操作探针（T_PROBE 取长些，
    # 否则 cat 的二次搬运量太小、测不出差异——256 步上两者持平，不构成证据）
    T_PROBE = 2048
    t0 = time.perf_counter()
    kv = torch.zeros(1, KV_HEADS, 1, HEAD_DIM)
    for _ in range(1, T_PROBE + 1):
        kv = torch.cat([kv, torch.randn(1, KV_HEADS, 1, HEAD_DIM)], dim=2)
    t_cat = time.perf_counter() - t0
    t0 = time.perf_counter()
    buf = torch.zeros(1, KV_HEADS, T_PROBE + 1, HEAD_DIM)
    for i in range(T_PROBE + 1):
        buf[:, :, i] = torch.randn(KV_HEADS, HEAD_DIM)
    t_buf = time.perf_counter() - t0
    print(f"    实现方式也是成本: {T_PROBE} 步追加式 torch.cat {t_cat * 1e3:.1f} ms"
          f" vs 预分配写入 {t_buf * 1e3:.1f} ms（{t_cat / t_buf:.0f}×，cat 每步搬运全量）"
          f" → 引擎一律预分配 KV 池")

    # ---- [2] batching 曲线：真实 batched forward ----
    print(f"\n[2] batching 曲线（每档 B 条序列 prefill 后齐步 decode {T_GEN} 步，"
          f"每档 3 遍取最快）")
    prompts = make_prompts(16)
    batched_decode_step([0, 0], [6, 7], [1, 1])      # 预热 batch 通路
    curve = {}
    for B in (1, 2, 4, 8, 16):
        ps, slots = prompts[:B], list(range(B))
        walls = []
        for _ in range(3):
            lasts = [prefill(p, s) for p, s in zip(ps, slots)]
            pos = [PROMPT_LEN] * B
            t0 = time.perf_counter()
            for _ in range(T_GEN):
                lasts = batched_decode_step(lasts, slots, pos)
                pos = [p + 1 for p in pos]
            walls.append(time.perf_counter() - t0)
        wall = min(walls)
        curve[B] = wall / T_GEN
        print(f"    B={B:>2}: 每迭代 {curve[B] * 1e3:6.2f} ms"
              f" | 吞吐 {B * T_GEN / wall:7.0f} tokens/s")
    solo = [decode_cached(p, 8, slot=i)[0] for i, p in enumerate(prompts[:4])]
    lasts = [prefill(p, s) for p, s in zip(prompts[:4], range(4))]
    batched = [[l] for l in lasts]
    pos = [PROMPT_LEN] * 4
    for _ in range(7):
        nv = batched_decode_step([b[-1] for b in batched], list(range(4)), pos)
        for bb, v in zip(batched, nv):
            bb.append(v)
        pos = [p + 1 for p in pos]
    match = all(b[:8] == s[:8] for b, s in zip(batched, solo))
    print(f"    batched(B=4) vs solo 逐序列一致: {match} ✅（batch 不改变每条序列的数学）")
    assert match
    a, b, r2 = fit_affine(curve)
    share16 = a / (a + 16 * b)
    print(f"    拟合 t(B) = {a * 1e3:.2f} + {b * 1e3:.2f}·B ms（R²={r2:.3f}）")
    print(f"    固定项占比 @B=16: {share16 * 100:.0f}%"
          f"（L0 模型 W_READ=1.0 → {1.0 / (1.0 + 16 * 0.02) * 100:.0f}%）")
    print(f"    → CPU 上固定项很小: decode 是 compute-bound，batch 摊薄的是算力；")
    print(f"      L0 的大固定项是 GPU「每步读一遍权重」的 HBM 物理（GPU 待真机验证）")

    # ---- [3] static vs continuous：真实请求、真实调度 ----
    reqs = REQ_LENGTHS
    prompts8 = make_prompts(len(reqs))
    print(f"\n[3] 真实调度（{len(reqs)} 个请求 lengths={reqs}，B={BATCH}，"
          f"prompt={PROMPT_LEN} token，各 3 遍取最快墙钟）")
    reps_s = [serve_static(prompts8, reqs) for _ in range(3)]
    reps_c = [serve_continuous(prompts8, reqs) for _ in range(3)]
    tok_s, it_s, waste_s, lat_s, wall_s = min(reps_s, key=lambda r: r[-1])
    tok_c, it_c, waste_c, lat_c, wall_c = min(reps_c, key=lambda r: r[-1])
    exp_it_s, exp_lat_s, exp_waste = ledger_static(reqs)
    exp_it_c, exp_lat_c, hist = ledger_continuous(reqs)
    print(f"    static    : 迭代 {it_s} | 空转 token-step {waste_s}"
          f" | 墙钟 {wall_s * 1e3:6.0f} ms | 完成迭代 {lat_s}")
    print(f"    continuous: 迭代 {it_c} | 空转 token-step {waste_c}"
          f" | 墙钟 {wall_c * 1e3:6.0f} ms | 完成迭代 {lat_c}")
    print(f"    L0 账本对照: static {exp_it_s} 迭代 / continuous {exp_it_c} 迭代"
          f" / bubble {exp_waste} —— 真实调度器逐位复现 L0 模拟器 ✅")
    print(f"    continuous 每迭代 decode 活跃数: {hist}（0 = 纯接纳迭代）")
    same_tok = all(ts == tc for ts, tc in zip(tok_s, tok_c))
    same_reps = all(r[0] == tok_c for r in reps_c) and all(r[0] == tok_s for r in reps_s)
    mark = "✅" if same_tok else "❌"
    print(f"    两种调度生成 token 逐请求一致: {same_tok} {mark}"
          f"（调度改变「何时算」，不改变「算什么」）")
    mark = "✅" if same_reps else "❌"
    print(f"    3 遍复跑生成内容逐位不变: {same_reps} {mark}（greedy 解码确定性）")
    assert it_s == exp_it_s and it_c == exp_it_c, "迭代账本必须与 L0 一致"
    assert lat_s == exp_lat_s and lat_c == exp_lat_c, "完成时刻必须与 L0 一致"
    assert waste_s == exp_waste and waste_c == 0
    assert same_tok and same_reps, "调度不得改变生成内容"
    uni = [5] * 8
    u_it_s, _, u_w = ledger_static(uni)
    u_it_c, _, _ = ledger_continuous(uni)
    assert u_it_s == u_it_c and u_w == 0
    print(f"    边界检查: 长度全为 5 时 static = continuous = {u_it_s} 迭代"
          f"（收益来自长度参差，与 L0 同款边界）")

    # ---- [4] 代价模型对照：用 [2] 的 (a,b) 预测 [3] 的墙钟 ----
    tp = []
    for _ in range(5):                               # 单条 prefill 墙钟（探针）
        t0 = time.perf_counter()
        prefill(prompts8[0], SLOTS - 1)
        tp.append(time.perf_counter() - t0)
    tp = min(tp)
    pred_s = sum(max(reqs[g0:g0 + BATCH]) - 1
                 for g0 in range(0, len(reqs), BATCH)) * (a + b * BATCH) \
        + len(reqs) * tp
    pred_c = sum((a + b * k) for k in hist if k > 0) + len(reqs) * tp
    err_s = (pred_s - wall_s) / wall_s
    err_c = (pred_c - wall_c) / wall_c
    print(f"\n[4] 用 [2] 拟合的 t(B) 预测 [3] 墙钟（decode 用拟合式，prefill 用探针最快"
          f" {tp * 1e3:.1f} ms）")
    print(f"    static    : 预测 {pred_s * 1e3:6.0f} ms / 实测 {wall_s * 1e3:6.0f} ms"
          f"（偏差 {err_s * 100:+.0f}%）")
    print(f"    continuous: 预测 {pred_c * 1e3:6.0f} ms / 实测 {wall_c * 1e3:6.0f} ms"
          f"（偏差 {err_c * 100:+.0f}%）")
    print(f"    结论: 仿射形式可拟合（R²={r2:.3f}）且能预测真实调度墙钟；")
    print(f"    残差来自 Python 调度循环与 batch 组装——真实引擎用 C++ 调度器压掉这层")

    print("\n" + "=" * 68)
    print("✅ self-check passed:")
    print("   cache==重算 逐 token 一致 / KV 池字节==公式 / batched==solo /")
    print("   迭代账本与完成时刻==L0 模拟器（23/16/44）/ 两调度生成一致 / 等长边界持平")
    print("=" * 68)
    print("\ntakeaway: L0 的机制在真实模型上全部成立——语义（算什么、多少迭代、")
    print("          多少浪费）由计划决定，与 L0 模拟器逐位一致；代价（墙钟、固定项")
    print("          占比）由硬件决定，CPU 与 GPU 物理不同。L2 上真实引擎（vLLM/SGLang,")
    print("          再在真实 GPU 环境把这份账对到生产数字。")


if __name__ == "__main__":
    with torch.no_grad():
        main()
