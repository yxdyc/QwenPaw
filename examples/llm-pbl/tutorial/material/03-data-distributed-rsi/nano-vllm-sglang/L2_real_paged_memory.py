"""nano-vllm-sglang · L2 真实分页内存管理
==========================================

目标：把 L1 的「每序列一整条连续预留 KV 槽」换成真正的分页内存管理——
    ① block table + 物理块池：KV 按块（BLOCK_SIZE token）按需分配，
       逻辑块→物理块映射住在 block table 里，forward 按表 gather/scatter；
    ② refcount + free queue：块完成即归还，归还的块立即可被下一个请求复用，
       且释放 ≠ 清空——ref_cnt=0 的块同时是空闲内存与缓存条目；
    ③ 前缀共享（content-hash 前缀缓存）：同前缀的请求 touch 同一物理块
       （ref_cnt+=1），后来者的 prefill 只算后缀——省显存也省算力；
    ④ copy-on-write：向共享块（ref_cnt>1）写入前先复制——共享只读、写时分裂；
    ⑤ 准入与抢占：块不够时抢占最晚的运行请求、释放其块，被抢占者稍后
       全量重算（recompute）恢复——语义不变，代价实付。

跨级别契约（机器断言）：
    - **paged decode 与 L1 连续池 decode 逐 token 一致**（同一权重、同一 prompt、
      greedy）——分页只改「KV 住在哪」，不改「算什么」；
    - 前缀共享 / CoW / 抢占重算后的生成序列与无共享无压力参考逐 token 一致。

声明（ROADMAP §三 可运行性契约）：
    - 模型为 L1 的随机初始化 GQA GPT（3,148,032 参数，state_dict 共享，
      逐参数断言相等）——真实权重 + 真实引擎（vLLM/SGLang on GPU）留
      Machine B 攒批通道 `[TODO: verify on real system]`。
    - 分页管理本身是真的：真实 block table / refcount / 块级 gather-scatter /
      内容哈希前缀缓存——不是对 vLLM 的 API 包装，是最小同构实现。
      与 vLLM V1 源码的逐条对照见 tutorial_L2 §7（2026-08-06 main 快照锚点）。
    - 本机 Apple Silicon CPU（torch fp32）。依赖仅 torch + 同目录 L1 模块
      （import 时设 sys.dont_write_bytecode，不落 __pycache__）。
    - greedy + 固定 seed：计数类输出确定；计时行（仅 [3]）随机器负载浮动。

运行：
    python L2_real_paged_memory.py
"""

import hashlib
import struct
import sys
import time
from collections import deque

sys.dont_write_bytecode = True          # import L1 不落 __pycache__（nano-slime 同款）
import torch

import L1_real_kv_batching as L1        # 跨级别契约的参照系（权重 + 连续池解码原语）

# ===== 配置 =====
BLOCK_SIZE = 4                          # nano 取小块让分配轨迹可见；vLLM 默认 16
NUM_BLOCKS = 64                         # 默认池：64 块 × 4 token
SEED = L1.SEED
BS = BLOCK_SIZE

torch.manual_seed(SEED)


# ===== 模型：L1 同款 GQA decoder，KV 改走 block table =====

class PagedMiniGPT(L1.MiniGPT):
    """与 L1.MiniGPT 逐参数同权重；forward 的 KV 前缀不再来自「每序列一条
    连续槽」，而是按 block table 从物理块池 gather，新 token 写回对应块。
    注意力数学与 L1 逐步一致 → 生成逐 token 一致（跨级别契约）。"""

    def __init__(self, pool):
        super().__init__()
        self.pool = pool                # BlockPool（持有 pool_k/pool_v 物理块张量）
        self.pool_k = None              # 屏蔽父类连续池，防误用
        self.pool_v = None

    def forward(self, x, pos_vec, block_tables):
        """x:[B,T]；pos_vec:[B] 各自已缓存前缀长度；block_tables: list[list[phys]]。
        调用前必须已为 pos_vec+T 分配好块（与 vLLM 先 allocate 后 forward 同序）。
        gather/scatter 用一次向量化索引完成（真实引擎把这步融进 attention kernel）。"""
        B, T = x.shape
        pos_vec = torch.as_tensor(pos_vec)
        max_pos = int(pos_vec.max())
        Klen = max_pos + T
        pos_id = pos_vec[:, None] + torch.arange(T)[None, :]
        h = self.tok(x) + self.pos(pos_id)
        j_idx = torch.arange(Klen)[None, None, :]
        allow = j_idx <= (pos_vec[:, None, None] + torch.arange(T)[None, :, None])
        # 每行的 (块号, 块内偏移) 索引：前缀 gather 用 pre_idx，新 token scatter 用 w_idx
        pre_idx, w_idx = [], []
        for b in range(B):
            pb = int(pos_vec[b])
            tbl = block_tables[b]
            if pb:
                blk = torch.tensor([tbl[p // BS] for p in range(pb)])
                pre_idx.append((blk, torch.arange(pb) % BS))
            else:
                pre_idx.append(None)
            wp = torch.arange(pb, pb + T)
            w_idx.append((torch.tensor([tbl[int(p) // BS] for p in wp.tolist()]),
                          wp % BS))
        for l in range(L1.LAYERS):
            g = self.norm1[l](h)
            q = self.wq[l](g).view(B, T, L1.HEADS, L1.HEAD_DIM).transpose(1, 2)
            k_new = self.wk[l](g).view(B, T, L1.KV_HEADS, L1.HEAD_DIM).transpose(1, 2)
            v_new = self.wv[l](g).view(B, T, L1.KV_HEADS, L1.HEAD_DIM).transpose(1, 2)
            k_full = torch.zeros(B, L1.KV_HEADS, Klen, L1.HEAD_DIM)
            v_full = torch.zeros(B, L1.KV_HEADS, Klen, L1.HEAD_DIM)
            for b in range(B):
                pb = int(pos_vec[b])
                if pb:
                    blk, off = pre_idx[b]
                    k_full[b, :, :pb] = self.pool.pool_k[l][blk, :, off].transpose(0, 1)
                    v_full[b, :, :pb] = self.pool.pool_v[l][blk, :, off].transpose(0, 1)
                k_full[b, :, pb:pb + T] = k_new[b]
                v_full[b, :, pb:pb + T] = v_new[b]
                wblk, woff = w_idx[b]
                self.pool.pool_k[l][wblk, :, woff] = k_new[b].transpose(0, 1)
                self.pool.pool_v[l][wblk, :, woff] = v_new[b].transpose(0, 1)
            k_rep = k_full.repeat_interleave(L1.HEADS // L1.KV_HEADS, dim=1)
            v_rep = v_full.repeat_interleave(L1.HEADS // L1.KV_HEADS, dim=1)
            s = q @ k_rep.transpose(-1, -2) / (L1.HEAD_DIM ** 0.5)
            s = s.masked_fill(~allow[:, None, :, :], float("-inf"))
            h = h + self.wo[l]((torch.softmax(s, -1) @ v_rep)
                               .transpose(1, 2).reshape(B, T, L1.HEADS * L1.HEAD_DIM))
            h = h + self.mlp[l](self.norm2[l](h))
        return self.head(self.final(h))


# ===== 块池：refcount + free queue + content-hash 前缀缓存 =====
# 对照 vLLM V1：KVCacheBlock(ref_cnt/block_hash) + FreeKVCacheBlockQueue +
# BlockPool.get_new_blocks/touch/free_blocks + hash_block_tokens（tutorial §7 锚点表）。

class Block:
    __slots__ = ("block_id", "ref_cnt", "block_hash")

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None          # 块被写满时登记的内容哈希（释放后仍保留）


class BlockPool:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free = deque(range(num_blocks))       # free queue（nano: FIFO）
        self.num_free = num_blocks
        self.hash_to_block = {}                    # 前缀缓存：block_hash -> block_id
        self.pool_k = [torch.zeros(num_blocks, L1.KV_HEADS, BS, L1.HEAD_DIM)
                       for _ in range(L1.LAYERS)]
        self.pool_v = [torch.zeros(num_blocks, L1.KV_HEADS, BS, L1.HEAD_DIM)
                       for _ in range(L1.LAYERS)]

    def bytes_total(self):
        return sum(t.nbytes for t in self.pool_k + self.pool_v)

    def get_new_blocks(self, n):
        """从 free queue 取 n 块，ref_cnt 0→1。不够即 ValueError——
        与 vLLM get_new_blocks 同款硬门槛（准入/抢占决策的依据）。"""
        if n > self.num_free:
            raise ValueError(f"Cannot get {n} free blocks from the pool "
                             f"(free={self.num_free})")
        ids = [self.free.popleft() for _ in range(n)]
        self.num_free -= n
        for i in ids:
            b = self.blocks[i]
            assert b.ref_cnt == 0
            if b.block_hash is not None:      # 重分配 → 旧缓存条目失效
                if self.hash_to_block.get(b.block_hash) == i:
                    del self.hash_to_block[b.block_hash]
                b.block_hash = None
            b.ref_cnt = 1
        return ids

    def touch(self, ids):
        """前缀缓存命中：ref_cnt += 1；若块曾在 free queue（ref_cnt=0）则移出——
        释放的块在真正被复用前仍是缓存（vLLM free_blocks/touch 语义）。"""
        for i in ids:
            b = self.blocks[i]
            if b.ref_cnt == 0:
                self.free.remove(i)
                self.num_free -= 1
            b.ref_cnt += 1

    def free_blocks(self, ids):
        """ref_cnt -= 1；归零即回 free queue。块内容与哈希保留——
        释放 ≠ 清空，ref_cnt=0 的块同时是「空闲内存」与「缓存条目」。"""
        for i in ids:
            b = self.blocks[i]
            assert b.ref_cnt > 0
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free.append(i)
                self.num_free += 1

    def cache_block(self, h, block_id):
        self.hash_to_block.setdefault(h, block_id)   # 同内容只认第一块
        self.blocks[block_id].block_hash = h

    def lookup(self, h):
        return self.hash_to_block.get(h)


def block_hash(parent, token_ids):
    """内容寻址：hash(parent_hash, 本块 token)——链式哈希使「同哈希」必然
    「同前缀内容」（对照 vLLM hash_block_tokens：parent_block_hash +
    curr_block_token_ids 进同一哈希）。"""
    hh = hashlib.sha256()
    hh.update(parent if parent is not None else b"\x00" * 32)
    hh.update(struct.pack(f"<{len(token_ids)}I", *token_ids))
    return hh.digest()


# ===== 序列状态与解码原语 =====
# 约定：seq.tokens = prompt + 已生成（含最后一个「其 KV 尚未缓存」的 token 之前
# 的全部）；num_computed = 已有 KV 的 token 数。原语返回下一个 token，
# **由调用者 append**（与 vLLM「调度器持有 token、worker 只算」的分工同构）。

class Seq:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.table = []                 # 逻辑块 -> 物理块
        self.num_computed = 0
        self.last_hash = None           # 已登记链的最后一个满块哈希
        self.num_cached_blocks = 0      # 已登记进前缀缓存的满块数
        self.done = False
        self.target = None              # [4] 调度用：目标总 token 数

    def n_blocks(self, total_tokens):
        return (total_tokens + BS - 1) // BS

    def ensure_blocks(self, pool, total_tokens):
        """按需分配：总块数 = ceil(total/BS)，只补差额（L0「用多少发多少」落地）。"""
        need = self.n_blocks(total_tokens)
        if need > len(self.table):
            self.table.extend(pool.get_new_blocks(need - len(self.table)))


def register_full_blocks(pool, seq, up_to):
    """把新写满的块登记进前缀缓存（只缓存满块——vLLM 经典策略；partial 条目
    是 V1 细粒度扩展，见 tutorial §7.4）。链式哈希从 seq.last_hash 续接。"""
    full = up_to // BS
    for j in range(seq.num_cached_blocks, full):
        h = block_hash(seq.last_hash, seq.tokens[j * BS:(j + 1) * BS])
        seq.last_hash = h
        pool.cache_block(h, seq.table[j])
    seq.num_cached_blocks = full


@torch.no_grad()
def paged_prefill(pool, model, seq, use_prefix_cache=True):
    """prefill：先前缀缓存查找（满块链式哈希命中），再做零副作用准入检查，
    然后 **先把命中块移出 free queue（touch）、再分配后缀块**，最后对后缀跑
    一次 forward。touch 必须先于分配：若先分配，FIFO free queue 可能弹出
    即将命中的块本身，同一物理块会在同一 block table 里一表双别名，后缀
    scatter 覆写命中 KV——静默发散（竞态实录见 tutorial_L3 §6；vLLM V1
    两阶段分配 issue #33775 同因）。准入不足即 raise，此前零副作用。
    全命中时对最后一个 token 重算一格（至少算一个——vLLM 同款约束）。
    返回下一个 token（调用者 append）。"""
    toks = seq.tokens
    hit_ids, h = [], None
    if use_prefix_cache:
        for j in range(len(toks) // BS):
            hh = block_hash(h, toks[j * BS:(j + 1) * BS])
            bid = pool.lookup(hh)
            if bid is None:
                break
            h, hit_ids = hh, hit_ids + [bid]
    hit_tokens = len(hit_ids) * BS
    n_new = seq.n_blocks(len(toks)) - len(hit_ids)
    # 准入检查（此刻零副作用）：仍留在 free queue 里的命中块（ref_cnt=0）
    # 马上要被 touch 收回，不能算可用空闲——与 vLLM 把「尚在 free queue 的
    # 命中块」计入容量检查同款（single_type_kv_cache_manager.py:L218-228，
    # 2026-08-13 抓取）。不足即 raise，调用者（调度器）据此抢占。
    in_queue = sum(1 for i in hit_ids if pool.blocks[i].ref_cnt == 0)
    if n_new > pool.num_free - in_queue:
        raise ValueError(f"Cannot get {n_new} free blocks from the pool "
                         f"(free={pool.num_free}, of which {in_queue} are "
                         f"hit blocks to be reclaimed first)")
    # touch-before-allocate：命中块先移出 free queue，get_new_blocks 就绝
    # 不会弹出它们（vLLM V1 allocate_slots 同序，见 tutorial_L2 §8）。
    if hit_ids:
        pool.touch(hit_ids)
    new_ids = pool.get_new_blocks(n_new)
    if hit_ids:
        seq.table = hit_ids + new_ids
        seq.last_hash = h
        seq.num_cached_blocks = len(hit_ids)
    else:
        seq.table = new_ids
    if hit_tokens < len(toks):
        suffix, pos = toks[hit_tokens:], hit_tokens
    else:                                   # 全命中：重算最后一个 token
        suffix, pos = [toks[-1]], hit_tokens - 1
        seq.num_computed = pos
    logits = model(torch.tensor([suffix]), torch.tensor([pos]), [seq.table])
    seq.num_computed = len(toks)
    register_full_blocks(pool, seq, len(toks))
    return int(logits[0, -1].argmax())


@torch.no_grad()
def paged_decode_step(pool, model, seq):
    """一个 decode 步：输入 = 最后 token（其 KV 尚未缓存），pos = num_computed。
    返回下一个 token（调用者 append）。"""
    pos = seq.num_computed
    seq.ensure_blocks(pool, pos + 1)
    logits = model(torch.tensor([[seq.tokens[-1]]]), torch.tensor([pos]),
                   [seq.table])
    seq.num_computed = pos + 1
    if seq.num_computed % BS == 0:
        register_full_blocks(pool, seq, seq.num_computed)
    return int(logits[0, -1].argmax())


def append_with_cow(pool, model, seq):
    """追加一个 token（decode 步）；若落点块被共享（ref_cnt>1）先 CoW：
    块级复制 → 原块 ref_cnt-=1 → block table 改指新块。
    返回 (是否 CoW, 源块, 目的块, 新 token)。"""
    pos = seq.num_computed
    j = pos // BS
    seq.ensure_blocks(pool, pos + 1)
    phys = seq.table[j]
    cow, src = False, None
    if pool.blocks[phys].ref_cnt > 1:       # 共享块禁写
        new = pool.get_new_blocks(1)[0]
        for l in range(L1.LAYERS):          # 块级复制（真实张量拷贝）
            pool.pool_k[l][new] = pool.pool_k[l][phys]
            pool.pool_v[l][new] = pool.pool_v[l][phys]
        pool.blocks[phys].ref_cnt -= 1
        seq.table[j] = new
        cow, src = True, phys
        phys = new
    tok = paged_decode_step(pool, model, seq)
    seq.tokens.append(tok)
    return cow, src, phys, tok


def fork(pool, parent):
    """parallel sampling 式 fork：复制 block table、逐块 touch（ref_cnt+=1）。
    子序列与父序列共享全部已有块，直到某一方写入（触发 CoW）。"""
    child = Seq(parent.tokens)
    child.table = list(parent.table)
    child.last_hash = parent.last_hash
    child.num_computed = parent.num_computed
    child.num_cached_blocks = parent.num_cached_blocks
    pool.touch(child.table)
    return child


def make_tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, L1.VOCAB, (n,), generator=g).tolist()


def main():
    print("=" * 68)
    print("nano-vllm-sglang L2 — 真实分页内存管理（block table + 物理块池）")
    print("=" * 68)
    print(f"torch {torch.__version__} | CPU fp32 | seed={SEED} | "
          f"BLOCK_SIZE={BS}（vLLM 默认 16：config/cache.py:L47）")
    print("声明: 权重 = L1 随机初始化 GQA GPT（state_dict 共享，逐参数断言相等）；")
    print("      分页管理为最小同构真实现（非 API 包装）；真实引擎 (vLLM/SGLang")
    print("      on GPU) 见 [TODO: verify on real system]；源码对照 tutorial_L2 §7")

    pool0 = BlockPool(NUM_BLOCKS)
    model = PagedMiniGPT(pool0)
    model.load_state_dict(L1.model.state_dict())
    model.eval()
    assert all(torch.equal(a, b) for a, b in
               zip(model.state_dict().values(), L1.model.state_dict().values()))

    # ---- [0] 池与模型：整池启动即划走，块为最小供给单位 ----
    kv_per_tok = 2 * L1.LAYERS * L1.KV_HEADS * L1.HEAD_DIM * 4      # fp32
    block_bytes = BS * kv_per_tok
    print(f"\n[0] 物理块池: {NUM_BLOCKS} 块 × {BS} token × {kv_per_tok} B/token"
          f" = {pool0.bytes_total() / 2**20:.1f} MiB（启动即划走，对齐真实引擎）")
    print(f"    每块 = {block_bytes:,} B；池字节 == 公式: "
          f"{pool0.bytes_total() == NUM_BLOCKS * block_bytes} ✅")
    print(f"    对照 L1 连续池: {L1.SLOTS} 槽 × {L1.MAX_T} token 整条预留"
          f"（每序列预留 {L1.MAX_T} token，不管实际多长）")
    assert pool0.bytes_total() == NUM_BLOCKS * block_bytes

    # ---- [1] 跨级别契约：paged == L1 连续池，逐 token 一致 ----
    T_NEW = 48
    prompt = L1.make_prompts(1)[0]                    # 8 token，L1 同款 seed=7
    gen_ref, _ = L1.decode_cached(prompt, T_NEW, slot=0)
    pool1 = BlockPool(NUM_BLOCKS)
    model.pool = pool1
    seq = Seq(prompt)
    first = paged_prefill(pool1, model, seq, use_prefix_cache=False)
    seq.tokens.append(first)
    gen_paged = [first]
    trace = [(seq.num_computed, len(seq.table))]      # (已有 KV 的 token 数, 块数)
    for _ in range(T_NEW - 1):
        t = paged_decode_step(pool1, model, seq)
        seq.tokens.append(t)
        gen_paged.append(t)
        trace.append((seq.num_computed, len(seq.table)))
    jumps = [(t, n) for (t, n), (_, pn) in zip(trace[1:], trace[:-1]) if n != pn]
    mark = "✅" if gen_paged == gen_ref else "❌"
    print(f"\n[1] 跨级别契约（prompt={len(prompt)} token，生成 {T_NEW}，greedy）")
    print(f"    paged decode == L1 连续池 decode（逐 token）: "
          f"{gen_paged == gen_ref} {mark}")
    print(f"    按需分配轨迹: prompt {len(prompt)} tok → {trace[0][1]} 块；"
          f"此后每写满 {BS} token 加 1 块")
    print(f"    加块时刻 (KV token 数 → 块数): {jumps}")
    expect = all(n == (t + BS - 1) // BS for t, n in trace)
    print(f"    块数 == ceil(tokens/{BS}) 全程成立: {expect} ✅")
    assert gen_paged == gen_ref, "分页不得改变生成内容"
    assert expect and trace[0][1] == (len(prompt) + BS - 1) // BS

    # ---- [2] 碎片与准入账本：同一预算，分页装得下、连续装不下 ----
    reqs = L1.REQ_LENGTHS                               # [6,2,9,3,14,5,1,8]
    toks = [L1.PROMPT_LEN + l for l in reqs]            # prompt + 生成
    need = [(t + BS - 1) // BS for t in toks]
    budget = sum(need)                                  # 分页峰值块数 = 32
    pool_p = BlockPool(budget)
    admitted_p = 0
    for t in toks:                                      # 全部并发，按到达分配峰值
        pool_p.get_new_blocks((t + BS - 1) // BS)
        admitted_p += 1
    max_t = max(toks)
    cont_blocks = (max_t + BS - 1) // BS                # 连续预留：每人按最长
    admitted_c = 0
    pool_c = BlockPool(budget)
    for _ in toks:
        try:
            pool_c.get_new_blocks(cont_blocks)
            admitted_c += 1
        except ValueError:
            break
    slots_p = budget * BS
    waste_p = slots_p - sum(toks)
    slots_c = len(toks) * cont_blocks * BS
    waste_c = slots_c - sum(toks)
    print(f"\n[2] 碎片与准入（{len(toks)} 个真实请求，tokens={toks}，Σ={sum(toks)}）")
    print(f"    分页  : 峰值 {sum(need)} 块 = {slots_p} 槽位"
          f"（内部碎片 {waste_p}，每请求 ≤ {BS - 1}）→ 预算 {budget} 块准入 "
          f"{admitted_p}/{len(toks)}")
    print(f"    连续  : 每请求预留 {max_t} tok = {cont_blocks} 块 → "
          f"{slots_c} 槽位（浪费 {waste_c}，占 {100 * waste_c / slots_c:.0f}%）"
          f"→ 同预算准入 {admitted_c}/{len(toks)}")
    print(f"    → 省下的块直接变成并发度（{admitted_p} vs {admitted_c}）；"
          f"L1 [2] 已实测吞吐随 B 单调上升——显存省下的部分就是吞吐")
    assert admitted_p == len(toks) and admitted_c < admitted_p
    assert waste_p <= len(toks) * (BS - 1)

    # ---- [3] 前缀共享（content-hash 前缀缓存）+ copy-on-write ----
    pool2 = BlockPool(NUM_BLOCKS)
    model.pool = pool2
    SHARED = make_tokens(128, 11)                       # 共享前缀 = 32 个满块
    pa, pb_ = SHARED + make_tokens(4, 12), SHARED + make_tokens(6, 13)
    seqA = Seq(pa)
    firstA = paged_prefill(pool2, model, seqA)          # A 先跑：33 满块全登记缓存
    seqA.tokens.append(firstA)
    free_after_A = pool2.num_free
    seqB = Seq(pb_)
    g0B = paged_prefill(pool2, model, seqB)             # B 命中共享前缀
    seqB.tokens.append(g0B)
    n_shared = len(SHARED) // BS                        # 命中的满块数 = 32
    shared_ids = seqB.table[:n_shared]
    ref_shared = [pool2.blocks[i].ref_cnt for i in shared_ids[:2]]
    new_blocks_B = len(seqB.table) - n_shared
    hit_tokens = n_shared * BS
    genB = [g0B]
    for _ in range(4):
        t = paged_decode_step(pool2, model, seqB)
        seqB.tokens.append(t)
        genB.append(t)
    # 无共享参考（独立池，全算）
    pool_ref = BlockPool(NUM_BLOCKS)
    model.pool = pool_ref
    seqBr = Seq(pb_)
    g = paged_prefill(pool_ref, model, seqBr, use_prefix_cache=False)
    seqBr.tokens.append(g)
    genB_ref = [g]
    for _ in range(4):
        t = paged_decode_step(pool_ref, model, seqBr)
        seqBr.tokens.append(t)
        genB_ref.append(t)
    # prefill 墙钟对照（各 3 遍取最快；命中版每次用全新池 + A 铺缓存）
    walls_full, walls_hit = [], []
    for _ in range(3):
        s = Seq(pb_)
        t0 = time.perf_counter()
        paged_prefill(BlockPool(NUM_BLOCKS), model, s, use_prefix_cache=False)
        walls_full.append(time.perf_counter() - t0)
        pt = BlockPool(NUM_BLOCKS)
        paged_prefill(pt, model, Seq(pa), use_prefix_cache=False)
        s2 = Seq(pb_)
        t0 = time.perf_counter()
        paged_prefill(pt, model, s2)
        walls_hit.append(time.perf_counter() - t0)
    print(f"\n[3] 前缀共享 + copy-on-write（BLOCK_SIZE={BS}，只缓存满块）")
    print(f"    A: prompt {len(pa)} tok（共享前缀 {len(SHARED)} + 私有 4）→ "
          f"{len(seqA.table)} 块全部写满、登记缓存（free {NUM_BLOCKS}→{free_after_A}）")
    print(f"    B: prompt {len(pb_)} tok，前 {hit_tokens} tok 链式哈希命中 → "
          f"touch 共享块（ref_cnt={ref_shared}），只新分配 {new_blocks_B} 块、"
          f"只计算后缀 {len(pb_) - hit_tokens} tok")
    print(f"    prefill 墙钟: 全算 {min(walls_full) * 1e3:.2f} ms / "
          f"命中后 {min(walls_hit) * 1e3:.2f} ms"
          f"（{min(walls_full) / min(walls_hit):.1f}×，命中段计算整个跳过）")
    print(f"    共享版 B 生成 == 无共享参考（逐 token）: {genB == genB_ref} ✅"
          f"（因果性使前缀 KV 与后续上下文无关 → 共享无损）")
    assert genB == genB_ref and all(r == 2 for r in ref_shared)
    assert new_blocks_B == seqB.n_blocks(len(pb_)) - len(shared_ids)

    # -- CoW：fork 两个孩子，写共享半块触发复制 --
    pool3 = BlockPool(NUM_BLOCKS)
    model.pool = pool3
    p10 = make_tokens(10, 14)                           # 2 满块 + 1 半块(2 tok)
    P = Seq(p10)
    gP = paged_prefill(pool3, model, P, use_prefix_cache=False)
    P.tokens.append(gP)                                 # 11 tok：KV 10 个 + 1 个已生成
    c1, c2 = fork(pool3, P), fork(pool3, P)             # 三块 ref_cnt: P+c1+c2 = 3
    ref_after_fork = [pool3.blocks[i].ref_cnt for i in P.table]
    snap = [pool3.pool_k[l][P.table[2]].clone() for l in range(L1.LAYERS)]
    cows = []
    for _ in range(3):
        cow, src, dst, _ = append_with_cow(pool3, model, c1)
        if cow:
            cows.append((1, src, dst, pool3.blocks[src].ref_cnt + 1))
        cow, src, dst, _ = append_with_cow(pool3, model, c2)
        if cow:
            cows.append((2, src, dst, pool3.blocks[src].ref_cnt + 1))
    snap_ok = all(torch.equal(s, pool3.pool_k[l][P.table[2]])
                  for l, s in enumerate(snap))
    pool_solo = BlockPool(NUM_BLOCKS)                   # 同起点 solo 参考
    model.pool = pool_solo
    S = Seq(p10)
    g = paged_prefill(pool_solo, model, S, use_prefix_cache=False)
    S.tokens.append(g)
    for _ in range(3):
        t = paged_decode_step(pool_solo, model, S)
        S.tokens.append(t)
    print(f"    CoW: P(10 tok = 2 满块 + 1 半块，已生成 1 tok) fork 两子"
          f"（fork 后各块 ref_cnt={ref_after_fork}，半块同被共享）")
    for (who, src, dst, ref_b) in cows:
        print(f"      child{who} 写入共享半块（写前 ref_cnt={ref_b} > 1）→ "
              f"复制块 {src} → {dst}，原块降为 {ref_b - 1}")
    print(f"    父 P 的半块内容在两次 CoW 后逐字节不变: {snap_ok} ✅"
          f"（共享块只读，写时分裂）")
    print(f"    child1/child2 全轨迹 == solo 参考（14 tok 逐个）: "
          f"{c1.tokens == S.tokens and c2.tokens == S.tokens} ✅")
    assert snap_ok and c1.tokens == S.tokens and c2.tokens == S.tokens
    assert len(cows) == 2, "半块被两子先后写入，应恰好触发两次 CoW"

    # ---- [4] 释放复用 + 抢占/重算 ----
    # [4a] 完成即释放；释放的块立即复用，且释放≠清空——仍是前缀缓存条目
    p12 = make_tokens(12, 15)                           # 12 tok = 3 满块
    pool4 = BlockPool(4)
    model.pool = pool4
    r0 = Seq(p12)                                       # 池仅 4 块
    g = paged_prefill(pool4, model, r0, use_prefix_cache=False)
    r0.tokens.append(g)
    for _ in range(2):
        t = paged_decode_step(pool4, model, r0)
        r0.tokens.append(t)
    used0 = list(r0.table)
    pool4.free_blocks(r0.table)                         # 完成 → 归还
    free_mid = pool4.num_free
    r1 = Seq(p12)                                       # 同前缀新请求
    g = paged_prefill(pool4, model, r1)                 # 命中「已释放」的块
    r1.tokens.append(g)
    reused = [b for b in r1.table if b in used0]
    print(f"\n[4] 释放复用 + 抢占/重算")
    print(f"    [4a] r0: {len(r0.tokens)} tok 用块 {used0} → 完成归还"
          f"（free 0→{free_mid}）")
    print(f"         同前缀 r1 进场: 3 满块全部命中已释放块 {reused}"
          f"（ref_cnt 0→1，移出 free queue，free {free_mid}→{pool4.num_free}），"
          f"只重算 1 个 token（全命中至少算一格）")
    assert free_mid == 4 and pool4.num_free == 1 and len(reused) == 3

    # [4b] 紧预算下的准入 + 抢占 + 重算（8 个真实请求，预算 12 块）
    BUDGET = 12

    def run_scheduler(budget, preempt=True):
        pool = BlockPool(budget)
        model.pool = pool
        seqs = [Seq(L1.make_prompts(1, seed=100 + i)[0]) for i in range(len(reqs))]
        for i, r in enumerate(seqs):
            r.target = L1.PROMPT_LEN + reqs[i]
        pending, running, done = list(range(len(seqs))), [], {}
        events, it = [], 0

        def preempt(victim):
            v = seqs[victim]
            events.append((it, victim, v.num_computed, len(v.table)))
            pool.free_blocks(v.table)
            v.table, v.num_computed = [], 0
            v.last_hash, v.num_cached_blocks = None, 0
            running.remove(victim)
            pending.insert(0, victim)

        while len(done) < len(seqs):
            it += 1
            assert it < 1000, "调度死循环"
            # 1) 运行中请求各前进一步
            for i in list(running):
                if i not in running:
                    continue                # 本阶段早先已被抢占
                r = seqs[i]
                if r.num_computed == 0:     # 新请求 / 抢占恢复：(重)prefill
                    is_rec = len(r.tokens) > L1.PROMPT_LEN
                    while True:
                        try:
                            tok = paged_prefill(pool, model, r,
                                                use_prefix_cache=False)
                            break
                        except ValueError:
                            if not preempt:
                                raise
                            victim = running[-1]        # FCFS: 抢占最晚者
                            preempt(victim)
                            if victim == i:
                                break
                    if i not in running:
                        continue
                    if is_rec:
                        events.append(("recompute", i, len(r.tokens)))
                else:
                    if r.num_computed + 1 > len(r.table) * BS:   # 要跨块
                        while True:
                            try:
                                r.ensure_blocks(pool, r.num_computed + 1)
                                break
                            except ValueError:
                                if not preempt:
                                    raise
                                victim = running[-1]
                                preempt(victim)
                                if victim == i:
                                    break
                        if i not in running:
                            continue
                    tok = paged_decode_step(pool, model, r)
                r.tokens.append(tok)
                if len(r.tokens) >= r.target:
                    r.done = True
            # 2) 完成的让位（释放块；满块哈希留在缓存）
            for i in [i for i in running if seqs[i].done]:
                pool.free_blocks(seqs[i].table)
                done[i] = it
                running.remove(i)
            # 3) 准入：free ≥ 该请求全程所需块（full_sequence_must_fit 同构门槛；
            #    不为运行中请求的未来增长预留 → 过载时由抢占兜底，见 tutorial §6.3）
            while pending:
                i = pending[0]
                if pool.num_free >= seqs[i].n_blocks(seqs[i].target):
                    pending.pop(0)
                    running.append(i)
                else:
                    break
        return seqs, events, it

    ref_reqs, _, _ = run_scheduler(NUM_BLOCKS, preempt=False)   # 无压力参考
    ref_tokens = [r.tokens for r in ref_reqs]
    pr_reqs, events, iters = run_scheduler(BUDGET, preempt=True)
    n_pre = [e for e in events if isinstance(e[0], int)]
    n_rec = [e for e in events if e[0] == "recompute"]
    rec_tokens = sum(e[2] for e in n_rec)
    same_all = all(p.tokens == r for p, r in zip(pr_reqs, ref_tokens))
    print(f"    [4b] 预算 {BUDGET} 块（8 请求峰值需 {sum(need)} 块）: "
          f"完成迭代 {iters} | 抢占 {len(n_pre)} 次（FCFS: 抢占最晚运行者）| "
          f"重算恢复 {len(n_rec)} 次（重算 prefill 共 {rec_tokens} tok）")
    for e in events:
        if isinstance(e[0], int):
            print(f"      iter {e[0]:>2}: 请求 {e[1]} 被抢占"
                  f"（已有 KV {e[2]} tok，释放 {e[3]} 块，num_computed 归 0）")
        else:
            print(f"      请求 {e[1]} 重算恢复: 对 prompt+已生成共 {e[2]} tok 重跑 prefill")
    print(f"    抢占/重算后 8 个请求最终 token == 无压力参考（逐个）: "
          f"{same_all} ✅（抢占改变「何时算」，不改变「算什么」）")
    assert same_all and len(n_pre) > 0, "预算必须紧到真的触发抢占"

    # ---- [5] self-check ----
    print("\n" + "=" * 68)
    print("✅ self-check passed:")
    print("   paged == L1 连续池逐 token 一致 / 块数 == ceil(tok/BS) 全程 /")
    print("   分页准入 8/8 vs 连续 5/8（同预算）/ 前缀命中 ref_cnt=2 且生成无损 /")
    print("   CoW 恰 2 次且父块字节不变 / 释放块被复用且仍是缓存 /")
    print("   抢占重算后全部 token 与无压力参考一致")
    print("=" * 68)
    print("\ntakeaway: 分页把「显存管理」从每序列连续预留变成块池 + block table：")
    print("          按需供给消内部碎片，refcount + 内容哈希让前缀可共享可复用，")
    print("          CoW 保住共享语义，准入/抢占把过载变成可恢复的代价而非崩溃。")
    print("          语义（生成什么）一分不差，代价（块数/重算）变成实数。")
    print("          L3 对照 vLLM block manager / SGLang RadixAttention 源码。")


if __name__ == "__main__":
    with torch.no_grad():
        main()
