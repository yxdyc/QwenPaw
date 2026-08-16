"""nano-vllm-sglang · L3 RadixAttention：radix tree 前缀缓存 + 多请求共享
=========================================================================

目标（README 阶梯 L3 行）：对照 vLLM block manager / SGLang RadixAttention 源码，
分析多请求共享前缀的场景。L2 已用「链式内容哈希 + FIFO free queue」实现了 vLLM 式
前缀缓存（块粒度、隐式驱逐）；L3 只加一层——**把前缀缓存的组织换成 radix tree**
（SGLang RadixAttention 的最小同构实现），并跑出三个哈希链做不到的场景：

    ① 并发共享：N 个同时在跑的请求共享同一 system prompt 的物理块，
       节点 lock_ref 保护运行中前缀不被驱逐（论文 Fig 2 / inc_lock_ref）；
    ② 叶子优先 LRU 驱逐：压力下来时先驱逐最久未用的**叶子**，公共祖先留在树里
       ——哈希链的块级 FIFO 按释放顺序复用：脊柱块最先被释放、最先被复用，
       整条命中链从头部断掉（[4] 实测）；
    ③ cache-aware 调度（LPM）：等待队列按「已命中前缀长度」降序准入，
       避免不相关请求交替进场造成 cache thrashing（论文 §3 / schedule_policy lpm）。

跨级别契约（机器断言，L1→L2→L3 三级同构）：
    - radix 缓存介入（命中 / 分裂 / 驱逐 / 加锁）后，每个请求的生成序列与
      L1 连续池参考、L2 哈希链引擎**逐 token 一致**——内存组织只改
      「KV 住在哪、何时算」，不改「算什么」；
    - 共享块全程只读：radix 世界不需要 copy-on-write（节点不可变，
      写入只发生在请求自己的新块上）——SYS 脊柱块字节全程不变（断言）。

声明（课程可运行性契约）：
    - 模型为 L1 的随机初始化 GQA GPT（3,148,032 参数，state_dict 共享，
      逐参数断言相等）——真实权重 + 真实引擎（vLLM/SGLang on GPU）留
      真实 GPU/多机环境 `[TODO: verify on real system]`。
    - radix tree 前缀缓存是真的：真实树结构 / match-split-insert / LRU 叶子驱逐 /
      lock_ref 保护 / 块级物理共享——不是 API 包装，是最小同构实现。
      与 SGLang radix_cache.py、vLLM v1 block_pool.py 的逐条对照见 tutorial_L3 §7
      （2026-08-13 main 抓取锚点）。
    - 块粒度（BLOCK_SIZE=4，继承 L2）而非 SGLang 的 token 粒度——取舍与原因
      见 tutorial_L3 §7（SGLang 现行 main 的 page_aligned 截断语义同因）。
    - 本机 Apple Silicon CPU（torch fp32）。依赖仅 torch + 同目录 L1/L2 模块
      （import 时设 sys.dont_write_bytecode，不落 __pycache__）。
    - L2 对照组（[4]）跑在修复后的 L2 引擎上：allocate-before-touch 竞态
      （半成品审计发现，机器证明在案）已修为 touch-before-allocate，与 vLLM V1
      两阶段分配同因（issue #33775）；竞态实录、修复与 L2 自场景输出不变性
      证明见 tutorial_L3 §6。
    - greedy + 固定 seed + 逻辑时钟：计数类输出确定；计时行（仅 [3]）以
      「elapsed」开头，可用 `sed '/^[[:space:]]*elapsed/d'` 整行掩码（继承 L2 口径）。

运行：
    python L3_radix_prefix_sharing.py
"""

import heapq
import sys
import time

sys.dont_write_bytecode = True          # import L1/L2 不落 __pycache__
import torch

import L1_real_kv_batching as L1        # 权重 + 连续池参考（跨级别契约的终点）
import L2_real_paged_memory as L2       # 分页引擎 + 哈希链前缀缓存（对照组）

BS = L2.BS                              # 4（vLLM 默认 16：config/cache.py:L48）
SEED = L1.SEED

torch.manual_seed(SEED)


# ===== 物理块池：SGLang allocator 同构——只管空闲块，不管共享语义 =====
# 对照 SGLang：token_to_kv_pool_allocator 只分配/回收 KV 槽位，「谁在引用、
# 何时可驱逐」全部由 radix tree 负责（与 L2 BlockPool 的块级 refcount+哈希
# 一肩挑相反）。nano 保留 ref_cnt 字段仅作 0/1 占用断言用。

class RadixBlockPool(L2.BlockPool):
    def get_new_blocks(self, n):
        if n > self.num_free:
            raise ValueError(f"Cannot get {n} free blocks (free={self.num_free})")
        ids = [self.free.popleft() for _ in range(n)]
        self.num_free -= n
        for i in ids:
            assert self.blocks[i].ref_cnt == 0
            self.blocks[i].ref_cnt = 1
        return ids

    def free_blocks(self, ids):
        for i in ids:
            b = self.blocks[i]
            assert b.ref_cnt == 1
            b.ref_cnt = 0
            b.block_hash = None
            self.free.append(i)
            self.num_free += 1


# ===== radix tree 前缀缓存（SGLang RadixAttention 最小同构） =====
# 对照 radix_cache.py（2026-08-13 main 抓取）：TreeNode:L216 / match_prefix:L352 /
# _insert_helper:L704 / evict:L562 / inc_lock_ref:L592 / _split_node:L674。
# 差异逐条见 tutorial_L3 §7；此处只保留论文骨架的五件事：
# match（可分裂）/ insert / LRU 叶子驱逐 / lock_ref / 逻辑时钟。

class Node:
    __slots__ = ("node_id", "key", "blocks", "children", "parent",
                 "lock_ref", "last_access")

    def __init__(self, node_id, key, blocks, parent):
        self.node_id = node_id
        self.key = key                  # token 元组（长度恒为 BS 的倍数）
        self.blocks = blocks            # 与 key 等长的物理块表（len == len(key)//BS）
        self.children = {}              # 首块 token 元组 -> 子节点
        self.parent = parent
        self.lock_ref = 0               # 多少个运行中请求覆盖此节点（论文 reference counter）
        self.last_access = 0            # 逻辑时钟（SGLang 用 time.monotonic；nano 用逻辑钟保确定）


class RadixCache:
    def __init__(self, pool):
        self.pool = pool
        self.root = Node(0, (), [], None)
        self._next_id = 1
        self.clock = 0
        self.evict_log = []             # (clock, node_id, 块数)

    def tick(self):
        self.clock += 1
        return self.clock

    def _new_node(self, key, blocks, parent):
        n = Node(self._next_id, tuple(key), list(blocks), parent)
        n.last_access = self.clock
        self._next_id += 1
        return n

    def _split(self, child, split_len):
        """边中分裂：new parent 继承前 split_len 个 token/块，child 保留其余。
        对照 _split_node:L674——key/blocks 切开、lock_ref 由父子同值持有
        （同一批请求同时覆盖两段）；nano 的 split_len 恒在块边界上（§7 差异 1）。"""
        new = Node(self._next_id, child.key[:split_len],
                   child.blocks[:split_len // BS], child.parent)
        new.last_access = self.clock
        new.lock_ref = child.lock_ref
        self._next_id += 1
        new.children = {child.key[split_len:split_len + BS]: child}
        child.parent.children[child.key[:BS]] = new
        child.key = child.key[split_len:]
        child.blocks = child.blocks[split_len // BS:]
        child.parent = new
        return new

    def _walk(self, toks):
        """match/insert 共用的走查：沿首块字典下行，边内逐 token 比较；
        边中失配 → 在最后一个块边界分裂（dict 键 = 首块 token，故命中边
        至少共享一整块，分裂点恒 ≥1 块——SGLang child_key(page_size) 同构）。
        返回 (终止节点, 已匹配 token 数)。沿路刷新 last_access（LRU touch）。"""
        node, pos, n_full = self.root, 0, (len(toks) // BS) * BS
        while pos < n_full:
            child = node.children.get(toks[pos:pos + BS])
            if child is None:
                break
            child.last_access = self.clock
            m = min(len(child.key), n_full - pos)
            common = 0
            while common < m and child.key[common] == toks[pos + common]:
                common += 1
            if common < len(child.key):
                node = self._split(child, (common // BS) * BS)
                pos += len(node.key)
            else:
                pos += len(child.key)
                node = child
        return node, pos

    def match_prefix(self, toks):
        """最长前缀匹配。返回 (命中 token 数, 命中物理块表, 末端节点)。
        对照 match_prefix:L352：命中可能触发分裂（结构细化、不复制数据）；
        块内失配截断到最后块边界（page_aligned 语义，§7 差异 1）。"""
        self.tick()
        node, pos = self._walk(tuple(toks))
        blocks = []
        p = node
        while p is not self.root:
            blocks = p.blocks + blocks
            p = p.parent
        assert len(blocks) * BS == pos
        return pos, blocks, node

    def insert(self, toks, blocks):
        """把 (token 序列, 块表) 登记进树：已匹配路径复用，其余挂新节点。
        对照 _insert_helper:L704 / cache_finished_req:L434（prompt+输出都入树）。
        返回末端节点（供 lock/dec_lock）。"""
        self.tick()
        toks = tuple(toks)
        node, pos = self._walk(toks)
        n_full = (len(toks) // BS) * BS
        if pos < n_full:
            child = self._new_node(toks[pos:n_full], blocks[pos // BS:n_full // BS],
                                   node)
            node.children[toks[pos:pos + BS]] = child
            node = child
        return node

    def evict(self, need_blocks):
        """显式 LRU 驱逐：候选 = lock_ref==0 的叶子，按 (last_access, node_id)
        建堆；弹出即整节点释放（块归还池、节点摘除），其父若变成新叶子且
        未加锁则入堆。对照 evict:L562 + 论文「evicts the least recently used
        leaf first …… re-use of their common ancestors until those ancestors
        become leaves」。返回实际释放块数（可能超过 need——整叶释放）。"""
        heap = []

        def collect(n):
            for c in n.children.values():
                if not c.children and c.lock_ref == 0:
                    heap.append((c.last_access, c.node_id, c))
                else:
                    collect(c)
        collect(self.root)
        heapq.heapify(heap)
        freed = 0
        while freed < need_blocks and heap:
            _, _, x = heapq.heappop(heap)
            if x.parent is None or x.children or x.lock_ref > 0:
                continue                        # 陈旧堆项（父节点入堆后状态又变）
            self.pool.free_blocks(x.blocks)
            self.evict_log.append((self.clock, x.node_id, len(x.blocks)))
            del x.parent.children[x.key[:BS]]
            p, x.parent = x.parent, None
            freed += len(x.blocks)
            if p is not self.root and not p.children and p.lock_ref == 0:
                heapq.heappush(heap, (p.last_access, p.node_id, p))
        return freed

    def inc_lock_ref(self, node):
        """沿父链加锁到 root（对照 inc_lock_ref:L592）。运行中请求覆盖的
        整条前缀路径不可驱逐——论文「A node is evictable if its reference
        counter is zero」的执行侧。"""
        while node is not self.root:
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node):
        while node is not self.root:
            assert node.lock_ref > 0
            node.lock_ref -= 1
            node = node.parent

    def tree_blocks(self):
        tot = [0]

        def rec(n):
            for c in n.children.values():
                tot[0] += len(c.blocks)
                rec(c)
        rec(self.root)
        return tot[0]

    def pretty(self):
        lines = ["root"]

        def rec(n, depth):
            for c in sorted(n.children.values(), key=lambda x: x.node_id):
                lines.append("    " * depth +
                             f"└ n{c.node_id} [{len(c.key)} tok / {len(c.blocks)} blk] "
                             f"lock={c.lock_ref} acc=t{c.last_access} "
                             f"key={list(c.key[:BS])}{'…' if len(c.key) > BS else ''}")
                rec(c, depth + 1)
        rec(self.root, 1)
        return "\n".join(lines)


# ===== 请求生命周期：scheduler + cache 的最小同构 =====
# 对照 SGLang 分工：match_prefix + inc_lock_ref 在批次加入时
# （schedule_policy.py:L936 _req_inc_lock_ref）、cache_finished_req 在完成时
# （radix_cache.py:L434）。nano 把「prompt 入树」提前到准入时刻——与 SGLang
# maybe_cache_unfinished_req（运行中入树供并发共享）同目的，取舍见 tutorial §7。

class RadixReq:
    def __init__(self, prompt, max_new):
        self.prompt = list(prompt)
        self.tokens = list(prompt)
        self.max_new = max_new
        self.table = []
        self.num_computed = 0
        self.last_node = None
        self.locked_node = None           # 准入时加锁的节点（完成时对同一路径解锁）
        self.hit_len = 0
        self.computed = 0               # prefill 实算 token 数（命中段跳过）
        self.done = False


def admit(cache, pool, req):
    """准入 = match → (LRU 驱逐补块) → 分配 → prompt 入树 → 加锁。
    块在准入时按 prompt+max_new 全程预分配（L2 full_sequence_must_fit 同款门槛；
    SGLang 按 token 增量分配、不足即驱逐——差异见 tutorial §7）。"""
    hit_len, hit_blocks, _ = cache.match_prefix(req.prompt)
    total = (len(req.prompt) + req.max_new + BS - 1) // BS
    need = total - len(hit_blocks)
    if pool.num_free < need:
        cache.evict(need - pool.num_free)
    new_blocks = pool.get_new_blocks(need)
    req.table = hit_blocks + new_blocks
    req.hit_len = hit_len
    req.last_node = cache.insert(req.prompt, req.table)
    cache.inc_lock_ref(req.last_node)
    req.locked_node = req.last_node
    return req


@torch.no_grad()
def radix_prefill(model, req):
    toks = req.tokens
    if req.hit_len < len(toks):
        suffix, pos = toks[req.hit_len:], req.hit_len
    else:                                   # 全命中：至少算一格（vLLM/SGLang 同款约束）
        suffix, pos = [toks[-1]], req.hit_len - 1
        req.num_computed = pos
    logits = model(torch.tensor([suffix]), torch.tensor([pos]), [req.table])
    req.num_computed = len(toks)
    req.computed = len(toks) - req.hit_len if req.hit_len < len(toks) else 1
    return int(logits[0, -1].argmax())


@torch.no_grad()
def radix_decode_step(model, req):
    pos = req.num_computed
    assert pos // BS < len(req.table), "块已在准入时全程预分配"
    logits = model(torch.tensor([[req.tokens[-1]]]), torch.tensor([pos]),
                   [req.table])
    req.num_computed = pos + 1
    return int(logits[0, -1].argmax())


def finish(cache, req):
    """完成 = (prompt+生成) 全量入树 + 解锁。对照 cache_finished_req:L434：
    输出也入树——多轮对话的下一轮、self-consistency 的兄弟采样都靠它命中。
    解锁对准准入时加锁的节点（新插入的 gen 节点天生未锁、即刻可驱逐）。"""
    req.last_node = cache.insert(req.tokens, req.table)
    cache.dec_lock_ref(req.locked_node)
    req.done = True


def run_radix(cache, pool, model, prompt, max_new):
    """顺序跑完一个请求，返回 (生成 token 列表, prefill 实算 token 数)。"""
    req = admit(cache, pool, RadixReq(prompt, max_new))
    gen = [radix_prefill(model, req)]
    req.tokens.append(gen[0])
    for _ in range(max_new - 1):
        t = radix_decode_step(model, req)
        req.tokens.append(t)
        gen.append(t)
    finish(cache, req)
    return gen, req.computed, req


# ===== L2 哈希链引擎的等价 harness（[4] 对照组） =====

def l2_hit_len(pool, toks):
    h, hit = None, 0
    for j in range(len(toks) // BS):
        hh = L2.block_hash(h, toks[j * BS:(j + 1) * BS])
        bid = pool.lookup(hh)
        if bid is None:
            break
        h, hit = hh, hit + 1
    return hit * BS


def run_l2(pool, model, prompt, max_new):
    """L2 引擎跑同一请求：哈希链前缀缓存 + FIFO free queue + 完成即 free。"""
    hit = l2_hit_len(pool, prompt)
    seq = L2.Seq(prompt)
    first = L2.paged_prefill(pool, model, seq)
    seq.tokens.append(first)
    gen = [first]
    for _ in range(max_new - 1):
        t = L2.paged_decode_step(pool, model, seq)
        seq.tokens.append(t)
        gen.append(t)
    computed = len(prompt) - hit if hit < len(prompt) else 1
    pool.free_blocks(seq.table)
    return gen, computed


def make_tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, L1.VOCAB, (n,), generator=g).tolist()


def main():
    print("=" * 68)
    print("nano-vllm-sglang L3 — RadixAttention：radix tree 前缀缓存 + 多请求共享")
    print("=" * 68)
    print(f"torch {torch.__version__} | CPU fp32 | seed={SEED} | BLOCK_SIZE={BS}")
    print("声明: 权重 = L1 随机初始化 GQA GPT（state_dict 共享，逐参数断言相等）；")
    print("      radix tree 为最小同构真实现（非 API 包装），对照 SGLang")
    print("      radix_cache.py / vLLM v1 block_pool.py（2026-08-13 main 抓取，")
    print("      tutorial_L3 §7）；真实引擎 (vLLM/SGLang on GPU) 见")
    print("      [TODO: verify on real system]")

    def mark(b):                            # ✅/❌ 一律由实测布尔派生，不硬编
        return "✅" if b else "❌"

    model = L2.PagedMiniGPT(RadixBlockPool(4))
    model.load_state_dict(L1.model.state_dict())
    model.eval()
    assert all(torch.equal(a, b) for a, b in
               zip(model.state_dict().values(), L1.model.state_dict().values()))

    # ---- [1] 跨级别契约：radix 命中路径 == L1 连续池，逐 token 一致 ----
    G = 4                                   # 每请求生成 4 token（恰 1 块）
    pool1 = RadixBlockPool(32)
    model.pool = pool1
    cache1 = RadixCache(pool1)
    p1 = make_tokens(24, 61)                # 6 个满块
    gen_a, comp_a, _ = run_radix(cache1, pool1, model, p1, G)
    gen_b, comp_b, _ = run_radix(cache1, pool1, model, p1, G)   # 全命中
    gen_ref, _ = L1.decode_cached(p1, G, slot=0)
    ok1 = gen_a == gen_b == gen_ref
    print(f"\n[1] 跨级别契约（prompt={len(p1)} tok，生成 {G}，greedy）")
    print(f"    第 1 次冷跑: 命中 0，实算 {comp_a} tok；第 2 次同 prompt: "
          f"全命中（{len(p1)} tok 全在树里），只重算最后 1 格（实算 {comp_b}）")
    print(f"    radix 命中路径 == radix 冷跑 == L1 连续池（逐 token）: "
          f"{ok1} {mark(ok1)}")
    assert ok1 and comp_b == 1

    # ---- [2] 并发共享：N 个运行中请求共享同一 system prompt 的物理块 ----
    N = 4
    SYS2 = make_tokens(24, 51)              # 6 块共享前缀
    prompts = [SYS2 + make_tokens(8, 52 + i) for i in range(N)]
    budget2 = 6 + N * 3                     # 恰 = 共享 6 块 + 每请求私有 3 块
    pool2 = RadixBlockPool(budget2)
    model.pool = pool2
    cache2 = RadixCache(pool2)
    reqs = [admit(cache2, pool2, RadixReq(p, G)) for p in prompts]
    spine = None                            # 找到 SYS2 脊柱节点
    for c in cache2.root.children.values():
        spine = c
    while spine.children and len(spine.children) == 1 and spine.lock_ref == N:
        spine = list(spine.children.values())[0]
    lock_spine = spine.lock_ref
    uniq = len(set(b for r in reqs for b in r.table))
    naive = sum(len(r.table) for r in reqs)
    snap = None                             # r0 prefill 写出脊柱 KV 后立即快照
    gens = []
    for step in range(G):                   # 轮转 decode，保持并发
        for i, r in enumerate(reqs):
            if step == 0:
                t = radix_prefill(model, r)
                r.tokens.append(t)
                gens.append([t])
                if i == 0:
                    snap = [pool2.pool_k[l][spine.blocks[0]].clone()
                            for l in range(L1.LAYERS)]
            else:
                t = radix_decode_step(model, r)
                r.tokens.append(t)
                gens[reqs.index(r)].append(t)
    for r in reqs:
        finish(cache2, r)
    snap_ok = all(torch.equal(s, pool2.pool_k[l][spine.blocks[0]])
                  for l, s in enumerate(snap))
    solo_ok = True
    for i, p in enumerate(prompts):
        ref, _ = L1.decode_cached(p, G, slot=0)
        solo_ok = solo_ok and gens[i] == ref
    print(f"\n[2] 并发共享（{N} 个请求同时在跑，共享 {len(SYS2)}-token system prompt）")
    print(f"    准入后树: 1 个脊柱节点持 {len(spine.blocks)} 个共享块，"
          f"lock_ref={lock_spine}（= 运行中请求数，驱逐禁区）")
    print(f"    物理块占用: 唯一块 {uniq} = 共享 6 + 私有 {N}×3；"
          f"若不共享 = {naive}（每请求各持一份）→ 省 {naive - uniq} 块 "
          f"({100 * (naive - uniq) // naive}%)")
    print(f"    共享块全程只读（并发 prefill+decode 后逐字节不变）: {snap_ok} "
          f"{mark(snap_ok)}；{N} 条轨迹 == 各自 solo 参考: {solo_ok} {mark(solo_ok)}")
    assert lock_spine == N and uniq == budget2 and naive == N * 9
    assert snap_ok and solo_ok

    # ---- [3] 树从 insert 里长出来：分裂让分支共享脊柱 ----
    SYS = make_tokens(32, 21)               # 8 块脊柱
    DOCA, DOCB = make_tokens(16, 22), make_tokens(16, 23)
    sA1, sB1 = make_tokens(4, 24), make_tokens(4, 25)
    pool3 = RadixBlockPool(30)
    model.pool = pool3
    cache3 = RadixCache(pool3)
    pA1, pB1 = SYS + DOCA + sA1, SYS + DOCB + sB1
    genA1, compA1, _ = run_radix(cache3, pool3, model, pA1, G)
    tree_after_1 = cache3.pretty()
    genB1, compB1, reqB1 = run_radix(cache3, pool3, model, pB1, G)
    tree_after_2 = cache3.pretty()
    # 墙钟对照（继承 L2 [3] 协议：更长 prompt、prefill 全程、3 遍取最快）——
    # 52-tok toy prompt 整请求计时被 Python 开销主导（实测 ~1.1×），撑不起
    # 「命中段计算整个跳过」的直觉，诚实定位见 tutorial_L3 §10。
    L_SYS = make_tokens(128, 34)                # 32 块长脊柱（L2 SHARED 同形）
    p_long = L_SYS + make_tokens(8, 35)
    walls_c, walls_h = [], []
    for _ in range(3):
        pool_t = RadixBlockPool(64)
        model.pool = pool_t
        cache_t = RadixCache(pool_t)
        t0 = time.perf_counter()
        run_radix(cache_t, pool_t, model, p_long, 1)     # 冷跑：全算 + 入树
        walls_c.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_radix(cache_t, pool_t, model, p_long, 1)     # 全命中：只重算最后 1 格
        walls_h.append(time.perf_counter() - t0)
    print(f"\n[3] 树从 insert 里长出来（rA1 = SYS+DOCA+sA1，rB1 = SYS+DOCB+sB1）")
    print(f"    rA1 冷跑后（一条链，{compA1} tok 全算）:")
    print(tree_after_1)
    print(f"    rB1 进场: 前 {reqB1.hit_len} tok 命中 SYS 脊柱 → 边中分裂"
          f"（DOCA/DOCB 在第 {len(SYS)} tok 分叉），只实算 {compB1} tok")
    print(tree_after_2)
    print(f"    elapsed: 长 prompt（{len(p_long)} tok）整请求（prefill+1 decode）"
          f"冷跑 {min(walls_c) * 1e3:.2f} ms / 全命中 {min(walls_h) * 1e3:.2f} ms"
          f"（{min(walls_c) / min(walls_h):.1f}×；toy 尺度比值定位见 tutorial §10）")
    assert compA1 == len(pA1) and compB1 == len(pB1) - len(SYS)
    assert reqB1.hit_len == len(SYS)

    # ---- [4] 叶子优先 LRU vs 块级 FIFO：压力下的前缀存活 ----
    # 工作负载（全新池+缓存，rA1/rB1 重建树；[3] 的树不续用）：两个 one-shot
    # 洪水请求 F1/F2（各 9 块 = 8 prompt + 1 gen）把压力顶满——洪水要大到
    # 把 L2 的 FIFO free queue 卷过 SYS 脊柱块（脊柱最先被释放 → 最先被复用），
    # 而 radix 的叶子优先驱逐只牺牲分支。参数经 scratch 原型网格搜索选定，
    # 使「FIFO 截断 vs 叶优先存活」在实测中成立（tutorial §4 实测推导）。
    sW, sA2, sB2 = make_tokens(4, 26), make_tokens(4, 27), make_tokens(4, 28)
    F1, F2 = make_tokens(32, 30), make_tokens(32, 31)     # one-shot 洪水（各 9 块）
    BUDGET = 20                     # 紧到 FIFO 必须复用「带哈希的缓存块」
    workload = [("rA1", pA1), ("rB1", pB1), ("F1", F1),
                ("rB_warm", SYS + DOCB + sW), ("F2", F2),
                ("rA2", SYS + DOCA + sA2), ("rB2", SYS + DOCB + sB2)]
    nocache = sum(len(p) for _, p in workload)

    def run_workload_radix():
        pool = RadixBlockPool(BUDGET)
        model.pool = pool
        cache = RadixCache(pool)
        comp, gens, spine_alive = [], {}, []
        spine_id = None
        for name, p in workload:
            g, c, req = run_radix(cache, pool, model, p, G)
            if name == "rA1":               # rA1 后树是单链；rB1 后分裂出脊柱
                spine_id = None
            if spine_id is None and name == "rB1":
                n = req.last_node
                while n.parent is not cache.root:
                    n = n.parent
                spine_id = n.node_id
            if spine_id is not None:
                ids = set()

                def rec(x):
                    ids.add(x.node_id)
                    for c in x.children.values():
                        rec(c)
                rec(cache.root)
                spine_alive.append(spine_id in ids)
            comp.append(c)
            gens[name] = g
        return comp, gens, spine_alive, cache, pool

    def run_workload_l2():
        pool = L2.BlockPool(BUDGET)
        model.pool = pool
        comp, gens, hits = [], {}, []
        for name, p in workload:
            hits.append(l2_hit_len(pool, p))
            g, c = run_l2(pool, model, p, G)
            comp.append(c)
            gens[name] = g
        return comp, gens, hits

    comp_r, gens_r, spine_alive, cache_r, pool_r = run_workload_radix()
    comp_l, gens_l, hits_l = run_workload_l2()
    gens_ref = {}
    for name, p in workload:
        g, _ = L1.decode_cached(p, G, slot=0)
        gens_ref[name] = g
    sem_r = all(gens_r[n] == gens_ref[n] for n, _ in workload)
    sem_l = all(gens_l[n] == gens_ref[n] for n, _ in workload)
    hits_r = [len(p) - c if c > 1 else len(p)             # radix 每请求命中（实测派生）
              for c, (_, p) in zip(comp_r, workload)]
    trunc = [(name, hits_l[i], hits_r[i])                 # FIFO 截断点（实测派生）
             for i, (name, _) in enumerate(workload) if hits_l[i] < hits_r[i]]
    print(f"\n[4] 叶子优先 LRU vs 块级 FIFO（{len(workload)} 请求同序列、同预算 "
          f"{BUDGET} 块、无缓存基线 {nocache} tok；L2 引擎含 touch-before-")
    print(f"    allocate 修复，见 §6）")
    print(f"    {'请求':<8} {'radix 实算':>10} {'L2 哈希链实算':>13}   说明")
    notes = {0: "冷跑（树空）", 1: "命中 SYS 脊柱",
             2: "one-shot 洪水 ①（9 块）", 3: "回访 B 分支",
             4: "one-shot 洪水 ②（9 块）", 5: "回访 A 分支",
             6: "回访 B 分支"}
    for i, (name, p) in enumerate(workload):
        print(f"    {name:<8} {comp_r[i]:>10} {comp_l[i]:>13}   {notes[i]}")
    print(f"    合计实算: radix {sum(comp_r)} tok / L2 哈希链 {sum(comp_l)} tok / "
          f"无缓存 {nocache} tok（radix 省 {100 * (nocache - sum(comp_r)) // nocache}%，"
          f"比哈希链再省 {sum(comp_l) - sum(comp_r)} tok）")
    print(f"    radix 驱逐日志（clock, 节点, 块数）: {cache_r.evict_log}")
    ok_spine = all(spine_alive)
    print(f"    SYS 脊柱在全部驱逐中存活: {ok_spine} {mark(ok_spine)}"
          f"（叶子优先 → 祖先可复用，直到自己变成叶子）")
    print(f"    命中轨迹（tok）: radix {hits_r} / L2 哈希链 {hits_l}")
    tdesc = "、".join(f"{n}（L2 hit {h} vs radix hit {r}）" for n, h, r in trunc)
    print(f"    FIFO 截断点: {tdesc if tdesc else '无'}——FIFO 按释放顺序复用，"
          f"脊柱块最先释放、最先被复用，哈希链从头部断掉（§4/§10 解读）")
    print(f"    语义不变: radix/L2 两引擎 {len(workload)} 个请求最终 token == L1 参考: "
          f"{sem_r}/{sem_l} {mark(sem_r and sem_l)}（内存组织不改『算什么』）")
    assert comp_r == [52, 20, 32, 20, 32, 20, 20]
    assert comp_l == [52, 20, 32, 52, 32, 52, 20]
    assert hits_l == [0, 32, 0, 0, 0, 0, 32]
    assert ok_spine and sem_r and sem_l and len(trunc) >= 2

    # ---- [5] cache-aware 调度：LPM（最长前缀优先）vs FCFS ----
    # 两组前缀 P1/P2，到达序刻意交错（A1,B1,A2,B2,A3,B3）。预算压到 15 块，
    # 使两组前缀无法共存——调度顺序直接决定 cache thrashing 程度。
    P1, P2 = make_tokens(24, 41), make_tokens(24, 42)
    sfx = [make_tokens(8, 43 + i) for i in range(6)]
    groups = [("A1", P1 + sfx[0]), ("B1", P2 + sfx[1]), ("A2", P1 + sfx[2]),
              ("B2", P2 + sfx[3]), ("A3", P1 + sfx[4]), ("B3", P2 + sfx[5])]

    def run_policy(policy):
        pool = RadixBlockPool(15)
        model.pool = pool
        cache = RadixCache(pool)
        pending = list(groups)
        order, comp, gens = [], [], {}
        while pending:
            if policy == "lpm":
                scored = []
                for name, p in pending:
                    hit, _, _ = cache.match_prefix(p)
                    scored.append((-hit, name, p))
                scored.sort(key=lambda x: (x[0], [n for n, _ in groups].index(x[1])))
                pick = scored[0]
                pending = [x for x in pending if x[0] != pick[1]]
                name, p = pick[1], pick[2]
            else:
                name, p = pending.pop(0)
            g, c, _ = run_radix(cache, pool, model, p, G)
            order.append(name)
            comp.append(c)
            gens[name] = g
        return order, comp, gens

    ord_f, comp_f, gens_f = run_policy("fcfs")
    ord_l, comp_l5, gens_l5 = run_policy("lpm")
    same_out = all(gens_f[n] == gens_l5[n] for n, _ in groups)
    ref_ok = True
    for name, p in groups:
        g, _ = L1.decode_cached(p, G, slot=0)
        ref_ok = ref_ok and gens_f[name] == g
    print(f"\n[5] cache-aware 调度（预算 15 块，两组前缀无法共存；到达序 "
          f"{[n for n, _ in groups]}）")
    print(f"    FCFS 准入序 {ord_f}: 实算 {comp_f} = {sum(comp_f)} tok"
          f"（每组前缀都被对方挤掉 → 每个请求都冷跑，cache thrashing）")
    print(f"    LPM  准入序 {ord_l}: 实算 {comp_l5} = {sum(comp_l5)} tok"
          f"（同组连续进场，前缀住在树里 → 组内只冷跑一次）")
    print(f"    调度只改『谁先算』: 六个请求最终 token 与顺序无关"
          f"（FCFS==LPM==L1 参考）: {same_out and ref_ok} {mark(same_out and ref_ok)}")
    assert ord_l == ["A1", "A2", "A3", "B1", "B2", "B3"]
    assert sum(comp_f) == 192 and sum(comp_l5) == 96 and same_out and ref_ok

    # ---- [6] self-check（数字全部派生自上方实测变量，不硬编） ----
    print("\n" + "=" * 68)
    print("✅ self-check passed:")
    print(f"   radix 命中路径 == L1 连续池逐 token 一致 / 并发 {N} 请求共享脊柱块")
    print(f"   （lock_ref={lock_spine}，物理块 {uniq} vs {naive}）且轨迹 == solo / "
          f"分裂让分支共享脊柱 /")
    print(f"   叶子优先 LRU 保住 SYS 脊柱（radix {sum(comp_r)} vs 哈希链 "
          f"{sum(comp_l)} vs 无缓存 {nocache} tok）/")
    print(f"   LPM 消除 thrashing（{sum(comp_l5)} vs {sum(comp_f)} tok）"
          f"且语义与顺序无关 / 全程零 CoW")
    print("=" * 68)
    print("\ntakeaway: 前缀缓存的两种权威组织——vLLM 用链式内容哈希（块级、隐式")
    print("          驱逐），SGLang 用 radix tree（显式 LRU 叶子驱逐 + lock_ref 保护")
    print("          + cache-aware 调度）。树把『前缀』变成一等数据结构：共享是节点,")
    print("          驱逐是剪叶，调度看树选请求——于是『省算力』从运气变成策略。")
    print("          语义（生成什么）一分不差，差别全在代价（实算 token / 块存活）。")


if __name__ == "__main__":
    with torch.no_grad():
        main()
