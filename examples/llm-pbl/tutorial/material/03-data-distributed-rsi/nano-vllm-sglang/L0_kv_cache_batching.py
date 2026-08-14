"""nano-vllm-sglang · L0 玩具实现
====================================

目标：用 ~170 行纯标准库抓住高吞吐 LLM 推理的四个最小机制——
    ① KV cache：decode 用显存换计算（K/V 投影从 O(T²) 降到 O(T)），但显存随长度线性涨；
    ② batching：decode 是 memory-bandwidth-bound，大 batch 摊薄权重读取 → 吞吐上升；
    ③ continuous batching：iteration 级调度，序列 EOS 即让位、新请求即刻进场；
    ④ PagedAttention 思想：KV 按页分配、按需供给，消除预留碎片 → 更大 batch → 吞吐。

四个实验 + 边界检查。这是 L0（玩具级）：代价模型 + 迭代模拟器，无真实张量/GPU，
所有数字都是 toy 模型输出（方向与相对量级），不是真机 benchmark。
与 vLLM / SGLang 的对应只到概念层，源码对照留 L2/L3 [TODO: verify source]。
参考：vLLM/PagedAttention 论文 arXiv:2309.06180（Kwon et al.）；
iteration-level scheduling 出自 Orca（Yu et al., OSDI 2022）；
SGLang 论文 arXiv:2312.07104（Zheng et al.）。

运行：
    python L0_kv_cache_batching.py
"""

import math

# ===== toy 代价模型与形状常量（显式 toy 口径）=====
# 「时间」单位 = 一次完整权重读取耗时；所有数值是模型输出，非真机秒数。

W_READ = 1.0      # 每个 decode 迭代：读一遍全部权重（memory-bandwidth-bound，与 batch 无关）
KV_STEP = 0.02    # 每个 decode 迭代：batch 里每条序列追加的 KV 读取 + attention 代价

# 每 token 的 KV cache 体积（Llama-2-7B 形状：32 层 / 32 KV head / head_dim=128 / fp16）
N_LAYERS, N_KV_HEADS, HEAD_DIM, BYTES = 32, 32, 128, 2


def kv_per_token_bytes() -> int:
    """每 token 的 KV cache 字节数 = 2(K,V) × 层数 × KV head 数 × head_dim × dtype 字节。"""
    return 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * BYTES


def iter_time(batch: int) -> float:
    """一个 decode 迭代的耗时 ≈ 权重读取（固定）+ batch × 每序列 KV 代价。"""
    return W_READ + batch * KV_STEP


# ===== 机制一：KV cache =====


def kv_projection_ops(T: int, use_cache: bool) -> int:
    """生成 T 个 token 的 K/V 投影计算量（单层，toy 计数）。
    无 cache：第 t 步重算 t 个前缀 token 的 K/V，总量 Σt = T(T+1)/2；
    有 cache：每步只算新 token 的 K/V（历史在 cache 里直接读），总量 T。"""
    return T if use_cache else T * (T + 1) // 2


def kv_cache_bytes(T: int) -> int:
    """缓存一条序列 T 个 token 的 KV 所需字节数——随 T 线性增长。"""
    return kv_per_token_bytes() * T


# ===== 机制三：static vs continuous batching =====


def sim_static(lengths, batch_size):
    """static batching：按到达顺序每 B 个打包；整批跑到最长序列结束，
    提前完成的序列空占槽位（bubble），且下一批必须等整批结束。"""
    t, latencies, bubbles = 0, [], 0
    for i in range(0, len(lengths), batch_size):
        grp = lengths[i:i + batch_size]
        t_end = t + max(grp)
        for L in grp:
            latencies.append(t + L)          # 自己的 token 在 t+L 时生成完
            bubbles += max(grp) - L          # 之后空等到 t_end
        t = t_end
    return t, latencies, bubbles


def sim_continuous(lengths, batch_size):
    """continuous batching（iteration 级调度）：每个迭代所有活跃序列前进一步，
    谁 EOS 谁让位，等待队列里的请求同一迭代立刻进场。"""
    pending, active = list(range(len(lengths))), {}
    t, latencies = 0, [None] * len(lengths)
    while pending or active:
        while pending and len(active) < batch_size:   # 有空位就放人进来
            active[pending.pop(0)] = 0
        t += 1
        for i in list(active):
            active[i] += 1
            if active[i] == lengths[i]:
                latencies[i] = t
                del active[i]
    return t, latencies


def main() -> None:
    print("=" * 64)
    print("nano-vllm-sglang L0 — 高吞吐推理的最小机制")
    print("=" * 64)

    # ---- [1] KV cache：用显存换计算 ----
    T = 512
    no_cache, with_cache = kv_projection_ops(T, False), kv_projection_ops(T, True)
    mib = kv_per_token_bytes() / 2**20
    print(f"\n[1] KV cache（生成长度 T={T}，单层 K/V 投影计数）")
    print(f"    无 cache（每步重算前缀）: {no_cache:,} 次")
    print(f"    有 cache（每步只算新 token）: {with_cache:,} 次   → 计算量 ÷{(no_cache / with_cache):.1f}"
          f"（精确比值 (T+1)/2）")
    print(f"    代价：每 token 要存 {mib:.1f} MiB KV"
          f"（{N_LAYERS} 层 × {N_KV_HEADS} KV head × {HEAD_DIM} dim × 2(K,V) × {BYTES}B）")
    print(f"    → 一条 4096 token 的序列 KV 占 {kv_cache_bytes(4096) / 2**30:.1f} GiB；"
          f"KV 显存成为 serving 的第一瓶颈")
    assert no_cache == T * (T + 1) // 2 and with_cache == T

    # ---- [2] batching：为什么吞吐随 batch 涨 ----
    print(f"\n[2] decode 迭代代价模型：iter_time = {W_READ}（读权重）+ B × {KV_STEP}（KV）")
    print(f"    {'B':>4} | {'迭代耗时':>8} | {'吞吐 tokens/单位时间':>12}")
    prev = 0.0
    for B in (1, 2, 4, 8, 16, 32, 64):
        tps = B / iter_time(B)
        print(f"    {B:>4} | {iter_time(B):>8.2f} | {tps:>12.1f}")
        assert tps > prev, "吞吐必须随 batch 单调上升（本模型内）"
        prev = tps
    print(f"    上界：B→∞ 时吞吐 → 1/{KV_STEP} = {1 / KV_STEP:.0f} tokens/单位时间（被 KV 代价封顶）")
    print(f"    → 想高吞吐就得把 batch 塞满；问题变成「怎么塞满又不浪费」")

    # ---- [3] static vs continuous batching ----
    lengths = [6, 2, 9, 3, 14, 5, 1, 8]          # 8 个请求的生成 token 数（长短不一）
    B = 4
    t_s, lat_s, bub_s = sim_static(lengths, B)
    t_c, lat_c = sim_continuous(lengths, B)
    print(f"\n[3] 8 个请求 lengths={lengths}，batch_size={B}")
    print(f"    static    : makespan = {t_s} 迭代 | 平均延迟 = {sum(lat_s) / len(lat_s):.2f}"
          f" | 最长等待 = {max(lat_s)} | bubble = {bub_s} 槽位·迭代")
    print(f"    continuous: makespan = {t_c} 迭代 | 平均延迟 = {sum(lat_c) / len(lat_c):.2f}"
          f" | 最长等待 = {max(lat_c)} | bubble = 0（有等待就不空槽）")
    print(f"    逐请求延迟 static → continuous: {list(zip(lat_s, lat_c))}")
    assert t_c < t_s and sum(lat_c) < sum(lat_s), "continuous 应同时改善吞吐与延迟"
    uni = [5] * 8
    u_s, _, u_b = sim_static(uni, B); u_c, _ = sim_continuous(uni, B)
    assert u_s == u_c and u_b == 0, "边界：长度全相等时两者应持平"
    print(f"    边界检查：长度全为 5 时 static = continuous = {u_s} 迭代"
          f"（收益来自长度参差，长度对齐时无增益）")

    # ---- [4] PagedAttention 思想：KV 按页分配，消除预留碎片 ----
    L_MAX, PAGE = 16, 4                          # 生成长度上限 16，页大小 4 token
    cont = sum(L_MAX for _ in lengths)           # 连续预分配：每人预留 L_MAX
    paged = sum(math.ceil(l / PAGE) * PAGE for l in lengths)
    need = sum(lengths)
    print(f"\n[4] 同样 8 条序列（ΣL={need}），L_MAX={L_MAX}，page={PAGE} token")
    print(f"    连续预分配: {cont} 槽位（浪费 {cont - need}，占 {100 * (cont - need) / cont:.0f}%）"
          f"——不管实际多长，一律按最长预留")
    print(f"    按页分配  : {paged} 槽位（浪费 {paged - need}，≤ 每请求 {PAGE - 1}）"
          f"——用多少发多少")
    budget = paged
    print(f"    显存预算 = {budget} 槽位时：paged 正好容纳 8 条并发；"
          f"contiguous 只能放 {budget // L_MAX} 条 → batch 减半，吞吐直接减半（回看 [2]）")
    assert cont == len(lengths) * L_MAX and paged < cont
    sweep = [(p, sum(math.ceil(l / p) * p for l in lengths)) for p in (1, 2, 4, 8, 16)]
    print(f"    页大小扫描 (page → 槽位): " + " | ".join(f"{p}:{s}" for p, s in sweep))
    assert sweep[-1][1] == cont, "边界：page = L_MAX 时 paged 退化为 contiguous"
    assert sweep[0][1] == need, "边界：page = 1 时零浪费，但 block table 条目最多"
    print(f"    → page 太大 = 内部碎片回潮；page = 1 = 元数据与碎片化访存开销"
          f"（vLLM 默认 block_size=16 token [TODO: verify source]）")

    print("\n" + "=" * 64)
    print("✅ self-check passed: KV cache 计算账 / 吞吐随 batch 单调 /")
    print("   continuous 全面优于 static（且长度对齐时持平）/ paging 省显存且边界正确")
    print("=" * 64)
    print("\ntakeaway: vLLM 的吞吐 = continuous batching（调度不空转）× PagedAttention")
    print("          （显存不碎片）→ batch 塞满 → 摊薄权重读取。SGLang 在此之上加前缀复用")
    print("          （RadixAttention）与结构化生成调度 [TODO: verify source]。")
    print("          L1 用真实 vLLM 跑小模型，把这份代价模型对到真实 tokens/s。")


if __name__ == "__main__":
    main()
