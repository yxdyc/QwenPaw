#!/usr/bin/env python3
"""
nano-slime L0 — 采样/训练解耦：data buffer + 版本化权重 + staleness 账本

零外部依赖（纯标准库），CPU 即跑，离散事件模拟、无随机、输出完全确定。

RL 后训练每一轮有两块大开销：rollout 一批轨迹（generate）与训练一步（train）。
lockstep（同步）让快的一方空等慢的一方；slime 的答案（README「Architecture
Overview」三模块，main 分支 2026-08-04 快照）：
  - training (Megatron)：从 Data Buffer 读数据，训练后把参数同步给 rollout；
  - rollout (SGLang + router)：生成数据存入 Data Buffer；
  - data buffer：连接两者的桥接模块。
本 toy 把这条数据通路抽成三件事：
  1. 解耦：generator 与 trainer 各跑各的，用 FIFO buffer 连接；
  2. 版本化权重：trainer 每完成一步 version+1，generator 开新批次前拉新版本；
  3. staleness：样本进 loss 时，version(消费时刻) - version(生成时刻) = 它过时了几步。
buffer 容量 C 是核心旋钮：它限住 staleness 的上界，但不改变稳态 makespan
（稳态由较慢的一方决定）——这个「买不到吞吐、只买到弹性与 off-policy 度」
的结论就是 buffer 尺寸权衡的本质。

toy 口径：G/T/S 是模拟时间常数（不是真机测量），目标是把权衡的结构量化；
真实吞吐与 slime 源码级对照留到 L1/L2/L3。
"""

from collections import deque

# ---------------- 模拟配置（toy 时间单位） ----------------
G, T, S, B = 4.0, 6.0, 1.0, 12   # 生成一批 / 训练一步 / 权重同步 / 总批数

# ---------------- [1] lockstep 基线 ----------------

def sim_lockstep(B, G, T, S):
    """同步：generate -> sync -> train 严格串行，每批一个 cycle。"""
    cycle = G + S + T
    return B * cycle, G / cycle, T / cycle    # makespan, gen 利用率, trainer 利用率

# ---------------- [2] 解耦：generator -> buffer(C) -> trainer ----------------

def sim_decoupled(B, G, T, S, C, g_list=None):
    """离散事件模拟。g_list 给每批不同的生成时间（确定性的「波动」）。
    返回 (makespan, gen利用率, trainer利用率, staleness列表, 消费事件, 同步次数)。
    规则（与 slime 数据通路同构）：
      - trainer 完成一步 => version+1；空闲且 buffer 非空就取最旧样本开训；
      - generator 开新批前，若手中版本落后于 version，花 S 拉新权重（计入忙碌）；
      - 背压：buffer 就绪数 + 在途批数 >= C 时 generator 停手。"""
    INF = float("inf")
    gs = g_list if g_list is not None else [G] * B
    t = 0.0
    version = gen_version = 0
    gen_until = None            # (完成时刻, 该批所用版本)
    trn_until = None            # 训练完成时刻
    buf, in_flight = deque(), 0
    started = consumed = syncs = 0
    staleness, events = [], []
    gen_busy = trn_busy = 0.0

    while consumed < B:
        # 先在当前时刻 t 尝试「启动」动作，再推进到下一个完成事件
        if trn_until is None and buf:                        # trainer 从 buffer 取批
            v = buf.popleft()
            staleness.append(version - v)
            events.append((consumed, t, v, version - v))
            trn_until = t + T
            trn_busy += T
        if gen_until is None and started < B and len(buf) + in_flight < C:
            cost = 0.0                                       # generator 开新批
            if gen_version != version:
                gen_version, syncs, cost = version, syncs + 1, S
            gen_until = (t + cost + gs[started], gen_version)
            in_flight += 1
            gen_busy += cost + gs[started]
            started += 1
        nxt = min(gen_until[0] if gen_until else INF, trn_until if trn_until else INF)
        assert nxt < INF, "deadlock: 无事件可推进但未完成"
        t = nxt
        if gen_until is not None and t == gen_until[0]:      # 一批 rollout 就绪
            buf.append(gen_until[1])
            in_flight -= 1
            gen_until = None
        if trn_until is not None and t == trn_until:         # 一步训练完成
            version += 1
            consumed += 1
            trn_until = None

    assert started == B and len(staleness) == B
    assert all(s >= 0 for s in staleness), "staleness 不能为负"
    return t, gen_busy / t, trn_busy / t, staleness, events, syncs

def mean(xs):
    return sum(xs) / len(xs)

# ---------------- 实验 ----------------

def main():
    print("=" * 64)
    print("nano-slime L0 — 采样/训练解耦：data buffer + 版本化权重")
    print("=" * 64)
    print(f"\n配置（toy 时间单位）: G={G} 生成一批 | T={T} 训练一步 | S={S} 权重同步 | B={B} 批")
    print("为何生成是大头：decode 每个 token 都是一次串行前向，G 随 response 长度线性涨；")
    print("训练一步只对定量数据做常数次前向/反向。长 response 的 RLVR 里常是 generate 主导。")

    # [1] lockstep：快的一方空等慢的一方
    mk_l, gu_l, tu_l = sim_lockstep(B, G, T, S)
    print(f"\n[1] lockstep（同步串行）: cycle = G+S+T = {G + S + T:.0f}")
    print(f"    makespan = {mk_l:.0f} | gen 利用率 {gu_l:.1%} | trainer 利用率 {tu_l:.1%}")
    print(f"    —— 每个 cycle 有 {1 - gu_l - tu_l:.1%} 的时间两边都在空转（sync + 互等）")

    # [2] 解耦（C=4）：buffer 连接，版本化权重，staleness 记账
    C0 = 4
    mk_d, gu_d, tu_d, stale, events, syncs = sim_decoupled(B, G, T, S, C0)
    print(f"\n[2] 解耦（buffer C={C0}）: makespan = {mk_d:.0f} | speedup = {mk_l / mk_d:.2f}x")
    print(f"    gen 利用率 {gu_d:.1%} | trainer 利用率 {tu_d:.1%} | 权重同步 {syncs} 次")
    print(f"    消费事件（批#, 时刻, 生成时版本, staleness）:")
    for i, (n, tt, v, s) in enumerate(events):
        print(f"      #{n:2d} t={tt:5.1f} v_gen={v} staleness={s}" + ("  ..." if i == 5 else ""))
        if i == 5:
            break
    print(f"    staleness: mean={mean(stale):.2f} max={max(stale)} —— buffer 里的样本是旧权重生的")

    # [3] buffer 容量扫描：吞吐买不到，off-policy 度随 C 涨
    print(f"\n[3] buffer 容量扫描（稳态由较慢一方决定）:")
    print(f"    {'C':>2} | {'makespan':>8} | {'mean stale':>10} | {'max stale':>9}")
    prev_mean = -1.0
    for C in (1, 2, 4, 8):
        mk, _, _, st, _, _ = sim_decoupled(B, G, T, S, C)
        print(f"    {C:>2} | {mk:8.0f} | {mean(st):10.2f} | {max(st):9d}")
        assert mean(st) >= prev_mean - 1e-9, "平均 staleness 应随 C 不减"
        prev_mean = mean(st)
    mk1, _, _, st1, _, _ = sim_decoupled(B, G, T, S, 1)
    assert max(st1) <= 1, "C=1 时最多只领先 trainer 一步"
    print(f"    C=1 把 staleness 钳在 ≤1（背压即 off-policy 闸门）；makespan 几乎不动")

    # [4] 反例
    print(f"\n[4] 反例：解耦不是万灵药")
    mk_g, _, _, st_g, _, _ = sim_decoupled(B, 20.0, 1.0, 1.0, 8)
    mk_gl, _, _ = sim_lockstep(B, 20.0, 1.0, 1.0)
    print(f"    a) 生成才是瓶颈（G=20, T=1）: lockstep {mk_gl:.0f} -> 解耦 {mk_g:.0f}"
          f"（speedup {mk_gl / mk_g:.2f}x，几乎白忙）")
    print(f"       => 该让生成引擎本身更快（SGLang/vLLM，见 nano-vllm-sglang L0），而不是加 buffer")
    assert mk_gl / mk_g < 1.2, "生成瓶颈下解耦收益应很小"
    gs_var = [2.0, 4.0, 9.0] * (B // 3)          # 确定性的批时长波动（均值 5，仍 < T=6）
    mk_v1, _, _, st_v1, _, _ = sim_decoupled(B, G, T, S, 1, g_list=gs_var)
    mk_v8, _, _, st_v8, _, _ = sim_decoupled(B, G, T, S, 8, g_list=gs_var)
    print(f"    b) 批时长波动 [2,4,9]: C=1 makespan {mk_v1:.0f}（被慢批卡住）"
          f" vs C=8 makespan {mk_v8:.0f}（buffer 吸收波动）")
    print(f"       弹性才是大 buffer 的真实收益；代价是 staleness {mean(st_v1):.2f} -> {mean(st_v8):.2f}")
    assert mk_v1 >= mk_v8, "波动下大 buffer 不应更慢"

    print("\n" + "=" * 64)
    print("✅ self-check passed: 事件推进无死锁 / staleness 非负 / C=1 上界 / 容量单调 / 两个反例")
    print("=" * 64)
    print(f"\ntakeaway: 解耦把每批耗时从 G+S+T 压到 max(较慢一方)，代价是样本带 staleness；")
    print(f"          buffer 容量限 staleness 上界、吸收波动，但不改稳态吞吐。")
    print(f"          真实 slime 里 trainer=Megatron、rollout=SGLang、中间就是 Data Buffer；")
    print(f"          IS ratio 可修正同一状态上的动作分布差异，但不能抹掉陈旧前缀；见 nano-verl L1。")

if __name__ == "__main__":
    main()
