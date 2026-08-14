#!/usr/bin/env python3
"""
nano-slime L2 — 引擎代价模型 × 同步/异步双 regime：update_weights_interval 是 staleness 旋钮，不是吞吐旋钮

L0 用模拟常数量化了解耦的结构（buffer 容量 vs staleness），L1 在真实小模型上实测了
G/T/S。L2 再向真实系统走一步，把 slime 源码里两条训练主循环的控制流变成同一个
模拟器里的两个 regime：
  - train.py（同步）：rollout_data_ref = ray.get(rollout_manager.generate.remote(...))
    阻塞等生成（train.py:L53），每 rollout 末尾 actor_model.update_weights()（:L85）；
  - train_async.py（1-step 异步）：循环前预取第 0 批（:L32），每轮先取上批（:L36）、
    立刻提前发起下一批（:L40），训练后按 (rollout_id+1) % update_weights_interval == 0
    的门（:L66）决定要不要推权重——推之前必须等在途生成完成（:L67-68 注释
    "sync generate before update weights to prevent update weight in the middle of
    generation"），因为引擎不能一边生成一边换权重。
同时把「生成时间 G」从一个常数升级成一个代价模型（借用 nano-vllm-sglang L0 的
iter_time = W_READ + B×KV_STEP，见该模块 L0_kv_cache_batching.py:L26-40）：引擎每个
decode 迭代读一遍全部权重（与 batch 无关），batch 里每条序列只追加 KV 代价——
所以 G 是「引擎怎么被喂」的函数，这是 L0/L1 都没有的维度。

三个定量结论（全部由本文件的模拟器产出，tutorial 逐条对照源码）：
  1. 1-step 异步的加速上界是 2x，且在 G≈T 处取到——它每轮只是把 min(G,T) 藏进
     重叠里；regime 越偏（G≫T 或 T≫G），收益越小。L1 实测的 G/T=2.3 落在
     「生成主导」侧，异步只值 ~1.4x。
  2. update_weights_interval=k 把 max staleness 钉在 k（结构性上界），稳态吞吐
     收益只有权重推送 S 的摊薄（每步 G+S → G+S/k）——它是 staleness 旋钮，
     不是吞吐旋钮（确定性稳态下）。
  3. 架构的第一性原理是让引擎忙：同步 regime 引擎利用率 = G/(G+T+S)，异步把它
     拉向 1。这也是 fully_async_rollout.py 背压设计（:L85-89/L148-152）的动机。

诚实口径：本文件是**可运行的本质模拟**（ROADMAP §三 L2 契约）——本机没有 GPU，
跑不了真实 SGLang/vLLM；真实引擎的 continuous batching、KV 管理、多引擎分片、
delta weight sync 均未建模。建模的是 slime 源码坐实的两个结构事实：两种训练循环
的控制流（行号见上，快照溯源见 tutorial §10）与引擎「权重读被 batch 摊薄」的
代价规律（nano-vllm-sglang L0 已独立讲透）。真机数字走 GPU 通道
[TODO: verify on real system]。
依赖：零（纯标准库）。CPU 瞬时（<0.1s）。
"""

import hashlib
import math
import time

# ---------------- 引擎代价模型（nano-vllm-sglang L0 同款常数） ----------------
# 溯源：tutorial/material/03-data-distributed-rsi/nano-vllm-sglang/L0_kv_cache_batching.py:L26-27
W_READ = 1.0      # 每个 decode 迭代：读一遍全部权重（memory-bandwidth-bound，与 batch 无关）
KV_STEP = 0.02    # 每个 decode 迭代：batch 里每条序列追加的 KV 读取 + attention 代价

# ---------------- workload 配置（与 L1 实测同形状） ----------------
N_ROLLOUTS = 16   # 一个 rollout batch 的条数（L1 实测批 16×L=128，tutorial_L1.md §5）
L_RESP = 128      # 每条 response 的 token 数
B = 12            # 总 rollout 批数（与 L0/L1 一致）
B_ENGINE = 16     # 引擎并发：一个 batch 的 16 条序列一次全进引擎（最优喂法）


def iter_time(b):
    """一个 decode 迭代的引擎代价（nano-vllm-sglang L0:L38-40 同款）。"""
    return W_READ + b * KV_STEP


def gen_time(b, n=N_ROLLOUTS, l=L_RESP):
    """引擎生成一批 rollout 的时间：n 条序列按并发 b 分 ceil(n/b) 波，每波 l 个迭代。"""
    return math.ceil(n / b) * l * iter_time(b)


# 训练一步与权重推送的代价（toy 时间单位，与引擎代价同纲）。
# T 取 L1 实测的比例：同批（16×L=128）G/T = 2.3（tutorial_L1.md §5 [3]）。
G16 = gen_time(B_ENGINE)
T = round(G16 / 2.3, 2)      # = 73.46
S = round(0.1 * T, 2)        # 一次权重推送 = 10% T（真实大规模 S 更大，见 [5]）


# ---------------- regime 1：同步（train.py 语义） ----------------

def sim_sync(B, G, T, S):
    """train.py：每 rollout 阻塞等生成（L53）→ 训练 → update_weights（L85）。
    循环前的首次 update_weights（train.py:L27）两边都不计（两 regime 同一起点）。"""
    cycle = G + T + S
    return {
        "makespan": B * cycle,
        "per_step": cycle,
        "engine_util": G / cycle,
        "trainer_util": T / cycle,
        "staleness": [0] * B,          # 每批都用刚推过的权重生成
        "pushes": B,
    }


# ---------------- regime 2：1-step 异步（train_async.py 语义） ----------------

def sim_async(B, G, T, S, interval, g_list=None):
    """train_async.py 的离散事件模拟（控制流逐行对应，见 docstring 行号）：
      - 循环前：发起生成 0（L32）；
      - 轮 i：等批 i（L36；若上门槛时已在手则免等）→ 立刻发起生成 i+1（L40）
        → 训练批 i（时长 T）→ 若 (i+1) % interval == 0（L66 门）：
          等在途生成 i+1 完成（L68）再推权重（L70，时长 S）。
    引擎 FIFO、一次跑一批；推送窗口内引擎静默（L67 注释：生成中途不换权重）。
    结构性保证（模拟正确性关键）：任一生成批要么在某个推送窗口开始前跑完
    （门槛等的就是它），要么在窗口结束后才开始（发起时刻必见 push_end）——
    不存在「生成跑一半被换权重」，与源码语义一致。
    staleness 定义承 L0：消费时刻的 trainer 版本 − 生成时引擎上的权重版本。"""
    gs = g_list if g_list is not None else [G] * B
    engine_free = 0.0        # 引擎下次空闲时刻
    push_end = 0.0           # 最近一次推送窗口结束时刻（引擎静默期）
    engine_busy = 0.0
    eng_version = 0          # 引擎上的权重版本
    gen_done, gen_ver = [], []
    train_end = 0.0
    stale, pushes = [], 0

    def issue(i, at):
        nonlocal engine_free, engine_busy
        start = max(at, engine_free, push_end)
        gen_ver.append(eng_version)    # 生成所用权重 = 开跑时引擎上的版本
        done = start + gs[i]
        engine_free = done
        engine_busy += gs[i]
        gen_done.append(done)

    issue(0, 0.0)                                   # train_async.py:L32
    for i in range(B):
        start = max(gen_done[i], train_end, push_end)   # L36（可能已被门槛提前等回）
        if i + 1 < B:
            issue(i + 1, start)                          # L40：start the next rollout early
        train_end = start + T
        stale.append(i - gen_ver[i])                     # 消费时 trainer 版本 = i
        if (i + 1) % interval == 0 and i + 1 < B:        # L66 门（末批后的推送不计 makespan）
            barrier = max(train_end, gen_done[i + 1])    # L68：等在途生成
            push_end = barrier + S                       # L70：update_weights
            eng_version = i + 1
            pushes += 1

    mk = train_end
    return {
        "makespan": mk,
        "per_step": mk / B,
        "engine_util": engine_busy / mk,
        "trainer_util": B * T / mk,
        "staleness": stale,
        "pushes": pushes,
    }


def mean(xs):
    return sum(xs) / len(xs)


def main():
    t0 = time.perf_counter()
    print("=" * 68)
    print("nano-slime L2 — 引擎代价模型 × 同步/异步双 regime（本质模拟）")
    print("=" * 68)
    print(f"workload: N={N_ROLLOUTS} 条/批 × L={L_RESP} | B={B} 批 | "
          f"T={T}（= G/2.3，L1 实测同比例）| S={S}（= 0.1·T）")
    print("口径: 可运行本质模拟——引擎代价模型借 nano-vllm-sglang L0，控制流逐行")
    print("      对应 slime train.py / train_async.py（行号快照见 tutorial §10）。")

    # ---------- [1] 引擎代价模型：G 是「怎么喂引擎」的函数 ----------
    print(f"\n[1] 引擎代价模型: iter_time(b) = {W_READ} + b×{KV_STEP}（读权重 + KV 增量）")
    print(f"    {'并发 b':>6} | {'G(一批) ':>9} | {'单条 G':>8} | {'引擎吞吐':>9}")
    for b in (1, 2, 4, 8, 16):
        g = gen_time(b)
        tps = b / iter_time(b)                       # tokens/单位时间（全批合计）
        print(f"    {b:>6} | {g:9.2f} | {g / N_ROLLOUTS:8.2f} | {tps:9.1f}")
    g1, g16 = gen_time(1), gen_time(16)
    floor = L_RESP * KV_STEP                         # b→∞ 时单条 G 的下界
    print(f"    b=1→16 压缩 {g1 / g16:.1f}x；b→∞ 单条下界 = L×KV_STEP = {floor:.2f}"
          f"（吞吐上界 1/KV_STEP = {1 / KV_STEP:.0f} tokens/单位时间）")
    print(f"    → 引擎 batching 是第一杠杆（L1 CPU 实测同向 2.6x，机理见 nano-vllm-sglang L0）")

    # ---------- [2] 双 regime：train.py vs train_async.py ----------
    G = G16
    rs = sim_sync(B, G, T, S)
    ra = sim_async(B, G, T, S, interval=1)
    print(f"\n[2] 双 regime（G={G:.2f}，生成主导，G/T={G / T:.1f}）")
    print(f"    同步  train.py      : makespan {rs['makespan']:8.2f} | 每步 {rs['per_step']:7.2f}"
          f" | 引擎 {rs['engine_util']:.1%} | trainer {rs['trainer_util']:.1%}")
    print(f"    异步  train_async.py: makespan {ra['makespan']:8.2f} | 每步 {ra['per_step']:7.2f}"
          f" | 引擎 {ra['engine_util']:.1%} | trainer {ra['trainer_util']:.1%}")
    sp = rs["makespan"] / ra["makespan"]
    print(f"    加速 {sp:.2f}x | 异步 staleness: mean={mean(ra['staleness']):.2f} "
          f"max={max(ra['staleness'])} | 推送 {ra['pushes']} 次（同步 {rs['pushes']} 次）")
    print(f"    异步每步 = max(G,T)+S = {max(G, T) + S:.2f}（稳态闭式；含首尾瞬态 {ra['per_step']:.2f}）")
    print(f"    同步把 trainer 闲在生成里（利用率 {rs['trainer_util']:.1%}）；"
          f"异步把训练藏进生成影子（引擎 {ra['engine_util']:.1%}）")

    # ---------- [3] update_weights_interval 扫描：staleness 旋钮，不是吞吐旋钮 ----------
    print(f"\n[3] update_weights_interval 扫描（arguments.py:L537-540 默认 1）")
    print(f"    {'k':>2} | {'每步':>7} | {'vs 同步':>7} | {'mean stale':>10} | {'max stale':>9} | 推送")
    per_k = {}
    for k in (1, 2, 4, 8):
        r = sim_async(B, G, T, S, k)
        per_k[k] = r["per_step"]
        print(f"    {k:>2} | {r['per_step']:7.2f} | {rs['makespan'] / r['makespan']:6.2f}x"
              f" | {mean(r['staleness']):10.2f} | {max(r['staleness']):9d} | {r['pushes']}")
    gain = (per_k[1] - per_k[8]) / per_k[1]
    print(f"    k=1→8 每步只省 {gain:.1%}（= S 的摊薄：G+S → G+S/8），max staleness 却 1→8")
    print(f"    → 稳态下 interval 是 staleness 旋钮；吞吐瓶颈始终是 max(G,T)")
    # 闭式核验（生成主导区）：引擎背靠背连跑，只有推送窗口让它空闲，
    # 故 makespan(k) = B·G + floor((B-1)/k)·S + T 应精确成立（事件模拟 vs 闭式）。
    for k in (1, 2, 4, 8):
        r = sim_async(B, G, T, S, k)
        closed = B * G + ((B - 1) // k) * S + T
        assert abs(r["makespan"] - closed) < 1e-6, \
            f"事件模拟应精确等于闭式（k={k}）: {r['makespan']:.6f} vs {closed:.6f}"

    # ---------- [4] regime 迁移：引擎变快时，异步的价值先升后降 ----------
    print(f"\n[4] regime 迁移: 引擎提速 f 倍（G' = G/f），异步(k=1) 相对同步的加速")
    h_gp, h_r = "G'", "G'/T"
    print(f"    {'f':>4} | {h_gp:>7} | {h_r:>5} | {'加速':>6} | {'引擎利用(异)':>12}")
    sps = {}
    for f in (4.0, 2.0, 1.0, 0.5, 0.25):
        g = G / f
        r_s = sim_sync(B, g, T, S)
        r_a = sim_async(B, g, T, S, 1)
        sps[f] = r_s["makespan"] / r_a["makespan"]
        print(f"    {f:>4} | {g:7.2f} | {g / T:5.2f} | {sps[f]:5.2f}x | {r_a['engine_util']:12.1%}")
    f_peak = max(sps, key=sps.get)
    print(f"    峰值在 f={f_peak}（G'={G / f_peak:.1f} ≈ T={T}）：1-step 异步每轮藏掉 min(G,T)，")
    print(f"    上界 2x 只在 G≈T 取到；G≫T（RLVR 常态）或 T≫G 时两头都趋 1")

    # ---------- [5] S 的量级：interval 的摊薄收益何时变大 ----------
    print(f"\n[5] S 量级敏感性（真实大规模全量权重跨节点推送可达 T 同量级）")
    print(f"    {'S':>6} | {'k=1 每步':>9} | {'k=8 每步':>9} | {'k=8 省':>7}")
    saves = []
    for s_frac in (0.1, 0.5, 1.0):
        s = round(s_frac * T, 2)
        p1 = sim_async(B, G, T, s, 1)["per_step"]
        p8 = sim_async(B, G, T, s, 8)["per_step"]
        saves.append((p1 - p8) / p1)
        print(f"    {s:>6.2f} | {p1:9.2f} | {p8:9.2f} | {saves[-1]:6.1%}")
    assert all(x > 0 for x in saves) and saves[0] < saves[1] < saves[2], \
        "S 越大，interval 摊薄收益应越大且单调"
    print(f"    → S 越大，interval 的吞吐收益越大（{saves[0]:.1%} → {saves[2]:.1%}）；"
          f"slime 的 delta weight sync（L3 主题）")
    print(f"      从另一头修这件事：把 S 本身变小，而不是忍受陈旧权重")

    # ---------- [6] self-check ----------
    print(f"\n[6] self-check")
    assert all(x >= 0 for x in ra["staleness"]), "staleness 不能为负"
    assert max(ra["staleness"]) <= 1, "interval=1 时 staleness 上界应为 1"
    print(f"    ✓ interval=1: staleness ∈ [0,1]（首批 0，其后 1——1-step off-policy）")
    for k in (2, 4, 8):
        r = sim_async(B, G, T, S, k)
        assert max(r["staleness"]) == k, f"interval={k} 时 max staleness 应恰为 {k}"
    print(f"    ✓ interval=k: max staleness == k（结构性上界，非测量拟合）")
    assert per_k[1] > per_k[2] > per_k[4] > per_k[8], "每步耗时应随 k 单调不增（S 摊薄）"
    assert gain < 0.05, "生成主导区 interval 的吞吐收益应很小（<5%）"
    print(f"    ✓ interval 吞吐收益单调且微小（{gain:.1%} < 5%）——旋钮定性正确")
    assert all(sps[f] < 2.0 for f in sps), "1-step 异步加速不应突破 2x 上界"
    assert f_peak in (2.0, 4.0), "峰值应落在 G'≈T 附近（f=2 → G/2，f=4 → G/4，夹住 T=G/2.3）"
    print(f"    ✓ 加速上界 2x 未突破（max {max(sps.values()):.2f}x），峰值在 G'≈T 处")
    steady = max(G, T) + S
    tail = ra["per_step"] - steady
    assert abs(tail - (T - S) / B) < 1e-9, "瞬态残差应精确 = (T−S)/B（首批等生成、末批省推送）"
    print(f"    ✓ 异步每步 vs 闭式 max(G,T)+S={steady:.2f}：瞬态残差 {tail:+.4f}"
          f" = (T−S)/B = {(T - S) / B:+.4f}（精确吻合）")
    assert rs["engine_util"] < ra["engine_util"], "异步应提高引擎利用率"
    print(f"    ✓ 引擎利用率：同步 {rs['engine_util']:.1%} → 异步 {ra['engine_util']:.1%}"
          f"（架构第一性 = 让引擎忙）")

    digest_src = "|".join([
        f"G16={G16:.2f}", f"T={T}", f"S={S}",
        f"sync_mk={rs['makespan']:.2f}", f"async_mk={ra['makespan']:.2f}",
        *[f"per_k{k}={per_k[k]:.2f}" for k in (1, 2, 4, 8)],
        *[f"sp_f{f}={sps[f]:.4f}" for f in (4.0, 2.0, 1.0, 0.5, 0.25)],
    ])
    digest = hashlib.md5(digest_src.encode()).hexdigest()
    print(f"\n    digest(metrics) = {digest}")

    print("\n" + "=" * 68)
    print("✅ self-check passed: staleness 上界 / interval 单调 / 2x 天花板 / 闭式吻合")
    print("=" * 68)
    print(f"\ntakeaway: 1-step 异步（train_async.py）每轮把 min(G,T) 藏进重叠——上界 2x、")
    print(f"          峰值在 G≈T；RLVR 的生成主导区（L1 实测 G/T=2.3）只值 {sp:.2f}x。")
    print(f"          update_weights_interval 把 max staleness 钉在 k、稳态只赚 S 摊薄，")
    print(f"          是 staleness 旋钮不是吞吐旋钮；真吞吐靠把 S 做小（delta sync，L3）")
    print(f"          或把引擎喂满（continuous batching，fully_async_rollout.py 背压设计）。")
    print(f"          真机验证（SGLang 引擎 + Megatron 权重同步）[TODO: verify on real system]")
    print(f"elapsed: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
