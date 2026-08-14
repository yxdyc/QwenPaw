# nano-slime · L2 教程：引擎代价模型 × 同步/异步双 regime——interval 是 staleness 旋钮，不是吞吐旋钮

> L0 用模拟常数量化了解耦的结构（buffer 容量 vs staleness），L1 在真实小模型上
> 把 G/T/S 测了出来（G∝L、batching 压缩 2.6x、同批 G/T=2.3）。L2 再向真实系统走一步：
> **把 slime 源码里两条训练主循环的控制流变成同一个模拟器里的两个 regime**，
> 并把 G 从一个常数升级成引擎代价模型——然后回答三个定量问题：
> 异步到底省了多少？`update_weights_interval` 这个旋钮拧的是什么？引擎变快时这套架构的价值怎么变？
> 前置：先跑过 [L0](tutorial_L0.md) 与 [L1](tutorial_L1.md)。对应实现：[L2_engine_cost_async_regimes.py](L2_engine_cost_async_regimes.py)。

---

## 1. 运行与输出

**可运行性契约声明（ROADMAP §三）**：本节是 **L2 允许的本质模拟**——本机没有 GPU，
跑不了真实 SGLang/vLLM 引擎。模拟核心（离散事件模拟器 + 引擎代价模型）本身可运行、
零依赖、输出完全确定；它建模的两个结构事实都有 slime 源码背书（§4/§10 逐行给锚点），
未建模的部分在 §13 逐条列清。真机（SGLang 引擎 + Megatron 权重同步）验证标
`[TODO: verify on real system]`，走 GPU 通道，不在写作轮内执行。

跑法（零依赖，纯标准库，任意 CWD，<0.1s）：

```bash
$ python3 -B L2_engine_cost_async_regimes.py
```

真实输出（2026-08-14；elapsed 行按既有口径 `sed '/^[[:space:]]*elapsed/d'` 掩码——
raw 72 行 → 掩码 71 行。4 跑 × 2 新建空独立 CWD 全 EXIT=0、stderr 0 B、
掩码输出 BYTE-IDENTICAL，掩码锚 `43a1a9ed02ad3eaaac1cf39e54d155c5`/71 行；
脚本自产 digest `d3c6be2606446b73cdb84cce0fbfabf1`）：

```text
====================================================================
nano-slime L2 — 引擎代价模型 × 同步/异步双 regime（本质模拟）
====================================================================
workload: N=16 条/批 × L=128 | B=12 批 | T=73.46（= G/2.3，L1 实测同比例）| S=7.35（= 0.1·T）
口径: 可运行本质模拟——引擎代价模型借 nano-vllm-sglang L0，控制流逐行
      对应 slime train.py / train_async.py（行号快照见 tutorial §10）。

[1] 引擎代价模型: iter_time(b) = 1.0 + b×0.02（读权重 + KV 增量）
      并发 b |    G(一批)  |     单条 G |      引擎吞吐
         1 |   2088.96 |   130.56 |       1.0
         2 |   1064.96 |    66.56 |       1.9
         4 |    552.96 |    34.56 |       3.7
         8 |    296.96 |    18.56 |       6.9
        16 |    168.96 |    10.56 |      12.1
    b=1→16 压缩 12.4x；b→∞ 单条下界 = L×KV_STEP = 2.56（吞吐上界 1/KV_STEP = 50 tokens/单位时间）
    → 引擎 batching 是第一杠杆（L1 CPU 实测同向 2.6x，机理见 nano-vllm-sglang L0）

[2] 双 regime（G=168.96，生成主导，G/T=2.3）
    同步  train.py      : makespan  2997.24 | 每步  249.77 | 引擎 67.6% | trainer 29.4%
    异步  train_async.py: makespan  2181.83 | 每步  181.82 | 引擎 92.9% | trainer 40.4%
    加速 1.37x | 异步 staleness: mean=0.92 max=1 | 推送 11 次（同步 12 次）
    异步每步 = max(G,T)+S = 176.31（稳态闭式；含首尾瞬态 181.82）
    同步把 trainer 闲在生成里（利用率 29.4%）；异步把训练藏进生成影子（引擎 92.9%）

[3] update_weights_interval 扫描（arguments.py:L537-540 默认 1）
     k |      每步 |   vs 同步 | mean stale | max stale | 推送
     1 |  181.82 |   1.37x |       0.92 |         1 | 11
     2 |  178.14 |   1.40x |       1.33 |         2 | 5
     4 |  176.31 |   1.42x |       2.17 |         4 | 2
     8 |  175.69 |   1.42x |       3.50 |         8 | 1
    k=1→8 每步只省 3.4%（= S 的摊薄：G+S → G+S/8），max staleness 却 1→8
    → 稳态下 interval 是 staleness 旋钮；吞吐瓶颈始终是 max(G,T)

[4] regime 迁移: 引擎提速 f 倍（G' = G/f），异步(k=1) 相对同步的加速
       f |      G' |  G'/T |     加速 |      引擎利用(异)
     4.0 |   42.24 |  0.58 |  1.47x |        50.5%
     2.0 |   84.48 |  1.15 |  1.70x |        86.8%
     1.0 |  168.96 |  2.30 |  1.37x |        92.9%
     0.5 |  337.92 |  4.60 |  1.19x |        96.3%
    0.25 |  675.84 |  9.20 |  1.10x |        98.1%
    峰值在 f=2.0（G'=84.5 ≈ T=73.46）：1-step 异步每轮藏掉 min(G,T)，
    上界 2x 只在 G≈T 取到；G≫T（RLVR 常态）或 T≫G 时两头都趋 1

[5] S 量级敏感性（真实大规模全量权重跨节点推送可达 T 同量级）
         S |    k=1 每步 |    k=8 每步 |   k=8 省
      7.35 |    181.82 |    175.69 |   3.4%
     36.73 |    208.75 |    178.14 |  14.7%
     73.46 |    242.42 |    181.20 |  25.3%
    → S 越大，interval 的吞吐收益越大（3.4% → 25.3%）；slime 的 delta weight sync（L3 主题）
      从另一头修这件事：把 S 本身变小，而不是忍受陈旧权重

[6] self-check
    ✓ interval=1: staleness ∈ [0,1]（首批 0，其后 1——1-step off-policy）
    ✓ interval=k: max staleness == k（结构性上界，非测量拟合）
    ✓ interval 吞吐收益单调且微小（3.4% < 5%）——旋钮定性正确
    ✓ 加速上界 2x 未突破（max 1.70x），峰值在 G'≈T 处
    ✓ 异步每步 vs 闭式 max(G,T)+S=176.31：瞬态残差 +5.5092 = (T−S)/B = +5.5092（精确吻合）
    ✓ 引擎利用率：同步 67.6% → 异步 92.9%（架构第一性 = 让引擎忙）

    digest(metrics) = d3c6be2606446b73cdb84cce0fbfabf1

====================================================================
✅ self-check passed: staleness 上界 / interval 单调 / 2x 天花板 / 闭式吻合
====================================================================

takeaway: 1-step 异步（train_async.py）每轮把 min(G,T) 藏进重叠——上界 2x、
          峰值在 G≈T；RLVR 的生成主导区（L1 实测 G/T=2.3）只值 1.37x。
          update_weights_interval 把 max staleness 钉在 k、稳态只赚 S 摊薄，
          是 staleness 旋钮不是吞吐旋钮；真吞吐靠把 S 做小（delta sync，L3）
          或把引擎喂满（continuous batching，fully_async_rollout.py 背压设计）。
          真机验证（SGLang 引擎 + Megatron 权重同步）[TODO: verify on real system]
```

---

## 2. 代码结构

单文件、零依赖（`hashlib`/`math`/`time` 标准库），四块：

1. **引擎代价模型**（`iter_time`/`gen_time`）——借 nano-vllm-sglang L0 的常数
   （`W_READ=1.0`、`KV_STEP=0.02`，溯源 `03-data-distributed-rsi/nano-vllm-sglang/L0_kv_cache_batching.py:L26-27`、
   `iter_time` 同文件 L38-40）。G 从此是「引擎怎么被喂」的函数：`G(b) = ⌈N/b⌉·L·(W_READ + b·KV_STEP)`。
2. **`sim_sync`**——train.py 语义：每 rollout 阻塞等生成 → 训练 → 推权重。
3. **`sim_async`**——train_async.py 语义的离散事件模拟：预取、提前发起、
   `(i+1) % interval == 0` 门、推前等在途生成。控制流逐行对应源码（§4）。
4. **实验 [1]–[6]**——代价模型扫描、双 regime 对照、interval 扫描、regime 迁移、
   S 量级敏感性、self-check（含事件模拟 vs 闭式公式的精确核验）。

workload 与 L1 同形状（16 条/批 × L=128，B=12 批），`T = G/2.3` 取 L1 实测的
同批 G/T 比例（tutorial_L1.md §5 [3]：`G_batch=236 ms vs T=103 ms → G/T = 2.3`）。
注意方向：L1 的 2.3 是 CPU 小模型实测，这里把它作为**模型参数**喂进模拟器——
比例可迁移（生成主导是 RLVR 的结构性事实），绝对值不迁移（§13）。

---

## 3. [1] 引擎代价模型：G 是「怎么喂引擎」的函数

L0/L1 里 G 是一个常数（模拟给定或实测所得）。L2 把它拆开：引擎跑一个 decode 迭代，
代价 = 读一遍全部权重（`W_READ`，memory-bandwidth-bound，与 batch 无关）+ batch 里
每条序列追加的 KV 读取与 attention（`KV_STEP`/条）。这套模型在 nano-vllm-sglang L0
已经独立讲透（为什么 decode 是 memory-bound、为什么吞吐上界是 `1/KV_STEP`），这里直接消费。

输出表里 b=1→16 把一批的 G 从 2088.96 压到 168.96（12.4x）：并发越大，那次
「读权重」被越多序列摊薄。b→∞ 时单条 G 的下界是 `L×KV_STEP = 2.56`——
再大的 batch 也压不掉每条序列自己的 KV 代价。

和 L1 的关系要讲诚实：L1 在 CPU 小模型上实测 batching 压缩 2.6x（tutorial_L1.md §4），
方向相同、量级远小于这里的 12.4x——CPU 小模型的「固定开销」是 Python/内核启动，
不是显存带宽，两个量级不可互推（L1 §1 声明 1 同款口径）。**可迁移的是结构**：
引擎 batching 是第一杠杆，这正是 L1 takeaway「吞吐的第一杠杆是 batching」的引擎侧机理。

---

## 4. [2] 双 regime：slime 两条训练主循环的逐行对照

这是本节的机制核心。slime 仓库根目录有两个训练入口，控制流差异就是「同步 vs 1-step 异步」
的全部本质（以下行号为 2026-08-14 main 分支快照，溯源 §10；引文逐字）：

**train.py（同步）**——循环体三行就是全部：

```python
# train.py:L53
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
# ...（train）...
# train.py:L85
actor_model.update_weights()
```

`ray.get(...)` 阻塞：trainer 干等这批 rollout 生成完；训练完立刻推权重（每 rollout 一推）。
循环前还有一次首推（train.py:L27 `actor_model.update_weights()`）——两 regime 都有，
模拟器里两边都不计（同一起点）。每批都用**刚推过的权重**生成，staleness 恒 0，
代价是每步 `G+T+S` 串行：引擎利用率 67.6%、trainer 利用率 29.4%（[2] 输出）——
trainer 有 70% 的时间在干等生成。

**train_async.py（1-step 异步）**——四个动作把训练藏进生成的影子：

```python
# train_async.py:L32（循环前预取第 0 批）
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # train_async.py:L36（取上批；若已被门槛等回则免等）
    rollout_data_curr_ref = ray.get(rollout_data_next_future)
    # train_async.py:L40（立刻提前发起下一批——重叠的来源）
    rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
    # ...（训练 curr 批）...
    # train_async.py:L66-70（权重更新门 + 推前屏障）
    if release_train or (rollout_id + 1) % args.update_weights_interval == 0:
        # sync generate before update weights to prevent update weight in the middle of generation
        rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
        rollout_data_next_future = None
        actor_model.update_weights()
```

四个要点：

1. **重叠只有一步**：第 i+1 批的生成与第 i 批的训练并行，仅此而已——这是
   「1-step off-policy」的字面含义。不是无限解耦（那是 fully async，§9）。
2. **L67 注释是工程约束的原文**：推权重前必须把在途生成**等完**，因为引擎不能
   一边生成一边换权重。模拟器的对应物是「推送窗口内引擎静默」——这条约束
   决定了异步不是免费的：每次推送都有一段引擎排队等它的时间。
3. **L68 的 walrus 细节**：门槛等回的下一批数据直接赋给 `rollout_data_curr_ref`、
   清空 future——下轮循环 L36 的 `if ... is not None` 因此跳过，数据不丢不重。
   控制流上「等权重」和「等数据」是同一个等待，slime 把它写成了一个表达式。
4. **L11 的断言**：`assert not args.colocate, "Colocation is not supported for async training."`
   ——异步模式要求训练与推理**分卡**（colocate 时 GPU 被训练占着，引擎没法并行跑）。
   这是异步的隐藏成本：用更多硬件换重叠。nano-verl 走的是另一条路
   （colocate + 分时复用，见 nano-verl L0/L3），两者是同一矛盾的两端。

模拟器 `sim_async` 与这段控制流一一对应：`issue(0, 0.0)` = L32；循环里
`start = max(gen_done[i], train_end, push_end)` = L36；`issue(i+1, start)` = L40；
门槛/屏障/推送 = L66-70。staleness 定义承 L0：消费时 trainer 版本 − 生成时引擎权重版本。

**[2] 输出怎么读**：G/T=2.3 的生成主导区里，异步把 makespan 从 2997.24 压到
2181.83（**1.37x**），staleness mean=0.92（首批 0、其后 1）、max=1。
引擎利用率 67.6%→92.9%：异步没有让任何一方「更快」，它只是让引擎几乎一直忙、
让训练搭了生成的顺风车。推送 11 次 vs 同步 12 次：末批之后无需再推
（没有下一批生成要用它）——模拟器按此口径，源码里末轮门槛命中时仍会推
（train_async.py:L66 对 `rollout_id = B-1` 不排除），该推送服务后续 eval/续训，不计训练 makespan。

---

## 5. 闭式：为什么异步每步 = max(G,T) + S

[2] 输出里有一行「异步每步 = max(G,T)+S = 176.31（稳态闭式；含首尾瞬态 181.82）」。
这不是拟合，是可以手推的：

生成主导区（G ≥ T）稳态下，第 i 轮的时间线是：生成 i+1 在引擎上排队连跑
（引擎背靠背，无空隙）→ 训练 i 在生成影子里完成（T < G，训练总是先完）→
门槛命中（interval=1 时每轮命中）：等在途生成完成（它刚好是瓶颈，等待 ≈ 0）→
推权重 S（引擎静默）。**每轮净增 = G + S**；训练时间被完全藏进生成。
T 主导区对称：每轮净增 = T + S（生成被藏进训练）。合并即 `max(G,T) + S`。

瞬态也可精确算：首批要干等第一个生成（G），末批省掉尾随推送（S），
摊到 B 批上，每步比稳态多 `(T−S)/B`。self-check 里这条是**精确等式断言**
（`+5.5092 = (73.46−7.35)/12`，误差 < 1e-9）——事件模拟与闭式互相验证，
模拟器的正确性不靠「看起来合理」背书。

[3] 的闭式核验更狠：生成主导区里
`makespan(k) = B·G + ⌊(B−1)/k⌋·S + T` **精确成立**（引擎总忙碌 B·G、
只有 ⌊(B−1)/k⌋ 个推送窗口让它空闲 S、末尾一步训练 T），四个 k 值
误差 < 1e-6。这条闭式是下面「interval 不是吞吐旋钮」结论的数学骨架。

---

## 6. [3] update_weights_interval：staleness 旋钮，不是吞吐旋钮

`--update-weights-interval`（slime/utils/arguments.py:L536-541，`default=1`，
help 原文 "Interval for updating the weights"）控制每几个 rollout 推一次权重。
直觉上它「应该」是吞吐旋钮：少推权重，少付 S，引擎少静默。**[3] 输出打脸**：

| k | 每步 | vs 同步 | mean stale | max stale |
|---|------|---------|-----------|-----------|
| 1 | 181.82 | 1.37x | 0.92 | 1 |
| 2 | 178.14 | 1.40x | 1.33 | 2 |
| 4 | 176.31 | 1.42x | 2.17 | 4 |
| 8 | 175.69 | 1.42x | 3.50 | 8 |

k=1→8，每步只省 3.4%（闭式一目了然：每步 `G+S → G+S/8`，S=0.1T 太小），
max staleness 却从 1 涨到 8。**吞吐瓶颈始终是 max(G,T)，interval 碰不到它**；
interval 真正拧的是样本的 off-policy 度。

`max staleness == k` 是**结构性上界**（self-check 对 k=2/4/8 精确断言，不是测量）：
两次推送之间引擎权重冻结在版本 v，其间生成、之后消费的所有批次，消费时 trainer
已前进 ≤ k 步，故 staleness ≤ k；而紧跟推送前生成的那一批恰好吃满 k。
L0 里「buffer 容量限 staleness 上界」的结论在这里有了训练侧的对应物：
**同步侧的闸门（interval）与数据侧的闸门（buffer 容量）是同一种东西——
用结构性约束把 off-policy 度钉死，而不是靠算法侧硬扛。**

那 interval>1 什么时候才值？[5] 给了答案的一半：S 涨到 0.5T/1.0T 时，
k=8 的摊薄收益从 3.4% 涨到 14.7%/25.3%——真实大规模里全量权重跨节点推送
完全可达 T 同量级（数十 GB 走 RDMA，见 §9 的 delta sync）。另一半在 L0 已经讲过：
**波动**。稳态确定性模型里 interval 买不到吞吐，但真实 G 有长尾（慢样本、
工具调用、环境交互），大 interval + 深流水能吸收波动——L0 [4]b 的结论原样迁移。

---

## 7. [4] regime 迁移：2x 天花板与峰值位置

把引擎提速 f 倍（G′ = G/f，对应换更强引擎/更大 batch/更多卡），异步(k=1)
相对同步的加速画出一条**先升后降**的曲线（[4] 输出）：

- f=4（G′/T=0.58，训练主导）：1.47x，引擎利用率只有 50.5%——引擎太快，
  一半时间在等训练，异步把生成藏完了也填不满它；
- f=2（G′/T=1.15 ≈ 平衡点）：**1.70x，峰值**；
- f=1（G′/T=2.3，L1 实测区）：1.37x；
- f=0.25（G′/T=9.2，深度生成主导）：1.10x——同步里 trainer 本来就闲，
  异步只是把「闲」从 trainer 挪走，makespan 仍被引擎钉死。

机制一句话：**1-step 异步每轮藏掉的是 min(G,T)**（§5 闭式的直接推论），
所以加速 = `(G+T+S)/(max(G,T)+S)`，上界 2x 当且仅当 G≈T 且 S→0 时取到。
self-check 断言了全扫描不破 2x、峰值落在夹住 G′=T 的 {f=2, f=4} 区间。

这条曲线是判断架构选型的尺子：RLVR 长 response 场景（G/T 2–10，
L1 实测 2.3）里 1-step 异步只值 1.1–1.4x，要更多就得换形态——
要么 fully async（§9，把流水深度从 1 放开），要么把引擎本身做快
（nano-vllm-sglang 的全部主题）。反过来，若任务 response 短、训练重
（G′/T < 1），异步接近 2x，是最划算的区间。

---

## 8. [5] S 的量级：delta weight sync 为什么存在

[5] 把 S 从 0.1T 扫到 1.0T：interval 摊薄收益 3.4% → 25.3%。这解释了
一个看似矛盾的工程事实：slime 一边提供 `--update-weights-interval`
让你忍受陈旧权重，一边又投入大量工程做 **delta weight sync**
（README:L45 特性清单链接 `docs/en/advanced/delta-weight-sync.md`，定义逐字：
"keeps non-colocated rollout engines up to date by shipping only the bytes
that changed between two syncs, instead of a full checkpoint each time"；
disk-transport、xor/overwrite 编码、zstd 压缩；实现细节是 L3 主题）——
因为**把 S 做小**和**少推几次**是同一个问题的两头，
前者不牺牲 on-policy 度。actor.py 里还有一处 interval 的消费点值得注意：
`keep_old_actor` 且 `update_weights_interval == 1` 时，slime 维护一个
队列式的三份权重轮转（actor.py:L640-645，日志原文
"updating model queue: rollout_actor -> old_actor, actor -> rollout_actor"）——
rollout 用的旧版权重被显式保留下来给 old-logprob 计算。这是框架层面的自供：
**staleness-1 是被当真处理的（IS 修正需要旧策略的 logprob），不是被忽略的。**
算法侧的修正（importance sampling）见 nano-verl L1。

---

## 9. 权威实现取舍表：nano 版没做什么

| 维度 | nano-slime L2（本文件） | slime 真实实现 | 差异原因 |
|------|------------------------|----------------|----------|
| 引擎 | 代价模型 `iter_time(b)`，FIFO 一次一批 | SGLang 引擎 + router：trainer 不跑引擎，向 HTTP endpoint 发请求（sglang_rollout.py:L158 `url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"`），continuous batching、KV cache 管理、多引擎分片 | 本机无 GPU；引擎内部机制已由 nano-vllm-sglang L0–L3 独立阶梯覆盖 |
| 异步形态 | 只有 1-step（train_async.py 语义） | 1-step 之外还有 fully async（train_async.py:L9 注释指向 examples/full_async；slime/rollout/fully_async_rollout.py 实现）：后台 worker 维持固定并发池，跨 rollout 边界不停 | fully async 的价值在长尾波动与 agentic 多轮场景，需要随机时长才显形；L0 [4]b 已用确定性波动量化过弹性收益 |
| 背压 | 隐式（门槛即闸门） | fully_async_rollout.py:L85-89 逐字注释："Unbounded on purpose: put() runs inside the event-loop thread (task done-callback), so a bounded queue that fills up would block the loop and freeze every in-flight generation. Backpressure lives in _loop() instead, which stops topping up while a full pool of completed groups is already waiting to be consumed."；L148-152 的 qsize 门落实 | 这是「队列故意无界 + 补货侧背压」的设计：背压放在**取 prompt 补新任务**一侧（`qsize < max_concurrent` 才补），而不是给完成队列加界——后者会阻塞事件循环、冻死所有在途生成。L0 的 buffer 容量背压是同一思想的另一形态 |
| 权重同步 | 常数 S、全量、同步阻塞 | Megatron→SGLang 跨引擎传输，`--update-weight-buffer-size` 分块（arguments.py:L527-535，默认 512 MiB，help 提及 MoE 场景）、delta sync、`keep_old_actor` 三份轮转（actor.py:L142/L640） | S 的内部结构（分块/增量/轮转）是 L3 主题；L2 只把 S 当黑盒代价 |
| 中止与重入 | 无 | fully async 里 ABORTED 的组回流 data buffer 而不是发给训练（fully_async_rollout.py:L199-206），权重刷新后重新生成 | 涉及权重更新信号（`GenerateState.aborted`）与样本状态机，超出 L2 的 regime 主题 |
| 硬件拓扑 | 无 | 异步要求非 colocate（train_async.py:L11）；placement group 分卡 | 模拟器不建模 GPU 分配；结论已在 §4 点明（异步用更多硬件换重叠） |
| 时长分布 | 确定性常数 | 真实 G 有长尾（样本长度差异、工具/环境交互），reward 计算占时 | 确定性是可控实验的前提；波动弹性 L0 已量化，随机模拟会引入噪声带、模糊闭式核验 |

---

## 10. 溯源与口径声明

**slime 源码快照**：codeload main 分支 tarball，2026-08-14 04:05 抓取，
6,010,274 B，md5 `5215fd1640781770486e4ce7ec2ea838`（存于写作 workspace
`faa68d5e…/s75_slimeL2/sources/slime-main/`，未入材料树）。**本文全部行号以此快照为准**，
2026-08-14 现场逐一 grep 核验（非凭记忆）：

| 锚点 | 内容 |
|------|------|
| train.py:L53 / L85 / L27 | 阻塞 generate / 每 rollout update_weights / 循环前首推 |
| train_async.py:L11 | `assert not args.colocate`（异步不支持 colocate） |
| train_async.py:L32 / L36 / L40 | 预取 / 取上批 / 提前发起下一批 |
| train_async.py:L66-70 | `(rollout_id + 1) % args.update_weights_interval == 0` 门 + L67 屏障注释（逐字引于 §4）+ update_weights |
| slime/utils/arguments.py:L536-541 | `--update-weights-interval`，`default=1`（L539） |
| slime/utils/arguments.py:L527-535 | `--update-weight-buffer-size`，默认 `512 * 1024**2` |
| slime/backends/megatron_utils/actor.py:L142 / L640-645 | interval==1 消费点 ×2 + keep_old_actor 队列式轮转 |
| slime/rollout/sglang_rollout.py:L158 | router `/generate` HTTP 接口 |
| slime/rollout/fully_async_rollout.py:L85-89 / L148-152 / L199-206 | 背压设计注释（逐字引于 §9）/ qsize 门 / ABORTED 回流 |
| README.md:L45 | Delta Weight Sync 特性条目（链接下面文档） |
| docs/en/advanced/delta-weight-sync.md | delta sync 定义（§8 逐字引文）：disk-transport、xor/overwrite、zstd |

**论文锚点**（引擎侧背景，非本节数字的来源）：vLLM PagedAttention `[2309.06180]`、
SGLang `[2312.07104]`——2026-08-14 04:05 经 export.arxiv.org API 核验
（标题逐字："Efficient Memory Management for Large Language Model Serving with
PagedAttention" / "SGLang: Efficient Execution of Structured Language Model Programs"，
证据件 `arxiv_two.xml` 同 workspace）。

**跨模块锚点**：引擎代价模型常数 = nano-vllm-sglang `L0_kv_cache_batching.py:L26-27`
（W_READ/KV_STEP）、`iter_time` L38-40；L1 实测比例 = 本目录 tutorial_L1.md §5
（G/T=2.3）、§4（batching 2.60x）；staleness 定义与 buffer 闸门 = 本目录
tutorial_L0.md §4-6；IS 修正 = nano-verl tutorial_L1.md。

**口径**：本节所有数字分两类——(a) 模拟器产出（paste 块内全部数字，可复现：
掩码锚 `43a1a9ed…`/71 行、digest `d3c6be26…`）；(b) 源码引文/行号（上表，快照内逐字）。
**没有任何真实系统的 benchmark 数字**——真机验证 `[TODO: verify on real system]`。

---

## 11. 费曼自检

**类比：外卖站的两口锅。** 骑手取餐（生成）比厨师炒菜（训练）慢得多。
同步店：厨师炒完一单就站在窗口等骑手取走、等骑手回来交单才开始下一单——
厨师大半时间在罚站。异步店：骑手还在路上跑这一单时，厨师已经炒下一单了；
但店里有条规矩——**换菜谱（推权重）之前，必须等在路上的骑手全部回店**
（不然骑手按旧菜谱接了单、店里却换了菜谱，做出来对不上）。于是换菜谱越频繁，
骑手被拦在门口的次数越多；换得越慢，骑手带出去的菜就越「旧菜谱」。
`update_weights_interval` 拧的就是「多久换一次菜谱」——它决定菜有多旧，
不决定出餐多快；出餐快慢始终被较慢的那口锅钉死。

自检三问（讲不出来就回 §4/§5/§6 重读）：

1. 为什么 1-step 异步的加速上界恰好是 2x？（提示：每轮藏掉的是什么？）
2. `interval=8` 时，哪一批样本恰好吃满 staleness=8？为什么不是平均值？
3. 推权重前为什么要等在途生成完成？如果不等，会破坏什么不变量？

---

## 12. 思考题

1. **闭式推导**：从 §5 的时间线出发，推导 T 主导区（T > G）的
   `makespan(k)` 闭式（提示：引擎不再背靠背，推送窗口不再制造空闲——
   闭式与生成主导区有何不同？用模拟器改 G/T 验证你的推导）。
2. **fully async 的增量**：1-step 异步的流水深度是 1。若允许深度 d
   （d 批同时在途），稳态吞吐与 max staleness 各变成什么？为什么 slime
   把 fully async 做成独立模块而不是 train_async.py 的参数？
   （提示：权重更新屏障 L66-70 与 ABORTED 回流 L199-206。）
3. **colocate 的另一端**：train_async.py:L11 禁止 colocate。nano-verl
   选了 colocate + 分时复用。把两条路线的硬件成本、切换开销、适用 regime
   列成对照表——G/T 在什么区间时各占优？
4. **S 的工程**：delta weight sync 只传「两次同步之间变化的字节」
   （xor/overwrite 编码 + 恒开 zstd 压缩，见 docs/en/advanced/delta-weight-sync.md）。
   但全参数 Adam 训练里每个参数每步都在变——「变化的字节」名义上就是全量，
   delta 到底省在哪？（提示：两个数值接近的 bf16/fp32 张量逐字节 XOR，
   结果里什么样的字节占多数？zstd 对这种数据为什么赚？）再想另一条正交的省法：
   `--update-weight-buffer-size` 的分块更新，它的 help 为什么特意提 MoE
   （arguments.py:L527-535 原文 "should be useful for MoE models"）？
5. **staleness 的算法账**：本模块只算了 staleness 的**系统账**（它值多少吞吐）。
   算法账（off-policy 偏差如何随 staleness 损害收敛）要怎么测？设计一个
   toy 实验：同一 reward 函数、同一初始模型，扫 interval ∈ {1,2,4,8}，
   比较最终 reward 与训练曲线。（提示：nano-verl L1 的 IS ratio 探针可复用。）

---

## 13. 反例与边界

1. **toy 尺度诚实声明**：T=73.46、S=7.35 是模型参数（取 L1 实测比例），
   不是任何真实系统的测量。所有加速比（1.37x/1.70x/…）只在
   「代价模型 + 确定性控制流」内成立，**不可外推为 slime 真机性能**；
   可外推的是结构结论（2x 上界、interval 的旋钮定性、max staleness == k），
   因为它们是控制流的数学性质，不是参数拟合。
2. **确定性 ≠ 真实**：真实 G 有长尾、S 依赖网络拓扑、reward 计算占时、
   引擎内部有 continuous batching 的动态并发——全部未建模。确定性换来了
   闭式精确核验（§5），这是本节的实验方法论选择，不是对真实系统的完整描述。
3. **interval 的吞吐收益在波动下会回来**：[3] 说「interval 不是吞吐旋钮」
   的限定词是**确定性稳态**。真实负载下大 interval 的流水深度能吸收长尾
   （L0 [4]b 已量化波动弹性），那时它同时是吞吐旋钮——别把本节结论用过头。
4. **模拟的边界就是 L3 的入口**：引擎内部（continuous batching、KV、
   radix 前缀共享）→ nano-vllm-sglang L0–L3；权重同步内部（分块/delta/轮转）
   与 data buffer 实现 → 本模块 L3（见 §14）；真机数字 → `[TODO: verify on real system]`
   （GPU 通道攒批验证，写作轮不 ssh）。

---

## 14. 阶梯预告

L3（README 阶梯表定义）：**对照 slime 源码做实现级深挖**——data buffer 实现、
delta weight sync 的传输路径与分块策略（arguments.py:L527-535 的参数如何落到
Megatron→SGLang 的张量搬运）、rollout 调度与 update 时机的完整状态机
（含 fully_async_rollout.py 的 ABORTED 回流与背压实现），并指出 nano 版
每一处简化在真实代码里长什么样。L2 给了控制流与代价的骨架，L3 填血肉。
