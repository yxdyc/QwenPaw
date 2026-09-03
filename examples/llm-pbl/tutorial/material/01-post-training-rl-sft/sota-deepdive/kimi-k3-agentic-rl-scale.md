# SOTA Deep-Dive — Kimi-K3：agentic RL 规模化

> **深挖对象**：01 轨 sota-deepdive README scope ②——Kimi-K3 的 agentic RL 规模化：rollout 吞吐、权重同步、长时程轨迹的 credit assignment、多步工具调用 reward 设计。
> **状态**：✅ 首版完成；2026-08-31 重新核对 K3 v2、OPD survey v4 与官方仓库。
> **SOTA 快照日期**：2026-08-31。前沿结论均为 dated snapshot，复用前应重查报告版本与官方实现。
> **与既有材料的关系**：主文档 [post-training-algorithm-evolution.md](post-training-algorithm-evolution.md) §7.4 与深化文档 [post-training-algorithm-evolution-deepening.md](post-training-algorithm-evolution-deepening.md) §6.2/§7 已提取 K3 的正文层证据片段（MOPD/partial rollout/verbosity）；本文是 scope ② 的**独立成篇**——把 K3 的 agentic RL 作为一个完整系统来解构：算法（K2.5 Eq.1）与 infra（co-located + partial rollout + AgentENV sandbox）如何互为存在条件。深化文档的机制面证据仍然有效，本文不重复其 OPD 合流分析，只在必要处交叉引用。
> **native sim**：[`kimi_k3_agentic_rl_sim.py`](kimi_k3_agentic_rl_sim.py)（本文 §1 paste 块 = 该 sim 实测输出，BYTE-IDENTICAL 机器证明见 §10.2）。

---

## §0 这篇 deepdive 解构什么

标准 RLVR（可验证 reward 的 RL）的隐含假设是：**一条轨迹便宜、短命、无状态**——生成一次、打分一次、丢掉。agentic RL 把这三条全部打破：

1. **轨迹变贵**：一条轨迹 = 数百上千次工具调用 + 百万级累计 context token（K3 §1 逐字：「often over hundreds or thousands of tool calls and millions of accumulated context tokens」）。生成一条轨迹的主要成本不再是模型前向，而是**环境交互的墙钟时间**（工具执行、页面加载、代码运行）。
2. **轨迹变长命**：长到单条轨迹的生成时间**超过一个训练迭代**——「an individual long-horizon trajectory naturally spans multiple iterations, introducing data staleness」（K3 §4.1.2 逐字）。陈旧度从 corner case 变成常态。
3. **环境变有状态**：sandbox 里有磁盘、有进程、有挂载——环境本身成了需要 checkpoint/fork/resume 的**状态对象**，而不是无状态的 reward 函数。

Kimi K3（2.8T 参数 MoE、104B 激活、1M context，K3 §1）是目前把这三件事做到最大公开规模的技术报告。本文沿四条坐标轴解构它：

- **§3 rollout 吞吐**：partial rollout 的 λ 旋钮——训练不再等 straggler，代价是陈旧度（sim [A]）。
- **§4 长时程 credit assignment 的算法地基**：K2.5 Eq.1 的 token 级 ratio 窗口 + 平方正则——「per-token regularization」为什么能容忍极端 off-policy（sim [B]）。
- **§5 权重同步**：co-located 设计如何把「权重同步问题」消解成「内存竞争问题」——与分卡异步路线（nano-slime 对照）是同一矛盾的两端。
- **§6 多步工具调用的 reward 设计**：预算控制、GRM 二元比较、MOPD dense reward、Toggle 交替（sim [C]）。
- **§7 sandbox 即 infra**：AgentENV 的 pause/resume/fork 经济学（sim [D]）。

一条主线贯穿全文：**K3 的算法选择与 infra 选择不是两件独立的事**。partial rollout（算法侧接受陈旧度）要求 sandbox 可暂停续跑（infra 侧）；per-token 正则（算法侧容忍 off-policy）让 partial rollout 敢把 λ 调小（infra 侧吞吐）；预算控制（reward 侧）直接改变 rollout 的长度分布（infra 侧 KV 压力）。主文档 §7「算法-infra 共演化」的论断在 K3 这里拿到了最完整的正文级证据链。

---

## §1 运行与输出

**可运行性契约**：这是本质模拟。四个机制面（partial rollout 权衡 / per-token 正则的符号无关性 / 预算与 verbosity / sandbox 经济学）全部是纯标准库、确定性、带 self-check 的 toy；seed=20260815 固定，无计时行，跨运行逐字节一致。论文侧数字（133 ms / 49 ms / 98% / 6.5× 等）只作 declared 参数入算或机制对照，**不外推量级**；sim 自己的数字与论文数字逐处区分。

```
$ python3 -B /path/to/sota-deepdive/kimi_k3_agentic_rl_sim.py   # Python 3.10+，任意空 CWD 可跑
```

以下为固定 seed 的真实输出；发布门禁要求两个新建空 CWD 均 exit 0、stderr 为空、stdout 逐字节一致：

```
========================================================================
kimi k3 agentic rl sim — partial rollout / per-token 正则 / 预算 / sandbox
========================================================================
toy: 纯标准库 | seed=20260815 | 机制回声: K3 [2607.24653] §4.1.2/§4.1.3/§5.3 + K2.5 [2602.02276] §4.4.2
全部数字为本 sim 实测或 declared 折算，论文侧只作机制对照、不外推量级。

[A] partial rollout：straggler 等待 vs 数据陈旧度
    toy: N=24 prompt × K=4 = 96 轨迹，重尾时长（Geom p=0.22 截断 24）
    机制回声: K3 §4.1.2 λ 暂停-续跑；陈旧度 = 消费迭代 − 生成迭代
    时长分布: min=1 median=4 p90=11 max=24（重尾：max/median = 6.0×）
    λ=1.0 : 首次训练@slot 24  训练步数=1  平均陈旧度=0.000 迭代 跨≥2迭代轨迹占比=0.000
    λ=0.75: 首次训练@slot 7   训练步数=2  平均陈旧度=0.292 迭代 跨≥2迭代轨迹占比=0.208
    λ=0.5 : 首次训练@slot 4   训练步数=2  平均陈旧度=0.351 迭代 跨≥2迭代轨迹占比=0.438
    λ=0.25: 首次训练@slot 2   训练步数=4  平均陈旧度=0.827 迭代 跨≥2迭代轨迹占比=0.656
    生成总时长 = max D = 24 slot，与 λ 无关（四 λ 逐位同一=True）
    λ=0.25 vs 同步: 首次训练 24→2 slot，训练步数 1→4，平均陈旧度 0.000→0.827

[B] per-token 正则：log-ratio 窗口裁剪为什么符号无关
    [B1] 梯度通图（|dL/dlogr|，log-ratio 网格 −2…+2，步长 0.25）
     pg A=+1:   0.14   0.22   0.37   0.61   1.00   1.65   2.72   4.48   7.39
     pg A=−1:   0.14   0.22   0.37   0.61   1.00   1.65   2.72   4.48   7.39
    ppo A=+1:   0.14   0.22   0.37   0.61   1.00   0.00   0.00   0.00   0.00
    ppo A=−1:   0.00   0.00   0.00   0.00   1.00   1.65   2.72   4.48   7.39
    k25 A=+1:   0.32   0.24   0.16   0.08   1.00   0.08   0.16   0.24   0.32
    k25 A=−1:   0.32   0.24   0.16   0.08   1.00   0.08   0.16   0.24   0.32

    [B2] 最坏情形梯度界（对 A 符号取 max 的 |dL/dlogr|）
     pg:   0.14   0.22   0.37   0.61   1.00   1.65   2.72   4.48   7.39
    ppo:   0.14   0.22   0.37   0.61   1.00   1.65   2.72   4.48   7.39
    k25:   0.32   0.24   0.16   0.08   1.00   0.08   0.16   0.24   0.32
    网格最大值: pg=7.39（∝r 无界） ppo=7.39（无界边） k25=1.32（有界）

    [B3] 陈旧批次多步训练（60 步，lr=0.06，400 条×6 token）
         压力测试：advantage 符号翻转（A→−A）= 陈旧到符号不可信
     pg 正常   KL: step0:0.000 step30:0.090 step59:0.346
    ppo 正常   KL: step0:0.000 step30:0.068 step59:0.135
    k25 正常   KL: step0:0.000 step30:0.084 step59:0.204
     pg 翻转   KL: step0:0.000 step30:0.089 step59:0.347
    ppo 翻转   KL: step0:0.000 step30:0.065 step59:0.122
    k25 翻转   KL: step0:0.000 step30:0.081 step59:0.169
    翻转放大系数: pg=1.00 ppo=0.90 k25=0.83
    口径说明: 本 toy 的正常漂移方向恰与 PPO 掩码边同向（掩码随 A 符号旋转），故 ppo 翻转不放大——K2.5 的符号无关性是结构性保证（B1/B2 机器证明），在 advantage 符号相对漂移方向不可信的 regime（train-inference mismatch + 跨迭代陈旧）才成为必需；k25 正常漂移 0.204 > ppo 0.135 系窗口更宽（[0.75,1.35] vs [0.8,1.2]）的超参权衡——窗口越宽传过的梯度越多、界越松，宽度是旋钮。

[C] 预算控制与 verbosity：reward hacking → 硬预算掉坑 → Toggle
    toy: P(success|ℓ,d) = 1−exp(−ℓ/s_d)，s_easy=3 / s_hard=12
    三臂同初始化 ℓ=6、同 lr=1.2、同 30 轮；预算 τ_b=2.0×b_0=12
      free: 终局 ℓ_easy= 17.40 ℓ_hard= 15.99 | 放量成功率 P(easy)=0.997 P(hard)=0.736
    budget: 终局 ℓ_easy=  6.68 ℓ_hard=  8.18 | 放量成功率 P(easy)=0.892 P(hard)=0.494
    toggle: 终局 ℓ_easy=  9.40 ℓ_hard= 10.97 | 放量成功率 P(easy)=0.956 P(hard)=0.599
    未见更难题（s=24）泛化: free=0.486 budget=0.289 toggle=0.367（硬预算臂早停习惯的代价）
    GRM 判负探针: ℓ_0=100, σ=1.5 → 阈值 150；候选 A(q=0.82,ℓ=170) 质量胜但超长 → 自动判负=True，胜者 A→B

[D] sandbox pause/resume 经济学（declared 折算，K3 §5.3.2 参数）
    轨迹寿命 L=3600s，等待占比 98%，切换 40 次 × (133+49)ms
    无 pause: 占用 3600 资源·s | 有 pause: 活跃 72.0 + 切换开销 7.28 = 79.28 资源·s
    超卖比 = 45.4×（98% 上界口径；K3 报告实测 up to 6.5×——真实负载等待占比低于上界，故实测更低，方向一致）

[E] self-check
    PASS  A1 同步 λ=1 陈旧度恰为 0（无轨迹跨迭代）  [stale=0.0, span2=0.0]
    PASS  A2 λ 越小首次训练越早（训练不再等 straggler）  [first_train=[24, 7, 4, 2]]
    PASS  A3 λ 越小同窗内训练步数越多（训练频率↑）  [steps=[1, 2, 2, 4]]
    PASS  A4 λ 越小平均陈旧度越高（权衡成立）  [stale=[0.0, 0.292, 0.351, 0.827]]
    PASS  A5 λ=0.25 时多数轨迹跨迭代（占比 >0.5）  [span2=0.656]
    PASS  A6 生成总时长与 λ 无关（partial rollout 解耦训练而非加速生成）  [total_slots=24]
    PASS  B1a PPO A=+1 只封右尾（左尾梯度照传、右尾掩码）  [left=0.135, right=0.000]
    PASS  B1b PPO A=−1 只封左尾（封的半边随 A 符号翻转）  [left=0.000, right=7.389]
    PASS  B1c K2.5 两端出窗后只剩平方正则（两端对称 = 2τ|logr|）  [ends=0.320/0.320, 2τ|logr|=0.320]
    PASS  B1d K2.5 窗口内 A 项照传（logr=0 处 signed = −A）  [A=+1:-1.000 / A=−1:1.000]
    PASS  B1e 符号无关性总账：K2.5 两符号通图逐位重合、PPO 不重合  [k25 两符号差=0.00e+00, ppo 两符号差=7.25]
    PASS  B2a PPO 最坏梯度随 |logr| 无界增长（= r，符号依赖漏边）  [max=7.389 = e^2=7.389]
    PASS  B2b K2.5 最坏梯度全网格有界（≤ 窗口内 A 项 + 正则）  [max=1.324（界内上界 β+2τ|logr_max|=1.670）]
    PASS  B2c 有界性差距 >4×（同一网格、同一最坏口径）  [ratio=5.58]
    PASS  B3a 正常陈旧批次：裸 PG 漂移最大（无任何界）  [KL_end pg=0.346]
    PASS  B3b 裁剪族把陈旧漂移压在 0.25 nats 内（四种情形全满足）  [max=0.204 vs pg=0.346]
    PASS  B3c K2.5 翻转近似不变（界不随 A 符号翻，结构性保证）  [amp=0.83]
    PASS  B3d 极端 off-policy 下 k25 漂移 ≪ 裸 PG  [k25_flip=0.169 vs pg_flip=0.347]
    PASS  C1 free 臂 easy 题长度膨胀（verbosity hacking）  [ell_easy=17.40 vs 初始 6]
    PASS  C2 硬预算臂 easy 题长度被压在预算附近（≤2.2×b_0）  [ell_easy=6.68（预算 12）]
    PASS  C3 硬预算臂 hard 题掉坑（length-overfitting）  [P(hard) budget=0.494 vs free=0.736]
    PASS  C4 Toggle 两全：easy 比 free 省、hard 比 budget 强  [ell_easy=9.40<17.40, P(hard)=0.599>0.494]
    PASS  C5 未见更难题：Toggle 泛化 > 硬预算（早停代价）  [harder: toggle=0.367 vs budget=0.289]
    PASS  C6 GRM verbosity 控制：超长候选质量胜也自动判负  [threshold=150, ell_A=170]
    PASS  D1 pause/resume 切换税 <1%（开销占比可忽略）  [overhead=7.28s / 3600s]
    PASS  D2 超卖比 >10×（98% 上界口径，机制方向坐实）  [overcommit=45.4]
    PASS  D3 开销 = 切换次数 × 0.182s（线性，机器证明）  [overhead=7.280]
    ✅ self-check passed (27/27)

digest(md5 of metrics) = 1c75f9c845e1cd8b2681f2a205411bf3
```

self-check 27/27 全过，digest `1c75f9c845e1cd8b2681f2a205411bf3`（md5 of metrics，跨运行不变）。下面按坐标轴解构。

---

## §2 为什么 agentic RL 是另一门学问（K3 的问题设定）

K3 的 RL 覆盖三个域 × 三个 reasoning effort 级别（§4.1.2 逐字）：

> (i) general tasks, spanning general experience, vision, reasoning, faithfulness, search capabilities, and knowledge work tasks; (ii) general agents, spanning long-horizon assistant tasks, deep research, and paragraph-level writing; and (iii) coding agents, spanning software engineering (SWE), coding experience, kernel tasks, and web development.
>
> Crossing these three domain experts with three reasoning effort levels in {low, high, max} yields a total of nine expert models.

训练环境包括「verifiable search and professional knowledge work, software engineering and kernel optimization, multimodal reasoning with vision-in-the-loop tool use, persistent assistant workflows, web development, and autonomous execution tasks」（K3 §1 逐字），而核心回路是「a general loop of reasoning, acting, observing, verifying, and adapting, often over **hundreds or thousands of tool calls** and **millions of accumulated context tokens**」（K3 §1 逐字）。

两个规模数字把「agentic RL 的 infra 问题」从修辞变成算术：

- **51,219,741 个 sandbox**、1,505,678 个镜像（K3 §5.3.2 逐字：「Throughout Kimi K3's training and evaluation, a total of 51,219,741 sandboxes across 1,505,678 images were created」）——训练期间创建的环境实例是千万量级。
- **100,000 并发 agent 任务**（K2.5 §4.5 逐字：「a dedicated Rollout Manager orchestrates up to 100,000 concurrent agent tasks during the RL process」，K3 沿用该框架）。

以及一个方向性证据——Fig. 8 的 caption（K3 §4.1.2 逐字）：

> By scaling RL FLOPs, tool-call steps scale up consistently, accompanied by a comprehensive improvement in the model's overall capability.

**RL FLOPs 增加 → 模型自发使用更多工具步 → 能力全面提升**：agentic RL 的 scaling law 作用在「交互步数」这个维度上，而不只是 response 长度。这意味着轨迹会越来越长、越来越贵——§3–§7 的全部工程都是在这个趋势下维持训练吞吐的对策。

---

## §3 [A] partial rollout：λ 旋钮的解剖（sim [A] 回声）

### 3.1 机制原文

K3 §4.1.2「Algorithm」（逐字）：

> To mitigate the long-tail latency that intensifies in long-horizon tasks, we extend the partial rollout scheme from our synchronous RL framework [117, 60]. During the rollout phase of each iteration, we sample K completions for each of N prompts, maintaining an active workload of N×K trajectories. Rather than waiting for all rollouts to terminate, the generation phase pauses as soon as a fraction λ∈(0,1) of trajectories completes (i.e., λNK), allowing policy optimization to proceed without execution stragglers. Paused rollouts are enqueued and prioritized for resumption at the start of the next iteration, powered by our sandbox infrastructure (§5.3.2). Once all K responses for a prompt complete, they are immediately dispatched for policy optimization, which follows the algorithm in Kimi K2.5 [60].

拆开看，这个方案有四个构件：

1. **恒定活跃负载 N×K**：池子始终满载，暂停的轨迹由新 prompt 的轨迹补位（sandbox 池 + 环境实例池支撑，K2.5：「Upon activation, each task acquires an environment instance from a managed pool」）。
2. **λ 触发训练**：完成比例达 λ 就推进，不等 straggler。λ=1 退化为同步 RL。
3. **暂停-续跑**：未完成轨迹带着已有进度进入下一迭代（不是丢弃重采）——这一步的存在条件是 §7 的 sandbox checkpoint/resume 与 §5 的 KV-cache 保留，**没有状态外化就没有 partial rollout**。
4. **prompt 粒度 dispatch**：一个 prompt 的 K 条全完成才进训练（组基线需要完整组，K2.5 Eq.1 的 r̄(x) 项）。

### 3.2 sim [A] 实测：权衡的定量形态

toy 设置：96 条轨迹（24 prompt × 4），重尾时长（median 4、max 24 slot，max/median = 6.0×——agentic 长尾的 toy 形态）。四个 λ 的实测：

| λ | 首次训练 | 训练步数 | 平均陈旧度（迭代） | 跨≥2迭代轨迹占比 |
|---|---|---|---|---|
| 1.0（同步） | @slot 24 | 1 | 0.000 | 0.000 |
| 0.75 | @slot 7 | 2 | 0.292 | 0.208 |
| 0.5 | @slot 4 | 2 | 0.351 | 0.438 |
| 0.25 | @slot 2 | 4 | 0.827 | 0.656 |

三个结论：

- **首次训练从 slot 24 提前到 slot 2**（λ=0.25）：同步方案里，训练被最慢那条轨迹（24 slot）完全阻塞；partial rollout 下训练在第 2 个 slot 就开始。长尾越重（max/median 比越大），这个收益越大——这正是 K3「long-tail latency that intensifies in long-horizon tasks」的算术内核。
- **陈旧度是 λ 的单调代价**（A4：0 → 0.292 → 0.351 → 0.827）：λ=0.25 时 65.6% 的轨迹跨迭代（A5），其早段 token 带着 ≥1 迭代的陈旧度进训练。**这就是 K3 必须配 per-token 正则的原因**——§4 接住。
- **诚实边界（A6）**：生成总时长 = max D = 24 slot，与 λ 逐位无关。**partial rollout 不加速生成，它解耦的是「训练何时开始」**。总生成吞吐由引擎与 sandbox 决定（§5/§7 的主题）；λ 买的是训练频率，付的是陈旧度。把 partial rollout 说成「加速 rollout」是流行但错的说法（§9.3 反例 1）。

### 3.3 与主文档的衔接

主文档 §3.3 讨论过 IS 的边界：「ratio 只修正同一 prefix 上的动作分布，prefix/state 仍是旧策略产生的」。partial rollout 的陈旧度正是这个边界的**生产形态**：一条跨 3 个迭代的轨迹，其中段 token 的 prefix 是两代前的策略生成的，ratio 修正不了这部分。K3 的对策不是修正它（没有更复杂的 IS 权重），而是**用 per-token 正则容忍它**（§4）——这是「限制而非纠正」的路线，与 DeepSeek-V4 在 OPD 上选全词表 KL 的「纠正」路线（深化文档 §6.3）形成方法论对照。

---

## §4 [B] per-token 正则：K2.5 Eq.1 的机制解剖（sim [B] 回声）

### 4.1 算法原文

K3 的 policy optimization 明确沿用 Kimi K2.5 [60] 的算法，所以算法本体见 K2.5 §4.4.2 Eq.1：

```
L_RL(θ) = E_x[ (1/N) Σ_j Σ_i Clip( π_θ(y_i^j | x, y_{0:i}^j) / π_old(y_i^j | x, y_{0:i}^j), α, β ) · ( r(x, y^j) − r̄(x) )
               − τ · ( log π_θ(y_i^j | ...) / π_old(y_i^j | ...) )² ]
```

其中 N = 批内总 token 数，r̄(x) = 组内 K 条的平均 reward（GRPO 族组基线，主文档 §3），α, β, τ > 0。这个 loss 有三层结构：

1. **组基线 + token 平均**：(r − r̄(x)) 是 GRPO 族组基线（深化文档 [B] 面实测过其偏移不变性与方差缩减）；1/N 按总 token 数归一（长度无偏聚合，主文档 §4 的 Dr. GRPO 机制类别）。
2. **token 级 ratio 窗口 Clip(r, α, β)**：log-ratio 出窗 [log α, log β] 的 token，policy gradient 直接掩码。K2.5 原文（逐字）：

   > The mechanism functions as a simple gradient masking scheme: policy gradients are computed normally for tokens with log-ratios within the interval [α,β], while gradients for tokens falling outside this range are zeroed out. Notably, a key distinction from standard PPO clipping [50] is that our method relies strictly on the log-ratio to explicitly bound off-policy drift, **regardless of the sign of the advantages**.

   （[50] = PPO arXiv 1707.06347。）
3. **每 token 平方 log-ratio 正则 −τ(log r)²**：把每个 token 的更新锚在 behavior 邻域——**这就是 K3 所说的「per-token regularization」**（K3 §4.1.2 逐字：「By constraining policy updates within a localized neighborhood, this regularization enables the algorithm to robustly handle highly stale data and sustains training stability」）。

谱系注记：log-ratio 窗口掩码与 CISPO（MiniMax-M1 [2506.13585] 内）同族；K2.5 把它放在近期稳定大规模 RL 的方法谱系中。本文不追单源方法名，只教机制类别：**ratio 窗口 + 符号无关掩码 + 逐 token 锚定**。

### 4.2 与 PPO clip 的本质区别：掩码边随不随 A 的符号转

PPO clip 是 `min(r·A, clip(r, 1−ε, 1+ε)·A)`——**哪一边被截断取决于 A 的符号**：A>0 封 r>1+ε 侧（r<1−ε 侧梯度照传），A<0 封 r<1−ε 侧（r>1+ε 侧照传）。sim [B1] 的梯度通图把这个区别画了出来（|dL/dlogr|，log-ratio 网格 −2…+2）：

```
 ppo A=+1:   0.14   0.22   0.37   0.61   1.00   0.00   0.00   0.00   0.00
 ppo A=−1:   0.00   0.00   0.00   0.00   1.00   1.65   2.72   4.48   7.39
 k25 A=+1:   0.32   0.24   0.16   0.08   1.00   0.08   0.16   0.24   0.32
 k25 A=−1:   0.32   0.24   0.16   0.08   1.00   0.08   0.16   0.24   0.32
```

- **PPO 两行不重合**（B1e：两符号差 7.25）——A 符号一翻，被掩码的半边就翻到对面。
- **K2.5 两行逐位重合**（B1e：差 0.00e+00）——掩码区域是 log-ratio 的纯函数，与 A 无关；窗口内 A 项照传（B1d：logr=0 处 signed = −A），窗口外只剩平方正则的对称锚力（B1c：两端 = 2τ|logr| = 0.320）。

为什么「符号无关」在 agentic RL 里是必需的？因为 partial rollout + train-inference mismatch 共同制造了一种 regime：**advantage 的符号相对当前策略的漂移方向不可信**。轨迹是几代前的策略生成的，advantage 用陈旧 reward 算，而 ratio 的分母 π_old 又是推理引擎的 logprob（K2.5 专门为此做「train-inference mismatch correction」：「We also record log probabilities for all inference engine outputs to perform train-inference mismatch correction」，§4.5 逐字）。在这种 regime 下，「掩码边随 A 翻转」意味着**保证的边界本身是数据依赖的**——而 K2.5 的窗口给出的界只依赖 log-ratio 本身。

[B2] 的最坏情形梯度界是这个区别的定量形态：对 A 符号取 max 后，PPO 的最大梯度 = e^2 ≈ 7.39（∝r 无界，漏边吃满），K2.5 = 1.32（全网格有界），差 5.58×（B2a–B2c）。**PPO 的无界边不是理论瑕疵——K2.5 明说这个机制「essential for maintaining training stability in complex domains requiring long-horizon, multi-step tool-use reasoning」（§4.4.2 逐字）**。

### 4.3 [B3] 陈旧批次训练：裁剪族 vs 裸 PG

同一陈旧批次（behavior 策略采的 400 条 × 6 token）反复训练 60 步，KL(π_θ||π_old) 终值：裸 PG 0.346，PPO 0.135，K2.5 0.204（正常）/ 0.169（advantage 符号翻转压力测试）。三个诚实结论：

- **裁剪族把陈旧漂移压在 0.25 nats 内**（B3b：四种情形全满足），裸 PG 无界（0.346 且趋势未收敛）。
- **K2.5 的翻转放大系数 0.83 ≈ 1**（B3c）：界不随 A 符号翻——结构性保证在动力学上可见。
- **口径诚实**：本 toy 的正常漂移方向恰与 PPO 掩码边同向（掩码随 A 旋转正好追上漂移方向），所以 PPO 翻转不放大（amp 0.90）、且正常漂移 0.135 < K2.5 0.204——后者是窗口宽度超参权衡（[0.75,1.35] vs PPO [0.8,1.2]，窗口越宽传过的梯度越多、界越松）。**符号无关性的价值不在「正常 regime 更紧」，而在「regime 变了界还在」**——这正是 partial rollout 把 λ 调小、轨迹跨更多迭代时发生的事。

### 4.4 优化器侧的配套：MuonClip

K2.5 使用 MuonClip 优化器；其参考文献 [29] 为 Muon，[33] 为 arXiv 2502.16982。本文不展开 Muon 细节（超出 scope）；只强调：ratio clip 约束**梯度信号**，MuonClip 约束**参数更新步长**，二者不是同一层面的限幅器。

---

## §5 权重同步：一个被 co-location 消解的问题

### 5.1 两种架构的对照

分离式架构（训练卡与推理卡分开）里，「权重同步」是一个显式且昂贵的步骤：每个训练步之后把新权重从 trainer 搬到 rollout 引擎。nano-slime tutorial_L2 §8 模型化了 delta weight sync，nano-verl 则采用 colocate + 分时复用。两条路线是同一矛盾的两端：**分卡买重叠（生成与训练并行），付同步；同卡买零同步，付分时切换**。

K3 在百万 token context + 2.8T 参数下选了 co-located 一端（§5.3.1 逐字）：

> We adopt co-located RL training [57] to keep each 1M-context Kimi K3 RL experiment within a few hundred GPUs, and use partial rollouts [117] to reduce tail latency from ultra-long trajectories. This design achieves good hardware utilization, but introduces a **memory usage contention** between rollout KV-cache that needs to be persisted for the next iteration, and the memory needed for training.

（[57] = Kimi K2 [2507.20534]。）

注意这段话的结构：**co-location 消解了权重同步问题（同卡，权重原地可见），但立刻生出了一个新问题——内存竞争**。训练态（权重 + 优化器态 + 梯度）与 rollout 态（百万 token 的 KV cache，且 partial rollout 要求它跨迭代保留）抢同一块显存。权重同步没有消失，它**变形**了。

### 5.2 变形后的三个对策（K3 §5.3.1 逐字机制）

1. **外部 KV cache 池（write-back）**：「Active decoding blocks remain in GPU KV cache, while reusable idle prefixes are written back to an external KV cache pool in CPU DRAM only when it is evicted from GPU, and is prefetched back before the next reuse」。为什么必须？——「At 1M-context multi-step rollout, a prefix KV-cache miss is extremely expensive. Partial rollout exacerbates this issue at the beginning of each iteration, due to many unfinished long prefill requests from the previous iteration arriving at the same time」。**partial rollout 的续跑轨迹在每个迭代开头集中回流，制造 prefill 风暴**——这是算法选择（λ 暂停）直接生成 infra 负载形态的实例。
2. **训练态 NVMe offload**：「we offload training states (model weights and optimizer states) to NVMe after a training iteration finishes. After a rollout iteration, the pool is released to avoid contention with training workloads」——训练与 rollout 分时使用显存/DRAM/NVMe 三级存储，状态在层级间搬迁。
3. **rollout 自动节流**：「In multi-step rollout, contexts grow progressively as the trajectory advances, making fixed concurrency based on the full-trajectory average length both hard to estimate and overly conservative early on... We therefore design an auto-throttling mechanism at the LLM request scheduling layer, using runtime signals such as active request count, queued request count, and KV cache utilization」——并发度随 KV 压力动态收放，防 preemption。这与 nano-vllm-sglang 阶梯的主题（KV 预算与准入）在机制层同构（03 轨只读交叉引用）。
4. 配套细节：非策略模型（reference model）权重放 CPU，用策略模型的 FP32 梯度缓冲做临时显存（「backing their parameter tensors with the policy model's FP32 gradient-buffer storage」）——前向时才物化，省常驻显存。

### 5.3 本质归纳

「权重同步」在 co-located 设计下不再是带宽问题，而是**状态生命周期管理问题**：谁的 KV 该跨迭代活下来（外部池）、谁的训练态该让位（NVMe）、谁的并发该收（节流）。这个归纳对 senior 的意义：选架构时不要问「同步带宽够不够」，要问「状态在时间轴上如何布局」——K3 的答案是**把 rollout 状态变成一等 infra 对象**（§7 的 sandbox 是同一条路线的环境侧）。深化文档 §7.2 的论断「长时程轨迹把 rollout 状态变成 infra 对象」在 §5.3.1 拿到了逐字证据。

---

## §6 [C] 多步工具调用的 reward 设计（sim [C] 回声）

agentic RL 的 reward 设计要同时回答三个问题：**长轨迹怎么打分**（§6.1 预算）、**不可验证任务怎么打分**（§6.2 GRM）、**dense 信号从哪来**（§6.3 MOPD）。

### 6.1 预算控制：T(y) 在 agentic 任务里计什么

K3 §4.1.2「Reasoning Effort RL」（逐字）：

> We associate each problem x with an initial token budget b_0(x) estimated from the cold-start model, and override the task reward with −1 for trajectories whose total token budget T(y) exceeds a scaled threshold τ·b_0(x). For general tasks, T(y) measures the number of thinking tokens, whereas **for agentic tasks, T(y) accounts for the cumulative output tokens, including both reasoning traces and tool-call arguments**.

两个细节值得停：

- **agentic 任务的长度计量包含 tool-call arguments**——工具调用参数也是模型输出，也占预算。这堵死了「把 verbosity 转移到工具调用里」的 hacking 路径（思考题 1）。
- **预算是 per-problem 的（b_0(x) 由 cold-start 模型估计），且 τ 走 stage-wise curriculum**：「We first train a max-budget variant with a relatively large τ, while still capping the maximum budget to suppress excessive overthinking. We then anneal τ to smaller values to obtain the high- and low-effort expert models. The adjustment of τ is configured per domain under human-in-the-loop guidance」——先放宽再收紧，产出 {low, high, max} 三档 effort 专家（§2 的九专家之一半来源）。

sim [C] 的三臂对照（同初始化 ℓ=6、同 30 轮）实测了这个机制族的完整动力学：

| 臂 | ℓ_easy | ℓ_hard | P(hard) 放量 | P(harder, s=24) 未见泛化 |
|---|---|---|---|---|
| free（无预算） | 17.40（膨胀，C1） | 15.99 | 0.736 | 0.486 |
| budget（硬预算 −1 覆写） | 6.68（C2） | 8.18 | 0.494（掉坑，C3） | 0.289 |
| toggle（Phase0/Phase1 交替） | 9.40 | 10.97 | 0.599（C4） | 0.367（C5） |

- **free 臂 verbosity hacking**：easy 题饱和长度 ~9，策略却膨胀到 17.4——多出来的长度不买成功率，只因为 reward 对长度单调不减（边际收益递减但为正），RL 就把「保险性多走几步」学进来。
- **硬预算臂 length-overfitting**：easy 题效率最好（6.68），但 hard 题掉坑（0.494 vs free 0.736），且在**未见过的更难难度**上泛化最差（0.289）——它学会了「早停」这个习惯，放量也改不掉。K2.5 对这个现象有逐字命名：「a **length-overfitting phenomenon**: models trained under rigid budget constraints often fail to generalize to higher compute scales... defaulting to truncated reasoning patterns」。
- **Toggle 是交替优化**（K2.5 §4.4.2 Eq.：Phase0 预算受限、Phase1 自由 scaling，每 m 轮交替；Phase0 还有豁免条件——组均 reward < λ 时不施加预算，「模型还不会做时先别谈效率」）：easy 比 free 省（9.40 < 17.40）、hard 比 budget 强（0.599 > 0.494）、未见难题泛化也更好（0.367 > 0.289）。K2.5 在 K2 Thinking 上的报告数字是「Toggle decreases output tokens by 25~30% with a negligible impact on performance」（K2.5 §4.4.2 逐字，[blog claims] 口径不适用——这是报告正文数字，标 [TODO: verify] 仅限其实验表逐项）。

### 6.2 GRM：不可验证任务的二元比较 + verbosity 自动判负

agentic 任务大量是「不可验证」的（写作、研究、助手类）——没有单元测试可跑。K3 用 Agentic Generative Reward Model（§4.1.2 逐字）：

> For non-verifiable general tasks, we adopt an Agentic Generative Reward Model (GRM), retaining the tournament-style group reward with binary comparisons as in Kimi K2.5 [57, 60]. Beyond generic agentic capabilities for enhanced judgment, the agentic judge is required to follow a mandatory protocol: (1) read the outcome, product, or text output; (2) generate a rubric; (3) score each candidate against the rubric; and (4) record the rubric-assigned scores in a scorepad.

judge 本身是 agent（读产物 → 生成 rubric → 按 rubric 打分 → 记分），协议强制化是为了压制 judge 的随意性。而 verbosity hacking 在二元比较里的对策是一行规则（逐字）：

> given an initial verbosity ℓ_0 estimated from the cold-start model and a multiplier σ, a candidate whose output length exceeds σ·ℓ_0 **automatically loses the binary comparison**.

sim [C] 的 GRM 探针把这个规则的算术形态跑了出来：候选 A 质量分 0.82 > B 的 0.80，但 ℓ_A=170 > σ·ℓ_0=150 → 自动判负，胜者 A→B（C6）。**长度规则先于质量比较生效**——这是「把预算写进 reward 结构」而不是「在 reward 上加惩罚项」：惩罚项可以被质量优势抵消，自动判负不可以。

### 6.3 MOPD dense reward：per-token 信号与 clip 家族合流

九专家（3 域 × 3 effort）最终要合并成一个统一模型，K3 用 Multi-Teacher On-Policy Distillation（§4.1.3；深化文档 §6.2 已有分析，此处只补 credit assignment 视角）。per-token OPD reward 见 Eq.15：

```
r_opd^d(y_t | e, x, y_{<t}) = clip( sg( log π_teacher^(d,e)(y_t | x, y_{<t}) / π_θ(y_t | e, x, y_{<t}) ), −R_max, R_max )
```

> where sg(·) denotes the stop-gradient operator, and R_max > 0 is a clipping threshold to constrain extreme advantage signals, thereby stabilizing RL training. This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as **partial rollout training** for long-horizon tasks.

三个观察：

- **长轨迹的 credit assignment 靠 dense per-token 信号**：轨迹级 reward（成功/失败）在百万 token 轨迹上信号太稀——每个 token 都有教师背书度作 reward，credit 不需要从轨迹末端反向摊派。这是「reward 设计」与「credit assignment」在 K3 里的合流点。
- **R_max clip 让 OPD 信号进 clip 家族**：连教师背书都要限幅——§4 的限幅器哲学一以贯之（深化文档 §6.2 同款论断）。注意 Eq.15 里 teacher 不 condition on effort e、student condition on e——教师给的是「该 prefix 下的标准答案分布」，学生学的是「在 effort 条件 e 下逼近它」（推断，基于公式形态；报告未展开解释此不对称）。
- **负结果**（逐字）：「While we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting」——与深化文档 §5.3「配方×容量共轭」互证：信号更细不必然更好。

---

## §7 [D] sandbox 即 infra：AgentENV 的经济学（sim [D] 回声）

### 7.1 为什么环境必须是 microVM

K3 §5.3.2 的开头给出了动机链（逐字）：

> As agents become more capable and tasks more difficult, they tend to explore more aggressively and may even attempt **reward hacking**. ... in our early experiments with traditional container-based sandbox runtimes, we observed several **kernel panics and deadlocks caused by unintended agent operations**. On the other hand, we want to permit as much exploration as possible so as not to constrain agent capability, and complex tasks require a sandbox close to a real-world environment — for example, agents should be able to mount disks, run containers, or even launch virtual machines at will. By running isolated microVMs with **Firecracker** [2], AgentENV provides a level of isolation and fidelity that container-based runtimes cannot match.

（[2] = Firecracker NSDI'20。）安全与保真度是**同一个需求的两个名字**：RL 训练会主动探索 reward hacking 路径（探索出 hacking 路径才知道 reward 哪里有洞），所以 sandbox 必须扛得住 agent 的任意操作，同时又要允许 agent 挂载磁盘、起容器、开虚机。**reward 设计的对抗性直接决定了 sandbox 的隔离等级**——这是 04 轨 harness engineering 与 01 轨 RL 的交点。

### 7.2 三个生命周期操作与一个算术

AgentENV 的三件套（K3 §5.3.2 逐字）：

- **Pause and Resume**：「a paused sandbox consumes no memory or CPU resources; a sandbox can therefore be paused while the agent is waiting for the model's inference result, which can account for **as much as 98% of the sandbox lifetime**」。增量 checkpoint/resume 低至 **133 ms / 49 ms**（「only memory pages dirtied since the last checkpoint are saved」）。
- **Fork**：「fork creates a new sandbox from the exact state of the original one while keeping the original running, which is useful for **reward judging without side effects**」——judge 在 fork 里检查环境状态，不污染原轨迹。
- **Snapshot**：定期快照做 error recovery。

sim [D] 的 declared 折算（98% 上界 + 133/49 ms 入算）：一条 1h 寿命轨迹，无 pause 占用 3600 资源·s；有 pause = 活跃 72 + 切换开销 40×0.182 = 79.28 资源·s，**超卖比 45.4×**（D1–D3：切换税 <1%，开销随切换次数线性）。报告实测口径是「a memory overcommit ratio of **up to 6.5×** in real workloads」——真实负载等待占比低于 98% 上界，故实测更低，方向一致（sim 输出已显式注明）。

这组算术解释了 §3 的一个悬案：**partial rollout 的「暂停轨迹下一迭代续跑」为什么付得起**——暂停本身几乎免费（暂停态零资源 + 恢复 49 ms），所以「保留未完成轨迹的完整环境状态」从奢侈操作变成默认选项。没有 133/49 ms 的增量 checkpoint，partial rollout 的续跑条款就是空头支票；反过来，没有 partial rollout 的需求，也没必要把 checkpoint 做到增量 133 ms。**算法与 infra 再次互为存在条件**。

规模侧：「tens of thousands of sandboxes... may need to be created within seconds」，用 OverlayBD 镜像格式（[67] DADI，USENIX ATC'20）+ 存储层共享 + P2P 传输做到亚秒级启动，CoW 内存 + page-cache 优化撑起超卖。全训练期 51,219,741 个 sandbox / 1,505,678 个镜像（§2 已引）。

### 7.3 开源双源核验

[AgentENV](https://github.com/kvcache-ai/AgentENV) 已公开。官方 README 称其用于 Kimi K3 的 agentic RL，并给出 snapshot-backed 环境的启动/恢复、暂停和生产内存超卖数据。README 的 9.6× 与 K3 报告的 6.5× 不是同一测量口径，因此必须分别标来源，不能合并成一个“官方常数”。

---

## §8 对照权威实现的取舍表

| 维度 | nano 侧（本仓库可运行锚） | K3/K2.5 生产选择 | 差异与原因 |
|---|---|---|---|
| rollout-训练耦合 | nano-slime L0–L2：lockstep vs 解耦的离散事件模拟，staleness 当模拟常数/旋钮 | partial rollout λ 暂停-续跑 + co-located 分时 | nano 把 λ 的权衡跑成算术（本文 sim [A]）；K3 的 λ 值未披露，本文不猜 |
| 权重同步 | nano-slime L2 §8 delta weight sync 代价模型（分卡路线）；nano-verl colocate+分时 | co-located：权重原地可见，问题变形为内存竞争（外部 KV 池 + NVMe offload + 自动节流） | §5.3 归纳：状态生命周期管理取代带宽问题 |
| off-policy 容忍 | nano-verl L1 IS + PPO clip；本文 sim [B] 三 loss 对照 | K2.5 Eq.1：log-ratio 窗口掩码（符号无关）+ 每 token 平方正则 | sim [B1]/[B2] 机器证明结构区别；窗口宽度是超参旋钮（[B3] 口径说明） |
| KV 管理 | nano-vllm-sglang L0–L3（03 轨）：paged KV、前缀缓存、radix | 外部 KV cache 池 write-back + KDA/MLA 双缓存联合管理 | 机制同构（准入/驱逐/复用）；K3 多了「跨迭代保留」与「prefill 回流风暴」两个 agentic 特有负载 |
| 长度控制 | 本文 sim [C]：free/budget/toggle 三臂 | per-problem b_0(x) + τ curriculum + −1 覆写 + GRM σ·ℓ_0 自动判负 + Toggle | nano 复现机制族动力学；K3 的 τ/σ 数值与 per-domain 配置未披露 [TODO: verify] |
| dense 信号 | nano-opd L0–L1：per-token OPD advantage | MOPD Eq.15：九教师 per-token reward + R_max clip | 深化文档 §6.2/§6.6 提供引用链；本文补 credit assignment 视角 |
| sandbox | 04 轨 agent-runtime L0/L1 覆盖 idempotency 与恢复面 | AgentENV：Firecracker microVM + 增量 checkpoint 133/49 ms + pause/fork/snapshot | nano 只覆盖契约，不宣称复刻 AgentENV 源码或生产性能 |

**nano 侧未做项（显式清单）**：KDA/MLA 双缓存、speculative decoding 与 prefix-block churn、MuonClip 参数层限幅、GRM 多 rubric 轮换（K2.5：「we employ multiple alternative GRM rubrics tailored to different task contexts」）、LLM Gateway 黑盒环境代理、OverlayBD 镜像栈——均为 K3/K2.5 报告正文机制，本文只作对照不作模拟（toy 尺度会扭曲其本质）。

---

## §9 费曼自检

### 9.1 讲给外行听

想象一个驾校同时教几千个学员练车（agentic RL）。三个难题：**其一**，有的学员练一圈要一小时（长时程轨迹），教练不能等所有人都练完才讲评——规矩改成「三成学员练完就先讲评，没练完的下节课接着练」（partial rollout）；代价是接着练的学员用的还是上节课教的开法（数据陈旧）。**其二**，为了让教练敢这么激进地插班讲评，评分规则必须防「学员已经换了开法、旧评分还照用」的失真——K3 的做法是给每个动作的「新旧开法差异」画一个窗口，差异出窗的动作直接不评分（不管教练当时夸还是骂），另外每个动作都交一点「别偏离原点开法太远」的保证金（per-token 正则）。PPO 的老规矩也画窗口，但窗口画在哪边取决于教练当时是夸是骂——教练的判断本身可能已经过时，所以 K3 换成与夸骂无关的窗口。**其三**，学员会钻空子：绕远路显得认真（verbosity hacking），规矩是「超出预算里程直接挂科」（−1 覆写）；但只卡里程又会教出「不敢走远路的司机」（length-overfitting），所以两档交替练（Toggle）。最后，每个学员的车（sandbox）在他等教练示范时（占 98% 时间）直接熄火入库、零油耗，轮到他就 49 毫秒点火——所以「把没练完的车留在场上」这件事几乎免费。四件事环环相扣：敢插班讲评是因为有防失真窗口；敢留车在场是因为点火便宜；留车在场又是插班讲评的前提。

### 9.2 思考题

1. K3 的 agentic 预算 T(y) 计「reasoning traces + tool-call arguments」。如果只计 reasoning traces 不计 tool-call arguments，verbosity hacking 会转移到什么形态？设计一个 toy 实验（在 sim [C] 上改一处）演示这个转移，并说明 σ·ℓ_0 自动判负规则要不要同步改。
2. sim [A] 的 A6 表明 partial rollout 不减少总生成时长。那么在什么条件下 λ 调小仍然**减少总训练墙钟**？（提示：训练与生成的重叠度、GPU 在同步方案里的空闲率；把「训练耗时」加回 toy——当前 toy 声明训练 0 slot，这是边界声明不是疏忽。）
3. [B3] 里 PPO 的翻转放大系数 0.90 < 1（翻转反而漂移更小）。给出一个机制解释（提示：翻转后 advantage 指向的行为与 behavior 策略的关系），并设计一个**能让 PPO 翻转放大 >1 的陈旧度结构**（提示：让 behavior 与初始策略不同——当前 toy 从 π_b 出发训练，漂移方向天然与掩码边同向）。
4. Toggle 的 Phase0 豁免条件是「组均 reward < λ 时不施加预算」。如果把豁免条件去掉（无条件施加），预测 hard 题会发生什么？用 sim [C] 验证（改一行），并对照 K2.5 原文「To prevent a premature sacrifice of quality for efficiency」解释这个条件存在的原因。
5. AgentENV 的 fork 用于「reward judging without side effects」。在什么任务上 judge **必须** fork 而不能只读产物文本？（提示：环境状态是 reward 的一部分——如代码 agent 的测试通过率依赖环境里留下的文件；fork 的成本结构 [增量快照] 为什么让这条路可行。）

### 9.3 反例（流行但错的说法）

1. **「partial rollout 加速了 rollout 生成」**：sim [A6] 机器证明总生成时长与 λ 逐位无关——它解耦的是训练触发时机，买的是训练频率，付的是陈旧度。加速生成是引擎与 sandbox 的事（§5/§7）。
2. **「PPO clip 已经解决了 off-policy 漂移，K2.5 的窗口只是换个写法」**：sim [B1]/[B2]——PPO 的掩码边随 A 符号翻转，最坏情形梯度 ∝r 无界（7.39）；K2.5 窗口符号无关、全网格有界（1.32）。在 advantage 符号不可信的 regime（跨迭代陈旧 + train-inference mismatch），「边界本身数据依赖」与「边界是 log-ratio 的纯函数」是两种保证等级。
3. **「给 reward 加长度惩罚项就能治 verbosity」**：惩罚项可被质量优势抵消（长而好的回答净 reward 仍为正，膨胀继续）；K3 的 −1 覆写与 σ·ℓ_0 自动判负是**结构性的**（长度规则先于质量生效，sim [C6]）。而且硬预算单用会制造 length-overfitting（sim [C3]/[C5]，K2.5 逐字命名）——治 verbosity 是个机制设计问题，不是调惩罚系数问题。

### 9.4 局限

- sim 为 toy 尺度（96 轨迹 / V=5×T=6 分解式策略 / 标量长度策略）：机制可迁移，量级不可外推；[A] 声明训练 0 slot、[D] 用 98% 上界入算，均为显式边界声明。
- K3 的 λ、τ、σ、α、β、R_max 等超参数值**报告未披露**，本文一律标 [TODO: verify]，不猜不拟合；sim 超参（α=0.75/β=1.35/τ=0.08/λ_tg=0.6 等）为 toy 自选并显式声明。
- K3 benchmark 数字表（Fig.1/Table 3 等）未逐项核验，本文未引用其 benchmark 分数。
- K2.5 Eq.1 的 Clip 语义按报告正文「gradient masking」描述实现（出窗梯度置零）；其与 CISPO 族的具体异同（窗口位置、双 clip 形态）未逐式对齐，标 [TODO: verify]。
- AgentENV 源码级核验（增量 checkpoint 实现、ublk 驱动）未做；本文引用边界仅为 K3 报告 + 官方仓库 README，不推断私有生产实现。

---

## §10 溯源与口径

### 10.1 一手来源快照（2026-08-31）

| 来源 | 版本/边界 | 本文用途 |
|---|---|---|
| [Kimi K3](https://arxiv.org/abs/2607.24653) | v2，2026-08-07；作者技术报告 | §2–§7 的 K3 架构、RL 与 AgentENV 声明 |
| [Kimi K2.5](https://arxiv.org/abs/2602.02276) | v2，2026-08-07；作者技术报告 | §4/§6 的 Eq.1、Toggle 与 Rollout Manager |
| [MOPD](https://arxiv.org/abs/2606.30406) | v1，2026-06-29 | 多教师 OPD 方法身份与配方语境 |
| [DeepSeek-V4](https://arxiv.org/abs/2606.19348) | 作者技术报告 | 白盒/黑盒信号源对照语境 |
| [OPD Survey](https://arxiv.org/abs/2604.00626) | v4，2026-06-18；持续更新的 survey | taxonomy 与开放问题，不作为单一生产事实的唯一证据 |
| [AgentENV](https://github.com/kvcache-ai/AgentENV) | 官方公开仓库；README 数字是仓库自述 | §7.3 的公开接口与 9.6× 口径 |
| [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) | 官方公开仓库 | 模型报告与发布入口 |

K3 参考文献身份映射：[2]=Firecracker NSDI'20 / [57]=Kimi K2 2507.20534 /
[60]=Kimi K2.5 2602.02276 / [67]=DADI(OverlayBD) ATC'20 / [74]=TM OPD
blog / [117]=Kimi K1.5 2501.12599 / [133]=MiMo-V2-Flash 2601.02780 /
[29]=DeepSeek-V4 2606.19348。K2.5 的 [50]=PPO 1707.06347。

**口径声明**：公式按 Eq.1/Eq.15 做教学化转写；报告数字、仓库 README 数字、
sim 输出与本文推断分别标注，不能互相替代。

### 10.2 复验方法

- 在两个新建空 CWD 中执行 §1 命令，要求 exit 0、stderr 为空、stdout `cmp`
  成功，且 self-check 为 27/27、digest 为 `1c75f9c845e1cd8b2681f2a205411bf3`。
- §1 输出块应与 stdout 全文一致；sim 无计时与本机路径，因此无需掩码。
- nano-slime、nano-verl 与深化文档只提供课程内交叉参照，不作为 K3 生产数字的证据。

### 10.3 四类信息区分

「原文声称」= 技术报告或官方仓库直接声明；「文献已有」= 已发表方法结论；
「推断」= 本文的机制归纳。仍未核验的边界包括：K3 超参数值
（λ/τ/σ/α/β/R_max）、benchmark 表逐项、K2.5 Eq.1 与 CISPO 逐式对齐、
AgentENV 源码行锚、K2.5 Toggle 实验表逐项。它们不进入已证事实表。
