# SOTA Deep-Dive 深化 — 后训练算法演进：可运行机制锚与 2026-08-13 重对齐

> **关系**：本文是 [post-training-algorithm-evolution.md](post-training-algorithm-evolution.md)（下称**主文档**，2026-08-11 首版）的深化 companion，不修改主文档正文。
> **状态**：✅ 首版完成并在 2026-08-13 重跑机制模拟与来源对齐。
> **SOTA 对齐日期**：2026-08-13（ROADMAP §八，B 层主题必做；主文档对齐日 2026-08-11）。18 个 arXiv ID 今日经 export.arxiv.org API 复验零漂移；4 篇一手报告全文今日现场重抓（Kimi K3 / MOPD / OPD survey v4 / DeepSeek-V4）；主文档 §10.3 五项 [TODO: verify] 闭合四项（详见 §6）。
> **sim 复现声明**：本文 §1 paste 块派生自 `post_training_evolution_sim.py`。2026-08-13 在两个新建空 CWD 中以 `python3 -B` 运行，均 EXIT=0、stderr 0 B、raw 82 行、self-check 24/24、digest `8ffa91ffddbf90d146516a65d9ee9285`，两次输出逐字节一致。该证据只覆盖当前 toy simulator，不外推生产训练吞吐或稳定性。

---

## §0 这篇深化是什么

主文档的机制论断锚在两类来源上：一手文献引文（18 个 arXiv ID + 两篇博客）与关联 nano 模块实测（nano-verl / nano-opd / nano-llamafactory，只读引用）。有两个缺口：

1. **GRPO 族变体修复的缺陷**（主文档 §4：ratio 粒度、长度偏置）只有摘要/博客层引文，没有本主题自己的实测锚——「token 级 ratio 更噪声」「sum 聚合 ∝ 长度」这些论断能不能在 toy 上直接跑出来？
2. **五个 [TODO: verify]**（主文档 §10.3：Kimi K3 正文层 / MOPD 正文层 / Qwen3.5 配方 / DeepSeek-V4 报告 / MOPD 同一性）——对齐两天后有没有机会闭合？

本文做两件事：

- **四机制面可运行 sim**（§1–§5）：`post_training_evolution_sim.py` 把 [A] IS+clip 地基、[B] GRPO 组基线、[C] ratio 粒度、[D] OPD 合流做成纯标准库、确定性、带 self-check 的本质模拟（ROADMAP §三 可运行性契约：toy 尺度演示机制、不外推量级）。这是 01 轨 deepdive 的 native sim，对齐 02/03 轨 deepdive 的既有形态（`deepseek_v3_mechanisms_sim.py` / `data_methodology_sim.py`）。
- **SOTA 重对齐**（§6–§7）：18 ID 复验零漂移 + 四篇一手报告全文重抓，闭合四个 [TODO: verify]，其中 DeepSeek-V4 的闭合强度超过原转述来源（V4 报告正文直接坐实），并录一次 survey v3→v4 版本漂移。

与主文档的分工：主文档讲「演进为什么走这条路」（三坐标轴 + 六机制面 + 2026 格局）；本文把其中「可以用算术和 toy 训练直接跑出来」的四件变成本主题自己的实测锚，并把摘要层证据推进到正文层。主文档正文不因本 companion 改写。

---

## §1 运行与输出

**可运行性契约声明（ROADMAP §三）**：本质模拟。toy 尺度（V=6 词表、T=8 位、分解式策略 = 每位置一个 softmax——「仍有 token 级 logprob 的最小语言模型」），纯标准库（math/random/hashlib），seed=3 固定，无计时行，跨运行逐字节一致。机制回声对象 = 主文档 §2（PPO）§3（GRPO）§4（GRPO 族）§6（OPD）；**全部数字为 sim 实测，论文侧只作机制对照、不外推量级**。关键算术用精确求和（分解式策略的 KL/期望有 O(T·V) 闭式；[D2] 缩小版全支撑 3^6=729 条枚举）——方法学继承 nano-opd L0「偏差不是噪声」。

```
$ python3 -B post_training_evolution_sim.py   # 任意空 CWD 可跑
```

以下为 2026-08-13 本轮实测输出（新建空独立 CWD，`-B`；两遍 BYTE-IDENTICAL，md5 `8638d805f1982569430b81a5a60f967e`/82 行）：

```
========================================================================
post-training evolution sim — IS+clip / GRPO baseline / ratio 粒度 / OPD
========================================================================
toy: V=6, T=8, 分解式策略(每位置 softmax) | seed=3 | 纯标准库
机制回声对象: deepdive §2(PPO) §3(GRPO) §4(GRPO族) §6(OPD)；
全部数字为本 sim 实测，论文侧只作机制对照、不外推量级。

[A] IS ratio + clipping：旧数据复用的算术地基
    精确期望（闭式）: E_old[f] = 3.774175  E_new[f] = 4.348107
    [A0] N=4000: mean(ratio) = 1.007439 (应≈1)
         IS 估计 E_new[f] = 4.403007 vs 精确 4.348107  (相对误差 1.26%)
    PASS  A0a E_old[ratio]≈1（旧数据复用的算术恒等式）  [mean_ratio=1.007439]
    PASS  A0b IS 重加权估计命中精确期望（<5%）  [est=4.4030 exact=4.3481]
    [A1] 同一旧批次 500 条 × 12 轮（lr=1.5, eps=0.2）:
         KL(old||cur) 第1/6/12轮  clipped   = 0.0283 / 0.0532 / 0.0691  | 批内最大 ratio 2.97
         KL(old||cur) 第1/6/12轮  unclipped = 0.0283 / 5.3351 / 1982.9741  | 批内最大 ratio 1617551.00
         真实目标 E_cur[f]: clipped 3.7742→4.1344 (+0.3602) | unclipped →6.8000 (+3.0258)
    PASS  A1a clipping 把信任域漂移压在 0.5 nats 内  [kl_clip=0.0691]
    PASS  A1b unclipped 漂移 > 2× clipped（限幅器在做事）  [1982.9741 vs 0.0691]
    PASS  A1c unclipped 批内最大 ratio 失控（> 2× clipped 的最大 ratio）  [1617551.00 vs 2.97]
    PASS  A1d clipped 复用仍能提升真实目标  [+0.3602]

[B] GRPO group baseline：偏移不变性（机器证明）+ 方差缩减（实测）
    [B0] 300 组 × G=16，reward 加 prompt 级常数 c：max|A − A'| = 2.22e-16
    PASS  B0 组基线对 reward 常数偏移精确不变（critic 要学的量，组均值免费消掉）  [max_diff=2.22e-16]
    [B1] 梯度估计组间离散 MSDEV: b=0 → 0.201647 | b=组均值 → 0.095846  (方差缩减 2.10×)
    PASS  B1 组均值基线降低梯度估计方差（>1.3×）  [ratio=2.10]
    [B2] 组均值基线的标准差: G=4 → 0.2548 | G=16 → 0.1235 | G=64 → 0.0586  (std64/std4 = 0.230, 理论 1/sqrt(16) = 0.250)
    PASS  B2 基线精度随 G 按 1/sqrt(G) 提升（采样预算换统计质量）  [decay=0.230]
    declared 算术（非实测，口径 = nano-verl tutorial_L3 §5 COST 模型，关联的后训练/预训练轨只读锚）:
    PPO 训练态 ≈ policy P + critic P + 两套 Adam m/v + 梯度 ≈ 8P；
    GRPO 去掉 critic 及其优化器态 → 4P 口径（nano-verl L3: 4P/N，7B@N=4 ≈ 34 GB/rank）。

[C] token 级 vs 序列级 ratio：粒度决定噪声与 clip 的对象
    陈旧度来源: [A1] 同款 clipped 训练续到 32 轮（KL(old||cur) = 0.0967 nats）
    [C0] 陈旧批次 N=2000（从旧策略采样，ratio = 当前策略/旧策略）:
         p95|log token ratio| = 0.2555 | p95|log seq ratio| = 0.1121  (token 级噪声宽 2.28×)
    [C0b] 单 token logprob 扰动 Δ=1.0: token ratio ×e^1=2.718 | seq ratio ×e^{Δ/T}=1.133  (传导系数比 = 8.0 = T)
    PASS  C0a token 级 ratio 噪声宽度 > 2× 序列级（几何平均吃掉单 token 噪声）  [2.28x]
    PASS  C0b 序列级对单 token 扰动的敏感性恰为 1/T（机器证明）  [transmit=8.0]
    [C1] eps=0.2: token 级 clip 比例 = 22.78% (3644/16000) | 序列级 clip 比例 = 0.00% (0/2000)
    PASS  C1 两种粒度 clip 的是不同对象（token 极端值 vs 序列整体漂移）  [tok=22.78% seq=0.00%]
    [C2] 每 token 优势同为 0.5: sum 聚合总推力 长/短 = 4.0/2.0 = 2.0（∝长度） | mean 聚合 = 0.50/0.50 = 1.0
    PASS  C2a sum 聚合使更新幅度 ∝ 响应长度（机器证明）  [ratio=2.0]
    PASS  C2b mean 聚合消除长度依赖（机器证明）  [ratio=1.0]

[D] OPD 合流：自采样+教师背书 → 锁模（opd_seq）；教师采样 → 掉谷（sft）；
    逐 token 配方的锁模需要学生有序列容量（opd_tok，实测对照）
    对照设置: 同一初始化（噪声±0.1 + mode-A 先验偏置 +0.2）、同 400 步、同 lr=0.1——信号配方 × 样本从谁采是变量
    [D0] opd_seq iter 100: 教师背书 = -0.8039 nats/token | 模质量 = 0.2400
    [D0] opd_seq iter 200: 教师背书 = -0.6442 nats/token | 模质量 = 0.3836
    [D0] opd_seq iter 300: 教师背书 = -0.6090 nats/token | 模质量 = 0.4145
    [D0] opd_seq iter 400: 教师背书 = -0.5900 nats/token | 模质量 = 0.4249
    [D1] 终局对照（同初始化/同 400 步/同 lr）:
         opd_seq(自采样+序列级adv): 背书 -2.9911→-0.5880 nats/token | 模质量 0.000002→0.4249 | 驻留率 = 0.809
         opd_tok(自采样+逐token adv): 模质量 →0.001352 | 驻留率 = 0.021
         sft(教师采样+NLL):          模质量 →0.002889 | 驻留率 = 0.036
         模质量比 opd_seq/sft = 147×
    实测发现（配方×容量共轭）: 逐 token 配方在无上下文学生上收敛到 mode-covering
    不动点（模质量与 sft 同量级）——TM 配方的锁模预设学生本身是序列模型
    （nano-opd L1 的真实 Transformer 学生即如此，token 级 OPD 锁 codebook）。
    PASS  D0a opd_seq 锁定教师的一个模（模质量 > 0.2）  [mass=0.4249]
    PASS  D0b opd_seq 采样驻留在模上（驻留率 > 0.5）  [dwell=0.809]
    PASS  D0c opd_seq 教师背书度大幅抬升（> +1.5 nats/token）  [-2.9911→-0.5880]
    PASS  D1a off-policy SFT 掉谷：opd_seq/sft 模质量比 > 30×  [147x]
    PASS  D1b 逐 token 变体在无上下文学生上不锁模（实测，模质量 < 0.05）  [mass_tok=0.001352]
    [D2] 缩小版（V=3, T=6，全支撑 729 条精确求和）:
         reverse KL 最优点 q*（锁B）: 模质量 = 0.5309 | KL(q*||p) = 0.6942
         对照：边缘匹配解（SFT 极限）KL = 4.5047  (reverse KL 偏好锁模解 6.5×)
         锁模点上: |g_right(学生采样)| = 0.002604（近驻点，模待得住）
                   |g_wrong(教师采样)| = 31.9413（偏差比 = 12265×）
         g_wrong 在谷方向(A模 token)分量和 = -55.749（负 = 梯度下降会把质量推向另一个模 = mode covering → 掉谷）
    PASS  D2a reverse KL 在受限学生族里的最优解是锁模（模质量 > 0.5）  [mass=0.5309]
    PASS  D2b reverse KL 偏好锁模解而非边缘匹配（KL 比 > 3×）  [4.505 vs 0.694]
    PASS  D2c 正确估计器在锁模点近驻点（|g| < 0.01）  [|g|=0.002604]
    PASS  D2d 教师采样估计器有系统性大偏差（>10× 真梯度）  [12265x]
    PASS  D2e 偏差方向指向谷（mode covering，分量和 < −0.1）  [valley_grad=-55.749]

[E] self-check
    ✅ self-check passed (24/24)

digest(md5 of metrics) = 8ffa91ffddbf90d146516a65d9ee9285
```

self-check 24/24 全过，digest `8ffa91ffddbf90d146516a65d9ee9285`（md5 of metrics，跨运行不变）。下面逐面解读。

---

## §2 [A] IS + clipping：旧数据复用的算术地基（回声主文档 §2）

### 2.1 [A0] 无偏性：一个算术恒等式的定量验证

`mean(ratio) = 1.007439`（应≈1）与 `IS 估计 4.403007 vs 精确 4.348107`（相对误差 1.26%）验证的是同一件事：

```
E_{y~π_old} [ (π_new(y)/π_old(y)) · f(y) ] = E_{y~π_new} [ f(y) ]
```

从旧策略采的样本，乘上 ratio，就能无偏地估计新策略下的期望——**这是「同一批 rollout 更新多轮」的全部算术基础**，PPO 用它、GRPO 用它、OPD 也用它（主文档 §2.1）。注意它只修正**同一 prefix 上的动作分布**：prefix/state 仍是旧策略产生的，ratio 不自动修正这部分（主文档 §3.3 的 IS 边界表述在 sim 层面的对应）。

### 2.2 [A1] 限幅器在做事：KL 漂移 0.0691 vs 1982.9741 nats

同一旧批次 500 条 × 12 轮更新：clipped 的 KL(old||cur) 停在 0.0691 nats、批内最大 ratio 2.97；unclipped 漂到 1982.9741 nats、最大 ratio 1,617,551——**四个数量级的差距，全部来自 `min(r, 1±ε)` 这一个操作**。clipping 把「策略离开采样分布的速度」限住了，这就是主文档 §2.1「防止某条数据把策略拉得太远」的定量形态。

### 2.3 诚实边界：为什么 unclipped 的 toy 目标反而更高

输出里有一个反直觉数字必须讲清：unclipped 的真实目标 E_cur[f] = 6.8000（+3.0258），clipped 只有 4.1344（+0.3602）——**不裁剪反而「更好」？**

不是。这个 toy 的 reward f 是**固定的 ground truth**（逐 token 可加函数，闭式可算），unclipped 大步漂移没有可以钻的空子：它优化的是真目标本身，所以到得更快。真实管线里 clip 防的恰恰是另一种情况——**目标本身只是局部可信的估计**：advantage 来自 critic/组基线（有偏差有方差），reward 来自 RM 或验证器（有拟合误差），策略一旦离开采样分布太远，被最大化的就是这些**估计器的误差**而不是真目标（reward hacking 的机制内核）。toy 没有建模估计误差，所以 unclipped 的代价只显示为「KL 漂移与 ratio 方差失控」这两个观测量，而不显示为「目标被钻空子」。

这正是本 sim 的**边界声明**（可运行性契约要求的显式注明）：[A1] 隔离的机制是「限幅器控制离开采样分布的速度」；「为什么离开太快是危险的」需要估计误差建模，不在本 toy 范围内。思考题 1 接这个边界。

---

## §3 [B] GRPO 组基线：critic 要学的量，组均值免费消掉（回声主文档 §3）

### 3.1 [B0] 偏移不变性是机器证明，不是实测

300 组 × G=16，每个「prompt」加不同的难度常数偏移 c：`max|A − A'| = 2.22e-16`（浮点噪声级）。`A_i = R_i − mean(R)` 对 prompt 级常数偏移**精确不变**——reward 模型的基线偏差、任务难度差异、prompt 固有的「容易/难」，这些 critic 需要花参数去学的量，在组内被一次减法精确消除。这是「组内均值 = 穷人版 critic」（主文档 §3.1 逐字）的精确含义：不是近似替代，是在**偏移这一维上严格等价**。

### 3.2 [B1]/[B2] 「采样换参数」的统计账

- **方差缩减**（实测）：同一批样本，b=0 vs b=组均值，梯度估计的组间离散 MSDEV 0.201647 → 0.095846，**2.10×**。基线不只消偏移，还降方差。
- **基线精度账**（实测）：组均值基线自身的标准差 G=4 → 0.2548、G=16 → 0.1235、G=64 → 0.0586，`std64/std4 = 0.230`，理论值 1/sqrt(16) = 0.250（偏差来自 400 组有限样本对两个 std 的估计噪声，量级 1/sqrt(400) ≈ 5%，见思考题 2）。

合起来的账：组基线的精度按 1/sqrt(G) 提升，代价是每组多采 G 个 response。**参数（critic）换成了采样（rollout）**——而采样恰恰可以被高吞吐推理引擎加速（主文档 §3.1/§7 的论点；nano-vllm-sglang 阶梯就是采样侧 infra 的可运行对照）。

### 3.3 declared 算术（显式声明：非实测）

sim 输出的 8P→4P 显存账（PPO 训练态 ≈ policy + critic + 两套 Adam m/v + 梯度 ≈ 8P；GRPO 去 critic 及其优化器态 → 4P 口径）引自 nano-verl `tutorial_L3.md` §5 的 COST 声明模型，并与主文档 §3.1 的口径一致。这是 declared 折算，不是 GPU 实测；不得据此推断真实硬件峰值。

---

## §4 [C] ratio 的粒度：GRPO 族变体在修什么的实测版（回声主文档 §4）

主文档 §4 按 §八 C 层纪律只教机制类别不追方法名；这一节把两个机制类别做成实测。

### 4.1 陈旧度的日常形态

[C] 的陈旧度来源是「[A1] 同款 clipped 训练在同一旧批次上续到 32 轮」（KL(old||cur) = 0.0967 nats）——**策略持续训练而 rollout 批次不刷新**。这不是极端假设：同步 RL 管线里每个 train step 之间、partial rollout 的跨迭代轨迹（Kimi K3 的百万 token 轨迹「naturally spans multiple iterations, introducing data staleness」，§6.2 引文）面对的都是同一种日常陈旧度。

### 4.2 [C0]/[C0b] token 级噪声 vs 序列级噪声

- **实测**：陈旧批次上 p95|log token ratio| = 0.2555 vs p95|log seq ratio| = 0.1121——**token 级噪声宽 2.28×**。序列级 ratio = token ratio 的几何平均（长度归一），单 token 的logprob 抖动被平均吃掉。
- **机器证明**：单 token logprob 扰动 Δ=1.0，token ratio 乘 e^1 = 2.718，序列 ratio 只乘 e^(Δ/T) = 1.133——**传导系数比恰为 T = 8**（1/T 敏感性是定义层面的恒等式，不是拟合）。

这就是 GSPO「token-level optimization objective is noisy」（官方博客逐字，今日重抓核验在位，§6.1）的 toy 定量形态。

### 4.3 [C1] 同一批、同一 ε：两种粒度 clip 的是不同对象

eps=0.2 下：token 级 clip 比例 22.78%（3644/16000）vs 序列级 0.00%（0/2000）。token 级 clip 抓的是**单 token 极端值**（陈旧批次里个别位置的概率剧变），序列级 clip 抓的是**序列整体漂移**（几何平均后的系统性偏离）——同一个 ε，两种粒度拦截的是不同形态的失真。

**诚实边界**：哪个粒度的 clip 比例更高**依赖陈旧度结构**，不是恒定的方向。本 toy 的陈旧度是「逐 token 噪声主导、序列级漂移相互抵消」，所以序列级 clip 率为 0；GSPO 官方博客报告的「被 clip 的 token 比例比 GRPO 高两个数量级」（今日重抓在位）是他们训练条件下的实测——真实长训里序列级似然可以有系统性方向漂移，序列级 clip 一旦触发就整条序列 T 个 token 计入。两边不矛盾：机制是「clip 的对象随粒度而变」，方向是条件性的（思考题 3）。clip 频率不是病，**clip 在错误的粒度上才是**（主文档 §4.1 博客引文的准确读法）。

### 4.4 [C2] 长度偏置：聚合归一决定长度是否进入梯度（Dr. GRPO 机制类别）

每 token 优势同为 0.5、长响应 T 位 vs 短响应 T/2 位：**sum 聚合总推力 4.0 vs 2.0 = 2.0×（∝ 长度）**；**mean 聚合 0.50 vs 0.50 = 1.0（长度无关）**——两个都是 1e-12 级机器证明。

这解释了 Dr. GRPO 的批评「artificially increases response length (especially for incorrect outputs)」（摘要原文，主文档 §4.1）的机制位置：**长度偏置不是模型「想变长」，是 sum 聚合把每 token 优势累加成了 ∝ 长度的总推力**——错误回答只要每 token 优势不为零，越长推力越大。修复方向（mean/长度归一聚合）在 sim 里是一行代码的事；它成为变体爆炸的 C 层现象，是因为真实系统里这个选择与 clip 粒度、基线设计耦合在一起（主文档 §4.2 的两个问题坐标）。

---

## §5 [D] OPD 合流：同一个 IS loss，reward 换成教师背书（回声主文档 §6）

### 5.1 设置：教师自回归双峰，学生分解式

教师是**序列级双峰**的自回归分布（两个交替 pattern A/B，一旦前缀定模就 0.9 概率沿模走，纯模序列 p≈0.215、混合序列概率指数级小）；学生是**分解式**策略（每位置独立 softmax，无跨位置关联）——学生族被故意限制，与 nano-opd L0「单峰受限学生」同源：在受限族里，机制差异才会以终局形态（锁模 vs 掉谷）显形。三臂同初始化（噪声 ±0.1 + mode-A 先验偏置 +0.2，显式声明：真实学生经 SFT 预热后总带先验；机制主张 = OPD 把小偏置放大成整模锁定）、同 400 步、同 lr=0.1，**唯一变量 = 信号配方 × 样本从谁采**。

### 5.2 三臂终局

| 臂 | 配方 | 模质量 | 驻留率 | 教师背书 |
|---|---|---|---|---|
| opd_seq | 学生自采样 + 序列级 adv（= 负 reverse KL，MiniLLM 配方 + GRPO 组基线） | 0.000002 → **0.4249** | **0.809** | −2.9911 → −0.5880 nats/token |
| opd_tok | 学生自采样 + 逐 token adv（TM 配方的逐字机制） | 0.001352 | 0.021 | — |
| sft | 教师采样 + 学生 NLL（off-policy 蒸馏） | 0.002889 | 0.036 | — |

- **opd_seq 锁模**：模质量比 sft **147×**，采样 80.9% 驻留在模上，教师背书度抬升 +2.40 nats/token——主文档 §6.1 的算术定理（「期望的下标是 q_θ，样本就必须从 q_θ 采」）在序列级训练上的终局形态。
- **sft 掉谷**：教师采样 + NLL 把质量堆进逐位置边缘（两个模的平均 = 谷），与 nano-opd L0 三配方对照（sft 驻留率 0.054）同构。
- **[D2] 驻点算术的序列级同源对照**：缩小版（V=3、T=6，全支撑 729 条精确求和、零采样）——reverse KL 最优点锁模（模质量 0.5309），KL(q*||p) = 0.6942 vs 边缘匹配解 4.5047（**reverse KL 偏好锁模解 6.5×**）；锁模点上正确估计器（学生采样）|g| = 0.002604（近驻点，模待得住），教师采样估计器 |g| = 31.9413（**偏差比 12265×**），且谷方向分量和 −55.749（负 = 梯度下降会把质量推向另一个模 = mode covering → 掉谷）。nano-opd L0 驻点算术（+0.002 vs +136.149）的序列级复现：**偏差不是噪声，在序列级同样成立**。

### 5.3 实测发现：配方 × 容量共轭（本文的一等新结论）

**opd_tok 在分解式（无上下文）学生上不锁模**（模质量 0.001352，与 sft 同量级）。机制原因（推断，与输出和 TM 配方文本一致）：TM 的逐 token advantage 是 `logp_teacher(y_t|y_{<t}) − logq_student(y_t|y_{<t})`——**它预设学生的条件分布依赖上下文**。分解式学生的 `logq_t(y_t)` 无前缀依赖，逐 token 信号退化为逐位置边缘匹配，不动点就是 mode covering（与 sft 极限重合）。

用主文档 §6 的语言说：**TM 配方的锁模预设学生本身是序列模型**。nano-opd L1 的真实 Transformer 学生有序列容量，所以 token 级 OPD 锁模（锁 codebook B，主文档 §6.2）；本 sim 的分解式学生没有，所以同一配方瘫在谷里。配方不变、容量变了、终局就变了——**信号粒度与学生容量是共轭变量，不存在「逐 token 严格优于序列级」的线性排序**（反例 2 接住）。这个发现与 §6.2 的 Kimi K3 一手负结果（top-k 细粒度蒸馏目标「no clear advantage」）相互印证。

---

## §6 SOTA 重对齐（2026-08-13）：零漂移 + 四项 [TODO: verify] 闭合

### 6.1 复验结果

- **18 个 arXiv ID**（主文档 §10.1 表全量）：export.arxiv.org API 单批复验（抓取件 90,577 B），标题/首发日期/修订日期与主文档录值**逐词零漂移**（含 2607.24653 v2 upd 2026-08-07、2604.00626 v4 upd 2026-06-18、2501.12948 v2 upd 2026-01-04 等全部版本录值）。对齐结论维持：**未发现取代 GRPO 族 / RLVR / OPD 的新一代范式**，且证据面比 08-11 更厚（下）。
- **两博客重抓**：GSPO blog 25,141 B、TM OPD blog 82,782 B——与 08-11 抓取件**同尺寸**；主文档所引三句 GSPO 引文与两句 TM 引文今日现场 grep 全部逐字在位（GSPO「noisy and inefficient」句页面源形为 `GRPO&rsquo;s`，HTML 实体归一后逐字吻合）。

### 6.2 Kimi K3 正文层 — 闭合（主文档 §10.3 项 1）

ar5iv 全文重抓（1,307,024 B）。主文档 §7.4 停在摘要层的论断全部推进到正文层（以下均今日抓取文本逐字）：

- **post-training 主干含 MOPD**：「We adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate these domain-specialized capabilities across varying reasoning efforts into a unified model [74, 133, 29]」，教师为「the corresponding teacher model among the **nine experts**」之一。
- **OPD advantage 进 clipping 家族**：per-token OPD reward（Eq.15）带 stop-gradient 与 clipping threshold τ——「a clipping threshold to constrain extreme advantage signals, thereby stabilizing RL training」。PPO 地基机制（主文档 §2）在 OPD 信号上回归：连教师背书都要限幅。
- **partial rollout 与陈旧度的生产处理**：「an individual long-horizon trajectory naturally spans multiple iterations, introducing data staleness that threatens training stability. Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a **per-token regularization**. By constraining policy updates within a localized neighborhood...」——主文档 §3.3 的 IS 边界讨论在百万 token 轨迹上的生产形态；infra 侧「co-located system combines partial rollouts, external KV-cache retention, adaptive throttling and **resumable microVM sandboxes** to preserve long-lived model and environment state」（主文档 §7.4「persistent rollout and sandbox states」的机制展开）。
- **长度/overthinking 控制**：per-problem token budget + 「budget-based verbosity control...automatically loses the binary comparison」——Dr. GRPO 长度偏置机制类别（本文 §4.4）的生产对策。
- **负结果**：更细粒度 top-k 蒸馏目标「no clear advantage in either convergence speed or final performance in our setting」（与本文 §5.3 配方×容量共轭互证）。
- MOPD 引用 = [74] TM blog + [133] MiMo-V2-Flash [2601.02780] + [29] DeepSeek-V4 [2606.19348]；**不引用 MOPD 论文 [2606.30406]**（身份问题见 §6.4）。

### 6.3 DeepSeek-V4 — 闭合，且强于原转述来源（主文档 §10.3 项 4）

主文档所录「DeepSeek-V4 采用 pure multi-teacher OPD」原系 survey 正文转述（nano-opd §15 已录 [TODO: verify]）。今日直接核验 V4 报告本身 [2606.19348]（DeepSeek-AI，2026-04-26，ar5iv 864,187 B），**坐实强度超过转述**（以下逐字）：

- §5.1：「the mixed Reinforcement Learning (RL) stage was **entirely replaced by On-Policy Distillation** (OPD)」——「pure」的直接一手形态。
- §5.1.2：「we employ **multi-teacher** On-Policy Distillation (OPD) as the primary technique for merging expert capabilities into the final model」；「Computing the **reverse KL** loss requires **sampling training trajectories from the student** to maintain on-policy learning」——主文档 §6.1 算术定理的逐字回声出现在生产报告里；「**more than ten teacher models** covering various domains are employed to distill a single student model」。
- **token 级 vs 全词表的显式取舍**（本文 §5.3 的生产对应）：「prior works usually simplify the full-vocabulary KL loss into a **token-level KL estimate**...reuse RL framework by replacing [it] as the per-token advantage estimate...Although this approach is resource-efficient, it leads to **high variance in gradient estimation** and often causes training instability. Therefore, we adopt **full-vocabulary logit distillation**」。
- §5.2.2 为全词表 OPD 付出的 infra 代价（§7 展开）：teacher 权重中心化分布式存储 + ZeRO-like 分片按需加载；只缓存 teacher **末层 hidden states**、训练时过 prediction head 重建全词表 logits（避免 V 维 logits 物化）；样本按 teacher index 排序，每 mini-batch 至多一个 teacher head 驻显存。
- OPD 引用 = Lu and Lab (2025)（TM blog）+ Gu et al. (2024)（MiniLLM，ICLR 版题「Knowledge Distillation of Large Language Models」）——与主文档 §6 的 A 层锚点吻合。

**版本漂移录值（survey v3 → v4）**：今日 survey v4 全文 grep「DeepSeek-V4」= **0 命中**——v3 的转述句在 v4（2026-06-18 修订）中已移除。原 claim 现改锚 V4 报告本身（更强的来源），survey 转述口径作废。

### 6.4 MOPD 正文层 + 同一性 — 闭合（带对主文档的更正）（主文档 §10.3 项 2/5）

- **正文层**：MOPD 论文 [2606.30406] ar5iv 全文重抓（204,940 B）。摘要逐字坐实主文档 §6.3 所引（per-domain RL teachers → 「distill these teachers into the student **on its own rollouts**」/「eliminates exposure bias and provides a dense optimization signal」/ 优于 Mix-RL、Cascade RL、Off-Policy Finetune、Param-Merge / deployed in MiMo-V2-Flash）。新增：「enables parallel, independent development of domain teachers, removing the cross-domain coupling」。
- **署名**：Peking University + **LLM Core, Xiaomi** + HKU + Renmin University（footnote: work done during internship at Xiaomi）。
- **同一性裁决证据链**：MiMo-V2-Flash 报告 [2601.02780]（Xiaomi LLM-Core Team，2026-01-06，v2 01-08，abs 页今日核验）摘要**自己命名**该范式：「introduces a novel Multi-Teacher On-Policy Distillation (**MOPD**) paradigm...domain-specialized teachers...provide dense and token-level reward」。MOPD 论文出自同一团队（Xiaomi LLM Core），是同一生产技术的论文化。**同一性成立**。
- **对主文档 §6.3 括注的更正**：「survey v3 方法表所载同名 MOPD」——今日 survey v4 全文 grep「MOPD」= **0 命中**；其方法表 MiMo-V2-Flash 行写作「Multi-teacher logit + reward」并引 MiMo-V2，**从未使用 MOPD 名**，且 v4 修订（06-18）早于 MOPD 论文（06-29）11 天、未引用。同一性的依据是「团队署名 + 技术内容 + 时间线」，不是 survey 命名。（主文档正文冻结不改，更正录于本文。）

### 6.5 Qwen3.5 配方细节 — 仍 [TODO: verify]（负结果录值）

export.arxiv.org API 检索（all:"Qwen3.5" AND ti:"Technical Report"，今日）仅返回 2604.15804（Qwen3.5-Omni，全模态主题，与主文档录值一致）。base Qwen3.5 的 post-training 配方**无 arXiv 一手文本**，qwen.ai 博客 JS 渲染维持——该 [TODO] 以负结果形态延续。

### 6.6 引用链：合流线的文献学证据

```
MiniLLM [2306.08543, 2023-06]（A 层经典锚点）
   → TM OPD blog [2025-10-27]（生产化配方：per-token adv = 负 reverse KL + RL IS loss）
      → MiMo-V2-Flash [2601.02780, 2026-01]（命名 MOPD，token 级 dense reward）
         → DeepSeek-V4 [2606.19348, 2026-04]（mixed RL 全换 OPD；引 TM blog + MiniLLM；全词表路线）
            → MOPD 论文 [2606.30406, 2026-06]（Xiaomi 论文化）
            → Kimi K3 [2607.24653, 2026-07]（9 教师 MOPD + clip；引 TM + MiMo + DSV4）
```

四个前沿实验室（Thinking Machines / Xiaomi / DeepSeek / Moonshot）收敛于同一机制族：**学生自采样 + reverse KL + 教师 dense 信号**（主文档 §1 合流线的文献学坐实）；实现分叉在信号粒度——token 级 advantage（TM / Kimi K3）vs 全词表 KL（DeepSeek-V4），即本文 §5.3 配方×容量共轭在生产尺度上的两个分支。

---

## §7 算法-infra 共演化补遗（2026-08-13 一手）

主文档 §7 的论断「算法选择直接增删 infra 步骤」今日获得两个新的正文级锚点：

1. **信号粒度决定 infra 形态**（对照主文档 §7.2 的 GSPO/verl 案例）：token 级 OPD（TM/K3 路线）复用 RL 框架本身——TM「call the RL importance-sampling loss function」逐字、K3「dense reward signal seamlessly integrates into our RL framework, naturally enabling...partial rollout training」；全词表 OPD（DSV4 路线）则必须新建一整套 teacher 调度 infra（§6.3 所列 §5.2.2 三件：权重按需加载 / hidden states 缓存重建 logits / teacher index 排序）。**同一个算法家族，信号粒度的选择直接决定要不要建「teacher  logits 供给系统」这个子系统**——这是「算法-infra 共演化」在 OPD 内部的可指认形态。
2. **长时程轨迹把 rollout 状态变成 infra 对象**（对照主文档 §7.4）：K3 的 partial rollout + external KV-cache retention + resumable microVM sandboxes，与 04 轨 harness engineering 的状态外化主题（主文档 §7.4 末段）在 infra 层合流——训练侧持久化 rollout 状态、推理侧外化 agent 状态，是同一枚硬币的两面（ROADMAP §一 RSI 闭环）。

---

## §8 费曼自检

### 8.1 讲给外行听

接着主文档 §9.1 的学徒厨师比喻：这一篇深化讲的是**打分方式的两场争论**。第一场：给学徒的每道菜打分，是**按每一刀评**（token 级）还是**按整道菜评**（序列级）？按每一刀评，评语噪声大——某一刀切歪了不代表这道菜差（sim [C]：token 级 ratio 噪声宽 2.28×）；按整道菜评，一道菜差就整道否掉，但学徒不知道差在哪一步。更微妙的是：按每一刀评**要求学徒记得自己之前切了什么**——如果学徒根本没有记忆（分解式学生），逐刀评语就退化成「每个位置的平均水平」，学徒永远学不会完整的菜（sim [D]：opd_tok 不锁模）。第二场：多个师傅各教一道拿手菜，怎么合到一个学徒身上？答案不是把师傅的菜谱抄一遍（off-policy 蒸馏掉谷，147× 差距），而是**学徒自己做、师傅们围着评**（multi-teacher OPD）——DeepSeek 家甚至要求师傅把每个可能动作的完整评分单都写出来（全词表 KL），为此专门建了一套传菜系统（§5.2.2 infra）；Moonshot 家则让师傅只评实际发生的那一步、但加了「评语过激要截断」的规矩（K3 的 clip）。而贯穿始终的那把限幅器（PPO clipping）没人拆掉——连师傅的评语都要经过它（K3 Eq.15 的 τ）。

### 8.2 思考题

1. [A1] 里 unclipped 的真实目标（6.8000）比 clipped（4.1344）高得多——这是不是说明 clipping 没必要？如果不是，指出这个 toy 没有建模的真实因素，并设计一个能把该因素加回 toy 的最小改动（提示：§2.3 的边界声明；advantage 估计误差可以显式注入——例如用另一个噪声函数代替 f 作为「被优化的目标」，用 f 作为「真目标」）。
2. [B2] 实测 decay 0.230 vs 理论 0.250。理论值为什么与 reward 的伯努利参数 p 无关（先证这个），再把 8% 的相对偏差归因到有限样本（提示：每个 std 由 400 组估计，1/sqrt(400) ≈ 5% 量级，两个 std 之比如何传播误差）。
3. [C1] 里序列级 clip 比例 0.00%，而 GSPO 官方博客说「被 clip 的 token 比例比 GRPO 高两个数量级」。这两个观察矛盾吗？（提示：§4.3 的边界声明——clip 比例的相对高低依赖陈旧度结构；博客数字是其训练条件下的实测；机制主张是「clip 的对象随粒度而变」而非「某粒度恒 clip 更多」。若要在本 sim 里复现博客方向，应制造什么样的陈旧度？）
4. opd_tok 在分解式学生上不锁模。给学生加一个 n-gram 上下文（容量介于分解式与全 Transformer 之间），预测会发生什么，并用 [D2] 的精确求和框架（V=3、T=6 可扩到 bigram 学生族）设计验证「容量阈值」假设的实验。你的预测依据是什么机制？（提示：§5.3 的共轭论证；逐 token advantage 需要学生的条件分布对前缀敏感。）
5. DeepSeek-V4 选全词表 KL、TM/Kimi K3 选 token 级 advantage。什么条件下全词表的额外 infra 成本（§5.2.2 三件套）值得？结合 K3 的 top-k 负结果（§6.2）与 sim [D] 两臂，给出你的判据（提示：梯度方差收益 vs teacher head 存储/重建成本；学生容量与任务对分布尾部敏感度的作用）。

### 8.3 反例（流行但错的说法）

1. **「OPD 就是蒸馏换了个新名字」**（主文档 §9.3.2 的 2026 加强版）：今日四家一手报告（TM / MiMo / DSV4 / K3）的实现全部要求学生自采样——DSV4 逐字「sampling training trajectories from the student to maintain on-policy learning」；本文 [D1] 同教师同学生，自采样锁模 vs 教师采样掉谷 147×。off-policy 蒸馏与 OPD 是两件事，生产界用四份独立实现投了票。
2. **「token 级 OPD 是序列级的严格细化，严格更好」**：本文 §5.3 实测发现——无上下文学生上 token 级配方不锁模（0.001352 vs 0.4249），K3 一手负结果 top-k 细粒度目标「no clear advantage」。信号粒度与学生容量共轭，不存在线性优劣序。
3. **「GRPO 族变体在竞争谁更新更强，应该追最新的」**（主文档 §9.3.3 的实测版）：四个 B 层变体修复的缺陷全部投影到本文 [C] 面的两个坐标（clip 粒度 / 聚合归一），且都能用 ≤10 行算术复现（C0b/C2 是机器证明）。值得教的是前沿模型采用证据，不是论文新旧。

### 8.4 局限

- sim 为 toy 尺度（V=6/T=8 分解式策略、纯标准库）：机制可迁移，量级不可外推；[A1] 的 unclipped 优势反转是 toy 边界（§2.3），[C1] 的 clip 比例方向是条件性的（§4.3）。
- Kimi K3 / MOPD / DeepSeek-V4 正文核验到机制节（post-training / OPD / infra 节逐字摘取）；三篇的 benchmark 数字表未逐项核验，引用时仍以各报告摘要/正文标注为准，标 [TODO: verify]。
- MOPD 论文实验表（Qwen3-30B-A3B 对比数字）本轮核验止于摘要+引言+署名层，正文实验表逐项核验标 [TODO: verify]。
- survey v4 与 v3 的差异录值限于「DeepSeek-V4 / MOPD 两个字符串的存废」，未做 v3/v4 全文 diff（v3 抓取件在 nano-opd §15 所录 08-05 workspace，非本轮现场件）。
- nano-verl L3 的 COST 数字属于声明式估算；本文只读引用，未把它升级为 GPU 实测。

---

## §9 溯源与口径

### 9.1 一手来源（2026-08-13 核验快照）

下表记录核验时的来源、字节数和用途；抓取件未随课程发布，应按 URL 和日期重新获取。

| 来源 | 通道 | 抓取件录值 | 用途 |
|---|---|---|---|
| arXiv API ×18 ID | export.arxiv.org/api/query（HTTPS 单批） | arxiv_18ids.xml，90,577 B | §6.1 零漂移复验 |
| Kimi K3 [2607.24653] | ar5iv 全文 | kimik3_ar5iv.html，1,307,024 B | §6.2 正文层闭合 |
| MOPD [2606.30406] | ar5iv 全文 | mopd_ar5iv.html，204,940 B | §6.4 正文层 + 署名 |
| OPD survey [2604.00626] v4 | ar5iv 全文 | opd_survey_ar5iv.html，640,235 B | §6.3/§6.4 版本漂移取证 |
| DeepSeek-V4 [2606.19348] | ar5iv 全文 + abs 页 | dsv4_ar5iv.html，864,187 B；abs_2606.19348.html，93,396 B | §6.3 直接一手闭合 |
| MiMo-V2-Flash [2601.02780] | abs 页 | abs_2601.02780.html，62,237 B | §6.4 MOPD 命名证据 |
| Kimi K2.5 [2602.02276] | abs 页 | abs_2602.02276.html，94,873 B | §6.2 RL 算法出处（K3「follows the algorithm in Kimi K2.5」） |
| GSPO blog | qwenlm.github.io/blog/gspo/ | gspo_blog.html，25,141 B（与 08-11 同尺寸） | §6.1 引文复验 |
| TM OPD blog | thinkingmachines.ai/blog/on-policy-distillation/ | tm_opd.html，82,782 B（与 08-11 同尺寸） | §6.1 引文复验 |
| Qwen3.5 检索 | export.arxiv.org API search | arxiv_qwen35.xml，3,289 B | §6.5 负结果 |

### 9.2 内部锚点口径

- **sim**：`post_training_evolution_sim.py` self-check 24/24；两次独立空 CWD 输出逐字节一致，边界见文首声明。
- **nano 交叉引用**：nano-verl 的 COST 是 declared model；nano-opd 的驻点算术与 2×2 因子设计来自对应教程。本文不把这些跨模块数字升级为新的独立实测。
- **主文档关系**：本文对主文档的补充只记录在本 companion 中，便于读者区分首版叙事与深化证据。

### 9.3 四类信息区分

「摘要/原文声称」= 今日或 08-11 抓取文本逐字；「文献已有」= 已发表结论；「推断」= 本文作者的机制推断（§2.3 边界解读、§5.3 机制原因、§6.6 合流线归纳，已逐处标明）；无猜测级内容入正文。[TODO: verify] 遗留四项：Qwen3.5 配方（负结果延续）、K3/MOPD/DSV4 benchmark 数字表逐项、MOPD 实验表逐项、survey v3/v4 全文 diff。
