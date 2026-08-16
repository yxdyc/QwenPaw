# nano-opd L0 — reverse KL vs forward KL vs SFT 蒸馏：为什么 OPD 必须「学生写、教师批」

> **模块定位**：OPD（on-policy distillation，学生自采样 + 教师分布监督）是 2025–2026
> 后训练的主流方向之一（时效性 B 层前沿选题，§八 SOTA 对齐记录见 §15）；
> MiniLLM / GKD 是它的经典锚点（A 层）而非前沿本身。
> 本模块只隔离 OPD 的**最小机制**——「散度怎么选、on-policy 从哪来」，
> 不覆盖完整蒸馏栈（数据、调度、multi-teacher 工程见 L1–L3 阶梯）。
>
> **K+1 起点（K）**：知道 KL 散度的定义、知道 MLE/SFT 是什么、见过 policy gradient
> 的基本形式。本节推进到 K+1：说得出「为什么 reverse KL 蒸馏必须学生自采样」，
> 并且能用算术（而不是断言）证明它。

---

## 1. 问题：同一个学生、同一个教师，三条蒸馏路线差在哪

蒸馏的朴素版本（SFT 蒸馏 / 离线 KD）人人会写：教师生成一批文本，学生做
next-token MLE 或拟合教师的输出分布。它的问题在**长程生成**上暴露：
学生只在「教师写过的状态」上被监督过，一旦自己生成时走偏到教师没写过的
前缀，就没有任何信号纠正它——exposure bias。

OPD 的处方：**学生自己采样轨迹，教师对学生真正写出来的东西给
（token 级）分布监督**。survey [2604.00626] 的界定原文是 "having the teacher
provide feedback on what the student actually produces"（让学生先产生输出，
教师针对学生实际产出提供反馈），并将其形式化为**学生自采样轨迹上的
f-divergence 最小化**。

但「换谁采样、优化什么散度」不是风格问题，是会算出完全不同结果的机制问题。
本节用一个极小的 toy 把三条路线放在一起跑：

| 配方 | 优化目标 | 信号从哪来 | 采样需求 |
|------|----------|-----------|----------|
| SFT 蒸馏 | 硬标签 MLE | 教师**生成的数据** | 无（离线） |
| forward KL | KL(p*\|\|q)，期望在**教师**分布下 | 教师**全分布** | 无（精确求和即可，离线） |
| reverse KL | KL(q\|\|p*)，期望在**学生**分布下 | 教师对学生样本**逐个打分** | **必须从学生自己采样**（on-policy 的由来） |

读完本节你应该能回答：**为什么第三条的「必须」是算术定理，不是工程习惯。**

---

## 2. 先跑起来

```bash
$ python3 L0_opd_divergence_choice.py
```

真实输出（纯标准库 math/random，固定 seed，跨运行逐字节确定）：

```text
========================================================================
nano-opd L0 — reverse KL vs forward KL vs SFT 蒸馏
========================================================================
toy 口径: 网格双峰教师 + 单峰受限学生(只有 mu 可动)；数字为现场算术，非 benchmark。

[1] 教师分布：两个模，中间是谷
    p*(x=0)=0.0044  p*(x=3)=0.1995  (模/谷 ≈ 45x)

[2] 三种配方，同一起点 mu0=+0.5：
sft  (教师样本+硬标签, 离线)                mu=-0.227 | 模区驻留率=0.054 | E_q[p*]=0.0214 | KL(q||p*)=3.198
fwd  (KL(p*||q), 精确求和, 离线)         mu=+0.000 | 模区驻留率=0.045 | E_q[p*]=0.0200 | KL(q||p*)=3.249
rev  (KL(q||p*), 学生自采样, OPD)       mu=+3.000 | 模区驻留率=0.955 | E_q[p*]=0.1559 | KL(q||p*)=0.736
    => SFT/fwd 把唯一的峰摆在两模中间的谷里(mode-covering, 生成教师不认账的样本)；
       rev 锁定一个模(mode-seeking, 生成的样本教师给高分)。

[3] 对称破缺：rev 锁哪个模由起点/早期样本决定，绝不骑墙
    mu0=+0.5 → mu=+3.000 | mu0=-0.5 → mu=-2.999

[4] 反例：reverse KL 为什么必须 on-policy
    a) 在锁模点 mu=+3 处看更新量的精确期望（无采样噪声）:
       正确估计器(从学生采): E[g]=+0.002  ≈0，模是驻点
       错误估计器(从教师采): E[g]=+136.149  ← mu -= lr*g，被拽向更小的 mu（离模进谷）
    b) 真跑起来(教师采样, 300 步): mu=-0.046 | 模区驻留率=0.045  ← 锁不住模，掉回谷里
       （若用主实验的 lr=0.15，有偏的大步长直接冲出网格——偏差不是噪声，是系统性的）

[5] self-check
✅ self-check passed: 三配方收敛形态 / 对称破缺 / 模区驻留率 / 教师认账度 / 估计器偏差分析

takeaway: reverse KL 的期望在学生分布下 → 必须学生自采样(on-policy)；
          教师只需给每个样本打分(logprob)。这一「学生写、教师批」的接口
          就是 OPD；L1 把它搬进真实序列模型，L2 加 multi-teacher。
```

三个 toy 指标的含义（后面反复用）：

- **模区驻留率** = 学生分布落在教师高概率区（|x∓3|≤1.5）的质量占比——学生生成的东西有多大比例在「教师的地盘」上；
- **E_q[p\*]** = 学生分布下教师的平均概率——学生生成的样本教师认不认账；
- **KL(q\|\|p\*)** = 以 reverse KL 口径度量的剩余差距。

---

## 3. toy 设置：双峰教师 + 容量受限学生

「token 空间」用一维离散网格 `GRID = [-6, ..., +6]`（13 个位置充当词表）。

**教师** p\* 是双峰：0.5·N(−3,1) + 0.5·N(+3,1) 离散化后归一。模/谷比 ≈ 45x
（p\*(±3)=0.1995 vs p\*(0)=0.0044）——两个「正确答案风格」，中间是教师自己
都不怎么去的荒原。

**学生** q_mu 是单峰 N(mu, 0.8)，**唯一可学参数是 mu**（峰的位置）；
峰宽、形状都动不了。这是刻意的：

> **容量受限不是缺陷，是设置。** mode-covering 与 mode-seeking 的差异只有在
> 「学生装不下两个模」时才显形。真实蒸馏里学生永远比教师小——同样的处境。
> 若学生容量足够（能精确表示 p\*），三条路线都收敛到 KL=0，差异消失（见 §13 反例三）。

**toy 口径声明**：全部数字是本脚本现场计算的玩具值，用于展示机制的方向与
相对量级，不是任何真实模型的 benchmark；「教师打分」在真实 LLM 里对应
teacher logprob，这里对应 `LOG_TEACHER[x]` 查表。

---

## 4. 配方一：SFT 蒸馏——抄范文抄到两篇范文的平均位置

```python
def train_sft(seed=7, n_data=256, steps=120, lr=0.5, mu0=0.5):
    """从教师采 n_data 个样本，硬标签 MLE。∇ E[log q] = E[(x-mu)/σ²]，
    最优解 mu = 样本均值——双峰数据的均值在两峰中间。"""
    rng = random.Random(seed)
    data = [sample(TEACHER, rng) for _ in range(n_data)]
    mu = mu0
    for _ in range(steps):
        mu += lr * sum(x - mu for x in data) / len(data) / (SIGMA_S * SIGMA_S)
    return mu
```

只需要教师**生成的数据**，连教师分布都不用给——这是最便宜的蒸馏。
单峰高斯的 MLE 不动点是 mu = 样本均值。数据从双峰教师采来，一半在 −3、
一半在 +3，均值在两峰中间。跑出来：`mu=-0.227`（seed=7 的样本均值），
模区驻留率只有 0.054——**学生 95% 的质量堆在教师自己概率都很低的谷里**。

注意这不是步数不够：不动点由数据均值代数决定，再训一万步 mu 也在谷里
（反例二，§13）。

---

## 5. 配方二：forward KL——期望在教师分布下，所以完全离线

```python
def train_forward(steps=120, lr=0.5, mu0=0.5):
    """KL(p*||q)：期望在教师分布下，精确求和即可，完全不需要学生采样。
    ∇ = -E_{p*}[(x-mu)/σ²] → 最优 mu = E_{p*}[x]。"""
    mu = mu0
    for _ in range(steps):
        g = -sum(p * (x - mu) for x, p in zip(GRID, TEACHER)) / (SIGMA_S * SIGMA_S)
        mu -= lr * g
    return mu
```

forward KL 的期望写在**教师分布** p\* 下——网格只有 13 个点，直接精确求和，
连采样都不需要，更不需要学生采样。这就是经典离线 KD 的形态。

不动点 mu = E_{p\*}[x]。教师对称，E_{p\*}[x] = 0，跑出来 `mu=+0.000`。
我在 mu∈[−6,6] 网格上直接算过目标形态：KL(p\*\|\|q(mu)) 的全局最小就在
mu=0（6.397），mu=±3 处是 13.43——**「选谷」是 forward KL 目标函数的算术
事实，不是优化没跑好**。

为什么 forward KL 天生 mode-covering？KL(p\*\|\|q) 对「p\* 有质量而 q 没
质量」的位置惩罚无穷大（log(p\*/q)→∞），所以优化逼着 q 把质量摊到 p\*
所有非零的地方。单峰学生摊不开两个模，只能摊在中间——每处都沾一点，
每处都不像。生成时教师认账度 E_q[p\*]=0.0200，和 SFT 一个量级。

---

## 6. 配方三：reverse KL——期望在学生分布下，on-policy 是算术要求

```python
def train_reverse(seed=7, mu0=0.5, steps=300, batch=32, lr=0.15, use_baseline=True):
    """∇ KL(q||p*) = E_{x~q}[(log q(x) - log p*(x) - b) · (x-mu)/σ²]
    其中 log p*(x) = 教师对学生每个样本的「打分」（真实 LLM 里 = teacher logprob）。
    b 是移动平均 baseline（方差缩减，policy gradient 的标准件）。"""
    rng = random.Random(seed)
    mu, b = mu0, 0.0
    for _ in range(steps):
        q = student_dist(mu)
        log_q = [math.log(v) for v in q]
        g, f_mean = 0.0, 0.0
        for _ in range(batch):
            x = sample(q, rng)                       # ← 学生自采样（on-policy）
            i = x - GRID[0]
            f = log_q[i] - LOG_TEACHER[i]            # log q - log p*
            f_mean += f
            g += (f - b) * (x - mu) / (SIGMA_S * SIGMA_S)
        g /= batch
        if use_baseline:
            b = 0.9 * b + 0.1 * f_mean / batch
        mu -= lr * g
    return mu
```

这一节的核心只有一行数学：

```
∇_θ KL(q_θ || p*) = E_{x~q_θ} [ ∇_θ log q_θ(x) · (log q_θ(x) − log p*(x)) ]
```

**期望的下标是 q_θ——学生自己的分布。** 这不是实现选择：散度定义里的
期望写在谁下面，无偏估计就必须从谁那里采样。想用 score-function
（REINFORCE）估计这个期望，样本必须来自当前学生——这就是 on-policy 的
全部由来。MiniLLM [2306.08543] 正是这条路线（survey 的转述：MiniLLM 把
优化重述为 REINFORCE，将 log(p_T/p_θ) 当作每步 reward）。

教师在这条路线里的角色变得非常轻：**不需要生成数据，不需要给梯度，只需对
学生写出的每个样本打分**——给出 log p\*(x)（真实 LLM 里就是教师对学生
token 序列的 logprob）。接口是「学生写、教师批」。

代价也立刻出现：每一步都要从**当前**学生采样——这正是 RL 的 rollout 问题
换了个名字。OPD 之所以能复用 RL infra（采样引擎、buffer、调度），原因在此
（nano-slime L0 / nano-vllm-sglang L0 讲的吞吐问题在这里同样成立）。

跑出来：`mu=+3.000`，模区驻留率 0.955，E_q[p\*]=0.1559——**学生 95% 的
生成落在教师地盘上，教师平均认账度是前两条路线的约 8 倍**。reverse KL
对「q 有质量而 p\* 没质量」的位置惩罚重（log(q/p\*)→∞），优化逼着 q 躲开
教师不去的地方——单峰学生唯一理性的策略就是挑一个模扎根。这就是
mode-seeking。

---

## 7. 「锁模」是算术定理：在模上算精确期望

「rev 能锁模、off-policy 不行」如果只靠一次运行演示，可能只是运气。
代码的 [4]a 把这件事做成了算术：在锁模点 mu=+3 处，直接对 13 个网格点
**精确求和**算更新量 g 的期望（无采样噪声）：

```
正确估计器（期望在学生分布下）: E[g] = +0.002   ≈ 0  → 模是驻点，待得住
错误估计器（期望换成教师分布）: E[g] = +136.149      → 被系统性拽离
```

+136.149 的符号可以手推：从教师采样时，约一半样本落在 x=−3（另一个模）。
在 mu=+3 处，学生对 x=−3 的概率极小 → log q(−3) 是大负数，而 log p\*(−3)
很大 → f = log q − log p\* 是大负数；再乘 (x−mu) = −6（也是负数），
负负得正，贡献一个大正项。于是 `mu -= lr*g` 把 mu 往小拽——**离开模、
走进谷**。教师采样把「另一个模上的质量」变成了持续的系统性拉力：
这恰恰是 forward KL 的 mode-covering 行为，被硬套在 reverse KL 的估计器上。

而正确估计器的样本集中在 x≈+3（学生自己的模），那里 f ≈ log q − log p\* ≈ 0，
正负贡献对称抵消——驻点。30 个不同 seed 全部锁模（|mu|>2.5，30/30），
排除了运气成分。

reverse KL 的目标形态同样是算术：KL(q(mu)\|\|p\*) 在 mu=±3.0 各有一个全局
最小（0.736），mu=0 是局部极大（3.249）。优化从哪边起步就滚进哪个盆地——
这解释了下一节的对称破缺。

---

## 8. 对称破缺：绝不骑墙

```
mu0=+0.5 → mu=+3.000
mu0=-0.5 → mu=-2.999
```

锁哪个模由起点和早期样本决定，但**一定锁一个**——reverse KL 的解空间里
「骑墙」（mu≈0）是局部极大，不是鞍点上的犹豫。

这对真实蒸馏的含义：容量受限的学生做 OPD，学到的是教师分布的**某一个
连贯风格/能力簇**，而不是所有风格的平均。这是 OPD 相对 SFT 蒸馏的行为级
差异——不是「学得更好」，是「学的东西形状不同」。要覆盖教师的多个模，
得靠更大的学生、多个学生、或 multi-teacher 课程（L2 的主题）。

---

## 9. 反例真跑：off-policy 估计器锁不住模

[4]b 把错误估计器真的跑起来（教师采样、小 lr、带边界钳制）：300 步后
`mu=-0.046`，模区驻留率 0.045——和 forward KL 一样掉回谷里。有偏梯度把
它从起点一路拽回两模中间，钳制只是防止它出界，救不回方向。

若把学习率换成主实验的 lr=0.15 且去掉钳制，我实测了发散路径：

```
step1: mu +0.500 → -4.351   （一步跳出模区）
step2: mu -4.351 → +32.486  （冲出网格，学生分布数值退化）
```

**偏差不是噪声。** 噪声型错误加大 batch 能摊薄，系统性偏差加到天荒地老
也是同一个方向——这正是 §7 里 +136.149 vs +0.002 的差距在动力学上的样子。

---

## 10. 费曼：能不能讲给外行听

**类比：师傅带学徒做两道招牌菜**（一道辣 x=+3、一道甜 x=−3，客人
99% 点这两道，点中间口味的不到 5%）。学徒只有一个灶、只练得精一种口味
（容量受限）。

- **SFT 蒸馏** = 学徒照师傅的成品菜抄 256 份。辣甜各半，抄出来的「平均
  口味」不辣不甜——客人（教师）不认账（驻留率 0.054）。
- **forward KL** = 师傅把完整菜单和点单概率给学徒，学徒要保证「菜单上
  每道菜都做得出」。一个灶cover不了两头，只能架在菜单正中间——每道都
  沾边、每道都不正宗。
- **reverse KL / OPD** = 学徒自己做菜，师傅每道尝一口打分（teacher
  logprob）。学徒很快发现：把辣菜做好，道道高分；做中间口味，道道口。
  于是专精辣菜（锁模 +3）——做出来的菜 95% 师傅认账。
- **off-policy 反例** = 学徒不自己做饭，反而去品尝师傅做的菜然后「反思」。
  反思的方向系统性错误：师傅两道都做，学徒被拽回中间，永远学不会任何一道。

**一句话版**：OPD 就是「学生写作文、老师批改」——**作文必须学生自己写，
因为要批改的是学生写得出来的东西**；老师的角色轻到只需打分，但学生
每走一步都得自己走。

**类比边界**：「一个灶、只能挪位置」对应单参数学生（真实学生是高维序列
分布）；「打分」对应 token 级 logprob；「客人点单分布」对应教师分布。
真实 OPD 里教师对每个 token 都打一次分，toy 里每次只对一个网格点打分——
机制同构，规模不同。

**反例版类比**（什么时候这个类比会误导你）：如果学徒有两个灶（容量足够），
「专精一道」就不再是必然策略——mode-seeking 的前提是装不下。类比在
「学生容量受限」这一点失效的地方，恰好对应 §13 反例三。

---

## 11. 与真实方法的对应（概念层）

toy 里每一块在真实 OPD 里都有对应物：

| toy | 真实系统 |
|------|----------|
| 网格 GRID | 词表上的序列空间（指数级大） |
| LOG_TEACHER[x] 查表 | 教师模型对学生 token 序列的 logprob（白盒教师） |
| `sample(q, rng)` | 学生模型的 rollout（真实推理引擎，见 nano-vllm-sglang） |
| score-function 梯度 + baseline | MiniLLM 的 REINFORCE 式策略梯度 [2306.08543] |
| 「换散度不换采样」 | GKD 的广义化：学生自采样 + 广义散度（含 JSD）[2306.13649] |
| 教师样本上做 token 级 loss | DistiLLM 在学生生成的前缀上算 loss + 自适应调度 [2402.03898] |
| multi-teacher（L2） | 多教师分布融合 / 路由（survey §5 的机制类别） |

**OPD 的当今定位（2026-08 视角，B 层前沿主流）**：OPD 已从论文进入生产
管线。OPD survey [2604.00626] 点名 Qwen3 [2505.09388]、DeepSeek-V4、
Gemma 2、MiMo-V2-Flash「均把 OPD 作为核心训练成分」（adoption cuts across
architectural lines）；文献两年内扩张到上百篇，沿 divergence 设计、
reward-guided optimization、self-play 三个方向展开。Thinking Machines 的
生产报告（2025-10-27）给出具体配方：Qwen3-8B 学生自采样、Qwen3-32B 教师
逐 token 打分、负 reverse KL 作为 advantage，把 AIME'24 从 SFT 后的 60%
拉到 70%；报告称相比继续 RL 微调可省约 9x 算力，并引 Qwen3 技术报告的
对照：RL 用 17,920 GPU 小时得 67.6，OPD 用 1,800 GPU 小时得 74.4
（以上数字为博客自述/转引，非本教程实测）。**注意经典 ≠ 前沿**：MiniLLM /
GKD 是机制地基（A 层经典锚点），当今生产配方在它们之上做了大量工程演进
（B 层），L3 再对照。

---

## 12. 思考题

1. **forward KL 的期望在教师分布下，精确求和就行，完全不需要学生采样——
   那为什么离线 KD 不能替代 OPD？** 提示：散度是在**谁的分布下被求值**的。
   forward KL 逼 q 覆盖 p\* 的所有非零处；容量受限的学生只能把质量摊进谷里，
   而生成时被评估的是 q 自己的样本——教师认账度 E_q[p\*] 说明了一切
   （回看 §2 的三个指标）。
2. **把 reverse KL 换成别的散度（比如 JSD），还需要学生自采样吗？**
   提示：问「期望写在谁下面」。GKD [2306.13649] 的广义化正是
   「采样保持 on-policy、散度可换」——JSD 在 forward/reverse 之间插值，
   缓解 reverse KL 的 zero-avoidance 过激。想清楚：换散度改变的是
   「覆盖 vs 聚焦」的权衡，不改变「样本必须来自学生」这件事。
3. **baseline b 为什么减掉也不破坏无偏性？** 提示：E_q[∇log q] = ∇Σq = 0。
   动手验证：把 `train_reverse(use_baseline=False)` 跑一遍——本 toy 里仍能
   锁模（实测 mu=+2.971），但轨迹方差更大。真实 LLM 里 token 级
   log-ratio 的量级波动远大于 toy，baseline / advantage 归一是必需件
   （与 nano-verl L1 的 GAE 同源）。
4. **toy 里教师是白盒的（logprob 随便查）。如果教师是黑盒 API、只肯给
   文本回复不给 logprob，OPD 还成立吗？** 提示：survey [2604.00626] §5.2
   「Black-Box and API-Constrained Distillation」一节给的机制类别；
   信号从「分布」退化成「评分/排序」时，OPD 与 RL 的边界就模糊了——
   这正是「蒸馏与 RL 合流」的具体含义。

---

## 13. 反例与边界

1. **off-policy 估计器（§7/§9）**：把 reverse KL 的期望错用教师样本估计，
   梯度偏差 +136.149 vs 驻点 +0.002——不是方差大，是方向错。真跑锁不住模
   （驻留率 0.045），大步长直接冲出网格。**教训：散度期望在谁下面，
   样本就必须从谁来——这条违反不得，违反了连「慢一点但对」都做不到。**
2. **SFT/forward KL 掉谷不是训练不足**：不动点 mu=E[x] 在两模中间是代数
   必然（MLE 与 forward KL 各自的最优性条件），加步数、调 lr 都救不了。
   **教训：目标函数选错时，优化越好越糟糕——forward KL 在 mu=0 处确实
   拿到了它的全局最小（6.397）。**
3. **差异的边界：学生容量足够时三条路线趋同。** 若学生族能精确表示 p\*
   （比如双峰混合模型），forward/reverse KL 都在 q=p\* 处取 0，on-policy
   与否只影响估计方差、不影响最优解。mode-covering vs mode-seeking 只在
   「装不下」时才是问题——而真实蒸馏的学生永远更小，所以这个问题永远在场。
   **教训：讨论散度选择前先问学生容量；容量充足时它是个假问题。**

---

## 14. 阶梯预告

- **L1**：把本 toy 搬进真实序列模型——真实 tokenizer、两个真实小模型
  作师生、真实梯度下降；教师 logprob 真实前向算出，对比 SFT 蒸馏 vs OPD
  的收敛与生成质量（验证「学生自采样 + token 级教师打分」在真数据上
  的形态）。依赖 torch，小模型 CPU/小显存可跑，配一键 fallback。
- **L2**：multi-teacher OPD——多教师分布融合 / 路由的最小机制。只教
  机制类别（survey taxonomy 里的分支），不追个别 2026 变体方法名
  （C 层单源变体按 §八 不立模块）。
- **L3**：对照生产配方与 survey taxonomy——divergence 选择、信号源
  （白盒/黑盒）、token 加权与效率稳定化；对照 verl / SWIFT 等框架的
  distillation 支持 `[TODO: verify]`。

**交叉引用**：off-policy 偏差的算法侧对策（importance sampling）→
[nano-verl](../nano-verl/) L1；学生自采样复用的 rollout infra →
[nano-slime](../nano-slime/) L0（staleness 账本对 OPD 同样成立）与
[nano-vllm-sglang](../../03-data-distributed-rsi/nano-vllm-sglang/) L0；
SFT 蒸馏的数据侧（模板 / loss mask）→ [nano-llamafactory](../nano-llamafactory/) L0。

---

## 15. 溯源与 §八 SOTA 对齐记录

**SOTA 对齐（课程的证据时效性分层，B 层选题写前必做）**

- 对齐日期：**2026-08-05**（全部一手来源当日现场核验）。
- 结论：未发现取代 OPD 的新一代范式；OPD 维持 **B 层前沿主流**定位，
  MiniLLM / GKD 为 **A 层经典锚点**（机制地基，非前沿本身）。

| 来源 | 内容 | 核验方式（2026-08-05） |
|------|------|------------------------|
| MiniLLM [2306.08543] | reverse KL + 学生自采样 + REINFORCE 式优化 | arxiv.org 标题页现场抓取逐字吻合：*MiniLLM: On-Policy Distillation of Large Language Models* |
| GKD [2306.13649] | 学生自采样 + 广义散度（self-generated mistakes） | 同上：*On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* |
| DistiLLM [2402.03898] | 学生生成前缀上的 token 级 loss + 自适应调度 | 同上：*DistiLLM: Towards Streamlined Distillation for Large Language Models* |
| OPD survey [2604.00626] | OPD = 学生自采样轨迹上的 f-divergence 最小化；方法 taxonomy；production adoption | v3（2026-05-18）HTML 全文抓取；v4 于 2026-06-18 修订（abs 页核验）。正文转述均出自 v3 抓取文本 |
| Qwen3 [2505.09388] | 生产配方采用 OPD（survey 点名） | arxiv.org 标题页核验：*Qwen3 Technical Report* |
| Thinking Machines blog | 生产级 OPD 配方（Qwen3-8B 学生 / 教师逐 token 打分 / 负 reverse KL 作 advantage；AIME'24 60%→70%；算力对照 17,920 vs 1,800 GPU 小时为博客转引 Qwen3 报告） | 2025-10-27 发布；数字标注为博客自述/转引，非本教程实测 `[blog claims]` |
| DeepSeek-V4 采用 pure multi-teacher OPD | survey 正文的转述 | 本报告未直接核验 DeepSeek-V4 技术报告 `[TODO: verify]` |
| multi-teacher 具体方法（MAD-OPD / MOPD / Uni-OPD） | 2026 年单篇变体，只作机制类别佐证 | survey v3 方法表转述；个别方法不单独立论（§八 C 层纪律） |

**本节全部 toy 数字**：`L0_opd_divergence_choice.py` 当日运行输出（三遍
逐字节一致）；KL 目标形态（forward 全局最小 mu=0: 6.397 / reverse 双谷
±3.0: 0.736、mu=0 局部极大 3.249）、off-policy lr=0.15 两步发散路径、
30/30 seed 锁模——均为独立脚本复算，非转引。
