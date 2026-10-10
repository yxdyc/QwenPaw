# Evaluation Gate L2 companion — 反复挑最好结果时，为什么赢家会自带乐观偏差

> **核心问题**：当同一评测面反复比较 operator、bucket、recipe 或 Agent 生成的实验方案时，
> “当前最好”究竟包含多少真实增益，多少只是被搜索放大的噪声？
>
> **先修**：[Evaluation Gate L0](tutorial_L0.md) 的 paired evidence 与 held-out 合同；
> [L2 主课](tutorial_L2.md) 的 cluster-level 检验与 alpha-spending。
>
> **运行**：`python3 L2_adaptive_search_lab.py`；纯标准库，CPU，通常一秒内。
>
> **验收**：能够复算最大噪声量级、paired $Z_L$、MDE、Kendall $\tau$ 与 mis-rank probability，
> 并解释它们分别约束搜索、停机、晋升和独立验收中的哪一步。
>
> **边界**：代码中的独立高斯噪声与小样本排序是教学模型，不代表真实训练 run 独立同分布，
> 也不证明 proxy 排序必然迁移到目标模型。
>
> 本文把 winner's curse、标准误、最小可检出效应、排序稳定性和独立 held-out 串成一条工程判断链。
> 公式用于决定是否继续搜索、扩量或冻结候选，不替代目标模型上的最终验证。

---

## 1. 最大噪声为什么约为 $\sigma\sqrt{2\log M}$

先做一个极端但有用的零假设：$M$ 个候选真实效果都等于 0，观测结果只有独立高斯噪声：

$$
Y_i=\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,\sigma^2).
$$

单个候选出现很大正噪声的概率不高；从 $M$ 个候选里专门挑最大值，问题已经变成极值统计。令
$Z_i=\epsilon_i/\sigma$，则

$$
P\!\left(\max_i Z_i\le x\right)=\Phi(x)^M.
$$

“典型最大值”大致出现在单个尾概率约为 $1/M$ 的位置。高斯尾部的主导项近似
$P(Z>x)\approx e^{-x^2/2}$，令它等于 $1/M$：

$$
e^{-x^2/2}\approx\frac1M
\quad\Longrightarrow\quad
x\approx\sqrt{2\log M}.
$$

因此最大纯噪声的量级约为 $\sigma\sqrt{2\log M}$。这不是精确期望值，也忽略了低阶修正；它表达的本质是：
**搜索次数越多，赢家越可能是“真实效果 + 一次幸运噪声”共同产生的。**

若把 40 个 operator、3 个 bucket 当成粗略上界，$M=120$，则
$\sqrt{2\log 120}\approx3.09$。即使所有候选都无真实增益，挑出来的最大观测值也可能接近 $3\sigma$。
更直观地，120 次独立比较中至少出现一次 $>3\sigma$ 噪声的概率约为
$1-\Phi(3)^{120}\approx15\%$。实际候选共享数据、模型和统计量，彼此相关，故有效候选数
$M_{\mathrm{eff}}<120$；但相关性不会自动消除选择偏差。

### 与 ADE 的关系：隔离搜索和验收，不宣称消灭选择偏差

ADE 的协议把反馈分成两层：

1. **in-loop** 指标允许研究过程反复读取，用来提案、修改、挑 checkpoint 和选 Trial；
2. 选中的 artifact 与 checkpoint 冻结后，才在研究过程不可见的 disjoint held-out 上评估；
3. held-out 一旦反过来指导下一轮方案，它就已经变成 in-loop，需要新的 untouched benchmark；
4. “NEW BEST”与“hypothesis supported”分开记录，避免把高分赢家直接解释成机制成立。

它针对的正是 adaptive search：允许在开发集上发生 winner's curse，但不拿同一个噪声赢家证明泛化。
这个协议仍不保证偏差归零：in-loop 最优值仍偏乐观；held-out 本身也有抽样误差；如果研究者反复用同一个
held-out 比较多个 ADE 配置，还会产生更高一层的 meta-overfitting。更稳妥的报告还应包含独立端到端重复、
跨 run 分布、冻结后的一次性 held-out、预算匹配的 independent-search baseline，以及 effect CI。

---

## 2. $Z_L>2$：把“loss 确实在动”变成一个信噪比检查

若短跑在若干独立 seed 或数据 shard 上得到 loss drop $d_1,\ldots,d_S$，定义

$$
Z_L=\frac{\bar d}{SE(\bar d)},
\qquad
SE(\bar d)=\frac{s_d}{\sqrt S}.
$$

- $s_d$ 是多次独立实验的 loss drop 的**样本标准差**，描述单次结果有多散；
- $SE$ 是**标准误**，描述当前均值估计有多不确定；样本数增加时按 $1/\sqrt S$ 缩小；
- 若 candidate 与 baseline 使用相同 seed、shard 和初始化，先逐对相减再算 $s_d$，通常比两组独立估计更省方差。

$Z_L>2$ 的工程含义是：观测均值约离 0 两个标准误。在近似正态、规则预先固定的条件下，它接近传统双侧
95% 阈值，但这里只把它当“训练信号已越过明显噪声”的初始门槛。它不证明下游能力提升，也不是看到结果后
可以反复调整的 stopping rule。

例：三个 paired runs 的 loss drop 为 $[0.02,0.08,0.05]$，则 $\bar d=0.05$，$s_d=0.03$，
$SE\approx0.0173$，$Z_L\approx2.89$。若只有一次 run，就无法从该 run 自身估计 seed-level SE；连续训练 step
高度自相关，也不能伪装成数百个独立样本。

---

## 3. MDE：当前预算究竟有能力看见多小的差异

两组各有 $S$ 次独立重复、单次指标标准差均为 $\sigma$ 时，均值差的标准误为

$$
SE(\bar Y_1-\bar Y_0)
=\sqrt{\frac{\sigma^2}{S}+\frac{\sigma^2}{S}}
=\sqrt{\frac{2\sigma^2}{S}}.
$$

要同时满足双侧显著性 $\alpha$ 与检验功效 $1-\beta$，真实差异除了跨过拒绝阈值
$z_{1-\alpha/2}SE$，还需留出 $z_{1-\beta}SE$ 的概率裕量。因此

$$
MDE\approx
\left(z_{1-\alpha/2}+z_{1-\beta}\right)
\sqrt{\frac{2\sigma^2}{S}}.
$$

在 $\alpha=0.05$、power $=0.8$ 时，两个 $z$ 值约为 1.96 和 0.84，总和为 2.80。若指标单次
$\sigma=0.5$，每组只有 4 个独立 runs，则 $MDE\approx0.99$：小于约 1 分的真实差异很可能看不清。
paired design 应把 $2\sigma^2$ 换成 paired difference 的方差，常能显著降低 MDE。

MDE 小于业务上有意义的最小增益，实验才真正有裁决能力。结果“不显著”且 MDE 很大，只能说证据不足，
不能说两个方案等价。比较很多候选时还需 alpha-spending、FDR 或最终独立 holdout，不能继续套单次公式。

---

## 4. Kendall $\tau$：相邻 checkpoint 的候选相对次序是否稳定

把 $N$ 个 recipe 分别在 checkpoint $t_1,t_2$ 排序。任取一对候选：

- 两个 checkpoint 给出相同先后次序，称 concordant；
- 次序翻转，称 discordant。

无 ties 时：

$$
\tau=\frac{C-D}{C+D}.
$$

$\tau=1$ 表示所有 pair 都保持次序，$0$ 表示接近随机，$-1$ 表示完全反转。例如
$t_1:A>B>C>D$，$t_2:A>C>B>D$，6 对里只有 B/C 翻转，所以 $\tau=(5-1)/6=0.667$。

连续两个甚至三个 checkpoint 的 $\tau$、top-k overlap 与候选效应方向同时稳定，说明短跑开始足以支撑
“筛谁进入下一阶段”。它不证明 target scale 的排序保持，因为两个 checkpoint 可能共享同一种 proxy bias。
实践用 Kendall $\tau_b$ 处理 ties，并把统计不可分的候选归为同一 tier，避免强行排全序。

---

## 5. 为什么 mis-rank 由 margin / noise 决定

设候选 $i$ 的真实效果比 $j$ 高 $\Delta_{ij}=\mu_i-\mu_j>0$，观测差为

$$
D_{ij}=\Delta_{ij}+\epsilon_{ij},
\qquad \epsilon_{ij}\sim\mathcal N(0,\sigma_{ij}^2).
$$

排错名次意味着 $D_{ij}<0$，因此

$$
P(\text{mis-rank }i,j)
=P(\epsilon_{ij}< -\Delta_{ij})
=\Phi\!\left(-\frac{\Delta_{ij}}{\sigma_{ij}}\right).
$$

$\Delta/\sigma$ 就是 pairwise signal-to-noise ratio：

- $\Delta=0.8,\sigma=0.4$ 时为 2，误排概率约 2.3%；
- $\Delta=0.2,\sigma=0.4$ 时为 0.5，误排概率约 30.9%。

排序“看上去稳定”不是 rank 指标自带魔法，而是公共扰动在 paired difference 中被抵消，且候选 margin 足够大。
若多个 recipe 使用相同 base、seed 和 shard，
$\operatorname{Var}(\epsilon_i-\epsilon_j)=\operatorname{Var}(\epsilon_i)+\operatorname{Var}(\epsilon_j)-2\operatorname{Cov}(\epsilon_i,\epsilon_j)$，
正相关公共噪声会被差分消掉。候选差距很小时，绝对分数误差即便不大，名次仍会频繁交换。

---

## 6. 一套可以实际执行的停止与晋升合同

1. 预先固定 checkpoints、主指标、能力 guardrails、seed/shard pairing 与候选预算；
2. 用 $Z_L$ 检查短跑是否出现可辨别训练响应；
3. 由 pilot 方差和业务最小增益算 MDE，确认实验有足够功效；
4. 查看 per-slice effect、Kendall $\tau_b$、top-k overlap、pairwise CI，而非只看总分；
5. 连续两个预登记 checkpoint 达到阈值，且 top tier 与 guardrails 稳定，才冻结候选；
6. 目标 family 短跑复验，再进入一次性 disjoint held-out；
7. held-out 若指导修改，升级为新的 in-loop，并启用新的 untouched holdout。

这条链把“训练是否有信号”“候选能否区分”“次序是否稳定”“搜索结果是否泛化”拆成四个问题，避免用一个
漂亮总分同时回答所有问题。

---

## 7. 费曼自检与参考答案

建议先独立作答，再用参考答案检查自己是否混淆了开发集筛选、统计功效与最终泛化证据。

### Q1. 120 个候选彼此高度相关，$3\sigma$ 还准确吗？

不准确。$3\sigma$ 是独立候选的直观上界；相关候选的有效比较数更小。但只要最后是根据同一噪声实现挑赢家，
selection bias 仍存在。可通过 block bootstrap、候选相关矩阵的有效维数、独立 run 和 held-out 评估偏差。

### Q2. 连续 checkpoint 的 $\tau=0.9$，可以直接扩到目标模型吗？

不能。它只说明当前 proxy trajectory 内部相对次序稳定。架构、tokenizer、训练阶段、数据 exposure 或 objective
变化都可能产生 scale interaction。应先做 target-family 短跑，再冻结候选进入独立评测。

### Q3. ADE 已有 held-out，为什么还要报告跨 run 分布？

held-out 防止同一搜索反馈直接充当最终证据；跨 run 分布回答搜索过程是否稳定、是否依赖一次幸运探索。
两者处理的随机性层级不同，缺一不可。
