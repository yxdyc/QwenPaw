# DeepSeek LLM Scaling Laws：怎样分配 model、data 与 compute

> **一手来源**：DeepSeek LLM: Scaling Open-Source Language Models with Longtermism，arXiv:2401.02954，48 页。
> **定位**：这份 2024 报告是 DeepSeek 系列中最直接回答 model/data/compute allocation 的材料；
> V3 报告重点已转向 MoE、MLA、FP8 与大规模训练系统，不能用 V3 的成本数字替代本文的 scaling 实验。

---

## 1. 它到底优化了什么

固定训练 compute $C$，模型做大意味着可消费的数据量变少；数据增多意味着每 token 可用模型计算变小。
报告要解的是：

$$
(M_{opt}(C),D_{opt}(C))
=\arg\min_{M,D:C=MD}L(M,D),
$$

其中 $D$ 是训练 token 数，$M$ 是 non-embedding FLOPs/token。它没有直接用参数量 $N$，原因是：

- $6N$ 近似没有准确计入 attention 随 sequence length 增加的开销；
- total parameters 里的 vocabulary projection 会增加 compute，却不等价于同样的主干容量；
- 在报告的小模型设置中，$6N$ 与真实 $M$ 的误差可达到约 50%，会扭曲 scaling fit。

因此报告用

$$
C=M\cdot D
$$

统一模型侧每 token 计算与数据侧 token exposure。这个变量选择本身就是重要经验：**scaling law 先要让横轴接近真实资源，
否则拟合精度高也可能是在拟合错误账本。**

---

## 2. 实验是怎样铺开的

### 2.1 先标定 batch size 与 learning rate

报告先在小规模实验上扫 batch size 与 learning rate，再把 generalization error 距最优值不超过 0.25% 的区域定义为
near-optimal band。拟合得到论文特定单位下的经验关系：

$$
\eta_{opt}=0.3118C^{-0.1250},\qquad
B_{opt}=0.2920C^{0.3271}.
$$

含义是 compute 增大时，最优 batch 倾向变大、learning rate 倾向变小；near-optimal 区域比较宽，不需要迷信唯一小数点。
这些系数绑定该模型族、optimizer、数据、tokenizer、sequence length 和 $C$ 的单位，不能复制到新项目直接开大跑。

### 2.2 再做 IsoFLOP profiles

报告选了 8 个 compute budgets，从 $10^{17}$ 到 $3\times10^{20}$ FLOPs；每个 budget 设计约 10 种 model/data allocation。
每条曲线使用前一步拟合的近优超参，在一份与训练分布相似、规模 100M tokens 的独立 validation set 上计算 bits-per-byte。

对每个 compute budget，取 loss 最低的 $(M,D)$，再分别拟合 $M_{opt}$ 与 $D_{opt}$ 随 $C$ 的幂律：

$$
M_{opt}=0.1715C^{0.5243},\qquad
D_{opt}=5.8316C^{0.4757}.
$$

指数相加约为 1，与 $C=MD$ 一致。compute 放大 10 倍时，这个 fit 对应模型侧每 token compute 放大约
$10^{0.5243}=3.34$ 倍、数据 token 放大约 $10^{0.4757}=2.99$ 倍。报告还用小规模拟合预测了约 1000×
compute 的 7B/67B 结果，并报告落点与曲线相符。

这不是“最优参数量等于某个固定值”，而是一个 ladder：每个 compute 档都要横向扫描 model/data allocation，
再用多个档位拟合趋势，最后保留目标规模的确认实验。

---

## 3. 最有价值的发现：数据变了，最优 exponent 也会变

报告比较三版数据得到：

| 数据 | model exponent $a$ | data exponent $b$ |
|---|---:|---:|
| early in-house | 0.450 | 0.550 |
| current in-house | 0.524 | 0.476 |
| OpenWebText2（报告内部质量判断更高） | 0.578 | 0.422 |

作者的经验解释是：数据质量提高后，在相同数据规模上可以支持更大的模型，因此新增 compute 的更大比例投向模型侧。
这里需要保留三条边界：

1. 这是三个数据分布上的经验规律，不是“质量越高，永远应该少用数据”的定理；
2. “质量”来自其内部数据 assessment，OpenWebText2 规模更小、处理更精细，不能化约成单一 scalar；
3. tokenizer、optimizer、架构和目标能力变化都可能改变 exponent，旧 fit 不能跨代硬复用。

一个反直觉但重要的结论是：**scaling law 不是模型独有属性，而是 model × data × optimization contract 的联合属性。**
所以新数据版本不能只看固定小模型 loss；它可能改变大规模 compute 的最佳分配方式。

---

## 4. 与 Sandbox / data scaling ladder 的关系

两者都用小规模实验减少大跑试错，但所估对象不同：

| | DeepSeek LLM scaling | Data-Juicer Sandbox |
|---|---|---|
| 决策变量 | model FLOPs/token、training tokens、batch、LR | operator、bucket、recipe、能力切片 |
| 主要响应 | validation bits-per-byte / loss | 下游能力向量与 recipe gain |
| 目标 | 固定 compute 下分配 model/data | 固定模型与预算下筛数据干预 |
| 主要风险 | fit 外推、数据/架构改变、近优区误判 | winner's curse、评测过拟合、operator interaction、rank flip |

一个完整 ladder 可以先用 Sandbox 类实验筛掉明显无效 recipe，再为保留的 2–3 个数据版本各做低成本 IsoFLOP 或
target-family 短跑，判断数据质量变化是否连带改变 model/data allocation。不能用一次 Sandbox 排名直接推出全局 scaling exponent，
也不能用 validation loss scaling 代替细粒度能力和产品 guardrails。

---

## 5. 如何在实际团队复现这套方法

### S0：冻结资源与数据账本

固定 tokenizer、sequence length、optimizer family、model shape family、数据版本、validation contract；将
attention、embedding/vocab 与激活重算的 FLOPs 口径写清楚。

### S1：超参 pilot

在 2–3 个小 compute 档上扫 LR×global batch，找到 near-optimal band；报告 band，不只报最优点。先排除不稳定训练，
否则 IsoFLOP 曲线比较的是“谁更没调好”。

### S2：IsoFLOP allocation sweep

每个 compute 档选 6–10 个 model/data allocations，覆盖明显 undersized 到 oversized model；相同数据版本上使用固定
validation loss，同时记录能力切片、吞吐和失败率。

### S3：fit 与不确定性

对每档的最优区间而非单点拟合；bootstrap compute levels 或对曲线做 profile likelihood，报告 exponent CI 与
相邻配置的 regret。若多个配置落在 noise band 内，保留成近优集合。

### S4：目标规模确认

目标规模保留至少一个 allocation 对照，检查 loss 预测、关键能力、MFU、稳定性和 token/data availability。
数据版本改变时重新做至少低成本 ladder；不要只改系数。

---

## 6. 机制检验题与参考答案

### Q1. 为什么不用 Chinchilla 的 0.5/0.5？

0.5/0.5 是特定模型、数据和优化合同上的经验近似。DeepSeek 的 current-data fit 是 0.524/0.476，已经很接近；
更关键的是它展示了换数据后指数从 0.450/0.550 到 0.578/0.422 漂移。工程上应把 Chinchilla 当 prior，
用本地 IsoFLOP evidence 更新，而不是当硬常数。

### Q2. 既然高质量数据让 model exponent 变大，是否应该激进过滤？

不能。激进过滤同时减少 unique tokens 和 coverage，可能迫使数据重复、损伤长尾能力。应把质量、可用数据上限、
重复 exposure、能力向量和 compute allocation 联合优化；“更干净”只有在目标训练合同下才有意义。

### Q3. 1000× 外推为什么可能成功，也为什么危险？

幂律在平滑、同构的训练 regime 内可跨档外推，且报告的 7B/67B 落点支持该 fit。危险来自 regime change：架构、
sequence length、tokenizer、数据质量、optimizer 或数值精度变化会改变曲面。外推越远，越应提供 target-scale anchor。

### Q4. 最新基模是否还会做类似 ladder？

公开报告常披露 architecture/compute/token 总量，却很少像这份报告一样公开完整 allocation sweep。方法本身仍是合理的
前沿训练流程：超参 pilot → proxy scaling → target anchor；是否被某一未披露模型采用，不能从最终模型表反推。
