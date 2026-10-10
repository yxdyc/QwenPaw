# 数据 Recipe 实验设计：从 operator 排名到可迁移的能力结论

> **核心问题**：一个数据 operator 在某个 bucket 上赢了，怎样判断它改进了什么、是否可组合、能否迁移到
> target model？本文以 Data-Juicer Sandbox 的 Probe–Analyze–Refine 为起点，补齐统计功效、混杂控制、
> interaction、能力切片和多模态覆盖的实验合同。

---

## 1. 三张表先分开：operator、bucket、capability slice

把一次数据干预写成三维响应，而非立即压成总分：

$$
Y[o,b,k]=\text{recipe }(o,b)\text{ 对能力切片 }k\text{ 的效果}.
$$

- **operator $o$**：改变数据的规则或模型，如相似度、aesthetics、NSFW、长度、去重；
- **bucket $b$**：同一统计量的强度或区间，如 low/mid/high、不同 retention rate；
- **capability slice $k$**：模型在特定输入条件下的可重复行为，如 OCR、主体一致性、动作绑定。

保留这个 response tensor 可以区分三类现象：同一 operator 随阈值产生 U 形响应；一个 operator 提升 alignment
却损伤 rare-object coverage；同一 recipe 在模型规模变化后发生 rank flip。总分只是业务权重下的投影
$U=\sum_k w_kY[o,b,k]$，不是数据质量的内禀属性。

### 能力切片的严格定义

一个可用切片至少包含：

$$
C_k=(\text{input distribution},\text{required behavior},\text{evaluation contract},
\text{failure interpretation},\text{guardrail}).
$$

“医疗能力”太粗；“长病历含干扰项时，引用正确证据并说明诊断依据”才可测。样本标签是协变量，能力切片是
模型行为的条件结果；两者需要建立可检验映射，不能画等号。

---

## 2. correlation analysis 与 importance analysis 回答不同问题

### Correlation analysis：哪些变量提供了重复信息

Sandbox 原文做了两类相关分析：

1. operator statistics 之间的 Pearson correlation，用来把高度冗余的 OP 分组；
2. evaluation metrics 之间的 correlation，用来把评测维度聚成 static、dynamic、quality / matching 等组。

它适合降维、选择互补候选，也能暴露重复计权。它回答“谁与谁一起变化”，不能回答“改变谁会让模型变好”。

### Importance analysis：在固定合同下，改变该变量带来多大效果

Fig. 1 的 importance 更接近受控 intervention：固定 base、样本数、训练预算和 evaluator，比较同一 operator 的
low/mid/high pool 相对 random pool 的模型变化。若进一步系统化，可估计：

$$
\tau_{o,b,k}=E[Y_k\mid do(o,b)]-E[Y_k\mid do(\text{random})].
$$

真实数据实验很难满足完全随机化，所以需要 matching/reweighting、paired seeds 和 source-aware uncertainty。
相关分析负责压缩候选空间；importance analysis 负责回答“值得不值得改 recipe”；两者不能互相替代。

---

## 3. 六个实验工具怎样落到一次具体 recipe 选择

假设要测试两个过滤器：

- $A$：保留高 image-text alignment；
- $B$：保留高 aesthetic score。

四个 arms 使用相同 base、token、steps、sampler 和评测合同：

| arm | 结果 $Y$ |
|---|---:|
| random：$(0,0)$ | 50.0 |
| 只用 $A$：$(A,0)$ | 52.0 |
| 只用 $B$：$(0,B)$ | 51.5 |
| 组合 $A+B$ | 52.2 |

### 3.1 Paired seeds

四个 arms 都使用相同 seed 集、相同 source shards 与相同评测 requests。对每个 seed 先算
$d_s=Y_{A,s}-Y_{0,s}$，再估计均值和方差。这样初始化难度、shard 难度等公共扰动会在差分中抵消。
“三次 run”只有在 seed / shard 真正提供独立随机单元时才是三个样本；同一 run 的训练 steps 不独立。

### 3.2 Bootstrap CI

对真正独立的单位重采样，例如 seed、source shard、用户、视频簇或 prompt family；每次重算 effect，取分位数形成 CI。
不能把同一视频拆出的数百帧、同一模板生成的数百题当独立样本逐行 bootstrap，否则区间会虚假变窄。

### 3.3 Threshold dose-response curve

不要只比 low/high。预先固定 retention 20/40/60/80/100% 或分位点阈值，画出：

$$
\text{intensity / retention}\longrightarrow
(\text{target gain},\text{coverage loss},\text{cost}).
$$

单调上升支持“越强越好”；先升后降表明高阈值开始损伤多样性；平台区说明继续过滤只有成本没有收益。
阈值选择应考虑曲线和业务约束，而非从多个点里事后挑峰值。

### 3.4 Matching 或 reweighting

高 alignment pool 可能同时更短、更英文、更高分辨率。直接比较会把 length/language/resolution 的影响归到
alignment。可在这些协变量上分层匹配、倾向分加权或固定 quota，使两组在已观测混杂因素上可比。
它只能平衡已记录变量，无法消除未观测混杂；最终仍需随机或准随机干预验证。

### 3.5 Negative-control operator

加入一个理论上不应改变语义或目标能力的稳定随机桶，例如按 document hash parity 分组，或只改变 metadata 字段顺序。
若 negative control 也稳定“提升”，优先怀疑采样、缓存、训练预算、evaluator leakage 或多重选择，而非宣布新规律。

### 3.6 Mixed-effects model

多个模型、seed 和 source 可写成：

$$
Y_{r,m,s,d}=\mu+\alpha_r+\beta_m+(\alpha\beta)_{r,m}+u_s+v_d+\epsilon.
$$

- $\alpha_r$：recipe 的平均 effect；
- $\beta_m$ 与 interaction：模型及 recipe×model 的可迁移性；
- $u_s,v_d$：seed、source/domain 的随机偏移。

它可把“总体有效”和“只在某 source / model 有效”拆开。层级少、样本极少时估计会不稳，应用 partial pooling 或
直接报告分层 effect 与区间，不要让复杂模型制造虚假精确度。

---

## 4. Operator interaction：为什么两个好算子组合后可能更差

二阶 interaction 定义为：

$$
I(A,B)=Y(A,B)-Y(A,0)-Y(0,B)+Y(0,0).
$$

代入上表：

$$
I=52.2-52.0-51.5+50.0=-1.3.
$$

若效果可加，组合应为 $50+2+1.5=53.5$；实际只有 52.2，多出的 $-1.3$ 表示冗余、冲突或共同造成的 coverage loss。
若组合结果为 54.5，则 $I=+1.0$，表明协同；若为 49，可能出现强冲突或保留数据过少。

interaction 不是抽象统计项，它对应具体机制：

- 两个 scorer 同源，重复筛掉同一批样本；
- 两个阈值叠加后保留率太低，训练 exposure 不足；
- alignment 与 aesthetics 同时提高，却把稀有实体、复杂关系或真实噪声清空；
- 顺序敏感的 mapper/filter 改变了后续算子的输入分布。

### 算子多了怎么办

$K$ 个二元 operator 的 full factorial 需要 $2^K$ 个 arms。预算有限时：

- **fractional factorial**：选一个平衡子集估计 main effects 与预先关心的少数交互，代价是某些效应彼此 alias；
- **稀疏 ANOVA / 正则回归**：假设大多数高阶交互接近 0，只恢复少数大效应；
- **sequential design**：第一轮筛 main effects，第二轮只展开 top candidates 的 pairwise / threshold interaction。

不能先按单算子排名挑 top-5，再只报告赢家组合；筛选过程本身必须进入不确定性和 held-out 合同。

---

## 5. 多模态常用词：typography、human anatomy 与 coverage

### Typography

指**生成图像里文字的视觉与语言正确性**：字母/汉字是否可读、拼写是否正确、字距和行距是否自然、字体风格是否一致、
换行和版式是否符合海报/路牌/UI 场景。它与“模型能否 OCR 输入文字”不同：前者是生成侧渲染质量，后者是理解侧读取能力。

### Human anatomy

指人体结构的可接受性与一致性：手指和肢体数量、关节连接、姿势范围、脸和身体比例、遮挡关系、多人身份一致性。
评测要覆盖手部、运动姿势、多人交互、局部遮挡等高风险条件，也要避免把单一体型、年龄或肤色当作“正常人体”。

### Concept、object、entity coverage

以 “a red Tesla driving past the Eiffel Tower in cyberpunk style” 为例：

| 覆盖 | 问什么 | 例子 | 常见测量 |
|---|---|---|---|
| concept | 是否覆盖语义概念、属性、动作、关系、风格或领域 | red、driving、cyberpunk、urban night | caption/tag embedding、taxonomy、关系抽取 |
| object | 是否出现具体可见物体类别/部件 | car、tower、road | detector/segmenter、object tags |
| entity | 是否出现可唯一链接的命名实体 | Tesla、Eiffel Tower | entity linker、OCR、知识库 |

entity 可能也是 object，但 object 不一定有名字；concept 更宽，可包含不可直接框出的风格、动作和抽象语义。
只统计 object presence 仍会遗漏 attribute、count、spatial relation 与 compositional coverage，实际数据账本应把这些维度分开。

---

## 6. Counterfactual perturbation 与 soft membership

### Counterfactual perturbation：只改一个因素，观察行为是否按预期改变

保持主语义尽量不变，对输入做最小成对修改：

- “two red cups left of the plate” 改为 three 或 right；
- 删除一条关键医学证据，或加入高度相似的干扰证据；
- 替换实体、否定词、工具返回值、图像 crop/occlusion；
- 对 Agent trajectory 注入 tool timeout、stale observation 或 schema change。

若答案对 two→three、left→right 不敏感，就发现了 counting / spatial binding 切片；不需要先给每个训练样本打一个硬标签。
反事实要避免同时改变多个因素，也要检查改写后的输入仍自然、标签仍有效。

### Soft membership：一个样本可同时属于多个切片

样本可能同时具有 OCR、layout、医学知识与长上下文需求。与其用阈值硬分成单一类别，可赋权：

$$
w_{ik}\in[0,1],\qquad
\widehat{Y}_k=\frac{\sum_i w_{ik}Y_i}{\sum_i w_{ik}}.
$$

例如某页医学报告对 OCR、layout、reasoning 的 membership 为 0.8、0.6、0.3。权重可来自校准分类器、embedding mixture、
evaluator rationale 或多标注共识。这样能表达重叠和边界不确定性，避免 0.49/0.51 在硬阈值两侧造成突变。
$w$ 只有在校准后才能解释成概率；否则应称 relevance weight，并同时记录来源和置信度。

---

## 7. 从 Sandbox 的公开观察到 T2I：哪些可迁移，哪些必须重做

Sandbox 没有把 text-to-image 作为主实验，因此下面是机制迁移假设，不是论文已验证结论：

1. **模态对齐大概率迁移**：T2I 同样依赖 caption–image 对齐，低 alignment 会让条件信号变噪；但 CLIP 同源筛选/评测会放大 scorer bias。
2. **输出模态质量更直接**：T2I 输出是图像，aesthetics、watermark、compression、resolution、typography、anatomy 更接近生成目标。
3. **coverage 与 fidelity 有冲突**：激进 aesthetics/NSFW/alignment 过滤可能损失 rare entity、复杂关系、真实人体姿态和文字场景。
4. **阈值和组合不可照搬**：T2V 的 motion/duration、I2T 的输入图像偏好与 T2I 的生成 objective 不同，需重新做 dose-response 和 interaction。
5. **评测需多视角**：至少分 prompt adherence、object/entity/concept/relations、typography、anatomy、aesthetics 与 safety；总分不能替代切片。

一个最小 T2I 验证可用小型 diffusion/flow model，在固定 images-seen 与 compute 下比较 alignment×aesthetics 的 2×2 factorial，
额外设置 rare-entity / typography / anatomy held-out 与 human pairwise eval。只有 target gain 和 guardrails 同时过门，才扩到全量。

---

## 8. 机制总结与迁移检查

完整的 recipe 实验保留 operator、bucket、能力切片三维响应。相关分析用于识别冗余和互补变量，importance 由等预算干预估计。实验先用 paired seeds 和 source-aware bootstrap 描述不确定性，再绘制阈值 dose-response；若 operator pool 在语言、长度、分辨率上不平衡，则使用 matching 或 reweighting。稳定随机桶作为 negative control，跨 model/source 的分层 effect 或 mixed-effects 用于检查迁移。组合实验显式估计二阶 interaction；算子多时使用 fractional factorial 或稀疏 ANOVA，并记录 alias 与选择偏差。最后冻结 top tier，在 target model 与独立 held-out 上验收。
