# 为什么模型几个月就能追平：能力增量、蒸馏与 benchmark validity

> **核心问题**：为什么旗舰模型能在几个月内显著进化、不同厂商 gap 很快缩小？“蒸馏刷榜”具体怎样发生，
> 什么时候是有效能力迁移，什么时候只是 benchmark-local imitation？
>
> **定位**：[Frontier Model Lifecycle](README.md) 的后训练归因专题；不做厂商排行榜，而把可行性、证据强度
> 和发布门拆开。
>
> **证据边界**：公开论文能证明某些机制可行，无法证明任何闭源厂商的全部数据来源、teacher、训练成本或
> benchmark 合规性。未公开部分只能作为带标签的机制推断。

---

## 1. “模型能力”不是一块每次从零铸造的铁

一次产品分数可以粗写成：

$$
Y=f(B,D_{pre},D_{post},R,T,S,H,E),
$$

其中 $B$ 是 base，$D$ 是数据，$R$ 是 SFT/RL/OPD recipe，$T$ 是 teacher/verifier，$S$ 是 inference-time
compute，$H$ 是 harness/tool，$E$ 是 evaluator。几个月内的大幅变化未必意味着重新训练了一个数量级更大的
base；常见情况是围绕同一强底座并行改动多个后半程变量：

- 修复数据覆盖、格式、拒答、工具协议和语言/domain 长尾；
- 增加 reasoning/agentic RL 与 inference compute；
- 用更强 teacher 生成、批改、比较或提供 logits；
- 改善 sandbox、搜索、代码执行、context compaction 和失败恢复；
- 重新做量化、serving、采样预算与产品路由；
- 修正 benchmark harness、prompt、timeout 或 judge。

其中一些是真模型增量，一些是系统增量，还有一些只是测量变化。它们都能让同一公开表格上升，所以
“发布间隔短”不能单独回答提升来自哪里。

---

## 2. 为什么追平速度会越来越快

### 2.1 后发团队面对的不是空白搜索空间

先行模型一旦公开论文、权重、API 行为和失败样例，后发者获得了四种压缩后的信息：

1. **架构先验**：哪些 attention/MoE/optimizer 路线值得试；
2. **行为 oracle**：强模型能给候选答案、轨迹、偏好和失败修复；
3. **评测坐标**：公开 benchmark 暴露能力象限和可复现 harness；
4. **系统模板**：vLLM/SGLang、verl/slime、sandbox 与数据框架降低实现固定成本。

这相当于把“发明问题 + 搜索方法 + 找数据 + 建 infra”缩成“在已知方向上做更密的实验”。它解释 convergence
的可行性，但不证明任一团队实际复制了别家的私有输出。

### 2.2 后训练的边际实验周期短于完整 pretraining

若 base 已经有知识和基本推理能力，SFT/RL/蒸馏主要改变“在什么状态下表现哪种行为”。单项实验可以只覆盖一个
domain、一个数据混合或一个 reward 版本，失败后回滚；完整 pretraining 则同时绑定数据、架构、optimizer 和大规模
并行，反馈周期更长。于是生产关系自然演化为：专项小队并行压能力象限，再做数据或 teacher 集成。

### 2.3 benchmark 是高带宽反馈，但也容易被过拟合

一个固定 benchmark 同时给出目标格式、难度分布与自动分数。它能加速诊断，也让局部优化非常便宜：只要在邻近
任务族中生成足够多的 teacher 数据、做 rejection sampling，再针对失败簇迭代，分数可以快速提升。若没有 fresh
holdout、变体和 contamination 审计，我们无法区分“学会任务族”与“记住测试流形”。

---

## 3. “蒸馏刷榜”通常是怎样一条流水线

“蒸馏”不是单个 loss，而是一族 teacher-assisted data/optimization 路线：

```text
目标能力/benchmark failure clusters
        ↓
任务扩增：同构变体、难度阶梯、反事实、格式扰动
        ↓
强 teacher 多采样：不同 effort / prompt / tool / seed
        ↓
规则 verifier + judge + 去污染 + 多样性过滤
        ↓
verified successes / preference pairs / repair suffix / teacher logits
        ↓
SFT → preference/RLVR → on-policy distillation
        ↓
固定 public dev + fresh hidden family + OOD + cost/failure gate
```

具体可以分四档。

### 3.1 黑盒 response distillation

teacher 输出完整答案或 CoT，经过 verifier/rejection sampling 后，student 做 masked next-token SFT：

$$
L_{SFT}=-\mathbb E_{(x,y_T)}\sum_t m_t\log p_\theta(y_{T,t}\mid x,y_{T,<t}).
$$

成本最低、最容易复用 API，但 student 只看到 teacher 访问过的 prefix。长生成时 student 自己犯错后进入的新状态没有
教师监督，产生 exposure gap。

### 3.2 preference / verifier distillation

对 student 或多个候选 rollout，由 teacher/judge/verifier 排序，形成 chosen/rejected、process feedback 或标量 reward。
它不要求 teacher 暴露 logits，能覆盖“哪个更好”，但 reward hacking 与 judge bias 会进入 student。

### 3.3 on-policy distillation

student 自己采样 prefix，teacher 在这些状态上给 token 分布或 sampled-token advantage：

$$
x,y\sim p_\theta,\qquad
L_{OPD}\approx D\bigl(p_\theta(\cdot\mid x,y_{<t}),p_T(\cdot\mid x,y_{<t})\bigr).
$$

它直接纠正 student 自己会走到的状态，和 RL 共用 rollout infra；代价是在线 teacher serving、版本/路由、logit
带宽和 staleness。课程 [nano-opd L0](../01-post-training-rl-sft/nano-opd/tutorial_L0.md) 与
[L1](../01-post-training-rl-sft/nano-opd/tutorial_L1.md) 已用真实梯度对照离线 SFT 与 OPD。

### 3.4 multi-teacher capability consolidation

数学、代码、Agent、医疗、写作等 teacher 可以并行训练，再由 student rollout 上的 router 选择对应 teacher。
这能降低专项团队之间的训练耦合，但不消除模型容量冲突：teacher routing、数据配比、shared representation 和最坏域
遗忘仍需联合检查。公开锚点包括 [MOPD](https://arxiv.org/abs/2606.30406)；当前课程的
[Capability Factory](../cross-track-capability-factory/) 把 teacher estimator、路由和 promotion gate 放在同一 toy 中。

---

## 4. 为什么少量 benchmark 可以“刷得很快”

假设 benchmark 只有 $M$ 个公开题型，每题能由 teacher 生成 $K$ 个解法，再做 $V$ 种语义保持变体，数据量近似：

$$
N\approx M\times K\times V.
$$

即使 $M$ 不大，$K$ 和 $V$ 也能快速放大监督量。自动 verifier 又让筛选成本远低于人工标注。更关键的是，训练不必
覆盖整个自然语言分布，只要在 benchmark 附近提高：

$$
p_\theta(y\mid x\in\mathcal N(B)).
$$

因此局部 5pp 并不神秘，也不自动代表作弊；真正问题是邻域 $\mathcal N(B)$ 有多窄，以及测试是否仍独立。

| 结果 | 最合理解释 | 证据强度 |
|---|---|---|
| public benchmark 上升，模板扰动即消失 | format/prompt overfit | 很弱 |
| 同题族 fresh hidden variants 也上升 | task-family learning | 中等 |
| 新机构、时间切分、语言与难度均迁移 | domain capability | 较强 |
| 长程环境成功、失败率和成本也改善 | deployable system capability | 更强 |
| 只报告 pass@大 k，单次成功不变 | inference/search 增益 | 不能写成单次 policy 增益 |

“刷榜容易”最准确的说法是：**固定测量表面的局部拟合样本效率很高；跨分布、单次可靠、低成本且无回归的能力
迁移仍然难。**

---

## 5. 哪些做法有效，哪些会越过 benchmark validity 边界

### 合法且有价值

- 根据公开 benchmark 暴露的能力类别，自建独立题目和环境；
- 用 teacher 生成解法，但测试题保持不可见；
- 从训练日志归纳 failure taxonomy，再生成反事实与难度阶梯；
- 用 public dev 调 recipe，用预先冻结的 fresh hidden family 做最终选择；
- 报告训练 benchmark、fresh in-domain、OOD 与部署任务四层结果。

### 必须隔离或披露

- teacher 直接看到正式 test prompt 并生成答案；
- 把 benchmark 解答、judge rationale 或 leaderboard 反馈回灌训练；
- 多轮手工试 prompt 后只报告最优结果，却不计 selection budget；
- evaluator 与训练 reward 同源，且没有独立 verifier；
- 失败/timeout 从分母消失，或不同模型得到不同工具、effort、pass@$k$。

“是否违规”取决于 benchmark 许可和提交规则；“是否仍能证明泛化”则是更严格的科学问题。即使规则允许用公开测试集
开发，重复在同一测试集选择 checkpoint 后，它也不再是独立泛化证据。

---

## 6. 蒸馏为什么可行，又为什么不是无限可行

### 可行条件

1. student base 已经具备承载能力，只缺激活、格式或策略；
2. teacher 在目标 slice 上确实更强，且 verifier 能筛掉主要错误；
3. 任务可生成大量多样变体，而非只复述固定答案；
4. student 容量足以同时表示多个 teacher mode；
5. 线上分布与训练/OPD rollout 的 state distribution 足够接近。

### 硬边界

- **容量边界**：小 student 无法无损容纳所有 teacher；reverse KL 还可能选一个 mode 而放弃另一个；
- **知识/环境边界**：teacher API 给的是行为样本，不给底层权重、隐状态和完整世界覆盖；
- **错误继承**：teacher hallucination、风格偏差和 reward exploit 会被高保真复制；
- **长程 exposure**：离线 teacher trajectory 覆盖不了 student 自己的全部偏移状态；
- **评测饱和**：越围绕固定榜单优化，新增分数越可能来自窄邻域和 selection overfit；
- **成本边界**：multi-teacher full-vocab OPD 需要大量 logits/hidden-state 服务，不再是“便宜抄答案”。

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 是 reasoning distillation 可行性的公开锚点；MiniLLM、GKD、
DistiLLM 与 MOPD 则说明采样分布、散度和多教师路由本身就是问题。论文结果证明各自设定，不应外推成“任意模型、
任意任务都能靠蒸馏追平”。

---

## 7. 生产上怎样防止“局部胜利被写成模型代际进化”

### 7.1 四张分开的账

| 账本 | 必须记录 |
|---|---|
| model/stage | base parent、CPT/SFT/RL/OPD checkpoint、teacher/router、数据 snapshot |
| inference | effort、temperature、pass@$k$、tools/search、context、timeout、cost |
| evaluation | prompt template、judge/verifier、失败分母、污染规则、版本 |
| selection | 尝试过的 recipe/checkpoint 数、public dev 使用次数、最终 hidden gate |

### 7.2 能力向量，不只看平均分

专项 teacher 合版后至少检查：

$$
\Delta=(\Delta_{domain_1},\ldots,\Delta_{domain_K},
\Delta_{general},\Delta_{safety},\Delta_{cost},\Delta_{failure}).
$$

promotion 不是 $\operatorname{mean}(\Delta)>0$，而是同时满足关键域 non-inferiority、hidden gain、失败率与成本门。
平均 +5pp 可能由一个 benchmark +20pp 和三个关键域 -5pp 构成。

### 7.3 fresh evidence ladder

```text
public benchmark
  → format perturbation
  → generated but independently authored variants
  → time/source split hidden set
  → live environment tasks
  → post-deployment fresh logs
```

越靠后越难被 benchmark-local imitation解释。若只有第一层，正确表述是“在该公开合同上提升”；只有经过后几层，才逐步
升级为 task-family、domain 或部署能力主张。

---

## 8. 对专项小队 → 数据合版/MOPD 生产模式的判断

这套认知总体正确，但应增加一个中间门：专项小队交付的不是“高了 5pp 的模型”，而是一个可审计 capability package：

```text
teacher checkpoint + parent lineage
data/reward/verifier snapshot
target slice + fresh holdout
paired gains + failure/cost vector
known conflicts + replay traces
```

然后再选合版方式：

- 冲突小、监督格式兼容、teacher 不必在线：优先数据配比混训；
- student 常进入 teacher 未覆盖状态，或需多个强专家动态纠偏：考虑 MOPD；
- 只有最终 checkpoint、没有数据/logits：parameter merge 可做候选生成，但必须重新过全套 gate；
- benchmark 增益过于局部、fresh variants 不复现：不应进入总模型主干，只保留诊断资产。

集成前先做 data-only 小预算试验通常最便宜；MOPD 的价值不是自动更强，而是让专项 teacher 在 student 自己的状态上
提供 dense supervision，并允许团队并行演进。代价则是 teacher serving、router lineage、staleness 与最坏域治理。

---

## 9. 费曼自检

1. 同一个 base 三个月后 benchmark +10pp，为什么不能直接说“预训练能力大幅进化”？
2. teacher 生成一百万条与 benchmark 相似但不重复的题，算不算数据泄漏？还缺哪些判断？
3. 为什么 OPD 比静态 teacher SFT 更适合纠正 student 自己的长程错误状态？
4. public benchmark、fresh variant 和 live task 分别支持多强的能力主张？
5. 一个 7B student 在数学 teacher 和写作 teacher 间发生冲突，增加 teacher 数为什么不能自动解决？
6. “后发厂商追平”最少要固定哪些 inference/evaluation 条件才可比较？

<details>
<summary>参考答案</summary>

1. 同 base 的变化可能来自后训练数据、RL/蒸馏、更多 inference compute、工具/harness 或评测口径；需要 stage lineage 和固定合同的 paired replay 才能归因。
2. 是否泄漏取决于原题/答案可达性、生成来源、许可和语义近重复；科学上还需独立作者/时间/机构的 hidden set、去污染与变体迁移，证明不是只拟合测试邻域。
3. OPD 在 student 实际采样的 prefix 上查询 teacher，训练分布更贴近部署时 student 会访问的状态；静态 SFT 只覆盖 teacher rollout 的状态。它仍受 teacher、容量和在线成本限制。
4. public 分数只证明固定合同；fresh 同题族支持 task-family；live 环境若固定失败分母、成本和工具合同，才能进一步支持部署系统能力。
5. 冲突来自 student 容量和 shared representation；更多 teacher 可能增加相互矛盾的梯度。需要 routing、权重、容量诊断与最坏域 gate。
6. 至少固定 model identity、prompt/harness、tools/search、effort/token budget、temperature/pass@$k$、timeout、judge/verifier、失败分母和 selection 次数。

</details>

一句话验收：**蒸馏让已知能力的复制与局部适配很快；fresh、长程、低成本、低失败且无回归的能力，仍必须靠独立证据逐层买回来。**
