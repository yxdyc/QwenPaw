# Reasoning effort 与长程 Agent 训练：从“拉满再压短”到条件计算策略

> **核心问题**：thinking effort、adaptive thinking 和长程 agent 能力在训练上是什么关系？面对低监督密度、
> 超时与稀有长成功轨迹，SFT/RFT 应怎样设 loss、采样和预算课程？
>
> **定位**：01 后训练的跨算法实操专题；把 [SFT mask L0](../nano-llamafactory/tutorial_L0.md)、
> [EpisodeRecord L2](../../cross-track-episode-record/tutorial_L2.md) 与
> [Kimi-K3 agentic RL](kimi-k3-agentic-rl-scale.md) 串成一条训练路线。
>
> **SOTA 快照**：2026-09-15。API 能力来自厂商官方文档；公开训练机制主要来自 Kimi K3/K2.5 报告。
> 闭源模型的内部训练 recipe 未披露处保持 `unknown`。

---

## 1. 第一性原理：effort 是策略的条件变量，不只是 `max_tokens`

对任务 $x$、第 $t$ 轮状态 $h_t$，真正想学的是一个计算分配策略：

$$
b_t\sim\pi_b(b\mid x,h_t,c),
$$

其中 $b_t$ 可以表示思考 token、工具步、墙钟、sandbox CPU 或总代价，$c$ 是用户指定的 effort/latency SLA。
动作策略再在预算条件下运行：

$$
a_t\sim\pi_\theta(a\mid x,h_t,b_t).
$$

部署目标通常不是“每题想得越久越好”，而是质量—成本约束：

$$
\min_\pi\ \mathbb E[C]\quad
\text{s.t.}\quad \mathbb E[Q]\ge q_0,
\quad Q_{critical}\ge q_{critical}.
$$

因此要分开三个概念：

- **hard cap**：`max_tokens`、最大轮数、timeout，超过就截断；
- **effort control**：用户告诉策略偏向 low/high/max；它影响搜索深度，但未必恰好用完一个固定 token 数；
- **adaptive thinking**：模型/服务根据任务和中间 observation 动态决定是否继续思考、是否再调工具。

`effort=high` 与 `max_tokens=20k` 不是同义词。前者是策略条件，后者是资源上限；正确系统同时记录 requested
effort、实际 reasoning/action/tool tokens、轮数、墙钟、termination 和结果质量。

---

## 2. 各家公开接口支持了什么，没公开什么

| 家族 | 当前可观测控制面 | adaptive / interleaved | 公开训练证据边界 |
|---|---|---|---|
| OpenAI GPT-6 Astra | 模型页列出 `low/medium/high/xhigh/max` | 外部可显式选 effort；服务内部如何按题动态分配未公开 | 参数、训练课程、effort expert 是否独立等保持 unknown |
| Claude Fable 5.1 | always-on `adaptive thinking`，默认 effort 为 high，并提供 per-message effort beta | 官方说明 effort 与 query complexity 联合校准；支持工具结果后的 interleaved thinking | 文档描述产品行为，不等于公开了 RL/SFT recipe |
| Kimi K3 | 报告训练 low/high/max 三档 reasoning-effort experts | 长程 rollout 中可依据 observation 继续 acting/verifying；具体 API 合同另查服务文档 | 报告公开 max-budget→anneal、九专家与 MOPD 主线，但数据、全部 reward/teacher 仍未开放 |
| GLM-5.3 | 官方文档列为 forced thinking；GLM 系列同时公开 interleaved/preserved thinking，较早部分版本支持逐轮开关 | 可在工具前后思考，并要求完整回传历史 reasoning content；逐轮开关是外部路由，不等于模型自动选预算 | 产品接口可核验；当前旗舰的完整训练 recipe 未披露 |

一手入口：

- [GPT-6 Astra model：reasoning effort 支持表](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [Claude Fable 5.1 model：always-on adaptive thinking](https://platform.claude.com/docs/en/models/fable-5-1/overview)
- [Claude prompting：effort、interleaved thinking 与长程 agent](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [Kimi K3 technical report](https://arxiv.org/abs/2607.24653)
- [GLM Thinking Mode](https://docs.z.ai/guides/capabilities/thinking-mode)

这张表刻意把“API 可控”与“训练时怎样得到”拆开。接口支持 low/high/max，不能反推出内部一定训练了三个独立
checkpoint；厂商公开了 max→low 课程，也不能推出线上一定直接加载这些专家之一。

---

## 3. 为什么“先把 thinking/round 拉满，再 long-to-short”只对了一半

先放宽预算的价值是真实的：早期策略还不会解题，过早惩罚长度会让它学会提前放弃。难题成功轨迹本来稀少，低预算
进一步把它们压成 all-fail，RLVR 组内没有相对信号。max-budget phase 因而适合完成三件事：

1. 发现任务是否可解以及成功路径大致多长；
2. 为 SFT/OPD 收集稀有成功与关键 recovery；
3. 给每题估计基础预算 $b_0(x)$，而不是全域共用一个长度阈值。

但“所有任务永远 max，再统一砍短”会制造三个问题：

- easy 题被训练成过度搜索，verbosity/tool-use hacking 固化；
- 全部数据来自 max-policy，low-effort deployment 遭遇 distribution shift；
- 机械截断长 CoT 不会产生正确短解，只会制造没有终局的坏 target。

更稳的路线是**质量优先、条件压缩、保留长尾**：

```text
Phase A  feasibility discovery
  high/max budget + 多 rollout → 找到真实成功、失败归因与 b0(x)

Phase B  behavior cloning / recovery bootstrap
  完整成功 SFT + 错误前缀(mask)→修复 suffix + 多预算控制标签

Phase C  on-policy RFT
  同题跨 low/high/max rollout；先过 success gate，再优化成本

Phase D  effort consolidation
  effort-conditioned student；必要时 multi-teacher OPD；保留 hard/max replay

Phase E  paired Pareto gate
  每难度层比较 quality/completion/cost/timeout，不让平均省 token 覆盖 hard regression
```

Kimi K3 报告给出一条公开生产锚：先训练较宽松的 max-budget variant，再逐阶段减小预算系数得到 high/low experts，
并按 domain 人工校准；K2.5 的 Toggle 机制又说明硬预算和自由 scaling 需要交替，避免 length-overfitting。本课程
[Kimi-K3 sim 的预算实验](kimi-k3-agentic-rl-scale.md) §6.1 已运行
`free/budget/toggle` 三臂：硬预算在 easy 题省长度，却在 hard 与未见更难题上掉坑；Toggle 位于二者之间。

所以答案不是固定的 long-to-short decay，而是：**先确认会做，再按题、按域、按阶段学习计算分配，同时持续回放
必须长做的任务。**

---

## 4. 长程 Agent SFT：监督稀疏不等于只能训 final

一条 100 轮 trajectory 中，system/user/tool observation 可能占大多数 token。它们不直接进 CE，却构成每个后续
动作的状态。推荐 mask：

```text
system/user                         0
environment observation/tool result 0
teacher/current desired thinking     1
teacher/current desired tool call    1
old-policy bad action                0
teacher repair suffix/final          1
```

这里的 `thinking` 指训练语料中由该模型显式生成、token-aligned 的 reasoning trace，不等于闭源 API 未返回的内部
chain-of-thought。若服务把 reasoning block 作为独立受保护对象，数据管线只能按其公开 API 合同保存/回传，不能假设
能读取后再当 SFT 标签。

关键不是“100 轮只有一个 final label”，而是每个正确的 model-generated action span 都可以提供 next-token 监督。
不过要区分三类数据：

| 数据 | 直接训练什么 | 不能证明什么 |
|---|---|---|
| 完整成功 expert trajectory | 正确状态下逐步行动、格式与停机 | 遇到自身错误仍能恢复 |
| old-policy bad prefix → teacher repair | 暴露偏移状态下的恢复 suffix | teacher 修复在当前 policy 下可达 |
| 单轮/短轨高密度样本 | 原子工具技能、局部决策 | 长程状态、credit 与副作用可靠性 |

SFT batch 不应只按 episode 个数采样。至少建立：

```text
task/domain × difficulty × horizon × effort × outcome × failure_attribution
```

的分层桶，并同时约束 episode quota 与 trainable-token quota。只按 token mean，少数超长成功轨迹可能吞掉梯度；
只按 episode mean，5-token tool-call 又可能与 500-token reasoning 等权。推荐先声明目标：是在模仿“时间占用”，
还是让“任务”各占一票；常见折中是 sample cap、长度分桶与 domain/trajectory-level 权重，而不是让 padding/packing
偶然决定权重。

---

## 5. 长程 RFT：terminal reward 能训练整条轨迹，但不是 dense credit

对 on-policy trajectory $\tau$，序列级回报可以广播给 action token：

$$
L_{PG}=-\frac{1}{\sum_tm_t}\sum_t m_t A_t\log\pi_\theta(a_t\mid h_t).
$$

这里 $m_t$ 只选当前 policy 生成的 thinking/tool-call/final token。observation 是条件；旧 policy prefix 和 teacher
repair 不能冒充当前 behavior action。把同一个 terminal reward 写到每个 token 并不等于知道哪一步做对了，
它只是共享 advantage。轨迹越长，credit variance、陈旧度和失败归因越难。

信号选择应按可信度逐级增加：

1. **terminal verifier**：最终代码测试、环境目标、医疗答案等；最可信但最稀疏；
2. **process checks**：只使用能客观验证的中间状态，不把 judge 偏好伪装成事实；
3. **pairwise/global reward model**：覆盖不可验证任务，但必须监控 Goodhart；
4. **on-policy distillation**：学生自己走到的 prefix 上，由 teacher 给 token-level dense signal；成本转移到 teacher serving；
5. **recovery SFT replay**：把高价值失败变成 mask prefix + verified suffix，作为下一轮 bootstrap。

GRPO/RLVR 选题也不能只看当前 pass rate。all-fail group 没有 relative gradient，但任务可能通过 teacher repair、
更高预算或 curriculum 变得可学。训练价值至少联合考虑当前可解性、学习增量、verifier 可信度和剩余 headroom。

---

## 6. timeout 与稀有长成功：不要静默删掉

长任务的数据分布天然是截尾分布。若只保留成功轨迹，得到的是：

$$
p(\tau\mid success,\ completion),
$$

不是部署时的 $p(\tau)$。最慢、最需要恢复的失败被 censor 后，模型看上去高效，实际只是数据管线没把失败送进来。

每条轨迹至少记录：

- `done`、`truncated`、timeout/cancel/tool-error/max-steps；
- completed rounds、模型 token、tool wait、wall-clock；
- environment snapshot 与 resume handle；
- policy version、每轮 sampled token IDs/log-prob；
- verifier/reward version与失败归因；
- 是否进入 SFT、policy loss、recovery queue 或 infra quarantine。

对稀有长成功，不应简单重复 N 次。优先保留任务权重不变，用 trajectory-aware sampler 增加被看到的机会，并对重复
次数、来源 checkpoint 和近重复簇做 lineage；否则某条偶然成功会被过拟合成模板。对 timeout，先区分模型策略、
工具延迟和 harness 故障；只有模型可归因的截断才适合作为训练难例。

---

## 7. adaptive effort 的训练数据怎样造

最有信息的数据不是给每题只跑一个预算，而是对同一任务做 paired budget sweep：

```text
same task/env/seed family
  ├── low  → quality, tokens, rounds, timeout, trace
  ├── high → quality, tokens, rounds, timeout, trace
  └── max  → quality, tokens, rounds, timeout, trace
```

由此标注“满足质量门的最小 effort”：

$$
e^*(x)=\min\{e:LCB(Q(x,e)-q_0)\ge0\}.
$$

然后训练两部分：

- `effort selector/router`：根据任务与当前状态预测 $e^*$ 或继续/停止；
- `effort-conditioned policy`：在 effort 控制下产生对应深度的行动。

如果只训练 selector 而底层 policy 没见过 low/max 两端，路由没有可控对象；如果只训练多个 effort experts 而不做
consolidation/control，线上需要多 checkpoint 调度。MOPD 是一种集成路线，但必须检查最坏 effort×domain 单元格，
不能只看总体均值。

一个实用 reward 结构是先门控质量，再谈成本：

$$
R(x,\tau)=
\begin{cases}
Q(x,\tau), & \text{尚未越过可解/质量门};\\
Q(x,\tau)-\lambda_d C(\tau), & \text{质量门后，且按 domain/difficulty 定 }\lambda_d.
\end{cases}
$$

这比从第一步就统一施加长度惩罚更稳。成本也不能只算 visible CoT：tool-call arguments、调用次数、环境等待、失败重试
都可能成为 verbosity 转移的新出口。

---

## 8. 推荐的最小实验矩阵

不要一上来同时改数据、reward、预算和 async infra。固定 parent、harness 与 evaluator，依次做：

| 实验 | 唯一主要变化 | 关键指标 | go/no-go |
|---|---|---|---|
| E0 baseline | 当前短 SFT/RFT | success、completion、cost、timeout | 建立分层基线 |
| E1 long-success SFT | 加完整长成功 | hard success、easy regression | hard 增益且 easy 不退 |
| E2 recovery SFT | bad prefix mask + verified suffix | rescue/harm、suffix reachability | net rescue 为正 |
| E3 max-budget RL | 放宽预算，不加效率罚 | solvability、长尾、dead groups | 找到可学习区而非只变长 |
| E4 Toggle/conditional cost | 质量门后加成本课程 | Pareto frontier、hard LCB | 省成本且 hard non-inferior |
| E5 adaptive selector | paired budget labels | selector calibration、oracle gap | 接近最小可行 effort |
| E6 OPD consolidation | 固定 experts/student 加 OPD | effort×domain worst-cell | 不以均值掩盖遗忘 |

每步保留 candidate-parent paired traces；promotion 依赖 fresh hidden tasks、失败率和成本，不依赖训练 benchmark 的单一
提点。长程任务还应把 harness timeout 和环境不稳定单列，避免把 infra 修复误归因为模型能力。

---

## 9. infra 最小合同

长程 adaptive training 会同时压力测试四种状态：

- **model state**：policy/teacher/reward/evaluator version；
- **token state**：append-only sampled IDs、behavior log-prob、mask、KV lineage；
- **environment state**：sandbox、文件、进程、工具副作用、pause/resume；
- **training state**：rollout group、advantage、partial-rollout staleness、admission 与 loss denominator。

推荐的所有权是：Harbor 保存真实 environment episode；canonical ledger 保存 token/provenance；slime 负责 rollout
调度、segment/packing 和 trainer adapter；Evaluation Gate 负责 candidate-parent 晋升。对象能 pickle 或 tensor 能
load 都不是恢复证明，必须重放下一动作/observation 和 next-batch identity。

partial rollout 解决的是训练不必等待 straggler，不是让一条轨迹更快结束。λ 越激进，旧 prefix 越陈旧；需要
per-token policy version、off-policy mask/regularization，以及可恢复 sandbox/KV。完整算术见
[Kimi-K3 deep-dive](kimi-k3-agentic-rl-scale.md)；flat/split 与 token mask 见
[EpisodeRecord L2](../../cross-track-episode-record/tutorial_L2.md)。

---

## 10. 常见错误

1. **只把 max CoT 截短当 long-to-short**：截断会删 terminal 和因果闭环；应重新采样/重写 verified short solution。
2. **把 final reward 当逐步真值**：广播 advantage 仍是稀疏 credit，不应声称知道关键步骤。
3. **成功轨迹全训、失败全丢**：制造 success-conditioned censoring；失败应进入归因、recovery 或 infra quarantine。
4. **tool result 进 loss**：会训练模型伪造环境；observation 只作条件。
5. **按轮简单平均 loss**：短工具轮被放大；先声明 token/episode/trajectory estimand。
6. **一个全局长度惩罚**：easy/hard 与 domain 的合理预算不同；过早效率化会造成 length-overfitting。
7. **只看平均 benchmark**：adaptive/OPD 最容易牺牲 hard/max 小单元格；必须报告 worst-cell 与 timeout。
8. **把 API effort 反推训练 recipe**：外部控制面不揭示内部专家、router、RL 和蒸馏实现。

---

## 11. 费曼自检

1. 为什么 adaptive thinking 不能简化为“让模型自己决定输出多少 token”？
2. 为什么 max-budget-first 有合理性，但 all-max SFT 又可能损害 low-effort deployment？
3. terminal reward 广播到全部 action token 后，为什么仍叫稀疏 credit？
4. 一条 old-policy 失败轨迹怎样同时贡献 SFT 与 RL，而不把 teacher suffix 塞进 PPO？
5. paired budget sweep 比每题随机分配一个 effort 多提供了什么可识别信息？
6. long-to-short 后平均 token 降 30%，还需要哪些证据才能晋升？

<details>
<summary>参考答案</summary>

1. adaptive 策略还要决定是否验证、是否调用工具、何时在 observation 后重新规划，并受质量和资源约束；visible token 长度只是成本的一部分。
2. 宽预算避免策略尚未学会时被迫早停；但只模仿 max-policy 会把过度搜索固化，并让 low 条件没有对应训练分布。应混合 effort、按题压缩并保留 hard/max replay。
3. 所有 token 收到同一个 outcome-derived advantage，并没有局部标签说明哪一步造成成功；轨迹越长，归因方差越大。
4. 保存同一完整上下文：旧错误动作与 observation mask；verified teacher repair 只进 SFT adapter；RL 重新从当前 policy rollout，只让带 exact behavior log-prob 的当前动作进入 policy mask。
5. 它观察同一任务在不同预算下的质量—成本响应曲线，可估计最小可行 effort 和边际收益；随机单点把任务难度与预算效应混在一起。
6. 至少要有同 parent/harness 的 paired quality、completion/timeout、hard/critical non-inferiority、未见难度泛化、成本与失败归因；还要确认减少不是由截断或 harness censoring 造成。

</details>

一句话验收：**长程训练先学会完成，再学会按状态分配计算；effort 是受质量门约束的策略变量，不是统一砍掉 CoT。**
