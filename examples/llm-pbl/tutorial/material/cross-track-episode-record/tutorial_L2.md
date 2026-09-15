# EpisodeRecord L2 — 真实多轮、伪多轮与训练切段

> **核心问题**：一条 `Action → Observation → Action` 轨迹，应作为一个 sample 还是拆成多轮？
>
> **先修**：[L0](tutorial_L0.md) 的 provenance/termination 合同与
> [L1](tutorial_L1.md) 的 record→tensor/loss reduction。
>
> **不变量**：是否真实多轮由 observation 的因果来源决定；sample 数量只是训练布局。拆段不得改变 prefix、
> token provenance、loss 分母、behavior log-prob 或 rollout-level reward。
>
> **运行**：`python3 -B L2_trajectory_segmentation.py`；纯标准库、CPU、固定输出。
>
> **验收**：12/12 self-check；flat 与 token-weighted split loss 一致，伪 observation、重 tokenize 和 reward
> 重复计数均被抓住。
>
> **边界**：L2 是 token/provenance 算术，不做真实模型 forward、在线工具调用、KV retention 或分布式存储。

---

## 0. 先拆掉一个混淆：episode 与 sample 不是同一层对象

真实 agent rollout 在时间上逐轮发生：

```text
模型生成 A1 → 环境执行 → 返回 O1 → 模型生成 A2 → 环境执行 → 返回 O2 → Final
```

trainer 最终可以收到一条 flat tensor：

```text
[P0 | A1 | O1 | A2_bad | O2 | A3_repair | Final]
  0    1    0      0       0        1          1     <- SFT loss mask
```

也可以收到两个带累计 prefix 的 physical segments：

```text
segment-1: [P0                           | A1]
segment-2: [P0 | A1 | O1 | A2_bad | O2 | A3_repair | Final]
```

两种布局都可能来自同一条真实轨迹。反过来，把十组无关 QA 写成交替角色，或者让模型自己续写
`Observation:`，即使字符串有十轮，也不是环境意义上的真实多轮。

所以系统应先保存语义对象：

```text
Episode / Trajectory  1 ── N  TrainingSegment
```

而不是在 schema 中写死 `Sample == Episode`。

---

## 1. 先跑：同一轨迹的两种布局

```bash
python3 -B L2_trajectory_segmentation.py
```

稳定输出：

```text
==============================================================================
EpisodeRecord L2 — real multi-turn -> flat/split training segments
==============================================================================

[1] One semantic trajectory, two training layouts
    episode=episode-medical-agent-007 rollout=rollout-policy-v12-007
    flat: tokens=17 sft_targets=7 ppo_targets=3
    split: segments=2 physical_tokens=25 (prefixes are repeated)
    observation source=environment and mask=0; old bad action stays in context and mask=0

[2] Loss equivalence requires token-count weighting
    flat token mean       = 0.771429
    split token-weighted  = 0.771429
    split simple turn mean= 0.725000 (different objective)

[3] Fail-closed provenance and token continuity
    model-fabricated observation -> REJECT
    re-tokenized prefix drift     -> REJECT

[4] Terminal reward belongs to the rollout
    naive per-segment copy=2.0 | rollout reducer=1.0

[5] self-check
    PASS | flat sample contains the complete trajectory
    PASS | SFT trains current action plus teacher repair/final
    PASS | PPO trains only exact current-policy sampled tokens
    PASS | environment observations are context-only
    PASS | all physical segments retain one rollout identity
    PASS | flat equals token-weighted split
    PASS | simple turn mean changes the objective
    PASS | turn expansion recomputes accumulated prefixes
    PASS | model-fabricated observation is rejected
    PASS | re-tokenized prefix drift is rejected
    PASS | terminal reward is counted once per rollout
    PASS | PPO targets retain exact behavior logprobs

SELF-CHECK: 12/12 PASS
digest=b05c11059dde6745
takeaway: multi-turn truth lives in environment provenance; sample count is a trainer layout choice.
```

toy 里 flat 只处理 17 个 physical token；逐轮展开因为第二段重复完整历史，总计处理 25 个。真实长轨若把每轮
都展开成 `prefix → current response`，前缀重算会随轮数累积，可能接近二次增长。短序列和灵活 batching 的收益，
要与重复 prefill/backward 的成本一起算。

---

## 2. token provenance 先于 role name

L2 不靠字符串里的 `assistant`、`tool` 猜监督范围，而给每段记录来源与算法 mask：

| span | source | SFT | PPO | 为什么 |
|---|---|---:|---:|---|
| system/user | system/user | 0 | 0 | 条件，不是目标行为 |
| 当前策略的 thinking/tool call | current policy | 1 | 1 | SFT 可模仿；PPO 有精确行为概率 |
| environment observation | environment | 0 | 0 | 只作下一动作的条件 |
| 旧策略错误动作 | old policy | 0 | 0 | 保留恢复现场，不直接模仿，也不冒充当前 on-policy token |
| teacher repair/final | teacher | 1 | 0 | 可作 SFT 修复目标；不是行为策略采样，不能直接塞进 PPO ratio |

这解释了为什么不能只存一个万能 `loss_mask`。SFT、PPO 与 OPD 消费的 estimand 不同；底层应存
`token_source`、`policy_version`、teacher identity 和行为 log-prob，再由 adapter 派生算法 mask。

一个实用 span schema 是：

```text
episode_id, rollout_id, segment_id, parent_segment_id, turn_id
token_ids, token_source, sft_mask, policy_mask, behavior_logprob
action_id, caused_by_action, policy_version, environment_version
terminal_reward, done, truncated, termination_reason
```

raw string 可以保留给审计和环境接口，但训练真值应是生成当时的 token IDs。对 PPO/IS，`mask=1` 的每个
current-policy token 必须有同一前缀、同一 policy version 下的 behavior log-prob。

---

## 3. flat 与 split 何时等价

设第 $j$ 个监督 response 含 $n_j$ 个 token：

$$
L_{flat}=-\frac{\sum_j\sum_{t\in A_j}\log p_\theta(x_t\mid x_{<t})}{\sum_jn_j}.
$$

若逐轮样本先各自取均值 $L_j$，split 只有在下面权重下才与 flat 等价：

$$
L_{split}=\sum_j\frac{n_j}{\sum_kn_k}L_j.
$$

本实验现场得到二者同为 `0.771429`。若改成简单轮均值：

$$
L_{turn}=\frac1J\sum_jL_j=0.725000,
$$

短工具轮与长推理轮各占一票，目标已经改变。等价还要求：

1. 每段包含与原轨迹逐 token 相同的完整 prefix；
2. chat template、tokenizer、position IDs、attention/boundary policy 不变；
3. 这些 loss 在同一次 optimizer update 中按同一分母累积；
4. 没有一条布局截断、另一条布局保留的历史；
5. dropout、并行规约等数值差异另行界定，不能把公式等价写成 bitwise 等价。

因此默认策略很简单：不超 context、无分支、无 compaction 时，一条 trajectory flatten 成一条 sample；只有
长度、分支、partial rollout 或调度约束要求时才切段，并携带共同 `rollout_id` 和全轨 loss 分母。

---

## 4. “伪多轮”至少有四种意思

这个词不是稳定术语，设计评审里必须展开成可检查的类型：

| 常被叫作“伪多轮”的东西 | 实际性质 | 能训练什么 | 主要风险 |
|---|---|---|---|
| 真实历史离线 flatten | offline multi-turn behavior cloning | 给定真实历史续写动作 | 没有 on-policy exposure |
| 每轮展开成 `完整 prefix → 当前回答` | physical single-turn samples from one trajectory | 与 flat 可等价 | prefix 重算、轮均值改权重 |
| 无关 QA 拼成交替角色 | formatting multi-turn | 对话格式、局部回答 | 没有状态依赖或长程 credit |
| 模型同时生成 Action 和 Observation | fabricated-environment completion | 最多是显式 world-model toy | agent 学会伪造工具结果 |

本 L2 用两条 admission gate 区分关键边界：

- observation 必须由 environment 产生，并引用紧邻的 `action_id`；
- 下一轮 token prefix 必须原样包含上一轮已采样 token；decode 成字符串后重新套模板导致的任何 token drift 都拒绝。

真实系统允许 context compaction，但它必须成为显式事件：记录 compactor/version、被替换区间、摘要 token、原始
artifact 指针与新状态起点。不能把 compaction 造成的 prefix 改写伪装成 append-only continuation。

---

## 5. mask 为 0 不等于 detach

对 masked observation，直接 CE 项为零：

$$
m_t=0\Rightarrow m_t\log p_\theta(x_t\mid x_{<t})=0.
$$

但后续 repair token 会 attend 到 observation 和旧错误动作。后续 loss 的梯度仍经过这些上下文的 hidden
representation。这正是训练“读懂工具结果”和“从错误历史恢复”的信号。

如果目标真的是让某段历史完全不影响训练，必须删除/改写它，或采用有明确语义的 frozen-prefix/cache detach；
单纯 `loss_mask=0` 做不到。反过来，不应把 tool observation 设为 target：那会训练模型复述甚至伪造环境输出。

---

## 6. 一条 rollout 切成两段，reward 不能变成两份

脚本把 terminal reward `1.0` 复制到两个 segment，按物理样本求和会得到 `2.0`。正确 reducer 先按
`rollout_id` 聚合，只记一次 `1.0`。生产 RL 还要决定 advantage 如何跨段：

- episode return、group membership、`done/truncated` 属于 trajectory；
- token advantage 可以落到 segment，但分母与 bootstrap 边界必须引用全轨合同；
- branch/root-to-leaf samples 应保留 parent/branch lineage，不能互相冒充独立 prompt group；
- partial rollout 的旧前缀通常 context-only，新鲜 current-policy suffix 才进入 policy loss。

否则 split 数量会成为隐式 sample weight：更长、切段更多的 episode 被训练更多次。

---

## 7. 对接 slime + Harbor 的三层接口

建议把职责固定为：

```text
Harbor episode plane
  real action / observation / env snapshot / timeout / replay artifact
              ↓
Canonical trajectory ledger
  exact token IDs / source / mask / logprob / version / branch lineage
              ↓
slime training assembler
  one-or-many Sample / packing / rollout reducer / trainer admission
```

Slime 的公开多轮示例采用“模型生成 action，工具执行并追加 observation”的真实循环；模型 token 进入 loss，
tool/environment token 只作条件。其 agentic RL 指南进一步强调保留逐轮的原始 prompt/output token IDs 与
log-prob，避免把 decode 后文本重新 tokenize。分支、context compaction 等场景允许一个 rollout 产生多个
`Sample`，但需要共同 rollout identity 与正确聚合。

对应的一手入口：

- [slime multi-turn rollout quick start](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md)
- [slime coding-agent token trajectory](https://github.com/THUDM/slime/blob/main/examples/coding_agent_rl/README.md)
- [slime customization: branches and multiple samples](https://github.com/THUDM/slime/blob/main/docs/en/get_started/customization.md)
- [slime loss aggregation options](https://github.com/THUDM/slime/blob/main/docs/en/get_started/usage.md)

这些链接是实现参照，不代表 L2 已运行 slime 或 Harbor。真正接入时至少增加以下 fail-closed tests：

1. `token_ids/source/mask/logprob` 长度逐位相等；
2. environment token 永远不进 policy/SFT target；
3. PPO mask 中每个 token 都有 admitted policy version 的 behavior log-prob；
4. 下一轮 prompt 满足精确 append-only prefix，或存在显式 compaction event；
5. flat 与 split 的 toy loss 在目标 reduction 下相等；
6. terminal reward、GRPO group 和 train admission 每 rollout 只计一次；
7. old-policy prefix 与 teacher suffix 不被误接入 PPO ratio。

---

## 8. 费曼自检

1. 为什么一条 tensor 可以是真多轮，而十条 tensor 仍可能是伪多轮？
2. split 时每轮 loss 简单平均，为什么会放大短 tool-call？
3. teacher repair 为什么能进入 SFT mask，却不能直接进入 PPO policy mask？
4. observation 已 mask，为什么后续 repair loss 仍能训练模型利用它？
5. 为什么 `rollout_id` 不能用 `segment_id` 代替？
6. context compaction 后不再满足 append-only prefix，应该怎样避免误报 token drift？

<details>
<summary>参考答案</summary>

1. 真多轮取决于 action 是否实际驱动环境产生 observation；tensor 数只描述 trainer 布局。真实轨迹可以 flatten，伪 observation 也可以拆成十段。
2. 简单轮均值令每轮权重均为 $1/J$；一个 5-token tool-call 与一个 500-token 推理各占一半。token mean 下二者权重分别为 $5/505$ 与 $500/505$。
3. SFT 可以把 teacher token 当监督标签；PPO ratio 要求 token 由明确的 behavior policy 在同一前缀采样并有对应 old log-prob。teacher token 不满足这个因果合同。
4. mask 只删除 observation 自身的直接 CE target；后续 token 的预测仍通过 attention 依赖它，梯度由后续 loss 经过其上下文表示传播。
5. 一个 rollout 可切成多个 segment。用 segment ID 聚合会重复 terminal reward、group return 和 admission；rollout ID 才代表语义上的一次环境试验。
6. 把 compaction 记录为显式状态转移，绑定原始区间、摘要/新 prompt token、compactor version 和 artifact digest；只在同一 append-only epoch 内检查严格 prefix。

</details>

一句话验收：**真实多轮是 rollout 的因果属性；flatten/split 是 trainer 的物理布局，后者不得篡改前者。**
