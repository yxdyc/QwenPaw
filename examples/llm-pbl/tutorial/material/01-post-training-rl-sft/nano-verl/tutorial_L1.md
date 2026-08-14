# nano-verl L1 — 最小 PPO 循环（真实可训练模型）

> 对应真实系统：[verl](https://github.com/verl-project/verl)（HybridFlow）
> 本文件：`tutorial/material/01-post-training-rl-sft/nano-verl/tutorial_L1.md`
> 可跑文件：`L1_minimal_ppo.py`

---

## 本节目标

L0 用 bandit 玩具讲清了 **actor/learner 为什么要分离**；L1 进入真实训练循环，回答：

- 一个最小 RLHF 训练循环长什么样？
- `generate → advantage → update` 三步各自在算什么？
- 为什么真实后训练通常先 SFT、再 RL？
- PPO 的 clip 和 GAE 到底在解决什么问题？

本节的模型是一个字符级 LSTM（约 28K 参数），依赖只有 `torch`，CPU/MPS 均可跑，不需要 transformers。

---

## 先跑起来

```bash
# 需要 torch；本例在 MPS/CPU 上都能跑
python3 L1_minimal_ppo.py
```

修复 state padding 后的一次真实输出（CPU，seed=42；不同 PyTorch 版本和设备的具体数值可能略有不同）：

```text
nano-verl L1 — minimal PPO on a tiny LSTM
device: cpu, model params: 27,966
prompt='go:' target='hello' max_resp_len=5
config: rollouts=32, outer_iter=80, ppo_epochs=4, lr=0.001
        warmup_steps=100

[before warmup] random samples:
  'knms<pad>'  reward=0.00
  'hjsmh'  reward=0.20
  'uxbzu'  reward=0.00
  '<pad>bwo:'  reward=0.00
  'g<sos>rsr'  reward=0.00

[warmup] cross-entropy loss=1.2933

[iter   0] reward=0.831 (best=1.000) policy_loss=-0.005 value_loss=1.073 entropy=0.549 approx_kl=0.0004
[iter  10] reward=0.888 (best=1.000) policy_loss=-0.015 value_loss=0.015 entropy=0.379 approx_kl=0.0056
[iter  20] reward=0.925 (best=1.000) policy_loss=-0.005 value_loss=0.019 entropy=0.197 approx_kl=0.0006
[iter  30] reward=1.000 (best=1.000) policy_loss=-0.002 value_loss=0.001 entropy=0.108 approx_kl=0.0000
[iter  40] reward=0.994 (best=1.000) policy_loss=-0.001 value_loss=0.001 entropy=0.071 approx_kl=0.0000
[iter  50] reward=0.969 (best=1.000) policy_loss=-0.006 value_loss=0.012 entropy=0.045 approx_kl=0.0004
[iter  60] reward=0.963 (best=1.000) policy_loss=-0.008 value_loss=0.010 entropy=0.041 approx_kl=0.0016
[iter  70] reward=1.000 (best=1.000) policy_loss=0.000 value_loss=0.001 entropy=0.031 approx_kl=0.0000
[iter  79] reward=0.988 (best=1.000) policy_loss=-0.010 value_loss=0.001 entropy=0.024 approx_kl=0.0016

[after PPO] samples:
  'hello'  reward=1.00 ✅
  'hello'  reward=1.00 ✅
  'hello'  reward=1.00 ✅
  'hello'  reward=1.00 ✅
  'hello'  reward=1.00 ✅
  'hello'  reward=1.00 ✅
  'hello'  reward=1.00 ✅
  'hello'  reward=1.00 ✅

takeaway: 极小 LSTM 经 SFT warmup 后，PPO 能把 'go:' 稳定续写成 'hello'。
          真实后训练也遵循同样顺序：SFT model → RL；L2 再把 rollout 与 train 拆开。
```

从随机乱码开始，经过 100 步 SFT warmup + 80 轮 PPO，模型把 `go:` 稳定续写成 `hello`。这是一个**真实可训练**的最小闭环。

---

## 机制拆解：从 SFT 到 PPO

### 1. 为什么先 SFT warmup？

PPO 依赖采样。如果初始策略完全随机，采样到高质量 response 的概率极低，训练信号就会很差。本节的 toy 任务尤其明显：随机策略几乎不可能一次性猜中 `hello`，模型很容易陷入「重复某个局部高奖励字符」（比如一直输出 `ooooo`）的局部最优。

因此真实后训练几乎总是：

```text
pretrain → SFT → RL（PPO/DPO/GRPO 等）
```

> **PPO 的当今定位（2026-08 视角）**：PPO（arXiv:1707.06347，2017）是经典锚点，但**已不是前沿 RLVR 的首选算法**——
> 前沿后训练（如 Qwen3、Kimi 系列的 RL 阶段）主流是 GRPO 族（DAPO / GSPO / CISPO 等）与 RLVR：
> 它们去掉了 value model，用组内相对 reward 估计 advantage。但 PPO 的核心思想——
> importance sampling ratio + clipping 限制策略更新幅度——**直接流入了 GRPO 族**，
> 读懂 PPO 是读懂这些新方法的前提。本节按经典教机制，不把它当作「当前前沿就是这么训的」。
> （GRPO：arXiv:2402.03300，DeepSeekMath 提出 `[TODO: verify 后续变体的 arXiv ID]`）

SFT 给模型一个**合理的初始策略**，RL 在此基础上做策略优化。本节用 100 步 teacher-force warmup 模拟这个环节：

```python
def supervised_warmup(model, optimizer, prompt_ids, device, steps):
    full_seq = torch.tensor([prompt_ids + encode(TARGET)], dtype=torch.long, device=device)
    input_ids = full_seq[:, :-1]
    target_ids = full_seq[:, 1:]
    for _ in range(steps):
        logits, _ = model.forward(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

> **关键认识**：SFT 不是「作弊」，而是 RL 的合理起点。没有它，PPO 要么学不动，要么学到奇怪局部最优。

### 2. Rollout：用旧策略采样

```python
def generate_rollout(model, prompt_ids, max_len, device):
    model.eval()
    buffer = RolloutBuffer()
    seq = list(prompt_ids)

    with torch.no_grad():
        for step in range(max_len):
            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            dist, value = model.last_dist_value(input_ids)
            action_t = dist.sample()
            action = action_t.item()
            log_prob = dist.log_prob(action_t).item()

            buffer.states.append(list(seq))
            buffer.actions.append(action)
            buffer.log_probs.append(log_prob)
            buffer.values.append(value.item())
            buffer.rewards.append(None)
            seq.append(action)

    step_rewards = reward_fn(decode(seq[len(prompt_ids):]))
    for step, r in enumerate(step_rewards):
        buffer.rewards[step] = r
    return buffer, sum(step_rewards), decode(seq[len(prompt_ids):])
```

这里收集的是 **on-policy 数据**：`log_prob` 来自当前策略，`value` 来自当前 value function。后续 PPO 更新会拿这些旧概率做重要性采样。

#### 批处理 state：为什么要右 padding + 真实长度？

一条 response 的各个 state 长度不同。例如生成 `hello` 时，前两个 state 可能是：

```text
s0 = [g, o, :]
s1 = [g, o, :, h]
```

为了组成 batch，L1 将它们**右 padding** 到相同长度，同时保存真实长度：

```python
lengths = [len(s) for s in states]
max_len = max(lengths)
padded = [s + [PAD_ID] * (max_len - len(s)) for s in states]
length_t = torch.tensor(lengths, dtype=torch.long, device=device)
```

PPO update 不取统一的最后一列，而是按 `lengths - 1` 找到每条 state 的最后一个真实 token：

```python
batch_idx = torch.arange(token_ids.size(0), device=token_ids.device)
last_idx = length_t - 1
last_logits = logits[batch_idx, last_idx, :]
last_value = values[batch_idx, last_idx]
```

这样单向 LSTM 在真实末尾产生的输出，与 rollout 时没有 padding 的计算一致；右侧 PAD 虽然随后会被 LSTM 处理，但其输出不会反向影响前面的真实位置。

这里不能直接使用未屏蔽的**左 padding**。如果把 `s0` 变成：

```text
[PAD, PAD, PAD, PAD, g, o, :]
```

LSTM 会先处理四个 PAD，导致进入 `g` 前的 hidden state 已经改变。此时 PPO 实际比较的是：

$$
\frac{\pi_{\text{new}}(a\mid PAD\ldots PAD+s)}
     {\pi_{\text{old}}(a\mid s)}
$$

而正确的 ratio 必须在同一个 state 上比较：

$$
\frac{\pi_{\text{new}}(a\mid s)}
     {\pi_{\text{old}}(a\mid s)}
$$

一个很好用的正确性自检是：在第一次 `optimizer.step()` 之前，新旧参数相同，因此所有 ratio 都应该约等于 1。若明显偏离，通常意味着 old/new log-prob 的输入、mask、位置或采样配置不一致。

本次修复后的 CPU 自检中，ratio 落在 `[0.999999762, 1.000000238]`，最大绝对误差约为 `2.38e-7`（浮点舍入量级）。

### 3. Reward：逐字符匹配

```python
def reward_fn(generated):
    rewards = []
    for i in range(MAX_RESP_LEN):
        c = generated[i] if i < len(generated) else "<pad>"
        rewards.append(1.0 / MAX_RESP_LEN if c == TARGET[i] else 0.0)
    return rewards
```

真实 LLM 后训练里，reward 通常来自 reward model（RM）或规则 verifier。这里用一个确定性的字符匹配函数代替，好处是可控、可复现，且不依赖外部 API。

### 4. GAE：把未来奖励折现成优势

```python
def compute_gae(buffers, gamma, lam):
    for buf in buffers:
        rewards = buf.rewards
        values = buf.values
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
```

GAE（Generalized Advantage Estimation）做两件事：
- 用 `gamma` 把未来奖励折现到现在；
- 用 `lambda` 在「真实回报」和「value 差分」之间做偏差-方差权衡。

`delta = r + gamma * V(s') - V(s)` 是 TD error：如果当前 step 的 reward 比 value 预测的好，这个动作就有正优势。

### 5. PPO update：clip 限制策略突变

```python
for _ in range(n_epochs):
    dist, last_values = model.last_dist_value(state_t)
    new_log_prob = dist.log_prob(action_t)
    ratio = torch.exp(new_log_prob - old_log_prob_t)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
```

- `ratio = π_new(a) / π_old(a)`：重要性采样权重；
- `surr1`：标准策略梯度；
- `surr2`：把 ratio 限制在 `[1-ε, 1+ε]` 内，防止某条数据把策略拉得太远；
- 取 `min`：只让正向优势（好动作）在限制范围内放大，避免过度优化。

这就是 PPO 的核心设计动机：**让同一批 rollout 可以更新多个 epoch，同时限制策略在这批数据上走得过远**。
clip 不等于严格的 KL 约束，因此代码仍输出 `approx_kl` 监控实际策略漂移。

#### Importance sampling：为什么旧样本还能再用几轮？

先固定一个状态 $s$。我们想算新策略下某个量 $f(s,a)$ 的期望，但手里的动作来自旧策略：

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi_{new}}[f(s,a)]
&= \sum_a \pi_{new}(a\mid s) f(s,a) \\
&= \sum_a \pi_{old}(a\mid s)
   \frac{\pi_{new}(a\mid s)}{\pi_{old}(a\mid s)} f(s,a).
\end{aligned}
$$

所以要给旧策略采到的动作乘一个修正权重：

$$
r_t=\frac{\pi_{new}(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
=\exp\!\left(\log\pi_{new}-\log\pi_{old}\right).
$$

在 LLM 中，$s_t$ 是 `prompt + response[:t]`，$a_t$ 是第 $t$ 个 token。rollout 时把
`old_log_prob` 存进 buffer；训练每个 epoch 用当前参数重算 `new_log_prob`。第一次更新前两者应几乎相等，
所以 ratio 约为 1；optimizer 改过参数后，后续 epoch 的 ratio 才开始偏离 1。

例如旧策略给某 token 的概率是 0.20，新策略变成 0.30，ratio 就是 1.5；若降到 0.10，ratio 就是 0.5。
PPO 再用 clip 阻止这些权重无限放大更新。

这个等式有重要边界：它只是在**相同状态 $s$ 上修正动作分布**。旧 rollout 的前缀本身仍来自旧策略；
整条序列的 ratio 连乘还会迅速爆炸或趋零；clip 也会主动引入偏差来换稳定性。因此 PPO 的 IS 不是
「任意旧数据都能安全重放」的许可证，真实系统仍需新鲜 rollout、版本号与 staleness 控制。

想继续深挖，可看[算法演进深挖 §2](../sota-deepdive/post-training-algorithm-evolution.md)及其
[可运行 IS 实验](../sota-deepdive/post_training_evolution_sim.py)；异步系统里「策略版本落后」如何产生，见
[nano-slime L0](../nano-slime/tutorial_L0.md)。

### 6. Value function 同步更新

```python
value_loss = F.mse_loss(last_values, returns)
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

同一个 LSTM 同时输出 policy head 和 value head。value 负责估计 GAE 里的 `V(s)`，必须和实际 return 对齐；entropy 奖励鼓励探索，防止模型过早坍塌到确定性输出。

---

## 与 verl 的对应关系

L1 已经触及真实 RL 训练循环的四要素，但仍然是单进程、单设备版本。verl / HybridFlow 在 L1 基础上增加了工程层面的拆分：

| 本节概念 | verl 中的对应物 | 说明 |
|---------|----------------|------|
| `generate_rollout` | actor/rollout model + 推理后端（vLLM/SGLang）| 负责高吞吐采样 |
| `ppo_update` | actor training worker / trainer | 负责反向传播和参数更新 |
| `RolloutBuffer` | trajectory `DataProto` / experience queue | 暂存 state、action、reward、value、old log-prob 等字段 |
| `supervised_warmup` | SFT checkpoint 初始化 | 本 toy 没有单独的 reference policy，也没有 KL-to-reference loss |
| 单设备串行 | HybridFlow scheduler | L2 演示 lockstep 角色拆分；L3 演示同步 colocate；异步重叠见 nano-slime L0 |

具体源码路径和 API 会在 L3 对照 verl 实现时给出。L1 先确保「循环本身是对的」。

---

## 费曼自检

### 讲给外行听：教鹦鹉说话

想象你教一只鹦鹉说 "hello"：
- **SFT warmup**：你先反复念 "hello"，让鹦鹉大致模仿出声音；
- **Rollout**：你让鹦鹉自己说一遍，录下来；
- **Reward**：你说对的字符给一颗瓜子，错的没有；
- **GAE / Advantage**：如果鹦鹉已经会说 "he"，那下一个字符说对 "l" 的「边际进步」比从完全乱叫开始更大；
- **PPO clip**：如果某次鹦鹉突然改掉太多发音，你只让它改一小步，防止它彻底不会说了。

PPO 不是让鹦鹉一次性背下整句话，而是让它在已有基础上，稳定地朝高分方向微调。

### 思考题

1. 如果把 `CLIP_EPS` 从 0.2 改成 2.0（基本不 clip），训练可能会出现什么现象？
2. `entropy_coef` 如果设为 0，模型输出会怎样变化？这在长序列生成里为什么危险？
3. 为什么 GAE 要从后往前算，而不是从前往后？

### 反例

> "SFT 已经能让模型输出 'hello' 了，为什么还要 PPO？直接 SFT 到 loss=0 不就行了吗？"

SFT 只能让模型模仿给定答案。真实后训练里，我们没有「标准答案」——只有 reward 信号（比如人类偏好、规则正确性、工具执行结果）。PPO 的价值在于**用 reward 优化策略**，而不是复制固定输出。本节用确定性 target 只是为了教学可控；一旦 reward 来自外部反馈，SFT 就无能为力。

---

## 下一步

L2 会把 `generate_rollout` 和 `ppo_update` 拆到不同进程/设备，用普通 PyTorch batch forward
演示角色隔离、权重同步与资源冲突；真正推理引擎的 continuous batching 在 03 轨展开。

---

## 溯源与声明

- verl 仓库：`https://github.com/verl-project/verl`
- PPO 原始论文：`arXiv:1707.06347`
- GAE 原始论文：`arXiv:1506.02438`
- 真实框架对应关系在 L3 的固定版本源码锚点中展开；当前 main 分支接口可能继续演进。
- 所有数字来自 `L1_minimal_ppo.py` 在 MPS 上的真实运行输出，同 seed 可复现。
- 模型参数量由 `sum(p.numel() for p in model.parameters())` 现场统计：27,966。
