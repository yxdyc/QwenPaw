# nano-verl L0 — 玩具 actor/learner 调度

> 对应真实系统：[verl](https://github.com/verl-project/verl)（HybridFlow）
> 本文件：`tutorial/material/01-post-training-rl-sft/nano-verl/tutorial_L0.md`
> 可跑文件：`L0_toy_hybridflow.py`

---

## 本节目标

用纯标准库写一个**玩具调度器**，体会后训练 infra 中最核心的一个工程取舍：

> **rollout（采样）和 train（梯度更新）要不要放在同一进程、同一批 GPU 里做？**

跑完这节，你应该能回答：
- 为什么 RLHF/RLVR 训练通常把「采样端」和「训练端」拆开；
- 拆开之后，如何用流水线重叠两者；
- 权重同步在整条链路里扮演什么角色。

L0 不做真实模型，也没有 PyTorch。所有时间都是确定性模拟，CPU 即跑。

---

## 先跑起来

```bash
python3 L0_toy_hybridflow.py
```

真实输出：

```text
nano-verl L0 — toy actor/learner scheduling
config: actions=5, best=2, batch=32, iters=8
        roll=3.0ms, train=5.0ms, sync=0.5ms

[Naive  ] total simulated time: 68.0 ms
         best-action prob: 0.238
         final weights:    [-0.04, -0.05, 0.18, -0.04, -0.05]
[Hybrid ] total simulated time: 47.0 ms
         best-action prob: 0.238
         final weights:    [-0.04, -0.05, 0.18, -0.04, -0.05]

speedup(Hybrid/Naive) = 1.45x

 takeaway: 把 rollout 和 train 放到不同资源并流水线化，
           可以把每轮耗时从 (roll+train+sync) 降到 max(roll,train)+sync。
```

同一个简单任务、同样随机种子，两种调度方式得到的策略权重一样（都是正确的 REINFORCE 更新），但 Hybrid 把总虚拟时间从 68 ms 降到 47 ms，**加速了约 1.45 倍**。这个倍数是 toy 的，真实系统可能更大，也可能更小——取决于 rollout 和 train 的耗时比例。本节不讨论具体倍数，只关注机制。

> **注意**：这里「两种调度最终策略一样」是 toy 的特例——因为我们在确定性种子下用同一个 `policy` 对象，且 REINFORCE 的梯度只依赖当前 rollout 的数据。在真实 LLM 后训练（如 PPO）中，actor 用旧权重 rollout、learner 用新权重训练会引入 **off-policy 偏差**，通常需要用重要性采样（importance sampling）或类似机制处理。L1 开始会涉及这一点。

---

## 机制拆解：rollout vs train

在 RL 后训练里，每一轮大体要干三件事：

1. **Rollout**：用当前 policy 生成响应（推理/采样）。
2. **Train**：拿 rollout 得到的 reward/advantage，算 loss，反向传播更新 policy。
3. **Sync**：把 learner 更新后的权重同步给 actor，让下一轮 rollout 用上新 policy。

 rollout 和 train 的硬件脾气很不一样：

| 维度 | Rollout（推理） | Train（训练） |
|------|----------------|--------------|
| 主要操作 | 自回归生成 token | 前向 + 反向传播 |
| 显存压力 | 相对小（KV cache + activations） | 大（activations + gradients + optimizer states） |
| 批大小偏好 | 小 batch、低延迟 | 大 batch、高吞吐 |
| 典型优化 | vLLM/SGLang 连续批、PagedAttention | FSDP/TP/PP、gradient accumulation |

如果强行把它们塞进同一个进程、同一块显存：
- 显存里既要放训练态的 optimizer states，又要放生成态的 KV cache，容易 OOM；
- 推理和训练 kernel 争抢计算单元，互相打断；
- batch size 迁就训练还是迁就推理，总有一头吃亏。

所以工程上常见的解法是 **actor-learner 分离**：
- actor 进程/资源专门负责 rollout；
- learner 进程/资源专门负责 train；
- 两个角色之间只传递「轨迹数据」和「权重同步」。

---

## 代码走读

### 1. 玩具策略：一个 5 臂 bandit

```python
class ToyBanditPolicy:
    def __init__(self, n=N_ACTIONS, best=BEST_ACTION, lr=0.15):
        self.w = [0.0] * n
        self.best = best
        self.lr = lr
        self.n = n

    def probs(self):
        return softmax(self.w)

    def sample(self):
        p = self.probs()
        r = random.random()
        cum = 0.0
        for i, pi in enumerate(p):
            cum += pi
            if r <= cum:
                return i
        return self.n - 1
```

没有神经网络，只有一个 softmax 策略。选到最优臂给 1.0 分，选到其他臂给 0.1 分。这个 toy 足够说明「policy 会根据 reward 更新」。

### 2. REINFORCE 更新

```python
def update(self, batch):
    baseline = sum(r for _, r in batch) / len(batch)
    p = self.probs()
    grad = [0.0] * self.n
    for a, r in batch:
        advantage = r - baseline
        for i in range(self.n):
            indicator = 1.0 if i == a else 0.0
            grad[i] += advantage * (indicator - p[i])
    for i in range(self.n):
        self.w[i] += self.lr * grad[i] / len(batch)
```

这就是策略梯度：
- `indicator - p[i]` 是 `∇ log π(a)` 对 softmax logits 的导数；
- `(r - baseline)` 降低方差；
- 如果某个动作带来高于平均的 reward，它的权重就往上涨。

### 3. 串行 Naive 调度

```python
def run_naive():
    random.seed(SEED)
    policy = ToyBanditPolicy()
    clock = 0.0

    for it in range(N_ITERS):
        batch = collect_batch(policy, BATCH_SIZE)
        clock += C_ROLL_MS
        policy.update(batch)
        clock += C_TRAIN_MS
        clock += C_SYNC_MS

    print_run("Naive  ", clock, policy)
    return clock
```

每一轮都是「先采样、再训练、再同步」。总时间就是三者简单相加。

### 4. HybridFlow 式流水线

```python
def run_hybrid():
    random.seed(SEED)
    policy = ToyBanditPolicy()
    clock = 0.0
    batches = []

    # startup：actor 先 rollout 第 0 个 batch
    batches.append(collect_batch(policy, BATCH_SIZE))
    clock += C_ROLL_MS

    # steady state：重叠 rollout_i 与 train_{i-1}
    for it in range(1, N_ITERS):
        prev = batches[-1]
        start = clock
        batches.append(collect_batch(policy, BATCH_SIZE))   # actor
        policy.update(prev)                                  # learner
        end_roll = start + C_ROLL_MS
        end_train = start + C_TRAIN_MS
        clock = max(end_roll, end_train) + C_SYNC_MS

    # drain：训练最后一个 batch
    policy.update(batches[-1])
    clock += C_TRAIN_MS + C_SYNC_MS

    print_run("Hybrid ", clock, policy)
    return clock
```

关键点：
- **startup/drain**：流水线不是从第一天就满负荷，需要预热和收尾；
- **重叠**：actor 和 learner 在同一时间窗里各干各的，每轮耗时变成 `max(roll, train) + sync`；
- **权重同步**：玩具里用同一个 `policy` 对象，所以不需要真的拷贝；真实系统里 learner 会调用 `actor.load_state_dict(...)` 或 RPC 广播新权重。

---

## 与 verl 的对应关系

verl 是一个真实可运行的 RL 训练框架，它的核心设计之一就是 **HybridFlow**：把 rollout 和 train 拆成不同的工作流，再在一个统一的调度层下协调 GPU 资源。

L0 玩具只保留了其中**最本质的三件事**：

| 玩具概念 | verl 中的对应物 | 说明 |
|---------|----------------|------|
| `actor` 采样 | actor model + 推理后端（如 vLLM/SGLang） | 负责生成 response/trajectory `[TODO: verify source]` |
| `learner` 训练 | trainer / worker group | 负责反向传播和参数更新 `[TODO: verify source]` |
| `sync` 权重 | weight synchronization / broadcast | learner 把新权重推给 actor `[TODO: verify source]` |
| 流水线重叠 | HybridFlow scheduler | 让 rollout 和 train 在时间上重叠 `[TODO: verify source]` |

具体源码路径、类名和调度 API 会在 L3 对照实现时给出。L0 阶段只需理解「为什么要分离」以及「分离后怎么省时间」。

---

## 费曼自检

### 讲给外行听：厨房与传菜

想象一家餐厅：
- **rollout** 像「服务员点菜、传菜」：轻量、频繁、需要快速响应；
- **train** 像「后厨炒菜」：重火力、一批一批做、需要时间；
- **sync** 像「把新菜单告诉服务员」：让下一轮点菜按最新规则来。

如果你让同一个服务员既要点菜又要炒菜，他会在厨房和餐桌之间来回跑，两头都耽误。于是餐厅把「前厅服务员」和「后厨厨师」分开，点菜和炒菜可以并行：服务员给客人 A 点菜的同时，厨师在做客人 B 的菜。这就是 actor-learner 分离的直觉。

### 思考题

1. 为什么真实系统里，rollout 和 train 很难用「完全相同的 batch size」？
2. 如果 `C_ROLL_MS == C_TRAIN_MS`，理想情况下 Hybrid 能比 Naive 快多少倍？为什么本节的输出是 1.45 倍而不是 2 倍？
3. 权重同步一次要 0.5 ms，如果同步成本涨到 10 ms，流水线还划算吗？什么情况下会不划算？

### 反例

> "既然 actor-learner 分离好，那我把 SFT 也拆成 actor 和 learner 好不好？"

SFT 没有 rollout 阶段：输入是固定的 `(prompt, completion)`，直接前向算 loss、反向更新。没有生成响应这一步，也就不存在「推理和训练争抢资源」的问题。强行拆分只会增加通信开销，没有意义。**技术取舍要看具体流程，不能照搬。**

---

## 下一步

L1 会进入真实训练循环：用一个真正可训练的小语言模型（字符级 LSTM，约 28K 参数）跑一个最小 PPO，体会 `generate → advantage → update` 的完整链路。

---

## 溯源与声明

- verl 仓库：`https://github.com/verl-project/verl`
- 源码级对应关系在 L0 阶段未逐行核对，已标 `[TODO: verify source]`，L3 会补齐。
- 所有数字来自 `L0_toy_hybridflow.py` 的真实运行输出，无编造。
