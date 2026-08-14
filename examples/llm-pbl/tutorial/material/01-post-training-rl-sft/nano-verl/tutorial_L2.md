# nano-verl L2 — actor/learner 分离 + batch inference

> 对应真实系统：[verl](https://github.com/verl-project/verl)（HybridFlow）
> 本文件：`tutorial/material/01-post-training-rl-sft/nano-verl/tutorial_L2.md`
> 可跑文件：`L2_actor_learner_split.py`

---

## 本节目标

L1 在单进程里跑通了最小 PPO。真实后训练系统不会这样干——因为 **rollout（推理采样）和 train（参数更新）的硬件脾气完全不同**。L2 回答：

- 为什么要把 rollout 和 train 拆到不同进程/设备？
- batch inference 到底在优化什么？
- 权重同步是怎么发生的？跨进程传 tensor 有哪些坑？
- 什么情况下 actor-learner 分离真的能加速，什么情况下反而会变慢？

本节仍然只用 `torch`，没有 transformers/vLLM/SGLang；用 multiprocessing + Queue + state_dict 传递，搭出分布式 RL 训练系统的最小原型。

> **范围先说清**：本实现仍是 `rollout → train → sync` 的严格 lockstep。它演示的是角色/设备隔离、
> batch forward 和权重快照，**没有让相邻 step 的 rollout 与 train 重叠**。同步 colocate 见
> [L3](tutorial_L3.md)；真正异步解耦、策略陈旧度与背压见 [nano-slime L0](../nano-slime/tutorial_L0.md)。

---

## 先跑起来

```bash
# 需要 torch；默认 actor 用 CPU，learner 用本机最快的 accelerator（CUDA/MPS/CPU fallback）
python3 L2_actor_learner_split.py
```

一次当前 CPU-only 输出（seed=42；wall time 只作本机参考）：

```text
nano-verl L2 — actor/learner split with batch inference
learner device: cpu, actor device: cpu, model params: 237,150
prompt='go:' target='hello' max_resp_len=5
config: rollouts=256, outer_iter=40, ppo_epochs=4, lr=0.001
        warmup_steps=100
hint: set NANO_VERL_ACTOR_DEVICE / NANO_VERL_LEARNER_DEVICE to experiment

[1/2] serial baseline (generate + train in the same process)
      elapsed: 8.43s
      final reward: 0.999 (best=1.000)

[2/2] actor-learner split (lockstep processes + weight sync)
      elapsed: 8.18s
      final reward: 0.999 (best=1.000)

[comparison]
  serial   : 8.43s
  split    : 8.18s
  speedup  : 1.03x
  reward parity: serial=0.999, split=0.999

takeaway: L2 完成角色/设备隔离与显式权重同步，但仍是 lockstep；
          batch inference 摊薄推理开销，CPU/GPU 差异会显著影响这个 toy 的计时。
          L3 讲同步 colocate；真正异步 overlap/staleness 见 nano-slime L0。
```

同一个任务、同一个模型、同为 CPU，8.43s 与 8.18s 基本可视作持平：当前实现并没有把两相变成流水线，
多进程隔离的收益被 Queue、权重快照和调度开销抵消。这反而是更干净的教学结果——**角色拆分不自动等于
overlap，更不自动等于加速**。真实模型上必须分别测 rollout、train、同步和空闲时间再归因。

> **测量口径（值得单独讲一句）**：两边的计时窗口是对齐的——都从「SFT warmup 结束之后」开始：
> serial 在 warmup 后取 `t0`；split 则等 learner 完成 warmup 并发出初始权重（`ready_event`）后才取 `t0`，
> 子进程 spawn、import torch、模型初始化都不计入。窗口内保留的是 split 版无法回避的真实开销：
> 第一条 rollout 的等待与生成。口径不对齐的加速比是自欺欺人——
> 如果 split 把 warmup 算进窗口、serial 不算，报出的 speedup 会系统性偏低。
> 测性能时先问「这个秒表是从哪个事件开始按的」，是比任何调优都优先的基本功。

---

## 机制拆解：从单进程到 actor-learner 分离

### 1. rollout 与 train 为什么「硬件脾气不同」

| 阶段 | 主要操作 | 批大小 | 显存模式 | 优化目标 |
|------|---------|--------|----------|----------|
| **rollout** | 自回归生成、forward-only | 大 batch（成千上万条 prompt） | 只需存模型 + KV cache | **吞吐（tokens/s）** |
| **train** | forward + backward + optimizer | 小 batch（受显存限制） | 要存参数、梯度、优化器状态、activations | **显存效率 + 收敛速度** |

把两者硬塞在同一进程、同一张卡上，会出现三种浪费：

1. **批大小不匹配**：rollout 想要大 batch，train 受显存只能小 batch；
2. **内存布局不匹配**：rollout 可以 inference-optimized（FP16/BF16、量化、KV cache），train 需要 master weight + optimizer state；
3. **调度目标冲突**：同步系统里两相有数据依赖，异步系统虽可跨 step 重叠，却会引入旧策略数据。

L2 先把角色拆开，但仍保持 lockstep，方便把设备选择、权重流动和 off-policy 边界看清楚。

### 2. batch inference：把单条生成改成批量生成

L1 里 `generate_rollout` 一次只生成一条序列；L2 改成 `generate_rollout_batch`，每轮把所有 prompt 拼成一个大 tensor，一次 forward 出所有位置的下一个 token：

```python
# 以下为简化展示：源码中同一循环里还会收集 log_probs / values
# （PPO 训练需要旧策略概率与价值估计），见 L2_actor_learner_split.py
def generate_rollout_batch(model, prompt_ids, n_rollouts, max_len, device):
    model.eval()
    buffers = [RolloutBuffer() for _ in range(n_rollouts)]
    seqs = [list(prompt_ids) for _ in range(n_rollouts)]

    with torch.no_grad():
        for step in range(max_len):
            input_ids = torch.tensor(seqs, dtype=torch.long, device=device)
            logits, values = model(input_ids)          # 一次 forward，n_rollouts 条序列
            last_logits = logits[:, -1, :]
            actions = torch.distributions.Categorical(logits=last_logits).sample()
            for i in range(n_rollouts):
                buffers[i].states.append(list(seqs[i]))
                buffers[i].actions.append(actions[i].item())
                seqs[i].append(actions[i].item())
```

这只是最基础的**静态 batch forward**：用 batching 摊薄 kernel launch 和权重读取开销。
vLLM/SGLang 还包含 continuous batching、paged KV cache、调度等机制，不能与这段 toy 代码画等号；
这些内容放在 03 轨 `nano-vllm-sglang`。

还有一个从 L1 继承的 PPO 正确性细节：不同 step 的 state 长度不同，batch 化时要**右 padding 并保存真实
`length`**；训练重算 log-prob 时按 `length - 1` 取每行最后一个真实 token。否则未屏蔽的左 PAD 会改变
普通单向 LSTM 的 hidden state，使 rollout 与 train 实际比较的不是同一个 $s_t$。完整推导见
[L1 的 padding 小节](tutorial_L1.md#批处理-state为什么要右-padding--真实长度)。

### 3. 权重同步：跨进程传 state_dict

actor 和 learner 是两个独立进程，要通过 Queue 传递一个**不会继续随 learner 更新而变化的权重快照**。
本教程选择 clone 后放到 CPU，这是对 `spawn`、CPU/MPS/CUDA 都直观且可移植的教学格式：

```python
# learner 进程：把新权重发给 actor
weight_queue.put({
    k: v.detach().cpu().clone()
    for k, v in model.state_dict().items()
})
```

actor 进程收到后，`load_state_dict` 会自动把权重放到 actor 自己的 device 上：

```python
# actor 进程：接收并加载最新权重
state_dict = weight_queue.get()
model.load_state_dict(state_dict)
```

CUDA tensor 并非原则上不能跨进程共享：PyTorch 在 `spawn`/`forkserver` 下支持 CUDA IPC，但发送进程必须在
接收方持有 tensor 期间存活，还要处理设备、生命周期与异常退出；MPS/平台支持又不同。生产系统通常也不会用
Python Queue 搬完整 state dict，而会用 NCCL broadcast/all-gather 或推理引擎的 `update_weights` 路径。

> **关键认识**：权重同步是 actor-learner 架构的「脐带」。同步间隔越长，actor 的 policy version 越旧；
> 同步越频繁，通信与重建推理权重的成本越高。

### 4. 为什么默认 actor=CPU、learner=accelerator？

本脚本默认把 actor 放在 CPU，learner 放在本机最快的 accelerator（CUDA/MPS）。这不是唯一正确配置，
而是用最少依赖展示「角色可以绑定不同设备」：

- CPU actor 负责生成 rollout；
- MPS/GPU learner 负责 PPO update；
- 两个进程在系统层面独立，但当前协议仍逐轮握手：actor 生成完才让 learner 训练，learner 同步完才开始下一轮。

因此当前时间线仍是：

```text
step k:     actor [ rollout ] ──> learner [ train + sync ]
step k+1:                                      actor [ rollout ] ──> ...
```

只有允许 actor 用稍旧权重继续生成，才可能形成：

```text
actor:    [ rollout k+1 ][ rollout k+2 ]
learner:          [ train k   ][ train k+1 ]
```

第二种会把总时长从近似 `T_roll + T_train` 压向 `max(T_roll, T_train)`，同时引入 policy staleness；
完整的异步 buffer 教学放在 [nano-slime L0](../nano-slime/tutorial_L0.md)。

如果想自己实验，可以改环境变量：

```bash
# 两者都放 CPU：会因进程竞争而变慢（见下方反例）
NANO_VERL_ACTOR_DEVICE=cpu NANO_VERL_LEARNER_DEVICE=cpu python3 L2_actor_learner_split.py

# 两者都放 MPS：同样会争 GPU，通常也看不到加速
NANO_VERL_ACTOR_DEVICE=mps NANO_VERL_LEARNER_DEVICE=mps python3 L2_actor_learner_split.py
```

### 5. 模型比 L1 更大，让 rollout 成本更容易观察

本节模型从 L1 的约 28K 参数增加到约 237K 参数（embed=64, hidden=128, n_layers=2），并把每轮 rollout 数从 32 放大到 256，目的是让 **采样时间足够显著**，便于观察 device/runtime 选择的差异。在真实 LLM 后训练中，rollout 经常占据大量时间，但是否是瓶颈取决于生成长度、采样数、训练 epoch、并行配置与推理引擎，必须实测。

---

## 与 verl 的对应关系

L2 已经触及分布式 RL 训练的最小工程形态。verl / HybridFlow 在此基础上做了大量工程化：

| 本节概念 | verl 中的对应物 | 说明 |
|---------|----------------|------|
| `actor_loop` + `generate_rollout_batch` | rollout worker + vLLM/SGLang 等后端 | 本节只有普通 PyTorch 静态 batch |
| `learner_loop` + `ppo_update` | actor training worker / trainer | 参数更新；真实系统还会使用 FSDP、Megatron 等并行 |
| `weight_queue` | parameter synchronizer / `update_weights` | toy 用 CPU snapshot；真实系统常走 NCCL 等设备通信 |
| `traj_queue` | trajectory / experience message queue | 配置容量虽为 2，但握手使同时在途 rollout 至多一轮；异步系统还需背压与 staleness 策略 |
| actor=CPU, learner=GPU | 分离资源池的一种 toy 映射 | L3 展开同卡同步 colocate；nano-slime 展开异步分离 |

具体源码路径和 API 会在 L3 对照 verl 实现时给出。L2 先确保「拆分后训练仍然正确、加速出现的位置可解释」。

---

## 费曼自检

### 讲给外行听：双灶厨房

想象一个小餐馆只有一位厨师（learner）和一个传菜窗口（actor）：

- **serial 版**：厨师先跑到窗口接单、切菜、炒菜，全部自己干。顾客少时还行，高峰期窗口排队，灶台却空着。
- **L2 actor-learner 版**：窗口专人负责接单、配菜（actor），厨师只负责炒菜（learner）；但架子只容一盘，窗口放下一盘后必须等厨师炒完并发回新菜谱，下一盘才开始。这是角色拆分，不是流水线重叠。
- **权重同步**：就是厨师每换一道新菜谱（新模型权重），都要喊一声让窗口同事按新配方配菜。如果不通知，窗口还在按旧菜单配，客人吃到的东西就和新菜谱对不上（off-policy）。
- **batch inference**：窗口不是接到一单就回厨房拿一次菜，而是攒几单一起拿，减少来回跑的次数。

actor-learner 分离能不能快，既取决于设备是否合适，也取决于是否允许多盘在途。L2 只验证前者；允许多盘在途会进一步引出旧菜谱（staleness）问题。

### 思考题

1. 如果 actor 和 learner 都跑在同一张 GPU 上，为什么本节通常观察不到加速？真实 verl 是怎么解决这个问题的？
2. 为什么 `weight_queue.put(model.state_dict())` 之前必须把 tensor 移到 CPU？在 CUDA 上是否也要这样做？
3. 当前实现里 actor 收到一次权重、生成一轮 rollout 就等下一次权重。如果让 actor 连续生成多轮 rollout 再用最新权重更新，会引入什么问题？

简答：

1. 当前没有跨步 overlap；同卡双进程还会争计算流和显存。verl 的同步 HybridFlow 可让 actor/rollout 角色 colocate，在阶段边界切换模式、同步权重并复用整组 GPU；fully-async 模式则给 trainer 与 rollouter 分配独立资源，以 staleness 换 overlap。
2. **不是 CUDA 上也必须如此**。CPU clone 是本教程最稳妥的快照/传输格式；CUDA IPC 有条件可用，生产方案通常直接使用 NCCL 等设备通信。
3. 吞吐可能提高，但后几轮来自更旧的行为策略。ratio 方差、clip 比例与 KL 往往上升，有效样本率下降；需要携带 policy version / `old_log_prob`，限制最大 staleness，并定义丢弃、降权或 partial rollout 续跑策略。

### 反例

> "actor-learner 分离一定能让训练更快。"

不一定。本次 CPU-only 运行是 8.43s→8.18s（1.03x，基本持平）；不同机器上也可能略慢。
这说明进程拆分本身还会增加调度与通信开销；它不是 overlap 的充分条件。

> "actor 用旧权重采样没关系，反正 learner 会更新。"

关系很大。actor 用旧权重采样出的 `log_prob` 是 "old policy" 的概率；learner 用新权重算出的 `log_prob` 是 "new policy" 的概率。两者差距太大时，PPO 的 importance sampling ratio 会剧烈波动，训练不稳定。真实系统会通过限制 KL、控制权重同步频率、或使用 off-policy 修正算法来缓解。

---

## 下一步

按下面的顺序继续即可：

- [L3](tutorial_L3.md)：同步 HybridFlow colocate，同一组 worker 在 rollout/train 间切换；
- [nano-slime L0](../nano-slime/tutorial_L0.md)：异步 actor/learner、buffer、背压与 staleness；
- [PPO/IS 深挖 §2](../sota-deepdive/post-training-algorithm-evolution.md)：importance sampling、clip 与 ratio 粒度。

---

## 溯源与声明

- verl 仓库：`https://github.com/verl-project/verl`
- verl HybridFlow 编程指南：`https://verl.readthedocs.io/en/latest/hybrid_flow.html`
- verl fully-async 指南：`https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md`
- PyTorch multiprocessing / CUDA tensor 共享说明：`https://docs.pytorch.org/docs/stable/notes/multiprocessing.html`
- PPO 原始论文：`arXiv:1707.06347`
- GAE 原始论文：`arXiv:1506.02438`
- 当前 verl 的 actor/rollout worker、参数更新与异步模式以官方文档为准；L3 另有固定版本源码锚点。
- 所列计时来自当前 CPU-only 运行；同 seed 保证训练随机性可控，不保证跨机器 wall time 相同。计时窗口两边对齐（均不含 warmup），而 1.03x 应视作基本持平，不作加速结论。
- 模型参数量由 `sum(p.numel() for p in model.parameters())` 现场统计：237,150。
- CPU-only 输出来自 `NANO_VERL_ACTOR_DEVICE=cpu NANO_VERL_LEARNER_DEVICE=cpu` 的真实运行。
