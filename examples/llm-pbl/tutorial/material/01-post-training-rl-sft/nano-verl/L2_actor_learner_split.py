#!/usr/bin/env python3
"""
L2_actor_learner_split.py — nano-verl L2

把 L1 的单进程 PPO 拆成 actor-learner 两个进程：
- actor 进程：持有 inference-optimized 模型，用 batch 推理高吞吐生成 rollout。
- learner 进程：接收 trajectory，执行 PPO update，把新权重同步回 actor。

核心要体会的工程点：
1. rollout（推理）和 train（训练）对计算/显存/批大小的诉求不同，值得拆成独立角色。
2. actor 必须用旧权重采样，learner 用新权重训练，二者之间需要显式权重同步。
3. 静态 batch forward 能摊薄部分推理开销；真实 vLLM/SGLang 还包含 continuous batching、
   paged KV cache 和调度，本文件不模拟这些引擎机制。
4. multiprocessing + Queue + state_dict 传递是分布式 RL 训练系统的最小原型。

边界：本文件采用「rollout -> train -> sync」的 lockstep 握手，只演示进程/设备隔离，
不声称 rollout 与 train 已经跨步重叠。同步 colocate 见 L3；真正的异步 buffer、
staleness 与跨步流水见同轨 nano-slime L0。

依赖：torch（CPU/MPS/GPU 均可；无 transformers）。
L3 对照同步 colocate；真实推理引擎内部机制在 03 轨展开。
"""

import math
import multiprocessing as mp
import os
import random
import time


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise SystemExit(
        "[error] torch is required to run this L2 script.\n"
        "        Install it with: pip install torch\n"
        "        (CPU/MPS/GPU all work; no transformers needed.)"
    ) from e


# ---------------------------------------------------------------------------
# 配置：比 L1 稍大，让 rollout 成本更容易观察
# ---------------------------------------------------------------------------
SEED = 42
PROMPT = "go:"
TARGET = "hello"
MAX_RESP_LEN = len(TARGET)

VOCAB = ["<pad>", "<sos>"] + [chr(ord("a") + i) for i in range(26)] + [":"]
PAD_ID = 0
SOS_ID = 1
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}

EMBED_DIM = 64
HIDDEN_DIM = 128
N_LAYERS = 2

N_ROLLOUTS = 256        # 每轮采样数：放大吞吐优势
N_EPOCHS = 4
N_ITERATIONS = 40       # L2 重点看时间，迭代数比 L1 少
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
KL_TARGET = 0.02
LR = 1e-3
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5
WARMUP_STEPS = 100

WEIGHT_QUEUE_SIZE = 2
TRAJ_QUEUE_SIZE = 2


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def encode(text):
    return [CHAR2ID[c] for c in text]


def decode(token_ids):
    return "".join(VOCAB[i] for i in token_ids)


def reward_fn(generated):
    """逐字符匹配奖励，每条 rollout 总奖励在 [0, 1] 之间。"""
    rewards = []
    for i in range(MAX_RESP_LEN):
        c = generated[i] if i < len(generated) else "<pad>"
        rewards.append(1.0 / MAX_RESP_LEN if c == TARGET[i] else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# 模型：和 L1 结构相同，但更深更宽，使 rollout 更耗时
# ---------------------------------------------------------------------------
class TinyLSTM(nn.Module):
    """策略 + 价值共享 backbone 的极小 LSTM。"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        out, _ = self.lstm(x)
        logits = self.policy_head(out)
        values = self.value_head(out).squeeze(-1)
        return logits, values


class RolloutBuffer:
    """单条 rollout 的缓存。"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []


# ---------------------------------------------------------------------------
# 核心函数：GAE / collate / PPO update / warmup（与 L1 逻辑一致）
# ---------------------------------------------------------------------------
def compute_gae(buffers, gamma, lam):
    all_advantages = []
    all_returns = []
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
        all_advantages.extend(advantages)
        all_returns.extend(returns)
    return torch.tensor(all_advantages, dtype=torch.float32), torch.tensor(all_returns, dtype=torch.float32)


def collate(buffers, device):
    states, actions, old_log_probs = [], [], []
    for buf in buffers:
        states.extend(buf.states)
        actions.extend(buf.actions)
        old_log_probs.extend(buf.log_probs)
    lengths = [len(s) for s in states]
    max_len = max(lengths)
    # 与 L1 一致：右 padding，并在 update 时取最后一个真实 token。
    # 左 padding 会先改变普通单向 LSTM 的 hidden state，污染 PPO ratio。
    padded = [s + [PAD_ID] * (max_len - len(s)) for s in states]
    state_t = torch.tensor(padded, dtype=torch.long, device=device)
    length_t = torch.tensor(lengths, dtype=torch.long, device=device)
    action_t = torch.tensor(actions, dtype=torch.long, device=device)
    old_log_prob_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    return state_t, length_t, action_t, old_log_prob_t


def ppo_update(model, optimizer, buffers, device, n_epochs, clip_eps, entropy_coef, value_coef, kl_target):
    advantages, returns = compute_gae(buffers, GAMMA, GAE_LAMBDA)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.to(device)
    returns = returns.to(device)

    state_t, length_t, action_t, old_log_prob_t = collate(buffers, device)

    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0
    approx_kl = 0.0

    for _ in range(n_epochs):
        logits, values = model(state_t)
        batch_idx = torch.arange(state_t.size(0), device=device)
        last_idx = length_t - 1
        last_logits = logits[batch_idx, last_idx, :]
        last_values = values[batch_idx, last_idx]
        dist = torch.distributions.Categorical(logits=last_logits)
        new_log_prob = dist.log_prob(action_t)
        ratio = torch.exp(new_log_prob - old_log_prob_t)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(last_values, returns)
        entropy = dist.entropy().mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        n_updates += 1

        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        if approx_kl > kl_target * 1.5:
            break

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": approx_kl,
    }


def supervised_warmup(model, optimizer, prompt_ids, device, steps):
    """teacher forcing SFT warmup，返回平均 loss。"""
    model.train()
    full_seq = torch.tensor([prompt_ids + encode(TARGET)], dtype=torch.long, device=device)
    input_ids = full_seq[:, :-1]
    target_ids = full_seq[:, 1:]
    total_loss = 0.0
    for _ in range(steps):
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / steps


def cpu_state_dict_snapshot(model):
    """为 Queue 发布一个与后续 optimizer 更新解耦的 CPU 权重快照。"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Batch rollout：只演示最基础的静态 batching（不模拟 continuous batching / KV 调度）
# ---------------------------------------------------------------------------
def generate_rollout_batch(model, prompt_ids, n_rollouts, max_len, device):
    """
    用当前 policy 自回归生成 n_rollouts 条 response。
    每个 step 一次性对所有序列做 forward（batch inference），摊薄 kernel launch 等开销。
    """
    model.eval()
    buffers = [RolloutBuffer() for _ in range(n_rollouts)]
    seqs = [list(prompt_ids) for _ in range(n_rollouts)]

    with torch.no_grad():
        for step in range(max_len):
            input_ids = torch.tensor(seqs, dtype=torch.long, device=device)
            logits, values = model(input_ids)
            last_logits = logits[:, -1, :]
            last_value = values[:, -1]
            dist = torch.distributions.Categorical(logits=last_logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            for i in range(n_rollouts):
                buffers[i].states.append(list(seqs[i]))
                buffers[i].actions.append(actions[i].item())
                buffers[i].log_probs.append(log_probs[i].item())
                buffers[i].values.append(last_value[i].item())
                buffers[i].rewards.append(None)
                seqs[i].append(actions[i].item())

    rewards = []
    for i in range(n_rollouts):
        resp = decode(seqs[i][len(prompt_ids):])
        step_rewards = reward_fn(resp)
        for step, r in enumerate(step_rewards):
            buffers[i].rewards[step] = r
        rewards.append(sum(step_rewards))

    return buffers, rewards


# ---------------------------------------------------------------------------
# 单进程串行版：用于与多进程版做 wall-time 对比
# ---------------------------------------------------------------------------
def run_serial(device_str):
    device = torch.device(device_str)
    set_seed(SEED)

    model = TinyLSTM(len(VOCAB), EMBED_DIM, HIDDEN_DIM, N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    prompt_ids = encode(PROMPT)

    supervised_warmup(model, optimizer, prompt_ids, device, WARMUP_STEPS)

    best_reward = 0.0
    results = []
    t0 = time.time()
    for iteration in range(N_ITERATIONS):
        buffers, rewards = generate_rollout_batch(model, prompt_ids, N_ROLLOUTS, MAX_RESP_LEN, device)
        mean_reward = sum(rewards) / len(rewards)
        best_reward = max(best_reward, max(rewards))
        stats = ppo_update(
            model, optimizer, buffers, device,
            n_epochs=N_EPOCHS,
            clip_eps=CLIP_EPS,
            entropy_coef=ENTROPY_COEF,
            value_coef=VALUE_COEF,
            kl_target=KL_TARGET,
        )
        results.append((mean_reward, best_reward, stats))
    elapsed = time.time() - t0
    return elapsed, results


# ---------------------------------------------------------------------------
# 多进程并行版：actor 与 learner 分离
# ---------------------------------------------------------------------------
def actor_loop(device_str, prompt_ids, weight_queue, traj_queue, n_iterations, stop_event):
    """
    actor 进程主循环。
    - 从 weight_queue 接收最新权重；
    - 用 batch inference 生成一批 rollout；
    - 把 (buffers, rewards) 放入 traj_queue；
    - 重复 n_iterations 次后退出。
    """
    device = torch.device(device_str)
    set_seed(SEED)
    model = TinyLSTM(len(VOCAB), EMBED_DIM, HIDDEN_DIM, N_LAYERS).to(device)
    model.eval()

    for it in range(n_iterations):
        # 等 learner 的最新权重；首次权重由 learner warmup 后发送
        state_dict = weight_queue.get()
        if state_dict is None:
            break
        model.load_state_dict(state_dict)

        buffers, rewards = generate_rollout_batch(
            model, prompt_ids, N_ROLLOUTS, MAX_RESP_LEN, device
        )
        traj_queue.put((buffers, rewards))

    stop_event.set()


def learner_loop(device_str, prompt_ids, weight_queue, traj_queue, result_queue, n_iterations,
                 ready_event):
    """
    learner 进程主循环。
    - 做 SFT warmup；
    - 把初始权重发给 actor；
    - 从 traj_queue 收 trajectory，做 PPO update；
    - 把新权重发回 weight_queue，并把统计量发到 result_queue。
    """
    device = torch.device(device_str)
    set_seed(SEED)

    model = TinyLSTM(len(VOCAB), EMBED_DIM, HIDDEN_DIM, N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    supervised_warmup(model, optimizer, prompt_ids, device, WARMUP_STEPS)
    # 发送初始权重，actor 拿到后才能开始第一次 rollout
    # 本 toy 用 CPU snapshot 作为跨设备、跨进程的通用交换格式。
    # CUDA IPC 并非做不到，但生命周期/设备同步更复杂；真实系统通常使用 NCCL。
    weight_queue.put(cpu_state_dict_snapshot(model))
    # 计时口径对齐：warmup 结束 + 初始权重已发出，才通知主进程开始计时
    # （与 run_serial「warmup 之后取 t0」保持同一口径）
    ready_event.set()

    best_reward = 0.0
    for iteration in range(n_iterations):
        buffers, rewards = traj_queue.get()
        mean_reward = sum(rewards) / len(rewards)
        best_reward = max(best_reward, max(rewards))
        stats = ppo_update(
            model, optimizer, buffers, device,
            n_epochs=N_EPOCHS,
            clip_eps=CLIP_EPS,
            entropy_coef=ENTROPY_COEF,
            value_coef=VALUE_COEF,
            kl_target=KL_TARGET,
        )
        # 新权重先发给 actor（让 actor 下一轮用新权重），再汇报结果
        weight_queue.put(cpu_state_dict_snapshot(model))
        result_queue.put((iteration, mean_reward, best_reward, stats))

    result_queue.put("DONE")


def run_parallel(actor_device_str, learner_device_str):
    """启动 actor + learner 两个 lockstep 进程，并汇总结果。

    两个角色可以跑在不同 device 上（例如 actor 在 CPU 做推理，learner 在 MPS/GPU
    做训练），但当前 Queue 依赖仍是 rollout -> train -> sync，没有跨步计算重叠。
    这个对照主要展示角色隔离、设备适配和 IPC 成本，而不是流水线加速。
    """
    mp.set_start_method("spawn", force=True)

    weight_queue = mp.Queue(WEIGHT_QUEUE_SIZE)
    traj_queue = mp.Queue(TRAJ_QUEUE_SIZE)
    result_queue = mp.Queue()
    stop_event = mp.Event()
    ready_event = mp.Event()   # learner warmup 完成后置位，主进程据此对齐计时窗口

    prompt_ids = encode(PROMPT)

    actor_proc = mp.Process(
        target=actor_loop,
        args=(actor_device_str, prompt_ids, weight_queue, traj_queue, N_ITERATIONS, stop_event),
    )
    learner_proc = mp.Process(
        target=learner_loop,
        args=(learner_device_str, prompt_ids, weight_queue, traj_queue, result_queue,
              N_ITERATIONS, ready_event),
    )

    actor_proc.start()
    learner_proc.start()

    # 计时口径：与 run_serial 对齐——两边都从「warmup 之后」开始计。
    # spawn / import torch / 模型初始化 / warmup 均不在窗口内；
    # 窗口内包含第一条 rollout 的等待与生成，这是并行版无法回避的真实开销。
    ready_event.wait()

    results = []
    t0 = time.time()
    while len(results) < N_ITERATIONS:
        msg = result_queue.get()
        if msg == "DONE":
            break
        results.append(msg)
    elapsed = time.time() - t0

    # 发送结束信号，清理子进程
    try:
        weight_queue.put(None, timeout=1.0)
    except Exception:
        pass
    actor_proc.join(timeout=5.0)
    learner_proc.join(timeout=5.0)
    if actor_proc.is_alive():
        actor_proc.terminate()
    if learner_proc.is_alive():
        learner_proc.terminate()

    return elapsed, results


# ---------------------------------------------------------------------------
# main：跑单进程 + 多进程，对比时间与收敛性
# ---------------------------------------------------------------------------
def main():
    learner_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    # actor 默认放 CPU，learner 放 accelerator；可通过环境变量覆盖
    actor_device_str = os.environ.get("NANO_VERL_ACTOR_DEVICE", "cpu")
    learner_device_str = os.environ.get("NANO_VERL_LEARNER_DEVICE", str(learner_device))

    n_params = sum(p.numel() for p in TinyLSTM(len(VOCAB), EMBED_DIM, HIDDEN_DIM, N_LAYERS).parameters())

    print("nano-verl L2 — actor/learner split with batch inference")
    print(f"learner device: {learner_device_str}, actor device: {actor_device_str}, model params: {n_params:,}")
    print(f"prompt='{PROMPT}' target='{TARGET}' max_resp_len={MAX_RESP_LEN}")
    print(f"config: rollouts={N_ROLLOUTS}, outer_iter={N_ITERATIONS}, ppo_epochs={N_EPOCHS}, lr={LR}")
    print(f"        warmup_steps={WARMUP_STEPS}")
    print(f"hint: set NANO_VERL_ACTOR_DEVICE / NANO_VERL_LEARNER_DEVICE to experiment\n")

    print("[1/2] serial baseline (generate + train in the same process)")
    t_serial, res_serial = run_serial(learner_device_str)
    print(f"      elapsed: {t_serial:.2f}s")
    print(f"      final reward: {res_serial[-1][0]:.3f} (best={res_serial[-1][1]:.3f})\n")

    print("[2/2] actor-learner split (lockstep processes + weight sync)")
    t_parallel, res_parallel = run_parallel(actor_device_str, learner_device_str)
    print(f"      elapsed: {t_parallel:.2f}s")
    print(f"      final reward: {res_parallel[-1][1]:.3f} (best={res_parallel[-1][2]:.3f})\n")

    print("[comparison]")
    print(f"  serial   : {t_serial:.2f}s")
    print(f"  split    : {t_parallel:.2f}s")
    speedup = t_serial / t_parallel if t_parallel > 0 else float("inf")
    print(f"  speedup  : {speedup:.2f}x")
    print(f"  reward parity: serial={res_serial[-1][0]:.3f}, split={res_parallel[-1][1]:.3f}")

    print("\ntakeaway: L2 完成角色/设备隔离与显式权重同步，但仍是 lockstep；")
    print("          batch inference 摊薄推理开销，CPU/GPU 差异会显著影响这个 toy 的计时。")
    print("          L3 讲同步 colocate；真正异步 overlap/staleness 见 nano-slime L0。")


if __name__ == "__main__":
    main()
