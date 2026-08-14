#!/usr/bin/env python3
"""
L1_minimal_ppo.py — nano-verl L1

真实小模型（字符级 LSTM，约 28K 参数）上的最小 PPO 循环：
    SFT warmup → generate rollout → GAE advantage → clipped policy update → value update。

这是 L0 玩具调度器的下一级：L0 用 bandit 讲清了 actor/learner 为什么要分离；
L1 用一个可训练的语言模型把「generate → advantage → update」的完整链路跑通。

为了让 PPO 在极小模型上稳定学会目标序列，先做一次极短的 SFT warmup
（teacher-force 目标序列）。这符合真实后训练流程：RL 几乎总是从一个 SFT model 开始。

依赖：torch（CPU 或 MPS 均可；无 transformers）。
对应真实系统：verl / HybridFlow 中的 RL core（L3 再对照源码）。
"""

import math
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise SystemExit(
        "[error] torch is required to run this L1 script.\n"
        "        Install it with: pip install torch\n"
        "        (CPU/MPS/GPU all work; no transformers needed.)"
    ) from e


SEED = 42
PROMPT = "go:"          # 固定 prompt，让模型学会续写
TARGET = "hello"        # 目标续写内容
MAX_RESP_LEN = len(TARGET)

VOCAB = ["<pad>", "<sos>"] + [chr(ord("a") + i) for i in range(26)] + [":"]
PAD_ID = 0
SOS_ID = 1
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}

EMBED_DIM = 32
HIDDEN_DIM = 64
N_LAYERS = 1

N_ROLLOUTS = 32         # 每次 update 前采多少条 rollout
N_EPOCHS = 4            # 每条数据复用多少次（PPO 的 epoch）
N_ITERATIONS = 80       # 外循环轮数
GAMMA = 0.99            # 折扣因子
GAE_LAMBDA = 0.95       # GAE 参数
CLIP_EPS = 0.2          # PPO clip
KL_TARGET = 0.02        # 早停 KL 阈值
LR = 1e-3
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5

WARMUP_STEPS = 100      # SFT warmup 步数，让 PPO 有一个合理的初始策略


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


class TinyLSTM(nn.Module):
    """策略 + 价值共享 backbone 的极小 LSTM。"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids):
        # token_ids: (B, T)
        x = self.embedding(token_ids)
        out, _ = self.lstm(x)
        logits = self.policy_head(out)             # (B, T, V)
        values = self.value_head(out).squeeze(-1)  # (B, T)
        return logits, values

    def last_dist_value(self, token_ids, lengths=None):
        """取每条序列最后一个真实 token 处的策略分布与价值估计。"""
        logits, values = self.forward(token_ids)
        if lengths is None:
            # rollout 时没有 padding，最后一列就是真实序列末尾。
            last_logits = logits[:, -1, :]  # (B, V)
            last_value = values[:, -1]      # (B,)
        else:
            # PPO update 使用右 padding；按真实长度取末尾，忽略右侧 PAD。
            batch_idx = torch.arange(token_ids.size(0), device=token_ids.device)
            last_idx = lengths.to(device=token_ids.device, dtype=torch.long) - 1
            last_logits = logits[batch_idx, last_idx, :]  # (B, V)
            last_value = values[batch_idx, last_idx]      # (B,)
        dist = torch.distributions.Categorical(logits=last_logits)
        return dist, last_value


class RolloutBuffer:
    """收集一条 rollout 的 (state, action, old_log_prob, reward, value)。"""
    def __init__(self):
        self.states = []      # list of token-id sequences (prompt + prefix)
        self.actions = []     # 实际采样的 token id
        self.log_probs = []   # 采样时的 log prob
        self.rewards = []     # token-level reward
        self.values = []      # value 估计


def generate_rollout(model, prompt_ids, max_len, device):
    """用当前 policy 自回归生成一条 response，同时记录旧策略的 log_prob 和 value。"""
    model.eval()
    buffer = RolloutBuffer()
    seq = list(prompt_ids)

    with torch.no_grad():
        for step in range(max_len):
            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            dist, value = model.last_dist_value(input_ids)
            action_t = dist.sample()  # (1,)
            action = action_t.item()
            log_prob = dist.log_prob(action_t).item()

            buffer.states.append(list(seq))
            buffer.actions.append(action)
            buffer.log_probs.append(log_prob)
            buffer.values.append(value.item())
            buffer.rewards.append(None)  # 占位，生成完整序列后回填

            seq.append(action)

    # 回填逐字符奖励
    step_rewards = reward_fn(decode(seq[len(prompt_ids):]))
    for step, r in enumerate(step_rewards):
        buffer.rewards[step] = r

    final_reward = sum(step_rewards)
    return buffer, final_reward, decode(seq[len(prompt_ids):])


def compute_gae(buffers, gamma, lam):
    """对每个 rollout 用 GAE 计算 advantage 和 return。"""
    all_advantages = []
    all_returns = []

    for buf in buffers:
        rewards = buf.rewards
        values = buf.values
        advantages = []
        gae = 0.0

        # 从后往前算；最后一个 state 的 next_value 视为 0（episode 结束）
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
    """把多个 rollout 的 state 右 pad 到相同长度，并保留真实长度。"""
    states = []
    actions = []
    old_log_probs = []

    for buf in buffers:
        states.extend(buf.states)
        actions.extend(buf.actions)
        old_log_probs.extend(buf.log_probs)

    lengths = [len(s) for s in states]
    max_len = max(lengths)
    # 必须右 padding：单向 LSTM 在最后一个真实 token 处的输出不会受后续 PAD 影响。
    # 若左 padding，rollout 时不存在的 PAD 会先改变 LSTM hidden state，污染 PPO ratio。
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

    for _ in range(n_epochs):
        # length_t 保证这里和 rollout 都在同一个真实 state 末尾计算概率/value。
        dist, last_values = model.last_dist_value(state_t, length_t)
        new_log_prob = dist.log_prob(action_t)
        # IS ratio：把 rollout 时 old policy 采到的 action，换算到当前 policy 的相对权重。
        # 第一次 optimizer.step() 前应约为 1；后续 PPO epoch 才会随参数更新偏离 1。
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

        # 在 old-policy 样本上的非负 KL 估计：KL(old||new)
        # = E_old[(ratio - 1) - log(ratio)]（有限样本下为近似）。
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
    """
    用 teacher forcing 对目标序列做极短 SFT，给 PPO 一个合理初始策略。
    输入：prompt + target 的前 T-1 个 token；目标：target 的后 T-1 个 token。
    返回平均 cross-entropy loss（最后一步 loss 在极小数据上不稳定，平均更合理）。
    """
    model.train()
    full_seq = torch.tensor([prompt_ids + encode(TARGET)], dtype=torch.long, device=device)
    input_ids = full_seq[:, :-1]
    target_ids = full_seq[:, 1:]

    total_loss = 0.0
    for _ in range(steps):
        logits, _ = model.forward(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / steps


def main():
    # 自动选择 CUDA / MPS / CPU；有 NVIDIA GPU 时优先 CUDA
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    set_seed(SEED)

    model = TinyLSTM(len(VOCAB), EMBED_DIM, HIDDEN_DIM, N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    prompt_ids = encode(PROMPT)

    print("nano-verl L1 — minimal PPO on a tiny LSTM")
    print(f"device: {device}, model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"prompt='{PROMPT}' target='{TARGET}' max_resp_len={MAX_RESP_LEN}")
    print(f"config: rollouts={N_ROLLOUTS}, outer_iter={N_ITERATIONS}, ppo_epochs={N_EPOCHS}, lr={LR}")
    print(f"        warmup_steps={WARMUP_STEPS}\n")

    # 训练前随机采样，看看 baseline
    model.eval()
    with torch.no_grad():
        print("[before warmup] random samples:")
        for _ in range(5):
            _, r, text = generate_rollout(model, prompt_ids, MAX_RESP_LEN, device)
            print(f"  '{text}'  reward={r:.2f}")
        print()

    # SFT warmup：让模型先"认识"目标序列
    warmup_loss = supervised_warmup(model, optimizer, prompt_ids, device, WARMUP_STEPS)
    print(f"[warmup] cross-entropy loss={warmup_loss:.4f}\n")

    # PPO 主循环
    best_reward = 0.0
    for iteration in range(N_ITERATIONS):
        buffers = []
        rewards = []
        for _ in range(N_ROLLOUTS):
            buf, r, _ = generate_rollout(model, prompt_ids, MAX_RESP_LEN, device)
            buffers.append(buf)
            rewards.append(r)

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

        if iteration % 10 == 0 or iteration == N_ITERATIONS - 1:
            print(f"[iter {iteration:3d}] reward={mean_reward:.3f} (best={best_reward:.3f}) "
                  f"policy_loss={stats['policy_loss']:.3f} value_loss={stats['value_loss']:.3f} "
                  f"entropy={stats['entropy']:.3f} approx_kl={stats['approx_kl']:.4f}")

    print()
    # 训练后采样
    model.eval()
    with torch.no_grad():
        print("[after PPO] samples:")
        for _ in range(8):
            buf, r, text = generate_rollout(model, prompt_ids, MAX_RESP_LEN, device)
            marker = " ✅" if text == TARGET else ""
            print(f"  '{text}'  reward={r:.2f}{marker}")

    print("\ntakeaway: 极小 LSTM 经 SFT warmup 后，PPO 能把 'go:' 稳定续写成 'hello'。")
    print("          真实后训练也遵循同样顺序：SFT model → RL；L2 再把 rollout 与 train 拆开。")


if __name__ == "__main__":
    main()
