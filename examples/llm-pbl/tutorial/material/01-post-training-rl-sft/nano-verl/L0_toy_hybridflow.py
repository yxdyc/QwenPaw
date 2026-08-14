#!/usr/bin/env python3
"""
L0_toy_hybridflow.py — nano-verl L0

用纯标准库模拟 RL 后训练中的两种调度方式：
1. Naive：同一进程/资源里 rollout -> train -> sync 顺序执行。
2. HybridFlow：actor（采样）与 learner（训练）分离，并通过流水线重叠二者。

核心要体会的点：
- rollout 是推理行为（前向、生成 token），train 是优化行为（反向、更新权重）。
- 二者对计算/显存/批大小的诉求不同，强行串行会浪费资源。
- actor-learner 分离后，用「权重同步」把 learner 的新权重交给 actor。

这是一个玩具：没有真实 GPU/模型，所有耗时都是确定性模拟。
对应真实系统：verl (HybridFlow) https://github.com/verl-project/verl
"""

import math
import random


SEED = 42
N_ACTIONS = 5
BEST_ACTION = 2
BATCH_SIZE = 32
N_ITERS = 8

# 模拟每轮各阶段的耗时（ms）。train 比 rollout 重，是常见情况。
C_ROLL_MS = 3.0      # actor 生成一个 batch 的轨迹
C_TRAIN_MS = 5.0     # learner 做一次 policy-gradient 更新
C_SYNC_MS = 0.5      # 权重从 learner 同步到 actor


def softmax(logits):
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    s = sum(e)
    return [v / s for v in e]


class ToyBanditPolicy:
    """一个 5 臂 bandit 的 softmax 策略，用 REINFORCE 更新。"""
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

    def update(self, batch):
        """batch: list of (action, reward)。REINFORCE with mean baseline。"""
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


def collect_batch(policy, size):
    """用当前策略采样一个 batch，返回 [(action, reward), ...]。"""
    batch = []
    for _ in range(size):
        a = policy.sample()
        r = 1.0 if a == policy.best else 0.1
        batch.append((a, r))
    return batch


def print_run(name, clock, policy):
    p = policy.probs()
    print(f"[{name}] total simulated time: {clock:.1f} ms")
    print(f"         best-action prob: {p[policy.best]:.3f}")
    print(f"         final weights:    {[round(x, 2) for x in policy.w]}")


def run_naive():
    """串行：rollout -> train -> sync，每一步独占时间线。"""
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


def run_hybrid():
    """
    HybridFlow 式流水线：
    - startup：actor 先 rollout 第 0 个 batch。
    - steady：actor rollout batch i 与 learner train batch i-1 重叠；完成后 sync。
    - drain：train 最后一个 batch 并 sync。
    """
    random.seed(SEED)
    policy = ToyBanditPolicy()
    clock = 0.0
    batches = []

    # startup
    batches.append(collect_batch(policy, BATCH_SIZE))
    clock += C_ROLL_MS

    # steady state
    for it in range(1, N_ITERS):
        prev = batches[-1]
        start = clock
        batches.append(collect_batch(policy, BATCH_SIZE))   # actor
        policy.update(prev)                                  # learner
        end_roll = start + C_ROLL_MS
        end_train = start + C_TRAIN_MS
        clock = max(end_roll, end_train) + C_SYNC_MS

    # drain
    policy.update(batches[-1])
    clock += C_TRAIN_MS + C_SYNC_MS

    print_run("Hybrid ", clock, policy)
    return clock


def main():
    print("nano-verl L0 — toy actor/learner scheduling")
    print(f"config: actions={N_ACTIONS}, best={BEST_ACTION}, batch={BATCH_SIZE}, iters={N_ITERS}")
    print(f"        roll={C_ROLL_MS}ms, train={C_TRAIN_MS}ms, sync={C_SYNC_MS}ms\n")

    t_naive = run_naive()
    t_hybrid = run_hybrid()
    print(f"\nspeedup(Hybrid/Naive) = {t_naive / t_hybrid:.2f}x")
    print("\n takeaway: 把 rollout 和 train 放到不同资源并流水线化，")
    print("           可以把每轮耗时从 (roll+train+sync) 降到 max(roll,train)+sync。")


if __name__ == "__main__":
    main()
