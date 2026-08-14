# nano-verl

> **抓的核心机制**：RLHF 的 **actor-learner 分离** + rollout/train 的**资源调度**（HybridFlow 思想）。
> **对应真实系统**：[verl](https://github.com/verl-project/verl)
> **轨道**：[01 后训练/RL/SFT](../README.md) · **状态**：L0–L3 ✅

---

## 环境依赖

- L0：纯标准库，零外部依赖。
- L1/L2：需要 `torch`（CPU/MPS/GPU 均可，无 transformers）。
- L2 额外使用 `multiprocessing`（Python 标准库）模拟 actor/learner 两个进程；在 macOS/Windows 上默认用 `spawn` 启动方式。
- L3：仅 `torch`，**强制 CPU**（跨遍 bit-level 确定性是 self-check 前提）；任意 CWD 可跑，~8s。L3 为「可运行的本质模拟 + 显式注明」（真实 verl 需 Ray+FSDP+vLLM+多 GPU，`[TODO: verify on real system]`）。

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | 玩具：用伪调度模拟「采样阶段」与「训练阶段」交替，理解为什么二者要分离 | ✅ `L0_toy_hybridflow.py` + `tutorial_L0.md` |
| **L1** | 单卡跑通一个最小 PPO 循环：generate rollout → 算 advantage → 更新 policy | ✅ `L1_minimal_ppo.py` + `tutorial_L1.md` |
| **L2** | 引入 lockstep actor-learner 分离：理解 batch inference、权重快照与资源冲突（尚未跨步 overlap） | ✅ `L2_actor_learner_split.py` + `tutorial_L2.md` |
| **L3** | 对照 verl HybridFlow（v0.7.1 源码），复现「同一组 worker 在 rollout/train 间复用」：单控制器+SPMD、DataProto chunk/concat、阶段边界 resharding、显存按相轮换 | ✅ `L3_hybridflow_colocate.py` + `tutorial_L3.md` |

## 核心要讲清的点

- 为什么 rollout（推理）和 train（训练）的硬件诉求不同（吞吐 vs 显存）
- 权重同步：learner 更新后如何把新权重发给 actor
- advantage 估计：GAE 在做什么
- importance sampling：旧 `log_prob`、新 `log_prob` 与 PPO ratio 如何让同批 rollout 更新多轮，以及它不能消除哪些 staleness
- （L3）严格新鲜 rollout 的同步两相串行：拆卡 = 每相一半算力 + 空转，复用 = 每相全部算力 + resharding 税
- （L3）colocate 的硬逻辑是显存算术：训练态 4P/N vs 拆卡的 4P/(N/2)，模型越大越必须复用
- （L3）调度机制数值透明（同 DP 宽度 bit 级不变）；跨 DP 宽度成立的是收敛可比性，不是轨迹相同

## 费曼自检

- 能不能解释「为什么不能边采样边训练（同一进程同一显存）」？
- （L3）能不能用「一间厨房、两份菜单」讲清 colocate 为什么是显存算术的必然，并说出类比的边界？

## 权威实现与延伸

- 对标源码：verl `https://github.com/verl-project/verl`（HybridFlow：单控制器+SPMD、两相复用、权重 resharding）
- L3 锚点基准：**v0.7.1**（2026-08-07 抓取核验，六文件 sha256 + 行号见 `tutorial_L3.md` §11）；论文 arXiv:2409.19256（标题/摘要亲验）。注意 main 分支已重构（`fsdp_workers.py` 等重组为 `engine_workers.py` 布局，最新 release v0.8.0），行号锚点不可外推到 main。
- 概念延伸：采样吞吐对接轨道 03 `nano-vllm-sglang`；真正 actor/learner 异步解耦与 staleness
  对接 [`nano-slime` L0](../nano-slime/tutorial_L0.md)；importance sampling 的公式、clip 与 token/sequence
  ratio 实验见 [`sota-deepdive` §2](../sota-deepdive/post-training-algorithm-evolution.md)。
