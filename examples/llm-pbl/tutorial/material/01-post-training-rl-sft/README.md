# 轨道 01 — 后训练 / RL / SFT Infra

> **一句话**：把一个 base model 变成「听话 + 会做事」的模型，所需的全部训练基础设施。
> **对标权威实现**：verl (HybridFlow) · slime · Trinity-RFT · LLaMA-Factory ｜ **SOTA 参照**：Kimi-K3（agentic RL 规模化）· GRPO 族 / RLVR · OPD（on-policy distillation）

---

## 这条线学什么

后训练（post-training）= SFT（监督微调）+ RLHF/RLVR（强化学习）+ 与 RL 正在合流的**蒸馏**（OPD, on-policy distillation：学生自己采样、教师给分布监督）。
工程难点不在算法公式，而在**如何让「采样 rollout」和「梯度更新」高效协同**——这正是 verl/slime/trinity-rft 这类框架要解决的本质问题；而 OPD 恰好复用同一套 rollout infra，是算法层的前沿（时效性 B 层，见 课程的证据时效性分层）。

| nano-* | 抓的核心机制 | 对标权威实现 |
|--------|-------------|--------------|
| `nano-llamafactory` | SFT/DPO 一站式 pipeline：数据→训练→导出 | LLaMA-Factory |
| `nano-verl` | actor-learner 分离 + rollout/train 资源调度 | verl (HybridFlow) |
| `nano-slime` | 高吞吐 rollout / 采样 infra，与训练解耦 | slime |
| `nano-trinity-rft` | 统一 SFT+RL 的训练框架，配置驱动 | Trinity-RFT |
| `nano-opd` | on-policy distillation：学生自采样 + 教师分布监督；reverse KL vs forward KL vs SFT 蒸馏；multi-teacher 融合 | 经典：MiniLLM / GKD / DistiLLM；前沿：OPD Survey / Qwen3 等生产配方 |

横切合同：[EpisodeRecord L0](../cross-track-episode-record/tutorial_L0.md) 用同一条 trajectory 解释
PPO、GRPO/RLVR 与 sampled-token OPD 分别消费哪些字段，并在训练前拒绝 termination / policy / teacher / router
版本错配；[Capability Factory L0](../cross-track-capability-factory/tutorial_L0.md) 把 multi-teacher 集成接到 promotion gate。

---

## 学习路径（K+1 阶梯）

```
前置：会 PyTorch 训练循环、懂 transformer、知道 PPO/DPO 是什么（K）
  │
  ▼
Step 1  nano-llamafactory L0–L3   ← 从 SFT 数据合同到 SFT/DPO/KTO stage dispatch
  │
  ▼
Step 2  nano-verl L0–L3           ← 引入 RL：PPO/IS、lockstep 拆分与同步 colocate
  │
  ▼
Step 3  nano-slime L0–L3          ← rollout 吞吐、异步解耦、staleness、partial rollout 与 delta sync
  │
  ▼
Step 4  nano-trinity-rft L0–L3    ← 统一框架，配置驱动 SFT+RL，对照权威实现源码
  │
  ▼
Step 5  nano-opd L0–L3            ← 从 reverse-KL 算术到 multi-teacher 与生产配方四轴
  │
  ▼
Step 6  sota-deepdive: Kimi-K3    ← 看 SOTA 如何把 agentic RL 规模化
```

---

## 完成标志

- [ ] 能用 single-file 跑通一个 toy SFT，并解释 data collator / loss mask 在做什么
- [ ] 能画出 verl 的 actor-learner 分离图，说清「为什么不能边采样边训练」
- [ ] 能解释 rollout 吞吐瓶颈，并说出 slime 用什么手段缓解
- [ ] 能用 trinity-rft 风格的配置跑通 SFT→RL 两阶段，并对照权威实现说明配置抽象的取舍
- [ ] 能讲清 Kimi-K3 在 agentic RL 规模化上的至少 2 个关键工程选择（基于一手技术报告）
- [ ] 能 single-file 跑通一个 toy OPD，解释「学生自采样 + reverse KL」为何在长程生成上优于静态教师文本的 SFT 蒸馏
- [ ] 能说出 multi-teacher OPD 的至少一种工程形态（多教师分布融合 / 路由），及其相对单教师的动机
- [ ] 能为同一条 EpisodeRecord 写出 PPO/GRPO/OPD adapter 的必填字段，并解释 `done` 与 `truncated`
- [ ] 能按课程的三层证据时效性分层，说出 PPO/DPO（经典层）与 GRPO 族/OPD（前沿层）在当今后训练格局中的定位差异

---

## 权威实现与 SOTA 参照

写材料须回到一手来源（源码 / 技术报告），拿不准标 `[TODO: verify]`：
- verl：`github.com/verl-project/verl`（HybridFlow 调度、actor-learner 分离）
- slime：`github.com/THUDM/slime`（rollout infra）
- LLaMA-Factory：`github.com/hiyouga/LLaMA-Factory`（SFT/DPO pipeline）
- Trinity-RFT：[agentscope-ai/Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT)
- OPD 经典锚点：MiniLLM `[arXiv 2306.08543]`（reverse KL + 学生自采样）、GKD `[arXiv 2306.13649]`、DistiLLM `[arXiv 2402.03898]` / DistiLLM-2 `[arXiv 2503.07067]`
- OPD 前沿：[OPD Survey 2604.00626](https://arxiv.org/abs/2604.00626)（v4 2026-06）；[Qwen3 2505.09388](https://arxiv.org/abs/2505.09388)；[MOPD 2606.30406](https://arxiv.org/abs/2606.30406)。单源变体只在核验后进入事实表。
- RLVR / GRPO 族：[DAPO 2503.14476](https://arxiv.org/abs/2503.14476) / [GSPO 2507.18071](https://arxiv.org/abs/2507.18071) / CISPO（[MiniMax-M1 2506.13585](https://arxiv.org/abs/2506.13585)）
- SOTA：[Kimi K3 技术报告 2607.24653](https://arxiv.org/abs/2607.24653)；课程解构见 [Kimi-K3 deep-dive](sota-deepdive/kimi-k3-agentic-rl-scale.md)

→ 深挖见 [sota-deepdive/](sota-deepdive/)
