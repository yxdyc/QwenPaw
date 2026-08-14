# EpisodeRecord：rollout、训练与评估之间的统一合同

> **核心问题**：同一条 trajectory 从环境进入 PPO、GRPO/RLVR 或 OPD 时，必须保存什么，什么错误必须在训练前拒绝？
>
> **定位**：跨 01 后训练、03 数据治理和 04 agent harness 的数据合同模块；L0 不引入新算法，而是把散落在
> rollout buffer、reward、teacher signal、tool trace 和 lineage 中的不变量收拢成一条可运行记录。
>
> **状态**：L0–L1 ✅；L2–L3 待持久存储、分布式 admission 与真实框架对照。

## 阶梯

| 级别 | 核心任务 | 验收 | 状态 |
|------|----------|------|------|
| L0 | typed EpisodeRecord；PPO/GRPO/OPD admission；done/truncated；版本错配与 dead group | 标准库 CPU；9/9 self-check | ✅ [代码](episode_record_lab.py) · [教程](tutorial_L0.md) |
| L1 | PyTorch tensor batch、padding/mask、GAE/GRPO/OPD adapter 与 round-trip serialization | 变长 batch 三种 view shape/mask 全对；resume 后逐位一致；13/13 self-check | ✅ [代码](L1_tensor_batch.py) · [教程](tutorial_L1.md) |
| L2 | append-only object store、schema evolution、exactly-once train admission、staleness/lease | 重复/过期/缺字段 record 被隔离；旧 schema 可显式迁移 | 🔲 |
| L3 | 对照真实 RL/agent 框架的 rollout schema 与存储路径 | 给出字段映射、丢失语义、故障注入和重放审计 | 🔲 |

## 推荐先修与后续

- PPO/GAE：[nano-verl L1](../01-post-training-rl-sft/nano-verl/tutorial_L1.md)
- GRPO/RLVR reward failure：[nano-trinity-rft L2](../01-post-training-rl-sft/nano-trinity-rft/tutorial_L2.md)
- OPD teacher signal：[nano-opd L0](../01-post-training-rl-sft/nano-opd/tutorial_L0.md)
- lineage：[nano-data-platform L0](../03-data-distributed-rsi/nano-data-platform/tutorial_L0.md)
- tool trace：[nano-agentscope L3](../04-llm-to-agent/nano-agentscope/tutorial_L3.md)
- promotion：[Capability Factory L0](../cross-track-capability-factory/tutorial_L0.md)

## 建议运行顺序

```bash
python3 episode_record_lab.py
python3 L1_tensor_batch.py
```

L0 先固定一条轨迹的事实与版本合同；L1 再演示多条可变长 record 怎样派生为 tensor batch，并把
episode-level GRPO 中心化与 token-level 长度加权的差别摆到输出中。两级都不声称已经解决重复消费；
append-only EpisodeStore、schema migration、lease 与 exactly-once train admission 留在 L2。
