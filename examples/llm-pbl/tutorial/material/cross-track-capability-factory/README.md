# Capability Factory：能力并行生产、集成与晋升

> **核心问题**：当领域专家可以并行训练后，怎样把它们集成成统一模型，同时证明集成结果值得替换 parent？
>
> **定位**：跨 01–04 四轨的毕业模块；它不把某一家模型的发布叙事当作普遍定理，而是把“能力工厂”
> 拆成可计算的 OPD 信号、可失败的 teacher routing、版本化 lineage 和独立 promotion gate。
>
> **状态**：L0 ✅；L1–L3 是后续真实模型与系统路线。

## 阶梯

| 级别 | 核心任务 | 运行与验收 | 状态 |
|------|----------|------------|------|
| L0 | full-vocabulary reverse-KL 与 sampled-token 估计；错误 teacher routing；candidate-parent gate | 纯标准库、CPU；9/9 self-check | ✅ [代码](capability_factory_lab.py) · [教程](tutorial_L0.md) |
| L1 | 同一 base 分出 2–3 个小模型专家，对照 mixed training / parameter merge / off-policy KD / sampled-token OPD / full-vocabulary OPD | GPU 可选；固定 data/token/compute budget，报告能力保留率和最坏领域回归 | 🔲 |
| L2 | 多教师服务、版本绑定、teacher routing、异步 rollout、logit/hidden-state transport 与故障恢复 | 注入 stale teacher、错路由、worker preemption；不能静默混用版本 | 🔲 |
| L3 | 受治理能力工厂：隐藏 sentinel、配对评估、promotion/rejection、rollback、evaluator succession、stopping | 至少一次拒绝和一次回滚；所有决策可重放、可追责 | 🔲 |

## 推荐先修

- OPD 的 reverse-KL 与 on-policy 来源：[nano-opd L0](../01-post-training-rl-sft/nano-opd/tutorial_L0.md)
- PPO/GRPO rollout record 与版本：[nano-verl](../01-post-training-rl-sft/nano-verl/)
- 模型状态、显存和通信：[nano-fsdp](../02-pretraining-cpt/nano-fsdp/) · [nano-megatron](../02-pretraining-cpt/nano-megatron/)
- snapshot / lineage：[nano-data-platform L0](../03-data-distributed-rsi/nano-data-platform/tutorial_L0.md)
- evaluator 与 Goodhart：[nano-trinity-rft L2](../01-post-training-rl-sft/nano-trinity-rft/tutorial_L2.md)
- harness、长轨迹与可恢复执行：[harness engineering](../04-llm-to-agent/sota-deepdive/harness-engineering.md)

## 模块边界

L0 证明的是两件结构性事实：不同 OPD 估计器有真实的方差/系统成本取舍；总分上涨不足以批准模型晋升。
它不证明 OPD 必然优于 mixed RL，不证明任一厂商模型的因果贡献，也不证明模型已经实现可靠 RSI。
