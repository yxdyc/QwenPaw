# Cross-track Evaluation Gate

> **核心问题**：candidate 在同一批任务上看似优于 parent，什么时候才有足够证据允许晋升？

这个模块把训练、数据、推理和 Agent 轨道共同缺少的治理边界独立出来。训练负责产生 candidate；
evaluation gate 只消费不可变快照和版本化证据，并输出 `PROMOTE` 或 `REJECT`，不能反向修改分数；
durable promotion 再把批准变成可恢复、幂等且可回滚的状态转移；evaluator governance 则保证连续试验没有
沿用已经漂移的尺子，也不能通过换 evaluator 重置误报预算。

| 级别 | 内容 | 验收 | 状态 |
|---|---|---|---|
| L0 | 配对差值、paired bootstrap、隐藏 sentinel、关键回归、成本门、不可变决策记录 | 公开均分上涨但隐藏关键项退化时拒绝；稳健候选晋升；重复证据 fail closed | ✅ [代码](evaluation_gate_lab.py) · [教程](tutorial_L0.md) |
| L1 | raw evidence durable store、PREPARE/ACTIVATE、崩溃恢复、stale-parent guard、实际 rollback | raw evidence 可重算；恢复只激活一次；rollback 后历史仍在 | ✅ [代码](L1_durable_promotion.py) · [教程](tutorial_L1.md) |
| L2 | cluster-aware exact paired test、anchor drift、evaluator epoch、显式 re-baseline 与 alpha-spending | 逐 task 显著但仅 2 cluster 时拒绝；独立 marginal trial 再被全局预算拒绝；漂移后冻结 | ✅ [代码](L2_evaluator_governance.py) · [教程](tutorial_L2.md) |
| L3a | 双 SQLite authority、outbox、lease/fencing、payload-bound idempotency、generation CAS、receipt reconcile | router commit 后丢 ACK 可恢复且只切一次；陈旧/冲突命令拒绝；未知 route drift 冻结 | ✅ [代码](L3_external_router.py) · [教程](tutorial_L3.md) |
| L3b | 真实模型盲测、多阶段 GPU canary、SLO 与 evaluator succession | quality、错误率、延迟、吞吐、成本、rollback time 与评估器换代均有可审计证据 | 🔲（需可用 GPU 环境） |

## 运行

```bash
python3 evaluation_gate_lab.py
python3 L1_durable_promotion.py
python3 L2_evaluator_governance.py
python3 L3_external_router.py
```

只依赖 Python 标准库，CPU 数秒内完成。L0 的 `hidden` split 是合成演示数据，在源码中当然可见；
它演示的是 gate 的接口和拒绝路径，不是实际保密方案。L1 使用真实 SQLite 事务，但 active pointer 是
本地发布控制的替身。L2 的 anchors、approval ids、cluster ids 与 paired deltas 也都是 synthetic：它证明的是
伪重复拒绝、冻结、换代、证据失效和 error-budget lineage，不证明字符串 cluster id 真的对应独立抽样。
L3a 使用两个真实 SQLite 事务域与逻辑时钟，但 router 仍是本地服务替身；
这些层级都不证明 evaluator 正确、审批者独立、网络协议或 GPU 盲测已经完成。

## 前后连接

- 上游事实：[EpisodeRecord](../cross-track-episode-record/) 固定 rollout、版本与终止语义；
- 上游 candidate：[Capability Factory](../cross-track-capability-factory/) 产生能力集成候选；
- 代理指标风险：[reward signals](../01-post-training-rl-sft/nano-trinity-rft/tutorial_L2.md)；
- 下游副作用：[transactional Agent runtime](../04-llm-to-agent/nano-agent-runtime/tutorial_L0.md) 说明外部路由、工具或服务不与 SQLite 同库时，还需要 outbox、幂等和补偿。

一句话验收：**candidate 先用配对证据获得批准，再通过可恢复事务激活；若评测坐标漂移则先冻结，换代后重评，
而 rollback、re-baseline 和失败 trial 都必须作为新事实留在同一 lineage。**
