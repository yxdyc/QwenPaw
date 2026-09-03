# nano-agent-runtime

> **核心机制**：LLM 输出只是 action intent；可靠 runtime 必须在可信边界内完成 authorization、幂等执行、
> prepare/commit、崩溃恢复与人工接管，才能把“会调用工具”升级成“可安全地产生副作用”。
>
> **轨道**：[04 LLM → Agent](../README.md) · **状态**：L0–L2 ✅，L3 待补。

## 阶梯

| 级别 | 目标 | 状态 |
|------|------|------|
| L0 | 本地事务 toy：default-deny、payload-bound idempotency、prepare/commit、hash-chained log、crash recovery、needs_human | ✅ [代码](L0_transactional_side_effects.py) · [教程](tutorial_L0.md) |
| L1 | SQLite/WAL + 真实 argv/stdout provider 子进程；进程级 kill/restart 与并发重复提交 | ✅ [代码](L1_durable_tool_runtime.py) · [教程](tutorial_L1.md) |
| L2 | 多 worker lease + provider-checked fencing epoch、原子 outbox/inbox、timeout/backoff、compensation、可信 control-plane binding | ✅ [代码](L2_distributed_runtime.py) · [教程](tutorial_L2.md) |
| L3 | 对照真实 agent/tool runtime 与事务/outbox 模式，做权限、重放、注入与审计演练 | 🔲 |

## 推荐先修与后续

- ReAct 与 tool observation：[nano-agentscope L0](../nano-agentscope/tutorial_L0.md)
- typed message / multi-agent contract：[nano-agentscope L3](../nano-agentscope/tutorial_L3.md)
- agent ledger 与对抗自检：[nano-qwenpaw L2](../nano-qwenpaw/tutorial_L2.md)
- 长程 harness：[harness engineering](../sota-deepdive/harness-engineering.md)
- 轨迹数据合同：[EpisodeRecord L0](../../cross-track-episode-record/tutorial_L0.md)

L2 用 lease 协调 worker，用单调 epoch 让 provider 拒绝 stale owner，再以 payload-bound idempotency
让安全重试收敛到同一 receipt。fencing 只在单机 SQLite provider 中模拟；没有密码学 token、网络分区或
分布式共识证据，不能把该 self-check 外推成生产保障。
