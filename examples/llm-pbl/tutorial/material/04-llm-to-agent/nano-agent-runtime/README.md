# nano-agent-runtime

> **核心机制**：LLM 输出只是 action intent；可靠 runtime 必须在可信边界内完成 authorization、幂等执行、
> prepare/commit、崩溃恢复与人工接管，才能把“会调用工具”升级成“可安全地产生副作用”。
>
> **轨道**：[04 LLM → Agent](../README.md) · **状态**：L0 ✅，L1–L3 待补。

## 阶梯

| 级别 | 目标 | 状态 |
|------|------|------|
| L0 | 本地事务 toy：default-deny、payload-bound idempotency、prepare/commit、hash-chained log、crash recovery、needs_human | ✅ [代码](L0_transactional_side_effects.py) · [教程](tutorial_L0.md) |
| L1 | SQLite/WAL + 真实 subprocess/HTTP mock server；进程级 kill/restart 与并发重复提交 | 🔲 |
| L2 | 多 worker lease、outbox/inbox、timeout/backoff、compensation、principal/session/purpose binding | 🔲 |
| L3 | 对照真实 agent/tool runtime 与事务/outbox 模式，做权限、重放、注入与审计演练 | 🔲 |

## 推荐先修与后续

- ReAct 与 tool observation：[nano-agentscope L0](../nano-agentscope/tutorial_L0.md)
- typed message / multi-agent contract：[nano-agentscope L3](../nano-agentscope/tutorial_L3.md)
- agent ledger 与对抗自检：[nano-qwenpaw L2](../nano-qwenpaw/tutorial_L2.md)
- 长程 harness：[harness engineering](../sota-deepdive/harness-engineering.md)
- 轨迹数据合同：[EpisodeRecord L0](../../cross-track-episode-record/tutorial_L0.md)
