# 轨道 04 — LLM → Agent

> **一句话**：把一个会说话的模型，变成一个会**可靠地做事**的 agent——编排、工具调用、上下文工程、以及事务化的执行保障。
> **对标权威实现**：AgentScope · qwenpaw ｜ **SOTA 参照**：harness engineering（上下文 / 工具 / 可靠性 / 评测）

---

## 这条线学什么

「LLM → Agent」的鸿沟不在 prompt，而在 **harness engineering**（脚手架工程）：
- **编排**：单 agent 的 ReAct loop，多 agent 的消息传递与角色分工。
- **工具与上下文**：工具注册/调用、上下文窗口管理、记忆。
- **可靠性**：失败重试、状态回滚、动作的事务化（把数据库的 commit/rollback 语义引入 agent 动作）。

| nano-* | 抓的核心机制 | 对标权威实现 |
|--------|-------------|--------------|
| `nano-agentscope` | 多 agent 编排：消息、pipeline、工具调用 | AgentScope |
| `nano-qwenpaw` | agent harness / coach：把 LLM 包成有方法论的执行体 | qwenpaw（本仓库同源） |
| `nano-agent-runtime` | trusted authorization、幂等副作用、prepare/commit、崩溃恢复与人工接管 | L0–L2：进程内协议 → SQLite/WAL → 多 worker outbox/fencing/compensation |

---

## 学习路径（K+1 阶梯）

```
前置：会调 LLM API、懂 ReAct（propose-observe）循环（K）
  │
  ▼
Step 1  nano-agentscope L0–L1   ← 写一个 ReAct 单 agent，跑通工具调用
  │
  ▼
Step 2  nano-agentscope L2      ← 多 agent 编排：消息传递 / 角色分工
  │
  ▼
Step 3  nano-qwenpaw L1–L2      ← harness：上下文工程 / 记忆 / 方法论注入
  │
  ▼
Step 4  nano-agent-runtime L0–L2 ← 从进程内 toy 到 SQLite/WAL、多 worker、fencing 与 compensation
  │
  ▼
Step 5  可靠性专题 L3            ← 网络分区、真实下游 conditional write 与权限/重放审计
  │
  ▼
Step 6  sota-deepdive           ← SOTA harness engineering 实践
```

---

## 完成标志

- [ ] 能用 single-file 写一个 ReAct agent，跑通「思考→调工具→观察→再思考」闭环
- [ ] 能设计一个 2-agent 协作（如 planner + executor），说清消息契约
- [ ] 能解释上下文工程：何时压缩、何时检索、记忆怎么放
- [ ] 能说出 agent 动作为什么需要事务语义，并设计 payload-bound idempotency、prepare/commit 与 needs_human
- [ ] 能讲清至少 2 个 SOTA harness 的工程选择（如上下文管理 / 工具设计 / 评测）

---

## 权威实现与 SOTA 参照

写材料须回到一手来源（源码 / 技术报告 / 官方文档），拿不准标 `[TODO: verify]`：
- AgentScope：`github.com/agentscope-ai/agentscope`（message / pipeline / 工具调用）
- qwenpaw：本仓库 `coach/`（harness / 方法论注入，同源材料）
- 事务化执行：把数据库 commit/rollback 语义引入 agent 动作（概念专题，可参照数据库事务文献）
- SOTA：代表性 harness / agent 框架工程博客或报告 `[TODO: verify]`；agent 评测基准（τ-bench / SWE-bench 等）`[TODO: verify]`

→ 深挖见 [sota-deepdive/](sota-deepdive/)
