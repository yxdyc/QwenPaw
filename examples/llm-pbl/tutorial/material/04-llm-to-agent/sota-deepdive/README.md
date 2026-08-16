# SOTA Deep-Dive — Harness Engineering

> **深挖对象**：SOTA agent harness 的工程实践——上下文工程、工具设计、记忆、评测、可靠性。
> **状态**：✅ 首版完成（SOTA 对齐日期 2026-08-11）
> **可运行对照**：[nano-agentscope L0–L3](../nano-agentscope/) + [nano-qwenpaw L0–L3](../nano-qwenpaw/)。

---

## 深挖什么（scope）

1. **上下文工程**：窗口管理、压缩/检索策略、system prompt 设计。
2. **工具设计**：工具 schema、错误处理、副作用管理。
3. **可靠性 / 事务化**：失败恢复、状态回滚——把 commit/rollback 事务语义引入 agent 动作。
4. **评测**：agent 能力评测的难点（轨迹评测 vs 结果评测）。

四项 scope 在首版 deepdive 中的落点：上下文工程 → [`harness-engineering.md`](harness-engineering.md) §2；工具设计 → §4（含 skill-as-data）；可靠性/事务化 → §3（状态外化）+ §5.3（基础设施噪声）；评测 → §5（含 pass^k 可靠性度量与评测基准谱系）。另加编排（§6）与 2026 格局/三层锚点定位（§7）。

## 信息溯源要求（反幻觉硬约束）

- 数字/结论必须来自一手来源（技术报告 / 开源代码 / 官方文档）。
- 拿不到就标 `[TODO: verify]`，绝不凭印象写 benchmark 分数。
- 区分：原文声称 / 文献已有 / 合理推断 / 猜测。

## 来源清单（首版已核验，2026-08-11 现场重抓）

- [x] Anthropic engineering 四篇一手文（2025-10-16 Agent Skills / 2025-11-26 Effective harnesses / 2026-02-05 infra noise / 2026-03-24 Harness design）——详见 deepdive §9.1 表。
- [x] agent 评测基准一手来源：SWE-bench `[2310.06770]` / GAIA `[2311.12983]` / OSWorld `[2404.07972]` / τ-bench `[2406.12045]` / τ²-bench `[2506.07982]` / Terminal-Bench 2.0 `[2601.11868]` / Agent-as-a-Judge `[2410.10934]`——全部经 export.arxiv.org API 当日批量核验。
- [x] 机制谱系：ReAct `[2210.03629]` / Toolformer `[2302.04761]` / Plan-and-Solve `[2305.04091]` / ToolLLM `[2307.16789]` / MemGPT `[2310.08560]` / SWE-agent `[2405.15793]` / AgentScope `[2402.14034]` / context engineering 综述 `[2507.13334]`。
- [x] 2026 前沿（C 层对待，只讲机制类别）：A-MEM `[2502.12110]` / The Last Harness You'll Ever Build `[2604.21003]`（摘要层，正文标 [TODO: verify]）。
- 负对照记录：`2506.07989` 为物理论文（非 τ²-bench），复验纠正，见 deepdive §9.1。

## 权威实现与延伸

- 轨道 [04](../README.md)；落地参照 nano-agentscope / nano-qwenpaw 的 L0–L3 可运行材料。
- 一手来源：Anthropic engineering 四篇（URL 与日期见 deepdive §9.1）；AgentScope 开源代码（行号锚以抓取日为准）；qwenpaw 本仓库 `coach/`
