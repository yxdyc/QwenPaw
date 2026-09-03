# SOTA Deep-Dive — 后训练 / Agentic RL 规模化

> **深挖对象**：01 轨两个主题——① 后训练算法演进 PPO→GRPO/RLVR→OPD（首版 ✅ + 深化 ✅）；② Kimi-K3 agentic RL 规模化（独立成篇 ✅）。
> **状态**：首版主题完成（2026-08-11）；深化与来源重对齐完成（2026-08-13，18 个 arXiv ID 复核，四项 `[TODO: verify]` 闭合）。
> **对照基础**：nano-verl、nano-opd、nano-llamafactory 均已覆盖 L0–L3。
> **深化落点**：[`post-training-algorithm-evolution-deepening.md`](post-training-algorithm-evolution-deepening.md)——四机制面可运行 sim（IS+clip / GRPO 组基线 / ratio 粒度 / OPD 合流，self-check 24/24）+ Kimi K3 / MOPD / DeepSeek-V4 正文层闭合 + survey v3→v4 版本漂移录值。

---

## 深挖什么（scope）

1. **后训练算法演进**（首版已覆盖 + 深化已覆盖）：PPO 奠基（IS + clipping）→ GRPO 族与 RLVR（去 value model、reward 换可验证信号）→ OPD（蒸馏与 RL 合流）；算法与 infra 共演化。落点：[`post-training-algorithm-evolution.md`](post-training-algorithm-evolution.md) §2–§7，2026 格局三层锚点 → §8；深化落点：[`post-training-algorithm-evolution-deepening.md`](post-training-algorithm-evolution-deepening.md)（四机制面 native sim 实测锚 + 2026-08-13 重对齐 + 引用链）。
2. **[Kimi-K3 agentic RL 规模化](kimi-k3-agentic-rl-scale.md)**（已完成）：rollout 吞吐、co-located 权重可见性与内存竞争、长轨迹 credit assignment、多步工具 reward、AgentENV sandbox；配套纯标准库 [native sim](kimi_k3_agentic_rl_sim.py) 覆盖 27 项 self-check。

## 信息溯源要求（反幻觉硬约束）

- 数字/结论必须来自一手来源（技术报告 / 开源代码 / 官方文档）。
- 拿不到就标 `[TODO: verify]`，绝不凭印象写 benchmark 分数。
- 区分：原文声称 / 文献已有 / 合理推断 / 猜测。

## 来源清单（首版已核验，2026-08-11 现场核验）

- [x] A 层经典锚点：PPO `[1707.06347]` / GAE `[1506.02438]` / DPO `[2305.18290]` / MiniLLM `[2306.08543]` / GKD `[2306.13649]` / DistiLLM `[2402.03898]` / HybridFlow `[2409.19256]`——全部经 export.arxiv.org API 当日核验，标题与 对应 nano 教程记录逐词吻合。
- [x] B 层前沿主流：DeepSeekMath/GRPO `[2402.03300]` / DeepSeek-R1 `[2501.12948]` / DAPO `[2503.14476]` / Dr. GRPO `[2503.20783]` / CISPO（MiniMax-M1 `[2506.13585]` 内）/ GSPO `[2507.18071]` / Qwen3 `[2505.09388]` / OPD Survey `[2604.00626]`（v4 2026-06-18）/ MOPD `[2606.30406]` / Kimi K3 `[2607.24653]`。
- [x] 一手博客当日重抓：GSPO 官方博客（qwenlm.github.io/blog/gspo/，2025-07-27）/ Thinking Machines On-Policy Distillation（2025-10-27，数字标 `[blog claims]`）。
- [x] 2026-08-13 重对齐闭合四项：Kimi K3 正文层（ar5iv 全文）/ MOPD 正文层 + 同一性（Xiaomi LLM Core 署名 + MiMo-V2-Flash 报告自命名）/ DeepSeek-V4（报告 §5.1/§5.1.2 直接一手，强于 survey 转述；并录 survey v4 已移除该转述的版本漂移）。
- [ ] 待核（负结果延续）：Qwen3.5 配方细节（arXiv 检索仅 Omni 报告、博客 JS 渲染，2026-08-13 录值）；另新增三项见深化文档 §8.4（K3/MOPD/DSV4 benchmark 表逐项、MOPD 实验表逐项、survey v3/v4 全文 diff）。
- C 层纪律：MAD-OPD / Uni-OPD 等单源变体无已核验 arXiv ID，只作机制类别出现，不补造 ID。

## native sim 复现声明（2026-08-13）

- `post_training_evolution_sim.py`：两个独立空 CWD 中以 `python3 -B` 运行，均 EXIT=0、self-check 24/24、输出 82 行且逐字节一致，digest `8ffa91ff…`。
- 深化文档 §1 paste 块来自该 sim；这只证明 toy 快照的确定性，不证明真实训练规模或生产吞吐。

## 权威实现与延伸

- 轨道 [01](../README.md)；落地参照 nano-verl / nano-opd / nano-llamafactory（均 L0–L3）。
- 一手来源：verl 开源代码（v0.7.1 锚点表见 nano-verl tutorial_L3 §11，行号以 2026-08-07 抓取日为准）；上述 arXiv 一手报告（详见 deepdive §10.1）
