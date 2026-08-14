# nano-opd

> **抓的核心机制**：OPD（on-policy distillation）的最小机制——**散度选择决定
> 采样需求**：reverse KL 的期望在学生分布下，所以必须学生自采样（on-policy 的
> 算术由来）；教师只需对学生样本逐个打分（logprob）。同一受限学生、同一教师，
> SFT 蒸馏 / forward KL / reverse KL 三条路线跑出三种收敛形态（mode-covering
> 掉谷 vs mode-seeking 锁模），off-policy 估计器的系统性偏差做成算术定理。
> **定位句**：OPD 是 2025–2026 后训练主流方向之一（时效性 B 层前沿选题，
> 对齐记录见 tutorial §15）；MiniLLM / GKD / DistiLLM 是经典锚点（A 层）而非
> 前沿本身。本模块只取散度选择与 on-policy 来源的最小机制，不覆盖完整蒸馏栈
> （真实模型 → L1，multi-teacher → L2，生产配方对照 → L3）。
> **对应一手来源**：MiniLLM `[arXiv 2306.08543]` · GKD `[arXiv 2306.13649]` ·
> DistiLLM `[arXiv 2402.03898]` · OPD Survey `[arXiv 2604.00626]`
> **轨道**：[01 后训练/RL/SFT](../README.md) · **状态**：L0–L1 ✅，L2–L3 待补

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | single-file 确定性 toy：双峰教师 + 单峰受限学生，三配方对比（SFT / forward KL / reverse KL）+ 锁模驻点算术 + off-policy 反例（零依赖，CPU 即跑） | ✅ [L0_opd_divergence_choice.py](L0_opd_divergence_choice.py) · [tutorial_L0.md](tutorial_L0.md) |
| **L1** | 搬进真实序列模型：真实 tokenizer + 两个真实小模型作师生（22x 容量差），教师 logprob 真实前向算出，真实梯度下降；2×2 因子设计隔离信号源与散度；教师背书度指标 + 相变动力学 + self-check 断言设计史 | ✅ [L1_real_opd_seqdistill.py](L1_real_opd_seqdistill.py) · [tutorial_L1.md](tutorial_L1.md) |
| **L2** | multi-teacher OPD：多教师分布融合 / 路由的最小机制（只教 survey taxonomy 的机制类别，不追 C 层单源变体） | 🔲 |
| **L3** | 对照生产配方（Qwen3 / MiMo-V2-Flash / Thinking Machines 报告）与 survey taxonomy：divergence 设计、信号源（白盒/黑盒）、token 加权、效率与稳定化 `[TODO: verify source]` | 🔲 |

## 环境依赖

- L0：零外部依赖（纯标准库 math/random），CPU 即跑，固定 seed 逐字节确定。
- L1：依赖 torch（真实小模型前向/梯度）。CPU 即跑，约 4 秒。已在 base（py3.13.13 + torch 2.13.0）和 longds（py3.12.13 + torch 2.4.1）两环境各 3 遍验证，EXIT=0，self-check 全绿。
- L2+：视实现需 torch；真机项走 Machine B 通道 `[TODO: verify on real system]`。

## 核心要讲清的点

- 散度期望写在谁下面，样本就必须从谁采——reverse KL → 学生自采样是算术要求，不是工程习惯
- 教师角色轻到只需打分（logprob），不需要生成数据、不需要梯度——「学生写、教师批」接口
- 容量受限学生下 mode-covering（forward KL / SFT 掉谷）vs mode-seeking（reverse KL 锁模）是必然，不是调参问题
- off-policy 错误估计器的偏差是系统性的（驻点 +0.002 vs 拽离 +136.149），不是方差问题
- OPD 复用 RL 的 rollout infra——采样吞吐问题与 nano-slime / nano-vllm-sglang 同源
- L1 新增：2×2 因子设计精确隔离「信号源」与「散度」两个变量——opd_off 是 L0 错误估计器的真实模型版
- L1 新增：教师背书度（连续量）vs 有效率（二值量）——背书度有理论依据（on-policy reverse KL 直接优化它），有效率在双方都收敛后差距消失
- L1 新增：相变动力学——有效率在 step 240–299 从 0 跳到 0.17，背书度是先行指标

## 费曼自检

- 能不能用「学徒学做菜」讲清：为什么学徒必须自己做菜让师傅尝（on-policy），
  而不是抄师傅的成品（SFT）或照着菜单全覆盖（forward KL）？
- 一句话版：OPD = 学生写作文、老师批改——作文必须学生自己写，因为要批改的
  是学生写得出来的东西。

## 权威实现与延伸

- 经典锚点（一手论文）：MiniLLM `[2306.08543]`（reverse KL + 学生自采样 +
  REINFORCE 式优化）、GKD `[2306.13649]`（自采样 + 广义散度）、
  DistiLLM `[2402.03898]`（学生前缀上的 token 级 loss + 自适应调度）。
- B 层一手来源：OPD Survey `[2604.00626]`（f-divergence over student-sampled
  trajectories 的形式化 + taxonomy + production adoption：Qwen3 `[2505.09388]` /
  DeepSeek-V4 / Gemma 2 / MiMo-V2-Flash）；Thinking Machines on-policy
  distillation 博客（2025-10-27，生产配方与算力对照，数字为博客自述/转引）。
  §八 SOTA 对齐记录（2026-08-05）见 [tutorial_L0.md](tutorial_L0.md) §15。
- 工程参考：verl / SWIFT 的 distillation 支持 `[TODO: verify]`（L3 对照）。
- 概念延伸：off-policy 修正（importance sampling）→ [nano-verl](../nano-verl/) L1；
  自采样的吞吐 infra → [nano-slime](../nano-slime/) L0 与
  [nano-vllm-sglang](../../03-data-distributed-rsi/nano-vllm-sglang/) L0；
  SFT 蒸馏的数据侧 → [nano-llamafactory](../nano-llamafactory/) L0。
