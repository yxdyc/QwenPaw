# Frontier Model Lifecycle：追踪能力从哪里来

> **核心问题**：一个新模型的提升，究竟来自 base architecture / pretraining、后训练、蒸馏、部署，还是评测口径？
> **快照日期**：2026-09-08；这里使用官方名称 **DeepSeek-V4、Qwen3.8-Flash-Next、Kimi K3、GLM-5.3 / 5.3-Flash、GPT-6 Astra、Claude Fable 5.1**。
> **先修**：[02 预训练](../02-pretraining-cpt/README.md) + [01 后训练](../01-post-training-rl-sft/README.md)。
> **本级边界**：L0 是纯标准库的 claim/evidence 合同；不下载权重，也不证明模型质量或系统吞吐。

## 先运行，再读模型故事

```bash
python3 -B tutorial/material/cross-track-frontier-model-lifecycle/L0_stage_claim_contract.py
```

验收看 12/12 checks。脚本把别名归一到可核验对象，分开总参数、激活参数和外置容量；遇到未披露配方就停在
`unknown`。对 GPT-6 Astra 和 Claude Fable 5.1，它还会守住另一条边界：API 可访问只说明服务可调用，参数规模和训练配方仍为空。
真实输出和逐段解释见 [tutorial_L0](tutorial_L0.md)，完整证据账本见 [RESEARCH](RESEARCH.md)。

## 沿生命周期读模型

```text
data/version
    ↓
base pretrain ── architecture / optimizer / precision / parallelism
    ↓
midtrain / long-context / multimodal continuation
    ↓
SFT ── task format / reasoning format / tool schema
    ↓
specialist reasoning RL / agentic RL / general RL
    ↓
OPD or cross-stage distillation ── consolidate without silent regression
    ↓
QAT / serving ── active params, KV/state, offload, sparse kernels
    ↓
failure-aware evaluation ── same base? same harness? completion in denominator?
```

可以把模型发布看成一次长途运输。最终 checkpoint 是到站的箱子；stage、parent、数据版本和评测合同则是沿途的物流单。
箱子到了，却没有物流单，我们就分不清提升来自新底座、后训练还是更宽松的测试条件。闭源 API 还多一道转运：服务端可能
改变模型、工具或 safeguard 路由，因此请求名、实际返回模型、fallback 和完成状态都要进入记录。

## 六家模型各自教什么

| 家族 | 最值得迁移的机制 | 代价 / 风险 | 课程中的作用 |
|---|---|---|---|
| DeepSeek-V4 | CSA/HCA 百万上下文、mHC/Muon；specialist → 多教师 full-vocabulary reverse-KL OPD | 训练/教师服务复杂；官方增益仍需独立复验 | 把 02 的长上下文/优化器接到 01 的 OPD |
| Qwen3.8-Flash-Next | linear+sparse hybrid、MoE、gated residual、外置 n-gram；候选按质量/效率/稳定三轴筛选 | 不能用 pretrain loss 单独选架构；公开报告未给细致 SFT/RL recipe | 教 architecture trial gate 与训练—部署共同设计 |
| Kimi K3 | KDA、Stable LatentMoE、原生视觉；reasoning/agentic/general RL 与长轨 infra | 轨迹陈旧、sandbox 与环境成本成为主项 | 把 01 的 agentic RL 和 05 的原生多模态接起来 |
| GLM-5.3 / Flash | 同底座后训练 delta；GLM-5 的 sequential RL、跨阶段蒸馏与异步 TITO；Flash 是新底座 | 5.3↔5.2 可做后训练归因，5.3↔Flash 不可 | 教 counterfactual attribution，而非品牌内横比 |
| GPT-6 Astra | 1.05M context、128K max output；async tool call、mid-turn steering、可变 reasoning effort | 架构、参数和训练 recipe 未公开；长提示还有分段价格 | 教闭源模型的 observable contract、长任务状态与 task-level ROI |
| Claude Fable 5.1 | 长时异步 agent、vision、fallback safeguard、模型退役合同 | 部分领域请求会路由到其他模型；默认保留期影响数据治理 | 教“请求模型”和“实际执行模型”的 provenance |

## 用 ROI 安排缺口

| 缺口 | 现有锚点 | 最便宜的下一份证据 | 优先级 |
|---|---|---|---|
| 同底座 stage attribution | Evaluation Gate 已有 paired gate；尚未绑定模型 stage | config/card lineage + fixed harness paired delta | P0 |
| sparse/linear attention 的检索保真 | Megatron 教切分，VLM 教 position；缺选择器反事实 | tiny dense-vs-sparse retrieval recall/cost factorial | P0 |
| Muon 参数分组与 batch scaling | pretraining lifecycle 目前以 AdamW 为主 | 同模型/数据预算的 AdamW-vs-Muon 小实验 | P1 |
| specialist → OPD 能力保留 | nano-opd 已有估计器/路由 | 固定学生、多个 teacher、最坏领域回归 | P0 |
| async RL staleness | nano-slime 已有 partial rollout / delta sync | TITO token version + direct IS mask 故障注入 | P1 |
| QAT 的 train/serve 一致性 | FSDP L3 有精度账；缺后训练期 QAT | 小模型 SFT 前后量化 paired eval | P1 |
| n-gram offload / prefetch | KV/cache 教程已有成本账 | host-memory/transfer/命中率的离散事件 sim | P2 |
| 闭源 API provenance | AgentScope 已有 OpenAI-compatible client；Evaluation Gate 已有 lineage | requested/returned model、fallback、tool trace、token/cost receipt | P0 |
| 长任务 steering / async tools | Agent runtime 已有 durable intent/outbox | pending tool call + mid-turn correction + resume 的状态机 | P1 |

P0 的工作先消除会改变路线的未知量。小型 factorial 和 API contract replay 通常已经够用；结果真的影响决策时，再支付 GPU、
数据和长期维护成本。

## L0 → L3 阶梯

| 级别 | 新增真实约束 | 项目与验收 | 仍不能推出 |
|---|---|---|---|
| **L0 已完成** | 名称、stage、参数口径、开放/API 边界与归因进入类型系统 | 7 张卡、12 项检查；错误别名/配方/比较/架构门均被拒绝 | 任何质量、速度或训练可行性 |
| **L1 计划** | 固定官方 revision 的 config/card ledger | 只拉 metadata；重算参数/序列/显存，记录 SHA 与 license | 大权重能在目标硬件运行 |
| **L2 计划** | 小模型上的因果对照 | dense↔sparse、AdamW↔Muon、sequential RL↔OPD、sync↔async 四组最小 factorial | 可外推到 frontier scale |
| **L3 计划** | 真实开源权重、固定 harness 与系统账 | 先通过 license/磁盘/GPU gate，再报告完成率、质量、峰值显存和成本 | 厂商未开放训练数据/recipe 已被复现 |

8×L20 最适合 L1 metadata ledger、L2 小模型机制实验和经过裁剪的推理/量化复验；它不能证明 0.3T–2.8T 级模型的
完整预训练或后训练系统。GPU 使用应由“还缺哪层证据”触发，而不是由机器可用触发。

## 用四句话检查自己

- “125B + 51B n-gram，所以每 token 激活 57B 参数。”——外置容量不是激活 backbone FLOPs。
- “Qwen3.8 已开放权重，所以它的详细 RL recipe 已知。”——报告没有披露就必须保持未知。
- “GLM-5.3-Flash 比 GLM-5.3 的差异证明了后训练方法。”——它们不是同一 base，比较被混杂。
- “某架构 pretrain loss 更低，所以应进入生产。”——还缺 downstream、训练/推理效率和稳定性三轴。

再加一条闭源模型检查：若有人从 GPT-6 Astra 的上下文长度或价格反推出参数规模，证据已经越界。对 Claude Fable 5.1，
若 safeguard 触发 fallback，评测记录必须写清实际由谁完成；否则同一模型名下混入了两个执行策略。
