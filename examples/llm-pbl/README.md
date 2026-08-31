# LLM-PBL — Project-Based Learning for LLM Systems

> **定位**：一套以「动手做出来」为唯一验收标准的 LLM 系统学习材料。
> 不是综述，不是讲义摘抄，而是**从零重写核心组件（nano-*）→ 跑通最小闭环 → 对标权威实现与 SOTA 工程实践**的阶梯式训练场。

---

## 目标画像：senior LLM scientist & engineer

材料的终点不是「会用某个框架」，而是把人训练到 senior 水平。具体拆成两面：

- **Scientist 面**：能读懂 SOTA 论文并抓住**本质机制**（而非复述摘要）；能判断一个方法的适用边界与失败模式；能设计实验去验证一个假设。
- **Engineer 面**：能读懂**权威开源实现的核心源码**；能在吞吐 / 显存 / 稳定性之间做工程取舍；能从 0 搭起可跑的训练 / 数据 / 推理 / agent 系统。

每一节材料都要同时服务这两面：既讲清「为什么」（scientist），又给可跑代码（engineer）。

---

## 为什么叫 PBL（Project-Based Learning）

看会 ≠ 做会。本仓库的每一节都要求学习者**亲手写代码、跑出结果、解释现象**。
评判一段材料好坏的标准只有一个：**一个具备前置知识的工程师，跟着做完之后，能不能独立复现、并迁移到自己的问题上。**

五条主线对应 LLM 系统栈的五个相互咬合层面：

| 轨道 | 主题 | nano-*（对标权威实现） | SOTA 深挖 |
|------|------|------------------------|-----------|
| **01** | 后训练 / RL / SFT infra | trinity-rft · slime · verl · llamafactory | Kimi-K3（agentic RL 规模化） |
| **02** | 预训练 / CPT infra | megatron · fsdp | DeepSeek（MoE + MLA + 训练稳定性） |
| **03** | 数据 / 分布式 / RSI / 数据平台工程 | data-juicer · ray · vllm-sglang · data-platform · orchestration · rag-retrieval | LLM 数据方法论 + data-model co-dev + 湖仓/MLOps |
| **04** | LLM → Agent | agentscope · qwenpaw | Harness engineering |
| **05** | 多模态理解与生成 | vlm-understanding · image-dit · video-dit · minimax-h3-capstone | VLM/DiT/Video DiT → MiniMax H3 综合系统 |

五条线不是孤立的：03 产出的数据喂给 02/01 训练，02 的 Transformer/并行底座与 03 的数据/推理服务共同支撑 05；
01/02/05 产出的语言与多模态能力在 04 里变成 agent，agent 的运行轨迹又回流成 03 的数据——这正是
**data-model co-development（recursive self-improvement）** 的闭环，也是本仓库的核心命题之一。

> **选题来源说明**：这五条线冷启于当前的研究兴趣，但这只是 topic 的初始画像，**不是全貌、更不是目标边界**。
> 材料的标准始终对标各领域最权威的实现与 SOTA，内容会随「什么最重要、最本质」而扩展，不被任何具体项目绑定。

---

## 设计理念（连接 QwenPaw Learning Coach）

本材料与 [QwenPaw Learning Coach](../../coach/README.md) 共用一组核心教学原则：

1. **K+1 黄金法则** —— 永远只比学习者当前水平（K）高一层。不跳级、不灌水。每个 nano-* 内部采用 L0→L3 阶梯，学习顺序见[总导航](tutorial/material/README.md)。
2. **费曼技巧** —— 每节末尾必须有「能不能讲给外行听」的自检。讲不清 = 没懂。
3. **Project-Based** —— 概念挂在真实可跑的项目上，不悬空。
4. **对抗式自检（Adversarial Self-Verification）** —— 产出与验证分离；独立验证者以“严苛教授”视角寻找反例和证据缺口。
5. **零容忍反幻觉（Anti-Hallucination）** —— 所有数字、API、行数、benchmark 分数必须可溯源；不确定处显式标 `[TODO: verify]`，绝不编造。

---

## 材料形态约定

- **code-based**：核心是代码，不是散文。每个 nano-* 至少有一个**可独立运行的最小实现**（single-file 优先）。
- **notebook-style**：教程以「叙述 + 代码块 + 运行输出 + 思考题」交替推进，像 Jupyter notebook 的阅读体验（即便用 `.md` 承载）。
- **阶梯递进（ladder）**：每个主题内部从 L0 玩具实现 → L1 单卡可跑 → L2 分布式/性能 → L3 对齐权威/SOTA，逐级加码，每级都能独立验收。
- **对标权威与 SOTA**：每个 nano-* 都明确对应一个**权威开源实现**（如 verl / Megatron-LM / vLLM / Data-Juicer），L3 级别要求对照其源码做工程取舍分析；每条轨道有 SOTA 深挖。
- **educational score 优先**：宁可少而透，不可多而浅。深度 / 正确性 / 完备性 > 广度 / 数量。

---

## 目录结构

```
LLM-PBL/
├── README.md                  # 本文件：定位 + 目标画像 + 设计理念
├── shared/                    # 跨轨共用：环境、术语与评测约定
└── tutorial/material/
    ├── README.md                     # 学习路线、知识地图与交叉阅读
    ├── cross-track-episode-record/   # 轨迹事实与训练视图合同
    ├── cross-track-capability-factory/ # 多教师能力集成
    ├── cross-track-evaluation-gate/  # 配对评测、晋升与回滚边界
    ├── 01-post-training-rl-sft/      # 后训练 / RL / SFT
    │   ├── nano-trinity-rft/  nano-slime/  nano-verl/  nano-llamafactory/  nano-opd/
    │   └── sota-deepdive/
    ├── 02-pretraining-cpt/           # 预训练 / 继续预训练
    │   ├── nano-pretraining-loop/  nano-megatron/  nano-fsdp/
    │   └── sota-deepdive/
    ├── 03-data-distributed-rsi/      # 数据 / 分布式 / 递归自改进 / 数据平台工程
    │   ├── nano-data-juicer/  nano-ray/  nano-vllm-sglang/
    │   ├── nano-data-platform/  nano-data-orchestration/  nano-rag-retrieval/
    │   └── sota-deepdive/
    ├── 04-llm-to-agent/              # LLM → Agent
    │   ├── nano-agentscope/  nano-qwenpaw/  nano-agent-runtime/
    │   └── sota-deepdive/
    └── 05-multimodal-understanding-generation/
        ├── nano-vlm-understanding/  nano-image-dit/  nano-video-dit/
        └── minimax-h3-capstone/
```

---

## 快速上手

- 想先看全貌、按问题选课或查看跨轨依赖 → 读 [tutorial/material 学习总导航](tutorial/material/README.md)
- 想从某一层切入 → 进对应 `tutorial/material/0X-*/README.md`
- 想理解“candidate 怎样被裁决、可靠激活、治理 evaluator，并发布到独立 router” → 跑 [Evaluation Gate L0→L3a](tutorial/material/cross-track-evaluation-gate/)
