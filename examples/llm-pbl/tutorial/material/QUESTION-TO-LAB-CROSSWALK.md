# LLM-PBL Question-to-Lab Crosswalk｜从技术问题定位到可运行实验

> 用途：从一个具体技术问题定位最短的机制实验，不替代五轨完整学习路径。先明确问题和不变量，
> 再运行 L0 验证核心机制，最后按证据缺口进入 L1–L3。

## 1. 怎样把一个问题变成可验证的技术解释

每个问题沿四层递进：

| 层次 | 需要澄清什么 | 对应课程证据 |
|---|---|---|
| 本质 | 输入、状态、计算、输出 | L0 的最小机制与不变量 |
| 取舍 | 收益来自哪里，成本转移到哪里 | 账本、ablation、counterexample |
| 工程 | 状态放置、版本、通信、失败与恢复 | L1/L2 的真实模型或分布式实验 |
| 边界 | toy、代理指标和真实系统分别能证明什么 | L3 对齐、证据分层与 TODO |

只写第一层得到的是术语定义。完整的学习产出应把第二、三层连成因果链，并用第四层限制结论范围。
课程中的数字需要绑定运行环境、baseline 和证据类型。

## 2. 技术问题到最短实验

| 技术问题 | 最小实验 | 进一步实验 | 完成后应能解释 |
|---|---|---|---|
| SFT 与预训练 loss 有什么差异 | [SFT mask L0](01-post-training-rl-sft/nano-llamafactory/tutorial_L0.md) | [DPO pairs L2](01-post-training-rl-sft/nano-llamafactory/tutorial_L2.md) | 两者都可使用 shifted CE；template 和 mask 决定哪些 token 产生梯度 |
| PPO/GRPO 中 reward、value、advantage 的关系 | [verl L0](01-post-training-rl-sft/nano-verl/tutorial_L0.md) | [PPO L1](01-post-training-rl-sft/nano-verl/tutorial_L1.md) · [reward L2](01-post-training-rl-sft/nano-trinity-rft/tutorial_L2.md) | reward 是反馈，value 预测 return，advantage 表示相对预期的好坏 |
| OPD 与普通蒸馏有什么差异 | [OPD L0](01-post-training-rl-sft/nano-opd/tutorial_L0.md) | [L1→L3](01-post-training-rl-sft/nano-opd/) | 学生在自己的轨迹上采样，教师提供分布监督；KL 方向与采样分布一起解释 |
| Rollout 为什么会 stale | [slime L0](01-post-training-rl-sft/nano-slime/tutorial_L0.md) | [verl actor-learner L2](01-post-training-rl-sft/nano-verl/tutorial_L2.md) · [slime L3](01-post-training-rl-sft/nano-slime/tutorial_L3.md) | 轨迹由旧 policy 产生，版本差扩大后梯度假设和行为分布偏离 |
| 一条轨迹怎样安全进入不同算法 | [EpisodeRecord L0](cross-track-episode-record/tutorial_L0.md) | [record→tensor L1](cross-track-episode-record/tutorial_L1.md) | 先固定 provenance、termination、mask 和版本，再生成 PPO/GRPO/OPD view |
| Packing 与 checkpoint 怎样影响 exact resume | [pretraining L0](02-pretraining-cpt/nano-pretraining-loop/tutorial_L0.md) | [distributed resume L2](02-pretraining-cpt/nano-pretraining-loop/tutorial_L2.md) | 权重、optimizer、RNG、sampler/data cursor 与 rank identity 共同定义训练状态 |
| ZeRO/FSDP 和 TP/PP/SP 各切什么 | [FSDP L0](02-pretraining-cpt/nano-fsdp/tutorial_L0.md) · [TP L0](02-pretraining-cpt/nano-megatron/tutorial_L0.md) | [PP L2](02-pretraining-cpt/nano-megatron/tutorial_L2.md) · [SP L3](02-pretraining-cpt/nano-megatron/tutorial_L3.md) | 前者切训练状态，后者切层内矩阵、层级或序列；切分对象决定通信 |
| 数据快照、lineage 与 manifest 的差异 | [data-platform L0](03-data-distributed-rsi/nano-data-platform/tutorial_L0.md) | [incremental L1](03-data-distributed-rsi/nano-data-platform/tutorial_L1.md) · [commit L2](03-data-distributed-rsi/nano-data-platform/tutorial_L2.md) | snapshot 固定输入，lineage 解释流转，manifest 声明已提交产物 |
| Operator/Recipe 为什么优于脚本堆叠 | [data-juicer L0](03-data-distributed-rsi/nano-data-juicer/tutorial_L0.md) | [distributed L2](03-data-distributed-rsi/nano-data-juicer/tutorial_L2.md) · [contract L3](03-data-distributed-rsi/nano-data-juicer/tutorial_L3.md) | 稳定逻辑 contract 与物理执行解耦，才能复用、审计和按资源调度 |
| 去重、质量、配比、去污染怎样区分 | [data methodology](03-data-distributed-rsi/sota-deepdive/data-methodology.md) | [Evaluation Gate L0](cross-track-evaluation-gate/tutorial_L0.md) | 四者分别改变重复曝光、准入、token 预算和评测可信度 |
| Ray task、actor、object store 怎么选 | [Ray L0](03-data-distributed-rsi/nano-ray/tutorial_L0.md) | [actor L2](03-data-distributed-rsi/nano-ray/tutorial_L2.md) · [object store L3](03-data-distributed-rsi/nano-ray/tutorial_L3.md) | 无状态工作用 task，长寿命状态用 actor，大对象按引用共享并管理生命周期 |
| KV cache、continuous batching、PagedAttention 的关系 | [serving L0](03-data-distributed-rsi/nano-vllm-sglang/tutorial_L0.md) | [paged memory L2](03-data-distributed-rsi/nano-vllm-sglang/tutorial_L2.md) | KV 复用历史计算，continuous batching 调度请求，paging 管理内存碎片与生命周期 |
| Agent 的 action 与 observation 为什么要分开 | [AgentScope L0](04-llm-to-agent/nano-agentscope/tutorial_L0.md) | [typed message L3](04-llm-to-agent/nano-agentscope/tutorial_L3.md) · [EpisodeRecord](cross-track-episode-record/tutorial_L0.md) | model action 可训练，environment observation 是事实条件；owner 决定 mask 和归因 |
| Tool 返回成功后为什么仍需 verifier | [Agent Runtime L0](04-llm-to-agent/nano-agent-runtime/tutorial_L0.md) | [SQLite/WAL L1](04-llm-to-agent/nano-agent-runtime/tutorial_L1.md) · [multi-worker L2](04-llm-to-agent/nano-agent-runtime/tutorial_L2.md) | 协议成功不等于环境状态达成，崩溃与响应丢失要求幂等、对账和不确定态 |
| Prompt 中的方法论怎样变成可执行约束 | [QwenPaw L2](04-llm-to-agent/nano-qwenpaw/tutorial_L2.md) | [Harness deep-dive](04-llm-to-agent/sota-deepdive/harness-engineering.md) | 将原则转成 gate、状态、预算、反例和降级路径，而非只写提示词 |
| 多模态模型是否真的使用了图像 | [VLM L0](05-multimodal-understanding-generation/nano-vlm-understanding/tutorial_L0.md) | [track research](05-multimodal-understanding-generation/RESEARCH.md) | 用 image-drop/swap、2D position 等反事实区分答对与视觉依赖 |
| 视频为什么比图片更难、更贵 | [Video DiT L0](05-multimodal-understanding-generation/nano-video-dit/tutorial_L0.md) | [H3 capstone L0](05-multimodal-understanding-generation/minimax-h3-capstone/tutorial_L0.md) | token 增加时间维，attention pairs 与时序一致性成为新约束 |

## 3. 三条渐进式学习路径

### 第一遍：建立核心不变量

1. SFT mask L0：训练谁、mask 谁。
2. Pretraining L0：文档怎样进入 loss，checkpoint 需要哪些状态。
3. FSDP L0 + TP L0：区分训练状态分片与模型计算分片。
4. Serving L0：prefill/decode、KV cache、continuous batching、paging。
5. Agent Runtime L0：协议成功、环境成功与幂等恢复。

### 第二遍：补齐系统边界

在第一遍后加入 EpisodeRecord L0、Ray L0/L3、Trinity reward L2、data methodology 和 AgentScope L0。
每篇至少构造一个反例，并沿状态、版本、数据移动和失败恢复解释结果。

### 第三遍：建立跨轨因果链

按[总导航的路线 A](README.md#路线-a最短机制主干)运行各 L0，再选择一条 L2：分布式 resume、PagedAttention、
Agent Runtime 或 Data-Juicer distributed pipeline。完成标准是能独立画出版本、状态、数据移动和恢复路径，并用实验说明边界。

## 4. 每个技术问题的最小验收结构

一次完整学习至少留下：

1. 一条本质定义；
2. 一个可画出的数据流或状态机；
3. 一项主要收益及对应代价；
4. 一个会失败的反例；
5. 一种验证或观测方法。

缺少第 2 项通常意味着只有术语；缺少第 4、5 项意味着结论尚未经过可证伪检查。
