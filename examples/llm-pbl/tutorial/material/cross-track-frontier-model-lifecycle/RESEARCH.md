# Frontier Model Lifecycle 证据账本

> 核验快照：2026-09-08。只把论文正文、官方模型卡/仓库中的内容写入事实栏；benchmark 数字不跨 harness 排名。
> “报告披露”“开放权重”“开放训练数据/代码”“第三方复现”是四种不同证据。

## 1. 名称先消歧

| 用户口语 | 本专题使用的可核验对象 | 为什么 |
|---|---|---|
| `dpsk` | DeepSeek-V4-Pro（必要时连同 V4-Flash） | 当前官方 V4 报告与权重对象，而不是泛指 DeepSeek 全家族 |
| `qwen3.8-next` | **Qwen3.8-Flash-Next** | 官方 repo、模型卡与报告的完整名称；它是 Qwen4 架构早期预览 |
| `kimi3` | **Kimi K3** | 官方技术报告名称 |
| `glm5.3` | **GLM-5.3**；另列 **GLM-5.3-Flash** | 5.3 是同 5.2 base 的后训练更新；Flash 是不同的新 base，不能合并归因 |
| `gpt-6` | **GPT-6 Astra**，API id `gpt-6-astra` | 官方文档列出的 GPT-6 旗舰模型；参数规模和训练架构未公开 |
| `claude5.1` | **Claude Fable 5.1**，API id `claude-fable-5-1` | Anthropic 当前一般可用的 Fable 级模型；与受限访问的 Mythos 5.1 分开 |

名称不是文案细节：不锁对象就无法锁 revision、license、参数口径、训练 stage 与比较 parent。

## 2. 一手事实表

| 对象 | 架构 / 预训练事实 | 后训练事实 | 开放边界 |
|---|---|---|---|
| DeepSeek-V4-Pro | 1.6T total / 49B active；1M context；CSA+HCA、mHC、Muon；报告称训练超过 32T token | specialist 先各自 fine-tune + GRPO；再把 10+ teachers 通过 OPD 合入 student；采用 full-vocabulary reverse KL | 官方开放权重、报告与模型卡；不等于教师数据、完整训练栈和报告增益均被独立复现 |
| Qwen3.8-Flash-Next | 125B total / 6B active backbone，另有 51B n-gram embedding 与 4B MTP；3 Gated DeltaNet : 1 QSA；512 experts 中 10 routed + 1 shared；262,144 native context | 官方报告聚焦架构与预训练方法；不能由 “pre+post training” 一句话补造具体 SFT/RL 配方 | 官方 repo/权重/报告可用；详细 post-training recipe 记为 **unknown** |
| Kimi K3 | 2.8T total / 104B active；KDA、Attention Residuals、Stable LatentMoE（16/896 routed）；原生视觉与 1M context | 报告覆盖 reasoning、agentic、coding/general 等后训练；长轨 partial rollout、sandbox 与环境并行是能力的系统条件 | 官方开放权重，受 K3 license 约束；训练数据与完整生产栈未因此开放 |
| GLM-5.3 | 官方卡声明使用与 GLM-5.2 相同 base，5.3 的提升来自 post-training | 这是本组最干净的 **same-base post-training delta** 候选，但仍须固定 harness/预算做 paired replay | 模型卡数字属于厂商声明；fallback 和 harness 差异不得隐去 |
| GLM-5.3-Flash | 新训练的 320B total / 18B active base；hybrid sparse+linear attention、mHC、原生多模态；30T multimodal pretraining token | 有 post-training，但相对 GLM-5.3 的差异混入 base、规模、架构和多模态数据 | 开放权重不消除上述混杂 |
| GPT-6 Astra | 官方 API 文档给出 1,050,000 context、128,000 max output、图像输入、reasoning effort `low` 至 `max`；未披露参数/架构/pretraining | 官方模型指南公开 async tool calling、mid-turn steering 和 conversation 内 reasoning effort 调整 | API 可用；权重和训练 recipe 未公开。价格与限额是 dated service facts |
| Claude Fable 5.1 | 官方页面称其面向长时异步 coding/knowledge work，支持 vision；未披露参数/架构/pretraining | 后训练 recipe 未公开；公开的是服务行为、safeguard、fallback 与 data-retention 合同 | API 可用、权重未开放；部分 cyber/biology 请求会路由到 Opus 系模型 |

所有数字以各自来源的参数口径为准。特别是：`total parameters`、`active backbone parameters`、外置 n-gram 容量、MTP 参数、
KV/state memory 和训练 optimizer state 不可加成一个“模型大小”再做速度推断。

## 3. 五条可迁移洞察

### 3.1 DeepSeek-V4：OPD 是能力合并系统，不只是一个 KL 公式

V4 的关键不是又出现一个 reverse KL，而是把 specialist 的训练、teacher 索引、全词表分布服务、student rollout 和能力保留
连成同一系统。报告说明其教师侧采用集中存储、按需类似 ZeRO 的分片、缓存最后层 hidden state 后重建 logits，并按 teacher index
排序请求。这解释了为什么 [nano-opd](../01-post-training-rl-sft/nano-opd/) 还需继续补真实 teacher serving 与 lineage：
估计器正确不等于多教师系统的吞吐、路由和版本正确。

### 3.2 Qwen3.8：pretrain loss 不是架构晋升门

Qwen 报告把候选判断拆为三轴：loss/downstream、training/prefill/decode efficiency、optimal hyperparameters/stability。
报告还指出某些改动在预训练观察上可接受，却在后训练后失效。可迁移结论是：架构 trial 必须一直追踪到 downstream 和 post-training，
同时记录系统成本；只按 loss 选 winner 会产生 stage-local overfitting。

外置 51B n-gram embedding 是另一个好反例：它可放在 accelerator 外并预取，增加容量却不等于每 token 激活 51B backbone 参数。
课程应分别记 storage、host-device transfer、lookup hit 与 active compute。

### 3.3 Kimi K3：长轨后训练把环境变成训练状态

K3 的 partial rollout 只有在 sandbox 能 pause/resume/fork 时才真正可用；per-token regularization 允许系统容忍一定 off-policy，
但不会神奇地纠正由旧策略生成的 prefix/state。现有 [K3 deep-dive](../01-post-training-rl-sft/sota-deepdive/kimi-k3-agentic-rl-scale.md)
负责完整算法—infra 解构；本专题只记录它在整个生命周期的位置，并补上原生视觉、Stable LatentMoE 与 QAT/权重口径。

### 3.4 GLM：用一对可归因比较和一对不可归因比较教学

GLM-5 报告披露 28.5T token 训练、长上下文 midtraining、Reasoning RL → Agentic RL → General RL，以及 on-policy
cross-stage distillation；异步系统采用解耦 rollout/train、token-in-token-out gateway、双侧 importance sampling 与区间外 mask。
这些是理解 GLM-5.3 后训练方向的上游证据，但不能假装成 5.3 每一项实现都逐字开放。

- `GLM-5.3 - GLM-5.2`：官方声明同 base，可优先设计 post-training paired attribution。
- `GLM-5.3 - GLM-5.3-Flash`：base/architecture/size/multimodal pretraining 同时变化，只能称系统比较，不能称后训练消融。

这比抄一张 benchmark 表更有教学价值：它让读者真正学会什么是可识别的 estimand。

### 3.5 闭源前沿：把透视镜换成行车记录仪

研究开放权重时，我们可以拆开“发动机”，检查参数、层结构与 kernel。GPT-6 Astra 和 Claude Fable 5.1 没有提供这扇窗，
继续猜参数量只会制造伪精确。课程改用行车记录仪：固定请求、工具合同、reasoning effort、上下文、返回模型、token/cost receipt、
完成状态和 fallback，再做 paired replay。

OpenAI 官方文档让这条路线很具体。GPT-6 Astra 支持异步工具调用，工具执行时模型仍可处理其他工作；mid-turn steering 可以在
运行中加入新要求；`configuration_update` 可改变 reasoning effort，同时保留原 prompt prefix 的缓存。这些是 agent runtime 的状态机
问题，适合接到 04 轨的 pending call、journal 和 recovery。

Anthropic Fable 5.1 提供了另一个难得的反例。其 safeguard 可能把 cyber 请求交给 Opus 4.8，把 biology 请求交给 Opus 5；官方
benchmark 说明也明确记录了置零或 fallback 口径。一次 API 调用因此至少有 `requested_model`、`executed_model` 和 `routing_reason`
三项身份。遗漏其中任何一项，candidate-parent 比较都会混入隐藏处理差异。默认 30 天数据保留也会直接影响企业数据能否进入评测。

两家在 2026-09-08 都给出每百万 input/output token 10/50 美元的标题价，任务 ROI 仍不能据此判平。OpenAI 的超长提示分段计价、
cache write/read，Anthropic 的 cache read 和 fallback 规则，加上完成 token、工具费、重试与失败率，最终共同决定每个成功任务的成本。

## 4. 课程覆盖与缺口

| 生命周期环节 | 已有课程锚 | 仍缺什么 |
|---|---|---|
| 数据顺序、packing、完整 checkpoint | `nano-pretraining-loop` L0–L2 | frontier recipe 的 data mixture/version 通常未开放；不可补造 |
| DP/TP/PP/SP、MoE/MLA、精度 | `nano-fsdp`、`nano-megatron`、DeepSeek-V3 deep-dive | hybrid linear/sparse attention、mHC、Muon 与 QAT 的独立小实验 |
| SFT 与 stage dispatch | `nano-llamafactory` L0–L3 | 同一 base 的 SFT checkpoint lineage 与量化前后 paired eval |
| reasoning / agentic RL | `nano-verl`、`nano-slime`、`nano-trinity-rft`、K3 deep-dive | GLM 式 TITO version/mask 与真实长轨 completion/cost |
| multi-teacher consolidation | `nano-opd`、Capability Factory | full-vocabulary teacher service、版本错配和最坏领域保留 |
| 原生多模态 | 05 轨 VLM/DiT/H3 | pretraining→post-training 的统一 multimodal data lineage |
| 晋升、回滚与发布 | Evaluation Gate L0–L3 | model-stage parent、harness revision 与 cost ledger 的直接接线 |
| 闭源模型服务 | AgentScope API client、Agent runtime、Evaluation Gate | returned model/fallback、async pending call、mid-turn steering、token/cost receipt 与保留期 |

优先级依据是 `证据缺口 × 决策影响 × 跨轨复用 / 总成本`。因此第一批应先做 config/lineage 和小型 factorial，
不是下载所有 frontier 权重或复制厂商榜单。

## 5. 一手来源

- DeepSeek：[DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348)；[DeepSeek-V4-Pro official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)。
- Qwen：[Qwen3.8-Flash-Next technical report](https://arxiv.org/abs/2608.30320)；[official repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)；[official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
- Kimi：[Kimi K3 technical report](https://arxiv.org/abs/2607.24653)；[official repository](https://github.com/MoonshotAI/Kimi-K3)；[official model card](https://huggingface.co/moonshotai/Kimi-K3)。
- GLM：[GLM-5 technical report](https://arxiv.org/abs/2602.15763)；[GLM-5 official repository](https://github.com/zai-org/GLM-5)；[GLM-5.3 official model card](https://huggingface.co/zai-org/GLM-5.3)；[GLM-5.3-Flash official model card](https://huggingface.co/zai-org/GLM-5.3-Flash)。
- OpenAI：[GPT-6 Astra official model page](https://developers.openai.com/api/docs/models/gpt-6-astra)；[official model guide](https://developers.openai.com/api/docs/guides/latest-model)。
- Anthropic：[Claude Fable 5.1 official page](https://www.anthropic.com/claude/fable)；[Claude model lifecycle](https://platform.claude.com/docs/en/about-claude/model-deprecations)。

## 6. 明确未知

- Qwen3.8-Flash-Next 的详细 SFT/RL 阶段配方未由当前官方报告公开；不沿用或猜测其他 Qwen 代际配方。
- GLM-5 报告提供上游生命周期设计，不等于 GLM-5.3 的每个训练细节都已披露。
- GPT-6 Astra 与 Claude Fable 5.1 的参数规模、内部架构、pretraining 和 post-training recipe 未公开；context、价格或 benchmark 无法补出这些空白。
- 闭源 API 的模型、价格、限额、fallback 和保留期会变化；课程记录的是 2026-09-08 快照，运行时需重查官方文档。
- 厂商 benchmark 表不是跨模型公平排名；测试 harness、工具/搜索 fallback、采样预算、失败率和 judge 必须一同固定。
- 本账本未运行任何权重，也未验证 8×L20 上的显存、吞吐、长上下文、量化或生成质量。
