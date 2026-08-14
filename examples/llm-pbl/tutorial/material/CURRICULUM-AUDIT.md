# `tutorial/material` 课程级走读与审计

> 审计日期：2026-08-14（结构计数与 EpisodeRecord L1 状态同步）
>
> 范围：`tutorial/material` 下 4 条轨道、17 个 nano 模块、2 个跨轨模块与 4 个 deep-dive；当前共
> 87 篇 Markdown，其中 53 篇 L0–L3 教程。本文审的是**课程结构与抽样到的高风险机制表述**，
> 不是对全部正文、全部外部引文和 56 个脚本重新做一次逐行独立复现。

---

## 结论先行

这套材料的**单模块 educational value 已经很高**：大部分成熟模块都有明确问题、可运行锚点、
机制推导、反例、费曼自检、权威实现对照与证据边界。尤其强的是“把抽象系统机制压成一个
可失败的最小实验”，而不是只列框架 API。

目前最影响整体质量的不是再给单篇加更多细节，而是三件课程级问题：

1. **L0 脊柱已建立，真实系统层尚未闭合**：总导航、pretraining lifecycle、事务化 runtime 与
   Capability Factory 已有最小可运行锚点，EpisodeRecord 已推进到真实 PyTorch tensor adapter L1；
2. **L1–L3 仍是主要缺口**：真实 Transformer exact resume、持久 EpisodeStore/exactly-once admission、
   SQLite/WAL 进程崩溃恢复，以及真实 multi-teacher integration / hidden evaluation 尚未完成；
3. **审计证据挤进主叙事**：大量 hash、抓取日期、字节数与逐字引文保证了可信度，却会遮住
   学习者第一次阅读时真正应抓住的不变量。

当前材料已补 [学习总导航](README.md)、[EpisodeRecord L0–L1](cross-track-episode-record/)、
[pretraining lifecycle L0](02-pretraining-cpt/nano-pretraining-loop/tutorial_L0.md)、
[transactional runtime L0](04-llm-to-agent/nano-agent-runtime/tutorial_L0.md) 与
[Capability Factory L0](cross-track-capability-factory/tutorial_L0.md)。后续应保持已发布实验的可复现
契约，把这些 L0 逐级接入真实模型、持久化存储和故障注入。

---

## 审计方法与证据边界

本轮做了以下检查：

- 盘点全部 Markdown、模块层级、L0–L3 实际落盘状态与 deep-dive；
- 提取各教程 H1–H3，检查是否有“问题 → 运行 → 机制 → 反例/边界 → 自检 → 溯源”链条；
- 检查原有材料中的 216 个相对 Markdown 链接；未发现真实的文件路径断链（朴素正则报告的 6 项均是
  代码/数学里的 `[...] (...)` 形态，不是 Markdown 导航链接）；
- 抽查 PPO/GAE/IS、ZeRO/FSDP、TP/PP/SP、Ray/object store、agent message/harness 等
  高风险机制与当前实现；
- 对照模块 README、轨道 README、`ROADMAP.md` 和最新审查账本，检查范围声明与实际覆盖。

没有在本轮重新联网核验每一处论文/API/源码行号，也没有把全部 56 个脚本重新跑一遍；因此，
下文把“结构事实”“抽样发现”“建议”分开写，不把抽样审计冒充全量正确性证明。

---

## 已经做得好的部分

### 1. 机制通常落在可观察量上

优秀例子很多：SFT 的 mask 直接落到 labels；PPO 的 old/new log-prob 落到 ratio；FSDP/TP
落到每卡字节与通信事件；Ray 落到对象住所和到达顺序；AgentScope 落到 message state 与
终止原因。学习者不是“相信作者说懂了”，而是能让断言失败。

### 2. 反例不是装饰

多篇教程把错误切分、错误 mask、staleness、dead group、Goodhart、lost update、活锁、
召回阈值等做成对照实验。它们比继续堆 happy-path API 更接近 senior 所需的判断力。

### 3. toy / simulation / real execution 的边界大多写得诚实

成熟材料会说明 CPU/gloo 数字不能外推 GPU/NCCL、调度 toy 没有真的并行、规则模型不是托管
LLM、模拟器证明的是结构而非生产吞吐。这条边界必须继续保持。

### 4. 当前 PPO 教学缺陷已得到实质修复

抽查时，`nano-verl/L1_minimal_ppo.py` 已采用**右 padding + 每条序列真实长度索引**，使 PPO
update 在与 rollout 相同的 LSTM state 上重算 log-prob/value；相应教程也解释了为什么左
padding 会污染 hidden state。这是“实现正确性直接提升 educational value”的好例子。

---

## P1：最值得补的核心知识

### P1-1 把数据飞轮补成受治理的 RSI 闭环

当前 03 轨已经有 data production、filter、distributed execution、inference、snapshot、DAG
和 agent trace 回流，但“数据回来了”只证明循环存在，不证明候选模型更好，更不证明长期可靠。

**2026-08-13 更新**：已新增跨轨 [Capability Factory L0](cross-track-capability-factory/tutorial_L0.md)，
用纯算术实验补上 candidate-parent 配对 gate、隐藏回归、最坏领域回归、lineage 与 rollback target；同时把
full-vocabulary OPD 和 sampled-token MOPD 的系统权衡、错 teacher routing 反例做成 9/9 self-check。
所以下表 L0 的控制流已有可运行锚点；L1–L3 的真实模型、隐藏评估、evaluator succession 与故障注入仍是缺口。

建议新增 `nano-rsi-governance`（名字可调整），最小阶梯：

| 级别 | 必须抓住的本质 | 验收 |
|------|----------------|------|
| L0 | candidate 与 parent 的**配对评估**；promotion / rejection；不可变 snapshot；append-only lineage | 固定 seed 下生成 3 代候选，至少一次拒绝；任何 promoted model 都能追溯 parent、data、config、evaluator version |
| L1 | 置信区间或 sequential test；隐藏 sentinel / anchor；回归与能力提升同时约束 | 构造“公开分上涨但隐藏集退化”的候选，gate 必须拒绝 |
| L2 | evaluator 漂移与 succession；rollback；停止 / re-baseline；预算与风险约束 | 模拟 evaluator 被优化后失真，系统回滚且不能静默更换评价标准 |
| L3 | 对照一个真实评测/训练编排系统，说明哪些只是 proxy、哪些有独立证据 | 给出 failure injection、审计日志与恢复演练，不用单个 endpoint score 宣称 RSI 成功 |

应交叉复用：

- [reward signal / Goodhart](01-post-training-rl-sft/nano-trinity-rft/tutorial_L2.md)
- [lakehouse snapshot / lineage](03-data-distributed-rsi/nano-data-platform/tutorial_L0.md)
- [DAG 状态机](03-data-distributed-rsi/nano-data-orchestration/tutorial_L0.md)
- [对抗自检与 ledger](04-llm-to-agent/nano-qwenpaw/tutorial_L2.md)
- [RAG evaluator 指标](03-data-distributed-rsi/nano-rag-retrieval/tutorial_L0.md)

### P1-2 02 轨要补“训练生命周期”，不只补更多并行策略

02 轨对 ZeRO/FSDP/TP/PP/SP 的覆盖已经很深，再加一种切分的边际收益不高。对“预训练 / CPT”
这个轨道名而言，更本质的空缺是：

- tokenization、packing、document boundary 与 causal mask；
- sample order、shuffling、data mixture 与 deterministic resume；
- AdamW、warmup / decay、gradient accumulation 与 global batch；
- train/validation loss、data contamination 与 checkpoint selection；
- checkpoint 的模型/优化器/scheduler/RNG/data cursor 完整状态；
- fault recovery、NaN/loss spike triage，以及吞吐、成本、稳定性的联合决策。

**L0 已补**：[nano-pretraining-loop](02-pretraining-cpt/nano-pretraining-loop/) 已把 document-local causal
shift、mixture/shuffle/cursor、gradient accumulation、AdamW、warmup/decay、validation 和完整 checkpoint
串成 183 行实验；连续训练与 JSON resume 最大参数差为 0，丢 Adam moments 或 data cursor 会确定性分叉。
下一步是 L1 真实小 Transformer + optimizer/scheduler/RNG exact-resume 边界，L2 再接现有 FSDP/Megatron。

### P1-3 04 轨需要可运行的副作用事务与安全边界

现有材料很强地覆盖了消息契约、终止、记忆和验证，但轨道 README 提到的 commit / rollback
还没有一个同等强度的最小实现。建议增加 `nano-agent-runtime` 或一个可靠性专题：

- tool intent 与 authorization 分离；default-deny 与最小权限；
- idempotency key、timeout、retry、重复交付；
- prepare → commit，或无法原子提交时的 compensating action；
- append-only event log、崩溃恢复与人工接管；
- prompt injection / tool output 作为不可信输入；
- “任务成功”与“副作用已提交”是两个不同状态。

验收反例：在“工具已执行、响应丢失”处注入崩溃，重试不能重复扣款/重复发消息；无权限的模型
输出不能自行扩大 tool scope；rollback 不可用时必须进入显式 `needs_human`，不能伪装成功。

**L0 已补**：[nano-agent-runtime](04-llm-to-agent/nano-agent-runtime/) 已运行上述反例：provider commit 后
崩溃并重试只转账一次；相同 idempotency key 换 payload、tool-output 伪授权均拒绝；不可查询/不可幂等
legacy effect 进入 NEEDS_HUMAN。L1 仍需 SQLite/WAL + 进程 kill/restart，L2 再补并发 worker、outbox 与 compensation。

### P1-4 01 轨还缺 evaluator 与 rollout data contract 的统一总账

相关零件已经散落在 PPO buffer、slime staleness、Trinity reward、OPD teacher signal 中。建议做一篇
横切教程，统一一条 EpisodeRecord 至少包含：prompt/source、response/actions、token mask、old/reference
log-prob、reward components、done/truncated、bootstrap value、group id、policy/reward/evaluator version、
environment/tool trace。然后用同一条记录分别解释 PPO、GRPO/RLVR、OPD 哪些字段消费、哪些字段不用。

这会把“算法名”重新压回更本质的问题：**样本来自谁、谁给分、用哪个分布重加权、数据可以旧多久。**

**L0–L1 已补**：[EpisodeRecord](cross-track-episode-record/) 已先用一条 typed record 驱动 PPO、GRPO 和
sampled-token OPD admission，并对 missing bootstrap、stale policy、teacher/router mismatch、broken tool trace、
missing provenance 与 GRPO dead group fail closed；L1 再把四条可变长 record collate 成 PyTorch tensor batch，
运行 GAE、GRPO group gate 与 sampled-token OPD view，显式展示 episode-level 中心化不等于 token-level
中心化，并验证 data-only round-trip 后 tensor、metadata 与三种 view 一致。L2 仍需 append-only EpisodeStore、
schema migration、lease/staleness 与 exactly-once train admission。

---

## P1：需要校准的关键措辞

以下是抽样发现，适合后续按模块逐项校准；当前审计先记录问题与建议改法：

1. **Ray “吞吐线性扩展”过强**：轨道 03 README 当前这样概括，但同轨教程已经实测并解释
   startup、serialization、task granularity、straggler 与资源瓶颈。应改成“在任务粒度足够、资源可用且
   调度/数据移动未主导时扩展；线性只是理想上界”。
2. **TP 切法“唯一决定”需要加条件**：`W1` 列切、`W2` 行切是在“标准 MLP、输入复制、隐藏维均匀
   分片、目标为块内最少集合通信”等约束下的经典通信高效方案，不是所有并行布局中的数学唯一解。
3. **rollout 显存“相对小”、偏好“小 batch/低延迟”不是普遍事实**：高并发长序列时 KV cache
   可以主导显存；吞吐型 rollout 同样追求大而动态的 continuous batch。应讲“状态组成与 batch 约束
   不同”，不要讲成 rollout 天生更小。
4. **`pretrain → SFT → RL（PPO/DPO/GRPO）` 分类不严**：DPO 通常是离线偏好优化，不需要
   online rollout 环；可以说它属于 post-training / preference optimization，但不应在未解释口径时
   与 PPO/GRPO 并列成同一种 RL 执行形态。
5. **“ZeRO-1/2 是免费午餐”“ZeRO-3 固定付 1.5× 通信”需要限定口径**：本质是相对本教程的
   DDP 通信量记账与特定 wrap/re-shard 策略；真实峰值与通信量受 bucket、prefetch、重算、参数复用、
   topology 和实现影响。标题可改为“不增加本节渐近通信量”，数字标明 toy / 近似条件。
6. **toy 中的能力天花板不要外推成 SFT 的普遍定理**：训练集覆盖会约束可监督信号，但模型可能
   泛化到未逐字出现的样本。结论应限定在构造任务和给定容量/优化条件下。

这些问题大多不是代码错，而是“把给定假设下的结论写成无条件定理”；对高级教学而言，补上假设
往往比再加一段公式更重要。

---

## P2：信息架构与认知负荷

### 1. 每篇开头统一六项，不再让读者自己找

建议后续新文统一一个短 header：

```text
核心问题：这一节只解决什么？
先修：必须已懂 / 已跑哪一节？
不变量：无论实现怎样换，什么必须保持？
运行：依赖、设备、预计耗时、真实/模拟口径
验收：哪几个 assert / metric 说明学会了？
边界：这节明确不证明什么？
```

现有材料通常包含这些信息，但位置不一致；统一入口能显著降低切换成本。

### 2. 把“可读教程”和“审计证据”分层

hash、抓取字节数、mtime、逐字引文、长输出对独立复核很重要，但不应抢占第一次阅读的主线。
建议保留正文里的最小证据，并把完整内容拆成同目录的 evidence manifest：

- 正文：关键公式、3–10 行代表性输出、解释与边界；
- evidence：完整输出、环境、hash、source snapshot、引用核验表；
- README：状态与一键运行命令。

证据不能删，只是改变呈现层级。

### 3. deep-dive 应先给“决策表”，再给来源密集的纵深

四篇 deep-dive 资料很扎实，但读者先需要一页回答：方法解决什么瓶颈、付什么代价、何时不要用、
证据强度是什么。之后再进入引文与源码锚。这样能避免“来源很多”替代“形成判断”。

---

## 建议的跨轨毕业项目

与其再写一篇综述，更建议把 [Capability Factory L0](cross-track-capability-factory/tutorial_L0.md)
逐级升级成一个受治理的小闭环作为总验收：

1. 从 raw snapshot 取数据，经 OP pipeline 生成训练集；
2. 训练 parent 与 candidate，记录完整 config/data/model lineage；
3. 用版本化 rollout 生成 EpisodeRecord；
4. candidate-parent 在公开集配对评估，同时跑隐藏 sentinel；
5. promotion gate 根据效果下界、回归上界、成本和安全约束作决定；
6. 晋升后保留旧 snapshot；下轮退化时自动 rollback；
7. evaluator 更换必须新建版本并 re-baseline，不能静默覆盖历史分数；
8. 连续若干轮无显著提升或预算耗尽时停止。

最终交付不是最高分 checkpoint，而是：不可变 snapshots、append-only lineage、评估报告、promotion
decision、rollback 演练和停止理由。这个项目能把四轨真正闭合，也能清楚区分“发现了新数据”与
“可靠地改进了系统”。

---

## 下一轮验收清单

- [ ] 新读者从 [学习总导航](README.md) 能在 3 分钟内选出一条路线；
- [ ] 每条轨道 README 的层级概览与实际文件一致，且标明快照日期；
- [ ] 上述 6 条关键措辞完成带假设的校准；
- [x] governed RSI / Capability Factory L0 有 candidate-parent 配对、promotion/rejection、lineage 与 rollback；
- [ ] Capability Factory L1–L3 用真实小模型验证 multi-teacher integration，并补 hidden sentinel、
      evaluator succession、stale teacher / 错路由 / preemption 注入和真实 rollback 演练；
- [x] 01/03/04 横切 EpisodeRecord L0–L1 统一 record admission、tensor mask、GAE/GRPO/OPD view 与 round-trip；
- [ ] EpisodeRecord L2 补 append-only EpisodeStore、schema migration、lease/staleness 与 exactly-once admission；
- [x] 02 轨有一个非并行主题的完整 pretraining lifecycle L0；
- [x] 04 轨有可失败的事务化副作用 L0；
- [ ] pretraining lifecycle 与 transactional runtime 分别推进真实模型 exact resume / SQLite-WAL 进程故障注入的 L1；
- [ ] 新增教程统一六项 header，并把完整抓取证据下沉为 evidence；
- [ ] 自动链接检查能忽略代码块/数学表达式，并对真实相对链接保持零断链。
