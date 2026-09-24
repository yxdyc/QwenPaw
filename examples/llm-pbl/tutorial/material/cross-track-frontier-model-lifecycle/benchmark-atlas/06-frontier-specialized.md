# 分册 06：前沿专项与新兴 benchmark

> **核心问题**：最新发版开始用尚未形成统一行业口径的科研、网络安全、职业 artifact、视觉工程和实时语音评测。怎样既看见新能力，又不把不同版本、工具和指标拼成一张伪排行榜？
>
> **快照**：2026-09-22。动态厂商成绩的完整上下文见[发布证据账](RELEASE_LEDGER.md)。本页只引用 benchmark 作者、数据集/代码维护者和模型厂商的一手材料。

## 0. 阅读约定

- `[一手事实]`：来自 benchmark 论文、官方仓库、官方数据卡或官方榜；`[厂商声明]`：来自模型厂商发布页/模型卡，尚不等于独立复现；`[证据推断]`：本课程依据两者作出的有限判断。卡片中未重复加标签的 metric 定义承接 `[一手事实]`，能力阶梯、局限与水位判断承接 `[证据推断]`；未知则明确写“未披露/需从 manifest 计数/metric 身份不全”。
- 所有“课程合成例题”均标为 `[课程合成]`，只模拟能力结构，**不是** benchmark 原题、改写题或泄露题。
- “最高水位”只在 `benchmark@revision + split + metric + tools + harness + budget + trials + judge + date` 相同时才有意义。条件不齐时，本页宁可写 `NOT_COMPARABLE`。
- 下文“厂商采用”只说明该发版把此项列入评测，不证明训练使用了题目，也不证明模型权重单独造成分数。

厂商数的直接一手入口：[Qwen](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)、[Step](https://www.stepfun.com/step-5-preview)、[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)、[Kimi](https://huggingface.co/moonshotai/Kimi-K3)、[GLM](https://huggingface.co/zai-org/GLM-5.3-Flash)、[OpenAI](https://openai.com/index/gpt-6-astra/)、[Google](https://deepmind.google/models/gemini/)、[Anthropic](https://www.anthropic.com/claude-fable-and-mythos-5-1)、[xAI](https://x.ai/news/grok-4-7)、[Hy4](https://github.com/Tencent-Hunyuan/Hy4-preview)。

## 1. 前沿推理、科学与广域检索

### 1.1 CritPt：研究级物理临界点推理

- **基本信息 / 体量**：`[一手事实]` [CritPt 官方仓库](https://github.com/CritPt-Benchmark/CritPt)含 71 个挑战、190 个 checkpoints；公开测试使用 70 个挑战，由 50 多位物理研究者、30 多家机构参与构造。
- **能力项 / 阶梯**：专家知识问答 → 多步建模 → 对开放研究问题选定假设、推导并做数值/符号核验。
- **课程合成典型题**：`[课程合成]` 给出一个二维材料的有限尺寸观测与不完备边界条件，要求判断更合适的标度律、写出验证步骤并指出哪项观测能区分两个机制。
- **指标**：`[一手事实]` 公共主指标是 70 个测试挑战上、每题 5 次运行的平均 challenge accuracy；工具/无工具配置必须分开。
- **厂商采用**：`[厂商声明]` [Step 5](https://www.stepfun.com/step-5-preview) `20.9`、[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) `23.4`、[Hy4](https://github.com/Tencent-Hunyuan/Hy4-preview) `16.9`；[Claude 5.1 系统卡](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20%26%20Claude%20Mythos%205.1%20System%20Card.pdf)另报内部修订 31/71 题面的 `CritPt-Corrected mean@16 88.4`。
- **局限**：Corrected 同时改题面、工具和 16 次聚合，不能用 `88.4-23.4` 表示模型代际增益；小而难的题集也需要逐题置信区间。
- **最高水位 / 口径**：截至 2026-09-22，公共 CritPt 的本账厂商候选最高为 Kimi `23.4`，但未锁相同 harness，仍是 B-vendor；Claude `88.4` 标记 `NOT_COMPARABLE`。

### 1.2 ArXivMath：MathArena 中按月更新的论文数学题

- **基本信息 / 体量**：`[一手事实]` [MathArena 官方仓库](https://github.com/eth-sri/matharena)把 ArXivMath 作为动态 competition 配置维护，当前可见 2026-01/02/03 等月度版本；它不是一个永久固定 N 的静态数据集，题量必须从所用配置 manifest 报告。
- **能力项 / 阶梯**：竞赛数学 → 读取最新论文记号 → 重建局部证明/计算 → 用代码或检索核验。
- **课程合成典型题**：`[课程合成]` 给一段新论文中的定义和引理，要求在额外约束下推导一个数值不变量，并说明何处使用了该引理。
- **指标**：短答案可做 exact/equivalence；证明型题用 judge。默认多次采样配置常见 `n=4`，必须区分 no-tools、tools、pass@k 与平均正确率。
- **厂商采用**：`[厂商声明]` Claude 5.1 报 no-tools/tools `91.33/93.88`；Hy4 报 `66.6`。
- **局限**：月份、论文版本、题型、工具和 judge 一变，分数身份即改变；新论文并不自动排除模型通过检索看到原文。
- **最高水位 / 口径**：Claude `93.88` 只是其 tools 配置的厂商候选；Hy4 未披露同一月度配置，二者 `NOT_COMPARABLE`。

### 1.3 MathArena：赛事容器，不是一张永恒总榜

- **基本信息 / 体量**：`[一手事实]` [MathArena](https://matharena.ai/)持续纳入 AIME、HMMT、Apex、ArXivMath、BrokenArxiv 和证明赛事；规模随赛事与时间增长，必须报 competition 名、年份/月与题数。
- **能力项 / 阶梯**：单题精确答案 → 跨主题竞赛推理 → 最新/破损题面鲁棒性 → 证明生成与裁决。
- **课程合成典型题**：`[课程合成]` 在一个组合计数题中先识别题面缺失条件，再分别给出“按原文无解”和“加入最小修复后”的答案。
- **指标**：按赛事采用 exact answer accuracy、judge correctness 或 pass@k；“MathArena 总分”必须附组成与权重。
- **厂商采用**：`[厂商声明]` DeepSeek-V4.1-Flash 报 MathArena Apex `65.6`；Hy4 报 MathArena `74.2`，另报 BrokenArXiv `54.6`。
- **局限**：Apex 单赛事与多赛事聚合并非同一构念；动态赛题会发生后验公开、题面修订和 contamination 边界变化。
- **最高水位 / 口径**：两条厂商数缺少共同 competition manifest，标记 `NOT_COMPARABLE`，不以 `74.2>65.6` 排名。

### 1.4 BioMysteryBench：从真实生物数据恢复科学结论

- **基本信息 / 体量**：`[一手事实]` [Anthropic 官方数据卡](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full)的 2026-07 v11 完整版含 90 个 bioinformatics 问题；审计后移除 9 题并编辑 24 题，字段区分 `human_solvable`，输入是匿名化真实生物数据。
- **能力项 / 阶梯**：读表/图 → 选择分析流程 → 跨文件统计与生物解释 → 在未知标签下恢复可验证结论。
- **课程合成典型题**：`[课程合成]` 给匿名转录组矩阵与实验元数据，要求找出最可能的处理组、列出两项可复算证据，并拒绝用已知论文结论替代数据分析。
- **指标**：按题的结构化最终答案/判分协议计正确；必须分 human-solvable、difficult 与 full，并报告工具、尝试次数和失败分母。
- **厂商采用**：`[厂商声明]` [Gemini 3.8 Flash](https://deepmind.google/models/gemini/)报 human-solvable/difficult `88.8/56.5`；Hy4 报 `71.3`。
- **局限**：`71.3` 若未注明 subset，不能与前两列比较；生物结论可能存在多条合理流程，终点正确也不证明机制解释可靠。数据卡另含 no-training 条款，使用者须遵守许可。
- **最高水位 / 口径**：截至 2026-09-22 没有跨厂同 split 水位；Google 两个分集与 Hy4 unspecified 记 `NOT_COMPARABLE`。

### 1.5 ZeroBench：故意让当代 VLM 接近零分的视觉难题

- **基本信息 / 体量**：`[一手事实]` [ZeroBench 官方页](https://zerobench.github.io/)含 100 个 main questions、334 个 subquestions；v1/v2/v3 分别于 2025-02/03/12 发布，2026-08 又更新 grading protocol 而未改问题。
- **能力项 / 阶梯**：细粒度感知 → 多图/图表关系 → 世界知识与长链推理 → 工具辅助裁剪、OCR、代码核验。
- **课程合成典型题**：`[课程合成]` 给一张含多个微小标记的工程照片，要求先定位正确部件，再结合图例和物理约束判断唯一故障模式。
- **指标**：main/subquestions 分榜；`pass@1` 是平均单次成功，`pass@5` 是五次至少一次，`pass^5` 是五次全对，三者不可混写。
- **厂商采用**：`[厂商声明]` DeepSeek 报 tools `Pass@5 49.0`；Kimi 报 no-tools/tools `Pass@5 23/41`。
- **局限**：官方页把作者运行与 externally reported 明确分开；工具、成本和每题 token 很容易让峰值改善但稳定性下降。
- **最高水位 / 口径**：`[一手事实]` 2026-09-22 官方榜 main 当前候选 GPT-5.6 Sol max 为 `pass@1 22.0 / pass@5 30.0 / pass^5 13.0`；厂商 tools 数属于 external，不能覆盖官方水位。

### 1.6 `$OneMillion-Bench`：经济后果显著的专家 agent 任务

- **基本信息 / 体量**：`[一手事实]` [论文](https://arxiv.org/abs/2603.07980)定义 400 个专家策划任务，覆盖法律、金融、工业、医疗与自然科学。
- **能力项 / 阶梯**：权威检索 → 冲突证据消解 → 领域规则应用 → 受约束专业决策与合规表达。
- **课程合成典型题**：`[课程合成]` 根据三份互相矛盾的监管文件与公司披露，给出一项是否可执行的建议，逐条说明适用日期、例外和风险。
- **指标**：rubric 评价 factual accuracy、logical coherence、practical feasibility、professional compliance；必须保留专家/工具配置、judge 与聚合权重。
- **厂商采用**：`[厂商声明]` Qwen3.8-Max 报 expert `52.5`；Hy4 报 tools `65.4`。
- **局限**：专家 rubric 仍依赖 judge；同一总分可由事实正确但不合规、或合规但不完整构成，不能只看均值。
- **最高水位 / 口径**：Qwen 的 `expert` 与 Hy4 的 `tools` 未证明同一模式；截至快照仅为两个 B-vendor 候选，`NOT_COMPARABLE`。

### 1.7 WideSearch：广而全的结构化网页搜集

- **基本信息 / 体量**：`[一手事实]` [WideSearch 官方页](https://widesearch-seed.github.io/)有 200 个任务、18 个行业；人工平均约 2.3 小时并查阅 44+ 页面，输出 ground-truth table。
- **能力项 / 阶梯**：找一个事实 → 批量覆盖实体 → 去重/规范化 → 形成完整、可核验的结构化表。
- **课程合成典型题**：`[课程合成]` 收集某地区所有符合三项条件的公开实验室，输出机构、设备、开放日期与逐格来源，遗漏一行也要计错。
- **指标**：Success Rate 要求整表 100% 匹配；另报 row-level F1、item-level F1，以及 Avg@N、Pass@N/Max@N。
- **厂商采用**：`[厂商声明]` Qwen3.8-Max 用 Qwen-Agent 报 `81.9`；Hy4 报 `83.9`。
- **局限**：若只写 `81.9`，无法知道它是 item F1、row F1 还是 SR；搜索引擎、网页日期、agent 并发和字段等价 judge 都会改分。
- **最高水位 / 口径**：官方原始榜的不同 agent 类型与厂商新跑法不统一；两条发版数 metric 身份不完整，均标 `NOT_COMPARABLE`。

## 2. 科研编程、研究复现与模型训练

### 2.1 MLS-Bench-Lite：算法改进必须跨设置迁移

- **基本信息 / 体量**：`[一手事实]` [MLS-Bench 官方仓库](https://github.com/Imbernoulli/MLS-Bench)全集 140 题、12 个 ML 研究领域；Lite 是覆盖全部 12 领域的 30 题子集，公共榜推荐每题 5 小时探索预算。
- **能力项 / 阶梯**：改一处代码 → 提升单个实验 → 跨 seed/dataset/scale 保持增益 → 不破坏约束和计算预算。
- **课程合成典型题**：`[课程合成]` 在固定训练框架中只改采样器，让三个数据集和四个 seed 的归一化指标总体提高，且不得调高训练步数。
- **指标**：各任务用自己的可执行科学指标，再相对基线归一化并做任务/领域聚合；2026-05 官方已把领域聚合从几何均值改为算术均值。
- **厂商采用**：`[厂商声明]` Qwen `41.0`、Step `40.5`、Kimi `48.3`。
- **局限**：GPU 型号、5h agent budget、verifier 时长、web access 和 scaffold 会改变搜索空间；“改进”也不等于提出可发表的新方法。
- **最高水位 / 口径**：三条厂商数未锁同 commit/harness；Kimi `48.3`只能称本账 B-vendor 最高，不能称公共 SOTA。

### 2.2 SciCode：把科学推导变成可执行代码

- **基本信息 / 体量**：`[一手事实]` [SciCode 官方仓库](https://github.com/scicode-bench/SciCode)含 80 个 main problems、338 个 subproblems，横跨 6 个领域、16 个子域；可选择提供 scientist-authored background。
- **能力项 / 阶梯**：读科学描述 → 分解子问题 → 实现数值/符号算法 → 通过主问题集成测试。
- **课程合成典型题**：`[课程合成]` 实现一个简化反应扩散求解器；先通过边界离散、稳定步长和守恒三个子测试，再通过完整轨迹测试。
- **指标**：测试驱动的 subproblem/main-problem pass；主问题是否成功依赖所需子组件，必须说明是否给 background。
- **厂商采用**：`[厂商声明]` Step `58.9`、Kimi `58.7`。
- **局限**：环境、数值容差与依赖版本会让边界题翻转；新版 SciCode-Verified 与原版不可静默合并。
- **最高水位 / 口径**：两条仅相差 0.2pp，且未给统一 trials/commit，结论是近似同档而非 Step 确定领先。

### 2.3 SWE-Atlas：QnA、写测试、重构是三种任务

- **基本信息 / 体量**：`[一手事实]` [SWE-Atlas 官方仓库](https://github.com/scaleapi/SWE-Atlas)公开 `qa`、`tw`、`rf` 三类数据；顶层 README 未给一个应永久引用的冻结总 N，复现实验应从指定 commit manifest 计数。
- **能力项 / 阶梯**：代码库问答 → 为行为写测试 → 保持功能的结构重构；比“修一个 issue”更能拆开仓库理解与修改能力。
- **课程合成典型题**：`[课程合成]` 先解释缓存失效的调用路径，再为一个漏测边界补测试，最后在不改公开 API 的情况下抽出重复逻辑。
- **指标**：QnA/部分 rubric 用 LLM judge；test-writing/refactoring 还依赖可执行测试与约束检查；三列不能平均成无说明的总分。
- **厂商采用**：`[厂商声明]` Step 报 QnA/Test `63.6/50.8`；Hy4 报 QnA/Test/Refactor `64.0/57.8/53.3`。
- **局限**：judge、repo revision、测试沙箱与允许修改范围都会改分；缺失 Refactor 列不能当 0，也不能与三列平均比较。
- **最高水位 / 口径**：在已披露同名分项中 Hy4 QnA/Test 数字较高，但没有 matched harness，仍为 B-vendor 候选。

### 2.4 PostTrainBench：让 agent 真正训练四个小模型

- **基本信息 / 体量**：`[一手事实]` [PostTrainBench v1.1](https://posttrainbench.com/?version=v1)给每个 agent 四个 base models、单张 H100 和每次 10 小时；形成 `4 models × 7 benchmarks` 的训练—评测矩阵。
- **能力项 / 阶梯**：选数据与 SFT 配方 → 调参/诊断 → 在固定 GPU 时间内训练 → 跨任务避免过拟合和污染。
- **课程合成典型题**：`[课程合成]` 在 10 小时内提升一个 3B base 的函数调用能力，同时保证数学与代码回归不超过预注册阈值，提交权重和完整训练日志。
- **指标**：AIME 2025、Arena Hard、BFCL、GPQA Main、GSM8K、HealthBench、HumanEval 的加权平均，再跨四个 base 聚合；v1.1 另做 contamination、API-use、benchmark-lookup 和 model-identity 审计。
- **厂商采用**：`[厂商声明]` Kimi 报 `36.6`、Hy4 报 `35.6`。
- **局限**：v1.1 会把违规 run 退回 base score；manual reprompt、native CLI 与拒答 fallback 都改变被测系统，v1 数不能直接进入 v1.1 榜。
- **最高水位 / 口径**：以带版本的官方 live leaderboard 为 source of record；Kimi/Hy4 发布材料未证明是审计后的 v1.1，同标 `NOT_COMPARABLE`。

### 2.5 PaperBench：完整复现与 Code-Dev 不是一个任务

- **基本信息 / 体量**：`[一手事实]` [OpenAI PaperBench](https://openai.com/index/paperbench/)完整任务要求从零复现 20 篇 ICML 2024 Spotlight/Oral 论文，作者共建的层级 rubric 含 8,316 个可判定条目；[官方 Code-Dev 变体](https://github.com/openai/frontier-evals/blob/main/project/paperbench/README.md#paperbench-code-dev)跳过独立执行 submission 和验证实验结果，只给 code-development requirements 打分。
- **能力项 / 阶梯**：Code-Dev 测“读论文→实现代码”；完整模式再增加“建环境→实际运行→匹配结果”。后者严格包含更多 failure surface，不能用前者代称。
- **课程合成典型题**：`[课程合成]` Code-Dev 只要求根据方法说明实现训练与评测仓库；完整模式还必须锁依赖、跑出主结果、提交原始日志并解释与论文的差异。
- **指标**：两种模式都用层级 rubric aggregate，但计分节点不同；必须报 mode、task/rubric revision、时间预算、容器和 judge 版本。
- **厂商采用**：`[厂商声明]` Qwen3.8-Max 报 `93.0`。
- **局限**：原始完整评测中最佳被测 agent 的 average replication score 是 `21.0%`；Qwen 的 `93.0` 已明确属于 Code-Dev。两个数字的差首先来自 task mode/计分节点和模型时代，不是可归因的 72pp 模型提升。
- **最高水位 / 口径**：完整 PaperBench 的原始 `21.0%` 只作历史锚；Qwen `93.0` 是 BasicAgent Code-Dev、Opus 4.6 judge、3 runs、每次最多 12h 的 B-vendor 候选。二者永久分栏，标 `NOT_COMPARABLE`。

### 2.6 RoadmapBench：跨版本升级的多目标软件工程

- **基本信息 / 体量**：`[一手事实]` [RoadmapBench 官方仓库](https://github.com/UniPat-AI/RoadmapBench)有 115 个版本升级任务，来自 17 个仓库、5 种语言；每题含多个相对独立的升级目标。
- **能力项 / 阶梯**：单依赖升级 → 多 API 迁移 → 处理迁移之间的依赖 → 全仓回归与部分进度诊断。
- **课程合成典型题**：`[课程合成]` 将一个库从框架 v2 升至 v3，分别迁移配置、异步接口和序列化格式，并保证旧数据仍可读。
- **指标**：Resolved Rate 要求任务满分；Completion Score 对各目标的 partial reward 聚合。两者回答“完整交付”和“完成多少”。
- **厂商采用**：`[厂商声明]` Step 5 报 `54.3`，但发布页未在 headline 中展开它对应 Resolved 还是 Completion。
- **局限**：版本生态会漂移；网络、依赖缓存、时间预算和目标权重可主导结果。只报一个百分数无法区分完全解决与半成品。
- **最高水位 / 口径**：Step `54.3` 记为 B-vendor 且 metric 身份不全；没有可宣告的统一公开水位。

## 3. 网络安全与生物安全 agent

> 这些评测具有明显双重用途。本页只讨论评测合同、隔离和防御性解释，不提供利用步骤、payload 或可操作攻击指南。

### 3.1 CyberGym：从漏洞描述生成可验证 PoC

- **基本信息 / 体量**：`[一手事实]` [CyberGym 论文 v3](https://arxiv.org/abs/2506.02548)含 1,507 个真实漏洞、188 个开源项目，主要任务是在给定代码库和漏洞文字描述后生成可复现漏洞的 PoC test。
- **能力项 / 阶梯**：代码定位 → 根因推理 → 构造最小触发输入 → 在隔离环境中稳定复现。
- **课程合成典型题**：`[课程合成]` 在玩具解析器中定位越界读取，提交只触发 sanitizer 的最小测试输入和根因说明；不得联网或接触真实服务。
- **指标**：可执行 verifier 判断 PoC 是否在目标环境触发；需报漏洞设置、项目子集、attempt/time budget 与超时分母。
- **厂商采用**：`[厂商声明]` Step `84.7`、DeepSeek `88.1`、Grok `80.3`、Hy4 `78.4`。
- **局限**：论文公开实验称最强组合约 20% success，而发版数达 78–88；这强烈指向 subset、提示信息、聚合或新 harness 已变，不是可直接比较的进步。
- **最高水位 / 口径**：四个发版数全部标 `NOT_COMPARABLE_TO_PAPER`；在任务 manifest 与 metric 公开前不得称 DeepSeek `88.1` 为公共 SOTA。

### 3.2 CyberGym-E2E：发现、复现、修复完整闭环

- **基本信息 / 体量**：`[一手事实]` [论文 v2](https://arxiv.org/abs/2606.04460)含 920 个真实漏洞、139 个项目；[官方仓库](https://github.com/sunblaze-ucb/cybergym-e2e)区分只给源码的 `e2e` 和给 crash log+PoC 的 `patch-only`。
- **能力项 / 阶梯**：源码审计 → 漏洞发现 → PoC → patch → 回归与 ground-truth exploit 阻断。
- **课程合成典型题**：`[课程合成]` 在封网玩具项目中发现一个输入验证缺陷，提交 `poc.bin` 与 `fix.patch`，修复后既不再触发又通过原测试。
- **指标**：四级验证：agent PoC 在未修复版触发、修复版不触发、项目测试通过、ground-truth PoC 也被阻断；e2e 与 patch-only 分栏。
- **厂商采用**：截至本账九家最新发版 headline 未单列 E2E；采用状态为“公共基准已发布，暂无本账厂商成绩”。
- **局限**：长程任务受编译失败、镜像、ASLR/sanitizer、网络隔离和安全拒答影响；patch-only 高分不能推导能自主发现漏洞。
- **最高水位 / 口径**：2026-09-22 本页不抄录一个缺少统一 run manifest 的峰值；以官方榜按 `e2e/patch-only`、920-task revision 报告。

### 3.3 SEC-bench Pro：三大复杂系统中的长程漏洞猎取

- **基本信息 / 体量**：`[一手事实]` [论文](https://arxiv.org/abs/2605.26548)有 344 个经验证漏洞，覆盖 V8、SpiderMonkey 和 Linux kernel，包括 memory safety、sandbox、JIT、race 与内核子系统问题。
- **能力项 / 阶梯**：理解漏洞报告 → 跨代码库追踪 → 生成工作 PoC → 在长时间预算中诊断失败。
- **课程合成典型题**：`[课程合成]` 对一个封闭教学 VM 的玩具 JIT 缺陷生成崩溃复现，并输出可审计的调用链；不要求获得真实权限。
- **指标**：总体 solved/success，并应同时报 completed、timeout 与 judge 类型；论文说明纯规则 judge 会误判，因而引入 LLM judge。
- **厂商采用**：`[厂商声明]` DeepSeek-V4.1-Flash 报 `62.8`。
- **局限**：LLM judge 会引入可攻击面；三类目标难度不同，超时处理与完成后条件化成功率可反转排名。
- **最高水位 / 口径**：论文版本最强公开实验为 GPT-5.5/Codex `58%`；DeepSeek `62.8` 是更晚厂商数，未证明同 revision/harness，暂列 B-vendor 候选而非直接改写公共水位。

### 3.4 ExploitGym：从崩溃触发升级到实际安全影响

- **基本信息 / 体量**：`[一手事实]` [ExploitGym 论文](https://arxiv.org/abs/2605.11086)有 898 个实例，来自 userspace、V8 和 Linux kernel，并改变各实例的防护配置；[官方仓库](https://github.com/sunblaze-ucb/exploitgym)提供容器化环境。
- **能力项 / 阶梯**：已知 crash → 控制程序状态 → 绕过逐级防护 → 达成 verifier 定义的 exploit impact。
- **课程合成典型题**：`[课程合成]` 给一个只能在隔离模拟器运行的内存错误输入，要求让教学程序读取预置假文件并由 verifier 判断，不输出真实系统利用细节。
- **指标**：working exploit 的实例成功率/成功数，并按目标与防护分组；时间上限和尝试次数是 score identity。
- **厂商采用**：`[厂商声明]` DeepSeek 报 `15.3`；GPT-6 Astra 的系统卡说明其配置取消了常见 6 小时时限。
- **局限**：取消时限会改变“能力×预算”；论文最强配置成功 157/898 与 120/898，拒答/fallback 又可能改变 checkpoint 身份。
- **最高水位 / 口径**：论文候选 157/898（约 17.5%）只属于其合同；DeepSeek `15.3` 和 Astra 无时限结果均需单列，不能汇成单榜。

### 3.5 CVE-Bench：真实 Web 应用漏洞的沙箱复现

- **基本信息 / 体量**：`[一手事实]` [CVE-Bench 论文](https://arxiv.org/abs/2503.17332)围绕 critical-severity 真实 Web 应用 CVE 建沙箱；官方常用集为 40 题，但有系统卡因基础设施只跑 34/40，必须显式报实际分母。
- **能力项 / 阶梯**：读应用/公告 → 部署与侦察 → 在沙箱触发漏洞 → 生成 verifier 可确认的效果。
- **课程合成典型题**：`[课程合成]` 在本地靶场中判断一个访问控制缺陷能否读取预置测试记录，并提交结构化证据；不接触公网目标。
- **指标**：task resolved / exploit success；zero-day 与 one-day 信息条件、40/34 分母及 infra failure policy 分栏。
- **厂商采用**：`[厂商声明]` Grok 4.7 报 xhigh/high `36.6/37.7`。
- **局限**：high 反高于 xhigh 不是置信区间；小样本、基础设施失败和不同先验信息会产生数个百分点波动。
- **最高水位 / 口径**：原论文最佳 agent 最高约 `13%`，Grok 两数显然是后续配置；无 matched contract 时只保留两条正式声明，不做代际差。

### 3.6 LatchBio：benchmark 家族，不是一个固定总集

- **基本信息 / 体量**：`[一手事实]` [LatchBio 官方研究页](https://latch.bio/)维护 SpatialBench、scBench、EpiBench、TxBench、VariantBench、BioSecBench 等持续扩展的家族；没有一个稳定名为“LatchBio aggregate”的固定 N。例如 BioSecBench-Surveillance 是 100 evals，scBench-Long 是 21 evals。
- **能力项 / 阶梯**：短程结构化分析 → 长程多组学复现 → 药理/变异决策 → capability 与 biosecurity refusal 的双轴评估。
- **课程合成典型题**：`[课程合成]` 给去标识的测序文件与稀疏上下文，选择分析流程并返回限定词表中的分类、置信依据与是否需要人工复核。
- **指标**：多数子集用 deterministic endpoint pass rate；部分另用 trajectory rubric。Refusal 与 capability 方向相反，不能直接平均。
- **厂商采用**：`[厂商声明]` Grok 4.7 报 `LatchBio aggregate 44.5`，并另报 refusal/surveillance/function/safety 分项。
- **局限**：未披露子 benchmark 列表、权重、有效运行分母和 fallback，就无法重建 `44.5`；家族还在月度新增任务。
- **最高水位 / 口径**：`44.5` 标 `AGGREGATE_UNDEFINED`；最高水位必须回到每个子集@日期、model×harness、三次尝试和 CI。

## 4. 职业技能、办公 artifact 与专业分析

### 4.1 SkillsBench：技能包有没有带来配对增益

- **基本信息 / 体量**：`[一手事实]` [SkillsBench 1.1](https://www.skillsbench.ai/blogs/skillsbench-1-1)固定 87 个原生 BenchFlow tasks、8 个领域、3 个难度层；同一任务在 no-Skills 与 curated-Skills 条件下各跑 3 次。
- **能力项 / 阶梯**：无额外说明完成任务 → 读取技能包 → 正确调用其中脚本/参考 → 在多领域形成稳定 skill lift。
- **课程合成典型题**：`[课程合成]` 在相同容器中制作一份带校验的报表：A 组只给需求，B 组额外挂载专家写的检查清单与脚本，比较成对成功差。
- **指标**：deterministic verifier 的 resolution rate；`Skill Lift = with-Skills - no-Skills`，另报 invocation rate 与 95% CI。
- **厂商采用**：`[厂商声明]` Qwen `70.2`、Hy4 `62.9`。
- **局限**：skill 内容、本身是否被读取、harness 预置技能与模型交互都属于处理；只有 with-Skills 一列不能归因于 skill。
- **最高水位 / 口径**：`[一手事实]` 2026-09-22 官方 v1.1 配对榜最高公开 with-Skills 为 OpenHands+GPT-5.5 `67.3`；Qwen `70.2` 若非同 roster/paired contract，标 B-vendor 而不覆盖它。

### 4.2 APEX-Agents：跨应用专业工作的 rubric 完成度

- **基本信息 / 体量**：`[一手事实]` [2026-02 论文](https://arxiv.org/abs/2601.14242)称开源版 `n=480`；[当前官方榜](https://www.mercor.com/apex/apex-agents-leaderboard/)则明确为 31 worlds、240 tasks，覆盖投行、咨询和公司法律。两个数字代表版本变化，不能择一抹去。
- **能力项 / 阶梯**：单文件分析 → 跨应用搜集与计算 → 生成客户级 artifact → 长程维护 world state。
- **课程合成典型题**：`[课程合成]` 根据 data room 的市场资料和财务表，更新估值模型、制作三页客户演示并写一封列出假设的邮件。
- **指标**：Mean Score 是平均 rubric criteria 通过率；Pass@1 仅在一题全部 criteria 通过时计 1。
- **厂商采用**：`[厂商声明]` Step `37.8`、Kimi `41.0`、Hy4 `37.1`，发布表未统一说明它们属于 480/240 与 Mean/Pass 哪一列。
- **局限**：LM judge、应用镜像、world 版本与任务数都变；平均 60% criteria 不等于 60% 文件可交付。
- **最高水位 / 口径**：`[一手事实]` 当前 240-task Mean Score 榜 2026-09-22 显示 Fable 5.1 Max `68.6±4.9`；三条约 40 的发版数标 `VERSION_OR_METRIC_UNKNOWN`。

### 4.3 SpreadsheetBench 2：不是填几个单元格，而是交付工作簿

- **基本信息 / 体量**：`[一手事实]` [SpreadsheetBench 2 官方页](https://spreadsheetbench.github.io/)有 321 题，覆盖 financial modeling/template、debugging、visualization；平均工作簿 11.8 个 sheets、需修改约 593.5 个 cells。
- **能力项 / 阶梯**：定位单元格 → 正确公式/格式 → 跨 sheet 依赖 → 保持未要求区域不变并交付可渲染文件。
- **课程合成典型题**：`[课程合成]` 修复一个滚动预测表的三处断链公式，新增敏感性图，并保证历史 sheet 的值与样式 hash 不变。
- **指标**：strict task accuracy 要求全部指定输出正确且无意外修改；另报 modification fraction 与 visualization assertion pass（常用 70% 阈值）。
- **厂商采用**：`[厂商声明]` Step `29.4`、Kimi `34.8`。
- **局限**：Excel/LibreOffice 重算、locale、公式缓存和图表 renderer 都会影响 verifier；partial cell correctness 不代表文件能打开或可审计。
- **最高水位 / 口径**：官方页截至快照的 headline 约 `34.89`；Kimi `34.8`与之量级一致，但仍应保存精确 model/harness 和 run date。

### 4.4 PresentBench：带背景材料的专业演示文稿

- **基本信息 / 体量**：`[一手事实]` [PresentBench 官方仓库](https://github.com/PresentBench/PresentBench)及[论文](https://arxiv.org/abs/2603.07244)含 238 个实例，附背景材料；每题平均约 54.1 个二元 checklist criteria。
- **能力项 / 阶梯**：内容抽取 → 结构叙事 → 图表/版式 → 全 deck 一致性与客户可用性。
- **课程合成典型题**：`[课程合成]` 用一份市场底稿制作六页策略 deck，必须含可追溯数字、统一图例、风险页和演讲者备注。
- **指标**：对输出 deck 按 checklist/rubric 聚合；需同时报 criterion pass、all-pass、渲染成功和是否人工/LLM judge。
- **厂商采用**：`[厂商声明]` Step 5 报 `76.8`。
- **局限**：高平均 checklist 可能掩盖一个致命数字错误；字体、renderer 和模板资源属于系统条件。
- **最高水位 / 口径**：仅有 Step 正式发版候选且 headline 未完整披露 aggregation；记 B-vendor，不宣布跨系统水位。

### 4.5 GDP.pdf：长 PDF 专业 artifact 的严格交付

- **基本信息 / 体量**：`[一手事实]` [GDP.pdf 论文](https://arxiv.org/abs/2607.11192)定义 100 个任务、10 个专业领域；公开论文是体量 source of record，页面未统一披露一个应跨发版复用的页数/criterion 总数。
- **能力项 / 阶梯**：读多份 PDF → 跨文档计算 → 生成专业文件 → 通过全部关键要求。
- **课程合成典型题**：`[课程合成]` 从三份合同与两份财务附件生成一份带引用的董事会备忘录，金额、条款和页码引用均为硬门槛。
- **指标**：rubric/criterion score 与 strict all-pass 必须分开；还应记录 OCR/原图输入、工具和最终 PDF renderer。
- **厂商采用**：`[厂商声明]` Step `14.8`；Gemini `35.0 all-pass`；Claude `85.4`（系统卡未在 headline 同栏明确 all-pass/average）。
- **局限**：`85.4` 和 `35.0 all-pass`很可能是不同聚合；PDF image 与 extracted text 也不是相同输入。
- **最高水位 / 口径**：三个数一律按 metric 身份分栏；Claude 数不得被称作 85.4% 整份交付，当前结论是 `NOT_COMPARABLE`。

### 4.6 OfficeQA：同名 base、Pro、ProV2 与输入表示要分开

- **基本信息 / 体量**：`[一手事实]` [OfficeQA 官方仓库](https://github.com/databricks/officeqa)维护 Full 246、Pro 133、ProV2 90 题；Pro/Full 基于 697 份 1939–2025 Treasury Bulletins，ProV2 扩到 1,435 份 1793–2024 文档。
- **能力项 / 阶梯**：单文档定位 → 跨期检索 → 表格/脚注计算 → 在长 PDF corpus 中给精确数字。
- **课程合成典型题**：`[课程合成]` 从多期财政月报找到某项目余额，处理单位换算和修订表，再只输出带年份的最终数值与来源页。
- **指标**：QA accuracy，以数值 fuzzy matching/容差为主；需锁 Full/Pro/ProV2、PDF image 或 extracted text、retriever 和工具。
- **厂商采用**：`[厂商声明]` Step Pro `60.3`、Kimi `63.3`、GLM `62.4`、Claude Office/Pro `80.2/69.0`、Hy4 `66.2`。
- **局限**：没有 suffix 的 `OfficeQA 80.2` 不能自动当 Pro；OCR、页序、retrieval top-k 与计算工具会移动结果。
- **最高水位 / 口径**：只在明确 Pro 的列中，Claude `69.0` 是本账 B-vendor 候选；其余未统一 subset/representation，不做总排名。

### 4.7 Finance Agent v2：投行分析的多源、精确数值工作

- **基本信息 / 体量**：`[一手事实]` [Vals AI 官方页](https://www.vals.ai/benchmarks/fabv2)含 927 个专家复核问题：27 public、450 private validation、450 held-out test，覆盖 9 类分析工作流。
- **能力项 / 阶梯**：filing 检索 → 数值计算 → 跨文档调整/可比公司 → 行业惯例下的模型与结论。
- **课程合成典型题**：`[课程合成]` 从两家公司 filings 调整租赁负债与少数股权，计算可比 EV/EBITDA，并列出每个输入的报告页与舍入规则。
- **指标**：主指标是 dealbreaker-gated、severity-weighted Partial Credit；All-Pass 要求全部 checks 通过。每模型 3 runs，三 judge jury 打分。
- **厂商采用**：`[厂商声明]` Kimi `54.4`、Gemini 3.8 Flash 发布快照 `61.4`；官方 live 页还列出多家模型与工具使用。
- **局限**：私有 test 无法本地逐题复现；judge、价格数据日期、六种工具和数值容差更新会造成 live 榜漂移。
- **最高水位 / 口径**：`[一手事实]` 2026-09-22 live 页的 Partial Credit headline 为 Muse Spark 1.2 `60.60%`、All-Pass `50.88%`；Gemini 发布 `61.4` 属更早/不同快照，二者不应静默择高。

## 5. 视觉工程、专业 GUI 与视频软件理解

### 5.1 Chartography：从复杂图表中恢复结构与答案

- **基本信息 / 体量**：`[一手事实]` [Chartography 论文](https://arxiv.org/abs/2608.10677)含 100 个专业图表任务，问题由从业者提出并经 3 位专家核验；论文评估 30 个配置、每题 20 trials。
- **能力项 / 阶梯**：读轴/图例 → 定位系列 → 多步视觉计算 → 主动 crop/OCR/Python 验证。
- **课程合成典型题**：`[课程合成]` 从一张多面板能源图找出两个地区在指定季度的拐点差，并解释使用了哪两个轴与哪条曲线。
- **指标**：任务 `pass@1` 及多 trial 聚合；no-tools 与 crop+Python 必须分栏。
- **厂商采用**：`[厂商声明]` DeepSeek tools `78.9`、GLM tools `78.0`、Claude no-tools/tools `42.6/86.2`。
- **局限**：论文统一配置的最佳约 `45.0 mean pass@1`，与厂商 78–86 明显不是同一工具/judge合同；工具增益不能归入裸视觉 encoder。
- **最高水位 / 口径**：公共论文合同保留 `45.0`；厂商候选单列，Claude tools `86.2` 不覆盖公共水位。注意：`42.6→86.2` 属 Chartography，不是 CharXiv。

### 5.2 BenchCAD：大规模机械 CAD 的生成、编辑与问答

- **基本信息 / 体量**：`[一手事实]` [BenchCAD 官方页](https://benchcad.com/)与[仓库](https://github.com/BenchCAD/BenchCAD-main)覆盖 17,900 个 parts、106 个 families、47 个工程标准；含 Vision2Code 17,900、CodeEdit 748、Vision/Code QA 2,400 等任务。
- **能力项 / 阶梯**：识图问答 → 从图生成可执行 CAD code → 编辑已有模型 → 满足几何与工程标准。
- **课程合成典型题**：`[课程合成]` 从带尺寸的二维支架图生成参数化实体，并在第二轮把孔径改为新公差而保持孔距。
- **指标**：按任务报 voxel IoU、normalized IoU、symmetric ratio accuracy 等确定性指标；百分制和 `[0,1]` 小数必须统一显示尺度。
- **厂商采用**：`[厂商声明]` GPT-6 Astra 报 `95.9`；Claude 报 `0.437/0.843` 两种配置。
- **局限**：三数可能对应不同 task/subset/工具，且 `0.843` 若换算百分制是 84.3；CAD 几何相似不等于可制造性。
- **最高水位 / 口径**：全部标 task/scale 依赖；不能写成“Astra 95.9 对 Claude 0.843”的 95 倍差。

### 5.3 CADGenBench：以有效 STEP/BREP 为交付合同

- **基本信息 / 体量**：`[一手事实]` [CADGenBench 官方仓库](https://github.com/huggingface/cadgenbench)含 generation（工程图→STEP）与 editing（STEP+修改请求→STEP）；顶层资料未披露一个稳定总 N，需从 `cadgenbench-data@revision` manifest 计数。
- **能力项 / 阶梯**：生成可解析几何 → 形状接近 → 配合面/禁入区正确 → 拓扑和编辑意图正确。
- **课程合成典型题**：`[课程合成]` 根据正视/侧视尺寸生成法兰 STEP，再把两个安装孔改为沉头孔且不改变中心距。
- **指标**：Validity 是门控；其后 CAD Score 组合 surface-distance F1、volume IoU、interface keep-in/out 与 topology/Betti 指标。
- **厂商采用**：`[厂商声明]` Grok 4.7 报 CADGen `44.4`。
- **局限**：hidden ground truth 与 server-side scoring 限制完全离线复验；几何分高仍可能违反材料、公差或加工约束。
- **最高水位 / 口径**：Grok `44.4` 为单一正式候选；无相同 data revision 的跨厂比较，不宣称全局水位。

### 5.4 EEBench：由仿真和 BOM 约束验证电子设计

- **基本信息 / 体量**：`[一手事实]` [EEBench methodology](https://www.eebench.org/methodology.html)的 V1 聚焦 requirements→design→test 的模拟闭环，输出 atopile design bundle；任务保持私有，官方未披露固定 N。
- **能力项 / 阶梯**：器件/电路理解 → 代码化设计 → 仿真 worst-case corners → 同时满足技术与成本约束。
- **课程合成典型题**：`[课程合成]` 为一个温度传感器设计信号调理前端，在隐藏器件容差角下同时满足增益、噪声和 BOM 成本约束。
- **指标**：deterministic build/simulation/BOM checks；总分 `0.65 × technical + 0.35 × cost-efficiency`，工作设计才有成本 credit。
- **厂商采用**：`[厂商声明]` Grok 4.7 发布页报 `64.0`，同次官方模型卡 xhigh 报 `66.0`。
- **局限**：私有 held-out 无法自助复验；各厂可用最强原生 scaffold，官方也展示 harness effect 可大于模型代际 effect。
- **最高水位 / 口径**：保留 `64.0` 与 `66.0` 的官方冲突，不能静默取 66；待 xAI/EEBench 给出同 run id 后再消歧。

### 5.5 ScreenSpot-Pro：专业高分辨率 GUI 的单步定位

- **基本信息 / 体量**：`[一手事实]` [论文](https://arxiv.org/abs/2504.07981)有 1,581 条 instruction、每条唯一 screenshot，覆盖 23 个应用、5 类行业、3 个 OS。
- **能力项 / 阶梯**：理解操作指令 → 在完整高分辨率屏幕找小目标 → 输出准确坐标；它是 GUI agent 的感知组件测试。
- **课程合成典型题**：`[课程合成]` 在完整 CAD 软件截图中定位“切换约束显示”的小图标，只返回坐标，不执行后续操作。
- **指标**：预测点是否落在 ground-truth bounding box 的 grounding accuracy；direct、crop/zoom/search 需分栏。
- **厂商采用**：`[厂商声明]` GPT-6 Astra 报 no-tools `92.7`。
- **局限**：正确点击不代表能完成多步任务；分辨率缩放、坐标格式、crop 与 planner 都会显著改分。
- **最高水位 / 口径**：论文 2025 的 ScreenSeekeR 为 `48.1%`，Astra `92.7` 是 2026 厂商候选；因模型/协议代际不同，只能日期化并列而非把差值全归因于权重。

### 5.6 MMVU：专家级多学科视频理解

- **基本信息 / 体量**：`[一手事实]` [MMVU 官方仓库](https://github.com/yale-nlp/MMVU)含 3,000 个专家标注 QA、1,529 个专业视频、27 个 subjects，覆盖科学、医疗、人文社科与工程四大类。
- **能力项 / 阶梯**：视频定位 → 时间/过程理解 → 领域知识 → 专家级因果或程序推理。
- **课程合成典型题**：`[课程合成]` 观看一段实验操作视频，指出导致曲线漂移的具体步骤，并结合仪器原理解释为何不是后续读数误差。
- **指标**：validation 输出用 GPT-4o judge 计算 accuracy；test 隐藏并由维护方运行。应报 direct/CoT、采帧策略和输入时长。
- **厂商采用**：`[厂商声明]` Kimi K3 `82.1`、GLM-5.3-Flash `80.5`。
- **局限**：模型可能只需少量关键帧，分数不等于完整视频持续理解；judge 与采帧策略都可能带来 1–2pp 以上差异。
- **最高水位 / 口径**：两条 B-vendor 数差 1.6pp 且无 matched harness，视为同一量级，不声称 Kimi 确定领先。

### 5.7 WorldVQA：把视觉世界知识与推理解耦

- **基本信息 / 体量**：`[一手事实]` [Moonshot 官方数据卡](https://huggingface.co/datasets/moonshotai/WorldVQA)公开 3,000 个 VQA pairs、8 类；原始版 3,500/9 类中的 People 因版权与系统性拒答被移除。
- **能力项 / 阶梯**：视觉 grounding → 命名实体 → head/long-tail 世界知识覆盖；尽量不靠复杂推理掩盖记忆缺口。
- **课程合成典型题**：`[课程合成]` 给一张不常见科研仪器部件的照片，要求给出精确类别名；不要求解释工作原理。
- **指标**：公开 8 类用官方脚本报告 overall F-score；若厂商另称 accuracy，必须先核对是否只是把同一 F-score 口语化。People 版不可与移除版混算。
- **厂商采用**：`[厂商声明]` Kimi K3 报 `51.0`。
- **局限**：官方数据卡仍写“无模型超过 50%”，与更晚 Kimi `51.0`形成日期/协议张力；版权移除也改变 denominator。
- **最高水位 / 口径**：Kimi `51.0` 是 2026-07 厂商候选；在官方榜刷新并绑定 3,000/8 类前，标 `NEWER_EXTERNAL_RESULT`，不改写旧页结论。

### 5.8 SWE-MM / SWE-bench Multimodal：视觉证据驱动的前端修复

- **基本信息 / 体量**：`[一手事实]` 本账的 `SWE-MM` 对应 [SWE-bench Multimodal 论文](https://arxiv.org/abs/2410.03859)：617 个真实 task instances、17 个 JavaScript libraries，issue 中的截图等视觉证据对修复必要。
- **能力项 / 阶梯**：理解 issue+图像 → 定位前端/可视化代码 → 修改仓库 → 通过可执行测试并匹配预期视觉行为。
- **课程合成典型题**：`[课程合成]` issue 附一张折线图错位截图；在仓库中定位响应式布局 bug，提交最小 patch 和回归测试。
- **指标**：与 SWE-bench 类似的 instance resolved rate，由容器中 fail-to-pass/pass-to-pass tests 判定；需报 exact split、agent 和图片输入方式。
- **厂商采用**：`[厂商声明]` [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)报 `38.6`。
- **局限**：测试未必捕获视觉质量；图像预处理、浏览器 renderer 与 agent 可否打开本地页面均影响结果。
- **最高水位 / 口径**：只有 Qwen 新发版候选，无法给统一水位；`38.6` 不应与文本 SWE-bench Verified/Pro 比较。

### 5.9 视觉专项的稳定合并原则

- BenchCAD 与 CADGenBench 可在“机械 CAD artifact”目录并列，但不可合分：前者有多任务大数据，后者以隐藏 STEP ground truth 和 validity-gated CAD Score 为核心。
- EEBench 与 EEE-Bench 不是同一基准：本页 EEBench 是 atopile/仿真/BOM 的 agent design；不要拿考试式多模态电工题的 accuracy 混入。
- ScreenSpot-Pro 是单步 grounding；OSWorld 是多步环境成功。WorldVQA 是原子视觉知识；MMVU 是专家视频推理。四者分数没有共同分母。

## 6. 原生 speech-to-speech 与 τ-Voice

### 6.1 Artificial Analysis Speech-to-Speech Index：四维合成指数

- **基本信息 / 体量**：`[一手事实]` [官方 methodology](https://artificialanalysis.ai/methodology/speech-to-speech-benchmarking)的 2026-08 v2.0 指数等权组合四部分：Big Bench Audio、τ-Voice、Speech Agent Arena、Task Success Rate；各子集体量/有效 call 数应随榜单 snapshot 报，不能把指数当题数。
- **能力项 / 阶梯**：音频推理 → 实时轮流说话/打断 → 工具型对话完成 → 人类偏好与端到端任务成功。
- **课程合成典型题**：`[课程合成]` 用户边说边更正航班日期，背景有噪声；系统需及时停说、确认新日期并只执行一次合法改签。
- **指标**：四项各 25% 的 composite；另报 Time to First Audio、输入/输出音频价格。BBA v1.0/1.1/1.2 的失败分母与 judge 也不同。
- **厂商采用**：`[厂商声明]` [Gemini 3.8 Live](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-8-live-gemini-3-8-live-extended-thinking/)报 Speech-to-Speech Quality Index `82.6`。
- **局限**：一个综合数可掩盖低任务成功或高延迟；ASR/TTS、流式 endpoint、地域网络和安全路由都是被测系统。
- **最高水位 / 口径**：`82.6` 只属于 AA Index v2.0@该日期；v1.x 三/四组件指数不得拼接，文本模型也不进入此榜。

### 6.2 τ-Voice：278 个可打断、带噪全双工 agent 任务

- **基本信息 / 体量**：`[一手事实]` [τ-Voice 论文](https://arxiv.org/abs/2603.13686)把 278 个 retail/airline/telecom 任务转为 clean/realistic 全双工音频交互；[官方指标文档](https://github.com/sierra-research/tau2-bench/blob/main/docs/interaction-metrics.md)同步实现交互质量指标。
- **能力项 / 阶梯**：听懂内容 → 合规 tool use → 处理口音/噪声 → 识别 backchannel → 被打断时让出话轮并恢复状态。
- **课程合成典型题**：`[课程合成]` 用户咨询退货时插话补充订单号，途中有旁人说话；agent 只把定向话语当指令，核验政策并完成合法退款。
- **指标**：task `pass^1/pass@1` 加 responsiveness、latency、interrupt rate、selectivity；clean 与 realistic、领域和实时 provider 分栏。
- **厂商采用**：`[厂商声明]` Gemini 3.8 Live 报 τ-Voice `68.6`、τ-Voice-banking `35.1`。
- **局限**：语音 user simulator、VAD、turn endpoint、噪声合成和通信 judge 会改变结果；backend 成功不等于自然、低延迟或不抢话。
- **最高水位 / 口径**：两数属于两个任务集，不能平均；截至 2026-09-22 只称 Google 官方系统候选，不与文本 τ³-Banking 比较。

## 7. Legacy / base probing：紧凑回归表

> `[一手事实]` 下表的体量与定义来自各 benchmark 官方论文/仓库；`[厂商声明]` “采用/水位”均指
> [DeepSeek-V4.1-Flash Base 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)的 2026-09-22 发版快照。
> 这些 probe 适合做低成本回归，不应与 Instruct/Agent 系统混表。每个课程例题仍为合成题而非原题。

| Benchmark 与基本信息/体量 | 能力项 / 阶梯；课程合成典型题 | 指标 | 厂商采用与日期化水位 | 主要局限 |
|---|---|---|---|---|
| [AGIEval](https://github.com/ruixiangcui/AGIEval)：20 类官方入学/资格考试；总 N 随中英子集 manifest 报告 | 考试知识→指令遵循→推理；`[课程合成]` 根据法规摘录选唯一合法程序 | MCQ accuracy / exact，按 zero/few-shot 分栏 | DeepSeek Base `83.4`；B-vendor，无同协议跨厂水位 | 老题公开、模板敏感；考试分不等于开放工作能力 |
| [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)：约 12K 题、14 个领域、10 个选项 | 学科知识→长链消歧；`[课程合成]` 在十个近似物理选项中选满足边界条件者 | accuracy，需报 CoT/answer extraction | DeepSeek Base `74.1`；B-vendor | 公开题污染与选项格式效应；不测工具或交付 |
| [C-Eval](https://cevalbenchmark.com/)：13,948 道中文选择题、52 学科 | 中文学科知识→专业考试；`[课程合成]` 判断一项中国会计处理 | accuracy，按学科/难度聚合 | DeepSeek Base `92.1`；高饱和 B-vendor | 文化/考试范围窄，接近饱和时小差值不稳 |
| [MultiLoKo](https://arxiv.org/abs/2504.10356)：31 种语言的本地知识；本页不固化 N，以论文/data revision 为准 | 多语表达→地方性事实；`[课程合成]` 用目标语言回答地区公共制度的稳定事实 | exact/judge accuracy，按语言与 head/long-tail 报告 | DeepSeek Base `45.5`；B-vendor | 翻译等价、知识时点与区域覆盖不均；不能用宏平均掩盖低资源语言 |
| [SimpleQA Verified](https://arxiv.org/abs/2509.07968)：1,000 个经复核短事实题，源自 4,326 题 SimpleQA | 短事实→知道自己不知道；`[课程合成]` 问一个有唯一、时点稳定答案的实体事实 | correct / incorrect / not-attempted 与 accuracy | DeepSeek Base `42.3`；B-vendor | 单事实不等于长报告 factuality；judge 与拒答策略影响分数 |
| [SuperGPQA](https://github.com/SuperGPQA/SuperGPQA)：26,529 题、13 大类、72 fields、285 个研究生 disciplines | 广度→长尾专家知识；`[课程合成]` 选出罕见材料表征方法的正确适用条件 | MCQ sample/field/discipline macro accuracy | DeepSeek Base `53.1`；B-vendor | STEM 占比高、学科宏平均选择会改排名；题目公开 |
| [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)：23 个从 BIG-Bench 筛出的难任务；N 依 task files | 规则归纳→多步符号推理；`[课程合成]` 根据四个示例推断隐藏字符串变换 | exact-match average across tasks | DeepSeek Base `86.1`；B-vendor | 已广泛用于训练/提示优化，趋于饱和；task 宏平均掩盖短板 |
| [BIG-Bench Extra Hard](https://github.com/google-deepmind/bbeh)：BBH 后继 hard suite；N 以官方 release 为准 | 更强组合推理→反捷径；`[课程合成]` 在冲突规则下追踪多对象状态 | 官方 task grader 的平均 accuracy | DeepSeek Base `27.2`；B-vendor | decoding、采样和 grader extraction 敏感；名称接近 BBH 但非同集 |
| [DROP](https://allenai.org/data/drop)：约 96K QA、约 6.7K passages | 段落检索→离散数值推理；`[课程合成]` 从赛季叙述算两队净胜分差 | Exact Match 与 token-level F1 | DeepSeek Base `F1 87.9`；B-vendor | F1 可奖励部分字符串；不要求可审计计算过程 |
| [HellaSwag](https://rowanzellers.com/hellaswag/)：约 70K 常识续写实例 | 情境理解→选择合理后续；`[课程合成]` 从四个动作中选物理可行的下一步 | 4-choice accuracy | DeepSeek Base `87.2`；B-vendor | 老题、高污染风险、接近饱和；分类不等于生成连贯计划 |
| [BigCodeBench](https://github.com/bigcode-project/bigcodebench)：1,140 个库调用型编程任务 | 函数生成→多库组合；`[课程合成]` 写函数解析压缩日志并生成统计表 | `pass@1`（完整测试）及 instruct/complete split | DeepSeek Base `60.6`；B-vendor | sandbox/dependency 与提示格式敏感；函数级不代表 repo 工程 |
| [HumanEval](https://github.com/openai/human-eval)：164 个 Python 函数题 | 规格→短函数实现；`[课程合成]` 实现稳定去重并保持首见顺序 | unit-test `pass@k` | DeepSeek Base `79.4`；B-vendor | 极小且公开、污染/饱和严重；隐藏测试覆盖有限 |
| [GSM8K](https://github.com/openai/grade-school-math)：约 8.5K 小学应用题（约 7.5K train/1.3K test） | 文本算术→短链推理；`[课程合成]` 计算两次折扣后的剩余数量 | final-answer exact accuracy | DeepSeek Base `93.0`；B-vendor | 接近饱和，答案提取与 CoT prompting 主导小差异 |
| [MATH](https://github.com/hendrycks/math)：12.5K 竞赛题（7.5K train/5K test） | 代数/几何→竞赛推理；`[课程合成]` 求一个参数多项式的整数根条件 | boxed-answer equivalence / accuracy | DeepSeek Base `61.1`；B-vendor | 公开训练集与衍生数据多；等价 checker 和题级难度影响大 |
| [MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm)：250 个对齐问题×10 种语言 | 算术→跨语迁移；`[课程合成]` 用斯瓦希里语描述并回答库存变化 | 各语言 exact accuracy 与 macro average | DeepSeek Base `80.2`；B-vendor | 翻译腔、每语仅 250 题、宏平均高方差；不是地方知识 |
| [LongBench v2](https://github.com/THUDM/LongBench)：503 个困难长上下文选择题，长度从约 8K 到 2M words | 长文定位→跨段推理；`[课程合成]` 在多份相似合同中找唯一例外并判断后果 | accuracy，按长度/领域/难度切片 | DeepSeek Base `45.2`；B-vendor | context truncation、packing 与最大输入决定可比性；选择题不等于长程 agent |
| [MMMU-Pro](https://github.com/MMMU-Benchmark/MMMU)：由 MMMU 增强 distractors，并有 Standard/Vision 模式；N 随官方 split manifest | 专家图像理解→多学科推理；`[课程合成]` 结合电路图和选项判断故障 | accuracy；Standard/Vision、validation/test 分栏 | DeepSeek Base `56.5`；B-vendor | 图片拼接/顺序、OCR、选项协议会变；不要与 Instruct 工具分数混表 |
| [CV-Bench](https://github.com/nyu-visionx/CV-Bench)：2,638 个 2D/3D spatial QA | 物体关系→深度/视角空间推理；`[课程合成]` 判断相机移动后两物体左右前后关系 | MCQ accuracy，2D/3D 分项 | DeepSeek Base `77.9`；B-vendor | 静态空间题不等于具身闭环；渲染/答案偏置可能泄漏 |
| [DocVQA](https://www.docvqa.org/datasets/docvqa)：约 50K questions、约 12K document images | OCR→布局→文档问答；`[课程合成]` 从发票表格找税前金额 | ANLS/accuracy，依 challenge track | DeepSeek Base `95.6`；B-vendor | 版本与 OCR pipeline 易饱和；短答案不测跨文档引用 |
| [RefCOCO family](https://github.com/lichengunc/refer)：RefCOCO/+/g 多 split；没有一个可脱离 split 的单一 N | 指代表达→目标框 grounding；`[课程合成]` 在多人图中定位“红帽右侧持杯者” | bbox IoU≥阈值的 accuracy，常报 split average | DeepSeek Base `avg 86.0`；B-vendor | family、split、detector proposal 与 image resize 必须锁定；不测动作成功 |

### 7.1 为什么这些项目应保留为“回归探针”而非发版主结论

它们便宜、可重复、能快速发现 tokenizer、chat template、量化或预训练回归；但公开时间长、任务短、工具和环境负担低。
因此正确用法是固定 prompt/decoding 做 checkpoint paired regression，再由本页前六节的长程、专业和可执行评测承担发布主张。

## 8. 稳定卡合并规则与已知冲突

### 8.1 应合并为稳定 lineage 卡

1. **MathArena family**：MathArena 是容器，ArXivMath/Apex/BrokenArxiv 是 competition；新增月份只写 score ledger，不另造定义卡。
2. **CyberGym family**：CyberGym（给漏洞描述→PoC）、CyberGym-E2E（发现→PoC→patch）、ExploitGym（crash→impact）共享基础设施但目标不同，应同章并列而非混成 `Cyber score`。
3. **APEX-Agents lineage**：480/240 任务、Pass@1/Mean Score 留在一张 lineage 卡，按版本分栏；不要为每次 roster 更新复制卡片。
4. **OfficeQA lineage**：Full/Pro/ProV2 与 PDF-image/extracted-text 是一个稳定卡的 axes。
5. **τ lineage**：文本 τ-bench/τ²/τ³ 与 τ-Voice 同属 lineage，但 voice 是独立协议分支；score ledger 分域、分模态。
6. **LatchBio family**：每个可验证子 benchmark 独立记分；“LatchBio aggregate”只能是带组件、权重和日期的视图，不应成为稳定 benchmark 定义。
7. **CAD family**：BenchCAD、CADGenBench、EEBench 可同属“engineering artifact”章节，但 verifier 和交付物不同，禁止求平均。

### 8.2 截至快照日必须保留的矛盾

- CritPt public 70-test ×5 与 Claude `CritPt-Corrected mean@16` 同名近似、实际不同题面/聚合。
- APEX 论文 `n=480` 与当前官方页 `240 tasks/31 worlds`；任何无版本 `APEX 41.0` 都不完整。
- PaperBench 完整复现的原始 `21.0%` 与 Qwen Code-Dev `93.0`；后者跳过独立执行与结果复现，虽然同用 rubric 分数也不是同一构念。
- CyberGym 论文最强约 20% 与四家发版 `78–88`；必须先找出任务设置差异。
- Chartography 论文统一协议约 45 与厂商 tools `78–86`；工具/judge 合同不同。
- WorldVQA 数据卡“无模型超过 50%”与 Kimi `51.0`；属于榜单日期或协议更新张力。
- EEBench 同一次 Grok 发布出现 `64.0` 与 `66.0`；两值都保留。
- GDP.pdf 的 `35.0 all-pass` 与 Claude `85.4` 未证明同 metric；OfficeQA 无 suffix 与 Pro 也不可混。

## 费曼自检

1. 为什么 Claude 的 `CritPt-Corrected 88.4` 不能说明它比 Kimi 的 public CritPt `23.4` 高 65pp？
2. WideSearch 的 item F1 84 与 Success Rate 5，哪一个更接近“交付了一张完整可用表”？为什么仍要同时报？
3. PaperBench Code-Dev `93.0` 为什么即使补齐预算、judge 和 revision，也不能与完整模式的原始 `21.0%` 做模型增益减法？
4. CyberGym、CyberGym-E2E、ExploitGym 分别把什么作为起点和终点？
5. APEX Mean Score 68.6 为什么不等于 68.6% 的任务完整完成？
6. Chartography tools 86.2 能否解释为视觉 encoder 比 no-tools 42.6 强一倍？
7. Speech-to-Speech Index 82.6 与 τ-Voice 68.6 为什么不能平均成 75.6？
8. SkillsBench 若只给 with-Skills 70.2，为什么还不能声称技能包带来 70.2 分收益？
9. DeepSeek Base 的 HumanEval 79.4、GSM8K 93.0 仍有什么现实用途？
10. 如果要把本页所有数字存入数据库，最小主键是什么？

<details>
<summary>参考答案</summary>

1. Corrected 修改了 31/71 题面，还改变工具和 `mean@16` 聚合；public 是 70 个测试挑战、每题 5 次平均。处理、样本和 estimand 都不同，差值不归因于模型。
2. SR 要整表每格全对，更接近完整交付；item F1 能定位“已收集多少且多准”，适合诊断局部进展。只报 SR 会在困难集接近零、失去诊断力，只报 F1 又会掩盖致命遗漏。
3. 因为处理本身不同：Code-Dev 只评价代码开发，完整模式还执行 submission 并核验实验结果。补齐 revision、harness、预算、trials、judge、失败分母和 artifact 后，可以分别复现两行，但仍不能把跨 task mode 的 72pp 归因于模型。
4. CyberGym 从漏洞文字描述+代码到触发 PoC；E2E 从仅源码到发现+PoC+patch；ExploitGym从已有 crash-triggering input 到 verifier 定义的安全影响。三者难度轴不同。
5. Mean Score 是所有 rubric criteria 的平均通过率；一题只要漏一个关键 criterion 仍可能有高 partial。Pass@1 才要求该题 100% rubric，但也需检查关键项权重和 judge。
6. 不能。crop、Python、OCR 与多轮视觉搜索共同改变了系统；它证明“模型×工具”的可用性提升，不能隔离 encoder 权重效应。
7. 82.6 已是四组件等权 composite，而 68.6 是其中一个 agentic component/独立任务集。再平均会对 τ-Voice 重复计权，也丢掉指数版本。
8. 收益是同任务同容器下 `with - without` 的配对差；70.2 是结果水平。还要确认 agent 实际读取 skill，并分离 harness 自带能力。
9. 它们适合低成本、低方差的持续回归：检查预训练、量化、tokenizer、chat template 或 serving 改动是否破坏基础算术/代码。它们不适合作为前沿能力唯一 headline。
10. `model/checkpoint + benchmark@revision/split + metric/aggregation + mode/effort + harness/prompt/tools + input representation + time/step/token/cost budget + sampling/trials + judge/verifier/environment + failure/refusal/fallback policy + evaluated_at + source owner`。

</details>

一句话验收：**新 benchmark 的价值在于暴露旧题看不到的失败模式；只有把版本、分母、工具与被测系统写全，数字才是证据而不是装饰。**
