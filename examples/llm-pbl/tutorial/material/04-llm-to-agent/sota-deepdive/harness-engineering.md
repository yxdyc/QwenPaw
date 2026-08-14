# SOTA Deep-Dive — Harness Engineering

> **深挖对象**：SOTA agent harness 的工程实践——上下文工程、状态外化与事务化、工具与技能设计、验证与评测、多角色编排。
> **状态**：✅ 首版完成（2026-08-11）
> **对照基础**：nano-agentscope L0–L3（25/25 满阶）+ nano-qwenpaw L0–L3（25/25 满阶）——本文所有「nano 实测」引用均指向这两个模块的可运行材料。
> **SOTA 对齐日期**：2026-08-11（ROADMAP §八）。全部一手来源当日现场重抓复验，见 §9 溯源表。

---

## §0 这篇文章是什么

Harness（agent 脚手架）指包住 LLM 调用的一切工程设施：system prompt、工具注册、记忆/上下文管理、自检与验证、终止条件、状态持久化。它不是「测试框架」义（见 shared/conventions.md 术语表）。

这篇 deepdive 回答三个问题：

1. **为什么 harness 是独立工程对象**——不是 prompt 技巧的集合，也不是框架选型问题（§1）；
2. **harness 的五个机制面**——上下文工程（§2）、状态外化与事务化（§3）、工具与技能（§4）、验证与评测（§5）、编排（§6），每一面都给一手来源 + nano 模块的可运行实证；
3. **2026 年的格局**——三层锚点定位 + 自动化 harness engineering 的兴起（§7）。

**复验记录（反幻觉口径）**：本文引用的 17 个 arXiv ID 于 2026-08-11 经 export.arxiv.org API 批量现场核验（标题/日期/摘要全部以当次抓取为准）；Anthropic 四篇一手文同日现场重抓全文。复验中确认一处 ID 误归属：`2506.07989` 实为物理论文（*Photon rings in a holographic toy model*，Detournay et al.），τ²-bench 的真 ID 是 `2506.07982`——本文只用后者。四类信息（原文声称 / 文献已有 / 合理推断 / 猜测）在正文中显式区分，推断标「推断」。

**阅读路径建议**：先跑 nano-qwenpaw L0（裸调用 vs harness 的最小差异），再读 §1；每节末尾的 nano 实证链接可以当场跑通。

---

## §1 harness 为什么是独立工程对象

### 1.1 失败模式定义了这个对象

Anthropic 确立该术语的一篇一手文《Effective harnesses for long-running agents》（2025-11-26，Justin Young）开宗明义：长时程 agent 任务的困难在于「each new session begins with no memory of what came before」（每个新会话都从零记忆开始）。该文没有给 harness 下形式化定义，而是把它具体化为 Claude Agent SDK 加上 prompts、文件、工具与 git 工作流的整体。文中记录的失败模式有两类：**试图一次做太多**（trying to do too much at once），以及**后续会话过早宣布完工**（later agents choose to "declare the job done" too early）。

注意这两个失败模式的归属：它们不是模型能力问题（模型在单会话内可以做得很好），而是**跨会话、跨步骤的工程结构问题**——记忆在哪里、终止条件由谁持有、进度如何被不可篡改地记录。这正是 harness 的研究对象。

同一来源给出的核心观察：Claude Agent SDK 自带 compaction（压缩），是一个「powerful, general-purpose agent harness」，但「compaction isn't sufficient」——压缩本身不足以支撑长时程任务。这句话的分量在 §2 展开。

### 1.2 harness 贵逾 20 倍，但买到了 solo 买不到的东西

《Harness design for long-running application development》（2026-03-24，Prithvi Rajasekaran，Anthropic Labs）给了罕见的完整成本账（原文声称，全部数字来自该文现场抓取）：

- 游戏制作对照：solo（单 agent 裸跑）= 20 分钟、$9；完整 harness（planner + builder + 独立 evaluator 三角色）= 6 小时、$200。「The harness was over 20x more expensive」——但 solo 没有产出可玩的核心，harness 产出了。
- 一次 DAW（数字音频工作站应用）构建全程 3 小时 50 分钟、$124.70，分项：planner 4.7 min/$0.46；build 1 2h7m/$71.08；QA 1 8.8 min/$3.24；build 2 1h2m/$36.89；QA 2 6.8 min/$3.09；build 3 10.9 min/$5.88；QA 3 9.6 min/$4.06。planner 的输入只有 1–4 句话，输出是 16 个 feature、10 个 sprint 的规格（契约粒度之细：仅 sprint 3 就有 27 条 criteria、覆盖 level editor，源文 "Sprint 3 alone had 27 criteria covering the level editor"）。

这组数字的本质（推断，但与数据一致）：**harness 是在用钱买「错误的可发现性」**。QA 轮只占总成本的约 8.3%（$10.39/$124.70 = 3.24+3.09+4.06，按上述分项求和计算），但没有它，build 1 的缺陷会静默流入 build 2。solo 便宜，是因为它把验证成本外部化给了人。

### 1.3 harness 不随模型变强而消失

同一篇 2026-03-24 文的两个判断值得逐字记录：

- 「Out of the box, Claude is a poor QA agent.」——开箱即用的模型做不好质检，QA 角色需要刻意调校。
- 「the space of interesting harness combinations doesn't shrink as models improve」——模型升级时，harness 组合的空间不缩小；正确的动作是升级后重新测试脚手架，拆掉不再「load-bearing」（承重）的部分、补上新能力。

这两句合起来划出了 harness engineering 的学科边界：它不是模型弱时的临时补丁，而是与模型能力**共同演化**的工程层。**推断**：这与训练侧的 data-model co-dev 同构——模型越强，「如何组织模型」的工程问题越重要而非越消失（ROADMAP §一 RSI 闭环的 04 轨视角）。

**nano 实证**：nano-qwenpaw L0 用同一个 mock LLM 对照「裸调用 vs 套上 system prompt + 输出自检的最小 harness」，行为差异完全来自 harness；L2/L3 进一步实测了方法论注入前后的行为差（见 §5.3）。跑一遍 `nano-qwenpaw/L0_harness_loop.py` 即可看到最小形态。

---

## §2 上下文工程：窗口是 cache，store 才是 memory

### 2.1 学科化：从 prompt design 到 context engineering

《A Survey of Context Engineering for Large Language Models》[2507.13334]（2025-07-17，Mei et al.）把 context engineering 形式化为一门学科：不止是 prompt 设计，而是「the systematic optimization of information payloads for LLMs」。其分类学（摘要声称）：基础组件 = 上下文检索与生成 / 上下文处理 / 上下文管理；系统集成形态 = RAG、记忆系统、tool-integrated reasoning、多 agent 系统。该文梳理了超过 1400 篇文献，并指出一个研究缺口：模型在**理解**复杂上下文上很强，在**生成**同等复杂的长输出上显著偏弱（asymmetry）。

### 2.2 OS 类比：虚拟上下文管理

MemGPT [2310.08560]（2023-10-12，Packer et al.）是「记忆系统」这条线的经典锚点（A 层）：受操作系统分层内存启发，提出 **virtual context management**——在有限上下文窗口内管理多层记忆，快慢内存之间搬移数据，并用 **interrupts** 管理模型与用户之间的控制流。评估域：超出窗口的文档分析 + 多会话聊天。

这个 OS 类比的本质（推断）：上下文窗口不是「模型的内存」，而是**缓存**——它小、贵、易失；真正的记忆在外部存储，窗口管理策略决定什么时刻把什么搬进缓存。

**nano 实证**：nano-qwenpaw L1 在同一真实语料、同一预算下实测三种窗口政策的损失谱——append-only 丢 50/60%（静默、按新近度单调丢头部）；summarize 丢 20/20%（损失不可预测：salience=词元稀有度，与任务重要性无关，实测 445←1282 字符不可逆压缩）；write-through+evict-index 100/100%（ctx 999≤1000 预算有界，store 24==24 不变量，FTS5 召回 byte-identical）。结论句「窗口是 cache，store 才是 memory」即出自该节：记忆可靠性是**存储设计**问题，不是模型属性。

### 2.3 compaction 不够，那够的是什么

Anthropic 两篇一手文给出的答案拼起来是：

- 2025-11-26 文：「compaction isn't sufficient」；解法见 §3（状态外化到文件/git）。
- 2026-03-24 文：记录了一个真实现象「context anxiety」（模型在上下文将满时行为退化——提前收尾、草率提交），并给出分级对策：出现 context anxiety 时用 **context resets + 「structured handoff」**（硬重置 = 清空上下文窗口、另起一个新 agent，并配一份携带前一 agent 状态与后续步骤的结构化交接，源文原词 "a structured handoff that carries the previous agent's state and the next steps"）；换更强的模型后，「automatic compaction」可能就够了。

**推断**：这两条合起来说明压缩策略的选择不是固定的最佳实践，而是**(模型能力 × 任务长度) 的函数**——与 §1.3「harness 随模型共同演化」是同一件事在上下文面的投影。

### 2.4 记忆的主动组织：A-MEM

A-MEM [2502.12110]（2025-02-17，Xu et al.，v11 2025-10-08）代表记忆系统的另一个方向：不止存取，还要**动态组织**——按 Zettelkasten 方法为每条新记忆生成带 contextual descriptions/keywords/tags 的笔记，分析历史记忆建立链接，且新记忆可以**反向更新旧记忆的表示**（memory evolution）。摘要声称在六个基础模型上优于现有 SOTA 基线。

**时效性定位**（§八 三层锚点）：A-MEM 目前按 C 层（单论文方法）对待——本文只教「记忆需要主动组织与演化」这一**机制类别**，不押注该具体方法；其晋升 B 层需要独立验证（前沿产品采用/权威框架集成）。MemGPT 的 OS 类比是 A 层锚点，nano-qwenpaw L1 的三政策实测是可运行锚点。

---

## §3 状态外化与事务化：让「中断」不再致命

这是 04 轨 placeholder 原定 scope 的「可靠性/事务化」面，也是长时程 agent 与一次性 chatbot 的分水岭。

### 3.1 Anthropic 的配方：文件系统 + git = 检查点

2025-11-26 文的初始设置配方（原文声称）：initializer agent 首次运行时创建 `init.sh`、`claude-progress.txt`、「an initial git commit」（初始 git commit，标示新增了哪些文件）、一份结构化需求文件；任务拆成 JSON feature checklist（该文实例「over 200 features」，初始全标 failing），后续会话的 coding agent 只能更新 `passes` 字段——「It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality」。每个新会话的 orientation 流程：读 git logs 和 progress 文件 → 挑一个未完成项 → 一次只做一个 feature → 完成后 git commit（descriptive message）+ 更新进度笔记 → 像真实用户一样验证（「use browser automation tools and do all testing as a human user would」，该文用 Puppeteer MCP 截图抓到了常规测试漏掉的 bug）→ 仓库留在 clean state。

拆出来的机制清单（推断，逐条可对应到事务语义）：

| 配方成分 | 事务语义 | 防的失败模式 |
|----------|----------|--------------|
| feature checklist 只增不改判据 | 验收标准是 schema，不是数据 | 过早宣布完工（改判据=改目标） |
| 一次一个 feature + commit | 原子提交点 | 一次做太多、中断后无法定位 |
| progress.txt + git log | 持久化的会话外记忆 | 「new session begins with no memory」 |
| 像用户一样验证 | 独立证据通道 | 自我验证的相关盲点（见 §5.3） |

2026-03-24 文把同一思想升级成角色间契约：planner 产出 spec，builder 与 evaluator 之间用协商出的「sprint contract」文档界定验收；状态外化到 files、「structured artifacts」、git，而不是依赖 chat history。sprint 边界、契约、evaluator 阈值、版本历史共同构成**可审查的停止点**（推断概括，非源文原词——源文分别给出这四个要素，合成命名是本文的机制提炼）。

### 3.2 终止即数据

nano 材料把「终止条件住在哪」做成了可运行实验：

- nano-agentscope L2：五种终止状态全部带日志返回；消息契约在边界强制校验（六种消息类型、类型化违规五种）。实测对照：活锁守卫 6 次 attempt 就能精确诊断，而预算保险丝要烧到 24 次才断且误标——**终止条件的信息质量决定调试成本**。
- nano-agentscope L3（对照 AgentScope v2.0.6 源码）：`ReplyFinishedReason` 是类型化的数据（`types/_reply.py:L10-16`，行号以 2026-08-10 抓取日为准），终止原因**住在消息上**而不是进程状态里——`exceed_max_iters` 是消息的属性，任何接收方都能读到。对照 v1.0.0：终止只是函数返回，第三方无从知晓。

**事务化小结**（推断）：commit/rollback 引入 agent 动作的本质不是「保存进度」，而是把**状态变更的可见性与可裁决性**从模型内部搬到外部介质——检查点让中断可恢复，不可篡改的判据让「宣布完工」成为一个可被第三方验证的事务，而不是模型的自信声明。这与 §1.1 的失败模式（过早宣布完工）严丝合缝：那个失败模式之所以是 harness 问题，正因为终止的裁决权应该在数据里，不在模型的置信度里。

---

## §4 工具设计与 skill-as-data：能力在接口里，不在模型里

### 4.1 工具使用的机制谱系（A 层锚点）

- **Toolformer** [2302.04761]（2023-02-09，Schick et al.）：模型**自监督**学会决定调哪个 API、何时调、传什么参数、如何把结果并入后续预测——每个 API 只需少量示范。机制贡献：工具调用是可以训练进模型的行为，不是外部强加的格式。
- **ReAct** [2210.03629]（2022-10-06，Yao et al.）：推理轨迹与动作交错生成——推理帮助规划/跟踪/更新计划与处理异常，动作让模型接触外部信息源。在 ALFWorld/WebShop 上以 1–2 个 in-context 示例超过模仿与 RL 基线（绝对成功率 +34%/+10%，摘要声称）。机制贡献：observe 步骤让错误可被模型自己看见。
- **ToolLLM** [2307.16789]（2023-07-31，Qin et al.）：16,464 个真实 RESTful API（RapidAPI，49 类）+ 自动构造的指令数据 + DFSDT 搜索 + ToolEval 自动评测。机制贡献：工具规模上来后，**API 检索与搜索策略**成为瓶颈，不只是「会调工具」。

### 4.2 ACI：接口设计是一等设计对象

SWE-agent [2405.15793]（2024-05-06，Yang et al.）的论点是 harness engineering 在工具面的基石（原文声称）：LM agent 是**新的一类终端用户**，有自己的需求与能力，需要为它们专门构建接口——agent-computer interface（ACI）。该文实测接口设计直接影响 agent 表现：在 SWE-bench 上 pass@1 12.5%、HumanEvalFix 87.7%（发表时 SOTA）。

对照时间线上的两个数字可以看 harness 的份量（两个数字均为各自论文摘要声称，非同一实验的受控对照，归因须谨慎）：SWE-bench 发布时（2023-10）最强模型 Claude 2 解出 1.96% [2310.06770]；七个月后 SWE-agent（专用 ACI + 当时的前沿模型）报 12.5% [2405.15793]。模型在进步，但**接口形态的改变是独立变量**——SWE-agent 论文的核心主张正是接口设计本身在起作用（「provide insight on how the design of the ACI can impact agents' behavior and performance」）。

### 4.3 Agent Skills：技能是可加载的数据

Anthropic《Equipping agents for the real world with Agent Skills》（2025-10-16；2025-12-18 更新为跨平台开放标准；Barry Zhang, Keith Lazuka, Mahesh Murag）定义了 skill 的形态（原文声称）：

- 「a skill is a directory that contains a `SKILL.md` file」；SKILL.md 必须以 YAML frontmatter 开头，含必需的 `name` 与 `description`。
- 加载机制 = **progressive disclosure**（渐进披露，原文称其为核心设计原则）：启动时只把每个 skill 的 name/description 预载进 system prompt；模型判断当前任务相关时，才把完整 SKILL.md 读进上下文；skill 目录可捆绑更多文件按名引用，「the amount of context that can be bundled into a skill is effectively unbounded」。
- skill 可以携带可执行代码（「Skills can also include code for Claude to execute as tools」）。
- 实践建议：从评测出发找能力缺口（「Start with evaluation」）；内容大了就拆文件引用（「Structure for scale」）；name/description 是模型看见的第一眼（「Pay special attention to the name and description」）；让 Claude 自己沉淀成功做法与常犯错误（「ask Claude to capture its successful approaches and common mistakes」）。安全面：只从可信来源安装 skill。

2025-12-18 的开放标准化是格局事件（推断）：skill 从单一产品的特性变成**跨平台可移植的工件格式**——这意味着「方法论即数据」的资产可以跨 harness 积累，与模型解耦。

### 4.4 nano 实证：skill 的五个实测性质

nano-qwenpaw L3（对照本仓库 qwenpaw coach 的 builder/registry/store 源码，行号锚 live 推导）实测了 skill-as-data 的五个性质，全部可运行复现：

1. **directory ≠ skill**：enablement 住在 manifest 里（真实 profile 实测 9 个目录 vs 8 个 enabled，codex-delegate 目录存在但从未生效）——目录扫描天然是 opt-out，manifest 天然是 opt-in，这是安全姿态的差别。
2. **SKILL.md 是准入门**：effective ≠ injectable；builder 记录原样日志（onboard 埋点实测被门拦下）。
3. **channel 过滤 reach**：同一个 workspace 按请求组装出两个 agent（console 3 skills vs voice 3 skills，差集恰为 feynman-check/voice-brief）。
4. **skills ride the prompt，不在 tools=**：技能注入走上下文而非工具注册（prompt delta 1145 = 1291 − 146 est-tokens 实测）——文本性能力与功能性能力是两条通道。
5. **capability 在注入文档，不在 harness**：同一份解释，在有验证文档的 console 席被验证（mastery +0.00），在缺文档的 voice 席被放行（+0.05）——行为差完全归因缺失的文档，mastery 通胀也是渠道问题（10 轮投影 0.75 vs 1.25）。

nano-agentscope L3 从工具侧补了一面：对照 AgentScope v2.0.6，`ToolCallBlock` 是带状态机的内容块（`message/_block.py:L138`，行号以 2026-08-10 抓取日为准），**agent 不能自推工具状态**——状态迁移由 harness 执行。这与 skill 的 enablement 同构：能力与状态的控制权都在 harness 侧，模型只负责提议。

---

## §5 验证与评测：harness 自身的质量问题

### 5.1 自评不可信：把做事的和判事的分开

2026-03-24 文的两个实测观察：自我审查时「agents tend to respond by confidently praising the work」（自信地夸自己的工作）；而「Separating the agent doing the work from the agent judging it」之后反馈才变得有用。该文的 evaluator 是活的：「the evaluator would navigate the page on its own」（用 Playwright MCP 自己操作页面），按 rubric 打分——rubric 不问「美不美」，问「does this follow our principles for good design?」，并对 design quality 与 originality 加权；用 few-shot examples 校准评分尺度，设硬性 pass/fail 线。前端循环每轮生成跑 5–15 次迭代，全程可 stretched 到 4 小时。

Agent-as-a-Judge [2410.10934]（2024-10-14，Zhuge et al.）把同一思想系统化：用 agent 系统评测 agent 系统——相比 LLM-as-a-Judge 的关键增量是**对全过程提供中间反馈**（不止看最终结果）。配套 DevAI 基准：55 个真实 AI 开发任务、365 条分层人工标注需求；摘要声称其可靠性与人工评测基线相当、显著优于 LLM-as-a-Judge，并视其为 agent 自我改进所需的 reward signal 来源。

**nano 实证**：nano-qwenpaw L2 实测了相关盲点的机制——「自己重读一遍自己的输出」只抓出 4 个隐患中的 1 个，而换独立证据通道抓出 4/4：**验证效力来自证据通道的异质性**（同一套计算复现同一个错误）。这与「worker/judge 分离」是同一原理的两种形态：角色分离是组织手段，通道异质是机制本质。

### 5.2 评测基准谱系：从结果评测到轨迹与可靠性评测

以下全部数字来自各论文摘要（2026-08-11 arXiv API 现场核验），按「评测什么」排列：

| 基准 | arXiv | 日期 | 评测对象 | 摘要关键数字 |
|------|-------|------|----------|--------------|
| SWE-bench | 2310.06770 | 2023-10 | 真实 GitHub issue 修复（12 个 Python 仓库、2,294 题） | 当时最强 Claude 2 解 1.96% |
| GAIA | 2311.12983 | 2023-11 | 通用助手（推理+多模态+浏览+工具）466 题 | 人类 92% vs GPT-4+plugins 15% |
| OSWorld | 2404.07972 | 2024-04 | 真实计算机环境（Ubuntu/Windows/macOS）369 任务 | 人类 72.36% vs 最强模型 12.24%（卡在 GUI grounding 与操作知识） |
| τ-bench | 2406.12045 | 2024-06 | 工具-agent-**用户**三方交互 + 领域规则遵从；数据库终态比对 | gpt-4o 级 agent 成功率 <50%；**pass^8 < 25%（retail）** |
| Agent-as-a-Judge/DevAI | 2410.10934 | 2024-10 | 轨迹级评测（中间反馈）55 任务/365 需求 | 与人工评测基线相当 |
| τ²-bench | 2506.07982 | 2025-06 | **双控环境**（agent 与用户都有工具、共享世界，Telecom 域建模为 Dec-POMDP） | 从 no-user 到 dual-control「显著性能下降」 |
| Terminal-Bench 2.0 | 2601.11868 | 2026-01 | 命令行硬核真实任务 89 个（每题人工写解 + 完备测试） | 前沿模型与 agent <65% |

三个机制性观察：

1. **pass^k 是可靠性度量，不是能力度量**（τ-bench 原文提出）：同一 agent 多试 k 次全过的概率。pass^8 < 25% 意味着单次成功 <50% 的系统在重复使用时一致性崩塌——**可靠性代数不是线性的**（nano-agentscope L1 实测：相关/sticky 失败下重试算术失效，iid 公式给出乐观上界）。
2. **评测在往「交互与过程」走**：τ-bench（用户模拟）→ τ²-bench（双控：双方都能改世界状态，考的是协调与引导）→ Agent-as-a-Judge（轨迹级）。τ²-bench 摘要声称从 no-user 到 dual-control 性能显著下降——「引导用户行动」要求的能力与「自己行动」不同（推断：它要求模型对自己的动作与对方的动作做联合规划，而多数 harness 只建模前者）。
3. **分数是 (模型 × harness × 环境) 三元组的属性**：OSWorld 的 12.24% 不是模型的数字，是当时最强 agent 系统在该环境下的数字；换 harness 换环境都变。下一节把这个观察量化。

### 5.3 基础设施噪声：分数里有多少是 VM 的贡献

《Quantifying infrastructure noise in agentic coding evals》（2026-02-05，Gian Segato；URL slug 为 `infrastructure-noise`，标题与 slug 不同形）是评测方法论的一手文，全部数字原文声称（Terminal-Bench 2.0 实验）：

- 「Two agents with different resource budgets and time limits aren't taking the same test.」——资源预算不同的 agent 考的不是同一张卷子。
- 最富与最穷配置的差距：**6 个百分点（p < 0.01）**；「as many as 6% of tasks were failing because of pod errors」——多达 6% 的任务因 pod 错误而失败，与模型能力无关。
- 根因之一：把请求资源同时设为最低保证与 kill 阈值——「zero headroom for transient spikes」。 infra 错误率：严格限制 5.8% → 3x headroom 2.1%（p < 0.001）→ 不封顶 0.5%，单调下降。
- 关键分段：**1x 到 3x，分数在噪声内波动（p=0.40）**——headroom 只是消除假失败；**约 3x 起趋势改变**：「success rates climb faster than infra errors decline」，额外资源开始让 agent 解出原本解不出的题——「limits can actually change what the eval measures」。不封顶相对 1x 总提升 +6pp（p < 0.01）。
- SWE-bench 同方向但幅度小：5x RAM 比 1x 高 1.54pp。
- 其他混杂因子：时间上限、集群状态、硬件、并发、带宽、延迟——「pass rates fluctuate with time of day, likely because API latency varies with traffic patterns and incidents」。
- 方法论建议：每任务给 floor + ceiling 两个参数而非单点值（「The band between them should be calibrated so that scores at the floor and ceiling fall within noise of each other」）；报告倍数与校准方法；跨时段跨天平均；**「leaderboard differences below 3 percentage points deserve skepticism until the eval configuration is documented and matched」**；收尾句：「A few-point lead might signal a real capability gap—or it might just be a bigger VM.」

**nano 实证**：nano-agentscope L1 的可靠性代数（iid 公式 vs sticky 失效实测）与 L2 的「活锁守卫 6 vs 预算保险丝 24」对照，是同一主题在玩具尺度的可运行版本：评测与守卫的**阈值设计**必须基于失败结构的实测，而不是想当然的均匀假设。infra noise 一文则把同样的教训放大到基础设施尺度：方差本身有结构（假失败段 vs 真能力段），混在一起报分数就是伪精度。

---

## §6 编排：从单循环到多角色

### 6.1 机制谱系

- **ReAct** [2210.03629]：单 agent 的 reasoning-acting 交错循环（A 层锚点；nano-agentscope L0 即其可运行最小形态）。
- **Plan-and-Solve** [2305.04091]（2023-05-06，Wang et al.）：先 devising plan 把任务拆成子任务、再按 plan 执行——zero-shot CoT 的三大坑（计算错误/漏步/语义误解）里专治漏步。这是 planner 角色的提示词级原型。
- **AgentScope** [2402.14034]（2024-02-21，Gao et al.）：以消息交换为核心通信机制的多 agent 平台，内置与可定制容错，actor-based 分布式框架（本地/分布式部署无感切换）。

### 6.2 Anthropic 两篇一手文的编排形态

- 2025-11-26 文：**initializer agent + coding agent**——注意两者的差别只是「different initial user prompts」，不是两套工具。该文同时把多 agent 架构（testing agent / QA agent / code cleanup agent）列为**开放问题**而非推荐——单通用 agent 是否最优没有定论。
- 2026-03-24 文：**planner + builder + 独立 evaluator** 三角色实证落地，planner 把 1–4 句话变成 16 features/10 sprints 的规格。

两篇合起来的口径（推断）：**角色分离首先发生在职责上（规划/执行/裁决），是否物化成多个 agent 是次要的工程选择**。nano-agentscope L3 的实测支持这一口径：广播（MsgHub）让 verifier 以 0 次额外发送获得全屋知识（p2p 需 4 次转发）——**认知状态由接线决定**；而「提前完成（无证据宣称完工）」这种失败只有第三角色抓得住（L2 的两方在协议层接受同一向量）。verifier 的价值不在它是独立进程，而在它的信息位置与利益位置。

### 6.3 契约类型化

nano-agentscope L2→L3 的阶梯增量正是编排面的工程进化（对照 AgentScope v1.0.0 → v2.0.6 双快照实测）：校验从「门口拦截」（运行时拒绝非法消息）前移到「出生拦截」（构造期校验，角色×块类型合法表把自我授权/伪造证据挡在构造器外）；v2.0.6 core 整体移除了 pipeline/msghub 而强化 Msg 契约——**上游的演化方向是「契约变厚、编排变薄」**（全树 grep 实证类名消失，见 nano-agentscope tutorial_L2/L3 §6 锚点记录）。协议级重试代数实测：k=0 时复合可靠性公式成立，k≥1 时 iid 只是上界（54.5→64.5→66.0%）——重试的收益随协议强度饱和。

---

## §7 2026 格局：三层锚点与自动化趋势

### 7.1 三层锚点定位（ROADMAP §八，对齐日 2026-08-11）

| 层 | 本主题的条目 | 处理 |
|----|--------------|------|
| **A 经典锚点** | ReAct [2210.03629]、Toolformer [2302.04761]、MemGPT [2310.08560]、SWE-bench [2310.06770]、GAIA [2311.12983]、τ-bench [2406.12045]、AgentScope [2402.14034] | 机制仍是地基，教本质；当今定位已在各节说明（如 ReAct 循环已内化进所有主流 harness） |
| **B 前沿主流** | Anthropic 四篇一手文（2025-10/11、2026-02/03）；context engineering 综述 [2507.13334]；τ²-bench [2506.07982]；Terminal-Bench 2.0 [2601.11868]；Agent Skills 开放标准（2025-12-18） | 多独立来源支撑（Anthropic 连续四篇 + 基准被前沿评测采用）；本文主体内容 |
| **C 中间状态** | A-MEM [2502.12110]（单论文记忆方法）；The Last Harness You'll Ever Build [2604.21003]（单论文自动化框架，见 §7.2） | 只讲机制类别（记忆的主动组织 / harness 工程的自动化），不押注具体方法；晋升 B 层需独立验证 |

### 7.2 自动化 harness engineering：对象本身在变

The Last Harness You'll Ever Build [2604.21003]（2026-04-22，Seong et al.，v3 2026-05-01）的摘要声称（本文只引摘要层，正文细节未读，标 [TODO: verify]）：每个新任务域都需要「painstaking, expert-driven harness engineering」（设计 prompt、工具、编排逻辑、评测标准）；该文提出两层框架——**Harness Evolution Loop**（Worker Agent 执行、Evaluator Agent 对抗性诊断失败并打分、Evolution Agent 基于全部历史尝试修改 harness）+ **Meta-Evolution Loop**（跨任务优化 evolution blueprint Λ=(W_H, H^(0), V, E) 本身），并形式化与 meta-learning 的对应。

这篇论文值得注意的不是其具体方法（C 层，未独立验证），而是它把 §5.1 的 worker/judge 分离 + §1.3 的「harness 随模型演化」推到了逻辑终点：**如果 harness 是工程对象，它就可以被工程化地生产**。对抗性 evaluator 在其中的角色与 nano-qwenpaw L2 的 claims gate 同构——验证者查的是 provenance（这次修改相对历史证据是否成立），不是 truth。

**时效性判断**（对齐日口径）：截至 2026-08-11，harness engineering 的「手工学科」形态（Anthropic 四篇所代表）是 B 层主流；「自动化学科」是刚出现的 C 层方向。两者不互斥——自动化框架的 evaluator 仍需要人定义「什么叫好」（2026-03-24 文的 rubric 正是人工产物）。[TODO: verify] 自动化方向是否有 ≥2 个独立机构的后续工作。

### 7.3 上游在动：锚点会漂移

nano-agentscope 阶梯的锚点记录了一个实证：AgentScope main 分支从 L1 写作（2026-08-06）到 L2 写作（2026-08-10）跃迁至 v2.0.6，行号锚漂移（`_agent.py` L858-874/L3019 → L863-869/L3050），v2 core 整体移除 pipeline/msghub。**harness 框架本身处于快速演化期**——这是本 deepdive 一切行号锚都带「以抓取日为准」声明的原因，也是 §八 季度再校准机制在本主题的具体形态。

---

## §8 费曼自检

### 8.1 讲给外行听

把 LLM 想成一个手艺极好但**没有长期记忆、不知道自己什么时候做错了**的工匠。harness 就是给他搭的工地：脚手架（上下文窗口管理）决定他此刻站得多稳；墙上的施工台账和已验收的房间（git + progress 文件 + feature checklist）让他明天来了知道干到哪、什么算干完；工具箱（tools/skills）按活计配发，说明书（SKILL.md）随箱附带；质检员（独立 evaluator）不是他自己，拿着验收标准（rubric/契约）一间一间查；工地规章（评测基准 + 基础设施规范）保证「今天量出的尺寸」和「明天量出的尺寸」可比。没有工地，工匠只能盖一间随时会塌的样板房；有了工地，他盖楼的速度仍然取决于手艺——**harness 不提高手艺，它让手艺可以累积**。

### 8.2 思考题

1. 「过早宣布完工」为什么是 harness 问题而不是模型问题？（提示：终止条件的裁决权住在哪——nano-agentscope L2 的五种终止状态、L3 的 `ReplyFinishedReason` 住在消息上；§3.2）
2. infra noise 一文建议 floor+ceiling 而非单点值。如果你设计一个 agent 评测平台，band 的校准标准是什么？为什么不能用「分数最大化」来选 ceiling？（提示：§5.3 的 3x 分段——ceiling 越过能力拐点后，评测测的不再是同一个东西）
3. skills ride the prompt 不在 tools=（nano-qwenpaw L3 实测）。那么 skill 注入的「能力」与 SFT 注入的「能力」，在生效条件、可追溯性、更新成本上有何异同？（提示：provenance 门禁、manifest opt-in、开放标准的可移植性）
4. 如果 harness engineering 被两层进化循环自动化（[2604.21003] 摘要声称），人类 harness 工程师的哪部分工作最先被自动化、哪部分最后？（提示：对抗 evaluator 的评分标准本身是谁写的——2026-03-24 文的 rubric 与「design principles」）
5. τ²-bench 的双控设置让性能显著下降。引导一个**也会动手**的用户，比自己动手多要求了什么？这对 harness 的消息契约设计意味着什么？（提示：联合规划 vs 单方规划；nano-agentscope L3 广播 vs p2p 的认知状态差别）

### 8.3 反例（流行但错的说法）

1. **「压缩（compaction）足以支撑长时程任务」**——Anthropic 2025-11-26 原文明确「compaction isn't sufficient」；nano-qwenpaw L1 实测 summarize 政策的损失不可预测（salience≠importance）。压缩是缓存策略，不是记忆系统。
2. **「让 agent 自己检查一遍输出就能保证质量」**——2026-03-24 文：自评时「confidently praising the work」；nano-qwenpaw L2 实测同通道重读只抓 1/4。验证效力来自证据通道异质性，角色分离是其组织形式。
3. **「排行榜上高 2 个点 = 更强的 agent」**——infra noise 一文：<3pp 的差距在评测配置未文档化、未对齐之前「deserve skepticism」；6pp 的差距可以纯粹来自 VM。
4. **「模型越强，harness 越可以省」**——2026-03-24 文：harness 组合空间不随模型进步缩小；且 ≥3x 资源段「limits can actually change what the eval measures」——环境参数都在改变有效能力，何况 harness。正确的动作是模型升级后**重测并拆掉不再承重的部分**，而不是整体拆除。
5. **「benchmark 分数是模型的属性」**——OSWorld 12.24%、SWE-bench 1.96%→12.5%、Terminal-Bench <65% 都是 (模型 × harness × 环境) 三元组的数字。把分数从三元组上剥下来归给模型，是 agent 评测最常见的归因错误。

### 8.4 局限

- 本文的 Anthropic 一手文均出自单一厂商视角，其配方（initializer/coding agent、sprint contract、rubric）在该厂商模型上验证；跨模型迁移性未独立验证 [TODO: verify]。
- [2604.21003] 只核验到摘要层（标题/日期/摘要为 2026-08-11 arXiv API 现场抓取），正文实验未读，正文声称标 [TODO: verify]。
- 四篇 Anthropic 文章的具体内部实验设置（除其自报数字外）无法独立复算，本文一律标「原文声称」。
- 行号类锚点（AgentScope/qwenpaw 源码）均带抓取日，漂移可检测但需要按季度再校准（§7.3 实证了漂移真实发生）。

---

## §9 溯源与口径

### 9.1 一手来源清单（全部 2026-08-11 现场重抓复验）

**Anthropic engineering（四篇，URL 均为 www.anthropic.com/engineering/ 下）**：

| 文章（页面实际标题） | slug | 发布/更新 | 作者 |
|----------------------|------|-----------|------|
| Effective harnesses for long-running agents | `effective-harnesses-for-long-running-agents` | 2025-11-26 | Justin Young |
| Equipping agents for the real world with Agent Skills | `equipping-agents-for-the-real-world-with-agent-skills` | 2025-10-16；2025-12-18 更新（开放标准） | Barry Zhang, Keith Lazuka, Mahesh Murag |
| Quantifying infrastructure noise in agentic coding evals | `infrastructure-noise` | 2026-02-05 | Gian Segato |
| Harness design for long-running application development | `harness-design-long-running-apps` | 2026-03-24 | Prithvi Rajasekaran（Labs） |

注：两篇的页面标题与 slug 不同形（infra noise / harness design），引用时以页面标题为准。

**arXiv（17 个 ID，2026-08-11 export.arxiv.org API 批量核验标题/日期/摘要）**：

| arXiv ID | 标题 | 首发日期 | 本文引用位置 |
|----------|------|----------|--------------|
| 2210.03629 | ReAct: Synergizing Reasoning and Acting in Language Models | 2022-10-06 | §4.1, §6.1 |
| 2302.04761 | Toolformer: Language Models Can Teach Themselves to Use Tools | 2023-02-09 | §4.1 |
| 2305.04091 | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | 2023-05-06 | §6.1 |
| 2307.16789 | ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs | 2023-07-31 | §4.1 |
| 2310.06770 | SWE-bench: Can Language Models Resolve Real-World GitHub Issues? | 2023-10-10 | §4.2, §5.2 |
| 2310.08560 | MemGPT: Towards LLMs as Operating Systems | 2023-10-12 | §2.2 |
| 2311.12983 | GAIA: a benchmark for General AI Assistants | 2023-11-21 | §5.2 |
| 2402.14034 | AgentScope: A Flexible yet Robust Multi-Agent Platform | 2024-02-21 | §6.1 |
| 2404.07972 | OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments | 2024-04-11 | §5.2 |
| 2405.15793 | SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering | 2024-05-06 | §4.2 |
| 2406.12045 | τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains | 2024-06-17 | §5.2 |
| 2410.10934 | Agent-as-a-Judge: Evaluate Agents with Agents | 2024-10-14 | §5.1, §5.2 |
| 2502.12110 | A-MEM: Agentic Memory for LLM Agents | 2025-02-17（v11 2025-10-08） | §2.4 |
| 2506.07982 | τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment | 2025-06-09 | §5.2 |
| 2507.13334 | A Survey of Context Engineering for Large Language Models | 2025-07-17 | §2.1 |
| 2601.11868 | Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces（摘要内自称 Terminal-Bench 2.0） | 2026-01-17 | §5.2, §5.3 |
| 2604.21003 | The Last Harness You'll Ever Build | 2026-04-22（v3 2026-05-01） | §7.2（摘要层） |

**负对照**：`2506.07989` = *Photon rings in a holographic toy model*（Stéphane Detournay et al.，2025-06-09）——物理论文，**不是** τ²-bench。04:30 轮研究转录曾误归属，本轮复验纠正并全程只用 `2506.07982`。

### 9.2 内部对照材料（本仓库可运行锚点）

- nano-qwenpaw L0–L3：`../nano-qwenpaw/`（harness 最小形态 / 记忆三政策实测 / 方法论注入与相关盲点 / skill-as-data 五性质；L3 输出锚 md5 `2c6780dc…`，见其 README 锚点表）
- nano-agentscope L0–L3：`../nano-agentscope/`（ReAct 循环 / 可靠性代数 / 消息契约与终止即数据 / 类型化契约与广播；L2 输出锚 `997344ec…`，见其 README）
- AgentScope 源码行号锚：以 2026-08-10 codeload tarball 抓取日为准（v2.0.6 + v1.0.0 双快照，详见 nano-agentscope README「权威实现与延伸」节）
- qwenpaw coach/arch 源码行号锚：以 2026-08-11 live 推导为准（sha256[:8] 与行号表见 nano-qwenpaw README）

### 9.3 口径声明

- 四类信息区分：「原文声称」= 一手文/摘要原文；「文献已有」= 已发表结论；「推断」= 本文作者的机制推断（已标明）；无「猜测」级内容入正文。
- 所有数字均可回溯到 §9.1 清单的当日抓取；无裸数字。
- [TODO: verify] 遗留三项：Anthropic 配方的跨模型迁移性；[2604.21003] 正文层；自动化 harness 方向的独立后续工作数。
- 本文不引入任何 C 层单论文方法作为教学内容（§八），A-MEM 与 [2604.21003] 只作为机制类别的载体出现。
