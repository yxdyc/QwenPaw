# 分册 03：Agent、工具与职业工作 benchmark

> **核心问题**：当模型能浏览、写文件、操作桌面和调用几十个 SaaS 工具时，分数究竟属于模型，还是整套 agent system？
> **范围**：检索、工具调用、业务状态、GUI/computer use 与职业 artifact；动态厂商水位见 [发布证据账](RELEASE_LEDGER.md)。
> **快照**：2026-09-22；“当前/截至快照”的水位只对该日期成立。

## 1. 能力阶梯：从找答案到承担结果

```text
给定工具找一个事实
  → 多跳浏览并综合证据
  → 在一个 API/网站中完成有状态任务
  → 跨多个应用执行完整业务流程
  → 在桌面/终端中长程观察—行动—恢复
  → 产出可被专业人士验收的多文件 artifact
```

任务越往后，底座模型越不是唯一被测对象。prompt、tool schema、浏览器/OS 镜像、context compaction、最大步数、
并发、重试、时间/费用、safeguard、fallback 和 verifier 共同定义 score。

## 2. 事实检索与 deep research

### 2.1 GAIA：466 道通用助手题的经典基线

- **基本信息**：[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) 有 466 题：166 道公开 validation、300 道隐藏答案 test，分三个难度等级；输入可含 PDF、图像、音频或网页。
- **能力项 / 阶梯**：L2/L3 浏览、文件解析、计算与多步工具推理。
- **课程合成例题**：从 PDF 脚注找到一个年份，再到网页查同期汇率，换算后只输出最终金额。
- **打分**：规范化后的短答案 exact/equivalence accuracy，按 level 与 overall 报告。
- **厂商采用**：历代通用 agent 常用；2026 头部发版已更多迁到 BrowseComp、ALE 和职业工作集。
- **局限**：网页漂移、账号/工具差异与公开 validation 污染；最终答案正确不证明引用完整、过程安全或低成本。
- **最高水位口径**：固定 validation/test、浏览器、search budget、grader 与日期；历史 GAIA 分不代表当前 live web 能力。

### 2.2 BrowseComp：1,266 道“搜不到第一屏答案”的短答案题

- **基本信息**：[OpenAI BrowseComp](https://openai.com/index/browsecomp/) 有 1,266 道人工构造的困难检索题，答案通常很短，但需要组合多个罕见线索。
- **能力项 / 阶梯**：L3 query reformulation、多跳浏览、消歧与证据综合。
- **课程合成例题**：根据会议年份、作者教育经历和一项实验细节，定位唯一论文标题。
- **打分**：短答案正确率，官方 simple-evals 用 judge 判断等价；single run、majority、best-of-N 必须分栏。
- **厂商采用**：GPT-6 Astra、Step 5、Kimi K3 等最新发版；Hy4 另有内部 BrowseComp-Pro2。
- **局限**：网页会变化；搜索 API、抓取器、上下文压缩和 token budget 可主导结果；短答案不等于可靠研究报告。
- **最高水位**：Astra 厂商表 `91.5`、Kimi `91.2`、Step `88.7`，但 harness/预算不同，只作 B-vendor 候选。

### 2.3 DRACO / deep-research rubric：从答案命中到研究过程

- **基本信息**：DRACO 以约 100 个 deep-research query 和 rubric/judge 评估搜索、证据与综合；版本与 judge 需跟发布材料绑定。
- **能力项 / 阶梯**：L3 长报告、引用、多源矛盾处理和研究计划。
- **课程合成例题**：比较三家公司的供应链风险，逐条给出处、时点和相互冲突的口径，而非只报结论。
- **打分**：rubric criterion pass/aggregate judge score；有的协议允许近 1M token。
- **厂商采用**：Claude Fable 5.1、Step 5、Hy4 等。
- **局限**：Anthropic 披露仅换 judge 就可能移动 10–25pp；引用存在不等于引用支持主张。
- **最高水位口径**：固定 query set、search corpus/date、judge、citation verifier 和 token budget，不能跨 judge 摘最大值。

## 3. 网站、手机与桌面操作

### 3.1 WebArena / WebArena-Verified：在网站中改变真实状态

- **基本信息**：[WebArena](https://github.com/web-arena-x/webarena) 原始套件有 812 个自托管网站任务；Verified 对任务与 grader 做人工复核，具体 N 必须绑定 snapshot。
- **能力项 / 阶梯**：L2 网站导航、表单、购物/论坛/GitLab 等有状态交互。
- **课程合成例题**：在项目站找到指定 issue、加标签并安排里程碑，同时不修改同名诱饵 issue。
- **打分**：终态 functional correctness；有些任务另检查 URL/页面状态。
- **厂商采用**：Qwen3.8-27B 报 WebArena-Verified `64.8`，使用 OSWorld scaffold 与官方 grader。
- **局限**：DOM/截图输入、网站镜像、登录态、最大步数和 grader 修复都会变；成功率不评价多余/危险动作。
- **最高水位口径**：只在相同网站 snapshot、任务 revision、观察空间与 action budget 内比较。

### 3.2 OSWorld lineage：classic / Verified 与 2.x 不是一张榜

- **基本信息**：[classic OSWorld](https://github.com/xlang-ai/OSWorld) 与 2025 年修订的 OSWorld-Verified 属 1.x 任务线；[OSWorld 2.x](https://github.com/xlang-ai/OSWorld-V2) 另有 108 个跨桌面、网页与应用的长程 workflow，论文报告人类完成时间中位数约 1.6 小时。官方在 2026-09-16 又发布 `osworld-v2.1` bug-fix manifest，要求代码、tasks、assets、website 与 VM image 全部同 revision。
- **能力项 / 阶梯**：L3 GUI grounding、跨应用状态、动态信息、失败恢复与长程计划。
- **课程合成例题**：读取迟到的邮件附件，修正 spreadsheet，再在另一个应用同步状态并导出 PDF。
- **打分**：1.x/Verified 常报 task success；2.x 在最多 500 actions 下分 strict binary completion 与 partial reward。三者必须连同 release manifest 命名。
- **厂商采用**：GPT-6 Astra、Claude 5.1、Kimi K3、Qwen3.8 VLM 等。
- **局限**：Astra 用 offline `v2026.08.08`，Claude 用修订 108 题，Qwen/Kimi 的 `OSWorld-Verified` 又属于 1.x 且可能换 scaffold；网页/软件非确定性高。课程快照晚于 v2.1 发布，但厂商旧数不能被静默“升级”为 v2.1 成绩。
- **最高水位口径**：当前**无单一可比最高**；Kimi/Qwen 的 Verified success、Fable `77.9 partial/41.7 strict` 与 Astra `72.6 offline partial` 不是同一实验。

### 3.3 AndroidWorld：116 个跨 20 个 app 的移动端任务

- **基本信息**：[AndroidWorld](https://github.com/google-research/android_world) 的基准任务由可重置模拟器状态和 app verifier 定义，经典套件为 116 题、20 个 Android app。
- **能力项 / 阶梯**：L2/L3 小屏视觉定位、输入、跨 app intent 与状态恢复。
- **课程合成例题**：从短信读取地址，在地图保存地点，再在日历创建带正确时区的事件。
- **打分**：task success；厂商表也可能跑 95-task public subset 或 `avg@3`。
- **厂商采用**：Qwen3.8-Max/27B/Flash-Next 等最新视觉 agent 发布。
- **局限**：116 与 95 子集不可混；模拟器、app 版本、无障碍树/截图、重试和 action granularity 都会改变难度。
- **最高水位口径**：必须写 task subset、设备镜像、观察/动作接口和 trials；不能把 Pass@3 当单次成功率。

## 4. Tool use：函数调用不等于完成业务

### 4.1 Toolathlon-Verified：108 题、32 个应用、604 个工具

- **基本信息**：[Toolathlon](https://toolathlon.xyz/) 的 Verified 版复核了 108 个跨应用任务，覆盖 32 apps、604 tools，平均约 20 轮。
- **能力项 / 阶梯**：L3 多工具规划、依赖顺序、状态查询与写操作。
- **课程合成例题**：读邮件附件、核对学习平台记录、更新成绩并向符合条件的人发送通知。
- **打分**：常见三次运行的 Pass@1、Pass@3、`Pass^3` 与轮数；Pass@3 是至少一次成功，Pass³ 是三次都成功。
- **厂商采用**：Qwen3.8、Step 5、Kimi K3、GLM-5.3-Flash、Claude 5.1、Hy4 等。
- **局限**：Verified 前后差异大；有厂商内部重实现 MCP tools；null attempt、100 steps、2h 与服务失败处理不统一。
- **最高水位口径**：厂商约 `72–78` 的 Pass@1 不构成统一榜；需同一 tool server/harness 做 paired replay。

### 4.2 MCP-Atlas：把 tool protocol 与长程调用分开测

- **基本信息**：[Scale MCP-Atlas](https://labs.scale.com/leaderboard/mcp_atlas) 全集 1,000 题（500 public + 500 private），覆盖 36 servers、220 tools；当前协议常允许最多 100 次 tool calls。
- **能力项 / 阶梯**：L3 MCP tool discovery、schema binding、跨 server 工作流与长程恢复。
- **课程合成例题**：从 cloud drive 找报价单，在 CRM 更新 deal，再在 project tracker 创建带依赖的任务。
- **打分**：task success / judge score，必须锁定 public/private、judge 与 2026-04 前后协议。
- **厂商采用**：Step 5、Kimi K3、GLM、Hy4；Kimi/Hy4 明确跑 public 500。
- **局限**：只写“MCP-Atlas 85”会隐藏 public/private、100-call cap、judge 更新与 server 失败。
- **最高水位口径**：同 leaderboard snapshot 和相同 provider/harness；私有成绩不能由 public 本地复现。

### 4.3 AutomationBench：600 个严格计分业务流程

- **基本信息**：[AutomationBench](https://github.com/zapier/AutomationBench) 有 600 个公开计分任务，Sales/Marketing/Ops/Support/Finance/HR 各 100，涉及 47 个模拟 SaaS 工具；另有 200 smoke tasks 与更难 private set。
- **能力项 / 阶梯**：L3 跨 CRM、邮件、日历、消息与表格的确定性业务自动化。
- **课程合成例题**：找到正确 lead、约会、改状态并通知 Slack，且不得触碰同名诱饵记录。
- **打分**：`partial_credit` 是状态断言通过比例；所有断言都通过才 `task_completed_correctly=1`。
- **厂商采用**：Qwen、Step、DeepSeek、Kimi、GLM、GPT、Claude、Hy4 最新发版几乎全覆盖。
- **局限**：public/private、v1.0.6 修复、最大步数、工具集和 null/error 分母不同；Kimi 自表 `30.8` 与 Step 复跑的 Kimi `46.7` 展示了系统差异。
- **最高水位口径**：固定 benchmark commit、public/private、tool simulator、step budget 与 failure policy，优先报 strict+partial。

### 4.4 τ lineage：文本政策、知识检索与全双工语音

- **基本信息**：[τ³ 官方仓库](https://github.com/sierra-research/tau2-bench) 延续 τ-bench 的 airline/retail 政策与 user simulator，τ² 增加双边交互和 telecom；τ³-Banking 再加入约 700 份、约 195K-token 的非结构化政策库与 97 个任务；[τ-Voice](https://arxiv.org/abs/2603.13686) 则把 278 个 retail/airline/telecom 任务改成可打断、带噪声/口音的全双工音频交互。
- **能力项 / 阶梯**：L3 工具调用、policy compliance、知识检索、信息收集与用户协商；voice 支路额外测听说重叠、打断、响应时延与选择性。
- **课程合成例题**：用户要求退一张不满足规则的票；agent 需核验身份、解释限制并给合法替代，不能为追求“帮忙”绕过政策。
- **打分**：最终数据库状态 + 沟通合同的 task success；文本线常报 `pass^k` 衡量连续多次都成功，voice 线还报告 clean/realistic 条件下的 task pass@1、responsiveness、latency、interrupt rate 与 selectivity。
- **厂商采用**：Kimi K3、Step 5 等报告 τ³-Banking；Gemini 3.8 Live 报 τ-Voice 与 τ-Voice-banking。它们不能与原 τ-bench 合并。
- **局限**：user simulator 与 communication judge 本身是模型；retriever、约 700 文档 snapshot、噪声/口音、实时 provider 和 Pass^k 的 k 都会改分。backend success 也不等于语音自然度。
- **最高水位口径**：锁领域、user model、retrieval、tools、policy/task revision、audio condition、k 和 seed。2026-09-22 的 Sierra 与 Artificial Analysis τ³-Banking 榜对 Qwen3.8-Max 分别显示 `55.2` 与 `51.3`，本身就说明 provider/协议快照要进 score identity；Gemini 3.8 Live 的 `68.6` τ-Voice 与 `35.1` τ-Voice-banking 是两个任务集。

## 5. 长程职业工作：平均 rubric 高仍可能不可交付

### 5.1 ALE：living benchmark，pass 与 score 必须同时报

- **基本信息**：[Agents' Last Exam](https://agents-last-exam.org/) 是持续扩展的 OS sandbox 专业任务；论文描述 1K+ 任务、55 子领域、13 行业簇，公开仓约 150 个 reference tasks，完整库已超过 1,500。厂商还可能使用 105-task snapshot。
- **能力项 / 阶梯**：L3 多小时 CLI/GUI、文档、媒体、工程与专业软件工作。
- **课程合成例题**：分析多份财务文件、更新模型、制作演示，并满足隐藏数值与格式检查。
- **打分**：隐藏 grader 的 `[0,1]` partial `Score`；满分任务比例为 `Pass Rate`。
- **厂商采用**：Qwen、Step、DeepSeek、Kimi、GLM、GPT、Hy4 等。
- **局限**：living set、公开/私有混合；软件许可证、OS 镜像、12h/500 turns、safeguard 与基础设施失败影响大。
- **最高水位口径**：Astra `59.3` 属其完整 computer-use system；其余 ALE/ALE-CLI snapshot 未锁同合同，不做横向减法。

### 5.2 GDPval-AA v2.1：220 个职业交付的相对偏好

- **基本信息**：[Artificial Analysis 公共榜](https://artificialanalysis.ai/evaluations/gdpval-aa) 从 GDPval 的 1,320 题中运行 220 题，覆盖 44 个职业、9 个行业；agent 通过 Stirrup 获得 shell 与 web，产出文档、表格、幻灯片和图示。
- **能力项 / 阶梯**：L3 经济相关职业 artifact 的研究、计算、写作与呈现。
- **课程合成例题**：根据 venue 和乐队需求制作 stage plot PDF；连接位置、尺寸、标签与交付格式分别进入验收。
- **打分**：同题匿名 submission 的盲评 pairwise，聚合为 Elo；v2.1 以 DeepSeek V4.1 Flash max=`1600` 锚定，不是正确率。
- **厂商采用**：Step、Kimi、GLM、GPT、Claude、Grok、Hy4、Qwen 等当前发布。
- **局限**：Elo 随对手池、judge、agent、renderer 与日期变化；专业“看起来更好”不等于所有硬约束都通过。
- **最高水位**：公共 v2.1 榜 2026-09-22 为 Fable 5.1 max `1735`；Claude 卡曾报 `1853`，属于不同 provider/pool snapshot，不回填到当前榜。

### 5.3 AA-Briefcase v1.1：四个多周项目、91 份独立交付

- **基本信息**：[AA-Briefcase](https://artificialanalysis.ai/evaluations/aa-briefcase) 是 4 个 multi-week 私有项目、共 91 tasks 与成千输入文件；每个 task 目前独立运行，不继承模型上周的输出。第五个 public-lite 场景只示范结构，不计正式分。
- **能力项 / 阶梯**：L3 数据科学、产品管理、公司战略中的多文件证据、冲突消解与成品质量。
- **课程合成例题**：从数十份尽调底稿制作市场结构图、财务模型和 briefing video；引用、结论、分析深度与版式分别评分。
- **打分**：binary rubric pass、analytical-quality pairwise Elo 与 presentation Elo，再合成 AA-Briefcase Elo。
- **厂商采用**：Step、Kimi、GPT、Claude、Grok、Qwen 等；Qwen3.8-Max 还出现在独立 v1.1 榜，而非其首发主表。
- **局限**：正式集私有，Elo 仍依赖 pool/judge；“multi-week”描述的是共享输入情境，目前并不测试模型跨周自我记忆。
- **最高水位**：公共榜快照为 Fable 5.1 max `1678`；Claude 发布卡的 `1694` 是更早/不同 pool，二者不做增量。

### 5.4 JobBench：专业人士真正想委托的 130 个任务

- **基本信息**：[JobBench 论文](https://arxiv.org/abs/2605.26329) 从 1,500+ 专业人士的委托偏好出发，覆盖 35 个职业、130 个 agentic tasks；公开 main/easy 为不同的 65/63 题平行 split，不是逐题难化版。
- **能力项 / 阶梯**：L3 在杂乱 dossier 中检索矛盾证据、展示推理链并做职业交付。
- **课程合成例题**：为审计师核对监管 PDF、SQLite 账目与多年 CSV，解释冲突并生成带出处的 exception memo。
- **打分**：每题 fact-anchored weighted rubric；论文平均每题 35.6 个 binary criteria，不能把 rubric pass 当 task all-pass。
- **厂商采用**：Qwen3.8、Step 5、Kimi、GLM、Hy4 等。
- **局限**：live web、harness、judge 与 timeout 强影响；main/easy 不能混合，平均 rubric 不能保证完整交付。
- **最高水位口径**：原论文统一实验最佳为 Opus 4.7 `45.9`；后续厂商表出现 Step `59.0`、Fable `57.4` 等，但 agent/revision 未锁同合同，只作 B-vendor。

### 5.5 Workspace-Bench 1.0：20,476 个文件中的依赖与版本谱系

- **基本信息**：[Workspace-Bench](https://arxiv.org/abs/2605.03596) 有 5 个职业 workspace、74 种文件类型、20,476 files、388 tasks 和 7,399 rubrics；Lite 为保持分布的 100 题子集，成本约低 70%。
- **能力项 / 阶梯**：L3 workspace exploration、跨文件语义/血缘关系、supporting/result files 选择与一致交付。
- **课程合成例题**：在数千文件中识别过期 `final_v2` 与当前定价表的依赖，更新预测表和管理层 memo，且保留来源链。
- **打分**：平均 rubric pass，并可看 dependency node/edge F1 与 task-level TCR@30/50/70/90/100。
- **厂商采用**：Qwen3.8-Max 等；模型表常只写 `WorkSpaceBench`，必须核对 full/Lite 与 harness。
- **局限**：不同 harness 的成本/turn 差异巨大；LLM judge、20GB workspace provisioning 与文件 renderer 都可能成为瓶颈。
- **最高水位**：论文 Lite 同协议最佳约 `68.7`、human-in-loop `80.7`；Qwen 首发表同样把 Fable `68.7`、Qwen Max `67.7` 记为厂商运行，不代表 full 388 题。

### 5.6 OfficeQA Pro / Pro V2：财务 PDF 检索、表格与计算

- **基本信息**：[OfficeQA 官方仓](https://github.com/databricks/officeqa) 分 Pro 133 题、Full 246 题与新语料 Pro V2 90 题；前两者使用 697 期 U.S. Treasury Bulletins，V2 使用 1,435 份更老、更难解析的 Federal Accounts 文档。
- **能力项 / 阶梯**：L2/L3 文档解析、跨期检索、表格数值对齐与 grounded calculation。
- **课程合成例题**：从数期财政公报找到口径改变前后的同一收支项，统一单位、计算差额并给出页码证据。
- **打分**：official reward 对规范化答案计 accuracy；agent-harness、oracle pages、extracted text 与 PDF image 是四种不同观察合同。
- **厂商采用**：Step、Kimi、GLM、Claude、Hy4 等。
- **局限**：OCR/表格表示、corpus indexing、web access 和绝对/相对误差阈值会移动分数；Pro 与 Pro V2 不合榜。
- **最高水位口径**：Fable 卡在 extracted-text harness 报 OfficeQA/OfficeQA Pro `80.2/69.0`；更新且换了语料的 Pro V2 官方仓快照最佳 `54.4`。三者不是难度刻度上的同一 split，不能挑最大值称“OfficeQA 80.2”。

## 6. 把 Agent 分数拆成四个分母

至少同时保存：

| 分母 | 问题 | 例子 |
|---|---|---|
| attempted | 所有计划运行是否都进入分母 | provider error、timeout、safeguard block 仍计 0 |
| completed | 成功跑完的 episode 有多准 | 诊断模型逻辑，不能替代端到端可靠率 |
| strict success | 是否满足全部关键 verifier | Automation/ALE/Legal all-pass |
| partial progress | 满足了多少非关键 criteria | FrontierSWE/OSWorld partial、rubric average |

产品发布通常还需要 harm/rescue：agent 是否误改诱饵对象、能否从工具失败恢复、人工接管后是否挽救。只有 success
而没有 side-effect 与 failure denominator 的榜单，不足以授权真实执行。

## 费曼自检

1. BrowseComp 91% 为什么不能推出 deep-research 报告 91% 可用？
2. Toolathlon Pass@3 上升、Pass³ 下降，可能说明什么？
3. OSWorld partial 77.9 与 strict 41.7，哪个更“真实”？
4. 为什么 GDPval Elo 1853 不能解释为 85.3% 正确？
5. 怎样区分底座变强与 harness 变强？

<details>
<summary>参考答案</summary>

1. BrowseComp 主要裁决短答案；报告还要求引用支持、覆盖 rubric、处理矛盾和结构可用。搜索预算、judge 和网页日期也属于协议。
2. 系统更容易在三次中撞中一次，却不能稳定重复成功；可能增加了随机探索或重试。前者适合 best-of 搜索，后者更接近重复部署可靠性，还需同时看成本。
3. 两者都真实但回答不同问题：partial 适合定位进展和 bottleneck，strict 更接近“用户是否拿到完整交付”。生产决策通常把 strict 作主门、partial 作诊断。
4. Elo 是特定对手池和 judge 下的相对位置，零点与尺度会随 pool 改变；既不是 task accuracy，也不能跨日期直接相减。
5. 固定 benchmark revision、harness、工具、预算、sampling 和 grader，只换 checkpoint 做 paired run；再固定 checkpoint 换 harness。两组 factorial 分别估计 model 与 scaffold effect，并保留交互项。

</details>

一句话验收：**Agent benchmark 的被测对象默认是整套执行系统；若要把增益归到模型权重，必须另做固定 harness 的配对实验。**
