# 分册 01：文本、知识、推理与长上下文 benchmark

> **核心问题**：怎样区分“知道答案”“能推导答案”“能遵守输出约束”和“能从百万 token 中找到并综合证据”？
> **范围**：受控问答、数学/科学、instruction following、专业 rubric 与 long context；动态水位见 [发布证据账](RELEASE_LEDGER.md)。
> **快照**：2026-09-22；“当前/截至快照”的水位只对该日期成立。

## 1. 通识基线与专家推理

### 1.1 MMLU-Pro：通识覆盖的回归测试，不再是 frontier 终点

- **基本信息**：[MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) 把传统 MMLU 扩为 12,032 道、14 个领域、最多 10 选 1，并提高 reasoning 比例、修正噪声。
- **能力项 / 阶梯**：L0 学科知识 + 中短推理；适合 base/instruct 的广覆盖 smoke test。
- **课程合成例题**：给一段经济学情境，从十个相近政策结果中选择最符合因果链的一项。
- **打分**：accuracy；必须写 few-shot/zero-shot、CoT、答案抽取与 macro/micro 聚合。
- **厂商采用**：常见于 base model 技术报告；2026 最新旗舰发版已减少用它做 headline，转向 HLE、agent 和专业工作。
- **局限**：公开多年、污染与饱和风险高；多选允许猜测；90% 不能推出开放任务可靠率 90%。
- **最高水位口径**：保留为历史/回归水位，不再用几个点差宣称 frontier reasoning SOTA。

### 1.2 GPQA Diamond：小而难的研究生科学闭卷题

- **基本信息**：[GPQA 论文](https://arxiv.org/abs/2311.12022) 主集 448 道生物、物理、化学专家多选题；Diamond 是 198 道最高质量子集。
- **能力项 / 阶梯**：L0 专业知识 + 多步科学推理。
- **课程合成例题**：给一个有机反应机理与实验条件，判断主产物立体构型，并从四项中选择。
- **打分**：四选一 accuracy，常用 pass@1/单次；需写 prompt、CoT、sampling 与是否 search。
- **厂商采用**：Qwen3.8、Kimi K3、Step 5 Preview、GPT-6 Astra 等均采用。
- **局限**：仅 198 题，1 题约 0.5pp；已进入 90%+ 饱和区，错题审计与污染比小差排名更重要。
- **最高水位**：截至 2026-09-22，厂商正式表中的候选高水位为 GPT-6 Astra 96.0%；这是 B-vendor，不等于统一公共榜。

### 1.3 HLE：跨百余学科的 frontier exam

- **基本信息**：[Humanity's Last Exam](https://epoch.ai/benchmarks/hle) 含 2,500 道专家编写题，跨 100+ 学科，约 14% 需要图像。
- **能力项 / 阶梯**：L0/L1 交界的长尾专家知识、推理与少量多模态。
- **课程合成例题**：给冷门数学对象的严格定义，要求推导唯一短答案；不是“解释一下”式开放作文。
- **打分**：exact / judge correctness；no-tools 与 with-search/code 必须分栏，有时还报告 calibration error。
- **厂商采用**：Step 5、Qwen3.8、Kimi K3、GLM、GPT、Claude 等最新发版反复采用。
- **局限**：工具版测系统；不同厂商用 text-only/full、不同 judge 和 token cap；题集公开后会逐步污染。
- **最高水位**：当前官方发布候选为 Fable 5.1 no-tools 60.9%、with-tools 65.0%；Step 的 text-only 不能直接加入 full-set 排名。

### 1.4 AIME：年度 30 题的数学推理探针

- **基本信息**：[MAA AIME](https://maa.org/math-competitions/american-invitational-mathematics-examination-aime) 每年通常 15+15 道高中竞赛整数答案题。
- **能力项 / 阶梯**：L0 代数、数论、几何、组合的多步推理。
- **课程合成例题**：求满足若干整除和计数约束的三位整数个数，答案格式为 `000–999`。
- **打分**：30 题 accuracy；常见 `avg@k`、`pass@k` 或 majority vote 必须分开。
- **厂商采用**：reasoning 模型常用，但 Kimi K3 Vendor Verifier 已明确不再用 AIME 2025 做供应商一致性检查。
- **局限**：30 题导致每题 3.33pp；年度题迅速公开，prompt/采样选择非常容易改变排名。
- **最高水位口径**：按年份、AIME I/II、采样与工具分栏；接近满分后更适合做回归而非总榜。

### 1.5 FrontierMath / ARC-AGI：接近饱和时先审题目与 harness

- **基本信息**：[FrontierMath](https://epoch.ai/frontiermath) 是私有/半私有高难数学；[ARC Prize](https://arcprize.org/) 用少量示例推断抽象网格变换，版本 1/2/3 不同。
- **能力项 / 阶梯**：L0 抽象归纳、数学发现与 inference-time search。
- **课程合成例题**：从三个输入—输出彩色网格归纳规则，生成第四个输出；或证明一个研究级数学命题的特例。
- **打分**：题目成功率；ARC 还需固定行动/搜索 harness，FrontierMath 要区分 tier/version。
- **厂商采用**：GPT-6 Astra 报 FrontierMath Tier 4 v2 97.6%、ARC-AGI-3 99.9%，均带专用运行合同。
- **局限**：接近 100% 后，题目版本、泄漏、工具/search budget 和 harness 的一处变化可主导结论。
- **最高水位口径**：只称“该版本/该 harness 的 vendor-reported 高水位”，并优先等待下一代 fresh set。

## 2. Instruction following 与专业 rubric

### 2.1 IFBench：可执行约束，而非主观“听话”

- **基本信息**：[Ai2 官方仓库](https://github.com/allenai/IFBench) 提供 300 个 prompt、58 个新的 OOD 可验证约束、7 个类别及 verifier；83 个 verifier 中另含 25 个旧 IFEval 约束，不能误写成 83 种新能力。
- **能力项 / 阶梯**：L0 精确遵循数量、位置、格式、禁用词、跨段关系等约束。
- **课程合成例题**：`写三段；每段恰好两句；第二段不能出现字母 e；最后一句以 17 结尾`。
- **打分**：常见 prompt-level loose accuracy；也可按 constraint-level strict/loose，必须标明。
- **厂商采用**：Qwen3.8、Artificial Analysis 最新指数等采用。
- **局限**：verifiable constraints 偏形式，不覆盖意图理解与内容质量；prompt-level 会被一项小错整体置零。
- **最高水位口径**：同 verifier registry、prompt set 与 strict/loose 定义内比较；不同 IFEval/IFBench 不合并。

### 2.2 HealthBench：rubric 覆盖优于“像医生”，但会奖励啰嗦

- **基本信息**：[OpenAI HealthBench](https://openai.com/index/healthbench/) 有 5,000 段对话和 48K+ physician-authored rubric criteria；Professional 子集约 525 对话。
- **能力项 / 阶梯**：L0/L3 交界的医疗回答完整性、正确性、沟通与安全。
- **课程合成例题**：用户描述胸痛及危险信号，回答必须同时覆盖急诊建议、不能自行驾车、关键信息追问与避免确诊。
- **打分**：逐 criterion judge 后聚合；最新发版还报告 length-adjusted score。
- **厂商采用**：GPT-6 Astra、Claude Fable 5.1、Grok 4.7 均报告，但 judge 与长度修正不同。
- **局限**：raw rubric score 会奖励长答案；LLM judge、拒答政策和 safety filter 使跨厂裸比失真；不是临床结局。
- **最高水位**：同表 length-adjusted 候选为 Astra 63.4%；Fable 62.1%、Grok 56.7 使用不同 judge/协议，不能视为统一榜。

### 2.3 PRBench：专业开放题的“逐条得分”，不是案件完成率

- **基本信息**：[PRBench / JusticeBench](https://www.justicebench.org/dataset/prbench) 公开 1,100 个专家题：金融 600、法律 500，来自 182 位从业者，覆盖 114 个国家和美国 47 个司法辖区；共 19,356 条 rubric，每题 10–30 条，并有金融 300 / 法律 250 题的 Hard 子集。
- **能力项 / 阶梯**：L0/L1 专业事实、程序、适用法律或财务分析、处理不确定性和可执行建议；它仍以一轮开放回答为主，不等于在文件环境中完成长期项目。
- **课程合成例题**：给出跨境收购的简化事实，要求列出适用审批、时间线、重大不确定性及每项依据；rubric 分别检查法域、程序、结论、风险披露和可操作性。
- **打分**：原子 criterion 二元判断后按 `-10…+10` 严重度加权聚合；需固定 rubric、judge、是否搜索/代码和归一化。平均 `58%` 表示获得约 58% 的 rubric 信号，不表示 58% 的整案完全可交付。
- **厂商采用**：Qwen3.8-Max 发版表列 PRBench-Legal `57.6`、PRBench-Finance `58.3`（B-vendor）。
- **局限**：LLM judge、开放答案风格和搜索工具会改变得分；地区覆盖广不等于每个法域都有足量样本；criterion average 会掩盖单个致命遗漏。
- **最高水位口径**：截至 2026-09-22，官方数据页给出的 Hard 子集已报告高值约为金融 `0.39`、法律 `0.37`，但其模型池/协议与 Qwen 表不同，不能拿 `39` 与 `58.3` 直接相减。

### 2.4 PLawBench：三类真实法律流程的细粒度 rubric

- **基本信息**：[PLawBench 论文](https://aclanthology.org/2026.acl-long.458/) 含 850 题、约 12,500 条专家 rubric，覆盖公共法律咨询、实务案件分析、法律文书生成三类、13 个场景。
- **能力项 / 阶梯**：L0/L1 争点与关键事实识别、结构化法律推理和文书一致性；比四选一法律知识题更接近律师工作，但仍不是带沙箱和多文件行动的 agent benchmark。
- **课程合成例题**：根据一段劳动争议事实，先识别请求权基础和举证责任，再生成含事实、规则、适用与风险提示的短意见书。
- **打分**：LLM evaluator 逐 rubric 判断并聚合；论文报告其与法律专家判断的一致性。跨表比较必须固定题集版本、judge、语言和回答预算。
- **厂商采用**：Qwen3.8-Max 正式表列 `73.2`（B-vendor）；不能与 PRBench 或 Legal Agent Benchmark 的百分数横比。
- **局限**：850 题仍主要测回答产物而非工具轨迹；judge 偏差、法域/语言分布和模板化写作都可能影响分数。
- **最高水位口径**：课程只保留论文同协议结果和厂商同表结果两条水位；在没有逐题 manifest 与统一复跑前，不宣布跨来源 SOTA。

### 2.5 Harvey LAB 与 LAB-AA：`criterion-pass` 高，不代表整项工作完成

- **基本信息**：[Harvey Legal Agent Benchmark](https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark) v1 公开框架包含 1,250 个长程任务、24 个执业领域和 75,000+ 条专家 rubric；instruction 平均约 50 词，agent 要在 client-matter 文件环境中产出 memo、redline、schedule 等交付物。Artificial Analysis 的 [Harvey LAB-AA](https://artificialanalysis.ai/evaluations/harvey-lab-aa) 是另一条运行合同：用 Stirrup harness 跑 Harvey 提供但不公开的 120 题私有集。
- **能力项 / 阶梯**：L3 多文件发现、长程规划、法律分析和最终文件交付。
- **课程合成例题**：在一组收购协议、邮件和披露表中找出控制权变更条款，计算合计风险暴露，生成 consent/waiver 时间线和带引用的审查报告。
- **打分**：同时看 criterion-pass 与 all-pass。前者是通过的 rubric 比例；后者只有一项任务的**全部** criterion 都通过才计 1。二者绝不能放在同一排行榜列。
- **厂商采用**：Kimi K3 表中的 Harvey LAB-AA `94.6` 是 criterion-pass；GPT/Claude/Grok 等发布还引用 Legal Agent all-pass 或其他 holdout，身份与 manifest 必须逐项核对。
- **局限**：AA 的 120 题 headline 数据不公开、使用单一 LLM judge；Harvey 公共 v1 又持续演进。高 criterion-pass 仍可能因一个关键风险遗漏而 all-pass 失败。
- **最高水位**：截至快照，LAB-AA criterion-pass 公共榜候选为 Muse Spark 1.3 `95.5%`、Kimi K3 `94.6%`；Harvey 自己的 2026-05 holdout all-pass 当时最高仅 `7.1%`。这组巨大差距首先说明 metric/题集不同，不是模型退化。

### 2.6 FrontierFinance：同名 benchmark 先查身份，再谈金融智能

- **基本信息**：2026 年存在至少两条同名谱系。Samaya 的[开放 rubric 版](https://arxiv.org/abs/2608.11683)有 220 个专家 query、11,543 条可溯源 rubric，覆盖投资者工作流的 6 类用例；另一篇[长程 computer-use 版](https://arxiv.org/abs/2604.05912)是 25 个真实金融工作任务。名称相同不代表同一数据或 metric。
- **能力项 / 阶梯**：前者偏 L1/L3 的公开数据检索、金融分析与带依据交付；后者偏 L3 浏览器/桌面操作和长程执行。
- **课程合成例题**：检索某公司多期公告与市场数据，统一会计口径，完成估值敏感性表，并让每个假设和数字都可回溯到来源。
- **打分**：220-query 版按 11,543 条 source-attributed rubric 聚合，还应同时报告成本、延迟和 harness；25-task 版按环境终态/任务 rubric 计分。两条分数不可合榜。
- **厂商采用**：Step 5 Preview 采用 220-query 外部集，并另报 FinStep LiveSearch、CorporateValuation、FinanceDR 等内部集；其他发版若只写 `FrontierFinance`，必须从来源或 task manifest 判定是哪一版。
- **局限**：模型、搜索源和 agent harness 共同决定结果；动态网页、市场时点和付费数据会破坏可复现性；rubric average 也不能证明投资决策有真实收益。
- **最高水位**：220-query 版论文在统一公共数据 harness 下报告 Samaya 系统 `56.0%`、最强单一 frontier model `49.2%`；它是系统级水位，不可解释为 base-model 独立能力。

## 3. Long context：容量、检索、综合与长期记忆是四件事

### 3.1 MRCR v2：多根相似针的顺序检索

- **基本信息**：[MRCR v2 官方说明](https://github.com/google-deepmind/eval_hub/blob/master/eval_hub/mrcr_v2/README.md) 在长对话中放入 2/4/8 个相同请求，最后要求复现第 $i$ 次请求对应的回答；长度桶可到 8M，总题数由生成配置决定，不应写一个固定 N。
- **能力项 / 阶梯**：L1 长上下文定位、共指/序号区分与精确复制。
- **课程合成例题**：20 万 token 中八次都问“写一首关于海的诗”，最后要求逐字返回第六次的回答。
- **打分**：常按目标与输出的相似度/正确率，在 128K、256K、512K、1M 等长度桶聚合。
- **厂商采用**：Qwen3.8、GPT-6 Astra、Claude 等长上下文发版采用。
- **局限**：任务复杂度固定、主要是检索复制；不能证明跨文档因果、代码修改或长期规划。
- **最高水位**：Astra 官方报 8-needle 512K–1M 为 96.3%；这是 B-vendor 且只支持 needle retrieval 主张。

### 3.2 LongBench v2：从 needle 走向真实长文推理

- **基本信息**：[官方仓库](https://github.com/THUDM/LongBench) v2 有 503 道多选题，上下文 8K–2M words，覆盖单/多文档 QA、长 ICL、长对话、代码库与结构化数据六类。
- **能力项 / 阶梯**：L1 多处证据综合与真实长输入推理。
- **课程合成例题**：给多个版本的法规与公司报告，问某政策变化如何改变两项财务指标，答案不在单一句子中。
- **打分**：四选一 accuracy；应按 32K 以下、32K–128K、128K+ 和任务类分桶。
- **厂商采用**：Qwen/Kimi/GLM 等长上下文模型发布采用。
- **局限**：503 题且上下文形态异质；模型可能用截断/RAG，需说明实际输入；人类 15 分钟基线不是无限时上限。
- **最高水位口径**：同 prompt、context truncation 和 direct/CoT 协议内比较；发布账只记录可复算快照。

### 3.3 AA-LCR v1.1：100 个约 100K-token 职业文档问题

- **基本信息**：[AA-LCR v1.1 数据集](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) 以 Apache-2.0 公开 100 题、答案、source URLs 与 judge system prompt；共 30 个 document sets、234 份文档，平均每题上下文约 99K tokens，覆盖公司/行业报告、政府咨询、学术、法律等。v1.1 修正了 16/100 个 answer keys，不能与 v1.0 直接相减。
- **能力项 / 阶梯**：L1 多文档证据连接、计算与综合。
- **课程合成例题**：比较三家公司两年报告中的口径变化，重算同口径毛利并解释差异来源。
- **打分**：LLM judge 判断与官方答案等价，取平均 pass rate；独立运行还报告 token、cost 和 time/task。
- **厂商采用**：Step 5、Kimi K3 等模型由 Artificial Analysis 独立运行。
- **局限**：100 题，小差有噪声；题集虽公开，动态 API、采样、模型 alias 与第三方运行时间仍限制逐次完全复现；不能外推 1M。
- **最高水位**：截至快照，独立 AA-LCR v1.1 为 Kimi K3 max `88.7`、Step 5 Preview `88.3`，可作带日期的 A-public candidate。Kimi 模型卡另列的 `74.7` 是 2026-07-23 厂商快照且未标 revision，不与 v1.1 合榜。

### 3.4 BEAM：百万 token 会话里的十类长期记忆

- **基本信息**：[BEAM 论文](https://arxiv.org/abs/2510.27246) 完整基准为 100 个会话、2,000 个验证问题，可生成至 10M；1M 档有 35 个会话。官方顶层资料未单列 1M 档总问题数；若使用 Kimi Vendor Verifier 的裁剪集，必须另记其 manifest，不能把它当 BEAM-1M 永久题量。
- **能力项 / 阶梯**：L1 abstention、矛盾消解、事件顺序、信息抽取、指令/偏好、知识更新、多 session、总结与时间推理。
- **课程合成例题**：一百万 token 的多次会话中，用户先喜欢 A、后明确改为 B；问当前偏好并引用最后一次更新。
- **打分**：generation 与 judge 解耦，按能力/长度聚合；供应商验证应固定 judge temperature。
- **厂商采用**：Kimi K3 用它检查不同推理服务商是否保持 1M 能力。
- **局限**：synthetic 长会话、judge 依赖、运行昂贵；原生 1M 与外置 memory/RAG 系统是不同对象。
- **最高水位口径**：full 100-conversation 与供应商 35-conversation 子集分栏；0.31 一类供应商分数不能当跨模型通用百分数。

## 4. Long context 的四级证据梯

```text
接口能接收 N token
  → needle / MRCR 能找回
  → LongBench / AA-LCR 能跨证据推理
  → ProgramBench / repo / live work 能在长轨中持续行动
```

每一层都是下一层的必要但不充分条件。只报告 context window 是产品规格，不是评测结果；只报告 MRCR 也不能回答
“为什么不直接 RAG”。真正决策还要比较质量、首 token 延迟、总成本、缓存命中、更新频率和可引用性。

## 费曼自检

1. GPQA 96% 和 HLE 60% 为什么不矛盾？
2. HLE with-tools 高于 no-tools，应该把增益记到模型还是系统？
3. 1M MRCR 96% 为什么仍可能在 100K 的 AA-LCR 上失败？
4. HealthBench raw score 高于 length-adjusted，说明了什么测量问题？

<details>
<summary>参考答案</summary>

1. 两者题目分布、规模、模态和难度不同；GPQA Diamond 只有 198 道三学科多选且趋近饱和，HLE 覆盖更长尾的 2,500 道专家题，不能用百分数直接比较难度。
2. 记为 `model+harness+tools+budget` 的系统增益；若要估计模型增益，需两模型在同一工具合同下 paired compare，并同时报告工具失败和成本。
3. MRCR 的核心是从多个相似 needle 中定位并复制；AA-LCR 要跨多份真实文档做多步推理与计算。窗口容量和简单检索可以成立，综合推理仍失败。
4. rubric judge 会奖励覆盖更多 criterion 的长回答，哪怕冗余降低真实可用性；长度校正试图把“更会写长”与“更正确”分开，因此两者都要报告。

</details>

一句话验收：**文本 benchmark 应沿“覆盖—专家推理—约束—专业交付—长输入行动”逐层买证据，不能让一个总分替五种能力发言。**
