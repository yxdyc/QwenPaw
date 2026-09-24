# 2026-09-22 前沿基模发版 Benchmark 证据账

> **用途**：回答“各家最新发版到底在用什么 benchmark、真正关注什么能力、最高水位在哪里”。
> **快照**：2026-09-22。模型、榜单和 living benchmark 会继续变化；引用时必须连同日期。
> **边界**：本页记录厂商正式发布材料与公共榜，不把厂商自测写成独立复现，也不把不同协议的最大数字拼成排行榜。

稳定的 benchmark 定义、课程合成例题和局限放在
[文本/推理](01-text-reasoning.md)、[代码](02-code-software.md)、[Agent](03-agent-tool-use.md)、
[多模态理解](04-multimodal-understanding.md)、[生成](05-multimodal-generation.md)与[前沿专项](06-frontier-specialized.md)六个分册；
Qwen3.8 的全部表另有[逐项导读](QWEN38_WALKTHROUGH.md)。本页只维护会随发版变化的
“模型身份—测试合同—结果”三元组。

## 1. 先辨认本次发版对象

| 厂商 | 截止快照日的最新对象 | 开放边界 | 本次 benchmark 重心 |
|---|---|---|---|
| Qwen | Qwen3.8-Max；开放对应物 Qwen3.8-2.4T-A95B | 2.4T/95B 开放权重版是 text-only、thinking-only；托管 Max 另有 vision、non-thinking、默认 1M 与内置工具 | coding、cowork、长程 agent、专业工作、长上下文 |
| StepFun | Step 5 Preview | 截至快照日是 preview；官网承诺 2026-10-15 开放权重 | coding/terminal、工具、办公、金融与长程 agent |
| DeepSeek | DeepSeek-V4.1-Flash | 开放权重；Base 与 Instruct 结果必须分表 | 统一长程 harness、coding、cyber、工具和视觉 agent |
| Moonshot | Kimi K3 | 7 月首发、7 月 27 日开放权重 | coding、research、office、原生视觉与 1M agent |
| Z.ai | GLM-5.3 与 GLM-5.3-Flash | 5.3 为旗舰后训练更新；Flash 是 320B/A18B 原生多模态新 base，不能并成一次消融 | coding/terminal、工具、office；Flash 增加视觉 |
| OpenAI | GPT-6 Astra | API/产品开放，参数、权重和训练配方未公开 | computer use、专业工作、coding/science、1M 检索、安全 |
| Google | Gemini 3.8 Flash；语音分支为 3.8 Live / Live Extended Thinking | API 服务；Flash 是最新通用工作模型，Live 是 speech-to-speech 专项，不能混表 | 高性价比 coding/agent、长视频、专业工作；实时语音 agent |
| Anthropic | Claude Fable 5.1 / Mythos 5.1 | 同底座不同 safeguard；Mythos 是 trusted-access；部分请求会 fallback | coding、长程终端、office/GUI、专业交付、安全 |
| SpaceXAI | Grok 4.7 | API 模型，500K text/image input；权重与配方未公开 | coding/engineering、CAD、电路、bio/cyber safety |
| Tencent | Hy4 preview | 770B/49B active、开放权重；通用基模 | coding、search、working agent、STEM 与 1M 上下文 |
| MiniMax | MiniMax H3 | 开放的是视频—音频生成 Base 权重的指定能力，不是本表其他模型那种通用问答基模 | FL2VA/Ref2VA、视频与音频联合生成 |

三个容易混淆的身份问题：

1. `Qwen3.8-Max` 是托管系统，`Qwen3.8-2.4T-A95B` 是开放权重 text-only checkpoint；不能把 Max 的 vision/工具能力自动赋给本地权重。
2. `GLM-5.3` 与 `GLM-5.3-Flash` 的 base、架构、规模和模态不同；“Flash 更晚”不表示它在所有任务上替代旗舰。
3. `Hy4 preview` 是腾讯当前通用基模；HunyuanImage 3.0、HunyuanVideo 与 Hunyuan3D-Buffalo 属于图像、视频、3D 生成产品线，不能混进通用基模总榜。

## 2. 从榜单迁移看前沿能力重心

| 能力面 | 旧式常见证据 | 最新发版正在迁向 | 为什么迁移 |
|---|---|---|---|
| 知识/推理 | MMLU、GSM8K、HumanEval | GPQA、HLE、CritPt、FrontierMath、Terminal-Science | 老题趋于饱和/污染；需要专家题与可执行科学工作流 |
| Coding | HumanEval、MBPP、SWE-bench Verified | DeepSWE、SWE-bench Pro、Terminal 2.1/4.0、FrontierCode/SWE、SWE-Marathon | 从函数生成迁到 repo、terminal、多小时交付和 scope control |
| Agent | GAIA、静态 tool calling | Toolathlon、MCP-Atlas、AutomationBench、ALE、Job/Workspace、GDPval/Briefcase | 从“会调用工具”迁到有状态业务流程与职业 artifact |
| 长上下文 | needle-in-haystack | MRCR、AA-LCR、BEAM-1M、ProgramBench、长程 rollout | 从容量/检索迁到区分相似证据、压缩、恢复与持续行动 |
| 多模理解 | VQAv2、MMBench | MMMU-Pro、MathVision、CharXiv、OmniDocBench、BabyVision、OSWorld | 从通用 VQA 迁到低层感知、文档、专业图表与行动闭环 |
| 多模生成 | FID、CLIPScore | GenEval/DPG/T2I-CompBench、VBench、分层盲评 | 单一 embedding 距离无法覆盖组合遵循、运动、同步与偏好 |
| 安全 | 单轮拒答率 | prompt injection、computer-use safety、cyber/bio 双用途、部署 routing | agent 有工具和环境权限后，风险对象从文字变成行动链 |

这不是“传统 benchmark 已无用”。MMLU/LiveCodeBench 仍适合做低成本回归；只是越接近发布主张，越需要更高层、更新鲜、
可执行且带失败分母的证据。

## 3. 各家最新发版：完整 headline inventory

### 3.1 Qwen3.8-Max：从聊天模型转向 coding + cowork

官方开放模型卡明确说明：下列 Max 数字来自托管系统或其指定 harness，不等于下载 2.4T checkpoint 后裸跑即可复现。

| 组别 | 发版报告的 Qwen3.8-Max 项目 | 读数 |
|---|---|---|
| Coding agent | Terminal-Bench 2.1 `86.6`；SWE-bench Pro `67.7`；DeepSWE 1.1 `56.6`；NL2Repo `55.9`；FrontierSWE v1 dominance `73.5`；MLS-Lite `41.0`；PaperBench Code-Dev `93.0`；AndroidBench `75.1` | repo/terminal/paper-to-code/移动端是一组系统能力；FrontierSWE v1 不是 v2 partial reward |
| 内部 coding | QwenSWE `80.7`；QwenQoder `58.4`；QwenReact Elo `1724`；QwenSVG Elo `1713` | C-internal；Elo 不是正确率 |
| General agent | CoWorkBench `74.8`；WorkspaceBench `67.7`；JobBench `53.4`；SkillsBench `70.2`；ALE pass/score `27.0/52.4`；Automation public `27.3`；Toolathlon `72.5`；WideSearch `81.9`；HLE tools `56.2` | 同时报告 pass 与 partial score 是好习惯；工具/agent 不统一时不能跨厂硬排 |
| General | GPQA `92.6`；HLE `43.6`；IFBench `82.8`；`$OneMillion-Bench` expert `52.5`；HealthBench `60.2`；PLaw `73.2`；PRBench-Legal `57.6`；Finance `58.3` | 知识题、遵循、专业 rubric 是不同构念 |
| Long context | MRCR v2 256K 8-needle `92.9`；LongBench v2 `66.3` | 证明指定协议下的检索/理解，不证明任意 1M 工作流 |

关键协议：Terminal 2.1 使用 Claude Code、`avg@10`、5 小时、131K max tokens；DeepSWE 在 Claude Code 与
mini-SWE-agent 中取较高值；Automation 只跑 600 题 public；SkillsBench 各家使用的 harness 不同；WideSearch 外部模型用
Claude Code、Qwen 用 Qwen-Agent。PaperBench `93.0` 明确是 BasicAgent **Code-Dev**：只评代码开发，跳过完整模式的
独立执行与实验结果复现，不能称作“93% 论文完整复现率”。因此这张表首先是一张“产品关注地图”，不是统一榜。

托管 **Qwen3.8-Max** 还有一张独立的 [VL Performance Chart](https://github.com/AlibabaCloud-Official/Qwen3.8-max)，
不能用开放 27B/Flash-Next 的表代替，也不能把这些分数归给 text-only 2.4T checkpoint。为避免丢项，这里保留官方图的
全部行名与 Max 数字；斜杠左右的 metric/mode 必须继续绑定图注，不能当置信区间：

| Max VL 组别 | 官方图中的完整 inventory |
|---|---|
| Multimodal reasoning | MMMU-Pro `82.3`；MathVision `95.2/97.7`；BabyVision `82.0/91.3`；HLE-VL tools `52.2`；ZeroBench Pass@5 `24.0/49.0`；ZeroBench-Sub `48.5`；LogicVista `91.9`；HiPhO `90.0`；PhyX `83.5`；SLAKE `90.8`；MedXpertQA-MM `80.4`；PMC-VQA `66.2` |
| Visual agent & coding | OSWorld-Verified `86.1`；OSWorld 2.0 `19.4/46.7`；ScreenSpot Pro `84.5`；WebArena-Verified `66.8`；AndroidWorld `85.3`；MobileWorld `77.8`；ClawEval-MM `77.2/74.8`；Vision2Web `69.0`；QwenBlenderBench `69.9`；Parametric CAD Bench `91.5`；RecreationBench `51.7`；PresentBench `79.6` |
| Document & office | CharXiv-RQ `88.4/93.5`；OmniDocBench 1.5 `92.1`；OCR-Bench-v2 EN/ZH `74.2/68.3`；CC-OCR-Bench-v2 `79.6`；MTVQA-Test `56.6`；MADQA `91.8`；QwenVisualOffice `44.6` |
| Real-world & spatial | RealWorldQA `88.0`；ERQA `77.8`；LingoQA `84.8`；SURDS `77.8` |
| Perception & grounding | SimpleVQA `75.0`；WorldVQA `53.2`；MMStar `85.9`；PerceptionBench `63.5`；CountQA `82.4`；RefAdv-S `80.2`；Dense200 `87.0`；COCO `78.7`；VisFactor `60.8`；VLMsAreBiased `88.3` |
| Video intelligence & agents | VideoMME w/sub `90.4`；VideoMME v2 w/sub `68.3`；VideoMMMU `88.7`；MMVU `82.4`；MLVU M-Avg `90.8`；TVBench `81.9`；LVBench `81.8`；LVBench w/memory `85.6`；EgoLife w/memory `80.3`；VideoDR w/search `73.2` |

逐项体量、指标、课程合成题与限制见 [Qwen3.8 专页](QWEN38_WALKTHROUGH.md)；其中 QwenBlenderBench、
RecreationBench、QwenVisualOffice 等内部集只作 C-internal 方向证据。

Qwen3.8 的**开放 VLM 变体必须另表**。27B 是 dense 原生 vision-language model；Flash-Next 是 125B total/6B
active backbone + 51B n-gram embedding 的架构预览。它们的官方 VL headline 为：

| 能力 | Qwen3.8-27B | Qwen3.8-Flash-Next | 解释 |
|---|---:|---:|---|
| Computer use | OSWorld-Verified `84.3` | OSWorld 2 binary/partial `19.4/52.3` | benchmark/revision/metric 不同，不能比较两格高低 |
| Browser/mobile | WebArena-Verified `64.8`；AndroidWorld `81.9` | AndroidWorld `84.5` | agent scaffold、设备 snapshot 与 step budget 是合同字段 |
| Multimodal tool | ClawEval-MM Pass@3/avg `57.4/56.9` | `64.4/60.4` | Pass@3 与三次平均回答不同问题 |
| App/web/code | Recreation `47.1`；Vision2Web `62.9`；SWE-MM `38.6` | Recreation `49.9`；Vision2Web `64.0` | Recreation 是内部集；Vision2Web 用 Claude Code + LLM judge |
| Visual math | MathVision no-CI/with-CI `90.0/94.6` | `90.6/95.7` | CI 是 code interpreter，with-CI 属系统能力 |
| Low-level/chart | BabyVision no-CI/with-CI `65.7/85.6`；CharXiv `83.7/90.2` | CharXiv `84.6/90.6` | 工具增益不能记到视觉 encoder 一项 |
| Document/world | OmniDocBench 1.5 `91.1`；RealWorldQA `85.9`；ERQA `65.5` | RealWorldQA `88.5`；ERQA `72.3`；LVBench `76.6` | 文档、静态空间、具身 QA、长视频是四种构念 |

这也解释了为何“Qwen3.8 用了哪些 benchmark”没有一张无歧义的表：先选 Max 托管系统、2.4T text-only 权重、27B
VLM 或 Flash-Next，再讨论成绩。官方来源见 [27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) 与
[Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。

### 3.2 Step 5 Preview：把发版页做成 Agent benchmark 集合

| 组别 | 官方 High-effort 项目 |
|---|---|
| Reasoning | GPQA `93.5`；HLE no-tools `46.5`；AA-LCR v1.1 `88.3`；CritPt `20.9` |
| Coding/terminal | DeepSWE 1.1 `67.7`；Terminal 2.1 `85.0`、4.0 `33.3`；CyberGym `84.7`；SciCode `58.9`；RoadmapBench `54.3`；ProgramBench 厂商标作 pass-rate `80.5`（未充分披露是 resolved、Almost 还是 average tests）；SWE-Marathon partial `72.7`；MLS-Lite `40.5`；SWE-Atlas QnA/Test-writing `63.6/50.8` |
| General agent | GDPval-AA v2 Elo `1571`；tau3-Banking `42.5`；Automation AA/public `51.0/44.0`；AA-Briefcase Elo `1417`；Toolathlon `74.1`；MCP-Atlas `85.6`；PresentBench `76.8`；OfficeQA Pro `60.3`；Spreadsheet v2 `29.4`；JobBench `59.0`；APEX `37.8`；DRACO `83.3`；BrowseComp `88.7`；HLE tools `59.4` |
| Computer use / vision | ALE-CLI `29.5`；MMMU-Pro `76.0`；GDP.pdf `14.8` |
| 内部集 | StepCodeBench `49.0`；StepCode-Daily `64.9`；StepCode-General `65.0`；FinStep LiveSearch `74.5`；CorporateValuation `60.6`；FinanceDR `55.8` |

StepCodeBench 披露 553 repos、9 类任务、20 领域、33 语言，并报 `avg@4`。FrontierFinance 是 220 个专家问题、
11,543 rubrics、6 类投资场景。两项 24 小时 demo——H100 kernel 达 508 TFLOPS、自动后训练把一个小模型 AIME
从 53.3 提到 60——是展示性轨迹，不是可推广的标准 benchmark。

### 3.3 DeepSeek-V4.1-Flash：最适合教学 harness 敏感性

**Instruct，max reasoning=100**：

| 组别 | 官方项目 |
|---|---|
| Reasoning | GPQA `90.9`；HLE full `36.8`、text-only `39.1`；Codeforces rating `3471`；MathArena Apex `65.6` |
| Coding/cyber | Terminal 2.1/3/4 `90.6/30.0/31.2`；DeepSWE `74.2`；ProgramBench Almost@1 `20.3`；NL2Repo 模型卡 `64.0`、更新页 `65.4`；CyberGym `88.1`；SEC-Bench Pro `62.8`；ExploitGym `15.3` |
| Tool/agent | HLE tools `63.9`；Automation `54.8`；ALE `31.8`；Chartography tools `78.9`；BabyVision tools `89.6`；ZeroBench-main tools Pass@5 `49.0` |

官方固定 `temperature=1, top_p=.95, max effort`，代码 agent 多用 DeepSeek Harness Minimal + 1M；但 DeepSWE
改用 mini-SWE-agent，SEC 改用 Claude Code。更重要的是官方把同一模型重跑在八种 scaffold：DeepSWE 从 `65.5`
到 `74.2`，Terminal 2.1 从 `84.1` 到 `90.6`。这接近 9pp 的差异**没有换模型权重**，直接证明 agent 榜必须记录
harness。

**Base** 是另一被测对象：AGIEval `83.4`、MMLU-Pro `74.1`、C-Eval `92.1`、MultiLoKo `45.5`、
SimpleQA-Verified `42.3`、SuperGPQA `53.1`、BBH `86.1`、BBEH `27.2`、DROP F1 `87.9`、HellaSwag
`87.2`、BigCodeBench `60.6`、HumanEval `79.4`、GSM8K `93.0`、MATH `61.1`、MGSM `80.2`、
LongBench-v2 `45.2`、MMMU-Pro `56.5`、CVBench `77.9`、DocVQA `95.6`、RefCOCO avg `86.0`。
这些多为 few-shot base probing，不能和 Instruct agent 分数串成同一能力曲线。

### 3.4 Kimi K3：1M 上下文必须进入真实工作流才有意义

| 组别 | 官方 max-effort 项目 |
|---|---|
| Reasoning/search | GPQA `93.5`；CritPt `23.4`；AA-LCR 厂商旧快照 `74.7`（截至 2026-07-23，表内未披露 revision；不得与当前 v1.1 的 `88.7` 相减）；HLE full no-tools/tools `43.5/56.0`；BrowseComp `91.2`；DeepSearchQA F1 `95.0`；ResearchRubrics `76.2` |
| Coding | DeepSWE `67.5`；ProgramBench average test pass `77.8`；Terminal 2.1 `88.3`；FrontierSWE v1 dominance `81.2`；SWE-Marathon H20-calibrated pre-v1.1 branch `42.0`；PostTrainBench `36.6`；MLS-Lite `48.3`；SciCode `58.7`；内部 Kimi Code Bench `72.9` |
| Working agent | GDPval-AA Elo `1686`；Toolathlon `76.5`；MCPMark `94.5`；MCP-Atlas `84.2`；Automation `30.8`；JobBench `54.3`；AA-Briefcase Elo `1548`；ALE `28.3`；APEX `41.0`；OfficeQA `63.3`；SpreadsheetBench2 `34.8`；OSWorld-Verified `84.8`；OSWorld2 `58.3`；SaaS-Bench `60.1`；tau3-Banking `33.4` |
| 专业工作 | Harvey Lab-AA `94.6`；CorpFin v2 `71.6`；Finance Agent v2 `54.4`；Legal Research `44.2` |
| Vision | WorldVQA `51.0`；OmniDocBench `91.1`；PerceptionBench public dataset、B-vendor run `58.5`；Video-MME subtitle `90.0`；MMVU `82.1`；BabyVision with Python `85.7`；MMMU-Pro `81.6/83.4`、CharXiv-RQ `84.8/91.3`、MathVision `94.3/97.8`、ZeroBench Pass@5 `23/41`（无/有工具） |

Kimi 提供了很有价值的 context-management 对照：BrowseComp 在 300K 触发压缩得 `91.2`，直接使用 1M、不管理
上下文为 `90.4`。窗口更大不自动更强；选择、压缩和保留状态仍是算法。其 DeepSWE 使用 Kimi Code，而竞品常取各自
最佳 harness；多模多数为 3 runs，工具列测的是系统而非裸 VLM。模型卡把 PerceptionBench 称作 in-house，但题集、
judge 与分项表现已公开；因此这里把**定义**记为 public，把 `58.5` 这次未附逐题 artifact 的运行仍记为 B-vendor。

### 3.5 GLM-5.3 / 5.3-Flash：同厂也要先分 checkpoint

旗舰 GLM-5.3 是 text-only，官方重点结果为 Terminal 2.1 `88.2`、Terminal 3 `28.3`、DeepSWE `66.9`、
ALE `28.5`、Automation `48.2`、HLE tools `62.5`、GDPval-AA v2 Elo `1769`。

最新 GLM-5.3-Flash 是原生多模态新 base，项目为 Terminal 2.1 `84.3`、DeepSWE `63.4`、NL2Repo `56.3`、
Toolathlon `78.4`、Automation v1.0.6 `48.8`、ALE `26.3`、HLE tools `55.3`、GDPval-AA v2 Elo `1773`、
OfficeQA `62.4`、CharXiv-RQ tools `89.4`、Chartography tools `78.0`、BabyVision `53.4`、MVBench `77.8`、
MMVU `80.5`。

Flash 的合同披露较细：HLE full，300K+context management，163,840 max output；NL2Repo 1M 且有 anti-hack；
DeepSWE 6 小时/400K；Terminal 2.1 用 Claude Code、6 小时；Toolathlon 三次独立 Pass@1；Automation 固定
v1.0.6 和 null-type 修复。对教学而言，这些脚注比 `84.3` 本身更重要。

### 3.6 GPT-6 Astra：从答案题进入 computer-use 与专业工作

| 组别 | 官方 headline | 证据边界 |
|---|---|---|
| Computer use | ALE `59.3`；OSWorld 2.0 offline partial `72.6`；ScreenSpot-Pro no-tools `92.7` | ALE 是完整系统；ScreenSpot 只证明截图定位，不证明任务闭环 |
| Professional | Automation `41.4`；BenchCAD `95.9`；BrowseComp `91.5`；OpenScore quartets `0.84`；内部 design `50.0`、data science `40.9` | internal 只作 C 级方向证据；BenchCAD 与 Claude 的修订子集不可裸比 |
| Coding/science | Terminal 4 `57.9`；DeepSWE `74.1`；FrontierCode Ext/Main `64.5/53.3`；Terminal-Science `64.6` | research/API harness，不等于 ChatGPT 默认产品 |
| Reasoning | FrontierMath T4v2 `97.6`；GPQA `96.0`；HLE tools `57.2`；ARC-AGI 1/2/3 `98.5/95.0/99.9` | 高度饱和时更应看版本、工具与 error slice |
| Long context | MRCR v2 8-needle：256–512K `100`、512K–1M `96.3` | 是相似对话检索与复现，不是百万 token 长程综合 |
| Health/science | HealthBench Pro length-adjusted `63.4`；GeneBench Pro `37.1`；LifeSciBench `60.3` | raw HealthBench Pro 是 `69.5`，长度校正防止啰嗦获奖 |

OpenAI 声明表中常取**任一 reasoning effort 的最大值**，且运行于 research/API 环境。FrontierCode 加了接近
Codex 的 developer message；ExploitGym 取消常见 6 小时时限；这些均属于 score identity。

### 3.7 Google Gemini 3.8：Flash 与 Live 是两条评测线

Gemini 3.8 Flash 是截至快照日最新通用 Flash 工作模型；它强调质量—延迟—价格 Pareto，而不是声称取代尚未发布的
3.5 Pro。官方同页的 headline 为：

| 组别 | Gemini 3.8 Flash | 关注点 |
|---|---:|---|
| Coding | DeepSWE 1.1 `73.7`；Terminal 2.1 `89.4`；Terminal 4 `19.1` | 2.1 与 4.0 难度/任务不同，不能解释成回退 |
| Professional | GDPval-AA v2 Elo `1545`；Finance Agent v2 `61.4`；Legal Agent all-pass `10.0`；GDP.PDF all-pass `35.0` | Elo、all-pass 与 rubric average 必须分开 |
| Multimodal | CharXiv no-tools `86.2`；LVBench agentic/static `87.8/87.1`；OSWorld 2 partial with batch tools `59.0` | agentic video 会主动选择片段；观察策略是系统的一部分 |
| Reasoning/science | HLE-Verified `54.9`；BioMystery human-solvable/difficult `88.8/56.5`；LABBench2 `86.2` | verified 版本和生物实验 harness 必须固定 |

Google 还发布了 3.8 Live 与 Live Extended Thinking：Artificial Analysis Speech-to-Speech Quality Index `82.6`、
τ-Voice `68.6`、τ-Voice-banking `35.1`。这些分数属于带 ASR/TTS、打断、延迟和对话状态的实时语音系统，不能与
文本 τ-bench 或 Gemini 3.8 Flash 的 terminal 分数合并。官方来源为
[Gemini model hub](https://deepmind.google/models/gemini/)、
[Gemini 3.8 Live release](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-8-live-gemini-3-8-live-extended-thinking/)
与[model-card index](https://deepmind.google/models/model-cards/)。

### 3.8 Claude Fable/Mythos 5.1：模型、路由与 safeguard 是不同对象

除另注外，系统卡多用 adaptive thinking max、默认 sampling、五次平均、最高 1M context。

| 组别 | 官方项目 |
|---|---|
| Coding | SWE-bench Pro `81.2`；Multilingual `89.1`；Multimodal `54.7`；DeepSWE `67.4`；FrontierCode Ext/Main `63.6/50.9`；FrontierSWE v2 `56.3`；Terminal 4 Fable/Mythos `55.8/60.9`；Terminal-Science `52.6` |
| Reasoning/research | CritPt-Corrected internal revision mean@16 `88.4`（专家修订 31/71 个题面，不是 public CritPt score）；ArXivMath no-tools/tools `91.33/93.88`；HLE no-tools/tools `60.9/65.0`；DRACO `87.7`；ARC-AGI 1/2 `97.5/90.0` |
| Long program | ProgramBench filtered 166-task average hidden-test pass `87.6`；单/固定五 agent/动态 subagent 对照 | 不是 87.6% fully resolved；五 agent 降低 latency 但增加总 token/cost |
| Vision/GUI | Chartography no-tools/tools `42.6/86.2`；BenchCAD `0.437/0.843`；OSWorld 2 partial/strict `77.9/41.7`；GDP.pdf `85.4` | crop/Python 能翻倍；subset、grader 与 task release 必须绑定 |
| Professional | OfficeQA `80.2`、Pro `69.0`；Legal core all-pass/criterion `19.09/90.81`；GDPval-AA Elo `1853`；Briefcase Elo `1694`；Toolathlon Pass@1/Pass@3/Pass³ `77.8/81.5/73.1`；Automation `31.4` | all-pass 与 criterion-pass 差 70pp，正是“完成”与“部分进展”的差别 |

Fable 与 Mythos 同底座但安全策略不同；部分 cyber/bio 请求会 fallback 到旧模型。fallback 是部署系统的正常能力，
但含 fallback 的分数不能再解释为单 checkpoint 水位。DRACO 更换 judge 可移动 10–25pp；OfficeQA 的 extracted text
与直接读 PDF image 也是两个任务。CritPt-Corrected 还同时改变题面、内部 task revision、工具与 16 次尝试聚合；它与
public CritPt 的 70-test-challenge 独立榜回答的是“修正后题面可解性”，不能拿 `88.4-20.9` 当模型代际增益。

### 3.9 Grok 4.7：工程 benchmark 很集中，通用 VLM/长上下文证据仍有空白

| 组别 | 官方项目 |
|---|---|
| Engineering | CursorBench 4 high/xhigh `43.9/46.3`；DeepSWE `71.0`；Terminal 4 `38.0`；FrontierSWE v2：xAI 卡 `29.0`、公共榜快照 `29.5`；SWE-Marathon `46.0`；Legal Agent `19.6`；CADGen `44.4` |
| Electrical/CAD | EEBench 发布页 `64.0`、模型卡 xhigh `66.0`；CADGen `44.4` | 官方源自相矛盾，必须保留两个值而非择高 |
| Professional/health | AA-Briefcase v1.1 Elo `1657`；HealthBench Pro `56.7` | judge 与 length adjustment 不同于 GPT/Claude |
| Bio/cyber | LatchBio aggregate `44.5`；CyberGym `80.3`；CVE-Bench xhigh/high `36.6/37.7`；多套 refusal/surveillance/function/safety 指标 | capability 与 refusal 方向相反；此处 high 反高于 xhigh，不能把两数解释成随机区间；internal safety set 不进公共 SOTA 表 |

Grok 4.7 支持 500K 与图像输入，但本次卡没有 MRCR/LongBench、GPQA/HLE/FrontierMath、MMMU/DocVQA/Video 等
标准证据。因此只能说**接口支持**，不能说“有效 500K 推理”或“通用视觉已到某水位”。模型使用匿名 Cursor workflow
数据并原生适配 Grok Bot harness，Cursor/Grok Build 结果应视为模型—脚手架联合优化。

### 3.10 Tencent Hy4 preview：通用基模，不是 Hunyuan 生成模型合集

| 组别 | 官方 appendix 项目 |
|---|---|
| Coding | SWE multilingual `82.9`、Pro `65.7`；DeepSWE `64.3`；SWE-Atlas QnA/Test/Refactor `64.0/57.8/53.3`；SWE-Marathon `31.9`；Terminal 2.1 `85.4`；NL2Repo `58.9`；CyberGym `78.4`；ProgramBench `17.5`（appendix 未标明三种 metric 中哪一种）；PostTrain `35.6`；Harbor-Index `39.6` |
| Search | WideSearch `83.9`；`$OneMillion-Bench` tools `65.4`；DRACO `77.2`；内部 LifeSearch `49.2`、BrowseComp-Pro2 `56.1` |
| Working agent | OfficeQA `66.2`；MCP-Atlas public `83.7`；Toolathlon `74.1`；APEX `37.1`；SkillsBench `62.9`；Job `61.7`；Workspace `60.2`；ALE-CLI `22.8`；GDPval Elo `1678`；Automation `32.1`；BankerToolBench `78.6` |
| STEM/reasoning | BioMystery `71.3`；HLE tools text-only `55.4`；CritPt `16.9`；GPQA `92.3`；HLE no-tools text-only `43.4`；SUPERChem `66.4`；ArXivMath `66.6`；HorizonMath pass@4 `8.8`；MathArena `74.2`；BrokenArXiv `54.6` |
| 内部工作 | Hy-Backend2 `35.2`；Hy-SWE-Max `64.2`；Hy-Company `62.4`；E-Bench/E-Bench-Code `77.1/79.0`；Hy-FinAgent `79.7`；Hy-Finmodel `57.0` | C-internal |

Hy4 的 appendix 记录了不少关键预算：Terminal 2.1 用 Claude Code、500 turns、12 小时；SWE-Marathon 重复到
8 个 valid runs；MCP-Atlas public 500/100 calls；ALE-CLI 是 105 题、12 小时；Toolathlon 是 108 verified、
100 steps、3-run Pass@1。数字只有连同这些条件才有意义。

### 3.11 MiniMax H3：生成 benchmark 必须另开坐标系

H3 是统一视频—音频生成模型，初始开放重点为 FL2VA/Ref2VA。它不应该与上面 GPQA、SWE-bench 或
OSWorld 模型做一个“综合智能”总分。课程将它放在
[MiniMax H3 capstone](../../05-multimodal-understanding-generation/minimax-h3-capstone/README.md)，按三层证据记录：

- 客观代理：时长/FPS、首尾帧误差、flicker、音频能量与视觉事件对齐；
- 固定 prompt 的盲评：指令、镜头、运动、视觉质量、音质、音画同步分栏；
- 系统合同：FL2VA/Ref2VA、768p 本地边界、scheduler、seed、revision、GPU、峰值显存与 artifact hash。

VBench/FVD/FAD/CLAP 不能单独证明观感与产品质量，详见[生成分册](05-multimodal-generation.md)。

## 4. “最高水位”表：先分 A/B/C，再看数值

### 4.1 可作为 A-public 时间戳的例子

| Benchmark@快照 | 当前公开候选 | Metric/规模 | 能支持的结论 |
|---|---|---|---|
| FrontierSWE v2 @ 2026-09-22 | GPT-6 Astra `65.5 ± 8.9`；Fable 5.1 `56.3 ± 11.1` | 34 tasks、5 trials、20h，mean partial；`±` 样式表示五次 trial 的 worst@5–best@5 范围 | 在该榜的统一 Proximus 合同下，Astra 当前领先；whisker 不是 CI/SE，若要统计不确定性需另算 task-cluster/bootstrap interval |
| SWE-Marathon v1.1 @ 2026-09-22 | Opus 5 `50.0`；Kimi K3 `48.1`；Fable 5.1 `45.6` | 20 tasks、8 trials，binary all-verifier resolution | 这是完整交付率；厂商表的 partial 不能加入 |
| ProgramBench public @ 2026-09-22 | Opus 5 resolved `4.5`、Almost `37.0`、average tests `74.7` | 200 tasks、248K+ tests，三种 metric | 同一模型三种数字揭示“局部进展 ≠ 完成交付” |
| AA-LCR v1.1 @ 2026-09-22 | Kimi K3 `88.7`、Step 5 `88.3` 等 | 长上下文 reasoning score | 只属于该版本与评测 provider；不证明所有 1M 工作流 |

公共榜也不是永恒真值：每行仍需保存榜单日期、agent version、provider、失败 task 与 raw artifact。

两个同名冲突必须原样保留：Grok 4.7 的 xAI model card 报 FrontierSWE v2 `29.0`，同日查看公共榜为
`29.5`（并显示 worst@5–best@5 范围 `±12.6`）；Kimi K3 的 SWE-Marathon `42.0` 来自 2026-07-09 的
H20-calibrated、最终 v1.1 之前分支，而公共 v1.1/Claude Code/8-trial 榜为 `48.1`。这不是“复跑误差”，而是
source snapshot 或 benchmark contract 已变化。

### 4.2 B-vendor 候选：可理解关注方向，不宜宣布全球 SOTA

| Benchmark | 看似最高的正式发布数字 | 为什么不直接排 |
|---|---|---|
| GPQA Diamond | GPT-6 Astra `96.0` | 小样本高分区；prompt、effort、tool/no-tool 与 sampling 未完全统一 |
| HLE no-tools | Claude Fable 5.1 `60.9` full | Step/Hy4 等明确是 text-only；judge、题集 revision 与 token cap 不同 |
| HLE tools | Fable 5.1 `65.0` | search/fetch/code、context management、judge 和工具失败分母不同 |
| DeepSWE 1.1 | DeepSeek `74.2` / Astra `74.1` | 同一 DeepSeek 权重仅换 scaffold 已差约 9pp |
| Terminal 2.1 | DeepSeek `90.6` | Claude Code/Kimi Code/DS harness、网络、500 steps、6/12h 不同 |
| Terminal 4 | Mythos `60.9`、Astra `57.9`、Fable `55.8` | Mythos 受限访问；trials 与 harness 不同 |
| OSWorld 2 | Fable partial `77.9`、Astra offline partial `72.6` | task release、offline subset、grader、partial/strict 不同 |
| HealthBench Pro | Astra length-adjusted `63.4` | Claude/Grok 的 judge、长度惩罚、安全拒答计零合同不同 |
| Toolathlon | 各厂约 `72–78`，另有 Pass@3/Pass³ | null attempt、内部 MCP 重实现、step budget 与 trials 不同 |
| GDPval-AA | Fable Elo `1853`、GLM 约 `1770` | Elo 随对手池、日期与 judge 变化，不是静态准确率 |
| MMMU-Pro | Kimi no-tools/tools `81.6/83.4`；Step `76.0` | standard/vision、Python/tools、图片顺序和 judge 未统一 |
| MathVision | Kimi no-tools/tools `94.3/97.8`；Qwen Flash-Next `90.6/95.7` | code interpreter 改变被测系统；corrected annotation 也须绑定 |
| CharXiv-RQ | Kimi `84.8/91.3`；Qwen Flash-Next `84.6/90.6` | crop/Python、validation/full 与 judge 会移动数十分 |
| 多模态生成 | 暂不给跨厂单一水位 | prompt/seed、分辨率/时长、采样器、候选选择和盲评池未统一；用分维度盲评 |

### 4.3 三类数字禁止进入“最高水位”主表

- **内部题集**：QwenSWE、StepCode、Hy-SWE、GPT internal design、Grok internal safety；它们可说明研发方向，不能支持外部排名。
- **不同 metric 共用百分号**：ProgramBench average vs Almost vs resolved；SWE-Marathon partial vs binary；ALE pass vs partial score。
- **不同被测系统**：裸 checkpoint、带工具模型、native harness、fallback router、多 agent；除非问题本来就是比较整套系统。

## 5. 八个决定结论的“反排行榜”案例

1. **DeepSWE scaffold delta**：同一 DeepSeek-V4.1-Flash 在不同 agent 中约 `65.5–74.2`；模型没变，headline 可动近 9pp。
2. **ProgramBench metric delta**：公共榜 fully resolved 远低于 average hidden-test pass；`80.5%` 若不写 metric 几乎没有意义。
3. **FrontierCode effort reversal**：更高 effort 提高 task correctness，却可能因乱改 docs/CI/邻近文件降低总 composite。
4. **OfficeQA representation delta**：extracted text 与 PDF image 不是同一输入；OCR/布局错误会改变构念。
5. **Chartography tool delta**：Fable 从 no-tools `42.6` 到 crop+Python `86.2`；这是系统可用性，不是裸视觉翻倍。
6. **Kimi context-management delta**：300K+压缩 `91.2` 高于不管理的 1M `90.4`；窗口是容量，选择策略才是有效记忆。
7. **Claude fallback identity**：路由提高部署可用性，却使结果不再能归于 Fable 单 checkpoint。
8. **Grok 官方冲突**：同次发版 EEBench 页面 `64.0`、模型卡 `66.0`；证据账应保留冲突，不能静默择高。

## 6. 发版成绩的最小数据库键

一行成绩的唯一键至少应是：

```text
model/checkpoint-or-api-id
+ benchmark@revision/split
+ metric/aggregation/direction
+ model-mode/reasoning-effort
+ prompt/template/harness
+ tools/context/output
+ time/step/cost budget
+ sampling/trials/retries
+ judge/verifier/environment
+ timeout/refusal/fallback/failure denominator
+ evaluated_at/source_owner
```

如果缺字段，结论降级：

- 字段齐、公共任务和 raw runs 可复验：A-public candidate；
- 正式模型卡但协议/逐题结果不完整：B-vendor；
- 内部数据、内部 judge 或只有宣传图：C-internal；
- 同名不同合同：`NOT_COMPARABLE`，不计算差值。

## 7. 一手来源

- Qwen：[Qwen3.8-2.4T-A95B model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)、[Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8)。
- StepFun：[Step 5 Preview](https://www.stepfun.com/step-5-preview)。页面未显示明确发布日期；本页只称截至快照日最新。
- DeepSeek：[DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)、[API updates](https://api-docs.deepseek.com/updates/)。
- Moonshot：[Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)、[official blog](https://www.kimi.ai/blog/kimi-k3)、[open-weight announcement](https://www.kimi.com/news/kimi-k3-open-source)。
- Z.ai：[GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash)、[GLM-5 repository](https://github.com/zai-org/GLM-5)。
- OpenAI：[GPT-6 Astra release](https://openai.com/index/gpt-6-astra/)、[system card](https://deploymentsafety.openai.com/gpt-6-astra)、[API model](https://developers.openai.com/api/docs/models/gpt-6-astra)。
- Google：[Gemini model hub](https://deepmind.google/models/gemini/)、[model-card index](https://deepmind.google/models/model-cards/)、[Gemini 3.8 Live release](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-8-live-gemini-3-8-live-extended-thinking/)。
- Anthropic：[Claude Fable/Mythos 5.1 release](https://www.anthropic.com/claude-fable-and-mythos-5-1)、[system card PDF](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20%26%20Claude%20Mythos%205.1%20System%20Card.pdf)。
- SpaceXAI：[Grok 4.7 release](https://x.ai/news/grok-4-7)、[model card PDF](https://media.x.ai/v1/website/card4p7-3a96f40b.pdf)、[API model](https://docs.x.ai/developers/models/grok-4.7)。
- Tencent：[Hy4 preview repository](https://github.com/Tencent-Hunyuan/Hy4-preview)、[official release](https://www.tencent.com/tencent-releases-and-open-sources-tencent-hy4-preview/)、[benchmark appendix](https://raw.githubusercontent.com/Tencent-Hunyuan/Hy4-preview/main/assets/benchmark-appendix.jpg)。
- MiniMax：[H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)。
- 独立动态榜：[FrontierSWE v2](https://www.frontierswe.com/)、[SWE-Marathon](https://www.swe-marathon.org/)、[ProgramBench](https://programbench.com/)、[AA-LCR](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning)。

## 费曼自检

1. 为什么 DeepSeek 的 DeepSWE `74.2` 与 Astra 的 `74.1` 不能据此断言前者模型权重更强？
2. 为什么 Kimi/Step 的 ProgramBench `70–80` 级数字与公共榜 `4.5% resolved` 可以同时为真？
3. 为什么 GPT-6 Astra 的 MRCR 1M 高分与 Kimi 的 BrowseComp context-management 对照回答了不同问题？
4. 如果一个模型在 HLE tools、Toolathlon、OSWorld 都提升，怎样判断是权重、harness 还是工具系统贡献？
5. 你会怎样向管理层汇报“当前最高水位”，既简洁又不制造虚假精确性？

<details>
<summary>参考答案</summary>

1. 两行的 harness、预算、重试和运行环境并未锁成同一合同；DeepSeek 还公开证明同一权重仅换 scaffold 可移动近 9pp。应先统一 agent 做 paired replay，再估计 checkpoint effect。
2. 它们可能分别是 average hidden-test pass、Almost@1 或 partial，而 `4.5%` 是 200 题全部测试通过的 fully resolved。分子、聚合和成功条件不同，共用百分号不代表同一量。
3. MRCR 测大量相似上下文中定位并复现指定答案；BrowseComp 是搜索、证据综合和上下文管理的 agent 系统。前者隔离 retrieval fidelity，后者包含行动与压缩策略。
4. 做四格或逐级消融：固定模型换 harness；固定 harness 换模型；逐一关闭搜索/代码/视觉；锁 token/time/attempt/failure denominator，并保存逐题 paired outcome。只有 matched delta 才可归因。
5. 分 A-public、B-vendor、C-internal 三栏；每个候选只报 benchmark revision、metric、系统身份、预算与日期，并显式写“不同协议不可合并”。对业务决策再加成本、失败率和自家 fresh holdout，而不是给一个总榜名次。

</details>

一句话验收：**最新发版越来越测“模型 × harness × 工具 × 环境 × 预算”的联合系统；完整 benchmark 材料的价值，是把这个乘号重新拆开。**
