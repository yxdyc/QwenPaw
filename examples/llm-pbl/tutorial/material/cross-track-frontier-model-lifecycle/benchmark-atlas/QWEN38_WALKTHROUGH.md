# Qwen3.8 发版到底用了哪些 benchmark

> **快照日期**：2026-09-22。
> **一句话答案**：Qwen3.8 不是一个可直接抄成单列排行榜的模型；发版材料同时覆盖 Max 托管系统、2.4T-A95B text-only 开放权重、27B dense VLM 与 Flash-Next 架构预览。先确定被测对象，再读 benchmark 数字。
> **证据边界**：下列分数来自 Qwen 官方 model card，属于厂商正式运行结果；除非另有公共榜复验，不能自动视为独立公共 SOTA。课程合成例题均为新写的题型说明，不复制正式测试题。

## 1. 先分清四个被测对象

| 对象 | 身份与能力边界 | 上下文与推理模式 | 本页怎样使用它的分数 |
|---|---|---|---|
| **Qwen3.8-Max** | 基于 2.4T-A95B 的托管系统；另有视觉输入、non-thinking、官方内置工具和产品级服务能力 | 默认 1M；具体发版表混合 Claude Code、OpenCode、Qwen-Agent、工具与 judge | [2.4T-A95B 卡][max-card]里的 benchmark 表实际以 **Qwen3.8-Max** 为被测列；表中分数不能自动归给下载权重 |
| **Qwen3.8-2.4T-A95B** | 2.4T total / 95B activated 的开放权重 causal LM；**text-only、thinking-only** | 原生 262,144，官方称可扩至 1,010,000；`xhigh` 为默认 reasoning effort | 官方卡没有给开放 checkpoint 单独一列 benchmark 数字；本页只记录身份，不把 Max 的视觉、工具或得分搬给它 |
| **Qwen3.8-27B** | Apache-2.0、27B dense、原生 image/video 的开放 VLM | 原生 262,144，可扩至 1M；thinking 默认开启但可关闭 | 使用 [27B 卡][vl-card]自身 Text Performance 与 VL Performance 两张表 |
| **Qwen3.8-Flash-Next** | 125B total / 6B activated，另有 51B n-gram embedding 与 4B MTP；原生 VLM；是 Qwen4 路线的实验性架构预览 | 原生 262,144，可扩至 1M；thinking 可控 | 使用 [Flash-Next 卡][flash-card]的 Language 与 Vision Language 两张表；托管 `Qwen3.8-Flash` 是基于它、带更多产品能力的另一系统身份 |

最重要的防错句是：**Max 分数是托管系统分；2.4T-A95B 是开放 checkpoint 身份；两者有谱系关系，但不是同一个测试合同。**

## 2. ROI 最高的阅读顺序

如果只有十分钟，按下面顺序读：

1. **先看上面的身份表**：避免把 Max 的 vision、工具和 1M 默认服务能力写到 2.4T 开放权重上。
2. **再看代码表的脚手架**：Terminal、SWE-Pro、DeepSWE、NL2Repo 的 headline 都是模型—harness—预算联合结果。
3. **看同 benchmark 的三 checkpoint 列**：27B 与 Flash-Next 的官方表更接近同一发布方的配对运行，但仍可能换 harness、prompt 或 corrected set。
4. **看 Agent 的 pass 与 partial**：ALE 的 `Pass@1` 和 `Score`、ClawEval 的 `Pass@3` 和三次平均回答不同问题。
5. **最后才看视觉工具增益**：MathVision、BabyVision、CharXiv 中的 `CI` 是 code interpreter，不是 confidence interval。
6. **内部 benchmark 放最后**：它们能显示产品目标，却不能进入公共 SOTA 主表。

本页统一写法：`—` 表示该 checkpoint 的官方表没有报告；`N 未披露` 表示 Qwen 卡没有给本次运行的题数，绝不从相近版本反推。

## 3. Coding：从算法题到多小时工程

### 3.1 公共或外部 benchmark

| Benchmark | 能力与体量 | Metric | 课程合成例题 | Qwen 协议与分数 | 局限与详细卡 |
|---|---|---|---|---|---|
| [Terminal-Bench 2.1](https://www.tbench.ai/) | 隔离终端中的构建、debug、系统管理；v2.1 为 **89 题** | 严格 task success；多 trial 时必须写聚合 | 修复离线容器的依赖锁定与健康检查，使最终 verifier 全绿 | **Max 86.6**：Claude Code、`avg@10`、5h、`max_tokens=131,072`；**27B 73.0**：卡标 `Terminus`，其余预算未披露；Flash-Next `—` | 版本、镜像、网络和 timeout 可主导结果；Max 与 27B 不是同 harness。[Max 卡][max-card]、[27B 卡][vl-card] |
| [SWE-bench Pro](https://arxiv.org/abs/2509.16941) | 跨文件专业 repo issue；论文总集 **1,865 题、41 repos**，Qwen refined set 的精确 N 未披露 | resolved / Pass@1 | 修复权限继承 bug，同时更新 migration、API、缓存和回归测试 | **Max 67.7；27B 61.7；Flash-Next 62.5**。均用 Claude Code、`temp=1.0`、`top_p=.95`、256K；Qwen 修正 problematic tasks 后重跑 baseline | 不能与原始 public、held-out、commercial 或其他 corrected set 裸比；这是 agent 系统分。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [DeepSWE 1.1](https://deepswe.datacurve.ai/) | 原创长程 repo engineering；**113 题、91 repos、5 种语言** | 所有 fail-to-pass 与 pass-to-pass tests 全绿才 resolved | 为异步库加入可取消读取，处理 shutdown、timer 和 multipart 交互 | **Max 56.6**：Claude Code 与 mini-SWE-agent 取高者，最好来自 Claude Code；**27B 42.2**：Claude Code；**Flash-Next 58.7**：两 harness 取高者，最好来自 mini-SWE-agent。共同披露 `temp=1.0`、`top_p=.95`、256K | “取两个 harness 最大值”不是单一固定系统；跨卡相减不能归因于权重。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [NL2Repo-Bench](https://arxiv.org/abs/2512.12730) | 从规格和空目录生成完整 Python repo；**104 题、9 类领域**，平均规格约 18.8K tokens | 平均 hidden-test pass；fully-correct repo 是另一指标 | 从需求文档实现带持久化、CLI、重试和插件接口的可安装包 | **Max 55.9；27B 42.3；Flash-Next 48.1**。Claude Code；禁用 `pip download/install`、`git clone` 等直接获取目标 repo 的 Bash 行为 | 平均测试通过率不是 repo 完成率；anti-hack 规则、依赖与测试覆盖决定解释。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [FrontierSWE v1](https://www.frontierswe.com/v1) | 多小时开放工程与研究；v1 **17 题** | 相对于当时对手池的 **dominance**，不是 v2 partial reward | 在固定算力内训练视频轨迹预测器并提交可复现实验 | **Max 73.5 dominance**：Claude Code；官方卡从 2026-08-03 的 v1 raw scores 用官方脚本重算 dominance | 不能与 v2 的 34-task `mean@5` 或“完成率”合并；对手池变化也会改 dominance。[Max 卡][max-card] |
| MLS-Bench-Lite | 长程机器学习软件任务；本次 N 未披露 | 卡只给 scalar score，精确聚合未披露 | 修复训练管线的 checkpoint 恢复，并让隐藏回归测试通过 | **Max 41.0**：Claude Code、5h、`max_tokens=131,072`；27B/Flash-Next `—` | 名称中的 Lite、task snapshot 与 leaderboard revision 必须一起保存；不可猜成 resolved rate。[Max 卡][max-card] |
| [PaperBench](https://openai.com/index/paperbench/) | 完整基准以 **20 篇论文、8,316 个 rubric nodes**为核心；但 Qwen 跑的是 **Code-Dev** 变体，只评代码开发，跳过单独执行 submission 与复现实验结果的阶段 | Code-Dev requirements 的 rubric aggregate；3 runs 平均 | 从论文实现训练/评测代码并提交可审查仓库；本合同不要求真的跑出论文表格 | **Max 93.0**：BasicAgent、Code-Dev、Claude Opus 4.6 judge、每次最多 12h | 93.0 不是“20 篇论文中 93% 完整复现”，也不能与完整 PaperBench 原始 `21.0%` 做减法；mode、judge、依赖和 rubric revision 都要锁定。[Max 卡][max-card]、[官方 Code-Dev 说明](https://github.com/openai/frontier-evals/blob/main/project/paperbench/README.md#paperbench-code-dev) |
| AndroidBench | 移动端 agent；Qwen 明确跑 **95-task public subset** | `avg@3` task score | 从短信提取地址，在地图收藏，再创建带正确时区的日历事件 | **Max 75.1**；27B/Flash-Next `—` | 不要与 116-task AndroidWorld 或其他 AndroidBench snapshot 合并；设备镜像与动作接口未在顶层表完全披露。[Max 卡][max-card] |
| [SWE-bench Multilingual](https://www.swebench.com/multilingual.html) | 跨语言 repo issue；**300 题、9 种语言** | resolved / Pass@1 | 在 Rust 项目修复并发边界，并保持既有跨平台测试通过 | **Flash-Next 81.0**；Flash 卡同表列 **27B 73.8**；Max `—`。mini-SWE-agent、`temp=1.0`、`top_p=.95`、256K | 跨语言平均会隐藏单语言退化；27B 值来源于 Flash 卡的配对表，而非 27B 卡原表。[Flash 卡][flash-card] |
| [LiveCodeBench v6](https://github.com/LiveCodeBench/LiveCodeBench) | 新近竞赛代码；v6 full code-generation **1,055 题** | 执行测试的 pass@1；卡未写 lite/full 与采样细节 | 实现动态图连通查询，通过隐藏时间与内存限制 | **27B 90.3；Flash-Next 91.9**；Max `—` | 算法题是低成本 code reasoning probe，不证明 repo agent 能力；需锁 release、日期窗口和抽取器。[27B 卡][vl-card]、[Flash 卡][flash-card] |

### 3.2 Qwen 内部 coding benchmark

| Benchmark | 能力与体量 | Metric | 课程合成例题 | Qwen 协议与分数 | 局限与详细卡 |
|---|---|---|---|---|---|
| QwenSWEBench | 软件工程；**内部题集，N 未披露** | `avg@3` | 在内部服务 repo 中修复跨模块缓存失效并补回归测试 | **Max 80.7**：Claude Code、8h、`max_tokens=32,768`、`temp=1.0`、256K；**27B 79.0**：同类协议；Flash-Next `—` | C-internal；题目、污染与完整 verifier 不可外审，不能写成公共 SWE SOTA。[Max 卡][max-card]、[27B 卡][vl-card] |
| QwenQoderBench | Qoder 用户体验与 coding workflow；**内部，N 未披露** | `avg@5` | 根据 IDE 中的多文件上下文完成重构并解释风险 | **Max 58.4**：Claude Code、6h、`max_tokens=32,768`、`temp=1.0`、256K | 只支持“该内部产品合同下”的结论；不能与外部 resolved rate 混列。[Max 卡][max-card] |
| QwenReactBench | React 项目构建；双语、**7 类**，题数未披露 | 自动渲染 + 多模态 judge 的 BT/Elo | 按产品说明实现响应式仪表盘，并由渲染结果检查交互和布局 | **Max Elo 1724**：Claude Code、EN/CN | Elo 随对手池和 judge 变化，不是 1724% 或正确率；内部 prompt 不支持公共排名。[Max 卡][max-card] |
| QwenSVGBench | SVG code generation；双语，题数未披露 | 自动渲染 + 多模态 judge 的 BT/Elo | 生成带图例和标注的可缩放流程图，检查结构与视觉一致性 | **Max Elo 1713** | renderer、字体、对手池和视觉 judge 都是合同；C-internal。[Max 卡][max-card] |

## 4. General Agent：完成任务，不只是答题

这一组按 Max 官方表共有九项；前八项在下表，第九项 `HLE w/ tools` 与无工具 HLE 放在下一节并单列合同。

| Benchmark | 能力与体量 | Metric | 课程合成例题 | Qwen 协议与分数 | 局限与详细卡 |
|---|---|---|---|---|---|
| CoWorkBench | 长程办公/生产力；Qwen 明确称 **in-house**，N 未披露 | 卡给 aggregate score，精确聚合未披露 | 汇总多份财报，更新 spreadsheet，再制作带引用的简报 | **Max 74.8；27B 70.7；Flash-Next 73.9** | C-internal；CS、金融、法律、医疗混合平均会隐藏失败类型，不可宣布公共 SOTA。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card] |
| WorkSpaceBench | 当前公开主集 **388 题、20,476 个文件、7,399 条 rubric**，另有 Lite-100；Qwen snapshot 的精确 manifest 未披露 | task/criterion score，卡未披露精确聚合 | 从共享目录找正确版本，更新表格公式并导出 PDF | **Max 67.7**；27B/Flash-Next `—` | public 总量不等于 Qwen 一定跑了全部；task split、renderer、工具和 judge 未写全；与 JobBench、OfficeQA 不同。[Max 卡][max-card]、[Agent 分册](03-agent-tool-use.md) |
| JobBench | 当前公开库 **130 题、35 个职业**；主榜 65 题、Easy 63 题可分栏，Qwen snapshot 的精确 manifest 未披露 | rubric/task aggregate，卡未披露精确聚合 | 根据客户底稿产出可审计 memo，并满足数字、引用和格式检查 | **Max 53.4；27B 33.4；Flash-Next 55.7** | 不同卡可能换 task subset 或 harness；平均分不等于整份 artifact 可交付。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card]、[Agent 分册](03-agent-tool-use.md) |
| SkillsBench v1.1 | 带可加载 skill 的 agent 任务；**87 题** | 每题 3 runs 的平均 score | 读取一个本地技能说明，调用指定工具完成数据清洗并生成报告 | **Max 70.2**：Qwen 系用 OpenCode；对照模型可能用 Claude Code/Codex；均为 Qwen 自测 | 横向列同时换模型和 harness；只可作产品系统比较。[Max 卡][max-card] |
| [Agents' Last Exam](https://agents-last-exam.org/) | living OS-sandbox 专业任务；完整库持续变化，**Qwen snapshot N 未披露** | `Pass@1 / Score`：满分任务率 / partial grader | 分析财务附件、更新模型、做演示并满足隐藏格式检查 | **Max 27.0 / 52.4；27B 20.4 / 42.9；Flash-Next 24.3 / 51.2** | pass 与 partial 不能择高；OS 镜像、licensed apps、时间与失败分母会改分。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [AutomationBench](https://github.com/zapier/AutomationBench) | **600 public scoring tasks**，6 类业务、47 个模拟 SaaS 工具 | 卡列 `Pass@1`；基准还可分 strict completion 与 partial assertions | 找到正确 lead、更新 CRM、安排会议并通知消息系统 | **Max 27.3**：600-task public subset | public/private、版本、step cap、null/provider error 是否计零必须锁定。[Max 卡][max-card] |
| [Toolathlon Verified](https://toolathlon.xyz/) | **108 题、32 apps、604 tools**，平均约 20 轮 | `Pass@1` | 读邮件附件、核对学习记录、更新成绩并发送通知 | **Max 72.5**；**Flash-Next 73.5**；Flash 卡同表列 **27B 67.1** | tool server、内部 MCP 重实现、step/time budget 与 null attempts 未统一；27B 值来自 Flash 卡配对表。[Max 卡][max-card]、[Flash 卡][flash-card] |
| WideSearch | 大规模搜索与多结果汇总；N 未披露 | 4 runs 的平均 **item-F1** | 搜集多个官方目录中的条目，去重并输出带来源的完整清单 | **Max 81.9**：Qwen 用 Qwen-Agent；外部模型用 Claude Code | 同表 harness 不同；item-F1 不是整项研究任务成功率。[Max 卡][max-card] |

## 5. General 与 long context：十一项合同

这里按发布表的十个 General/Long Context 项，再加单独的 `HLE w/ tools` 系统态，共十一行。这样既不漏项，也不会把带工具和闭卷 HLE 合成一个分数。

| Benchmark | 能力与体量 | Metric | 课程合成例题 | Qwen 协议与分数 | 局限与详细卡 |
|---|---|---|---|---|---|
| [HLE w/ tools](https://epoch.ai/benchmarks/hle) | 跨 100+ 学科；HLE full **2,500 题**，约 14% 含图；Qwen工具子集/工具清单未披露 | 工具系统下的 correctness/judge score | 遇到冷门材料问题时先搜索官方手册，再用代码核算并给短答案 | **Max 56.2**；27B/Flash-Next `—` | 这是 General Agent 第九项；工具、检索源、judge 和 full/text-only 不明时只能作 B-vendor 系统分。[Max 卡][max-card] |
| [GPQA Diamond](https://arxiv.org/abs/2311.12022) | 专家科学推理；**198 题** | 四选一 accuracy / 单次正确率 | 根据反应机理与实验条件判断主产物构型 | **Max 92.6；27B 89.2；Flash-Next 91.7** | 1 题约 0.5pp 且已接近饱和；prompt、CoT、sampling 与污染会主导小差。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [HLE](https://epoch.ai/benchmarks/hle) | 长尾专家知识与推理；**2,500 题、100+ 学科** | correctness / judge；卡中与 `HLE w/ tools` 分栏 | 从严格定义推导唯一短答案，不调用外部工具 | **Max 43.6；27B 30.8；Flash-Next 35.9**。27B/Flash 卡注明 GPT-4o judge | Max 卡没有逐字写 `no-tools`，但它与 `HLE w/ tools` 是不同表项；不得把 43.6→56.2 全归因于权重。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [IFBench](https://github.com/allenai/IFBench) | **300 prompts、83 verifiers**，其中 58 个为新 OOD 约束 | IFBench score；Qwen卡未披露 prompt/constraint-level 与 strict/loose | 写三段，每段两句，第二段禁用某字符，末句以指定数字结尾 | **Max 82.8；27B 79.5；Flash-Next 81.3** | 形式约束通过不代表内容质量；聚合口径未写全时不要与别表小数点排名。[Max 卡][max-card]、[27B 卡][vl-card]、[Flash 卡][flash-card] |
| `$OneMillion-Bench` | 高经济价值专业 agent 任务，并非“百万 token”测试；论文/公开仓库为 **400 题、5 大领域**，官网 leaderboard 概览仍写 200 题，Qwen run 的精确 manifest 未披露 | 加权 rubric 的 **expert score**；公开榜另分 pass rate、average 与 economic value | 检索多份权威资料，解决冲突证据并完成带出处的估值意见 | **Max 52.5**：官方脚注称用 `gemini-3.1-pro-preview` 评估 | 名称中的 OneMillion 指估算专家劳动价值接近 100 万美元，不是上下文长度；公开源自身有 200/400 题版本漂移，必须锁 manifest。[Max 卡][max-card]、[专项分册](06-frontier-specialized.md) |
| [HealthBench](https://openai.com/index/healthbench/) | **5,000 对话、48K+ 医生 rubric criteria** | rubric aggregate；Qwen卡未说明是否 length-adjusted | 对含胸痛危险信号的咨询同时给出急诊建议、追问和安全边界 | **Max 60.2** | LLM judge 与回答长度会影响分数；不是临床结局或诊断准确率。[Max 卡][max-card] |
| [PLawBench](https://aclanthology.org/2026.acl-long.458/) | **850 题、13 个法律场景、约 12,500 条专家 rubric**；咨询、案件分析、文书生成三类 | LLM evaluator 逐 rubric 判断后聚合；Qwen 卡未披露完整 prompt/聚合 | 对比两版法规，列出合同条款需要修改的地方并给依据 | **Max 73.2**：`gemini-3.1-pro-preview` 评估 | 法域/语言分布、judge 与回答预算影响结果；不是带沙箱的法律 agent 完成率。[Max 卡][max-card]、[文本分册](01-text-reasoning.md) |
| [PRBench-Legal](https://www.justicebench.org/dataset/prbench) | PRBench 共 **1,100 题、19,356 条 rubric**，其中法律 500 题、Hard 250 题 | 严重度加权 criterion aggregate；Qwen 卡未披露完整归一化合同 | 根据合同、法规和判例制作逐条风险清单 | **Max 57.6** | criterion average 可能掩盖关键条款遗漏；不可当整份交付通过率。[Max 卡][max-card]、[文本分册](01-text-reasoning.md) |
| [PRBench-Finance](https://www.justicebench.org/dataset/prbench) | 同一 PRBench 中金融 **600 题**、Hard 300 题；总库来自 182 位专家、覆盖 114 国 | 严重度加权 criterion aggregate；Qwen 卡未披露完整归一化合同 | 统一两家公司会计口径后重算现金流并说明调整 | **Max 58.3** | 资料时点、计算工具和 judge 未写全；必须保留完整 benchmark 名，不能只写 `Finance`。[Max 卡][max-card]、[文本分册](01-text-reasoning.md) |
| [MRCR v2 256K, 8-needle](https://github.com/google-deepmind/eval_hub/blob/master/eval_hub/mrcr_v2/README.md) | 在多次相似请求中定位目标轮；题数由生成配置决定，**无固定 N** | 目标回答复现的相似度/正确率 | 20 万 token 中多次提出相同请求，最后复现指定序号那次回答 | **Max 92.9**：256K、8-needle | 主要测定位与复制，不证明跨文档综合；开放 2.4T 的 262K 原生窗口也不能据此获得 Max 分数。[Max 卡][max-card] |
| [LongBench v2](https://github.com/THUDM/LongBench) | 真实长文推理；**503 道四选一**，8K–2M words、六类任务 | accuracy，应按长度和任务类分桶 | 比较多版法规与公司报告，判断政策变化对财务指标的影响 | **Max 66.3** | 截断、RAG、prompt 和实际输入长度未写清时，单一 overall 不能证明 1M 有效推理。[Max 卡][max-card] |

## 6. Qwen3.8-Max VL Performance：托管多模态系统

这一节逐行转录 [Qwen3.8-Max 官方 VL Performance Chart][max-vl-card]。图中实际有 **六组、55 行**；这些都是 **Qwen3.8-Max 托管系统**的厂商运行结果，不能归给 text-only 的 2.4T-A95B 开放权重。图中粗体只表示所展示模型列中的最佳值，不能单独证明公共 SOTA；内部 benchmark 仍按 C-internal 处理。

斜杠必须保留原身份：`MathVision / BabyVision / ZeroBench / CharXiv` 是 **Without CI / With CI**（`CI` 为 code interpreter）；`OSWorld 2.0` 是 **Binary / Partial**；`ClawEval-MM` 是 **Pass@3 / Average**；`OCR-Bench-V2` 是 **EN / ZH**。其余没有清晰脚注的体量或 metric 均明写“未披露”，不从 benchmark 惯例反推。

### 6.1 Multimodal Reasoning（12/12）

| Benchmark | 能力与体量 | Metric | 课程合成题型 | Max 分数与协议 | 局限与详细卡 |
|---|---|---|---|---|---|
| [MMMU-Pro](https://github.com/MMMU-Benchmark/MMMU) | 多学科图文专家推理；公开集常见总量约 1,730，Qwen 本次 manifest 未披露 | 多选 accuracy；图未写 prompt/input-order 细节 | 结合电路图与文字条件判断故障来源 | **82.3**；官方脚注说明 Max 为发布方内部评估 | 压缩、选项顺序和专家学科配比会改分；只按厂商系统分读取。[Max VL 卡][max-vl-card] |
| [MathVision](https://github.com/YerongLi/MathVision) | 视觉数学；full 3,040 题、testmini 304，Qwen split 未披露 | **Without CI / With CI**；数学等价/答案判定 | 从几何图读辅助线，推导角度并输出规范答案 | **95.2 / 97.7**；`CI`=code interpreter；Qwen 使用逐步推理并把最终答案放入 `\boxed{}` 的固定提示；官方称人工核验后修正少量错误标注 | 这是两套系统合同，且 corrected revision、固定提示与 split 都影响结果。[Max VL 卡][max-vl-card] |
| BabyVision | 低层视觉、路径与空间模式；本次 N 未披露 | **Without CI / With CI**；具体聚合未披露 | 沿迷宫箭头追踪路径，判断最终出口 | **82.0 / 91.3**；`CI`=code interpreter | 工具增益不能归因于视觉编码器；图像缩放与 judge 合同未完整披露。[Max VL 卡][max-vl-card] |
| HLE-VL (w/ Tools) | HLE 中含视觉输入的长尾专家题；工具子集精确 N 未披露 | 工具系统 correctness/judge score；精确聚合未披露 | 识别专业仪器图后检索手册并核算唯一答案 | **52.2**；工具包括 code interpreter 与搜索 | 工具调用细节、检索源、VL 子集与 judge 未在图中完整披露；是系统分。[Max VL 卡][max-vl-card] |
| ZeroBench (Pass@5) | 高难视觉推理；本次 N 未披露 | **Pass@5；Without CI / With CI** | 对复杂机械示意图进行五次独立求解，看是否至少一次正确 | **24.0 / 49.0**；斜杠为无/有 code interpreter | Pass@5 受采样数与相关性影响，不等于单次成功率；CI 版不可当裸模型分。[Max VL 卡][max-vl-card] |
| ZeroBench-Sub | ZeroBench 子集；具体子集与 N 未披露 | 官方图仅给 scalar score；metric 未披露 | 对高难空间拼接题给出唯一选项 | **48.5** | 不能与 full-set Pass@5 做减法；subset、k 与 judge 均需锁定。[Max VL 卡][max-vl-card] |
| LogicVista | 图形逻辑与抽象视觉推理；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 根据三幅图的变化规则补全第四幅 | **91.9** | 题型分布、答案抽取与聚合未披露，不能据分数反推错误数。[Max VL 卡][max-vl-card] |
| HiPhO | 高阶图文推理；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 从多对象关系图推断隐藏约束并选择可行状态 | **90.0** | 名称、版本、prompt 与评分细节不足；只保留厂商报告值。[Max VL 卡][max-vl-card] |
| PhyX | 物理图示与现象推理；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 根据受力图判断物体下一时刻的运动方向 | **83.5** | 未披露 exact split 与聚合，不能直接横比其他物理 QA。[Max VL 卡][max-vl-card] |
| SLAKE | 医学影像视觉问答；Qwen 本次 N 未披露 | 图中未写具体 VQA 聚合 | 根据标注过的医学影像回答器官相对位置 | **90.8** | 开放/封闭式题、语言切片和答案归一化会移动分数。[Max VL 卡][max-vl-card] |
| MedXpertQA-MM | 多模态医学专家问答；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 结合病理图与化验单选择下一步检查 | **80.4** | 不是临床结局；病例切片、judge 与安全性未由单一分数覆盖。[Max VL 卡][max-vl-card] |
| PMC-VQA | 生物医学论文图表问答；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 阅读论文中的生存曲线并判断实验组差异方向 | **66.2** | 论文图泄漏、OCR、答案归一化与 split 未在图中说明。[Max VL 卡][max-vl-card] |

### 6.2 Visual Agent & Coding（12/12）

| Benchmark | 能力与体量 | Metric | 课程合成题型 | Max 分数与协议 | 局限与详细卡 |
|---|---|---|---|---|---|
| [OSWorld-Verified](https://github.com/xlang-ai/OSWorld) | 桌面 GUI agent；Qwen snapshot N 未披露 | task success | 打开邮件附件，修改表格公式并导出 PDF | **86.1** | 这是 Max+scaffold+VM 的系统分；app 版本、动作空间和 retry 都会改分。[Max VL 卡][max-vl-card] |
| [OSWorld 2.0](https://github.com/xlang-ai/OSWorld-V2) | 108 个跨桌面、网页和应用的长程 workflow | **Binary / Partial** | 跨两个应用修正文件，并满足所有终态条件 | **19.4 / 46.7**；binary=满奖励任务率，partial=部分奖励聚合 | 两个数不能择高；2.0 manifest、action cap 与 VM 必须绑定。[Max VL 卡][max-vl-card] |
| ScreenSpot Pro | 专业 GUI 截图定位；本次 N 未披露 | grounding/点击定位 score；图未写具体聚合 | 在密集 IDE 截图中定位“运行当前测试”按钮 | **84.5**；官方脚注说明 Max 为发布方内部评估 | 坐标容差、分辨率和输入协议未完整披露。[Max VL 卡][max-vl-card] |
| [WebArena-Verified](https://github.com/web-arena-x/webarena) | 自托管网站行动；原始 WebArena 812 题，Verified snapshot N 未披露 | functional task success | 在项目站点找到指定 issue、加标签并设 milestone | **66.8**；官方脚注指向 OSWorld scaffold 与 WebArena-Verified grader | 网站快照、登录态、DOM/截图观察和 grader revision 都是合同。[Max VL 卡][max-vl-card] |
| [AndroidWorld](https://github.com/google-research/android_world) | Android 应用行动；经典套件 116 题，Qwen 是否完整运行未披露 | task success | 从短信提取地址，在地图收藏并建立日历事件 | **85.3** | 不能与 AndroidBench 95-task subset 合并；模拟器与 app 版本影响结果。[Max VL 卡][max-vl-card] |
| MobileWorld | 移动端跨应用 agent；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 在购物与日历应用间核对订单并建立提醒 | **77.8** | 设备、app 池、动作接口和失败重试未披露；不可当模型裸能力。[Max VL 卡][max-vl-card] |
| ClawEval-MM | 多模态工具调用；N 未披露 | **Pass@3 / Average**：三次至少一次通过 / 三次平均分 | 看懂截图后选工具修复配置，并独立尝试三次 | **77.2 / 74.8** | Pass@3 受 k 加成，不是单次可靠率；average 也不是 binary pass。[Max VL 卡][max-vl-card] |
| Vision2Web | 前端、单页与整站视觉开发；N 未披露 | 三类任务平均 score | 根据参考图实现响应式网页并通过交互检查 | **69.0**；Claude Code；三类平均；`gpt-5.4-2026-03-05` judge | 模型、coding harness、renderer 与 judge 联合决定分数。[Max VL 卡][max-vl-card] |
| QwenBlenderBench | Blender 视觉建模 agent；**内部，N 未披露** | aggregate score；精确 metric 未披露 | 依据参考图在 Blender 中搭建带材质的简单场景 | **69.9** | C-internal；资产、动作接口、渲染和 judge 不可独立外审。[Max VL 卡][max-vl-card] |
| Parametric CAD Bench | 参数化 CAD 建模；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 根据尺寸图创建带约束的可编辑机械零件 | **91.5** | CAD 内核、单位、约束检查和可编辑性 rubric 未在图中披露。[Max VL 卡][max-vl-card] |
| RecreationBench | 跨 Ubuntu/macOS/Windows/Android/Web 的应用重建；**内部，N 未披露** | aggregate recreation score；精确聚合未披露 | 根据截图与交互说明重建一个轻量应用 | **51.7** | C-internal；平台池、参考行为与 judge 不能独立审计。[Max VL 卡][max-vl-card] |
| PresentBench | 演示文稿理解/生成；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 根据品牌规范和数据制作三页可编辑演示稿 | **79.6** | 模板、字体、渲染、可编辑性与 judge 合同未披露。[Max VL 卡][max-vl-card] |

### 6.3 Document & Office Intelligence（7/7）

| Benchmark | 能力与体量 | Metric | 课程合成题型 | Max 分数与协议 | 局限与详细卡 |
|---|---|---|---|---|---|
| [CharXiv (RQ)](https://github.com/princeton-nlp/CharXiv) | 2,323 张论文图；常用 validation 1,000 图、RQ 1,000 题，Qwen split 未披露 | **Without CI / With CI** 的 reasoning-question judge score | 比较四个 subplot，判断方法在哪个数据区才领先 | **88.4 / 93.5**；`CI`=code interpreter；官方称人工核验后修正少量错误标注 | RQ、crop、judge 与 corrected revision 必须锁定；工具增益不是视觉编码器增益。[Max VL 卡][max-vl-card] |
| [OmniDocBench 1.5](https://github.com/opendatalab/OmniDocBench) | 文档解析；1,355 页、9 类文档、4 类布局、3 种语言 | overall；底层含 OCR、公式、表格、layout/reading-order | 把双栏论文页还原成保留公式和表格结构的 Markdown | **92.1** | overall 会隐藏表格或阅读顺序崩溃；v1.5 不能与其他 revision 混排。[Max VL 卡][max-vl-card] |
| OCR-Bench-V2 (EN/ZH) | 英文/中文 OCR；本次 N 未披露 | **EN / ZH** 两个语言切片，不是两种 metric | 分别读取英文发票和中文表单中的指定字段 | **74.2 / 68.3** | 必须保留语言身份；图像压缩、答案规范与脚本版本会移动分数。[Max VL 卡][max-vl-card] |
| CC-OCR-Bench-V2 | 复杂场景 OCR；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 读取弯曲招牌和低照度票据中的关键文本 | **79.6** | 语言、场景切片和匹配规则未在图中展开，不能反推字符错误率。[Max VL 卡][max-vl-card] |
| MTVQA-Test | 多语种文字视觉问答；test N 未披露 | 官方图仅给 scalar score；metric 未披露 | 根据多语种街景招牌回答店铺营业信息 | **56.6** | 语言分布、OCR 与 QA 聚合被压成一数，低资源语言表现不可见。[Max VL 卡][max-vl-card] |
| MADQA | 文档/多模态问答；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 从带表格和批注的合同页抽取冲突条款 | **91.8** | 题集版本、答案归一化和页面输入方式未披露。[Max VL 卡][max-vl-card] |
| QwenVisualOffice | 办公文件视觉操作；**内部，N 未披露** | aggregate score；精确 metric 未披露 | 找出表格截图中的公式错误并生成修订说明 | **44.6** | C-internal；文件集、office runtime、工具与 grader 不可外审。[Max VL 卡][max-vl-card] |

### 6.4 Real-World & Spatial Understanding（4/4）

| Benchmark | 能力与体量 | Metric | 课程合成题型 | Max 分数与协议 | 局限与详细卡 |
|---|---|---|---|---|---|
| [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA) | 真实场景空间关系；test 765 条 | exact accuracy | 判断街景中红车与卡车的相对距离 | **88.0** | 样本小且有场景偏置；缩放、抽取和语言先验影响分数。[Max VL 卡][max-vl-card] |
| [ERQA](https://github.com/embodiedreasoning/ERQA) | 静态具身空间推理；400 道四选一 | A/B/C/D exact accuracy | 从桌面双视角判断机械臂抓取路径是否受阻 | **77.8** | QA 不是机器人闭环成功率；选项可猜，图像顺序需固定。[Max VL 卡][max-vl-card] |
| LingoQA | 驾驶场景视觉问答；Qwen 本次 N 未披露 | 官方图仅给 scalar score；metric 未披露 | 根据行车视频帧解释为何此刻不能变道 | **84.8** | 视频/帧输入、驾驶场景分布与 judge 未在图中完整披露。[Max VL 卡][max-vl-card] |
| SURDS | 空间/真实世界理解；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 从室内图判断绕开障碍后到目标物的方向 | **77.8** | 任务定义、split 和评分合同在图中未展开；只保留厂商值。[Max VL 卡][max-vl-card] |

### 6.5 Visual Perception & Grounding（10/10）

| Benchmark | 能力与体量 | Metric | 课程合成题型 | Max 分数与协议 | 局限与详细卡 |
|---|---|---|---|---|---|
| SimpleVQA | 基础视觉问答；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 识别图片中对象的颜色与数量 | **75.0** | “简单”题也受答案归一化和图片质量影响；不能外推复杂推理。[Max VL 卡][max-vl-card] |
| WorldVQA | 世界知识与视觉问答；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 识别地标后回答其所在国家 | **53.2** | 视觉识别与记忆知识混合，单分数不能分解两类错误。[Max VL 卡][max-vl-card] |
| MMStar | 多模态综合能力；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 根据图表、对象关系和文字选择唯一答案 | **85.9** | 数据污染控制、能力切片和抽取细节未在图中披露。[Max VL 卡][max-vl-card] |
| [PerceptionBench](https://github.com/MoonshotAI/PerceptionBench) | 细粒度视觉感知；公开集 3,000 题 | benchmark score；官方图未写具体聚合 | 比较近似对象的局部形状并判断差异 | **63.5**；官方脚注说明 Max 为发布方内部评估 | 输入分辨率、reasoning 配置和聚合需随 snapshot 保存。[Max VL 卡][max-vl-card] |
| CountQA | 视觉计数；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 统计遮挡场景中满足指定属性的物体数量 | **82.4** | 重叠、尺度、答案抽取和计数范围会显著影响结果。[Max VL 卡][max-vl-card] |
| RefAdv-S | 对抗式指代表达定位；N 未披露 | grounding score；精确 metric 未披露 | 在多个同类物体中定位带否定条件的目标 | **80.2** | 容差、框/点协议和对抗模板未在图中披露。[Max VL 卡][max-vl-card] |
| Dense200 | 密集目标感知/定位；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 在拥挤货架图中定位指定的小包装商品 | **87.0** | 名称不等于 200 道题；输入分辨率与匹配阈值未披露。[Max VL 卡][max-vl-card] |
| COCO | 通用对象感知/定位；Qwen 使用的 split 与 N 未披露 | 官方图仅给 scalar score；不能猜成 COCO AP | 根据自然图像返回指定对象的位置 | **78.7** | COCO 有多种任务与指标；没有 split/metric 就不能与检测 AP 或 caption 分数比较。[Max VL 卡][max-vl-card] |
| VisFactor | 视觉因素分解与感知；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 分离材质、光照和视角变化，判断真正改变的属性 | **60.8** | 任务版本、因素切片和聚合未披露。[Max VL 卡][max-vl-card] |
| VLMsAreBiased | 视觉语言模型偏置/稳健性；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 在交换性别或背景后检查对象判断是否保持一致 | **88.3** | 高分不等于整体公平；偏置维度、方向与聚合必须看原任务卡。[Max VL 卡][max-vl-card] |

### 6.6 Video Intelligence & Agents（10/10）

| Benchmark | 能力与体量 | Metric | 课程合成题型 | Max 分数与协议 | 局限与详细卡 |
|---|---|---|---|---|---|
| [VideoMME (w/ Sub.)](https://github.com/MME-Benchmarks/Video-MME) | 长短视频理解；v1 约 900 视频、2,700 QA，Qwen 本次 manifest 未披露 | 官方图仅给带字幕条件的 scalar score；精确 metric 未披露 | 结合字幕与镜头判断人物行动的先后顺序 | **90.4**；`w/ Sub.`=带字幕 | 字幕是额外输入；帧采样、视频可得性与 token budget 都影响分数。[Max VL 卡][max-vl-card] |
| [VideoMME v2 (w/ Sub.)](https://github.com/MME-Benchmarks/Video-MME-v2) | 更新版视频理解；约 800 视频、3,200 QA，Qwen 本次 manifest 未披露 | 官方图仅给带字幕条件的 scalar score；精确 metric 未披露 | 结合跨镜头信息和字幕回答事件因果 | **68.3**；`w/ Sub.`=带字幕 | v2 与 v1 任务难度/组成不同，不能把 90.4→68.3 当模型退化。[Max VL 卡][max-vl-card] |
| VideoMMMU | 多学科视频理解；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 看实验演示视频后回答学科推理题 | **88.7** | 学科切片、帧/音频/字幕输入与抽取规则未披露。[Max VL 卡][max-vl-card] |
| MMVU | 多模态视频理解；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 追踪多镜头中的对象状态并判断最终结果 | **82.4** | 帧采样和输入模态决定可见证据；不能只按模型名比较。[Max VL 卡][max-vl-card] |
| MLVU (M-Avg) | 多任务长视频理解；N 未披露 | **M-Avg**；图未进一步定义聚合 | 在长视频中定位事件并总结跨段关系 | **90.8** | 保留 M-Avg 原标签；未披露任务权重时不要改写成简单准确率。[Max VL 卡][max-vl-card] |
| TVBench | 时间理解与视频推理；N 未披露 | 官方图仅给 scalar score；metric 未披露 | 判断两个动作发生的先后与持续时间 | **81.9** | 时间采样、clip 长度与答案抽取未披露。[Max VL 卡][max-vl-card] |
| LVBench | 最长约两小时视频的检索与全局理解；本次 N 未披露 | QA score；精确聚合未披露 | 在完整比赛录像中定位战术首次和末次出现 | **81.8** | 帧采样、字幕、可访问视频与 token budget 都会移动分数。[Max VL 卡][max-vl-card] |
| LVBench (w/ Mem.) | LVBench 加外部记忆系统；N 同上但 manifest 未披露 | 带 memory plugin 的 QA score | 将长视频分段记忆后跨段检索同一事件 | **85.6**；官方脚注称用基于 Qwen 插件构建的记忆系统 | 这是模型+记忆插件，不是裸模型分；不可把 +3.8 全归因于权重。[Max VL 卡][max-vl-card] |
| EgoLife (w/ Mem.) | 第一视角长视频/生活记忆；N 未披露 | 带 memory plugin 的 score；精确 metric 未披露 | 从一天的第一视角视频回忆物品最后出现的位置 | **80.3**；官方脚注称用基于 Qwen 插件构建的记忆系统 | 记忆切分、检索、视频隐私与 scorer 均是系统合同。[Max VL 卡][max-vl-card] |
| VideoDR (w/ Search) | 视频深度检索/研究 agent；N 未披露 | 带 search 的 aggregate score；精确 metric 未披露 | 搜索长视频片段与辅助资料，回答带时间证据的问题 | **73.2**；官方脚注说明在浏览器搜索工具条件下评估 | 搜索索引、工具、来源与召回上限会改变结果；不能当闭卷视频分。[Max VL 卡][max-vl-card] |

## 7. 多模态 Agent：看见之后还要行动

| Benchmark | 能力与体量 | Metric | 课程合成例题 | Qwen 协议与分数 | 局限与详细卡 |
|---|---|---|---|---|---|
| [OSWorld-Verified](https://github.com/xlang-ai/OSWorld) | 1.x 桌面环境任务；Qwen snapshot 的 N 未披露 | task success | 从邮件附件读取数据，修改 spreadsheet 后导出 PDF | **Max 86.1；27B 84.3**；Flash-Next `—` | 属 1.x/Verified，不是 OSWorld 2.0；VM、app、动作接口与 scaffold 会移动分数。Max 值来自独立 VL chart，27B 值来自 27B 卡。[Max VL 卡][max-vl-card]、[27B 卡][vl-card] |
| [OSWorld 2.0](https://github.com/xlang-ai/OSWorld-V2) | **108 个**跨桌面、网页和应用的长程 workflow | `Binary / Partial` | 在两个应用间核对状态，修正文件并完成最终提交 | **Flash-Next 19.4 / 52.3**；Flash 卡同表列 27B `19.4 / 48.0`，但这不是 27B 的 Verified 84.3 | binary 与 partial 相差大；2.0/2.1 manifest、500-action cap 与 VM 必须绑定。[Flash 卡][flash-card] |
| [WebArena-Verified](https://github.com/web-arena-x/webarena) | 自托管网站行动；原始 WebArena **812 题**，Verified 的精确 snapshot N 未披露 | 终态 functional success | 找到指定 issue、加标签并安排 milestone，不动同名诱饵 | **27B 64.8**：OSWorld scaffold + official WebArena-Verified grader | DOM/截图、网站快照、登录态、step cap 和 grader revision 都是被测系统的一部分。[27B 卡][vl-card] |
| [AndroidWorld](https://github.com/google-research/android_world) | 经典套件 **116 题、20 apps**；Qwen卡未说明是否完整集 | task success | 从短信取地址，在地图收藏并建立带时区的日历事件 | **27B 81.9；Flash-Next 84.5** | 不要与 Max 的 `AndroidBench 95-task subset` 合并；模拟器、app version 与观察接口需一致。[27B 卡][vl-card]、[Flash 卡][flash-card] |
| ClawEval-MM | 多模态工具调用；N 未披露 | `Pass@3 / Average`：三次至少一次通过 / 三次平均 score | 看懂截图后选择工具修复配置，并在三次独立尝试中验收 | **27B 57.4 / 56.9；Flash-Next 64.4 / 60.4** | Pass@3 会随 k 增大，不能当单次可靠率；average 与 binary pass 也不同。[27B 卡][vl-card]、[Flash 卡][flash-card] |
| RecreationBench | 跨 Ubuntu/macOS/Windows/Android/Web 的应用重建；**内部，N 未披露** | aggregate recreation score | 根据产品截图与行为说明重建一个带交互的轻量应用 | **27B 47.1；Flash-Next 49.9** | C-internal；平台池、参考行为、judge 和任务选择不可独立审计。[27B 卡][vl-card]、[Flash 卡][flash-card] |
| Vision2Web | 前端、单页和整站视觉开发；N 未披露 | 三类平均 score | 根据页面参考图实现响应式网页并通过视觉与交互检查 | **27B 62.9；Flash-Next 64.0**：Claude Code，`gpt-5.4-2026-03-05` judge | 这是模型+Claude Code+judge 的系统分；不同 viewport、renderer 或 judge 不可横比。[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [SWE-bench Multimodal / SWE-MM](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal) | issue 含 screenshot/mockup；Qwen跑 public dev split，精确 N 见数据 revision | resolved / task score | 根据 UI 截图修复布局和交互，并通过代码与视觉 verifier | **27B 38.6**：Claude Code，采用 Claude Opus 4.7 system card Appendix 8.3 的修改；Flash-Next `—` | image-dependent slice、渲染、修改版任务和 harness 必须锁定；不能当纯视觉分。[27B 卡][vl-card] |

## 8. 多模态理解：视觉、文档、空间与长视频

| Benchmark | 能力与体量 | Metric | 课程合成例题 | Qwen 协议与分数 | 局限与详细卡 |
|---|---|---|---|---|---|
| [MathVision](https://github.com/YerongLi/MathVision) | 真实视觉数学；full **3,040 题**，testmini 304；Qwen split 未披露 | exact/数学等价；`Without CI / With CI` | 从带辅助线的几何图求角度，并输出规范数值 | **27B 90.0 / 94.6；Flash-Next 90.6 / 95.7**。`CI`=code interpreter；Qwen固定 step-by-step + boxed prompt；卡称修正少量错误标注 | 工具版是系统能力；split、prompt 和 corrected annotations 不同就不能比较。[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [BabyVision](https://github.com/UniPat-AI/BabyVision) | 低层视觉、追踪和空间模式；Qwen本次理解集 N 未披露 | boxed answer + judge；`Without CI / With CI` | 沿迷宫路径判断最终出口，不依赖百科知识 | **27B 65.7 / 85.6**；Flash-Next `—` | code interpreter 增益不能归到 vision encoder；judge、压缩和 fine-grained split 会改分。[27B 卡][vl-card] |
| [CharXiv (RQ)](https://github.com/princeton-nlp/CharXiv) | **2,323 张**真实论文图；常用 validation 为 1,000 图，含 1,000 道 RQ | reasoning-question judge score；`Without CI / With CI` | 比较四个 subplot，判断哪种方法只在低数据区领先 | **27B 83.7 / 90.2；Flash-Next 84.6 / 90.6**。卡称人工修正少量错误标注 | `CI` 是 code interpreter；RQ、descriptive、crop、judge 与 corrected revision 必须分栏。[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [OmniDocBench 1.5](https://github.com/opendatalab/OmniDocBench) | 文档解析；v1.5 **1,355 页、9 类文档、4 类布局、3 种语言** | 卡报 overall；底层含 OCR、公式、表格、layout/reading-order 指标 | 把双栏论文页还原成保留公式和表格结构的 Markdown | **27B 91.1**；Flash-Next `—` | overall 可隐藏 table TEDS 或 reading-order 崩溃；v1.5 不能与 v1.7 榜混用。[27B 卡][vl-card] |
| [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA) | 真实场景空间关系；test **765 条** | exact accuracy | 判断行车照片里最近红车与卡车的相对距离 | **27B 85.9；Flash-Next 88.5** | 小样本且场景偏置明显；图片缩放、答案抽取和语言先验影响结果。[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [ERQA](https://github.com/embodiedreasoning/ERQA) | 静态具身空间推理；**400 道四选一** | A/B/C/D exact accuracy | 从桌面两个视角判断机械臂抓取路径会被什么阻挡 | **27B 65.5；Flash-Next 72.3** | 是 QA，不是机器人闭环成功率；多选可猜，image ordering 必须固定。[27B 卡][vl-card]、[Flash 卡][flash-card] |
| [LVBench](https://github.com/zai-org/LVBench) | 最长约两小时视频的检索与全局理解；本次 N 未披露 | QA accuracy | 在完整比赛录像中定位战术首次与末次出现并解释变化 | **Flash-Next 76.6**；Flash 卡同表列 **27B 72.4** | 帧采样、字幕、视频链接可得性与 token budget 会改分；27B 值来自 Flash 卡配对表。[Flash 卡][flash-card] |

## 9. 从这套发版表能推出什么，不能推出什么

可以推出：

- Qwen3.8 的发版重点已从短答案转向 repo/terminal、长程办公、多工具、视觉 agent 与长上下文。
- 27B 与 Flash-Next 都是原生 VLM；Flash-Next 还用较低 activated parameters 探索新的效率架构。
- 同一 benchmark 在不同 checkpoint 上有一组厂商内配对读数，可用于形成复现实验假设。

不能推出：

- Max 的 `86.6` Terminal 分数等于本地 2.4T checkpoint 的裸能力。
- Flash-Next 比 27B 高出的每一点都来自 QSA、n-gram embedding 或参数效率；训练数据、后训练、harness 与采样也同时变化。
- `With CI` 比 `Without CI` 高出的差值等于视觉 encoder 提升；它测的是加 code interpreter 后的系统。
- QwenReact/QwenSVG/CoWork/Recreation 的内部数字证明公共 SOTA。
- 原生/可扩 1M context 加上 MRCR 高分，足以证明任意百万 token 工作流可靠。

## 10. 费曼自检：问题与答案

> **参考答案**：为方便逐项核对，每道问题后直接给出答案，而不是把答案集中折叠到文末。

1. **为什么不能把 Max 的全部表格写成 2.4T-A95B 开放权重成绩？**
   因为官方明确把 Max 定义为基于该 checkpoint、但另加 vision、non-thinking、默认 1M、官方工具和托管运行时的系统；卡中 benchmark 列也叫 Qwen3.8-Max，并没有开放 checkpoint 的独立列。

2. **Max、27B、Flash-Next 的 Terminal/SWE 数字能否直接比较参数规模？**
   不能。它们常换 Claude Code、Terminus 或 mini-SWE-agent，还会换 context、timeout、corrected task set 与“两个 harness 取最大值”规则。

3. **为什么 Max 的 FrontierSWE 73.5 不是 v2 领先 73.5%？**
   它是 v1 的 dominance，依赖当时对手池；v2 是 34 题、每题 partial reward 后做 `mean@5`，两个量没有共同尺度。

4. **为什么 Max 的 AndroidBench 75.1 不能和 27B 的 AndroidWorld 81.9 做减法？**
   Max 卡写的是 AndroidBench 95-task public subset、`avg@3`；27B 卡写 AndroidWorld，任务池与合同名称都不同。

5. **ClawEval 的 64.4 与 60.4 分别回答什么？**
   64.4 是三次中至少一次成功的 Pass@3；60.4 是三次 benchmark score 的平均。前者受搜索次数加成，后者描述平均表现。

6. **MathVision/CharXiv 的 `CI` 是置信区间吗？**
   不是，是 code interpreter。`90.6/95.7` 是无/有代码解释器两套系统合同，不是不确定性上下界。

7. **ALE 的 `27.0/52.4` 为什么要保留两个数？**
   `27.0` 是满分任务比例，`52.4` 是 partial score。只报后者会把“做了一部分”误写成“完整交付”。

8. **Qwen 内部 benchmark 有什么价值？**
   它们显示团队在优化 repo engineering、Qoder 体验、React/SVG 渲染和 cowork；但题集、污染、judge 与对手池不可完整外审，所以不能支撑公共跨厂排名。

9. **27B 与 Flash-Next 的配对分数能证明 Flash 架构因果优越吗？**
   不能。配对表减少了发布方差异，却没有冻结训练数据、后训练、参数规模、推理实现和每项 harness；它只提出值得做受控消融的假设。

10. **看到“原生 262K、可扩 1M”后，还要看哪些证据？**
    至少分四层：接口容量、MRCR 类定位、LongBench 类跨证据推理、repo/agent 的长轨行动；前一层是后一层的必要但不充分条件。

## 11. 覆盖核对

- Max public coding：8/8；internal coding：4/4。
- Max General Agent：9/9，其中 HLE-tools 为避免合同混淆，在 General/Long Context 首行单列。
- Max General/Long Context：官方十个非 agent 表项 + HLE-tools，共 11 行合同。
- Max VL Performance Chart：55/55；Multimodal Reasoning 12、Visual Agent & Coding 12、Document & Office Intelligence 7、Real-World & Spatial Understanding 4、Visual Perception & Grounding 10、Video Intelligence & Agents 10。
- 27B Text Performance 指定项：Terminal 2.1、SWE-Pro、NL2Repo、DeepSWE、QwenSWE、CoWork、Job、ALE、IFBench、GPQA、HLE、LiveCodeBench v6，12/12。
- Flash-Next Language 指定项：DeepSWE、SWE-Pro、SWE multilingual、NL2Repo、CoWork、Job、ALE、Toolathlon、IFBench、GPQA、HLE、LiveCodeBench v6，12/12。
- VLM 指定族：OSWorld 两条 lineage、WebArena、AndroidWorld、ClawEval、Recreation、Vision2Web、SWE-MM、MathVision、BabyVision、CharXiv、OmniDoc、RealWorldQA、ERQA、LVBench，全部覆盖；未报告的 checkpoint 明写 `—`。

## 12. 官方一手来源

- [Qwen3.8-2.4T-A95B / Max benchmark card][max-card]
- [Qwen3.8-Max 官方卡与 VL Performance Chart][max-vl-card]
- [Qwen3.8-27B model card][vl-card]
- [Qwen3.8-Flash-Next model card][flash-card]
- [Qwen3.8-Max 发布博客](https://qwen.ai/blog?id=qwen3.8)
- [Qwen3.8-Flash-Next 技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

[max-card]: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B#benchmark-results
[max-vl-card]: https://github.com/AlibabaCloud-Official/Qwen3.8-max/blob/main/README.md
[vl-card]: https://huggingface.co/Qwen/Qwen3.8-27B#benchmark-results
[flash-card]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next#benchmark-results
