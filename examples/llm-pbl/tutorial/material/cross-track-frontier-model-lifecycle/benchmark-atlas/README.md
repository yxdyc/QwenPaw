# Frontier Benchmark Atlas：模型发版表到底测了什么

> **核心问题**：一张“模型 A 85.0、模型 B 82.0”的表，何时是在比较能力，何时只是在比较 harness、工具、预算或失败处理？
> **快照日期**：2026-09-22；动态型号与分数集中在 [发布证据账](RELEASE_LEDGER.md)，稳定的 benchmark 定义放在领域分册。
> **先修**：[Frontier Model Lifecycle](../README.md) 的 stage/identity 概念；建议随后进入 [Evaluation Gate](../../cross-track-evaluation-gate/README.md)。
> **证据边界**：本专题解释测量合同，不把厂商自报表当独立排行榜，也不把 benchmark 高分自动升级为生产能力。

## 先运行：同名分数为何仍不可比较

```bash
python3 -B tutorial/material/cross-track-frontier-model-lifecycle/benchmark-atlas/L0_benchmark_contract.py
```

验收看 9/9 checks：缺 `prompt_template` 的 99 分先被 fail closed；同一 `Terminal-Bench` 名称下，`pass@8`、
加搜索、换 prompt/环境/verifier、换 task manifest 或排除失败任务都不能进入基线排序。只有题集身份、执行系统、
预算与裁决合同全部一致的两行才进入同一个“最高水位”。逐段解释见 [L0 教程](tutorial_L0.md)。

## 学习阶梯与当前状态

| 级别 | 新增真实约束 | 项目与验收 | 状态 |
|---|---|---|---|
| L0 合同反例 | synthetic score rows、完整字段、缺字段高分 | 9/9；invalid 不进入排序，不同合同不做减法 | **已完成** |
| L1 公开 manifest | 固定一个公共 benchmark revision、task manifest、prompt 与环境镜像 | 生成可哈希 run manifest；缺字段/任务漂移必须拒绝 | 规划中 |
| L2 同协议 replay | 两个真实 checkpoint 在同一 harness、预算、verifier 下运行 | paired task delta、完成率、成本和区间可复算；再固定模型换 harness | 规划中 |
| L3 fresh gate | family-disjoint holdout、失败簇、canary 与 rollback lineage | 公共集只作诊断，fresh paired evidence 接入 Evaluation Gate | 规划中 |

升级原则：L1 先证明“实验能被重建”，L2 才估计 checkpoint 与 harness effect，L3 再回答“能否为目标流量发布”。

## 1. 一次分数其实是一个条件函数

把 headline 写全，应该是：

$$
\hat S = f(M,D,V,Q,P,H,T,B,R,E,J,F),
$$

其中：

- $M$：模型 checkpoint 或实际执行的 API model；
- $D,V,Q$：数据 split、benchmark revision 与精确 task manifest；
- $P,H$：prompt/template 与 agent harness；
- $T$：搜索、代码解释器、终端、浏览器等工具；
- $B$：context、输出 token、时间、成本与 tool-call 预算；
- $R$：reasoning effort、温度、采样次数、重试和 `pass@k`；
- $E$：环境镜像与 verifier revision；
- $J$：judge revision；
- $F$：timeout、拒答、fallback、safeguard 和未完成任务怎样进入分母。

只报 $(M,\hat S)$，等于把其余十一个变量藏进脚注。越接近 coding/agent/GUI 任务，隐藏变量对结论的影响通常越大。

## 2. 能力阶梯：不是“题越难级别越高”

这里按**状态空间、交互长度、验证成本与开放度**分层，而不是按当前 SOTA 分数排序。

| 层级 | 被测对象 | 代表 benchmark | 核心失败模式 |
|---|---|---|---|
| L0 受控回答 | 短上下文知识、约束遵循、数学/科学推理、短代码 | MMLU-Pro、GPQA Diamond、HLE、IFBench、AIME、LiveCodeBench | 污染、饱和、答案抽取、`pass@k` 混用 |
| L1 长输入与复杂感知 | 长文检索/综合、OCR、图表、空间、长视频 | MRCR、LongBench v2、AA-LCR、BEAM-1M、MMMU-Pro、CharXiv、OmniDocBench、Video-MME | “装得下”冒充“用得好”、模态 token/工具不等价、judge bias |
| L2 有状态工程任务 | repo patch、终端、网页、手机与桌面环境 | SWE-bench、SWE-bench Pro、Terminal-Bench、WebArena、AndroidWorld、OSWorld | harness、镜像、依赖、timeout 与基础设施噪声主导 |
| L3 长程开放工作 | 跨工具/文件/应用的职业交付和用户交互 | GAIA、BrowseComp、Toolathlon、ALE、AutomationBench、JobBench | 小样本、失败分母、fallback、LLM judge 与任务泄漏 |
| L4 开放式生成 | 图像/视频/音频的质量、遵循、一致性和安全 | GenEval、DPG-Bench、T2I-CompBench、VBench、FVD/FAD | 代理指标与人类偏好错位、reference bias、不可复现的 prompt/seed |

同一模型可能 L0 很强、L2 很弱；同一 benchmark 也可能因加工具从“模型闭卷能力”变成“系统能力”。阶梯的用途是
定位证据，不是制造一个新的总榜。

安全不是第六级，而是横切每一级的另一根轴。至少分开：危险能力是否存在、模型是否在不当请求上拒绝、是否对正常请求
过度拒绝、工具动作是否越权、prompt injection 是否成功，以及部署 router/monitor 是否阻断。`attack success rate`、
`harmful compliance`、`refusal recall`、`benign refusal` 的好坏方向并不相同。合成例题可以是“邮件正文夹带让 agent
外传联系人”的间接注入；既要检查 agent 有没有泄露，也要检查它是否还能完成正常邮件摘要。GPT-6 Astra、Claude 5.1 与
Grok 4.7 的最新 system/model card 都把 safety suite 与能力榜并列，但内部攻击集只能作为 C-internal 证据。

## 3. 指标不是百分数的不同拼法

| 指标 | 它回答的问题 | 常见误读 |
|---|---|---|
| accuracy / exact match | 一次输出能否匹配 gold 或选项 | 忽略答案抽取、部分正确与多解 |
| F1 / ANLS / edit similarity | token/span/OCR 的部分重合程度 | 把相似度当事实正确性 |
| `pass@1` | 单次代码样本通过全部测试的概率 | 和贪心 `accuracy`、多次平均混为一谈 |
| `pass@k` | $k$ 次采样中至少一次成功 | 写成“单次可靠率”；$k$ 越大越像搜索系统 |
| `pass^k` | $k$ 次都成功的稳定性 | 与 `pass@k` 正好相反却只差一个符号 |
| resolved / task success rate | agent 是否完成环境终态或测试合同 | 不记录 harness、timeout、环境失败与分母 |
| partial / rubric score | 一个复杂交付满足多少检查项 | 与“完整完成率”混在同一百分数列 |
| Elo / Bradley–Terry | 成对偏好下的相对位置 | 当作绝对正确率；忽略对手池和 judge 变化 |
| LLM-as-judge score | judge 对开放答案/视觉质量的判断 | judge 与被测模型同偏、位置/风格偏差 |
| economic outcome | 长期环境里的利润、余额或累计 reward | 把单个随机轨迹当稳定能力；忽略风险暴露 |

若每题是独立二元结果，近似标准误为

$$
SE(\hat p)\approx\sqrt{\frac{\hat p(1-\hat p)}{n}}.
$$

但 agent 任务常按 repo、网站或工作流成簇，也含环境失败；直接把每个 task 当 iid 往往会低估不确定性。

## 4. Benchmark card 的最小合同

领域分册里的每张 card 尽量回答十一件事：

1. 名称、版本与一手来源；
2. 任务量、split、语言、模态和环境；
3. 真正测量的构念；
4. 所处能力阶梯；
5. **课程合成例题**，只展示题型，不复制受版权保护的正式题；
6. metric、聚合方法与成功条件；
7. prompt、harness、环境/verifier、工具、预算、采样、timeout、judge 与失败分母；
8. 哪些最新发版采用过；
9. 截止快照日的 protocol-specific 水位；
10. contamination、饱和、小样本、judge/environment 等局限；
11. 证据等级。

证据等级统一为：

- **A-public**：题目/环境、harness、协议和结果可由公共榜或独立运行复验；
- **B-vendor**：来自厂商正式模型卡/发布页，但协议或逐题结果不完整；
- **C-internal**：厂商内部题集、内部 judge 或只给摘要；只说明厂商关注方向。

### 4.1 发版表常见别名：先正规化名称，仍不能省略 revision

| 发版表短名 | 本专题正规化身份 | 仍须保留的差异 |
|---|---|---|
| `Spreadsheet v2` | SpreadsheetBench 2 | task/data revision、Excel/LibreOffice、strict/partial |
| `WorkSpaceBench` / `Workspace` | Workspace-Bench | full/Lite、文件 manifest、renderer、judge |
| `Job` / `JobBench` | JobBench | main/Easy、task snapshot、rubric aggregate |
| `OSWorld2` | OSWorld 2.x | 2.0/2.1、binary/partial、offline/full、VM image |
| `SWE-MM` | SWE-bench Multimodal | split、图片输入、renderer、agent harness |
| Qwen 表中的 `Finance` | PRBench-Finance | PRBench split、rubric normalization、judge 与工具 |
| `Harvey Lab-AA` | Artificial Analysis 的 Harvey LAB 实现 | 120 题私有集、criterion/all-pass；不等于公共 LAB v1 |
| `GDPval-AA` / `AA-Briefcase` | Artificial Analysis 的独立实现 | 版本、Elo 对手池、Stirrup/harness 与日期 |
| `FrontierSWE` | 必须继续写 v1 或 v2 | v1 dominance 与 v2 partial `mean@5` 没有共同尺度 |

正规化只解决“它大概是哪一家族”；只要上表最后一列不同，score identity 仍不同，不能做差。

## 5. “最高水位”必须是复数

本专题不用一个粗体数字宣布 SOTA，而分四栏：

| 水位 | 可以支持什么 | 不可以支持什么 |
|---|---|---|
| public leaderboard verified | 在该公共 revision 与统一 harness 下的当前领先结果 | 换版本、换工具或厂商自定义 harness 后仍领先 |
| vendor-reported | 该厂商按披露协议得到的结果 | 独立复现、跨厂公平排名 |
| internal benchmark | 厂商发现并优化某能力切面的证据 | 外部泛化或题集无污染 |
| human/reference ceiling | 量表的解释锚点 | 把不同人群、工具或时间预算当同一上限 |

若版本/task manifest、split、prompt、环境/verifier、工具、harness、effort、`pass@k` 或失败分母不同，
表中写“**不可合并**”，而不是挑最大数字。
每个动态水位都必须带日期；旧水位仍可保留为时间序列，但不能继续叫“当前最高”。

## 6. 分册导航

- **从这次问题直接进入**：[Qwen3.8 发版 benchmark 逐项导读](QWEN38_WALKTHROUGH.md)。它先拆开 Max、2.4T-A95B、27B 与 Flash-Next，再逐项给出能力、体量、metric、课程合成题、Qwen 协议/成绩与局限。
- [文本、知识、推理与长上下文](01-text-reasoning.md)：MMLU-Pro、GPQA、HLE、IFBench、AIME、MRCR、LongBench v2、AA-LCR、BEAM-1M。
- [代码与软件工程](02-code-software.md)：LiveCodeBench、SWE-bench、SWE-bench Pro、Terminal-Bench、DeepSWE、NL2Repo、FrontierSWE 等。
- [Agent、工具与职业工作](03-agent-tool-use.md)：GAIA、BrowseComp、Toolathlon、ALE、AutomationBench、OSWorld、Job/Workspace/office 工作流。
- [多模态理解](04-multimodal-understanding.md)：MMMU-Pro、MathVision、CharXiv、OCR/文档、RealWorldQA 与长视频。
- [多模态生成](05-multimodal-generation.md)：GenEval、DPG/T2I-CompBench、VBench、FVD/FAD 以及盲评合同。
- [前沿专项与新兴协议](06-frontier-specialized.md)：科学发现、机器学习研究、网络安全、办公交付、CAD/电子、实时语音及 living benchmark 的版本陷阱。
- [2026-09-22 最新发版证据账](RELEASE_LEDGER.md)：Qwen3.8、Step 5 Preview、DeepSeek、Kimi、GLM、GPT、Gemini、Claude、Grok、Hy4 与 MiniMax。

## 7. 从发版表走到自己的 Evaluation Gate

推荐顺序不是“选一个总榜第一”，而是：

```text
产品失败簇
  → 对应能力阶梯与 benchmark card
  → 冻结 revision / task manifest / prompt / harness / environment / verifier / budget / failure denominator
  → 公共 benchmark 做诊断
  → fresh family-disjoint task 做选择
  → paired candidate-parent + cost/failure gate
  → canary / rollback / 线上 fresh evidence
```

公共 benchmark 适合给研发团队一张共享地图；真正的发布裁决仍需与目标流量匹配的 task mixture、独立 holdout、
成本/失败率以及可回滚 lineage。具体实现见 [Evaluation Gate L0–L3](../../cross-track-evaluation-gate/README.md)。

## 费曼自检

1. 为什么同一个 benchmark 名称和同一个百分数单位，仍可能完全不可比？
2. `pass@8=90%` 与 `pass@1=70%` 哪个模型更可靠？为什么题目信息不足？
3. 为什么 GPQA/HLE 的上升不能替代 SWE-bench/OSWorld 的证据？
4. 为什么 internal benchmark 仍值得记录，却不应进入公共最高水位？

<details>
<summary>参考答案</summary>

1. 分数还条件于 revision、split、prompt、harness、工具、effort、token/time budget、采样、环境、judge 和失败分母；任一项变化都可能改变被估计对象。
2. 无法直接判断。`pass@8` 是八次搜索至少一次成功，`pass@1` 是单次成功；还需同题、同采样分布、同预算，并同时报告稳定性与总成本。
3. GPQA/HLE 主要测受控的专家问答与推理；SWE-bench/OSWorld 还要求在有状态环境中观察、行动、恢复并满足终态 verifier，构念和失败面不同。
4. internal benchmark 能暴露厂商的目标任务和真实 failure clusters，常比旧学术题更贴产品；但题目、污染、judge 和协议不可审计，因此只能作为 C 级发布声明，不能支持跨厂排名。

</details>

一句话验收：**benchmark 不是一个分数，而是一份可执行测量合同；先证明两行在估计同一个量，再讨论谁更高。**
