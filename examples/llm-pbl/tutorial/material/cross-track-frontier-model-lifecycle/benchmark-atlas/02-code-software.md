# 分册 02：代码与软件工程 benchmark

> **核心问题**：从“补一个函数”到“二十小时完成可合并工程”，分数的被测对象怎样从 code model 变成完整 agent system？
> **范围**：算法题、issue repair、终端、从零建库、程序重建与超长程工程；动态厂商水位见 [发布证据账](RELEASE_LEDGER.md)。
> **快照**：2026-09-22；“当前/截至快照”的水位只对该日期成立。

## 1. 能力阶梯先于榜单

```text
短函数 / 算法题
  → 已知 repo 的局部 issue patch
  → terminal 中配置、调试、构建与实验
  → 空目录按需求生成完整 repo
  → 给行为 oracle 重建完整程序
  → 多小时开放式工程 / 研究优化
```

越往后，模型权重只占结果的一部分；harness、shell、搜索、context compaction、重试、时间/成本、容器和 verifier
共同决定分数。

## 2. 短代码与持续更新

### 2.1 LiveCodeBench：按时间更新的竞赛代码探针

- **基本信息**：[官方仓库](https://github.com/LiveCodeBench/LiveCodeBench) 持续收集 LeetCode、AtCoder、Codeforces 新题，并版本化时间窗口；`release_v6` full code-generation 集为 1,055 题，另有 lite/日期子集与 self-repair、execution、test-output prediction。
- **能力项 / 阶梯**：L0 算法设计、实现与局部调试。
- **课程合成例题**：实现一个在 $2\times10^5$ 节点上求动态图连通性的函数，通过隐藏时间/内存测试。
- **打分**：执行测试的 `pass@1`/`pass@5`，按 easy/medium/hard 与时间窗口分桶。
- **厂商采用**：Qwen/DeepSeek/Kimi 等 base/instruct 模型卡常用；Qwen3.8 base report 使用 v6。
- **局限**：竞赛题不要求读 repo、理解用户意图或做可维护 patch；timeout 可造成小幅波动，`pass@5` 不是单次可靠率。
- **最高水位口径**：只在相同 release/time window、语言、采样与 timeout 内比较；它是 code reasoning 下界，不是 SWE agent 排名。

## 3. Issue repair：测试通过仍不一定可合并

### 3.1 SWE-bench Verified：500 个经人工筛选的 Python issue

- **基本信息**：[SWE-bench](https://www.swebench.com/) 从真实 GitHub issue/PR 构造环境；Verified 是 500 个经人工确认较可解、test 合同较可靠的实例。
- **能力项 / 阶梯**：L2 repo 导航、复现 bug、跨文件修改、运行测试。
- **课程合成例题**：某 Python 库在 timezone-aware datetime 下序列化错误，需找实现、补 patch 并通过 fail-to-pass 与 pass-to-pass tests。
- **打分**：resolved rate / pass@1；通常要求新增失败测试通过且既有测试不回归。
- **厂商采用**：历代 GPT、Claude、Qwen、DeepSeek 等；最新发版逐渐迁到 Pro、DeepSWE、FrontierCode 等更新集。
- **局限**：500 题高度公开并趋于饱和；agent scaffold 与 repo-specific setup 影响大；测试通过不检查全部 code quality。
- **最高水位口径**：只保留为历史可复现基线，并用 fresh/live family 做晋升证据。

### 3.2 SWE-bench Pro：更真实，也暴露 benchmark 本身会坏

- **基本信息**：[论文](https://arxiv.org/abs/2509.16941) 描述 1,865 个问题、41 个专业 repo，分 public、held-out 与 commercial；任务往往跨多文件、需小时到天。
- **能力项 / 阶梯**：L2/L3 企业级 issue resolution。
- **课程合成例题**：给 B2B 服务的权限继承 bug，需同时改 schema migration、API、缓存和回归测试。
- **打分**：通常 `pass@1`/resolved；不同发布又使用 public、corrected、verified 或私有子集。
- **厂商采用**：Qwen3.8、Claude Fable、Grok、GLM 等最新 coding 发布采用。
- **局限**：[OpenAI 2026 审计](https://openai.com/index/separating-signal-from-noise-coding-evaluations/) 估计约 30% 原始任务存在问题；早期统一 scaffold 最高约 23%，最新 corrected/厂商表可到 60–80%，这首先说明 task set/protocol 已变。
- **最高水位口径**：**当前不给单一数字**；必须写 public/held-out、原始/corrected/Pro Verified、harness 与 trials。Fable 5.1 的 81.2% 只属于其披露合同。

### 3.3 SWE-bench Multilingual / Multimodal：语言与视觉是新增变量

- **基本信息**：[Multilingual](https://www.swebench.com/multilingual.html) 使用 300 个、9 种语言的 repo issue；[Multimodal](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal) 的 issue 含 screenshot/mockup 等视觉证据。
- **能力项 / 阶梯**：L2 多语言代码生态；L2 视觉定位/UI 修复。
- **课程合成例题**：根据前端截图修复布局与交互，或在 Rust/Go repo 中修复并发边界。
- **打分**：resolved rate；Multimodal 要固定图片输入/渲染和 GUI verifier。
- **厂商采用**：Claude Fable 5.1 报 Multilingual 89.1%、Multimodal 54.7%；同表 Opus 5 为 89.5/59.4。
- **局限**：跨语言平均会隐藏低资源语言退化；截图题仍可能靠 issue 文本完成，需 image-drop。
- **最高水位口径**：分语言、分 image-dependent slice；同一 harness 和 task revision 内比较。

### 3.4 DeepSWE 1.1：113 个跨 91 repo 的原创长程任务

- **基本信息**：[官方站](https://deepswe.datacurve.ai/) 当前 v1.1 有 113 个原创任务，覆盖 91 个活跃 repo、Go/Python/TypeScript/Rust/JavaScript 五种语言。
- **能力项 / 阶梯**：L2 长程 repo engineering，强调比旧 SWE-bench 更大的解法与更多输出 token。
- **课程合成例题**：给异步运行库增加可取消 body read，处理 shutdown、timer 与 form-data 多条交互路径。
- **打分**：每个任务所有 fail-to-pass 与 pass-to-pass tests 全绿才 resolved；公开榜常统一 mini-SWE-agent。
- **厂商采用**：Step 5、DeepSeek V4、Kimi K3、Qwen3.8、GLM 5.3、GPT/Claude/Grok 最新发布均使用。
- **局限**：厂商有时使用 Kimi Code、SWE-agent、Codex、Claude Code 而非统一 mini agent；同名分数因此是系统比较。
- **最高水位**：GPT-6 Astra 厂商表 74.1%；Grok 统一 mini-SWE-agent 71.0%；先标 harness，再谈差值。

## 4. Terminal 与科学工作流

### 4.1 Terminal-Bench：任务版本比小数点重要

- **基本信息**：[Terminal-Bench](https://www.tbench.ai/) 在隔离终端完成构建、debug、数据处理、系统管理等工作；2.1 有 89 题并修复 2.0 中 28/89 题的问题，3.0、4.0 又更换任务与方法，4.0 当前为 66 题。
- **能力项 / 阶梯**：L2 shell/tool use、环境探索、持久调试与终态提交。
- **课程合成例题**：修复一个失败的多阶段 Docker build，离线恢复依赖并让服务健康检查通过。
- **打分**：严格 task success rate；多 trials、agent harness 与长 timeout 常见。
- **厂商采用**：几乎所有 2026 coding/agent 旗舰；Qwen3.8 使用 2.1，GLM 新表可用 3.0，GPT/Claude/Grok 最新使用 4.0。
- **局限**：**2.1 的 88 分和 4.0 的 58 分没有高低关系**；容器、网络、provider failure 与 timeout 造成大噪声。
- **最高水位**：4.0 厂商同表中 Mythos 5.1 60.9%、Astra 57.9%、Fable 55.8%；Mythos 是 trusted-access，且 trials 不同。

### 4.2 Terminal-Bench-Science 0.1：从写代码走向做科研工作流

- **基本信息**：[官方仓库](https://github.com/harbor-framework/terminal-bench-science) 当前有 70 个专家任务，覆盖五类科学领域。
- **能力项 / 阶梯**：L2/L3 分析数据、运行模拟、拟合模型、解释结果与产生可验证 artifact。
- **课程合成例题**：读取实验 CSV，发现仪器 drift，拟合校正模型并生成满足统计检验的报告与图。
- **打分**：terminal task success；当前标准误可达约 3.5–4.5pp/model。
- **厂商采用**：GPT-6 Astra、Claude Fable/Mythos 5.1 最新发布。
- **局限**：70 题且执行昂贵；依赖 scientific stack 与 wall time；它测的是 agent+environment，不是纯科学知识。
- **最高水位**：Astra 64.6%、Fable 52.6%，但必须附 trials、harness、成本和不确定性。

## 5. 从空目录到完整系统

### 5.1 NL2Repo：自然语言需求 → 可安装 Python repo

- **基本信息**：[论文](https://arxiv.org/abs/2512.12730) 含 104 个 Python repo 重建任务、9 类领域，平均规格约 18.8K tokens；给 agent 一份需求文档和空 workspace，要求设计架构、依赖和多模块实现。
- **能力项 / 阶梯**：L3 repository generation、全局一致性与长程计划。
- **课程合成例题**：从规格实现一个带持久化、CLI、重试与插件接口的 Python 包，没有预给 skeleton。
- **打分**：主要是所有 hidden tests 的**平均通过率**，另看整个 repo fully correct。
- **厂商采用**：DeepSeek V4、Qwen3.8、GLM 等最新 coding 发布采用。
- **局限**：55% average test pass 不等于 55% repo 完成；局部测试可通过但架构、安装或跨模块失败。
- **最高水位**：Qwen3.8-Max 厂商表报 55.9 average test pass；只能按该 harness 解释。

### 5.2 ProgramBench：只有 binary 与文档，重建程序

- **基本信息**：[官方站](https://programbench.com/) 有 200 个任务、248K+ behavioral tests；从小 CLI 到 FFmpeg/SQLite/PHP，agent 不能上网、读/反编译 reference binary。
- **能力项 / 阶梯**：L3 黑盒实验、接口发现、架构设计与全 repo 实现。
- **课程合成例题**：只给一个 CLI 二进制和 `--help`，通过反复输入观察行为，重建兼容实现和 build script。
- **打分**：primary `Resolved` 要全部 hidden behavior tests 通过；另报 `Almost (≥95%)` 和 average hidden-test pass。
- **厂商采用**：Kimi K3、Claude Fable 等最新发布开始采用；Anthropic 的 filtered/no-time-limit 166-task 口径不同于公共 200-task 榜。
- **局限**：平均 test pass 与 fully resolved 差距巨大；部分行为可能受不可观测常量/格式限制，且单任务成本可极高。
- **最高水位**：公共 200-task 榜当前 Opus 5 resolved 4.5%、almost 37.0%、average pass 74.7%；不能把 Fable 的 87.6% filtered average 写成 87.6% 完成率。

### 5.3 FrontierCode 1.1：测试通过之外，还要“可 merge”

- **基本信息**：[Cognition](https://cognition.com/frontiercode) 由开源 maintainer 编写真实任务与 1,000+ criteria；Extended 150 题，Main 为最难 100 题，1.1 修订网络与评分。
- **能力项 / 阶梯**：L3 functional correctness、code quality、scope control 与 maintainer judgment。
- **课程合成例题**：实现新缓存策略；即使测试通过，若顺手改 CI、文档或无关 API，也会因 blocker/out-of-scope 扣分。
- **打分**：held-out tests + weighted rubric + blocker 的 composite，常 `mean@5`。
- **厂商采用**：GPT-6 Astra、Claude、Grok 等最新 coding 发布。
- **局限**：不是 binary resolved rate；网络允许范围、developer prompt 与 criteria revision 会变；高 effort 可能因越界修改反而降分。
- **最高水位**：Main 当前约 53.5 级，Astra Extended 厂商表 64.5；Main/Extended 不可比较。

## 6. 超长程工程：partial progress 与完整交付分开

### 6.1 FrontierSWE lineage：v1 dominance 与 v2 partial reward

- **基本信息**：[v1 存档](https://www.frontierswe.com/v1) 有 17 题，按 per-task average rank 与“随机抽一题、随机抽对手时获胜”的 dominance 排名；[v2 官方榜](https://www.frontierswe.com/) 扩至 34 个开放式技术/研究任务，每模型每题 5 trials、每 trial 20 小时，含视觉赛车、脑信号、天气模型等。
- **能力项 / 阶梯**：L3 超长程研究工程、实验迭代、资源管理与恢复。
- **课程合成例题**：训练一个从视频预测台球轨迹的系统，在固定 compute 内优化验证指标并提交可复现实验。
- **打分**：v1 dominance 是相对于当时对手池的 win-rate；v2 每题给 $[0,1]$ partial reward，leaderboard 为 `mean@5`，不是“完整解决率”。v2 的 whisker 是 5 trials 的 worst@5–best@5，不是置信区间。
- **厂商采用**：GPT-6、Claude、GLM、Grok、Kimi、Qwen、DeepSeek 最新系统均出现。
- **局限**：v2 仅 34 题，模型/agent/时间/数百万 token 强纠缠；v1 dominance 随对手池变化，不能与 v2 reward 或新版模型的绝对完成度串线。
- **最高水位**：截至 2026-09-22，v2 公共榜为 Astra `65.5`、Fable `56.3`；Grok 4.7 的厂商卡为 `29.0`、同日公共榜为 `29.5`，两者应作为 source-snapshot 冲突保存。榜上 `±` 只表示 trial 极值范围。

### 6.2 SWE-Marathon 1.1：20 个多小时任务，必须全部 verifier 通过

- **基本信息**：[官方站](https://www.swe-marathon.org/) 有 20 个真实多小时任务，每题 8 trials；包含 full-stack clone 与长程 case study。
- **能力项 / 阶梯**：L3 完整交付、持续调试与 reward-hacking resistance。
- **课程合成例题**：克隆一个带浏览器交互的产品，unit tests 与 computer-use rubric 都必须通过。
- **打分**：binary resolution；任何 verifier fail 即 0，另有未校准 partial 仅用于诊断。
- **厂商采用**：Kimi K3、GPT、Claude、GLM、Grok 等最新 coding agent。
- **局限**：20 题导致高方差；8 trials 与巨大 token 预算不代表单次产品可靠率。
- **最高水位**：公共 v1.1 当前 Opus 5 `50.0`、Kimi K3 `48.1`、Fable 5.1 `45.6`；要同时报 agent 与 `k=8`。Kimi 模型卡的 `42.0` 来自 H20-calibrated、最终 v1.1 前分支，不能与公共 `48.1` 当作同合同复跑。

## 7. 厂商内部 coding bench 怎样记录

QwenSWE/QwenQoder/QwenReact/QwenSVGBench、Z.ai Code Bench、DeepSeek DSBench、GPT/Grok 内部 migration/design
题集能说明厂商在优化什么，但缺少公开 tasks、harness 或 judge 时只能标 **C-internal**。Elo 型 React/SVG 分数不是
正确率；“内部提升 50%”也可能是相对旧模型，而非对公共 frontier 的绝对水位。

## 费曼自检

1. SWE-bench Pro 从 23% 涨到 81%，为什么不能直接说一年内能力涨了 58pp？
2. ProgramBench average tests 87.6% 为什么可能只有很低 fully resolved？
3. FrontierSWE v1 的 dominance 与 v2 的 0–1 partial reward 为什么不能串成趋势线？
4. Terminal-Bench 4.0 分数比 2.1 低，为什么可能是好消息？

<details>
<summary>参考答案</summary>

1. task subset、broken-task 清理、harness、effort、trials 和 agent 都变了；先做同 revision、同 scaffold 的 paired replay，才能估计模型增益。
2. average 允许每个 repo 通过不同部分；resolved 是所有隐藏行为的合取，一个关键测试失败就整题未完成。前者测进展，后者测交付可靠性。
3. v1 dominance 是相对对手的排序统计，v2 是每题 verifier 的绝对 partial reward；题目、harness 与 20 小时协议也变了，estimand 不同。
4. 新版可能移除饱和/污染题、加入更难真实任务并修复 verifier，使分数重新有区分度。版本升级后下降不代表模型退化。

</details>

一句话验收：**coding 分数越接近真实工程，越属于“模型 × harness × 环境 × 预算 × verifier”；任何单因素归因都必须另做对照。**
