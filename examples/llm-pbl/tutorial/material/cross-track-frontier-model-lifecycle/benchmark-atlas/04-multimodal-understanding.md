# 分册 04：多模态理解 benchmark

> **核心问题**：模型答对了，是看懂图像/视频，还是靠文字先验、OCR、工具或 benchmark 捷径猜中了？
> **范围**：静态视觉、OCR/文档、图表/数学、空间/具身与长视频；动态厂商水位见 [发布证据账](RELEASE_LEDGER.md)。
> **快照**：2026-09-22；“当前/截至快照”的水位只对该日期成立。

## 1. 多模态评测的第一性原理

一条视觉题至少有四段误差链：

```text
pixels / frames
  → perception / OCR / localization
  → binding / temporal or spatial relation
  → reasoning / world knowledge
  → answer extraction / judge
```

只看最终 accuracy，无法知道失败在哪层。高质量评测至少增加 `image-drop`、`image-swap`、crop/zoom、顺序打乱和
no-tools/with-tools 对照；否则语言先验可能在没有看图时也答对。

## 2. 通用视觉与专家推理

### 2.1 MMMU-Pro：抗文字捷径的多学科视觉题

- **基本信息**：[MMMU 官方仓库](https://github.com/MMMU-Benchmark/MMMU) 同时维护 MMMU 与 MMMU-Pro；Pro 约 1,730 道多选题，通过过滤 text-only 可答题、扩充选项和改写设置提升视觉依赖。
- **能力项 / 阶梯**：L1 专家视觉理解 + 学科推理；艺术、工程、医学、科学等多领域。
- **课程合成例题**：给一张电路图，问改变开关后哪条支路电流增大；四个选项都使用相近专业术语。
- **打分**：通常为多选 accuracy；应区分 standard、vision-only、input order 和 no-tools/with-tools。
- **厂商采用**：Qwen3.8、Kimi K3 等原生多模态模型发布/供应商一致性检查采用；Kimi Vendor Verifier 固定 `MMMU Pro Vision`。
- **局限**：最终分仍混合 perception、知识和推理；不同图片分辨率、拼接顺序和答案抽取会移动分数。
- **最高水位口径**：只在官方同 split、同 prompt/input order 和同工具合同中排；“MMMU”不能与“MMMU-Pro”合并。

### 2.2 RealWorldQA：真实场景空间关系

- **基本信息**：[xAI 官方数据集](https://huggingface.co/datasets/xai-org/RealworldQA) 当前 test 为 765 条图像问答，包含车载与日常真实场景。
- **能力项 / 阶梯**：L1 物理世界对象、距离、方向、数量和可通行性。
- **课程合成例题**：车辆第一视角照片中，问“最近的红车是否比最近的卡车更近”。
- **打分**：单词/数字或多选 exact accuracy。
- **厂商采用**：源自 xAI Grok-1.5V 发布，后被多个 VLM 模型卡沿用；当前 release 若未重跑不能沿用旧代分数。
- **局限**：765 题较小；相机视角和交通场景偏置明显；答案可由对象共现猜中时需 image-swap 对照。
- **最高水位口径**：官方 765-row revision 内比较；不要把 MME-RealWorld 等同名相近数据集混入。

### 2.3 PerceptionBench：把视觉原子能力从推理中剥离

- **基本信息**：[Moonshot 官方仓库](https://github.com/MoonshotAI/PerceptionBench) 有 3,000 道人工验证、短而唯一答案的问题，平衡十种原子感知能力。
- **能力项 / 阶梯**：L1 视觉关系、计数、属性、深度、定位、比较、细粒度识别、上下文、OCR 与感知幻觉。
- **课程合成例题**：`图中离蓝色圆最近的三角形是什么颜色？`；不再额外要求百科知识或长推理。
- **打分**：open-ended short answer accuracy；官方以 judge 对 gold 判定，并做人工一致性审计。
- **厂商采用**：由 Moonshot/Kimi 团队发布，Kimi K3 是首批统一协议评测对象之一。
- **局限**：judge 仍可能在同义词/单位上出错；原子题高分不代表多步组合或 agent 行动能力。
- **最高水位口径**：采用官方 unified prompt、最高 reasoning budget 与同 judge；记录 leaderboard 日期。

## 3. 视觉数学、图表与基础空间

### 3.1 MathVision：真实竞赛图形数学

- **基本信息**：[官方仓库](https://github.com/YerongLi/MathVision) 收录 3,040 道真实数学竞赛视觉题，覆盖 16 学科、5 个难度级；`testmini` 为 304 题。
- **能力项 / 阶梯**：L1 几何图、函数图、拓扑/组合图与数学推理。
- **课程合成例题**：给一个带辅助线但不按比例绘制的圆几何图，求阴影角度并输出数值。
- **打分**：多选 exact；开放答案需数学等价抽取/judge。必须标 full 还是 testmini、是否使用 Python 工具。
- **厂商采用**：Qwen3.8 等 VLM 发版常报告 MathVision；with-tools 的代码执行分数是系统分。
- **局限**：OCR/图形解析与数学推理纠缠；LLM judge 和 answer extractor 可带来非模型误差。
- **最高水位口径**：full/testmini、no-tools/with-Python 分栏；不能只摘最大值。

### 3.2 CharXiv：真实论文图表，不只是读刻度

- **基本信息**：[官方仓库](https://github.com/princeton-nlp/CharXiv) 使用 2,323 张真实 arXiv 图表，每图 4 道描述题和 1 道推理题；常用 validation 是 1,000 图、4,000 道描述题和 1,000 道推理题。
- **能力项 / 阶梯**：L1 图表感知、信息抽取、跨 subplot 综合与科学推理。
- **课程合成例题**：从四个子图中判断“哪种方法只在低数据区领先，并估算转折点”。
- **打分**：descriptive 和 reasoning 分开；开放回答常由 LLM judge/规则联合评分。
- **厂商采用**：Qwen3.8、Kimi K3、Claude 等发布报告 reasoning 或 no-tools/with-Python 变体。
- **局限**：crop、OCR 与代码解释器可能显著改变图表任务分数，但具体增益必须来自同一 CharXiv revision 的配对运行；`42.6 → 86.2` 是 **Chartography** 的公开案例，不属于 CharXiv。
- **最高水位口径**：明确 `RQ`/descriptive、tools、crop 和 judge；不同协议不合榜。

### 3.3 BabyVision：三岁儿童直觉仍难

- **基本信息**：[官方仓库](https://github.com/UniPat-AI/BabyVision) 同时有理解与生成 track，覆盖细粒度辨别、视觉追踪、空间知觉和视觉模式四类。
- **能力项 / 阶梯**：L1 迷宫、连线、影子、折纸、积木计数和图案补全，尽量减少语言知识作用。
- **课程合成例题**：给迷宫图，只允许视觉追踪，问从入口出发最终到哪个带编号出口。
- **打分**：理解 track 输出 boxed answer，再由 judge 对 ground truth；生成 track 另有 280 个标注任务。
- **厂商采用**：Qwen3.8 最新 VLM 发版使用该类诊断，说明学术知识高分仍不能替代低层视觉证据。
- **局限**：judge 和图像压缩仍影响结果；“儿童水平”取决于年龄组和测试程序，不能只凭 benchmark 名称作拟人化结论。
- **最高水位口径**：full/fine-grained、工具与 judge 必须一致；人类锚点单列。

## 4. OCR 与文档

### 4.1 OCRBench / OCRBench v2：看到字不等于读懂文档

- **基本信息**：[OCRBench](https://github.com/qywh2023/OCRBench) 经典版含 1,000 个人工核验 QA，覆盖文字识别、scene-text VQA、文档 VQA、关键信息抽取和手写公式；[v2 论文](https://arxiv.org/abs/2501.00321) 扩至 10,000 条双语 QA 与 31 场景。
- **能力项 / 阶梯**：L1 OCR、版面与文字推理。
- **课程合成例题**：票据图中读取总额、税率与日期，再判断两项金额是否相加一致。
- **打分**：经典版常按命中累计/accuracy；v2 有更细能力指标。供应商一致性检查有时归一到 0–1。
- **厂商采用**：Kimi K3 Vendor Verifier 用经典 OCRBench；许多 OCR/VLM 模型报告 v1/v2。
- **局限**：版本与量纲极易混淆；纯字符串命中惩罚格式差异，OCR 高分也不保证 reading order/table structure。
- **最高水位口径**：必须写 v1/v2、归一化方式、图片缩放和答案规范化。

### 4.2 OmniDocBench：把 PDF 解析拆成模块

- **基本信息**：[官方仓库](https://github.com/opendatalab/OmniDocBench) 的版本差异很大：v1.0 为 981 页，v1.5 为 1,355 页、9 类文档、4 类布局、3 种语言，后续版本再扩到 1,651 页。稳定比较必须固定如 `v1.5`，而不是只写“OmniDocBench”。
- **能力项 / 阶梯**：L1 文本 OCR、公式、表格、layout 与 reading order 的端到端结构解析。
- **课程合成例题**：把双栏论文页还原成按阅读顺序排列的 Markdown，同时保留公式 LaTeX 与表格 HTML。
- **打分**：normalized edit distance、BLEU/METEOR、TEDS、公式/布局指标等；有些模型卡报告 `(1-NED)×100` 聚合。
- **厂商采用**：Qwen/Kimi 系 VLM、专用 OCR 模型广泛使用；Qwen3.8 卡采用特定 1.5 口径时不可与当前 1.7 榜直接比较。
- **局限**：依赖 TeX、ImageMagick、Ghostscript 和 evaluator version；一个 overall 可隐藏表格或 reading-order 崩溃。
- **最高水位口径**：标 v1.5/v1.6/v1.7、pipeline 或 end-to-end、各模块分数；不同版本不合并。

## 5. 长视频与具身空间

### 5.1 Video-MME / Video-MME v2：长视频、多模态输入与采样预算

- **基本信息**：[经典 Video-MME](https://github.com/MME-Benchmarks/Video-MME) 有 900 个视频、254 小时、2,700 个 QA，时长从 11 秒到 1 小时；[v2](https://github.com/MME-Benchmarks/Video-MME-v2) 为 800 视频、3,200 QA，并提供逐词时间戳字幕。
- **能力项 / 阶梯**：L1 短/中/长视频的时序、事件、知识和跨模态理解。
- **课程合成例题**：在 45 分钟教程中，问“第二次参数回退发生在什么操作之后”，要求区分多个相似事件。
- **打分**：多选 accuracy，通常按时长/类别及 no-subtitle/with-subtitle 分栏。
- **厂商采用**：多代 GPT/Gemini/Qwen 等 VLM 发布采用；帧数、分辨率、字幕和音频是否输入是核心合同。
- **局限**：384 帧与 32 帧并非同一观察预算；均匀采样会漏掉短事件；v1/v2 不可合榜。
- **最高水位口径**：固定 v1/v2、frame selector、最大帧数、字幕/音频和 context；报告每时长桶而非只报 overall。

### 5.2 LVBench：最长两小时的检索与全局理解

- **基本信息**：[官方仓库](https://github.com/zai-org/LVBench) 专门评估最长约两小时视频的理解与信息提取。
- **能力项 / 阶梯**：L1 长时事件定位、跨片段关系与全局叙事。
- **课程合成例题**：在完整比赛录像中定位战术首次出现和最后一次出现，并解释中间策略如何改变。
- **打分**：按官方题型的 QA accuracy 聚合；需绑定视频可得性、帧采样与字幕策略。
- **厂商采用**：Qwen3.8 等长视频模型发布使用；适合与短视频 benchmark 形成长度阶梯。
- **局限**：高分可能来自字幕检索；公开视频链接失效和采样实现会造成基础设施噪声。
- **最高水位口径**：固定视频 snapshot 与采样器；subtitle-only、vision-only、combined 分栏。

### 5.3 ERQA：具身空间推理的 400 题诊断

- **基本信息**：[官方仓库](https://github.com/embodiedreasoning/ERQA) 公开 400 个由一张或多张交错图片与文本组成的四选一题，来自 Gemini Robotics 的具身推理评测。
- **能力项 / 阶梯**：L1/L2 交界的真实场景空间关系与机器人世界知识；没有实际控制环境时仍是 QA。
- **课程合成例题**：给桌面前后两个视角，问机械臂若从右侧抓杯子，哪个物体会阻挡路径。
- **打分**：A/B/C/D exact accuracy。
- **厂商采用**：Qwen3.8 等 VLM 报告；适合与 OSWorld/机器人环境区分“看图推理”和“真正行动”。
- **局限**：多选可猜；静态图片无法测试闭环感知—行动—重观察；400 题的不确定性不可忽略。
- **最高水位口径**：同 TFRecord、相同 image ordering/prompt；不可把它写成机器人任务成功率。

## 6. 必做的视觉反事实矩阵

| 对照 | 若分数几乎不降，优先怀疑什么 |
|---|---|
| image-drop | 语言先验或题干泄漏 |
| image-swap | 模型忽略具体图像，只识别题型 |
| patch/frame shuffle | 位置/时间关系没有被使用 |
| crop/zoom | 分辨率/token budget 或小目标瓶颈 |
| OCR text-only | 最终推理可能强，视觉 OCR 才是短板 |
| no-tools vs Python/search | 裸模型与系统能力被混写 |
| subtitle-only vs vision-only | 视频分数来自字幕检索而非视觉时序 |

## 费曼自检

1. MMMU-Pro 90% 为什么仍不能推出视觉 encoder 已经“解决”？
2. Video-MME 多给十倍帧后分数上涨，应该归因给模型还是系统？
3. OmniDocBench overall 不变，但 table TEDS 暴跌，生产上可能发生什么？
4. ERQA 和 OSWorld 都有图像与空间关系，为什么不在同一层？

<details>
<summary>参考答案</summary>

1. 最终 accuracy 仍混合 OCR/感知、知识、推理和答案抽取；还要看 image-drop/swap、原子 perception slice 和分辨率敏感性，才能定位视觉模块贡献。
2. 这是观察预算或 frame-selector 与模型的联合增益；除非同一输入 token/帧预算下做配对，不能只归因权重。应报告质量—帧数—延迟 Pareto。
3. 文本段落可能补偿总体平均，但财报/发票里的表格结构已不可用；生产 gate 应把关键模块设 non-inferiority floor，而非只看 overall。
4. ERQA 是给定静态观察后回答多选；OSWorld 要在有状态桌面中执行动作、观察反馈并到达终态。后者多了 policy、工具、恢复和环境噪声。

</details>

一句话验收：**视觉 benchmark 的分数只有在证明模型真的依赖那张图、那段时间和那个空间关系后，才配叫多模态证据。**
