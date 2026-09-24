# 多模态 Long Context：何时真需要 512K/1M，何时应该用 RAG

> 证据快照：2026-09-15。本章不把“模型能接收 1M token”等同于“能稳定利用每个 token”，
> 也不把 RAG 视为 long context 的替代品。目标是在证据完整性、质量、延迟和成本之间做可检验选择。
> 配套的纯标准库实验见 [nano-long-context-routing L0](nano-long-context-routing/tutorial_L0.md)。

本章的 token 账从模型 processor 的输出开始。文件码率、解码放大、FPS/分辨率抽样与 content-group split 见
[《大规模多模态数据管线》](MEDIA_DATA_PIPELINE.md)；先决定保留哪些证据，再决定这些证据进入 long context 还是 RAG。

## 0. PBL 任务：不先选架构，先识别证据形状

给定三个请求：

1. 从 10 万份合同中查某一条客户的解约条款；
2. 判断一段 90 分钟会议中第 12 分钟的承诺是否在第 78 分钟被推翻；
3. 审查一个仓库改动是否同时违反 API schema、调用方假设和历史迁移约束。

任务 1 的证据很稀疏，通常 RAG 性价比更高；任务 2 需要时序和远距依赖，单次 top-k
容易丢掉因果链；任务 3 往往适合“代码搜索/图索引 → 候选文件与邻域 → 较长上下文联合推理”。
先判断证据是**稀疏局部**还是**稠密全局**，再决定是 RAG、long context 还是 hybrid。

## 1. 512K/1M 提供的是容量上限，不是有效利用保证

对长度 $N$ 的 dense self-attention，prefill 的主要 attention 关系数随 $N^2$ 增长；KV cache 与输入长度
近似线性增长，而生成每个新 token 也要读更长的 KV。稀疏/线性 attention、sequence parallel、
KV 量化和 prompt cache 能改变常数或某一项复杂度，不会使多余上下文变成免费。

更长窗口至少有四个彼此独立的问题：

- **capacity**：序列能否被接收，不 OOM；
- **retrieval**：模型能否找到埋在中间的证据；
- **reasoning**：能否联立多处、跨越较远的证据；
- **economics**：首 token 延迟、显存、计费和吞吐是否可接受。

[Lost in the Middle](https://arxiv.org/abs/2307.03172) 表明相关证据所在位置会影响长上下文利用；
[RULER](https://arxiv.org/abs/2404.06654) 进一步把单 needle 扩展为多种检索与聚合任务。因此官方声称的
maximum context 不应直接进入“有效上下文”一列。

## 2. 多模态确实会把 token 预算迅速吃掉

### 2.1 图像不是“一张一 token”

[Qwen3-VL 官方 processor 用法](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md)给出空间压缩口径 32，
所以经 resize 后的单图视觉 token 可粗略写成：

$$
N_{image}\approx\frac{H'W'}{32^2}.
$$

本课 Qwen3-VL L1 的 768×448 图像实测为 336 visual tokens；1024×1024 在不考虑 padding/尺寸对齐时
约为 1024 token。一张图通常不需要 512K，但高清扫描件、数百页报告、多图对照和 GUI 轨迹会线性累加。

### 2.2 视频是“帧数 × 每帧 token”

用 $f$ 表示抽帧率、$s$ 表示每帧视觉 token、$m_t$ 表示时间合并率，最小预算为：

$$
N_{video}\approx\frac{duration\times f\times s}{m_t}.
$$

例如 30 分钟、1 FPS、每帧 196 token、无额外时间合并的教学账已经是 352,800 token；2 FPS 则
超过 700K。真实 processor 会动态 resize、抽帧或做 temporal merge，所以必须记录实际 `grid_thw`，
不能把这个示例写成某个模型的固定比率。[Qwen3-VL 报告](https://arxiv.org/abs/2511.21631)
把长文档和长视频列为 256K 原生上下文的主要用途，这比“更长聊天记录”更能解释多模态需求。

### 2.3 音频也有自己的 token rate

音频 encoder/codec 可能按固定 Hz 生成 latent；[H3 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)
公开的 audio latent 是每声道 40 Hz。这个口径下，一小时是每声道 144K latent，双声道合计 288K channel-major rows。
这只是 rate 换算，不表示 H3 支持一小时单次生成。它是 H3 生成合同，不能外推到所有音频理解模型；但它说明，
保留说话人、语调、环境声和时间对齐时，输入远比一份 ASR 文本长。只用 transcript 更便宜，但会丢掉非语义证据。

## 3. 哪些现实任务真能用到超长上下文

| 任务 | 为什么会长 | 为什么不一定能只靠 RAG |
|---|---|---|
| 长视频/会议/播客审查 | 帧、字幕、说话人、音频事件与时间戳同时累加 | 问题可能依赖遥远的前因后果，检索一个片段不够 |
| 多页扫描件/图表报告 | 每页同时有 OCR、layout、图表和图注 | 条款、表头、附注可跨页引用，错 chunk 会破坏结构 |
| 代码库改造/事故追溯 | 代码、schema、issue、log、diff 和测试共同构成证据 | 需要联立多文件不变量，但全库直塞通常仍不是最优 |
| 长时间 Agent 任务 | 观测、工具返回、失败、计划与用户约束不断增长 | 某个早期决定可能改变后续语义，需要 lineage，不只是语义相似片段 |
| 整体一致性/穷尽性审计 | 任务要求“全部”而不是找一个答案 | top-k 天然可能漏项，需要 map-reduce 或全局覆盖证据 |
| many-shot / 稀有规则的 in-context learning | 大量样例共同定义临时任务 | 在不训练权重时，样例集就是任务规格 |

[Gemini 1.5 技术报告](https://arxiv.org/abs/2403.05530) 用多文档、长视频、长音频和大代码库验证 million-token
能力。这些是有意义的上限用例，却不能证明日常问答也应该直接投喂 1M token。

## 4. 什么时候 RAG 明显更划算

当 relevant-token density

$$
\rho=\frac{\text{answer-required tokens}}{\text{corpus tokens}}
$$

很低，并且证据可以用关键词、embedding、metadata 或结构化过滤找到时，RAG 通常胜出：

- 一个问题只依赖百万文档中的几个片段；
- 同一知识库需被反复查询，索引成本可以摊销；
- 数据持续更新，需要新鲜度、权限过滤、引用和可删除性；
- TTFT、显存或按输入 token 计费是主要约束；
- 问题可以先被转成明确的检索条件。

但 RAG 把错误提前到 retrieval stage。一旦正确 chunk 没进 top-k，后面的 LLM 无法补救。
[检索增强与长上下文对照研究](https://arxiv.org/abs/2407.16833) 的重要结论不是某一方永远胜出，而是资源充足时
long context 可提供更高上限，RAG 成本更低，路由式 hybrid 能改善质量—成本折中。

## 5. 为什么 hybrid 通常是生产默认答案

```text
corpus / video / trajectory
  → 结构化解析（页、章节、shot、speaker、symbol、event）
  → 粗检索（BM25 + embedding + metadata/graph）
  → 取回命中项的邻域、上级摘要和时间前后文
  → 在可控的长上下文中联合推理
  → 证据覆盖/引用检查；缺口触发第二轮检索
```

这个系统同时避免两个瓶颈：

- RAG 的 **selection bottleneck**：只给一个孤立 chunk，丢掉章节、表头、时序和代码调用邻域；
- long context 的 **utilization bottleneck**：把全部原始数据塞入，让模型在噪声中找证据。

[上下文化检索](https://www.anthropic.com/engineering/contextual-retrieval) 也可以理解为一种边界修复：建索引前为 chunk
补少量文档级语境，减少切块后的指代和语义丢失。对视频的类比做法是为 shot 保留时间范围、人物和前后事件，
而不是只向量化一帧。

## 6. 一个可验收的三路对照

不要用不同问题比较 RAG 和 long context。固定同一模型、问题、语料与输出合同：

先运行 [L0 selector surrogate](nano-long-context-routing/tutorial_L0.md)，观察 evidence recall 怎样先于模型推理决定
答案是否可得；再把 oracle readout 依次替换为真实 tokenizer、retriever 和 LLM。

| 实验臂 | 输入 | 主要失败 |
|---|---|---|
| A whole-context | 按原顺序放入全部可容纳语料 | lost-in-the-middle、高 TTFT/成本 |
| B RAG | top-k chunks | retrieval miss、chunk 失去邻域 |
| C hybrid | top-k + parent/neighbor + 局部原文/帧 | 系统更复杂，但可同时修复两类失败 |

每条问题报告：

- answer correctness 和 abstention；
- evidence recall：必要证据有多少真的进入模型；
- citation/grounding correctness；
- input tokens、retrieval latency、TTFT、decode latency 和计费；
- 证据位置和跨证据距离；
- completion rate，不删除 OOM、timeout 和 parser failure。

至少构造四种样例：单个稀疏 needle、跨远程序列的两跳问题、需要全量遍历的审计、与问题很像但无关的干扰项。
只测 needle retrieval 会高估真实的全局推理能力。

## 7. 简化决策树

```text
证据是否位于大型、可重复查询的语料库？
├─ 否：小于可控预算 → 直接 long context，但要测位置敏感性
└─ 是：问题只需少数局部证据？
   ├─ 是 → RAG + rerank + citation
   └─ 否：依赖顺序、多跳或穷尽覆盖？
      ├─ 是 → hierarchical retrieval/map-reduce + 局部 long context
      └─ 不确定 → 三路 paired evaluation，用质量—成本前沿选择
```

对大多数企业知识问答，默认不应是 1M raw prompt，而是结构化检索加可控长上下文。
对长视频、长 Agent 轨迹与全局审计，超长窗口是有价值的安全裕量，但仍需要索引、压缩、分层和工具调用。

## 8. 费曼自检

1. 一张图只有几百个 token，为什么视频会轻易逼近 1M？
2. RAG 只给模型 8K token，为什么结果可能比直接给 512K 更好？
3. 什么任务不应使用普通 top-k RAG？
4. prompt caching 为什么没有让 1M context 变成免费？
5. 怎样证明 hybrid 的收益来自证据选择，而不是给了更多 token？

<details>
<summary>参考答案</summary>

1. 视频 token 近似是抽样帧数与每帧 token 的乘积；时长和 FPS 稍增就线性放大。动态 resize、镜头筛选和时间合并
   是为了控制这本账，但过强压缩又可能丢失短暂事件。
2. RAG 先移除无关 token，降低噪声、TTFT 和 attention 竞争。前提是必要证据被召回；若 retrieval miss，8K 内的推理再强也无用。
3. 证据遍布全文、需要时间顺序、跨段多跳、穷尽性检查，或查询本身无法表达要找的证据时，孤立 top-k 容易失败。
   应加 parent/neighbor、时间窗、图检索或 map-reduce。
4. cache 可避免重复 prefill 某些前缀，但缓存需占用存储/KV，decode 仍要在长历史上取数，而首次请求的视觉编码、prefill 与网络传输不会消失。
5. 固定最终 LLM input-token 预算，比较随机/截断上下文、top-k 与 top-k+neighbor；同时报 evidence recall、answer correctness 和延迟。
   若 hybrid 在等 token 下提高必要证据覆盖并提高答案质量，才能归因于选择。

</details>

## 9. 完成标志

学习者不应用“模型支持 1M”回答架构问题，而应交付：语料规模、实际模态 token 账、证据密度、顺序/多跳需求、
retrieval recall、有效上下文测试、TTFT/显存/计费，以及 whole-context、RAG、hybrid 的同条件对照。
