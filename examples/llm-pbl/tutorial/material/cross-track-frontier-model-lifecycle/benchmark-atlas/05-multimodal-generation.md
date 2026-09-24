# 分册 05：多模态生成 benchmark

> **核心问题**：生成模型没有唯一 gold answer，怎样把“好看”拆成可诊断、可复现又不冒充人类偏好的证据？
> **范围**：文生图、图像编辑、文/图生视频与音频生成；模型发布动态水位见 [发布证据账](RELEASE_LEDGER.md)。
> **快照**：2026-09-22；“当前/截至快照”的水位只对该日期成立。

## 1. 先把“生成质量”拆成五个坐标

一张图或一段视频的质量至少包含：

1. **技术质量**：清晰度、噪声、压缩伪影、闪烁；
2. **条件遵循**：对象、数量、属性、文字、空间/时间关系是否命中；
3. **一致性**：身份、背景、结构、运动和音画是否随时间保持；
4. **偏好与美学**：构图、风格、真实感与可用性；
5. **安全与溯源**：敏感内容、偏见、版权/水印与生成披露。

FID、CLIPScore 或一个 VLM judge 都只覆盖其中一部分。生产发布应把自动代理、成对盲评、失败 taxonomy 和成本延迟
分栏，而不是先加权成一个“总质量”。

## 2. 图像生成与编辑

### 2.1 GenEval 2：对象原语与组合遵循

- **基本信息**：官方 [GenEval 2](https://github.com/facebookresearch/GenEval2) 含 800 个不同组合度的提示，覆盖对象、属性、关系与计数；它更新了原始 553-prompt GenEval。
- **能力项 / 阶梯**：L4 受控组合生成；比自由美学更像“视觉单元测试”。
- **课程合成例题**：`画一只蓝色杯子在两本红书左侧，桌上恰好还有三颗梨`。
- **打分**：Soft-TIFA 把 prompt 分解成视觉问答原语，再软聚合各原语是否满足。
- **厂商采用**：原始 GenEval 常见于开放图像模型卡；使用时必须注明 GenEval 还是 GenEval 2、检测/VQA scorer revision 和每 prompt seed 数。
- **局限**：VQA scorer 可能看漏小物体/文字；对象原语得分高不代表摄影、美学、手部结构或商业可用。
- **最高水位口径**：只在相同 prompt suite、分辨率、每题样本数和 Soft-TIFA checkpoint 内排名；不把 legacy GenEval 分数拼入 GenEval 2。

### 2.2 T2I-CompBench++：属性绑定与复杂关系

- **基本信息**：[官方仓库](https://github.com/Karine-Huang/T2I-CompBench) 覆盖颜色/形状/纹理绑定、空间关系、非空间关系和复杂组合；++ 版本扩展 prompt 与 evaluator。
- **能力项 / 阶梯**：L4 compositional binding；检查模型是否把“红色”绑给正确对象，而非只在画面中出现红色。
- **课程合成例题**：`金属质感的绿色茶壶在木碗后面；碗里没有水果`。
- **打分**：不同维度使用检测、CLIP/VQA 或 3-in-1 等专门指标，再分维度报告。
- **厂商采用**：DALL·E 3、Stable Diffusion 3 等发布曾采用该类组合指标；当前模型必须重跑同 revision，不能抄历史对手分。
- **局限**：多个 scorer 的误差会被总分掩盖；否定、文字渲染、细粒度关系和开放世界长尾仍弱。
- **最高水位口径**：保留每个子维度，不用总平均隐藏“对象有了但绑定错了”。

### 2.3 DPG-Bench：长而密的 prompt 遵循

- **基本信息**：[ELLA 官方仓库](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench) 提供 dense prompt、分解问答和 evaluator；规模应从固定 commit 的 `dpg_bench.csv` 复算，不能只写流传的题数。
- **能力项 / 阶梯**：L4 dense semantic coverage；一条 prompt 同时含主体、动作、服饰、场景、视角与光照。
- **课程合成例题**：`雨夜车站，一位穿橙色雨衣的老人左手提透明箱，箱内两只白鸽；低机位、霓虹倒影`。
- **打分**：把 prompt 转成多个 yes/no 问题，以 mPLUG 等 VQA 模型逐点裁决并聚合 DPG score；常见协议每 prompt 生成多图。
- **厂商采用**：常见于中文/开源 T2I 模型发布；scorer、prompt revision、图片数与 grid 方式都是合同字段。
- **局限**：judge 既可能漏检，也可能被画面中文字诱导；dense prompt 得分不等于自然用户 prompt 的偏好。
- **最高水位口径**：不同 VQA judge 或 prompt rewrite 不可直接横比；只公布同 commit 复算的 protocol-specific 水位。

### 2.4 人工 GSB / pairwise blind：最终产品偏好

- **基本信息**：给评审同 prompt 的 A/B 结果，选择 Good/Same/Bad 或胜/平/负；例如 [HunyuanImage 3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) 官方披露了 1,000 prompts、单次生成、100+ 专业评审的 GSB，以及独立的结构语义自动评测。
- **能力项 / 阶梯**：L4 综合可用性；能够覆盖自动 scorer 没定义的构图、真实感和明显瑕疵。
- **课程合成任务**：对同 seed policy 的两组匿名图，分别判断 prompt 遵循、结构、美学和文字可读性，不只问“更喜欢哪张”。
- **打分**：win/tie/loss、胜率或 Bradley–Terry；应给评审数、prompt 数、每 prompt seed、置信区间与一致性。
- **厂商采用**：图像/视频发布广泛使用；HunyuanImage 3.0 还把 SSAE 与 GSB 分栏，是比单一自动分更好的报告形态。
- **局限**：自有 prompt、评审招募、显示设备、位置顺序和 cherry-pick 都可改变结论；厂商内部 GSB 属 C-internal。
- **最高水位口径**：不同对手池的胜率/Elo 不可拼接；只能说“在该盲评池与协议中领先”。

## 3. 视频生成

### 3.1 VBench / VBench 2.0：把“视频好”拆成维度

- **基本信息**：[VBench 官方仓库](https://github.com/Vchitect/VBench) 的经典套件使用标准 prompt suite，覆盖 16 个维度；完整经典口径常见为 946 prompts。VBench 2.0 又增加物理、常识、人类运动和创意组合等内在可信度。
- **能力项 / 阶梯**：L4 视频技术质量 + 语义 + 时序；维度包括主体/背景一致、闪烁、运动平滑/幅度、美学、画质、对象/动作/颜色/空间/场景/风格等。
- **课程合成例题**：`玻璃球从斜坡滚下，撞倒两块木牌后停在红线前；镜头持续跟随且球的花纹不变`。
- **打分**：每维使用特定模型/视觉算子；质量分与语义分经固定归一化和权重形成 total score。
- **厂商采用**：HunyuanVideo 等开放视频模型常报告 VBench；必须注明 VBench、VBench++、2.0、I2V 还是 Long。
- **局限**：自动 evaluator 可能奖励“少运动所以少闪烁”；归一化上下界与权重会改变总榜；短视频高分不证明长程叙事。
- **最高水位口径**：以官方同版本 leaderboard 为准并记录日期；不同帧数、时长、分辨率、prompt rewrite 和采样数不可合并。

### 3.2 专业盲评：Text Alignment / Motion / Visual Quality

- **基本信息**：[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) 发布曾用 1,533 prompts、单次生成、60+ 专业评审，在文字遵循、运动质量、视觉质量和 overall 上比较。
- **能力项 / 阶梯**：L4 端到端视频体验；尤其补足 VBench 自动指标对物理与自然运动的盲点。
- **课程合成任务**：把“主体是否正确”“运动是否合理”“是否清晰稳定”分开打分，再额外给 overall，而非让 overall 反推三个子项。
- **打分**：通常是成对胜率/GSB；应固定视频时长、分辨率、FPS、音频、生成次数与展示顺序。
- **厂商采用**：视频生成发布的常见主证据；HunyuanVideo 官方同时提醒其评测使用高质量版而非当时公开 fast 版。
- **局限**：不同产品版本、默认 prompt enhancer 或候选挑选会混入系统差异；内部 prompt 不支持公共 SOTA。
- **最高水位口径**：只对同一次盲评池作相对判断；不把“对 A 胜率”和“对 B 胜率”直接排序。

### 3.3 FVD / temporal feature distance：分布像不像，不是题意对不对

- **基本信息**：FVD 比较真实与生成视频在预训练视频特征空间中的分布距离，可视作视频版 FID 思路。
- **能力项 / 阶梯**：L4 数据集级视觉/运动分布相似度。
- **课程合成例题**：不是逐 prompt 问答；用一批“跑步”真实视频与同条件生成集比较特征均值/协方差。
- **打分**：距离越低通常越好；必须固定 feature extractor、clip 长度、预处理和样本量。
- **厂商采用**：研究论文常见，现代产品发布越来越倾向与 prompt adherence、VBench 和人工盲评并用。
- **局限**：对 encoder、样本量和预处理敏感；可以通过生成常见但不遵循 prompt 的视频获得较好分布距离。
- **最高水位口径**：跨数据集或 extractor 的 FVD 没有共同尺度。

## 4. 音频与视听联合生成

### 4.1 FAD / CLAPScore：音质分布与文字对齐

- **基本信息**：FAD 比较真实/生成音频 embedding 分布；CLAPScore 用对齐模型估计文本—音频相似度。
- **能力项 / 阶梯**：L4 音质/事件分布与语义遵循。
- **课程合成例题**：`雨声由远及近，2 秒处玻璃碎裂，随后右声道传来犬吠`；分别检查事件存在、时间、空间声道和整体音质。
- **打分**：FAD 越低、CLAP 相似度越高；需固定采样率、声道、时长、embedding checkpoint 和裁剪策略。
- **厂商采用**：音频/视频带声模型常用，但 H3 这类视听联合系统还需事件时间对齐与人工听评。
- **局限**：CLAP 可能只感知“有狗叫”而忽略 2 秒/右声道；FAD 不证明 prompt 遵循或语音可懂度。
- **最高水位口径**：不同 reference corpus、时长和 encoder 不可横比。

### 4.2 视听事件对齐：峰值重合只是下界

- **基本信息**：比较视觉事件时间与音频能量/事件检测时间，或由人工判断因果同步。
- **能力项 / 阶梯**：L4 audio-video temporal binding。
- **课程合成例题**：画面中锤子三次击钉，三次冲击声应分别落在接触帧附近；环境底噪不应制造假峰。
- **打分**：事件 precision/recall、时间偏差分布、同步偏好；需报告标注容差窗口。
- **厂商采用**：适用于原生音视频生成，但当前跨厂公共协议仍不统一。
- **局限**：能量峰值不是语义事件；无声事件、持续声和音乐节拍会破坏简单峰值匹配。
- **最高水位口径**：在没有公共同协议榜单时明确写“暂无可比水位”，比创造一个综合分更诚实。

## 5. 一个最低可用的生成评测包

```text
frozen prompt suite + prompt taxonomy
seed list + samples/prompt + sampler/steps/guidance
resolution + duration + FPS + audio contract
raw generations + output SHA256
automatic metrics by dimension
blind pairwise rubric + rater agreement
failure taxonomy + safety review
latency/VRAM/cost + cherry-pick policy
```

模型只要改了 prompt rewrite、negative prompt、upscaler、frame interpolation 或候选选择器，系统合同就变了。可以比较，
但必须把结论写成“生成系统”而非“DiT 权重”的单因素增益。

## 费曼自检

1. 为什么 FID/FVD 下降和 prompt adherence 上升可能同时不成立？
2. VBench 总分更高，为什么仍可能生成更少运动的视频？
3. 为什么每 prompt 生成 8 个候选后挑最好的一张，必须和单次生成分开报告？
4. HunyuanImage 的厂商 GSB 有什么价值，为什么又不能当公共 SOTA？

<details>
<summary>参考答案</summary>

1. 分布距离只问生成集在某特征空间是否像 reference，模型可以生成常见、高质量但与具体 prompt 无关的内容；遵循需对象/关系/文字等条件检查。
2. 闪烁与主体一致指标可能偏爱静态画面；若 dynamic degree 没有作为硬门，总加权可由其他维度补偿。应看维度向量与 motion floor。
3. best-of-8 引入搜索与选择器，成功机会、token/GPU 成本都提高；它估计“预算化系统最好结果”，单次生成估计默认用户可靠率。
4. GSB 能捕捉自动指标遗漏的综合可用性，并揭示厂商目标场景；但自有 prompt、评审池、对手与未公开逐样本结果使其不可独立复验，只能算 C-internal。

</details>

一句话验收：**生成评测的最小单位不是一张好图，而是 prompt、seed、采样系统、分维度裁决与盲评共同绑定的证据包。**
