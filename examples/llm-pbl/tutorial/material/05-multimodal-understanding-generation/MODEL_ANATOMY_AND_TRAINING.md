# 多模态模型解剖与训练：从 readout 到现代 SOTA 配方

> 资料快照：2026-09-15。本文回答四个容易混在一起的问题：readout 是什么；视觉 encoder/VAE 是否开放、怎样训练；
> 模型参数中的“视觉占比”怎样计算；当前多模态训练是否仍只是 LLaVA 式多阶段 VQA。
> “最新发布”“榜单 SOTA”“开放权重”“本课程可复现”是四个不同标签，进入实验前仍须按 revision 重核。

## 0. PBL 场景：你要训练的究竟是哪一段

假设团队拿到 8×L20，有三项需求：让 8B LLM 看懂截图、让 DiT 生成带文字的图片、验证 H3 的视听生成合同。
最昂贵也最常见的错误，是把三个任务都写成“训练一个多模态模型”。先完成这张因果图：

```text
理解：pixels → vision encoder → visual tokens → connector → LLM hidden states → readout → text/box/action
生图：pixels → VAE encoder → image latents ─┐
      prompt → condition encoder ───────────┼→ DiT learns velocity/noise → sampler → VAE decoder → pixels
视频：在 image latents 上增加 time，并加入时空压缩、长序列与跨帧一致性
H3： context encoder + video/audio VAE → packed rows → joint flow → separate media decoders
```

图上的每个箭头都有不同训练目标。需求失败时，应先定位信息在哪一段丢失，再决定解冻谁；直接做全参数 SFT 通常是
最高成本、最低可诊断性的第一步。

## 1. readout：把内部表示变成任务答案的路径

readout 不是某个厂商组件名，而是一个功能概念：**从已经融合的 hidden states 中选取、聚合并映射出任务输出的规则**。

- 分类模型可能用 `[CLS] hidden → linear head → class logits`；
- decoder-only VLM 通常用最后位置的 hidden state 经共享 `lm_head` 产生下一个 token，再自回归得到文字、坐标或动作；
- grounding 模型可以把 box 离散化为文本 token，也可以另接坐标 head；两者都是 readout；
- 本课程 L0 用问题 query 对视觉 token 做 attention，再对 value 加权求和；这是可检查的手工 readout。

因此，“LLM 是 decoder，所以 readout 就是 decoder”只对了一半。decoder 负责反复更新整段表示，readout 说明**读哪个状态、
用什么头、以什么输出协议解释它**。真实 VLM 的生成式 readout 可简写为：

$$
h_i = \operatorname{LLM}(v_{1:m},q_{1:n},y_{<i}),\qquad
p(y_i)=\operatorname{softmax}(W_{\text{vocab}}h_i).
$$

若 visual tokens 根本没有保存小字，后面的 readout 再强也读不回来；若视觉证据已在 hidden state 中但坐标序列格式错误，
问题更可能位于 SFT/readout contract。区分两者需要 hidden-state probe、image-swap 和格式正确率，而不是只看最终 accuracy。

## 2. 三种“encoder”不能混为一谈

| 组件 | 压缩什么 | 典型目标 | 推理时输出给谁 | 常见开放情况 |
|---|---|---|---|---|
| VLM vision encoder | 像素 → 语义 patch/token | contrastive、caption/识别及多模态 NTP | connector/LLM | 许多开放权重 VLM 随整包发布 |
| DiT condition encoder | prompt/参考图 → 条件表示 | 先做语言/视觉预训练，生成训练时常冻结或低学习率 | DiT cross/joint attention | 可能复用开放 VLM，也可能仅随 pipeline 分发 |
| VAE encoder/decoder | 像素/帧 ↔ 连续 latent | reconstruction + perceptual，部分模型再用 adversarial/KL | encoder 给 DiT 训练 latent；decoder 还原媒体 | 常随开放生成 checkpoint 分发，原始数据和完整预训练脚本未必开放 |

“权重可下载”只证明可执行，不等于训练资产完整开放。至少分别检查：模型权重、架构源码、训练入口、数据配方、数据本体、
损失与超参、许可证。缺少数据与完整 recipe 时，可以复现 inference 或做 LoRA，不能声称从零复现了 foundation model。

一个很好的具体反例来自 [Qwen-Image 技术报告](https://arxiv.org/html/2508.02324)：它用冻结的 Qwen2.5-VL 提取语义条件，
又用 VAE encoder 保留可重建细节；其 VAE 复用 Wan-2.1 encoder 并冻结，只微调 image decoder，损失使用重建与感知项。
这说明 encoder 并不总是随主 DiT 一起端到端更新，也说明“语义相似”和“像素可还原”需要两套表示。

## 3. 参数规模：先定义分母，再谈视觉占比

“视觉相关参数占多少”至少有四种口径：独立 vision tower、所有会处理视觉 token 的参数、每 token 激活参数、完整 pipeline
驻留参数。对统一 Transformer 来说，大量 LLM/DiT 权重同时服务多个模态，强行切成“视觉”和“非视觉”会失真。

| 公开模型 | 可核验参数口径 | 能得出的结论 | 不能直接得出的结论 |
|---|---|---|---|
| InternVL3.5 | 多数档 vision tower 约 0.3B，38B/241B 档约 5.5B | 独立视觉塔占总参数约 1%–27%，且随语言主干变化很大 | 其余参数都与视觉无关 |
| Qwen3-VL-235B-A22B | 235B total、22B active；vision/merger 与 LLM 联合工作 | 应同时报告 total、active 与视觉 token 成本 | 用一个“视觉百分比”解释推理成本 |
| Qwen3.8-Flash-Next | 125B 主模型 + 51B n-gram embedding，6B/token active | MoE 的磁盘、驻留和每 token 计算是三本账 | 只用 active=6B 推出可在小显存完整加载 |
| Qwen-Image（初代） | 7B 条件 VLM、VAE encoder 54M/decoder 73M、MMDiT 20B | DiT 主干远大于 VAE，VAE 小却决定重建上限 | 把条件 VLM 全算作“生成 decoder” |
| HunyuanImage-3.0 | 80B total、13B active 的原生多模态自回归生图/编辑模型 | 生图不只有 VAE+DiT/flow 一条架构路线 | 把 13B active 当成完整权重驻留量 |
| Wan2.2-A14B | 两个约 14B denoising experts，约 27B total、每步约 14B active | expert 按噪声阶段切换，active 与 total 不同 | 套用语言 MoE 的逐 token routing 解释 |
| HunyuanVideo-1.5 | 8.3B DiT，另有 3D causal VAE 与条件组件 | “8.3B”主要是生成主干口径 | 把 8.3B 当完整 pipeline 驻留量 |
| MiniMax H3 | 32B Qwen3-VL encoder + 33B Omni Transformer，另有视听 VAE；Transformer 中约 13B 为 AdaLN 分支 | H3 是多组件系统，不能只报 33B | 把共享 Transformer 精确分摊成视觉/音频百分比 |

InternVL 的逐组件数字来自 [InternVL3.5 技术报告](https://arxiv.org/html/2508.18265)；Qwen3-VL 与 Qwen3.8 的参数口径分别见
[官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)和[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)；
Qwen-Image、HunyuanImage-3.0、Wan2.2、HunyuanVideo-1.5 与 H3 的数字分别见各自
[技术报告](https://arxiv.org/html/2508.02324)、[官方仓库](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)、
[官方仓库](https://github.com/Wan-Video/Wan2.2)、
[技术报告](https://arxiv.org/html/2511.18870)和[官方仓库](https://github.com/MiniMax-AI/MiniMax-H3)。

真正与推理 ROI 相关的账是：权重 bytes、KV/attention 激活、visual token 数、并行通信、VAE decode 峰值和成功率。
独立视觉塔只占几个百分点，也可能因高分辨率产生上万 token，成为 prefill 的主要成本。

## 4. 当前 VLM 训练：保留 LLaVA 骨架，但主训练已不是 VQA 微调

原始 LLaVA 的两阶段方案很清楚：59.5 万图文对只训练 projector；随后用 15.8 万视觉指令样本更新 projector 和 LLM，
视觉 encoder 始终冻结。它证明了“预训练视觉塔 + 语言模型 + 小桥接层”可以快速获得视觉对话能力，详见
[LLaVA 论文](https://arxiv.org/html/2304.08485)。

现代开放前沿通常采用下面的课程：

| 阶段 | 常见可训练参数 | 数据/目标 | 这一阶段主要学什么 |
|---|---|---|---|
| A. encoder 预训练/续训 | vision/audio encoder | 图文对比、caption、OCR、视频/音频对齐 | 保留可被下游读取的感知证据 |
| B. adapter warm-up | connector/merger；主干冻结 | caption、OCR、短图文对 | 建立模态坐标接口，避免随机 connector 冲击 LLM |
| C. multimodal CPT | 常见做法是全参数解冻 | 图文交错、文档、grounding、VQA、STEM、视频和纯文本 | 把跨模态条件依赖写入整个模型 |
| D. context/skill curriculum | 全参数或分组学习率 | 高分辨率、长视频、多图、GUI、工具轨迹 | 学长序列、空间/时间和行动合同 |
| E. SFT / CoT / distillation | 全参、部分参数或 LoRA | 高质量回答、推理轨迹、专项 teacher 输出 | 学回答策略、格式与任务分解 |
| F. preference / online RL | policy 参数 | preference pair、规则/执行器/verifier、人类反馈 | 压低坏输出并改善交互；不能补回缺失视觉证据 |

大多数 decoder-only VLM 仍把任务统一为对目标文本 token 的 next-token prediction：

$$
\mathcal L_{\mathrm{NTP}}=-\sum_{i\in\text{target text}}
\log p_\theta(x_i\mid x_{<i},v_{1:m}).
$$

VQA 仍在数据混合中，但更多时候只是 serialization interface。caption、OCR 字符、box、时间戳、GUI action、拒答和 CoT
都能序列化成 token。训练上限越来越取决于数据混合、反事实 grounding、分辨率/长度 curriculum 与 post-training，
而不是把更多 benchmark QA 拼进一个 JSON。

### 两个公开配方锚

- [Qwen3-VL](https://arxiv.org/html/2511.21631)：先只训 merger（约 67B token），再全参数训练约 1T token；随后继续
  约 1T token 的 32K 长上下文训练和 100B token 的 256K 适配。数据混合包含纯文本、图文交错、grounding、VQA、
  STEM、视频与 Agent；后训练再接长 CoT SFT、teacher distillation 与 RL。
- [InternVL3.5](https://arxiv.org/html/2508.18265)：预训练联合更新全部参数，约 250B token；SFT 约 56M samples/
  130B token，再用 offline MPO 暖机和 online GSPO 精炼。它仍是 ViT–MLP–LLM，但优化对象已从 projector 扩到完整策略。

原生 omni 会进一步把音频/视频从早期就混入训练。[Qwen3.5-Omni 报告](https://arxiv.org/html/2604.15804)公开的流程是：
固定 LLM 训练 adapter/encoder；再全参数混合约 4T token；再做 262K 长上下文；后训练包含专项 teacher 蒸馏、跨模态
on-policy distillation 与交互 RL。这与“最后加一批语音 VQA”有本质区别。

## 5. Image/Video DiT 怎样训练

生成模型的典型训练也多阶段，但主目标不是回答文本：

1. **先训练或复用 VAE**：让 $D(E(x))\approx x$，在压缩率与重建细节间取舍；视频 VAE 还要处理时间因果与分块边界。
2. **训练基础 DiT/flow**：采样数据 latent $z_1$ 和噪声 $z_0$，构造 $z_t=(1-t)z_0+tz_1$，学习条件速度
   $v^*=z_1-z_0$。文本/VLM encoder 与 VAE 常被冻结，主要更新 DiT。
3. **逐步扩分辨率、时长与数据质量**：先低分辨率学分布，再高分辨率学细节；图像与视频混训用于保留静态质量和运动。
4. **加入编辑/控制任务**：T2I、I2I、TI2I 或 T2V/I2V 共训；首帧、参考图、姿态和音频成为额外条件。
5. **post-training**：用精选数据、偏好对、reward model/RL、few-step distillation 调整审美、指令遵循与推理成本。

[Qwen-Image](https://arxiv.org/html/2508.02324)展示了从 256p 到 640p/1328p、从非文字到文字密集数据、再到
T2I/I2I/TI2I 多任务和生成 RL 的渐进路线。[HunyuanVideo-1.5](https://arxiv.org/html/2511.18870)则联合训练
T2I/T2V/I2V，使用大规模清洗与重标注数据、渐进 pre/post-training 和 Muon；其开放仓库提供继续训练/LoRA 入口，
但开源训练脚本不等于原始数据与全量 foundation run 可复现。

MiniMax H3 公开了 encoder/VAE、packed sequence、33B Omni Transformer、视听双 flow 和部分最终训练设计；目前没有公开
足够完整的数据组成、阶段 token/step、优化器和消融来重建全套训练 recipe。因此课程只把已公开架构写成事实，不根据
checkpoint 文件反推未披露训练过程。

## 6. “最新”快照：怎样选教学锚而不追版本号

| 厂商/系列 | 截至快照日的定位 | 课程决策 |
|---|---|---|
| Qwen understanding | Qwen3.8-Flash-Next 是 2026-08-26 开放的最新多模态架构预览；Qwen3-VL 是较早的专用 VLM 系列 | 保留 2B Qwen3-VL 作为低成本实证；前沿章节追踪 Qwen3.8，不冒充同 checkpoint |
| Qwen omni | Qwen3.5-Omni 是更新的音视频理解/语音输出系统，但公开服务、报告与可下载权重边界不同 | 用报告讲原生 omni 训练；本地实验只选已确认开放 checkpoint |
| Qwen image | Qwen-Image-2.0 是更新的模型/报告；官方开源仓库当前本地 quick start 仍以 2512 为最新 T2I 权重 | 2.0 作前沿研究，2512 作 L2 可复现基线 |
| InternVL | InternVL3.5 仍是最新专用开放 VLM 家族；InternVL-U 是 2026-03 开放的更新统一理解/生图/编辑研究线 | 二者并列，不能用 InternVL-U 的“更新”否定 3.5 的专用 VLM 定位 |
| Wan video | Wan3.0 是当前托管 All-in-One 视频主线；Wan2.7 为上一代 API；Wan2.2 仍是官方开源组织的主要通用权重/源码基线 | API 评测跟踪 Wan3.0，本地机制/资源实证保留 Wan2.2，两者不合并为“已复现” |
| Hunyuan text | Hy4 preview 是 770B total/49B active、1M context 的开放文本 LLM | 用于长上下文/稀疏 attention 对照，不写成 VLM/Video DiT |
| Hunyuan image | HunyuanImage-3.0 是开放的 80B/13B-active 原生多模态自回归生图线 | 列为 DiT 之外的架构对照，先做资源/revision gate |
| Hunyuan video/3D | 当前服务分为 HY-Video-1.5 与 HY-3D-3.1；HunyuanVideo-1.5 是开放视频基座，Buffalo 1.0 是统一 3D 理解/生成/编辑研究线 | 不再用“Hunyuan 最新”跨文本、图像、视频和 3D 作总排名 |
| MiniMax | M3 是理解/Agent 方向的原生多模态模型；H3 是 2026-07 发布的开放视听生成系统 | H3 capstone 只负责生成系统，不与通用 VLM 榜单直接排序 |

“SOTA”必须带任务、数据版本、输入预算、推理预算和开放性。厂商平均榜单只能形成候选，不能替代本课程固定 prompt、
counterfactual、completion、显存/延迟与人工 rubric 的同条件比较。完整来源账见 [RESEARCH.md](RESEARCH.md)；
超长上下文的现实任务、模态 token 账和 RAG/hybrid 取舍见 [LONG_CONTEXT_OR_RAG.md](LONG_CONTEXT_OR_RAG.md)。

## 7. 失败定位与最低成本实验

| 现象 | 优先怀疑 | 最低成本反证 | 不应立刻做 |
|---|---|---|---|
| 常识题对，换图答案不变 | 图像依赖/数据捷径 | image-drop/swap pair | 扩大 VQA SFT |
| OCR 漏小字 | 分辨率、vision encoder、token 压缩 | 提高 crop/visual budget 的 paired run | 先做语言 RL |
| box 内容对但格式错 | readout/SFT contract | normalized 与 strict accuracy 分栏 | 重训 ViT |
| 生图语义对但字糊 | VAE 重建与文字数据 curriculum | 原图 VAE round-trip + OCR | 只调 CFG |
| 单帧好看但视频闪烁 | 时空建模/数据/采样 | 固定 seed 对比 frame-wise 与 joint temporal | 用 CLIP 总分下结论 |
| H3 本地只能 768p | deployment/open boundary | 核验模块与 revision | 把托管 2K 写成本地能力 |

在 8×L20 上，最高 ROI 顺序通常是：CPU/L0 先验合同 → 2B/8B VLM 固定诊断 → connector/LoRA paired experiment →
tiny Image/Video DiT 训练 → 真实开放生成模型最小 inference。几十 B 的 H3/大 MoE 更适合做固定 revision 的推理与资源账；
没有训练数据、优化状态和长期预算时，不应把“8 卡放得下权重”误写成“具备 foundation training 条件”。

## 8. 费曼自检：能否从失败反推训练阶段

先独立回答，再展开参考答案：

1. 为什么 connector alignment 成功仍不能证明模型具有可靠 OCR？
2. 一个 VLM 的 vision tower 只占总参数 3%，为什么视觉输入仍可能主导延迟？
3. 为什么 VAE decoder 容易理解，不代表 VAE encoder 可以忽略？
4. 现代 VLM 为什么仍保留纯文本数据？RL 又为什么不能补回被视觉压缩丢掉的小字？
5. 为什么 Qwen3.8 比 Qwen3-VL 更新，却不应立即替换本课程的 2B L1？

<details>
<summary>参考答案</summary>

1. connector 只把已有视觉特征搬到 LLM 可消费的坐标系。若分辨率不足、patch merge 过强或视觉 encoder 没保存字符笔画，
   映射再准确也没有信息可读；必须用真实 OCR、crop/token-budget 对照和 image-swap 验证。
2. 参数占比描述权重存储，延迟还受 token 数和 attention 激活支配。高分辨率或长视频会产生大量 visual tokens，增加
   encoder 计算和 LLM prefill；小视觉塔也能制造很长的下游序列。
3. decoder 只从 latent 重建它收到的信息。encoder 决定哪些细节被压进有限 latent，以及 latent 是否容易被 DiT 学习；
   字符或运动在 encoder 处丢失后，decoder 只能猜。应先做 VAE round-trip，再评估 DiT。
4. 全参数多模态 CPT 会改变 LLM 分布，纯文本 replay 用于保持语言、代码和长文能力。RL 只能重新加权模型可产生的行为；
   输入表示没有小字信息时，奖励无法恢复不存在的证据，只可能鼓励更自信地猜。
5. 最新模型可能更大、许可证不同、训练 recipe 不完整或超出硬件预算。Qwen3-VL-2B 已有固定 revision、六例原始输出和
   可重复失败，适合教学因果对照；Qwen3.8 应先进入前沿追踪和独立小样本 gate，再决定是否升级实验锚。

</details>

## 9. 完成标志

学习者应能为任一多模态系统交付一张表：组件、参数口径、输入输出表示、训练阶段、可训练参数、数据类型、损失、开放资产、
最低成本反例和不可外推项。缺少其中任一列时，先补证据，不用“SOTA”“端到端”或“全模态”替代未知信息。
