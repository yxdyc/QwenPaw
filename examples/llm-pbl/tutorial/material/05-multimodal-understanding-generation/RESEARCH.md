# 05 多模态理解与生成：研究谱系与证据账本

> 对齐日：2026-08-31。本文把**论文机制、官方发布事实、源码/配置事实、课程推断和开放缺口**分开。
> 前沿模型、仓库和许可证会变化；进入 L1–L3 前必须固定 revision 并重新核验。

## 0. 不是模型动物园：三条技术谱系

```text
图文理解：pixel/patch → visual encoder → connector/resampler → LLM token fusion → multimodal pretrain/SFT
文生图：  pixels → VAE latent → latent patches → DiT + time/text condition → rectified-flow sampling → decode
文生视频：frames/audio → causal VAE/codec latent → (t,h,w) patches → long-sequence DiT → multi-flow decode
```

三条线共享 Transformer，却不能混为一谈：理解模型从媒体提取证据再生成文本；生成模型从噪声/条件预测连续 latent flow；
视频系统还要处理时间一致性、序列成本和视听同步。

## 1. 图文理解谱系：谁解决了哪一段接口

| 锚点 | 一手来源 | 教学定位 | 不应过度推出 |
|---|---|---|---|
| CLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | 大规模图文对比预训练把 image/text 映射到可对齐空间 | 对齐 embedding 本身不是生成式 VQA/grounding |
| Flamingo | [Flamingo](https://arxiv.org/abs/2204.14198) | resampler + gated cross-attention 让冻结语言模型读取交错视觉上下文 | cross-attention 不是所有 VLM 的唯一融合方式 |
| BLIP-2 | [BLIP-2](https://arxiv.org/abs/2301.12597) | Q-Former 在冻结视觉 encoder 与冻结 LLM 之间做轻量桥接 | 小 connector 不能自动保证细粒度空间能力 |
| LLaVA | [LLaVA](https://arxiv.org/abs/2304.08485) | 线性 projector + visual instruction tuning 展示简洁的视觉 token→LLM 路线 | instruction following 分数不等于 OCR/grounding 全覆盖 |
| Qwen3-VL | [官方仓库](https://github.com/QwenLM/Qwen3-VL) · [技术报告](https://arxiv.org/abs/2511.21631) | 本课程真实源码锚：动态视觉输入、空间/视频理解及当前 VLM 系统实验 | 具体 API、token budget、模型大小须在 L1/L2 固定 revision 后复核 |

### 研究问题，而不是单一总分

- **OCR**：字符是否可读、顺序/版面是否保持；
- **grounding/空间**：答案是否对应正确区域、坐标/相对关系是否稳定；
- **计数**：是否被重复纹理与遮挡欺骗；
- **视频理解**：是否利用时间顺序，而不是抽一帧猜测；
- **图像依赖**：image-drop/swap/shuffle 后输出是否按因果预期变化。

总 accuracy 会混合技能分布。L1 起固定分技能样本和 counterfactual pair；拒答也要单列，不能把无证据时的自信回答算“流畅”。

## 2. 文生图谱系：从 latent diffusion 到 DiT/rectified flow

| 锚点 | 一手来源 | 机制增量 | 课程用途 |
|---|---|---|---|
| Latent Diffusion | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 先用 autoencoder 压缩像素，再在 latent 空间扩散 | 建立 pixel/latent token 与重建损失账 |
| DiT | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | 把 latent patch 当 Transformer token；用 AdaLN 等方式接入 timestep/class 条件 | L0 的 latent patch、AdaLN 和序列成本锚 |
| SD3 / MMDiT | [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | rectified flow + multimodal diffusion Transformer，文本/图像表示共同参与 | L0 校准 flow 方向和 CFG；L1 才学习速度场 |
| Qwen-Image | [官方仓库](https://github.com/QwenLM/Qwen-Image) · [技术报告](https://arxiv.org/abs/2508.02324) | VLM 条件编码与图像生成/编辑系统 | L2 真实开放实验锚 |

### 可复现基线与前沿追踪要分开

- **实验基线**：课程默认采用官方仓库可取得、许可证明确的 **Qwen-Image-2512**；执行时仍需固定 commit/model revision。
- **前沿研究**：Qwen-Image 2.0 报告见 [arXiv:2605.10730](https://arxiv.org/abs/2605.10730)；2.0/3.0 的模型名、
  权重开放状态和接口必须在实验当日重核。未确认开放的条目只作调研，不进入“可复现完成”表。
- **评测边界**：CLIP similarity、OCR/VLM judge、aesthetic predictor 都是代理。文字、空间约束、组合遵循和总体偏好
  要用固定 prompt pair 与盲评 rubric；vendor leaderboard 作为厂商声明单列。

## 3. 文生视频谱系：时间轴改变了什么

| 组件 | 核心问题 | 验收方法 |
|---|---|---|
| 3D causal VAE | 时间压缩是否泄漏未来、长视频能否分块解码、重建怎样影响运动 | 记录时间/空间压缩率、重建误差、边界 artifact |
| 时空 patch / 3D position | token 同时属于哪一帧、哪一空间位置 | token ledger + position round-trip |
| 长序列 attention | 帧数增加使 full attention 关系数近似按 $T^2$ 增长 | 理论 pairs 与实测显存/延迟分栏 |
| 首尾帧/参考控制 | 条件是否在中间帧持续生效 | endpoint error + 中间轨迹/遮挡检查 |
| 训练/推理并行 | sequence parallel、offload、tiling 是否保持语义与确定性 | 固定 prompt/seed/revision 的 paired run |

源码对照采用 [HunyuanVideo 1.5 官方仓库](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
（报告 [arXiv:2511.18870](https://arxiv.org/abs/2511.18870)）与 [Wan2.2 官方仓库](https://github.com/Wan-Video/Wan2.2)。
课程不会仅凭仓库 README 的展示样例宣称质量领先；L2 用固定提示集、完成率、资源账和盲评分栏复核。

## 4. MiniMax H3：综合系统证据分层

### 4.1 官方已公开事实

主要锚点是 [MiniMax H3 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)、
[官方仓库](https://github.com/MiniMax-AI/MiniMax-H3) 与 [开放权重公告](https://www.minimax.io/news/minimax-h3-open-source)。

| 事实 | 来源类型 | 课程表述 |
|---|---|---|
| H3-Base 为 33B dense single-stream Omni Transformer | 官方模型卡/公告 | 称“33B 单流 omni Transformer”；不据此推断训练数据或成本 |
| encoder 使用完整 Qwen3-VL-32B，并取 hidden layer 50 | 官方模型卡 | 作为输入条件编码事实；具体 tensor shape 在固定 revision 后复核 |
| 文本、参考媒体、video latent 与 audio latent 统一打包 | 官方模型卡 | 用 packed rows 建教学合同，不猜私有 row schema |
| visual VAE 空间压缩 16×、时间压缩 4×、24 latent channels（f16t4d24），随后做 `1×2×2` patch；audio 每声道 40 Hz latent | 官方模型卡 | 用于 L0 token 公式；明确 `d24` 不是空间因子，真实 padding/layout 交给 L2 源码核验 |
| 模型约 13B 参数用于 modality-specific AdaLN 分支，使用 3D MM-RoPE | 官方模型卡 | 说明“共享主干 + 模态分支”，不把 toy RoPE 数值当官方坐标 |
| 输出默认短边 768、24 FPS、32 kHz stereo；2K 通过 Regenerate | 官方模型卡 | L3 本地只预注册 768p；2K 明确归 hosted boundary |

官方 [发布博客](https://www.minimax.io/blog/minimax-h3) 中的能力、效率和 benchmark 数字属于厂商报告；若课程后续引用，
必须保留测试条件并标“官方声明”，不能替代本地复现或盲评。

### 4.2 配置与源码实现事实

| 实现事实 | 一手锚点 | 课程落点 |
|---|---|---|
| packed sequence 使用 full self-attention，不依赖 cross-attention | [Diffusers H3 Transformer 文档](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/models/minimax_h3_transformer3d.md) | L0 明示 `full_self_attention` / `cross_attention=false` |
| 模态差异主要位于输入/输出投影、row tag、AdaLN/head | 同上 | 缺 tag 必须失败 |
| 视频/音频在同一次 Transformer 调用中使用各自 rectified-flow scheduler | [Diffusers scheduler 文档](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/schedulers/minimax_h3.md) | L0 同时返回两条 flow 合同 |
| 视频 scheduler shift=12，音频 shift=3 | [官方 video config](https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/scheduler/scheduler_config.json) · [audio config](https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/audio_scheduler/scheduler_config.json) | 错配必须被拒绝 |
| checkpoint 为 CFG-distilled，pipeline 使用单次条件前向 | [Diffusers pipeline 文档](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/minimax_h3.md) | 记录 forward calls=1，不伪造 unconditional branch |
| 结构/组件装配以公开配置为准 | [Transformer config](https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/transformer/config.json) · [model_index.json](https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/model_index.json) | L1 固定 revision 复算，不从课程常量倒推 |

### 4.3 开放边界：首版不等于完整系统开放

- 初始发布开放 H3-Base 的 FL2VA / Ref2VA 权重；课程统一称**开放权重**。
- H3-Context-IR 与 H3-Regenerate-2K 是托管模块，首版未随权重开放；L0 的 `TeachingContextIR` 固定
  `surrogate=true`，只表示课程自建中间表示。
- 官方说明中的稀疏注意力实现未随首版发布；没有实现/固定 revision 证据前，不写成可用本地优化。
- 截至本对齐日，课程没有取得可作为现成一手证据的完整技术报告；模型卡、博客、配置和源码各按自身证据层使用，
  不互相补齐未公开细节。

### 4.4 许可证是实验前置，不是脚注

[MiniMax H3 官方许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)单列核验。
课程不分发权重；不把 H3 输出用于训练其他 AI 模型。地域适用、商业展示/标识、安全义务与输出披露要求均以执行时
重新读取的官方许可证为准。本文是课程工程边界，不构成法律意见。

## 5. 从 L0 到真机：证据升级表

| 阶段 | 新增证据 | 仍然不能声称 |
|---|---|---|
| L0 | 机制合同、反例、确定性 token/metric 账 | 权重能力、真实质量、官方私有 schema |
| L1 | 真实 Qwen3-VL 小模型 / tiny trained DiT / H3 metadata | 大模型生产性能、真实视频 SOTA |
| L2 | 固定 revision 的开放系统生成、资源测量、盲评 | H3 托管 Context-IR/2K 已本地复现 |
| L3 | H3 FL2VA 单 checkpoint 三案例真机 manifest | 广泛提示分布、Ref2VA、完整系统或商业可用性 |

每一级都必须同时记录 completion/reliability。失败的 OOM、decode、依赖或许可证检查不能从质量均值的分母里删掉。

## 6. 决策门

- **Go L1**：四个 L0 的稳定 JSON、反例和教程输出全部通过 fresh-CWD 验收。
- **Go L2**：模型/代码/数据 revision 和许可证可固定；真实样本与合成样本、代理指标与人工 rubric 已分栏。
- **Go H3 真机**：只读硬件/磁盘/依赖检查通过，FL2VA 许可与下载范围获确认，媒体输出目录在仓库外。
- **Stop/Pivot**：任何组件只能由付费 hosted API 获得，就改为接口/边界分析；不得把 API 结果写成本地复现。
