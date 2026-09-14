# 05 多模态理解与生成：研究谱系与证据账本

> 对齐日：2026-09-14。本文把**论文机制、官方发布事实、源码/配置事实、课程推断和开放缺口**分开。
> 前沿模型、仓库和许可证会变化；进入 L1–L3 前必须固定 revision 并重新核验。
> 本轮刷新模型身份、开放层级与训练证据；2026-09-04 的 Qwen3-VL L1 精确 revision 和真机结果保持原实验快照。

## 0. 不是模型动物园：三条技术谱系

```text
图文理解：pixel/patch → visual encoder → connector/resampler → LLM token fusion → multimodal pretrain/SFT
文生图：  pixels → VAE latent → latent patches → DiT + time/text condition → rectified-flow sampling → decode
文生视频：frames/audio → causal VAE/codec latent → (t,h,w) patches → long-sequence DiT → multi-flow decode
```

三条线共享 Transformer，却不能混为一谈：理解模型从媒体提取证据再生成文本；生成模型从噪声/条件预测连续 latent flow；
视频系统还要处理时间一致性、序列成本和视听同步。

### 0.1 “最新”“SOTA”“开放”“可复现”不是同义词

| 词 | 本账本的判定方法 | 常见误判 |
|---|---|---|
| 最新发布 | 官方公告/仓库给出日期和模型身份 | 新日期自动覆盖所有旧任务族 |
| SOTA | 固定任务、数据版本、输入/推理预算与比较协议 | 厂商平均榜单等于全场景第一 |
| 开放权重 | 官方提供 checkpoint，并单独核验许可证 | 权重可下等于数据和训练 recipe 全开 |
| 课程可复现锚 | revision、依赖、硬件与验收预算能固定 | 最大、最新的 checkpoint 必然最适合教学 |

截至本对齐日的模型定位如下。这里不做跨任务总排名：

| 系列 | 当前身份核验 | 本课程处理 |
|---|---|---|
| Qwen 理解 | [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next) 于 2026-08-26 开放权重，是更新的多模态 MoE 架构预览；Qwen3-VL 是更早的专用 VLM 家族 | 前者进入前沿追踪；后者 2B 继续作为已有真机证据的低成本锚 |
| Qwen Omni | [Qwen3.5-Omni 报告](https://arxiv.org/abs/2604.15804)给出更新的原生音视频训练路线；官方服务与公开权重边界不能混写 | 用报告讲训练；本地实验只用已确认可下载的 checkpoint |
| Qwen Image | [Qwen-Image-2.0](https://arxiv.org/abs/2605.10730)是更新模型/报告；[官方仓库](https://github.com/QwenLM/Qwen-Image)当前本地 T2I quick start 仍指向 2512 | 2.0 作前沿研究，2512 保持 L2 开放权重基线 |
| InternVL | InternVL3.5 仍是专用开放 VLM 家族；[InternVL-U](https://github.com/OpenGVLab/InternVL-U)是 2026-03 开放的 4B 理解—生图—编辑统一研究线 | 两条路线并列，不用发布日期替代任务定义 |
| Wan | [Wan2.2](https://github.com/Wan-Video/Wan2.2)仍是官方通用开放视频 foundation family；更晚的 [Animate-2](https://github.com/Wan-Video/Wan-Animate-2) / [Dancer](https://github.com/Wan-Video/Wan-Dancer) 属专项派生线 | T2V/I2V 对照保留 Wan2.2，专项模型另立任务 |
| Hunyuan | [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)仍是最新开放基础视频生成线；[OmniWeaving](https://github.com/Tencent-Hunyuan/OmniWeaving)等更新项目解决不同问题 | 基础视频 L2 保留 1.5，不从论文日期推出全面替代 |
| MiniMax | [MiniMax M3](https://www.minimax.io/blog/minimax-m3)偏理解/Agent；[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)是 2026-07 发布的开放视听生成系统 | H3 做生成 capstone，不与通用助手榜单直接排序 |

下一次 source refresh 应重新读取官方发布页、仓库和模型文件列表。搜索结果摘要、第三方排行榜和模型名猜测不能单独改变
课程实验锚。

## 1. 图文理解谱系：谁解决了哪一段接口

| 锚点 | 一手来源 | 教学定位 | 不应过度推出 |
|---|---|---|---|
| CLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | 大规模图文对比预训练把 image/text 映射到可对齐空间 | 对齐 embedding 本身不是生成式 VQA/grounding |
| Flamingo | [Flamingo](https://arxiv.org/abs/2204.14198) | resampler + gated cross-attention 让冻结语言模型读取交错视觉上下文 | cross-attention 不是所有 VLM 的唯一融合方式 |
| BLIP-2 | [BLIP-2](https://arxiv.org/abs/2301.12597) | Q-Former 在冻结视觉 encoder 与冻结 LLM 之间做轻量桥接 | 小 connector 不能自动保证细粒度空间能力 |
| LLaVA | [LLaVA](https://arxiv.org/abs/2304.08485) | 线性 projector + visual instruction tuning 展示简洁的视觉 token→LLM 路线 | instruction following 分数不等于 OCR/grounding 全覆盖 |
| Qwen3-VL | [官方仓库](https://github.com/QwenLM/Qwen3-VL) · [技术报告](https://arxiv.org/abs/2511.21631) | 本课程真实源码锚：动态视觉输入、空间/视频理解及当前 VLM 系统实验 | 具体 API、token budget、模型大小须在 L1/L2 固定 revision 后复核 |
| Qwen3.8-Flash-Next | [官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next) · [技术报告](https://arxiv.org/abs/2608.30320) | 当前前沿追踪：GDN + QSA、gated residual、n-gram embedding、Muon/AdamW 分工 | 更新且开放不代表能在 L20 预算内替换 2B 教学锚 |
| InternVL3.5 | [官方仓库](https://github.com/OpenGVLab/InternVL) · [技术报告](https://arxiv.org/abs/2508.18265) | 显式视觉参数账、全参数 NTP、SFT 与 offline→online Cascade RL | 厂商报告的 SOTA 结论仍需同预算复验 |

### 研究问题，而不是单一总分

- **OCR**：字符是否可读、顺序/版面是否保持；
- **grounding/空间**：答案是否对应正确区域、坐标/相对关系是否稳定；
- **计数**：是否被重复纹理与遮挡欺骗；
- **视频理解**：是否利用时间顺序，而不是抽一帧猜测；
- **图像依赖**：image-drop/swap/shuffle 后输出是否按因果预期变化。

总 accuracy 会混合技能分布。L1 起固定分技能样本和 counterfactual pair；拒答也要单列，不能把无证据时的自信回答算“流畅”。

### 现代训练证据：VQA 是数据接口，不是完整配方

| 模型 | 一手训练事实 | 课程解释 |
|---|---|---|
| LLaVA | 595K 图文对仅训练 projector；158K 指令数据更新 projector+LLM，vision encoder 保持冻结 | 低成本两阶段基线，重点是接通视觉与指令接口 |
| Qwen3-VL | merger-only 67B token → 全参数约 1T → 全参数长上下文约 1T → 256K 适配 100B；再接 SFT、蒸馏、RL | 主要能力来自全参数多模态 CPT、长度/数据 curriculum 与后训练 |
| InternVL3.5 | 全参数 NTP 预训练约 250B token；SFT 约 56M samples/130B token；offline MPO 后接 online GSPO | ViT–MLP–LLM 外形仍在，优化对象和证据规模已经变化 |
| Qwen3.5-Omni | 固定 LLM 训练 adapter/encoder → 约 4T token 全参数 omni pretrain → 262K 长上下文 → 专项/在线蒸馏和交互 RL | 原生 omni 从早期混合模态，不能简化成末尾增加音频/视频 VQA |

上述数字来自各模型公开报告，只描述报告中的训练口径，不表示课程能够获得相同数据或复现 foundation run。系统推导、
readout、参数分母和逐阶段可训练参数见 [MODEL_ANATOMY_AND_TRAINING.md](MODEL_ANATOMY_AND_TRAINING.md)。

### Qwen3-VL L1 的可复现锚（2026-09-04 局部刷新）

- 模型身份固定为 [Qwen/Qwen3-VL-2B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)，
  权重/配置 revision 固定为
  [`89644892e4d85e24eaac8bacfd4f463576704203`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/commit/89644892e4d85e24eaac8bacfd4f463576704203)。
- [官方仓库用法](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md)采用
  `AutoModelForImageTextToText`、`AutoProcessor` 与 `apply_chat_template`，并要求 Transformers 4.57.0 及以上；
  课程 L1 跟随这条公开接口，但仍在运行记录中写出实际安装版本。
- 真机无法访问 Hugging Face，因此从 [Qwen 官方 ModelScope 仓库](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)
  转移快照后离线加载。ModelScope `master` 是可变分支：课程不把它写成 pinned revision，而是验证权重 SHA256
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` 与 HF 固定 revision 相同，并逐文件
  交叉核验 10 个运行时关键文件；13 文件本地快照 manifest 为
  `b4f1f572206cd2e60255e7357166ee41bf2ffe3b8b52fa9da739370af895a99f`。
- 两个独立离线进程的 8/8 checks、答案、token 账和 digest 均一致。六例 normalized semantic accuracy 为 0.833，
  原始 OCR 回答保留为 `CODE`（期望 `CODE 7319`）。这形成固定小诊断上的实现证据；API、revision 与哈希只解决
  provenance，六个 synthetic diagnostics 仍不能外推成自然图像 benchmark。

## 2. 文生图谱系：从 latent diffusion 到 DiT/rectified flow

| 锚点 | 一手来源 | 机制增量 | 课程用途 |
|---|---|---|---|
| Latent Diffusion | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 先用 autoencoder 压缩像素，再在 latent 空间扩散 | 建立 pixel/latent token 与重建损失账 |
| DiT | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | 把 latent patch 当 Transformer token；用 AdaLN 等方式接入 timestep/class 条件 | L0 的 latent patch、AdaLN 和序列成本锚 |
| SD3 / MMDiT | [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | rectified flow + multimodal diffusion Transformer，文本/图像表示共同参与 | L0 校准 flow 方向和 CFG；L1 才学习速度场 |
| Qwen-Image | [官方仓库](https://github.com/QwenLM/Qwen-Image) · [技术报告](https://arxiv.org/abs/2508.02324) | VLM 条件编码与图像生成/编辑系统 | L2 真实开放实验锚 |

### 可复现基线与前沿追踪要分开

- **实验基线**：课程默认采用官方仓库可取得、许可证明确的 **Qwen-Image-2512**；执行时仍需固定 commit/model revision。
- **前沿研究**：Qwen-Image 2.0 已有[官方报告](https://arxiv.org/abs/2605.10730)和官方产品发布，但当前
  [Qwen-Image 开放仓库](https://github.com/QwenLM/Qwen-Image)没有给出 2.0 本地权重 quick start。2.0 只作调研，
  不进入“可复现完成”表；没有一手发布的后续型号不写入事实表。
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
- 截至本对齐日，官方仓库与模型卡已披露更多架构、输入合同和部分训练设计，但仍没有足够完整的数据组成、stage
  token/step、优化器与消融来复建 foundation recipe；这些缺口不得由 checkpoint 文件或宣传材料补齐。

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
