# 大规模多模态数据管线：从压缩字节到可学习证据

这份材料补齐 05 轨此前缺少的数据平面：图片、音频、视频文件怎样经过探测、解码、抽样、去重和打包，最终变成模型真正消费的张量或 token。

配套可运行实验：[`nano-multimodal-data-pipeline/tutorial_L0.md`](nano-multimodal-data-pipeline/tutorial_L0.md)。通用的数据清洗、编排和治理仍由 [03 Data × Distributed × RSI](../03-data-distributed-rsi/README.md) 负责；这里专讲媒体特有的表示转换和成本。

---

## 0. 先抓住本质：管线不是“搬文件”，而是“保存证据”

假设对象存储里有一段 60 秒视频。训练失败时，常见的直觉是“磁盘慢”或“GPU 不够”。但在模型看到它之前，至少出现了三种语义完全不同的对象：

```text
压缩文件               解码后的物理信号                  模型视图
MP4/JPEG/FLAC  ──解码──> RGB 帧 / PCM 波形  ──处理器──> patch / embedding / token
   几十 MiB                 几 GiB                        几千到几百万位置
```

这三种对象不能混为一谈：

- **文件字节**回答“存储和网络要搬多少”。
- **解码信号**回答“CPU、解码器、内存带宽要生产和搬多少像素/采样点”。
- **模型位置**回答“GPU 要对多少位置做投影、注意力和反向传播”。

因此，多模态管线的第一原则是：

> 不按“文件个数”管理成本，要同时记文件、信号、模型三本账；不按“画质越高越好”保留信号，要按任务所需证据分配预算。

---

## 1. 四个经常混淆的词

### 1.1 容器、编码、解码、模型 tokenizer

| 名词 | 费曼式理解 | 例子 | 它不负责什么 |
|---|---|---|---|
| 容器 container | 一个带目录和时间轴的盒子 | MP4、MKV、WebM | 不规定模型 token |
| 编解码器 codec | 把稠密信号压缩成较少字节，再近似还原 | JPEG、H.264、AV1、AAC、Opus | 不决定训练样本如何抽帧 |
| demux / decode | 从盒子取出某路压缩包，再还原像素或波形 | 视频帧、PCM 音频 | 不等于视觉/音频 tokenizer |
| 模型 processor / tokenizer | 把信号变成模型约定的尺寸、patch、特征或离散码 | resize、mel、ViT patch、audio codec token | 不负责通用文件兼容性 |

一个 `.mp4` 文件可以同时装视频流、音频流、字幕流和时间戳。`demux` 是把这些流分开；视频 decoder 再把压缩包还原为帧。模型的视觉 processor 随后才决定抽哪些帧、缩到多大、切成多少 patch。

所以“我们已经把视频 token 化”可能指两件完全不同的事：

1. 用 H.264/AV1 把视频压成文件字节；
2. 用模型 processor 把解码帧变成视觉 token。

讨论成本时必须说清楚是哪一层。

### 1.2 为什么压缩文件很小，训练仍可能很慢

以 RGB8 为例：

```text
单张图片解码字节 ≈ H × W × 3
音频 PCM 字节     ≈ 时长 × 采样率 × 声道数 × 每采样字节
视频 RGB 字节     ≈ 时长 × FPS × H × W × 3
```

一段 60 秒、1080p、30 FPS 的视频一共有 1,800 帧。即使文件只有约 72 MiB，全部解码成 RGB8 也约为 10.4 GiB，放大约 149 倍。这里没有神秘的“模型开销”，仅仅是压缩被展开了。

这解释了两个生产现象：

- 对象存储带宽看起来够，GPU 仍等数据：瓶颈可能在解码、resize 或主机内存搬运。
- 把所有媒体预解码并缓存通常不可行：它把便宜的压缩存储换成了巨大的派生数据集。

---

## 2. 三本账：任何设计评审都应同时给出的数字

### 2.1 文件账本：`bytes/file` 与 `bytes/s`

记录压缩大小、来源、格式、码率、下载失败率、对象读取延迟。这决定对象存储容量、网络和冷启动代价。

但文件账本不能预测训练成本。相同 10 MiB：可能是一张超高分辨率 JPEG，也可能是数分钟低码率视频。

### 2.2 信号账本：`pixels/s`、`audio-seconds/s`、`decoded bytes/s`

记录解码后的宽高、帧率、时长、采样率、声道和位深。它决定 decoder、CPU、内存、PCIe 和预处理压力。

最有用的中间指标通常不是“每秒多少文件”，而是：

- image megapixels/s；
- decoded video frames/s 或 megapixels/s；
- decoded audio hours/s；
- 无效/损坏/超限媒体比例。

### 2.3 模型账本：`model positions/sample` 与 `tokens/batch`

模型成本取决于 processor 的最终视图。例如用一个简化的 `32×32` 视觉 cell：

```text
单帧视觉位置 ≈ ceil(H / 32) × ceil(W / 32)
视频视觉位置 ≈ 采样帧数 × 单帧视觉位置
```

空间边长各放大 2 倍，会产生约 4 倍像素和视觉位置；若帧率也放大 2 倍，视频位置约变成 8 倍。后续注意力若直接作用于这些位置，代价还可能增长得更快。

音频也一样：48 kHz 波形每秒有 48,000 个采样点，但模型可能先转成几十到几百个声学帧或 codec 时间步。**采样点、声学帧、离散音频码不是同一种 token。**

---

## 3. 一条可审计的端到端管线

```text
不可变原始资产
  │
  ├─ 1. ingest + hash：保留来源、许可、抓取时间和原始字节
  ├─ 2. probe：只读头部/元数据，识别格式、流、时长、尺寸、时间基
  ├─ 3. validate + quarantine：限制资源，隔离损坏、炸弹和不支持输入
  ├─ 4. decode/canonicalize：按需解码，统一颜色、方向、采样率和时间戳语义
  ├─ 5. quality/safety：模糊、静音、冻结帧、黑边、语言、许可、安全策略
  ├─ 6. dedup/group：精确、感知、近语义重复形成 content group
  ├─ 7. segment/align：镜头、语音段、字幕、图文或音视频时间对齐
  ├─ 8. split：以 content group 为单位切 train/val/test
  ├─ 9. manifest/shard：小元数据索引 + 大块顺序读取的数据分片
  └─ 10. model view：按任务在线抽帧、裁剪、增强、tokenize、动态组 batch
```

关键点是每一步都产生**可追溯派生物**，而不是就地覆盖前一步。

### 3.1 最小 manifest 合同

| 字段 | 为什么需要 |
|---|---|
| `asset_id`, `source_uri`, `source_hash` | 找回原始证据并判定字节是否变化 |
| `license`, `provenance`, `ingest_time` | 合规、撤回和来源审计 |
| `media_type`, `codec`, `duration`, `shape`, `time_base` | 预估解码和任务成本 |
| `transform_name`, `transform_version`, `parent_id` | 重建派生样本和 lineage |
| `quality_flags`, `failure_reason` | 可解释过滤，不把失败静默吞掉 |
| `content_group_id` | 去重后再切分，阻断近重复泄漏 |
| `segment_start/end`, `alignment_ids` | 对齐到原始时间轴，而非复制整份媒体 |
| `split`, `shard_id`, `offset` | 可重复抽样和高效读取 |

原则是：manifest 保存**事实和决策依据**；模型训练 recipe 再决定如何消费。不要把某次实验的固定 resize 结果误当作唯一真相。

### 3.2 资产不等于训练样本

一段视频可以派生多个镜头、音频段、关键帧、字幕对齐、问答和不同分辨率视图；同一图片也能派生 OCR、caption、grounding
和编辑前后对。应明确两层 ID：

```text
asset_id：原始证据与许可的单位
sample_id：某个训练目标消费的语义单位，带 parent asset/segment/annotation/version
```

这解决三个常见错误：

- **切分泄漏**：同一 asset 的不同 crop/clip 被当作独立样本分到 train 与 test；
- **权重失真**：长视频切出 1,000 个 clip，在按 sample 均匀采样时意外获得 1,000 倍权重；
- **伪标签漂移**：caption/OCR/VLM annotation 换模型后覆盖旧结果，无法知道能力变化来自数据还是模型。

生成式 annotation 应像代码一样版本化，至少记录 teacher、prompt/template、sampling 参数、输入视图、输出、过滤理由和人工抽检。
质量不能只测语言流畅度，还要测 groundedness、时空对齐、覆盖率、矛盾率和不同来源/模态桶的偏差。数据混合时同时报告
asset 数、sample 数、原始时长/像素和最终 model positions；否则“50% 视频数据”没有可执行含义。

---

## 4. 三种模态最容易踩的坑

### 4.1 图片：方向、颜色和“解压炸弹”

- 相机方向可能只写在 EXIF 中；忽略它会让标注框、OCR 坐标和像素错位。
- RGB、CMYK、灰度、带 alpha 的图像不能靠文件扩展名猜测。
- 极小压缩文件可能声明极大画布；在受限 worker 中先 probe 并设像素上限。
- JPEG 重编码后字节 hash 会变，但内容可能几乎不变，所以仅用 SHA-256 不够做数据去重。

### 4.2 音频：采样率不是“质量分数”

采样率决定可表达的最高频率范围，位深影响量化噪声，声道决定空间信息。任务需要什么，才保留什么：

- 语音识别可能不需要保留超声段或多声道空间感；
- 音乐理解、音效生成、说话人和空间任务可能需要更高带宽或声道信息；
- 重采样之前应先记录原始采样率，并使用确定、版本化的 resampler；
- VAD 切段能减少静音，却可能切掉呼吸、环境声和跨段语义，必须按任务验证。

### 4.3 视频：FPS 不是时间，帧序号也不是时间戳

- 可变帧率视频里，`frame_index / nominal_fps` 不一定是真实时刻；应以 PTS/time base 对齐。
- 随机 seek 常从关键帧附近开始解码，不是 O(1) 读取任意帧。
- 先转成恒定帧率可能简化训练，却会复制或丢弃帧；必须记录变换。
- 音画对齐应使用同一时间轴，不能分别“数第几个样本”。
- 只看均匀抽帧容易漏掉短事件；全帧训练又会把大量预算花在静止和重复画面上。

---

## 5. 分辨率、采样率和帧率：追求的是证据带宽，不是数字最大

人类对“够清晰”的感受依赖观看距离、屏幕、内容、压缩和任务，没有一个普适阈值。训练更应问：**目标事件的最小空间尺度、最快时间尺度、最高相关声学频率是什么？**

### 5.1 粗到细比全局稠密更划算

一个常用结构是：

1. 低分辨率全局视图发现候选区域；
2. 对文字、物体或事件窗口取高分辨率 ROI；
3. 对短时事件局部提高 FPS；
4. 在总视觉 token 预算内动态装 batch。

配套 L0 中，一张 `4096×3072` 图片的全量 toy 视图需要 12,288 个位置，而 `1024×768` 全局图加一个 `512×512` ROI 只需 1,024 个。这个数字不是质量保证；它展示的是“先定位，再花预算”的结构性收益。

### 5.2 超分和插帧的边界

超分辨率、去噪、插帧可以让输入更适合人看或让下游模型更稳定，但不能凭空恢复已经丢失的证据。它们可能生成貌似合理的文字、纹理或中间动作。

因此：

- 展示、美学和生成任务可以把它视为一种增强；
- OCR、取证、医学、计数和精细时序任务必须保留原始证据，并单独评估伪细节；
- 不要用增强后的内容反过来做“真实标签”；
- 若训练时使用，manifest 必须记录模型、版本、参数和 parent asset。

### 5.3 可落地的起始配置，不是统一 SOTA 答案

下面是做预算实验时可用的**起始点**，不是对任一当前模型的配置声明。最终值要由 processor 合同、硬件和下游消融决定：

| 场景 | 低成本起始视图 | 何时必须加预算 |
|---|---|---|
| 图片质量扫描/粗分类 | 长边约 256–512 的 thumbnail | 小字、细粒度缺陷、密集目标或版面结构 |
| VLM 全局理解 | 约 0.25–1 MP 全局图，再按 processor 对齐 | OCR、GUI、图表、遥感等需要原图 tile/ROI |
| 图片生成训练 | 先低分辨率学构图，再用 512/1024 等 bucket 学细节 | 输出规格、文字和局部纹理确实受益且算力允许 |
| 语音语义任务 | 16 kHz mono 常可作为对照起点 | 韵律、音乐、环境声、空间声或高频细节 |
| 通用音频/音乐 | 保留 44.1/48 kHz 原始资产，模型视图再按任务重采样 | 生成、混音、声学事件或立体空间要求更高 |
| 长视频索引/粗理解 | 场景切分 + 约 0.5–2 FPS 全局抽样 | 短动作、手势、体育、UI 操作与快速镜头变化 |
| 事件级视频理解 | 粗采样发现候选，窗口内约 4–16 FPS 或原始 FPS | detector 漏召回、细粒度动作或精确时序 |
| 人类交付视频 | 24/25/30 FPS 是常见基线，快运动常用 50/60 FPS | 高速运动、交互低延迟或慢动作需求 |

这些数字最重要的用法是形成**配对消融**：固定样本、模型、训练 token 和评测，只改变一个分辨率/FPS/采样率档位，
比较质量增益、位置数、吞吐、显存和 data-wait。不要拿“输入更高清”替代能力证据，也不要把不同总 token 预算的结果
误写成纯分辨率因果效应。

当前具体模型的 native FPS、动态像素范围、VAE 压缩率和音频 latent rate 会随版本变化，应在运行前从固定 revision 的
processor/config 重算；本轨把这些版本化事实放在 [RESEARCH.md](RESEARCH.md)，而不把易漂移数字硬编码成课程定理。

### 5.4 高画质、高细节和高美感是三个变量

- **高画质**偏技术完整性：清晰度、噪声、压缩伪影、曝光、色彩和音画同步。
- **高细节**偏可恢复信息：小字、纹理、短事件、弱声学信号是否仍存在。
- **高美感**偏构图、光影、节奏、风格一致性与人群偏好；单纯增加像素或采样率不会自动提高它。

数据选择时应把这些 score 分栏，并保留分布与置信度。若用单一 aesthetic score 过滤，容易牺牲长尾、真实噪声、文化多样性
和对任务重要但“不漂亮”的证据；若只追高分辨率，又会重复为相同语义支付更多解码和模型位置成本。

---

## 6. 去重必须早于 split

仅按字节 hash 去重，只能发现完全相同的文件。裁剪、转码、加水印、改分辨率都会绕过它。

生产上通常分层做：

1. **精确去重**：字节 hash，便宜且无歧义；
2. **感知去重**：图像/关键帧/音频指纹，发现轻微变换；
3. **近语义分组**：embedding 或多模态证据，召回更广但有误合并风险；
4. **group-aware split**：同一内容组、同一视频切片、同一对话或同一来源家族只进一个 split。

顺序不能反：先随机 split，再在每个 split 内去重，会保留 train 与 test 之间的近重复，得到虚高评测。

近语义去重也不能无脑越强越好。它可能删除“外观相似但标签不同”的困难样本。应保存 pair、距离、规则版本和抽样人工审计结果，让阈值可回滚。

---

## 7. 离线做什么，在线做什么

判断标准不是“能不能离线”，而是**复用次数、稳定性、随机性和存储放大**。

| 更适合离线 | 更适合在线 |
|---|---|
| 来源校验、probe、hash、许可和安全元数据 | 随机 crop、mask、颜色增强 |
| 昂贵且跨实验复用的镜头/VAD/去重索引 | 随 epoch 变化的抽帧和样本混合 |
| 确定性的时间对齐和高价值特征 | 与当前模型 token 预算绑定的 resize/tiling |
| 失败隔离、manifest、shard 索引 | 少量按需 decode 与设备侧变换 |

稳妥的分层是：

```text
L0 raw       不可变压缩资产
L1 manifest  小而可查询的事实、lineage、质量和分组
L2 derived   可重建且高复用的索引、片段或特征缓存
L3 model view 由当前 recipe 动态产生的 tensor/token
```

缓存键至少应包含 `source_hash + transform_version + parameters`。否则代码升级后，旧缓存会悄悄混入新实验。

---

## 8. 大规模吞吐：最快 stage 决定不了速度，最慢 stage 才能

流水线稳态吞吐近似：

```text
throughput ≤ min(download, decode, transform, tokenize, host_to_device, model)
```

只提高某一段的并发，可能只是把等待搬到下一条队列并撑爆内存。应同时观察：

- 各 stage 的输入/输出速率、P50/P95/P99 延迟；
- 队列深度、backpressure 时间和 worker 利用率；
- compressed MB/s、decoded MPix/s、audio/video seconds/s；
- model positions/s、padding 比例和 GPU data-wait；
- 按 codec、来源、时长桶统计的失败率与隔离原因。

### 8.1 分片与 batch

- 数十亿小文件会把延迟浪费在 open/list 请求；用可索引的大 shard 做顺序读取。
- shard 不应成为唯一真相：manifest 才是逻辑目录，shard 是可重建的物理布局。
- 先在 shard 间 shuffle，再用有限 buffer 在 shard 内 shuffle，避免把全量索引装入内存。
- batch 应按最终模型 token 或像素/时长预算组装，而不只是固定样本数。
- 对超长、超大、稀有 codec 单独分桶，避免一个异常样本拖住整批。

### 8.2 可靠性

- 每个 stage 幂等：相同输入和版本重复运行得到相同逻辑结果。
- 错误进入 quarantine 并带 reason code；不能让 `except: pass` 静默改变数据分布。
- 设资源上限：最大像素、时长、流数、解码时间、内存和递归层数。
- 许可、同意、PII/人脸/儿童安全与来源策略绑定到 asset；撤回时沿 lineage 删除所有 segment、annotation、shard 和训练清单引用。
- 发布数据集时冻结 manifest、规则版本、统计摘要和抽样 QA；能从样本回溯到原始资产。

---

## 9. 症状到根因：最低成本的排查顺序

| 症状 | 首先怀疑 | 最低成本检查 |
|---|---|---|
| GPU 周期性空转 | decode 尾延迟或 shard 倾斜 | 分 stage 队列等待和时长桶 |
| 存储吞吐不高但 CPU 满 | 解码/resize，而非 I/O | MPix/s、codec 分桶、perf profile |
| 主机内存逐步上涨 | 预取无 backpressure 或缓存 decoded tensor | 队列上限与对象生命周期 |
| 每 batch 用时波动大 | 按样本数而非 token/时长组 batch | batch 总位置数和最大样本 |
| 离线评测异常高 | 近重复跨 split | content group 的 split 交叉率 |
| 同一 recipe 无法复现 | transform/cache 未版本化 | source hash、参数和 lineage |
| OCR/短事件能力差 | 过早降采样或均匀抽帧漏证据 | 原始资产上的 ROI/时间窗审计 |
| 数据越“高清”训练越慢但无收益 | 模型位置增加，信息增益不足 | 固定算力下做分辨率/FPS 消融 |

---

## 10. 发布前的 principles / best-practice 清单

1. **三本账**：文件字节、解码信号、模型位置是否都被测量？
2. **证据优先**：分辨率/FPS/采样率是否由任务所需证据决定？
3. **原始不可变**：能否从任一训练样本追溯到 source hash？
4. **廉价失败优先**：是否先 probe、限额、过滤，再做昂贵 decode？
5. **去重后切分**：content group 是否跨 train/val/test 为零？
6. **离线/在线有边界**：稳定高复用工作离线，随机且任务相关工作在线？
7. **物理布局可替换**：manifest 与 shard 是否解耦？
8. **按真实成本 batch**：是否按 token、像素或时长，而非文件个数？
9. **失败可见**：quarantine、reason code、分桶失败率是否齐全？
10. **声明证据边界**：toy、oracle、近似 detector 与真实模型结果是否分开？

如果只能记一句：

> 好的多模态数据管线不是把所有信号变得最稠密，而是在可追溯、可复现的前提下，把有限计算准确花到任务需要的证据上。

---

## 11. 与课程其他模块的边界

- 本文与 [`nano-multimodal-data-pipeline`](nano-multimodal-data-pipeline/README.md)：媒体字节、解码、抽样、去重、manifest、shard 与模型视图。
- [`nano-vlm-understanding`](nano-vlm-understanding/README.md)：模型已拿到图片后，视觉 encoder、projector、LLM 怎样连接。
- [`nano-long-context-routing`](nano-long-context-routing/README.md)：模型位置过多后，长上下文、RAG 和路由如何取舍。
- [03 的 `nano-data-juicer`](../03-data-distributed-rsi/nano-data-juicer/README.md)：通用数据算子和质量治理。
- [03 的 `nano-data-orchestration`](../03-data-distributed-rsi/nano-data-orchestration/README.md)：分布式执行、backfill、状态和失败恢复。

外部实现参考：[FFmpeg formats](https://ffmpeg.org/ffmpeg-formats.html)、[FFmpeg codecs](https://ffmpeg.org/ffmpeg-codecs.html)、[Data-Juicer](https://github.com/modelscope/data-juicer)、[WebDataset](https://github.com/webdataset/webdataset)、[PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html)。这些工具提供机制；具体阈值仍需由任务、数据和硬件实验决定。
