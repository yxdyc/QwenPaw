# L0｜从媒体文件到模型 token：先把三本账算清楚

> **核心问题**：为什么 72 MiB 的视频能让数据 worker 搬 10.4 GiB，又能在抽帧后只给模型几万个视觉位置？为什么“按 hash 去重”仍可能污染测试集？
>
> **前置知识**：会运行 Python；知道图片由像素组成即可。
>
> **不变量**：原始媒体不变；改变的只是读取、抽样、分组和模型视图策略。
>
> **运行**：`python3 -B tutorial/material/05-multimodal-understanding-generation/nano-multimodal-data-pipeline/L0_media_pipeline_ledger.py`
>
> **验收**：退出码为 `0`；最后一行 `RESULT_JSON=`；八个 `checks` 全为 `true`。
>
> **证据边界**：这是确定性算术 surrogate，不调用真实 codec 或模型；事件窗口和内容组是 oracle。

---

## 1. 先做一个反直觉实验

脚本构造一段 60 秒、1920×1080、30 FPS、10 Mbps 的视频。

压缩文件大小约为：

```text
10,000,000 bit/s × 60 s ÷ 8 ≈ 71.5 MiB
```

若全部解码成 RGB8：

```text
1920 × 1080 × 3 byte × 30 frame/s × 60 s ≈ 10.4 GiB
```

同一段内容出现了两个都正确、却相差 149 倍的“大小”。原因是：前者在数 codec 压缩后的字节，后者在数解压后的像素。

模型又有第三种大小。若 toy processor 每个 `32×32` 区域产生一个视觉位置，那么每帧约有：

```text
ceil(1920 / 32) × ceil(1080 / 32) = 60 × 34 = 2,040 positions
```

30 FPS 全读 60 秒就是 3,672,000 个视觉位置。GPU 不关心 MP4 文件是 72 MiB 还是 100 MiB，它主要关心最终送进网络多少位置、什么 dtype、怎样做 attention。

这就是本课的第一条不变量：

> 文件字节、解码信号和模型位置必须分别核算，任何一个都不能代表另外两个。

---

## 2. 阅读代码：四个最小机制

文件：[`L0_media_pipeline_ledger.py`](L0_media_pipeline_ledger.py)

### 2.1 信号展开

代码直接计算压缩视频和解码 RGB 的大小：

```python
compressed_video = bitrate * duration_s // 8
decoded_video = width * height * 3 * fps * duration_s
```

这里假设解码结果是每通道 1 byte 的 RGB。真实 decoder 也可能输出 YUV、多平面格式、不同位深，内存账本要按真实布局算。

### 2.2 模型位置

toy vision tokenizer 是：

```python
ceil(width / 32) * ceil(height / 32)
```

这不是某个生产模型的固定配置，只是让“像素预算怎样传导为模型预算”可计算。真实模型可能使用 patch merge、动态切块、缩放、ROI、多尺度 encoder 或 latent tokenizer。

### 2.3 三种视频策略

脚本比较：

| 策略 | 帧与尺寸 | 视觉位置 | 事件召回 |
|---|---:|---:|---:|
| 稠密 | 30 FPS，1080p | 3,672,000 | 1.000 |
| 均匀 | 1 FPS，640×352 | 13,200 | 0.333 |
| 自适应 | 平时 1 FPS，事件附近 8 FPS，640×352 | 18,480 | 1.000 |

三段事件里有两个不足 0.25 秒。每个整数秒取一帧会漏掉它们；事件窗口加密后，toy 召回恢复到 1.0，而视觉位置仍不到稠密策略的 1%。

这里必须读懂边界：脚本已经知道事件在哪里。真实系统还要付出“如何发现候选窗口”的代价，并测 detector 的漏召回。L0 证明的是**若能把预算路由到稀疏事件，收益可能很大**，不是证明某个 detector 已经可用。

### 2.4 group-before-split

脚本构造两个视觉内容相同、编码不同的文件：

```text
original.mp4     byte_hash=sha256-a  content_group=scene-7  split=train
reencoded.webm   byte_hash=sha256-b  content_group=scene-7  split=eval
```

精确 hash 看不到重复，因为转码改变了字节；内容分组能看到 `scene-7` 同时跨 train 和 eval。

正确顺序是：

```text
先形成 content group -> 再按 group 切 split
```

而不是：

```text
先随机切 split -> 每个 split 内部各自去重
```

后者会造成评测泄漏。

---

## 3. 亲手运行

从 `examples/llm-pbl` 目录执行：

```bash
python3 -B tutorial/material/05-multimodal-understanding-generation/nano-multimodal-data-pipeline/L0_media_pipeline_ledger.py
```

本仓库实测输出：

```text
THREE LEDGERS
video: file=71.526 MiB -> decoded=10678.711 MiB (149.30x)
dense_30fps_1080p: frames=1800, tokens=3672000, event_recall=1.000
uniform_1fps_640x352: frames=60, tokens=13200, event_recall=0.333
adaptive_1_to_8fps_640x352: frames=84, tokens=18480, event_recall=1.000
dedup: exact_hash_cross_split=False, semantic_group_cross_split=True
RESULT_JSON={"audio_model_view": {"codec_temporal_steps": 1500, "resampled_waveform_samples": 960000, "source_pcm_mib": 10.986}, "checks": {"adaptive_sampling_recovers_events": true, "adaptive_tokens_lt_1pct_dense": true, "byte_hash_misses_reencode": true, "codec_steps_are_not_waveform_samples": true, "decode_amplification_gt_100x": true, "global_plus_roi_lt_full_image_tokens": true, "group_before_split_blocks_leak": true, "uniform_sampling_misses_short_events": true}, "dedup": {"content_group_detects_cross_split_copy": true, "exact_hash_detects_cross_split_copy": false, "grouped_split_has_leak": false}, "digest": "a06d00908ef1d7d7", "evidence_boundary": "Deterministic arithmetic surrogate: event windows and content groups are oracle labels, not learned production detectors.", "image_model_view": {"full_tokens": 12288, "global_plus_roi_tokens": 1024}, "three_ledgers": {"image_decoded_rgb_mib": 36.0, "image_file_mib": 3.0, "video_decode_amplification": 149.3, "video_decoded_rgb_mib": 10678.711, "video_file_mib": 71.526}, "video_policies": [{"event_recall": 1.0, "frame_shape": [1080, 1920], "frames": 1800, "name": "dense_30fps_1080p", "selected_rgb_mib": 10678.711, "visual_tokens": 3672000}, {"event_recall": 0.333, "frame_shape": [352, 640], "frames": 60, "name": "uniform_1fps_640x352", "selected_rgb_mib": 38.672, "visual_tokens": 13200}, {"event_recall": 1.0, "frame_shape": [352, 640], "frames": 84, "name": "adaptive_1_to_8fps_640x352", "selected_rgb_mib": 54.141, "visual_tokens": 18480}]}
```

`RESULT_JSON` 不是装饰。它让 CI 或后续实验能机器读取 checks、metrics、证据边界和 digest，而不是靠肉眼猜“好像跑通了”。

---

## 4. 图片和音频为什么也需要三本账

### 4.1 图片：全图高清还是全局图 + ROI

toy 图片为 `4096×3072`：

```text
压缩文件                 3 MiB
解码 RGB                 36 MiB
全图 32×32 cell          12,288 positions
1024×768 全局 + 512² ROI 1,024 positions
```

“全局 + ROI”少 12 倍位置，但只有 ROI 覆盖真正证据时才不损伤任务。OCR 小字、密集计数或全图细粒度检索可能需要多个 ROI，甚至需要保留全图高分辨率。策略的验收指标必须是下游能力，而不是 token 越少越好。

### 4.2 音频：波形点不是 codec token

60 秒音频在 16 kHz 下有 960,000 个单声道采样点。toy audio codec 若每秒输出 25 个时间步，则只有 1,500 个时间步。

但不要把“1,500”直接理解为最终 token 数：真实 codec 可能每个时间步有多个 codebook，声学 encoder 也可能输出连续 embedding。正确问法是：

1. 原始 PCM 有多少采样点/字节？
2. 特征或 codec 的时间步率是多少？
3. 每个时间步有多少通道、codebook 或向量维度？
4. 最终哪些位置进入 Transformer？

---

## 5. 从 toy 升级到真实数据管线

### L1：真实 probe 与受限 decode

选择少量可公开媒体，记录：

- 容器、codec、宽高、时长、帧率、PTS/time base；
- 压缩字节与解码字节；
- 完整 decode、随机 seek、均匀抽帧的墙钟时间；
- 损坏文件、超限文件和可变帧率样例。

验收不是“FFmpeg 能打开”，而是同一时间戳策略可重现、错误有 reason code、worker 有像素/时长/内存上限。

### L2：分片和吞吐

固定相同资产与读取顺序，比较：

- 每个媒体一个对象；
- tar/shard 顺序读取；
- 不同 worker 数、prefetch 深度和 decode 后端；
- 按样本数 batch 与按视觉位置/音频时长 batch。

同时测 storage MB/s、decoded MPix/s、model positions/s、GPU data-wait 和 P99，不能只报 samples/s。

### L3：治理闭环

把每个派生样本接入 catalog/manifest：

```text
source_hash
  -> probe_version
  -> transform_version + parameters
  -> content_group + split
  -> shard_id + offset
  -> training run + metric
```

当阈值、decoder 或抽帧策略改变时，只 backfill 受影响的资产；新版本先 shadow 统计，再 canary 训练，最后由固定评测门禁决定是否提升。

完整生产原则见 [`../MEDIA_DATA_PIPELINE.md`](../MEDIA_DATA_PIPELINE.md)。

---

## 6. 思考题

### 6.1 为什么不能用 MP4 文件大小估算 GPU 训练成本？

**参考答案：** codec 压缩率由画面可压缩性、码率和编码设置决定；GPU 看到的是抽样、resize、tiling 后的模型位置。同样大的 MP4 可以有不同分辨率、时长和帧率；同一 MP4 也能产生完全不同的模型视图。应分别记文件字节、解码信号和模型位置。

### 6.2 “先全部解码缓存，训练就不会等 decoder”为什么通常不是免费优化？

**参考答案：** 它把压缩存储放大为像素/PCM 存储，可能是几十到数百倍；还带来缓存生成、版本失效、网络读取和 lineage 成本。高复用且稳定的派生物可以缓存，任务相关、随机或高放大的视图通常应按需生成，并用有界 prefetch 隐藏延迟。

### 6.3 为什么均匀 1 FPS 能看见长事件，却漏掉两个短事件？提高到固定 8 FPS 就够了吗？

**参考答案：** 采样点必须落入事件持续区间才会命中。短事件位于整数秒之间，因此 1 FPS 漏掉。固定 8 FPS 会提高召回，但也把整段 60 秒成本放大 8 倍；自适应方案只在候选窗口加密。它仍依赖候选 detector 的召回，真实验收需把 detector 成本和漏检一起算。

### 6.4 为什么 JPEG/视频转码会绕过 SHA-256 去重？应该怎样切分？

**参考答案：** SHA-256 比较文件字节，转码会改变字节，即使人看到的内容近似不变。应结合精确 hash、感知指纹和近语义分组形成 `content_group_id`，再按 group 切 train/val/test。近语义规则会误合并，必须保存阈值、pair 和人工抽检证据。

### 6.5 超分后 OCR 更清楚，是否可以把超分图当作更真实的训练标签？

**参考答案：** 不可以直接这样推断。超分模型可能生成符合先验但原图不存在的笔画。它可以作为增强视图，但原始资产必须保留；对 OCR、取证等任务要专门测伪细节和标签一致性，不能让增强结果自证正确。

### 6.6 如果 GPU 利用率低，为什么先增加 DataLoader worker 可能让系统更差？

**参考答案：** 真瓶颈可能是单路 codec、内存带宽、下游 tokenize 或不均匀长样本。盲目加 worker 会增加争用和预取队列，把主机内存撑满。先测各 stage 速率、队列等待、MPix/s、P99 和 GPU data-wait，再对最慢 stage 扩容，并设置 backpressure。

---

## 7. 费曼复述：不用术语讲给同学听

请先遮住下面答案，用一分钟解释：

> 为什么一段 72 MiB 的视频既可能在 CPU 侧变成 10.4 GiB，又可能在模型侧缩成 18,480 个位置？这个“缩小”会付出什么风险？

**参考讲法：**

视频文件像一只抽过真空的衣物袋。打开袋子后，每一帧的每个像素都要展开，所以内存量暴涨；这叫解码。训练时我们又不一定把每一帧、每一个像素都给模型，而是缩图、抽帧并切成较大的小格，所以只剩有限的视觉位置。这样省计算，但若小字或短动作刚好落在被丢掉的区域和时刻，模型永远看不到它。好的管线不是一味压缩，而是先知道任务需要什么证据，再把高分辨率和高帧率预算花在那些地方，并保存原始文件以便审计和重做。
