# nano-multimodal-data-pipeline

> 用一个可运行的三账本实验，理解压缩文件、解码信号和模型 token 为什么是三种不同成本；再把抽帧、去重、切分、manifest 和 shard 串成一条可审计的数据管线。

## 立即运行

```bash
python3 -B tutorial/material/05-multimodal-understanding-generation/nano-multimodal-data-pipeline/L0_media_pipeline_ledger.py
```

验收条件：退出码为 `0`，最后一行以 `RESULT_JSON=` 开头，且 `checks` 全部为 `true`。脚本只依赖 Python 标准库，CPU、离线、确定性运行。

## 它解决什么问题

多模态训练常把下面三件事都叫“数据量”：

1. 对象存储中的 JPEG、MP4、FLAC 压缩字节；
2. 解码后的 RGB 像素、视频帧和 PCM 采样点；
3. 送入模型的 patch、声学帧或离散 token。

它们可能相差数百倍，而且分别压住网络、CPU/内存和 GPU。L0 用同一组 toy 资产同时记这三本账，并构造两个反例：均匀抽帧漏掉短事件；字节 hash 漏掉转码后的跨 split 重复。

完整原理与生产清单见 [`../MEDIA_DATA_PIPELINE.md`](../MEDIA_DATA_PIPELINE.md)。

## 学习阶梯

| 层级 | 目标 | 状态 |
|---|---|---|
| L0 | 三账本、稠密/均匀/自适应抽帧、group-before-split | 已实现 |
| L1 | 用真实图片/音频/视频做 probe、受限 decode 与时间戳抽样 | 规划中 |
| L2 | 比较小文件、tar shard、并行 decode、prefetch 与 backpressure | 规划中 |
| L3 | 接入 manifest/catalog、增量 backfill、质量门禁和 lineage | 规划中 |

## 与相邻模块的接口

```text
03 数据治理/编排
        │
        ▼
本模块：media bytes -> decoded signal -> model view
        │
        ├──> nano-vlm-understanding：视觉 encoder/projector/LLM
        └──> nano-long-context-routing：位置预算、长上下文与 RAG
```

- 通用清洗和算子治理：[`../../03-data-distributed-rsi/nano-data-juicer/`](../../03-data-distributed-rsi/nano-data-juicer/)
- 分布式数据编排：[`../../03-data-distributed-rsi/nano-data-orchestration/`](../../03-data-distributed-rsi/nano-data-orchestration/)
- 模型侧图文理解：[`../nano-vlm-understanding/`](../nano-vlm-understanding/)
- 长上下文路由：[`../nano-long-context-routing/`](../nano-long-context-routing/)

## 证据边界

L0 不读取真实媒体，也不声称 toy 的 `32×32` cell、8 FPS 事件窗口或 content group 是生产配置。事件窗口和内容组是 oracle，用来隔离机制；真实系统必须用真实 decoder、detector/fingerprint、硬件吞吐和下游任务指标重新验证。
