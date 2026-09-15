# nano-video-dit

图片 DiT 增加时间轴后，困难不只是“多几张图”：token 变成 $(t,h,w)$，attention 序列迅速增长，首尾帧约束必须
沿时间传播，逐帧看似正确也可能闪烁。本模块用最小 latent 轨迹把这些差别量化。

## 立即运行

```bash
python3 -B L0_spatiotemporal_latent_dit.py
```

仅依赖 Python 3.10+ 标准库，CPU、离线可跑。教程见 [tutorial_L0.md](tutorial_L0.md)。

## L0–L3 阶梯

| 级别 | 项目 | 验收重点 | 状态 |
|---|---|---|---|
| L0 | 时空 latent toy | 3D patch/position、首尾帧、逐帧 vs 联合、flicker 与 $N^2$ 成本 | 已完成 |
| L1 | moving-digit 微型 Video DiT | 运动条件、首尾帧、held-out temporal consistency | 规划中 |
| L2 | HunyuanVideo 1.5 / Wan2.2 | 3D VAE、offload、tiling、固定提示集、显存与延迟 | 规划中 |
| L3 | 并行与源码对照 | 长序列 attention、稀疏/序列并行、revision 固定与真机账本 | 规划中 |

这张表的 L2 列的是**开放权重/源码实验锚**，不是厂商产品最新榜。截至 2026-09-15，
[Wan3.0 Video](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/wan3-video-generation-guide) 已是官方最新 All-in-One 托管视频主线，
Wan2.7 为上一代 API；Wan3.0 的公开产品仓目前只有 README/许可证，Wan2.2 才有本地 checkpoint 与推理源码。
课程对前者只做 API 合同/计费/质量评测，
对后者才做本地结构与资源复现。Hunyuan 也按 HY-Video-1.5 服务、HunyuanVideo-1.5 开放基座与 OmniWeaving 控制扩展分栏。

## L0 量化合同

- 首尾帧误差：约束是否被满足。
- trajectory roughness：二阶差分绝对值均值。
- temporal flicker：相邻变化量偏离平均速度的绝对偏差。
- full-attention pairs：$N^2$，明确它只是复杂度账，不是实测 FLOPs/延迟。

L0 的联合轨迹是确定性线性插值，不是训练后的运动先验；“更平滑”不等于更真实。

## 文件

- [L0_spatiotemporal_latent_dit.py](L0_spatiotemporal_latent_dit.py)
- [tutorial_L0.md](tutorial_L0.md)
- [上级概念教程](../MODEL_ANATOMY_AND_TRAINING.md)：3D VAE、Video DiT 渐进训练、参数与开放资产口径。
- [超长上下文与 RAG](../LONG_CONTEXT_OR_RAG.md)：帧率、每帧 token、temporal merge 与长视频检索。
- [上级研究账本](../RESEARCH.md)
