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

## L0 量化合同

- 首尾帧误差：约束是否被满足。
- trajectory roughness：二阶差分绝对值均值。
- temporal flicker：相邻变化量偏离平均速度的绝对偏差。
- full-attention pairs：$N^2$，明确它只是复杂度账，不是实测 FLOPs/延迟。

L0 的联合轨迹是确定性线性插值，不是训练后的运动先验；“更平滑”不等于更真实。

## 文件

- [L0_spatiotemporal_latent_dit.py](L0_spatiotemporal_latent_dit.py)
- [tutorial_L0.md](tutorial_L0.md)
- [上级研究账本](../RESEARCH.md)
