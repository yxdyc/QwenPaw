# nano-image-dit

这个模块把文生图拆成四件事：latent token 为什么更便宜、DiT 怎样接收时间/文本条件、rectified flow 的方向是什么、
CFG 为什么既能增强条件也会过冲。

## 立即运行

```bash
python3 -B L0_rectified_flow_dit_oracle.py
```

仅依赖 Python 3.10+ 标准库，CPU、离线可跑。教程见 [tutorial_L0.md](tutorial_L0.md)。

## L0–L3 阶梯

| 级别 | 项目 | 验收重点 | 状态 |
|---|---|---|---|
| L0 | oracle rectified-flow DiT | latent patch、AdaLN、Euler、CFG、错误符号与 token 账 | 已完成 |
| L1 | 微型可训练 DiT | PyTorch CPU/GPU 训练；公开真实样本与合成条件分栏 | 规划中 |
| L2 | Qwen-Image-2512 真实生成 | 文字、空间、组合遵循、延迟、显存；盲评 + 代理指标 | 规划中 |
| L3 | 源码/系统对照 | rectified flow、MMDiT/VLM conditioning、offload/tiling 与 revision 固定 | 规划中 |

Qwen-Image-2512 是当前预定真实开放基线；Qwen-Image 2.0/3.0 只进入前沿追踪，开放状态未复核前不进入可复现阶梯。

## L0 量化合同

- latent reconstruction MAE：检查流方向和积分。
- condition attribute hit rate：检查条件目标，而不是只看重建误差。
- pixel/latent token ratio：显式记录 attention 之前的 token 成本。
- 失败件：反向 Euler 符号、CFG=3 过冲。

L0 使用显式 oracle velocity，没有学习任何生成分布；零重建误差是代数自检，不是生成质量证明。

## 文件

- [L0_rectified_flow_dit_oracle.py](L0_rectified_flow_dit_oracle.py)
- [tutorial_L0.md](tutorial_L0.md)
- [上级研究账本](../RESEARCH.md)
