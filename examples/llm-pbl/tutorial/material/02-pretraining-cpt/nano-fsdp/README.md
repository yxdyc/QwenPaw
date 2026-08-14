# nano-fsdp

> **抓的核心机制**：**ZeRO / FSDP 分片**——把参数、梯度、优化器状态切到多卡，理解「显存去哪了」。
> **对应真实系统**：[PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) / DeepSpeed ZeRO
> **轨道**：[02 预训练/CPT](../README.md) · **状态**：L0–L3 ✅（阶梯完成）

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | 玩具：手算给定模型/卡数下，参数+梯度+优化器状态的显存账本（Adam 为何是 16× 参数量） | ✅ [`L0_memory_ledger.py`](L0_memory_ledger.py) · [`tutorial_L0.md`](tutorial_L0.md) |
| **L1** | 用真实 PyTorch `FSDP` 包一个小模型，在 CPU 上模拟 2 卡训练，对比 `DDP` 的每 rank 模型状态内存 | ✅ [`L1_single_card_fsdp.py`](L1_single_card_fsdp.py) · [`tutorial_L1.md`](tutorial_L1.md) |
| **L2** | 实现 ZeRO-1/2/3 的分片差异，测量各阶段显存与通信量 | ✅ [`L2_zero_stages.py`](L2_zero_stages.py) · [`tutorial_L2.md`](tutorial_L2.md) |
| **L3** | 对照 FSDP 的 mixed-precision + activation checkpointing，分析端到端显存：激活账本 fp32 1,750,532→298,500 B（-82.9%）且与 ZeRO 分片正交叠加（FSDP 下逐字节同值）、bf16 Adam 参数冻结实证 master weights 硬需求、mixed 模型状态仍 16Ψ（实测）而 gather 通信恰减半（bf16）、FSDP2 fully_shard DTensor 契约真跑 + 非重入 checkpoint 重算早停机制（36/36 自检） | ✅ [`L3_mixed_precision_checkpointing.py`](L3_mixed_precision_checkpointing.py) · [`tutorial_L3.md`](tutorial_L3.md) |

## 环境依赖

- **L0**：零外部依赖（纯标准库），CPU 即跑。
- **L1**：需要 `torch`（CPU 即可）。本机使用 `python`（torch 2.4.1）实测通过；GPU 非必须。
- **L2**：需要 `torch`（本机 `python3`，torch 2.13.0 实测）。
  **真实多进程**（`mp.spawn` 2 个进程 + gloo backend，CPU 即跑，~3s，任意 CWD）；
  macOS 默认走 `lo0` loopback（Linux 删除脚本里 `GLOO_SOCKET_IFNAME` 一行即可）。
  GPU 上的吞吐结论标 `[TODO: verify on real system]`。
- **L3**：需要 `torch`（本机 `python3`，torch 2.13.0 实测）。
  **真实多进程**（`mp.spawn` 2 个进程 + gloo backend，CPU 即跑，~3s，任意 CWD）；
  混合精度走真实 bf16 kernel、activation checkpointing 走官方 `apply_activation_checkpointing`。
  macOS 上 FSDP2 `fully_shard` 需显式 CPU device mesh（默认 mesh 自动探测走 torch.mps 路径，
  本机 build 不可用，见 tutorial_L3 §8）。GPU 绝对数字标 `[TODO: verify on real system]`。

## 核心要讲清的点

- Adam 显存账本：参数(fp16) + 梯度(fp16) + master(fp32) + m + v ≈ 16× 参数量
- ZeRO 三阶段：分别切优化器状态 / 梯度 / 参数
- all-gather vs all-reduce：FSDP 前向需 all-gather 参数，反向 reduce-scatter 梯度
- 与 TP 的取舍：FSDP 不切算子，通用但通信多

## 费曼自检

- 能不能口算：7B 模型、8 卡、Adam、fp16 训练，ZeRO-3 下每卡显存大约多少？

## 权威实现与延伸

- 对标源码：PyTorch FSDP `docs.pytorch.org/docs/stable/fsdp.html`；DeepSpeed ZeRO `github.com/deepspeedai/DeepSpeed`
- 概念延伸：与 `nano-megatron` 对比（ZeRO vs TP）
