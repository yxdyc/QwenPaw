# nano-megatron

> **抓的核心机制**：**张量并行 (TP) + 流水线并行 (PP) + 序列并行 (SP)** 的最小可跑实现，理解「切分如何通信」。
> **对应真实系统**：[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（论文 arXiv:1909.08053）
> **轨道**：[02 预训练/CPT](../README.md) · **状态**：L0–L3 ✅（阶梯完成）

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | 玩具：纯 Python 把 MLP 做 Megatron 式列切/行切，用矩阵逐元素和模拟 all-reduce，验证与 dense 数值等价 + 行切反例 + 通信/显存账本 | ✅ [`L0_tp_mlp.py`](L0_tp_mlp.py) · [教程](tutorial_L0.md) |
| **L1** | torch.distributed 多进程（gloo，CPU 即可）跑张量并行 MLP：真实 all-reduce 前向 + f/g 算子反向 + 行切反例 + 16P/N 账本实测 + 通信墙钟对照 L0 账本 | ✅ [`L1_tp_real_allreduce.py`](L1_tp_real_allreduce.py) · [教程](tutorial_L1.md) |
| **L2** | 流水线并行：4 块按层切成 2 个真实 stage（gloo p2p），GPipe/1F1B 双调度步后权重 bit 级相同，bubble 实测对公式 (N-1)/(m+N-1)，PP 通信 524,288 B/rank/step = 1/8 TP，1F1B 死锁复现 + batch_isend_irecv 非阻塞修复（40/40 自检） | ✅ [`L2_pipeline_microbatch.py`](L2_pipeline_microbatch.py) · [教程](tutorial_L2.md) |
| **L3** | 序列并行 + TP/PP/SP 组合 + MFU：AR≡RS+AG 恒等式下 LN/Dropout 切序列维，dW 分片 bit 级等价，区域激活恰 1/t，反向重放 gather 使 wire 2m→2.5m（+25%），PP 接缝 SP 下减半（schedules.py seq//tp），γ/β 梯度部分和跨 TP 组归约（finalize_model_grads），GEMM 标定峰值 + Megatron FLOPs 公式的三段 MFU 分解（24/24 自检） | ✅ [`L3_sp_tp_pp_mfu.py`](L3_sp_tp_pp_mfu.py) · [教程](tutorial_L3.md) |

**环境依赖**：L0 零外部依赖（纯标准库），CPU 即跑：`python3 L0_tp_mlp.py`；
L1 需 `torch`（CPU 即可，gloo 多进程，无需 GPU）：
`python L1_tp_real_allreduce.py`；
L2 同样只需 `torch`（CPU，gloo 2 进程，实测 ~2s）：
`python3 L2_pipeline_microbatch.py`
（torch 2.13.0 实测；计时行 14 行浮动，掩码口径见 tutorial_L2 §12）。
L3 同样只需 `torch`（CPU，gloo 4 进程 = PP2×TP2，实测 ~3-15s）：
`python3 L3_sp_tp_pp_mfu.py`
（torch 2.13.0 实测；计时行 9 行浮动，掩码口径见 tutorial_L3 §13）。

### 可选 GPU/NCCL 证伪探针

- [L1_gpu_verify.py](L1_gpu_verify.py)：用 `--world-size 2/4/8` 在单机可见 CUDA GPU
  上复验 TP 前反向等价、f/g 通信次数、16P/N 显存账与 NCCL all-reduce scaling。
- [L2_gpu_verify.py](L2_gpu_verify.py)：在单机两卡上复验 PP 数值等价、P2P
  字节账、GPipe/1F1B live-microbatch 上界与 bubble 观测。
- 两者都支持 `--master-port`（或 `NANO_MEGATRON_MASTER_PORT`），不绑定机器、
  私有环境路径或 GPU 型号；末行输出 `RESULT_JSON=`。时间/带宽只对运行时记录的
  torch、CUDA、NCCL、driver 与设备成立，不是 L1/L2 CPU 教程完成状态的前提。

#### 2026-08-31 单机 L20 复验

环境快照为 2 张可见 NVIDIA L20（机器共 8 张、每次实验只使用 2 张），driver 550.90.07、
PyTorch 2.9.1+cu128、CUDA 12.8、NCCL 2.27.5。L1 与 L2 各独立运行两次，四次均
exit 0、stderr 为空；以下只把跨运行稳定的正确性账作为验收证据：

| 探针 | 稳定结果 | 两次共同 digest |
|---|---|---|
| L1 / TP=2 | dense 对照相对误差约 `1.9e-7`；错误的 reduce-before/after-GeLU 顺序产生 `1.667e+02` 误差，修正后为 `7.629e-05`；每 rank 参数+梯度+Adam 账为 256 KiB | `c86a4b72d7086762e3bbb147dc6f257e` |
| L2 / PP=2 | 42/42；同 microbatch mirror 参数逐位相同；改变 FP32 归约形状的 full-batch 对照 `max|Δ|=2.876e-05 ≤ 5.166e-05`；每 rank P2P 账为 524,288 B/step | `02f8fe71ec5202ea740e078aa2f121b6` |

L1 的 16 MiB all-reduce 两次样本约 0.782–0.785 ms（21.4–21.5 GB/s）；L2 的
GPipe/1F1B 墙钟与 bubble 会随调度噪声变化，只验证 microbatch 增多时 bubble 整体下降，
不要求实测幅度等于理想公式。这一子批单独证明单机 TP2/PP2 toy 正确性与通信账；
它没有使用 4/8 卡，也不是生产训练吞吐基准。后续扩展实验见下一节。

#### 2026-09-03 TP2/4/8 scaling 复验

同一软件栈上，参数化后的 L1 probe 对 TP2、TP4、TP8 各运行两次；六次均 exit 0、
stderr 为空、7/7。脚本 SHA256 为
`d093ef1d951d343c4753d246444ca19b571739f97859aa9adc5d775d5032b8ec`，
远端运行副本与仓库逐字一致。

| TP | 每 rank 参数+梯度+Adam | 前向相对误差 | FP32 16 MiB latency | busbw | 两次共同 digest |
|---:|---:|---:|---:|---:|---|
| 2 | 256 KiB | `1.9e-7` | 0.784–0.785 ms | 21.4 GB/s | `be3aa882721b9e67cf7279097286271f` |
| 4 | 128 KiB | `2.1e-7` | 1.874–1.875 ms | 13.4 GB/s | `e3742f02a539ecca9c1b9d155519929e` |
| 8 | 64 KiB | `1.5e-7` | 2.067–2.069 ms | 14.2 GB/s | `9b9a41d443b5a89d49036ccfe17ede55` |

本机拓扑没有 NVLink：GPU0–1 为 `PIX`，GPU0–3 扩展到同 NUMA 的 `NODE`，8 卡跨
NUMA `SYS`。因此表中既有参与 rank 数变化，也有路径层级变化；它们共同出现，但本实验
不做单因果归因。完整解释与 `busbw` 口径见 [L1 教程 §7.1](tutorial_L1.md)。

## 核心要讲清的点

- TP 切 MLP：列并行切 W1，行并行切 W2，all-reduce 插在激活上
- TP 切 Attention：按 head 切，天然并行
- PP 的 bubble：为什么 micro-batch 能减小空转
- SP：在 LayerNorm/Dropout 上切序列维，省 activation 显存
- SP 的代价：AR≡RS+AG 分解通信中性，但反向重放 gather 使 wire +25%（显存-通信 tradeoff）
- 组合效应：SP 下 PP 接缝字节减半（stage 间传的就是分片形态）；γ/β 梯度变部分和须跨 TP 组归约
- MFU：GEMM 标定峰值 × Megatron FLOPs 公式，把并行效率压成一个可追责的数字

## 费曼自检

- 能不能手画一个 2 卡 TP 的 MLP 前向，标出 all-reduce 的位置和通信量？
- 能不能把一块 LN+MLP 的 SP 前向画出来：哪里 AG、哪里 RS、哪些激活是 [T/t,H]？
  反向多出来的那次 gather 是为了算什么？不重放行不行（代价是什么）？

## 权威实现与延伸

- 对标源码：Megatron-LM `github.com/NVIDIA/Megatron-LM`（TP/PP/SP 切分与通信）
  - L2 锚点（main 分支 2026-08-08 现场抓取核验，行号以抓取日为准）：
    `schedules.py:L2129`（without_interleaving 调度）/ `L2252-2253`（warmup 公式 N-rank-1）/
    `p2p_communication.py:L17-52`（`_batched_p2p_ops`）/ `L257-262`（wait 与 race-condition guard）
  - L3 锚点（main 分支 2026-08-10 现场抓取核验，行号以抓取日为准）：
    `mappings.py:L118/L159/L280/L300/L355`（序列维 gather/reduce-scatter 与三个 SP 算子）/
    `layers.py:L565-573`（融合线性 fwd gather）/ `L609-618`（bwd 重放 gather 算 wgrad）/
    `L1472-1478`（RowParallel SP 分支）/ `transformer_block.py:L595-598`（SP 区域 RNG fork）/
    `schedules.py:L2122-2123`（PP p2p 形状 SP → seq//tp）/
    `training.py:L391/L428-431/L548`（FLOPs 公式：MLP 4·exp·tokens·h²，×3）/
    `finalize_model_grads.py:L416/L451-453`（SP 下 LN 梯度跨 TP 组 SUM 归约）
- 论文：SP = arXiv:2205.05198（Reducing Activation Recomputation in Large
  Transformer Models）；MFU 口径 = PaLM arXiv:2204.02311
- 概念延伸：与 `nano-fsdp` 对比（TP vs ZeRO 的取舍）
