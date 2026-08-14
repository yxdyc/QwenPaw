# nano-fsdp L0 — ZeRO 显存账本

> **核心机制**：用纯 Python 手算「参数 + 梯度 + 优化器状态」占多少显存，理解 Adam fp16 训练为什么是 **16 × 参数量**。
>
> **运行要求**：零外部依赖，CPU 即跑。文件：`L0_memory_ledger.py`。

---

## 1. 本节目标

学完这一节，你能口算出：

- 一个 7B 模型用 Adam + fp16 训练，**单副本**需要多少显存；
- ZeRO-1/2/3 分别把什么切了，每卡显存怎么变；
- 为什么真实系统里 ZeRO-3 不是免费的——它省了显存，但多了通信。

---

## 2. 显存都去哪了？

训练 LLM 时，显存不只放「模型权重」。以 Adam + mixed precision（fp16/bf16）为例，每个参数旁边还站着四样东西：

| 内容 | dtype | 每参数字节 |
|------|-------|-----------|
| parameter | fp16 | 2 |
| gradient | fp16 | 2 |
| master parameter（Adam 需要 fp32） | fp32 | 4 |
| momentum `m` | fp32 | 4 |
| variance `v` | fp32 | 4 |
| **合计** | — | **16** |

所以常说「Adam fp16 训练 ≈ 16 × 参数量」。这个数字是口算显存的起点。

> **注意**：这 16 倍只算**模型状态**（model states），不算 activations、通信 buffer、临时张量、OS 开销。L1/L2 会接着量这些。

---

## 3. 跑代码：先看 Adam 账本

```bash
python3 tutorial/material/02-pretraining-cpt/nano-fsdp/L0_memory_ledger.py
```

输出节选：

```
============================================================
1. Adam + fp16 训练的显存账本（按每参数计）
============================================================
component               dtype    bytes/param
--------------------    -----    ------------
parameter               fp16     2
gradient                fp16     2
master parameter        fp32     4
momentum (m)            fp32     4
variance (v)            fp32     4
----------------------------------------------
total                            16

=> 结论：adam_training_state_bytes(P) = 16 × P bytes = 16 byte/param
```

代码里这行就是本质：

```python
def adam_training_state_bytes(params: int) -> int:
    return params * 16
```

不是魔法，只是把上面五列相加。

---

## 4. 跑代码：ZeRO 分片怎么省显存

DeepSpeed ZeRO / PyTorch FSDP 的核心思想是：**数据并行时，没必要每张卡都存一份完整的 16× 状态**。按分片激进程度分四档：

| ZeRO stage | 分片内容 | 每卡显存（近似） |
|-----------|---------|----------------|
| ZeRO-0（DDP） | 不分片 | 16P |
| ZeRO-1 | 优化器状态（master + m + v） | 4P + 12P/W |
| ZeRO-2 | 优化器状态 + 梯度 | 2P + 14P/W |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 16P/W |

其中 `P` 是总参数量，`W` 是 GPU 数。

脚本用 7B 模型、1/2/4/8 卡算了一张表：

```
============================================================
3. 7B 模型在不同 ZeRO stage 下的每卡显存
============================================================
ZeRO stage         1 GPU      2 GPUs      4 GPUs      8 GPUs
------------------------------------------------------------
ZeRO-0            112.0 GB       112.0 GB       112.0 GB       112.0 GB
ZeRO-1            112.0 GB        70.0 GB        49.0 GB        38.5 GB
ZeRO-2            112.0 GB        63.0 GB        38.5 GB        26.2 GB
ZeRO-3            112.0 GB        56.0 GB        28.0 GB        14.0 GB

说明：
  - ZeRO-0 = 普通 DDP，每张卡都存完整副本，不随卡数减少。
  - ZeRO-3 把参数也分片，显存随卡数近似线性下降（7B/8卡 = 14 GB）。
  - OS 显存工具通常按 GiB（1024^3）显示，数值会比 GB 小约 7%；这里用 GB 方便心算。
  - 实际还要加 activations、通信 buffer、OS 开销；这里只算模型状态。
```

**重点看两个数**：

- 7B/8卡/ZeRO-3：模型状态只要 **14 GB**。这解释了为什么单卡 24 GB/40 GB 也能参与训练大模型。
- ZeRO-0 永远是 112 GB：它跟 DDP 没区别，只是复制。

---

## 5. 口算练习：80B 模型、64 卡、ZeRO-3

脚本给出的答案是：

```
============================================================
4. 口算练习：80B 参数，64 卡，ZeRO-3
============================================================
params = 80B, gpus = 64, ZeRO-3
model state per GPU ≈ 20.00 GB
=> 若 activations 再占 ~10–20 GB，单卡仍需 >30 GB（A100/H100 80G 才舒服）。
```

心算过程：80B × 16 bytes = 1280 GB 总状态；÷ 64 卡 = 20 GB/卡。加上 activations、buffer，A100 40G 会比较紧张，80G 才宽松。这个估算能力是调分布式训练时的基本功。

---

## 6. 与权威实现的对应关系（L0 只到概念层）

| 概念 | nano-fsdp L0 | 权威实现 `[TODO: verify source]` |
|------|--------------|----------------------------------|
| 16× 显存估算 | 脚本里的 `adam_training_state_bytes(P)` | PyTorch FSDP 文档中的 mixed-precision memory footprint；DeepSpeed ZeRO 论文 |
| ZeRO-1/2/3 分片 | `zero_memory_per_gpu(..., stage=...)` | DeepSpeed ZeRO stage-1/2/3 参数切分策略 `[TODO: verify source]` |
| all-gather / reduce-scatter | 本脚本未模拟，只讲显存结果 | FSDP `all_gather` 前向拼参数、`reduce_scatter` 反向传梯度 `[TODO: verify source]` |

L0 不展开源码行级对应，那是 L2/L3 的任务。这里先建立「显存账本」的直觉。

---

## 7. 费曼自检

### 讲给外行听：搬书

想象一群工人（GPU）要抄一本 1000 页的书（模型）。

- **ZeRO-0**：每人手里都放一本完整的书、一份完整笔记、一套完整文具。房间（显存）很快堆满。
- **ZeRO-1**：每人只保留自己负责那几页的文具（优化器状态分片），需要更新别人的页时再临时汇总。
- **ZeRO-2**：笔记也分片，每人只记自己负责那几页。
- **ZeRO-3**：书也拆开，每人只拿几页；需要看别的页时，临时从同事手里借（all-gather）。

代价：ZeRO-3 借书最频繁，所以**通信变多**；但每人占用的桌子（显存）最小。

### 思考题

1. 如果换成 **SGD** 而不是 Adam，每参数训练状态是几倍？为什么 FSDP/ZeRO 文献里常拿 Adam 说事？
2. 7B 模型 ZeRO-3/8 卡只要 14 GB 模型状态，为什么真实训练往往还要留大量显存余量？余量可能被什么吃掉？
3. ZeRO-3 下，如果 GPU 数从 8 变 16，模型状态显存会怎么变？通信量会怎么变？

### 反例

> 「把 ZeRO-3 开到无限多卡，每卡显存就能无限趋近于 0。」

**错**。当卡数超过参数总量（按分片粒度算），每张卡只存几个参数，但前向/反向时为了计算一个 token 的 activation，还是要临时把整层参数 all-gather 回来。通信次数和延迟会爆炸，而 activations 本身并不随 ZeRO 减少。所以 ZeRO-3 的显存收益有**边际递减**，不是无限免费。

---

## 8. 下一步（L1）

L1 会用真实 PyTorch 包一个小模型，分别用 DDP 和 FSDP 跑单卡/双卡训练，对比 `nvidia-smi` 看到的显存差异，并体会：

- `FullyShardedDataParallel` 的 `auto_wrap_policy`；
- mixed precision 下 master weight 的去向；
- 为什么 L0 的 16× 估算和真实 `nvidia-smi` 不完全一样。

---

## 9. 溯源与不确定声明

- Adam 16× 显存估算：混合精度训练与 ZeRO 的常见结论，来源 DeepSpeed ZeRO 论文 `[TODO: verify arXiv]` 与 PyTorch FSDP 文档。
- PyTorch FSDP 文档链接：`https://docs.pytorch.org/docs/stable/fsdp.html`（已验证可达）。
- DeepSpeed ZeRO 仓库：`https://github.com/deepspeedai/DeepSpeed`（已验证可达）。
- 源码级对应（FSDP 内部 all-gather/reduce-scatter 实现）：本 L0 未展开，标 `[TODO: verify source]`，L2/L3 补齐。
