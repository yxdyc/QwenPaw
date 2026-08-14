# nano-fsdp L1 — 真实 PyTorch FSDP：把 L0 的账本跑成分片数字

> **核心机制**：用真实 `torch.distributed.fsdp.FullyShardedDataParallel` 训练一个
> TinyLM，量出 DDP 与 FSDP 下每 rank 的「模型状态」内存，验证 L0 的 16× 公式。
>
> **运行要求**：需要 `torch`（CPU 即可）。本机命令：
> ```bash
> python L1_single_card_fsdp.py
> ```
> 真实生产在 GPU 上运行，本机用 CPU 做本地可复现 demo，分片机制相同。

---

## 1. 本节目标

L0 手算了 ZeRO 分片的显存账。L1 把账本接上真实 PyTorch：

- 用 `DistributedDataParallel`（DDP）跑一个完整副本，验证每 rank 真的接近 **16 × 参数字节**；
- 用 `FullyShardedDataParallel`（FSDP）按 Transformer block 分片，验证每 rank 接近 **16P / W**；
- 看清楚 FSDP **没有消灭** 内存，只是把同一份 16P 切成 W 份；
- 体会 `auto_wrap_policy` 为什么存在——它决定「多大的盒子（module）被包成一层 FSDP」。

---

## 2. 跑代码

```bash
python \
  tutorial/material/02-pretraining-cpt/nano-fsdp/L1_single_card_fsdp.py
```

输出：

```
=================================================================
nano-fsdp L1 — real PyTorch FSDP vs DDP on CPU
=================================================================
TinyLM: vocab=128 dim=64 layers=2 | total params = 116,480
Adam training state (L0 formula) = 16 bytes/param
Running on CPU with 2 processes

[DDP] per-rank model state (params + grads + optimizer)
-----------------------------------------------------------------
  rank 0  params=  0.44 MiB  grads=  0.44 MiB  optimizer=  0.89 MiB  total=  1.78 MiB
  rank 1  params=  0.44 MiB  grads=  0.44 MiB  optimizer=  0.89 MiB  total=  1.78 MiB
  sum across ranks = 3.55 MiB
  L0 expected full replica = 1.78 MiB

[FSDP] per-rank model state (params + grads + optimizer)
-----------------------------------------------------------------
  rank 0  params=  0.22 MiB  grads=  0.22 MiB  optimizer=  0.44 MiB  total=  0.89 MiB
  rank 1  params=  0.22 MiB  grads=  0.22 MiB  optimizer=  0.44 MiB  total=  0.89 MiB
  sum across ranks = 1.78 MiB
  L0 expected full replica = 1.78 MiB
  L0 expected per-rank shard = 0.89 MiB

✅ self-check passed
```

---

## 3. 数字解读

### 3.1 DDP = ZeRO-0

两张 rank 都接近 **1.78 MiB**，等于 L0 预言的完整副本：

```
116,480 params × 16 bytes/param = 1,863,680 bytes ≈ 1.78 MiB
```

DDP 每张卡都保存完整的参数、梯度、Adam 状态；它只是多卡并行做 all-reduce 梯度，**没有分片**。

### 3.2 FSDP = ZeRO-3

每张 rank 接近 **0.89 MiB**，是完整副本的一半：

```
1.78 MiB / 2 ranks ≈ 0.89 MiB
```

FSDP 把每个 Transformer block 的参数、梯度、Adam 状态都切到两个 rank 上。前向时 all-gather 拼回整层参数算完再丢掉；反向时 reduce-scatter 把梯度写回自己的 shard。

### 3.3 关键结论：总内存没有减少

注意 `sum across ranks`：

- DDP：1.78 × 2 = **3.55 MiB**（两张卡各自存一套完整书）
- FSDP：0.89 × 2 = **1.78 MiB**（两张卡合起来只存一套书）

FSDP 的节省是**每张卡**的节省，不是集群总内存的魔术。这正好对应 L0 的费曼反例：「无限多卡不能把显存压到 0」——因为这本书总得有人拿着。

### 3.4 外推到 7B/8 卡

L0 已经算过：7B 参数、Adam、8 卡 FSDP，每卡模型状态约 14 GiB。本节的 TinyLM 只是把同样的公式缩到能在 CPU 秒开的大小。

---

## 4. 代码结构：三个真实 PyTorch API

### 4.1 `DistributedDataParallel` —— 完整副本

```python
model = DDP(raw_model, device_ids=None)  # CPU 上 device_ids=None
```

DDP 在反向时做 all-reduce 梯度，让每张卡的梯度都变成全局平均。内存上它与单卡训练一样完整，只是复制了 W 份。

### 4.2 `FullyShardedDataParallel` —— 分片

```python
model = FSDP(
    raw_model,
    auto_wrap_policy=fsdp_wrap_policy,
    device_id=torch.device('cpu'),
)
```

`auto_wrap_policy` 决定哪些子 module 被单独包成 FSDP 单元。这里把每个 `TransformerEncoderLayer` 包一层：

```python
def fsdp_wrap_policy(module, recurse, nonwrapped_numel):
    return isinstance(module, nn.TransformerEncoderLayer)
```

真实大模型常配 `size_based_auto_wrap_policy(min_num_params=...)` 或 `transformer_auto_wrap_policy`，让每一层/每一块独立分片；如果不配 `auto_wrap_policy`，整个模型只被包成**一个** FSDP 单元，仍能 shard，但通信粒度粗。

### 4.3 Adam 状态才是那 16 倍的大头

脚本里的 `model_state_bytes` 把三部分拆开：

```python
param_bytes + grad_bytes + opt_bytes
```

其中 `opt_bytes` 是 Adam 的 `exp_avg` 和 `exp_avg_sq`，每个各占 4 bytes/param，合计 8 bytes/param。加上参数和梯度各 4 bytes/param，就是 L0 的 **16 bytes/param**。

---

## 5. 与权威实现的对应关系

| nano-fsdp L1 | 真实 PyTorch API / 文档 |
|--------------|--------------------------|
| DDP 完整副本 | `torch.nn.parallel.DistributedDataParallel` |
| FSDP 分片 | `torch.distributed.fsdp.FullyShardedDataParallel` |
| 按层自动包装 | `torch.distributed.fsdp.wrap.size_based_auto_wrap_policy`（或自定义 lambda policy） |
| 进程组初始化 | `torch.distributed.init_process_group` |
| Adam 状态 | `torch.optim.Adam` 的 `exp_avg` / `exp_avg_sq` |
| 官方文档 | `https://docs.pytorch.org/docs/stable/fsdp.html`（L0 已验证可达） |

L1 只到 API 行为和内存数字层面；源码级 all-gather / reduce-scatter 实现、mixed-precision + activation checkpointing 的端到端账本是 L2/L3 的任务。

---

## 6. 费曼自检

### 讲给外行听：搬书升级版

L0 把 ZeRO 比作「书、笔记、文具」分片。L1 用真实 PyTorch 做了验证：

- **DDP**：两个工位各放一套完整的书 + 笔记 + 文具。两人各自抄完全书，然后对答案（all-reduce 梯度）。每人桌子都占 1.78 MiB。
- **FSDP**：两个工位只各放半套书、半本笔记、半套文具。需要看另一半时，临时从隔壁借（all-gather），看完还回去；算出的局部笔记再汇总（reduce-scatter）。每人桌子只占 0.89 MiB，但**整个房间里的书并没有变少**。

### 思考题

1. 为什么脚本里 FSDP 的 `sum across ranks` 等于 DDP 的 `per-rank total`？这个等式在真实 7B/8 卡训练里还成立吗？
2. `auto_wrap_policy` 如果不按 Transformer block 包装，而是只包整个模型，内存数字会怎么变？通信次数会怎么变？
3. 为什么 DDP 的 `sum across ranks` 是 FSDP 的两倍？集群总显存预算紧张时，这是选 FSDP 的核心理由吗？
4. 脚本里 Adam 用 fp32，所以参数 + 梯度 + m + v 刚好 16 bytes/param。如果改用 mixed precision（fp16 参数 + fp32 master + m + v），每 rank 的哪一栏会变化？

### 反例

> 「FSDP 让训练大模型的总显存需求减半。」

**错**。FSDP 让**每张卡**的模型状态减半（近似 16P/W），但所有卡合起来仍然存一份完整模型状态（16P）。它解决的是「单卡装不下」而不是「集群总需求减少」。真正减少总内存的是**模型压缩**（剪枝、量化、MoE 稀疏化），不是数据并行分片。

---

## 7. 下一步（L2/L3）

- **L2**：手写 ZeRO-1/2/3 的显存与通信量差异，把 all-gather / reduce-scatter 的次数和带宽算出来。
- **L3**：对照 PyTorch FSDP 源码，看 `FlatParameter` 怎么拼参数、`_post_backward_hook` 怎么做 reduce-scatter，以及 mixed precision + activation checkpointing 如何改变 16× 这个数字。

---

## 8. 溯源与不确定声明

- PyTorch FSDP 文档：`https://docs.pytorch.org/docs/stable/fsdp.html`（L0 已验证可达）。
- DDP / FSDP CPU 可跑性：本脚本在本机 `torch 2.4.1` + gloo backend 上实测通过；多 GPU 生产环境机制相同，数字按 GiB 放大。
- 本脚本未覆盖：真实 GPU 上的 `nvidia-smi` 显存（含 activations / buffer / fragmentation）、mixed precision 下的 fp32 master weight、activation checkpointing；这些属 L2/L3，标 `[TODO: verify on real GPU]`。
