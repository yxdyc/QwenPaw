# nano-megatron · L1 教程：张量并行的真实代价——真进程、真 all-reduce、真反向

> **本节目标（L1）**：把 L0 的 toy 切分搬上**真实的分布式通信**——
> `torch.distributed`（gloo backend）起多个进程，每个进程只持有 1/N 的权重，
> 用**真实 `dist.all_reduce`** 完成前向与反向，回答 L0 留下的问题：
> ① 真实通信下数值等价还成立吗？② 反向的 all-reduce 藏在哪？③ all-reduce 到底多贵？
> **前置**：[nano-megatron L0](tutorial_L0.md)（切法与两个账本）；
> [nano-fsdp L1](../nano-fsdp/tutorial_L1.md) 的多进程模式同款，读过更好。
> **本节 K+1**：从「模拟 all-reduce 的语义」到「付出 all-reduce 的代价」：
> 真实进程、真实集合通信、真实墙钟，外加 L0 没展开的反向（f/g 算子）。

---

## 1. L0 模拟了什么，L1 要付出什么

L0 里 `mat_sum`（矩阵逐元素和）模拟了 all-reduce 的**语义**：把 N 份部分和加成一份。
但它跳过了真实系统里最要紧的三件事：

1. **通信是真实进程之间的数据搬运**（TCP/共享内存），不是内存里一次加法——有墙钟代价；
2. **反向也要通信**。L0 只引用了论文「每层 fwd 2 + bwd 2 次 all-reduce」的结论，
   反向那 2 次从哪来没有落地；
3. **每个 rank 真实持有的内存**是多少——L0 是算术，L1 要逐字节量出来。

L1 全部用真的：gloo 多进程、`dist.all_reduce`、手写 f/g 两个 autograd 算子
（Megatron `mappings.py` 的同构最小版）、逐字节的 params/grads/Adam 账本、墙钟计时。
没有 mock——这是 CPU 上的真实分布式训练切片，只是互连是 loopback 而非 NVLink。

---

## 2. 先跑起来

文件：`L1_tp_real_allreduce.py`，依赖仅 `torch`（CPU 即可）。先起 TP=2 跑完整套件
（前向/反向/反例/账本/计时），再起 TP=4 复验等价性与 P/N 账本。

```bash
$ python L1_tp_real_allreduce.py
```

真实输出（seed=7，run1；计时行以外的所有行连跑 3 遍逐字节一致，计时行波动区间见 §7）：

```text
====================================================================
nano-megatron L1 — real tensor parallel MLP (torch.distributed/gloo)
====================================================================
shape: X[128,64]  W1[64,256]  W2[256,64]  (f = 4h, GeLU MLP) | fp32 | seed=7
MLP params P = 32,768 | torch 2.4.1, gloo, CPU
[TP=2] forward: W1 列切 + W2 行切 + 1 次真实 all-reduce
    max|Y_tp - Y_ref| = 1.602e-04 (相对 3.9e-07)  ✅ 舍入级等价
[TP=2] backward: f/g autograd 算子
    max|dX - dX_ref| = 1.526e-04   (f 反向 all-reduce 汇合)
    max|dW1 分片 - dW1_ref 对应块| = 0.000e+00   (本地即完整，零通信)
    max|dW2 分片 - dW2_ref 对应块| = 0.000e+00   (本地即完整，零通信)
    集合通信计数: fwd all-reduce = 1, bwd all-reduce = 1
[TP=2] 反例: W1 行切（切输入维）
    naive（先 GeLU 再 all-reduce）: err = 1.684e+02  ❌ 错
    fixed（先 all-reduce 再 GeLU）: err = 9.155e-05  ✅ 对
    design                    fwd all-reduce    message           params/rank
    column-row (Megatron)     1                 32.0 KiB [T,H]    16,384
    row-first naive           1 (wrong result)  128.0 KiB [T,F]   24,576
    row-first fixed           1                 128.0 KiB [T,F]   24,576
[TP=2] 账本实测: 每 rank params + grads + Adam 状态
    rank 0: params=64.0 KiB  grads=64.0 KiB  optimizer=128.0 KiB  total=256.0 KiB
    rank 1: params=64.0 KiB  grads=64.0 KiB  optimizer=128.0 KiB  total=256.0 KiB
    dense 完整副本 = 16 x 32,768 = 512.0 KiB | TP 各 rank 之和 = 512.0 KiB
[TP=2] 计时: 真实 all-reduce 墙钟（gloo loopback, fp32, max over ranks, 5 次均值）
    fp16 all_reduce 探测（gloo）: supported, sum(1 x 2 ranks) = 2.0  ✅ 数值正确（本机 torch 2.4.1 + gloo 实测）
    msg   1.0 MiB:     2.49 ms/call
    msg  16.0 MiB:    23.69 ms/call
    msg  32.0 MiB:    46.51 ms/call
    toy fwd 计算（每 rank, 本形状）:    0.038 ms
[TP=4] forward: W1 列切 + W2 行切 + 1 次真实 all-reduce
    max|Y_tp - Y_ref| = 1.831e-04 (相对 4.4e-07)  ✅ 舍入级等价
[TP=4] backward: f/g autograd 算子
    max|dX - dX_ref| = 1.526e-04   (f 反向 all-reduce 汇合)
    max|dW1 分片 - dW1_ref 对应块| = 0.000e+00   (本地即完整，零通信)
    max|dW2 分片 - dW2_ref 对应块| = 0.000e+00   (本地即完整，零通信)
    集合通信计数: fwd all-reduce = 1, bwd all-reduce = 1
[TP=4] 反例: W1 行切（切输入维）
    naive（先 GeLU 再 all-reduce）: err = 2.725e+02  ❌ 错
    fixed（先 all-reduce 再 GeLU）: err = 1.068e-04  ✅ 对
    design                    fwd all-reduce    message           params/rank
    column-row (Megatron)     1                 32.0 KiB [T,H]    8,192
    row-first naive           1 (wrong result)  128.0 KiB [T,F]   20,480
    row-first fixed           1                 128.0 KiB [T,F]   20,480
[TP=4] 账本实测: 每 rank params + grads + Adam 状态
    rank 0: params=32.0 KiB  grads=32.0 KiB  optimizer=64.0 KiB  total=128.0 KiB
    rank 1: params=32.0 KiB  grads=32.0 KiB  optimizer=64.0 KiB  total=128.0 KiB
    rank 2: params=32.0 KiB  grads=32.0 KiB  optimizer=64.0 KiB  total=128.0 KiB
    rank 3: params=32.0 KiB  grads=32.0 KiB  optimizer=64.0 KiB  total=128.0 KiB
    dense 完整副本 = 16 x 32,768 = 512.0 KiB | TP 各 rank 之和 = 512.0 KiB

====================================================================
✅ self-check passed: 前向/反向与 dense 舍入级等价 · 行切反例在真实通信上依旧错且可修复 · 每 rank 状态恰为 16P/N · fwd/bwd 各 1 次 all-reduce
====================================================================
```

**L1 基线指标**：真实 all-reduce 下的前向最大误差 `1.602e-04`（相对 `3.9e-07`，舍入级）；
每块前向/反向各 `1` 次 all-reduce（计数器实测）；每 rank 训练状态 `256 KiB = 16P/2`（实测）。
对照 L0：误差从纯 Python fp64 的 `1.110e-16` 变成 fp32+BLAS+gloo 的舍入级——等价性的**判据**
从来不是零，而是舍入量级（§3）。

---

## 3. 前向：真实 all-reduce 下的数值等价

每个 rank 只持有 `W1` 的一个列块 `[h, f/N]` 和 `W2` 的一个行块 `[f/N, h]`（真实切片、
真实独立进程、真实只占 1/N 内存），前向就是 L0 的公式搬上 torch：

```python
Y_tp = ReduceFromTensorParallelRegion.apply(          # g 算子：fwd = all-reduce
    F.gelu(CopyToTensorParallelRegion.apply(X_in)     # f 算子：fwd = identity
           @ W1_r) @ W2_r)                            # 列并行 -> GeLU -> 行并行
```

`max|Y_tp - Y_ref| = 1.602e-04`，除以 `max|Y_ref|` 后相对误差 `3.9e-07`——
fp32 的机器精度（eps ≈ 1.19e-7）量级。**为什么不是 0？** 两处舍入来源：

- 行并行把 `W2` 的内积拆成 N 段分别算再相加，加法顺序与 dense 不同；
- gloo all-reduce 的求和顺序也与「内存里一次 `mat_sum`」不同。

浮点加法不满足结合律，顺序一变，舍入就变——这正是 L0 §3 预告的现象在真实通信上的样子。
工程判据：**切分正确性看相对误差是否在舍入级**（脚本 assert 相对 < 1e-5），
要求逐 bit 相同既不可能也不必要。TP=4 时相对误差 `4.4e-07`，同量级——
等价性不随 degree 退化。

---

## 4. 反向：f/g 算子——「fwd 1 次 + bwd 1 次」第一次落地

L0 引用了 Megatron 论文「每层 fwd 2 + bwd 2 次 all-reduce」，反向那半一直没有实体。
L1 把它做出来：两个 `torch.autograd.Function`，与 Megatron `mappings.py` 同构：

```python
class CopyToTensorParallelRegion(torch.autograd.Function):      # f 算子
    @staticmethod
    def forward(ctx, x):            # 前向 identity：输入本就各 rank 复制
        return x
    @staticmethod
    def backward(ctx, grad_output):  # 反向 all-reduce：汇合各 rank 的 dX 部分和
        g = grad_output.contiguous().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        COMM["bwd"] += 1
        return g

class ReduceFromTensorParallelRegion(torch.autograd.Function):  # g 算子
    @staticmethod
    def forward(ctx, x):             # 前向 all-reduce：行并行部分和拼成完整输出
        x = x.contiguous().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        COMM["fwd"] += 1
        return x
    @staticmethod
    def backward(ctx, grad_output):  # 反向 identity：dY 各 rank 本就相同
        return grad_output
```

对照权威实现（2026-08-05 main 分支实测，见 §9）：`_CopyToModelParallelRegion`
（mappings.py:L201）与 `_ReduceFromModelParallelRegion`（mappings.py:L221），
前向/反向的 identity/reduce 组合逐条一致；它们内部调用的 `_reduce`（L22）
就是一句 `torch.distributed.all_reduce`（L35）。

跑完 `Y_tp.sum().backward()`，输出给出三个关键事实：

1. **`max|dX - dX_ref| = 1.526e-04`（舍入级）**：dX 是每个 rank 的部分贡献
   `dP_r @ W1_rᵀ` 之和，靠 **f 的反向 all-reduce** 汇合——这就是反向那 1 次通信。
2. **`max|dW1 分片 - dW1_ref 对应块| = 0.000e+00`，dW2 同样**：不是「很小」，
   是**逐 bit 相等**。原因：分片参数的梯度只依赖本地已有的量——
   `dW1_r = Xᵀ @ dP_r`（X 各 rank 完整复制、dP_r 本地算出）、
   `dW2_r = H_rᵀ @ dY`（H_r、dY 都在本地）。**每个 rank 的权重分片梯度
   无需任何额外通信就是完整的。**
3. **计数器恰为 `fwd = 1, bwd = 1`**：一块「列并行+行并行」前向 1 次、反向 1 次。
   Transformer 每层有 attention 输出投影、MLP 第二线性两个这样的块（L0 §6），
   乘 2 就是论文的「每层 fwd 2 + bwd 2」——账本在这里闭环。

f/g 的不对称值得盯住看：**g 前向 reduce、反向 identity；f 前向 identity、反向 reduce。**
规律一句话：**前向里被复制的量（输入 X），反向里要聚合它的梯度；
前向里被求和的量（部分和），反向里梯度直接通过（因为各 rank 的 dY 本来就相同）。**
通信总是出现在「复制 ⇄ 求和」的转换处，前向反向各一次，不多不少。

> 诚实备注：TP=2 与 TP=4 的 dX 误差在本次 seed=7 下恰好同为 `1.526e-04`，
> 这是舍入误差的数据依赖巧合——换 seed=99 复跑即分离为 `1.373e-04`（TP=2）
> 与 `1.984e-04`（TP=4）。列出来是为了提醒：误差数字是实测，不是公式推出来的。

---

## 5. 反例搬上真实通信：错切法不仅错，还更贵

L0 的定理——「非线性破坏可加性，`GeLU(p1+p2) ≠ GeLU(p1)+GeLU(p2)`」——
在真实 `dist.all_reduce` 上原样复现：

```text
naive（先 GeLU 再 all-reduce）: err = 1.684e+02  ❌ 错
fixed（先 all-reduce 再 GeLU）: err = 9.155e-05  ✅ 对
```

但 L1 能量化 L0 没量化的东西——**通信与权重的完整代价**：

```text
design                    fwd all-reduce    message           params/rank
column-row (Megatron)     1                 32.0 KiB [T,H]    16,384
row-first naive           1 (wrong result)  128.0 KiB [T,F]   24,576
row-first fixed           1                 128.0 KiB [T,F]   24,576
```

- **消息量**：行切的 all-reduce 对象是 pre-activation `[T, f]`，而列切→行切设计
  把 all-reduce 挪到 W2 之后，对象缩成 `[T, h]`。`f = 4h` 时**差 4 倍**——
  「列切→行切」不只是「能 1 次通信算对」，还是消息量最小的切法。
- **权重足迹**：行切方案里每个 rank 必须持有完整 `W2` 才能完成最后一乘，
  每 rank 权重 = `h·f/N + h·f`，TP=2 时是列切方案的 1.5 倍，TP=4 时 2.5 倍
  （`(N+1)/2` 随 N 增长）——degree 越大越亏。

结论比 L0 更强：**「W1 列切 → W2 行切」是正确性、通信量、权重足迹三项同时最优的切法**，
由 GeLU 的位置唯一决定。attention 同构（按 head 列切 + 输出投影行切，L0 §5）。

---

## 6. 账本实测：每 rank 恰好 16P/N

每个 rank 上真实创建 Adam、走一步 `step()` 物化优化器状态，然后逐字节清点：

```text
rank 0: params=64.0 KiB  grads=64.0 KiB  optimizer=128.0 KiB  total=256.0 KiB   (TP=2)
dense 完整副本 = 16 x 32,768 = 512.0 KiB | TP 各 rank 之和 = 512.0 KiB
```

- 每 rank = `16P/N`：params `4P/N` + grads `4P/N` + Adam m/v `8P/N`，与 L0 [5]
  的算术逐位吻合；TP=4 时减半到 `128 KiB`，P/N 随 degree 线性下降实测成立。
- **守恒**：各 rank 之和 = dense 完整副本。这和 [nano-fsdp L1](../nano-fsdp/tutorial_L1.md)
  在 FSDP 上测到的是同一条守恒律：**并行不减少集群总内存，只把同一份 16P 切开**。
- 差别在切的东西：FSDP 切「状态」，算每层时仍要 all-gather 整层权重；
  TP 切的是权重与计算本身，前向不需要拼回任何完整权重——
  代价就是本节量出来的 all-reduce。两者互补而非竞争，这是「机器内 TP + 跨机 ZeRO」
  组合的账本根源。

---

## 7. 计时：all-reduce 在关键路径上，有多贵

gloo loopback（真实 TCP，本机 CPU）上三个消息大小的实测墙钟
（warmup 2 次后取 5 次均值，跨 rank 取 max；连跑 3 遍的波动区间）：

| 消息大小 | run1 | 3 遍区间 | 折算带宽（run1 算术） |
|---|---|---|---|
| 1 MiB | 2.49 ms | 2.35–2.49 ms | —（延迟主导） |
| 16 MiB | 23.69 ms | 23.59–23.95 ms | ≈ 708 MB/s |
| 32 MiB | 46.51 ms | 46.04–47.16 ms | ≈ 721 MB/s |

三个读数各说一件事：

1. **延迟地板**：1 MiB 也要 2.4 ms。消息放大 32 倍，时间只放大 ~19 倍——
   固定延迟占比显著。而 toy 形状下每 rank 前向计算仅 `0.038 ms`：
   通信延迟地板已是计算的 ≈65 倍。L0 §11「batch 小 / 序列短时通信占比升高」
   在这里有了真实数字——toy 形状完全被通信压死。
2. **16 MiB 正是 L0 账本里「一次 all-reduce 的消息量」**（h=4096, seq=2048, fp16）。
   在本机这条消息要 ~24 ms。按 L0 的核算（每层 4 次 × 32 层 = 128 次/microbatch）
   纯算术外推：本机口径下每 microbatch 纯通信 ≈ 128 × 23.7 ms ≈ 3.0 s。
   这是本机 CPU/loopback 的基线数字，用来说明量级关系；单机 L20/NCCL 的实测见 §7.1。
3. **fp16 探测**：本机 torch 2.4.1 + gloo **支持** fp16 all_reduce 且数值正确
   （`1.0 + 1.0 = 2.0` 实测）。L1 仍用 fp32，为的是舍入分析干净；
   生产混合精度训练是 NCCL/GPU 的事，行为不互相推定。

「TP 只在机器内用」这笔账（L0 §6）现在有了实测注脚：all-reduce 既躲不掉
（每层 4 次，全在关键路径上），又贵（延迟地板 + 带宽需求），只能靠机器内
高带宽互连把它压到计算能容忍的水平。

### 7.1 单机 L20：TP2 → TP4 → TP8 时，省下什么、付出什么？

2026-09-03 在同一台 8×NVIDIA L20 上，把 [GPU probe](L1_gpu_verify.py) 分别设为
`--world-size 2/4/8`，每个配置独立运行两次。软件栈为 driver 550.90.07、
PyTorch 2.9.1+cu128、CUDA 12.8、NCCL 2.27.5；六次均 exit 0、stderr 0、7/7。
每次只暴露参与实验的 GPU，避免把“机器有 8 卡”和“本次用了几卡”混为一谈。

| TP | 每 rank 训练状态 | 前向相对误差 | FP32 16 MiB all-reduce | ring-normalized busbw | 两次共同 digest |
|---:|---:|---:|---:|---:|---|
| 2 | 256 KiB | `1.9e-7` | 0.784–0.785 ms | 21.4 GB/s | `be3aa882721b9e67cf7279097286271f` |
| 4 | 128 KiB | `2.1e-7` | 1.874–1.875 ms | 13.4 GB/s | `e3742f02a539ecca9c1b9d155519929e` |
| 8 | 64 KiB | `1.5e-7` | 2.067–2.069 ms | 14.2 GB/s | `9b9a41d443b5a89d49036ccfe17ede55` |

这里的 `busbw` 沿用 NCCL-tests 的 ring all-reduce 归一化：

$$
\text{algbw}=\frac{B}{t},\qquad
\text{busbw}=\text{algbw}\cdot\frac{2(N-1)}{N}.
$$

这张表支持三个结论：

1. **内存收益是精确的**：每 rank 状态按 $1/N$ 从 256 → 128 → 64 KiB，所有 rank
   相加始终是 512 KiB；TP 切分没有凭空消灭全局状态。
2. **数值语义保持**：三种规模均与 dense 达到 `1.5e-7–2.1e-7` 相对误差；错误的
   GeLU-before-reduce 反例仍显著失败，正确顺序仍恢复。
3. **通信不是免费扩容**：固定 16 MiB 消息从 TP2 到 TP4/TP8 分别约变为 2.39×/2.64×
   延迟。拓扑中 GPU0–1 为 `PIX`，GPU0–3 扩到同 NUMA 的 `NODE`，8 卡还跨 NUMA `SYS`；
   带宽变化与拓扑层级变化同时出现，但本实验不把相关性写成单一因果归因。

这不是端到端训练 speedup：MLP 仍是 `[128,64]×[64,256]` 的微小 toy，表中 16 MiB
是独立 collective probe，且没有 attention、通信计算重叠或多节点。它回答的是
“状态按 $1/N$ 下降时，collective 代价如何变化”，不是“8 卡训练一定比 2 卡快多少”。

---

## 8. 与真实 Megatron 的对应（行号均为 2026-08-05 main 分支实测）

| nano-megatron L1 | Megatron-LM / PyTorch 对应 | 说明 |
|---|---|---|
| `CopyToTensorParallelRegion`（f） | `megatron/core/tensor_parallel/mappings.py:L201` `_CopyToModelParallelRegion`：fwd identity（L210-213）/ bwd `_reduce`（L218） | 语义逐条一致 |
| `ReduceFromTensorParallelRegion`（g） | 同文件 `:L221` `_ReduceFromModelParallelRegion`：fwd `_reduce`（L232）/ bwd identity（L236） | 同上 |
| 算子内部的 `dist.all_reduce` | `_reduce`（mappings.py:L22，调用点 L35） | 一句 `torch.distributed.all_reduce` |
| f 插在列并行前向入口 | `layers.py:L869` `ColumnParallelLinear`，前向调用 f 于 `layers.py:L1148` | 包装函数 `copy_to_tensor_model_parallel_region`（mappings.py:L492） |
| g 插在行并行前向出口 | `layers.py:L1250` `RowParallelLinear`，前向调用 g 于 `layers.py:L1482` | 包装函数 `reduce_from_tensor_model_parallel_region`（mappings.py:L498）；L1478 是序列并行分支的 reduce-scatter，L3 展开 |
| 「每块 fwd 1 + bwd 1」计数 | Megatron 论文通信分析（arXiv:1909.08053）：每层 fwd 2 + bwd 2 | L1 用计数器把「1」做实 |

**nano 与权威实现的差异（为什么它那样选）**：

1. Megatron 的 `ColumnParallelLinear` / `RowParallelLinear` 还带
   `gather_output` / `input_is_parallel` / 序列并行分支（layers.py:L377 的
   reduce-scatter 路径）等开关，服务 TP×SP 组合与块间拼接；nano 只保留主干，
   因为 L1 的目标是把 f/g 的通信语义做实，组合调度是 L2/L3 的事。
2. 真实实现有通信与计算的重叠（异步 all-reduce / `gradient_accumulation_fusion`
   把 dW 的 GEMM 与归约融合等优化）；nano 全部同步执行——先看清楚「通信在关键路径上」
   这件事本身，再看怎么把它藏起来（后者属 L2+ 的性能话题）。
3. PyTorch 侧全部使用稳定公开 API：`torch.distributed.all_reduce`
   （<https://docs.pytorch.org/docs/stable/distributed.html>）与
   `torch.autograd.Function`（<https://docs.pytorch.org/docs/stable/notes/extending.html>），
   无版本性 hack。

---

## 9. 费曼：讲给外行听

**类比：分科阅卷 + 电话汇总，正向反向各打一次电话。**

L0 的类比是阅卷分工：各科老师（rank）独立算自己那科的部分和，班主任（all-reduce）
汇总一次出总分（前向）。L1 加上反向——**改错反馈也要打电话**：

- 总分算错了一点（loss 有误差），反馈从班主任那里**原样**传回每位老师
  （g 的反向是 identity：每位老师拿到的都是同一份完整反馈，不需要再打电话）；
- 但每位老师要把反馈转给自己负责的那部分答题过程，算出的「题目本身该怎么改」
  （dX）只是自己这科的版本——最后必须**再打一次电话**把所有科目的意见加起来
  （f 的反向 all-reduce），才能得到对题目的完整修改意见；
- 至于老师自己手里的评分标准怎么调（dW1/dW2），每人本地就能算全，不用打电话。

**一句话版**：TP 的通信长在「复制 ⇄ 求和」的转换点上——前向哪里求和，
反向就在对称的地方聚合，各一次；权重怎么调，本地说了算。

**反例版**：若让两位老师分摊同一科（W1 行切），标准化（GeLU）在分数合并前就做不了——
强行先做，总分错出 168 分（naive err = 1.684e+02）；合并了再做，电话费（消息量）
又要多付 4 倍。分工方式不是任意的，它被「哪一步必须看全量」唯一决定。

---

## 10. 思考题

1. 为什么 g 的反向是 identity 而 f 的反向是 all-reduce，不能反过来？
   （提示：前向里哪个张量是「各 rank 相同」的，哪个是「各 rank 部分和」的；
   反向梯度的复制/求和结构恰好镜像。）
2. dW1_r / dW2_r 为什么零通信就是完整的？如果换成**各 rank 复制**的参数
   （如 LayerNorm 的 γ/β、embedding），它们的梯度需要什么额外操作才能保持一致？
   （提示：复制 ⇒ 各 rank 梯度只是部分视角，需要聚合。Megatron 具体在哪一步做，
   L3 对照源码验证 `[TODO: verify]`。）
3. 用 §7.1 的 TP2/4/8 表分别计算“每 rank 状态缩小倍数”和“16 MiB collective
   延迟放大倍数”。为什么前者不能直接推出端到端 speedup？还缺哪些 GEMM、attention、
   overlap 与拓扑证据？
4. TP 与 FSDP（nano-fsdp L1）都把每 rank 状态压到 ≈ 16P/N，且都满足「各 rank 之和 = 16P」
   的守恒。两者的通信分别花在什么上（激活级 vs 权重级）？为什么大规模训练要
   机器内 TP + 跨机 FSDP/ZeRO 组合，而不是二选一？

---

## 11. 边界与局限

- **CPU 与 GPU 数字不可混用**：§7 来自 CPU + gloo + loopback，§7.1 来自固定的
  L20/NCCL 单机栈；两者都只代表各自环境，不代表其它互连或生产 workload。
- **toy 形状**：`[128, 64] × [64, 256]`，计算微秒级，通信延迟完全主导；
  真实形状下计算/通信比会变化，但「每块 fwd 1 + bwd 1」的结构不变。
- **未含 attention 块与 bias/LayerNorm**：attention 的 TP 是同构切法（L0 §5），
  bias 与 norm 参数属「复制参数」，其梯度聚合议题见思考题 2。
- PP（按层切 + micro-batch + bubble）是 L2，SP（序列维再切）与 MFU 分析是 L3。

---

## 12. 溯源

- 运行环境：`python`，torch 2.4.1，
  gloo backend，CPU（15 核机器，每进程 `torch.set_num_threads(4)` 防争抢），seed=7。
- **输出保真**：§2 粘贴为 run1 实跑输出；连跑 3 遍，除计时两节外逐字节一致
  （mask 计时行后 md5 三遍相同：`e7f279a1ff29d40c30161575002b2700`）；
  计时行 3 遍区间如实列于 §7 表。
- **Megatron-LM 源码锚点**（NVIDIA/Megatron-LM main 分支，2026-08-05 现场抓取核验，
  与 L0 于 2026-08-04 实测的行号一致、无漂移）：`mappings.py:L201/L221`（f/g 类）、
  `L22/L35`（`_reduce` → `dist.all_reduce`）、`L492/L498`（包装函数）、
  `layers.py:L869/L1250`（`ColumnParallelLinear`/`RowParallelLinear`）、
  `L1148/L1482`（f/g 前向调用点）。仓库：<https://github.com/NVIDIA/Megatron-LM>。
- **论文**：Megatron-LM arXiv:1909.08053（列/行并行、f/g 算子、每层 fwd 2 + bwd 2 核算）。
- **巧合声明**：§4 备注的 TP=2/TP=4 dX 误差相等为 seed=7 巧合，
  seed=99 复跑实测分离（`1.373e-04` / `1.984e-04`）。
- fp16 支持结论为本机实测（torch 2.4.1 + gloo），不推广到其它后端/版本。
- **GPU 补充证据**：§7.1 的脚本 SHA256 为
  `d093ef1d951d343c4753d246444ca19b571739f97859aa9adc5d775d5032b8ec`；TP2/4/8
  各两遍，稳定指标与 digest 一致，计时只按两次区间报告。未执行多机实测。
