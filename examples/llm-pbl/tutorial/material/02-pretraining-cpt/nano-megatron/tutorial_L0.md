# nano-megatron · L0 教程：张量并行——怎么切、在哪通信、为什么这样切

> **本节目标（L0）**：用 ~200 行纯 Python 抓住 Megatron 张量并行（TP）的核心机制——
> 把一个 MLP 真正切开、分算、再拼回来，回答三个问题：
> ① 切法由什么唯一决定？② 通信插在哪里、量有多大？③ 为什么 TP 只在机器内用？
> **前置**：无硬前置；读过 [nano-fsdp L0](../nano-fsdp/tutorial_L0.md)（显存账本）会更好。
> **本节 K+1**：从「模型并行 = 把模型分开放」到「切分必须数值等价，通信位置由非线性决定」。

---

## 1. 问题：ZeRO 省了显存，为什么还不够？

nano-fsdp L0 讲过：Adam + fp16 训练要 16 bytes/param，ZeRO-3 把这份状态切到每卡 16P/N。
看起来问题解决了？还差一层：

- **ZeRO/FSDP 切的是「状态」**（参数副本、梯度、优化器状态）。但每次前向/反向算某一层时，
  仍要把该层**完整权重** all-gather 回来，算完再丢——计算本身仍是「每卡算整层」。
- 当一层本身就很大（隐藏维 h 上万），或者单卡算力喂不饱时，就需要把**计算本身**切开：
  每卡只持有 1/N 的权重、只做 1/N 的乘加，然后用通信把部分结果拼起来。

这就是张量并行（tensor parallelism，TP）：切分对象从「数据/状态」进入了**算子内部**。
Megatron-LM（arXiv:1909.08053）给出了 Transformer 上最经典的切法，本节在最小 MLP 上把它做出来。

---

## 2. 先跑起来

文件：`L0_tp_mlp.py`，纯标准库（连 numpy 都不需要），CPU 即跑。

```bash
$ python3 L0_tp_mlp.py
```

真实输出（seed=42）：

```text
================================================================
nano-megatron L0 — tensor parallel MLP: 怎么切，在哪通信
================================================================

toy shape: X[3x4]  W1[4x8]  W2[8x4]  (GeLU MLP)

[1] dense 参照: Y_ref[0] = [-0.076672, -0.522249, -0.811754, 0.726474]

[2] Megatron 式 TP (2 ranks): W1 列切 + GeLU + W2 行切 + 1 次 all-reduce
    max|Y_tp - Y_ref| = 1.110e-16   ✅ 数值严格一致

[3] 反例：W1 按行切（切输入维）
    naive（对部分和直接 GeLU）: max|err| = 0.448225  ✗ 错得离谱
    fixed（GeLU 前先 all-reduce）: max|err| = 2.220e-16  ✅ 又对了，但多一次通信
    => 「W1 列切 → W2 行切」是前向只需 1 次 all-reduce 的切法

[4] 通信账本（h=4096, seq=2048, batch=1, layers=32, fp16）
    一次 all-reduce 量 [b*s, h]          =     16.0 MiB
    TP 每层 fwd+bwd（4 次 all-reduce）   =     64.0 MiB
    TP 每 microbatch（32 层）           =   2048.0 MiB ≈ 2.0 GiB
    对照 PP（8 stages）：边界点对点     =    224.0 MiB
    ring all-reduce 每卡收发量：
      TP= 2:   16.0 MiB (= 1.00 × 消息量)
      TP= 4:   24.0 MiB (= 1.50 × 消息量)
      TP= 8:   28.0 MiB (= 1.75 × 消息量)
    => 加卡不省流量（趋近 2× 消息量）→ TP 靠机器内 NVLink；PP 流量小 → 跨机器

[5] 显存账本（h=4096, layers=32 → 总参数 ≈ 6.44e+09，约 6.4B）
    TP=1: 每卡 params = 6.442e+09 → fp16 权重 12.00 GiB
    TP=2: 每卡 params = 3.221e+09 → fp16 权重  6.00 GiB
    TP=4: 每卡 params = 1.611e+09 → fp16 权重  3.00 GiB
    TP=8: 每卡 params = 8.053e+08 → fp16 权重  1.50 GiB
    结合 nano-fsdp L0：Adam+fp16 训练状态 16 bytes/param → TP 下每卡 16P/N

================================================================
✅ self-check passed: TP 数值等价 / 反例显著出错且可修复 / 显存线性切分
================================================================

takeaway: TP 把「计算」本身切进各卡，前向每块只需 1 次 all-reduce；
          切法（W1 列 → W2 行）由 GeLU 的逐元素性唯一决定；通信量不随卡数下降，所以 TP 只在机器内用，跨机器交给 PP（L2）。
```

**L0 基线指标（toy metric）**：TP 与 dense 的最大误差 `1.110e-16`（浮点舍入级，见 §3 末尾说明）；
反例误差 `0.448225`（显著错误）。后续级别的对照基线：L1 会把这个误差搬到真实通信上复验。

盯住输出里的三个数字往下看：**1 次 all-reduce**、**16 MiB**、**P/N**。

---

## 3. 机制一：W1 列切——为什么不需要通信

toy MLP：`Y = GeLU(X @ W1) @ W2`，形状 `X[t,h] · W1[h,f] · W2[f,h]`。

把 `W1` 沿**输出维**（列）切成 N 块：`W1 = [W1_1 | W1_2 | ... | W1_N]`，每块 `[h, f/N]`。
rank r 持有 `W1_r` 和完整的 `X`（输入在 TP 里是各卡复制的），独立计算：

```python
# 列并行：rank r 持有 W1 的第 r 列块 [h, f/N]
W1_r = [row[r * shard:(r + 1) * shard] for row in W1]
# GeLU 逐元素：半份 pre-activation 可独立过非线性，无需通信
H_r = gelu_mat(matmul(X, W1_r))
```

这一步成立靠两个性质：

1. **矩阵乘对输出维可分块**：`X @ [W1_1 | W1_2] = [X@W1_1 | X@W1_2]`——
   每个输出列只依赖 `X` 和 `W1` 的对应列块，与其它列块无关。
2. **GeLU 是逐元素函数**：`GeLU([A | B]) = [GeLU(A) | GeLU(B)]`——
   非线性作用在每个元素上，不跨元素混合，所以「先切再过非线性」和「先过非线性再切」等价。

两条合起来：每个 rank 拿着 1/N 的权重就能独立算出 1/N 的隐层激活，**中间零通信**。

> 顺带解释输出里的 `1.110e-16`：TP 与 dense 在**代数上严格相等**，但浮点加法的结合顺序不同
> （先加谁后加谁），舍入误差在 1e-16 量级冒出来。这是「数值等价」的正常形态，不是切错了——
> 判断切分正确性的合理阈值是舍入级（代码里 assert < 1e-9），而不是要求逐 bit 相同。

---

## 4. 机制二：W2 行切——部分和 + 唯一一次 all-reduce

隐层 `H = [H_1 | H_2 | ... | H_N]` 已经沿列切好了（每块 `[t, f/N]`）。
现在要乘 `W2[f, h]`。把 `W2` 沿**输入维**（行）切成与 `H` 对齐的块：

```python
# 行并行：rank r 持有 W2 的第 r 行块 [f/N, h]（与自己的 H_r 列数对齐）
W2_r = W2[r * shard:(r + 1) * shard]
partials.append(matmul(H_r, W2_r))   # 部分和 [t, h]
...
return mat_sum(partials)             # ← 前向唯一一次 all-reduce
```

为什么能这样拼？分块矩阵乘法：

```text
Y = H @ W2 = [H_1 | H_2] @ [W2_1; W2_2] = H_1 @ W2_1 + H_2 @ W2_2
```

每个 rank 算出的 `H_r @ W2_r` 是**完整输出的一个部分和**（形状都是 `[t, h]`），
最后对 N 个部分和做一次 all-reduce(sum) 就得到与 dense 完全一致的 `Y`。

**「列切 → 行切」是咬合设计**：第一个矩阵切输出维，产出的列块恰好是第二个矩阵按输入维
切块后各自需要的输入。两个线性层 + 中间一个逐元素非线性，整个块的前向**只需 1 次 all-reduce**。

---

## 5. 反例：把 W1 按行切会怎样——切法由非线性位置唯一决定

如果换个切法：把 `W1` 沿**输入维**（行）切，`X` 也对应按列切。每个 rank 得到的是
**部分的 pre-activation**：`P_r = X_r @ W1_r`，且 `P_1 + P_2 + ... = X @ W1`。

天真的做法是直接对部分和过 GeLU 再相加。代码实测（输出 [3]）：

```text
naive（对部分和直接 GeLU）: max|err| = 0.448225  ✗ 错得离谱
```

错因一句话：**非线性破坏可加性**。`GeLU(p1 + p2) ≠ GeLU(p1) + GeLU(p2)`——
GeLU 不是线性算子，不能穿过求和号。半份 pre-activation 上「独立过 GeLU」在数学上根本不成立。

修复方法只有一条路：GeLU 之前先把部分和 all-reduce 拼回完整 pre-activation：

```text
fixed（GeLU 前先 all-reduce）: max|err| = 2.220e-16  ✅ 又对了，但多一次通信
```

这就是结论：**「W1 列切 → W2 行切」不是审美选择，而是被 GeLU 的位置唯一决定的通信最优解**——
它让非线性恰好落在「不需要跨 rank 混合」的切分方向上，整个前向只剩 1 次 all-reduce。
任何其它切法要么错、要么多通信。

Attention 是同一个模式的复刻：QKV 投影按 **head 列切**（每个 rank 拿若干个完整的 head），
softmax 在每个 head 内部沿序列维归一化、**不跨 head 混合**，所以各 rank 独立算完自己的 head；
输出投影再按 head 维**行切**，部分和 all-reduce。head 机制让 attention 天然适配 TP——
这也是 Megatron 论文里 attention 与 MLP 采用同构切法的原因。

---

## 6. 通信账本：TP 为什么只在机器内用

通信量按真实规模算（代码 [4]，h=4096、seq=2048、batch=1、32 层、fp16）：

- **一次 all-reduce 的消息量 = 激活大小**：`b·s·h·dtype = 1×2048×4096×2 = 16 MiB`。
- **每层 4 次 all-reduce**：Transformer 每层有两个「列并行+行并行」块（attention 块 + MLP 块），
  每块前向 1 次、反向对称 1 次 → 每层 `4 × 16 MiB = 64 MiB`；
  「每层 fwd 2 + bwd 2」的核算出自 Megatron 论文（arXiv:1909.08053）的通信分析。
- **每 microbatch（32 层）≈ 2 GiB**——全部压在每一层的计算关键路径上。

加卡能摊薄吗？看 ring all-reduce 的经典结论：每张卡实际收发量 ≈ `2(N-1)/N × 消息量`：

```text
TP=2: 16.0 MiB (= 1.00 ×)    TP=4: 24.0 MiB (= 1.50 ×)    TP=8: 28.0 MiB (= 1.75 ×)
```

**加卡把计算砍半，流量却几乎不降**（N→∞ 时趋近 2× 消息量）。这意味着 TP 的通信瓶颈无法
靠堆卡缓解，只能靠带宽解决——所以 TP 只部署在机器内（NVLink 级别的互连），
跨机器交给流量小一个量级的 PP（8 stages 边界点对点仅 224 MiB/microbatch）和 DP。
**「TP 进机器、PP/DP 跨机器」的 3D 并行标准拓扑，就是从这笔账里推出来的。**

---

## 7. 显存账本：参数随 N 线性下降

Transformer 每层参数 ≈ `12h²`（attention QKV+输出投影 `4h²` + MLP 两个 `h×4h` 矩阵 `8h²`，
不含 embedding/LayerNorm）。`h=4096, L=32` → `12 × 4096² × 32 ≈ 6.44e9`，约 6.4B。

TP=N 时每卡只存 `P/N` 参数（代码 [5]）：TP=8 时 fp16 权重从 12 GiB 降到 1.5 GiB。
结合 nano-fsdp L0 的账本——Adam+fp16 训练状态 16 bytes/param——TP 下每卡训练状态就是 `16P/N`。

和 ZeRO-3 对照着看（两者每卡都近似 P/N 级别的参数负担）：

| | ZeRO-3 / FSDP | TP |
|---|---|---|
| 切什么 | 状态（参数副本/梯度/优化器） | 计算本身（权重+乘加） |
| 每层计算时 | all-gather 整层权重，算完丢弃 | 不需要整层权重，各算 1/N |
| 通信形态 | 权重级（每层 param 大小） | 激活级（每层 `4×b·s·h`） |
| 典型位置 | 跨机器 | 机器内 |

两者不是竞争关系，而是切在不同维度上的互补手段——真实大模型训练同时用（3D 并行）。

---

## 8. 与真实 Megatron 的对应（概念层 + 已验证入口）

| nano 实现 | Megatron 对应 | 说明 |
|-----------|--------------|------|
| `mlp_tp` 的 W1 列切 | `ColumnParallelLinear`（`megatron/core/tensor_parallel/layers.py:L869`） | NVIDIA/Megatron-LM main 分支，2026-08-04 实测行号 |
| `mlp_tp` 的 W2 行切 + `mat_sum` | `RowParallelLinear`（同文件 `:L1250`，前向在输出处做 all-reduce） | 同上，2026-08-04 实测 |
| 前向 all-reduce / 反向 identity | g 算子 `reduce_from_tensor_model_parallel_region`：RowParallelLinear 前向调用点 `layers.py:L1482`；定义在 `mappings.py`（`_ReduceFromModelParallelRegion:L221`） | f/g 双算子设计见论文 §3 |
| 前向 identity / 反向 all-reduce | f 算子 `copy_to_tensor_model_parallel_region`：ColumnParallelLinear 前向调用点 `layers.py:L1148`；定义在 `mappings.py`（`_CopyToModelParallelRegion:L201`） | 反向的梯度聚合就藏在这里 |
| 每层 4 次 all-reduce 的核算 | 论文通信分析（fwd 2 + bwd 2） | arXiv:1909.08053 |
| attention 按 head 列切 | 同论文 §3 的 attention 并行化 | head 数需被 TP degree 整除 |

L0 只到机制层。源码级的细节——梯度如何沿 f/g 算子流动、TP 与 PP 组合时的调度、
序列并行 SP 为何要再切序列维——留给 L2/L3 对照源码补齐。

---

## 9. 费曼：讲给外行听

**类比：分科阅卷 + 加权汇总总分。**

要给全校学生算加权总分。最笨的办法是一个人算所有科目（dense）。分工的办法是：

- **列切（每人负责一科）**：数学老师拿**所有学生**的数学成绩（输入 X 人手一份完整副本），
  独立做「单科标准化」（GeLU——逐学生逐分数处理，不需要看别人的卷子）；
- **行切（每人只握自己那科的权重）**：数学老师把标准化分数乘以「数学的权重占比」（W2 的一块），
  得到每个学生总分里的**数学部分和**；
- **all-reduce（班主任汇总）**：各科老师把部分和报给班主任，一次加总，得到最终总分——
  这就是唯一的通信点。

反例版：如果让两位老师**分摊同一科的卷子**（W1 按行切），「标准化」就做不下去了——
划线/等级换算依赖完整的一科分数，半份分数算不出正确的等级
（`GeLU(p1+p2) ≠ GeLU(p1)+GeLU(p2)`）。除非两人先把分数合并复原（多一次通信）再处理。
所以分工方式不是随便定的：**哪一步需要看全量，通信就必须插在那一步之前**。

---

## 10. 思考题

1. TP 下哪些张量是**每卡完整复制**的？（提示：输入 `X`、输出 `Y`，以及 transformer 里
   LayerNorm/Dropout 的激活——它们不在任何切分块内部。）这会引出什么浪费？
   （这正是序列并行 SP 的动机：把 LayerNorm/Dropout 的激活沿序列维再切开，L3 展开。）

2. 为什么实践中 TP degree 通常不超过 8、且几乎总在单机内？把 TP=16 强行跨两台机器会发生什么？
   （提示：§6 的流量饱和 + 每层 all-reduce 都在关键路径上 + 跨机带宽断崖；
   另外 head 数必须能被 degree 整除，代码里的 `assert f % n_ranks == 0` 就是这件事的 toy 版。）

3. TP 和 ZeRO-3 都能把每卡参数负担压到 ~P/N，为什么大规模训练还要**同时**用两者
   （如机器内 TP=8、跨机 ZeRO/FSDP）？各自的通信付在了什么不同的东西上？

---

## 11. 反例：TP degree 不是免费的

> 「TP 开得越大，训练越快。」

错。三层原因：① 每层 all-reduce 都在计算关键路径上，batch 小 / 序列短时，计算量缩了，
通信延迟却不会同比例缩，占比反而升高；② 流量 `2(N-1)/N × 消息量` 随 N 饱和，
加卡摊不薄通信，只能摊计算——收益递减来得很快；③ degree 受 head 数与整除约束，
不是想开几就开几。**TP 是「用机器内高带宽通信换单卡装不下/算不完」的交易，
不是免费午餐**；跨机器做 TP 更是直接亏本（见 §6 的 PP 对照）。

---

## 12. 下一步 L1

L1 把这个 toy 搬上**真实通信**：用 `torch.distributed`（gloo backend，CPU 多进程即可）
起 2 个 rank 跑列/行并行 MLP，复验数值等价，并实测一次 all-reduce 的真实耗时，
和本节账本里的 16 MiB 对照——从「模拟 all-reduce 的语义」走到「付出 all-reduce 的代价」。
L2 再加流水线并行（按层切 + micro-batch + bubble），L3 加序列并行并对照 Megatron 源码。

---

## 13. 溯源

- 运行输出来自本机真实执行：`python3 L0_tp_mlp.py`（seed=42，
  纯确定性计算，输出可逐位复现）。通信/显存账本数字全部由代码现场算出，公式在 §6/§7 给出，可手算复核。
- Megatron-LM 论文：arXiv:1909.08053（Shoeybi et al.，2019-09-17 提交，已在线核验标题与作者）；
  「每层 fwd 2 + bwd 2 次 all-reduce」、列/行并行切法、attention 按 head 切均出自该论文。
- ring all-reduce 每卡收发量 `2(N-1)/N × 消息量` 为经典结论（reduce-scatter `(N-1)/N` +
  all-gather `(N-1)/N`），代码 `ring_bytes_per_gpu` 即此式。
- 每层 `12h²` 参数：QKV 投影 `3h²` + 输出投影 `h²` + MLP `2 × 4h²`（不含 bias/embedding/LayerNorm，
  是估算口径，代码 [5] 已注明「约 6.4B」）。
- Megatron-LM 源码入口于 2026-08-04 在 NVIDIA/Megatron-LM main 分支在线实测：
  `megatron/core/tensor_parallel/layers.py` 的 `ColumnParallelLinear:L869` /
  `RowParallelLinear:L1250`，f 算子调用点 `layers.py:L1148`、g 算子调用点 `layers.py:L1482`；
  f/g 算子定义在 `megatron/core/tensor_parallel/mappings.py`
  （`_CopyToModelParallelRegion:L201` / `_ReduceFromModelParallelRegion:L221`）。
  行号为当日 main 分支快照，上游迭代后可能漂移；源码级细读留 L3。
- 仓库：<https://github.com/NVIDIA/Megatron-LM>。
