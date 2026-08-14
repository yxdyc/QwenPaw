# nano-megatron · L3 教程：序列并行——把 all-reduce 拆成两半，中间插上切碎的 LayerNorm

> **本节目标（L3）**：L1 切了层内参数（TP），L2 切了层间深度（PP），
> 但 TP 有个盲区：MLP/Attention **内部**的激活被切了（`[T, FF/t]`），
> 而 LayerNorm / Dropout / 残差这些「TP 未切区域」的激活在每个 rank 上
> 都是**完整副本**——t 个 rank 存 t 份一模一样的 `[T, H]`。
> 序列并行（Sequence Parallelism, SP，arXiv:2205.05198）把这些区域沿
> **序列维**切碎：进 MLP 前 all-gather 拼回、出 MLP 后 reduce-scatter 散回。
> 本节做出 Megatron 的真实组合 **TP × PP × SP**（4 个真实进程），
> 并用 **MFU** 把整套并行化「值不值」量化成一个可溯源的数字。
> **前置**：[nano-megatron L1](tutorial_L1.md)（f/g 算子与 all-reduce 账本）、
> [nano-megatron L2](tutorial_L2.md)（PP 调度与 batched p2p）。
> **本节 K+1**：从「切参数 / 切深度」到「切序列」——
> AR ≡ RS+AG 恒等式、重放 gather 的代价、γ/β 梯度的部分和归约、
> PP 接缝在 SP 下减半、以及 MFU 的三段分解。

---

## 1. TP 的盲区：没切到的地方，显存照样翻倍

回顾 L1 的 TP：一个 GeLU MLP，W1 列切、W2 行切，激活变成 `[T, FF/t]`——
MLP **内部**的显存除以了 t。但一个真实 Transformer 块不只有 MLP：

```
x ──▶ LayerNorm ──▶ MLP(TP 切) ──▶ Dropout ──▶ + ──▶ 下一层
      [T, H] 全复制          [T, FF/t] 已切     [T] 全复制
```

LayerNorm 逐 token 归一化（统计量只在 H 维上算），Dropout 逐位置丢弃，
残差逐元素相加——它们都**不涉及跨 token 的混合**，所以 TP 没有理由去切它们；
但后果是：这些区域的激活在每个 TP rank 上都是完整的 `[T, H]` 副本。
t=8 时，8 个 rank 存 8 份一模一样的 LayerNorm 激活——
在长序列大 batch 下，这部分「复制激活」是激活显存的大头。

SP 的洞察（arXiv:2205.05198 §4.2.2 Sequence Parallelism）很简单：**这些区域可以沿序列维切**。
LayerNorm 逐 token 独立，切了照样对；Dropout 的掩码本来就是逐位置的；
残差更是逐元素。于是：

```
非 SP:  x[T,H] 全复制 ──LN──▶ MLP(TP) ──▶ 全复制 ──▶ ...
SP:     x[T/t,H] 分片 ──LN──▶ AG ──▶ MLP(TP) ──▶ RS ──▶ 分片 ──▶ ...
```

`AG` = all-gather（进 MLP 前把序列拼回全量），`RS` = reduce-scatter
（出 MLP 后把行并行的部分和**顺便**散回序列分片）。
关键恒等式：

```
all-reduce ≡ reduce-scatter + all-gather
```

非 SP 的 TP 在 MLP 出口做一次 all-reduce（g 算子）；
SP 把它拆成两半——出口的 RS 加上（下一块）入口的 AG——
**拆开后，中间正好能插上序列维切分的 LN/Dropout 区域**。
这不是近似，是恒等式：本节 [0] 实测 SP 与 TP 的 dW 分片 **bit 级相同**。

---

## 2. 先跑起来

文件：`L3_sp_tp_pp_mfu.py`，依赖仅 `torch`（CPU 即可，gloo 4 进程）。

```bash
$ python3 L3_sp_tp_pp_mfu.py
```

真实输出（seed=7，本次运行；除计时行外所有行在 2 个独立 CWD 连跑逐字节一致——
计时行 = 计时导出行，共 9 行：`elapsed[` 行 4 + [5] 解读行 1（含实测 wall 比值）+
含 `MFU sanity` 的 self-check 行 2 + 含 `MFU SP/非SP` 的 self-check 行 1 +
total wall 行，掩码口径见 §14；digest `51b674545d3295004557e525ab161842`
多遍相同）：

```text
========================================================================
nano-megatron L3 — sequence parallelism: TP×PP×SP combined + MFU
========================================================================
model: 4 x (LN + GeLU-MLP) blocks (H=128, FF=512, p_drop=0.1) | P = 525,312 | fp32 | seed=7
cluster: 4 ranks = PP2 x TP2 (gloo, CPU; SP 与 TP 同组同 degree) | T = 512
SP primitives: all-gather / reduce-scatter (Megatron mappings.py style); AR ≡ RS+AG

[0] SP mechanism (single LN+MLP block, TP=2): 等价性
    dW1/dW2 shard: SP 与 TP 各自 vs dense 参照 max|Δ| = 0.0e+00 (bit 级)
    dγ/dβ: max|Δ| = 6.1e-05 (相对 1.7e-07, fp32 归约顺序差)
    dX: SP 分片 vs TP 全序列对应切片 max|Δ| = 0.0e+00
    SP γ/β 梯度 = 序列分片部分和: all-reduce 前 vs dense Δ = 3.574e+02 ❌, all-reduce 后 Δ = 6.1e-05 ✅
    结论: all-reduce ≡ reduce-scatter + all-gather 不只是通信恒等式——
          拆开后中间能插序列维切分的 LN/Dropout，数学不变

[1] communication: 分解中性，重放加价
    TP/block: 1 AR fwd + 1 AR bwd = wire 2m = 524288 B
    SP/block: fwd(AG+RS) + bwd(AG+RS) + 1 重放AG(wgrad) = wire 2.5m = 655360 B = TP×1.25
    重放 = 不存 gathered 输入的代价（layers.py:L609-618）——存了就把省下的显存吐回去

[2] activation ledger: SP 的收益只在 TP 未切区域
    区域激活/block/rank (LN xhat+rstd+输出+掩码): TP = 528384 B → SP = 264192 B (= 1/2)
    TP 已切激活/block/rank (gelu_in+a, [T, FF/t]): 1048576 B, 两者相同

[3] dropout 掩码必须随序列切（反例）
    非 SP 误用 per-rank forked 掩码: dW1 Δ = 3.378e+01 ❌ (复制流分叉)
    正确做法: 掩码是位置数据——全序列生成、按分片切片 (SP 区域 fork RNG, transformer_block.py:L595)

[4] combined: PP2 × TP2 = 4 ranks, GPipe m=2, 一步 Adam
    步后权重: SP vs 非SP max|Δ| = 2.3e-10; vs dense 镜像参照: 非SP 6.1e-07 / SP 6.1e-07
    per-mb losses = ['5.841236', '5.854410']  (SP vs 非SP Δ = 0.0e+00)
    PP 接缝字节/step/rank: 非SP = 524288 B [mb,H], SP = 262144 B [mb/t,H] = 1/2
    LN γ/β 梯度: SP 下为序列分片部分和, 步前跨 TP 组 all-reduce (finalize_model_grads.py:L416)

[5] MFU (GEMM-calibrated peak, Megatron FLOPs 公式, per-rank 口径)
    model FLOPs/step = 1,610,612,736 = 3 × 4 blocks × 4·4·T·h² (training.py:L428/L548); per rank = /TP/PP = 402,653,184
    elapsed[calib]: GEMM peak = 665.2 GFLOP/s (fp32, 512³×10×3, 4 threads)
    elapsed[mfu-dense]: 9.9 ms/step → MFU(dense, 无通信) = 24.46%
    elapsed[mfu-nosp]: 15.3 ms/step → achieved = 26.32 GFLOP/s, MFU = 3.96%
    elapsed[mfu-sp]: 29.8 ms/step → achieved = 13.51 GFLOP/s, MFU = 2.03%
    解读: MFU(dense) = 计算效率上界; 分布式 MFU 再扣通信/调度; SP wall/非SP wall = 1.95× (CPU/gloo list 版 AG/RS 开销 + 集合 5 vs 2; GPU/NCCL 通信量仅 1.25×)
    CPU/gloo 绝对值低 = 通信主导; GPU/NCCL 真机 MFU 与 SP 显存收益 [TODO: verify on real system]

[6] self-check
    PASS  single-block dW1: SP & TP both vs dense ref within 1e-6 (max Δ = 0.00e+00)
    PASS  single-block dW2: SP & TP both vs dense ref within 1e-6 (max Δ = 0.00e+00)
    PASS  single-block dgamma: SP & TP both vs dense ref within rel 1e-6 (Δ = 1.91e-05, |dgamma| max = 110.9, rel = 1.72e-07; fp32 归约顺序差)
    PASS  single-block dbeta: SP & TP both vs dense ref within rel 1e-6 (Δ = 6.10e-05, |dbeta| max = 692.2, rel = 8.82e-08; fp32 归约顺序差)
    PASS  single-block dX: SP shard vs TP full-slice Δ = 0.00e+00 ≤ 1e-6
    PASS  TP comm/block: 1 AR fwd + 1 AR bwd (got ar_fwd=1, ar_bwd=1)
    PASS  SP comm/block: fwd AG+RS, bwd AG+RS + 1 重放AG(wgrad) (got ag_fwd=1, rs_fwd=1, ag_bwd=2, rs_bwd=1)
    PASS  TP wire/block = 2m = 524,288 B (got 524288)
    PASS  SP wire/block = 2.5m = 655,360 B = TP×1.25 (got 655360; AR≡RS+AG 分解中性, 重放 gather 加价)
    PASS  region activations/block: TP = 528384 B = 2 × SP 264192 B (恰 1/t)
    PASS  TP-sharded activations/block 不变: 1048576 B (SP 的收益只在未切区域)
    PASS  counterexample: 非 SP 用 per-rank forked 掩码 → dW1 Δ = 3.378e+01 (复制流分叉, 显著错)
    PASS  SP γ/β grads pre-allreduce = partial sums (Δ vs dense = 3.574e+02 > 1e-3, 本地只见过 T/2 个 token)
    PASS  combined: 步后权重 SP vs 非SP max|Δ| = 2.33e-10 ≤ 1e-6
    PASS  combined: 非SP vs dense 镜像参照 max|Δ| = 6.15e-07 < 1e-5
    PASS  combined: SP vs dense 镜像参照 max|Δ| = 6.15e-07 < 1e-5
    PASS  combined: 非SP vs true full-batch max|Δ| = 6.78e-07 < 1e-5 (fp32 归约形状差, 同 L2 [0])
    PASS  combined: per-mb losses SP vs 非SP Δ = 0.00e+00 ≤ 1e-6 (归约分组显式对齐)
    PASS  combined p2p bytes 非SP = 2(N-1)·T·H·4 = 524,288 (got 524288)
    PASS  combined p2p bytes SP = 2(N-1)·(T/t)·H·4 = 262,144 = 非SP/2 (got 262144; schedules.py:L2122 seq//tp)
    PASS  FLOPs/step = 3·L·4·exp·T·h² = 1,610,612,736 (Megatron 公式口径)
    PASS  MFU sanity (sp=False): 0 < 3.957% < 100% (CPU/gloo)
    PASS  MFU sanity (sp=True): 0 < 2.031% < 100% (CPU/gloo)
    PASS  MFU SP/非SP = 0.51 ∈ (0.25, 4.0) (CPU/gloo: list 版 AG/RS 单次开销大 + 集合次数 5 vs 2 → SP 实测更慢; GPU/NCCL 上通信量仅 1.25×, 吞吐应近似 [TODO: verify on real system])
    ✅ self-check passed (24/24)

digest(md5 of metrics) = 51b674545d3295004557e525ab161842

total wall = 3.0s
```

24/24 自检全过。下面拆开讲。

---

## 3. 等价性为什么成立：把算子画出来

非 SP 的 TP（L1 的 f/g 算子）与 SP 的算子布局，逐块对照：

```
非 SP（每块）                          SP（每块）
x [T,H] 全复制                         x [T/t,H] 分片
│                                      │
├─ f: identity (bwd: AR)               ├─ LN 在分片上算（逐 token 独立，结果逐位同）
├─ LN [T,H]                            ├─ AG: 拼回 [T,H]          ← fwd 集合 ①
├─ W1 列切 matmul → [T,FF/t]           ├─ W1 列切 matmul → [T,FF/t]
├─ GeLU                                ├─ GeLU
├─ W2 行切 matmul → [T,H] 部分和        ├─ W2 行切 matmul → [T,H] 部分和
├─ g: AR → [T,H] 全量     ← fwd 集合①  ├─ RS: 部分和→分片 [T/t,H]  ← fwd 集合 ②
├─ Dropout(全掩码)                      ├─ Dropout(掩码切片)
└─ x + ···  全复制                      └─ x_shard + ···  分片
```

反向完全对称：AG 的反向是 RS，RS 的反向是 AG；f 的反向 AR 对应
SP 里「AG 反向 + RS 反向」的组合。每一处通信都**恰好**对上，
求和的项与顺序一致（t=2 时都是两项相加），所以：

- **dW1/dW2 分片 bit 级相同**（§2 [0]：Δ = 0.0e+00）——
  权重梯度只依赖本 rank 的 matmul 输入输出，SP 的 AG 拼回的就是
  非 SP 那份全量激活（concat 是精确的）。
- **dX 分片 vs 全序列对应切片 bit 级相同**（Δ = 0.0e+00）。
- **dγ/dβ 相对差 ~1e-7**（舍入级）：γ/β 梯度是对 token 维的求和，
  SP 是「两个半区和再相加」，dense 是「一次全和」——fp32 归约顺序差，
  与 L2 [0] 的「fp32 归约形状差」同族，非错误。

一个必须显式处理的细节：**SP 下 γ/β 梯度是部分和**。
LN 的 γ/β 不被 TP 切（Megatron 同款：LN 在 TP 区域之外），
非 SP 时每个 rank 见过全部 T 个 token，本地算出的 dγ 就是完整的；
SP 时每个 rank 只见过 T/t 个 token，本地 dγ 只是**部分和**——
§2 [0] 实测 all-reduce 前与 dense 差 3.574e+02（完全不对），
跨 TP 组 all-reduce 后降到 6.1e-05（舍入级）。
Megatron 在 `finalize_model_grads` 里做这件事：带 `sequence_parallel`
属性的参数（LN 权重）梯度跨 TP 组 SUM 归约
（`finalize_model_grads.py:L416/L451-453`，§10 表）。

---

## 4. 通信账：分解中性，重放加价

SP 常被说成「通信量不变」——这话对一半。把账算到实现层面（§2 [1]）：

| | 每块集合次数 | ring 等价字节（m = T·H·4 = 262,144 B，t=2） |
|---|---|---|
| 非 SP | fwd 1 AR + bwd 1 AR = 2 | AR(m) = 2m(t-1)/t = m → 共 **2m = 524,288 B** |
| SP | fwd AG+RS，bwd AG+RS，**+ bwd 重放 AG** = 5 | AG/RS 各 m(t-1)/t = m/2 → 共 **2.5m = 655,360 B = 1.25×** |

「AR ≡ RS+AG」这个分解本身是通信中性的（2m ↔ m/2+m/2 两半对两半）；
多出来的 0.5m 是**反向重放的 all-gather**——为什么必须重放？

算 wgrad 需要 gathered 的全量输入：`dW1 = X_full.T @ dGeLU`。
前向 gathered 出的 `[T, H]` 输入有两条路：

1. **存下来**给反向用——省了通信，但省下的激活显存原样吐回去，SP 白做；
2. **反向重新 gather**（只存 `[T/t, H]` 分片输入）——多一次 AG 通信，显存收益保住。

Megatron 选了 2：融合线性算子前向把输入 gather 进全局 buffer 做 matmul
（`layers.py:L565-573`），**只保存分片输入**，反向再 all-gather 一次算 wgrad
（`layers.py:L609-618`，靠 `CUDA_DEVICE_MAX_CONNECTIONS=1` 让 gather 与
后续计算重叠）。所以准确的表述是：**SP 用 ~25% 的额外 TP 通信换未切区域
1/t 的激活显存**——这是显存-通信 tradeoff，不是免费午餐。

---

## 5. 显存账：收益只在未切区域

激活账本在「为 backward 保存的张量」的保存点实测（§2 [2]）：

| 每块每 rank | 非 SP | SP | 比值 |
|---|---|---|---|
| 区域激活（LN xhat+rstd+LN 输出+dropout 掩码） | 528,384 B | 264,192 B | **恰 1/2 = 1/t** |
| TP 已切激活（gelu_in + a，`[T, FF/t]`） | 1,048,576 B | 1,048,576 B | 1×（不变） |

两个直接推论：

1. **SP 的收益与「未切区域占比」成正比**。本节 toy 块只有 LN+Dropout，
   真实 Transformer 还有 attention 的 softmax 区域（未切，`[s, s]` 级）——
   长序列下那才是大头，SP 收益更显著（arXiv:2205.05198 §4.1/§4.3 的显存分析）。
2. **t 越大收益越大**（1/t），但 TP 的通信代价也随 t 涨——
   这就是为什么 Megatron 里 SP 与 TP **绑定同组同 degree**
   （`sequence_parallel` 只在 TP 组内生效），而不是独立维度。

---

## 6. Dropout 掩码必须随序列切（反例）

SP 区域的 dropout 有个隐蔽的坑：**掩码是位置数据，必须和激活一起切**。
本节的掩码按位置确定（专用 generator，`mask[i]` 只由位置 i 决定）：
非 SP 用全量 `[T]`，SP 用切片 `mask[r·T/t:(r+1)·T/t]`——同一份数据，切法不同。

反例（§2 [3]）：把「SP 区域各 rank 独立 RNG」的做法误搬到**非 SP**——
非 SP 的激活是全序列**复制**，两个 TP rank 必须用**同一份**掩码；
若各 rank 用独立掩码，复制流从 dropout 处分叉，后续所有激活不一致，
梯度从分叉的激活上算出来——实测 dW1 Δ = 3.378e+01（显著错）。

反过来，SP 区域**必须**各 rank 独立（每个 rank 只持有不同的序列片段，
掩码自然不同）——Megatron 在 SP 区域 fork RNG 状态
（`transformer_block.py:L595-598`：`if config.sequence_parallel:
rng_context = get_cuda_rng_tracker().fork()`）。

一句话：**复制的区域要同一份随机性，切分的区域要各管各的随机性**——
随机性的「切分语义」必须和激活的切分语义一致。

---

## 7. 组合：TP × PP × SP 一起上

Megatron 的真实形态是三维组合（再加 DP 就是 4D）。本节做
**4 rank = PP2 × TP2**，SP 与 TP 同组，一步真实 Adam（GPipe m=2，
调度沿用 L2，非本节重点）：

```
stage 0 (rank 0,1 = TP 组)          stage 1 (rank 2,3 = TP 组)
块 0,1：LN→AG→MLP切→RS→drop         块 2,3：同左
        └──── p2p: [mb/t, H] 分片 ────┘   ← SP 下接缝传的是分片形态
```

三个实测结果（§2 [4]）：

1. **正确性保持**：SP vs 非SP 步后权重 Δ = 2.3e-10，两者 vs dense 镜像参照
   6.1e-07（舍入级）——三维组合不引入新的数值问题，每一维的等价性独立成立。
2. **PP 接缝字节减半**：非 SP = 524,288 B，SP = 262,144 B = 1/2。
   原因直接：stage 输出本来就是 SP 分片形态 `[mb/t, H]`，p2p 传的就是它。
   Megatron 在 `get_tensor_shapes` 里显式把 seq 维除以 TP size
   （`schedules.py:L2122-2123`：`if config.sequence_parallel:
   effective_seq_length //= tp_group.size()`）——**SP 的显存收益顺着 PP
   接缝传导**，这是纯 TP 分析里看不到的组合效应。
3. **γ/β 梯度归约**在步前执行（§3 已讲的 finalize 机制在组合里同样生效）。

---

## 8. MFU：把「值不值」算成一个数字

并行化做了一堆切分，到底值不值？**MFU（Model FLOPs Utilization，
PaLM arXiv:2204.02311 的口径）**给出单一答案：

```
MFU = 模型有用 FLOPs / (硬件峰值 FLOPs × 墙钟时间)
```

三个可溯源性要点（本节全部实测，不引用厂商标称）：

1. **峰值用 GEMM 现场标定**（§2 [5] `elapsed[calib]`）：
   512³ fp32 matmul × 10 × 3 轮取最快，本机 ~570–665 GFLOP/s
   （4 线程，随机器浮动）。
2. **有用 FLOPs 用 Megatron 的公式**（`training.py:L391`
   `num_floating_point_operations`）：MLP 前向 = `4·expansion·tokens·h²`
   （`L428-431`，只数 GEMM，LN/Dropout 的 O(T·H) 忽略——与 Megatron 同口径），
   fwd+bwd ×3（`L548`）。本节 = 3 × 4 块 × 16·T·h² = 1,610,612,736 FLOPs/step；
   per-rank 再除以 TP·PP（每 rank 算 1/4 的模型）。
3. **三段分解**（§2 [5]，本机数字，GPU 会完全不同）：
   - **MFU(dense) ≈ 24%**：单进程无通信，计算效率上界——
     toy 形状（H=128）的 GEMM 太小，吃不满峰值；
   - **MFU(TP+PP) ≈ 4%**：再扣通信与调度——CPU/gloo 上小消息集合通信
     延迟主导（L1/L2 已反复见到）；
   - **MFU(TP+PP+SP) ≈ 2–3%**：SP 实测更慢（wall 比 1.5–2×）——
     **这是后端 artifact，不是 SP 的本质**：gloo 的 list 版 AG/RS 单次开销大，
     且 SP 集合次数 5 vs 2。GPU/NCCL 上 SP 通信量仅 1.25×，
     吞吐应近似非 SP `[TODO: verify on real system]`。

MFU 的价值正在于此：它把 bubble、通信开销、小算子低效、后端实现质量
**全部压进一个数字**——24% → 4% → 2% 的落差里，每一段都有明确的物理来源，
而不是「感觉变慢了」。真机训练里 MFU 30–50% 是常见目标区间，
低于预期时按这三段拆解排查（GPU 数字标 `[TODO: verify on real system]`，
走 ROADMAP §三 Machine B 通道）。

---

## 9. 与真实 Megatron 的对应（行号均为 2026-08-10 main 分支现场抓取核验，行号以抓取日为准）

| nano-megatron L3 | Megatron-LM 对应 | 说明 |
|---|---|---|
| `GatherFromSP`（fwd AG / bwd RS） | `mappings.py:L300` `_GatherFromSequenceParallelRegion`（fwd `_gather_along_first_dim` L118 / bwd reduce-scatter） | SP 入口算子 |
| `ReduceScatterToSP`（fwd RS / bwd AG） | `mappings.py:L355` `_ReduceScatterToSequenceParallelRegion`（fwd `_reduce_scatter_along_first_dim` L159 / bwd all-gather） | SP 出口算子，替代 g 的 AR |
| `SPColumnLinear` fwd 内 gather | `layers.py:L565-573`（融合线性 fwd：`dist_all_gather_func` 进全局 buffer 再 matmul） | Megatron 把 gather 融进线性 |
| `SPColumnLinear` bwd 重放 gather | `layers.py:L609-618`（bwd 重放 all-gather 算 wgrad，async + `CUDA_DEVICE_MAX_CONNECTIONS=1` 重叠） | 本节 +25% 通信的来源 |
| RowParallel 出口 SP 分支 | `layers.py:L1472-1478`（`sequence_parallel` → `reduce_scatter_to_sequence_parallel_region`，否则 all-reduce） | RS 替代 AR 的决策点 |
| LN 在分片上算、γ/β 不被 TP 切 | LN 逐 token 独立；γ/β 梯度带 `sequence_parallel` 属性 | 机制前提 |
| γ/β 梯度步前跨 TP 组 all-reduce | `finalize_model_grads.py:L416/L451-453`（`_allreduce_non_tensor_model_parallel_grads`，SUM） | nano 版 finalize |
| SP 区域 fork RNG | `transformer_block.py:L595-598`（`get_cuda_rng_tracker().fork()`） | dropout 的切分语义 |
| PP 接缝传 `[mb/t, H]` | `schedules.py:L2097/L2122-2123` `get_tensor_shapes`（SP → `seq // tp_size`），without_interleaving（`L2129`）经 `L2277/L2285` 调用 | SP 收益顺 PP 传导 |
| FLOPs 公式与 MFU | `training.py:L391/L428-431/L548`（`num_floating_point_operations`：MLP `4·exp·tokens·h²`，×3）+ `L2865-2870`（TFLOP/s/GPU）；MFU 口径 PaLM arXiv:2204.02311 | per-device MFU |

**nano 与权威实现的差异（为什么它那样选）**：

1. Megatron 的 gather 融进线性算子并用 CUDA 全局 buffer + 异步重叠
   （`layers.py:L565-573/L609-618`）；nano 把 gather 写成独立 autograd
   Function，语义相同但无重叠——CPU/gloo 上没有重叠的空间，如实声明。
2. Megatron 的 SP 只与 TP 绑定（`sequence_parallel` 要求 tp_size > 1，
   `layers.py:L1056-1062`），本节同款（SP 与 TP 同组同 degree）；
   真实系统另有 Context Parallel（CP）沿序列维做 attention 级切分，
   与 SP 正交，不在本节范围。
3. Megatron 的 attention 区域（softmax 的 `[s,s]` 激活）也受 SP 收益；
   nano 块无 attention（L0 §5 讲过 attention 的 TP 同构切法），
   收益区只有 LN/Dropout——机制相同，占比不同，如实声明。
4. 真机 SP 的显存收益与 MFU 绝对值需 GPU 验证
   （`[TODO: verify on real system]`，Machine B 通道）；
   本节的结构性结论（等价性、通信账 2.5m、账本 1/t、接缝减半）
   与后端无关。

---

## 10. 费曼：讲给外行听

**类比：图书馆抄书。**

TP（L1）像把一本厚书的**每一章**拆给两个人合抄：你抄左半页、我抄右半页，
每抄完一段要碰头对一次答案（all-reduce）。但每章开头的**目录页**
（LayerNorm/Dropout）两人各抄了一份一模一样的——浪费纸。

SP 的做法：目录页也切开——你抄第 1–50 行的目录，我抄第 51–100 行。
到了正文（MLP）需要完整目录时，两人把目录条拼起来（all-gather）；
正文抄完，把结果按目录行分回各人（reduce-scatter）。
纸省了一半，跑腿次数多了一点（反向还要再拼一次目录算总账）。

PP（L2）是把**不同的章**分给不同的人；组合起来（TP×PP×SP）：
4 个人，两人一组合抄一章（TP），两组各抄两章（PP），
连目录都是切开存的（SP）——组与组之间传递的只是**半份目录厚的纸条**
（PP 接缝字节减半）。

MFU 是问：这套分工到底比一个人单干快多少？
答案 = 真正抄字的秒数 ÷（理论手速 × 总耗时）——
开会（通信）、等上游（bubble）、纸条太小不好抄（小 GEMM 低效）全在里面。

**一句话版**：SP 把 all-reduce 拆成 RS+AG 两半，中间插上切碎的 LN/Dropout；
显存省 1/t，通信多 25%（反向重放 gather）；γ/β 梯度变部分和要再归约；
PP 接缝跟着减半；MFU 把整套并行的效率压成一个可追责的数字。

---

## 11. 思考题

1. 为什么 SP 必须和 TP 绑定同组同 degree，而不能独立成一个并行维度？
   （提示：AG/RS 的 group 就是 TP group——SP 的「切」发生在 TP 的「切」
   留下的未切区域里；没有 TP 就没有「未切区域」，也没有现成的通信组。
   若要独立的序列维并行，那是 Context Parallel 的领域，切的是 attention。）
2. 反向重放 gather 多付 0.5m 通信，换回什么？若某模型的未切区域激活
   只占总激活 5%，SP 还值得开吗？
   （提示：收益 = 未切区域 × (1-1/t)，代价 = 25% TP 通信 + 实现复杂度。
   占比小时收益 < 代价——Megatron 把 SP 做成开关而非默认，正是这个权衡。）
3. §7 说 SP 让 PP 接缝字节减半。若 PP=4、TP=8、序列长 s，
   写出每 step 每 rank 的 PP p2p 字节（非 SP vs SP）。
   （提示：`2·(N-1)·s·h·4` vs `2·(N-1)·(s/8)·h·4`——SP 的收益随 TP degree 放大。）
4. MFU(dense)=24% 说明即使没有通信，toy 形状也只吃到峰值的 1/4。
   若把 H 从 128 提到 4096（真实模型尺寸），MFU(dense) 会怎么变？
   分布式 MFU 与 dense MFU 的比值呢？
   （提示：GEMM 变大 → 计算效率升 → MFU(dense) 升；通信/计算比下降 →
   分布式/dense 比值也升。这就是「大模型更容易训出高 MFU」的算术根源。）

---

## 12. 边界与局限

- **本机数字 ≠ GPU 数字**：全部计时来自 CPU + gloo + loopback；
  list 版 AG/RS 的单次开销远大于 NCCL 的融合实现，SP 实测更慢
  （wall 比 1.5–2×）是后端 artifact。GPU/NCCL 上 SP 通信量仅 1.25×，
  吞吐应近似非 SP；SP 的真机显存收益（1/t 未切区域激活）与 MFU 绝对值
  均标 `[TODO: verify on real system]`（ROADMAP §三 Machine B 通道）。
- **toy 形状**：`H=128, FF=512, T=512, 4 块, PP2×TP2`——
  计算微秒级、通信延迟主导，MFU 绝对值低；结构性结论
  （等价性、wire 2m→2.5m、账本 1/t、接缝减半、γ/β 部分和）与形状无关。
- **无 attention**：SP 对 attention softmax 区域（`[s,s]` 激活）的收益
  未覆盖；nano 块只有 LN+Dropout 未切区域，收益占比小于真实 Transformer。
- **无 CP/DP**：Context Parallel（attention 级序列切分）与 DP（数据并行）
  不在本节；Megatron 完整形态是 DP×TP×PP×(CP) 加 SP 绑定 TP。
- **MFU 峰值口径**：GEMM 标定的是「本机 torch fp32 matmul 可达峰值」，
  不是硬件厂商标称峰值（后者含 bf16/AMX 等，口径不同不可比）——
  本节刻意用同口径标定，保证 MFU 分子分母可比。
- **gloo 的 reduce_scatter/all_gather 为 list 版**：torch 2.13.0 实测可用、
  数值正确；`*_into_tensor` 单缓冲版在 gloo 上的支持情况未验证
  `[TODO: verify]`，Megatron 用单缓冲 + 全局 buffer 池（`layers.py:L565-573`）。

---

## 13. 溯源

- 运行环境：`python3`，torch 2.13.0，
  gloo backend，CPU（每进程 `torch.set_num_threads(4)` 防争抢），seed=7，
  MASTER_PORT=29550，4 进程 = PP2 × TP2。
- **输出保真**：§2 粘贴为本次运行实跑输出；2 个独立 CWD（/tmp/l3-run2、
  /var/tmp/l3-run3）连跑全 EXIT=0、stderr 0 B，除计时行外逐字节一致。
  掩码口径（paste 块与运行输出两侧同施；计时行共 9 行 = `elapsed[` 行 4 +
  [5] 解读行 1 + `MFU sanity` self-check 行 2 + `MFU SP/非SP` self-check 行 1 +
  total wall 行）：
  `sed '/elapsed\[/d; /解读: MFU/d; /MFU sanity/d; /MFU SP\/非SP/d; /^total wall/d'`，
  掩码后余 64 行确定性核心，md5 `a1a0df6100d1984adbbd539633efc0f5`；
  digest（代码内对 deltas/comm/act/bytes/losses/flops 等非计时指标计算的 md5）
  多遍相同：`51b674545d3295004557e525ab161842`。
- **Megatron-LM 源码锚点**（NVIDIA/Megatron-LM main 分支，2026-08-10
  现场抓取核验，行号以抓取日为准；抓取件 md5：mappings.py `a10dd8e5…`/621 行、
  layers.py `1e5ca2e0…`/1522 行、transformer_block.py `cd0ce574…`/800 行、
  training.py `6add7e23…`/4744 行、finalize_model_grads.py `5aed2187…`/30,894 B；
  schedules.py 使用同日独立抓取件，并以 md5 复核零漂移）：
  `mappings.py:L118/L159`（`_gather_along_first_dim` / `_reduce_scatter_along_first_dim`）、
  `L280/L300/L355`（scatter/gather/reduce-scatter 三个 SP autograd 算子）；
  `layers.py:L565-573`（融合线性 fwd gather）、`L609-618`（bwd 重放 gather）、
  `L1472-1478`（RowParallel SP 分支）、`L1056-1062`（SP 要求 tp>1）；
  `transformer_block.py:L595-598`（SP 区域 RNG fork）；
  `schedules.py:L2097/L2122-2123`（`get_tensor_shapes` SP → seq//tp）、
  `L2129`（without_interleaving）、`L2277/L2285`（recv/send 形状调用点）；
  `training.py:L391`（`num_floating_point_operations`）、`L428-431`（mlp_layer_flops）、
  `L548`（×3）、`L2865-2870`（TFLOP/s/GPU）；
  `finalize_model_grads.py:L416/L451-453/L491`（LN 梯度跨 TP 组 SUM 归约）。
  仓库：<https://github.com/NVIDIA/Megatron-LM>。
- **论文**：SP = *Reducing Activation Recomputation in Large Transformer Models*
  arXiv:2205.05198（§4.2.2 序列并行机制；§4.1/§4.3 激活显存分析）；MFU 口径 = PaLM
  arXiv:2204.02311（per-device Model FLOPs Utilization）；
  Megatron-LM arXiv:1909.08053（TP/PP 基础）；GPipe arXiv:1811.06965。
  两篇新引论文标题经 arXiv API 现场复抓逐词核验。
- **与 L1/L2 的衔接**：f/g 算子与 all-reduce≡2×msg 口径直接沿用 L1；
  PP 接缝字节公式 `2·(N-1)·T·H·4` 与 batched p2p 直接沿用 L2
  （本节 SP 下将其推广为 `2·(N-1)·(T/t)·H·4`）；
  「fp32 归约形状差」口径同 L2 [0] / nano-fsdp L2 [4a]/[4b]。
- 未 ssh 远端；真机验证走 Machine B 攒批通道。
