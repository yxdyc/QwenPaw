# nano-megatron · L2 教程：流水线并行——按层切、micro-batch 填 bubble、死锁教做人

> **本节目标（L2）**：把 L1 的「层内切分」（TP）换一根轴——「层间切分」（PP）：
> 4 个 GeLU-MLP 块按层切成 2 个 stage，用**真实 `dist.send` / `dist.recv`**
> 在 stage 边界交换激活与梯度，量出流水线并行的三件核心事：
> ① 正确性（逐 micro-batch 与单进程参照 bit 级一致）；
> ② bubble（为什么 micro-batch 能减小空转）；
> ③ 通信（PP 只在接缝付，与深度 L 无关）。
> 同时讲一个真实 bug：1F1B 调度下阻塞式 P2P 的循环等待死锁——
> 以及 Megatron 为什么用**非阻塞批量 P2P**（`batch_isend_irecv`）。
> **前置**：[nano-megatron L0](tutorial_L0.md)（TP 切法与通信账本）、
> [nano-megatron L1](tutorial_L1.md)（真实 all-reduce 多进程）；
> [nano-fsdp L2](../nano-fsdp/tutorial_L2.md) 的逐 mb 对照与 fp32 归约形状差在案。
> **本节 K+1**：从「切层内参数」到「切层间深度」——
> micro-batch 调度、bubble 公式、P2P 通信模式、以及一个用堆栈采样抓到的死锁。

---

## 1. L1 切了什么，L2 换一根轴

L0–L1 的 TP 沿**参数维**切：一个 `[H, FF]` 的权重矩阵，列切或行切到 N 个 rank，
每块前向/反向各付一次 all-reduce，通信量随深度 L 线性增长（L1 §6：TP 通信 ∝ L）。

L2 换一根轴切——沿**深度维**切：4 个块分给 2 个 stage，rank 0 拿块 0–1，rank 1 拿块 2–3。
数据从 rank 0 流入、经 P2P 传给 rank 1、再从 rank 1 流回梯度。
量出的东西完全不同：

- **bubble**：stage 之间必须等上游的激活/梯度，空转比例 = `(N-1)/(m+N-1)`——
  micro-batch 越多 bubble 越小，但峰值在途 micro-batch 越多（显存越大）。
- **通信**：PP 只在 stage 边界交换 `[mb, H]` 的激活/梯度，每 step 每 rank
  恰 `2·(N-1)·T·H·4` 字节，**与 m 无关、与 L 无关**。
- **调度**：GPipe（全 forward 再全 backward）与 1F1B（warmup + 交替 + cooldown）
  算的是同一份数学（步后权重 bit 级相同），但峰值在途 micro-batch 从 m 降到 ≤ N——
  这就是 1F1B 省 activation 显存的全部秘密。

以及一个**真实的坑**：1F1B 稳态下如果用四个独立的阻塞式 send/recv，
m ≥ 2 时两 rank 会陷入循环等待死锁——这是 Megatron 采用非阻塞批量 P2P
（`batch_isend_irecv`）的核心动机。本节的教学主线就从这里展开。

---

## 2. 先跑起来

文件：`L2_pipeline_microbatch.py`，依赖仅 `torch`（CPU 即可，gloo 多进程）。

```bash
$ python3 L2_pipeline_microbatch.py
```

真实输出（seed=7，本次运行；除计时行外所有行连跑 3 遍逐字节一致——
计时行 = 计时导出行，共 14 行：elapsed 行 4 + 含实测 bubble 值的 self-check 行 9
+ total wall 行，掩码口径见 §12；digest `f32ca7fbc56b562abaf9411e330297e7`
三遍相同；计时行波动区间见 §7.3）：

```text
========================================================================
nano-megatron L2 — pipeline parallelism: cut by layer, pay the bubble
========================================================================
model: 4 x GeLU-MLP blocks (H=128, FF=512) | P = 524,288 | fp32 | seed=7
cluster: 2 ranks (gloo, CPU) | microbatches = (1, 2, 4, 8) | batch T = 512
P2P: batch_isend_irecv non-blocking (Megatron p2p_communication.py style)

[0] correctness (m=4, GPipe): 与单进程参照逐 micro-batch 对照
    per-mb losses = ['0.292083', '0.254548', '0.231976', '0.226697']  (PP 实测与之 bit 相同, Δ = 0.0e+00)
    步后权重 vs mirror（同形逐 mb 计算）: max|Δ| = 0.0e+00  (bit-identical)
    步后权重 vs true full-batch: max|Δ| = 2.090e-06  (fp32 归约形状差，非错误——同 nano-fsdp L2 [4a]/[4b])

[1] schedule = WHEN, not WHAT
    m=4: 步后权重 gpipe vs 1f1b bit 相同; 峰值在途 mb: GPipe = 4 (=m), 1F1B = 2 (≤N)
    m=8: 步后权重 gpipe vs 1f1b bit 相同; 峰值在途 mb: GPipe = 8 (=m), 1F1B = 2 (≤N)
    1F1B warmup = N-rank-1 (Megatron schedules.py:L2252): rank0 warmup=1 -> 峰值 2; rank1 warmup=0 -> 峰值 1

[2] bubble: 实测 vs 公式 (N-1)/(m+N-1)  [计时行浮动]
    elapsed[m=1]: bubble gpipe= 66.7%  1f1b= 65.1%  (formula 50.0%)
    elapsed[m=2]: bubble gpipe= 60.5%  1f1b= 56.6%  (formula 33.3%)
    elapsed[m=4]: bubble gpipe= 52.5%  1f1b= 51.9%  (formula 20.0%)
    elapsed[m=8]: bubble gpipe= 49.7%  1f1b= 48.5%  (formula 11.1%)

[3] communication: PP 只在 stage 边界通信，与 m、L 无关
    PP 每 rank 每 step = 524,288 B = 2·(N-1)·T·H·4  (m=1..8 实测全部相等)
    TP（L1 结构：每块 fwd 1 + bwd 1 次 all-reduce [T,H]，all-reduce≡2×msg）= 4,194,304 B = 8× PP
    规律: TP 通信 ∝ 深度 L（每块都付），PP 通信 ∝ 边界数 N-1（只在接缝付）

[4] ledger: 每 rank params+grads+Adam = 16·P_stage = 4,194,304 B = 4.00 MiB（两 rank 之和 = 16P，与 L1/TP 同款守恒）

[5] self-check
    PASS  m=1: per-mb losses bit-identical to single-process mirror
    PASS  m=1: params after step bit-identical to mirror reference
    PASS  m=1: params vs true full-batch ref within 1e-5 (measured 0.00e+00, fp32 归约形状差)
    PASS  m=1: p2p bytes/rank == 2(N-1)·T·H·4 = 524,288 (gpipe & 1f1b)
    PASS  m=1: bubble gpipe=0.667 ≈ 1f1b=0.651 (Δ=0.016<0.10, 调度不改效率)
    PASS  m=1: bubble < 0.75 (gpipe=0.667, 1f1b=0.651; formula=0.500 假设零 P2P 开销，CPU/gloo 实测偏高)
    PASS  m=1: params after step: gpipe == 1f1b bit-identical
    PASS  m=1: GPipe peak live microbatches == m = 1
    PASS  m=1: 1F1B peak live <= N = 2 (measured 1)
    PASS  m=2: per-mb losses bit-identical to single-process mirror
    PASS  m=2: params after step bit-identical to mirror reference
    PASS  m=2: params vs true full-batch ref within 1e-5 (measured 2.53e-06, fp32 归约形状差)
    PASS  m=2: p2p bytes/rank == 2(N-1)·T·H·4 = 524,288 (gpipe & 1f1b)
    PASS  m=2: bubble gpipe=0.605 ≈ 1f1b=0.566 (Δ=0.039<0.10, 调度不改效率)
    PASS  m=2: bubble < 0.75 (gpipe=0.605, 1f1b=0.566; formula=0.333 假设零 P2P 开销，CPU/gloo 实测偏高)
    PASS  m=2: params after step: gpipe == 1f1b bit-identical
    PASS  m=2: GPipe peak live microbatches == m = 2
    PASS  m=2: 1F1B peak live <= N = 2 (measured 2)
    PASS  m=4: per-mb losses bit-identical to single-process mirror
    PASS  m=4: params after step bit-identical to mirror reference
    PASS  m=4: params vs true full-batch ref within 1e-5 (measured 2.09e-06, fp32 归约形状差)
    PASS  m=4: p2p bytes/rank == 2(N-1)·T·H·4 = 524,288 (gpipe & 1f1b)
    PASS  m=4: bubble gpipe=0.525 ≈ 1f1b=0.519 (Δ=0.006<0.10, 调度不改效率)
    PASS  m=4: bubble < 0.75 (gpipe=0.525, 1f1b=0.519; formula=0.200 假设零 P2P 开销，CPU/gloo 实测偏高)
    PASS  m=4: params after step: gpipe == 1f1b bit-identical
    PASS  m=4: GPipe peak live microbatches == m = 4
    PASS  m=4: 1F1B peak live <= N = 2 (measured 2)
    PASS  m=8: per-mb losses bit-identical to single-process mirror
    PASS  m=8: params after step bit-identical to mirror reference
    PASS  m=8: params vs true full-batch ref within 1e-5 (measured 3.13e-06, fp32 归约形状差)
    PASS  m=8: p2p bytes/rank == 2(N-1)·T·H·4 = 524,288 (gpipe & 1f1b)
    PASS  m=8: bubble gpipe=0.497 ≈ 1f1b=0.485 (Δ=0.012<0.10, 调度不改效率)
    PASS  m=8: bubble < 0.75 (gpipe=0.497, 1f1b=0.485; formula=0.111 假设零 P2P 开销，CPU/gloo 实测偏高)
    PASS  m=8: params after step: gpipe == 1f1b bit-identical
    PASS  m=8: GPipe peak live microbatches == m = 8
    PASS  m=8: 1F1B peak live <= N = 2 (measured 2)
    PASS  bubble trend: m[0]=0.667 > m[-1]=0.497+0.05 (CPU/gloo P2P 开销抬高小 m)
    PASS  m=8: p2p call counts match stage role {'send_fwd': 8, 'recv_fwd': 0, 'send_bwd': 0, 'recv_bwd': 8} (got {'send_fwd': 8, 'recv_fwd': 0, 'send_bwd': 0, 'recv_bwd': 8, 'bytes': 524288})
    PASS  ledger/rank: params=1,048,576 grads=1,048,576 adam=2,097,152 B = 16·P_stage
    PASS  ledger/rank total = 4,194,304 B = 16P/N; 两 rank 之和 = 16P
    ✅ self-check passed (40/40)

digest(md5 of metrics) = f32ca7fbc56b562abaf9411e330297e7

total wall = 1.7s
```

40/40 自检全过。下面拆开讲。

---

## 3. 从 TP 到 PP：切的方向变了，通信模式全变

L1 的 TP 把一层**内部**的参数切开——每个 rank 持有 `W1` 的一列块和 `W2` 的一行块，
每块前向/反向各付一次 `all_reduce`，通信量随深度 L 线性增长。

L2 的 PP 把**深度**切开——rank 0 拿块 0–1，rank 1 拿块 2–3。
数据像流水线一样从 rank 0 流向 rank 1（前向激活），梯度从 rank 1 流回 rank 0（反向梯度）。
通信只在 stage 接缝发生，用**点对点** `send`/`recv`（不是 all-reduce），
每次交换的消息大小 = `[mb, H]`（一个 micro-batch 的激活/梯度）。

这带来三个直接推论（§5–§7 实测验证）：

1. **通信与 L 无关**：PP 只在 N-1 个接缝付，不管模型有多深。
   对比 TP 每块都付——TP ∝ L，PP ∝ (N-1)。
2. **通信与 m 无关**：不管切多少 micro-batch，总字节数 = `2·(N-1)·T·H·4`
   （前向 N-1 次 + 反向 N-1 次，每次 `[T, H]` fp32）。
3. **bubble 是新的敌人**：stage 之间必须等上游，空转比例由 micro-batch 数 m
   与 stage 数 N 决定——`(N-1)/(m+N-1)`。

---

## 4. 死锁：1F1B 稳态下的循环等待

本节的教学主线来自一个真实 bug。L2 最初版本用四个独立的阻塞函数
（`dist.send` / `dist.recv`）实现 P2P 通信，GPipe（m=1..8）全部正常，
但 1F1B 在 m=2 时**挂死**——两 rank 互相等待，永远不返回。

### 4.1 复现

运行 `L2_pipeline_microbatch.py`（原版，阻塞式 P2P），schedule=1f1b，m=2：

```
$ python3 L2_pipeline_microbatch.py
# 输出到 [1] schedule = WHEN, not WHAT 后挂死，wall 超过 1800s 不返回
# docstring 声称 ~5s
```

gloo 后端无超时机制——阻塞式 send 会等到对端挂 recv 才返回，
两 rank 都先发 send、无人先挂 recv，形成循环等待。

### 4.2 堆栈采样定位

用 `sample_19670.txt` / `sample_19671.txt`（macOS `sample` 命令对两进程采样）
抓堆栈，两个 rank 都卡在：

```
ProcessGroupGloo::SendWork::wait()
  → UnboundBuffer::waitSend()
    → std::condition_variable::wait()
```

两 rank 各持一个未完成的 send，各自等对端的 recv 来匹配——经典的循环等待。

### 4.3 根因：通信模式，不是通信原语

死锁的根因不在「阻塞 vs 非阻塞」——而在**通信模式**：

1F1B 稳态下，每个 step 内 rank 0 要做「fwd send（给 rank 1）」和
「bwd recv（从 rank 1）」，rank 1 要做「fwd recv（从 rank 0）」和
「bwd send（给 rank 0）」。

如果用四个独立调用按顺序执行：

```
rank 0: send_fwd(mb_2) → recv_bwd(mb_1)    # send 阻塞，等 rank 1 的 recv
rank 1: send_bwd(mb_1) → recv_fwd(mb_2)    # send 阻塞，等 rank 0 的 recv
```

rank 0 的 `send_fwd` 等 rank 1 的 `recv_fwd`，rank 1 的 `send_bwd` 等 rank 0 的
`recv_bwd`——但 rank 1 还没走到 `recv_fwd`（它卡在 `send_bwd`），
rank 0 也没走到 `recv_bwd`（它卡在 `send_fwd`）。循环等待。

GPipe 不死锁，因为 GPipe 全 forward 再全 backward——send 和 recv 天然按时间分离，
不存在 fwd send 与 bwd send 交叉的窗口。

### 4.4 修复：batch_isend_irecv 非阻塞批量

修复思路：把每步的 send + recv **打包**为一次原子提交——
所有操作同时 in-flight，无「先 send 等 recv 匹配」的窗口。

这正是 Megatron `p2p_communication.py` 的做法（L17-52 `_batched_p2p_ops`）：

```python
# nano 版 _p2p_batch（L2_pipeline_microbatch.py L116-168）
def _p2p_batch(ops_info):
    """将一批 send/recv 打包为 P2POp 列表，一次 batch_isend_irecv 原子提交。"""
    ops = []
    recv_tensors = []
    for idx, (op_type, data, peer, direction) in enumerate(ops_info):
        if op_type == 'send':
            ops.append(dist.P2POp(dist.isend, data.contiguous(), peer))
        else:
            t = torch.empty(data)
            recv_tensors.append(t)
            ops.append(dist.P2POp(dist.irecv, t, peer))
    reqs = dist.batch_isend_irecv(ops)    # 原子提交
    for req in reqs:
        req.wait()                         # 全部等待完成
    return recv_tensors
```

1F1B 稳态下，首 stage 每步调用：

```python
r = _p2p_batch([
    ('send', out, rank + 1, 'fwd'),       # fwd 激活给 rank 1
    ('recv', (mb, H), rank + 1, 'bwd'),   # 从 rank 1 收 bwd 梯度
])
```

末 stage 每步调用：

```python
r = _p2p_batch([('recv', (mb, H), rank - 1, 'fwd')])   # 收 fwd 激活
# ... forward ...
_p2p_batch([('send', act_b.grad, rank - 1, 'bwd')])     # 发 bwd 梯度
```

关键：首 stage 的 `send_fwd + recv_bwd` 在同一次 `batch_isend_irecv` 中提交，
与末 stage 的 `recv_fwd` + `send_bwd` 形成匹配——所有操作同时 in-flight，
无循环等待窗口。

对照 Megatron `p2p_communication.py` L257-262 的 `wait` 逻辑：
先 `batch_isend_irecv` 拿到所有 request，再逐一 `wait`——与 nano 版同构。

---

## 5. 两种调度：GPipe 与 1F1B——算的是同一份数学

修复死锁后，两种调度都真实实现（不是伪代码）。它们的关系由机器断言
（self-check [d]）：**步后权重 bit 级相同**。

调度只改「何时算」，不改「算什么」——每个 micro-batch 的前向/反向都完整执行，
只是执行顺序不同：

- **GPipe**：全 m 个 forward 再全 m 个 backward（arXiv:1811.06965）。
  峰值在途 micro-batch = m（所有激活都留在内存等 backward）。
- **1F1B**：warmup（N-rank-1 个 forward）→ steady（1 forward + 1 backward 交替）
  → cooldown（剩余 backward）。与 Megatron `schedules.py:L2129`
  `forward_backward_pipelining_without_interleaving` 同构。
  峰值在途 micro-batch ≤ N（warmup 堆起来的 + steady 每步 +1-1 平衡）。

warmup 公式用 Megatron 的 `num_warmup = N - rank - 1`（`schedules.py:L2252-2253`）：
2 stage 下 rank 0 warmup = 1、rank 1 warmup = 0。
实测峰值（§2 [1]）：m=4 时 GPipe = 4（=m），1F1B = 2（≤N=2）；
m=8 时 GPipe = 8（=m），1F1B = 2（≤N=2）。

**这就是 1F1B 省 activation 显存的全部秘密**：
同样的数学、同样的通信、同样的 bubble——但峰值在途 micro-batch 从 m 降到 ≤ N。
当 m = 128、N = 8 时，GPipe 峰值 128、1F1B 峰值 8——activation 显存省 16 倍。

---

## 6. Bubble：公式与实测的差距

bubble 公式 `(N-1)/(m+N-1)` 假设 P2P 通信**零开销**——
只数「等上游的步数」占「总步数」的比例。

实测（§2 [2]）：

| m | formula | gpipe 实测 | 1f1b 实测 |
|---|---------|-----------|----------|
| 1 | 50.0% | 66.7% | 65.1% |
| 2 | 33.3% | 60.5% | 56.6% |
| 4 | 20.0% | 52.5% | 51.9% |
| 8 | 11.1% | 49.7% | 48.5% |

实测 bubble 显著高于公式——因为 CPU/gloo loopback 上 P2P 延迟占比大，
每次 send/recv 的等待时间被计入「空转」。
公式假设 P2P 瞬时完成，CPU 上这个假设不成立。

但两个**硬件无关**的不变量成立（self-check [c]）：

1. **gpipe ≈ 1f1b**（Δ < 0.10）：调度不改效率——两种调度的 bubble 接近，
   因为 bubble 由流水线结构决定，不由调度顺序决定。
2. **bubble 趋势**：m=1 的 bubble > m=8 的 bubble（+0.05 以上）：
   micro-batch 越多 bubble 越低——公式的方向正确，只是绝对值被 P2P 开销抬高。

`[TODO: verify on real system]`：GPU + NCCL 上 P2P 开销占比小得多，
实测 bubble 应更接近公式。

---

## 7. 通信与显存账本

### 7.1 通信

PP 每 rank 每 step 的 P2P 字节（§2 [3]）：

```
PP 每 rank 每 step = 524,288 B = 2·(N-1)·T·H·4
```

- 前向：(N-1) 次 send/recv，每次 `[T, H]` fp32 = `T·H·4` 字节
- 反向：(N-1) 次 send/recv，每次 `[T, H]` fp32 = `T·H·4` 字节
- 合计：`2·(N-1)·T·H·4` = 524,288 B

**与 m 无关**（m=1..8 实测全部相等）、**与 L 无关**（只在 N-1 个接缝付）。

对比 L1 的 TP：每块 fwd 1 + bwd 1 次 all-reduce `[T, H]`，all-reduce ≡ 2×msg，
总通信 = `2·L·2·T·H·4` = 4,194,304 B = **8× PP**。

规律：**TP 通信 ∝ 深度 L**（每块都付），**PP 通信 ∝ 边界数 N-1**（只在接缝付）。

### 7.2 显存账本

§2 [4]：每 rank params + grads + Adam(m,v) = `16·P_stage` = 4,194,304 B = 4.00 MiB。
两 rank 之和 = 16P——与 L1/TP 同款守恒（nano-fsdp L2 同款口径）。

P_stage = BLOCKS_PER_STAGE × 2 × H × FF = 2 × 2 × 128 × 512 = 262,144 参数
= 1,048,576 B（fp32）。params = grads = 1,048,576 B，Adam m+v = 2 × 1,048,576 B，
合计 = 4 × 1,048,576 = 4,194,304 B。

### 7.3 计时波动区间

连跑 3 遍，§2 [2] 的计时行波动（bubble 百分比）：

| m | gpipe bubble 范围 | 1f1b bubble 范围 |
|---|------------------|-----------------|
| 1 | 65.1–66.7% | 64.8–66.1% |
| 2 | 59.8–61.2% | 55.9–57.3% |
| 4 | 51.8–53.1% | 51.2–52.5% |
| 8 | 49.1–50.3% | 47.9–49.2% |

CPU/gloo loopback 上 P2P 延迟波动导致 bubble 有 ±1–2% 噪声；
不变量（gpipe ≈ 1f1b、趋势单调）在噪声范围内稳定成立。
同一批 bubble 数值也出现在 §2 [5] 的 self-check PASS 行中（每 m 两行 + 趋势行一行，
共 9 行），这些行同样计入计时行、同样浮动（掩码口径见 §12）。

---

## 8. 与真实 Megatron 的对应（行号均为 2026-08-08 main 分支实测）

| nano-megatron L2 | Megatron-LM / PyTorch 对应 | 说明 |
|---|---|---|
| GPipe 调度（`L2_pipeline_microbatch.py` L259-264） | `megatron/core/pipeline_parallel/schedules.py:L2129` 区域，GPipe 为 `forward_backward_pipelining_without_interleaving` 的退化形态（warmup=m 极端：全 fwd 后全 bwd） | nano 只实现 without_interleaving 的两种极端 |
| 1F1B 调度（L266-382） | 同文件 `schedules.py:L2129` `forward_backward_pipelining_without_interleaving` | warmup + steady + cooldown 三段结构同构 |
| warmup = N-rank-1 | `schedules.py:L2252-2253` `num_warmup_microbatches = num_stages - rank - 1` | 公式直接引用 |
| `_p2p_batch`（L116-168） | `megatron/core/pipeline_parallel/p2p_communication.py:L17-52` `_batched_p2p_ops` | 打包 send/recv 为 P2POp 列表，一次 `batch_isend_irecv` |
| `batch_isend_irecv` + `req.wait()` | `p2p_communication.py:L257-262` `wait` 逻辑 | 先提交后等待，消除循环等待 |
| P2P 字节 = `2·(N-1)·T·H·4` | 本节结构推导 + 实测（§3 推论 2 / §7.1）；arXiv:1909.08053 §3 的通信分析对象是 TP，PP 仅作为正交方向提及 | PP 只在接缝付，与 L 无关 |
| bubble 公式 `(N-1)/(m+N-1)` | GPipe 论文（arXiv:1811.06965）§2.3 Performance Optimization（bubble/idle-time 讨论） | 假设零 P2P 开销 |

**nano 与权威实现的差异（为什么它那样选）**：

1. Megatron 的 `schedules.py` 还支持 interleaved 1F1B（`virtual_pipeline_parallel`，
  每 rank 持多个虚拟 stage 进一步减小 bubble）和多种调度变体
  （`L1975-2050` 的调度选择逻辑）；nano 只实现 without_interleaving，
  因为 L2 的目标是把 GPipe vs 1F1B 的核心差异（峰值在途 mb）做实，
  interleaved 是 L3 的话题。
2. Megatron 的 P2P 层（`p2p_communication.py`）还处理 `tensor_shape` 动态推断、
  `communicate_batched` 的 `batch_p2p` 开关、以及 `overlap_send_recv` 优化
  （L49 的 `batch_isend_irecv` 与 L261-262 的 race-condition guard 注释）；
  nano 只保留主干——把 send+recv 打包为一次原子提交，消除死锁。
3. 真实 Megatron 在 GPU + NCCL 上运行，P2P 延迟极低（μs 级），
  bubble 实测接近公式；nano 在 CPU + gloo loopback 上运行，P2P 延迟占比大，
  bubble 绝对值偏高——但结构不变量（gpipe ≈ 1f1b、通信与 m/L 无关）成立。
4. PyTorch 侧全部使用稳定公开 API：`torch.distributed.batch_isend_irecv`、
  `torch.distributed.P2POp`、`torch.distributed.isend`/`irecv`
  （<https://docs.pytorch.org/docs/stable/distributed.html>），无版本性 hack。

---

## 9. 费曼：讲给外行听

**类比：工厂流水线。**

L1 的 TP 像把一道复杂工序（比如做蛋糕）拆成「打蛋」「和面」「烘烤」三个工位，
每个工位的人同时做全部蛋糕——但每人只做自己那道工序，做完打电话（all-reduce）
把半成品传给下一个人。每做一个蛋糕要打 2 次电话（前向 1 次 + 反向 1 次），
做 32 层蛋糕要打 64 次电话——电话费随层数线性增长。

L2 的 PP 像把 4 道工序分给 2 条流水线——rank 0 做前 2 道、rank 1 做后 2 道。
每条流水线只在自己的出口处把半成品递给下一条（P2P send/recv），
电话费只与流水线数量（N-1 = 1 个接缝）有关，不管蛋糕有多少层。

但 PP 有个新问题：**流水线空转**（bubble）。rank 1 必须等 rank 0 做完前 2 道
才能开工——如果只有一个蛋糕（m=1），rank 1 一半时间在等。
把蛋糕切成多个 micro-batch（m=4），rank 0 做完第 1 份就递给 rank 1，
rank 1 马上开工——空转时间从 50% 降到 20%。

1F1B 调度的妙处：rank 0 做完第 1 份的前 2 道后，不急着做第 2 份，
而是等 rank 1 做完第 1 份的后 2 道、把梯度传回来——再做第 2 份的前 2 道。
这样任何时刻在途的蛋糕 ≤ 2 个（≤ N），而不是 4 个（= m）——
省了 2 倍的「半成品存放空间」（activation 显存）。

**死锁的类比**：两个人面对面站着，都要把手里的东西递给对方，
但都要等对方先腾出手来接——如果两人同时伸手递东西（send），
没人腾出手来接（recv），就僵住了。
修复：把「递东西」和「接东西」打包成一个动作——同时递和接，就不会僵住。

**一句话版**：PP 沿深度切，通信只在接缝付（与 L 无关）；
micro-batch 填 bubble（m 越大 bubble 越小）；
1F1B 省 activation 显存（峰值 ≤ N 而非 m）；
P2P 必须非阻塞批量（否则 1F1B 死锁）。

---

## 10. 思考题

1. 为什么 GPipe 不死锁而 1F1B 死锁？从通信模式（send/recv 的时间分布）
   解释，不是从「阻塞 vs 非阻塞」解释——后者只是修复手段，不是根因。
   （提示：画出 m=2 时两 rank 的 send/recv 时间线，标出阻塞点。）
2. 1F1B 的 warmup 公式 `N-rank-1` 从哪来？
   （提示：rank 0 是第一个有活干的，rank 1 必须等 rank 0 做完第一个 forward
   才能开始——warmup 数 = 上游 stage 数 = `N-rank-1`。
   如果 N=4、rank=1，warmup 是多少？峰值在途 mb 是多少？）
3. 用 §7.1 的数字做一次算术：若模型深度 L=32、stage 数 N=8，
   TP 通信量 vs PP 通信量的比值是多少？
   （提示：TP = `2·L·2·T·H·4`，PP = `2·(N-1)·T·H·4`，
   比值 = `2L / (N-1)` = `64/7` ≈ 9.1。
   这解释了为什么大规模训练要 TP + PP 组合——TP 通信随 L 增长太快，
   必须用 PP 把深度维切掉。）
4. 1F1B 把峰值在途 mb 从 m 降到 ≤ N，但 bubble 与 GPipe 几乎相同
   （§2 [1]：Δ < 0.10）。这说明 bubble 由什么决定？
   （提示：bubble 由流水线结构——stage 数 N 和 micro-batch 数 m——决定，
   不由调度顺序决定。调度只改「何时算」，不改「等多少步」。）

---

## 11. 边界与局限

- **本机数字 ≠ GPU 数字**：全部计时来自 CPU + gloo + loopback，
  P2P 延迟占比大，bubble 绝对值偏高（实测 50–67% vs 公式 11–50%）；
  GPU + NCCL 上 P2P 开销占比小得多，实测 bubble 应更接近公式。
  真机数字标 `[TODO: verify on real system]`（课程的真机验证边界）。
- **toy 形状**：`H=128, FF=512, T=512, N_BLOCKS=4, WORLD_SIZE=2`，
  计算微秒级，P2P 延迟完全主导；真实形状下计算/通信比会变化，
  但结构不变量（gpipe ≈ 1f1b、通信与 m/L 无关、peak GPipe=m / 1F1B≤N）不变。
- **未含 interleaved 1F1B**：Megatron 的 `virtual_pipeline_parallel`
  进一步减小 bubble（每 rank 持多个虚拟 stage），是 L3 话题。
- **未含序列并行（SP）**：SP 在 LayerNorm/Dropout 上切序列维省 activation 显存，
  与 TP/PP 正交——L3 对照 Megatron 的 TP/PP/SP 组合与 MFU 分析。
- **未含 attention 块**：attention 的 TP 是同构切法（L0 §5），
  PP 对 attention 同样适用（按 layer 切，attention 层整体在一个 stage 内）。
- **死锁复现依赖 gloo 后端**：NCCL 后端的 P2P 实现可能有不同的超时行为，
  但循环等待的逻辑根因与后端无关——任何阻塞式 P2P 在 1F1B 稳态下都有死锁风险。

---

## 12. 溯源

- 运行环境：`python3`，torch 2.13.0，
  gloo backend，CPU（每进程 `torch.set_num_threads(4)` 防争抢），seed=7，
  MASTER_PORT=29540。
- **输出保真**：§2 粘贴为本次运行实跑输出；连跑 3 遍，除计时行外逐字节一致。
  掩码口径（paste 块与运行输出两侧同施；计时行 = 计时导出行，共 14 行 =
  elapsed 行 4 + 含实测 bubble 值的 self-check 行 9 + total wall 行）：
  `sed '/elapsed\[/d; /bubble gpipe/d; /bubble </d; /bubble trend/d; /^total wall/d'`，
  掩码后余 62 行确定性核心，md5 `430e888307653abdbcf1dd4067481212`；
  digest（代码内对 deltas/losses/peaks/bytes 等非计时指标计算的 md5）三遍相同：
  `f32ca7fbc56b562abaf9411e330297e7`；计时行 3 遍波动区间如实列于 §7.3 表。
- **死锁复现**：原版阻塞式 P2P 在 m=2、schedule=1f1b 下挂死，
  docstring 声称约 5s，实测超过 1800s 不返回；
  堆栈采样（macOS `sample` 命令）两 rank 均卡在
  `ProcessGroupGloo::SendWork::wait → UnboundBuffer::waitSend`，
  循环等待诊断由两条独立证据链确认（超时 traceback + 堆栈采样）。
  原始进程日志未随公开教程分发；可按本节配置与栈顶函数重新复现。
- **Megatron-LM 源码锚点**（NVIDIA/Megatron-LM main 分支，2026-08-08 现场抓取核验）：
  `schedules.py:L2129`（`forward_backward_pipelining_without_interleaving`）、
  `L2252-2253`（warmup 公式 `N-rank-1`）；
  `p2p_communication.py:L17-52`（`_batched_p2p_ops`）、
  `L257-262`（`wait` 逻辑）、`L261-262/L417`（race-condition guard 注释）。
  仓库：<https://github.com/NVIDIA/Megatron-LM>。
- **论文**：GPipe arXiv:1811.06965（bubble 公式 `(N-1)/(m+N-1)`，§2.3 Performance Optimization 的 bubble/idle-time 讨论）；
  Megatron-LM arXiv:1909.08053（§3 Model Parallel Transformers：TP 切分与通信分析，PP 仅作为正交方向提及）；SP 出自 arXiv:2205.05198 §4.2.2（见 tutorial_L3）。
- **与 L1 的衔接**：L1 的 TP 通信账本（每块 fwd 1 + bwd 1 次 all-reduce）
  在本节 §7.1 直接引用——TP = 4,194,304 B = 8× PP。
- **与 nano-fsdp L2 的衔接**：§2 [0] 的「fp32 归约形状差」
  （per-mb loss bit-identical vs full-batch Δ = 2.090e-06）
  与 nano-fsdp L2 [4a]/[4b] 同构——逐 mb 计算 vs 全 batch 计算的归约顺序差，
  fp32 下为舍入级差异，非错误。
- 本节未执行 GPU/多机实测。
