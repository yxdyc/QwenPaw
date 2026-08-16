# nano-slime · L1 教程：实测 G/T/S——真实小模型上的 generate vs train

> L0 把 G（生成一批 rollout）/ T（训练一步）/ S（权重同步）当**模拟常数**，
> 演示了解耦的结构。L1 在一台真实机器上，用一个真实的小模型把这三个数**测出来**，
> 再灌回 L0 的模拟器——看「真实数字下的解耦」长什么样。
> 前置：先跑过 [L0](tutorial_L0.md)。对应实现：[L1_real_gen_train_timing.py](L1_real_gen_train_timing.py)。

---

## 1. 三个声明（读数字之前必须知道的事）

1. **绝对数字不可外推，结构可以。** 本机是 CPU、单线程、0.8M 参数的小模型。
   「0.35 ms/token」到了 GPU + 数十亿参数上完全不同（那里每 token 的代价主要由
   读权重的显存带宽决定，见 §9）。可外推的是**结构**：G 随 L 线性、batching 压缩 G、
   同批 G > T、生成主导区里解耦买不到吞吐——这些是机制，不是机器常数。
2. **探针模型是真实的，不是 mock。** 一个现场训练的 char-level 小 GPT，语料是
   2026-08-06 抓取的 THUDM/slime README 真实文本（溯源见 §14）。rollout 是从
   学到的分布里真实采样出的文本。用它而不用随机权重，是因为 L1 验收要求
   「接真实小模型」，也因为「训练一步 T」必须测在真实的训练过程上。
3. **生成内容确定，计时值浮动。** greedy + 固定 seed → 输出逐字节可复现；
   墙钟随机器负载浮动。计时协议见 §3.3，三遍连跑区间见 §8。

跑法（依赖仅 torch，CPU 即跑，约 2.5 分钟，其中 ~47s 是探针模型预训练）：

```bash
$ python3 L1_real_gen_train_timing.py
```

真实输出（run1，2026-08-06；计时行以外的所有行连跑 3 遍逐字节一致——
mask 计时行后 md5 三遍相同：`9cd878fd25855f3210b515032b7f5f85`；计时行波动见 §8）：

```text
====================================================================
nano-slime L1 — 实测 G/T/S：真实小模型上的 generate vs train
====================================================================
env: torch 2.13.0 | CPU | threads=1 | seed=7 | greedy decode（内容确定，计时浮动）

[0] 探针模型: char-level GPT | vocab=66 | d=128 | layers=4 | heads=8 | params=797,184
    语料: THUDM/slime README 真实文本切片 3362 bytes（溯源见 tutorial §14）
    预训练: 1200 步 × batch8×seq128 | 末步 loss=0.022 nats/char | 耗时 46.6s（不计入任何 G/T 测量）
    rollout 展示（prompt | 生成 96 字符，greedy）:
      prompt  : ...'**slime** is an LLM post-trainin'
      生成续写: 'g framework for RL scaling, providing two core capabilities:\n\n1.  **High-Performance Training**:'
    确定性: 重生成一次，逐 token 相同 ✓

[1] G(L): 串行生成 1 条 rollout（B=1，KV cache，round-robin ×5 取中位数）
       L |   G (ms) | ms/token
      16 |      4.4 |     0.27
      32 |      8.8 |     0.27
      64 |     19.6 |     0.31
     128 |     38.2 |     0.30
     256 |     88.5 |     0.35
    线性拟合 G = -2.8 + 0.350·L (ms) | R² = 0.9958
    截距 a = 每批固定开销（Python/内核启动）；斜率 b = 每 token 串行前向价

[2] batching 压缩: 共 16 条 rollout × L=128，变批大小（threads=1）
      B |  单条 G (ms) |  vs B=1
      1 |      39.26 |   1.00x
      2 |      48.26 |   0.81x
      4 |      28.65 |   1.37x
      8 |      18.67 |   2.10x
     16 |      15.09 |   2.60x
    压缩率 G(B=1)/G(B=16) = 2.60x —— 固定开销与权重读被摊薄
    探针（threads=8）: B1 52.3  B2 86.8  B4 54.7  B8 32.6  B16 21.0 ms/条
    → 本模型尺寸下多线程在每个批大小都更慢（matmul 太小，调度开销主导）；
      线程开始赚钱的拐点取决于 模型×批 的形状。批与线程是耦合的旋钮，
      GPU 引擎干脆用数千线程的 SIMD 重写这条曲线（L2 / nano-vllm-sglang）。

[3] T(L): 一个训练步（fwd+bwd+Adam，batch=16，round-robin ×5 取中位数）
       L |   T (ms) |  ms/token(全批)
      16 |     24.3 |         0.032
      32 |     33.0 |         0.032
      64 |     49.4 |         0.032
     128 |    102.9 |         0.040
     256 |    273.3 |         0.059
    线性拟合 T = -7.2 + 1.046·L (ms) | R² = 0.9743
    同一批（16×L=128）: G_batch=236 ms vs T=103 ms → G/T = 2.3
    训练一步把 16×(32+128) token 并行过一遍；生成要 128 步串行前向——这就是 rollout 主导的墙钟根源

[4] S: 权重同步（trainer→rollout 侧参数拷贝）= 0.09 ms（3.0 MB，31.51 GB/s）
    对照: S/T = 0.001 —— 本机 CPU 上同步不是瓶颈
    （真实 slime 里是 Megatron→SGLang 跨引擎传输，见 L3 的 delta weight sync）

[5] 实测 G/T/S 灌回 L0 模拟器（12 批，buffer C=4）
    lockstep : makespan 4.07s | gen 利用率 69.6% | trainer 30.4%
    解耦 C=4 : makespan 2.94s | speedup 1.39x | gen 96.5% | trainer 42.1% | 同步 10 次
    staleness: mean=0.92 max=1（生成主导 → buffer 积不起来，off-policy 度天然低）
    反事实（引擎不 batching，G=0.63s/批）: lockstep makespan 8.8s
    → batching 把 makespan 从 8.8s 压到 4.1s（2.2x）——第一杠杆是批量化，不是解耦

[6] self-check
    ✓ G 随 L 线性：R²=0.9958 > 0.99
    ✓ B≥4 单调压缩至 2.60x ≥ 1.33（B=16 vs B=1）
    ✓ B=1↔2 在噪声带内：单条 decode 固定开销的方差与本征差相当（见 tutorial §4.2）
    ✓ 每 token 墙钟：生成 0.350 > 训练 0.040 ms/token（8.7x，串行 L 步 vs 并行 1 遍）
    ✓ S < T（0.09 < 103 ms）
    ✓ 生成主导区解耦 speedup=1.39x ∈ (1.0, 1.6)：buffer 买不到吞吐

====================================================================
✅ self-check passed: 线性 / 压缩 / G>T / S<T / 解耦增益有限
====================================================================

takeaway: 实测确认 L0 的三条口头声明——G∝L（b=0.350 ms/token）、
          batching 压缩 2.6x、同批 G/T=2.3。解耦的价值在
          staleness 管理与弹性，不在吞吐；吞吐的第一杠杆是 batching 与
          更快的引擎（L2 接 SGLang/vLLM，对照 nano-vllm-sglang）。
```

---

## 2. [0] 探针模型：为什么先训练，「真实」指什么

模型本身 ~50 行（`L1_real_gen_train_timing.py` 的 `TinyGPT`）：4 层、d=128、
8 头、权重绑定的 char-level GPT，797,184 参数。两个工程选择值得说：

**KV cache 是手写的**（`Attention.forward` 的双路：全序列带因果掩码走训练，
单步拼缓存走 decode）。这不是装饰——L1 要测的正是「带缓存的串行 decode」
的墙钟，缓存实现错了（比如每步重算全序列），测出来的就不是真实引擎的物理。

**语料是 slime 自己的 README**。小模型学的就是它所在框架的架构描述——
续写 `'g framework for RL scaling, providing two core capabilities:...'`
是对 README 首句的真实采样。1200 步后 loss=0.022 nats/char：强烈过拟合，
这是**有意**的——我们要的是可读的 demo rollout 与真实的训练步，不是一个好模型。
预训练耗时 ~47s，不计入任何 G/T 测量（探针的「制造成本」与「使用成本」分开）。

确定性检查（`demo == demo2` 逐 token）保证后面所有计时都建立在可复现的计算上：
同一 seed 下 torch CPU 单线程的算子序列逐位确定，浮动的只有墙钟。

---

## 3. [1] G ∝ L：串行 decode 的墙钟

### 3.1 读数

B=1、greedy、KV cache，response 长度 L 从 16 扫到 256：G 从 4.4ms 涨到 88.5ms，
线性拟合 **G = -2.8 + 0.350·L (ms)，R² = 0.9958**。L0 里那句口头的
「G 随 response 长度线性涨」，现在是一个斜率：**每 token 0.35ms 的串行前向价**。

### 3.2 机制：为什么是线性，以及线性从哪里开始漏

每个 token 都是一次完整的前向，且必须等上一个 token 出来（autoregressive 的
因果性）。带 KV cache 时，第 t 步的代价 ≈ **读全部权重（常数）+ 读 t 个历史
KV（随 t 涨）**。于是：

```
G(L) ≈ Σ_t (c0 + c1·t) = c0·L + c1·L²/2
```

线性只是「c0 项主导」的表象。本实验里 c1 项已经露头：ms/token 从 L=16 的 0.27
爬到 L=256 的 0.35（+30%）。用两点外推（run1 数字）：c1 ≈ 5.9e-4 ms/token²，
c0 ≈ 0.271 ms/token，**KV 读与权重读平起平坐的交点约在 L ≈ 900 token**——
但这个两点外推对噪声极敏感（三遍连跑给出 ~0.9k–3.8k 的区间，见 §8），
只能当量级估计。真正的教训是方向性的：response 越长，KV 项越凶，这正是真实
系统里 GQA/MQA、KV 量化、paged attention 存在的理由（它们都在砍 c1）。

截距 a=-2.8ms 是拟合产物（负值 = 噪声范围内），其物理对应是「每批固定开销」：
Python 循环、内核启动、缓存初始化。它不随 L 变，所以是 batching 摊薄的对象（§4）。

### 3.3 测量方法本身：round-robin + 中位数

共享机器上，测量协议是实验的一部分。第一版代码按 L 逐块测量（每个 L 连测 7 次
再换下一个），结果 B=16 块的墙钟在两次完整运行间漂了 2 倍（14.6→28.3ms）——
机器的慢漂移/负载尖峰整块砸在一个配置头上。修复是 `sweep_median`：
**每一轮按序测完所有配置，再重复，每配置取中位数**——漂移被摊到所有配置上，
谁也不占便宜。warmup 2 次丢掉冷启动。本教程所有计时都是这个协议。

---

## 4. [2] batching 压缩：第一杠杆

### 4.1 读数与机制

同样 16 条 rollout × L=128，只变批大小：单条 G 从 39.3ms 压到 15.1ms，
**压缩 2.60x**。机制：decode 每步要读全部权重——批内 B 条序列**共享这一次读**，
每条序列摊到的权重流量 ÷B；每步的 Python/内核固定开销同理摊薄。
这就是「真实引擎的第一杠杆」：slime 的 rollout 侧用 SGLang 大并发批处理，
不是为了「并行好看」，是因为不批就贵。

### 4.2 B=1↔2 是噪声带（run1 就在输出里）

run1 的 B=2 是 48.26ms，比 B=1 还慢（0.81x）；另两遍是 36.24 / 43.37ms
（1.05x / 0.88x）。B=1 与 B=2 的**本征差接近零**（该模型尺寸下向量化收益
从 B≥4 才起步），而单条 decode 固定开销的测量方差有 ±15%——信噪比 <1 的
地方不配谈单调性。所以 self-check 只断言结构性质：**B≥4 单调压缩**、
**B=16 显著压缩（≥1.33x）**、任何批大小不灾难性变慢（>1.5x）。
教训：断言要落在结构性质上，不要落在噪声带上——实验中两次被这条咬到
（第一次还误以为是 B=2 的物理异常，探针实验证明是线程调度，见下）。

### 4.3 探针：批与线程是耦合的旋钮

同一个扫描在 threads=8 下重测：B1 52.3 / B2 86.8 / B4 54.7 / B8 32.6 / B16 21.0——
**本模型尺寸下多线程在每个批大小都更慢**。matmul 太小时，线程调度开销
吃掉甚至超过并行收益；「多线程赚钱」的拐点取决于 模型×批 的形状。
（第一版代码全程 threads=8，B=2 出现稳定的 0.87x 凹陷——换到 threads=1
后凹陷消失，证明那是调度伪影而非 batching 物理。）GPU 引擎干脆用数千线程的
SIMD 重写这条曲线：权重读被摊到海量线程上，batching 的收益曲线完全变形——
那是 L2 与 nano-vllm-sglang 的地盘。

---

## 5. [3] T(L) 与 G/T：并行一遍 vs 串行 L 步

训练步 = 同一批 16 条 rollout 上的 fwd+bwd+Adam（teacher forcing，所有 token
一次并行前向）。T 也随 L 涨（**T = -7.2 + 1.046·L ms，R²=0.9743**），
但注意 ms/token(全批)：0.032→0.059——**每个 token 的训练墙钟比生成便宜一个
数量级**（run1：0.350 vs 0.040 ms/token，**8.7x**）。原因一句话：
训练把 16×(32+L) 个 token 摊进**一次**前向/反向；生成要走 **L 步串行**前向。

同一批口径（16×L=128）：**G_batch=236ms vs T=103ms，G/T = 2.3**。
为什么批级比值（2.3）小于每 token 比值（8.7）？因为 batching 已经把 G 摊薄了
2.6x——而训练本来就是「批处理」的。这两个数字合起来才是完整的账：

- 每 token 口径：串行 decode 的物理代价（8.7x）——机制；
- 同一批口径：管线实际消费的时间比（2.3x）——工程现状（已 batching）。

T 的 R²（0.974）比 G（0.996）低，且 ms/token 从 0.032 爬到 0.059：
训练的超线性来自 attention 的 O(L²) 项（序列 32+L，反向再翻一遍）。
L=256 时它已经可见——真实长上下文训练里它是序列并行 / context parallel
存在的理由（nano-megatron L2/L3 的方向）。

还有一个口径要声明：真实 RL 的 T 要乘 PPO epoch 数（同批数据走 k 遍——
我们的 [nano-verl L1](../nano-verl/L1_minimal_ppo.py) 实现取 `N_EPOCHS = 4`）；
但 k 是小常数，而 G 随任务的 L 无界增长（长程 agent 的 response 以千计）
——「rollout 主导」的 regime 结论对 k 稳健。

---

## 6. [4] S：权重同步

trainer 侧全部参数（3.0MB）拷贝到 rollout 侧缓冲：**0.09ms，~31GB/s**，
S/T = 0.001。本机口径下同步不可见。真实 slime 里这一步是
**Megatron→SGLang 的跨引擎权重传输**（README L28 把 "weight synchronization"
与 "high-throughput rollout" 并列为生产验证项——这两个词出现在 README 里，
正因为它们是瓶颈），涉及 GPU 间传输、层间流水线与增量（delta）同步——
源码级对照是 L3 的事 `[TODO: verify source]`。L1 只确立一件事：
同步代价与「参数量 × 传输带宽」有关，与 L 无关——它不随 response 变长而恶化。

---

## 7. [5] 实测值灌回 L0 模拟器：解耦在真实 regime 里值多少

把实测 G=236ms / T=103ms / S=0.09ms 灌进 L0 的 `sim_lockstep` / `sim_decoupled`
（直接 import，一行不改——L0 模拟器的接口就是为这一刻设计的）：

- lockstep makespan 4.07s → 解耦 2.94s，**speedup 1.39x**；
- 解耦后 trainer 利用率 42.1%——**过半时间空等样本**（生成主导的直接读数）；
- staleness mean=0.92 max=1：buffer 积不起来，off-policy 度天然低——
  L0 里 C=1 闸门是人为背压出来的，这里它是 regime 的自然结果；
- 反事实：引擎不 batching（G=16×单条=0.63s/批）时 makespan 8.8s——
  **batching 一个杠杆压下 2.2x，解耦只给 1.39x**。

这与 L0 反例[4]a（G=20, T=1 时解耦只有 1.05x）逐字呼应：
**解耦治不了 G，它管的是 staleness 与弹性；吞吐的第一杠杆是 batching，
第二杠杆是更快的引擎（L2）。** 而 trainer 利用率 42% 这个数字给出了
L2 的量化目标：要把 trainer 拉到 90%，需 G ≤ T/0.9 ≈ 114ms，
即再压缩 ~2.1x——更大的批、更快的 kernel、或更短的 response。

---

## 8. 计时区间（连跑 3 遍校准，2026-08-06 同日同机）

三遍均 EXIT=0、self-check 全绿；非计时行逐字节一致（mask 后 md5 同为
`9cd878fd25855f3210b515032b7f5f85`）。计时行区间（run1 / 三遍范围）：

| 量 | run1 | 三遍区间 |
|---|---|---|
| 预训练耗时 | 46.6s | 44.6–46.7s |
| G 斜率 b（ms/token） | 0.350 | 0.330–0.350 |
| G 拟合 R² | 0.9958 | 0.9958–0.9985 |
| 单条 G @B=1（ms） | 39.26 | 38.12–39.26 |
| 单条 G @B=2（ms，噪声带） | 48.26 | 36.24–48.26 |
| 单条 G @B=16（ms） | 15.09 | 14.10–15.09 |
| 压缩率 | 2.60x | 2.60–2.71x |
| threads=8 探针 @B=16（ms） | 21.0 | 17.7–21.0 |
| T @L=128（ms） | 102.9 | 97.8–102.9 |
| G_batch @16×128（ms） | 236 | 232–236 |
| G/T | 2.3 | 2.3–2.4 |
| 每 token 生成/训练比 | 8.7x | 8.6–8.7x |
| S（ms） | 0.09 | 0.09–0.10 |
| 解耦 speedup | 1.39x | 1.37–1.39x |
| 解耦 trainer 利用率 | 42.1% | 40.2–42.1% |
| 反事实 makespan（不 batching） | 8.8s | 8.5–8.8s |

机制结论全部不变：线性（R²>0.995）、压缩方向、G>T、解耦增益有限。
KV 交点外推（§3.2）对噪声敏感：三遍给 ~0.9k / 2.3k / 3.8k token——
只取量级（10³–10⁴），不取精确值。

**定稿后第 4 遍确认跑**（文件定稿后复验）：b=0.287 ms/token、压缩 2.41x、
G/T=2.2、解耦 speedup 1.39x——压缩率与 G/T 略出上表三遍区间。如实记录：
三遍区间是**样本值，不是保证区间**；共享机器上计时分布的尾部比三样本
估计的更宽。断言全部仍为绿（它们只落在结构性质上），机制结论不变。

---

## 9. 与真实 slime 的对应，以及外推边界

| 本实验 | slime（README 2026-08-06 快照锚点） |
|---|---|
| `TinyGPT.generate`（批 greedy decode + KV cache） | rollout（SGLang + router，README L91） |
| `one_step`（fwd+bwd+Adam） | training（Megatron，README L90） |
| `dst.copy_(p.data)` 参数拷贝 | weight synchronization（README L28/L90 "synchronizes parameters to the rollout module after training"） |
| L0 `sim_decoupled` 的 FIFO buffer | data buffer（README L92 "A bridge module"） |
| 固定长度 greedy rollout | 真实 rollout：采样 + 变长 + 早停 + 多轮 agent 循环（L91 "multi-turn loops, tool calls..."） |

**可外推的**（机制，与机器无关）：G∝L 的串行结构（GPU 上每 token 读全部权重，
更凶——memory-bound）；batching 摊薄权重读；训练每 token 墙钟远低于生成；
生成主导 regime 里解耦增益有限、staleness 天然低。**不可外推的**（本机常数）：
0.35ms/token 的绝对值；S=memcpy 与跨引擎 GPU 传输的差距；threads×batch 的
拐点位置；以及我们的「固定长度」口径——真实 rollout 有 EOS 早停，平均 L 更短，
但长尾更长（长程任务），变长由 continuous batching 承接（L2）。
生态参照：README L130 的 Relax（基于 slime）"fully-async training at
configurable staleness"——把 staleness 直接做成可配置旋钮，正是 L0/L1
这套账本在工业侧的延伸。

---

## 10. 费曼：讲给外行听

后厨（rollout 引擎）做菜必须一道一道炒（decode 串行）；一次给 16 桌备料
（batching）比 16 次单做快 2.6 倍——锅和火（权重读）是共享的。
传菜员（trainer）上菜快得多，所以大半时间（58%）站在后厨门口等。
把点菜和炒菜拆成两个班次（解耦）只让整体快 1.39 倍——**因为瓶颈是炒菜，
不是传菜**。要让客人吃上热菜，办法是更大的锅（batching）和更猛的灶
（更快的引擎），不是多雇传菜员。

类比边界：真实后厨的菜不会「过时」，但 RL 的样本会——传菜员等得越久，
端出去的是越旧菜谱（权重版本）炒的菜，这就是 staleness（L0 的账本）。

---

## 11. 思考题

1. §3.2 的 KV 交点外推用了 L=16 与 L=256 两点。如果给你 G(512) 的实测值，
   你会如何改进估计？为什么真实大模型上交点来得更早（提示：c1 与
   层数×头维成正比，c0 与总参数量成正比，两者的缩放速度不同）？
2. 生成主导 regime 里解耦只值 1.39x——那 slime 为什么还要解耦？
   （提示：弹性、故障隔离、staleness 的**可控性**——Relax 的
   configurable staleness；以及 trainer 不再被 rollout 的失败重启拖住。）
3. 把 greedy 换成 T=0.7 采样，G 的测量值会变吗？rollout 的**长度分布**会吗？
   哪个量才是管线吞吐的真正输入？（提示：计算形状不变 → G 不变；
   EOS 早停使平均 L 缩短——管线消费的是长度分布，不是单条最坏值。）
4. 用 run1 的数字验算：解耦 trainer 利用率 42.1% ≈ T/G_batch = 103/236。
   若要把利用率拉到 90%，G 要压到多少？这对应现实中哪些手段？
   （答案：G ≤ ~114ms，再压缩 ~2.1x——更大批 / 更快引擎 / 更短 response。）

---

## 12. 反例与边界

1. **绝对数字不是 GPU 的预言。** 0.35ms/token 是 CPU 小模型数；GPU 大模型上
   每 token 要读 GB 级权重，绝对值由显存带宽钉死（nano-vllm-sglang 有实测）。
   本节卖的是斜率的存在与方向，不是斜率的值。
2. **线性不是定律，是区间现象。** L≤256 内线性项主导（R² 0.996）；KV 二次项
   已在爬（ms/token +30%），长上下文下它会接管——外推 G 到千 token 级时，
   线性拟合会**低估**真实代价。
3. **batching 压缩有天花板。** 本实验 B=16 还在赚（2.6x），因为瓶颈是
   固定开销与权重读的摊薄；一旦进入 compute-bound（GPU 大批量区），
   压缩饱和，且每请求延迟会先付出代价——同一机制的另一面，
   见 nano-vllm-sglang L1 的 batching 实测。
4. **固定长度 greedy 是「最坏长度」口径。** 真实 rollout 采样 + 早停，
   平均更短、长尾更长；变长批处理（continuous batching）是 L2 的内容，
   本节的 G 应读作「每条都生成满 L 时的上界式估计」。

---

## 13. 阶梯预告与交叉引用

- **L2**：把自写的 `generate()` 换成真实推理引擎（SGLang/vLLM）做 rollout，
  对照 nano-vllm-sglang 的代价模型；真机部分需在真实 GPU/多机环境验证
  `[TODO: verify on real system]`。
- **L3**：对照 slime 源码——data buffer 实现、权重同步（delta weight sync）、
  rollout 调度与 update 时机 `[TODO: verify source]`。
- 交叉引用：[nano-verl L1](../nano-verl/tutorial_L1.md)（off-policy 偏差的算法侧
  修正：importance sampling——本节测出的 staleness 是它的输入）、
  [nano-vllm-sglang](../../03-data-distributed-rsi/nano-vllm-sglang/)（引擎为什么快）、
  [nano-megatron L1](../../02-pretraining-cpt/nano-megatron/tutorial_L1.md)（同为
  「真实测量」范式：all-reduce 墙钟）、[L0](tutorial_L0.md)（本节灌回的模拟器）。

---

## 14. 溯源与校准

- **训练语料**：`https://raw.githubusercontent.com/THUDM/slime/main/README.md`，
  2026-08-06 curl 抓取，全文件 19093 bytes，
  sha256 `8989972638bb73f06ecd4bfb3092ce49ca42f55ff14f660cdaf28a3d37c93d21`；
  切片 = L9–17 + L19–28 + L84–92（3362 bytes，嵌入代码 `CORPUS` 常量）；
  两块 junction 空行做过归一化（L17/L19 交界删一个空行、L84 交界前插一个空行），
  内容行逐字一致（独立 diff 核验）。
- **README 锚点**（2026-08-06 快照）：L28（high-throughput rollout, weight
  synchronization）、L67（vision blog）、L84（Architecture Overview）、
  L90–92（training / rollout / data buffer 三模块描述）、L130（Relax
  fully-async at configurable staleness）。与 L0 于 2026-08-04 记录的行号
  **逐项一致，无漂移**（README 期间有内容更新：GLM-5.2 / Qwen3.6 等生态条目）。
- **运行环境**：Apple M5 Pro / Python 3.13.13 / torch 2.13.0 / CPU /
  `torch.set_num_threads(1)` / seed=7 / greedy。总运行时长 ~2.5 分钟。
- **测量协议**：warmup 2 + round-robin ×5 取中位数（§3.3）；三遍连跑区间见 §8；
  mask 计时行后 md5 `9cd878fd25855f3210b515032b7f5f85` 三遍相同。
- **self-check 断言设计史**（如实记录，两轮真实调试）：
  (a) 初版全程 threads=8，B=2 稳定 0.87x——误判为 batching 物理，
  探针实验（threads=1 vs 8）证明是线程调度伪影 → 基线改单线程；
  (b) 「单条 G 随 B 单调不增」与「任何配置不慢于 B=1 的 15%」两条断言
  先后被 B=1↔2 噪声带咬穿（run3 / run6 实测越界）→ 断言改为结构性质
  （B≥4 单调、B=16 显著压缩、无灾难性回归）。
  教训：**断言落在结构上，不落在噪声带上**——与 nano-opd L1 的
  「断言落在理论预测上，不落在经验猜测上」同构。
- **未核验项**：slime 源码级细节（data buffer / delta weight sync 实现）
  留 L3 `[TODO: verify source]`；本机无 GPU，所有 GPU 相关表述为机制转述。
