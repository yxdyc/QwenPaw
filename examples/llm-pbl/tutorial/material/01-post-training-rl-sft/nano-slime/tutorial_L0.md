# nano-slime · L0 教程：采样/训练解耦——data buffer、版本化权重与 staleness 账本

> **本节目标（L0）**：用 ~160 行纯标准库的确定性离散事件模拟，抓住 RL 后训练
> 系统（slime 一类）数据通路的最小机制：**generator 与 trainer 解耦、用
> FIFO data buffer 连接、权重按版本同步、每个样本带 staleness（off-policy 度）**，
> 并量化 buffer 容量这个核心旋钮到底买得到什么、买不到什么。
> **前置**：知道 RL 后训练「先采样一批轨迹、再训一步」的循环即可；
> 读过 nano-verl L0（actor-learner 流水线）与 nano-vllm-sglang L0（推理引擎
> 为什么快）会更顺。
> **本节 K+1**：从「知道 RL 要 rollout」到「说得出 rollout 与 train 为什么必须
> 解耦、解耦的代价是什么、buffer 尺寸在权衡什么」。

---

## 1. 问题：一轮 RL 后训练的两个脾气

RL 后训练（RLHF/RLVR）每一轮做两件事：

1. **rollout**：用当前 policy 生成一大批轨迹（prompt → response，可能带工具调用、
   多轮交互），并算出 reward；
2. **train**：拿这批轨迹做一次策略更新（PPO/GRPO 族，见 nano-verl L1）。

两块开销的脾气完全不同。decode 是自回归的——每个 token 一次串行前向
（nano-vllm-sglang L0 §1），所以生成时间 G 随 response 长度**线性涨**；
而训练一步只对这批定量数据做常数次前向/反向，耗时 T 相对平稳。
长 response 的 RLVR 场景里，G 往往是绝对大头。

最朴素的组织方式是 **lockstep（同步串行）**：生成一批 → 同步权重 → 训练一步，
循环。问题一目了然：**快的一方必须空等慢的一方**。生成时训练资源闲着，
训练时生成资源闲着，权重同步期间两边都不干正事。

两个改进方向：**把引擎本身做快**（nano-vllm-sglang L0 的主题：KV cache /
continuous batching / 分页都在压缩 G），以及**把两阶段解耦**——本节做后者。
nano-verl L0 演示过「时间线上的流水线重叠」；slime 走得更彻底：两边各跑各的，
中间用一条 **data buffer** 连接，代价是一种新的账——**staleness**。

---

## 2. 先跑起来

文件：`L0_data_buffer_decouple.py`，纯标准库，CPU 即跑。

```bash
$ python3 L0_data_buffer_decouple.py
```

真实输出（离散事件模拟，无随机，跨运行确定）：

```text
================================================================
nano-slime L0 — 采样/训练解耦：data buffer + 版本化权重
================================================================

配置（toy 时间单位）: G=4.0 生成一批 | T=6.0 训练一步 | S=1.0 权重同步 | B=12 批
为何生成是大头：decode 每个 token 都是一次串行前向，G 随 response 长度线性涨；
训练一步只对定量数据做常数次前向/反向。长 response 的 RLVR 里常是 generate 主导。

[1] lockstep（同步串行）: cycle = G+S+T = 11
    makespan = 132 | gen 利用率 36.4% | trainer 利用率 54.5%
    —— 每个 cycle 有 9.1% 的时间两边都在空转（sync + 互等）

[2] 解耦（buffer C=4）: makespan = 76 | speedup = 1.74x
    gen 利用率 72.4% | trainer 利用率 94.7% | 权重同步 7 次
    消费事件（批#, 时刻, 生成时版本, staleness）:
      # 0 t=  4.0 v_gen=0 staleness=0
      # 1 t= 10.0 v_gen=0 staleness=1
      # 2 t= 16.0 v_gen=0 staleness=2
      # 3 t= 22.0 v_gen=1 staleness=2
      # 4 t= 28.0 v_gen=2 staleness=2
      # 5 t= 34.0 v_gen=3 staleness=2  ...
    staleness: mean=2.33 max=4 —— buffer 里的样本是旧权重生的

[3] buffer 容量扫描（稳态由较慢一方决定）:
     C | makespan | mean stale | max stale
     1 |       76 |       0.92 |         1
     2 |       76 |       1.75 |         2
     4 |       76 |       2.33 |         4
     8 |       76 |       2.33 |         4
    C=1 把 staleness 钳在 ≤1（背压即 off-policy 闸门）；makespan 几乎不动

[4] 反例：解耦不是万灵药
    a) 生成才是瓶颈（G=20, T=1）: lockstep 264 -> 解耦 251（speedup 1.05x，几乎白忙）
       => 该让生成引擎本身更快（SGLang/vLLM，见 nano-vllm-sglang L0），而不是加 buffer
    b) 批时长波动 [2,4,9]: C=1 makespan 90（被慢批卡住） vs C=8 makespan 75（buffer 吸收波动）
       弹性才是大 buffer 的真实收益；代价是 staleness 0.92 -> 1.67

================================================================
✅ self-check passed: 事件推进无死锁 / staleness 非负 / C=1 上界 / 容量单调 / 两个反例
================================================================

takeaway: 解耦把每批耗时从 G+S+T 压到 max(较慢一方)，代价是样本带 staleness；
          buffer 容量限 staleness 上界、吸收波动，但不改稳态吞吐。
          真实 slime 里 trainer=Megatron、rollout=SGLang、中间就是 Data Buffer；
          IS ratio 可修正同一状态上的动作分布差异，但不能抹掉陈旧前缀；见 nano-verl L1。
```

> **toy 口径声明**：G/T/S 是**模拟时间常数**，不是真机测量；全部数字是这个
> 离散事件模拟的算术输出，用来量化**权衡的结构**（什么随什么涨、什么不随什么动），
> 不是 benchmark。真实吞吐与 slime 源码级对照留 L1/L2/L3。与 nano-megatron L0
> 「用账本代替实测」的口径相同。

**L0 基线指标（toy metric）**：同一组配置（G=4, T=6, S=1, B=12），lockstep
makespan 132 → 解耦（C=4）76，speedup 1.74x；trainer 利用率 54.5% → 94.7%；
代价是 staleness mean 2.33 / max 4。buffer 容量从 1 加到 8，makespan 不动（76），
mean staleness 从 0.92 涨到 2.33 后饱和。

---

## 3. 模拟器：三条规则构成数据通路

整个模拟只有两个 actor 加一条 FIFO，规则与真实系统的数据通路同构：

```python
# trainer：完成一步 => version+1；空闲且 buffer 非空就取最旧样本开训
# generator：开新批前，若手中版本落后于 version，花 S 拉新权重（计入忙碌）
# 背压：buffer 就绪数 + 在途批数 >= C 时 generator 停手
```

事件循环本身有一条值得学的纪律——**先在当前时刻尝试「启动」动作，再推进到
下一个完成事件**：

```python
while consumed < B:
    if trn_until is None and buf:        # trainer 取批（记录 staleness）
        ...
    if gen_until is None and started < B and len(buf) + in_flight < C:
        ...                              # generator 开新批（可能先付 sync）
    nxt = min(完成事件们)                # 然后才推进时间
    t = nxt
    ...                                  # 处理完成事件：buf.append / version+1
```

反过来写（先推进时间再尝试启动）会在 t=0 撞上「无事件可推进」的死锁——
任何离散事件模拟都要先想清楚：**每个时刻，动作与事件的处理顺序是什么**。
（nano-ray L0 §5.1 讲的是并发触发的时序；这里是单线程事件循环的时序，
同一种「顺序即正确性」的教训。）

**staleness 的定义**落在 trainer 取批的那一刻：

```python
staleness = version(消费时刻) - version(该批生成时所用)
```

即：这个样本的 policy 落后当前 policy 几步。它是解耦的**核心代价**，下面全程记账。

---

## 4. 机制一：解耦把「每批 G+S+T」变成「max(较慢一方)」

lockstep 每批一个 cycle：`G + S + T = 11`，12 批 = 132。两边的利用率都低：
generator 36.4%、trainer 54.5%——**每个 cycle 近一半时间在互等**。

解耦后（[2]，C=4）：makespan 76，恰好等于 `4 + 12×6`——第一批 t=4 就绪，
此后 trainer 每 6 个单位消费一批，直到第 12 步结束。**稳态节奏完全由较慢的
一方（trainer，T=6）决定**，generator（G=4）再快也只是把 buffer 填满然后被
背压拦住。利用率翻转：trainer 94.7%（几乎不空转），generator 72.4%
（被背压停掉的时间就是它「太快」的部分）。

这就是解耦的全部收益来源：**把「串行相加」变成「取 max」**。
理论上限是 makespan → max(G, T)×B + 尾部，本例 132 → 72+4 = 76，已贴近上限；
speedup 上界 = (G+S+T)/max(G,T) = 11/6 ≈ 1.83x，实测 1.74x（差额是首批 ramp
与 7 次权重同步）。

---

## 5. 机制二：版本化权重——staleness 是怎么生出来的

看 [2] 的消费事件表，逐行读：

- `#0 t=4 v_gen=0 staleness=0`：第一批由初始权重生成、被第一步训练消费，新鲜；
- `#1 t=10 v_gen=0 staleness=1`：第二批也是 v0 生的（generator 开批时 trainer
  还没完成过更新），但消费它时 trainer 已经更新过 1 次——**落后 1 步**；
- `#2 t=16 v_gen=0 staleness=2`：同上，落后 2 步；
- `#3 t=22 v_gen=1 staleness=2`：generator 终于拉了一次新权重（version=1）
  才开始生成，但 buffer 积压 + 生成也要时间，消费时仍落后 2 步。

staleness 不需要任何人犯错，它是**结构性的**：样本必须在生成之后才能被训练，
而训练一直在推进 version——只要两边异步，样本落地时天然落后。max=4 意味着
最旧的一批样本是用「落后 4 步的 policy」采的。这对 RL 是真实的正确性问题：
样本上的 log-prob 来自旧 policy，直接进梯度会产生 off-policy 偏差。真实系统的
两条对策：**算法侧**用 importance sampling ratio 修正相同 prefix 上的动作概率差异
（但不能抹掉旧策略造成的 prefix/state 分布漂移，见 [nano-verl L1](../nano-verl/tutorial_L1.md#importance-sampling为什么旧样本还能再用几轮)），
**系统侧**限制 staleness 上界——这就是下一个机制。

---

## 6. 机制三：buffer 容量——买不到吞吐，只买到弹性与 off-policy 度

[3] 的容量扫描是本节最重要的一张表：

| C | makespan | mean staleness | max staleness |
|---|----------|----------------|---------------|
| 1 | 76 | 0.92 | 1 |
| 2 | 76 | 1.75 | 2 |
| 4 | 76 | 2.33 | 4 |
| 8 | 76 | 2.33 | 4 |

三个结论，每个都能从机制推出来：

1. **makespan 恒为 76，buffer 大小买不到吞吐**。稳态吞吐被较慢一方（trainer，
   每 T=6 一批）钉死；buffer 只能挪相移，不能提速率。想提吞吐只有两条路：
   让慢的一方变快（换硬件/优化训练），或换结构（加并行 trainer——超出 L0）。
2. **staleness 随 C 单调上升，且会饱和**：C=4 与 C=8 完全相同——因为稳态下
   buffer 积压根本到不了 8（generator 每 4 单位产一批、trainer 每 6 单位消一批，
   积压上界约 4）。**超过稳态积压的容量是死重**。
3. **C=1 是 off-policy 闸门**：背压让 generator 最多领先 trainer 一步，
   staleness 被钳在 ≤1（代码里的 assert 验证）。想要近 on-policy，就把 buffer 收小——
   代价是失去弹性（§7 反例 b）。

一句话：**buffer 容量是「弹性 + off-policy 度」的旋钮，不是吞吐的旋钮。**

---

## 7. 反例：两个「想当然」

- **「解耦总能加速」**——[4]a：生成才是瓶颈（G=20, T=1）时，lockstep 264 →
  解耦 251，speedup 只有 **1.05x**。trainer 快得用不完样本，buffer 常年空着，
  staleness ≈ 0，解耦除了省掉一点互等什么都赚不到。瓶颈在 G，解药是把生成
  引擎做快——KV cache / continuous batching / 分页（nano-vllm-sglang L0 的四件套），
  这正是 slime 用 SGLang 作 rollout 引擎的原因。**先诊断哪边慢，再决定买什么药。**
- **「buffer 越大越好」**——[4]b 的另一半：把批时长改成确定性的波动序列
  [2,4,9]（均值 5，仍 < T=6），C=1 的 makespan 从 76 恶化到 90（每个 9 单位的
  慢批都让 trainer 干等），C=8 则维持在 75（慢批之前攒下的存货顶上去）。
  大 buffer 的真实收益是**吸收波动**；但同一张表里 staleness 从 0.92 涨到 1.67——
  弹性不是免费的，它用 off-policy 度付账。

---

## 8. 与真实 slime 的对应（README 一手来源）

slime（`github.com/THUDM/slime`）是完整的 RL 后训练框架，不只做采样——
但它的骨架正是本节模拟的这条数据通路。README（main 分支，2026-08-04 快照）
「Architecture Overview」一节原文（L90–92）：

> - **training (Megatron)**: Responsible for the main training process, reads data
>   from the Data Buffer, and synchronizes parameters to the rollout module after training.
> - **rollout (SGLang + router)**: Generates new data (including rewards/verifier
>   outputs) and stores it in the Data Buffer.
> - **data buffer**: A bridge module that manages prompt initialization, custom data,
>   and rollout generation methods (including agentic workflows that produce samples
>   through the same interface).

逐条对上：

| nano 实现 | 真实 slime | 说明 |
|-----------|-----------|------|
| generator + FIFO buffer | rollout (SGLang + router) + data buffer | 生成数据存入 buffer；真系统还带 reward/verifier 输出 |
| trainer 每步消费一批 | training (Megatron) 从 Data Buffer 读数据 | README L90 原话 |
| version+1 后 generator 拉新权重（代价 S） | 训练后同步参数到 rollout（weight synchronization，README L28） | 真实同步是 GB 级参数传输，还有 delta weight sync 等优化 `[TODO: verify source]` |
| staleness 记账 | 生态内已把 staleness 当一等旋钮：基于 slime 的 Relax 自称支持 "fully-async training at configurable staleness"（README L130 引述） | 佐证 staleness 是这类系统的核心设计变量 |
| 未覆盖 | reward/verifier 路径、agentic 多轮 generate、rollout-only/train-only 调试 | 完整框架的另一半，L2/L3 主题 `[TODO: verify source]` |

slime 源码级的对照（data buffer 的数据结构、权重同步时机与 update barrier、
SGLang server 的部署形态）全部留到 L2/L3，标 `[TODO: verify source]`。

---

## 9. 费曼：讲给外行听

**类比：回转寿司。**

- **厨师 = rollout 引擎**：不停做寿司（样本），只要传送带还有空位就继续做；
- **客人 = trainer**：按自己的节奏取寿司、吃完给反馈（一步训练）；
- **传送带 = data buffer**：长度就是容量 C；
- **换配方 = 权重更新**：客人每吃完一轮，反馈让厨师的配方升一版（version+1）；
  厨师**开始做下一盘时**才用新配方（生成批次边界拉权重），正在做的那盘不打断；
- **staleness = 过时度**：你手里这盘寿司是几版配方之前做的。带子转得越慢、
  带子越长，拿到手的寿司相对当前配方就越旧；
- **带子加长不能让客人吃得更快**（稳态由吃得慢的一方决定），只能让客人
  「永远有得拿」——以及在换配方时，带上积着的那几盘全是旧配方做的。

一句话版本：**解耦 = 厨师和客人各干各的，中间一条传送带；带子长度决定
你拿到多旧的寿司，不决定你吃多快。**

---

## 10. 思考题

1. 本节 trainer 按 FIFO 消费**最旧**的样本。如果改成「staleness 超过阈值就
   丢弃重采」（一些 replay-buffer 方案的做法），你失去什么、得到什么？
   （提示：被丢的样本消耗的是 generator 的时间——在 [3] 的利用率账本里，
   这部分算谁的浪费？off-policy 度下降能否换回等量的训练质量？）
2. 为什么 generator 在**开始新批次时**才拉新权重，而不是 version 一变化就
   立刻打断当前生成？（提示：打断意味着已完成 token 的计算与 KV cache 作废；
   权重传输本身还要占用引擎。真实引擎如何安排 update 时机 `[TODO: verify]`，
   是 L2/L3 的问题。）
3. 本节配置里 trainer 是慢的一方（T=6 > G=4），staleness 随 C 增长。把配置
   翻成 G=20、T=1（[4]a），稳态下 buffer 占用和 staleness 各是多少？为什么
   瓶颈方不仅决定吞吐、还决定 off-policy 度？（提示：跑一遍 [4]a 看
   staleness 输出；buffer 常年空着说明什么。）

---

## 11. 下一步 L1

L1 把 G/T 从模拟常数变成**实测数字**，验证本节的输入假设：

1. 在真实小模型上（字符级 LSTM 起步，torch 即可）串行 generate N 条 rollout，
   测 wall-time 随 response 长度与 batch 大小的变化——确认「G 随长度线性涨」；
2. 再测 batched generate 对 G 的压缩（权重读取摊薄，nano-vllm-sglang L0 §4
   的机制在小模型上同样成立，只是绝对值小）；
3. 用实测的 G/T 重跑本节模拟器，看解耦收益的预测是否跟着变。

L2 接真实推理引擎（SGLang/vLLM）做 rollout、走 Machine B 真机验证通道
`[TODO: verify on real system]`；L3 对照 slime 源码（data buffer 实现、
权重同步、rollout 调度），摘掉全部 `[TODO: verify source]`。

---

## 12. 溯源

- 运行输出来自本机真实执行：`python3
  L0_data_buffer_decouple.py`，离散事件模拟无随机，连跑两遍逐字一致。
- slime 仓库：<https://github.com/THUDM/slime>。README（main 分支，2026-08-04
  抓取）：架构三模块原文 L90–92、weight synchronization L28、vision blog 链接
  L67、Relax "configurable staleness" 引述 L130。仓库自述 "an LLM post-training
  framework for RL scaling"，并称是 GLM-4.5/4.6/4.7/5/5.1/5.2 等模型发布背后的
  RL 框架——**后者为 README 自述，未独立核验** `[TODO: verify]`。
- vision blog：README 给出 <https://lmsys.org/blog/2025-07-09-slime/>
  （*slime: An SGLang-Native Post-Training Framework for RL Scaling*），
  本轮网络不稳未核验可达性 `[TODO: verify]`。
- 全部时间/吞吐数字为 toy 模拟常数的算术输出（§2 toy 口径声明），非真机
  benchmark；真机数字留 L1/L2。
- 概念交叉引用：nano-verl L0（actor-learner 流水线重叠）、nano-verl L1
  （importance sampling ratio 的局部修正与边界）、nano-vllm-sglang L0
  （推理引擎为何快）——均为本仓库已交付材料。
