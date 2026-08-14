# nano-trinity-rft L0 — 统一 SFT+RL：三组件协同 + 配置驱动

> **本节抓的机制**：Trinity-RFT 为什么能把 SFT 和 RL 放进**同一个框架**——
> 统一的样本数据流 + Explorer/Trainer/Buffer 三组件 + 配置驱动的配方切换。
> 学完你应该能：用一张 config 说清「从全量 SFT 到叠加 RL」要改哪几个字段，
> 并用算术解释「SFT 的天花板由 teacher 数据覆盖画死」。
>
> **前置**：无硬前置；读过 [nano-verl tutorial_L0](../nano-verl/tutorial_L0.md)
> （调度视角）与 [nano-slime tutorial_L0](../nano-slime/tutorial_L0.md)
> （buffer 解耦与 staleness 视角）会更好——本节补的是**数据流与算法切换**视角，
> 三者互不重复。
>
> **权威实现**：[Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT)
> （arXiv:2505.17826，旧地址 `modelscope/Trinity-RFT` 已 301 重定向至此）。

---

## 1. 为什么需要「统一」

真实后训练很少是「纯 SFT」或「纯 RL」：生产配方通常是 SFT 打底 → RL 拔高，
或者两种信号按阶段/比例混合（Trinity 的算法表里甚至有专门的 mix 类条目，
见 §9 引用）。如果 SFT 和 RL 是两套互不相通的系统，每次切换都要重写数据管线、
重搭训练循环、重做 checkpoint 搬运。**统一框架的本质是：让"换算法"变成"换配置"。**

Trinity-RFT 把 RFT 解耦成三个协同组件（README L21–25，引用见 §9）：

- **Explorer** —— 通过 agent-环境交互产生经验数据；
- **Trainer** —— 在数据上最小化 loss 以更新模型权重；
- **Buffer** —— 贯穿 RFT 全生命周期的数据管线。

本节用约 280 行零依赖 Python 把这三组件搭出来，并验证一件事：
**同一份样本记录、同一个 Buffer、同一个 Trainer，只改 config 字段，
训练配方就在 `sft_only` / `rl_only` / `sft_then_rl` / `mix` 之间切换。**

## 2. 先跑起来

```bash
python3 L0_unified_sft_rl_loop.py
```

**Toy 口径声明**：本节没有真实模型与真实数据——策略是 6 context × 4 action 的
表格 softmax，环境是确定性 0/1 reward。所有数字都是这个 toy 的算术输出，
只演机制，不代表任何真实系统的性能。固定 seed，**逐字节可复现**
（本节粘贴输出与程序输出经 diff 机器核对）。

以下为其中一次运行的完整输出（三次运行逐字节一致）：

```text
========================================================================
nano-trinity-rft L0 — 统一 SFT+RL：三组件协同 + 配置驱动
========================================================================
toy 口径: 表格 softmax（6 ctx × 4 act），reward∈{0,1}；teacher 只覆盖 ctx [0, 1, 2]，ctx 3–5 是数据空洞。
config 基底: lr=4.0 capacity=256 max_reuse=2 sft_k=8 rl_k=6

[1] 配置驱动两阶段：sft_then_rl（sft_rounds=3 → rl_rounds=6）
round  mode  new(sft/rl)  batch(sft/rl)  eval_reward  version
   1   sft    24/0         24/0         0.5986      v1
   2   sft    24/0         48/0         0.6053      v2
   3   sft    24/0         48/0         0.6097      v3
   4   rl      0/36        24/24        0.6393      v4
   5   rl      0/36         0/48        0.7007      v5
   6   rl      0/36         0/48        0.8060      v6
   7   rl      0/36         0/48        0.9150      v7
   8   rl      0/36         0/48        0.9546      v8
   9   rl      0/36         0/48        0.9774      v9
=> SFT 阶段在 teacher 覆盖处快爬；RL 阶段把其余 ctx 拉到 ~1.0；
   注意 r4：RL 阶段第一轮的 batch 里仍混着存量 SFT 样本——数据流跨阶段统一。

[2] 配置消融阶梯（同样 9 轮，只改 config 字段，循环代码一字不动）
recipe        eval@r1   eval@r2   eval@r3   eval@r9   per-context final [c0..c5]
sft_only      0.5986    0.6053    0.6097    0.6190    [0.98 0.99 0.98 0.26 0.25 0.26]
rl_only       0.4118    0.5492    0.7129    0.9896    [0.95 1.00 1.00 1.00 1.00 1.00]
sft_then_rl   0.5986    0.6053    0.6097    0.9774    [0.97 0.97 0.99 0.99 0.98 0.96]
=> 同一套 Explorer/Trainer/Buffer，config 字段一换就是不同配方可直接对比；
   SFT 模仿快但被 teacher 覆盖画死，RL 起步慢但能越过 teacher。

[3] mix 模式（每轮同时产出 sft+rl 样本，一个 batch、一步更新）
round  mode  new(sft/rl)  batch(sft/rl)  eval_reward  version
   1   mix    24/36        24/24        0.5656      v1
   2   mix    24/36        24/24        0.6390      v2
   3   mix    24/36        24/24        0.6838      v3
   4   mix    24/36        24/24        0.7217      v4
   5   mix    24/36        24/24        0.8374      v5
   6   mix    24/36        24/24        0.9256      v6
   7   mix    24/36        24/24        0.9639      v7
   8   mix    24/36        24/24        0.9787      v8
=> 两种信号在同一 batch 里叠加——Trinity 的 mix 类算法（如 CHORD）即此形态。

[4] 反例：把 sft_only 加到 15 轮——更多 SFT 轮数填不上空洞
sft_only@15r: eval=0.6217 per-context=[0.98 0.99 0.99 0.26 0.25 0.26]
算术预测: 覆盖 3 ctx→1.0，空洞 3 ctx→0.25 → mean=(3×1.0+3×0.25)/6=0.625
=> SFT 的天花板由 teacher 数据覆盖画死；要突破只能靠 RL 探索（或补数据）。

[账本] [1] 的 buffer: 进入 288 条 | 训练 408 次（max_reuse=2）| 退役 192 条 | 现存 96 条

✅ self-check passed: 两阶段收敛 / warm start 更快 / sft_only 天花板≈0.625 / mix 双信号共存 / 样本复用不超 max_reuse
```

toy 世界只有三样东西：

- **策略**：6 个 context × 4 个 action 的表格 softmax，初始权重全 0（= 均匀分布，
  此时 eval 期望恰为 0.25 = 1/4）；
- **环境**：每个 context 有唯一正确 action，reward ∈ {0, 1}；
- **teacher**：SFT 数据源，但**只覆盖 ctx 0–2**；ctx 3–5 没有任何监督数据，
  是「数据空洞」——只能靠 RL 探索填上。这个不对称是全节戏剧冲突的来源。

评测用固定 rng 的采样式评测（每 context 采 1024 个 action 取平均 reward），
与训练用的 rng 完全分离，保证评测值只反映策略本身。

## 3. 三组件 + 统一数据协议

代码里三个类与 Trinity 的三组件一一对应，它们之间只靠一种记录流动：

```python
class Sample:
    """统一数据协议：SFT 与 RL 样本走同一种记录。
    SFT 样本的 reward 只是登记值（teacher 给的 target 必对，恒 1.0），不参与 loss；
    version = -1 表示 teacher 数据（与策略版本无关，天然无 staleness 问题）。"""
    __slots__ = ("kind", "ctx", "act", "reward", "version", "trains")
```

| toy 组件 | 职责 | 对应 Trinity（README 行号见 §9） |
|----------|------|----------------------------------|
| `Explorer.rl_rollout` | 用**当前策略**与环境交互产经验，记录 `version`（产生时的权重版本） | Explorer（README L23） |
| `Explorer.sft_data` | 从 teacher 取监督对，`version=-1` | SFT 数据路径（`algorithm_type: sft`，README L121） |
| `Buffer` | FIFO 容量 + `select(batch_size)` + `trains` 计数 + `max_reuse` 退役 | Full-Lifecycle Data Pipelines（README L102–105） |
| `Trainer.step` | 同一个优化步消化两种 loss：SFT 交叉熵 + RL REINFORCE | Trainer（README L24）+ 可插拔算法（README L109） |

两个值得停下来看的细节：

**① `version` 字段区分两类数据。** RL 样本产生于某个权重版本，策略更新后它就
「过时」了一点——这正是 [nano-slime L0](../nano-slime/tutorial_L0.md) 里 staleness
账本记的东西。而 SFT 样本来自 teacher，与策略版本无关（`version=-1`），
**天然没有 staleness 问题**。同一个 buffer 里两种数据的「保鲜期」性质完全不同，
这正是统一数据协议要管理的事。

**② RL 的 baseline 取每个 context 组内的 reward 均值**——同一 context 的样本互相对比：
做对的（advantage > 0）被强化，做错的（advantage < 0）被压低；全对或全错的组
advantage 全为 0，该组这一步没有对比信号。「组内相对」正是 GRPO 的核心思想
（arXiv:2402.03300；GRPO 还额外除以组内标准差做归一化，toy 里省略）。
全零 reward 的组没有梯度信号，是这类方法已知的失败模式之一（见思考题 1）。

## 4. 配置驱动：换配方 = 换字段

四个实验用的是**同一个** `run(cfg)` 循环，唯一输入是 config dict：

```python
BASE = dict(lr=4.0, capacity=256, max_reuse=2, sft_k=8, rl_k=6, batch_size=48)

CFG1    = dict(BASE, sft_rounds=3, rl_rounds=6, mix=False)   # [1] 两阶段
CFG_S   = dict(BASE, sft_rounds=9, rl_rounds=0, mix=False)   # [2] 纯 SFT
CFG_R   = dict(BASE, sft_rounds=0, rl_rounds=9, mix=False)   # [2] 纯 RL
CFG_M   = dict(BASE, rounds=8, mix=True)                     # [3] 每轮双信号
```

回答 scaffold 留下的费曼问题——「从全量 SFT 到叠加 RL 要改哪几个字段」：
`sft_rounds`、`rl_rounds`、`mix` 三个字段。全量 SFT = `sft_rounds=N, rl_rounds=0`；
SFT 打底叠 RL = 两者都给值；每轮双信号混合 = `mix=True`。循环代码一字不动。

Trinity 的真实配置体系同理：它把算法抽象成 `algorithm_type` 配置项，
PPO / GRPO / SFT / DPO / mix 类算法并列在同一张支持表里（README L117–123，见 §9）——
「换算法」被设计成「换配置」，这是统一框架对用户最直接的兑现。

## 5. 实验 [1] 逐轮读：两阶段 + 跨阶段的数据流

`[1]` 的表是本节的主干，三个信息层叠在一起：

**SFT 阶段（r1–r3）**：eval 从初始 0.25 跳到 0.5986 并缓慢爬到 0.6097。
第 1 轮就把 teacher 覆盖的 3 个 ctx 拉到 ~0.95（模仿是一步到位的），
之后只有微涨——SFT 在覆盖范围内很快饱和。为什么到不了 0.625+？
因为还有 3 个空洞 ctx 停在 ~0.25：`(3×0.98 + 3×0.25)/6 ≈ 0.62`，与观测吻合。

**RL 阶段（r4–r9）**：eval 从 0.6393 一路爬到 0.9774——空洞被逐个填上。
RL 样本带来的梯度信号只流向策略还没掌握的 ctx，覆盖处则被继续巩固。

**跨阶段数据流（r4 的 batch = 24/24）**：这是最容易被忽略的一行。
r4 已经切换到 RL 模式，但 batch 里还有 24 条 SFT 样本——它们是 r3 进入 buffer、
只被训过一次的存量样本。`select` 按 FIFO 取最旧，于是**上一阶段的 SFT 数据
自然地流进了下一阶段的训练 batch**。r5 起这批样本训满 `max_reuse=2` 退役，
batch 变成 0/48。数据流没有因为「换阶段」被切断——这就是「统一」的具体含义：
阶段是 config 的属性，数据流是连续的。

## 6. 实验 [2]：消融阶梯——SFT 与 RL 的分工

同样 9 轮预算，只换 config，三条曲线画出完整分工：

- **sft_only**：r1 冲到 0.5986 后几乎原地踏步，9 轮后 0.6190。
  per-context 末态 `[0.98 0.99 0.98 | 0.26 0.25 0.26]`——覆盖处近 1.0，
  空洞处纹丝不动。起步快（r1 就 0.5986），但天花板钉死。
- **rl_only**：r1 只有 0.4118（冷启动要先探索），随后持续爬升，
  9 轮后 0.9896，per-context 全线 ≥0.95。起步慢，但没有天花板。
- **sft_then_rl**：前 3 轮走 SFT 的快速模仿，后 6 轮 RL 填洞，终值 0.9774。

两个结论都有算术背书，不是印象：第 1 轮 0.5986 > 0.4118（代码里有 assert），
说明**在 teacher 覆盖的范围内，模仿永远比探索快**；而 sft_only 终值 0.6190
与算术天花板 0.625 之差只有 0.006，说明**这个天花板是数据覆盖画死的，
与训练轮数无关**（实验 [4] 会直接验证这一点）。

一个诚实的提醒：在这个 toy 里 rl_only 追得很快（r3 就到 0.7129），因为环境确定、
reward 即时、每轮 36 个样本总能撞到正确答案。真实 RFT 里探索要贵得多
（生成成本高、reward 可能稀疏甚至延迟），rl_only 的曲线会难看很多——
这正是「SFT 打底」在生产配方里普遍存在的理由。toy 演示的是机制方向，不是幅度。

## 7. 实验 [3]：mix——两种信号同一步叠加

`mix=True` 时每轮同时产出 24 条 SFT + 36 条 RL 样本，FIFO 组出 24/24 的 batch，
**一次 `trainer.step` 里两种 loss 相加**。eval 8 轮到 0.9787。

注意 batch 里 SFT 占了一半，等于 RL 信号被稀释：mix 前 4 轮（0.7217）明显慢于
[1] 的 RL 阶段同期——混合比例本身就是一个需要调的超参。Trinity 的算法支持表里
有专门的 mix 类条目（`algorithm_type: mix_chord`，README L123，见 §9），
说明「SFT 信号与 RL 信号在同一步共存」不是 toy 的发明，而是生产框架的一等公民。

## 8. 实验 [4]：天花板是数据画死的（本节的算术高潮）

把 sft_only 加到 15 轮——比 [2] 多 67% 的训练量——结果：

```text
sft_only@15r: eval=0.6217 per-context=[0.98 0.99 0.99 0.26 0.25 0.26]
```

而算术预测是：覆盖 3 ctx → 1.0，空洞 3 ctx → 0.25（SFT 梯度从不流向它们，
权重保持初始均匀分布），`mean = (3×1.0 + 3×0.25)/6 = 0.625`。
实测 0.6217 与预测差 0.0033，在评测采样噪声范围内
（每 ctx 1024 次采样，单 ctx 标准差 ≈ √(0.25×0.75/1024) ≈ 0.014）。

这个 0.625 值得记住：**它由数据覆盖决定，不由训练计算量决定**。
看到 SFT loss 还在降、eval 却贴住某个值不动时，先去查 teacher 数据覆盖了什么，
而不是加轮数。要突破天花板只有两条路：补数据（扩大 teacher 覆盖），
或者换信号（RL 探索）——这正是「SFT → RL」配方存在的根本原因。

## 9. buffer 生命周期账本：需求 > 供给 = 积压

`[账本]` 一行：进入 288 条 | 训练 408 次 | 退役 192 条 | 现存 96 条。
数字之间对得上账，而且藏着一个工程教训。

先对账（全部可由上式推出）：288 = 3 轮×24 SFT + 6 轮×36 RL；
退役 192 条都训满了 `max_reuse=2`，现存 96 条中 24 条训过 1 次，
于是训练次数 = 192×2 + 24×1 = 408 ✓；剩 288 − 192 − 24 = **72 条从未被训练** ✓。

72 条积压是哪来的？**需求/供给算术**：RL 阶段每轮新产 36 条，每条要训 2 次
（max_reuse=2），需求 = 36×2 = 72 个 batch 槽位/轮；供给 = batch_size = 48 槽位/轮。
需求 > 供给，差额 24 槽位/轮 ÷ 每轮新产 36 条 = 每轮积压 12 条，6 轮恰积压 72 条。

教训很直接：**提高样本复用率（max_reuse）不是免费的**——要么同步加大 batch_size，
要么降低 explorer 产量，否则 buffer 里会堆起永远轮不到的数据（真实系统里它们
最终会被容量驱逐，等于白采）。这与 [nano-slime L0](../nano-slime/tutorial_L0.md)
的容量结论互为镜像：那里 buffer 容量买不到吞吐，这里复用率要配预算。

## 10. 费曼自检

**类比：一家考前辅导班。** teacher 的范例答案是 SFT 数据（只覆盖部分题型）；
学生自己刷题 + 对答案拿分数是 RL rollout + reward；错题本和题库管理
（题收进来、每道题做了几遍、做够两遍就归档）是 Buffer；
**课程表就是 config**——先上几节「抄范例课」、后上几节「自习刷题课」、
还是每节课两样都练，全由课表字段决定，教室、题库、学生都是同一套。

**类比的边界**：toy 里「题型」只有 6 种、答案是离散选项、对错即时可知；
真实 RFT 的回答是整个序列空间的生成，reward 可以稀疏、延迟、甚至由另一个模型打出
（后者是 L2 的主题）。

**类比的反例版**：如果课表只排「抄范例课」（sft_only），学生在老师没讲过的题型上
永远停在瞎蒙水平（~0.25）——多排十节抄讲课也不会变。这就是实验 [4]。

自检三问（讲不出就回到对应小节）：

1. 换配方到底改了系统的什么？（答：只改 config 的三个字段；样本协议、
   buffer、trainer、循环全部复用——§4）
2. 为什么 r4 的 batch 里有 SFT 样本？（答：buffer 是连续的，阶段切换不清空数据；
   FIFO 让存量旧样本继续被消费——§5）
3. 0.625 是怎么算出来的？（答：`(3×1.0 + 3×0.25)/6`，天花板由覆盖决定——§8）

## 11. 思考题

1. **baseline 的范围**：把 RL 的 baseline 从「context 组内均值」改成
   「整个 batch 的均值」，在 mix 的 batch（SFT 样本不参与 RL loss）里，
   样本少的 ctx 和样本多的 ctx 谁的学习速率会被扭曲？进一步：GRPO
   （arXiv:2402.03300）在组内均值之外还除以组内标准差——想想它解决的是什么问题
   （提示：不同组的 reward 方差不同时，梯度的相对尺度）。
2. **复用与 staleness**：把 `max_reuse` 从 2 改成 100，账本里的训练次数会怎么变？
   注意 RL 样本的 `version` 字段——被反复训练的样本产生于旧版权重。
   对照 [nano-slime L0](../nano-slime/tutorial_L0.md) 的 staleness 事件表：
   复用 = 拿陈旧样本训练，真实系统需要 importance sampling 之类的机制校正
   （见 [nano-verl L1](../nano-verl/tutorial_L1.md) 的 ratio 与 clip）。
3. **teacher 不只缺，还可能错**：本节空洞是「teacher 没覆盖」。如果换一个场景——
   teacher 给的 target 与环境 reward **冲突**（teacher 说选 A，reward 只奖 B），
   `sft_then_rl` 会发生什么？提示：谁掌握 labels 谁就是监督者，对照
   [nano-llamafactory L0](../nano-llamafactory/tutorial_L0.md) 里
   「mask 边界画在 labels 空间」的机制。

## 12. 反例节

1. **SFT 填不上数据空洞（实测）**：[4] 已给出——15 轮 sft_only 停在 0.6217，
   与算术天花板 0.625 只差噪声。加大 SFT 计算量不产生新信息。
2. **复用率要配预算（实测）**：§9 的 72 条积压。盲目调高 `max_reuse`
   而不动 `batch_size`，buffer 会堆满轮不到的样本。
3. **toy 的 RL 探索太便宜（口径声明，非实测数字）**：本节 rl_only 追得很快，
   因为环境确定、reward 即时、样本撞对答案的概率高。真实环境里探索可能昂贵、
   有安全约束、reward 稀疏——不要把 toy 里「RL 冷启动三轮就追上」的幅度外推。
   方向是机制给的，幅度必须实测。

## 13. 溯源与映射

**权威实现映射**（源码级路径对照留 L3，此处只到概念层）：

| toy | Trinity-RFT | 对应层级 |
|-----|-------------|---------|
| `Explorer.rl_rollout` | Explorer：agent-environment interaction（README L23） | 概念 |
| `Explorer.sft_data` | `algorithm_type: sft` 数据路径（README L121） | 概念 |
| `Buffer`（容量/复用/退役） | Full-Lifecycle Data Pipelines（README L102–105） | 概念 |
| `Trainer.step` 双 loss | 可插拔算法与 mix 类（README L109 / L123） | 概念 |
| `cfg` dict | trinity 配置体系（`algorithm_type` 等，README L117–123） | 概念 |
| explorer/trainer/buffer 真实源码路径 | `[TODO: verify source]`，L3 对照 | — |

**引用清单（全部 2026-08-05 现场核验）**：

- 论文：arXiv:2505.17826，标题页核验为 *Trinity-RFT: A General-Purpose and Unified
  Framework for Reinforcement Fine-Tuning of Large Language Models*；
  abstract 自述 modular and decoupled design（RFT-core 统一 sync/async、on/off-policy、
  online/offline 模式 + agent-environment interaction + systematic data pipelines）。
- README：`agentscope-ai/Trinity-RFT` main 分支（raw.githubusercontent.com 抓取）：
  L18–20 定位句（L18 为 *What is Trinity-RFT?* 标题、L19 空行、定位句实为 L20）、
  L21 「decouples RFT into three components」、L23–25 三组件定义、
  L102–105 Full-Lifecycle Data Pipelines（含 active data management 与 multi-task
  joint learning）、L109 plug-and-play decoupled architecture、
  L117 PPO / L118 GRPO / L121 SFT（`trinity/algorithm/policy_loss_fn/sft_loss.py`）/
  L122 DPO / L123 CHORD（`algorithm_type: mix_chord`）。
  README 快照可能随上游迭代漂移，行号以 2026-08-05 为准。
- GRPO：arXiv:2402.03300（Trinity README L118 自身引用；本仓库于 2026-08-04
  曾独立核验该 ID 为 DeepSeekMath 论文）。本 toy 只用组内均值、未除标准差，是简化版。
- 仓库迁移：`modelscope/Trinity-RFT` → 301 → `agentscope-ai/Trinity-RFT`（curl 实测）。
  本机对 github.com 的 HEAD 请求超时（与历史轮次一致），源文件经 raw.githubusercontent.com
  抓取成功即可达性实证。
- CHORD 论文 ID（arXiv:2508.11408）为 Trinity README L123 自身引用，本节未展开其机制，
  仅引用「mix 类条目存在」这一事实。`[TODO: verify]` 其内容细节（如需在 L1+ 使用）。

## 14. 下一步

L1：接真实小模型（如字符级 LSTM 或 0.5B 级开源模型），用同一套
「config 驱动两阶段」跑通 SFT→RL：SFT 阶段产出 checkpoint，RL 阶段从 checkpoint
续训，验证 loss/reward 曲线与 checkpoint 衔接（scaffold 原 L0 目标里的
「几步优化」一并上移到这里做）。届时本节 toy 的每个组件都有真实对应物：
`Trainer.step` 变成真实的优化器步，`version` 变成真实的 checkpoint 版本号。
