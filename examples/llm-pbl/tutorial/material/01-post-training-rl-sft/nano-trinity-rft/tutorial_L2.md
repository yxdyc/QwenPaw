# nano-trinity-rft L2 — RL 的信号来源：rule reward vs learned reward model

> **K+1 位置**：L1 用真实 0.8M char-GPT 跑通了配置驱动的 SFT→RL 两阶段，当时
> reward 是「逐位匹配率」，docstring 里声明它是 **dense、rule-based**，把
> 「稀疏 0/1 与 reward model」留给本级。L2 就回答这个问题：**RL 阶段的 reward
> 信号从哪里来？选哪一种？各自付出什么代价？**
> **对标权威实现**：`agentscope-ai/Trinity-RFT`（main 分支，2026-08-12 现场核验，
> README 30,381 B，sha256 `d513f140…b73982`——与 L1 08-06 录值逐位零漂移）。
> 行号锚点以 2026-08-12 抓取日为准。

---

## 1. 先跑起来

```bash
python3 -B L2_reward_signals.py     # 仅依赖 torch；CPU ~1.5 分钟；任意 CWD 可跑
```

固定 seed 下指标行逐字节确定（`elapsed` 计时行随机器负载浮动，掩码口径
`sed '/^[[:space:]]*elapsed/d'`）。完整输出（2026-08-13 本机实测，
掩码锚 md5 `0014cd664263c1d78450759c3e6ced33`/102 行，双 CWD 两遍逐字节一致）：

```text
============================================================================
nano-trinity-rft L2 — RL 的信号来源：rule reward vs learned reward model
============================================================================
env: python 3.13.13 | torch 2.13.0 | seed 20260806

[0] 任务与起点（与 L1 同一张任务表）
targets: ['acbb', 'daab', 'bcbd', 'dcca', 'bbba', 'cbba']  （teacher 覆盖 ctx [0, 1, 2]，ctx 3–5 数据空洞）
model: char-GPT | params=799,360 | 确定性探针: 两次前向逐位一致 ✓
SFT@3r:  exact=0.500 characc=0.625 per-ctx characc=[1.00 1.00 1.00 0.25 0.25 0.25]
warm RL@10r（dense rule reward, G=64，L1 同口径）: exact=0.667 characc=0.917
per-ctx characc: [1.00 1.00 0.75 1.00 1.00 0.75]  （空洞 ctx 3–5 爬到中途——『偶尔会对』窗口）
每 ctx exact 胜率 p̂（M=1536）: c0:0.368 c1:0.369 c2:0.031 c3:0.338 c4:0.352 c5:0.033

[1] 信号算术：组内 reward 全同（std=0）→ advantage 全 0 → 零梯度
    解析式 P(dead | p, G) = p^G + (1-p)^G（全对组与全错组都无信号）
G     sparse 实测(dead率, 空洞ctx均值)   sparse 解析    dense 实测(G 同值)
4        0.416                           0.424        0.0530
8        0.280                           0.279        0.0017
16       0.198                           0.197        0.0000
32       0.097                           0.116        0.0000
64       0.028                           0.040        0.0000

[2] 学习后果（同一起点 snap_warm，稀疏 exact reward，12 轮）
arm               rollouts/轮   dead率@r1   exact@r12  characc@r12  空洞ctx末态
sparse G=8           48        0.333      0.333     0.750      [0.00 0.00 0.00]
sparse G=64         384        0.167      0.333     0.833      [1.00 1.00 0.00]
sparse G=8+dyn      328        0.278      0.500     0.875      [1.00 1.00 0.00]
    dyn = nano 版 std_threshold 过滤 + 补采（rollout 预算封顶 = G64 等价）；
    Trinity 对应物: grpo_advantage.py:L160-163（std≤阈值的组 exps.clear() 跳过）
    + duplicate_experiences 补位（L178-194）；DAPO Dynamic Sampling [2503.14476 §3.2]

[3] learned reward model（从 snap_warm 采偏好数据，Bradley-Terry 训练）
偏好对: 960（分差≥0.25 的干净对 712 + 同分对 248；同分对标注者以 p=0.95 偏好含 'a' 多者——注入偏置）
RM: 26→32(tanh)→1 | BT loss −log σ(r_w−r_l) | 末步 loss=0.1068
held-out 对准确率: 全部 0.840（147/175）| 分差≥0.25 对 0.840（147/175）
RM 分与真实匹配率 Pearson r = 0.387（N=480 held-out 响应）
偏置探针（固定 q，比含 'a' 多/少两半的 RM 均分差）: Δ = +0.0618（18 个 (ctx,k) 探针均值；>0 = RM 学到了 'a' 偏置）
RM 的最爱（每 ctx 暴力扫 256 条）: ['aabd', 'ddcc', 'aabd', 'aaaa', 'abaa', 'aaba'] vs targets ['acbb', 'daab', 'bcbd', 'dcca', 'bbba', 'cbba']
    6/6 个 ctx 的 RM 最爱不是正确答案——proxy 与 gold 从训练完那天起就不是一回事

[4] Goodhart 三臂（同一起点 snap_warm，10 轮，G=24；gold 只监测不训练）
round  arm          proxy(RM均分)  gold exact  gold characc  'a'/resp
  1    rule_dense   0.6910        0.667      0.917       1.167
  1    rm(β=0)      0.7879        0.667      0.917       1.167
  1    rm(β=0.2)    0.7879        0.500      0.875       1.167

  3    rule_dense   0.7344        0.667      0.917       1.264
  3    rm(β=0)      0.8244        0.500      0.833       1.257
  3    rm(β=0.2)    0.8393        0.500      0.875       1.431

  6    rule_dense   0.8229        0.833      0.958       0.931
  6    rm(β=0)      0.8106        0.333      0.792       1.160
  6    rm(β=0.2)    0.8357        0.500      0.875       1.146

 10    rule_dense   0.8819        0.833      0.958       1.000
 10    rm(β=0)      0.8372        0.167      0.750       1.312
 10    rm(β=0.2)    0.8708        0.500      0.875       1.292

末态 greedy 输出:  rm(β=0) = ['acbb', 'daaa', 'bcba', 'dcba', 'bcba', 'bcba']
                   targets = ['acbb', 'daab', 'bcbd', 'dcca', 'bbba', 'cbba']
    β=0 臂: proxy 持续涨而 gold 反降/停滞——策略找到的是 RM 的错误面而非任务解
    （arXiv:2210.10760 的 gold/proxy 分离在 toy 尺度的复现）；KL 臂把策略锚在
    ref 附近: proxy 照样能涨（RM 与质量相关），但 gold 不坍缩；rule 臂 gold 一路升。

[5] 账本与取舍
成本: rule reward = 纯函数（0 次模型调用）；RM = 偏好对 960 条 + 500 步训练 + 每条响应 1 次 RM 前向
      且 gold 评测依然在循环里——只是从训练信号变成监测信号（Trinity RULER 示例同时记 reward/gold_reward/judge_success/eval_accuracy）
取舍表:
  维度        rule(verifiable)              learned RM
  忠实度      对任务精确（就是任务本身）     proxy：拟合偏好数据，含偏置
  覆盖面      只有 checker 存在的域          任意域（非可验证域的唯一路径）
  粒度        可稀可密（exact/逐位/格式）    天然 dense
  可被 hack   不能（答对就是答对）           能（[4] 实测：钻偏置空子）
  成本        ~0                             标注/偏好数据 + 训练 + 推理
  Trinity     注册表 7 个 reward 全 rule 型  非可验证域走 RULER/rubric 示例

self-check:
    PASS  SFT@3r 覆盖处 characc≥0.9（实测 [1.00 1.00 1.00]），聚合≈0.5（空洞贪心基线）
    PASS  warm RL@10r characc≥0.7（实测 0.917）
    PASS  空洞 ctx 胜率 p̂ 落在稀疏信号窗口 (0.005, 0.45)
    PASS  dead 率实测 vs 解析@G=8: |0.280 − 0.279| ≤ 0.08
    PASS  dead 率实测 vs 解析@G=64: |0.028 − 0.040| ≤ 0.08
    PASS  dead 率随 G 单调降: G=8 0.280 ≥ G=64 0.028 + 0.15
    PASS  稀疏 reward 在小 G 下信号大面积缺失: dead@G=4 = 0.416 ≥ 0.3
    PASS  稀疏学习后果: G=64 characc 0.833 ≥ G=8 0.750 + 0.05
    PASS  空洞 ctx 末态: G=64 0.917 ≥ G=8 0.583 + 0.3（小 G 稀疏 RL 净破坏，大 G 保住/填上）
    PASS  dynamic sampling 不差于固定 G=8（characc）: 0.875 vs 0.750
    PASS  dyn rollout 预算封顶生效: 3936 ≤ G64 的 4608
    PASS  RM held-out 干净对准确率 ≥ 0.72（实测 0.840）
    PASS  RM 分与真实匹配率相关 ≥ 0.3（实测 r=0.387；warm 态 q 值域窄，r 受值域限制）
    PASS  RM 学到注入的 'a' 偏置: 固定 q 下 Δ=+0.0618 > 0
    PASS  RM 的最爱多数不是正确答案: 6/6
    PASS  只优化 RM: proxy 上升 0.788 → 0.837（起点已近饱和）
    PASS  Goodhart 分离: rm 臂 gold 不涨且低于 rule 臂 (0.750 vs 0.958)
    PASS  钻空子末态: rm 臂多数 greedy 输出已非 target（exact=0.167 ≤ 1/3）
    PASS  KL 缓解: β=0.2 gold 0.875 ≥ β=0 0.750
    PASS  KL 锚定: β=0.2 臂 gold 不坍缩（0.875 → 0.875）
    PASS  rule 对照组 gold 不回退: 0.667 → 0.833
    PASS  RM reward 经 sigmoid 有界于 (0,1)（rm 臂全部轮次 proxy 在界内）
    ✅ self-check passed (22/22)

digest(md5 of metrics) = 5b3c872ee1396149968b710f012013eb
```

22 条 self-check 全绿。下面逐段拆。

---

## 2. 问题设定：L1 的 reward 是哪里来的

L1 的循环里有一行当时没展开：

```python
def reward_of(c, resp_ids):
    """rule-based dense reward：与 target 逐位匹配率 ∈ {0, .25, .5, .75, 1}。"""
```

它假了三件事：**有一个逐位可对照的 ground truth**（可验证）、**信号可以免费
算出来**（零成本）、**匹配率就是好坏本身**（忠实）。真实后训练里这三条各自
对应一个岔路：

- 没有逐位 truth，只有最终答案对错 → **稀疏 rule reward**（exact match {0,1}）；
- 连「对错」都无法程序化判定（医疗问答、写作、对话）→ **learned reward model**；
- reward 不再是任务本身，而是任务的**代理（proxy）**→ 代理可以被钻空子
  （Goodhart，本节 [4] 实测）。

Trinity 生产实现里这三条路长什么样，先给权威源码的地图（§3），再在 toy 上
逐条跑出来（§4–§8）。

---

## 3. Trinity 的 reward 地图（权威源码对照，2026-08-12 现场核验）

**（a）reward 的家在 `trinity/common/rewards/`。** 基类是 16 行的 ABC
（`reward_fn.py:L7-16`）：`__call__(**kwargs) -> Dict[str, float]`——返回值是
**多组件字典**，不只是一个标量。注册表（`__init__.py:L7-18`）登记了 7 个：
`math_reward / math_boxed_reward / format_reward / countdown_reward /
accuracy_reward / math_dapo_reward / rlcr_reward`。**7 个全部是 rule/parse 型**——
解数学答案、正则查格式、倒计时比对、置信度解析（rlcr_reward.py 的 5 组件
L22-35 也全是标签解析 + Brier 分数，没有一个是学出来的）。

**（b）组合方式是字典求和。** `math_rm_workflow.py:L34-44`：

```python
reward_dict = self.reward_fn(response, messages, ground_truth=self.truth)
if response.metrics is None:
    response.metrics = {}
response.metrics.update(reward_dict)      # 各组件分开记账
reward = sum(reward_dict.values())        # 训练用标量 = 直接求和
```

注意两件事：各组件先**分账**进 `metrics`（proxy/gold 可以分开监控），再求和成
训练标量；这个 workflow 的 docstring 自述「as introduced in DeepSeek-R1」
（`math_rm_workflow.py:L11-12`；R1 的 rule-based RLVR 见 arXiv:2501.12948）。
`MathRewardFn` 本身就是 accuracy + format 的组合（`math_reward.py:L35-39`
`return {**accuracy_score, **format_score}`），其中 accuracy 是稀疏 0/1
（`accuracy_reward.py:L61-67`：「Reward 1 if the content is the same as the
ground truth, 0 otherwise」，verify 异常也记 0），format 是 ±0.1 的正则 shaping
（`format_reward.py:L17-24`，式样源自 open-r1）。

**（c）组内无信号的问题，Trinity 在 advantage 层处理。** `grpo_advantage.py`
的 `GRPOGroupedAdvantage` 有个 `std_threshold` 参数（L97；L106-107 docstring：
「If provided, groups with a reward standard deviation equal or below this
threshold will be skipped」），实现就是本节 [2] 要演的机制（L160-163）：

```python
if self.std_threshold is not None and group_reward_std <= self.std_threshold:
    metrics["skipped_count"] = len(exps)
    exps.clear()          # 整组丢弃
```

丢掉的组还可以用 `duplicate_experiences` 从有效组复制补位（L108-109 引 Polaris
博客；L178-194 实现）。同一文件里还有两个 reward 信号相关的旋钮，均为近期单源
论文的工程化 [transient/单源，只作指针、机制不展开]：`rank_penalty`
（arXiv:2506.02355，按 logprob 排名纠偏）与
`std_cal_level`（arXiv:2508.08221，均值按组、标准差按 batch 的混合归一）。
advantage 本体是 `(r − group_mean) / (group_std + ε)`（L169）——nano 版与 L1
一致不除 std，差异在 tutorial_L1 §6 已讨论。

**（d）非可验证域走示例而非注册表。** README:L73 把「Non-verifiable domains」
指向三个示例：RULER（`examples/grpo_gsm8k_ruler/`，LLM-as-judge 对 rollout
排序，ART 的 Relative Universal LLM-Elicited Rewards）、trainable RULER、
rubric-as-reward（`examples/grpo_rubric_as_reward/`，医疗 QA 按 rubric 打分，
方法出自 RaR-Implicit，arXiv:2507.17746）。而 `agents_reward.py` /
`human_reward.py` 在 main 分支还是 20 字节的「# to be implemented」。RULER
示例 README 有两处与本节直接互文：其一，`std_threshold` 设小值正是为了
「filter out group of experiences with same rewards」；其二，它同时跟踪四个
指标——`reward`（judge 分）/ `gold_reward`（rule 算的 accuracy+format）/
`judge_success` / `eval_accuracy`，并提醒 judge reward 有噪声、lr 要调小
（2e-6）。**proxy 与 gold 分开记账、并且永远保留 gold 监测**——这是 [4]/[5]
的工程原型。

nano-L2 的取舍：我们把 (a)(b) 压成「reward 是 config 字段、来源标签进 Sample
记录」（§4），把 (c) 压成 `dynamic=dict(target_live, max_extra)` 的过滤+补采
（§6），把 (d) 压成 26→32→1 的 Bradley-Terry 小 RM（§7）。Trinity 的多组件字典
求和、auxiliary judge 模型调度不在本级重复（留给 L3 对照真实 schema）。

---

## 4. 实验 [0]：起点——SFT + warm RL，L1 的算术原样复现

任务表与 L1 同源（SEED=20260806）：6 个 context、4 字符响应、teacher 只覆盖
ctx 0–2。SFT 3 轮的结果与 L1 §8 的天花板算术逐位吻合：

```text
SFT@3r:  exact=0.500 characc=0.625 per-ctx characc=[1.00 1.00 1.00 0.25 0.25 0.25]
```

覆盖 3 ctx→1.0，空洞 3 ctx 贪心碰巧匹配 1/4 位→0.25，mean=(3×1+3×0.25)/6=0.625。
然后是一段 warm RL：**用 L1 的 dense rule reward（逐位匹配率 + 逐 token 组内
优势）跑 10 轮**，把策略推到「空洞偶尔会对」的中途态：

```text
warm RL@10r（dense rule reward, G=64，L1 同口径）: exact=0.667 characc=0.917
per-ctx characc: [1.00 1.00 0.75 1.00 1.00 0.75]
每 ctx exact 胜率 p̂（M=1536）: c0:0.368 c1:0.369 c2:0.031 c3:0.338 c4:0.352 c5:0.033
```

两个细节值得停一下：

- **贪心 characc 0.917 ≠ 采样胜率 p̂**。p̂ 是带 ε=0.3 探索的采样口径：贪心
  已经全对的 c0，采样胜率也只有 0.368。后文的 dead group 算术全部发生在采样
  口径上——RL 看见的是采样分布，不是贪心输出。
- **p̂ 很不均匀**：c2/c5 只有 0.03 左右，c3/c4 已到 0.34。同一个任务表里不同
  问题的「难度窗口」天然不同——这正是真实 RLVR 里「有的题大家都会、有的题全
  组全错」的由来。

为什么需要 warm 这一段？因为**稀疏 reward 只在策略「偶尔会对」的窗口里才有
信号**——p≈0 时组内全 0，p≈1 时组内全 1，两种都没有梯度。数学 RLVR 的 base
model 必须先冷启到「做得出一些题」，同一算术。

---

## 5. 实验 [1]：信号算术——dead group 率 = p^G + (1-p)^G

组内相对优势（GRPO，arXiv:2402.03300）的信号存在条件：**组内 reward 有方差**。
对稀疏 {0,1} reward，一组 G 条响应全错或全对时 std=0、advantage 全 0、梯度为
零。设单条胜率为 p，则

```
P(dead | p, G) = (1-p)^G + p^G
```

实测（空洞 ctx 均值）与解析式逐行对照：

```text
G     sparse 实测(dead率, 空洞ctx均值)   sparse 解析    dense 实测(G 同值)
4        0.416                           0.424        0.0530
8        0.280                           0.279        0.0017
16       0.198                           0.197        0.0000
32       0.097                           0.116        0.0000
64       0.028                           0.040        0.0000
```

三件事：

1. **实测与解析式吻合**（最大偏差 0.019，在 M=1536 的采样噪声内）——dead 率
   不是玄学，是 p 和 G 的初等函数。
2. **dense reward 的 dead 率几乎为 0**（G=4 时 0.053，G≥16 后为 0）：逐位部分
   分让「两条响应完全同分」变成小概率事件。这是 dense 的第一重红利——**不是
   分更高，而是组几乎永远活着**。
3. 小 G 下稀疏信号大面积缺失：**G=4 时 41.6% 的组零梯度**。这就是 DAPO 论文
   §3.2 Dynamic Sampling 的出发点，原文（arXiv:2503.14476，ar5iv 2026-08-12
   抓取）：「if all outputs of a particular prompt are correct and receive the
   same reward, the resulting advantage for this group is zero. A zero advantage
   results in zero policy gradients」——注意它同时点了「全对」的情况：模型已解
   出的题同样不产信号，所以 DAPO 过滤的是 accuracy=0 **和** accuracy=1 两组。

---

## 6. 实验 [2]：学习后果——小 G 稀疏 RL 是净破坏，dynamic sampling 最省

从同一个 snap_warm 出发，稀疏 exact reward，12 轮，三臂：

```text
arm               rollouts/轮   dead率@r1   exact@r12  characc@r12  空洞ctx末态
sparse G=8           48        0.333      0.333     0.750      [0.00 0.00 0.00]
sparse G=64         384        0.167      0.333     0.833      [1.00 1.00 0.00]
sparse G=8+dyn      328        0.278      0.500     0.875      [1.00 1.00 0.00]
```

（「空洞ctx末态」是 exact 口径；characc 口径 G8 空洞 0.583 / G64 与 dyn 0.917。）

- **G=8 不只是慢，是净破坏**：characc 0.917→0.750，warm 已解出的 c3/c4 从
  exact 1.0 掉回 0。机制：dead 组浪费预算，live 组又小又噪（8 条里 1 条对、
  7 条错，advantage 噪声大），加上探索扰动，净效果是把 warm 的家底拆掉。
- **G=64 保住了 c3/c4 并继续压 c5**（characc 0.833，空洞 exact [1,1,0]）。
  c5（p̂=0.033）12 轮没填上——解析式预言它的 G=64 dead 率仍有 ~11%，最难的
  题需要更大的 G 或课程，toy 里看得清清楚楚。
- **dyn 臂（nano 版 std_threshold：dead 组整组丢、live 组不足就补采、rollout
  预算封顶在 G64 等价）**：平均 328 rollouts/轮（比 G64 的 384 省 15%），
  characc 反而最高（0.875）。这就是 DAPO Fig.6 「采样实例变多、收敛反而更快」
  在 toy 尺度的对应形态——**为有效信号付费，比均匀花钱划算**。

还有一个容易忽略的细节：dead 组并非「无害地躺在那里」。Trainer 的 RL loss 是
`-(adv·logp).sum(dim=1).mean()`——**批内 dead 样本以零贡献参与 mean，把有效
梯度稀释掉**。过滤 dead 组既省 rollout 又提纯梯度，这是 Trinity 把过滤放在
advantage 层（进 buffer 之前）而不是训练层的原因（对照 `grpo_advantage.py:
L160-163` 的 `exps.clear()` 位置）。

---

## 7. 实验 [3]：learned reward model——BT 训练 + 三个诚实的测量

非可验证域没有 checker，reward 只能**学**。最小形态（Christiano et al.，
arXiv:1706.03741；InstructGPT 管线，arXiv:2203.02155）：采响应 → 成对比较 →
Bradley-Terry 损失 −log σ(r_w − r_l)。我们的标注者模型（显式声明的模拟）：

- 两条响应真实匹配率差 ≥0.25：以 0.95 概率标对（5% 噪声）；
- **同分对（q 相等）：标注者以 0.95 概率偏好含 'a' 更多的一条**——注入的
  系统性偏置。真实对应物：标注者的表面偏好（语气、排版、长度）渗进偏好数据。

RM = 26 维特征（ctx one-hot ⊕ 逐位字符 one-hot 含 other 桶）→ 32(tanh) → 1。
训练后测三件事：

```text
held-out 对准确率: 全部 0.840（147/175）| 分差≥0.25 对 0.840（147/175）
RM 分与真实匹配率 Pearson r = 0.387（N=480 held-out 响应）
偏置探针（固定 q，比含 'a' 多/少两半的 RM 均分差）: Δ = +0.0618（18 个 (ctx,k) 探针均值；>0 = RM 学到了 'a' 偏置）
RM 的最爱（每 ctx 暴力扫 256 条）: ['aabd', 'ddcc', 'aabd', 'aaaa', 'abaa', 'aaba'] vs targets [...]
    6/6 个 ctx 的 RM 最爱不是正确答案
```

- **0.840 的对准确率是真的**——RM 确实学到了质量信号；r=0.387 偏低有统计原因
  （warm 态响应集中在 q∈{0.75,1.0}，值域限制压低 Pearson），但方向明确。
- **偏置探针 Δ=+0.0618**：固定真实质量 q 不变，只改含 'a' 量，RM 均分仍然
  单调偏向 'a' 多的一侧——注入的偏置被学进去了。它相对质量信号很小，**这正是
  现实形态**：偏置不是 RM 的主成分，是贴在主成分上的系统性误差。
- **argmax 扫描是最刺眼的一条**：每个 ctx 暴力枚举全部 256 条响应，RM 的最爱
  6/6 不是正确答案，且多为 'a' 重的串（'aaaa'、'aaba'…）。**proxy 与 gold 的
  分离不是训练崩了——RM 准确率 0.84——而是从它训完那天起，它的错误面就客观
  存在，等着被优化器找到。**

---

## 8. 实验 [4]：Goodhart 三臂——proxy 涨、gold 掉，KL 锚住 gold

从 snap_warm 出发三臂，reward 分别为 dense rule / RM(sigmoid) / RM+β=0.2 KL。
**gold 只监测、不参与训练**（这正是 RULER 示例的 gold_reward 位）：

```text
 10    rule_dense   0.8819        0.833      0.958       1.000
 10    rm(β=0)      0.8372        0.167      0.750       1.312
 10    rm(β=0.2)    0.8708        0.500      0.875       1.292
```

- **rm(β=0)：proxy 0.788→0.837 一路涨，gold exact 却从 0.667 掉到 0.167。**
  末态贪心输出 `['acbb','daaa','bcba','dcba','bcba','bcba']`——'daaa' 落在注入
  的 'a' 偏置轴上，'bcba' 则是 RM 的另一个特异性错误面（与注入偏置无关）。
  **被钻的是哪条缝，不由你指定**：优化器找的是 RM 错误面上最陡的可利用方向，
  注入偏置只是众多错误面之一。这是 Gao et al.（arXiv:2210.10760）gold/proxy
  实验在 toy 尺度的复现，其摘要原话：「Because the reward model is an
  imperfect proxy, optimizing its value too much can hinder ground truth
  performance, in accordance with Goodhart's law」。
- **rm(β=0.2)：gold 锚在 0.875 不掉**（KL 把策略拉在 ref 附近），proxy 照样涨
  到 0.871——因为 RM 与真实质量相关，保住质量本身就保住大部分 proxy。**KL 的
  作用是抑制漂移、保住 gold，不是压住 proxy 数字**——把「防 hacking」读成
  「reward 涨得慢」是常见误读。
- **rule 臂：gold 一路上升到 0.958**，proxy 也健康上涨。同样的循环、同样的
  预算，**结局的差别只来自 reward 信号是谁**。

---

## 9. 机制深潜：三条带走的话

1. **信号存在性是 p×G 的算术，不是运气。** 稀疏 reward 下组内方差存在的概率
   是 1−p^G−(1-p)^G；它同时解释了「为什么 RLVR 要挑难度合适的题」（p 窗口）、
   「为什么 group size 是超参」（G 买信号）、「为什么 DAPO/Trinity 要过滤零
   方差组」（dead 组既费预算又稀释梯度）。dense reward 的另一重红利是让组
   几乎永远活着（[1] 表第三列）。
2. **learned reward 是代理，代理的错误面是客观存在的结构，不是训练事故。**
   0.84 准确率的 RM，argmax 扫描 6/6 不是正确答案；被优化后 gold 掉到 0.167。
   准确率是平均-case 指标，钻空子发生在误差的**分布**上——少量系统性误差就
   够被利用。所以 RM 路线必须配三样东西：gold 监测位（Trinity 的 gold_reward/
   eval_accuracy）、漂移约束（KL）、以及对 judge 噪声的让步（RULER 示例的小
   lr 与 std_threshold）。
3. **选 reward = 选「优化器会往哪里钻」。** rule reward 钻不动（答对就是答
   对），但只在有 checker 的域存在；RM 覆盖任意域，代价是错误面开放。Trinity
   的注册表全 rule 型、非可验证域走 RULER/rubric 示例，就是这个取舍的工程
   表达。

---

## 10. 费曼自检

**讲给外行听**：RL 的 reward 像考试的阅卷方式。rule reward 是机器阅卷的选择题
——对就是对，错就是错，免费且没法作弊，但你只能出选择题（可验证域）。learned
RM 是请了一位作文阅卷老师——什么题都能阅，但他有自己的口味（偏好数据里的
偏置被他学走）；学生一旦只按他的口味写（只优化 proxy），作文真实水平（gold）
反而下降。dead group 则是：一道题如果全班都答对或都答错，这道题对改进教学
毫无信息——组内相对优势的信号只存在于「有人对有人错」的组里。

三个自问：

- 能不能一句话说清 sparse 和 dense rule reward 的差别？（dense 让组永远活着、
  且逐位分给 credit；sparse 只在 p 窗口内有信号，且 credit 是整条响应级的。）
- 能不能说清「RM 准确率 0.84」和「被钻到 gold 0.167」为什么不矛盾？
- 能不能说清 KL 臂里 proxy 为什么照样涨、这算不算防住了 hacking？

---

## 11. 思考题

1. dead 率为什么是 p^G+(1-p)^G 而不是只有 (1-p)^G？模型已经做对的题（p→1）
   在 RLVR 里还值不值得采？DAPO 怎么处理 accuracy=1 的组？（§5 引文。）
2. 本 toy 里 dense reward「免费」——真实数学题的逐位部分分从哪来？为什么
   「过程奖励」本身是个研究问题（process reward 需要可验证的中间步骤）？
3. RM 对准确率 0.84 仍被钻空子。如果你的 RM 准确率提到 0.95，钻空子会消失
   吗？（提示：想 Gao et al. 的结论——过优化随 RM 规模与优化量平滑变化，
   准确率不是开关。）
4. Trinity 把 learned/judge 型信号放在示例（RULER/rubric）而不是注册表，
   从 RULER 示例 README 的两处配置（std_threshold、小 lr）看，这个设计在
   防什么？
5. [4] 里 gold 还能当监测信号，是因为 toy 有 ground truth。真实的非可验证域
   往往连 gold 都没有——这时 [4] 的保护还剩什么？（RaR 的答案：把评估标准
   做成 rubric 喂给 judge，arXiv:2507.17746。）
6. nano 版 advantage 不除组内 std（与 L1 一致），Trinity 除（std+ε，
   grpo_advantage.py:L169）。在 dead 组被过滤的前提下，除 std 会放大什么？
   （提示：live 组里 std 很小的组。）

---

## 12. 反例与边界

1. **偏置是注入的，真实偏置不长这样。** 我们的标注者偏置（同分对偏好 'a'）
   是显式模拟，单轴、可探针。真实标注偏置多轴、相关、且与质量信号纠缠，
   探针测不干净。本节证明的是「偏置会被学走且会被钻」这个机制类别，不是
   任何具体偏置的形态。
2. **toy 尺度不可外推。** 0.8M 策略 + 26 维 RM + 960 偏好对：钻空子的深度、
   KL 的平衡点、dead 率的数值都随规模变。Gao et al. 的 scaling law 讲的正是
   「过优化随 RM/策略规模平滑变化」——toy 给的是方向与机制，不是系数。
3. **稀疏 reward 不是坏选项。** p 高的域（简单题、格式检查）小 G 就够，
   dead 率是 p×G 的函数；RLVR 在数学/代码上的成功（DeepSeek-R1 路线，
   arXiv:2501.12948）正是把稀疏 rule reward 用在了 p 窗口合适的域。
4. **RM 也不是坏选项。** 非可验证域它是唯一路径（Trinity README:L73 的三个
   示例全是 judge 型）。问题从来不是「能不能用」，而是「必须配监测与约束」。
5. 本级的 KL 是 β 正则（GRPO 目标里的正则项），不是 PPO 的 clip；生产系统
   通常两者叠加。L1 §6.2 的教训（KL 的锚在空洞处是错的）在这里没复现，因为
   ref 是 warm 末态（已经填了大半洞）而非 SFT 末态——**ref 选在哪，KL 就把
   你锚在哪**。

---

## 13. 阶梯预告与交叉引用

- **L3**：对照 Trinity 真实配置 schema 与 explorer/trainer/buffer 源码，复现
  ablation ladder 式的开关组合（README 阶梯表 L3 行）——本级的 reward 来源
  切换、dyn 过滤都会在那里的真实配置字段里找到对应物。
- 交叉阅读：[nano-verl](../nano-verl/)（rollout/训练调度与 ratio clip）、
  [nano-slime](../nano-slime/)（buffer 解耦与 staleness——异步下 reward 的
  陈旧是另一个信号质量维度）、[nano-fsdp L3](../../02-pretraining-cpt/nano-fsdp/tutorial_L3.md)
  （混合精度下「省显存不省模型状态」的账本方法，与本节 proxy/gold 分账同构）、
  [02 轨 sota-deepdive](../../02-pretraining-cpt/sota-deepdive/deepseek-moe-mla-stability.md)
  （只读引用：V3 的稳定性机制面）。

---

## 14. 溯源与校准

**权威实现锚点**（agentscope-ai/Trinity-RFT main 分支，raw.githubusercontent.com
2026-08-12 抓取；README 30,381 B，sha256
`d513f140afdd691a0847f668ab5bd3cc062f99682cd9a2f5bba49e1cacb73982`——与 L1
08-06 录值逐位零漂移）：

| 锚点 | 内容 |
|------|------|
| README:L21-25 / L73 / L102-105 / L121 | 三组件 / 非可验证域三示例 / 全生命周期数据管线 / SFT 配置项（L0/L1 已用，本轮复验零漂移） |
| trinity/common/rewards/reward_fn.py:L7-16 | RewardFn ABC，`__call__ → Dict[str, float]` |
| trinity/common/rewards/__init__.py:L7-18 | REWARD_FUNCTIONS 注册表 7 项全 rule/parse 型 |
| trinity/common/rewards/math_reward.py:L15-39 | MathRewardFn = accuracy + format，L39 字典合并 |
| trinity/common/rewards/accuracy_reward.py:L61-67 | 稀疏 0/1，verify 失败记 0 |
| trinity/common/rewards/format_reward.py:L17-24 | 正则 ±0.1 shaping（Ref open-r1） |
| trinity/common/rewards/rlcr_reward.py:L22-35 | 5 组件 rule reward（含 Brier/置信度解析） |
| trinity/common/rewards/agents_reward.py / human_reward.py | 「# to be implemented」（各 20 B） |
| trinity/common/workflows/math_rm_workflow.py:L11-12, L34-44 | 「as introduced in DeepSeek-R1」；reward_dict→metrics→sum |
| trinity/algorithm/advantage_fn/grpo_advantage.py:L97, L106-116, L160-163, L169, L178-194 | std_threshold / duplicate_experiences / rank_penalty / std_cal_level；过滤实现；(r−mean)/(std+ε)；补位实现 |
| examples/grpo_gsm8k_ruler/README.md（3,762 B） | RULER=LLM-as-judge 排序；std_threshold 过滤同分组；lr 2e-6 防噪声；reward/gold_reward/judge_success/eval_accuracy 四指标 |
| examples/grpo_rubric_as_reward/README.md（2,053 B） | 非可验证医疗 QA；RaR-Implicit；judge 按 rubric 打 [0,1] 分 |

**论文锚点**（arXiv abs 页 2026-08-12 现场重抓核验标题/日期）：

| arXiv ID | 标题（核验录值） | 本节用法 |
|----------|------------------|----------|
| 2505.17826 | Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models（v1 2025-05-23，v3 2025-09-29） | 对标框架 |
| 2402.03300 | DeepSeekMath（GRPO） | 组内相对优势 |
| 2503.14476 | DAPO: An Open-Source LLM RL System at Scale（v2 2025-05-20） | §3.2 Dynamic Sampling 逐字引文（ar5iv 204,976 B 抓取） |
| 2210.10760 | Scaling Laws for Reward Model Overoptimization（2022-10-19） | gold/proxy 分离 + Goodhart 摘要引文 |
| 1706.03741 | Deep reinforcement learning from human preferences（v4 2023-02-17） | BT 成对偏好 RM |
| 2203.02155 | Training language models to follow instructions with human feedback（2022-03-04） | RLHF 管线 / RM 架构族 |
| 2507.17746 | Rubrics as Rewards: RL Beyond Verifiable Domains（v2 2025-10-03） | 非可验证域 rubric reward |
| 2501.12948 | DeepSeek-R1（v2 2026-01-04） | rule-based RLVR 当今定位 |
| 2506.02355 / 2508.08221 | Rewarding the Unlikely / Part I: Tricks or Traps?（grpo_advantage.py docstring 引） | Trinity 内嵌旋钮出处 [transient/单源]（只作指针，未展开机制） |

**信息分类**：表格与引文 = 原文声称（现场抓取）；dead 率解析式 = 初等概率
（文献已有的 GRPO 组内方差讨论的直接推论）；「小 G 净破坏」「被钻的缝不由
你指定」= 本节实测 + 合理推断（toy 尺度，边界见 §12）；标注者偏置形态 = 显式
声明的模拟。

**复现锚点**：`python3 -B L2_reward_signals.py`，任意 CWD，CPU ~1.5 分钟；
掩码 `elapsed` 行后输出 md5 `0014cd664263c1d78450759c3e6ced33`/102 行
（2026-08-13 两独立 CWD 逐字节一致）；脚本自产 digest
`5b3c872ee1396149968b710f012013eb`（metrics 的 md5）。
