# nano-llamafactory L2 — DPO：偏好对住在同一套遮罩上

> **K+1 位置**：L0/L1 建立了 SFT 的数据侧三件套——chat template 定 response
> 边界、labels 的 `-100` 遮罩定 loss、collator 的 pad 双层遮罩——并验证了
> 「SFT 与预训练共用同一个 next-token loss，差别只在 labels 遮罩」。本级只加
> 一个新问题：**如果数据不再是「标准答案」，而是「哪个回答更好」的成对比较，
> 同一套数据侧机制要改什么？** 答案是：几乎什么都不改——一个偏好对就是两条
> 普通 SFT 行；真正新的只是 loss 怎么消费它们。这就是 DPO（Direct Preference
> Optimization，arXiv 2305.18290）。
>
> **对标权威实现**：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
> main 分支 @`0bbe481e`（全 HEAD `0bbe481e6e621527284d37f1e13a6b9556c303ec`，
> 2026-08-13 codeload tarball 抓取，并经独立复验确认零漂移）。本文行号锚点以
> **2026-08-13 抓取日**为准；引用录值均保留来源与快照口径。
>
> **时效性定位（课程的经典证据层经典锚点，必读 §8.2）**：DPO 是无可置疑的经典，
> 但**经典 ≠ 前沿**——前沿模型的生产配方已转向 GRPO/RLVR 族。

---

## 1. 先跑起来

**可运行性契约声明（课程可运行性契约）**：本级的训练全部是**真实 torch 梯度下降**，
不是 mock——DPO 的构造与优化每一步都真跑。toy 的只是规模（74,496 参数字符级
GPT、6 对单位数加法），目的是让机制在 CPU 秒级、固定 seed 下逐字节确定地显现。
真实框架的多卡 / 全模型路径本机不覆盖，标 `[TODO: verify on real system]`
（需在真实 GPU/多机环境验证）；本节机制与 LLaMA-Factory 源码逐行对照（§5）。

```bash
python3 -B L2_dpo_preference_pairs.py   # 仅依赖 torch；CPU 实测 ~4s；任意 CWD 可跑
```

固定 seed 下除 `elapsed` 计时行外逐字节确定（掩码口径
`sed '/^[[:space:]]*elapsed/d'`）。完整输出（2026-08-13 本机实测，
python 3.13.13 / torch 2.13.0；掩码锚 md5 `f8b50175a6ee5d1c307a84d827a6ea76`/51 行，
累计七次运行、四份独立输出均逐位吻合）：

```text
====================================================================
nano-llamafactory L2 — DPO: preference pairs on the same mask
====================================================================
vocab size = 26
model params = 74,496
pairs = 6 (chosen = correct sum, rejected = off-by-one)

[0] pairwise collate: a preference pair = two ordinary SFT rows
    batch rows = 12 = 2 x 6 (first 6 = chosen, last 6 = rejected; collator.py:L564 顺序)
    supervised tokens per row = [3]  (answer + \n + <|eot|>; prompt 全 -100)
    every row identical to its L1-style SFT row (pre-pad): True

[1] SFT baselines: clean data vs noisy data (rejected also fed as positive)
    clean SFT: loss 3.5309 -> 0.0003 (6 rows x 300 epochs)
    noisy SFT: loss 3.5268 -> 0.2334 (12 rows x 300 epochs)
    model        win(p_c>p_r)  greedy  mean_p_chosen  mean_p_rejected
    clean SFT    6/6          6/6     0.9992         0.0000
    noisy SFT    4/6          4/6     0.4785         0.5205

[2] DPO from noisy ref (beta=0.1 = LLaMA-Factory pref_beta default):
    loss = -log sigmoid(beta * (margin_policy - margin_ref))
    step   0: loss=0.6931  margin=-0.0864  pair_acc=1/6
    step  40: loss=0.0065  margin=+54.4843  pair_acc=6/6
    step  80: loss=0.0008  margin=+78.4252  pair_acc=6/6
    step 120: loss=0.0005  margin=+83.0357  pair_acc=6/6
    step 160: loss=0.0004  margin=+86.5248  pair_acc=6/6
    step 199: loss=0.0003  margin=+89.4095  pair_acc=6/6
    after DPO:  win=6/6  greedy=6/6  margin=+89.4783
    p_rejected: 0.5205 (noisy ref) -> 4.23e-33 (DPO)
    drift KL(policy || ref) at answer position = 0.9230 nats
    implicit reward gap beta*(margin_policy-margin_ref) = +8.9565

[3] beta sweep: separation vs drift (each from a fresh copy of ref)
    beta=0.1 : margin=+89.4783  drift=0.9230 nats  win=6/6  greedy=6/6
    beta=0.5 : margin=+38.5336  drift=7.6404 nats  win=6/6  greedy=2/6
    beta=2.0 : margin=+8.5994  drift=5.4233 nats  win=6/6  greedy=3/6
    answer-position dist for 'Compute 2+2' (chosen '4', rejected '3'):
        noisy ref : '4'=0.5053  '3'=0.4943  '2'=0.0001
        beta=0.1  : '4'=0.9400  '7'=0.0599  '\n'=0.0000
        beta=0.5  : 'assistant.'=0.9998  '2+3'=0.0002  '1+2?'=0.0000
        beta=2.0  : '4'=0.5710  '\n'=0.3360  '5'=0.0364
    pair loss 只约束 p(chosen) vs p(rejected)，pair 之外的质量完全自由。

[4] counter-example: reversed pairs teach the model to be confidently wrong
    reversed-DPO: win=0/6  greedy=0/6  p_rejected=0.6649
    example: 'What is 1+1?' -> '3'  (chosen '2', rejected '3')
    example: 'Compute 2+2' -> '3'  (chosen '4', rejected '3')

====================================================================
[self-check] 15/15 PASS
digest: 9353e071cfb4054a6a3649a28c2cc6e7
```

15 条 self-check 全绿。下面逐段拆。

---

## 2. 代码结构

单文件 524 行，七个角色：

| 角色 | 代码 | 对照 LLaMA-Factory |
|------|------|--------------------|
| 数据 | `pairs`（6 对：chosen=正确和，rejected=off-by-one 错答） | 偏好数据集（chosen/rejected 双字段） |
| 行构造 | `make_row` = L1 原样（template + labels 遮罩） | `data/processor/pairwise.py:L66` |
| 批构造 | `pairwise_collate` → 2n 行，前 n=chosen | `data/collator.py:L564` PairwiseDataCollatorWithPadding |
| 模型 | `TinyLM` = L1 原样（2 层 causal Transformer） | 任意 causal LM |
| logps | `seq_logps`（eval 侧 no_grad）/ 训练循环内联（带梯度） | `train/trainer_utils.py:L592` get_batch_logps |
| loss | `dpo_loss` = −log σ(β·(margin_policy − margin_ref)) | `train/dpo/trainer.py:L187` 委托 trl DPOTrainer.dpo_loss |
| 训练 | `train_dpo`：deepcopy ref → 预算一次 ref logps → 200 步 | `concatenated_forward`（trainer.py:L219-253） |
| 测量 | `evaluate_pairs` / `drift_kl` / `answer_position_dist` | rewards/accuracies 指标（trainer.py:L305-308） |

两个设计决定值得先看：

**ref logps 只算一次**。ref 模型冻结不动，所以训练前 `no_grad` 前向一遍把
`ref_chosen/ref_rejected` 存下来，循环里只跑 policy 前向。LLaMA-Factory 的
`compute_reference_log_probs` 每步走一遍 ref 前向（或用独立 ref 进程），语义相同、
工程上更通用（ref 可以是不同架构 / 不同并行度的模型）；nano 版吃定「ref 冻结」
这条不变量换 CPU 上的速度。

**测量面比训练面大**。`evaluate_pairs` 同时看 win（p_chosen > p_rejected 的对数）
与 greedy（第一个生成 token 是否等于 chosen 答案）——[3] 会证明这两个指标可以
剧烈背离，只看 win 会被骗。

---

## 3. 输出解读

### 3.1 [0] 偏好对 = 两条普通 SFT 行（机器证明）

```text
[0] pairwise collate: a preference pair = two ordinary SFT rows
    batch rows = 12 = 2 x 6 (first 6 = chosen, last 6 = rejected; collator.py:L564 顺序)
    supervised tokens per row = [3]  (answer + \n + <|eot|>; prompt 全 -100)
    every row identical to its L1-style SFT row (pre-pad): True
```

三行输出是一个连续性证明：

- **2n 行、前 n = chosen**：与 LLaMA-Factory `PairwiseDataCollatorWithPadding`
  的 docstring 逐字同义（collator.py:L564，2026-08-13 抓取）：
  > "We generate 2 * n examples where the first n examples represent chosen
  > examples and the last n examples represent rejected examples."
- **每行只监督 3 个 token**（答案 + `\n` + `<|eot|>`），prompt 全 `-100`——
  与 L1 的遮罩规则逐位相同。偏好学习没有发明新的 mask 语义。
- **`every row identical ... : True`**：pairwise 批的每一行（去 pad 后）与按
  L1 单条 SFT 路径构造的行逐位相等。这是「偏好对 = 两条普通 SFT 行」的机器证明，
  不是口头声明。

LLaMA-Factory 侧的同构点在 `data/processor/pairwise.py:L66`：

```python
chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
```

——和 supervised processor 的 label 构造一模一样，只是每个 prompt 渲染两次
（chosen 一次、rejected 一次）。**数据侧三件套原封不动，DPO 的新东西全在 loss 侧。**

### 3.2 [1] 含噪 SFT 会 mode-cover：概率质量在对错之间劈开

```text
    clean SFT: loss 3.5309 -> 0.0003 (6 rows x 300 epochs)
    noisy SFT: loss 3.5268 -> 0.2334 (12 rows x 300 epochs)
    model        win(p_c>p_r)  greedy  mean_p_chosen  mean_p_rejected
    clean SFT    6/6          6/6     0.9992         0.0000
    noisy SFT    4/6          4/6     0.4785         0.5205
```

同一个 toy 模型，两份数据：

- **clean**：只喂 6 条 chosen 行。loss 降到 0.0003，win 6/6、greedy 6/6，
  p_chosen 0.9992——L1 的结论复现：遮罩正确的 SFT 把答案背下来。
- **noisy**：把 6 条 rejected 行（off-by-one 错答）也当正例混进去。每个问题
  都有一对「对答/错答」同时被监督，SFT 的最优解是**按数据比例分配概率质量**：
  p_chosen 0.4785 ≈ p_rejected 0.5205，几乎对半劈开。win 只剩 4/6，greedy 4/6。

这就是 **mode-covering**：SFT 的最小化目标是逐 token 交叉熵，它忠实地拟合数据
里的每一个模式——包括错误模式。你没法靠「多喂几遍对的」来抵消，因为错的也在
被等强度监督。注意 noisy 的 loss（0.2334）并没有爆炸：从训练曲线看它「收敛得
很好」。**低 loss 再次不等于行为正确（L1 §3.3 的 off-by-one 同课）。**

这设定了 DPO 的考题：**在数据含噪、且拿不到干净目标时，怎么修？**

### 3.3 [2] DPO 从含噪 ref 修复：只靠「哪个更好」

```text
[2] DPO from noisy ref (beta=0.1 = LLaMA-Factory pref_beta default):
    loss = -log sigmoid(beta * (margin_policy - margin_ref))
    step   0: loss=0.6931  margin=-0.0864  pair_acc=1/6
    ...
    step 199: loss=0.0003  margin=+89.4095  pair_acc=6/6
    after DPO:  win=6/6  greedy=6/6  margin=+89.4783
    p_rejected: 0.5205 (noisy ref) -> 4.23e-33 (DPO)
    drift KL(policy || ref) at answer position = 0.9230 nats
    implicit reward gap beta*(margin_policy-margin_ref) = +8.9565
```

从含噪模型（深拷贝）出发，200 步 DPO，beta=0.1（= LLaMA-Factory `pref_beta`
默认值，finetuning_args.py:L171-173）：

- **step 0 的 loss = 0.6931 = ln 2 是 DPO 的冷启动指纹**：policy 与 ref 是同一份
  权重，margin_policy − margin_ref = 0，−log σ(0) = ln 2。任何实现跑 DPO，第 0 步
  都应该是 ln 2——不是的话，说明两次前向数值不一致或 ref 接错了。
- **起点 margin = −0.0864**：含噪 ref 平均甚至**略微偏爱错答**。DPO 要逆着
  ref 的偏见爬。这正是 ref 作为 margin 基线的意义——loss 消费的是「相对起点的
  改善量」，不是绝对概率。
- **200 步后**：pair_acc 6/6，margin +89.5，p_rejected 从 0.5205 被压到
  4.23e-33（30 多个数量级），greedy 修复到 6/6。**全程没有干净目标**——
  信号只有「chosen 比 rejected 好」这一种比较。
- **drift = 0.9230 nats**：答案决策位上 KL(policy‖ref) 的读数。β 是 KL 缰绳
  （§4.4），这个数就是缰绳松紧的直接测量。
- **implicit reward gap = +8.9565**：β·(margin_policy − margin_ref)，即 DPO
  视角下「隐式奖励」的改善量——reward 没有显式模型，它隐含在 policy 与 ref 的
  对数概率差里（§4.4 的推导）。

### 3.4 [3] beta 扫描 + 答案位分布探针：win 6/6 与 greedy 崩坏同时成立

```text
    beta=0.1 : margin=+89.4783  drift=0.9230 nats  win=6/6  greedy=6/6
    beta=0.5 : margin=+38.5336  drift=7.6404 nats  win=6/6  greedy=2/6
    beta=2.0 : margin=+8.5994  drift=5.4233 nats  win=6/6  greedy=3/6
    answer-position dist for 'Compute 2+2' (chosen '4', rejected '3'):
        noisy ref : '4'=0.5053  '3'=0.4943  '2'=0.0001
        beta=0.1  : '4'=0.9400  '7'=0.0599  '\n'=0.0000
        beta=0.5  : 'assistant.'=0.9998  '2+3'=0.0002  '1+2?'=0.0000
        beta=2.0  : '4'=0.5710  '\n'=0.3360  '5'=0.0364
```

三个 beta 各从 ref 的新鲜拷贝独立训练，两个观察：

**观察一：beta 越大，margin 越小，sigmoid 越早饱和。** margin +89.5（β=0.1）→
+38.5（β=0.5）→ +8.6（β=2.0）。loss = −log σ(β·Δmargin)：β 越大，Δmargin 只需
很小就能把 sigmoid 推到饱和、梯度归零，训练就「提前收工」。β 小则要求更大的
分离度才饱和。这是 self-check「sigmoid saturation: margin(0.1) > margin(0.5) >
margin(2.0)」盯住的算术。

**观察二（本节最重要的反直觉）：β=0.5 win 6/6，greedy 却只剩 2/6。** 探针给出
答案位分布：β=0.5 的模型把 0.9998 的质量倒在了 `'assistant.'` 上——一个跟答案
毫无关系的 token。pair loss 只约束 p(chosen) 与 p(rejected) 的**相对**大小；
质量漏去 pair 之外的第三个 token，loss 完全失明。win 率是 pair 内指标，行为
（greedy）是全局指标——**loss 正常、行为崩坏**就是这么发生的。

drift 的排序也由此解释：β=0.1 走「干净路径」（压 rejected、chosen 留在原处，
drift 0.92 nats）；β=0.5/2.0 走「偷懒路径」（质量整体搬家到无关 token，
drift 7.64/5.42 nats）。drift 大小不是 β 的直接单调函数，而是「优化路径把质量
搬去了哪」的结果—— toy 尺度、固定 seed 下的路径现象，机制可迁移、系数不可
外推（§8.1）。

诊断启示：只盯 pair 内指标（win / rewards/accuracies）会被骗。LLaMA-Factory 的
指标面（trainer.py:L305-312：rewards、logps、logits 三族）比 win 率宽，但同样
不直接测「质量是否漏出 pair」——生产里靠 held-out 生成评测兜底。

### 3.5 [4] 反例：颠倒的偏好对把模型教成「自信地答错」

```text
    reversed-DPO: win=0/6  greedy=0/6  p_rejected=0.6649
    example: 'What is 1+1?' -> '3'  (chosen '2', rejected '3')
    example: 'Compute 2+2' -> '3'  (chosen '4', rejected '3')
```

把每对的 chosen/rejected 字段互换，其余不变：DPO 照样收敛得「很好」——只是把
偏好学了个反：win 0/6、greedy 0/6，p_rejected 0.6649，模型**自信地输出错答**。

比较信号本身没有方向感。loss 只关心「被标为 chosen 的那行概率要高于被标为
rejected 的那行」，标签对不对它无从判断。[2] 与 [4] 用完全相同的机制跑出完全
相反的结果——**偏好数据的质量就是 DPO 的天花板**。这是真实管线里 RM 过滤、
多标注者一致性、按可验证规则筛数据（RLVR 路线）存在的根本原因之一；对照
nano-trinity-rft L2 的 reward 信号来源讨论（`../nano-trinity-rft/tutorial_L2.md`）。

---

## 4. 机制深挖

### 4.1 偏好对构造：数据侧零发明

[0] 的机器证明已给出结论；补两个权威实现的细节：

- LLaMA-Factory 的 pairwise 预处理（pairwise.py:L58-69）先做长度预算
  （`infer_seqlen` 在 `cutoff_len` 内给 prompt/response 分额度），再各自拼
  `prompt_ids + chosen_ids` / `prompt_ids + rejected_ids`——**chosen 与 rejected
  共享同一个 prompt 渲染、同一套 IGNORE_INDEX 遮罩**（pairwise.py:L66）。
- collator 侧（collator.py:L564 起）把 n 个 feature 展开成 2n 行：先遍历
  `("chosen", "rejected")` 两个 key、再遍历 features——所以批内顺序是
  「全部 chosen 在前、全部 rejected 在后」，而不是逐对交错。这个顺序是后面
  `split(batch//2)` 能工作的前提。

### 4.2 concatenated_forward：一次前向，劈成两半

LLaMA-Factory `train/dpo/trainer.py:L219-253`（2026-08-13 抓取）：

```python
labels = batch.pop("labels")  # dpo do not need compute loss in forward
all_logits: torch.Tensor = model(**batch, ...).logits.to(torch.float32)
all_logps, valid_length = get_batch_logps(logits=all_logits, labels=labels, ...)
if self.loss_type in ["ipo", "orpo", "simpo"]:
    all_logps = all_logps / valid_length            # L234-235：长度归一族
batch_size = batch["input_ids"].size(0) // 2
chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)   # L237-238
```

2n 行**一次前向**再 `split(batch//2)`，而不是 chosen/rejected 各跑一遍：一次
kernel 启动、批内 padding 统一、chosen 与 rejected 在完全相同的数值条件下计算。
nano 的 `train_dpo` 逐字同构：`logps.split(n, dim=0)`。另注意 logits 先转
fp32 再算 logps——概率对数值精度敏感，混合精度训练下这是稳定性细节。

### 4.3 get_batch_logps：shifted gather、-100 遮罩、求和

`train/trainer_utils.py:L592`（2026-08-13 抓取），核心四步（L607-611）：

```python
labels = labels[:, 1:].clone()          # shift：labels 空间
logits = logits[:, :-1, :]
loss_mask = labels != label_pad_token_id     # -100 遮罩
labels[labels == label_pad_token_id] = 0     # dummy token，gather 安全
per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
```

默认分支（sigmoid 族）对 mask 内 token **求和**（L634：
`logps = (per_token_logps * loss_mask).sum(-1)`）；`ld_alpha` 分支
（L614-632）对「chosen/rejected 公共长度之外」的 token 打折加权，是长度偏置
校正的近期扩展。nano 的 `seq_logps` 就是默认分支的逐行复刻。

**「和」vs「平均」的语义差**：ipo/orpo/simpo 族在 `concatenated_forward` 里再除
`valid_length`（trainer.py:L234-235），margin 变成「平均每 token 的偏好强度」；
sigmoid 族用「和」，长回答的 margin 天然更大。chosen/rejected 长度差异大时，
两种聚合给出不同的训练动力学——这是 LLaMA-Factory 把聚合方式与 loss 族绑定的
原因。

### 4.4 DPO loss 与 reference model 的三个角色

**损失**（arXiv 2305.18290 Eq. 7；ar5iv 2026-08-13 抓取逐字核验录值，见 §10）：

```
loss = -log sigmoid( beta * ( margin_policy - margin_ref ) )
margin = logp(chosen) - logp(rejected)
```

LLaMA-Factory 经 `compute_preference_loss`（trainer.py:L187）分派：
`use_ref_model=False` 走 orpo/simpo（不需要 ref）；否则委托 trl 的
`DPOTrainer.dpo_loss`（trainer.py:L27 import），默认 sigmoid 族
（finetuning_args.py:L183-186）。`use_ref_model` 不是用户字段而是推导量
（finetuning_args.py:L593）：

```python
self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
```

**为什么这个 loss 长这样（本质）**：KL 约束下的奖励最大化
`max_π E[r(y)] − β·KL(π‖ref)` 有闭式解 `π*(y|x) ∝ ref(y|x)·exp(r(y)/β)`；
反解出 `r(y) = β·log(π*(y)/ref(y)) + β·log Z(x)`——**奖励隐含在 policy 与 ref
的对数概率差里**。把它代入 Bradley-Terry 偏好模型 `P(y_w ≻ y_l) = σ(r_w − r_l)`，
配分函数 Z(x) 在差值里相消，就得到上面的 logistic loss。这就是「DPO 不需要显式
reward model」的准确含义：RM 被 (π, ref) 这对模型隐式参数化了。

**reference model 的三个角色**（[2] 的输出逐一对应）：

1. **policy 的起点**：nano 与 LF 都从 ref 深拷贝出 policy。step 0 的 ln 2 指纹
   正是「两者同权」的体现。
2. **margin 的基线**：loss 消费的是 margin_policy − margin_ref。含噪 ref 的
   margin = −0.0864（略偏爱错答），DPO 必须逆着这个偏见爬——没有 ref，
   「相对起点的改善」就无从定义。
3. **KL 球的锚**：β 是 KL 约束的乘子，ref 是约束的中心。drift KL(policy‖ref)
   = 0.9230 nats 就是「policy 离开锚点多远」的直接读数；β 越小缰绳越松、
   允许的漂移越大（但 [3] 表明漂移大小最终取决于优化路径把质量搬去哪）。

**DPO 的优缺点同根**：它只看 pair 内两行——不需要 RM、不需要在线采样，工程上
极简；也正因为只看 pair 内两行，pair 之外的概率质量完全失明（[3] 的质量泄漏）、
标签方向错了也无从察觉（[4]）。off-policy 是它的省钱之处，也是它的天花板。

---

## 5. 权威实现取舍表（nano-L2 vs LLaMA-Factory）

| 维度 | nano-L2 | LLaMA-Factory（2026-08-13 抓取锚点） |
|------|---------|--------------------------------------|
| 偏好行构造 | `make_row` + `pairwise_collate` | pairwise.py:L66（`[IGNORE_INDEX]*source_len + ids`）+ collator.py:L564（2n 行 chosen-first） |
| 前向 | 2n 行一次前向 + `split(n)` | `concatenated_forward` 一次前向 + `split(batch//2)`（trainer.py:L219-253）；logits 转 fp32 |
| logps | shifted gather + mask + 求和 | `get_batch_logps`（trainer_utils.py:L592）；sigmoid 族求和（L634），ipo/orpo/simpo ÷valid_length（trainer.py:L234-235）；ld_alpha 长度打折分支（L614-632） |
| loss 族 | 仅 sigmoid | `pref_loss` Literal 六项：sigmoid/hinge/ipo/kto_pair/orpo/simpo（finetuning_args.py:L183-186），经 trl DPOTrainer.dpo_loss（trainer.py:L27、L187） |
| ref 模型 | deepcopy + 训练前预算一次 | `compute_reference_log_probs` 每步前向；`use_ref_model` 推导链（finetuning_args.py:L593）——orpo/simpo 免 ref |
| beta | 0.1（主实验）+ 0.5/2.0 扫描 | `pref_beta` 默认 0.1（finetuning_args.py:L171-173） |
| 数据规模 | 6 对、词表 26、全批 | 真实数据集 + cutoff_len 长度预算（pairwise.py:L58-63） |
| 并行/精度 | CPU 单进程 | DeepSpeed/FSDP 接入（trainer.py:L28-29 prepare_deepspeed/prepare_fsdp）+ 混合精度 `[TODO: verify on real system]` |
| 指标 | win/greedy/margin/drift/答案位分布 | rewards、logps、logits 三族（trainer.py:L305-312） |

**nano 侧未做项（LF 有、本级不实现，列出原因）**：

- **pref_ftx**（finetuning_args.py:L175）：DPO 里叠一路 chosen 上的 SFT 辅助
  loss（trainer.py:L301-302，`ftx_gamma > 1e-6` 时生效），防 policy 忘记「怎么
  说话」。nano 的 toy 没有「遗忘」压力，加了反而模糊焦点。
- **dpo_label_smoothing**（finetuning_args.py:L187-190，cDPO 鲁棒化；仅 sigmoid
  族可用，L609 有互斥校验）：对偏好标签本身的不确定性建模——与 [4] 的「标签
  可能错」直接相关，但机制面已由 [4] 的反例覆盖。
- **pref_bco_weight**（finetuning_args.py:L179，BCO 二分类偏好）与 **ld_alpha**
  （长度依赖加权）：单源扩展，机制类别已分别被「比较信号」与「长度归一」覆盖。
- **真实 tokenizer / 模板库 / 多轮**：L3 的领域（配置体系对照）。

---

## 6. 费曼自检

**类比：不看标准答案的教练，和一张入职基线。**

L1 的 SFT 像「老师拿着标准答案逐字批改」。DPO 换了个教练：他不写标准答案，
只看两份答卷说哪份更好。他打分有个规矩——**跟你入职时的基线比**（reference
model），不是跟绝对满分比：「你比基线更倾向好答案了多少」。beta 是这条规矩的
严格程度：beta 大，稍微好一点就满意（sigmoid 早饱和，margin 小）；beta 小，
非要拉开大差距才罢休（margin 大）。

这个类比自带三个推论，都能对上输出：

- 教练只看两份答卷的**相对**好坏——你把答卷顺序故意颠倒（[4]），他照样认真
  地把你往反方向教：比较信号没有方向感，方向来自标签。
- 教练只盯 A、B 两份卷——你把分数全刷到无关的 C 卷上（[3] 的 `'assistant.'`
  =0.9998），他照样判你赢：pair 之外的质量他看不见。
- 基线本身可以是歪的（含噪 ref margin −0.0864）——教练量的是「相对基线的
  改善」，所以歪基线也能教出好学生，只要比较信号方向对。

**自检问**（讲给外行听之前先过自己这关）：

1. 为什么 DPO「不需要 reward model」却**仍然需要一个 reference model**？
   一句话说清 ref 在 loss 里的位置。
2. 为什么 step 0 的 loss 恰好是 ln 2？它不是 ln 2 说明什么？
3. win 6/6 为什么不能保证模型「会答题」？质量可以漏去哪？

---

## 7. 思考题

1. **ln 2 指纹**：DPO step 0 loss = 0.6931 来自 policy ≡ ref。如果你在自己的
   实现里跑出的 step 0 loss 明显偏离 ln 2，列出至少三个可能的接线错误。
2. **质量泄漏诊断**：β=0.5 的模型 win 6/6、greedy 2/6。若你要在训练侧加一个
   廉价探针来捕捉这类「loss 正常、行为崩坏」，你会测什么？（提示：答案位分布、
   pair 外质量、drift；对照 LLaMA-Factory 指标面 trainer.py:L305-312 的盲区。）
3. **「和」vs「平均」**：sigmoid 族用 token logps 之和，ipo/orpo/simpo 除以
   valid_length。构造一个 chosen 比 rejected 长 3 倍的假想批，推演两种聚合下
   margin 的差别，并解释为什么长度归一族对「长回答刷分」更不敏感。
4. **ref 的起点效应**：本实验从含噪 ref（margin −0.0864）出发。若把 ref 换成
   clean SFT 模型（margin 已大幅为正），DPO 还能改善什么、drift 会变大还是变
   小？设计一个实验验证你的猜想（本节代码可直接改）。
5. **防颠倒**：[4] 证明颠倒的偏好对把模型教成自信地答错。真实管线里有哪些
   数据侧机制在防这件事？（提示：RM 打分过滤、多标注者一致性、可验证规则
   RLVR；对照 `../nano-trinity-rft/tutorial_L2.md` 的 reward 来源与
   `../nano-verl/tutorial_L3.md` 的 policy-loss 族。）

---

## 8. 反例与边界

### 8.1 toy 尺度诚实声明

- **74,496 参数字符级 GPT、6 对单位数加法、300 epochs 全批记忆式训练**。
  输出里的数值——margin 数十 nats、p_rejected 压到 4.23e-33、drift 排序
  0.92 < 5.42 < 7.64——是 toy 尺度 + 固定 seed 下让机制**确定性显现**的手段，
  **系数一律不可外推到真实模型**。可迁移的是机制：偏好对 = 两条 SFT 行、
  margin/β/ref 的语义、质量泄漏与方向盲目两个失败模式。
- **drift 排序是优化路径现象**：「β=0.1 走得干净」依赖本 seed 下三条训练路径
  的具体走向；换 seed 或换尺度，排序可能变。机制结论（pair loss 对 pair 外
  质量失明）不依赖排序，探针输出是直接证据。
- **确定性**：固定 seed、CPU、`-B` 任意 CWD，除 `elapsed` 行外逐字节确定
  （掩码锚 `f8b50175…`/51 行，7 跑 × 4 源收敛，见 §1 与 §10）。
- **未覆盖**：在线偏好采样、iterative DPO、真实 tokenizer/模板库、多卡与混合
  精度（`[TODO: verify on real system]`）——前者是算法面扩展，后两者是 L3 与
  真实 GPU/多机环境的领域。

### 8.2 DPO 的当今定位（课程的经典证据层强制声明）

DPO（2305.18290，2023-05）是**经典锚点**，不是当前前沿：

- **仍是地基的**：「reward model 隐于 policy」（r = β·log(π/ref) + const）的
  洞察，以及「chosen/rejected 对 + margin loss」的偏好学习范式。LLaMA-Factory
  的 pref_loss 六族、trl 的 DPOTrainer、SimPO/ORPO 等后续方法全是它的直系
  演化；离线偏好对齐场景（有标注对、无验证器）它仍是默认选项之一。
- **已被替代的**：前沿模型的生产配方转向 **GRPO/RLVR 族**——用可验证规则
  奖励（数学/代码/格式）替代人工偏好标注，用组内相对优势替代 pairwise 比较，
  在线采样替代离线对。锚点：nano-verl L3（verl 的 policy-loss 注册表与
  PPO→GRPO 演化，`../nano-verl/tutorial_L3.md`）、nano-trinity-rft L2/L3
  （DAPO 配置与 reward 来源，`../nano-trinity-rft/`）、01 轨 sota-deepdive
  「后训练算法演进 PPO→GRPO/RLVR→OPD」（`../sota-deepdive/`，课程内交叉引用）。
- **一句话**：学 DPO 学的是偏好学习的机制地基；但别以为前沿模型现在还是拿
  偏好对这么训的——**经典 ≠ 前沿**。

---

## 9. 阶梯预告

L3 = 本模块 README 阶梯表 L3 行：**对照 LLaMA-Factory 配置体系——一个配置
切换 SFT/DPO/PPO 的抽象取舍**（`stage` 字段如何决定数据 processor、trainer、
loss 族的装配；`[TODO: verify source]` 待 L3 动笔时现场核定）。L2 留下的钩子：
本节所有「loss 族 / ref 用不用 / 聚合方式」的分叉（§5 表）在 LF 里都是配置字段
而非代码分支——这正是 L3 的主题。

---

## 10. 溯源与口径声明

- **代码**：`L2_dpo_preference_pairs.py`（524 行，md5
  `fdbbb798516e098369bcd3fd82cac7c6`；2026-08-13 修正一处 docstring 耗时描述，
  与前一版本仅 L29 不同）。
- **运行环境**：`python3`（python 3.13.13，
  torch 2.13.0），CPU。输出保真：§1 paste 块 = 掩码口径
  （`sed '/^[[:space:]]*elapsed/d'`）运行输出，BYTE-IDENTICAL；掩码锚 md5
  `f8b50175a6ee5d1c307a84d827a6ea76`/51 行；digest
  `9353e071cfb4054a6a3649a28c2cc6e7`（脚本自产，raw 输出 L51）。
  七次运行、四份独立输出逐位吻合。
- **LLaMA-Factory 源码**：main @`0bbe481e6e621527284d37f1e13a6b9556c303ec`
  （2026-08-13 codeload tarball 抓取；HEAD 录值经独立复验无漂移）。本文全部
  行号锚点以 **2026-08-13 抓取日**为准，并逐条复验在位：
  `data/collator.py:L564`（PairwiseDataCollatorWithPadding，2n 行 chosen-first
  docstring 逐字）、`data/processor/pairwise.py:L66`（chosen_labels 构造）、
  `train/dpo/trainer.py:L27`（trl DPOTrainer import）/`L187`
  （compute_preference_loss）/`L219-253`（concatenated_forward）/`L234-235`
  （ipo/orpo/simpo 长度归一）/`L237-238`（split(batch//2)）/`L301-302`（pref_ftx
  叠加）/`L305-312`（rewards/logps/logits 指标）、`train/trainer_utils.py:L592`
  （get_batch_logps）/`L607-611`（shifted gather）/`L614-632`（ld_alpha 分支）/
  `L634`（sigmoid 族求和）、`hparams/finetuning_args.py:L171-173`（pref_beta
  0.1）/`L175`（pref_ftx）/`L179`（pref_bco_weight）/`L183-186`（pref_loss
  Literal 六项，默认 sigmoid）/`L187-190`（dpo_label_smoothing）/`L593`
  （use_ref_model 推导）/`L609`（label smoothing 仅 sigmoid 族校验）、
  `extras/constants.py:L50`（IGNORE_INDEX=-100）。
  与 README 08-05 快照录值的漂移一处：`PairwiseDataCollatorWithPadding`
  08-05 录 L553 → 08-13 抓取 L564（录新值，原因 = main 分支上游漂移）。
- **论文**：DPO = arXiv 2305.18290（Rafailov et al., "Direct Preference
  Optimization: Your Language Model is Secretly a Reward Model"），loss 为
  Eq. 7——ar5iv 2026-08-13 抓取逐字核验录值（L2 代码 docstring 载录）；
  export.arxiv.org API 的一次复验尝试超时，因此保留抓取日期并不把网络失败解释为新证据。
  课程的实现参照与证据分层的经典锚点录值同 ID。
- **课程内交叉引用**：L0/L1 数据侧三件套（`tutorial_L0.md` §6/§11、`tutorial_L1.md`
  §3-§5）；nano-verl L3；nano-trinity-rft L2/L3；01 轨 sota-deepdive
  「后训练算法演进」。这些链接提供机制对照，不替代本节 DPO 自身的运行证据。
