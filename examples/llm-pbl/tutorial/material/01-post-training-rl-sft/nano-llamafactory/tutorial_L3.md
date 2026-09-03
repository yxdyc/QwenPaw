# nano-llamafactory L3 — 一个配置，三种方法：stage 分发层的抽象取舍与它不许触碰的数字

> **K+1 位置**：L0–L2 建立了数据侧三件套（template / labels mask / collator）、
> 真实 SFT 循环、DPO 偏好对——但三级都是「手写」的：每个方法一条独立代码路径。
> 本级回答一个框架级问题：**为什么 LLaMA-Factory 用一个 `stage` 配置字段就能切换
> SFT/DPO/KTO，而不是一套方法一个代码库？** 拆到底，dispatch 只有三层——
> 配置层（推导标志 + fail-loud）、分发层（表，不是分支；数据层按**数据形状**分发）、
> 执行层（一个 `pref_loss` 字符串同时决定三件事）。而本级最核心的命题要机器证明：
> **好的分发层在数值上是惰性的——它改变代码组织，不改变一个数字**（跨级
> bit-for-bit 锚：L3 经配置分发跑出的 sft 与 dpo-sigmoid 与 L2 手写路径逐位相同）。
>
> **对标权威实现**：[LlamaFactory](https://github.com/hiyouga/LlamaFactory) 固定 revision
> [`f28afaf6355af515454dfb16c97d728307c93897`](https://github.com/hiyouga/LlamaFactory/tree/f28afaf6355af515454dfb16c97d728307c93897)，
> 以及 TRL v0.24.0。本文源码行号只对该 dated snapshot 有效，不能当作当前 main 的永久行号。
>
> **时效性定位（必读 §8）**：LlamaFactory 是工程参照；
> 但 **DPO 族 = A 层经典机制 ≠ 前沿**——前沿生产配方（GRPO 族 / RLVR / OPD）不走
> 这里的 `pref_loss` 路径。

---

## 1. 先跑起来

**可运行性契约声明**：本级的三个 trainer（sft/dpo/kto）全部是
**真实 torch 梯度下降**，不是 mock——配置分发、loss 族切换、KTO 的 KL 基线，每一步
都在真训练里走。toy 的只是规模（74,496 参数字符级 GPT、6 对单位数加法 / 12 条
单侧反馈、CPU 秒级），目的是让「分发层的数值惰性」在固定 seed 下**逐字节确定地**
显现。真实框架的 LoRA / 量化 / deepspeed / 多卡路径不在本级证据范围内；本节机制与
固定 LlamaFactory revision 对照（§7）。

```bash
python3 -B L3_stage_dispatch.py   # 仅依赖 torch（CPU 即可，无 transformers）；实测 ~6s；任意 CWD 可跑
```

固定 seed（SEED=42）+ 全批训练下，除 `elapsed` 计时行外逐字节确定（掩码口径
`sed '/^[[:space:]]*elapsed/d'`）。2026-08-31 在两个新建空 CWD 中以 `-B` 复验，
均 EXIT=0、stderr 0 B；掩码输出 md5 `56611328be39c8d75cb85cb74f14526a`/64 行/3,598 B，
逐字节一致，digest `6a9e099db7c7c7cb14e076ea4e063134`：

```text
====================================================================
nano-llamafactory L3 — one config, three methods: stage dispatch
====================================================================
vocab size = 26
model params = 74,496

[0] config layer: derived flags + fail-loud validation
    (a) use_ref_model derivation (finetuning_args.py:L593 mirror):
        pref_loss=sigmoid -> use_ref_model=True
        pref_loss=ipo     -> use_ref_model=True
        pref_loss=orpo    -> use_ref_model=False
        pref_loss=simpo   -> use_ref_model=False
    (b) fail-loud at config time (not mid-training):
        dpo+ipo+smoothing=0.1 -> ConfigError: `dpo_label_smoothing` is only valid for sigmoid loss function.
        stage=grpo            -> ConfigError: Unknown task: grpo.
    (c) silent-inert field, made loud by nano (LF stays silent):
        [nano warning] pref_ftx=0.5 has no effect at stage=sft (only the dpo trainer reads it)

[1] dispatch layer: tables, not branches
    STAGE2DATA_STAGE = {'sft': 'sft', 'dpo': 'rm', 'kto': 'kto'}
    (dpo -> 'rm': the data layer dispatches on DATA SHAPE, not method
     name; dpo/workflow.py:L45 calls get_dataset(..., stage='rm'))
    dpo resolves to the SAME processor object as rm: True
    one row under three processors (sft / dpo / kto):
        input_ids+attention_mask+labels byte-identical: True  (supervised tokens = 3)

[2] stage=sft (cross-level anchor: must reproduce L2 noisy SFT bit-for-bit)
    final loss = 0.2334  (L2 recorded 0.2334)
    win=4/6  greedy=4/6  p_chosen=0.4785  p_rejected=0.5205
    cross-level match vs L2: True

[3] stage=dpo pref_loss=sigmoid beta=0.1 (cross-level anchor: must reproduce L2 [2] bit-for-bit)
    step   0: loss=0.6931  margin=-0.0864  pair_acc=1/6
    step  40: loss=0.0065  margin=+54.4843  pair_acc=6/6
    step  80: loss=0.0008  margin=+78.4252  pair_acc=6/6
    step 120: loss=0.0005  margin=+83.0357  pair_acc=6/6
    step 160: loss=0.0004  margin=+86.5248  pair_acc=6/6
    step 199: loss=0.0003  margin=+89.4095  pair_acc=6/6
    final: win=6/6  greedy=6/6  margin=+89.4783  p_rejected=4.23e-33
    drift KL = 0.9230 nats   implicit reward gap = +8.9565
    cross-level match vs L2: True
    (dispatch layer is numerically INERT: it reorganizes code,
     it does not change a single number)

[4] one string, three behaviors: pref_loss sweep (same data, same seed)
    family   needs_ref  aggregate  init_loss   final_loss  win  greedy  margin
    sigmoid  True       sum           0.6931     0.0004  6/6  6/6  +85.7116
    ipo      True       avg          25.0000     0.0000  6/6  4/6  +14.9140
    orpo     False      avg           0.3286     0.0001  6/6  6/6   +9.6273
    simpo    False      avg           0.9759     0.0119  6/6  4/6  +170.7879
    init-loss analytical anchors (policy == ref => margin cancels to 0):
        sigmoid : computed 0.693147  == ln 2 = 0.693147  (True)
        ipo     : computed 25.000000  == 1/(4 beta^2) = 25.000000  (True)
    ref forwards used: orpo=0, simpo=0, sigmoid=1, ipo=1  (config-derived use_ref_model respected: True)

[5] stage=kto: unpaired feedback + cyclic-shift KL baseline
    kl baseline row[i] == response[i-1 mod 12] (cyclic shift, feedback.py:L87 mirror): True
    after 200 steps: loss 0.5000 -> 0.2409  (kl estimate = 0.0000 >= 0 by clamp)
    beta*logratio: desirable mean = +0.0748, undesirable mean = -7.5179
    greedy correct = 6/6  (noisy ref was 4/6; no pairs used, tags only)

====================================================================
[self-check] 14/14 PASS
digest: 6a9e099db7c7c7cb14e076ea4e063134
```

14 条 self-check 全绿。下面按三层 dispatch 逐段拆，最后证明核心命题（§6）。

---

## 2. 问题设定：L2 留下三个债

L2 手写了一条 DPO 路径：数据侧复用三件套、loss 侧手写 `-log σ(β·(margin_policy −
margin_ref))`。能跑、能讲清机制，但它欠了三笔框架级的债：

1. **切换债**：真实系统里没人手写 DPO——用户改一行配置（`stage: dpo`）就切换了
   方法。这个「一个字段切方法」的机制，抽象代价是什么？会不会改变训练的数字？
2. **形状债**：L2 只讲了偏好对这一种数据形状。KTO（arXiv 2402.01306）的数据是
   **单侧响应 + 二值 tag**，没有「对」——它怎么住进同一套数据侧机制？
3. **惰性债**：L2 的 digest 是 `9353e071…`。如果 L3 把同一套训练重新组织进
   「配置 → 分发 → 执行」三层，跑出来的数字还和 L2 逐位相同吗？

第 3 笔债是本级的灵魂。**抽象不是免费的**：每一层分发都是一个可以悄悄引入数值
漂移的机会（种子位点不同、collate 顺序不同、归一化时机不同）。如果重构后数字
变了，要么重构引入了 bug，要么「抽象」本身偷偷参与了计算。L3 用内置的
`L2_ANCHOR`（15 个录值）把这笔债机器化：`cross-level match vs L2: True` 不是
日志，是断言——不匹配就 EXIT≠0。

代码结构（单文件 899 行）：数据侧（与 L0–L2 逐位同构的 template/mask/collator/
TinyLM/seq_logps）→ 配置层 `NanoFactoryConfig` → 分发层 `STAGE2DATA_STAGE` +
`DATA_STAGE2PROCESSOR` 两张表 + `build_trainer` → 执行层 `PREF_LOSS_REGISTRY` +
三个真 trainer → 测量面（与 L2 定义逐位相同的 evaluate_pairs / drift_kl /
first_gen_token）→ `L2_ANCHOR` + 14 条 self-check + digest。

---

## 3. [配置层] 配置自己算出自己的后果

### 3.1 一个扁平 dataclass：方法族字段与 stage 同住

LlamaFactory 的 `FinetuningArguments`（`hparams/finetuning_args.py`）把**所有**
训练方法相关的字段放在同一个扁平 dataclass 里：

```python
# finetuning_args.py:L460（@f28afaf6，2026-08-16 抓取件）
stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(...)
# finetuning_args.py:L183
pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(...)
```

nano 版取它的 dispatch 相关切片（`NanoFactoryConfig`）：`stage` / `pref_loss` /
`pref_beta`（默认 0.1，LF L171）/ `pref_ftx`（默认 0.0，L175）/
`dpo_label_smoothing`（默认 0.0，L187）/ `kto_chosen_weight`、`kto_rejected_weight`
（默认 1.0，L191/L195）/ `simpo_gamma`（默认 0.5，L199）。

为什么扁平而不是「每个方法一个配置类」？因为**方法族字段之间有共享结构**：
`pref_beta` 同时服务 sigmoid/ipo/orpo/simpo/kto 五族；`use_ref_model` 由 stage 与
pref_loss **联合**决定。继承树（DPOConfig/KTOConfig/...）会把共享字段复制五份，
而扁平 dataclass 让「哪些字段被哪个方法读」变成一个可以在一张表里审查的问题。
代价是配置空间里存在**非法组合**（`stage=sft + pref_loss=ipo`、
`dpo+ipo+label_smoothing>0`）——这由配置层的第二件事兜住。

### 3.2 推导标志：use_ref_model 不是用户传的，是配置算出来的

```python
# finetuning_args.py:L593（逐字）
self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
```

nano 逐字镜像（`_post_init`）。输出 `[0] (a)` 演示推导结果：

```text
pref_loss=sigmoid -> use_ref_model=True
pref_loss=ipo     -> use_ref_model=True
pref_loss=orpo    -> use_ref_model=False
pref_loss=simpo   -> use_ref_model=False
```

为什么**推导**而不是让用户显式传 `use_ref_model=True/False`？因为它是
`(stage, pref_loss)` 的**函数**，不是独立自由度：orpo/simpo 的数学定义里就没有
ref（§5），传 `use_ref_model=True + pref_loss=orpo` 是一个无意义的组合。让配置
「算出自己的后果」（single source of truth），非法组合在定义上就不存在——这比
「允许传但校验」更彻底。后面 §5 会看到，这个推导标志如何一路传导成「orpo/simpo
全程 0 次 ref 前向」的机器记录。

### 3.3 fail-loud 在配置解析时，不在训练中途

```python
# finetuning_args.py:L609-610（逐字，含消息）
if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
    raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")
```

nano 逐字镜像消息（`ConfigError`），输出 `[0] (b)` 演示两条 fail-loud：

```text
dpo+ipo+smoothing=0.1 -> ConfigError: `dpo_label_smoothing` is only valid for sigmoid loss function.
stage=grpo            -> ConfigError: Unknown task: grpo.
```

第二条镜像 `tuner.py:L150-151` 的 else 分支（`raise ValueError(f"Unknown task:
{finetuning_args.stage}.")`，逐字同款消息）——nano 把它提前到配置解析时。
**错误发现的成本随阶段指数上升**：配置时报错 = 用户改一行 yaml；训练中途报错 =
几小时 GPU 时间 + 半截 checkpoint；训练完才暴露（比如发现 smoothing 被静默忽略）=
一次实验的结论作废。配置层是错误成本曲线的最低点，所以边界校验全部堆在这里。

注意 `[0] (b)` 的第二条输出同时是 §8 的机器证据：**`stage=grpo` 在 LF 的 stage
体系里是 `Unknown task`**——RLVR 主流方法不住在这套分发体系里。

### 3.4 nano 增补（LF 没有）：静默无效字段大声化

```text
[nano warning] pref_ftx=0.5 has no effect at stage=sft (only the dpo trainer reads it)
```

`pref_ftx`（SFT 辅助 loss 叠加系数，只有 dpo trainer 读它，LF `trainer.py:L300-302`）
在 `stage=sft` 下设了也白设——LF 对此**保持沉默**，nano 选择大声化。这是 nano
版唯一主动偏离权威实现的地方，显式声明：教学上，「框架的沉默不是承诺」是比
「与 LF 逐字一致」更值钱的教训（§11 反例 2 展开）。

---

## 4. [分发层] 表，不是分支；数据层的键是数据形状，不是方法名

### 4.1 stage → workflow：一条 if/elif 链的诚实形态

LF 的 `tuner.py:L138-151` 就是六个 `elif finetuning_args.stage == "..."` 加一个
`else: raise`。没有注册表魔法，没有插件系统——**分发层的本质是一张有限表**，
if/elif 只是它的代码形态。nano 的 `build_trainer` 同款：

```python
def build_trainer(vocab, cfg):
    if cfg.stage == "sft":  return SFTTrainer(vocab, cfg)
    if cfg.stage == "dpo":  return DPOTrainer(vocab, cfg)
    if cfg.stage == "kto":  return KTOTrainer(vocab, cfg)
    raise ConfigError(f"Unknown task: {cfg.stage}.")   # tuner.py:L150-151 同款
```

表驱动的价值不在「好看」，在**可审查**：一个新人要回答「这个框架支持哪些训练
方法」，只需读这一张表（LF 是 14 行，nano 是 8 行），不用全仓库 grep。

### 4.2 数据层没有 "dpo" 这个键

这是本级第二个反直觉点。nano 的训练 stage → 数据 stage 映射：

```python
STAGE2DATA_STAGE = {"sft": "sft", "dpo": "rm", "kto": "kto"}
```

**dpo 的数据走 `stage="rm"` 的 pairwise 管线**。LF 源码里这是白纸黑字：

```python
# train/dpo/workflow.py:L45（逐字）
dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
```

LF 数据层的 stage Literal 是 `["pt","sft","rm","ppo","kto"]`（`data/loader.py:L169`）
——五项，**没有 dpo**。为什么？因为**数据层的分发键是数据形状，不是方法名**：
偏好对（chosen/rejected）就是 reward model 训练的数据形状，dpo 与 rm 消费同一种
行结构（pairwise processor：每条偏好对渲染成 chosen 行 + rejected 行，
`data/processor/pairwise.py:L66` 的 `chosen_labels = [IGNORE_INDEX] * source_len
+ chosen_ids`）。方法名（rm 用 pair 训打分器、dpo 用 pair 算 margin）是**消费侧**
的差异，与数据形状无关。

输出 `[1]` 给出两个机器证明：

```text
dpo resolves to the SAME processor object as rm: True
one row under three processors (sft / dpo / kto):
    input_ids+attention_mask+labels byte-identical: True  (supervised tokens = 3)
```

第一条：`resolve_processor("dpo")` 与 `DATA_STAGE2PROCESSOR["rm"]` 是**同一个
对象**（`is`，不是「相等」）。第二条：同一条 `(system, "Compute 2+2", "4")`，
经 sft / dpo / kto 三个 processor 渲染，`input_ids + attention_mask + labels`
逐位相同——数据侧三件套（L0–L1 的遗产）在三个 stage 之间**零改动复用**，
supervised tokens = 3（answer + `\n` + `<|eot|>`，prompt 全 -100）。

这个设计的回报：**新增方法不触碰数据层**。明天 LF 加第七种 pref_loss 族，
只要它吃偏好对，数据层一行不改。反过来想（§10 思考题 1）：如果数据层按方法名
分发，每加一个方法都要回答「它的数据键叫什么」，数据层就从「五种形状」膨胀成
「N 种方法」——把变化的维度（方法）耦合进了稳定的维度（形状）。

---

## 5. [执行层] 一个字符串决定三件事

### 5.1 PREF_LOSS_REGISTRY：loss 族的三个后果

nano 把每个 pref_loss 族注册成一条三后果的记录：

```python
PREF_LOSS_REGISTRY = {
    "sigmoid": dict(needs_ref=True,  aggregate="sum", fn=_sigmoid_loss, paper="2305.18290"),
    "ipo":     dict(needs_ref=True,  aggregate="avg", fn=_ipo_loss,     paper="2310.12036"),
    "orpo":    dict(needs_ref=False, aggregate="avg", fn=_orpo_loss,    paper="2403.07691"),
    "simpo":   dict(needs_ref=False, aggregate="avg", fn=_simpo_loss,   paper="2405.14734"),
}
```

三个后果在 LF 里住在三个地方：`needs_ref` 镜像 `finetuning_args.py:L593`（推导
标志）+ `trainer.py:L195-208` 的无 ref 分支；`aggregate` 镜像 `trainer.py:L234-235`
（`if self.loss_type in ["ipo","orpo","simpo"]: all_logps = all_logps / valid_length`，
逐字）——ipo/orpo/simpo 用**长度归一后**的均值 logp，sigmoid 用求和；loss 公式
本身：sigmoid/ipo 对照 trl v0.24.0 `dpo_trainer.py`（sigmoid 支 L1110-1114：
`-F.logsigmoid(self.beta * logits) * (1 - label_smoothing) - F.logsigmoid(-self.beta
* logits) * label_smoothing`，label_smoothing=0 时即 `-logsigmoid(β·logits)`；
ipo 支 L1135-1137：`(logits - 1 / (2 * self.beta)) ** 2`，逐字），orpo/simpo 是
LF 原生实现（`trainer.py:L150-158` odds_ratio_loss / `L160-166` simpo_loss）。

**为什么一个字符串同时决定三件事，而不是三个独立开关？** 因为这三件事是同一个
数学定义的三个工程后果：orpo 的公式里**没有 ref 项**（`sft_loss + β·odds_ratio`，
ref 无从出现）、它的 logps **必须**长度归一（odds ratio 对长度敏感）。如果把
「用不用 ref」「求和还是平均」拆成独立配置项，配置空间就会出现「ipo + 不用 ref +
求和」这种数学上无意义的组合——与 §3.2 同理：**后果跟着定义走，不给用户拆开关**。

### 5.2 一串三行为：pref_loss 扫描的机器证据

输出 `[4]` 是同数据、同 seed 下四族扫描：

```text
family   needs_ref  aggregate  init_loss   final_loss  win  greedy  margin
sigmoid  True       sum           0.6931     0.0004  6/6  6/6  +85.7116
ipo      True       avg          25.0000     0.0000  6/6  4/6  +14.9140
orpo     False      avg           0.3286     0.0001  6/6  6/6   +9.6273
simpo    False      avg           0.9759     0.0119  6/6  4/6  +170.7879
```

三个观察。第一，**四族真的不同**（self-check 第 14 条断言末步 margin 不全相等）：
ipo/simpo 的 greedy 只有 4/6——长度归一改变了长答案的相对代价，win 6/6 与
greedy 4/6 并存是 L2 §[3] 讲过的「pair loss 对 pair 之外质量失明」在方法族维度
的重演。第二，**init_loss 有闭式锚**：初始时 policy == ref，有 ref 的族里
margin_policy 与 margin_ref 逐项相消为 0——sigmoid 得 `-log σ(0) = ln 2 =
0.693147`，ipo 得 `(0 − 1/(2β))² = 1/(4β²) = 25.000000`（β=0.1），两条
`computed == formula` 全 True。无 ref 族（orpo/simpo）没有这个相消，init loss
取决于真实（非零）margin，**没有闭式锚**——代码里只对两族断言，不是偷懒，是
数学边界。第三，**ref 前向次数是机器记录的**：

```text
ref forwards used: orpo=0, simpo=0, sigmoid=1, ipo=1  (config-derived use_ref_model respected: True)
```

`n_ref_forwards` 由 trainer 自己计数：配置推导 `use_ref_model=False` 的族，
**一次 ref 前向都不跑**（不是「跑了不用」，是 `ref_sum = None` 直接不算）。
§3.2 的推导标志在这里兑现成可测量的计算量差异——生产尺度上，这就是「orpo/simpo
省一半前向」的机制来源（ref 模型整个可以不加载）。

### 5.3 KTO：第三种数据形状，和一个循环移位技巧

KTO（arXiv 2402.01306）的数据没有「对」：每条样本是 `(prompt, response, tag)`，
tag ∈ {desirable, undesirable}。数据形状变了，所以它是第三个 stage 而不是第六个
pref_loss——`feedback_process` 返回 `(rows, kl_rows, tags)`，比 pairwise 多带一个
KL 基线样本。

KTO 的 loss（对照 trl v0.24.0 `kto_trainer.py:L1150-1191`）：

```text
kl = mean(policy_kl_logps - ref_kl_logps).detach().clamp(min=0)     # L1150-1151
chosen_losses   = 1 - sigmoid(beta * (logratio_c - kl))             # L1161
rejected_losses = 1 - sigmoid(beta * (kl - logratio_r))             # L1179
losses = cat(lambda_D * chosen_losses, lambda_U * rejected_losses)  # L1189-1191
```

KL 基线样本从哪来？LF 数据侧的答案是**批内循环移位**：

```python
# data/processor/feedback.py:L87（逐字）
kl_response = [examples["_response"][-1]] + examples["_response"][:-1]
```

第 i 条样本的 KL 响应 = 第 i−1 条的响应（模批大小）。输出 `[5]` 第一行是**双向
索引机器证明**：`rows[i] == kl_rows[(i+1) % 12]` 正向全过 + `kl_rows[i] ==
rows[(i−1) % 12]` 反向全过。为什么循环移位？因为 KL 基线需要「同分布但不同
内容」的响应——批内任意一条别人的响应都近似满足，循环移位是最便宜的确定性
取法（无随机、无额外前向）。

训练结果（无偏好对，只有 tag）：

```text
after 200 steps: loss 0.5000 -> 0.2409  (kl estimate = 0.0000 >= 0 by clamp)
beta*logratio: desirable mean = +0.0748, undesirable mean = -7.5179
greedy correct = 6/6  (noisy ref was 4/6; no pairs used, tags only)
```

初始 loss 0.5000 = 两侧各半的 `1 − σ(0)` 加权和（闭式，与 sigmoid 的 ln2 同族
的解析锚）；末步 desirable 的 β·logratio 均值 +0.0748 > undesirable 的
−7.5179——**单侧二值信号足以把六种加法的 greedy 从 4/6 推到 6/6**，全程没有
一对偏好对。`kl estimate = 0.0000` 是 clamp 贴零：toy 尺度 policy 漂移小，
KL 批均值 <0 被截到 0——演示的是 clamp 机制（KTO 论文要求 KL 基线非负），
不是生产里 KL 真为 0（§11 反例 5）。

---

## 6. 核心命题：分发层的数值惰性（跨级 bit-for-bit 锚，本课程首例）

现在兑现 §2 的第 3 笔债。命题：

> **好的分发层在数值上是惰性的：它改变代码组织，不改变一个数字。**

机器证明在输出 `[2]` 与 `[3]`：

```text
[2] stage=sft ...  cross-level match vs L2: True
[3] stage=dpo pref_loss=sigmoid ...  cross-level match vs L2: True
```

`L2_ANCHOR` 内置 15 个录值（noisy SFT 的 final loss / win / greedy / p_chosen /
p_rejected 五值 + DPO 的 step0 loss / step0 margin / step0 pair_acc / 末步 margin /
win / greedy / final margin / p_rejected / drift / gap 十值），来源是 L2 教程 §1
paste 块（L2 digest `9353e071cfb4054a6a3649a28c2cc6e7`）。L3 经「配置 →
build_trainer → DPOTrainer.fit」分发路径跑出的每一个数字，与 L2 手写
`train_dpo(...)` 路径**逐位相同**：0.2334 / win4 / greedy4 / 0.4785 / 0.5205 /
0.6931 / −0.0864 / +89.4095 / +89.4783 / 4.23e-33 / 0.9230 / +8.9565。
L2 自身的 15/15 self-check 与 digest 是跨级声明源；L3 的两个独立 fresh-CWD 运行又逐位命中
内置的 `L2_ANCHOR`，共同把这条命题变成机器可判决的契约。

为什么这条命题值得单独证明？因为**每一层分发都是一个漂移机会**：

- 种子位点：L3 的 `set_seed(SEED)` 必须落在与 L2 `main()` 相同的逻辑位点
  （noisy SFT 前、DPO 前各一次），否则初始化不同，全线漂移；
- 数据顺序：`pairwise_process` 必须 chosen 前 6 行、rejected 后 6 行（LF
  `collator.py:L564` 的 chosen-first 顺序，L2 §[0] 已证），否则 `split(n)` 语义反转；
- 归一化时机：`aggregate="sum"` 族必须与 L2 一样不做长度归一，一个 `/vlen`
  就是另一条曲线。

bit-for-bit 相同是「抽象惰性」的**最强**保证——比「loss 曲线看起来差不多」强，
因为后者容忍 O(ε) 的静默漂移，而 O(ε) 正是重构 bug 的典型尺度。工程推论有两条：
其一，LF 可以放心重构分发层（换注册表、换 workflow 组织），只要复现录值逐位吻合，
可复现性记录就不作废；其二，**「重构前后数字逐位对齐」是重构的验收标准**，
「看起来一样」不是。

命题的边界见 §11 反例 1：惰性是**分发层**的性质，不是整个系统的性质——它成立
的前提是同 seed / 同数据顺序 / 同精度 / 同计算顺序。

---

## 7. 权威实现取舍表：nano 版没做什么

行锚全部针对 LlamaFactory revision `f28afaf6`；读者应通过上方固定 revision 链接复核，
不要把这些行号套到当前 main。

| 维度 | nano L3 的选择 | LlamaFactory 的选择（行锚） | 差异原因 |
|------|---------------|------------------------------|----------|
| 配置载体 | 手写 `NanoFactoryConfig`（只留 dispatch 相关 8 字段） | HfArgumentParser + 多组 dataclass（`finetuning_args.py` 全文 600+ 行，stage/pref_* 只是其中切片：L460/L183/L171/L175/L187/L191/L195/L199） | nano 只要「能演示 dispatch」的字段；LF 要承载真实训练的全部旋钮 |
| stage 集合 | sft/dpo/kto 三种 | pt/sft/rm/ppo/dpo/kto 六种（L460） | rm 要打分器头、ppo 要生成 rollout + value 模型、pt 是无遮罩数据流——各自需要 nano 不演示的额外 infra |
| pref_loss 族 | sigmoid/ipo/orpo/simpo 四族 | 六族（+hinge/kto_pair，L183） | 六族机制同构（registry 同款），nano 用四族演示机制，不重复实现 |
| 推导标志 | `use_ref_model` 逐字镜像（L593） | 同 | 无差异——这是配置层的灵魂，必须逐字 |
| fail-loud | 两条（smoothing@非 sigmoid L609-610；Unknown task L150-151），消息逐字 | 同（LF 在 `__post_init__` 与 tuner else 分支） | nano 额外增补 pref_ftx 静默无效警告（LF 没有，§3.4 声明） |
| 数据层 | `STAGE2DATA_STAGE` 三键表，dpo→rm | 数据层五键（loader.py:L169 Literal 无 dpo），dpo workflow 显式传 `stage="rm"`（workflow.py:L45） | 无本质差异——nano 把 LF 的隐含映射显式成表 |
| ref 前向 | 冻结 ref **预算一次**（`ref_sum` 复用，n_ref_forwards 机器记录） | 每步走 ref 前向（`trainer.py:L187` compute_preference_loss 每步调用） | LF 的统一代码路径不为「ref 冻结」特化；nano 利用 L2 已证的「ref 冻结 ⇒ 预算一次」不变量省掉每步前向——数值结果不变（§6），成本不同 |
| sigmoid/ipo 公式 | 手写（镜像 trl） | 走 trl DPOTrainer（`pyproject.toml:L47` 钉死 `trl>=0.18.0,<=0.24.0`；trl dpo_trainer sigmoid L1110-1114 / hinge L1132-1133 / ipo L1135-1137） | LF 对 trl 是「能借则借」：sigmoid/ipo/hinge 等复用 trl 的成熟实现 |
| orpo/simpo 公式 | 手写（镜像 LF） | **LF 原生实现**（trainer.py:L150-158 odds_ratio_loss / L160-166 simpo_loss，不走 trl） | trl 当年没有 orpo/simpo，LF 自己补——「缺什么补什么」的混合策略 |
| KTO KL 基线 | 循环移位逐字镜像（feedback.py:L87）+ 双向索引证明 | 同（数据侧 feedback.py:L87 + loss 侧走 trl kto_trainer L1150-1191） | 无本质差异 |
| pref_ftx 训练路径 | 代码实现了（L474-477 镜像 trainer.py:L300-302），但 demo **未行使**（只走配置警告路径） | 完整行使 | 行使它需要构造「smoothing 族 + ftx」组合实验，与本级命题（dispatch 惰性）无关；诚实清单在此 |
| 其余未做 | — | LoRA/量化/deepspeed 集成、WebUI、megatron bridge、rm/ppo/pt 三个 stage、bco/ld_alpha 等 trl 更新选项 | 与「stage dispatch 的抽象取舍」这一核心机制无关；真实系统验证走 `[TODO: verify on real system]`（§13） |

---

## 8. LlamaFactory 的时效性定位

三层定位，分开说清：

**工程锚点（§五 权威实现）**：LlamaFactory 是后训练 infra 的权威开源实现之一，
本模块对标的就是它的**配置-分发体系**——「一个 stage 字段切方法 + 扁平
dataclass + 数据层按形状分发」是当今一站式微调框架的主流工程实践形态。
SOTA 对齐日期 = 2026-08-16（tarball 抓取日）；顶 commit「[v1] add FSDPTurbo
EP/EFSDP plugin for MoE training (#10676)」（2026-08-13T12:45:55Z）说明上游
仍在高频演进（v1 架构重构线活跃），行锚以抓取日为准、漂移需重抓核对。

**DPO 族 = A 层经典机制 ≠ 前沿**：sigmoid/ipo/orpo/simpo/kto 教的是偏好学习的
机制本质——margin、ref 锚、KL 约束、长度归一的代价——这些概念流入了一切后续
方法。但**前沿模型的生产配方（GRPO 族 / RLVR / OPD）不走 llamafactory 的
`pref_loss` 路径**。脚本证据就在输出里：`stage=grpo -> ConfigError: Unknown
task: grpo.`——LF 的 stage 体系（截至 08-16 快照）根本不包含 RLVR 方法。
「经典 ≠ 前沿」的分工：学机制本质来本级，学前沿生产配方去 nano-verl L2/L3
（RLVR infra）与 nano-opd（on-policy distillation）。

**上游改名声明**：repo 于 2026 年从 `hiyouga/LLaMA-Factory` 改名为
`hiyouga/LlamaFactory`（lf_commits.atom title「Recent Commits to
LlamaFactory:main」坐实当今名），包名仍为 `llamafactory`。L2 教程引用的旧名
快照 `0bbe481e`（08-13 抓取）录值在其抓取日语境下续真。

---

## 9. 费曼自检

**类比：一家餐厅的点单系统**。客人只说一句话——「麻辣香锅，微辣」（= 一个
`stage` + 一个 `pref_loss` 字段）。前台（配置层）从这句话**推导出**一串后果：
要不要问花生过敏（`use_ref_model`——是这句话算出来的，不是客人另外填的表）、
走哪个备餐间、按份还是按重计费。如果你点「油炸冰淇淋」，前台**当场拒绝**
（fail-loud 在点单时，不在厨房开工后——错误发现的成本随阶段指数上升）。
点单票通过转单窗口（分发层）到对应档口——窗口是**一张表**不是一个人，而且
「麻辣香锅」和「冒菜」在数据档口是**同一种原料形状**（dpo 与 rm 共用 pairwise
管线：分发键是原料形状，不是菜名）。最后，「微辣」这**一个字符串同时决定
三件事**：配方（loss 公式）、要不要备一份对照餐（用不用 ref）、按份还是按重
计费（求和还是平均）——因为这三件事都是「微辣」这个定义本身的后果，不给客人
拆成三个开关。而整家餐厅最重要的纪律：**无论怎么重组转单流程，菜的味道不许
变**（分发层数值惰性）——味道变了，说明「重组」偷偷改了配方，那是事故不是
重构。

自检三问（讲不出来就回对应节重读）：

1. 为什么 `use_ref_model` 是**推导**出来的而不是让用户传？（§3.2：它是
   (stage, pref_loss) 的函数，不是独立自由度——传了就会出现数学上无意义的组合）
2. 为什么 dpo 的数据走 `stage="rm"` 而不是自建一条「dpo 数据管线」？
   （§4.2：数据层按数据形状分发，偏好对就是 rm 的形状；方法名是消费侧差异）
3. 「分发层数值惰性」是什么意思？它的前提条件是什么？（§6/§11 反例 1：
   重构前后数字逐位相同；前提 = 同 seed / 同序 / 同精度 / 同计算顺序）

---

## 10. 思考题

1. **加一族要动几个文件？** 假设要给 LF 加第七个 pref_loss 族（比如某新变体），
   列出必须改的文件与不必改的文件。（提示：必改 = `finetuning_args.py` 的
   Literal + loss 实现处（trl 或 trainer.py）+ 可能的聚合分支；不必改 = 数据层
   全部——只要新族吃偏好对。这正是 §4.2 设计回报的兑现。）
2. **ref 预算一次的代价**：nano 把冻结 ref 预算成一次 `ref_sum`，LF 每步走 ref
   前向。什么场景下 LF 的「浪费」是必要的？（提示：ref 不冻结的算法 [某些
   定期更新 ref 的变体]；以及 LoRA 场景下 ref 前向可能有更便宜的等价物——
   具体 LF 如何实现属开放问题 `[TODO: verify]`，不要凭印象回答。）
3. **设计一个生产级惰性探针**：十亿参数 + 混合精度下 bit-for-bit 不再可能，
   你怎么验证「重构没改数字」？给出从强到弱的三级方案。（提示：同 seed 同序的
   张量级 allclose 容差 → 关键指标锚 [eval/loss 曲线容差带] → 统计检验
   [多次 seed 的分布比较]；每一级各能抓住哪类漂移？）
4. **KTO 的 batch=1 退化**：循环移位 KL 基线在 batch size = 1 时发生什么？
   （提示：单条样本的移位是自己，kl_logps − ref_kl_logps = 0，KL 基线恒 0——
   这是退化还是设计边界？LF 真实小批处理 `[TODO: verify]`，别凭印象。）
5. **沉默还是大声**：`pref_ftx@sft` 在 LF 里静默、nano 选择警告。如果你是 LF
   维护者，选哪个？各自的成本是什么？（提示：警告的成本 = 老配置升级时的误报
   噪声与向后兼容负担；沉默的成本 = 用户带着无效配置跑完整个实验。哪个成本
   更高取决于你的用户画像——这就是工程取舍，没有标准答案。）

---

## 11. 反例与边界

1. **「数值惰性」命题的边界**：惰性是分发层的性质，前提 = 同 seed / 同数据顺序 /
   同精度 / 同计算顺序。反例构造很容易：把 `pairwise_process` 改成 rejected 在前
   chosen 在后，`split(n)` 语义立刻反转，全线数字面目全非——数据顺序是数值契约
   的一部分，不是「实现细节」。命题说的是「**分发层**不引入漂移」，不是「随便
   重构都不漂移」。
2. **配置字段不是承诺**：`pref_ftx=0.5 + stage=sft` 在 LF 里被**静默丢弃**
   （只有 dpo trainer 读它）——「字段存在」≠「字段有效」。LF 的校验覆盖是
   不完全的：`dpo_label_smoothing@ipo` 被抓（L609-610，输出 `[0](b)` 演示），
   `pref_ftx@sft` 不被抓。真实框架的验证层永远是「已知非法组合的清单」，不是
   「所有无意义组合的拦截器」。
3. **形状分发的失败模式**：若 dpo 数据不是偏好对形状（比如只给了 chosen 侧），
   错误在**数据层**就被抓住（pairwise processor 解不出四元组），而不是漂到 loss
   层变成 NaN——分发键是形状，形状错误的拦截点就在形状检查处。这是「按形状
   分发」附赠的故障定位性。
4. **toy 尺度诚实声明**：margin +89、p_rejected 4.23e-33 是 6 对数据 +
   74,496 参数 + 200 步全批的 toy 数字，**不可外推**到生产尺度；init 闭式锚
   （ln2、1/(4β²)）的绝对值有意义（数学性质，与尺度无关），训练曲线的绝对值
   没有。
5. **kl_last = 0.0000 不是「KTO 的 KL 真为 0」**：toy 尺度 policy 漂移小，
   KL 批均值 <0 被 clamp 截到 0——演示的是 clamp 机制（KTO 要求非负 KL 基线），
   生产尺度上 KL 估计通常显著为正。

---

## 12. 阶梯预告

nano-llamafactory 至此 L0–L3 完整：数据侧三件套（L0）→ 真实 SFT 循环（L1）→
DPO 偏好对（L2）→ stage 分发层的抽象取舍（本级）。同一机制光谱的另外两端：
RLVR/GRPO 族的 infra 取舍在 [nano-verl](../nano-verl/) L2/L3，on-policy
distillation 的生产配方在 [nano-opd](../nano-opd/)。真机侧：真实 LoRA SFT/DPO
跑通、多 stage yaml 一键切换的实测，标 `[TODO: verify on real system]`（GPU
通道攒批，写作轮不 ssh）。

---

## 13. 溯源与口径声明

**LlamaFactory 快照**：固定 revision
[`f28afaf6355af515454dfb16c97d728307c93897`](https://github.com/hiyouga/LlamaFactory/tree/f28afaf6355af515454dfb16c97d728307c93897)，
抓取日 2026-08-16。上游仓库后来使用 `hiyouga/LlamaFactory` 名称，Python 包名仍为
`llamafactory`。本文行号只对该 revision 负责。

**TRL 对照**：[v0.24.0](https://github.com/huggingface/trl/tree/v0.24.0)。正文只用它解释
sigmoid/hinge/IPO/KTO 等 loss 分支的机制；行号同样是版本化快照事实。

**运行复验**：代码 md5 `19035b47e01490cd2e814c6858438668`/899 行/42,447 B。
2026-08-31 在两个新建空 CWD 以 `-B` 运行，均 EXIT=0、stderr 0 B；掩码输出
`56611328be39c8d75cb85cb74f14526a`/64 行/3,598 B，逐字节一致，digest
`6a9e099db7c7c7cb14e076ea4e063134`。这证明当前文件的教学重构没有改变数值路径。

**arXiv 五 ID**（2026-08-16 export.arxiv.org API live 核验，标题逐位吻合）：
DPO 2305.18290「Direct Preference Optimization: Your Language Model is Secretly
a Reward Model」/ IPO 2310.12036「A General Theoretical Paradigm to Understand
Learning from Human Preferences」/ KTO 2402.01306「KTO: Model Alignment as
Prospect Theoretic Optimization」/ ORPO 2403.07691「ORPO: Monolithic Preference
Optimization without Reference Model」/ SimPO 2405.14734「SimPO: Simple
Preference Optimization with a Reference-Free Reward」。

**数字口径**：本教程所有数字出自 §1 paste 块与当前双 fresh-CWD 运行；无任何外部 benchmark 数字；toy 常数（β=0.1 =
LF pref_beta 默认、simpo_gamma=0.5 = LF 默认）声明在 §5。

**真机项**：真实 LoRA SFT/DPO 跑通、多 stage yaml 切换、deepspeed/多卡下的
ref 前向成本不在本级证据范围内，必须在固定框架与模型 revision 后另行验证。
