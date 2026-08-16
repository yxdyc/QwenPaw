# nano-trinity-rft L3 — 配置即实验台：schema、注册表与 ablation ladder

> **K+1 位置**：L2 回答了「RL 的 reward 信号从哪里来」（rule vs learned RM、
> dead group 算术、Goodhart）。本级换一个完全不同的问题：**为什么 Trinity-RFT
> 能用一份 YAML 就切换十余种算法、跑一组消融？** 答案不在算法里，在配置系统
> 里——schema、注册表、三层优先级合并、stages 覆盖，这四样东西让「改一个字段
> = 换一个实验」成为可能。L3 就在 toy 上复现这套机制，并用它跑一个 DAPO 式的
> ablation ladder：阶梯每一格 = `dapo.yaml` 里的一行开关。
> **对标权威实现**：`agentscope-ai/Trinity-RFT`（main 分支，2026-08-13 现场克隆
> 核验，HEAD `009850b1`，末 commit 2026-07-31；README 30,381 B，sha256
> `d513f140…b73982`——与 L1 08-06 / L2 08-12 录值逐位零漂移）。行号锚点以
> 2026-08-13 抓取日为准。

---

## 1. 先跑起来

```bash
python3 -B L3_config_ablation.py     # 仅依赖 torch；CPU ~2 分钟；任意 CWD 可跑
```

固定 seed 下指标行逐字节确定（`elapsed` 计时行随机器负载浮动，掩码口径
`sed '/^[[:space:]]*elapsed/d'`）。完整输出（2026-08-13 本机实测，
掩码锚 md5 `7ee8aa0952b2acf68049e90b3535b7c5`/101 行，双 CWD 两遍逐字节一致）：

```text
============================================================================
nano-trinity-rft L3 — 配置即实验台：schema、注册表与 ablation ladder
============================================================================
env: python 3.13.13 | torch 2.13.0 | seed 20260806

[0] schema 与三层优先级：algorithm_type 是宏开关，展开成微观组合
    三层 = 用户显式设置 > algorithm_type 默认 > 全局兜底
    （config_validator.py:L385-398：check_config → update → set_if_none）
  grpo 全默认             → repeat_times=2(a) policy_loss_fn=ppo(a) advantage_fn=grpo(a) kl_loss_fn=k2(a) loss_agg_mode=token-mean(a)
  grpo 用户改 G=8         → repeat_times=8(u) policy_loss_fn=ppo(a) advantage_fn=grpo(a) kl_loss_fn=k2(a) loss_agg_mode=token-mean(a)
  dapo 全默认             → repeat_times=16(a) policy_loss_fn=ppo(a) advantage_fn=grpo(a) kl_loss_fn=none(a) loss_agg_mode=token-mean(a)
  dapo 用户改 clip        → repeat_times=16(a) policy_loss_fn=ppo(a) advantage_fn=grpo(a) kl_loss_fn=none(a) loss_agg_mode=token-mean(a)
  dpo 用户设 kl=none      → repeat_times=2(u) policy_loss_fn=dpo(a) advantage_fn=ppo(g) kl_loss_fn=k2(u) loss_agg_mode=token-mean(a)
      check_config 修复: kl_loss_fn: none → k2（DPO must use KL loss，algorithm.py:L302-304）
      check_config 修复: repeat_times → 2（Fake repeat times，algorithm.py:L300-301）
    (u)=user 显式 / (a)=algorithm 默认 / (g)=全局兜底
  sft+mode=both → ValueError: `algorithm_type: sft` does not support `mode: both`（镜像 SFT...
    nano 注册表 4 项 ['dapo', 'dpo', 'grpo', 'sft']；Trinity ALGORITHM_TYPE 24 项
    （algorithm/__init__.py:L9-36：sft/cpt/ppo/grpo/dapo/.../on_policy_distill/jsd）

[1] 起点：SFT@3r + warm RL（dense 逐位优势, G=32, 64 步）→ 快照 snap_warm
SFT@3r: exact=0.500 characc=0.667 per-ctx=[1.00 1.00 1.00 0.25 0.50 0.25]
warm RL@16r: exact=0.667 characc=0.917 p̂(M=768)=[0.32 0.35 0.30 0.00 0.24 0.00]

[2] ablation ladder（同一 snap_warm 出发，G=8，20 轮，ppo_epochs=3）
    开关表对照 examples/dapo_math/README.md:L11-16（paper technique → wiring）
R0 grpo 基线             exact=0.667 characc=0.875 H(r1→r20)=0.000→0.000 drift=47.5653 clipfrac=0.049 dead均=0.358 len=5.25 rollouts=960
R1 +kl_loss=none       exact=0.667 characc=0.917 H(r1→r20)=0.000→0.000 drift=220.7239 clipfrac=0.052 dead均=0.350 len=5.27 rollouts=960
R2 +clip_high=0.28     exact=0.667 characc=0.875 H(r1→r20)=0.000→0.000 drift=465.9230 clipfrac=0.049 dead均=0.367 len=5.40 rollouts=960
R3 +dyn_sampling       exact=0.667 characc=0.917 H(r1→r20)=0.000→0.000 drift=206.6399 clipfrac=0.078 dead均=0.383 len=5.19 rollouts=960
R4 +overlong           exact=0.667 characc=0.875 H(r1→r20)=0.000→0.001 drift=40.3948 clipfrac=0.075 dead均=0.150 len=5.06 rollouts=960
    R0→R1 去 KL（DAPO 默认）；R1→R2 Clip-Higher（非对称 clip，防熵坍缩）；
    R2→R3 Dynamic Sampling（std_threshold 过滤 + 活组复制补位，不花新 rollout）；
    R3→R4 Overlong 软惩罚（reward 加 format_score 分量，max=8/cache=2/factor=1）

[3] 批内机制演示（loss_agg_mode / overlong 曲线 / mask_response_truncated）
(a) loss_agg_mode 梯度质量份额（批=[短5tok, 长8tok]，adv=1、ratio=1）:
    token-mean:          批损失 -1.0000，长序列份额 0.615（∝ 长度，长响应主导梯度）
    seq-mean-token-sum:  批损失 -6.5000，长序列份额 0.615（同族，全局尺度不同）
    seq-mean-token-mean: 批损失 -1.0000，长序列份额 0.500（每序列 1/L 归一 → 等权）
    DAPO §3.3『token-level loss』= 不做 1/L 归一的 token 加和族 → Trinity dapo.yaml
    接 token-mean（dapo_math/README.md:L13 录此接线）；长度归一族会稀释长响应梯度
(b) overlong 软惩罚曲线（max=8, cache=2, factor=1；dapo_reward.py:L71-97 原式）:
    len:     [4, 5, 6, 7, 8, 9]
    penalty: ['0.00', '0.00', '0.00', '-0.50', '-1.00', '-1.00']
    len<max−cache=6 → 0；6→8 线性降到 −1；>max → −1（截断响应另被 mask）
(c) mask_response_truncated: 截断响应（8 token 无 EOS）mask 和 8 → 0（reward_filter.py:L158-172：action_mask 置 0，
    该样本对 policy loss 零贡献——DAPO §3.4 第一道闸；软惩罚是第二道闸

[4] stages：SFT→RL 课程 = 两条 StageConfig，循环代码零改动
  stage 'char-gpt-curriculum/sft_warmup'（mode=train, policy_loss_fn=sft(algorithm)） → exact=0.500 characc=0.667
  stage 'char-gpt-curriculum/rl'（mode=both, algorithm_type=grpo, G=8, kl=none）@12r → exact=0.500 characc=0.667 dead均=0.514 rollouts=576
    （RL 阶段曲线平 = 冷启 SFT 直接上稀疏 RL：空洞组全 dead、覆盖组已饱和，
    L1 §5 同课——这正是生产课程要 warm/dense 过渡层的原因；本演示的主张是
    stages 机制本身：同一 Config 迭代出两段、交接权重、循环代码零改动）
  名字后缀（Config.__iter__ L988-990）: char-gpt-curriculum/sft_warmup / char-gpt-curriculum/rl
  Trinity 在 stage 边界强制 save_hf_checkpoint='last'（config.py:L993-994），
  nano 对应物 = 内存 state_dict 交接；gsm8k.yaml 尾部注释即同款 stages 模板

[5] 账本与取舍
成本: 5 臂 × 20 轮 × G=8 × 6 ctx = 4800 rollouts/臂（R3/R4 的 dup 补位不花 rollout）；
      schema/resolve/stages 演示零训练成本——配置系统的『实验成本』是解析配置，
      这正是 ablation 便宜的原因：改 YAML 字段，不改代码、不重训基座。
取舍表（nano vs Trinity）:
  维度            nano-L3                    Trinity（源码锚点）
  schema          6 dataclass、10 算法字段    34 dataclass、1063 行（config.py）
  注册表          ALGORITHM_TYPE 4 项         24 项（algorithm/__init__.py:L9-36）
  三层合并        resolve_algorithm 同序      config_validator.py:L385-398
  PPO 损失        ratio+非对称clip+clip_c     ppo_policy_loss.py:L69-95（改自 verl）
  聚合            4 模式同语义                utils.py:L9-49
  advantage       grpo + std_threshold + dup  grpo_advantage.py:L160-194
  过滤位置        advantage 层（进训练前）     两处：advantage_fn_args 或管线算子
                                              dapo_dynamic_sampling（二选一，README:L18
                                              警告不可叠加；算子按 metrics['accuracy']
                                              判对错，不用塑形后总 reward——R4 的 nano
                                              std_threshold 恰是反例，见输出）
  调度/buffer     批内复用 ppo_epochs=3       queue buffer + Ray actor + 权重同步
                                              （trainer.py:L104-178，对照 nano-verl/slime）

self-check:
    PASS  三层合并: grpo 全默认时 repeat_times=2 来自 algorithm 层（非用户）
    PASS  三层合并: 用户 G=8 覆盖算法默认（prov=user）
    PASS  宏开关展开: dapo 默认 kl_loss_fn=none、G=16（algorithm.py:L162-174 口径）
    PASS  check_config 修复: dpo+kl=none 被强制改回 k2（algorithm.py:L302-304）
    PASS  check_config 拦截: sft+mode=both 抛 ValueError（algorithm.py:L67-76）
    PASS  SFT 天花板: 覆盖 ctx 全 1、空洞 0.25 附近（L1/L2 同构算术）
    PASS  warm 窗口: ≥1 个空洞 ctx 胜率 p̂ 落在 (0.005, 0.6)（稀疏信号存在性前提；最难空洞 p̂=0 = L2 §6 『最难的题需要更大 G 或课程』的同构）
    PASS  R1 去 KL 后漂移更大: drift(R1) ≥ drift(R0)（KL loss 的锚定作用）
    PASS  Clip-Higher 降低截断率: clipfrac(R2) ≤ clipfrac(R1)（上界 0.2→0.28）
    PASS  Clip-Higher 保熵: H_r12(R2) ≥ H_r12(R1)（DAPO 防熵坍缩的 toy 对应）
    PASS  dyn sampling 生效: R3 有活组复制补位（dup>0）且训练照常
    PASS  dyn sampling 不更差: characc(R3) ≥ characc(R2) − 0.1
    PASS  overlong 改变长度或 dead 结构: len(R4) < len(R3) 或 dead 均(R4) < dead 均(R3)
    PASS  学习有效: 最优臂末态 characc ≥ 起点 warm（RL 在填洞）
    PASS  token 加和族: 长序列梯度份额 = L长/(L短+L长) = 0.615（长度偏置存在）
    PASS  长度归一族: seq-mean-token-mean 长序列份额 = 0.5（每序列等权）
    PASS  overlong 曲线: len=6 → 0、len=8 → −1（Trinity 原式算术）
    PASS  mask_response_truncated: 截断响应 mask 置 0（零损失贡献）
    ✅ self-check passed (18/18)

digest(md5 of metrics) = 666a72423df8753e7012d82d439dcc8c
```

18 条 self-check 全绿。下面逐段拆。

---

## 2. 问题设定：L2 留下的那个「便宜」

L2 做了三组对比实验（G 的大小、reward 来源、KL 的有无）。回头看一下那些实验
是怎么「换配方」的：改一个参数、跑一遍、记一行。在 toy 里这很自然，因为循环
是我们自己写的。但 Trinity 生产线上有十余种算法（PPO/GRPO/DAPO/DPO/RAFT/
GiGPO/on-policy distill...）、四种聚合模式、七种 KL 估计、可插拔的 advantage /
policy loss / reward——如果每种组合都要改代码，消融研究根本做不动。

Trinity 的答案是：**所有算法决策都是配置字段，训练循环只消费解析后的配置。**
于是「GRPO 去掉 KL」「DAPO 把 clip 上界放到 0.28」这类消融，就是改 YAML 里
一行字段的事。本级要复现的就是这套机制，拆开看有四件：

1. **schema**——决策即字段（§3a）；
2. **注册表 + 宏开关**——`algorithm_type` 一个字段展开成一整套微观组合（§3b）；
3. **三层优先级合并 + check_config**——用户 > 算法默认 > 全局兜底，非法组合被
   拦截或修复（§3c、[0] 实验）；
4. **stages**——多阶段课程 = 基座配置 + 覆盖列表（§3d、[4] 实验）。

然后用这套机制跑一个 ablation ladder（[2] 实验）：每一格只动一个开关，对照
`examples/dapo_math/` 里 DAPO 的开关表（§3e）。

---

## 3. Trinity 的配置系统地图（权威源码对照，2026-08-13 现场核验）

**(a) schema：决策即字段。** `trinity/common/config.py`（1063 行、34 个
dataclass）把训练的一切决策都写成了具名字段。与算法消融直接相关的是
`AlgorithmConfig`（L618-662）：

```python
algorithm_type: str = "ppo"          # 宏开关
repeat_times: int = 1                # GRPO 组大小 G
advantage_fn / advantage_fn_args     # "grpo" + {std_threshold, duplicate_experiences, ...}
policy_loss_fn / policy_loss_fn_args # "ppo" + {clip_range_low/high, clip_ratio_c, loss_agg_mode}
kl_penalty_fn / kl_loss_fn           # 两个独立的 KL 旋钮：reward 侧 vs loss 侧
entropy_loss_fn / loss_agg_mode      # 熵正则 / 四种聚合模式
bypass_old_logprobs / rollout_correction  # 旧 logprob 复用 / off-policy 修正
```

注意 `kl_penalty_fn` 与 `kl_loss_fn` 是**两个**字段：前者把 KL 算进 reward
（PPO 经典形态），后者把 KL 加进 loss（GRPO 形态）。消融「KL 放哪里」不需要
碰任何代码。`ExplorerConfig`（L729-790）管 eval_interval / over_rollout /
concurrent_mode；`TrainerConfig`（L806-849）管 grad_clip / offload 开关 /
Megatron 并行度；`BufferConfig`（L708-727）管 batch_size 与两个输入口
（explorer_input 的 taskset / trainer_input 的 experience_buffer）。

**(b) 注册表 + 宏开关展开。** `ALGORITHM_TYPE` 注册表
（`trinity/algorithm/__init__.py:L9-36`）登记了 24 个 algorithm_type，每个是一
个 `AlgorithmType` 子类（`algorithm.py`），带两类信息：

- **类级标志**：`use_critic` / `use_reference` / `compute_advantage_in_trainer`
  / `schema`——决定这个算法需要什么资源、吃什么数据。GRPO `use_reference=True`
  而 DAPO `use_reference=False`（L142-175：DAPO 不用 reference 模型，这是它
  `kl_loss_fn: none` 默认的根因）；SFT/DPO 的 `schema` 分别是 `"sft"`/`"dpo"`
  而非 `"experience"`。
- **`default_config()`**：宏开关展开成的微观组合。GRPO（L120-140）=
  `{repeat_times: 2, advantage_fn: grpo, policy_loss_fn: ppo, kl_loss_fn: k2,
  entropy_loss_fn: default}`；DAPO（L162-174）= `{repeat_times: 16,
  kl_loss_fn: none, ...}`。**一个 algorithm_type 字段 = 一整行消融配方。**

**(c) 三层优先级合并 + check_config。** 解析发生在
`AlgorithmConfigValidator.validate`（`config_validator.py:L385-398`）：

```python
algorithm = ALGORITHM_TYPE.get(config.algorithm.algorithm_type)
algorithm.check_config(config)                 # ① 校验 + 修复（会改写 config！）
default_config = { ...全局兜底... }             # ② 全局默认（ppo/ppo/none/k2/default/token-mean）
default_config.update(algorithm.default_config())  # ③ 算法默认覆盖全局
for key, value in default_config.items():
    set_if_none(config.algorithm, key, value)  # ④ 只填用户没设的字段
```

优先级 = **用户显式设置 > algorithm_type 默认 > 全局兜底**。`check_config` 不是
被动校验，它会**动手改**：`DPOAlgorithm.check_config`（algorithm.py:L293-305）
强制 `kl_loss_fn = "k2"`（「DPO must use KL loss」）、强制 `repeat_times = 2`
（「Fake repeat times」）；`SFTAlgorithm.check_config`（L67-76）直接拒绝
`mode != "train"`（SFT 没有 rollout，`both`/`explore` 模式无意义）。**配置系统
是契约，不是便签**——非法组合在进训练循环之前就被拦截或修复。

**(d) stages：多阶段 = 基座 + 覆盖。** `Config.stages: List[StageConfig]`
（config.py:L971）+ `Config.__iter__`（L978-995）：对每个 stage，deepcopy 基座
配置、把 stage 里非 None 的字段覆盖上去、名字加 `/stage_name` 后缀，然后强制
`trainer.save_hf_checkpoint = "last"`（L993-994）——保证下一阶段能从 HF
checkpoint 加载。SFT→RL 课程就是两条 stage 记录；`examples/grpo_gsm8k/
gsm8k.yaml` 尾部就有一段注释掉的 stages 模板（sft_warmup → rft）。

**(e) DAPO 开关表：消融阶梯的权威样板。** `examples/dapo_math/README.md:L11-16`
把 DAPO 论文（arXiv:2503.14476）的四个 technique 逐行映射到 `dapo.yaml` 的
字段：

| paper technique | Trinity wiring（dapo.yaml） |
|-----------------|------------------------------|
| GRPO 组相对优势、无 KL | `algorithm_type: dapo` → `advantage_fn: grpo`、`kl_loss_fn: none` |
| Clip-Higher | `policy_loss_fn_args.clip_range_low/high: 0.2 / 0.28` |
| Token-level policy loss | `policy_loss_fn_args.loss_agg_mode: token-mean` |
| Dynamic sampling | 管线算子 `dapo_dynamic_sampling` |
| Overlong filter（截断不计损失） | 管线算子 `mask_response_truncated` |
| Soft overlong reward | `math_dapo_reward` + `reward_fn_args` |

这张表就是 [2] 实验的设计图。两处值得停下来：其一，`dapo.yaml` 的
`loss_agg_mode` 配的是 `token-mean`——DAPO §3.3 说的「token-level loss」在
Trinity 里接的是 token-mean（[3a] 会解释为什么这合理）；其二，README:L18
明确警告**不要**同时设 `advantage_fn_args.std_threshold` 和管线过滤——同一个
机制有两个接线位置，叠加会双重过滤。

**源码考古发现（如实记录）**：`algorithm.py:L146-148` 的 DAPOAlgorithm
docstring 与 `examples/dapo_math/README.md:L3` 都引用
`docs/dapo_trinity_implementation_spec.md`，该文件在 2026-08-13 抓取的 main 树
（HEAD `009850b1`）中不存在——dangling reference（可能已移动或删除，未深追）。

nano-L3 的取舍：schema 压成 6 个 dataclass、注册表压成 4 项（sft/grpo/dapo/
dpo）、三层合并与 check_config **同序复现**（`resolve_algorithm`）、stages 的
deepcopy-覆盖-后缀同构；PPO 损失按 `ppo_policy_loss.py:L69-95` 实现（ratio
截断 ±20、非对称 clip、clip_ratio_c 只作用 adv<0、四模式聚合按
`utils.py:L9-49`）；GRPO advantage 按 `grpo_advantage.py:L160-194`（除
std+ε、std_threshold 过滤、duplicate_experiences 复制补位）；KL 用 k2
（`kl_fn.py` K2Fn = 0.5·(logp−ref_logp)²）。调度/buffer 不重复（对照
nano-verl / nano-slime）。

---

## 4. 实验 [0]：三层 resolve 与 check_config——配置为什么长这样

[0] 不训练，只解析。五个配置逐个过 `resolve_algorithm`，每个字段标注出处
（`(u)`=user / `(a)`=algorithm 默认 / `(g)`=全局兜底）：

```text
  grpo 全默认             → repeat_times=2(a) policy_loss_fn=ppo(a) advantage_fn=grpo(a) kl_loss_fn=k2(a) loss_agg_mode=token-mean(a)
  grpo 用户改 G=8         → repeat_times=8(u) ...（其余同上）
  dapo 全默认             → repeat_times=16(a) ... kl_loss_fn=none(a) ...
  dpo 用户设 kl=none      → repeat_times=2(u) policy_loss_fn=dpo(a) advantage_fn=ppo(g) kl_loss_fn=k2(u) ...
      check_config 修复: kl_loss_fn: none → k2（DPO must use KL loss，algorithm.py:L302-304）
      check_config 修复: repeat_times → 2（Fake repeat times，algorithm.py:L300-301）
  sft+mode=both → ValueError: `algorithm_type: sft` does not support `mode: both`
```

四个观察：

1. **grpo 全默认时没有一个字段来自用户**——`algorithm_type: grpo` 一行就展开
   出完整配方。这是「宏开关」的字面意思。
2. **dpo 行里 `kl_loss_fn=k2(u)` 的 `(u)` 是个陷阱**：用户明明设了 `none`，
   为什么出处显示 user/k2？因为 `check_config` 直接**改写了 config 对象**
   （algorithm.py:L302-304 原地赋值），改写之后再做三层合并，读到的就是修复后
   的值。Trinity 同语义——validator 跑完，你手里的 config 已经不是原来那份。
   修复日志（`fixed` 列表）是 nano 侧加的显式账本，生产实现里只有 warning 日志。
3. **`advantage_fn=ppo(g)` 出现在 dpo 行**：dpo 的 `default_config` 没有
   advantage_fn 这一项（DPO 不需要），落到全局兜底 `ppo`——一个无害的残留字段。
   真实系统里这类「兜底残留」不少，读配置解析结果时要分清哪些字段真的被消费。
4. **sft+mode=both 被拒绝**：SFT 没有 rollout 环节，`both` 模式在语义上就不
   成立。错误在**解析期**抛出，不是跑了三小时才在 trainer 里崩——这是 schema
   作为契约的直接回报。

---

## 5. 实验 [1]：起点——SFT + warm，L2 同构的窗口

任务表与 L1/L2 同源（SEED=20260806，targets 逐位一致）。本级唯一扩展：
**响应变长**——采样到 EOS 或 8 token 为止（L1/L2 固定 4 字符）。为什么必须
变长：`loss_agg_mode` 与 overlong 惩罚这两个开关在定长响应上**没有任何可观察
差异**（所有序列等长，聚合权重与长度 shaping 都退化为常数）。变长是它们生效的
最小前提。SFT 目标 = 4 个目标字符 + EOS——「答完就停」本来就是 SFT 目标的一
部分（L1/L2 的 SFT 损失本就含 EOS 位），口径未变。

```text
SFT@3r: exact=0.500 characc=0.667 per-ctx=[1.00 1.00 1.00 0.25 0.50 0.25]
warm RL@16r: exact=0.667 characc=0.917 p̂(M=768)=[0.32 0.35 0.30 0.00 0.24 0.00]
```

与 L2 的 warm 态同构（L2：exact 0.667 / characc 0.917 / p̂ c3 0.338、c5
0.033）：覆盖 ctx 全 1，空洞 c4 进入「偶尔会对」窗口（p̂=0.24），最难的
c3/c5 仍为 0——L2 §6 的教训原样：最难的题需要更大的 G 或课程。warm 用逐位
dense 优势（L1/L2 的 credit assignment，nano 侧继承；Trinity 的 advantage_fn
只消费标量 reward，逐位 dense 是 toy 压缩）+ KL k2 锚（防 GRPO 除 std 在 toy
尺度放大噪声）。

一个 toy 尺度的发现，后面会反复用到：**SFT 把覆盖 ctx 的目标路径饱和到
p≈1.0**——logp 恒为 0、梯度归零、沿该路径的任何漂移探针恒读 0。所以本级的
drift 探针放在空洞 ctx（3–5）的目标路径上；熵探针 H 也基本恒 0（§7 再谈它
的叙事后果）。

---

## 6. 实验 [2]：ablation ladder——每一格 = dapo.yaml 的一行开关

五个臂从同一个 `snap_warm` 出发，G=8、20 轮、ppo_epochs=3（批内复用来让
ratio/clip 存在），**训练循环对所有臂一字不改，变的只有 config 字段**：

```text
臂      相对上一格多动的字段                     末态关键指标
R0      grpo 基线（clip 0.2/0.2，kl k2）         drift=47.6  clipfrac=0.049  dead均=0.358
R1      kl_loss_fn: k2 → none                   drift=220.7（4.6×）
R2      clip_range_high: 0.2 → 0.28             clipfrac=0.049（≤R1 0.052）drift=465.9
R3      + std_threshold=1e-6 + duplicate        characc=0.917（并列最高）dup 补位生效
R4      + overlong 软惩罚（max=8/cache=2）       dead均 0.383 → 0.150（打破 dead 平局）
```

逐格拆机制：

**R0→R1（去 KL）**：drift 从 47.6 涨到 220.7——同一个循环、同一份数据，仅
`kl_loss_fn` 从 k2 变 none，策略离开 warm 快照的距离就放大 4.6 倍。这就是
KL loss 的锚定作用被「配置化」后的样子：DAPO 默认 `kl_loss_fn: none`
（algorithm.py:L162-174，`use_reference=False` 的推论——连 reference 模型都
不加载），是在**接受更大漂移**换探索自由度；GRPO 默认 k2 是相反取舍。toy 里
两臂 characc 都健康（0.875/0.917），因为 20 轮太短、漂移还没酿成遗忘——真实
尺度上这个差距会以「覆盖域退化」的形式兑现（L2 §8 的 KL 臂对照同构）。

**R1→R2（Clip-Higher）**：非对称 clip 把上界从 1.2 放到 1.28。可观察量是
clipfrac（被 clip 的 token 占比，Trinity 的 `pg_clipfrac` 指标，
ppo_policy_loss.py:L69（pg_clip_frac 计算）/L109（metrics 导出））：0.052→0.049，上界放宽后正向更新的截断变少。drift
反而升到 465.9——约束更松、单步走得更远，与 Clip-Higher 的设计意图一致
（DAPO §3.1：放宽上界让高 advantage token 的概率能涨上去，维持策略更新能力）。
**诚实声明**：DAPO 论文对 Clip-Higher 的核心论据是「防熵坍缩」
（dapo_math/README.md:L38「Policy entropy ... should not collapse early with
Clip-Higher」），但本 toy 的策略熵从 SFT 起就≈0（§5 的饱和发现），熵保留叙事
在本尺度**不可观察**——clip 机制只能由 clipfrac 与 drift 度量。这是 toy 的
边界，不是开关的边界。

**R2→R3（Dynamic Sampling）**：`advantage_fn_args.std_threshold=1e-6` +
`duplicate_experiences=True`（grpo_advantage.py:L160-163 过滤 + L178-194
补位）。dead均（观测口径，组内 reward 全同的组占比）~0.38——这些组在 R0–R2
里以零优势参与 mean、稀释梯度（L2 §6 的账），在 R3 里被整组丢弃、缺的组数从
活组随机复制补回（**不花新 rollout**——对照 L2 的 dyn 是补采新 rollout，Trinity
的 `duplicate_experiences` 是复制已有活组，代价是重复样本）。characc 0.917
并列最高。注意 nano 侧过滤发生在 advantage 层（进训练前）；Trinity 还有第二个
接线位置——管线算子 `dapo_dynamic_sampling`（reward_filter.py:L65-148），两者
**二选一**（README:L18 警告）。

**R3→R4（Overlong 软惩罚）**：reward 字典加 `format_score` 分量
（dapo_reward.py:L71-97 原式，toy 参数 max=8/cache=2/factor=1），最意外的读数
是 dead均 0.383→0.150：**长度塑形打破了全错组的 reward 平局**——原本全 0 的
组里，长响应拿到 −0.5/−1.0，组内 std 不再是 0，dead 组变 live。这既是塑形
的红利（多了一路信号），也是一个**泄漏警示**：R4 的 std_threshold 过滤现在
筛的是「塑形后总 reward」的方差，不是「对错」的方差——组活过来是因为长度，
不是因为答案。**这正是 Trinity 把 DAPO 过滤做成管线算子、并明确按
`metrics["accuracy"]` 而非总 reward 判定（reward_filter.py:L70-71、L100-104）
的原因**；README:L18 的「用 accuracy 不用 length-shaped total reward」与
「不要叠加两种过滤」两条警告，在 R4 的输出里有了 toy 尺度的实证。

---

## 7. 实验 [3]：批内机制演示——三个在学习尺度不生效的开关

有三个开关在 toy 的 20 轮学习里看不出结局差异（定长主导的任务 + 饱和策略），
但它们的机制可以在**单个批内**直接测量。

**(a) loss_agg_mode 的两族。** 批 = [短 5 token, 长 8 token] 两条样本、
adv=1、ratio=1（纯聚合权重测量）：

```text
token-mean:          批损失 -1.0000，长序列份额 0.615（∝ 长度，长响应主导梯度）
seq-mean-token-sum:  批损失 -6.5000，长序列份额 0.615（同族，全局尺度不同）
seq-mean-token-mean: 批损失 -1.0000，长序列份额 0.500（每序列 1/L 归一 → 等权）
```

关键在**族**的划分：token-mean 与 seq-mean-token-sum 的**相对**权重完全相同
（长序列份额都是 8/13=0.615），只差一个全局尺度（1/总token 数 vs 1/批大小，
即有效学习率不同）——它们同属「token 加和族」：长响应拿到的梯度质量 ∝ 长度。
seq-mean-token-mean 是「长度归一族」：每条序列先除以自身长度，长短等权。
DAPO §3.3 的「token-level loss」= 不做 1/L 归一的 token 加和族——所以
Trinity 的 dapo.yaml 接 `token-mean`（dapo_math/README.md:L13 录此接线）
不是随意之选：在相对权重意义上，token-mean 就是 token-level。而想要「每条
响应等权」，要用 seq-mean-token-sum-norm（utils.py:L40-43，除以固定
normalizer）或 seq-mean-token-mean。

**(b) overlong 软惩罚曲线。** Trinity 原式（dapo_reward.py:L71-97）在 toy
参数下的形状：`len < max−cache = 6` → 0；6→8 之间线性降到 −1；超过 max →
−1。注意惩罚窗口挂在**尾部**（`max_response_length − cache_length` 起算），
不是从 cache_length 起算——直觉上容易读反。

**(c) mask_response_truncated。** 截断响应（8 token 无 EOS）的 action_mask
整体置 0（reward_filter.py:L158-172），对 policy loss 零贡献。这是 DAPO §3.4
的**第一道闸**（截断样本干脆不进梯度），软惩罚是第二道闸（接近截断的样本拿
负分）。两道闸的分工：前者管「已经截断的」，后者管「快要截断的」。

---

## 8. 实验 [4]：stages——课程是配置列表，不是代码分支

```text
stage 'char-gpt-curriculum/sft_warmup'（mode=train, policy_loss_fn=sft(algorithm)） → exact=0.500 characc=0.667
stage 'char-gpt-curriculum/rl'（mode=both, algorithm_type=grpo, G=8, kl=none）@12r → exact=0.500 characc=0.667 dead均=0.514
名字后缀: char-gpt-curriculum/sft_warmup / char-gpt-curriculum/rl
```

机制主张全部兑现：同一个 `Config` 经 `__iter__` 迭代出两段（deepcopy 基座 →
覆盖非 None 字段 → 名字加后缀，config.py:L978-995 同构），stage 边界做权重
交接（Trinity 强制 `save_hf_checkpoint="last"`，nano 对应物 = 内存
state_dict），**训练循环代码没有任何 `if stage == ...` 分支**。

而学习曲线是平的（0.667→0.667，dead均 0.514）——这不是 stages 的失败，是
L1 §5 的老课：**冷启 SFT 直接上稀疏 RL 没有信号**（空洞组全 dead、覆盖组已
饱和）。生产课程需要 warm/dense 过渡层（本文件 [2] 的 snap_warm 就是），
stages 机制的价值恰在于：加一层过渡 = 列表里多一条 StageConfig，循环依旧
零改动。`examples/grpo_gsm8k/gsm8k.yaml` 尾部注释的 sft_warmup→rft 模板就是
这个用法。

---

## 9. 机制深潜：三条带走的话

1. **algorithm_type 是宏开关，消融是宏开关的枚举。** 一个字段展开成
   policy_loss × advantage × kl × entropy × G 的整行组合（algorithm.py 的
   `default_config`），三层合并保证用户只写差异项（config_validator.py:
   L385-398），check_config 保证组合合法（DPO 强制 KL、SFT 拒绝 both）。
   「跑一组消融」因此 = 枚举几个字段值，训练循环一字不动——DAPO 开关表
   （dapo_math/README.md:L11-18）就是这种枚举的样板。
2. **同一个机制可以有多个接线位置，选错位置就是 bug。** dynamic sampling
   可以接在 advantage_fn_args（std_threshold）或管线算子
   （dapo_dynamic_sampling），README:L18 警告二选一；R4 实测了接错的效果——
   塑形后的总 reward 让过滤按「长度方差」而非「对错方差」筛组（dead均
   0.383→0.150 是泄漏的信号）。Trinity 把 DAPO 过滤做成按 `metrics["accuracy"]`
   判定的管线算子，就是对这类泄漏的工程防御。读权威实现时，**找到机制的所有
   接线位置、以及文档警告你不要叠加的那些**，和找到机制本身一样重要。
3. **配置的便宜是解析期的便宜。** [0]/stages 演示零训练成本；ladder 五臂的
   差异全部在 resolve 阶段就定型（provenance 表可查）。ablation 研究的可信度
   有一半来自这里：每个臂「到底改了什么、其他字段从哪层来」是可审计的——
   而不是散落在代码 diff 里。

---

## 10. 费曼自检

**讲给外行听**：想象一个巨大的调音台，每首曲子（算法）其实是一组推杆位置
（clip 多少、要不要 KL、组多大…）。Trinity 的做法是给每首曲子存一张**预设卡**
（algorithm_type），卡上只写和默认不同的推杆；调音台还有安全锁（check_config）
——你把「人声」推杆拔了它会给你装回去（DPO 必须有 KL），你说「放伴奏但别放
曲子」（SFT+both 模式）它直接拒绝播放。做消融实验就是把预设卡一张张换着放、
比较声音——不用每次重新接线。

三个自问：

- 能不能说清「用户设了 kl=none 的 DPO 配置，resolve 之后 kl_loss_fn 为什么
  是 k2、出处还标 user」？（check_config 原地改写 config 在先，三层合并在后。）
- 能不能说清 token-mean 和 seq-mean-token-sum 为什么是「同族」？DAPO 的
  token-level loss 接 token-mean 凭什么合理？（相对权重都 ∝ 序列长度，只差
  全局尺度；token-level 的要义是不做 1/L 归一。）
- R4 的 dead均掉到 0.150 是好事还是坏事？（既是塑形多给了一路信号，也是过滤
  开始按长度而非对错筛组——所以 Trinity 的 DAPO 过滤按 accuracy 判定。）

---

## 11. 思考题

1. Trinity 的 `kl_penalty_fn`（reward 侧）和 `kl_loss_fn`（loss 侧）是两个
   字段。设计一个消融：固定其他字段，枚举 {penalty 开/关} × {loss 开/关}，
   预测四格里哪两格行为接近？（提示：两者都是对策略漂移的惩罚，但作用路径
   不同——一个进 reward 参与 advantage 归一，一个直接进 loss。）
2. `check_config` 会改写用户配置（DPO 强制 k2）。从「配置是契约」的角度，
   改写 + warning 和直接报错拒绝，各适合什么样的非法组合？为什么 SFT+both
   是拒绝、DPO+kl=none 是修复？
3. R4 展示了塑形泄漏进过滤。如果你要在 nano 里实现 Trinity 的管线算子版
   DAPO 过滤（按 accuracy 判定），需要给 Sample 记录加什么字段？（提示：
   `metrics` 分账——reward_dict 各组件分开记录，math_rm_workflow.py:L36-44，
   L2 §3(b) 已见过。）
4. stages 之间 Trinity 强制 `save_hf_checkpoint="last"`（config.py:L993-994）。
   为什么是 HF 格式而不是原生 checkpoint？（提示：下一阶段可能换训练后端/
   并行策略——HF 格式是跨后端的最小公分母。）
5. 本 toy 的 SFT 把覆盖 ctx 饱和到 p≈1.0（logp=0、梯度归零）。真实 LLM 的
   SFT 为什么通常不会饱和到这个程度？（提示：词表大小、label smoothing、
   数据多样性、dropout——以及为什么这反而让 RL 的梯度有地方可去。）
6. [2] 的 ladder 每臂 4800 rollouts、结论以 drift/clipfrac/dead均 呈现。
   如果预算砍到每臂 960 rollouts（5 轮），哪些结论还站得住、哪些会先垮？
   （提示：drift 是累积量、clipfrac 是批内瞬时量、characc 是离散探针。）

---

## 12. 反例与边界

1. **toy 的熵叙事不可观察。** DAPO Clip-Higher 的核心论据是防熵坍缩，但本
   toy 的策略熵从 SFT 起就≈0（饱和），H 探针全程 0.000——「保熵」在 toy 里
   没有可区分的对象。clip 机制改由 clipfrac/drift 度量，这是退而求其次，
   不是等价替代。
2. **drift 的绝对值不可外推。** 47.6 / 220.7 / 465.9 是 k2 式
   0.5·(Δlogp)² 在 toy 词表（14 token）上的值，量纲和尺度都随词表、模型、
   轮数变。可迁移的是**相对关系**（去 KL → drift 放大 4.6×）和探针设计
   （避开饱和路径）。
3. **ε-探索掩盖了策略内部的探索差异。** toy 的采样多样性主要来自外注入的
   ε（L1 的教训），不是策略分布本身。因此任何依赖「策略自发探索」的机制
   （熵正则、温度调度）在本尺度都测不出差异——L3 没有把 entropy_loss_fn
   排进 ladder，就是这个原因。
4. **ladder 的 characc 区分度低（0.875/0.917 交替）。** 20 轮 toy 训练里，
   五臂的结局差异主要体现在 drift/clipfrac/dead均 这些过程量上，而不是末态
   准确率——真实消融同样常看过程量（熵曲线、clip 曲线、pass@k 中途值），
   末态指标往往要规模上去才分开。
5. **stages 演示的平曲线是设计内的反例。** 它复现的是「冷启稀疏 RL 无信号」
   （L1 §5），不是 stages 机制的缺陷；机制主张（迭代/覆盖/后缀/交接）由同段
   输出的其余部分兑现。
6. **nano 的 duplicate_experiences 用 deepcopy 复制样本（含 old_logp）**，
   与 Trinity 同语义（grpo_advantage.py:L178-194 也是 deepcopy）；但 toy 的
   ppo_epochs 批内复用里，复制样本与源样本共享同一份 old_logp——off-policy
   程度随 epoch 数加深，真实系统用 buffer 周转与权重同步缓解（对照
   nano-slime 的 staleness 讨论）。

---

## 13. 阶梯预告与交叉引用

- nano-trinity-rft 的阶梯到此完成（L0 统一数据流 → L1 真实模型上的配置驱动 →
  L2 reward 信号来源 → L3 配置系统与消融阶梯）。回看一条线：L0 说「改 config
  字段就能换配方」，L3 回答了「这为什么在生产尺度也成立」——schema + 注册表 +
  三层合并 + stages。
- 交叉阅读：[nano-verl](../nano-verl/)（actor/learner 调度——Trinity
  TrainerConfig 背后的那类问题）、[nano-slime](../nano-slime/)（buffer 解耦与
  staleness——duplicate_experiences 的重复样本在异步下的形态）、
  [nano-megatron](../../02-pretraining-cpt/nano-megatron/)（TrainerConfig 里
  的 Megatron 并行度字段指向的世界）、[L2 tutorial](tutorial_L2.md)
  （reward 来源与 dead group 算术，本级的 warm 窗口直接继承它）。

---

## 14. 溯源与校准

**权威实现锚点**（agentscope-ai/Trinity-RFT main 分支，2026-08-13 现场克隆，
HEAD `009850b112593133d042d6ca02384d71ae5fa988`，末 commit 2026-07-31；
README 30,381 B，sha256
`d513f140afdd691a0847f668ab5bd3cc062f99682cd9a2f5bba49e1cacb73982`——与
L1 08-06 / L2 08-12 录值逐位零漂移；行号以 2026-08-13 抓取日为准）：

| 锚点 | 内容 |
|------|------|
| trinity/common/config.py:L618-662 / L708-727 / L729-790 / L806-849 / L927-938 / L940-1052 | AlgorithmConfig / BufferConfig / ExplorerConfig / TrainerConfig / StageConfig / Config（含 stages L971、__iter__ L978-995、save_hf_checkpoint 强制 L993-994） |
| trinity/common/config_validator.py:L385-398 | AlgorithmConfigValidator：check_config → 全局兜底 → 算法默认 → set_if_none 三层合并 |
| trinity/algorithm/__init__.py:L9-36 | ALGORITHM_TYPE 注册表 24 项 |
| trinity/algorithm/algorithm.py:L48-118 / L120-140 / L142-175 / L263-305 | SFT / GRPO / DAPO / DPO 的 default_config + 类级标志 + check_config（DPO 强制 k2 L302-304、repeat_times L300-301；SFT 拒绝 both L67-76） |
| trinity/algorithm/policy_loss_fn/ppo_policy_loss.py:L19-36, L69-95 | 非对称 clip 参数 / ratio 截断 ±20、clip(1−low,1+high)、clip_ratio_c（adv<0）、pg_clipfrac（改自 verl core_algos.py） |
| trinity/algorithm/utils.py:L9-49 | aggregate_loss 四模式（token-mean / seq-mean-token-sum / seq-mean-token-mean / seq-mean-token-sum-norm） |
| trinity/algorithm/advantage_fn/grpo_advantage.py:L97, L106-116, L160-163, L166-169, L178-194 | std_threshold / duplicate_experiences 参数与 docstring；过滤实现；(r−mean)/(std+ε)；复制补位实现（L2 锚点复验零漂移） |
| trinity/algorithm/kl_fn/kl_fn.py（K2Fn）/ kl_fn/__init__.py:L4-15 | k2 = 0.5·(logp−ref_logp)²；KL_FN 注册表 7 项 |
| trinity/common/rewards/dapo_reward.py:L34-97 | MathDAPORewardFn：accuracy ±1 + format_score；overlong 软惩罚原式（max−cache 起算） |
| trinity/buffer/operators/__init__.py:L7-19 / operators/filters/reward_filter.py:L65-148, L151-172 | EXPERIENCE_OPERATORS 注册表 8 项（含 data_juicer）；DAPODynamicSamplingFilter（按 metrics["accuracy"] 判对错、dropped_all_correct/wrong 分账）；MaskResponseTruncatedOperator |
| examples/dapo_math/README.md:L11-16, L18, L38 / dapo.yaml | paper technique → Trinity wiring 六行表；accuracy 判定与「勿叠加过滤」警告；熵观察指标；clip 0.2/0.28、kl none、token-mean 实例 |
| examples/grpo_gsm8k/gsm8k.yaml（尾部注释） | sft_warmup → rft 的 stages 模板 |
| trinity/explorer/explorer.py:L404-405 / trinity/trainer/trainer.py:L104-178 | need_eval（step % eval_interval）/ Trainer.train 主循环（sample→train_step→need_sync→sync_weight→need_save→save_checkpoint） |

**论文锚点**（export.arxiv.org API 2026-08-13 现场重抓标题核验）：

| arXiv ID | 标题（核验录值） | 本节用法 |
|----------|------------------|----------|
| 2505.17826 | Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models | 对标框架 |
| 2503.14476 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale（2026-08-13 export.arxiv.org API 官方全称；L2 08-12 录值为缩写口径『LLM RL System』；v1/v2 abs 页标题 2026-08-13 现场核验逐位同一，无版本间演进） | §3.1 Clip-Higher / §3.2 Dynamic Sampling / §3.3 Token-Level Loss / §3.4 Overlong Reward Shaping |
| 2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | GRPO 组内相对优势 |
| 1707.06347 | Proximal Policy Optimization Algorithms（A 层经典锚点，L2 已验） | ratio clip 的出处；当今定位 = GRPO/DAPO 的 policy_loss_fn 仍是它的变体 |

**信息分类**：表格与引文 = 原文声称（现场克隆/抓取）；「token-mean 与
seq-mean-token-sum 相对权重同族」= 本节实测 + 初等代数（utils.py 语义的直接
推论）；「overlong 打破 dead 平局」「去 KL → drift 放大 4.6×」= 本节实测
（toy 尺度，边界见 §12）；「SFT 饱和到 p≈1.0」= 本节实测的 toy 现象（真实
LLM 通常不至此，§12.1/思考题 5）；dangling reference（dapo_trinity_
implementation_spec.md）= 现场核验的事实记录。

**复现锚点**：`python3 -B L3_config_ablation.py`，任意 CWD，CPU ~2 分钟；
掩码 `elapsed` 行后输出 md5 `7ee8aa0952b2acf68049e90b3535b7c5`/101 行
（2026-08-13 两独立 CWD 逐字节一致）；脚本自产 digest
`666a72423df8753e7012d82d439dcc8c`（metrics 的 md5）。
