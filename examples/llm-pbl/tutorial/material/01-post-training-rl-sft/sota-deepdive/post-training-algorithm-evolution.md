# SOTA Deep-Dive — 后训练算法演进：PPO → GRPO/RLVR → OPD

> **深挖对象**：2023–2026 后训练算法的机制演进主线——PPO 奠基（importance sampling + clipping）→ GRPO 族与 RLVR（去掉 value model、reward 换成可验证信号）→ OPD（on-policy distillation，蒸馏与 RL 的合流）。
> **状态**：✅ 首版完成（SOTA 对齐日期 2026-08-11）
> **可运行对照**：[nano-verl L0–L3](../nano-verl/) + [nano-opd L0–L1](../nano-opd/) + [nano-llamafactory](../nano-llamafactory/)；文中的「nano 实测」均指向这些模块的可运行产出。
> **SOTA 对齐日期**：2026-08-11（课程的证据时效性分层，B 层主题必做）。18 个 arXiv ID 当日经 export.arxiv.org API 现场核验（标题/日期/作者/摘要），两篇一手博客当日现场重抓，见 §10 溯源表。对齐结论：**未发现取代 GRPO 族 / RLVR / OPD 的新一代范式**。

---

## §0 这篇文章是什么

后训练（post-training）指预训练之后、把 base model 变成可用模型的全部训练阶段：SFT、偏好对齐、RL、蒸馏。这篇 deepdive 不复述论文摘要，而是回答三个问题：

1. **为什么演进走这条路**——不是「新方法取代旧方法」的线性故事，而是三个坐标轴的组合演化：样本从谁那里采（on/off-policy）、reward 信号有多密（dense/sparse）、reward 从谁来（人类偏好 / 可验证答案 / 教师模型）（§1）；
2. **六个机制面**——PPO 奠基（§2）、GRPO 与 RLVR（§3）、GRPO 族的修正机制类别（§4）、DPO 的离线支线定位（§5）、OPD 合流（§6）、算法与 infra 的共演化（§7），每一面都给一手来源 + nano 模块的可运行实证；
3. **2026 年的格局**——三层锚点定位（§8）。

**复验记录（反幻觉口径）**：本文引用的 18 个 arXiv ID 于 2026-08-11 经 export.arxiv.org API 批量现场核验（标题/日期/作者/摘要以当次抓取为准）；GSPO 官方博客与 Thinking Machines OPD 博客同日现场重抓全文。nano-verl / nano-opd 的实测数字逐字引自相关教程材料（文内标注文件与节号）。四类信息（原文声称 / 文献已有 / 合理推断 / 猜测）在正文中显式区分，推断标「推断」；无猜测级内容入正文。

**阅读路径建议**：先跑 `nano-verl/L1_minimal_ppo.py`（最小 PPO 闭环，torch 单卡/CPU 可跑），再跑 `nano-opd/L0_opd_divergence_choice.py`（散度选择的算术定理，零依赖确定性 toy），然后按 §2 → §6 读机制面；§7 需要 nano-verl L2/L3 的 colocate 实测背景。

---

## §1 演进主线：三个坐标轴，一条合流线

Thinking Machines 的 OPD 一手文（2025-10-27，Kevin Lu）给出过一张极简分类表（原文声称，下表为逐字整理）：

| 方法 | 采样 | reward 信号 |
|------|------|-------------|
| Supervised finetuning | off-policy | dense |
| Reinforcement learning | on-policy | sparse |
| On-policy distillation | on-policy | dense |

这张表就是本文的坐标系：**采样轴**（训练用的样本是不是当前学生/策略自己生成的）与 **信号轴**（每个 token 都有监督，还是只有轨迹终局一个分数）。第三轴是 **reward 的来源**：人类偏好标注（RLHF 经典形态）、可验证答案（RLVR：数学题对错、代码测试通过与否）、教师模型打分（OPD）。

三代演进在这三个轴上的走位（时间线全部为 §10 核验日期）：

- **PPO（2017-07）**：on-policy + sparse + 人类偏好（经 reward model）。奠定 importance sampling ratio + clipping 的更新控制机制——这是后面一切的地基（§2）。
- **DPO（2023-05）**：把偏好对齐改写成离线闭式解，绕开 RL 循环——离线支线（§5）。
- **GRPO（2024-02，DeepSeekMath）**：PPO 的变体，去掉 value model，用组内相对 reward 估计 advantage；原始动机之一是 PPO 的内存开销（摘要原文，§3）。
- **DeepSeek-R1（2025-01）**：RLVR 分水岭——纯 RL、不依赖人工标注推理轨迹，reward 来自可验证任务（§3）。
- **GRPO 族修正（2025-03–07）**：DAPO / Dr. GRPO / CISPO / GSPO 各自修复 token 级 importance sampling 的一个具体缺陷（§4）。
- **OPD（2023 锚点 → 2025–26 生产化）**：reverse KL + 学生自采样的经典机制（MiniLLM/GKD/DistiLLM）在 Qwen3、Thinking Machines、MiMo-V2-Flash 等生产配方中成为核心成分（§6）。

合流线（推断，与上述全部一手来源一致）：**采样轴一路向 on-policy 收敛**（SFT→RL→OPD 都要求用当前策略自己的样本），**信号轴一路向 dense 收敛**（终局 reward→可验证中间奖励→教师逐 token 打分），而 reward 来源的演化（人类偏好→可验证答案→教师）决定了每一代方法能教什么能力。PPO 的 IS+clipping 机制则一路随行——GRPO 族与 OPD 的实现都在用它（§2 末与 §6 的 TM 伪代码是直接证据）。

---

## §2 PPO：importance sampling + clipping 的地基（A 层锚点）

### 2.1 机制：用旧数据多次更新，同时控制更新幅度

PPO [1707.06347]（2017-07-20，Schulman et al.）的核心问题：on-policy 采样昂贵，一批样本想用多次，但策略一旦偏离采样分布，梯度估计就失真。解法两件：

- **importance sampling ratio**：`ratio = π_new(a|s) / π_old(a|s)`。在固定的同一状态 $s$ 且旧策略覆盖新策略支持集时，旧策略动作样本乘 ratio 可以改写新策略下的条件期望——这是「同批数据更新多轮」的算术基础。它不自动修正旧策略产生的 prefix/state 分布；PPO clipping 还会主动引入偏差来换稳定性，因此不是任意陈旧轨迹的无偏重放器。
- **clipping**：把 ratio 限制在 `[1-ε, 1+ε]` 再取 min，「防止某条数据把策略拉得太远；取 min：只让正向优势（好动作）在限制范围内放大，避免过度优化」（nano-verl tutorial_L1 机制拆解 §5 逐字）。advantage 由 GAE [1506.02438]（2015-06-08，Schulman et al.）估计：用 γ 折现未来奖励、用 λ 在真实回报与 value 差分之间做偏差-方差权衡（nano-verl tutorial_L1 机制拆解 §4 表述）。

### 2.2 nano 实证：一个 28K 参数 LSTM 的最小 PPO 闭环

nano-verl L1（`L1_minimal_ppo.py`，真实输出 Apple MPS，seed=42）在字符级 LSTM（27,966 参数，现场 `sum(p.numel())` 统计）上跑通 generate → advantage → update 全循环：配置 rollouts=32、outer_iter=80、ppo_epochs=4、lr=0.001、CLIP_EPS=0.2；先 100 步 SFT warmup（动机：「PPO 依赖采样。如果初始策略完全随机，采样到高质量 response 的概率极低，训练信号就会很差」——tutorial_L1 机制拆解 §1 逐字），再 80 轮 PPO。实测曲线（tutorial_L1「先跑起来」节输出节选，省略中间字段）：

```
[warmup] cross-entropy loss=1.2933
[iter   0] reward=0.831 ... value_loss=1.073 entropy=0.549 approx_kl=0.0004
[iter  30] reward=1.000 ... value_loss=0.001 entropy=0.108 approx_kl=0.0000
[iter  79] reward=0.988 ... value_loss=0.001 entropy=0.024 approx_kl=0.0016
```

三条曲线各讲一件事：value_loss 1.073→0.001 是 **critic 的学习曲线**（GRPO 将整条省掉，§3）；entropy 0.549→0.024 表示这个单答案 toy 的策略逐渐集中，真实长序列任务若过早出现同样趋势则可能是探索坍缩；approx_kl 监控新旧策略的实际漂移。它很小与 clip 目标一致，但**不能单独证明** clipping 正在起作用，还应结合 ratio/clip fraction 或关闭 clip 的对照实验。

### 2.3 当今定位（A 层必注）

「PPO（arXiv:1707.06347，2017）是经典锚点，但**已不是前沿 RLVR 的首选算法**——前沿后训练（如 Qwen3、Kimi 系列的 RL 阶段）主流是 GRPO 族（DAPO / GSPO / CISPO 等）与 RLVR：它们去掉了 value model，用组内相对 reward 估计 advantage。但 PPO 的核心思想——importance sampling ratio + clipping 限制策略更新幅度——**直接流入了 GRPO 族**」（nano-verl tutorial_L1 机制拆解 §1「PPO 的当今定位」段逐字；该判断由 §3/§4 的一手来源逐条坐实）。教 PPO 是教机制地基，不是教「前沿就是这么训的」。

---

## §3 GRPO 与 RLVR：去掉 value model，reward 换成可验证的

### 3.1 GRPO：组内相对 advantage 是「穷人版 critic」

DeepSeekMath [2402.03300]（2024-02-05，Shao et al.）摘要原文：GRPO 是「a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO」。机制：对同一 prompt 采一组（group）response，advantage 取组内相对值（均值作 baseline），不再训练 value model——「GRPO 路径则干脆不要 critic（组内均值当 baseline）」（nano-verl tutorial_L3 §4 逐字）。

为什么这是真进步而不是偷懒（推断，与两处一手来源一致）：critic 是 PPO 里第二个与 policy 同量级的模型——参数、优化器状态、训练目标、失败模式全部翻倍。nano-verl L3 的 declared 显存算术给了量级感：训练态每 rank 居住 4P/N（参数分片 + Adam m/v + 梯度），7B 模型 N=4 卡时约 34 GB/rank（tutorial_L3 §5 COST 模型；该 34 含激活 6 GB 声明常数，表口径 = 4P/N + 6 GB）——多养一个 critic 意味着训练态显存接近翻倍。GRPO 用「组内相对」这个统计量替代了一个学习出来的 value 函数，代价是每组要多采样（采样换参数），而采样恰恰可以被高吞吐推理引擎加速（§7）。该文报告 DeepSeekMath 7B 在 competition-level MATH 达 51.7%（无外部工具/投票），64 样本 self-consistency 60.9%（摘要数字）。

### 3.2 DeepSeek-R1：RLVR 的分水岭

DeepSeek-R1 [2501.12948]（2025-01-22，DeepSeek-AI，v2 2026-01-04）摘要原文：推理能力可以「incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories」——纯 RL、不需要人工标注推理轨迹；训练中涌现 self-reflection、verification、dynamic strategy adaptation；且大模型的涌现推理模式「can be systematically harnessed to guide and enhance the reasoning capabilities of smaller models」（蒸馏伏笔，§6 接住）。

RLVR（RL with verifiable rewards）的本质变化不在算法、在 **reward 来源**（推断）：从「人类偏好标注 + reward model」换成「答案对错 / 测试通过」这类可程序化验证的信号。后果有三：reward 不再需要标注产能与 RM 训练（成本结构变了）；reward 不会被 RM 的拟合误差污染（reward hacking 的形态变了）；任务必须可验证——所以 RLVR 的主场是数学/代码/STEM（R1 摘要的「verifiable tasks such as mathematics, coding competitions, and STEM fields」）。DAPO 摘要的旁证：「key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report)」——R1 之后社区复现潮本身就是 RLVR 成为主流的场证据。

### 3.3 nano 实证：critic 的存在感与 on-policy 的刚性

nano-verl 阶梯把 GRPO 的两个机制面都做成了可运行对照：L1 的 value_loss 曲线（§2.2）是 critic 存在感的直接测量；L3 则固定「每轮 rollout 紧跟最新权重」的同步口径，展示两相串行时的 colocate 算术。放宽到异步可以跨 step overlap，但会产生 policy staleness；IS ratio 只能局部修正同一 prefix 上的动作分布，不能把任意旧轨迹变回新策略轨迹（见 nano-slime L0 与 tutorial_L1 IS 小节）。GRPO 去掉 critic 并没有消除这个采样分布与训练分布的权衡。

---

## §4 GRPO 族的机制类别：变体在修什么

2025 年 GRPO 变体爆炸。按 §八 C 层纪律，本文不逐个教方法名，只教**机制类别**：每个有 B 层证据（前沿模型采用 / 大规模开源）的变体，修复的都是 token 级 importance sampling 的一个具体缺陷。

### 4.1 四个 B 层变体各自修什么（一手来源摘要/官方博客）

| 变体 | arXiv | 修复的缺陷 | 采用证据 |
|------|-------|-----------|----------|
| DAPO | 2503.14476（2025-03-18） | Decoupled clip + Dynamic sAmpling（摘要只给技术名；「分离上下裁剪界、过滤零优势样本组」为领域通行解读，正文层细节标 [TODO: verify]） | Qwen2.5-32B base 达 AIME 2024 50 分；系统完全开源、构建于 verl 框架之上（摘要声称） |
| Dr. GRPO | 2503.20783（2025-03-26） | GRPO 的 optimization bias——「artificially increases response length (especially for incorrect outputs)」；提出无偏估计 | 7B base 达 AIME 2024 43.3%（摘要声称）；其批评是长度控制类 trick 的机制依据 |
| CISPO | 2506.13585 内（MiniMax-M1，2025-06-16） | 「clips importance sampling weights rather than token updates」——裁剪对象从更新量换成 IS 权重本身 | MiniMax-M1 全 RL 训练：512 H800、3 周、租金 $534,700（摘要声称） |
| GSPO | 2507.18071（2025-07-24，Qwen 团队） | token 级 ratio 的噪声与 MoE 不稳定：ratio 改定义在 sequence likelihood（token ratio 的几何平均、长度归一）上，clipping/rewarding/optimization 全部序列级 | 「contributed to the remarkable improvements in the latest Qwen3 models」（摘要逐字）；官方博客 2025-07-27 |

GSPO 官方博客（qwenlm.github.io/blog/gspo/，2025-07-27 发布，当日现场重抓）的三个一手观察值得逐字记录：

1. GRPO 长训「exhibit severe instability issues during long training and lead to irreversible model collapse」——这是 Qwen 团队换序列级目标的直接动机。
2. GSPO 被 clip 的 token 比例比 GRPO **高两个数量级**，训练效率反而更高——「GRPO's token-level optimization objective is noisy and inefficient」，clip 频率高不是病，clip 在错误的粒度上才是。
3. GRPO 训练 MoE 需要 **Routing Replay**（缓存 π_old 激活的专家路由、在 π_θ 计算 ratio 时重放）才能收敛，GSPO 完全消除该依赖——因为序列级似然对单 token 似然不敏感。

### 4.2 机制类别总结（推断，逐条有上述一手来源支撑）

把四个变体投影到两个问题上：

- **clip 在什么粒度**：token 级（GRPO/PPO）→ 解耦的高低 advantage clip（DAPO）→ IS 权重本身（CISPO）→ 序列级（GSPO）。趋势是**把更新控制从 token 噪声里拔出来**。
- **估计在哪里有偏**：Dr. GRPO 指出 GRPO 存在会人为拉长回答（尤其错误回答）的 optimization bias（摘要原文，§4.1）——advantage 估计的偏差不是方差问题，加大组采样也消不掉（与 nano-opd L0 的 off-policy 偏差定理同构，§6.1）。

**nano 实证**：nano-verl L1 的监控面（approx_kl/entropy 逐 iter 输出，§2.2）正是 token 级更新控制所需的观测面——entropy 0.549→0.024 显示策略在这个单答案 toy 上集中；而 GSPO 声称可省掉的 old_log_prob 训练侧重算步骤，在 verl v0.7.1 源码里就是 `ray_trainer.py:L1132 _compute_old_log_prob`（行号以 2026-08-07 抓取日为准，锚点见 nano-verl tutorial_L3 §11）——**算法选择直接增删 infra 步骤**，这是 §7 主题的预告。

---

## §5 DPO：离线支线，以及它为什么不是 RLVR 主场

DPO [2305.18290]（2023-05-29，Rafailov et al.）是 A 层经典锚点。机制（摘要层）：KL 约束下的偏好优化存在闭式解，language model「is secretly a reward model」——于是可以跳过显式 reward model 与 RL 循环，直接在偏好对 (chosen, rejected) 上做监督式优化。

**当今定位**（§八 A 层必注，推断与文献已有区分如下）：DPO 系方法在「静态偏好数据充足、探索价值低」的场景仍是高效选择（文献已有：DPO 及其变体在工业对齐管线中广泛使用——此为领域常识级表述，不引具体数字）；但在 RLVR 主场（数学/代码/推理）它让位给在线 RL，机制原因有二（推断，基于两处摘要对照）：其一，DPO 的训练信号来自**已标注的偏好对**，而推理任务的价值恰在探索出标注里没有的解法——R1 摘要的「pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories」正是对这一点的反向陈述；其二，离线数据与当前策略的分布差会随训练扩大，DPO 没有 IS 机制去修正它（PPO/GRPO 有 ratio，OPD 有 on-policy 采样）。OPD survey [2604.00626] 摘要把这条线缝了回来：该文显式讨论「the connection between OPD and KL-constrained reinforcement learning」——KL 约束闭式解的思想在 OPD 框架里继续活着。

**nano 锚点**：偏好对的数据侧构造（chat template / loss mask / pairwise collator）在 nano-llamafactory L0–L1 已可运行（SFT 数据侧三件套 ✅）；DPO 本身的可运行对照见 [nano-llamafactory L2](../nano-llamafactory/tutorial_L2.md)，其中明确区分“经典机制地基”和“当前前沿配方”。

---

## §6 OPD：蒸馏与 RL 的合流点

### 6.1 算术定理：散度期望在谁下面，样本就从谁采

OPD 的最小机制不是工程习惯，是一条算术（nano-opd tutorial_L0 §6 逐字）：

```
∇_θ KL(q_θ || p*) = E_{x~q_θ} [ ∇_θ log q_θ(x) · (log q_θ(x) − log p*(x)) ]
```

「**期望的下标是 q_θ——学生自己的分布。** 这不是实现选择：散度定义里的期望写在谁下面，无偏估计就必须从谁那里采样。」reverse KL 的期望在学生分布下 → 必须学生自采样（on-policy）；教师只需给每个样本打分（logprob）——「学生写、教师批」的接口。

nano-opd L0（零依赖确定性 toy：双峰教师 0.5·N(−3,1)+0.5·N(+3,1)，单峰受限学生唯一可学参数是峰位 mu，模/谷比 ≈45x）把这条定理做成了三组实证（tutorial_L0 §2 输出 + §4–§6 逐配方，数字逐字）：

- **三配方对照**：sft（教师样本+硬标签）mu=−0.227、模区驻留率 0.054；fwd（KL(p*||q) 精确求和）mu=+0.000、驻留率 0.045——两条 off-policy 路线都把 95% 的质量堆进教师的谷；rev（KL(q||p*)，学生自采样）mu=+3.000、驻留率 0.955、教师认账度 E_q[p*]=0.1559 ≈ 前两者的 8 倍。
- **驻点算术定理**（tutorial_L0 §7：在锁模点 mu=+3 对 13 个网格点精确求和，无采样噪声）：正确估计器 E[g]=+0.002 ≈0（模是驻点，待得住）；错误估计器（期望换成教师分布）E[g]=+136.149（被系统性拽离）。「**偏差不是噪声。** 噪声型错误加大 batch 能摊薄，系统性偏差加到天荒地老也是同一个方向」（§9 逐字）。
- **off-policy 真跑反例**（tutorial_L0 §9）：教师采样 300 步，mu=−0.046、驻留率 0.045——锁不住模，掉回谷里。30 个 seed 全部锁模（|mu|>2.5，30/30，§7）排除了运气解释。

### 6.2 真实模型验证：2×2 因子设计隔离「信号源 × 散度」

nano-opd L1（`L1_real_opd_seqdistill.py`，真实 tokenizer V=66 + 两个真实小 Transformer：教师 110,530 / 学生 4,914 参数，22x 容量差；同一初始权重、各 300 步、batch=32、lr=0.002）的四格结果（tutorial_L1 §1 输出逐字）：

```
sft        有效率=0.226 | A模占有效=0.664 | 教师背书=-1.774 nats/token
kd         有效率=0.263 | A模占有效=0.638 | 教师背书=-1.642 nats/token
opd_off    有效率=0.053 | A模占有效=0.301 | 教师背书=-2.461 nats/token
opd        有效率=0.168 | A模占有效=0.165 | 教师背书=-1.704 nats/token
```

因子解读（tutorial_L1 §1 逐字）：信号源固定（教师序列）换散度，kd(fwd) 0.263 vs opd_off(rev) 0.053——reverse KL 用在教师前缀上不锁模；散度固定（rev）换信号源，opd_off 0.053 vs opd（学生自采样）0.168——**on-policy 是 reverse KL 的算术要求**，L0 的玩具定理在真实模型上逐格复现。opd 的 A 模占有效 0.165——只有 16.5% 的有效输出落在 A 模，即「**锁定了 codebook B（大写）**」（tutorial_L1 §4.4 逐字），这是 mode-seeking 在真实模型上的形态。两个附加实测：**相变动力学**——有效率在 step 240–299 之间从 0 跳到 0.171，而教师背书度（= 教师对学生采样序列的平均 token logprob）从 −9.342 一路升到 −1.703，是先行指标（§5.2）；**跨环境可复现**——base（py3.13.13+torch 2.13.0）与 longds（py3.12.13+torch 2.4.1）两环境各 3 遍 EXIT=0，定性结论一致（opd 绝对值 0.168 vs 0.214，背书最高、锁模、opd_off 瘫；§11.1 两环境对照表）。

### 6.3 生产配方：Qwen3、Thinking Machines、MiMo-V2-Flash

OPD 在 2025–26 成为生产成分，一手证据链：

- **Qwen3 技术报告** [2505.09388]（2025-05-14）摘要声称：「by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models」——旗舰模型知识下沉是小模型构建的核心手段（摘要未展开算法细节；OPD survey 点名 Qwen3 采用 OPD，见 nano-opd tutorial_L0 §15 2026-08-05 核验记录）。
- **Thinking Machines OPD 一手文**（2025-10-27，Kevin Lu，当日现场重抓 82,782 B）：实现三步 = 学生采样 → 教师 compute_logprobs → 「We set the per-token advantage to the negative reverse KL, and call the RL importance-sampling loss function to perform the training update on the student model」（逐字）——**OPD 的 loss 就是 PPO 的 IS loss，reward 换成负 reverse KL**，§1 合流线在此闭环。数字（博客自述 + 转引 Qwen3 报告 Table 21，标 `[blog claims]`）：同一 SFT 初始化上，off-policy distillation AIME'24 55.0% / GPQA-Diamond 55.6%；+RL 67.6% / 61.3% / 17,920 GPU hours；+OPD **74.4% / 63.3% / 1,800 GPU hours**——「reaching a higher score of 74.4 on AIME'24 at one-tenth the cost of RL」。TM 自测（Qwen3-8B-Base 学生，OpenThoughts-3/QwQ-32B 教师数据 SFT-400K → AIME'24 60%，再 OPD → 70%）：baseline cost reduction 9x（CE 比 9–30×，教师 logprob 计算 FLOPs 计入）。
- **OPD survey** [2604.00626]（2026-04-01，v4 2026-06-18，Song et al.）摘要声称：把 OPD 形式化为「f-divergence minimization over student-sampled trajectories」，沿三轴组织领域（what to optimize / where the signal comes from / how to stabilize），并给出 exposure bias 的量化动机——静态模仿教师文本的错误「scale roughly with the square of sequence length」，OPD 把复合项压向线性；open problems 包括 distillation scaling laws、uncertainty-aware feedback、agent-level distillation、KD-RL overlap。
- **MOPD** [2606.30406]（2026-06-29，Ma et al.）摘要声称：multi-teacher OPD——先 per-domain RL 训出领域教师，再在学生自身 rollout 上蒸馏（「eliminates exposure bias and provides a dense optimization signal」）；在 Qwen3-30B-A3B 上优于 Mix-RL / Cascade RL / Off-Policy Finetune / Param-Merge；**已部署于 MiMo-V2-Flash（industrial-scale frontier model）**。multi-teacher OPD 的其他单篇变体（MAD-OPD / Uni-OPD 等）无已核验 arXiv ID，按 C 层只作机制类别（「多教师融合」）出现。（survey v3 方法表所载同名 MOPD 与本论文是否同一工作：时间线吻合，合理推断为同一，标 [TODO: verify]。）

### 6.4 机制观点：OPD = RL 的采样 infra + 教师的 dense reward

推断（与上述全部来源一致）：OPD 不是「蒸馏换了个名字」，而是把 RLVR 的稀疏可验证 reward 换成教师的 dense 背书——采样侧完全不变，所以「OPD 复用 RL 的 rollout infra——采样吞吐问题与 nano-slime / nano-vllm-sglang 同源」（nano-opd README 逐字）。这解释了两个生产现象：TM 文中 OPD 的 1,800 vs 17,920 GPU hours 差距主要来自 dense 信号消灭了「靠海量采样换一个可验证成功」的 RL 成本结构；而 MOPD 能作为 post-training primitive 出现，是因为领域教师的 rollout 与学生 rollout 走的是同一套管线。reward 来源轴（§1）从「可验证答案」走到「教师打分」，能力边界就从可验证任务扩到了不可验证任务（对话风格、领域知识）——这是 OPD 在 2026 格局里的位置。

---

## §7 算法与 infra 的共演化

算法选择不是纸面数学，它直接决定集群怎么搭；反过来，infra 的约束也写进了算法的动机里。这一面 nano-verl 阶梯提供了全谱实测。

### 7.1 算法的内存画像塑造架构：GRPO 动机 → colocate 算术

GRPO 的原始动机之一就是「optimizing the memory usage of PPO」（§3.1 摘要逐字）——去掉 critic 直接改变训练态显存画像。在严格新鲜 rollout 的同步口径中，两相串行把问题推给调度：拆卡（disagg）= 每相一部分算力 + 其余卡等待；复用（colocate）= 每相全部算力 + 阶段边界付 resharding 税。异步口径则是另一条以 staleness 换 overlap 的路线。

nano-verl L3（对照 verl v0.7.1 源码，单控制器+SPMD、DataProto chunk/concat、阶段边界 resharding；强制 CPU、~8s、self-check 10/10）的 declared 算术（tutorial_L3 §5 COST 模型，7B/13B 折算，非 GPU 实测）：

```
scale   colo-train  colo-rollout  disagg-train  disagg-rollout  budget(GB)
7B            34.0          34.0          62.0            34.0        80
13B           58.0          46.0         110.0            46.0        80
disagg trainer OOM @7B: False   OOM @13B: True
```

「7B 时 62 GB 还能挤（这就是为什么小模型时代拆卡架构活得下去），13B 时 110 GB > 80 GB，**直接 OOM**」（tutorial_L3 §5 逐字；4P/(N/2) = 2× colocate 的 4P/N）。时钟侧（tutorial_L3 §1）：同口径 colocate 每步 2100ms vs 拆卡同步 3900ms（η=1.0，**1.86x 全部来自一半卡空转**；η=0.85 仍 1.69x）。resharding 税有 toy 尺度实测对账：30 步训练搬运 101,602 KB，手工对账 104,040,960 B 与程序输出逐位吻合——「**resharding 是 colocate 的税；税率由互连带宽决定，税基由模型大小决定**」（tutorial_L3 §6 逐字）。结论句（tutorial_L3「先跑起来」节输出 takeaway 逐字）：「colocate 不是性能 trick，是显存算术的必然」。

### 7.2 算法的数值特性塑造管线：GSPO 免重算 vs verl 的重算步

GSPO 的序列级 ratio「much more tolerant of precision discrepancies」，因此「makes it possible to directly use likelihoods returned by inference engines for optimization, eliminating the need for recomputation with training engines」（官方博客逐字）——特别利好 partial rollout、multi-turn RL、训推分离框架。对照 verl v0.7.1：训练引擎重算 old_log_prob 是管线的实在步骤（`ray_trainer.py:L1132`，锚点见 nano-verl tutorial_L3 §11，行号以 2026-08-07 抓取日为准）。**同一个 infra 步骤，token 级算法必须有、序列级算法可以删**——算法与 infra 共演化不是抽象口号，是可以用行号指认的具体增删。

### 7.3 actor-learner 分离：权重同步是脐带

nano-verl L2（当前 CPU-only 输出，seed=42）演示 lockstep 两进程分离：模型 237,150 参数、rollouts=256、outer_iter=40；serial 8.43s vs split 8.18s = **1.03x（基本持平）**，reward parity 0.999/0.999。协议仍是 `rollout → train → sync`，因此角色拆分没有自动形成 overlap；Queue、权重快照和调度本身还有成本。权重同步在 toy 中用 `detach().cpu().clone()` 生成稳定、可移植的 Queue 快照；CUDA IPC 原则上可用但有进程生命周期约束，生产系统通常使用 NCCL 等设备通信。同步间隔则在通信成本与 policy staleness 之间取舍。

### 7.4 2026 的 infra 前沿：agentic RL 把轨迹拉长到百万 token

Kimi K3 技术报告 [2607.24653]（2026-07-27，v2 2026-08-07，Kimi Team）摘要声称：2.8T MoE / 104B 激活 / 1M 上下文；post-training = 「reinforcement learning across general, agentic, and coding domains and multiple reasoning-effort levels」；infra 侧的关键新词是「**million-token agentic RL with persistent rollout and sandbox states**」——agentic RL 把 rollout 从「一问一答」拉长到跨环境状态的长时程轨迹，rollout 引擎要能持久化沙箱与中间状态。（当前证据只覆盖摘要，正文细节标 [TODO: verify]。）这与 04 轨 harness engineering 的长时程主题（状态外化、检查点、可审查停止点）是同一枚硬币的两面：训练侧要持久化 rollout 状态，推理侧要外化 agent 状态——长轨迹既是训练数据也是工程对象（总导航的四轨闭环 RSI 闭环：agent 轨迹回流数据侧）。DAPO 摘要的生态证据补一句：该开源 RL 系统「built on the verl framework」——verl 系 infra 是 RLVR 开源复现的事实底座（nano-verl 对标的正是它）。

---

## §8 2026 格局：三层锚点定位

对齐日 2026-08-11（§10 全部当日核验）。结论先行：**未发现取代 GRPO 族 / RLVR / OPD 的新一代范式**；OPD 在向 multi-teacher / agent-level 扩张（survey open problems），RL 在向 agentic 长时程扩张（Kimi K3）。

| 层 | 本主题的条目 | 处理 |
|----|--------------|------|
| **A 经典锚点** | PPO [1707.06347]、GAE [1506.02438]、DPO [2305.18290]、MiniLLM [2306.08543]、GKD [2306.13649]、DistiLLM [2402.03898]；verl/HybridFlow [2409.19256] 实现 | 机制仍是地基，教本质；当今定位已逐条注明（PPO 思想流入 GRPO 族；DPO 为离线支线；MiniLLM/GKD/DistiLLM 是 OPD 的经典锚点而非前沿本身） |
| **B 前沿主流** | GRPO [2402.03300 内提出] → DAPO [2503.14476] / Dr. GRPO [2503.20783] / CISPO [2506.13585] / GSPO [2507.18071]；RLVR [2501.12948]；OPD 生产化 [2505.09388 / TM blog 2025-10-27 / 2604.00626 / 2606.30406]；Kimi K3 agentic RL [2607.24653] | 多独立来源支撑：GRPO 族变体各有前沿模型采用证据（Qwen3/M1/MiMo-V2-Flash）；OPD 有 survey + 三家生产配方；本文主体内容 |
| **C 中间状态** | XPO 类离线偏好变体、MAD-OPD / Uni-OPD 等单源 OPD 变体（无已核验 arXiv ID） | 只讲机制类别（token 级重加权 / 多教师融合），不追单论文微创新；晋升 B 层需 ≥2 个独立验证 |

存在性数据点：Qwen3.5 系列已发布（qwen.ai blog「Qwen3.5: Towards Native Multimodal Agents」；Qwen3.5-Omni Technical Report [2604.15804]，2026-04-17，摘要主题为全模态）——尚未提取其 post-training 配方的一手文本（博客 JS 渲染），标 [TODO: verify]。DeepSeek-V4 采用 pure multi-teacher OPD 的说法系 survey 正文转述（nano-opd §15 已录 [TODO: verify]），本文维持转述口径。

---

## §9 费曼自检

### 9.1 讲给外行听

把训练一个模型想成培养一个学徒厨师。**SFT** 是照着师傅的成品菜谱抄——抄得快，但学徒从没自己做过菜，一遇到菜谱没写的情况就露馅（exposure bias：错误随序列长度平方累积）。**RL（RLVR）** 是让学徒自己做菜、按答题卡打分：菜能不能吃（答案对不对）是硬标准，但一道菜只得到一个「对/错」，不知道哪一步做错了（sparse reward），只好疯狂试错（海量采样）。**OPD** 是请师傅站在旁边，学徒自己做，师傅每一口都尝、每一刀都评（教师逐 token 打分，dense reward）——学徒做的必须是自己做得出来的东西（on-policy 的算术由来：要批改的是学生的作文，不是抄来的范文），所以「学生写、教师批」省不掉学生自己写这一步。而贯穿三代的那把「更新幅度限幅器」（PPO 的 clipping）始终装在灶台上：不管打分方式怎么变，一次尝菜不能把菜谱改得面目全非。GRPO 族的种种变体，是在争论「打分应该按每一刀还是按整道菜」（token 级 vs 序列级）——争论的裁判是厨房的实际情况：灶台（MoE 路由）稳不稳、传菜（推理引擎与训练引擎）要不要对两次账。

### 9.2 思考题

1. GRPO 用组内均值当 baseline，省掉了 critic。为什么「多采样几个 response」比「多训练一个 value model」在工程上更划算？（提示：§3.1 的显存画像 + §7 的 rollout 吞吐 infra；nano-verl L3 declared 算术 4P/N）
2. Dr. GRPO 指出 GRPO 会人为拉长回答（尤其错误回答）。如果你在自己的 RLVR 管线里怀疑长度偏置，会观测哪几个量、做什么对照实验？（提示：§4.2 偏差 vs 方差；nano-opd L0 驻点算术「偏差不是噪声」的对照设计——精确求和 E[g] 而非多跑几遍）
3. GSPO 被 clip 的 token 比例比 GRPO 高两个数量级，训练反而更好。这说明「clip 频率」和「clip 是否有效」是什么关系？（提示：§4.1 博客引文——clip 在错误的粒度上才是病）
4. OPD survey 说静态模仿的 exposure bias「scale roughly with the square of sequence length」。用一阶近似推一遍这个平方律从哪来，并解释为什么 OPD 能把它压向线性。（提示：每步错误概率 × 剩余步数；nano-opd L0 的 off-policy 反例是它的玩具版）
5. TM 文把 OPD 的 loss 写成「per-token advantage = 负 reverse KL + RL IS loss」。对照 nano-opd L1 的 token 级实现与 MiniLLM 的序列级 REINFORCE，说出两者在方差与计算成本上的取舍。（提示：nano-opd tutorial_L1 §6「L0 实现了序列级版本，L1 实现了 token 级版本（方差更低，生产配方更常用）」）

### 9.3 反例（流行但错的说法）

1. **「GRPO 就是去掉 critic 的 PPO，仅此而已」**——去掉 critic 只是算法面；RLVR 的实质变化在 reward 来源（人类偏好 → 可验证信号），它改变了成本结构、reward hacking 形态与任务边界（§3.2）。只讲「去 critic」会把 R1 讲成一个 trick。
2. **「OPD 是蒸馏换了个新名字」**——off-policy 蒸馏（SFT on 教师文本）与 OPD 是两件事：nano-opd L0 实测同教师同学生，SFT 路线 95% 质量掉谷、OPD 路线 95.5% 锁模；驻点算术 +0.002 vs +136.149 证明差别是算术级的（§6.1）。
3. **「GRPO 变体越多越强，应该追最新的」**——变体爆炸是 §八 C 层现象。机制类别只有几个（clip 粒度、偏差修正、采样策略）；DAPO/GSPO/CISPO 值得教是因为有前沿模型采用证据，不是因为论文新（§4.2）。
4. **「DPO 已经取代 RLHF/RL」**——DPO 是离线支线（§5）；前沿推理模型的 post-training 主干是 RLVR + OPD（Qwen3 / R1 / Kimi K3 / MiMo-V2-Flash 的一手证据链，§3/§6）。
5. **「算法改进不影响 infra，infra 不影响算法」**——GRPO 的动机里写着 PPO 的内存（摘要原文）；GSPO 删掉了 verl 管线的 old_log_prob 重算步（§7.2 行号锚点）；colocate 是显存算术的必然（nano-verl L3）。算法-infra 共演化是双向的、可指认的。

### 9.4 局限

- Kimi K3 [2607.24653] 与 MOPD [2606.30406] 只核验到摘要层（标题/日期/作者/摘要为 2026-08-11 arXiv API 现场抓取），正文实验细节标 [TODO: verify]。
- Thinking Machines 博客数字（55.0/67.6/74.4、17,920/1,800 GPU hours、9x）为博客自述与转引 Qwen3 报告 Table 21，非独立复算，标 `[blog claims]`。
- Qwen3.5 系列 post-training 配方细节未提取到一手文本（博客 JS 渲染），标 [TODO: verify]；DeepSeek-V4 的 OPD 采用系 survey 转述，未直接核验（nano-opd §15 已录 [TODO: verify]，维持）。
- nano 实证全部为 toy 尺度（28K–237K 参数模型、CPU/MPS）：机制可迁移，量级不可外推；nano-verl L3 的 7B/13B 显存/时钟数字是 declared COST 模型算术，非 GPU 实测（其教程已显式区分 declared vs real toy，本文遵守该口径）。
- MOPD 与 survey v3 方法表同名方法的同一性为合理推断（时间线吻合），标 [TODO: verify]。

---

## §10 溯源与口径

### 10.1 一手来源清单（全部 2026-08-11 现场核验）

**arXiv（18 个 ID，export.arxiv.org API 批量核验标题/日期/作者/摘要）**：

| arXiv ID | 标题 | 首发日期 | 本文引用位置 |
|----------|------|----------|--------------|
| 1506.02438 | High-Dimensional Continuous Control Using Generalized Advantage Estimation | 2015-06-08 | §2.1 |
| 1707.06347 | Proximal Policy Optimization Algorithms | 2017-07-20 | §2.1, §8 |
| 2305.18290 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | 2023-05-29 | §5, §8 |
| 2306.08543 | MiniLLM: On-Policy Distillation of Large Language Models | 2023-06-14（v6 2026-01-31） | §8（A 层锚点） |
| 2306.13649 | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | 2023-06-23 | §8（A 层锚点） |
| 2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | 2024-02-05 | §3.1, §4 |
| 2402.03898 | DistiLLM: Towards Streamlined Distillation for Large Language Models | 2024-02-06 | §8（A 层锚点） |
| 2409.19256 | HybridFlow: A Flexible and Efficient RLHF Framework | 2024-09-28 | §7（verl 对标） |
| 2501.12948 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 2025-01-22（v2 2026-01-04） | §3.2, §5, §8 |
| 2503.14476 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | 2025-03-18 | §4.1, §7.4 |
| 2503.20783 | Understanding R1-Zero-Like Training: A Critical Perspective | 2025-03-26 | §4.1, §4.2, §8 |
| 2505.09388 | Qwen3 Technical Report | 2025-05-14 | §6.3, §8 |
| 2506.13585 | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention | 2025-06-16 | §4.1（CISPO） |
| 2507.18071 | Group Sequence Policy Optimization | 2025-07-24 | §4.1, §7.2 |
| 2604.00626 | A Survey of On-Policy Distillation for Large Language Models | 2026-04-01（v4 2026-06-18） | §5, §6.3, §8 |
| 2604.15804 | Qwen3.5-Omni Technical Report | 2026-04-17 | §8（存在性数据点） |
| 2606.30406 | MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training | 2026-06-29 | §6.3, §8 |
| 2607.24653 | Kimi K3: Open Frontier Intelligence | 2026-07-27（v2 2026-08-07） | §7.4, §8 |

**一手博客（当日现场重抓全文）**：

| 文章 | 来源 | 发布 | 抓取件 |
|------|------|------|--------|
| GSPO: Towards Scalable Reinforcement Learning for Language Models | qwenlm.github.io/blog/gspo/（Qwen Team） | 2025-07-27 | 抓取件 25,141 B |
| On-Policy Distillation | thinkingmachines.ai/blog/on-policy-distillation/（Kevin Lu in collaboration with others at Thinking Machines） | 2025-10-27 | 抓取件 82,782 B |

**负对照与交叉核验**：18 个 ID 全部返回真实条目且标题与 对应 nano 教程记录逐词吻合（MiniLLM/GKD/DistiLLM 标题 vs nano-opd tutorial_L0 §15；HybridFlow 标题 vs nano-verl tutorial_L3 §11；OPD survey v4 updated 2026-06-18 vs nano-opd §15 录值）——无一处 ID 误归属。

### 10.2 课程内可运行对照

- nano-verl L0–L3：`../nano-verl/`（L0 调度 toy 68.0→47.0ms/1.45x；L1 最小 PPO 闭环 27,966 参数/reward 0.831→0.988 + IS 边界；L2 lockstep actor-learner 角色拆分，CPU-only 8.43s→8.18s/1.03x、基本持平；L3 HybridFlow colocate declared 显存算术 7B/13B + resharding 税 101,602 KB 对账 + self-check 10/10，metrics md5 `5cba79e6…`；verl v0.7.1 源码锚点表见其 tutorial_L3 §11，行号以 2026-08-07 抓取日为准）
- nano-opd L0–L1：`../nano-opd/`（L0 三配方对照/驻点算术 +0.002 vs +136.149/30/30 seed 锁模；L1 2×2 因子设计四格/相变 step 240–299/背书度 −9.342→−1.703/两环境复现；§15 SOTA 对齐记录 2026-08-05 为本文对齐的前置基线）
- nano-llamafactory L0–L1：`../nano-llamafactory/`（SFT 数据侧三件套；DPO 对照排 L2 待补，本文未预支）

### 10.3 口径声明

- 四类信息区分：「摘要/原文声称」= arXiv 摘要或博客原文（2026-08-11 当日抓取）；「文献已有」= 已发表结论；「推断」= 本文作者的机制推断（已逐处标明）；无猜测级内容入正文。
- nano 实测数字来自所链接教程（文内标注文件与节号）；本文没有把这些数字重新解释为 GPU 或生产规模证据。复现命令、环境口径和输出锚见各模块 README。
- declared vs real toy 严格区分：nano-verl L3 的 GB/ms 数字为 COST 声明模型算术，toy 流量为实测但规模是 28K 参数——两类数字在本文均按其原始口径标注。
- [TODO: verify] 遗留五项：Kimi K3 正文层；MOPD 正文层；Qwen3.5 配方细节；DeepSeek-V4 技术报告；MOPD 与 survey 同名方法的同一性。
- 本文不引入任何 C 层单论文方法作为教学内容（§八）；MAD-OPD / Uni-OPD 等只作为机制类别的载体出现，不补造 arXiv ID。
