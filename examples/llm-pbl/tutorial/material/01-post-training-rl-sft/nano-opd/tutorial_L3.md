# nano-opd L3 — 生产配方的四个旋钮：同一个循环，Qwen3 / MiMo / TM / DSv4 各拧到哪

> L0–L2 已经证明 OPD 的核心算术：散度选择决定采样需求（L0）、真实序列模型上
> 的 2×2 因子验证（L1）、多教师融合/路由（L2）。L3 问生产问题：**同一个
> on-policy 循环上有四个可调旋钮，生产配方各自拧到了哪里、为什么？**
> 四个旋钮 = survey `[2604.00626]` taxonomy 的机制类别重组（映射见 §8）：
> divergence 设计 / 信号源（白盒/黑盒）/ token 加权 / 效率与稳定化。

---

## 1. 先跑起来

```bash
# Python 3.10+ 与 PyTorch；任意空目录可跑，-B 防止 __pycache__
$ cd $(mktemp -d)
$ python3 -B /path/to/nano-opd/L3_production_recipe_axes.py
```

**可运行性契约**：教师/学生的前向、反向与采样是实际 torch 计算，
「黑盒 API」是接口约束（`teacher_logprobs_api` 只向外暴露采样 token 的
logprob，全词表 logits 永不离开该函数），不是假数据。任务是 toy（封闭双语
密码本），只证明固定机制和反例可复现，不证明生产模型质量或吞吐（§11）。
CPU 运行通常需要数分钟；group 采样路线的 ×16 采样开销是机制的一部分（§4）。

真实输出（固定 seed；stdout 不含墙钟与本机路径）：

```text
========================================================================
nano-opd L3 — 生产配方的四个旋钮（survey taxonomy 机制类别 ×4）
========================================================================
任务: 4 位数字 → 小写 codebook A 或大写 codebook B 都算对
词表 V=66 | 响应长 9 | 学生自采样 on-policy 循环（全轴共用）

[1] 双语教师（1000 步）params = 110,530（教师/学生 ≈ 22x，学生 4,914）
    教师自采样: valid_A=0.531 valid_B=0.455 frac_A=0.539（教师逐样本随机选模，分域精确匹配 ≈ 0.5×准确率 是构造使然）

[2] 轴1 divergence 设计（同一 on-policy 循环，各 500 步）
    β-JSD 族探针（初始化学生、teacher-forced，无需训练）
      对称性 β=0.5: max|JSD(p‖q)−JSD(q‖p)| = 0.00e+00（float32 机器精度）
      端点标度（β=1e-5，均值相对差；max 括号内）: JSD_β/β vs FKL = 4.28e-04 (8.17e-04)；JSD_β/(1−β) vs RKL = 5.29e-03 (1.29e-02)
      有界性: 对称 JSD ∈ [0.545, 0.682]（ln 2 = 0.693）
    容量边缘学生: 3,318 参（mode-seeking/covering 对比只在容量压力下出现）
    熵门控 λ 剖面（教师条件熵/ln V）: 位置0 = 0.168，其余位均值 = 0.003（门控集中在模选择位）
    recipe   valid_A  valid_B   total  frac_A  lock      背书 maxTokLoss
    fwd        0.057    0.023   0.080   0.707  0.41  -2.144      8.962
    mix25      0.080    0.039   0.119   0.672  0.34  -2.023      8.284
    jsd        0.648    0.000   0.648   1.000  1.00  -0.584      0.691
    mix75      0.252    0.010   0.262   0.963  0.93  -1.413      8.879
    rev        0.824    0.000   0.824   1.000  1.00  -0.460     10.314
    ada        0.717    0.002   0.719   0.997  0.99  -0.677     12.462

[3] 轴2 信号源（白盒 / 黑盒裸 REINFORCE / 黑盒+group 采样，各 600 步）
    恒等式探针 ∇KL(q‖p) = E_{y~q}[(log q − log p)·∇log q]（固定学生）
      白盒解析 vs 全词表枚举: 相对差 = 2.21e-07（机器精度）
      黑盒 MC(N=1024) vs 白盒: 余弦 = 0.9509，相对范数差 = 0.330
    梯度方差探针（M=16 独立抽取，每位置 1 样本 = 训练形态）
      黑盒 RMS 相对误差 = 6.159（白盒 = 0，解析量）
    route             valid_A  valid_B   total  lock      背书     KL均值     KL标准差
    whitebox            0.520    0.428   0.947  0.10  -0.139    1.289    1.6466
    blackbox-naive      0.000    0.000   0.000  1.00  -8.348    5.203    0.2258
    blackbox-group16    0.088    0.564   0.652  0.73  -0.604    3.181    1.4888
    whitebox 采样: IOFXGQDZ<end>  pjoioisk<end>
    blackbox-naive 采样: o3yaorIL<end>  yazJmJeU<end>
    blackbox-group16 采样: rmyaXGFX<end>  veoioisk<end>

[4] 轴3 token 加权（rev-KL 循环；300 步快照 + 600 步终态）
    weight      300步字母位  300步<end>   300步位loss离散   终态total   终态lock
    uniform       0.855      1.000        1.1095     0.947     0.10
    gap           0.947      1.000        0.0802     0.934     0.20
    norm          0.588      1.000        2.9636     0.857     0.11
    norm_div      0.557      1.000        3.4305     0.000     1.00

[5] 轴4 效率与稳定化（rev-KL 白盒，总梯度步 600）
    config  valid_A  valid_B   total  lock      陈旧度
    K1        0.520    0.428   0.947  0.10   0.0000
    K2        0.516    0.426   0.941  0.10   0.0118
    K4        0.480    0.465   0.945  0.02   0.0244
    K8        0.500    0.445   0.945  0.06   0.0519
    K4+IS     0.498    0.445   0.943  0.06   0.0241
    陈旧度损害探针（容量边缘学生 3,318 参，K1/K4/K4+IS/K8）
      K1   : total=0.859 lock=1.00 陈旧度=0.0000
      K4   : total=0.877 lock=1.00 陈旧度=0.0302
      K4+IS: total=0.812 lock=1.00 陈旧度=0.0308
      K8   : total=0.744 lock=1.00 陈旧度=0.0682
    稳定化：黑盒 group16 路线 advantage 裁剪 τ 谱系（各 600 步）
      config    终态total    loss标准差    被裁token占比
      noclip      0.652     56.171       0.0000
      clip2       0.266     12.922       0.4532
      clip5       0.584     38.503       0.3854

[6] self-check（断言设计见 tutorial §7）
✅ self-check passed: JSD族数学/恒等式/白盒黑盒(group修复+崩溃反例)/加权谱序与不动点/复用陈旧度(边缘学生损害)/IS/τ裁剪谱系/散度端点结局

digest(关键指标) = 19dd73cd65d209f6

takeaway: 同一个 on-policy 循环，四个旋钮各有一个生产答案：
          散度：生产主流拧在 reverse 端（Qwen3/DSv4/TM），JSD 有界但中庸；
          信号源：reverse KL 是唯一「黑盒可跑」的散度（恒等式，机器精度），
                  但裸 per-token REINFORCE 在本 toy 确定性崩溃（方差爆炸 +
                  OOD 自强化）——可行性 = 恒等式 + 方差代价；group 采样
                  （×16 教师打分）修复之，代价 = 同步数预算终态仍逊于白盒；
          加权：不改不动点（有界权重），只改路径——gap 把尺度差当信号最快，
                EMA 归一化族把尺度差当噪声、在本 toy 实测有害（无界形态崩溃）；
          效率：rollout 复用省采样但样本变陈旧（off-policy 偏差定理），
                容量冗余吸收损害、容量边缘学生使损害显形；IS 修偏差付方差；
                advantage 裁剪 τ 须与尺度匹配（τ=5 压波动不损终态，τ=2 崩坏）。
```

这段输出来自固定 seed 的 CPU 运行。发布门禁要求两个新建空 CWD 均 exit 0、
stderr 为空且 stdout 逐字节一致；digest `19dd73cd65d209f6` 是关键指标摘要，
不是代码或数据完整性签名。硬件、PyTorch 版本或数值内核变化时应重新验收。

---

## 2. 问题设定：L2 留下一个循环，L3 问四个旋钮

L2 的结论是：多教师冲突时，**加号写在哪**决定成败——loss 级/logit 级混合
塌缩、概率级混合瘫痪、输出路由自确认锁模，只有 context routing 活下来。
所有这些实验共用同一个底层循环：**学生自采样 → 教师打分 → token 级 loss 反传**。

L3 不再问「第二个教师怎么进来」，而是问：**这个循环本身在生产里被怎么调？**
OPD survey `[2604.00626]` 的自述 taxonomy 是三正交轴：feedback signal
（logit-based / outcome-based / self-play）× teacher access（white-box /
black-box / teacher-free）× loss granularity（token / sequence / hybrid），
外加统一 f-divergence 框架。本文件把它重组成**四个教学旋钮**（映射见 §8）：

1. **轴1 divergence 设计**——同一循环上换 token 级散度（fwd / mix / jsd / rev /
   熵门控 ada）。散度只换「批作业怎么改」，不换「谁来写作业」：统一目标里
   采样分布与散度选择解耦（survey §2.5 统一目标，alttext eq136：
   `L_OPD(θ)=E_{y~π_mix}[Σ_t D_f(p_T(·|x,y_<t), p_θ(·|x,y_<t))]`——π_mix 与
   D_f 各管各的）。
2. **轴2 信号源**——教师接口两种形态：白盒（全词表 logits）vs 黑盒（只对学生
   采样 token 打 logprob，API 形态）。本轴有本文件最深的两个结果：一个恒等式
   （黑盒可行的算术原因）和一个确定性崩溃（裸黑盒的方差代价，§4）。
3. **轴3 token 加权**——同一散度下哪些 token 算数（survey §4.1.2：「Orthogonal
   to the choice of divergence is the question of which tokens matter most」）。
4. **轴4 效率与稳定化**——OPD 比 SFT 贵的两处是学生采样与教师打分
   （survey §8 成本算例：70B→7B / 1B tokens，off-policy ≈300 GPU-hours vs
   on-policy 1,200–1,500 GPU-hours，4–5× overhead [survey 算例]）；复用、
   IS 修正、advantage 裁剪是三个稳定化旋钮。

任务与模型承 L1/L2：双语密码本（4 位数字 → 小写 A 或大写 B 都算对），双语
教师 110,530 参数（逐样本随机选模，两套都会），默认学生 4,914 参数
（教师/学生 ≈ 22x），轴1 与陈旧度探针用容量边缘学生 3,318 参数（d_model=12
恰在「装不下两套密码本」的边缘，L0 §8 对称破缺声明继承）。

---

## 3. 轴1 divergence 设计：六个配方，一条插值轴

### 3.1 β-JSD 族的三条数学性质（无需训练的探针）

轴1 的散度族以 β-JSD 为骨架：`JSD_β(p‖q) = β·KL(p‖m_β) + (1−β)·KL(q‖m_β)`，
`m_β = β·p + (1−β)·q`。β=0.5 即对称 JSD。探针在初始化学生、teacher-forced
gold 序列上验证三条性质（[2] 输出第一段）：

- **对称性**：β=0.5 时 `max|JSD(p‖q)−JSD(q‖p)| = 0.00e+00`——float32 下
  逐位相等，不是近似。
- **端点标度**：β=1e-5 时 `JSD_β/β ≈ KL(p‖q)`（fwd，均值相对差 4.28e-04），
  `JSD_β/(1−β) ≈ KL(q‖p)`（rev，5.29e-03）——**插值轴的两个端点恰是
  fwd/rev KL**。这是「mix 轴与 jsd 轴是同族」的数学根据：一阶展开下
  JSD_β 的端点行为就是两个 KL。
- **有界性**：对称 JSD ∈ [0.545, 0.682] ⊂ [0, ln 2 = 0.693]。survey §2.4
  选 JSD 的理由原文：「bounded…stable gradient field…preventing extreme
  gradient explosions」。

（该族与 GKD `[2306.13649]` 的广义 JSD 同族；本页不复刻 GKD 的 β 参数化，
只对这里明确定义的公式做数值自证。）

### 3.2 容量边缘学生上的六配方结局

六配方在同一 on-policy 循环、同一容量边缘学生上各跑 500 步（[2] 输出表）：
fwd 两模都留质量（lock 0.41，total 仅 0.080——覆盖但装不下）、mix25/mix75
沿插值轴连续过渡、jsd 与 rev 都锁一模（lock 1.00；rev 锁 A 域 0.824，
jsd 亦 A 域 0.648）、ada（熵门控）软化锁模：lock 0.99、A 域 0.717，B 域
还留着 0.002 的质量——用 0.105 的终态 total 换「不彻底选边」。软化幅度
不大，因为教师的高熵位只有位置 0（§3.3 的 λ 剖面预言了这一点）。
self-check (a)–(d) 把这四个结局类别钉成断言（§7）。

**maxTokLoss 列是 JSD 有界性的训练态对照**：jsd 的逐 token loss 最大 0.691
（≤ ln 2 = 0.693，断言 (c)），而 rev 达 10.314、ada 达 12.462——off-manifold
位上 reverse KL 无界，JSD 把它钳在 ln 2。这是「bounded gradient field」的
逐 token 实例化。

### 3.3 熵门控：λ 剖面预言了门控往哪拧

ada 配方（survey §4.1.1 entropy-gated mixture 机制类别）：教师熵低 → reverse
精确模仿；教师熵高 → forward 覆盖全部合理选项。λ_t = H_t/ln V。探针实测
（[2] 输出）：位置0（模选择位，双模边缘 ½/½）λ = 0.168，其余位均值 0.003
——门控**只**在模选择位生效，其余位近纯 reverse。survey §4.1.1 报告
Qwen3-4B-Base 上 entropy-gated 混合较 reverse 基线 +5.05 Pass@8
[survey 声称]；本 toy 的 λ 剖面解释了该机制的作用位置：只在教师自己
不确定的位置引入 forward 分量。

---

## 4. 轴2 信号源：一个恒等式、一个崩溃、一个修复

这是 L3 的核心轴，三个结果按因果链排列。

### 4.1 恒等式：黑盒为什么可能

reverse KL 对学生参数求梯度（采样前缀视为常数，承 L1/L2/GKD 约定）：

```
∇_θ Σ_t KL(q_t ‖ p_t) = Σ_t E_{y~q_t} [ (log q_t(y) − log p_t(y)) · ∇_θ log q_t(y) ]
```

右端只需要**采样 token 的 log q 与 log p**——不需要教师的全词表分布。
`probe_gradient_identity` 用三种算法对照验证（[3] 输出第一段）：

- `g_exact`：白盒解析梯度（autograd，全词表 KL）；
- `g_enum`：恒等式右端对全词表 V=66 精确枚举（toy 特权）——与 g_exact
  相对差 **2.21e-07，机器精度**，证明恒等式本身成立；
- `g_mc`：黑盒 MC——每位置 1024 个采样 token、只经 `teacher_logprobs_api`
  （同形于 Thinking Machines 的 compute_logprobs：输入学生采样的序列，
  只返回逐 token logprob）——与白盒余弦 **0.9509**，无偏但噪。

这正是 TM 生产配方可行的算术原因。TM 博客（Kevin Lu，2025-10-27）自述
（2026-08-15 重抓 82,782 B，引文逐字）：「We use a sampling client, as we
do not need to propagate logprobs through the teacher model」；「query the
teacher client with compute_logprobs…returns the teacher's logprobs on the
tokens x sampled by the student」；「We set the per-token advantage to the
negative reverse KL, and call the RL importance-sampling loss function」；
「we do not consider logit (top-k) distillation in any of our experiments」。
——per-token advantage = 负 reverse KL 被积函数 `c_t = log q − log p`，
教师只做打分客户端。本文件的黑盒路线就是这句话的最小实现。

### 4.2 崩溃：裸 per-token REINFORCE 在 toy 尺度确定性失败（反例教材）

恒等式说「可行」，没说「便宜」。把黑盒路线按最朴素的方式跑——每步每位置
1 个采样 token（= 训练形态），600 步，结果（[3] 输出 blackbox-naive 行）：

```
blackbox-naive      0.000    0.000   0.000  1.00  -8.348    5.203    0.2258
blackbox-naive 采样: o3yaorIL<end>  yazJmJeU<end>
```

**total = 0.000，采样退化成 OOD 乱码，KL 全程平台（尾部均值 5.203，600 步
不降），教师背书 −8.348（完全不认账）。** 五个边界探针
（clip τ=2 / 1200 步 / lr 5e-4 / 全局均值基线 / 幅度归一）
全部崩溃（valid_total 全 0.000）；同目标函数的白盒路线同 seed 同预算
达 0.947——**目标函数可学，崩的是估计器**。

机制深挖（三个环节自增强）：

1. **方差爆炸**：梯度方差探针实测 `bb_rms_rel = 6.159`——单次 REINFORCE
   抽取的梯度 RMS 误差是真实梯度范数的 **6 倍**（白盒 = 0，解析量）。
   噪声是信号的 6 倍时，每步方向近似随机。
2. **Adam 把噪声变成符号步**：Adam 用二阶矩归一化梯度尺度——对 6 倍噪声，
   归一化后每参数仍是满步长更新，噪声被「整形」成近似随机符号步，随机游走
   把学生推出教师流形。
3. **OOD 前缀自强化（不可逆环节）**：on-policy 的前缀由学生自己采样。学生
   一旦漂出域（采样出 `o3ya…` 这类不在任何码本里的序列），教师在这些前缀
   上的信号失去引导结构——KL 平台（5.203）就是「教师怎么说学生都听不懂」
   的度量。越漂越远，回不来。

**教材化声明**：本崩溃被刻意保留为主流程的一部分（断言 `naive total < 0.1`
+ `KL 平台 > 4.0` + `背书 < −5.0`，固定 seed 确定性复现）。它回答了一个
只看 TM 博客不会问的问题：**既然恒等式保证无偏，为什么生产黑盒 OPD 必须
配 RL infra？** 答案在 4.3。

### 4.3 修复：group 采样——方差代价的显式计价

修复方向来自代码内证据：g_mc 探针把每位置样本数从 1 提到 1024，余弦即达
0.9509——**多样本累积可把方差降到可用**。训练循环里的对应实现 = group
采样（GRPO group-sampling 机制类别）：每 prompt 采 16 条响应、loss 对全部
序列平均（`group_m=16`）。每位置样本 ×16 → REINFORCE 方差 ~1/16。结果
（[3] 输出 blackbox-group16 行）：

```
blackbox-group16    0.088    0.564   0.652  0.73  -0.604    3.181    1.4888
blackbox-group16 采样: rmyaXGFX<end>  veoioisk<end>
```

**学会了**（total 0.652 > 0.4，锁 B 模 lock 0.73，KL 从平台变成真实下降：
3.181 < naive 5.203），采样回到码本流形附近（`rmyaXGFX` 只差一个字母、
`veoioisk` 差两位——不完美但在学）。

**方差代价现在可以计价了**，它不是抽象的：

| 代价项 | 白盒 | 黑盒 naive | 黑盒 group16 |
|--------|------|-----------|--------------|
| 每步教师前向 | 1 × batch | 1 × batch | **16 × batch** |
| 教师透传量 | 全词表 logits | 采样 token logprob | 采样 token logprob |
| 600 步终态 total | 0.947 | 0.000（崩溃） | 0.652 |
| 轨迹 KL std | 1.6466 | 0.2258 | 1.4888 |

同步数预算下 group16 仍逊于白盒（0.652 vs 0.947）——**黑盒省的是 logits
透传（API 形态的硬约束），付的是采样次数与终态质量**。这就是 DSv4 报告
§5.1.2 自述取舍的 toy 实例化（下述为短摘，公式与结论以原报告为准）：
token-level advantage 路线「resource-efficient…leads to high variance in
gradient estimation and often causes training instability. Therefore, we
adopt full-vocabulary logit distillation…more stable gradient estimates」。
——DSv4 选白盒，TM 选黑盒 + RL infra（大 batch rollout + advantage 工程），
两条路都在付 4.2 的方差账单，只是付法不同。

### 4.4 错误代理附赠课：轨迹 KL 波动 ≠ 估计器方差

看 [3] 表的最后一列会发现反直觉的事：**轨迹最平稳的（kl_std 0.2258）恰是
崩溃路线**。轨迹 KL std 度量「动没动」，不度量「噪不噪」——崩溃路线原地
打转，当然「稳」。DSv4「high variance」声称的直接探针是 `bb_rms_rel`
（梯度估计器方差，6.159 ≫ 0.05 断言），不是轨迹 std。选错探针会把
「崩溃」读成「最稳的训练」——本文件把这个教训钉成断言
（`naive.kl_std < group16.kl_std`，§7）。

---

## 5. 轴3 token 加权：不动点不变，路径分裂，还有一个无界陷阱

### 5.1 理论：加权为什么不改变目标

token 级加权 loss `Σ_t w_t·ℓ_t`（w_t > 0 且**有界**、sg  detached）：
q = p 时 ℓ_t = 0，加权后仍是 0——**不动点不变**。加权改变的是路径：
哪些位置的梯度预算更多。这是 [4] 表终态列「同量级」的理论根据。

### 5.2 实测谱序：把尺度差当信号 vs 当噪声

四个配方同一循环各 600 步（[4] 输出）：

| weight | 300步字母位 | 300步位loss离散 | 终态total | 机制类别 |
|--------|------------|-----------------|-----------|----------|
| uniform | 0.855 | 1.1095 | 0.947 | 基线 |
| gap | **0.947** | **0.0802** | 0.934 | AdaKD（survey §4.1.2）：w ∝ sg[ℓ]，把尺度差当**信号** |
| norm | 0.588 | 2.9636 | 0.857 | EMA 尺度相对加权（DistiLLM adaptive loss 机制类别的 nano 构造）：w = sg[ℓ/EMA]，有界 |
| norm_div | 0.557 | 3.4305 | **0.000** | 同族真归一化形态 ℓ/EMA：有效步长 ∝ 1/EMA，**无界** |

谱序 `gap.spread300 (0.0802) < uniform (1.1095) < norm (2.9636)` 是实测派生
断言（固定 seed），它把两类加权哲学分开了：

- **gap 把尺度差当信号**：loss 高的位置 = 还没学会的位置，给它更多预算
  → 落后位被追平（300 步字母位 0.947 全场最快）、逐位离散塌到 0.0802。
- **norm 把尺度差当噪声**：按运行尺度相对加权，本意是「不让高尺度位置
  主导」。但本 toy 的封闭码本上，位置间尺度差**恰好就是**「学会没学会」
  的信息本身——把它归一化掉等于扔掉信号：字母位收敛最慢（0.588）、
  离散反而最大（2.9636）。终态 0.857 与均匀同量级（差 0.090 ≤ 0.12，
  不动点断言仍过）——**有界权重守住了不动点，但路径明显变差**。

**生产语境声明（反幻觉）**：此处只引用 DistiLLM `[2402.03898]`
adaptive loss 的**机制类别**
（按运行尺度重标定 token loss），norm/norm_div 的精确参数化是 nano 构造、
非原文公式复现。DistiLLM 面向真实 LLM 蒸馏（长序列、混合域、极端尺度
token），那里尺度差更多是标定伪影而非学习信号——**同一机制类别在不同
regime 下符号相反**，这正是本轴要教的判断力。

### 5.3 norm_div：无界有效步长如何违反不动点定理的前提

norm_div（ℓ/EMA 除法）是「真归一化」：梯度 = ∇ℓ_t/EMA_t，各位置梯度尺度
被拉平——听起来正是 equalization。实测却 total = 0.000 崩溃（[4] 表末行）。
机制：EMA 跟踪损失有 ~10 步滞后（0.9/0.1）；当某位置损失快速下降，
1/EMA 相对放大——**有效学习率 ∝ 1/EMA_t 在 EMA 衰减处无界**，近收敛位置
被过步长振荡破坏，学生滑出流形。「w_t > 0 ⇒ 不动点不变」的定理隐含
**有界**前提，norm_div 违反它。生产实现里这类归一化总带 clamp/下界，
原因即此。

---

## 6. 轴4 效率与稳定化：陈旧度、IS、与 τ 的尺度匹配

### 6.1 rollout 复用：陈旧度单调，但损害需要容量压力才显形

采一次、K 步梯度（reuse_k=K）。复用即 off-policy：第 2..K 步的样本来自
旧策略 q_gen，而 loss 按当前 q_now 计算。陈旧度 = `E|log q_now − log q_gen|`
实测单调（[5] 输出：K2 0.0118 / K4 0.0244 / K8 0.0519，断言过）——这是
L0 off-policy 偏差定理的序列空间版。

但默认学生（4,914 参）上 K1 vs K8 终态差仅 0.002（0.947 vs 0.945）——
**损害低于可测阈**：容量冗余的学生有足够「缓冲」吸收陈旧样本的偏差。
换容量边缘学生（3,318 参，轴1 同款），损害显形：

```
K1: total=0.859   K8: total=0.744   →  差 0.115（断言 > 0.05）
```

**陈旧度损害是「容量 × 陈旧度」的交互项**：模型越勉强，复用越伤。
生产里 partial rollout / 复用是常态（Kimi K3 报告称其通过 per-token
regularization 容忍极端 off-policy，详见同轨 deep-dive）——本探针解释了为什么它们必须配
正则化/IS 修正。

### 6.2 clipped IS：修偏差，付方差

K4+IS（clipped 重要性比 `min(q_now/q_gen, 2)`）在默认学生上 0.943 ≥
K4 0.945 − 0.01（断言过，陈旧度 0.0241）——IS 至少不劣。但注意它
没有「修复」到 K1 水平：clip 截断了比的尾部，偏差只修了一部分，而乘进
loss 的 ratio 本身带来方差。**IS 是偏差-方差交换，不是免费午餐**——
容量边缘学生上这个交换实测为负收益（[5] 陈旧度损害探针：K4+IS 0.812 <
K4 0.877，实测派生断言）：容量越勉强，ratio 噪声的伤害越大。生产里
IS/partial rollout 总是与正则化、advantage 工程成对出现（§6.1 Kimi K3
引文），原因即此。

### 6.3 advantage 裁剪 τ：阈值必须与 advantage 尺度匹配

Kimi K3 报告描述了用于限制极端 advantage 的 clipping threshold；这里不复刻
未披露的生产阈值，只在 toy 中扫描 τ。
在黑盒 group16 路线上跑 τ 谱系（[5] 输出）：

| config | 终态total | loss标准差 | 被裁token占比 |
|--------|-----------|-----------|---------------|
| noclip | 0.652 | 56.171 | 0.0000 |
| clip τ=2 | **0.266** | 12.922 | **0.4532** |
| clip τ=5 | 0.584 | 38.503 | 0.3854 |

- **τ=5 是「压方差不损终态」**：loss_std 56.171 → 38.503（−31%），终态
  0.584 vs 0.652（差 0.068 ≤ 0.10，断言过）。
- **τ=2 是阈值失配的反例**：被裁 token 占 45.3%——剪掉的不是「极端尾部」
  而是**信号主体**（off-manifold token 的 c = log q − log p 天然在 5–10
  量级，那正是「把错误 token 压下去」的力），终态崩到 0.266（断言
  < noclip − 0.2）。

**τ 不是一个可以照抄的超参，它必须对照 advantage 的经验分布定**——
看被裁占比，而不是只看 loss 曲线。

---

## 7. self-check 断言设计史：每条断言一个锚

| # | 断言 | 锚（理论/实测派生） |
|---|------|--------------------|
| 1 | 教师 total > 0.9、两模均衡 | 构造：双语教师两套码本都学（[1] 0.986/0.539） |
| 2 | JSD 对称/端点标度/有界 | 数学事实：β-JSD 族定义（§3.1，机器精度） |
| 3 | 恒等式 rel_enum < 1e-4 | 数学事实：∇KL 的 score-function 形式（§4.1，2.21e-07） |
| 4 | cos_mc > 0.95、bb_rms_rel > 0.05 | 无偏性 + 方差存在性（§4.1/§4.2，0.9509/6.159） |
| 5 | whitebox total > 0.4 | 动力学结局：白盒可学（0.947） |
| 6 | naive total < 0.1、KL平台 > 4.0、背书 < −5.0 | **实测派生**：确定性崩溃三签名（§4.2，0.000/5.203/−8.348） |
| 7 | group16 total > 0.4、KL 下降 | 实测派生：group 修复可学（§4.3，0.652） |
| 8 | naive.kl_std < group16.kl_std | 实测派生：错误代理反例（§4.4，0.2258 < 1.4888） |
| 9 | gap 不慢于 uniform（−0.02 容差） | 预算集中机制（§5.2，0.947 vs 0.855） |
| 10 | norm 慢于 uniform；谱序 gap < uniform < norm | 实测派生：尺度差=信号（§5.2） |
| 11 | norm_div total < 0.1、离散扩大 | 实测派生：无界有效步长（§5.3，0.000/3.4305） |
| 12 | gap/norm 终态 ±0.12 同量级 | 理论：有界权重不动点不变（§5.1） |
| 13 | 陈旧度 K8 > K2 > 0 单调 | off-policy 程度定义（§6.1） |
| 14 | 默认学生 K1/K8 差 ≤ 0.05；边缘学生差 > 0.05 | 实测派生：容量×陈旧度交互（§6.1，0.002 vs 0.115） |
| 15 | K4+IS ≥ K4 − 0.01 | IS 至少不劣（§6.2，0.943 vs 0.945） |
| 16 | clip5 loss_std 下降、终态 ±0.10；clip2 崩坏 | 实测派生：τ 尺度匹配（§6.3） |
| 17 | self-check (a)–(d)：rev 锁模/fwd 覆盖/插值连续/JSD 有界 vs rev 无界/熵门控软化 | L0 算术定理的序列空间版（承 L0 §6，绿色项未动） |

**实测派生断言口径声明**：动力学结局类断言在固定 seed 确定性复现的前提下
锚定该实现的结局类别（崩溃/学会/损害可测/谱序），阈值留有余量
（如崩溃签名 total < 0.1 对实测 0.000），不是对单个浮点值的拟合。

---

## 8. 与权威实现和生产配方的对应

### 8.1 四轴 → survey taxonomy 映射（教学重组的闭合依据）

survey `[2604.00626]` v4 自述 taxonomy = 三正交轴（feedback signal ×
teacher access × loss granularity）+ 统一 f-divergence 框架。本文件四轴
是其教学重组；映射以当前版本的一手论文为准：

| 本文件轴 | survey 落点 | 关键主张（教学转述） |
|----------|-------------|--------------------------------------|
| 轴1 divergence 设计 | §2.4 f-divergence 族 + §2.5 统一目标 + §4.1.1 adaptive divergence | 「bounded…stable gradient field…preventing extreme gradient explosions」（JSD）；「GKD…empirically testing Forward KL, Reverse KL, and JSD. Setting [λ=0] makes GKD purely on-policy」 |
| 轴2 信号源 | §3.2 Teacher Access + §5.1 + §4.1.4 邻域 G-OPD 视角 | 「Teacher-internal-state availability dictates the allowable mathematical formulations」；「the teacher serves exclusively as a scoring function or preference ranker over the student's on-policy trajectories」 |
| 轴3 token 加权 | §4.1.2 Token Weighting and Selection + §2.5 DistiLLM adaptive loss | 「Orthogonal to the choice of divergence is the question of which tokens matter most」 |
| 轴4 效率与稳定化 | §4.3.1 Compute Efficiency + §7.2 + §8 成本算例 + DSv4 §5.1.2 + Kimi K3 τ-clip | 「The most direct bottleneck of OPD is the cost of generating full student rollouts at every training step」；「The primary systemic bottleneck of OPD is the teacher forward pass on dynamically generated student tokens」 |

### 8.2 生产配方的四个数据点（全部在盘引文，逐字可 grep）

| 生产方 | 旋钮选择 | 逐字引文（来源与抓取日见 §13） |
|--------|----------|-------------------------------|
| **Qwen3** `[2505.09388]` | OPD 进生产栈（轴全开） | survey 工业表行：「Qwen3 (64) \| Qwen3-32B / 235B-A22B \| Qwen3-0.6B–14B, 30B-A3B \| On-Policy Distillation \| AIME, MATH, LiveCode」；TM 博客转引 Table 21：RL 67.6% @17,920 GPU-hours vs OPD 74.4% @1,800 GPU-hours——「reaching a higher score of 74.4 on AIME'24 at one-tenth the cost of RL」`[blog claims]` |
| **MiMo-V2-Flash** `[2601.02780]` | 多教师 OPD（轴2 白盒 + 轴3 域路由，L2 主题的生产形态） | 摘要自述：「introduces a novel Multi-Teacher On-Policy Distillation (MOPD) paradigm…domain-specialized teachers (e.g., trained via large-scale reinforcement learning) provide dense and token-level reward」 |
| **Thinking Machines**（blog，2025-10-27） | **黑盒** + per-token advantage + RL IS loss（轴2 黑盒路线的生产形态） | 见 §4.1 四句引文；算力自述「9-30×」便宜、「a baseline cost reduction of 9x when the SFT dataset is given」`[blog claims]` |
| **DeepSeek-V4** `[2606.19348]` | **白盒** full-vocabulary + loss 级多教师加权（轴2 白盒路线的生产形态） | 「the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)」；目标 eq29 `L_OPD(θ) = Σ_i w_i · D_KL(π_θ ‖ π_Ei)`；「more than ten teacher models covering various domains…distill a single student model」；白盒取舍自述见 §4.3 |

**两条生产路线在轴2 上分叉**：TM 选黑盒（教师只做 compute_logprobs 客户端，
配 RL infra 付方差账单），DSv4 选白盒（full-vocabulary logits 工程，配
ZeRO-like 教师权重分片 + last-layer hidden 缓存 + TileLang exact-KL kernel
等 infra 付透传账单，见 DSv4 §5.2.2）。本文件 §4 的三路线表就是这次分叉的
toy 缩影：没有「黑盒还是白盒」的抽象优劣，只有方差账单怎么付。

### 8.3 nano 侧未做（诚实清单）

- 黑盒路线的**序列级** IS loss（TM 的 RL IS loss 是序列级重要性比；本文件
  黑盒是纯 REINFORCE，IS 只在轴4 白盒复用路径上演示）。
- advantage 的 group 内中心化/标准化（GRPO 的 advantage 工程；本文件 group
  只做平均——token 级常数基线的无偏性见 §10 思考题 4）。
- 真实模型规模、真实 rollout infra（异步采样/教师打分流水线）、多教师
  生产形态（L2 只到机制）。
- DistiLLM adaptive loss / GKD 广义 JSD 的**原文公式级**复刻；本页只覆盖
  自己明确定义并可执行的机制类别。

---

## 9. 费曼：能不能讲给外行听

**类比：一个函授班，四个旋钮。** 学生自己写作业（on-policy 采样），老师
批改（教师打分）。四个旋钮：

- **批改标准**（轴1）：按「覆盖所有写法」批（forward）学生样样通样样松；
  按「揪住你写得最像的」批（reverse）学生会锁死一种写法；折中（JSD/混合）
  在两者之间连续可调。
- **老师能看到什么**（轴2）：老师能看到你全篇草稿（白盒 logits）就能精确
  指出每处怎么改；老师只能看到你交上来的成稿、逐字打分（黑盒 logprob）——
  **恒等式说这足够教会你（无偏）**，但每次只凭一份卷子打分，噪声是信号的
  6 倍（bb_rms_rel 6.159），裸跑必然教崩（naive 0.000）。办法：每道题让你
  写 16 份再平均（group 采样）——教得会（0.652），但老师批改量 ×16，而且
  同样课时下还是不如能看草稿的老师（0.947）。**黑盒不是免费，是把账单从
  「透传草稿」换成了「多写多批」。**
- **哪些字算数**（轴3）：错得多的字多批（gap）最快；把「错得多」当成
  噪声去归一化（norm 族）反而扔掉信息——在本 toy 里尺度差就是「会没会」。
- **省力气与防翻车**（轴4）：一份卷子批 K 遍（复用）省纸但卷子越批越旧
  （陈旧度单调），学生脑子越不够用越吃亏（容量边缘损害 0.115）；给批改
  力度装限位器（τ-clip）——限位器拧太紧（τ=2）把正常批改也限没了
  （45.3% 被裁、终态 0.266），拧合适（τ=5）只压住过激批语。

**一句话版**：同一个「学生写、老师批」的循环，生产配方的全部分歧就是：
批改标准拧在 reverse 端、老师看不看草稿（看 = 白盒稳，不看 = 黑盒省但要
多写多批）、把错误分布当信号还是噪声、以及省力气时愿意付多少陈旧度。

**自检问**：
1. 不看草稿的老师（黑盒）凭什么「足够」？（恒等式，§4.1——梯度只需采样
   token 的 logprob。）
2. 为什么「每道题写 16 份」能救裸黑盒，而调学习率/加裁剪救不了？
   （方差是 1/N 量级的统计量，调参改不了信噪比的结构，§4.2/§4.3。）
3. 轨迹「最稳」的路线为什么恰是最差的？（kl_std 度量动没动，不度量噪不噪，§4.4。）

**类比边界**：真实老师批改的不止「对/错」，还有风格/安全/格式（多维
reward）；真实学生的「写 16 份」是并行 rollout infra 问题（nano-slime /
nano-vllm-sglang 主题），本 toy 里只是顺序多跑；真实黑盒 API 的 logprob
还可能有数值精度/截断问题（本 toy 是精确 float32）。

---

## 10. 思考题

1. **方差的计价**：group16 把黑盒从 0.000 救到 0.652，代价是教师打分 ×16。
   若教师前向成本是学生前向的 100 倍（真实规模），白盒（1 次教师全词表）
   vs 黑盒 group16（16 次 logprob 打分）的成本比怎么算？DSv4 的
   last-layer hidden 缓存 + TileLang exact-KL（§8.2）在改变哪个因子？
   （提示：logprob 打分仍需教师全前向，省的只是透传与存储。）
2. **谱序的 regime 依赖**：本 toy 上 gap < uniform < norm（离散谱序），
   因为尺度差 = 学习信号。构造一个尺度差 = 标定伪影的场景（提示：不同
   位置的词表大小不同，或混合长短序列使 loss 基线不同），预测谱序如何
   反转，并设计最小验证。
3. **IS 的负收益**：容量边缘学生上 K4+IS (0.812) < K4 (0.877)（§6.2，
   [5] 探针实测）。用偏差-方差分解解释；若给你「K4+IS 的 ratio 不 clip
   上界 2 而是 clip 到 1.2」，预测会发生什么并验证。
4. **基线的合法性**：REINFORCE 基线 b 要求「与采样 token 无关」才保无偏
   （E_{y~q}[∇log q] = 0）。group 内按位置中心化（每个 prompt 的 16 条
   响应在同一位置取均值作基线）为什么**不是**合法基线？（提示：组内成员
   前缀不同，位置 t 的条件分布不同——合法基线只能是目前缀的函数。）
5. **τ 的自适应**：设计一个「按 advantage 经验分位数定 τ」的自适应裁剪
   （如永远只裁最大 5%），预测它在本 toy 的 clip_frac/终态表现，与固定
   τ=5 对比。Kimi K3 的「clipping threshold」是固定还是自适应，从引文
   能判断吗？（提示：引文只说 threshold，未说调度——不确定就标
   [TODO: verify]。）

---

## 11. 反例与边界

1. **toy 尺度诚实声明**：V=66 词表、4,914 参学生、600 步预算。全部数字
   是机制的实例化，不是任何真实模型的 benchmark；「6 倍噪声」「×16 采样」
   等量值随任务/规模变化，机制方向（恒等式无偏、方差 ~1/N、陈旧度单调、
   有界权重不动点）不随规模变化。
2. **toy 黑盒路线与 TM 生产形态的差距（诚实声明）**：本文件的黑盒修复 =
   group 平均（toy 里最干净的降方差实现）；TM 生产形态 = RL infra 上的
   序列级 IS loss + per-token advantage 工程 + 大规模并行 rollout
   （「We set the per-token advantage to the negative reverse KL, and call
   the RL importance-sampling loss function」）。两者同属「用多样本/多步
   工程付方差账单」的机制类别，但 toy 版没有序列级 IS、没有 advantage
   标准化、没有异步 infra。把 toy 数字外推到「黑盒 OPD 在生产需要 ×16
   采样」是错误的——本 toy 只证明**方差账单存在且必须有人付**。
3. **崩溃的边界**：裸 per-token REINFORCE 在本 toy 确定性崩溃（五个边界
   探针全崩），但崩溃不是数学必然——更大 batch、更小 lr、
   更短 horizon 会推迟崩溃（方差账单的连续谱），本 toy 只是把账单拉到了
   600 步内可见。反过来，group 采样也不是万能的：样本数阈值本身是任务
   方差的函数——边界探针中 group_m=8 仍崩（total 0.000），group_m=16 才过线
   （0.652）。**「需要多少样本」没有普适答案，只有针对任务方差的实测。**
4. **§八 OPD B 层当今定位声明**（对齐日期 2026-08-15，承 sota notes §4）：
   OPD 维持 B 层前沿主流——生产采用面 2025-05（Qwen3）→ 2025-10（TM 复现）
   → 2026-01（MiMo-V2-Flash）→ 2026-06（MOPD 论文化 / DSv4 主干替换 RL）
   持续加厚，未发现取代范式。经典锚点 MiniLLM / GKD / DistiLLM 仍为 A 层
   地基（survey §2.5 统一视角把三者收编为同一目标的不同参数化——机制仍是
   当前生产配方的参数选择空间，非过时内容）。2026 年 OPD 微创新论文变体
   爆炸，本节按 C 层纪律只教机制类别、以生产数据点锚定，不追单源方法名。
5. **真机验证**：多卡 rollout infra、真实教师 API 延迟/精度下的黑盒路线
   尚未由本 toy 验证，必须单独记录硬件、软件栈、成本与质量指标。

---

## 12. 阶梯预告

- nano-opd 阶梯至此 L0–L3 完整：L0 散度选择算术 → L1 真实序列模型 2×2
  因子 → L2 多教师融合/路由 → L3 生产配方四旋钮。
- 横向延伸：轴2 黑盒路线的采样吞吐问题 = [nano-slime](../nano-slime/) 与
  [nano-vllm-sglang](../../03-data-distributed-rsi/nano-vllm-sglang/) 的
  主题；轴4 的 IS 修正完整形态 = [nano-verl](../nano-verl/) L1；
  后训练算法演进的全景（PPO→GRPO/RLVR→OPD）见 01 轨 sota-deepdive。
- 本模块无 L4 规划：L3 已对齐权威/生产，再往上是真实 infra 工程
  （非 nano 形态）。

---

## 13. 溯源与口径声明

**一手来源快照（2026-08-31 复核；版本可能继续变化）**：

| arXiv ID | 标题（API 官方全称） | v / 日期 | 本节用途 |
|----------|----------------------|----------|---------|
| [2306.13649](https://arxiv.org/abs/2306.13649) | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | v3 / 2023-06-23（upd 2024-01-17） | GKD：on-policy 循环与广义散度族；本页不复刻其 β 约定 |
| [2402.03898](https://arxiv.org/abs/2402.03898) | DistiLLM: Towards Streamlined Distillation for Large Language Models | v2 / 2024-02-06（upd 2024-07-03） | adaptive loss 机制类别（§5.2） |
| [2604.00626](https://arxiv.org/abs/2604.00626) | A Survey of On-Policy Distillation for Large Language Models | v4 / 2026-04-01（upd 2026-06-18） | taxonomy 三轴、统一目标与成本讨论（§8.1） |
| [2505.09388](https://arxiv.org/abs/2505.09388) | Qwen3 Technical Report | v1 / 2025-05-14 | 生产数据点（§8.2）；具体配方按报告与 survey 分层引用 |
| [2601.02780](https://arxiv.org/abs/2601.02780) | MiMo-V2-Flash Technical Report | v2 / 2026-01-06（upd 2026-01-08） | 多教师 OPD 生产数据点（§8.2） |
| [2606.19348](https://arxiv.org/abs/2606.19348) | DeepSeek-V4 Technical Report | 2026 | 白盒 full-vocabulary 与 multi-teacher OPD 数据点（§4/§8） |
| [2607.24653](https://arxiv.org/abs/2607.24653) | Kimi K3: Open Frontier Intelligence | v2 / 2026-08-07 | partial rollout 与 per-token regularization 对照（§6） |

**Thinking Machines 博客**（Kevin Lu，2025-10-27，on-policy distillation）：
本文中的算力数字（17,920 vs 1,800 GPU-hours、「9-30×」、
「one-tenth the cost」）均标作 `[blog claims]`，不能替代论文或本课程实测。

**公式边界**：本文件的 β-JSD 族以代码中的定义为准，并只声称三个被机器
验证的性质（对称性、端点标度、有界性）；它不是 GKD 原文公式的复刻。

**复验方法**：在两个新建空 CWD 中运行上方命令，分别保存 stdout/stderr，
要求两个 exit code 都为 0、stderr 都为空且 `cmp stdout.1 stdout.2` 成功；再核对
`digest(关键指标) = 19dd73cd65d209f6` 与本页输出块。若 PyTorch/平台升级导致
浮点末位漂移，先审计 self-check 与机制结局，不要只追旧哈希。

**时效性边界**：这里的“生产主流”是 2026-08-31 的 dated snapshot；前沿配方
可能变化，复用课程前应重新检查 survey 版本、模型报告与官方实现。
