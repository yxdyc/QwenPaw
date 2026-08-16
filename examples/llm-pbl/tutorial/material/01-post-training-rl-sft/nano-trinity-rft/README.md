# nano-trinity-rft

> **抓的核心机制**：**统一数据流 + 配置驱动**——SFT 与 RL 走同一份样本记录、
> 同一个 Buffer、同一个 Trainer；只改 config 字段，训练配方即可切换。
> **对标权威实现**：[Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT)
> （arXiv:2505.17826；旧地址 `modelscope/Trinity-RFT` 已 301 重定向至此）
> **轨道**：[01 后训练/RL/SFT](../README.md) · **状态**：L0–L3 ✅

> **定位说明**：Trinity-RFT 是通用的统一 RFT 框架（Explorer/Trainer/Buffer
> 三组件 + 统一数据流，支持 PPO/GRPO/SFT/DPO/mix 等十余种 algorithm_type）。
> 本模块取两个最小机制：L0–L2 取**统一 SFT+RL 数据流与配置驱动切换**（含
> reward 信号来源），L3 取**配置系统本身**（schema / 注册表 / 三层合并 /
> stages——配置驱动为什么在生产尺度成立）；调度（对照 nano-verl）、异步与
> staleness（对照 nano-slime）不在本模块重复。

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | 玩具：Explorer/Trainer/Buffer 三组件 + 统一样本记录，config 切换 sft_only / rl_only / sft_then_rl / mix 四配方 | ✅ [L0_unified_sft_rl_loop.py](L0_unified_sft_rl_loop.py) · [tutorial_L0.md](tutorial_L0.md) |
| **L1** | 真实 0.8M char-GPT 上配置驱动跑通 SFT→RL 两阶段：RL 逐位填洞至全 1.0、checkpoint 续训与连续跑 20/20 轮逐位一致、四配方消融（含 mix 最快与 rl_only 中途反超两个真实现象）；附 RL 失败模式与实现陷阱（探索死亡 / KL 双刃 / IS 比率陷阱 / 回放干扰） | ✅ [L1_real_unified_sft_rl.py](L1_real_unified_sft_rl.py) · [tutorial_L1.md](tutorial_L1.md) |
| **L2** | RL 的信号来源：稀疏 rule reward 的 dead group 算术（p^G+(1-p)^G，实测 vs 解析）与学习后果（小 G 净破坏 / dynamic sampling 最省）、Bradley-Terry learned RM（注入偏置可探针）、Goodhart 三臂（proxy 涨 gold 掉 / KL 锚 gold / rule 对照）；对照 Trinity rewards 注册表 / std_threshold / RULER-rubric 示例 | ✅ [L2_reward_signals.py](L2_reward_signals.py) · [tutorial_L2.md](tutorial_L2.md) |
| **L3** | 配置即实验台：nano 版 schema + ALGORITHM_TYPE 注册表 + 三层优先级合并（用户>算法默认>全局兜底）+ check_config 修复/拦截 + stages 课程，全部对照 Trinity config.py/algorithm.py/config_validator.py 源码；并复现 DAPO 式 ablation ladder（no-KL / Clip-Higher / Dynamic Sampling / Overlong 各一格开关），实测 KL 锚 drift 4.6×、overlong 打破 dead 平局与塑形泄漏 | ✅ [L3_config_ablation.py](L3_config_ablation.py) · [tutorial_L3.md](tutorial_L3.md) |

## 核心要讲清的点

- 为什么把 SFT 和 RL 放进同一框架（共享数据流 / checkpoint / 配置）——L0 ✅
- 配置驱动如何支持 ablation（开关式切换训练配方，逐级对比）——L0 ✅
- SFT 的天花板由 teacher 数据覆盖画死，RL 探索才能越过——L0 ✅（反例 [4]）；
  L1 在真实梯度上复现同一算术（exact 0.500 / characc 0.625，与 L0 的 0.625 同构）✅
- 真实模型的三笔债：探索会死（SFT 坍缩策略 → ε-探索是显式机制）、共享权重会打架
  （回放干扰 / phase-aware lr）、checkpoint 必须带走权重+优化器+版本+RNG 才能逐位
  精确衔接——L1 ✅（tutorial §4/§6，全部实测）
- reward 的来源：rule-based vs model-based——L2 ✅（tutorial_L2 §5–§9：dead group
  算术实测吻合解析式 / RM 偏置可探针 / Goodhart 三臂 proxy 涨 gold 掉，全部实测）
- 配置驱动为什么在生产尺度也成立：schema + 注册表 + 三层合并 + check_config +
  stages——L3 ✅（tutorial_L3 §3/§4/§8：宏开关一行展开成配方 / DPO 强制 KL、
  SFT 拒绝 both / 课程 = 配置列表，全部实测）；ablation ladder 复现 DAPO 开关表，
  实测去 KL drift 4.6×、overlong 打破 dead 平局与塑形泄漏（tutorial_L3 §6）

## 费曼自检

- 能不能用一张配置说明「从全量 SFT 到叠加 RL」要改哪几个字段？
  （L0 答案：`sft_rounds` / `rl_rounds` / `mix` 三个字段，循环代码一字不动）
- 能不能说清「toy 里免费的探索，为什么真实模型里必须显式维护」？
  （L1 答案：SFT 把策略坍缩成错误吸引子 → 组内方差归零 → advantage 归零 →
  RL 停摆；ε-探索用机制保证方差。见 tutorial_L1 §6.3）
- 能不能一句话说清「RM 对准确率 0.84」和「被钻到 gold 0.167」为什么不矛盾？
  （L2 答案：准确率是平均-case 指标，钻空子发生在误差的分布上——RM 训完那天
  错误面就客观存在，等着被优化器找到。见 tutorial_L2 §7/§12）
- 能不能说清「用户设了 kl=none 的 DPO，resolve 后为什么是 k2、出处还标 user」？
  （L3 答案：check_config 原地改写 config 在先、三层合并在后——配置是契约，
  validator 跑完你手里的 config 已经不是原来那份。见 tutorial_L3 §4）

## 环境依赖

- **L0**：零外部依赖（Python 标准库 `math`/`random`），CPU 即跑，固定 seed 逐字节可复现。
- **L1**：仅 `torch`（CPU，~2 分钟，任意 CWD）；固定 seed 下指标行逐字节确定
  （mask `elapsed` 计时行后连跑多遍 md5 相同），任务为合成口径（显式声明）。
- **L2**：仅 `torch`（CPU，~1.5 分钟，任意 CWD，`-B`）；固定 seed 下指标行逐字节
  确定（mask `elapsed` 行后 md5 `0014cd66…`/102 行，双 CWD 逐字节一致；digest
  `5b3c872e…`）；任务为 L1 同源合成口径，标注者偏置为显式注入的模拟（显式声明）。
- **L3**：仅 `torch`（CPU，~2 分钟，任意 CWD，`-B`）；固定 seed 下指标行逐字节
  确定（mask `elapsed` 行后 md5 `7ee8aa09…`/101 行，双 CWD 逐字节一致；digest
  `666a7242…`）；任务为 L1/L2 同源合成口径 + 变长响应扩展（显式声明）。
  源码对照需克隆 Trinity-RFT 仓库（只读，tutorial_L3 §14 锚点表）。

## 权威实现与延伸

- 对标源码：`agentscope-ai/Trinity-RFT`——三组件定义见 README L21–25；
  SFT 作为 `algorithm_type: sft` 配置项见 README L121（`trinity/algorithm/policy_loss_fn/sft_loss.py`）；
  全生命周期数据管线见 README L102–105
- 仓库核验：2026-08-05 初测（L0）；2026-08-06 复测——README 30,381 bytes、
  sha256 `d513f140…b73982`，上述锚点**逐项零漂移**；arXiv:2505.17826 标题页存活吻合
  （细节见 tutorial_L1 §13）；2026-08-12 三次复测——README sha256 逐位零漂移，
  新增 reward 侧锚点（rewards 注册表 / math_rm_workflow / grpo_advantage std_threshold /
  RULER-rubric 示例）与 10 个 arXiv ID 全部现场重抓核验（细节见 tutorial_L2 §14）；
  2026-08-13 四次复测——现场克隆（HEAD `009850b1`，末 commit 2026-07-31），
  README sha256 逐位零漂移，新增配置系统锚点（config.py schema / ALGORITHM_TYPE
  24 项 / config_validator 三层合并 / dapo_math 开关表 / DAPO 管线算子）与
  3 个 arXiv ID 现场重抓核验（细节见 tutorial_L3 §14）
- 交叉阅读：[nano-verl](../nano-verl/)（actor/learner 调度）、[nano-slime](../nano-slime/)（buffer 解耦与 staleness）、[nano-llamafactory](../nano-llamafactory/)（SFT 数据侧机制）
