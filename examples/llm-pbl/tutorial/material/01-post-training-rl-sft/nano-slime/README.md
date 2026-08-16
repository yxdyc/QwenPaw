# nano-slime

> **抓的核心机制**：RL 后训练的**采样/训练解耦**——rollout 引擎与 trainer 各跑各的，
> 用 **data buffer** 连接、**版本化权重**同步，代价是样本带 **staleness**（off-policy 度）。
> 注意：slime 本身是完整的后训练框架（Megatron 训练 + SGLang rollout + Data Buffer
> + reward/verifier 数据通路），本模块只取其数据通路的骨架做阶梯。
> **对应真实系统**：[slime](https://github.com/THUDM/slime)（自述 "an LLM post-training framework for RL scaling"）
> **轨道**：[01 后训练/RL/SFT](../README.md) · **状态**：L0–L3 ✅

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | single-file 确定性离散事件模拟：lockstep vs 解耦（data buffer + 版本化权重），量化 buffer 容量与 staleness 的权衡（零依赖，CPU 即跑） | ✅ [L0_data_buffer_decouple.py](L0_data_buffer_decouple.py) · [tutorial_L0.md](tutorial_L0.md) |
| **L1** | 把 G/T 从模拟常数变实测：真实小模型上串行 generate N 条 rollout，验证「G 随 response 长度线性涨」，batching 对 G 的压缩 | ✅ [L1_real_gen_train_timing.py](L1_real_gen_train_timing.py) · [tutorial_L1.md](tutorial_L1.md)——0.8M char-GPT（slime README 真实语料现场训练）实测 G∝L（R²>0.995）/ batching 压缩 2.6x / 同批 G/T=2.3 / S≪T，实测值灌回 L0 模拟器得解耦 speedup 1.39x（生成主导区 buffer 买不到吞吐） |
| **L2** | 引擎代价模型（借 nano-vllm-sglang L0 iter_time）× 同步/异步双 regime：slime train.py/train_async.py 控制流逐行对照，量化 1-step 异步 2x 上界 + update_weights_interval = staleness 旋钮（可运行本质模拟；真机 `[TODO: verify on real system]`） | ✅ [L2_engine_cost_async_regimes.py](L2_engine_cost_async_regimes.py) · [tutorial_L2.md](tutorial_L2.md) |
| **L3** | 对照 slime 源码（@ 2fa9a442，2026-08-16 抓取）：data buffer 回收（partial rollout + 版本戳 staleness）× delta weight sync（diff→编码→压缩→校验→滚动基线，xor 对合 vs overwrite 幂等，delta⊥colocate 决策规则）（可运行本质模拟；真机 `[TODO: verify on real system]`） | ✅ [L3_buffer_delta_sync.py](L3_buffer_delta_sync.py) · [tutorial_L3.md](tutorial_L3.md) |

## 环境依赖

- L0：零外部依赖（纯标准库），CPU 即跑。
- L1：torch（CPU 单线程基线，threads=1；约 2.5 分钟，含 ~47s 探针模型预训练）。
  绝对毫秒数为 CPU 小模型口径，结构结论（线性/压缩/G≫T）可外推，绝对值不可。
- L2：零外部依赖（纯标准库），CPU 瞬时（<0.1s）。本级为可运行的本质模拟（本课程 L2 可运行性契约）：
  建模 slime 源码背书的双 regime 控制流 + 引擎代价模型；真实 SGLang/Megatron 验证
  `[TODO: verify on real system]` 走 GPU 通道。
- L3：零外部依赖（纯标准库），CPU 瞬时（<0.25s）。本级为可运行的本质模拟（本课程 L3 可运行性契约）：
  对照 slime 源码（THUDM/slime @ 2fa9a442，2026-08-16 codeload 抓取）逐行核验 buffer 回收与
  delta sync 机制；掩码输出锚 `1c85efaf…`/58 行、digest `482ddb8b…`（2 遍 × 2 新建空独立 CWD
  BYTE-IDENTICAL）。真实 SGLang `/pull_weights` + Megatron gather + 共享盘验证
  `[TODO: verify on real system]` 走 GPU 通道。

## 核心要讲清的点

- 为什么 RL 里 rollout 往往是大头（decode 串行、G 随 response 长度线性涨）
- 解耦把「每批 G+S+T」变成「max(较慢一方)」；稳态吞吐由较慢一方钉死
- staleness 是解耦的结构性代价；buffer 容量限其上界、吸收波动，但买不到吞吐
- off-policy 偏差的两条对策：算法侧 importance sampling（nano-verl L1）、系统侧限 staleness
- （L1）实测三件套：G∝L 的斜率、batching 压缩率、同批 G/T——生成主导 regime 里
  解耦只值 1.39x，吞吐的第一杠杆是 batching，第二杠杆是更快的引擎（L2）
- （L1）测量方法论：round-robin + 中位数摊漂移；断言落在结构性质上不落在噪声带上
  （B=1↔2 噪声带、threads×batch 耦合探针为实例）
- （L2）1-step 异步（train_async.py）每轮藏掉 min(G,T)：加速上界 2x、峰值在 G≈T；
  生成主导区（L1 实测 G/T=2.3）只值 ~1.4x——异步不是万灵药，它的价值随 regime 迁移
- （L2）update_weights_interval 把 max staleness 钉在 k（结构性上界）、稳态只赚 S 摊薄：
  是 staleness 旋钮不是吞吐旋钮；真吞吐靠把 S 做小（delta sync）或把引擎喂满（背压设计）

## 费曼自检

- 能不能用回转寿司讲清：为什么加长传送带不能让客人吃得更快、只能让寿司更旧？
- （L1）能不能用后厨与传菜员讲清：为什么多雇传菜员（解耦）不如换更大的锅（batching）？
- （L2）能不能用外卖站的两口锅讲清：为什么「多久换一次菜谱」（update_weights_interval）
  决定菜有多旧、却不决定出餐多快？换菜谱前为什么要等在路上的骑手全部回店？
- （L3）能不能用快递站讲清：为什么「在途件回仓」（partial rollout）买不到吞吐却值得开？
  为什么合同只寄「红线 diff 页」（delta sync）在改得太多时反而比寄全本贵？
  两家公司共用一个文件柜（colocate）时为什么红线账本是纯浪费？

## 权威实现与延伸

- 对标源码：slime `github.com/THUDM/slime`——README「Architecture Overview」：
  training (Megatron) 从 Data Buffer 读数据、训练后同步参数给 rollout；
  rollout (SGLang + router) 生成数据存入 Data Buffer；data buffer 为桥接模块。
- 概念延伸：采样引擎为什么快 → 轨道 03 [nano-vllm-sglang](../../03-data-distributed-rsi/nano-vllm-sglang/)；
  流水线重叠的另一种解法 → [nano-verl](../nano-verl/) L0；off-policy 修正 → nano-verl L1。
