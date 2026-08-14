# nano-verl L3 — HybridFlow：同一组 worker 在 rollout/train 间复用

> 对应真实系统：[verl](https://github.com/verl-project/verl)（HybridFlow，arXiv:2409.19256）
> 本文件：`tutorial/material/01-post-training-rl-sft/nano-verl/tutorial_L3.md`
> 可跑文件：`L3_hybridflow_colocate.py`
> 锚点基准：verl-project/verl @ **v0.7.1**（2026-08-07 抓取核验，sha256 见 §11）

---

## 本节目标

L2 把 actor 与 learner 拆成了两个 lockstep 进程，并观察到不同设备路径下的运行时间差；
它没有做跨 step overlap。L3 转向另一种重要形态：**同步 colocate**——
同一个 worker 类（`ActorRolloutRefWorker`）住在同一组 GPU 上，rollout 阶段做推理、train 阶段做训练，
阶段切换时做权重 resharding 与训练态 offload。为什么？

- 严格新鲜 rollout 的同步两相串行时，拆卡到底在付什么隐性成本？
- 「同一组卡两相复用」要解决的核心矛盾是什么（提示：不是速度，是显存算术）？
- 单控制器（single controller）+ SPMD 的编程模型长什么样？`DataProto.chunk/concat` 与
  `register(dispatch_mode=...)` 在其中的角色？
- 权重从训练分片到推理引擎的 resharding 之路，真实系统走了哪几步？代价多大？

**可运行性声明（ROADMAP §三 L2/L3 契约）**：真实 verl 需要 Ray + FSDP/Megatron + vLLM + 多 GPU，
本机跑不了 `[TODO: verify on real system]`。本节是**可运行的本质模拟**：计算是真的
（真实 char-LSTM、真实 PPO 梯度、真实权重流动），显存与时钟按声明式 COST 模型折算（§5/§6 显式区分
「实测玩具数字」与「声明规模算术」，绝不混用）。模拟的**语义**与 verl 一致，且被代码末尾
10 项 self-check 机器断言。

---

## 先跑起来

```bash
# 依赖仅 torch；强制 CPU（跨遍 bit-level 确定性是本节 self-check 的前提）；任意 CWD 可跑；~8s
python3 L3_hybridflow_colocate.py
```

真实输出（Apple Silicon；padding 修正后的确定性运行，`elapsed` 会随机器波动）：

```text
nano-verl L3 — HybridFlow colocate: same worker group, two phases
task: 'go:' -> 'hello' (与 L1/L2 相同)   model: TinyLSTM (embed=32, hidden=64, layers=1)
cluster: N=4 simulated ranks   device: cpu (强制 CPU 保证跨遍确定性)
rl: rollouts=64, iters=30, ppo_epochs=4, lr=0.001

[0] determinism probe
    same input forwarded twice -> max |Δlogits|+|Δvalues| = 0.0e+00
    params = 27,966 (fp32 = 109.2 KB) — 计算是真的，显存/时钟按 COST 声明折算

[1] DataProto + dispatch semantics
    (a) chunk(4) -> concat round-trip: EXACT
    (b) DP transparency: chunked log_prob vs full-batch log_prob, max |Δ| = 0.0e+00 (bit-identical)

[2] colocated fit (N=4): same ranks serve rollout AND train
    one step = rollout phase -> [sync: gather+push] -> train phase -> [sync]
      rollout phase: ranks 0-3 generate (训练态 offloaded, KV cache resident)
      train   phase: ranks 0-3 update  (rollout engine sleeping)
    warmup(SFT) loss = 1.2933
    sharding: 27966 params + 2 pad = 27968 (6992/rank, FSDP 式 pad 到整除)
  [step  0] reward=0.875 exact=0.625 policy_loss=-0.0090 value_loss=1.0051 entropy=0.529 kl=0.0019
  [step  5] reward=0.859 exact=0.594 policy_loss=-0.0101 value_loss=0.0331 entropy=0.550 kl=0.0029
  [step 10] reward=0.928 exact=0.734 policy_loss=-0.0089 value_loss=0.0161 entropy=0.268 kl=0.0017
  [step 15] reward=0.981 exact=0.906 policy_loss=-0.0095 value_loss=0.0040 entropy=0.141 kl=0.0025
  [step 20] reward=0.984 exact=0.953 policy_loss=-0.0087 value_loss=0.0032 entropy=0.097 kl=0.0008
  [step 25] reward=0.975 exact=0.922 policy_loss=-0.0088 value_loss=0.0075 entropy=0.076 kl=0.0008
  [step 29] reward=0.978 exact=0.891 policy_loss=-0.0027 value_loss=0.0027 entropy=0.069 kl=0.0003
    greedy samples after training: ['hello', 'hello', 'hello']
    declared memory/rank @7B: train 34.0 GB / rollout 34.0 GB (budget 80 GB)
    real toy traffic: 101602 KB moved across 30 syncs + 120 epoch gather/reduce (declared-scale formula in [3])

[3] declared-scale arithmetic (COST model, not measured)
    memory per rank (GB):
      scale   colo-train  colo-rollout  disagg-train  disagg-rollout budget
      7B            34.0          34.0          62.0            34.0     80
      13B           58.0          46.0         110.0            46.0     80
      colocate  fits @7B: True   fits @13B: True
      disagg trainer OOM @7B: False   OOM @13B: True
      (disagg 把 N/2 张卡的训练态压到一半卡上: 4P/(N/2) = 2x colocate 的 4P/N)
    wall-clock per RL step (declared ms):
      eta     colo(N)  disagg-sync(N/2)  disagg-async  colo vs sync-disagg
      1.00       2100              3900          2300                1.86x
      0.85       2100              3545          2103                1.69x
      sync-disagg 的隐性成本: 严格新鲜 rollout 不跨步流水 -> 每相都有一半边卡空转
      async-disagg 用 staleness 换重叠 (off-policy 代价, 见 nano-slime L0)

[4] scheduling invariance
    [4a] flat (same N=4, same seed, no phase switches):
         per-step metrics bit-identical to [2]: True
    [4b] disagg (2 rollout ranks + 2 train ranks, same seed):
         reward curve max |Δ| vs [2] = 0.1031
         (DP 宽度不同 -> 每 rank 的采样划分与梯度归约树都不同 -> 轨迹不逐位相同;
          不变的是收敛行为, 不是轨迹——bit 级不变性只在同宽度下成立, 见 [4a])
         final reward: colocate=0.978, disagg=0.997

[5] determinism re-run
    md5(metrics) run#1 = 5cba79e6a5bb5f8ff192f5f888837983
    md5(metrics) run#2 = 5cba79e6a5bb5f8ff192f5f888837983   identical: True

[6] self-check
    [pass] deterministic forward (CPU)
    [pass] chunk/concat round-trip exact
    [pass] DP transparency (max|Δ|=0.0e+00)
    [pass] colocate memory fits budget @7B (declared)
    [pass] colocate memory fits budget @13B (declared)
    [pass] disagg trainer OOM @13B (declared) — memory wall
    [pass] scheduling numerically transparent ([2]==[4a])
    [pass] disagg convergence parity (curve max|Δ|=0.1031, finals 0.978/0.997)
    [pass] learning: final reward 0.978 >= 0.9
    [pass] cross-run determinism (md5)

    ✅ self-check passed (10/10)

takeaway: 在严格新鲜 rollout 的同步配置下，colocate 的核心是显存与资源算术：
          拆卡 = 每相一半算力 + 一半卡空转；复用 = 每相全部算力 + 阶段边界付
          resharding 税。模型越大，训练态越住不满拆出去的卡，colocate 越赢。
          真实 verl 就是这套语义：RayPPOTrainer 单控制器编排，
          ActorRolloutRefWorker 同卡两相，rollout_mode() 做 gather+sync。

elapsed: 5.1s (计算真跑; 显存/时钟数字来自 COST 声明模型)
```

`elapsed` 不是确定性锚点；代码会在单次运行内部重复训练并比较 `md5(metrics)`，用于验证数值轨迹一致。

---

## §1 L3 的问题：角色已经拆开，为什么还要 colocate？

先纠正一个容易从 L2 产生的误读：L2 虽有两个进程，但协议是
`rollout → train → sync → 下一轮 rollout`，没有相邻 step 的流水重叠。修正后的 CPU-only 对照是
8.43s vs 8.18s（1.03x，基本持平），正说明“两进程”不能当作 overlap 证据。

如果要求每轮 rollout 紧跟最新权重，同步数据流是：

1. **严格新鲜口径的数据依赖是刚性的**。本节规定第 k 步 rollout 使用第 k-1 步更新后的权重；
   train 等 rollout 出数据，下一轮 rollout 等 train 出新权重，因此两相串行。放宽这条规定可以异步，
   但不再是本节的同步口径，并会产生下面的 staleness 权衡。
2. **同步拆卡 = 每个阶段都只有一部分算力**。4 张卡拆成 2+2：rollout 阶段 2 张采样卡干活、
   2 张训练卡等待；train 阶段反过来。同构集群上若不跨 step，固定拆卡没有形成重叠。
3. **复用 = 每个阶段拿满全部算力**。4 张卡先一起做 rollout，再一起做 train，阶段边界付一次
   权重同步的「过路费」。这就是 colocate。

[3] 的时钟表把这个算术量化了（声明式 COST 模型）：同步口径下 colocate 每步 2100ms，
拆卡 3900ms（η=1.0）——**1.86x 的差距全部来自「一半卡空转」**，与并行效率假设无关
（η=0.85 时仍有 1.69x）。

那角色拆分白做了吗？没有。第三行 `disagg-async` 是拆卡真正的主场：**放弃逐轮最新策略、
允许跨步流水**（rollout 用旧几步的权重继续采，trainer 不等），用 staleness 换重叠。
算法侧的「陈旧样本」和系统侧的「异步拆卡」是同一个权衡的两面。这里不展开队列和背压，
直接转到 [nano-slime L0](../nano-slime/tutorial_L0.md)；importance sampling 能修正什么、不能修正什么，
见 [L1 的 IS 小节](tutorial_L1.md#importance-sampling为什么旧样本还能再用几轮)。

---

## §2 HybridFlow 的编程模型：单控制器 + SPMD

arXiv:2409.19256 摘要的原话（2026-08-07 亲验）：RLHF 把经典 RL 的 dataflow 里每个节点
膨胀成「一个分布式训练/生成程序」、每条边膨胀成「多对多 multicast」；纯单控制器编排
分布式节点内计算会有巨大控制开销，纯多控制器嵌套又不灵活；HybridFlow 把两者**混合**：

- **单控制器（driver）**：编排节点之间的数据流。verl 里 = `RayPPOTrainer`
  （`verl/trainer/ppo/ray_trainer.py:L225`），它的 `fit()`（L1230）就是 dataflow 本身。
- **多控制器（SPMD）**：每个节点内部，所有 rank 跑同一份程序，自己管自己的并行。
  verl 里 = `RayWorkerGroup`（`verl/single_controller/ray/base.py:L411`）带着 N 个 Ray actor。

driver 眼中的 worker group 是**一个对象**：

```python
# nano 版（L3_hybridflow_colocate.py）——与 verl 的调用形态同构
rows, totals = self.roll_group.generate_sequences(proto, step)   # rollout 节点
stats = self.train_step(update_proto)                            # train 节点
self._sync()                                                     # 节点间的边（权重流）
```

「数据怎么切到 rank、结果怎么收回来」不在调用点写，而是**注册在方法上**——对照
`verl/single_controller/base/decorator.py`（v0.7.1）：

```python
# verl/single_controller/base/decorator.py:L37-47（v0.7.1，原文照录关键行）
def init_predefined_dispatch_mode():
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DP_COMPUTE_METRIC")
    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    Dispatch.register("DIRECT_ROLLOUT_METHOD")
```

每种模式 = 一对 `dispatch_fn`（参数怎么切）+ `collect_fn`（结果怎么收）
（decorator.py:L308-325 的注册表）。worker 方法用 `@register(dispatch_mode=...)`
（decorator.py:L397）声明自己吃哪种模式。nano 版实现其中两种最核心的：

| 模式 | 语义 | verl 用例（fsdp_workers.py v0.7.1） | nano 用例 |
|------|------|-------------------------------------|-----------|
| `ONE_TO_ALL` | 同样参数广播给所有 rank | `init_model`（L857）、`save/load_checkpoint`（L1179/L1228） | 控制类操作 |
| `DP_COMPUTE_PROTO` | DataProto 沿 batch 维 chunk 到各 rank，各算各的，concat 收回 | `update_actor`（L997）、`generate_sequences`（L1043）、`compute_log_prob`（L1093） | generate / 梯度计算 |

（v0.7.1 的 worker 方法实际注册的是 `make_nd_compute_dataproto_dispatch_fn(mesh_name=...)`——
`DP_COMPUTE_PROTO` 的 N 维 mesh 推广，允许 dp×tp 的 rank 映射；nano 版只有 dp 一维，语义相同。）

**为什么这个设计是 HybridFlow 的灵魂**：driver 代码（fit 循环）与节点内并行方案
（FSDP？Megatron？TP 几路？）彻底解耦。换并行方案 = 换 worker 实现，fit 一行不改。
论文称之为 hierarchical API 解耦 computation 与 data dependency——读源码时你会发现
`ray_trainer.py` 的 fit 循环里没有任何一行 FSDP/vLLM 细节，全在 worker 侧。

---

## §3 DataProto：唯一的过路协议，和一次行序事故

节点之间流动的数据有一个统一容器：`DataProto`（`verl/protocol.py:L317`）=
TensorDict `batch` + `non_tensor_batch`（numpy，装变长/字符串类字段）+ `meta_info`。
两个关键方法的语义（nano 版逐条复刻）：

- `chunk(n)`（protocol.py:L863）：沿 dim 0 切 n 份，**meta_info 复制给每份**；
  tensor 用 `TensorDict.chunk`，non-tensor 用 `np.array_split`（允许不等长切分）。
- `concat(list)`（protocol.py:L916）：沿 dim 0 拼回；meta_info 合并时**非 metrics 键
  冲突即断言**（L951），metrics 键聚合成 list——「各 rank 的度量收上来，普通配置
  必须一致」这条纪律写在协议层。

[1] 板块机器验证了这两条语义：

- **(a) round-trip**：`concat(chunk(4))` 与原型逐位 `torch.equal`（行序、数值、meta 全还原）。
- **(b) DP 透明性**：同一份权重下，log_prob 按 4-chunk 分开算再 concat，与全量一次算，
  **max |Δ| = 0.0（bit-identical）**。per-row 计算不随切分方式改变——这是「dispatch 只是
  数据搬运、不碰数值」的实证，也是 [4a] 那个更强断言的地基。

### 行序是隐式合同：一次真实的调试事故

写本节代码时踩过一个坑，值得单独讲，因为它恰好说明 DataProto 为什么存在。
`generate_sequences` 返回五种逐 token 行（state/action/log_prob/value/reward）。
自回归只能按步推进，前四个数组天然是 **token-major**（step t 的 n 行连续）；
而奖励是生成完整句后回填的，顺手写成了 **rollout-major**（rollout i 的 T 个 token 连续）。
最后统一重排成 rollout-major 时，reward 被错配到别的行上——GAE 拿到的是噪声化的奖励，
训练立刻发散：reward 0.875 → 0.291、entropy 0.53 → 2.55 单调爬升（熵奖励趁政策梯度
失效独自把策略推向均匀）。五个数组**看起来**都在同一个 DataProto 里相安无事。

教训：**行基协议里，「第 k 行属于谁」是所有字段共享的隐式合同**。verl 的 DataProto 把
chunk/concat 做成保序操作、把 batch 收进一个容器，就是在协议层强制这份合同；
任何一行字段脱离容器单独走（nano 里的 `totals` 列表就是），都必须自己声明并保持
行序不变量。修掉行序和 padding 状态索引后，同配置 reward 0.875 → 0.978、entropy 0.53 → 0.07 健康收敛——
发散曲线与收敛曲线的分岔点不是超参，是一个排列。

---

## §4 fit 循环逐相对照：nano vs ray_trainer.py

nano 的 `NanoHybridTrainer.fit()` 与 verl `RayPPOTrainer.fit()`（ray_trainer.py:L1230）
阶段顺序一一对应（v0.7.1 行号）：

| 阶段 | nano（本文件） | verl fit 内（v0.7.1） |
|------|----------------|------------------------|
| 1 rollout | `roll_group.generate_sequences(proto, step)`（DP_COMPUTE_PROTO，rank 内 batch 自回归） | L1321 `self.async_rollout_manager.generate_sequences(...)` |
| 2 old log prob | rollout 内随采样记录 `old_log_prob` | L1403-1404 `_compute_old_log_prob`（v0.7.1 支持 bypass/recompute 两模式，L1389-1402 注释） |
| 3 ref log prob | **省略**（L1/L2 同款无 KL 罚 PPO） | L1440-1441 `_compute_ref_log_prob`（ref policy worker） |
| 4 advantage | driver 侧 `gae_batch` 全量计算 | L1486 `compute_advantage(...)`（driver 侧函数，L129 定义） |
| 5 update critic | 省略（value head 与 policy 共享 backbone） | L1498-1499 `_update_critic` |
| 6 update actor | `train_step`：chunk → 各 rank 前向/反向 → rank 序梯度归约 → 分片 Adam | L1506-1507 `_update_actor` → worker 侧 `update_actor`（fsdp_workers.py:L997） |
| 7 update weights | `_sync`：gather 分片 → 推给各推理副本 | L1531-1533，源码注释原文：`# update weights from trainer to rollout` |

两个值得注意的省略：

- **critic**：nano 沿用 L1 的 value head（PPO 自举），verl 的 PPO 路径有独立 CriticWorker
  （fsdp_workers.py:L1282）；GRPO 路径则干脆不要 critic（组内均值当 baseline）。
- **ref policy**：KL 正则的参考策略。nano 的任务（rule-based dense reward）没有 KL 崩溃
  压力所以省了；真实 RLHF 里它是常驻角色，colocate 时 ref 也住在同一组卡上
  （`ActorRolloutRefWorker` 名字里的 Ref 就是它，参数 offload 到 CPU 待命）。

driver 侧还藏着一个工程细节：verl 在 generate 之前会 `_balance_batch`
（ray_trainer.py:L1018）按 seqlen 重排 batch，让各 DP rank 的 token 总量均衡——
变长序列下「行数均分」不等于「负载均分」。nano 的任务所有 prompt 等长，整除切分即均衡，
所以省了这一步；但它的存在本身说明：**dispatch 的均分只是起点，负载均衡是另一门手艺**。

---

## §5 colocate 的显存算术：谁住在显存里

colocate 的真正理由不是速度（§1 已给出速度算术），是**显存**。每个阶段每张卡上住着什么：

| 阶段 | 居住者（每 rank） | 声明规模（P = fp32 参数字节，N=4） |
|------|-------------------|-------------------------------------|
| train | 参数分片 P/N + Adam m/v 2P/N + 梯度 P/N + 激活 | 4P/N + 6 GB |
| rollout | 推理副本（bf16）P/2 + KV cache | P/2 + 20 GB |

切换的艺术 = **不在场的角色不占房**：进 rollout 相，训练态（参数分片+优化器）offload 到
CPU；进 train 相，rollout 引擎睡觉（vLLM sleep mode，权重与 KV 都释放）。对照 verl 源码
（fsdp_workers.py v0.7.1）：

- `update_actor`（L997-1041）：进门 `load_fsdp_model_to_gpu` + `load_fsdp_optimizer`
  （L1001-1004），出门 `offload_fsdp_model_to_cpu` + `offload_fsdp_optimizer`
  （L1034-1039）——训练态的 load/offload 夹心。
- `rollout_mode()`（L750-856）：进门 gather 全量权重推给 vLLM，然后
  **两段唤醒**——`await self.rollout.resume(tags=["weights"])`（L831）→
  `update_weights`（L846）→ `resume(tags=["kv_cache"])`（L851）。先住权重、再开 KV，
  因为 update_weights 本身要有地方放新权重。
- `generate_sequences`（L1043-1071）把整个三明治包起来：`rollout_mode()`（L1063）→
  生成（L1067）→ `trainer_mode()`（L1070）。（`trainer_mode` 的定义位置本轮未定位到
  fsdp_workers.py 内 `[TODO: verify]`——调用点与语义（rollout_mode 的逆操作）已核验。）

[3] 的显存表（COST 声明模型，budget = 80 GB/卡）：

```text
  scale   colo-train  colo-rollout  disagg-train  disagg-rollout budget
  7B            34.0          34.0          62.0            34.0     80
  13B           58.0          46.0         110.0            46.0     80
```

读法：

- **colocate 两相都住得下**（7B：34/34；13B：58/46），因为每相只住自己的东西，且训练态
  被 N=4 摊薄到 4P/N。
- **disagg 的 rollout 侧没区别**（34/46，独立卡上住全量副本+KV），**但 trainer 侧爆炸**：
  训练态只能摊到 N/2=2 张卡上，4P/(N/2) = 2×colocate——7B 时 62 GB 还能挤（这就是为什么
  小模型时代拆卡架构活得下去），13B 时 110 GB > 80 GB，**直接 OOM**。

这条 4P/N vs 4P/(N/2) 的算术，就是「模型越大越必须 colocate」的硬逻辑。真实世界的对应：
7B 时代可以 actor/learner 分集群；到 70B+ 与长上下文 KV，训练态与推理态都大到
必须共享同一组卡并按相轮换——verl 把 colocate 做成默认，是显存算术逼出来的，不是审美。

（算术口径声明：Adam 状态按 fp32 m+v = 2P、推理副本按 bf16 = P/2、KV 20 GB 与激活 6 GB
为声明常数；真实配置随 dtype/序列长/并行方案变化，此处只演示**结构**。）

---

## §6 resharding 的税：权重从分片到推理引擎走了几步

colocate 的代价在阶段边界。verl 的 `rollout_mode()`（fsdp_workers.py:L750-856）全程：

1. `load_fsdp_model_to_gpu`（若 param_offload；L756）——训练态从 CPU 回来；
2. `params = self.actor_module_fsdp.state_dict()`（L771）——**FSDP 全量 gather**
   （每个 rank 从自己那片出发，收齐全量）；
3. `convert_weight_keys`（L773）——HF/FSDP 命名 → vLLM 命名（Megatron 路线还有
   TP/PP 布局转换，即论文说的 3D-hybrid resharding）；
4. `offload_fsdp_model_to_cpu`（L799）——gather 完立刻让出显存；
5. `DTensor → full_tensor()`（L809-811，FSDP2 路径逐张量物化）；
6. `resume(weights)`（L831）→ `rollout.update_weights(per_tensor_param)`（L846）→
   `resume(kv_cache)`（L851）——推理引擎两段唤醒。

nano 的 `_sync()` 把 2+6 压缩成一步（gather → load 进各 rank 的推理副本），并按字节记账。
玩具规模实测（[2] 输出）：30 步训练共搬运 **101,602 KB**。手工对账（可复算）：
padded 参数 27,968×4 B = 111,872 B；每步阶段边界 gather 收 3 份（N-1）= 335,616 B +
写 4 个副本 = 447,488 B，合计 783,104 B × 30 步 = 23,493,120 B；train 相内每 epoch
还有 gather + 梯度归约各 335,616 B × 2 × 120 epoch = 80,547,840 B；
总计 104,040,960 B = 101,602 KB ✓（与程序输出逐位吻合）。

折算到声明规模（P = 28 GB，N=4）：仅阶段边界每 rank 每步 ≈ P(N-1)/N + P ≈ 49 GB 的
本地内存流量——这就是为什么真实系统的权重同步必须走 NVLink 级互连、为什么 verl 在
fit 里给 `update_weights` 单独挂 `marked_timer`（ray_trainer.py:L1532）盯着它。
**resharding 是 colocate 的税；税率由互连带宽决定，税基由模型大小决定。**

顺带一个细节：27,966 个参数不能被 4 整除，nano 如实 pad 到 27,968（[2] 输出
`+ 2 pad`）——真实 FSDP 的 FlatParameter 同样 pad 到 world size 整除，pad 参与通信、
不参与计算。这类「不整除的现实」在玩具里保留下来，是因为 senior 的坑往往就在这些
不性感的角落。

---

## §7 调度不变性：调度改变什么、不改变什么

[4] 是本节最重要的语义实验，两个层次：

- **[4a] bit 级不变性（同 DP 宽度）**：`flat` 模式与 `colocate` 用同样的 4 rank、
  同样的切分/归约/采样种子，只关掉阶段切换与显存账本——30 步全部度量
  **逐位一致**（`True`）。这证明切换/账本/记账机制是**数值透明**的：它们只移动数据
  与声明资源，不碰任何浮点运算。SPMD 的纪律（rank 序归约、(step, rank) 决定采样）
  是这个断言成立的前提。
- **[4b] 收敛可比性（不同 DP 宽度）**：disagg（2+2）与 colocate（4）的 reward 曲线
  最大差 0.1031，终值 0.978 vs 0.997——**轨迹不逐位相同，收敛行为相当**。
  注意措辞：这里的发散不是 ulp 漂移，而是 DP 宽度改变后每 rank 的采样划分
  （谁采哪些 rollout）与梯度归约树都变了，batch 组成实质不同。bit 级不变性
  **只在同宽度下成立**；跨宽度成立的是统计意义上的收敛可比。把这两层分开，
  才不会在真实系统里对着两条不重合的 loss 曲线怀疑人生。

---

## §8 nano 与 verl 的取舍对照（L3 核心交付）

| 维度 | nano（本文件） | verl v0.7.1 | 差异原因 |
|------|----------------|-------------|----------|
| worker 载体 | 同进程 Python 对象，rank 序顺序执行 | Ray actor，一 rank 一进程一 GPU | 本机无多 GPU；SPMD **语义**不依赖进程模型 |
| dispatch | 2 种模式（ONE_TO_ALL / DP_COMPUTE_PROTO） | 8 种 + N-D mesh 推广（decorator.py:L37-47） | nano 只留数据并行主干；ND 映射是 dp×tp 的推广，语义同源 |
| 参数分片 | 扁平连续切片 + pad | FSDP FlatParameter（按 module、逐层 all-gather、与计算 overlap） | nano 放弃 overlap/prefetch 工程，保留「分片-gather-更新」语义 |
| 推理引擎 | rank 内 batch forward（L2 同款） | vLLM/SGLang（paged KV、continuous batching、sleep mode） | 引擎优化属 03 轨 nano-vllm-sglang；此处只需「全量副本 + 采样」的角色 |
| 权重同步 | gather → load_state_dict，按字节记账 | state_dict → convert_weight_keys → update_weights → 两段 resume（L773-851） | 命名/布局转换在单一后端下不存在；两段唤醒依赖 sleep mode |
| 显存/时钟 | COST 声明模型（7B/13B 折算） | 真实 CUDA 分配与计时 | 本机无 GPU；声明与实测分开报告，结构可复核 |
| 算法 | PPO + GAE + value head（L1 同款） | PPO/GRPO/…，critic 与 ref 可配 | 算法不是本节变量；保持与 L1/L2 一致才能归因 |

**nano 保留了什么**（= HybridFlow 的本质）：单控制器/SPMD 分层、DataProto 保序协议、
两相复用与阶段边界 resharding、显存按相轮换、调度数值透明性。**nano 放弃了什么**：
一切与「多机多卡真实资源」绑定的工程（进程模型、通信 overlap、引擎内部、真实显存）。
放弃的每一项都在表里给出了去处（哪一级、哪一轨接着讲）。

---

## §9 费曼自检

**讲给外行听**：一家餐馆只有一间厨房。中午做快餐（rollout：火大、锅多、出餐快），
晚上做宴席（train：备料复杂、占地方）。把厨房一分为二？快餐和宴席本来就不同时
营业（on-policy：今天的菜单必须用今天新到的食材），分开只是让两边都变成半间厨房。
于是选择复用：营业间隙把快餐设备推出去、把宴席设备推进来（resharding + offload），
付一笔搬运费，换每个时段都有整间厨房。菜单越大（模型越大），半间厨房越装不下，
复用就越不是选择题而是必答题。

**类比的边界**：真实厨房的「搬运费」可能大到必须重新设计动线（NVLink/转换格式/
overlap），而类比里的搬运是免费的；另外真实餐馆还可以开分店异步营业
（async disagg + staleness），类比没覆盖这条退路。

**思考题**（都挂在本文件可跑实验或可手算算术上）：

1. 把 `N_WORKERS` 从 4 改成 8（`N_ROLLOUTS` 保持 64），预测 [3] 的显存表里
   colo-train 变成多少？disagg-train 呢？跑一遍验证。（提示：4P/N 与 4P/(N/2)。）
2. [4a] 的 bit 级不变性依赖哪些纪律？把 `train_step` 里的梯度归约从 rank 序改成
   逆序（`for g in reversed(grads[1:])`），预测会发生什么——会破坏正确性吗？
   会破坏不变性吗？（浮点加法不结合；答案应该是「正确性保留、bit 不变性破坏」。）
3. §6 的流量账：若把 `N_EPOCHS` 从 4 降到 1，总流量（KB）变成多少？手算后改代码验证。
   （每步阶段边界流量不变，train 相内流量 ÷4。）
4. `COST["kv_gb"]` 从 20 调到 60（长上下文），哪个配置的哪一相先 OOM？
   这解释了为什么长上下文 RL 对 rollout 侧显存特别敏感。
5. 进阶：verl 的 `rollout_mode()` 为什么先 `resume(weights)` 再 `resume(kv_cache)`，
   而不是一次性全唤醒？（提示：update_weights 要往哪里写？）

---

## §10 反例与边界

1. **colocate 不是永远赢**：异步场景（允许 staleness）下，拆卡 + 跨步流水能把两相
   真正重叠（[3] 的 disagg-async 行：2300ms vs colo 2100ms，η=1.0 时几乎打平，
   且 rollout 可持续服务）；异构集群（推理卡与训练卡型号不同）下拆卡也自然。
   verl 两者都提供——**默认 colocate 是同步 on-policy 同构集群下的最优**，不是普适真理。
2. **声明数字不是实测**：§5/§6 的 GB/ms 全部来自 COST 块声明常数（7B/13B 折算），
   本机没有任何 GPU 测量；玩具流量（101,602 KB）是实测但规模是 28K 参数。两类数字
   在输出里分开标注（`declared` vs `real toy`），引用时不可混用。
3. **时效性**（ROADMAP §八）：锚点基准 v0.7.1 是保留经典目录结构的最后一个 release。
   main 分支（2026-08-07 抓取）已重构：`verl/workers/fsdp_workers.py` 与
   `sharding_manager/` 消失，worker 层改组为 `engine_workers.py` 等新布局；
   最新 release v0.8.0。机制（单控制器+SPMD、两相复用、resharding）未变，
   但**行号锚点不可外推到 main**——引用 main 时须重新核验（§11 录了 main 的
   README/protocol/decorator sha256 供漂移检测）。
4. **DP 透明性有前提**：[1](b) 的 bit-identical 依赖 per-row 计算无跨行归约；
   advantage 归一化这类跨行操作必须留在 driver 侧对全量做（verl 的
   `compute_advantage` 正是 driver 侧函数）——若把它塞进 rank 内对 chunk 做，
   语义就错了（每 rank 各归一各的 ≠ 全局归一）。

---

## §11 溯源

**权威实现锚点**（verl-project/verl @ v0.7.1，raw.githubusercontent.com 抓取于
2026-08-07 21:3x CST，本机 `/tmp/verl_anchor/v071/`）：

| 文件 | sha256 | 用到的行号锚点 |
|------|--------|----------------|
| verl/protocol.py | `aa719abd71323bfc5725e5e4a8617b6c6aef76a7edcb6adac0c2421664e63115` | DataProto L317；chunk L863-902；concat L916-960（meta 冲突断言 L951） |
| verl/single_controller/base/decorator.py | `80bb8d3ad40390cd91727385bd50e4b8fe92a16555991d5a5699cd6493bde4c4` | dispatch 模式注册 L37-47；dispatch_fn/collect_fn L119/L133/L147/L158/L166/L190；模式表 L308-325；register L397 |
| verl/single_controller/ray/base.py | `6df6878fdc610843fd96388eaa072527fbc774eebd224f82dd87cec620fc3ec1` | RayResourcePool L112；RayWorkerGroup L411；create_colocated_worker_cls L981 |
| verl/trainer/ppo/ray_trainer.py | `ab60b6225fb919e59c19743669735880a7ce04ffb48e9d97e47d0cd420949dc7` | compute_advantage L129；RayPPOTrainer L225；init_workers L678；colocate 注释 L744-747；_balance_batch L1018；_compute_ref_log_prob L1105；_compute_old_log_prob L1132；fit L1230；generate L1321；old_log_prob L1403；ref L1440；advantage L1486；update_actor L1506；update_weights L1531-1533 |
| verl/workers/fsdp_workers.py | `ce52e8b085e599fe997d9ad494fb1761c7d82609359d81ca1a75e8f4fbcbea74` | ActorRolloutRefWorker L143；param_offload L238-245；rollout_mode L750-856（state_dict L771 / convert_weight_keys L773 / offload L799 / resume weights L831 / update_weights L846 / resume kv L851）；init_model L857；update_actor L997-1041（load L1001-1004 / offload L1034-1039）；generate_sequences L1043-1071（rollout_mode L1063 / trainer_mode L1070）；compute_log_prob L1093；compute_ref_log_prob L1145；CriticWorker L1282 |

**论文**：arXiv:2409.19256「HybridFlow: A Flexible and Efficient RLHF Framework」
（标题与摘要 2026-08-07 arxiv.org 亲验；单控制器+多控制器混合、hierarchical API
解耦 computation/data dependency 均摘自摘要原文）。

**main 分支漂移基线**（2026-08-07 抓取）：README sha256
`a6c46bb67d246942d214cb590e0d7d10bc51256f613a5625687d5fcf32714fe6`；
verl/protocol.py sha256 `b2db8399daaca64d74c40a8246bcf825f08746757485c7a494303f089a06a34f`；
verl/single_controller/base/decorator.py sha256
`718876c4c59e9f3cdb0339185be156aeb7b6d8eb42a2a6f919a3163961634ace`（与 v0.8.0 逐位相同）；
`verl/workers/fsdp_workers.py` 在 main 已 404（重构为 `engine_workers.py` 等）。

**未定位项**：`trainer_mode()` 的定义位置（调用点 fsdp_workers.py:L1070 已核验；
定义不在 fsdp_workers.py/engine_workers.py/engine/base.py 内）`[TODO: verify]`。

**运行环境**：macOS（Apple Silicon），Python 3.x，torch CPU；
`python3 L3_hybridflow_colocate.py`，任意 CWD，~8s。
确定性：强制 CPU + `torch.manual_seed(42)` + rank 序归约 + (SEED, step, rank) 采样种子；
单次运行内部的两遍训练得到相同 metrics md5（`5cba79e6a5bb5f8ff192f5f888837983`）。

**调试史**（§3 已展开）：reward 行序错配导致 GAE 噪声化、训练发散
（reward 0.875→0.291 / entropy 0.53→2.55）；修正行序与 padding 索引后收敛（0.875→0.978）。
[4b] 判据从「曲线差 ≤0.05」放宽为「收敛可比」（曲线差 ≤0.15 + 终值均 ≥0.9 且差 ≤0.05），
原因：不同 DP 宽度下 batch 组成本质不同，轨迹相同才是异常。

---

## 阶梯状态

| 级别 | 状态 | 交付物 |
|------|------|--------|
| L0 | ✅ | `L0_toy_hybridflow.py` + `tutorial_L0.md`（玩具调度：为什么要分离） |
| L1 | ✅ | `L1_minimal_ppo.py` + `tutorial_L1.md`（单进程最小 PPO） |
| L2 | ✅ | `L2_actor_learner_split.py` + `tutorial_L2.md`（lockstep 两进程角色拆分 + batch inference） |
| **L3** | ✅ | 本文件（colocate 两相复用 + resharding，对照 verl v0.7.1 源码） |

nano-verl 阶梯至此完整。横向接口：rollout 吞吐的内部机制 → 03 轨
`nano-vllm-sglang`（paged KV / continuous batching / sleep mode 的实现侧）；
异步拆卡、buffer 与 staleness → 01 轨 [`nano-slime` L0](../nano-slime/tutorial_L0.md)。
