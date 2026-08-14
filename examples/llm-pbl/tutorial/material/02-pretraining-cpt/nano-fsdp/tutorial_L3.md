# nano-fsdp L3 — 混合精度 + 激活检查点：显存账本的另一半

> **核心机制**：L0–L2 把「模型状态」这本账算透了（16 bytes/param、ZeRO 各 stage 的
> 切法与通信量）。但训练时显存里还有**另一半**——激活（activations），它随
> batch×seq×层数增长，ZeRO 对它一个字节都动不了；而 mixed precision 重写的不是
> 「算得快」，是账本里每一行的 dtype 构成。本节用真实 PyTorch（真 bf16 kernel、
> 真 gloo 集合通信、真 FSDP1/FSDP2）量出三件事：
>
> 1. **activation checkpointing** 用一次重算换激活显存（fp32 实测省 82.9%），
>    与 ZeRO 分片**正交叠加**（FSDP 包裹下省出的字节与单进程逐字节相同）；
> 2. **mixed precision 不减少模型状态总量**（fp32 16Ψ = 混合精度 16Ψ，实测），
>    它减少的是激活（bf16 减半）与 all-gather 通信（bf16 减半，实测恰 /2）；
> 3. **fp32 master copy 是数值硬需求**：bf16 参数上的 Adam 更新被舍入直接吞掉
>    （实测参数冻结 5 步），FSDP 的 MP policy 把 fp32 shard 直接当 master weights。
>
> **运行要求**：`torch`（CPU 即可，本机实测 torch 2.13.0）。命令：
> ```bash
> python3 L3_mixed_precision_checkpointing.py   # ~3s
> ```
> 分布式部分是 2 个真实进程 + gloo（任意 CWD 可跑）；真实生产在 GPU 上，
> dtype 流转语义在源码层面与设备无关，绝对数字（吞吐/峰值显存）按 GPU 放大，
> 标 `[TODO: verify on real system]`。

---

## 1. 本节目标（K+1：从「模型状态账本」到「端到端显存」）

L2 结尾留了四个明确缺口（tutorial_L2 §9 表与 §12「下一步」逐条列名）：

1. **mixed precision 如何改写 16× 账本**——L2 全程 fp32，而 ZeRO 论文的 16Ψ
   本身就是混合精度口径（fp16 参数/梯度 + fp32 master/m/v），两种口径的映射
   L2 §4.2 只给了表，没给机制；
2. **激活显存**——ZeRO 不解决的另一半（L2 §9 表最后一行）；
3. **FSDP1 → FSDP2**——composable `fully_shard` 与「包 wrapper」的 API 形态差异
   （L2 §12 留了 `[TODO: verify FSDP1 的官方 deprecation 时间表]`，本节修订）；
4. L2 思考题 5 的**峰值**问题在混合精度下的形态（MP 的 unshard 峰值是 bf16 还是 fp32）。

L3 全部用真跑回答。答案用一张表预告（下面是机器验证，不是断言）：

| 实验块 | 量什么 | 权威对照 |
|--------|--------|----------|
| [0] | 激活账本：dtype × checkpoint 四组合 | ZeRO §3.2 / arXiv:1604.06174 / arXiv:2205.05198 §4 |
| [1] | bf16 Adam 为什么必须有 master weights | ZeRO §3.1 / DeepSpeed `stage_1_and_2.py` |
| [2] | 16Ψ 的三种 dtype 拆法（实测 bytes/param） | ZeRO §3.1（K=12） |
| [3] | 真 FSDP MP：每 rank 模型状态与存储 dtype | FSDP `MixedPrecision` / FSDP2 `MixedPrecisionPolicy` |
| [4] | 真 FSDP MP：每步通信字节（dtype 感知） | `_flat_param.py` 低精度 shard 路径 |
| [5] | FSDP + 官方 activation checkpointing | `apply_activation_checkpointing` |
| [6] | FSDP2 `fully_shard`：DTensor 契约 | FSDP2 文档 user contract |

---

## 2. 先跑起来

```bash
python3 \
  tutorial/material/02-pretraining-cpt/nano-fsdp/L3_mixed_precision_checkpointing.py
```

输出（完整 119 行；计时行——以 `elapsed` 开头的 3 行：`elapsed[single-process]:` /
`elapsed[distributed]:` / `elapsed:` 总计——随机器浮动，锚点口径为整行删除后 md5：
`sed '/^[[:space:]]*elapsed/d'`，见 §13）：

```
========================================================================
nano-fsdp L3 — mixed precision + activation checkpointing
========================================================================
TinyLM: vocab=128 dim=64 layers=2 | P = 116,480 params
cluster: W = 2 real processes, gloo backend, CPU (真实多进程 + 真实集合通信；生产在 GPU，机制相同)
steps: 1 warmup + 3 measured | seed = 7 (与 L1/L2 一致)

elapsed[single-process]: 0.6s ([0][1][2] 真 kernel; 计时行浮动)

[0] ACTIVATION LEDGER（单进程，saved_tensors_hooks 实测；batch=(8, 16)）
    dtype     ckpt    saved after fwd  n_pack  body  hook             loss
    ------------------------------------------------------------------------
    float32   False         1,750,532      60     2     2  5.0277075767517
    float32   True            298,500      14     4     2  5.0277075767517
    bfloat16  False           878,338      60     2     2  5.0000000000000
    bfloat16  True            150,274      14     4     2  5.0000000000000
    checkpoint 省激活：fp32 1,750,532→298,500 B（-82.9%），bf16 878,338→150,274 B
    重算计数：ckpt 下 forward 体执行 4 次（2 块 × 首遍+重算），而 forward hook 只触发 2 次——重算被 _StopRecomputationError 提前终止，hook 永远等不到 forward 返回（机制见 tutorial §5）

[1] WHY MASTER WEIGHTS（bf16 参数上跑真 Adam，lr=1e-3，5 步，初值 1.0）
    bf16 params: p[0] = 1.0（Δ = +0.0e+00，更新被舍入吞掉——参数冻结）| exp_avg dtype = bfloat16
    fp32 params: p[0] = 0.9950000643730164（Δ = -0.0050，正常前进）| exp_avg dtype = float32
    分辨率：eps(bf16) = 0.0078125，eps(fp16) = 0.0009765625，eps(fp32) = 1.192e-07（torch.finfo 实测）
    torch 梯度 dtype 守卫：bf16 参数赋 fp32 梯度 → RuntimeError（消息含 'grad_dtype'：True）

[2] MIXED-PRECISION LEDGER（n = 1024 参数，真跑一步 Adam 后盘点；B/param）
    regime       params   grads  master     m+v |   total
    ----------------------------------------------------
    fp32            4.0     4.0     0.0     8.0 |    16.0
    bf16-naive      2.0     2.0     0.0     4.0 |     8.0
    mixed           2.0     2.0     4.0     8.0 |    16.0
    ZeRO 论文（arXiv:1910.02054 §3.1）口径：2Ψ+2Ψ+KΨ = 16Ψ，K=12——mixed 一行就是它的实测展开
    注意 total：mixed(16) == fp32(16) ≠ 减半——混合精度不省模型状态，省的是激活（[0]）与通信（[4]）；bf16-naive 的 8 B/param 以参数冻结为代价（[1]），不可用

[3] per-rank MODEL STATE（真 FSDP FULL_SHARD，W=2；params+grads+Adam）
    mode           params     grads       opt |     total storage dtype  formula
    ------------------------------------------------------------------------------
    fsdp3_f32       0.22M     0.22M     0.44M |     0.89M float32        16P/W
    fsdp3_mp        0.22M     0.22M     0.44M |     0.89M float32        16P/W
    fsdp2_f32       0.22M     0.22M     0.44M |     0.89M float32        16P/W
    fsdp2_mp        0.22M     0.22M     0.44M |     0.89M float32        16P/W
    （M = MiB；16P/W = 0.89 MiB。**mp 两行存储仍是 fp32**——低精度只发生在计算与 all-gather 通信上，fp32 shard 就是 master weights）

[4] per-step COMM BYTES（dtype 感知计量；口径同 L2：gather/rs 各记完整张量）
    mode              gather  reduce-scatter        total gather dtype
    ----------------------------------------------------------------------
    fsdp3_f32       2.597 MB        1.398 MB     3.995 MB float32
    fsdp3_mp        1.299 MB        1.398 MB     2.696 MB bfloat16
    mixed precision 通信 = fp32 的 67.5%：gather 减半（bf16），reduce-scatter 不变（reduce_dtype=fp32 保数值稳定）

[5] FSDP + ACTIVATION CHECKPOINTING（官方 apply_activation_checkpointing）
    场景                                  saved after fwd
    ------------------------------------------------------
    单进程 fp32（[0] 参照）                          1,750,532
    单进程 fp32 + checkpoint（[0] 参照）               298,500
    FSDP FULL_SHARD fp32                      1,750,532
    FSDP FULL_SHARD fp32 + AC                   298,500
    loss（rank 平均）：no-AC 5.056958 vs AC 5.056958——fp32 重算逐位一致
    激活账本在 FSDP 包裹下与单进程逐字节相同：ZeRO 切模型状态、checkpoint 切激活，两笔账正交叠加

[6] FSDP2 fully_shard（composable API，DTensor 分片，真跑）
    type(param) = DTensor；本 rank local shard 元素总数 = 58,240 = P/2（116,480/2）
    fsdp2_f32 每 rank 状态 = 0.89 MiB（= 16P/W）；fsdp2_mp local shard dtype = float32（MP policy 下存储仍 fp32——sharded 高精度参数即 master weights，FSDP2 文档原话见 tutorial §9）

[7] correctness：各模式 vs 单进程参照（loss 为 rank 平均 = 全 batch loss）
    mode        step losses                   max|Δ| vs ref
    --------------------------------------------------------
    fsdp3_f32   4.865972 4.695927 4.533425        7.153e-07
    fsdp3_mp    4.843750 4.703125 4.546875        3.125e-02
    fsdp2_f32   4.865972 4.695927 4.533425        7.153e-07
    fsdp2_mp    4.843750 4.703125 4.546875        3.125e-02
    ref_fp32    4.865972 4.695926 4.533425   (ground truth)
    ref_bf16    4.875000 4.687500 4.531250        (bf16 参照)
    fsdp3_mp/fsdp2_mp 的 Δ 是 bf16 量化噪声（恰 1 ULP：loss ∈ [4,8) 的 ULP = eps×4 = 0.03125）；fp32 模式的 Δ 是归约结构舍入（同 L2 §8）

[8] self-check
    PASS  ckpt saves less activation (fp32)
    PASS  ckpt saves less activation (bf16)
    PASS  bf16 activation ≈ fp32/2 (ratio 0.5018)
    PASS  ckpt recompute bit-exact loss (fp32)
    PASS  ckpt recompute bit-exact loss (bf16)
    PASS  no-ckpt forward body execs == 2
    PASS  ckpt forward body execs == 2×2 (recompute real)
    PASS  ckpt forward hooks fire only on first pass (early-stop abort)
    PASS  ckpt packs fewer saved tensors
    PASS  bf16 Adam: param frozen (update swallowed by rounding)
    PASS  fp32 Adam: param advances
    PASS  torch enforces grad dtype == param grad_dtype (RuntimeError)
    PASS  Adam state follows param dtype (bf16)
    PASS  fp32 regime = 16 B/param (measured)
    PASS  bf16-naive regime = 8 B/param (measured)
    PASS  mixed regime = 16 B/param (measured)
    PASS  mixed total == fp32 total (MP 不省模型状态)
    PASS  fsdp3_f32 per-rank state 931888 within 1.5% of 16P/W=931840
    PASS  fsdp3_mp per-rank state 931888 within 1.5% of 16P/W=931840
    PASS  fsdp2_f32 per-rank state 931952 within 1.5% of 16P/W=931840
    PASS  fsdp2_mp per-rank state 931952 within 1.5% of 16P/W=931840
    PASS  FSDP1 MP: storage stays fp32 (master weights)
    PASS  MP gather bytes exactly halved (1298688 vs 2597376)
    PASS  reduce-scatter bytes unchanged (fp32): 1397760
    PASS  MP total comm < fp32 total comm
    PASS  MP gather collective runs in bf16
    PASS  FSDP activation bytes == single-process (sharding 不动激活)
    PASS  FSDP+AC activation bytes == single-process ckpt (正交叠加)
    PASS  AC loss bit-exact vs no-AC (fp32)
    PASS  FSDP2 params are DTensor
    PASS  FSDP2 local shards sum to P/W
    PASS  FSDP2 MP: storage stays fp32
    PASS  fsdp3_f32 losses match fp32 reference (<1e-5)
    PASS  fsdp2_f32 losses match fp32 reference (<1e-5)
    PASS  fsdp3_mp losses within 1 bf16 ULP of bf16 reference
    PASS  fsdp2_mp losses within 1 bf16 ULP of bf16 reference
    ✅ self-check passed (36/36)

digest(md5 of metrics) = 47e7ffd99c93628cea14f9feac4716e4

elapsed[distributed]: 2.7s (6 模式真跑; 计时行浮动)

elapsed: 3.2s total (计算真跑; 计时行浮动, 锚点口径见 tutorial §2)
```

36 项 self-check 全绿。下面逐块拆。

---

## 3. [0] 激活账本：ZeRO 碰不到的那一半

### 3.1 怎么量：saved_tensors_hooks

autograd 为 backward 保存的每个张量都要经过 `saved_tensors_hooks` 的 pack/unpack
——在外层挂一对钩子记字节数，就得到「forward 结束后为 backward 驻留的激活」的
精确账本（脚本 `SaveMeter`，与 10:00 轮 scratch 实验同构，见 §13 溯源）。
这不是估算：`after_fwd` 就是 backward 开始前驻留的保存张量总字节。

实测（batch=(8,16)，TinyLM 2 层）：

| dtype | checkpoint | saved after fwd | n_pack | loss |
|-------|-----------|-----------------|--------|------|
| fp32 | 无 | 1,750,532 B | 60 | 5.0277075767517 |
| fp32 | 有 | 298,500 B（**-82.9%**） | 14 | 5.0277075767517 |
| bf16 | 无 | 878,338 B（fp32 的 0.5018） | 60 | 5.0000000000000 |
| bf16 | 有 | 150,274 B | 14 | 5.0000000000000 |

三个结论，各自对应一个权威来源：

- **checkpoint 用重算换显存**：包了 `checkpoint` 的 block 只在 forward 时保存
  「输入 + 边界张量」（n_pack 60→14），backward 需要中间量时**把这层 forward
  重跑一遍**现场再生。这就是 Chen et al.（arXiv:1604.06174）的结论：n 层网络
  可以用 O(√n) 内存训练，代价是每个 mini-batch 多一遍 forward。本节实测的
  「-82.9%」是 2 层全 checkpoint 的形态；层数越深，分段 checkpoint 的 O(√n)
  结构越重要（思考题 2）。
- **bf16 激活减半**：激活的字节 = 元素数 × dtype 尺寸，bf16 直接 /2
  （实测比值 0.5018，零头来自 cross_entropy 内部 fp32 上采样的保存张量）。
  这是「mixed precision 省显存」的**真实出处**——省在激活，不在模型状态（§5）。
- **重算不改变数值**：两行 loss 逐位一致（fp32 与 bf16 各自内部 bit-exact）。
  fp32 显然；bf16 也逐位一致是因为重算用同样的 kernel、同样的输入、同样的
  归约顺序——checkpoint 的正确性保证是「重放」，不是「近似」。

### 3.2 激活账本在文献里的位置

ZeRO 论文（arXiv:1910.02054）§3.1 算完 16Ψ 模型状态后，§3.2「Residual Memory
Consumption」第一句就是激活：「Activations can take up a significant amount of
memory during training」（ar5iv 2026-08-11 现场抓取，节号结构见 §13）。ZeRO 把
激活归入「剩余显存」，由 ZeRO-R 的 partitioned activation checkpointing（§6.1）
处理——而 Megatron 一系对激活显存的闭式分析在 arXiv:2205.05198 §4.1（每层激活
内存）与 §4.3（总量），其 §5 进一步提出 **selective activation recomputation**：
「most of this redundant compute is unnecessary」（论文摘要原话）——只重算那些
「保存贵、重算便宜」的部分（典型：attention 的 softmax 中间量）。本节的 2 层
TinyLM 用全 checkpoint 演示机制本身，选择性重算的思想在思考题 2/3 延伸。

---

## 4. 一个反直觉的发现：重算时 forward hook 为什么不触发

`[0]` 表里藏着本节最值钱的机制细节：**ckpt 组合的 `body` 列 = 4，`hook` 列 = 2**。

- `body` 在 forward 函数体内部打点：2 个 block × (首遍 + 重算) = 4——重算**真的
  执行了** forward 代码；
- `hook` 是 `register_forward_hook` 的触发次数：只有 2——重算那遍 hook 没响。

用 hook 给「forward 跑了几遍」计数的人（profiler、FLOPs 统计、调试打点）在这里
会少数一半。原因要读 torch 2.13.0 的 `torch/utils/checkpoint.py` 源码（本机安装，
行号 2026-08-11 现场核验）：

1. 非重入式 checkpoint（`use_reentrant=False`，现行默认）的重算是**惰性的**：
   backward 需要哪个保存张量，`unpack_hook` 才触发重算（checkpoint.py:L1179-1185）；
2. 重算时，`_recomputation_hook` 的 pack 端逐个对账：重算出的第 i 个保存张量
   对应首遍的第 i 个 holder。**对完最后一个的瞬间抛出
   `_StopRecomputationError`**（checkpoint.py:L1084 类定义、L1131 raise）——
   后面的 forward 不用算了，backward 要的东西已经齐了；
3. 这个异常从 forward 体内部一路穿出，在 `unpack_hook` 被捕获
   （checkpoint.py:L1184 `except _StopRecomputationError: pass`）；
4. 关键在于它穿过了 `nn.Module._call_impl`：`inner()` 抛出异常后走的是
   **exception 分支**（module.py:L1884 `return inner()` → except），该分支只补跑
   `always_call=True` 的 hook（module.py:L1900 循环），普通 forward hook 不在其中
   ——而正常返回路径上的 hook 调用块（`result = forward_call(...)` 之后）永远
   执行不到，因为 forward **没有正常返回**。

`early_stop` 参数默认为 `True`（checkpoint.py:L362 签名）——即「按需重算、提前
终止」是默认行为。重入式（`use_reentrant=True`，旧默认）走的是另一条路：
`CheckpointFunction.backward` 里 `outputs = ctx.run_function(*detached_inputs)`
（checkpoint.py:L314）把 forward **完整跑完**，异常提前终止不存在，hook 会正常
触发（思考题 3 可亲手验证）。

senior 要点有二：**异常可以做控制流**（PyTorch 自己就用它实现「算够就停」，
比标志位干净——穿栈即生效）；**计量重算要用 op 级探针**（saved_tensors_hooks /
profiler），模块级 hook 在非重入 checkpoint 下系统性少计。

---

## 5. [1][2] master weights：不是精度保险，是数值硬需求

### 5.1 实测：bf16 参数上的 Adam 直接冻结

`[1]` 块把机制裸露到最小：初值 1.0 的参数，常数梯度 0.01，真 `torch.optim.Adam`
（lr=1e-3）跑 5 步：

- **bf16 参数：p[0] = 1.0，Δ = +0.0e+00——五步纹丝不动**。Adam 的每步更新量
  ≈ lr × m̂/√v̂ ≈ 1e-3，而 bf16 在 1.0 附近的 ULP = eps = 0.0078125（torch.finfo
  实测），1.0 + 0.001 舍入回 1.0——更新被吞掉。雪上加霜：torch Adam 的状态
  跟随参数 dtype，`exp_avg` 也是 bf16（实测），动量本身也在丢精度。
- **fp32 参数：p[0] = 0.9950000643730164**，正常前进 5×1e-3。

`[1]` 还量了 torch 2.x 的一道守卫：给 bf16 参数赋 fp32 梯度直接 RuntimeError
（消息含 `grad_dtype`）——梯度 dtype 必须匹配参数的 `grad_dtype`。这正是 FSDP
混合精度 policy 要替你管理的语义面：低精度计算产生的梯度走什么 dtype 回优化器，
是 policy 的一部分（`reduce_dtype`），不是用户手动 cast 的细节。

### 5.2 为什么 bf16 而不是 fp16：两种 dtype 各自失败的方式

`[1]` 打印了三种 dtype 的 eps：bf16 = 0.0078125（2^-7），fp16 = 0.0009765625
（2^-10），fp32 = 1.19e-07。注意 **fp16 的尾数比 bf16 细**——单看「吞更新」，
fp16 反而轻（1e-3 ≈ 1 个 fp16 ULP，部分存活）。那为什么大模型训练的主流是 bf16？
因为 fp16 失败在**另一头**：5 位指数，动态范围窄（格式上限 65,504；小梯度
underflow 到 0），必须配 loss scaling 把梯度抬进可表示区间；bf16 的 8 位指数与
fp32 同范围，不需要 loss scaling，代价是 7 位尾数粗——**所以 bf16 的 master
weights 比 fp16 的更不可商量**。两种低精度各自失败的方式不同，但解药相同：
fp32 master copy。这就是 ZeRO 论文 §3.1 的账本（ar5iv 现场抓取原文）：

> 「Mixed precision training of a model with Ψ parameters using Adam requires
> enough memory to hold an fp16 copy of the parameters and the gradients, with
> memory requirements of 2Ψ and 2Ψ bytes respectively. In addition, it needs to
> hold the optimizer states: an fp32 copy of the parameters, momentum and
> variance, with memory requirements of 4Ψ, 4Ψ, and 4Ψ bytes, respectively.
> Let's use K to denote the memory multiplier of the optimizer states…
> Mixed-precision Adam has K = 12. In total, this results in
> 2Ψ + 2Ψ + KΨ = 16Ψ bytes.」

**L0 的 16 bytes/param 从第一天起就是混合精度账本**——fp32 那 12 bytes 不是
「额外开销」，而是训练能进行的最低条件。

### 5.3 [2] 三种体制的实测账本

`[2]` 块对 n=1024 参数真跑一步 Adam 后盘点（B/param）：

| regime | params | grads | master | m+v | total |
|--------|--------|-------|--------|-----|-------|
| fp32 | 4 | 4 | – | 8 | **16** |
| bf16-naive | 2 | 2 | – | 4 | 8（参数冻结，不可用） |
| mixed | 2 | 2 | 4 | 8 | **16** |

核心结论（self-check 机器断言）：**mixed 的 total == fp32 的 total == 16**。
混合精度**不减少模型状态**——它把 16 bytes 重新分配（2+2+4+4+4），总量不变。
它真正减少的是：激活（§3，bf16 /2）与参数通信（§6，all-gather /2）。DeepSpeed
的 master 分片在源码里就是一个构造动作（`stage_1_and_2.py` master 分支
2026-08-11 抓取）：

```python
# deepspeed/runtime/zero/stage_1_and_2.py:L486-490（注释原文）
# A partition of the fp32 master weights that will be updated by this process.
# Note that the params in single_partition_of_fp32_groups is cloned and detached
# from the origin params of the model.
weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id].detach().clone().to(
    device=self.device, dtype=self.master_weights_and_grads_dtype)
```

append 进 `single_partition_of_fp32_groups`（L506），然后
`param_group['params'] = [self.single_partition_of_fp32_groups[i]]`（L511-513）
——**优化器从此只看 fp32 分片**，低精度参数只是它的影子（每步写回，`[2]` 的
mixed 体制就是这套布局的最小复现）。

---

## 6. [3][4] 真 FSDP mixed precision：存储、计算、通信三条 dtype 通道

### 6.1 实测：存储没变，变的是计算与 gather

`[3]` 块是真 FSDP1 `FULL_SHARD` + `MixedPrecision(param_dtype=bf16,
reduce_dtype=fp32, buffer_dtype=bf16)`（2 进程 gloo 真跑）：

| mode | params | grads | opt | total | storage dtype |
|------|--------|-------|-----|-------|---------------|
| fsdp3_f32 | 0.22M | 0.22M | 0.44M | 0.89M | float32 |
| fsdp3_mp | 0.22M | 0.22M | 0.44M | 0.89M | **float32** |

**MP 模式下每 rank 模型状态与纯 fp32 逐字节同量（均 931,888 B ≈ 16P/W），
存储 dtype 仍是 fp32**。第一次见到这个结果的人会以为 MP「没生效」——它生效了，
生效在三条不同的 dtype 通道上：

- **存储通道（fp32）**：sharded flat param 保持 fp32——它就是 master weights；
- **计算通道（bf16）**：forward/backward 前，`pre_unshard` 走
  `_use_low_precision_shard()`（`_flat_param.py:L1333`，torch 2.13.0），把 shard
  cast 成 bf16 再 all-gather——所以 `[7]` 的 MP loss 是 bf16 量化值
  （4.843750 vs fp32 4.865972）；
- **归约通道（fp32）**：梯度 reduce-scatter 用 `reduce_dtype=fp32`，
  `_cast_grad_to_param_dtype`（`_runtime_utils.py:L1030`）再把分片梯度 cast 回
  参数 dtype 交优化器——dtype 语义由 `_init_param_reduce_dtypes`
  （`_flat_param.py:L906-940`）的 `_fwd_bwd_param_dtype` / `_reduce_dtype`
  两个字段钉死。

FSDP2 文档对这套设计有一句点睛（`distributed.fsdp.fully_shard.html`，torch 2.13
文档页，2026-08-11 抓取，页面标注 Last Updated 2026-04-24）：

> 「FSDP works well with module-level mixed precision since it keeps the
> high-precision sharded parameters in memory anyway. In other words, FSDP does
> not require any extra memory to keep a high-precision copy of the parameters
> for the optimizer step.」

**sharded 高精度参数即 master weights——MP policy 不花一个额外字节就拿到了
master copy**。回看 `[2]`：mixed 体制的 16 B/param 里那 4 bytes master，在 FSDP
里就是参数分片本身，不是第二份拷贝。这是「为什么 FSDP 的 MP 是免费的」的精确
含义——免费的是 master copy，不是精度。

### 6.2 [4] 通信字节：gather 恰减半，reduce-scatter 不动

L2 的 CommMeter 升级为 dtype 感知（记每次集合通信的完整张量元素数 × dtype），
实测每步：

| mode | gather | reduce-scatter | total | gather dtype |
|------|--------|----------------|-------|--------------|
| fsdp3_f32 | 2,597,376 B | 1,397,760 B | 3,995,136 B | float32 |
| fsdp3_mp | 1,298,688 B | 1,397,760 B | 2,696,448 B | bfloat16 |

- **gather 字节恰 /2**（1,298,688 × 2 == 2,597,376，机器断言）：all-gather 的
  输入是 bf16 低精度 shard（§6.1 的 cast 发生在 gather 之前），元素数不变
  （L2 §5.2 的 2×blocks+root 结构照旧），字节减半；
- **reduce-scatter 逐字节不变**：`reduce_dtype=fp32` 是刻意的数值稳定选择——
  跨 rank 求和是误差放大点，低精度归约的 loss spike 风险大于省下的带宽；
- 合计 **67.5%**。若 `reduce_dtype` 也设 bf16，total 将到 50%（思考题 1 亲手验）。

### 6.3 决策表（L3 的工程落点）

| 瓶颈在哪 | 选 | 理由（用上面的数字） |
|----------|-----|----------------------|
| 激活撑爆显存（长 seq/大 batch） | activation checkpointing | -82.9%（§3），代价一遍重算 |
| 激活 + 带宽都紧 | MP + checkpoint 叠加 | 激活 /2 与 gather /2 独立生效（§3/§6） |
| 梯度归约不稳（loss spike） | `reduce_dtype=fp32` | 多付 1.3 MB/step 买稳定（§6.2） |
| 模型状态本身 | 别指望 MP | 16Ψ 不变（§5.3），回到 L2 的 ZeRO 阶梯 |

---

## 7. [5] 正交叠加：FSDP + 官方 activation checkpointing

`apply_activation_checkpointing`（`torch.distributed.algorithms._checkpoint.
checkpoint_wrapper:L239`，torch 2.13.0）是 FSDP1 的官方 AC 入口：`check_fn`
选中 `TransformerEncoderLayer`，FSDP 单元内部 forward 即被 checkpoint 包裹。
实测（batch=(8,16)，与 `[0]` 同形，外层 saved_tensors_hooks 计量）：

| 场景 | saved after fwd |
|------|-----------------|
| 单进程 fp32（[0] 参照） | 1,750,532 |
| 单进程 fp32 + checkpoint（[0] 参照） | 298,500 |
| FSDP FULL_SHARD fp32 | **1,750,532** |
| FSDP FULL_SHARD fp32 + AC | **298,500** |

两组数字**逐字节相同**（机器断言）——这就是「正交」的精确含义：

- ZeRO 切的是**模型状态**（params/grads/opt，16P → 16P/W），激活它碰都不碰：
  FSDP 包裹下的激活账本与单进程分毫不差（每个 rank 为自己的 microbatch 保存
  激活，分片不分到它头上）；
- checkpoint 切的是**激活**（保存点从层内中间量退到层边界），模型状态它碰都不碰；
- 两者叠加，每 rank 端到端 = **16P/W 的模型状态 + checkpoint 后的激活
  + unshard 瞬时峰值**（一个 block 的完整参数，L2 思考题 5 的测量点；MP 下这个
  峰值是 bf16——思考题 5 的 L3 版）。loss 侧 no-AC 与 AC 逐位一致
  （5.056958 == 5.056958，fp32 重算精确）。

ZeRO 论文把这件事放在 ZeRO-R（§6.1 partitioned activation checkpointing）：
把 checkpoint 的**重算计算**也分到各 rank（配合模型并行），论文结论是其通信增量
「in general less than one tenth of the baseline MP」。本节的数据并行形态不需要
那一层——每个 rank 重算自己的 microbatch，AC 不产生任何额外通信（`[4]` 的
comm 表在 AC 下不变，思考题 4 可验）。

---

## 8. [6] FSDP1 → FSDP2：同一个数学，新的契约

`[6]` 块真跑了 `torch.distributed._composable.fsdp.fully_shard`（FSDP2）：

- `type(param) = DTensor`：`fully_shard(model)` **原位**把 `model.parameters()`
  从 plain Tensor 变成 DTensor（官方 user contract，FSDP2 文档页原文：「fully_shard
  converts model.parameters() from plain torch.Tensor to DTensor in-place」），
  优化器必须建在 DTensor 参数上、step 也在 DTensor 上做；
- 本 rank local shard 元素总数 = 58,240 = P/2（逐参数 `_local_tensor` 求和，
  机器断言）——per-parameter sharding，不再拼 FlatParameter；
- 每 rank 状态 = 0.89 MiB = 16P/W，与 FSDP1 同量；`[7]` 的 loss 与 FSDP1-fp32
  **逐位相同**（4.865972/4.695927/4.533425，同为 7.153e-07 的归约舍入）——
  API 换了，数学一个字节没变；
- MP policy（`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`）下
  local shard dtype 仍是 fp32——与 FSDP1 同款语义（§6.1 的文档引文就是 FSDP2 的）。

API 形态差异（L2 §12 的承诺兑现）：FSDP1 是「包 wrapper」——`FSDP(module,
auto_wrap_policy=...)` 返回一个新模块；FSDP2 是「逐层 mark」——自底向上对每个
层调 `fully_shard`，最后对 root 调一次（文档：「User should apply fully_shard in
a bottom-up manner」），参数分组与通信单元由 mark 结构直接表达，与
`torch.compile` 友好（FSDP 论文 arXiv:2304.11277 记录了 FSDP1 的设计权衡，
compile 集成方向见 SimpleFSDP，arXiv:2411.00284，标题级引用）。

**FSDP1 deprecation 时间表（修订 L2 §12 的 `[TODO]`）**：截至 torch 2.13.0
（2026-08-11 现场核验两条通道）——docs `stable/fsdp.html`（重定向至 2.13 版，
HTTP 200）仍完整收录 FSDP1，页面内的 deprecated 字样均为**参数级**
（`ignored_modules` / `optim_input` 等），无类级 deprecation 声明；本机安装源码
里唯一的策略级 deprecation 是 `NO_SHARD`（`_init_utils.py:L441`）。同时 FSDP2
文档页明确写「If you are currently using FSDP1, consider migrating to FSDP2 using
our migration guide」。**结论：方向已定（FSDP2 是继任者）、迁移有官方指南，但
torch 2.13 没有公布 FSDP1 的移除时间表**——本教程继续用 FSDP1 教 ZeRO 分级
（`ShardingStrategy` 枚举把 stage 语义显式化），FSDP2 对照着学，两者语义一致。

一个本机实测的坑（写进溯源以免后人再踩）：macOS 上 `fully_shard` 不传 mesh 时，
`_init_default_mesh`（`_fsdp_init.py:L206`）自动探测设备走到
`torch.mps.is_initialized`（`device_mesh.py:L460`），本机 torch 2.13.0 build 无此
属性，AttributeError——解法是显式 `init_device_mesh('cpu', (W,))`（脚本已做）。
GPU 机器上默认 mesh 正常工作，标 `[TODO: verify on real system]`。

---

## 9. 取舍分析：nano 版 vs 权威实现（L3 硬性要求）

| 维度 | nano-fsdp L3 | 权威实现（FSDP / DeepSpeed / Megatron） | 差异原因 |
|------|--------------|------------------------------------------|----------|
| 激活计量 | saved_tensors_hooks 逻辑字节 | `torch.cuda.max_memory_allocated` / nvidia-smi（含 allocator 碎片、buffer、NCCL 缓冲） | nano 要可复现的精确账；真实显存还有账外项 |
| 重算调度 | 全 block checkpoint | Megatron selective recomputation（arXiv:2205.05198 §5，只重算 attention 段） | nano 演示机制；生产要「省得值」 |
| MP 通信计量 | 逻辑字节（dtype×元素数） | NCCL 实流量 + overlap/prefetch（FSDP streams、limit_all_gathers） | 逻辑口径才能对公式；物理口径是 GPU 攒批的事 |
| master weights | [2] 最小复现布局 | DeepSpeed fp32 分片 + bucket 化 + CPU offload（ZeRO-Offload 系） | nano 露语义；DeepSpeed 要吞吐与显存极限 |
| AC 与分片组合 | FSDP1 官方 API 直用 | FSDP2 + torch.compile 图级处理 / Megatron 与 TP/SP 复合 | 组合形态在演进，语义（正交叠加）不变 |
| 设备 | CPU/gloo（确定性、可复现） | GPU/NCCL | 机制同构，绝对数字不可外推 |

一句话：**L3 保留了「三条 dtype 通道 + 两笔正交账本」的全部骨架，剥掉的是
allocator/overlap/selective 三层皮肤**——读权威源码时，§4/§6 的行号锚点就是骨架
位置，皮肤为什么存在（吞吐、碎片、带宽）表里已逐条给出。

---

## 10. 费曼自检

### 讲给外行听：书桌、 shorthand 与账本

接 L1/L2 的搬书类比：参数 = 书，梯度 = 笔记，优化器状态 = 文具，ZeRO 把这三样
**分着放**（每人工位只留 1/W）。L3 加两件事：

- **书桌大小（激活）**：你读书时摊开在桌面上的纸页。ZeRO 分管的是书架（模型
  状态），桌面它管不着——batch 越大、书越厚，桌面越先爆。**checkpoint 就是
  只记页码不摊纸**：需要某页时重新翻到那里（重算一遍 forward），用跑腿换桌面。
  实测桌面从 1.75 MB 降到 0.30 MB（-82.9%），代价是每层多翻一遍。
- **shorthand 与账本（混合精度）**：笔记用速记写（bf16：快、省纸），但**总账
  必须用正楷**（fp32 master）——速记的笔画太粗，账目上 1 厘钱的改动
  （lr=1e-3 的更新）在速记里根本写不进去（实测参数冻结 5 步）。而且你会发现
  总账并没有因为速记变薄：16 页还是 16 页（2+2+4+4+4），只是换了一种分配——
  速记真正省的是**桌面**（激活减半）和**跑腿时的包裹体积**（all-gather 减半）。

类比边界：真实系统里「重新翻书」（重算）可以和别的阅读并行（GPU stream
overlap），类比里是串行的——所以「多一遍 forward」的时间代价在 GPU 上部分可藏；
selective recomputation 则是「只重翻最容易皱的那几页」。

### 反例

1. 「混合精度让显存减半。」——错。模型状态 16Ψ 不变（[2] 实测 mixed == fp32）；
   减半的是激活（[0]）与 all-gather 通信（[4]）。想减模型状态，回 L2 的 ZeRO 阶梯。
2. 「bf16 就是粗糙版 fp16，哪个精度高用哪个。」——错。两者失败方式不同：bf16
   尾数粗（吞更新，§5.1 实测），fp16 范围窄（梯度 underflow，要 loss scaling）；
   选 bf16 是选**动态范围**，不是选精度——且两者都需要 fp32 master。
3. 「checkpoint 包得越多越好。」——错。重算是真实计算（[0] body=4 vs 2）；
   2205.05198 §5 的 selective recomputation 就是为「省得值」而生。
4. 「FSDP MP 把参数存成 bf16。」——错（[3] 实测）。存储是 fp32，低精度只在计算
   与 gather 通道；fp32 shard 就是 master weights（FSDP2 文档原话，§6.1）。
5. 「forward hook 触发次数 = forward 执行次数。」——错（[0] hook=2 vs body=4）。
   非重入 checkpoint 的重算被 `_StopRecomputationError` 提前终止，hook 等不到
   forward 返回（§4 全机制）。

---

## 11. 思考题（都能动手验证）

1. 把 `mode_fsdp_mp` 的 `reduce_dtype` 改成 `torch.bfloat16`，先**预测** total
   通信字节（提示：rs 也减半 → 恰 50%），再跑一遍验证。想一想为什么生产上
   很少这么干（§6.2 的数值稳定账）。
2. 把 `LAYERS` 改成 8，预测 `[0]` 的 ckpt 节省比例会怎么变（用
   arXiv:1604.06174 的 O(√n) 结构推理：保存点从「所有中间量」退到「每层边界」，
   层内中间量占比随深度上升），跑一遍验证。
3. 给 `run_activation_combo` 的 `checkpoint(...)` 加 `use_reentrant=True`，预测
   `hook` 列变成几（提示：§4 的 L314 全量重算路径），跑一遍验证，并解释为什么
   PyTorch 把默认换成了非重入（梯度图结构 / 双重 backward 支持 / hook 语义）。
4. `[1]` 里把 lr 改成 0.01（更新量 > bf16 ULP），bf16 参数从第几步开始动？这个
   实验说明「精度不够」和「完全不训练」之间是什么关系？（提示：更新偶尔存活
   ≠ 训练正确——舍入噪声进了梯度方向。）
5. 在 `mode_fsdp_mp` 的 forward 中加一个测量点（unshard 后瞬间：一个 block 的
   完整参数驻留），量 fsdp3_mp 与 fsdp3_f32 的**峰值** resident 字节各是多少
   （L2 思考题 5 的 MP 版）。预测谁的峰值高、高多少（bf16 unshard vs fp32
   unshard；fp32 shard 常驻不变）。GPU 上的 `max_memory_allocated` 口径标
   `[TODO: verify on real system]`。

---

## 12. 下一步（02 轨满阶之后）

- **与 nano-megatron 合流**：ZeRO（切状态）与 TP/SP（切算子/切序列）是正交的
  两维——nano-megatron L3 实测了 SP 把激活按序列维切开（TP 未切区域激活恰
  1/t），与 L3 的 checkpoint/MP 三轴可以组合成完整的显存策略空间
  （HYBRID_SHARD + TP + SP + selective AC 是 7B+ 训练的常见配方）。
- **真机验证攒批**（Machine B 通道，独立活动）：GPU 上的
  `torch.cuda.max_memory_allocated` 全账（含 allocator 碎片）、NCCL 物理带宽下
  的 67.5% 通信节省兑现为多少吞吐、FSDP2 默认 mesh 在 GPU 的行为。
- **02 轨状态**：nano-fsdp L0–L3 阶梯完成（本节为满阶最后缺件），与
  nano-megatron L0–L3 并列为 02 轨两个满阶模块。

---

## 13. 溯源与不确定声明

**运行环境**：macOS（arm64），`python3`
（Python 3.13.13，torch 2.13.0），gloo backend，2 个真实进程，CPU。全部计算真跑
（真实 bf16/fp32 kernel、真实 gloo 集合通信、真实 FSDP1/FSDP2），无 mock；
GPU 绝对数字与吞吐结论标 `[TODO: verify on real system]`（Machine B 真机验证
通道，独立攒批）。

**输出锚点**：整行删除口径 `sed '/^[[:space:]]*elapsed/d'`（3 行计时浮动：
`elapsed[single-process]:` / `elapsed[distributed]:` / `elapsed:` 总计）；掩码后
输出 md5 = `5320c6d6b95a218e90d175e2a40f8245`（116 行；在 3 个独立
CWD 逐字节一致）；raw 119 行；指标 digest = `47e7ffd99c93628cea14f9feac4716e4`
（输出内印，跨遍一致）。

**scratch 出处**：`[0]` 的 SaveMeter 与 10:00 轮 scratch 实验同构——workspace
`9d88b729-2b4d-479d-a0d2-620d54968fb2/fsdpL3/exp1_local.py`（3,778 B，md5
`1a7030e3998039215fa6b5f5fc550735`，2026-08-11 记录；多次独立运行复验）。现场复跑与既有记录逐位吻合
（fp32 1,750,532→298,500 B / bf16 减半 / loss ckpt 前后逐位一致）；exp1b 的
grad_dtype 探针并入 `[1]`（修正了 scratch 中赋值在 try 外的瑕疵），exp1 的
fwd_calls 计数疑点（ckpt 下 hook 不增）由 §4 的源码机制解释。

**源码锚点**（torch 2.13.0 本机安装，全部 2026-08-11 现场核验）：

- checkpoint 机制：`torch/utils/checkpoint.py` L362（`early_stop: bool = True`
  默认）/ L1084（`class _StopRecomputationError`）/ L1131（pack 端 raise）/
  L1179-1185（unpack 端触发重算 + catch）/ L314（reentrant 全量重算
  `outputs = ctx.run_function(*detached_inputs)`）；`torch/nn/modules/module.py`
  L1884（`return inner()` → exception 分支）/ L1900（always_call hook 补跑循环，
  普通 hook 缺席）。
- FSDP1 混合精度：`torch/distributed/fsdp/_flat_param.py` L1307（`pre_unshard`）/
  L1333（`_use_low_precision_shard()` 调用位）/ L1342（def）/ L906-940
  （`_init_param_reduce_dtypes`：`_fwd_bwd_param_dtype` / `_reduce_dtype` 语义）；
  `torch/distributed/fsdp/_runtime_utils.py` L1030（`_cast_grad_to_param_dtype`，
  L917 调用位）。
- 官方 AC 入口：`torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py`
  L239（`apply_activation_checkpointing`）。
- FSDP2 macOS 坑：`torch/distributed/fsdp/_fully_shard/_fsdp_init.py` L206
  （`_init_default_mesh`）→ `torch/distributed/device_mesh.py` L460
  （`torch.mps.is_initialized` AttributeError，本机 build）。

**DeepSpeed**：`deepspeedai/DeepSpeed` master 分支
`deepspeed/runtime/zero/stage_1_and_2.py`（2026-08-11 经 raw.githubusercontent.com
抓取，实测 156,651 B / 3,153 行，git blob sha
`960854ad248cec11e5e573354568c158bcf2c6df`）：`class DeepSpeedZeroOptimizer`
L134、`partition_gradients` L222、「ZeRO-2 if partition_grads else ZeRO-1」L223
（L2 锚点行号级零漂移）；L3 新锚点：fp32 master 分片注释与构造 L486-490、
`single_partition_of_fp32_groups.append` L506、优化器 param_group 指向 fp32 分片
L511-513、`bf16_master_weights_and_gradients` / `bf16_optimizer_states` 标志
L178-179、reduce-scatter 合法 dtype L305。上游持续演进（08-08 抓取为 154,425 B /
blob `85dd6ffb…`，三日 +2,226 B），行号以抓取日为准、本轮逐一现场核验在位。

**arXiv**（全部经 export.arxiv.org API 2026-08-11 现场核验标题/日期）：

- 1910.02054「ZeRO: Memory Optimizations Toward Training Trillion Parameter
  Models」（Rajbhandari, Rasley, Ruwase, He；2019-10-04）。章节结构经 ar5iv
  现场抓取：§3.1 Model States（16Ψ 账本，引文见 §5.2）/ §3.2 Residual Memory
  Consumption（激活）/ §6.1 Partitioned Activation Checkpointing（ZeRO-R）/
  §7.2 ZeRO-DP Communication Volume（「no additional communication using P_os
  and P_g…a maximum of 1.5×」——L2 已实测复现）。
- 1604.06174「Training Deep Nets with Sublinear Memory Cost」（Chen et al.,
  2016-04-21）：摘要原话「costs O(sqrt(n)) memory to train a n layer network,
  with only the computational cost of an extra forward pass per mini-batch」。
- 2205.05198「Reducing Activation Recomputation in Large Transformer Models」
  （2022-05-10）：ar5iv 章节结构现场核验 §4.1（每层激活内存）/ §4.2.2
  （Sequence Parallelism）/ §4.3（总激活内存）/ §5（Selective Activation
  Recomputation，摘要原话「most of this redundant compute is unnecessary」）。
  节号与 nano-megatron L3 修订后口径（ar5iv 独立复抓）一致。
- 2304.11277「PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel」
  （2023-04-21）——FSDP 论文，本轮新录；2411.00284「SimpleFSDP: Simpler Fully
  Sharded Data Parallel with torch.compile」（2024-11-01）——标题级引用
  （compile 集成方向，内容未展开，不引其数字）。

**文档**（2026-08-11 抓取）：`docs.pytorch.org/docs/stable/fsdp.html`（重定向至
`/docs/2.13/fsdp.html`，HTTP 200，910,287 B）——FSDP1 无类级 deprecation（仅
参数级 + `NO_SHARD` 策略级，后者在本机源码 `_init_utils.py:L441`）；
`/docs/2.13/distributed.fsdp.fully_shard.html`（HTTP 200，764,215 B，页面标注
Created 2024-12-04 / Last Updated 2026-04-24）——FSDP2 user contract 与
MixedPrecisionPolicy 引文出处（§6.1/§8）。

**未覆盖**：真实 GPU 显存全账（allocator/buffer/碎片）、NCCL 物理带宽、
selective recomputation 实现、ZeRO-R 的 partitioned activation（需模型并行
配合）、HYBRID_SHARD 跨节点形态——真机攒批与后续模块处理。
