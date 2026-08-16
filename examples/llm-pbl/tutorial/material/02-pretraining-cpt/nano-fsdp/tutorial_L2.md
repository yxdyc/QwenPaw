# nano-fsdp L2 — ZeRO 分级：到底切了什么，代价是什么

> **核心机制**：ZeRO-0/1/2/3 的区别只有一个变量——**哪一类张量被分片**（参数 / 梯度 /
> 优化器状态）。分片对象每多一类，每 rank 显存按 L0 的公式下降，但通信量只在「切参数」
> 这一步才上涨（2P → 3P）。本节用 5 个真实多进程模式（真 gloo 集合通信，无 mock）把
> 显存与通信量**同时**量出来，并用单进程参照验证「任何 ZeRO stage 数学上等价于在全
> batch 上训练」。
>
> **运行要求**：`torch`（CPU 即可，本机实测 torch 2.13.0）。命令：
> ```bash
> python3 L2_zero_stages.py   # ~3s
> ```
> 真实生产在 GPU 上；CPU/gloo 跑的是**同一套 PyTorch 分布式代码路径**，机制相同，
> 绝对数字（显存、时延）按 GPU 放大。

---

## 1. 本节目标（K+1：从 L1 的「两档对比」到「整条阶梯」）

L0 手算了账本：Adam 训练 16 bytes/param，ZeRO 各 stage 的闭式公式。
L1 用真实 FSDP 量了两档：DDP（全副本 16P）与 FSDP FULL_SHARD（16P/W）。

L1 留下两个没回答的问题，正是 L2 的全部目标：

1. **中间两级（ZeRO-1/2）到底切了什么？** 只切优化器状态、再切梯度，各省多少？
2. **省的代价是什么？** 每个 stage 每步付多少通信量——这是「该切到哪一级」的决策依据。

L2 的答案用一张表预告（下面是机器验证，不是断言）：

| mode | 实现 | 分片对象 | 每 rank 模型状态 (fp32) | 通信量/step |
|------|------|----------|------------------------|-------------|
| `ddp` | 真实 torch DDP | 无 | 16P | 2P |
| `zero1` | 手写真实 ZeRO-1 | 优化器状态 | 8P + 8P/W | 2P |
| `zero2` | 手写真实 ZeRO-2 | + 梯度 | 4P + 12P/W | 2P |
| `fsdp2` | 真实 FSDP `SHARD_GRAD_OP` | + 梯度（ZeRO-2 血统） | 16P/W（稳态） | 2P |
| `fsdp3` | 真实 FSDP `FULL_SHARD` | 全部（ZeRO-3） | 16P/W（稳态） | ≈3P |

两个先剧透的 senior 级观察，下面逐一验证：

- **ZeRO-1/2 的显存节省是通信免费的**（都是 2P，与 DDP 相同）；只有切到参数（ZeRO-3）
  才多付 ~1.5× 通信。这就是 ZeRO 论文（arXiv:1910.02054）的核心通信量结论，我们亲手复现。
- **`fsdp2` 与 `fsdp3` 的稳态每 rank 存储完全相同**——它们的差异不在「存多少」，
  而在「峰值驻留多少」与「通信多少」。多数人以为 SHARD_GRAD_OP「切得少所以存得多」，
  实测恰恰不是。

---

## 2. 先跑起来

```bash
python3 \
  tutorial/material/02-pretraining-cpt/nano-fsdp/L2_zero_stages.py
```

输出（完整 79 行；计时行——以 `elapsed` 开头的 6 行：5 行 `elapsed[mode]:` + 1 行
`elapsed:` 总计——随机器浮动，
锚点口径为掩码后 md5：`sed '/^[[:space:]]*elapsed/d'`，见 §13）：

```
========================================================================
nano-fsdp L2 — ZeRO stages: what gets sharded, what it costs
========================================================================
TinyLM: vocab=128 dim=64 layers=2 | P = 116,480 params | fp32
cluster: W = 2 real processes, gloo backend, CPU (真实多进程 + 真实集合通信；生产在 GPU，机制相同)
steps: 1 warmup + 3 measured | seed = 7 (与 L1 一致)

[ref] single-process ground truth: losses = 4.8660 4.6959 4.5334

[0] per-rank MODEL STATE after training（params+grads+Adam, fp32; L0/L1 口径）
    mode       params     grads       opt |     total formula (fp32)    expected
    ------------------------------------------------------------------------------
    ddp         0.44M     0.44M     0.89M |     1.78M 16P                  1.78M
    zero1       0.44M     0.44M     0.44M |     1.33M 8P + 8P/W            1.33M
    zero2       0.44M     0.22M     0.44M |     1.11M 4P + 12P/W           1.11M
    fsdp2       0.22M     0.22M     0.44M |     0.89M 16P/W (steady)       0.89M
    fsdp3       0.22M     0.22M     0.44M |     0.89M 16P/W (steady)       0.89M
    （M = MiB；16P = 1.78 MiB，与 L1 的 DDP/FSDP 数字一致；opt 列含 Adam step 计数器的每参数张量几 B 零头）

[1] per-step COMM VOLUME（口径：all-reduce=2P, reduce-scatter=P, all-gather=P）
    mode    collectives/step                         volume     x P
    ----------------------------------------------------------------
    ddp     hook_all_reduce=1                      0.932 MB  2.000x
    zero1   gather=1 rs=1                          0.932 MB  2.000x
    zero2   gather=1 rs=1                          0.932 MB  2.000x
    fsdp2   gather=3 rs=3                          0.932 MB  2.000x
    fsdp3   gather=5 rs=3                          1.332 MB  2.858x

[2] correctness：五个分布式模式 vs 单进程参照（同一数学，不同切法）
    mode    step losses (全 batch)         max|Δparams| vs ref
    ----------------------------------------------------------
    ddp     4.8660 4.6959 4.5334                  1.760e-04  inter-rank Δ=0.0e+00
    zero1   4.8660 4.6959 4.5334                  5.740e-05  inter-rank Δ=0.0e+00
    zero2   4.8660 4.6959 4.5334                  5.740e-05  inter-rank Δ=0.0e+00
    fsdp2   4.8660 4.6959 4.5334                  5.740e-05
    fsdp3   4.8660 4.6959 4.5334                  5.740e-05
    ref     4.8660 4.6959 4.5334               (ground truth)

[3] timing（CPU/gloo 仅供相对比较；生产在 GPU，见 tutorial §8）
    elapsed[ddp]:      6.4 ms/step
    elapsed[zero1]:      8.3 ms/step
    elapsed[zero2]:      7.6 ms/step
    elapsed[fsdp2]:     11.1 ms/step
    elapsed[fsdp3]:     11.6 ms/step

[4] self-check
    PASS  ddp per-rank state 1863792 within 1.5% of 16P=1863680
    PASS  zero1 per-rank state 1397764 within 1.5% of 8P + 8P/W=1397760
    PASS  zero2 per-rank state 1164804 within 1.5% of 4P + 12P/W=1164800
    PASS  fsdp2 per-rank state 931852 within 1.5% of 16P/W (steady)=931840
    PASS  fsdp3 per-rank state 931852 within 1.5% of 16P/W (steady)=931840
    PASS  memory ladder monotone: ddp > zero1 > zero2 > fsdp*
    PASS  fsdp2/fsdp3 steady-state storage identical (差异在峰值与通信，不在稳态)
    PASS  ddp comm volume == 2P (931840 B)
    PASS  zero1 comm volume == 2P (931840 B)
    PASS  zero2 comm volume == 2P (931840 B)
    PASS  fsdp2 comm volume == 2P (931840 B)
    PASS  fsdp3 comm volume ~3P (measured 2.858x P)
    PASS  fsdp3 calls: gather=5==2*blocks+1, rs=3==units
    PASS  fsdp2 calls: gather=3==units, rs=3==units
    PASS  ddp step losses match reference
    PASS  ddp final params within 5e-4 of reference
    PASS  zero1 step losses match reference
    PASS  zero1 final params within 5e-4 of reference
    PASS  zero2 step losses match reference
    PASS  zero2 final params within 5e-4 of reference
    PASS  fsdp2 step losses match reference
    PASS  fsdp2 final params within 5e-4 of reference
    PASS  fsdp3 step losses match reference
    PASS  fsdp3 final params within 5e-4 of reference
    PASS  ddp ranks hold identical params
    PASS  zero1 ranks hold identical params
    PASS  zero2 ranks hold identical params
    PASS  sum of FSDP shards across ranks == full replica (16P)
    ✅ self-check passed (28/28)

digest(md5 of metrics) = e6ec3bc5b4b8fca17aac7fe35d907d79

elapsed: 2.6s total (计算真跑; 计时行浮动, 锚点口径见 tutorial §2)
```

28 项 self-check 全绿。下面逐块拆。

---

## 3. 五个模式：只有一个变量在变

脚本用同一个 TinyLM（与 L1 逐字相同，P = 116,480）、同一份每 rank 数据
（seed = SEED+rank）、同一个 Adam，只改「状态怎么分布到 W=2 个进程」：

- `ddp`：真实 `DistributedDataParallel`。每 rank 全量参数/梯度/Adam 状态，
  反向时 all-reduce 梯度。ZeRO-0。
- `zero1` / `zero2`：**手写**的真实 ZeRO-1/2——真实多进程、真实
  `reduce_scatter` / `all_gather`（gloo 上真跑），只是把 DeepSpeed 的 bucket/overlap
  工程剥掉，露出 stage 语义本身（§5 逐行走读）。
- `fsdp2`：真实 `FSDP(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)`。
  FSDP 官方文档对它的定义（`torch/distributed/fsdp/api.py:L32-64`，torch 2.13.0）：
  「Gradients and optimizer states are sharded during computation, and additionally,
  parameters are sharded outside computation」——计算时（forward/backward）参数是全的，
  计算之外参数也切片。这就是 ZeRO-2 血统。FSDP 内部甚至直接管它叫 zero2：
  相邻的 `_HYBRID_SHARD_ZERO2` 策略文档写明「This is like HYBRID_SHARD, except...
  the unsharded parameters are not freed after the forward pass」（docstring api.py:L59-62，枚举成员 api.py:L69）。
- `fsdp3`：真实 `FSDP(sharding_strategy=ShardingStrategy.FULL_SHARD)`，ZeRO-3。
  与 L1 相同的 per-layer auto-wrap（每个 `TransformerEncoderLayer` 单独成 FSDP 单元，
  外加 root 单元 = embed+norm+head，共 3 个单元）。

为什么 ZeRO-0 用真 DDP 而不是 FSDP `NO_SHARD`？两者梯度同步语义相同
（`NO_SHARD` 的梯度路径就是 `dist.all_reduce`，`_runtime_utils.py:L932-940`），
而 DDP 是学习者实际会用的 API。我们用官方 `register_comm_hook` 截获它的
all-reduce 做精确计量（DDP 的集合通信在 C++ 侧发起，python 层拦不到，comm hook
是官方计量点）。

**计量原理（脚本的 CommMeter）**：FSDP1 运行时通过 `dist.xxx(...)` 模块属性调用
集合通信，替换 `torch.distributed` 的对应属性即可全量截获真实调用（不是模拟——
拦截的就是真流量）。体积口径按 ZeRO 论文：all-reduce(P) 记 2P（ring 上等价于
reduce-scatter + all-gather），reduce-scatter(P)、all-gather(→P) 各记 P。
DDP 走 comm hook 单独计。两条路径的数字都与理论公式对上了（§4/§5）。

---

## 4. 显存结果：阶梯为什么长这样

### 4.1 实测 vs 公式（fp32 口径）

`[0]` 块的五行就是 L0 公式的 fp32 版实测（P = 116,480，W = 2，单位 MiB）：

| mode | 实测 total | 公式 (fp32) | 数值 |
|------|-----------|-------------|------|
| ddp | 1.78M | 16P | 1,863,680 B |
| zero1 | 1.33M | 8P + 8P/W | 1,397,760 B |
| zero2 | 1.11M | 4P + 12P/W | 1,164,800 B |
| fsdp2 | 0.89M | 16P/W（稳态） | 931,840 B |
| fsdp3 | 0.89M | 16P/W（稳态） | 931,840 B |

实测与公式误差 ≤ 112 B（Adam 的 `step` 计数器：每个参数张量 4 B；DDP 有 28 个
参数张量，FSDP 有 3 个 FlatParameter）。self-check 以 1.5% 容差断言全部命中。

### 4.2 L0 是 fp16 口径，这里是 fp32——映射表

L0 的公式（ZeRO-1 = 4P+12P/W 等）假设 mixed-precision：fp16 参数 2P + fp16 梯度 2P +
fp32 master/m/v 12P。本节全程 fp32（params 4P + grads 4P + m/v 8P），**fp32 里参数
本身就是 master**，于是「opt 列」从 12P 缩到 8P，「params 列」从 2P 涨到 4P：

| stage | L0 口径（fp16） | L2 口径（fp32） | 为什么 |
|-------|----------------|-----------------|--------|
| ZeRO-0 | 16P | 16P | 总数不变 |
| ZeRO-1 | 4P + 12P/W | 8P + 8P/W | 只切 opt：fp16 的 opt 是 12P，fp32 的是 8P |
| ZeRO-2 | 2P + 14P/W | 4P + 12P/W | 切 opt+grad：grad 在 fp16 是 2P，fp32 是 4P |
| ZeRO-3 | 16P/W | 16P/W | 全切：口径无关 |

看到 16P/W 在两种口径下相同不是巧合——ZeRO-3 把**所有**模型状态均匀切片，
与每字节是什么 dtype 无关。

### 4.3 两个 senior 级细节

**fsdp2 与 fsdp3 稳态存储相同**（实测都是 931,852 B）。`SHARD_GRAD_OP` 的名字让人
以为「参数不切 → 存得多」，但 FSDP 文档写得清楚：参数只是「计算期间」不切——
backward 结束照样 reshard，步与步之间参数仍是 1/W 的 shard。差异在别处（§5/§7）：
峰值驻留与通信量。

**FSDP 会 padding**。每个 FSDP 单元的 FlatParameter 长度要能被 W 整除
（`_runtime_utils.py:L890-905` 的 `_get_reduce_scatter_tensors` 做 pad）。本节
三个单元（49,984 / 49,984 / 16,512 元素）恰好都是偶数，padding 为 0——所以
fsdp 实测 = 931,840 + 12 B 零头，分毫不差。换个不能整除的维度就会出现 padding 零头，
这正是 nano-verl L3 里「27,966 + 2 pad」的同款机制。

---

## 5. 通信量结果：ZeRO-1/2 是免费午餐，ZeRO-3 要加钱

### 5.1 实测：2P / 2P / 2P / 2P / 2.858P

`[1]` 块（每行都是真实截获的集合通信，按 §3 口径折算）：

- `ddp`：1 次 all-reduce（P 个元素的梯度 bucket，comm hook 实测）→ 2P。
- `zero1` / `zero2`：1 次 reduce-scatter（完整梯度，P）+ 1 次 all-gather（更新后
  参数 slice 拼回，P）→ 2P。**注意**：这与 DDP 的单次 all-reduce **同量**——
  all-reduce 本来就是 reduce-scatter + all-gather 的合成。ZeRO-1/2 只是把
  「all-reduce 的后半段 all-gather」换成了「all-gather 参数」，总量不变。
  这就是 ZeRO 论文的结论：切优化器状态、切梯度，通信量与标准数据并行相同。
- `fsdp2`：每单元 1 次 all-gather（forward 前取回参数）+ 1 次 reduce-scatter
  （backward 后切梯度）= 3+3 次、合计 2P——与手写 zero2 **逐字节同量**。
- `fsdp3`：5 次 all-gather + 3 次 reduce-scatter = **2.858P ≈ 3P**。多出来的
  那组 all-gather 就是「切参数」的账单：FULL_SHARD 在 forward 后把参数 reshard
  （释放显存），backward 前得再 gather 一次。

### 5.2 为什么是 2.858 而不是 3.000：root 单元每步只 gather 一次

脚本把每次集合通信都记了下来，`fsdp3` 每步的完整序列（元素数）：

```
gather:16512(root)  gather:49984(b1)  gather:49984(b2)     ← forward
gather:49984(b2)    gather:49984(b1)                        ← backward（重新 gather）
rs:49984(b2)        rs:49984(b1)        rs:16512(root)      ← 梯度 reduce-scatter
```

block 单元 gather 两次（forward + backward 各一次），**root 单元只 gather 一次**——
root 是最外层模块，forward 取回后跨过 backward 一直驻留（reshard 发生在 backward
结束时，见 `_runtime_utils.py:L277-331` 的 `_unshard`/`_reshard`）。所以：

```
volume = (2×blocks + root) + (blocks + root)
       = (2×99,968 + 16,512) + 116,480 = 332,928 元素 = 2.858 × P
```

模型越深（blocks 占比越大），这个数越逼近 3.000。2 层的 TinyLM 让 root 占了 14%，
所以偏差肉眼可见——**这正是一个好教材案例：公式的渐近行为与有限尺寸修正都能量出来**。

### 5.3 决策表（L2 的工程落点）

| 瓶颈在哪 | 选 | 理由（用上面的数字） |
|----------|-----|----------------------|
| 优化器状态撑爆显存（Adam 占 8/16 B/param） | ZeRO-1 | 省 8P→8P/W，通信 0 增长 |
| 梯度 buffer 也撑爆 | ZeRO-2 / `SHARD_GRAD_OP` | 再省梯度，通信仍 0 增长 |
| 参数本身单卡放不下 | ZeRO-3 / `FULL_SHARD` | 唯一选择，付 1.5× 通信 |
| 跨节点带宽紧张 | `HYBRID_SHARD` | 节点内 FULL_SHARD + 节点间复制，把 3P 关在 NVLink 内（api.py:L55-58 文档原文） |

外推到 7B/Adam/fp16（L0 公式）：ZeRO-2 每卡 = 2P + 14P/W = 14 + 12.25 = 26.2 GB @8卡；
ZeRO-3 = 14 GB @8卡——省 12 GB 的代价是每步通信 28 GB → 42 GB（2P→3P，fp16 字节计）。
这 50% 的带宽在 NVLink 上常常可以接受，在以太网集群上就是训练速度的天花板。

---

## 6. 手写 ZeRO-1/2：stage 之间只差一个布尔值

`ZeROShardedAdam`（L2_zero_stages.py）每步四步，全部真实集合通信：

```python
# 1. backward 产生完整梯度（autograd 必然如此——这是 ZeRO-1/2 共同的瞬时峰值）
loss.backward()
# 2. 拍平 → reduce-scatter：每 rank 拿到自己 slice 的求和梯度
dist.reduce_scatter(self.grad_shard, list(g_flat.chunk(W)))
self.grad_shard.div_(W)          # sum → mean，对齐 DDP 的平均语义
# 3. 只在自己的 slice 上跑真实 torch.optim.Adam（状态只有 1/W）
self.shard.grad = self.grad_shard; self.opt.step()
# 4. all-gather 各自更新的 slice，拼回完整参数
dist.all_gather(parts, parts[self.rank])
```

**ZeRO-1 与 ZeRO-2 的唯一区别**：`keep_full_grads` 一个布尔值——ZeRO-1 在
reduce-scatter 后仍保留完整梯度（resident 4P），ZeRO-2 立即释放、只留 1/W 的 shard
（resident 4P/W）。这就是 `[0]` 块里 zero1 与 zero2 的 grads 列 0.44M vs 0.22M。

这不是我发明的简化——**DeepSpeed 就是这么实现的**：stage 1 与 stage 2 共用同一个
`DeepSpeedZeroOptimizer` 类（`deepspeed/runtime/zero/stage_1_and_2.py:L134`，
master 分支 2026-08-08 抓取），区分两者的是一个构造参数：

```python
# deepspeed/runtime/zero/stage_1_and_2.py:L222-223
self.partition_gradients = partition_grads
self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"
```

手写版与 DeepSpeed 的真实差异（都在工程层，不在机制层）：DeepSpeed 按 bucket 在
backward **过程中**做 reduce-scatter（与计算 overlap，也压低完整梯度的峰值驻留）、
支持 fp16 通信与 predivide/postdivide 数值技巧；手写版是步级串行 fp32，为的是把
stage 语义裸露出来。

---

## 7. 对照 FSDP 源码：测出来的序列对应哪几行

以下行号锚点基于 torch 2.13.0（本机安装与 GitHub `v2.13.0` tag 逐字节一致，
4 个文件 md5 见 §13）：

| 测到的行为 | 源码位置（pytorch v2.13.0） |
|-----------|------------------------------|
| 参数 unshard（forward/backward 前的 all-gather） | `torch/distributed/fsdp/_runtime_utils.py:L277` `_unshard` → `_flat_param.py:L1440` `_all_gather_flat_param` |
| CPU 上走 list 版 `dist.all_gather`（GPU 走 `all_gather_single`） | `_flat_param.py:L1466-1476`（L1466 注释原文：「HACK this should be handled by C10D」） |
| 梯度 reduce-scatter（post-backward hook） | `_runtime_utils.py:L858` `dist.reduce_scatter_single(...)` |
| reduce-scatter 的 padding | `_runtime_utils.py:L890-905` `_get_reduce_scatter_tensors` |
| `NO_SHARD` 梯度 all-reduce（= 我们的 ddp 档） | `_runtime_utils.py:L932-940` `_reduce_grad_no_shard` |
| reshard 时机（FULL_SHARD forward 后 / SHARD_GRAD_OP backward 后） | `_runtime_utils.py:L310` `_reshard`；策略语义 `api.py:L32-64` |
| 集合通信 python 入口 | `torch/distributed/distributed_c10d.py`：`all_reduce:L3156` / `all_gather:L4192` / `all_gather_single:L4292` / `reduce_scatter:L4790` / `reduce_scatter_single:L4847` |

两处值得盯着看：

1. **`SHARD_GRAD_OP` 的文档语义与实测完全咬合**（api.py:L43-49 原文）：
   「unshards before the forward, does not reshard them after the forward, and only
   reshards them after the backward computation」——所以它每单元每步只有 1 次
   all-gather（§5.2 序列里 fsdp2 是 gather=3），而 FULL_SHARD 是「reshards after the
   forward, unshards before the backward」（api.py:L37-42）→ 每 block 2 次。
2. **CPU 与 GPU 走不同的集合通信 API**（`_flat_param.py:L1466` 的 HACK）：CPU 用
   list 版 `all_gather`，GPU 用单张量 `all_gather_single`。语义相同、API 不同——
   任何在 CPU 上做的 FSDP 实验（包括本节）都要意识到这条分叉；计量代码里两种
   入口都得拦（CommMeter 的 OPS 表）。

---

## 8. 正确性：五种切法，同一份数学

`[2]` 块是 L2 最值钱的一张表：五个分布式模式与**单进程参照**（无分布式，直接在全
batch = 两个 rank 数据的拼接上训练）逐步 loss 完全一致（4.8660 / 4.6959 / 4.5334），
最终参数 max|Δ| ≤ 1.76e-4。

这机器验证了一个 senior 必须内化的事实：**数据并行（任何 ZeRO stage）在数学上就是
「在 union batch 上训练」**——梯度在各 rank 分别对本地 batch 求平均，跨 rank 求和再
除以 W，恰好等于全 batch 的平均梯度。ZeRO 切的是**状态存储与通信编排**，不动这个
等式。

Δ 为什么不是 0？fp32 加法不满足结合律：ref 一次 backward 累加 8 个 token 梯度，
DDP 是「各 rank 累加 4 个 → ring 上相加 → ÷2」，FSDP 是 reduce-scatter 的求和树——
**不同的归约结构 = 不同的舍入路径**。实测 ddp Δ=1.76e-4、其余四档 Δ=5.74e-5，
量级一致但数值不同，正是归约结构不同的指纹。这与 nano-verl L3 的 [4a]/[4b] 切分
同源：bit 级不变性只在「同一归约结构」内成立（本脚本 inter-rank Δ=0.0e+00 就是
同结构的两 rank），跨结构只有「收敛可比」。别指望跨并行策略逐位复现——但
O(1) 的偏差一定意味着数学错了（调试时先用这个表自查）。

计时（`[3]`，CPU/gloo）：ddp 6.4 < zero1/zero2 ~8 < fsdp2 11.1 < fsdp3 11.6 ms/step。
趋势与通信量/编排开销一致，但**绝对值不可外推到 GPU**：生产环境里 all-gather 与
计算 overlap（FSDP 的 backward prefetch 等），ZeRO-3 的 1.5× 通信大部分可以藏起来；
CPU/gloo 没有这种 overlap，开销全裸。所以本节只声明相对趋势，GPU 吞吐结论标
`[TODO: verify on real system]`。

---

## 9. 取舍分析：nano 版 vs 权威实现（L2 硬性要求）

| 维度 | nano-fsdp L2 | 权威实现（FSDP1 / DeepSpeed） | 差异原因 |
|------|--------------|-------------------------------|----------|
| 梯度归约时机 | 步级（backward 后一次性） | bucket 级、与 backward overlap（DDP bucket / DeepSpeed `overlap_comm`） | nano 要裸露语义；权威要吞吐 |
| 通信/计算重叠 | 无（串行） | 有（独立 stream + prefetch） | 同上；这也是 GPU 上 ZeRO-3 不慢 1.5× 的原因 |
| padding | 手写版假设整除（assert） | FlatParameter 自动 pad 到 W 倍数 | nano 用断言把复杂度挡在门外，FSDP 必须处理任意形状 |
| dtype | 全程 fp32 | mixed precision（fp16/bf16 计算 + fp32 累积）、通信 predivide/postdivide | 数值稳定性工程，L3 主题 |
| ZeRO-1 | 手写真实实现 | FSDP 无原生 ZeRO-1 档（只有 SHARD_GRAD_OP≈ZeRO-2 / FULL_SHARD≈ZeRO-3）；DeepSpeed 有 stage 1 | FSDP 的设计重心在「参数也切」；只切 opt 的场景 DeepSpeed 覆盖 |
| 激活显存 | 不计 | activation checkpointing 等 | ZeRO 只切模型状态，激活是另一笔账（L3） |

一句话：**nano 版保留了 stage 语义的全部骨架（切什么、通信多少、正确性等价），
剥掉的是 overlap / padding / 数值工程三层皮肤**。读权威源码时，先认出骨架（§7 的
锚点就是骨架位置），再理解皮肤为什么存在。

---

## 10. 费曼自检

### 讲给外行听：搬书升级版

一个图书馆要抄一套书（训练一个模型）。书 = 参数，笔记 = 梯度，文具 = 优化器状态。

- **DDP**：两个抄写员各备一套完整的书+笔记+文具，各抄各的，定期互相对答案。
  桌子都占满（16P），但对答案一次就够（2P 通信）。
- **ZeRO-1**：书和笔记仍人手一套，但**文具分着放**——每人只保管自己负责那几章的
  文具。用完交换成品（all-gather 参数）。桌子省了文具的一半，**跑腿次数没变**（2P）。
- **ZeRO-2**：笔记也分——新写的笔记只留自己负责章节的，其余当场交出去
  （reduce-scatter 后释放）。桌子再省一截，跑腿仍不变（2P）。
- **ZeRO-3**：**书本身也拆成两半分放**。谁要读某一章，先得从两人那里凑齐
  （forward all-gather），读完放回；校订（backward）时再凑一次——跑腿多了 50%（3P）。
  桌子最省（16P/W），但「借书」成了新开销。

类比边界：真实系统里「借书」（通信）可以和「抄写」（计算）同时进行（overlap），
类比里是串行的——所以类比预言「ZeRO-3 慢 50%」，真实 GPU 上通常远小于 50%。

### 反例

1. 「ZeRO stage 越高越好。」——错。ZeRO-3 付 1.5× 通信；显存够用时 ZeRO-2 往往更快
   （本机实测 fsdp2 11.1 < fsdp3 11.6 ms/step；GPU 上差距取决于 overlap 程度）。
2. 「SHARD_GRAD_OP 切得少，所以每 rank 存得多。」——错。两者稳态存储逐字节相同
   （实测均 931,852 B）；差异在峰值驻留（SHARD_GRAD_OP 让整层参数跨 forward+backward
   驻留）与通信（2P vs 2.858P）。
3. 「手写 ZeRO-2 和 FSDP SHARD_GRAD_OP 是两种东西。」——错。每步通信构成完全相同
   （gather P + reduce-scatter P，实测同量）；差的只是 bucket/overlap/padding 工程。

---

## 11. 思考题（都能动手验证）

1. 把 `WORLD_SIZE` 改成 4（`mp.spawn(nprocs=4)`），先**预测**再实测：zero1 与 fsdp3
   的每 rank total 各变成多少？通信量/step 变不变？（提示：通信量的 leading term
   与 W 无关——这是 ZeRO 论文的另一结论，量一下 2P/3P 是否纹丝不动。）
2. fsdp3 测到 2.858×P 而非 3.000×P。把 `LAYERS` 改成 8，预测新数值
   （用 §5.2 的公式手算），再跑一遍验证。
3. 在 `ZeROShardedAdam.reduce_grads` 里注释掉 `div_(W)`，loss 曲线会变成什么？
   （DDP/FSDP 内部也有对应的 predivide/postdivide，见 `_runtime_utils.py:L852/L879`
   的 `_gradient_predivide_factor` / `_gradient_postdivide_factor`。）
4. L1 的 FSDP 没用 per-layer auto-wrap（只有 root 一个单元）。预测它的每步通信量
   是多少，改 L1 代码加上 CommMeter 验证——为什么「只包 root 的 ZeRO-3」通信量
   退化成了 2P？（root 每步只 gather 一次，§5.2 的机制。）
5. 本表测的是**稳态**模型状态。要测**峰值**（forward 中某一刻：一个单元的完整参数
   + 其余 shard），测量点应加在哪？预测 fsdp2 与 fsdp3 谁的峰值高、高多少
   （一个 block = 49,984×4 B ≈ 0.19 MiB）。

---

## 12. 下一步（L3）

- **FlatParameter 内部**：参数怎么被拍平、拼接、padding（`_flat_param.py`），
  为什么 FSDP 的 `parameters()` 返回的是 shard 而不是原参数。
- **mixed precision**：fp16/bf16 计算 + fp32 累积如何改写 16× 账本
  （`MixedPrecision` 配置），通信 dtype 与 predivide/postdivide 的数值技巧。
- **activation checkpointing**：ZeRO 不解决的另一半显存（激活），与 ZeRO 正交叠加。
- **FSDP1 → FSDP2**：torch 2.13 同时提供 composable 的
  `torch.distributed._composable.fsdp.fully_shard`（FSDP2）；本教程用 FSDP1 是因为
  它的 `ShardingStrategy` 枚举把 ZeRO 分级直接显式化（且与 L1 衔接）。FSDP2 的
  分片语义与 FSDP1 一致，API 从「包 wrapper」变为「逐层 mark」——L3 对照讲。
  `[TODO: verify FSDP1 的官方 deprecation 时间表]`

---

## 13. 溯源与不确定声明

**运行环境**：macOS（arm64），`python3`（Python 3.13.13，
torch 2.13.0），gloo backend，2 个真实进程，CPU。全部计算真跑，无 mock；
显存只计模型状态（params+grads+Adam m/v，与 L0/L1 口径一致），不含
activations/buffer/OS 开销。GPU 上的绝对数字与吞吐结论标
`[TODO: verify on real system]`（真实 GPU/多机环境，后续验证）。

**输出锚点**：掩码口径 `sed '/^[[:space:]]*elapsed/d'`（6 行计时浮动：5 行
`elapsed[mode]:` + 1 行 `elapsed:` 总计）；掩码后输出 md5 =
`73e551e4071cdbc99a33dfcf74d55d1c`（连续运行 3 遍，输出逐字节一致）；
指标 digest = `e6ec3bc5b4b8fca17aac7fe35d907d79`（输出内印，跨遍一致）。

**源码锚点**（全部 2026-08-08 现场核验）：

- PyTorch：本机 torch 2.13.0 安装与 GitHub `pytorch/pytorch v2.13.0` tag 逐字节一致
  （md5：`api.py` = `e99fd345ac06588e3921860a5179e468`，`_runtime_utils.py` =
  `e29b685499d6e32c2f2036adfd97cbc5`，`_flat_param.py` = `1453f4283997568cf173f4c53cd82b46`，
  `distributed_c10d.py` = `310583209071c4880abe2456b3e3721a`），行号锚点见 §7 表，
  对 tag 与本机安装同时有效。
- ZeRO 论文：arXiv:1910.02054「ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models」（Rajbhandari, Rasley, Ruwase et al.），标题/作者经
  export.arxiv.org API 逐词核验。通信量口径（all-reduce≡2P、ZeRO-1/2 = baseline、
  ZeRO-3 = 1.5×）为论文通信量分析的标准结论，本脚本实测复现（2P/2P/2P/2P/2.858P）。
- FSDP 文档：`https://docs.pytorch.org/docs/stable/fsdp.html`（HTTP 200，2026-08-08）。
- DeepSpeed：`deepspeedai/DeepSpeed` master 分支
  `deepspeed/runtime/zero/stage_1_and_2.py`（2026-08-08 抓取，实测 154,425 B / 3,116 行，
  git blob sha `85dd6ffb46b5eac7f24fca917cee822bc6b4c6f4`；尺寸以抓取时点为准）：
  `class DeepSpeedZeroOptimizer` L134、`partition_gradients` L222、
  「ZeRO-2 if partition_grads else ZeRO-1」L223（同日复核在位，逐字有效）。master 在持续演进，
  行号以抓取日为准。勘误：此处初版误录「50,481 B」，系抓取被截断后把截断长度录成文件尺寸
  （锚点行均在前 50 KB 内，行号不受影响）；已经 api.github.com 通道独立重抓改正。
- 通信量是**逻辑体积**（leading term），不是物理线上字节数；ring 实现还有
  (W−1)/W 系数，W=2 时恰为 1/2，两种口径的相对结论一致。
- 未覆盖：真实 GPU 显存（nvidia-smi 口径）、mixed precision、activation、
  HYBRID_SHARD 跨节点形态、FSDP2 对照——L3 与真实 GPU/多机验证处理。
