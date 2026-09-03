# L2：两进程数据并行的 exact resume、rank 绑定与回滚

> **核心问题**：单进程已经能 exact resume，为什么一进入 data parallel，仍可能“checkpoint 都在、恢复却错了”？
> **先修**：[L1 真实小 Transformer 生命周期](tutorial_L1.md)，尤其是 optimizer/scheduler/RNG/data cursor。
> **新增约束**：global batch、梯度平均、rank-local sampler/RNG、checkpoint rank/world identity。
> **运行**：Python 3.10+、PyTorch（含 `torch.distributed`/gloo）、CPU、loopback；当前实验约 8 秒。
> **验收**：10/10 checks；连续与 resume/clean replay 的最大参数差均为 0；错误 rank shard fail closed。
> **边界**：这是真实两进程 gloo，不是 GPU/NCCL、FSDP 参数分片或吞吐 benchmark。

---

## 1. 从 L1 到 L2，只多一个 rank 吗？

不是。L1 的状态合同是：

$$
S_t = (\theta_t, O_t, H_t, R_t, C_t, t),
$$

其中 $\theta$ 是模型，$O$ 是 optimizer，$H$ 是 scheduler，$R$ 是随机状态，$C$ 是数据 cursor。
两路 data parallel 后，合同变成：

$$
S_t^{\mathrm{DP}}
=
\left(
\theta_t,
O_t,
H_t,
\{R_t^{(r)}, C_t^{(r)}\}_{r=0}^{W-1},
W,
t
\right).
$$

关键不是把同一份文件复制 $W$ 次，而是明确哪些状态被复制、哪些状态属于 rank：

| 状态 | 本实验中的布局 | 恢复时的风险 |
|---|---|---|
| model / optimizer / scheduler | DP rank 间数值复制 | 漏项会让所有 rank 一起偏离 |
| Python / torch RNG | 每 rank 单独保存 | dropout、采样等随机流可能错位 |
| sampler epoch/cursor | 每 rank 单独保存 | rank 会继续消费错误的数据位置 |
| rank / world size | checkpoint 身份元数据 | 错文件若静默加载，错误很难从 loss 发现 |
| consumed tokens | per-rank 账 + global 派生账 | 少乘 $W$ 会低报全局训练量 |

因此，本节刻意称它为 **rank-local checkpoint set**，不称“FSDP sharded checkpoint”。这里每个
rank 文件仍含完整 model/optimizer tensor；真正按 tensor 分片的 checkpoint 请到
[nano-fsdp](../nano-fsdp/) 学习。

---

## 2. 运行

在本模块目录执行：

```bash
python3 -B L2_distributed_exact_resume.py
```

脚本占用连续 5 个 loopback 端口；并行运行多个副本时显式错开：

```bash
python3 -B L2_distributed_exact_resume.py --base-port 29720
```

也可设置 `NANO_PRETRAINING_L2_BASE_PORT`。端口只负责本机进程 rendezvous，不是对外服务。
所有 checkpoint 和进程结果都位于自动清理的临时目录，不在当前工作目录留下模型文件。

一次验收输出如下。`torch` 版本和 loss 数字属于该环境；应跨同环境比较 contract 与 digest，
不要把这些 CPU 数字外推成 GPU 性能。

```text
==============================================================================
Pretraining lifecycle L2 — distributed exact resume & fault triage
==============================================================================
[0] environment: torch=2.13.0 world_size=2 backend=gloo device=cpu
    real: torch.distributed / manual all-reduce / rank-local checkpoint set
    toy : world_size=2, loopback interconnect, tiny model
    DP model/optimizer tensors are replicated; sampler/RNG identity remains rank-local

[1] continuous distributed run (world_size=2)
    step  global_train   lr        global_val
       1       16.7884  0.00200    19.4889
       5       14.3304  0.00290    14.8264
      10        9.5362  0.00191     9.6534
      15        7.5653  0.00060     7.6030
      20        6.9847  0.00000     7.2412
    final val=7.2412  tokens/rank=2560 global_tokens=5120

[2] exact resume from rank-local checkpoint set@step8
    resumed final val=7.2412
    max param diff vs continuous = 0.000e+00

[3] fault injection: load shard_0.pt on all ranks
    rejected non-owner loads=1/1
    fail-closed rank binding prevents silent sampler/RNG identity corruption

[4] anomaly branch from checkpoint@step10: inject at step12
    pre-spike train=9.2019 spike_train=399.8253 ratio=43.5x

[5] rollback: discard anomaly branch and replay step11..20
    clean replay final val=7.2412
    max param diff vs continuous = 0.000e+00

[6] self-check
    PASS | distributed training improves validation loss
    PASS | warmup raises the learning rate
    PASS | cosine decay lowers the learning rate
    PASS | per-rank token ledger closes arithmetically
    PASS | global token ledger includes every data-parallel rank
    PASS | rank-local exact resume matches continuous run
    PASS | resumed final val equals continuous final val
    PASS | wrong-rank checkpoint loads fail closed
    PASS | injected loss spike is observable before promotion
    PASS | clean rollback replay rejoins the continuous trajectory

SELF-CHECK: 10/10 PASS
digest(sha256 of metrics): 9cb7ab4783da46c7
takeaway: data parallelism changes the bookkeeping, not the contract:
          same full distributed state == same distributed training run.
RESULT_JSON={"checks": {"passed": 10, "total": 10}, "digest": "9cb7ab4783da46c7", "evidence_boundary": "Real two-process torch.distributed/gloo on CPU; tiny model and loopback only. DP tensors are replicated, not FSDP-sharded, and timing is not benchmarked.", "metrics": {"clean_replay_param_diff": 0.0, "final_validation_loss": 7.24115, "global_tokens": 5120, "rank_mismatch_rejections": 1, "resume_param_diff": 0.0, "spike_train_ratio": 43.450319, "tokens_per_rank": 2560}, "module": "nano_pretraining_loop_l2", "schema_version": 1}
```

---

## 3. Data parallel 的数值合同

rank $r$ 在本地 micro-batch 上得到梯度 $g_r$，同步更新使用：

$$
\bar g = \frac{1}{W}\sum_{r=0}^{W-1} g_r.
$$

代码把所有梯度摊平成一个 bucket，只做一次 all-reduce，再切回每个参数：

```python
grad_params = [p for p in model.parameters() if p.grad is not None]
flat = torch.cat([p.grad.detach().reshape(-1) for p in grad_params])
dist.all_reduce(flat, op=dist.ReduceOp.SUM)
flat /= world_size
```

真实 DDP 会用多个 bucket，并把反向计算与通信重叠。本节用一个 bucket 是为了在 CPU 上保留正确的
数值合同，同时避免“每个参数一次 collective”把教学实验拖成通信延迟测试。

global train loss 也不能简单平均 rank loss；每个 rank 的有效 token 数可能不同。正确口径是：

$$
L_{global} =
\frac{\sum_r L_r N_r}{\sum_r N_r},
$$

其中 $N_r$ 是未被 boundary mask 掉的 target token 数。

---

## 4. 为什么错误 rank 文件要拒绝，而不是等它自然发散？

每个文件显式写入：

```python
shard = {
    "rank": rank,
    "world_size": world_size,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "torch_rng": torch.get_rng_state(),
    "py_rng": random.getstate(),
    "sampler": sampler.state(),
    "step": step,
    "consumed_tokens": consumed_tokens,
}
```

加载时先验证 world size，再验证 rank：

```python
if shard.get("rank") != rank:
    raise RuntimeError(
        f"rank-local shard rank={shard.get('rank')} loaded by rank={rank}"
    )
```

这个反例有意不再检查“错误加载后参数差是否大于阈值”。原因是 DP 的 model/optimizer tensor 原本就复制；
某些小实验里，错误 rank 文件可能暂时看不出参数差。依赖后续 divergence 才发现身份错误，是 fail open。
正确行为是在状态身份不成立时立即拒绝。

更完整的生产 manifest 还应包含 data snapshot、代码/config revision、tensor shape/dtype、文件 hash、
完成标记和原子发布协议。本节只教 rank/world identity，不声称已经实现完整 checkpoint service。

---

## 5. 两本 token 账必须同时闭合

本实验每个 rank 消费：

$$
20\ \text{steps}
\times 2\ \text{micro-batches}
\times 4\ \text{blocks}
\times 16\ \text{tokens}
=2560.
$$

两路 data parallel 的全局消费量是：

$$
N_{global}=W N_{rank}=2\times2560=5120.
$$

`tokens_per_rank` 适合恢复单个 rank 的 cursor；`global_tokens` 适合说明总训练预算。把前者命名为
`consumed_tokens across ranks` 会少算 $W$ 倍，也是本节修掉的一个典型账本错误。

---

## 6. loss spike 与 rollback 是两个分支，不是“继续训就算恢复”

脚本在 step 10 保存 rollback anchor，然后构造两个分支：

```text
checkpoint@10
├── anomaly: step12 loss ×100 → 观测 spike，不晋升
└── clean replay: 丢弃异常分支，重跑 step11..20 → 与 continuous 完全重合
```

异常分支的 train loss 放大 43.5 倍，说明监控能看见它；真正的恢复证据是 clean replay 的最终参数与
continuous run 最大差为 0。仅展示“spike 后 loss 又下降”不能证明 rollback，因为那可能只是继续训练。

真实训练还要定义：谁触发 rollback、回滚多少 step、异常数据是否隔离、已消费预算怎样记账、恢复后
validation/hidden sentinel 是否重新通过。本节只证明“完整 anchor 能重放同一轨迹”。

---

## 7. 这 10 个 check 分别证明什么？

| check | 支持的结论 | 不支持的外推 |
|---|---|---|
| train/val、warmup、decay | 分布式 loop 确实在学习且 scheduler 生效 | 大模型收敛质量 |
| per-rank/global token 账 | global batch 口径闭合 | 数据质量或利用率 |
| resume diff = 0 | 同环境、同 world size、完整状态可 exact resume | 跨版本/跨拓扑 resume |
| wrong-rank reject | checkpoint identity fail closed | tensor 文件完整性已全面验证 |
| spike ratio | 构造异常可观测 | 自动定位真实 NaN/loss spike 根因 |
| clean replay diff = 0 | 丢弃异常分支后可回到参考轨迹 | 生产 checkpoint 永不损坏 |

---

## 8. 失败边界与下一步

- gloo 只走单机 loopback；没有 NCCL、RDMA、拓扑或 straggler 证据。
- model/optimizer 在 DP rank 间复制；不要把文件名 `shard_*.pt` 误读为 FSDP tensor sharding。
- 使用固定 `world_size=2`；elastic resize 需要重建 sampler 和状态映射。
- checkpoint 写入没有临时文件 + manifest commit 的原子发布协议，也未注入半写/磁盘损坏。
- 数值相同依赖同一 PyTorch/算子/设备环境；跨版本 bit-for-bit 不是本节承诺。
- timing 被明确排除，CPU 运行时不能用于估计 GPU MFU 或扩展效率。

下一步按顺序交叉阅读：

1. [FSDP L2](../nano-fsdp/tutorial_L2.md)：模型/梯度/optimizer 真正分片后的 checkpoint/通信；
2. [Megatron L2](../nano-megatron/tutorial_L2.md)：pipeline stage、micro-batch schedule 与 bubble；
3. [数据平台 L0](../../03-data-distributed-rsi/nano-data-platform/tutorial_L0.md)：训练状态如何绑定 immutable data snapshot。

费曼自检：如果 model/optimizer 在所有 DP rank 上完全相同，为什么仍不能删除所有非零 rank 的
checkpoint？若你的答案没有提到 sampler/RNG、rank identity 和未来并行布局，说明还没真正掌握本节。
