#!/usr/bin/env python3
"""
L3_hybridflow_colocate.py — nano-verl L3

对照 verl（HybridFlow）源码，复现其核心调度机制：**同一组 worker（GPU）在
rollout 阶段与 train 阶段之间复用**（colocate），阶段切换时做权重 resharding
（训练分片 → 全量 gather → 推理引擎副本）与训练态 offload。

可运行性契约（课程可运行性契约）：L3 = 可运行的本质模拟 + 显式注明。
- 真实 verl 需要 Ray + FSDP/Megatron + vLLM + 多 GPU，本机不可跑
  [TODO: verify on real system]（真机验证需在真实 GPU/多机环境验证，后续验证）。
- 本文件可运行的是「本质模拟」：**计算是真的**（真实 char-LSTM、真实 PPO 梯度、
  真实权重在 trainer→rollout 间流动），**显存/时钟是声明式成本模型折算**
  （COST 块，显式注明，与实测玩具字节数分开报告）。单控制器 + SPMD worker
  group + DataProto chunk/concat + resharding 的**语义**与 verl 一致，并被
  末尾 self-check 机器断言。

对照的权威实现（2026-08-07 抓取核验，sha256 见 tutorial_L3.md §11）：
  verl-project/verl @ v0.7.1
  - verl/protocol.py:L317/L863/L916               DataProto / chunk / concat
  - verl/single_controller/base/decorator.py:L37-47/L119/L166/L190/L397
                                                  Dispatch 模式 / dispatch_fn / register
  - verl/single_controller/ray/base.py:L411/L981  RayWorkerGroup / create_colocated_worker_cls
  - verl/trainer/ppo/ray_trainer.py:L225/L1230/L1321/L1486/L1506/L1531-1533
                                                  RayPPOTrainer / fit 循环各阶段
  - verl/workers/fsdp_workers.py:L143/L750-856/L997-1041/L1062-1071
                                                  ActorRolloutRefWorker / rollout_mode()
                                                  （gather→convert→update_weights→resume）
                                                  / update_actor 的 load-offload 夹心 /
                                                  generate_sequences 的两相切换
  - arXiv:2409.19256（HybridFlow 论文：单控制器 + 多控制器混合范式）

与 L2 的关系（K+1）：L2 把 actor 与 learner 拆成两个 lockstep 进程（不同资源），
并未实现跨 step overlap；L3 问「同步训练时为什么常常不拆卡？」——严格新鲜 rollout
让两相在 step 内串行，固定拆卡会让一部分卡在每个阶段空转；把同一组卡在两相间复用，
每个阶段都拿到全部算力，代价是阶段边界的 resharding 同步，以及必须精确管理「谁现在住在显存里」。
这正是 verl 的默认架构（colocate 的 ActorRolloutRefWorker）。

依赖：torch（CPU）。为保证跨遍 bit-level 确定性，L3 强制 CPU
（L1/L2 可自动选 device；L3 的 self-check 需要跨遍逐位一致）。
运行：python3 L3_hybridflow_colocate.py   # ~8s
"""

import hashlib
import sys

sys.dont_write_bytecode = True  # 仓库卫生约定：不落 __pycache__

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise SystemExit(
        "[error] torch is required.\n        Install: pip install torch (CPU is enough.)"
    ) from e


# ---------------------------------------------------------------------------
# 任务与模型：与 L1/L2 完全相同（go: -> hello）。L3 只换系统，不换任务——
# 这样训练曲线的任何差异都只能归因于调度/数据切分，而不是任务变了。
# ---------------------------------------------------------------------------
SEED = 42
PROMPT = "go:"
TARGET = "hello"
MAX_RESP_LEN = len(TARGET)

VOCAB = ["<pad>", "<sos>"] + [chr(ord("a") + i) for i in range(26)] + [":"]
PAD_ID = 0
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}

EMBED_DIM = 32
HIDDEN_DIM = 64
N_LAYERS = 1

N_ROLLOUTS = 64          # 全局 batch（driver 视角）
N_EPOCHS = 4
N_ITERATIONS = 30
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
KL_TARGET = 0.02
LR = 1e-3
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5
WARMUP_STEPS = 100       # SFT warmup = 真实流程里「RL 从 SFT checkpoint 出发」

N_WORKERS = 4            # 模拟集群：4 个 rank（verl 里 = 4 张 GPU 上的 4 个 worker）

DEVICE = torch.device("cpu")  # 强制 CPU：跨遍 bit-level 确定性


# ---------------------------------------------------------------------------
# COST MODEL（声明式，不是实测）：真实显存/时钟按 7B/13B 规模折算，计算用
# 28K 参数玩具模型真跑。两套数字分开报告，绝不混用——这是模拟的诚实边界。
# ---------------------------------------------------------------------------
COST = {
    "budget_gb": 80.0,            # 单卡显存预算（如 H100-80GB）
    "p_gb_7b": 28.0,              # 7B fp32 参数 ≈ 28 GB（记作 P）
    "p_gb_13b": 52.0,             # 13B ≈ 52 GB
    "optim_factor": 2.0,          # Adam m+v（fp32）= 2P
    "rollout_dtype_factor": 0.5,  # 推理副本 bf16 = P/2
    "kv_gb": 20.0,                # rollout 相 KV cache 声明值
    "act_gb": 6.0,                # train 相激活/工作区声明值
    "roll_ms": 1000.0,            # N worker 并行时一步 rollout 声明耗时
    "train_ms": 800.0,            # N worker 并行时一步 train 声明耗时
    "sync_ms": 300.0,             # 一次权重 resharding+同步声明耗时
    "etas": (1.0, 0.85),          # 并行效率指数：cost(k) = base * (N/k)^eta
}


def set_seed(seed=SEED):
    torch.manual_seed(seed)


def encode(text):
    return [CHAR2ID[c] for c in text]


def decode(token_ids):
    return "".join(VOCAB[i] for i in token_ids)


def reward_fn(generated):
    """逐字符匹配奖励（与 L1/L2 相同）：每位 1/T，总和 ≤ 1。"""
    rewards = []
    for i in range(MAX_RESP_LEN):
        c = generated[i] if i < len(generated) else "<PAD>"
        rewards.append(1.0 / MAX_RESP_LEN if c == TARGET[i] else 0.0)
    return rewards


class TinyLSTM(nn.Module):
    """与 L1/L2 同构：policy + value 共享 backbone。"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        out, _ = self.lstm(x)
        return self.policy_head(out), self.value_head(out).squeeze(-1)


def new_model():
    return TinyLSTM(len(VOCAB), EMBED_DIM, HIDDEN_DIM, N_LAYERS).to(DEVICE)


# ---------------------------------------------------------------------------
# 参数扁平化与分片：FSDP 的前提是把参数看成平坦的表。真实 FSDP 按 module
# 粒度组织 FlatParameter；这里按连续切片，本质相同——每个 rank 只持有 1/N
# 的参数与优化器状态（ZeRO/FSDP 的 optimizer-state sharding）。
# ---------------------------------------------------------------------------
def flatten_state_dict(model):
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    spec = [(name, tuple(p.shape), p.numel()) for name, p in model.named_parameters()]
    return flat, spec


def unflatten_to_state_dict(flat, spec):
    sd, offset = {}, 0
    for name, shape, numel in spec:
        sd[name] = flat[offset:offset + numel].reshape(shape).clone()
        offset += numel
    assert offset == flat.numel()
    return sd


def shard_params(flat, n):
    """FSDP 分片。参数总数不一定整除 n——真实 FSDP 把 FlatParameter pad 到
    可整除（padding 参与通信、不参与计算）；nano 版如实照做，pad 全程可追踪。"""
    pad = (-flat.numel()) % n
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    size = flat.numel() // n
    return [flat[i * size:(i + 1) * size].clone() for i in range(n)], pad


# ---------------------------------------------------------------------------
# DataProto（nano 版）——对照 verl/protocol.py:L317（v0.7.1）
# 真实版：TensorDict batch + non_tensor_batch + meta_info，是 driver 与 worker
# group 之间**唯一**的数据协议。nano 版保留两个关键语义：
#   chunk(n)：沿 dim 0 切 n 份，meta_info 复制给每份（verl L863-902 同语义）；
#   concat(list)：沿 dim 0 拼回，meta_info 合并且非 metrics 键要求一致（L916-960）。
# ---------------------------------------------------------------------------
class DataProto:
    def __init__(self, batch, meta_info=None):
        self.batch = batch
        self.meta_info = meta_info if meta_info is not None else {}

    def __len__(self):
        if not self.batch:
            return 0
        return next(iter(self.batch.values())).shape[0]

    def chunk(self, chunks):
        n = len(self)
        assert n % chunks == 0, (
            f"nano 版要求整除切分（Got {n} / {chunks}）。不整除时 verl 会 pad，"
            "训练侧另有 _balance_batch 按 seqlen 均衡负载（ray_trainer.py:L1018）。"
        )
        out = []
        for sub in zip(*[torch.chunk(v, chunks, dim=0) for v in self.batch.values()]):
            out.append(DataProto(dict(zip(self.batch.keys(), sub)), dict(self.meta_info)))
        return out

    @staticmethod
    def concat(protos):
        batch = {}
        for k in protos[0].batch:
            batch[k] = torch.cat([p.batch[k] for p in protos], dim=0)
        merged = {}
        for p in protos:
            for k, v in p.meta_info.items():
                if k in merged:
                    assert merged[k] == v, f"meta_info key '{k}' conflict on concat"
                else:
                    merged[k] = v
        return DataProto(batch, merged)


# ---------------------------------------------------------------------------
# Dispatch 模式（nano 版）——对照 verl/single_controller/base/decorator.py:L37-47
# verl 注册 8 种模式；这里实现两种最核心的：
#   ONE_TO_ALL      ：同样的参数广播给所有 rank（init/save/load 这类控制操作）；
#   DP_COMPUTE_PROTO：DataProto 沿 batch 维 chunk 到各 rank，各算各的，concat 收回
#                    （generate / compute_log_prob / update_actor 这类数据并行计算）。
# 这就是 HybridFlow 论文的「单控制器编排 + SPMD 节点内计算」：driver 只写一行
# wg.generate_sequences(batch)，数据怎么切、怎么收，由方法上注册的 dispatch 决定。
# ---------------------------------------------------------------------------
class Dispatch:
    ONE_TO_ALL = "ONE_TO_ALL"
    DP_COMPUTE_PROTO = "DP_COMPUTE_PROTO"


def register(dispatch_mode):
    """verl decorator.py:L397 的最小对应：把 dispatch 模式钉在方法上。"""
    def deco(fn):
        fn.__dispatch_mode__ = dispatch_mode
        return fn
    return deco


class WorkerGroup:
    """单控制器视角的 worker 组——对照 verl/single_controller/ray/base.py:L411
    RayWorkerGroup。driver 调 wg.method(...)，group 按方法上注册的 dispatch 模式
    决定切分/广播/收集。真实版跨 Ray actor 远程调用；nano 版是同进程对象按 rank
    序顺序执行——语义相同（SPMD：所有 rank 执行同一方法），且完全确定性。"""

    def __init__(self, ranks, name):
        self.ranks = ranks
        self.name = name
        self._methods = {}
        for attr in dir(type(ranks[0])):
            fn = getattr(type(ranks[0]), attr, None)
            if callable(fn) and hasattr(fn, "__dispatch_mode__"):
                self._methods[attr] = fn

    def __getattr__(self, item):
        if item in ("ranks", "name", "_methods"):
            raise AttributeError(item)
        fn = self._methods[item]
        mode = fn.__dispatch_mode__

        def call(*args, **kwargs):
            if mode == Dispatch.ONE_TO_ALL:
                outs = [fn(r, *args, **kwargs) for r in self.ranks]
                return outs[0]
            # DP_COMPUTE_PROTO：第一个位置参数必为 DataProto
            proto = args[0]
            assert isinstance(proto, DataProto)
            chunks = proto.chunk(len(self.ranks))
            outs = [fn(r, chunks[i], *args[1:], **kwargs) for i, r in enumerate(self.ranks)]
            first = outs[0]
            if isinstance(first, tuple):
                # (DataProto, extra)：proto 按 rank 序 concat，extra 按 rank 序收集
                protos = [o[0] for o in outs]
                extras = [o[1] for o in outs]
                return DataProto.concat(protos), extras
            if isinstance(first, DataProto):
                return DataProto.concat(outs)
            return outs
        return call


# ---------------------------------------------------------------------------
# ColocatedRank：一个 rank = 一张模拟 GPU 上住着的全部角色。
# 对照 fsdp_workers.py:L143 ActorRolloutRefWorker——同一个 worker 类里住着
# actor（训练）+ rollout（推理）+ ref，阶段切换时轮流占用显存。
# ---------------------------------------------------------------------------
class ColocatedRank:
    def __init__(self, rank_id, param_shard, param_spec):
        self.rank_id = rank_id
        # --- 训练态（FSDP 分片 + 分片 Adam）---
        self.param_shard = param_shard
        self.param_spec = param_spec
        self.m = torch.zeros_like(param_shard)   # Adam 一阶动量（只存 1/N）
        self.v = torch.zeros_like(param_shard)   # Adam 二阶动量（只存 1/N）
        self.adam_step = 0
        # --- 推理态（全量副本 = vLLM 角色的本质：完整权重 + KV cache）---
        self.rollout_model = new_model()
        self.phase = "init"

    def rollout_load(self, sd):
        self.rollout_model.load_state_dict(sd)

    # ---- 推理：batch 自回归采样（L2 的 batch inference，按 rank 的 chunk 执行）----
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, proto, step):
        """verl fsdp_workers.py:L1043 的本质模拟。
        返回 (逐token行 DataProto, 每条 rollout 的总奖励列表)。
        行序 = rollout-major（rollout i 的 T 个 token 连续），concat 后 driver
        可直接 reshape(R, T)。随机性只由 (SEED, step, rank_id) 决定。"""
        g = torch.Generator(device="cpu").manual_seed(SEED * 1000003 + step * 97 + self.rank_id)
        prompts = proto.batch["input_ids"]              # (n, prompt_len)
        n = prompts.shape[0]
        model = self.rollout_model
        model.eval()

        # token-major 暂存（自回归只能按步推进），最后重排为 rollout-major
        tm_state, tm_action, tm_lp, tm_val, tm_rew = [], [], [], [], []
        seqs = [t.tolist() for t in prompts]
        prompt_lens = [len(t) for t in prompts]
        with torch.no_grad():
            for _ in range(MAX_RESP_LEN):
                inp = torch.tensor(seqs, dtype=torch.long, device=DEVICE)
                logits, values = model(inp)
                dist = torch.distributions.Categorical(logits=logits[:, -1, :])
                probs = torch.softmax(dist.logits, dim=-1)
                actions = torch.multinomial(probs, 1, generator=g).squeeze(-1)
                lp = dist.log_prob(actions)
                for i in range(n):
                    tm_state.append(list(seqs[i]))
                    tm_action.append(actions[i].item())
                    tm_lp.append(lp[i].item())
                    tm_val.append(values[i, -1].item())
                    seqs[i].append(actions[i].item())

        # 逐 token 奖励回填（与 L1 同口径）。注意：tm_rew 必须按 token-major
        # （下标 t*n+i）存放，与 tm_state/tm_action/... 同序，否则下面的重排会
        # 把奖励错配到别的行上（GAE 会拿到噪声）。
        rew_matrix = []
        totals = []
        for i in range(n):
            resp = decode(seqs[i][prompt_lens[i]:])
            step_rew = reward_fn(resp)
            rew_matrix.append(step_rew)
            totals.append(sum(step_rew))
        for t in range(MAX_RESP_LEN):
            for i in range(n):
                tm_rew.append(rew_matrix[i][t])

        # token-major -> rollout-major：行 (t, i) 的全局下标 = t*n + i
        order = [t * n + i for i in range(n) for t in range(MAX_RESP_LEN)]
        rows_state = [tm_state[k] for k in order]
        lengths = [len(s) for s in rows_state]
        max_len = max(lengths)
        # 与 L1/L2 一致：普通单向 LSTM 用右 padding，并保存真实长度。
        padded = [s + [PAD_ID] * (max_len - len(s)) for s in rows_state]
        proto_out = DataProto({
            "state": torch.tensor(padded, dtype=torch.long),
            "length": torch.tensor(lengths, dtype=torch.long),
            "action": torch.tensor([tm_action[k] for k in order], dtype=torch.long),
            "old_log_prob": torch.tensor([tm_lp[k] for k in order], dtype=torch.float32),
            "value": torch.tensor([tm_val[k] for k in order], dtype=torch.float32),
            "reward": torch.tensor([tm_rew[k] for k in order], dtype=torch.float32),
        }, meta_info={})
        return proto_out, totals


def shard_adam_step(rank, reduced_grad):
    """ZeRO/FSDP 式分片 Adam：每个 rank 只拿归约后梯度的自己那片，
    只维护那片的动量。Adam 是逐元素算法，分片执行与整块执行逐位等价
    （只要梯度归约相同）——这个性质在 [4a] 被机器验证。"""
    size = rank.param_shard.numel()
    start = rank.rank_id * size
    g = reduced_grad[start:start + size]
    rank.adam_step += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    rank.m = b1 * rank.m + (1 - b1) * g
    rank.v = b2 * rank.v + (1 - b2) * g * g
    m_hat = rank.m / (1 - b1 ** rank.adam_step)
    v_hat = rank.v / (1 - b2 ** rank.adam_step)
    rank.param_shard = rank.param_shard - LR * m_hat / (v_hat.sqrt() + eps)


def rank_grad_and_metrics(rank, full_padded, proto, total_rows):
    """单 rank 的一次 PPO 前向/反向（= 真实 update_actor 里每个 DP rank 干的事）：
    用 gather 出的全量参数 forward/backward，得到全量梯度交给 driver 归约。
    真实 FSDP 是逐层 all-gather + reduce-scatter；nano 一次性 gather，语义相同。
    loss_rank = Σ_{行∈chunk} loss_row / 全局总行数 —— 整除切分下，各 rank 梯度
    之和即全局均值梯度（与真实 DP 的 local-mean 再 all-reduce-mean 等价）。
    梯度末尾补零对齐 padded 分片布局（pad 位无参数、梯度恒 0）。"""
    n_real = sum(numel for _, _, numel in rank.param_spec)
    pad = full_padded.numel() - n_real
    model = new_model()
    model.load_state_dict(unflatten_to_state_dict(full_padded[:n_real], rank.param_spec))
    model.train()

    logits, values = model(proto.batch["state"])
    batch_idx = torch.arange(len(proto), device=DEVICE)
    last_idx = proto.batch["length"] - 1
    last_logits = logits[batch_idx, last_idx, :]
    last_values = values[batch_idx, last_idx]
    dist = torch.distributions.Categorical(logits=last_logits)
    new_lp = dist.log_prob(proto.batch["action"])
    ratio = torch.exp(new_lp - proto.batch["old_log_prob"])

    adv = proto.batch["advantage"]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
    policy_rows = -torch.min(surr1, surr2)
    value_rows = (last_values - proto.batch["return"]) ** 2
    entropy_rows = dist.entropy()

    loss = (policy_rows.sum() + VALUE_COEF * value_rows.sum()
            - ENTROPY_COEF * entropy_rows.sum()) / total_rows
    loss.backward()
    flat_grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    if pad:
        flat_grad = torch.cat([flat_grad, torch.zeros(pad, dtype=flat_grad.dtype)])

    kl_rows = (ratio - 1) - torch.log(ratio)   # L1 同款 KL 估计
    part = {
        "policy": policy_rows.sum().item(),
        "value": value_rows.sum().item(),
        "entropy": entropy_rows.sum().item(),
    }
    return flat_grad, part, kl_rows.sum().item()


def gae_batch(values, rewards, gamma, lam):
    """(R, T) 批量 GAE，episode 在 T 处终止（next_value=0），与 L1 同口径。
    在 driver 侧对全量 batch 计算——对照 ray_trainer.py:L129/L1486
    compute_advantage 是 driver（单控制器）侧的函数。"""
    R, T = values.shape
    advantages = torch.zeros_like(values)
    gae = torch.zeros(R)
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros(R)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    return advantages, advantages + values


def supervised_warmup(model, optimizer, prompt_ids, steps):
    """与 L1/L2 相同的 teacher-forcing warmup。"""
    model.train()
    full_seq = torch.tensor([prompt_ids + encode(TARGET)], dtype=torch.long, device=DEVICE)
    input_ids, target_ids = full_seq[:, :-1], full_seq[:, 1:]
    total = 0.0
    for _ in range(steps):
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / steps


# ---------------------------------------------------------------------------
# 显存账本（声明式）：只回答一个问题——每个阶段，每张卡上住着什么？
# colocate 的全部艺术：train 相 rollout 引擎在睡觉（vLLM sleep mode，权重/KV
# 让位），rollout 相训练态（参数分片+优化器）已 offload 到 CPU。
# 对照 fsdp_workers.py:L1001-1004（进 train 上卡）/ L1034-1039（出 train 回 CPU）/
# L831-851（rollout 侧 resume weights -> update -> resume kv_cache 两段唤醒）。
# ---------------------------------------------------------------------------
class MemoryLedger:
    def __init__(self, n_shards, p_gb, budget_gb):
        self.n_shards = n_shards
        self.p = p_gb
        self.budget = budget_gb

    def train_gb(self):
        # 参数分片 P/k + Adam 2P/k + 梯度缓冲 P/k + 激活
        return (1.0 + COST["optim_factor"] + 1.0) * self.p / self.n_shards + COST["act_gb"]

    def rollout_gb(self):
        return self.p * COST["rollout_dtype_factor"] + COST["kv_gb"]

    def fits(self, phase):
        gb = self.train_gb() if phase == "train" else self.rollout_gb()
        return gb, gb <= self.budget


# ---------------------------------------------------------------------------
# driver：fit 循环——对照 ray_trainer.py:L1230 RayPPOTrainer.fit 的阶段顺序：
#   generate_sequences(L1321) -> old_log_prob(rollout 内记录) ->
#   compute_advantage(L1486, driver 侧) -> update_actor(L1506) ->
#   update_weights(L1531-1533, 源码注释「update weights from trainer to rollout」)
# nano 版省略 critic 与 ref（L1/L2 同款 PPO+GAE；verl 在哪插入见 tutorial §4）。
# ---------------------------------------------------------------------------
class NanoHybridTrainer:
    """mode:
       colocate — N 个 rank 两相复用（verl 默认，本文件主角）；
       flat     — 同 N 个 rank、同样切分与同步，但跳过阶段切换/账本
                  （数值透明性对照：证明调度机制不碰训练数值）；
       disagg   — L2 的世界观：N/2 卡专做 rollout + N/2 卡专做 train。
    """

    def __init__(self, mode, n_workers=N_WORKERS, n_iterations=N_ITERATIONS,
                 p_gb=None, verbose=True):
        assert mode in ("colocate", "flat", "disagg")
        self.mode = mode
        self.n = n_workers
        self.n_iter = n_iterations
        self.verbose = verbose
        set_seed()

        # driver 建完整模型并做 SFT warmup = 「RL 从 SFT checkpoint 出发」
        # （真实流程：SFT 产物是 RL 的输入；对应 L1 的 warmup、trinity L1 的 sft_then_rl）
        model = new_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.warmup_loss = supervised_warmup(model, optimizer, encode(PROMPT), WARMUP_STEPS)

        flat, self.spec = flatten_state_dict(model)
        self.n_params = flat.numel()

        self.n_train = n_workers if mode != "disagg" else n_workers // 2
        self.n_roll = n_workers if mode != "disagg" else n_workers // 2
        train_shards, self.pad_len = shard_params(flat, self.n_train)
        train_ranks = [ColocatedRank(i, train_shards[i], self.spec) for i in range(self.n_train)]
        self.train_group = WorkerGroup(train_ranks, "train")
        if mode == "disagg":
            roll_ranks = [ColocatedRank(i, train_shards[i], self.spec) for i in range(self.n_roll)]
            self.roll_group = WorkerGroup(roll_ranks, "rollout")
        else:
            self.roll_group = self.train_group
        # 初始推理副本 = warmup 后的全量权重（首次 sync 的等价物）
        sd0 = unflatten_to_state_dict(flat, self.spec)
        for r in self.roll_group.ranks:
            r.rollout_load(sd0)

        self.traffic = {
            "sync_gather_bytes": 0,      # 阶段边界：分片 -> 全量 gather
            "sync_replica_write_bytes": 0,  # 阶段边界：全量 -> 各推理副本
            "train_gather_bytes": 0,     # train 相内：每 epoch 前向的参数 gather
            "grad_reduce_bytes": 0,      # train 相内：每 epoch 反向的梯度归约
            "syncs": 0,
        }
        self.p_bytes = (self.n_params + self.pad_len) * 4   # fp32，含 pad（通信按 padded 计）
        self.p_gb = p_gb if p_gb is not None else COST["p_gb_7b"]
        self.ledger = MemoryLedger(self.n_train, self.p_gb, COST["budget_gb"])

    # ---- 阶段切换：colocate 才真切；flat 跳过（数值对照）；disagg 无需切 ----
    def _switch(self, phase):
        if self.mode != "colocate":
            return
        gb, ok = self.ledger.fits(phase)
        assert ok, f"显存账本超预算：{phase} 相 {gb:.1f} GB > {self.ledger.budget} GB"
        for r in self.train_group.ranks:
            r.phase = phase

    # ---- 权重同步（resharding）----
    # 对照 fsdp_workers.py:L750-856 rollout_mode()：gather 全量 -> （convert keys）
    # -> rollout.update_weights；nano 版显式做成一步并按字节计数。
    def _sync(self):
        full_padded = torch.cat([r.param_shard for r in self.train_group.ranks], dim=0)
        sd = unflatten_to_state_dict(full_padded[:self.n_params], self.spec)
        targets = self.roll_group.ranks
        for r in targets:
            r.rollout_load(sd)
        # 流量账（真实 all-gather：每 rank 收 (k-1)/k 份全量）
        self.traffic["sync_gather_bytes"] += self.p_bytes * (self.n_train - 1)
        self.traffic["sync_replica_write_bytes"] += self.p_bytes * len(targets)
        self.traffic["syncs"] += 1

    def train_step(self, proto):
        """PPO 更新（n_epochs × 全 rank DP）。梯度归约按 rank 序，确定性。"""
        total_rows = len(proto)
        sums = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        n_updates = 0
        approx_kl = 0.0
        for _ in range(N_EPOCHS):
            full_padded = torch.cat([r.param_shard for r in self.train_group.ranks], dim=0)
            # train 相内的 gather/reduce 流量（真实 FSDP 每次前向/反向都有）
            self.traffic["train_gather_bytes"] += self.p_bytes * (self.n_train - 1)
            self.traffic["grad_reduce_bytes"] += self.p_bytes * (self.n_train - 1)

            chunks = proto.chunk(self.n_train)
            grads, kl_sum = [], 0.0
            for i, r in enumerate(self.train_group.ranks):
                g, part, kl_local = rank_grad_and_metrics(r, full_padded, chunks[i], total_rows)
                grads.append(g)
                for k, v in part.items():
                    sums[k] += v
                kl_sum += kl_local
            reduced = grads[0]
            for g in grads[1:]:
                reduced = reduced + g                  # rank 序求和（确定性）
            norm = reduced.norm()
            if norm > 0.5:
                reduced = reduced * (0.5 / norm)       # 全局梯度裁剪（reduce 后）
            for r in self.train_group.ranks:
                shard_adam_step(r, reduced)            # 每 rank 只更新自己的 1/N
            n_updates += 1
            approx_kl = kl_sum / total_rows
            if approx_kl > KL_TARGET * 1.5:            # L1 同款 KL 早停
                break
        return {
            "policy_loss": sums["policy"] / total_rows / n_updates,
            "value_loss": sums["value"] / total_rows / n_updates,
            "entropy": sums["entropy"] / total_rows / n_updates,
            "approx_kl": approx_kl,
        }

    def fit(self):
        prompt_ids = encode(PROMPT)
        metrics_log = []
        for step in range(self.n_iter):
            # ---- phase 1: rollout（对照 fit L1321 generate_sequences）----
            self._switch("rollout")
            proto = DataProto({
                "input_ids": torch.tensor([prompt_ids] * N_ROLLOUTS, dtype=torch.long)
            }, meta_info={"step": step})
            rows, totals_per_rank = self.roll_group.generate_sequences(proto, step)
            totals = [t for sub in totals_per_rank for t in sub]   # rank 序展平

            # ---- phase 2: driver 侧 advantage（对照 fit L1486）----
            R, T = N_ROLLOUTS, MAX_RESP_LEN
            values = rows.batch["value"].reshape(R, T)
            rew = rows.batch["reward"].reshape(R, T)
            advantages, returns = gae_batch(values, rew, GAMMA, GAE_LAMBDA)
            adv_flat = advantages.reshape(-1)
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

            # ---- phase 3: train（对照 fit L1506 update_actor）----
            self._switch("train")
            update_proto = DataProto({
                "state": rows.batch["state"],
                "length": rows.batch["length"],
                "action": rows.batch["action"],
                "old_log_prob": rows.batch["old_log_prob"],
                "advantage": adv_flat,
                "return": returns.reshape(-1),
            }, meta_info={"step": step})
            stats = self.train_step(update_proto)

            # ---- phase 4: 权重同步（对照 fit L1531-1533 update_weights）----
            self._sync()

            rewards_t = torch.tensor(totals, dtype=torch.float32)
            mean_reward = rewards_t.mean().item()
            exact = sum(1 for t in totals if t > 0.999) / N_ROLLOUTS
            metrics_log.append({
                "step": step, "reward": mean_reward, "exact": exact,
                "policy_loss": stats["policy_loss"], "value_loss": stats["value_loss"],
                "entropy": stats["entropy"], "approx_kl": stats["approx_kl"],
            })
            if self.verbose and (step % 5 == 0 or step == self.n_iter - 1):
                print(f"  [step {step:2d}] reward={mean_reward:.3f} exact={exact:.3f} "
                      f"policy_loss={stats['policy_loss']:+.4f} "
                      f"value_loss={stats['value_loss']:.4f} "
                      f"entropy={stats['entropy']:.3f} kl={stats['approx_kl']:.4f}")
        return metrics_log

    def greedy_samples(self, n=3):
        """最终策略的贪心输出（确定性）， eyeball 用。"""
        model = self.roll_group.ranks[0].rollout_model
        model.eval()
        seq = encode(PROMPT)
        outs = []
        with torch.no_grad():
            for _ in range(n):
                s = list(seq)
                for _ in range(MAX_RESP_LEN):
                    logits, _ = model(torch.tensor([s], dtype=torch.long, device=DEVICE))
                    s.append(int(logits[0, -1, :].argmax()))
                outs.append(decode(s[len(seq):]))
        return outs


def fmt_metric(m):
    return (f"{m['step']}|{m['reward']:.6f}|{m['exact']:.6f}|"
            f"{m['policy_loss']:.6f}|{m['value_loss']:.6f}|"
            f"{m['entropy']:.6f}|{m['approx_kl']:.6f}")


def metrics_md5(log):
    return hashlib.md5("\n".join(fmt_metric(m) for m in log).encode()).hexdigest()


# ---------------------------------------------------------------------------
# 声明式算术：显存表 + 时钟表（[3] 用）。纯函数，不依赖玩具模型。
# ---------------------------------------------------------------------------
def memory_table(n_workers):
    rows = []
    for label, p in (("7B", COST["p_gb_7b"]), ("13B", COST["p_gb_13b"])):
        colo_train = MemoryLedger(n_workers, p, COST["budget_gb"])
        colo_roll = MemoryLedger(n_workers, p, COST["budget_gb"])
        disagg_train = MemoryLedger(n_workers // 2, p, COST["budget_gb"])
        disagg_roll = MemoryLedger(n_workers // 2, p, COST["budget_gb"])
        rows.append({
            "scale": label,
            "colo_train": colo_train.train_gb(),
            "colo_roll": colo_roll.rollout_gb(),
            "disagg_train": disagg_train.train_gb(),
            "disagg_roll": disagg_roll.rollout_gb(),
            "budget": COST["budget_gb"],
        })
    return rows


def clock_table(n_workers):
    """一步 RL 的声明耗时。cost(k) = base * (N/k)^eta（base 定义在 k=N）。
    sync-colocate : 两相串行，但每相拿满 N 卡 + 一次同步。
    sync-disagg   : 两相串行且各只有 N/2 卡（严格新鲜 rollout 不跨步流水，另一半空转）。
    async-disagg  : 跨步流水（容忍 off-policy staleness），取两相 max。"""
    out = []
    for eta in COST["etas"]:
        half = (n_workers / (n_workers / 2)) ** eta   # = 2^eta
        t_colo = COST["roll_ms"] + COST["train_ms"] + COST["sync_ms"]
        t_disagg_sync = (COST["roll_ms"] + COST["train_ms"]) * half + COST["sync_ms"]
        t_disagg_async = max(COST["roll_ms"], COST["train_ms"]) * half + COST["sync_ms"]
        out.append({"eta": eta, "colo": t_colo, "disagg_sync": t_disagg_sync,
                    "disagg_async": t_disagg_async})
    return out


# ---------------------------------------------------------------------------
# main：[0]-[6] 板块 + self-check
# ---------------------------------------------------------------------------
def main():
    print("nano-verl L3 — HybridFlow colocate: same worker group, two phases")
    print(f"task: '{PROMPT}' -> '{TARGET}' (与 L1/L2 相同)   model: TinyLSTM "
          f"(embed={EMBED_DIM}, hidden={HIDDEN_DIM}, layers={N_LAYERS})")
    print(f"cluster: N={N_WORKERS} simulated ranks   device: {DEVICE} (强制 CPU 保证跨遍确定性)")
    print(f"rl: rollouts={N_ROLLOUTS}, iters={N_ITERATIONS}, ppo_epochs={N_EPOCHS}, lr={LR}")

    # ------------------------------------------------------------------ [0]
    print("\n[0] determinism probe")
    set_seed()
    probe = new_model()
    x = torch.tensor([encode(PROMPT + "he")], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        l1, v1 = probe(x)
        l2, v2 = probe(x)
    diff = (l1 - l2).abs().max().item() + (v1 - v2).abs().max().item()
    print(f"    same input forwarded twice -> max |Δlogits|+|Δvalues| = {diff:.1e}")
    assert diff == 0.0, "CPU 前向必须逐位确定"
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"    params = {n_params:,} (fp32 = {n_params*4/1024:.1f} KB) — 计算是真的，"
          f"显存/时钟按 COST 声明折算")

    # ------------------------------------------------------------------ [1]
    print("\n[1] DataProto + dispatch semantics")
    set_seed()
    base = new_model()
    full_len = len(PROMPT) + MAX_RESP_LEN
    rows, lengths = [], []
    for i in range(MAX_RESP_LEN + 1):
        ids = encode(PROMPT + TARGET[:i])
        lengths.append(len(ids))
        rows.append(ids + [PAD_ID] * (full_len - len(ids)))
    states = torch.tensor(rows * 2, dtype=torch.long)   # 12 行，可被 chunk(4) 整除
    state_lengths = torch.tensor(lengths * 2, dtype=torch.long)
    actions = torch.randint(0, len(VOCAB), (states.shape[0],), generator=torch.Generator().manual_seed(7))
    proto = DataProto({"state": states, "length": state_lengths, "action": actions},
                      meta_info={"probe": 1})

    # (a) chunk -> concat round-trip：行序与数值逐位还原
    rebuilt = DataProto.concat(proto.chunk(4))
    rt_ok = torch.equal(rebuilt.batch["state"], proto.batch["state"]) and \
        torch.equal(rebuilt.batch["length"], proto.batch["length"]) and \
        torch.equal(rebuilt.batch["action"], proto.batch["action"]) and \
        rebuilt.meta_info["probe"] == 1
    print(f"    (a) chunk(4) -> concat round-trip: {'EXACT' if rt_ok else 'FAIL'}")
    assert rt_ok

    # (b) DP 透明性：同一权重下，log_prob 按 rank chunk 算 vs 全量一次算
    def log_prob_rows(model, st, lens, ac):
        model.eval()
        with torch.no_grad():
            logits, _ = model(st)
            idx = torch.arange(st.size(0), device=st.device)
            d = torch.distributions.Categorical(logits=logits[idx, lens - 1, :])
            return d.log_prob(ac)

    base.eval()
    full_lp = log_prob_rows(base, states, state_lengths, actions)
    chunked = []
    for c in proto.chunk(4):
        chunked.append(log_prob_rows(base, c.batch["state"], c.batch["length"],
                                     c.batch["action"]))
    chunked_lp = torch.cat(chunked, dim=0)
    max_diff = (full_lp - chunked_lp).abs().max().item()
    print(f"    (b) DP transparency: chunked log_prob vs full-batch log_prob, "
          f"max |Δ| = {max_diff:.1e} {'(bit-identical)' if max_diff == 0.0 else ''}")
    assert max_diff <= 1e-6, "per-row 前向不应随 chunk 方式改变（CPU）"
    dp_bit_identical = (max_diff == 0.0)

    # ------------------------------------------------------------------ [2]
    print("\n[2] colocated fit (N=4): same ranks serve rollout AND train")
    print("    one step = rollout phase -> [sync: gather+push] -> train phase -> [sync]")
    print("      rollout phase: ranks 0-3 generate (训练态 offloaded, KV cache resident)")
    print("      train   phase: ranks 0-3 update  (rollout engine sleeping)")
    trainer = NanoHybridTrainer("colocate", n_workers=N_WORKERS)
    print(f"    warmup(SFT) loss = {trainer.warmup_loss:.4f}")
    print(f"    sharding: {trainer.n_params} params + {trainer.pad_len} pad = "
          f"{trainer.n_params + trainer.pad_len} "
          f"({(trainer.n_params + trainer.pad_len)//N_WORKERS}/rank, FSDP 式 pad 到整除)")
    log_colo = trainer.fit()
    print(f"    greedy samples after training: {trainer.greedy_samples()}")
    led = trainer.ledger
    print(f"    declared memory/rank @7B: train {led.train_gb():.1f} GB / "
          f"rollout {led.rollout_gb():.1f} GB (budget {led.budget:.0f} GB)")
    tb = sum(trainer.traffic[k] for k in
             ("sync_gather_bytes", "sync_replica_write_bytes", "train_gather_bytes", "grad_reduce_bytes"))
    print(f"    real toy traffic: {tb/1024:.0f} KB moved across {trainer.traffic['syncs']} syncs "
          f"+ {N_ITERATIONS*N_EPOCHS} epoch gather/reduce "
          f"(declared-scale formula in [3])")

    # ------------------------------------------------------------------ [3]
    print("\n[3] declared-scale arithmetic (COST model, not measured)")
    print("    memory per rank (GB):")
    mem = memory_table(N_WORKERS)
    print(f"      {'scale':6s} {'colo-train':>11s} {'colo-rollout':>13s} "
          f"{'disagg-train':>13s} {'disagg-rollout':>15s} budget")
    for r in mem:
        print(f"      {r['scale']:6s} {r['colo_train']:11.1f} {r['colo_roll']:13.1f} "
              f"{r['disagg_train']:13.1f} {r['disagg_roll']:15.1f} {r['budget']:6.0f}")
    fit_colo_7b = mem[0]['colo_train'] <= mem[0]['budget'] and mem[0]['colo_roll'] <= mem[0]['budget']
    fit_colo_13b = mem[1]['colo_train'] <= mem[1]['budget'] and mem[1]['colo_roll'] <= mem[1]['budget']
    oom_disagg_7b = mem[0]['disagg_train'] > mem[0]['budget']
    oom_disagg_13b = mem[1]['disagg_train'] > mem[1]['budget']
    print(f"      colocate  fits @7B: {fit_colo_7b}   fits @13B: {fit_colo_13b}")
    print(f"      disagg trainer OOM @7B: {oom_disagg_7b}   OOM @13B: {oom_disagg_13b}")
    print("      (disagg 把 N/2 张卡的训练态压到一半卡上: 4P/(N/2) = 2x colocate 的 4P/N)")

    print("    wall-clock per RL step (declared ms):")
    clk = clock_table(N_WORKERS)
    print(f"      {'eta':5s} {'colo(N)':>9s} {'disagg-sync(N/2)':>17s} {'disagg-async':>13s} "
          f"{'colo vs sync-disagg':>20s}")
    for r in clk:
        print(f"      {r['eta']:<5.2f} {r['colo']:9.0f} {r['disagg_sync']:17.0f} "
              f"{r['disagg_async']:13.0f} {r['disagg_sync']/r['colo']:19.2f}x")
    print("      sync-disagg 的隐性成本: 严格新鲜 rollout 不跨步流水 -> 每相都有一半边卡空转")
    print("      async-disagg 用 staleness 换重叠 (off-policy 代价, 见 nano-slime L0)")

    # ------------------------------------------------------------------ [4]
    print("\n[4] scheduling invariance")
    print("    [4a] flat (same N=4, same seed, no phase switches):")
    trainer_flat = NanoHybridTrainer("flat", n_workers=N_WORKERS, verbose=False)
    log_flat = trainer_flat.fit()
    same = [fmt_metric(a) for a in log_colo] == [fmt_metric(b) for b in log_flat]
    print(f"         per-step metrics bit-identical to [2]: {same}")
    assert same, "调度机制（切换/账本）不得改变训练数值"

    print("    [4b] disagg (2 rollout ranks + 2 train ranks, same seed):")
    trainer_dis = NanoHybridTrainer("disagg", n_workers=N_WORKERS, verbose=False)
    log_dis = trainer_dis.fit()
    max_reward_diff = max(abs(a["reward"] - b["reward"]) for a, b in zip(log_colo, log_dis))
    final_colo, final_dis = log_colo[-1]["reward"], log_dis[-1]["reward"]
    print(f"         reward curve max |Δ| vs [2] = {max_reward_diff:.4f}")
    print(f"         (DP 宽度不同 -> 每 rank 的采样划分与梯度归约树都不同 -> 轨迹不逐位相同;")
    print(f"          不变的是收敛行为, 不是轨迹——bit 级不变性只在同宽度下成立, 见 [4a])")
    print(f"         final reward: colocate={final_colo:.3f}, disagg={final_dis:.3f}")
    disagg_parity = (max_reward_diff <= 0.15 and final_colo >= 0.9
                     and final_dis >= 0.9 and abs(final_colo - final_dis) <= 0.05)
    assert disagg_parity, "不同 DP 宽度应收敛可比（同任务同 seed 族）"

    # ------------------------------------------------------------------ [5]
    print("\n[5] determinism re-run")
    trainer2 = NanoHybridTrainer("colocate", n_workers=N_WORKERS, verbose=False)
    log_colo2 = trainer2.fit()
    h1, h2 = metrics_md5(log_colo), metrics_md5(log_colo2)
    print(f"    md5(metrics) run#1 = {h1}")
    print(f"    md5(metrics) run#2 = {h2}   identical: {h1 == h2}")
    assert h1 == h2, "同 seed 跨遍必须逐位一致（CPU）"

    # ------------------------------------------------------------------ [6]
    print("\n[6] self-check")
    checks = [
        ("deterministic forward (CPU)", True),
        ("chunk/concat round-trip exact", rt_ok),
        (f"DP transparency (max|Δ|={max_diff:.1e})", max_diff <= 1e-6),
        ("colocate memory fits budget @7B (declared)", fit_colo_7b),
        ("colocate memory fits budget @13B (declared)", fit_colo_13b),
        ("disagg trainer OOM @13B (declared) — memory wall", oom_disagg_13b),
        ("scheduling numerically transparent ([2]==[4a])", same),
        (f"disagg convergence parity (curve max|Δ|={max_reward_diff:.4f}, "
         f"finals {final_colo:.3f}/{final_dis:.3f})", disagg_parity),
        (f"learning: final reward {final_colo:.3f} >= 0.9", final_colo >= 0.9),
        ("cross-run determinism (md5)", h1 == h2),
    ]
    all_ok = True
    for name, ok in checks:
        print(f"    [{'pass' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    assert all_ok, "self-check failed"
    print(f"\n    ✅ self-check passed ({len(checks)}/{len(checks)})")
    print("\ntakeaway: 在严格新鲜 rollout 的同步配置下，colocate 的核心是显存与资源算术：")
    print("          拆卡 = 每相一半算力 + 一半卡空转；复用 = 每相全部算力 + 阶段边界付")
    print("          resharding 税。模型越大，训练态越住不满拆出去的卡，colocate 越赢。")
    print("          真实 verl 就是这套语义：RayPPOTrainer 单控制器编排，")
    print("          ActorRolloutRefWorker 同卡两相，rollout_mode() 做 gather+sync。")


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nelapsed: {time.time()-_t0:.1f}s (计算真跑; 显存/时钟数字来自 COST 声明模型)")
