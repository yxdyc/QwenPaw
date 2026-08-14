#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nano-trinity-rft L3 — 配置即实验台：schema、注册表与 ablation ladder

L0–L2 讲了统一数据流（L0/L1）与 reward 信号来源（L2）。L3 回答最后一个问题：
**为什么 Trinity-RFT 能用一份 YAML 就切换十余种算法、跑一组消融？** 答案是
配置系统本身被当成一等公民来设计：
  (1) schema——每个算法决策都是 Config 里的具名字段（algorithm_type /
      repeat_times / policy_loss_fn / kl_loss_fn / loss_agg_mode / stages ...，
      trinity/common/config.py 共 1063 行、34 个 dataclass）；
  (2) 注册表 + 宏开关展开——algorithm_type 是宏开关，经 ALGORITHM_TYPE 注册表
      （trinity/algorithm/__init__.py:L9-36，24 个条目）解析成
      policy_loss_fn / advantage_fn / kl_loss_fn / ... 的微观组合；
      三层优先级 = 用户显式设置 > 算法默认 > 全局兜底
      （config_validator.py:L385-398：先 check_config 校验修复，再
      default_config.update + set_if_none 合并）；
  (3) 多阶段 = stages 列表覆盖——Config.__iter__ 对每个 StageConfig deepcopy
      基座、覆盖非 None 字段、名字加后缀（config.py:L978-995），SFT→RL 课程
      就是两条 stage 记录（examples/grpo_gsm8k/gsm8k.yaml 尾部注释即模板）。
本级在 toy 上复现这套机制，并用它跑一个 DAPO 式的 ablation ladder：
阶梯每一格 = dapo.yaml 里的一行开关（no-KL / Clip-Higher / Dynamic Sampling /
Overlong Shaping，对照 examples/dapo_math/README.md 的「paper technique →
Trinity wiring」表），全部由 nano schema 的字段切换驱动，训练循环一字不改。

对标权威实现（agentscope-ai/Trinity-RFT main 分支，2026-08-13 现场克隆核验，
HEAD 009850b1，README 30,381 B sha256 d513f140…b73982 与 L1/L2 录值逐位零漂移；
行号以 2026-08-13 抓取日为准）：
  - schema：config.py:L618-662 AlgorithmConfig / L729-790 ExplorerConfig /
    L806-849 TrainerConfig / L927-938 StageConfig / L940-1052 Config；
  - 宏开关：algorithm.py:L48-118 SFT / L120-140 GRPO / L142-175 DAPO /
    L263-305 DPO（default_config + use_critic/use_reference/schema 类级标志 +
    check_config 修复——DPO 强制 kl_loss k2、SFT 拒绝 both 模式）；
  - 三层合并：config_validator.py:L385-398（AlgorithmConfigValidator.validate）；
  - PPO 损失：policy_loss_fn/ppo_policy_loss.py:L69-95（ratio=exp(logp−old_logp)
    截断 ±20、非对称 clip(1−low, 1+high)、clip_ratio_c 只作用 adv<0、
    loss_agg_mode 聚合），改自 verl core_algos.py；
  - 聚合四模式：algorithm/utils.py:L9-49（token-mean / seq-mean-token-sum /
    seq-mean-token-mean / seq-mean-token-sum-norm）；
  - GRPO advantage：advantage_fn/grpo_advantage.py:L160-163（std≤阈值整组
    clear）/ L166-169（(r−mean)/(std+ε)）/ L178-194（duplicate_experiences
    从活组随机复制补位）；
  - KL k2：kl_fn/kl_fn.py K2Fn = 0.5·(logp−ref_logp)²；KL_FN 注册表 7 项
    （kl_fn/__init__.py:L4-15）；
  - 多组件 reward：workflows/math_rm_workflow.py:L36-44（reward_dict 分账进
    metrics、sum 成训练标量）；dapo_reward.py:L58-97（accuracy ±1 +
    overlong 软惩罚：len < max−cache → 0，线性降到 max 处 −factor）；
  - DAPO 开关表：examples/dapo_math/README.md:L11-16（paper technique →
    Trinity wiring 六行表）+ dapo.yaml（clip 0.2/0.28、kl none、
    dapo_dynamic_sampling + mask_response_truncated 管线算子、overlong
    reward_fn_args）；dapo_dynamic_sampling 实现 = buffer/operators/filters/
    reward_filter.py:L65-148（按 metrics["accuracy"] 判对错，保留
    0<#correct<G 的组，dropped_all_correct / dropped_all_wrong 分开记账）；
  - 三组件循环：explorer/explorer.py:L404-405 need_eval（step % eval_interval）/
    trainer/trainer.py:L104-178 train 主循环（sample→train_step→need_sync→
    sync_weight→need_save→save_checkpoint）。
其余引用：DAPO arXiv:2503.14476（§3.2 Dynamic Sampling / §3.3 Token-Level
Loss / §3.4 Overlong Reward Shaping）；GRPO arXiv:2402.03300；PPO clip
arXiv:1707.06347；Trinity 论文 arXiv:2505.17826。
源码考古发现（如实记录）：algorithm.py:L146-148 DAPOAlgorithm docstring 与
examples/dapo_math/README.md:L3 均引用 docs/dapo_trinity_implementation_spec.md，
该文件在 2026-08-13 抓取的 main 树中不存在（dangling reference）。

Toy 口径（显式声明：合成任务，只演机制）：沿用 L1/L2 的字符级 GPT（~0.8M 参数）
与同一张任务表（SEED=20260806 同源，targets 逐位一致）。本级唯一扩展：
**响应变长**——采样到 EOS 或 8 token 为止（L1/L2 固定 4 字符）。原因：
loss_agg_mode 与 overlong 惩罚这两个开关在定长响应上没有任何可观察差异，
变长是它们生效的最小前提。SFT 目标 = 4 个目标字符 + EOS（L1/L2 的 SFT 本就
含 EOS 位，口径未变）。
依赖：仅 torch。CPU 单文件，任意 CWD 可跑（python3 -B）。固定 seed → 指标行
逐字节确定，elapsed 计时行随机器负载浮动（掩码口径
sed '/^[[:space:]]*elapsed/d'，继承 L1/L2 先例）。
"""

import sys
import time
import math
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
SEED = 20260806          # 与 L1/L2 同源：同一张任务表、同一条初始化 RNG 流

# ----------------------------- [A] 任务与词表（L1/L2 同构 + 变长扩展） -----------------------------
ALPHABET = "abcd"
RESP_LEN = 4             # 目标长度
MAX_RESP_LEN = 8         # 采样上限（超过即 truncated，对应 Trinity 的 max_response_tokens）
N_CTX = 6
TEACHER_CTX = [0, 1, 2]


def make_targets(seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(ALPHABET), (N_CTX, RESP_LEN), generator=g)
    return ["".join(ALPHABET[int(i)] for i in row) for row in idx]


TARGETS = make_targets(SEED)   # 与 L1/L2 逐位一致：['acbb','daab','bcbd','dcca','bbba','cbba']

VOCAB = ["<pad>", "<bos>", "<eos>", ":", "k"] + [str(d) for d in range(N_CTX)] + list(ALPHABET)
ID = {c: i for i, c in enumerate(VOCAB)}
PAD, BOS, EOS = ID["<pad>"], ID["<bos>"], ID["<eos>"]
ALPHABET_IDS = [ID[c] for c in ALPHABET]


def prompt_ids(c):
    return [ID["k"], ID[str(c)], ID[":"]]


def target_ids(c):
    """SFT 目标 = 4 个字符 + EOS（L1/L2 同口径：SFT 本就教『答完就停』）。"""
    return [ID[ch] for ch in TARGETS[c]] + [EOS]


def decode(resp):
    """响应文本 = 首个 EOS 之前的 token（无 EOS 则全部）。"""
    out = []
    for t in resp:
        if t == EOS:
            break
        out.append(VOCAB[t])
    return "".join(out)


# ----------------------------- [B] nano config schema（字段名对齐 trinity/common/config.py） -----------------------------
@dataclass
class OptimizerConfig:                      # config.py:L90
    lr: float = 1e-3


@dataclass
class AlgorithmConfig:                      # config.py:L618-662 的 nano 切片
    algorithm_type: str = "grpo"            # 宏开关
    repeat_times: int = None                # GRPO 组大小 G（Trinity GRPO 默认 2，gsm8k.yaml 设 8）
    policy_loss_fn: str = None              # "ppo" / "sft"
    policy_loss_fn_args: dict = None        # clip_range_low/high、loss_agg_mode 的家
    advantage_fn: str = None                # "grpo"
    advantage_fn_args: dict = None          # std_threshold / duplicate_experiences 的家
    kl_loss_fn: str = None                  # "k2" / "none"
    kl_loss_fn_args: dict = None            # kl_coef
    loss_agg_mode: str = None               # token-mean / seq-mean-token-sum（config.py:L652-654）
    reward_fn: str = "accuracy"             # nano 侧 reward 选择（Trinity 在 workflow/taskset 层）
    reward_fn_args: dict = None             # enable_overlong_penalty 等（dapo.yaml 口径）


@dataclass
class ExplorerConfig:                       # config.py:L729-790 的 nano 切片
    eps: float = 0.3                        # ε-探索（L1 教训：toy 必须显式维护探索）
    eval_interval: int = 1                  # config.py:L765-766


@dataclass
class TrainerConfig:                        # config.py:L806-849 的 nano 切片
    total_steps: int = None
    grad_clip: float = 1.0                  # config.py:L822
    ppo_epochs: int = 3                     # 每批 rollout 训练步数（ratio/clip 因此存在）
    lr_scale_rl: float = 0.3                # RL 阶段 lr 缩放（token-mean 聚合把梯度除以总
                                            # token 数，有效步长比 L2 的 sum 语义小 ~5 倍，
                                            # toy 档位相应放大；所有臂同口径，公平对照）


@dataclass
class BufferConfig:                         # config.py:L708-727 的 nano 切片
    batch_size: int = 8                     # 每步任务数（nano：每 ctx 一组）


@dataclass
class StageConfig:                          # config.py:L927-938
    stage_name: str = ""
    mode: str = None                        # explore / train / both
    algorithm: AlgorithmConfig = None
    buffer: BufferConfig = None
    trainer: TrainerConfig = None


@dataclass
class Config:                               # config.py:L940-1052 的 nano 切片
    mode: str = "both"                      # config.py:L943
    name: str = "rft"
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    explorer: ExplorerConfig = field(default_factory=ExplorerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    stages: list = field(default_factory=list)   # config.py:L971

    def __iter__(self):
        """多阶段 = 基座 + stage 覆盖（镜像 Config.__iter__，config.py:L978-995）：
        deepcopy 基座 → 覆盖 stage 中非 None 的同名字段 → 名字加 /stage_name 后缀。
        Trinity 还强制 save_hf_checkpoint='last'（L993-994）保证下一阶段能加载，
        nano 侧对应物 = 内存 state_dict 交接（[D3] 实验）。"""
        for stage in self.stages:
            new_config = deepcopy(self)
            for f in fields(stage):
                stage_value = getattr(stage, f.name)
                if stage_value is not None and hasattr(new_config, f.name):
                    setattr(new_config, f.name, stage_value)
            if stage.stage_name:
                new_config.name = f"{self.name}/{stage.stage_name}"
            new_config.stages = []
            yield new_config


# ----------------------------- [C] 注册表与宏开关（nano 版 algorithm.py + __init__.py） -----------------------------
class Registry:
    """trinity/utils/registry.py:L7-47 的最小形态：字符串 → 类。"""

    def __init__(self, name):
        self.name, self._modules = name, {}

    def register(self, key, cls):
        self._modules[key] = cls
        return cls

    def get(self, key):
        if key not in self._modules:
            raise KeyError(f"[{self.name}] unknown key: {key!r} "
                           f"(registered: {sorted(self._modules)})")
        return self._modules[key]

    def keys(self):
        return sorted(self._modules)


ALGORITHM_TYPE = Registry("algorithm")      # trinity/algorithm/__init__.py:L9-36（24 项）


class AlgorithmType:
    """algorithm.py:L21-46 的 nano 形态：类级标志 + default_config + check_config。"""
    use_reference: bool = False
    schema: str = "experience"

    @classmethod
    def default_config(cls):
        raise NotImplementedError

    @classmethod
    def check_config(cls, cfg):
        """校验并修复（Trinity 同语义：check_config 会改写 config，
        如 DPOAlgorithm 强制 kl_loss k2，algorithm.py:L293-305）。返回修复日志。"""
        return []


class SFTAlgorithm(AlgorithmType):          # algorithm.py:L48-118
    use_reference = False
    schema = "sft"

    @classmethod
    def default_config(cls):
        return {"policy_loss_fn": "sft", "kl_loss_fn": "none", "loss_agg_mode": "token-mean"}

    @classmethod
    def check_config(cls, cfg):
        fixed = []
        if cfg.mode != "train":
            raise ValueError("`algorithm_type: sft` does not support "
                             f"`mode: {cfg.mode}`（镜像 SFTAlgorithm.check_config，"
                             "algorithm.py:L67-76：SFT 只支持 train 模式）")
        return fixed


class GRPOAlgorithm(AlgorithmType):         # algorithm.py:L120-140
    use_reference = True
    schema = "experience"

    @classmethod
    def default_config(cls):
        return {"repeat_times": 2, "policy_loss_fn": "ppo", "advantage_fn": "grpo",
                "kl_loss_fn": "k2", "loss_agg_mode": "token-mean"}


class DAPOAlgorithm(AlgorithmType):         # algorithm.py:L142-175
    use_reference = False                   # DAPO 不用 reference（kl none 的根因）

    @classmethod
    def default_config(cls):
        return {"repeat_times": 16, "policy_loss_fn": "ppo", "advantage_fn": "grpo",
                "kl_loss_fn": "none", "loss_agg_mode": "token-mean"}


class DPOAlgorithm(AlgorithmType):          # algorithm.py:L263-305
    use_reference = True
    schema = "dpo"

    @classmethod
    def default_config(cls):
        return {"policy_loss_fn": "dpo", "kl_loss_fn": "k2", "loss_agg_mode": "token-mean"}

    @classmethod
    def check_config(cls, cfg):
        fixed = []
        if cfg.algorithm.kl_loss_fn in ("none", None):
            cfg.algorithm.kl_loss_fn = "k2"
            fixed.append("kl_loss_fn: none → k2（DPO must use KL loss，"
                         "algorithm.py:L302-304）")
        if cfg.algorithm.repeat_times != 2:
            cfg.algorithm.repeat_times = 2
            fixed.append("repeat_times → 2（Fake repeat times，algorithm.py:L300-301）")
        return fixed


ALGORITHM_TYPE.register("sft", SFTAlgorithm)
ALGORITHM_TYPE.register("grpo", GRPOAlgorithm)
ALGORITHM_TYPE.register("dapo", DAPOAlgorithm)
ALGORITHM_TYPE.register("dpo", DPOAlgorithm)

# 全局兜底默认（config_validator.py:L387-395 的 nano 切片）
GLOBAL_ALGO_DEFAULTS = {
    "policy_loss_fn": "ppo",
    "advantage_fn": "ppo",
    "kl_loss_fn": "k2",
    "loss_agg_mode": "token-mean",
}


def resolve_algorithm(cfg):
    """三层优先级合并（镜像 AlgorithmConfigValidator.validate，
    config_validator.py:L385-398）：
      用户显式设置 > algorithm_type 默认 > 全局兜底。
    顺序 = check_config（校验修复）→ default_config.update(全局) → set_if_none。
    返回 (resolved dict, provenance dict)；provenance 记录每个字段的出处，
    这是『配置为什么长这样』的可追溯账本。"""
    algo_cls = ALGORITHM_TYPE.get(cfg.algorithm.algorithm_type)
    fixed = algo_cls.check_config(cfg)
    merged = dict(GLOBAL_ALGO_DEFAULTS)          # 层 3：全局兜底
    merged.update(algo_cls.default_config())     # 层 2：算法默认覆盖全局
    resolved, prov = {}, {}
    user = {f.name: getattr(cfg.algorithm, f.name) for f in fields(AlgorithmConfig)}
    for k, v in merged.items():
        if user.get(k) is not None:              # 层 1：用户显式设置最高优先
            resolved[k] = user[k]
            prov[k] = "user"
        else:
            resolved[k] = v
            prov[k] = "algorithm" if k in algo_cls.default_config() else "global"
    for k, v in user.items():                    # 不在 merged 表里的用户字段（reward_fn 等）
        if v is not None and k not in resolved:
            resolved[k], prov[k] = v, "user"
    return algo_cls, resolved, prov, fixed


# ----------------------------- [D] reward（多组件字典：分账 + 求和） -----------------------------
def reward_accuracy(c, resp):
    """稀疏 0/1（AccuracyReward 形态，accuracy_reward.py:L61-67）。"""
    return {"accuracy": 1.0 if decode(resp) == TARGETS[c] else 0.0}


def reward_dense(c, resp):
    """逐位匹配率（L1/L2 的 dense rule reward；warm 阶段用，把策略推进
    『偶尔会对』窗口——稀疏信号的存在性前提，L2 §4）。"""
    t = TARGETS[c]
    m = sum(1 for j in range(RESP_LEN)
            if j < len(resp) and VOCAB[resp[j]] == t[j])
    return {"accuracy": m / RESP_LEN}


def overlong_penalty(resp_len, max_len, cache_len, factor):
    """Trinity 原式（dapo_reward.py:L71-97）：len < max−cache → 0；
    之后线性降到 max 处 −factor；超过 max → −factor（截断响应另由
    mask_response_truncated 处理，reward_filter.py:L151-172）。"""
    expected = max_len - cache_len
    if resp_len < expected:
        return 0.0
    if resp_len > max_len:
        return -factor
    return (expected - resp_len) / cache_len * factor


def make_reward_fn(resolved):
    """reward = 多组件字典求和（math_rm_workflow.py:L36-44：各组件先分账进
    metrics，再 sum 成训练标量）。nano 侧组件 = accuracy（rule）+ 可选
    format_score（overlong 软惩罚，dapo.yaml reward_fn_args 口径）。"""
    base = reward_dense if resolved.get("reward_fn") == "dense" else reward_accuracy
    rargs = resolved.get("reward_fn_args") or {}
    enable_overlong = rargs.get("enable_overlong_penalty", False)

    def fn(c, resp):
        d = dict(base(c, resp))
        if enable_overlong:
            d["format_score"] = overlong_penalty(
                len(resp), rargs["max_response_length"],
                rargs["cache_length"], rargs["penalty_factor"])
        return d
    return fn


# ----------------------------- [E] 模型与探针（L2 同构 + 变长采样） -----------------------------
class Block(nn.Module):
    def __init__(self, d, nhead, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=128, nhead=4, nlayers=4, ff=512, maxpos=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxpos, d)
        self.blocks = nn.ModuleList([Block(d, nhead, ff) for _ in range(nlayers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        n = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(n))
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


@torch.no_grad()
def generate(model, c, greedy=False, temp=1.0, eps=0.0):
    """变长采样：直到 EOS 或 MAX_RESP_LEN。eps>0 时按 ε-探索（L1/L2  lineage）：
    以概率 eps 从字母表均匀采（探索不产生 EOS——toy 里『跑题』的形态）。"""
    ids = [BOS] + prompt_ids(c)
    out = []
    for _ in range(MAX_RESP_LEN):
        logits = model(torch.tensor([ids]))[0, -1]
        if greedy:
            t = int(logits.argmax())
        elif torch.rand(1).item() < eps:
            t = ALPHABET_IDS[int(torch.randint(len(ALPHABET_IDS), (1,)))]
        else:
            t = int(torch.multinomial((logits / temp).softmax(-1), 1))
        ids.append(t)
        out.append(t)
        if t == EOS:
            break
    return out


def evaluate(model):
    ex, ca = [], []
    for c in range(N_CTX):
        resp = generate(model, c, greedy=True)
        d = reward_accuracy(c, resp)
        ex.append(d["accuracy"])
        ca.append(reward_dense(c, resp)["accuracy"])
    return sum(ex) / N_CTX, sum(ca) / N_CTX


@torch.no_grad()
def entropy_probe(model):
    """策略熵：6 ctx × 4 目标位，沿贪心前缀的逐位分布熵均值。
    在 toy 里它首先是饱和诊断量：SFT 把目标路径打到 p≈1.0 → H≈0，DAPO 的
    『熵坍缩/Clip-Higher 保熵』叙事（dapo_math/README.md:L38）在本尺度不可观察，
    clip 机制改由 clipfrac 与填洞速度度量（§[2] 注）。"""
    hs = []
    for c in range(N_CTX):
        ids = [BOS] + prompt_ids(c)
        for j in range(RESP_LEN):
            p = model(torch.tensor([ids]))[0, -1].softmax(-1)
            hs.append(float(-(p * (p + 1e-12).log()).sum()))
            ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))
    return sum(hs) / len(hs)


@torch.no_grad()
def drift_probe(model, ref):
    """k2 式漂移：0.5·(logp_θ − logp_ref)² 在空洞 ctx（3–5）目标位上的均值
    （K2Fn，kl_fn.py:L165-177）。度量『RL 把策略从 warm 快照拉出去多远』。
    测空洞而非覆盖 ctx：SFT 把覆盖 ctx 的目标路径饱和到 p=1.0（logp 恒 0、
    梯度归零），探针在那边恒为 0——饱和本身是 toy 尺度的一个发现（§[2]）。"""
    ks = []
    for c in [3, 4, 5]:
        ids = torch.tensor([[BOS] + prompt_ids(c) + target_ids(c)])
        logp = F.log_softmax(model(ids), dim=-1)
        rlogp = F.log_softmax(ref(ids), dim=-1)
        pos = slice(3, 3 + len(target_ids(c)))
        tgt = ids[:, 4:4 + len(target_ids(c))]
        lp = logp[:, pos].gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        rp = rlogp[:, pos].gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        ks.append(float((0.5 * (lp - rp).square()).mean()))
    return sum(ks) / len(ks)


# ----------------------------- [F] Explorer / advantage / Trainer -----------------------------
class Sample:
    __slots__ = ("kind", "ctx", "resp", "mask", "reward", "comps", "adv",
                 "old_logp", "truncated")

    def __init__(self, kind, ctx, resp):
        self.kind, self.ctx, self.resp = kind, ctx, resp
        # action_mask：响应位（含 EOS）为 1；truncated 响应可被整体置 0
        # （mask_response_truncated，reward_filter.py:L158-172）
        self.mask = [1] * len(resp)
        self.truncated = (len(resp) == MAX_RESP_LEN and resp[-1] != EOS)
        self.reward, self.comps, self.adv = 0.0, {}, [0.0] * len(resp)
        self.old_logp = None


class Explorer:
    def __init__(self, model, reward_fn, eps):
        self.model, self.reward_fn, self.eps = model, reward_fn, eps

    def sft_batch(self):
        return [Sample("sft", c, target_ids(c)) for c in TEACHER_CTX]

    @torch.no_grad()
    def _logp_of(self, c, resp):
        ids = torch.tensor([[BOS] + prompt_ids(c) + resp])
        logp = F.log_softmax(self.model(ids), dim=-1)
        return logp[0, 3:3 + len(resp)].gather(1, torch.tensor(resp).unsqueeze(-1)).squeeze(-1)

    def group(self, c, g):
        """一个 ctx 的 g 条 rollout（repeat_times=G）：采响应 → 多组件 reward
        求和 → 记录 old_logp（ratio 的分母，rollout 时刻的策略）。"""
        samps = []
        for _ in range(g):
            resp = generate(self.model, c, eps=self.eps)
            s = Sample("rl", c, resp)
            s.comps = self.reward_fn(c, resp)
            s.reward = sum(s.comps.values())        # math_rm_workflow.py:L43
            s.old_logp = self._logp_of(c, resp)
            samps.append(s)
        return samps

    def rollout(self, g):
        per_ctx = [self.group(c, g) for c in range(N_CTX)]
        return per_ctx


def grpo_advantage(per_ctx, resolved):
    """GRPO 组内优势（grpo_advantage.py 语义）：
      score = (r − group_mean) / (group_std + ε)          （L166-169）
      std ≤ std_threshold 的组 exps.clear() 整组丢弃       （L160-163）
      duplicate_experiences：从活组随机复制补回组数        （L178-194）
    返回 (kept_samples, n_dead, n_dup_groups)。
    注：L2 的 nano dyn 是『过滤+补采新 rollout』；Trinity 的
    duplicate_experiences 是『从已有活组复制』——复制不花 rollout，
    但带来重复样本（nano 侧 max_reuse 限制兜底）。"""
    aargs = resolved.get("advantage_fn_args") or {}
    thr = aargs.get("std_threshold", None)
    dup = aargs.get("duplicate_experiences", False)
    kept, n_dead, groups = [], 0, []
    for gs in per_ctx:
        rs = torch.tensor([s.reward for s in gs])
        std = float(rs.std()) if len(gs) > 1 else 0.0
        if std <= 1e-12:
            n_dead += 1                             # 观测口径：组内 reward 全同（L2 §5 的 dead 组）
        if thr is not None and std <= thr:
            continue                                # exps.clear()（过滤只在设阈值时发生）
        groups.append(gs)
    n_dup = 0
    if dup and groups:
        # 被丢的组用活组随机复制补位（_duplicate_experiences，L178-194）
        n_missing = len(per_ctx) - len(groups)
        for _ in range(n_missing):
            src = groups[int(torch.randint(len(groups), (1,)))]
            copies = [deepcopy(s) for s in src]
            groups.append(copies)
            n_dup += 1
    for gs in groups:
        rs = torch.tensor([s.reward for s in gs])
        mean, std = float(rs.mean()), float(rs.std()) if len(gs) > 1 else 0.0
        for s, r in zip(gs, rs.tolist()):
            score = (r - mean) / (std + 1e-6)
            s.adv = [score] * len(s.resp)           # outcome 级广播（GRPO 形态）
        kept.extend(gs)
    return kept, n_dead, n_dup


def warm_advantage(per_ctx):
    """warm 阶段专用：dense reward 的逐位组内优势（L1/L2 的 credit assignment，
    tutorial_L1 §6：toy 的空洞只有逐位信号才填得动）。Trinity 的 advantage_fn
    只消费标量 reward（grpo_advantage.py:L166-169），逐位 dense 是 nano 侧压缩——
    ladder 各臂用稀疏 accuracy reward，走纯 GRPO outcome 优势，不受此影响。"""
    kept = []
    for c, gs in enumerate(per_ctx):
        g = len(gs)
        matches = [[1.0 if j < len(s.resp) and VOCAB[s.resp[j]] == TARGETS[c][j] else 0.0
                    for j in range(RESP_LEN)] for s in gs]
        mean_j = [sum(m[j] for m in matches) / g for j in range(RESP_LEN)]
        for s, m in zip(gs, matches):
            adv4 = [m[j] - mean_j[j] for j in range(RESP_LEN)]
            s.adv = adv4 + [0.0] * (len(s.resp) - RESP_LEN)   # 4 位之后（含 EOS）零优势
        kept.extend(gs)
    return kept, 0, 0


class Trainer:
    """PPO-ratio 训练器（ppo_policy_loss.py:L69-95 的 nano 形态）：
      ratio = exp(clamp(logp − old_logp, ±20))
      L = max(−adv·ratio, −adv·clip(ratio, 1−low, 1+high))   （非对称 clip）
      adv<0 时再与 −adv·clip_ratio_c 取 min（DAPO Clip Ratio C）
      + kl_loss（k2：0.5·(logp−ref)²）· kl_coef
      聚合按 loss_agg_mode（utils.py:L9-49 四模式）。
    SFT 走同一训练器的 nll 分支（algorithm_type=sft → policy_loss_fn=sft）。"""

    def __init__(self, model, resolved, lr):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        pargs = resolved.get("policy_loss_fn_args") or {}
        self.clip_low = pargs.get("clip_range_low", 0.2)
        self.clip_high = pargs.get("clip_range_high", self.clip_low)
        self.clip_c = pargs.get("clip_ratio_c", 3.0)
        self.agg = resolved.get("loss_agg_mode", "token-mean")
        kargs = resolved.get("kl_loss_fn_args") or {}
        self.kl_coef = kargs.get("kl_coef", 0.001)
        self.use_kl = resolved.get("kl_loss_fn", "none") != "none"
        self.policy_loss = resolved.get("policy_loss_fn", "ppo")
        self.grad_clip = 1.0
        self.ref = None
        self.version = 0
        self.clipfrac_sum = self.clipfrac_n = 0

    def set_ref(self, ref):
        self.ref = ref

    def _aggregate(self, vals, mask):
        """utils.py:L9-49 的四模式聚合（nano：vals/mask 为变长 list of tensor）。
        分子保持 tensor（autograd 图）；只有分母取 float。"""
        if self.agg == "token-mean":
            tot = sum((v * m).sum() for v, m in zip(vals, mask))
            den = sum(float(m.sum()) for m in mask) + 1e-8
            return tot / den
        elif self.agg == "seq-mean-token-sum":
            return sum((v * m).sum() for v, m in zip(vals, mask)) / len(vals)
        elif self.agg == "seq-mean-token-mean":
            return sum((v * m).sum() / (float(m.sum()) + 1e-8)
                       for v, m in zip(vals, mask)) / len(vals)
        else:  # seq-mean-token-sum-norm：总和 / 固定 normalizer（最大长度）
            tot = sum((v * m).sum() for v, m in zip(vals, mask))
            return tot / (MAX_RESP_LEN + 1e-8)

    def step_sft(self, batch):
        ids = torch.tensor([[BOS] + prompt_ids(s.ctx) + s.resp for s in batch])
        L = len(batch[0].resp)
        logits = self.model(ids)
        loss = F.nll_loss(logits[:, 3:3 + L].reshape(-1, len(VOCAB)),
                          ids[:, 4:4 + L].reshape(-1))
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        self.version += 1
        return float(loss.detach())

    def step_rl(self, batch):
        vals, masks = [], []
        kl_tot, n_tok = 0.0, 0
        for s in batch:
            ids = torch.tensor([[BOS] + prompt_ids(s.ctx) + s.resp])
            logp = F.log_softmax(self.model(ids), dim=-1)
            pos = slice(3, 3 + len(s.resp))
            lp = logp[0, pos].gather(1, torch.tensor(s.resp).unsqueeze(-1)).squeeze(-1)
            neg_kl = torch.clamp(lp - s.old_logp, -20.0, 20.0)   # ppo_policy_loss.py:L72-76
            ratio = neg_kl.exp()
            adv = torch.tensor(s.adv)
            l1 = -adv * ratio
            l2 = -adv * torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
            lc = torch.maximum(l1, l2)
            if self.clip_c < 1e5:                   # clip_ratio_c 只作用 adv<0（L88-92）
                l3 = -adv * self.clip_c
                lc = torch.where(adv < 0, torch.minimum(l3, lc), lc)
            mask = torch.tensor(s.mask, dtype=torch.float)
            vals.append(lc)
            masks.append(mask)
            self.clipfrac_sum += float(((l2 > l1).float() * mask).sum())
            self.clipfrac_n += float(mask.sum())
            if self.use_kl and self.ref is not None:
                with torch.no_grad():
                    rlogp = F.log_softmax(self.ref(ids), dim=-1)
                rp = rlogp[0, pos].gather(1, torch.tensor(s.resp).unsqueeze(-1)).squeeze(-1)
                kl_tok = 0.5 * (lp - rp).square()   # K2Fn（kl_fn.py:L165-177）
                kl_tot = kl_tot + (kl_tok * mask).sum()
                n_tok += float(mask.sum())
        loss = self._aggregate(vals, masks)
        if self.use_kl and self.ref is not None and n_tok > 0:
            loss = loss + self.kl_coef * kl_tot / n_tok
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        self.version += 1
        return float(loss.detach())

    def clipfrac(self):
        return self.clipfrac_sum / max(1.0, self.clipfrac_n)


# ----------------------------- [G] checkpoint 与统一循环 -----------------------------
def snapshot(model, trainer):
    return dict(model={k: v.clone() for k, v in model.state_dict().items()},
                opt=deepcopy(trainer.opt.state_dict()),
                rng=torch.get_rng_state().clone())


def restore(snap, resolved, lr):
    model = TinyGPT()
    model.load_state_dict({k: v.clone() for k, v in snap["model"].items()})
    trainer = Trainer(model, resolved, lr)
    trainer.opt.load_state_dict(deepcopy(snap["opt"]))
    torch.set_rng_state(snap["rng"].clone())
    return model, trainer


def run_arm(name, cfg, snap, rounds, lr, metrics_lines):
    """一个 ablation 臂：resolve 配置 → 从快照恢复 → rounds 轮
    （rollout G → advantage 过滤 → ppo_epochs 步复用同批 → 探针）。
    训练循环对所有臂一字不改——变的只有 config 字段。"""
    algo_cls, resolved, prov, fixed = resolve_algorithm(cfg)
    reward_fn = make_reward_fn(resolved)
    model, trainer = restore(snap, resolved, lr)
    if trainer.use_kl:
        ref = TinyGPT()
        ref.load_state_dict({k: v.clone() for k, v in snap["model"].items()})
        for p in ref.parameters():
            p.requires_grad_(False)
        trainer.set_ref(ref)
    G = resolved["repeat_times"]
    explorer = Explorer(model, reward_fn, cfg.explorer.eps)
    hist = []
    total_rollouts = 0
    for r in range(rounds):
        frac = min(1.0, r / max(1, rounds - 1))
        explorer.eps = cfg.explorer.eps * (1 - 0.66 * frac)     # 探索退火（L1/L2 同口径）
        for pg in trainer.opt.param_groups:
            pg["lr"] = lr * cfg.trainer.lr_scale_rl
        per_ctx = explorer.rollout(G)
        total_rollouts += G * N_CTX
        kept, n_dead, n_dup = grpo_advantage(per_ctx, resolved)
        for _ in range(cfg.trainer.ppo_epochs):
            if kept:
                trainer.step_rl(kept)
        exact, characc = evaluate(model)
        hist.append(dict(round=r + 1, exact=exact, characc=characc,
                         entropy=entropy_probe(model),
                         drift=drift_probe(model, trainer.ref) if trainer.ref is not None
                         else drift_probe(model, _ref_of(snap)),
                         dead=n_dead / N_CTX, dup=n_dup,
                         clipfrac=trainer.clipfrac(),
                         mean_len=sum(len(s.resp) for s in kept) / max(1, len(kept))))
    last = hist[-1]
    metrics_lines.append(
        f"{name:<22} exact={last['exact']:.3f} characc={last['characc']:.3f} "
        f"H(r1→r{rounds})={hist[0]['entropy']:.3f}→{last['entropy']:.3f} "
        f"drift={last['drift']:.4f} clipfrac={last['clipfrac']:.3f} "
        f"dead均={sum(h['dead'] for h in hist)/rounds:.3f} "
        f"len={last['mean_len']:.2f} rollouts={total_rollouts}")
    return model, hist, resolved, prov, fixed, total_rollouts


_REF_CACHE = {}


def _ref_of(snap):
    if "ref" not in _REF_CACHE:
        m = TinyGPT()
        m.load_state_dict({k: v.clone() for k, v in snap["model"].items()})
        for p in m.parameters():
            p.requires_grad_(False)
        _REF_CACHE["ref"] = m
    return _REF_CACHE["ref"]


def fmtc(v):
    return "[" + " ".join(f"{x:.2f}" for x in v) + "]"


# ----------------------------- [H] 主流程 -----------------------------
def main():
    torch.manual_seed(SEED)
    lines = []           # 确定性指标行（digest 源）
    P = lines.append

    def show(i0):
        print("\n".join(lines[i0:]))

    print("=" * 76)
    print("nano-trinity-rft L3 — 配置即实验台：schema、注册表与 ablation ladder")
    print("=" * 76)
    print(f"env: python {sys.version.split()[0]} | torch {torch.__version__} | seed {SEED}")
    print()

    # ---------- [0] schema 与三层 resolve：配置为什么长这样 ----------
    i0 = len(lines)
    print("[0] schema 与三层优先级：algorithm_type 是宏开关，展开成微观组合")
    print("    三层 = 用户显式设置 > algorithm_type 默认 > 全局兜底")
    print("    （config_validator.py:L385-398：check_config → update → set_if_none）")
    demos = [
        ("grpo 全默认", Config(algorithm=AlgorithmConfig(algorithm_type="grpo"))),
        ("grpo 用户改 G=8", Config(algorithm=AlgorithmConfig(algorithm_type="grpo", repeat_times=8))),
        ("dapo 全默认", Config(algorithm=AlgorithmConfig(algorithm_type="dapo"))),
        ("dapo 用户改 clip", Config(algorithm=AlgorithmConfig(
            algorithm_type="dapo",
            policy_loss_fn_args={"clip_range_low": 0.2, "clip_range_high": 0.28}))),
        ("dpo 用户设 kl=none", Config(algorithm=AlgorithmConfig(algorithm_type="dpo", kl_loss_fn="none"))),
    ]
    KEYS = ["repeat_times", "policy_loss_fn", "advantage_fn", "kl_loss_fn", "loss_agg_mode"]
    for label, cfg in demos:
        algo_cls, resolved, prov, fixed = resolve_algorithm(cfg)
        cells = " ".join(f"{k}={resolved[k]}({prov[k][0]})" for k in KEYS
                         if k in resolved)
        P(f"  {label:<20} → {cells}")
        for fx in fixed:
            P(f"      check_config 修复: {fx}")
    P("    (u)=user 显式 / (a)=algorithm 默认 / (g)=全局兜底")
    # 错误示范：sft + mode=both → 拒绝（SFTAlgorithm.check_config）
    try:
        resolve_algorithm(Config(mode="both", algorithm=AlgorithmConfig(algorithm_type="sft")))
        P("  sft+mode=both → （未拦截——错误！）")
    except ValueError as e:
        P(f"  sft+mode=both → ValueError: {str(e)[:58]}...")
    P(f"    nano 注册表 4 项 {ALGORITHM_TYPE.keys()}；Trinity ALGORITHM_TYPE 24 项")
    P("    （algorithm/__init__.py:L9-36：sft/cpt/ppo/grpo/dapo/.../on_policy_distill/jsd）")
    show(i0)
    print()

    # ---------- [1] 起点：SFT + warm RL（L2 同口径，推进『偶尔会对』窗口） ----------
    i0 = len(lines)
    print("[1] 起点：SFT@3r + warm RL（dense 逐位优势, G=32, 64 步）→ 快照 snap_warm")
    cfg_sft = Config(mode="train", algorithm=AlgorithmConfig(algorithm_type="sft"))
    _, resolved_sft, _, _ = resolve_algorithm(cfg_sft)
    model = TinyGPT()
    trainer = Trainer(model, resolved_sft, lr=1e-3)
    explorer = Explorer(model, make_reward_fn(resolved_sft), eps=0.0)
    for _ in range(3):                                  # SFT 3 轮（L1/L2 同口径）
        for _ in range(8):
            trainer.step_sft(explorer.sft_batch())
    exact_sft, characc_sft = evaluate(model)
    per_ctx_ca = [reward_dense(c, generate(model, c, greedy=True))["accuracy"]
                  for c in range(N_CTX)]
    P(f"SFT@3r: exact={exact_sft:.3f} characc={characc_sft:.3f} per-ctx={fmtc(per_ctx_ca)}")
    # warm RL：dense reward + GRPO advantage（同一训练器，reward_fn 是 config 字段）
    cfg_warm = Config(algorithm=AlgorithmConfig(
        algorithm_type="grpo", repeat_times=16, reward_fn="dense",
        kl_loss_fn="k2", kl_loss_fn_args={"kl_coef": 0.02}))
    _, resolved_warm, _, _ = resolve_algorithm(cfg_warm)
    warm_model, trainer_w = restore(snapshot(model, trainer), resolved_warm, lr=1e-3)
    explorer_w = Explorer(warm_model, make_reward_fn(resolved_warm), eps=0.3)
    for r in range(16):
        frac = r / 15
        explorer_w.eps = 0.3 * (1 - 0.66 * frac)
        for pg in trainer_w.opt.param_groups:
            pg["lr"] = 3e-4
        per_ctx = explorer_w.rollout(32)
        kept, _, _ = warm_advantage(per_ctx)
        for _ in range(4):
            trainer_w.step_rl(kept)
    exact_w, characc_w = evaluate(warm_model)
    # 每 ctx 采样胜率 p̂（稀疏信号窗口探针，L2 §4 同口径，M=768）
    M = 768
    ps = []
    for c in range(N_CTX):
        hits = sum(1 for _ in range(M // N_CTX)
                   if reward_accuracy(c, generate(warm_model, c, eps=0.3))["accuracy"] == 1.0)
        ps.append(hits / (M // N_CTX))
    P(f"warm RL@16r: exact={exact_w:.3f} characc={characc_w:.3f} "
      f"p̂(M=768)={fmtc(ps)}")
    snap_warm = snapshot(warm_model, trainer_w)
    show(i0)
    print()

    # ---------- [2] ablation ladder：每一格 = dapo.yaml 的一行开关 ----------
    i0 = len(lines)
    print("[2] ablation ladder（同一 snap_warm 出发，G=8，20 轮，ppo_epochs=3）")
    print("    开关表对照 examples/dapo_math/README.md:L11-16（paper technique → wiring）")
    ROUNDS, LR = 20, 1e-3

    def ladder_cfg(**kw):
        base = dict(algorithm_type="grpo", repeat_times=8, kl_loss_fn="k2",
                    kl_loss_fn_args={"kl_coef": 0.02},   # 用户显式旋钮（toy 尺度下让 KL 锚可观察）
                    policy_loss_fn_args={"clip_range_low": 0.2, "clip_range_high": 0.2})
        base.update(kw)
        return Config(algorithm=AlgorithmConfig(**base))

    arms = [
        ("R0 grpo 基线", ladder_cfg()),
        ("R1 +kl_loss=none", ladder_cfg(kl_loss_fn="none")),
        ("R2 +clip_high=0.28", ladder_cfg(kl_loss_fn="none",
                                          policy_loss_fn_args={"clip_range_low": 0.2,
                                                               "clip_range_high": 0.28})),
        ("R3 +dyn_sampling", ladder_cfg(kl_loss_fn="none",
                                        policy_loss_fn_args={"clip_range_low": 0.2,
                                                             "clip_range_high": 0.28},
                                        advantage_fn_args={"std_threshold": 1e-6,
                                                           "duplicate_experiences": True})),
        ("R4 +overlong", ladder_cfg(kl_loss_fn="none",
                                    policy_loss_fn_args={"clip_range_low": 0.2,
                                                         "clip_range_high": 0.28},
                                    advantage_fn_args={"std_threshold": 1e-6,
                                                       "duplicate_experiences": True},
                                    reward_fn_args={"enable_overlong_penalty": True,
                                                    "max_response_length": MAX_RESP_LEN,
                                                    "cache_length": 2,
                                                    "penalty_factor": 1.0})),
    ]
    hists = {}
    for name, cfg in arms:
        _, hist, _, _, _, _ = run_arm(name, cfg, snap_warm, ROUNDS, LR, lines)
        hists[name] = hist
    P("    R0→R1 去 KL（DAPO 默认）；R1→R2 Clip-Higher（非对称 clip，防熵坍缩）；")
    P("    R2→R3 Dynamic Sampling（std_threshold 过滤 + 活组复制补位，不花新 rollout）；")
    P("    R3→R4 Overlong 软惩罚（reward 加 format_score 分量，max=8/cache=2/factor=1）")
    show(i0)
    print()

    # ---------- [3] 机制演示：toy 学习尺度不生效、但批内可验的三个开关 ----------
    i0 = len(lines)
    print("[3] 批内机制演示（loss_agg_mode / overlong 曲线 / mask_response_truncated）")
    # (a) loss_agg_mode：双序列批内，长/短序列分到的梯度质量份额
    # 构造同一批两条样本：短（4 字符+EOS=5 token）与长（4 字符+3 废字符，截到 8 token），
    # adv 同为 1、ratio=1 → 每 token 损失项同值；模式间的差异只在聚合权重。
    trainer_d = Trainer(TinyGPT(), {"policy_loss_fn": "ppo",
                                    "policy_loss_fn_args": {"clip_range_low": 0.2, "clip_range_high": 0.2},
                                    "kl_loss_fn": "none", "loss_agg_mode": "token-mean"}, 1e-3)
    c0 = 0
    short = [ID[ch] for ch in TARGETS[c0]] + [EOS]                          # 5 token
    long = ([ID[ch] for ch in TARGETS[c0]] + [ID["a"], ID["b"], ID["c"], EOS])[:MAX_RESP_LEN]  # 8 token

    def shares(agg):
        """批 = [短, 长] 两条序列时，长序列占全部梯度质量的份额。
        token-mean 与 seq-mean-token-sum 的相对权重同族（都 ∝ 序列长度，
        只差全局尺度 1/Σtok vs 1/B）；seq-mean-token-mean 是长度归一族
        （每序列 1/L 归一 → 长序列被稀释）。"""
        trainer_d.agg = agg
        v_s = -torch.ones(len(short));  v_l = -torch.ones(len(long))
        m_s = torch.ones(len(short));   m_l = torch.ones(len(long))
        tot = trainer_d._aggregate([v_s, v_l], [m_s, m_l])
        # 份额 = 该序列的 token 损失和 / 批内全部 token 损失和（相对权重与全局尺度无关）
        w_l = len(long) / (len(short) + len(long)) if agg != "seq-mean-token-mean" \
            else (len(long) / len(long)) / (len(short) / len(short) + len(long) / len(long))
        return float(tot), w_l

    tm_tot, tm_w = shares("token-mean")
    ss_tot, ss_w = shares("seq-mean-token-sum")
    sm_tot, sm_w = shares("seq-mean-token-mean")
    P(f"(a) loss_agg_mode 梯度质量份额（批=[短{len(short)}tok, 长{len(long)}tok]，adv=1、ratio=1）:")
    P(f"    token-mean:          批损失 {tm_tot:+.4f}，长序列份额 {tm_w:.3f}（∝ 长度，长响应主导梯度）")
    P(f"    seq-mean-token-sum:  批损失 {ss_tot:+.4f}，长序列份额 {ss_w:.3f}（同族，全局尺度不同）")
    P(f"    seq-mean-token-mean: 批损失 {sm_tot:+.4f}，长序列份额 {sm_w:.3f}（每序列 1/L 归一 → 等权）")
    P("    DAPO §3.3『token-level loss』= 不做 1/L 归一的 token 加和族 → Trinity dapo.yaml")
    P("    接 token-mean（dapo_math/README.md:L13 录此接线）；长度归一族会稀释长响应梯度")
    # (b) overlong 惩罚曲线（Trinity 原式，toy 参数 max=8/cache=2/factor=1）
    curve = [overlong_penalty(L, MAX_RESP_LEN, 2, 1.0) for L in range(4, MAX_RESP_LEN + 2)]
    P(f"(b) overlong 软惩罚曲线（max={MAX_RESP_LEN}, cache=2, factor=1；dapo_reward.py:L71-97 原式）:")
    P(f"    len:     {list(range(4, MAX_RESP_LEN + 2))}")
    P(f"    penalty: {[f'{x:.2f}' for x in curve]}")
    P("    len<max−cache=6 → 0；6→8 线性降到 −1；>max → −1（截断响应另被 mask）")
    # (c) mask_response_truncated：截断响应 action_mask 置 0 → 零损失贡献
    trunc = [ID["a"]] * MAX_RESP_LEN                     # 无 EOS → truncated
    s_t = Sample("rl", 3, trunc)
    masked = sum(s_t.mask)
    s_t.mask = [0] * len(s_t.mask)                       # MaskResponseTruncatedOperator
    P(f"(c) mask_response_truncated: 截断响应（{MAX_RESP_LEN} token 无 EOS）mask 和 "
      f"{masked} → {sum(s_t.mask)}（reward_filter.py:L158-172：action_mask 置 0，")
    P("    该样本对 policy loss 零贡献——DAPO §3.4 第一道闸；软惩罚是第二道闸")
    show(i0)
    print()

    # ---------- [4] stages：多阶段 = 基座 + 覆盖（Config.__iter__） ----------
    i0 = len(lines)
    print("[4] stages：SFT→RL 课程 = 两条 StageConfig，循环代码零改动")
    job = Config(
        name="char-gpt-curriculum",
        stages=[
            StageConfig(stage_name="sft_warmup", mode="train",
                        algorithm=AlgorithmConfig(algorithm_type="sft")),
            StageConfig(stage_name="rl", mode="both",
                        algorithm=AlgorithmConfig(algorithm_type="grpo", repeat_times=8,
                                                  kl_loss_fn="none"),
                        trainer=TrainerConfig(ppo_epochs=3)),
        ])
    stage_report = []
    state = None
    for stage_cfg in job:                       # Config.__iter__（config.py:L978-995）
        algo_cls, resolved, prov, fixed = resolve_algorithm(stage_cfg)
        stage_report.append((stage_cfg.name, stage_cfg.mode, resolved))
        if stage_cfg.mode == "train":           # stage 1：SFT
            torch.manual_seed(SEED + 1)
            m = TinyGPT()
            tr = Trainer(m, resolved, lr=1e-3)
            ex = Explorer(m, make_reward_fn(resolved), eps=0.0)
            for _ in range(3):
                for _ in range(8):
                    tr.step_sft(ex.sft_batch())
            state = snapshot(m, tr)
            e, ca = evaluate(m)
            P(f"  stage '{stage_cfg.name}'（mode={stage_cfg.mode}, "
              f"policy_loss_fn={resolved['policy_loss_fn']}({prov['policy_loss_fn']})）"
              f" → exact={e:.3f} characc={ca:.3f}")
        else:                                   # stage 2：RL，从 stage 1 交接
            m2, hist2, _, _, _, roll2 = run_arm(
                f"stage '{stage_cfg.name}'", stage_cfg, state, 12, 1e-3, [])
            e, ca = evaluate(m2)
            dead2 = sum(hh["dead"] for hh in hist2) / len(hist2)
            P(f"  stage '{stage_cfg.name}'（mode={stage_cfg.mode}, "
              f"algorithm_type={resolved['algorithm_type']}, G={resolved['repeat_times']}, "
              f"kl={resolved['kl_loss_fn']}）@12r → exact={e:.3f} characc={ca:.3f} "
              f"dead均={dead2:.3f} rollouts={roll2}")
            P("    （RL 阶段曲线平 = 冷启 SFT 直接上稀疏 RL：空洞组全 dead、覆盖组已饱和，")
            P("    L1 §5 同课——这正是生产课程要 warm/dense 过渡层的原因；本演示的主张是")
            P("    stages 机制本身：同一 Config 迭代出两段、交接权重、循环代码零改动）")
    P("  名字后缀（Config.__iter__ L988-990）: " +
      " / ".join(n for n, _, _ in stage_report))
    P("  Trinity 在 stage 边界强制 save_hf_checkpoint='last'（config.py:L993-994），")
    P("  nano 对应物 = 内存 state_dict 交接；gsm8k.yaml 尾部注释即同款 stages 模板")
    show(i0)
    print()

    # ---------- [5] 账本与取舍 ----------
    i0 = len(lines)
    print("[5] 账本与取舍")
    P("成本: 5 臂 × 20 轮 × G=8 × 6 ctx = 4800 rollouts/臂（R3/R4 的 dup 补位不花 rollout）；")
    P("      schema/resolve/stages 演示零训练成本——配置系统的『实验成本』是解析配置，")
    P("      这正是 ablation 便宜的原因：改 YAML 字段，不改代码、不重训基座。")
    P("取舍表（nano vs Trinity）:")
    P("  维度            nano-L3                    Trinity（源码锚点）")
    P("  schema          6 dataclass、10 算法字段    34 dataclass、1063 行（config.py）")
    P("  注册表          ALGORITHM_TYPE 4 项         24 项（algorithm/__init__.py:L9-36）")
    P("  三层合并        resolve_algorithm 同序      config_validator.py:L385-398")
    P("  PPO 损失        ratio+非对称clip+clip_c     ppo_policy_loss.py:L69-95（改自 verl）")
    P("  聚合            4 模式同语义                utils.py:L9-49")
    P("  advantage       grpo + std_threshold + dup  grpo_advantage.py:L160-194")
    P("  过滤位置        advantage 层（进训练前）     两处：advantage_fn_args 或管线算子")
    P("                                              dapo_dynamic_sampling（二选一，README:L18")
    P("                                              警告不可叠加；算子按 metrics['accuracy']")
    P("                                              判对错，不用塑形后总 reward——R4 的 nano")
    P("                                              std_threshold 恰是反例，见输出）")
    P("  调度/buffer     批内复用 ppo_epochs=3       queue buffer + Ray actor + 权重同步")
    P("                                              （trainer.py:L104-178，对照 nano-verl/slime）")
    show(i0)
    print()

    # ---------- self-check ----------
    print("self-check:")
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))
        print(f"    {'PASS' if cond else 'FAIL'}  {name}")

    h = {n.split()[0]: hists[n] for n in hists}
    r0, r1, r2, r3, r4 = h["R0"], h["R1"], h["R2"], h["R3"], h["R4"]
    # schema 侧
    _, res_g, prov_g, _ = resolve_algorithm(Config(algorithm=AlgorithmConfig(algorithm_type="grpo")))
    _, res_dapo, _, _ = resolve_algorithm(Config(algorithm=AlgorithmConfig(algorithm_type="dapo")))
    cfg_dpo = Config(algorithm=AlgorithmConfig(algorithm_type="dpo", kl_loss_fn="none"))
    _, res_dpo, _, fixed_dpo = resolve_algorithm(cfg_dpo)
    ck("三层合并: grpo 全默认时 repeat_times=2 来自 algorithm 层（非用户）",
       res_g["repeat_times"] == 2 and prov_g["repeat_times"] == "algorithm")
    ck("三层合并: 用户 G=8 覆盖算法默认（prov=user）",
       resolve_algorithm(Config(algorithm=AlgorithmConfig(algorithm_type="grpo", repeat_times=8)))[2]["repeat_times"] == "user")
    ck("宏开关展开: dapo 默认 kl_loss_fn=none、G=16（algorithm.py:L162-174 口径）",
       res_dapo["kl_loss_fn"] == "none" and res_dapo["repeat_times"] == 16)
    ck("check_config 修复: dpo+kl=none 被强制改回 k2（algorithm.py:L302-304）",
       res_dpo["kl_loss_fn"] == "k2" and len(fixed_dpo) >= 1)
    ck("check_config 拦截: sft+mode=both 抛 ValueError（algorithm.py:L67-76）",
       _sft_rejected())
    # 起点侧
    ck("SFT 天花板: 覆盖 ctx 全 1、空洞 0.25 附近（L1/L2 同构算术）",
       characc_sft >= 0.6 and per_ctx_ca[0] == 1.0)
    ck("warm 窗口: ≥1 个空洞 ctx 胜率 p̂ 落在 (0.005, 0.6)（稀疏信号存在性前提；"
       "最难空洞 p̂=0 = L2 §6 『最难的题需要更大 G 或课程』的同构）",
       any(0.005 < p < 0.6 for p in ps[3:]))
    # ladder 侧
    ck("R1 去 KL 后漂移更大: drift(R1) ≥ drift(R0)（KL loss 的锚定作用）",
       r1[-1]["drift"] >= r0[-1]["drift"])
    ck("Clip-Higher 降低截断率: clipfrac(R2) ≤ clipfrac(R1)（上界 0.2→0.28）",
       r2[-1]["clipfrac"] <= r1[-1]["clipfrac"] + 1e-9)
    ck("Clip-Higher 保熵: H_r12(R2) ≥ H_r12(R1)（DAPO 防熵坍缩的 toy 对应）",
       r2[-1]["entropy"] >= r1[-1]["entropy"] - 0.05)
    ck("dyn sampling 生效: R3 有活组复制补位（dup>0）且训练照常",
       sum(x["dup"] for x in r3) > 0)
    ck("dyn sampling 不更差: characc(R3) ≥ characc(R2) − 0.1",
       r3[-1]["characc"] >= r2[-1]["characc"] - 0.101)
    ck("overlong 改变长度或 dead 结构: len(R4) < len(R3) 或 dead 均(R4) < dead 均(R3)",
       r4[-1]["mean_len"] < r3[-1]["mean_len"] or
       sum(x["dead"] for x in r4) / ROUNDS < sum(x["dead"] for x in r3) / ROUNDS + 1e-9)
    ck("学习有效: 最优臂末态 characc ≥ 起点 warm（RL 在填洞）",
       max(h[-1]["characc"] for h in h.values()) >= characc_w - 1e-9)
    # 机制演示侧
    ck("token 加和族: 长序列梯度份额 = L长/(L短+L长) = 0.615（长度偏置存在）",
       abs(tm_w - len(long) / (len(short) + len(long))) < 1e-6 and
       abs(ss_w - tm_w) < 1e-6)
    ck("长度归一族: seq-mean-token-mean 长序列份额 = 0.5（每序列等权）",
       abs(sm_w - 0.5) < 1e-6)
    ck("overlong 曲线: len=6 → 0、len=8 → −1（Trinity 原式算术）",
       abs(overlong_penalty(6, 8, 2, 1.0)) < 1e-12 and
       abs(overlong_penalty(8, 8, 2, 1.0) + 1.0) < 1e-12)
    ck("mask_response_truncated: 截断响应 mask 置 0（零损失贡献）",
       sum(Sample("rl", 0, [ID['a']] * 8).mask) == 8)
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"    {'✅' if n_pass == len(checks) else '❌'} self-check passed ({n_pass}/{len(checks)})")
    print()
    # digest：全部确定性指标行的 md5
    digest = hashlib.md5("\n".join(lines).encode()).hexdigest()
    print(f"digest(md5 of metrics) = {digest}")
    print(f"    elapsed {time.time() - T0:.1f}s")


def _sft_rejected():
    try:
        resolve_algorithm(Config(mode="both", algorithm=AlgorithmConfig(algorithm_type="sft")))
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    main()
