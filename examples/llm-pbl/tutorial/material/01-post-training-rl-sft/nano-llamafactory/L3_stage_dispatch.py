#!/usr/bin/env python3
"""
L3_stage_dispatch.py — nano-llamafactory L3

对照 LLaMA-Factory 配置体系：**一个配置切换 SFT/DPO/KTO 的抽象取舍**。
L0-L2 分别讲了数据侧三件套（template/labels mask/collator）、真实 SFT 循环、
DPO 偏好对。本级回答框架级问题：为什么 LLaMA-Factory 用**一个 `stage` 配置字段**
就能切换训练方法，而不是一套方法一个代码库？拆到底，dispatch 只有三层：

    [配置层]  FinetuningArguments.stage / pref_loss / ...（一个扁平 dataclass）
              -> 推导标志（use_ref_model）+ 边界校验（__post_init__ fail-loud）
    [分发层]  tuner._training_function 按 stage 选 workflow（if/elif 表）
              workflow 再按「数据形状」选 processor/collator——
              注意 dpo 的数据走 stage="rm" 的 pairwise 管线（dpo/workflow.py:L45）：
              数据层的分发键是**数据形状**，不是方法名
    [执行层]  trainer 内部按 pref_loss 字符串选 loss 族（sigmoid/ipo/orpo/simpo），
              同一个字符串同时决定三件事：loss 公式 / 用不用 ref / 求和还是平均

本文件 = 可运行的「nano-factory」：配置 dataclass + 分发表 + 三个真 trainer
（sft/dpo/kto，全部真实 torch 梯度下降，CPU，固定 seed）。两个机器证明：
    A. 跨级锚（抽象不变性）：经配置分发跑出的 sft 与 dpo-sigmoid 结果与
       L2_dpo_preference_pairs.py 手写路径**逐位相同**——
       好的分发层在数值上是惰性的：它改变代码组织，不改变数字。
    B. 一个字符串三个行为：pref_loss 扫描证明同一配置字段切换
       loss 公式 / ref 使用 / 聚合方式，且 orpo/simpo 全程零次触碰 ref。

与权威实现的对应（LlamaFactory main @f28afaf6，2026-08-16 codeload tarball 抓取；
HEAD f28afaf6355af515454dfb16c97d728307c93897。上游 2026 年改名：
github.com/hiyouga/LLaMA-Factory -> github.com/hiyouga/LlamaFactory，
包名仍为 llamafactory；L2 快照 0bbe481e 录值以 2026-08-13 抓取日为准）：
    hparams/finetuning_args.py:L460  stage: Literal["pt","sft","rm","ppo","dpo","kto"]
    hparams/finetuning_args.py:L183  pref_loss: Literal[六项]，默认 sigmoid
    hparams/finetuning_args.py:L593  use_ref_model = stage=="dpo" and pref_loss not in ["orpo","simpo"]
    hparams/finetuning_args.py:L609  dpo_label_smoothing 仅 sigmoid 族校验
    train/tuner.py:L138-151          stage if/elif 分发链（else: Unknown task）
    train/dpo/workflow.py:L45        get_dataset(..., stage="rm")——dpo 复用 pairwise 管线
    data/loader.py:L189-226          _get_dataset_processor（数据层五阶段，无 dpo）
    data/processor/feedback.py:L87   KTO 的 KL 基线 = 批内循环移位响应
    train/dpo/trainer.py:L80/L195-208/L234-235  loss_type 装配 / 无 ref 分支 / 长度归一
    train/dpo/trainer.py:L150-166    orpo/simpo 原生实现
    trl v0.24.0 dpo_trainer.py:L1110-1137  sigmoid(cDPO)/hinge/ipo 公式
    trl v0.24.0 kto_trainer.py:L1150-1191  KTO loss（kl 批均值 clamp + 双侧 1-sigmoid）
    （trl 依赖区间 trl>=0.18.0,<=0.24.0：LlamaFactory pyproject.toml:L47）

论文锚点（arXiv，2026-08-16 export.arxiv.org API live 核验标题逐位吻合）：
    DPO   2305.18290    IPO 2310.12036    ORPO 2403.07691
    SimPO 2405.14734    KTO 2402.01306

依赖：torch（CPU 即可）。输出确定性：固定 seed + 全批训练，除 elapsed 行外逐字节确定。
"""

import copy
import hashlib
import math
import random
import re
import time

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise SystemExit(
        "[error] torch is required to run this L3 script.\n"
        "        Install it with: pip install torch\n"
        "        (CPU is enough; no transformers needed.)"
    ) from e


SEED = 42
PAD = "<pad>"
SYS, USR, ASST, EOT = "<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"
IGNORE_INDEX = -100  # LlamaFactory extras/constants.py:L50（L2 2026-08-13 复验零漂移）

TOKEN_RE = re.compile(r"<\|[\w]+\|>|\n|\S+")

# 与 L1/L2 同构的超参数（74,496 参数 TinyLM）——跨级锚要求模型/优化器逐位同款
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
MAX_LEN = 64
LR = 5e-3
SFT_EPOCHS = 300
DPO_STEPS = 200
SWEEP_STEPS = 150
KTO_STEPS = 200
BETA = 0.1       # = LlamaFactory pref_beta 默认值（finetuning_args.py:L171-173）
SIMPO_GAMMA = 0.5  # = LlamaFactory simpo_gamma 默认值（finetuning_args.py:L199-201）


# ======================================================================
# 数据侧：与 L0/L1/L2 完全相同的 template + labels 遮罩机制
# （本级的论点正是：这一层在 sft/dpo/kto 三个 stage 之间**零改动**复用）
# ======================================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text):
    return TOKEN_RE.findall(text)


def apply_chat_template(system, user, assistant=None):
    """与 L1/L2 完全相同的 ChatML 风格 toy 模板。"""
    text = f"{SYS}\n{system}\n{EOT}\n{USR}\n{user}\n{EOT}\n{ASST}\n"
    if assistant is not None:
        text += f"{assistant}\n{EOT}"
    return text


def build_vocab(texts):
    specials = [PAD, SYS, USR, ASST, EOT]
    words = sorted({t for tx in texts for t in tokenize(tx)
                    if t not in specials and t != "\n"})
    return specials + words + ["\n"]


def make_row(vocab, system, question, answer):
    """渲染一条 (prompt+answer) 的未 pad 行：(input_ids, attention_mask, labels)。
    与 L1/L2 的构造逐位同构（prompt 全 -100，response 含 eot 进 loss）。"""
    tok2id = {t: i for i, t in enumerate(vocab)}
    full = apply_chat_template(system, question, answer)
    prompt = apply_chat_template(system, question)
    ids = [tok2id[t] for t in tokenize(full)]
    prompt_ids = [tok2id[t] for t in tokenize(prompt)]
    assert ids[:len(prompt_ids)] == prompt_ids, "推理 prompt 必须是训练串的真前缀"
    labels = [IGNORE_INDEX] * len(prompt_ids) + ids[len(prompt_ids):]
    am = [1] * len(ids)
    return ids, am, labels


def collate_rows(rows, vocab):
    """与 L1/L2 相同的右 pad 双层遮罩 collator。rows = [(ids, am, labels), ...]"""
    pid = vocab.index(PAD)
    L = max(len(ids) for ids, _, _ in rows)
    batch_ids, batch_am, batch_labels = [], [], []
    for ids, am, labels in rows:
        n = L - len(ids)
        batch_ids.append(ids + [pid] * n)
        batch_am.append(am + [0] * n)
        batch_labels.append(labels + [IGNORE_INDEX] * n)
    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(batch_am, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )


class TinyLM(nn.Module):
    """与 L1/L2 完全相同的极小 causal LM（74,496 参数）。"""

    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 max_len=MAX_LEN):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        key_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(
            torch.ones((L, L), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        h = self.blocks(h, mask=causal_mask, src_key_padding_mask=key_mask)
        return self.head(self.norm(h))


def seq_logps(model, input_ids, attention_mask, labels):
    """每条序列的 label 对数概率之和（shifted gather + -100 遮罩）。
    对照 LlamaFactory get_batch_logps（train/trainer_utils.py；L2 快照录
    L592，2026-08-13 锚）。定义与 L2 的 seq_logps 逐位相同。"""
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != IGNORE_INDEX).float()
    safe = shift_labels.clamp(min=0)
    per_tok = torch.gather(
        shift_logits.log_softmax(-1), dim=2, index=safe.unsqueeze(2)
    ).squeeze(2)
    return (per_tok * mask).sum(-1)


# ======================================================================
# [配置层] NanoFactoryConfig —— 对照 hparams/finetuning_args.py
# 要点一：方法族字段（pref_*）与 stage 同住一个扁平 dataclass；
# 要点二：__post_init__ 做两件 LF 同款事——推导标志 + 边界 fail-loud。
# ======================================================================

STAGES = ("sft", "dpo", "kto")  # nano 实现的 stage；pt/rm/ppo 只在分发表演示
PREF_LOSSES = ("sigmoid", "ipo", "orpo", "simpo")  # LF 六族中的四族（见 tutorial §7）


class ConfigError(ValueError):
    pass


class NanoFactoryConfig:
    """nano 版 FinetuningArguments 切片：只保留与 dispatch 相关的字段。

    对照 LlamaFactory hparams/finetuning_args.py（@f28afaf6，2026-08-16）：
        L460  stage: Literal["pt","sft","rm","ppo","dpo","kto"]，默认 "sft"
        L183  pref_loss: Literal["sigmoid","hinge","ipo","kto_pair","orpo","simpo"]
        L171  pref_beta 默认 0.1；L175 pref_ftx 默认 0.0
        L187  dpo_label_smoothing 默认 0.0；L199 simpo_gamma 默认 0.5
        L191/L195 kto_chosen_weight / kto_rejected_weight 默认 1.0
    """

    def __init__(self, stage="sft", pref_loss="sigmoid", pref_beta=BETA,
                 pref_ftx=0.0, dpo_label_smoothing=0.0,
                 kto_chosen_weight=1.0, kto_rejected_weight=1.0,
                 simpo_gamma=SIMPO_GAMMA):
        self.stage = stage
        self.pref_loss = pref_loss
        self.pref_beta = pref_beta
        self.pref_ftx = pref_ftx
        self.dpo_label_smoothing = dpo_label_smoothing
        self.kto_chosen_weight = kto_chosen_weight
        self.kto_rejected_weight = kto_rejected_weight
        self.simpo_gamma = simpo_gamma
        self.warnings = []
        self._post_init()

    def _post_init(self):
        # ---- 推导标志：配置自己算出自己的后果 ----
        # 逐字镜像 finetuning_args.py:L593：
        #   self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
        self.use_ref_model = (
            self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
        )
        # ---- 边界校验：fail-loud 在配置解析时，不在训练中途 ----
        # 镜像 finetuning_args.py:L609-610（消息逐字同款）：
        if (self.stage == "dpo" and self.pref_loss != "sigmoid"
                and self.dpo_label_smoothing > 1e-6):
            raise ConfigError(
                "`dpo_label_smoothing` is only valid for sigmoid loss function."
            )
        # 镜像 tuner.py:L150-151 的 else 分支（消息逐字同款）：
        if self.stage not in STAGES:
            raise ConfigError(f"Unknown task: {self.stage}.")
        # ---- nano 增补（LF 没有）：静默无效字段大声化，见 tutorial §6 ----
        if self.pref_ftx > 1e-6 and self.stage != "dpo":
            self.warnings.append(
                f"pref_ftx={self.pref_ftx} has no effect at stage={self.stage} "
                f"(only the dpo trainer reads it)"
            )


# ======================================================================
# [分发层] 表，不是分支
# 对照：tuner.py:L138-151（stage -> workflow 的 if/elif 链）
#       data/loader.py:L189-226（数据层五阶段 -> processor 类）
#       train/dpo/workflow.py:L45（dpo 的数据走 stage="rm"）
# ======================================================================

# 训练 stage -> 数据 stage。注意 dpo -> "rm"：数据层没有 "dpo" 这个键
# （loader.py:L169 的 Literal 是 ["pt","sft","rm","ppo","kto"]），
# 因为数据层按**数据形状**分发：偏好对就是 rm 的形状。
STAGE2DATA_STAGE = {"sft": "sft", "dpo": "rm", "kto": "kto"}


def supervised_process(vocab, examples):
    """sft 数据处理器：examples = [(system, question, answer)] -> rows。
    对照 data/processor/supervised.py（`<bos> X Y <eos>` / `<ignore>... Y <eos>`
    约定，L109-110 注释逐字）。"""
    return [make_row(vocab, s, q, a) for (s, q, a) in examples]


def pairwise_process(vocab, examples):
    """rm/dpo 数据处理器：examples = [(system, question, chosen, rejected)]
    -> chosen rows + rejected rows。对照 data/processor/pairwise.py:L66
    （chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids）。"""
    rows = [make_row(vocab, s, q, c) for (s, q, c, _r) in examples]
    rows += [make_row(vocab, s, q, r) for (s, q, _c, r) in examples]
    return rows


def feedback_process(vocab, examples):
    """kto 数据处理器：examples = [(system, question, answer, tag)]
    -> rows + kl_rows + tags。对照 data/processor/feedback.py：
    每条样本额外携带一条 KL 基线样本 = **批内循环移位的响应**（L87：
    kl_response = [examples["_response"][-1]] + examples["_response"][:-1]）。"""
    rows = [make_row(vocab, s, q, a) for (s, q, a, _t) in examples]
    shifted = [examples[-1]] + examples[:-1]  # 循环移位：第 i 条的 KL 响应 = 第 i-1 条
    kl_rows = [make_row(vocab, s, q, a) for (s, q, a, _t) in shifted]
    tags = torch.tensor([1 if t == "desirable" else 0
                         for (_s, _q, _a, t) in examples], dtype=torch.long)
    return rows, kl_rows, tags


# 数据 stage -> processor（对照 loader.py:L198-224 的 if/elif 选择）
DATA_STAGE2PROCESSOR = {
    "sft": supervised_process,
    "rm": pairwise_process,   # rm 与 dpo 共用：形状相同
    "kto": feedback_process,
}


def resolve_processor(stage):
    return DATA_STAGE2PROCESSOR[STAGE2DATA_STAGE[stage]]


# ======================================================================
# [执行层] loss 族注册表 —— 对照 train/dpo/trainer.py 的 loss_type 装配
# 一个 pref_loss 字符串同时决定三件事：
#   needs_ref  —— 镜像 finetuning_args.py:L593 + trainer.py:L195-208 的分支
#   aggregate  —— 镜像 trainer.py:L234-235（ipo/orpo/simpo 除以 valid_length）
#   loss 公式  —— sigmoid/ipo 镜像 trl v0.24.0 dpo_trainer.py:L1110-1137；
#                 orpo/simpo 镜像 LlamaFactory 原生实现 trainer.py:L150-166
# ======================================================================

def _sigmoid_loss(pol_c, pol_r, ref_c, ref_r, cfg):
    """DPO（arXiv 2305.18290 Eq. 7；trl dpo_trainer.py:L1110-1114，
    label_smoothing=0 时即 -logsigmoid(beta * logits)）。"""
    logits = cfg.pref_beta * ((pol_c - pol_r) - (ref_c - ref_r))
    return -nn.functional.logsigmoid(logits).mean()


def _ipo_loss(pol_c, pol_r, ref_c, ref_r, cfg):
    """IPO（arXiv 2310.12036；trl dpo_trainer.py:L1135-1137：
    (logits - 1/(2*beta))**2；logps 为长度归一后的均值 logp）。"""
    logits = (pol_c - pol_r) - (ref_c - ref_r)
    return ((logits - 1.0 / (2.0 * cfg.pref_beta)) ** 2).mean()


def _orpo_loss(pol_c, pol_r, _ref_c, _ref_r, cfg):
    """ORPO（arXiv 2403.07691；LlamaFactory 原生实现 trainer.py:L150-158：
    sft_loss + beta * odds_ratio_loss，无 ref）。"""
    log_odds = (pol_c - pol_r) - (
        torch.log1p(-torch.exp(pol_c)) - torch.log1p(-torch.exp(pol_r))
    )
    sft_loss = -pol_c
    odds_ratio_loss = -nn.functional.logsigmoid(log_odds)
    return (sft_loss + cfg.pref_beta * odds_ratio_loss).mean()


def _simpo_loss(pol_c, pol_r, _ref_c, _ref_r, cfg):
    """SimPO（arXiv 2405.14734；LlamaFactory 原生实现 trainer.py:L160-166：
    -logsigmoid(beta * (pi_logratios - gamma/beta))，无 ref，长度归一）。"""
    pi_logratios = pol_c - pol_r
    logits = pi_logratios - cfg.simpo_gamma / cfg.pref_beta
    return -nn.functional.logsigmoid(cfg.pref_beta * logits).mean()


PREF_LOSS_REGISTRY = {
    "sigmoid": dict(needs_ref=True, aggregate="sum", fn=_sigmoid_loss,
                    paper="2305.18290"),
    "ipo": dict(needs_ref=True, aggregate="avg", fn=_ipo_loss,
                paper="2310.12036"),
    "orpo": dict(needs_ref=False, aggregate="avg", fn=_orpo_loss,
                 paper="2403.07691"),
    "simpo": dict(needs_ref=False, aggregate="avg", fn=_simpo_loss,
                  paper="2405.14734"),
}


# ======================================================================
# 三个 trainer（真实 torch 训练循环）
# ======================================================================

class SFTTrainer:
    """stage=sft。与 L2 的 train_sft 数值路径逐位相同（跨级锚的前提）。"""

    def __init__(self, vocab, cfg):
        self.vocab = vocab
        self.cfg = cfg

    def fit(self, rows, epochs=SFT_EPOCHS):
        input_ids, attn_mask, labels = collate_rows(rows, self.vocab)
        model = TinyLM(len(self.vocab))
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        losses = []
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            logits = model(input_ids, attention_mask=attn_mask)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return model, losses


class DPOTrainer:
    """stage=dpo。对照 LlamaFactory CustomDPOTrainer 的装配：
    loss_type = finetuning_args.pref_loss（trainer.py:L80）->
    concatenated_forward（L219-252：2n 行一次前向、ipo/orpo/simpo 长度归一、
    split 出 chosen/rejected）-> compute_preference_loss（L187-216：
    use_ref_model 分支）-> 可选 pref_ftx 叠加（L300-302）。
    sigmoid 路径与 L2 的 train_dpo 数值路径逐位相同（跨级锚的前提）。"""

    def __init__(self, vocab, cfg):
        assert cfg.stage == "dpo"
        assert cfg.pref_loss in PREF_LOSS_REGISTRY, cfg.pref_loss
        self.vocab = vocab
        self.cfg = cfg
        self.family = PREF_LOSS_REGISTRY[cfg.pref_loss]
        self.n_ref_forwards = 0  # 机器记录：ref 到底被前向了几次

    def _forward_logps(self, model, input_ids, attn_mask, labels):
        """带梯度的 sum-logps + valid_length（对照 concatenated_forward 的
        get_batch_logps 调用，L231-233）。"""
        logits = model(input_ids, attention_mask=attn_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = (shift_labels != IGNORE_INDEX).float()
        safe = shift_labels.clamp(min=0)
        per_tok = torch.gather(
            shift_logits.log_softmax(-1), dim=2, index=safe.unsqueeze(2)
        ).squeeze(2)
        return (per_tok * mask).sum(-1), mask.sum(-1)

    def fit(self, pairs, ref_model, steps=DPO_STEPS, log_every=None):
        rows = pairwise_process(self.vocab, pairs)
        input_ids, attn_mask, labels = collate_rows(rows, self.vocab)
        n = len(pairs)
        agg = self.family["aggregate"]

        ref = copy.deepcopy(ref_model)
        ref.eval()
        if self.family["needs_ref"]:
            # ref 冻结 => 预算一次（与 L2 同款不变量；LF 每步走 ref 前向，
            # 见 tutorial §7 取舍表）
            ref_sum = seq_logps(ref, input_ids, attn_mask, labels)
            self.n_ref_forwards = 1
        else:
            ref_sum = None  # orpo/simpo：配置推导出不需要 ref，就一次都不跑

        def normalize(logps_sum):
            if agg == "avg":
                # 镜像 trainer.py:L234-235：all_logps = all_logps / valid_length
                with torch.no_grad():
                    mask = (labels[:, 1:] != IGNORE_INDEX).float()
                    vlen = mask.sum(-1)
                return logps_sum / vlen
            return logps_sum

        ref_c, ref_r = None, None
        if ref_sum is not None:
            ref_norm = normalize(ref_sum)
            ref_c, ref_r = ref_norm.split(n, dim=0)

        policy = copy.deepcopy(ref_model)
        opt = torch.optim.Adam(policy.parameters(), lr=LR)
        curve = []
        for step in range(steps):
            policy.train()
            opt.zero_grad()
            logps_sum, vlen = self._forward_logps(policy, input_ids, attn_mask, labels)
            logps = logps_sum / vlen if agg == "avg" else logps_sum
            pol_c, pol_r = logps.split(n, dim=0)
            loss = self.family["fn"](pol_c, pol_r, ref_c, ref_r, self.cfg)
            if self.cfg.pref_ftx > 1e-6:
                # 镜像 trainer.py:L300-302：sft_loss = -policy_chosen_logps_avg
                chosen_avg = pol_c if agg == "avg" else pol_c / vlen[:n]
                loss = loss + self.cfg.pref_ftx * (-chosen_avg.mean())
            loss.backward()
            opt.step()

            if log_every is not None and (step % log_every == 0 or step == steps - 1):
                # 与 L2 train_dpo 同款录值语义：用本步（更新前）前向张量
                margin = float((pol_c - pol_r).mean().item())
                if ref_sum is not None:
                    pair_acc = int(((pol_c - pol_r) > (ref_c - ref_r)).sum().item())
                else:
                    pair_acc = int((pol_c > pol_r).sum().item())
                curve.append((step, loss.item(), margin, pair_acc))
        return policy, curve


class KTOTrainer:
    """stage=kto。KTO（arXiv 2402.01306）：无偏好对，只有单侧二值反馈。
    loss 对照 trl v0.24.0 kto_trainer.py:L1150-1191：
        kl = mean(policy_kl_logps - ref_kl_logps).detach().clamp(min=0)
        chosen_losses   = 1 - sigmoid(beta * (logratio_c - kl))
        rejected_losses = 1 - sigmoid(beta * (kl - logratio_r))
        losses = cat(lambda_D * chosen_losses, lambda_U * rejected_losses)
    KL 基线样本 = 批内循环移位响应（LF 数据侧 feedback.py:L87）。"""

    def __init__(self, vocab, cfg):
        assert cfg.stage == "kto"
        self.vocab = vocab
        self.cfg = cfg
        self.n_ref_forwards = 0

    def fit(self, feedback_examples, ref_model, steps=KTO_STEPS):
        rows, kl_rows, tags = feedback_process(self.vocab, feedback_examples)
        input_ids, attn_mask, labels = collate_rows(rows, self.vocab)
        kl_ids, kl_am, kl_labels = collate_rows(kl_rows, self.vocab)

        ref = copy.deepcopy(ref_model)
        ref.eval()
        ref_main = seq_logps(ref, input_ids, attn_mask, labels)
        ref_kl = seq_logps(ref, kl_ids, kl_am, kl_labels)
        self.n_ref_forwards = 2

        policy = copy.deepcopy(ref_model)
        opt = torch.optim.Adam(policy.parameters(), lr=LR)
        curve = []
        for step in range(steps):
            policy.train()
            opt.zero_grad()
            main_logps, _ = self._forward_logps(policy, input_ids, attn_mask, labels)
            kl_logps, _ = self._forward_logps(policy, kl_ids, kl_am, kl_labels)
            # trl L1150-1151：kl 是批均值、detach、clamp(min=0)
            kl = (kl_logps - ref_kl).mean().detach().clamp(min=0)
            logratio = main_logps - ref_main
            des = tags == 1
            und = ~des
            chosen_losses = 1 - torch.sigmoid(
                self.cfg.pref_beta * (logratio[des] - kl))
            rejected_losses = 1 - torch.sigmoid(
                self.cfg.pref_beta * (kl - logratio[und]))
            losses = torch.cat((
                self.cfg.kto_chosen_weight * chosen_losses,
                self.cfg.kto_rejected_weight * rejected_losses,
            ), 0)
            loss = losses.mean()
            loss.backward()
            opt.step()
            curve.append((step, loss.item(), float(kl.item())))
        return policy, curve

    def _forward_logps(self, model, input_ids, attn_mask, labels):
        logits = model(input_ids, attention_mask=attn_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = (shift_labels != IGNORE_INDEX).float()
        safe = shift_labels.clamp(min=0)
        per_tok = torch.gather(
            shift_logits.log_softmax(-1), dim=2, index=safe.unsqueeze(2)
        ).squeeze(2)
        return (per_tok * mask).sum(-1), mask.sum(-1)


# nano-factory 的「run_exp」：配置进，trainer 出（对照 tuner.py:L138-151）
def build_trainer(vocab, cfg):
    if cfg.stage == "sft":
        return SFTTrainer(vocab, cfg)
    if cfg.stage == "dpo":
        return DPOTrainer(vocab, cfg)
    if cfg.stage == "kto":
        return KTOTrainer(vocab, cfg)
    raise ConfigError(f"Unknown task: {cfg.stage}.")  # tuner.py:L150-151 同款


# ======================================================================
# 测量面（定义与 L2 逐位相同：evaluate_pairs / drift_kl / first_gen_token）
# ======================================================================

@torch.no_grad()
def first_gen_token(model, vocab, system, question):
    tok2id = {t: i for i, t in enumerate(vocab)}
    prompt = apply_chat_template(system, question)
    ids = [tok2id[t] for t in tokenize(prompt)]
    model.eval()
    x = torch.tensor([ids], dtype=torch.long)
    am = torch.ones_like(x)
    logits = model(x, attention_mask=am)
    return int(logits[0, -1].argmax().item())


def evaluate_pairs(model, vocab, pairs):
    """与 L2 的 evaluate_pairs 定义逐位相同（跨级锚的测量口径）。"""
    tok2id = {t: i for i, t in enumerate(vocab)}
    rows = pairwise_process(vocab, pairs)
    input_ids, attn_mask, labels = collate_rows(rows, vocab)
    n = len(pairs)
    logps = seq_logps(model, input_ids, attn_mask, labels)
    pc, pr = logps.split(n, dim=0)
    wins = int((pc > pr).sum().item())
    greedy_ok = 0
    for (sys_, q, c, _r) in pairs:
        if first_gen_token(model, vocab, sys_, q) == tok2id[c]:
            greedy_ok += 1
    return {
        "win": wins, "n": n, "greedy": greedy_ok,
        "p_chosen": float(pc.exp().mean().item()),
        "p_rejected": float(pr.exp().mean().item()),
        "margin": float((pc - pr).mean().item()),
    }


def drift_kl(policy, ref, vocab, pairs):
    """与 L2 的 drift_kl 定义逐位相同：答案决策位 KL(policy||ref)，nats。"""
    tok2id = {t: i for i, t in enumerate(vocab)}
    total = 0.0
    for (sys_, q, _c, _r) in pairs:
        prompt = apply_chat_template(sys_, q)
        ids = torch.tensor([[tok2id[t] for t in tokenize(prompt)]], dtype=torch.long)
        am = torch.ones_like(ids)
        policy.eval(); ref.eval()
        with torch.no_grad():
            lp = policy(ids, attention_mask=am)[0, -1].log_softmax(-1)
            lr = ref(ids, attention_mask=am)[0, -1].log_softmax(-1)
        total += float((lp.exp() * (lp - lr)).sum().item())
    return total / len(pairs)


# ======================================================================
# 跨级锚：L2 tutorial_L2.md §1 paste 块的录值（L2 digest 9353e071…）
# 抽象不变性命题：经配置分发跑出的结果必须与 L2 手写路径逐位相同。
# ======================================================================
L2_ANCHOR = {
    "noisy_final_loss": "0.2334",
    "noisy_win": 4, "noisy_greedy": 4,
    "noisy_p_chosen": "0.4785", "noisy_p_rejected": "0.5205",
    "dpo_step0_loss": "0.6931", "dpo_step0_margin": "-0.0864",
    "dpo_step0_pair_acc": 1,
    "dpo_last_margin": "+89.4095",
    "dpo_win": 6, "dpo_greedy": 6, "dpo_margin": "+89.4783",
    "dpo_p_rejected": "4.23e-33",
    "dpo_drift": "0.9230", "dpo_gap": "+8.9565",
}


def main():
    t0 = time.perf_counter()
    set_seed(SEED)
    print("=" * 68)
    print("nano-llamafactory L3 — one config, three methods: stage dispatch")
    print("=" * 68)

    system = "You are a helpful assistant."
    # 与 L2 完全相同的 6 对偏好数据（chosen=正确和，rejected=off-by-one）。
    # 词表构造顺序也与 L2 相同 => vocab 逐位相同（跨级锚前提之一）。
    pairs = [
        (system, "What is 1+1?", "2", "3"),
        (system, "Compute 2+2", "4", "3"),
        (system, "What is 1+2?", "3", "4"),
        (system, "Compute 2+3", "5", "6"),
        (system, "What is 3+3?", "6", "5"),
        (system, "Compute 1+6", "7", "6"),
    ]
    texts = [apply_chat_template(sys_, q, a) for (sys_, q, a, _r) in pairs] + \
            [apply_chat_template(sys_, q, r) for (sys_, q, _c, r) in pairs]
    vocab = build_vocab(texts)
    print(f"vocab size = {len(vocab)}")
    print(f"model params = {sum(p.numel() for p in TinyLM(len(vocab)).parameters()):,}")

    # ---------------- [0] 配置层 ----------------
    print("\n[0] config layer: derived flags + fail-loud validation")
    print("    (a) use_ref_model derivation (finetuning_args.py:L593 mirror):")
    derived = {}
    for fam in PREF_LOSSES:
        c = NanoFactoryConfig(stage="dpo", pref_loss=fam)
        derived[fam] = c.use_ref_model
        print(f"        pref_loss={fam:7s} -> use_ref_model={c.use_ref_model}")
    print("    (b) fail-loud at config time (not mid-training):")
    try:
        NanoFactoryConfig(stage="dpo", pref_loss="ipo", dpo_label_smoothing=0.1)
        raise AssertionError("should have raised")
    except ConfigError as e:
        msg_smooth = str(e)
        print(f"        dpo+ipo+smoothing=0.1 -> ConfigError: {msg_smooth}")
    try:
        NanoFactoryConfig(stage="grpo")
        raise AssertionError("should have raised")
    except ConfigError as e:
        msg_stage = str(e)
        print(f"        stage=grpo            -> ConfigError: {msg_stage}")
    print("    (c) silent-inert field, made loud by nano (LF stays silent):")
    c_inert = NanoFactoryConfig(stage="sft", pref_ftx=0.5)
    for w in c_inert.warnings:
        print(f"        [nano warning] {w}")

    # ---------------- [1] 分发层 ----------------
    print("\n[1] dispatch layer: tables, not branches")
    print(f"    STAGE2DATA_STAGE = {STAGE2DATA_STAGE}")
    print("    (dpo -> 'rm': the data layer dispatches on DATA SHAPE, not method")
    print("     name; dpo/workflow.py:L45 calls get_dataset(..., stage='rm'))")
    same_proc = resolve_processor("dpo") is DATA_STAGE2PROCESSOR["rm"]
    print(f"    dpo resolves to the SAME processor object as rm: {same_proc}")
    # 同一条数据，三个 processor 渲染出的行必须逐位相同（数据侧零改动复用）
    probe = (system, "Compute 2+2", "4")
    row_sft = supervised_process(vocab, [probe])[0]
    row_dpo = pairwise_process(vocab, [(system, "Compute 2+2", "4", "3")])[0]
    fb_examples = [(system, "Compute 2+2", "4", "desirable"),
                   (system, "Compute 2+2", "3", "undesirable")]
    row_kto = feedback_process(vocab, fb_examples)[0][0]
    row_same = (row_sft == row_dpo == row_kto)
    n_sup = sum(1 for x in row_sft[2] if x != IGNORE_INDEX)
    print(f"    one row under three processors (sft / dpo / kto):")
    print(f"        input_ids+attention_mask+labels byte-identical: {row_same}"
          f"  (supervised tokens = {n_sup})")

    # ---------------- [2] sft stage：跨级锚 A ----------------
    print("\n[2] stage=sft (cross-level anchor: must reproduce L2 noisy SFT bit-for-bit)")
    noisy_rows = (supervised_process(vocab, [(s, q, c) for (s, q, c, _r) in pairs])
                  + supervised_process(vocab, [(s, q, r) for (s, q, _c, r) in pairs]))
    set_seed(SEED)  # 与 L2 main() 的 noisy SFT 同款种子位点
    sft_trainer = build_trainer(vocab, NanoFactoryConfig(stage="sft"))
    model_noisy, losses = sft_trainer.fit(noisy_rows)
    ev_noisy = evaluate_pairs(model_noisy, vocab, pairs)
    print(f"    final loss = {losses[-1]:.4f}  (L2 recorded {L2_ANCHOR['noisy_final_loss']})")
    print(f"    win={ev_noisy['win']}/6  greedy={ev_noisy['greedy']}/6  "
          f"p_chosen={ev_noisy['p_chosen']:.4f}  p_rejected={ev_noisy['p_rejected']:.4f}")
    sft_match = (
        f"{losses[-1]:.4f}" == L2_ANCHOR["noisy_final_loss"]
        and ev_noisy["win"] == L2_ANCHOR["noisy_win"]
        and ev_noisy["greedy"] == L2_ANCHOR["noisy_greedy"]
        and f"{ev_noisy['p_chosen']:.4f}" == L2_ANCHOR["noisy_p_chosen"]
        and f"{ev_noisy['p_rejected']:.4f}" == L2_ANCHOR["noisy_p_rejected"]
    )
    print(f"    cross-level match vs L2: {sft_match}")

    # ---------------- [3] dpo stage（sigmoid）：跨级锚 B ----------------
    print(f"\n[3] stage=dpo pref_loss=sigmoid beta={BETA} "
          f"(cross-level anchor: must reproduce L2 [2] bit-for-bit)")
    set_seed(SEED)  # 与 L2 main() 的 DPO 同款种子位点
    dpo_cfg = NanoFactoryConfig(stage="dpo", pref_loss="sigmoid")
    dpo_trainer = build_trainer(vocab, dpo_cfg)
    policy, curve = dpo_trainer.fit(pairs, model_noisy, steps=DPO_STEPS, log_every=40)
    ev_dpo = evaluate_pairs(policy, vocab, pairs)
    dpo_drift = drift_kl(policy, model_noisy, vocab, pairs)
    gap = BETA * (ev_dpo["margin"] - ev_noisy["margin"])
    for (step, loss, margin, pair_acc) in curve:
        print(f"    step {step:3d}: loss={loss:.4f}  margin={margin:+.4f}  "
              f"pair_acc={pair_acc}/6")
    print(f"    final: win={ev_dpo['win']}/6  greedy={ev_dpo['greedy']}/6  "
          f"margin={ev_dpo['margin']:+.4f}  p_rejected={ev_dpo['p_rejected']:.3g}")
    print(f"    drift KL = {dpo_drift:.4f} nats   implicit reward gap = {gap:+.4f}")
    step0, last = curve[0], curve[-1]
    dpo_match = (
        f"{step0[1]:.4f}" == L2_ANCHOR["dpo_step0_loss"]
        and f"{step0[2]:+.4f}" == L2_ANCHOR["dpo_step0_margin"]
        and step0[3] == L2_ANCHOR["dpo_step0_pair_acc"]
        and f"{last[2]:+.4f}" == L2_ANCHOR["dpo_last_margin"]
        and ev_dpo["win"] == L2_ANCHOR["dpo_win"]
        and ev_dpo["greedy"] == L2_ANCHOR["dpo_greedy"]
        and f"{ev_dpo['margin']:+.4f}" == L2_ANCHOR["dpo_margin"]
        and f"{ev_dpo['p_rejected']:.3g}" == L2_ANCHOR["dpo_p_rejected"]
        and f"{dpo_drift:.4f}" == L2_ANCHOR["dpo_drift"]
        and f"{gap:+.4f}" == L2_ANCHOR["dpo_gap"]
    )
    print(f"    cross-level match vs L2: {dpo_match}")
    print("    (dispatch layer is numerically INERT: it reorganizes code,")
    print("     it does not change a single number)")

    # ---------------- [4] 一个字符串三个行为：pref_loss 扫描 ----------------
    print("\n[4] one string, three behaviors: pref_loss sweep (same data, same seed)")
    print("    family   needs_ref  aggregate  init_loss   final_loss  win  greedy  margin")
    sweep = {}
    for fam in PREF_LOSSES:
        cfg = NanoFactoryConfig(stage="dpo", pref_loss=fam)
        set_seed(SEED)
        tr = build_trainer(vocab, cfg)
        pol, cv = tr.fit(pairs, model_noisy, steps=SWEEP_STEPS,
                         log_every=SWEEP_STEPS - 1)  # 录 step 0 与末步两点
        ev = evaluate_pairs(pol, vocab, pairs)
        sweep[fam] = (cfg, tr, ev)
        print(f"    {fam:8s} {str(cfg.use_ref_model):10s} "
              f"{PREF_LOSS_REGISTRY[fam]['aggregate']:9s}  "
              f"{cv[0][1]:9.4f}  {cv[-1][1]:9.4f}  "
              f"{ev['win']}/6  {ev['greedy']}/6  {ev['margin']:+8.4f}")
    # 解析锚：init（policy==ref）时**有 ref 的 loss 族**有闭式值——
    # 因为 margin_policy 与 margin_ref 逐项相消为 0。无 ref 族（orpo/simpo）
    # 没有这个相消，init loss 取决于真实（非零）margin，无闭式锚。
    init_anchors = {
        "sigmoid": (-math.log(0.5), "ln 2"),
        "ipo": (1.0 / (4.0 * BETA ** 2), "1/(4 beta^2)"),
    }
    print("    init-loss analytical anchors (policy == ref => margin cancels to 0):")
    init_ok = True
    for fam, (val, formula) in init_anchors.items():
        # 重新单步计算 init loss（确定性，便宜）
        cfg = NanoFactoryConfig(stage="dpo", pref_loss=fam)
        rows = pairwise_process(vocab, pairs)
        ids, am, labs = collate_rows(rows, vocab)
        n = len(pairs)
        agg = PREF_LOSS_REGISTRY[fam]["aggregate"]
        sums = seq_logps(model_noisy, ids, am, labs)
        if agg == "avg":
            mask = (labs[:, 1:] != IGNORE_INDEX).float()
            sums = sums / mask.sum(-1)
        pc, pr = sums.split(n, dim=0)
        rc, rr = (pc.clone(), pr.clone()) if PREF_LOSS_REGISTRY[fam]["needs_ref"] \
            else (None, None)
        got_val = float(PREF_LOSS_REGISTRY[fam]["fn"](pc, pr, rc, rr, cfg).item())
        ok = abs(got_val - val) < 1e-6
        init_ok = init_ok and ok
        print(f"        {fam:8s}: computed {got_val:.6f}  == {formula} = {val:.6f}  ({ok})")
    ref_free_ok = (sweep["orpo"][1].n_ref_forwards == 0
                   and sweep["simpo"][1].n_ref_forwards == 0
                   and sweep["sigmoid"][1].n_ref_forwards == 1
                   and sweep["ipo"][1].n_ref_forwards == 1)
    print(f"    ref forwards used: orpo={sweep['orpo'][1].n_ref_forwards}, "
          f"simpo={sweep['simpo'][1].n_ref_forwards}, "
          f"sigmoid={sweep['sigmoid'][1].n_ref_forwards}, "
          f"ipo={sweep['ipo'][1].n_ref_forwards}  "
          f"(config-derived use_ref_model respected: {ref_free_ok})")

    # ---------------- [5] kto stage：第三种数据形状 ----------------
    print("\n[5] stage=kto: unpaired feedback + cyclic-shift KL baseline")
    # KTO 数据 = 同样的 12 条响应，但拆成带 tag 的单侧样本（不再是「对」）
    fb_examples = []
    for (s, q, c, r) in pairs:
        fb_examples.append((s, q, c, "desirable"))
        fb_examples.append((s, q, r, "undesirable"))
    rows, kl_rows, tags = feedback_process(vocab, fb_examples)
    # 机器证明：第 i 条的 KL 响应 == 第 i-1 条的响应（循环移位，feedback.py:L87）
    shift_ok = all(
        rows[i][0] == kl_rows[(i + 1) % len(rows)][0] for i in range(len(rows))
    )
    # 上式等价于 kl_rows[i] == rows[i-1]；这里用正向索引再验一遍
    shift_ok = shift_ok and all(
        kl_rows[i][0] == rows[(i - 1) % len(rows)][0] for i in range(len(rows))
    )
    print(f"    kl baseline row[i] == response[i-1 mod {len(rows)}] "
          f"(cyclic shift, feedback.py:L87 mirror): {shift_ok}")
    set_seed(SEED)
    kto_cfg = NanoFactoryConfig(stage="kto")
    kto_trainer = build_trainer(vocab, kto_cfg)
    policy_kto, kto_curve = kto_trainer.fit(fb_examples, model_noisy)
    kl_last = kto_curve[-1][2]
    # 测量：逐条 beta*logratio（desirable 应升、undesirable 应降）+ greedy
    ids, am, labs = collate_rows(rows, vocab)
    pol_logps = seq_logps(policy_kto, ids, am, labs)
    ref_logps = seq_logps(model_noisy, ids, am, labs)
    logratio = (pol_logps - ref_logps) * BETA
    des_mask = tags == 1
    lr_des = float(logratio[des_mask].mean().item())
    lr_und = float(logratio[~des_mask].mean().item())
    tok2id = {t: i for i, t in enumerate(vocab)}
    greedy_kto = sum(
        first_gen_token(policy_kto, vocab, s, q) == tok2id[c] for (s, q, c, _r) in pairs
    )
    print(f"    after {KTO_STEPS} steps: loss {kto_curve[0][1]:.4f} -> "
          f"{kto_curve[-1][1]:.4f}  (kl estimate = {kl_last:.4f} >= 0 by clamp)")
    print(f"    beta*logratio: desirable mean = {lr_des:+.4f}, "
          f"undesirable mean = {lr_und:+.4f}")
    print(f"    greedy correct = {greedy_kto}/6  (noisy ref was 4/6; no pairs used, "
          f"tags only)")

    # ---------------- [6] self-check ----------------
    print("\n" + "=" * 68)
    checks = [
        ("use_ref_model derivation matches L593 (sigmoid/ipo True, orpo/simpo False)",
         derived == {"sigmoid": True, "ipo": True, "orpo": False, "simpo": False}),
        ("config rejects smoothing+non-sigmoid with LF's exact message",
         msg_smooth == "`dpo_label_smoothing` is only valid for sigmoid loss function."),
        ("config rejects unknown stage with tuner.py's exact message",
         msg_stage == "Unknown task: grpo."),
        ("dpo resolves to the same processor object as rm (data-shape dispatch)",
         same_proc),
        ("one row byte-identical under sft/dpo/kto processors", row_same),
        ("sft stage reproduces L2 noisy SFT bit-for-bit", sft_match),
        ("dpo-sigmoid stage reproduces L2 [2] bit-for-bit", dpo_match),
        ("init losses match closed forms via ref-cancellation (sigmoid ln2, ipo 1/4b^2)",
         init_ok),
        ("ref usage matches config derivation (orpo/simpo 0, sigmoid/ipo 1)",
         ref_free_ok),
        ("kto cyclic-shift KL baseline property", shift_ok),
        ("kto kl estimate >= 0 (clamp)", kl_last >= 0.0),
        ("kto separates tags: beta*logratio desirable > undesirable", lr_des > lr_und),
        ("kto learns from tags alone: greedy >= 5/6", greedy_kto >= 5),
        ("loss families actually differ (final losses not all equal)",
         len({f"{sweep[f][2]['margin']:.4f}" for f in PREF_LOSSES}) > 1),
    ]
    passed = sum(1 for _name, ok in checks if ok)
    for name, ok in checks:
        assert ok, f"self-check failed: {name}"
    digest_src = (
        f"sft:{losses[-1]:.4f}/{ev_noisy['win']}/{ev_noisy['greedy']}|"
        f"dpo:{ev_dpo['win']}/{ev_dpo['greedy']}/{ev_dpo['margin']:.4f}/"
        f"{dpo_drift:.4f}/{gap:.4f}|"
        + "|".join(f"{f}:{sweep[f][2]['win']}/{sweep[f][2]['greedy']}/"
                   f"{sweep[f][2]['margin']:.4f}" for f in PREF_LOSSES)
        + f"|kto:{greedy_kto}/{lr_des:.4f}/{lr_und:.4f}/{kl_last:.4f}"
    )
    digest = hashlib.md5(digest_src.encode()).hexdigest()
    print(f"[self-check] {passed}/{len(checks)} PASS")
    print(f"digest: {digest}")
    print(f"    elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
