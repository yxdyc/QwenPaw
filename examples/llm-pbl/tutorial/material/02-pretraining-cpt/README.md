# 轨道 02 — 预训练 / CPT Infra

> **一句话**：从随机初始化训出一个 base model（pre-training），或在已有模型上灌入领域知识（continual pre-training, CPT），所需的并行与稳定性工程。
> **对标权威实现**：Megatron-LM · PyTorch FSDP / DeepSpeed ZeRO ｜ **SOTA 参照**：DeepSeek（MoE + MLA + 训练稳定性）

---

## 这条线学什么

预训练的工程核心是**训练生命周期、并行与稳定性**：
- 文档要经过 boundary-aware tokenization/packing、deterministic sampling、causal loss 和 optimizer/schedule，
  checkpoint 必须恢复模型、优化器、RNG 与 data cursor，而不只是权重。
- 模型大到单卡放不下 → 需要 TP（张量并行）/ PP（流水线并行）/ SP（序列并行）/ FSDP（ZeRO 分片）。
- 训练动辄数周 → 稳定性工程（loss spike 恢复、梯度裁剪、checkpoint、MFU 优化）决定成败。

| nano-* | 抓的核心机制 | 对标权威实现 |
|--------|-------------|--------------|
| `nano-pretraining-loop` | 文档→causal targets→sample order/mixture→AdamW/schedule→完整 checkpoint/resume | L0 自包含机制；L1+ 对照 PyTorch/Megatron data/checkpoint stack |
| `nano-megatron` | TP + PP + SP 的最小可跑实现，理解切分如何通信 | Megatron-LM |
| `nano-fsdp` | ZeRO / FSDP 参数-梯度-优化器分片，理解显存账本 | PyTorch FSDP / DeepSpeed ZeRO |

---

## 学习路径（K+1 阶梯）

```
前置：会单卡 transformer 训练、懂 all-reduce/all-gather 是什么（K）
  │
  ▼
Step 1  nano-pretraining-loop L0 ← 先建立文档到完整 checkpoint 的生命周期
  │
  ▼
Step 2  nano-fsdp L0–L3        ← 理解「显存去哪了」，ZeRO/FSDP 分片
  │
  ▼
Step 3  nano-megatron L0–L3    ← 张量/流水线/序列并行与 MFU
  │
  ▼
Step 4  pretraining-loop L1–L2  ← 真实 Transformer + distributed exact-resume 边界
  │
  ▼
Step 5  sota-deepdive: DeepSeek ← MoE + MLA + 训练稳定性的 SOTA 工程
```

---

## 完成标志

- [ ] 能画出 ZeRO-1/2/3 的显存账本，算出给定模型/卡数的每卡占用
- [ ] 能解释 document boundary、data mixture、global batch 与 causal labels 怎样共同定义训练目标
- [ ] 能列出可恢复 checkpoint 的模型/优化器/scheduler/RNG/data cursor 状态，并说明何时不能 bitwise resume
- [ ] 能用 single-file 跑通一个 2 卡张量并行 MLP，说清 all-reduce 插在哪
- [ ] 能解释序列并行（SP）为什么能省 activation 显存
- [ ] 能列出一套 loss spike 的诊断与恢复流程
- [ ] 能讲清 DeepSeek 的 MLA（多头潜在注意力）和 MoE 在训练上的至少 2 个工程要点

---

## 权威实现与 SOTA 参照

写材料须回到一手来源（源码 / 技术报告），拿不准标 `[TODO: verify]`：
- Megatron-LM：`github.com/NVIDIA/Megatron-LM`（TP/PP/SP 切分与通信）
- PyTorch FSDP：`docs.pytorch.org/docs/stable/fsdp.html`；DeepSpeed ZeRO：`github.com/deepspeedai/DeepSpeed`
- 跨轨概念依赖：CPT 的领域语料清洗见轨道 03 `nano-data-juicer`
- SOTA：DeepSeek-V3 技术报告（MoE / MLA / 训练稳定性）`[TODO: verify arXiv]`

→ 深挖见 [sota-deepdive/](sota-deepdive/)
