# nano-llamafactory

> **抓的核心机制**：SFT 的**数据侧三件套**——chat template（定 response 边界）/
> loss mask（-100 定 loss）/ collator（pad 双层遮罩），以及「SFT 与预训练共用
> 同一个 next-token loss，差别只在 labels 遮罩」这一本质。
> 注意：LLaMA-Factory 本身是全功能微调框架（SFT/DPO/PPO/评估/导出一站式 +
> Web UI），本模块只取其 SFT 数据侧的最小机制做阶梯——它是后续一切训练方法
> （含 L2 的 DPO 偏好对构造）的数据侧地基。
> **对应真实系统**：[LlamaFactory](https://github.com/hiyouga/LlamaFactory)
> **轨道**：[01 后训练/RL/SFT](../README.md) · **状态**：L0–L3 ✅

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | single-file 确定性 toy：template 两种渲染 + labels 遮罩（shift 边界）+ collator 双层遮罩；用 coasting 模型量化 mask 的作用（零依赖，CPU 即跑） | ✅ [L0_sft_data_pipeline.py](L0_sft_data_pipeline.py) · [tutorial_L0.md](tutorial_L0.md) |
| **L1** | 真实小模型 + torch 梯度下降跑最小 SFT 循环：在本节构造的 (input_ids, attention_mask, labels) 上训练，验证遮罩决定模型学到什么（scaffold 原「几步优化」目标在此落地） | ✅ [L1_minimal_sft.py](L1_minimal_sft.py) · [tutorial_L1.md](tutorial_L1.md) |
| **L2** | DPO：偏好对用同一套 template/mask 机器构造（真实入口 `collator.py:L564` PairwiseDataCollatorWithPadding，2026-08-13 抓取；08-05 快照录 L553，上游漂移），reference-model KL 约束，对比 SFT vs DPO；按课程的经典证据层要求写明 DPO 当今定位 | ✅ [L2_dpo_preference_pairs.py](L2_dpo_preference_pairs.py) · [tutorial_L2.md](tutorial_L2.md) |
| **L3** | 对照固定 LlamaFactory revision 的配置体系：一个 `stage` 分发 SFT/DPO/KTO，`pref_loss` 切换偏好 loss 族，并用跨级锚证明分发层数值惰性 | ✅ [代码](L3_stage_dispatch.py) · [教程](tutorial_L3.md) |

## 环境依赖

- L0：零外部依赖（纯标准库），CPU 即跑，输出确定。
- L1：torch（真实小模型 SFT）。
- L2：torch（CPU 即可，实测 ~4s）；偏好对在脚本内构造（6 对单位数加法），
  固定 seed 确定性输出，掩码锚 `f8b50175…`/51 行、digest `9353e071…`。
- L3：torch（CPU 即可）；源码对照钉住 LlamaFactory revision，行号不外推到当前 main。

## 核心要讲清的点

- SFT = labels 被遮罩过的预训练：训练循环不变，差别全在数据侧
- 模板定边界：推理 prompt 必须是训练串的真前缀（train/test 一致性）
- mask 边界画在 labels 空间（被预测端），shift 发生在 loss 计算时——混淆两个空间就丢第一个 response token
- pad 双层遮罩（attention_mask 管前向、labels 管反向）；正确 collate 不改变 loss 数值
- coasting 实验：不遮罩时 loss 被模板稀释，低 loss ≠ 会回答
- 偏好对 = 两条普通 SFT 行（机器证明）：数据侧零发明，DPO 的新东西全在 loss 侧
- DPO loss = −log σ(β·(margin_policy − margin_ref))；ref 的三个角色 = policy 起点
  + margin 基线 + KL 球锚（β 是缰绳，drift 是读数）
- pair loss 的两个失明：对 pair 之外的概率质量失明（质量泄漏，win 6/6 与 greedy
  崩坏可同时成立）、对标签方向失明（颠倒偏好对教出自信答错的模型）

## 费曼自检

- 能不能用「批改答题卡」讲清：为什么题干（prompt）不算分、为什么空白补位区（pad）既不阅读也不给分？
- 能不能用「不看标准答案、只跟入职基线比的教练」讲清：为什么 DPO 不需要 reward
  model 却仍需要 reference model？为什么 win 6/6 不保证模型会答题？

## 权威实现与延伸

- 对标源码：LLaMA-Factory `github.com/hiyouga/LLaMA-Factory`（main，2026-08-05
  快照实测）：`data/processor/supervised.py:L109`（`<bos> X Y <eos>` /
  `<ignore>...<ignore> Y <eos>` 约定）、`extras/constants.py:L50`
  （IGNORE_INDEX=-100）、`data/collator.py:L137/L492/L553`、
  `data/template.py:L60/L132`。详见 tutorial_L0.md §6/§11。
- L2 源码锚点（main @`0bbe481e`，2026-08-13 codeload tarball 抓取，HEAD
  `0bbe481e6e621527284d37f1e13a6b9556c303ec`）：`data/collator.py:L564`
  （2n 行 chosen-first；08-05 快照录 L553，上游漂移）、
  `data/processor/pairwise.py:L66`、`train/dpo/trainer.py:L187/L219-253`、
  `train/trainer_utils.py:L592`、`hparams/finetuning_args.py:L171/L183/L593`。
  详见 tutorial_L2.md §5/§10。
- 概念延伸：SFT warmup → RL 的顺序 → [nano-verl](../nano-verl/) L1；
  训练好的模型进入 agent → 轨道 04。
