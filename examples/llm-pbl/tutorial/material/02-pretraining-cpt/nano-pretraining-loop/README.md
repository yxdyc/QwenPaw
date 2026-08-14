# nano-pretraining-loop

> **核心机制**：模型能放进集群只是起点；完整预训练还必须把文档变成合法 causal targets，确定数据顺序与
> mixture，执行 optimizer/schedule/gradient accumulation，并保存能恢复同一计算过程的完整 checkpoint。
>
> **轨道**：[02 预训练 / CPT](../README.md) · **状态**：L0 ✅，L1–L3 待补。

## 阶梯

| 级别 | 目标 | 状态 |
|------|------|------|
| L0 | 纯 Python bigram LM：document boundary、causal shift、mixture/shuffle/cursor、AdamW、warmup/decay、gradient accumulation、validation、完整 resume | ✅ [代码](L0_pretraining_lifecycle.py) · [教程](tutorial_L0.md) |
| L1 | 单卡真实小 Transformer：tokenizer/packing/mask、AMP、真实 optimizer/scheduler/RNG checkpoint 与 exact-resume 边界 | 🔲 |
| L2 | 接 FSDP/Megatron：distributed sampler、global batch、sharded checkpoint、故障恢复和 loss spike triage | 🔲 |
| L3 | 对照权威框架的数据 loader、checkpoint schema、训练日志与吞吐/稳定性控制 | 🔲 |

## 先修与后续

- 本模块先于 [nano-fsdp](../nano-fsdp/) 和 [nano-megatron](../nano-megatron/) 回答“被切分的完整训练过程是什么”。
- 数据快照与 lineage 见 [nano-data-platform L0](../../03-data-distributed-rsi/nano-data-platform/tutorial_L0.md)。
- L1 后再把完整状态搬进 FSDP/TP/PP，而不是把“参数能分片”误作“训练能恢复”。
