# SOTA Deep-Dive — 预训练 / MoE + MLA + 训练稳定性

> **深挖对象**：DeepSeek-V3（MoE 路由 + MLA 压缩 + FP8 训练 + 训练稳定性，首版 ✅）；DeepSeek-V4 为更新一代替代（摘要级，不作教学主体）。
> **状态**：首版完成（SOTA 对齐日期 2026-08-11/12）
> **对照基础**：nano-megatron L0–L3（25/25 转正，关联的后训练/预训练轨材料，关联的数据/Agent 轨只读引用、标锚点；L2 锚点门槛超额满足，ROADMAP §四.4）

---

## 阶梯状态

| 文件 | 状态 | 说明 |
|------|------|------|
| [`deepseek-moe-mla-stability.md`](deepseek-moe-mla-stability.md) | ✅ 首版（2026-08-12） | 机制面 ×4（MoE 路由 + aux-loss-free 偏置负载均衡 / MLA 低秩压缩 + absorbed + 解耦 RoPE / FP8 细粒度 + 高精度累加 / 梯度裁剪与训练稳定性），每面一手来源逐字引文 + sim 实测双证 + nano-megatron 实测锚交叉引用；费曼四件齐备；V4 定位 + T-Rex 单源标注 |
| [`deepseek_v3_mechanisms_sim.py`](deepseek_v3_mechanisms_sim.py) | ✅ 冻结基线（2026-08-12 00:57） | 四个机制面的可运行本质模拟（toy 尺度 + 真实格式语义，真实 `float8_e4m3fn` 量化-反量化）；仅依赖 torch、CPU 即跑、seed=3 跨运行逐字节一致、self-check 20/20 |

## 环境依赖

- **sim**：仅 `torch`（本机 torch 2.13.0 实测；CPU 即跑，秒级；fp8 用真实 E4M3 格式做量化-反量化，矩阵计算在 fp32/fp64——模拟量化误差，不模拟硬件 kernel）。
- **运行**：`python3 -B deepseek_v3_mechanisms_sim.py`（任意 CWD，`-B` 防 pycache；无计时行，跨运行逐字节一致）。
- **输出锚**：md5 `45cf39f335c5b8940068506fc8df24c4` / 4,957 B；digest `1e5fffacca552774c0fce81d6f9f3e35`；sim 文件 md5 `2a9ce9ceedfee9b91f063c88e000b218` / 448 行 / 27,099 B。

## 深挖什么（scope）

1. **MoE 训练工程**（首版已覆盖）：sigmoid 路由 + 组限制 top-K + aux-loss-free 偏置负载均衡（V3 §2.1.2 Eq.16）；EP / 跨节点 all-to-all 与组限制路由的协同设计（V3 §3.2.2）。
2. **MLA**（首版已覆盖）：低秩 KV 联合压缩 + absorbed 推理路径 + 解耦 RoPE 的必要性（V2 §2.1/§2.1.3）；KV cache 压缩比双口径分离（93.3% vs 67B / 71.1× V3 config）。
3. **FP8 训练**（首版已覆盖）：细粒度（1×128 tile / 128×128 block）在线量化 + 高精度累加（promotion to CUDA Cores）（V3 §3.3）。
4. **训练稳定性**（首版已覆盖）：梯度裁剪（clip norm = 1.0，V3 唯一披露的稳定性旋钮，V3 §4.2）；零 irrecoverable loss spike 的一揽子归因。
5. **可迁移的工程选择**（首版 §5 已覆盖）：与 nano-megatron L0–L3 实测锚（TP×PP×SP 组合 / SP 通信账 / MFU 三段分解 / PP bubble）的对应关系。

## 信息溯源要求（反幻觉硬约束）

- 数字/结论必须来自一手来源（技术报告 arXiv / 开源代码）。
- 拿不到就标 `[TODO: verify]`，绝不凭印象写。
- 区分：原文声称 / 文献已有 / 合理推断 / 猜测（见 deepdive §8.4）。

## 来源清单（首版已核验，2026-08-12 现场重抓；arXiv 经 export.arxiv.org API，论文经 ar5iv，源码经 raw.githubusercontent.com）

- [x] **DeepSeek-V3 Technical Report** `[2412.19437]`（v2，2024-12-27 / 2025-02-18）——机制面规范锚点；ar5iv 470,563 B 原文 12+ 处逐条命中（Eq.16 + bias 只选路 / bias update speed / γ=0.001→0.0 @14.3T/500B / α=0.0001 / M=4 / clip norm 1.0 / K=4096 近 2% / 14 bits / Online Quantization / promotion to CUDA Cores / no irrecoverable loss spikes / batch 3072→15360 / 2.788M H800 GPU hours / 14.8T tokens）。
- [x] **DeepSeek-V2** `[2405.04434]`（v5）——MLA 原始提出文；93.3% 口径（vs DeepSeek 67B）+ §2.1/§2.1.2/§2.1.3 节号实在。
- [x] **DeepSeekMoE** `[2401.06066]` / **Aux-Loss-Free Load Balancing** `[2408.15664]`——A 层经典锚点。
- [x] **官方推理源码**（github.com/deepseek-ai/DeepSeek-V3，main 2026-08-12 抓取）：`inference/model.py`（32,831 B，md5 `18498c730ab8e3460b93de313c2bc6cc`，行锚点 L483/L484/L486/L494-495/L535/L564/L566-598/L585/L594 逐一在位）+ `inference/configs/config_671B.json`（503 B，md5 `bb3ea9736753cadf24f8cd6f4275bd6c`，17 字段与 sim 逐项吻合）。行号以 2026-08-11/12 抓取日为准。
- [x] **DeepSeek-V4** `[2606.19348]`（2026-04-26）——更新一代替代，存在性坐实；机制细节（CSA/HCA/mHC/Muon）仅摘要级、标 `[TODO: verify]`、不作教学主体（ROADMAP §八 B 层处理）。
- [x] **SLAI T-Rex** `[2607.20145]`（2026-07-22 / v2 2026-07-30）——`[transient/单源]`，仅录不展开。

## 权威实现与延伸

- 轨道 [02](../README.md)；落地参照 nano-megatron（L0–L3 满阶可运行锚点，关联的后训练/预训练轨只读）/ nano-fsdp（混合精度显存账，关联的后训练/预训练轨只读）/ nano-vllm-sglang（paged KV cache，关联的数据/Agent 轨）
- 一手来源：DeepSeek-V3 技术报告 `[2412.19437]` + 官方推理源码（详见 deepdive §8.1/§8.2）
