# nano-vlm-understanding

这个模块回答一个基础问题：二维像素怎样变成 LLM 能消费、又能被问题选择的 token？它先拆开
`patch → projector → 2D position → packed sequence → causal attention → answer`，再用反事实证明模型是否真的看图。

## 立即运行

```bash
python3 -B L0_visual_tokens_to_language.py
```

仅依赖 Python 3.10+ 标准库，CPU、离线可跑。先读 [notebook-style 教程](tutorial_L0.md)，再改代码。

## L0–L3 阶梯

| 级别 | 项目 | 验收重点 | 状态 |
|---|---|---|---|
| L0 | 固定 visual projector + causal readout | patch/token、2D 位置、image-drop/swap/shuffle 与分技能 EM | 已完成 |
| L1 | Qwen3-VL-2B-Instruct 小样本推理 | OCR、空间、计数、图像交换、证据不足拒答 | 实现已就绪，待 L20 真机验收（不计完成） |
| L2 | Qwen3-VL 系统实验 | 动态分辨率、visual token budget、DeepStack、interleaved MRoPE、batching、connector/LoRA | 规划中 |
| L3 | 源码对照与评测边界 | 固定 revision，解释视觉编码/融合路径；盲评与代理指标分栏 | 规划中 |

## L0 量化合同

- `skill_exact_match`：按 top-left / center / bottom-right 分技能，而非只报总平均。
- `image_dependence_gain`：baseline 与 image-drop 的 EM 差。
- `counterfactual_sensitivity`：换图后答案改变的比例；它衡量图像依赖，不等于正确率。
- 固定反例：image-drop、image-swap、patch-shuffle、移除二维位置。

L0 是手工 projector/readout 的机制模拟，不是训练后的 VLM，也不证明 OCR、grounding 或开放世界视觉能力。

## L1 staging 状态

[L1_qwen3_vl_real_probe.py](L1_qwen3_vl_real_probe.py) 已固定
`Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`，并生成 6 个 synthetic diagnostics：
OCR、空间、计数、同问题 image-swap pair 与空白图拒答。脚本保留全部失败在 completion 分母，重复两轮 greedy 推理，
分开输出 exact match、prediction stability、swap sensitivity、峰值显存和每个原始回答。

当前**不把 L1 标为完成**：只有 L20 上真实 checkpoint 两轮运行、教程真实输出和静态 QA 全部闭环后才能转正。

## 文件

- [L0_visual_tokens_to_language.py](L0_visual_tokens_to_language.py)：≤200 行单文件实验。
- [tutorial_L0.md](tutorial_L0.md)：推导、真实输出、反例与练习。
- [L1_qwen3_vl_real_probe.py](L1_qwen3_vl_real_probe.py)：真实 2B checkpoint 诊断脚本；等待 GPU gate。
- [上级研究账本](../RESEARCH.md)：CLIP → Flamingo → BLIP-2 → LLaVA → Qwen3-VL 的证据谱系。
