# nano-vlm-understanding

这个模块回答一个基础问题：二维像素怎样变成 LLM 能消费、又能被问题选择的 token？它先拆开
`patch → projector → 2D position → packed sequence → causal attention → answer`，再用反事实证明模型是否真的看图。
其中 readout 指从融合后的 hidden states 选择、聚合并映射为文字/坐标/动作的路径，不等同于视觉 encoder，也不必是独立层。

## 立即运行

```bash
python3 -B L0_visual_tokens_to_language.py
```

仅依赖 Python 3.10+ 标准库，CPU、离线可跑。先读 [notebook-style 教程](tutorial_L0.md)，再改代码。

## L0–L3 阶梯

| 级别 | 项目 | 验收重点 | 状态 |
|---|---|---|---|
| L0 | 固定 visual projector + causal readout | patch/token、2D 位置、image-drop/swap/shuffle 与分技能 EM | 已完成 |
| L1 | Qwen3-VL-2B-Instruct 小样本推理 | OCR、空间、计数、图像交换、证据不足拒答 | 已完成（单张 L20，两独立进程） |
| L2 | Qwen3-VL 系统实验 | 动态分辨率、visual token budget、DeepStack、interleaved MRoPE、batching、connector/LoRA | 规划中 |
| L3 | 源码对照与评测边界 | 固定 revision，解释视觉编码/融合路径；盲评与代理指标分栏 | 规划中 |

## L0 量化合同

- `skill_exact_match`：按 top-left / center / bottom-right 分技能，而非只报总平均。
- `image_dependence_gain`：baseline 与 image-drop 的 EM 差。
- `counterfactual_sensitivity`：换图后答案改变的比例；它衡量图像依赖，不等于正确率。
- 固定反例：image-drop、image-swap、patch-shuffle、移除二维位置。

L0 是手工 projector/readout 的机制模拟，不是训练后的 VLM，也不证明 OCR、grounding 或开放世界视觉能力。

## L1 L20 真机证据

[L1_qwen3_vl_real_probe.py](L1_qwen3_vl_real_probe.py) 已固定
`Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`，并生成 6 个 synthetic diagnostics：
OCR、空间、计数、同问题 image-swap pair 与空白图拒答。脚本保留全部失败在 completion 分母，重复两轮 greedy 推理，
分开输出 normalized semantic accuracy、strict-format accuracy、prediction stability、swap sensitivity、
“swap 后既改变又答对”、视觉 patch/token 账、端到端延迟、峰值显存和每个原始回答。

这些量回答不同问题：semantic accuracy 看答案内容，strict-format 看指令遵循，sensitivity 只看模型是否随图改变，
counterfactual correctness 才要求两张图都答对。任何一个都不能替代其余三个。

评测尺子仍可在下载权重前单独验证；该命令只依赖 Python 3.10+ 标准库：

```bash
python3 -B L1_qwen3_vl_real_probe.py --self-test-metrics
```

它固定覆盖四种易混淆情形：normalized 不等于 strict、换图后又改变又正确、换图后改变但两边都错、
两张图给同一答案，并检查两个 token-merge 正反例。通过只说明 evaluator 合同成立，不说明图像处理或模型推理已经运行。

2026-09-04 的最终验收固定 HF revision
`89644892e4d85e24eaac8bacfd4f463576704203`。因运行机无法访问 Hugging Face，权重从 Qwen 官方 ModelScope
仓库转移到仓库外临时目录；课程用官方 HF 权重 SHA256、10 个运行时关键文件逐文件哈希和 13 文件完整 manifest
三层复核，不把可变的 ModelScope `master` 冒充 immutable revision。两个独立离线进程各重复两轮 greedy inference：

- exit code 均为 0、stderr 均为空、8/8 系统 checks，稳定 digest 均为 `5ee6a7c212010936`；
- completion 1.0，normalized semantic / strict-format accuracy 均为 0.833；
- image-swap sensitivity 与 correctness 均为真；OCR 两次都只答 `CODE`，漏掉 `7319`；
- 每图 1,344 个 raw patches，经 merge size 2 得 336 visual tokens，跨进程账本一致；
- 峰值 allocated VRAM 4.044 GiB；约 0.070–0.071 s/case 只描述本机六个短输出，不是 serving benchmark。

完整环境、命令、原始回答和边界见 [L1 教程](tutorial_L1.md)。这组证据只支持“真实 2B checkpoint 的固定六例
诊断已经闭环”，不能外推自然图像 OCR、生产吞吐或更大 Qwen3-VL 的质量。

## 文件

- [L0_visual_tokens_to_language.py](L0_visual_tokens_to_language.py)：≤200 行单文件实验。
- [tutorial_L0.md](tutorial_L0.md)：推导、真实输出、反例与练习。
- [L1_qwen3_vl_real_probe.py](L1_qwen3_vl_real_probe.py)：真实 2B checkpoint 诊断脚本与可离线自测的评测合同。
- [tutorial_L1.md](tutorial_L1.md)：五类 estimand、token 账、L20 实测、复验命令和失败归因。
- [上级概念教程](../MODEL_ANATOMY_AND_TRAINING.md)：readout、encoder/VAE、参数口径与现代训练阶段。
- [Long Context / RAG 决策教程](../LONG_CONTEXT_OR_RAG.md)：把图像、视频和音频 token 账接到架构与成本选择。
- [上级研究账本](../RESEARCH.md)：经典谱系、当前模型身份与一手证据边界。
