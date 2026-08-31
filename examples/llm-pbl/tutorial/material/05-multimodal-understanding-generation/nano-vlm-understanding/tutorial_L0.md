# L0：从视觉 token 到语言答案

> 目标：不用神经网络框架，亲手跑通 `pixel grid → patch → projector → 2D position → packed tokens → causal readout`，
> 并证明答案不是只从问题文本猜出来的。

## 0. 先预测失败模式

一个模型即使图像全丢了也可能答对常识题，所以只报 VQA accuracy 不足以证明“看了图”。运行前先写下预测：

1. baseline 应在三个位置技能上全对；
2. image-drop 应显著伤害 exact match；
3. image-swap 应改变答案；
4. patch 内容被打乱、或二维位置被移除，都应伤害空间读取。

## 1. 图像怎样进入 causal token 流

3×3 灰度网格先变成 9 个 1×1 patch。固定 projector 为每个 patch 产生内容标量和 6 维位置 key：

```python
def patchify(image: list[list[int]]) -> list[dict]:
    """Turn a 3x3 pixel grid into 1x1 patches with explicit 2D coordinates."""
    return [
        {"value": float(value), "row": row, "col": col}
        for row, line in enumerate(image)
        for col, value in enumerate(line)
    ]
```

问题 token 放在 9 个图像 token 后面，所以最小 causal readout 只允许最后的问题查询前面的视觉 token。
对目标坐标 $(r,c)$，query 与 patch 的 row/column one-hot key 做点积，再经 softmax：

$$
a_i = \operatorname{softmax}_i(12 q^\top k_i),\qquad
\hat y = \operatorname{round}\left(\sum_i a_i v_i\right).
$$

这里的 12 只是让 toy attention 足够尖锐，不是从数据学习的温度。固定权重有意把“序列怎样接起来”与“能力怎样学出来”分开。

## 2. 四个反事实分别在问什么

| 干预 | 保留 | 破坏 | 能排除的捷径 |
|---|---|---|---|
| image-drop | 问题与位置 | 图像内容 | 只靠文本先验 |
| image-swap | 问题不变 | 当前图像证据 | 对图像不敏感 |
| patch-shuffle | token 数与位置槽 | 内容—位置对应 | bag-of-patches |
| remove-2D-position | 图像内容集合 | 空间索引 | 只看全局统计 |

注意：counterfactual sensitivity 高，只说明换图会改答案；若两次都错，它仍可能很高。因此必须与 exact match 分栏。

## 3. 运行

在本目录执行：

```bash
python3 -B L0_visual_tokens_to_language.py
```

真实输出：

```text
L0 visual tokens -> language
patches/image=9 packed_tokens/example=10
baseline skill EM={'top_left': 1.0, 'center': 1.0, 'bottom_right': 1.0}
image-drop mean EM=0.333
image-swap counterfactual sensitivity=1.000
patch-shuffle mean EM=0.333
remove-2D-position mean EM=0.333
checks=5/5
RESULT_JSON={"checks":{"baseline_all_correct":true,"drop_hurts":true,"no_2d_position_hurts":true,"patch_shuffle_hurts":true,"swap_changes_answer":true},"digest":"133e887a3d4b206c","evidence_boundary":"fixed projector/readout mechanism simulation; not a trained VLM","metrics":{"counterfactual_sensitivity":1.0,"image_dependence_gain":0.667,"skill_exact_match":{"bottom_right":1.0,"center":1.0,"top_left":1.0}},"module":"nano-vlm-understanding/L0","schema_version":"1.0"}
```

baseline 是 9/9；drop、shuffle、无位置各只剩 3/9。这里的 0.333 不是自然数据集上的估计，而是三个精心构造网格的确定性结果。

## 4. 从 toy 迁移到真实 VLM

真实系统会把固定 projector 换成视觉编码器与可训练 connector/resampler，把 one-hot 坐标换成 2D/多模态位置编码，
并在 multimodal pretraining/SFT 中学会 readout。但同一诊断仍成立：固定问题，交换图像；固定图像，扰乱空间；按 OCR、
计数、空间、grounding 分技能报分，不能让平均数藏住失败。

## 5. 动手题与边界

1. 把 softmax 温度 12 改为 1，解释为何 attention 泄漏到同行/同列 patch。
2. 新增“左上是否大于右下”问题：需要一次还是两次 readout？
3. 构造 image-swap 不改变答案的例子，说明 sensitivity 为何不能单独作为质量分。

**证据边界**：本实验没有训练、自然图像、OCR 字符、视觉 encoder 或生成式 LLM。它只证明这些合同和反例在一个
可检查的最小系统里按预期工作；不证明真实 VLM 的准确率、鲁棒性或 grounding 能力。
