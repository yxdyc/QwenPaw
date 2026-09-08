# L1：用真实 Qwen3-VL checkpoint 区分“答对”与“看图”

> **状态：L1 已完成（2026-09-04）。** 评测器合同先通过 CPU 自测；最终脚本随后在单张 NVIDIA L20 上启动
> 两个独立离线进程，每个进程重复两轮 greedy inference。两次均 exit 0、stderr 为空、8/8 checks，
> 稳定 digest、答案与 token 账一致。

| 第一屏 | 内容 |
|---|---|
| 核心问题 | 一个真实 VLM 答对时，怎样判断它确实用了图像，并把格式错误、推理失败和不稳定分开？ |
| 先修 | 先完成 [L0 visual token 机制实验](tutorial_L0.md)，理解 patch、2D position 与 image-swap |
| 不变量 | 所有失败留在分母；模型与 revision 固定；语义、格式、反事实、稳定性、完成率分别报告 |
| 低成本入口 | `python3 -B L1_qwen3_vl_real_probe.py --self-test-metrics` |
| GPU 验收 | 已在单张 L20 上完成；同一本地快照启动两个独立进程，各跑两轮 greedy inference |
| 证据边界 | 真实 2B checkpoint + 六个合成诊断，不是自然图像 benchmark、生产吞吐或大模型质量结论 |

## 1. 第一性原理：一次回答至少混合五个问题

给定图像 $x$、问题 $q$、模型输出 $\hat y=f(x,q)$ 和期望答案 $y$，最容易犯的错误是把一切压成一个
`accuracy`。本实验拆成五个 estimand：

1. **语义正确 $C$**：规范化后是否等于期望答案；
2. **格式遵循 $F$**：原始输出是否逐字等于要求的短答案；
3. **视觉依赖 $D$**：保持问题不变、把图像从 $x_a$ 换成 $x_b$ 后，答案是否改变；
4. **重复稳定 $S$**：同一输入的两次 greedy 输出是否相同；
5. **执行完整 $R$**：预处理、forward、decode 是否真的结束，失败是否仍留在分母。

对应的最小定义是：

$$
C_i=\mathbf 1[\operatorname{norm}(\hat y_i)=\operatorname{norm}(y_i)],\qquad
F_i=\mathbf 1[\hat y_i=y_i]
$$

$$
D=\mathbf 1[\operatorname{norm}(f(x_a,q))\ne\operatorname{norm}(f(x_b,q))]
$$

但 $D=1$ 只说明**输出随图改变**，不说明两边正确。因此还要报告：

$$
D_{correct}=D\land C_a\land C_b
$$

| 现象 | 语义正确 | swap sensitivity | swap correctness | 应怎样解释 |
|---|---:|---:|---:|---|
| triangle→`TRIANGLE`，circle→`CIRCLE` | 是 | 是 | 是 | 这组反事实同时支持视觉依赖与正确性 |
| triangle→`CIRCLE`，circle→`TRIANGLE` | 否 | 是 | 否 | 模型看图后改变了输出，但映射错了 |
| 两张图都答 `TRIANGLE` | 至多一边 | 否 | 否 | 可能依赖语言先验或忽略了图像 |
| 输出 `The answer is TRIANGLE` | 规范化后仍可能不匹配 | 视另一图而定 | 视两边而定 | 还暴露了格式遵循问题 |

本质洞察是：**正确性是任务结果，反事实敏感性是证据依赖，completion 是系统可靠性。三者不是替代指标。**

## 2. 为什么只用六个诊断

脚本在临时目录确定性生成 768×448 RGB 图像，不下载数据：

| case | 改变的最小变量 | 期望答案 | 主要诊断 |
|---|---|---|---|
| `ocr_exact` | 黑白 bitmap 文本 | `CODE 7319` | OCR 与字符顺序 |
| `spatial_left` | 左红方、右蓝圆 | `RED SQUARE` | 颜色、形状与相对位置绑定 |
| `count_triangles` | 三个绿色三角形 | `3` | 计数 |
| `swap_triangle` | 单个红三角形 | `TRIANGLE` | 与下一例组成同问题反事实 |
| `swap_circle` | 单个蓝圆形 | `CIRCLE` | 与上一例组成同问题反事实 |
| `no_evidence_refusal` | 无文字灰图 | `NOT VISIBLE` | 证据不足时拒答 |

六例的 ROI 很高：它们能快速暴露 processor 接线、视觉依赖、格式和拒答问题；但样本太少且过于干净，不能估计
真实分布上的 OCR、grounding、计数或安全能力。这里的目标是让测量链先可信，不是制造一个好看的总分。

## 3. 从 image grid 到 LLM input：把 token 成本算清

官方 processor 返回 `image_grid_thw=(T,H,W)`。脚本先记录送入视觉 encoder 前的 patch 数：

$$
P_{raw}=T\times H\times W
$$

若 spatial merge size 为 $m$，进入语言序列的视觉 token 数应满足：

$$
N_{visual}=\frac{P_{raw}}{m^2}
$$

脚本同时记录：

- `raw_visual_patches`：merge 前 patch 数；
- `spatial_merge_size`：processor 的空间合并因子；
- `visual_tokens_after_merge`：按上式复算的视觉 token；
- `input_tokens`：包含图像占位、文本与特殊 token 的完整输入长度。

`input_tokens` 不应被简单写成“文本 token + $N_{visual}$”：chat template 和特殊 token 也占位置。真正需要稳定的是
同一 revision、同一输入的账本在独立进程间一致；因此首轮 token ledger 被纳入稳定 digest，而计时不进入 digest。

## 4. 先验证尺子，再测模型

运行：

```bash
python3 -B L1_qwen3_vl_real_probe.py --self-test-metrics
```

真实输出：

```text
METRIC_SELF_TEST_JSON={"checks":{"normalized_not_strict":true,"same_answer_not_sensitive":true,"sensitive_and_correct":true,"sensitive_but_wrong":true,"token_merge_exact":true,"token_merge_mismatch_detected":true},"evidence_boundary":"evaluator semantics only; no image processing or model inference","module":"nano-vlm-understanding/L1-metric-self-test","schema_version":"1.0"}
```

这个自测只使用标准库，验证四个评测逻辑反例和两个 token merge 账本；它不创建图像、不导入 PyTorch，
也不证明真实 checkpoint 能运行。

## 5. GPU 环境与运行方法

### 5.1 依赖合同

- Python 3.10+；
- 与机器 CUDA/driver 匹配的 PyTorch；
- Transformers 4.57.0+、Accelerate 与 Pillow；
- [Qwen3-VL 官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)；
- 固定 revision
  [`89644892e4d85e24eaac8bacfd4f463576704203`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/commit/89644892e4d85e24eaac8bacfd4f463576704203)。

不要从课程命令盲装 CUDA 版 PyTorch；先按当前 GPU、driver 和 CUDA 选择兼容 wheel/container，再记录实际版本。
[官方仓库示例](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md)使用
`AutoModelForImageTextToText`、`AutoProcessor` 与 `apply_chat_template`，脚本沿用该公开接口。

### 5.2 两阶段下载与复验

若运行机可访问 Hugging Face，首次运行允许下载，并把缓存放在仓库外的隔离目录：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/qwen3-vl-l1-cache \
python3 -B L1_qwen3_vl_real_probe.py --device cuda:0 --repeat 2
```

缓存完整后，启动两个**独立进程**复验；两次都加 `--local-files-only`，避免第二次暗中补文件或漂移 revision：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/qwen3-vl-l1-cache \
python3 -B L1_qwen3_vl_real_probe.py --device cuda:0 --repeat 2 --local-files-only
```

`repeat=2` 检查同一进程内的预测稳定；两个独立进程则额外检查加载、processor 和缓存边界。性能字段允许浮动，
但 `digest`、normalized answers、token ledger、requested/resolved revision 应一致。

本次 L20 机器无法访问 Hugging Face，但可访问 [Qwen 官方 ModelScope 仓库](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)。
因此使用 `modelscope-hub==0.4.0` 下载 13 文件快照，再从本地目录离线加载。`master` 是可变分支，不能冒充
HF commit；课程采用三重约束：

1. `model.safetensors` SHA256 必须等于 HF 固定 revision 官方值
   `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`；
2. 权重、模型配置、image/video processor、tokenizer、chat template 与 generation config 共 10 个运行时关键文件，
   均与 HF 固定 revision 逐文件 SHA256 相同；
3. 13 文件本地快照 manifest 固定为
   `b4f1f572206cd2e60255e7357166ee41bf2ffe3b8b52fa9da739370af895a99f`。

实际离线命令如下；路径只在运行机上解释，不进入 `RESULT_JSON`：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -B L1_qwen3_vl_real_probe.py \
  --model-path <LOCAL_MODEL_DIR> \
  --artifact-source qwen_modelscope \
  --artifact-source-revision master \
  --expected-weight-sha256 7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0 \
  --expected-manifest-sha256 b4f1f572206cd2e60255e7357166ee41bf2ffe3b8b52fa9da739370af895a99f \
  --device cuda:0 --repeat 2 --local-files-only
```

### 5.3 L20 实测环境与结果

| 项 | 实测值 |
|---|---|
| 物理 GPU | 8× NVIDIA L20；每次只暴露 GPU 0 |
| 单卡显存 / driver | 46,068 MiB / 550.90.07 |
| Python / CUDA | 3.12.4 / 12.8 |
| torch / torchvision | 2.9.1+cu128 / 0.24.1+cu128 |
| Transformers / Accelerate | 4.57.6 / 1.14.0 |
| artifact source | Qwen ModelScope `master` transfer；HF pinned artifact hashes 交叉核验 |
| artifact 验证 | 每遍 3.020–3.042 s；weight 与 13-file manifest 均匹配 |
| model load | 1.195–1.339 s |
| 峰值显存 | allocated 4.044 GiB；reserved 4.131 GiB |
| 两遍中位性能 | end-to-end 0.070–0.071 s/case；generation 0.062–0.064 s/case；47.539–48.136 generated tokens/s |

第一遍代表性输出：

```text
L1 Qwen3-VL real visual diagnostics
model=Qwen/Qwen3-VL-2B-Instruct revision_requested=89644892e4d85e24eaac8bacfd4f463576704203 revision_resolved=None load_mode=verified_local_snapshot
artifact_source=qwen_modelscope files=13 weight_sha256=7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0 manifest_sha256=b4f1f572206cd2e60255e7357166ee41bf2ffe3b8b52fa9da739370af895a99f
device=NVIDIA L20 torch=2.9.1+cu128 transformers=4.57.6
ocr_exact: answer='CODE' normalized_match=False strict_format=False raw_patches=1344 visual_tokens=336 input_tokens=361 e2e_s=0.419
spatial_left: answer='RED SQUARE' normalized_match=True strict_format=True raw_patches=1344 visual_tokens=336 input_tokens=363 e2e_s=0.073
count_triangles: answer='3' normalized_match=True strict_format=True raw_patches=1344 visual_tokens=336 input_tokens=358 e2e_s=0.058
swap_triangle: answer='TRIANGLE' normalized_match=True strict_format=True raw_patches=1344 visual_tokens=336 input_tokens=360 e2e_s=0.08
swap_circle: answer='CIRCLE' normalized_match=True strict_format=True raw_patches=1344 visual_tokens=336 input_tokens=360 e2e_s=0.067
no_evidence_refusal: answer='NOT VISIBLE' normalized_match=True strict_format=True raw_patches=1344 visual_tokens=336 input_tokens=363 e2e_s=0.079
metrics={"completion_rate": 1.0, "normalized_semantic_accuracy": 0.833, "prediction_stability": true, "skill_normalized_accuracy": {"count": 1.0, "image_swap": 1.0, "ocr": 0.0, "refusal": 1.0, "spatial": 1.0}, "strict_format_accuracy": 0.833, "swap_counterfactual_correct": true, "swap_counterfactual_sensitivity": true}
performance={"median_end_to_end_s": 0.071, "median_generation_s": 0.064, "median_tokens_per_s": 47.539}
checks=8/8 peak_vram_allocated_gib=4.044 load_s=1.339
```

程序化比较两次最终运行：

```text
FINAL_COMPARISON_JSON={"answers_and_token_ledgers_equal":true,"checks_all_true":true,"digest_equal":true,"digests":["5ee6a7c212010936","5ee6a7c212010936"],"metrics_equal":true,"peak_allocated_gib":[4.044,4.044],"stderr_bytes":[0,0]}
```

结果不是 6/6：两遍 OCR 都只回答 `CODE`，漏掉 `7319`；其余五例均正确，所以 normalized semantic 与
strict-format accuracy 都是 0.833，OCR skill 为 0.0。swap pair 两边均正确，sensitivity 与 correctness 都为真。
这证明实验能保留真实失败，而不是证明该模型 OCR 普遍只有 0%；后者需要字体、尺度、噪声和自然版面上的更大样本。

## 6. 验收不是“必须 100 分”

脚本只用系统合同决定退出码，不用准确率阈值隐藏模型缺陷。以下 checks 必须全部为 `true`：

- `all_attempts_counted`：恰有 $6\times repeat$ 次尝试；
- `all_completed`：没有预处理、forward 或 decode 失败；
- `all_answers_nonempty`：完成的回答非空；
- `repeat_coverage_complete`：每个 case 都有完整重复；
- `six_case_ids_present`：六个诊断都在；
- `model_artifact_verified`：本地权重和完整 snapshot manifest 均匹配预期 SHA；
- `visual_token_ledger_complete`：patch、merge 后视觉 token 与 input token 均可观测；
- `visual_token_merge_exact`：$P_{raw}=N_{visual}m^2$ 精确成立。

准确率低仍可以是一次**成功而有价值的实验**：它说明真实 2B checkpoint 在这个固定诊断上失败。反之，即使 6/6，
也不能据此声称模型在自然图像上可靠。课程验收关心“结果是否可信、失败是否可归因”，不是把模型调到满分。

## 7. 怎样读 `RESULT_JSON`

| 分区 | 是否应跨进程稳定 | 用途 |
|---|---|---|
| `metrics` | 是 | 语义、格式、swap、completion、重复稳定 |
| `checks` | 是 | 执行与 token 不变量；决定退出码 |
| `digest` | 是 | 覆盖 metrics、answers、revision 和首轮 token ledger |
| `evidence` | 版本/设备应稳定，加载时间可变 | GPU、CUDA、torch/vision、transformers、artifact source、revision、SHA 与峰值显存 |
| `performance` | 否 | 中位端到端/生成延迟与短输出 tokens/s，仅作本机描述 |
| `records` | 答案/token 应稳定，计时可变 | 保留每次原始回答和失败，不做成功样本筛选 |

短回答的 `tokens_per_s` 对调度开销很敏感，不能当成 serving benchmark；`peak_vram_allocated_gib` 与
`peak_vram_reserved_gib` 也只对应当前进程、输入尺寸和 PyTorch allocator。

## 8. 失败归因与下一步

| 观察 | 优先检查 | 不要立刻做什么 |
|---|---|---|
| Import/load 失败 | wheel/CUDA 兼容、缓存完整性、revision | 不要把“GPU 不行”当结论 |
| completion < 1 | 原始 error、OOM、processor 输入、decode | 不要从分母删除失败 case |
| semantic 高、strict 低 | prompt 格式遵循和原始回答 | 不要把 normalized accuracy 政名 exact match |
| sensitivity 高、correctness 低 | shape label 映射、prompt、模型错误 | 不要声称已经证明可靠看图 |
| token merge check 失败 | processor version、`image_grid_thw`、merge size | 不要继续比较吞吐 |
| 两进程 digest 不同 | revision、缓存、processor、答案和 token ledger | 不要先增加样本或升级 L2 |

只有真实 L1 闭环后，L2 才值得引入自然图像、动态分辨率、visual token budget、batching、DeepStack 与
interleaved MRoPE。完整技术谱系和证据边界见 [研究账本](../RESEARCH.md)。

## 9. 已形成与仍未形成的证据

本页已经形成真实 2B checkpoint 在固定六例上的 artifact、completion、答案、token、显存与单机延迟证据；两个独立
进程的稳定 digest 为 `5ee6a7c212010936`。它仍未形成自然图像 benchmark、不同分辨率 scaling、batch serving、
长视频理解、DeepStack 消融或 connector/LoRA 训练证据；这些属于 L2，不能由本次 12 次短回答推断。

## 10. 费曼自检：这次 GPU 实验究竟证明了什么

1. 为什么 OCR 错一例，反而比把输出规则放宽到 6/6 更有教学价值？
2. `visual_token_merge_exact=true` 与“模型正确看懂图像”之间缺了哪几层证据？
3. 两个进程的 digest 相同，能排除哪些问题，又排除不了哪些问题？

<details>
<summary>参考答案</summary>

1. 固定答案合同后保留 `CODE` 对 `CODE7319` 的失败，能区分语义/格式能力并暴露小字 OCR 短板；事后放宽规则会改变 estimand，使满分失去可解释性。单例失败仍只说明这组固定刺激上的行为，不能外推总体 OCR 水平。
2. merge 不变量只证明 processor 报告的 raw patch 与合并后 token 数在当前版本一致。还需要内容随换图正确改变、分技能准确、定位证据、自然分布样本和干预消融，才能逐步支持视觉依赖与理解能力。
3. 它能发现答案、token 账、模型 revision 或评估逻辑的非预期漂移；不能排除两个进程共享同一系统性错误，也不能证明其他 GPU、依赖版本、图片分布或长序列上可复现。计时和 allocator 状态本来就不应进入稳定 digest。

</details>
