# Benchmark Contract L0 — 96 分为什么可能输给 85 分

> **核心问题**：怎样把一张只有模型名和分数的榜单，恢复成可审计、可比较的测量合同？
> **先修**：理解 accuracy 和 `pass@k`；不需要统计学库。
> **不变量**：只有测量合同相同的 run 才能排序；缺字段必须 fail closed。
> **运行**：`python3 -B L0_benchmark_contract.py`；纯标准库、CPU、无网络。
> **验收**：9/9 checks；缺 `prompt_template` 的 99 分行被拒绝，naive valid winner 为 `model-d`，contract-aware winner 为 `model-b`。
> **边界**：所有模型名和分数均为 synthetic toy；脚本不验证任何真实模型或外部榜单。

---

## 1. 先看反直觉结果

```text
raw headlines (missing contracts are ineligible)
  model-invalid: 99.0 task_success_rate | INVALID: prompt_template
  model-d: 96.0 pass@8
  model-c: 91.0 task_success_rate
  model-f: 90.0 task_success_rate
  model-g: 89.0 task_success_rate
  model-e: 88.0 task_success_rate
  model-h: 87.0 task_success_rate
  model-b: 85.0 task_success_rate
  model-a: 82.0 task_success_rate
raw headline winner: model-invalid (99.0)
naive valid winner: model-d (96.0)

contract audit against model-a
  model-b: COMPARABLE | same contract
  model-c: NOT_COMPARABLE | tools
  model-d: NOT_COMPARABLE | samples_per_task, temperature, metric
  model-e: NOT_COMPARABLE | failure_policy
  model-f: NOT_COMPARABLE | prompt_template
  model-g: NOT_COMPARABLE | environment, verifier
  model-h: NOT_COMPARABLE | task_manifest
  model-invalid: INVALID | missing prompt_template
comparable winner: model-b (85.0, n=100)
illustrative 95% Wilson interval: [76.7, 90.7]
boundary: interval assumes independent binary tasks; agent tasks often violate this
PASS | missing contract fails closed
PASS | same contract remains comparable
PASS | prompt template is part of contract
PASS | environment and verifier are part of contract
PASS | task manifest is part of contract
PASS | tool augmentation breaks comparability
PASS | pass@k is not pass@1
PASS | failure denominator is part of score
PASS | contract-aware winner differs from naive winner
RESULT_JSON={"checks": {"passed": 9, "total": 9}, "digest": "cea6f541da87cb34", "evidence_boundary": "synthetic contracts only; no external benchmark score or model quality is validated", "metrics": {"comparable_to_baseline": 2, "contract_winner": "model-b", "invalid_runs": 1, "naive_valid_winner": "model-d", "raw_headline_winner": "model-invalid", "runs": 9, "valid_runs": 8}, "module": "benchmark-contract-l0", "schema_version": "1.0"}
```

99 分先因缺 `prompt_template` 被 fail closed，不能进入任何排序。96 分也没有“被降权”：它根本没有进入与
82/85 分相同的排序集合，因为 `model-d` 每题可试八次且 metric 是 `pass@8`。其余行分别改变工具、prompt、
环境/verifier、任务 manifest 或失败分母；它们可以各自是合法实验，却回答了不同问题。

## 2. 为什么要保存十八个合同字段

`BenchmarkRun` 不只保存 benchmark、model 和 score，还保存：

```python
CONTRACT_FIELDS = (
    "benchmark", "revision", "split", "task_manifest", "prompt_template",
    "harness", "environment", "tools",
    "reasoning_effort", "context_budget", "output_budget",
    "samples_per_task", "temperature", "timeout_seconds",
    "verifier", "judge", "failure_policy", "metric",
)
```

这些字段可以分成四层：

- **题目身份**：revision、split、task manifest；防止把不同版本或不同任务子集拼在一起；
- **系统能力**：prompt template、harness、environment、tools、effort 与预算；说明分数来自哪套完整执行系统；
- **随机与执行**：samples、temperature、timeout；决定搜索强度与完成机会；
- **裁决**：verifier、judge、failure policy、metric；决定什么算成功、失败是否还留在分母。

`contract_diff()` 采用最严格的相等比较。真实系统可以定义受控的兼容规则，例如只比较相同容器镜像的 patch revision；
但兼容规则本身也必须版本化，不能在看到结果后临时放宽。

## 3. 三个最常见的榜单错觉

### 3.1 加工具后仍写成“模型分数”

搜索、Python、终端和浏览器是有效能力放大器，但分数此时属于 $(model+harness+tools)$。比较裸模型与工具系统，
不能归因“模型推理更强”。正确做法是并列报告 no-tools / with-tools，或固定相同工具合同。

### 3.2 `pass@k` 冒充单次可靠率

若每次独立成功概率为 $p$，至少一次成功的理想化概率是：

$$
pass@k = 1-(1-p)^k.
$$

当 $p=0.7$、$k=8$ 时，这个值接近 100%；代价却是最多八倍采样与 verifier/search。现实样本还相关，不能用该式
反推出精确 `pass@1`。发布表必须同时报告 $k$、聚合器、总 token/时间/费用。

### 3.3 失败从分母消失

假设 100 题中 12 题 API error，剩余 88 题完成 77 题。`77/88=87.5%` 看起来优于 `77/100=77%`，但前者回答
“成功运行后有多准”，后者才接近用户看到的端到端可靠率。两者都可报告，不能只保留较好的一项。

## 4. 区间为什么也不能自动修复坏合同

脚本对 100 个二元任务的 85% 给出 Wilson 区间 `[76.7, 90.7]`。它只表达有限样本的不确定性，不会修复：

- 题目被训练数据污染；
- 两个模型使用不同 harness；
- repo/网站内任务相关而非 iid；
- judge 有系统偏差；
- 100 个 candidate 中只汇报最幸运的一个。

先固定 estimand 和执行合同，再估计不确定性；顺序不能反过来。

## 5. 动手改造

1. 把 `model-c` 的工具改回 `terminal-only`。它是否立刻可比较？检查其他字段。
2. 给 `model-b` 同时改成 20-task manifest 与 `tasks=20`，观察可比性和区间；解释任务身份与证据强度是两个问题。
3. 新增一行 `metric="partial_score"`。即使数值也是 85.0，为什么不能与 success rate 排序？
4. 再把一个 run 的 `verifier` 置空，确认它与缺 prompt 的 99 分一样进入 invalid 列表。

## 6. 费曼自检

1. 为什么脚本没有把 96 分“归一化”到 85 分的尺度上？
2. 同一个公开数据集、同一个 metric，为什么 harness 不同仍不能做模型归因？
3. 什么时候可以故意比较不同工具合同？应该怎样表述结论？
4. Wilson 区间覆盖什么不确定性，没覆盖什么？

<details>
<summary>参考答案</summary>

1. `pass@8` 与单次 task success 的 estimand 不同，没有无假设的通用换算；采样相关、verifier 和预算都会影响关系。最诚实的做法是分表，并额外报告成本—质量 Pareto。
2. prompt、harness、环境与 verifier 决定观察怎样压缩、工具怎样执行、错误怎样恢复和结果怎样裁决；任一项不同，差值都同时含模型与系统增益。
3. 当产品问题正是“哪套端到端系统更好”时可以比较；结论应写“系统 A 在固定成本/失败分母下完成率更高”，不能写成“底座模型 A 推理更强”。
4. 在独立二元任务假设下，它覆盖从有限题样本估计成功率的抽样不确定性；不覆盖污染、协议差异、任务聚类、judge bias、模型选择和环境漂移。

</details>

一句话验收：**不同合同的高分无需被打折；它们需要被分开命名。**
