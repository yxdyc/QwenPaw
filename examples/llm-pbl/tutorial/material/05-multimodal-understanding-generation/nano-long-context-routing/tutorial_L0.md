# L0：用证据形状裁决 Long Context、RAG 与 Hybrid

> 核心问题：回答所需证据是稀疏局部、时序多跳，还是要求穷尽覆盖？

> 先修：能运行 Python 3.10+，理解 top-k 检索；不需要向量数据库、GPU 或模型权重。

> 运行：`python3 -B L0_context_rag_hybrid.py`。

> 验收：能用 evidence recall 和输入 token 代理解释三个 selector 的胜负，并为失败选择下一步修复。

> 边界：本页使用 oracle readout 和词交集检索；不证明真实 tokenizer、embedding、attention 或 LLM 的质量。

## 1. 先判断必要证据会不会到齐

把回答所需证据集合记为 $E^*$，selector 实际交给模型的集合记为 $S$：

$$
\operatorname{evidence\ recall}=\frac{|E^*\cap S|}{|E^*|}.
$$

如果 recall 小于 1，最强的下游模型也缺少至少一项必要证据。L0 故意使用 oracle readout：只要 $E^*\subseteq S$
就判为“可答”，从而把 retrieval/selection failure 与模型推理 failure 分开。

## 2. 三个反例为什么互补

| Case | 证据形状 | 预期 |
|---|---|---|
| sparse clause | 一个可被关键词直接命中的合同条款 | RAG 用很少 token 找全 |
| temporal reversal | 承诺、安全审查、撤回分散在相邻会议片段 | top-k 漏掉指代片段，hybrid 用邻域补回 |
| exhaustive audit | 三个回滚项分布在整个变更记录 | top-k 和局部邻域都不保证“全部”，需要全局/map-reduce |

这三个 case 刻意不产生“万能赢家”。RAG 的瓶颈是 selection，whole context 的瓶颈是成本与有效利用，hybrid
只是折中，不会自动满足穷尽性。

## 3. 运行

在任意空目录执行绝对路径也应得到相同输出：

```bash
python3 -B /absolute/path/to/L0_context_rag_hybrid.py
```

真实输出：

```text
LONG-CONTEXT ROUTING L0
case=sparse_clause method=whole_context recall=1.000 correct=true tokens=115 selected=contract/cover,contract/sla,contract/privacy,contract/exit,meeting/0,meeting/1,meeting/2,meeting/3,meeting/4,repo/0,repo/1,repo/2,repo/3,repo/4
case=sparse_clause method=rag_top2 recall=1.000 correct=true tokens=18 selected=contract/exit,contract/cover
case=sparse_clause method=hybrid_neighbor recall=1.000 correct=true tokens=35 selected=contract/cover,contract/sla,contract/privacy,contract/exit
case=temporal_reversal method=whole_context recall=1.000 correct=true tokens=115 selected=contract/cover,contract/sla,contract/privacy,contract/exit,meeting/0,meeting/1,meeting/2,meeting/3,meeting/4,repo/0,repo/1,repo/2,repo/3,repo/4
case=temporal_reversal method=rag_top2 recall=0.667 correct=false tokens=18 selected=meeting/1,meeting/3
case=temporal_reversal method=hybrid_neighbor recall=1.000 correct=true tokens=43 selected=meeting/0,meeting/1,meeting/2,meeting/3,meeting/4
case=exhaustive_audit method=whole_context recall=1.000 correct=true tokens=115 selected=contract/cover,contract/sla,contract/privacy,contract/exit,meeting/0,meeting/1,meeting/2,meeting/3,meeting/4,repo/0,repo/1,repo/2,repo/3,repo/4
case=exhaustive_audit method=rag_top2 recall=0.667 correct=false tokens=16 selected=repo/0,repo/2
case=exhaustive_audit method=hybrid_neighbor recall=0.667 correct=false tokens=30 selected=repo/0,repo/1,repo/2,repo/3
RESULT_JSON={"checks":{"global_view_needed_for_exhaustive_case":true,"hybrid_recovers_temporal_chain":true,"sparse_rag_saves_tokens":true},"digest":"075f9caac9b1783d","evidence_boundary":"selection surrogate only; oracle readout, no tokenizer, attention, retrieval model, or LLM","metrics":{"hybrid_neighbor":{"accuracy":0.666667,"mean_evidence_recall":0.888889,"mean_input_tokens_surrogate":36.0},"rag_top2":{"accuracy":0.333333,"mean_evidence_recall":0.777778,"mean_input_tokens_surrogate":17.333},"whole_context":{"accuracy":1.0,"mean_evidence_recall":1.0,"mean_input_tokens_surrogate":115.0}},"module":"nano-long-context-routing/L0","schema_version":"1.0"}
```

## 4. 怎样升级成真实 L1

保持同一问题、语料和输出 schema，再逐项替换 surrogate：

1. 用模型 tokenizer 替代词数成本；
2. 用 BM25/embedding/reranker 替代词交集；
3. 让真实 LLM 输出 answer + cited chunk IDs，不再使用 oracle readout；
4. 对 whole context 改变必要证据的位置，测 lost-in-the-middle；
5. 同时报 correctness、evidence recall、citation、TTFT、decode latency、峰值显存与计费；
6. OOM、timeout、parser failure 全部留在 completion 分母。

只有这样才能区分：“证据没召回”“证据到了但模型没找到”“找到但不会联立推理”与“答案正确但成本不可接受”。

## 5. 费曼自检

1. 为什么 L0 使用 oracle readout，反而让实验结论更干净？
2. `evidence_recall=1` 为什么仍不能证明答案正确？
3. hybrid 为什么能修复会议反例，却可能漏掉穷尽性审计？
4. 若 whole context 在三例都正确，为什么不能宣布它是生产默认方案？
5. 怎样验证 hybrid 的收益来自选择质量，而非仅仅输入更多 token？

<details>
<summary>参考答案</summary>

1. 它固定下游推理为“证据齐全即成功”，所以失败只能归因于 selector；若一开始接真实 LLM，召回、位置利用和推理错误会纠缠。
2. recall 只证明必要证据在输入中。模型可能忽略中间位置、误解否定、无法跨片段联立，或输出无依据结论。
3. 邻域扩展能补回命中片段周围的指代和时间前后文；穷尽任务的证据可能散落在多个远端区域，局部邻域没有覆盖保证。
4. L0 没有测长输入的 TTFT、显存、计费、吞吐和 lost-in-the-middle；真实语料也可能超过物理窗口。正确上限不等于经济最优。
5. 固定最终输入 token 预算，比较随机/截断、top-k 与 top-k+neighbor，同时报告 evidence recall 和 answer correctness。

</details>

## 6. 完成标志

- 三个 checks 全为 `true`；
- 两次 fresh-CWD 运行 stdout 逐字节一致、stderr 为空；
- 能为每个 case 指出 selection bottleneck；
- 不把 oracle correctness 写成真实模型能力。
