# nano-long-context-routing

这个模块不先问“模型窗口有多长”，而是先问“回答所需证据长什么样”。L0 用同一语料和问题对比：

- whole context：证据覆盖高，但输入成本最大；
- top-k RAG：适合稀疏局部证据，但 retrieval miss 后无法靠 readout 补救；
- hybrid：为命中项补 parent/neighbor/time-window，修复部分切块与时序断裂。

## 立即运行

```bash
python3 -B L0_context_rag_hybrid.py
```

仅依赖 Python 3.10+ 标准库，CPU、离线、确定性运行。完整推导和真实输出见 [L0 教程](tutorial_L0.md)。

## 量化合同

- `evidence_recall`：回答必需证据进入上下文的比例；
- `correct_if_oracle_readout`：只有证据全部到齐才算可答，用来隔离 selection failure；
- `input_tokens_surrogate`：简单词数成本，只用于三种 selector 的相对比较；
- 三个固定反例：稀疏条款、跨时序撤回、穷尽性回滚审计。

这里的 readout 是 oracle；脚本没有真实 tokenizer、embedding、attention 或 LLM。因此它能证明证据选择合同和反例，
不能证明某个模型在 512K/1M 内能正确推理，也不能给出真实延迟、显存或价格。

## 阶梯

| 级别 | 项目 | 新增约束 | 状态 |
|---|---|---|---|
| L0 | 确定性 selector surrogate | selection recall、邻域扩展、成本代理 | 已完成 |
| L1 | 小模型 + 真实 tokenizer | 位置敏感性、答案/引用、TTFT 与 token 账 | 规划中 |
| L2 | 真实多模态资料 | page/shot/speaker/symbol 索引与长视频、多页文档 | 规划中 |
| L3 | 生产路由 | 权限、新鲜度、cache、成本回执、失败重检索 | 规划中 |

## 延伸阅读

- [Long Context 还是 RAG](../LONG_CONTEXT_OR_RAG.md)：容量、有效利用、多模态 token 账与生产决策。
- [Qwen3-VL L1](../nano-vlm-understanding/tutorial_L1.md)：真实视觉 token ledger。
- [03 轨 nano-rag-retrieval](../../03-data-distributed-rsi/nano-rag-retrieval/)：检索、索引和 provenance 的纵深实现。
