# nano-rag-retrieval

> **抓的核心机制**：embedding 索引、向量检索、混合检索、重排序、检索评估（课程的数据系统教学约定）。
> L0 用纯 Python 裸出检索器的内核：**embedding = 把文本映进几何空间**（lexical 版：词哈希 + 字符 trigram，feature hashing）并让它的失败模式可见（同义词盲 / 词序盲 / 哈希碰撞假相似——神经 embedding 的动机）+ **flat 暴力精确 kNN = 一切 ANN 的基线**（Milvus 文档称 FLAT 为 Brute-Force）+ **检索评估 first-class**（recall@k / precision@k / MRR，8 条标注查询的 toy 基线）+ **治理 first-class**（索引字节 / 扫描 op 成本账本 + 分数阈值「不知道」语义）。
> **对应真实系统**：[Milvus](https://github.com/milvus-io/milvus) / [OpenSearch](https://github.com/opensearch-project/OpenSearch) / [Weaviate](https://github.com/weaviate/weaviate)；RAG 架构参照 Lewis et al. arXiv:[2005.11401]。
> **轨道**：[03 数据/分布式/RSI/数据平台工程](../README.md) · **状态**：L0–L2 ✅

---

## 为什么从「lexical embedding + 暴力精确 kNN」开始

LLM 有两个知识边界（训练截止 + 私有数据），RAG 是最直接的工程回答，而检索器是 RAG 的 R 半。L0 的选择是先看见**天花板**：lexical embedding 注定跨不过同义词鸿沟（实测 car vs automobile cosine = 0.0000），暴力精确 kNN 注定随 N 线性变慢（实测 + 算术外推）——两个「注定」分别定义了 L1（真实 embedding 模型）与 L2（ANN 索引）的存在理由。先有精确基线与可量化失败，突破才有参照系（tutorial §10 反例）。

---

## 阶梯（L0–L2）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | single-file 玩具（161 行，纯标准库）：lexical embedding（词 + trigram 哈希，md5 确定性）+ 四点光谱探针（1.0 / 0.2449 / 0.0000 / 0.2052）；flat 暴力精确 kNN（ANN 基线）；recall@k / precision@k / MRR 评估（toy 基线 recall@1 = 0.625 / MRR = 0.6875，含弱信号被噪声淹没与两条纯同义词 MISS 的逐查询分析）；成本账本（28672 B 索引 / 3584 ops/query）；分数阈值「不知道」语义；确定性 digest | ✅ `L0_vector_index_and_recall.py` + `tutorial_L0.md` |
| **L1** | 接真实小 embedding 模型（CPU 可跑，依赖显式声明 + 轻量 fallback）：同一索引与评估集上量化 lexical → 语义的 recall 突破（L0 两条 MISS = 验收靶点）+ 索引文件级持久化与增量 add + 词序/同义词失败模式的修复验证 | ✅ [代码](L1_semantic_embedding_and_persistence.py) · [教程](tutorial_L1.md) |
| **L2** | 对照权威实现源码做取舍分析：Milvus（FLAT/HNSW/IVF 索引族、segment 与 payload 分离）+ OpenSearch（k-NN + BM25 混合）+ Weaviate（HNSW 实现）；混合检索（BM25+dense 融合）+ 重排序 + recall/latency 权衡实测；可运行的本质模拟 + 显式注明 | ✅ [代码](L2_hybrid_search_hnsw_and_rerank.py) · [教程](tutorial_L2.md) |

**环境依赖分级**：L0 零依赖；L1 的 real 路径需要 sentence-transformers、torch 与钉住的模型快照，缺失时明确降级到手写 fallback（fallback 指标不代表模型质量）；L2 的 BM25/HNSW/融合/重排核心为 CPU 可运行实现，真实集群行为不在本模块证据范围内。L1 默认使用临时索引目录，不污染当前 CWD；设置 `NANO_RAG_INDEX_DIR` 才保留产物。

---

## L0 快速开始

```bash
python3 L0_vector_index_and_recall.py
```

预期输出（toy 指标基线）：embedding 光谱 `1.0000 / 0.2449 / 0.0000 / 0.2052`（完全一致 / 词形变化 / 同义词盲 / 哈希碰撞）；检索评估 `recall@1 = 0.625 / recall@3 = 0.750 / MRR = 0.688`（5 条 rank-1 + 1 条被碰撞噪声压到 rank 2 + 2 条纯同义词 MISS）；成本账本 `28672 B 索引 / 3584 ops/query`；retrieval digest `603eda71b56457f2`；`self-check: 13/13 PASS`。逐步拆解见 `tutorial_L0.md`。

---

## 费曼自检

- 能不能用「图书馆与坐标」一段话讲清 embedding / 暴力 kNN / 评估 / 阈值 / 成本各自的角色？（见 `tutorial_L0.md` §10）
- 「embedding 的质量决定检索的天花板」——哪个实测数字是天花板的直接证据？（0.0000：car vs automobile 正交。）
- 为什么 L0 坚持用注定失败的 lexical embedding，而不是直接上真实模型？

## 权威实现与延伸

- 对标源码（L2 展开）：milvus-io/milvus（FLAT/HNSW/IVF 索引族）、opensearch-project/OpenSearch（k-NN + BM25）、weaviate/weaviate（HNSW）；Milvus 文档 *Index Explained*（FLAT = Brute-Force 判据）
- 论文锚点：RAG [2005.11401] · feature hashing [0902.2206] · HNSW [1603.09320]
- 姊妹模块：[nano-data-platform](../nano-data-platform/)（语料 = 版本化快照，索引才可复现）· [nano-data-orchestration](../nano-data-orchestration/)（索引重建 = 一条 DAG 任务）
- 轨道：[03 数据/分布式/RSI/数据平台工程](../README.md)
