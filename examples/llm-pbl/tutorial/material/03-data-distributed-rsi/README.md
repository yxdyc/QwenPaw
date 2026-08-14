# 轨道 03 — 数据 / 分布式处理 / 递归自改进（RSI）/ 数据平台工程

> **一句话**：LLM 的「数据飞轮」——多源接入 → 湖仓治理 → rule-based 与 llm-based 算子清洗 → 分布式调度跑大规模 → 高吞吐推理引擎做采样，最终让 data 与 model 相互迭代（recursive self-improvement）。
> **对标权威实现**：Data-Juicer · Ray · vLLM / SGLang · Iceberg / Delta Lake / Airflow · Milvus / OpenSearch ｜ **SOTA 参照**：LLM 数据方法论 + data-model co-dev + 湖仓 / MLOps

---

## 这条线学什么

数据是 LLM 的燃料，但「数据处理」本身是个系统工程：
- **算子（OP）化**：把清洗/过滤/去重/打分抽象成可组合算子（rule-based 快而确定，llm-based 强而贵）。
- **分布式**：用 Ray 把算子调度到集群，吞吐线性扩展。
- **高吞吐推理**：vLLM/SGLang 让 llm-based 算子和 RL rollout 的采样成本可接受。
- **数据平台工程**：多源连接器、湖仓分层（raw/curated）、SQL 分析、权限/成本/稳定性治理、DAG 编排、CI/CD、RAG 检索供给。
- **RSI 闭环**：模型产出数据 → 数据筛选 → 训练更强模型，data-model co-development。

| nano-* | 抓的核心机制 | 对标权威实现 |
|--------|-------------|--------------|
| `nano-data-juicer` | OP 抽象（mapper/filter/dedup/aggregator）+ 配置驱动 pipeline | Data-Juicer |
| `nano-ray` | task/actor 模型、分布式调度、object store | Ray |
| `nano-vllm-sglang` | PagedAttention / continuous batching / 高吞吐采样 | vLLM / SGLang |
| `nano-data-platform` | 多源接入 → 湖仓分层 → 治理（权限/成本/质量）→ 训练/检索消费 | Iceberg / Delta Lake / dbt；云原生参考 AWS Glue/Redshift/Snowflake/ClickHouse/Airbyte/Fivetran |
| `nano-data-orchestration` | DAG 编排、依赖调度、失败重试、CI/CD、Agentic 管线自愈 | Apache Airflow / Dagster / Prefect；GitHub Actions / GitLab CI |
| `nano-rag-retrieval` | embedding 索引、向量检索、混合检索、重排序、检索评估 | Milvus / OpenSearch / Weaviate |

---

## 学习路径（K+1 阶梯）

```
前置：会 pandas 数据处理、懂 LLM 推理基本流程（K）
  │
  ▼
Step 1  nano-data-juicer L0–L1        ← 写一个 filter/mapper OP，跑通配置驱动 pipeline
  │
  ▼
Step 2  nano-ray L0–L2                ← 把 OP 分布式化，理解 task/actor 与 object store
  │
  ▼
Step 3  nano-vllm-sglang L0–L2        ← llm-based OP 的吞吐引擎，PagedAttention 直觉
  │
  ▼
Step 4  RSI 闭环专题 L2–L3             ← data-model co-dev，数据路由 / 配比 / 自改进
  │
  ▼
Step 5  sota-deepdive                 ← SOTA LLM 数据方法论 + 核心工程技术
  │
  ▼
扩展（可并行，不阻塞核心路径）
  ├── nano-data-platform L0–L2        ← 湖仓 / 多源接入 / 治理 / MLOps infra
  ├── nano-data-orchestration L0–L2   ← DAG / CI/CD / Agentic workflow
  └── nano-rag-retrieval L0–L2        ← 向量检索 / RAG
```

---

## 完成标志

- [ ] 能写一个自定义 Data-Juicer 风格 OP（rule-based + llm-based 各一），配置驱动跑通
- [ ] 能用 Ray 把单进程 pipeline 改成分布式，解释 object store 为何避免数据拷贝
- [ ] 能解释 PagedAttention / continuous batching 如何提升推理吞吐
- [ ] 能画出 data-model co-dev 闭环，说清「数据如何回流改进模型」
- [ ] 能讲清「数据路由 / 选择」（按信号给不同数据分配训练权重）在 co-dev 闭环中扮演什么角色
- [ ] 能设计 raw/curated 两层湖仓 schema，说明增量同步与全量回测的取舍
- [ ] 能用 Airflow 风格 DAG 编排一个数据管线，并说明失败重试与 CI/CD 测试策略
- [ ] 能用向量数据库实现 RAG 检索链路，并解释 hybrid search / rerank 的必要性

---

## 权威实现与 SOTA 参照

写材料须回到一手来源（源码 / 技术报告），拿不准标 `[TODO: verify]`：
- Data-Juicer：`github.com/modelscope/data-juicer`（本地参考：`${DATA_JUICER_REPO}`）
- Ray：`github.com/ray-project/ray`（task/actor、object store）
- vLLM：`github.com/vllm-project/vllm`（PagedAttention）；SGLang：`github.com/sgl-project/sglang`（RadixAttention）
- 数据湖仓：Apache Iceberg `github.com/apache/iceberg`、Delta Lake `github.com/delta-io/delta`、dbt `github.com/dbt-labs/dbt-core`
- 数据接入：Airbyte `github.com/airbytehq/airbyte`、Fivetran（闭源，文档为准）`[TODO: verify]`
- 数据仓库/分析：Snowflake（闭源）、ClickHouse `github.com/ClickHouse/ClickHouse`、AWS Redshift/Glue/S3/IAM/CloudWatch（官方文档）`[TODO: verify]`
- 编排：Apache Airflow `github.com/apache/airflow`、Dagster `github.com/dagster-io/dagster`、Prefect `github.com/PrefectHQ/prefect`
- 向量检索：Milvus `github.com/milvus-io/milvus`、OpenSearch `github.com/opensearch-project/OpenSearch`、Weaviate `github.com/weaviate/weaviate`
- SOTA：代表性数据方法论报告（FineWeb / DCLM / Nemotron 等）`[TODO: verify arXiv]`；RSI / self-improvement 代表工作 `[TODO: verify]`；湖仓与 MLOps 平台最佳实践 `[TODO: verify]`

→ 深挖见 [sota-deepdive/](sota-deepdive/)
