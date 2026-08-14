# LLM-PBL 总路线图

> 四条轨道的依赖关系、阶梯定义、**educational-score 评分细则**、以及权威实现 / SOTA 参照表。
> 贡献者按此图推进，独立验证者按同一评分标准复核。
> **北极星**：把学习者训练到 senior LLM scientist & engineer（见 README「目标画像」）。

---

## 一、四轨依赖图

```
        ┌─────────────────────────────────────────────────────────────────┐
        │  03 数据 / 分布式 / RSI / 数据平台工程                              │
        │  data-juicer · ray · vllm-sglang · data-platform · orchestration  │
        │  （多源接入 → 湖仓治理 → 算子清洗 → 分布式调度 → 高吞吐采样）        │
        └───────────────────────────────┬─────────────────────────────────┘
                                        │ 数据 / rollout
              ┌─────────────────────────┴──────────────────────────┐
              ▼                                                    ▼
   ┌──────────────────────┐                           ┌──────────────────────┐
   │ 02 预训练 / CPT        │   base model              │ 01 后训练 / RL / SFT   │
   │ megatron · fsdp        │ ────────────────────────▶ │ trinity · slime ·     │
   │ （从 0 训 / 继续预训练） │                           │ verl · llamafactory   │
   └──────────────────────┘                           └───────────┬──────────┘
                                                                  │ 训好的模型
                                                                  ▼
                                                   ┌──────────────────────────┐
                                                   │ 04 LLM → Agent            │
                                                   │ agentscope · qwenpaw      │
                                                   │ （harness + 可靠执行）      │
                                                   └──────────────┬───────────┘
                                                                  │ agent 轨迹
                                                                  └──────▶ 回流 03（RSI 闭环）
```

**关键咬合点**：
- 03 的 `nano-vllm-sglang` 为 01 的 RL rollout 提供高吞吐采样；为 04 的 agent 提供推理后端。
- 01 训出的 policy 在 04 里被包装成 agent；agent 运行产生的轨迹（带 reward 信号）回流 03 做数据筛选。
- 02 提供 base model，是 01 的输入；CPT（continual pre-training）又依赖 03 的领域数据清洗。
- 03 内部的 `nano-data-platform` 与 `nano-data-orchestration` 是「数据飞轮」的基础设施层：多源接入、湖仓分层、权限/成本/稳定性治理，以及 DAG 编排与 CI/CD 自动化。它们不直接训练模型，但决定 data-model co-dev 能否在生产环境持续跑起来。

这个闭环就是 **data-model co-development（recursive self-improvement, RSI）** 的工程具象——它本身是 LLM 系统的核心命题，不依附于任何具体项目。

---

## 二、阶梯定义（每个 nano-* 内部的 L0–L3）

所有 nano-* 模块统一采用四级阶梯，**每级都能独立运行、独立验收**：

| 级别 | 名称 | 目标 | 验收标准 |
|------|------|------|----------|
| **L0** | 玩具实现 | 用最少代码抓住核心机制（≤200 行，单文件，CPU 可跑） | 能跑通一个 toy 例子，能口头讲清「它在模拟真实系统的哪一面」 |
| **L1** | 单卡可跑 | 接真实小模型 / 小数据，单 GPU 端到端 | 在真实小样本上产出正确结果，loss/指标曲线合理 |
| **L2** | 分布式 / 性能 | 引入并行 / 分片 / 高吞吐，触及工程难点 | 多卡或高并发下正确且更快（本机不具备时，允许「可运行的本质模拟 + 显式注明」，见 §三），能解释 scaling 行为 |
| **L3** | 对齐权威 / SOTA | 对照一个权威开源实现或 SOTA 系统的工程选择，复现其关键 trick | 能指出「我的 nano 版与权威实现的差异在哪、为什么它那样选」 |

**K+1 映射**：学习者应先定位自己当前所在的级（K），材料只推进到 K+1，不跳级。

---

## 三、Educational-Score 评分细则

每节材料按 5 个维度打分，各 0–5 分，满分 25。**低于 18 分打回重写。**

| 维度 | 5 分（优秀） | 3 分（及格） | 0 分（不合格） |
|------|-------------|-------------|---------------|
| **可跑性 (Runnability)** | 代码复制即跑，依赖/数据/命令齐全，贴有真实输出 | 能跑但需补环境，输出缺失 | 跑不通 / 伪代码冒充 |
| **阶梯清晰度 (Ladder)** | L0→L3 边界清楚，每级独立验收，K+1 平滑 | 有分级但跨级跳跃 | 一锅炖，无阶梯 |
| **机制深度 (Depth)** | 讲清「为什么这样设计」，对照权威/SOTA 有取舍分析，触及本质 | 只讲「是什么」 | 浮于 API 调用 |
| **费曼自检 (Feynman)** | 有「讲给外行」的类比 + 思考题 + 反例 | 有小结 | 无自检，读完仍讲不出 |
| **反幻觉 (Anti-hallucination)** | 数字/API/行数全部可溯源，不确定处标 `[TODO: verify]` | 个别未溯源 | 编造 benchmark / 虚构 API |

**附加硬性门槛（任一不满足直接打回，不论总分）**：
- 代码无法运行 / 依赖未声明
- 出现编造的数字、API、论文结论
- 用大段 mock 或假数据冒充「跑通」

**可运行性分级契约（2026-08-03 晚用户拍板：思想本质讲透 + 干净漂亮，可优先于严格跑通；反幻觉底线不变）**：
- **L0 / L1 必须可跑**（玩具级 / 单机级，这是 PBL 的核心，成本也低）；L1 若真实模型 / 数据过重，须提供小规模 fallback（随机初始化小模型 / 内置微样本），保证一键跑通。
- **L2 / L3 允许「可运行的本质模拟 + 显式注明」**：真实系统本机跑不了（多卡 / 集群 / 重引擎）时，模拟代码本身必须可运行、能演示核心机制，并明确写「此处模拟真实系统 Y 的机制 X，真实实现见 repo/path」；对照权威源码的分析仍是硬性要求。真机验证标 `[TODO: verify on real system]`。GPU/集群 smoke 由维护者在显式配置的环境中执行：先检查设备和占用，使用隔离目录，不干扰其他任务，并把可公开的软件/硬件条件写入质量报告；主机、用户名、端口和私有工作目录不得进入课程正文。
- **显式注明的 mock 任何级别可用**；若可跑版本会扭曲机制本质或显著损害干净漂亮，优先把思想讲透，但整节「纯伪代码、无任何可跑核心」仍不允许——每个 nano-* 至少有一个可跑锚点（L0 就是）。
- 硬门槛相应缩限解释：「代码无法运行」指**应可跑的部分**（L0/L1、模拟核心）跑不通；显式注明的模拟 / mock 不属于「冒充跑通」。

**senior 加成（独立验证者判断，用于区分 18 分与 25 分）**：
材料是否让人逼近 senior——即不仅「会跑」，还能读得懂权威实现源码、说得清 SOTA 为何这样选、判得出方法的边界与失败模式。

---

## 四、推进策略：广度优先轮转 + 级内 K+1

每次迭代只做一个 (模块, 级别) 并**做透**（保证深度、不留半成品），但“选哪个”按**广度优先**，让四条轨道齐头并进：

1. **先补短板**：优先选「还没有任何可跑内容」的轨道，给它开第一个模块的 L0。目标是让四条轨道**尽快都有 L0**，而不是把一轨做深做透再换下一轨。
2. **再均衡推进**：四轨都有 L0 后，选「最高完成级别最低 / 已覆盖模块最少」的轨道推进；同轨道内按 K+1 从 L0 往 L1、L2 走。
2.5. **L0 扫盲冻结**：只要还有核心模块没有任何 L0，就**冻结**已有模块的 L1+ 升级（收尾半成品除外），先把所有缺的 L0 按广度扫完——不允许「有的模块 degree K 还缺、却去写 K+2」。公开版以本文件的模块矩阵与各轨道 README 为准。
3. **每轨内部的模块顺序**（先 foundational）：
   - 01：nano-verl / nano-trinity-rft → nano-llamafactory → nano-slime → nano-opd（前沿算法层，时效性 B 层，见 §八）
   - 02：nano-pretraining-loop → nano-fsdp → nano-megatron
   - 03：nano-data-juicer → nano-ray → nano-vllm-sglang → nano-data-platform / nano-data-orchestration / nano-rag-retrieval（平台工程模块可并行，不阻塞核心 L0–L1 阶梯）
   - 04：nano-agentscope → nano-qwenpaw → nano-agent-runtime
4. **sota-deepdive**：对应 nano-* 到 L2 后才深挖（Kimi-K3 / DeepSeek / 后训练算法演进 PPO→GRPO/RLVR→OPD / 数据方法论 / harness engineering / 数据平台工程），避免悬空。

> 广度由「先补短板、均衡轮转」保证，深度由「每轮做透一级」保证——两者不冲突。用户随时应能看到每一轨都在前进。

---

## 五、权威实现与 SOTA 参照表

每个 nano-* 对标一个权威开源实现；L3 与 sota-deepdive 须回到这些**一手来源**（源码 / 技术报告），不得凭印象。

| 轨道 | nano-* | 权威开源实现（一手源码） | SOTA 深挖对象（一手报告） |
|------|--------|--------------------------|---------------------------|
| 01 | nano-verl | [verl](https://github.com/volcengine/verl)（HybridFlow） | Kimi-K3 技术报告 `[TODO: verify arXiv]`；GRPO 族 / RLVR `[TODO: verify arXiv]` |
| 01 | nano-slime | [slime](https://github.com/THUDM/slime) | 同上 |
| 01 | nano-trinity-rft | Trinity-RFT（开源仓库待核 `[TODO: verify repo]`） | 同上 |
| 01 | nano-llamafactory | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | DPO `[TODO: verify arXiv]` |
| 01 | nano-opd | 经典锚点（论文为一手来源）：MiniLLM `[2306.08543]` / GKD `[2306.13649]` / DistiLLM `[2402.03898]`；工程参考：verl / SWIFT 的 distillation 支持 `[TODO: verify]` | OPD Survey `[2604.00626]`；Qwen3 `[2505.09388]`；multi-teacher OPD（MAD-OPD / MOPD / Uni-OPD 等）`[TODO: verify arXiv]` |
| 02 | nano-pretraining-loop | L0 自包含训练生命周期；L1+ 对照 PyTorch/Megatron data loader、optimizer/scheduler 与 distributed checkpoint | 大规模预训练数据/恢复/稳定性报告 `[TODO: verify]` |
| 02 | nano-megatron | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | DeepSeek-V3 技术报告 `[TODO: verify arXiv]` |
| 02 | nano-fsdp | [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) / DeepSpeed ZeRO | 同上 |
| 03 | nano-data-juicer | [Data-Juicer](https://github.com/modelscope/data-juicer) | FineWeb / DCLM / Nemotron 数据报告 `[TODO: verify arXiv]` |
| 03 | nano-ray | [Ray](https://github.com/ray-project/ray) | 同上 |
| 03 | nano-vllm-sglang | [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | 同上 |
| 03 | nano-data-platform | [Apache Iceberg](https://github.com/apache/iceberg) / [Delta Lake](https://github.com/delta-io/delta) / [dbt](https://github.com/dbt-labs/dbt-core)；云原生参考 AWS Glue / Redshift / Snowflake / ClickHouse / Airbyte / Fivetran（权威实现需按实际版本核对）`[TODO: verify repo]` | 数据湖仓与 MLOps 平台最佳实践 `[TODO: verify]` |
| 03 | nano-data-orchestration | [Apache Airflow](https://github.com/apache/airflow) / [Dagster](https://github.com/dagster-io/dagster) / [Prefect](https://github.com/PrefectHQ/prefect)（工作流编排）；CI/CD 参考 GitHub Actions / GitLab CI `[TODO: verify repo]` | Agentic workflow / 自动化数据管线 `[TODO: verify]` |
| 03 | nano-rag-retrieval | [Milvus](https://github.com/milvus-io/milvus) / [OpenSearch](https://github.com/opensearch-project/OpenSearch) / [Weaviate](https://github.com/weaviate/weaviate) `[TODO: verify repo]` | RAG 检索系统与 embedding 索引 `[TODO: verify]` |
| 04 | nano-agentscope | [AgentScope](https://github.com/modelscope/agentscope) | harness / agent 评测 `[TODO: verify]` |
| 04 | nano-qwenpaw | qwenpaw（本仓库 `coach/`） | 同上 |
| 04 | nano-agent-runtime | 事务/outbox/idempotency 模式；L1+ 对照持久化 tool runtime `[TODO: verify repo]` | agent 副作用安全、恢复与审计 `[TODO: verify]` |

> 写材料时引用权威实现须给**真实 repo / 源码路径**；引用论文用 arXiv ID。拿不准就标 `[TODO: verify]`，绝不虚构。

---

## 六、约定

- **引用格式**：论文用 arXiv ID（如 `[2507.xxxxx]`）；权威实现源码用 `repo/path/file.py:Lxx`。
- **不确定标记**：`[TODO: verify]`（需独立验证的声明）、`[TODO: verify repo]`（需核对的仓库地址）。
- **语言**：中文叙述，技术术语保留英文。
- **代码优先**：能用代码说清的不用散文；每个 nano-* 至少一个 single-file 可跑实现。
- **本质优先**：先讲清机制的「为什么」，再谈 API 的「怎么用」；对标权威/SOTA 时做取舍分析而非功能罗列。

---

## 七、数据平台与 MLOps 工程扩展（2026-08-03 纳入规划）

数据平台 / MLOps / 数据工程主题纳入 03 轨道，作为核心阶梯之外的**深度扩展模块**。它们不替代 `nano-data-juicer → nano-ray → nano-vllm-sglang` 的核心路径，但在 L1/L2 之后提供工程落地视角。

### 新增 nano-* 模块

| nano-* | 覆盖关键词 | 抓的核心机制 | 对标权威实现 |
|--------|-----------|-------------|--------------|
| `nano-data-platform` | 多源数据接入；结构化/非结构化连接器；增量同步与全量回测；稳定性；湖仓架构 raw/curated zone；性能调优；鉴权；面向 AI 训练与检索供给；SQL；Redshift / Snowflake / ClickHouse；PySpark / AWS Glue；S3 / EC2；secrets manager / IAM / CloudWatch；Terraform HCL；Airbyte / Fivetran | 数据接入 → 分层存储 → 治理（成本/权限/质量）→ 训练/检索消费 | Iceberg / Delta Lake / dbt；云原生参考 AWS/Glue/Redshift/Snowflake/ClickHouse/Airbyte/Fivetran |
| `nano-data-orchestration` | Airflow；DAG；CI/CD；Agentic 能力 | 工作流编排、依赖调度、失败重试、自动化测试/部署、Agent 驱动的管线自愈 | Apache Airflow / Dagster / Prefect；GitHub Actions / GitLab CI |
| `nano-rag-retrieval` | RAG；OpenSearch / Milvus | embedding 索引、向量检索、混合检索、重排序、检索评估 | Milvus / OpenSearch / Weaviate |

### 与核心阶梯的关系

```
核心路径（必修）：
nano-data-juicer L0–L3 → nano-ray L0–L2 → nano-vllm-sglang L0–L2 → RSI 闭环专题 → sota-deepdive

平台工程扩展（选修/深度）：
nano-data-platform L0–L2（湖仓 + 接入 + 治理）
nano-data-orchestration L0–L2（DAG + CI/CD + Agentic）
nano-rag-retrieval L0–L2（向量检索 + RAG）
```

### 写作原则

- **不堆概念清单**：每个 nano-* 仍按 L0–L3 产出可跑代码，不允许只列技术名词。
- **云厂商实现作为参照，不锁定**：Redshift/Snowflake/ClickHouse/Glue/S3/IAM 等按 AWS / Snowflake / ClickHouse 官方文档与开源等价物（Iceberg/MinIO）对照讲解，避免材料变成单一云厂商说明书。
- **Terraform / HCL 只在 L2/L3 触及**：L0 用纯 Python 模拟 infra-as-code 的状态管理思想，L1 起才接触真实 HCL/provider。
- **安全与成本是 first-class 议题**：secrets manager、IAM least-privilege、CloudWatch 可观测性、存储/计算成本权衡，必须作为机制的一部分讲清，而非附录。
- **反幻觉门槛不变**：所有云厂商 API、价格、性能数字须给官方文档链接或实测；不确定标 `[TODO: verify]`，绝不凭印象写 "AWS Glue 比 Spark 快 X 倍" 这类不可溯源结论。

---

## 八、技术时效性策略：三层锚点 + 定期校准（2026-08-03 用户提出）

LLM 后训练演进很快（2023 PPO/DPO → 2024–25 GRPO/RLVR → 2025–26 OPD on-policy distillation）。材料须避免两种失败模式：**追新**（把中间状态的单论文方法当 SOTA 教）与**过时**（把经典方法当当前前沿教）。用户方针：**新版本/新论文 + 无可置疑的经典实现/经典论文搭配；中间状态要小心**。选题与写作按三层锚点执行：

| 层 | 定义 | 例子（截至 2026-08） | 材料中的处理 |
|----|------|---------------------|--------------|
| **A 经典锚点** | 无可置疑的经典论文与经典实现，机制仍是现代方法的地基 | PPO `[1707.06347]`、DPO `[2305.18290]`、MiniLLM `[2306.08543]`、GKD `[2306.13649]`；verl / Megatron / Ray / vLLM 等实现 | 长期保留，教机制本质；但**必须说明其当今定位**——如「PPO 已非前沿 RLVR 首选，但其 clipping / importance sampling 思想直接流入 GRPO」 |
| **B 前沿主流** | 新选题优先：近 12 个月被多个独立来源验证、或被前沿模型采用 | GRPO 族（DAPO / GSPO / CISPO）、RLVR、agentic RL、OPD / multi-teacher OPD、Qwen3 / Kimi-K3 的生产配方 | 须有一手技术报告支撑，不以单论文自称为准；写作前先做「SOTA 对齐」（见下） |
| **C 中间状态** | 新出现的单论文方法、变体爆炸（XPO 数十变体、2026 年 OPD 微创新论文上百篇） | 单来源的 XPO / OPD 变体 | **不单独立模块、不追**；提及时一律标 `[transient/单源]`，只教「机制类别」（如 token 级重加权这一类）不教个别方法名；晋升 B 层需 ≥2 个独立验证（前沿模型采用 / 权威框架集成 / 多机构复现） |

**执行机制**：

- **SOTA 对齐（写 B 层选题前必做）**：检索近 6 个月一手报告，确认是否有更新一代替代；材料中写明引用的 arXiv ID 与对齐日期；不确定标 `[TODO: verify]`。
- **时效性审计**：每轮检查材料引用的前沿方法是否已被取代，A 层材料是否说明了当今定位。
- **季度再校准**：每季度把 §五 参照表与最新格局对一遍，过期的 B 层条目降级 C 层或归档。
- **经典 ≠ 前沿**：教 PPO/DPO 本身没错（它们是 K 层），但不得让学习者误以为「前沿模型现在就是这么训的」。

## 九、公开维护与验证协议

本公开镜像只保留理解、运行和审计课程所需的信息；个人调度、逐轮写作日志、机器地址和
原始审查对话属于私有运维状态，不是课程前置条件。

1. **一次只推进一个层级**：每个变更明确对应一个模块和 L0–L3 层级，避免把多个机制、
   重构和证据更新混在一起。
2. **产出与验证分离**：重要机制至少由另一条运行路径、独立实现或失败注入复核；验证结论
   必须说明适用的源码快照、环境和未覆盖边界。
3. **不可把局部锚点升级成全量证明**：hash、固定 seed 和重复输出可证明快照一致性，不能
   证明外部系统、不同硬件或生产吞吐具有相同行为。
4. **正文与证据分层**：正文保留学习所需的不变量、公式、代表性输出和反例；完整输出、
   source snapshot 与环境摘要放入 evidence manifest 或质量报告。
5. **公开数据最小化**：只使用公开、合成或明确获准的数据；API key 从环境变量读取；不得
   提交用户名、私有主机、绝对工作目录、内部项目代号或协作账本。
6. **时效性复核**：涉及前沿方法、API 或源码行号时记录核验日期；无法重验的陈述继续保留
   `[TODO: verify]`，不得用旧缓存或二手摘要冒充当前事实。

当前覆盖和已知缺口见 [QUALITY-REPORT.md](QUALITY-REPORT.md)。
