# LLM-PBL 课程完备性与质量审计

> 快照日期：2026-09-03
> 审计范围：`tutorial/material` 的结构、阅读路径、发布卫生与证据分层。
> 重要边界：本页不等于“全部实验已重跑”或“全部外部主张已重新联网核验”。

## 结论先行

LLM-PBL 最有价值的教学模式已经稳定：把一个抽象机制压缩成可运行、可失败、可量化的最小实验，
再逐级接到真实 tensor、分布式运行或真实系统。01–04 轨已有一批完整 L0–L3 纵深，05 多模态轨也补齐了
“图文理解 → Image DiT → Video DiT → MiniMax H3”四个 L0 机制锚。

下一阶段的最高收益不再是继续增加并列的综述或 toy，而是闭合三类证据缺口：

1. 把仍停在机制层的模块接到真实小模型、持久化状态、跨进程恢复或真机执行；
2. 把现有单机 TP2/4/8 collective 证据继续接到端到端 workload，并等待完整本地模型后再测真实推理引擎；
3. 对正文中的计划标记和来源账逐项分级，优先消除会阻断结论的证据债，而不是机械清零所有标记。

## 1. 当前课程快照

| 维度 | 当前数量 | 口径 |
|---|---:|---|
| 主轨 | 5 | 01 后训练、02 预训练、03 数据/分布式/RSI、04 Agent、05 多模态 |
| `nano-*` 模块 | 20 | 递归目录名计数；H3 capstone 不在该命名口径内 |
| 跨轨模块 | 3 | Capability Factory、EpisodeRecord、Evaluation Gate |
| deep-dive 目录 | 4 | 01–04 各一处；05 以 `RESEARCH.md` 承担研究账本 |
| Markdown | 117 | `tutorial/material` 全树 |
| `tutorial_L*.md` | 75 | 包含 Evaluation Gate 的敏感性补充教程 |
| Python | 84 | 课程材料树内脚本，不含仓库级校验器 |

数量只说明“内容存在”，不说明“生产可用”。本课程继续使用三层证据口径：

- **机制证据**：标准库或小型模拟证明控制流、账本、反例和不变量；
- **实现证据**：真实框架、小模型或持久化组件实际运行；
- **系统证据**：固定硬件、依赖与 revision 后的 GPU/多进程/故障注入结果。

低一层证据不能自动外推为高一层能力。

## 2. 覆盖矩阵：哪里已经深，哪里仍然浅

| 轨道 | 已形成纵深 | 当前最重要缺口 | 下一步高 ROI |
|---|---|---|---|
| 01 后训练 | 5 个核心模块均有 L0–L3；覆盖 SFT、PPO/RLVR、rollout、RFT、OPD 与 Kimi K3 案例 | 多教师路由与生产 recipe 已有教学锚，但真实 teacher 服务、隐藏评估和端到端成本证据仍有限 | 用固定小模型做一次 multi-teacher 对照，并把失败率、成本和增益同表报告 |
| 02 预训练 | FSDP、Megatron 已到 L3；pretraining lifecycle 已到两进程 gloo exact resume L2；Megatron 已有 PP2 与 TP2/4/8 L20/NCCL 证据 | 复制式 DP 与 FSDP/Megatron checkpoint schema 尚未闭合；TP scaling 仍是 toy collective 而非端到端训练 | 把 rank-local 状态合同迁移到真实分片 manifest，再以实际层形状测 compute/communication overlap |
| 03 数据/分布式/RSI | Data-Juicer、Ray、vLLM/SGLang 到 L3；平台、编排、RAG 到 L2 | 跨组件 schema 演进、离线/在线一致性和真实引擎证据仍分散 | 用一条 EpisodeRecord 贯穿 snapshot → retrieval → rollout → admission，并补真实 SGLang 固定提示集 |
| 04 Agent | AgentScope、QwenPaw 到 L3；transactional runtime L2 已覆盖多 worker、outbox、compensation 与 provider-checked fencing epoch | fencing 仍是单机 SQLite 机制证据；网络分区、真实 token 与外部 runtime 尚未实证 | 以 HTTP mock/真实 runtime 注入 stale owner、响应丢失、权限重放与补偿失败 |
| 05 多模态 | VLM、Image DiT、Video DiT、H3 四个 L0 可独立学习 | 目前全是机制层；真实小模型、真实图像/视频质量和 H3 配置账尚未落地 | 依次完成 Qwen3-VL 小样本、微型 DiT 训练、moving-video DiT、H3 config-only 复算 |

跨轨部分已经承担“系统闭环”而非补充阅读：

- [EpisodeRecord](cross-track-episode-record/) 统一 PPO、GRPO、OPD、工具轨迹与 provenance 的数据契约；
- [Capability Factory](cross-track-capability-factory/) 产生可追溯 candidate；
- [Evaluation Gate](cross-track-evaluation-gate/) 用配对证据、隐藏 sentinel、回滚与激活日志约束晋升。

这里最值得继续做的是把三者接到同一条可恢复的端到端实验，而不是再复制一套概念定义。

## 3. 六个质量维度

### 3.1 Educational value

强项是“反例可运行”：错误 padding、陈旧 policy、错误 teacher routing、重复副作用、丢失 modality tag、
错误 flow scheduler 等都能让检查确定性失败。这比只展示 happy path 更接近真实工程判断。

需要改进的是跨级验收的一致性。每一级应明确回答：

1. 本级新增了哪一种真实约束？
2. 哪个指标或断言证明该约束被满足？
3. 哪个结论仍然不能由本级证据推出？

### 3.2 易读性与 simple-but-deep

多数教程已经包含问题、公式、运行输出和边界，但第一次阅读仍可能被长证据表、哈希和来源摘录打断。
后续应把每篇开头收敛为六项：核心问题、先修、不变量、运行、验收、边界；完整来源账和长输出放在后半部。

“简单”应来自更少的状态变量和更清楚的因果对照，不是删掉关键假设；“深入”应来自失败模式和可证伪性，
不是增加名词密度。

### 3.3 材料组织

[学习总导航](README.md) 已把五轨组织成“数据 → 训练/生成 → 评估 → Agent → 反馈”的闭环。
仍需避免两类漂移：

- 文件已经存在，但模块 README 没有把它纳入正式阶梯；
- README 标为完成，但教程没有 fresh-CWD、输出同步或真实依赖边界。

因此，**模块 README 是发布状态的唯一入口**；孤立脚本不能据此算作某一级完成。

### 3.4 SOTA 覆盖

课程不追求列出所有新模型，而是追踪可迁移的方法谱系。当前已覆盖现代 RLVR/OPD、分布式训练与推理、
data/RSI governance、Agent runtime，以及 VLM、rectified-flow DiT、Video DiT 和 H3 packed omni flow。

05 轨的 [研究账本](05-multimodal-understanding-generation/RESEARCH.md) 特别需要保持三种事实分离：
论文/官方模型卡声明、公开源码实现、课程推断。开放权重也不能写成整个托管系统全部开源。

### 3.5 学习 ROI 与冗余

优先保留能改变决策的材料：数据/状态契约、错误对照、成本账、恢复边界和 promotion gate。以下新增内容收益较低：

- 再写一篇只做模型列表的综述；
- 为同一机制复制第二个没有新反例的 toy；
- 用单个 endpoint score 或单个自动 judge 代替配对评估与人工 rubric；
- 把硬件计时写成与环境无关的定理。

deep-dive 的第一屏应先给“解决什么、代价什么、何时不用、证据多强”的决策表，再进入来源密集的纵深。

### 3.6 证据质量

当前材料包含 357 个常用计划标记**关键字出现次数**，分布在 102 个文件、351 个源文本行；
它们不是 357 个独立缺陷，也不能按非零枚举简单等同为 357 个任务。按轨道的出现次数为：01=86、02=53、
03=140、04=78、05=0。

建议逐项标为四类：

| 类型 | 是否阻断发布 | 处理方式 |
|---|---|---|
| claim-blocking | 是 | 未核验数字、API 或源码语义不得进入事实表 |
| real-system debt | 视声明而定 | 保留明确边界，排入真实框架/GPU/故障实验 |
| source-refresh | 通常否 | 固定 revision；仅在相关教程发布或版本变化时刷新 |
| planned-level | 否 | 移入模块路线图，避免混在正文事实中 |

优先清偿 claim-blocking，而不是为了得到“零标记”去删除诚实边界。

## 4. 当前发布与真机证据边界

四个补充 GPU 探针已纳入课程：pretraining lifecycle L1、Megatron L1/L2 与 SGLang L2/L3；
后三者使用显式 CLI 参数、失败即停和 `RESULT_JSON`。Pretraining L1 已于 2026-09-03
在单张 L20 独立运行两次，均 5/5 且稳定输出一致。Megatron PP2 已在单机 2×L20 完成两次复验；TP probe 又于 2026-09-03 在同机完成
TP2/4/8 各两次、共六次 7/7 复验。稳定 digest、数值误差、显存账、collective timing 与拓扑边界见
[nano-megatron](02-pretraining-cpt/nano-megatron/README.md)。
这构成单机 TP scaling 的正确性、状态账与固定消息 collective 证据，不构成端到端训练 speedup 或多机证据。

真机记录至少应包含：

- GPU 型号/数量、driver、CUDA、Python、框架与模型 revision；
- 命令、seed、输入规模、退出码、正确性 checks；
- 吞吐、峰值显存、通信/缓存命中指标及其测量口径；
- 失败日志的公开摘要，不包含本机路径、凭据或内部工作流元数据。

SGLang 探针还必须使用可公开复现的本地模型，并同时报告完成 token 数和 matched prompt token budget；
不能用请求的最大 token 数冒充实际吞吐，也不能用长度悬殊的 prompt 声称证明 prefix cache 收益。
本轮机器只有 SGLang wheel 与不完整的模型仓库元数据，缺少完整本地权重，因此按合同不运行、不补数。

## 5. 接下来两轮的优先级

### P0：闭合而不是扩张

1. 把 `nano-pretraining-loop` L2 的 rank-local 合同接到 FSDP/Megatron manifest，补半写、world-size 与版本不兼容反例。
2. 把 `nano-agent-runtime` L2 已验证的 fencing 合同迁到 HTTP/真实 provider，加入响应丢失与网络分区。
3. 获得完整、固定 revision 的小模型后再运行 SGLang；Megatron 后续只在加入真实层形状、overlap 或多机变量时继续真机实验。

### P1：让证据债可管理

1. 使用仓库级 [材料校验器](../../scripts/validate_material.py) 固定 AST、Markdown fence、相对链接、Git 可见性、敏感信息和产物卫生检查。
2. 为计划标记增加四类标签与 owner-free 的处理状态；不要恢复内部协作流程元数据。
3. 对当前导航中的新增教程做两次空 CWD、`python -B`、stderr 为空和输出同步验收。

### P2：推进 05 轨真实层

按依赖顺序推进：Qwen3-VL 小样本推理 → 微型 rectified-flow Image DiT → moving-video DiT → H3 config-only。
H3 大权重真机实验必须单独核验许可证、磁盘、依赖与 revision；本地 768p 不能写成托管 2K 系统复现。

## 6. 本次审计如何复验

从仓库根目录运行：

```bash
python3 -B scripts/validate_material.py
```

校验器只依赖 Python 标准库与仓库已有的 Git，最后输出稳定的 `RESULT_JSON=`。当前检查覆盖 Python AST、
Markdown fence、代码块外的真实相对文件链接及其 Git 可见性、公开内容中的本机/内部元数据，以及
`__pycache__`/`.pyc` 等产物。
它会报告计划标记数量但不因此失败。

仍需人工或专项实验完成的部分包括：84 个脚本的全量运行、外部引文逐项刷新、GPU 真机结果、视觉质量盲评，
以及“自动指标能否支持结论”的构念效度判断。静态全绿只是发布必要条件，不是课程正确性的充分条件。
