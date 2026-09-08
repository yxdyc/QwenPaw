# LLM-PBL 课程完备性与质量审计

> 快照日期：2026-09-08
> 审计范围：`tutorial/material` 的结构、阅读路径、发布卫生与证据分层。
> 重要边界：本页不等于“全部实验已重跑”或“全部外部主张已重新联网核验”。

## 结论先行

LLM-PBL 最有价值的教学模式已经稳定：把一个抽象机制压缩成可运行、可失败、可量化的最小实验，
再逐级接到真实 tensor、分布式运行或真实系统。01–04 轨已有一批完整 L0–L3 纵深，05 多模态轨也补齐了
“图文理解 → Image DiT → Video DiT → MiniMax H3”四个 L0 机制锚，并已把图文理解推进到真实 Qwen3-VL-2B L1。
跨轨 Frontier Model Lifecycle 又把 DeepSeek-V4、Qwen3.8-Flash-Next、Kimi K3、GLM-5.3/Flash 按
architecture → pretraining → post-training → serving → evaluation 串联，并把 GPT-6 Astra、Claude Fable 5.1
放进闭源 API 的可观测合同。课程由此同时覆盖 stage 血缘、增益归因和路由 provenance，而不需要猜测未公开的模型内部。

下一阶段的最高收益不再是继续增加并列的综述或 toy，而是闭合三类证据缺口：

1. 把仍停在机制层的模块接到真实小模型、持久化状态、跨进程恢复或真机执行；
2. 把现有单机 TP2/4/8 collective 证据继续接到端到端 workload，并等待完整本地模型后再测真实推理引擎；
3. 对正文中的计划标记和来源账逐项分级，优先消除会阻断结论的证据债，而不是机械清零所有标记。

## 1. 当前课程快照

| 维度 | 当前数量 | 口径 |
|---|---:|---|
| 主轨 | 5 | 01 后训练、02 预训练、03 数据/分布式/RSI、04 Agent、05 多模态 |
| `nano-*` 模块 | 20 | 递归目录名计数；H3 capstone 不在该命名口径内 |
| 跨轨模块 | 4 | Capability Factory、EpisodeRecord、Evaluation Gate、Frontier Model Lifecycle |
| deep-dive 目录 | 4 | 01–04 各一处；05 以 `RESEARCH.md` 承担研究账本 |
| Markdown | 121 | `tutorial/material` 全树 |
| `tutorial_L*.md` | 77 | 包含 Evaluation Gate 的敏感性补充教程、VLM L1 真机教程与 Frontier Lifecycle L0 |
| Python | 85 | 课程材料树内脚本，不含仓库级校验器 |

数量只说明“内容存在”，不说明“生产可用”。本课程继续使用三层证据口径：

- **机制证据**：标准库或小型模拟证明控制流、账本、反例和不变量；
- **实现证据**：真实框架、小模型或持久化组件实际运行；
- **系统证据**：固定硬件、依赖与 revision 后的 GPU/多进程/故障注入结果。

低一层证据不能自动外推为高一层能力。

## 2. 覆盖矩阵：哪里已经深，哪里仍然浅

| 轨道 | 已形成纵深 | 当前最重要缺口 | 下一步高 ROI |
|---|---|---|---|
| 01 后训练 | 5 个核心模块均有 L0–L3；覆盖 SFT、PPO/RLVR、rollout、RFT、OPD 与 Kimi K3；Frontier Lifecycle 已接入 V4 multi-teacher OPD 和 GLM same-base attribution | 真实 teacher 服务、同底座 paired replay、隐藏评估和端到端成本证据仍有限 | 先固定 model/stage metadata，再用小模型做 multi-teacher 与 same-base 对照 |
| 02 预训练 | FSDP、Megatron 已到 L3；pretraining lifecycle 到 gloo exact resume L2；DeepSeek/Qwen/Kimi/GLM 已有跨代机制/血缘地图；Megatron 有 PP2 与 TP2/4/8 L20/NCCL 证据 | hybrid sparse/linear attention、Muon、mHC、QAT 仍缺独立实验；分片 checkpoint schema 尚未闭合 | metadata ledger 后，每次只做一个小型 architecture/optimizer factorial |
| 03 数据/分布式/RSI | Data-Juicer、Ray、vLLM/SGLang 到 L3；平台、编排、RAG 到 L2 | 跨组件 schema 演进、离线/在线一致性和真实引擎证据仍分散 | 用一条 EpisodeRecord 贯穿 snapshot → retrieval → rollout → admission，并补真实 SGLang 固定提示集 |
| 04 Agent | AgentScope、QwenPaw 到 L3；transactional runtime L2 已覆盖多 worker、outbox、compensation 与 provider-checked fencing epoch | fencing 仍是单机 SQLite 机制证据；网络分区、真实 token 与外部 runtime 尚未实证 | 以 HTTP mock/真实 runtime 注入 stale owner、响应丢失、权限重放与补偿失败 |
| 05 多模态 | 四个 L0 可独立学习；Qwen3-VL-2B L1 已有单张 L20、双独立进程的真实 checkpoint 证据 | VLM 仍只有六例 synthetic diagnostics；真实图像/视频质量和 H3 配置账尚未落地 | 训练微型 Image DiT，再做 moving-video DiT 与 H3 config-only 复算 |

跨轨部分已经承担“系统闭环”而非补充阅读：

- [EpisodeRecord](cross-track-episode-record/) 统一 PPO、GRPO、OPD、工具轨迹与 provenance 的数据契约；
- [Capability Factory](cross-track-capability-factory/) 产生可追溯 candidate；
- [Evaluation Gate](cross-track-evaluation-gate/) 用配对证据、隐藏 sentinel、回滚与激活日志约束晋升。
- [Frontier Model Lifecycle](cross-track-frontier-model-lifecycle/) 把 model identity、parent、stage、参数口径与开放边界放进同一 claim contract。

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

本轮加入非阻断的 [风格审计器](../../scripts/audit_tutorial_style.py)，扫描 77 篇 `tutorial_L*.md` 的首屏语义字段和两类惯用句。
首批改写 22 篇高优先级教程后，首屏覆盖从“问题 44 / 先修 41 / 运行 54 / 验收 38 / 边界 37”提升到
“问题 61 / 先修 62 / 运行 63 / 验收 60 / 边界 55”；固定“不是…而是…”句式由 83 处降到 68 处，
正文双破折号信号由 2326 降到 2282。这些计数只用于发现入口过密或信息缺位，不能替代人工判断；
合理的逻辑对照、长推导和表格不会因为命中规则就自动成为坏文风。

本轮也把“费曼自检”从结尾问题清单升级为可自学的反馈回路：77 篇教程全部含自检，77 个自检段均给出显式参考答案。
学习者应先独立作答再展开对照；答案重点解释因果链、反例和不可外推项，不要求背诵固定措辞。
[费曼覆盖审计器](../../scripts/audit_feynman_answers.py) 会阻断“整篇缺自检”或“有问题无答案”的回归，
但覆盖率不能证明答案正确、解释充分，也不能证明学习者真的先思考过。

### 3.3 材料组织

[学习总导航](README.md) 已把五轨组织成“数据 → 训练/生成 → 评估 → Agent → 反馈”的闭环。
仍需避免两类漂移：

- 文件已经存在，但模块 README 没有把它纳入正式阶梯；
- README 标为完成，但教程没有 fresh-CWD、输出同步或真实依赖边界。

因此，**模块 README 是发布状态的唯一入口**；孤立脚本不能据此算作某一级完成。

### 3.4 SOTA 覆盖

课程不追求列出所有新模型，而是追踪可迁移的方法谱系。当前已覆盖现代 RLVR/OPD、分布式训练与推理、
data/RSI governance、Agent runtime，以及 VLM、rectified-flow DiT、Video DiT 和 H3 packed omni flow。
Frontier Model Lifecycle 进一步用开放权重模型建立“架构候选三轴门—base/midtrain—specialist/agentic RL—OPD—QAT/serve—paired eval”主线；
GPT-6 Astra 与 Claude Fable 5.1 则提供闭源对照：只教学官方 API 行为、路由/fallback、工具轨迹、缓存和成本回执，
参数量、架构与训练 recipe 继续记为未知。官方声明、源码事实、未披露项和课程推断分栏，避免用厂商榜单替代机制证据。

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

当前材料包含 341 个常用计划标记**关键字出现次数**，分布在 102 个文件；
它们不是 341 个独立缺陷，也不能按非零枚举简单等同为 341 个任务。按轨道的出现次数为：01=77、02=48、
03=138、04=78、05=0。

建议逐项标为四类：

| 类型 | 是否阻断发布 | 处理方式 |
|---|---|---|
| claim-blocking | 是 | 未核验数字、API 或源码语义不得进入事实表 |
| real-system debt | 视声明而定 | 保留明确边界，排入真实框架/GPU/故障实验 |
| source-refresh | 通常否 | 固定 revision；仅在相关教程发布或版本变化时刷新 |
| planned-level | 否 | 移入模块路线图，避免混在正文事实中 |

优先清偿 claim-blocking，而不是为了得到“零标记”去删除诚实边界。

### 3.7 第一性原理与学习 ROI

课程推进的最小单位不是“再多一个文件”，而是**消除一个会改变判断的未知量**。可用下面的序关系排队：

$$
\text{priority}\ \propto\
\frac{\text{evidence gap}\times\text{decision impact}\times\text{cross-track reuse}}
{\text{GPU/data cost}+\text{engineering cost}+\text{maintenance burden}}
$$

这不是可比较到小数点的评分公式，而是强迫维护者回答五个问题：当前缺的到底是哪层证据；失败会不会改变路线；
结果能否被多个模块复用；最便宜的反证是什么；新依赖和版本债由谁承担。按当前快照，推荐顺序是：

| 候选批次 | 先做它能消除什么未知量 | 成本/风险 | 当前判断 |
|---|---|---|---|
| Frontier Lifecycle L0 | 模型昵称、参数口径、stage、API 开放边界和跨版本比较是否可审计 | 纯标准库；不提供质量证据 | **已闭合**；7 张模型卡、12/12 claim checks |
| Frontier Lifecycle L1 metadata | 官方 config/card 是否支持参数、序列、显存、license 与 parent 复算 | 不下载大权重；有版本维护债 | **当前最高 ROI** |
| Qwen3-VL-2B 六例 L1 | L0 的视觉依赖反事实能否迁移到真实 processor/checkpoint；语义、格式、敏感性与 completion 能否分开量 | 单卡、小合成集；不能外推自然图像 | **已闭合**；真实 OCR 失败已留在证据中 |
| 微型 rectified-flow Image DiT L1 | oracle velocity 的方向/条件合同能否迁移到真实优化，并在 held-out condition 上学习 | 单卡小时内；合成数据不代表真实画质 | **05 轨下一项最高 ROI** |
| checkpoint manifest bridge | exact resume 合同能否跨 FSDP/Megatron 分片、半写和 world-size 变化成立 | 多进程与格式维护成本中等 | 高复用，紧随其后 |
| Agent HTTP/provider 故障注入 | SQLite fencing 在响应丢失、stale owner 和权限重放下是否仍守住副作用 | 需要 mock/真实 provider 双层边界 | 高决策价值 |
| H3 config-only 复算 | 33B packed sequence、video/audio latent 与显存账能否由公开 metadata 重算 | 不需大权重，但需固定版本和事实审计 | 先于 H3 权重 |
| H3/Video 大权重 smoke | 真实生成能否在固定硬件上完成最小合同 | 下载、GPU、人工评测和许可证成本高 | 前置 gate 未闭合前不抢跑 |
| 新增并列模型综述 | 主要增加名词覆盖 | 时效债高、决策增量低 | 暂缓 |

每个教程也应用同一原则：第一屏只保留核心问题、先修、不变量、运行、验收、边界；长输出、哈希与来源账后置。
这样做不是降低深度，而是让读者先获得可运行的因果骨架，再按需要支付细节成本。

## 4. 当前发布与真机证据边界

补充 GPU 探针已覆盖 pretraining lifecycle L1、Megatron L1/L2、SGLang L2/L3 与 Qwen3-VL L1；
它们使用显式 CLI 参数、失败即停和 `RESULT_JSON`。Pretraining L1 已于 2026-09-03
在单张 L20 独立运行两次，均 5/5 且稳定输出一致。Megatron PP2 已在单机 2×L20 完成两次复验；TP probe 又于 2026-09-03 在同机完成
TP2/4/8 各两次、共六次 7/7 复验。稳定 digest、数值误差、显存账、collective timing 与拓扑边界见
[nano-megatron](02-pretraining-cpt/nano-megatron/README.md)。
这构成单机 TP scaling 的正确性、状态账与固定消息 collective 证据，不构成端到端训练 speedup 或多机证据。

Qwen3-VL L1 已于 2026-09-04 在单张 L20 启动两个独立离线进程，每个进程重复两轮；均 exit 0、stderr 为空、
8/8 checks，稳定 digest `5ee6a7c212010936`，峰值 allocated VRAM 4.044 GiB。六例准确率 0.833 来自明确的
OCR 漏数字失败，image-swap sensitivity/correctness 均为真。该结果只对应固定 synthetic diagnostics，不构成
自然图像 OCR、生产吞吐或完整 Qwen3-VL 系列能力证据。

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

1. 完成 Frontier Lifecycle L1：固定七个模型身份的官方 metadata revision、license、parent/stage 与 API 合同；只对公开 config 复算参数、序列和最小显存账，不下载 frontier 权重，也不补写闭源参数。
2. 把 Image DiT L0 的 oracle velocity 改成可训练的微型 rectified-flow DiT；固定训练预算、held-out condition、
   reconstruction/condition metrics 与错误符号反例，不把合成图形升级成真实画质声明。
3. 把 `nano-pretraining-loop` L2 的 rank-local 合同接到 FSDP/Megatron manifest，补半写、world-size 与版本不兼容反例。
4. 把 `nano-agent-runtime` L2 已验证的 fencing 合同迁到 HTTP/真实 provider，加入响应丢失与网络分区。
5. 获得完整、固定 revision 的小模型后再运行 SGLang；Megatron 后续只在加入真实层形状、overlap 或多机变量时继续真机实验。

### P1：让证据债可管理

1. 使用仓库级 [材料校验器](../../scripts/validate_material.py) 固定 AST、Markdown fence、相对链接、Git 可见性、敏感信息和产物卫生检查。
2. 为计划标记增加四类标签与 owner-free 的处理状态；不要恢复内部协作流程元数据。
3. 对当前导航中的新增教程做两次空 CWD、`python -B`、stderr 为空和输出同步验收。

### P2：推进 05 轨真实层

按依赖顺序推进：微型 rectified-flow Image DiT → moving-video DiT → H3 config-only。Qwen3-VL L1 已完成，下一次
VLM 扩展应直接进入自然图像、动态分辨率和 token-budget 对照，而不是继续增加同分布合成样本。
H3 大权重真机实验必须单独核验许可证、磁盘、依赖与 revision；本地 768p 不能写成托管 2K 系统复现。

## 6. 本次审计如何复验

从仓库根目录运行：

```bash
python3 -B scripts/validate_material.py
python3 -B scripts/audit_tutorial_style.py
python3 -B scripts/audit_feynman_answers.py
```

校验器只依赖 Python 标准库与仓库已有的 Git，最后输出稳定的 `RESULT_JSON=`。当前检查覆盖 Python AST、
Markdown fence、代码块外的真实相对文件链接及其 Git 可见性、公开内容中的本机/内部元数据，以及
`__pycache__`/`.pyc` 等产物。
它会报告计划标记数量但不因此失败。

风格审计器同样只依赖标准库，并输出稳定的 `RESULT_JSON=`。它只检查首屏是否容易找到问题、先修、运行、验收和边界，
以及少数容易滥用的句式；它是编辑提示，不是发布门禁或写作质量分数。

费曼覆盖审计器要求每篇 `tutorial_L*.md` 至少有一段费曼自检，且每段都含 `参考答案`、`参考讲法` 或 `答案要点` 标记。
它验证反馈回路没有结构性缺口，不对答案的事实性、教学深度或学习效果背书。

仍需人工或专项实验完成的部分包括：85 个脚本的全量运行、外部引文逐项刷新、其余模块的 GPU 真机结果、视觉质量盲评，
以及“自动指标能否支持结论”的构念效度判断。静态全绿只是发布必要条件，不是课程正确性的充分条件。
