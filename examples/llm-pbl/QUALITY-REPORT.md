# LLM-PBL 公开版质量报告

> 审计日期：2026-08-14
> 对象：`examples/llm-pbl/` 公开镜像
> 结论：可作为个人维护的公开 learning workflow / usage example；它不是 QwenPaw engine 插件，也不代表全部外部来源与生产硬件已经持续验证。

## 1. 发布边界

公开树只包含课程正文、导航、可运行代码、原创/可再分发 fixture、质量报告和摘要清单。
下列内容保留在私有作者工作区，不进入镜像：

- 写作进度、逐轮审查、调度/watchdog 与多人协作记录；
- 用户名、主机、端口、SSH 命令、私有工作目录与本机环境绝对路径；
- 本地数据集、运行日志、备份、缓存和临时文件；
- 真实 learner state、会话、凭据或 API key。

发布清单见 [`PUBLICATION-MANIFEST.sha256`](PUBLICATION-MANIFEST.sha256)。该文件列出除自身外每个公开文件的 SHA-256，可用于检查发布树是否被意外增删或改写。

## 2. 内容与教育价值

发布树共 152 个文件（含本报告与摘要清单）：

- 4 条主轨：后训练，预训练，数据/分布式/RSI，LLM→Agent；
- 17 个 `nano-*` 模块、2 个跨轨模块；
- 54 篇 `tutorial_L*.md`、57 个 Python 实现、5 篇非 README 的 SOTA deep dive；
- 统一的阶梯、可运行性契约、术语与评测约定。

当前材料最有价值的不是框架 API 罗列，而是反复复现同一组系统不变量：

1. **状态与边界**：mask、版本、watermark、checkpoint、消息 schema、cache ownership；
2. **失败关闭**：格式错误、缺 provenance、stale policy、错误凭据、越权路径和不完整回复均显式拒绝；
3. **代价可见**：显存、通信、重试调用、数据扫描、缓存命中和调度代价进入可量化账本；
4. **跨层闭环**：episode record 与 capability factory 把数据、训练、推理、agent 行为连接起来；
5. **证据边界**：toy、同构模拟、真实单机实现、外部源码锚和待真机项分开陈述。

综合判断：**教育价值高，核心系统知识覆盖较强，但不是完整的 LLM 教科书**。它更适合作为已有 Transformer/优化基础后的 systems practicum。仍值得优先补齐的核心面包括：

- 独立的 evaluation science 主线：污染/泄漏、paired eval、置信区间、回归门与长期漂移；
- serving 的量化、speculative decoding、真实 continuous batching 与端到端 SLO；
- 多机 GPU 的 topology-aware 实测与故障恢复，而不只是单机/同构模拟；
- agent security：prompt injection、工具权限、状态隔离与审计/回滚；
- 将各模块散落的 lineage、promotion、rollback、stopping 统一成一个受治理的 RSI 闭环。

更细的课程结构审计见 [`tutorial/material/CURRICULUM-AUDIT.md`](tutorial/material/CURRICULUM-AUDIT.md)。

## 3. 本次公开校准

- 把私有作者树蒸馏为 `examples/llm-pbl/`，不修改并行写作线维护的课程源文件；
- 删除协作轮次、内部状态、临时证据路径和维护者环境信息，仅保留可复核的技术结论；
- 命令统一为 `python3`，外部数据/源码通过 `--data`、`LLM_PBL_DATA_PATH`、`DATA_JUICER_REPO` 等显式参数传入；
- key 示例统一为 `CHANGEME` 或明确的测试占位符，不分发真实 endpoint credential；
- 将 phone fixture 从号码形态改为不可拨号的 `PHONE-DEMO-*`，并重算受影响的 digest/输出锚；
- 将 ReAct 摘要原文 fixture 改为原创释义，只保留论文标题与 arXiv ID；
- 修复公开目录多一层导致 nano-qwenpaw 错判 repo root 的可移植性问题；
- 修复 nano-qwenpaw L3 粗略 token 估算的整数取整断言（整段估算与分块估算允许相差 1）；
- 修复 nano-agentscope 的 OOV 边界：UNK 与 PAD 使用独立 special index，均从生成分布屏蔽；
- 同步 nano-agentscope L3 的跨级调用接口、loss/可靠性指标和教程输出；
- 在 coach 与课程之间增加双向定位，明确前者是 profile showcase，后者是完整 PBL 材料。

## 4. 2026-08-14 QA 证据

| 检查 | 结果 | 口径 |
|---|---:|---|
| Python 语法树 | 57/57 PASS | 对全部 `.py` 执行 `ast.parse`，不生成 pyc |
| Markdown fence | 92/92 PASS | 每个 Markdown 的 fenced-code 边界闭合 |
| 本地 Markdown 链接 | 369/369 PASS | 只检查仓库内相对路径；外部 URL 不计入 |
| 全部 L0 | 17/17 PASS | 从独立临时 CWD、Python 3.13.13 执行 |
| 非 L0 / 跨轨抽测 | 10/10 PASS | capability factory、episode record L0/L1、数据平台 L1、编排 L1、nano-qwenpaw L1/L2/L3、nano-agentscope L1/L3 |
| 总运行覆盖 | 27/57 PASS | 其余 30 个脚本未在本发布轮全部执行 |
| nano-data-platform 脱敏后 | 13/13 + 25/25 PASS | L0/L1 self-check；digest 与文档已同步 |
| nano-agentscope 脱敏后 | L1 + L3 PASS | torch 2.13.0 CPU；L1 含 loopback HTTP 契约；L3 120 行 self-check 输出 |
| 公开边界扫描 | PASS | 0 个维护者用户名/非 loopback 主机/绝对私有路径/真实邮箱或电话/高风险 key 形态命中 |
| 排除项泄漏 | PASS | 公开树中无私有进度、审查、watchdog、备份或缓存文件 |

允许且有意保留的安全相关字符串只有：API **环境变量名**、`CHANGEME`/测试占位符，以及 local contract test 使用的 `127.0.0.1`。这些不是凭据或外部地址。

本轮未使用 GPU，也没有调用真实模型 API。GPU/多机/真实 serving 结论仍以文中 `[TODO: verify on GPU]` / `[TODO: verify on real system]` 为准。

## 5. 已知限制与发布风险

1. **运行覆盖不是全量**：27/57 个脚本在本轮执行；其余脚本有的需要 Ray、多进程/GPU、外部数据、较长训练或真实 API。AST 通过不等于运行正确。
2. **外部证据是日期快照**：源码行号、main 分支 hash、论文/博客录值可能漂移。本轮没有重新抓取所有外部来源；材料保留抓取日期，读者应在重要使用前复核。
3. **硬件数字不可外推**：Apple Silicon CPU 或 toy 模型上的时间、吞吐和内存账不是生产 GPU benchmark。
4. **模拟边界仍在**：部分 L2/L3 是“可运行的本质模拟”，可证明控制流/不变量，不能证明真实框架的性能或故障行为。
5. **regex 扫描不是安全证明**：本次扫描显著降低误传风险，但不能替代 GitHub secret scanning、历史提交扫描和人工许可审查。
6. **第三方引用**：正文保留论文标题、短引文和链接；ReAct fixture 已改为原创释义。新增第三方语料前仍需单独核对许可，不能默认继承仓库 Apache-2.0。

## 6. 下一轮建议

按杠杆率排序：

1. 增加一个公开 `qa` 入口，把 AST、链接、隐私模式、全部 L0 与快速抽测变成 CI；
2. 将 30 个未运行脚本按 `fast-cpu / slow-cpu / gpu / network / external-data` 分层，逐层形成可重复矩阵；
3. 建立 evaluation/promotion gate 跨轨模块，用 paired candidate-parent 评测、hidden sentinels、lineage 和 rollback 把 RSI 从概念闭环升级为受治理闭环；
4. 在不泄漏机器信息的前提下补一次 GPU smoke：记录公开硬件型号、软件版本、命令、峰值显存与验收断言；
5. 对 SOTA deep dive 做一次来源刷新，把“历史录值仍可定位”和“当前 main 仍一致”分成两列。
