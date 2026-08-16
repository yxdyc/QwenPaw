# nano-agentscope

> **抓的核心机制**：**多 agent 编排**——消息契约、pipeline、工具调用，把单个 LLM 组成协作系统。
> **对应真实系统**：[AgentScope](https://github.com/agentscope-ai/agentscope)
> **轨道**：[04 LLM→Agent](../README.md) · **状态**：L0–L3 ✅（阶梯完成）

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | 玩具：single-file 写一个 ReAct 单 agent（思考→调工具→观察→再思考），用 mock LLM 跑通闭环 | ✅ [`L0_react_loop.py`](L0_react_loop.py) · [`tutorial_L0.md`](tutorial_L0.md) |
| **L1** | 把真模型换进循环：现场训练真实 char-level 小 LM（采样输出、可测违规率）+ 2 个真实工具（沙箱磁盘 I/O），harness 六条防御在故障注入下实测，可靠性代数（iid 公式 vs sticky 失效）+ 成本账；OpenAI-compatible API 客户端经本地契约服务器验证（真端点待 key） | ✅ [`L1_real_agent_loop.py`](L1_real_agent_loop.py) · [`tutorial_L1.md`](tutorial_L1.md) |
| **L2** | 多 agent：planner + executor 协作，消息契约与终止条件——六种消息类型在边界强制校验（类型化违规五种），五种终止状态全部带日志返回；实测摘要：89 行确定性输出 / happy path 恰 6 条验证消息 / 活锁守卫 6 次 attempt 精确诊断 vs 预算保险丝 24 次才烧断且误标。全部 agent 为声明的 rule-based 测试向量（按构造的协议失败），工具是真实磁盘 I/O | ✅ [`L2_planner_executor.py`](L2_planner_executor.py) · [`tutorial_L2.md`](tutorial_L2.md) |
| **L3** | 对照 AgentScope 的 message/pipeline 抽象，复现一个编排模式（双快照：v2.0.6 typed blocks + 构造期校验 + 消息级终止 / v1.0.0 MsgHub 广播 + SequentialPipeline 对照），并把真模型接回 planner 席（L1 配方重训，params/loss 逐位同一）；实测摘要：120 行确定性输出 / 贪心 happy path 3 次调用 6 条 crossing 15 次广播观察 / hub 以 0 次额外发送买到 verifier 全屋知识（p2p 需 4 次转发）/ 协议级重试代数 k=0 复合成立、k≥1 iid 为上界（72.5→76.5→76.5%）/ 提前完成只有第三角色抓得住（L2 两方协议接受同一向量）/ `exceed_max_iters` 住在消息上 | ✅ [`L3_typed_msghub.py`](L3_typed_msghub.py) · [`tutorial_L3.md`](tutorial_L3.md) |

## 环境依赖（分级）

- **L0**：零外部依赖（纯标准库），CPU 即跑。
- **L1**：`pip install torch`（CPU 即可，实测 torch 2.13.0 / Python 3.13）；训练约 2 分钟，
  全程确定性（seeded），总运行约 3 分钟。真实托管模型路径需
  `DASHSCOPE_API_KEY` / `OPENAI_API_KEY`（代码就绪，`[TODO: needs key]`）。
  数据文件 [`corpus.txt`](corpus.txt)（ReAct 论文 arXiv:2210.03629 标题与本仓库原创释义；
  论文元数据于 2026-08-06 在 arxiv.org 核验）。
- **L2**：零外部依赖（纯标准库），CPU 即跑，秒级（无训练）；无随机源、无计时行，
  跨运行输出逐字节一致（输出 md5 锚点见 tutorial_L2 §11）。
- **L3**：`pip install torch`（经 L1 import，CPU 即可）；训练约 2 分钟 + [4] 扫描约 1.5 分钟，
  总运行约 3-4 分钟；全程 seeded、无计时行，跨运行输出逐字节一致（输出 md5 锚点见
  tutorial_L3 §11）。真模型 planner 席由 L1 配方重训的 TinyReActLM 担任（确定性 fallback）；
  托管 planner 路径需 `DASHSCOPE_API_KEY` / `OPENAI_API_KEY`（代码就绪，`[TODO: needs key]`）。

## 核心要讲清的点

- ReAct loop：propose-observe 如何收敛，何时终止
- 工具调用：schema 注册、参数解析、结果回填
- **可靠性是流程的属性**（L1 实测）：真实模型是分布，违规率 p 可测且随温度变化；
  harness 的 critique-retry / 工具错误回流 / loop guard / 预算把 p 变成可定价的系统可靠性；
  相关（sticky）失败使「重试算术」失效
- 多 agent 消息契约：谁对谁说话、消息里带什么、如何避免死循环（L2）
- **契约类型化 + 编排即接线**（L3）：校验从「门口拦截」前移到「出生拦截」（角色×块类型
  合法表把自我授权/伪造证据挡在构造器外）；tool call 是带状态机的 block，agent 不能自推
  状态；广播 = 重接订阅表而非消息路由，verifier 的认知状态由接线决定；终止原因住在消息上
- **模型入席后的协议实测**（L3）：可靠性代数跨边界复合（k=0 成立），但 iid 是上界不是
  等式（sticky + 修复只在训练过的位置有效）；提前完成（无证据宣称）在野外可测（T=0.7
  step0 谱线 final:9/200）且只有第三角色抓得住

## 费曼自检

- 能不能解释「agent 的『可靠性』问题，和写普通函数调用，本质区别在哪」？（提示：动作有副作用、可能失败；且组件本身是随机的——L1 实测 p≈0.13@T=0.7）
- 能不能解释「为什么 critique-retry 只对听得懂批评的模型有效」？（L1 §5.2：critique 轨迹入训练集前后，retry 恢复能力的差异）
- 能不能解释「审计员在群里」比「审计员被逐个通知」强在哪——不只是方便，而是认知状态的
  差别？（L3 §7：best-effort 转发 vs by-construction 共享账本）

## 权威实现与延伸

- 对标源码：AgentScope `https://github.com/agentscope-ai/agentscope`（message / pipeline / 工具调用；原 `modelscope/agentscope` 已 301 重定向）。L1 已核验锚点（2026-08-06 main）：统一 `Agent` 的 reasoning-acting 循环（`agent/_agent.py:L858-874`）、`ReActConfig.max_iters=20` + `structured_output_grace_iters=5`（`agent/_config.py:L282-303`）、`_json_loads_with_repair`（`_utils/_common.py:L86`）、`Toolkit`（`tool/_toolkit.py:L66`）、`Msg`（`message/_base.py:L67`）——详见 tutorial_L1 §6。
- L2 已核验锚点（2026-08-10，main 已跃迁至 **v2.0.6**，行号取自当日 codeload tarball 双快照）：`Msg`（`message/_base.py:L67`）+ 构造期校验 `validate_role_content`（`message/_base.py:L117`）；终止即数据 `ReplyFinishedReason`（`types/_reply.py:L10-16`）；类型化内容块（`message/_block.py`：`TextBlock` L11 / `ToolCallState` L128 / `ToolCallBlock` L138 / `ToolResultState` L185 / `ToolResultBlock` L195）；`ReActConfig.max_iters=20`（L285-290）+ `structured_output_grace_iters=5`（L293-302）（`agent/_config.py:L282`）；reasoning-acting 循环 `while True` L863 + `match next_action` L869、`_next_action` L3050（`agent/_agent.py`——L1 记录的 L858-874/L3019 在 v2.0.6 漂移到 L863-869/L3050，机制不变）；v1.0.0 tag 编排原语 `MsgHub`（`pipeline/_msghub.py:L11`）/ `SequentialPipeline`（`pipeline/_class.py:L8`）/ `sequential_pipeline`（`pipeline/_functional.py:L7`）——v2.0.6 core 已整体移除 pipeline/msghub（全树 grep 零命中）而 Msg 契约加强，L3 对照因此须横跨 v1/v2 双快照——详见 tutorial_L2 §6。
- L3 已核验锚点（2026-08-10 18:3x 重新抓取 codeload 双 tarball）：main 7,525,108 B（较 00:36 快照 +158 B，`diff -rq` 定位漂移仅 `model/_gemini/_model.py` + 其测试，材料锚点零漂移，`__version__` 仍 2.0.6）；v1.0.0 tag 7,702,712 B 尺寸一致（tag 不可变）。新增锚点：**v1 广播机制**——`_subscribers`（`agent/_agent_base.py:L152`）/ reply 后 `_broadcast_to_subscribers`（L239-251）/ `reset_subscribers` 排除自己（L439-447）/ `MsgHub.broadcast` 含发送者（`pipeline/_msghub.py:L115-123`）；**v1→v2 契约演化**——v1 `Msg` 为普通类（`message/_message_base.py:L21`）、blocks 为 TypedDict 无 state 无运行时强制（`message/_message_block.py:L9/L79/L92`）→ v2 pydantic + 状态机 + 构造期校验；v2 按 role 的块合法断言 `_assert_user_content_blocks`（`message/_base.py:L33-39`，user 只许 text/data）；`ToolCallBlock.input` 为原始 JSON 字符串、状态迁移图在其 docstring（`message/_block.py:L138`）；v2 core 移除 pipeline/msghub 的复验结论仍成立（全树类名 grep 零命中）——详见 tutorial_L3 §6。
- 论文：ReAct arXiv:2210.03629（2026-08-06 核验，08-10 复验标题页）；Plan-and-Solve arXiv:2305.04091（L2 于 08-10 复验标题页，逐词一致）
- 概念延伸：可靠性 / 事务化执行见轨道 04「可靠性专题」
