# nano-agentscope L2 — planner + executor：消息契约与终止条件

> **级别**：L2（K+1：L1 的单 agent + 真模型 + harness → L2 的多 agent 协议层）
> **文件**：[`L2_planner_executor.py`](L2_planner_executor.py)（608 行，零外部依赖——纯标准库，全确定性：无任何随机源，跨运行输出逐字节一致）

---

## 1. L2 换掉了什么，以及三个声明

L1 把真实（微小）模型换进了**单个** agent 的循环，测出了违规率 p，算出了 harness 用
retry 买可靠性的价格。但那一切都发生在一个 agent 的内部。真实的 agent 系统几乎总是
**多个角色**的：一个 planner 负责拆任务、一个 executor 负责执行、可能还有 verifier、
critic……「planner 先出计划再执行」这条路线本身有论文谱系——Plan-and-Solve
（arXiv:2305.04091，2026-08-10 于 arxiv.org 核验）在 zero-shot CoT 上证明
「先拆解、后执行」优于直接推理。

一旦拆成两个 agent，一类**全新的失败**出现了，它们和模型聪不聪明无关：

- 两个 agent 互相确认「你确定吗？」「你确定吗？」——**活锁**（livelock）；
- executor 回了一条缺字段的「结果」，planner 把它存进状态——**契约被静默破坏**；
- 子任务失败后整个计划作废，或者 planner 无限改计划——**终止条件缺失**。

关键命题（本节全部实验都在证明它）：

> **部件的可靠性不蕴含整体的可靠性。** 两个零随机、永不违规的 agent，
> 放在一个没有契约和终止条件的协议里，照样可以永远循环下去。
> 整体可靠性是**协议**的属性，必须单独设计、单独测试。

三个显式声明（对应 ROADMAP §三契约，代码头部同文）：

1. **PlannerAgent / ExecutorAgent / StuckExecutor / CorruptingExecutor /
   RecklessPlanner 都是声明过的 rule-based 测试向量，不是模型**——与 L1 的
   Playback/FaultModel 同一纪律。本节的每个失败都是**按构造的协议失败**；
   把真模型接进 planner 是 L3 的工作。
2. **工具是真实的磁盘 I/O**（L1 纪律：冻结的 `list_dir` + realpath 沙箱的
   `read_file`），`[5]` 里的 FileNotFoundError 是真异常。
3. **L2 刻意不 import L1**：L1 的 TinyReActLM 把一条固定任务前缀背进了权重，
   它无法跟随 planner 现场组出的子任务 prompt（字符级记忆零泛化）——import 它
   只会拖进 torch 和 2 分钟训练，对「协作」教不出任何真的东西。协议层值得单独
   隔离研究，这正是本节的 K+1 边界。工具因此在 L2 里按同一纪律重新实现。

---

## 2. 运行与输出（逐字粘贴）

```bash
$ python3 L2_planner_executor.py   # 秒级，无训练
```

以下为定稿后 run1 的逐字输出（3 遍 × 3 个不同 CWD 两两 diff 为空——无随机源、
无计时行，输出 md5 = `997344ec19996ada46bef2a8e1321f01`，89 行）：

```text
=====================================================================
nano-agentscope L2 — planner + executor: contracts & termination
=====================================================================
python 3.13.13 | stdlib only, no randomness
declarations: all agents are declared rule-based test vectors
  (NOT models) — every failure below is a protocol failure by
  construction; tools are real disk I/O (L1 discipline).

[0] the message contract (enforced at the boundary)
    plan     requires metadata: plan_id, steps
    subtask  requires metadata: plan_id, subtask_id, tool, args
    result   requires metadata: plan_id, subtask_id, status
    clarify  requires metadata: plan_id, subtask_id
    answer   requires metadata: plan_id
    abort    requires metadata: plan_id, reason
    roles: planner, executor | every crossing is validated BEFORE it may touch the shared log

[1] happy path: planner + executor on the L1 task
    #1 planner  plan     2 steps: s1=list_dir, s2=read_file(corpus.txt)
    #2 planner  subtask  s1 list_dir {}
    #3 executor result   ok s1 <- ['L0_react_loop.py', 'L1_real_agent_loop.py', 'L2_planne
    #4 planner  subtask  s2 read_file {"path": "corpus.txt"}
    #5 executor result   ok s2 <- ReAct: Synergizing Reasoning and Acting in Language Mode
    #6 planner  answer   reasoning and acting
    status=answered | attempts=6 | messages=6 | violations=0 | replans=0 | answer='reasoning and acting'

[2] contract violation at the boundary (CorruptingExecutor)
    REJECTED executor-1: missing_field — result requires metadata['subtask_id']
    status=answered | attempts=7 | messages=6 | violations=1 | replans=0 | answer='reasoning and acting'

[3] livelock: StuckExecutor, with and without the guard
    guard ON  (window=4):  status=no_progress | attempts=6 | messages=6 | violations=0 | replans=0 | answer=None
    guard OFF (budget=24): status=budget_exhausted | attempts=24 | messages=24 | violations=0 | replans=0 | answer=None
    guard-off tail: clarify / subtask / clarify / subtask  <- clarify/re-dispatch forever; only the budget fuse stopped it, and it mislabeled
    the failure. The guard stopped 18 crossings earlier WITH the correct diagnosis.

[4] budget sweep: a fuse, not a controller (happy path)
    budget   status            attempts  messages  answer
    2        budget_exhausted  2         2         None
    4        budget_exhausted  4         4         None
    6        answered          6         6         'reasoning and acting'
    8        answered          6         6         'reasoning and acting'
    16       answered          6         6         'reasoning and acting'
    below the task's message distance (6) the budget amputates;
    at/above it, it never fires — healthy flows are governed by
    semantics (T3), the fuse only bounds the pathological cases.

[5] failure isolation + replan (real FileNotFoundError)
    #1 planner  plan     2 steps: s1=list_dir, s2=read_file(notes.txt)
    #2 planner  subtask  s1 list_dir {}
    #3 executor result   ok s1 <- ['L0_react_loop.py', 'L1_real_agent_loop.py', 'L2_planne
    #4 planner  subtask  s2 read_file {"path": "notes.txt"}
    #5 executor result   error s2 <- FileNotFoundError: no such file in module dir: 'notes.tx
    #6 planner  subtask  s3 read_file {"path": "corpus.txt"} [replan]
    #7 executor result   ok s3 <- ReAct: Synergizing Reasoning and Acting in Language Mode
    #8 planner  answer   reasoning and acting
    status=answered | attempts=8 | messages=8 | violations=0 | replans=1 | answer='reasoning and acting'
    the failure stayed inside ONE subtask; the rest of the plan
    survived, and the fallback parsed the REAL list_dir output.
    zero replan budget: status=aborted | abort reason='replan_budget_exhausted' | partial log kept=6 msgs
    RecklessPlanner (ignores budget): status=replans_exhausted | replans=2 — T4 fired even though the planner
    kept going: the orchestrator does not trust agents with liveness.

[6] the coordination ledger
    L1 single agent : 3 model calls, zero boundaries to validate.
    L2 decomposition: 6 messages for the same task — every one
    crossed a validated boundary. What the extra messages bought,
    measured above: a corrupted crossing caught BEFORE it touched
    shared state [2]; a livelock diagnosed, not just truncated [3];
    a failed subtask replanned locally while the plan survived [5].
    In real systems each planner/executor turn is itself a model
    call — L1's calls/success ledger generalizes to messages/task.

=====================================================================
✅ self-check passed:
   happy path answers correctly in exactly 6 validated messages /
   boundary rejects the corrupted crossing, then recovers /
   livelock guard: no_progress at 6 attempts vs budget fuse at 24 /
   budget sweep: fuse never fires on the healthy flow /
   real FileNotFoundError isolated + replanned / abort is a state /
   T4 backstop fires on a budget-ignoring planner
=====================================================================

takeaway: put two reliable agents in a room and they can still
          loop forever, shout past each other, or hand each other
          malformed notes. Reliability of the PARTS does not imply
          reliability of the WHOLE — that takes a typed contract at
          every crossing and termination conditions owned by the
          orchestrator, not by the agents.
```

---

## 3. 代码结构（608 行，四个板块 + 实验）

| 板块 | 内容 | 关键点 |
|------|------|--------|
| [A] 真工具 | `list_dir` / `read_file` | L1 纪律原样：`list_dir` 冻结于 L2 定稿时刻（八件清单，含本级交付物；观察值喂给回退启发式与打印轨迹，冻结使锚与目录状态解耦，L3 进目录不再击穿）；`read_file` 保持 live + realpath 沙箱 |
| [B] 消息契约 | `CONTRACT` / `validate_msg` | 六种消息类型 × 必需 metadata 字段；**类型化违规**（`bad_shape / bad_role / bad_kind / missing_field / bad_status`）——L1 `parse_block` 的思想从「解析单块模型输出」升维到「校验 agent 间每一次穿越」 |
| [C] 声明式 agents | `PlannerAgent` / `ExecutorAgent` + 三个测试向量 | planner 策略 P1–P6 显式写出（含**故意天真**的 P5：对 clarify 无条件重派——活锁必须由守卫而非 planner 终结）；StuckExecutor / CorruptingExecutor / RecklessPlanner 各构造一种协议失败 |
| [D] Orchestrator | T1–T4 + abort | 终止条件全部显式为**数据**：`budget_exhausted / no_progress / answered / replans_exhausted / aborted`，每个都带完整消息日志返回——失败是有产出的状态，不是异常 |

---

## 4. 逐段解读

**[1] happy path**。同一个任务，L1 的单 agent 用 3 次模型调用走完；L2 的
planner + executor 走了 **6 条消息**：plan → subtask(s1) → result → subtask(s2)
→ result → answer。多出来的一倍流量买到了什么？先看 plan 这条消息本身：它是
planner 对 executor 的**公开承诺**（「我将要你做这两件事」），从此 orchestrator
和两个 agent 共享同一份计划文本——单 agent 的「内心打算」变成了可审计的 crossing。
每条消息都在进入共享日志**之前**过一遍 `validate_msg`，这就是 [2] 的伏笔。

**[2] 边界违规**。CorruptingExecutor 的第一条 result 被抽掉了
`metadata['subtask_id']`——边界拒绝它（`missing_field`），把违规详情作为
critique 退回给 executor，executor 重发合规版本，任务照常完成：
`attempts=7, messages=6, violations=1`。注意被拒消息**没有进入共享日志、
没有触碰任何 agent 状态**——这是 L1 critique-retry 的跨 agent 版本：L1 里
critique 发生在单 agent 的 prompt 内，这里 critique 发生在**边界**上，违规
消息连「存在过」都不算。

**[3] 活锁**。StuckExecutor 对每个子任务都回「which file exactly?」，naive
planner（P5）每次都重新派发同一个子任务。双方都在**合规地说话**——没有一条
消息违规——但系统在原地打转。守卫的定义是核心：**progress 事件 = 被接受的
plan / 某 subtask_id 的首次派发 / 任何 result**；重复派发已见过的 subtask
不算 progress。连续 4 个 turn 无 progress → `no_progress`，6 次 attempt 就
停，诊断精确。关掉守卫对比：预算保险丝在第 24 次才烧断，状态被**误标**为
`budget_exhausted`（听起来像「任务太大」，实际是「协议死锁」），还多烧了
18 次穿越。

**[4] 预算扫描**。 happy path 的「消息距离」是 6：预算 <6 时保险丝切断任务
（无答案）；≥6 时它**从不触发**——健康流由语义终止（T3 answered）统治，预算
只兜底病态情形。这就是「保险丝不是控制器」的精确含义。AgentScope 在这件事上
有一个精致的补丁：`max_iters` 之外另有 `structured_output_grace_iters=5`——
触顶后再给 5 次宽限，专门用来**把结论交出来**（见 §6），因为「预算烧断时
什么都不产出」对下游是最坏的结果。

**[5] 失败隔离 + 重规划**。planner 以为要读 `notes.txt`（不存在）：executor
回一条 `status=error` 的 result（内容是**真实的** FileNotFoundError），planner
用 list_dir 的**真实输出**做回退（解析冻结清单、取第一个没失败过的 `.txt`
→ corpus.txt），打上 `[replan]` 标记重派——任务完成，`replans=1`。失败被
隔离在**一个子任务**里，计划的其余部分存活。两个变体把终止路径补全：
planner 自身 `replan_budget=0` 时，它发出 `abort` 消息——**放弃是一个有
reason、有残留日志的状态**；RecklessPlanner 无视预算无限改计划时，
orchestrator 的 T4 在 `replans=2` 时强行终结——**orchestrator 不信任 agent
的账本**，防御纵深的最后一层永远在自己手里。

---

## 5. 机制深挖

### 5.1 契约为什么必须在边界执行，而不是靠发送方自觉

`validate_msg` 的位置是整个 [B] 板块唯一的重点：它在**消息进入共享日志之前**
执行。如果改成「planner 收到结果后自己检查字段齐不齐」，会发生三件事：
违规消息已经进入了可被其他组件读到的状态；每个接收方都得重复写一遍检查；
以及——发送方和接收方是**同一个模型生态**里的角色，它们的违规是相关的
（L1 测过 sticky 失败：难 prompt 一直难）。边界校验把契约从「N 个角色的
自觉」收敛成「1 个执行点」，这正是数据库把约束放在表上而不是放在每个
应用里的原因。类型化违规（五种 kind）则是 L1 的教训复用：只有违规可枚举，
critique 才能点名、恢复策略才能挂接、统计才有意义。

### 5.2 终止是语义：五种状态，每一种都有产出

本节的 orchestrator 永远**返回**，从不**抛异常**，且每个返回都带完整日志：

| 状态 | 触发 | 语义 |
|------|------|------|
| `answered` | T3：验证过的 answer 消息 | 成功 |
| `aborted` | planner 主动发 abort | 语义级放弃（知道做不下去，交出部分结果） |
| `no_progress` | T2：活锁守卫 | 协议死锁诊断 |
| `replans_exhausted` | T4：重规划超限 | orchestrator 对 planner 的否决 |
| `budget_exhausted` | T1：总预算 | 最后的保险丝（信息量最低的状态） |

注意这个排序是有意的：**信息量高的终止条件先检查**。`budget_exhausted`
几乎什么都没告诉你（「反正停了」），所以它只该在别的诊断都没触发时出现——
[3] 的对比实验里，关掉守卫后状态退化为 budget_exhausted，就是诊断能力
退化的直接体现。生产系统里这个排序决定 oncall 的效率：拿到一个失败状态，
应该能立刻知道下一步查什么。

### 5.3 活锁与 progress：为什么「还在动」≠「有进展」

L1 讲过 sticky 失败让「重试算术」失效；L2 的活锁是它的孪生兄弟——
**sticky 无进展**：每个 turn 都合规、每个角色都在干活，但状态空间没有
前进。守卫的唯一设计问题就是「什么叫进展」。本节的定义
（plan / 新 subtask 派发 / 任何 result）有一个值得注意的性质：它是
**可机器判定的**——orchestrator 不需要理解消息内容，只看类型和 id。
这是有意为之：任何需要「理解语义」的 progress 判定都会把活锁检测
变成一个模型问题，而活锁检测必须是协议问题。思考题 1 会问这个定义
的边界在哪。

### 5.4 分解的代价：messages/task 账本

同一任务：L1 = 3 次模型调用、0 个边界；L2 = 6 条消息、每条都过校验。
代价是双倍的流量 + orchestrator 本身；买到的三样东西都在本节实测过：
违规在触碰共享状态前被拦截 [2]、活锁被诊断而非截断 [3]、失败被局部
重规划而非全局作废 [5]。真实系统里每次 planner/executor 发言本身就是一次
模型调用——L1 的 `calls/success` 账本自然推广为 `messages/task`。这也给出
「什么时候**不该**拆 agent」的判据：如果任务没有可隔离的失败域、没有需要
不同权限/工具集的角色、没有需要独立预算的子目标，拆出来的边界只会变成
纯粹的税。多 agent 不是架构时尚，是对「失败隔离 + 权限分离 + 预算独立」
的需求的回应。

---

## 6. 与权威实现对照（AgentScope v2.0.6 main + v1.0.0 tag，2026-08-10 codeload tarball 现场核验）

L1 的锚点核验于 08-06；本轮（08-10）重新抓取 main 时发现版本已跃迁到
**v2.0.6**（`_version.py`），且有一处架构级变化值得单独讲（见差异 3）。
全部行号取自 08-10 快照。

| nano L2 部件 | AgentScope 对应物（核验行号） |
|------|------|
| 消息契约（四字段 dict） | `src/agentscope/message/_base.py:L67` `class Msg(BaseModel)`，docstring 自述 *"responsible for information storage and transmission among different agents"*。字段分三组：**context**（`name` L75 / `content: list[ContentBlock]` L77 / `role` L79 / `id` L81）、**metadata**（`metadata` L89 / `created_at` L91 / `usage` L93）、**workflow control**（`finished_at` L101 / `finished_reason` L103 / `structured_output` L106 / `error` L112） |
| 构造期校验 | `message/_base.py:L117` `validate_role_content`（pydantic `model_validator`）：按 role 断言 content blocks 的合法性——权威实现同样**不信任发送方**，校验发生在消息构造时 |
| 终止即数据 | `src/agentscope/types/_reply.py:L10-16` `ReplyFinishedReason` StrEnum：`COMPLETED / INTERRUPTED / EXCEED_MAX_ITERS / ERROR`——终止原因**住在消息里**（`Msg.finished_reason`），docstring 明言 *"None until a REPLY_END event is applied"* |
| 类型化消息内容 | `message/_block.py`：`TextBlock` L11 / `ToolCallState` L128（`PENDING→ASKING→ALLOWED→SUBMITTED→FINISHED` 状态机）/ `ToolCallBlock` L138 / `ToolResultState` L185（`SUCCESS/ERROR/INTERRUPTED/DENIED/RUNNING`）/ `ToolResultBlock` L195——v2 把类型化推进到消息**内容**层：一次工具调用是消息体内一个**带状态机、过权限系统**的 block |
| 预算与宽限 | `agent/_config.py:L282` `ReActConfig`：`max_iters` 默认 **20**（L285-290）、`structured_output_grace_iters` 默认 **5**（L293-302）、`stop_on_reject`（L305+）——L1 锚点零漂移 |
| 循环与决策/副作用分离 | `agent/_agent.py:L863` `while True:` + L869 `match next_action:`（case Exit/Reasoning/Acting）；`_next_action` L3050，docstring 自述 *"Read-only: all side effects are performed by the caller"*（L1 记录的 L858-874/L3019 在 v2.0.6 漂移到 L863-869/L3050，机制不变） |
| 步骤状态 | `agent/_utils.py:L26/L32/L39` `Acting / Reasoning / Exit` pydantic 模型——L1 锚点零漂移 |
| 编排原语（v1） | v1.0.0 tag：`pipeline/_msghub.py:L11` `class MsgHub`（*"controlled the subscription of the participated agents"*，`broadcast` L115）+ `pipeline/_class.py:L8` `SequentialPipeline` + `pipeline/_functional.py:L7` `sequential_pipeline` |

**nano 与权威实现的差异（为什么它那样选）**：

1. **校验的载体不同，机制相同**。AgentScope 用 pydantic 模型把 schema 编译进
   类型系统（构造即校验）；我们用 dict + 边界函数。生产系统选前者：schema
   数量一大，手写校验函数会漂移，而 pydantic/JSON Schema 可以从单一来源生成
   文档、客户端、校验器。我们的版本胜在**可读**——五种违规 kind 一眼到底，
   这正是教学版该付的取舍。
2. **终止状态的粒度不同**。AgentScope 把 `finished_reason` 放在**每条回复消息**
   上（消息级），我们把 status 放在**会话结果**里（会话级）。消息级粒度服务的是
   流式与多轮：一个 agent 的 reply 可能在流式中途被 INTERRUPTED、可能因
   EXCEED_MAX_ITERS 带着不完整的 structured_output 结束——这些状态必须跟着消息
   走，因为下游消费的是消息而不是会话。我们的会话级 status 是它的最小形态。
3. **编排原语从核心退场——本轮核验发现的真演化**。v1.0.0 的
   `pipeline/ + msghub` 是 AgentScope 的招牌抽象（顺序/扇出/广播的组合子）；
   v2.0.6 的 core 包里它们**已被整体移除**（全树 grep 零命中，examples 也换成
   agent_service/rag/web_ui 等服务化形态）。与此同时，`Msg` 契约不但保留还
   加强了（typed content blocks + workflow-control 字段）。这个对比本身就是一课：
   **编排层是演化快的层，契约是演化慢的层**——把稳定性投资放在契约上，把
   灵活性留给编排。README 阶梯表里 L3 的「message/pipeline 抽象」对照因此
   需要横跨 v1/v2 两个快照做（v1 的 MsgHub/SequentialPipeline + v2 的
   typed blocks），L3 照此执行。
4. **信任边界的位置相同**。我们的 T4（orchestrator 不信 planner 的账本）对应
   AgentScope 的 `max_iters` 由 Agent 循环自身强制执行、`ToolCallBlock` 必须
   经 `PENDING→ALLOWED` 的权限状态机才能执行——权威实现同样**从不把活性与
   权限托付给模型侧的自觉**。

---

## 7. 费曼自检

**类比：两个外包供应商 + 一个项目经理**。planner 和 executor 是两家外包：
planner 开任务单（subtask），executor 回交付单（result）。公司不祈祷外包
靠谱，而是靠三样东西：① 前台收发文员（边界校验）——单据格式不对**当场退回**，
根本进不了档案室（共享日志）；② 项目经理盯着「进展台账」——如果最近几张单
都在问同一个问题、没有任何新交付，判定扯皮、叫停（活锁守卫），而不是等
经费烧完；③ 经费上限（预算）——它不是项目的完成条件，只是公司的止损线。
**一句话版**：两个靠谱的外包 + 一套没有收发规则和止损线的流程 = 一个可以
永远扯皮下去的项目；整体可靠性是流程的属性，得单独设计。

**反例版**（两条都能用本文实测证伪）：

1. 「每个部件可靠，整体就可靠」——[3] 实测：两个 rule-based agent 零随机、
   零违规，照样活锁；守卫不在场时只有预算能停它。
2. 「有预算就够了」——[3] 实测：预算把活锁误标成 budget_exhausted 还多烧
   18 次穿越；「能停」和「停得明白」是两回事，oncall 靠后者。

**自检问题**：你能不能向一个只写过单体函数的工程师解释——为什么「两个
agent 互相礼貌地确认」比「一个 agent 犯错」更难防？（提示：单点错误是
**事件**，格式校验抓得住；礼貌死锁是**模式**，只在消息序列上才显现——
所以防它的守卫必须维护历史，而不是只看当前这条消息。）

---

## 8. 思考题

1. **progress 的定义边界**：本节把「任何 result」都算 progress。构造一个
   场景：executor 反复回 `status=error` 的 result（每次都算 progress），
   守卫会不会失效？如果要防住「错误循环」，progress 定义该怎么改？
   （提示：result 需要带上「新信息」判据——同样的 error 重复出现不算进展；
   这和 T4 的 replan 上限是什么分工关系？）
2. **真模型进场**：本节的 planner 策略 P1–P6 里，把 planner 换成真实 LLM
   后，哪一条最可能最先被违反——P1 的分解结构、P4 的回退启发式、还是 P5？
   harness 需要为每条各加什么防御？（提示：P1 违反 = 消息结构问题，[B] 管得住；
   P4/P5 违反 = 策略问题，需要 orchestrator 侧的模式检测——这正是 L1 类型化
   parser 与 L2 守卫的分工在真模型下的放大。）
3. **终止的粒度**：AgentScope 把 `finished_reason` 放在消息上，我们放在会话上。
   给出一个**必须**用消息级终止状态的场景，和一个会话级就够的场景。
   （提示：流式中途被打断 / 一个 agent 对多个下游分别交付。）
4. **动手题**：把 `guard_window` 改成 2，用本节全部五个实验复跑一遍——
   有没有哪个**正常**流被误伤？先推导本节实现中正常流的最大 streak
   （提示：只有 [2] 的被拒消息会产生 streak=1），再跑一遍验证你的推导。
   这个余量设计和你给生产系统选窗口时考虑的因素一样吗？
5. **演化题**：v1 的 pipeline/msghub 在 v2 被移出核心，Msg 契约反而加强。
   如果你在设计自己的 agent 框架，这个观察会如何影响你的分层决策？
   （提示：哪些层该做成稳定 API，哪些层该留给用户代码——以及判断标准。）

---

## 9. 局限（诚实清单）

1. **agents 是规则写的**：本节测量的是**协议属性**（契约、终止、隔离），
   不是模型行为。planner 换成真模型后的失败模式（幻觉子任务、忘记计划、
   提前宣布完成）留 L3——那需要 L1 的可靠性代数 + 本节的协议骨架合体。
2. **固定两角色拓扑**：planner→executor 顺序回合制。群聊式协作
   （v1 MsgHub 的 broadcast 语义）、动态角色、并发回合均未覆盖。
3. **手写 dict schema**：契约用 dict + 校验函数表达，机制与 pydantic/
   JSON Schema 同款，工程化程度更低（无 schema 生成、无版本演化）。
4. **活锁构造是声明的**：StuckExecutor 是最干净的活锁形态；真实活锁常带
   部分进展（进两步退一步），对窗口式守卫更隐蔽——思考题 1 触及了这个方向。
5. **无并发/异步**：真实多 agent 系统的消息时序问题（乱序、竞态）在本节
   不存在，因为回合是严格顺序的。

---

## 10. 下一级预告与交叉引用

- **L3**：对照 AgentScope 的 message 抽象复现一个编排模式。因 v2.0.6 已移除
  pipeline/msghub（§6 差异 3），L3 的对照横跨两个快照：v1.0.0 的
  MsgHub/SequentialPipeline 组合子 + v2.0.6 的 typed content blocks 与
  workflow-control 字段；把真模型接回 planner（L1 的可靠性代数 ×
  L2 的协议骨架）。
- 交叉引用：L1（critique-retry 的跨 agent 版本 = 本节 [2] 的边界拒绝；
  `calls/success` → `messages/task`）；nano-vllm-sglang L1–L2（每条消息在
  真实系统里是 token，retry/replan 烧的是推理引擎省出来的吞吐余量）；
  03 轨 RSI（orchestrator 的消息日志 = 结构化轨迹，是回流训练数据的
  04 侧出口——比 L1 的单 agent traj 多了角色与契约维度）。

---

## 11. 溯源与口径

- **AgentScope 快照（双份，2026-08-10 于 codeload.github.com 现场抓取核验）**：
  main 分支 tarball（7,524,950 B，`__version__ = "2.0.6"`）与 v1.0.0 tag
  tarball（7,702,712 B）。§6 全部行号取自这两份快照；L1 教程记录的 08-06
  锚点在本轮复验中：`ReActConfig` / `Acting·Reasoning·Exit` /
  `_json_loads_with_repair`（`_utils/_common.py:L86`）/ `Toolkit`
  （`tool/_toolkit.py:L66`）/ `Msg`（L67）零漂移，`while True` 循环与
  `_next_action` 分别漂移到 L863-869 / L3050（机制不变）。canonical 仓库
  `github.com/agentscope-ai/agentscope`。
- **论文**：ReAct arXiv:2210.03629（L1 于 08-06 首验，本轮 08-10 于
  arxiv.org 复验标题页）；Plan-and-Solve arXiv:2305.04091（本轮 08-10 于
  arxiv.org 核验标题：*Plan-and-Solve Prompting: Improving Zero-Shot
  Chain-of-Thought Reasoning by Large Language Models*）——planner-executor
  「先拆解后执行」的谱系锚点。
- **测量口径**：L2 无随机源、无计时行——3 遍 × 3 个不同 CWD（/tmp、
  /var/tmp、/）运行输出逐字节一致（md5 `997344ec19996ada46bef2a8e1321f01`，
  89 行）；全部断言在代码内（self-check 块）。`list_dir` 冻结于 L2 定稿
  时刻的八件清单（含 `L2_planner_executor.py` 与 `tutorial_L2.md` 自身），
  声明在代码内，纪律沿用 L1。
- **写作过程修掉的真实 bug**（self-check 当场抓出）：CorruptingExecutor
  初版漏掉 first-only 条件，每条 result 都被污染——[2] 实测 violations=2
  与断言冲突，一次复跑定位。
