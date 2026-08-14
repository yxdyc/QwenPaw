# nano-agentscope L3 — typed messages × broadcast wiring × 真模型

> **级别**：L3（K+1：L2 的协议层（rule-based agents）→ L3 的「类型化契约 × 编排模式 × 模型入席」三合一）
> **文件**：[`L3_typed_msghub.py`](L3_typed_msghub.py)（967 行，依赖 torch——来自 L1 import；全程 seeded、无任何计时行，跨运行输出逐字节一致）
> **对照权威实现**：AgentScope **双快照**——v2.0.6 main（typed blocks / 构造期校验 / 消息级终止）+ v1.0.0 tag（MsgHub 广播 / SequentialPipeline 组合子）。L2 §6 发现 v2 已把 pipeline/msghub 移出 core，所以 L3 复现的是**跨两个快照的编排模式**，这正是 tutorial_L2 §10 预告的对照方案。

---

## 1. L3 换掉了什么，以及四个声明

L1 把真实（微小）模型放进**单 agent** 循环，测出违规率 p，算出 harness 的可靠性代数；L2 把 agent 拆成 planner + executor，证明**整体可靠性是协议的属性**——但 L2 的 agent 全是规则写的，消息是裸 dict，拓扑是固定两角色点对点。L3 一次换掉三样，每一样都对着权威实现的一个真实设计决策：

1. **消息从 dict 变成类型化对象**（v2.0.6 快照）：content 是 typed blocks 的列表，tool call 是**带状态机的 block**（必须过 permission 一步才能执行），终止原因 `finished_reason` **住在消息里**而不是会话结果里；校验发生在**构造期**——畸形消息根本无法出生（L2 是「门口拦截」，这里是「出生拦截」）。
2. **编排从点对点变成广播群聊**（v1.0.0 快照）：`MsgHub` 在进入时**重接每个参与者的订阅表**，此后每条发言自动扇出给全房间；第三个角色 verifier 因此**免费**获得全屋知识。`SequentialPipeline`（线性 fold）作为对照组——同样的 agent、同样的契约，三种接线，三种认知状态。
3. **真模型接回 planner 席**：L1 的 TinyReActLM 按**同一配方重训**（params/loss 与 L1 锚点逐位一致），贪心路径上它的每个输出都变成一次过校验的类型化 crossing；采样路径上测协议级的可靠性代数。L2 §9 遗留的「真模型 planner 失败模式」在这里第一次有实测数据。

四个显式声明（对应 ROADMAP §三契约，代码头部同文）：

1. **planner 后端 = L1 的 TinyReActLM 重训**——真实学习分布、采样输出，但字符级记忆、不推理。托管模型路径 `HostedBackend` 代码就绪、`[TODO: needs key]`（复用 L1 [6] 已过本地契约服务器验证的 OpenAI-compatible 客户端）；无 key 时确定性 fallback = 本地小模型（L1 先例）。
2. **ExecutorAgent / VerifierAgent / RulePlanner / PrematurePlanner / WrongAnswerPlanner 都是声明的 rule-based 测试向量，不是模型**——L1/L2 同款纪律，每个失败都是构造出来的，且构造处都有标签。
3. **工具 = L1 的磁盘 I/O 层整体 import**（冻结六件清单的 `list_dir` + realpath 沙箱 `read_file`）。L3 **刻意不**重新冻结自己的九件清单：模型是在 L1 的观察串上训练的，planner 必须被喂「它受训于其中的那个世界」的观察——这是模型支撑的 agent 的真实约束（模型自带一份世界快照，harness 必须遵守）。
4. **确定性**：全程 seeded，**无任何计时行**，输出逐字节可复现（L2 锚点纪律）；训练锚点与 L1 逐位一致（93,731 params / loss 0.0218）。

---

## 2. 运行与输出（逐字粘贴）

```bash
$ python3 L3_typed_msghub.py   # ~3-4 分钟（训练 ~2 分钟 + [4] 扫描 ~1.5 分钟）
```

以下为公开脱敏版 2026-08-14 完整复跑的逐字输出（输出 md5 = `8786f888db882b3710f4bed2dbe23595`，120 行；跨 torch/硬件版本需重新核验）：

```text
=====================================================================
nano-agentscope L3 — typed messages x broadcast wiring x model
=====================================================================
python 3.13.13 | torch 2.13.0
declarations: planner backend = L1's TinyReActLM retrained on
  the SAME recipe (real ~94K-param char-LM; memorizes, does not
  reason); hosted planner path code-ready [TODO: needs key];
  executor/verifier/Rule/Premature/WrongAnswer planners are
  declared rule-based test vectors; tools = L1's real disk I/O
  (frozen list_dir + sandboxed read_file), imported wholesale.

[0] train the planner's model (L1 recipe, verbatim, ~2 min)
    transcripts=75 (60 clean + 15 critique-repair) | params=93,731 | final loss=0.0218
    cross-level anchor: params/loss match L1's anchored values
    bit-for-bit — L3's planner IS L1's model, no re-derivation.
    planner backend: local TinyReActLM fallback [TODO: needs key for hosted]

[1] the typed message layer: validation at construction
    REJECTED at birth  empty_content                                        -> empty_content
    REJECTED at birth  bad_block_type                                       -> bad_block_type
    REJECTED at birth  bad_block_field                                      -> bad_block_field
    REJECTED at birth  role_block_mismatch (executor forging a tool_call)   -> role_block_mismatch
    REJECTED at birth  role_block_mismatch (planner fabricating a tool_result) -> role_block_mismatch
    REJECTED at birth  bad_finished_reason                                  -> bad_finished_reason
    state machine legal path: pending -> allowed -> finished  (final state: finished)
    REJECTED transition: pending -> submitted (illegal_transition) — the permission system cannot be skipped
    two validation layers: [layer 1] raw model text -> typed
    crossing (L1 parse kinds); [layer 2] message construction
    (the five kinds above). A malformed message cannot be born.

[2] the orchestration pattern: same agents, three wirings
    hub          verdict=verified      verifier knowledge=5 msgs | extra sends=0
    p2p          verdict=not_verified  verifier knowledge=1 msgs | extra sends=0
    p2p+forward  verdict=verified      verifier knowledge=5 msgs | extra sends=4
    sequential    each stage sees exactly the previous stage's output (verifier saw 1 msg: the result, never the task; a fold has no shared log)
    broadcast buys the verifier the room's knowledge at ZERO
    extra sends; point-to-point pays one explicit send per
    crossing to get the same epistemic state.

[3] happy path: the real model in the planner seat (greedy)
    #1 planner  subtask  s1 list_dir {} [finished]
    #2 executor result   s1 <- ['L0_react_loop.py', 'L1_real_agent_loop.py', 'READM [success]
    #3 planner  subtask  s2 read_file {"path": "corpus.txt"} [finished]
    #4 executor result   s2 <- ReAct: Synergizing Reasoning and Acting in Language  [success]
    #5 planner  answer   reasoning and acting [finished_reason=completed]
    #6 verifier verdict  verified (supported_by_evidence)
    (block states shown as of end-of-run: every tool_call has
    traversed pending -> allowed -> finished, [1] legal path)
    status=verified | model_calls=3 | crossings=6 | violations=0 | answer='reasoning and acting'
    observations delivered by broadcast: 15 (task announcement to 3 + 6 crossings x 2 others = 15);
    finished_reason on the answer message: 'completed' — termination is message data.

[4] reliability algebra at the protocol level (T=0.7)
    p(step0) = 0.155   [compliant:169, final:9, no_action:22]
    p(step1) = 0.100   [compliant:180, final:3, no_action:15, unknown_tool:2]
    p(answer) = 0.160   [compliant:15, final:168, no_action:17]
    task runs (200 each, fresh seeded model per task, hub wiring):
     k   measured   formula   mean_calls
    0     72.5%     63.9%     2.67
    1     76.5%     94.1%     2.88
    2     76.5%     99.1%     2.88
    k=0: per-site failure probabilities COMPOSE across boundaries
    — measured within sampling noise of prod_i (1-p_i). The algebra
    that priced L1's single loop prices the protocol too.
    k>=1: the iid formula is an upper BOUND, not an equality —
    retries are not fresh iid draws (failures are sticky, L1) and
    repair only works where critique-repair was trained (the answer
    position was not). Retries still buy monotone uplift; the iid
    magnitude overpromises. Same lesson as L1, one level up:
    rewiring the loop does not restore independence.
    note: step0 spectrum 'final:9' — organic premature
    completion: at T=0.7 the real model jumps to Final Answer with
    no evidence; every such draw is rejected by the verifier below.

[5] premature completion: only a third role can catch it
    hub + verifier:   status=not_verified | verdict=no_evidence | crossings=2
    L2 two-party:     status=answered | answer='reasoning and acting' — ACCEPTED: with no verifier role and no
    evidence check, this failure mode cannot even be expressed.
    wrong answer:     status=not_verified | verdict=unsupported — evidence existed but did
    not support the claim (no_evidence vs unsupported are different diagnoses).

[6] termination lives in the message (T=1.3, zero retries)
    partial message text: '<last raw> [Step.0 0] Thought: Let man ain Laren lools ar int it to the'
    message.finished_reason = 'exceed_max_iters' | session status = exceed_max_iters | verdict = incomplete_reply
    the verifier rendered a verdict ON the failed message — any
    downstream consumer reads termination off the message itself
    (v2 ReplyFinishedReason), no session context needed.

[7] the coordination ledger (live numbers)
    level  model_calls  crossings  parties  verifier  observations
    L1     3            0          1        -         -
    L2     0 (rules)    6          2        -         -
    L3     3            6          3        yes       15
    L2's explicit plan message became implicit (the plan lives
    in the model's weights; the protocol sees only crossings).
    What L3's extra wiring bought, measured above: evidence-based
    acceptance instead of trust [5], knowledge without forwarding
    [2], termination as message data [6]. What it costs: every
    observation is tokens in the observer's context — the budget
    nano-vllm-sglang's KV cache pays for.

=====================================================================
✅ self-check passed:
   retrained model matches L1's anchor bit-for-bit (93,731 / 0.0218) /
   six typed construction errors + the permission-skip rejected /
   hub verifies at zero extra sends; p2p starves the verifier /
   greedy real-model run: 3 calls, 6 crossings, verified /
   retry algebra: composition holds at k=0, iid upper bound at k>=1 /
   premature completion caught ONLY when a verifier sees the room /
   exceed_max_iters lives on the message, verdict still rendered
=====================================================================

takeaway: contracts become typed, orchestration becomes wiring.
          A message that cannot be born malformed, a tool call
          that cannot execute itself, a room where everyone hears
          everything, and a verifier whose knowledge IS the room's
          — reliability is no longer a property of any participant.
          It is a property of the wiring. The wiring is the part
          that evolves fast (v1's pipelines are gone in v2); the
          contract is the part you invest in.
```

---

## 3. 代码结构（967 行，四个板块 + 实验）

| 板块 | 内容 | 关键点 |
|------|------|--------|
| [A] 类型化消息层 | `Msg` / blocks / `transition` | v2.0.6 镜像：content = typed blocks（text / tool_call / tool_result）；构造期校验五种类型化错误（`empty_content / bad_block_type / bad_block_field / role_block_mismatch / bad_finished_reason`）+ 状态机非法迁移 `illegal_transition`；`finished_reason` 取 v2 的 `ReplyFinishedReason` 四值 |
| [B] 编排组合子 | `AgentBase` / `MsgHub` / `sequential_pipeline` | v1.0.0 镜像：**广播 = 重接订阅表**（`reset_subscribers` 排除自己，v1 `_agent_base.py:L447`），hub 不路由消息、只在进出时改线；显式 `broadcast` 含发送者（v1 L115-123），reply 广播不含 |
| [C] agents | `ModelPlanner` + 四个声明向量 + `HostedBackend` | planner 的 prefix 与 L1.Harness **逐字符同构**（字符级记忆对 prompt 漂移零容忍）；critique 通道做了**词表安全化**（见 §5.4 与 §11 的真实 bug） |
| [D] `HubOrchestrator` | crossings / permission / verdicts | 每条被接受的 crossing 过 hub 广播；`pending→allowed` 由 orchestrator 独占（permission 一步）；verifier 只凭**自己观察到的**内容裁决；终止状态：`verified / not_verified / exceed_max_iters / loop_detected / budget_exhausted` |

---

## 4. 逐段解读

**[0] 训练锚点**。L3 的 planner 不是新模型——是 L1 配方的逐位复现（同 seed、同 transcript 构造、同训练函数），`params=93,731 / final loss=0.0218` 与 L1 锚点逐位一致。这不是偷懒，是**跨级别同一性证明**：L3 协议层承受的模型失败率，就是 L1 实测过的那个分布，两级数据可直接互引。

**[1] 出生拦截**。六个构造失败演示 + 一个状态机非法迁移。注意两种 `role_block_mismatch`：executor 想构造 `tool_call`（**自我授权**）与 planner 想构造 `tool_result`（**伪造证据**）——角色 × 块类型的合法表把这两种攻击在构造期就挡住，不依赖任何一方的自觉。这是 v2 `validate_role_content`（按 role 断言 content blocks 合法性）的 nano 镜像：v2 规定 user 消息只能 text/data、system 只能 text（`_base.py:L33-48`），assistant 不设限——权威实现同样把「谁能说什么」编译进类型系统。状态机演示里 `pending → submitted` 被拒：v2 的迁移图（`_block.py` ToolCallBlock docstring）规定 submitted 只能从 allowed 到达——**跳过 permission 直接提交外部执行**是非法边。

**[2] 同一组 agent，三种接线**。这是 L3 的核心实验：agent 与契约全部 held constant，只变接线——
- **hub**：verifier 观察到全部 5 条 crossing（广播），裁决 `verified`，额外发送 0；
- **p2p**：verifier 只收到 answer（orchestrator 咨询它时递的），**看到主张、没看到证据** → `no_evidence`；
- **p2p+forward**：每条 crossing 显式转发给 verifier，4 次额外发送换回与 hub 相同的认知状态；
- **sequential**（v1 fold）：每一级只看到上一级的输出，verifier 拿到 result 却拿不到 task——fold 没有共享日志。
广播买到的不是「发消息更方便」，是**认知状态的免费同步**：房间里的每个成员自动拥有房间的完整账本。代价在 [7] 结算。

**[3] 真模型入席（贪心）**。模型走 L1 的三步轨迹（list_dir → read_file → Final Answer），但每一步都变成类型化 crossing：`tool_call` block 从 pending 起步，orchestrator 授予 allowed，executor 在**确认 state==allowed 之后**才执行真实工具，产出 `tool_result` block 并把 call 推进到 finished。answer 消息带上 `finished_reason=completed`——终止是消息数据。3 次模型调用、6 条 crossing、15 次观察广播（announcement 3 + 6×2），verifier 裁决 supported_by_evidence。对比 L2：同样的「6 条消息」，但 L2 的 plan 消息（planner 的公开承诺）消失了——**计划沉进了模型权重**，协议只看得到 crossing。这个差异是 L3 最重要的结构性变化，§5.5 展开。

**[4] 协议级可靠性代数**。三个位置各采 200 个样本测违规率：p(step0)=0.155 / p(step1)=0.100 / p(answer)=0.160（谱线里能看到真实的违规种类分布）。然后 200 任务 × k∈{0,1,2} 重试：
- **k=0**：实测 72.5% vs 乘积公式 Π(1-pᵢ)=63.9%——各位置首试失败率**跨边界复合**，偏差在三个 200-样本 p 估计的复合噪声内。L1 为单循环定价的代数，在协议层同样成立；
- **k≥1**：iid 公式是**上界而非等式**——实测 76.5%/76.5% 远低于 94.1%/99.1%。原因有二，都是 L1 的旧课：失败是 sticky 的（难 context 一直难），且 critique 修复只在**训练过修复的位置**有效（answer 位置的修复没进训练集）。`mean_calls` 在 k=1/2 时均为 2.88 也是同款证据：相关失败在**单个位置快速烧完**重试，而不是把重试摊到各位置。重试预算从 k=0 到 k=1 买到提升，但 k=2 已无增益（72.5→76.5→76.5），iid 的量级是过度承诺；
- 谱线彩蛋：step0 有 **19/200 的 'final'**——模型在 T=0.7 下会**有机地**跳过证据直接宣布答案。[5] 的 PrematurePlanner 是声明向量，但这个失败在野外真的发生，且每一次都被 verifier 拒绝。

**[5] 提前完成：只有第三角色抓得住**。PrematurePlanner 一开口就是正确答案（失败不是无知，是**无证据的宣称**）：hub + verifier 裁决 `no_evidence`（日志里没有任何 tool_result）；同一向量放进 L2 的两方协议**被接受**（T3 只校验消息形状，无人核对证据）——这个失败模式在两方拓扑里**无法被表达**。WrongAnswer 变体区分了第二种诊断：证据存在但不支持主张 → `unsupported`。`no_evidence`（没做就说）与 `unsupported`（做了但说的不对）是两种不同的病，oncall 靠这个区分查不同的方向。

**[6] 终止住在消息里**。T=1.3、零重试：第一次违规即耗尽预算，planner 发出**部分消息**——文本是最后一次原始输出（采样噪声下支离破碎的真实文本），`finished_reason=exceed_max_iters` **在消息自身**，verifier 仍能对这条失败消息裁决（`incomplete_reply`）。对照 L2：status 只会话级——下游想知道「为什么结束」必须拿到整个会话上下文。消息级粒度服务的是流式与多下游：一条 reply 可以在中途 INTERRUPTED、可以带着不完整的 structured_output 结束 EXCEED_MAX_ITERS，消费方（verifier、日志、RSI 轨迹收集器）只读消息本身就够了。

**[7] 协调账本**。L1：3 次调用、0 个边界；L2：6 条 crossing、2 方、0 观察；L3：6 条 crossing、3 方、15 次观察。crossing 数没变，多出来的是**房间**——以及房间的账单：每次观察都是观察者 context 里的 token，nano-vllm-sglang 的 KV cache 付的就是这笔钱。

---

## 5. 机制深挖

### 5.1 出生拦截 vs 门口拦截：校验位置再前移一级

L2 的 `validate_msg` 在**消息进入共享日志之前**执行（门口）；L3/v2 把校验挪到**消息构造的那一刻**（出生）。两者的执行点论证相同（不信任发送方），但出生拦截多买到两样东西：其一，违规消息**连对象都不是**——不存在「已被某个组件瞥见但尚未入账」的中间态，并发下也没有窗口；其二，角色 × 块类型合法表把**权限分离**编译进了构造器——executor 物理上构造不出 tool_call（自我授权）、planner 构造不出 tool_result（伪造证据），这比「事后检查谁发了什么」强一个量级。代价是 schema 与代码同生共死：v2 用 pydantic 把 schema 编译进类型系统（构造即校验、单一来源生成文档/客户端/校验器），nano 用手写校验换五种违规 kind 一眼到底的可读性——教学版付的取舍，工程化时选前者。

### 5.2 广播 = 重接订阅表，不是消息路由

读 v1 源码最容易想错的地方：MsgHub **不转发任何消息**。它在进入时对每个参与者调 `reset_subscribers(participants)`（排除自己），在退出时清空——此后每条 reply 由**发言者自己**扇出给订阅者（`_agent_base.py:L239-251`）。hub 是一个**接线员**，不是邮差。这个设计把「谁听到谁」从消息路径里抽出来变成**可动态修改的图**：`hub.add/delete` 中途改房间成员，只动订阅表，不动任何在途消息。nano 逐字镜像了这套机制（含「reply 广播排除自己、显式 broadcast 含发送者」的不对称）。p2p+forward 实验量化了它的价值：同样的认知状态，hub 花 0 次额外发送，点对点花 4 次——**O(1) 接线 vs O(接收者) 转发**，且房间越大差距越大。

### 5.3 状态机：工具调用是一个有生命周期的 block

v2 把 tool call 从「消息里的一行 JSON」升格为**带状态机的 block**：`PENDING→ASKING→ALLOWED→SUBMITTED→FINISHED`，迁移图写在 `ToolCallBlock` 的 docstring 里——pending 可被 permission DENY 直达 finished、可 ASK 进 asking（等人确认）、可 ALLOW 进 allowed；allowed 本地执行到 finished 或外部提交到 submitted。nano 复用全量词汇、实跑 `pending→allowed→finished` 主路，声明 asking/submitted 不演练。本质只有一条：**agent 不能单方面推进自己请求的世界状态**——executor 拒绝执行 state≠allowed 的 call，`pending→submitted`（跳过 permission）是非法边。这与 L2 的 T4（orchestrator 不信 planner 的账本）是同一课的不同讲法：活性与权限永远不托付给模型侧的自觉。v2 还给了 `suggested_rules: list[PermissionRule]`——permission 是规则系统，不是一个 if。

### 5.4 critique 通道是未设防的输入通道（写作中真抓的 bug）

L1 的 critique 模板把 parser 的 `(kind: payload)` 原样回填进 prompt。L3 的宽扫描（600 个 T=0.7 任务）第一次踩中后果：模型生成未闭合 JSON 时，`json.JSONDecodeError` 的消息是 **"Unterminated string starting at..."**——大写 'U' 不在字符级模型的词表里，critique 一进 prompt 就在 embedding 查表处 KeyError。L1 的 seeded 扫描从未抽到这种 payload，所以 PASS 材料里埋着这颗哑弹。修复是声明式的：critique 在进 prompt 前**按后端词表安全化**（char-level 后端把越界字符换成 '?'；托管 tokenizer 后端没有 .stoi、不需要）。一般化教训：**凡是把异常文本回流进模型 prompt 的通道（parser 错误、工具报错、验证反馈），都是未设防的输入通道**——字符表、编码、长度、注入面，全都可能在那里出事。真实系统里这通常表现为「retry 路径偶发崩溃且难以复现」，因为只有在特定违规种类下才触发。

### 5.5 计划沉进权重之后，协议必须扛起全部可观察性

L2 的 planner 先发一条 plan 消息——那是它对 executor 和 orchestrator 的**公开承诺**，可审计、可对照。L3 的模型 planner 没有这条消息：计划是权重里的记忆轨迹，协议只看到一条条 crossing。这不是中性的结构变化：**当智能变成统计的，协议只能依赖穿过边界的东西**。L2 可以「拿计划和执行对照」，L3 只能「拿证据和结论对照」——这正是 verifier 角色在 L3 出现的深层原因（L2 不需要 verifier，因为 plan 本身就是对照物）。v2 的 `Msg.structured_output` 字段可以读成对这个问题的现代回答：把模型的结构化决策（计划、工具选择、结论）显式放进消息的工作流控制区，让「计划」重新变成可审计的 crossing——想清楚你的系统里计划住在哪，是 multi-agent 设计的第一问。

---

## 6. 与权威实现对照（双快照，2026-08-10 18:3x codeload 现场重抓核验）

2026-08-11 在写作前**重新抓取**双 tarball：main 7,525,108 B（md5 `bb76351534b542c79a2391670a5e126c`）——较早快照（7,524,950 B）**+158 B 漂移**，`diff -rq` 定位漂移仅限 `model/_gemini/_model.py` + `tests/model_gemini_test.py`（gemini 适配器，与本节无关），**材料所用全部锚点零漂移**，`__version__` 仍 2.0.6；v1.0.0 tag 7,702,712 B（md5 `ea8ddb7e636bead9eec37c58bc2bf873`）与早先快照字节尺寸一致（tag 不可变）。下表行号全部取自此次新鲜快照。

| nano L3 部件 | AgentScope 对应物（核验行号） |
|------|------|
| 类型化 Msg（构造期校验） | v2 `message/_base.py:L67` `class Msg(BaseModel)`；字段三组：context（`name` L75 / `content: list[ContentBlock]` L77 / `role` L79 / `id` L81）、metadata（L89/91/93）、**workflow control**（`finished_at` L101 / `finished_reason` L103 / `structured_output` L106 / `error` L112）；`validate_role_content` L117（model_validator）+ 按 role 的块合法断言 `_assert_user_content_blocks` L33-39（user 只许 text/data）/ `_assert_system_content_blocks` L42-48（system 只许 text） |
| 终止即消息数据 | v2 `types/_reply.py:L10-16` `ReplyFinishedReason` StrEnum：`COMPLETED / INTERRUPTED / EXCEED_MAX_ITERS / ERROR`；docstring 明言 *"None until a REPLY_END event is applied"* |
| typed blocks + 状态机 | v2 `message/_block.py`：`TextBlock` L11 / `ToolCallState` L128（五态，迁移图在 `ToolCallBlock` L138 的 docstring 内：pending→asking/allowed/finished，allowed→finished/submitted，submitted→finished）/ `ToolResultState` L185（SUCCESS/ERROR/INTERRUPTED/DENIED/RUNNING，默认 RUNNING）/ `ToolResultBlock` L195；`ToolCallBlock.input` 是**原始 JSON 字符串**（流式累积）——nano 同款 |
| 预算与宽限 | v2 `agent/_config.py:L282` `ReActConfig`：`max_iters=20`（L285-290）/ `structured_output_grace_iters=5`（L293-302）——L1 锚点零漂移 |
| 循环与决策/副作用分离 | v2 `agent/_agent.py:L863` `while True:` + L869 `match next_action:`；`_next_action` L3050（*"Read-only: all side effects are performed by the caller"*）；`Acting/Reasoning/Exit` `agent/_utils.py:L26/32/39` |
| 广播群聊（编排模式） | v1 tag `pipeline/_msghub.py:L11` `class MsgHub`（*"controlled the subscription of the participated agents"*）：`__aenter__` L56-65 重接订阅 + 广播 announcement；`broadcast` L115-123（**含发送者**）；订阅机制在 `agent/_agent_base.py`：`_subscribers` L152 / reply 后 `_broadcast_to_subscribers` L239-251 / `reset_subscribers` L439-447（**L447 排除自己**） |
| 顺序组合子（对照组） | v1 `pipeline/_class.py:L8` `SequentialPipeline` / `pipeline/_functional.py:L7` `sequential_pipeline`（fold：前一级输出 = 后一级唯一输入） |
| v1→v2 契约演化 | v1 `Msg` 是普通类（`message/_message_base.py:L21`，content: str \| blocks，role 仅 assert）；v1 的 blocks 是 **TypedDict**（`message/_message_block.py:L9/L79/L92`，无 state 字段、无运行时强制）→ v2 全部升格为 pydantic 模型 + 状态机 + 构造期校验 |

**nano 与权威实现的差异（为什么它那样选）**：

1. **pydantic vs 手写校验**（L2 差异 1 的升级版）：v2 把 schema 编译进类型系统，构造即校验、单一来源派生文档与客户端；nano 手写校验换可读性。机制同构，工程化选前者。
2. **v1 广播的 nano 镜像是逐字的**（订阅表重接 + 两个广播语义的不对称），但同步化：v1 全异步（`async def observe/broadcast`），nano 回合严格顺序——并发时序问题（乱序、竞态）不在本节范围（§9）。
3. **状态机只跑主路**：v2 的 asking（human-in-the-loop 确认）与 submitted（外部执行）路径声明不演练；`suggested_rules: list[PermissionRule]` 的规则系统简化为 orchestrator 单点授权。本质（agent 不能自推状态）保留，外围（人审、外部分发、拒绝即 finished）留给真实系统。
4. **`finished_reason` 的设置者**：v2 由 REPLY_END 事件应用到 reply 消息；nano 由 orchestrator 在裁决点设置。消息级粒度这一关键属性两侧相同。
5. **planner 的模型是字符级记忆模型**：真托管模型的行为（幻觉子任务、忘记计划）只能 `[TODO: needs key]`——适配器代码就绪，确定性 fallback 是 L1 先例。

**v2 移除 pipeline/msghub 的复验**：18:3x 新鲜 main 快照全树 grep `MsgHub / SequentialPipeline / sequential_pipeline` = **0 类名命中**（仅 middleware docstring 里 "pipeline" 一词的散文用法）——L2 §6 的演化观察（编排层快变、契约层慢变）在四天后的快照上维持。

---

## 7. 费曼自检

**类比：一个公司群 + OA 系统 + 流程审批 + 群里的审计员**。hub 就是公司群：谁在群里发话，所有人自动看到（不用逐个转发——入群那一刻「订阅关系」就接好了）；typed blocks 是 OA 表单：缺字段的单子**根本提交不出去**（不是前台退回，是系统不让你生成）；状态机是审批流：你发起的申请停在「待审批」，**你自己不能把它改成「已办结」**——审批权不在发起人手里；verifier 是群里的审计员：他不干活，但群里每条消息他都看见，最后验收时他只认群里的证据。「他声称做完了」和「群里有他做完的证据」是两回事。
**一句话版**：表单管住「什么话能说」（出生拦截），审批流管住「谁能推进世界」（状态机），群聊管住「谁知道什么」（广播），审计员管住「说的和证据对不对」（verifier）——可靠性不再属于任何一个人，而属于这套接线。

**反例版**（都能用本文实测证伪）：

1. 「点对点把消息都转发一遍，效果就和广播一样」——[2] 实测：认知状态确实相同，但代价是每条 crossing 一次显式发送（4 次额外发送 vs 0）；房间加到 N 个人时这个差是 O(N) 的——「效果一样」和「代价一样」是两回事。
2. 「重试预算按 iid 公式买就够了」——[4] 实测：k=1 时公式承诺 94.1%，实测 76.5%——失败是 sticky 的、修复只在训练过的位置有效，iid 是上界不是等式。

**自检问题**：你能不能向一个只写过 RPC 调用的工程师解释——为什么「审计员在群里」比「审计员被逐个通知」不只是方便，而是**认知状态**的差别？（提示：p2p 世界里审计员的知识 = 别人**记得**转发给它的东西——转发是义务，就会漏；hub 世界里审计员的知识 = 房间的账本——漏不了。前者是 best-effort，后者是 by-construction。）

---

## 8. 思考题

1. **广播的账单**：hub 里每条 crossing 被 N-1 个其他成员观察，观察 = 进 context = token。推导 N 个 agent、M 条 crossing 的 hub 的总观察成本，与 p2p「只递给该递的人」的成本对比；当 N=6、M=50 时差多少倍？这解释了真实框架里 announcement/memory 管理为什么是性能问题而不只是语义问题（联系 nano-vllm-sglang L2：这些重复观察在 KV cache 里长什么样？prefix caching 能救多少？）
2. **verifier 的边界**：子串规则是故意粗糙的。构造一个**能通过**子串检查但语义错误的 answer（提示：answer 是证据的子串 ≠ 证据支持 answer——否定、量词、截断），然后把 verifier 升级成仍然机器可判定的更强规则（提示：L2 的 `extract_answer` 是现成的确定性抽取——「answer == extract(evidence)」比「answer ⊂ evidence」强在哪？什么时候连它也不够、必须上模型？）
3. **submitted 的代价**：v2 的 `allowed→submitted→finished` 路径服务外部执行。在分布式环境里，submitted 引入了哪种 allowed→finished 本地执行不存在的失败模式？谁拥有 submitted→finished 这条迁移？（提示：执行方崩溃在 submit 之后、result 之前——迁移的幂等性、事件溯源、以及为什么 v2 把状态做成 event-driven 而不是字段赋值。）
4. **重试预算买在哪**：[4] 的谱线给出三个位置的 p 与修复训练覆盖（step0/step1 有 critique-repair 训练、answer 没有）。如果只能给**一个**位置配 k=1 重试预算，买在哪期望收益最大？用实测数字算一遍，并据此回答：真实系统里 critique-retry 应该优先铺在哪些 crossing 上？（提示：收益 = p × 修复率；answer 位置 p 最高但修复率近零。）
5. **快层与慢层**：v2 砍掉了 v1 的 pipeline/msghub（快层），保留并加强了 typed Msg + 状态机（慢层）。给以下四项分层并说明判据：消息 schema、传输/广播机制、permission 策略、编排 DSL。（提示：判据 = 「改它的时候，多少下游代码会碎」+ 「它的语义是否随模型能力演化」。）

---

## 9. 局限（诚实清单）

1. **planner 是记忆模型**：幻觉子任务、忘记计划、提前完成里只有最后一种能在本节实测（step0 谱线 final:9/200，有机发生）；前两种需要真托管模型——`HostedBackend` 代码就绪、`[TODO: needs key]`。
2. **同步回合**：v1 全异步；本节的并发时序问题（乱序、竞态、部分广播）不存在，因为回合严格顺序。
3. **状态机只跑主路**：asking（人审）/submitted（外部执行）声明不演练；permission 是 orchestrator 单点，不是规则系统。
4. **verifier 子串规则粗糙**：思考题 2 的方向；生产 verifier 通常是「确定性抽取 + 模型裁决」混合。
5. **观察账本按消息计，不按 token 计**：真实成本在 token（思考题 1）；本节只给结构不给量。
6. **p 估计各 N=200**：±3pp 量级采样噪声；k=0 的复合偏差 6.8pp 在这个噪声尺度内解读。
7. **词表安全化是 char-level 特有**：托管 tokenizer 后端无此问题——声明在代码内（`getattr(backend, "stoi", None)`），不是通用机制。

---

## 10. 交叉引用与下一级

- **阶梯完成**：nano-agentscope L0–L3 全阶（L0 mock ReAct → L1 真模型单循环 → L2 协议层 → L3 类型化契约 × 广播 × 模型入席）。04 轨 sota-deepdive（**harness engineering**）的锚点门槛（agentscope ≥ L2）早已满足，L3 落地后对照材料完整——deepdive 可直接引用本阶四级的实测数据。
- L1：可靠性代数（本节 [4] 是它的协议级版本；sticky 教训原样生效）；critique 通道词表安全化是 L1 CRITIQUE 模板的加固。
- L2：协议骨架（边界校验 → 出生拦截是同一执行点论证的前移；T4「不信 agent 的账本」→ 状态机「agent 不能自推状态」）。
- nano-vllm-sglang L1–L2：广播的每次观察都是 token；重试/重规划烧的是推理引擎省出来的吞吐余量（[7] 的账本在那边结算）。
- 03 轨 RSI：hub 的消息日志 = **多角色结构化轨迹**（角色、契约、裁决俱全），是回流训练数据的 04 侧出口——比 L1 的单 agent traj 多了 verifier 标签，天然可筛「有证据支撑的完成」vs「无证据的宣称」。

---

## 11. 溯源与口径

- **AgentScope 快照（双份，2026-08-10 18:3x 于 codeload.github.com 现场重抓）**：main tarball 7,525,108 B（md5 `bb76351534b542c79a2391670a5e126c`；较同日 00:36 快照 7,524,950 B 漂移 +158 B，`diff -rq` 定位仅 `model/_gemini/_model.py` + `tests/model_gemini_test.py`，材料锚点零漂移；`__version__ = "2.0.6"`）；v1.0.0 tag tarball 7,702,712 B（md5 `ea8ddb7e636bead9eec37c58bc2bf873`，与 00:36 快照尺寸一致，tag 不可变）。§6 全部行号取自这两份 18:3x 快照；canonical 仓库 `github.com/agentscope-ai/agentscope`（原 modelscope/agentscope 301 重定向）。
- **论文**：ReAct arXiv:2210.03629（*ReAct: Synergizing Reasoning and Acting in Language Models*，Yao et al.）与 Plan-and-Solve arXiv:2305.04091（*Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models*，Wang et al.）——双 ID 标题页本轮 08-10 于 arxiv.org 复抓核验，与 L1/L2 录值逐词一致。
- **测量口径**：全程 seeded、无计时行；公开脱敏版于 2026-08-14 完整复跑一次（md5 `8786f888db882b3710f4bed2dbe23595`，120 行），全部断言在代码内（self-check 块）。训练锚点与 L1 逐位同一（93,731 params / loss 0.0218——同 seed 同配方，L3 的 planner 就是 L1 的模型）。环境：Python 3.13.13 / torch 2.13.0 / CPU；跨版本/硬件需重新核验。
- **工具层口径**：L1 的 `list_dir` 冻结清单（六件，2026-08-06 定稿时刻）+ realpath 沙箱 `read_file` 整体 import——L3 **不**重新冻结目录清单，原因 = 模型在 L1 观察串上受训（§1 声明 3）；L1/L2 的冻结清单继续保护各自锚点，L3 进目录不击穿任何上级锚。
- **写作过程修掉的真实 bug**（self-check 当场抓出，§5.4 展开）：① critique 通道 KeyError 'U'——`json.JSONDecodeError` 的 "Unterminated string..." 携带词表外字符回流进 char-LM 的 prompt（L1 seeded 扫描未触发、L3 宽扫描触发），修复 = 按后端词表安全化 critique；② RulePlanner 的 `n_subtask` 双计数（`subtask_msg` 与 `observe_result` 各推进一次），首跑 AttributeError/流程错位抓出，修复 = 单一推进点。两处都是对抗式自检（先跑再交付）的直接收益。
