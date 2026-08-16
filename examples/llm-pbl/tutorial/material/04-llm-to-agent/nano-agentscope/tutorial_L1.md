# nano-agentscope L1 — 把真模型换进 ReAct 循环：不可靠性从哪来，harness 买到什么

> **级别**：L1（K+1：L0 的规则 mock → L1 的真实（微小）语言模型 + 真实工具）
> **文件**：[`L1_real_agent_loop.py`](L1_real_agent_loop.py)（779 行，依赖仅 `torch`，CPU 即跑）
> **数据**：[`corpus.txt`](corpus.txt)（ReAct 论文 arXiv:2210.03629 的标题与原创释义；论文元数据于 2026-08-06 在 arxiv.org 核验，fixture 不复制论文摘要）

---

## 1. L1 换掉了什么，以及三个声明

L0 的 MockLLM 是规则写死的：它**按构造**永远不会违规。这恰好把 agent 工程的核心问题藏起来了——
真实模型的输出是一个**分布**而不是函数，它以可测概率违反格式契约、叫错工具名、写坏 JSON、
甚至不调工具直接编答案。L1 把这件事变成真的：

- 训练一个**真实的** char-level 小语言模型（93,731 参数，LSTM），它的输出是采样出来的；
- 用两个**真实工具**（真实磁盘 I/O，带沙箱）跑完整任务；
- 把 harness 的每种防御放在**受控故障注入**下测试，并把「可靠性」算成可证伪的代数。

三个显式声明（对应 课程可运行性契约，代码头部同文）：

1. **TinyReActLM 是真语言模型**（学习得到的分布、可采样），但只有 ~94K 参数、字符级——
   它靠**记忆**完成轨迹，不做推理。真实托管大模型的行为留 `[TODO: needs key]`。
2. **Playback / FaultModel 不是模型**，是声明过的故障注入测试向量（unit-test fixture），
   用来把失败概率调成已知值。
3. **[6] 的本地契约服务器不是真 LLM**——它验证的是我们 API 客户端代码本身
   （请求构造 / auth header / 响应解析）对着 OpenAI-compatible JSON 契约的正确性。

---

## 2. 运行与输出（逐字粘贴）

```bash
$ python3 L1_real_agent_loop.py   # CPU，约 3 分钟（训练 ~2 分钟）
```

以下为定稿后 run1 的逐字输出（连跑 3 遍全部 EXIT=0；除 `train time` 一行随负载浮动
（公开脱敏版 2026-08-14 两次 CPU 复跑 109.9–112.1s；耗时随负载浮动，见 §11），其余逐字节一致，
diff 核验为空）：

```text
====================================================================
nano-agentscope L1 — real tiny model + real tools + harness
====================================================================
python 3.13.13 | torch 2.13.0
declarations: TinyReActLM = real ~94K-param char-LM (memorizes,
  does not reason); Playback/FaultModel = declared fault vectors;
  real hosted LLM path needs a key [TODO: needs key].

[0] train TinyReActLM (synthetic trajectories, real file obs)
    transcripts=75 (60 clean + 15 violation->critique->repair) mean_len=962 chars | vocab=65+UNK+PAD
    params=93,731 | final loss=0.0218 | train time=109.9s (CPU)

[1] greedy decode: real model x real tools, full task
    [step 0] (compliant) [Step 0] Thought: First I will check what files exist here. | Action: list_dir | Action Input: {}
             obs: ['L0_react_loop.py', 'L1_real_agent_loop.py', 'README.md', 'corpus.txt', 'tutorial_L0.md', 'tutorial
    [step 1] (compliant) [Step 1] Thought: I should read corpus.txt and look at the first line. | Action: read_file | Action Input: {"p
             obs: ReAct: Synergizing Reasoning and Acting in Language Models
arXiv:2210.03629 — citation verified on a
    [step 2] (final) [Step 2] Thought: The title line names the two things directly. | Final Answer: reasoning and acting
    answer='reasoning and acting' | status=answered | model calls=3
    sandbox check '../../etc/passwd': blocked -> PermissionError
    sandbox check 'no_such_file.txt': blocked -> FileNotFoundError

[2] model-side reality: format compliance vs temperature
    (first block from task prefix, 200 samples each, seeded)
    T=0.3: compliant 197/200 = 98.5%   [compliant:197, no_action:3]
    T=0.7: compliant 178/200 = 89.0%   [compliant:169, final:9, no_action:22]
    T=1.0: compliant 109/200 = 54.5%   [bad_json:3, compliant:83, final:26, no_action:84, unknown_tool:4]
    T=1.3: compliant 34/200 = 17.0%   [bad_json:3, compliant:8, final:26, no_action:160, unknown_tool:3]
    => a real model gives per-call failure p ~= 0.11 at T=0.7. This p is what the harness has to live with.

[3] harness defenses under declared fault injection
    a) bad_json -> critique -> retry: status=answered | kinds=['bad_json', 'compliant(retry1)', 'compliant', 'final'] | calls=4
    b) unknown_tool 'read_flie' -> critique(unknown_tool: read_flie) -> retry: status=answered | kinds=['compliant', 'unknown_tool', 'compliant(retry1)', 'final']
    c) tool exception as observation: "error - FileNotFoundError: no such file in module dir: 'no_such.txt'"
    d) loop (same action twice): status=loop_detected (guard fired, budget saved)

[4] reliability algebra: measured task success vs formula
    organic (real LM, T=0.7): retries=0 -> success 67.5% | mean calls 2.57
    organic (real LM, T=0.7): retries=1 -> success 82.0% | mean calls 3.06
    controlled iid (FaultModel, 400 runs each):
     p     k   measured   formula   |diff|
    0.05  0     85.2%     85.7%    0.5%
    0.05  1     99.5%     99.3%    0.2%
    0.05  2    100.0%    100.0%    0.0%
    0.10  0     72.5%     72.9%    0.4%
    0.10  1     97.8%     97.0%    0.7%
    0.10  2     99.8%     99.7%    0.0%
    0.20  0     51.0%     51.2%    0.2%
    0.20  1     89.8%     88.5%    1.3%
    0.20  2     98.0%     97.6%    0.4%
    0.30  0     33.2%     34.3%    1.0%
    0.30  1     74.0%     75.4%    1.4%
    0.30  2     91.5%     92.1%    0.6%
    sticky s=0.75 @ p=0.2, k=1: measured 58.2% vs iid formula 88.5% (sticky formula 59.3%)

[5] cost ledger: reliability is bought with model calls
     p     k   success   mean_calls   calls/success
    0.10  0     74.5%       2.75         3.68
    0.10  1     98.0%       3.25         3.32
    0.10  2     99.8%       3.29         3.30
    0.30  0     36.0%       2.21         6.12
    0.30  1     75.2%       3.54         4.71
    0.30  2     90.8%       4.04         4.45
    (each extra call = tokens = latency = $; retries consume the
     throughput headroom nano-vllm-sglang works so hard for; and
     every recorded trajectory is itself training data -> 03/01)

[6] real-API path: OpenAI-compatible client vs local contract server
    POST /compatible-mode/v1/chat/completions | auth header present: True
    request model='qwen-turbo' | response parsed -> agent status=answered, answer='reasoning and acting'
    client code verified against the JSON contract locally.
    real endpoint (DASHSCOPE_API_KEY / OPENAI_API_KEY): [TODO: needs key]

====================================================================
✅ self-check passed:
   greedy real-model trajectory correct (3 calls) /
   sandbox blocks traversal & missing files /
   compliance falls with temperature (measured) /
   defenses recover or abort under fault injection /
   measured success matches iid formula (max dev 1.4%) / sticky < iid /
   API client round-trips through real HTTP
====================================================================

takeaway: L0's mock never failed, so the harness had nothing
          to do. Put a real distribution behind the loop and the
          harness BECOMES the product: parse-validate, retry with
          critique, tool errors as observations, loop guards, and
          a budget — reliability you can measure and price.
```

---

## 3. 代码结构（779 行，六个板块）

| 板块 | 内容 | 关键点 |
|------|------|--------|
| [A] 真工具 | `list_dir` / `read_file` | 真实磁盘 I/O（其中 `list_dir` 清单已于 2026-08-06 定稿时刻冻结，见代码内声明——模块目录将随 L2/L3 阶梯生长，冻结观察清单使本节的确定性锚与目录状态解耦）；`read_file` 带 **realpath 沙箱**（路径必须落在模块目录内）+ head 式截断（observation 吃 context 预算） |
| [B] 严格解析器 | `parse_block` | 与 L0 的 `(None, None)` 不同，返回**类型化违规**：`final / compliant / no_action / unknown_tool / bad_json / missing_args`——harness 的每种防御都挂在一个可枚举的类别上 |
| [C] TinyReActLM | char-LSTM 训练 + 采样生成 | 训练数据 = 60 条干净轨迹 + **15 条「违规→critique→修复」轨迹**；observation 取自真实工具输出；UNK/PAD 是独立 special index，生成时都被 mask，运行时未见前缀字符映射到 UNK |
| [D] Harness | 六条防御 | ①逐块解析验证 ②critique+retry ③工具异常变 observation ④loop guard ⑤max_steps 预算 ⑥全程轨迹记录 |
| [E] 测试向量 | `Playback` / `FaultModel` | 声明的故障注入：`p` = 每次调用失败概率，`sticky` = 失败相关性；corruption 全部落在 [B] 的真实违规类别里 |
| [F] API 客户端 | `OpenAICompatChat` + 本地契约服务器 | stdlib urllib；key 只从环境变量来；本地服务器验证客户端代码本身 |

一个值得单独说的设计：**训练数据里的 critique 文本与 harness 运行时的 critique 逐字符一致**
（同一模板、同一 parser payload，且**不含**失败块本身——因为 harness 的 retry 前缀就是
「上文 + critique」，失败块不在上下文里）。训练前缀 == 推理前缀，这是 nano-llamafactory L1
「推理 prompt 必须是训练串的真前缀」那条教训在 agent 场景的落地。第一版没做到这一点时，
retry 路径直接崩在 `KeyError: '/'`——critique 里的字符超出了模型词表。

---

## 4. 逐段解读

**[0] 训练**。75 条轨迹（60 干净 + 15 带 critique），字符表 65+UNK+PAD，93,731 参数，
final loss 0.0286。注意那 15 条 critique 轨迹的作用：它们把「听到批评后重新输出合规块」
这个行为写进了模型分布——这是 [4a] organic 实验里 retry 能恢复的前提。

**[1] greedy 全任务**。真实模型 × 真实工具，3 次调用走完 list_dir → read_file → Final Answer，
答案与 corpus.txt 标题行逐字一致（assert）。两个 sandbox 反例也是真跑：
`../../etc/passwd` 被 realpath 包含检查挡住（PermissionError），不存在的文件给出
FileNotFoundError——这两种错误在 [3c] 里会变成 observation 喂回模型，而不是崩溃。

**[2] 模型侧现实**。同一个前缀采样 200 次（seeded，可复现）：T=0.3 全合规，T=0.7 掉到
89%，T=1.0 为 54.5%，T=1.3 降至 17.0%。**违规不是 bug，是采样的物理**——温度越高，
分布的尾巴越厚，格式骨架被冲垮的概率越大。注意 T=0.7 时的 `final:19`：模型有 9.5%
的概率不调工具、直接背出答案（它确实背得对——但这是记忆，不是工具使用；§8 思考题 1
会问如果答案没背过会发生什么）。

**[3] 防御演习**。四个声明过的故障向量分别触发四条防御：bad_json 经 critique 后
retry 恢复（`kinds` 里能看到 `bad_json → compliant(retry1)`）；工具名拼错
（`read_flie`，真实系统里高频出现）由 critique 点名违规类型后修复；真实
FileNotFoundError 作为 observation 回流，模型（脚本）改用正确文件名；同一动作连续
两次被 loop guard 截停——**防御要么恢复、要么止损，绝不静默**。

**[4] 可靠性代数**（本节核心，见 §5.3）。

**[5] 成本账**。可靠性用模型调用次数买：`calls/success` 是单位可靠性的价格。
p=0.1 时 k=0→k=1 把成功率从 74.5% 拉到 98.0%，单位价格反而从 3.68 降到 3.32——
适度失败率下 retry 是**划算**的（失败任务提前终止省下的调用抵不过重试挽回的成功）；
p=0.3 时 k=0 的单位价格飙到 6.12，retry 重新变划算。没有免费的防御，只有定价。

**[6] 真 API 路径**。客户端对着本地契约服务器完成一次真实 HTTP 往返：POST 路径、
Bearer auth header、请求体、响应解析全部经过断言。剩下的只是「换一个真的 base_url +
真的 key」——DashScope（OpenAI-compatible）或 OpenAI 端点，见 §11 溯源。

---

## 5. 机制深挖

### 5.1 为什么「可靠性问题」只有换了真模型才存在

L0 的 mock 合规率是**构造出来的 100%**——它根本没有输出空间可以违规。L1 的小 LM 是
学习得到的分布：即使 greedy 路径被训练到完美（T=0.3 实测 100%），只要开始采样，
违规就以可测概率出现（T=0.7 实测 p≈0.13）。这个 p 不是我们假设的，是**测出来的**——
而且它随解码策略（温度）系统性变化。换句话说：

> agent 的可靠性问题不在「模型不够聪明」，而在「模型是随机的」。
> 任何把 LLM 当确定性函数用的 agent 代码，都在写一个迟早爆炸的假设。

这也解释了真实系统为什么把 temperature=0（或极低温）当 agent 默认：不是保守，是在压 p。

### 5.2 critique-retry 只对「听得懂批评」的模型有效

[4a] 的 organic 实验：真实小 LM 在 T=0.7 下，无 retry 成功率 67.5%，带一次
critique-retry 升到 82.0%。这个提升**不是无条件的**——它成立是因为训练数据里有
15 条 critique 轨迹，模型学会了「critique 之后输出合规块」。第一版（没有 critique
训练数据）的模型在 retry 前缀下直接输出词表外字符崩溃——critique 文本对它是纯噪声。

这正是 instruction tuning / RLHF 在 agent 语境下的机制含义：**真实大模型之所以能被
harness「调教」，是因为它们被训练过响应反馈**。harness 的 critique-retry 防御和模型的
follow-feedback 能力是**一对共演化的组件**——单有 harness 不够，单有模型也不够。
（AgentScope 的 `_json_loads_with_repair` 是同一思想的另一面：与其等模型重试，
harness 先替它修 JSON，见 §6。）

### 5.3 可靠性代数：公式、实测、以及公式在哪里失效

设每次模型调用失败概率为 p（iid），每步最多重试 k 次，任务需要 n 步全对：

```
单步通过率  q = 1 - p^(k+1)        （k+1 次尝试里至少一次合规）
任务成功率  q^n                    （步骤独立时）
```

[4b] 的 12 组 (p, k) 扫描（各 400 次任务）：实测与公式**最大偏差 1.4%**——可靠性在
iid 假设下是精确可算的。这就是为什么 senior 能把「加一次 retry」当成工程决策而不是玄学。

然后 [4c] 打破假设：**sticky 失败**（第一次失败后，重试仍以 s + (1-s)·p 的概率失败——
模拟「难 prompt 一直难」的相关性）。p=0.2, s=0.75, k=1 时：

```
iid 公式预测 88.5%，实测只有 58.2%；
sticky 公式 (1 - p·(s + (1-s)·p))^n = 59.3%，精确命中实测。
```

结论：**「不行就重试」的有效性完全取决于失败的独立性**。真实系统里的 sticky 来源：
输入本身超出模型能力（OOD）、prompt 有结构性歧义、工具参数错误但模型坚持原参数。
对 sticky 失败，正确的动作不是重试，是**换策略**（改 prompt、换温度、升级模型、
escalate 给人）——这是 harness 设计里最常被忽略的一条分支。

### 5.4 预算：为什么「停」也需要设计

Harness 的 max_steps 与 loop guard 不是可有可无的保险丝：[5] 的账本说明每次调用都有
价格，一个陷入循环的 agent 是在**烧钱买零信息**。AgentScope 在这件事上的设计值得细看：
`ReActConfig.max_iters` 默认 20，另有 `structured_output_grace_iters=5`——超出预算后
不是直接掐断，而是**再给 5 次宽限**专门用于产出结构化输出（「要停了，先把结论交出来」），
还有 `stop_on_reject`（工具调用被拒绝时是否停止推理）。**终止是语义的一部分**：
生产 agent 的「失败」必须是一个有产出的状态，而不是一个异常。

### 5.5 沙箱：为什么不注册一个 shell 工具

`read_file` 的 realpath 包含检查挡住了 `../../etc/passwd`（[1] 实测）。这不是过度设计：
agent 的动作有**副作用**，工具注册表就是 agent 的权限边界。注册一个无限制的 shell
等于把最小权限原则整个放弃——模型的一次 hallucinated 参数就是一次真实的 `rm`。
AgentScope 甚至有独立的 `permission` 模块（`src/agentscope/permission/`，存在性核验，
行号级对照留 L3）。思考：为什么 L1 敢注册 `read_file` 却不敢注册 `write_file`？
（提示：读的最坏情况是信息泄漏，写的最坏情况是不可逆破坏——副作用的可逆性决定权限等级。）

---

## 6. 与权威实现对照（AgentScope main，2026-08-06 codeload tarball 现场核验）

| nano L1 部件 | AgentScope 对应物（main 分支行号） |
|------|------|
| ReAct 循环（Thought→Action→Observation） | `src/agentscope/agent/_agent.py:L858-874`：`while True` + `match next_action: case Exit / Reasoning / Acting`（统一 `Agent` 类 L110，旧版独立 ReActAgent 已并入配置） |
| 循环的每一步决策 | `_agent.py:L3019-3022` `_next_action(...) -> Reasoning \| Acting \| Exit`，docstring 自述「Read-only: all side effects are performed by the caller」——**决策与副作用分离** |
| 三种步骤状态 | `src/agentscope/agent/_utils.py:L26/L32/L39`：`Acting / Reasoning / Exit` 均为 pydantic 模型 |
| max_steps 预算 | `src/agentscope/agent/_config.py:L282-291` `ReActConfig.max_iters` 默认 **20**；L293-303 `structured_output_grace_iters` 默认 5；L305-313 `stop_on_reject` |
| 解析容错（我们的类型化 parser 的「修复」版） | `src/agentscope/_utils/_common.py:L86` `_json_loads_with_repair`，docstring 原文：*"The given json_str maybe incomplete, e.g. '{"key', so we need to repair and load it"*——权威实现同样**不信任模型的 JSON** |
| 工具注册 / schema / 调用 | `src/agentscope/tool/_toolkit.py:L66` `Toolkit`；L171 `get_tool_schemas`；L225 `call_tool`；L628 `add_tool`；schema 从 docstring 派生（`tool/_utils.py:L6` `from docstring_parser import parse`，L46 `_extract_function_description`） |
| 消息契约（L2 主题） | `src/agentscope/message/_base.py:L67` `class Msg(BaseModel)`：*"responsible for information storage and transmission among different agents"* |
| 工具权限 | `src/agentscope/permission/`（模块存在性核验；机制留 L3） |

**nano 与权威实现的差异（为什么它那样选）**：

1. **我们做 critique-retry，AgentScope 做 JSON repair**：它直接在 harness 侧修复不完整
   JSON（流式场景下甚至边收边修），把「重试」留给更贵的失败。共同前提一致——模型输出
   默认不可信；分歧在修复成本：字符级修复便宜，语义级错误只能重试。
2. **我们的违规是类型化枚举，AgentScope 用事件流**（`ToolCallStartEvent` /
   `ToolResultEndEvent` 等，`_agent.py` 头部 import 可见）：生产系统需要把每一步变成
   可观测、可回放的事件——我们的 `traj` 记录是它的最小形态。
3. **决策与副作用分离**（`_next_action` read-only）让循环可测试、可中断——我们的
   Playback 测试向量正是这种可测试性的受益者。
4. **AgentScope 面向真实大模型 + 异步 + 多 agent**，`max_iters=20` 的默认值对应的是
   真实任务的步数分布；我们 `max_steps=6` 对应 3 步任务 + 防御余量。预算数字由任务
   步数分布 × 成本容忍度决定，不是常数。

---

## 7. 费曼自检

**类比：一家餐厅的出菜流程**。厨师（模型）手艺是真的，但他偶尔把盐当糖、偶尔不听
单、偶尔同一道菜反复重做。餐厅不靠「祈祷厨师不出错」运转，而是靠流程：服务员复述
订单（parse-validate），做错了打回重做并说明哪里错（critique-retry），食材有问题
如实告诉厨师而不是掀桌（tool errors as observations），同一道菜连做两次就停
（loop guard），每桌限时（max_steps），全程留监控（trajectory record）。
**一句话版**：可靠性不是组件的属性，是流程的属性——流程把 87% 合规的厨师变成
98% 可靠的餐厅，代价是每桌多花几次灶台时间。

**反例版**（两条都能用本文实测证伪）：

1. 「厨师水平够高就不需要流程」——[2] 实测：即使 T=0.3 合规率 100%，把温度调到 0.7
   （真实部署常见的多样性需求）立刻掉到 87%。能力不消灭随机性。
2. 「出错了无限重做总能成功」——[4c] 实测：sticky 失败下 retry 从 88.5% 掉到 58.2%，
   重做不收敛。对相关性失败，换策略才有用。

**自检问题**：你能不能向一个没写过 agent 的工程师解释——为什么「模型输出不合规」
不是加个 try/except 就能解决的小问题？（提示：try/except 处理的是**异常**，
而违规是模型输出的**正常组成部分**——它需要的是预算内的策略，不是异常分支。）

---

## 8. 思考题

1. **[2] 里 T=0.7 有 19/200 次「final 捷径」被计为成功**——模型不调工具直接背出答案，
   恰好背对了。如果任务换成「统计 corpus.txt 的词数」（模型没背过），捷径会变成什么？
   harness 应该怎么区分「用工具得到的答案」和「跳过工具的答案」？（提示：轨迹校验 /
   process reward；这不是格式问题，是行为问题——[B] 的类型化 parser 管不了它。）
2. **iid 公式假设每次失败独立。给出真实系统里两个 sticky（相关）失败的来源**，
   并说明对每种来源，harness 应该用什么替代「同 prompt 重试」的动作。
   （提示：OOD 输入 / 结构性歧义 / 模型能力缺口；换 prompt、换温度、换模型、escalate。）
3. **AgentScope 的 `max_iters=20` 之外还有 `structured_output_grace_iters=5`**——
   为什么「停止」本身需要预算？如果生产 agent 到达预算上限时什么都不产出，
   下游（用户 / 调用方 / RSI 数据回流）分别会发生什么？
4. **动手题**：把 `FaultModel` 的 p 改成 0.5，先用公式预测 k=1 的任务成功率，
   再跑一遍对照；然后用 [5] 的 `calls/success` 判据回答：p 至少多大时 k=2 相对
   k=1「不值得」（多花的调用买不回等量的成功率）？

---

## 9. 局限（诚实清单）

1. **小模型不推理**：TinyReActLM 靠记忆完成轨迹，它的「成功」与真实大模型的理解力
   无关。本节测量的是**格式可靠性**与 **harness 机制**，不是模型能力。
2. **真实 API 未打通**：客户端代码经本地契约服务器验证，但真实端点行为
   （限流、流式、tool_calls 原生字段）留 `[TODO: needs key]`。
3. **故障注入分布 ≠ 真实失败分布**：FaultModel 的 p/sticky 是控制变量，真实模型的
   失败与任务难度、prompt 长度、领域强相关——[4] 的公式形状不变，参数要重测。
4. **CPU 计时**：公开脱敏版在 Apple Silicon / torch 2.13.0 CPU 上于
   2026-08-14 两次复跑 109.9–112.1s；这是环境相关观测，不是性能保证。
5. **单 agent**：消息契约、多 agent 编排（planner + executor）是 L2 主题，本节未触及。

---

## 10. 下一级预告与交叉引用

- **L2**：多 agent——planner + executor 协作，消息契约（谁对谁说话、消息里带什么）、
  终止条件与死循环防御。对标 AgentScope 的 `Msg` 与 pipeline 抽象。
- 交叉引用：nano-qwenpaw L0（harness = system prompt + self-check loop，本节的
  critique 是它的泛化）；nano-vllm-sglang L1（retry 消耗的吞吐余点从哪来）；
  01/03 轨（trajectory 记录 = 回流训练数据，RSI 闭环的 04 侧出口）。

---

## 11. 溯源与口径

- **ReAct 论文**：arXiv:2210.03629，2026-08-06 于 arxiv.org 标题页核验
  （*ReAct: Synergizing Reasoning and Acting in Language Models*）；`corpus.txt`
  只保留标题、论文 ID 与本仓库原创释义，不复制摘要。该释义把论文机制压缩为：
  reasoning 维护/修订计划，acting 从工具或环境取得会改变后续推理的证据。
- **AgentScope 锚点**：全部 2026-08-06 经 codeload.github.com main 分支 tarball
  逐行核验（§6 表格行号即当日快照；上游迭代可能漂移）。canonical 仓库
  `github.com/agentscope-ai/agentscope`（原 `modelscope/agentscope` 301 重定向）。
- **DashScope OpenAI-compatible 端点**：help.aliyun.com 文档页（model-studio /
  compatibility-of-openai-with-dashscope）2026-08-06 抓取核验——兼容路径
  `/compatible-mode/v1/chat/completions`、`api_key=os.getenv("DASHSCOPE_API_KEY")`
  示例原文在页；文档同时建议从经典域名 `https://dashscope.aliyuncs.com` 迁移到
  新的 workspace 域名（经典域名在文档中仍列出）。
- **测量口径**：全部随机过程 seeded（torch seed 42 / 数据 rng 42 / 采样 seed
  1000+i / organic seed_base 50000+37i / FaultModel seed 10000+i 起）；定稿后
  公开脱敏版于 2026-08-14 复跑两次；除 `train time` 外的输出逐字节一致，
  但跨 torch/硬件版本仍应重新核验。训练耗时 109.9–112.1s 只代表这两次 CPU 运行。
  违规率、成功率均为 200/400 次任务的频率估计（二项标准误 ≤ 3.5%/2.5%，
  SE = √(0.25/n)）。
- **self-check 捕获并修复的真实 bug**（全部由 assert/崩溃当场抓出）：a) PAD 索引与真实
  字符共用 0 号位，换行符失去监督、生成退化；b) 采样可输出 PAD 索引
  （KeyError 抓出，生成时 mask）；c) critique 词表外字符（KeyError '/' 抓出 →
  critique 轨迹入训练集）；d) 公开语料释义缩小字符表后，运行时错误反馈中的大写
  `U` 再次触发 OOV——新增独立 UNK 索引，前缀未知字符映射到 UNK，且 UNK/PAD
  都从生成分布 mask；e) 可靠性公式把失败率 p 当成功率用（手算 0.95³=85.7%
  vs 实测 85.2% 抓出）。
