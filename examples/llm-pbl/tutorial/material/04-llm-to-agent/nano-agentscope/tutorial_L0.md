# nano-agentscope L0 — ReAct 单 agent 玩具闭环

> 目标：用**标准库**写一个最小 ReAct agent，让「思考 → 调工具 → 观察 → 再思考」这个循环真实跑起来。这里的 LLM 是 mock，只用于演示 loop 形态；L1 再接入真实 API。

---

## 1. 为什么需要 ReAct？

大模型能写文案、能推理，但遇到两件事会翻车：

1. **算术不准**：`(15 + 27) * 3` 它可能直接口算错。
2. **信息不新鲜**：它不知道今天的股价、天气，也不知道你私有数据库里的内容。

ReAct（Reasoning + Acting）的思路是：让模型把问题拆成步骤，每步决定「是该继续想，还是调用一个可靠工具」。工具负责精确计算或查外部信息，模型负责编排。这样就把 LLM 的"模糊推理"和外部系统的"精确执行"接在了一起。

本节只做一个**单 agent**、**两个工具**、**一个固定问题**的玩具，但已经包含真实 agent 系统的全部骨架：

- 工具注册表（Tool registry）
- 工具调用解析（Action / Action Input）
- 观察回填（Observation）
- 终止判断（Final Answer 或 max_steps）

---

## 2. 运行

文件：`L0_react_loop.py`，零外部依赖，CPU 即跑。

```bash
$ python3 L0_react_loop.py
```

真实输出：

```
================================================================
nano-agentscope L0 — ReAct loop with a mock LLM
================================================================

⚠️  MOCK LLM: 这不是真实语言模型，只是规则 mock。
    它的唯一作用是展示 ReAct loop 的运转形态。
    L1 会把 mock 替换成真实 LLM API。

Task: What is (15 + 27) * 3?

--- Step 0 ---
  Thought: I should first add 15 and 27.
  Action: add
  Action Input: {"a": 15, "b": 27}
  Observation: 42
--- Step 1 ---
  Thought: Now I multiply 42 by 3.
  Action: multiply
  Action Input: {"a": 42, "b": 3}
  Observation: 126
--- Step 2 ---
  Thought: The calculation is complete; I can answer.
  Final Answer: 126

================================================================
Final Answer: 126
================================================================

✅ self-check passed: answer correct and loop terminated.
```

---

## 3. 代码骨架

### 3.1 Tool 与注册表

```python
class Tool:
    def __init__(self, name: str, fn: Callable, schema: Dict[str, str]):
        self.name = name
        self.fn = fn
        self.schema = schema
```

工具就是 `(name, function, schema)` 三元组。schema 在这里只做参数名检查，真实系统里还要做类型校验、JSON Schema 校验、错误兜底。L1 会扩展。

### 3.2 Mock LLM（L0 专用）

以下为简化展示（源码先把 `observations[-1]` 赋给变量 `last` 再逐条判断，返回文本也是完整的，此处从简）：

```python
class MockLLM:
    def __call__(self, prompt: str) -> str:
        observations = re.findall(r"Observation:\s*(\S+)", prompt)
        if not observations:
            return (
                "Thought: I should first add 15 and 27.\n"
                "Action: add\n"
                'Action Input: {"a": 15, "b": 27}'
            )
        if observations[-1] == "42":
            return "... multiply 42 by 3 ..."
        if observations[-1] == "126":
            return "... Final Answer: 126"
```

⚠️ **再次强调**：这个 mock 只认 `(15 + 27) * 3` 这一条轨迹。它通过正则读取 prompt 里最新的 `Observation`，然后返回硬编码的下一步。换一个问题就会答非所问。L1 会把它换成真实 LLM。

### 3.3 Action 解析

```python
def parse_action(text: str):
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if action_match and input_match:
        args = json.loads(input_match.group(1))
        return action_match.group(1), args
    return None, None
```

真实系统里这一步可能由 LLM 的 function calling 接口完成（OpenAI `tool_calls`、Claude `tool_use` 等），不需要自己正则解析。但 ReAct 论文 [arXiv:2210.03629] 里就是这种文本格式，理解它有助于理解 function calling 的前身。

### 3.4 ReAct loop

```python
for step in range(max_steps):
    response = llm(build_prompt(task, history))
    history.append(response)

    if "Final Answer" in response:
        return answer

    tool_name, args = parse_action(response)
    result = tools[tool_name].run(args)
    history.append(f"Observation: {result}")
```

核心不变量：**agent 永远根据完整历史决定下一步**。history 里既有模型的 Thought/Action，也有工具返回的 Observation。这保证了模型"看到"了之前的执行结果，而不是每次都从头猜。

---

## 4. 与 AgentScope 的对应关系（概念层）

AgentScope 真实仓库：`https://github.com/agentscope-ai/agentscope`（已从 `modelscope/agentscope` 迁移）`[TODO: verify source]`。

| nano 概念 | AgentScope 概念 | 说明 |
|-----------|----------------|------|
| `Tool` | `agentscope.tools.ServiceToolkit` / 工具函数包装 | 真实系统会有 schema 生成、类型校验、错误兜底 `[TODO: verify source]` |
| `ReActAgent.history` | `Msg` 对象列表 / conversation memory | AgentScope 的消息是结构化的 `Msg(name, content, role)`，不只是字符串 `[TODO: verify source]` |
| `parse_action` | LLM function calling 或 ReAct parser | 真实系统可能直接调用模型提供的 tool_calls 字段 `[TODO: verify source]` |
| `max_steps` | agent 运行时的终止条件配置 | 真实系统还会配超时、错误重试、对话轮数上限 `[TODO: verify source]` |

L0 只到概念映射；源码级对应（具体文件路径、类名、API）留到 L3 再核对。

---

## 5. 费曼：讲给外行听

**类比：叫外卖做算术题**

想象你有一个很聪明但不擅长心算的朋友（LLM），和一个算盘很准的室友（工具）。你问朋友：「(15 + 27) * 3 等于多少？」

- 朋友想了想说：「我先算 15+27」。但他心算不准，于是**拿起电话（Action）打给室友**，报出数字 15 和 27。
- 室友拨完算盘回话：「42」。这就是 **Observation**。
- 朋友听到 42，又说：「那我再算 42*3」，再次打电话问室友。
- 室友回：「126」。
- 朋友说：「好了，答案是 126。」

ReAct 就是这个过程的自动化：朋友负责**拆步骤和决策**，室友负责**精确执行**。二者通过「请求 → 结果」这条线反复沟通，直到朋友觉得可以给出最终答案。

---

## 6. 思考题

1. 如果 MockLLM 第二步返回 `Action: divide`，但工具注册表里没有 `divide`，代码会怎么处理？这个兜底在真实系统里为什么重要？
2. 为什么 history 里必须同时保存 Thought、Action 和 Observation，而不是只保存 Observation？
3. 如果 `max_steps` 设成 1，程序会输出什么？这说明了 agent 系统的什么工程问题？

---

## 7. 反例：ReAct 不是万能药

**反例 1：工具返回错误信息时，agent 可能越陷越深**

如果 `add` 工具返回 `"error: NaN"`，MockLLM 不认识这个 Observation，就会直接回答 `"I don't know"`。真实 LLM 可能更糟糕——它会继续调用别的工具试图"修复"，结果产生更多无效调用。这说明 agent 的可靠性不仅取决于模型，还取决于**工具错误处理、重试、终止条件**的设计。

**反例 2：没有工具时，ReAct 退化成纯文本编造**

如果你把工具表清空，agent 没有任何可调用的工具，但 loop 仍然会跑。MockLLM 会因为没有 Observation 而反复输出第一步。真实 LLM 则可能在没有任何工具的情况下继续"假装"调用工具，或者编造 Observation。这提醒我们：**工具定义和模型行为必须配对**。

---

## 8. 下一步 L1

L1 会把这个 mock LLM 替换成真实 API（例如 DashScope / OpenAI），并把工具换成真实可调用函数（比如查天气、调用本地数据库）。届时我们要解决三个新问题：

1. **Prompt 工程**：如何让真实 LLM 稳定输出 `Action:` / `Action Input:` 格式？
2. **错误兜底**：API 超时、JSON 解析失败、工具抛异常时怎么办？
3. **成本控制**：每多一轮循环就多一次 API 调用，如何设置合理的 `max_steps` 和 early stop？

---

## 9. 溯源

- ReAct 论文：`arXiv:2210.03629`，*ReAct: Synergizing Reasoning and Acting in Language Models*。
- AgentScope 仓库：`https://github.com/agentscope-ai/agentscope`（canonical URL，原 `modelscope/agentscope` 已 301 重定向）`[TODO: verify source]`。
- 源码级对应：L0 未给出具体文件路径，L3 再对照 AgentScope 的 message / pipeline / tool 实现 `[TODO: verify source]`。
