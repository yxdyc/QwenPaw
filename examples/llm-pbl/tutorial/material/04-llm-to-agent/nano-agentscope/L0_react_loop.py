"""
nano-agentscope L0 — ReAct single-agent loop with a mock LLM
==============================================================
目标：用最少代码展示 ReAct 的核心闭环——
      Thought → Action(tool call) → Observation → Thought → ... → Final Answer
依赖：Python 标准库（json / re / typing），零外部包，CPU 即跑。

⚠️  重要声明：下面的 LLM 是一个 hard-coded 的规则 mock，
    只用于演示 ReAct loop 的形态。它不会理解任意问题。
    真实 LLM API 调用将在 L1 中接入。
"""

import json
import re
from typing import Callable, Dict, List


class Tool:
    """一个最小工具：名字、可调用函数、参数 schema。"""

    def __init__(self, name: str, fn: Callable, schema: Dict[str, str]):
        self.name = name
        self.fn = fn
        self.schema = schema

    def run(self, kwargs: Dict) -> str:
        # 极简参数校验：只检查 key 是否齐全
        missing = [k for k in self.schema if k not in kwargs]
        if missing:
            raise ValueError(f"missing args for {self.name}: {missing}")
        return str(self.fn(**kwargs))


class MockLLM:
    """
    L0 专用 mock LLM：它根据 prompt 里已经出现的 Observation
    决定下一步该说什么。它不是真实语言模型，只能走预设的
    (15 + 27) * 3 这条轨迹。
    """

    def __call__(self, prompt: str) -> str:
        # 从 prompt 里提取所有 Observation 值
        observations = re.findall(r"Observation:\s*(\S+)", prompt)

        if not observations:
            return (
                "Thought: I should first add 15 and 27.\n"
                "Action: add\n"
                'Action Input: {"a": 15, "b": 27}'
            )

        last = observations[-1]
        if last == "42":
            return (
                "Thought: Now I multiply 42 by 3.\n"
                "Action: multiply\n"
                'Action Input: {"a": 42, "b": 3}'
            )
        if last == "126":
            return (
                "Thought: The calculation is complete; I can answer.\n"
                "Final Answer: 126"
            )

        return (
            "Thought: I don't recognize this intermediate result.\n"
            "Final Answer: I don't know."
        )


def parse_action(text: str):
    """
    从 LLM 输出中解析工具调用。
    格式：
        Action: <tool_name>
        Action Input: {<json>}
    返回 (tool_name, args) 或 (None, None)。
    """
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if action_match and input_match:
        try:
            args = json.loads(input_match.group(1))
            return action_match.group(1), args
        except json.JSONDecodeError:
            return None, None
    return None, None


class ReActAgent:
    """最简 ReAct agent：维护历史，循环调用 LLM 与工具。"""

    def __init__(self, llm: Callable[[str], str], tools: Dict[str, Tool], max_steps: int = 6):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: List[str] = []

    def _build_prompt(self, task: str) -> str:
        lines = [f"Task: {task}"]
        lines.append(f"You can use tools: {', '.join(self.tools.keys())}.\n")
        lines.extend(self.history)
        lines.append("What do you do next?")
        return "\n".join(lines)

    def run(self, task: str) -> str:
        print(f"Task: {task}\n")
        for step in range(self.max_steps):
            prompt = self._build_prompt(task)
            response = self.llm(prompt)
            self.history.append(f"[Step {step}] {response}")

            print(f"--- Step {step} ---")
            for line in response.splitlines():
                print(f"  {line}")

            if "Final Answer" in response:
                return response.split("Final Answer:", 1)[-1].strip()

            tool_name, args = parse_action(response)
            if tool_name and tool_name in self.tools:
                try:
                    result = self.tools[tool_name].run(args)
                    obs = f"Observation: {result}"
                except Exception as e:
                    obs = f"Observation: error - {e}"
            elif tool_name:
                obs = f"Observation: tool '{tool_name}' not found"
            else:
                obs = "Observation: no action detected"

            self.history.append(obs)
            print(f"  {obs}")

        return "Agent reached max steps without a final answer."


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("nano-agentscope L0 — ReAct loop with a mock LLM")
    print("=" * 64)
    print("\n⚠️  MOCK LLM: 这不是真实语言模型，只是规则 mock。")
    print("    它的唯一作用是展示 ReAct loop 的运转形态。")
    print("    L1 会把 mock 替换成真实 LLM API。\n")

    tools = {
        "add": Tool("add", add, {"a": "int", "b": "int"}),
        "multiply": Tool("multiply", multiply, {"a": "int", "b": "int"}),
    }

    agent = ReActAgent(MockLLM(), tools, max_steps=6)
    task = "What is (15 + 27) * 3?"
    answer = agent.run(task)

    print("\n" + "=" * 64)
    print(f"Final Answer: {answer}")
    print("=" * 64)

    # 简单自检查：答案必须正确，步数不能超
    assert answer == "126", f"expected 126, got {answer!r}"
    assert len(agent.history) <= 12, "too many steps"
    print("\n✅ self-check passed: answer correct and loop terminated.")


if __name__ == "__main__":
    main()
