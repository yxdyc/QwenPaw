"""
nano-qwenpaw L0 — agent harness: system prompt + self-check loop
===============================================================
目标：用最少代码展示 harness 的核心价值——
      同一个 LLM，套上「system prompt + 输出自检」后，行为更可控。
依赖：Python 标准库（re / typing），零外部包，CPU 即跑。

⚠️  重要声明：下面的 LLM 是一个 hard-coded 的规则 mock，
    只用于演示 harness 的形态。它不会理解任意问题。
    真实 LLM API 调用将在 L1 中接入。
"""

import re
from typing import Callable, List, Tuple


class MockLLM:
    """
    L0 专用 mock LLM。
    它会根据 prompt 里有没有 system prompt / 自检反馈，
    返回不同质量的回答，用来展示 harness 的作用。
    """

    def __call__(self, prompt: str) -> str:
        has_system = "system:" in prompt.lower()
        has_critique = "critique:" in prompt.lower()

        if has_critique:
            # 收到 harness 的 critique 后，给出符合格式要求的答案
            return (
                "Step 1: A common year has 365 days.\n"
                "Step 2: Each day has 24 hours.\n"
                "Step 3: 365 * 24 = 8760 hours.\n"
                "Final Answer: 8760"
            )

        if has_system:
            # 有 system prompt，但 mock 第一次仍漏掉 Final Answer
            return (
                "Step 1: 365 days in a year.\n"
                "Step 2: 24 hours per day.\n"
                "Step 3: 365 * 24 = 8760."
            )

        # 裸调用：随意回答，不含 Final Answer
        return "A year has about 8760 hours."


class Harness:
    """
    最小 harness：给 LLM 套上 system prompt，并对其输出做格式自检。
    自检失败时，把 critique 写回 history，让 LLM 重试。
    """

    SYSTEM_PROMPT = (
        "system: You are a careful assistant. "
        "Solve the problem step by step and end with 'Final Answer: <number>'."
    )

    def __init__(self, llm: Callable[[str], str], max_iterations: int = 3):
        self.llm = llm
        self.max_iterations = max_iterations
        self.history: List[str] = []

    def _check(self, output: str) -> Tuple[bool, str]:
        """
        自检：输出必须包含逐步推理痕迹和 Final Answer。
        返回 (passed, critique)。
        """
        has_steps = bool(re.search(r"(?i)(step\s*\d|first,|then,|therefore)", output))
        has_final = bool(re.search(r"(?i)Final Answer:\s*\d+", output))

        if has_steps and has_final:
            return True, ""

        missing = []
        if not has_steps:
            missing.append("step-by-step reasoning")
        if not has_final:
            missing.append("'Final Answer: <number>'")

        critique = (
            f"critique: Your response is missing {', '.join(missing)}. "
            "Please revise according to the system instruction."
        )
        return False, critique

    def run(self, task: str) -> str:
        messages = [self.SYSTEM_PROMPT, f"user: {task}"]

        for i in range(self.max_iterations):
            prompt = "\n".join(messages + self.history)
            output = self.llm(prompt)
            self.history.append(f"assistant (try {i + 1}): {output}")

            passed, critique = self._check(output)
            status = "PASS" if passed else "FAIL"
            print(f"--- Iteration {i + 1} ---")
            print(f"  {output}")
            print(f"  check: {status}")
            if not passed:
                print(f"  -> {critique}")

            if passed:
                return output

            self.history.append(critique)

        return "Harness failed: max iterations reached without passing self-check."


def bare_call(llm: Callable[[str], str], task: str) -> str:
    """裸调用 LLM，没有任何 harness。"""
    return llm(task)


def main():
    print("=" * 64)
    print("nano-qwenpaw L0 — harness: system prompt + self-check")
    print("=" * 64)
    print("\n⚠️  MOCK LLM: 这不是真实语言模型，只是规则 mock。")
    print("    它的唯一作用是展示 harness 如何改变模型输出形态。")
    print("    L1 会把 mock 替换成真实 LLM API。\n")

    task = "How many hours are there in a year?"
    llm = MockLLM()

    print("1. Bare LLM call (no harness):")
    print("-" * 64)
    bare = bare_call(llm, task)
    print(f"  {bare}\n")

    print("2. With harness (system prompt + self-check loop):")
    print("-" * 64)
    harness = Harness(llm, max_iterations=3)
    result = harness.run(task)

    print("\n" + "=" * 64)
    print("Final harness output:")
    print(result)
    print("=" * 64)

    # 自检查：harness 最终必须输出合规答案
    assert "Final Answer: 8760" in result, (
        f"expected compliant output containing 'Final Answer: 8760', got {result!r}"
    )
    print("\n✅ self-check passed: harness converted non-compliant output into compliant output.")


if __name__ == "__main__":
    main()
