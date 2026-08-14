# nano-qwenpaw L0 — agent harness：system prompt + 输出自检

> 目标：用**标准库**写一个最小 agent harness，展示「同一个 LLM，套上 system prompt + 输出自检后，行为更可控」。这里的 LLM 是 mock，只用于演示 harness 形态；L1 再接入真实 API。

---

## 1. 为什么需要 harness？

调用大模型最简单的方式是裸调用：把问题丢进去，把回答拿出来。但生产环境里这样做经常踩坑：

1. **格式不稳定**：让它「逐步推理」，有时照做，有时直接给答案。
2. **关键信息漏掉**：要它输出 `Final Answer: <number>`，它可能只说「答案是 8760」。
3. **难以复查**：没有统一结构，下游解析代码得写一堆 `if/else`。

**harness** 就是包在 LLM 外面的一层脚手架：先给模型一个明确的 system prompt（方法论），再对它的输出做检查（自检），检查不通过就写回 critique 让它重试。这样把「希望模型做什么」从「暗知识」变成「可执行流程」。

本节只做一个**单轮任务**、**一个格式检查**、**最多重试 3 次**的玩具，但已经包含真实 agent harness 的全部骨架：

- system prompt 注入
- 输出格式自检
- critique 反馈循环
- 最大迭代兜底

---

## 2. 运行

文件：`L0_harness_loop.py`，零外部依赖，CPU 即跑。

```bash
$ python3 L0_harness_loop.py
```

真实输出：

```
================================================================
nano-qwenpaw L0 — harness: system prompt + self-check
================================================================

⚠️  MOCK LLM: 这不是真实语言模型，只是规则 mock。
    它的唯一作用是展示 harness 如何改变模型输出形态。
    L1 会把 mock 替换成真实 LLM API。

1. Bare LLM call (no harness):
----------------------------------------------------------------
  A year has about 8760 hours.

2. With harness (system prompt + self-check loop):
----------------------------------------------------------------
--- Iteration 1 ---
  Step 1: 365 days in a year.
Step 2: 24 hours per day.
Step 3: 365 * 24 = 8760.
  check: FAIL
  -> critique: Your response is missing 'Final Answer: <number>'. Please revise according to the system instruction.
--- Iteration 2 ---
  Step 1: A common year has 365 days.
Step 2: Each day has 24 hours.
Step 3: 365 * 24 = 8760 hours.
Final Answer: 8760
  check: PASS

================================================================
Final harness output:
Step 1: A common year has 365 days.
Step 2: Each day has 24 hours.
Step 3: 365 * 24 = 8760 hours.
Final Answer: 8760
================================================================

✅ self-check passed: harness converted non-compliant output into compliant output.
```

---

## 3. 代码骨架

### 3.1 Mock LLM（L0 专用）

```python
class MockLLM:
    def __call__(self, prompt: str) -> str:
        has_system = "system:" in prompt.lower()
        has_critique = "critique:" in prompt.lower()

        if has_critique:
            return (
                "Step 1: A common year has 365 days.\n"
                "Step 2: Each day has 24 hours.\n"
                "Step 3: 365 * 24 = 8760 hours.\n"
                "Final Answer: 8760"
            )
        if has_system:
            return (
                "Step 1: 365 days in a year.\n"
                "Step 2: 24 hours per day.\n"
                "Step 3: 365 * 24 = 8760."
            )
        return "A year has about 8760 hours."
```

⚠️ **再次强调**：这个 mock 是规则驱动，只会根据 prompt 里是否出现 `system:` / `critique:` 返回三段固定文本。它不理解「一年有多少小时」。L1 会把它换成真实 LLM。

### 3.2 Harness 核心

```python
class Harness:
    SYSTEM_PROMPT = (
        "system: You are a careful assistant. "
        "Solve the problem step by step and end with 'Final Answer: <number>'."
    )

    def _check(self, output: str) -> Tuple[bool, str]:
        has_steps = bool(re.search(r"(?i)(step\s*\d|first,|then,|therefore)", output))
        has_final = bool(re.search(r"(?i)Final Answer:\s*\d+", output))
        ...
        return passed, critique

    def run(self, task: str) -> str:
        messages = [self.SYSTEM_PROMPT, f"user: {task}"]
        for i in range(self.max_iterations):
            prompt = "\n".join(messages + self.history)
            output = self.llm(prompt)
            # 模型需要看到自己上一次的输出，下一轮才能修正它
            self.history.append(f"assistant (try {i + 1}): {output}")
            passed, critique = self._check(output)
            if passed:
                return output
            self.history.append(critique)
        return "Harness failed: max iterations reached..."
```

（以上省略了源码中的打印逻辑，完整实现见 `L0_harness_loop.py`。）

核心不变量：**harness 不直接生成答案，它生成「约束 + 检查 + 反馈」**。真正写答案的还是 LLM，但 harness 决定了答案必须满足什么格式、失败时如何修正。

---

## 4. 与 qwenpaw coach 的对应关系（概念层）

qwenpaw 本仓库参考：`coach/profile/SOUL.md`（7 条编号原则：K+1 / 费曼 / PBL / 对抗自检 / 反幻觉 / 学习者自主 / 持续改进；2026-08-06 核验 `## 1`–`## 7` 共 7 条，principle 5 = Anti-Hallucination）。

| nano 概念 | qwenpaw coach 概念 | 说明 |
|-----------|-------------------|------|
| `Harness.SYSTEM_PROMPT` | coach 的 SOUL.md / system 原则注入 | 把「该怎么做」写进模型上下文，而不是只藏在代码注释里 `[TODO: verify source]` |
| `Harness._check` | 对抗自检（Adversarial Self-Verification） | 每个输出先过一道质量门，再决定是否交付 `[TODO: verify source]` |
| `Harness.history` + critique | 多轮对话 / 记忆 | 把之前的失败和反馈保留下来，让模型在下一轮修正 `[TODO: verify source]` |
| `max_iterations` | 终止条件与兜底 | 防止无限循环，和 coach 的「不信任模型永远一次做对」一致 `[TODO: verify source]` |

L0 只到概念映射；源码级对应（具体文件路径、类名、API）留到 L3 再核对。

---

## 5. 费曼：讲给外行听

**类比：考试阅卷**

想象一个学生（LLM）做一道数学题。你告诉他：「必须写步骤，最后写『最终答案：xxx』」。

- **裸调用**：学生只写了一句「答案是 8760」。你不知道他怎么算的，也无法用机器自动批改。
- **harness**：你相当于在考场里加了一个自动阅卷机。学生第一次交了只写结果的卷子，阅卷机打回并批注：「缺少步骤和最终答案格式」。学生第二次按要求写了步骤和格式，阅卷机通过。

harness 不是替学生做题，而是**把「怎么做对题」变成可检查、可反馈的流程**。模型还是那个模型，但行为被约束得更可靠。

---

## 6. 思考题

1. 如果把 `Harness.SYSTEM_PROMPT` 去掉，只保留 `_check` 和 critique 循环，程序还能收敛吗？这说明了 system prompt 和 feedback 各自的什么作用？
2. `_check` 目前用正则检查关键词。真实系统里还可能做哪些检查？（提示：JSON Schema、事实一致性、安全策略。）
3. `max_iterations` 是 3。如果 mock 永远返回不合规输出，harness 会怎么收场？这个兜底在生产环境里为什么必不可少？

---

## 7. 反例：harness 不是万能药

**反例 1：自检函数本身可能漏检**

`_check` 只检查有没有 `Step` / `Final Answer` 和数字。如果模型输出：

```
Step 1: guess.
Final Answer: 9999
```

自检会 PASS，但答案明显错误。这说明 **harness 的上限取决于检查函数的质量**——格式检查容易，事实正确性检查难。

**反例 2：system prompt 和 critique 不一致时，模型会无所适从**

如果 system prompt 要求「用一句话回答」，critique 又要求「写出详细步骤」，真实 LLM 可能来回摇摆，永远通不过。这提醒我们：**harness 里的所有约束必须自洽**，否则会变成「自己跟自己打架」。

---

## 8. 下一步 L1

L1 会把这个 mock LLM 替换成真实 API（例如 DashScope / OpenAI），并扩展 harness 的能力：

1. **真实 system prompt 工程**：如何写才能让真实模型稳定输出指定格式？
2. **多轮记忆管理**：跨任务、跨用户轮次时，history 该保留什么、丢掉什么？
3. **更复杂的自检**：用 JSON Schema 校验结构化输出，或调用外部工具验证事实。

---

## 9. 溯源

- qwenpaw coach 参考：`coach/profile/SOUL.md`（本仓库），7 条编号原则：K+1、费曼、PBL、对抗自检、反幻觉、学习者自主、持续改进（2026-08-06 核验）。
- harness / structured output 概念：OpenAI API `response_format` / function calling，Claude system prompt 等（公开文档，无需 arXiv）。
- 源码级对应：L0 未给出 qwenpaw coach 的具体文件路径，L3 再对照 `coach/profile/skills/` 与 `agent.json` 的实现 `[TODO: verify source]`。
