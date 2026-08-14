"""nano-data-juicer · L1 真实数据 + LLM-based OP
==================================================

K+1 目标（相对 L0）：
    L0 用 6 条内联 toy 数据 + 3 个纯规则 OP，抓住「OP 可组合 + 配置驱动」。
    L1 在此基础上：
      1. 接真实小样本（一份医学 SFT 数据，10 条 JSONL）
      2. 处理嵌套结构（messages 格式，非扁平 text 字段）
      3. 引入 LLM-based OP：用 LLM 给数据打质量分（真实 API 调用）
      4. 展示 rule-based OP 与 llm-based OP 在同一 pipeline 里无缝组合

运行：
    # 真实模式（需要 API key）：
    export DASHSCOPE_API_KEY="CHANGEME"   # 或 OPENAI_API_KEY + OPENAI_BASE_URL
    python L1_real_data.py --data /path/to/train.jsonl

    # Mock 模式（无需 API，用启发式打分代替 LLM，仅用于演示 pipeline 流程）：
    python L1_real_data.py --mock

注意：--mock 模式下的打分是启发式规则，不是真实 LLM 输出。
      真实模式下，每条数据会调用一次 LLM API 进行质量评估。

依赖：纯标准库（json, urllib, os, sys, re, argparse）。无第三方包。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# 基础类型（与 L0 一致：OP = list[Sample] -> list[Sample]）
# ---------------------------------------------------------------------------

Sample = Dict[str, Any]
OP = Callable[[List[Sample]], List[Sample]]

# The dataset is intentionally not bundled. Resolve it explicitly rather than
# falling back to a maintainer-specific checkout path.
DATA_PATH_ENV = os.environ.get("LLM_PBL_DATA_PATH")
DATA_PATH = (
    Path(DATA_PATH_ENV).expanduser() if DATA_PATH_ENV else None
)


# ---------------------------------------------------------------------------
# Formatter：从 JSONL 文件加载数据（L0 是内联 list，L1 接真实文件）
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Sample]:
    """从 JSONL 文件加载样本。每行一个 JSON 对象。"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_id"] = i  # 保留行号，方便溯源
            samples.append(obj)
    return samples


# ---------------------------------------------------------------------------
# Rule-based OPs（处理 messages 嵌套结构）
# ---------------------------------------------------------------------------

def make_extract_fields_mapper() -> OP:
    """Mapper：从 messages 格式中提取扁平字段，供后续 OP 使用。

    输入格式（真实 SFT 数据）：
        {"messages": [{"role": "user", "content": [...]}, {"role": "assistant", "content": [...]}]}

    提取：
        - question_text: 用户问题的纯文本
        - answer_text: 助手回答的纯文本
        - thinking_text: <think>...</think> 内的推理过程
        - thinking_chars: 推理文本字符数
        - answer_chars: 最终回答字符数
    """

    def extract_text(content) -> str:
        """content 可能是 str 或 list[{"type":"text","text":...}]"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        return ""

    def op(samples: List[Sample]) -> List[Sample]:
        out = []
        for s in samples:
            s = dict(s)  # 不修改原样本
            messages = s.get("messages", [])
            user_msg = ""
            assistant_msg = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = extract_text(msg.get("content", ""))
                elif msg["role"] == "assistant":
                    assistant_msg = extract_text(msg.get("content", ""))

            s["question_text"] = user_msg
            s["answer_text"] = assistant_msg

            # 提取 <think>...</think> 内容
            think_match = re.search(r"<think>(.*?)</think>", assistant_msg, re.DOTALL)
            s["thinking_text"] = think_match.group(1).strip() if think_match else ""
            s["thinking_chars"] = len(s["thinking_text"])

            # 最终回答 = 去掉 think 部分
            final_answer = re.sub(r"<think>.*?</think>", "", assistant_msg, flags=re.DOTALL).strip()
            s["final_answer"] = final_answer
            s["answer_chars"] = len(final_answer)

            out.append(s)
        return out

    return op


def make_thinking_length_filter(min_chars: int = 100) -> OP:
    """Filter：过滤掉推理过程太短的样本（可能是低质量/截断数据）。"""

    def op(samples: List[Sample]) -> List[Sample]:
        return [s for s in samples if s.get("thinking_chars", 0) >= min_chars]

    return op


def make_answer_format_filter() -> OP:
    """Filter：只保留最终回答符合 \\boxed{X} 格式的样本。

    这是 rule-based 质量检查：如果 SFT 数据的回答格式不对，
    说明生成过程有问题，不应进入训练集。
    """
    pattern = re.compile(r"\\boxed\{[A-E]\}")

    def op(samples: List[Sample]) -> List[Sample]:
        return [s for s in samples if pattern.search(s.get("final_answer", ""))]

    return op


# ---------------------------------------------------------------------------
# LLM-based OP（L1 核心新增）
# ---------------------------------------------------------------------------

def call_llm(prompt: str, api_key: str, base_url: str, model: str) -> str:
    """调用 OpenAI-compatible API。纯 urllib 实现，无第三方依赖。"""
    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def make_llm_quality_scorer(
    api_key: str,
    base_url: str,
    model: str,
    threshold: int = 3,
) -> OP:
    """LLM-based OP：让 LLM 评估每条 SFT 数据的推理质量（1-5 分）。

    这是 Data-Juicer 中 llm-based OP 的最小内核：
    - 输入：一批样本
    - 对每个样本：构造 prompt → 调用 LLM → 解析分数
    - 输出：分数 >= threshold 的样本（附带 llm_score 字段）

    真实 Data-Juicer 的 llm-based OP（如 nlpaug_en_mapper、chinese_convert_mapper）
    也是同样的模式：对每条数据调用外部模型做变换/评估，只是接口更复杂。
    """

    PROMPT_TEMPLATE = """你是一个 SFT 训练数据质量评估员。请评估以下医学 MCQ 训练样本的推理质量。

评估维度：
1. 推理链是否完整（从题目分析到最终答案）
2. 医学知识引用是否合理
3. 最终答案与推理是否一致

请只输出一个 1-5 的整数分数（1=很差, 5=优秀），不要输出其他内容。

--- 样本开始 ---
【题目】{question}
【推理过程（截取前 500 字）】{thinking}
【最终答案】{answer}
--- 样本结束 ---

分数："""

    def op(samples: List[Sample]) -> List[Sample]:
        out = []
        for i, s in enumerate(samples):
            question = s.get("question_text", "")[:200]
            thinking = s.get("thinking_text", "")[:500]
            answer = s.get("final_answer", "")

            prompt = PROMPT_TEMPLATE.format(
                question=question, thinking=thinking, answer=answer
            )

            try:
                response = call_llm(prompt, api_key, base_url, model)
                # 解析分数：取第一个 1-5 的数字
                score_match = re.search(r"[1-5]", response)
                score = int(score_match.group()) if score_match else 0
            except Exception as e:
                print(f"  [warn] sample {i} LLM call failed: {e}", file=sys.stderr)
                score = 0

            s = dict(s)
            s["llm_score"] = score
            out.append(s)
            print(f"  [llm_scorer] sample {i}: score={score}")
            time.sleep(0.5)  # rate limit 友好

        # 过滤低于阈值的
        kept = [s for s in out if s["llm_score"] >= threshold]
        print(f"  [llm_scorer] {len(out)} scored, {len(kept)} kept (threshold={threshold})")
        return kept

    return op


def make_mock_quality_scorer(threshold: int = 3) -> OP:
    """⚠️ MOCK 模式：用启发式规则模拟 LLM 打分。仅用于演示 pipeline 流程。

    这不是真实 LLM 输出。真实版使用 make_llm_quality_scorer()。
    启发式：thinking_chars > 1000 → 5分, > 500 → 4分, > 200 → 3分, else 2分。
    """

    def op(samples: List[Sample]) -> List[Sample]:
        out = []
        for i, s in enumerate(samples):
            tc = s.get("thinking_chars", 0)
            if tc > 1000:
                score = 5
            elif tc > 500:
                score = 4
            elif tc > 200:
                score = 3
            else:
                score = 2

            s = dict(s)
            s["llm_score"] = score
            s["_scored_by"] = "mock_heuristic"  # 显式标记：这是 mock
            out.append(s)
            print(f"  [mock_scorer] sample {i}: thinking_chars={tc}, score={score}")

        kept = [s for s in out if s["llm_score"] >= threshold]
        print(f"  [mock_scorer] {len(out)} scored, {len(kept)} kept (threshold={threshold})")
        print("  ⚠️  MOCK MODE: 分数由启发式规则生成，非真实 LLM 输出。")
        return kept

    return op


# ---------------------------------------------------------------------------
# Pipeline（与 L0 相同的配置驱动抽象，增加统计能力）
# ---------------------------------------------------------------------------

@dataclass
class Pipeline:
    steps: List[tuple]  # [(name, OP), ...]
    stats: Dict[str, int] = field(default_factory=dict)

    def run(self, samples: List[Sample]) -> List[Sample]:
        cur = samples
        self.stats = {"input": len(cur)}
        print(f"\n{'='*60}")
        print(f"[pipeline] input: {len(cur)} samples")
        print(f"{'='*60}")
        for name, op in self.steps:
            cur = op(cur)
            self.stats[name] = len(cur)
            print(f"[{name}] -> {len(cur)} samples")
        print(f"{'='*60}")
        print(f"[pipeline] output: {len(cur)} samples")
        print(f"{'='*60}\n")
        return cur


# ---------------------------------------------------------------------------
# 配置 + 注册表
# ---------------------------------------------------------------------------

def build_pipeline(config: List[Dict[str, Any]], mock: bool = False) -> Pipeline:
    """从配置构建 pipeline。与 L0 相同的 registry 模式，但 OP 种类更丰富。"""

    # 解析 API 配置（真实模式）
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("LLM_PBL_MODEL", "qwen-plus")

    registry: Dict[str, Callable[[Dict], OP]] = {
        "extract_fields_mapper": lambda cfg: make_extract_fields_mapper(),
        "thinking_length_filter": lambda cfg: make_thinking_length_filter(
            cfg.get("min_chars", 100)
        ),
        "answer_format_filter": lambda cfg: make_answer_format_filter(),
        "llm_quality_scorer": lambda cfg: (
            make_mock_quality_scorer(cfg.get("threshold", 3))
            if mock
            else make_llm_quality_scorer(
                api_key=api_key,
                base_url=base_url,
                model=model,
                threshold=cfg.get("threshold", 3),
            )
        ),
    }

    steps = []
    for item in config:
        name, cfg = item["op"], item.get("params", {})
        if name not in registry:
            raise KeyError(f"unknown op: {name} (available: {list(registry)})")
        steps.append((name, registry[name](cfg)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def print_sample_summary(samples: List[Sample]) -> None:
    """打印样本摘要，方便观察 pipeline 效果。"""
    print(f"\n{'─'*60}")
    print(f"最终保留 {len(samples)} 条样本：")
    print(f"{'─'*60}")
    for s in samples:
        q = s.get("question_text", "")[:60].replace("\n", " ")
        score = s.get("llm_score", "?")
        tc = s.get("thinking_chars", 0)
        scored_by = s.get("_scored_by", "llm")
        print(f"  [line {s.get('_line_id', '?')}] score={score}({scored_by}) "
              f"thinking={tc}chars | {q}...")
    print()


def main():
    parser = argparse.ArgumentParser(description="nano-data-juicer L1: real data + LLM-based OP")
    parser.add_argument("--mock", action="store_true",
                        help="使用启发式 mock 打分（无需 API key，仅演示流程）")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "JSONL 数据文件路径；也可设置 LLM_PBL_DATA_PATH。"
            "仓库不附带私有或受限数据。"
        ),
    )
    args = parser.parse_args()

    # 1. 先检查 API key（mock 模式除外），避免在真实模式下加载大文件后才发现无法调用
    if args.mock:
        print("\n⚠️  MOCK MODE: LLM 打分将由启发式规则代替，非真实 LLM 输出。")
        print("    去掉 --mock 并设置 DASHSCOPE_API_KEY 环境变量可启用真实 LLM 打分。\n")
    elif not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("[error] 未检测到 API key。请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。",
              file=sys.stderr)
        print("        或使用 --mock 模式演示 pipeline 流程。", file=sys.stderr)
        sys.exit(1)

    # 2. 加载真实数据
    data_path = Path(args.data).expanduser() if args.data else DATA_PATH
    if data_path is None:
        print(
            "[error] 未指定数据文件。请使用 --data，或设置 "
            "LLM_PBL_DATA_PATH。",
            file=sys.stderr,
        )
        sys.exit(1)
    if not data_path.exists():
        print(f"[error] 数据文件不存在: {data_path}", file=sys.stderr)
        print("请指定 --data 路径，或确认数据文件位置。", file=sys.stderr)
        sys.exit(1)

    print(f"[load] {data_path}")
    samples = load_jsonl(data_path)
    print(f"[load] {len(samples)} samples loaded")

    # 3. 配置 pipeline
    # 顺序：提取字段 → 格式过滤 → 长度过滤 → LLM 打分过滤
    # 注意：rule-based 先跑（便宜），llm-based 后跑（贵）——这是真实数据 pipeline 的成本意识
    # min_chars=4500：要求推理链足够充实（这批数据 thinking 长度 2474–6889，
    # 阈值 4500 会过滤掉推理较短的 3 条，保留 7 条——演示 pipeline 的实际筛选效果）
    config = [
        {"op": "extract_fields_mapper"},
        {"op": "answer_format_filter"},
        {"op": "thinking_length_filter", "params": {"min_chars": 4500}},
        {"op": "llm_quality_scorer", "params": {"threshold": 3}},
    ]

    # 4. 执行
    pipeline = build_pipeline(config, mock=args.mock)
    result = pipeline.run(samples)

    # 5. 输出摘要
    print_sample_summary(result)

    # 6. 统计（L0 没有的：量化 pipeline 效果）
    print("[stats] pipeline 漏斗：")
    for stage, count in pipeline.stats.items():
        print(f"  {stage}: {count}")

    # 费曼自检（见 tutorial_L1.md）：
    #   Q: 为什么 rule-based OP 要放在 llm-based OP 前面？
    #   A: 成本。rule-based 是 CPU 微秒级，llm-based 是网络秒级 + 花钱。
    #      先用便宜 OP 过滤掉明显不合格的，减少 LLM 调用次数。
    #      这和真实 Data-Juicer 的最佳实践一致：rule filter 前置，llm op 后置。


if __name__ == "__main__":
    main()
