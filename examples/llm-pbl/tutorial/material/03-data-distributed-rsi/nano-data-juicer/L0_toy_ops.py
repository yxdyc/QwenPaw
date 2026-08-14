"""nano-data-juicer · L0 玩具实现
====================================

目标：用 ~150 行抓住 Data-Juicer 的核心机制——
    1. 把数据处理抽象成「算子 (OP)」：每个 OP 是一个 callable，吃一个样本列表，吐一个样本列表。
    2. 用「配置 (config)」驱动 pipeline：不写死流程，按配置顺序串联 OP。

这是 L0（玩具级）：单文件、纯 Python、CPU 即跑、无外部依赖。
真实 Data-Juicer 还有 Formatter/Aggregator、Arrow schema、分布式等，见 README 阶梯 L1-L3。

运行：
    python L0_toy_ops.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict, Any

# 一个样本就是一个 dict。真实系统里可能是 Arrow 行 / jsonl 行，这里用最朴素的 dict。
Sample = Dict[str, Any]
# 一个 OP 就是 list[Sample] -> list[Sample] 的函数。
OP = Callable[[List[Sample]], List[Sample]]


# ---------------------------------------------------------------------------
# 三个最小 OP：filter / mapper / deduplicator
# 它们对应 Data-Juicer 三大类算子的「最小可运行内核」。
# ---------------------------------------------------------------------------

def make_length_filter(min_chars: int) -> OP:
    """Filter：按文本长度过滤。返回一个 OP（闭包携带参数）。"""

    def op(samples: List[Sample]) -> List[Sample]:
        return [s for s in samples if len(s.get("text", "")) >= min_chars]

    return op


def make_lowercase_mapper() -> OP:
    """Mapper：对每个样本做变换（不改变条数）。"""

    def op(samples: List[Sample]) -> List[Sample]:
        out = []
        for s in samples:
            s = dict(s)  # 不修改原样本，避免副作用
            s["text"] = s.get("text", "").strip().lower()
            out.append(s)
        return out

    return op


def make_deduplicator(key: str = "text") -> OP:
    """Deduplicator：按某个 key 去重，保留首次出现。"""

    def op(samples: List[Sample]) -> List[Sample]:
        seen = set()
        out = []
        for s in samples:
            k = s.get(key)
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    return op


# ---------------------------------------------------------------------------
# 配置驱动的 pipeline：核心抽象
# ---------------------------------------------------------------------------

@dataclass
class Pipeline:
    """按顺序执行一串 (name, op)。名字仅用于打印，方便观察每步效果。"""

    steps: List[tuple]  # [(name, OP), ...]

    def run(self, samples: List[Sample]) -> List[Sample]:
        cur = samples
        print(f"[input] {len(cur)} samples")
        for name, op in self.steps:
            cur = op(cur)
            print(f"[{name}] -> {len(cur)} samples")
        return cur


def build_pipeline_from_config(config: List[Dict[str, Any]]) -> Pipeline:
    """从一个「配置列表」构造 pipeline。

    这就是 Data-Juicer「配置驱动」的最小内核：
    用户只描述「要哪些 OP、什么参数、什么顺序」，不写流程代码。
    """
    registry = {
        "length_filter": lambda cfg: make_length_filter(cfg.get("min_chars", 1)),
        "lowercase_mapper": lambda cfg: make_lowercase_mapper(),
        "deduplicator": lambda cfg: make_deduplicator(cfg.get("key", "text")),
    }
    steps = []
    for item in config:
        name, cfg = item["op"], item.get("params", {})
        if name not in registry:
            raise KeyError(f"unknown op: {name} (available: {list(registry)})")
        steps.append((name, registry[name](cfg)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# 跑一个 toy 例子
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data: List[Sample] = [
        {"text": "Hello World", "id": 1},
        {"text": "hi", "id": 2},                 # 太短，会被过滤
        {"text": "hello world", "id": 3},        # 规范化后与 id=1 重复
        {"text": "Data-Juicer is a data system", "id": 4},
        {"text": "short", "id": 5},              # 太短
        {"text": "Data-Juicer is a data system", "id": 6},  # 与 id=4 重复
    ]

    # 配置：先规范化 -> 再过滤短文本 -> 最后去重
    # 注意顺序很重要：先去重再 lowercase 会漏掉大小写不同的重复。
    config = [
        {"op": "lowercase_mapper"},
        {"op": "length_filter", "params": {"min_chars": 6}},
        {"op": "deduplicator", "params": {"key": "text"}},
    ]

    pipeline = build_pipeline_from_config(config)
    result = pipeline.run(data)

    print("\n[final]")
    for s in result:
        print("  ", s)

    # ------------------------------------------------------------------
    # 反例（见 tutorial_L0.md §5）：把 dedup 挪到 lowercase 之前，
    # 大小写不同的重复对会漏网——OP 顺序是 pipeline 语义的一部分。
    # ------------------------------------------------------------------
    print("\n=== 反例：先 dedup 再 lowercase ===")
    bad_config = [
        {"op": "deduplicator", "params": {"key": "text"}},
        {"op": "lowercase_mapper"},
        {"op": "length_filter", "params": {"min_chars": 6}},
    ]
    bad_result = build_pipeline_from_config(bad_config).run(data)
    print("final:", [s["id"] for s in bad_result])

    # 费曼自检题（见 README）：
    #   Q: 加一个「按语言过滤」的 OP，需要改 Pipeline.run 吗？
    #   A: 不需要——只需新写一个 make_xxx_filter，并注册进 registry。
    #      这正是「OP 可组合 + 配置驱动」相对写死 if-else 的价值。
