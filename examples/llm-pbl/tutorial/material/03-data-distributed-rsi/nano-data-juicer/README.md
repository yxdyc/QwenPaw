# nano-data-juicer

> **抓的核心机制**：把数据处理抽象成**可组合算子（OP）**，用**配置（dict/yaml）驱动** pipeline；L2 起把 pipeline **分布式化**（分区 + 并行局部 OP + 全局 OP 收敛点）；L3 对齐权威 OP 接口（注册表 + 两段式 Filter + stats 命名空间 + 配置 schema）。
> **对应真实系统**：[Data-Juicer](https://github.com/modelscope/data-juicer)（本地参考：`${DATA_JUICER_REPO}`）
> **轨道**：[03 数据/分布式/RSI](../README.md) · **状态**：L0–L3 ✅

---

## 为什么从这个开始

Data-Juicer 是用户日常工具，门槛低、价值直接。理解了「OP + 配置驱动」这个抽象，
就能看懂 Data-Juicer 全部 200+ 算子的骨架——它们只是同一套接口的不同实现。

> 真实 Data-Juicer 的 OP 分类（跨模态总数，仅作背景，非 text-only 精确值 `[TODO: verify text-subset counts]`）：
> Formatter / Mapper / Filter / Selector / Deduplicator / Aggregator 等。

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | single-file 玩具：3 个 OP（filter/mapper/dedup）+ 配置驱动 pipeline，CPU 即跑 | ✅ `L0_toy_ops.py` + `tutorial_L0.md` |
| **L1** | 接真实小样本（10 条医学 SFT 数据），加一个 llm-based OP（质量打分） | ✅ `L1_real_data.py` + `tutorial_L1.md` |
| **L2** | 分布式 pipeline：分区 + 真实多进程并行局部 OP + 全局 OP（dedup）收敛点（convergence vs shuffle）+ 分区级容错；对照 Data-Juicer `RayExecutor` / `PartitionedRayExecutor` 源码（对接 [nano-ray](../nano-ray/README.md) 的调度视角） | ✅ `L2_distributed_pipeline.py` + `tutorial_L2.md` |
| **L3** | 对照 Data-Juicer 真实 OP 接口，复现一个 filter（text_length_filter）的完整行为 + 配置 schema：Registry/load_ops 构造链、两段式 compute_stats→process、stats 复用、区间与 reversed 语义、NON_STATS_FILTERS、`__init_subclass__` 守卫；含与 L2 漏斗逐位一致的跨级别契约 | ✅ `L3_filter_interface.py` + `tutorial_L3.md` |

**环境依赖分级**：L0 零依赖（纯标准库）；L1 纯标准库，真实 LLM 模式需 `DASHSCOPE_API_KEY` / `OPENAI_API_KEY`（`--mock` 无需）；L2 纯标准库（`multiprocessing` 真实 worker 进程，CPU 即跑；不把 Ray 设为必装依赖，多进程承载同一套分布式执行语义，显式声明见 `tutorial_L2.md` §10）；L3 纯标准库（同目录 import L2 模块，单进程复现接口语义，声明见 `tutorial_L3.md` §10）。

---

## L0 快速开始

```bash
python L0_toy_ops.py
```

预期输出（漏斗 `6 → 6 → 4 → 2`）：先 lowercase 规范化（条数不变），
再过滤短文本（6→4），最后去重（4→2）。逐步拆解见 `tutorial_L0.md`。

---

## 费曼自检

- 能不能用一句话说清「OP 抽象比写一堆 if-else 处理脚本好在哪」？
- 如果让你加一个「按语言过滤」的 OP，你需要改 pipeline 主流程吗？（答案应是否——这正是可组合性的价值）
- （L3）为什么 filter 要分 compute_stats 和 process 两段？如果配置里把 `min_len` 拼成了 `min_lne`，框架会报错吗？（答案：不会，静默掉回默认值——见 `L3_filter_interface.py` [4]，这正是没有 schema 的代价）

## 权威实现与延伸

- 轨道：[03 数据/分布式/RSI](../README.md)
- 对标源码：Data-Juicer `github.com/modelscope/data-juicer`（本地参考：`${DATA_JUICER_REPO}`）
