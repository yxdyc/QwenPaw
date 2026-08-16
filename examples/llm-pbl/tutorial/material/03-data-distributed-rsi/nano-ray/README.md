# nano-ray

> **抓的核心机制**：**task / actor 编程模型** + **分布式调度** + **object store**（零拷贝共享）。
> **对应真实系统**：[Ray](https://github.com/ray-project/ray)（论文 arXiv:1712.05889）
> **轨道**：[03 数据/分布式/RSI](../README.md) · **状态**：L0 ✅，L1 ✅，L2 ✅，L3 ✅

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | 玩具：纯标准库实现 remote function（future）+ object store（传引用）+ 动态任务图调度，理解「把函数变成可调度单元」 | ✅ [`L0_task_graph.py`](L0_task_graph.py) · [教程](tutorial_L0.md) |
| **L1** | 用**真实 Ray** 跑 `nano-data-juicer` L2 的同一执行计划（同语料、同漏斗），测出启动/序列化/提交的真实成本 | ✅ [`L1_ray_pipeline.py`](L1_ray_pipeline.py) · [教程](tutorial_L1.md) |
| **L2** | 引入 actor：有状态算子（如全局去重的 index）用 actor 承载；对照「先到先留（顺序敏感）vs min-row_id（顺序免疫）」两条更新规则，实测 actor RPC 成本 | ✅ [`L2_actor_dedup_index.py`](L2_actor_dedup_index.py) · [教程](tutorial_L2.md) |
| **L3** | 理解 object store / plasma 零拷贝，分析数据密集任务的内存与调度 | ✅ [`L3_object_store_zero_copy.py`](L3_object_store_zero_copy.py) · [教程](tutorial_L3.md) |

**环境依赖**：L0 零外部依赖（纯标准库），CPU 即跑：`python3 L0_task_graph.py`。
L1 需要 `pip install ray`（实测 ray==2.56.1 / Python 3.13 / macOS arm64 单机 4 CPU），
`python3 L1_ray_pipeline.py` 全自动自检，无需数据文件、无需 GPU、无需多机。
L2 依赖同 L1（ray==2.56.1），import 复用同目录 `L1_ray_pipeline`（语料生成 + 漏斗常量对照），
`python3 L2_actor_dedup_index.py` 全自动自检，单机 4 CPU 即可。
L3 依赖同 L1（ray==2.56.1 / numpy），import 复用同目录 `L1_ray_pipeline`（语料 + 漏斗常量），
`python3 L3_object_store_zero_copy.py` 全自动自检（两个 Ray 会话：默认 store + 75 MB 小 store），
任意 CWD 可跑；`[5]` 多节点 locality 为显式注明的本质模拟，真机验证需在真实 GPU/多机环境验证。

## 核心要讲清的点

- task（无状态函数）vs actor（有状态进程）如何选择
- object store 为何能避免 worker 间数据拷贝（共享内存 + 引用传递）
- 调度：Ray 如何把 task 放到有资源的节点
- actor 默认串行执行（threaded `max_concurrency=1`）为何是「免费」的并发安全保证；
  调大并发后丢更新如何被 barrier 确定性复现，而不是靠调度运气
- 顺序敏感 vs 顺序免疫的更新规则：keep-first 的语义住在 RPC 到达时序里，
  min-row_id 聚合 + 收敛排序把语义还给规则本身

## 费曼自检

- 能不能解释「把一个大 DataFrame **传值**给 10 个 task 要付 10 次序列化，
  而 `ray.put` 一次 + 传 10 个 ObjectRef 只付 1 次」？「重复传同一个对象」能省吗？
- 能不能解释「同一个执行计划换执行器（串行 → multiprocessing → 真 Ray），
  为什么漏斗数字必须一分不差」？差一条说明什么？
- 能不能解释「同一份去重计划反向喂给 actor 索引，为什么 first-seen 恰好翻转
  236 条 keeper、min-row_id 一条不翻」？236 这个数从哪个恒等式来？
- 能不能解释「这个规模下 actor 路线与 task 收敛点的墙钟只在同一量级打转
  （11–24 ms，先后随负载翻转），为什么还值得写」？买到了什么、代价是什么、
  什么规模下账会翻过来？
- 能不能解释「两次 `ray.get` 同一个大数组，为什么返回的是『同一块』内存、
  为什么必须只读」？store 装不下时 put 为什么不报错——它把什么换成了什么？

## 权威实现与延伸

- 对标源码：Ray `github.com/ray-project/ray`（task/actor、object store、调度器）
- 概念延伸：把 `nano-data-juicer` 的 OP 分布式化
