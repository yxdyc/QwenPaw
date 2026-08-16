# nano-data-juicer · Tutorial L2 — 分布式 pipeline：分区、并行、全局 OP 收敛点

> **级别**：L2（分布式 / 性能）。前置：本模块 [L0](tutorial_L0.md)（OP 可组合 + 配置驱动）、[L1](tutorial_L1.md)（真实数据 + LLM OP）、[nano-ray L0](../nano-ray/tutorial_L0.md)（任务图调度）。
> **K+1 声明**：L0/L1 的 pipeline 是单进程串行的——`op(list) -> list`，全量数据必须装进一个进程。L2 只加一层：**把同一个 pipeline 搬到多个 worker 上跑，并且保证结果和串行逐位一致**。不碰真实集群、不碰 GPU。
> **可运行性声明（课程可运行性契约）**：L2 默认不依赖 Ray，而用标准库 `multiprocessing` 的**真实 worker 进程**实现分布式执行语义（分区 / 并行 / 收敛 / 重算）；真实多机 Ray 行为标 `[TODO: verify on real system]`。语料为固定 seed 的合成数据（L2 的主题是执行语义，需要足够样本量才能测出并行加速；L1 的 10 条真实样本测不出加速）。

---

## 1. 问题：串行 pipeline 在哪里断掉

L1 的 pipeline 长这样：

```python
cur = samples
for name, op in steps:
    cur = op(cur)          # 全量 list 一次过，单进程
```

数据到 TB 级时，这条路在三个地方断掉：

1. **装不下**：全量数据进不了单机内存，必须切块（partition）流式处理；
2. **太慢**：单机 CPU 核数有限，必须把块分给多个 worker（多进程 / 多机）并行；
3. **不都是可并行的**：Mapper / Filter 逐样本独立，切块各做各的没问题；但 **Deduplicator 需要看到全量数据**——重复的两条可能落在任意两个不同的块里，块内各做各的去重是**错的**。

所以 L2 的核心不是「把循环改成并行」这么浅，而是一个执行计划问题：

> **哪些 OP 可以分区并行？哪些 OP 是全局的、必须在它前面插入收敛点（convergence point）？坏了怎么重算？**

这正是 Data-Juicer 从 `DefaultExecutor`（单机多进程）走向 `RayExecutor` / `PartitionedRayExecutor`（分布式）时解决的事。

---

## 2. 运行与输出

```bash
$ python3 L2_distributed_pipeline.py
```

以下为一次真实运行逐字粘贴（连跑 3 遍：除计时行外逐字节一致；计时波动区间见 §5.4）：

```text
====================================================================
nano-data-juicer L2 — distributed pipeline
====================================================================
config: docs=3000 dups=360 partitions=4 workers=4 seed=42
声明: 合成语料(固定seed) + multiprocessing 真实进程并行；
      分布式语义的本质模拟，真实 Ray 集群见 [TODO: verify on real system]

[1] 语料: 3360 docs, 3.74 MB (含 360 条注入重复), 生成耗时     60.1 ms
[2] 分区: 4 partitions, sizes=[840, 840, 840, 840], round-trip 无损 ✅

[3] 局部 OP 阶段 (normalize_mapper + length_filter)
    漏斗: 3360 -> 2358 条 (length_filter 过滤 1002 条短文档)
    serial  :    301.4 ms
    parallel:    118.3 ms  (speedup 2.55x)
    并行结果 == 串行结果: True ✅（顺序逐位一致）

[4] 全局 OP 阶段 (exact_deduplicator)
    a) 反例: 把 dedup 当局部 OP 分区各做各的
       结果条数 = 2346, 泄漏跨分区重复 236 条 ❌
    b) convergence 策略 (union 后全局去重): 2110 条
    c) shuffle 策略 (按 hash(key) 重分区): 2110 条
    串行基准: 2110 条
    两策略与串行基准逐位一致 ✅
    重复对账本(过滤后幸存者): 同分区内 12 对 | 跨分区 236 对
    naive 只抓得到同分区对，跨分区对每对泄漏 1 条 => 泄漏数 236 vs 跨分区对数 236

[5] 端到端 pipeline 计时
    serial              :    290.0 ms
    distributed(conv)   :     97.3 ms  speedup 2.98x
    distributed(shuffle):    106.5 ms  speedup 2.72x
    执行计划 (convergence):
      - normalize_mapper: local, parallel on 4 partitions
      - length_filter: local, parallel on 4 partitions
      - exact_deduplicator: GLOBAL -> convergence (union 4 partitions, serial dedup)
    Amdahl: convergence 版的全局去重段是串行的，
    并行加速只作用于局部 OP 段 => speedup 有上限。

[6] 分区级容错 (注入 partition 2 首次执行崩溃)
    [fault] [injected] worker for partition 2 crashed
    attempts per partition: {0: 1, 1: 1, 2: 2, 3: 1}
    recomputed partitions : [2]
    其余 3 个分区结果直接复用，未重算 ✅

====================================================================
✅ self-check passed:
   分区无损 round-trip / 局部 OP 并行==串行逐位一致 /
   naive 分区去重必泄漏 / convergence==shuffle==串行 /
   重复对账本吻合 / 容错只重算崩溃分区
====================================================================
```

---

## 3. 输出逐段解读

**[1][2] 语料与分区。** 3000 条基础文档 + 360 条注入重复 = 3360 条、3.74 MB。重复的注入方式是「复制某条文本、做大小写/空白扰动、放到更后的位置」——原文不相等、**规范化后相等**，所以 dedup 必须发生在 normalize 之后才抓得到（L0「OP 顺序是语义的一部分」在分布式下的回声）。分区是连续 chunk 切分（840×4），round-trip 拼回必须无损——这是后面一切正确性断言的地基。

**[3] 局部 OP 并行。** normalize_mapper + length_filter 都是逐样本独立的**局部 OP**：4 个分区丢进 4 个 worker 进程各做各的，按分区顺序拼回。关键断言不是加速比，而是 **`并行结果 == 串行结果: True` 逐位一致**——包括顺序。加速 2.55x（随机器负载浮动，观测区间约 2.4x–3.3x），不是 4x，原因见 §5.4。

**[4] 全局 OP 是本节的戏眼。**

- **a) 反例**：不识别 dedup 的全局性、把它当局部 OP 分区各做各的——结果 2346 条，比正确答案 2110 条多出 **236 条跨分区重复**。这些重复对的两份拷贝躺在不同分区里，任何一个分区的局部视野都看不到「对方」，自然去不掉。
- **账本**：过滤后幸存的重复对共 248 对，其中同分区内 12 对（naive 能抓到）、跨分区 236 对（naive 全漏）。泄漏数 = 跨分区对数，一分不差——「naive 错在哪、错多少」从断言变成了算术。
- **b) convergence**：Data-Juicer partitioned executor 的做法——全局 OP 前把所有分区 union 成一个整体，串行做全局去重。
- **c) shuffle**：Spark/MapReduce 的做法——按 `hash(dedup_key) % P` 重分区，相同 key 必落同一分区，于是「分区内去重」重新变成正确的，全程并行，最后按 `row_id` 排序还原全局顺序。
- 两种策略结果与串行基准**逐位一致**（不只是条数相等，是逐样本、逐顺序相等）。

**[5] 端到端。** 串行 290 ms；convergence 版 97 ms（2.98x）；shuffle 版 106 ms（2.72x）。注意这里 **convergence 反而比 shuffle 快**——不是写反了，机制见 §5.5 反例 3。执行计划日志把「哪个 OP 是 local、哪个触发 GLOBAL 收敛」打成了显式的一行行 plan。

**[6] 分区级容错。** 注入 partition 2 的 worker 首次执行崩溃：executor 只重算 partition 2（attempts `{0:1, 1:1, 2:2, 3:1}`），其余 3 个分区的结果直接复用。重算的最小单位是**分区**——这正是 Ray lineage 重算与 Data-Juicer per-partition checkpoint 的粒度。

---

## 4. 代码结构：OP 的类别决定执行语义

### 4.1 OpSpec：把「能不能并行」变成 OP 的一等属性

```python
@dataclass
class OpSpec:
    name: str
    kind: str          # "mapper" | "filter" | "deduplicator"
    fn_map:  Optional[Callable[[Sample], Sample]] = None
    fn_keep: Optional[Callable[[Sample], bool]]  = None
    dedup_key: Optional[Callable[[Sample], str]] = None

def is_global_operation(op: OpSpec) -> bool:
    return op.kind == "deduplicator" or "dedup" in op.name.lower()
```

L0/L1 里 OP 只是 `list -> list` 的函数；L2 给 OP 加了**类别**。执行器不再「对每个 OP 一视同仁地循环」，而是先问一句：*你是局部的还是全局的？* 这句问话在 Data-Juicer 里真实存在——`is_global_operation()`（`data_juicer/core/executor/dag_execution_strategies.py:L441-471`），判定优先级三级：显式标志 → `isinstance(op, Deduplicator)` 基类判定 → 名字模式兜底（`deduplicator` / `global_` / `full_dataset_`）。

### 4.2 分区与 worker 函数

```python
def partition(docs, n):
    size = (len(docs) + n - 1) // n
    return [docs[i*size:(i+1)*size] for i in range(n) if i*size < len(docs)]
```

连续 chunk 切分，保持分区内顺序——对应 Ray Data 的 `dataset.split(n)`（`ray_executor_partitioned.py:L1292`）。

worker 侧只有两个函数，都是**顶层函数**：

```python
def _apply_local_op(args):        # (partition, op) -> processed partition
    part, op = args
    if op.kind == "mapper":
        return [op.fn_map(s) for s in part]
    if op.kind == "filter":
        return [s for s in part if op.fn_keep(s)]

def _bucket_dedup(args):          # (bucket, key) -> deduped bucket
    bucket, key = args
    best = {}
    for s in bucket:
        k = key(s)
        if k not in best or s["row_id"] < best[k]["row_id"]:
            best[k] = s
    return list(best.values())
```

一个真实的工程坑：**spawn 模式下闭包无法 pickle 进 worker 进程**。第一版把 `_bucket_dedup` 写成 `distributed_run` 内的闭包（捕获外层 `key`），直接报错；改成顶层函数、把 `key` 作为参数传进去才好。分布式代码里「什么能被序列化送到别处」是第一性问题——Ray 的 remote 函数同理。

### 4.3 distributed_run：执行计划 = 局部并行 + 全局收敛

骨架只有十几行：

```python
for op in ops:
    if not is_global_operation(op):
        cur_parts = pool.map(_apply_local_op, [(p, op) for p in cur_parts])
    else:                                   # ---- 收敛点 ----
        if dedup_strategy == "convergence":
            merged = [s for p in cur_parts for s in p]      # union
            cur_parts = [_dedup_keep_first(merged, op.dedup_key)]
        elif dedup_strategy == "shuffle":
            buckets = repartition_by_hash(cur_parts, key)   # hash(key) % P
            deduped = pool.map(_bucket_dedup, [(b, key) for b in buckets])
            merged = sorted(concat(deduped), key=row_id)    # 还原全局顺序
            cur_parts = [merged]
```

两个策略都保证「保留全局顺序下的首次出现」：convergence 靠 union 后按原顺序扫描；shuffle 靠每个 bucket 内保留 `row_id` 最小者 + 最后按 `row_id` 排序。**`row_id` 是全局顺序的唯一载体**——分区打乱物理位置之后，语义顺序必须由数据自己携带。

### 4.4 worker 池复用

`main()` 里 `mp.Pool(4)` 只建一次、全程复用。每个 stage 新建池会反复支付进程 spawn 成本（macOS 上每进程约 0.1–0.3 s）——对应真实系统里常驻 worker / Ray actor 的存在理由。

---

## 5. 机制深挖：四个「为什么」

### 5.1 为什么 Mapper/Filter 可以分区并行

`normalize_mapper` 和 `length_filter` 是**逐样本函数**：输出只依赖当前这一条输入，不依赖任何其他样本。于是对任意分区方案，「先分区各做各的、再按分区顺序拼回」与「全量串行」产生**同一个输出序列**——这不是实测巧合，是函数独立性的算术必然。本文件把它写成硬断言（`dist_local == serial_local` 逐位相等），每次运行都验证一遍。

Data-Juicer 的单机并行就是这个机制：`Mapper.run` 里 `dataset.map(self.process, num_proc=self.runtime_np(), ...)`（`ops/base_op.py:L606, L703`），HF datasets 在 `num_proc` 个进程上分片跑同一个 map。Filter 是两段：先并行 `compute_stats`（`L836`），再并行 `filter`（`L856`）——统计与判定分离，统计结果还能落盘复用。

### 5.2 为什么 Deduplicator 是全局 OP

看 Data-Juicer 自己的 `Deduplicator.run`（`ops/base_op.py:L867-879`）：

```python
new_dataset = dataset.map(self.compute_hash, num_proc=self.runtime_np(), ...)
# ...
new_dataset, dup_pairs = self.process(new_dataset, show_num)
```

**指纹计算是并行的（`compute_hash` 逐样本），去重判定是全局的（`process(dataset)` 吃整个 dataset）**。权威实现把这条边界画得清清楚楚：能并行的部分（算 hash）尽量并行，不能并行的部分（判定谁是重复）老实收拢。

为什么判定必须全局？重复对 `{a, b}` 的判定依赖「a 和 b 同时出现在视野里」。分区方案对此一无所知——a、b 可能落在任意两个分区。任何「分区内各做各的」方案都存在一个反例输入（把一对重复拆到两个分区），所以它在**语义上**就是错的，不是实现不够好。[4]a 的 236 条泄漏就是这个反例的实测。

识别全局 OP 并插入收敛点，在 Data-Juicer 里是一条真实代码路径：`_detect_convergence_points`（`dag_execution_mixin.py:L236-243`）扫描 OP 列表，凡是 `is_global_operation(op)` 为真的位置记为收敛点；partitioned executor 在 run 时据此分派 `_process_with_convergence`（`ray_executor_partitioned.py:L493-497, L872`）。

### 5.3 两种收敛策略的本质差异

| | convergence（union 后全局做） | shuffle（按 key 重分区后并行做） |
|---|---|---|
| 数据移动 | 所有分区合并成一份 | 全量按 hash 重分区（一次 shuffle） |
| 全局阶段 | 串行扫描（Amdahl 串行段） | 并行（bucket 间独立） |
| 顺序还原 | 天然保持（union 按分区序） | 需按 row_id 排序 |
| 实现复杂度 | 低 | 高 |
| 适用 | 全局 OP 本身轻量（exact dedup = O(N) 扫描） | 全局 OP 昂贵（如两两相似度）或单点装不下 |

Data-Juicer 的 partitioned executor 选了 convergence：pre-convergence 的 OP 分区并行跑完 → union 合并 → post-convergence 的 OP 在合并后的数据集上跑（`ray_executor_partitioned.py:L872-922`）。这个选择在 exact dedup 场景是合理的：去重判定本身是 O(N) 的哈希集合扫描，串行不贵；而 shuffle 要付一次全量重分区 + 排序。本文件的实测也支持这一点（§5.5 反例 3）。

但 convergence 的代价在**规模**上：合并意味着单个节点要装下全量中间数据。数据装不下时，shuffle（或 dedup 专用的两级方案：先局部去重、再对指纹做全局聚合）才是出路。真实集群上两者的交叉点在哪，标 `[TODO: verify on real system]`（真实 GPU/多机环境）。

### 5.4 为什么 4 worker 只测出 2.4x–3.1x

三个诚实的原因：

1. **IPC 序列化**：3.74 MB 的样本要 pickle 送进 worker、结果再 pickle 送回。数据搬运是分布式的固有成本——真实系统里它变成网络传输，更贵。
2. **Amdahl**：分区切分、结果拼接、（convergence 版的）全局去重段都是串行的。
3. **任务数 = 分区数 = 4**：没有更细的粒度给调度器做负载均衡（Data-Juicer 的 partition 默认 size=5000 条、上限 64 MB——`ray_executor_partitioned.py:L290-293`——就是在控制这个粒度）。

计时随机器负载浮动（机器更空闲时会高于初次记录时观测的区间）：[3] 局部阶段 speedup 观测区间约 2.4x–3.3x；[5] 端到端 convergence 约 2.9x–3.2x、shuffle 约 2.6x–3.0x。区间覆盖初次多次运行与后续独立复测（最高 3.28x / 3.21x / 3.00x）多批测量。区间下界不是硬保证：2026-08-06 一次复跑在并发只读命令的负载干扰下落到 1.80x（parallel 段 171.9 ms vs 正常约 92 ms）。趋势不变：**并行显著快于串行，但达不到 worker 数**。

### 5.5 反例与边界（三件）

1. **全局 OP 当局部 OP 跑 ⇒ 静默错误**。这是本节最重要的反例：naive 方案不报错、不崩溃、条数看起来也「差不多」（2346 vs 2110），但训练集里混进了 236 条重复。数据 pipeline 的错误往往不是异常而是**分布污染**——所以 [4] 的账本断言（泄漏数 = 跨分区对数）必须是自动自检的一部分。
2. **并行不一定快**。把 [3] 的 mapper 换成 µs 级轻操作，IPC 会吞掉全部收益（nano-ray L0 [4] 的粒度反例在数据 pipeline 里的同款）。本文件刻意让 mapper 做真实 CPU 负载（字符二元组直方图 + 双层哈希 + 词表统计），因为 L2 要测的是「并行机制」而非「序列化开销」。
3. **convergence 不一定比 shuffle 差**。实测 convergence（95–100 ms）快于 shuffle（104–109 ms）：全局去重本身太轻，shuffle 的 bucketing + 排序反而更贵。「shuffle 更先进」是错的直觉——**选策略要看全局 OP 的计算复杂度和数据规模，不是看谁更分布式**。

---

## 6. 与 Data-Juicer 权威实现的对应

> 行号全部按 `github.com/modelscope/data-juicer` **main 分支**口径标注，2026-08-05 两次现场核验：第一次 raw 文件抓取，同日经 codeload main tarball 解包逐条复核，**29 处锚点无一漂移**。注意本地 checkout 为 `report_enhance@4e40654`（2026-05-11），行号与 main 有漂移（`base_op.py` main 约高 51 行、`dag_execution_mixin.py` 约高 16 行、`ray_executor_partitioned.py` 非均匀）——本教程引用一律以 main 为准，本地 checkout 只作机制交叉阅读。上游迭代可能再漂移。

| toy 部件 | Data-Juicer 对应 | 源码锚点 |
|---|---|---|
| `executor_type` 选择执行器 | `ExecutorFactory`：`default` / `ray` / `ray_partitioned` | `core/executor/factory.py:L8-14`；默认值 `config/config_all.yaml:L82` |
| 串行 OP 循环（L1 语义） | `NestedDataset.process`：`for idx, op in enumerate(operators): dataset = op.run(...)`，逐 OP 打耗时与剩余条数日志 | `core/data/dj_dataset.py:L254-349`（循环 L289、执行 L303、日志 L310-312） |
| `NUM_WORKERS`（cfg） | `self.np = cfg.get("np", None) or 1` | `core/executor/default_executor.py:L59` |
| 局部 OP 并行 | `Mapper.run` → `dataset.map(self.process, num_proc=self.runtime_np())`；`Filter.run` 两段（compute_stats + filter） | `ops/base_op.py:L606, L703`（Mapper）；`L717, L836, L856`（Filter）；`runtime_np` 自动并发度 `L523` |
| 指纹并行 + 判定全局 | `Deduplicator.run`：`dataset.map(self.compute_hash, num_proc=...)` 后 `self.process(dataset)` | `ops/base_op.py:L867, L922, L926` |
| `is_global_operation` | 同名函数，三级判定（标志 → Deduplicator 基类 → 名字模式） | `core/executor/dag_execution_strategies.py:L441-471` |
| 执行计划里的收敛点 | `_detect_convergence_points` | `core/executor/dag_execution_mixin.py:L236-243` |
| `partition()` chunk 切分 | `dataset.data.split(self.num_partitions)` + 确定性开关 `preserve_order=True` | `core/executor/ray_executor_partitioned.py:L1292, L1174-1175` |
| 分区默认值（P=4） | `num_of_partitions=4`、`size=5000`、`max_size_mb=64` | `ray_executor_partitioned.py:L290-293` |
| convergence 策略 | `_process_with_convergence`：pre-convergence OP 分区跑 → union 合并 → post-convergence OP 全局跑 | `ray_executor_partitioned.py:L872-922`；union 合并 `L600`；分派 `L493-497` |
| [6] 分区级重算 | per-partition checkpoint（按 partition_id 存取、按 op group 恢复） | `ray_executor_partitioned.py:L923-1058`；分区元数据（行数 + 首行哈希）校验 `L1192, L1231-1277` |
| RayExecutor 的能力边界 | 类 docstring 自述：「Support Filter, Mapper and Exact Deduplicator operators for now；Only support loading .json files；checkpoint not supported」 | `core/executor/ray_executor.py:L50-52` |

**nano 版与权威实现的差异及原因**：

- **单机多进程 vs Ray 集群**：本机无 Ray，用 `multiprocessing.Pool` 承载同一套语义。差异在数据移动成本（进程间 pickle vs 集群网络）与故障域（进程退出 vs 节点宕机），执行计划的形状 相同。
- **shuffle 策略是本文件加的教学对照组**：Data-Juicer 的 partitioned executor 只实现 convergence（union 后全局做）。加 shuffle 是为了把「全局 OP 的两条出路」都摆出来让学习者比较——实测也说明 convergence 在 exact dedup 场景不落下风（§5.5 反例 3）。
- **checkpoint 只在内存**：[6] 的「分区结果缓存」是内存 dict，不落盘；Data-Juicer 的 `RayCheckpointManager` 落盘且带分区元数据校验（行数 + 首行哈希，防止恢复时分区方案漂移）。
- **分区验证没做**：Data-Juicer 恢复作业时会校验「重新切分出的分区与上次一致」（否则 checkpoint 作废）——这是「确定性切分」在工程上的真正用途，toy 版只在 §5.3 提及。

---

## 7. 费曼自检

**类比：快递分拣中心。** 本地 OP 就像各站点处理自己片区的包裹——拆包检查（mapper）、拒收违禁品（filter），站点之间互不依赖，片区怎么划都不影响最终结果。但「查找重复下单的包裹」是全局问题：同一个客户可能在城东城西各下了一单，任何单站只看自己的件，永远发现不了跨站重复。两条出路：要么把全部面单**汇到总仓**统一查（convergence）——简单，但总仓要装得下所有面单；要么**按运单号尾数重新分拣**，尾数相同的必进同一站，站内查重就一定完备（shuffle）——多了一次全量重分拣，但没有任何一站需要装下全局面单。至于「某站的扫描枪坏了」——只让那一站重扫一遍，其他站的结果不动（分区级重算）。

**一句话版**：逐样本的活可以分着干，跨样本的活必须先凑齐再干（或按 key 重排让同组相遇）；坏了只重算坏的那一块。

**边界声明**：类比里「总仓装不装得下」对应单节点内存上限，真实系统还叠加网络带宽与序列化成本；「按尾数分拣」假设 key 分布均匀，真实数据倾斜（hot key）会让某个 bucket 过载——这是 shuffle 方案的经典失败模式，toy 里未模拟。

**反例版**：如果有人说「每个站点各自查重，最后把结果拼起来也一样」——那就是 [4]a 的 naive 方案，实测泄漏 236 条。跨站重复在局部视野里**不存在**，拼多少个局部视野都补不出来。

---

## 8. 思考题

1. **两个全局 OP 的执行计划**：如果 pipeline 是 `normalize → exact_dedup → simhash_dedup`（两个全局 OP），执行计划里有几个收敛点？第一个全局 OP 之后数据已经合并成单分区，后面的局部 OP（如果有）还想并行，需要做什么？（提示：看 `ray_executor_partitioned.py:L872-922` 里 post-convergence OP 跑在什么形态的数据上。）
2. **row_id 排序的必要性**：shuffle 策略最后为什么要按 `row_id` 排序？如果需求从「保留首次出现」放宽为「每个重复组随便保留一条」，这次排序能省吗？省掉之后输出还确定吗？（提示：bucket 内 `dict` 的迭代顺序与输入顺序的关系。）
3. **worker 池复用**：把 `main()` 里的 `pool = mp.Pool(4)` 挪进 `distributed_run` 每次新建，预测 [5] 的 speedup 会怎么变，实测验证。这个成本在真实 Ray 集群上对应什么？
4. **能力边界**：Data-Juicer 的 `RayExecutor` 自述只支持 Filter / Mapper / Exact Deduplicator（`ray_executor.py:L50-52`）。结合本节的局部/全局分类，猜一猜 Selector / Aggregator 为什么不在支持列表里——它们是哪类 OP？（提示：Selector 从全量里选 top-k，Aggregator 把多条聚合成一条。）

---

## 9. 阶梯预告

- **L3（✅ 已交付）**：[tutorial_L3.md](tutorial_L3.md)——对照 Data-Juicer 真实 OP 接口，复现一个 filter 的完整行为 + 配置 schema（stats 字段、区间判定语义、NON_STATS_FILTERS 等），把 L2 的执行语义与 L3 的 OP 语义拼成完整图景；含与 L2 漏斗逐位一致的跨级别契约。
- **解锁**：本模块到 L2 后，03 轨 sota-deepdive（数据方法论：FineWeb / DCLM / Nemotron 数据报告）的开写门槛满足。

**交叉引用**：[nano-ray L0](../nano-ray/tutorial_L0.md)——数据在 worker 之间怎么流动（object store / 任务图）是 L2「谁搬数据」的另一半；[nano-vllm-sglang L0](../nano-vllm-sglang/tutorial_L0.md)——batching 是另一种「攒起来一起处理」，与分区互补；本模块 [L0](tutorial_L0.md) / [L1](tutorial_L1.md)——OP 组合与顺序语义是 L2 执行计划的前提。

---

## 10. 溯源与口径声明

- **源码锚点**：§6 表格全部行号于 2026-08-05 现场核验两次——第一次 `raw.githubusercontent.com/modelscope/data-juicer/main/...` 抓取；同日（raw 服务不可达时段）改经 codeload main tarball 解包复核，29 处锚点零漂移。本地 checkout `report_enhance@4e40654`（2026-05-11）行号与 main 有漂移，仅作交叉阅读，引用口径以 main 为准。
- **toy 口径**：所有计时数字是本机（Apple Silicon, Python 3.13, multiprocessing spawn）真实运行输出，非 benchmark；合成语料固定 seed=42，除计时外逐字节可复现（连跑 3 遍 diff 验证）。
- **[TODO: verify on real system]**（真实 GPU/多机环境验证）：① 真实 Ray 集群上 `split` + convergence 的多节点行为与本机模拟的差异；② convergence vs shuffle 在「单节点装不下全量」规模下的交叉点；③ `preserve_order=True` 在真实 Ray Data 上的性能开销。
- **未核验项如实标注**：Data-Juicer partitioned executor 在 main 分支的启用方式（配置入口）未逐行核验，本教程只引用其源码机制，不声称「生产默认启用」。
