# 分布式去重与多模态数据治理：从 group-by 到可撤销的相似性图

> **对齐日期**：2026-09-24
> **覆盖问题**：Spark vs Ray group-by；Data-Juicer BTS MinHash；论文中的 `3.3×`；
> Data-Juicer × Ray Data；`map` vs `map_batches`；PB 级增量去重；图像、视频、音频、PDF、
> 图文/音视频 pair 与 interleaved 数据的去重、清洗和打标。
> **前置材料**：[数据方法论](data-methodology.md)解释 MinHash/LSH、质量过滤、配比和去污染；
> [多模态媒体数据管线](../../05-multimodal-understanding-generation/MEDIA_DATA_PIPELINE.md)
> 解释解码、采样、manifest 与基础媒体账本。本文只补足分布式物理执行、增量状态和分模态治理。

---

## 0. 先给结论

1. **Spark 更适合 group-by，不是因为 hash 函数更快，而是因为它把**
   `局部聚合 → shuffle → spill → 最终聚合 → 失败重算 → commit`
   **做成了成熟的系统协议**。这个判断主要成立于 Spark SQL/DataFrame 对 plain Ray Core；
   Ray Data 已经有 group-by、aggregate 和 shuffle，差距比“Spark vs Ray”这句话暗示的小。
2. Data-Juicer 的 Ray MinHash 去重，不只是 `groupby(band_hash)`：它先生成 LSH 候选边，
   再把 connected components 问题交给一组 Ray actor，以 BTS 式负载均衡 union-find 收敛，
   并用 hash aggregation 压掉重复、碎片化的 union 通信。
3. Data-Juicer 2.0 的 `3.3×` 是**特定版本、特定 baseline 下的端到端实验结论**。
   论文给出了优化机制，却没有披露足够的机器、数据、Ray 版本、参数与逐阶段时间，
   因而不能推出“所有数据上都比当前 Ray Data 快 3.3×”。
4. 增量去重不是“给新数据查一次 Bloom filter”，而是持续维护一个
   **有版本、可审计、可撤销的相似性图**。必须同时处理 `new↔old`、`new↔new`、历史簇合并、
   canonical 更换、删除造成的簇分裂以及算法升级。
5. 多模态去重没有一个通用 embedding 能解决全部问题。至少要分开维护：
   **资产重复**、**局部/变换后重复**、**语义重复**、**pair 重复**与**泄漏/污染**。
6. 清洗和打标也不是“一个大模型给分”。可靠的生产漏斗是：
   `格式/解码硬门 → 信号统计 → 小模型打标 → 大模型语义审计 → 分层抽检 → 下游训练消融`。

本文使用以下证据标签：

- **[论文原文]**：论文明确报告；
- **[源码事实]**：当前主分支代码可以直接观察；
- **[官方文档]**：项目文档明确声明；
- **[工程推断]**：由机制推导出的设计建议，不冒充作者结论；
- **[未披露]**：公开证据不足，不能反推。

---

## 1. 为什么 group-by 是这组问题的共同地基

MinHash 去重、图连通分量、每来源质量统计、多模态 embedding 聚类最终都会落到同一个物理问题：

> 大量记录原本散在各个分区；现在需要让相同 key 的状态在某处汇合。

设输入共 $N$ 行、$P$ 个输入分区、$R$ 个输出分区。先区分两种名字相同、代价完全不同的操作。

### 1.1 可合并聚合：可以先把百万行压成一个 accumulator

例如：

```text
groupBy(key).count/sum/min/max
```

若聚合满足结合律，每个 map 分区可以先做 partial aggregation。设第 $p$ 个分区有 $U_p$ 个不同 key：

$$
\text{shuffle bytes}_{partial}
\approx \sum_{p=1}^{P}U_p\,(|key|+|accumulator|)
$$

没有局部合并时则接近：

$$
\text{shuffle bytes}_{raw}\approx N\,(|key|+|value|)
$$

若一亿条记录只有十万个 key，局部 combine 可能把网络量降低几个数量级。

### 1.2 整组 UDF：最大 hot key 决定能否活下来

例如：

```text
groupBy(key).mapGroups(fn)
groupByKey(key) -> Iterable[rows]
```

它不能一般性压缩原始值，网络量仍接近 $O(N)$，且最大组的内存下界约为：

$$
M_{max}\approx n_{max}\,|row|
$$

一个包含数亿条空网页的 `hash(empty_text)`，足以让 Spark、Ray 或任何引擎的单 reducer OOM。
所以讨论“谁更擅长 group-by”前，先问：**我要 accumulator，还是要整个 group？**

---

## 2. Spark 为什么通常更适合 group-by；边界又在哪里

### 2.1 Spark SQL 看得见聚合语义

典型物理计划是：

```text
scan
  -> filter/project
  -> partial HashAggregate
  -> Exchange hashpartitioning(key)
  -> final HashAggregate
  -> write
```

Spark SQL 的 planner 知道 grouping key、aggregate buffer、partial/final mode，因此可以做列裁剪、
predicate pushdown、代码生成、hash/sort aggregate 选择。其
[Spark SQL 论文](https://people.eecs.berkeley.edu/~matei/papers/2015/sigmod_spark_sql.pdf)
解释了 Catalyst 与声明式物理规划；当前
[`HashAggregateExec`](https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/execution/aggregate/HashAggregateExec.scala)
源码也明确包含内存压力下向 sort-based 聚合回退及 spill 指标。

plain Ray Core 提供的是 task、actor 与 object reference。它能表达这套协议，但不会从任意 Python task
自动推断“这些任务共同构成一个可局部合并的关系聚合”。若直接用 Ray Core 自建，就要自己实现：

```text
partition -> local combine -> shard routing -> backpressure/spill
          -> reducer lifecycle -> skew handling -> retry -> commit
```

这不是 Ray 做不到，而是做完后已经在实现一个数据引擎。

### 2.2 Spark 把 shuffle 当一等执行阶段

Spark 的 sort-based shuffle 会按目标 partition 写带索引的 map output，内存不足时 spill 并 merge，
而不是朴素地产生 $P\times R$ 个微小文件。实现约束见
[`SortShuffleManager`](https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/shuffle/sort/SortShuffleManager.scala)。

它还把 stage、shuffle block 生命周期、lineage 重算和 speculative execution 组合成 bulk analytics 的
默认恢复协议。Ray task 也有 lineage reconstruction，不能说“Ray 没容错”；差别是 actor 状态、
object owner、shuffle commit 等边界通常需要应用或 Ray Data 层处理。

### 2.3 Spark 的成熟优势不等于“groupByKey 随便用”

Spark 官方同样建议可聚合场景优先 `reduceByKey`/`aggregateByKey`，避免 materialize 全组，见
[RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide)。AQE 可以根据实际
shuffle statistics 合并小分区，并改善部分 skew 情况，见
[Spark SQL Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning)；
但它无法魔法般消除一个必须逻辑汇合的超热 key。常见处理仍是：

- map-side combine；
- heavy-hitter 预检测与独立路径；
- salting 后两阶段聚合；
- 只传 accumulator/代表样本，而不是完整成员；
- 把大簇成员清单分页写出，canonical 选择另算。

### 2.4 Ray Data 已经不是 plain Ray Core

Ray Data 是运行在 Ray Core 之上的批/流式数据执行层，已经提供 `groupby`、built-in aggregation、join、
repartition 与 shuffle。其[数据内部机制](https://docs.ray.io/en/latest/data/data-internals.html)说明经典
hash shuffle 会让 block 按 key 切分并送往聚合 actor；`AggregateFnV2` 则支持 block 级 accumulator
和 combine。当前 Shuffle v2 进一步把中间数据放入 object store、利用 spill 并合并同节点小 shard，
但官方仍把它标为 Alpha，部署前要验证 head-node 小对象压力与恢复路径。

Ray 官方对 [`map_groups`](https://docs.ray.io/en/latest/data/api/doc/ray.data.grouped_data.GroupedData.map_groups.html)
的边界也写得很直接：它比专用 aggregate 慢，而且单个 group 必须装入一个节点内存。

因此选择表应写成：

| 场景 | 默认倾向 | 原因 |
|---|---|---|
| PB 级 SQL/ETL，反复 join/group-by | Spark SQL | 成熟 planner、shuffle、spill、AQE、运维生态 |
| 数据已在 Ray，前后接 GPU inference/Ray Train | Ray Data | 避免跨系统搬运；task/actor/GPU 统一调度 |
| 每个 group 内是昂贵模型推理或动态工作流 | Ray | 模型计算压过 shuffle，actor 复用模型状态 |
| plain Ray Core 写常规聚合 | 通常不建议 | 需要自行重造数据执行协议 |
| 单个 group 巨大 | 两者都需改算法 | 引擎选择不能消除 $M_{max}$ 下界 |

**可证伪的基准方式**：固定数据、机器、序列化格式、分区数和语义，分别报告 scan、partial、shuffle、
spill、final 与 commit 时间；同时报告 shuffle bytes、spill bytes、peak RSS、失败重试。只比较总 wall time
无法解释差异来自 planner、网络、Python UDF 还是缓存命中。

---

## 3. MinHash 去重从概率问题变成图问题

### 3.1 候选召回

将文档变为 token shingle 集合 $S_d$，Jaccard 为：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

MinHash 用 $K$ 个 hash minimum 组成定长 sketch。将 $K=br$ 个值切成 $b$ 个 band、每 band $r$ 行，
相似度为 $s$ 的文档至少一 band 相同的概率为：

$$
P(candidate\mid s)=1-(1-s^r)^b
$$

LSH 只是**候选生成器**。正确的漏斗是：

```text
document
  -> shingles
  -> MinHash signature
  -> (band_id, band_hash, doc_id)
  -> group same band key
  -> candidate pairs
  -> exact Jaccard/edit/containment verifier
  -> duplicate edges
  -> connected components / constrained clustering
```

### 3.2 为什么最后必须解 connected components

假设 A 与 B、B 与 C 都过阈值，即使 A 与 C 没有在任一 band 碰撞，工程上通常仍需要给三者一个
共同 duplicate cluster。这把问题从“查表”变成了图连通分量：

```text
A -- B -- C      =>      root(A)=root(B)=root(C)
```

朴素做法把每个 LSH bucket 内 pair 全部送给统一 union-find，容易同时出现：

- hot bucket 产生 $O(m^2)$ pair；
- 同一条边从多个 band 重复出现；
- 图边跨 worker 往返；
- 某个根节点吸收大簇，worker 极度倾斜；
- 反复小 union 造成调度与对象开销。

这正是 Data-Juicer 的 BTS 路径要优化的部分。

---

## 4. Data-Juicer 的 BTS MinHash Ray 去重到底做了什么

### 4.1 Data-Juicer 与 Ray 的结合层

**[论文原文]** [Data-Juicer 2.0](https://arxiv.org/abs/2501.14755)把 `DJDataset` 作为统一 facade，
下接 Hugging Face Dataset、Ray Data 与 MaxFrame；分布式执行路径使用 Ray Dataset 与 RayExecutor。

**[源码事实]** 当前
[`RayDataset`](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/ray_dataset.py)
内部持有真正的 `ray.data.Dataset`，常规 operator 会落到 Ray Data 的 `map`、`map_batches`、`filter`
等接口：

- 纯 CPU、无长期状态的 operator 通常用 task pool；
- 需要加载模型或 GPU 的 callable class 通常用 actor pool，以复用模型权重；
- `num_cpus`、`num_gpus`、`memory`、`runtime_env` 等透传给 Ray；
- 去重这类需要全局 shuffle/迭代收敛的 operator 不硬塞进逐行 map，而走专门的 engine-native `run`。

因此关系是：

```text
Data-Juicer config/operator semantics
          |
       DJDataset
          |
  RayDataset + RayExecutor
          |
 Ray Data blocks/map_batches/filter ---- Ray Core tasks/actors/object store
          |
 special global op: BTS MinHash actors + iterative graph convergence
```

“Data-Juicer 使用 Ray”和“所有东西都直接用 Ray Data built-in”不是同一句话。常规局部算子复用 Ray Data，
而 BTS 去重会下沉到 Ray actor/remote call 来表达专门的图算法。

### 4.2 当前 BTS 实现的阶段图

以当前
[`ray_bts_minhash_deduplicator.py`](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/ops/deduplicator/ray_bts_minhash_deduplicator.py)
为源码锚，机制可压缩为六步：

```text
[1] map_batches
    文档 -> tokenize/shingle -> MinHash signature -> band records
                         |
[2] hash route          v
    (band_hash, uid) -> owner union actor
                         |
[3] local hash aggregation
    同 bucket UID 聚合；高频 bucket 达 threshold 后压缩 union
                         |
[4] edge redistribution
    按 UID owner 重新路由 parent/edge
                         |
[5] balanced union-find
    actor 间迭代传播 root，直到本轮无变化
                         |
[6] final filter
    以最小 UID 为 root；parent 集中的非 root 被过滤
```

在 Data-Juicer v1.0.3 的公开实现快照中，默认 MinHash 为 256 维；bucket 达到
`union_threshold=256` 后会立即 union，并只留代表元，避免热门桶无限增长。BTS 的 owner/color 为
`floor(uid / 1000) mod P`：同 color 先选 local root，再让 local roots 指向全局最小 root。
这些是该版本的实现参数，不是 MinHash/BTS 的普适常数。

代码还使用 `ray.wait` 控制 pending remote calls，避免 producer 无界地产生 object refs；用 PyArrow
batch/zero-copy 路径减少 Python 行级开销。当前主分支还可能包含 GPU MinHash、C++ operator 与显式
memory reservation 等后续能力，**不能倒推它们都属于论文 `3.3×` 的实验快照**。

### 4.3 BTS 的本质优化：移动状态，而不是搬完整大簇

BTS 原论文
[Balanced Tree-based Strategy](https://kmudmlab.github.io/assets/papers/ICDE24_cekim.pdf)
针对分布式 connected components/union-find：把顶点和边按规则分配，让树结构与工作量更均衡，
优先在本地做 union，再交换必要的根/边状态。其目标是降低：

1. **重复通信**：同一 bucket、同一 component 的冗余边先本地压缩；
2. **碎片化 union**：不让每条小边都触发一次跨 actor 状态更新；
3. **根节点倾斜**：根据 UID/owner 把图状态分散到 actor；
4. **driver 压力**：收敛发生在 actor 间，而非把所有 pair 拉回 driver。

Data-Juicer 再叠加 hash aggregation：LSH bucket 内先聚合、去掉重复边/重复 union 请求，再进入 BTS。
这比“Ray 原生 groupby 后对每组逐对 union”的 baseline 更贴合该图问题。

这里的 baseline 要说准确。公开历史提交
[`31338d1`](https://github.com/datajuicer/data-juicer/commit/31338d147dd477b0fb1980ff899c04034d8d0e46)
中，Data-Juicer 自己的 vanilla Ray 版大致是：

```text
map_batches(compute_stats)
-> map_batches(expand each doc into num_bands rows)
-> groupby(band_hash).map_groups(...)
-> randomly choose a Union-Find actor per group and synchronously union
-> tree-merge all actor states
-> collect duplicate nodes to driver
-> filter
```

它不是“Ray 官方提供了一个 MinHash deduplicator”。主要开销来自全局 groupby shuffle、band expansion、
细粒度同步 RPC、actor 最终树式合并与 hot bucket 倾斜。后续公开提交展示了优化演化：

- [`1395072`](https://github.com/datajuicer/data-juicer/commit/139507204bb94a35e23424c73ede64595f098b54)：引入 BTS，但仍保留 groupby；
- [`fea44eb`](https://github.com/datajuicer/data-juicer/commit/fea44eb1fc9406cb36ff8cfb20baa37023084a95)：阈值式 partial union 与 actor spread；
- [`7fb0c59`](https://github.com/datajuicer/data-juicer/commit/7fb0c59f2a5378c024266736d2b94f4237259108)：去掉 groupby，band pair 直接 hash-route 到 actor；
- [`73d3f83`](https://github.com/datajuicer/data-juicer/commit/73d3f8325e8271fdeea971313399f5bffc851f95)：hash table 合入 BTS actor，热桶随到随压；
- [`d880b0d`](https://github.com/datajuicer/data-juicer/commit/d880b0df03d643ff95a986159891b33274c2ac7d)：UID 分配与 MinHash 计算合并为一 pass。

这些提交能建立机制因果链，却仍不能给 BTS 与 remove-groupby 各自分配一个独立 speedup；论文没有报告
完整的逐项 ablation。

### 4.4 `3.3×` 应该怎样正确解读

**[论文原文]** Data-Juicer 2.0 报告：其 MinHash 去重通过 load-balanced union-find 与 hash
aggregation，相比作者使用的 vanilla Ray 路径实现最高约 `3.3×` 加速。附录还解释了 fuzzy dedup
跨越 `map/filter/groupby/aggregate/join`，BTS 避免 native groupby 形成的 fragmented unions。

**它证明了**：

- 在论文实验条件里，专门的图算法与通信压缩显著优于其 vanilla Ray baseline；
- 性能瓶颈不只在 MinHash 计算，也在候选边聚合和 component 收敛；
- actor 可承载长期 union-find 状态，Ray Data 可承载列式批处理，两层组合是合理的。

**它没有证明**：

- 任意数据分布上稳定 `3.3×`；
- 比 Spark/GraphX、当前 Ray Data Shuffle v2 或任意第三方实现快 `3.3×`；
- 只改一个开关即可复现；
- 精度、候选召回与 canonical 规则完全不变。

**[未披露]** 公开论文没有给出足够完整的独立复现合同：至少缺少逐阶段 profile、所有 Ray/代码版本、
完整数据分布、LSH 参数、cluster 拓扑、baseline 的具体实现与置信区间。因此课程应把 `3.3×` 写成
“论文报告的 workload-specific speedup”，不能写成库的固有常数。

Data-Juicer 的[分布式文档](https://github.com/datajuicer/data-juicer/blob/main/docs/Distributed_ZH.md)
另给出 TB 级工程数据：MinHash 在 8 节点、每节点 160 cores 条件下约 3 小时量级；表中 1 TB 从
4 节点约 50.83 分钟降到 8 节点约 30.08 分钟，5 TB 从约 285.43 分钟降到约 168.10 分钟。
这组数能说明扩展性，但与 `3.3×` baseline ablation 是不同实验，不能混算。
当前英文分布式文档使用更保守的“约 `2–3×`”表述；这可能来自版本/措辞漂移，也不能自行与论文
精确 `3.3×` 等同。

### 4.5 怎样做可信复现

至少冻结以下 manifest：

```yaml
code:
  data_juicer_commit: ...
  ray_version: ...
algorithm:
  tokenizer: ...
  ngram: 5
  num_minhash: 256
  bands: ...
  rows_per_band: ...
  union_threshold: ...
execution:
  nodes: ...
  cpu_per_node: ...
  object_store_bytes: ...
  block_size: ...
data:
  documents: ...
  bytes: ...
  length_p50_p99: ...
  bucket_size_p50_p99_max: ...
semantics:
  candidate_pair_digest: ...
  cluster_membership_digest: ...
  keeper_policy: ...
```

对 vanilla、hash aggregation、BTS、二者同时打开做 $2\times2$ 消融，报告：signature time、shuffle、
candidate edges、union rounds、network bytes、spill、peak object-store、总时长以及输出 digest。
只有“更快且语义账本相同”才是有效优化。

---

## 5. `map` 与 `map_batches`：不是语法糖，而是摊销边界

Ray Data 官方分别提供
[`Dataset.map`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html) 与
[`Dataset.map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)。

| 维度 | `map` | `map_batches` |
|---|---|---|
| 用户函数输入 | 单行/单样本 | Arrow/Pandas/NumPy 等一批样本 |
| Python 调用次数 | 约 $N$ | 约 $N/B$ |
| 向量化 | 很弱 | 可用 SIMD、矩阵运算、tokenizer/model batching |
| 模型调用 | 单样本延迟高 | 一次 forward 吃 batch，吞吐通常高 |
| 内存峰值 | 低 | 输入 batch + 中间 tensor + 输出共同驻留 |
| 坏样本影响 | 通常只影响一行 | 可能让整批失败/重试 |
| 尾延迟 | 小粒度 | straggler 或动态 padding 会放大 |
| 调试/归因 | 直接 | 需保留 batch 内 sample ID 与逐样本错误 |

可用一个简单账本理解收益。若每次 Python/调度/序列化固定开销为 $t_o$，每样本实际计算为 $t_c$：

$$
T_{map}\approx N(t_o+t_c)
$$

batch size 为 $B$ 且计算可向量化到加速因子 $v(B)$ 时：

$$
T_{batch}\approx \frac{N}{B}t_o+\frac{N}{v(B)}t_c
$$

但 $B$ 不是越大越好。模型输入长度差异会造成 padding，峰值显存近似受
`B × max_length_in_batch` 控制；媒体解码还可能出现一条 4K 视频拖累整批。

生产建议：

1. 按长度、分辨率、时长先 bucketing，再 batch；
2. CPU 轻算子从小批量开始，GPU 模型用吞吐/显存曲线找拐点；
3. 为 batch UDF 实现逐样本 error envelope，避免一条坏数据毒死整批；
4. zero-copy batch 不能原地修改只读 buffer；需要改列时显式复制目标列；
5. block boundary/batch boundary 不是语义 group，不可依赖“同组刚好在同批”；
6. callable class + actor pool 用于复用模型；普通函数更适合无状态 task；
7. GPU `map_batches` 明确设置整数 `batch_size`，并记录 OOM retry，不能只看成功吞吐。

Data-Juicer 2.0 附录曾报告其 batched processing 相对旧 single-sample 路径最高缩短约 84% 时间，
batch size 到 100 后趋于平台，并推荐过 1000 的工程默认值。这个实验同时跨框架版本，不能改写成
“Ray `map_batches` 对 `map` 必然快 84%”，也不能把 1000 当成图像、长文本与 GPU 模型的共同最优值。

---

## 6. PB 级增量去重：维护版本化相似性图

### 6.1 典型场景

- Common Crawl 每月新增 snapshot，要与数年历史比较；
- 多供应商、搜索抓取与用户上传持续出现同一资产；
- SFT/RL rollout 与 agent trajectory 每天追加；
- 新闻、商品、代码仓库持续更新，旧版本仍需 lineage；
- 隐私删除、许可证撤回、eval decontamination 需要反向定位所有派生物；
- normalizer、embedding 或阈值升级，需要 shadow/backfill，而非静默覆盖。

第一性原理上先冻结四个定义：

1. **unit**：文档、段落、turn、trajectory、图片、镜头还是固定 token span？
2. **equivalence**：字节相同、归一化相同、近词法、语义相似，还是 benchmark 泄漏？
3. **representative policy**：同簇保留谁，按许可证、质量、来源还是时间？
4. **scope**：批内、跨批、跨来源、跨语言、跨模态还是只在某 manifest 内？

FineWeb 的经验提醒我们：去重同时在做采样和重加权。其最初按时间全局 MinHash 会大幅删除旧 crawl，
但保留下来的部分质量结构并不理想，最后采用逐 snapshot 去重，见
[FineWeb](https://arxiv.org/abs/2406.17557)。算法“检测正确”不代表保留策略“分布正确”。

### 6.2 不要混淆三个 ID

```text
Occurrence                Signature                  Cluster/Decision
---------------------     ----------------------     -------------------------
occurrence_id             occurrence_id              cluster_id
source/URI/offset         signature_version          canonical_occurrence_id
ingest_batch_id           normalizer/tokenizer       keep/drop/quarantine
raw_blob_pointer          shingle/hash seeds         reason/evidence
raw/normalized hash       minhash/band hashes        valid_from/to
license/provenance        embedding/version          snapshot_id
active/tombstone
```

- `occurrence_id` 表示“这次来源中的这条记录”，相同内容也不能合并身份，否则许可证与删除语义丢失；
- `content_hash` 表示内容相同；
- `cluster_id` 表示相似性图中的簇，不能直接等于 canonical，因为 canonical 会被删或换优；
- 所有 signature namespace 必须包含算法/模型/tokenizer/seeds/version。

### 6.3 分层索引

```text
new batch
   |
   +--> immutable lake snapshots
   |    occurrences / signatures / verified_edges /
   |    cluster_aliases / decisions / tombstones / manifests
   |
   +--> serving indexes
        exact KV + Bloom negative cache
        LSH inverted postings
        ANN vector index
        cluster root/canonical cache
```

Bloom positive 不能直接判重：false positive 会误删唯一数据；它只适合跳过“肯定不存在”的查询。
湖表是事实源与重建源，KV/ANN 是可再生 serving state。

### 6.4 每批的正确协议

设历史稳定快照为 $S_t$，新批为 $\Delta_t$：

1. **冻结快照**：所有 worker 查询同一个 `index_snapshot=t`；
2. **摄入幂等**：按 `(source, source_record_id, source_version)` 去消息重放；
3. **批内 exact/normalized-exact**：对 hot hash 只聚合 count、best candidate 与分页 refs；
4. **new↔old**：新 signature 查询历史 exact/LSH/ANN；
5. **new↔new**：同批按 band/embedding partition 建候选；逐条“查完即写”会导致顺序依赖；
6. **精确验证**：LSH/ANN 只负责召回，之后计算真实 Jaccard、edit、containment 或跨模态判定；
7. **写 pair edge**：规范化为 `(min_id,max_id,method_version)` 并先去重；
8. **合并簇**：允许一个新节点桥接两个历史簇；
9. **确定 canonical**：许可证资格 > 来源优先级 > 质量 > 完整性 > 时间 > stable ID tie-break；
10. **原子发布**：occurrence、signature、edge、alias、decision、manifest 一次 snapshot commit；
11. **幂等重跑**：同一 `batch_id` 重跑必须得到逻辑等价 digest。

Data-Juicer 的
[`document_minhash_deduplicator_with_uid`](https://github.com/datajuicer/data-juicer/blob/main/docs/operators/deduplicator/document_minhash_deduplicator_with_uid.md)
允许外部稳定 UID：给历史 A 较小 UID、新批 B 较大 UID，就能保持“旧数据优先保留”，并减少中间 I/O。
但它仍是**有稳定优先级的 A+B 联合去重**，并不等于已经维护一个永久在线的 LSH posting service；
真正的增量系统还需上面的持久索引、版本和 commit 协议。

### 6.5 最难的不是查重，而是图的非传递与删除

#### 相似不传递

可能出现：

$$
sim(A,B)\ge\tau,\quad sim(B,C)\ge\tau,\quad sim(A,C)<\tau
$$

connected components 会把三者全部合并，造成 chaining over-merge。高风险语料可要求新成员也与
canonical/medoid 过阈值，或对大簇二次聚类；最重要的是保留 pair edge，使 cluster 成为可重算 view。

#### union-find 会合并，不会自然分裂

若 B 是 A 与 C 的唯一桥，删除 B 后 active graph 应分裂；普通 DSU 不会。需要：

- tombstone 与 active/historical 两套视图；
- 保存 verified edges；
- 删除 bridge 时对受影响 component 局部重建；
- canonical 被删时重新选择，而不是丢掉 cluster identity。

#### 只索引 canonical 会漏召回

新样本可能只与某个非 canonical 成员相似。可选策略是索引全部 active members、每簇多个 diverse
representatives，或 cluster-level sketch；三者在状态量、召回和维护复杂度间取舍。

### 6.6 热点、失败与算法升级

| 失败模式 | 后果 | 防护 |
|---|---|---|
| 空文档/模板形成 hot exact key | 单 reducer/actor OOM | map-side combine、heavy-key side path、分页 refs |
| LSH bucket 大小为 $m$ | 朴素 pair 为 $O(m^2)$ | 最短文档门槛、停用高频 shingle、canopy、quarantine |
| candidate cap 静默截断 | recall 不可知地下降 | 记录 truncation rate；大桶走二级算法 |
| 先写 postings、decision 失败 | retry 后自撞/重复删除 | staging + checksum + atomic publish |
| first-seen 当 canonical | 并发/批次顺序改变结果 | 确定性多字段 policy |
| 不同 MinHash seeds 混用 | 相似度空间失真 | versioned namespace + dual-write/backfill/cutover |
| watermark 当永久语料边界 | 多年后重复漏检 | hot recent index + cold full-history index |

### 6.7 监控：不能只看“删了多少”

质量：candidate recall、verifier precision/recall、pair F1、cluster purity/B-cubed、false removal、
duplicate miss、canonical regret、各语言/来源/长度/时间 retention delta、eval contamination recall。

效率：docs/s、TB/hour、CPU-hours/TB、shuffle/input ratio、spill bytes、candidate/doc P50/P99、
bucket size P99/max、union rounds、index bytes/doc、compaction amplification、retry、commit latency。

正确性：orphan postings、tombstone 仍可 lookup、canonical 指向 inactive occurrence、alias chain/cycle、
signature version coverage、delete propagation SLA、partial snapshot visibility、rerun digest。

---

## 7. 多模态去重：先问“什么算同一个东西”

文本通常有较稳定的 token/shingle。媒体则同时存在编解码、裁剪、重采样、字幕、水印、镜头拼接、
语义近似与跨模态配对，所以至少分五层：

```text
L0 byte identity        同一文件
L1 decoded identity     容器/编码不同，像素或 PCM 相同
L2 transformed reuse    裁剪、缩放、转码、加水印、变速、局部片段
L3 semantic similarity  内容/事件相近但不是同一资产
L4 pair/sequence reuse  资产相同但 caption/上下文不同，或组合序列重复
```

L3 不应默认“删除”：两张不同角度的同一手术图像可能是有价值的多样性；两个语义相同、出处不同的
新闻视频则可能是重复采样。去重策略必须绑定训练目标。

NVIDIA NeMo Curator 的[去重概览](https://docs.nvidia.com/nemo/curator/curate-text/process-data/deduplication)
也明确区分 exact、fuzzy 与 semantic dedup，并为 text、image、video 提供不同工作流；这支持“分层，
而非一个统一 hash”的设计。

### 7.1 分模态矩阵

| 模态/单位 | exact / decoded | 近重复候选 | 最终验证与特殊风险 |
|---|---|---|---|
| 图像 | file hash；规范化 RGB pixel hash | pHash/PDQ 一类感知 hash；DINO/CLIP/SigLIP embedding ANN | crop/overlay 对全图 hash 不稳；用局部特征/区域重叠；截图含文字另做 OCR MinHash |
| 视频 | container hash；规范化帧流+音轨 hash | keyframe/shot perceptual signatures；video embedding；音频 fingerprint | 必须检出子片段、拼接、变速、重编码；需 temporal alignment，不能只平均帧 embedding |
| 音频/语音 | file hash；规范化 PCM hash | acoustic fingerprint；CLAP/语音 embedding ANN | 变速、变调、噪声、截取；同转写不等于同录音，privacy/说话人风险独立 |
| PDF/扫描文档 | file hash；解析文本 hash | page image pHash；OCR text MinHash；layout embedding | 同内容不同排版/扫描；页重复、模板页、阅读顺序与表格结构 |
| 图文 pair | `(image_asset_id,text_hash)` | image cluster × text similarity；cross-modal embedding | 同图不同 caption 可能是互补监督，也可能是模板污染；不可只按 image 删除整簇 |
| 音视频 pair | asset/track IDs | AV embedding、ASR text、shot+audio fingerprints | 配音/字幕版本、A/V 不同步；只删视频可能留下重复音轨 |
| interleaved 文档/轨迹 | ordered asset/text ID sequence hash | window MinHash/sequence alignment | 相同素材不同教学顺序可能有价值；需区分 asset leakage 与 sequence duplication |

### 7.2 图像：三个 index 通常都要

```text
raw_hash index      -> 低成本、零容忍 exact
perceptual index    -> resize/recompress/light color transform
semantic ANN        -> candidate discovery / diversity analysis
```

对 crop、拼图、meme overlay，应增加局部 region/patch 特征或 OCR 文本路径。语义 ANN 的高相似只说明
“值得检查”，不自动等价：同一商品不同角度、连续医学切片、同一人物不同时刻都可能是目标多样性。

### 7.3 视频：把视频先拆成时间结构

可靠表示应包含：

```text
video_id
  -> shots
      -> keyframes/perceptual hashes
      -> visual embeddings
      -> ASR/OCR/audio fingerprint
      -> [start,end] timestamps
```

候选后再做局部时间对齐，判断 overlap duration、coverage 与 speed ratio。平均全片 embedding 对“十分钟视频
中重复三十秒”很不敏感，也会把同一主题的不同视频误判为重复。去重决定宜输出 pair evidence：

```text
same_full_asset | contains_clip | shared_intro | semantic_only | uncertain
```

共享片头/片尾通常应做 span removal 或降权，而不是删除整条视频。

### 7.4 音频：指纹、转写与语义是三个不同问题

- PCM hash：抓规范化后完全相同音频；
- acoustic fingerprint：抓转码、轻噪声、局部片段；
- ASR text MinHash：抓“说了相同的话”，但不能证明录音相同；
- CLAP/语音 embedding：用于语义/声学候选和多样性，不应直接删除；
- speaker embedding：用于说话人分布与隐私审计，不宜混作内容去重。

同一段话由不同说话人朗读，对 TTS/ASR 可能是宝贵多样性；对知识预训练则可能是不必要重复。

### 7.5 PDF/文档：多视图去重

同一论文的 arXiv PDF、出版社排版、扫描件与网页 HTML，file hash 全不同。建议同时维护：

1. 规范化抽取文本的 exact/MinHash；
2. page image 的感知 hash；
3. 标题、作者、DOI 等 metadata blocking；
4. layout/table/formula signature；
5. 页级边和 document-level aggregation。

这能区分“同一文献不同载体”“同一模板不同内容”“同一 PDF 内重复页”。

### 7.6 split 必须在 component 之后

若先随机拆 train/val/test，再各自去重，同一资产的转码版、裁剪版或同源 pair 会跨 split 泄漏。
正确顺序是：

```text
build evidence graph -> connected/constrained components -> group-aware split
```

split key 应至少能绑定 asset family、来源事件/文档与派生链，而不只是最终行 ID。

### 7.7 公开前沿实践：方法解决的是哪一层

| 实践 | 公开做法 | 应该学什么；不应外推什么 |
|---|---|---|
| [DINOv2/LVD-142M](https://arxiv.org/abs/2304.07193) | PCA hash/安全与人脸处理后，以 SSCD 去副本；用 ViT embedding、k-means、FAISS IVF-PQ 做数据检索 | 多级 exact/copy/semantic pipeline；其检索比例与阈值不是通用默认值 |
| [SSCD](https://arxiv.org/abs/2202.10261) | 学习 compact descriptor，针对裁剪、编辑、重压缩后的图像副本 | 它是 copy detector，不是通用 semantic dedup |
| [SemDeDup](https://arxiv.org/abs/2303.09540) | embedding 聚类后簇内按 cosine 裁剪；LAION 实验可大量减量 | 语义裁剪能省算力，但不证明被删样本是同一作品 |
| [FairDeDup](https://arxiv.org/abs/2404.16123) | 显示语义去重会改变群体公平性，并加入 fairness-aware 选择 | threshold 与 canonical policy 必须按人群/语言/域审计 |
| [MLT-Dedup](https://arxiv.org/html/2606.12215v1) | 稀疏 clip embedding/HNSW 召回，细 frame embedding 按需加载，再定位复制时间段 | 视频应粗召回+细时序定位；论文阈值只属于其平台和模型 |
| [Whisper](https://arxiv.org/abs/2212.04356) | 音频/文本 LID、机器 transcript 过滤、fuzzy transcript dedup、30 秒切片、初模反查坏源 | transcript 去重与 acoustic 去重必须分开 |
| [MINT-1T](https://arxiv.org/abs/2406.11271) | PDF 文本/布局抽取、paragraph Bloom dedup、图片 SHA、source/snapshot 内处理 | 披露清楚但会漏跨 source/snapshot 重复，适合作反例审计 |
| [mmc4](https://arxiv.org/abs/2304.06939) | pHash、图像硬门、页内 sentence×image CLIP、linear assignment | interleaved 文档不能被扁平化为独立 pair top-1 |
| [OBELICS](https://github.com/huggingface/OBELICS) | URL/image-set/document/paragraph 多层去重，NSFW 与 opt-out 处理 | asset、document、paragraph 和退出权需要不同账本 |

一个重要实践边界：SemDeDup 一类 semantic pruning 更适合**限频、降权或多样性采样**；SSCD、
acoustic fingerprint、时序局部匹配等 near-copy 证据达到高 precision operating point 后，才更适合
硬删除。两者都叫 dedup，却优化不同目标。

---

## 8. 多模态清洗与打标：SOTA 是级联，不是一只万能模型

公开工具会快速变化，所以本节把具体模型作为 **operator exemplar**，不声称某一个在所有域都是永久 SOTA。
真正稳定的是操作维度、验证合同和成本分层。

### 8.1 共用的六层漏斗

| 层 | 解决什么 | 典型输出 |
|---|---|---|
| 0 格式/来源 | 能否读、是否有权用 | decode_ok、license、provenance、PII flags |
| 1 信号质量 | 媒体本身是否可学 | 分辨率、SNR、blur、clipping、fps、时长、OCR confidence |
| 2 内容安全 | 是否应进入某训练域 | NSFW、暴力、医疗敏感、人脸、儿童、版权/水印 |
| 3 语义标签 | 里面有什么 | language、objects、scene、action、topic、speaker、shot、document type |
| 4 跨模态关系 | pair 是否对齐 | image-text/AV alignment、caption specificity、contradiction、grounding |
| 5 价值/多样性 | 是否值得占 token budget | aesthetic、information density、rarity、cluster weight、uncertainty |

Data-Juicer 2.0 报告覆盖 text/image/audio/video 以及跨模态 operators，包括 motion score、phrase
grounding recall、视频摘要、NSFW/face blur 等；这说明生产系统应把“可组合 operator + stats cache”
作为基本单元，而非把所有判断揉进一次大模型调用。

### 8.2 图像

建议字段：

```text
decode / width / height / aspect / entropy / blur / exposure
watermark / OCR area / face count / NSFW / violence
CLIP-or-SigLIP alignment / aesthetic / object-scene tags
grounding coverage / caption specificity / duplicate cluster
```

实践上：

- 规则处理损坏、极小图、极端宽高比；
- 感知模型处理 blur/aesthetic/watermark/safety；
- CLIP/SigLIP 一类双塔模型适合大规模图文 alignment 与 ANN；
- DINOv2 一类视觉自监督 embedding 更偏视觉相似与聚类；
- VLM 适合生成细粒度 caption、解释矛盾与 difficult-case 审核，但成本高且会 hallucinate；
- grounding/detector 用于验证 caption 中实体是否有区域证据。

可用的开放 operator exemplars 还包括：

- [Q-Align/OneAlign](https://arxiv.org/abs/2312.17090)：IQA/美学/视频质量的人类主观分数代理；
- [RAM++](https://arxiv.org/abs/2310.15200)：开放词汇图像标签；
- [SigLIP2](https://arxiv.org/abs/2502.14786)：多语 image-text embedding、定位与在线数据治理；
- [ShieldGemma 2](https://ai.google.dev/gemma/docs/shieldgemma/model_card_2)：图像安全分类。

它们各自只覆盖一个轴。安全模型的语言/policy prompt 会影响结论；美学模型会把文化偏好编码成“质量”。
因此保留连续原始 score、模型 revision 与域内 calibration，避免一次性全局 top-x 删除。

[DataComp](https://arxiv.org/abs/2304.14108)的核心贡献之一正是固定模型训练协议、比较数据筛选策略，
提醒我们不能只凭过滤模型自身分数认定“数据更好”；最终仍需固定训练预算做 downstream ablation。

### 8.3 视频

建议字段：

```text
decode_ok / duration / fps / resolution / codec / black-frame ratio
shot boundaries / motion score / optical-flow statistics / freeze ratio
audio present / AV sync / ASR / OCR / language
NSFW/violence / faces / actions / scene / temporal caption
caption-event grounding / repeated intro-outro / duplicate spans
```

关键 trade-off：均匀抽 8 帧便宜，却会错过短事件；全帧推理召回高，但成本近似随时长线性增长。
常用折中是 shot-aware sampling：低成本检测镜头/运动，静态镜头少采、快速动作多采；在候选片段上再跑
video encoder/VLM。时序标签必须保存 `[start,end]`，否则无法支持片段去重、局部过滤和训练采样。

[Stable Video Diffusion 的数据报告](https://arxiv.org/abs/2311.15127)披露了 scene cut、2 fps
光流、首/中/末帧 CLIP/美学与 OCR 过滤；[Panda-70M](https://arxiv.org/abs/2402.19479)
将长视频切成语义一致 clips，并融合视频描述、字幕与帧 caption；
[VideoPrism](https://arxiv.org/abs/2402.13217)可作为通用 video embedding/tagging exemplar。
这些报告里的 bottom 25%/50%、OCR 面积阈值等都是小规模消融选出的 operating point，不能跨数据集复制。

### 8.4 音频/语音

建议字段：

```text
sample_rate / channels / duration / clipping / silence / SNR
VAD segments / speech-music-noise / language / ASR confidence
speaker count / diarization / overlap / emotion / acoustic events
PII in transcript / profanity / alignment / duplicate spans
```

Whisper 类 ASR、VAD、speaker diarization、CLAP 类 audio-text embedding分别解决转写、语音边界、
说话人结构与跨模态语义，不能用一个分数互相替代。高 ASR confidence 也不等于音质好：模型可能
在模板化语音上很自信；低 confidence 可能来自口音，而非无效数据。应按语言/口音/领域分桶校准阈值。

[CLAP](https://arxiv.org/abs/2206.04769)可作 audio-text alignment/semantic embedding；
[Panako](https://github.com/JorenSix/Panako)一类声学指纹更适合抗压缩、噪声、轻微变速/变调的
near-copy；[DNSMOS P.835](https://arxiv.org/abs/2110.01763)给出语音感知质量代理。
感知质量仍不等于 ASR 可学性；同一 transcript 的不同说话人也不能默认去掉。

### 8.5 PDF、网页截图与富文档

除文本质量外，还要记录：

```text
parse success / OCR confidence / reading-order confidence
page type / text-image-table-formula ratio / repeated header-footer
table structure / equation extraction / figure-caption links
scanned-vs-digital / language / document genre / source lineage
```

MinerU、Docling、Marker 等解析器可以作为候选实现，但“谁是 SOTA”强依赖扫描件、公式、表格和语言。
正确评估单位不是 parser 自报成功率，而是页/区块级标注：reading order、表格 cell、公式、caption link，
再加下游 RAG/预训练消融。

公开生产锚点可看 [olmOCR](https://github.com/allenai/olmocr) 与
[Dolma3 PDF recipe](https://github.com/allenai/dolma3/blob/main/datasets/dolma3_mix/pools/9T/README.md)：
其 recipe 披露了 PDF SHA、解析失败容忍、Poppler fallback、5-gram MinHash、PII/质量分类与 denylist。
注意 OCR 会把原本藏在图片中的姓名、证件号变成可检索文本，解析后必须重新跑 PII，而不是沿用原文件
入库时的文本扫描结果。

### 8.6 Pair 与 interleaved 数据：标签必须描述关系

单模态都合格，不代表 pair 合格。图文/音视频数据至少打：

- relevance：是否同主题；
- entailment/contradiction：文本是否被媒体支持；
- specificity：是泛化描述还是精确到实体/动作；
- coverage：caption 覆盖多少重要区域/时间事件；
- hallucination：是否提及不存在内容；
- temporal alignment：声音、字幕、事件是否同步；
- provenance consistency：pair 是否原生共现还是后合成；
- teaching value：问答是否需要看媒体才能回答。

只用 CLIP score 会偏爱表面名词重叠和显著物体，可能淘汰长尾、文字密集图、图表与复杂关系；只用 VLM
judge 则成本高、版本漂移，并可能把自己的先验当作媒体证据。建议：双塔召回/粗筛 → detector/OCR/ASR
结构证据 → VLM difficult-case → 人工分层校准。

### 8.7 阈值与预算的 trade-off

| 选择 | 收益 | 代价/偏差 |
|---|---|---|
| 更严格质量阈值 | 平均质量提高 | 数据量、长尾、多语/低资源分布下降 |
| 更强语义模型 | 标签细、召回高 | GPU 成本、版本漂移、不可解释误判 |
| 更激进去重 | 降低记忆化与重复算力 | 合法变体/多视角/少数群体被误删 |
| 更密视频采样 | 短事件召回高 | 成本、存储与相邻帧冗余上升 |
| 自动 caption 重写 | 文本更流畅 | 抹掉原始 provenance；引入 teacher 偏见/幻觉 |
| cluster 均匀采样 | 提升多样性 | 稀有但高价值大簇可能被过度降权 |

阈值不是模型常数，而是预算与目标函数的控制旋钮。每次变更都要输出 retention curve：

```text
threshold
 -> retained bytes/tokens/hours
 -> slice distribution
 -> estimated label precision/recall
 -> training loss and downstream delta at fixed compute
```

---

## 9. 推荐的统一生产架构

```text
                   immutable raw objects
                            |
                    ingest manifest
                            |
          +-----------------+------------------+
          |                                    |
   cheap structural ops                      metadata/provenance
 decode/hash/stats/VAD/shot/OCR              license/consent/lineage
          |                                    |
          +-----------------+------------------+
                            |
              batch model enrichment (Ray)
     embedding/caption/ASR/safety/quality/grounding
                            |
          feature lake: versioned, reusable, no overwrite
                            |
          +-----------------+------------------+
          |                                    |
 similarity graph                     policy/quality filters
 exact/LSH/ANN/temporal               rules + calibrated models
          |                                    |
          +-----------------+------------------+
                            |
          canonical/weight/split decisions
                            |
              immutable training manifest
                            |
        fixed-budget proxy train + eval + audit
                            |
              PromotionGate / rollback
```

### 9.1 Spark 与 Ray 的合理分工

- Spark：超大表 join/group-by、离线 feature 汇总、历史 snapshot/backfill、稳定 spill；
- Ray Data：CPU/GPU 混合模型推理、`map_batches`、actor 模型复用、动态多模态 pipeline；
- 专用 actor/服务：BTS union-find、ANN/KV serving index 等有长期状态的算法；
- 湖表：唯一事实源、原子 snapshot 与 time travel；
- 不让 actor 内存成为唯一真相，也不让 Spark/Ray 的临时 block ID 成为数据身份。

跨引擎边界必须 materialize 为版本化表与 manifest；不要在失败恢复时依赖两个 runtime 的隐式缓存状态。

### 9.2 发布门

一版新 operator/模型/阈值只有同时满足以下条件才 promote：

1. 输入、代码、模型、参数与输出 digest 可复现；
2. pair/cluster/label gold set 上达到预先冻结的 precision/recall；
3. retention slice 没有未解释的语言、来源、人口或时间偏斜；
4. 固定 compute proxy training 优于或不劣于 baseline，并给出不确定性；
5. 成本、吞吐、spill、OOM/retry 在预算内；
6. 支持 tombstone、rollback 与历史 manifest 重放；
7. shadow 运行一个完整 ingest 周期，没有 partial visibility 或 lineage 断裂。

---

## 10. 费曼检验：能否不用术语说清楚

### Q1：为什么 Spark group-by 常比 plain Ray task 省心？

因为 group-by 不只是“把相同 key 放一起”，还要局部压缩、跨机器搬运、内存不足写盘、机器坏了重算、
最后只发布一次。Spark 把整套流程做成默认协议；plain Ray 给你的是可以搭协议的积木。Ray Data 则已经
搭了相当一部分，所以不能把它与 Ray Core 混为一谈。

### Q2：Data-Juicer 的 `3.3×` 是不是 MinHash 算得快 3.3 倍？

不是。主要优化发生在 MinHash 之后：如何聚合同桶候选、压缩重复 union、分散图状态并让 connected
components 收敛。公开论文不足以把 `3.3×` 分解到每个优化，也不能外推到所有版本和数据。

### Q3：为什么 LSH collision 不能直接删除？

LSH 故意让相似对象“有较高概率”碰撞，换取少比较很多 pair。概率候选不是确定判定；还需算真实
Jaccard/edit/containment，并保存证据。

### Q4：为什么增量去重不能只拿新数据查历史库？

因为同一新批内部也可能重复，而且一个新样本可能同时连接两个旧簇。只做 `new↔old` 会漏掉
`new↔new`，也无法正确合并历史组件。

### Q5：为什么删掉一个文档后 union-find 不够？

union-find 只擅长把集合并起来。如果被删文档是两个子图之间唯一的桥，集合应该拆开；普通 DSU 不会
自动拆分，所以必须保存边并局部重算。

### Q6：为什么图片 CLIP 相似度高不等于 duplicate？

它可能只说明语义相近。两张“医生看 X 光片”的图可以是完全不同病例；删除会损害多样性。感知 hash、
局部匹配和 provenance 才更接近资产复用证据。

### Q7：什么时候 `map_batches` 反而更差？

单样本极不均匀、batch padding 很重、内存不足、坏样本会让整批重试，或 UDF 根本无法向量化时。
batch 的收益来自摊销与向量化，不来自函数名字。

### Q8：多模态清洗为何不能只看过滤器离线分数？

过滤会同时改变数据量和分布。分类准确不代表训练收益；必须在固定模型、token/step/compute 和评测合同
下做 downstream ablation，并检查长尾切片有没有被静默删光。

---

## 11. 一手来源与证据边界

- [Data-Juicer 2.0 论文](https://arxiv.org/abs/2501.14755)：DJDataset、Ray/MaxFrame 支持、BTS + hash aggregation 与 `3.3×` 声称。
- [Data-Juicer 分布式文档](https://github.com/datajuicer/data-juicer/blob/main/docs/Distributed_ZH.md)：RayDataset/RayExecutor 与 TB 级运行表。
- [Data-Juicer RayDataset 源码](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/ray_dataset.py)：Ray Data 接口、task/actor pool 与资源参数。
- [Data-Juicer BTS 源码](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/ops/deduplicator/ray_bts_minhash_deduplicator.py)：当前 actor、路由、union 与 backpressure 实现。
- [BTS ICDE 2024](https://kmudmlab.github.io/assets/papers/ICDE24_cekim.pdf)：balanced tree 分布式图算法；其 `3.1–261.9×` 对比与 Data-Juicer `3.3×` 不是同一实验。
- [Spark SQL 论文](https://people.eecs.berkeley.edu/~matei/papers/2015/sigmod_spark_sql.pdf)、[RDD guide](https://spark.apache.org/docs/latest/rdd-programming-guide)、[SQL tuning](https://spark.apache.org/docs/latest/sql-performance-tuning)：planner、shuffle、partial aggregate、AQE。
- [Ray Data internals](https://docs.ray.io/en/latest/data/data-internals.html)、[aggregation](https://docs.ray.io/en/latest/data/aggregating-data.html)、[`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)：当前数据引擎语义与边界。
- [FineWeb](https://arxiv.org/abs/2406.17557)、[Lee et al. 去重](https://aclanthology.org/2022.acl-long.577.pdf)：MinHash/LSH、exact substring、数据分布效应。
- [NeMo Curator dedup](https://docs.nvidia.com/nemo/curator/curate-text/process-data/deduplication)、[DataComp](https://arxiv.org/abs/2304.14108)：多模态 semantic dedup 与固定训练协议的数据筛选评估。

**最后的边界**：公开资料足以解释机制和设计复现实验，但不足以知道头部实验室全部私有数据规则，
也不足以宣布某个闭源模型在所有模态上“最 SOTA”。生产选择应以自己的 gold slice、固定预算训练和
可撤销 manifest 为准。
