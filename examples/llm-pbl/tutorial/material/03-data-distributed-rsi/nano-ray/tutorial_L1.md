# nano-ray · Tutorial L1 — 真实 Ray 执行 OP pipeline：语义不变，代价变成实数

> **K+1 定位**：L0 用 270 行标准库把 Ray 编程模型的**语义**搭对了（future / 传引用 /
> 完成即触发），但有三样东西是假的：worker 是线程（躲不开 GIL）、object store 是同进程
> 字典（没有共享内存）、调度与序列化没有真实成本。L1 只加一层：把同一套执行计划搬到
> **真实 Ray**（ray 2.56.1，真 worker 进程 + 真 object store）上，回答两个问题——
> 语义会变吗（不会）？代价是多少（全部变成实数）？
>
> **跨模块契约**：本文件与 `nano-data-juicer/L2_distributed_pipeline.py` 使用同一个
> 工作负载（seed=42 合成语料，3360 条）与同一个执行计划（分区 → 局部 OP 并行 → 全局
> OP 收敛）。三个执行器（串行 / L2 multiprocessing / L1 Ray）必须给出同一个漏斗：
> **3360 → 2358 → 2110**。执行语义住在「计划」里，不在 runtime 里——这就是 L1 的核心
> 教学点，也是真实系统里换执行引擎不改数据语义的底气来源。

---

## 1. 问题：L0 的三处「假」，L1 逐一换真

| L0（玩具） | L1（真实 Ray） | 换真之后多付的钱 |
|---|---|---|
| `ThreadPoolExecutor` 线程 worker | 独立 worker **进程**（各有自己的 pid） | 进程启动与 IPC |
| 内存字典 store，传引用 = 传 Python 对象 | object store（plasma），传引用 = ObjectRef 句柄 | 驱动端序列化 |
| submit 零成本 | submit 走 gRPC 提交任务 | 每任务提交开销 |
| 没有启动阶段 | `ray.init` 拉起 raylet + worker | 秒级一次性成本 |

L0 的实验结论（并行等价、依赖顺序、粒度反例）在 L1 全部要重新验证一遍——因为真实
runtime 的任何一处语义偏差，都会在「并行结果 == 串行结果」这类断言上现形。

## 2. 运行与输出

环境：`pip install ray`（本机实测 ray==2.56.1 / Python 3.13.13 / macOS arm64，
单机模式，4 CPU）。语料与 nano-data-juicer L2 完全同一构造，无需任何数据文件。

```bash
python L1_ray_pipeline.py
```

```
====================================================================
nano-ray L1 — 真实 Ray 执行 nano-data-juicer 的 OP pipeline
====================================================================
python 3.13.13 | ray 2.56.1
声明: 真实 Ray 单机模式（本地 raylet + worker 进程），无 mock；
      多节点行为见 [TODO: verify on real system]

[0] ray.init 启动成本: 2.7s (resources: CPU=4, object_store=2.0 GiB)
    语料: 3360 docs, 3.74 MB (含 360 条注入重复) —— 与 nano-data-juicer L2 同一构造

[1] object store 喂数据 (corpus 3.74 MB, 序列化负载 ~3.8 MB)
    ray.put 一次: 1.4 ms（驱动端序列化 1 份进 store）
    传引用提交 4 任务: 提交耗时 0.3 ms（只传 ~28B 的 ObjectRef，数据不再动）
    传值提交 4 任务: 提交耗时 7.1 ms（每次提交都把 ~3.8 MB 语料完整序列化一遍，
    随任务 RPC 内联发出——< 10MB 内联上限 task_rpc_inlined_bytes_limit，
    ray_config_def.h@ray-2.56.1；超限参数改走自动 put，见 L3）
    两种方式结果一致: [840, 840, 840, 840] ✅ | worker 进程数 = 4（跨进程执行 ✅）
    自动解引用现场 a: 任务内对自己的 int 参数调 ray.get ->
      ValueError: Invalid type of object refs, <class 'int'>, is given. 'object_refs' must either be an ObjectRef or a list of ObjectRefs.
    自动解引用现场 b: 对已解引用的 list 调 ray.get ->
      TypeError: Attempting to call `get` on the value 1, which is not an ray.ObjectRef.
    → ObjectRef 作为参数在任务执行前已被 runtime 解引用，任务体拿到
      的直接是数据；b 更危险——list 会被 ray.get 当成「ref 列表」逐个
      再解一次，若元素恰好是 ref 就发生静默二次解引用。想在任务里持有
      ref 本身需包一层容器（L3 讨论）。

[2] 局部 OP 阶段 (normalize_mapper + length_filter)
    漏斗: 3360 -> 2358 条 | serial 304.9 ms / ray 86.4 ms (speedup 3.53x)
    并行结果 == 串行结果: True ✅（顺序逐位一致）
    worker pid 证据: 4 个不同进程承载 4 个分区任务（L0 的线程共享同一进程，这里是真进程）

[3] 全局 OP (exact_deduplicator)
    a) naive 分区各做各的: 2346 条, 泄漏 236 条 ❌
    b) 收敛点任务 ray_global_dedup(*part_refs): 2110 条 ✅
    串行基准: 2110 条 | 逐位一致: True ✅
    重复对账本(过滤后幸存者): 共 248 对 | 跨分区 236 对 => 泄漏数 236 = 跨分区对数
    （全局去重串行段 0.2 ms：收敛点后的 Amdahl 串行段，与 L2 同结论）

[4] 任务粒度扫描 (同一语料，变分区数 P；4 CPU)
     P |  wall(ms) |   docs/s | 每任务条数
     1 |     312.6 |    10748 | 3360
     2 |     156.9 |    21413 | 1680
     4 |      82.0 |    40952 | 840
     8 |      83.7 |    40134 | 420
    16 |      86.5 |    38844 | 210
    最快: P=4；P 过大时每任务开销（提交/调度/序列化）抬头，
    P=1 没有并行——最优粒度在两者之间，且随负载而变。

[5] runtime 成本摊销: ray.init 2.7s vs 单轮 pipeline 工作 305 ms => 约 9 轮才回本
    小作业别付集群启动费；Ray 的价值在长生命周期集群 + 大负载
    （Data-Juicer 分区默认 size=5000 条/上限 64 MB，本语料 3.74 MB
    按该默认只会切出 1 个分区——默认参数面向的是 GB 级语料）。

====================================================================
✅ self-check passed:
   漏斗 3360->2358->2110 与 nano-data-juicer L2 完全一致 /
   并行==串行逐位一致 / naive 泄漏 236=跨分区对数 /
   收敛点==串行 pipeline 逐位一致 / 粒度扫描条数不变
====================================================================

takeaway: 同一个执行计划（分区→局部 OP 并行→全局 OP 收敛）换到真实
          Ray 上，语义一分不差，代价（启动/提交/序列化）变成实数。
          L2 把有状态算子搬进 actor。
```

三遍连跑（记录当日）：掩掉计时数字后输出逐字节一致；所有计数类输出（漏斗、泄漏账本、
worker 数）三遍完全相同；「最优 P」在同批三遍相同，跨批次独立复跑时可能在
噪声级差异下偏移一档（见 §4 [4] 与 §10 计时口径）。

## 3. 代码结构：同一计划，换个执行器

`L1_ray_pipeline.py` 分四段：

1. **语料与 OP（与 L2 同构）**。`make_corpus` / `normalize_map` / `length_keep` /
   `local_ops` / `dedup_keep_first` / `partition` 与 nano-data-juicer L2 是同一套实现，
   同 seed 生成同一份语料。这是「跨执行器语义不变」能被机器检验的前提：只要两边
   漏斗数字差一条，就说明计划或 OP 里藏着对 runtime 的隐含假设。
2. **五个 `@ray.remote` 函数**。`slice_count`（切片计数，[1] 用）、`ray_local_ops`
   （局部 OP 阶段，附 worker pid）、`ray_partition_dedup`（naive 反例）、
   `ray_global_dedup`（收敛点）、`bad_get`（自动解引用现场）。全部是顶层函数，
   可被 cloudpickle 送进 worker 进程。
3. **实验 [0]–[5]**，见 §4 逐段解读。
4. **硬编码期望值断言**：`EXPECTED_AFTER_FILTER=2358`、`EXPECTED_AFTER_DEDUP=2110`、
   `EXPECTED_NAIVE=2346`、`EXPECTED_LEAK=236`——这些数字来自已验证的 L2，
   在 L1 里充当**跨模块一致性契约**：Ray 跑出的漏斗与 multiprocessing 跑出的
   逐位一致，才许打勾。

收敛点的签名值得盯一眼：

```python
@ray.remote
def ray_global_dedup(*parts: List[Sample]) -> List[Sample]:
    merged = [s for p in parts for s in p]
    return dedup_keep_first(merged)

# 调用：
conv_out = ray.get(ray_global_dedup.remote(*mapped_refs))
```

`*mapped_refs` 传进去的是四个 ObjectRef，但任务体里 `parts` 已是解引用后的真实分区
数据——**ObjectRef 作为任务参数会在任务执行前被 runtime 自动解引用**。所以「收敛点」
在 Ray 里的物化形式极其朴素：*一个依赖所有分区的任务*。不需要 barrier API，不需要
shuffle 原语，依赖边本身就是同步语义（L0「完成即触发」的回声）。

## 4. 输出逐段解读

**[0] 启动成本。** `ray.init` 花 2.5–2.9s（多批测量区间，下同：区间覆盖初次多次运行与
独立复跑批次，见 §10）拉起本地 raylet、object store
（2.0 GiB 共享内存）与 worker 进程池。对照 [5]：单轮 pipeline 的串行工作量约 0.3s，
所以这套 runtime 要连跑约 9 轮才「回本」。这是所有分布式 runtime 的共性账单：
**固定启动费 ÷ 单轮工作量 = 盈亏平衡轮数**。

**[1] 三种喂数据方式的账。** 这是 L1 对 L0「object store = 字典」的最大修正。
同一个 3.74 MB 语料（序列化负载 ~3.8 MB）：

- `ray.put` 一次：驱动端序列化 1 份进 store，实测 1.4–2.0 ms；
- 传引用提交 4 任务：每次提交只带 ~28B 的 ObjectRef，4 次合计 0.3–0.4 ms；
- 传值提交 4 任务：**每次提交都把语料完整序列化一遍**随任务 RPC 发出，
  4 次合计 7.0–7.9 ms，约为单次序列化的 4 倍。

注意测量卫生：实验先用两个小任务预热 submit/put 通路，把「首次使用」的一次性初始化
成本挡在计时窗外——探针实测冷首提交 ~50 ms，预热后同尺寸提交降到几 ms。若不做预热，
第一个数字会把一次性成本错记到序列化头上（§5.1 细说）。
另一个探针结论值得记住：**驱动端没有按对象身份缓存序列化结果**——同一对象重复传值，
每次都付全量序列化的钱。所以「传值 N 次 = 序列化 N 次」不是修辞，是账本。

**[1b] 自动解引用的两种失败现场。** 任务体内对自己已被解引用的参数再调 `ray.get`：
参数是 int 时报 `ValueError: Invalid type of object refs...`；参数是 list 时报
`TypeError: Attempting to call 'get' on the value 1...`——后者暴露了更危险的语义：
`ray.get` 见到 list 就当成「ref 列表」逐个再解一次。你的数据若恰好是一个列表，
它会被静默地二次解释。

**[2] 局部 OP 并行。** 3360 → 2358 条，与串行逐位一致（`par_mapped == ser_mapped`，
顺序敏感）。speedup 三遍观测 3.37x–3.54x——与 L2 multiprocessing 的 2.4x–3.3x 同一
量级，同样达不到 4。worker pid 证据：4 个分区任务落在 4 个不同进程上（L0 的线程池
全在一个进程里，GIL 下 CPU 密集任务拿不到真并行；这里是真进程）。

**[3] 全局 OP 与泄漏账本。** naive 做法（各分区各做各的去重）得到 2346 条，泄漏 236
条；收敛点任务得到 2110 条，与串行 pipeline 逐位一致。账本闭合：过滤后幸存者中共
248 个重复对，其中跨分区的恰为 236 对——**泄漏数 = 跨分区重复对数**，一分不多一分
不少。这个算术与 L2 完全相同，因为它是「计划」的性质，不是 runtime 的性质。

**[4] 粒度扫描。** P=1 无并行（~300 ms）；P=2 减半；P=4 触底（~82 ms，≈4 倍加速）；
P=8/16 不再变快反而微升——每任务条数降到 420/210 时，提交、调度、数据搬移的固定
开销开始抬头。多批测量下最优为 **P=4（= CPU 数）附近 ± 一档**：独立复跑
有一遍 P=8 以 81.2 vs 81.6 ms 险胜 P=4——差 0.4 ms，纯噪声级；ms 级差异下 argmin
对噪声敏感，但机制结论（最优在 CPU 数附近、P>4 后开销主导）不变。L0 实验 [4] 用
µs 级小任务演示的「任务太细，开销吃掉收益」，在真实 Ray 上的位置是 P>4 之后的这段曲线。

**[5] 摊销与 Data-Juicer 默认参数的对照。** 本语料 3.74 MB，而 Data-Juicer 的分区
默认是 size=5000 条 / 上限 64 MB——按该默认，本语料只会被切成 1 个分区。默认参数
面向的是 GB 级语料；拿小语料上分布式，付的是启动费，赚不到并行。

## 5. 机制深挖

### 5.1 驱动端序列化的账：为什么「传值 4 次」不是 4 份 store 拷贝

Ray 提交任务时，参数走两条路：**内联**进任务 RPC，或先 put 进 store 再传引用。
分界线由 `task_rpc_inlined_bytes_limit`（10 MB）控制——单个任务 RPC 内联对象总字节
不超过该值（`src/ray/common/ray_config_def.h` @ ray-2.56.1，注释原文
"Max number bytes of inlined objects in a task rpc request/response."）。本语料
pickle 后 ~3.8 MB < 10 MB，所以传值参数**内联在每个任务 RPC 里**：提交 4 个任务，
驱动端就把 3.8 MB 序列化并随 RPC 发出 4 次。这解释了 7.1–7.7 ms ≈ 4 × 单次序列化。

「重复提交同一对象会不会便宜些？」——不会。探针用「同一对象重复提交」与
「等值新对象提交」对照，耗时在噪声范围内一致；源码侧也对得上：
`SerializationContext.serialize()`（`ray/_private/serialization.py:L771-789`，
本机安装 ray 2.56.1 验证）每次调用都走 msgpack 序列化路径，没有按对象身份的缓存。
**结论：想让 N 个任务共享一份数据，唯一的正路是 `ray.put` 一次 + 传 N 个 ref。**

> **方法论插话（这次差点翻车）**：第一版实验先 `ray.put` 再测传值提交，得到
> 「传值 4 次仅 9.2 ms」的假账——因为冷启动的一次性成本落在了 put 头上，传值段
> 看似免费。后来把预热分离、三路对照重测，才得到上面的真实比例。**测量分布式
> 系统时，一次性成本落在哪个计时区间里，结论就长成那个区间的形状。**

### 5.2 自动解引用：便利背后的两个坑

ObjectRef 参数在哪一层被解开？Python 侧没有这个逻辑——`flatten_args`
（`ray/_common/signature.py:L104-137`，ray 2.56.1 验证）对 ObjectRef 没有任何特殊
分支，ref 被原样打包进任务描述；解引用发生在 C++ core worker 于任务执行前完成的
依赖解析（机制细节见 L3 `[TODO: verify source]`）。Python 侧能看到的全部证据就是
[1b] 的两种报错，而这两处报错代码都能精确指认（ray 2.56.1 安装路径
`ray/_private/worker.py`）：

- `L2962-2966`：参数既非 ObjectRef 也非 list → `ValueError: Invalid type of
  object refs, <class 'int'>, is given...`（对应 bad_get(42) 现场 a）；
- `L2943` 起 list 分支 + `L992-998`：list 被当作「ref 列表」逐元素校验，
  元素不是 ObjectRef → `TypeError: Attempting to call 'get' on the value 1...`
  （对应 bad_get([1,2,3]) 现场 b）。

坑 b 比坑 a 危险：a 是「对数据调 get」立即炸；b 是「数据恰好长得像 ref 列表」时，
`ray.get` 会**逐个元素再解一次**——元素若真是 ObjectRef，就发生静默二次解引用，
程序不炸、语义已变。这也是 [1b] 结尾那句「想持有 ref 本身需包一层容器」的由来
（容器本身不是 ObjectRef 也不是 ObjectRef 列表，不会被误伤；L3 展开）。

### 5.3 为什么 4 worker 只测出 3.4x–3.5x

与 L2 同一答案的两半：

1. **Amdahl 串行段**。本实验 [2] 只计时局部 OP 并行段（3.37–3.54x），但完整
   pipeline 还有收敛点后的全局 dedup 串行段（~0.2 ms）与结果合并；[4] 里 P=1 的
   ~300 ms 是纯串行基准，P=4 的 ~82 ms 也不是 300/4=75 ms——任务提交、分区数据
   搬进搬出（每分区 pickle ~1.5 MB）都在并行段里占真实份额。
2. **返回值也走 store**。每个 `ray_local_ops` 的返回约 1.5 MB（pickle 实测），
   超过 100 KB 的内联阈值，按 ray-2.56.1 的 `ray_config_def.h` 注释原文，
   超过 `max_direct_call_object_size` 的返回值「are stored in plasma instead」——
   即并行段的输出不是顺着任务回复带回驱动，而是写进 object store 再由驱动取回。
   输入输出两头都要过 store，这是「数据密集任务」区别于「计算密集任务」的成本结构。

### 5.4 粒度扫描读出的两段曲线

把 [4] 的表画成心智图：P 从 1 到 4，wall time 按近直线下降（并行主导区）；
P 越过 4（=CPU 数）后曲线走平并微升（开销主导区：任务更碎，每任务的提交/调度/
序列化固定份额上升，且 4 个 CPU 也吃不下 8 个并发 CPU 密集任务）。**最优粒度不在
「切得越细越好」的方向上，而在并行收益与每任务开销的交点附近**——这与 L0 实验 [4]
（µs 任务跑不赢调度开销）是同一条物理规律的两种呈现。

## 6. 与 Ray 权威实现的对应

| 机制 | nano-ray L1 对应 | Ray 2.56.1 锚点 | 验证方式 |
|---|---|---|---|
| object store 喂数据 | [1] put/ref/传值三路对照 | `serialize()` 无身份缓存 `ray/_private/serialization.py:L771-789`；内联上限 `task_rpc_inlined_bytes_limit=10MB`（`src/ray/common/ray_config_def.h` @ ray-2.56.1） | 安装包行号 + tag 页面 + 探针 |
| ObjectRef 参数自动解引用 | [1b] 两种失败现场、[3] `*mapped_refs` 收敛点 | Python 侧无特殊分支 `ray/_common/signature.py:L104-137`；报错 `ray/_private/worker.py:L2962-2966` / `L2943`+`L992-998` | 安装包行号（本机 2026-08-05 验证） |
| 大返回值走 plasma | [2] 每分区返回 ~1.5 MB | `max_direct_call_object_size=100KB`（tag 注释：更大的返回值 stored in plasma） | tag 页面 + pickle 实测 |
| 全局 OP 收敛点 | `ray_global_dedup(*part_refs)` | 依赖解析在 C++ core worker，Python 无对应物 | 行为验证；细节 `[TODO: verify source]` → L3 |
| 多节点 / plasma 零拷贝 | 未覆盖 | — | `[TODO: verify on real system]`（真实 GPU/多机环境）→ L3 |

跨模块对照：漏斗 3360→2358→2110、naive 泄漏 236 = 跨分区对数，与 nano-data-juicer
L2（复现运行，其 29 处 Data-Juicer 锚点已两轮验证）逐位一致。

## 7. 费曼自检

- 能否解释：为什么把同一个 3.8 MB 语料**传值**提交给 10 个任务，驱动端要付 10 次
  全量序列化，而 `put` 一次 + 传 10 个 ref 只付 1 次？为什么「重复传同一对象」
  也不能省？
- 能否用 236/248 的账本解释：为什么全局去重不能分区各做各的？泄漏的到底是哪些样本？
- 能否说出 `ray_global_dedup(*mapped_refs)` 为什么就是「收敛点」的最小实现？
  它依赖的同步机制是哪一个（提示：不是 barrier，L0 就演示过）？
- 能否解释 [1b] 现场 b 为什么比现场 a 危险？什么样的真实数据会踩中？
- 能否用 [5] 的摊销算术，解释 Data-Juicer 默认分区参数（5000 条 / 64 MB）为什么对
  3.74 MB 语料「无效」？

## 8. 思考题

1. 传值参数超过 10 MB 内联上限时会发生什么（提示：参数会改走 object store）？
   设计一个实验验证：对比提交耗时与 store 对象数量（观测 store 需要
   `ray[default]` 的 state API，本机未装，`[TODO: verify on real system]`）。
2. 把 [4] 的扫描放到 8 CPU 机器上（`ray.init(num_cpus=8)`），预测 docs/s 在哪一档 P
   饱和，以及 P=16 时开销主导会不会更明显。`[TODO: verify on real system]`
3. 收敛点若从「一个任务收全部分区」改成「树状两两合并 + 去重」，wall time 可能更好，
   但 `keep-first` 语义还保得住吗？（提示：重复对保留哪一份取决于全局顺序；
   树状合并要怎么补偿顺序信息？）
4. [2] 的计时包含了分区数据搬进 worker、结果搬回 store 再取回的全过程，仍有 3.4x。
   若语料 ×100，估计哪一项开销的占比会显著上升？（提示：序列化与 store 读写是
   随数据量线性的，任务提交是随任务数线性的。）

## 9. 阶梯预告

- **L2（actor）**：全局去重的 sig 表目前每个任务各持一份、随收敛点合并；当索引本身
  是**有状态且要被多任务并发读写**的对象时，task 模型不够用，要换成 actor（有状态
  进程）。核心问题：task vs actor 的选择标准、actor 的并发语义。
- **L3（object store / plasma）**：[1] 只摸了 store 的账单，没摸它的构造——共享内存、
  零拷贝读取、引用计数与逐出。届时回来填本文的 `[TODO]`（C++ 依赖解析细节、
  超限参数路径、多节点行为、更大机器上的扫描），并解释 [1b] 的「包一层容器」到底包的是什么。

## 10. 溯源与口径声明

- **Ray 版本**：ray==2.56.1（pip 安装，Python 3.13.13，macOS arm64，单机 4 CPU）。
  文中所有安装包行号（`ray/_private/serialization.py`、`ray/_private/worker.py`、
  `ray/_common/signature.py`）于 2026-08-05 在本机 site-packages 逐条核对。
- **GitHub 锚点**：`src/ray/common/ray_config_def.h` 的两个配置常量与注释引自
  tag ray-2.56.1（2026-08-05 获取）；Ray 论文 arXiv:1712.05889（Moritz et al.）。
- **语料**：seed=42 合成语料（非真实语料）；L1 主题是执行机制（真进程 / store /
  提交成本），语料内容不影响验证。
- **计时口径**：所有 ms/s 为多批测量的观测区间，随机器负载浮动；区间覆盖复验环境
  连跑与另一次独立复跑（2026-08-05 同日同机）——
  init 2.5–2.9s；[1] put 1.4–2.0 ms、传值 4 次合计 7.0–7.9 ms、传引用合计 0.3–0.4 ms；
  [2] speedup 3.37x–3.54x（L2 multiprocessing 同段为 2.4x–3.3x）；[4] 最优 P=4 附近
  ± 一档（档位差 0.4 ms 级，噪声级；机制结论不变）；[5] 摊销约 9–10 轮。
  计数类输出（漏斗、账本、worker 数）逐字节一致，不随计时波动。
- **范围外**：多节点行为、plasma 零拷贝、C++ 依赖解析细节 → `[TODO: verify on
  real system]` / `[TODO: verify source]`，真实 GPU/多机环境与 L2/L3 接续。
