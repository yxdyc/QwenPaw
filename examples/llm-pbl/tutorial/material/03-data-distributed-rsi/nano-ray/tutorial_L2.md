# nano-ray · Tutorial L2 — actor：把有状态算子（全局去重索引）搬进有状态进程

> **K+1 定位**：L1 把同一套 OP pipeline 搬上了真实 Ray，全局去重用**收敛点任务**
> （一个任务收下全部分区）一次算完——语义正确，但有两个结构性限制：① 全部幸存者
> **数据**被搬进一个任务（数据向单点集中）；② sig 表只是该任务的局部变量，任务结束，
> 状态清零——索引活不过一次调用。L2 只加一层：把索引从「收敛点任务里的局部变量」
> 换成 **actor（有状态进程）**，回答三个问题——状态住在哪里才是「共享」的（[1]）？
> actor 的并发语义长什么样（[2]：默认串行是免费的并发安全，max_concurrency 打开
> 竞争）？同一个去重语义，更新规则对调度顺序的敏感度差多少（[3]：first-seen 反向喂
> 翻转 236 条，min-row_id 三种喂法逐位一致）？外加一笔诚实的成本账（[4]）。
>
> **跨模块契约**：与 `nano-data-juicer/L2_distributed_pipeline.py`、`nano-ray/L1_ray_pipeline.py`
> 同一语料（seed=42，3360 条）、同一执行计划、同一个漏斗：**3360 → 2358 → 2110**。
> 本文件直接 import L1 的语料构造、OP 与漏斗期望值（`EXPECTED_*`），跨模块一致性
> 在 [0] 处以机器断言复验。去重语义的「正确答案」不来自 actor，来自计划本身——
> actor 只是让状态有了一个可以常驻的家。

---

## 1. 问题：收敛点的两处结构性限制

L1 [3] 的收敛点是对的，也是小的：

```python
conv_out = ray.get(ray_global_dedup.remote(*mapped_refs))   # L1 的收敛点
```

- **数据向单点集中**。四个分区的全部幸存者（本语料 pickle 后 6.14 MB）搬进一个任务，
  结果（5.50 MB）再搬回来。数据量涨到 GB 级时，这个单点就是内存墙。
- **状态随任务生灭**。sig 表是任务体内的局部变量：任务一返回，索引蒸发。下一批数据
  想复用这张表？没有下一批——task 模型里没有「活着等下一次调用」的东西。

三类真实需求会立刻撞墙：多个生产者**并发**向同一个索引写入；索引要**跨批次常驻**
（增量去重 / 在线服务）；结果必须与**到达顺序无关**（调度不该改变语义）。这三件事
的共同前提是：有一个「有状态、可被多方访问、活得比任何一次调用更久」的对象——
在 Ray 里，这就是 **actor**。

| L1（task 模型） | L2（actor 模型） | 变化的本质 |
|---|---|---|
| sig 表 = 收敛点任务的局部变量 | sig 表 = actor 进程的实例属性 | 状态的住所：栈帧 → 常驻进程 |
| 数据搬进收敛点（6.14 MB） | 只搬 (sig, row_id) 索引（94 KB） | 集中的是数据 → 集中的是知识 |
| 一次性，收齐才算 | 常驻，随到随注册 | 批处理 → 可增量/在线 |
| 顺序由「任务一次收齐」天然保证 | 顺序由**更新规则**自己负责 | 语义责任从 runtime 移回算法 |

最后一行是 L2 的核心教训，[3] 用一个 236 条翻转的反例把它钉死。

## 2. 运行与输出

环境：`pip install ray`（本机实测 ray==2.56.1 / Python 3.13.13 / macOS arm64，单机
模式，4 CPU）。复用同目录 `L1_ray_pipeline.py`（import 前设 `sys.dont_write_bytecode`，
全树零 `__pycache__`）。任意 CWD 可运行。

```bash
python L2_actor_dedup_index.py
```

```
====================================================================
nano-ray L2 — actor：把有状态算子（全局去重索引）搬进有状态进程
====================================================================
python 3.13.13 | ray 2.56.1
声明: 真实 Ray 单机模式（本地 raylet + worker 进程），无 mock；
      复用 L1_ray_pipeline 的语料/OP/漏斗常量（跨模块一致性契约）。

[0] ray.init 2.9s | 语料 3360 docs -> 过滤后 2358 -> 去重后 2110（漏斗与 L1 逐位一致 ✅）
    重复对账本: 共 248 对 | 跨分区 236 对（== EXPECTED_LEAK，契约复验 ✅）

[1] actor = 专属有状态进程：状态住在进程里，跨调用存活
    4 次 incr -> [1, 2, 3, 4]（状态跨调用累积 ✅）
    4 次调用全落在同一个进程（唯一 pid 数 = 1），且该进程 ≠ 驱动进程 ✅
    （对照 L1：4 个分区任务散在 4 个 worker 进程；actor 是一个常驻进程）
    task 路线对照：同样数到 4，但状态以参数/返回值 8 次穿越驱动端；
    actor 路线 0 次——多个生产者要共享同一份可变状态时，task 模型逼
    所有人经驱动端中转，actor 给出唯一的共享住所。

[2] actor 的并发语义
    a) 默认串行 actor：8 生产者 × 25 条 = 200 条，丢失 0 条 ✅
       每个 caller 的 seq 严格递增（FIFO 保持）✅ | 8 个生产者交错排队成立（切换次数 ≥ 8 ✅）
    b) max_concurrency=2 + 决定性 barrier：两次 +1，x 停在 1 ❌（两次调用都读到 0、都写回 1：[(0, 1), (0, 1)]）
       对照（默认串行 actor 同样两次 bump）：x = 2 ✅（[(0, 1), (1, 2)]）
    → 默认串行 = 免费的并发安全；max_concurrency 打开吞吐也打开竞争，
      原子性要自己负责（方法内加锁，或保持串行、用批量摊薄 RPC）。

[3] 全局去重做成 actor 服务（喂的是索引，不是数据）
    a) first-seen 索引（到达顺序敏感）：
       正向逐条喂 2358 次 offer（811 ms）
       == 串行 dedup 逐位一致 ✅（到达顺序 == 全局顺序时语义重合）
       反向按分区喂：仍 2110 条，但 236 个重复对的
       幸存者翻成了 copy ❌——翻转数 == 跨分区对数 236：
       反序喂让高分区（row_id 更大）的 copy 先到；同分区的 12 对内部顺序未变。条数对、内容错——
       「第一次出现」是全局顺序语义，单阶段规则把顺序假设藏进了 RPC 时序。
    b) min-row_id 索引（两阶段：可交换聚合 + 收敛排序）：
       同样反向喂：keeper 序列 == 串行 pipeline ✅
       4 任务并发喂（14 ms，到达先后不确定）：
       仍逐位一致 ✅ | 索引条目 2110 == 去重后条数
       → keep-first 被拆成「register: min 聚合（顺序免疫）+ keepers:
         收敛后按 row_id 排序输出」，顺序语义不再住在 RPC 时序里。
    c) 组装：按 keeper row_id 回原分区取样本 == 串行 pipeline 逐位
       一致 ✅ —— 漏斗 3360->2358->2110 与 L1 / nano-data-juicer L2 完全一致

[4] 成本账：actor 路线搬「知识」，收敛点搬「数据」
    actor 创建（进程拉起+首次调用）: 441 ms（一次性）
    单次方法调用 RPC: 0.37 ms（预热后 50 次均值）
    搬进: 收敛点 6.14 MB（2358 条样本全体） vs 索引路线 94 KB（2358 条 (sig,row_id)），差 65x
    搬出: 收敛点返回 5.50 MB 样本 vs 索引路线返回 6.2 KB row_id
    墙钟（本语料 3.7 MB / 2358 条）: 收敛点 11.5 ms | actor 批量喂 14.4 ms | actor 逐条喂 811 ms
    逐条喂预测对账: 2358 x 0.37 ms ≈ 872 ms（实测 811 ms，同量级 ✅）
    → 收敛点：数据向一个任务集中；actor：知识向一个进程集中、数据留
      在原分区。本规模下吞吐不是选 actor 的理由——RPC 笔数是真成本，
      批量是标准解（L1 [4] 任务粒度账的 actor 版回声）；选 actor 的理由是
      语义（并发生产者 / 增量喂入 / 顺序免疫 / 索引常驻），不是这里的吞吐。

[5] task vs actor 选择标准
    无状态、可并行、无共享可变需求        -> task（L1 全图）
    状态跨调用存活、多生产者并发读写      -> actor（默认串行=免费并发安全）
    一次性收齐的纯收敛计算                -> 收敛点 task（[4]：与批量 actor 同量级、无常驻成本）
    索引要增量/在线/并发喂、且顺序免疫    -> actor + 可交换更新规则（[3b]）
    为吞吐开 max_concurrency              -> 先证明竞争无害（[2b] 是反例模板）

====================================================================
✅ self-check passed:
   漏斗 3360->2358->2110 与 L1 / nano-data-juicer L2 一致 /
   actor 状态跨调用 + 单进程证据 / 串行 actor 200 条零丢失 /
   lost update 决定性复现 x=1（对照串行 x=2）/
   first-seen 反向喂翻转 236 == 跨分区对数 /
   min-row_id 反向喂+并发喂 == 串行 / 组装 == 串行 pipeline 逐位一致
====================================================================

takeaway: 同一个去重语义——task 写法把状态藏在一次性任务里（收敛点），
          actor 写法把状态放进常驻进程里（索引服务）。语义正确与否不取
          决于 runtime，取决于更新规则有没有把顺序假设藏进 RPC 时序：
          first-seen 藏了，236 条翻转给你看；min-row_id 没藏，顺序与
          并发都免疫。L3 进 object store / plasma。
```

2026-08-07 的 3 次连续运行 + 1 次任意 CWD 复跑：全部 EXIT=0；掩掉计时
数字后输出逐字节一致（diff 核验）；计时行随机器负载浮动（区间见 §11）。[2b] 的
lost update 是 barrier 构造的**决定性**复现，不依赖调度巧合，三遍跑三遍同。

**同日独立复跑 7 遍**（含任意 CWD 3 遍）：全部 EXIT=0。7 遍跨两次
措辞修正：前 3 遍为修正前基线，第 4–5 遍在 [5] 一行修正后，第 6–7 遍在 [4] 收尾
修正后——掩计时后同版本内逐字节一致，跨版本差异恰且仅为被修正的行。最后 2 遍与
上方粘贴输出掩计时后 diff 零差异。早期运行中的「收敛点 task（本规模墙钟更优）」
没有在后续复测中稳定成立——收敛点 14.4–24.3 ms vs 批量喂 12.2–14.8 ms，同量级、先后随负载
翻转（定稿批恰是收敛点略快）；代码与上方粘贴已同步改为「与批量 actor 同量级、
无常驻成本」与「本规模下吞吐不是选 actor 的理由」——只陈述跨批稳定的事实。两批
计时并集见 §11。

## 3. 代码结构：一个服务、两种规则、一笔账

`L2_actor_dedup_index.py` 分四段：

1. **复用 L1（import，不重跑）**。`import L1_ray_pipeline as L1`：语料 `make_corpus`、
   局部 OP `local_ops`、串行去重参照 `dedup_keep_first`、分区 `partition`、漏斗常量
   `EXPECTED_AFTER_FILTER/DEDUP/LEAK` 全部来自 L1——跨模块契约不是口头约定，是
   import 进来的断言基线。`sys.dont_write_bytecode = True` 保证 import 不落
   `__pycache__`（全树零 pyc 约定）。
2. **六个 actor + 四个 task，全部自包含**。remote 代码被 cloudpickle 送进 worker 时，
   `__main__` 里的类按值序列化；为不依赖「worker 能 import L1」，所有 actor 方法体与
   task 体**不引用 L1 的名字**（`converge_dedup` 甚至内联复刻了去重循环）。角色表：
   `StatefulCounter`（[1] 状态跨调用 + [4] RPC 探针）、`incr_task`（[1] task 路线对照）、
   `SerialRecorder` + `caller_loop`（[2a] 8 生产者并发写账本）、`RacyCounter`
   （[2b] max_concurrency=2 + barrier 决定性 lost update）、`SafeCounter`（[2b] 串行对照）、
   `FirstSeenIndex`（[3a] 到达顺序敏感）、`MinRowIndex`（[3b] 可交换聚合）、
   `feed_partition`（[3b] 生产者任务，吃 ObjectRef + actor handle）、
   `converge_dedup`（[4] 收敛点复刻）。
3. **实验 [0]–[5]**，见 §4 逐段解读。
4. **硬编码期望值断言**：`EXPECTED_PAIRS=248`（过滤后幸存者中的重复对总数，来自
   L1 [3] 的账本算术，在 [0] 复验）；其余 `EXPECTED_*` 从 L1 import。断言失败即
   说明「actor 路线悄悄改了语义」——这是 L2 的跨级别契约。

两个索引 actor 的规则值得并排看一眼：

```python
# first-seen：谁先到谁留下 —— 语义 = f(到达顺序)
def offer(self, sig, row_id):
    if sig in self.first:
        return False
    self.first[sig] = row_id
    return True

# min-row_id：只记最小 row_id —— min 是可交换聚合，与到达顺序无关
def register(self, sig, row_id):
    cur = self.best.get(sig)
    if cur is None or row_id < cur:
        self.best[sig] = row_id
```

`offer` 的正确性依赖「到达顺序 == 全局 row_id 顺序」这个**外部条件**；`register` 的
正确性只依赖 min 的结合律/交换律，条件全在规则内部。[3] 的实验就是把这个区别
变成可观测的 236 条翻转。

## 4. 输出逐段解读

**[0] 契约复验。** `ray.init` 2.6–4.5s（两批独立测量分别为 2.6–2.9s、3.0–4.5s，
随负载浮动；当日首跑冷启动到 4.5s，如实记录）。串行参照漏斗 3360→2358→2110 与
重复对账本（248 对 / 跨分区 236 对）在 L2 现场重算并断言——actor 实验还没开始，
基线先钉死。

**[1] 状态住在进程里。** 同一个 actor 连调 4 次 `incr` 得到 `[1,2,3,4]`：状态跨调用
累积，且 4 次调用全落在同一个进程（唯一 pid），该进程 ≠ 驱动进程。对照 L1：4 个
分区 task 散在 4 个 worker 进程上，task 之间没有共享状态可言。task 路线的等价写法
（`incr_task` 把 n 作为参数传入、返回值带出）也能数到 4，但状态以参数/返回值形式
**8 次穿越驱动端**——每一跳都是一次序列化 + 一次 RPC。actor 路线 0 次：状态住在
actor 进程里，调用方只需要 handle。单个计数器看不出差别；当生产者变成 8 个、状态
变成一张 2110 条的索引时，「谁持有状态」就是架构问题，不再是修辞。

**[2a] 默认串行 = 免费的并发安全。** 8 个 `caller_loop` 任务并发向同一个
`SerialRecorder` 写 25 条账目：200 条全额落账、零丢失，每个 caller 的 seq 严格递增，
caller 之间交错排队（切换次数 ≥ 8）。200 条无损本身就是串行执行的证据——若方法
并发执行，8 路写同一个 list 会互相践踏。这一条对应 ray 2.56.1 的默认值：threaded
actor 的 `max_concurrency` 默认为 1（`ray/_private/ray_constants.py:L468`，
`DEFAULT_MAX_CONCURRENCY_THREADED = 1`；设置逻辑在 `ray/actor.py:L1808-1813`）。
**串行不是性能缺陷，是并发安全的默认形态**——索引这类「多方写、状态敏感」的对象，
默认就站在安全的一侧。

**[2b] max_concurrency 打开的是竞争，不只是吞吐。** `RacyCounter` 用
`@ray.remote(max_concurrency=2)` 声明，两个 `bump` 在 actor 进程内**真的并发**了：
barrier 强制两个线程都先读到 `x=0` 再写回，两次 +1 只生效一次，x 停在 1，两次调用
都返回 `(0, 1)`。对照的 `SafeCounter`（默认串行）同样两次 bump 得到 `[(0,1),(1,2)]`、
x=2。barrier 的构造细节见 §5.3——关键性质是**决定性**：不靠调度巧合，重跑必现。
源码侧的对应：`max_concurrency > 1` 时 Ray 明确声明执行顺序不再保证
（`ray/actor.py:L1662-1666` docstring 原文 "the execution order is not guaranteed
when max_concurrency > 1"；`allow_out_of_order_execution` 在该条件下默认为 True，
`ray/actor.py:L2037-2038`）。

**[3a] first-seen：条数对、内容错。** 正向逐条喂 2358 次 `offer`，结果与串行
`dedup_keep_first` 逐位一致——因为到达顺序恰好就是全局 row_id 顺序。把喂入改成
**分区反序**（分区 3→0，分区内部仍正序），条数仍是 2110，但 **236 个重复对的幸存
者翻成了 copy**：翻转数恰等于跨分区重复对数（EXPECTED_LEAK）。证明一句话：copy 的
row_id 恒大于 source，跨分区时 copy 必在更高分区；反序喂让更高分区先到，于是每个
跨分区对的「第一次出现」变成了 copy。同分区的 12 对内部顺序未变，幸存者不变。
这是全部实验里最值得盯住的一行：**条数对、内容错**——漏斗数字全绿，语义已经变了。
单阶段 first-seen 规则把「全局顺序」这个假设藏进了 RPC 时序，生产者喂法一变就翻车。

**[3b] min-row_id：把顺序语义拆成可交换聚合。** 两阶段：`register` 只对每个 sig 记
最小 row_id（min 与到达顺序、并发交错完全无关），`keepers` 在收敛后按 row_id 排序
输出。本 pipeline 的全局顺序就是 row_id 顺序，所以 keep-first ≡ keep-min-row_id。
同一个**反向喂**（[3a] 翻车的那次），min 版 keeper 序列 == 串行 pipeline；再换成
4 个任务**并发喂**（到达先后不确定），仍逐位一致；索引条目 2110 == 去重后条数。
顺序语义从「RPC 时序」搬回了「数据本身的序」——调度怎么抖，结论不变。

**[3c] 组装。** 数据从头到尾没挪过窝：样本留在原分区，只有 keeper row_id 集合回传，
按 row_id 回原分区取样本，结果与串行 pipeline 逐位一致。漏斗 3360→2358→2110 与
L1 / nano-data-juicer L2 完全一致——跨模块契约第三次闭合。

**[4] 成本账（见 §5.5 深挖）。** 搬进：收敛点 6.14 MB vs 索引路线 94 KB（65x）；
搬出：5.50 MB vs 6.2 KB。但墙钟上 actor 路线开不出差距：收敛点与批量喂**同量级、
先后随负载翻转**（第一批 11.5 vs 14.4 ms，第二批 14.4–24.3 vs 12.2–14.8 ms）≪
逐条喂（811–1022 ms）——**RPC 笔数是真成本**（第一批：2358 × 0.37 ms ≈ 872 ms 对
实测 811 ms；第二批：2358 × 0.40–0.69 ms ≈ 939–1616 ms 对实测 877–1022 ms，同量级
对账两批都成立），批量把 4 笔 RPC 摊到与收敛点同量级。选 actor 的理由从来不是
这个量级的吞吐。

## 5. 机制深挖

### 5.1 状态的三个住所

L2 把「状态」的住所摊开成三行账：

| 住所 | 载体 | 生命周期 | 共享方式 | 本文件的实例 |
|---|---|---|---|---|
| 数据里 | 参数/返回值 | 一次调用 | 谁拿到数据谁有 | `incr_task`（n 来回 8 次穿越驱动端） |
| 任务栈帧 | 局部变量 | 任务存活期 | 不可共享 | L1 收敛点的 sig 表 |
| 常驻进程 | actor 实例属性 | actor 存活期 | 所有持 handle 者 | `FirstSeenIndex` / `MinRowIndex` |

task 模型里状态只能住前两处：要么随数据搬（每个想用它的人都得经驱动端中转），
要么随任务生灭（任务一返回就蒸发）。多个生产者要**并发读写同一份可变状态**时，
前两处都塌：数据路线把所有人串成驱动端的一条队列，栈帧路线根本没有「所有人」。
actor 是 Ray 给第三种住所的名字——一个有 pid、有实例属性、能被多方持有的进程。

### 5.2 默认串行：免费但真实的并发安全

为什么 [2a] 的 200 条账一条不丢？因为 threaded actor 的方法默认**一次只执行一个**：
`max_concurrency` 的默认值是 1（`ray/_private/ray_constants.py:L468`；asyncio actor
则是 1000，`ray/_common/ray_constants.py:L2`——async 的并发模型不同，本文件不涉及）。
8 个生产者的调用在 actor 门口排成一条队，逐个进入——list.append 永远不会被另一路
并发写踩中。**这不是「还没优化」，是语义合同**：Ray 用串行换来了「actor 方法内
不需要锁」的默认保证，代价是方法级吞吐上限 = 单次方法耗时。

`max_concurrency=N` 是对这份合同的**显式违约声明**：N 个同名方法可在 actor 进程内
并发执行（threaded actor 走进程内线程；并发执行的精确 C++ 位置尚未逐行定位，
`[TODO: verify source]`，行为侧由 [2b] 的 barrier 实验实证——两个线程确实同时在
方法体内）。换来的后果也写在同一份 docstring 里：执行顺序不再保证
（`ray/actor.py:L1662-1666`），且 `allow_out_of_order_execution` 自动翻成 True
（`ray/actor.py:L2037-2038`）。[2b] 的 x=1 就是违约的账单。

### 5.3 决定性 lost update：barrier 怎么把调度钉死

lost update 的通常写法（两个并发 +1，「有时」得到 1）依赖调度巧合，跑十遍可能
十遍都对——教学上等于没讲。L2 用 barrier 把交错**构造**出来：

```python
def bump(self):                       # max_concurrency=2
    v = self.x                        # ① 读
    with self._lock:
        self._readers += 1
        first = (self._readers == 1)
    if first:
        self._gate.wait(timeout=5.0)  # ② 第一个读者停在闸前
    else:
        self._gate.set()              # ③ 第二个读者放行
    self.x = v + 1                    # ④ 写（两人的 v 都是 0）
    return v, self.x
```

两次调用同时提交，actor 进程内两个线程各占一个并发槽。无论谁先拿到锁记为「第一个
读者」，因果链都相同：第一个读者读完 v=0 后停在 ②，**写（④）被闸挡住**；第二个
读者此时进入、读到的必然是 0（第一个读者还没写）、在 ③ 放行。两个线程各自写回
0+1=1——x 停在 1，两次调用都返回 `(0,1)`。超时分支（5s）只是保险丝：即便走到，
两边写的仍是 1，结果不变，只伤墙钟。所以这个反例**重跑必现**，适合当模板：任何
「读-改-写」方法开了并发，都等价于把 ②③ 之间交给调度器摆布。修复只有两条路：
把读-改-写收进锁里（原子性自己负责），或干脆不开并发、用批量摊薄 RPC。

### 5.4 顺序敏感 vs 顺序免疫：236 的算术

[3a] 的翻转数为什么恰是 236？把账摊开（与 L1 [3] 同一本账）：

- 过滤后幸存者中的重复对共 248 对，其中跨分区 236 对、同分区 12 对；
- 每对的 source 恒有较小 row_id（copy 恒在 source 之后构造，row_id 即全局位置）；
- 分区是连续 chunk：分区 p 持有 row_id ∈ [840p, 840(p+1))，所以跨分区对的 copy
  必在**更高**编号的分区；
- 分区反序喂（3→0、区内正序）：同分区对内部顺序未变 → 12 对幸存者不变；跨分区对
  的 copy 先于 source 到达 → 236 对的幸存者全部翻成 copy。

于是得到那个最阴的失败形态：**条数 2110 一分不差，236 条内容已经换人**。任何只看
漏斗数字的监控都会放行它。

min-row_id 的免疫性来自一条代数性质：`min` 是可交换、可结合的聚合——更新序列任意
重排、任意并发交错，每个 sig 的终值不变。两阶段拆分（register 聚合 + keepers 收敛
排序）把「keep-first」里的顺序语义从执行时序中剥离，重新挂回数据自带的 row_id 全序。
这个思路不是去重专属：**凡是「全局第一次/最小/求和」类语义要上并发写入，先把更新
规则化成可交换聚合，再在收敛点排序输出**——顺序假设从 runtime 挪回数据，调度就
再也伤不到语义。（方向上与 CRDT 的 state-based 思路同源；这里不展开，只记同向。）

### 5.5 数据集中 vs 知识集中：65x 的账与 RPC 的账

两条路线搬的东西完全不同：

| 路线 | 搬进 | 搬出 | 数据留在哪 |
|---|---|---|---|
| 收敛点 task | 6.14 MB（2358 条样本全体） | 5.50 MB（2110 条样本） | 全搬走了 |
| actor 索引 | 94 KB（2358 条 (sig,row_id)） | 6.2 KB（2110 个 row_id） | 原分区，没挪窝 |

65x 的差距来自一个观察：**去重需要的是知识（谁和谁相同），不是数据本身**。样本
留在原分区（object store 里躺着），只有签名上行、row_id 下行。语料越大这个差越悬殊
——sig 是定长 32 字节，样本是变长 KB 级。

但账还有另一半：RPC 笔数。逐条喂 = 2358 次方法调用 × ~0.37 ms/次 ≈ 872 ms 的预测，
与实测 811 ms 同量级对账（另一批单价漂到 0.40–0.69 ms，对账仍成立）——**每次方法
调用都是一笔真实的 RPC**（序列化、提交、调度、返回），actor 不是本地对象。批量
（4 次 `register_many`）把这笔账摊到与收敛点同量级。所以本规模的诚实结论是：

- 墙钟：收敛点与批量 actor **同量级**（11–24 ms 带，两批之间先后翻转）≪ 逐条
  actor（0.64–1.02 s）。本规模 actor 路线买不到吞吐；
- actor 的创建还有一次性成本：实测 55–630 ms，双峰——取决于 Ray 预启动 worker 池
  有没有余量，池空时付一次完整的进程拉起（§9 有探针记录）；
- 选 actor 的理由是**语义**：并发生产者、增量喂入、顺序免疫、索引常驻。这些能力
  在收敛点模型里不存在，不是快慢问题。

顺带一个实测发现（值得记住，也值得警惕）：**ray 2.56.1 的默认 actor 不占 CPU 配额**。
选项表里 actor 的 `num_cpus` 没有默认值（`ray/_common/ray_option_utils.py:L137` 的
common 项无 default；task 才有 `default_value=1`，`L179`；`_actor_only_options`
（`L226` 起）不覆盖该项），运行时 metadata 实测为 None。行为侧两个探针：8 个默认
actor 在 4-CPU 预算上全部并发运行（8 个不同 pid，`available_resources` 的 CPU 恒为
4.0）；而显式声明 `num_cpus=2` 的 actor 第 3 个起就阻塞在准入上（预算只容得下 2 个）。
**资源配额是准入券，不是物理隔离**——默认 actor 拿到的是「不限」的准入券。这解释
了本文件 8+ 个 actor 并存为何毫无压力，也提醒：配额要自己声明，别指望默认值替你
做容量规划。

## 6. 与 Ray 权威实现的对应

| 机制 | nano-ray L2 对应 | Ray 2.56.1 锚点 | 验证方式 |
|---|---|---|---|
| threaded actor 默认串行 | [2a] 200 条零丢失 | `DEFAULT_MAX_CONCURRENCY_THREADED = 1`（`ray/_private/ray_constants.py:L468`）；默认值设置 `ray/actor.py:L1808-1813` | 安装包行号（2026-08-07 本机验证）+ 行为断言 |
| max_concurrency 的顺序代价 | [2b] lost update 反例 | docstring "execution order is not guaranteed when max_concurrency > 1"（`ray/actor.py:L1662-1666`）；`allow_out_of_order_execution` 默认翻转 `ray/actor.py:L2037-2038` | 同上 |
| threaded 并发执行的载体 | [2b] barrier 双线程 | 进程内线程并发；精确 C++ 位置未逐行定位 | 行为实证；`[TODO: verify source]` |
| actor 方法白名单解析 | §9.1 实现陷阱（`safe.get` AttributeError） | `inspect.getmembers` 收集 `ray/actor.py:L1221`；`_method_shells` `L2281`；`__getattr__` 严格校验 `L2442-2476` | 安装包行号 + 报错现场复现 |
| 默认 actor num_cpus=None | §5.5 双探针 | `ray/_common/ray_option_utils.py:L137`（无 default）vs `L179`（task default=1）；`_actor_only_options` `L226` 起无覆盖 | 安装包行号 + runtime metadata + 行为探针 |
| actor handle 作为 task 参数 | [3b] `feed_partition` | —（行为级特性） | 行为验证 |
| ObjectRef 参数自动解引用 | [3b] pair_refs 进 task | 同 L1（`ray/_common/signature.py` 无特殊分支、C++ core worker 解引用） | L1 已验证，此处复用 |
| 漏斗契约 3360→2358→2110 | [0]/[3c] 断言 | 跨模块：nano-data-juicer L2 / nano-ray L1 | 三执行器逐位一致 |

## 7. 费曼自检：户籍处与现场清点

把全局去重想象成一次**人口登记**：

- **L1 的收敛点** = 现场清点：所有部门把**整本户籍册**（样本全体）搬到一间会议室，
  清点人当场翻册子去重。数是对的——但册子搬来搬去（6.14 MB），而且会议一散，
  清点人手写的底稿（sig 表）就扔了。下次清点，从头再来。
- **L2 的 actor** = 常年开门的**户籍处**：各部门只需上报「户号 + 编号」
  （sig + row_id，94 KB），册子留在各部门不动，户籍处维护唯一一本台账，随时可查。
- **first-seen 规则** = 「谁先来报到就给谁上户口」——结果取决于**报到的顺序**。
  通知部门的顺序一换，同一户里登记的人就换了（236 户换了人），而总户数（2110）
  不变——上级看报表，一片祥和。
- **min-row_id 规则** = 「不管谁来报，同一户号只认编号最小的那位」——谁来谁先来，
  甚至几个人同时挤进门，登记的结论都一样。顺序从「门口的队伍」搬回了「编号本身」。
- **默认串行** = 户籍处只开一个窗口：队伍再长也不会两人同时改同一页台账——慢，
  但永远不会写错。**max_concurrency** = 多开窗口：快了，但两个职员同时改一页时，
  得自己定规矩（锁），否则 [2b] 的账就是你的账。

类比的边界：真实户籍处的「台账」有持久化，nano 的 actor 状态只在进程内存里
（actor 死了状态就没了——持久化是 L3 之后的话题）；「一个窗口」对应的是方法级
串行，actor 进程里其实还有别的线程设施（barrier 实验用的就是它们）。

## 8. 思考题

1. **fire-and-forget 的顺序**：把 `caller_loop` 的逐条 `ray.get` 改成 fire-and-forget
   （不等待），预测每个 caller 的 seq 是否仍然严格递增，跑三遍验证；再去 Ray 文档/
   源码里找「同一 caller 到同一 actor 的调用顺序」这条保证的准确表述
   `[TODO: verify]`。
2. **给 RacyCounter 上锁**：在 `bump` 的「读-改-写」外加一把 `threading.Lock`，
   保持 `max_concurrency=2`。x 还会丢吗？墙钟相对串行版本变快、变慢还是持平
   （提示：本规模下锁的开销与并发的收益都是 ms 级以下）？锁保护的是什么，
   付出的又是什么？
3. **保留 first-seen 语义的修法**：下游真的需要「全局第一次出现」（而不是
   「最小 row_id」恰好等价的形式），且必须并发喂入。给生产者加什么信息、把 actor
   规则改成什么，能在任意到达顺序下复现正向喂的结果？（提示：把顺序**变成数据**
   随请求携带——本题与 [3b] 是同一条路的两次走法。）
4. **规模算术**：语料 ×100（23.6 万幸存者）时，收敛点路线与 actor 路线各自的
   **字节数**和 **RPC 笔数**如何增长？哪条路线先撞单任务内存上限？
   （提示：收敛点把全部数据搬进一个任务；actor 路线的每笔 RPC 只搬签名。）
5. **准入券的代价**：默认 actor `num_cpus=None`（§5.5）。如果建 100 个索引 actor
   且同时活跃，物理上会发生什么？给哪类 actor 显式声明 `num_cpus` 是合理的容量
   规划，哪类声明了反而浪费？（提示：准入券 vs 物理核；IO 密集 vs CPU 密集。）

## 9. 反例与边界

1. **self-check 捕获的失败**：`SafeCounter` 第一版漏写了 `get` 方法，
   `safe.get.remote()` 当场抛 `AttributeError: 'ActorHandle' object has no
   attribute 'get'`。第一反应怀疑 runtime，查了 `ray/actor.py` 才确认是自己漏写：
   ray 2.56 的 actor 方法解析是**白名单**——`inspect.getmembers` 在装饰时收集方法
   （`L1221`），`__getattr__` 严格校验、查无即报（`L2442-2476`）。这比 duck typing
   友好：拼错方法名在取属性当场就炸，不会拖到 worker 里。教训：actor 报错先看
   handle 侧，再看 runtime 侧。
2. **测量卫生（L1 教训再次应验）**：`t_feed_conc` 第一版测出 71 ms vs 447–514 ms
   的巨幅波动——新建 `mr_conc` actor 的**进程拉起**（一次性成本）落在了计时窗内。
   加一次 `size()` 预热后稳定在 11–15 ms。一次性成本落在哪个计时区间里，结论就
   长成那个区间的形状——这句话从 L1 的 put/传值实验一路跟到 L2。
3. **actor 创建成本的双峰**：[4] 的「进程拉起 + 首次调用」实测 55–630 ms：Ray 有
   预启动的 worker 池，池有余量时创建只要几十 ms，池空时付一次完整的进程启动。
   含义：**actor 要复用，不要随用随建**；把创建成本当一次性预算，别放进热路径。
4. **默认 actor 不占 CPU 配额**：§5.5 的双探针结论（8 个默认 actor 共存于 4-CPU
   预算 / 显式 num_cpus=2 则阻塞准入）。这是对「一个 actor 占一核」直觉的修正；
   也再次说明：Ray 的资源是调度层的准入券，不做物理隔离。版本边界：本结论锚在
   ray 2.56.1 的选项默认值上，跨版本引用前需复核。
5. **本规模下 actor 路线买不到吞吐**：[4] 的墙钟是诚实的——3.7 MB / 2358 条的量级上，
   收敛点与批量 actor 同量级（11–24 ms 带，先后随负载翻转），逐条 actor 慢两个量级。
   actor 的收益在语义（并发/增量/顺序免疫/常驻），且要到数据更大、生产者更多、
   索引要跨批次复用的场景才兑现成吞吐或架构优势。别拿本文的墙钟去推销 actor。
6. **范围外**：多节点行为、plasma 零拷贝、actor 状态持久化、async actor
   （max_concurrency 默认 1000 的另一套并发模型）→ `[TODO: verify on real system]`
   （真实 GPU/多机环境）与 L3。

## 10. 阶梯预告

- **L3（object store / plasma）**：L1 [1] 只摸了 store 的账单，没摸它的构造——
  共享内存、零拷贝读取、引用计数与逐出。届时一并填：L1 [1b] 承诺的「包一层容器」
  到底包的是什么、C++ core worker 依赖解析的精确位置（L1/L2 两处
  `[TODO: verify source]`）、以及本文 [4] 的字节账在零拷贝口径下怎么改写。
- 交叉引用：本文的「可交换聚合 + 收敛排序」与 nano-data-juicer L2 的全局算子收敛
  同构（那边用进程间队列，这边用 actor）；「可靠性/语义正确是设计出来的属性」与
  nano-vllm-sglang L2（抢占后语义不变）、nano-qwenpaw L1（store 才是 memory）同族。

## 11. 溯源与口径声明

- **Ray 版本**：ray==2.56.1（pip 安装，Python 3.13.13，macOS arm64，单机 4 CPU）。
  文中所有安装包行号（`ray/_private/ray_constants.py:L468`、`ray/_common/ray_constants.py:L2`、
  `ray/actor.py:L1221/L1662-1666/L1808-1813/L2037-2038/L2281/L2442-2476`、
  `ray/_common/ray_option_utils.py:L137/L179/L226`）于 2026-08-07 在本机
  site-packages 逐条核对。
- **GitHub 锚点**：tag ray-2.56.1（2026-08-07 raw 通道可达，已抓取
  `src/ray/core_worker/core_worker.cc` 与 `python/ray/_raylet.pyx` 用于定位并发执行
  路径；threaded actor 并发执行的精确 C++ 位置尚未逐行钉死，标
  `[TODO: verify source]`）。Ray 论文 arXiv:1712.05889（Moritz et al.）。
- **语料**：seed=42 合成语料（非真实语料），与 nano-data-juicer L2 / nano-ray L1
  同一构造；L2 主题是 actor 的状态与并发语义，语料内容不影响验证。
- **计时口径**：所有 ms/s 为 2026-08-07 **两批**同机观测的并集区间
  （第一批 3 遍；第二批 7 遍，其中任意 CWD 3 遍），随机器负载浮动：
  init 2.6–4.5s（当日首跑冷启动 4.5s 如实记录）；单次 RPC 0.29–0.69 ms；逐条喂
  641–1022 ms；批量并发喂 11–15 ms（预热后）；收敛点 11.0–24.3 ms；actor 创建
  55–630 ms（双峰，见 §9.3）。收敛点与批量喂的先后顺序在两批之间翻转（同量级
  噪声），本文选材结论不依赖该顺序。计数类输出（漏斗、账本、翻转数、RPC 对账的
  结构部分）两批共 11 遍掩计时后逐字节一致（diff 核验），不随计时波动。
- **范围外**：多节点行为、plasma 零拷贝、actor 持久化、async actor → `[TODO:
  verify on real system]`，真实 GPU/多机环境与 L3 接续。
