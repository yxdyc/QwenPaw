# nano-ray · L0 教程：任务图、对象存储，与「把并行变成调度问题」

> **本节目标（L0）**：用 ~270 行纯 Python（含约 60 行触发时序回归测试）抓住
> Ray 编程模型的三个核心机制——
> 把函数变成可调度的任务（future）、让数据按引用传递（object store）、
> 让调度器按依赖自动触发下游（dynamic task graph）。
> **前置**：无硬前置；读过 nano-data-juicer L1（串行 pipeline）更容易理解动机。
> **本节 K+1**：从「手写线程池代码」到「描述任务图，并行调度交给系统」。

---

## 1. 问题：串行 pipeline 与手写并行

nano-data-juicer L1 里有一条串行 pipeline：OP 一个接一个，在单进程里处理整份数据。
数据一大、OP 一慢（尤其调 LLM 打分的 OP），就想并行。手写并行的路数是这样：

```python
with ThreadPoolExecutor(8) as pool:
    futures = [pool.submit(process, chunk) for chunk in chunks]
    results = [f.result() for f in futures]
```

独立任务这样能跑，但三个问题马上冒出来：

- **依赖要手搓**：结果之间有先后（先打分、再过滤、最后汇总）时，等待与排序全得自己写；
- **数据要搬来搬去**：一个大对象（embedding 矩阵、索引）被多个任务用到，传值 = 复制 N 份；
- **代码被线程细节淹没**：pool / queue / join 把「算什么」埋进了「怎么跑」。

Ray 的回答是给三个一等公民抽象：**remote function（任务）、object store（对象存储）、
dynamic task graph（动态任务图）**——你只描述「算什么、谁依赖谁」，并行调度变成系统的事。
本节把这三件事从零做出来。

---

## 2. 先跑起来

文件：`L0_task_graph.py`，纯标准库，CPU 即跑。

```bash
$ python3 L0_task_graph.py
```

真实输出（其中一次运行）：

```text
================================================================
nano-ray L0 — task graph / object store / 调度
================================================================

[1] 8 个独立任务（各模拟 0.15s IO），4 workers
    serial   =  1.22s | parallel =  0.31s | speedup = 3.97x
    结果与串行逐位一致 ✅（submit 立即返回，get 才阻塞）

[2] map-reduce 任务图：8 map + 7 reduce = 15 个节点，全部自动调度
    total = 1322800（与串行一致 ✅），完成顺序 oids = [9, 10, 11, 12, 14, 17, 18, 21, 13, 19, 15, 16, 20, 22, 23]
    每个 reduce 都在其两个 map 之后完成 ✅（依赖由调度器保证，非手工排序）

[3] object store：put 一次 200,000 元素，4 个任务按 ref 取用
    identity 检查：4 个任务拿到的是同一个对象（id=4392167040）✅
    传值语义要拷 4 份 = 800,000 元素；store 只存 1 份 + 传 4 个句柄
    → 真实 Ray 里这对应 plasma 共享内存：跨进程也不拷贝 [TODO: verify source]

[4] 反例：1000 个 µs 级小任务，8 workers
    serial =   0.03ms | parallel =  12.34ms | speedup = 0.00x
    对照 [1] 的 3.97x：任务太细时，submit/队列/线程唤醒的开销
    比任务本身还贵——remote 化只值得花在「单个足够重」的任务上

[5] 触发时序回归：300 个随机 DAG + 动态提交，全部任务完成 ✅
    反例（修复前顺序）：窗口内提交的 B 永远不被触发，成为孤儿 ❌（确定性复现）
    同一时机，修复后顺序：B 被放行后的快照捕获，正常完成 ✅（submit ⇒ 必执行）

================================================================
✅ self-check passed: 并行等价 / 依赖顺序正确 / 零拷贝 identity / 粒度反例成立
                      + 触发时序回归通过（submit ⇒ 必执行；旧顺序确定性复现丢触发）
================================================================

takeaway: Ray 的编程模型 = remote 函数（future）+ object store（传引用）
          + 动态任务图（完成即触发）。L1 用真 Ray 把 data-juicer pipeline 跑起来。
```

**L0 基线指标（toy metric）**：独立任务 speedup 约 `3.8x–4.0x`（8 任务 / 4 worker，理想 4x，
多次运行依机器负载在 3.8x–4.0x 间波动）；15 节点 map-reduce 图全自动调度且结果与串行
一致（total=1322800）；object store 让 4 个任务共享同一份 200k 元素对象，省掉 3 份拷贝；
细粒度反例 speedup ≈ 0；触发时序回归 300 个随机 DAG 全绿（submit ⇒ 必执行）。
注意 `[2]` 的完成 oid 顺序**每次运行都不同**（兄弟节点并行、先后不定），
但「reduce 一定在它的两个 map 之后」永远成立——这正是调度器该保证的不变量。

> **toy 口径声明**：`[1]` 用 `time.sleep(0.15)` **显式模拟**数据分片读取 / API 调用等
> IO 型耗时（sleep 会释放 GIL，线程并行真实成立），不是在演示 CPU 计算加速。
> 这与 nano-verl L0 用模拟时间的口径相同。L1 会接触真实 Ray：worker 是进程级的，
> 根本不存在 GIL 问题。

---

## 3. 机制一：remote function 与 future——submit 立即返回

```python
def submit(self, fn, *args) -> ObjectRef:
    """remote 调用：立即返回未来结果的 ObjectRef（不阻塞、不执行）。"""
    task = Task(fn, args, ObjectRef())
    with self.lock:
        self.tasks.append(task)
    self._try_start(task)
    return task.out
```

`submit(fn, *args)` **不执行 fn**——它把任务登记进图里，立刻返回一个结果的占位符
`ObjectRef`（future）。`ref.get()` 才阻塞等结果。

心智模型由此改变：串行代码是「调用 → 等 → 拿结果 → 下一步」；
Ray 式代码是「**先把整张图描述完 → 最后再取**」。实验 `[1]` 的循环里，
8 个 submit 本身都在微秒级返回，真正的计算已经在 worker 里并发跑起来了。

`ObjectRef` 同时是依赖的载体：任何任务的参数里出现 `ObjectRef`，
就自动成为它的依赖（见 `Task.__init__` 里的 `self.deps`）——**依赖关系藏在数据流里，
不需要单独声明**。这是后面动态调度的基础。

---

## 4. 机制二：object store——传引用，不传值

```python
big = list(range(200_000))                     # ~200k 元素的"大"对象
big_ref = sched2.put(big)                      # put 一次
ids = [sched2.submit(lambda x: id(x), big_ref) for _ in range(4)]
# 4 个任务的 id(x) == id(big) ✅ —— 拿到的是同一个对象
```

传值语义下，把 `big` 交给每个任务都要复制一份——4 个任务 = 80 万元素的拷贝。
object store 的做法：数据**放进 store 一次**，任务只拿句柄（ObjectRef），
执行时由调度器把句柄解析成 store 里的对象。实验 `[3]` 的 identity 检查
（`id(x) == id(big)`）证明了单进程内的零拷贝。

为什么这在真实分布式系统里重要：跨机器搬数据的代价是网络与内存。
Ray 的 object store（plasma）把对象放在**共享内存**里，同机器的不同 worker
进程访问的是同一块物理内存——跨进程也零拷贝 `[TODO: verify source，L3 补源码路径]`。
这就是「把大 DataFrame 传给 10 个任务不会复制 10 份」的答案：
DataFrame 在 store 里只有一份，10 个任务拿的是 10 个句柄。

---

## 5. 机制三：动态任务图——完成即触发下游

```python
def _run(self, task):
    vals = [a.get() if isinstance(a, ObjectRef) else a for a in task.args]
    result = task.fn(*vals)
    with self.lock:                        # 先记账再放行：finished 顺序严格尊重依赖
        self.finished.append(task.out.oid)
    task.out._set(result)
    # 触发时序不变量：pending 快照必须在 _set 之后取（为什么？见下一小节）
    with self.lock:
        pending = [t for t in self.tasks if not t.started]
    for t in pending:                      # 完成即触发下游——动态图的关键
        self._try_start(t)
```

两个要点：

1. **依赖就绪才开始**：`_try_start` 检查 `all(d.ready() for d in task.deps)`。
   worker 线程从不阻塞在依赖上——等待发生在调度逻辑里，不占用 worker。
2. **完成即触发**：任何一个任务完成，都会去扫描并触发新近就绪的下游任务。
   图**不需要预先声明完整**——跑完一批 map、根据结果再 submit reduce，完全合法；
   图是运行时动态生长的。这正是 Ray 所称的 dynamic task graph：
   RL rollout、agentic workflow、递归数据处理里，下一批任务往往要等上一批
   跑完才知道长什么样，静态 DAG 表达不了——这是 Ray 论文（arXiv:1712.05889）
   面向 emerging AI workloads 的核心动机之一。

盯着 `[2]` 的输出看：8 map + 7 reduce 共 15 个节点，没人手写执行顺序，
「每个 reduce 在其两个 map 之后」的断言却恒成立；而 4 个 map 的 oid 次序每次都变——
**兄弟非确定、依赖恒有序**，这是调度器的正确形态，不是 bug。

对照静态 DAG 系统（Airflow 式：先声明全图再执行）：静态图适合稳定的批处理；
动态图适合任务数量依赖运行时数据的场景（「读完文件才知道切几个分片」）。
Ray 的底座是后者。

### 5.1 触发时序：「完成即触发」的事件顺序本身就是机制

上面的 `_run` 看似平平无奇，但**事件顺序差一步，正确性就没了**。这个 toy 的第一版
把 pending 快照取在 `_set` **之前**（与 `finished.append` 同一个锁块里）：

```python
with self.lock:
    self.finished.append(task.out.oid)
    pending = [t for t in self.tasks if not t.started]   # ← 快照过早
task.out._set(result)                                    # ← 竞态窗口：快照 → 放行
for t in pending:
    self._try_start(t)
```

后果：若依赖本任务的新任务 T 恰在「快照之后、`_set` 之前」提交——T 自己的
`_try_start` 见依赖未就绪而返回，而触发循环拿的又是**不含 T 的旧快照**——
T 的触发被永久丢失。T 没有下游就静默成孤儿（`submit ⇒ 必执行` 被破坏），
有下游就级联挂死在 `get()` 上。实验 `[5b]` 用拉宽的窗口**确定性复现**了它。

修复只动一处：**把快照移到 `_set` 之后**。此刻两种交错都被覆盖——
此前提交的依赖任务，要么提交时已见依赖就绪而自行启动，要么落进这份
放行后的快照被触发（`submit` 内部是「先入 `tasks` 再查就绪」，顺序同样关键）。
实验 `[5a]` 用 300 个随机 DAG + 图中途动态提交做回归：修复后全绿，
同样的负载在旧顺序下会偶发丢触发。

教训超出这个 toy：真实系统里「事件顺序与通知」同样是机制的一部分——
Ray 的 ownership 协议要处理的正是同一类问题（谁负责通知、通知会不会丢、
乱序到达怎么办）`[TODO: verify source]`。写调度器时，先问清楚每个事件的
**记账、放行、快照、触发**四步的顺序，再写代码。

---

## 6. 与真实 Ray 的对应（概念层）

| nano 实现 | Ray 对应 | 说明 |
|-----------|---------|------|
| `ToyScheduler.submit` | `@ray.remote` 装饰的函数调用 | 调用立即返回 ObjectRef |
| `ObjectRef` / `ref.get()` | `ray.ObjectRef` / `ray.get` | future 语义一致 |
| `ToyScheduler.put` | `ray.put` | 对象入 store，返回句柄 |
| `ThreadPoolExecutor` worker | Ray worker 进程 | toy 用线程 + sleep 模拟 IO；真 Ray 用进程（GIL / 隔离 / 资源核算） |
| 完成触发 `_try_start` | 分布式调度（ownership-based） | toy 是单进程扫描；真 Ray 由 owner 跟踪依赖、分布式实现 `[TODO: verify source]` |
| 进程内 identity 零拷贝 | plasma 共享内存零拷贝 | toy 验证语义；真实实现要共享内存 `[TODO: verify source]` |

L0 只到概念层；源码级对照（ownership 协议、plasma、GCS）留 L3 补齐。

---

## 7. 费曼：讲给外行听

**类比：外卖平台。**

- **submit = 下单**：你下单，平台立刻给你一个订单号（ObjectRef），厨房不会当场把菜递给你。
  你可以继续下别的单（submit 其它任务），真要吃的时候才凭订单号取餐（get）；
- **object store = 共享原料仓**：餐厅不为每个订单单独采购一份原料；
  常用原料放在共享仓里（put 一次），每个订单只记「凭编号取用」（传引用）——
  10 个订单引用同一锅汤底，不会熬 10 锅；
- **动态任务图 = 派单系统**：接单即开始做、做好即打包、打包好即派骑手——
  每个环节由上一环节的完成自动触发，没有人事先拿着一张全局流程表指挥；
  某个订单缺料（依赖另一个订单的半成品）就自动等它就绪。

一句话版本：**Ray 把「怎么并行」从代码细节变成系统的调度问题；你只写「算什么、谁依赖谁」。**

---

## 8. 思考题

1. 真实 Ray 的 worker 为什么用**进程**，而 toy 用线程？
   （提示：Python GIL 让线程无法并行 CPU 计算；进程隔离让一个任务崩溃不带走别人；
   资源（CPU/GPU/内存）按进程核算与分配。toy 用线程只是为了 L0 零依赖即跑，
   sleep 型负载恰好释放 GIL，才让并行演示成立。）

2. 实验 `[4]` 里细粒度任务的 speedup ≈ 0。实践中怎么定任务粒度？
   （提示：单任务实际工作量要远大于调度/入队/唤醒开销，真实系统还有序列化与网络成本；
   经验口径是「单任务至少毫秒级、最好更大」——这是定性经验，具体阈值依集群与负载而变
   `[TODO: verify]`。）

3. 如果某个任务中途失败了，整张图会怎样？
   （提示：重试、按血缘重算、故障隔离——真实 Ray 有 task/actor 的监管与重建机制，
   超出 L0 范围，L2 引入 actor 时再碰 `[TODO: verify source]`。）

---

## 9. 反例：「什么都 remote 化、越细越好」——错

实验 `[4]` 就是直接打脸：1000 个微秒级任务，串行 0.05ms，并行反而花了 22ms——
submit、锁、线程池队列、唤醒的固定开销完全淹没了工作量本身。再补两个常见误区：

- **「并行了就一定快」**：不。并行的收益要先付调度与通信成本，
  任务成本摊不平它时，并行反而更慢（对照 `[1]` 的 3.95x 与 `[4]` 的 0.00x）。
- **「object store 是万能的」**：store 里的对象是**不可变**的——
  只有不可变，才能安全地被多任务/多机共享。要修改共享状态，那是 actor 的活
  （L2 的主题），不是把可变对象塞进 store。

---

## 10. 下一步 L1

L1 把这个 toy 搬上真实 Ray：

1. `pip install ray`，把 nano-data-juicer L1 的 OP pipeline 包成 `@ray.remote`——
   OP 接口仍是 `list[Sample] → list[Sample]`，只是每个 OP 变成 task、数据集变成 ObjectRef；
2. 在真实数据上测吞吐提升，对照本节 `[1]` 的账本；
3. 看 Data-Juicer 官方怎么做同一件事：`data_juicer/core/executor/ray_executor.py`
   （本地 checkout 实测存在，2026-08-04）。

---

## 11. 溯源

- 运行输出来自本机真实执行：`python3 L0_task_graph.py`。
  speedup 多次运行有 ±百分之几的波动（约 3.8x–4.0x，依机器负载而异），
  `total=1322800`、identity 检查与全部 self-check 确定不变。
- `time.sleep` 作为 IO 模拟为**显式声明的 toy 口径**（见 §2）；sleep 释放 GIL 是
  CPython 的既有行为，因此线程并行在本节真实成立。
- Ray 论文：arXiv:1712.05889（Moritz et al.，*Ray: A Distributed Framework for
  Emerging AI Applications*）；摘要原文称要为下一代 AI 应用提供
  "single dynamic execution engine"，dynamic task graph 是本教程对该动机的转述。
- plasma / ownership 调度器 / GCS 的源码级对应均标 `[TODO: verify source]`，L3 补齐；
  仓库：<https://github.com/ray-project/ray>。
- Data-Juicer 的 Ray executor 路径 `data_juicer/core/executor/ray_executor.py`
  于本地 checkout `${DATA_JUICER_REPO}` 实测存在（2026-08-04）。
