# nano-data-orchestration L0 — DAG 调度器：任务状态机 + 调和循环 + 失败语义（纯 Python 本质模拟）

> **前置**：无（先读 nano-data-platform L0 会更容易代入 fixture，但不必须）。Python 3.10+，纯标准库，CPU 秒级。
> **运行**：`python3 L0_dag_scheduler_state_machine.py`（任意目录可跑，输出确定，复跑逐字节一致）。
> **本文件是 notebook-style 教程**：叙述 + 代码摘录 + 真实运行输出 + 思考题交替推进。

---

## §1 为什么编排是数据飞轮的「神经系统」

nano-data-platform 回答了数据**住在哪、谁能碰、花了多少钱、训练用的是哪一版**。但飞轮不是跑一次就完，而是**持续地跑**：接入 → 质检 → 构建 → 训练 → 评估 → 部署 → agent 轨迹回流再接入（ROADMAP §一 的 RSI 闭环）。每一步都可能失败——网络抖动、源系统宕机、权限过期、代码 bug——「持续跑」因此变成一个硬问题：**失败发生时，系统替谁做什么决定？**重试？止损？把失败传播给下游？还是拒绝执行？

编排器（orchestrator）就是回答这组问题的组件。如果说数据平台是飞轮的器官（存储与供给），编排就是神经系统：什么时候动哪块肌肉、疼了怎么缩手。清洗算子写得再好（nano-data-juicer），如果「何时执行、失败怎么办」没有机制保障，管线要么停摆，要么——更糟——**在坏数据上继续跑**。

本模块抓的核心机制链条（ROADMAP §七）：

```
工作流编排、依赖调度、失败重试、自动化测试/部署、Agent 驱动的管线自愈
```

L0 裸出前三项的本质（DAG 调度的状态机内核）；CI/CD 门与 Agentic 自愈在 L1/L2 展开。

先看三大编排器对自己的定位（官方 repo 页面标题逐字，2026-08-12 抓取）：

- **Airflow**："A platform to programmatically author, schedule, and monitor workflows"（author / schedule / monitor 三动词）
- **Dagster**："An orchestration platform for the development, production, and observation of data assets"（observation 被放进定义）
- **Prefect**："Prefect is a workflow orchestration framework for building resilient data pipelines in Python."（resilient 被放进定义）

三家不约而同把「调度 + 可观测 + 韧性」写进一句话定位——没有一家说自己是「脚本运行器」。L0 要裸出的正是这句话里的机制。

## §2 L0 模拟真实系统的哪四面

L0 的验收标准是「能口头讲清它在模拟真实系统的哪一面」。本实现模拟四面，刻意不模拟其余（§9 列边界）：

| # | 机制面 | nano 实现 | 真实系统对应 |
|---|--------|-----------|--------------|
| [1] | DAG = 一等公民的依赖结构：环 / 未知依赖执行前被拒 | `topo_validate` | Airflow/Dagster 的 DAG 静态校验（fail fast） |
| [2] | 任务状态机 + 调和循环：每 tick 扫描状态、施加转移规则 | `run()` 规则 A/B/C | Airflow TaskInstance 状态机 + scheduler 周期扫描 |
| [3] | 失败语义：transient/permanent 分类 → 有界重试 vs 立即失败；上游失败向下游传播 | `TransientError`/`PermanentError` + `UPSTREAM_FAILED` | Airflow 的 `up_for_retry` / `upstream_failed` 状态 |
| [4] | 治理 first-class：capability default-deny + attempt 成本账本 | `needs`/`grants` + `cost_report` | IAM least-privilege + 成本可观测 |

先跑一遍，建立全局印象（完整输出；以下各节的输出块均从此同一次运行中截取）：

```bash
$ python3 L0_dag_scheduler_state_machine.py
== nano-data-orchestration L0: DAG 调度器——状态机 + 调和循环 + 失败语义（纯 Python 本质模拟）==
...
self-check: 15/15 PASS
```

demo 的剧本：一条 CI/CD 风格的数据管线（9 个任务，动作序列呼应 nano-data-platform L0：ingest → gate → build → deploy/publish），从 t=0 开始调和——两个源一个抖动（重试救回）、一个彻底坏掉（止损 + 下游锥传播），一个发布任务因未授权被拒（0 成本），最终 5 成功 / 2 失败 / 2 上游失败，成本账本结账。

> **fixture 声明**：PIPELINE 是内嵌的演示管线，**失败日程是显式实验设计**（同 nano-data-platform 刻意埋缺陷的口径）：`flaky_source` 只败第 1 次（transient 原型）；`broken_source` 永远败、且被作者**故意误分类**为 transient（演示误分类的代价与重试上限的止损作用）；`publish_report` 故意不授权 `metrics_write`（演示 default-deny）。这不是「假数据冒充跑通」——L0 的机制对象就是「调度器在失败面前做什么决定」，失败本身是实验设计的一部分。L1 会换真实 subprocess 任务复现同一套状态机语义。

---

## §3 机制面 [1]：DAG = 一等公民的依赖结构

```python
# ---- [1] DAG 校验：依赖是一等公民，环 / 未知依赖必须死在执行前 ----
def topo_validate(tasks):
    for n, spec in sorted(tasks.items()):
        for d in spec["deps"]:
            if d not in tasks:
                raise ValueError(f"unknown dep '{d}' of '{n}' —— 拼写错误就是生产事故，校验期拒绝")
    indeg = {n: len(spec["deps"]) for n, spec in tasks.items()}
    order, ready = [], sorted(n for n, k in indeg.items() if k == 0)
    while ready:
        n = ready.pop(0); order.append(n)
        for m in sorted(tasks):
            if n in tasks[m]["deps"]:
                indeg[m] -= 1
                if indeg[m] == 0: ready.append(m)
        ready.sort()
    if len(order) != len(tasks):
        raise ValueError(f"cycle detected: {sorted(set(tasks) - set(order))} —— 环上的任务永不收敛，校验期拒绝")
    return order
```

```text
[1] fail fast：环与未知依赖在执行前被拒（校验成本 0；运行时拒绝成本 = 整次运行）
  rejected as expected: cycle detected: ['a', 'b', 'c'] —— 环上的任务永不收敛，校验期拒绝
  [check 01] PASS  环必须在执行前被拒
  rejected as expected: unknown dep 'ghost' of 'a' —— 拼写错误就是生产事故，校验期拒绝
  [check 02] PASS  未知依赖必须在执行前被拒
```

**为什么结构错误必须死在执行前？** 三笔账：

1. **成本账**：校验是纯静态的（扫依赖表 + Kahn 拓扑排序），成本相对一次真实运行约等于 0；而运行时才撞上环或拼错的依赖名，代价是**整次运行**——部分任务可能已经产生副作用（写了数据、花了钱、发了通知）。
2. **语义账**：环上的任务**永不收敛**——a 等 c、c 等 b、b 等 a，没有任何一个能先跑。如果校验期不拒，调度器只能在运行时死锁（§4 的死锁守卫只是兜底，不该被到达）。
3. **事故账**：依赖名是字符串，字符串会拼错。`unknown dep 'ghost'` 在生产里对应「上游任务改名了，下游配置没跟上」——这类事故的正确拦截点就是校验期，而不是等运行到那个任务时报「找不到数据」。

这正是 DAG 作为**声明式结构**（而非命令式调用序列）的红利：结构是静态可分析的，结构错误就能静态拦截。真实系统里 Airflow/Dagster 都要求 DAG 定义可被解析期检查，机制动机相同（合理推断，见 §11）。

**思考题 3.1**：`topo_validate` 为什么每一步都对 ready 集合 `sort()`？（参考方向：拓扑序不唯一，不排序则同一 DAG 多次运行可能得到不同顺序，输出不可复现；排名字序是 nano 的确定性策略。代价是放弃了优先级 / 资源感知等调度策略自由度——那是 L2 的课题，但「同输入同调度」始终是审计能力的前提。）

---

## §4 机制面 [2]：状态机 + 调和循环 —— 调度器唯一的活

调度器不是「按顺序跑脚本的进程」。本实现里它只做一件事：**每个 tick 扫描全部任务的状态，施加转移规则**。状态集合：

```
PENDING ─(规则B: 全上游 SUCCESS)─► RUNNABLE ─(规则C: 执行)─► RUNNING ─ok─► SUCCESS
   │                                   ▲                        │
   └─(规则B: 任一上游失败)─► UPSTREAM_FAILED                     ├─ permanent/deny ─► FAILED
                                       │                        ├─ transient 且未耗尽 ─► RETRYING ─┐
                              (规则A: 退避到期)◄────────────────┴─ transient 且耗尽 ─► FAILED      │
                                       └──────────────────────────────────────────────────────────┘
```

```python
# ---- [2][3][4] 状态机 + 调和循环：调度器唯一的活 = 每 tick 扫描状态、施加转移规则 ----
TERMINAL = {"SUCCESS", "FAILED", "UPSTREAM_FAILED"}

def run(tasks, grants, log=True):
    topo_validate(tasks)
    state = {n: "PENDING" for n in tasks}
    attempts, retry_at, retry_waits = {n: 0 for n in tasks}, {}, {}
    exec_order, events = [], []
    tick = 0
    while True:
        if all(state[n] in TERMINAL for n in tasks): break
        did = 0
        for n in sorted(tasks):                                # 规则 A：RETRYING 唤醒（退避到期）
            if state[n] == "RETRYING" and retry_at[n] <= tick:
                state[n] = "RUNNABLE"; did += 1
        for n in sorted(tasks):                                # 规则 B：依赖解析——全 success 才可跑；任一失败立即传播
            if state[n] != "PENDING": continue
            ds = [state[d] for d in tasks[n]["deps"]]
            if any(s in ("FAILED", "UPSTREAM_FAILED") for s in ds):
                state[n] = "UPSTREAM_FAILED"; did += 1
                events.append((tick, n, "UPSTREAM_FAILED", "上游终态失败——依赖是承诺：不在坏数据上跑"))
            elif all(s == "SUCCESS" for s in ds):
                state[n] = "RUNNABLE"; did += 1
```

跑一条带两种故障的管线，事件按 tick 回放：

```text
[2] 调和循环跑一条带两种故障的数据管线（tick = 逻辑时钟，事件按 tick 回放）
  [t=0] ingest_crm     -> RETRYING        attempt 1 transient: connection reset —— transient —— +1 tick 后重试
  [t=0] ingest_web     -> RETRYING        attempt 1 transient: source 拒绝所有读取——看似 transient 实则 permanent（误分类教训） —— +1 tick 后重试
  [t=0] unit_tests     -> SUCCESS         attempt 1
  [t=1] ingest_crm     -> SUCCESS         attempt 2  <- 重试救回
  [t=1] ingest_web     -> RETRYING        attempt 2 transient: source 拒绝所有读取——看似 transient 实则 permanent（误分类教训） —— +2 tick 后重试
  [t=2] gate_crm       -> SUCCESS         attempt 1
  [t=3] build_curated  -> SUCCESS         attempt 1
  [t=3] ingest_web     -> FAILED          重试上限 (2) 耗尽，止损——上限是误分类的最后防线
  [t=4] gate_web       -> UPSTREAM_FAILED 上游终态失败——依赖是承诺：不在坏数据上跑
  [t=4] normalize_web  -> UPSTREAM_FAILED 上游终态失败——依赖是承诺：不在坏数据上跑
  [t=4] deploy         -> SUCCESS         attempt 1
  [t=4] publish_report -> FAILED(deny)    capability missing (default-deny): ['metrics_write'] —— permanent 不重试，0 attempt 0 成本
```

```text
[3] 终态向量: SUCCESS=['build_curated', 'deploy', 'gate_crm', 'ingest_crm', 'unit_tests']
    FAILED=['ingest_web', 'publish_report']  UPSTREAM_FAILED=['gate_web', 'normalize_web']
  [check 03] PASS  5 SUCCESS（含重试救回的 ingest_crm）
  [check 04] PASS  2 FAILED（ingest_web 止损 / publish_report 被拒）
  [check 05] PASS  2 UPSTREAM_FAILED（坏源的爆炸半径）
```

**为什么是「扫描状态 + 施加规则」，而不是「写一段控制流按序执行」？** 这是编排器设计里最深的分岔：

**（a）状态是完整记录。** 控制流式调度器的「进度」活在调用栈里——进程一死，进度蒸发，只能从头重跑。状态机式调度器的进度活在 `state` 这个显式记录里：崩溃恢复（从 state 继续调和）、可观测性（每个事件是 `(tick, task, 新状态, 为什么)` 四元组，上面 [t=0]–[t=4] 的回放就是它）、重试（RETRYING 只是另一个状态，不需要特殊控制流）——全部从状态长出来。Airflow 的 TaskInstance 状态表（官方 tasks 文档，2026-08-12 抓取）就是这套语义的工业形态，几个条目逐字：

> "success: The task finished running without errors" …… "failed: The task had an error during execution and failed to run"
> "up_for_retry: The task failed, but has retry attempts left and will be rescheduled."
> "upstream_failed: An upstream task failed and the Trigger Rule says we needed it"
> —— https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html

nano 的 RETRYING/UPSTREAM_FAILED 与 Airflow 的 `up_for_retry`/`upstream_failed` 一一对应——状态名不同，机制同构。

**（b）调和是幂等的。** 同样的状态套同样的规则，得到同样的转移。于是「重启进程再调和一遍」是安全的——L1 的崩溃续跑就是「状态落盘 + 重启后继续调和」，机制在 L0 已经就位。

**（c）RUNNING 在 L0 里一闪而过。** 注意事件流里没有 RUNNING 事件：L0 的任务是纯函数，同步执行、瞬时完成。真实任务会跑几分钟几小时，RUNNING 是长时间停留的状态——于是「RUNNING 还活着吗」本身成了状态问题。Airflow 为此有 heartbeat / zombie task 机制（同一文档页逐字）：

> "Airflow will find these periodically, clean them up, and mark the TaskInstance as failed or retry it if it has available retries."

调度器周期扫描、清理僵死、按规则改状态——和 nano 的 while 循环是同一个内核。nano 把 RUNNING 压缩成瞬时，是为了让状态机主干（而非 liveness 检测）当主角；这一面在 §9 记为 L1/L2 课题。

**确定性声明**：tick 是逻辑时钟（不是 wall-clock），同一 tick 内就绪的任务按名字序执行——于是全部输出可复现（§8 的 digest 与双跑锚点）。真实系统用 wall-clock + 并发，但「状态决定下一步发生什么」这条不变。

**思考题 4.1**：如果调度器进程在 t=3 时突然被 kill（L0 的 state 在内存里），丢了什么？真实系统把什么落盘，才能在重启后回答「我们进行到哪一步了」？（参考方向：L0 全丢，只能整条重跑；真实系统持久化 TaskInstance 状态 + 事件日志到数据库，重启 = 从状态继续调和——即 L1 的崩溃续跑。zombie 引文说明 Airflow 连「RUNNING 但实际已死」都要靠状态记录兜底。）

---

## §5 机制面 [3a]：错误分类 → 重试策略

```python
class TransientError(Exception): pass   # 可重试：网络抖动 / 对端重启 ——「重试可能成功」
class PermanentError(Exception): pass   # 不可重试：权限被拒 / 校验失败 ——「重试 N 次同果」，分类是任务作者责任
```

```python
        for n in sorted(tasks):                                # 规则 C：执行（L0 串行；并行 / 资源池 → L1/L2）
            if state[n] != "RUNNABLE": continue
            spec = tasks[n]; did += 1
            try:
                missing = sorted(spec.get("needs", set()) - grants.get(n, set()))
                if missing: raise PermanentError(f"capability missing (default-deny): {missing}")
                attempts[n] += 1; state[n] = "RUNNING"
                if n not in exec_order: exec_order.append(n)
                spec["fn"](attempts[n])
                state[n] = "SUCCESS"
                events.append((tick, n, "SUCCESS", f"attempt {attempts[n]}" + ("  <- 重试救回" if attempts[n] > 1 else "")))
            except PermanentError as e:                        # [4a] permanent：0 重试；拒绝发生在计算之前
                state[n] = "FAILED"
                events.append((tick, n, "FAILED(deny)", f"{e} —— permanent 不重试，0 attempt 0 成本"))
            except TransientError as e:                        # [3] transient：有界重试 + 指数退避
                if attempts[n] <= spec["retries"]:
                    wait = 2 ** (attempts[n] - 1)              # 退避 = 2^(k-1)：给故障源恢复时间，确定性可复算
                    state[n] = "RETRYING"; retry_at[n] = tick + wait
                    retry_waits.setdefault(n, []).append(wait)
                    events.append((tick, n, "RETRYING", f"attempt {attempts[n]} transient: {e} —— +{wait} tick 后重试"))
                else:
                    state[n] = "FAILED"
                    events.append((tick, n, "FAILED", f"重试上限 ({spec['retries']}) 耗尽，止损——上限是误分类的最后防线"))
```

```text
[4] 错误分类 -> 重试策略：transient 指数退避有界重试，permanent 立即失败
  ingest_crm:     attempts=2 backoff=[1] -> 重试救回
  ingest_web:     attempts=3 backoff=[1, 2] -> 上限耗尽止损（误分类的代价）
  publish_report: attempts=0 -> 拒绝发生在计算之前：permanent 0 重试 0 成本
  [check 06] PASS  transient 被救回: ingest_crm 第 2 次成功
  [check 07] PASS  指数退避 = 2^(k-1): [1] 与 [1, 2]
  [check 08] PASS  permanent 永不重试: publish_report attempts == 0
  [check 09] PASS  重试上限 = 止损线: ingest_web attempts == retries+1 == 3
```

三个本质点：

**（a）分类是任务作者的责任，调度器只执行分类的后果。** 调度器看不见任务内部——它不知道「连接被重置」是对端重启（等一下就好）还是防火墙永久封禁（等多久都白搭）。只有任务作者知道「重试是否可能成功」，于是分类以异常类型声明（`TransientError`/`PermanentError`），调度器按类型施加策略。Airflow 的 retry policy 官方示例（同一 tasks 文档页）用的是同一个思想：按异常类型 / HTTP 状态码分类——官方示例代码注释逐字 "# server error -- worth retrying"（5xx 重试）与 "# client error -- not retryable"（4xx 不重试）；三个动作里 FAIL 的语义逐字为 "fail immediately, skipping any remaining retries"，且重试永远受上限约束——"a policy can fail earlier but cannot extend past the configured maximum"。nano 的 `retries` 上限 + 两类异常，是这套语义的最小核。Airflow 还原生提供 `retry_exponential_backoff` 参数（同页在案）——指数退避不是 nano 的发明，是工业标配。

**（b）指数退避 = 2^(k-1)：给故障源恢复的时间。** 第 1 次失败后等 1 tick，第 2 次后等 2 tick（check 07：`[1]` 与 `[1, 2]`）。立刻重试大概率撞上同一个故障窗口；退避让「对端重启」「限流解除」这类 transient 故障有时间自愈。选 2^(k-1) 还因为它确定性可复算——任何时候都能从重放中算出每次重试该在哪个 tick 醒来。

**（c）重试上限 = 止损线，更是误分类的最后防线。** `broken_source` 是故意埋的误分类：它永远失败，却被作者标成 transient。调度器忠实地重试——烧掉 3 次 attempt（backoff `[1, 2]`），然后在 `retries=2` 耗尽时止损（check 09：`attempts == retries+1 == 3`）。没有上限，这条管线永远不会到达终态（思考题 5.1）。**误分类不可避免——上限让误分类的代价有界。**

**思考题 5.1**：为什么不能把重试上限设得「尽量大」？（参考方向：其一，成本——每次 attempt 烧钱，§7 的账本会看到 3 coins 全浪费；其二，副作用放大——非幂等任务重试会重复副作用，幂等是重试的隐含前提，L1 的真实 subprocess 任务会正面撞上；其三，收敛性——上限是「全部到达终态」的保证，没有它管线可能无限运行。）

---

## §6 机制面 [3b]：失败传播 = 下游锥

```python
def downstream_cone(tasks, root):  # 爆炸半径：root 失败后，哪些任务随之失去存在意义
    cone, stack = set(), [root]
    while stack:
        n = stack.pop()
        for m, spec in tasks.items():
            if n in spec["deps"] and m not in cone:
                cone.add(m); stack.append(m)
    return cone - {root}
```

```text
[5] 失败传播：爆炸半径 = 下游锥；重试救回的任务保住它的整个下游
  若无重试，ingest_crm 失败将阻塞 ['build_curated', 'deploy', 'gate_crm', 'publish_report']（4 个任务）
  ingest_web 失败实际阻塞 ['gate_web', 'normalize_web']（2 个任务）——全部 0 attempt，不在坏数据上浪费计算
  [check 10] PASS  重试保住整个下游锥（4 任务获得执行机会）
  [check 11] PASS  坏源恰好阻塞其下游锥且 0 attempt，不波及其他分支
  [check 12] PASS  拓扑不变式：每个跑过的任务都在全部上游之后
```

**（a）急切传播：不在坏数据上跑，且 0 成本。** `ingest_web` 在 t=3 确认 FAILED 后，`gate_web` 与 `normalize_web` 在**同一个 tick**（t=4）被标 UPSTREAM_FAILED——不是「等轮到它们执行才发现没数据」。依赖是承诺：上游给的是坏数据（或没有数据），下游的执行就**失去存在意义**。注意代价面：锥内两个任务 `attempts == 0`（check 11）——传播发生在调度层，**没有发生任何计算**。UPSTREAM_FAILED 因此不是「失败的惩罚」，而是「免于浪费」：Airflow 状态表里这个条目的定义（§4 已引）说的正是「上游失败且 Trigger Rule 声明我需要它」。nano L0 把依赖承诺硬编码为最严格形态（全部上游 SUCCESS 才可跑）；Airflow 用 trigger_rule 把这一形态做成显式配置——状态表条目里 "the Trigger Rule says we needed it" 即此机制。

**（b）对称的事实：重试救回的不是一个任务，是整个下游锥。** check 10 的对照实验：如果 `ingest_crm` 没有重试（第 1 次失败即终态），它的下游锥 `{gate_crm, build_curated, deploy, publish_report}` 共 4 个任务全部失去执行机会——连 unit_tests 都救不了 build_curated（它还需要 gate_crm）。**重试的价值 = 被救任务下游锥的大小**，这是「为什么值得为 transient 故障付重试成本」的定量答案。

**（c）拓扑不变式（check 12）**：每个真正执行过的任务，都在它全部上游之后执行（exec_order 上验证）。调度器可以决定「何时跑」，但不能违反「在谁之后跑」——DAG 边是硬约束，调度策略是软自由度。

**思考题 6.1**：`normalize_web` 是 `gate_web` 的下游、`gate_web` 是 `ingest_web` 的下游——为什么 UPSTREAM_FAILED 能在同一个 tick 跨两级传播（t=4 两者同时落定）？如果把规则 B 的扫描顺序反过来会怎样？（参考方向：规则 B 按名字序扫描，gate_web 先于 normalize_web 被判定，后者同轮就能看到前者的新状态——传播规则在一轮调和内施加到不动点。扫描顺序反过来，normalize_web 要等下一 tick 才传播，但**终态不变**：终态与扫描顺序无关，过程与扫描顺序有关。这是调和循环的合流性质，也是「重启后继续调和」安全的原因之一。）

---

## §7 机制面 [4]：治理 first-class —— default-deny + 成本账本

ROADMAP §七 的硬性写作原则：**安全与成本不是附录，是机制的一部分**。L0 用两段兑现。

**（a）capability default-deny：拒绝发生在计算之前。** 规则 C 的第一行不是执行任务，是查权限（上面代码摘录的前两行）：任务声明 `needs`（需要的能力），调度器对照 `grants`（授权表），缺能力直接抛 `PermanentError`——注意它发生在 `attempts[n] += 1` **之前**，于是被拒任务 `attempts == 0`（check 08）。

```python
GRANTS = {"deploy": {"prod_deploy"}}   # 最小权限：只有 deploy 持 prod 能力；metrics_write 无人授权 -> 拒绝演练
```

```text
  [t=4] publish_report -> FAILED(deny)    capability missing (default-deny): ['metrics_write'] —— permanent 不重试，0 attempt 0 成本
```

授权表体现最小权限：只有 `deploy` 持有 `prod_deploy` 能力（碰生产环境的只有它）；`metrics_write` 无人授权——`publish_report` 的失败是**演练出来的拒绝**，不是事故。能力泄露的爆炸半径 = 持有该能力的任务集合，于是「谁持 prod 能力」本身就是设计决策。CI/CD 系统里的部署权限门（GitHub Actions 的 environments / protection rules 等，机制同类，概念性提及）是同一机制在发布流程的形态。**default-deny 的反面**（默认放行、逐个打补丁）安全态势随系统膨胀单调恶化——nano-data-platform L0 §7 在数据消费侧演示过同一思想，这里是执行侧的对应物。

**（b）成本账本：重试不是免费的。**

```python
# ---- [4b] 成本账本：每次 attempt 烧钱。toy 单价 1 coin/attempt，教学设定非真实云价 ----
def cost_report(record):
    wasted = sum(a for n, a in record["attempts"].items() if record["state"][n] == "FAILED")
    recovery = sum(a - 1 for n, a in record["attempts"].items() if record["state"][n] == "SUCCESS")
    return record["coins"], wasted, recovery
```

```text
[6] 成本账本（toy 单价 1 coin/attempt，教学设定非真实云价）：重试不是免费的
  总 9 = 有效 5 + 重试救回 1 + 浪费 3（ingest_web 的 3 次 attempt 全部无效）
  [check 13] PASS  成本恒等式: 9 = 5 + 1 + 3
```

**反幻觉声明**：1 coin/attempt 是教学设定的 toy 单价，**不是任何云厂商的真实价格**；真实成本结构（计算时长 × 规格单价 + 重试放大 + 排队等待）复杂得多，须查官方价目页（概念性指针，不引任何数字 `[TODO: verify 具体价目]`）。但恒等式本身是机制，不依赖单价：

```
总成本 = 有效计算 + 重试救回 + 浪费
  9    =    5     +     1     +   3
```

- **有效 5**：最终 SUCCESS 的 5 个任务各贡献成功那次 attempt（ingest_crm 的有效 coin 是其第 2 次 attempt，失败的第 1 次记入「重试救回」）；
- **重试救回 1**：`ingest_crm` 的第 2 次 attempt——这 1 coin 买回了整个下游锥（§6(b)），最值的一笔；
- **浪费 3**：`ingest_web` 的 3 次 attempt 全部无效——误分类的代价在账本上直接可见。

这就是「为什么生产编排必须有成本可观测」：没有账本，你不知道重试策略是在救管线还是在烧钱；`wasted/recovery` 两个量是调 retries 上限和退避策略的依据。

**思考题 7.1**：如果把 capability 检查改成「先跑任务，跑完再审计权限、不合规则回滚」，demo 里会坏掉什么？（参考方向：publish_report 将烧掉至少 1 次 attempt [账本 +1]，且副作用已发生——报告可能写了一半，回滚不免费也不总可行；default-deny 把安全边界推到「计算发生之前」，`attempts == 0` 就是它的机器证明。）

---

## §8 CI/CD 门 = 依赖边 + 确定性收尾锚点

```text
[7] CI/CD 门 = 依赖边：deploy 等 build_curated 与 unit_tests 双就绪；调和是确定性的
  [check 14] PASS  deploy 在数据与测试双就绪后才跑
  run digest: 0e0b34e0c9eb016f (ticks=5, coins=9)  两遍一致: True
  [check 15] PASS  调和确定性：两遍运行 digest 逐位一致

self-check: 15/15 PASS
```

`build_curated` 的依赖是 `["gate_crm", "unit_tests"]`——数据质量门**和**自动化测试双就绪才构建；`deploy` 又等 `build_curated`。CI/CD 的「门禁」在这里不需要专门的门禁系统：**依赖边就是门**，「什么条件下才允许走下一步」被表达成 DAG 结构，由同一套状态机执行（check 14 验证 deploy 严格在双就绪之后）。自动化测试/部署（ROADMAP §七 关键词）在 L0 即以这种最小形态在场。

**收尾锚点（toy 指标基线，shared/conventions 要求的可量化 L0 指标）**：终态向量 5 SUCCESS / 2 FAILED / 2 UPSTREAM_FAILED；`ticks=5`、`coins=9`；run digest `0e0b34e0c9eb016f`（state/attempts/ticks/coins 的 sha256 前 16 位）。脚本整体输出确定性：两个独立 CWD、`python3 -B` 双跑，EXIT=0、stderr 0 B，stdout 54 行、md5 `802aac9f48d5a7c81a5e61f695c8903d`，逐字节一致（RUN1==RUN2 BYTE-IDENTICAL）。

**为什么确定性是验收标准而不是锦上添花**：调度器是信任基础设施——「这个任务为什么没跑 / 为什么这时候跑」必须可回答、可重放。事件日志 + 确定性调和给出审计能力：同一份状态与规则，重放得到同样的历史。L1 引入真实 wall-clock 与 subprocess 后，墙钟时间不再确定，但「给定同样的状态序列，转移唯一」这条不变——那才是可审计性的真正载体。

---

## §9 它模拟了什么、刻意没模拟什么（L0 边界 → L1/L2）

**模拟了**（本教程的验收内容）：DAG 静态校验（环 / 未知依赖 fail fast）；任务状态机 + 调和循环（状态是完整记录）；错误分类 → 指数退避有界重试 vs 立即失败；上游失败急切传播（下游锥，0 成本）；capability default-deny（拒绝先于计算）；attempt 成本账本（9=5+1+3 恒等式）；CI/CD 门 = 依赖边；确定性（逻辑时钟 + digest）。

**刻意没模拟**（每一面都是更高阶梯的课题，不是遗漏）：

| 没模拟 | 为什么 L0 不做 | 哪一级做 |
|--------|----------------|----------|
| 持久化状态 / 崩溃续跑 | L0 状态在内存，进程死即丢（思考题 4.1）| L1（状态落盘 + 重启续调和）|
| 真实 subprocess 任务（shell/python，有副作用与时长）| L0 任务是纯函数，突出状态机本身；副作用带来幂等问题 | L1 |
| wall-clock / cron 触发、sensor | 逻辑时钟保证确定性 | L1（真实时间触发）|
| RUNNING liveness / zombie 检测 | L0 执行瞬时完成（§4(c)）| L1/L2（heartbeat 机制）|
| 真实并行 / 资源池 / 优先级 | 串行突出「状态决定转移」；并行引入资源竞争与并发安全 | L2（对照 Airflow executor/pool、Dagster concurrency）|
| SLA / backfill / trigger rules 配置化 | 独立机制面 | L2 |
| Agentic 自愈（agent 驱动的管线修复）| 需要 L0–L1 机制为前置 | L2（ROADMAP §七 关键词）|

## §10 费曼自检

**讲给外行听**：编排器是工地上盯进度板的总调度，不是亲自搬砖的工人。他每天早上（tick）看一遍进度板（状态），按固定规则派活：墙体没验收，电工不许进场（依赖是承诺）；水泥车临时堵在路上，记一笔「两小时后再来」（transient 重试，指数退避）；设计图本身画错了，立刻停掉这条线、不许「再试一次」（permanent 止损）；上游工序报废了，下游工序就不开工，一砖一瓦都不浪费（UPSTREAM_FAILED，0 attempt）；哪个施工队有哪个区域的钥匙提前定好，没钥匙的不许进（default-deny，拒绝发生在动工之前）；每个队出工几次都记账，返工的钱单独标红（成本账本，重试不是免费的）。总调度自己一砖不碰——**调度器不执行计算，只做状态转移**。

**思考题汇总**（正文内另有 3.1 / 4.1 / 5.1 / 6.1 / 7.1）：

1. 一句话说清：「调度器跑脚本」与「调度器调和状态」的本质区别是什么？（要点：进度活在调用栈里 vs 活在状态记录里——前者进程死即从头重跑，后者进程死后继续调和；崩溃恢复 / 可观测性 / 重试都从状态长出来。）
2. 本实现里哪两个数据结构分别对应 Airflow 的「TaskInstance 状态」与「调度器事件日志」？（`state` dict / `events` 列表——每 tick 打印的事件流就是它的序列化；L1 的持久化版本就是真实的 scheduler log 与状态库。）
3. 把 `broken_source` 例子里的 retries 上限去掉，管线会怎样？（ingest_web 无限重试：永远到不了「全部终态」，coins 无限烧，gate_web/normalize_web 永远悬在 PENDING——终态概念是一次运行能够结束的前提，上限是收敛保证，不只是成本控制。）

**反例（一个常见错误直觉）**：「编排就是写个 shell 脚本按顺序调几个命令，最多加个 nohup 就能上生产。」——错在三点：其一，**没有状态**：失败即进度丢失，重跑 = 全量重放 + 副作用重复（脚本不知道「上次跑到哪了」）；其二，**没有失败隔离**：一个命令失败整条链停摆，表达不了「web 分支坏了但 crm 分支照常推进」（本 demo 里 ingest_web 失败对 crm 分支零影响，check 11）；其三，**没有重试 / 权限 / 成本语义**：transient 故障靠人肉值守，权限靠「脚本里恰好没写那个命令」，成本无从记账。编排器的本质不是「按序调命令」，而是**把状态、失败、权限、成本都变成一等公民**。

## §11 溯源

| 声明 | 类型 | 来源 |
|------|------|------|
| Airflow 状态表四条引文（§4：success / failed / up_for_retry / upstream_failed 条目释义） | 文献已有（逐字引文） | https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html ，2026-08-12 抓取（156,580 B） |
| zombie / heartbeat 引文「Airflow will find these periodically, clean them up, and mark the TaskInstance as failed or retry it if it has available retries.」（§4） | 文献已有（逐字引文；原文 TaskInstance 带 `<code>` 标记，引文为去标记文本） | 同上（Task Instance Heartbeat Timeout 节） |
| retry policy 引文「fail immediately, skipping any remaining retries」与「a policy can fail earlier but cannot extend past the configured maximum」（§5） | 文献已有（逐字引文；后者原文有换行伪影 "but\ncannot"，换行归一后逐字） | 同上（Retries 节） |
| 官方 retry policy 示例注释「# server error -- worth retrying」/「# client error -- not retryable」（§5） | 文献已有（逐字引文） | 同上 |
| `retry_exponential_backoff` 参数名（§5） | 文献已有（参数名在页面在案） | 同上 |
| Airflow / Dagster / Prefect 一句话定位（§1） | 文献已有（官方 repo 页面标题逐字） | github.com/apache/airflow（585,673 B）/ github.com/dagster-io/dagster（378,485 B）/ github.com/PrefectHQ/prefect（373,640 B），2026-08-12 抓取 |
| Airflow / Dagster / Prefect 为权威参照实现；CI/CD（GitHub Actions / GitLab CI）关键词 | 纲领已有 | ROADMAP §五/§七 参照表 |
| 「结构错误前置到校验期是 DAG 声明式的红利」（§3）；「终态与扫描顺序无关」（§6 思考题答） | 合理推断 | 机制层面概括 / 本教程自论证，无外部引文 |
| 全部 tick / attempts / coins / digest / backoff 数字与 1 coin/attempt toy 单价 | 本实现实测（toy 设定） | `L0_dag_scheduler_state_machine.py` 本次运行输出；非真实云价、不可外推 |

下一站：**L1**——真实 subprocess 任务（有副作用与时长）+ 状态落盘 + 崩溃续跑（kill 调度器后重启续调和）+ wall-clock 退避，复现同一套状态机与失败语义；**L2**——对照 Airflow（scheduler loop / TaskInstance 状态机 / trigger rules / executor 与 pool）/ Dagster / Prefect 源码的取舍分析 + 真实并行与资源池 + CI/CD 参照 + Agentic 自愈（见 README 阶梯表）。
