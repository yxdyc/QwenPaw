# nano-data-orchestration L1 — 真实 subprocess + 状态落盘 + 崩溃续跑 + wall-clock 退避 + 幂等正面登场

> **前置**：`tutorial_L0.md` 必读（本级的状态机规则 A/B/C、失败语义、成本账本全部沿用 L0 语义）。先读 nano-data-platform L1 会更容易代入「状态落盘」的动机，但不必须。Python 3.10+（实测 3.13.13 通过），纯标准库，CPU 约 9s（实测 ~8.7s，含 1.5s 固定 on-call 响应窗）。
> **运行**：`python3 L1_subprocess_state_and_crash_recovery.py`（任意目录可跑；workdir 用 tempdir、跑完自清理、路径不入输出）。
> **本文件是 notebook-style 教程**：叙述 + 代码摘录 + 真实运行输出 + 思考题交替推进。

---

## §1 L0 留下的两笔债

L0 把调度器的状态机内核裸了出来（DAG 校验 / 调和循环 / 失败语义 / default-deny / 成本账本），但它有两笔**故意的债**——L0 思考题 4.1 与 5.1 各自记了一笔：

**债一：进度活在调用栈里。** L0 的 `state` 是内存里的 dict，进程一死全丢，只能整条重跑。思考题 4.1 问「真实系统把什么落盘，才能在重启后回答『我们进行到哪一步了』」——L0 给不了答案，因为它的任务瞬时完成，「重启」没有意义。真实任务会跑几分钟几小时（L0 §4(c)），调度器自己也可能死——**崩溃是持续运行系统的常态，不是异常**。

**债二：任务是纯函数。** L0 的 `spec["fn"](attempt)` 没有进程边界：没有副作用、没有时长、没有「任务死了但调度器不知道」。一旦任务变成真实进程，三件事立刻找上门：结果怎么跨过进程边界传回来（exit code）？副作用重试会不会重复（幂等）？RUNNING 状态里那个 pid 还活着吗（zombie）？

L1 的定义（README 阶梯行）就是还清这两笔债：

> 任务换成真实 subprocess（shell/python，有副作用与时长）：状态落盘（JSON/SQLite）+ 崩溃续跑（kill -9 调度器后重启续调和）+ wall-clock 退避 + 幂等问题正面登场（非幂等任务重试的副作用重复），复现 L0 状态机与失败语义

注意「复现 L0 状态机与失败语义」——L1 的验收不是「又写了一个调度器」，而是**同一套语义在真实进程基底上逐字成立**：L0 的终态向量、成本恒等式、退避算术，在 L1 全部有机器断言对应（check 01–03 / 07 / 05）。语义不变、基底更换——这正是「状态决定转移，基底决定代价」的机制观。

---

## §2 先跑一遍

**可运行性契约声明（ROADMAP §三）**：L0/L1 必须可跑。本文件是**真 L1，无 mock**——调度器与任务都是真实进程，kill -9 是真 kill（SIGKILL 整个进程组），续跑是真续跑（新调度器进程从盘上状态继续调和）。唯一的实验设计有二：其一是**失败日程显式**（同 L0 fixture 声明的口径——`ingest_crm` 只败第 1 次、`ingest_web` 永远败且被故意误分类为 transient、`publish_report` 故意不授权），失败是机制对象不是事故；其二是**故障模型选定为「宿主死亡」**（kill -9 整个进程组，调度器与正在跑的任务同死）——另一种故障模型（调度器死、任务活）在 §5(c) 与 §11 明确为 L2 课题。

```bash
$ python3 L1_subprocess_state_and_crash_recovery.py
```

完整输出如下（elapsed 掩码行已按口径 `sed '/^[[:space:]]*elapsed/d'` 删除——墙钟与 pid 是不确定量，不进 check 路径；掩码口径与双跑锚点见 §12。以下各节的输出块均从此同一次运行中截取，逐字子序列）：

```text
== nano-data-orchestration L1: 真实 subprocess + 状态落盘 + 崩溃续跑 + wall-clock 退避 + 幂等正面登场 ==
  （L0 状态机语义不变，基底换成真实进程：exit code 是分类通道，state.json 是 single source of truth）

[1] Run A 干净基线：L0 fixture 与失败语义在真实 subprocess 上复现
  --- 调度器事件流（Run A，调度器为独立子进程，此处为其 stdout 原文）---
  [seq 01] ingest_crm     -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/ingest_crm.attempt1.log）
  [seq 02] ingest_crm     -> RETRYING        attempt 1 exit=75 (EX_TEMPFAIL transient) —— 计划退避 0.60s（wall-clock）
  [seq 03] ingest_web     -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/ingest_web.attempt1.log）
  [seq 04] ingest_web     -> RETRYING        attempt 1 exit=75 (EX_TEMPFAIL transient) —— 计划退避 0.60s（wall-clock）
  [seq 05] unit_tests     -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/unit_tests.attempt1.log）
  [seq 06] unit_tests     -> SUCCESS         attempt 1 exit=0
  [seq 07] ingest_crm     -> RUNNING         attempt 2 subprocess 已启动（pid 已落 state.json；输出 -> logs/ingest_crm.attempt2.log）
  [seq 08] ingest_crm     -> SUCCESS         attempt 2 exit=0  <- 重试救回
  [seq 09] gate_crm       -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/gate_crm.attempt1.log）
  [seq 10] gate_crm       -> SUCCESS         attempt 1 exit=0
  [seq 11] build_curated  -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/build_curated.attempt1.log）
  [seq 12] build_curated  -> SUCCESS         attempt 1 exit=0
  [seq 13] deploy         -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/deploy.attempt1.log）
  [seq 14] deploy         -> SUCCESS         attempt 1 exit=0
  [seq 15] ingest_web     -> RUNNING         attempt 2 subprocess 已启动（pid 已落 state.json；输出 -> logs/ingest_web.attempt2.log）
  [seq 16] ingest_web     -> RETRYING        attempt 2 exit=75 (EX_TEMPFAIL transient) —— 计划退避 1.20s（wall-clock）
  [seq 17] publish_report -> FAILED(deny)    capability missing (default-deny): ['metrics_write'] —— permanent 不重试，0 attempt 0 成本，subprocess 从未启动
  [seq 18] ingest_web     -> RUNNING         attempt 3 subprocess 已启动（pid 已落 state.json；输出 -> logs/ingest_web.attempt3.log）
  [seq 19] ingest_web     -> FAILED          attempt 3 exit=75 —— 重试上限 (2) 耗尽，止损——上限是误分类的最后防线
  [seq 20] gate_web       -> UPSTREAM_FAILED 上游终态失败——依赖是承诺：不在坏数据上跑
  [seq 21] normalize_web  -> UPSTREAM_FAILED 上游终态失败——依赖是承诺：不在坏数据上跑
  终态向量: SUCCESS=['build_curated', 'deploy', 'gate_crm', 'ingest_crm', 'unit_tests']
            FAILED=['ingest_web', 'publish_report']  UPSTREAM_FAILED=['gate_web', 'normalize_web']
  [check 01] PASS  终态向量与 L0 逐字一致: 5 SUCCESS（含重试救回的 ingest_crm）
  [check 02] PASS  2 FAILED（ingest_web 止损 / publish_report 被拒）
  [check 03] PASS  2 UPSTREAM_FAILED（坏源的爆炸半径）
  [check 04] PASS  exit code 分类通道: ingest_crm attempt1 exit=75 -> 重试；ingest_web 三次 exit=75 全部在案
  [check 05] PASS  wall-clock 退避算术确定: 计划等待 ingest_crm [0.60] / ingest_web [0.60, 1.20]（= 0.6*2^(k-1)）
  [check 06] PASS  default-deny: publish_report attempts==0 且无日志文件——subprocess 从未启动
  成本账本（成本单位 = 1 次 subprocess 启动）: 总 9 = 有效 5 + 重试救回 1 + 浪费 3 + 崩溃税 0
  [check 07] PASS  成本恒等式 9 = 5 + 1 + 3 + 0（复现 L0 check 13）
  [check 08] PASS  state.json 与事件流收敛到同一终态；seq 1..N 单调无 gap；调度器干净退出
  终态向量 digest: ac4a0b3ac09bf47b（Run B 崩溃续跑后必须收敛到同一值）

[2] Run B 崩溃续跑：kill -9 整个进程组（宿主死亡模型）-> 重启从盘上继续调和
  kill 点快照（state.json 读值）: build_curated=RUNNING deploy=PENDING gate_crm=SUCCESS gate_web=PENDING ingest_crm=SUCCESS ingest_web=RUNNABLE normalize_web=PENDING publish_report=PENDING unit_tests=SUCCESS
  [check 09] PASS  kill 点 = 确定逻辑点: build_curated RUNNING（attempt 1 执行中），4 任务已终态，ingest_web 已唤醒待串行槽（RUNNABLE）
  [check 10] PASS  kill -9 后: 调度器返回 -SIGKILL；state.json 完整在盘，build_curated 留下 stale 态（RUNNING + 录 pid）
  盘上 stale 态（build_curated 行投影）: state=RUNNING attempts=1 pid=<已录, 已死> exit_code=None —— 重启的第一件事是识别它
  --- 调度器事件流（Run B 重启后，stdout 原文；seq 接续崩溃前编号）---
  [seq 12] build_curated  -> RETRYING        zombie 识别: state=RUNNING 而录 pid 已死（宿主死亡模型）——归 transient，计划退避 0.60s
  [seq 13] ingest_web     -> RUNNING         attempt 2 subprocess 已启动（pid 已落 state.json；输出 -> logs/ingest_web.attempt2.log）
  [seq 14] ingest_web     -> RETRYING        attempt 2 exit=75 (EX_TEMPFAIL transient) —— 计划退避 1.20s（wall-clock）
  [seq 15] build_curated  -> RUNNING         attempt 2 subprocess 已启动（pid 已落 state.json；输出 -> logs/build_curated.attempt2.log）
  [seq 16] build_curated  -> SUCCESS         attempt 2 exit=0  <- 重试救回
  [seq 17] deploy         -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/deploy.attempt1.log）
  [seq 18] deploy         -> SUCCESS         attempt 1 exit=0
  [seq 19] ingest_web     -> RUNNING         attempt 3 subprocess 已启动（pid 已落 state.json；输出 -> logs/ingest_web.attempt3.log）
  [seq 20] ingest_web     -> FAILED          attempt 3 exit=75 —— 重试上限 (2) 耗尽，止损——上限是误分类的最后防线
  [seq 21] gate_web       -> UPSTREAM_FAILED 上游终态失败——依赖是承诺：不在坏数据上跑
  [seq 22] normalize_web  -> UPSTREAM_FAILED 上游终态失败——依赖是承诺：不在坏数据上跑
  [seq 23] publish_report -> FAILED(deny)    capability missing (default-deny): ['metrics_write'] —— permanent 不重试，0 attempt 0 成本，subprocess 从未启动
  [check 11] PASS  zombie 识别: state=RUNNING 而录 pid 已死 -> 归 transient 回重试通道（计划退避 0.60s）
  [check 12] PASS  终态向量 == Run A（同一 digest，崩溃不改变收敛点）
  [check 13] PASS  已完成的工作不重做: 全部任务日志文件数 == attempts（重启不给已完成任务追加启动）
  [check 14] PASS  崩溃税 = 恰好 1: Run B coins 10 = Run A 9 + 1（被 kill 的 attempt 也烧了钱）
  [check 15] PASS  被 kill 的 attempt 零输出零产物: attempt1 日志 0 字节；attempt2 完整（含副作用行）
  [check 16] PASS  原子发布在崩溃下收敛: curated.jsonl 恰 3 行且全部 layer=curated（kill 点落在计算中，未发布）
  [check 17] PASS  崩溃前副作用不受崩溃影响: raw_crm.jsonl 与 Run A 逐字节一致
  [check 18] PASS  events.jsonl 跨两个调度器进程 seq 单调无 gap: 历史不被擦除，编号不重启

[3] Run C 幂等正面登场：同一调度器、同一重试策略，任务侧幂等与否决定副作用是否重复
  --- 调度器事件流（Run C，stdout 原文）---
  [seq 01] export_append  -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/export_append.attempt1.log）
  [seq 02] export_append  -> RETRYING        attempt 1 exit=75 (EX_TEMPFAIL transient) —— 计划退避 0.60s（wall-clock）
  [seq 03] export_atomic  -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/export_atomic.attempt1.log）
  [seq 04] export_atomic  -> RETRYING        attempt 1 exit=75 (EX_TEMPFAIL transient) —— 计划退避 0.60s（wall-clock）
  [seq 05] export_broken  -> RUNNING         attempt 1 subprocess 已启动（pid 已落 state.json；输出 -> logs/export_broken.attempt1.log）
  [seq 06] export_broken  -> FAILED          attempt 1 exit=1 —— permanent 立即止损，不重试
  [seq 07] export_append  -> RUNNING         attempt 2 subprocess 已启动（pid 已落 state.json；输出 -> logs/export_append.attempt2.log）
  [seq 08] export_append  -> SUCCESS         attempt 2 exit=0  <- 重试救回
  [seq 09] export_atomic  -> RUNNING         attempt 2 subprocess 已启动（pid 已落 state.json；输出 -> logs/export_atomic.attempt2.log）
  [seq 10] export_atomic  -> SUCCESS         attempt 2 exit=0  <- 重试救回
  report_append.txt（非幂等: 副作用=append，失败在副作用之后）= 'row-42\nrow-42\n' -> 2 行逐字重复
  report_atomic.txt（幂等: 工作可重做 + 原子发布）          = 'row-42\n' -> 恰 1 行
  [check 19] PASS  非幂等任务重试 = 副作用重复: report_append.txt 恰 2 行逐字相同（attempts=2，重试救回但副作用加倍）
  [check 20] PASS  幂等任务重试 = 收敛: report_atomic.txt 恰 1 行内容正确（attempts=2，重试无害）
  [check 21] PASS  崩溃垃圾可见: attempt1 的 temp 文件失去主人仍在盘（无害但需 GC——真实系统的 orphan 清理课题）
  [check 22] PASS  permanent 不重试: export_broken exit=1，retries=2 也 attempts==1 立即止损

[4] 跨 run 不变量：收敛点相同，代价账不同
  [check 23] PASS  拓扑不变式（Run B 事件流）: 每个跑过的任务都在全部上游之后首启
  [check 24] PASS  状态持久化往返: 重读 state.json 重算 digest 不变（逻辑投影序列化稳定）
  终态向量 digest: Run A = ac4a0b3ac09bf47b  Run B = ac4a0b3ac09bf47b（相同）
  attempts digest: Run A = f96c37e75f400691  Run B = a283a3ff01a5543c（不同 = 崩溃税，正确）

self-check: 24/24 PASS
```

三幕结构：**Run A** 干净基线——L0 fixture（deps/retries/needs 逐字沿用）在真实 subprocess 上复现，终态向量与成本恒等式和 L0 逐字一致；**Run B** 崩溃续跑——驱动在 `build_curated` attempt 1 执行中 kill -9 整个进程组，等一个固定的 on-call 响应窗后重启调度器，验证 zombie 识别、已完成工作不重做、收敛点不变；**Run C** 幂等对照——同一调度器、同一重试策略，任务侧幂等与否决定副作用是否重复。

> **fixture 声明（承 L0 口径）**：失败日程是显式实验设计，按 attempt 编号决定行为（确定性可复算）：`ingest_crm` attempt 1 exit 75（transient 原型）；`ingest_web` 永远 exit 75 且被作者故意误分类为 transient（止损教训）；`publish_report` 故意不授权 `metrics_write`（default-deny 演练）；`build_curated` 是 1.2s 长任务 = Run B 的崩溃受害者；Run C 的 `export_append`/`export_atomic`/`export_broken` 分别是「非幂等 + 副作用后失败」「幂等 + 发布前失败」「permanent exit 1」三个对照组。这不是假数据冒充跑通——L1 的机制对象就是「调度器在真实进程、真实崩溃、真实副作用面前做什么决定」。

---

## §3 机制面 [1]：任务 = 真实 subprocess，exit code 是通用分类通道

L0 的错误分类靠异常类型（`TransientError`/`PermanentError`）——异常活在同一个进程里，调度器 `try/except` 就接得住。真实任务跨过进程边界，异常传不回来：**进程边界上唯一通用的结果通道是 exit code**（POSIX 约定，0 = 成功；非 0 = 失败，具体语义由契约定义——机制层归纳）。L1 的分类契约：

```
exit 0   -> 成功
exit 75  -> transient（EX_TEMPFAIL：值得重试）
其余非 0 -> permanent（立即止损）
```

75 不是 nano 的发明。BSD 系标准头文件 sysexits.h 早就把「临时失败、请重试」标准化成了一个 exit code（本机 macOS SDK 在盘，2026-08-14 核验，见 §12）：

```c
#define EX_TEMPFAIL	75	/* temp failure; user is invited to retry */
```

「user is invited to retry」——头文件注释把重试语义直接写进了数字里。任务侧因此长这样（摘录自 `TASK_PROGRAM`，任务以 `[sys.executable, -B, -c]` 启动，上下文经环境变量 `NANO_TASK / NANO_ATTEMPT / NANO_WORKDIR` 传递）：

```python
if t == "ingest_crm":
    if a == 1: say("connection reset（模拟 transient 故障）"); sys.exit(75)   # EX_TEMPFAIL
    atomic_write(os.path.join(w, "raw_crm.jsonl"),
                 "".join(json.dumps({"id": f"crm-{i:03d}", "src": "crm"}) + "\n" for i in range(1, 4)))
    say("side effect: raw_crm.jsonl 写入 3 行（原子发布）"); sys.exit(0)
if t == "ingest_web":
    say("source 拒绝所有读取（永久损坏，作者却误分类为 transient）"); sys.exit(75)
```

调度器侧的分类体（`launch_and_wait` 摘录）——L0 的两个 `except` 分支换成了对 `rc` 的三分：

```python
    rc = p.wait(); log_f.close()
    t["pid"] = None; t["exit_code"] = rc
    if rc == 0:                                    # exit code = 通用分类通道（L0 的异常类型在进程边界的对应物）
        t["state"] = "SUCCESS"
        append_event(wd, name, "SUCCESS", f"attempt {a} exit=0" + ("  <- 重试救回" if a > 1 else ""), a)
    elif rc != EX_TEMPFAIL:                        # permanent：立即止损
        t["state"] = "FAILED"
        append_event(wd, name, "FAILED", f"attempt {a} exit={rc} —— permanent 立即止损，不重试", a)
    elif t["attempts"] <= spec["retries"]:         # transient：wall-clock 指数退避（计划等待是确定算术）
        wait = BACKOFF_BASE * 2 ** (t["attempts"] - 1)
        t["state"] = "RETRYING"; t["retry_after"] = time.time() + wait
        t["backoffs"].append(wait)
        append_event(wd, name, "RETRYING", f"attempt {a} exit=75 (EX_TEMPFAIL transient) —— 计划退避 {wait:.2f}s（wall-clock）", a)
```

Airflow 的 BashOperator 是同一形态（官方文档，2026-08-14 抓取，逐字，见 §12）：

> "In general a non-zero exit code produces an AirflowException and thus a task failure."

差别在一处：Airflow 把**所有**非 0 都当失败，重试与否另由 retry policy 决定；nano 把 transient/permanent 的区分压进 exit code 契约（75 vs 其余）。两种设计都成立——共同点是**分类必须事先约定成跨进程契约**：调度器看不见任务内部，任务作者必须把「重试是否可能成功」编码进进程边界传得出去的东西里。L0 §5(a) 的「分类是任务作者的责任，调度器只执行分类的后果」在进程边界下不但成立，而且更硬了——异常可以携带丰富的上下文，exit code 只有 8 位，契约不事先定好，信息就永远丢了。

Run C 的 `export_broken`（exit 1，permanent）验证了分类通道的另一半（上面输出 [seq 05]–[seq 06]）：`retries=2` 也 `attempts==1` 立即止损（check 22）——permanent 不重试的语义从 L0 check 08（`publish_report` 的 deny 路径）扩展到了 exit code 路径。

**思考题 3.1**：exit code 比异常「弱」在哪，为什么它反而是唯一正确的跨进程通道？（参考方向：异常携带类型、消息、栈，但随进程而死；exit code 是 8 位整数，任何语言任何运行时都产得出、任何编排器都读得到。放弃的是结构化错误信息，换来的是**通道通用性**——于是分类必须前移为事先契约 [75/其余]。接口窄不是缺陷，是进程边界的物理事实；幻想「把异常序列化传回来」就走向了私有协议，失去通用性。）

---

## §4 机制面 [2]：状态落盘——进度从调用栈搬到盘上

L0 思考题 4.1 的答案现在可以给了：落盘两样东西——**state.json（现在：每个任务的状态与簿记）**与 **events.jsonl（历史：每次转移的 append-only 日志）**。两者是同一真相的两个视图：状态是快照，日志是轨迹（L0 思考题汇总第 2 题的「`state` dict / `events` 列表」在 L1 的持久化形态）。

```python
def save_state(wd, st):
    tmp = state_path(wd) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False, sort_keys=True, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, state_path(wd))                # 原子替换 = 调度器层的幂等发布（与任务的 atomic_write 同构）

def append_event(wd, task, to, why, attempt):
    # seq 以 events.jsonl 本身为源（行数 + 1）：重启不重新编号、崩溃不擦除历史——日志是编号的 single source of truth
    seq = 0
    if os.path.exists(events_path(wd)):
        with open(events_path(wd)) as f: seq = sum(1 for _ in f)
    seq += 1
    with open(events_path(wd), "a") as f:
        f.write(json.dumps({"seq": seq, "task": task, "to": to, "why": why, "attempt": attempt,
                            "ts": round(time.time(), 3)}, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    print(f"  [seq {seq:02d}] {task:14s} -> {to:15s} {why}")
```

三个设计决策，各答一个「为什么」：

**（a）state.json 的原子写 = temp + fsync + replace。** 如果直接覆写 state.json，崩溃落在写一半的瞬间，重启读到的是半新半旧的 JSON——解析失败，恢复没有起点。原子写保证「要么旧版完整、要么新版完整」，这是可恢复性的前提。注意同构性：任务侧的 `atomic_write`（§8 的幂等武器）与调度器侧的 `save_state` 是**同一个模式**——调度器和任务面对同一个敌人（崩溃），就必须用同一件武器。nano-data-platform L1 的「落数据 → 推游标」单事务顺序在这里的对应物是「先落 pid 再 wait」：

```python
    t["state"] = "RUNNING"; t["started_at"] = time.time()
    p = subprocess.Popen([sys.executable, "-B", "-c", TASK_PROGRAM],
                         stdout=log_f, stderr=subprocess.STDOUT, env=env)
    t["pid"] = p.pid
    save_state(wd, st)                             # 先落 pid 再 wait——崩溃识别依赖这一步（§5 的窗口分析）
    append_event(wd, name, "RUNNING", ...)
    rc = p.wait(); log_f.close()
```

顺序反了会怎样？先 wait 再落 pid：任务跑完之前崩溃，重启时状态还是 RUNNABLE——调度器会**再启动一次**同一个任务，而第一次启动的子进程可能还活着（宿主未死的故障模型下）或已留下副作用：重复执行。先落 pid 再 wait 把「状态不知道的子进程」窗口压到 Popen 与 save_state 之间的微秒级——压不掉（没有两阶段提交），但窗口大小是工程可控量。这个残余风险在 §8 会与幂等合流：**窗口内的重复执行，最终靠任务幂等兜底**。

**（b）seq 以日志为源，不以状态为源。** `append_event` 的编号 = events.jsonl 行数 + 1——日志本身是编号的 single source of truth。如果 seq 存在 state.json 里，崩溃落在「状态已写、日志未附」的缝里，重启后 seq 会重复或跳号；以日志为源则天然免疫（重启只是继续数行数）。这与 nano-data-platform L1「逻辑时钟从 catalog 重建」是同一思想：**确定性不依赖进程存活，而依赖可从盘上重建的结构**。check 18 是它的机器断言：Run B 的 events.jsonl 跨两个调度器进程 seq 单调无 gap（崩溃前 seq 1–11，重启后续 12–23）。

**（c）stale 字段只在状态守卫下被读。** Run B kill 点的真实 state.json（探针复现，pid 与时间戳脱敏为 `<pid>`/`<epoch>`；注意 `ingest_crm` 已 SUCCESS 却仍留着 `retry_after`——它是 RETRYING 时代的 stale 字段）：

```json
{
 "tasks": {
  "build_curated": {"attempts": 1, "backoffs": [], "exit_code": null, "pid": "<pid>",
                    "retry_after": null, "started_at": "<epoch>", "state": "RUNNING"},
  "ingest_crm":    {"attempts": 2, "backoffs": [0.6], "exit_code": 0, "pid": null,
                    "retry_after": "<epoch>", "started_at": "<epoch>", "state": "SUCCESS"},
  "ingest_web":    {"attempts": 1, "backoffs": [0.6], "exit_code": 75, "pid": null,
                    "retry_after": "<epoch>", "started_at": "<epoch>", "state": "RUNNABLE"}
 }
}
```

`ingest_crm.retry_after` 没被清理，为什么无害？因为规则 A 只在 `state == "RETRYING"` 时读它——**字段的语义由状态守卫赋予，不由字段值本身携带**。真实系统的状态表里同样躺满 stale 列（Airflow 的 TaskInstance 表在状态转移后也不逐列清零，机制同类——合理推断），读任何字段前先问「这个状态下它有效吗」。stale 态真正危险的是 `build_curated` 那行：`state=RUNNING + pid 已录`——进程已经死了，状态还说活着。这就是下一节的 zombie。

events.jsonl 一行长这样（ts 脱敏）：

```json
{"seq": 2, "task": "ingest_crm", "to": "RETRYING", "why": "attempt 1 exit=75 (EX_TEMPFAIL transient) —— 计划退避 0.60s（wall-clock）", "attempt": 1, "ts": "<epoch>"}
```

**思考题 4.1**：state.json 与 events.jsonl 都落盘，是不是冗余？只留其一会丢什么？（参考方向：只留状态 = 有现在没历史——「这个任务为什么是 FAILED」回答不了，审计能力蒸发（L0 §8 的可重放性）；只留日志 = 有历史没现在——每次启动要重放全日志才能知道当前态，恢复成本 O(历史)。状态是日志的物化视图，两者收敛到同一终态正是 check 08 的断言。）

---

## §5 机制面 [3]：崩溃续跑——kill -9 进程组、zombie 识别、宿主死亡模型

Run B 的实验设计：驱动启动调度器（独立子进程、独立进程组），轮询 state.json 直到 `build_curated` 进入 RUNNING（确定逻辑点：此刻 `ingest_crm/gate_crm/unit_tests` 已 SUCCESS、`ingest_web` 已唤醒待串行槽），然后 `os.killpg(pgid, SIGKILL)`——**调度器与它正在跑的子进程同组俱灭**，模拟宿主死亡。等一个固定的 on-call 响应窗（1.5s，模拟值班响应不是瞬时的），再以同一命令、同一 workdir 重启调度器。

kill -9 之后、重启之前，盘上是这样的（上面输出 [2] 段）：state.json 完整在盘（check 10），`build_curated` 留下 stale 态——`state=RUNNING`、pid 已录、进程已死。重启的调度器第一件事是识别它：

```python
def recover_zombies(tasks, st, wd):
    for name in sorted(tasks):
        t = st["tasks"][name]
        if t["state"] != "RUNNING": continue
        if t["pid"] is not None and pid_alive(t["pid"]):
            raise RuntimeError(
                f"orphan task {name}: 调度器死而子进程活——exit code 已丢失，L1 不做孤儿收养（需 heartbeat/结果通道，L2 课题）")
        t["pid"] = None; t["exit_code"] = None
        if t["attempts"] <= tasks[name]["retries"]:
            wait = BACKOFF_BASE * 2 ** (t["attempts"] - 1)
            t["state"] = "RETRYING"; t["retry_after"] = time.time() + wait
            t["backoffs"].append(wait)
            append_event(wd, name, "RETRYING",
                         f"zombie 识别: state=RUNNING 而录 pid 已死（宿主死亡模型）——归 transient，计划退避 {wait:.2f}s", t["attempts"])
```

三个本质点：

**（a）zombie = 状态说活着、进程已死。** Airflow 对同一问题的官方描述（tasks 文档，2026-08-14 重抓 156,580 B 与 L0 08-12 录值同尺寸零漂移，逐字，见 §12）：

> "TaskInstances may get stuck in a running state despite their associated jobs being inactive (for example if the TaskInstance's worker ran out of memory). Such tasks were formerly known as zombie tasks. Airflow will find these periodically, clean them up, and mark the TaskInstance as failed or retry it if it has available retries."

「stuck in a running state despite their associated jobs being inactive」——nano 的 kill 点 stale 态逐字对应这句话。差别在检测形态：Airflow 周期扫描 + heartbeat 超时判定（连续形态），nano 只在重启时检查 pid 死活（点状形态）——L1 的故障模型里崩溃必然伴随重启，点状检测已经完备；连续形态的 liveness 检测是 L2 课题（§11）。

**（b）zombie 归 transient，因为「被外部 kill」不是任务自己的错。** 分类永远回答同一个问题：**重试是否可能成功？** attempt 死于宿主死亡（外部原因），换一次执行环境就可能成功 → transient；如果 attempt 每次都跑到同一内存水位被 OOM kill（确定性资源超限），重试 N 次同果，正确动作是扩资源改调度而不是重试——那该归 permanent（思考题 5.1）。zombie 识别后回重试通道、走同一套 wall-clock 退避（check 11：`backoffs == [0.6]`），状态机主干不需要为崩溃新增任何分支——**崩溃恢复不是新机制，是状态机在更大时间尺度上的一次调和**。

**（c）孤儿不收养，是诚实不是偷懒。** `pid_alive` 为真时 `recover_zombies` 直接 raise：调度器死而子进程活的故障模型下，父子关系已断，**exit code 永远拿不回来**（`waitpid` 等不了别人的孩子）——「跑完了」与「还在跑」不可区分，任何状态转移都是猜。Airflow 的答案是 heartbeat：任务进程周期上报心跳，liveness = 心跳新鲜度而非 pid 存在（同页下文即列 heartbeat timeout 的诸原因，含 "The Airflow worker ran out of memory and was OOMKilled"，逐字在盘）——结果不走父子关系，走独立通道。nano L1 把这个答案留给 L2（§11），本级只模拟宿主死亡模型：kill -9 进程组保证「RUNNING 的 pid 必然已死」，`pid_alive` 分支在 demo 中不被触发但代码在位、边界明示。

**（d）合流：过程与计时有关，终态与计时无关。** 对照 Run A 与 Run B 的事件流：Run A 里 `publish_report` 的 deny 在 seq 17（`ingest_web` attempt 3 之前），Run B 里在 seq 23（全部失败传播之后）——wall-clock 下退避到期时刻与任务时长交错，事件顺序真的变了。但 check 12 断言终态向量 digest 逐位相同（`ac4a0b3ac09bf47b`）。L0 思考题 6.1 说「终态与扫描顺序无关，过程与扫描顺序有关」，那时是纸面论证；L1 用真实 wall-clock 跑出了两个不同过程、同一终态——**调和循环的合流性质在真实时间下经受住了**。这是「重启后继续调和是安全的」的深层原因：安全不来自「重启恰好没改变什么」，而来自「转移规则由状态唯一决定」。

**思考题 5.1**：zombie 一律归 transient 对吗？给出一个 zombie 应该归 permanent 的场景，并说出判据。（参考方向：任务每次运行到确定性资源墙 [如固定输入下的 OOM] 被 kill——外部 kill 只是表象，根因是任务自身的确定性行为，重试同果。判据始终是「重试是否可能成功」：kill 的原因在环境侧 [宿主死、网络断、被误杀] → transient；在任务的确定性行为侧 → permanent。误分类的代价 L0 已经演过：transient 误判烧钱 [ingest_web]，permanent 误判丢数据。）

---

## §6 崩溃不能改变的，与必须留下的

Run B 的 check 12–18 是一组「崩溃审计」——崩溃**不能改变**的：

- **收敛点**（check 12）：终态向量 digest 与 Run A 逐位相同；
- **已完成的工作**（check 13）：全部任务的日志文件数 == attempts——重启不给任何已完成任务追加进程启动（`ingest_crm` 仍是 attempt 1+2 两份日志，没有 attempt 3）；
- **崩溃前的副作用**（check 17）：`raw_crm.jsonl` 与 Run A 逐字节一致；
- **历史**（check 18）：events.jsonl 跨进程 seq 单调无 gap。

崩溃**必须留下**的：

- **崩溃税**（check 14）：Run B coins = 10 = Run A 9 + 1——被 kill 的 attempt 消耗了一次真实进程启动，账必须记（§9 展开）；
- **被 kill attempt 的零产物证据**（check 15/16）：`build_curated.attempt1.log` 0 字节（kill 落在 1.2s 计算中，第一行 stdout 都没打出）、`curated.jsonl` 恰 3 行全对——因为发布是原子 rename，kill 点落在「私有 temp 区计算」阶段，读者永远看不到半成品。

后两条把「原子发布」从 §4 的调度器侧口号变成了任务侧的生存技能——下一节正面讲它，以及它的反面。

---

## §7 机制面 [4]：wall-clock 退避——确定的算术，不确定的醒来

L0 规则 A 比较的是逻辑 tick；L1 换成 epoch：

```python
        for n in sorted(tasks):                                # 规则 A：RETRYING 唤醒——比较对象从逻辑 tick 换成 epoch
            t = st["tasks"][n]
            if t["state"] == "RETRYING" and t["retry_after"] <= now:
                t["state"] = "RUNNABLE"; changed += 1
```

一字之改（`tick` → `now`），语义全变：**计划等待是确定算术，实际醒来是不确定事件**。`retry_after = 失败时刻 + 0.6*2^(k-1)` 落进 state.json，任何时候都能复算「这次重试计划等多久」（check 05：`[0.60]` 与 `[0.60, 1.20]` 从盘上读回、逐位吻合）；但调度器实际在 `retry_after + ε` 才醒来（sleep 粒度、进程启动、串行槽排队）——ε 不确定，于是 ε 以及一切墙钟量（总耗时、kill 点时刻）全部落 `elapsed` 掩码行，不进 check 路径、不进 digest。**可审计性没有消失，只是搬了家**：L0 的审计载体是「tick + 事件序」，L1 的审计载体是「state.json + events.jsonl 的逻辑字段」——墙钟只是注释。

退避基数选 0.6s 是 demo 尺度（总时长 ~9s 的约束下让 0.6/1.2 两档退避都可观察）；`0.6*2^(k-1)` 与 L0 的 `2^(k-1)` tick 同构——指数退避的机制动机（给故障源恢复时间、确定性可复算）在 L0 §5(b) 已讲透，L1 只换时钟。

Run B 的 on-call 响应窗 `ON_CALL_WINDOW = 1.5` 是固定常数，双重身份：叙事上它模拟「值班响应不是瞬时的」；机制上它是双保险——本 fixture 在 kill 点 `ingest_web` 已是 RUNNABLE（第一轮退避早已到期），没有任何 pending 退避横跨崩溃点，重启后的事件顺序本就已钉死、与重启延迟无关（§5(d) 的顺序差异来自 run 间计时交错，不来自重启时机）；常数窗把「重启不是瞬时的」也变成显式可控量。**把确定性让渡给显式常数，而不是让渡给运气**——这是 demo 级确定性设计的一般姿态。

**思考题 7.1**：如果把规则 A 的比较改成 `retry_after <= now + 1`（提前 1s 唤醒），会坏掉什么？改成迟到唤醒呢？（参考方向：提前唤醒 = 退避语义失效——重试撞回故障窗的概率回升，且账本上记的「计划退避」与实际行为不符，审计失真；迟到唤醒只损失吞吐，语义仍然正确。生产系统容忍调度迟到、不容忍调度提前——这也是为什么 Airflow 的 `retry_delay` 是下界语义而非精确时刻。）

---

## §8 机制面 [5]：幂等正面登场

L0 思考题 5.1 预警过「副作用放大——非幂等任务重试会重复副作用」；L1 用 Run C 把它跑出来。同一调度器、同一重试策略（retries=1、exit 75 触发），三个对照任务：

```python
IDEMPOTENCE_DEMO = {
    "export_append":  dict(deps=[], retries=1),  # 非幂等：副作用 = append，且失败发生在副作用之后（最坏情形）
    "export_atomic":  dict(deps=[], retries=1),  # 幂等：  工作可重做，发布 = 原子 rename
    "export_broken":  dict(deps=[], retries=2),  # permanent：exit 1（校验失败）——retries=2 也不重试
}
```

```python
if t == "export_append":                           # 非幂等对照：副作用先发生，失败随后
    with open(os.path.join(w, "report_append.txt"), "a") as f: f.write("row-42\n")
    say("side effect: report_append.txt append 1 行")
    if a == 1: say("副作用之后才失败（最坏情形：副作用已落地）"); sys.exit(75)
    sys.exit(0)
if t == "export_atomic":                           # 幂等对照：工作可重做，发布前失败不留痕
    tmp = os.path.join(w, f"report_atomic.txt.tmp.attempt{a}")
    with open(tmp, "w") as f: f.write("row-42\n")
    say("工作完成（temp 区，尚未发布）")
    if a == 1: say("发布之前失败（temp 文件成为孤儿垃圾）"); sys.exit(75)
    os.replace(tmp, os.path.join(w, "report_atomic.txt"))
    say("side effect: report_atomic.txt 原子发布"); sys.exit(0)
```

结果（上面输出 [3] 段）：`report_append.txt = 'row-42\nrow-42\n'`——**两次 attempt 各 append 一行，逐字重复**（check 19）；`report_atomic.txt = 'row-42\n'`——恰 1 行（check 20），attempt 1 的 temp 文件 `report_atomic.txt.tmp.attempt1` 作为崩溃垃圾留在盘（check 21：无害、可见、需 GC）。两个任务都被「重试救回」（attempts 都是 2），但救回的代价完全不同：**非幂等任务的「救回」是带利息的——副作用加倍**。

这不是 nano 特有的坑——Airflow 在接口层就把幂等前提内化了（XComs 文档，2026-08-14 抓取，逐字，见 §12）：

> "If the first task was not successful then on every retry task XComs will be cleared to make the task run idempotent."

编排器主动清空上一次运行留下的接口数据，「to make the task run idempotent」——**重试的前提是每次 attempt 面对同样的初始条件**。Airflow 替任务清理它管得到的那份状态（XComs）；任务自己管的状态（文件、外部系统），只有任务自己能负责——这就是「幂等是任务作者的责任」在 L1 的准确含义。

工程上让任务幂等的三种形态（机制层归纳，非某文档逐字）：**天然幂等**（副作用是「置为某值」而非「追加某量」——rename/覆写天然幂等，append/转账天然不幂等）；**去重键**（副作用携带唯一键，落地前查键——append 语义也能幂等化：写前查 `row-42` 是否已在）；**原子发布**（工作可重做、发布是原子切换——`export_atomic` 的 temp+rename，也是 §4 `save_state` 与 nano-data-platform L1 物化层的同构模式）。Run B 的 `build_curated` 是第三种形态在崩溃下的实证：kill 点落在计算中，重试后产物分毫不差（check 16）。

还要把 §4(a) 的残余风险接回来：Popen 与 save_state 之间的微秒窗口里崩溃，会产生「状态不知道的子进程」→ 可能的重复执行。**幂等是这个窗口的兜底**——调度器侧把窗口压到微秒级，任务侧用幂等让「万一重复」无害。两层防御缺一不可：只压窗口不写幂等，窗口再小也是敞开的；只写幂等不压窗口，重复执行的成本（§9 的账）会教做人。

**思考题 8.1**：给 `export_append` 做一个不改 append 语义的幂等化改造（提示：去重键），并说明 Airflow 清 XComs 属于三种形态里的哪一种。（参考方向：写前查「row-42 是否已在文件中」或给行带唯一键、落地用「查键 + 条件追加」——去重键形态；XComs 清空属于「重置执行环境」，是天然幂等的变体：让每次 attempt 从同样的初始状态出发，而不是让副作用本身可重入。）

---

## §9 成本账本扩展：崩溃税

L0 的成本恒等式（`总 9 = 有效 5 + 重试救回 1 + 浪费 3`，L0 check 13）在 L1 的 Run A 逐字复现（check 07，成本单位从 toy coin 换成「1 次 subprocess 启动」——仍然不是真实云价，真实成本结构须查官方价目页 `[TODO: verify 具体价目]`，恒等式本身是机制不依赖单价）。Run B 给恒等式加了一项：

```
总成本 = 有效计算 + 重试救回 + 浪费 + 崩溃税
  10   =    5     +     1     +  3   +   1
```

**崩溃税 = 被 kill 的 attempt 消耗的启动**（check 14：恰好 1）。为什么「恰好 1」是必须精确等于 1 的预期，而不是「大概 1」？因为它是 at-least-once 语义的机器证明：**>1** 说明重启重跑了已完成任务（状态丢失，check 13 会先炸）；**0** 说明被 kill 的 attempt 没被记账（账本丢失，check 14 会炸）。等于 1 = 「崩溃只烧了它确实烧掉的那一次，其余分毫不动」——恢复的正确性被一个等式钉死。

这也给出「崩溃要不要重试」的成本视角：崩溃税与重试救回在账本上是同一类东西（都是「为最终成功多付的 attempt」），区别只在触发源（外部 kill vs 任务 exit 75）。生产里决定「宿主崩溃后是否自动续跑」的，正是这本账：续跑的代价 = 崩溃税 + 可能的幂等改造成本；不续跑的代价 = 整条运行作废 + 人工介入。**有账本，才有得选**——这是 L0 §7(b)「没有账本，你不知道重试策略是在救管线还是在烧钱」在崩溃维度的续章。

---

## §10 费曼自检

**讲给外行听**：L0 的工地总调度升级了。进度不再记在他脑子里，而是钉在工地墙上的看板（state.json）+ 一本只准往后写的值班日志（events.jsonl）——调度员下班、甚至换一个人来，看板还在，日志还在。工人也不再是他的手势，是真实的施工队（subprocess）：每队自带工牌（pid）、自带施工记录（stdout 日志），干完没干完看验收单上的章（exit code：0 = 合格，75 = 「缓一缓再来」，其他章 = 红灯不许返工）。一晚工地整体停电（kill -9 宿主死亡），恢复供电后新调度员第一件事是对看板查现场：看板上写「施工中」的队伍，现场根本没人（zombie）——记一笔「重新进场」，按墙钟排时间（不是「第 3 个 tick」，是「3 点半」）；已验收的队伍绝不重新施工（日志份数 = 进场次数，一份不多）。最重要的教训来自浇筑：混凝土没一次成型就浇两次，你会得到双倍混凝土（非幂等的副作用重复）——所以关键构件在预制场里随便重做，最后一次性吊装到位（temp + 原子 rename）：重做免费，发布原子。至于停电时正在浇筑的那一车——账本照记（崩溃税），因为那车混凝土确实烧掉了。

**思考题汇总**（正文内另有 3.1 / 4.1 / 5.1 / 7.1 / 8.1）：

1. 一句话说清：L0 → L1，进度的「住处」从哪搬到哪？新住处带来了哪两个新问题？（要点：调用栈 → 盘上；带来 stale 态问题 [状态说活着、进程已死，§5 的 zombie] 与窗口问题 [状态不知道的进程，§4(a)/§8]——状态外化是可恢复性的价格，senior 的判断力在于知道价格是多少。）
2. Run A 与 Run B 的事件顺序不同（`publish_report` 的 deny 在 A 是 seq 17、在 B 是 seq 23），终态却逐位相同——为什么这是「理应如此」而不是「碰巧」？（要点：转移由状态唯一决定，计时只影响过程不影响终态——L0 思考题 6.1 的合流性质在真实 wall-clock 下被观测到，是「重启续调和安全」的根据。）
3. 「崩溃税 = 1」为什么是精确预期而非大致预期？（要点：at-least-once 的机器证明——>1 = 重做了已完成工作 [状态丢失]，0 = 被 kill 的 attempt 没记账 [账本丢失]，恰 1 = 恢复正确。）

**反例（一个常见错误直觉）**：「崩溃续跑嘛，重启前检查输出文件在不在，在就跳过任务——这不就是幂等？」错在三个假设全部不成立：其一，**存在 ≠ 正确**——文件可能是写了一半的 torn write，存在性检查会把半成品当完成品（§4(a) 的原子写正是防这个）；其二，**存在 ≠ 全部副作用**——任务的副作用可能没有产物（通知已发、钱已付、外部 API 已调用），文件永远「不在」，跳过永远轮不到；其三，这是「产物级查重」不是「任务级幂等」——真正需要的是**重做收敛到同一结果**（天然幂等 / 去重键 / 原子发布，§8），存在性检查只在「产物完整 + 产物是唯一副作用 + 产物能代表成功」三条全真时才碰巧正确。Run C 的 `export_append` 就是反例的实体化：产物在（第一行已 append），但「跳过」会丢第二行数据，「不跳过」会重复——两条路都错，只有把任务本身改成幂等才有对的路。

---

## §11 它模拟了什么、刻意没模拟什么（L1 边界 → L2）

**模拟了**（本教程的验收内容）：真实 subprocess 任务（exit code 分类通道，75 = EX_TEMPFAIL 契约）；状态落盘（state.json 原子写 + events.jsonl append-only，seq 以日志为源）；崩溃续跑（kill -9 进程组 → zombie 识别 → 回重试通道，已完成工作不重做）；wall-clock 退避（计划等待确定 / 醒来时刻不确定，掩码设计）；幂等正面登场（非幂等副作用重复 vs 原子发布收敛，崩溃垃圾可见）；成本账本扩展（崩溃税恒等式）；L0 全部失败语义复现（终态向量 / default-deny / 止损 / 传播逐字一致）。

**刻意没模拟**（每一面都是 L2 的课题，不是遗漏）：

| 没模拟 | 为什么 L1 不做 | L2 怎么做 |
|--------|----------------|-----------|
| 孤儿收养（调度器死、子进程活）| exit code 已丢失，「完成/在跑」不可区分；L1 对此 raise 并明示（§5(c)）| heartbeat / 结果通道：liveness 与结果都不走父子关系（Airflow heartbeat 机制，§5(a) 引文同页）|
| RUNNING 连续形态 liveness 检测 | L1 故障模型下崩溃必伴重启，重启时点状检测已完备 | 周期扫描 + heartbeat 超时（Airflow zombie 处理的连续形态）|
| 真实并行 / 资源池 / 优先级 | 串行规则 C 是 L1 的确定性选择（同 L0 口径）| Airflow executor/pool、Dagster concurrency 源码对照 |
| cron 触发 / sensor | L0 §9 曾指向 L1；本级兑现其 wall-clock 面（退避/epoch/elapsed）；「何时触发一次全新运行」属触发面，与「运行内如何调和」是两层机制 | Airflow scheduler loop / triggerer 对照（口径调整在此明示）|
| SLA / backfill / trigger rules 配置化 | 独立机制面 | L2 |
| 状态后端升级（state.json → SQLite/Postgres）| 单文件 JSON 在 toy 规模下机制等价、可读性最高 | Airflow metadata DB 形态对照；原子写模式不变 |
| Agentic 自愈（ROADMAP §七 关键词）| 需要 L0–L1 全部机制为前置 | L2（agent 驱动的管线修复）|

---

## §12 溯源

| 声明 | 类型 | 来源 |
|------|------|------|
| zombie 引文「TaskInstances may get stuck in a running state despite their associated jobs being inactive (for example if the TaskInstance's worker ran out of memory). Such tasks were formerly known as zombie tasks. Airflow will find these periodically, clean them up, and mark the TaskInstance as failed or retry it if it has available retries.」（§5） | 文献已有（逐字引文；原文 TaskInstance 带 `<code>` 标记，引文为去标记文本，源页换行已并接、弯撇号已归一为直引号（承 L0 §11 口径）） | https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html ，2026-08-14 抓取（156,580 B，与 L0 08-12 录值同尺寸零漂移；h1 "Tasks"） |
| heartbeat timeout 原因句「The Airflow worker ran out of memory and was OOMKilled」（§5(c)） | 文献已有（逐字引文） | 同上（Task Instance Heartbeat Timeout 节） |
| BashOperator 引文「In general a non-zero exit code produces an AirflowException and thus a task failure.」（§3） | 文献已有（逐字引文） | https://airflow.apache.org/docs/apache-airflow-providers-standard/stable/operators/bash.html ，2026-08-14 抓取（131,531 B；h1 "BashOperator"） |
| XComs 引文「If the first task was not successful then on every retry task XComs will be cleared to make the task run idempotent.」（§8） | 文献已有（逐字引文） | https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html ，2026-08-14 抓取（118,246 B；h1 "XComs"） |
| `EX_TEMPFAIL 75 /* temp failure; user is invited to retry */`（§3） | 文献已有（逐字引文，本机文件在盘核验） | BSD 系标准头文件 sysexits.h，本机路径 /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sysexits.h:L111（5,472 B，2026-08-14 核验） |
| L0 状态机规则 A/B/C、PIPELINE/GRANTS fixture、终态向量期望值（checks 01–03）、成本恒等式（check 07）、合流性质 | 纲领/前级已有 | `L0_dag_scheduler_state_machine.py`（冻结锚 `a391f8e6…`/191 行）+ `tutorial_L0.md` §4–§7 |
| 「落数据 → 推游标」「逻辑时钟从 catalog 重建」的同构类比（§4） | 姊妹模块已有 | nano-data-platform L1 `tutorial_L1.md`（锚 `d6bf53b0…`/382 行，只读引用） |
| 「exit code 是进程边界唯一通用通道」「两种故障模型（宿主死亡 vs 孤儿）」「幂等三形态（天然/去重键/原子发布）」「stale 字段由状态守卫赋义」「崩溃税的 at-least-once 解读」 | 合理推断 | 机制层归纳 / 本教程自论证，无外部引文；POSIX exit code 约定为通用常识 |
| 全部 seq / attempts / backoffs / digest / coins 数字与 0.6s 退避基数、1.5s on-call 窗、1.2s 长任务时长 | 本实现实测（toy 设定） | `L1_subprocess_state_and_crash_recovery.py` 本次运行输出（§2 paste 块即其掩码形态）；非真实云价、时长为 demo 尺度、不可外推 |
| 双跑确定性锚 | 本实现实测 | 两个新建空独立 CWD、`python3 -B` 双跑：全 EXIT=0、stderr 0 B；raw 98 行/10,510 B（md5 因 elapsed 行不同）；掩码口径 `sed '/^[[:space:]]*elapsed/d'` 后 md5 `9e1bec41263dca2108190e0262590914`/92 行/10,139 B，RUN1==RUN2 BYTE-IDENTICAL（Python 3.13.13，2026-08-14） |

下一站：**L2**——对照权威实现源码做取舍分析：Airflow（scheduler loop / TaskInstance 状态机 / trigger rules / executor 与 pool / heartbeat-zombie 连续形态）+ Dagster（asset graph / concurrency）+ Prefect（flow run 状态）；真实并行与资源池；CI/CD 参照（GitHub Actions / GitLab CI）；Agentic 管线自愈（ROADMAP §七）；按可运行性契约允许「可运行的本质模拟 + 显式注明」（见 README 阶梯表）。
