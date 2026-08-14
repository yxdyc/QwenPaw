#!/usr/bin/env python3
"""nano-data-orchestration L1 — 真实 subprocess + 状态落盘 + 崩溃续跑 + wall-clock 退避 + 幂等正面登场。

状态机语义完全沿用 L0（依赖调和 / 错误分类 / 失败传播 / default-deny / 成本账本），换掉的是基底——
任务从纯函数变成真实进程，L0 刻意没模拟的四件事（README 阶梯行 = L1 定义）正面登场：

  [1] 任务 = 真实 subprocess：exit code 成为结果分类的通用通道
      （0 成功 / 75 = EX_TEMPFAIL transient / 其余 permanent——75 不是发明，是 BSD sysexits.h 的
      "temp failure; user is invited to retry"，见 tutorial §3 溯源）；Airflow BashOperator 同形态：
      "In general a non-zero exit code produces an AirflowException and thus a task failure."（tutorial §3）
  [2] 状态落盘：state.json（原子写）+ events.jsonl（append-only）——进度不再活在调用栈里（L0 思考题 4.1 的答案）；
      seq 编号以 events.jsonl 为源、重启重建（呼应 nano-data-platform L1「逻辑时钟从 catalog 重建」）。
  [3] 崩溃续跑：kill -9 整个进程组（宿主死亡模型）→ 重启识别 zombie（state=RUNNING 而 pid 已死）→ 回重试通道。
      Airflow 对应机制见 tutorial §6 逐字引文（zombie/heartbeat）。只模拟宿主死亡模型；孤儿收养（调度器死、
      子进程活、exit code 丢失）需要 heartbeat/结果通道 → L2（tutorial §11 边界表）。
  [4] wall-clock 退避：retry_after = epoch 时间戳；**计划等待的算术是确定的**，实际醒来时刻不确定——
      不确定量（wall-clock / pid）一律落 elapsed 掩码行或 state.json，不进 check 路径（确定性口径沿用 L0）。
  [5] 幂等正面登场：非幂等任务重试 → 副作用重复（机器可见）；原子发布（temp + rename）在重试与崩溃下都收敛。
      Airflow 在接口层把这条前提内化：重试时先清空 XComs "to make the task run idempotent"（tutorial §8 逐字引文）。

刻意不模拟：孤儿收养 / heartbeat liveness / 真实并行与资源池 / cron 触发与 sensor / SLA——L2 课题（tutorial §11）。
零依赖（纯标准库），CPU 约 9s（三 run 实测 ~8.7s，含 1.5s 固定 on-call 响应窗）；输出确定（elapsed 掩码行除外，
掩码口径 sed '/^[[:space:]]*elapsed/d'）。任意 CWD 可跑；workdir 用 tempdir、跑完自清理、路径不入输出。

用法：python3 L1_subprocess_state_and_crash_recovery.py            # demo 驱动（Run A/B/C + 24 self-checks）
      python3 L1_subprocess_state_and_crash_recovery.py sched --workdir DIR --dag pipeline|idem
                                                                   # 调度器本体（调和至全部终态；由驱动以子进程方式调用）
"""
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time

EX_TEMPFAIL = 75          # sysexits.h: "temp failure; user is invited to retry"（溯源见 tutorial §3）
BACKOFF_BASE = 0.6        # wall-clock 退避基数（秒）：计划等待 = 0.6 * 2^(k-1)，算术确定
TERMINAL = {"SUCCESS", "FAILED", "UPSTREAM_FAILED"}
ON_CALL_WINDOW = 1.5      # Run B：kill 与重启之间的固定等待（模拟 on-call 响应；兼钉死事件顺序，tutorial §6）

CHECKS = []
def check(name, cond):
    CHECKS.append(bool(cond))
    print(f"  [check {len(CHECKS):02d}] {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: raise SystemExit("self-check failed: " + name)

# ---------- DAG fixture：deps/retries/needs 逐字沿用 L0 PIPELINE；失败日程从纯函数移进 subprocess 的 exit code ----------
PIPELINE = {
    "ingest_crm":     dict(deps=[],                         retries=2),  # attempt 1 exit 75（transient 原型）
    "ingest_web":     dict(deps=[],                         retries=2),  # 永远 exit 75（故意误分类为 transient）
    "gate_crm":       dict(deps=["ingest_crm"],             retries=0),
    "gate_web":       dict(deps=["ingest_web"],             retries=0),
    "normalize_web":  dict(deps=["gate_web"],               retries=0),
    "unit_tests":     dict(deps=[],                         retries=0),
    "build_curated":  dict(deps=["gate_crm", "unit_tests"], retries=1),  # 长任务（1.2s）= Run B 的崩溃受害者
    "deploy":         dict(deps=["build_curated"],          retries=1, needs={"prod_deploy"}),
    "publish_report": dict(deps=["build_curated"],          retries=2, needs={"metrics_write"}),
}
GRANTS = {"deploy": {"prod_deploy"}}   # 最小权限同 L0：metrics_write 无人授权 -> publish_report 拒绝演练
IDEMPOTENCE_DEMO = {
    "export_append":  dict(deps=[], retries=1),  # 非幂等：副作用 = append，且失败发生在副作用之后（最坏情形）
    "export_atomic":  dict(deps=[], retries=1),  # 幂等：  工作可重做，发布 = 原子 rename
    "export_broken":  dict(deps=[], retries=2),  # permanent：exit 1（校验失败）——retries=2 也不重试
}

# 任务程序：以 [sys.executable, -B, -c] 启动；上下文经环境变量传递（NANO_TASK / NANO_ATTEMPT / NANO_WORKDIR）。
# 失败日程是显式实验设计（同 L0 fixture 声明）：按 attempt 编号决定行为，确定性可复算。
TASK_PROGRAM = r'''
import json, os, sys, time
t = os.environ["NANO_TASK"]; a = int(os.environ["NANO_ATTEMPT"]); w = os.environ["NANO_WORKDIR"]
def say(m): print(f"[{t} attempt {a}] {m}", flush=True)
def atomic_write(path, text):
    tmp = f"{path}.tmp.attempt{a}"                 # 每 attempt 独立 temp 名：崩溃残留互不覆盖（tutorial §7）
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)                          # 原子发布：读者永远看到完整旧版或完整新版
if t == "ingest_crm":
    if a == 1: say("connection reset（模拟 transient 故障）"); sys.exit(75)   # EX_TEMPFAIL
    atomic_write(os.path.join(w, "raw_crm.jsonl"),
                 "".join(json.dumps({"id": f"crm-{i:03d}", "src": "crm"}) + "\n" for i in range(1, 4)))
    say("side effect: raw_crm.jsonl 写入 3 行（原子发布）"); sys.exit(0)
if t == "ingest_web":
    say("source 拒绝所有读取（永久损坏，作者却误分类为 transient）"); sys.exit(75)
if t in ("gate_crm", "gate_web"):
    src = os.path.join(w, "raw_crm.jsonl" if t == "gate_crm" else "raw_web.jsonl")
    if not os.path.exists(src) or os.path.getsize(src) == 0:
        say("quality gate: 无数据可用（permanent）"); sys.exit(1)
    n = sum(1 for _ in open(src)); say(f"quality gate: {n} 行通过"); sys.exit(0)
if t == "normalize_web":
    say("normalize: 本 fixture 中不可达（上游是坏源）"); sys.exit(0)
if t == "unit_tests":
    time.sleep(0.1); say("unit tests: 3 passed"); sys.exit(0)
if t == "build_curated":
    time.sleep(1.2)                                # 模拟聚合计算 = 崩溃窗口（L1 的 RUNNING 是长时间停留的状态）
    rows = [json.loads(l) for l in open(os.path.join(w, "raw_crm.jsonl"))]
    atomic_write(os.path.join(w, "curated.jsonl"),
                 "".join(json.dumps({"id": r["id"], "layer": "curated"}) + "\n" for r in rows))
    say(f"side effect: curated.jsonl 写入 {len(rows)} 行（原子发布）"); sys.exit(0)
if t == "deploy":
    say("deploy: curated 版本上线（行使 prod_deploy 能力）"); sys.exit(0)
if t == "publish_report":
    say("report 已发布"); sys.exit(0)
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
if t == "export_broken":
    say("schema 校验失败（permanent：重试多少次同果）"); sys.exit(1)
say(f"unknown task {t}"); sys.exit(2)
'''

# ---------- DAG 校验：与 L0 同款（结构错误死在执行前）----------
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

# ---------- 状态落盘：state.json 原子写 + events.jsonl append-only（进度活在盘上，不在调用栈里）----------
def state_path(wd): return os.path.join(wd, "state.json")
def events_path(wd): return os.path.join(wd, "events.jsonl")

def save_state(wd, st):
    tmp = state_path(wd) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False, sort_keys=True, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, state_path(wd))                # 原子替换 = 调度器层的幂等发布（与任务的 atomic_write 同构）

def load_state(wd):
    if not os.path.exists(state_path(wd)): return None
    with open(state_path(wd)) as f: return json.load(f)

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

def pid_alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True            # 活着但不是我们的进程

def state_digest(st):                              # 逻辑投影：只看状态向量（不含 attempts/pid/时间戳）
    body = json.dumps({n: t["state"] for n, t in st["tasks"].items()}, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()[:16]

def attempts_digest(st):
    body = json.dumps({n: t["attempts"] for n, t in st["tasks"].items()}, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()[:16]

# ---------- 规则 C 的执行体：L0 的 spec["fn"](attempt) 换成真实 subprocess ----------
def launch_and_wait(name, spec, grants, st, wd):
    t = st["tasks"][name]
    missing = sorted(spec.get("needs", set()) - grants.get(name, set()))
    if missing:                                    # default-deny：拒绝发生在 Popen 之前——0 attempt 0 成本
        t["state"] = "FAILED"
        append_event(wd, name, "FAILED(deny)",
                     f"capability missing (default-deny): {missing} —— permanent 不重试，0 attempt 0 成本，subprocess 从未启动", 0)
        return
    t["attempts"] += 1; a = t["attempts"]
    os.makedirs(os.path.join(wd, "logs"), exist_ok=True)
    log_name = f"{name}.attempt{a}.log"
    log_f = open(os.path.join(wd, "logs", log_name), "w")
    env = dict(os.environ, NANO_TASK=name, NANO_ATTEMPT=str(a), NANO_WORKDIR=wd)
    t["state"] = "RUNNING"; t["started_at"] = time.time()
    p = subprocess.Popen([sys.executable, "-B", "-c", TASK_PROGRAM],
                         stdout=log_f, stderr=subprocess.STDOUT, env=env)
    t["pid"] = p.pid
    save_state(wd, st)                             # 先落 pid 再 wait——崩溃识别依赖这一步（tutorial §5 的窗口分析）
    append_event(wd, name, "RUNNING", f"attempt {a} subprocess 已启动（pid 已落 state.json；输出 -> logs/{log_name}）", a)
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
    else:
        t["state"] = "FAILED"
        append_event(wd, name, "FAILED", f"attempt {a} exit=75 —— 重试上限 ({spec['retries']}) 耗尽，止损——上限是误分类的最后防线", a)

# ---------- 重启识别 zombie：state=RUNNING 而 pid 已死（宿主死亡模型；Airflow 对应 heartbeat/zombie 机制）----------
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
        else:
            t["state"] = "FAILED"
            append_event(wd, name, "FAILED", "zombie 识别: state=RUNNING 而录 pid 已死——重试上限耗尽，止损", t["attempts"])

# ---------- 调和循环：L0 规则 A/B/C 的 wall-clock 版（规则本身逐字同构）----------
def reconcile_until_terminal(wd, tasks, grants, poll=0.05):
    topo_validate(tasks)
    st = load_state(wd)
    if st is None:
        st = {"dag": wd, "tasks": {n: {"state": "PENDING", "attempts": 0, "backoffs": [], "retry_after": None,
                                       "pid": None, "exit_code": None, "started_at": None} for n in tasks}}
        save_state(wd, st)
    recover_zombies(tasks, st, wd)                 # 首次启动是 no-op；重启时识别 zombie
    while True:
        if all(st["tasks"][n]["state"] in TERMINAL for n in tasks): break
        now = time.time(); changed = 0
        for n in sorted(tasks):                                # 规则 A：RETRYING 唤醒——比较对象从逻辑 tick 换成 epoch
            t = st["tasks"][n]
            if t["state"] == "RETRYING" and t["retry_after"] <= now:
                t["state"] = "RUNNABLE"; changed += 1
        for n in sorted(tasks):                                # 规则 B：依赖解析（与 L0 逐字同构）
            t = st["tasks"][n]
            if t["state"] != "PENDING": continue
            ds = [st["tasks"][d]["state"] for d in tasks[n]["deps"]]
            if any(s in ("FAILED", "UPSTREAM_FAILED") for s in ds):
                t["state"] = "UPSTREAM_FAILED"; changed += 1
                append_event(wd, n, "UPSTREAM_FAILED", "上游终态失败——依赖是承诺：不在坏数据上跑", 0)
            elif all(s == "SUCCESS" for s in ds):
                t["state"] = "RUNNABLE"; changed += 1
        for n in sorted(tasks):                                # 规则 C：串行执行（真实并行 / 资源池 -> L2）
            if st["tasks"][n]["state"] == "RUNNABLE":
                launch_and_wait(n, tasks[n], grants, st, wd); changed += 1
                break
        save_state(wd, st)
        if changed == 0:
            if not any(t["state"] in ("RETRYING", "RUNNING") for t in st["tasks"].values()):
                raise RuntimeError("死锁：无变化且无等待——校验应已排除此情形")
            future = [t["retry_after"] for t in st["tasks"].values() if t["state"] == "RETRYING"]
            time.sleep(min([poll] + [max(0.0, r - time.time()) for r in future]))

# ---------- demo 驱动 ----------
def start_scheduler(wd, dag):
    return subprocess.Popen([sys.executable, "-B", os.path.abspath(__file__), "sched", "--workdir", wd, "--dag", dag],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)

def run_scheduler(wd, dag):
    p = start_scheduler(wd, dag); out, _ = p.communicate(); return p.returncode, out

def read_events(wd):
    with open(events_path(wd)) as f: return [json.loads(l) for l in f if l.strip()]

def log_files(wd, name):
    d = os.path.join(wd, "logs")
    if not os.path.isdir(d): return []
    return sorted(f for f in os.listdir(d) if re.fullmatch(re.escape(name) + r"\.attempt\d+\.log", f))

def main():
    print("== nano-data-orchestration L1: 真实 subprocess + 状态落盘 + 崩溃续跑 + wall-clock 退避 + 幂等正面登场 ==")
    print("  （L0 状态机语义不变，基底换成真实进程：exit code 是分类通道，state.json 是 single source of truth）")
    t0 = time.time()
    WA = tempfile.mkdtemp(prefix="nano_orch_l1_A_")
    WB = tempfile.mkdtemp(prefix="nano_orch_l1_B_")
    WC = tempfile.mkdtemp(prefix="nano_orch_l1_C_")
    try:
        # ---------------- [1] Run A：干净基线 ----------------
        print("\n[1] Run A 干净基线：L0 fixture 与失败语义在真实 subprocess 上复现")
        tA = time.time()
        rcA, outA = run_scheduler(WA, "pipeline")
        print("  --- 调度器事件流（Run A，调度器为独立子进程，此处为其 stdout 原文）---")
        print(outA, end="")
        stA, evA = load_state(WA), read_events(WA)
        vec = {s: sorted(n for n in stA["tasks"] if stA["tasks"][n]["state"] == s) for s in sorted(TERMINAL)}
        print(f"  终态向量: SUCCESS={vec['SUCCESS']}\n            FAILED={vec['FAILED']}  UPSTREAM_FAILED={vec['UPSTREAM_FAILED']}")
        check("终态向量与 L0 逐字一致: 5 SUCCESS（含重试救回的 ingest_crm）",
              vec["SUCCESS"] == ["build_curated", "deploy", "gate_crm", "ingest_crm", "unit_tests"])
        check("2 FAILED（ingest_web 止损 / publish_report 被拒）", vec["FAILED"] == ["ingest_web", "publish_report"])
        check("2 UPSTREAM_FAILED（坏源的爆炸半径）", vec["UPSTREAM_FAILED"] == ["gate_web", "normalize_web"])
        crm_a1 = [e for e in evA if e["task"] == "ingest_crm" and e["attempt"] == 1 and e["to"] == "RETRYING"]
        web_75 = [e for e in evA if e["task"] == "ingest_web" and "exit=75" in e["why"]]
        check("exit code 分类通道: ingest_crm attempt1 exit=75 -> 重试；ingest_web 三次 exit=75 全部在案",
              rcA == 0 and len(crm_a1) == 1 and "exit=75" in crm_a1[0]["why"] and len(web_75) == 3)
        check("wall-clock 退避算术确定: 计划等待 ingest_crm [0.60] / ingest_web [0.60, 1.20]（= 0.6*2^(k-1)）",
              stA["tasks"]["ingest_crm"]["backoffs"] == [0.6] and stA["tasks"]["ingest_web"]["backoffs"] == [0.6, 1.2])
        check("default-deny: publish_report attempts==0 且无日志文件——subprocess 从未启动",
              stA["tasks"]["publish_report"]["attempts"] == 0 and log_files(WA, "publish_report") == [])
        coins = sum(t["attempts"] for t in stA["tasks"].values())
        eff = sum(1 for t in stA["tasks"].values() if t["state"] == "SUCCESS")
        wasted = sum(t["attempts"] for t in stA["tasks"].values() if t["state"] == "FAILED")
        crash_tax = len([e for e in evA if e["why"].startswith("zombie 识别")])
        recovery = sum(t["attempts"] - 1 for t in stA["tasks"].values() if t["state"] == "SUCCESS") - crash_tax
        print(f"  成本账本（成本单位 = 1 次 subprocess 启动）: 总 {coins} = 有效 {eff} + 重试救回 {recovery} + 浪费 {wasted} + 崩溃税 {crash_tax}")
        check("成本恒等式 9 = 5 + 1 + 3 + 0（复现 L0 check 13）", (coins, eff, recovery, wasted, crash_tax) == (9, 5, 1, 3, 0))
        recon = {}
        for e in evA: recon[e["task"]] = "FAILED" if e["to"] == "FAILED(deny)" else e["to"]
        check("state.json 与事件流收敛到同一终态；seq 1..N 单调无 gap；调度器干净退出",
              recon == {n: t["state"] for n, t in stA["tasks"].items()}
              and [e["seq"] for e in evA] == list(range(1, len(evA) + 1)))
        digA = state_digest(stA)
        print(f"  终态向量 digest: {digA}（Run B 崩溃续跑后必须收敛到同一值）")
        print(f"  elapsed: Run A wall-clock {time.time() - tA:.2f}s（墙钟不确定量 -> 掩码行）")

        # ---------------- [2] Run B：kill -9 进程组（宿主死亡模型）-> 重启续调和 ----------------
        print("\n[2] Run B 崩溃续跑：kill -9 整个进程组（宿主死亡模型）-> 重启从盘上继续调和")
        tB = time.time()
        proc = start_scheduler(WB, "pipeline")
        while True:                                            # 驱动侧等待「build_curated 进入 RUNNING」这一确定逻辑点
            st_mid = load_state(WB)
            if st_mid is not None and st_mid["tasks"]["build_curated"]["state"] == "RUNNING": break
            if proc.poll() is not None: raise RuntimeError("调度器在 build_curated 进入 RUNNING 前退出: " + proc.stdout.read())
            time.sleep(0.02)
        snap = {n: t["state"] for n, t in sorted(st_mid["tasks"].items())}
        victim_pid = st_mid["tasks"]["build_curated"]["pid"]
        print("  kill 点快照（state.json 读值）: " + " ".join(f"{k}={v}" for k, v in snap.items()))
        check("kill 点 = 确定逻辑点: build_curated RUNNING（attempt 1 执行中），4 任务已终态，ingest_web 已唤醒待串行槽（RUNNABLE）",
              snap == {"build_curated": "RUNNING", "deploy": "PENDING", "gate_crm": "SUCCESS", "gate_web": "PENDING",
                       "ingest_crm": "SUCCESS", "ingest_web": "RUNNABLE", "normalize_web": "PENDING",
                       "publish_report": "PENDING", "unit_tests": "SUCCESS"})
        os.killpg(proc.pid, signal.SIGKILL)                    # 宿主死亡模型：调度器与它的 subprocess 同组俱灭
        rc_kill = proc.wait(); proc.stdout.read()
        while pid_alive(victim_pid): time.sleep(0.01)
        print(f"  elapsed: kill 点 @ Run B 调度器启动 +{time.time() - tB:.2f}s（build_curated attempt 1 执行中）")
        st_stale = load_state(WB)
        check("kill -9 后: 调度器返回 -SIGKILL；state.json 完整在盘，build_curated 留下 stale 态（RUNNING + 录 pid）",
              rc_kill == -signal.SIGKILL and st_stale["tasks"]["build_curated"]["state"] == "RUNNING"
              and isinstance(st_stale["tasks"]["build_curated"]["pid"], int))
        print(f"  盘上 stale 态（build_curated 行投影）: state=RUNNING attempts={st_stale['tasks']['build_curated']['attempts']} "
              f"pid=<已录, 已死> exit_code=None —— 重启的第一件事是识别它")
        time.sleep(ON_CALL_WINDOW)                             # 模拟 on-call 响应窗（固定常数；兼钉死事件顺序，tutorial §6）
        tR = time.time()
        rcB, outB = run_scheduler(WB, "pipeline")              # 重启：同一 workdir、同一命令——状态从盘上继续
        print(f"  elapsed: on-call 响应窗 {ON_CALL_WINDOW:.2f}s（固定常数）+ 重启调度 {time.time() - tR:.2f}s")
        print("  --- 调度器事件流（Run B 重启后，stdout 原文；seq 接续崩溃前编号）---")
        print(outB, end="")
        stB, evB = load_state(WB), read_events(WB)
        check("zombie 识别: state=RUNNING 而录 pid 已死 -> 归 transient 回重试通道（计划退避 0.60s）",
              any(e["task"] == "build_curated" and e["why"].startswith("zombie 识别") for e in evB)
              and stB["tasks"]["build_curated"]["backoffs"] == [0.6])
        check("终态向量 == Run A（同一 digest，崩溃不改变收敛点）", rcB == 0 and state_digest(stB) == digA)
        check("已完成的工作不重做: 全部任务日志文件数 == attempts（重启不给已完成任务追加启动）",
              all(len(log_files(WB, n)) == t["attempts"] for n, t in stB["tasks"].items()))
        coinsB = sum(t["attempts"] for t in stB["tasks"].values())
        check("崩溃税 = 恰好 1: Run B coins 10 = Run A 9 + 1（被 kill 的 attempt 也烧了钱）", coinsB == 10 and coinsB == coins + 1)
        b1 = os.path.join(WB, "logs", "build_curated.attempt1.log")
        b2 = os.path.join(WB, "logs", "build_curated.attempt2.log")
        check("被 kill 的 attempt 零输出零产物: attempt1 日志 0 字节；attempt2 完整（含副作用行）",
              os.path.getsize(b1) == 0 and "side effect: curated.jsonl" in open(b2).read())
        curated = [json.loads(l) for l in open(os.path.join(WB, "curated.jsonl"))]
        check("原子发布在崩溃下收敛: curated.jsonl 恰 3 行且全部 layer=curated（kill 点落在计算中，未发布）",
              len(curated) == 3 and all(r["layer"] == "curated" for r in curated))
        check("崩溃前副作用不受崩溃影响: raw_crm.jsonl 与 Run A 逐字节一致",
              open(os.path.join(WB, "raw_crm.jsonl"), "rb").read() == open(os.path.join(WA, "raw_crm.jsonl"), "rb").read())
        check("events.jsonl 跨两个调度器进程 seq 单调无 gap: 历史不被擦除，编号不重启",
              [e["seq"] for e in evB] == list(range(1, len(evB) + 1)))
        print(f"  elapsed: Run B wall-clock {time.time() - tB:.2f}s（含 kill + on-call 窗 + 续跑）")

        # ---------------- [3] Run C：幂等正面登场 ----------------
        print("\n[3] Run C 幂等正面登场：同一调度器、同一重试策略，任务侧幂等与否决定副作用是否重复")
        tC = time.time()
        rcC, outC = run_scheduler(WC, "idem")
        print("  --- 调度器事件流（Run C，stdout 原文）---")
        print(outC, end="")
        stC = load_state(WC)
        rep_append = open(os.path.join(WC, "report_append.txt")).read()
        rep_atomic = open(os.path.join(WC, "report_atomic.txt")).read()
        print(f"  report_append.txt（非幂等: 副作用=append，失败在副作用之后）= {rep_append!r} -> 2 行逐字重复")
        print(f"  report_atomic.txt（幂等: 工作可重做 + 原子发布）          = {rep_atomic!r} -> 恰 1 行")
        check("非幂等任务重试 = 副作用重复: report_append.txt 恰 2 行逐字相同（attempts=2，重试救回但副作用加倍）",
              rep_append == "row-42\nrow-42\n" and stC["tasks"]["export_append"]["attempts"] == 2)
        check("幂等任务重试 = 收敛: report_atomic.txt 恰 1 行内容正确（attempts=2，重试无害）",
              rep_atomic == "row-42\n" and stC["tasks"]["export_atomic"]["attempts"] == 2)
        orphan = os.path.join(WC, "report_atomic.txt.tmp.attempt1")
        check("崩溃垃圾可见: attempt1 的 temp 文件失去主人仍在盘（无害但需 GC——真实系统的 orphan 清理课题）",
              os.path.exists(orphan) and open(orphan).read() == "row-42\n")
        check("permanent 不重试: export_broken exit=1，retries=2 也 attempts==1 立即止损",
              stC["tasks"]["export_broken"]["attempts"] == 1 and stC["tasks"]["export_broken"]["state"] == "FAILED")
        print(f"  elapsed: Run C wall-clock {time.time() - tC:.2f}s")

        # ---------------- [4] 跨 run 不变量 ----------------
        print("\n[4] 跨 run 不变量：收敛点相同，代价账不同")
        order = []
        for e in evB:
            if e["to"] == "RUNNING" and e["task"] not in order: order.append(e["task"])
        check("拓扑不变式（Run B 事件流）: 每个跑过的任务都在全部上游之后首启",
              all(order.index(d) < order.index(n) for n in PIPELINE for d in PIPELINE[n]["deps"] if n in order))
        check("状态持久化往返: 重读 state.json 重算 digest 不变（逻辑投影序列化稳定）",
              state_digest(load_state(WB)) == digA and attempts_digest(stA) != attempts_digest(stB))
        print(f"  终态向量 digest: Run A = {digA}  Run B = {state_digest(stB)}（相同）")
        print(f"  attempts digest: Run A = {attempts_digest(stA)}  Run B = {attempts_digest(stB)}（不同 = 崩溃税，正确）")
        print(f"  elapsed: 总 wall-clock {time.time() - t0:.2f}s")
    finally:
        for w in (WA, WB, WC): shutil.rmtree(w, ignore_errors=True)
    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sched":
        ap = argparse.ArgumentParser()
        ap.add_argument("sched"); ap.add_argument("--workdir", required=True); ap.add_argument("--dag", required=True)
        args = ap.parse_args()
        dags = {"pipeline": (PIPELINE, GRANTS), "idem": (IDEMPOTENCE_DEMO, {})}
        reconcile_until_terminal(args.workdir, *dags[args.dag])
    else:
        main()
