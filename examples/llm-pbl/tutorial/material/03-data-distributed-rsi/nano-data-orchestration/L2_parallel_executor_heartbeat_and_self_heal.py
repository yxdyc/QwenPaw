#!/usr/bin/env python3
"""nano-data-orchestration L2 — 并行 executor + 资源池 + trigger rules + heartbeat/zombie/孤儿收养 + 并发键 + Agentic 自愈。

L0 裸出状态机内核，L1 把「真实进程 / 持久化 / wall-clock」请回来。L2 回答 L1 §11 边界表留下的每一笔债——
做法不是功能罗列，而是把三个权威系统的机制本质裸成可跑代码，并逐条对照源码说明「它那样选、我这样模拟」：

  [1] 并行 executor 与资源池 —— L1 的规则 C 是串行（确定性选择）；L2 放开并行：
      RUNNABLE →（pool 空槽）→ QUEUED →（executor open_slots）→ RUNNING。
      Airflow 同构：open_slots = parallelism - len(running)（base_executor.py:L350，heartbeat() 是「触发新任务的时刻」L348-358）；
      Pool.occupied = running + queued（pool.py:L269-316），slots=-1 → 无限（pool.py:L209 float("inf")）；
      出队按 priority 排序（base_executor.py:L428-442 sorted by priority_weight, reverse=False——nano 约定「权重小者先出」，
      Airflow 的权重方向语义由其 priority 计算族治理，此处只引排序事实）。
      Dagster 的对应物是 executor max_concurrent（executor_definition.py:L426 "tells the execution engine how many
      processes may run"）+ 并发键 claim/release（见 [4]）。
  [2] trigger rules 配置化 —— L0/L1 的依赖语义是写死的 all_success；L2 把「上游终态向量 → 本任务命运」抽成可配置规则，
      语义逐条对照 Airflow TriggerRuleDep 的分派表（trigger_rule_dep.py:L429-488 flag 分支 + L507-566 等待分支）。
      新终态 SKIPPED 随之登场（branching 未选中分支）；注意 Airflow 里 SKIPPED ∈ success_states（state.py:L222-224）
      但 ALL_SUCCESS 仍会被 skipped 上游染成 SKIPPED（trigger_rule_dep.py:L433-434）——统计口径与调度口径不是一回事。
  [3] heartbeat 连续形态 + 孤儿收养 —— L1 的 zombie 检测是重启时点状（宿主死亡模型：killpg 全组俱灭）；L2 请回两笔债：
      a) 连续形态：周期扫描 RUNNING 的 last_heartbeat，超时即处置（Airflow _find_and_purge_task_instances_without_heartbeats：
         state ∈ {RUNNING, RESTARTING} 且 last_heartbeat_at < now - timeout，scheduler_job_runner.py:L3516-3519，
         周期 10s 定时器 L1723-1726）；进程还活着但心跳停 = stuck，杀之重试（对应「kill externally」路径，
         base_executor.py:L456-458 注释）。
      b) 孤儿收养：调度器死、子进程活——L1 对此只能 raise。Airflow 的答案：liveness 与结果都不走父子关系——
         心跳走 DB（Job.latest_heartbeat，job.py:L100 定义/L141 更新/L168 使用），调度器死亡由心跳超时判定（scheduler_job_runner.py:L3269
         Job.latest_heartbeat < now - timeout → 标 FAILED），孤儿按 adoptable_states（state.py:L229-231）行锁抢占
         （L3300-3303 skip_locked）后 try_adopt（L3310-3311）：能收养则 queued_by_job_id 换成新调度器（L3328），
         不能则 state=None 回炉重调（L3322）。nano 在 [7] 用真实 subprocess 复现：杀调度器不杀子进程，
         重启后靠「pid 存活 + heartbeat 文件新鲜」收养、「原子结果文件」验收——exit code 不再是唯一结果通道。
  [4] 并发键（Dagster concurrency）—— claim/release 走槽位账本：available = slot_count - pending - launched - in_progress
      （op_concurrency_limits_counter.py:L220-225），全部根键都满才阻塞出队（L230-231）；claim 失败的退避是指数步进
      1 + (1.1^n - 1)、上限 15s（instance_concurrency_context.py:L178-189）——与 L1 的指数退避同族；
      上下文退出只释放 pending claims，不释放已持有槽位（因为进程可能还在跑，L28-31 docstring）——L1「崩溃垃圾」的同款诚实。
  [5] CI/CD 参照（不锁定）—— GitHub Actions concurrency group + cancel-in-progress（"ensure that only a single job or
      workflow using the same concurrency group will run at a time" / "cancel any currently running job"，docs.github.com
      workflow-syntax 页）；GitLab CI resource_group（"Limit job concurrency"）与 interruptible（"can be canceled when made
      redundant by a newer run"）只作对照（process mode 表：unordered/oldest_first/newest_first，docs.gitlab.com/ci/resource_groups/）。
  [6] Agentic 管线自愈（本课程约定）—— healer = 观察者（结构化事件日志）+ 诊断（playbook 模式匹配）+ 执行器（白名单 DAG patch）。
      安全边界 first-class：capability 缺失的修复动作 = escalate_to_human——default-deny 不可被 agent 自动授权绕过（上述原则）。
      真实系统里 LLM 坐在 diagnose 的位置增加泛化，机制骨架（观察→诊断→白名单行动→复验）不变；此处用确定性策略裸出骨架。

本课程可运行性契约：runs [1]–[6] = 逻辑时钟下的本质模拟（确定、可复现、字节级可锚），显式注明
「模拟 Airflow/Dagster 机制，真实集群路径须另行验证」；run [7] = 真实 subprocess 并行 + 真实 kill + 真实收养，
wall-clock 不确定量落 elapsed 掩码行（口径承 L1：sed '/^[[:space:]]*elapsed/d'）。零依赖（纯标准库），CPU ~6s。

跨级锚：run [1] 以 parallelism=1 退化运行 L1 逐字同款 PIPELINE/GRANTS fixture，终态向量 digest 必须复现
L1 录值 ac4a0b3ac09bf47b（L1 state_digest 公式逐字同款），成本恒等式 9=5+1+3+0 复现 L0 check 13 / L1 check 08。

用法：python3 L2_parallel_executor_heartbeat_and_self_heal.py            # demo 驱动（runs [1]–[7] + self-checks）
      python3 L2_parallel_executor_heartbeat_and_self_heal.py sched --workdir DIR
                                                                          # 真进程调度器本体（run [7] 由驱动以子进程方式调用）
"""
import argparse, copy, hashlib, json, os, subprocess, sys, tempfile, time, shutil, signal

CHECKS = []
def check(name, cond):
    CHECKS.append(bool(cond))
    print(f"  [check {len(CHECKS):02d}] {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: raise SystemExit("self-check failed: " + name)

def state_digest(tasks):                       # 与 L1 state_digest 逐字同款（逻辑投影：只看状态向量）
    body = json.dumps({n: t["state"] for n, t in tasks.items()}, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()[:16]

def attempts_digest(tasks):
    body = json.dumps({n: t["attempts"] for n, t in tasks.items()}, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()[:16]

# ---------- L1 fixture 逐字沿用（deps/retries/needs 同款；失败日程从 exit code 移回脚本层）----------
PIPELINE = {
    "ingest_crm":     dict(deps=[],                         retries=2),
    "ingest_web":     dict(deps=[],                         retries=2),
    "gate_crm":       dict(deps=["ingest_crm"],             retries=0),
    "gate_web":       dict(deps=["ingest_web"],             retries=0),
    "normalize_web":  dict(deps=["gate_web"],               retries=0),
    "unit_tests":     dict(deps=[],                         retries=0),
    "build_curated":  dict(deps=["gate_crm", "unit_tests"], retries=1),
    "deploy":         dict(deps=["build_curated"],          retries=1, needs={"prod_deploy"}),
    "publish_report": dict(deps=["build_curated"],          retries=2, needs={"metrics_write"}),
}
GRANTS = {"deploy": {"prod_deploy"}}           # publish_report 需要的 metrics_write 无人授权 → default-deny 演练

# 脚本化进程层（逻辑 runs 的「真实进程替身」——时长/失败日程/心跳模式都是显式实验设计，确定性可复算）
def sim_plan(name):
    P = {"duration": 1.0, "fails": {}, "hb_stop_attempt": None, "dies_at_attempt": None, "signature": ""}
    if name == "ingest_crm":  P["fails"] = {1: "transient"}                     # attempt1 transient，重试救回（同 L1）
    if name == "ingest_web":  P["fails"] = {1: "transient", 2: "transient", 3: "transient"}  # 永远 transient（误分类坏源）
    if name == "gate_web":    P["fails"] = {1: "permanent"}                     # 无数据（不会跑：急切传播先至）
    if name == "build_curated": P["duration"] = 2.0                             # 长任务（L1 崩溃窗口的逻辑对应）
    return P

TERMINAL = {"SUCCESS", "FAILED", "UPSTREAM_FAILED", "SKIPPED", "CANCELLED"}
FAILED_SET = {"FAILED", "UPSTREAM_FAILED"}     # = Airflow State.failed_states（state.py:L215-217）
HB_TIMEOUT = 2.0                               # 逻辑层心跳超时（tick）；真进程层用 0.6s（run [7]）

# ---------- trigger rule：上游终态向量 → 本任务命运（语义对照 trigger_rule_dep.py:L429-488 / L507-566）----------
def trigger_decision(rule, c, done, total):
    """c = {success, failed, upstream_failed, skipped} 计数；返回 run / wait / skip / upstream_failed。
    ALL_SUCCESS:  任何失败 → UPSTREAM_FAILED；任何 skipped → SKIPPED（L430-434）
    ALL_DONE:     等全部终态，然后无条件跑（L557-566 只等待）
    ONE_SUCCESS:  ≥1 成功即跑；全 done 无成功 → UPSTREAM_FAILED；全 skipped → SKIPPED（L441-447）
    NONE_FAILED:  失败 → UPSTREAM_FAILED；skipped 容忍（L454-456）
    NONE_FAILED_MIN_ONE_SUCCESS: 失败 → UPSTREAM_FAILED；全 skipped → SKIPPED；done 且 0 成功 → UPSTREAM_FAILED（L457-463）
    ALWAYS:       不等上游（L113 短路）"""
    f = c["failed"] + c["upstream_failed"]
    if total == 0 or rule == "always": return "run"          # 无上游任务 = 根任务，直接跑（Airflow 根任务不等待）
    if rule == "all_success":
        if f:                 return "upstream_failed"
        if c["skipped"]:      return "skip"
        return "run" if c["success"] == total else "wait"
    if rule == "all_done":    return "run" if done == total else "wait"
    if rule == "one_success":
        if c["success"]:      return "run"
        if done == total:     return "skip" if c["skipped"] == total else "upstream_failed"
        return "wait"
    if rule == "none_failed":
        if f:                 return "upstream_failed"
        return "run" if done == total else "wait"
    if rule == "none_failed_min_one_success":
        if f:                 return "upstream_failed"
        if c["skipped"] == total: return "skip"
        if done == total:     return "run" if c["success"] else "upstream_failed"
        return "wait"
    raise ValueError(f"unknown trigger_rule: {rule}")

# ---------- 并发键账本（Dagster 机制的本质模拟：claim/release + 指数步进退避）----------
class KeyLedger:
    def __init__(self, slots):                 # slots: {key: n}（对照 slot_count，op_concurrency_limits_counter.py:L220-225）
        self.slots, self.held, self.claims, self.releases, self.max_held = slots, {}, 0, 0, 0
    def claim(self, key, task):
        held = self.held.setdefault(key, set())
        if len(held) < self.slots[key]:
            held.add(task); self.claims += 1; self.max_held = max(self.max_held, len(held)); return True
        return False                           # 满 → 调用方记退避（1 + (1.1^n - 1)，上限 15，对照 L178-189）
    def release(self, key, task):
        if key in self.held and task in self.held[key]:
            self.held[key].discard(task); self.releases += 1

# ---------- 逻辑层调度器：L0/L1 状态机的并行版（规则 A/B 语义不变，规则 C 换成 executor 出队）----------
class Scheduler:
    def __init__(self, spec, grants=None, parallelism=1, pools=None, keys=None, scheduler_id=1, tasks=None):
        self.spec, self.grants, self.parallelism = spec, grants or {}, parallelism
        self.pools = pools or {}               # {pool_name: slots}（slots=-1 → 无限，对照 pool.py:L209）
        self.keys = keys or KeyLedger({})
        self.sid, self.t, self.seq = scheduler_id, 0.0, 0
        self.events, self.dequeue_log, self.pool_ledger = [], [], []
        self.tasks = tasks or {n: dict(state="PENDING", attempts=0, backoffs=[], retry_after=None, queued_by=None,
                                       last_hb=None, t_start=None, hb_stop=None, dies_at=None, killed=False,
                                       pool=spec[n].get("pool"), priority=spec[n].get("priority", 0),
                                       trigger_rule=spec[n].get("trigger_rule", "all_success"),
                                       key=spec[n].get("key"), claim_retry=None, claim_backoffs=[],
                                       branch_of=spec[n].get("branch_of"), pool_wait_announced=False) for n in spec}
        self._validate()

    def add_tasks(self, extra):                # 运行中注入新 run 的任务（[5] r2 在 t=1 提交；DAG 动态扩展的玩具形态）
        self.spec.update(extra)
        for n in extra:
            self.tasks[n] = dict(state="PENDING", attempts=0, backoffs=[], retry_after=None, queued_by=None,
                                 last_hb=None, t_start=None, hb_stop=None, dies_at=None, killed=False,
                                 pool=extra[n].get("pool"), priority=extra[n].get("priority", 0),
                                 trigger_rule=extra[n].get("trigger_rule", "all_success"),
                                 key=extra[n].get("key"), claim_retry=None, claim_backoffs=[],
                                 branch_of=extra[n].get("branch_of"), pool_wait_announced=False)
        self._validate()

    def _validate(self):                       # 与 L0/L1 topo_validate 同款：结构错误死在执行前
        for n, s in sorted(self.spec.items()):
            for d in s["deps"]:
                if d not in self.spec: raise ValueError(f"unknown dep '{d}' of '{n}'")
        indeg = {n: len(s["deps"]) for n, s in self.spec.items()}
        order, ready = [], sorted(n for n, k in indeg.items() if k == 0)
        while ready:
            n = ready.pop(0); order.append(n)
            for m in sorted(self.spec):
                if n in self.spec[m]["deps"]:
                    indeg[m] -= 1
                    if indeg[m] == 0: ready.append(m)
            ready.sort()
        if len(order) != len(self.spec): raise ValueError(f"cycle detected: {sorted(set(self.spec) - set(order))}")

    def event(self, task, to, why):
        self.seq += 1
        self.events.append({"seq": self.seq, "t": round(self.t, 2), "task": task, "to": to, "why": why,
                            "attempt": self.tasks[task]["attempts"]})
        print(f"  [seq {self.seq:02d}] t={self.t:<4.1f} {task:18s} -> {to:16s} {why}")

    def pool_occupied(self, p):                # occupied = QUEUED + RUNNING（对照 Pool.occupied_slots，pool.py:L269-316）
        return sum(1 for t in self.tasks.values() if t["pool"] == p and t["state"] in ("QUEUED", "RUNNING"))
    def pool_open(self, p):
        slots = self.pools[p]
        return (slots == -1) or (self.pool_occupied(p) < slots)
    def open_slots(self):                      # open_slots = parallelism - len(running)（base_executor.py:L350）
        return self.parallelism - sum(1 for t in self.tasks.values() if t["state"] == "RUNNING")

    def _sim(self, name):
        s = self.spec[name]; t = self.tasks[name]; p = sim_plan(name)
        p = dict(p, **{k: v for k, v in s.items() if k in ("duration", "fails", "hb_stop_attempt", "dies_at_attempt", "signature")})
        return p

    def _launch(self, name):
        t = self.tasks[name]
        missing = sorted(self.spec[name].get("needs", set()) - self.grants.get(name, set()))
        if missing:                            # default-deny：拒绝发生在启动前——0 attempt 0 成本（L0/L1 同款）
            t["state"] = "FAILED"
            self.event(name, "FAILED(deny)", f"capability missing (default-deny): {missing} —— 0 attempt 0 成本")
            return
        t["attempts"] += 1; t["state"] = "RUNNING"; t["t_start"] = self.t; t["last_hb"] = self.t
        t["queued_by"] = self.sid; t["killed"] = False
        p = self._sim(name)
        t["hb_stop"] = (self.t + 1.0) if p["hb_stop_attempt"] == t["attempts"] else None
        t["dies_at"] = (self.t + 1.0) if p["dies_at_attempt"] == t["attempts"] else None
        self.event(name, "RUNNING", f"attempt {t['attempts']} 启动（queued_by=sched#{self.sid}）")

    def _finish(self, name, how):
        t = self.tasks[name]; p = self._sim(name)
        if t["key"]: self.keys.release(t["key"], name)
        fail = p["fails"].get(t["attempts"]) if how == "complete" else "transient"
        if how == "complete" and fail is None:
            t["state"] = "SUCCESS"
            self.event(name, "SUCCESS", f"attempt {t['attempts']} 完成" + ("  <- 重试救回" if t["attempts"] > 1 else ""))
        elif fail == "transient" or how != "complete":
            if t["attempts"] <= self.spec[name]["retries"]:
                wait = 2.0 ** (t["attempts"] - 1)          # 逻辑退避：1,2,4… tick（L1 wall-clock 0.6*2^(k-1) 的逻辑对应）
                t["state"] = "RETRYING"; t["retry_after"] = self.t + wait; t["backoffs"].append(wait)
                why = (f"attempt {t['attempts']} " + (f"exit=75 transient（{p['signature']}）" if how == "complete"
                       else "被杀（heartbeat 超时进程仍存活 = stuck）") + f" —— 计划退避 {wait:.0f} tick")
                self.event(name, "RETRYING", why)
            else:
                t["state"] = "FAILED"
                self.event(name, "FAILED", f"attempt {t['attempts']} —— 重试上限 ({self.spec[name]['retries']}) 耗尽，止损")
        else:                                            # permanent
            t["state"] = "FAILED"
            self.event(name, "FAILED", f"attempt {t['attempts']} permanent（{p['signature']}）—— 立即止损，不重试")

    def recover(self, at):                     # 孤儿收养（对照 adopt_or_reset_orphaned_tasks，scheduler_job_runner.py:L3246-3345）
        self.t = at
        for n in sorted(self.tasks):
            t = self.tasks[n]
            if t["state"] != "RUNNING": continue
            p = self._sim(n)
            alive = (t["dies_at"] is None or at < t["dies_at"]) and not t["killed"]
            fresh = t["last_hb"] is not None and (at - t["last_hb"]) <= HB_TIMEOUT
            if alive and fresh:                # 能收养：心跳新鲜 + 进程存活 → queued_by 换新调度器（对照 L3328）
                t["queued_by"] = self.sid
                self.event(n, "RUNNING", f"adopted（sched#{self.sid} 收养：心跳新鲜 + 进程存活，attempt {t['attempts']} 继续，不重启）")
            else:                              # 不能收养：state 回炉（对照 L3322 ti.state = None → nano 回 RETRYING 通道）
                self._finish(n, "killed")
                self.events[-1]["why"] += "  <- zombie 识别（调度器重启：心跳陈旧或进程已死）"

    def step(self):
        t = self.t
        for n in sorted(self.tasks):                        # [a] 完成处理（先于心跳扫描：同 tick 完成的不误杀）
            tk = self.tasks[n]
            if tk["state"] != "RUNNING": continue
            p = self._sim(n)
            stuck = tk["hb_stop"] is not None and t > tk["hb_stop"]        # 心跳停 = 无进展，不会「正常完成」
            dead = tk["dies_at"] is not None and t > tk["dies_at"]         # 进程已死，完成无从谈起
            if stuck or dead: continue
            if t >= tk["t_start"] + p["duration"]: self._finish(n, "complete")
        for n in sorted(self.tasks):                        # [b] heartbeat 连续形态扫描（对照 L3503-3529 周期 purge）
            tk = self.tasks[n]
            if tk["state"] != "RUNNING": continue
            alive = (tk["dies_at"] is None or t <= tk["dies_at"]) and not tk["killed"]
            beating = alive and not (tk["hb_stop"] is not None and t > tk["hb_stop"])
            if beating: tk["last_hb"] = t                   # 活着且未 stuck：每步一跳（对照 last_heartbeat_at 更新）
            if t - tk["last_hb"] > HB_TIMEOUT:
                if alive:
                    tk["killed"] = True; self._finish(n, "killed")         # stuck：杀之重试（「kill externally」路径）
                else:
                    self._finish(n, "killed")
                    self.events[-1]["why"] = (f"zombie 识别: 心跳停 {t - tk['last_hb']:.0f} tick 且进程已死"
                                              f" —— 归重试通道（L1 同款语义，连续形态）")
        for n in sorted(self.tasks):                        # [c] 规则 A：RETRYING 唤醒（L0/L1 同款）
            tk = self.tasks[n]
            if tk["state"] == "RETRYING" and tk["retry_after"] is not None and tk["retry_after"] <= t:
                tk["state"] = "RUNNABLE"; tk["retry_after"] = None
        for n in sorted(self.tasks):                        # [d] 规则 B：依赖解析 + trigger rule（L2 配置化）
            tk = self.tasks[n]
            if tk["state"] != "PENDING": continue
            s = self.spec[n]
            if tk["branch_of"] and self.tasks.get(tk["branch_of"], {}).get("state") == "SUCCESS":
                if self.spec[tk["branch_of"]].get("choose") != n:
                    tk["state"] = "SKIPPED"; self.event(n, "SKIPPED", f"branch 未选中（{tk['branch_of']} 选择另一分支）")
                    continue
            ds = [self.tasks[d]["state"] for d in s["deps"]]
            done = sum(1 for x in ds if x in TERMINAL)
            c = {"success": ds.count("SUCCESS"), "failed": ds.count("FAILED"),
                 "upstream_failed": ds.count("UPSTREAM_FAILED"), "skipped": ds.count("SKIPPED")}
            # 注：nano 的 success 计数不含 SKIPPED——Airflow success_states 含 SKIPPED 是 DagRun 统计口径（state.py:L222-224），
            # trigger 分派单独数 skipped（trigger_rule_dep.py 的 counter 族），两口径不混用（tutorial §5 展开）。
            dec = trigger_decision(tk["trigger_rule"], c, done, len(ds))
            if dec == "run": tk["state"] = "RUNNABLE"
            elif dec == "skip":
                tk["state"] = "SKIPPED"; self.event(n, "SKIPPED", f"trigger_rule={tk['trigger_rule']}: skipped 上游传染")
            elif dec == "upstream_failed":
                tk["state"] = "UPSTREAM_FAILED"
                self.event(n, "UPSTREAM_FAILED", f"trigger_rule={tk['trigger_rule']}: 上游失败——依赖是承诺，不在坏数据上跑")
        for n in sorted(self.tasks, key=lambda x: (self.tasks[x]["priority"], x)):   # [e] 入池：RUNNABLE → QUEUED
            tk = self.tasks[n]
            if tk["state"] != "RUNNABLE": continue
            if tk["key"] and tk["claim_retry"] is not None and tk["claim_retry"] > t: continue
            if tk["pool"] and not self.pool_open(tk["pool"]):
                if not tk["pool_wait_announced"]:
                    tk["pool_wait_announced"] = True
                    self.event(n, "RUNNABLE", f"pool '{tk['pool']}' 无空槽（occupied={self.pool_occupied(tk['pool'])}）——等待")
                continue
            if tk["key"]:
                if not self.keys.claim(tk["key"], n):
                    k = len(tk["claim_backoffs"])
                    wait = min(15.0, 1.0 + (1.1 ** k - 1.0))     # Dagster 退避公式（instance_concurrency_context.py:L178-189）
                    tk["claim_backoffs"].append(round(wait, 2)); tk["claim_retry"] = t + wait
                    self.event(n, "RUNNABLE", f"并发键 '{tk['key']}' 无空槽 —— claim 被拒（第 {k + 1} 次），退避 {wait:.2f} tick")
                    continue
                tk["claim_retry"] = None
            tk["state"] = "QUEUED"; tk["pool_wait_announced"] = False
            self.event(n, "QUEUED", f"入 executor 队列（priority={tk['priority']}" + (f"，pool={tk['pool']}" if tk["pool"] else "") + "）")
        for n in sorted(self.tasks, key=lambda x: (self.tasks[x]["priority"], x)):   # [f] 出队：QUEUED → RUNNING
            tk = self.tasks[n]
            if tk["state"] != "QUEUED": continue
            if self.open_slots() <= 0: break                     # parallelism 门槛（base_executor.py:L350）
            self.dequeue_log.append((round(t, 2), n)); self._launch(n)
        for p in sorted(self.pools):
            self.pool_ledger.append((round(t, 2), p, self.pool_occupied(p)))

    def run(self, stop_at=None):
        guard = 0
        while not all(tk["state"] in TERMINAL for tk in self.tasks.values()):
            if stop_at is not None and self.t >= stop_at: return
            self.step()
            if all(tk["state"] in TERMINAL for tk in self.tasks.values()): break   # 停表于最后事件，不走幻影 tick
            cands = [self.t + 1.0]
            for tk in self.tasks.values():
                if tk["state"] == "RETRYING" and tk["retry_after"] is not None: cands.append(tk["retry_after"])
                if tk["claim_retry"] is not None: cands.append(tk["claim_retry"])
                if tk["state"] == "RUNNING": cands.append(tk["t_start"] + self._sim_duration(tk))
            nxt = min(c for c in cands if c > self.t)
            self.t = round(nxt, 2)
            guard += 1
            if guard > 500: raise RuntimeError("不收敛——fixture 设计错误")

    def _sim_duration(self, tk):
        return self._sim_by_name(tk)["duration"]
    def _sim_by_name(self, tk):
        for n, t in self.tasks.items():
            if t is tk: return self._sim(n)
        raise AssertionError

    def cancel_group(self, prefix):            # GHA cancel-in-progress 的本质模拟（新 run 取消同 group 未终态旧 run）
        for n in sorted(self.tasks):
            tk = self.tasks[n]
            if not n.startswith(prefix) or tk["state"] in TERMINAL: continue
            was_running = tk["state"] == "RUNNING"
            if tk["key"]: self.keys.release(tk["key"], n)
            tk["state"] = "CANCELLED"
            self.event(n, "CANCELLED", f"concurrency group 被新 run 抢占（cancel-in-progress: true）——"
                                       + ("运行中被取消（attempt 已花钱）" if was_running else "排队中被取消（0 attempt）"))

def cost_report(tasks, events):
    coins = sum(t["attempts"] for t in tasks.values())
    eff = sum(1 for t in tasks.values() if t["state"] == "SUCCESS")
    zombie_tax = len([e for e in events if e["why"].startswith("zombie") or "zombie 识别" in e["why"]])
    recovery = sum(t["attempts"] - 1 for t in tasks.values() if t["state"] == "SUCCESS") - zombie_tax
    wasted = sum(t["attempts"] for t in tasks.values() if t["state"] in ("FAILED", "CANCELLED"))
    return coins, eff, recovery, wasted, zombie_tax

# ---------- Agentic 自愈：观察（事件日志）→ 诊断（playbook）→ 行动（白名单 patch）→ 复验 ----------
def healer_diagnose(spec, tasks, events):
    denied = [n for n, t in tasks.items() if t["attempts"] == 0 and t["state"] == "FAILED"
              and any(e["task"] == n and e["to"] == "FAILED(deny)" for e in events)]
    if denied:                                             # P0：capability 缺失 = 安全边界
        return {"pattern": "P0_capability_missing", "tasks": sorted(denied), "action": "escalate_to_human",
                "reason": "capability 授权是安全边界（default-deny）——agent 不得自动授权，人工审批后重跑"}
    for n, t in sorted(tasks.items()):
        if t["state"] != "FAILED" or t["attempts"] <= spec[n]["retries"]: continue
        sigs = {e["why"] for e in events if e["task"] == n and "transient" in e["why"]}
        cone = [m for m, x in tasks.items() if x["state"] == "UPSTREAM_FAILED" and _in_cone(spec, n, m)]
        if sigs and cone:                                  # P1：重试耗尽 + 签名一致 + 下游锥饿死 = 坏源
            return {"pattern": "P1_bad_source", "task": n, "cone": sorted(cone), "action": "reroute_to_fallback",
                    "patch": {"quarantine": n, "add": n + "_fallback",
                              "repoint": {m: [d if d != n else n + "_fallback" for d in spec[m]["deps"]]
                                          for m in sorted(tasks) if n in spec[m]["deps"]}}}
    return {"pattern": "unknown", "action": "escalate_to_human", "reason": "无匹配 playbook——不做未授权动作"}

def _in_cone(spec, root, m):
    seen, stack = set(), [m]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        if root in spec[x]["deps"]: return True
        stack.extend(spec[x]["deps"])
    return False

def apply_patch(spec, patch):
    s = copy.deepcopy(spec)
    q = patch["quarantine"]
    del s[q]                                               # 隔离 = 移出 DAG（失败历史留在事件日志）
    s[patch["add"]] = dict(deps=[], retries=0, signature="fallback 源")
    for m, deps in patch["repoint"].items():
        if m in s: s[m]["deps"] = deps
    return s

# ---------- run [7] 真进程层：任务程序 + sched 模式（并行 + heartbeat 文件 + 孤儿收养 + 结果通道）----------
REAL_TASK = r'''
import json, os, sys, time
t = os.environ["NANO_TASK"]; dur = float(os.environ["NANO_DURATION"]); w = os.environ["NANO_WORKDIR"]
hb_log = os.path.join(w, "hb_" + t + ".log")
def beat(n):
    with open(hb_log, "a") as f:
        f.write(f"t={time.time():.3f} n={n}\n"); f.flush(); os.fsync(f.fileno())
n = 0
t_end = time.time() + dur
while time.time() < t_end:                     # heartbeat 通道：周期落盘（对照 Airflow TI last_heartbeat_at）
    beat(n); n += 1; time.sleep(0.15)
beat(n)
tmp = os.path.join(w, f"result_{t}.json.tmp")  # 结果通道：原子发布（L1 atomic_write 同款；exit code 之外的第二通道）
with open(tmp, "w") as f:
    json.dump({"task": t, "ok": True}, f); f.flush(); os.fsync(f.fileno())
os.replace(tmp, os.path.join(w, f"result_{t}.json"))
sys.exit(0)
'''
REAL_DAG = {"p_short1": 0.3, "p_short2": 0.3, "p_long": 2.0}   # 时长分离 → 事件顺序确定（并行度证据 = 心跳区间重叠）
REAL_PARALLELISM, REAL_HB_TIMEOUT, ON_CALL_WINDOW = 3, 0.6, 0.8

def real_state_path(wd): return os.path.join(wd, "state.json")
def real_events_path(wd): return os.path.join(wd, "events.jsonl")

def real_save(wd, st):
    tmp = real_state_path(wd) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False, sort_keys=True, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, real_state_path(wd))

def real_load(wd):
    if not os.path.exists(real_state_path(wd)): return None
    with open(real_state_path(wd)) as f: return json.load(f)

def real_event(wd, task, to, why, attempt):
    seq = 0
    if os.path.exists(real_events_path(wd)):
        with open(real_events_path(wd)) as f: seq = sum(1 for _ in f)
    seq += 1
    with open(real_events_path(wd), "a") as f:
        f.write(json.dumps({"seq": seq, "task": task, "to": to, "why": why, "attempt": attempt}, ensure_ascii=True) + "\n")
        f.flush(); os.fsync(f.fileno())

def pid_alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True

def hb_fresh(wd, name, timeout, since=None):
    # since = 启动时刻宽限：心跳文件尚未落盘时以 launched_at 为基准（对照 Airflow last_heartbeat_at 入队即初始化，
    # 解释器启动延迟不算 stuck）；since=None 时要求真实心跳存在（收养判定用）。
    p = os.path.join(wd, f"hb_{name}.log")
    if not os.path.exists(p):
        return since is not None and (time.time() - since) <= timeout
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    if not lines:
        return since is not None and (time.time() - since) <= timeout
    last = float(lines[-1].split()[0][2:])
    return (time.time() - last) <= timeout

def real_reconcile(wd):
    st = real_load(wd)
    if st is None:
        st = {"tasks": {n: {"state": "PENDING", "attempts": 0, "pid": None} for n in REAL_DAG}}
        real_save(wd, st)
    recovered = []
    for n in sorted(st["tasks"]):                              # 重启恢复：结果通道优先 / 孤儿收养 / zombie 回炉（序确定，无时序竞态）
        t = st["tasks"][n]
        if t["state"] != "RUNNING": continue
        result = os.path.join(wd, f"result_{n}.json")
        if os.path.exists(result):                             # 结果在盘 = 已完成——与 pid 活否、心跳新旧无关（zombie 心跳
            t["state"] = "SUCCESS"; t["pid"] = None            # 可能尚新鲜：先查 pid 会把已完成任务误判成 adopted，时序敏感）
            real_event(wd, n, "SUCCESS", "result channel 验证（原子结果文件在盘）——收养的第二种形态", t["attempts"])
            recovered.append((n, "result channel 验证 SUCCESS（原子结果文件在盘）"))
        elif t["pid"] and pid_alive(t["pid"]) and hb_fresh(wd, n, REAL_HB_TIMEOUT):
            recovered.append((n, "adopted（pid 存活，heartbeat 新鲜）——liveness 与结果都走文件通道，不走父子关系"))
        else:
            t["state"] = "PENDING"                             # zombie：进程死、无结果 → 回重试通道（L1 同款）
            real_event(wd, n, "RETRYING", "zombie 识别（pid 已死且无结果文件）", t["attempts"])
            recovered.append((n, "zombie 回炉（pid 已死且无结果文件）"))
        real_save(wd, st)
    for n, msg in recovered:
        print(f"  [recover] {n}: {msg}")
        if msg.startswith("adopted"): real_event(wd, n, "RUNNING", "adopted（重启调度器收养，attempt 继续）", st["tasks"][n]["attempts"])
    guard = 0
    while True:
        if all(t["state"] in ("SUCCESS",) for t in st["tasks"].values()): break
        guard += 1
        if guard > 1200: raise RuntimeError("real_reconcile 不收敛——fixture 设计错误")
        changed = 0
        for n in sorted(st["tasks"]):                          # 完成/失踪检测：result 通道优先（zombie 反例，tutorial §7 教材化）
            t = st["tasks"][n]
            if t["state"] != "RUNNING": continue
            result = os.path.join(wd, f"result_{n}.json")
            if os.path.exists(result):                          # 原子结果在盘 = 完成，与 pid 活否无关——未收割 zombie 对
                t["state"] = "SUCCESS"; t["pid"] = None         # os.kill(pid,0) 恒返回 True，pid 探测不可靠，结果文件才是权威
                real_event(wd, n, "SUCCESS", f"attempt {t['attempts']} exit=0（result channel 验收）", t["attempts"])
                changed += 1
                continue
            if t["pid"] and pid_alive(t["pid"]):
                if not hb_fresh(wd, n, REAL_HB_TIMEOUT, since=t.get("launched_at")):   # stuck 路径：首跳前以 launched_at 宽限，之后看真实心跳（机制在位，本 fixture 不触发）
                    try: os.kill(t["pid"], signal.SIGKILL)
                    except ProcessLookupError: pass
                    t["state"] = "PENDING"; t["pid"] = None
                    real_event(wd, n, "RETRYING", "heartbeat 超时（进程存活 = stuck）——杀之回炉", t["attempts"])
                    changed += 1
                continue
            t["state"] = "PENDING"                              # pid 已死且无结果文件 → zombie 回炉（L1 同款语义）
            real_event(wd, n, "RETRYING", "进程已死且无结果文件——zombie 回炉", t["attempts"])
            changed += 1
        for n in sorted(st["tasks"]):                          # 启动（并行度门槛 = open_slots）
            t = st["tasks"][n]
            if t["state"] != "PENDING": continue
            running = sum(1 for x in st["tasks"].values() if x["state"] == "RUNNING")
            if running >= REAL_PARALLELISM: break
            t["attempts"] += 1; t["state"] = "RUNNING"; t["launched_at"] = time.time()   # 启动宽限基准（对照 Airflow last_heartbeat_at 入队即初始化，job.py:L141）
            os.makedirs(os.path.join(wd, "logs"), exist_ok=True)
            log_f = open(os.path.join(wd, "logs", f"{n}.attempt{t['attempts']}.log"), "w")
            env = dict(os.environ, NANO_TASK=n, NANO_DURATION=str(REAL_DAG[n]), NANO_WORKDIR=wd)
            p = subprocess.Popen([sys.executable, "-B", "-c", REAL_TASK], stdout=log_f, stderr=subprocess.STDOUT, env=env)
            t["pid"] = p.pid
            real_save(wd, st)                                  # 先落 pid 再放手——收养依赖这一步（L1 同款窗口分析）
            real_event(wd, n, "RUNNING", f"attempt {t['attempts']} 启动（pid 已落 state.json）", t["attempts"])
            changed += 1
        real_save(wd, st)
        if changed == 0: time.sleep(0.05)
    print(f"  [done] terminal: SUCCESS={sorted(n for n, t in st['tasks'].items() if t['state'] == 'SUCCESS')}")

# ---------- demo 驱动 ----------
def main():
    print("== nano-data-orchestration L2: 并行 executor + 资源池 + trigger rules + heartbeat/收养 + 并发键 + Agentic 自愈 ==")
    print("  （runs [1]–[6] = 逻辑时钟本质模拟，确定性可锚；run [7] = 真实 subprocess 并行 + 真实 kill + 真实孤儿收养）")
    t0 = time.time()

    # ---------------- [1] 跨级锚：parallelism=1 退化运行 L1 fixture ----------------
    print("\n[1] 跨级锚：L1 fixture 在 L2 状态机上退化运行（parallelism=1）——终态 digest 必须复现 L1 录值")
    s1 = Scheduler(PIPELINE, GRANTS, parallelism=1)
    s1.run()
    d1 = state_digest(s1.tasks)
    coins, eff, rec, wasted, ztax = cost_report(s1.tasks, s1.events)
    vec = {s: sorted(n for n in s1.tasks if s1.tasks[n]["state"] == s) for s in sorted(TERMINAL) if any(
        s1.tasks[n]["state"] == s for n in s1.tasks)}
    print(f"  终态向量: SUCCESS={vec.get('SUCCESS')}\n            FAILED={vec.get('FAILED')}  UPSTREAM_FAILED={vec.get('UPSTREAM_FAILED')}")
    print(f"  成本账本: 总 {coins} = 有效 {eff} + 重试救回 {rec} + 浪费 {wasted} + 崩溃税 {ztax}")
    check("跨级锚: 终态向量 digest == L1 录值 ac4a0b3ac09bf47b（L1 state_digest 公式逐字同款）", d1 == "ac4a0b3ac09bf47b")
    check("成本恒等式 9 = 5 + 1 + 3 + 0（复现 L0 check 13 / L1 check 08）", (coins, eff, rec, wasted, ztax) == (9, 5, 1, 3, 0))
    check("default-deny 在并行状态机下不变: publish_report attempts==0（QUEUED 之前就拒）",
          s1.tasks["publish_report"]["attempts"] == 0)
    check("事件流 seq 单调无 gap；终态 5/2/2 与 L0/L1 逐字一致",
          [e["seq"] for e in s1.events] == list(range(1, len(s1.events) + 1))
          and len(vec.get("SUCCESS", [])) == 5 and len(vec.get("FAILED", [])) == 2 and len(vec.get("UPSTREAM_FAILED", [])) == 2)
    ticks1 = s1.t
    print(f"  （串行调和用时 {ticks1:.0f} tick——[2] 的并行必须更快且收敛点不变）")

    # ---------------- [2] 并行 executor + pool + priority ----------------
    print("\n[2] 并行 executor + pool + priority：同一 fixture，parallelism=3，pool 'src' 槽位=1，unit_tests 优先级最高")
    spec2 = copy.deepcopy(PIPELINE)
    for n in ("ingest_crm", "ingest_web", "unit_tests"): spec2[n]["pool"] = "src"
    spec2["unit_tests"]["priority"] = 0; spec2["ingest_crm"]["priority"] = 5; spec2["ingest_web"]["priority"] = 5
    s2 = Scheduler(spec2, GRANTS, parallelism=3, pools={"src": 1})
    s2.run()
    d2 = state_digest(s2.tasks)
    max_occ = max(occ for (_, p, occ) in s2.pool_ledger if p == "src")
    first_dequeue = s2.dequeue_log[0][1]
    print(f"  pool 'src' 占用峰值 = {max_occ}（slots=1）；首个出队任务 = {first_dequeue}（priority=0 先于 priority=5）")
    check("并行不改变收敛点: 终态 digest == [1]（调度策略是路径，不是语义）", d2 == d1)
    check("pool 不变量: 'src' 占用峰值 ≤ slots（occupied = QUEUED+RUNNING，对照 Pool.occupied_slots）", max_occ <= 1)
    check("priority 出队顺序: 权重小者先出（nano 约定；Airflow sorted by priority_weight, reverse=False 的排序事实）",
          first_dequeue == "unit_tests")
    check("attempts 向量逐字不变 + 本 fixture 临界路径（ingest_web 退避链）两模式同长，故 makespan 不快于串行",
          attempts_digest(s2.tasks) == attempts_digest(s1.tasks) and s2.t <= ticks1)
    probe = {f"x{i}": dict(deps=[], retries=0, duration=2.0) for i in (1, 2, 3)}
    p_ser, p_par = Scheduler(copy.deepcopy(probe), parallelism=1), Scheduler(copy.deepcopy(probe), parallelism=3)
    p_ser.run(); p_par.run()
    check("并行的量化证据（无共享资源探针）: 3 独立任务 × 2 tick，串行 6 tick -> 并行 2 tick（3× 加速）",
          p_ser.t == 6.0 and p_par.t == 2.0 and state_digest(p_par.tasks) == state_digest(p_ser.tasks))
    print(f"  elapsed: 逻辑时钟 [1] {ticks1:.0f} tick -> [2] {s2.t:.0f} tick（本 fixture 省不下 = 临界路径是退避链，不是算力）")

    # ---------------- [3] trigger rules 配置化 ----------------
    print("\n[3] trigger rules：上游终态向量 -> 本任务命运（语义对照 Airflow TriggerRuleDep 分派表）")
    spec3 = {
        "branch":             dict(deps=[], retries=0, choose="left"),
        "left":               dict(deps=["branch"], retries=0, branch_of="branch"),
        "right":              dict(deps=["branch"], retries=0, branch_of="branch"),
        "bad":                dict(deps=[], retries=0, fails={1: "permanent"}, signature="校验失败"),
        "solo_all_success":   dict(deps=["left"], retries=0, trigger_rule="all_success"),
        "join_all_success":   dict(deps=["left", "right"], retries=0, trigger_rule="all_success"),
        "join_none_failed":   dict(deps=["left", "right"], retries=0, trigger_rule="none_failed"),
        "join_one_success":   dict(deps=["left", "right"], retries=0, trigger_rule="one_success"),
        "join_all_done":      dict(deps=["left", "right", "bad"], retries=0, trigger_rule="all_done"),
        "join_nfmos":         dict(deps=["left", "bad"], retries=0, trigger_rule="none_failed_min_one_success"),
        "join_always":        dict(deps=["bad"], retries=0, trigger_rule="always"),
    }
    s3 = Scheduler(spec3, parallelism=8)
    s3.run()
    got3 = {n: t["state"] for n, t in sorted(s3.tasks.items())}
    exp3 = {"branch": "SUCCESS", "left": "SUCCESS", "right": "SKIPPED", "bad": "FAILED",
            "solo_all_success": "SUCCESS", "join_all_success": "SKIPPED", "join_none_failed": "SUCCESS",
            "join_one_success": "SUCCESS", "join_all_done": "SUCCESS", "join_nfmos": "UPSTREAM_FAILED",
            "join_always": "SUCCESS"}
    for n in sorted(exp3): print(f"  {n:20s} -> {got3[n]}")
    check("ALL_SUCCESS 被 skipped 上游染成 SKIPPED（trigger_rule_dep.py:L433-434；skipped 不算成功调度口径）",
          got3["join_all_success"] == "SKIPPED" and got3["solo_all_success"] == "SUCCESS")
    check("branching: 未选中分支 = SKIPPED（新终态，L0/L1 没有）", got3["right"] == "SKIPPED")
    check("NONE_FAILED 容忍 skipped、ONE_SUCCESS 见好就跑（两者都 SUCCESS）",
          got3["join_none_failed"] == "SUCCESS" and got3["join_one_success"] == "SUCCESS")
    check("ALL_DONE 等全部终态后无条件跑（含 FAILED 上游）；NONE_FAILED_MIN_ONE_SUCCESS 见 failed 即 UPSTREAM_FAILED",
          got3["join_all_done"] == "SUCCESS" and got3["join_nfmos"] == "UPSTREAM_FAILED")
    check("ALWAYS 不等上游（bad 尚未终态时 join_always 已完成）",
          got3["join_always"] == "SUCCESS"
          and next(e for e in s3.events if e["task"] == "join_always" and e["to"] == "RUNNING")["t"] == 0.0)

    # ---------------- [4] heartbeat 连续形态 + 孤儿收养（逻辑层）----------------
    print("\n[4] heartbeat 连续形态：stuck（活着不跳）杀之重试 / 死掉不跳 = zombie 回炉；调度器重启 -> 孤儿收养")
    spec4 = {
        "hb_ok":    dict(deps=[], retries=1, duration=3.0),
        "hb_stuck": dict(deps=[], retries=1, duration=3.0, hb_stop_attempt=1),   # attempt1 心跳停在 t+1，进程仍活
        "hb_dead":  dict(deps=[], retries=1, duration=3.0, dies_at_attempt=1),   # attempt1 进程 t+1 暴毙（无结果）
    }
    s4 = Scheduler(spec4, parallelism=3)
    s4.run()
    kill_ev = next(e for e in s4.events if e["task"] == "hb_stuck" and "被杀" in e["why"])
    zom_ev = next(e for e in s4.events if e["task"] == "hb_dead" and e["why"].startswith("zombie 识别"))
    check("stuck 检测: 心跳停于 t=1，超时=2 -> 恰在 t=4 被杀（检测时刻 = last_hb + timeout 后的首个事件点，确定）",
          kill_ev["t"] == 4.0 and "stuck" in kill_ev["why"])
    check("zombie 连续形态: 进程暴毙 + 心跳陈旧 -> 同一时刻识别回炉（L1 点状语义的连续化）",
          zom_ev["t"] == 4.0 and s4.tasks["hb_dead"]["attempts"] == 2)
    check("两者重试后都收敛 SUCCESS；hb_ok 一次通过（心跳正常 = 免打扰）",
          s4.tasks["hb_stuck"]["state"] == "SUCCESS" and s4.tasks["hb_dead"]["state"] == "SUCCESS"
          and s4.tasks["hb_ok"]["attempts"] == 1)
    print("  --- 调度器崩溃 @ t=2（逻辑模拟）：丢弃 sched#1 内存，sched#2 从状态重建并收养 ---")
    spec4b = {"long_a": dict(deps=[], retries=1, duration=6.0), "short_b": dict(deps=[], retries=0, duration=1.0)}
    sa = Scheduler(spec4b, parallelism=2)
    sa.run(stop_at=2.0)                                      # sched#1 跑到 t=2：short_b 已 SUCCESS，long_a RUNNING
    snap = copy.deepcopy(sa.tasks)
    sb = Scheduler(spec4b, parallelism=2, scheduler_id=2, tasks=snap)
    sb.recover(2.0)                                          # sched#2 入场第一件事：孤儿收养判定（对照 adopt_or_reset）
    sb.run()
    check("孤儿收养: long_a 被 sched#2 收养——queued_by 换人、attempt 不增（工作不重做）",
          sb.tasks["long_a"]["queued_by"] == 2 and sb.tasks["long_a"]["attempts"] == 1
          and sb.tasks["long_a"]["state"] == "SUCCESS")
    check("收养凭据 = 心跳新鲜 + 进程存活（对照 adoptable_states + try_adopt）；short_b 已终态不受扰动",
          any(e["task"] == "long_a" and "adopted" in e["why"] for e in sb.events)
          and sb.tasks["short_b"]["state"] == "SUCCESS" and sb.tasks["short_b"]["attempts"] == 1)

    # ---------------- [5] 并发键（Dagster 机制）+ cancel-in-progress（GHA 机制）----------------
    print("\n[5] 并发键 claim/release + 指数步进退避（Dagster）；concurrency group cancel-in-progress（GitHub Actions）")
    spec5 = {f"c{i}": dict(deps=[], retries=0, duration=2.0, key="gpu") for i in (1, 2, 3)}
    keys = KeyLedger({"gpu": 2})
    s5 = Scheduler(spec5, parallelism=3, keys=keys)
    s5.run()
    c3 = s5.tasks["c3"]
    print(f"  键 'gpu' slots=2: c3 被拒 {len(c3['claim_backoffs'])} 次，退避序列 {c3['claim_backoffs']} tick，持有峰值 {keys.max_held}")
    check("槽位不变量: 持有峰值 ≤ slot_count（claim/release 账本平衡 claims==releases==3）",
          keys.max_held == 2 and keys.claims == 3 and keys.releases == 3)
    check("退避是指数步进（Dagster 公式 1+(1.1^n-1)，上限 15）: c3 录值 [1.0, 1.1]", c3["claim_backoffs"] == [1.0, 1.1])
    check("阻塞窗口精确: c3 在 t=2.1 拿到槽（c1/c2 t=2 释放，退避到点即 claim）且最终 SUCCESS",
          next(e for e in s5.events if e["task"] == "c3" and e["to"] == "RUNNING")["t"] == 2.1
          and all(t["state"] == "SUCCESS" for t in s5.tasks.values()))
    spec5b = {f"r1.d{i}": dict(deps=[], retries=0, duration=3.0 if i == 1 else 1.0) for i in (1, 2, 3)}
    s5b = Scheduler(spec5b, parallelism=1)
    s5b.run(stop_at=1.0)                                     # r1 开跑：d1 RUNNING，d2/d3 QUEUED
    s5b.cancel_group("r1.")                                  # t=1：r2 提交，cancel-in-progress: true
    s5b.add_tasks({f"r2.e{i}": dict(deps=[], retries=0, duration=1.0) for i in (1, 2, 3)})   # r2 此刻才进入 DAG
    s5b.run()
    cancelled = sorted(n for n in s5b.tasks if s5b.tasks[n]["state"] == "CANCELLED")
    print(f"  cancel-in-progress: {cancelled}（运行中 1 + 排队中 2）；r2 = {sorted(n for n in s5b.tasks if n.startswith('r2.') and s5b.tasks[n]['state'] == 'SUCCESS')}")
    check("GHA 语义: 同 group 旧 run 全部取消（RUNNING 的 attempt 已花钱记浪费，QUEUED 的 0 attempt）",
          cancelled == ["r1.d1", "r1.d2", "r1.d3"] and s5b.tasks["r1.d1"]["attempts"] == 1
          and s5b.tasks["r1.d2"]["attempts"] == 0)
    check("新 run 正常跑完: r2 三个 SUCCESS（cancel 不伤新 run）",
          all(s5b.tasks[f"r2.e{i}"]["state"] == "SUCCESS" for i in (1, 2, 3)))

    # ---------------- [6] Agentic 自愈：观察 -> 诊断 -> 白名单行动 -> 复验 ----------------
    print("\n[6] Agentic 自愈：坏源误分类重试耗尽 -> playbook P1 改道 fallback；capability 缺失 -> 升级人工（不自动授权）")
    spec6 = {"ingest_bad": dict(deps=[], retries=2, fails={1: "transient", 2: "transient", 3: "transient"},
                                signature="source 拒绝所有读取（永久损坏）"),
             "gate_bad":   dict(deps=["ingest_bad"], retries=0),
             "report_bad": dict(deps=["gate_bad"], retries=0)}
    s6a = Scheduler(spec6, parallelism=2)
    s6a.run()
    diag = healer_diagnose(spec6, s6a.tasks, s6a.events)
    print(f"  诊断: {json.dumps(diag, ensure_ascii=False)}")
    check("诊断命中 P1_bad_source: 重试耗尽 + 签名一致 + 下游锥饿死（观察全部来自结构化事件日志）",
          diag["pattern"] == "P1_bad_source" and diag["cone"] == ["gate_bad", "report_bad"])
    spec6_fixed = apply_patch(spec6, diag["patch"])
    s6b = Scheduler(spec6_fixed, parallelism=2)
    s6b.run()
    check("白名单 patch 生效: 隔离坏源 + fallback 改道后全 SUCCESS（复验 = 重跑收敛，不是口头保证）",
          all(t["state"] == "SUCCESS" for t in s6b.tasks.values()) and "ingest_bad" not in s6b.tasks)
    spec6c = {"cap_task": dict(deps=[], retries=0, needs={"prod_deploy"})}
    s6c = Scheduler(spec6c, grants={}, parallelism=1)
    s6c.run()
    diagc = healer_diagnose(spec6c, s6c.tasks, s6c.events)
    print(f"  诊断（capability 缺失）: {json.dumps(diagc, ensure_ascii=False)}")
    check("安全边界 first-class: capability 缺失 -> escalate_to_human，agent 无授权动作（default-deny 不可绕过，attempts 仍 0）",
          diagc["action"] == "escalate_to_human" and s6c.tasks["cap_task"]["attempts"] == 0)

    # ---------------- [7] 真进程锚：并行 + kill 调度器（不杀子进程）+ 孤儿收养 ----------------
    print("\n[7] 真进程锚：parallelism=3 真实并行；kill -9 只杀调度器 -> 子进程成为孤儿 -> 重启收养（L1 做不到的那笔债）")
    WA = tempfile.mkdtemp(prefix="nano_orch_l2_A_")
    WB = tempfile.mkdtemp(prefix="nano_orch_l2_B_")
    try:
        tA = time.time()
        procA = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "sched", "--workdir", WA],
                               capture_output=True, text=True)
        rcA = procA.returncode
        # fail-loud：先验 rcA 再碰任何产物文件。否则 sched 崩溃时，下面开 hb_*.log 会抛 FileNotFoundError
        # 的表面症状，掩盖「sched 为什么失败」的真因（诊断必须带 sched 自己的 stdout/stderr）。
        if rcA != 0:
            raise RuntimeError(f"sched 干净 run 失败 rc={rcA}\n--- sched stdout ---\n{procA.stdout}"
                               f"\n--- sched stderr ---\n{procA.stderr}")
        stA = real_load(WA)
        d7 = state_digest(stA["tasks"])
        iv = {n: [float(l.split()[0][2:]) for l in open(os.path.join(WA, f"hb_{n}.log")) if l.strip()] for n in REAL_DAG}
        overlap = min(max(v) for v in iv.values()) - max(min(v) for v in iv.values())
        print(f"  干净 run: 3 任务全 SUCCESS，digest {d7}；三心跳区间存在共同存活窗（overlap > 0.1s = 真实并行的机器证据）")
        print(f"  elapsed: 干净 run wall-clock {time.time() - tA:.2f}s；三心跳区间重叠 {overlap:.2f}s（parallelism=3，p_long 2.0s 主导）")
        check("真进程并行: 三任务心跳区间存在共同存活窗（overlap > 0.1s；串行模型下三区间首尾相接、重叠≈0）",
              rcA == 0 and overlap > 0.1 and all(t["state"] == "SUCCESS" for t in stA["tasks"].values()))
        tB = time.time()
        proc = subprocess.Popen([sys.executable, "-B", os.path.abspath(__file__), "sched", "--workdir", WB],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        while True:                                          # kill 点 = 确定逻辑点：三任务全部 RUNNING（心跳都已开始）
            st_mid = real_load(WB)
            if st_mid is not None and all(t["state"] == "RUNNING" for t in st_mid["tasks"].values()): break
            if proc.poll() is not None: raise RuntimeError("调度器过早退出: " + proc.stdout.read())
            time.sleep(0.02)
        victim_pid = st_mid["tasks"]["p_long"]["pid"]
        os.kill(proc.pid, signal.SIGKILL)                    # 只杀调度器（对照 L1 killpg 全组俱灭 = 两种故障模型的分岔点）
        rc_kill = proc.wait(); proc.stdout.read()
        while pid_alive(victim_pid) is False or not hb_fresh(WB, "p_long", REAL_HB_TIMEOUT):
            time.sleep(0.02)                                 # 等孤儿稳定：p_long 活着且心跳在跳
        st_stale = real_load(WB)
        print(f"  kill 点: p_long RUNNING（pid 已录盘）；调度器被 kill -9，子进程 p_long/p_short1/p_short2 成为孤儿")
        check("调度器死而子进程活: rc=-SIGKILL，state.json 完整在盘，p_long 留 stale RUNNING 态（L1 在此只能 raise）",
              rc_kill == -signal.SIGKILL and st_stale["tasks"]["p_long"]["state"] == "RUNNING"
              and pid_alive(victim_pid))
        time.sleep(ON_CALL_WINDOW)                           # on-call 响应窗（固定常数，承 L1 口径）
        tR = time.time()
        out2 = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "sched", "--workdir", WB],
                              capture_output=True, text=True)
        print(out2.stdout, end="")
        print(f"  elapsed: on-call 窗 {ON_CALL_WINDOW:.2f}s + 重启收养与收尾 {time.time() - tR:.2f}s")
        stB = real_load(WB)
        evB = [json.loads(l) for l in open(real_events_path(WB)) if l.strip()]
        adopted = [e for e in evB if e["task"] == "p_long" and "adopted" in e["why"]]
        coinsB = sum(t["attempts"] for t in stB["tasks"].values())
        check("孤儿收养成功: p_long 被重启的调度器收养（pid 存活 + heartbeat 新鲜），attempt 不增——工作不重做",
              out2.returncode == 0 and len(adopted) == 1 and stB["tasks"]["p_long"]["attempts"] == 1)
        check("收敛点不变: 终态 digest == 干净 run（崩溃模型不同，收敛点相同——L1 同款不变量）",
              state_digest(stB["tasks"]) == d7)
        check("崩溃税 = 0（对照 L1 killpg 模型税=1）: 孤儿活下来了，收养省下了重做的 attempt（总 attempts == 3）",
              coinsB == 3)
        check("结果通道二形态: 短任务在重启前已完成 = result 文件验收（pid 已死 + 原子结果在盘 -> SUCCESS）",
              any(e["task"] == "p_short1" and "result channel" in e["why"] for e in evB)
              and stB["tasks"]["p_short1"]["attempts"] == 1 and stB["tasks"]["p_short2"]["attempts"] == 1)
        print(f"  elapsed: run [7] 孤儿线总 wall-clock {time.time() - tB:.2f}s")
    finally:
        for w in (WA, WB): shutil.rmtree(w, ignore_errors=True)
    print(f"\n  elapsed: 总 wall-clock {time.time() - t0:.2f}s")
    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sched":
        ap = argparse.ArgumentParser()
        ap.add_argument("sched"); ap.add_argument("--workdir", required=True)
        args = ap.parse_args()
        print(f"  [sched] start: tasks={len(REAL_DAG)} parallelism={REAL_PARALLELISM}")
        real_reconcile(args.workdir)
    else:
        main()
