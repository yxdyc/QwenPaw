#!/usr/bin/env python3
"""nano-data-orchestration L0 — DAG 调度器：任务状态机 + 调和循环 + 失败语义，纯 Python 本质模拟。

它在模拟真实系统的哪一面（L0 验收标准，ROADMAP §二）：
  [1] DAG = 一等公民的依赖结构：环与未知依赖在执行前被拒（fail fast，校验成本 0，运行时拒绝成本 = 整次运行）；
  [2] 任务状态机 + 调和循环（reconciliation loop）：调度器不是「跑一个脚本」，而是每 tick 扫描状态、施加转移
      规则——状态是完整记录，崩溃恢复 / 可观测性 / 重试都从状态长出来（Airflow 的 TaskInstance 状态机即此本质）；
  [3] 失败语义：transient/permanent 错误分类 → 指数退避有界重试 vs 立即失败；上游失败急切传播
      （UPSTREAM_FAILED），爆炸半径 = 下游锥；
  [4] 治理 first-class：capability default-deny（拒绝发生在计算之前，0 attempt 0 成本）+ attempt 成本账本（重试不是免费的）。
刻意不模拟：真实并行、持久化状态/崩溃续跑（L1）、wall-clock/cron 触发、池/SLA、executor 内部（L2 对照
Airflow/Dagster/Prefect 源码）——见 README 阶梯表。
零依赖（纯标准库），CPU 秒级；输出确定（tick = 逻辑时钟，同 tick 就绪集按名字序执行），复跑逐字节一致。
"""
import hashlib, json

CHECKS = []
def check(name, cond):
    CHECKS.append(bool(cond))
    print(f"  [check {len(CHECKS):02d}] {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: raise SystemExit("self-check failed: " + name)

class TransientError(Exception): pass   # 可重试：网络抖动 / 对端重启 ——「重试可能成功」
class PermanentError(Exception): pass   # 不可重试：权限被拒 / 校验失败 ——「重试 N 次同果」，分类是任务作者责任

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

def downstream_cone(tasks, root):  # 爆炸半径：root 失败后，哪些任务随之失去存在意义
    cone, stack = set(), [root]
    while stack:
        n = stack.pop()
        for m, spec in tasks.items():
            if n in spec["deps"] and m not in cone:
                cone.add(m); stack.append(m)
    return cone - {root}

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
        if log:
            for t, n, st, why in events: print(f"  [t={t}] {n:14s} -> {st:15s} {why}")
        events.clear()
        if did == 0 and not any(state[n] == "RETRYING" for n in tasks):
            break                                              # 死锁守卫：无变化且无等待（验证已排除，不应到达）
        tick += 1
    record = {"state": state, "attempts": attempts, "ticks": tick, "coins": sum(attempts.values()),
              "retry_waits": retry_waits, "exec_order": exec_order}
    body = json.dumps({k: record[k] for k in ("state", "attempts", "ticks", "coins")},
                      sort_keys=True, ensure_ascii=False).encode()
    record["digest"] = hashlib.sha256(body).hexdigest()[:16]
    return record

# ---- [4b] 成本账本：每次 attempt 烧钱。toy 单价 1 coin/attempt，教学设定非真实云价 ----
def cost_report(record):
    wasted = sum(a for n, a in record["attempts"].items() if record["state"][n] == "FAILED")
    recovery = sum(a - 1 for n, a in record["attempts"].items() if record["state"][n] == "SUCCESS")
    return record["coins"], wasted, recovery

# ---- demo fixture：一条 CI/CD 风格的数据管线（任务名呼应 nano-data-platform L0 的动作序列，见 tutorial §1）----
# 失败日程是显式实验设计（同 data-platform 刻意埋缺陷）：flaky 只败第 1 次；broken 永远败（且被作者误分类为 transient）。
def flaky_source(attempt):
    if attempt == 1: raise TransientError("connection reset —— transient")
def broken_source(attempt):
    raise TransientError("source 拒绝所有读取——看似 transient 实则 permanent（误分类教训）")
def ok(_attempt): pass

PIPELINE = {
    "ingest_crm":     dict(deps=[],                         retries=2, fn=flaky_source),
    "ingest_web":     dict(deps=[],                         retries=2, fn=broken_source),
    "gate_crm":       dict(deps=["ingest_crm"],             retries=0, fn=ok),
    "gate_web":       dict(deps=["ingest_web"],             retries=0, fn=ok),
    "normalize_web":  dict(deps=["gate_web"],               retries=0, fn=ok),
    "unit_tests":     dict(deps=[],                         retries=0, fn=ok),
    "build_curated":  dict(deps=["gate_crm", "unit_tests"], retries=1, fn=ok),  # CI 门：数据与测试都就绪才构建
    "deploy":         dict(deps=["build_curated"],          retries=1, needs={"prod_deploy"}, fn=ok),
    "publish_report": dict(deps=["build_curated"],          retries=2, needs={"metrics_write"}, fn=ok),
}
GRANTS = {"deploy": {"prod_deploy"}}   # 最小权限：只有 deploy 持 prod 能力；metrics_write 无人授权 -> 拒绝演练

def main():
    print("== nano-data-orchestration L0: DAG 调度器——状态机 + 调和循环 + 失败语义（纯 Python 本质模拟）==")
    print("\n[1] fail fast：环与未知依赖在执行前被拒（校验成本 0；运行时拒绝成本 = 整次运行）")
    cyclic = {"a": dict(deps=["c"], retries=0, fn=ok), "b": dict(deps=["a"], retries=0, fn=ok),
              "c": dict(deps=["b"], retries=0, fn=ok)}
    try:
        run(cyclic, {}); check("环必须被拒", False)
    except ValueError as e:
        print(f"  rejected as expected: {e}"); check("环必须在执行前被拒", True)
    try:
        run({"a": dict(deps=["ghost"], retries=0, fn=ok)}, {}); check("未知依赖必须被拒", False)
    except ValueError as e:
        print(f"  rejected as expected: {e}"); check("未知依赖必须在执行前被拒", True)
    print("\n[2] 调和循环跑一条带两种故障的数据管线（tick = 逻辑时钟，事件按 tick 回放）")
    r1 = run(PIPELINE, GRANTS)
    st, vec = r1["state"], {s: sorted(n for n in r1["state"] if r1["state"][n] == s) for s in TERMINAL}
    print(f"\n[3] 终态向量: SUCCESS={vec['SUCCESS']}\n    FAILED={vec['FAILED']}  UPSTREAM_FAILED={vec['UPSTREAM_FAILED']}")
    check("5 SUCCESS（含重试救回的 ingest_crm）", vec["SUCCESS"] == ["build_curated", "deploy", "gate_crm", "ingest_crm", "unit_tests"])
    check("2 FAILED（ingest_web 止损 / publish_report 被拒）", vec["FAILED"] == ["ingest_web", "publish_report"])
    check("2 UPSTREAM_FAILED（坏源的爆炸半径）", vec["UPSTREAM_FAILED"] == ["gate_web", "normalize_web"])
    print("\n[4] 错误分类 -> 重试策略：transient 指数退避有界重试，permanent 立即失败")
    print(f"  ingest_crm:     attempts={r1['attempts']['ingest_crm']} backoff={r1['retry_waits']['ingest_crm']} -> 重试救回")
    print(f"  ingest_web:     attempts={r1['attempts']['ingest_web']} backoff={r1['retry_waits']['ingest_web']} -> 上限耗尽止损（误分类的代价）")
    print(f"  publish_report: attempts={r1['attempts']['publish_report']} -> 拒绝发生在计算之前：permanent 0 重试 0 成本")
    check("transient 被救回: ingest_crm 第 2 次成功", r1["attempts"]["ingest_crm"] == 2 and st["ingest_crm"] == "SUCCESS")
    check("指数退避 = 2^(k-1): [1] 与 [1, 2]", r1["retry_waits"]["ingest_crm"] == [1] and r1["retry_waits"]["ingest_web"] == [1, 2])
    check("permanent 永不重试: publish_report attempts == 0", r1["attempts"]["publish_report"] == 0)
    check("重试上限 = 止损线: ingest_web attempts == retries+1 == 3", r1["attempts"]["ingest_web"] == 3)
    print("\n[5] 失败传播：爆炸半径 = 下游锥；重试救回的任务保住它的整个下游")
    saved, blocked = downstream_cone(PIPELINE, "ingest_crm"), downstream_cone(PIPELINE, "ingest_web")
    print(f"  若无重试，ingest_crm 失败将阻塞 {sorted(saved)}（{len(saved)} 个任务）")
    print(f"  ingest_web 失败实际阻塞 {sorted(blocked)}（{len(blocked)} 个任务）——全部 0 attempt，不在坏数据上浪费计算")
    check("重试保住整个下游锥（4 任务获得执行机会）", saved == {"gate_crm", "build_curated", "deploy", "publish_report"})
    check("坏源恰好阻塞其下游锥且 0 attempt，不波及其他分支",
          blocked == {"gate_web", "normalize_web"} and r1["attempts"]["gate_web"] == r1["attempts"]["normalize_web"] == 0)
    check("拓扑不变式：每个跑过的任务都在全部上游之后",
          all(r1["exec_order"].index(d) < r1["exec_order"].index(n)
              for n in PIPELINE for d in PIPELINE[n]["deps"] if n in r1["exec_order"]))
    print("\n[6] 成本账本（toy 单价 1 coin/attempt，教学设定非真实云价）：重试不是免费的")
    coins, wasted, recovery = cost_report(r1)
    print(f"  总 {coins} = 有效 {coins - wasted - recovery} + 重试救回 {recovery} + 浪费 {wasted}（ingest_web 的 3 次 attempt 全部无效）")
    check("成本恒等式: 9 = 5 + 1 + 3", (coins, wasted, recovery) == (9, 3, 1))
    print("\n[7] CI/CD 门 = 依赖边：deploy 等 build_curated 与 unit_tests 双就绪；调和是确定性的")
    check("deploy 在数据与测试双就绪后才跑",
          r1["exec_order"].index("deploy") > r1["exec_order"].index("build_curated") > r1["exec_order"].index("unit_tests"))
    r2 = run(PIPELINE, GRANTS, log=False)
    print(f"  run digest: {r1['digest']} (ticks={r1['ticks']}, coins={r1['coins']})  两遍一致: {r1['digest'] == r2['digest']}")
    check("调和确定性：两遍运行 digest 逐位一致", r1["digest"] == r2["digest"])
    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")

if __name__ == "__main__":
    main()
