"""nano-ray · L0 玩具实现
====================================

目标：用 ~270 行纯 Python 抓住 Ray 编程模型的三个核心机制——
    ① remote function：submit 立即返回 future（ObjectRef），先描述图、后取结果；
    ② object store：数据 put 一次、按引用传给任务，N 个任务不复制 N 份；
    ③ dynamic task graph：依赖由调度器在任务完成时自动触发下游，无需手工排序。

五个实验：独立任务并行 / map-reduce 动态任务图 / 传引用 vs 传值 / 细粒度反例 /
触发时序回归（含修复前顺序的确定性丢触发反例）。
这是 L0（玩具级）：单文件、纯标准库、CPU 即跑，自检全部带 assert。
真实 Ray（进程级 worker、plasma object store、ownership 调度器）见 L1–L3
[TODO: verify source]。参考：Ray 论文 arXiv:1712.05889（Moritz et al.）。

运行：
    python L0_task_graph.py
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

# ===== 机制一的对象：ObjectRef = object store 里对象的句柄（future）=====


class ObjectRef:
    """任务间传递的不是数据本身，而是这个句柄。get() 阻塞到结果就绪——future 语义。"""
    __slots__ = ("oid", "_event", "_value")
    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self) -> None:
        with ObjectRef._id_lock:
            ObjectRef._next_id += 1
            self.oid = ObjectRef._next_id
        self._event = threading.Event()
        self._value = None

    def _set(self, value: Any) -> None:
        self._value = value
        self._event.set()

    def get(self) -> Any:
        self._event.wait()
        return self._value

    def ready(self) -> bool:
        return self._event.is_set()

    def __repr__(self) -> str:
        return f"ObjectRef({self.oid})"


# ===== 机制二/三的对象：object store + 动态任务图调度器 =====


class Task:
    __slots__ = ("fn", "args", "out", "deps", "started")

    def __init__(self, fn: Callable, args: tuple, out: ObjectRef) -> None:
        self.fn, self.args, self.out = fn, args, out
        self.deps = [a for a in args if isinstance(a, ObjectRef)]
        self.started = False


class ToyScheduler:
    """最小动态任务图调度器：依赖就绪即提交 worker；任务完成再触发下游。

    与真实 Ray 的对应（概念层）：submit ≈ @ray.remote 调用，
    put/get ≈ ray.put/ray.get，完成触发 ≈ ownership 调度器的依赖解析。
    """

    def __init__(self, n_workers: int) -> None:
        self.pool = ThreadPoolExecutor(max_workers=n_workers)
        self.tasks: List[Task] = []
        self.lock = threading.Lock()
        self.finished: List[int] = []          # 完成顺序（oid），供观察依赖是否被尊重

    def put(self, value: Any) -> ObjectRef:
        """把对象放进 store 一次，之后所有任务按 ref 取用（对应 ray.put）。"""
        ref = ObjectRef()
        ref._set(value)
        return ref

    def submit(self, fn: Callable, *args: Any) -> ObjectRef:
        """remote 调用：立即返回未来结果的 ObjectRef（不阻塞、不执行）。"""
        task = Task(fn, args, ObjectRef())
        with self.lock:
            self.tasks.append(task)
        self._try_start(task)
        return task.out

    def _try_start(self, task: Task) -> None:
        with self.lock:
            if task.started or not all(d.ready() for d in task.deps):
                return
            task.started = True
        self.pool.submit(self._run, task)

    def _run(self, task: Task) -> None:
        vals = [a.get() if isinstance(a, ObjectRef) else a for a in task.args]
        result = task.fn(*vals)
        with self.lock:                        # 先记账再放行：finished 顺序严格尊重依赖
            self.finished.append(task.out.oid)
        task.out._set(result)
        # 触发时序不变量：pending 快照必须在 _set 之后取。结合「submit 先入 tasks
        # 再查依赖就绪」，两种交错都被覆盖：此前提交的依赖任务要么已见依赖就绪而
        # 自行启动，要么落进这份快照被我们触发——submit ⇒ 必执行（实验 [5] 回归）。
        with self.lock:
            pending = [t for t in self.tasks if not t.started]
        for t in pending:                      # 完成即触发下游——动态图的关键
            self._try_start(t)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


class LegacyScheduler(ToyScheduler):
    """仅用作反例：修复前的触发顺序——pending 快照取在 _set 之前。

    竞态窗口 = 快照 → _set 之间。窗口内提交的依赖任务：自己的 _try_start
    见依赖未就绪而返回，快照里又没有它 → 触发永久丢失（孤儿任务 / 下游挂死）。
    widen_sec 人为拉宽窗口以便确定性复现（实验 [5b]）；这就是修复前的真实代码路径。
    """

    def __init__(self, n_workers: int, widen_sec: float = 0.3) -> None:
        super().__init__(n_workers)
        self.widen_sec = widen_sec

    def _run(self, task: Task) -> None:
        vals = [a.get() if isinstance(a, ObjectRef) else a for a in task.args]
        result = task.fn(*vals)
        with self.lock:
            self.finished.append(task.out.oid)
            pending = [t for t in self.tasks if not t.started]   # ← 缺陷：快照过早
        time.sleep(self.widen_sec)                               # 拉宽竞态窗口
        task.out._set(result)
        for t in pending:
            self._try_start(t)


# ===== 任务负载 =====


def io_heavy_count(chunk: str, io_sec: float) -> int:
    """模拟「读一个数据分片 + 统计」：sleep 显式模拟 IO 耗时（释放 GIL），ord 和是真实小计算。"""
    time.sleep(io_sec)
    return sum(ord(c) for c in chunk)


def add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    print("=" * 64)
    print("nano-ray L0 — task graph / object store / 调度")
    print("=" * 64)

    text = ("data juicer ray verl megatron fsdp " * 400)
    chunks = [text[i::8] for i in range(8)]                    # 8 个数据分片

    # ---- [1] 独立任务并行：先提交整张图，再统一取结果 ----
    sched = ToyScheduler(n_workers=4)
    t0 = time.perf_counter()
    refs = [sched.submit(io_heavy_count, c, 0.15) for c in chunks]
    par_results = [r.get() for r in refs]
    t_par = time.perf_counter() - t0
    t0 = time.perf_counter()
    ser_results = [io_heavy_count(c, 0.15) for c in chunks]
    t_ser = time.perf_counter() - t0
    assert par_results == ser_results, "并行结果必须与串行一致"
    speedup = t_ser / t_par
    print(f"\n[1] 8 个独立任务（各模拟 0.15s IO），4 workers")
    print(f"    serial   = {t_ser:5.2f}s | parallel = {t_par:5.2f}s | speedup = {speedup:.2f}x")
    print(f"    结果与串行逐位一致 ✅（submit 立即返回，get 才阻塞）")
    assert speedup > 2.5, "4 worker 跑 8 个 IO 任务，加速应接近 4x"

    # ---- [2] map-reduce 动态任务图：完成即触发下游 ----
    sched2 = ToyScheduler(n_workers=4)
    map_refs = [sched2.submit(io_heavy_count, c, 0.05) for c in chunks]
    edges, level = [], map_refs
    while len(level) > 1:                                      # 两两归约成树
        nxt = []
        for i in range(0, len(level), 2):
            parent = sched2.submit(add, level[i], level[i + 1])
            edges.append((parent.oid, [level[i].oid, level[i + 1].oid]))
            nxt.append(parent)
        level = nxt
    total = level[0].get()
    assert total == sum(ser_results), "归约结果必须与串行一致"
    pos = {oid: i for i, oid in enumerate(sched2.finished)}
    for parent, children in edges:                             # 父必晚于所有子
        assert pos[parent] > max(pos[c] for c in children), "依赖顺序被破坏"
    print(f"\n[2] map-reduce 任务图：8 map + 7 reduce = 15 个节点，全部自动调度")
    print(f"    total = {total}（与串行一致 ✅），完成顺序 oids = {sched2.finished}")
    print(f"    每个 reduce 都在其两个 map 之后完成 ✅（依赖由调度器保证，非手工排序）")

    # ---- [3] object store：传引用 vs 传值 ----
    big = list(range(200_000))                                 # ~200k 元素的"大"对象
    big_ref = sched2.put(big)
    ids = [sched2.submit(lambda x: id(x), big_ref) for _ in range(4)]
    seen = [r.get() for r in ids]
    assert all(s == id(big) for s in seen), "任务应拿到 store 里同一个对象"
    print(f"\n[3] object store：put 一次 {len(big):,} 元素，4 个任务按 ref 取用")
    print(f"    identity 检查：4 个任务拿到的是同一个对象（id={id(big)}）✅")
    print(f"    传值语义要拷 4 份 = {4 * len(big):,} 元素；store 只存 1 份 + 传 4 个句柄")
    print(f"    → 真实 Ray 里这对应 plasma 共享内存：跨进程也不拷贝 [TODO: verify source]")

    # ---- [4] 反例：任务粒度太细，调度开销吃掉收益 ----
    micro = ToyScheduler(n_workers=8)
    t0 = time.perf_counter()
    rs = [micro.submit(add, i, 1) for i in range(1000)]        # 每个任务 ~µs 级
    _ = [r.get() for r in rs]
    t_micro_par = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = [add(i, 1) for i in range(1000)]
    t_micro_ser = time.perf_counter() - t0
    micro_speedup = t_micro_ser / t_micro_par
    print(f"\n[4] 反例：1000 个 µs 级小任务，8 workers")
    print(f"    serial = {t_micro_ser * 1e3:6.2f}ms | parallel = {t_micro_par * 1e3:6.2f}ms"
          f" | speedup = {micro_speedup:.2f}x")
    print(f"    对照 [1] 的 {speedup:.2f}x：任务太细时，submit/队列/线程唤醒的开销")
    print(f"    比任务本身还贵——remote 化只值得花在「单个足够重」的任务上")
    assert micro_speedup < speedup, "细粒度任务的加速比必须显著低于粗粒度"

    # ---- [5] 回归：触发时序——submit ⇒ 必执行 ----
    # (a) 修复后顺序：300 个随机 DAG，图生长途中动态提交，不允许任何丢触发
    rng = random.Random(0)
    for trial in range(300):
        s = ToyScheduler(n_workers=rng.randint(1, 4))
        prev = [s.put(0)]
        for _ in range(rng.randint(2, 4)):
            prev = [s.submit(add, p, 1) for p in prev for _ in range(rng.randint(1, 2))]
            prev[0].get()                    # 等部分上游完成，再动态提交下一层
        deadline = time.perf_counter() + 5.0
        for r in prev:
            while not r.ready():
                assert time.perf_counter() < deadline, f"trial {trial}: 丢触发 {r}"
                time.sleep(0.001)
        s.shutdown()
    print(f"\n[5] 触发时序回归：300 个随机 DAG + 动态提交，全部任务完成 ✅")

    # (b) 反例：修复前顺序确定性丢触发——窗口内提交的 B 永远不被触发
    legacy = LegacyScheduler(n_workers=2)
    a = legacy.submit(lambda: 0)
    while not legacy.finished:               # 等 A 进入「快照已取、尚未放行」的窗口
        time.sleep(0.001)
    b = legacy.submit(add, a, 1)             # B 在窗口内提交（窗口被拉宽到 0.3s）
    assert a.get() == 0                      # A 正常完成
    time.sleep(0.05)                         # 宽限：若有任何触发，此刻必已发生
    assert not legacy.tasks[-1].started and not b.ready(), "修复前顺序应丢掉 B 的触发"
    print(f"    反例（修复前顺序）：窗口内提交的 B 永远不被触发，成为孤儿 ❌（确定性复现）")
    fixed = ToyScheduler(n_workers=2)
    a2 = fixed.submit(lambda: 0)
    while not fixed.finished:
        time.sleep(0.001)
    b2 = fixed.submit(add, a2, 1)            # 同样时机，修复后顺序
    assert b2.get() == 1                     # B 被放行后的快照捕获，正常完成
    print(f"    同一时机，修复后顺序：B 被放行后的快照捕获，正常完成 ✅（submit ⇒ 必执行）")
    legacy.shutdown(); fixed.shutdown()

    sched.shutdown(); sched2.shutdown(); micro.shutdown()
    print("\n" + "=" * 64)
    print("✅ self-check passed: 并行等价 / 依赖顺序正确 / 零拷贝 identity / 粒度反例成立")
    print("                      + 触发时序回归通过（submit ⇒ 必执行；旧顺序确定性复现丢触发）")
    print("=" * 64)
    print("\ntakeaway: Ray 的编程模型 = remote 函数（future）+ object store（传引用）")
    print("          + 动态任务图（完成即触发）。L1 用真 Ray 把 data-juicer pipeline 跑起来。")


if __name__ == "__main__":
    main()
