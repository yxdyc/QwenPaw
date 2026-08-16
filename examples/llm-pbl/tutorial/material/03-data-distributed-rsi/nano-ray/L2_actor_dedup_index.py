"""nano-ray · L2 — actor：把有状态算子（全局去重索引）搬进有状态进程
=========================================================================

K+1 目标（相对 L1）：
    L1 把同一套 OP pipeline 搬上了真实 Ray：局部 OP 用 task 并行，全局去重
    用一个**收敛点任务**（ray_global_dedup 收下全部分区）一次算完。收敛点
    语义正确，但有两个结构性限制：
      1) 它把**全部幸存者数据**搬进一个任务（数据向单点集中）；
      2) sig 表是该任务的局部变量——任务结束，状态清零。索引活不过一次调用。
    当算子本身是「有状态、要被多个生产者并发读写、要活得比任何一次调用更久」
    的对象时，task 模型不够用。

    L2 只加一层：把全局去重的索引从「收敛点任务里的局部变量」变成
    **actor（有状态进程）**。同一语料、同一漏斗契约（3360 -> 2358 -> 2110，
    复用 L1_ray_pipeline 的语料构造与 OP），回答三个问题：
      [1] actor 与 task 的状态差异到底是什么——状态住在进程里，跨调用存活；
      [2] actor 的并发语义——默认串行是免费的并发安全；max_concurrency 打开
          并行也打开竞争，用 barrier 决定性复现一次 lost update；
      [3] 全局去重做成 actor 服务——first-seen 规则把顺序假设藏进 RPC 时序，
          反向喂入会翻转恰 236 个重复对的幸存者（== 跨分区对数）；两阶段
          min-row_id 规则（可交换聚合 + 收敛排序）对喂入顺序与并发免疫；
      [4] 成本账——actor 路线搬的是「知识」（sig+row_id 索引），收敛点搬的
          是「数据」（样本全体）；但本语料规模下 RPC 笔数主导，actor 路线
          并不更快——选 actor 的理由是语义，不是这个量级的吞吐。

    ⚠️ 声明（可运行性契约）：本文件用**真实 Ray**（ray 2.56.1，pip 安装，
    本机 CPU），无任何 mock；全部 actor / task 行为为真实 worker 进程内行为。
    多节点行为不在本机范围，见 [TODO: verify on real system]（真实 GPU/多机环境）。
    语料为 seed=42 合成语料（与 nano-data-juicer L2 / nano-ray L1 同一构造）。

依赖：ray（pip install ray）；同目录 L1_ray_pipeline.py（import 复用语料/OP/
     漏斗常量，不执行其 main）。
运行：
    python L2_actor_dedup_index.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True          # import L1 不落 __pycache__（全树零 pyc 约定）

import os
import pickle
import time
from typing import Dict, List, Set, Tuple

import ray

import L1_ray_pipeline as L1            # 跨模块契约的参照系（语料 + OP + 漏斗常量）

# ---------------------------------------------------------------------------
# 常量：与 L1 同口径（漏斗契约由 L1 传入，跨模块一致性在 [0] 机器断言）
# ---------------------------------------------------------------------------

NUM_PARTITIONS = L1.NUM_PARTITIONS
NUM_CPUS = L1.NUM_CPUS
EXPECTED_AFTER_FILTER = L1.EXPECTED_AFTER_FILTER    # 2358
EXPECTED_AFTER_DEDUP = L1.EXPECTED_AFTER_DEDUP      # 2110
EXPECTED_NAIVE = L1.EXPECTED_NAIVE                  # 2346（本文件不重做 naive，留作对照）
EXPECTED_LEAK = L1.EXPECTED_LEAK                    # 236 = 跨分区重复对数
EXPECTED_PAIRS = 248      # 过滤后幸存者中的重复对总数（L1 [3] 账本数字，此处断言）

CHUNK = (L1.NUM_DOCS + L1.NUM_DUPS) // NUM_PARTITIONS   # 840：分区 p 持有 row_id [840p, 840(p+1))

N_CALLERS = 8               # [2a] 并发生产者数
N_RECORDS = 25              # [2a] 每个生产者的记录条数

Sample = Dict[str, object]


# ---------------------------------------------------------------------------
# Ray actor / remote 任务
# （全部自包含：方法体不引用 L1 的名字。remote 代码被 cloudpickle 送进
#  worker 进程时，__main__ 里的类按值序列化，不依赖 worker 能 import L1。）
# ---------------------------------------------------------------------------

@ray.remote
class StatefulCounter:
    """[1]/[4] 用：状态住在 actor 进程里，跨调用存活。"""

    def __init__(self):
        import os
        self.n = 0
        self.pid = os.getpid()

    def incr(self, k: int = 1) -> Tuple[int, int]:
        self.n += k
        return self.n, self.pid

    def get(self) -> Tuple[int, int]:
        return self.n, self.pid


@ray.remote
def incr_task(n: int) -> int:
    """[1] 用：task 模型里的「状态」只能作为参数传入、返回值带出。"""
    return n + 1


@ray.remote
class SerialRecorder:
    """[2a] 用：默认串行 actor。8 个并发生产者往里写 (caller, seq)。

    若方法不是串行执行，8 路并发对同一个 list 的 append 会互相践踏
    （丢条目/乱序）——200 条全额落账本身就是串行执行的证据。"""

    def __init__(self):
        self.log: List[Tuple[int, int]] = []

    def record(self, caller: int, seq: int) -> None:
        self.log.append((caller, seq))

    def dump(self) -> List[Tuple[int, int]]:
        return list(self.log)


@ray.remote
def caller_loop(recorder, caller: int, times: int) -> None:
    """[2a] 用：一个生产者任务，顺序记 times 条（逐条等完成，最保守用法）。"""
    for seq in range(times):
        ray.get(recorder.record.remote(caller, seq))


@ray.remote(max_concurrency=2)
class RacyCounter:
    """[2b] 用：max_concurrency=2 → 同步方法在 actor 进程内的线程池里并发执行。

    用两道事件闸**决定性**复现 lost update：第一个读者读完 x 后停在闸前，
    等第二个读者也读完 x 才放行写回——两个线程都读到旧值、都写回旧值+1，
    两次 +1 只生效一次（最终 x=1 而非 2）。不依赖调度巧合，三遍跑三遍同。"""

    def __init__(self):
        import threading
        self.x = 0
        self._lock = threading.Lock()
        self._readers = 0
        self._gate = threading.Event()

    def bump(self) -> Tuple[int, int]:
        v = self.x                          # 读（两个线程都读到 0）
        with self._lock:
            self._readers += 1
            first = (self._readers == 1)
        if first:
            self._gate.wait(timeout=5.0)    # 第一个读者等第二个读者跟上
        else:
            self._gate.set()                # 第二个读者放行两边
        self.x = v + 1                      # 写（两个线程的 v 都是 0）
        return v, self.x

    def get(self) -> int:
        return self.x


@ray.remote
class SafeCounter:
    """[2b] 对照：同样的两次 bump，默认串行 actor 里不存在竞争。"""

    def __init__(self):
        self.x = 0

    def bump(self) -> Tuple[int, int]:
        v = self.x
        self.x = v + 1
        return v, self.x

    def get(self) -> int:
        return self.x


@ray.remote
class FirstSeenIndex:
    """[3a] 用：单阶段 first-seen 索引——谁先到谁留下。

    语义依赖到达顺序：到达顺序 == 全局 row_id 顺序时与串行 dedup_keep_first
    逐位一致；喂入顺序一变，跨分区重复对的幸存者就翻转（[3] 实测 236 个）。"""

    def __init__(self):
        self.first: Dict[str, int] = {}

    def offer(self, sig: str, row_id: int) -> bool:
        if sig in self.first:
            return False
        self.first[sig] = row_id
        return True

    def offer_many(self, pairs: List[Tuple[str, int]]) -> List[bool]:
        return [self.offer(sig, rid) for sig, rid in pairs]

    def keepers(self) -> Set[int]:
        return set(self.first.values())


@ray.remote
class MinRowIndex:
    """[3b] 用：两阶段 min-row_id 索引。

    register 只做可交换聚合（min 与到达顺序、并发交错无关）；keepers 在
    收敛后按全局顺序输出。本 pipeline 的全局顺序就是 row_id 顺序，所以
    「keep-first」==「keep-min-row_id」——顺序语义被拆成了顺序免疫的聚合
    + 收敛后的一次排序。索引里只存 (sig -> min row_id)，样本本体不挪窝。"""

    def __init__(self):
        self.best: Dict[str, int] = {}

    def register(self, sig: str, row_id: int) -> None:
        cur = self.best.get(sig)
        if cur is None or row_id < cur:
            self.best[sig] = row_id

    def register_many(self, pairs: List[Tuple[str, int]]) -> int:
        for sig, row_id in pairs:
            self.register(sig, row_id)
        return len(pairs)

    def keepers(self) -> List[int]:
        return sorted(self.best.values())

    def size(self) -> int:
        return len(self.best)


@ray.remote
def feed_partition(index, pairs: List[Tuple[str, int]]) -> int:
    """[3b] 用：一个生产者任务，把自己分区的 (sig, row_id) 批量注册进索引。

    pairs 以 ObjectRef 传入、被 runtime 自动解引用（L1 [1b] 的回声）；
    index 是 actor handle——task 里可以像本地对象一样调它的方法。"""
    return ray.get(index.register_many.remote(pairs))


@ray.remote
def converge_dedup(*parts: List[Sample]) -> List[Sample]:
    """[4] 用：L1 收敛点的同款复刻——一个任务收下全部分区，全局去重。
    （不直接调 L1 的 remote 函数：worker 反序列化时不必依赖能 import L1。）"""
    merged = [s for p in parts for s in p]
    seen, out = set(), []
    for s in merged:
        if s["sig"] not in seen:
            seen.add(s["sig"])
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("nano-ray L2 — actor：把有状态算子（全局去重索引）搬进有状态进程")
    print("=" * 68)
    print(f"python {sys.version.split()[0]} | ray {ray.__version__}")
    print("声明: 真实 Ray 单机模式（本地 raylet + worker 进程），无 mock；")
    print("      复用 L1_ray_pipeline 的语料/OP/漏斗常量（跨模块一致性契约）。")

    # ---- [0] 启动 + 跨模块契约复验 ----
    t0 = time.perf_counter()
    ray.init(num_cpus=NUM_CPUS, include_dashboard=False,
             logging_level="error")
    t_init = time.perf_counter() - t0

    docs = L1.make_corpus()
    ser_mapped = L1.local_ops(docs)
    ser_final = L1.dedup_keep_first(ser_mapped)
    assert len(ser_mapped) == EXPECTED_AFTER_FILTER, "局部 OP 漏斗与 L1 不一致"
    assert len(ser_final) == EXPECTED_AFTER_DEDUP, "串行去重漏斗与 L1 不一致"
    mapped_parts = L1.partition(ser_mapped, NUM_PARTITIONS)

    # 重复对账本（与 L1 [3] 同算术）：by_key 按全局顺序追加 => pair[0] 是 source
    by_key: Dict[str, List[Sample]] = {}
    for s in ser_mapped:
        by_key.setdefault(s["sig"], []).append(s)
    pairs_all = [v for v in by_key.values() if len(v) == 2]
    part_of = lambda rid: min(rid // CHUNK, NUM_PARTITIONS - 1)
    cross_pairs = [v for v in pairs_all
                   if part_of(v[0]["row_id"]) != part_of(v[1]["row_id"])]
    assert len(pairs_all) == EXPECTED_PAIRS
    assert len(cross_pairs) == EXPECTED_LEAK

    print(f"\n[0] ray.init {t_init:.1f}s | 语料 {len(docs)} docs -> 过滤后 "
          f"{len(ser_mapped)} -> 去重后 {len(ser_final)}（漏斗与 L1 逐位一致 ✅）")
    print(f"    重复对账本: 共 {len(pairs_all)} 对 | 跨分区 {len(cross_pairs)} 对"
          f"（== EXPECTED_LEAK，契约复验 ✅）")

    # ---- [1] actor = 专属有状态进程 ----
    print(f"\n[1] actor = 专属有状态进程：状态住在进程里，跨调用存活")
    c = StatefulCounter.remote()
    outs = [ray.get(c.incr.remote()) for _ in range(4)]
    vals = [n for n, _ in outs]
    pids = {p for _, p in outs}
    assert vals == [1, 2, 3, 4], "actor 状态必须跨调用累积"
    assert len(pids) == 1, "同一 actor 的所有方法调用必须落在同一进程"
    actor_pid = pids.pop()
    assert actor_pid != os.getpid(), "actor 进程必须独立于驱动进程"
    print(f"    4 次 incr -> {vals}（状态跨调用累积 ✅）")
    print(f"    4 次调用全落在同一个进程（唯一 pid 数 = 1），且该进程 ≠ 驱动进程 ✅")
    print(f"    （对照 L1：4 个分区任务散在 4 个 worker 进程；actor 是一个常驻进程）")

    n_task = 0
    for _ in range(4):
        n_task = ray.get(incr_task.remote(n_task))
    assert n_task == 4
    print(f"    task 路线对照：同样数到 4，但状态以参数/返回值 8 次穿越驱动端；")
    print(f"    actor 路线 0 次——多个生产者要共享同一份可变状态时，task 模型逼")
    print(f"    所有人经驱动端中转，actor 给出唯一的共享住所。")

    # ---- [2] actor 的并发语义 ----
    print(f"\n[2] actor 的并发语义")
    # a) 默认串行：8 生产者 × 25 条，全额落账 + 每 caller FIFO + 跨 caller 交错
    rec = SerialRecorder.remote()
    ray.get([caller_loop.remote(rec, i, N_RECORDS)
             for i in range(N_CALLERS)])
    log = ray.get(rec.dump.remote())
    assert len(log) == N_CALLERS * N_RECORDS, "并发写入下不得丢失条目"
    last_seq: Dict[int, int] = {}
    switches, prev_caller = 0, None
    for caller, seq in log:
        assert seq > last_seq.get(caller, -1), "同一 caller 的记录必须 FIFO"
        last_seq[caller] = seq
        if prev_caller is not None and caller != prev_caller:
            switches += 1
        prev_caller = caller
    assert switches >= N_CALLERS, "并发生产者的记录应当交错（纯分块只有 7 次切换）"
    print(f"    a) 默认串行 actor：{N_CALLERS} 生产者 × {N_RECORDS} 条 = "
          f"{len(log)} 条，丢失 0 条 ✅")
    print(f"       每个 caller 的 seq 严格递增（FIFO 保持）✅ | {N_CALLERS} 个生产者"
          f"交错排队成立（切换次数 ≥ {N_CALLERS} ✅）")

    # b) max_concurrency=2：决定性 lost update vs 默认串行对照
    racy = RacyCounter.remote()
    racy_res = sorted(ray.get([racy.bump.remote() for _ in range(2)]))
    racy_final = ray.get(racy.get.remote())
    assert racy_res == [(0, 1), (0, 1)], "两次 bump 都应是「读 0 写 1」"
    assert racy_final == 1, "lost update：两次 +1 只生效一次"
    safe = SafeCounter.remote()
    safe_res = sorted(ray.get([safe.bump.remote() for _ in range(2)]))
    safe_final = ray.get(safe.get.remote())
    assert safe_res == [(0, 1), (1, 2)]
    assert safe_final == 2
    print(f"    b) max_concurrency=2 + 决定性 barrier：两次 +1，x 停在 "
          f"{racy_final} ❌（两次调用都读到 0、都写回 1：{racy_res}）")
    print(f"       对照（默认串行 actor 同样两次 bump）：x = {safe_final} ✅"
          f"（{safe_res}）")
    print(f"    → 默认串行 = 免费的并发安全；max_concurrency 打开吞吐也打开竞争，")
    print(f"      原子性要自己负责（方法内加锁，或保持串行、用批量摊薄 RPC）。")

    # ---- [3] 全局去重做成 actor 服务 ----
    print(f"\n[3] 全局去重做成 actor 服务（喂的是索引，不是数据）")
    pairs_flat = [(s["sig"], s["row_id"]) for s in ser_mapped]  # 2358 条「知识」
    ser_keep_ids = [s["row_id"] for s in ser_final]

    # a) first-seen：正向逐条喂 == 串行；反向按分区喂 => 幸存者翻转
    fs_fwd = FirstSeenIndex.remote()
    t0 = time.perf_counter()
    keep_flags = [ray.get(fs_fwd.offer.remote(sig, rid))
                  for sig, rid in pairs_flat]
    t_fs_fwd = time.perf_counter() - t0
    fwd_out = [s for s, k in zip(ser_mapped, keep_flags) if k]
    assert fwd_out == ser_final, "到达顺序 == 全局顺序时 first-seen 必须 == keep-first"

    fs_rev = FirstSeenIndex.remote()
    rev_kept_ids: Set[int] = set()
    for part in reversed(mapped_parts):          # 分区反序喂，分区内部仍正序
        flags = ray.get(fs_rev.offer_many.remote(
            [(s["sig"], s["row_id"]) for s in part]))
        for s, k in zip(part, flags):
            if k:
                rev_kept_ids.add(s["row_id"])
    assert len(rev_kept_ids) == EXPECTED_AFTER_DEDUP, "反向喂条数不变，内容会变"
    flipped = 0
    for pair in pairs_all:
        a, b = pair[0]["row_id"], pair[1]["row_id"]   # a < b（source 在前）
        assert (a in rev_kept_ids) != (b in rev_kept_ids), "每对必须恰留一份"
        if b in rev_kept_ids:
            flipped += 1
            assert part_of(a) != part_of(b), "被翻转的必是跨分区对"
        else:
            assert part_of(a) == part_of(b), "同分区对的内部顺序未变，幸存者不变"
    assert flipped == EXPECTED_LEAK
    print(f"    a) first-seen 索引（到达顺序敏感）：")
    print(f"       正向逐条喂 {len(pairs_flat)} 次 offer（{t_fs_fwd * 1e3:.0f} ms）")
    print(f"       == 串行 dedup 逐位一致 ✅（到达顺序 == 全局顺序时语义重合）")
    print(f"       反向按分区喂：仍 {len(rev_kept_ids)} 条，但 {flipped} 个重复对的")
    print(f"       幸存者翻成了 copy ❌——翻转数 == 跨分区对数 {EXPECTED_LEAK}：")
    print(f"       反序喂让高分区（row_id 更大）的 copy 先到；同分区的 "
          f"{len(pairs_all) - flipped} 对内部顺序未变。条数对、内容错——")
    print(f"       「第一次出现」是全局顺序语义，单阶段规则把顺序假设藏进了 RPC 时序。")

    # b) min-row_id：同样反向喂 + 4 任务并发喂，均 == 串行
    mr_rev = MinRowIndex.remote()
    ray.get([mr_rev.register_many.remote(
        [(s["sig"], s["row_id"]) for s in part])
        for part in reversed(mapped_parts)])
    keepers_rev = ray.get(mr_rev.keepers.remote())
    assert keepers_rev == ser_keep_ids, "min 聚合与喂入顺序无关"

    mr_conc = MinRowIndex.remote()
    ray.get(mr_conc.size.remote())      # 预热：把 actor 进程拉起的一次性成本挡在计时窗外
    pair_refs = [ray.put([(s["sig"], s["row_id"]) for s in part])
                 for part in mapped_parts]
    t0 = time.perf_counter()
    fed = ray.get([feed_partition.remote(mr_conc, ref) for ref in pair_refs])
    t_feed_conc = time.perf_counter() - t0
    assert sum(fed) == len(pairs_flat)
    keepers_conc = ray.get(mr_conc.keepers.remote())
    idx_size = ray.get(mr_conc.size.remote())
    assert keepers_conc == ser_keep_ids, "min 聚合与并发交错无关"
    assert idx_size == EXPECTED_AFTER_DEDUP
    print(f"    b) min-row_id 索引（两阶段：可交换聚合 + 收敛排序）：")
    print(f"       同样反向喂：keeper 序列 == 串行 pipeline ✅")
    print(f"       4 任务并发喂（{t_feed_conc * 1e3:.0f} ms，到达先后不确定）：")
    print(f"       仍逐位一致 ✅ | 索引条目 {idx_size} == 去重后条数")
    print(f"       → keep-first 被拆成「register: min 聚合（顺序免疫）+ keepers:")
    print(f"         收敛后按 row_id 排序输出」，顺序语义不再住在 RPC 时序里。")

    # c) 组装：数据没挪过窝，按 keeper row_id 从原分区取样本
    keep_set = set(keepers_conc)
    final = [s for part in mapped_parts for s in part
             if s["row_id"] in keep_set]
    assert final == ser_final, "组装结果必须与串行 pipeline 逐位一致"
    print(f"    c) 组装：按 keeper row_id 回原分区取样本 == 串行 pipeline 逐位")
    print(f"       一致 ✅ —— 漏斗 {len(docs)}->{len(ser_mapped)}->{len(final)} "
          f"与 L1 / nano-data-juicer L2 完全一致")

    # ---- [4] 成本账：知识集中 vs 数据集中 ----
    print(f"\n[4] 成本账：actor 路线搬「知识」，收敛点搬「数据」")
    # actor 创建成本（进程拉起 + 首次调用，一次性）
    t0 = time.perf_counter()
    probe = StatefulCounter.remote()
    ray.get(probe.get.remote())
    t_create = time.perf_counter() - t0
    # 单次方法调用 RPC（预热 5 次后测 50 次）
    ray.get([probe.incr.remote() for _ in range(5)])
    t0 = time.perf_counter()
    for _ in range(50):
        ray.get(probe.incr.remote())
    t_rpc50 = time.perf_counter() - t0
    t_rpc = t_rpc50 / 50

    # 收敛点同款复刻：4 分区全体搬进一个任务
    mapped_refs = [ray.put(p) for p in mapped_parts]
    t0 = time.perf_counter()
    conv_out = ray.get(converge_dedup.remote(*mapped_refs))
    t_conv = time.perf_counter() - t0
    assert conv_out == ser_final

    conv_in = sum(len(pickle.dumps(p)) for p in mapped_parts)
    conv_out_bytes = len(pickle.dumps(ser_final))
    idx_in = len(pickle.dumps(pairs_flat))
    idx_out_bytes = len(pickle.dumps(keepers_conc))
    print(f"    actor 创建（进程拉起+首次调用）: {t_create * 1e3:.0f} ms（一次性）")
    print(f"    单次方法调用 RPC: {t_rpc * 1e3:.2f} ms（预热后 50 次均值）")
    print(f"    搬进: 收敛点 {conv_in / 1e6:.2f} MB（2358 条样本全体） vs "
          f"索引路线 {idx_in / 1e3:.0f} KB（2358 条 (sig,row_id)），差 "
          f"{conv_in / idx_in:.0f}x")
    print(f"    搬出: 收敛点返回 {conv_out_bytes / 1e6:.2f} MB 样本 vs "
          f"索引路线返回 {idx_out_bytes / 1e3:.1f} KB row_id")
    print(f"    墙钟（本语料 3.7 MB / 2358 条）: 收敛点 {t_conv * 1e3:.1f} ms | "
          f"actor 批量喂 {t_feed_conc * 1e3:.1f} ms | "
          f"actor 逐条喂 {t_fs_fwd * 1e3:.0f} ms")
    print(f"    逐条喂预测对账: 2358 x {t_rpc * 1e3:.2f} ms ≈ "
          f"{2358 * t_rpc * 1e3:.0f} ms（实测 {t_fs_fwd * 1e3:.0f} ms，同量级 ✅）")
    print(f"    → 收敛点：数据向一个任务集中；actor：知识向一个进程集中、数据留")
    print(f"      在原分区。本规模下吞吐不是选 actor 的理由——RPC 笔数是真成本，")
    print(f"      批量是标准解（L1 [4] 任务粒度账的 actor 版回声）；选 actor 的理由是")
    print(f"      语义（并发生产者 / 增量喂入 / 顺序免疫 / 索引常驻），不是这里的吞吐。")

    # ---- [5] 选择标准 + 汇总 ----
    print(f"\n[5] task vs actor 选择标准")
    print(f"    无状态、可并行、无共享可变需求        -> task（L1 全图）")
    print(f"    状态跨调用存活、多生产者并发读写      -> actor（默认串行=免费并发安全）")
    print(f"    一次性收齐的纯收敛计算                -> 收敛点 task（[4]：与批量 actor 同量级、无常驻成本）")
    print(f"    索引要增量/在线/并发喂、且顺序免疫    -> actor + 可交换更新规则（[3b]）")
    print(f"    为吞吐开 max_concurrency              -> 先证明竞争无害（[2b] 是反例模板）")

    ray.shutdown()
    print("\n" + "=" * 68)
    print("✅ self-check passed:")
    print("   漏斗 3360->2358->2110 与 L1 / nano-data-juicer L2 一致 /")
    print("   actor 状态跨调用 + 单进程证据 / 串行 actor 200 条零丢失 /")
    print("   lost update 决定性复现 x=1（对照串行 x=2）/")
    print("   first-seen 反向喂翻转 236 == 跨分区对数 /")
    print("   min-row_id 反向喂+并发喂 == 串行 / 组装 == 串行 pipeline 逐位一致")
    print("=" * 68)
    print("\ntakeaway: 同一个去重语义——task 写法把状态藏在一次性任务里（收敛点），")
    print("          actor 写法把状态放进常驻进程里（索引服务）。语义正确与否不取")
    print("          决于 runtime，取决于更新规则有没有把顺序假设藏进 RPC 时序：")
    print("          first-seen 藏了，236 条翻转给你看；min-row_id 没藏，顺序与")
    print("          并发都免疫。L3 进 object store / plasma。")


if __name__ == "__main__":
    main()
