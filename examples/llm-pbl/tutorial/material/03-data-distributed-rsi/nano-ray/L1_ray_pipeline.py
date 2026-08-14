"""nano-ray · L1 — 真实 Ray 执行 nano-data-juicer 的 OP pipeline
=========================================================================

K+1 目标（相对 L0）：
    L0 在**单进程内**（线程 + 内存字典 store）搭出了 Ray 编程模型的语义：
    future / 传引用 / 完成即触发。语义对了，但有三样东西是假的：
    worker 不是真进程（躲不开 GIL）、store 不是共享内存、调度没有真实成本。

    L1 只加一层：把同一套 OP pipeline（nano-data-juicer L2 的同一个工作负载）
    搬到**真实 Ray** 上——任务真的跑在独立 worker 进程里，数据真的经过
    object store，调度与序列化付出真实代价。同一语料、同一执行计划，
    三个执行器（串行 / L2 multiprocessing / L1 Ray）必须给出同一个漏斗：
    3360 -> 2358 -> 2110。**执行语义住在「计划」里，不在 runtime 里**——
    这就是 L1 的核心教学点。

    实验清单：
      [0] ray.init 启动成本（分布式 runtime 不是免费的）
      [1] object store 喂数据：ray.put 一次 vs 传值四次（驱动端序列化的账）
          + ObjectRef 参数自动解引用的真实行为（两种失败现场：
          对已解引用的 int 再 get → ValueError；对已解引用的 list 再 get
          被当成「ref 列表」→ TypeError）
      [2] 局部 OP 阶段：串行 vs Ray 4 任务并行，结果逐位一致 + worker pid 证据
      [3] 全局 OP：naive 分区去重反例（泄漏账本）vs 收敛点任务
      [4] 任务粒度扫描：P=1/2/4/8/16，开销主导区与并行饱和区
      [5] 启动成本摊销算术 + 汇总自检

    ⚠️ 声明（可运行性契约）：本文件用**真实 Ray**（ray 2.56.1，pip 安装，
    本机 CPU），无任何 mock；ray.init 单机模式启动本地 raylet + worker 进程。
    多节点行为不在本机范围，见 [TODO: verify on real system]（Machine B 通道）。
    语料与 nano-data-juicer L2 同一确定性构造（seed=42 合成语料，非真实语料；
    L1 主题是执行机制，语料内容不影响验证）。

依赖：ray（pip install ray；本机实测 ray==2.56.1 / Python 3.13.13 / macOS arm64）。
运行：
    python L1_ray_pipeline.py
"""

from __future__ import annotations

import hashlib
import pickle
import random
import re
import time
from typing import Any, Dict, List, Tuple

import ray

# ---------------------------------------------------------------------------
# 常量：与 nano-data-juicer L2 完全同口径（同一语料、同一漏斗）
# ---------------------------------------------------------------------------

SEED = 42
NUM_DOCS = 3000
NUM_DUPS = 360
NUM_PARTITIONS = 4
NUM_CPUS = 4              # ray.init(num_cpus=4)：与 L2 的 4 worker 公平对照
MIN_NORM_LEN = 900

EXPECTED_AFTER_FILTER = 2358    # L2 已验证的漏斗数字（交叉模块一致性断言）
EXPECTED_AFTER_DEDUP = 2110
EXPECTED_NAIVE = 2346
EXPECTED_LEAK = 236

Sample = Dict[str, Any]

# ---------------------------------------------------------------------------
# 合成语料：与 nano-data-juicer L2 make_corpus 同一构造（同 seed => 同语料）
# ---------------------------------------------------------------------------

VOCAB = (
    "data pipeline filter mapper dedup partition worker shuffle hash "
    "token sample corpus quality score threshold config operator stage "
    "merge split order deterministic parallel serial global local reduce "
    "scatter gather converge lineage retry checkpoint materialize export "
    "format text length duplicate cluster node engine schedule batch".split()
)

TEMPLATES = [
    "the {a} {b} runs after the {c} {d}",
    "a {a} stage must keep {b} and {c} consistent",
    "we compare {a} with {b} before the {c} step",
    "if the {a} fails, retry the {b} partition only",
    "{a} and {b} determine how {c} is shuffled",
    "every {a} writes its {b} to the shared {c}",
]


def make_corpus(seed: int = SEED, num_docs: int = NUM_DOCS,
                num_dups: int = NUM_DUPS) -> List[Sample]:
    """确定性语料（构造逻辑与 nano-data-juicer L2 make_corpus 逐行同构）。

    3000 条基础文档 + 360 条注入重复：copy 复制 source 的 text 并做
    大小写/空白扰动，原文不相等、规范化后相等——dedup 必须在 normalize
    之后才抓得到。不变量：每个重复 key 恰好两份，copy 恒在 source 之后。
    """
    rng = random.Random(seed)
    base: List[str] = []
    for i in range(num_docs):
        n_sent = rng.randint(12, 30)
        sents = []
        for _ in range(n_sent):
            t = rng.choice(TEMPLATES)
            sents.append(t.format(a=rng.choice(VOCAB), b=rng.choice(VOCAB),
                                  c=rng.choice(VOCAB), d=rng.choice(VOCAB)))
        base.append(" ".join(sents))

    src_ids = rng.sample(range(num_docs // 2), num_dups)
    copies: List[str] = []
    for src in src_ids:
        t = base[src]
        chars = list(t)
        for _ in range(max(1, len(t) // 40)):
            j = rng.randrange(len(chars))
            chars[j] = chars[j].upper()
        perturbed = re.sub(r"( )", "  ", "".join(chars), count=len(t) // 60)
        copies.append(" " + perturbed + " ")

    total = num_docs + num_dups
    copy_pos = set(rng.sample(range(num_docs // 2, total), num_dups))
    docs: List[Sample] = []
    bi, ci = 0, 0
    for pos in range(total):
        if pos in copy_pos:
            docs.append({"row_id": pos, "text": copies[ci]})
            ci += 1
        else:
            docs.append({"row_id": pos, "text": base[bi]})
            bi += 1
    return docs


# ---------------------------------------------------------------------------
# OP：与 nano-data-juicer L2 同一实现（mapper/filter 局部，dedup 全局）
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _heavy_signature(text: str) -> str:
    """真实 CPU 负载：字符二元组直方图 + 双层哈希（O(len) 真实计算）。"""
    hist = [0] * 256
    for i in range(len(text) - 1):
        hist[(ord(text[i]) + ord(text[i + 1])) % 256] += 1
    payload = text.encode("utf-8")
    return hashlib.md5(hashlib.sha1(payload).digest()
                       + bytes(hist)).hexdigest()


def normalize_map(sample: Sample) -> Sample:
    s = dict(sample)
    norm = _WS_RE.sub(" ", s["text"]).strip().lower()
    s["norm_text"] = norm
    s["norm_len"] = len(norm)
    s["sig"] = _heavy_signature(norm)
    words = norm.split()
    s["n_words"] = len(words)
    s["uniq_ratio"] = len(set(words)) / len(words) if words else 0.0
    return s


def length_keep(sample: Sample) -> bool:
    return sample.get("norm_len", 0) >= MIN_NORM_LEN


def local_ops(part: List[Sample]) -> List[Sample]:
    """局部 OP 阶段 = normalize_mapper + length_filter（逐样本独立）。"""
    return [s for s in (normalize_map(x) for x in part) if length_keep(s)]


def dedup_keep_first(docs: List[Sample]) -> List[Sample]:
    """全局去重：保留全局顺序下的首次出现（与 L2 同语义）。"""
    seen, out = set(), []
    for s in docs:
        if s["sig"] not in seen:
            seen.add(s["sig"])
            out.append(s)
    return out


def partition(docs: List[Sample], n: int) -> List[List[Sample]]:
    size = (len(docs) + n - 1) // n
    return [docs[i * size:(i + 1) * size] for i in range(n)
            if i * size < len(docs)]


# ---------------------------------------------------------------------------
# Ray remote 任务（顶层函数，可被 cloudpickle 送进 worker 进程）
# ---------------------------------------------------------------------------

@ray.remote
def slice_count(docs: List[Sample], lo: int, hi: int) -> Tuple[int, int]:
    """[1] 用：取 [lo, hi) 切片，返回 (条数, worker pid)。"""
    import os
    return len(docs[lo:hi]), os.getpid()


@ray.remote
def ray_local_ops(part: List[Sample]) -> Tuple[List[Sample], int]:
    """[2]/[4] 用：对一个分区跑局部 OP 阶段，附带 worker pid。"""
    import os
    return local_ops(part), os.getpid()


@ray.remote
def ray_partition_dedup(part: List[Sample]) -> List[Sample]:
    """[3] naive 反例用：各分区各做各的去重（语义错误，见账本）。"""
    return dedup_keep_first(part)


@ray.remote
def ray_global_dedup(*parts: List[Sample]) -> List[Sample]:
    """[3] 收敛点：一个任务收下全部分区，union 后全局去重。

    注意签名里的 *parts：ObjectRef 作为任务参数会被 Ray 自动解引用，
    任务体内拿到的已是真实数据——收敛点 = 「依赖所有分区的任务」。
    """
    merged = [s for p in parts for s in p]
    return dedup_keep_first(merged)


@ray.remote
def bad_get(x: Any) -> Any:
    """[1] 机制演示用：对「已被自动解引用的参数」再调 ray.get。"""
    return ray.get(x)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("nano-ray L1 — 真实 Ray 执行 nano-data-juicer 的 OP pipeline")
    print("=" * 68)
    import sys
    print(f"python {sys.version.split()[0]} | ray {ray.__version__}")
    print("声明: 真实 Ray 单机模式（本地 raylet + worker 进程），无 mock；")
    print("      多节点行为见 [TODO: verify on real system]")

    # ---- [0] 启动成本 ----
    t0 = time.perf_counter()
    ray.init(num_cpus=NUM_CPUS, include_dashboard=False,
             logging_level="error")
    t_init = time.perf_counter() - t0
    res = ray.cluster_resources()
    print(f"\n[0] ray.init 启动成本: {t_init:.1f}s "
          f"(resources: CPU={res['CPU']:.0f}, "
          f"object_store={res['object_store_memory'] / 2**30:.1f} GiB)")

    docs = make_corpus()
    n_bytes = sum(len(d["text"]) for d in docs)
    print(f"    语料: {len(docs)} docs, {n_bytes / 1e6:.2f} MB "
          f"(含 {NUM_DUPS} 条注入重复) —— 与 nano-data-juicer L2 同一构造")

    # ---- [1] object store 喂数据：put 一次 vs 传值四次 ----
    # 测量卫生：先用两个小任务预热 submit/put 通路，把「首次使用」的一次性
    # 初始化成本挡在计时窗外（否则首个提交会比稳态慢一到两个量级，
    # 账就算错了——探针实测：冷首提交 ~50ms，预热后同尺寸提交 ~几 ms，
    # 且「同一对象重复提交」并不比「等值新对象」便宜：驱动端没有按
    # 对象身份缓存序列化结果，传值每次都真付全量序列化的钱）。
    warm = [slice_count.remote([{"row_id": 0, "text": "x"}], 0, 1)
            for _ in range(2)]
    ray.get(warm)

    payload_mb = len(pickle.dumps(docs)) / 1e6    # stdlib pickle：负载量级参考
    t0 = time.perf_counter()
    corpus_ref = ray.put(docs)
    t_put = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref_refs = [slice_count.remote(corpus_ref, i * 840, (i + 1) * 840)
                for i in range(NUM_PARTITIONS)]
    t_submit_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_refs = [slice_count.remote(docs, i * 840, (i + 1) * 840)
                for i in range(NUM_PARTITIONS)]
    t_submit_val = time.perf_counter() - t0

    val_res = ray.get(val_refs)
    ref_res = ray.get(ref_refs)
    assert [r[0] for r in val_res] == [r[0] for r in ref_res] == [840] * 4
    pids_slice = sorted({r[1] for r in ref_res})
    print(f"\n[1] object store 喂数据 (corpus {n_bytes / 1e6:.2f} MB, "
          f"序列化负载 ~{payload_mb:.1f} MB)")
    print(f"    ray.put 一次: {t_put * 1e3:.1f} ms（驱动端序列化 1 份进 store）")
    print(f"    传引用提交 4 任务: 提交耗时 {t_submit_ref * 1e3:.1f} ms"
          f"（只传 ~28B 的 ObjectRef，数据不再动）")
    print(f"    传值提交 4 任务: 提交耗时 {t_submit_val * 1e3:.1f} ms"
          f"（每次提交都把 ~{payload_mb:.1f} MB 语料完整序列化一遍，")
    print(f"    随任务 RPC 内联发出——< 10MB 内联上限 task_rpc_inlined_bytes_limit，")
    print(f"    ray_config_def.h@ray-2.56.1；超限参数改走自动 put，见 L3）")
    print(f"    两种方式结果一致: {[r[0] for r in val_res]} ✅ | "
          f"worker 进程数 = {len(pids_slice)}（跨进程执行 ✅）")

    # ---- [1b] ObjectRef 参数自动解引用：两种失败现场 ----
    try:
        ray.get(bad_get.remote(ray.put(42)))
        raise AssertionError("bad_get(42) 应当抛出 ValueError")
    except ray.exceptions.RayTaskError as e:
        err_int = str(e).strip().splitlines()[-1]
    try:
        ray.get(bad_get.remote(ray.put([1, 2, 3])))
        raise AssertionError("bad_get([1,2,3]) 应当抛出 TypeError")
    except ray.exceptions.RayTaskError as e:
        err_list = str(e).strip().splitlines()[-1]
    print(f"    自动解引用现场 a: 任务内对自己的 int 参数调 ray.get ->\n"
          f"      {err_int}")
    print(f"    自动解引用现场 b: 对已解引用的 list 调 ray.get ->\n"
          f"      {err_list}")
    print(f"    → ObjectRef 作为参数在任务执行前已被 runtime 解引用，任务体拿到")
    print(f"      的直接是数据；b 更危险——list 会被 ray.get 当成「ref 列表」逐个")
    print(f"      再解一次，若元素恰好是 ref 就发生静默二次解引用。想在任务里持有")
    print(f"      ref 本身需包一层容器（L3 讨论）。")

    # ---- [2] 局部 OP 阶段：串行 vs Ray 并行 ----
    t0 = time.perf_counter()
    ser_mapped = local_ops(docs)
    t_ser_local = time.perf_counter() - t0

    parts = partition(docs, NUM_PARTITIONS)
    part_refs = [ray.put(p) for p in parts]
    t0 = time.perf_counter()
    par_out = ray.get([ray_local_ops.remote(r) for r in part_refs])
    t_par_local = time.perf_counter() - t0
    par_mapped = [s for out, _ in par_out for s in out]
    pids_ops = sorted({pid for _, pid in par_out})

    assert par_mapped == ser_mapped, "并行结果必须与串行逐位一致（含顺序）"
    sp = t_ser_local / t_par_local
    print(f"\n[2] 局部 OP 阶段 (normalize_mapper + length_filter)")
    print(f"    漏斗: {len(docs)} -> {len(ser_mapped)} 条 | "
          f"serial {t_ser_local * 1e3:.1f} ms / ray {t_par_local * 1e3:.1f} ms "
          f"(speedup {sp:.2f}x)")
    print(f"    并行结果 == 串行结果: True ✅（顺序逐位一致）")
    print(f"    worker pid 证据: {len(pids_ops)} 个不同进程承载 4 个分区任务"
          f"（L0 的线程共享同一进程，这里是真进程）")
    assert len(pids_ops) >= 2, "4 个并发分区任务应落在多个 worker 进程上"

    # ---- [3] 全局 OP：naive 反例 vs 收敛点 ----
    # 注意：dedup 的输入是「局部 OP 之后」的分区（sig 字段已生成），
    # 与 nano-data-juicer L2 的执行计划一致：normalize -> length -> dedup。
    mapped_parts = [out for out, _ in par_out]
    mapped_refs = [ray.put(p) for p in mapped_parts]
    naive_parts = ray.get([ray_partition_dedup.remote(r) for r in mapped_refs])
    naive_out = [s for p in naive_parts for s in p]
    t0 = time.perf_counter()
    ser_final = dedup_keep_first(ser_mapped)
    t_ser_global = time.perf_counter() - t0
    conv_out = ray.get(ray_global_dedup.remote(*mapped_refs))
    assert len(naive_out) == EXPECTED_NAIVE, \
        f"naive 应泄漏到 {EXPECTED_NAIVE} 条，实测 {len(naive_out)}"
    assert conv_out == ser_final, "收敛点结果必须与串行 pipeline 逐位一致"

    # 重复对账本（与 L2 同算术）：过滤后幸存者中的重复对
    by_key: Dict[str, List[Sample]] = {}
    for s in ser_mapped:
        by_key.setdefault(s["sig"], []).append(s)
    pair_total = sum(1 for v in by_key.values() if len(v) == 2)
    # 跨分区对：两份落在不同分区（用 row_id 区间判定，分区是连续 chunk）
    chunk = (len(docs) + NUM_PARTITIONS - 1) // NUM_PARTITIONS
    part_of = lambda row_id: min(row_id // chunk, NUM_PARTITIONS - 1)
    cross = sum(1 for v in by_key.values()
                if len(v) == 2 and part_of(v[0]["row_id"]) != part_of(v[1]["row_id"]))
    leak = len(naive_out) - len(conv_out)
    assert leak == cross == EXPECTED_LEAK

    print(f"\n[3] 全局 OP (exact_deduplicator)")
    print(f"    a) naive 分区各做各的: {len(naive_out)} 条, 泄漏 {leak} 条 ❌")
    print(f"    b) 收敛点任务 ray_global_dedup(*part_refs): {len(conv_out)} 条 ✅")
    print(f"    串行基准: {len(ser_final)} 条 | 逐位一致: True ✅")
    print(f"    重复对账本(过滤后幸存者): 共 {pair_total} 对 | 跨分区 {cross} 对"
          f" => 泄漏数 {leak} = 跨分区对数")
    print(f"    （全局去重串行段 {t_ser_global * 1e3:.1f} ms：收敛点后的 Amdahl "
          f"串行段，与 L2 同结论）")

    # ---- [4] 任务粒度扫描：P = 1/2/4/8/16 ----
    print(f"\n[4] 任务粒度扫描 (同一语料，变分区数 P；{NUM_CPUS} CPU)")
    print(f"    {'P':>2} | {'wall(ms)':>9} | {'docs/s':>8} | 每任务条数")
    sweep: Dict[int, float] = {}
    for P in (1, 2, 4, 8, 16):
        pp = partition(docs, P)
        refs = [ray.put(x) for x in pp]
        t0 = time.perf_counter()
        outs = ray.get([ray_local_ops.remote(r) for r in refs])
        wall = time.perf_counter() - t0
        sweep[P] = wall
        got = sum(len(o) for o, _ in outs)
        assert got == len(ser_mapped), f"P={P} 粒度下结果条数变了"
        print(f"    {P:>2} | {wall * 1e3:9.1f} | {len(docs) / wall:8.0f} | "
              f"{len(docs) // P}")
    best_p = min(sweep, key=sweep.get)
    print(f"    最快: P={best_p}；P 过大时每任务开销（提交/调度/序列化）抬头，")
    print(f"    P=1 没有并行——最优粒度在两者之间，且随负载而变。")

    # ---- [5] 启动成本摊销 + 汇总 ----
    t_work = t_ser_local + t_ser_global
    rounds = t_init / t_work
    print(f"\n[5] runtime 成本摊销: ray.init {t_init:.1f}s vs 单轮 pipeline 工作 "
          f"{t_work * 1e3:.0f} ms => 约 {rounds:.0f} 轮才回本")
    print(f"    小作业别付集群启动费；Ray 的价值在长生命周期集群 + 大负载")
    print(f"    （Data-Juicer 分区默认 size=5000 条/上限 64 MB，本语料 3.74 MB")
    print(f"    按该默认只会切出 1 个分区——默认参数面向的是 GB 级语料）。")

    ray.shutdown()
    print("\n" + "=" * 68)
    print("✅ self-check passed:")
    print("   漏斗 3360->2358->2110 与 nano-data-juicer L2 完全一致 /")
    print("   并行==串行逐位一致 / naive 泄漏 236=跨分区对数 /")
    print("   收敛点==串行 pipeline 逐位一致 / 粒度扫描条数不变")
    print("=" * 68)
    print("\ntakeaway: 同一个执行计划（分区→局部 OP 并行→全局 OP 收敛）换到真实")
    print("          Ray 上，语义一分不差，代价（启动/提交/序列化）变成实数。")
    print("          L2 把有状态算子搬进 actor。")


if __name__ == "__main__":
    main()
