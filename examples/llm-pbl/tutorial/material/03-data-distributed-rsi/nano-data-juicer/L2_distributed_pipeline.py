"""nano-data-juicer · L2 分布式 pipeline：分区 + 并行局部 OP + 全局 OP 收敛点
==================================================================================

K+1 目标（相对 L1）：
    L0/L1 的 pipeline 是单进程串行的：op(list) -> list，一次处理全量数据。
    数据到 TB 级时这条路走不通，必须把 pipeline 分布式化。L2 回答三个问题：

      1. 怎么切？   把数据集切成 P 个 partition，分给 worker 并行处理。
      2. 哪些 OP 能并行？ Mapper / Filter 是「局部 OP」——逐样本独立，
         分区各做各的，结果按分区顺序拼回来就等于串行结果（顺序确定性）。
         Deduplicator 是「全局 OP」——重复对可能横跨任意两个分区，
         分区内各做各的去重是**错的**，必须在执行计划里插入收敛点。
      3. 坏了怎么办？ 分区级容错：某个 partition 的 worker 挂了，
         只重算那一个 partition（lineage 重算），其余结果复用。

    与真实系统的对应（详见 tutorial_L2.md §6，行号为 main 分支 2026-08-05 快照）：
      - Data-Juicer DefaultExecutor 的并行 = HF datasets map(num_proc=np)；
      - Data-Juicer PartitionedRayExecutor：split() 切分 → 逐 partition 处理
        → union 合并；遇到 Deduplicator 等全局 OP 时触发 convergence point，
        先把分区收敛合并再跑全局 OP；确定性由 preserve_order=True 保证。

    ⚠️ 显式声明（可运行性契约）：
      - 为保持零额外依赖，本文件用标准库 multiprocessing 的**真实
        worker 进程**实现同一套执行语义（分区/并行/收敛/重算），是分布式
        执行机制的本质模拟；真实多机 Ray 执行见 [TODO: verify on real system]。
      - 语料为 seed 固定的合成数据（非真实语料）。L2 的主题是执行语义
        （切分/并行/收敛/容错），需要足够样本量才能测出并行加速，
        故不用 L1 的 10 条真实样本；数据内容不影响执行语义的正确性验证。

运行：
    python L2_distributed_pipeline.py

依赖：纯标准库（multiprocessing / hashlib / json / re / random / time）。
      无需 GPU、无需 Ray、无需网络。
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# 常量（全部固定 seed / 固定规模，保证确定性）
# ---------------------------------------------------------------------------

SEED = 42
NUM_DOCS = 3000          # 语料规模：够大才能测出并行加速
NUM_DUPS = 360           # 注入的重复样本数（~12%）
NUM_PARTITIONS = 4       # 分区数 P（对应 Data-Juicer 默认 num_of_partitions=4）
NUM_WORKERS = 4          # worker 进程数
MIN_NORM_LEN = 900       # length_filter 阈值（按规范化后字符数）

Sample = Dict[str, Any]


# ---------------------------------------------------------------------------
# [0] 合成语料：确定性生成 + 注入「跨分区重复」
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
    """生成确定性语料。

    每条样本：{"row_id": int, "text": str}。
    重复注入方式：复制某条样本的 text，做大小写/空白扰动后放到更后的位置。
    扰动保证「原文不相等、规范化后相等」——dedup 必须发生在 normalize 之后
    才能抓到它们，这正是 L0「OP 顺序是语义的一部分」在分布式下的回声。

    不变量（[4] 重复对账本算术的基础）：每个重复 key 恰好出现 2 次，
    且 copy 的 row_id 恒大于 source。为此 source 只从前半区取、copy 位置
    只在后半区，且 copy 列表独立构造——绝不在变动中的列表上二次取样，
    否则会复制出「copy 的 copy」，破坏「每 key 恰好两份」的账本。
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

    # source 取自前半区、copy 位置在后半区 => copy 必在 source 之后
    src_ids = rng.sample(range(num_docs // 2), num_dups)
    copies: List[str] = []
    for src in src_ids:
        t = base[src]                       # 只从 base 取，绝不取 copy
        # 扰动：随机大写 + 双空格 + 首尾空白（规范化后与原文相同）
        chars = list(t)
        for _ in range(max(1, len(t) // 40)):
            j = rng.randrange(len(chars))
            chars[j] = chars[j].upper()
        perturbed = re.sub(r"( )", "  ", "".join(chars), count=len(t) // 60)
        copies.append(" " + perturbed + " ")

    # 归位：total 个槽位，copy 占随机后半槽位，其余按序放 base
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
# [1] OP 抽象：kind 决定执行语义（局部 vs 全局）
# ---------------------------------------------------------------------------

@dataclass
class OpSpec:
    """OP = 名字 + 类别 + 执行函数。

    kind:
      "mapper"       局部 OP：逐样本变换（可分区并行）
      "filter"       局部 OP：逐样本判定（可分区并行）
      "deduplicator" 全局 OP：需要看到全量数据（需要收敛点）
    """
    name: str
    kind: str
    fn_map: Optional[Callable[[Sample], Sample]] = None
    fn_keep: Optional[Callable[[Sample], bool]] = None
    dedup_key: Optional[Callable[[Sample], str]] = None


def is_global_operation(op: OpSpec) -> bool:
    """判定是否全局 OP。

    与 Data-Juicer 的 is_global_operation 同构（三级判定优先级：
    显式标志 → 基类 → 名字模式；dag_execution_strategies.py:L441-471）。
    这里只有 kind 一个信号，但语义相同：全局 OP 需要收敛点。
    """
    return op.kind == "deduplicator" or "dedup" in op.name.lower()


# ---------------------------------------------------------------------------
# [2] 具体 OP 实现
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _heavy_signature(text: str) -> str:
    """真实 CPU 负载：字符二元组直方图 + 双层哈希。

    这不是假 sleep——是 O(len) 的真实计算，让并行加速可测量。
    对应真实 pipeline 里那些 CPU-bound 的清洗 OP（语言检测、
    perplexity 打分、simhash 指纹等）。
    """
    hist = [0] * 256
    for i in range(len(text) - 1):
        hist[(ord(text[i]) + ord(text[i + 1])) % 256] += 1
    payload = text.encode("utf-8")
    return hashlib.md5(hashlib.sha1(payload).digest()
                       + bytes(hist)).hexdigest()


def normalize_map(sample: Sample) -> Sample:
    """Mapper：规范化文本 + 多特征抽取（CPU-bound）。

    特征抽取是真实清洗 mapper 的常态（一条 OP 往往算一组特征：
    长度、词表、重复度……），这里全部是 O(len) 的真实计算。
    """
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
    """Filter：规范化后长度 >= 阈值才保留。"""
    return sample.get("norm_len", 0) >= MIN_NORM_LEN


def dedup_key_by_sig(sample: Sample) -> str:
    """Deduplicator 的 key：规范化文本的指纹（对应 compute_hash）。"""
    return sample["sig"]


def build_pipeline() -> List[OpSpec]:
    """配置驱动的 pipeline（与 L0/L1 同一思想：OP 列表即配置）。

    顺序有意设计：normalize（mapper）→ length（filter）→ dedup（全局）。
    dedup 依赖 normalize 的产物，且它是唯一的全局 OP——执行计划必须
    在它前面插入收敛点。
    """
    return [
        OpSpec("normalize_mapper", "mapper", fn_map=normalize_map),
        OpSpec("length_filter", "filter", fn_keep=length_keep),
        OpSpec("exact_deduplicator", "deduplicator",
               dedup_key=dedup_key_by_sig),
    ]


# ---------------------------------------------------------------------------
# [3] 分区：chunk 切分 + 轮转还原
# ---------------------------------------------------------------------------

def partition(docs: List[Sample], n: int) -> List[List[Sample]]:
    """连续 chunk 切分（对应 Ray Data 的 split(n)：保持分区内顺序）。"""
    size = (len(docs) + n - 1) // n
    return [docs[i * size:(i + 1) * size] for i in range(n)
            if i * size < len(docs)]


# ---------------------------------------------------------------------------
# [4] worker 侧：分区粒度的局部 OP（顶层函数，可被 pickle 传进进程池）
# ---------------------------------------------------------------------------

def _apply_local_op(args) -> List[Sample]:
    """worker 进程里对单个 partition 应用一个局部 OP。"""
    part, op = args
    if op.kind == "mapper":
        return [op.fn_map(s) for s in part]
    if op.kind == "filter":
        return [s for s in part if op.fn_keep(s)]
    raise ValueError(f"not a local op: {op.name}")


def _bucket_dedup(args) -> List[Sample]:
    """worker 进程里对单个 hash bucket 去重（shuffle 策略用）。

    同一 dedup key 必落在同一 bucket，所以 bucket 内去重即全局正确；
    保留 row_id 最小者 = 保留全局顺序下的首次出现。
    （顶层函数而非闭包：spawn 模式下闭包无法 pickle 进 worker 进程。）
    """
    bucket, key = args
    best: Dict[str, Sample] = {}
    for s in bucket:
        k = key(s)
        if k not in best or s["row_id"] < best[k]["row_id"]:
            best[k] = s
    return list(best.values())


# ---------------------------------------------------------------------------
# [5] 两种执行器：串行（L1 语义）vs 分布式（L2 新增）
# ---------------------------------------------------------------------------

def serial_run(docs: List[Sample], ops: List[OpSpec]) -> List[Sample]:
    """串行执行：L0/L1 的语义，全量数据一次过。"""
    cur = docs
    for op in ops:
        if op.kind == "mapper":
            cur = [op.fn_map(s) for s in cur]
        elif op.kind == "filter":
            cur = [s for s in cur if op.fn_keep(s)]
        elif op.kind == "deduplicator":
            seen, out = set(), []
            for s in cur:                       # 全局扫描，保留首次出现
                k = op.dedup_key(s)
                if k not in seen:
                    seen.add(k)
                    out.append(s)
            cur = out
        else:
            raise ValueError(f"unknown kind: {op.kind}")
    return cur


def _dedup_keep_first(docs: List[Sample], key: Callable[[Sample], str]
                      ) -> List[Sample]:
    """全局去重：保留全局顺序下的首次出现（row_id 最小者）。"""
    seen, out = set(), []
    for s in docs:
        k = key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def distributed_run(docs: List[Sample], ops: List[OpSpec],
                    num_partitions: int = NUM_PARTITIONS,
                    num_workers: int = NUM_WORKERS,
                    dedup_strategy: str = "convergence",
                    pool: Optional[mp.pool.Pool] = None,
                    ) -> Dict[str, Any]:
    """分布式执行：分区 → 局部 OP 并行 → 全局 OP 收敛。

    dedup_strategy:
      "convergence"  Data-Juicer partitioned executor 的做法：
                     全局 OP 前把所有分区 union 成一个整体再处理
                     （ray_executor_partitioned.py:L872-922, main 分支
                     2026-08-05 快照）。
                     正确，但全局阶段是串行的（Amdahl 的串行段）。
      "shuffle"      Spark/MapReduce 式：按 hash(key) % P 重分区，
                     相同 key 必落同一分区 → 分区内去重即全局正确，
                     全程并行，代价是一次全量 shuffle + 最后按
                     row_id 排序还原全局顺序。
    """
    parts = partition(docs, num_partitions)
    own_pool = False
    if pool is None:
        pool = mp.Pool(num_workers)
        own_pool = True

    plan = []                       # 执行计划日志（教程里逐行解读）
    try:
        cur_parts = parts
        for op in ops:
            if not is_global_operation(op):
                # ---- 局部 OP：分区并行（map 语义，顺序由分区序保证）----
                cur_parts = pool.map(_apply_local_op,
                                     [(p, op) for p in cur_parts])
                plan.append(f"{op.name}: local, parallel on "
                            f"{len(cur_parts)} partitions")
            else:
                # ---- 全局 OP：收敛点 ----
                if dedup_strategy == "convergence":
                    merged = [s for p in cur_parts for s in p]  # union
                    plan.append(f"{op.name}: GLOBAL -> convergence "
                                f"(union {len(cur_parts)} partitions, "
                                f"serial dedup)")
                    cur_parts = [_dedup_keep_first(merged, op.dedup_key)]
                elif dedup_strategy == "shuffle":
                    key = op.dedup_key
                    buckets: List[List[Sample]] = [[]
                                                   for _ in range(num_partitions)]
                    for p in cur_parts:
                        for s in p:
                            h = int(hashlib.md5(
                                key(s).encode()).hexdigest(), 16)
                            buckets[h % num_partitions].append(s)
                    plan.append(f"{op.name}: GLOBAL -> shuffle "
                                f"(repartition by hash(key)%{num_partitions}, "
                                f"parallel dedup)")
                    # 分区内去重：同 key 都在本分区，保留 row_id 最小者
                    deduped = pool.map(_bucket_dedup,
                                       [(b, key) for b in buckets])
                    merged = [s for p in deduped for s in p]
                    merged.sort(key=lambda s: s["row_id"])  # 还原全局顺序
                    cur_parts = [merged]
                else:
                    raise ValueError(f"unknown strategy: {dedup_strategy}")
        result = [s for p in cur_parts for s in p]
        return {"result": result, "plan": plan}
    finally:
        if own_pool:
            pool.close()
            pool.join()


def naive_partition_dedup(docs: List[Sample], ops: List[OpSpec],
                          num_partitions: int = NUM_PARTITIONS
                          ) -> List[Sample]:
    """反例：不识别全局 OP，把 dedup 当局部 OP 分区各做各的。"""
    parts = partition(docs, num_partitions)
    for op in ops:
        if op.kind == "deduplicator":
            parts = [_dedup_keep_first(p, op.dedup_key) for p in parts]
        else:
            parts = [_apply_local_op((p, op)) for p in parts]
    out = [s for p in parts for s in p]
    return out


# ---------------------------------------------------------------------------
# [6] 分区级容错：worker 挂了只重算那一个 partition（lineage 重算）
# ---------------------------------------------------------------------------

def run_with_partition_failure(docs: List[Sample], op: OpSpec,
                               fail_pid: int,
                               num_partitions: int = NUM_PARTITIONS
                               ) -> Dict[str, Any]:
    """模拟某个 partition 的 worker 首次执行崩溃，executor 只重算它。

    显式声明：崩溃是注入的（第一次处理 fail_pid 时抛异常），
    用于演示分区级失败隔离 + lineage 重算语义——对应 Ray 的
    lineage 重建与 Data-Juicer partitioned executor 的
    per-partition checkpoint（ray_executor_partitioned.py:L923-1058,
    main 分支 2026-08-05 快照）。
    这里在进程内模拟以保确定性，重算语义与真实系统一致。
    """
    parts = partition(docs, num_partitions)
    attempts = {i: 0 for i in range(len(parts))}
    cache: Dict[int, List[Sample]] = {}
    failed_once = {fail_pid: False}

    def compute_one(pid: int) -> List[Sample]:
        attempts[pid] += 1
        if pid == fail_pid and not failed_once[fail_pid]:
            failed_once[fail_pid] = True
            raise RuntimeError(f"[injected] worker for partition {pid} crashed")
        return _apply_local_op((parts[pid], op))

    # 第一遍：并行提交（模拟），fail_pid 崩溃，其余成功进缓存
    for pid in range(len(parts)):
        try:
            cache[pid] = compute_one(pid)
        except RuntimeError as e:
            print(f"    [fault] {e}")

    # 重算：只重做缺失的分区（缓存命中的不重算）
    recomputed = []
    for pid in range(len(parts)):
        if pid not in cache:
            cache[pid] = compute_one(pid)
            recomputed.append(pid)

    result = [s for pid in range(len(parts)) for s in cache[pid]]
    return {"result": result, "attempts": attempts,
            "recomputed": recomputed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _fmt_ms(sec: float) -> str:
    return f"{sec * 1000:8.1f} ms"


def main():
    print("=" * 68)
    print("nano-data-juicer L2 — distributed pipeline")
    print("=" * 68)
    print(f"config: docs={NUM_DOCS} dups={NUM_DUPS} partitions={NUM_PARTITIONS} "
          f"workers={NUM_WORKERS} seed={SEED}")
    print("声明: 合成语料(固定seed) + multiprocessing 真实进程并行；")
    print("      分布式语义的本质模拟，真实 Ray 集群见 [TODO: verify on real system]")

    # ---- [1] 造数据 ----
    t0 = time.time()
    docs = make_corpus()
    t_gen = time.time() - t0
    n_bytes = sum(len(d["text"]) for d in docs)
    print(f"\n[1] 语料: {len(docs)} docs, {n_bytes / 1e6:.2f} MB "
          f"(含 {NUM_DUPS} 条注入重复), 生成耗时 {_fmt_ms(t_gen)}")

    # ---- [2] 分区 + round-trip ----
    parts = partition(docs, NUM_PARTITIONS)
    roundtrip = [s for p in parts for s in p]
    assert roundtrip == docs, "partition round-trip must be lossless"
    sizes = [len(p) for p in parts]
    print(f"[2] 分区: {NUM_PARTITIONS} partitions, sizes={sizes}, "
          f"round-trip 无损 ✅")

    ops = build_pipeline()

    # worker 池全程复用：进程启动开销只付一次（对应真实系统里
    # 常驻 worker / Ray actor 的做法；每个 stage 新建池会反复付 spawn 成本）
    pool = mp.Pool(NUM_WORKERS)

    # ---- [3] 局部 OP：串行 vs 并行，结果必须逐字节一致 ----
    print(f"\n[3] 局部 OP 阶段 (normalize_mapper + length_filter)")
    local_ops = [op for op in ops if not is_global_operation(op)]

    t0 = time.time()
    serial_local = serial_run(docs, local_ops)
    t_serial_local = time.time() - t0

    t0 = time.time()
    dist_local = distributed_run(docs, local_ops, pool=pool)["result"]
    t_dist_local = time.time() - t0

    identical = dist_local == serial_local
    speedup_local = t_serial_local / t_dist_local if t_dist_local else float("inf")
    print(f"    漏斗: {len(docs)} -> {len(serial_local)} 条 "
          f"(length_filter 过滤 {len(docs) - len(serial_local)} 条短文档)")
    print(f"    serial  : {_fmt_ms(t_serial_local)}")
    print(f"    parallel: {_fmt_ms(t_dist_local)}  "
          f"(speedup {speedup_local:.2f}x)")
    print(f"    并行结果 == 串行结果: {identical} ✅（顺序逐位一致）")
    assert identical, "parallel local ops must equal serial exactly"

    # ---- [4] 全局 OP：反例 + 两种收敛策略 ----
    print(f"\n[4] 全局 OP 阶段 (exact_deduplicator)")
    serial_final = serial_run(docs, ops)

    naive = naive_partition_dedup(docs, ops)
    leaked = len(naive) - len(serial_final)
    print(f"    a) 反例: 把 dedup 当局部 OP 分区各做各的")
    print(f"       结果条数 = {len(naive)}, 泄漏跨分区重复 {leaked} 条 ❌")
    assert leaked > 0, "cross-partition duplicates must leak in naive mode"

    conv = distributed_run(docs, ops, dedup_strategy="convergence", pool=pool)
    shuf = distributed_run(docs, ops, dedup_strategy="shuffle", pool=pool)
    print(f"    b) convergence 策略 (union 后全局去重): {len(conv['result'])} 条")
    print(f"    c) shuffle 策略 (按 hash(key) 重分区): {len(shuf['result'])} 条")
    print(f"    串行基准: {len(serial_final)} 条")
    assert conv["result"] == serial_final, "convergence must equal serial"
    assert shuf["result"] == serial_final, "shuffle must equal serial"
    print(f"    两策略与串行基准逐位一致 ✅")

    # 重复对账本：同分区 vs 跨分区（解释 naive 到底漏了什么）
    pid_of = {}
    for pid, p in enumerate(partition(docs, NUM_PARTITIONS)):
        for s in p:
            pid_of[s["row_id"]] = pid
    seen_first: Dict[str, int] = {}
    same_part, cross_part = 0, 0
    for s in serial_local:               # 已按 row_id 升序（filter 不改顺序）
        k = s["sig"]
        if k in seen_first:
            if pid_of[s["row_id"]] == pid_of[seen_first[k]]:
                same_part += 1
            else:
                cross_part += 1
        else:
            seen_first[k] = s["row_id"]
    print(f"    重复对账本(过滤后幸存者): 同分区内 {same_part} 对 | "
          f"跨分区 {cross_part} 对")
    print(f"    naive 只抓得到同分区对，跨分区对每对泄漏 1 条 "
          f"=> 泄漏数 {leaked} vs 跨分区对数 {cross_part}")
    assert leaked == cross_part, "leak count must equal cross-partition pairs"

    # ---- [5] 端到端：串行 vs 分布式（两种策略）计时 ----
    print(f"\n[5] 端到端 pipeline 计时")
    t0 = time.time()
    serial_run(docs, ops)
    t_serial = time.time() - t0

    t0 = time.time()
    conv_run = distributed_run(docs, ops, dedup_strategy="convergence",
                               pool=pool)
    t_conv = time.time() - t0

    t0 = time.time()
    distributed_run(docs, ops, dedup_strategy="shuffle", pool=pool)
    t_shuf = time.time() - t0

    print(f"    serial              : {_fmt_ms(t_serial)}")
    print(f"    distributed(conv)   : {_fmt_ms(t_conv)}  "
          f"speedup {t_serial / t_conv:.2f}x")
    print(f"    distributed(shuffle): {_fmt_ms(t_shuf)}  "
          f"speedup {t_serial / t_shuf:.2f}x")
    print(f"    执行计划 (convergence):")
    for line in conv_run["plan"]:
        print(f"      - {line}")
    print(f"    Amdahl: convergence 版的全局去重段是串行的，")
    print(f"    并行加速只作用于局部 OP 段 => speedup 有上限。")

    pool.close()
    pool.join()

    # ---- [6] 分区级容错 ----
    print(f"\n[6] 分区级容错 (注入 partition 2 首次执行崩溃)")
    fault = run_with_partition_failure(docs, local_ops[0], fail_pid=2)
    ref_all = [normalize_map(s) for p in parts for s in p]
    assert fault["result"] == ref_all, "recomputed result must equal clean run"
    print(f"    attempts per partition: {fault['attempts']}")
    print(f"    recomputed partitions : {fault['recomputed']}")
    print(f"    其余 {NUM_PARTITIONS - len(fault['recomputed'])} 个分区结果"
          f"直接复用，未重算 ✅")
    assert fault["recomputed"] == [2]
    assert fault["attempts"][0] == 1 and fault["attempts"][1] == 1
    assert fault["attempts"][2] == 2 and fault["attempts"][3] == 1

    # ---- self-check 汇总 ----
    print("\n" + "=" * 68)
    print("✅ self-check passed:")
    print("   分区无损 round-trip / 局部 OP 并行==串行逐位一致 /")
    print("   naive 分区去重必泄漏 / convergence==shuffle==串行 /")
    print("   重复对账本吻合 / 容错只重算崩溃分区")
    print("=" * 68)


if __name__ == "__main__":
    main()
