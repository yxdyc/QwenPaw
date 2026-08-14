"""nano-data-juicer · L3 — OP 接口与配置 schema：一个 Filter 的完整行为
=========================================================================

K+1 目标（相对 L2）：
    L2 把 OP 当「执行单元」：OpSpec = 名字 + 类别 + 函数指针，执行器按类别
    决定并行还是收敛。但「OP 是什么」仍是裸 dataclass：没有类层级、没有接口
    约定（stats 怎么算、判定怎么读、存到哪）、没有配置到实例的构造路径。
    L3 对齐 Data-Juicer 真实 OP 接口，只加一层：

        一个 filter 在成熟框架里从配置文件走到判定输出，每一步长什么样。

    五个机制（与真实系统的逐条对应见 tutorial_L3.md §6，行号为 main 分支
    2026-08-07 双通道核验快照）：
      1. Registry + 配置驱动：OPERATORS 注册表绑定 名字→类；配置
         [{op_name: args}] 经 load_ops 装配成实例。OP 的名字是注册表
         赋的（registry.py 的 module_cls._name = module_name），不是类自报的。
      2. Filter 两段式：compute_stats（把统计写进样本的 Fields.stats
         命名空间列）→ process（只读 stats 判定 keep/drop）；reduce=False
         时只算统计不删样本；stats 按 key 复用（已有则跳过重算）。
      3. 区间语义：get_keep_boolean —— min/max 阈值 + 开闭区间开关 +
         reversed_range（先把两端开闭取反，再整体取非）。
      4. NON_STATS_FILTERS：不产 stats 的 filter（如按文件后缀过滤）双注册，
         OP.run 不给它们注入 stats 列。
      5. 接口守卫：__init_subclass__ 禁止子类直接重写 compute_stats /
         process，强制实现 *_single / *_batched——接口契约在类定义期生效。

    跨级别契约（机器断言，与 nano-ray L1/L2 同款做法）：
        导入 L2 的 make_corpus / normalize_map / serial_run / build_pipeline。
        L3 的 TextLengthFilter(text_key='norm_text', min_len=900) 的幸存者
        row_id 序列 == L2 串行漏斗（normalize_mapper + length_filter）的
        幸存者 row_id 序列——3360 -> 2358，逐位一致，不止条数相等。

    ⚠️ 显式声明（可运行性契约）：
      - 复现的是**接口语义**，不是执行性能：nano 的 Filter.run 是单进程
        list 循环；真实 Data-Juicer 是 HF datasets map/filter + num_proc
        多进程（base_op.py 的 runtime_np()）。执行形态差异见教程 §7。
      - 真实 dataset 是 HF Arrow 表；nano 用 list[dict] + dict-of-lists
        列式批次复现同一 _batched_op 约定（含 list<->dict 双向转换）。
      - 语料复用 L2 的固定 seed 合成语料（import make_corpus），保证
        跨级别可比；L3 主题是 OP 语义，不做并行计时（L2 已覆盖）。

运行：
    python L3_filter_interface.py

依赖：纯标准库（sys / typing）+ 同目录 import L2_distributed_pipeline
      （L2 亦为纯标准库）。无 GPU、无网络、无第三方包。
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List

sys.dont_write_bytecode = True          # import L2 不落 __pycache__（全树零 pyc 约定，nano-ray L2 同款）

# 跨级别契约：直接复用 L2 的语料与串行基准（L2 有 __main__ 守卫，导入安全）
from L2_distributed_pipeline import (
    MIN_NORM_LEN,
    build_pipeline,
    is_global_operation,
    make_corpus,
    normalize_map,
    serial_run,
)

Sample = Dict[str, Any]


# ---------------------------------------------------------------------------
# [0] 常量：命名空间前缀与 stats 键（constant.py：DEFAULT_PREFIX / Fields / StatsKeys）
# ---------------------------------------------------------------------------

DEFAULT_PREFIX = "__dj__"


class Fields:
    """样本上的框架保留列。带命名空间前缀，避免与用户字段（text/images…）撞名。"""
    stats = DEFAULT_PREFIX + "stats__"
    suffix = DEFAULT_PREFIX + "suffix__"


class StatsKeys:
    """stats 列内部的统计量键名。全局约定：一个键只有一种语义。"""
    text_len = "text_len"


# ---------------------------------------------------------------------------
# [1] Registry：名字 -> 类 的注册表（registry.py）
# ---------------------------------------------------------------------------

class Registry:
    """与 data_juicer/utils/registry.py 同构的最小注册表。

    关键行为逐条复现：
      - 重复注册抛 KeyError（除非 force=True）；
      - 注册成功时 **module_cls._name = module_name**——OP 的名字是注册表
        赋的，不是类自报的。一个类注册到两个名字下就会有两个「身份」。
    """

    def __init__(self, name: str):
        self._name = name
        self._modules: Dict[str, type] = {}

    @property
    def modules(self) -> Dict[str, type]:
        return self._modules

    def _register_module(self, module_name: str, module_cls: type,
                         force: bool = False) -> None:
        if module_name in self._modules and not force:
            raise KeyError(
                f"{module_name} is already registered in {self._name}")
        self._modules[module_name] = module_cls
        module_cls._name = module_name          # 名字来自注册表

    def register_module(self, module_name: str = None, module_cls: type = None,
                        force: bool = False):
        """既能当装饰器 @REG.register_module("name")，也能直接调用。"""
        if module_cls is None:
            def _wrapper(cls):
                self._register_module(module_name, cls, force=force)
                return cls
            return _wrapper
        self._register_module(module_name, module_cls, force=force)
        return module_cls


OPERATORS = Registry("Operators")
NON_STATS_FILTERS = Registry("Non-stats Filters")
DEFAULT_BATCH_SIZE = 1000      # 真实值（base_op.py），nano 单批全量，仅展示常量

# nano 观测注入：统计量真实计算次数（真实 DJ 无此计数器；复用与否靠它可视）
STATS_COMPUTE_COUNT = {"text_len": 0}


# ---------------------------------------------------------------------------
# [2] 批次形态转换：list[dict] <-> dict-of-lists（base_op.py 的两个 convert_*）
# ---------------------------------------------------------------------------

def convert_list_dict_to_dict_list(samples: List[Sample]) -> Dict[str, list]:
    """「行式」转「列式」：_batched_op 收到的批次是列式的。"""
    keys = samples[0].keys()
    return {key: [s[key] for s in samples] for key in keys}


def convert_dict_list_to_list_dict(samples: Dict[str, list]) -> List[Sample]:
    """「列式」转回「行式」。"""
    keys = list(samples.keys())
    return [{key: samples[key][i] for key in samples}
            for i in range(len(samples[keys[0]]))]


# ---------------------------------------------------------------------------
# [3] OP 基类：run() 负责注入 stats 列（只对「产 stats 的 Filter」）
# ---------------------------------------------------------------------------

class OP:
    """OP 基类（对应 base_op.py 的 class OP）。

    nano 只保留与 Filter 主线相关的职责：
      - text_key 等字段键从 kwargs 取（默认 "text"）；
      - run() 里给「产 stats 的 Filter」注入 Fields.stats 列——判定条件与
        真实 OP.run 逐字同构：isinstance(self, Filter)
        and self._name not in NON_STATS_FILTERS.modules。
    省略项（真实 OP.run 还有，与 L3 主题无关，见教程 §7 差异表）：
      NestedDataset 包装、TAGGING_OPS 的 meta 列、index_key 索引。

    ⚠️ 注意 __init__ 全程 kwargs.get(...)：未知参数**静默吸收**——这是
    真实行为的忠实复现，也是 [5] 反例的主角。
    """

    _name = ""                 # 由 Registry 赋值
    _batched_op = False

    def __init__(self, *args, **kwargs):
        self.text_key = kwargs.get("text_key", "text")
        self.batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)

    def is_batched_op(self) -> bool:
        return self._batched_op

    def run(self, dataset: List[Sample]) -> List[Sample]:
        if (isinstance(self, Filter)
                and self._name not in NON_STATS_FILTERS.modules):
            for s in dataset:
                s.setdefault(Fields.stats, {})   # 已存在则复用（对应 features 检查）
        return dataset


# ---------------------------------------------------------------------------
# [4] Filter 基类：区间语义 + 两段式 run + 接口守卫
# ---------------------------------------------------------------------------

class Filter(OP):
    """Filter 基类（对应 base_op.py 的 class Filter）。

    三段复现：
      __init__         区间开关 + 运行时方法绑定（batched/single 分派）
      get_keep_boolean 区间判定（逐字同构）
      run              compute_stats → (reduce) process 两段式
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # filter strategy：开闭区间 + 反转（base_op.py Filter.__init__）
        self.min_closed_interval = kwargs.get("min_closed_interval", True)
        self.max_closed_interval = kwargs.get("max_closed_interval", True)
        self.reversed_range = kwargs.get("reversed_range", False)
        if self.reversed_range:                 # 反转前先取反两端开闭
            self.min_closed_interval = not self.min_closed_interval
            self.max_closed_interval = not self.max_closed_interval

        # 运行时绑定：真实 DJ 在这里套异常捕获装饰器（catch_map_*_exception），
        # nano 只做 batched/single 分派，异常捕获省略（教程 §7）。
        if self.is_batched_op():
            self.compute_stats = self.compute_stats_batched
            self.process = self.process_batched
        else:
            self.compute_stats = self.compute_stats_single
            self.process = self.process_single

    # 接口守卫：子类不得直接重写 compute_stats / process（__init_subclass__
    # 在类定义期触发——错误在 import/定义时炸，不是跑数据时炸）
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        not_allowed_list = ["compute_stats", "process"]
        for method_name in not_allowed_list:
            if method_name in cls.__dict__:
                raise TypeError(
                    f"Method {method_name} cannot be overridden by subclass "
                    f"{cls.__name__}. Please implement {method_name}_single "
                    f"or {method_name}_batched.")

    def get_keep_boolean(self, val, min_val=None, max_val=None) -> bool:
        """区间判定——与 base_op.py Filter.get_keep_boolean 逐字同构。"""
        res_bool = True
        if min_val is not None:
            res_bool = res_bool and (
                val >= min_val if self.min_closed_interval else val > min_val)
        if max_val is not None:
            res_bool = res_bool and (
                val <= max_val if self.max_closed_interval else val < max_val)
        if self.reversed_range:
            res_bool = not res_bool
        return res_bool

    def run(self, dataset: List[Sample], reduce: bool = True) -> List[Sample]:
        """两段式：compute_stats（全量算统计）→ reduce 时 process（判定删留）。

        真实签名 run(dataset, *, exporter=None, tracer=None, reduce=True)；
        nano 省略 exporter（stats 落盘导出）与 tracer（样本级追踪）。
        """
        dataset = super().run(dataset)          # 注入 stats 列（条件见 OP.run）
        if self.is_batched_op():
            batch = convert_list_dict_to_dict_list(dataset)
            batch = self.compute_stats(batch)   # 列式批，原地写 stats
            dataset = convert_dict_list_to_list_dict(batch)
        else:
            dataset = [self.compute_stats(s) for s in dataset]
        if reduce:
            if self.is_batched_op():
                batch = convert_list_dict_to_dict_list(dataset)
                keep = list(self.process(batch))
                dataset = [s for s, k in zip(dataset, keep) if k]
            else:
                dataset = [s for s in dataset if self.process(s)]
        return dataset


# ---------------------------------------------------------------------------
# [5] 具体 Filter：text_length_filter（filter/text_length_filter.py 逐行复现）
# ---------------------------------------------------------------------------

@OPERATORS.register_module("text_length_filter")
class TextLengthFilter(Filter):
    """按文本长度区间过滤。真实实现的每个行为点都保留：

    - _batched_op = True：列式批处理；
    - min_len=10 / max_len=sys.maxsize 的默认值；
    - compute_stats_batched 的 stats 复用：`if StatsKeys.text_len in stat:
      continue`——已有就跳过，这是跨 OP 复用统计量的唯一入口；
    - process_batched 只读 stats、返回 get_keep_boolean 的 map 对象。
    """

    _batched_op = True

    def __init__(self, min_len: int = 10, max_len: int = sys.maxsize,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_len = min_len
        self.max_len = max_len

    def compute_stats_batched(self, samples: Dict[str, list]):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        for i, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.text_len in stat:
                continue
            else:
                samples_stats[i][StatsKeys.text_len] = len(samples_list[i])
                STATS_COMPUTE_COUNT["text_len"] += 1   # nano 观测注入
        return samples

    def process_batched(self, samples: Dict[str, list]):
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(
                stat[StatsKeys.text_len], self.min_len, self.max_len),
            samples[Fields.stats],
        )


# ---------------------------------------------------------------------------
# [6] NON_STATS_FILTERS 例子：suffix_filter（不产 stats，双注册）
# ---------------------------------------------------------------------------

_OP_NAME = "suffix_filter"


@NON_STATS_FILTERS.register_module(_OP_NAME)
@OPERATORS.register_module(_OP_NAME)
class SuffixFilter(Filter):
    """按样本来源后缀过滤——不产生任何 stats。

    真实行为复现点：
      - 双注册：同时在 OPERATORS 与 NON_STATS_FILTERS（装饰器顺序同真实源码）；
      - compute_stats_single 原样返回样本（走流程但不产统计）；
      - process_single 直接判 Fields.suffix，且**自己**处理 reversed_range
        （不走 get_keep_boolean，与真实源码一致）。
    """

    def __init__(self, suffixes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if suffixes is None:
            self.suffixes = []
        elif isinstance(suffixes, str):
            self.suffixes = [suffixes]
        else:
            self.suffixes = list(suffixes)

    def compute_stats_single(self, sample: Sample) -> Sample:
        return sample

    def process_single(self, sample: Sample) -> bool:
        if self.suffixes:
            res_bool = sample[Fields.suffix] in self.suffixes
            if self.reversed_range:
                res_bool = not res_bool
            return res_bool
        return True


# ---------------------------------------------------------------------------
# [7] 配置驱动：load_ops（ops/load.py——DefaultExecutor 的真实入口）
# ---------------------------------------------------------------------------

def load_ops(process_list: List[Dict[str, dict]]) -> List[OP]:
    """与 data_juicer/ops/load.py 同构：[{op_name: args}] -> [op 实例]。

    两个忠实复现的细节：
      - OPERATORS.modules[op_name](**args)：未注册的 op 名在这里 KeyError；
      - op._op_cfg = op_cfg：把原始配置存回实例（可追溯「我是哪行配置生的」）。
    """
    ops: List[OP] = []
    new_process_list = []
    for process in process_list:
        op_name, args = list(process.items())[0]
        ops.append(OPERATORS.modules[op_name](**args))
        new_process_list.append(process)
    for op_cfg, op in zip(new_process_list, ops):
        op._op_cfg = op_cfg
    return ops


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("nano-data-juicer L3 — OP interface: a filter's full behavior")
    print("=" * 68)
    print(f"声明: 复现 Data-Juicer Filter 接口语义（单进程 list 循环承载），")
    print(f"      语料复用 L2 合成语料 (seed=42)，跨级别契约见 [7]")

    # ---- [1] Registry + 配置驱动构造 ----
    print("\n[1] Registry + 配置驱动 (load_ops)")
    process_cfg = [
        {"text_length_filter": {"text_key": "norm_text",
                                "min_len": MIN_NORM_LEN}},
    ]
    ops = load_ops(process_cfg)
    op = ops[0]
    print(f"    config        : {process_cfg}")
    print(f"    实例类型      : {type(op).__name__}")
    print(f"    op._name      : {op._name!r}   <- 注册表赋值，非类自报")
    print(f"    op._op_cfg    : {op._op_cfg}   <- load_ops 存回实例")
    print(f"    注册表成员    : {sorted(OPERATORS.modules.keys())}")
    try:
        @OPERATORS.register_module("text_length_filter")
        class _Dup(TextLengthFilter):
            pass
    except KeyError as e:
        print(f"    重复注册      : KeyError ✅ ({e})")

    # ---- [2] 两段式执行 + stats 复用 ----
    print("\n[2] 两段式执行 (compute_stats -> process) + stats 复用")
    docs = [normalize_map(s) for s in make_corpus()]
    n0 = len(docs)

    STATS_COMPUTE_COUNT["text_len"] = 0
    stats_only_op = TextLengthFilter(text_key="norm_text", min_len=MIN_NORM_LEN)
    stats_only = stats_only_op.run(docs, reduce=False)     # 只算统计，不删
    n_compute_1 = STATS_COMPUTE_COUNT["text_len"]
    print(f"    reduce=False  : {len(stats_only)} 条全保留，"
          f"text_len 实算 {n_compute_1} 次")
    assert len(stats_only) == n0
    assert n_compute_1 == n0
    eg = stats_only[0]
    print(f"    stats 落点    : sample[{Fields.stats!r}] = "
          f"{eg[Fields.stats]}   <- 命名空间列，随样本走")

    kept = TextLengthFilter(text_key="norm_text",
                            min_len=MIN_NORM_LEN).run(docs)  # 复用已有 stats
    n_compute_2 = STATS_COMPUTE_COUNT["text_len"] - n_compute_1
    print(f"    reduce=True   : {n0} -> {len(kept)} 条；"
          f"第二个 filter 实算 text_len {n_compute_2} 次（全复用）✅")
    assert n_compute_2 == 0, "stats 已存在时必须复用，不得重算"

    # ---- [3] 区间语义边界表 (get_keep_boolean) ----
    print("\n[3] 区间语义边界表 (min_len=900, 只看 min 端)")
    variants = [
        ("闭区间 [900, +inf)       ",
         TextLengthFilter(min_len=MIN_NORM_LEN)),
        ("左开   (900, +inf)       ",
         TextLengthFilter(min_len=MIN_NORM_LEN, min_closed_interval=False)),
        ("reversed [900, +inf] 取反",
         TextLengthFilter(min_len=MIN_NORM_LEN, reversed_range=True)),
    ]
    names = [name for name, _ in variants]
    print("    val  | " + " | ".join(f"{name:<24}" for name in names))
    for val in (MIN_NORM_LEN - 1, MIN_NORM_LEN, MIN_NORM_LEN + 1):
        cells = []
        for _, f in variants:
            verdict = "keep" if f.get_keep_boolean(val, f.min_len, None) \
                else "drop"
            cells.append(f"{verdict:<24}")
        print(f"    {val} | " + " | ".join(cells))
    f_rev = variants[2][1]
    assert f_rev.get_keep_boolean(899, 900, None) is True
    assert f_rev.get_keep_boolean(900, 900, None) is True   # 边界点两侧都留
    assert f_rev.get_keep_boolean(901, 900, None) is False
    print("    reversed 语义 : 先把两端开闭取反、再整体取非 => 闭区间 [900,+inf)")
    print("                    反转为 (-inf,900]。注意边界点 900 在反转前后都被保留")
    print("                    —— reversed 不是集合补集，这是取反机制的算术结果 ✅")

    # ---- [4] 反例：配置拼写错误被静默吸收（schema 宽松的真实代价） ----
    print("\n[4] 反例: 参数拼写错误 -> 静默用默认值")
    typo_ops = load_ops([
        {"text_length_filter": {"text_key": "norm_text",
                                "min_lne": MIN_NORM_LEN}},   # typo!
    ])
    typo_op = typo_ops[0]
    print(f"    配置写 min_lne=900，未报错；op.min_len = {typo_op.min_len}"
          f"（掉回默认值 10）")
    assert typo_op.min_len == 10
    typo_docs = [normalize_map(s) for s in make_corpus()]
    typo_kept = typo_op.run(typo_docs)
    print(f"    后果: 保留 {len(typo_kept)}/{len(typo_docs)} 条"
          f"（正确配置应保留 {len(kept)} 条）——静默的分布污染 ❌")
    assert len(typo_kept) > len(kept)

    # ---- [5] NON_STATS_FILTERS：不产 stats 的 filter ----
    print("\n[5] NON_STATS_FILTERS: suffix_filter（双注册，无 stats 列）")
    print(f"    双注册: 'suffix_filter' in OPERATORS = "
          f"{'suffix_filter' in OPERATORS.modules}, in NON_STATS_FILTERS = "
          f"{'suffix_filter' in NON_STATS_FILTERS.modules}")
    suf_docs = [
        {"row_id": 0, "text": "clinical note a", Fields.suffix: ".txt"},
        {"row_id": 1, "text": "scan report",     Fields.suffix: ".pdf"},
        {"row_id": 2, "text": "discharge sum",   Fields.suffix: ".txt"},
        {"row_id": 3, "text": "lab results",     Fields.suffix: ".csv"},
    ]
    suf_kept = SuffixFilter(suffixes=[".txt"]).run(suf_docs)
    print(f"    过滤 suffixes=['.txt']: {len(suf_docs)} -> "
          f"{[s['row_id'] for s in suf_kept]}")
    has_stats = any(Fields.stats in s for s in suf_docs)
    print(f"    样本里出现 stats 列: {has_stats} —— NON_STATS filter "
          f"不注入 ✅")
    assert [s["row_id"] for s in suf_kept] == [0, 2]
    assert not has_stats

    # ---- [6] 接口守卫：__init_subclass__ 禁止直接重写 process ----
    print("\n[6] 接口守卫 (__init_subclass__)")
    try:
        class BadFilter(Filter):
            def process(self, sample):        # 直接重写 => 类定义期就炸
                return True
    except TypeError as e:
        print(f"    class BadFilter(Filter) 定义即抛:")
        print(f"    TypeError: {e}")

    # ---- [7] 跨级别契约：L3 filter 判定 == L2 串行漏斗 ----
    print("\n[7] 跨级别契约 (L3 filter == L2 serial funnel)")
    docs_c = make_corpus()
    ref = serial_run(
        docs_c, [o for o in build_pipeline() if not is_global_operation(o)])
    l3_out = TextLengthFilter(text_key="norm_text",
                              min_len=MIN_NORM_LEN).run(
        [normalize_map(s) for s in make_corpus()])
    ref_ids = [s["row_id"] for s in ref]
    l3_ids = [s["row_id"] for s in l3_out]
    print(f"    L2 串行漏斗 : {len(make_corpus())} -> {len(ref)} 条")
    print(f"    L3 filter   : -> {len(l3_out)} 条")
    print(f"    row_id 序列逐位一致: {ref_ids == l3_ids} ✅")
    assert ref_ids == l3_ids
    assert all(a["text"] == b["text"] for a, b in zip(ref, l3_out))

    # ---- [8] 边界反例：stats 键是全局名，不按 text_key 隔离 ----
    print("\n[8] 边界: stats 键全局命名 —— 先到先得，语义由首个计算者定义")
    docs_d = [normalize_map(s) for s in make_corpus()]
    _ = TextLengthFilter(text_key="norm_text",
                         min_len=MIN_NORM_LEN).run(docs_d)          # 先算
    reuse_kept = TextLengthFilter(text_key="text",
                                  min_len=MIN_NORM_LEN).run(docs_d)  # 后复用
    fresh_raw = TextLengthFilter(text_key="text",
                                 min_len=MIN_NORM_LEN).run(
        make_corpus())                                   # 无 stats，真算 raw
    print(f"    text_key='norm_text' 先算 -> 同批 text_key='text' 复用: "
          f"保留 {len(reuse_kept)} 条（仍是 norm 口径）")
    print(f"    干净语料上 text_key='text' 真算: 保留 {len(fresh_raw)} 条"
          f"（raw 口径，含扰动空白）")
    assert len(reuse_kept) == len(kept)
    assert len(fresh_raw) >= len(kept)
    print("    结论: StatsKeys.text_len 只认键名不认字段——真实 DJ 同样如此，")
    print("          stats 键的全局语义靠约定维护，混用 text_key 会静默错判。")

    # ---- self-check 汇总 ----
    print("\n" + "=" * 68)
    print("✅ self-check passed:")
    print("   registry 名字即身份 / 重复注册 KeyError / stats 复用零重算 /")
    print("   区间与 reversed 边界逐点吻合 / typo 静默吸收被当场抓出 /")
    print("   NON_STATS 不注入 stats / 接口守卫定义期拦截 /")
    print("   L3 判定 == L2 漏斗逐位一致 / stats 键全局性被实测复现")
    print("=" * 68)


if __name__ == "__main__":
    main()
