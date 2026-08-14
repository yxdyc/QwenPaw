# nano-data-juicer · Tutorial L3 — OP 接口与配置 schema：一个 Filter 的完整行为

> **级别**：L3（对齐权威实现）。前置：本模块 [L0](tutorial_L0.md)（OP 可组合 + 配置驱动）、[L1](tutorial_L1.md)（真实数据 + LLM OP）、[L2](tutorial_L2.md)（分布式执行语义）。
> **K+1 声明**：L2 的 OP 是「装着函数指针的 dataclass」——执行器按 kind 决定怎么跑，但 OP 本身长什么样没人管。L3 只加一层：**对齐 Data-Juicer 真实的 OP 接口，把一个 filter 从配置文件到判定输出的每一步复现出来**——类层级、注册表、两段式接口、stats 命名空间、区间判定、配置 schema。不碰并行执行（那是 L2 的事）。
> **可运行性声明**：纯标准库 + 同目录 import `L2_distributed_pipeline`（L2 有 `__main__` 守卫，导入安全）。复现的是**接口语义**：nano 用单进程 list 循环承载，真实 Data-Juicer 是 HF datasets `map`/`filter` + `num_proc` 多进程——执行形态差异逐条列在 §7。语料复用 L2 固定 seed 合成数据（跨级别契约要求同源）。

---

## 1. 问题：L2 的 OP 抽象在哪里不够用

L2 的 `OpSpec` 长这样：

```python
@dataclass
class OpSpec:
    name: str
    kind: str          # "mapper" | "filter" | "deduplicator"
    fn_map / fn_keep / dedup_key   # 函数指针
```

执行器按 `kind` 分派并行还是收敛，够用——但它回答不了一组成熟框架必须回答的问题：

1. **OP 从哪来？** 配置文件写 `text_length_filter: {min_len: 900}`，谁负责把它变成一个能跑的对象？名字和类怎么绑定？写错名字怎么办？写错参数怎么办？
2. **filter 的判定依据从哪来？** 长度过滤要先量长度——「量」和「判」是同一件事吗？量出来的结果存哪？第二个 filter 想用同一个统计量，重算还是复用？
3. **接口契约靠什么保证？** 如果某个 filter 作者直接重写了框架的调度入口，错误会在什么时候暴露——定义时、跑第一条数据时、还是跑完三小时之后？

这些问题 Data-Juicer 都有明确答案，而且答案就写在 `ops/base_op.py` 的类层级里。L3 把这层结构复现出来：**OP 不是函数，是注册过名字、由配置构造、按两段式接口执行的类**。

---

## 2. 运行与输出

```bash
$ python3 L3_filter_interface.py
```

以下为一次真实运行逐字粘贴（连跑 3 遍 `diff` 逐字节一致；L3 无计时声明）：

```text
====================================================================
nano-data-juicer L3 — OP interface: a filter's full behavior
====================================================================
声明: 复现 Data-Juicer Filter 接口语义（单进程 list 循环承载），
      语料复用 L2 合成语料 (seed=42)，跨级别契约见 [7]

[1] Registry + 配置驱动 (load_ops)
    config        : [{'text_length_filter': {'text_key': 'norm_text', 'min_len': 900}}]
    实例类型      : TextLengthFilter
    op._name      : 'text_length_filter'   <- 注册表赋值，非类自报
    op._op_cfg    : {'text_length_filter': {'text_key': 'norm_text', 'min_len': 900}}   <- load_ops 存回实例
    注册表成员    : ['suffix_filter', 'text_length_filter']
    重复注册      : KeyError ✅ ('text_length_filter is already registered in Operators')

[2] 两段式执行 (compute_stats -> process) + stats 复用
    reduce=False  : 3360 条全保留，text_len 实算 3360 次
    stats 落点    : sample['__dj__stats__'] = {'text_len': 805}   <- 命名空间列，随样本走
    reduce=True   : 3360 -> 2358 条；第二个 filter 实算 text_len 0 次（全复用）✅

[3] 区间语义边界表 (min_len=900, 只看 min 端)
    val  | 闭区间 [900, +inf)          | 左开   (900, +inf)         | reversed [900, +inf] 取反
    899 | drop                     | drop                     | keep
    900 | keep                     | drop                     | keep
    901 | keep                     | keep                     | drop
    reversed 语义 : 先把两端开闭取反、再整体取非 => 闭区间 [900,+inf)
                    反转为 (-inf,900]。注意边界点 900 在反转前后都被保留
                    —— reversed 不是集合补集，这是取反机制的算术结果 ✅

[4] 反例: 参数拼写错误 -> 静默用默认值
    配置写 min_lne=900，未报错；op.min_len = 10（掉回默认值 10）
    后果: 保留 3360/3360 条（正确配置应保留 2358 条）——静默的分布污染 ❌

[5] NON_STATS_FILTERS: suffix_filter（双注册，无 stats 列）
    双注册: 'suffix_filter' in OPERATORS = True, in NON_STATS_FILTERS = True
    过滤 suffixes=['.txt']: 4 -> [0, 2]
    样本里出现 stats 列: False —— NON_STATS filter 不注入 ✅

[6] 接口守卫 (__init_subclass__)
    class BadFilter(Filter) 定义即抛:
    TypeError: Method process cannot be overridden by subclass BadFilter. Please implement process_single or process_batched.

[7] 跨级别契约 (L3 filter == L2 serial funnel)
    L2 串行漏斗 : 3360 -> 2358 条
    L3 filter   : -> 2358 条
    row_id 序列逐位一致: True ✅

[8] 边界: stats 键全局命名 —— 先到先得，语义由首个计算者定义
    text_key='norm_text' 先算 -> 同批 text_key='text' 复用: 保留 2358 条（仍是 norm 口径）
    干净语料上 text_key='text' 真算: 保留 2364 条（raw 口径，含扰动空白）
    结论: StatsKeys.text_len 只认键名不认字段——真实 DJ 同样如此，
          stats 键的全局语义靠约定维护，混用 text_key 会静默错判。

====================================================================
✅ self-check passed:
   registry 名字即身份 / 重复注册 KeyError / stats 复用零重算 /
   区间与 reversed 边界逐点吻合 / typo 静默吸收被当场抓出 /
   NON_STATS 不注入 stats / 接口守卫定义期拦截 /
   L3 判定 == L2 漏斗逐位一致 / stats 键全局性被实测复现
====================================================================
```

---

## 3. 输出逐段解读

**[1] 名字即身份。** 配置就是一个 list：`[{op名: 参数字典}]`——和真实 Data-Juicer 的 `cfg.process` 同形。`load_ops` 逐条查注册表、按参数实例化。注意 `op._name` 不是类自己声明的，是**注册表在注册成功那一刻赋的**（`module_cls._name = module_name`）；`op._op_cfg` 则把原始配置存回实例——任何一个 OP 都能回答「我是哪行配置生的」。重复注册抛 KeyError：注册表是名字到类的**单射**，`force=True` 才允许覆盖。

**[2] 两段式 + stats 复用。** 本节的戏眼之一。`reduce=False` 只算统计不删样本：3360 条全在，每条的 `__dj__stats__` 列里多了 `{'text_len': ...}`（第一条是 805，低于 900，注定被过滤）。`reduce=True` 才执行判定，3360 → 2358。关键断言是第二个 filter **实算 0 次**——`compute_stats_batched` 里的 `if StatsKeys.text_len in stat: continue` 是复用的唯一入口，stats 已经长在样本身上，后来的 OP 直接读。

**[3] 区间边界表。** 三列是三种配置在 899/900/901 上的判定。闭区间与左开的差别只在 900 这一个点；reversed 列藏着一个反直觉事实：**边界点 900 在反转前后都被保留**。机制是「先把两端开闭取反、再整体取非」：闭区间 `[900,+∞)` → 取反成开区间 `(900,+∞)` → 取非得 `(-∞,900]`。两次取反在边界点上互相抵消，所以 reversed **不是集合补集**——这是从代码算术推出来、再被断言钉死的（写作时第一版断言就写错了，被自检当场抓住）。

**[4] 静默吸收反例。** 配置把 `min_len` 拼成 `min_lne`：不报错，`op.min_len` 掉回默认值 10，全量 3360 条一条不删——正确配置本该只剩 2358 条。这就是「没有 schema」的真实代价：错误不是异常，是**静默的分布污染**。与 L2 [4]a 的 naive 去重同款——数据 pipeline 的错误形态往往不是崩溃。

**[5] NON_STATS_FILTERS。** `suffix_filter` 同时注册进两个注册表：`OPERATORS`（能按名字构造）+ `NON_STATS_FILTERS`（声明「我不产 stats」）。`OP.run` 的注入条件因此跳过它——4 条样本过滤完，身上没有任何 `__dj__stats__` 列。它的 `compute_stats_single` 原样返回样本：流程照走，统计为零。

**[6] 接口守卫。** 试图在子类里直接重写 `process`——**类定义语句本身**抛 TypeError。错误在 import/定义期爆炸，而不是跑数据时。对照 [4]：typo 的代价是跑完才显现（甚至永不显现），接口违规的代价是零数据成本——框架把两类错误的拦截时机刻意拉开了。

**[7] 跨级别契约。** L3 的 `TextLengthFilter(text_key='norm_text', min_len=900)` 与 L2 的 `normalize_mapper + length_filter` 串行漏斗，幸存者 `row_id` 序列**逐位一致**（3360 → 2358）。这不是「条数碰巧相等」，是两套实现在同一语料上的判定逐样本对账——L3 的 OP 语义与 L2 的执行语义在契约上焊死。

**[8] stats 键的全局性。** 边界反例：先让 `text_key='norm_text'` 的 filter 算出 `text_len`，再让 `text_key='text'` 的 filter 跑同一批数据——后者**复用了 norm 口径的统计量**，判定结果仍是 2358；而干净语料上真算 raw 长度是 2364（多出的 6 条是注入重复的扰动拷贝：raw 含双空格更长，规范化后跌破 900）。`StatsKeys.text_len` 只认键名不认字段，真实 Data-Juicer 的代码同样如此（`if StatsKeys.text_len in stat: continue`，没有 text_key 参与）。stats 键的语义靠全局约定维护——这是 [2] 复用能力的另一面。

---

## 4. 代码结构：类层级就是接口契约

### 4.1 Registry：名字 → 类的单射

```python
def _register_module(self, module_name, module_cls, force=False):
    if module_name in self._modules and not force:
        raise KeyError(f"{module_name} is already registered in {self._name}")
    self._modules[module_name] = module_cls
    module_cls._name = module_name        # 名字来自注册表
```

与 `data_juicer/utils/registry.py` 同构。`register_module` 同时支持装饰器用法（`@OPERATORS.register_module("text_length_filter")`）与直接调用。两个注册表：`OPERATORS` 管「能构造」，`NON_STATS_FILTERS` 管「不产 stats」——后者是一个**性质标记表**，成员身份本身就是声明。

### 4.2 OP → Filter：run() 里的注入条件与两段式

```python
class OP:
    def run(self, dataset):
        if (isinstance(self, Filter)
                and self._name not in NON_STATS_FILTERS.modules):
            for s in dataset:
                s.setdefault(Fields.stats, {})
        return dataset
```

stats 列不是凭空出现的——`OP.run` 按条件注入，条件与真实 `base_op.py` 逐字同构（真实版还多一个 `Fields.stats not in dataset.features` 的已有列检查，nano 用 `setdefault` 表达同一语义）。

`Filter.__init__` 做三件事：收区间开关（`min/max_closed_interval`、`reversed_range`，reversed 时**先**把两端开闭取反）、按 `_batched_op` 把 `self.compute_stats` / `self.process` 绑定到 `*_batched` 或 `*_single`。注意绑定的是**实例属性**——这正是 [6] 守卫存在的原因（§5.4）。

```python
def run(self, dataset, reduce=True):
    dataset = super().run(dataset)                 # 注入 stats 列
    dataset = self.compute_stats(...)              # 阶段一：全量算统计
    if reduce:
        keep = list(self.process(...))             # 阶段二：只读 stats 判定
        dataset = [s for s, k in zip(dataset, keep) if k]
    return dataset
```

真实签名是 `run(dataset, *, exporter=None, tracer=None, reduce=True)`；nano 省略 exporter（stats 落盘）与 tracer（样本级追踪），保留 `reduce`——它是「只要统计不要过滤」的 profiling 模式开关。

### 4.3 TextLengthFilter：逐行复现的完整行为

```python
@OPERATORS.register_module("text_length_filter")
class TextLengthFilter(Filter):
    _batched_op = True

    def __init__(self, min_len: int = 10, max_len: int = sys.maxsize,
                 *args, **kwargs):
        ...
    def compute_stats_batched(self, samples):      # samples 是列式批
        for i, stat in enumerate(samples_stats):
            if StatsKeys.text_len in stat:         # 复用入口，唯一
                continue
            samples_stats[i][StatsKeys.text_len] = len(samples_list[i])
        return samples

    def process_batched(self, samples):
        return map(lambda stat: self.get_keep_boolean(
            stat[StatsKeys.text_len], self.min_len, self.max_len),
            samples[Fields.stats])
```

与真实 `ops/filter/text_length_filter.py` 逐行对应：`_batched_op = True`、默认值 `min_len=10 / max_len=sys.maxsize`、复用跳过、`process_batched` 返回 **map 对象**（调用侧要 `list()` 才能消费——真实源码的同款细节）。唯一的 nano 注入是计算计数器（`STATS_COMPUTE_COUNT`），让「复用与否」从日志变成数字。

列式批次（dict-of-lists）与行式样本（list[dict]）之间用两个转换器往返——这两个函数在真实 `base_op.py` 里同名存在，供异常捕获包装器使用。

### 4.4 load_ops：配置 schema 的两个面

```python
def load_ops(process_list):
    for process in process_list:
        op_name, args = list(process.items())[0]
        ops.append(OPERATORS.modules[op_name](**args))   # 查表构造
    for op_cfg, op in zip(new_process_list, ops):
        op._op_cfg = op_cfg                              # 配置回溯
    return ops
```

与 `ops/load.py` 同构。这段代码就是 schema 的全部执行者，于是边界一清二楚：**未知 OP 名**在 `OPERATORS.modules[op_name]` 处 KeyError（唯一硬的验证）；**参数拼错**被 `**args` → `kwargs.get` 链条静默吸收（[4]）。真实 Data-Juicer 的 `config/config.py` 里没有 jsonschema（2026-08-07 对 main 分支全文件检索确认）——宽松不是疏忽，是注册表 + 签名风格的自然结果，代价由用户承担。

### 4.5 守卫与例外：__init_subclass__ 和 suffix_filter

```python
def __init_subclass__(cls, **kwargs):
    for method_name in ["compute_stats", "process"]:
        if method_name in cls.__dict__:
            raise TypeError(f"Method {method_name} cannot be overridden ...")
```

检查的是 `cls.__dict__`——子类**自己定义**了这两个名字才触发；`Filter.__init__` 里的实例属性赋值不受影响。`suffix_filter` 演示例外通道：双注册进 `NON_STATS_FILTERS` 后，`compute_stats_single` 原样返回样本，`process_single` 直接读 `Fields.suffix` 判定（且自己处理 `reversed_range`——不走 `get_keep_boolean`，与真实源码一致）。

---

## 5. 机制深挖：四个「为什么」

### 5.1 为什么 compute_stats 和 process 要分两段

把「量」和「判」拆开，一次性买到三样东西：

1. **复用**：统计量写进样本就变成共享资产。第二个同键 filter 实算 0 次（[2]）；推广到真实 pipeline，N 个长度相关 filter 只付一次计算。
2. **profiling 模式**：`reduce=False` 只算不删——想看数据分布而不想动数据时，同一个 OP 换个参数就是分析工具。
3. **执行形态自由**：两个阶段各自是可并行的逐样本操作（L2 术语：都是局部 OP），真实 DJ 里它们各自走 `dataset.map` / `dataset.filter` + `num_proc`（`base_op.py:L834-840, L855-857`）；判定阶段只读 stats，极轻。

对照 `Deduplicator.run` 会更清楚这条设计线：指纹计算并行（`compute_hash` map），判定全局（`process(dataset)`）——**能并行的尽量并行，不能并行的老实收拢，而「算」与「判」永远分层**（L2 §5.2 的同一主题）。

### 5.2 为什么 stats 要长在样本的命名空间列上

stats 若存在 OP 实例里，数据集一流动就丢了；写进样本的 `__dj__stats__` 列，则：下游 OP 看得见（复用的前提）、落盘导出后可人工审查（data profiling）、OP 实例保持无状态（可任意重建/并行复制）。`__dj__` 前缀防止与用户字段（`text`、`images`…）撞名。

代价在 [8]：**键是全局约定，不按 OP、不按 text_key 隔离**。`text_len` 的语义由第一个计算者定义，后来者无条件复用。真实 DJ 用 `StatsKeys` 的全局命名表维护这套约定——复用能力与碰撞风险是同一个设计的两面。

### 5.3 为什么名字由注册表赋，而不是类自报

配置语言的核心是「名字 → 行为」的单射。注册表在注册成功时赋 `_name`（`registry.py` 的 `module_cls._name = module_name`），保证：类与名字一一对应（重复注册 KeyError）、名字错了立刻炸（load_ops 查表）、实例可回溯配置（`_op_cfg`）。如果让类自报名字，「两个类报同一个名字」只能在运行时撞车，而注册表把它提前到 import 期。

这也解释了 schema 的严格/宽松分界：**名字层严格**（查表失败即 KeyError），**参数层宽松**（`kwargs.get` 静默吸收）。前者是单射的数学要求，后者是 `**kwargs` 透传风格的自然结果——真实 DJ 的 `OP.__init__` 同样全程 `kwargs.get`。

### 5.4 为什么接口错误要在类定义期爆炸

`Filter.__init__` 按 `is_batched_op()` 把 `self.compute_stats` / `self.process` 动态绑定到批式或单样本实现——绑定的是**实例属性**。若子类直接重写类级 `process`，这层调度会被静默绕过：作者以为写了个 filter，框架实际跑的却是另一条路。`__init_subclass__` 在子类**定义时**检查 `cls.__dict__`，违规即 TypeError——错误成本为零条数据。

对照 [4] 的 typo：那类错误要到跑完全量数据才显形（甚至永不显形）。框架设计的一条原则藏在两者的时差里：**能把错误提前到定义期，就绝不留到数据期**。

---

## 6. 与 Data-Juicer 权威实现的对应

> 行号全部按 `github.com/modelscope/data-juicer` **main 分支**口径，2026-08-07 现场双通道核验：通道一 `raw.githubusercontent.com/modelscope/data-juicer/main/...` 抓取 9 个文件；通道二 codeload main tarball（57 MB）解包逐文件 `diff`，**9/9 字节一致，锚点零漂移**。本地 checkout `report_enhance@4e40654`（2026-05-11）与 main 有漂移（`base_op.py` 本地 1059 行 / main 1110 行），仅作交叉阅读，引用口径以 main 为准。

| toy 部件 | Data-Juicer 对应 | 源码锚点（main 分支） |
|---|---|---|
| `Registry`：名字→类、`_name` 赋值、重复注册 KeyError | `Registry._register_module` / `register_module` | `utils/registry.py:L66-83`（KeyError L80；`module_cls._name = module_name` L82-83）、`L85` |
| `OPERATORS` / `NON_STATS_FILTERS` / `DEFAULT_BATCH_SIZE` | 同名定义 | `ops/base_op.py:L22, L24, L27` |
| `Fields.stats = "__dj__stats__"`、`StatsKeys.text_len` | `DEFAULT_PREFIX` + `Fields.stats`；`StatsKeys.text_len` | `utils/constant.py:L12, L21, L345` |
| `convert_list_dict_to_dict_list` 与逆变换 | 同名函数（异常包装器用） | `ops/base_op.py:L30, L39` |
| `load_ops([{op名: 参数}])` | 同名函数：`OPERATORS.modules[op_name](**args)`；`op._op_cfg = op_cfg` | `ops/load.py:L18-19, L22-24`；调用点 `core/executor/default_executor.py:L168` |
| stats 列注入条件 | `OP.run`：`isinstance(self, Filter) and self._name not in NON_STATS_FILTERS.modules and Fields.stats not in dataset.features` | `ops/base_op.py:L560`（run）、`L577-580`（条件） |
| 区间开关 + reversed 翻转 | `Filter.__init__`：`min/max_closed_interval` 默认 True；reversed 时两端取反 | `ops/base_op.py:L748-753` |
| `get_keep_boolean` | 同名方法（逐字复现） | `ops/base_op.py:L786-794` |
| 两段式 `Filter.run(reduce=...)` | `run(dataset, *, exporter, tracer, reduce=True)`：先 map compute_stats，reduce 时 filter | `ops/base_op.py:L832`（签名）、`L834-840`（stats 段）、`L843, L855-857`（reduce/filter 段） |
| batched/single 运行时分派 | `__init__` 里按 `is_batched_op()` 绑定（真实版外裹异常捕获包装） | `ops/base_op.py:L755-770` |
| `__init_subclass__` 接口守卫 | Filter 禁 `compute_stats`/`process`；Mapper 禁 `process` | `ops/base_op.py:L772-781`（Filter）、`L639-648`（Mapper） |
| `TextLengthFilter` 完整行为 | `_batched_op=True`；`min_len=10`/`max_len=sys.maxsize`；stats 复用；`process_batched` 只读 stats | `ops/filter/text_length_filter.py:L8, L18, L20, L37-47`（复用 L42-43）、`L49-54` |
| `SuffixFilter` 双注册、不产 stats | `compute_stats_single` 原样返回；判定读 `Fields.suffix` | `ops/filter/suffix_filter.py:L10-11`（双注册）、`L38-39, L41-48` |
| NON_STATS 名单 | `["suffix_filter", "video_tagging_from_frames_filter"]` | `ops/filter/__init__.py:L123-126` |
| stats meta 排除 NON_STATS | `load_ops_with_stats_meta` 按 `NON_STATS_FILTERS.modules` 过滤 | `config/config.py:L1171-1183`（排除 L1180） |
| 无 jsonschema：schema = 注册表 + `__init__` 签名 | 2026-08-07 对 main 分支 `config/config.py` 全文件检索无 jsonschema | `config/config.py`（全文件，缺失性验证） |

**nano 版与权威实现的差异及原因**：

- **单进程 list 循环 vs HF datasets 多进程**：真实 `Filter.run` 的两个阶段都是 `num_proc` 并行的 `map`/`filter`（并发度由 `runtime_np()` 决定，`base_op.py:L523`）。L3 的主题是接口语义，执行并行已由 L2 覆盖，此处不重复。
- **list[dict] vs Arrow 表**：真实 dataset 是 HF Arrow；nano 用行式 list + 列式批转换复现同一 `_batched_op` 约定。转换器本身是真实 `base_op.py` 里同名存在的函数。
- **异常捕获包装省略**：真实 `__init__` 绑定的是 `catch_map_batches_exception` / `catch_map_single_exception` 包装后的方法（样本级容错，`skip_op_error` 控制吞错还是抛错）；nano 异常直通，样本级容错不是本节主题。
- **exporter / tracer 省略**：真实 `Filter.run` 支持 `stats_export_path` 落盘导出统计量（`base_op.py:L745, L841-842`）与样本级 tracer；复用语义在内存中已可完整演示。
- **`OP.run` 的其余职责省略**：NestedDataset 包装、TAGGING_OPS 的 meta 列、`index_key` 索引——与 filter 主线无关。
- **`STATS_COMPUTE_COUNT` 是 nano 注入的观测计数器**：真实系统没有；没有它，「复用零重算」只能靠间接推断。
- **语料复用 L2 合成数据**：跨级别契约要求同源；接口语义不依赖语料内容。

---

## 7. 费曼自检

**类比：质检站与任务单。** 注册表是监管局的**机构名录**：质检站的名字是监管局发的证，不是自己印的；冒名注册当场吊销（KeyError）。配置文件是当天的**任务单**：「机构名 + 参数」，`load_ops` 照单派活，完工后把任务单存根钉在机构墙上（`_op_cfg`）。两段式执行像**仪表与判定员分工**：仪表只负责把读数记到每件产品的档案卡上（stats 列随产品走），判定员只读档案卡决定放行——下一道工序要用同一个读数，直接看卡，不用重新测量（第二个 filter 实算 0 次）。区间语义是仪表的**合格线刻度**：边界值算不算合格，由开闭开关决定；把合格线「反转」时，先换刻度再翻结论，于是边界值在反转前后都留在合格区——反直觉，但算术如此。NON_STATS 是「看包装有没有破损」这类目测项目：流程照走，不填档案卡。接口守卫是监管局规章「仪表只能校准、不能私改线路」——违规在**挂牌之前**就被拦下，不是出了事故才追查。

**一句话版**：OP 不是函数，是注册了名字、由配置构造、按「算统计 → 读统计判定」两段执行的类；stats 是随样本流动的共享资产，判定只读不写。

**边界声明**：类比里「档案卡」在真实系统里是 Arrow 表的列，复用跨 run 还能落盘再读回；「监管局」只管名字与接口，不管参数对错——参数拼错照样派活（[4]），这正是真实系统的宽松面，不是类比的失真。

**反例版**：「参数写错了框架总会报错吧？」——[4] 实测：`min_lne` 拼错，静默用默认值 10，3360 条全保留，正确配置本该只剩 2358 条。「stats 算过一次，换个字段再算也一样吧？」——[8] 实测：`text_len` 只认键名不认字段，后到的 `text_key='text'` filter 复用了 norm 口径，判定静默错位。

---

## 8. 思考题

1. **stats 键命名**：如果把 `StatsKeys.text_len` 改成按字段隔离的名字（如 `text_len@{text_key}`），[8] 的碰撞消失——但真实 DJ 为什么仍用全局键？从 stats 落盘导出、跨 OP 复用、`load_ops_with_stats_meta` 的 meta 预注册（`config/config.py:L1171-1183`）三个角度各想一条理由。
2. **reversed 的边界点**：[3] 的表证明闭区间反转后边界点两侧都保留。对任意闭区间 `[a, b]`，证明反转后 `a`、`b` 两点仍被保留；若要求严格的集合补集（边界点恰好留一侧），`get_keep_boolean` 至少要改哪几行？改了之后旧的配置文件语义会发生什么？
3. **把静默吸收变响**：不改 `OP.__init__` 的 `kwargs.get` 风格，只在 `load_ops` 层加一道参数名校验（提示：`inspect.signature`）。写出来，然后回答：这道校验会误伤哪两类合法用法？（提示：基类透传的公共参数、子类新增参数的向后兼容。）
4. **契约外推**：把 `TextLengthFilter` 换成按**词数**过滤的 `words_num_filter`，[7] 的跨级别契约要改哪两处才能继续成立？（提示：`compute_stats_batched` 里的统计量算法，以及 L2 `length_keep` 的口径。）

---

## 9. 阶梯预告

- **本模块 L0–L3 完成**：L0 组合与配置、L1 真实数据与 LLM OP、L2 执行语义（分区/并行/收敛/容错）、L3 OP 语义（接口/stats/schema）——**执行与 OP 两半拼合**。此后读真实 Data-Juicer 的 `DefaultExecutor` 只需一步：`run` 循环逐 OP 调 `op.run`（见 [tutorial_L2.md §6](tutorial_L2.md) 的 `NestedDataset.process` 锚点，2026-08-05 口径），而 L3 的 `Filter.run` 就是其中一个 `op.run` 的内部。
- **解锁**（承 L2 §9）：03 轨 sota-deepdive（数据方法论：FineWeb / DCLM / Nemotron 数据报告）的开写门槛已满足。

**交叉引用**：[nano-ray L0](../nano-ray/tutorial_L0.md)——数据在 worker 间怎么流动是「谁搬 stats」的另一半；[nano-vllm-sglang L0](../nano-vllm-sglang/tutorial_L0.md)——batching 是另一种攒批，与 `_batched_op` 的列式批互为镜像；本模块 [L0](tutorial_L0.md) / [L1](tutorial_L1.md) / [L2](tutorial_L2.md)——OP 顺序语义与执行语义是 L3 接口契约的前提与归宿。

---

## 10. 溯源与口径声明

- **源码锚点**：§6 表格全部行号于 2026-08-07 现场双通道核验——通道一 `raw.githubusercontent.com/modelscope/data-juicer/main/...` 抓取 9 个文件（base_op / text_length_filter / load / suffix_filter / constant / registry / config / filter `__init__` / default_executor）；通道二 codeload main tarball 解包逐文件 `diff`，9/9 字节一致，锚点零漂移。本地 checkout `report_enhance@4e40654`（2026-05-11）与 main 有漂移（`base_op.py` 本地 1059 行 / main 1110 行），仅作交叉阅读。上游迭代可能再漂移。
- **toy 口径**：所有输出是本机（Apple Silicon, Python 3.13, `python3 -B`）真实运行结果；连跑 3 遍 `diff` 逐字节一致（L3 无计时声明，性能话题归 L2）。
- **[TODO: verify on real system]**（Machine B 通道攒批）：① 真实 DJ 全链路（yaml → launcher → executor）下 typo 参数是否同样静默——本文件按 `load_ops` + `OP.__init__` 逐字机制复现，但未在真实环境跑通全链路；② stats 键与 text_key 碰撞在官方 OP 集里的实际影响面（官方 text_length_filter 几乎总是用默认 text_key，影响面未核验）；③ stats 落盘持久化跨 run 复用的真实提速。
- **未核验项如实标注**：「无 jsonschema」仅覆盖 main 分支 `config/config.py` 当日检索，仓库其余位置未检索；`get_keep_boolean` 的 reversed 边界行为由逐字复现 + 断言实测得出，未见官方文档描述。
