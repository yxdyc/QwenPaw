# nano-ray · Tutorial L3 — object store：共享内存、零拷贝读取、引用计数与逐出

> **K+1 定位**：L1/L2 把 object store 当「账单」用——put 一次付 1 次序列化、传引用
> 只搬 ~28 B 句柄（L1 §5.1）、收敛点搬 6.14 MB（L1 [3]）——但从没摸过 store 的
> **构造**：数据到底住在哪？为什么读它不用拷贝、能不能写？装不下时会发生什么？
> 参数里的 ObjectRef 到底在哪一层被解开？L3 只加一层：把 store 从账单变成**实体**，
> 回答五个问题（[1] 物理构造 / [2] 字节账改写 / [3] 解引用规则 / [4] 装不下 / [5] 数据
> 密集调度），并兑现 L2 §10 的三条预告：L1 [1b]「包一层容器」包的到底是什么、
> L1/L2 两处 `[TODO: verify source]`（依赖解析的精确 C++ 位置）、以及 L2 [4] 的
> 字节账在零拷贝口径下怎么改写。
>
> **跨模块契约**：与 `nano-data-juicer/L2_distributed_pipeline.py`、`nano-ray/L1_ray_pipeline.py`、
> `nano-ray/L2_actor_dedup_index.py` 同一语料（seed=42，3360 条）、同一执行计划、
> 同一个漏斗：**3360 → 2358 → 2110**。本文件 import L1 的语料构造、OP 与漏斗期望值
> （`EXPECTED_*`），跨模块一致性在 [0] 处以机器断言复验。

---

## 1. 问题：store 一直在付账，但它住在哪里？

L1 的字节账（tutorial_L1 §5.1）把 store 当成一个黑盒收费处：

- `ray.put(x)`：付 1 次序列化，换一个 ObjectRef；
- 传 ref 给任务：只搬句柄，不搬数据；
- 传值给任务：每次提交都重新序列化一遍。

账是对的，但黑盒留下了四个答不出的问题：

1. **数据住哪？** 「put 进 object store」——store 是堆内存？是文件？是另一台机器？
   这决定了容量规划怎么做、OOM 长什么样。
2. **为什么读不用拷贝？** L1 测出传引用比传值快，但「快」只是结果。机制是什么？
   如果 get 返回的就是 store 里的数据本身，那写它会发生什么？
3. **装不下呢？** store 满了，put 是报错、丢数据、还是别的？L2 的 actor 索引常驻
   内存，生产环境里 store 压力是常态，这个行为必须知道。
4. **ref 在哪一层被解开？** L1 [1b] 留了个坑：把 ref 当参数传，任务收到的是值；
   把 ref 藏进 list 再 `ray.get`，会触发二次解引用的危险语义——所以「想持有 ref
   本身需包一层容器」。为什么包一层就行？分界线画在哪？

这四个问题的共同答案是：**object store 是一块有构造、有容量策略、有生命周期规则
的共享内存**，不是「另一个进程的堆」。L3 逐一拆开。

| L1/L2（store 是账单） | L3（store 是实体） | 变化的本质 |
|---|---|---|
| put 付 1 次序列化（黑盒） | store = init 时整体预分配的共享内存文件 | 数据的住所：抽象 → 物理（mmap 文件） |
| 传引用更快（观测） | get 返回 store 的只读 mmap 视图，零拷贝 | 机制：拷贝 → 映射 |
| store 满了？不知道 | 80% 触发 spill 到磁盘，get 透明 restore | 容量策略：报错 → 分层 |
| ref 自动解引用（黑盒） | 只有顶层参数位的 ref 进依赖表 | 语义边界：参数位 vs 容器内 |

---

## 2. 运行与输出

环境：`pip install ray`（本机实测 ray==2.56.1 / Python 3.13.13 / numpy 2.5.0 / macOS arm64）。
复用同目录 `L1_ray_pipeline.py`（import 前设 `sys.dont_write_bytecode`，全树零
`__pycache__`）。任意 CWD 可运行。

```bash
python L3_object_store_zero_copy.py
```

**可运行性契约声明**：`[0]–[4]` 用**真实 Ray**（本机 raylet + worker + plasma，
无任何 mock）——零拷贝、spilling、restore、依赖等待全部是真实行为，证据含 raylet
自身日志行；`[5]` 多节点 locality 本机不可跑，用**显式注明的本质模拟**（决策规则
逐条对照 ray 2.56.1 源码），真机验证标 `[TODO: verify on real system]`（真实 GPU/多机环境
通道）。下面 paste 块为掩去 `elapsed` 计时行后的输出（口径
`sed '/^[[:space:]]*elapsed/d'`），该掩码输出在双独立 CWD 三遍运行逐字节一致
（锚点见 §11）：

```
====================================================================
nano-ray L3 — object store：共享内存、零拷贝、引用计数与逐出
====================================================================
python 3.13.13 | ray 2.56.1 | numpy 2.5.0
声明: [0]–[4] 真实 Ray 单机模式（本地 raylet + worker + plasma），无 mock；
      [5] 多节点 locality 为显式注明的本质模拟（决策规则逐条对照
      ray 2.56.1 源码），真机行为 [TODO: verify on real system]。

[0] 跨模块契约: 语料 3360 docs -> 过滤后 2358 -> 去重后 2110（漏斗与 L1/L2 逐位一致 ✅）

[1] store 的物理构造：共享内存 + 零拷贝读取
    a) 32,000,000 B numpy：两次 ray.get 共享同一块内存 ✅
       （np.shares_memory=True、数据指针相同、只读、写入被 ValueError 拒绝）
       → get 没有把 32 MB 拷进驱动堆：返回的是 store 的 mmap 视图
    b) 4 个 worker 进程并发读同一 ref：全部只读视图、
       sum/nbytes 逐位一致 ✅（跨进程共享 = 每个 worker mmap 同一物理页）
    c) RSS 证据: 256 MB 对象 ray.get 3 次且每次全量触摸，
       峰值 RSS 增长 < 64 MB ✅（若每次 get 拷贝应 +768 MB）

[2] 字节账改写（L1 [1] 的零拷贝口径版）
    a) 同一 Python 对象 put 两次 -> 两个不同 ObjectRef ✅
       （store 的单位是 put，不是对象身份——L1 §5.1「无身份缓存」的 store 侧回声）
    b) 8 任务读 4 分区: 传引用（put 4 次 + ref 8 次）== 传值（8 次全量序列化）
       结果逐位一致 ✅ | 传值慢于传引用 ✅
    c) 改写口径: N 个任务共享一份数据 = 1 次序列化 + 1 份 store 拷贝 +
       N 次零拷贝读（同节点）；传值 N 次 = N 次序列化 + N 份 store 拷贝
       （> 100 KB 的参数自动 put，见 [4]；内联上限 10 MB，见 tutorial §5）

[3] 参数中 ObjectRef 的解引用规则
    a) 顶层 ref 参数: 任务在对象就绪后才开始 ✅
       （慢对象 sleep 2.0s；任务开始被对象就绪门控，必须晚于提交 ≥ 1.8s）
    b) ref 藏进 dict: 任务立即开工、不等对象 ✅（开始于提交后 < 1.0s，而对象要 2.0s 才就绪）
    c) 任务内 ray.get(嵌套 ref): 阻塞到就绪才返回 ✅（返回时刻必须晚于提交 ≥ 1.8s）——等待的责任从 runtime 移回任务自己
    d) 解引用只发生在「顶层参数位」: 顶层 -> int ✅；dict/tuple/list/
       自定义类里 -> 仍是 ObjectRef ✅（它们只进引用计数，不进依赖表）
    e) ray.put(ref) 被禁止，报错原文自带官方写法:
       "If you really want to do this, you can wrap the ray.ObjectRef
        in a list and call 'put' on it."
    f) 容器写法闭环: outer = ray.put([inner_ref]) -> 任务收到 [ObjectRef]，
       自己决定何时 ray.get ✅（L1 [1b]「包一层容器」包的这就是这个）

[4] 装不下时：75 MB 小 store 灌 128 MB（auto-put -> spill -> restore）
    a) store 是一个文件: raylet 日志声明 create_and_mmap_buffer(78643208, /tmp/ray/plasmaXXXXXX)
       请求 75 MiB -> 实得容量 78.6 MB，后备文件 78643208 B（≥ 容量 ✅，init 即整体预分配 mmap）
       目录里找不到它 ✅：mmap 后立即 unlink，映射经 fd 存活（POSIX 共享
       内存惯用法，lsof 可见 raylet 持有该 REG 映射）；Linux 上默认住
       /dev/shm——object store 从来不是「堆内存」
    b) 显式 put 8 × 16 MB = 128 MB > 75 MB store：全部成功 ✅
       日志坐实 spill 发生（pinned 对象不可逐出，put 要继续只能 spill）；
       最早的对象 get 回来逐字节完好 ✅（spill -> restore 层级透明），
       8 个对象逐一取回全部完好 ✅
    c) 8 任务 × 16 MB 传值参数（> 100 KB 全部 auto-put）:
       全部完成、结果无损 ✅——参数也是 store 对象，也吃同一套 spill 经济
    d) 机制链（raylet 日志逐行坐实）: 装不下 -> spill 到磁盘（对象仍计为
       primary、逻辑上随时可用）-> get 触发 restore -> 数据逐字节完好。
       「装不下」不是错误，是 store 的正常工作状态。

[5] 数据密集任务的调度：pull data vs move computation
    声明: 多节点本机不可跑——以下为显式注明的本质模拟，决策规则逐条
    对照 ray 2.56.1 源码（见 print 内锚点）；真机 [TODO: verify on real system]
    T1: grant-local      <- R1 参数全本地
    T2: wait-local+pull  <- R2 拉取 B（代价 0.6s）
    T3: wait-local+pull  <- R2 拉取 B（代价 0.6s）
    T4: spillback->N1    <- R3 C 拉不动，任务去数据节点
    pull 去重账: T2/T3 同依赖 B -> 网络流量 1 × 0.6 GB（不是 2 ×）✅
    → Ray 的 locality 是双向的：先拉数据（R2），拉不动就把任务送过去
      （R3）；「数据不动计算动」与「计算不动数据动」由同一套规则裁决。

[6] object store 使用标准
    多任务共享同一份数据          -> put 一次 + 传 ref（[2]：1 次序列化）
    大参数（> 100 KB）            -> 反正会 auto-put；显式 put 更可控（[4]）
    想让任务持有 ref 本身          -> 藏进容器传（[3d]）或 ray.put([ref])（[3f]）
    想让任务等数据就绪再开工        -> ref 放顶层参数位（[3a]：免费的依赖等待）
    容量规划                      -> store 是预分配文件；80% 起 spill（[4]）
    数据密集多节点                -> 依赖解析自带 locality（[5] R1–R3）

====================================================================
✅ self-check passed:
   漏斗 3360->2358->2110 与 L1/L2 一致 /
   两次 get 共享内存 + 只读 + 写入被拒 / 4 worker 只读视图一致 /
   256 MB × 3 get RSS 零增长 / put 无身份去重（两 ref）/
   传引用 == 传值 结果逐位一致且传值更慢 /
   顶层 ref 等就绪、嵌套 ref 立即开工、任务内 get 自阻塞 /
   dict/tuple/list/Box 内 ref 不被解引用 /
   ray.put(ref) 禁止且报错给出官方容器写法 / boxed 闭环 /
   75 MB store 灌 128 MB pinned + 再 128 MB auto-put 瞬态流过全绿：后备文件日志声明值 ≥ 容量、目录零残留、
   spill 触发/完成/restore 日志三行在位、数据逐字节完好
====================================================================

takeaway: object store 不是「堆内存的远房亲戚」，是一块 init 时就
          按容量整体预分配的共享内存文件：put 付 1 次序列化，之后谁读
          都是零拷贝的只读视图；装不下就 spill 到磁盘、按需 restore，
          数据完整性对上层透明。参数里的 ObjectRef 只在顶层参数位被
          解引用（附赠依赖等待）；藏进容器就只进引用计数——「包一层
          容器」包的正是这条分界线。L1/L2 的字节账就此改写：成本不在
          「读」，在「写进 store 的那一次」和「跨节点拉取的那一次」。
```

---

## 3. 代码结构：两个 Ray 会话 + 一个模拟体

文件分三段：

- **Session A（默认 store）**：`[0]` 跨模块契约复验 → `[1]` 零拷贝构造三证
  （双 get 共享 / 4 worker 跨进程 / RSS 零增长）→ `[2]` 字节账改写
  （put 无身份去重 + 传引用 vs 传值）→ `[3]` 解引用规则六证（a–f）。
- **Session B（75 MB 小 store）**：`ray.init(object_store_memory=75 MiB)`——
  75 MiB 是 ray 允许的最小 store（`ray_constants.py:L94`
  `OBJECT_STORE_MINIMUM_MEMORY_BYTES = 75 * 1024 * 1024`），故意造一个
  「装不下」的世界：`[4]` 灌 128 MB pinned 数据 + 128 MB auto-put 瞬态参数，
  读 raylet 自己的日志验证 store 的文件构造、spill、restore。
- **模拟体（`[5]`）**：两节点 `SimCluster`，四条决策规则 R1–R4 逐条镜像
  ray 2.56.1 的 `LocalLeaseManager` / `PullManager`（规则与源码锚点见 §5.5）。

全部 remote 函数自包含（不引用模块级名字），断言全部是机制断言（事件必须发生，
`wait_for` 超时即失败，不静默）——没有「跑完看输出猜对错」的环节。

---

## 4. 输出逐段解读

**[0] 契约。** 漏斗 3360 → 2358 → 2110 与 L1/L2 逐位一致。L3 换掉的是
「store 的认知」，不是数据平面——数据平面的一致性由断言钉死。

**[1] store 的物理构造。** 三证递进：

- a) 32 MB numpy `ray.put` 后连 get 两次：`np.shares_memory(v1, v2)=True`、
  数据指针相同、`flags.writeable=False`、手写 `v1[0] = -1` 被 `ValueError` 拒绝。
  若 get 是拷贝，两次 get 应是两块独立堆内存——实测是**同一块**，且只读。
- b) 4 个 worker 进程并发读同一 ref：全部落在不同 pid、全部只读、sum/nbytes
  逐位一致。**跨进程**共享同一份数据而不拷贝，唯一解释是共享内存映射。
- c) 256 MB 对象 get 3 次且每次全量触摸（`view.sum()` 强制访问每一页）：
  峰值 RSS 增长 < 64 MB（三遍实测均为 0 MB）。若每次 get 拷贝一份，
  3 × 256 MB = 768 MB 私有内存增长是逃不掉的。RSS 零增长 = 触摸的页
  从来不是驱动私有的——它们是 store 文件的映射页。

**[2] 字节账改写。** 同一 Python 对象 put 两次得到两个不同 ObjectRef——
store 的记账单位是 **put 这个动作**，不是对象身份（L1 §5.1「`serialize()` 无身份
缓存」在 store 侧的回声）。8 任务读 4 分区：传引用（4 次 put + 8 个 ref）与
传值（8 次全量序列化）结果逐位一致，传值恒慢（本日五跑：传引用 5.8–8.4 ms，
传值 7.9–17.1 ms）。改写后的口径（[2]c）就是 L1 账单的零拷贝版。

**[3] 解引用规则。** 六证画出那条分界线，见 §5.3。

**[4] 装不下。** 75 MB store 灌 128 MB pinned 数据：put 全部成功，raylet 日志
坐实 spill；最早的对象 get 回来逐字节完好（restore 耗时 1.3–1.6 s——磁盘层级
的价格，见 §5.4）；再来 8 × 16 MB 传值参数（全部 > 100 KB → auto-put），
任务全部无损完成。机制链：**装不下 → spill → restore，数据完整性对上层透明**。

**[5] locality。** 模拟体四任务演示 R1–R4（§5.5），核心结论：Ray 的 locality
是**双向**的——先拉数据，拉不动就把任务送过去。

---

## 5. 机制深挖

### 5.1 store 是一个文件：init 即整体预分配，mmap 后立即 unlink

`[4]a` 的第一手证据来自 raylet 自己的日志（本机 2026-08-12 运行实录）：

```
(raylet) dlmalloc.cc:153: create_and_mmap_buffer(78643208, /tmp/ray/plasmaXXXXXX)
```

这一行对应 ray 2.56.1 源码 `src/ray/object_manager/plasma/dlmalloc.cc:L140-166`
（POSIX 路径）：

```cpp
void create_and_mmap_buffer(int64_t size, void **pointer, int *fd) {
  // Create a buffer. This is creating a temporary file and then
  // immediately unlinking it so we do not leave traces in the system.
  ...
  RAY_LOG(INFO) << "create_and_mmap_buffer(" << size << ", " << file_template << ")";   // L153
  ...
  // Immediately unlink the file so we do not leave traces in the system.               // L164
  if (unlink(&file_name[0]) != 0) { ... }                                                // L165
```

三个信息点：

1. **store 在 `ray.init` 时就按容量整体预分配**——不是按需增长。请求 75 MiB，
   后备文件 78,643,208 B（≥ 容量 78,643,200 B；多出的 8 B 是分配器块头开销，
   合理推断，未逐行核）。这解释了容量规划为什么必须「先算好再 init」：
   store 大小是 init 参数，不是运行时属性。
2. **mmap 后立即 unlink**：目录项消失、映射经 fd 存活——POSIX 共享内存惯用法。
   所以 `[4]a` 断言「目录里找不到它」（macOS dlmalloc 路径实测零残留项）。
   好处：进程退出内核自动回收，不留垃圾文件；坏处：`df` / `ls` 看不见它，
   排障得靠 `lsof`。
3. **Linux 上默认住 `/dev/shm`**（`ray/_private/services.py:L2223-2231`：
   「/dev/shm on Linux, unless the shared-memory file system is too small」→
   `plasma_directory = "/dev/shm"`，安装包行号）——object store 从来不是
   「堆内存」，容量吃的是 shm 配额（容器里 `/dev/shm` 默认 64 MB，这是
   「容器里 ray 起不来」的常见根因之一，合理推断的应用提示）。

历史口径：Ray 论文（arXiv:1712.05889）里 plasma 是独立的共享内存存储服务；
ray 2.56.1 里它已并入 raylet 进程——实证就在上面那行日志的前缀 `(raylet)`：
创建 store 的是 raylet 自己，没有独立 plasma 进程。

### 5.2 零拷贝读取：get 返回的是 store 的只读视图

`[1]` 三证指向同一机制：`ray.get` 对大 numpy 不产生拷贝，返回的数组是
store 映射内存上的**视图**（`v1.base is not None`、数据指针 == store 页）。
「只读」不是约定，是强制：对象不可变性由内存保护落地，写视图直接
`ValueError`。这带来三个推论：

- **共享免费，写不可能**。多 worker 读同一 ref = 各自 mmap 同一物理页，
  零拷贝、零同步开销；代价是 store 对象**不可变**——要改数据就 put 新对象。
  不可变同时让故障恢复变简单：对象内容永远等于它被 put 时的样子，
  lineage 重算不用考虑中间被改过。
- **读的成本 ≈ 0，写的成本 = 1 次序列化 + 1 次进 store 的拷贝**。
  L1 账单里「put 付 1 次序列化」的那一笔，买的正是之后所有读的零拷贝。
- **传值参数不走这条路**：每次提交序列化出一份新的 store 对象（> 100 KB
  auto-put，§5.4），N 次传值 = N 份拷贝——`[2]b` 的「传值恒慢」就是这笔账。

### 5.3 解引用的分界线：顶层参数位 vs 容器内（填 L1 [1b] 的坑）

`[3]` 六证把 L1 [1b] 的「包一层容器」从口诀还原成机制。分界线的两侧：

**顶层参数位的 ref = 解引用 + 依赖等待。** `[3]a`：慢对象要 2.0 s 才就绪，
以它为顶层参数的任务**在对象就绪前根本不开始**（开始时刻晚于提交 ≥ 1.8 s，
实测 2.00–2.02 s）。这不是「任务里等你」，是**调度层不授权**——源码在
raylet 的 `LocalLeaseManager`（`src/ray/raylet/scheduling/local_lease_manager.cc`，
tag ray-2.56.1）：

```cpp
bool args_ready = lease_dependency_manager_.RequestLeaseDependencies(          // L105
    lease_id, lease.GetLeaseSpecification().GetDependencies(), ...);
if (args_ready) {                                                              // L110
  RAY_LOG(DEBUG) << "Args already ready, lease can be granted " << lease_id;
  leases_to_grant_[scheduling_key].emplace_back(std::move(work));              // L112
} else {
  ...  // 进 waiting_lease_queue_ 等依赖
```

而「哪些东西算依赖」由 task spec 定义（`src/ray/common/task/task_spec.cc:L363-371`）：

```cpp
std::vector<rpc::ObjectReference> TaskSpecification::GetDependencies() const {
  std::vector<rpc::ObjectReference> dependencies;
  for (size_t i = 0; i < NumArgs(); ++i) {
    if (ArgByRef(i)) {                                    // 只有 by-ref 的顶层参数
      dependencies.push_back(message_->args(i).object_ref());
    }
  }
  return dependencies;
}
```

这就同时填了 L1 §5.2 的 `[TODO: verify source]`，并修正其口径：L1 说「依赖解析
在 C++ core worker」——更精确地说，ref 的**所有权与引用计数**住在 core worker
（owner 模型），而「对象没就绪就不让任务开始」这道**闸门**住在 raylet 的
lease 调度里。Python 侧没有任何对应分支，所以 L1 当时只能看到报错。

**藏进容器的 ref = 只进引用计数，不进依赖表。** `[3]b`：ref 藏进 dict，任务
**立即开工**（开始于提交后 < 1.0 s，实测 0.003–0.004 s，对象还要 2.0 s 才就绪）；
`[3]c`：任务内自己 `ray.get` 那个嵌套 ref，阻塞到就绪才返回——等待的责任从
runtime 移回任务自己。`[3]d` 确认容器谱系：dict / tuple / list / 自定义类里的
ref 都保持 ObjectRef 身份，只有顶层参数位被解引用。机制上：容器里的 ref 只是
被序列化字节流「携带」，`GetDependencies` 的 `ArgByRef` 扫描看不见它们，
所以不进调度依赖；但 core worker 的引用计数仍会追踪它们（否则对象会被提前
回收）——「只进引用计数、不进依赖表」这半句由此得名。

**ray.put(ref) 被禁止，报错即文档。** `[3]e` 的报错原文
（`ray/_private/worker.py:L836-841`，安装包行号，GitHub tag 同文）：

```python
# Make sure that the value is not an object ref.
if isinstance(value, ObjectRef):
    raise TypeError(
        "Calling 'put' on an ray.ObjectRef is not allowed. "
        "If you really want to do this, you can wrap the "
        "ray.ObjectRef in a list and call 'put' on it."
    )
```

`[3]f` 按报错指路走完闭环：`outer = ray.put([inner_ref])` → 任务收到
`[ObjectRef]`，自己决定何时 get。**「包一层容器」包的正是这条分界线**：
容器本身不是顶层 ObjectRef，不会被解引用，也不会进依赖表——ref 的语义
（何时等、谁来等）完全交还给任务代码。L1 [1b] 的两个坑（对已解引用的值再
get、list 被当 ref 列表二次解引用）在这里收口：想持有 ref 本身，要么藏进
容器传（[3d]），要么 `ray.put([ref])`（[3f]），没有第三条路。

### 5.4 装不下：spill 是 store 的正常工作状态

Session B 把 store 压到 75 MiB（ray 允许的最小值），灌进 128 MB pinned 数据。
机制链逐环有源码与日志双证：

1. **何时触发**：raylet 在 primary 对象占用超过阈值时启动 spill
   （`src/ray/raylet/node_manager.cc:L2394-2409`，tag ray-2.56.1）：

   ```cpp
   void NodeManager::SpillIfOverPrimaryObjectsThreshold() {
     ...
     const float allocated_percentage =
         static_cast<float>(local_object_manager_.GetPrimaryBytes()) /
         object_manager_.GetMemoryCapacity();
     if (allocated_percentage >= RayConfig::instance().object_spilling_threshold()) {   // L2403
       RAY_LOG(INFO) << "Triggering object spilling because current usage " ...;        // L2405-2406
       local_object_manager_.SpillObjectUptoMaxThroughput();
     }
   }
   ```

   阈值 = **0.8**（`src/ray/common/ray_config_def.h:L749`
   `RAY_CONFIG(float, object_spilling_threshold, 0.8)`）——store 用到 80% 就
   开始往磁盘搬，不等满。

2. **谁被 spill**：驱动持有 ref 的对象是 pinned 的，不能被直接逐出（逐出 =
   数据没了），只能 spill（写磁盘、腾内存、对象仍计为 primary、逻辑上随时
   可用）。`[4]b` 的 8 个显式 put 对象全被驱动 pin 住，所以 128 MB 装进
   75 MB 的唯一出路就是 spill——raylet 日志 `:info_message:Spilled ...` 行
   （格式出自 `src/ray/raylet/local_object_manager.cc:L250-255`）坐实。
   机制下界可算术：128 MB 存活装进 75 MB，至少 53 MiB / ≥4 个对象必须
   离开内存（`[4]d` 的断言就是这个下界，实测累计 122–137 MiB / 8–9 对象）。

3. **restore 透明**：`ray.get(refs2[0])` 把最早（LRU 首位、必已 spill）的对象
   取回，逐字节完好；8 个对象逐一取回全部完好。restore 日志行
   （`local_object_manager.cc:L505-511`「Restored ... read throughput」）在位。
   价格是延迟：最早对象 get 实测 1311–1615 ms——内存层级是 ms 级，
   磁盘 restore 是秒级。**spill 买的是「put 不失败」，付的是「get 变慢」**。

4. **传值大参数走同一条路**：`[4]c` 的 8 × 16 MB 传值参数全部 > 100 KB，
   触发 auto-put——参数路径的决策在 `python/ray/_raylet.pyx`（tag ray-2.56.1）：

   ```python
   put_threshold = RayConfig.instance().max_direct_call_object_size()      # L801
   rpc_inline_threshold = RayConfig.instance().task_rpc_inlined_bytes_limit()  # L803
   ...
   if <int64_t>size <= put_threshold and \
           (<int64_t>size + total_inlined <= rpc_inline_threshold):        # L848-849
       ...  # 内联进任务 RPC
   else:
       put_id = ... put_serialized_object_and_increment_local_ref(...)     # L866-878
       args_vector.push_back(... CTaskArgByReference(put_id, ...))         # 变成 store 对象
   ```

   两个常数：`max_direct_call_object_size = 100 * 1024`（`ray_config_def.h:L245`，
   注释原文「values larger than this are stored in plasma instead」，L243-244）；
   `task_rpc_inlined_bytes_limit = 10 * 1024 * 1024`（`ray_config_def.h:L637`，
   单次任务 RPC 内联总量上限）——这就是 `[2]c` 里「> 100 KB 自动 put、
   内联上限 10 MB」的出处。auto-put 意味着**传值大参数也是 store 对象**，
   也吃同一套 spill 经济（`[4]c` 全绿：8 任务无损完成）。观察口径：瞬态参数
   自身是否被 spill 属异步观察项（本日两跑累计账 8 对象 vs 9 对象均出现；
   「80% 阈值触发」日志行在位与否也随批次浮动），断言只保机制下界。

**为什么 Ray 选择 spill 而不是报错？** store 的角色是「跨任务数据共享层」，
不是硬容量上限：对调用方承诺「put 成功 = 数据永远取得回」，把容量压力转成
延迟压力（restore 变慢）而不是失败压力（put 报错）。这是典型的**用分层换
可用性**——与虚拟内存的 swap 同构（合理推断的类比；spill 的对象选择、
触发时机是 raylet 自己的策略，不是 OS 页换出）。反面：若工作集长期大于
store，restore 延迟会吃掉全部收益——`[4]b` 的 1.3–1.6 s 就是预告。

### 5.5 数据密集任务的调度：pull data vs move computation（本质模拟）

多节点行为本机不可跑（`[TODO: verify on real system]`，真实 GPU/多机环境），
`[5]` 用显式注明的本质模拟，四条规则逐条镜像 ray 2.56.1 源码：

| 规则 | 模拟行为 | ray 2.56.1 源码锚点（tag ray-2.56.1） |
|---|---|---|
| R1 参数全本地 | 就地授权 | `local_lease_manager.cc:L105-112`（`args_ready` → `leases_to_grant_`） |
| R2 参数在远端且可拉取 | 本地等待 + pull | 同文件 `waiting_lease_queue_` 分支（L113-118）+ `pull_manager.h:L52-54`（「responsible for managing the policy around when to send pull requests and to whom」） |
| R3 拉不动（blocked） | 任务迁移到数据节点 | `local_lease_manager.cc:L443`（`SpillWaitingLeases`）、L463-468（blocked 判定，注释原文「we should force the lease onto a remote feasible node, even if we have enough resources available locally」）、L481（`/*exclude_local_node*/ lease_dependencies_blocked`） |
| R4 pull 按对象去重 | 两任务一对象，网络只拉一份 | `pull_manager.h:L479`（`object_pull_requests_` 以 ObjectID 为键）、L489（ObjectID → 等待它的请求集合） |

模拟输出：T1（本地依赖）就地授权；T2/T3（同依赖远端对象 B）本地等 pull，
pull 账记在**对象** B 上（`cl.pulls == {"B": 2}` 而 pull 对象数 = 1——网络
流量 1 × 0.6 GB 不是 2 ×）；T4（依赖 blocked 对象 C）反向迁移到 N1。

**senior 视角**：Ray 的 locality 是**双向**的——「数据不动计算动」与
「计算不动数据动」由同一套规则裁决（先 pull，blocked 才 spillback）。
R3 的注释值得逐字读：即使本地资源足够，只要依赖拉不动，就把任务送走——
调度器承认「有些数据根本拉不过来」（owner 失联、对象太大），与其让任务
在本地饿死，不如让计算搬家。模拟体省略了真实系统的资源准入、spread 策略
（`local_lease_manager.cc:L484-489`：spread 调度宁可本地等 pull 也不
spillback，以免破坏均匀分布——规则之外的二阶权衡）与多轮重试，真机行为
待 真实 GPU/多机环境 验证。

---

## 6. 与 Ray 权威实现的对应（取舍分析）

| 机制 | 本文实测 / 模拟 | ray 2.56.1 锚点（2026-08-12 核验） | 验证方式 |
|---|---|---|---|
| store = 预分配共享内存文件 | `[4]a` raylet 日志声明值 ≥ 容量、目录零残留 | `plasma/dlmalloc.cc:L140/L153/L164-166`；`ray_constants.py:L94`（75 MiB 下限）；`services.py:L2223-2231`（Linux 默认 /dev/shm） | 日志行号与源码逐位吻合 + 安装包行号 |
| 零拷贝只读视图 | `[1]` 双 get 共享 / 4 worker 跨进程 / RSS 零增长 | mmap 视图 + 不可变对象设计（§5.2 机制推断，行为全部实测） | 真实 ray 实测三证 |
| auto-put：>100 KB 进 store，内联上限 10 MB | `[4]c` 8 × 16 MB 传值参数全 auto-put | `_raylet.pyx:L801/L803/L848-849/L866-878`；`ray_config_def.h:L245/L637` | tag 源码 + 行为实测 |
| 顶层 ref = 解引用 + 依赖等待 | `[3]a` 门控 ≥1.8 s 实测 | `task_spec.cc:L363-371`（依赖 = ArgByRef）；`local_lease_manager.cc:L105-112`（args_ready 闸门） | tag 源码 + 计时实测 |
| 容器内 ref = 只进引用计数 | `[3]b/c/d` 立即开工 + 自阻塞 + 类型探针 | 同上（`ArgByRef` 扫描看不见容器内 ref） | 真实 ray 实测六证 |
| `ray.put(ref)` 禁止 + 官方容器写法 | `[3]e/f` 报错原文 + boxed 闭环 | `worker.py:L836-841`（安装包行号 = GitHub tag 行号） | 报错现场 + 源码逐字 |
| 80% 阈值 spill / restore 透明 | `[4]b/d` 128 MB 灌 75 MB 全绿 | `ray_config_def.h:L749`；`node_manager.cc:L2394-2409`；`local_object_manager.cc:L250-255/L505-511` | tag 源码 + raylet 日志 |
| locality 双向决策（R1–R4） | `[5]` 本质模拟 | `local_lease_manager.cc:L105-112/L443/L463-468/L481`；`pull_manager.h:L52-54/L479/L489` | 模拟规则逐条对照 tag 源码；真机 `[TODO: verify on real system]` |

**nano/教程侧没做的（差异与原因）**：

- **分布式引用计数（ownership）只碰到行为面**。驱动持 ref = pinned 是实测的，
  但 owner 表、跨节点 ref 传递、引用计数消息协议（core worker 的
  `ReferenceCounter`）不在本级展开——它们是「对象何时被回收」的完整理论，
  需要多节点真机才能演活，留给 真实 GPU/多机环境。
- **spill 的 IO 栈没展开**（对象序列化到哪个目录、并发 restore 的吞吐核算、
  `min_spilling_size` 等旋钮）——本机单节点下这些是配置项而非机制面。
- **plasma 的对象布局 / 分配器**（dlmalloc 的块头、对齐）只取了「+8 B」
  这一个观察点，标注为合理推断。

---

## 7. 费曼自检：共享仓库与它的三条规矩

把 object store 讲成**城市的共享仓库**：

- **入仓（put）**：你把货送到仓库，付一次运费（序列化），拿到一张仓单
  （ObjectRef）。仓库是 init 时就整块租好的地皮（预分配文件），不是
  你家客厅（堆内存）。
- **取货（get）**：谁拿着仓单都能**就地看货**——仓库给你开的是「 viewing
  窗口」（mmap 视图），不是把货搬回你家（拷贝）。所以看一千次都不累
  （零拷贝），但**只许看不许改**（只读）——要改？重新入一件新货。
- **仓满（spill）**：仓库满了不会拒收——把暂时没人取的货搬到地下室
  （磁盘），仓单照样有效，取的时候从地下室搬回来（restore），货一件
  不少，就是慢（秒级）。被仓单押着的货（pinned）不能扔，只能搬地下室。
- **仓单怎么用（解引用规则）**：把仓单**直接递**给工人（顶层参数），
  工头会等货就位才派工（依赖等待）；把仓单**塞进货箱夹层**（容器），
  工头看不见，工人自己决定何时去取。「包一层容器」包的就是工头的视线。

自检三问：能不能解释「两次 `ray.get` 为什么返回『同一块』内存、为什么必须
只读」？能不能解释「藏进 dict 的 ref 和放在顶层的 ref，各自进了哪张表
（引用计数表 / 依赖表），命运差在哪」？能不能解释「『装不下』为什么不是
错误、store 为此放弃了什么（延迟换可用）」？讲不出就回 §5.1 / §5.3 / §5.4。

---

## 8. 思考题

1. `[1]c` 用 RSS 峰值证明零拷贝。如果改成「get 后把视图 `copy()` 一份再释放」，
   RSS 曲线长什么样？`np.copy` 的钱付在哪一层（store 侧还是驱动侧）？
2. `[2]a` 说 put 无身份去重。假设你要处理 100 个任务共享同一份 10 GB 权重，
   正确的姿势是什么？如果不小心在循环里对同一对象 put 了 100 次，store 里
   会有什么、代价多大？（提示：store 的记账单位是 put 这个动作。）
3. `[3]b` 的嵌套 ref 让任务立即开工。什么场景下这是**优点**而不是坑？
   （提示：任务可以先做不依赖该对象的准备工作，把等待推迟到真正需要的
   那一刻——等待粒度从「任务级」细化到「语句级」。）
4. `[4]` 的 spill 把容量压力转成延迟压力。若工作集长期是 store 容量的 3 倍，
   系统行为会退化成什么？这时该调 `object_store_memory`、调数据布局
   （分片 + 流式处理），还是换存储层？各自的判据是什么？
5. `[5]` 的 R3 把任务送去数据节点。反过来想：什么情况下「拉数据」永远
   优于「搬计算」？（提示：比较对象大小与任务计算量；R2 的 pull 代价
   0.6 s vs 任务本身的运行时间。）

---

## 9. 反例与边界

1. **写只读视图**：`v1[0] = -1` → `ValueError`（`[1]a` 实测）。想「修改」
   store 对象的唯一路径是 put 新对象——不可变是设计，不是 bug。
2. **把 store 当无限容量**：spill 的 restore 是秒级（1.3–1.6 s 实测），
   比内存读慢三个数量级以上。把 store 当「大内存」用的代码，在容量线
   附近会发生断崖式变慢——容量规划必须在 init 前做完。
3. **嵌套 ref + 任务超时**：`[3]b/c` 的嵌套 ref 把等待责任交给任务自己；
   若任务内 `ray.get` 的对象永远不就绪（上游任务失败），等待会表现为
   任务挂起/异常——调试方向是「谁持有那个 ref」而不是「调度器卡了」。
4. **`ray.get(list)` 的二次解引用**（L1 [1b] 坑 b 的边界）：数据恰好长得像
   「ref 列表」时会被逐元素再解一次。L3 的 `[3]d/f` 给出正路，但没改
   `ray.get` 的这个语义——它还在（worker.py 的 list 分支），用 list 装
   普通数据时仍要当心。
5. **本机结论的边界**：`[1]–[4]` 全部是单机单 raylet 行为；多节点的
   pull/spillback（`[5]）`、跨机零拷贝（对象需先 pull 到本地 store 才能
   映射）、分布式引用计数，都标 `[TODO: verify on real system]`。
6. **toy 尺度不可外推**：32 MB / 256 MB 的 RSS 证据、75 MB store 的
   spill 账，都是为「让机制可见」选的尺寸；生产尺度（百 GB store、
   TB 级 spill）的吞吐数字以官方基准与真机实测为准，本文不外推。

---

## 10. 阶梯预告

nano-ray 的 L0–L3 阶梯到此完整：L0 任务图（把函数变成可调度单元）→
L1 真实 Ray 流水线（启动/序列化/提交的真实成本）→ L2 actor（状态的住所
与并发语义）→ L3 object store（数据的住所与容量策略）。四轨视角下，
03 轨的下一站是 **RSI 闭环专题**（agent 轨迹回流成训练数据）与
sota-deepdive（数据方法论，已开写）；交叉引用：本文的「不可变 + 零拷贝」
与 nano-vllm-sglang 的 KV cache 共享（只读页表项）同族，「spill 分层」与
nano-data-juicer L3 的缓存逐出同构——store 的经济学在栈的每一层都重演。

---

## 11. 溯源与口径声明

- **Ray 版本**：ray==2.56.1（pip 安装，Python 3.13.13，numpy 2.5.0，macOS
  arm64，单机分别以 4 CPU 与 2 CPU 配置运行）。GitHub 锚点一律 tag **ray-2.56.1**，
  raw 通道 2026-08-12 现场重抓；
  安装包行号（`ray_constants.py:L94`、`worker.py:L836-841`、
  `services.py:L2223-2231`）同日在本机 site-packages 逐条核对，与 GitHub tag
  行号双通道吻合。Ray 论文 arXiv:1712.05889（Moritz et al.，经典锚点）。
- **源码锚点清单（全部 2026-08-12 现场核验在位）**：
  `python/ray/_private/ray_constants.py:L94`（75 MiB 下限）/
  `src/ray/common/ray_config_def.h:L243-245`（100 KB auto-put 阈值 + 注释原文）、
  `L636-637`（10 MB 内联上限）、`L749`（0.8 spill 阈值）/
  `python/ray/_raylet.pyx:L801、L803、L848-849、L866-878`（参数路径）/
  `python/ray/_private/worker.py:L836-841`（put(ref) 禁止，报错逐字）/
  `src/ray/common/task/task_spec.h:L268` + `task_spec.cc:L363-371`（依赖 = ArgByRef）/
  `src/ray/raylet/scheduling/local_lease_manager.cc:L105-112、L443、L463-468、L477-483`
  （args_ready 闸门 / SpillWaitingLeases / blocked 判定 / exclude_local_node）/
  `src/ray/object_manager/pull_manager.h:L52-54、L479、L489`（pull 策略与按对象去重）/
  `src/ray/raylet/node_manager.cc:L2394-2409`（80% 触发）/
  `src/ray/raylet/local_object_manager.cc:L250-255、L505-511`（Spilled/Restored 日志格式）/
  `src/ray/object_manager/plasma/dlmalloc.cc:L140-166`（create_and_mmap_buffer + unlink）。
- **本机实证日志**：raylet.out 行 `(raylet) dlmalloc.cc:153:
  create_and_mmap_buffer(78643208, /tmp/ray/plasmaXXXXXX)`（2026-08-12 运行
  实录，行号与 tag 源码逐位吻合）；「Triggering object spilling」/「Spilled」/
  「Restored」日志行同批在位。
- **计时口径**：所有 ms/s 为记录当日（2026-08-12）**五跑**（run3–run7，
  含双独立 CWD）观测区间，随机器负载浮动：ray.init 2.5–5.8 s；小 store
  init 2.4–5.3 s；传引用 5.8–8.4 ms vs 传值 7.9–17.1 ms（传值恒慢，五跑无
  翻转）；顶层 ref 开始延迟 2.00–2.02 s；嵌套 ref 开工延迟 0.003–0.004 s；
  restore get 1311–1615 ms；8 任务墙钟 1.33–1.37 s；spill 写吞吐 196–365
  MiB/s。计数/结构类输出（漏斗、断言、类型探针、日志声明值）五跑掩计时后
  **逐字节一致**：掩码口径 `sed '/^[[:space:]]*elapsed/d'`，掩码输出 md5
  `465bff182328ac12e6dcff221b269a69`/96 行（run5/run6/run7 三跑 × 双 CWD
  收敛；§2 paste 块即该掩码输出）。异步观察项（累计 spill 账 122–137 MiB /
  8–9 对象、「Triggering」日志行在位与否）随批次浮动，断言只保机制下界
  （≥53 MiB / ≥4 对象），如实记录。
- **语料**：seed=42 合成语料（非真实语料），与 nano-data-juicer L2 /
  nano-ray L1/L2 同一构造；L3 主题是 store 的构造与容量策略，语料内容
  不影响验证。
- **修复记录**：早期版本（md5 `3b2bda79389ca38ab8ec4ddea30fb75a`/565/31,232 B）有三处 print 把实测
  计时值写进了非掩码行（[3]a/b/c，掩码锚会随负载漂移）+ self-check 行一处
  无溯源数字（「128+96 MB」之 96 无从推导）+ 文件头 docstring 两处与实测
  不符的表述（「被 spill 的包括 8 个传值大参数」——实测累计 spill 账的
  对象数 = pinned 显式 put 数，瞬态参数是否被 spill 属异步浮动）。本批
  已全部修复为 print/docstring 路径同行替换，机制内容、断言、实验设计未改动；
  定版 md5 `b54e634970af1d890a6a486a9fc4b229`/566/31,447 B。
- **范围外**：多节点 pull/spillback 真机行为、分布式引用计数协议、spill
  IO 栈细节 → `[TODO: verify on real system]`（真实 GPU/多机环境）。
