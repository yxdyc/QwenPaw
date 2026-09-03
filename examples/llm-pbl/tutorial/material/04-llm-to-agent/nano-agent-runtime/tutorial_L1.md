# nano-agent-runtime L1 — 进程被 kill -9 之后，谁来保证副作用只做一次？

> **核心问题**：L0 的崩溃是同一进程内的异常；真实的崩溃是整个进程消失。当 runtime、provider
> 都可能在任意时刻被 SIGKILL 时，怎样让每个外部副作用「恰一次」？
>
> **先修**：[L0](tutorial_L0.md) 的 prepare/commit、payload-bound idempotency、needs_human 四件套。
>
> **不变量（继承 L0，全部落盘）**：模型只提出 intent；授权来自可信 policy；idempotency key 绑定
> payload 指纹；不可确定状态不得自动成功——区别是这些判断现在写在 SQLite/WAL 里，进程死了也在。
>
> **运行**：`python3 -B L1_durable_tool_runtime.py`，必须在**空目录**里跑（会创建 runtime.db /
> provider.db 及其 -wal/-shm 文件）。纯标准库（sqlite3/subprocess），CPU，固定输出。
>
> **验收**：20/20 self-check；三个 kill 窗口（worker 死于 provider 提交后 / provider 死于自己提交前 /
> provider 死于自己提交后回复前）各恢复一次，余额账逐分不差；并发重复提交只扣一次款。
>
> **边界**：单机 SQLite、toy 余额账、子进程级 kill；没有网络分区、没有真支付 API、没有掉电测试。
> L2 才做多 worker lease / outbox / compensation。

---

## 1. 运行与输出

可运行性契约：L1 必须可跑。下面的输出块来自空目录中的真实运行，并由两个新的空 CWD
复跑确认逐字节一致（输出无计时行，无需掩码；复现锚点见 §13）。

```bash
python3 -B L1_durable_tool_runtime.py
```

```text
==============================================================================
Agent runtime L1 — durable tool runtime: SQLite/WAL, real kill, real recovery
==============================================================================
toy: stdlib only (sqlite3/subprocess) | provider = separate process (bank)
protocol inherits L0: typed intent -> authorize -> PREPARED -> provider -> COMMITTED

[1] Happy path: one intent, one durable commit
    worker rc=0 receipt=prov-receipt-1
    replay of committed key rc=0 receipt=prov-receipt-1 (no new commit, no provider call)
    balances={'attacker': 0, 'research': 175, 'vendor': 25}

[2] SIGKILL the worker after provider commit (response lost)
    worker rc=-9 (SIGKILL) | runtime.db-wal persisted after kill=True
    pre-recovery: commit rows for key=0, latest event=PREPARED
recovery start: chain_ok=True events=3
recover key=inv-2 via=query receipt=prov-receipt-2
recovery complete: query=1 replay=0
    balances={'attacker': 0, 'research': 150, 'vendor': 50}  (no double debit)

[3] SIGKILL the provider before its commit (effect not durable)
    provider killed mid-transaction | worker rc=3 (UNCERTAIN) | provider.db-wal persisted after kill=True
recovery start: chain_ok=True events=6
recover key=inv-3 via=replay receipt=prov-receipt-3
recovery complete: query=0 replay=1
    balances={'attacker': 0, 'research': 125, 'vendor': 75}  (applied exactly once)

[4] SIGKILL the provider after its commit, before reply (effect durable, reply lost)
    provider killed after commit | worker rc=3 (UNCERTAIN)
recovery start: chain_ok=True events=9
recover key=inv-4 via=query receipt=prov-receipt-4
recovery complete: query=1 replay=0
    balances={'attacker': 0, 'research': 100, 'vendor': 100}  (no double debit)

[5] Concurrent duplicate submission: 2 workers, same key+payload
    worker rcs=[0, 0] receipts=['prov-receipt-5', 'prov-receipt-5']
    commit rows for key=1 | provider receipts for key=1
    balances={'attacker': 0, 'research': 75, 'vendor': 125}  (debited exactly once)

[6] Idempotency is payload-bound; authority is default-deny
    same key + different amount -> rc=6 REJECTED idempotency key reused with different payload
    out-of-scope source -> rc=5 REJECTED source is outside principal scope
    provider receipts unchanged=5

[7] Legacy effect without query/replay support -> NEEDS_HUMAN survives restarts
marked key=legacy-mail-9 NEEDS_HUMAN
    worker attempt rc=4 (BLOCKED uncertain side effect requires human resolution (key=legacy-mail-9))
    recover pass 1: needs_human unchanged=True
    recover pass 2: needs_human unchanged=True
    commit rows for key=0

[8] Durability evidence + cross-level anchor + self-check
    journal_mode: runtime=wal provider=wal
    integrity_check after all kills: runtime=ok provider=ok
    event hash chain verifies=True (18 events)
    cross-level anchor (L0 crash+retry == L1 kill+recover): c2e95d2a0c5b0809 match=True
    PASS | [1] happy path committed once, balances moved once
    PASS | [1] replay of committed key returns stored receipt without new effect
    PASS | [2] worker SIGKILL after provider commit was exercised
    PASS | [2] runtime.db-wal persisted after the kill, before recovery opened it
    PASS | [2] pre-recovery state was a PREPARED orphan with zero commit rows
    PASS | [2] recovery adopted the provider's receipt via query (no second debit)
    PASS | [3] provider SIGKILL before commit left the worker UNCERTAIN
    PASS | [3] provider.db-wal with the uncommitted transaction persisted after the kill
    PASS | [3] recovery replayed the lost effect exactly once
    PASS | [4] provider kill after commit resolved via query, not a second apply
    PASS | [5] concurrent duplicates both succeeded with the identical receipt
    PASS | [5] exactly one commit row and one provider receipt for the duplicated key
    PASS | [5] concurrent duplicate debited the account exactly once
    PASS | [6] same key with different payload rejected (payload-bound idempotency)
    PASS | [6] out-of-scope source rejected by default-deny authority
    PASS | [6] rejections produced no provider-side effect
    PASS | [7] NEEDS_HUMAN persisted across restarts and was never auto-committed
    PASS | [8] journal_mode=wal active on both durable stores
    PASS | [8] integrity_check=ok on both stores after all SIGKILLs
    PASS | [8] hash chain verifies end-to-end; L0 and L1 agree on the observable semantics

SELF-CHECK: 20/20 PASS
digest(md5 of metrics) = aa1b4f2517ed5f0b48154028e939f8f7
takeaway: durability is a protocol, not a hope — WAL makes records survive kills, payload-bound keys make replays safe, and recovery turns uncertainty into exactly-once or needs_human.
```

读法：`[1]` 是参照系（一次 intent 一次 commit）；`[2][3][4]` 是三个 kill 窗口各杀一个真实子进程
（`rc=-9` 就是 SIGKILL 的返回码，不是模拟）；`[5]` 两个 worker 子进程并发提交同一个 key；`[6][7]`
是安全边界；`[8]` 给出持久性证据与跨级锚。余额账从 `research=200` 开始，五笔 25 的转账后
`research=75 / vendor=125`——任何一次恢复若重复执行，这个数字就会错，self-check 会红。

---

## 2. 从 L0 到 L1：协议没变，变的是「谁记得」

L0 与 L1 跑的是同一个协议：typed intent → authorize → PREPARED → provider effect → COMMITTED。
差别全在持久性这一维：

| 维度 | L0 | L1 |
|------|----|----|
| event log | Python list，进程死即失忆 | SQLite 表 + hash chain，`PRAGMA journal_mode=WAL` |
| commit 记录 | dict `receipts` | `commits(key PRIMARY KEY, fingerprint, receipt, method)` |
| 崩溃方式 | 同进程抛 `SimulatedCrash` | `os.kill(os.getpid(), SIGKILL)` 杀真实子进程 |
| provider | 同进程对象 | 独立子进程 + 自己的 provider.db（可独立被杀） |
| 恢复 | 同进程再调一次 | **新进程** `recover`：读盘 → 查询/重放 → 补 commit |
| 并发 | 不存在 | 两个 worker 子进程同 key 竞态 |

K+1 的要点：L1 没有发明新机制，只是把 L0 的每个判断从「内存里的约定」变成「磁盘上的事实」。
跨级锚（`[8]` 的 `cross-level anchor`）把 L0 的 crash+retry 与 L1 的 kill+recover 的**可观察语义**
（余额变化量、commit 行数、receipt 稳定性）做 canonical JSON 摘要比对，两级逐位一致
（`c2e95d2a0c5b0809`）——持久化升级不改变协议语义，这正是「机制不变、载体升级」的机器证明。

---

## 3. 为什么是 WAL：commit 的解剖

SQLite 默认的原子提交靠 rollback journal：先把原页面抄进日志，再原地改库文件，删日志即提交。
WAL 把这个方向倒过来——原数据库页暂时不动，改动追加进 `-wal` 文件；当 WAL 追加一条表示
提交的特殊记录时，事务才算提交。这个定义来自 [SQLite WAL 官方文档](https://sqlite.org/wal.html)，
也是 L1 持久性论证的地基：**提交 = WAL 中出现 commit record**。于是——

- 进程在 commit record 追加前被杀：WAL 里只有未提交的帧，下次打开时恢复流程丢弃它们，
  就像从未发生。`[3]` 里 provider 正是死在这个窗口（`BEGIN IMMEDIATE` 之后、`conn.commit()`
  之前自杀），`provider.db-wal persisted after kill=True` 记录的就是这些「 staged 但从未生效」
  的帧；恢复后 `integrity_check=ok`、余额分文未动。
- 进程在 commit record 追加后被杀：改动已持久，哪怕回复永远没发出去。`[4]` 就是这个窗口。
- `PRAGMA synchronous=FULL` 会在每次事务提交时同步 WAL；NORMAL 省略这次同步，断电时可能
  丢失最近提交，但不会破坏数据库。L1 选 FULL：对副作用 runtime 而言，**丢一条已承诺的
  commit 记录比慢一点贵得多**。
- 最后一个连接异常退出后，新连接首次打开数据库会执行恢复，并在恢复期间持有排他锁。
  `recover` 进程打开 runtime.db 的那一刻，WAL 恢复由 SQLite 自动完成；这不是 L1 自己实现的逻辑。

`[2]`/`[3]` 里检查 `-wal` 文件在 kill 后、恢复打开前**确实存在于磁盘**，是为了把「WAL 在崩溃
后仍在」从文档断言变成实验事实。为什么 WAL 而不是默认 rollback journal？两个理由：其一，
WAL 的提交先追加、不直接覆盖原数据库页，恢复期外同一数据库的读者可以与 writer 并行；
其二，追加式 commit record 让「提交点」成为
一个离散的、可引用的事实，与 prepare/commit 协议天然同构。

---

## 4. 三个 kill 窗口：不确定性的分类学

一次跨进程副作用的完整时间线上，有三个「死了就说不清」的窗口：

```text
worker: PREPARED ──► provider apply ──► [w1] ──► commits.insert ──► COMMITTED
provider:            BEGIN ──► 写余额/receipt ──► [p1] COMMIT [p2] ──► 打印 receipt
```

| 窗口 | 谁死 | 外部世界 | 本地记录 | 恢复动作 | 对应场景 |
|------|------|----------|----------|----------|----------|
| `[w1]` worker 死于 provider 返回后 | worker | 已提交（有 receipt） | 只有 PREPARED | **query**：问 provider 要 receipt，补 COMMITTED | `[2]` |
| `[p1]` provider 死于自己提交前 | provider | 未发生 | PREPARED + UNCERTAIN | **replay**：同 key 重放（provider 幂等，安全） | `[3]` |
| `[p2]` provider 死于提交后、回复前 | provider | 已提交 | PREPARED + UNCERTAIN | **query**：receipt 已在 provider 账上 | `[4]` |

三个窗口只有两种药：**能查就查（query），查不到才重放（replay），既不能查又不能安全重放
就 NEEDS_HUMAN**。`recover` 的决策树就是这三行——它不是「重试逻辑」，是**状态判别器**：
对每个没有 commit 记录的 key，先问外部世界「你那边有吗」，再决定采纳、重放还是升级人工。

`[3]` 与 `[4]` 都让 worker 以 rc=3 报 UNCERTAIN——注意 UNCERTAIN 是一种**持久状态**而非错误：
它被写进 event log（`[3]` 的 events=6 含 UNCERTAIN 事件），重启后依然可辨。L0 说过
「不可确定状态不得自动成功」，L1 把它落盘成「不可确定状态**重启后仍然**不得自动成功」。

`[7]` 是第三种 provider：既无幂等键也不能查询（legacy 邮件网关这类）。`legacy-mark` 把它标成
NEEDS_HUMAN 后，两次 `recover`（两个新进程）都输出 `needs_human unchanged`，commit 行数始终
为 0——人工介入是唯一出口，恢复进程绝不「为了完成任务」盲重放。

---

## 5. 恢复是一个新进程，不是一次重试

`recover` 模式刻意以**全新进程**运行：它不继承任何内存状态，唯一输入是 runtime.db /
provider.db 两个文件。这是「重启」的字面意义，也逼出两条设计约束：

1. **PREPARED 事件必须携带完整 intent**。重放需要 source/destination/amount，这些信息若只在
   死掉进程的内存里，恢复就退化成 NEEDS_HUMAN。L1 的 PREPARED data 里存着 canonical intent +
   fingerprint + principal——durable intent record 是安全重放的前提。
2. **commits 表是权威状态，event log 是审计历史**。`commits(key PRIMARY KEY)` 回答「这个 key
   到底提交没有」；event log 回答「发生过什么」。`[5]` 的并发场景里两者会短暂分叉：两个
   worker 都 PREPARED、都拿到同一 receipt、都尝试 `INSERT INTO commits`——UNIQUE 约束保证恰
   一个成功，败者改走 `COMMIT_OBSERVED` 事件（它观察到了提交，而非又做了一次提交）。最终
   commit 行数 = 1、provider receipt 数 = 1、扣款一次；event log 里 4 条事件完整保留了竞态
   的审计痕迹。**把「权威状态」与「审计历史」分成两个结构，是并发下仍能说清「恰一次」的关键。**

hash chain 在 L1 继续承担防篡改/防错序的职责：`recovery start: chain_ok=True` 是 recover 的
第一行输出——先验日志完整性，再谈恢复。恢复本身也走同一条链：每条补记的 COMMITTED 事件
都带 `method=recovered-via-query / recovered-via-replay`，事后审计能区分「直接提交」与
「恢复提交」，而两者的业务效果完全相同。

---

## 6. 并发重复提交：两道闸，两层机制

`[5]` 用 `Popen` 同时拉起两个 worker 子进程，提交同一 key+payload。竞态要过两道闸，
两层的机制必须分开讲：

- **第一道闸，provider 层——锁内原子幂等**。`provider_apply` 在碰 receipts 表之前先执行
  `BEGIN IMMEDIATE`（代码 L126）：先拿写锁，再查 `receipts(key PRIMARY KEY)`。check-then-act 因此在写锁内
  成为一段原子区间：第一个 apply 插入 `prov-receipt-5` 并提交；第二个 apply 经
  `busy_timeout=10000` 排队拿到锁后，只能看到两种事实——「还没插入（我来插）」或
  「已插入」。后者就是锁内读到同 key 同 fingerprint → 原样返回同一 receipt（`row[1]`），
  无新余额变动、无新 receipt 行。两个 worker 从 provider 拿到的 receipt 因此**真的相同**
  （输出里 `receipts=['prov-receipt-5','prov-receipt-5']`）——这个「相同」的致因是
  **锁的原子性**：`BEGIN IMMEDIATE` 把「查 + 插」串行成一段独占区间，后到者没有第三种
  状态可见。
- **第二道闸，commits 层——竞态仍在，观察纠偏**。两个 worker 都拿到真 receipt 后几乎
  同时 `INSERT INTO commits(key PRIMARY KEY)`：PK 唯一约束保证恰一个赢家；败者捕获
  `IntegrityError`，读出胜者的 fingerprint：相同 → 记 COMMIT_OBSERVED、返回胜者存储的
  receipt（rc=0）；不同 → REJECTED（rc=6）。这一层的竞态**没有被消除，也不需要消除**——
  demo 的 inv-5 实际记录的事件链是 `PREPARED×2 + COMMITTED(method=direct,
  prov-receipt-5) + COMMIT_OBSERVED(prov-receipt-5)`（runtime.db events seq 11–14，
  本教程在自己的跑后 CWD 现场复核过）：恰一次直接提交、恰一次观察纠偏，竞态的审计痕迹
  完整保留。

两层不可混淆：provider 层保证「每个 worker 拿到同一个真 receipt」（原子幂等），commits 层
保证「表里只存一条 commit 记录」（PK + 观察纠偏）。若没有 UNIQUE 约束而只有「先查后插」，
两个进程会同时查到空、同时插入——**检查-然后-行动（check-then-act）在并发下不是原子的，
唯一约束才是**；而要让「查」本身原子，写锁必须前置于检查之前（`BEGIN IMMEDIATE`）。这是
L0 的单线程世界完全看不到的工程难点，也是 L1 必须真并发跑一遍的原因。

### 6.1 反例：一行笔误如何藏进多次全绿运行（输出全绿 ≠ 机制正确）

上面「后到者原样返回同一 receipt」这句，最初并不真。第一版代码把幂等快路径的返回值写成
`print(row[0][1])`：`row = (fingerprint, receipt)`，而 `row[0][1]` 是**指纹字符串的第二
字符**，不是 receipt（正确写法是 `row[1]`）。也就是说，第二次 provider apply 返回的是垃圾，
两个 worker 手里的 receipt 实际一真一假、并不相同。

为什么常规端到端测试可能一直不红？因为存在一条**掩盖链**：先 commit 的 provider 先把真
receipt 返回给自己的 worker →
该 worker 因果居先、先赢 commits 竞态 → 持垃圾的 worker 恒走败者路径，COMMIT_OBSERVED
输出胜者存储的真 receipt → s5 恒 PASS。掩盖是**因果稳健**的，但**不是逻辑必然**的：若调度
抖动让持垃圾者赢了 commits 竞态，commits 表将存进垃圾字符，s5 会非确定性变红（0/22 观测，
概率非零）。而且无论 demo 红不红，provider 的幂等契约在直接调用下已经破损——任何绕过
runtime commits 门口、直读 provider 响应的重放调用者都会拿到垃圾。稳定的全绿输出只能说明
观测到的调度路径收敛，不能替代对 provider 幂等返回值契约的直接验证。

击穿它不需要并发，只需要一个**契约级探针**：
在一个全新空目录里直接调用 provider 自己的接口两次——直调绕过了 runtime 的授权门口
（provider 是外部世界，授权是 runtime 的闸），探针打的正是 provider 层自己的返回值契约：

```bash
python3 -B L1_durable_tool_runtime.py provider apply demo-key abcdef1234567890 research vendor 500 none
# → prov-receipt-1 (rc=0)
python3 -B L1_durable_tool_runtime.py provider apply demo-key abcdef1234567890 research vendor 500 none
# → 修复前此位输出 'b'（指纹 "abcdef1234567890" 的第二字符，rc=0）；修复后输出 prov-receipt-1 (rc=0)
```

一击击穿。修复（`row[0][1]`→`row[1]`，并把 `BEGIN IMMEDIATE` 前置于检查之前）后本教程
复跑自证：判决性探针第二次 apply 返回同一 receipt；并发探针 12 轮 × 同 key+同指纹双 apply
并发 0 mismatch，receipts 恰 12 行、余额账恰一次闭合。

这个反例值得带走三条教训：① **输出全绿 ≠ 机制正确**——self-check 只覆盖可观察输出，
掩盖路径上的 bug 要靠契约级探针（直调该层接口、核其返回值契约）才会现形；② **因果稳健 ≠
逻辑必然**——「观察上恒 PASS」与「不可能失败」之间隔着一层调度抖动；③ 上一节的
check-then-act 教训正是本 bug 的注脚——第一版快路径的检查在锁外（先查后锁），真正兜底的
是 receipts 的 PK；修复后写锁前置，残留的 TOCTOU 竞态路径（先查后锁之间对方 commit →
provider IntegrityError → worker UNCERTAIN → recover query 采纳）**从根消除：
IntegrityError 路径不再存在**。

---

## 7. 持久化的 payload 绑定与 default-deny

`[6]` 把 L0 的两条安全边界搬到持久层：

- 用已提交 key `inv-1` 携带 `amount=26`（原为 25）再来：worker 在 **commits 门口**就比对
  fingerprint，不同 → REJECTED 事件落盘、rc=6，provider 根本不会被调用。即使穿透 runtime，
  provider 侧 `receipts` 的 fingerprint 比对是第二道闸（`provider apply` 直接返回
  `FINGERPRINT_MISMATCH`）。纵深两层，任何一层失守另一层仍在。
- `source=attacker` 的 intent：authorize 拒绝（source 不在 principal scope），REJECTED 落盘、
  rc=5。授权对象 `AUTHORITY` 是源码里的可信常量，从不解析模型/工具输出——L0 的
  prompt-injection 结论在 L1 原样继承：**权限来自可信控制面，磁盘上也一样**。

`provider receipts unchanged=5` 是这两次拒绝的副作用账证明：拒绝不产生外部效果。

---

## 8. 权威实现对照：三种「持久执行」的取舍

durable agent runtime 不是新发明——工作流引擎十年前就在解同一题。三家权威给出三种答案：

| 维度 | LangGraph checkpointer | Temporal | nano L1 |
|------|------------------------|----------|---------|
| 持久的是什么 | **图状态快照**（per thread/superstep 的 checkpoint） | **事件历史**（Event History，每步一条） | **协议状态**（PREPARED/UNCERTAIN/COMMITTED + commits 表） |
| 恢复靠什么 | 读回最近 checkpoint 继续执行 | 确定性重放（replay）事件历史 | 扫描孤儿 key → query/replay provider |
| 副作用恰一次谁负责 | **工具自己**（checkpoint 不 dedupe 节点内的副作用） | **Activity 自己**（框架保证 at-least-once 投递） | **协议双方**（runtime 幂等键 + provider 幂等账本） |
| 崩溃模型 | 进程重启后从 checkpoint 续跑 | worker 崩溃 → 新 worker 重放历史 | 任意进程 SIGKILL → recover 新进程 |

一手来源与取舍分析：

- **LangGraph**（[langgraph-checkpoint-sqlite 3.1.1 项目页](https://pypi.org/project/langgraph-checkpoint-sqlite/3.1.1/)，
  2026-08-31 核验）：项目页将它定义为 SQLite-backed checkpoint saver，支持同步与异步接口；
  示例使用 `SqliteSaver.from_conn_string(...)`，checkpoint 结构含
  `channel_values/channel_versions/versions_seen`——即**状态快照**语义：
  恢复 = 回到最近的快照继续走。它的 README 同时给了一个安全注脚：要求设置
  `LANGGRAPH_STRICT_MSGPACK=true` 限制 checkpoint 反序列化的类型白名单——**持久化状态本身就是
  攻击面**（被篡改的库 = 被注入的图状态），这与 L1 的 hash chain 动机同构。取舍点：checkpoint
  模型对「节点内部已发生的外部副作用」无感知——节点执行到一半崩溃，重跑节点会重做副作用，
  所以工具幂等仍是使用方的义务。这里是由 checkpoint 边界推出的**教学推断**，并非项目页
  对任意工具副作用作出的保证；本教程也不对其 SQLite PRAGMA 等源码细节下结论。
- **Temporal**（[官方文档](https://docs.temporal.io/) 与
  [Retry Policies](https://docs.temporal.io/encyclopedia/retry-policies)，2026-08-31 核验）：
  Workflow 依靠 Event History 在故障后恢复进度；Activity 可能因失败而重试，因此外部副作用仍需
  幂等或业务去重。取舍点：Temporal 把「工作流重放安全」揽进框架（事件历史 +
  确定性约束），代价是编程模型约束（workflow 代码必须确定性）；L1 不要求 provider 确定性，
  只要求 provider 幂等可查——约束更弱、适用更杂的真实工具（邮件、支付、部署）。
- **nano L1 的选择**：既不快照图状态，也不重放执行历史，而是把「副作用协议」本身持久化：
  PREPARED 是 durable 的意图，commits 是 durable 的事实，中间的不确定性由 query/replay 收敛。
  这更接近支付/库存类系统的 outbox+幂等键传统（L2 展开），也最贴合 agent 场景的现实——
  工具五花八门，多数只提供「带 key 重试」和「按 key 查询」这两点微弱保证。

---

## 9. 费曼自检

**类比**：快递员把货到付款的包裹放在门口，还没拍照留证就被车撞走了（SIGKILL）。新来的同事
接手（recover 新进程），他不能直接再送一遍（可能收两次款），也不能当没发生（客户可能已付）。
他先查台账（query provider）：有签收记录 → 补一张凭证归档；没有签收记录 → 重新送一次
（replay，因为单号没变，仓库只会出库一次）；如果这单走的是个不留记录的老物流，那就只能
挂起等人工（NEEDS_HUMAN）。台账本身写在防撕毁的账本上（WAL）：写到一半人倒了，那半页
自动作废；写了「已确认」三个字之后人倒了，这三个字不会丢。

自检问：

1. 为什么「恢复」必须是一个新进程，而不是原进程里的 try/except？
2. `[3]` 和 `[4]` 的 worker 都报 UNCERTAIN，为什么恢复动作一个是 replay 一个是 query？
3. 若把 `commits` 表的 UNIQUE 约束去掉、改成「先查后插」，`[5]` 会发生什么？
4. `synchronous=NORMAL` 在什么场景下可以接受？在副作用 runtime 里为什么不行？
5. LangGraph 的 checkpoint 能替你保证工具副作用恰一次吗？谁来做这件事？

---

## 10. 思考题

1. `[2]` 中 worker 死于拿到 receipt 之后——receipt 在内存里随进程消失。恢复为什么仍能拿到
   正确的 receipt？若 provider 也不存 receipt（只存余额变动），恢复还能成功吗？
2. event log 里 `[5]` 的 key 有 4 条事件（两个 PREPARED + COMMITTED + COMMIT_OBSERVED），
   而 commits 表只有 1 行。若审计要求「每个 key 的事件链线性无分叉」，你会怎么改 schema？
   代价是什么？
3. provider 的 `before-commit` kill 里，未提交帧留在 `-wal` 里等恢复丢弃。若 kill 发生在
   `conn.commit()` 系统调用**内部**（内核写了一半），结果会不同吗？（提示：commit record
   的原子性由什么保证？）
4. 把 AUTHORITY 的 `max_amount` 从 30 改成 20，哪些场景的行为会变？这个改动需要迁移
   durable 状态吗？policy 版本化该怎么做？
5. Temporal 要求 workflow 代码确定性，L1 只要求 provider 幂等可查。构造一个「provider 幂等
   但不可查询」的现实例子，说明 L1 协议会把它归入哪一类、为什么。

---

## 11. 反例与边界（toy 尺度诚实声明）

- **单机假设**：SQLite WAL 的跨进程协调依赖同一文件系统与 POSIX 锁；跨主机（NFS/对象存储）
  不成立，真实多机需要换存储或加协调服务。`[TODO: verify on real system]`
- **掉电 ≠ SIGKILL**：SIGKILL 杀进程，磁盘还在；掉电考验的是 fsync 语义与硬件缓存。
  `synchronous=FULL` 在文档层面覆盖掉电，但本教程未做掉电测试（无硬件注入手段）。
- **余额账是 toy**：真实支付有 pending/settled/reversed 状态机、货币/汇率/租户维度；
  L1 的 COMMITTED 只表示「provider 已接受并给了持久 receipt」，不等于业务终态（继承 L0 口径）。
- **并发规模是 2**：只演示竞态的分类与收拢，不测吞吐；SQLite 单写者模型在高并发写下会成为
  瓶颈（这正是 L2 多 worker lease / 分片要回答的问题）。
- **provider 协议是 argv/stdout**：真实 provider 是 HTTP/RPC，传输层会带来超时、半开连接等
  新的不确定性窗口——但恢复侧的决策树（query/replay/needs_human）不变，这是协议与传输解耦
  的好处。
- **时间维度只进了一半**：worker 对 provider 调用设了 30 秒 timeout，timeout 按 UNCERTAIN
  处理（rc=3）——慢 provider 与被 kill 的 provider 在 worker 视角不可区分（代码 L191 注释
  口径），同样进恢复决策树；recover 侧相反，timeout = **fail-loud by design**：
  TimeoutExpired 直接传播、recover 非零退出——receipt 到手前本地零写入，intent 留孤儿，
  重跑 recover（幂等）再来（代码 L248–250 注释口径）。仍缺的是 backoff/lease：PREPARED
  孤儿多久算「该恢复」、谁来接管？真实系统需要时钟与租约，归 L2。

---

## 12. 阶梯预告

- **L2**（README 阶梯行）：多 worker lease（恢复不再靠人工触发，而是抢锁接管）、outbox/inbox
  模式（把「调用 provider」也变成 durable 消息）、timeout/backoff、compensation 的持久化
  编排、principal/session/purpose 完整 token binding。
- **L3**：对照真实 agent/tool runtime 与事务/outbox 模式的权威实现，做权限、重放、注入与
  审计的完整演练。

---

## 13. 溯源与口径声明

**复现锚点（2026-08-31 复验）**：代码 `L1_durable_tool_runtime.py` md5
`8bf0cf3c5e8b27df890fe0fade2a3844`/534 行/28,806 B。两个新建空 CWD 均以 `-B` 运行，
全部 EXIT=0、stderr 0 B，stdout BYTE-IDENTICAL；输出 md5
`66329d15509e11519793a648a53088aa`/79 行/4,719 B，digest
`aa1b4f2517ed5f0b48154028e939f8f7`，跨级锚 `c2e95d2a0c5b0809`（由代码
`cross_level_digest()` 对 L0 机制锚计算）。输出不含计时行，无需掩码。§6.1 记录的 bug
解释了为什么稳定输出不能单独证明 provider 契约正确；判决性证据是直接重放探针返回相同 receipt。

**一手来源（2026-08-31 重新核验；事实与教学推断分开）**：

- [SQLite WAL 官方文档](https://sqlite.org/wal.html)：commit record、恢复、FULL/NORMAL 与
  单 writer/并发 reader 的事实。
- [langgraph-checkpoint-sqlite 3.1.1 项目页](https://pypi.org/project/langgraph-checkpoint-sqlite/3.1.1/)：
  版本、SQLite checkpointer 定位、用法与安全提示；工具副作用边界是本文推断。
- [Temporal 官方文档](https://docs.temporal.io/) 与
  [Retry Policies](https://docs.temporal.io/encyclopedia/retry-policies)：故障恢复与 Activity 重试边界。

**口径**：全部数字为本脚本实测（余额账、receipt 计数、rc、-wal 存在性、integrity_check）；
权威实现侧只引一手文档逐字句与元数据，未核验处显式标注。L0 材料零改动（锚复验在案）。
