# nano-data-platform L1 — 增量接入 + 持久化 catalog + 重放对账（SQLite catalog，纯标准库）

> **前置**：`tutorial_L0.md`（L0 的分层契约与漏斗语义是本级的 K 层）。Python 3.10+，纯标准库（`sqlite3` 内置），CPU 秒级。
> **运行**：`python3 L1_incremental_sync_catalog.py`（任意目录可跑，输出确定，复跑逐字节一致）。
> **本文件是 notebook-style 教程**：叙述 + 代码摘录 + 真实运行输出 + 思考题交替推进。

---

## §1 K+1：L0 留下了哪两笔债

L0 把湖仓分层与声明式状态裸了出来，但所有状态都活在内存里。这留下两笔债，正是生产环境 day-1 就会撞上的：

1. **进程一重启，世界蒸发。** watermark（读到哪了）、血缘（哪批数据从哪来）、快照版本全在 Python 对象里。重启之后接入器不知道上次读到哪，只能从头全量重拉——数据量上来之后这就是灾难。
2. **每次 build 都全量重算。** L0 的 `CuratedZone.build` 把**所有** raw batch 重放一遍。raw 是只增不减的，build 成本随历史单调上涨，而每次真正的新信息只有最新那批。

L1 用四个机制面还清这两笔债（L1 验收标准，README 阶梯表）：

| # | 机制面 | nano 实现 | 真实系统对应 |
|---|--------|-----------|--------------|
| [1] | watermark/cursor 增量接入 | `sync_source`（游标之后才拉 + 单事务推游标） | Airbyte/Fivetran 类连接器的 incremental sync |
| [2] | 持久化 catalog（SQLite 级） | `open_catalog` 五张表 + `catalog_digest` | Hive Metastore / Glue Data Catalog / Iceberg catalog |
| [3] | 增量物化（merge/upsert 语义） | `build_version`（上一版快照 + 仅新 batch 折叠） | dbt incremental model（`unique_key`/merge） |
| [4] | 双层对账（reconciliation） | `full_replay` + `reconcile_source_to_raw` | 全量回测对账 / 周期性 full refresh |

刻意不模拟的（§11 列边界）：并发 commit 冲突（L2 对照 Iceberg commit protocol）、schema evolution（L2）、真实列存与分布式 catalog。

---

## §2 先跑一遍：完整输出

```bash
$ python3 L1_incremental_sync_catalog.py
```

```text
== nano-data-platform L1: 增量接入 + 持久化 catalog + 重放对账（SQLite catalog，纯标准库） ==

[1] 持久化 catalog（SQLite）：watermark/血缘/快照全部落盘（路径不入输出，复跑逐字节一致）
  5 张表: sources / batches / raw_records / curated_versions / curated_records
  声明源: crm (cursor=lsn) / web_log (cursor=lsn) —— 游标 = 增量接入的『读到哪了』

[2] 首次同步 = 全量回灌（watermark=0 → 拉全部）+ v1 物化（复现 L0 漏斗）
  denied as expected: sync denied: source 'crm' 凭据错误 —— 认证是接入边界第一道闸
  [check 01] PASS  错误凭据必须被拒
  b001: 拉取 9 events (lsn 1–9)，watermark 0 → 9
  [check 02] PASS  首次同步 = 全量回灌 9 events
  gate 拦截: 重复 id: t003
  gate 拦截: 必填字段缺失/为空: t006
  gate 拦截: 必填字段缺失/为空: t008
  v1: raw 9 条 → curated 6 条 (sha256=4599c15439c026c8)
  [check 03] PASS  v1 漏斗 9→6，拦截 3 条（复现 L0 语义）
  [check 04] PASS  跨级锚: v1 digest == L0 v1 4599c15439c026c8
  [check 05] PASS  内层对账: v1 增量物化 == raw 全量重放

[3] 幂等：同一游标重拉 = no-op（源端重试安全）
  re-sync → 0 new events / 新增 batch 0 个（watermark 未动）
  [check 06] PASS  重拉幂等: 0 new、batch 数不变

[4] 进程重启：close → reopen，catalog 从盘上原样重建（L0 内存态做不到的事）
  reopen: crm watermark=9, v1 = 6 条 (sha256=4599c15439c026c8) —— 全部从盘上读回
  [check 07] PASS  重启后 watermark 与快照幸存
  [check 08] PASS  重启前后 catalog digest 不变

[5] 新源 web_log 首同 + 增量物化 v2（跨级锚 + 内层对账）
  b002: 3 events（web_log 首同 = 全量，其 watermark 从 0 起）
  v2: curated 8 条 (sha256=a12337250f5d4d79)，本轮新增拦截: ['重复 id: t007']
  [check 09] PASS  v2 = 8 条，新增拦截恰 1 条跨源重复
  [check 10] PASS  跨级锚: v2 digest == L0 v2 a12337250f5d4d79
  [check 11] PASS  内层对账: v2 增量 == 全量重放

[6] 外层对账：源全量导出 vs raw 账目（接入完备性的保险丝）
  crm: 源侧真相 8 条 vs raw 账目 → 分歧 0 条
  [check 12] PASS  crm 源↔raw 对账: 0 分歧
  web_log: 源侧真相 3 条 vs raw 账目 → 分歧 0 条
  [check 13] PASS  web_log 源↔raw 对账: 0 分歧

[7] wave3：update（回填时间戳）+ insert → lsn 游标增量拉取 → v3
  b003: 增量拉取 2 events (lsn [10, 11]) —— 只拉游标之后，不重扫世界
  [check 14] PASS  lsn 游标增量: 恰 2 events (lsn 10, 11)
  v3: curated 9 条 (sha256=8e60d023ac8e576d)；t002 原位更新 → 'charged twice on my card, order #4217'（位置 1 不变）
  [check 15] PASS  v3 = 9 条，t002 原位覆写且位置不变（merge 语义）
  [check 16] PASS  内层对账: v3 增量 == 全量重放（update 语义双路径一致）
  [check 17] PASS  PII 投影: 全部版本 curated 无 phone

[8] 血缘变 SQL：t007 从哪来？v3 由哪些 batch 派生？
  t007 ← crm lsn8 (b001)
  t007 ← web_log lsn3 (b002)
  [check 18] PASS  跨源重复血缘: t007 恰 2 条 raw 记录（crm + web_log）
  v3.from_batches = ['b001', 'b002', 'b003']（源: ['crm', 'web_log']）
  [check 19] PASS  v3 派生自全部 3 个 batch / 2 个源

[9] 成本视角：增量物化 vs 全量重算（toy 数字，讲机制不讲规模）
  全量重算扫 14 events / 1370 B；v3 增量只扫 2 events / 209 B（15.3% 字节）
  [check 20] PASS  增量扫描字节 < 全量扫描字节

[10] 反例探针：updated_at 游标的盲区 + 对账捕获 + 回补自愈（独立 scratch catalog）
  updated_at 游标 (wm=90): 只拉到 lsn [11] —— 回填 update lsn10（updated_at=20）被静默漏掉
  [check 21] PASS  盲区复现: updated_at 游标漏掉回填 update
  [check 22] PASS  内层对账仍绿——raw 自己就缺事件，重放当然一致（内层对账的边界）
  外层对账: 源全量导出 vs raw → 分歧 ['t002']（源侧已修正的 t002，raw 里还是旧的）
  [check 23] PASS  外层对账捕获分歧: 恰 t002
  回补自愈: 全量重拉补进 1 event (b003) → 重建 v3（7 条，t002 已修正）→ 双层对账全绿
  [check 24] PASS  自愈后: 分歧清零且内外对账一致

catalog digest: 1ad07870b421fcf3  (5 张表的逻辑哈希；SQLite 文件字节不参与，输出确定)
  [check 25] PASS  catalog 已落盘且非空

self-check: 25/25 PASS
```

demo 剧本：建 catalog → crm 首同（全量回灌）物化 v1 → 幂等重拉 → **进程重启** → web_log 首同 + 增量物化 v2 → 双层对账 → wave3 增量（update + insert）物化 v3 → 血缘 SQL → 成本对比 → 反例探针（updated_at 盲区 → 对账捕获 → 回补自愈）。

> **fixture 声明（本地样本，跨级锚设计）**：L1 的接入样本与 L0 的 BATCH1/2/3 **逐字相同**（客服工单，phone 字段只使用不可拨号的占位符 `PHONE-DEMO-XX`，刻意埋缺陷）。这不是偷懒：README 对 L1 的要求是「复现 L0 的漏斗语义」，而「复现」的最强证明是**字节级**的——同款样本 + 同款折叠规则下，L1 增量物化出的 v1/v2 与 L0 全量重算的 v1/v2 digest 逐位相等（check 04/10，§6）。增量接入、持久化、对账机制与数据内容无关，wave3 的新事件（update/insert）则专门覆盖 L0 没有的语义。

---

## §3 机制面 [2]：持久化 catalog —— 把账本从脑子里搬进保险柜

L0 的「账本」全在内存对象里；L1 把它落进一个 SQLite 文件，五张表：

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
  source TEXT PRIMARY KEY, cursor_kind TEXT NOT NULL,
  credential_name TEXT NOT NULL, watermark INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS batches(
  batch_id TEXT PRIMARY KEY, source TEXT NOT NULL, n INTEGER NOT NULL,
  bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
  lsn_lo INTEGER NOT NULL, lsn_hi INTEGER NOT NULL, ingested_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS raw_records(
  source TEXT NOT NULL, lsn INTEGER NOT NULL, op TEXT NOT NULL,
  rid TEXT NOT NULL, payload TEXT NOT NULL, batch_id TEXT NOT NULL,
  PRIMARY KEY(source, lsn));
...
"""
```

五张表各管一摊：`sources` = 接入器状态（**watermark 在这里**）；`batches` = 批次血缘（源、条数、字节、内容哈希、lsn 区间）；`raw_records` = raw zone 本体（PK = `(source, lsn)`，§4 会用到）；`curated_versions` / `curated_records` = 版本化快照（L0 `CuratedZone.versions` 的落盘版）。

**为什么 catalog 是独立一层，而不是「数据库里的一张表」？** 因为湖仓的本质结构是**存储与元数据分离**：数据文件躺在便宜的对象存储上，「这张表在哪、有哪些快照、每个快照含哪些文件」由 catalog 管理。Iceberg table spec（页面标题 "Spec - Apache Iceberg™"，https://iceberg.apache.org/spec/ ，2026-08-13 重抓 883,921 B，与 L0 2026-08-12 录值零漂移）对 catalog 的定位（逐字引文）：

> "The table's location may be fixed in table metadata or inferred, but is intended to be managed and supplied by a catalog."

真实世界的 Hive Metastore / AWS Glue Data Catalog / Iceberg REST catalog 都是这一层的具象（云厂商实现作参照不锁定）。nano 版用 SQLite 是因为它恰好满足 catalog 的三个本质要求：**嵌入式**（无服务进程，`import sqlite3` 即用）、**事务性**（§4 的单事务推游标靠它）、**SQL 查询**（§8 的血缘查询）。DuckDB 也常在这个位置出现（README 阶梯表的「SQLite/DuckDB 级」）。

两个细节值得盯住：

**（a）连逻辑时钟都从 catalog 重建。** L0 用 `Clock.tick()` 保证输出确定，但钟在内存里，重启归零。L1 的 `open_catalog` 打开文件时执行 `Clock.rebase(MAX(ingested_at), MAX(built_at))`——**编号从账本里接着往下走**。确定性不依赖进程存活，这才是「状态落盘」的彻底形态。

**（b）catalog digest 是逻辑哈希，不是文件哈希。** SQLite 文件的字节布局（页分配、空闲链、头计数）不确定，同一逻辑内容两次写盘字节可以不同。所以 `catalog_digest` 对五张表的**行内容**做规范化 JSON 再哈希（check 08 用它证明重启前后内容零漂移）。真实系统同理：Iceberg 的 metadata 一致性靠 JSON metadata 内容与快照哈希，不靠底层文件字节 `[TODO: verify L2 源码锚点]`。

**思考题 3.1**：如果把 watermark 存在接入器的本地配置文件里（而不是 catalog 里），换一台机器接管接入会发生什么？（要点：接入器状态与平台状态分离 = 单点故障 + 不可审计。catalog 的另一个价值是**状态集中可查**——运维一条 SQL 就能看到所有源读到哪了。）

---

## §4 机制面 [1]：watermark 增量接入 —— 「读到哪了」是接入器的全部记忆

Airbyte 官方文档对 cursor 的定义（https://docs.airbyte.com/platform/using-airbyte/core-concepts/sync-modes/incremental-append ，2026-08-13 抓取，53,491 B，标题 "Incremental Sync - Append | Airbyte Docs"，逐字引文）：

> "A cursor is the value used to track whether a record should be replicated in an incremental sync." …… "A common example of a cursor would be a timestamp from an updated_at column in a database table."

nano 版把源系统建模为 **append-only 变更日志**（lsn 单调递增，CDC 的本质形态），接入器只干一件事——拉游标之后的事件：

```python
def sync_source(con, source, src, secrets, cursor_kind=None, force_wm=None):
    kind0, cred, wm0 = con.execute(
        "SELECT cursor_kind, credential_name, watermark FROM sources WHERE source=?", (source,)).fetchone()
    ...
    if secrets.get(cred) != src.token:
        raise PermissionError(f"sync denied: source '{source}' 凭据错误 —— 认证是接入边界第一道闸")
    events = src.changes_since(wm, kind)
    if not events:
        return 0, None  # 幂等：没有新变更 = no-op，源端重试安全
    ...
    with con:  # 单事务：raw 落盘 + batch 血缘 + watermark 推进，要么全成要么全不成
        for e in events:
            cur = con.execute(
                "INSERT OR IGNORE INTO raw_records VALUES(?,?,?,?,?,?)", ...)
            if cur.rowcount: inserted.append(e)
        ...
        con.execute("UPDATE sources SET watermark=? WHERE source=?", (max(new_wm, wm), source))
```

输出 §2 的 [2][3][4] 段演示了三个性质：

**（a）首次同步 = 全量回灌，而且不是特例。** watermark 初始为 0，「拉游标之后」自然就是拉全部——全量首同与增量后续是**同一条代码路径**（check 02）。真实连接器同理：初始加载（initial load / backfill）就是 watermark 为零的第一次增量。

**（b）幂等来自接口形状，不靠纪律。** 源端重试、网络超时重发，同一批事件再次到达时，`INSERT OR IGNORE` 撞上 PK `(source, lsn)` 直接 no-op（check 06：重拉 0 new、batch 数不变）。于是投递语义可以是 **at-least-once**（便宜、好实现），存储语义仍然是 **exactly-once**（PK 去重兜底）——这对组合是绝大多数生产接入器的真实形态。

**（c）数据落盘与游标推进在同一事务里。** 这是 exactly-once 物化的核心，拆开看两种失败：先推游标后落数据，进程死在中间 → 事件**永久丢失**（游标已经越过去了）；先落数据后推游标，进程死在中间 → 下次重拉 → PK 去重兜住，**无损**。所以顺序必须是「落数据 → 推游标」且原子化（`with con:` 单事务），配合（b）的去重，得到「最多重拉、绝不丢」的语义（check 07：重启后 watermark=9 与快照原样幸存）。

**思考题 4.1**：如果把 `UPDATE sources SET watermark` 挪到 `with con:` 块之外（事务外），上面哪种失败模式会回来？为什么 PK 去重救不了它？（要点：事务外 = 先落数据、事务提交、再单独写游标——两步之间崩溃就回到「丢数据」分支；PK 只能救「重拉」，救不了「游标已越过未落盘事件」。）

---

## §5 机制面 [3]：增量物化 —— 新版 = 旧版 + 新 batch，不重算世界

L0 的 build 全量重放所有 batch；L1 的 `build_version` 只折叠**上一版没见过**的 batch：

```python
def build_version(con, dataset):
    """增量物化：上一版快照状态 + 仅新 batch 的事件 → 新版（不重算世界）。"""
    prev = con.execute(
        "SELECT version, from_batches FROM curated_versions WHERE dataset=? ORDER BY built_at DESC LIMIT 1",
        (dataset,)).fetchone()
    known, state = set(), {}
    if prev:
        known = set(json.loads(prev[1]))
        for rid, payload in con.execute(
                "SELECT rid, payload FROM curated_records WHERE dataset=? AND version=? ORDER BY pos", ...):
            state[rid] = json.loads(payload)
    new_batches = [r[0] for r in con.execute(
        "SELECT batch_id FROM batches ORDER BY ingested_at, batch_id") if r[0] not in known]
    problems = fold_events(state, _batch_events(con, new_batches))
    return _persist_version(con, dataset, state, problems, known | set(new_batches))
```

关键在折叠函数 `fold_events` 的 **merge/upsert 语义**——这正是 dbt incremental model 的核心决策。dbt 官方文档（https://docs.getdbt.com/docs/build/incremental-models ，2026-08-13 抓取，120,752 B，逐字引文）：

> "Defining the optional `unique_key` parameter enables updating existing rows instead of just appending new rows." …… "If new information arrives for an existing `unique_key`, that new information can replace the current information instead of being appended to the table."

没有 merge 语义的增量模型会把「更新」当成「新增」追加进去——同一条工单在 curated 里出现两个版本，下游训练数据悄悄膨胀（这正是 dbt 的 `unique_key` 机制存在要解决的问题，引文见上；Airbyte 的 Incremental-Append 模式干脆声明不处理："In this flavor of incremental, records in the warehouse destination will never be deleted or mutated"，把去重推给下游 dbt——这是「接入层只做 append、转换层做 merge」的真实分工）。nano 版在物化层直接做 merge：

```python
def fold_events(state, events):
    """state = {rid: 投影后记录}（保持插入序）。
    insert: 重复 id / 必填缺失 → 拦截（L0 同款文案）；update: 原位覆写、位置不变（merge 语义），
    坏更新 = 逐出 curated（层 = 质量承诺，L0 §5 契约）。PII 投影在层边界（phone 不落 curated）。"""
    problems = []
    for op, r in events:
        rid = r.get("id", "?")
        missing = any(r.get(k) in (None, "") for k in REQUIRED)
        if op == "insert":
            if rid in state: problems.append(f"重复 id: {rid}")
            elif missing: problems.append(f"必填字段缺失/为空: {rid}")
            else: state[rid] = {k: v for k, v in r.items() if k != "phone"}
        else:  # update
            if rid not in state: problems.append(f"update 指向不存在的 id: {rid}")
            elif missing:
                problems.append(f"update 未过质量门: {rid}")
                del state[rid]
            else: state[rid] = {k: v for k, v in r.items() if k != "phone"}  # dict 覆写保持原插入位置
    return problems
```

wave3 的 update 事件（源侧修正 t002 的 text）演示了 merge 的三个细节（§2 [7] 段）：

1. **原位覆写、位置不变**（check 15：t002 仍在位置 1）。Python dict 覆写保持原插入位置，「更新不重排」自然涌现——下游消费者看到的行序稳定，diff 最小。
2. **update 也要过质量门**：坏更新（必填字段被改空）不是放行脏数据，而是把该行**逐出 curated**——L0 §5 的「层 = 质量承诺」契约在增量语义下依然成立。
3. **update 指向不存在的 id = 拦截**（乱序/孤儿更新），不静默插入——insert 与 update 的边界是明确的。

**思考题 5.1**：dbt 有 `--full-refresh` 逃生门（丢弃增量状态全量重建）。本实现里哪个函数是它的对应物？为什么「增量系统必须永远保留全量重建路径」？（要点：`full_replay`——增量状态一旦损坏/语义变更，全量重算是唯一的自愈手段；这也解释了为什么 raw 必须不可变保留，L0 §4 的可重建性在 L1 变成了运维逃生门。）

---

## §6 跨级锚：增量物化 == L0 全量重算（字节级）

L1 最硬的验收不是「跑通了」，而是**增量路径与 L0 全量路径产出字节级一致**：

```text
  [check 04] PASS  跨级锚: v1 digest == L0 v1 4599c15439c026c8
  [check 10] PASS  跨级锚: v2 digest == L0 v2 a12337250f5d4d79
```

两个 digest 的来源都可溯源：`4599c15439c026c8` 是 L0 tutorial §5 的录值（L0 运行时输出）；`a12337250f5d4d79` 由只读探针复算——import 冻结的 `L0_lakehouse_and_iac_state.py`（`main()` 有 `__main__` 守卫，import 无副作用），按 L0 代码同款口径（`json.dumps(kept, sort_keys=True, ensure_ascii=False)` → sha256 前 16 位）对 BATCH1+2+3 重算得到。L1 把这两个值写进 self-check，意味着：

- **漏斗语义逐字复现**：质量门（重复/空字段拦截、L0 同款文案）、PII 投影（phone 不落 curated）、去重（含跨源重复 t007）——insert-only 事件流下 `fold_events` 与 L0 `quality_gate` 逐条同行为；
- **增量没有引入任何语义漂移**：v1/v2 的 kept 集合、顺序、字段、规范化 JSON 与 L0 全量重算逐位相同。

这就是「复现 L0 的漏斗语义」的机器证明形态——不是教程里的一句「与 L0 一致」，而是跑一次就能证伪的断言。

---

## §7 机制面 [4]：双层对账 —— 增量是开环近似，对账是反馈回路

增量接入的本质是一笔交易：**用完备性换成本**——不再全量扫描，代价是「游标之后的世界」这个假设一旦不成立就会漏数据。让这笔交易安全的保险丝是**对账**。L1 有两层，各保一件事：

**内层：raw 全量重放 == 增量 curated 账目（物化正确性）。** `full_replay` 不读任何增量状态，从 raw 独立重放同一折叠规则，digest 与最新版比对（check 05/11/16 三次全绿，含 update 语义的 v3）。它回答：「我的增量折叠逻辑有没有算错？」注意它**永远可以执行**——前提正是 L0 §4 的 raw 不可变：只要 raw 在，参照系就在。

**外层：源全量导出 == raw 账目（接入完备性）。** `reconcile_source_to_raw` 拿源侧真相（`export_full`，日志按 lsn 折叠、后写覆盖）与 raw 折叠结果逐行比对（§2 [6] 段：两源 0 分歧）。它回答：「我有没有漏接/错接？」

两层的边界在 §8 的反例探针里被精确暴露：**内层对账抓不住接入层的漏**（check 22：raw 自己就缺事件，重放当然自洽）。这对应生产事故的常见形态——物化逻辑全绿、数据照样少，因为问题在更上游。对账必须分层做，每层只保自己那一段：外层保「源 → raw」，内层保「raw → curated」，合起来才是完整血缘链上的账目闭合。

**思考题 7.1**：外层对账全绿，数据就一定「对」吗？给一个反例。（要点：对账只保证 raw == 源，不保证源 == 世界——源系统自己错了（比如重复扣款的脏数据在源里就是错的），对账抓不住。质量门（§5 的 gate）与对账是两种正交的防线：gate 管「内容合不合格」，对账管「账目平不平」。）

---

## §8 反例探针：updated_at 游标的盲区、捕获与自愈

§2 [10] 段是全教程最重要的一段——它演示一个**看起来完全正确**的增量方案如何静默漏数据。

探针用独立 scratch catalog 重跑同一条 crm 流，但游标换成 `updated_at`（Airbyte 文档钦点的 "common example"）。wave1 首同后 watermark=90；wave3 源侧推来两个事件：

- lsn10：update t002（修正 text），但 `updated_at=20`——**回填时间戳**（离线修正任务重写该行时沿用了原始时间戳，或副本时钟回拨）；
- lsn11：insert t011，`updated_at=100`。

`updated_at` 游标拉「updated_at > 90」：只拿到 lsn11。回填修正 lsn10 **被静默漏掉**（check 21）——没有报错、没有警告，watermark 照常推进到 100，从此 lsn10 永远不会被拉到（它的 updated_at=20 已经永远落在游标后面）。而内层对账依旧全绿（check 22）——raw 里根本没有这个事件，重放当然自洽。

捕获靠外层对账：源侧全量导出里 t002 已是修正版，raw 里还是旧的 → 分歧 `['t002']`（check 23）。自愈靠回补：以 lsn 全量重拉（`force_wm=0`），`INSERT OR IGNORE` 自动去重已有的、补进缺的 lsn10，重建 v3，双层对账全绿（check 24）。注意自愈后游标采取**保守姿态**（宁可让后续重拉、PK 去重兜底，也不冒进）——对账-回补循环把「漏」从永久事故降级为**一个对账周期内的暂时偏差**。

**游标选型矩阵**（senior 判断力的核心，一句话一行）：

| 游标 | 源侧要求 | 盲区 | 典型形态 |
|------|----------|------|----------|
| 单调 lsn / LSN（变更日志位点） | 源侧维护 append-only log（有保留期！） | log 被截断/过期后的窗口 | CDC（Debezium 类）、本实现主路径 |
| `updated_at` 时间戳 | 只需一个索引列，最便宜 | 回填/时钟回拨/回写旧时间戳（§8 探针实测） | 查询型连接器（Airbyte 文档的 common example） |
| 周期性全量对账 | 源侧支持全量导出 | 无（成本就是代价） | 本实现 `reconcile_source_to_raw` + 回补 |

三者不是单选题：生产接入器常见组合是「便宜游标做日常 + 周期全量对账兜底」，或「CDC log 做日常 + log 过期时退化全量」。

**思考题 8.1**：delete 怎么捕获？两种游标都看不见删除（行只是消失了，没有事件）。用本实现的组件给出一个方案。（参考方向：`export_full` 的 id 集合 diff——外层对账天然就是删除检测机制；CDC 场景则靠 log 里的 delete 事件。想清楚「raw 里的已删行该不该物理删除」——提示：不该，raw 不可变，删除语义在物化层表达。）

**思考题 8.2**：回补为什么用「lsn 全量重拉 + INSERT OR IGNORE」而不是把 updated_at 游标重置为 0？两者都能补回 lsn10，工程上的差别是什么？（要点：前者对任何游标类型通用、且 PK 去重让重拉天然幂等；后者耦合具体游标语义，且若游标字段本身不可信 [回填就是它不可信的证据]，重置它也未必可靠。自愈机制要建立在比故障机制更可靠的假设上。）

---

## §9 血缘变 SQL + 成本视角

持久化 catalog 把 L0 里「遍历对象才能回答的问题」变成 SQL（§2 [8] 段）：

```sql
SELECT source, lsn, batch_id FROM raw_records WHERE rid='t007' ORDER BY source, lsn;
-- t007 ← crm lsn8 (b001)
-- t007 ← web_log lsn3 (b002)   ← 跨源重复的两条血缘都在案
```

「t007 从哪来」「v3 由哪些 batch 派生」（`curated_versions.from_batches`）——数据事故归因（「这批训练数据里怎么混进了重复样本？」）从翻代码变成一条查询。这是 catalog 作为**控制面**的日常价值。

成本视角（§2 [9] 段，toy 数字声明同 L0 §7：只讲机制、不可外推）：v3 的增量物化只扫 b003 的 2 events / 209 B，全量重算要扫 14 events / 1370 B——**15.3% 的字节**。toy 尺度差距已有 6 倍多；生产尺度下 raw 以 GB/TB 计而日增量以 MB 计时，这就是「没有人全量重算」的原因——也是「对账必须周期性而非每次做」的原因（对账的成本正是它全量扫描的那部分）。

---

## §10 费曼自检

**讲给外行听**：把数据平台想象成一家连锁书店的进销存。L0 时代，进货台账（raw）和净库存（curated）都记在会计脑子里（内存）——会计一换班（进程重启），所有账目蒸发，只能把供应商的货重新全点一遍。L1 做了四件事：其一，账本搬进保险柜（SQLite catalog），换班照样接得上（§3/§4 的 check 07/08）；其二，进货不再每次全点，只点「上次点到的货号之后的新货」（watermark——§4），重复送货单直接丢弃（PK 去重，§4b）；其三，库存卡不用每次从头誊抄，只在旧卡上追加新货、修正更正（增量物化 merge——§5）；其四，每月盘点两次：一次「按进货单重算库存卡」查誊抄错误（内层对账），一次「和供应商对账单」查漏进货（外层对账——§7）。探针演示了漏进货的经典桥段：供应商补送了一张**日期倒填**的送货单（updated_at 回填），「只收新日期」的收货员会静默漏掉它——唯有对账单能抓出来，然后按全量送货单补进（§8）。

**思考题汇总**（正文内另有 3.1 / 4.1 / 5.1 / 7.1 / 8.1 / 8.2）：

1. 一句话说清：「watermark 存在 catalog 里」与「watermark 存在接入器本地」的本质差别是什么？（要点：平台状态 vs 组件状态——前者集中、可审计、可接管；后者是散落的状态碎片，故障时无人知道全局读到哪了。）
2. 本实现里哪两个东西分别对应 Iceberg 的「snapshot 序列」与真实 catalog 的「table location 管理」？（`curated_versions` 的版本链 / `sources`+`batches` 的元数据层——L2 对照 Iceberg 源码展开。）
3. 如果把内外两层对账合并成一层（只比「源全量导出 vs curated」），会丢失什么信息？（要点：故障定位能力——两层对账能回答「漏在接入还是错在物化」，合并后只知道「账不平」。分层对账与分层存储是同一个思想：每层边界都是一个可独立验证的契约。）

**反例（一个常见错误直觉）**：「游标字段选对了，增量同步就永远不需要全量对账。」——§8 的探针就是反驳：`updated_at` 是 Airbyte 文档举的标准游标例子，「选得对」；但游标的正确性依赖一个**关于源侧写入模式的假设**（时间戳单调反映变更顺序），回填/时钟回拨/离线修正都会打破它，而且打破时没有任何报错。对账不是多余的双重保险，是增量这个开环近似的**反馈回路**——没有反馈的开环系统，误差无界。反过来，「每次都全量重算最安全」也不成立：§9 的数字说明成本随历史单调上涨，生产尺度下直接不可行。senior 的答案在中间：**增量为主干 + 对账为反馈 + 全量重建为逃生门**，三者缺一不可。

---

## §11 边界与下一站

**模拟了**（本教程验收内容）：watermark/cursor 增量接入（lsn 主路径 + updated_at 反例）；首同=全量的统一路径；at-least-once 投递 + PK 去重 = exactly-once 物化；单事务游标推进；持久化 catalog（重启幸存 + 逻辑时钟重建 + 血缘 SQL）；增量物化 merge/upsert 语义（位置保持、坏更新逐出、孤儿更新拦截）；跨级 digest 锚复现 L0 漏斗；双层对账（物化正确性 / 接入完备性）+ 回补自愈；增量 vs 全量的扫描成本对比。

**刻意没模拟**（每行都是更高阶梯的课题）：

| 没模拟 | 为什么 L1 不做 | 哪一级做 |
|--------|----------------|----------|
| 并发写入 / commit 冲突 | 需要乐观并发控制与重试协议 | L2 对照 Iceberg commit protocol（spec 载有此句，此处逐字引录："When a snapshot is created for a commit, it is optimistically assigned the next sequence number"——快照提交时乐观分配下一个 sequence number） |
| schema evolution | 需要 schema 版本协商 | L2（Iceberg schema evolution） |
| 真实列存（Parquet/ORC）与谓词下推 | 独立课题 | 超出本模块阶梯 |
| 分布式 catalog / state locking | 需要多节点环境 | 云厂商实现仅作参照（§九 of L0），不锁定 |
| 真实连接器生态（Airbyte/Fivetran 的 connector 协议细节） | 环境依赖重 | 概念与引文已对照，实操非本模块目标 |

**L2 预告**（README 阶梯表 L2 行）：对照 Apache Iceberg / Delta Lake 源码的 snapshot/manifest/commit protocol（乐观并发、time travel、schema evolution）+ dbt 的分层派生模型 + Terraform HCL/provider/state locking 实操（课程的数据系统教学约定：HCL 到 L2 才触及）——本级的 `curated_versions` / `catalog_digest` / 单事务提交，都会在 L2 找到它们的工业级对应物并做取舍分析。

---

## §12 溯源

| 声明 | 类型 | 来源 |
|------|------|------|
| cursor 定义两句引文 + append flavor「never be deleted or mutated」引文（§4/§5/§8） | 文献已有（逐字引文） | https://docs.airbyte.com/platform/using-airbyte/core-concepts/sync-modes/incremental-append ，2026-08-13 抓取（53,491 B，标题 "Incremental Sync - Append \| Airbyte Docs"） |
| dbt `unique_key` 两句引文（§5） | 文献已有（逐字引文） | https://docs.getdbt.com/docs/build/incremental-models ，2026-08-13 抓取（120,752 B） |
| Iceberg spec catalog 定位引文（§3）+ 快照乐观提交引文（§11） | 文献已有（逐字引文） | https://iceberg.apache.org/spec/ ，2026-08-13 重抓（883,921 B，标题 "Spec - Apache Iceberg™"，与 L0 2026-08-12 录值零漂移）；行号级源码锚点 `[TODO: verify L2 源码锚点]` |
| Hive Metastore / Glue Data Catalog / Iceberg REST catalog / DuckDB 为 catalog 层的真实对应（§3） | 纲领已有 + 合理推断 | 课程的实现参照与数据系统约定 参照表；概念性提及，未引数字 |
| CDC / Debezium 类、lsn/LSN 游标形态（§8 矩阵） | 合理推断（机制类别归纳） | 未引具体数字；Debezium 为概念性提及 |
| 「首同 = watermark 为零的第一次增量」（§4a）与「接入层 append、转换层 merge」分工（§5） | 合理推断（机制归纳） | 与 Airbyte incremental 文档行为一致，未引数字 |
| L0 v1 digest `4599c15439c026c8` | 文献已有（L0 录值） | `tutorial_L0.md` §5 输出块（L0 代码 `c5adc7d8…` 冻结件运行输出） |
| L0 v2 digest `a12337250f5d4d79` | 本仓只读探针复算 | import 冻结 `L0_lakehouse_and_iac_state.py`（`main()` 有守卫，import 无副作用），按 L0 代码 L65-71 同款口径重算 |
| 全部漏斗/扫描/digest 数字（9→6、12→8、14 events/1370 B、209 B/15.3%、`43567fee…`、`b69ebeb7…` 等） | 本实现实测（toy 设定） | `L1_incremental_sync_catalog.py` 本次运行输出（§2 paste 块与运行输出 BYTE-IDENTICAL），非真实云价、不可外推 |
| Fivetran 与 Airbyte 同类（增量同步连接器） | 纲领已有（课程的数据系统教学约定 关键词） | 概念性提及；其文档页为 JS 渲染、未获正文引文 `[TODO: verify]` |

**运行锚点**（2026-08-13，Python 3.13.13 / SQLite 3.53.1 实测）：代码 md5 `f3696d73d2b28458459d2a7b1625f802`/397 行；双跑 2 遍 × 新建空独立 CWD（`-B`）全 EXIT=0、stderr 0 B、stdout 73 行/4,495 B、md5 `b02aad91a525ad34d72168f46f916477`，RUN1==RUN2 BYTE-IDENTICAL；self-check 25/25 PASS。

下一站：**L2**——对照 Iceberg/Delta/dbt 权威源码的 snapshot/commit/schema evolution 取舍分析 + Terraform HCL/state 实操（见 README 阶梯表）。
