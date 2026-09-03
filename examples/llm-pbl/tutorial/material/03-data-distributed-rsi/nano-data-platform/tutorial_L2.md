# nano-data-platform L2 — Iceberg 式 commit protocol + schema evolution + Terraform HCL/state locking（可运行的本质模拟）

> **前置**：`tutorial_L0.md`（分层契约 + 声明式 plan/apply）与 `tutorial_L1.md`（watermark 增量 + 持久化 catalog + 对账）。Python 3.10+，纯标准库，CPU 秒级。
> **运行**：`python3 L2_commit_protocol_schema_evolution.py`（任意目录可跑，输出确定——逻辑时钟、无随机，复跑逐字节一致）。
> **本文件是 notebook-style 教程**：叙述 + 代码摘录 + 真实运行输出 + 思考题交替推进。
> **可运行性契约声明**：本文件是「可运行的本质模拟」——单进程脚本化交错模拟多 writer 并发（真实系统是多进程/集群上的真并发）；对象存储 = 本地 tempdir 内容寻址文件（真实 = S3/HDFS 类对象存储）；HCL 只解析教学子集（真实 Terraform 是 Go + HCL 全语法 + 远端 backend 锁）。模拟核心本身可运行、本节全部数字来自 §2 的真实运行输出；真多进程竞争与对象存储原子性必须在对应真实系统另行验证。

---

## §1 K+1：L1 留下了哪三笔债

L1 把账本搬进了保险柜（SQLite catalog），但「快照」还只是追加的版本号，提交是单线程的、schema 是固定的。这留下三笔债，正是多 writer 生产环境 day-1 就会撞上的：

1. **没有人并发提交。** L1 的 build 是独占的。两个接入器/两个训练数据构建任务同时物化同一张表，L1 没有任何机制保证「各恰一次、不丢不重」——这是湖仓格式存在的原始动机。
2. **schema 不能演化。** 加一列、改一个列名，在「按列名读文件」的世界里要重写所有历史数据文件——或者干脆禁止。而训练数据管线里 schema 天天在变（加标注列、改口径名）。
3. **元数据是一锅 SQLite 行 + 声明式配置停在玩具形态。** L0/L1 的 plan/apply 没有 state locking（两个 apply 并发会互相踩），也没有真实 HCL 语法；L2 才引入 HCL 教学子集和 state locking。

L2 抓四个机制面（README 阶梯表 L2 行），每一面都回答一个「为什么这样设计」：

| # | 机制面 | nano 实现 | 真实系统对应 |
|---|--------|-----------|--------------|
| [1] | 元数据树 + 内容寻址存储 | `ObjectStore`（按内容哈希命名、写后不可变、同内容去重） | Iceberg metadata → snapshot → manifest list → manifest → data file；S3/HDFS 对象存储 |
| [2] | commit protocol = 原子指针交换 + 乐观并发 | `Catalog.commit`（check-and-put）+ `commit_with_retry`（refresh + re-apply） | Iceberg `SnapshotProducer.commit()` 重试循环；Delta 日志原子版本 |
| [3] | schema evolution by field id + time travel | `SchemaOp` / `RollbackOp` + `Table.scan(snapshot_seq, at_schema)` | Iceberg schema evolution（列按 id 引用）、time travel / rollback |
| [4] | Terraform HCL + state locking | `parse_hcl`（真实 HCL 子集）+ `TfState`（serial/lineage/lock nonce） | Terraform plan/apply + backend 锁（DynamoDB/Consul/Terraform Cloud） |

刻意不模拟的（§14 列边界）：真实多进程竞争、对象存储上 put-if-absent/rename 的实际原子性、指数退避、partition spec、row-level deletes、branch/tag 全语义。

---

## §2 先跑一遍：完整输出

```bash
$ python3 L2_commit_protocol_schema_evolution.py
```

下方 paste 块来自真实运行输出；2026-08-31 在两个新建空 CWD 中以 `-B` 复验，stdout 逐字节一致（本码无 elapsed 行，无需掩码）：

```text
== nano-data-platform L2: Iceberg 式 commit protocol + schema evolution + Terraform HCL/locking（本质模拟） ==

[1] 元数据树 + 首次 commit：table metadata → snapshot → manifest list → manifest → data file（全内容寻址）
  gate 拦截: 重复 id: t003
  gate 拦截: 必填字段缺失/为空: t006
  gate 拦截: 必填字段缺失/为空: t008
  append 6 行 → snapshot seq1（metadata V1，1 attempt）
  元数据树: metadata cacd9b59… → snapshot seq1 → mlist e8f10515… → manifest → data file
  [check 01] PASS  跨级锚: seq1 内容 digest == L0/L1 v1 4599c15439c026c8
  [check 02] PASS  首次 commit 1 attempt（无冲突）

[2] 第二次 append：manifest 复用 = 快照便宜的本质（不重写世界）
  gate 拦截: 重复 id: t007
  seq2: 8 行 (digest a12337250f5d4d79)；新增字节 987 B（新 data file + 新 manifest + 新 mlist + 新 metadata）
  对比全量重写世界 2442 B —— 快照边际成本 40.4%
  [check 03] PASS  跨级锚: seq2 内容 digest == L0/L1 v2 a12337250f5d4d79
  [check 04] PASS  manifest 复用: seq2 的 mlist 含 seq1 的 manifest（引用复用，零重写）
  [check 05] PASS  快照边际字节 < 全量世界字节

[3] 乐观并发：A/B 同 base 并发 append —— 一个成功，另一个 CAS 冲突 → refresh + re-apply 重试
  writer-B 先提交 → seq3（1 attempt）
    commit 冲突（base c9f8b9ce… 过期）→ refresh + 重新 apply（attempt 1/4 已用）
  writer-A 第 1 次 attempt CAS 冲突 → 第 2 次 attempt 基于新 base 重做 → seq4
  attempt 2 重放 attempt 1 已写的 data/manifest 对象: 内容相同 → 去重命中 2 次（内容寻址把『重做』变『复用』）
  [check 06] PASS  B 无冲突 1 attempt / A 恰 2 attempts（1 冲突 + 1 重试）
  [check 07] PASS  双方数据各恰一次（无丢失无重复）
  [check 08] PASS  snapshot-log 严格递增（历史串行化）

[4] 不是所有更新都能 rebase：并发 schema 更新 —— add-column 成功，rename 验证失败被拒
  writer-A add column priority → s2（metadata V5，无新快照）
    commit 冲突（base dc51de41… 过期）→ refresh + 重新 apply（attempt 1/4 已用）
  writer-B rename(label→category) 基于 s1：base(s1) 与 current(s2) 之间 schema 已变：此更新不可 rebase，须基于新 schema 重新发起
  [check 09] PASS  基于过期 schema 的 rename 必须被拒
  writer-B 基于新 schema 重新发起 rename → s3（1 attempt）
  [check 10] PASS  最终 schema = id/text/category/priority（两次演化按提交序生效）
  [check 11] PASS  schema 更新不产生快照（快照数仍 4）

[5] schema evolution by field id：旧文件零重写，rename 纯元数据（机器证明）
  当前 schema 读旧文件: t001 = {'id': 't001', 'text': 'how to reset password', 'category': 'auth', 'priority': None}（label 已改名 category，priority 读出 null —— 列按 field id 解析）
  [check 12] PASS  rename 后旧数据按新名可读、新增列读 null
  [check 13] PASS  演化未重写任何数据文件（data 字节数不变）
  新写 t013（带 priority）→ seq5: {'id': 't013', 'text': 'password reset email in spam', 'category': 'auth', 'priority': 'high'}
  [check 14] PASS  新 schema 写入的新行携带 priority
  [check 15] PASS  写新行只增不改（旧 data 对象哈希集合不变）

[6] time travel + rollback：旧快照按当时 schema 可读；rollback 是一个普通 commit
  as-of seq2（schema s1）: 8 行，digest a12337250f5d4d79 —— 与 L0/L1 v2 逐字节同一
  [check 16] PASS  time travel 复现 L0/L1 v2 锚（快照隔离 + 历史可重放）
  [check 17] PASS  旧快照在当前 schema 下读出 category（field id 穿越）
  rollback → seq2（metadata V8，1 attempt）：current = 8 行
  [check 18] PASS  rollback 后当前视图 = seq2 的 8 行（按当时 schema s1 读复现 v2 锚）
  [check 19] PASS  rollback 不删历史：快照仍 5 个，seq5 仍可前向 time travel
  [check 20] PASS  snapshot-log 记录回滚轨迹

[7] Terraform：HCL 子集 → plan/apply + state locking（serial/lineage/nonce）
  解析 platform.hcl: [('dp_dataset', ['sft_support']), ('dp_grant', ['ingestor', 'trainer'])]
  [check 21] PASS  HCL 子集解析: 3 blocks、属性类型正确
  plan: create:dp_dataset            sft_support
  plan: create:dp_grant              ingestor
  plan: create:dp_grant              trainer
  [check 22] PASS  首次 apply = 3 actions，serial 0→1
  apply-P2 被拒: state 已被 'apply-P1-long' 锁定 (lock_id=lock-002) —— 拒绝而非等待损坏
  [check 23] PASS  持锁期间的并发 apply 必须被拒
  伪造 nonce 解锁被拒: release 拒绝：lock_id 'lock-999' 与当前锁 nonce 不符 —— 解锁只认 nonce
  [check 24] PASS  非持锁者不能解锁（nonce 校验）
  apply-P2 重试: 0 actions（幂等 no-op），serial 维持 1
  [check 25] PASS  解锁后重试 = 幂等 no-op、serial 不推进
  drift 检测（带外删掉 trainer 授权）: plan = [('create:dp_grant', 'trainer')] —— 最小 diff 修复，不重建世界
  [check 26] PASS  drift 修复 = 恰 1 个 action
  [check 27] PASS  授权落 state 后在消费边界执行: trainer 可读 / intern 被拒

[8] 成本与孤儿账本（toy 字节账，讲机制不讲规模）
  对象存储字节账: data 903 B / manifest 270 B / mlist 360 B / metadata 12610 B（去重命中 2 次，均来自 [3] 重试 re-put）
  metadata 对象 11 个，其中 2 个是冲突遗留孤儿（[3] writer-A 冲突 attempt + [4] rename 过期 base attempt；真实 Iceberg 由 cleanUncommitted 清理，SnapshotProducer.java:L524）
  [check 28] PASS  去重命中恰 2：内容寻址把冲突重试的『重做』变『复用』（同内容不重复计字节）
  [check 29] PASS  冲突孤儿恰 2：乐观构建的 metadata 未提交即弃（cleanUncommitted 的清理对象）

platform L2 state digest: dafdf0c9bfdecd5e  (catalog 指针 + snapshot-log + schema 序列 + tf state 的逻辑哈希)
  [check 30] PASS  state digest 非空且 catalog 提交数 = 8（成功 commit；失败的 rename 验证在 apply() 内抛错、不计 commit）

self-check: 30/30 PASS
```

demo 剧本：首次 append 建元数据树（[1]）→ 第二次 append 算快照边际字节账（[2]）→ A/B 并发 append 演示 CAS 冲突与重试（[3]）→ 并发 schema 更新演示「不可 rebase」的验证（[4]）→ field id 让 rename 零重写数据（[5]）→ time travel + rollback（[6]）→ HCL plan/apply + 锁与 nonce（[7]）→ 成本与孤儿账本（[8]）。

> **fixture 声明（跨级锚设计）**：BATCH1/2/3 与 L0/L1 **逐字相同**（客服工单，明显假号 `138-0000-00XX`，刻意埋缺陷），转换层 `staging`/`fold_events` 与 L1 逐字同款。这不是偷懒：L2 的验收之一是「阶梯间语义同一性的机器证明」——同 fixture 同折叠规则下，L2 快照内容 digest 与 L0/L1 字节级一致（check 01/03，§7）。L2 新增的 T011–T013 专门覆盖 L0/L1 没有的语义（并发、新列、priority 字段）。

---

## §3 机制面 [1]：内容寻址 —— 快照为什么便宜

L0/L1 的「快照」是一个版本号加一锅数据；L2 把元数据拆成一棵**不可变的、按内容哈希命名的树**：

```text
table metadata（catalog 指针唯一指向的可变入口）
 └─ snapshot seq N（不可变，内容寻址）
     └─ manifest list（本快照包含哪些 manifest）
         └─ manifest（一组 data file 的清单）
             └─ data file（行数据本体）
```

`ObjectStore.put` 是全部机制的地基，只有六行语义：

```python
def put(self, kind, obj):
    body, h = canon(obj), sha16(canon(obj))
    if h in self.paths:          # 同内容已存在：零成本复用（快照共享 manifest 的基础）
        self.dedup_hits += 1
        return h
    self.paths[h] = f"{self.root}/{kind}-{h}.json"
    with open(self.paths[h], "wb") as f: f.write(body)
    ...
```

对象名 = 内容哈希，带来三个性质，每一个都是后续机制的前提：

**（a）写后不可变。** 同内容必同名，同名必同内容——任何对象一旦写下就不需要再改。快照因此可以安全共享：seq2 的 manifest list 引用 seq1 的 manifest（check 04），没有人能「改坏」被共享的那份。

**（b）同内容自动去重。** [3] 段 writer-A 冲突重试时把同样的 data/manifest 对象再 put 一遍——内容相同 → 哈希相同 → 直接复用，去重命中 2 次（check 28）。**内容寻址把乐观并发的「重做」变成了「复用」**：重试的边际成本趋近于零，这是「冲突了大不了重来」敢于成立的账本基础。

**（c）快照的边际成本 = 新增的那几个对象，不是重写世界。** [2] 段的字节账是本级成本 first-class 的核心实测（toy 尺度，讲机制不讲规模）：第二次 append 只新增 987 B（新 data file + 新 manifest + 新 mlist + 新 metadata），而「全量重写世界」要 2442 B——**快照边际成本 40.4%**。toy 尺度下旧世界还很小（8 行），所以边际占比高达四成；生产尺度下表以 GB/TB 计而单次 append 的 manifest 增量以 KB/MB 计，这个比值会小到可以忽略——这正是「湖仓里快照可以随便打、time travel 默认全保留」的经济学根据。反过来，任何「打快照要复制全表」的系统（传统数据库的物理快照）成本模型完全不同，time travel 只能当奢侈品卖。

云厂商参照（不锁定）：这里的 `ObjectStore` 本质模拟 S3/HDFS 类对象存储的「按 key 存对象、对象不可变」语义，用本地 tempdir 承载——机制与云无关，真对象存储上的原子性差异见 §14 `[TODO: verify on real system]`。

**思考题 3.1**：如果把对象名从「内容哈希」改成「递增序号」（`data-001`、`data-002`），上面 (a)(b)(c) 三个性质各会坏掉哪个？（要点：(a) 坏——同名不再意味着同内容，共享变成可变别名，快照隔离崩塌；(b) 坏——重试重放无法识别「同内容」，去重消失，冲突重试开始真实消耗字节；(c) 部分坏——快照仍可引用旧对象，但「两个 writer 独立写出相同内容」不再收敛到同一对象，孤儿与冗余膨胀。）

---

## §4 机制面 [2]a：commit = 原子指针交换 + CAS —— 并发的全部秘密在一个指针上

整棵树里**唯一可变**的东西是 catalog 里的一个指针：`table → 当前 metadata 哈希`。commit 只做一件事——check-and-put：

```python
def commit(self, table, base, new):
    if self.pointer[table] != base:
        raise CommitConflictError(f"base {base[:8]}… 已不是当前版本（当前 {self.pointer[table][:8]}…）")
    self.pointer[table] = new
```

先验「我以为的 base 仍是当前」，再交换指针。这就是 CAS（compare-and-swap）语义：交换本身是原子的，并发性被压缩成「谁的 base 过期了」这一个二值判断——不存在「两个提交各写了一半」的中间态，因为除了指针没有任何可变状态。

Iceberg table spec（https://iceberg.apache.org/spec/ ，抓取件 `iceberg_spec.html` 883,921 B，2026-08-15 抓取）对提交语义的总纲（逐字引文）：

> "Updates verify the conditions under which they can be applied to a new version and retry if those conditions are met. Append operations have no requirements and can always be applied."

两句话各管一半：**前半句**是乐观并发——先构建、后验证、冲突重试；**后半句**是 append 的特权——没有任何前置条件，所以永远可以 rebase（§5 展开它的反面）。

[3] 段是这段机制的完整演示：writer-A 与 writer-B 读到同一个 base（`base0`，模拟并发读），B 先提交成功（1 attempt）；A 的第一次 attempt 撞上 `CommitConflictError`（base `c9f8b9ce…` 过期）→ refresh 拿到新 base → 重新 apply → 第二次 attempt 成功（check 06：B 恰 1 / A 恰 2）。结果 check 07：t011/t012 各恰一次，无丢失无重复；check 08：snapshot-log `[1,2,3,4]` 严格递增——**并发提交在历史上被串行化了**，每个快照都有唯一的序号与父链。

**为什么选乐观并发而不是悲观锁？** 因为湖仓的提交是「长构建、短交换」：一次 append 可能写几万个文件、花几分钟，而指针交换是毫秒级。悲观锁要求 writer 持锁几分钟——锁的持有期就是系统的串行瓶颈，且持锁者崩溃会留下死锁。乐观并发把冲突代价推迟到提交瞬间：绝大多数时候没有冲突（各写各的分区/批次），冲突了也只是重来一次（而 §3(b) 保证了重来的字节成本趋近零）。这个取舍在 nano 版里被刻意保留：`commit_with_retry` 的重试是逻辑步进（无 wall-clock 等待），真实 Iceberg 是指数退避（§10 行锚）。

**思考题 4.1**：如果把 check-and-put 改成「先交换指针、再验 base」（put-then-check），并发场景下会坏成什么样？（要点：两个过期 base 的提交会先后「成功」，历史分叉成两个都声称自己是 seq N 的快照——不是丢数据，是历史本身失去线性，任何 time travel/对账都失去参照系。CAS 的顺序不可颠倒：验证必须在交换生效之前。）

---

## §5 机制面 [2]b：不是所有更新都能 rebase —— schema 更新的验证规则

§4 引文的后半句说 append「永远可以应用」——那什么更新**不**可以？[4] 段演示了答案：writer-A 与 writer-B 都基于 schema s1 发起 schema 更新，A 的 add-column 先成功（s1→s2）；B 的 rename(label→category) 仍基于 s1，`SchemaOp.apply` 在构建时验证：

```python
if m["current-schema-id"] != self.base_schema_id:
    raise CommitValidationError(
        f"base(s{self.base_schema_id}) 与 current(s{m['current-schema-id']}) 之间 schema 已变：此更新不可 rebase，须基于新 schema 重新发起")
```

注意这个错误**不在重试循环里消化**——`CommitValidationError` 不是 `CommitConflictError`，重试循环不会 refresh 后重放它（check 09：基于过期 schema 的 rename 必须被拒）。B 必须**基于新 schema 重新发起**（`SchemaOp("writer-B", m5["current-schema-id"], "rename", ...)` → s3，1 attempt；check 10：最终 schema = id/text/category/priority，两次演化按提交序生效）。

为什么 append 可以无脑 rebase、schema 更新不行？看两者在「base 与 current 之间发生了别的提交」时的语义：

- **append** 只新增文件，不与任何已有结构交互——无论中间发生了什么，把新文件挂到最新 metadata 上语义不变。spec 同节（逐字引文）："Append operations have no requirements and can always be applied."
- **schema 更新**的语义依赖「我是基于哪个 schema 演化的」。B 的 rename 说「把 s1 的 field 3 改名」，若中间 A 已经加了一列（s1→s2），把 B 的更新直接 rebase 到 s2 上**可能**无害（改的还是 field 3），但也可能有害（若 A 删了/改了 field 3，B 的更新指向的东西已不存在或已变）。Iceberg 的选择是不做这种逐 case 的危害分析，一律验证拒绝。spec 同节（逐字引文，全句）：

> "Table schema updates and partition spec changes must validate that the schema has not changed between the base version and the current version."

（代码 docstring 引文以 "Table schema updates …" 省略了句中 "and partition spec changes"，省略口径在此声明；本节引全句。）

这是一个可以背下来的 senior 判断：**乐观并发的可重试性 = 更新语义对中间历史的无关性**。append 无关 → 永远可重试；schema/partition 更新有关 → 必须验证。同样的判据可以推广到你自己系统里的任何「乐观更新」设计：先问「这个操作 rebase 到别的 base 上语义还成立吗」，成立才配进重试循环，不成立就让它快速失败、由发起方重新决策。

两个附带观察（都在 [4] 段输出里）：schema 更新**不产生快照**（check 11：快照数仍 4，只有 metadata 版本推进 V4→V5→V6）——schema 是表的属性而非数据的状态；add-column 的冲突（base `dc51de41…` 过期）走的是普通 refresh-retry 路径（因为验证在 apply 内基于新 base 重做后通过了）——**验证失败与提交冲突是两条不同的错误路径**，前者拒绝重试，后者重试消化。

**思考题 5.1**：如果把 schema 更新也改成「无条件 rebase」（像 append 一样），给一个具体的提交交错序列，让 rename 产生错误结果。（提示：A 发起 rename(field 3→category) 基于 s1；中间 B drop 了 field 3 又 add 了一个新列恰好拿到 id 4……想清楚「按名字」与「按 id」两种引用下，无条件 rebase 各会错在哪。）

---

## §6 机制面 [3]：field id —— rename 为什么零重写数据

schema evolution 能「便宜」的全部秘密在一个设计决策：**列按 id 引用，不按名字**。数据文件里存的是 `{field_id: value}`，名字只活在 schema 元数据里：

```python
rows.append({f["name"]: row.get(str(f["id"])) for f in schema["fields"]})
```

读出的行按**当前 schema 的 field id** 去旧文件里取值，再把 id 映射成当前名字。于是三种演化都变成纯元数据操作：

- **add column**：新 schema 多一个 id，旧文件里没有这个 id → 读出 null（[5] 段：t001 的 `priority: None`）。不重写。
- **rename**：只改 schema 里 id→name 的映射，数据文件字节零触碰。**机器证明**：check 13（演化前后 data 字节数不变）+ check 15（写新行后旧 data 对象哈希集合不变，只增不改）。
- **drop**：schema 里删掉该 id（required 列拒删），旧文件里那个 id 的值还在、只是不再被读——历史快照按当时 schema 读仍完整（§6 下半 time travel 的前提）。

[5] 段的 t001 是这段机制的具象：`{'id': 't001', 'text': 'how to reset password', 'category': 'auth', 'priority': None}`——这是**写在 rename 之前**的旧文件，读出来却带着 rename 之后的名字 `category`（check 12），新列 priority 读出 null。写入端则相反：新行按当前 schema 列名书写（t013 直接写 `category` 与 `priority: "high"`，check 14），名称映射发生在写入边界（真实管线 = dbt 模型的 `SELECT label AS category`）——**读按 id 解析所以历史原样保留，写按名映射所以新数据对齐当前口径**，两边各管一半。

**time travel 与 rollback**（[6] 段）是「不可变 + 指针」设计的免费赠品：

- `scan(snapshot_seq=2, at_schema=1)` = as-of 读：旧快照按**当时** schema 读出 8 行，digest `a12337250f5d4d79` 与 L0/L1 v2 逐字节同一（check 16）——快照隔离让历史可精确重放。
- `scan(snapshot_seq=2)`（不传 at_schema）= 旧快照 × **当前** schema：seq2 的行读出 `category`（check 17）——field id 穿越历史，rename 对全部历史生效。
- **rollback 是一个普通 commit**：`RollbackOp` 只把 `current-snapshot-seq` 指回 seq2，不新建快照、不删历史、数据文件零触碰（metadata V7→V8）。权威实现同款语义——SnapshotProducer.java:L479 注释（抓取件逐字）："this is a rollback operation"。check 19：rollback 后快照仍 5 个，seq5 仍可前向 time travel；check 20：snapshot-log `[1,2,3,4,5,2]` 完整记录回滚轨迹——**「回滚」本身也是可审计的历史事件**，而不是把历史抹掉假装没发生。

**思考题 6.1**：如果列按名字引用（数据文件存 `{name: value}`），rename 的成本模型变成什么？add column 对旧文件的影响呢？（要点：rename 必须重写所有历史数据文件——成本从 O(元数据) 变 O(全表字节)，且重写期间新旧文件并存、提交窗口变长、冲突面变大；add column 要么也重写填 null、要么读取端永远特判「缺列=null」。field id 是把「列的身份」与「列的显示名」解耦——身份稳定、名字可变，这与「内容寻址把对象身份与存储位置解耦」是同一个思想的两面。）

---

## §7 跨级锚：L2 快照内容 == L0/L1 全量重算（字节级）

L2 最硬的验收不是「跑通了」，而是**换了整套存储/提交机制之后，数据语义零漂移**：

```text
  [check 01] PASS  跨级锚: seq1 内容 digest == L0/L1 v1 4599c15439c026c8
  [check 03] PASS  跨级锚: seq2 内容 digest == L0/L1 v2 a12337250f5d4d79
  [check 16] PASS  time travel 复现 L0/L1 v2 锚（快照隔离 + 历史可重放）
  [check 18] PASS  rollback 后当前视图 = seq2 的 8 行（按当时 schema s1 读复现 v2 锚）
```

四个锚各证一件事：check 01/03 = L2 的元数据树 + 内容寻址 + 乐观并发这套全新机制，物化出的表内容与 L0 全量重算、L1 增量物化**逐字节相同**（同 fixture 同 `fold_events`，digest 口径 `sha256(canonical_json)[:16]` 三级同款）；check 16 = time travel 读出的历史视图也逐字节复现；check 18 = rollback 之后的当前视图同样复现——**机制可以换代，语义必须守恒**，这是阶梯间「K+1 没有偷偷改 K」的机器证明形态。

---

## §8 机制面 [4]：Terraform HCL + state locking —— 并发 apply 为什么被拒而不是损坏

L0 的 plan/apply 是单线程玩具；L2 补上两块真东西：**真实 HCL 子集**与 **state locking**。

**（a）HCL 子集解析。** `HCL_CONFIG` 是一段真实 HCL 语法的配置（`resource "<type>" "<name>" { ... }` block + attribute）：

```hcl
resource "dp_dataset" "sft_support" {
  gate     = "required_fields+dedup"
  pii_drop = ["phone"]
}
resource "dp_grant" "trainer" {
  read = ["sft_support"]
}
```

`parse_hcl` 解析 block/attribute/字符串/列表/布尔/整数（check 21：3 blocks、属性类型正确），不支持表达式/变量引用/嵌套 block——教学子集够演示「声明式 desired state」的全部机制，真实 HCL 全语法见 hashicorp/hcl 的 spec（抓取件 `hcl_spec.md`）。

**（b）state = serial + lineage + lock。** `TfState` 的三个字段各管一摊：`serial` 每次状态真变 +1（no-op apply 不推进，check 25）；`lineage` 是状态的身份（防止拿错 state 文件）；`lock` = `{owner, lock_id}`，**lock_id 是 nonce**。Terraform 官方文档（https://developer.hashicorp.com/terraform/language/state/locking ，抓取件 `tf_locking.html` 171,791 B，页面 dateModified 2025-11-19，逐字引文）：

> "If supported by your backend, Terraform will lock your state for all operations that could write state. This prevents others from acquiring the lock and potentially corrupting your state." …… "If state locking fails, Terraform does not continue."

注意第二句的语义：**锁失败 = 拒绝，不是等待、更不是继续**。[7] 段演示了完整的三连（check 23/24/25）：

- apply-P1-long 持锁期间，apply-P2 的 `acquire` 直接抛 `LockError`（"拒绝而非等待损坏"）——并发 apply 被拒，state 零损坏；
- 伪造 `lock-999` 解锁被拒（"解锁只认 nonce"）——官方文档同页对 lock ID 的定位（逐字引文）："This lock ID acts as a nonce, ensuring that locks and unlocks target the correct lock." nonce 语义 = **只有持锁者能解锁**：锁记录里存着 lock_id，release 必须报出同一个 id，任何第三方（包括「上一个 apply 的残影」）都解不开；
- 解锁后 apply-P2 重试 = 0 actions 幂等 no-op、serial 维持 1——desired state 没变，世界就不动（L0 的幂等契约在加锁之后依然成立）。

**为什么「拒绝」优于「等待」？** 等待型互斥（排队拿锁）把故障藏起来了：死锁的持锁者会让所有人无限等待，或者等超时后强行接管——强行接管就是两个 writer 并存的开始。拒绝型把问题立刻抛回给人/上层系统（"state 被谁锁着、lock_id 是什么"都在错误信息里），配合 nonce 保证「强解」只能是有意识的 `force-unlock` 操作而非意外。真实 Terraform 的 force-unlock 同页文档明确要求 unique lock ID（引文见上）——**安全阀也是认 nonce 的**。

**（c）drift 检测 = 最小 diff 修复。** 带外删掉 state 里的 trainer 授权（模拟有人绕过平台手动改了世界），`plan` 给出的修复是**恰 1 个 action**（check 26：`[('create:dp_grant', 'trainer')]`）——只补缺的那一格，不重建世界。这是声明式的另一半价值：world 与 desired 的 diff 永远可计算，修复成本与偏差量成正比、与世界的总规模无关。

**（d）命名映射：HCL 下划线 vs 表名连字符（消费边界的显式映射，不静默抹平）。** 注意一个细节：HCL 里的资源名是 `sft_support`（下划线），catalog 里的表名是 `sft-support`（连字符）。这不是笔误——HCL spec 的 Identifiers 节（抓取件 `hcl_spec.md` L90-109，逐字引文）：

> "The dash character `-` is additionally allowed in identifiers, even though that is not part of the unicode `ID_Continue` definition. This is to allow attribute names and block type names to contain dashes, although underscores as word separators are considered the idiomatic usage."

下划线是 HCL 资源名的惯用词分隔符（idiomatic），而表名用连字符是湖仓/SQL 世界的常见形态——**两个命名空间各有各的惯例**。nano 版的选择是在消费边界显式映射（`consume` 内 `name.replace("_", "-")`，代码 L454-456 注释即此设计），而不是在某一层静默统一：静默抹平会让「HCL 里写的名字」与「平台上生效的名字」之间的映射规则变得不可见，出问题时无法审计；显式映射把「谁的名字在哪个边界变成什么」钉在代码里。授权本身在消费边界 default-deny 执行（check 27：trainer 可读 8 行 / intern 抛 `PermissionError`）——**安全 first-class：授权声明落 state 只是记账，真正的闸门在读数据的那一刻**。

**思考题 8.1**：如果把 lock 从 state 文件里的一个字段改成「创建 lock 文件」来实现（本地文件系统的常见土法），在多进程/多机场景下会引入什么新风险？（提示：创建文件的原子性依赖文件系统语义——NFS 上 `O_CREAT|O_EXCL` 的历史行为、以及「持锁者崩溃后 lock 文件还在」的死锁形态；真实 backend 用 DynamoDB 条件写 / Consul session 这类提供原子 CAS + TTL 的原语，正是为了把锁的正确性建立在比文件系统更强的保证上。`[TODO: verify on real system]`：真多机锁行为归真机验证攒批。）

---

## §9 成本与孤儿账本（cost first-class）

[8] 段把对象存储的字节账摊开（toy 尺度声明同 §3(c)：讲机制不讲规模）：

```text
data 903 B / manifest 270 B / mlist 360 B / metadata 12610 B（去重命中 2 次）
metadata 对象 11 个，其中 2 个是冲突遗留孤儿
```

两笔账值得盯住：

**（a）metadata 比 data 大一个数量级（12610 B vs 903 B）。** 这不是 bug，是内容寻址元数据树的固有成本结构：每次 commit 都要写一份新的完整 metadata（含全部 snapshot/schema 历史），而 data 只写增量。toy 尺度下 metadata 显得巨大，是因为 8 行数据太小；生产尺度下 data 以 TB 计、metadata 以 MB 计，占比翻转——但**「metadata 是每次提交的全量、data 是增量」这个结构不变**，它解释了真实湖仓的两类运维动作：metadata 压缩/过期快照清理（控制 metadata 增长），以及 manifest 重写（小文件合并）。

**（b）冲突孤儿是乐观并发的已知成本，不是泄漏事故。** 2 个孤儿 metadata（[3] writer-A 冲突 attempt + [4] rename 过期 base attempt）都是「乐观构建出来、提交失败即弃」的对象。真实 Iceberg 由 `cleanUncommitted` 清理（SnapshotProducer.java:L524，抓取件逐字：`cleanUncommitted(Sets.newHashSet(saved.allManifests(ops.io())));`）；nano 版刻意**保留**孤儿并在账本里报数——因为「孤儿从哪来、为什么必然有」本身就是乐观并发的成本教材：**每一次冲突重试都可能在存储上留下未提交的残骸，清理是协议的一部分而非可选运维**。check 28/29 把两笔账都钉成断言：去重命中恰 2（重做=复用）、孤儿恰 2（失败 attempt 的残骸）。

state digest `dafdf0c9bfdecd5e` = catalog 指针 + snapshot-log + schema 序列 + tf state 的逻辑哈希（check 30），是「平台当前状态」的一个指纹——与 L1 的 catalog digest 同款思想：逻辑哈希而非文件哈希，复跑收敛。

---

## §10 权威实现取舍表：nano 版 vs Apache Iceberg / Terraform

行锚基于 **apache-iceberg-1.11.0 tag**（抓取件 `iceberg_tags.atom` 确认该 tag 存在，2026-08-15 抓取）；源码路径 `core/src/main/java/org/apache/iceberg/`，行号为 2026-08-15 抓取件录值（§八 时效性声明见 §14）。

| 机制 | nano 版（本文件） | 权威实现 | 差异与原因 |
|------|-------------------|----------|-----------|
| commit 重试循环 | `commit_with_retry`：逻辑步进重试，冲突 → refresh → 重新 `apply()` | `SnapshotProducer.commit()`（SnapshotProducer.java:L459-501）：`Tasks.foreach(ops).retry(...)`（L465）+ 每次 attempt 重新 `Snapshot newSnapshot = apply();`（L475）+ `taskOps.commit(base, updated.withUUID())`（L501）CAS | 结构同款（重试内重做 apply + CAS 提交）；nano 用逻辑步进替代**指数退避**（`.exponentialBackoff(...)`，L466-470）——退避是 wall-clock 行为，会破坏输出确定性，机制上它只影响「多久重试一次」不影响「重试什么」 |
| 重试次数 | `max_retries=4` | `COMMIT_NUM_RETRIES = "commit.retry.num-retries"`、`COMMIT_NUM_RETRIES_DEFAULT = 4`（TableProperties.java:L89-90） | 默认值对齐（4），且同为表属性可配 |
| rollback | `RollbackOp`：指针指回旧快照的普通 commit，不删历史 | SnapshotProducer.java:L479 注释 "this is a rollback operation"——rollback 走同一条 commit 路径（`base.snapshot(id) != null` 分支） | 语义同款；Iceberg 还有 branch/tag 引用的完整语义（nano 只有 current 指针） |
| 失败清理 | 保留孤儿并报账（[8] 段） | `cleanUncommitted(Sets.newHashSet(saved.allManifests(ops.io())))`（SnapshotProducer.java:L524）+ 清理多余 manifest list | nano 刻意不清理：孤儿是乐观并发的成本教材，清掉就看不见了 |
| schema 更新验证 | `SchemaOp.apply` 内 `current-schema-id != base_schema_id` → `CommitValidationError`（不进重试） | spec 规则（§5 引文）："Table schema updates and partition spec changes must validate that the schema has not changed between the base version and the current version." | nano 只验 schema（partition spec 未建模）；验证时机同款——构建期验证、失败拒绝 rebase |
| 元数据树 | metadata → snapshot → mlist → manifest → data，JSON 内容寻址 | 同款五层树（Iceberg spec），真实载体 = Avro manifest + JSON metadata，存储 = S3/HDFS/GCS | 层级与引用语义同款；nano 用 JSON + 本地 tempdir 替代 Avro + 对象存储（可运行性契约，§14） |
| catalog | 内存 dict 指针 + check-and-put | Hive Metastore / Glue / REST catalog / Hadoop 文件系统原子 rename（多实现） | CAS 语义同款；真实 catalog 的原子性由各自 backend 保证（DynamoDB 条件写、RDBMS 事务、文件系统 rename）`[TODO: verify on real system]` |
| state locking | `TfState.lock` 字段 + nonce（单文件） | Terraform backend 锁：DynamoDB/Consul/Terraform Cloud（官方 locking 文档，§8 引文） | 语义同款（锁失败即拒 + nonce 解锁）；nano 把锁放进 state 文件字段，真实 backend 用外部原语获得跨进程/跨机原子性与 TTL |
| HCL | `parse_hcl` 教学子集（block/attribute/标量/列表） | hashicorp/hcl 全语法（表达式、变量、函数、嵌套 block，hcl_spec.md） | §七：HCL 到 L2 才触及——触及的是「声明式 desired state + plan/apply」机制，不是语法全覆盖 |
| dbt 分层 | `staging()` = raw 事件 → 质量门 + PII 投影 → curated 行（L0/L1 同款漏斗） | dbt 的 staging/intermediate/marts 分层模型（dbt 官方项目结构指南，抓取件 `dbt_structure.html`："Staging — ...building blocks, from source data / Intermediate — stacking layers of logic... / Marts — bringing together our modular pieces..."） | nano 两层（raw→curated）对应 dbt staging/marts 的最小形态；dbt 的模型物化增量（incremental/merge）在 L1 `build_version` 已对照过 |

---

## §11 交叉对照：Delta Lake —— 同款乐观并发的另一具象

Iceberg 不是唯一答案。Delta Lake 用**事务日志**而非元数据树达到同一目标，官方协议文档（delta-io/delta `PROTOCOL.md`，master，抓取件 `delta_protocol.md` 209,306 B，2026-08-15 抓取，L151-152/L157 逐字引文）：

> "Delta's transactions are implemented using multi-version concurrency control (MVCC)." …… "As a table changes, Delta's MVCC algorithm keeps multiple copies of the data around rather than immediately replacing files that contain records that are being updated or removed."

> "First, they optimistically write out new data files or updated copies of existing ones."

对照三个关键点：**（a）不可变 + 多版本**是共同的底层思想——Delta「保留多份副本而非就地替换文件」与 Iceberg 的不可变快照树同构，L2 的 §3(a) 性质在两边都成立；**（b）提交载体不同**——Delta 的每次提交是向 `_delta_log` 追加一个原子编号的 JSON action 文件（日志即历史，版本 = 日志长度），Iceberg 是指针交换 + 不可变元数据树（历史 = 快照链）；**（c）冲突检测点不同**——Delta 在提交时对比「自己读时的版本」与「日志当前末尾」之间的 action 序列做兼容性判定（协议文档的 conflict 规则族），Iceberg 在 CAS 指针时判定 + 按更新类型决定是否可重试（§5 的 spec 规则）。**选型判断（senior 口径）**：两者都是生产级湖仓格式，机制上都能承载本教程的全部语义；差异更多在生态绑定、catalog 集成与演进路线上——本教程以 Iceberg 为解剖对象是因为其 spec 与元数据树结构对教学最透明，不构成对 Delta 的优劣判定；云厂商实现（Databricks/Snowflake/Redshift/Glue 等）一律作参照不锁定，价目与性能数字以官方文档为准或标 `[TODO: verify]`。

---

## §12 费曼自检

**讲给外行听**：把数据平台想象成一座**档案馆**。L1 时代，档案柜（SQLite）是结实的，但全馆只有一支笔（单线程提交）：两个人同时要归档新材料，只能排队。L2 做了四件事。其一，所有档案一律**用内容盖章编号**（内容寻址）：同一份材料只有一个编号、盖过章的永不涂改——于是「拍一张全馆快照」不再是复印全部档案，而是**记一张新的目录卡**，卡上大部分条目指向旧档案（§3：快照边际成本 = 新目录卡，不是复印世界）。其二，全馆唯一可变的东西是门口「当前目录卡」的**指针**：归档比赛（并发提交）的规则是「交材料时先报你依据的是哪张旧卡」——报对了才换指针，报错了就拿最新卡重来（§4/§5）；而「改档案分类法」（schema 更新）必须验证「我改的时候分类法没被人动过」，因为分类法改动不能像加档案那样随便重来（§5）。其三，档案按**编号**而非标题上架：给某类档案改标题（rename）只改目录卡、不动一份档案（§6：零重写机器证明）；想看「三个月前馆里长什么样」随时可调（time travel），「撤销上次归档」也只是把指针指回旧卡、历史一页不撕（rollback）。其四，馆里的**门禁与规章**用声明式配置管理（HCL）：两个人同时要改规章，门锁（state lock）会把后到的那个直接拒之门外而不是让两人同时涂改（§8）；门禁卡带防伪码（nonce），不是持卡本人刷不开锁；规章里写了「谁能进哪个库房」，在**真正进门那一刻**查验（消费边界 default-deny）。

**自检问**（答不上来就回对应小节）：

1. 为什么「快照便宜」与「冲突重试便宜」是同一个机制（内容寻址）的两个推论？（§3(a)(b)(c)）
2. CAS 的 check 与 put 为什么不能颠倒顺序？（§4 思考题 4.1）
3. append 永远可重试、schema 更新必须验证——判据是什么一句话？（§5：更新语义对中间历史的无关性）
4. rename 零重写数据，靠的是哪个设计决策？按名字引用会怎样？（§6 + 思考题 6.1）
5. state lock 失败时为什么「拒绝」优于「等待」？nonce 防的是谁？（§8(b)）

---

## §13 思考题（×5）

1. **（重试的边界）** `commit_with_retry` 对 `CommitConflictError` 重试、对 `CommitValidationError` 不重试。如果把两者都纳入重试，给一个让系统产生错误结果的交错序列；如果把两者都不重试，并发吞吐会付出什么代价？（要点：前者让「基于过期 schema 的更新」在 refresh 后以新 base 重放——而发起方的意图可能已经失效，静默重放 = 静默改变语义；后者把每次冲突都变成上层失败，高并发下 append 吞吐崩塌。两条错误路径的区分就是「可自动消化的冲突」与「必须人工/发起方重新决策的冲突」的边界。）
2. **（孤儿的治理）** nano 版保留孤儿报账，真实 Iceberg 由 `cleanUncommitted` 清理（SnapshotProducer.java:L524）。但「提交成功后的清理」只能清自己这次的对象——崩溃在「提交成功、清理之前」的孤儿谁来清？设计一个兜底机制（提示：按「对象是否被任何已提交 metadata 引用」做周期性 GC——这正是真实 Iceberg `remove_orphan_files` 过程的存在理由；想清楚为什么不能只靠提交路径内的清理）。
3. **（锁的粒度）** 本实现的锁是「整个 state 一把锁」。真实 Terraform 可以用 `-target` 做局部 apply，但锁仍是全局的——为什么「局部 apply + 全局锁」是合理的中间形态，而不是按资源分锁？（提示：资源间有依赖图，按资源分锁要解决死锁/升级问题；全局锁的代价是串行化，但 apply 本身是低频运维动作——锁粒度的取舍取决于临界区频率与依赖结构，不是越细越好。）
4. **（field id 的代价）** field id 永不复用是 Iceberg 的硬规则（drop 掉的 id 不能被新列占用）。为什么？如果允许复用，给一个让 time travel 读出错误语义的场景。（要点：旧文件里那个 id 还有值——新列复用 id 后，time travel 到新列加入前的快照会把旧值读成新列的值，类型都可能不同。id 复用省的是 schema 里的编号空间，赔掉的是历史的可信度。）
5. **（声明式的边界）** drift 检测（§8(c)）能抓住「带外删除」，但抓不住什么？给一个 state 显示全绿、世界实际已错的场景。（要点：state 只记录「我声明过的资源在不在、配置对不对」——带外**新增**一个 state 不知道的资源（影子授权、私开的存储桶），plan 的 diff 里它不存在；这与 L1 §8「内层对账抓不住接入层的漏」同构：每层校验只覆盖自己的声明范围，声明之外的世界要靠审计/巡检。安全 first-class 的另一半：default-deny 让「未声明」等价于「不允许」。）

---

## §14 反例与边界

**一个常见错误直觉**：「有了快照和 time travel，数据永远不会丢、错账永远能回滚。」——反驳在 §9(b) 与 §5：rollback 能撤销的是**提交**，撤销不了「错误数据已经被下游消费」的事实（训练任务按 seq5 跑完的梯度不会回滚）；且快照链本身依赖 catalog 指针与元数据的存活——元数据丢了（catalog 损坏且无备份），不可变对象堆得再齐也拼不回表。**time travel 是账本的可回溯性，不是业务的时光机。**

**toy 尺度诚实声明**：本节全部数字（987 B / 2442 B / 40.4% / 12610 B / 11 个 metadata / 2 个孤儿）来自 §2 的真实运行输出，但 fixture 只有 11 行——**讲机制、不可外推**。生产尺度下的量级关系（metadata 占比翻转、manifest 合并的必要性、锁竞争频率）见 §9/§10 的定性讨论，真实数字以官方文档或实测为准。

**模拟了**（本教程验收内容）：内容寻址元数据树（不可变 + 去重 + 引用共享）；快照边际字节账；catalog CAS 提交 + 乐观并发重试（refresh + re-apply）；append 可 rebase vs schema 更新验证拒绝（spec 双引文）；field id schema evolution（add/rename/drop，零重写机器证明）；time travel（as-of × 当时/当前 schema）与 rollback（普通 commit、历史不删）；HCL 子集 plan/apply；state locking（锁失败即拒 + nonce 解锁 + 幂等 no-op + drift 最小 diff）；授权在消费边界 default-deny 执行；成本与孤儿账本。

**刻意没模拟**（每行都是更高阶梯的课题或真机验证项）：

| 没模拟 | 为什么 | 归属 |
|--------|--------|------|
| 真多进程/多机并发提交 | 单进程脚本化交错是本质模拟；真并发涉及网络分区、时钟、backend 原子性 | 需在目标对象存储与 catalog 上另行验证 |
| 对象存储 put-if-absent / rename 的实际原子性 | 本地 tempdir 无此语义差异 | 同上 |
| 指数退避与 jitter | wall-clock 行为破坏输出确定性 | 机制说明在 §10（真实 Iceberg L466-470），行为不模拟 |
| partition spec / hidden partitioning | 独立机制面，与 commit protocol 正交 | 超出本模块阶梯（spec 引文已带 "and partition spec changes"） |
| row-level deletes（equality/position delete） | 独立机制面（merge-on-read） | 超出本模块阶梯 |
| branch/tag 全语义、snapshot 过期与 GC 实操 | nano 只有 current 指针 + 孤儿报账 | 概念已在 §10/§13 对照，实操非本模块目标 |
| Terraform provider/远端 backend 生态 | 环境依赖重 | §七：HCL 到 L2 触及的是机制，不是生态实操 |

**§八 时效性声明**：Iceberg 行锚基于 **apache-iceberg-1.11.0 tag**（2026-08-15 抓取 `iceberg_tags.atom` 确认 tag 在位），行号为该 tag 抓取件录值——其他版本行号可能漂移，引用时以抓取日为准。定位声明（§八 A 层口径）：Iceberg 的 snapshot/commit/schema-evolution 机制是湖仓格式的**经典锚点**——spec 的提交语义（§4/§5 双引文）至今是理解一切湖仓格式的地基，不存在「已过时被替代」；但各引擎/云厂商的实现与扩展（REST catalog、variant 类型、PV3 等）持续演进，本文只锚 spec 级与 1.11.0 源码级事实，不锚任何厂商特性。Delta 引文基于 master 分支 PROTOCOL.md（2026-08-15 抓取），Terraform locking 引文基于官方文档页（dateModified 2025-11-19），HCL 引文基于 hashicorp/hcl spec。

---

## §15 阶梯预告

L2 是 nano-data-platform 阶梯的顶级（README 阶梯表 L0–L2）。三级走完的机制清单：L0 裸出「湖仓分层 + 声明式状态管理」两个本质；L1 还清「重启蒸发 + 全量重算」两笔债（watermark 增量 + 持久化 catalog + 双层对账）；L2 换上内容寻址元数据树、CAS 乐观并发、field id 演化与 state locking，并用跨级 digest 锚证明**机制换代、语义守恒**。真实 Iceberg 集群、Terraform backend 与湖仓运维实操应进入独立 deep-dive；本教程的 §10 与 §14 是起点清单。

---

## §16 溯源与口径声明

| 声明 | 类型 | 来源 |
|------|------|------|
| spec 双引文（§4/§5："Append operations have no requirements and can always be applied." / "Table schema updates and partition spec changes must validate that the schema has not changed between the base version and the current version."） | 文献已有（逐字引文） | https://iceberg.apache.org/spec/ ，抓取件 `iceberg_spec.html` 883,921 B，2026-08-15 抓取（与 L1 2026-08-13 重抓同尺寸）；代码 docstring 引文 "Table schema updates …" 的 "…" 省略 "and partition spec changes"，口径已声明 |
| SnapshotProducer.java 行锚（§10：retry / apply / commit / cleanup） | 文献已有（源码） | [apache-iceberg-1.11.0 `SnapshotProducer.java`](https://github.com/apache/iceberg/blob/apache-iceberg-1.11.0/core/src/main/java/org/apache/iceberg/SnapshotProducer.java)；正文行号以该 tag 为准 |
| TableProperties.java:L89-90（`COMMIT_NUM_RETRIES` 默认 4） | 文献已有（源码逐字） | 同上 tag，`core/src/main/java/org/apache/iceberg/TableProperties.java`，2026-08-15 抓取 |
| apache-iceberg-1.11.0 tag 存在 | 文献已有 | https://github.com/apache/iceberg/tags atom feed，抓取件 `iceberg_tags.atom`，2026-08-15 抓取 |
| Delta 三句引文（§11：MVCC / 保留多副本 / 乐观写） | 文献已有（逐字引文） | delta-io/delta `PROTOCOL.md`（master），抓取件 `delta_protocol.md` 209,306 B（"Delta Transaction Log Protocol"），2026-08-15 抓取，L151/L152/L157 |
| Terraform locking 三句引文（§8：自动锁 / 锁失败不继续 / lock ID acts as a nonce） | 文献已有（逐字引文） | https://developer.hashicorp.com/terraform/language/state/locking ，抓取件 `tf_locking.html` 171,791 B（页面 dateModified 2025-11-19），2026-08-15 抓取 |
| HCL Identifiers 引文（§8(d)：下划线为惯用词分隔符） | 文献已有（逐字引文） | hashicorp/hcl spec（`hclsyntax/spec.md` Identifiers 节），抓取件 `hcl_spec.md` L90-109，2026-08-15 抓取 |
| dbt staging/intermediate/marts 分层（§10 表） | 文献已有（短语引文） | dbt 官方项目结构指南，抓取件 `dbt_structure.html` 60,923 B，2026-08-15 抓取 |
| L0/L1 digest 锚 `4599c15439c026c8` / `a12337250f5d4d79` | 文献已有（L0/L1 录值） | `tutorial_L0.md` §5 / `tutorial_L1.md` §6（跨级锚，check 01/03/16/18 机器证明） |
| 全部字节账/attempt 数/digest（987 B、2442 B、40.4%、903/270/360/12610 B、去重 2、孤儿 2、commits 8、`dafdf0c9bfdecd5e` 等） | 本实现实测（toy 设定） | `L2_commit_protocol_schema_evolution.py` §2 paste 块（与运行输出 BYTE-IDENTICAL），非真实云价、不可外推 |
| 「乐观并发可重试性 = 更新语义对中间历史的无关性」（§5 末） | 合理推断（机制归纳） | 由 spec 双引文与 [3]/[4] 实测归纳，非 spec 原文 |
| 云厂商实现（Glue/Redshift/Snowflake/Databricks 等）提及 | 概念性参照 | 未引任何价目或性能数字 |

**运行锚点**（2026-08-31，`-B`）：代码 md5 `cace8d26d1e702d0c3305389b8111f2f`/488 行/33,437 B；两个新建空 CWD 均 EXIT=0、stderr 0 B；stdout md5 `582752ca4d46fac422b5ade14e18abb1`/81 行/6,230 B，逐字节一致；self-check 30/30 PASS，state digest `dafdf0c9bfdecd5e`，跨级锚 check 01/03/16/18 PASS。

下一站：**「数据平台工程」sota-deepdive**——湖仓格式生态选型、真实 catalog 架构与 MLOps 工程化；本教程 §10/§14 为起点清单。
