#!/usr/bin/env python3
"""nano-data-platform L1 — 增量接入（watermark/cursor）+ 持久化 catalog（SQLite）+ 重放对账。

L0 把「分层契约 + 声明式状态」裸了出来，但所有状态都在内存里：进程一重启，watermark、
血缘、快照全部蒸发，下次接入只能从头全量重拉。L1 抓四个机制面（L1 验收，README 阶梯表）：

  [1] watermark/cursor 增量接入：只拉「游标之后」的变更事件。游标选型是本质决策——
      单调 lsn（变更日志位点）无盲区但要求源侧有 append-only log；updated_at 游标便宜
      （源侧只需一个索引列）但对回填/时钟回拨有盲区（[10] 反例探针实测）。
      对应 Airbyte/Fivetran 类连接器的 incremental sync（cursor 定义见 tutorial §4 引文）。
  [2] 持久化 catalog（SQLite 级）：sources/batches/raw_records/curated_versions/
      curated_records 五张表落盘——watermark、血缘、版本在进程重启后原样可重建，
      血缘查询变成 SQL。对应真实系统的 catalog 层（Hive Metastore / Glue Data Catalog /
      Iceberg catalog：管理表的位置与快照序列，见 tutorial §3）。
  [3] 增量物化（merge/upsert 语义）：新版 curated = 上一版快照 + 仅新 batch 的增量折叠，
      不重算世界；update 事件原位覆写（dbt incremental 的 unique_key/merge 思想，§5 引文）；
      漏斗语义（质量门 + PII 投影 + 去重）逐字复现 L0——跨级 digest 锚机器证明（§6）。
  [4] 双层对账（reconciliation）：内层「raw 全量重放 == 增量 curated 账目」（物化正确性），
      外层「源全量导出 == raw 账目」（接入完备性）——增量接入用成本换完备性，对账是
      让这笔交易安全的保险丝；发现分歧 → 全量回补（backfill）自愈。

刻意不模拟（更高阶梯课题，见 README 阶梯表）：并发 commit 冲突与乐观并发控制（L2，
Iceberg commit protocol）、schema evolution（L2）、真实列存格式（Parquet/ORC）、
分布式 catalog 与 state locking、真实连接器生态。
零依赖（纯标准库，sqlite3 内置），CPU 秒级；输出确定（逻辑时钟，且钟本身可从 catalog
重建——重启不断编号），任意 CWD 可跑，复跑逐字节一致。
"""
import hashlib, json, os, shutil, sqlite3, tempfile

CHECKS = []
def check(name, cond):
    CHECKS.append(bool(cond))
    print(f"  [check {len(CHECKS):02d}] {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: raise SystemExit("self-check failed: " + name)

class Clock:
    """逻辑时钟：确定性输出。L1 增量点：钟本身可从 catalog 重建（重启不断编号）。"""
    t = 0
    @classmethod
    def tick(cls):
        cls.t += 1
        return cls.t
    @classmethod
    def rebase(cls, base):
        cls.t = max(cls.t, base)

# ---- secrets manager：与 L0 同款机制（代码里只有凭据名，值在集中存储，绝不硬编码） ----
class SecretStore:
    def __init__(self): self._s = {}
    def put(self, name, value): self._s[name] = value
    def get(self, name):
        if name not in self._s: raise KeyError(f"secret '{name}' 未注册 —— 拒绝接入而非静默失败")
        return self._s[name]

# ---- 源系统模拟：append-only 变更日志（lsn 单调递增）+ 全量导出（对账的源侧真相） ----
class SourceSystem:
    def __init__(self, name, token):
        self.name, self.token, self.log = name, token, []
    def push(self, op, row, updated_at):
        assert op in ("insert", "update")
        self.log.append(dict(lsn=len(self.log) + 1, op=op, row=row, updated_at=updated_at))
        return self.log[-1]
    def changes_since(self, wm, cursor_kind):
        key = (lambda e: e["lsn"]) if cursor_kind == "lsn" else (lambda e: e["updated_at"])
        return [e for e in self.log if key(e) > wm]
    def export_full(self):
        """源侧真相 = 变更日志按 lsn 折叠（后写覆盖）。外层对账的参照系。"""
        rows = {}
        for e in self.log: rows[e["row"]["id"]] = e["row"]
        return rows

# ---- [2] 持久化 catalog：五张表落盘（SQLite），进程重启后一切可重建 ----
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
CREATE TABLE IF NOT EXISTS curated_versions(
  dataset TEXT NOT NULL, version TEXT NOT NULL, n INTEGER NOT NULL,
  sha256 TEXT NOT NULL, built_at INTEGER NOT NULL,
  from_batches TEXT NOT NULL, problems TEXT NOT NULL,
  PRIMARY KEY(dataset, version));
CREATE TABLE IF NOT EXISTS curated_records(
  dataset TEXT NOT NULL, version TEXT NOT NULL, pos INTEGER NOT NULL,
  rid TEXT NOT NULL, payload TEXT NOT NULL,
  PRIMARY KEY(dataset, version, rid));
"""
def open_catalog(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    base = 0
    for sql in ("SELECT MAX(ingested_at) FROM batches", "SELECT MAX(built_at) FROM curated_versions"):
        v = con.execute(sql).fetchone()[0]
        base = max(base, v or 0)
    Clock.rebase(base)  # 逻辑时钟从 catalog 重建——重启不断编号
    return con

def register_source(con, source, cursor_kind, secrets, token):
    secrets.put(f"{source}/credential", token)
    with con:
        con.execute("INSERT INTO sources VALUES(?,?,?,0)", (source, cursor_kind, f"{source}/credential"))

# ---- [1] watermark 增量接入：只拉游标之后；落数据与推游标在同一事务（exactly-once 物化的核心） ----
def sync_source(con, source, src, secrets, cursor_kind=None, force_wm=None):
    kind0, cred, wm0 = con.execute(
        "SELECT cursor_kind, credential_name, watermark FROM sources WHERE source=?", (source,)).fetchone()
    kind = cursor_kind or kind0
    wm = force_wm if force_wm is not None else wm0
    if secrets.get(cred) != src.token:
        raise PermissionError(f"sync denied: source '{source}' 凭据错误 —— 认证是接入边界第一道闸")
    events = src.changes_since(wm, kind)
    if not events:
        return 0, None  # 幂等：没有新变更 = no-op，源端重试安全
    inserted, batch_id = [], "b%03d" % (con.execute("SELECT COUNT(*) FROM batches").fetchone()[0] + 1)
    with con:  # 单事务：raw 落盘 + batch 血缘 + watermark 推进，要么全成要么全不成
        for e in events:
            cur = con.execute(
                "INSERT OR IGNORE INTO raw_records VALUES(?,?,?,?,?,?)",
                (source, e["lsn"], e["op"], e["row"]["id"],
                 json.dumps(e["row"], sort_keys=True, ensure_ascii=False), batch_id))
            if cur.rowcount: inserted.append(e)
        if inserted:
            payload = json.dumps([e["row"] for e in inserted], sort_keys=True, ensure_ascii=False).encode()
            con.execute("INSERT INTO batches VALUES(?,?,?,?,?,?,?,?)",
                        (batch_id, source, len(inserted), len(payload),
                         hashlib.sha256(payload).hexdigest()[:16],
                         min(e["lsn"] for e in inserted), max(e["lsn"] for e in inserted), Clock.tick()))
        new_wm = max((e["lsn"] if kind == "lsn" else e["updated_at"]) for e in events)
        con.execute("UPDATE sources SET watermark=? WHERE source=?", (max(new_wm, wm), source))
    if not inserted:
        return 0, None  # 段内全是已收过的重复事件：游标照推，不产生空 batch
    return len(inserted), batch_id

# ---- [3] 增量物化：单遍折叠。insert-only 流下与 L0 quality_gate 逐条同行为（跨级锚的前提） ----
REQUIRED = ("id", "text", "label")
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

def _batch_events(con, batch_ids):
    evs = []
    for b in batch_ids:
        for op, payload in con.execute(
                "SELECT op, payload FROM raw_records WHERE batch_id=? ORDER BY lsn", (b,)):
            evs.append((op, json.loads(payload)))
    return evs

def _persist_version(con, dataset, state, problems, from_batches):
    kept = list(state.values())
    body = json.dumps(kept, sort_keys=True, ensure_ascii=False).encode()
    n = con.execute("SELECT COUNT(*) FROM curated_versions WHERE dataset=?", (dataset,)).fetchone()[0]
    version = f"v{n + 1}"
    digest = hashlib.sha256(body).hexdigest()[:16]
    with con:
        con.execute("INSERT INTO curated_versions VALUES(?,?,?,?,?,?,?)",
                    (dataset, version, len(kept), digest, Clock.tick(),
                     json.dumps(sorted(from_batches)), json.dumps(problems, ensure_ascii=False)))
        for pos, (rid, rec) in enumerate(state.items()):
            con.execute("INSERT INTO curated_records VALUES(?,?,?,?,?)",
                        (dataset, version, pos, rid, json.dumps(rec, sort_keys=True, ensure_ascii=False)))
    return dict(version=version, n=len(kept), sha256=digest, problems=problems,
                from_batches=sorted(from_batches))

def build_version(con, dataset):
    """增量物化：上一版快照状态 + 仅新 batch 的事件 → 新版（不重算世界）。"""
    prev = con.execute(
        "SELECT version, from_batches FROM curated_versions WHERE dataset=? ORDER BY built_at DESC LIMIT 1",
        (dataset,)).fetchone()
    known, state = set(), {}
    if prev:
        known = set(json.loads(prev[1]))
        for rid, payload in con.execute(
                "SELECT rid, payload FROM curated_records WHERE dataset=? AND version=? ORDER BY pos",
                (dataset, prev[0])):
            state[rid] = json.loads(payload)
    new_batches = [r[0] for r in con.execute(
        "SELECT batch_id FROM batches ORDER BY ingested_at, batch_id") if r[0] not in known]
    problems = fold_events(state, _batch_events(con, new_batches))
    return _persist_version(con, dataset, state, problems, known | set(new_batches))

def full_replay(con):
    """[4a] 内层对账参照系：不读任何增量状态，从 raw 全量重放（同一折叠规则，独立代码路径）。"""
    batches = [r[0] for r in con.execute("SELECT batch_id FROM batches ORDER BY ingested_at, batch_id")]
    state = {}
    fold_events(state, _batch_events(con, batches))
    kept = list(state.values())
    body = json.dumps(kept, sort_keys=True, ensure_ascii=False).encode()
    return kept, hashlib.sha256(body).hexdigest()[:16]

def raw_fold(con, source):
    """raw 侧物化真相（不过质量门——外层对账比的是接入忠实度，不是清洗结果）。"""
    rows = {}
    for payload in con.execute(
            "SELECT payload FROM raw_records WHERE source=? ORDER BY lsn", (source,)):
        r = json.loads(payload[0])
        rows[r["id"]] = r
    return rows

def reconcile_source_to_raw(con, source, src):
    """[4b] 外层对账：源全量导出 vs raw 账目。返回分歧 id 列表（增量接入的保险丝）。"""
    truth, rawm = src.export_full(), raw_fold(con, source)
    div = []
    for rid in sorted(set(truth) | set(rawm)):
        a = json.dumps(truth.get(rid), sort_keys=True, ensure_ascii=False) if rid in truth else None
        b = json.dumps(rawm.get(rid), sort_keys=True, ensure_ascii=False) if rid in rawm else None
        if a != b: div.append(rid)
    return div

def catalog_digest(con):
    orders = {"sources": "source", "batches": "batch_id", "raw_records": "source, lsn",
              "curated_versions": "dataset, version", "curated_records": "dataset, version, pos"}
    dump = {t: con.execute(f"SELECT * FROM {t} ORDER BY {o}").fetchall() for t, o in orders.items()}
    body = json.dumps(dump, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(body).hexdigest()[:16]

# ---- demo fixture：与 L0 BATCH1/2/3 逐字相同（跨级 digest 锚的前提，见 tutorial §6 声明） ----
BATCH1 = [{"id": "t001", "text": "how to reset password", "label": "auth", "phone": "PHONE-DEMO-01"},
          {"id": "t002", "text": "charged twice on my card", "label": "billing", "phone": "PHONE-DEMO-02"},
          {"id": "t003", "text": "cannot export csv from dashboard", "label": "export", "phone": "PHONE-DEMO-03"},
          {"id": "t004", "text": "login page loops on safari", "label": "auth", "phone": "PHONE-DEMO-04"},
          {"id": "t005", "text": "upgrade plan from team to biz", "label": "billing", "phone": "PHONE-DEMO-05"}]
BATCH2 = [{"id": "t003", "text": "cannot export csv from dashboard", "label": "export", "phone": "PHONE-DEMO-03"},
          {"id": "t006", "text": "api rate limit question", "label": None, "phone": "PHONE-DEMO-06"},
          {"id": "t007", "text": "webhook retries after 500", "label": "integration", "phone": "PHONE-DEMO-07"},
          {"id": "t008", "text": "", "label": "billing", "phone": "PHONE-DEMO-08"}]
BATCH3 = [{"id": "t009", "text": "sso with okta setup help", "label": "auth", "phone": "PHONE-DEMO-09"},
          {"id": "t010", "text": "invoice pdf is blank", "label": "billing", "phone": "PHONE-DEMO-10"},
          {"id": "t007", "text": "webhook retries after 500", "label": "integration", "phone": "PHONE-DEMO-07"}]
# wave3：源侧回填修正（updated_at 回填到早于当前 watermark——§[10] 盲区探针的素材）+ 新增一条
T002_FIXED = {"id": "t002", "text": "charged twice on my card, order #4217", "label": "billing", "phone": "PHONE-DEMO-02"}
T011 = {"id": "t011", "text": "two-factor sms not arriving", "label": "auth", "phone": "PHONE-DEMO-11"}

def main():
    print("== nano-data-platform L1: 增量接入 + 持久化 catalog + 重放对账（SQLite catalog，纯标准库） ==")
    tmp = tempfile.mkdtemp(prefix="nano_dp_L1_")
    try:
        secrets = SecretStore()
        db = f"{tmp}/catalog.db"
        con = open_catalog(db)
        print("\n[1] 持久化 catalog（SQLite）：watermark/血缘/快照全部落盘（路径不入输出，复跑逐字节一致）")
        register_source(con, "crm", "lsn", secrets, "crm-token-demo")
        register_source(con, "web_log", "lsn", secrets, "web-token-demo")
        print("  5 张表: sources / batches / raw_records / curated_versions / curated_records")
        print("  声明源: crm (cursor=lsn) / web_log (cursor=lsn) —— 游标 = 增量接入的『读到哪了』")

        print("\n[2] 首次同步 = 全量回灌（watermark=0 → 拉全部）+ v1 物化（复现 L0 漏斗）")
        crm = SourceSystem("crm", "crm-token-demo")
        for i, r in enumerate(BATCH1 + BATCH2): crm.push("insert", r, updated_at=10 * (i + 1))
        try:
            sync_source(con, "crm", SourceSystem("crm", "wrong-or-stolen-token"), secrets)
            check("错误凭据必须被拒", False)
        except PermissionError as e:
            print(f"  denied as expected: {e}"); check("错误凭据必须被拒", True)
        n, b = sync_source(con, "crm", crm, secrets)
        wm = con.execute("SELECT watermark FROM sources WHERE source='crm'").fetchone()[0]
        print(f"  {b}: 拉取 {n} events (lsn 1–9)，watermark 0 → {wm}")
        check("首次同步 = 全量回灌 9 events", n == 9 and wm == 9)
        v1 = build_version(con, "sft-support")
        for p in v1["problems"]: print(f"  gate 拦截: {p}")
        print(f"  {v1['version']}: raw 9 条 → curated {v1['n']} 条 (sha256={v1['sha256']})")
        check("v1 漏斗 9→6，拦截 3 条（复现 L0 语义）", v1["n"] == 6 and len(v1["problems"]) == 3)
        check("跨级锚: v1 digest == L0 v1 4599c15439c026c8", v1["sha256"] == "4599c15439c026c8")
        _, rd = full_replay(con)
        check("内层对账: v1 增量物化 == raw 全量重放", rd == v1["sha256"])

        print("\n[3] 幂等：同一游标重拉 = no-op（源端重试安全）")
        n2, b2 = sync_source(con, "crm", crm, secrets)
        nb = con.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        print(f"  re-sync → {n2} new events / 新增 batch {nb - 1} 个（watermark 未动）")
        check("重拉幂等: 0 new、batch 数不变", n2 == 0 and b2 is None and nb == 1)

        print("\n[4] 进程重启：close → reopen，catalog 从盘上原样重建（L0 内存态做不到的事）")
        d_before = catalog_digest(con)
        con.close()
        con = open_catalog(db)  # 新进程视角：手里只有文件
        wm = con.execute("SELECT watermark FROM sources WHERE source='crm'").fetchone()[0]
        v1r = con.execute("SELECT n, sha256 FROM curated_versions WHERE dataset='sft-support' AND version='v1'").fetchone()
        print(f"  reopen: crm watermark={wm}, v1 = {v1r[0]} 条 (sha256={v1r[1]}) —— 全部从盘上读回")
        check("重启后 watermark 与快照幸存", wm == 9 and v1r == (6, "4599c15439c026c8"))
        check("重启前后 catalog digest 不变", catalog_digest(con) == d_before)

        print("\n[5] 新源 web_log 首同 + 增量物化 v2（跨级锚 + 内层对账）")
        web = SourceSystem("web_log", "web-token-demo")
        for i, r in enumerate(BATCH3): web.push("insert", r, updated_at=15 + 10 * i)
        n, b = sync_source(con, "web_log", web, secrets)
        v2 = build_version(con, "sft-support")
        print(f"  {b}: {n} events（web_log 首同 = 全量，其 watermark 从 0 起）")
        print(f"  {v2['version']}: curated {v2['n']} 条 (sha256={v2['sha256']})，本轮新增拦截: {v2['problems']}")
        check("v2 = 8 条，新增拦截恰 1 条跨源重复", v2["n"] == 8 and v2["problems"] == ["重复 id: t007"])
        check("跨级锚: v2 digest == L0 v2 a12337250f5d4d79", v2["sha256"] == "a12337250f5d4d79")
        _, rd = full_replay(con)
        check("内层对账: v2 增量 == 全量重放", rd == v2["sha256"])

        print("\n[6] 外层对账：源全量导出 vs raw 账目（接入完备性的保险丝）")
        for s, sys_ in (("crm", crm), ("web_log", web)):
            div = reconcile_source_to_raw(con, s, sys_)
            print(f"  {s}: 源侧真相 {len(sys_.export_full())} 条 vs raw 账目 → 分歧 {len(div)} 条")
            check(f"{s} 源↔raw 对账: 0 分歧", div == [])

        print("\n[7] wave3：update（回填时间戳）+ insert → lsn 游标增量拉取 → v3")
        crm.push("update", T002_FIXED, updated_at=20)  # 回填修正：updated_at=20 早于当前 watermark=90
        crm.push("insert", T011, updated_at=100)
        n, b = sync_source(con, "crm", crm, secrets)
        lsns = [e[0] for e in con.execute("SELECT lsn FROM raw_records WHERE batch_id=? ORDER BY lsn", (b,))]
        print(f"  {b}: 增量拉取 {n} events (lsn {lsns}) —— 只拉游标之后，不重扫世界")
        check("lsn 游标增量: 恰 2 events (lsn 10, 11)", lsns == [10, 11])
        v3 = build_version(con, "sft-support")
        ids3 = [r[0] for r in con.execute(
            "SELECT rid FROM curated_records WHERE dataset='sft-support' AND version='v3' ORDER BY pos")]
        t002 = json.loads(con.execute(
            "SELECT payload FROM curated_records WHERE dataset='sft-support' AND version='v3' AND rid='t002'").fetchone()[0])
        print(f"  {v3['version']}: curated {v3['n']} 条 (sha256={v3['sha256']})；t002 原位更新 → '{t002['text']}'（位置 {ids3.index('t002')} 不变）")
        check("v3 = 9 条，t002 原位覆写且位置不变（merge 语义）",
              v3["n"] == 9 and ids3.index("t002") == 1 and t002["text"].endswith("order #4217"))
        _, rd = full_replay(con)
        check("内层对账: v3 增量 == 全量重放（update 语义双路径一致）", rd == v3["sha256"])
        check("PII 投影: 全部版本 curated 无 phone",
              all("phone" not in json.loads(r[0]) for r in con.execute("SELECT payload FROM curated_records")))

        print("\n[8] 血缘变 SQL：t007 从哪来？v3 由哪些 batch 派生？")
        lin = con.execute("SELECT source, lsn, batch_id FROM raw_records WHERE rid='t007' ORDER BY source, lsn").fetchall()
        for s, l, bb in lin: print(f"  t007 ← {s} lsn{l} ({bb})")
        check("跨源重复血缘: t007 恰 2 条 raw 记录（crm + web_log）",
              len(lin) == 2 and {x[0] for x in lin} == {"crm", "web_log"})
        fb = json.loads(con.execute(
            "SELECT from_batches FROM curated_versions WHERE dataset='sft-support' AND version='v3'").fetchone()[0])
        srcs = [r[0] for r in con.execute(
            f"SELECT DISTINCT source FROM batches WHERE batch_id IN ({','.join('?' * len(fb))})", fb)]
        print(f"  v3.from_batches = {fb}（源: {sorted(srcs)}）")
        check("v3 派生自全部 3 个 batch / 2 个源", fb == ["b001", "b002", "b003"] and sorted(srcs) == ["crm", "web_log"])

        print("\n[9] 成本视角：增量物化 vs 全量重算（toy 数字，讲机制不讲规模）")
        tot_b, tot_n = con.execute("SELECT SUM(bytes), SUM(n) FROM batches").fetchone()
        new_b, new_n = con.execute("SELECT bytes, n FROM batches WHERE batch_id='b003'").fetchone()
        print(f"  全量重算扫 {tot_n} events / {tot_b} B；v3 增量只扫 {new_n} events / {new_b} B（{100.0 * new_b / tot_b:.1f}% 字节）")
        check("增量扫描字节 < 全量扫描字节", new_b < tot_b)

        print("\n[10] 反例探针：updated_at 游标的盲区 + 对账捕获 + 回补自愈（独立 scratch catalog）")
        pcon = open_catalog(f"{tmp}/probe_catalog.db")
        register_source(pcon, "crm", "updated_at", secrets, "crm-token-demo")
        pcrm = SourceSystem("crm", "crm-token-demo")
        for i, r in enumerate(BATCH1 + BATCH2): pcrm.push("insert", r, updated_at=10 * (i + 1))
        sync_source(pcon, "crm", pcrm, secrets)  # 首同（wm=0 全量），updated_at 游标 → 90
        build_version(pcon, "sft-support")
        pcrm.push("update", T002_FIXED, updated_at=20)  # 同一回填修正：updated_at=20 < wm=90
        pcrm.push("insert", T011, updated_at=100)
        n, b = sync_source(pcon, "crm", pcrm, secrets)
        got = [e[0] for e in pcon.execute("SELECT lsn FROM raw_records WHERE batch_id=? ORDER BY lsn", (b,))]
        rawlsns = {r[0] for r in pcon.execute("SELECT lsn FROM raw_records WHERE source='crm'")}
        missed = [e["lsn"] for e in pcrm.changes_since(0, "lsn") if e["lsn"] not in rawlsns]
        print(f"  updated_at 游标 (wm=90): 只拉到 lsn {got} —— 回填 update lsn{missed[0]}（updated_at=20）被静默漏掉")
        check("盲区复现: updated_at 游标漏掉回填 update", got == [11] and missed == [10])
        pv2 = build_version(pcon, "sft-support")
        _, prd = full_replay(pcon)
        check("内层对账仍绿——raw 自己就缺事件，重放当然一致（内层对账的边界）", prd == pv2["sha256"])
        div = reconcile_source_to_raw(pcon, "crm", pcrm)
        print(f"  外层对账: 源全量导出 vs raw → 分歧 {div}（源侧已修正的 t002，raw 里还是旧的）")
        check("外层对账捕获分歧: 恰 t002", div == ["t002"])
        n, b = sync_source(pcon, "crm", pcrm, secrets, cursor_kind="lsn", force_wm=0)  # 回补 = 全量重拉，PK 去重
        pv3 = build_version(pcon, "sft-support")
        div2 = reconcile_source_to_raw(pcon, "crm", pcrm)
        _, prd3 = full_replay(pcon)
        print(f"  回补自愈: 全量重拉补进 {n} event ({b}) → 重建 {pv3['version']}（{pv3['n']} 条，t002 已修正）→ 双层对账全绿")
        check("自愈后: 分歧清零且内外对账一致", div2 == [] and prd3 == pv3["sha256"] and pv3["n"] == 7)
        pcon.close()

        print(f"\ncatalog digest: {catalog_digest(con)}  (5 张表的逻辑哈希；SQLite 文件字节不参与，输出确定)")
        check("catalog 已落盘且非空", os.path.getsize(db) > 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")

if __name__ == "__main__":
    main()
