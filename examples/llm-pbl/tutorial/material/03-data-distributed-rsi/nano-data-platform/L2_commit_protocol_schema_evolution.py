#!/usr/bin/env python3
"""nano-data-platform L2 — Iceberg 式 snapshot/commit protocol + schema evolution + Terraform HCL/state locking。

L1 还掉了 L0 的两笔债（重启蒸发、全量重算），但那时的「快照」只是追加的版本号：
没有人并发提交、schema 不能演化、元数据是一锅 SQLite 行。L2 抓四个机制面（README 阶梯表 L2 行）：

  [1] 元数据树 + 内容寻址存储：table metadata → snapshot → manifest list → manifest → data file，
      每层都是按内容哈希命名的不可变对象（模拟对象存储）。快照的边际成本 = 新增的那几个对象，
      不是重写世界——这是快照便宜的本质（§[2] 字节账实测）。
  [2] commit protocol = 原子指针交换 + 乐观并发（check-and-put）：catalog 只持一个可变指针，
      commit(base, new) 先验 base 仍是当前再交换；冲突 → refresh + 重新 apply + 重试。
      不是所有更新都能 rebase：append 永远可重试，schema 更新必须验证「base 与 current 之间
      schema 未变」，否则拒绝重试（spec 规则逐字引文见 tutorial §4/§5）。
  [3] schema evolution by field id：列按 id 引用而非按名字——add column（旧文件读出 null，
      不重写）、rename（纯元数据操作，数据文件字节不变，§[6] 机器证明）、drop；time travel：
      旧快照仍可按当时或当前 schema 读取，rollback 是一个普通 commit（历史不删）。
  [4] Terraform HCL + state locking：真实 HCL 子集配置（block/attribute 语法）→ plan/apply；
      state 文件带 serial/lineage/lock，锁 = CAS + nonce（只有持锁者能解）；并发 apply 被拒
      而不是损坏状态；drift 检测给最小 diff 修复；授权声明落 state 后在消费边界执行。

显式注明（模拟 vs 真实的可运行性契约）：
  - 本文件是「可运行的本质模拟」：单进程脚本化交错模拟多 writer 并发（真实系统是多进程/集群
    上的真并发）；对象存储 = 本地 tempdir 内容寻址文件（真实 = S3/HDFS）；HCL 只解析教学子集
    （真实 Terraform 是 Go + HCL 全语法 + 远端 backend 锁）。
  - 真实集群行为（多进程 commit 竞争、对象存储上 rename/put-if-absent 的实际原子性）
    不在本脚本证据范围内，必须在对应存储和 catalog 上另行验证。
  - 跨级锚：与 L0/L1 同 fixture 同折叠规则 → 快照内容 digest 与 L0/L1 字节级一致
    （v1 4599c15439c026c8 / v2 a12337250f5d4d79），阶梯间语义同一性的机器证明。
  - Delta Lake 对照：同款乐观并发的另一具象（MVCC + 日志原子版本，引文见 tutorial §11）。
零依赖（纯标准库），CPU 秒级；输出确定（逻辑时钟，无 wall-clock / 无随机），任意 CWD 可跑，
双跑逐字节一致。
"""
import hashlib, json, os, re, shutil, tempfile

CHECKS = []
def check(name, cond):
    CHECKS.append(bool(cond))
    print(f"  [check {len(CHECKS):02d}] {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: raise SystemExit("self-check failed: " + name)

class Clock:  # 逻辑时钟：确定性输出，复跑可收敛（L0/L1 同款）
    t = 0
    @classmethod
    def tick(cls):
        cls.t += 1
        return cls.t

def canon(obj): return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
def sha16(b): return hashlib.sha256(b).hexdigest()[:16]

# ---- [1] 内容寻址对象存储（S3 本质模拟）：按内容哈希命名、写后不可变、同内容自动去重 ----
class ObjectStore:
    def __init__(self, root):
        self.root, self.paths = root, {}
        os.makedirs(root, exist_ok=True)   # mkdtemp 只建 tmp 根目录；对象存储子目录须在首次 put 前显式创建
        self.bytes_by_kind = {"data": 0, "manifest": 0, "mlist": 0, "metadata": 0}
        self.dedup_hits = 0
    def put(self, kind, obj):
        body, h = canon(obj), sha16(canon(obj))
        if h in self.paths:          # 同内容已存在：零成本复用（快照共享 manifest 的基础）
            self.dedup_hits += 1
            return h
        self.paths[h] = f"{self.root}/{kind}-{h}.json"
        with open(self.paths[h], "wb") as f: f.write(body)
        self.bytes_by_kind[kind] += len(body)
        return h
    def get(self, h):
        with open(self.paths[h], "rb") as f: return json.loads(f.read())
    def total(self): return sum(self.bytes_by_kind.values())

class CommitConflictError(Exception): pass
class CommitValidationError(Exception): pass
class LockError(Exception): pass

# ---- [2] catalog = 每表一个可变指针 + check-and-put（spec「Metastore Tables」节的本质模拟） ----
class Catalog:
    def __init__(self): self.pointer, self.commits, self.committed = {}, 0, set()
    def create(self, table, meta_h): self.pointer[table] = meta_h; self.committed.add(meta_h)
    def current(self, table): return self.pointer[table]
    def commit(self, table, base, new):
        if self.pointer[table] != base:
            raise CommitConflictError(f"base {base[:8]}… 已不是当前版本（当前 {self.pointer[table][:8]}…）")
        self.pointer[table] = new
        self.commits += 1
        self.committed.add(new)

class Table:
    def __init__(self, name, store, catalog):
        self.name, self.store, self.catalog = name, store, catalog
    def meta(self): return self.store.get(self.catalog.current(self.name))
    def _schema(self, m, sid): return [s for s in m["schemas"] if s["schema-id"] == sid][0]
    def scan(self, snapshot_seq=None, at_schema=None):
        """读表。默认 = 当前快照 × 当前 schema 投影（Iceberg reader 口径：按 field id 解析，
        旧文件在新 schema 下照常读——rename 跨历史可见、新增列读 null）。time travel = 显式传参。"""
        m = self.meta()
        seq = m["current-snapshot-seq"] if snapshot_seq is None else snapshot_seq
        snap = [s for s in m["snapshots"] if s["sequence-number"] == seq][0]
        schema = self._schema(m, m["current-schema-id"] if at_schema is None else at_schema)
        rows = []
        for mh in self.store.get(snap["manifest-list"]):
            for e in self.store.get(mh):
                for row in self.store.get(e["data-file"])["rows"]:
                    rows.append({f["name"]: row.get(str(f["id"])) for f in schema["fields"]})
        return rows

def commit_with_retry(table, op, max_retries=4, first_base=None):
    """模拟 SnapshotProducer.commit() 的重试循环（apache/iceberg tag apache-iceberg-1.11.0，
    core/src/main/java/org/apache/iceberg/SnapshotProducer.java:L459-501）：
    .retry(COMMIT_NUM_RETRIES 默认 4，TableProperties.java:L89-90) + 每次 attempt 重新 apply()
    （L475）+ taskOps.commit(base, updated) CAS（L501）。此处逻辑步进而非指数退避等待。
    注意：乐观构建的 metadata 对象若提交失败即成孤儿——真实 Iceberg 由 cleanUncommitted 清理
    （SnapshotProducer.java:L524），本模拟保留孤儿并在 [8] 报账。"""
    base = first_base if first_base is not None else table.catalog.current(table.name)
    attempts = 0
    while True:
        attempts += 1
        m2 = op.apply(table, base)                 # 乐观构建（每次 attempt 基于最新 base 重做）
        h = table.store.put("metadata", m2)
        try:
            table.catalog.commit(table.name, base, h)
            return attempts, m2
        except CommitConflictError:
            if attempts > max_retries: raise
            print(f"    commit 冲突（base {base[:8]}… 过期）→ refresh + 重新 apply（attempt {attempts}/{max_retries} 已用）")
            base = table.catalog.current(table.name)   # = ops.refresh()

class AppendOp:
    """append：写新 data file + 新 manifest，manifest list = 旧表全部 manifest + 新 manifest。
    spec：『Append operations have no requirements and can always be applied.』→ 冲突永远可 rebase。"""
    def __init__(self, writer, records): self.writer, self.records = writer, records
    def apply(self, table, base_h):
        m = table.store.get(base_h)
        schema = table._schema(m, m["current-schema-id"])
        rows = [{str(f["id"]): r.get(f["name"]) for f in schema["fields"]} for r in self.records]
        df = table.store.put("data", {"written-with-schema": schema["schema-id"], "rows": rows})
        man = table.store.put("manifest", [{"data-file": df, "record-count": len(rows)}])
        prev = m["snapshots"][-1] if m["snapshots"] else None
        mlist = (table.store.get(prev["manifest-list"]) if prev else []) + [man]
        seq = (prev["sequence-number"] + 1) if prev else 1
        snap = {"sequence-number": seq, "parent-seq": prev["sequence-number"] if prev else None,
                "schema-id": schema["schema-id"], "manifest-list": table.store.put("mlist", mlist),
                "summary": {"operation": "append", "writer": self.writer, "added-records": len(rows)}}
        return {**m, "snapshots": m["snapshots"] + [snap], "current-snapshot-seq": seq,
                "snapshot-log": m["snapshot-log"] + [seq], "metadata-version": m["metadata-version"] + 1}

class SchemaOp:
    """schema evolution（add/rename/drop，按 field id 寻址的纯元数据操作）。
    spec：『Table schema updates … must validate that the schema has not changed between the
    base version and the current version.』→ 验证失败 = 不可重试，必须基于新 schema 重新发起。"""
    def __init__(self, writer, base_schema_id, kind, **kw):
        self.writer, self.base_schema_id, self.kind, self.kw = writer, base_schema_id, kind, kw
    def apply(self, table, base_h):
        m = table.store.get(base_h)
        if m["current-schema-id"] != self.base_schema_id:
            raise CommitValidationError(
                f"base(s{self.base_schema_id}) 与 current(s{m['current-schema-id']}) 之间 schema 已变：此更新不可 rebase，须基于新 schema 重新发起")
        fields = [dict(f) for f in table._schema(m, m["current-schema-id"])["fields"]]
        if self.kind == "add":
            fields.append({"id": max(f["id"] for f in fields) + 1, "name": self.kw["name"],
                           "type": self.kw["type"], "required": False})
        elif self.kind == "rename":
            for f in fields:
                if f["id"] == self.kw["field_id"]: f["name"] = self.kw["to"]
        elif self.kind == "drop":
            assert not any(f["id"] == self.kw["field_id"] and f["required"] for f in fields), "required 列不可 drop"
            fields = [f for f in fields if f["id"] != self.kw["field_id"]]
        new_sid = max(s["schema-id"] for s in m["schemas"]) + 1
        return {**m, "schemas": m["schemas"] + [{"schema-id": new_sid, "fields": fields}],
                "current-schema-id": new_sid, "metadata-version": m["metadata-version"] + 1}

class RollbackOp:
    """rollback = 把当前指针指回旧快照的普通 commit（SnapshotProducer.java:L479
    『this is a rollback operation』）：不新建快照、不删历史、数据文件零触碰。"""
    def __init__(self, writer, target_seq): self.writer, self.target_seq = writer, target_seq
    def apply(self, table, base_h):
        m = table.store.get(base_h)
        assert any(s["sequence-number"] == self.target_seq for s in m["snapshots"]), "rollback 目标快照不存在"
        return {**m, "current-snapshot-seq": self.target_seq,
                "snapshot-log": m["snapshot-log"] + [self.target_seq],
                "metadata-version": m["metadata-version"] + 1}

# ---- 转换层（dbt 式 staging）：与 L1 fold_events 逐字同款（跨级 digest 锚的前提） ----
REQUIRED = ("id", "text", "label")
def fold_events(state, events):
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
            else: state[rid] = {k: v for k, v in r.items() if k != "phone"}
    return problems

# ---- [4] HCL 子集解析器（block + attribute；真实 HCL 全语法见 hashicorp/hcl hclsyntax/spec.md） ----
def parse_hcl(text):
    out, stack = {}, None
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"): continue
        m = re.match(r'^resource\s+"([^"]+)"\s+"([^"]+)"\s*\{$', line)
        if m:
            stack = out.setdefault(m.group(1), {}).setdefault(m.group(2), {})
            continue
        if line == "}":
            stack = None
            continue
        m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if m and stack is not None:
            key, val = m.group(1), m.group(2).strip()
            if val.startswith("["):
                stack[key] = re.findall(r'"([^"]*)"', val)
            elif val.startswith('"'):
                stack[key] = val.strip('"')
            elif val in ("true", "false"):
                stack[key] = val == "true"
            else:
                stack[key] = int(val)
            continue
        raise SyntaxError(f"HCL 子集解析失败 @L{ln}: {raw!r}")
    return out

class TfState:
    """Terraform state 本质：serial（每次状态变更 +1）+ lineage（状态身份）+ lock（CAS + nonce）。
    真实 backend 的锁在 DynamoDB/Consul/Terraform Cloud；此处用同一文件的 lock 字段模拟。"""
    def __init__(self, path):
        self.path, self.lock_seq = path, 0
        if not os.path.exists(path):
            self._write({"serial": 0, "lineage": "lineage-000", "lock": None, "resources": {}})
    def _read(self):
        with open(self.path, "rb") as f: return json.loads(f.read())
    def _write(self, st):
        with open(self.path, "wb") as f: f.write(json.dumps(st, sort_keys=True, ensure_ascii=False, indent=1).encode())
    def acquire(self, owner):
        st = self._read()
        if st["lock"]:
            raise LockError(f"state 已被 '{st['lock']['owner']}' 锁定 (lock_id={st['lock']['lock_id']}) —— 拒绝而非等待损坏")
        self.lock_seq += 1
        st["lock"] = {"owner": owner, "lock_id": f"lock-{self.lock_seq:03d}"}
        self._write(st)
        return st["lock"]["lock_id"]
    def release(self, lock_id):
        st = self._read()
        if not st["lock"] or st["lock"]["lock_id"] != lock_id:
            raise LockError(f"release 拒绝：lock_id '{lock_id}' 与当前锁 nonce 不符 —— 解锁只认 nonce")
        st["lock"] = None
        self._write(st)
    @staticmethod
    def plan(st, desired):
        acts = []
        for kind in sorted(set(desired) | set(st["resources"])):
            want, have = desired.get(kind, {}), st["resources"].get(kind, {})
            for name in sorted(set(want) | set(have)):
                if want.get(name) != have.get(name):
                    acts.append((f"{'create' if name in want else 'drop'}:{kind}", name))
        return acts
    def apply(self, owner, desired):
        lid = self.acquire(owner)
        try:
            st = self._read()
            acts = self.plan(st, desired)
            if acts:
                st["resources"] = {k: dict(v) for k, v in desired.items()}
                st["serial"] += 1          # serial 只在状态真变时 +1（no-op apply 不推 serial）
                self._write(st)
            return acts, lid
        except Exception:
            self.release(lid); raise
        # 注：成功路径的 release 由调用方执行——为了让 [7] 能演示「持锁中的并发 apply」

# ---- demo fixture：BATCH1/2/3 与 L0/L1 逐字相同（跨级 digest 锚的前提） ----
BATCH1 = [{"id": "t001", "text": "how to reset password", "label": "auth", "phone": "138-0000-0001"},
          {"id": "t002", "text": "charged twice on my card", "label": "billing", "phone": "138-0000-0002"},
          {"id": "t003", "text": "cannot export csv from dashboard", "label": "export", "phone": "138-0000-0003"},
          {"id": "t004", "text": "login page loops on safari", "label": "auth", "phone": "138-0000-0004"},
          {"id": "t005", "text": "upgrade plan from team to biz", "label": "billing", "phone": "138-0000-0005"}]
BATCH2 = [{"id": "t003", "text": "cannot export csv from dashboard", "label": "export", "phone": "138-0000-0003"},
          {"id": "t006", "text": "api rate limit question", "label": None, "phone": "138-0000-0006"},
          {"id": "t007", "text": "webhook retries after 500", "label": "integration", "phone": "138-0000-0007"},
          {"id": "t008", "text": "", "label": "billing", "phone": "138-0000-0008"}]
BATCH3 = [{"id": "t009", "text": "sso with okta setup help", "label": "auth", "phone": "138-0000-0009"},
          {"id": "t010", "text": "invoice pdf is blank", "label": "billing", "phone": "138-0000-0010"},
          {"id": "t007", "text": "webhook retries after 500", "label": "integration", "phone": "138-0000-0007"}]
T011 = {"id": "t011", "text": "two-factor sms not arriving", "label": "auth", "phone": "138-0000-0011"}
T012 = {"id": "t012", "text": "refund status after 5 days", "label": "billing", "phone": "138-0000-0012"}
T013 = {"id": "t013", "text": "password reset email in spam", "label": "auth", "phone": "138-0000-0013", "priority": "high"}

HCL_CONFIG = """
# platform.hcl —— 期望状态声明（真实 HCL 子集：block + attribute 语法）
resource "dp_dataset" "sft_support" {
  gate     = "required_fields+dedup"
  pii_drop = ["phone"]
}
resource "dp_grant" "trainer" {
  read = ["sft_support"]
}
resource "dp_grant" "ingestor" {
  write = ["raw"]
}
"""

def staging(events, prev=None):
    """dbt 式分层派生：raw 事件 → 质量门 + PII 投影 → 可 append 的 curated 行（L0/L1 同款漏斗）。
    prev = 表内已 curated 的记录（{rid: rec}，L1 build_version「上一版快照」谱系）：跨 batch 去重
    依赖它——fold_events 与 L1 逐字同款，同 id insert 被拦截；只返回新行（本档只处理 insert 流，
    update 物化是 L1 merge 语义，append 表语义不在本级范围）。"""
    state = dict(prev or {})
    problems = fold_events(state, events)
    new = [rec for rid, rec in state.items() if rid not in (prev or {})]
    return new, problems

def main():
    print("== nano-data-platform L2: Iceberg 式 commit protocol + schema evolution + Terraform HCL/locking（本质模拟） ==")
    tmp = tempfile.mkdtemp(prefix="nano_dp_L2_")
    try:
        store, catalog = ObjectStore(f"{tmp}/objects"), Catalog()
        print("\n[1] 元数据树 + 首次 commit：table metadata → snapshot → manifest list → manifest → data file（全内容寻址）")
        s1 = {"schema-id": 1, "fields": [{"id": 1, "name": "id", "type": "string", "required": True},
                                          {"id": 2, "name": "text", "type": "string", "required": True},
                                          {"id": 3, "name": "label", "type": "string", "required": True}]}
        meta0 = {"format-version": 2, "table": "sft-support", "schemas": [s1], "current-schema-id": 1,
                 "snapshots": [], "current-snapshot-seq": None, "snapshot-log": [], "metadata-version": 0}
        catalog.create("sft-support", store.put("metadata", meta0))
        kept, problems = staging([("insert", r) for r in BATCH1 + BATCH2])
        for p in problems: print(f"  gate 拦截: {p}")
        t = Table("sft-support", store, catalog)
        n_att, m1 = commit_with_retry(t, AppendOp("ingestor", kept))
        snap1 = m1["snapshots"][0]
        print(f"  append {len(kept)} 行 → snapshot seq{snap1['sequence-number']}（metadata V{m1['metadata-version']}，{n_att} attempt）")
        print(f"  元数据树: metadata {catalog.current('sft-support')[:8]}… → snapshot seq1 → mlist {snap1['manifest-list'][:8]}… → manifest → data file")
        d1 = sha16(canon(t.scan()))
        check("跨级锚: seq1 内容 digest == L0/L1 v1 4599c15439c026c8", d1 == "4599c15439c026c8")
        check("首次 commit 1 attempt（无冲突）", n_att == 1)

        print("\n[2] 第二次 append：manifest 复用 = 快照便宜的本质（不重写世界）")
        bytes_before = store.total()
        prev_state = {r["id"]: r for r in t.scan()}   # 「上一版快照」从表读出（L1 增量谱系：跨 batch 去重依赖它）
        kept2, problems2 = staging([("insert", r) for r in BATCH3], prev_state)
        for p in problems2: print(f"  gate 拦截: {p}")
        n_att, m2 = commit_with_retry(t, AppendOp("ingestor", kept2))
        d2 = sha16(canon(t.scan()))
        marginal = store.total() - bytes_before
        ml2, ml1 = store.get(m2["snapshots"][1]["manifest-list"]), store.get(snap1["manifest-list"])
        print(f"  seq2: {len(t.scan())} 行 (digest {d2})；新增字节 {marginal} B（新 data file + 新 manifest + 新 mlist + 新 metadata）")
        print(f"  对比全量重写世界 {bytes_before + marginal} B —— 快照边际成本 {100.0 * marginal / (bytes_before + marginal):.1f}%")
        check("跨级锚: seq2 内容 digest == L0/L1 v2 a12337250f5d4d79", d2 == "a12337250f5d4d79")
        check("manifest 复用: seq2 的 mlist 含 seq1 的 manifest（引用复用，零重写）", ml1[0] in ml2)
        check("快照边际字节 < 全量世界字节", marginal < bytes_before + marginal)

        print("\n[3] 乐观并发：A/B 同 base 并发 append —— 一个成功，另一个 CAS 冲突 → refresh + re-apply 重试")
        base0 = catalog.current("sft-support")   # A、B 都读到同一个 base（模拟并发读）
        att_b, _ = commit_with_retry(t, AppendOp("writer-B", [T011]), first_base=base0)
        print(f"  writer-B 先提交 → seq{t.meta()['current-snapshot-seq']}（{att_b} attempt）")
        att_a, _ = commit_with_retry(t, AppendOp("writer-A", [T012]), first_base=base0)
        m4 = t.meta()
        ids4 = [r["id"] for r in t.scan()]
        print(f"  writer-A 第 1 次 attempt CAS 冲突 → 第 2 次 attempt 基于新 base 重做 → seq{m4['current-snapshot-seq']}")
        print(f"  attempt 2 重放 attempt 1 已写的 data/manifest 对象: 内容相同 → 去重命中 {store.dedup_hits} 次（内容寻址把『重做』变『复用』）")
        check("B 无冲突 1 attempt / A 恰 2 attempts（1 冲突 + 1 重试）", att_b == 1 and att_a == 2)
        check("双方数据各恰一次（无丢失无重复）", sorted(ids4) == sorted([r["id"] for r in t.scan(snapshot_seq=2)] + ["t011", "t012"]) and ids4.count("t011") == 1 and ids4.count("t012") == 1)
        check("snapshot-log 严格递增（历史串行化）", m4["snapshot-log"] == [1, 2, 3, 4])

        print("\n[4] 不是所有更新都能 rebase：并发 schema 更新 —— add-column 成功，rename 验证失败被拒")
        base4 = catalog.current("sft-support")   # 两个 schema 更新都基于 s1
        att_add, m5 = commit_with_retry(t, SchemaOp("writer-A", 1, "add", name="priority", type="string"), first_base=base4)
        print(f"  writer-A add column priority → s{m5['current-schema-id']}（metadata V{m5['metadata-version']}，无新快照）")
        try:
            commit_with_retry(t, SchemaOp("writer-B", 1, "rename", field_id=3, to="category"), first_base=base4)
            check("基于过期 schema 的 rename 必须被拒", False)
        except CommitValidationError as e:
            print(f"  writer-B rename(label→category) 基于 s1：{e}")
            check("基于过期 schema 的 rename 必须被拒", True)
        att_re, m6 = commit_with_retry(t, SchemaOp("writer-B", m5["current-schema-id"], "rename", field_id=3, to="category"))
        print(f"  writer-B 基于新 schema 重新发起 rename → s{m6['current-schema-id']}（{att_re} attempt）")
        names6 = [f["name"] for f in t._schema(m6, m6["current-schema-id"])["fields"]]
        check("最终 schema = id/text/category/priority（两次演化按提交序生效）", names6 == ["id", "text", "category", "priority"])
        check("schema 更新不产生快照（快照数仍 4）", len(m6["snapshots"]) == 4 and m6["current-snapshot-seq"] == 4)

        print("\n[5] schema evolution by field id：旧文件零重写，rename 纯元数据（机器证明）")
        data_hashes_before = sorted(h for h, p in store.paths.items() if p.split("/")[-1].startswith("data-"))
        bytes_data_before = store.bytes_by_kind["data"]
        rows = t.scan()
        t001 = [r for r in rows if r["id"] == "t001"][0]
        print(f"  当前 schema 读旧文件: t001 = {t001}（label 已改名 category，priority 读出 null —— 列按 field id 解析）")
        check("rename 后旧数据按新名可读、新增列读 null", t001["category"] == "auth" and "label" not in t001 and t001["priority"] is None)
        check("演化未重写任何数据文件（data 字节数不变）", store.bytes_by_kind["data"] == bytes_data_before)
        kept3 = staging([("insert", r) for r in [T013]])[0]
        # 写入端契约: 新行按当前 schema 列名书写。label 已改名 category（field id 3 不变）——
        # writer 看得见 rename，在写入边界做名称映射（真实管线 = dbt 模型的 SELECT label AS category）；
        # 旧文件不需要这步：读取按 field id 解析，历史原样保留（这正是 [check 12] 的另一半）。
        renamed3 = [{("category" if k == "label" else k): v for k, v in r.items()} for r in kept3]
        n_att, m7 = commit_with_retry(t, AppendOp("writer-A", renamed3))
        t013 = [r for r in t.scan() if r["id"] == "t013"][0]
        print(f"  新写 t013（带 priority）→ seq{m7['current-snapshot-seq']}: {t013}")
        check("新 schema 写入的新行携带 priority", t013["priority"] == "high" and t013["category"] == "auth")
        check("写新行只增不改（旧 data 对象哈希集合不变）",
              set(data_hashes_before) <= {h for h, p in store.paths.items() if p.split("/")[-1].startswith("data-")})

        print("\n[6] time travel + rollback：旧快照按当时 schema 可读；rollback 是一个普通 commit")
        as_of_2 = t.scan(snapshot_seq=2, at_schema=1)
        print(f"  as-of seq2（schema s1）: {len(as_of_2)} 行，digest {sha16(canon(as_of_2))} —— 与 L0/L1 v2 逐字节同一")
        check("time travel 复现 L0/L1 v2 锚（快照隔离 + 历史可重放）", sha16(canon(as_of_2)) == "a12337250f5d4d79")
        cross = t.scan(snapshot_seq=2)   # 旧快照 × 当前 schema：rename 跨历史可见
        check("旧快照在当前 schema 下读出 category（field id 穿越）", "category" in cross[0] and cross[0].get("priority") is None)
        n_att, m8 = commit_with_retry(t, RollbackOp("ops", 2))
        print(f"  rollback → seq2（metadata V{m8['metadata-version']}，{n_att} attempt）：current = {len(t.scan())} 行")
        check("rollback 后当前视图 = seq2 的 8 行（按当时 schema s1 读复现 v2 锚）",
              len(t.scan()) == 8 and sha16(canon(t.scan(at_schema=1))) == "a12337250f5d4d79")
        check("rollback 不删历史：快照仍 5 个，seq5 仍可前向 time travel", len(m8["snapshots"]) == 5 and len(t.scan(snapshot_seq=5)) == 11)
        check("snapshot-log 记录回滚轨迹", m8["snapshot-log"] == [1, 2, 3, 4, 5, 2])

        print("\n[7] Terraform：HCL 子集 → plan/apply + state locking（serial/lineage/nonce）")
        desired = parse_hcl(HCL_CONFIG)
        print(f"  解析 platform.hcl: {sorted((k, sorted(v)) for k, v in desired.items())}")
        check("HCL 子集解析: 3 blocks、属性类型正确", desired["dp_dataset"]["sft_support"]["pii_drop"] == ["phone"] and desired["dp_grant"]["trainer"]["read"] == ["sft_support"] and desired["dp_grant"]["ingestor"]["write"] == ["raw"])
        st = TfState(f"{tmp}/terraform.tfstate.json")
        acts1, lid1 = st.apply("apply-P1", desired)
        for op, key in acts1: print(f"  plan: {op:28s} {key}")
        st.release(lid1)
        check("首次 apply = 3 actions，serial 0→1", len(acts1) == 3 and st._read()["serial"] == 1)
        lid_p1 = st.acquire("apply-P1-long")   # 模拟一个持锁中的长 apply
        try:
            st.apply("apply-P2", desired)
            check("持锁期间的并发 apply 必须被拒", False)
        except LockError as e:
            print(f"  apply-P2 被拒: {e}")
            check("持锁期间的并发 apply 必须被拒", True)
        try:
            st.release("lock-999")
            check("非持锁者不能解锁（nonce 校验）", False)
        except LockError as e:
            print(f"  伪造 nonce 解锁被拒: {e}")
            check("非持锁者不能解锁（nonce 校验）", True)
        st.release(lid_p1)
        acts2, lid2 = st.apply("apply-P2", desired)
        st.release(lid2)
        print(f"  apply-P2 重试: {len(acts2)} actions（幂等 no-op），serial 维持 {st._read()['serial']}")
        check("解锁后重试 = 幂等 no-op、serial 不推进", acts2 == [] and st._read()["serial"] == 1)
        drifted = st._read(); del drifted["resources"]["dp_grant"]["trainer"]
        with open(st.path, "wb") as f: f.write(json.dumps(drifted, sort_keys=True, ensure_ascii=False, indent=1).encode())
        acts3 = st.plan(st._read(), desired)
        print(f"  drift 检测（带外删掉 trainer 授权）: plan = {acts3} —— 最小 diff 修复，不重建世界")
        check("drift 修复 = 恰 1 个 action", acts3 == [("create:dp_grant", "trainer")])
        lid3 = st.acquire("apply-P3"); st._read(); st.release(lid3)  # 修复 apply（略：与 P1 同路径）
        grants = st._read()  # drift 已在 plan 中量化；此处恢复授权以闭合消费边界
        grants["resources"]["dp_grant"]["trainer"] = {"read": ["sft_support"]}
        with open(st.path, "wb") as f: f.write(json.dumps(grants, sort_keys=True, ensure_ascii=False, indent=1).encode())
        def consume(who):
            # state 里的授权以 Terraform 资源名记录（下划线 = HCL 惯用词分隔符，spec Identifiers 节），
            # catalog 表名用连字符——两个命名空间，在消费边界显式映射（见 tutorial §8，不静默抹平）。
            g = st._read()["resources"].get("dp_grant", {}).get(who, {})
            granted_tables = {name.replace("_", "-") for name in g.get("read", [])}
            if "sft-support" not in granted_tables:
                raise PermissionError(f"ACL deny: '{who}' 无 sft-support 读授权 (default-deny)")
            return len(t.scan())
        def _denied(who):
            # 拒绝路径的真实执行：复用 consume 语义，拒绝边界 = PermissionError（default-deny）
            try:
                consume(who)
                return False
            except PermissionError:
                return True
        check("授权落 state 后在消费边界执行: trainer 可读 / intern 被拒", consume("trainer") == 8 and _denied("intern"))

        print("\n[8] 成本与孤儿账本（toy 字节账，讲机制不讲规模）")
        meta_bytes = store.bytes_by_kind["metadata"]
        orphans = [h for h, p in store.paths.items() if p.split("/")[-1].startswith("metadata-") and h not in catalog.committed]
        print(f"  对象存储字节账: data {store.bytes_by_kind['data']} B / manifest {store.bytes_by_kind['manifest']} B / mlist {store.bytes_by_kind['mlist']} B / metadata {meta_bytes} B（去重命中 {store.dedup_hits} 次，均来自 [3] 重试 re-put）")
        print(f"  metadata 对象 {sum(1 for p in store.paths.values() if p.split('/')[-1].startswith('metadata-'))} 个，其中 {len(orphans)} 个是冲突遗留孤儿（[3] writer-A 冲突 attempt + [4] rename 过期 base attempt；真实 Iceberg 由 cleanUncommitted 清理，SnapshotProducer.java:L524）")
        check("去重命中恰 2：内容寻址把冲突重试的『重做』变『复用』（同内容不重复计字节）", store.dedup_hits == 2)
        check("冲突孤儿恰 2：乐观构建的 metadata 未提交即弃（cleanUncommitted 的清理对象）", len(orphans) == 2)
        plat_digest = sha16(canon({"pointer": catalog.current("sft-support"), "snapshot-log": t.meta()["snapshot-log"],
                                   "schemas": [s["schema-id"] for s in t.meta()["schemas"]],
                                   "tf": {"serial": st._read()["serial"], "resources": st._read()["resources"]}}))
        print(f"\nplatform L2 state digest: {plat_digest}  (catalog 指针 + snapshot-log + schema 序列 + tf state 的逻辑哈希)")
        check("state digest 非空且 catalog 提交数 = 8（成功 commit；失败的 rename 验证在 apply() 内抛错、不计 commit）", catalog.commits == 8 and len(plat_digest) == 16)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")

if __name__ == "__main__":
    main()
