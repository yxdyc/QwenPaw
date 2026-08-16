#!/usr/bin/env python3
"""nano-data-platform L0 — 湖仓分层（raw/curated）+ infra-as-code 状态管理的纯 Python 本质模拟。

它在模拟真实系统的哪一面（L0 验收标准，L0 验收契约）：
  [1] raw zone 不可变、只追加、带血缘（source/batch/sha256）——湖仓 bronze 层的本质；
  [2] curated zone = 从 raw 派生的版本化快照（snapshot），质量门是晋升硬门槛——Iceberg/Delta snapshot 思想；
  [3] 声明式期望状态 + plan/apply + state 文件，幂等且最小 diff——Terraform 状态管理本质（无 HCL，HCL 到 L2）；
  [4] 治理 first-class：secrets manager、最小权限 ACL（default-deny）、成本账本（toy 价格，非真实云价）。
刻意不模拟：并发/事务、真实存储引擎、schema evolution、分布式 catalog——见 README 阶梯表。
零依赖（纯标准库），CPU 秒级；输出确定（逻辑时钟，无 wall-clock / 无随机），复跑逐字节一致。
"""
import hashlib, json, shutil, tempfile

CHECKS = []
def check(name, cond):
    CHECKS.append(bool(cond))
    print(f"  [check {len(CHECKS):02d}] {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: raise SystemExit("self-check failed: " + name)

class Clock:  # 逻辑时钟：确定性输出，复跑可收敛
    t = 0
    @classmethod
    def tick(cls):
        cls.t += 1
        return cls.t

# ---- [4a] secrets manager 本质：代码里只有凭据名，值在集中存储，绝不硬编码 ----
class SecretStore:
    def __init__(self): self._s = {}
    def put(self, name, value): self._s[name] = value
    def get(self, name):
        if name not in self._s: raise KeyError(f"secret '{name}' 未注册 —— 拒绝接入而非静默失败")
        return self._s[name]

# ---- [1] raw zone：只有 append 接口，没有 update/delete —— 不可变由接口保证 ----
class RawZone:
    def __init__(self): self.batches = []
    def ingest(self, source, records, credential, secrets):
        if credential != secrets.get(f"{source}/credential"):
            raise PermissionError(f"ingest denied: source '{source}' 凭据错误 —— 认证是接入边界第一道闸")
        payload = json.dumps(records, sort_keys=True, ensure_ascii=False).encode()
        b = dict(batch_id=f"b{len(self.batches) + 1:03d}", source=source, ingested_at=Clock.tick(),
                 n=len(records), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()[:16],
                 records=list(records))
        self.batches.append(b)
        return b

# ---- [2a] 质量门：raw→curated 晋升的硬门槛（挡住 = 不落层，不是警告后放行） ----
REQUIRED = ("id", "text", "label")
def quality_gate(records):
    kept, problems, seen = [], [], set()
    for r in records:
        if any(r.get(k) in (None, "") for k in REQUIRED):
            problems.append(f"必填字段缺失/为空: {r.get('id', '?')}")
        elif r["id"] in seen:
            problems.append(f"重复 id: {r['id']}")
        else:
            seen.add(r["id"]); kept.append(r)
    return kept, problems

# ---- [2b] curated zone：版本化快照，只从 raw 派生；旧版本保留 = time travel ----
class CuratedZone:
    def __init__(self): self.versions = {}
    def build(self, dataset, raw):
        recs = [r for b in raw.batches for r in b["records"]]
        kept, problems = quality_gate(recs)
        kept = [{k: v for k, v in r.items() if k != "phone"} for r in kept]  # PII 投影：phone 不落 curated
        vs = self.versions.setdefault(dataset, [])
        body = json.dumps(kept, sort_keys=True, ensure_ascii=False).encode()
        v = dict(version=f"v{len(vs) + 1}", built_at=Clock.tick(), n=len(kept),
                 sha256=hashlib.sha256(body).hexdigest()[:16], records=kept, problems=problems)
        vs.append(v)
        return v
    def read_version(self, dataset, version):
        for v in self.versions[dataset]:
            if v["version"] == version: return v
        raise KeyError(f"{dataset}@{version} not found")

# ---- [3] 声明式期望状态 + plan/apply + state 文件（infra-as-code 本质，无 HCL） ----
class Platform:
    def __init__(self, state_path):
        self.state_path = state_path
        self.state = {"datasets": {}, "grants": {}}
    def plan(self, desired):
        acts = []
        for ds, spec in sorted(desired["datasets"].items()):
            if ds not in self.state["datasets"]: acts.append(("create_dataset", ds))
            elif self.state["datasets"][ds] != spec: acts.append(("update_dataset", ds))
        for ds in sorted(set(self.state["datasets"]) - set(desired["datasets"])):
            acts.append(("drop_dataset", ds))
        for who, perms in sorted(desired["grants"].items()):
            if self.state["grants"].get(who) != perms: acts.append(("set_grant", who))
        return acts
    def apply(self, desired):
        acts = self.plan(desired)
        for op, key in acts:
            if op in ("create_dataset", "update_dataset"): self.state["datasets"][key] = desired["datasets"][key]
            elif op == "drop_dataset": del self.state["datasets"][key]
            else: self.state["grants"][key] = desired["grants"][key]
        body = json.dumps(self.state, sort_keys=True, ensure_ascii=False, indent=1).encode()
        with open(self.state_path, "wb") as f: f.write(body)
        return acts, len(body)
    # ---- [4b] 最小权限消费：default-deny，训练/检索只暴露 curated 快照 ----
    def consume(self, who, dataset, version, curated):
        if dataset not in self.state["grants"].get(who, {}).get("read", []):
            raise PermissionError(f"ACL deny: '{who}' 无 '{dataset}' 读授权 (default-deny)")
        return curated.read_version(dataset, version)

# ---- [4c] 成本账本：存储成本 first-class。单价 = toy coin/(B·月)，教学设定非真实云价 ----
PRICE = {"raw": 1.0, "curated": 3.0}  # toy 设定：curated 含质量保障+索引，单价 3x（真实价目见 tutorial §7）
def cost_report(raw, curated):
    rb = sum(b["bytes"] for b in raw.batches)
    cb = sum(len(json.dumps(v["records"], sort_keys=True, ensure_ascii=False).encode())
             for vs in curated.versions.values() for v in vs)
    return rb, cb, rb * PRICE["raw"] + cb * PRICE["curated"]

# ---- demo fixture：客服工单。刻意埋 3 类缺陷（重复/空字段/跨源重复）+ PII（phone，明显假号） ----
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

def main():
    print("== nano-data-platform L0: 湖仓分层 + infra-as-code 状态管理（纯 Python 本质模拟） ==")
    tmp = tempfile.mkdtemp(prefix="nano_dp_L0_")
    try:
        secrets, raw, curated = SecretStore(), RawZone(), CuratedZone()
        plat = Platform(f"{tmp}/state.json")
        print("\n[1] 声明式期望状态 → plan/apply（infra-as-code：先看 diff 再动手，幂等）")
        desired = {"datasets": {"sft-support": {"from": "raw", "gate": "required_fields+dedup", "pii_drop": ["phone"]}},
                   "grants": {"trainer": {"read": ["sft-support"]}, "retrieval_svc": {"read": ["sft-support"]},
                              "ingestor": {"write": ["raw"]}}}
        acts, sb = plat.apply(desired)
        for op, key in acts: print(f"  plan: {op:15s} {key}")
        print(f"  apply: {len(acts)} actions 落盘, state 文件 {sb} B（此后 drift 检测的依据）")
        check("首次 apply = create_dataset + 3 set_grant", len(acts) == 4)
        check("二次 apply 幂等: plan = 0 actions", plat.apply(desired)[0] == [])
        print("\n[2] 接入：凭据来自 secrets manager；raw 只追加、带血缘")
        secrets.put("crm/credential", "crm-token-demo")
        b1 = raw.ingest("crm", BATCH1, "crm-token-demo", secrets)
        print(f"  {b1['batch_id']} source={b1['source']} n={b1['n']} sha256={b1['sha256']} at={b1['ingested_at']}（逻辑时钟）")
        try:
            raw.ingest("crm", BATCH1, "wrong-or-stolen-token", secrets)
            check("错误凭据必须被拒", False)
        except PermissionError as e:
            print(f"  denied as expected: {e}"); check("错误凭据必须被拒", True)
        raw.ingest("crm", BATCH2, "crm-token-demo", secrets)
        check("raw 共 9 条 / 2 batches（crm 源）", sum(b["n"] for b in raw.batches) == 9 and len(raw.batches) == 2)
        print("\n[3] 质量门 + 分层派生：curated v1（坏数据挡在层外）")
        v1 = curated.build("sft-support", raw)
        for p in v1["problems"]: print(f"  gate 拦截: {p}")
        print(f"  {v1['version']}: raw {sum(b['n'] for b in raw.batches)} 条 → curated {v1['n']} 条 (sha256={v1['sha256']})")
        check("v1: 9→6，拦截 3 条（重复/空 label/空 text）", v1["n"] == 6 and len(v1["problems"]) == 3)
        check("PII 投影: phone 不落 curated", all("phone" not in r for r in v1["records"]))
        print("\n[4] 增量接入（新源 web_log）→ 快照 v2：训练可复现 = 钉住版本号")
        secrets.put("web_log/credential", "web-token-demo")
        raw.ingest("web_log", BATCH3, "web-token-demo", secrets)
        v2 = curated.build("sft-support", raw)
        print(f"  {v2['version']}: raw {sum(b['n'] for b in raw.batches)} 条 → curated {v2['n']} 条（新增拦截: {v2['problems'][-1]}）")
        check("v2: 12→8，较 v1 多拦 1 条跨源重复", v2["n"] == 8 and len(v2["problems"]) == 4)
        pinned = plat.consume("trainer", "sft-support", "v1", curated)
        latest = plat.consume("trainer", "sft-support", "v2", curated)
        print(f"  trainer 钉住 v1 → {pinned['n']} 条；新任务用 v2 → {latest['n']} 条（两版本共存，互不污染）")
        check("快照钉住: v1 不受 v2 影响", pinned["n"] == 6 and latest["n"] == 8)
        print("\n[5] 最小权限消费：default-deny")
        try:
            plat.consume("intern", "sft-support", "v2", curated)
            check("未授权消费者必须被拒", False)
        except PermissionError as e:
            print(f"  denied as expected: {e}"); check("未授权消费者必须被拒", True)
        check("trainer 只有 curated 读权 / ingestor 只有 raw 写权",
              plat.state["grants"]["trainer"] == {"read": ["sft-support"]}
              and plat.state["grants"]["ingestor"] == {"write": ["raw"]})
        print("\n[6] 成本账本（toy coin/(B·月)：raw 1.0 / curated 3.0，教学设定非真实云价）")
        rb, cb, coins = cost_report(raw, curated)
        print(f"  raw {rb} B ×1.0 + curated(v1+v2) {cb} B ×3.0 = {coins:.0f} toy-coins/月")
        check("curated 单条字节 < raw 单条字节（PII 投影生效）",
              cb / (v1["n"] + v2["n"]) < rb / sum(b["n"] for b in raw.batches))
        print("\n[7] drift 与最小 diff：新增一条授权，plan 恰好 1 个 action（不重建世界）")
        desired["grants"]["analyst"] = {"read": ["sft-support"]}
        acts3, _ = plat.apply(desired)
        for op, key in acts3: print(f"  plan: {op:15s} {key}")
        check("最小 diff = 恰好 1 个 set_grant", acts3 == [("set_grant", "analyst")])
        state_bytes = open(plat.state_path, "rb").read()
        print(f"\nplatform state digest: {hashlib.sha256(state_bytes).hexdigest()[:16]}  (state 文件 {len(state_bytes)} B @ tempdir)")
        check("state 文件已落盘且非空", len(state_bytes) > 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nself-check: {sum(CHECKS)}/{len(CHECKS)} PASS")

if __name__ == "__main__":
    main()
