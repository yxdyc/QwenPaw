# nano-data-platform L0 — 湖仓分层 + infra-as-code 状态管理（纯 Python 本质模拟）

> **前置**：无。Python 3.10+，纯标准库，CPU 秒级。
> **运行**：`python3 L0_lakehouse_and_iac_state.py`（任意目录可跑，输出确定，复跑逐字节一致）。
> **本文件是 notebook-style 教程**：叙述 + 代码摘录 + 真实运行输出 + 思考题交替推进。

---

## §1 为什么数据平台是 LLM 系统的生命线

训练一个模型，算法论文给你 loss 函数，但**没人给你数据**。真实世界里数据散落在 CRM、日志、爬虫、人工标注、agent 轨迹里，格式各异、质量参差、权限敏感。数据平台（data platform）解决的就是这件事：**把散乱的数据源，变成训练和检索可以安全、可复现、算得起账来消费的东西**。

在 LLM-PBL 的四轨依赖图里（总导航的四轨闭环），03 轨产出的数据喂给 02 预训练 / 01 后训练，04 轨 agent 的运行轨迹又回流 03——这个 data-model co-development（RSI）闭环**能不能在生产环境持续转起来，取决于数据平台这一层**。清洗算子写得再好（nano-data-juicer），如果数据接不进来、版本不可复现、权限管不住、成本算不清，飞轮就是空转。

本模块抓的核心机制链条（课程的数据系统教学约定）：

```
数据接入 → 分层存储（raw/curated）→ 治理（成本/权限/质量）→ 训练/检索消费
```

## §2 L0 模拟真实系统的哪四面

L0 的验收标准是「能口头讲清它在模拟真实系统的哪一面」。本实现模拟四面，刻意不模拟其余（§9 列边界）：

| # | 机制面 | nano 实现 | 真实系统对应 |
|---|--------|-----------|--------------|
| [1] | raw zone：不可变、只追加、带血缘 | `RawZone.ingest` | 对象存储（S3 类）上的湖仓 bronze 层 |
| [2] | curated zone：质量门 + 版本化快照派生 | `quality_gate` + `CuratedZone.build` | Iceberg/Delta 的 snapshot；dbt 的分层 model |
| [3] | 声明式期望状态 + plan/apply + state 文件 | `Platform.plan/apply` | Terraform 的状态管理（本教程无 HCL，HCL 到 L2） |
| [4] | 治理：secrets / 最小权限 ACL / 成本账本 | `SecretStore` / `Platform.consume` / `cost_report` | secrets manager / IAM least-privilege / 成本可观测 |

先跑一遍，建立全局印象（完整输出；以下各节的输出块均从此同一次运行中截取）：

```bash
$ python3 L0_lakehouse_and_iac_state.py
== nano-data-platform L0: 湖仓分层 + infra-as-code 状态管理（纯 Python 本质模拟） ==
...
self-check: 13/13 PASS
```

demo 的剧本：声明一个平台（1 个 dataset + 3 条授权）→ 从两个源接入 3 批共 12 条客服工单（其中刻意埋了重复、空字段、跨源重复三类缺陷和 PII 字段）→ 质量门构建 curated v1/v2 → 训练者钉住版本消费 → 未授权访问被拒 → 成本账本结账 → 配置变更时 plan 给出最小 diff。

> **fixture 声明**：BATCH1/2/3 是内嵌的演示数据（客服工单，phone 字段只使用不可拨号的占位符 `PHONE-DEMO-XX`），刻意埋缺陷以演示质量门。这不是「假数据冒充跑通」——L0 的机制对象就是「缺陷如何被分层治理挡住」，缺陷本身是实验设计的一部分。L1 会用同一小样本复现同一套语义。

---

## §3 机制面 [3]：infra-as-code —— 先声明期望状态，再 plan/apply

先看最反直觉的一点：**平台不是「手动建出来的」，是「声明出来的」**。你不敲命令逐个建 dataset、逐个授权，而是写一份期望状态（desired state），让系统自己去「现实 → 期望」的差量：

```python
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
```

demo 的第一步就是声明平台并 apply：

```text
[1] 声明式期望状态 → plan/apply（infra-as-code：先看 diff 再动手，幂等）
  plan: create_dataset  sft-support
  plan: set_grant       ingestor
  plan: set_grant       retrieval_svc
  plan: set_grant       trainer
  apply: 4 actions 落盘, state 文件 316 B（此后 drift 检测的依据）
  [check 01] PASS  首次 apply = create_dataset + 3 set_grant
  [check 02] PASS  二次 apply 幂等: plan = 0 actions
```

三个本质点，逐个拆开：

**（a）声明式 vs 命令式。** 命令式脚本说「执行 create_dataset(...)」，声明式配置说「我要的世界长这样」。差别在**重复执行**时显现：同一份 desired 再 apply 一次，plan 算出 0 个 action（check 02）——这就是**幂等**。命令式脚本重跑会报「已存在」或重复创建；声明式系统重跑是安全的 no-op。运维上这意味着：配置可以进版本库、可以 review、可以无脑重放。

**（b）plan 在 apply 之前。** plan 是纯函数：`(当前 state, desired) → action 列表`，不动任何真实资源。这给你一个**动手前看 diff** 的机会——生产平台上「apply 前先看 plan」是铁律，因为 plan 列表就是这次变更的爆炸半径。

**（c）state 文件是现实世界的账本。** apply 之后，当前状态序列化落盘（316 B 的 state.json）。Terraform 官方文档对 state 的定位（2026-08-12 抓取原文）：

> "Terraform uses your workspace's state to map real world resources to your configuration" …… "Terraform uses state to determine which changes to make to your infrastructure."
> —— https://developer.hashicorp.com/terraform/language/state

没有 state 文件，下一次 plan 就不知道现实长什么样，只能全量重建。**state 是「差量计算」的前提**——这一点 §8 的 drift 演示会再用到。

注意本教程的 desired state 是一个 Python dict，不是 HCL。这不是偷懒：HCL 只是 Terraform 的配置**语法**，状态管理（desired/state/plan/apply）才是**机制**。课程的数据系统教学约定 约定 Terraform/HCL 到 L2 才触及，L0 先把机制裸出来。

**思考题 3.1**：如果两个人同时改 desired state 再各自 apply，会发生什么？（提示：state 文件没有锁。真实系统用 state locking / 后端远端化解决——这正是 Terraform remote backend 存在的理由之一，L2 展开。）

---

## §4 机制面 [1]：raw zone —— 不可变、只追加、带血缘

接入侧。两个源（crm / web_log）的数据进湖：

```python
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
```

```text
[2] 接入：凭据来自 secrets manager；raw 只追加、带血缘
  b001 source=crm n=5 sha256=ca3a13554610b12f at=1（逻辑时钟）
  denied as expected: ingest denied: source 'crm' 凭据错误 —— 认证是接入边界第一道闸
  [check 03] PASS  错误凭据必须被拒
  [check 04] PASS  raw 共 9 条 / 2 batches（crm 源）
```

**为什么 raw 必须不可变（immutable, append-only）？** 注意 `RawZone` 的接口设计：只有 `ingest`（append），**没有 update/delete 方法**——不可变不是靠纪律，是靠接口形状保证的。理由有三层：

1. **可重建性**：curated 是 raw 的派生物（§5）。只要 raw 在，curated 任何时候都能重算。如果 raw 被就地修改过，「重建」出来的东西和当初消费的东西就不一样了——整个派生链失去意义。
2. **审计与血缘**：每个 batch 带 `source / batch_id / ingested_at / n / bytes / sha256`。出了数据事故（比如模型行为突变），你能回答「这批数据什么时候、从哪个源、以什么内容进来的」——sha256 让「内容有没有被动过」变成可机器验证的问题。
3. **接入与治理解耦**：源系统重试、格式脏乱，都先原样收下（raw 的职责只有一个：忠实），清洗判断留给下一层。这就是业界常说的 bronze/silver/gold medallion 分层（Databricks 推广的术语）里 bronze 层的角色。

**认证是接入边界的第一道闸**：凭据不硬编码在代码里，而是存在 `SecretStore`（secrets manager 的本质——代码里只有凭据**名**，值在集中存储），ingest 时校验。错误凭据直接拒绝（check 03）。真实系统里这对应 IAM 的 source 侧授权 + secrets manager（AWS Secrets Manager / HashiCorp Vault 等，机制同类）；**认证**（你是谁）与 §7 的**授权**（你能读什么）是两道独立的闸，L0 把它们分别放在接入侧和消费侧演示。

**思考题 4.1**：demo 里 `ingested_at` 用逻辑时钟（`Clock.tick()`）而不是 wall-clock，为什么？（提示：为了输出确定、复跑逐字节一致——本教程双跑 stdout md5 均为 `3f6aa19e606647c0f6def83ed5561dc9`。真实系统用 wall-clock，但「时间戳只是元数据，不参与语义」这一点不变。）

---

## §5 机制面 [2a]：质量门 —— 晋升 curated 的硬门槛

```python
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
```

```text
[3] 质量门 + 分层派生：curated v1（坏数据挡在层外）
  gate 拦截: 重复 id: t003
  gate 拦截: 必填字段缺失/为空: t006
  gate 拦截: 必填字段缺失/为空: t008
  v1: raw 9 条 → curated 6 条 (sha256=4599c15439c026c8)
  [check 05] PASS  v1: 9→6，拦截 3 条（重复/空 label/空 text）
  [check 06] PASS  PII 投影: phone 不落 curated
```

**toy 指标（L0 基线）**：漏斗 `9 → 6`，拦截率 33%。三类缺陷各命中一个：源内重试导致的重复（t003）、空 label（t006）、空 text（t008）。

两个设计决策值得盯住：

**（a）gate 失败 = 挡住，不是警告后放行。** `quality_gate` 返回 `(kept, problems)`，被拦的记录**不进入 curated**，只留拦截记录。为什么不做成「警告但放行」？因为 curated 的消费者（训练/检索）默认信任这一层——一旦坏数据混进 curated，每个下游都要重新怀疑数据质量，层的存在意义就崩塌了。「层 = 质量承诺」是分层架构的契约。（思考题 5.1 会讨论这个规则的边界。）

**（b）PII 投影在晋升时做。** `CuratedZone.build` 里 `phone` 字段被投影掉（check 06）：raw 保留全量忠实记录（审计需要），curated 只暴露训练真正需要的字段。**暴露面最小化发生在层边界**，而不是靠每个消费者自觉——这和 §7 的最小权限 ACL 是同一个思想在数据内容维度的应用。

**（c）curated 是派生物，不是独立存储。** `build` 的输入只有 `raw`——curated 从不接受直接写入。这保证了一条铁律：**curated 的任何一行都能追溯到 raw 的某个 batch**（血缘闭环），且 curated 永远可以重建。真实系统里 dbt 的 model 就是这个语义：`dbt run` 从上游声明式派生，没人手改 dbt 产出的表（手改即 drift，见 §8）。

**思考题 5.1**：给一个「警告后放行」反而合理的场景。（参考方向：人工审核流程——被拦数据进 quarantine 区等人审，审过再晋升。关键区别是 quarantine 区仍然在 curated **之外**，「层 = 质量承诺」的契约没破。）

---

## §6 机制面 [2b]：快照版本化 —— 训练可复现的工程答案

第三个源（web_log）增量接入后，重新 build 出 v2：

```text
[4] 增量接入（新源 web_log）→ 快照 v2：训练可复现 = 钉住版本号
  v2: raw 12 条 → curated 8 条（新增拦截: 重复 id: t007）
  [check 07] PASS  v2: 12→8，较 v1 多拦 1 条跨源重复
  trainer 钉住 v1 → 6 条；新任务用 v2 → 8 条（两版本共存，互不污染）
  [check 08] PASS  快照钉住: v1 不受 v2 影响
```

注意 v2 新增拦截的 `重复 id: t007` 是**跨源重复**——同一条工单既在 crm 又在 web_log 出现。去重必须在全局（raw 全量）视角做，这正是「集中入湖再治理」优于「各源头自行清洗」的一个具体理由。

**核心问题：训练任务消费的到底是哪份数据？** 如果 curated 是就地更新（in-place update）的表，那么「上周训练用的数据」就不存在了——表已经被新 build 覆盖。模型复现、A/B 对照、事故归因全部失效。L0 的答案是**版本化快照**：每次 build 追加一个新 version（v1、v2……），旧版本原样保留；消费者显式声明读哪个版本（`consume("trainer", "sft-support", "v1", ...)`）。check 08 验证了关键不变量：**v2 的出现不改变 v1 的任何字节**。

这就是 Apache Iceberg / Delta Lake 的 snapshot 机制在 toy 尺度的投影。Iceberg 的 table spec（页面标题 "Spec - Apache Iceberg™"，https://iceberg.apache.org/spec/ ，2026-08-12 抓取确认）把一张表的状态组织成**快照序列**：每次对表的提交（commit）产生一个新 snapshot，snapshot 指向一组不可变的 manifest（数据文件清单），历史 snapshot 保留下来即支持 time travel。nano 版里 `CuratedZone.versions[dataset]` 这个列表就是 metadata 层的玩具对应物，`sha256` 字段是 manifest 校验和的玩具对应物（行号级源码锚点 L2 再核 `[TODO: verify L2 源码锚点]`）。

**「训练用了哪版数据」必须是一等公民的问题**——在 LLM-PBL 的 RSI 闭环里，01/02 轨每次训练都要能回答这个问题，否则 data-model co-dev 的实验全部不可复现。

**思考题 6.1**：快照无限保留会把存储成本撑爆（§7 的账本会看到 v1+v2 都在计费）。真实系统怎么权衡？（参考方向：retention policy / snapshot expiration——Iceberg 有 `expire_snapshots` 类操作；保留策略本身也是平台治理的一部分，而不是事后补救。）

---

## §7 机制面 [4]：治理 first-class —— 最小权限、secrets、成本账本

本课程的一条硬约束是：**安全与成本不是附录，而是机制的一部分**。L0 用三段演示兑现。

**（a）最小权限消费（default-deny）：**

```python
    # ---- [4b] 最小权限消费：default-deny，训练/检索只暴露 curated 快照 ----
    def consume(self, who, dataset, version, curated):
        if dataset not in self.state["grants"].get(who, {}).get("read", []):
            raise PermissionError(f"ACL deny: '{who}' 无 '{dataset}' 读授权 (default-deny)")
        return curated.read_version(dataset, version)
```

```text
[5] 最小权限消费：default-deny
  denied as expected: ACL deny: 'intern' 无 'sft-support' 读授权 (default-deny)
  [check 09] PASS  未授权消费者必须被拒
  [check 10] PASS  trainer 只有 curated 读权 / ingestor 只有 raw 写权
```

授权表（grants）是声明式配置的一部分（§3），每个消费者**只拿它需要的最小集合**：trainer 只有 curated dataset 的读权（check 10）——它根本看不到 raw，于是 raw 里的 PII（phone）对训练任务从根上不可见，而不是靠训练代码「自觉不读」。ingestor 只有 raw 写权，读不了 curated。未声明者一律拒绝（default-deny，check 09）——授权模型的反面是「默认放行、逐个打补丁」，那种模型的安全态势随系统膨胀单调恶化。真实系统里这对应 IAM least-privilege（AWS IAM / GCP IAM 等，机制同类；云厂商实现作参照不锁定，§九）。

**（b）成本账本（存储/计算成本权衡）：**

```python
# ---- [4c] 成本账本：存储成本 first-class。单价 = toy coin/(B·月)，教学设定非真实云价 ----
PRICE = {"raw": 1.0, "curated": 3.0}  # toy 设定：curated 含质量保障+索引，单价 3x（真实价目见 tutorial §7）
def cost_report(raw, curated):
    rb = sum(b["bytes"] for b in raw.batches)
    cb = sum(len(json.dumps(v["records"], sort_keys=True, ensure_ascii=False).encode())
             for vs in curated.versions.values() for v in vs)
    return rb, cb, rb * PRICE["raw"] + cb * PRICE["curated"]
```

```text
[6] 成本账本（toy coin/(B·月)：raw 1.0 / curated 3.0，教学设定非真实云价）
  raw 1161 B ×1.0 + curated(v1+v2) 1021 B ×3.0 = 4224 toy-coins/月
  [check 11] PASS  curated 单条字节 < raw 单条字节（PII 投影生效）
```

**反幻觉声明**：`PRICE` 是教学设定的相对单价（curated 3× raw），**不是任何云厂商的真实价格**，toy-coins 是虚构计量单位。真实价目因厂商、区域、存储类别、访问模式而异，须查官方价目页（如 AWS S3 Pricing：https://aws.amazon.com/s3/pricing/ ，2026-08-12 抓取确认页面存在 `[TODO: verify 具体价目]`）。本教程只用相对价格讲清**机制**：

1. **raw 便宜、curated 贵**是湖仓的常态结构——raw 躺在对象存储上（低频访问、按字节计费），curated 承担质量保障、索引、服务化，单位成本高。账本里 1161 B 的 raw 只计 1161 coins，而 1021 B 的 curated 计 3063 coins。
2. **但 curated 的字节数更小**（check 11：单条 curated 字节 < 单条 raw 字节——PII 投影 + 质量门在缩层）。分层的成本逻辑不是「多存一份」，而是「用便宜的层存忠实全量，用贵的层存小而精的可消费集」。
3. **每个版本都在计费**（v1+v2 都算钱）——§6 的 time travel 不是免费的，快照保留策略是成本决策。这就是「存储/计算/保留」三角权衡在账本上的直接体现。

**思考题 7.1**：如果老板说「raw 太占钱，删掉吧，反正有 curated」，用 §4 的三条理由反驳他。（答案要点：curated 不可重建 → 质量门规则变更后无法回溯重算 → 审计血缘断裂。正确做法是把冷 raw 迁到更便宜的存储类别——这是存储分层，不是删除。）

---

## §8 drift 与最小 diff：infra-as-code 的下半场

配置变了（新增 analyst 授权），再 apply：

```text
[7] drift 与最小 diff：新增一条授权，plan 恰好 1 个 action（不重建世界）
  plan: set_grant       analyst
  [check 12] PASS  最小 diff = 恰好 1 个 set_grant

platform state digest: 112fb2c779d3f592  (state 文件 372 B @ tempdir)
  [check 13] PASS  state 文件已落盘且非空

self-check: 13/13 PASS
```

plan 恰好给出 1 个 action——不重建 dataset、不重放已有授权。**最小 diff 的前提是 §3 的 state 文件**：plan 拿 state（现实账本）和 desired（期望）做差，差多少动多少。

「drift」指现实偏离了声明的期望——比如有人绕过平台手改了一张 curated 表（对应 §5 说的「dbt 产出不可手改」）。有 state 文件就有检测 drift 的基线：refresh 现实、对比 state、diff 即 drift。Terraform 在每次操作前刷新 state 以对齐现实（同一文档页："Terraform uses state to determine which changes to make to your infrastructure"），本 demo 的 `plan()` 就是这一步的玩具版。

**收尾锚点**：platform state digest `112fb2c779d3f592`（state 文件 372 B）。脚本整体输出确定性：两次独立 CWD、`python3 -B` 双跑，stdout 48 行、md5 `3f6aa19e606647c0f6def83ed5561dc9`，逐字节一致（RUN1==RUN2 BYTE-IDENTICAL）。

---

## §9 它模拟了什么、刻意没模拟什么（L0 边界 → L1/L2）

**模拟了**（本教程的验收内容）：raw 不可变追加 + 血缘；质量门硬晋升 + PII 投影；curated 版本化快照 + 钉住消费；声明式 desired/state/plan/apply + 幂等 + 最小 diff；secrets 接入认证；default-deny 最小权限；相对成本账本。

**刻意没模拟**（每一面都是更高阶梯的课题，不是遗漏）：

| 没模拟 | 为什么 L0 不做 | 哪一级做 |
|--------|----------------|----------|
| 并发写入 / 事务提交 | 需要 commit protocol（Iceberg 的乐观并发控制）| L2 对照 Iceberg 源码 |
| 真实存储引擎与格式（Parquet/ORC）| 列存与谓词下推是独立课题 | L1 起可接 DuckDB/SQLite |
| schema evolution | 需要 schema 版本协商机制 | L2（Iceberg schema evolution）|
| 增量 build（只算新 batch）| L0 全量重算以突出「派生」语义 | L1（watermark/增量物化）|
| HCL / provider / remote state | §七 约定 Terraform 语法到 L2 | L2 |
| 真实 IAM/secrets 后端 | 需要云环境 | 云厂商实现仅作参照（§九），不锁定 |

## §10 费曼自检

**讲给外行听**：把数据平台想象成连锁餐饮的中央厨房。原料到货只能进冷库登记入库（raw zone，只进不出、每批留小票和来源——§4）；质检把坏掉的、来路重复的丢掉，坏的不许进操作间（质量门——§5）；处理好的净菜按批次封装贴版本号（curated snapshot——§6），门店做菜只能从净菜柜取，不许进冷库，更没有生熟混放的权限（最小权限——§7）；总部不是每天打电话指挥每家店「今天进什么货、给谁发钥匙」，而是发一张配置单，门店照着配置单对齐现状，差什么补什么，已经对齐的不动（声明式 plan/apply——§3/§8）；财务每月按冷库和净菜柜分别算账（成本账本——§7）。

**思考题汇总**（正文内另有 3.1 / 4.1 / 5.1 / 6.1 / 7.1）：

1. 一句话说清：「curated 是 raw 的派生物」这句话如果改成「curated 是独立维护的另一份数据」，会坏掉哪些机制？（要点：可重建性、血缘、质量门的一致性——两份数据会各自漂移。）
2. 本实现里哪两个东西分别对应 Iceberg 的「metadata 层」和 Terraform 的「state 文件」？（`CuratedZone.versions` 的快照序列 / `Platform.state` 落盘的 state.json。）
3. 如果把质量门从「挡住」改成「打分排序、全部放行」，这个系统还叫湖仓分层吗？（这是 nano-data-juicer 的算子语义与本模块层语义的边界：算子管「数据怎么处理」，层管「质量承诺在哪兑现」。两者互补，不是替代。）

**反例（一个常见错误直觉）**：「数据平台就是一个更大的数据库，把数据都倒进去，用的时候现查现洗。」——错在三点：其一，现洗意味着每个消费者重复实现清洗逻辑，质量口径必然漂移（§5 的层契约不存在了）；其二，没有快照，训练不可复现（§6）；其三，没有权限与成本的层边界，PII 暴露面和存储账单都失控（§7）。湖仓的本质不是「大库」，而是**分层 + 派生 + 声明式治理**——存储可以很便宜（对象存储），价值在层与层之间的契约。

## §11 溯源

| 声明 | 类型 | 来源 |
|------|------|------|
| Terraform state 两句引文（§3/§8） | 文献已有（逐字引文） | https://developer.hashicorp.com/terraform/language/state ，2026-08-12 抓取 |
| Iceberg table spec 页存在、标题 "Spec - Apache Iceberg™"（§6） | 文献已有 | https://iceberg.apache.org/spec/ ，2026-08-12 抓取；行号级锚点 `[TODO: verify L2 源码锚点]` |
| AWS S3 Pricing 页存在（§7，仅作真实价目指针，未引任何价格数字） | 文献已有 | https://aws.amazon.com/s3/pricing/ ，2026-08-12 抓取 `[TODO: verify 具体价目]` |
| Iceberg/Delta/dbt/Terraform 为权威参照实现 | 纲领已有 | 课程的实现参照与数据系统约定 参照表（apache/iceberg、delta-io/delta、dbt-labs/dbt-core） |
| medallion（bronze/silver/gold）为 Databricks 推广的术语（§4） | 合理推断（术语溯源未给链接） | 概念性提及，不作数字声明 |
| PRICE = {raw:1.0, curated:3.0}、toy-coins、全部漏斗数字（9→6、12→8、1161/1021 B、4224、digest 等） | 本实现实测（toy 设定） | `L0_lakehouse_and_iac_state.py` 本次运行输出，非真实云价、不可外推 |
| 「Airbyte/Fivetran 类连接器解决多源接入」 | 纲领已有（课程的数据系统教学约定 关键词） | 概念性提及，未引数字 |

下一站：**L1**——真实小数据集 + 增量物化（watermark）+ 可持久化的 catalog（DuckDB/SQLite 级），复现同一套漏斗语义；**L2**——对照 Iceberg/Delta/dbt 源码的 snapshot/commit/schema evolution 取舍分析 + Terraform HCL/state 实操（见 README 阶梯表）。
