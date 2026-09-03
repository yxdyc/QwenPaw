# L2：多 worker 的 outbox、fencing、backoff 与 compensation

> **核心问题**：L1 已能在单进程崩溃后恢复；多个 dispatcher 同时工作时，怎样避免丢任务、陈旧 owner、重复副作用和越权补偿？
> **先修**：[L1 SQLite/WAL durable runtime](tutorial_L1.md)，尤其是 payload-bound idempotency 与 provider query。
> **新增约束**：多进程竞争、durable outbox/inbox、lease takeover、单调 fencing epoch、退避、补偿和可信控制面身份。
> **运行**：Python 3.10+，只用标准库；必须从空目录运行，约 7 秒。
> **验收**：11/11 checks；真实子进程被 SIGKILL 后恢复；陈旧 epoch 被 provider 拒绝；重复提交只有一个 provider receipt。
> **边界**：这是单机 SQLite/WAL toy；provider-side fencing 是机制模拟，不是分布式租约服务，也没有密码学 token。

---

## 1. durable 不等于 distributed-safe

L1 解决的是：一个 runtime 在 provider commit 前后崩溃，重启后怎样查询并收敛到明确状态。
L2 再加入多个独立进程后，会出现四类新裂缝：

1. 两个 worker 同时看见同一个 `PENDING` intent；
2. worker 拿到任务后死亡，任务永久停在 `LEASED`；
3. provider 暂时失败，立即重试造成 retry storm；
4. effect 已 commit，业务后来要求“撤销”，但数据库 rollback 已经来不及。

本节的最小控制流是：

```text
model intent + trusted runtime context
             │ authorize + fingerprint
             ▼
 PREPARED event + PENDING outbox  ──同一 SQLite transaction
             │
             ▼
 lease(owner, epoch) → provider(key, epoch, fingerprint, payload)
             │                     ├─ epoch < max → STALE_FENCE
             │                     └─ current epoch → effect / same receipt
             │ crash/retry
             ▼
 COMMITTED + inbox + outbox DONE + lease release ──同一 SQLite transaction
             │
             └─ optional compensation = 新 intent + 新 key + 新授权
```

注意：worker 可以调用 provider 多次；安全目标不是“网络只发送一次”，而是同一个
`(idempotency_key, payload fingerprint)` 最多产生一个外部 effect，并最终得到同一 receipt。

---

## 2. 运行

脚本有意在当前目录创建 `runtime.db`、`provider.db` 和 WAL 文件，所以必须使用空目录：

```bash
tmp_dir="$(mktemp -d)"
cd "$tmp_dir"
python3 -B /path/to/L2_distributed_runtime.py
```

不要在课程源码目录直接跑；“目录非空”本身不构成问题，但若已有同名数据库，脚本会 fail fast，防止旧状态
污染验收。下面是两次 fresh-CWD 验收中逐字节一致的一次输出：

```text
==============================================================================
Agent runtime L2 — lease+fencing / outbox / backoff / compensation / control-plane binding
==============================================================================
toy: stdlib only (sqlite3/subprocess) | provider = separate process
protocol inherits L1: typed intent -> authorize -> PREPARED -> provider -> COMMITTED
new in L2: outbox + lease/fencing + recovery + backoff + compensation + trusted context

[1] Happy path: submit -> dispatcher -> commit exactly once
    balances={'attacker': 0, 'research': 475, 'vendor': 25} outbox={'inv-1': {'attempts': 1, 'lease_owner': None, 'status': 'DONE'}}

[2] Dispatcher dies holding lease; recovery takes over and finishes
    dispatcher killed rc=-9 (SIGKILL=-9)
recovery start: chain_ok=True events=3
lease takeover key=inv-2 reason=dead-owner
recovery complete: reset_leases=1 processed=1

    balances={'attacker': 0, 'research': 450, 'vendor': 50} commits={'method': 'outbox', 'principal': 'demo-agent', 'purpose': 'invoice-payment', 'receipt': 'prov-receipt-2', 'session': 'demo-session'}

[3] Provider transient failures: retry with exponential backoff, then succeed
    after 2 provider kills: attempts=2 status=PENDING
    after successful retry: attempts=3 status=DONE receipt=prov-receipt-3

[4] Concurrent duplicate submission: lease + outbox unique key dedupes
    submit outputs: 'SUBMITTED key=inv-4' / 'SUBMITTED key=inv-4'
    same key / changed payload: rc=6 REJECTED idempotency key reused with different payload
    balances={'attacker': 0, 'research': 400, 'vendor': 100} provider_receipts_for_key=1

[5] Compensation: commit a transfer, then undo it with a separately authorized refund
    before_comp={'attacker': 0, 'research': 375, 'vendor': 125} after_comp={'attacker': 0, 'research': 400, 'vendor': 100} baseline_before_inv5={'attacker': 0, 'research': 400, 'vendor': 100}
    compensations={'comp-inv-5': {'original_key': 'inv-5', 'status': 'DONE'}}

[6] Control-plane binding: model purpose is scoped; identity is runtime-owned
    bad purpose rc=5 REJECTED purpose is not allowed
    bad source rc=5 REJECTED source is outside purpose scope

[7] Max attempts exceeded -> NEEDS_HUMAN (simulated by provider crash every time)
    outbox={'attempts': 5, 'lease_owner': None, 'status': 'NEEDS_HUMAN'}

[8] Expired owner is rejected by a provider-side fencing epoch
    epochs: stale=14 fresh=15 reconciled=16; fresh_rc=0 stale_rc=3 stale_reply=STALE_FENCE receipts=1 stale_key_receipts=0 same_scope_blocked=True
    balances={'attacker': 0, 'research': 399, 'vendor': 101} outbox={'attempts': 1, 'lease_owner': None, 'status': 'DONE'}

[9] Durability evidence + self-check
    journal_mode: runtime=wal provider=wal
    integrity_check: runtime=ok provider=ok
    event hash chain verifies=True (20 events)
    leases={} outbox={'comp-inv-5': {'attempts': 1, 'lease_owner': None, 'status': 'DONE'}, 'inv-1': {'attempts': 1, 'lease_owner': None, 'status': 'DONE'}, 'inv-2': {'attempts': 2, 'lease_owner': None, 'status': 'DONE'}, 'inv-3': {'attempts': 3, 'lease_owner': None, 'status': 'DONE'}, 'inv-4': {'attempts': 1, 'lease_owner': None, 'status': 'DONE'}, 'inv-5': {'attempts': 1, 'lease_owner': None, 'status': 'DONE'}, 'inv-8': {'attempts': 5, 'lease_owner': None, 'status': 'NEEDS_HUMAN'}, 'inv-9': {'attempts': 1, 'lease_owner': None, 'status': 'DONE'}}
    PASS | [1] commit is DONE and bound to trusted principal/session/purpose context
    PASS | [2] dead dispatcher's lease taken over and intent committed exactly once
    PASS | [3] provider failures retried with backoff, then committed
    PASS | [4] duplicate payload deduped; same key with changed payload rejected
    PASS | [5] compensation reversed the committed transfer, balances restored
    PASS | [6] wrong purpose and out-of-scope source rejected without commits
    PASS | [7] max attempts exceeded escalates to NEEDS_HUMAN and never commits
    PASS | [8] provider rejects stale fencing epoch; recovery converges to one effect
    PASS | [9] WAL enabled on both stores
    PASS | [9] integrity_check ok after all kills
    PASS | [9] event hash chain verifies end-to-end

SELF-CHECK: 11/11 PASS
digest(sha256 of metrics) = e8276261bbc1dfd6
takeaway: leases coordinate workers; a monotonic provider-checked epoch fences stale owners; payload-bound idempotency makes retries converge to one effect. The outbox makes work durable, backoff absorbs transient failures, compensation undoes committed effects, and trusted context binds principal/session/purpose.
RESULT_JSON={"checks": {"passed": 11, "total": 11}, "digest": "e8276261bbc1dfd6", "evidence_boundary": "Real local processes and SQLite/WAL; toy single-host provider, wall-clock leases, provider-side fencing simulated in SQLite, and no cryptographic token verification or distributed fencing service.", "metrics": {"s1_committed": true, "s1_context_bound": true, "s1_outbox_done": true, "s2_dispatcher_killed": true, "s2_no_duplicate_receipt": true, "s2_recovered": true, "s3_backoff_attempts": true, "s3_eventually_committed": true, "s4_one_commit_row": true, "s4_one_provider_receipt": true, "s4_payload_mismatch_rejected": true, "s5_balances_restored": true, "s5_compensation_committed": true, "s5_original_committed": true, "s6_bad_purpose_rejected": true, "s6_bad_scope_rejected": true, "s6_no_commits": true, "s7_needs_human": true, "s7_no_commit": true, "s8_epoch_monotonic": true, "s8_one_effect": true, "s8_scope_exclusion": true, "s8_stale_rejected": true}, "module": "nano_agent_runtime_l2", "schema_version": 1}
```

---

## 3. outbox 真正保证的是什么？

`PREPARED` event 和 `PENDING` outbox 必须在同一个本地事务中提交；等价骨架如下：

```python
conn.execute("BEGIN IMMEDIATE")
append_event(conn, "PREPARED", key, event_data, commit=False)
conn.execute("INSERT INTO outbox (...) VALUES (...)", values)
conn.commit()
```

否则存在一个经典 crash window：journal 已写“准备完成”，但 outbox 尚不存在；恢复器既看见意图事实，
又没有可派发工作。反过来，只有 outbox 没有 provenance event，也会让审计链无法解释任务来源。

provider 返回 receipt 后，本地的 `commits`、`inbox`、`outbox=DONE`、event 和 lease release 同样在一个
SQLite transaction 中完成。这个原子性只覆盖 **runtime.db 内部**，不覆盖 provider.db；两个事务域之间
仍靠 provider 的 idempotency 与 query 收敛。

因此可以把保证拆成三层：

| 层 | 保证 | 机制 |
|---|---|---|
| queue admission | 一个 key 只有一个待处理 payload | outbox UNIQUE key + fingerprint mismatch reject |
| external effect | 同 key+fingerprint 只产生一个 effect | provider receipt table 的 payload-bound idempotency |
| local convergence | receipt 最终进入 commit/inbox，outbox 最终 DONE | retry/recovery + 本地原子事务 |

“exactly once”如果不说明是哪一层，通常是不完整的声明。

---

## 4. lease 与 fencing epoch 分别解决什么？

lease 让大多数时候只有一个 worker 处理 key：

```text
PENDING → acquire lease(owner, expires_at) → LEASED → release
```

但设想 worker A 卡顿超过 TTL，worker B 取得新 lease；随后 A 又恢复。仅检查“当前是否有 lease”无法撤回
A 手里已经拿到的请求。本 toy 因而把两个机制明确拆开：

1. runtime 每次成功获取或接管 lease，都把该资源 scope 的持久化 counter 加一，得到 $f_{new}>f_{old}$；
2. provider 在产生 effect 的同一事务里按资源 scope 持久化 `max_epoch`，并先拒绝所有 $f<f_{max}$ 的请求；
3. payload-bound idempotency 再让同一 epoch 或更新 epoch 的重试返回同一个 receipt，而不是重复 effect。

本例把 source account 当作受保护的资源 scope，而不是把 idempotency key 当作 scope；因此换一个请求 key
也不能绕过陈旧 epoch。场景 [8] 显式构造旧 owner 过期、新 owner 先到 provider、旧 owner随后恢复：
下游返回 `STALE_FENCE`；runtime 再以更高 epoch 对账，只保留一个 receipt。这个反例同时说明：

- lease 负责减少并发冲突，epoch 负责让下游识别“你已经过期”；
- fencing 不替代 idempotency——新 owner 可能在收到响应前断线，仍需安全重试；
- idempotency 也不替代 fencing——同一资源的陈旧 owner 可能携带不同合法 key，仍须由资源版本拒绝。

这里的 provider 与 lease store 仍是同一主机上的两个 SQLite 数据库。它验证协议形状和失败方向，不能证明
网络分区下的线性一致性，也不能替代 etcd/Consul/数据库序列或真实下游的 conditional write。

另一个容易忽略的错误是把 `time.monotonic()` 写进数据库。某些运行时的 monotonic 基准是进程局部的，
进程 A 写入的 deadline 不能由进程 B 比较。本节使用跨进程可比较的 wall clock 做单机演示：

```python
def durable_now() -> float:
    return time.time()
```

wall clock 仍会受时钟回拨影响；生产中应优先使用数据库/server time，并把安全交给 fencing epoch，而不是
只相信一个过期时间。

---

## 5. retry/backoff：重试是状态机，不是 while True

第 $a$ 次失败后，本实验设置：

$$
d(a)=\min(2^a, 30\text{s}).
$$

outbox 同时记录 `attempts` 和 `next_attempt_at`。dispatcher 只选择到期的 `PENDING` 行；超过
`MAX_ATTEMPTS=5` 后进入 `NEEDS_HUMAN`，不能伪装成成功，也不能无限消耗 provider。

实验为快速构造 2 次/5 次连续失败，`dispatch-one` 测试入口可以直接触发一次 attempt；真实 dispatcher
仍遵守 `next_attempt_at`。生产系统还应加入 jitter、错误分类、熔断、全局预算和 dead-letter queue。

---

## 6. compensation 不是数据库 rollback

provider effect commit 后，runtime 无法把外部世界“回滚到事务开始前”。补偿是一个新的、可能失败的 effect：

```python
{
    "operation": "transfer",
    "source": original["destination"],
    "destination": original["source"],
    "amount": original["amount"],
    "purpose": "refund",
    "idempotency_key": f"comp-{original['idempotency_key']}",
}
```

它必须有自己的 key、fingerprint、purpose authorization、outbox 和 receipt。本脚本把 compensation plan、
`PREPARED` event、outbox row 与 `COMPENSATION_SCHEDULED` event 放在同一个 SQLite transaction 中；
但如果 refund provider 永久失败，系统仍只能进入人工接管，不能宣称原 effect 从未发生。

---

## 7. 模型 intent 与可信控制面必须分开

模型可以提议：operation、source、destination、amount、purpose；它不能自行声明“我是哪个 principal，
属于哪个 session”。本实验的 `principal/session` 来自 runtime 的可信上下文，写入 outbox、commit 和 event：

```text
model intent purpose=invoice-payment
        +
runtime context principal=demo-agent, session=demo-session
        ↓
purpose scope: invoice-payment 只能从 research 账户付款，amount ≤ 30
```

错误 purpose 和越界 source 都 fail closed。同 key 换 payload 也被拒绝，防止先授权小额请求、再用相同 key
替换成另一笔 effect。

这里没有签名 token、密钥轮换、issuer/audience/expiry 验证。`AUTHORITY` 是教学用可信控制面替身，不能当作
生产 authentication 实现。

---

## 8. 九组场景如何对应真实 crash window

| 场景 | 注入点 | 关键验收 |
|---|---|---|
| happy path | 无 | commit identity 绑定 + outbox DONE |
| dispatcher death | provider 已 commit，本地尚未 commit | dead lease takeover；provider 返回同 receipt |
| transient failure | provider commit 前 SIGKILL | attempts 增长、backoff、最终 commit |
| concurrent duplicate | 两进程同时 submit | 同 payload 合并；换 payload 拒绝；provider receipt 仅 1 行 |
| compensation | original 已 commit | 新授权 refund 完成后余额恢复 |
| scope attack | purpose/source 越权 | 无 commit |
| retry exhaustion | provider 每次 commit 前死亡 | `NEEDS_HUMAN`，无 commit |
| stale owner | epoch 1 过期，epoch 2 先到下游，epoch 1 后到 | provider 返回 `STALE_FENCE`；对账后仅一个 receipt/effect |
| durability | 多次 SIGKILL 与 stale request 之后 | 两库 WAL/integrity ok，event hash chain 完整 |

hash chain 只能暴露已有 event 的篡改/断链；它不是签名、时间戳服务，也不能证明“应该记录但从未记录”的事件。

---

## 9. 证据边界

- SQLite 与 provider 都在同一主机；没有网络分区、leader election 或分布式共识。
- fencing epoch 在 provider SQLite 事务内真实检查，但 lease store/provider 都是单机 toy；没有证明网络分区、
  跨 region 或独立共识系统下的线性一致性。
- provider 天生支持 payload-bound idempotency 和 query；不具备这两个能力的 legacy effect 需要更保守的
  `NEEDS_HUMAN` 策略。
- wall clock 适合本 demo 的跨进程 deadline，不足以承担生产安全。
- compensation 可能失败，也可能无法真正恢复业务语义；“余额相等”只是本构造任务的验收。
- event hash chain 无密码学身份；控制面 token 也没有签名验证。
- stdout 的确定性证明控制路径可重复，不证明高并发吞吐或线性扩展。

下一步 L3 应把 SQLite provider 换成真实 agent/tool runtime 或 HTTP mock，并加入网络分区、响应丢失、
outbox relay crash、permission replay、prompt-injected tool output 和人工接管演练；fencing 要落到下游
conditional write，而不是只存在于 runtime 自己的数据库。

费曼自检：如果 lease 已保证同一时刻只有一个 dispatcher，为什么 provider 仍必须做 idempotency？
若答案没有覆盖 TTL 过期、stale owner、网络响应丢失和重试，就还没有抓住本节的安全边界。
