# Evaluation Gate L2 companion — 聚类口径一换，结论为什么会翻转

> **核心问题**：同一份 paired rows、同一个平均增益，按 task、source、user 或 domain 作为独立单位时，
> p-value 为什么会完全不同？遇到这种分歧，系统应该选最显著的口径，还是先冻结？
>
> **先修**：[EpisodeRecord L0](../cross-track-episode-record/tutorial_L0.md) 的 provenance/版本合同；
> [Evaluation Gate L2 主课](tutorial_L2.md) 的 cluster-level sign flip 与 alpha-spending。
>
> **运行**：`python3 L2_cluster_sensitivity.py`；纯标准库 + 同目录 L2 模块，CPU，通常一秒内。
>
> **验收**：26/26 self-check；同一批 12 行只有逐 task 口径通过固定 0.05，真实 lineage 口径均拒绝，
> policy disagreement 输出 `FREEZE`；manifest→policy→evidence→audit 必须按序持久化，重启可重放，且 evidence
> 出现后不能新增 policy 或原地改写记录。

---

## 1. 这不是再发明一条 gate，而是审计 gate 的问题定义

[L2 主课](tutorial_L2.md#4-p-value-从哪里来独立单位是-cluster不是-row) 已把正式检验从逐 task 改为
cluster-level exact sign flip。但它故意留下一个更基础的问题：**cluster 到底按什么定义？**

一条 task 可能同时属于：

- 原始文档 `source_id`；
- 真实用户 `user_id`；
- 产品域 `domain_id`；
- 当前 task 本身 `task_id`。

这四个字段都能把 rows 分组，却不回答同一个问题。若一个 source 派生十道题，逐 task 口径估计的是“随机抽一题”
的表现；按 source 等权更接近“随机抽一个来源”的表现。先看结果再选口径，就是 researcher degrees of freedom：
失败时合并、成功时拆分，最终 p-value 不再拥有原先的错误率含义。

companion lab 因而不替正式 gate 作 promotion。它只做三件事：

1. 从 outcome 之外的 lineage manifest 派生 cluster，禁止调用方自由命名；
2. 并列计算多个**预登记** policy 的效应量、有效 cluster 数与 p-value；
3. 只要 pass/reject 结论不一致，就输出 `FREEZE`，要求先明确 estimand 或补独立证据。

---

## 2. 两张表必须分开：lineage manifest 与 paired evidence

代码把输入拆成：

```text
LineageRow(task_id, source_id, user_id, domain_id, traffic_weight)
PairedDelta(task_id, delta)
```

第一张表描述样本从哪里来、产品流量怎样加权；第二张表才包含 candidate-parent 的结果差 $d_i$。拆开是为了让
cluster policy 能在看见 $d_i$ 之前登记。`manifest_id` 对排序后的 lineage rows 做内容寻址；`evidence_id`
单独绑定 paired outcomes。

`ClusterPolicy` 又固定：

- 选择哪个 lineage axis；
- 固定 $\alpha$；
- 簇内如何聚合：本 lab 是 equal-row mean；
- 簇间如何聚合：本 lab 是 equal-cluster mean；
- 使用哪种检验：one-sided exact sign flip。

这些字段连同 `manifest_id` 进入 `policy_id`。改 axis、改权重、改阈值或改算法，都是新 policy，不允许把旧 id
贴到新口径上。

内容 hash 只能证明“这份字节是什么”，不能证明“何时登记”。所以本 lab 不停在 hash，而是把对象按下一节的
顺序写入一个真实 SQLite registry；但生产系统仍需独立时间戳与权限域，事后自建数据库不叫预注册。

---

## 3. 用持久化顺序把“预登记”变成可验收合同

`PreregistrationLedger` 用一条 append-only event log 和四张对象表保存：

```text
REGISTER_MANIFEST -> REGISTER_POLICY x 4 -> RECORD_EVIDENCE -> AUDIT
```

每个对象表行都引用创建它的 `event_seq`；事件又把前一事件 hash 纳入下一事件 hash。manifest、policy、evidence、
report 与 event 表都装有拒绝 `UPDATE`/`DELETE` 的 trigger，并在同一 SQLite transaction 里同时追加 event 与对象，
避免“事件记了但对象没写”或相反的半提交。

关键权限边界在 `record_evidence`：

- manifest 必须已经登记；
- 必须已有至少两个不同 lineage axis 的 policy；
- event 把当时**全部** policy id 固定下来；
- 从 evidence 成功落账起，只允许 payload-identical 的旧 policy 重试，新增 policy 永久拒绝；
- `run_registered_audit(evidence_id)` 只从 registry 加载 policy，不接受调用者临时传一套口径。

这使三个常见失败可测试：没有 policy 就塞 evidence 会拒绝；看完 evidence 再登记更有利口径会拒绝；超时重试同一
policy/evidence/audit 不会追加重复事件。`verify_registration_order()` 从 event 头开始重放并比对对象表投影，关闭再
打开数据库后仍须得到同一 `FREEZE` report。

它比“README 里说我们预登记了”强，但不是可信第三方证明。掌握数据库文件和程序的人仍可离线重建一条全新的自洽
hash chain，SQLite 顺序也不知道调用者是否早已在别处偷看 outcome。生产上应把 registry 放入独立服务，由认证身份、
不可回拨时钟和访问控制签发 receipt；跨服务发布还需 [L3a outbox/receipt](tutorial_L3.md)，而不是让本地数据库冒充
共识或不可抵赖日志。[L1 durable promotion](tutorial_L1.md) 展示了相邻的事务/恢复原则，但它的决策账本也不能替代
独立预注册 authority。

---

## 4. 两个效应量回答不同问题

对 traffic weight $w_i$（$\sum_i w_i=1$），row-weighted effect 是：

$$
\widehat\Delta_{\text{row}}=\sum_i w_i d_i.
$$

对 policy 派生出的 $G$ 个 cluster，先求每簇简单均值：

$$
\bar d_g=\frac{1}{|I_g|}\sum_{i\in I_g}d_i,
$$

再让每个 cluster 等权：

$$
\widehat\Delta_{\text{cluster}}=\frac{1}{G}\sum_{g=1}^{G}\bar d_g.
$$

二者可能不同。例如 source A 有 10 道题且全为 `+0.01`，source B 有 2 道题且全为 `-0.01`：

$$
\widehat\Delta_{\text{row}}=\frac{10(0.01)+2(-0.01)}{12}=0.006667,
$$

而 source 等权是：

$$
\widehat\Delta_{\text{source}}=\frac{0.01+(-0.01)}{2}=0.
$$

这里没有哪个数字“算错”。前者说随机一题平均变好；后者说随机一个 source 没有平均增益。真正的错误是没有先
声明目标，却在 outcome 出来后把两个 estimand 混用。

---

## 5. 同一 row effect，为什么 p-value 仍会变

lab 的 12 个 delta 固定为 10 个 `+0.01`、2 个 `-0.01`，traffic weight 均为 $1/12$。所以无论按哪个 axis
查看，`row_effect` 都是 `+0.006667`。变化的是**可独立翻转的证据单位**：

| axis | cluster 结构 | equal-cluster effect | exact p | fixed 0.05 |
|---|---|---:|---:|---|
| task | 12 个 task | +0.006667 | 0.019287 | PASS |
| source | 10 正题同源 + 2 负题同源，共 2 source | 0 | 0.750000 | REJECT |
| user | 3 个全正 user + 1 个全负 user，共 4 user | +0.005000 | 0.312500 | REJECT |
| domain | 1 个全正 domain + 1 个 4 正 2 负 domain | +0.006667 | 0.250000 | REJECT |

逐 task 时仍有主课推导的：

$$
p_{\text{task}}=\frac{\binom{12}{10}+\binom{12}{11}+\binom{12}{12}}{2^{12}}
=\frac{79}{4096}\approx0.019287.
$$

按 source 时只有 $(+0.01,-0.01)$ 两个 cluster mean，观察统计量为 0。四种符号组合中有三种统计量不小于 0，
故 $p=3/4=0.75$。按 domain 时两个 cluster mean 都为正，只有全正组合达到观察值，故 $p=1/4$；这个值仍
不可能低于 0.05。**重复 rows 可以让 task 数变大，却没有制造新的 source/domain。**

少 cluster 带来的离散、低功效不是检验缺陷，而是数据对该问题确实提供不了更细的证据。把 2 个 source 拆成
12 个字符串，只是在账面上制造样本量。

---

## 6. 为什么分歧必须 `FREEZE`

若 audit 返回：

```text
task_id   -> PASS
source_id -> REJECT
user_id   -> REJECT
domain_id -> REJECT
```

系统不能说“至少一种方法显著，所以晋升”。这等于在四次尝试中挑一个成功结果，而且选中的恰是最容易伪重复的
task axis。`FREEZE` 的语义是：

- 当前 promotion 依据对问题定义敏感；
- 在 policy owner 明确目标 population 之前，没有自动晋升权限；
- 可以补新的独立 source/user/domain，或用事先约定的主 estimand 重做 fresh evaluation；
- 不能删除不利 policy，也不能把其权重改到结论一致。

`FREEZE` 不等于 candidate 必然更差。它是权限判断：现有证据不足以支持一个对口径稳健的自动动作。

---

## 7. provenance contract 能防什么

代码只接受 manifest 允许的四个 axis，并从 `LineageRow` 自己派生 `{task_id: cluster_id}`。如果上游还提交一份
cluster assignment，它必须与派生结果逐项一致。以下情况在统计计算前 fail closed：

- task id 重复；
- source/user/domain 任一字段为空；
- paired evidence 覆盖与 frozen manifest 不完全相同；
- traffic weight 非正、非有限或不归一；
- 调用方把 `task-00` 从 `source-a` 重标成 `source-b`；
- 使用 manifest 合同之外的 axis。

这与 [EpisodeRecord](../cross-track-episode-record/) 的连接是：EpisodeRecord 应在 rollout/采集时记录 task 和
provenance identity；本 audit 消费那个 frozen view，而不是让 evaluator 根据分数反推 cluster。

但 provenance 字段相同不等于 causal independence 已被证明。两个不同 `source_id` 仍可能是同一模板改写，多个
user 也可能共用组织、设备或事件冲击。生产审计还需检查采样设计、生成谱系、时间批次与污染关系。

---

## 8. fixed 0.05 为什么只用于这个反例

本 companion 故意让四种 policy 都在同一个 fixed 0.05 下比较，以隔离“cluster definition 导致的翻转”。它没有
领取或修改 [L2 主 gate](tutorial_L2.md#3-为什么连续试-candidate-不能每次都用-005) 的全局 trial counter。

真实流程应是：

```text
EpisodeRecord freezes provenance
        -> sensitivity audit validates the preregistered unit/estimand
        -> one primary cluster policy enters L2 global alpha-spending
        -> durable PROMOTE/REJECT continues to L1/L3a publication
```

不能把本 audit 的四个 p-value 当四次正式 promotion trial，也不能在它们中做最低 p-value selection。若组织确实
要同时保护多种 population，应预先设计多重终点控制、层级 gate 或 intersection rule。

---

## 9. 参考运行输出

```text
[1] provenance and policies are content-addressed before outcomes
    manifest=manifest-5d506f745d4b policies=4 evidence=evidence-7c741dec794c report=sensitivity-e4f964619fe2
[2] the same paired rows answer different questions
    axis=task_id   tasks=12 clusters=12 row_effect=+0.006667 cluster_effect=+0.006667 p=0.019287 pass=True
    axis=source_id tasks=12 clusters= 2 row_effect=+0.006667 cluster_effect=+0.000000 p=0.750000 pass=False
    axis=user_id   tasks=12 clusters= 4 row_effect=+0.006667 cluster_effect=+0.005000 p=0.312500 pass=False
    axis=domain_id tasks=12 clusters= 2 row_effect=+0.006667 cluster_effect=+0.006667 p=0.250000 pass=False
[3] disagreement is a stop signal, not a policy-selection menu
    status=FREEZE passing_axes=task_id
[4] provenance contract rejects post-hoc relabeling
    wrong_assignment=True duplicate_task=True missing_lineage=True invalid_weights=True incomplete_evidence=True
[5] weights are part of policy identity
    original=policy-85a3177bafaa reweighted=policy-e1001411a2bd changed=True
[6] durable preregistration closes before evidence
    events=7 policies=4 no_policy_evidence=True late_policy=True policy_retry=True immutable=True
    chain=True replay=True idempotent=True reopened=True
[7] structural self-check
    PASS | manifest identity ignores row ordering
    PASS | task-level exact p-value is 79/4096
    PASS | naive task policy passes fixed 0.05
    PASS | source lineage leaves only two clusters
    PASS | equal-source estimand cancels to zero
    PASS | source-level exact p-value is 0.75
    PASS | user-level evidence does not pass
    PASS | domain-level evidence does not pass
    PASS | row-weighted effect is stable across policies
    PASS | policy disagreement freezes selection
    PASS | lineage-derived ids reject caller relabeling
    PASS | duplicate task ids fail closed
    PASS | missing lineage fails closed
    PASS | invalid traffic weights fail closed
    PASS | incomplete evidence coverage fails closed
    PASS | changing frozen weights changes policy identity
    PASS | registry stores one manifest, four policies, evidence, report, and events
    PASS | evidence requires preregistered sensitivity axes
    PASS | new policy is rejected after evidence
    PASS | payload-identical policy retry remains idempotent after closure
    PASS | retries do not append duplicate events
    PASS | stored rows reject in-place UPDATE and DELETE
    PASS | event hash chain verifies
    PASS | event replay verifies registration order
    PASS | stored report round-trips
    PASS | close and reopen preserves counts, report, chain, and replay
SELF-CHECK: 26/26 PASS
takeaway: preregistered provenance derives clusters; disagreement freezes the gate; durable order is auditable after restart.
```

输出中的短 hash 是教学用 content address，不是签名、权限证明或可信时间戳。关键验收不是记住 hash，而是解释：
为什么 row effect 不变、source effect 却归零；为什么只有 task policy 通过时不能自动选择 task policy。

---

## 10. 费曼类比：十二张小票来自几桌客人

餐厅收到 12 张满意度小票：10 张好评来自同一桌团建客人，2 张差评来自另一桌。按“小票”平均，满意度很高；
按“每桌一票”，一桌好、一桌差，证据打平。

若经理看到结果后才决定“今天按小票算，明天按桌算”，统计规则就变成了业绩美化工具。正确做法是事先说明：
要优化的是每位用餐者、每桌订单，还是每个企业客户；身份关系在结账时记录，不能在评分出来后改桌号。

自检：

1. 为什么 10 张同桌好评不能当 10 桌独立好评？
2. row-weighted effect 与 equal-cluster effect 哪个更接近你的产品目标？依据是什么？
3. 两种合理 policy 结论相反时，`FREEZE` 比选择较小 p-value 多保护了什么？

---

## 11. 动手改造与反例

1. 把 source B 的两个负 delta 改成正值，观察四种 policy 是否转为一致；说明 `CONSISTENT` 仍不等于因果正确。
2. 新增 20 个都来自 source A 的正 task，比较 task p 与 source p；画出“行数增长但有效 source 数不变”。
3. 把 `traffic_weight` 改成 source A 合计 0.5、source B 合计 0.5，解释 row estimand 怎样变化、policy id 为什么必须变。
4. 给 manifest 增加 `template_family_id`，构造 source 不同但 template 相同的反例；讨论更上游的相关性单位。
5. 设计 primary axis + 两个 safety axis：primary 显著但 safety axis 方向为负时应怎样 gate，而不是怎样挑 p-value。
6. 把本地 policy registry 换成独立服务，让它签发绑定 principal、manifest/policy ids 与服务端时钟的 receipt；
   构造调用者先看 outcome、再重建一条自洽本地 hash chain 的攻击，解释为什么单机日志无法识别它。

---

## 12. 本 companion 证明了什么，仍缺什么

它通过可运行反例证明：

- 同一 paired evidence 的显著性会随独立抽样单位改变；
- cluster assignment 可以由 frozen provenance 派生并校验，而不是由调用方任意提交；
- lineage axis、traffic weights、检验与阈值必须进入 policy identity；
- 多个预登记 policy 结论翻转时，可以 fail closed，而不是 post-hoc selection。
- 单机 registry 可以持久化并重放 manifest→policy→evidence→audit 顺序，拒绝 late policy 与原地改写。

它没有证明：

- source/user/domain 字段真的对应因果独立单元；
- equal-cluster 或 traffic-weighted estimand 哪个符合具体业务价值；
- 四个 synthetic policy 构成完整的敏感性空间；
- fixed 0.05 控制持续 candidate 搜索误报；正式控制仍在 L2 主课；
- 单机 SQLite/hash chain 提供可信时间、身份认证或不可抵赖；
- 2–4 个 cluster 具有足够统计功效；
- candidate 在真实 hidden blind set、真实 evaluator 或线上 serving 中可靠。

一句话验收：**先用 provenance 决定哪些行能算独立证据，再用预登记 estimand 计算；若合理口径让结论翻转，
冻结并补证据，而不是挑最显著的那一种。**
