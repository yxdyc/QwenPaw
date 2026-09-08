# L0：给模型发布稿配一张物流单

> **核心问题**：看到“新架构、新 base、新后训练、新 API、榜单提升”时，怎样判断究竟证明了什么？
> **先修**：知道 pretraining、SFT、RL、distillation、MoE 的基本含义。
> **运行**：纯标准库、CPU、无网络与模型下载；Python 3.9+。
> **验收**：12/12 checks，两个空 CWD 运行逐字节一致，最后一行是稳定 `RESULT_JSON=`。
> **边界**：这是 dated evidence/claim contract，不是模型训练、推理或 benchmark。

## 0. 先写下预测

运行前判断下面六句话会被接受还是拒绝：

1. “Qwen3.8 的 51B n-gram capacity 应加到 6B active parameters 里算每 token FLOPs。”
2. “GLM-5.3 与 GLM-5.2 同底座，因此适合做后训练 paired attribution。”
3. “GLM-5.3-Flash 也叫 5.3，所以与 GLM-5.3 的差异主要来自后训练。”
4. “一个候选 pretrain loss 更低，因此可直接晋升到生产架构。”
5. “GPT-6 Astra 有 1.05M context，因此可以反推出它的参数规模。”
6. “请求 Claude Fable 5.1，评测记录只写这个请求名就够了。”

判断依次是：拒绝、进入 paired replay、拒绝、拒绝、拒绝、拒绝。后面用一张“物流单”解释这些判断：模型是货物，
base 和 stage 是中转站，harness 与 fallback 是运输条件。少一段记录，终点分数就很难归因。

## 1. 第一性原理：要估计的是 stage effect

模型终态能力可粗写成：

$$
Y = f(B, D, A, P, R, Q, S, H),
$$

其中 $B$ 是 base lineage，$D$ 是数据，$A$ 是架构，$P$ 是 pre/midtraining，$R$ 是 SFT/RL/OPD，
$Q$ 是量化与部署，$S$ 是采样预算，$H$ 是 harness。想估计后训练增益，目标其实是：

$$
\Delta_R = Y(B,D,A,P,R_1,Q,S,H)-Y(B,D,A,P,R_0,Q,S,H).
$$

若 base、架构或 harness 跟着 $R$ 一起变化，观测差异就无法识别为 $\Delta_R$。本 L0 先检查
**parent、stage 和评测合同能否对齐**，分数比较排在这一步之后。

## 2. 三本账不能混

`ModelCard` 分开记录：

- `total_b`：权重总容量；
- `active_b`：一次 token 前向激活的 backbone 参数口径；
- `external_capacity_b`：外置/查表容量，不自动等于激活计算。

Qwen3.8-Flash-Next 因而写成 `125B total / 6B active / 51B external n-gram`，而不是“176B 模型”或
“57B active”。同理，训练态 optimizer/gradient、KV/state 与 checkpoint 存储还应另开账。

## 3. 声明门：unknown 必须保持 unknown

核心函数宁可失败，也不从同家族旧版本补造当前 recipe：

```python
def require_posttraining_recipe(card: ModelCard) -> str:
    if not card.posttraining_recipe_disclosed:
        raise ValueError(f"post-training recipe is undisclosed: {card.name}")
    return card.posttraining
```

这条规则把 “weights are available” 与 “full recipe is open” 分开。开放权重允许我们研究 checkpoint；训练数据、
teacher、reward、stage checkpoint、分布式训练代码和厂商 harness 仍各自需要证据。

## 4. 归因门：名字相近不等于同底座

toy classifier 只有在同 `base_lineage` 且官方明确将候选描述为 post-training gain 时，才返回
`posttraining-focused`；其他情况返回 `confounded`。这个标签只决定“这对比较是否值得进入更昂贵的 paired replay”，尚未构成因果证明。

- GLM-5.3 ← GLM-5.2：同 base，进入下一层 paired attribution。
- GLM-5.3-Flash ↔ GLM-5.3：新 base、架构/规模/多模态预训练均变化，在 L0 就拒绝当作后训练消融。

## 5. 闭源 API：记录仪比结构猜谜更有用

GPT-6 Astra 和 Claude Fable 5.1 的参数量、内部架构与训练配方没有出现在官方公开材料里。脚本给这些字段保留 `None`，
并检查 API access 没有被写成 open weights。上下文长度、价格和 benchmark 都属于服务侧观测，无法填补结构字段。

闭源模型仍然能教很深的系统问题。GPT-6 Astra 的异步工具调用和 mid-turn steering 要求 runtime 保存 pending call 与新指令的
先后关系；Claude Fable 5.1 的 safeguard 会触发模型 fallback，评测记录因此需要 `requested_model`、实际执行模型与 routing reason。
这像封着引擎盖测试汽车：看不到活塞，但刹车距离、仪表读数、驾驶条件和维修记录都可以严格测量。

## 6. 架构门：至少跨三个轴

Qwen3.8 报告给出一个很有迁移价值的研究纪律：候选不能只看 loss，还要同时过质量、效率、稳定性。L0 将其压缩为：

```python
def approve_architecture_trial(signals: dict[str, bool]) -> bool:
    required = {"quality", "efficiency", "stability"}
    return required.issubset(signals) and all(signals[key] for key in required)
```

真实 L2 应把每个布尔量展开：quality 至少含 downstream/post-training 后效果；efficiency 分 train/prefill/decode 与内存；
stability 包含最优超参区间、loss spike、恢复和跨 seed 方差。

## 7. 运行与真实输出

从仓库根目录运行：

```bash
python3 -B tutorial/material/cross-track-frontier-model-lifecycle/L0_stage_claim_contract.py
```

也可以从任意空 CWD 用脚本绝对路径运行。固定输出如下：

```text
frontier lifecycle L0 — architecture → pretrain → post-train → serve → evaluate
snapshot=2026-09-08 | standard-library teaching contract | no model execution
CARD DeepSeek-V4-Pro: active/total=3.062%; posttrain_recipe_disclosed=True; full_recipe_open=False
CARD Qwen3.8-Flash-Next: active/total=4.800%; posttrain_recipe_disclosed=False; full_recipe_open=False
CARD Kimi K3: active/total=3.714%; posttrain_recipe_disclosed=True; full_recipe_open=False
CARD GLM-5.3: active/total=undisclosed; posttrain_recipe_disclosed=False; full_recipe_open=False
CARD GLM-5.3-Flash: active/total=5.625%; posttrain_recipe_disclosed=False; full_recipe_open=False
CARD GPT-6 Astra: active/total=undisclosed; posttrain_recipe_disclosed=False; full_recipe_open=False
CARD Claude Fable 5.1: active/total=undisclosed; posttrain_recipe_disclosed=False; full_recipe_open=False
PAIR GLM-5.3 <- GLM-5.2: posttraining-focused
PAIR GLM-5.3-Flash <> GLM-5.3: confounded
FAILURES: unknown alias, undisclosed recipe, loss-only gate, and confounded attribution rejected
CHECKS 12/12
RESULT_JSON={"checks":{"active_is_not_total":true,"aliases_normalized":true,"api_access_is_not_open_weights":true,"closed_parameters_remain_unknown":true,"loss_only_architecture_rejected":true,"new_base_comparison_is_confounded":true,"open_weights_not_full_recipe":true,"qwen_external_capacity_not_active":true,"same_base_delta_enters_replay":true,"three_axis_architecture_accepted":true,"undisclosed_recipe_rejected":true,"unknown_identity_rejected":true},"digest":"7025f9f4ad825311","evidence_boundary":"Dated metadata and claim-typing simulation; no weights, training, benchmark, throughput, or production capability were tested.","metrics":{"confounded_pairs_detected":1,"known_active_ratios":4,"model_cards":7,"posttraining_focused_pairs":1,"posttraining_recipes_disclosed":2},"module":"frontier_model_lifecycle_l0","schema_version":"1.0"}
```

## 8. 十二个 checks 分别防什么

| 检查 | 防止的错误 |
|---|---|
| aliases normalized / unknown rejected | 用模糊昵称拼接了错误配置或 license |
| active is not total | 以总参数直接推断计算量 |
| Qwen external capacity not active | 把 host lookup 容量当 backbone FLOPs |
| undisclosed recipe rejected | 从旧代或博客补造当前后训练配方 |
| open weights not full recipe | 把开放权重写成全栈开源 |
| closed parameters remain unknown | 从 context、价格或分数倒推闭源架构 |
| API access is not open weights | 把可调用服务写成可检查 checkpoint |
| same-base delta enters replay | 连 parent 都未固定就声称 stage gain；通过本检查也只表示值得进入 paired replay |
| new-base comparison confounded | 品牌/版本号相近造成伪消融 |
| loss-only rejected / three-axis accepted | 用局部 proxy 提前晋升架构 |

## 9. 反例与下一阶实验

L0 故意注入四类失败，但只证明规则能发现它们。L1 应固定官方 metadata revision、license 和 config SHA；L2 才做
dense↔sparse、AdamW↔Muon、sequential RL↔OPD、sync↔async 的小模型 factorial；L3 才碰真实 frontier 权重与固定 harness。

优先做 GLM-5.3/5.2 same-base replay 和 Qwen 三轴 architecture gate，因为它们最直接改善“提升来自哪里”的判断；
下载多个数百 B/数 T 权重却没有 counterfactual，不会提供同等 ROI。

## 10. 费曼自检

1. 为什么 `total parameters / active parameters` 仍不足以预测真实延迟？列出至少四个遗漏变量。
2. 若 5.3 与 5.2 同 base，但 5.3 benchmark 使用更多 tool fallback，后训练效应还可识别吗？
3. 为什么一个 sparse-attention 方案的 pretrain loss 持平，仍可能在 post-training 后失败？
4. 给 DeepSeek-V4 multi-teacher OPD 设计一个最坏领域回归门：parent、样本单位、指标、停止条件分别是什么？
5. Claude safeguard 触发 fallback 时，怎样定义一次可复算的 candidate-parent pair？

<details>
<summary>参考答案</summary>

1. 至少还缺序列长度与 KV/attention 形态、batch/并发、dtype/量化、kernel 与硬件拓扑、内存带宽、通信、tool/IO 等待和调度开销。active parameters 只近似一部分算力，不是 wall-clock 模型。
2. 不能直接识别。更多 tool fallback 同时改变了执行策略与可用外部能力，观察到的差值是“后训练 + harness/routing”组合效应。要么固定 tool policy 重测，要么把目标 estimand 明确写成整个产品系统，而不归因给后训练。
3. pretrain loss 只测训练分布上的 next-token 拟合。稀疏模式可能丢失 post-training 所需的长程 credit、检索、工具轨迹或视觉对齐信息，也可能在新长度/新 mask 下出现优化与 kernel 不稳定；需要 downstream factorial 和失败面。
4. 固定同一 parent/student、教师集合与 router revision；以任务或 episode cluster 为配对单位，分别报告每个领域 success/completion、最坏领域 delta 与成本。若任一关键域的置信下界低于预登记 non-inferiority margin，或出现安全回归，立即停止并拒绝 candidate，不让平均提升覆盖它。
5. 每个 pair 记录相同输入/seed、requested model、实际 executed model、fallback reason、tool policy、token/cost receipt 和 completion。若 parent/candidate 触发不同 fallback，必须按路由层分层报告；要测模型本体就排除或固定 fallback，要测产品系统则把 fallback 纳入 treatment 定义。

</details>

答案的共同核心是：**先锁定 estimand 和 lineage，再用最便宜的反事实逐层买证据。**
