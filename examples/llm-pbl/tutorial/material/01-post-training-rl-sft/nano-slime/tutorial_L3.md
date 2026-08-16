# nano-slime · L3 教程：data buffer 回收 × delta weight sync——把 L2 的两个抽象对象拆到源码级

> L0 用离散事件模拟建立了解耦的结构（buffer 容量 vs staleness），L1 在真实小模型上
> 测出 G/T/S，L2 把 slime 两条训练主循环建模成双 regime、证明了 interval 是
> staleness 旋钮不是吞吐旋钮。L2 留下两个「只知其名」的对象：**data buffer**
> （L0 的抽象队列，L2 只在 §9 提了一句 ABORTED 回流）与 **delta weight sync**
> （L2 §8 只论证了它为什么存在 = 把 S 做小）。L3 对照 slime 源码
> （HEAD `2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d`，2026-08-16 codeload tarball 抓取）
> 把这两个对象的**实现机制**变成可运行的本质模拟，并回答四个问题：
> partial rollout 回收的到底是什么账？delta 的 wire bytes 由什么决定？
> xor 与 overwrite 两种编码的本质差异是什么？delta 在什么 regime 下恒亏？
> 前置：先跑过 [L0](tutorial_L0.md)、[L1](tutorial_L1.md) 与 [L2](tutorial_L2.md)。
> 对应实现：[L3_buffer_delta_sync.py](L3_buffer_delta_sync.py)。

---

## 1. 运行与输出

**可运行性契约声明**：本节是 **L3 允许的本质模拟**——本机没有
GPU/多机/共享文件系统，跑不了真实 SGLang `/pull_weights`、Megatron gather 与 NCCL。
模拟核心本身可运行、零依赖（纯标准库）、输出完全确定；它建模的每个结构事实都有
slime 源码行号背书（§13 行锚表逐项核验），
压缩用标准库 zlib-1 **显式代替** zstd-1（只取「压缩吃零字节」这一结构性质，
绝对压缩率不可外推，§11）。真机验证标 `[TODO: verify on real system]`，
需在具备相应依赖的 GPU 环境另行执行。

跑法（零依赖，任意 CWD，CPU 瞬时 <0.25s）：

```bash
$ python3 -B L3_buffer_delta_sync.py
```

真实输出（2026-08-16；elapsed 行按既有口径 `sed '/^[[:space:]]*elapsed/d'` 掩码——
raw 59 行 → 掩码 58 行。在两个新建独立 CWD 各运行两遍，均 EXIT=0、stderr 0 B；
四次掩码输出逐字节一致，md5
`1c85efaf9ef1c2807380ac258e87a33e`。脚本自产 digest
`482ddb8ba574d4ddff9177859c26aac1`；elapsed 是机器相关观测量，不纳入确定性比较）：

```text
====================================================================
nano-slime L3 — data buffer 回收 × delta weight sync（源码级机制模拟）
溯源：THUDM/slime @ 2fa9a442（2026-08-16 抓取）；行锚见各段注释
====================================================================
[0] data buffer：buffer-first 采样 × partial rollout 回收
    workload: 24 组 dataset / 每轮取 8+2 组（过采样）/ 连续批（每迭代各组 +1 tok）/ 6 轮；长尾组 total∈{128,224,288,352}
    ✓ 两变体各训 48 组 = 6×8（吞吐结构不变——buffer 买不到吞吐，承 L1/L2）；过采样余量去向 = abort 5 + 超额完成 7（超额完成不回库：sglang_rollout.py:L439-440 NOTE 同款语义）
    ✓ 守恒律：off 引擎 10912 tok = 训练 7712 + abort 丢弃 1536 + 超额 1664；on = 训练 8224 + 超额 1888 + 残留 288
      abort 丢弃的 1536 tok 在 off 里永久作废（dataset 槽位同失）；on 里 abort 进度 0 丢弃——半截工作带戳回库
    ✓ 回收组 4 个，staleness = train_round − start_rollout_id ∈ [1]——版本戳就是 off-policy 度数（mask-offpolicy-in-partial-rollout 是算法侧对策，arguments.py:L467-474）
    ✓ 游标守恒：off dataset 取 60 组（游标 60）；on buffer 命中 4 次顶替 dataset 消费（游标 56）——回收另省 dataset 槽位（prompt 昂贵时 [agent 轨迹] 这是第二笔账）

[1] delta chain：seed → diff → 编码 → 压缩 → 校验 → 滚动基线
    世界：6 tensor / 全量 573,440 B；基线 = seed（首调不发布，L84-88）
    ✓ v1: 变 ['t0', 't2', 't4']（t1/t3/t5 未变 → 零代价跳过），density=0.0125，wire=28,164 B（全量 573,440 B 的 4.91%）
    ✓ engine apply v1：md5 逐字节吻合 trainer（版本链 0→1）
    ✓ v2: 变 ['t0', 't1']，density=0.0068，wire=17,661 B——diff 对 v1 基线（快照滚动 L247），t2/t4 的上轮变更不再上线
    ✓ 版本链拒绝乱序：v2 施加在 v0 基座 → ValueError（version 000002 只允许施加在 base 000001 上（当前 0））
    ✓ engine apply v2（基座 v1）：md5 逐字节吻合——链 0→1→2 闭合
    ✓ checksum fail-loud：base 被篡改 1 字节 → 新态 adler32 不符（宁可报错，不发坏权重）

[2] 编码代数：xor = 对合（须恰好一次） vs overwrite = 幂等（可重复施加）
    N=262,144 B 单 tensor，**均匀随机**变更，密度 d 下 raw / 压缩后 wire（zlib-1）：
      d=0.100%: xor raw= 262,144 zip=  2,495 | overwrite raw=    1,314 zip=  1,092 | xor 更省：False
      d=0.996%: xor raw= 262,144 zip= 11,654 | overwrite raw=   13,059 zip=  8,129 | xor 更省：False
      d=4.978%: xor raw= 262,144 zip= 41,396 | overwrite raw=   65,254 zip= 33,350 | xor 更省：False
      d=19.919%: xor raw= 262,144 zip=109,295 | overwrite raw=  261,084 zip=126,601 | xor 更省：True
    ✓ raw 账：xor ≡ N；overwrite = 4+5c（c=实际变更字节数）——与 overwrite_encode 布局逐字节对账
    ✓ 均匀随机变更下交叉点在 1%–20% 之间——**overwrite 在低密度反而更省**，与 docs:L72『xor smallest wire』相反？看下一探针
    块状探针（d=0.994%，8 个连续块）：xor zip=4,132 vs overwrite zip=6,584 → xor 更省：True
    ✓ 交叉位置取决于**变更分布**：均匀散布利于 overwrite、块状聚集利于 xor——slime 默认 xor（arguments.py:L175）的依据是真实 diff 结构 + zstd-1 实测（docs:L33 profiling 压过 lz4/gzip/snappy/brotli）；nano 的均匀随机 toy 复现机制、不复现 profiling 数字（绝对压缩率不可外推）
    ✓ xor 施加两次 → 还原回 base（involution 机器证明）——『必须对正确 base 恰好一次』（docs:L72-74）
    ✓ overwrite 施加两次 → 仍是新态（idempotent 机器证明）——重试/断点续传安全（docs:L75-78）

[3] regime 整合：S_delta 灌回 max(G,T)+S，delta ⊥ colocate
    承 L2：G=168.96 / T=73.46 / S_full=7.35；BW_NET=78,019 B/toy-t，BW_SCAN=8×BW_NET（toy），C_flush=0.5
    d=0.0125 时：wire=28,164 B → S_delta=1.280 vs S_full=7.35（省 82.6%）；其中扫描底 0.919 是 delta 躲不掉的
    ✓ breakeven（toy 实测标定）：α = wire/density = 2,248,102 B（v1 标定），wire_be = (S_full−扫描底)×BW_NET = 501,760 B → d* = 22.3%——密度高于 d* 时 diff 扫描底 + wire 反超全量直推
      interval=1: 每步 full=176.81 → delta=170.74（省 6.070；staleness 上界 = k，结构性，承 L2）
      interval=2: 每步 full=172.89 → delta=169.85（省 3.035；staleness 上界 = k，结构性，承 L2）
      interval=4: 每步 full=170.92 → delta=169.40（省 1.518；staleness 上界 = k，结构性，承 L2）
    ✓ flush 与 S 同被 interval 摊薄，但都不改 staleness 上界——interval 仍是 staleness 旋钮
    ✓ colocate：S=0.3（handle-only）→ delta 强加扫描底 0.919 → 1.219 > 0.3——『delta bookkeeping 是纯开销』（arguments.py:L2058-2060 原句坐实）
    ✓ 决策规则（机器验证）：delta 仅当 ① 字节真要过网络/盘（nccl/disk disaggregated）且 ② 变化密度 < d* ≈ 22.3%；colocate 永远 full-IPC

    digest(metrics) = 482ddb8ba574d4ddff9177859c26aac1

====================================================================
✅ self-check passed: buffer-first/回收零丢弃/版本戳 staleness≥1 · 未变跳过/滚动基线/版本链/checksum · xor 对合 vs overwrite 幂等 · delta⊥colocate/breakeven 密度
====================================================================

takeaway: data buffer 的价值不在吞吐（L1/L2 已证 buffer 买不到吞吐），而在把
          abort 的半截长尾工作带版本戳回库——staleness≥1 是它的价格，
          mask-offpolicy 是算法侧对策。delta sync 把 S 从『全量过网』降成
          『扫描底 + 密度×全量过网』：本 toy 密度 1.25% 时省 82.6%（breakeven d*≈22.3%，实测标定），但 diff
          躲不掉全量扫描，密度高于 d* 反亏；colocate 走 CUDA IPC
          （handle-only）时 delta 是纯开销——slime 直接 raise 禁止该组合。
          真机验证（SGLang /pull_weights + Megatron gather + 共享盘）[TODO: verify on real system]
```

---

## 2. 代码结构

单文件、零依赖（`hashlib`/`random`/`struct`/`time`/`zlib` 标准库），四块实验 + 汇总：

1. **[0] `DataSourceWithBuffer` + `sim_rounds`**——data_source.py
   `RolloutDataSourceWithBuffer` 的最小忠实版：`get_samples` 先 buffer（pop_first FIFO）
   后 dataset（游标 + epoch 回绕），`add_samples` 回库。`sim_rounds` 是连续批语义的
   离散事件模拟：每轮取 TARGET+OVER 组（过采样），所有在途组每迭代各推进 1 token，
   第 TARGET 个完成即轮末，剩余在途组 abort——partial rollout 开/关两变体对照。
2. **[1] `DeltaChain`**——trainer 侧 `publish`（diff against 快照 → 编码 → zlib-1 →
   新态 checksum → index 元数据 → 快照滚动）+ engine 侧 `apply`（版本链校验 →
   解码施加 → checksum fail-loud）。共享文件系统用内存 dict 模拟。
3. **[2] 编码代数探针**——xor/overwrite 的 encode/apply 各一对，raw 账逐字节对账 +
   均匀随机密度扫描 + 块状变更探针 + 施加两次的代数性质验证。
4. **[3] regime 整合**——把 [1] 产出的 wire bytes 灌回 L2 的 `max(G,T)+S` 闭式，
   标定 breakeven 密度 d*，机器证明 delta ⊥ colocate 决策规则。

与 L2 的衔接有一处显式差异：L2 把 S 当黑盒（`S=7.35` 含一切推送开销），L3 把
flush 拆出来单列（`C_FLUSH=0.5`，对应 pause + flush_cache + reload 的固定代价）——
所以 [3] 里 interval=1 的 full 每步是 176.81 = L2 的 176.31 + 0.5，两处口径相差
恰一个 C_FLUSH，不是数字漂移。

---

## 3. [0] data buffer：三笔账——吞吐账、资产账、槽位账

slime 的 data buffer 在源码里朴素得惊人：`self.buffer = []`
（data_source.py:L171），采样**先 buffer 后 dataset**（L182-188），默认 filter 是
pop_first FIFO（L225-229）。真正的机制不在 buffer 本身，而在**谁往 buffer 里写**：
partial rollout（`--partial-rollout`，arguments.py:L456-465，help 原文
"If set, the unfinished samples during dynamic sampling will be recycled back to
data buffer. This is useful for long responses."）。一轮 rollout 凑满
`rollout_batch_size` 后（sglang_rollout.py:L407 `while len(data) < target_data_size`
退出），abort 掉所有在途请求（L451 `aborted_samples = await abort(args, rollout_id)`）；
未完成的组带着 `start_rollout_id` 版本戳（L363-364：
`sample.metadata["start_rollout_id"] = rollout_id`）回收进 buffer（L648
`data_source.add_samples(aborted_samples)`），下一轮 buffer-first 优先消费。

[0] 输出把这笔账拆成三层，每层都有断言守着：

**吞吐账：buffer 买不到吞吐。** 两变体各训 48 组 = 6×8，逐位相等——回收不改变
「每轮训 target 组」的吞吐结构。这是 L1/L2 结论的原样迁移：稳态吞吐被
max(G,T) 钉死，buffer 只改变样本**从哪来**，不改变引擎**跑多快**。

**资产账：abort 的半截工作从「丢弃」变「库存」。** off 变体里 abort 丢弃的
1536 tok 永久作废（引擎算力白付、dataset 槽位白占）；on 变体里 abort 进度
0 丢弃——4 个长尾组带戳回库、下轮优先消费并训完，期末 buffer 还残留 288 tok
在制品。守恒律是严格的机器证明（不是近似）：off 引擎 10912 = 训练 7712 +
丢弃 1536 + 超额 1664；on = 训练 8224 + 超额 1888 + 残留 288——每个 token
都有去向。注意「超额完成不回库」这条语义：同一迭代同时完成的组若超过 TARGET，
超出的直接丢弃、不进 buffer——这是 sglang_rollout.py:L439-440 NOTE 原文
"# NOTE: here we have not stored all the unused samples back to the data buffer."
的忠实实现，toy 的 `extra_waste` 就是它。

**槽位账：回收另省 dataset 消费。** on 变体 buffer 命中 4 次顶替 dataset 取组
（游标 56 vs off 的 60）。prompt 廉价时这笔账无所谓；prompt 昂贵时
（agentic RL 的多轮工具调用轨迹、环境交互上下文），「不用重新取 prompt」
是回收的第二笔收入——这也是为什么 partial rollout 的 help 特意写
"long responses"：response 越长，abort 时沉没的 prompt 成本越高。

**价格：staleness ≥ 1。** 回收组的 staleness = train_round − start_rollout_id，
实测 ∈ [1]——生成在旧版权重、训练在新版权重，版本戳就是 off-policy 度数。
算法侧的对策是 `--mask-offpolicy-in-partial-rollout`（arguments.py:L467-474，
help 原文 "If set, only on-policy generated tokens will be used in training"）：
把上一轮生成的前缀 token 从 loss 里 mask 掉，只训本轮权重下生成的部分。
系统侧回收、算法侧买单——和 L2 的 interval 一样，slime 对 staleness 的态度
始终是「显式定价、显式对冲」，不是假装它不存在。

---

## 4. [1] delta chain：seed → diff → 编码 → 压缩 → 校验 → 滚动基线

delta weight sync 的定义（docs/en/advanced/delta-weight-sync.md:L3-6 逐字）：
"keeps non-colocated rollout engines up to date by shipping only the bytes that
changed between two syncs, instead of a full checkpoint each time. It targets
large-model training/inference disaggregation across clusters or datacenters,
where writing the whole actor every sync is the dominant cost." —— 它只属于
**训推分离**（disaggregated）regime，这个限定词是后面 [3] 决策规则的伏笔。

slime 的实现链路（update_weight_from_disk_delta.py）与 toy `DeltaChain` 的对应：

**seed：基线从 hf_checkpoint 抓，不从 GPU 权重抓。** 首次 `update_weights`
只抓基线、不发布（L84-88：`if not self._baseline_captured: ... return`）。
`_capture_baseline` 的 docstring（L95-101）说明了种子选择的讲究：快照 seeded
from `--hf-checkpoint`——那正是每个 rollout host materialize 本地 base 的来源，
于是不变量 `snapshot == engine base` 从第 0 版就成立；即使 Megatron→HF 的
round-trip 不是字节精确的（embed/lm_head 的 vocab-padding 行会被 trim），
diff 也不会产生「假变更」。docs:L63-65 同款表述。toy 的世界里 trainer/engine
字节天然同一，没有 round-trip 精度问题，所以构造时直接取当前世界为基线——
这是 toy 可以简化的地方，但简化的理由本身要讲清（真实系统里 seed 来源是
正确性问题，不是性能问题）。

**publish：未变 tensor 整块跳过 + 快照滚动。** 每次 sync 对每个 tensor
diff against 上一版快照；`if not changed: return name, new, None, None, 0`
（L240-241）——未变 tensor 零代价（连压缩都不进）。变了的走编码 →
zstd-1 压缩（L242）→ 新态 checksum（L243）→ 写成 canonical HF 目录
（`weight_v{N:06d}/`，index.json 带 version/base_version/delta_encoding/
compression_format/checksum_format 元数据，L157-167）→ 快照滚动推进到新值
（L247：`snapshot[name] = new  # becomes the next sync's base`）。
[1] 输出逐条验证：v1 只含 t0/t2/t4（t1/t3/t5 未变跳过），v2 只含 t0/t1——
t2/t4 的 v1 变更**不再上线**，因为 diff 的基线已经滚动到 v1。滚动基线是
delta 链的灵魂：不滚动，每轮都会把历史变更重复付费。

**apply：版本链 + checksum 双闸。** docs:L82-86 逐字："The trainer stores a
per-tensor checksum of each tensor's new state in the version. After applying,
every host recomputes the checksum and **raises on any mismatch** — the failure
propagates through the `/pull_weights` response, so a corrupt delta or a wrong
base fails loud instead of serving bad weights. The apply also refuses to run
out of order: a version only applies on top of its declared base version."
toy 把两种失败模式各演了一遍：v2 施加在 v0 基座 → 版本链 ValueError 拒绝；
base 被篡改 1 字节 → 新态 adler32 不符 fail-loud。宁可训练中断，不让坏权重
流入引擎——权重分发系统的正确性底线是「错得响亮」，不是「尽量兼容」。

**reload：pull → pause → flush_cache → reload → continue。** 引擎侧时序
（L176-189）：`pull_weights`（每个 engine 把 delta apply 到所有 host 的本地
checkpoint）→ `pause_generation` → `flush_cache` → `update_weights_from_disk`
（走**普通**的 disk 加载路径，weight loader 永远看不见 delta 格式，
docs:L60-61）→ `continue_generation`。其中 `flush_cache` 是 prefix cache
失效代价（nano-vllm-sglang L2/L3 的主题：radix cache 在权重更换后全部作废），
toy 把它与 pause/reload 一起建模为 `C_FLUSH=0.5` 的固定代价。

---

## 5. [2] 编码代数：xor 是对合，overwrite 是幂等——交叉点取决于变更分布

两种编码都是字节级、dtype-blind（docs:L69-70），raw 账可以逐字节对账：

- **xor**：payload = `new ^ old`，raw ≡ N（不管变了多少字节，payload 恒等于
  tensor 全长）。docs:L72-74 逐字："writes `new ^ old`. Smallest wire and
  fastest to apply (sequential, cache-friendly; the unchanged bytes are zeros
  the compressor crushes). It is an involution, so it must be applied
  **exactly once** against the correct base — applying it twice reverts."
- **overwrite**：payload = u4 变更数 + u4 位置序列 + 新值字节
  （disk_delta.py:L21-25 同款布局与 docstring："The 'overwrite' delta:
  changed-position count (u4), positions (u4 each), then new values.
  Idempotent to apply, unlike xor (an involution)"），raw = 4+5c（c = 实际
  变更字节数）。docs:L75-78 逐字："writes the changed positions and their new
  absolute values. Larger on the wire and a less cache-friendly scattered
  apply, but **idempotent**: re-applying it (or finishing a partially-applied
  delta) converges to the same state regardless of how many times it runs."

代数性质的机器证明（[2] 输出末两行）：xor 施加两次还原回 base（对合，
involution）——所以版本链必须保证「对正确 base 恰好一次」，重试对 xor 是
**危险**操作；overwrite 施加两次仍是新态（幂等，idempotent）——断点续传、
重复投递都安全，重试是**免费**操作。这一条代数差异决定了两种编码的运维语义，
比 wire size 的差异更本质。

wire size 的故事有一个反直觉的转折。[2] 的均匀随机扫描显示：d≈1% 时
overwrite 压缩后 8,129 B < xor 11,654 B，d≈4.98% 时仍反超不了
（33,350 < 41,396），直到 d≈19.9% xor 才赢（109,295 < 126,601）——
**低密度下 overwrite 反而更省**，与 docs:L72 "Smallest wire" 的说法相反？
块状探针给出答案：同密度 ~1%，变更聚集成 8 个连续块（真实权重 diff 的形态：
整行/整块参数一起动）时，xor zip=4,132 vs overwrite zip=6,584，xor 反超。
机制：块状变更下 xor 的 payload 是「长零游程 + 密集变更块」，两段都极好压；
overwrite 的位置列表（4B/位置）在块状下毫无压缩收益，仍要照付。

**结论：wire 的交叉位置取决于变更分布，不只是密度。** slime 默认 xor
（arguments.py:L175 `default="xor"`）的真实依据不是「xor 无条件最小」，
而是真实权重 diff 的结构 + zstd-1 的实测 profiling（docs:L33 逐字：
"Deltas are always zstd-compressed (level 1); profiling showed it dominates
lz4 / gzip / snappy / brotli on both wire size and decompress speed for this
data, so it is not a knob."）。nano 的均匀随机 toy 复现的是**机制**
（分布依赖性、对合 vs 幂等），不复现 profiling 数字——zlib-1 显式代替 zstd-1，
绝对压缩率不可外推（§11）。这也是读文档的正确姿势：docs 的 "smallest wire"
带着「真实 diff 结构」的隐含前提，把文档结论当无条件定理用，就会在自己的
场景里选错编码。

---

## 6. [3] regime 整合：S_delta 灌回 max(G,T)+S，delta ⊥ colocate

L2 §8 留了半个问题：delta sync 为什么存在？[3] 给出定量答案。delta 一次
sync 的 toy 时间 = 扫描底（diff 要扫全量，memory-bandwidth-bound，
disk_delta.py:L11-13 原文 "The delta phases (diff, zstd, checksum) are
memory-bandwidth bound"）+ wire 过网络：

- d=1.25%（v1 实测密度）：S_delta = 1.280 vs S_full = 7.35，**省 82.6%**；
  其中扫描底 0.919 是 delta 躲不掉的——即使 wire = 0，diff 也要把全量读一遍。
- **breakeven 密度 d\* ≈ 22.3%**（toy 实测标定：α = wire/density = 2,248,102 B，
  wire_be = (S_full − 扫描底)×BW_NET = 501,760 B）。密度高于 d\* 时，
  扫描底 + wire 反超全量直推——delta 不是永远赢，它有明确的适用域。
  注意 α 是 v1 单点标定的 toy 常数（假设 wire 与密度线性），真实压缩率随密度
  非线性变化，d\* 是定性决策边界，不是精确阈值。
- **interval 摊薄**：k=1/2/4 时 full 每步 176.81/172.89/170.92 → delta
  170.74/169.85/169.40。(S + C_flush) 一起被摊薄，但 staleness 上界 = k 的
  结构性事实分毫未动（承 L2 §6：interval 是 staleness 旋钮）。delta 与
  interval 是同一个问题（S 太大）的两条正交解法：delta 把 S 本身做小、
  不牺牲 on-policy 度；interval 把 S 摊薄、代价是陈旧度上界抬升。
- **delta ⊥ colocate：机器证明的决策规则。** colocate 时权重走 CUDA IPC，
  只有 handle 跨进程（S_COLOC=0.3），delta 的 snapshot+diff+encode 变成
  白付的扫描底：0.3 + 0.919 = 1.219 > 0.3，**恒亏**。slime 源码对此的态度
  是直接禁止（arguments.py:L2057-2061，raise 原句逐字）：
  "--update-weight-mode=delta is not supported with --colocate. Colocate
  transfers weights via CUDA IPC (only a handle crosses processes), so the
  delta bookkeeping (snapshot + diff + encode) is pure overhead."
  toy 的断言与源码的 raise 表达同一条决策规则：**delta 仅当 ① 字节真要过
  网络/盘（nccl/disk disaggregated）且 ② 变化密度 < d\*；colocate 永远
  full-IPC。** 一个工程特性该不该有开关，看它在哪些 regime 有正收益——
  slime 连开关都不给，因为 colocate+delta 在任何参数下都是负收益。

---

## 7. 权威实现取舍表：nano 版没做什么

| 维度 | nano-slime L3（本文件） | slime 真实实现 | 差异原因 |
|------|------------------------|----------------|----------|
| buffer | list + pop_first FIFO + group 为 dict | `RolloutDataSourceWithBuffer`（data_source.py:L171/L182-188/L225-229），group = n_samples_per_prompt 条样本，filter 可插拔（`--buffer-filter-path`，默认 pop_first） | nano 取默认 filter 的共相；优先级/staleness-aware filter 留作思考题 1 |
| partial rollout | 离散事件：轮内 target 达成后在途组带进度回库，round 号作版本戳 | `abort()` 异步中止在途请求（sglang_rollout.py:L451）+ `metadata["start_rollout_id"]`（L363-364）+ `add_samples` 回库（L648）；依赖 SGLang patch 的 `/abort_request` endpoint（lmsys blog 逐字："Reclaiming partially generated content, which enables partial rollouts"） | 真实 abort 是请求级异步操作；toy 的「每迭代各组 +1 tok」是 continuous batching 的本质抽象 |
| delta publish | 内存 dict + zlib-1 + adler32 | canonical HF 目录（safetensors + index.json，L157-167）+ zstd-1（L242）+ xxh3-128 默认 checksum（arguments.py:L186-198）+ 原子写 + ThreadPool 并行（disk_delta.py:L11-13：diff/zstd/checksum 释放 GIL，线程池回收带宽） | zlib-1 显式代替 zstd-1（只取结构性质）；adler32 是 slime 的可选算法之一（interop 场景）；真实 publish 作用于 Megatron TP/EP gather 后的 HF 张量（L199-273），toy 无并行布局 |
| seed | 构造时取当前世界 | hf_checkpoint 种子 + `pull_weights(0)` 与快照捕获重叠（L95-101） | toy 世界字节自洽、无 Megatron→HF round-trip 精度问题；真实 seed 设计为「snapshot == engine base」不变量服务（vocab-padding trim） |
| apply | 版本链校验 + checksum fail-loud（忠实复现） | `/pull_weights` fan-out 到引擎所有 host、per-host 文件锁折叠同机 rank、per-tensor 并行 apply + 逐 tensor 验证，全部 host 验证通过才报成功（docs:L49-53） | 共享文件系统用内存 dict 模拟，多 host 一致性未建模；「reload 走普通 disk 路径、loader 不见 delta 格式」的解耦在 toy 的 publish/apply 分离中保留 |
| 引擎窗口 | C_FLUSH 常数 | pause_generation → flush_cache → update_weights_from_disk → continue_generation（L176-189） | prefix cache 失效代价依赖 workload（nano-vllm-sglang L2/L3 主题），toy 取常数 |
| colocate 守卫 | [3] 断言机器证明 | arguments.py 直接 raise（L2057-2061） | 语义同一：delta ⊥ colocate |
| 时长/分布 | 确定性常数（长尾组 total ∈ {128,224,288,352}） | 真实 response 长尾、工具/环境交互 | 确定性换断言精确（守恒律逐字节对账），承 L2 §13 同款口径 |

---

## 8. slime 当今定位（证据时效性声明）

slime 属 **B 层前沿主流**（对齐日期 2026-08-16，本次源码快照日期）：
LMSYS Org 官方博客发布（2025-07-09，"slime: An SGLang-Native Post-Training
Framework for RL Scaling"，抓取件 45,411 B）；THUDM/slime main 持续活跃
（该快照顶 commit `2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d`，
slime_commits.atom 首条，与代码头部声明逐位吻合）。delta weight sync
是 main 分支较新加入的特性（docs + 实现均在 2fa9a442 快照在盘；具体合入
时点未考古 `[TODO: verify]`）。

与经典锚点的关系：slime 的机制地基仍是 RLVR 族 RL 算法（GRPO 族，A/B 层
锚点见 nano-verl tutorial_L3 §八）+ 训推分离 rollout 架构；delta sync 与
partial rollout 是**工程传输层**的优化，与算法层正交——算法换了（如从 GRPO
换 DAPO），这套 buffer/delta 机制原样成立。这正是本模块只取数据通路骨架做
阶梯的原因：传输层机制的半衰期比算法名长。

---

## 9. 费曼自检

**类比：快递站的两本账。**

partial rollout 是「在途件回仓」：快递员没能在截单前送完（本轮 target 凑满，
abort 在途请求），站点不把半送的包裹扔掉，而是带回仓库、贴上「上一班次」的
标签（start_rollout_id 版本戳），下一班次**优先派送**（buffer-first）。
包裹不浪费了，但它已经旧了一个班次——标签就是陈旧度。算法侧还有个补丁：
真按旧班次路线生成的那几单，结算时不算提成（mask-offpolicy）。

delta sync 是「红线修订合同」：两家公司签长期合同（trainer 与 rollout
engine），每次改条款不再邮寄整本 500 页合同，只寄红线 diff 页
（xor/overwrite + 压缩）。但每次仍要**翻一遍整本合同**才能找出改了哪几页
（扫描底，memory-bandwidth-bound，躲不掉）；改得太多（密度 > d\*）时，
寄 diff 反而比寄新全本贵。diff 页必须带版本号（版本链）——「第 2 次修订
只能贴在第 1 版合同上」，贴错了立刻报错（checksum fail-loud），绝不把坏
合同投入使用。而如果两家公司**共用一个文件柜**（colocate），直接递柜子
钥匙就行（CUDA IPC handle）——这时维护红线账本纯属浪费，所以 slime 源码
直接禁止这种组合。

自检三问（讲不出来就回 §3/§5/§6 重读）：

1. 为什么 partial rollout 买不到吞吐、却仍然值得开？（提示：三笔账分开算。）
2. xor 为什么必须「对正确的 base 恰好一次」？施加两次会发生什么？
   overwrite 为什么不怕重复施加？哪种编码的重试是免费的？
3. delta 在 wire = 0 的理想情况下为什么仍然有成本？这个成本在 colocate
   时意味着什么？

---

## 10. 思考题

1. **buffer filter 设计**：slime 的默认 filter 是 pop_first FIFO，但
   `--buffer-filter-path` 可插拔。设计一个按「剩余工作量 + staleness」
   排序的优先 filter（修改 `DataSourceWithBuffer.get_samples`），用模拟器
   论证：它相对 FIFO 能改善什么指标？会引入什么风险？（提示：长尾组永远
   排后面会发生什么？staleness 分布怎么变？）
2. **breakeven 的解析 vs 实测**：把 [1] 的 mutate 密度改成
   {0.05, 0.1, 0.3, 0.5} 重测，观察 wire/density 的 α 是否恒定；
   用 toy 闭式 d\* = (S_full − 扫描底)×BW_NET/α 预测 breakeven，
   与逐密度实测对比——解析值为什么会偏？（提示：压缩率随密度的非线性，
   以及 xor raw ≡ N 与 overwrite raw = 4+5c 的交叉。）
3. **checksum 选型**：slime 默认 xxh3-128，可选 blake3 / adler32
   （arguments.py:L186-198 help 原文 "this is a digest-property choice,
   not a speed one"）。为什么校验和不是速度选型？什么场景需要 cryptographic
   强度？（提示：共享盘是否可信——docs:L90-91 "blake3 is cryptographic,
   for untrusted storage"。）
4. **版本链的重置**：docs:L55-59 说一个「普通全量 checkpoint」版本会
   reset 整条链——迟加入的新 host 从最新全量版 seed，而不是回放所有 delta。
   为什么必须有这个设计？delta 链无界增长会付出什么代价？（类比
   `git clone --depth=1` 与全量历史回放。）
5. **架构选型尺子**：结合 L2 §4（异步要求非 colocate）与 [3] 的决策规则，
   列一张「colocate + full-IPC vs disaggregated + delta」的选型表：
   硬件成本、S 的量级、staleness 容忍度、prefix cache 行为各占什么权重？
   G/T 在什么区间、集群规模在什么量级时各占优？（提示：delta 的收益
   ∝ 全量推送的字节账，集群越大、模型越大，disaggregated 越划算。）

---

## 11. 反例与边界

1. **toy 尺度诚实声明**：zlib-1 显式代替 zstd-1（slime 实测 zstd-1 在
   wire size 与解压速度上压过 lz4/gzip/snappy/brotli 且不可调，docs:L33；
   nano 只取「压缩吃零字节」这一结构性质，**绝对压缩率不可外推**）。
   BW_SCAN=8×BW_NET、C_FLUSH=0.5、S_COLOC=0.3、OVER=2 均为 toy 常数
   （代码注释逐处声明），结构结论可外推、绝对值不可。breakeven d\*≈22.3%
   只在 toy 标定内成立。
2. **均匀随机变更 ≠ 真实权重 diff**：[2] 的块状探针证明交叉位置取决于
   变更分布。nano 的均匀随机假设在低密度下得出「overwrite 更省」，与 docs
   的 "xor smallest wire" 表面相反——两者都对，前提不同（§5）。slime 默认
   xor 的依据是真实 diff 结构 + zstd-1 profiling，toy 复现机制、不复现
   profiling 数字。
3. **单 host 模拟**：真实 `/pull_weights` fan-out 到引擎所有 host
   （per-host 文件锁、per-tensor 并行验证、全部 host 通过才成功，
   docs:L49-53），toy 用内存 dict 模拟共享文件系统、不建模多 host 一致性。
4. **未建模 filter/reward 路径**：dynamic filter / filter_hub（奖励过滤、
   过采样补样）不在本节主题内；[0] 的过采样余量只区分 abort 与超额完成。
5. **partial rollout 的真实触发条件是长 response**（arguments.py help
   "This is useful for long responses"）；toy 的长尾组是确定性常数，
   真实长尾来自任务结构（工具调用、多轮交互）。

---

## 12. 阶梯预告

nano-slime 至此 L0–L3 完整：L0 结构解耦（buffer 容量 vs staleness 上界）→
L1 实测 G/T/S（真实小模型）→ L2 双 regime + interval 旋钮（控制流逐行对照）→
L3 源码级 buffer 回收 + delta sync（本节）。继续深挖的三个方向：
fully async 的背压设计（L2 §9 已引 fully_async_rollout.py:L85-89 逐字注释）
与引擎内部机制 → 轨道 03 [nano-vllm-sglang](../../03-data-distributed-rsi/nano-vllm-sglang/)；
off-policy 的算法侧修正（IS ratio / mask-offpolicy 的损失账）→
[nano-verl](../nano-verl/) L1/L3；真机 delta sync（SGLang `/pull_weights` +
Megatron gather + 共享盘）`[TODO: verify on real system]`，走 GPU 通道攒批验证。

---

## 13. 溯源与口径声明

**slime 源码快照**：codeload main 分支 tarball，2026-08-16 10:06 抓取，
6,010,379 B。公开教程不分发第三方源码归档，可按顶 commit
`2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d` 重建；同日抓取的 commits feed
首条 `<id>` 与该 commit 逐位吻合。**本文全部行号以此快照为准**，下表锚点已逐项核验：

| 锚点 | 内容 |
|------|------|
| data_source.py:L171 / L182-188 / L225-229 | `buffer=[]` / get_samples 先 buffer 后 dataset / pop_first FIFO |
| sglang_rollout.py:L407 / L439-440 / L451 | `while len(data) < target_data_size` / 超额完成不回库 NOTE（§3 逐字引）/ abort |
| sglang_rollout.py:L356-357 / L363-364 / L648 | 非 partial rollout 直接 continue / start_rollout_id 戳 / add_samples 回库 |
| arguments.py:L175 / L186-198 | `--update-weight-delta-encoding` default="xor" / checksum 三选一 help（§10 逐字引） |
| arguments.py:L456-465 / L467-474 | `--partial-rollout` help（§3 逐字引）/ `--mask-offpolicy-in-partial-rollout` help（§3 逐字引） |
| arguments.py:L2057-2061 | colocate+delta raise（§6 逐字引，原句 L2059-2061） |
| update_weight_from_disk_delta.py:L84-88 / L95-101 | 首调只抓基线 / hf_checkpoint 种子 + pull_weights(0) 重叠 |
| update_weight_from_disk_delta.py:L157-167 / L176-189 | index 元数据 / pull→pause→flush_cache→reload→continue |
| update_weight_from_disk_delta.py:L231-239 / L240-241 / L242 / L243 / L247 | xor/overwrite 分支 / 未变跳过 / zstd-1 / 新态 checksum / 快照滚动 |
| utils/disk_delta.py:L11-13 / L21-25 | memory-bandwidth-bound + GIL / overwrite 布局 + 幂等 docstring（§5 逐字引） |
| docs/en/advanced/delta-weight-sync.md:L3-6 / L33 / L37-41 / L55-59 / L63-65 / L67-78 / L82-86 | 定义 / zstd-1 profiling / seed / 全量版重置链 / hf-checkpoint 正确性 / 编码代数 / integrity（§4/§5 逐字引） |
| README.md:L45 | Delta Weight Sync 特性条目（承 L2 §10 锚） |

**lmsys blog**：2026-08-16 12:51 抓取，45,411 B（"slime: An SGLang-Native
Post-Training Framework for RL Scaling"，LMSYS Org，2025-07-09）；§7 引文
"Reclaiming partially generated content, which enables partial rollouts"
逐字（源页标点空格归一）。

**arXiv 探索声明**：探索件 `[2505.16312]`（EquivPruner，agent 搜索剪枝）/
`[2603.24477]`（Composer 2 Technical Report）与 title:"slime" 检索命中
（多为黏菌文献）均与本模块机制无直接相关，未引用；delta sync 的一手来源
就是 slime repo 源码与 docs，无需论文支撑。

**口径**：本节所有数字分三类——(a) 模拟器产出（paste 块内全部数字，可复现：
掩码锚 `1c85efaf…`/58 行、digest `482ddb8b…`）；(b) 源码/docs/blog 引文与
行号（上表，快照内逐字）；(c) toy 常数（BW_SCAN 倍率、C_FLUSH、S_COLOC、
OVER，代码注释逐处声明）。**没有任何真实系统的 benchmark 数字**——真机验证
`[TODO: verify on real system]`（SGLang `/pull_weights` + Megatron gather +
共享盘，需在可用 GPU/多机环境另行验证）。
