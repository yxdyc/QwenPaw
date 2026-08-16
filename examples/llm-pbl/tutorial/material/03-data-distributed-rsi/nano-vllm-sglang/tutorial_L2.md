# nano-vllm-sglang L2 — 分页内存管理：block table、refcount 与抢占

> L0 用算术说明分页省显存，L1 用真实模型测出 KV cache / batching / 调度的代价。
> 但 L1 的 KV 池仍是「每序列一整条连续预留」——真实引擎不这么做。
> L2 把内存管理本身换成真的：**物理块池 + block table + refcount + 内容哈希前缀缓存 + CoW + 抢占/重算**，
> 并用一条机器断言钉住语义：**分页前后，生成逐 token 一致**。

---

## 1. 先跑为敬

文件：`L2_real_paged_memory.py`，依赖仅 `torch` + 同目录 L1 模块（import 时设
`sys.dont_write_bytecode`，不落 `__pycache__`）。CPU 即跑，约 1 分钟。

```bash
$ python3 L2_real_paged_memory.py
```

真实输出（2026-08-07 run1，逐字节粘贴）：

```text
====================================================================
nano-vllm-sglang L2 — 真实分页内存管理（block table + 物理块池）
====================================================================
torch 2.13.0 | CPU fp32 | seed=42 | BLOCK_SIZE=4（vLLM 默认 16：config/cache.py:L47）
声明: 权重 = L1 随机初始化 GQA GPT（state_dict 共享，逐参数断言相等）；
      分页管理为最小同构真实现（非 API 包装）；真实引擎 (vLLM/SGLang
      on GPU) 见 [TODO: verify on real system]；源码对照 tutorial_L2 §7

[0] 物理块池: 64 块 × 4 token × 2048 B/token = 0.5 MiB（启动即划走，对齐真实引擎）
    每块 = 8,192 B；池字节 == 公式: True ✅
    对照 L1 连续池: 16 槽 × 512 token 整条预留（每序列预留 512 token，不管实际多长）

[1] 跨级别契约（prompt=8 token，生成 48，greedy）
    paged decode == L1 连续池 decode（逐 token）: True ✅
    按需分配轨迹: prompt 8 tok → 2 块；此后每写满 4 token 加 1 块
    加块时刻 (KV token 数 → 块数): [(9, 3), (13, 4), (17, 5), (21, 6), (25, 7), (29, 8), (33, 9), (37, 10), (41, 11), (45, 12), (49, 13), (53, 14)]
    块数 == ceil(tokens/4) 全程成立: True ✅

[2] 碎片与准入（8 个真实请求，tokens=[14, 10, 17, 11, 22, 13, 9, 16]，Σ=112）
    分页  : 峰值 32 块 = 128 槽位（内部碎片 16，每请求 ≤ 3）→ 预算 32 块准入 8/8
    连续  : 每请求预留 22 tok = 6 块 → 192 槽位（浪费 80，占 42%）→ 同预算准入 5/8
    → 省下的块直接变成并发度（8 vs 5）；L1 [2] 已实测吞吐随 B 单调上升——显存省下的部分就是吞吐

[3] 前缀共享 + copy-on-write（BLOCK_SIZE=4，只缓存满块）
    A: prompt 132 tok（共享前缀 128 + 私有 4）→ 33 块全部写满、登记缓存（free 64→31）
    B: prompt 134 tok，前 128 tok 链式哈希命中 → touch 共享块（ref_cnt=[2, 2]），只新分配 2 块、只计算后缀 6 tok
    prefill 墙钟: 全算 5.92 ms / 命中后 2.31 ms（2.6×，命中段计算整个跳过）
    共享版 B 生成 == 无共享参考（逐 token）: True ✅（因果性使前缀 KV 与后续上下文无关 → 共享无损）
    CoW: P(10 tok = 2 满块 + 1 半块，已生成 1 tok) fork 两子（fork 后各块 ref_cnt=[3, 3, 3]，半块同被共享）
      child1 写入共享半块（写前 ref_cnt=3 > 1）→ 复制块 2 → 3，原块降为 2
      child2 写入共享半块（写前 ref_cnt=2 > 1）→ 复制块 2 → 4，原块降为 1
    父 P 的半块内容在两次 CoW 后逐字节不变: True ✅（共享块只读，写时分裂）
    child1/child2 全轨迹 == solo 参考（14 tok 逐个）: True ✅

[4] 释放复用 + 抢占/重算
    [4a] r0: 15 tok 用块 [0, 1, 2, 3] → 完成归还（free 0→4）
         同前缀 r1 进场: 3 满块全部命中已释放块 [0, 1, 2]（ref_cnt 0→1，移出 free queue，free 4→1），只重算 1 个 token（全命中至少算一格）
    [4b] 预算 12 块（8 请求峰值需 32 块）: 完成迭代 17 | 抢占 8 次（FCFS: 抢占最晚运行者）| 重算恢复 3 次（重算 prefill 共 29 tok）
      iter  2: 请求 7 被抢占（已有 KV 0 tok，释放 0 块，num_computed 归 0）
      iter  2: 请求 6 被抢占（已有 KV 0 tok，释放 0 块，num_computed 归 0）
      iter  3: 请求 5 被抢占（已有 KV 8 tok，释放 2 块，num_computed 归 0）
      iter  3: 请求 4 被抢占（已有 KV 8 tok，释放 2 块，num_computed 归 0）
      请求 4 重算恢复: 对 prompt+已生成共 9 tok 重跑 prefill
      请求 5 重算恢复: 对 prompt+已生成共 9 tok 重跑 prefill
      iter  5: 请求 7 被抢占（已有 KV 0 tok，释放 0 块，num_computed 归 0）
      iter  5: 请求 6 被抢占（已有 KV 0 tok，释放 0 块，num_computed 归 0）
      iter  7: 请求 5 被抢占（已有 KV 10 tok，释放 3 块，num_computed 归 0）
      请求 5 重算恢复: 对 prompt+已生成共 11 tok 重跑 prefill
      iter  8: 请求 7 被抢占（已有 KV 0 tok，释放 0 块，num_computed 归 0）
    抢占/重算后 8 个请求最终 token == 无压力参考（逐个）: True ✅（抢占改变「何时算」，不改变「算什么」）

====================================================================
✅ self-check passed:
   paged == L1 连续池逐 token 一致 / 块数 == ceil(tok/BS) 全程 /
   分页准入 8/8 vs 连续 5/8（同预算）/ 前缀命中 ref_cnt=2 且生成无损 /
   CoW 恰 2 次且父块字节不变 / 释放块被复用且仍是缓存 /
   抢占重算后全部 token 与无压力参考一致
====================================================================

takeaway: 分页把「显存管理」从每序列连续预留变成块池 + block table：
          按需供给消内部碎片，refcount + 内容哈希让前缀可共享可复用，
          CoW 保住共享语义，准入/抢占把过载变成可恢复的代价而非崩溃。
          语义（生成什么）一分不差，代价（块数/重算）变成实数。
          L3 对照 vLLM block manager / SGLang RadixAttention 源码。
```

**声明（课程可运行性契约）**——输出开头也打印了同样的声明：

- **权重是 L1 的随机初始化 GQA GPT**（3,148,032 参数，`state_dict` 共享，逐参数断言相等）。
  真实权重 + 真实引擎（vLLM/SGLang on GPU）仍待真实 GPU/多机环境验证 `[TODO: verify on real system]`。
- **分页管理不是模拟、不是 API 包装**：block table / refcount / 块级 gather-scatter /
  内容哈希前缀缓存 / CoW / 抢占重算都是真实现——只是规模小、跑在 CPU。
  与 vLLM V1 源码的逐条对照见 §8。
- 计时行仅 [3] 的 prefill 墙钟及其派生比值，随机器负载浮动（共独立运行 4 遍，
  比值区间 2.2×–2.6×）；其余全部为计数类输出，跨运行逐字节一致
  （§14 确定性记录）。

---

## 2. K+1：L1 留下什么，L2 接住什么

L1 的 KV 池是**每序列一条连续预留**：`pool[slot]` 形状 `[KV_HEADS, MAX_T=512, HEAD_DIM]`——
一条序列进场就占走一个 512-token 的槽，不管它实际生成多长。L1 靠「池子够大」（16 槽 × 512）
绕开了这个问题，但 L0 [4] 的算术已经算过账：按最长预留会把显存浪费在空气上。

L2 只加一层：**把「每序列连续预留」换成「块池 + block table」**。

| | L1 | L2 |
|---|---|---|
| KV 住在哪 | `pool[slot][:, :len]` 连续条 | 物理块池 `[N_BLOCKS, KV_HEADS, BS, HEAD_DIM]`，按块散布 |
| 谁记录位置 | slot 号（隐式连续） | block table：逻辑块 → 物理块（显式映射） |
| 何时分配 | 进场即整条 | 写满当前块才要下一块（按需） |
| 序列结束后 | 槽位空等复用 | 块归还 free queue，**哈希留在前缀缓存** |
| 共享 | 无 | 内容哈希命中 → touch（ref_cnt+=1） |

**跨级别契约做成机器断言**：同一权重、同一 prompt、greedy——
`paged decode == L1 连续池 decode（逐 token）`（[1] 板块）。
分页改变「KV 住在哪」，不改变「算什么」：gather 只是把同一批数值按同一顺序摆回
`k_full`，注意力其后的每一步算术与 L1 逐位相同，所以生成必须逐 token 一致——
不一致就是分页实现错了。这条断言是 L2 的地基，后面所有实验（共享 / CoW / 抢占）
都复用同款断言：「参考轨迹 vs 机制介入后的轨迹，逐 token 比」。

---

## 3. 块池与 block table：按需供给（[0][1]）

物理块池启动即划走整块（0.5 MiB = 64 块 × 4 tok × 2048 B/tok，字节数 == 公式断言）——
与真实引擎「启动即占满 KV 显存」同构（L1 [0] 的池也是预分配）。**块是最小供给单位**：
序列要 KV 空间，只能整块地要。

[1] 板块拍下按需分配的全过程：prompt 8 tok 占 2 块；此后第 9、13、17…个 token 写入时
各加 1 块——**加块时刻 = 写满当前块的那一刻**，轨迹与 `ceil(tokens/BS)` 全程逐位一致
（断言）。没有预留、没有超发：序列在任意时刻占用的槽位 = 它实际写过的 token 数向上取整。

一个工程细节：forward 前**必须先分配**（调度器 `ensure_blocks` 在 `forward` 之前），
因为 scatter 的目标物理块必须已存在。vLLM 同序：scheduler 先 `allocate_slots`，
worker 再跑 kernel——分配失败（块不够）是调度层的事件，不是 kernel 的异常（§7）。

gather/scatter 用一次向量化索引完成（`pool[blk_idx, :, offset]`）——真实引擎把这步
**融进 paged-attention kernel**：kernel 拿着 block table 直接读散布的物理块，
根本不拼连续缓冲。nano 的向量化索引是它在 CPU 上的最小同构体；逐块 Python 循环
是它的反面教材（§12.2，第一轮实现真的踩过）。

---

## 4. 碎片与准入：省下的块就是吞吐（[2]）

8 个真实请求（L0/L1 同款 `lengths=[6,2,9,3,14,5,1,8]` + 8-token prompt，总 token
`[14,10,17,11,22,13,9,16]`，Σ=112），两种分配策略在**同一预算**（分页峰值 32 块）下比准入：

- **分页**：峰值 32 块 = 128 槽位，内部碎片 16（每请求 ≤ BS−1 = 3）→ **准入 8/8**。
- **连续预留**：每请求按最长（22 tok = 6 块）预留 → 192 槽位、浪费 80（42%）→ **准入 5/8**。

账本与 L0 [4] 的纯算术完全对上（那里算过「page 太大 = 内部碎片回潮」），但 L2 把它
推进了一步：**准入差直接就是并发度差**（8 vs 5）。L1 [2] 已经实测吞吐随 B 单调上升
（decode 摊薄权重读取），于是链条闭合：

```
分页 → 内部碎片消失 → 同预算更多并发 → 摊薄固定读取 → 吞吐
```

这就是 PagedAttention 论文（arXiv:2309.06180）的核心主张在 nano 上的完整复现：
**显存效率不是孤立指标，它通过并发度换算成吞吐**。

---

## 5. 前缀共享：内容寻址 + 因果性 = 无损共享（[3] 上半）

A 先跑：132-token prompt（128 共享前缀 + 4 私有）写满 33 块，**每个满块登记一个链式哈希**：

```
h_j = sha256(h_{j-1}, 块 j 的 token 序列)，h_{-1} = 0
```

链式哈希使「同哈希」必然「同前缀内容」——块的内容寻址不只看自己，还看整条前缀链
（对照 vLLM `hash_block_tokens(parent_block_hash, curr_block_token_ids)`，§8）。

B 进场：134-token prompt 的前 128 token 与 A 相同 → 链式哈希逐块命中 32 块 →
**touch**（ref_cnt 1→2，块从 free queue 移出）→ 只新分配 2 块、prefill 只算后缀 6 token。

两个实测数字：

1. **省显存**：B 的 34 块里 32 块是借的（ref_cnt=[2,2] 在盘），物理块只多花 2 块。
2. **省算力**：prefill 墙钟 5.92 ms → 2.31 ms（**2.6×**，4 次运行区间 2.2×–2.6×）。

为什么只有 2.6× 而不是 134/6 ≈ 22×？因为命中路径**仍要付三笔钱**：
(a) 前缀 KV 的 gather（32 块要读回来）；(b) 6 个后缀 token 对全部 134 个位置的
attention 读取——**缓存省的是命中段的计算，不省新 token 对前缀的读取**；
(c) 每次 forward 的固定开销。prompt 越长，(a)(c) 占比越小，加速比越逼近
「全长计算 / 后缀计算」——这正是生产里前缀缓存对长 system prompt / few-shot
前缀的 TTFT 收益来源。同时注意 (b)：前缀缓存不改变「decode 是带宽瓶颈」的物理
（L0/L1 的主线），它只砍 prefill 计算。

**共享为什么无损？** 断言 `共享版 B 生成 == 无共享参考（逐 token）` 通过不是巧合：
因果 attention 下，位置 t 的 KV 只依赖 ≤t 的 token——共享前缀的 KV 在 A 的上下文里
算什么，在 B 的上下文里还是什么。**因果性是前缀共享的物理基础**；反之，任何
双向/非因果的编码（如 encoder 的 cross-attention）都不能这样共享。

---

## 6. Copy-on-Write：共享块只读，写时分裂（[3] 下半）

共享引入了一个新问题：**两个序列指着同一物理块，一方要写怎么办？**

实验：P 有 10 个 token（2 满块 + 1 半块，半块里有 2 个 token 的 KV），fork 出两个孩子
（parallel sampling 的最小形态）：复制 block table、逐块 touch——三块全部共享，
ref_cnt = [3, 3, 3]（P + c1 + c2）。然后两个孩子各自生成：

```
child1 写入共享半块（写前 ref_cnt=3 > 1）→ 复制块 2 → 3，原块降为 2
child2 写入共享半块（写前 ref_cnt=2 > 1）→ 复制块 2 → 4，原块降为 1
```

规则只有一条：**目标块 ref_cnt > 1 → 先块级复制、原块降引用、block table 改指新块，
再写**（copy-on-write）。于是：

- 父 P 的半块内容在两次 CoW 后**逐字节不变**（张量快照比对断言）——共享块只读；
- 两个孩子各自持有私有副本后，后续写入不再触发 CoW（ref_cnt=1 = 独占）；
- child1/child2 全轨迹 == solo 参考（14 token 逐个）——CoW 没有改变任何语义。

vLLM V1 里 CoW 的活实例是**部分前缀命中**：缓存块里存着别的请求算出的 16 个 token，
新请求的前缀在块中间分叉（只命中前 k < 16 个），要往块尾写自己的 KV → 先复制
（`_pending_cow_copies` 登记 (src, dst)，forward 前在 GPU 上执行块拷贝，§8 锚点）。
PagedAttention 论文则把 CoW 用于 parallel sampling——两处用途，同一条规则。

---

## 7. 释放复用与抢占/重算：过载是可恢复的代价（[4]）

**[4a] 释放 ≠ 清空**。r0 完成 → 4 块归还 free queue（free 0→4）；同前缀的 r1 进场，
3 个满块**全部命中「已释放」的块**——ref_cnt 0→1、移出 free queue、只重算 1 个 token
（全命中时至少要算一格才能产出下一个 token，vLLM 同款约束）。关键设计：
**ref_cnt=0 的块同时是「空闲内存」和「缓存条目」**——块的内容与哈希在释放后保留，
直到被真正重分配时才失效（nano 在 `get_new_blocks` 里删旧哈希条目，对照 vLLM
`_maybe_evict_cached_block`）。这让前缀缓存零额外空间：缓存不是另一份副本，
是「还没被覆盖的空闲块」。

**[4b] 抢占与重算**。预算压到 12 块（8 请求峰值需 32 块），调度器循环：
准入（free ≥ 该请求全程所需块，`full_sequence_must_fit` 同构门槛）→ 每迭代各前进一步
→ 跨块时分配 → **分配失败 → 抢占最晚的运行请求**（FCFS，对照 vLLM `running.pop()`）：
释放其全部块、`num_computed` 归 0、回等待队列；恢复时对 prompt+已生成 token
**全量重跑 prefill**（recompute）。实测：

```
完成迭代 17 | 抢占 8 次 | 重算恢复 3 次（重算 prefill 共 29 tok）
抢占/重算后 8 个请求最终 token == 无压力参考（逐个）: True ✅
```

三个值得盯着看的点：

1. **语义不变，代价实付**：被抢占者已生成的 token 保留在请求里（vLLM 的
   `Request.token_ids` 同构），重算的是 KV 不是输出——greedy 下重算后的轨迹与
   无压力参考逐 token 一致（断言）。抢占付出的 = 29 个 token 的重复 prefill。
2. **抢占是准入的镜像**：准入检查只看「当前 free」，不为运行中请求的未来增长预留
   → 必然出现过载 → 抢占兜底。若把门槛改成「free ≥ 新请求全程 + 运行中请求剩余增长」，
   抢占会消失（思考题 2）——vLLM 的 `reserved_blocks` 参数正是这类预留的旋钮。
   nano 刻意用简单门槛，让抢占机制真的被跑到。
3. **被抢占多次的请求**（请求 5：iter 3 被抢、恢复、iter 7 再被抢、再恢复）——
   重算成本随已生成长度线性涨（9 tok → 11 tok），这是「为什么生产调度要尽量
   别抢占长序列」的算术根源。

---

## 8. 与 vLLM V1 源码对照（2026-08-06 main 快照核验）

权威实现：`github.com/vllm-project/vllm`（V1 引擎）。锚点全部在 2026-08-06 下载的
main 快照上逐行核验（快照与核验细节见 §14；`config/cache.py:L47` 另有 2026-08-06
raw 通道独立核验，零漂移；§8 全部锚点在 2026-08-07 再经 raw 通道复核，结论见 §14）。

| nano L2 | vLLM V1 | 锚点 | 差异与原因 |
|---------|---------|------|-----------|
| `BlockPool.get_new_blocks`（free queue popleft、ref_cnt 0→1、不够即 ValueError） | `BlockPool.get_new_blocks` 同款（`popleft_n` + ref_cnt+=1 + raise） | `v1/core/block_pool.py:L647` | 同构；nano 的失败异常是调度层抢占的触发器，vLLM 中由 scheduler 捕获处理 |
| `Block(ref_cnt, block_hash)` | `KVCacheBlock`（ref_cnt / block_hash / free-list 双向指针） | `v1/core/kv_cache_utils.py:L118` | nano 无 null block / metrics 字段 |
| free queue = `deque` FIFO | `FreeKVCacheBlockQueue` 双向链表 | `v1/core/kv_cache_utils.py:L184`（`popleft_n`:L273 / `prepend_n`:L349 / `append_n`:L370） | **vLLM 的释放顺序有讲究**：有哈希的块 append 到尾部（LRU 保缓存）、无哈希的 prepend 到头部（先被复用）（`free_blocks`:L719）；nano 一律 FIFO——驱逐顺序影响前缀缓存命中率，见思考题 5 |
| `touch`（ref_cnt+=1，ref_cnt=0 则移出 free queue） | `touch` 逐字同款语义 | `v1/core/block_pool.py:L702` | 同构 |
| `free_blocks`（ref_cnt-=1，归零回队，哈希保留） | `free_blocks` 同款（哈希块/无哈希块分队列位置） | `v1/core/block_pool.py:L719` | 见上行差异 |
| 重分配时删旧哈希条目 | `_maybe_evict_cached_block`（reset hash + 移出缓存表） | `v1/core/block_pool.py:L679` | 同构——不做这步，前缀命中会拿到内容已被覆盖的块（静默错误） |
| 链式 `block_hash`（sha256(parent, tokens)） | `hash_block_tokens(parent_block_hash, curr_block_token_ids)` | `v1/core/kv_cache_utils.py:L576`（哈希函数 `sha256_cbor`/`xxhash_cbor`:L18） | 同思想；vLLM 的 extra_keys 还可挂 LoRA 等上下文 |
| 只缓存满块 | `cache_full_blocks` | `v1/core/block_pool.py:L225` | V1 另有细粒度 partial 条目（`_block_hash_num_tokens`，块内前缀也可入缓存，`resolve_kv_cache_block_sizes`:L606）——nano 留经典策略 |
| 准入：free ≥ 全程所需块 | `allocate_slots(..., full_sequence_must_fit=True)`；返回 None = 拒绝 | `v1/core/kv_cache_manager.py:L344`（watermark:L171） | nano 无 watermark / chunked prefill / reserved_blocks |
| 抢占最晚运行者（FCFS） | `preempted_req = self.running.pop()` | `v1/core/sched/scheduler.py:L616`（PRIORITY 策略则按优先级选） | 同款 FCFS 默认 |
| 抢占 = 释放块 + `num_computed=0`（重算） | `_preempt_request`：`_free_request_blocks` + `num_computed_tokens = 0` | `v1/core/sched/scheduler.py:L1275/L1291/L1295` | 同构——V1 抢占即重算（token 保留在请求里） |
| CoW：写共享块前先复制 | partial 前缀命中 CoW：`_pending_cow_copies` 登记 (src,dst)，forward 前 GPU 块拷贝 | `v1/core/single_type_kv_cache_manager.py:L114-117`；`v1/worker/gpu/model_runner.py:L959-966`（现行 main 位移至 L954-960，见 §14） | nano 用 fork/parallel-sampling 场景演示同一条规则（论文原始用途）；V1 的触发点是部分命中 |
| `BLOCK_SIZE=4`（轨迹可见） | `DEFAULT_BLOCK_SIZE = 16` | `vllm/config/cache.py:L47` | 块大小 = 内部碎片 vs block table 开销 vs kernel 效率的三方折中（L0 [4] 扫过页大小） |
| — | SGLang RadixAttention：前缀组织成 radix tree | 论文 arXiv:2312.07104 | 源码对照留 L3（README 阶梯定义）`[TODO: verify]` |

**nano 与权威实现最大的三处分歧**（都是刻意的教学取舍）：
其一，free queue 用 FIFO 而非「哈希块沉底」的 LRU——少了驱逐策略变量，前缀缓存
命中率的讨论才有一个干净的基线；其二，单 KV 组、全注意力——V1 的
`kv_cache_coordinator` 要协调 sliding-window / MLA / mamba 多组缓存，那是 L3 的
复杂度；其三，调度是教学用的三阶段循环（前进→释放→准入），vLLM 的 scheduler
还要管 token budget / chunked prefill / spec decode——但「分配失败 → 抢占最晚者 →
重算恢复」这条主回路，两边是同一条。

---

## 9. 机制深潜：分页是把内存管理变成数据结构问题

把四个实验并排，L2 讲的其实是一件事——**当显存成为第一瓶颈，内存管理从
「给每序列划一条」变成一组数据结构，每个结构对应一个语义承诺**：

| 数据结构 | 语义承诺 | 被哪个断言检查 |
|----------|----------|----------------|
| block table（逻辑→物理） | 分页不改变注意力数学 | paged == L1 逐 token |
| 按需分配 + 块级 gather | 占用 = 实际使用（±BS−1） | 块数 == ceil(tok/BS) 全程 |
| ref_cnt | 共享块只读 | CoW 恰 2 次 + 父块字节不变 |
| 内容哈希（链式） | 同哈希 = 同前缀 = 可复用 | 命中后生成 == 无共享参考 |
| free queue + 哈希保留 | 释放 ≠ 清空（空闲即缓存） | 已释放块被同前缀命中 |
| 准入/抢占 | 过载可恢复、语义不变 | 抢占后 token == 无压力参考 |

三条 senior 判断：

1. **refcount 一肩挑三个角色**——共享（>1）、在用（≥1）、空闲且可缓存（=0 且有哈希）。
   vLLM 没有为前缀缓存单独建一套副本管理，缓存就是「还没被复用的空闲块」——
   零额外空间的缓存不是优化技巧，是数据结构的定义方式。
2. **内容寻址把「能不能共享」变成了可判定的查询**。链式哈希让共享判定 = 一次
   dict lookup，与序列身份、到达顺序无关——同一个 system prompt 的一千个请求
   天然共享，不需要任何协调。RadixAttention 用树、vLLM 用哈希表，载体不同，
   「前缀是内容寻址的」这一点相同。
3. **抢占把「拒绝服务」改成了「延迟服务」**。块不够时不是 OOM 崩溃也不是拒绝准入，
   而是付出重算代价换系统继续运转——且代价可预算（已生成 token 数 × 重算单价）。
   这是「可靠性 = 存储/调度设计，不是祈祷」的又一例（与 04 轨 nano-qwenpaw L1
   的记忆可靠性同构：可靠性是设计出来的属性）。

---

## 10. 费曼：标准托盘仓库

把 KV 显存想象成一个**仓库**，token 是货物。

- **L1 的做法**：每个客户（序列）进场，仓库给他**围一整片地**（512 格），
  不管他实际放多少货——地围多了空着，围少了不够放。
- **L2 的做法**：仓库改用**标准托盘**（块，4 格/托）。客户的货放在哪些托盘上，
  记在一张**提货单**（block table）上——托盘可以分散在仓库任何角落。
  放满一托才领下一托（按需分配）；客户走了托盘收回**待用区**（free queue），
  但托盘上的货和**标签**（哈希）先不撕——下个存同样货的客户来了，
  直接指着托盘说「这些算我的」（touch，refcount+1），不用重新进货（省算力）。
  两个客户共用一托时，谁要往托上加新货，仓库先**复制一托**给他改（CoW）——
  共用的货永远不动。仓库爆满时，最晚来的客户被请出去等（抢占），
  他的货位让给别人；轮到他回来时，按订单重新进一遍货（重算）——
  订单（token 序列）一直在他手里，所以最终交付的货一分不差。

一句话版：**窗口时代的 KV 管理是「圈地」，分页时代是「托盘 + 提货单 + 共享标签」**。

类比的边界：真实仓库的「复制托盘」是 GPU 块拷贝（一次 kernel），nano 里是 CPU 张量
拷贝；真实仓库的待用区排序有讲究（LRU + 哈希沉底），nano 是 FIFO；真实仓库还要
处理多种货架（sliding window / MLA 多组），nano 只有一种。

---

## 11. 思考题

1. **块大小算术**：BLOCK_SIZE 从 4 改 16（vLLM 默认），[2] 的内部碎片上界
   （每请求 ≤ BS−1）与 block table 条目数各变成多少？回看 L0 [4] 的页大小扫描，
   为什么真实引擎选 16 而不是 1 或 512？（提示：kernel 的访存合并与表开销。）
2. **消灭抢占**：把 [4b] 的准入门槛改成「free ≥ 新请求全程块数 + 所有运行中请求
   的剩余增长」（reserved 版）。预测抢占次数会变成几，跑一遍验证；再想：
   vLLM 为什么默认不做满预留？（提示：短序列提前完成时，预留会白白压低并发。）
3. **命中加速的上界**：[3] 的命中加速是 2.6× 而非 134/6≈22×。列出命中路径仍要付
   的三笔成本；若共享前缀从 128 涨到 4096 token、后缀仍 6 token，加速比会向哪个
   方向走？哪一笔成本最终成为天花板？（提示：attention 读取 ∝ 前缀长。）
4. **链式哈希的强度**：证明——两个请求的逻辑块 j 映射到同一物理块，当且仅当
   它们的前 (j+1)×BS 个 token 完全相同（在哈希无碰撞的前提下）。如果只用
   「本块 token」做哈希（不链 parent），会出什么错？（提示：同块内容、不同前缀
   的 KV 相同吗？）
5. **驱逐顺序的代价**：vLLM `free_blocks` 把有哈希的块放队尾、无哈希的放队头
   （`block_pool.py:L719-742`）。构造一个请求序列说明：FIFO（nano）与
   「哈希沉底」（vLLM）在前缀缓存命中率上的差异；nano 的 [4a] 为什么观察不到
   这个差异？（提示：池里只有一种前缀。）

---

## 12. 反例与边界

1. **CPU 口径不可外推**：所有墙钟（[3] 的 2.6×）是 CPU 小模型口径——真实引擎的
   gather 融进 paged-attention kernel、GPU 带宽物理不同。机制（块/refcount/CoW/抢占）
   可外推，绝对数字不可。真机验证 `[TODO: verify on real system]`（真实 GPU/多机环境）。
2. **逐块 Python gather 是反面教材**：第一版实现用逐块切片循环，[3] 命中加速只有
   1.1×——每块一次的 Python 分发开销在两条路径都付，把计算差吞了。改向量化索引
   （更接近 kernel 融合）后才显出 2.6×。**测量协议决定你测出什么**（L1 同款教训：
   探针形状决定结论）。
3. **nano 的调度不建模真实 SLO**：抢占选「最晚者」是 FCFS 教学版；生产调度要权衡
   优先级 / 延迟 SLO / 重算成本（vLLM 有 PRIORITY 策略分支）。token budget /
   chunked prefill / spec decode 一律不建模。
4. **贪心 + 随机权重**：跨级别契约依赖 greedy 确定性；采样下「轨迹逐 token 一致」
   要改成「同分布」，但内存机制本身与采样无关。
5. **fork 实验的第一版 bug（如实记录）**：P 的 prefill token 忘了 append 进 tokens
   就 fork，两个孩子的轨迹整体错位一格——而「生成列表相比」的初版断言因两侧
   同样错位而险些放行。改成**全轨迹比较**（child.tokens == solo.tokens）后暴露。
   教训：断言要钉在「完整状态」上，不是钉在「增量」上。

---

## 13. 阶梯预告 + 交叉引用

- **L3（对照源码做多请求前缀共享分析）**：对照 vLLM `kv_cache_manager` /
  `kv_cache_coordinator` 与 SGLang RadixAttention（radix tree vs 哈希表的前缀组织
  取舍、partial 条目、多 KV 组协调）`[TODO: verify]`；真机吞吐与 KV 占用测量
  `[TODO: verify on real system]`。
- 交叉引用：分页省显存→并发→吞吐的链条，L0 [4] 是算术版、L1 [2] 是实测吞吐曲线版；
  「可靠性是存储/调度设计」与 [nano-qwenpaw L1](../../04-llm-to-agent/nano-qwenpaw/tutorial_L1.md)
  （窗口是 cache，store 才是 memory）同构；RL rollout 的生成侧成本（01 轨 nano-slime L1
  的 G/T 账）最终由本节这类引擎买单。

---

## 14. 溯源与校准

**源码快照（vLLM）**：`vllm-project/vllm` main 分支，2026-08-06 本地下载快照
（本地 vLLM 源码快照，含 `v1/core/` 完整 V1 引擎目录）。初次记录时（2026-08-07 00:3x–01:1x，
tutorial 定稿 01:10）`raw.githubusercontent.com` 不可达（curl 多次 exit 28），§8 锚点全部
基于该快照逐行核验；其中 `config/cache.py:L47`（`DEFAULT_BLOCK_SIZE = 16`）另有
2026-08-06 raw 通道独立核验记录，与快照零漂移。
**raw 通道双通道复核已完成（修订）**：2026-08-07 raw 通道恢复后，
现场抓取 vllm main 6 文件复核 §8 全部锚点——`block_pool.py` / `kv_cache_manager.py` /
`single_type_kv_cache_manager.py` / `config/cache.py` 与 08-06 快照逐字节相同（零漂移）；
`kv_cache_utils.py` / `sched/scheduler.py` / `worker/gpu/model_runner.py` 同日微漂移
（vllm main 提交频繁），锚点仍于行号级有效；model_runner CoW 段现行 main 位于 L954-960
（快照行号 L959-966，代码内容相同）。SGLang `radix_cache.py` 同因初次记录时不可达，
源码对照按 README 阶梯定义留 L3。

**锚点清单**（快照行号级）：`block_pool.py:L647`（get_new_blocks）/ `L679`
（_maybe_evict_cached_block）/ `L702`（touch）/ `L719`（free_blocks）；
`kv_cache_utils.py:L118`（KVCacheBlock）/ `L184`（FreeKVCacheBlockQueue，
popleft_n:L273 / prepend_n:L349 / append_n:L370）/ `L576`（hash_block_tokens）/
`L606`（resolve_kv_cache_block_sizes）；`kv_cache_manager.py:L344`（allocate_slots）/
`L171`（watermark）；`sched/scheduler.py:L616`（FCFS running.pop()）/ `L1275/L1291/L1295`
（_preempt_request / free / num_computed_tokens=0）；`single_type_kv_cache_manager.py:L114-117`
（_pending_cow_copies）；`worker/gpu/model_runner.py:L959-966`（CoW 块拷贝执行）。

**论文**：PagedAttention / vLLM arXiv:2309.06180、SGLang arXiv:2312.07104
（abs 页 2026-08-06 核验）；iteration-level scheduling：Orca（OSDI 2022，
usenix 页 2026-08-06 核验）。

**确定性记录**：greedy + seed=42 + 固定 prompts（独立 Generator）——计数类输出
跨运行逐字节一致；计时行仅 [3] prefill 墙钟及其派生比值（3 次连续运行：5.92/2.31 ms
→ 2.6×，6.11/2.35 ms → 2.6×，第三遍比值 2.3×；定稿后确认跑 4.24/1.89 ms → 2.2×，
比值出 3 遍区间下界——该跑与只读命令并发、负载略高，如实补注；计数类 mask 后
四遍全部逐字节一致）。已知实现陷阱：fork off-by-one（§12.5）+ 逐块 gather 吞掉
命中差（§12.2）+ 块重分配必须失效旧哈希（否则前缀命中拿到被覆盖的块——机制先行，
虽然 nano 调度路径未开缓存）。

**环境**：Apple M5 Pro / Python 3.13.13 / torch 2.13.0（`python3`）/
CPU fp32 / seed=42。真实引擎与 GPU 数字：`[TODO: verify on real system]`。
