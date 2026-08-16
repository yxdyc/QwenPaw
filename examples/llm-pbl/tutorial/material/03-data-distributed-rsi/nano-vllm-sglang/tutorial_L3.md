# nano-vllm-sglang L3 — RadixAttention：radix tree 前缀缓存与多请求共享

> L2 把「前缀缓存」做成了链式内容哈希 + FIFO free queue：同前缀的请求 touch 同一物理块，
> 释放的块带着哈希留在队里等复用。这套机制能省，但**省多少全凭运气**——FIFO 不懂「前缀」，
> 它只按释放顺序发货。L3 把缓存的组织换成一棵 **radix tree**（SGLang RadixAttention 的最小
> 同构实现）：共享是节点、驱逐是剪叶、调度看树选请求——「省算力」从运气变成策略。
> 顺带，L3 的压力场景还从 L2 里挖出一个真竞态（§6）——这是本节的重要教学材料。

---

## 1. 先跑为敬

文件：`L3_radix_prefix_sharing.py`，依赖仅 `torch` + 同目录 L1/L2 模块（import 时设
`sys.dont_write_bytecode`，不落 `__pycache__`）。CPU 即跑（本机实测约 10 秒）。

```bash
$ python3 L3_radix_prefix_sharing.py
```

**可运行性契约声明（课程可运行性契约）**——输出开头也打印了同样的声明：

- **权重是 L1 的随机初始化 GQA GPT**（3,148,032 参数，`state_dict` 共享，逐参数断言相等）。
  真实权重 + 真实引擎（vLLM/SGLang on GPU）仍待真实 GPU/多机环境验证 `[TODO: verify on real system]`。
- **radix tree 不是模拟、不是 API 包装**：真实树结构 / match-split-insert / LRU 叶子驱逐 /
  lock_ref 保护 / 块级物理共享都是最小同构真实现——只是规模小、跑在 CPU、请求顺序执行。
- **[4] 的 L2 对照组跑在修复后的 L2 引擎上**：L2 的 `paged_prefill` 曾被本模块的压力场景
  挖出 allocate-before-touch 竞态（§6 实录），已修为 touch-before-allocate（vLLM V1 两阶段
  分配同因，issue #33775）。L2 自场景输出不变性证明见 §6.5。
- 计时行仅 [3] 的 `elapsed` 行（随机器负载浮动），可用 `sed '/^[[:space:]]*elapsed/d'`
  整行掩码（继承 L2 口径）。下面 paste 块即掩码后的输出，双独立 CWD 两遍运行逐字节一致
  （锚点见 §12）。

真实输出（2026-08-13 复现运行，掩码后逐字节粘贴）：

```text
====================================================================
nano-vllm-sglang L3 — RadixAttention：radix tree 前缀缓存 + 多请求共享
====================================================================
torch 2.13.0 | CPU fp32 | seed=42 | BLOCK_SIZE=4
声明: 权重 = L1 随机初始化 GQA GPT（state_dict 共享，逐参数断言相等）；
      radix tree 为最小同构真实现（非 API 包装），对照 SGLang
      radix_cache.py / vLLM v1 block_pool.py（2026-08-13 main 抓取，
      tutorial_L3 §7）；真实引擎 (vLLM/SGLang on GPU) 见
      [TODO: verify on real system]

[1] 跨级别契约（prompt=24 tok，生成 4，greedy）
    第 1 次冷跑: 命中 0，实算 24 tok；第 2 次同 prompt: 全命中（24 tok 全在树里），只重算最后 1 格（实算 1）
    radix 命中路径 == radix 冷跑 == L1 连续池（逐 token）: True ✅

[2] 并发共享（4 个请求同时在跑，共享 24-token system prompt）
    准入后树: 1 个脊柱节点持 6 个共享块，lock_ref=4（= 运行中请求数，驱逐禁区）
    物理块占用: 唯一块 18 = 共享 6 + 私有 4×3；若不共享 = 36（每请求各持一份）→ 省 18 块 (50%)
    共享块全程只读（并发 prefill+decode 后逐字节不变）: True ✅；4 条轨迹 == 各自 solo 参考: True ✅

[3] 树从 insert 里长出来（rA1 = SYS+DOCA+sA1，rB1 = SYS+DOCB+sB1）
    rA1 冷跑后（一条链，52 tok 全算）:
root
    └ n1 [52 tok / 13 blk] lock=0 acc=t3 key=[457, 207, 312, 260]…
        └ n2 [4 tok / 1 blk] lock=0 acc=t3 key=[193, 58, 290, 288]
    rB1 进场: 前 32 tok 命中 SYS 脊柱 → 边中分裂（DOCA/DOCB 在第 32 tok 分叉），只实算 20 tok
root
    └ n3 [32 tok / 8 blk] lock=0 acc=t6 key=[457, 207, 312, 260]…
        └ n1 [20 tok / 5 blk] lock=0 acc=t4 key=[373, 132, 300, 448]…
            └ n2 [4 tok / 1 blk] lock=0 acc=t3 key=[193, 58, 290, 288]
        └ n4 [20 tok / 5 blk] lock=0 acc=t6 key=[83, 230, 40, 457]…
            └ n5 [4 tok / 1 blk] lock=0 acc=t6 key=[0, 87, 0, 87]

[4] 叶子优先 LRU vs 块级 FIFO（7 请求同序列、同预算 20 块、无缓存基线 324 tok；L2 引擎含 touch-before-
    allocate 修复，见 §6）
    请求         radix 实算      L2 哈希链实算   说明
    rA1              52            52   冷跑（树空）
    rB1              20            20   命中 SYS 脊柱
    F1               32            32   one-shot 洪水 ①（9 块）
    rB_warm          20            52   回访 B 分支
    F2               32            32   one-shot 洪水 ②（9 块）
    rA2              20            52   回访 A 分支
    rB2              20            20   回访 B 分支
    合计实算: radix 196 tok / L2 哈希链 260 tok / 无缓存 324 tok（radix 省 39%，比哈希链再省 64 tok）
    radix 驱逐日志（clock, 节点, 块数）: [(7, 2, 1), (7, 1, 5), (7, 5, 1), (7, 4, 5), (10, 7, 1), (10, 6, 8), (13, 9, 1), (13, 8, 5), (16, 11, 1), (16, 10, 8)]
    SYS 脊柱在全部驱逐中存活: True ✅（叶子优先 → 祖先可复用，直到自己变成叶子）
    命中轨迹（tok）: radix [0, 32, 0, 32, 0, 32, 32] / L2 哈希链 [0, 32, 0, 0, 0, 0, 32]
    FIFO 截断点: rB_warm（L2 hit 0 vs radix hit 32）、rA2（L2 hit 0 vs radix hit 32）——FIFO 按释放顺序复用，脊柱块最先释放、最先被复用，哈希链从头部断掉（§4/§10 解读）
    语义不变: radix/L2 两引擎 7 个请求最终 token == L1 参考: True/True ✅（内存组织不改『算什么』）

[5] cache-aware 调度（预算 15 块，两组前缀无法共存；到达序 ['A1', 'B1', 'A2', 'B2', 'A3', 'B3']）
    FCFS 准入序 ['A1', 'B1', 'A2', 'B2', 'A3', 'B3']: 实算 [32, 32, 32, 32, 32, 32] = 192 tok（每组前缀都被对方挤掉 → 每个请求都冷跑，cache thrashing）
    LPM  准入序 ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']: 实算 [32, 8, 8, 32, 8, 8] = 96 tok（同组连续进场，前缀住在树里 → 组内只冷跑一次）
    调度只改『谁先算』: 六个请求最终 token 与顺序无关（FCFS==LPM==L1 参考）: True ✅

====================================================================
✅ self-check passed:
   radix 命中路径 == L1 连续池逐 token 一致 / 并发 4 请求共享脊柱块
   （lock_ref=4，物理块 18 vs 36）且轨迹 == solo / 分裂让分支共享脊柱 /
   叶子优先 LRU 保住 SYS 脊柱（radix 196 vs 哈希链 260 vs 无缓存 324 tok）/
   LPM 消除 thrashing（96 vs 192 tok）且语义与顺序无关 / 全程零 CoW
====================================================================

takeaway: 前缀缓存的两种权威组织——vLLM 用链式内容哈希（块级、隐式
          驱逐），SGLang 用 radix tree（显式 LRU 叶子驱逐 + lock_ref 保护
          + cache-aware 调度）。树把『前缀』变成一等数据结构：共享是节点,
          驱逐是剪叶，调度看树选请求——于是『省算力』从运气变成策略。
          语义（生成什么）一分不差，差别全在代价（实算 token / 块存活）。
```

---

## 2. K+1：L2 留下什么，L3 接住什么

L2 的前缀缓存（tutorial_L2 §5/§7）是**链式内容哈希**：每个满块登记
`hash(父块哈希, 本块 token)`，后来者沿链查找，命中即 touch。它已经做到三件事：
同前缀共享物理块、释放 ≠ 清空（ref_cnt=0 的块仍是缓存条目）、重分配即失效哈希。

但 L2 的 [4] 场景（tutorial_L2 §7）只演示了「释放复用 + 抢占」，**没有给缓存上压力**——
64 块池子里 8 个小请求，FIFO 从未卷到缓存块头上。一旦预算收紧、洪水请求反复冲刷池子，
哈希链方案有三个做不到的事，正是 L3 的三个场景：

1. **并发共享的保护粒度**。L2 的 ref_cnt 住在**块**上：运行中请求持有的块 ref_cnt≥1，
   不会被 get_new_blocks 弹出。但「一条前缀作为一个整体不可驱逐」这件事没有名字——
   调度器无法回答「这段 system prompt 现在被几个请求用着」。SGLang 的答案是树节点上的
   `lock_ref`：论文 §3 原文——"each node maintains a reference counter indicating how
   many running requests are using it. A node is evictable if its reference counter is
   zero."（ar5iv 208,367 B 抓取件 grep 在位，§12）。[2] 跑出 lock_ref=4 的脊柱。
2. **驱逐的顺序**。L2 的 FIFO 按**释放顺序**复用块——它不知道哪些块属于同一条前缀，
   更不知道「脊柱块比叶子块值钱」。SGLang 的答案是叶子优先 LRU：论文 §3 原文——
   "we introduce a simple LRU eviction policy that evicts the least recently used leaf
   first. By evicting leaves first, we enable the re-use of their common ancestors
   until those ancestors become leaves and are also evicted." [4] 用 20 块紧预算 +
   两轮洪水实测：radix 的 SYS 脊柱全程存活，哈希链的脊柱被 FIFO 卷走、回访命中归零。
3. **调度看缓存**。L2 的调度（tutorial_L2 §7）只看块预算，不看命中。SGLang 的答案是
   cache-aware 调度：论文 §3 原文——"if the request scheduler frequently switches
   between different, unrelated requests, it can lead to cache thrashing and a low hit
   rate"，解法是 "sort the requests by matched prefix length and prioritize requests
   with longer matched prefixes instead of using a first-come, first-served schedule"，
   即 LPM（longest prefix match，schedule_policy.py:L198）。[5] 用交错到达序实测：
   FCFS 192 tok vs LPM 96 tok，且六个请求最终 token 与顺序无关。

L3 只加这一层：**缓存的组织从哈希链换成 radix tree**。模型、分页引擎、跨级别契约
（逐 token 一致）全部继承 L1/L2，一行不改——L3 改的只是「KV 住在哪、何时算、谁先算」。

---

## 3. 代码结构：一个池、一棵树、一条生命周期

```
L3_radix_prefix_sharing.py（672 行）
├── RadixBlockPool(L2.BlockPool)      # 物理块池：只管空闲块，不管共享语义
│     get_new_blocks / free_blocks    #   （对照 SGLang token_to_kv_pool_allocator）
├── Node / RadixCache                 # radix tree 前缀缓存（SGLang RadixAttention 最小同构）
│     _walk / _split                  #   走查 + 边中分裂
│     match_prefix / insert           #   最长前缀匹配 / 登记入树
│     evict                           #   LRU 叶子驱逐（堆 + 整叶释放）
│     inc_lock_ref / dec_lock_ref     #   运行中保护（沿父链到 root）
├── RadixReq / admit / radix_prefill / radix_decode_step / finish / run_radix
│                                     # 请求生命周期：准入(匹配→驱逐→分配→入树→加锁)
│                                     #   → prefill(命中段跳过) → decode → 完成(全量入树→解锁)
├── l2_hit_len / run_l2               # L2 哈希链引擎的等价 harness（[4] 对照组）
└── main: [1]–[6]                     # 六个场景，全部机器断言
```

三个刻意的设计选择（与权威实现的差异逐条见 §7）：

- **池与树分工**。`RadixBlockPool` 只认空闲块（ref_cnt 仅作 0/1 占用断言），「谁在引用、
  何时可驱逐」全部由树负责——这是 SGLang 的分工（allocator 不管共享语义，radix tree
  管），与 L2 `BlockPool` 的「refcount + 哈希一肩挑」相反。分工的代价是两次簿记，
  收益是驱逐策略可以独立演化（SGLang 现行 main 的 priority-aware eviction 就长在树上）。
- **块粒度（BS=4，继承 L2）**而非 SGLang 的 token 粒度。SGLang 论文 §3 原文："These
  KV cache tensors are stored in a non-contiguous, paged layout, where the size of each
  page is equivalent to one token."；但现行 main 的 `page_aligned`（radix_cache.py:L135）
  在 page_size>1 时把查询 key 截断到页倍数——块粒度是它自己支持的受保护模式。nano 取
  块粒度是为了与 L2 同构对照，差异见 §7。
- **逻辑时钟**。`last_access` 用自增逻辑钟而非 `time.monotonic`——保确定性（同一输入
  跨运行逐字节一致），语义与 LRU 完全等价。

---

## 4. 输出逐段解读

**[1] 跨级别契约**。24-tok prompt 冷跑命中 0、实算 24；同 prompt 再跑全命中、只重算
最后 1 格（全命中至少算一格——vLLM/SGLang 同款约束，tutorial_L2 §1/§7 已遇）。三条路径
（radix 命中 / radix 冷跑 / L1 连续池）逐 token 相等。**缓存改的永远是「算什么」的代价，
不是「算什么」本身**——这条断言贯穿 L1→L2→L3。

**[2] 并发共享**。4 个请求共享 24-tok system prompt，准入后树里只有 1 个脊柱节点持
6 个共享块，`lock_ref=4`——驱逐禁区。物理账：唯一块 18 = 共享 6 + 私有 4×3，不共享则
36，省 50%。两个语义断言：共享块在并发 prefill+decode 全程**逐字节不变**（快照比对，
radix 世界写入只发生在请求自己的新块上 → 不需要 CoW）；4 条轨迹 == 各自 solo 参考。
「并发」在这里是轮转 decode 的顺序模拟（§10 边界），但「N 个请求同时持有一条前缀」的
簿记（lock_ref 沿父链到 root）是真的。

**[3] 树从 insert 里长出来**。rA1 冷跑后是一条链（n1 持 52 tok/13 blk + gen 节点 n2）；
rB1 进场命中前 32 tok（SYS 脊柱），在边中分裂：n3（SYS，8 blk）成为新父，n1 瘦身为
DOCA+sA1（5 blk）挂左，DOCB+sB1（n4，5 blk）挂右——**两个分支从此共享脊柱的物理块，
分裂不复制任何 KV 数据**（只切 key/blocks 表）。对照 SGLang `_split_node`
（radix_cache.py:L674）：key/value 按 split_len 切开、`new_node.lock_ref =
child.lock_ref`（同一批请求同时覆盖两段）。计时行见 §10 诚实定位。

**[4] 叶子优先 LRU vs 块级 FIFO（本节的肉）**。7 个请求同序列、同预算 20 块，radix 与
修复后的 L2 哈希链跑同一工作负载：rA1/rB1 建树 → F1/F2 两轮 one-shot 洪水（各 9 块）
→ rB_warm/rA2/rB2 回访。实测：

- radix 合计实算 **196** tok（省 39%），L2 哈希链 **260** tok（省 20%），无缓存 324。
  radix 比哈希链再省 64 tok = 恰好两个回访请求的 SYS 段（2×32）。
- **命中轨迹**：radix `[0,32,0,32,0,32,32]`，L2 `[0,32,0,0,0,0,32]`。截断点在
  rB_warm 与 rA2：L2 hit 0，radix hit 32。
- **驱逐日志**逐条可读：t7（F1 准入）弹出 n2/n1/n5/n4——全是 rA1/rB1 的**叶子与分支**
  （gen 1 块 + DOCA/sB1 系 5 块，共 12 块 ≥ 需 9）；t10/t13/t16 依次弹出上一轮洪水与
  回访的枝叶。**脊柱 n3 从未进过候选堆**——它不是叶子。
- **语义不变**：两引擎 7 个请求最终 token == L1 参考（True/True）——L2 引擎带上
  touch-before-allocate 修复后，压力场景下契约重新成立（修复前不成立，§6）。

为什么 L2 在 rB_warm/rA2 命中归零？FIFO 按**释放顺序**复用：SYS 脊柱块在 rA1 完成时
**最先**被释放（free queue 头部），rB1 命中时 touch 把它们移出队列、完成时又**按 table
顺序重新 append**——但两轮洪水（各 9 块）的需求量足以把队列卷过脊柱位置：F1 弹出
队首的脊柱块复用，`get_new_blocks` 同时删掉其哈希条目（L2 的「重分配即失效」），
链式查找在第一环就断了——**哈希链从头部断掉，后面所有块瞬间全失效**。radix 侧同样
缺水，但驱逐候选只有叶子：树枝（DOCA/DOCB 分支、gen 节点）先被剪，脊柱留到最后。
这就是论文那句话的工程含义：祖先的可复用性由「它何时变成叶子」决定，而不是由
「它何时被释放」决定。

一个更细的实测值得盯住：**L2 的 rB2 命中了 32 tok**——别误会，那不是原脊柱：
rB_warm/rA2 命中归零后全量重算了 SYS 段，把新块重新登记成链，rB2 命中的是 rA2
刚登记的副本。FIFO 的命中是**或然的**（取决于队列位置的运气），radix 的存活是
**结构的**（只要不是叶子就不可驱逐）。同一个 workload，一边是运气，一边是保证。

**[5] cache-aware 调度**。预算 15 块、两组前缀 P1/P2 无法共存，到达序刻意交错
（A1,B1,A2,B2,A3,B3）。FCFS 照单全收：每组前缀刚住进树就被对方挤掉，六个请求全冷跑
（192 tok）；LPM 按已命中前缀长度降序准入，同组连续进场（A1,A2,A3,B1,B2,B3），组内
只冷跑一次（96 tok）。调度只改「谁先算」：六个请求最终 token 与顺序无关
（FCFS==LPM==L1 参考）——**调度是代价的函数，不是语义的函数**。

**[6] self-check**。数字全部派生自上方实测变量（早期版本曾误用硬编码，现已从根上消除）：
196/260/324、96/192、lock_ref=4、18 vs 36 都来自 [2][4][5] 的
运行值。

---

## 5. 机制深挖：radix 五件套，逐行对照 SGLang 源码

对照锚点：sgl-project/sglang main，2026-08-13 02:57 raw 抓取（行号以抓取日为准；
抓取件 md5/字节数见 §12）。五件套 = match / insert / evict / lock_ref / 时钟。

**① match：最长前缀匹配，可分裂**。nano `match_prefix`（走查 + 边内逐 token 比较 +
块边界分裂）对照 `match_prefix`:L352——docstring 明说 "may mutate internal structure
by splitting an existing node"（命中即可能分裂，结构细化、不复制数据）；
`_match_prefix_helper`:L648 沿路径刷新 `last_access_time`（LRU touch 内建于查找）。
nano 的 `_walk` 同款：沿首块字典下行、边内比较、沿路刷 `last_access`。差异：nano 的
分裂点恒在块边界（dict 键 = 首块 token，命中边至少共享一整块）——SGLang token 粒度
可在任意 token 处分裂，page_size>1 时由 `page_aligned`:L135 截断到页倍数（**同因**：
页内不可验证的尾部不能进缓存）。

**② insert：走查-分裂-挂新节点**。nano `insert` 对照 `_insert_helper`:L704 /
`cache_finished_req`:L434——**输出也入树**：`cache_finished_req` 把
`origin_input_ids + output_ids` page-aligned 插入（注释 "Radix Cache takes one ref in
memory pool"）。这是多轮对话下一轮命中、self-consistency 兄弟采样命中的来源。nano 的
`finish` 同款（prompt+gen 全量入树）。nano 的取舍：prompt 在**准入时**就入树
（`admit` 内 insert），SGLang 在运行中经 `maybe_cache_unfinished_req`（scheduler.py
import:L266 / 调用:L2917）把未完成请求入树供并发共享——目的相同（让并发请求命中
运行中的前缀），nano 用准入即入树近似，代价是 chunked prefill 语义缺失（§7）。

**③ evict：LRU 叶子，整叶释放**。nano `evict`：候选 = lock_ref==0 的叶子，按
`(last_access, node_id)` 建堆，弹出即整节点释放（块归还池、节点摘除），父变叶且未锁
则入堆。对照 `evict`:L562：evictable leaves 建堆（priority 来自 eviction_strategy）、
pop → free_segment → `_delete_leaf`、父节点变叶且 lock_ref==0 入堆——逐条同构。
论文依据即 §2 所引 "evicts the least recently used leaf first…"。nano 与 SGLang 现行
main 的差异：后者的堆优先级可插拔（priority-aware eviction），nano 固定 LRU。

**④ lock_ref：运行中保护**。nano `inc_lock_ref` 沿父链到 root 逐节点 +1，对照
`inc_lock_ref`:L592（走到 root：lock_ref+=1，evictable_size_→protected_size_）/
`dec_lock_ref`:L607。加锁时机：nano 在准入时（`admit` 末），SGLang 在批次加入时
（schedule_policy.py:L936-937 `_req_inc_lock_ref`，`tree_cache.inc_lock_ref(
req.last_node)`）。语义同款：**被运行中请求覆盖的整条前缀路径不可驱逐**——论文
"A node is evictable if its reference counter is zero" 的执行侧。

**⑤ 时钟与树身份**。nano 用逻辑钟（确定性），SGLang 用 `time.monotonic`
（TreeNode:L216 `last_access_time`；`__lt__` 即 last_access_time 比较，堆序所在）。
另有 nano 没有的两件：`RadixKey`:L59 的 `extra_key` 命名空间（LoRA/cache_salt 隔离，
同 token 不同上下文的键空间）与 `evicted` property:L245（`value is None`——驱逐后节点
可留骨架）。

**与 vLLM 哈希链的对照**（vllm-project/vllm main 同日抓取，行号以抓取日为准）：
同一段机制，vLLM V1 用另一套数据结构表达——`KVCacheBlock`:L119（ref_cnt/block_hash/
free-list 双向指针）+ `FreeKVCacheBlockQueue`:L185 + `hash_block_tokens`:L577
（parent_hash + 本块 token 进同一哈希，与 L2 的 `block_hash` 同构）+
`cache_full_blocks`:L225 / `get_new_blocks`:L647 / `touch`:L702 / `free_blocks`:L719
（L728-743：无哈希块 prepend 先复用、有哈希块 append 保 LRU 缓存）。**树与链的本质
差异不在「能不能缓存」，而在三件事**：驱逐的单位（节点/枝叶 vs 块）、驱逐的顺序
（叶优先 LRU vs 释放序 FIFO）、以及调度能否「看见」前缀结构（LPM 需要树，哈希链
给不出「已命中长度」的排序键——vLLM 的 LPM 等价物长在 scheduler 的哈希前缀匹配里，
不在 block_pool）。

---

## 6. touch 必须先于 allocate：一个从 L2 里挖出来的竞态（一等教材）

这一节记录一个关键发现：**L3 的压力场景首次暴露了 L2 引擎一个潜伏的引擎层
bug**——L2 自己的场景从未触发它，但 L2 docstring 自载的
跨级别契约（「生成逐 token 一致」）在压力场景下被它违反。仪表化探针独立复现并验证了修复。
**结论先行：free queue 不是普通队列——它里面的
块带着缓存条目，任何消费者必须先结算命中块的缓存主张，再消费空闲块。**

### 6.1 现象：rA2 在第二个 token 静默发散

对早期 L3 版本运行旧 workload（[4]：F1/F2 = 16 tok，budget 20）：
`[4]` 的 L2 对照组里 rA2（命中 SYS 32 tok）生成 `[436, 188, 4, 195]`，L1 参考
`[436, 346, 507, 21]`——**tok0 相同、tok1 起发散**，且无任何报错。一个交叉引用 L1/L2
的独立仪表化探针复现了相同事件序：

```
rA2  hit=32  events=[('get_new', [12, 0, 1, 2, 3]),
                     ('touch', [0, 1, 2, 3, 4, 5, 6, 7]),
                     ('get_new', [14])]   DUP_IN_TABLE=True
rA2  sem_ok=False  gen=[436, 188, 4, 195]  ref=[436, 346, 507, 21]
```

### 6.2 根因：allocate-before-touch，一表双别名

旧 L2 `paged_prefill` 的顺序是**先 `get_new_blocks`、后 `touch` 命中块**。rA2 进场时
SYS 的 8 个命中块（0-7）正躺在 FIFO free queue 里（上一个使用者释放后尚未被复用）：

1. `get_new_blocks(5)` 从队首弹出 `[12, 0, 1, 2, 3]`——**0-3 正是即将 touch 的命中块**，
   此刻 touch 还没执行，它们仍是「空闲块」；
2. `touch([0..7])` 随后执行：见 0-3 的 ref_cnt 已被 get_new 置 1，只 +1（双重记账），
   不再做「移出 free queue」动作；
3. block table 变成 `[0,1,2,3,4,5,6,7, 12,0,1,2,3]`——**块 0-3 一表双别名**：既是
   命中段（逻辑位置 0-15）又是「新」后缀块（逻辑位置 36-51）；
4. prefill forward 内 gather 先读位置 0-31（SYS KV 尚完好，故 tok0=436 正确），
   scatter 再写位置 36-51——**落进块 0-3 的块内 offset 0-3，覆写 SYS 位置 0-15 的 KV**；
5. tok1 的 decode gather 读到被污染的脊柱 KV → 静默发散。

触发条件 = **缓存命中 + 块压力**同时成立（命中块恰在 FIFO 头部 + 后缀需要分配）。
双对照坐实：每请求 fresh pool（无跨请求缓存）7/7 语义正确；budget 放宽到 60
（无复用压力）7/7 语义正确。L2 自身场景两个条件从不同叠加，所以重复运行均通过。

### 6.3 权威实现怎么处理同一个问题：vLLM 的两阶段分配

这不是 nano 独有的坑——**vLLM V1 踩过同因的竞态并留下了修复痕迹**（以下行号均为
2026-08-13 抓取/核验）：

- `kv_cache_coordinator.py`:L224-227（复现运行 抓取，37,705 B，md5 `c109880f…`）
  注释原文："Two-phase allocation (issue #33775): first touch every group's local
  cache-hit blocks, then allocate external blocks for every group. This ensures an
  earlier group's external `get_new_blocks` cannot evict a later group's not-yet-touched
  cache-hit blocks." ——**先 touch 所有命中块、再 get_new_blocks**，理由逐字写明：
  防止分配弹出尚未 touch 的命中块。issue #33775 经 GitHub API 同日核验在案：
  标题 "[KVConnector] Fix data race when we have both local and external cache hit"，
  state=closed，created 2026-02-04。
- touch 的执行点：`single_type_kv_cache_manager.py`:L267 `self.block_pool.touch(
  new_computed_blocks)`（注释 L265 "Touch the computed blocks to make sure they
  won't be evicted."），发生在 `add_local_computed_blocks`:L230 内。
- 准入的零副作用：`allocate_slots`（kv_cache_manager.py:L345）先做全量预算检查，
  不足直接 `return None`（L525-529），**检查在任何状态变更之前**；L534-537 注释明说
  命中块延迟提交是 "to avoid the case where the new blocks cannot be allocated"。
- 自由块计数的细账：vLLM 把「尚在 free queue 的命中块」计入容量检查——
  `single_type_kv_cache_manager.py`:L218-221 注释原文："If a computed block is an
  eviction candidate (in the free queue and ref_cnt == 0), it will be removed from
  the free queue when touched by the allocated request, so we must count it in the
  free-capacity check."（`num_evictable_blocks` 并入返回值，L228）。**不算这笔账，
  准入检查会把即将被 touch 收回的块当成可用空闲，放行一个注定失败的分配。**

### 6.4 修复：touch-before-allocate + 零副作用准入检查

L2 `paged_prefill` 的修复（复现运行 落盘）与 vLLM 同序：

```python
n_new = seq.n_blocks(len(toks)) - len(hit_ids)
# 准入检查（此刻零副作用）：仍留在 free queue 里的命中块（ref_cnt=0）
# 马上要被 touch 收回，不能算可用空闲（vLLM num_evictable_blocks 同款）
in_queue = sum(1 for i in hit_ids if pool.blocks[i].ref_cnt == 0)
if n_new > pool.num_free - in_queue:
    raise ValueError(...)                     # 不足 → raise，此前零副作用
# touch-before-allocate：命中块先移出 free queue，get_new_blocks 就绝不会弹出它们
if hit_ids:
    pool.touch(hit_ids)
new_ids = pool.get_new_blocks(n_new)          # 检查已保证不会失败
```

选择「先检查后变更」而非「先变更后回滚」：检查在任何状态变更之前，失败路径天然
零副作用，不需要恢复 free queue 位置的复杂回滚逻辑——这与 vLLM 的
`required_blocks > available_blocks → return None` 同款。修复后探针复跑：事件序变成
`touch → get_new`（命中块已不在队列，不可能被弹出），`DUP_IN_TABLE=False`，
7/7 请求 `sem_ok=True`。

### 6.5 修复后的不变性证明

修复修改了 `L2_real_paged_memory.py`，因此额外验证原有 L2 场景不变：

- **修复前锚点**：md5 `d9921fb049af1df08e76b942ddfed38a`/669 行。
- **自场景输出不变性**：L2 场景不触发竞态（§6.2 触发条件），故修复前后 L2 自场景
  输出应逐位不变——实测：修复前基线跑与修复后复跑（外部 CWD、`-B`）掩码输出
  （口径 `sed '/prefill 墙钟/d'`）**cmp BYTE-IDENTICAL**，md5 均 =
  `cced3a908c09aa543d42a6ab549f8d87`/63 行（五次独立运行均收敛）。
- **新文件锚**：`L2_real_paged_memory.py` 修复后 md5
  `24d37d15c001c5b9cffff9c46b69f47e`/686 行（改动仅 `paged_prefill` docstring +
  函数体 21 行，print 路径零改动）。
- **行号引用核验**：tutorial_L2.md / README.md 对代码行号的引用全部指向 vLLM 源文件
  （§8 锚点表），无 L2 本文件行号引用，零失效。

---

## 7. 权威实现取舍表（含 nano 侧未做项）

对照锚点：SGLang `python/sglang/srt/mem_cache/radix_cache.py` /
`managers/schedule_policy.py` / `managers/scheduler.py`，vLLM `v1/core/block_pool.py` /
`kv_cache_manager.py` / `kv_cache_coordinator.py` / `single_type_kv_cache_manager.py` /
`kv_cache_utils.py` / `config/cache.py`——均 main 分支 2026-08-13 抓取（md5/字节数见
§12，行号以抓取日为准）。

| 机制面 | nano L3 | SGLang 现行 main | vLLM V1 现行 main |
|---|---|---|---|
| 缓存组织 | radix tree（Node：key/blocks/children/lock_ref/last_access） | radix tree（TreeNode:L216，另有 hit_count / evicted property:L245） | 链式内容哈希（KVCacheBlock:L119 + hash_block_tokens:L577） |
| 粒度 | 块（BS=4，继承 L2） | token（论文 §3）；page_size>1 时 page_aligned:L135 截断到页倍数 | 块（DEFAULT_BLOCK_SIZE=16，config/cache.py:L48） |
| 匹配 | `_walk` 走查 + 块边界分裂 | match_prefix:L352（可分裂）+ _match_prefix_helper:L648 | scheduler 侧哈希前缀匹配（block_pool 不感知） |
| 驱逐 | 显式 LRU 叶子，整叶释放，堆 `(last_access, node_id)` | evict:L562，堆优先级可插拔（priority-aware） | 隐式：free queue FIFO 复用即驱逐（free_blocks:L719 无哈希 prepend/有哈希 append） |
| 运行中保护 | 节点 lock_ref，准入时沿父链加锁 | 同款（inc_lock_ref:L592；批次加入时 _req_inc_lock_ref:L936-937） | 块级 ref_cnt（touch:L702） |
| 分配顺序 | touch-before-allocate + 零副作用准入检查（§6.4） | 准入增量分配、不足即驱逐 | 两阶段分配（issue #33775，coordinator:L224-227）+ allocate_slots 先检查后提交（L345/L525-529）+ num_evictable_blocks 计数（L218-228） |
| 准入 | full-fit 预分配（prompt+max_new 全程，L2 同款门槛） | token 增量分配，运行中不足 → 驱逐/重试 | full_sequence_must_fit（allocate_slots 参数）+ watermark |
| 调度 | LPM demo（[5]，手写排序） | CacheAwarePolicy.LPM:L198 + calc_priority:L232；队列 >128 退回 FCFS（_determine_active_policy:L285-288） | 调度不依赖树结构（哈希匹配给 computed blocks） |
| 完成入树 | finish：prompt+gen 全量 insert + 解锁 | cache_finished_req:L434 同款 | cache_full_blocks:L225（块级登记） |

**nano 侧未做项（点名不展开，均为 SGLang/vLLM 现行 main 扩展）**：extra_key 命名空间
（RadixKey:L59，LoRA/cache_salt 隔离）；priority-aware eviction（堆优先级策略可插拔）；
HiCache host backup / kv events / EAGLE bigram key（SGLang）；watermark /
reserved_blocks / chunked prefill / 多 KV 组协调（vLLM）；metrics collector
（block_pool 各处 on_block_allocated/on_block_evicted 钩子）。教学以论文骨架
（tree/match/insert/evict/lock_ref）为主体——对齐结论（2026-08-13）：PagedAttention
与 RadixAttention 仍是 2026-08 推理引擎前缀缓存/KV 内存管理的两大权威范式，均为
A 层经典锚点（机制仍是现役实现的地基：vLLM V1 block_pool / SGLang radix_cache 现行
main 即其直接演化）。

---

## 8. 费曼自检：家谱、借出的书、与仓底先发的货

**讲给外行听**：想象一个图书馆把「内容相同的书」只买一本。

- **L2 的哈希链**像一排储物柜：每个柜子的钥匙由「上一个柜子的钥匙 + 本柜内容」
  配出来，同前缀的人拿同一串钥匙开同一排柜子。省，但管理员（FIFO）只会按
  「哪个柜子最先空出来」发货——他不懂哪些柜子属于同一串钥匙。
- **L3 的 radix tree**像家谱：共同的祖先（system prompt）只写一页，后代分支挂下去；
  要腾地方时**先剪最外围的叶子**，祖先留着——因为「姓这个姓的人」还可能来。
  `lock_ref` 是挂在节点上的「借出中」牌子：只要还有一个读者在用，这一支就不许清理。
- **LPM 调度**是前台的叫号策略：与其让姓张的、姓李的、姓张的、姓李的交替来（每次
  都得把上一家的档案收起来再搬出来），不如让同姓的一起来——家谱摊开一次，服务全家。

三条自测：

1. 为什么「先剪叶子」比「按入库顺序发货」保住的前缀多？（叶子是局部损失，
   祖先是全局枢纽；FIFO 恰恰最先发走最早入库的祖先——脊柱块最先被释放。）
2. lock_ref 为什么必须沿父链一路加到 root，只锁命中的末端节点行不行？
   （不行：驱逐是自底向上剪的，末端节点的祖先若可驱逐，剪到它时整条路径连根失效。）
3. [4] 里 L2 的 rB2 命中了 32 tok，为什么说这是「运气」不是「保证」？
   （命中的是 rA2 刚重算重登记的副本块；FIFO 的命中取决于队列位置，
   radix 的存活取决于树结构。）

---

## 9. 思考题

1. **[4] 的洪水尺寸**：F1/F2 从 16 tok 加到 32 tok 才让 FIFO 截断在实测中成立
   （16 tok 洪水卷不过队列——脊柱块前面总有足够的非脊柱空闲块挡着）。推演：
   给定预算 B、脊柱 S 块、每轮洪水 F 块，FIFO 卷过脊柱的最小 F 是多少？
   为什么这个阈值本身就说明「FIFO 的缓存存活是队列位置的函数」？
2. **整叶释放的代价**：radix 驱逐弹出整叶（可能释放超过所需——[4] 日志里需 9 释 12）。
   为什么 SGLang 不做「切半个节点精确凑数」？（对照 `_split_node`:L674 的使用场景：
   分裂服务于匹配，不服务于驱逐精度；整叶释放换来驱逐路径 O(1) 决策与结构简洁。）
3. **LPM 的退回阈值**：SGLang 在 waiting_queue > 128 时 LPM 退回 FCFS
   （schedule_policy.py:L285-288，注释 "Turn off the expensive prefix matching and
   sorting when the #queue is large"）。匹配成本随队列线性涨，命中收益为什么
   不跟着涨？给一个「队列很长时排序边际收益递减」的论证。
4. **块粒度的命中损失**：BS=4 时两个 prompt 在第 3 个 token 分叉，命中 0 tok；
   SGLang token 粒度可命中 3。构造一个 workload 让块粒度损失可见，并解释
   `page_aligned`:L135 为什么宁可截断也不缓存「页内未验证的尾部」。
5. **检查-变更 vs 变更-回滚**：§6.4 选了「先检查后变更」（失败零副作用）。若改成
   「先 touch 再分配、失败则回滚 touch」，回滚必须恢复 free queue 的**位置**
   （touch 从队列中间 remove，append 回去会改变 FIFO 序）。写一个最小例子说明
   位置漂移如何改变后续驱逐顺序——从而理解 vLLM 为什么把检查放在一切变更之前。

---

## 10. 反例与边界

**toy 尺度计时，诚实定位**。[3] 的 elapsed 行用 136-tok 长 prompt、3 遍取最快测得
冷跑 vs 全命中 ≈ 3.9×（参考运行 4.22/1.07 ms）——比 52-tok 整请求计时的 ~1.1× 有意义得多
（后者被 Python 开销主导，撑不起「计算整个跳过」的直觉），但仍远小于 token 比
（136/1）。原因与 L2 §5「为什么只有 2.6×」讨论同款：命中路径仍要付树操作 + 至少一格重算 +
decode 的钱，且 CPU 小模型下固定开销占比高。**所有墙钟不可外推到 GPU kernel 尺度**，
真实引擎的命中收益见 [TODO: verify on real system]（真实 GPU/多机环境）。

**竞态的边界教训（本节一等教材）**。§6 的竞态揭示了 block_pool 类设计的一条边界：
**带缓存条目的 free queue 不可简化为一个普通 free list**。空闲块同时是缓存条目
（释放 ≠ 清空），任何从队列取块的消费者（get_new_blocks）与任何主张缓存所有权的人
（touch）之间存在顺序契约——touch 必须先于 allocate，否则同一物理块会同时承担
「命中段」与「新块」两个身份，scatter 覆写命中 KV，且**全程无报错**。vLLM 在
多 KV 组场景以 issue #33775 的形式踩过同因竞态（先行的组 get_new 弹走了后面组尚未
touch 的命中块），修复即两阶段分配。nano 单组场景的 touch-before-allocate 是同一条
原则的最小形式。这也回答了「为什么 vLLM 的 allocate_slots 那么长」：准入检查、
watermark、命中块延迟提交、evictable 计数——全是围绕这条顺序契约的防御工事。

**顺序执行模拟并发**。[2] 的「并发」是准入 4 个请求后轮转 decode——lock_ref 簿记、
共享块只读、轨迹正确性都是真的，但没有真实 batched attention kernel、没有真并行
抢占。真并发行为 [TODO: verify on real system]。

**准入预分配 vs SGLang 增量**。nano 准入时按 prompt+max_new 全程预分配（L2
full_sequence_must_fit 同款门槛），SGLang 按 token 增量分配、不足即驱逐。预分配让
「不足 → 拒绝」发生在准入（无副作用），代价是长请求可能被过早拒绝——真实引擎用
chunked prefill + 增量分配绕开，nano 留门槛语义。

**LPM 无退回、无 partial 条目、单 KV 组**。见 §7 未做项清单；均为现行 main 扩展，
不属论文骨架。

---

## 11. 阶梯预告

nano-vllm-sglang 的阶梯到此走完：L0 算术账（KV 显存/吞吐曲线/分页思想）→ L1 真实
小模型（开销分解/gemm 悬崖/调度墙钟）→ L2 真实分页内存（block table/refcount/
哈希链前缀缓存/CoW/抢占）→ L3 radix tree 前缀缓存（共享/叶优先驱逐/LPM）+ 一条
从压力场景里挖出来的竞态教材。下一步不在本模块内：真实引擎（vLLM/SGLang on GPU）
的吞吐/KV 显存测量需在真实 GPU/多机环境验证 `[TODO: verify on real system]`；03 轨的
数据方法论深挖（sota-deepdive）已覆盖「数据侧怎么喂引擎」；01 轨 RL rollout 与
04 轨 agent 的推理后端需求，是这套机制的下游买主。

---

## 12. 溯源与口径声明

**论文（arXiv，export.arxiv.org API 2026-08-13 核验）**：

| arXiv ID | 标题 | 录值 |
|---|---|---|
| 2309.06180（v1，2023-09-12） | Efficient Memory Management for Large Language Model Serving with PagedAttention（Kwon, Li, Zhuang, Sheng, Zheng 等 9 人） | 抓取件 arxiv_both.xml 5,507 B；两次独立 API 调用所得尺寸一致 |
| 2312.07104（v2，2024-06-06） | SGLang: Efficient Execution of Structured Language Model Programs（Zheng, Yin, Xie, Sun, Huang 等 12 人） | 同上；ar5iv 全文 208,367 B（§3 RadixAttention 逐字引文 grep 在位） |

**源码抓取件（全部 2026-08-13 raw.githubusercontent.com main 分支；行号以抓取日为准，
md5 为现场复算值；早期记录的 md5 列有 6/8 归属错位，内容、行锚和字节数本身正确）**：

| 文件 | 字节 | md5 | 抓取时点 |
|---|---|---|---|
| sglang mem_cache/radix_cache.py | 30,700 | `7d4f521ac6f747be2bebee7540c7f10b` | 02:57（两次独立抓取 BYTE-IDENTICAL） |
| sglang managers/schedule_policy.py | 62,395 | `328c19b81bed35c0451fc4f839947ce9` | 02:58 |
| sglang managers/scheduler.py | 218,585 | `d8e3ac5706d27641b88a60797eae95a6` | 02:57 |
| vllm v1/core/block_pool.py | 33,139 | `afabd86019b814ac37a3e71ee0202b94` | 02:57（与 08-06 快照零漂移） |
| vllm v1/core/kv_cache_manager.py | 38,852 | `b0f2048c84568ef97c6dbdb8d79faee9` | 02:57 |
| vllm v1/core/kv_cache_utils.py | 93,070 | `0a76c79ac8bf826bfa2e7b8820cc9bfd` | 02:57 |
| vllm v1/core/sched/scheduler.py | 138,310 | `659cd9ccdda610c372654a2ab02cebb1` | 02:57 |
| vllm config/cache.py | 13,934 | `3f59d08fa119b5aeb3436ae293d6c7e2` | 02:57 |
| vllm v1/core/kv_cache_coordinator.py | 37,705 | `c109880f0008d9500b7d6793ab89fc9b` | 同日补抓（§6 两阶段分配锚点） |
| vllm v1/core/single_type_kv_cache_manager.py | 85,253 | `68861f9d61c503f4db6003d3540119d3` | 同日补抓（touch:L267 / num_evictable_blocks:L218-228） |

**GitHub issue**：#33775（vllm-project/vllm）经 api.github.com 2026-08-13 核验：
标题 "[KVConnector] Fix data race when we have both local and external cache hit"，
state=closed，created 2026-02-04T07:44:35Z。

**行锚清单（抓取日口径）**：radix_cache.py RadixKey:L59 / page_aligned:L135 /
TreeNode:L216 / evicted:L245 / RadixCache:L279 / match_prefix:L352 / insert:L412 /
cache_finished_req:L434 / evict:L562 / inc_lock_ref:L592 / dec_lock_ref:L607 /
_match_prefix_helper:L648 / _split_node:L674 / _insert_helper:L704；
schedule_policy.py LPM:L198 / calc_priority:L232 / _sort_by_longest_prefix:L260-261 /
_determine_active_policy:L285-288 / _req_inc_lock_ref:L936-937；sglang scheduler.py
maybe_cache_unfinished_req import:L266 / 调用:L2917；vllm block_pool.py
cache_full_blocks:L225 / get_new_blocks:L647 / _maybe_evict_cached_block:L679 /
touch:L702 / free_blocks:L719（L728-743 prepend/append）；kv_cache_utils.py
KVCacheBlock:L119 / FreeKVCacheBlockQueue:L185 / hash_block_tokens:L577；
kv_cache_manager.py allocate_slots:L345 / 检查后 return None:L525-529 /
命中块延迟提交注释:L534-537；kv_cache_coordinator.py 两阶段注释:L224-227 /
add_local_computed_blocks 调用:L229；single_type_kv_cache_manager.py
get_num_blocks_to_allocate:L142 / num_evictable_blocks:L218-228 /
add_local_computed_blocks:L230 / block_pool.touch:L267；cache_config.py
DEFAULT_BLOCK_SIZE=16:L48。vllm 行号 vs tutorial_L2 §14 的 08-06 快照有 ±1–17 微漂移
（main 提交频繁），本教程全部按 2026-08-13 抓取重录。

**可复现锚**：L3 掩码输出锚 `90c2fcffeac2d328e81116f53543c784`/67 行（口径
`sed '/^[[:space:]]*elapsed/d'`，raw 68 行删 1 行；双独立 CWD 两遍 BYTE-IDENTICAL，
本节 paste 块即该掩码输出）；L3 代码 md5 `14c9b7450f1f906375fe90e309bc7d96`/672 行；
L2 修复前后自场景掩码输出不变，锚 `cced3a908c09aa543d42a6ab549f8d87`/63 行；
L2 修复后代码 md5 `24d37d15c001c5b9cffff9c46b69f47e`/686 行。竞态探针逐位记录
事件序、`DUP_IN_TABLE` 与发散 token；[4] 参数经网格搜索（flood ∈ {24,32,40,48} × budget ∈
{18,20,24}，选 flood=32/budget=20：对比成立且预算继承原值）。跨模块引用：
L2 前缀缓存与 2.6× 计时先例（tutorial_L2 §1/§5/§7，锚点在其 §14）；L1 连续池参考
（tutorial_L1）。

**环境**：Apple M5 Pro / Python 3.13.13 / torch 2.13.0
（`python3`）/ macOS arm64，CPU fp32，greedy + seed=42
+ 逻辑时钟——计数类输出跨运行逐字节一致（两遍独立 CWD 机器证明在案）。
