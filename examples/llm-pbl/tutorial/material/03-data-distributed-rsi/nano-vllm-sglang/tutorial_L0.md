# nano-vllm-sglang · L0 教程：高吞吐推理的最小机制——KV cache、continuous batching 与分页

> **本节目标（L0）**：用 ~170 行纯标准库抓住高吞吐 LLM 推理的四个最小机制：
> KV cache（用显存换计算）、batching（摊薄权重读取）、continuous batching
> （iteration 级调度）、PagedAttention 思想（按页分配消除碎片）。
> **前置**：知道 transformer 自回归生成是「一次吐一个 token」即可；读过 nano-verl L2
> 的 batch inference 段落更容易理解动机。
> **本节 K+1**：从「会调 generate」到「说得出推理引擎为什么长这样」。

---

## 1. 问题：decode 的脾气

一次 LLM 推理分两个阶段，脾气完全不同：

- **prefill**：prompt 的所有 token 一次性并行通过模型，是大矩阵乘，**compute-bound**；
- **decode**：自回归一次只生成一个 token。每生成一个 token，都要把**全部权重**从显存
  读进计算单元，却只做「一条序列 × 一个 token」的算术——算术强度极低，
  **memory-bandwidth-bound**。

decode 的脾气推出两个核心问题：

1. 第 t 个 token 要对前 t-1 个 token 做 attention。如果每步都重算前缀的 K/V，
   生成 T 个 token 的 K/V 投影总量是 1+2+…+T = O(T²)——**怎么不重算？**
2. 每步都要读一遍全部权重，只服务一条序列太亏——**怎么让一次读取服务更多序列？**

问题 1 的答案是 KV cache（§3），问题 2 的答案是 batching（§4）；而 batching 立刻
引出下一个问题——**怎么把 batch 塞满又不浪费**——这就是 continuous batching（§5）
与 PagedAttention（§6）的位置。四个机制环环相扣，本节把它们逐个做出来。

---

## 2. 先跑起来

文件：`L0_kv_cache_batching.py`，纯标准库，CPU 即跑。

```bash
$ python3 L0_kv_cache_batching.py
```

真实输出（纯计算，跨运行确定）：

```text
================================================================
nano-vllm-sglang L0 — 高吞吐推理的最小机制
================================================================

[1] KV cache（生成长度 T=512，单层 K/V 投影计数）
    无 cache（每步重算前缀）: 131,328 次
    有 cache（每步只算新 token）: 512 次   → 计算量 ÷256.5（精确比值 (T+1)/2）
    代价：每 token 要存 0.5 MiB KV（32 层 × 32 KV head × 128 dim × 2(K,V) × 2B）
    → 一条 4096 token 的序列 KV 占 2.0 GiB；KV 显存成为 serving 的第一瓶颈

[2] decode 迭代代价模型：iter_time = 1.0（读权重）+ B × 0.02（KV）
       B |     迭代耗时 | 吞吐 tokens/单位时间
       1 |     1.02 |          1.0
       2 |     1.04 |          1.9
       4 |     1.08 |          3.7
       8 |     1.16 |          6.9
      16 |     1.32 |         12.1
      32 |     1.64 |         19.5
      64 |     2.28 |         28.1
    上界：B→∞ 时吞吐 → 1/0.02 = 50 tokens/单位时间（被 KV 代价封顶）
    → 想高吞吐就得把 batch 塞满；问题变成「怎么塞满又不浪费」

[3] 8 个请求 lengths=[6, 2, 9, 3, 14, 5, 1, 8]，batch_size=4
    static    : makespan = 23 迭代 | 平均延迟 = 10.50 | 最长等待 = 23 | bubble = 44 槽位·迭代
    continuous: makespan = 16 迭代 | 平均延迟 = 8.25 | 最长等待 = 16 | bubble = 0（有等待就不空槽）
    逐请求延迟 static → continuous: [(6, 6), (2, 2), (9, 9), (3, 3), (23, 16), (14, 8), (10, 7), (17, 15)]
    边界检查：长度全为 5 时 static = continuous = 10 迭代（收益来自长度参差，长度对齐时无增益）

[4] 同样 8 条序列（ΣL=48），L_MAX=16，page=4 token
    连续预分配: 128 槽位（浪费 80，占 62%）——不管实际多长，一律按最长预留
    按页分配  : 64 槽位（浪费 16，≤ 每请求 3）——用多少发多少
    显存预算 = 64 槽位时：paged 正好容纳 8 条并发；contiguous 只能放 4 条 → batch 减半，吞吐直接减半（回看 [2]）
    页大小扫描 (page → 槽位): 1:48 | 2:52 | 4:64 | 8:80 | 16:128
    → page 太大 = 内部碎片回潮；page = 1 = 元数据与碎片化访存开销（vLLM 默认 block_size=16 token [TODO: verify source]）

================================================================
✅ self-check passed: KV cache 计算账 / 吞吐随 batch 单调 /
   continuous 全面优于 static（且长度对齐时持平）/ paging 省显存且边界正确
================================================================

takeaway: vLLM 的吞吐 = continuous batching（调度不空转）× PagedAttention
          （显存不碎片）→ batch 塞满 → 摊薄权重读取。SGLang 在此之上加前缀复用
          （RadixAttention）与结构化生成调度 [TODO: verify source]。
          L1 用真实 vLLM 跑小模型，把这份代价模型对到真实 tokens/s。
```

> **toy 口径声明**：本节是**代价模型 + 迭代模拟器**，没有真实张量、没有 GPU。
> 「时间」的单位是「一次完整权重读取耗时」（`W_READ=1.0`），吞吐等全部数字是
> 模型的算术输出，用来演示**方向与相对量级**，不是真机 benchmark——真机数字
> 留给 L1/L2 实测（`[TODO: verify on real system]`）。这与 nano-megatron L0
> 用账本代替实测的口径相同。

**L0 基线指标（toy metric）**：KV cache 把 K/V 投影计算从 131,328 次降到 512 次
（÷256.5，即 (T+1)/2 倍）；batch 从 1→64 时吞吐 1.0→28.1 tokens/单位时间（上界 50）；
continuous batching 把 makespan 从 23 降到 16 迭代、平均延迟 10.50→8.25、
bubble 44→0；分页把 KV 槽位从 128 降到 64（同样预算下并发翻倍）。

---

## 3. 机制一：KV cache——用显存换计算

attention 需要每个历史 token 的 K/V 向量。两种做法：

```python
def kv_projection_ops(T, use_cache):
    """生成 T 个 token 的 K/V 投影计算量（单层，toy 计数）。"""
    return T if use_cache else T * (T + 1) // 2
```

- **无 cache**：第 t 步把 t 个前缀 token 的 K/V 全部重算一遍，总量 Σt = T(T+1)/2；
- **有 cache**：每步只算**新 token** 的 K/V，写进缓存；历史 K/V 直接读。总量 T。

T=512 时相差 (T+1)/2 = 256.5 倍，且 T 越大差距越大——这就是「没有 KV cache 就没有现代
LLM serving」的原因。但代价立刻出现：cache 要显存，而且随长度线性涨：

```python
def kv_per_token_bytes():
    # 2(K,V) × 32 层 × 32 KV head × 128 dim × 2B(fp16) = 524,288 B = 0.5 MiB
    return 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * BYTES
```

按 Llama-2-7B 的公开配置（32 层 / 32 KV head / head_dim=128 / fp16）算：
**每 token 0.5 MiB，一条 4096 token 的序列占 2 GiB**。KV cache 由此取代权重，
成为 serving 的第一显存瓶颈——这个结论会一直贯穿到 §6 和 L2。

---

## 4. 机制二：batching——把权重读取摊薄

decode 每步的固定大头是「读一遍全部权重」，这笔开销**与 batch 里有几条序列无关**
（memory-bandwidth-bound 的直接推论）。toy 代价模型：

```python
def iter_time(batch):
    return W_READ + batch * KV_STEP     # 1.0 + B × 0.02
```

吞吐 = B / iter_time(B)：B=1 时 1.0，B=64 时 28.1，随 B 单调上升但**次线性**，
上界是 1/KV_STEP = 50（B→∞ 时权重读取被完全摊薄，只剩 KV 代价）。
真实 GPU 上封顶因素更复杂（显存带宽、attention 计算、kernel 效率），
但「**batch 越大吞吐越高，直到某个上限**」的方向一致。

结论只有一句话：**想高吞吐，就得把 batch 塞满**。于是问题变成：
请求长短不一、来来去去，怎么让 batch 一直满着、又不让谁空等？

---

## 5. 机制三：continuous batching——iteration 级调度

**static batching** 的做法：凑齐 B 个请求打包，整批跑到**最长**的序列结束。
三种浪费同时发生：短序列完成后其槽位空转（bubble）；空出的槽位不能进人；
下一批必须等整批结束。模拟器逐行对应：

```python
def sim_static(lengths, batch_size):
    for i in range(0, len(lengths), batch_size):
        grp = lengths[i:i + batch_size]
        t_end = t + max(grp)             # 整批耗时由最长序列决定
        for L in grp:
            bubbles += max(grp) - L      # 提前完成 = 空等
        t = t_end
```

**continuous batching**（Orca 论文称之为 iteration-level scheduling，
OSDI 2022）把调度粒度从「请求级」改成「迭代级」：**每个 decode 迭代**结束时，
谁 EOS 谁让位，等待队列里的请求同一迭代立刻进场：

```python
def sim_continuous(lengths, batch_size):
    while pending or active:
        while pending and len(active) < batch_size:   # 有空位就放人进来
            active[pending.pop(0)] = 0
        t += 1                                        # 一个迭代：全体前进一步
        for i in list(active):
            active[i] += 1
            if active[i] == lengths[i]:
                del active[i]                         # EOS 即让位
```

拿 `lengths=[6, 2, 9, 3, 14, 5, 1, 8]`、B=4 手工对一遍关键时刻：
t=2 时长度 2 的序列完成，**长度 14 的请求当步进场**（static 里它要等到 t=9 整批结束）；
t=3 长度 3 的让位，长度 5 的进场……最终 makespan 23→16 迭代、平均延迟 10.50→8.25、
bubble 44→0，且**后进场请求的延迟改善最大**（长度 14 的等待从 23 降到 16）。

**边界检查**（反幻觉纪律：机制的适用边界要能跑出来）：长度全为 5 时，
static = continuous = 10 迭代——**长度对齐时 continuous batching 零增益**，
收益完全来自长度参差。真实负载的生成长度几乎必然参差，所以这个机制才成为标配。

---

## 6. 机制四：PagedAttention 思想——把碎片变回 batch

continuous batching 保证了「有空位就进人」，但还有个前提：**显存里要有空位**。
KV cache 是随生成长度动态增长的，传统做法是给每条请求按最大长度 `L_MAX`
**连续预留**——实际长度往往远小于 L_MAX，预留出来的全是碎片（内部碎片）。

PagedAttention（vLLM，arXiv:2309.06180）的回答就是操作系统分页那一套：

- KV cache 切成固定大小的**物理块**（block/page，toy 里 1 block = 4 token）；
- 每条序列维护一张 **block table**（逻辑块 → 物理块），用多少块发多少块；
- 序列结束，物理块立刻归还池子，给下一条序列用。

```python
cont  = sum(L_MAX for _ in lengths)                 # 8 × 16 = 128 槽位
paged = sum(math.ceil(l / PAGE) * PAGE for l in lengths)   # = 64 槽位
```

同样 8 条序列（ΣL=48）：连续预分配 128 槽位（浪费 62%），按页分配 64 槽位。
把显存预算钉在 64：paged 正好 8 条并发，contiguous 只放得下 4 条——
**省下的显存直接换成 batch 大小，batch 换成吞吐（回看 §4 的曲线）**。
这就是 vLLM 论文的核心逻辑链：内存效率 → 更大并发 → 更高吞吐。

页大小本身是个取舍（实验 [4] 的扫描）：page=1 零浪费，但 block table 条目最多、
访存最碎；page=L_MAX 就退化成连续预分配，收益清零。vLLM 的默认
block_size=16 token `[TODO: verify source]`，是碎片与元数据的折中。

---

## 7. 与真实 vLLM / SGLang 的对应（概念层）

| nano 实现 | 真实系统 | 说明 |
|-----------|---------|------|
| `sim_continuous` 的迭代循环 | vLLM `LLMEngine` 的调度循环 / Orca 的 iteration-level scheduling | 每迭代重组 batch；真机有抢占、优先级等 `[TODO: verify source]` |
| 槽位 + 连续预分配 vs 分页 | vLLM block manager + PagedAttention kernel | 逻辑/物理块映射、按需分配 `[TODO: verify source]` |
| KV 字节账本 | 引擎的 KV cache 容量估算与 `gpu_memory_utilization` 划分 | 真机还要扣权重与 activation `[TODO: verify source]` |
| 未覆盖 | SGLang 的 RadixAttention（多请求共享前缀）、结构化生成调度 | L3 主题 `[TODO: verify source]` |

L0 只到概念层；源码级对照（block manager 数据结构、调度器状态机）留 L2/L3。

---

## 8. 费曼：讲给外行听

**类比：一家餐厅。**

- **KV cache = 备菜备忘单**：做菜是「接着上一步做」，不是每步从杀鸡开始重做——
  每一步的成果记在备忘单上（cache），代价是备忘单占操作台（显存），订单越长单子越厚；
- **batching = 一锅出**：烧热一口锅的成本是固定的，锅里煮 1 份还是 8 份差不多——
  权重读取就是那口锅，batch 越大，每份菜摊到的烧锅成本越低；
- **continuous batching = 翻台**：客人随到随坐、吃完就走，新客人立刻入座；
  而不是「凑满 8 人开一桌、全桌最慢的人吃完才一起清桌」；
- **PagedAttention = 编号储物柜**：行李（KV）存进统一规格的柜子，按需用、用完还，
  而不是给每位客人圈一整排货架——圈整排时，半排空着也不能给别人用。

一句话版本：**推理引擎的吞吐 = 让每次权重读取服务尽可能多的序列，
而 KV cache 的显存决定了能同时服务多少条——四个机制全在攻这两点。**

---

## 9. 思考题

1. 实验 [2] 里吞吐随 B 次线性增长、被 KV 代价封顶。真实 GPU 上还有哪些因素会提前
   封顶？（提示：显存带宽读不动更大的 KV；attention 计算量随 batch×长度上升；
   本 toy 只建模了前两项。）
2. continuous batching 在长度对齐时零增益（边界检查已证），那它在无增益时仍要付出
   什么成本？（提示：batch 形状每个迭代都在变，kernel 与显存布局更难静态优化；
   调度本身有开销。static batching 的「形状固定」反而是它的优点。）
3. 100 路多轮对话共享同一个 system prompt，它们的 KV cache 里有一大段完全相同。
   分页解决了「每条序列自己的碎片」，那「跨序列的重复」谁来解？
   （提示：前缀共享/复用——SGLang 的 RadixAttention、vLLM 的 prefix caching，
   这正是 L3 的主题。）

---

## 10. 反例：三个「想当然」

- **「KV cache 是免费的」**：它把 O(T²) 计算变成了 O(T) 显存。按 §3 的账，
  4096 token 一条序列 2 GiB，40 路并发就是 80 GiB——长上下文 + 高并发下
  KV 显存先于权重爆掉。解法方向都是「省 KV」：更少的 KV head（GQA/MQA）、
  KV 量化、前缀复用（思考题 3）。
- **「continuous batching 永远赚」**：长度对齐时零增益（§5 边界检查实测），
  还要付动态形状的代价（思考题 2）。它赚的前提是**负载长度参差且持续有新请求**。
- **「page 越小越好」**：page=1 时浪费确实为 0，但 block table 条目数等于 token 数，
  元数据与碎片化访存的开销反过来吃性能（§6 扫描：1:48 → 16:128 之间的取舍）。

---

## 11. 下一步 L1

L1 把这个 toy 对上真实引擎：

1. `pip install vllm`（或 sglang），在带 GPU 的机器上 serve 一个小模型
   （Qwen2.5-0.5B 量级），测 tokens/s 随并发数的曲线，对照本节 [2] 的代价模型；
2. 观察真实引擎里「batch 塞不满」的两种情形（请求不足 / KV 显存不足），
   判断各属于本节哪个机制的管辖范围；
3. 本机（Mac，无 NVIDIA GPU）跑不了 CUDA 版 vLLM，真机部分走 Machine B
   攒批验证通道 `[TODO: verify on real system]`。

---

## 12. 溯源

- 运行输出来自本机真实执行：`python3
  L0_kv_cache_batching.py`，纯计算确定性，连跑两遍逐字一致。
- vLLM / PagedAttention 论文：arXiv:2309.06180（Kwon et al.，*Efficient Memory
  Management for Large Language Model Serving with PagedAttention*），
  2026-08-04 于 arxiv.org 核验标题。
- SGLang 论文：arXiv:2312.07104（Zheng et al.，*SGLang: Efficient Execution of
  Structured Language Model Programs*），2026-08-04 于 arxiv.org 核验标题。
- continuous batching 的调度思想（iteration-level scheduling）出自 Orca
  （Yu et al., OSDI 2022，<https://www.usenix.org/conference/osdi22/presentation/yu>，
  2026-08-04 核验）。
- KV 账本基于 Llama-2-7B 的公开配置（32 层 / 32 KV head / head_dim=128 / fp16），
  字节数为现场算术（2×32×32×128×2 = 524,288 B/token），非引用数字。
- 「时间 / 吞吐」全部是 toy 代价模型输出（显式 toy 口径，见 §2），非真机 benchmark；
  真机吞吐数字一律留 L1/L2 实测。
- vLLM 默认 block_size=16、block manager / 调度器源码级对应、RadixAttention 细节
  均标 `[TODO: verify source]`，L2/L3 补齐；仓库：
  <https://github.com/vllm-project/vllm>、<https://github.com/sgl-project/sglang>
  （2026-08-04 核验可达）。
