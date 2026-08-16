# nano-vllm-sglang · L1 教程：把 L0 的代价模型换上真实模型——语义不变，代价变成实数

> **本节目标（L1）**：L0 用代价模型 + 迭代模拟器推出四个机制；L1 把其中三个
> （KV cache / batching / static vs continuous 调度）搬到一个**真实的小 GPT** 上，
> 用真实 forward、真实 KV 张量、真实墙钟重新付一遍账，并检验 L0 代价模型的预测力。
> **前置**：读过本模块 L0（四个机制的 toy 版）；知道 transformer 自回归解码即可。
> **本节 K+1**：从「代价模型里算账」到「真实模型上测量 + 用模型做预测」。
> PagedAttention 的**真实内存管理**（block table / 物理块池 / 前缀共享）留 L2——
> L0 讲了它的思想，L1 不重复。

---

## 1. 定位与三个声明

L0 的「时间」是 toy 单位（一次权重读取 = 1.0），L1 的时间是秒表。这一换，三件事
必须先声明清楚（都是反幻觉纪律）：

1. **模型是随机初始化的小 GPT**（3.1M 参数，GQA 8Q/2KV，固定 seed）。真实权重 +
   真实引擎（vLLM/SGLang on GPU）留 L2 / 真实 GPU/多机环境
   `[TODO: verify on real system]`。但机制本身是真的：真实 forward、真实 KV 张量、
   真实 greedy 解码、真实墙钟——与 L0 的纯算术模拟是质的不同。
2. **本机是 Apple Silicon CPU**。CPU decode 是 **compute-bound**，GPU decode 是
   **memory-bandwidth-bound**——曲线的「形状」可以对照，物理成因不同。凡涉及
   GPU 的数字一律 `[TODO: verify on real system]`，本节不冒充。
3. **eager PyTorch 的每步开销不可忽略**。3.1M 参数的模型每 token 计算量极小，
   墙钟里一大块是 Python/算子分发开销——这本身成为实验 [1] 的分析对象，
   而不是被藏起来的脏数据。

本节的骨架与 nano-ray L1 同构：**同一个计划，换一个基底**。那里的计划是
「分区→局部 OP→全局 OP 收敛」，基底从 multiprocessing 换成真 Ray；这里的计划是
L0 的调度语义，基底从代价模型换成真实模型。验收标准也一样——**语义逐位一致，
代价变成实数**。

---

## 2. 先跑起来

文件：`L1_real_kv_batching.py`，依赖仅 `torch`（本机实测 torch 2.13.0，
Python 3.13，CPU fp32，threads=4）。

```bash
$ python3 L1_real_kv_batching.py
```

真实输出（run1；计时行随负载浮动，计数类输出三遍逐字节一致，见 §11 口径）：

```text
====================================================================
nano-vllm-sglang L1 — 真实小模型实测 L0 的三个机制
====================================================================
torch 2.13.0 | CPU fp32 | threads=4 | seed=42
声明: 随机初始化小 GPT（真实 forward / 真实 KV 张量 / 真实墙钟）；
      真实权重 + 真实引擎 (vLLM/SGLang on GPU) 见 L2 / [TODO: verify on real system]

[0] 模型: 3,148,032 params | 4 层 | GQA 8Q/2KV | head_dim=32
    KV 池: 16 slots × 512 tokens 预分配 = 16.0 MiB（启动即划走，对齐真实引擎做法）
    每 token KV = 2(K,V)×4层×2KV头×32dim×4B = 2048 B；池字节数 == 公式 ✅
    同式外推 Qwen2.5-0.5B(bf16): 12288 B/token = 12.0 KiB → 4096 token 占 48 MiB
    对照 L0 的 Llama-2-7B: 0.5 MiB/token → GQA 把每 token KV 缩小 42.7×（2 KV 头 vs 32 + 更小 head_dim）

[1] KV cache（prompt=1 token，生成 T=256，greedy）
    无 cache（每步重算整段前缀） vs 预分配 cache（每步只算 1 个新 token）
    生成结果逐 token 一致: True ✅（256 个 token 全同）
    墙钟: 无 cache     932 ms / 有 cache    181 ms → 实测加速 5.2×
    算术上界 (T+1)/2 = 128.5×（token-step 数 Σt = T(T+1)/2 vs T）
    差距解释: 无 cache 单步 ≈ 固定开销 0.58 ms + 每 token 计算 0.024 ms（256 步最小二乘拟合）；cache 单步下界 0.51 ms
    外推 本 toy（实测区间）: 重建加速 =    6.8×（实测 5.2×；重建未含 cache 侧 O(t) gather，故偏高）
    外推 每 token 计算 ×100（≈真实大模型区间）: 重建加速 =  105.9× → 逼近算术上界
      结论: 算术上界 (T+1)/2 在「计算主导」区间成立；toy 模型每 token 计算太小，
      eager 开销两条路径都付，加速被压扁——这正是真实引擎要压开销的原因
    单步对照（无 cache 单步随前缀长度线性涨；cache 单步近恒定，
    微涨来自逐行 gather 拷贝 O(t)——真实引擎把这步融进 attention kernel）:
      t=  8: 无 cache   1.69 ms / cache  0.54 ms →   3.1×（计算比 ≈ 8×）
      t= 32: 无 cache   1.16 ms / cache  0.79 ms →   1.5×（计算比 ≈ 32×）
      t= 64: 无 cache   1.64 ms / cache  0.69 ms →   2.4×（计算比 ≈ 64×）
      t=128: 无 cache   3.01 ms / cache  0.81 ms →   3.7×（计算比 ≈ 128×）
      t=256: 无 cache   7.02 ms / cache  0.77 ms →   9.1×（计算比 ≈ 256×）
    实现方式也是成本: 2048 步追加式 torch.cat 32.6 ms vs 预分配写入 4.7 ms（7×，cat 每步搬运全量） → 引擎一律预分配 KV 池

[2] batching 曲线（每档 B 条序列 prefill 后齐步 decode 48 步，每档 3 遍取最快）
    B= 1: 每迭代   0.56 ms | 吞吐    1772 tokens/s
    B= 2: 每迭代   1.57 ms | 吞吐    1278 tokens/s
    B= 4: 每迭代   1.89 ms | 吞吐    2122 tokens/s
    B= 8: 每迭代   2.34 ms | 吞吐    3424 tokens/s
    B=16: 每迭代   3.13 ms | 吞吐    5116 tokens/s
    batched(B=4) vs solo 逐序列一致: True ✅（batch 不改变每条序列的数学）
    拟合 t(B) = 1.01 + 0.14·B ms（R²=0.840）
    固定项占比 @B=16: 31%（L0 模型 W_READ=1.0 → 76%）
    → CPU 上固定项很小: decode 是 compute-bound，batch 摊薄的是算力；
      L0 的大固定项是 GPU「每步读一遍权重」的 HBM 物理（GPU 待真机验证）

[3] 真实调度（8 个请求 lengths=[6, 2, 9, 3, 14, 5, 1, 8]，B=4，prompt=8 token，各 3 遍取最快墙钟）
    static    : 迭代 23 | 空转 token-step 44 | 墙钟     47 ms | 完成迭代 [6, 2, 9, 3, 23, 14, 10, 17]
    continuous: 迭代 16 | 空转 token-step 0 | 墙钟     35 ms | 完成迭代 [6, 2, 9, 3, 16, 8, 7, 15]
    L0 账本对照: static 23 迭代 / continuous 16 迭代 / bubble 44 —— 真实调度器逐位复现 L0 模拟器 ✅
    continuous 每迭代 decode 活跃数: [0, 4, 3, 3, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1]（0 = 纯接纳迭代）
    两种调度生成 token 逐请求一致: True ✅（调度改变「何时算」，不改变「算什么」）
    3 遍复跑生成内容逐位不变: True ✅（greedy 解码确定性）
    边界检查: 长度全为 5 时 static = continuous = 10 迭代（收益来自长度参差，与 L0 同款边界）

[4] 用 [2] 拟合的 t(B) 预测 [3] 墙钟（decode 用拟合式，prefill 用探针最快 1.3 ms）
    static    : 预测     44 ms / 实测     47 ms（偏差 -6%）
    continuous: 预测     32 ms / 实测     35 ms（偏差 -9%）
    结论: 仿射形式可拟合（R²=0.840）且能预测真实调度墙钟；
    残差来自 Python 调度循环与 batch 组装——真实引擎用 C++ 调度器压掉这层

====================================================================
✅ self-check passed:
   cache==重算 逐 token 一致 / KV 池字节==公式 / batched==solo /
   迭代账本与完成时刻==L0 模拟器（23/16/44）/ 两调度生成一致 / 等长边界持平
====================================================================

takeaway: L0 的机制在真实模型上全部成立——语义（算什么、多少迭代、
          多少浪费）由计划决定，与 L0 模拟器逐位一致；代价（墙钟、固定项
          占比）由硬件决定，CPU 与 GPU 物理不同。L2 上真实引擎（vLLM/SGLang,
          再在真实 GPU 环境把这份账对到生产数字。
```

**L1 基线指标（真实测量）**：KV cache 与重算前缀 256 个 token 逐位一致，
实测加速 5.2–5.9×（算术上界 128.5×，差距由开销分解定量解释，§4.1）；
batch 1→16 吞吐 1772→5116 tokens/s（B=2 处有 kernel 悬崖，§4.2）；
static vs continuous 迭代账本 23/16、bubble 44/0、完成时刻与 L0 逐位一致，
墙钟 47→35 ms；拟合式预测调度墙钟偏差 −6%～−9%。

---

## 3. 代码结构：真实版的最小件

`L1_real_kv_batching.py`（508 行）分五段：

1. **MiniGPT**：4 层 pre-norm decoder，GQA（8 个 Q 头共享 2 组 KV，4:1），
   RMSNorm + 2 层 MLP + 因果 attention。位置编码用可学习绝对嵌入——
   Qwen2 实际用 RoPE（其 config 里 `rope_theta=1e6`），但 **PE 选型不影响
   KV cache 机制**：cache 存的都是「已经算好的历史 K/V」，与位置怎么编码无关。
   选绝对嵌入只为省代码，不冒充 Qwen 的完整结构。
2. **KV 池**：`pool_k/pool_v` 每层一个 `[SLOTS=16, KV_HEADS, MAX_T=512, HEAD_DIM]`
   张量，**初始化时一次性 `torch.zeros` 出来**——对齐真实引擎「启动即划走整块
   显存」的做法（vLLM 的 `_allocate_kv_cache` 就是启动时一把 `torch.zeros`，§6）。
   每条序列占一个 slot，`forward` 按行读写。
3. **解码原语**：`decode_no_cache`（每步整段重跑，反例基线）/ `prefill`
   （prompt 一次并行进模型）/ `decode_cached`（每步 1 token）/
   `batched_decode_step`（一个 decode 迭代，**各行 pos 可以参差**——这是
   continuous batching 的常态，mask 按行构造）。
4. **两个调度器**：`serve_static`（整批跑到最长，finished 留在 batch 空转）/
   `serve_continuous`（每迭代：让位→接纳即 prefill→其余前进）。迭代语义与
   L0 模拟器**严格同构**：每迭代每条活跃序列恰好产出 1 个 token，接纳迭代由
   prefill 产出 token #1。另有纯账本版 `ledger_static/ledger_continuous`
   给出 L0 的期望值，供断言对照。
5. **拟合与预测**：纯 Python 最小二乘 `t(B) = a + b·B`（不引入 numpy 依赖），
   用拟合参数预测 [3] 的墙钟。

一个实现细节值得点破：`forward` 里从池子读历史 K/V 是**逐行 Python 循环拷贝**
（`k_full[b,:,:pb] = pool[slot_b,:,:pb]`）。真实引擎把这一步融进
paged-attention kernel（按 block table 直接 gather 进 attention 计算，
不落地中间拷贝）。nano 版保留显式拷贝是为了让机制可读，代价是 cache 单步
随前缀长度有 O(t) 的微涨（输出里看得见，§4.1）——这个「实现方式本身是成本」
的教训在 cat vs 预分配探针里又出现一次。

---

## 4. 输出逐段解读

### 4.1 [1] KV cache：数值逐位一致，墙钟为何只快 5 倍

两条路径的数学完全相同：cache 路径的历史 K/V 是在更早的步里用同样的权重、
同样的输入算出来存进池子的，重算路径每步重新算一遍——fp32 下逐 token
argmax 一致（256/256）。**一致性是机制的，加速是硬件的**，两件事分开验收。

实测加速 5.2–5.9×，远不到算术上界 (T+1)/2 = 128.5×。这不是机制失效，是
**开销在两条路径上都付**：对 256 步做最小二乘拟合，无 cache 单步 ≈
固定开销 0.52–0.73 ms + 每 token 计算 0.024 ms。toy 模型每 token 只有
~6 MFLOP 计算（2×3.1M 参数），0.024 ms 就算完了；剩下全是 eager 分发、
张量分配、mask 构造。用拟合参数做外推：计算放大 100 倍（≈真实 7B 级模型
每 token 的量级区间），重建加速 106–109×，逼近 128.5 的上界——
**算术上界在「计算主导」区间成立**，toy 区间被开销压扁。这正是真实引擎
拼命压每步开销（kernel 融合、C++ 调度、静态图）的原因：模型越大，
开销占比越小，但长序列 × 高并发下省掉的每一毫秒都乘以迭代数。

两个顺带的实测：cache 单步从 0.5 ms 微涨到 0.8 ms，是逐行 gather 拷贝的
O(t)（真实 kernel 里不存在这笔）；2048 步追加式 `torch.cat` 比预分配写入
慢 7×（32.6 vs 4.7 ms）——cat 每步搬运全量历史，二次复杂度。256 步上两者
持平（0.6 vs 0.6 ms，测不出差异），探针特意加长到 2048 才让证据出现：
**测不出差异 ≠ 没有差异，是样本量不够**。

### 4.2 [2] batching 曲线与 B=2 悬崖

吞吐随 B 上升（1772→5116 tokens/s），方向与 L0 一致；但曲线不是光滑仿射——
**B=1→2 有一个真实的悬崖**（0.56→1.57 ms，吞吐反而下跌），B≥4 后才恢复
近似线性增长。这不是噪声（每档 3 遍取最快仍稳定复现，threads=1 下同样存在），
profiler 给出了成因：`aten::mm` 每 forward 调用 25 次（4 层×6 个投影 + lm_head），
B=1 时单次 2.96 µs（M=1 走 gemv 快路径），B=2 时单次 19.3 µs（切换 gemm 路径，
小矩阵下效率骤降）——**kernel 分发悬崖**（本机 torch 2.13.0 CPU 实测，
探针见 §11）。真实引擎的 benchmark 曲线上那些「不平滑的台阶」，很多就是这类
kernel/编译器切换点；这也是 kernel autotuning（Triton 自动调参、torch.compile）
存在的理由。拟合 R²=0.84–0.86 而不是 0.99，主要账就记在这个悬崖上。

固定项占比 @B=16 实测 29–31%，L0 模型是 76%——**这是本节最重要的「对照失配」**，
下一节单独拆。

### 4.3 [3] 调度：语义住在计划里（跨级别契约）

同一组请求（L0 原样的 lengths=[6,2,9,3,14,5,1,8]，B=4），两个真实调度器：

- **迭代账本逐位复现 L0**：static 23 迭代 / continuous 16 迭代 / bubble 44，
  完成时刻 `[6,2,9,3,23,14,10,17]` 与 `[6,2,9,3,16,8,7,15]`——和 L0 模拟器
  一个数不差。调度语义是**计划的性质**：L0 的 20 行模拟器抓住的就是它，
  真实 forward 只是让每个迭代「变贵」，没有改变任何账目。
- **生成内容逐请求一致**：static 与 continuous 对每个请求产出完全相同的
  token 序列（3 遍复跑也逐位不变）。调度改变「何时算」，不改变「算什么」——
  序列之间无交互（无 cross-attention），greedy 解码对单条序列是确定的。
  这条性质是 serving 正确性的地基：用户拿到的输出不应取决于当时队列里
  排了谁。（采样解码下随机种子同理需按请求隔离，机制相同。）
- **代价的差异只在墙钟与浪费计算量**：static 为 finished 序列空转了 44 个
  token-step（真实 forward，计算照付），墙钟 47 vs 35 ms。L0 里 bubble 是
  账本上的数字，L1 里它是秒表上的毫秒——同一个东西的两种物态。
- **边界检查**（与 L0 同款）：长度全为 5 时 static = continuous = 10 迭代，
  continuous 的收益完全来自长度参差。

这与 nano-ray L1 的跨执行器契约、nano-data-juicer L2 的跨执行器漏斗契约
是同一个工程性质：**换基底（模拟器→真模型、串行→并行→分布式）不换语义，
且用断言把语义钉死**。数据/推理系统换引擎时真正依赖的就是它。

### 4.4 [4] 代价模型是科学对象：拟合它，然后预测

用 [2] 拟合的 t(B) = a + b·B（a≈0.98–1.03 ms，b≈0.14–0.15 ms）去预测 [3]
的墙钟：decode 段按每迭代活跃数计费（continuous 用真实活跃数历史
`[0,4,3,3,4,4,3,3,3,2,2,2,2,2,2,1]`），prefill 用探针值，得 static 偏差 −6%、
continuous −9%——**系统性偏低**，缺的那 ~9% 是调度器 Python 循环
（让位/接纳/组 batch）的开销，它不在 `batched_decode_step` 的拟合域内。
这个残差结构本身就是结论：L0 的仿射形式在真实硬件上**可拟合、可预测、
且误差可归因**。代价模型不是修辞，是可以被证伪的对象。

---

## 5. 机制深挖：CPU 与 GPU 的物理差异

### 5.1 为什么固定项占比 30% ≠ L0 的 76%

L0 的 `W_READ=1.0` 建模的是 GPU decode 的物理：**每步把全部权重从 HBM 读进
计算单元**，这笔开销与 batch 里有几条序列无关（memory-bandwidth-bound）。
batch 的价值就是摊薄这笔固定读取。

CPU 上这笔账不成立：3.1M 参数 fp32 只有 12.6 MB，完全放得进 L2/L3 cache，
「读权重」不发生 HBM 级别的搬运；batch 增大时计算量实打实地 ×B
（compute-bound），没有大固定项可摊薄。所以实测 t(B) 的固定项只是
eager 分发开销（~1 ms），斜率项才是主体——曲线接近线性而非 L0 的
「次线性趋饱和」。

这不推翻 L0，而是划清了它的适用域：**L0 是 GPU 的模型**。GPU 上的曲线
（固定项主导、吞吐随 B 次线性上升、被显存带宽/KV 读取封顶）留 L2 真机验证
`[TODO: verify on real system]`。一个 senior 的判断力正体现在这里：
拿到一条吞吐曲线，先问「固定项的物理来源是什么」，再谈 batch 策略。

### 5.2 三个「实现方式」教训

- **预分配 vs 追加**：引擎启动即划走 KV 池（vLLM `_allocate_kv_cache`，§6），
  因为 cat 式增长是二次搬运；
- **gather 融进 kernel**：nano 的逐行拷贝让 cache 单步微涨，真实 paged
  kernel 按 block table 直接读，不落地中间张量；
- **调度器用 C++**：[4] 的 −9% 残差就是 Python 调度循环的账，真实引擎
  把它压到 µs 级。

三条都是「机制正确，实现付费」的例子——L2/L3 的主题正是真实实现怎么付。

---

## 6. 与权威实现的对应（源码锚点）

| nano 部件 | 权威实现 | 锚点（2026-08-06 核验） |
|-----------|---------|------------------------|
| KV 池启动预分配 | vLLM `_allocate_kv_cache`：启动时 `torch.zeros` 整块 backing（注释原文 "Allocate once; all packed tensors alias the same backing"） | `vllm/v1/worker/gpu/attn_utils.py:L183`（def）/ L190-197（torch.zeros）@ main |
| KV 显存预算 | vLLM `determine_available_memory`：profiling 定 KV 预算，docstring 明示 `gpu_memory_utilization` 控制 | `vllm/v1/worker/gpu_worker.py:L460-461`（装饰器+def）@ main |
| block 大小 | vLLM `DEFAULT_BLOCK_SIZE = 16`（未显式指定时的解析值） | `vllm/config/cache.py:L47`（ClassVar）/ L261（解析）@ main |
| GQA 缩 KV 账 | Qwen2.5-0.5B config：24 层 / 2 KV 头 / hidden 896 / 14 注意力头（head_dim = 896/14 = 64，transformers 按 hidden÷heads 计算） | huggingface.co/Qwen/Qwen2.5-0.5B config.json |
| continuous batching 思想 | Orca iteration-level scheduling（OSDI 2022）；vLLM 论文 arXiv:2309.06180；SGLang 论文 arXiv:2312.07104 | L0 已核验（2026-08-04 arxiv.org / usenix.org），本节转引 |

注：vLLM main 已全面转向 V1 引擎，早期 `vllm/worker/cache_engine.py` 路径
不复存在，上表锚点均按 V1 目录结构。block manager / 调度器状态机 /
PagedAttention kernel 的源码细读是 L2/L3 主题。

**nano 与权威实现的差异（及原因）**：

1. 调度器：nano 是 ~60 行 Python 循环（可读优先），vLLM 是 C++/Python 混合的
   生产级调度（抢占、优先级、chunked prefill）——[4] 的 −9% 残差量化了这层差距；
2. KV 读取：nano 逐行拷贝（显式、可审计），vLLM 融进 paged-attention kernel；
3. 位置编码：nano 绝对嵌入，Qwen2 用 RoPE——不影响本节任何机制结论；
4. 硬件：CPU compute-bound vs GPU bandwidth-bound，§5.1 专门拆解，不混报数字。

---

## 7. 费曼：讲给外行听

**类比：同一本菜谱，两个厨房。**

L0 写了本菜谱（代价模型 + 调度模拟器）：几道菜、每道菜几口锅、谁先谁后、
会浪费几个灶眼。L1 真的开火做菜了——菜谱没改一个字，所以**账本完全一样**
（23 口锅 vs 16 口锅，浪费 44 个灶眼 vs 0 个）；但两个厨房的**脾气不同**：

- 这个厨房（CPU）里，请厨师本身不贵（权重在缓存里），多做一份菜就实打实
  多花一份时间——所以「一锅出 16 份」省不出多少；隔壁厨房（GPU）请厨师
  极贵（每步从 HBM 读权重），一锅多出几份几乎免费，batch 才那么值钱；
- 这个厨房换装备有台阶：做 1 份用顺手的小刀（gemv），从 2 份起必须换
  大刀架（gemm），换的瞬间反而变慢——B=2 悬崖就是换刀架的那一下；
- 备菜备忘单（KV cache）确实省了重做（256 步逐位一致还快 5 倍），但厨房
  太小、每道菜太简单，「掏备忘单」的固定动作占了大头——菜越复杂（模型越大），
  备忘单省得越多（外推 ×100 → 106 倍，逼近理论 128.5 倍）。

一句话版本：**菜谱（调度语义）决定做什么、浪费多少；厨房（硬件）决定花多久。
换厨房不改菜谱，账本逐位不变——这是可以断言的。**

反例版：如果有人声称「continuous batching 在真实模型上会改变输出」——
本节的逐请求 token 一致性断言直接证伪（调度只改时序，序列间无交互）。
如果有人声称「KV cache 的加速就是 (T+1)/2 倍」——实测 5.2× 证伪，
且差距被开销分解定量解释（上界只在计算主导区间成立）。

---

## 8. 思考题

1. **[2] 的曲线在 GPU 上会长什么样？** 提示：固定项（读权重）变成大头，
   斜率项（KV 读取 + attention）相对变小——曲线从「近线性」变回 L0 的
   「次线性趋饱和」。封顶因素换成什么？（显存带宽、KV 读取量、kernel 效率。）
   真机数字留 L2 `[TODO: verify on real system]`。
2. **B=2 悬崖怎么治？** 提示：kernel 分发是形状触发的——要么让编译器
   为小矩阵 gemm 生成专门 kernel（torch.compile / Triton autotune），
   要么干脆别在 B=2 停留（调度上快速越过）。查一查 vLLM 是否有类似处理
   `[TODO: verify source]`。
3. **[4] 的 −9% 残差为什么是系统性的而不是噪声？** 提示：预测模型只拟合了
   `batched_decode_step`，调度器的让位/接纳/组 batch 逻辑不在拟合域内；
   每迭代都付一点，方向恒为「实测 > 预测」。如果把调度循环也计时建模，
   残差会收敛到什么量级？（动手：给 serve_continuous 的循环体加 perf_counter。）
4. **GQA 的 42.7× 换成并发上限是多少？** 算术题：同样 80 GiB KV 预算、
   4096 token 上下文，Llama-2-7B 形状（0.5 MiB/token）能并发多少路？
   Qwen2.5-0.5B 形状（12 KiB/token）呢？（答案：40 vs ~1706 路——
   80 GiB ÷ (0.5 MiB×4096) = 80 GiB ÷ 2 GiB = 40，正是 L0 §6 的基线；
   80 GiB ÷ (12 KiB×4096) = 81920 MiB ÷ 48 MiB ≈ 1706。并发比 1706/40 ≈ 42.7，
   就是每 token KV 之比——「省 KV 换并发、并发换吞吐」的链条在 GQA 上的量化，
   回看 L0 §6。）

---

## 9. 局限与边界

- **随机初始化 ≠ 真实权重**：logits 分布、argmax margin、生成长度分布都与
  真实模型不同；本节验证的是**机制与账本**，不是生成质量。
- **CPU 数字 ≠ GPU 数字**：所有墙钟/吞吐仅对本机（Apple Silicon CPU，
  torch eager）成立；GPU 行为全部 `[TODO: verify on real system]`。
- **eager 开销是实现产物不是机制**：0.5–0.7 ms 的每步固定开销随 PyTorch
  版本、线程数变化（torch 2.13.0 / threads=4 口径），机制结论不依赖它。
- **fp32**：真实 serving 用 fp16/bf16，数值路径不同（两条路径仍各自确定，
  一致性验收方式不变）。
- 调度器未建模：抢占（preemption）、chunked prefill、prefix 共享——
  分别是 L2/L3 的主题。

---

## 10. 下一步 L2

1. **真实分页内存管理**：block table（逻辑块→物理块）+ 物理块池 +
   分配/回收，把 L0 §6 的分页思想变成真实张量操作；再进一步是前缀共享
   （SGLang RadixAttention 的最小版）。
2. **真机验证**：在真实 GPU/多机环境上跑 vLLM/SGLang + 小模型
   （Qwen2.5-0.5B 量级），把 [2] 的曲线与 [1] 的加速比对到 GPU 数字
   `[TODO: verify on real system]`。

---

## 11. 溯源与口径声明

- **运行环境**：torch 2.13.0（pip 安装，测试机
  macOS arm64，Python 3.13.13），CPU fp32，`torch.set_num_threads(4)`，
  seed=42。全部输出为本机真实运行。
- **三遍一致性**：连跑 3 遍全部 EXIT=0；计数类输出（迭代账本 23/16/44、
  完成时刻列表、token 一致性、边界 10、KV 字节账、参数量）三遍逐字节一致；
  计时类输出随负载浮动，观测区间（3 遍 + 定稿前多轮）：[1] 加速 5.2–5.9×、
  固定开销 0.52–0.73 ms、每 token 计算 0.024 ms（稳定）、cat 探针 6–7×；
  [2] B=1 0.55–0.58 ms、B=16 3.07–3.23 ms，拟合 a 0.98–1.03、b 0.14–0.15、
  R² 0.84–0.86（拟合参数为计时派生值，故同为区间）；[3] static 45–48 ms、
  continuous 34–35 ms；[4] 偏差 −6%～−9%。区间为 2026-08-06 初次多次运行批次
  （同日同机）的样本，随机器负载浮动，不是硬保证：另一次独立运行
  独立复跑（13:4x）计时派生值多处出区间——[1] 加速 5.0×、固定开销 0.79 ms、
  [2] B=16 2.86–2.90 ms、拟合 a 0.93、[3] 墙钟 44/32 ms——而计数类输出跨批次
  逐字节一致，机制结论不受影响。
- **B=2 悬崖探针**（2026-08-06，本机，threads=1 隔离）：torch.profiler
  实测 `aten::mm` 20 步各 500 次调用，B=1 单次均值 2.956 µs / B=2 单次均值
  19.345 µs（self CPU time 1.465 ms vs 9.658 ms）——gemv→gemm 分发悬崖的
  直接证据；threads=4 与 threads=1 下曲线形状一致（悬崖均在 B=2）。
- **源码锚点**（2026-08-06 现场核验）：vLLM main 经 raw.githubusercontent.com
  抓取 `vllm/config/cache.py`（DEFAULT_BLOCK_SIZE L47/L261）+ codeload main
  tarball 复验（同文件零漂移；`vllm/v1/worker/gpu/attn_utils.py:L183`
  `_allocate_kv_cache` 与 L190-197 torch.zeros、`vllm/v1/worker/gpu_worker.py:L460-461`
  `determine_available_memory`（L460 为 `@torch.inference_mode()` 装饰器、L461 为 def）
  逐行核对）。Qwen2.5-0.5B config.json 于
  huggingface.co 当日抓取（24 层 / num_key_value_heads=2 / hidden 896 /
  14 头；head_dim=64 为 hidden÷heads 的算术派生，config 未显式列出）。
- **论文/文档**：arXiv:2309.06180（vLLM/PagedAttention）、arXiv:2312.07104
  （SGLang）、Orca OSDI 2022——L0 于 2026-08-04 在 arxiv.org/usenix.org
  核验标题，本节转引未重复核验。
- **toy 口径**：模型随机初始化（非真实权重）；请求长度为 L0 同款固定清单
  （非真实负载分布）；墙钟仅对本机 CPU 成立。真实权重/真实引擎/GPU 数字
  一律 `[TODO: verify on real system]`，需在真实 GPU/多机环境验证。
