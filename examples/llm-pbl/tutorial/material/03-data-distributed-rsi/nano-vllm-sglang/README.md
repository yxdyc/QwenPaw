# nano-vllm-sglang

> **抓的核心机制**：高吞吐推理引擎——**PagedAttention**（vLLM）与 **continuous batching / RadixAttention**（SGLang），让 llm-based 算子和 RL 采样的成本可接受。
> **对应真实系统**：[vLLM](https://github.com/vllm-project/vllm)（论文 arXiv:2309.06180）/ [SGLang](https://github.com/sgl-project/sglang)（论文 arXiv:2312.07104）
> **轨道**：[03 数据/分布式/RSI](../README.md) · **状态**：L0–L3 ✅

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | 玩具：代价模型 + 迭代模拟器抓住四个最小机制——KV cache 计算/显存账、batching 吞吐曲线、static vs continuous batching、PagedAttention 分页思想（纯标准库） | ✅ [`L0_kv_cache_batching.py`](L0_kv_cache_batching.py) · [教程](tutorial_L0.md) |
| **L1** | 把 L0 的三个机制换上真实小模型（随机初始化 GQA GPT，3.1M 参数）：KV cache 数值一致 + 开销分解、batching 真实曲线（含 gemv→gemm 悬崖）、static vs continuous 真实调度——迭代账本与 L0 逐位一致，并用拟合代价模型预测调度墙钟 | ✅ [`L1_real_kv_batching.py`](L1_real_kv_batching.py) · [教程](tutorial_L1.md) |
| **L2** | 真实分页内存管理（block table + 物理块池 + refcount + 内容哈希前缀缓存 + CoW + 抢占/重算）：跨级别契约 paged==L1 逐 token 一致、碎片准入 8/8 vs 5/8、前缀命中省算力 2.2–2.6×、CoW 恰 2 次父块字节不变、抢占重算后语义不变——对照 vLLM V1 源码行号级核验；L3 压力场景挖出 allocate-before-touch 竞态，已修为 touch-before-allocate（vLLM issue #33775 同因，实录与不变性证明见 tutorial_L3 §6）；真机吞吐/KV 显存 `[TODO: verify on real system]` | ✅ [`L2_real_paged_memory.py`](L2_real_paged_memory.py) · [教程](tutorial_L2.md) |
| **L3** | radix tree 前缀缓存 + 多请求共享（SGLang RadixAttention 最小同构）：并发共享 lock_ref=4 脊柱块只读、叶子优先 LRU 压力下保住 SYS 脊柱（实测 radix 196 vs 哈希链 260 vs 无缓存 324 tok，FIFO 截断点实测派生）、LPM 消除 thrashing（96 vs 192 tok）且语义与顺序无关——对照 SGLang radix_cache.py / vLLM V1 分配流源码行级核验（2026-08-13 抓取）；真机引擎 `[TODO: verify on real system]` | ✅ [`L3_radix_prefix_sharing.py`](L3_radix_prefix_sharing.py) · [教程](tutorial_L3.md) |

**环境依赖**：L0 零外部依赖（纯标准库），CPU 即跑：`python3 L0_kv_cache_batching.py`。
L1 依赖仅 `torch`（实测 torch 2.13.0 / macOS arm64），CPU 即跑：
`python3 L1_real_kv_batching.py`（greedy + 固定 seed，计数类输出确定）。
L2 依赖仅 `torch` + 同目录 L1 模块（import 不落 `__pycache__`），CPU 即跑约 1 分钟：
`python3 L2_real_paged_memory.py`（计数类输出确定，计时行仅 [3] prefill 墙钟）。
L3 依赖同 L2（`torch` + 同目录 L1/L2 模块），CPU 即跑约 10 秒：
`python3 L3_radix_prefix_sharing.py`（计数类输出确定，计时行仅 [3] elapsed 行，
口径 `sed '/^[[:space:]]*elapsed/d'` 可掩码）；真实引擎（vLLM/SGLang on GPU）
验证需在真实 GPU/多机环境验证 `[TODO: verify on real system]`。

### 可选真实引擎探针

[L2L3_gpu_verify.py](L2L3_gpu_verify.py) 用 SGLang offline Engine 测单模型的
batch decode 与共享前缀缓存。它要求显式 `--model`，并支持 `--device`、
`--attention-backend`、`--mem-fraction`、`--quick`；默认只写 stdout，只有显式
`--log` 才落盘。末行 `RESULT_JSON=` 同时记录模型、SGLang 版本、后端、设备、
吞吐与 cache 指标。绝对时间只属于该次 model × revision × backend × driver ×
GPU 组合，不能用来证明 vLLM 与 SGLang 的普遍优劣；没有真实 package 和模型时
应保留为未运行，而不是用 mock/fallback 补数。

截至 2026-08-31，本轮 L20 环境虽有可离线安装的 SGLang wheel，但本地 Hugging Face
缓存只有模型仓库元数据，没有完整开放权重，因此没有执行或发布吞吐数字。该状态不是探针失败，
而是其“真实 package + 完整本地模型”前置合同未满足；后续拿到固定 revision 后再运行。

## 核心要讲清的点

- decode 为何 memory-bandwidth-bound；KV cache 显存为何随 batch×seq 线性增长，PagedAttention 如何用「分页」减少碎片
- continuous batching：iteration 级动态进出，而非等一批凑齐（Orca，OSDI 2022）
- 前缀复用（RadixAttention）：system prompt 共享时如何省算力
- **分页 = 数据结构问题**（L2）：block table 保证「分页不改变数学」（paged==L1 逐 token 断言）；
  refcount 一肩挑共享/在用/空闲三角色；链式内容哈希使「同前缀」可判定可复用，
  因果性是前缀共享无损的物理基础
- **释放 ≠ 清空**（L2）：ref_cnt=0 的块同时是空闲内存与缓存条目——前缀缓存零额外空间；
  块重分配才失效哈希（对照 vLLM `_maybe_evict_cached_block`）
- **共享块只读，写时分裂**（L2）：CoW 规则一条（ref_cnt>1 → 先复制再写），
  父块字节不变 + 子轨迹 == solo 断言
- **过载是可恢复的代价**（L2）：准入（full-fit）+ 抢占最晚者（FCFS）+ 重算恢复——
  语义不变（token == 无压力参考），代价实付（重算 token 数）
- **前缀是一等数据结构**（L3）：radix tree 把共享变成节点、驱逐变成剪叶、
  调度看树选请求——lock_ref 沿父链保护运行中前缀，叶子优先 LRU 让祖先
  活到变成叶子为止，LPM 用「已命中长度」排序消除 thrashing
- **touch 必须先于 allocate**（L3 §6）：free queue 里的块带着缓存条目，
  先分配后 touch 会让同一物理块一表双别名、scatter 覆写命中 KV——
  静默发散（vLLM issue #33775 同因；带缓存条目的 free queue 不可简化为
  普通 free list）

## 费曼自检

- 能不能解释「为什么推理引擎的吞吐瓶颈往往是显存带宽而非算力」？
- 能不能解释「省下的 KV 显存是怎么一步步变成吞吐的」？
- 能不能解释「为什么前缀共享必须配 copy-on-write，而 CoW 又不改变任何生成结果」？
  （L2 §6：共享块只读、写时分裂；因果性 + greedy 使轨迹逐 token 不变）
- 能不能解释「释放的块为什么还能被前缀命中，这份额外的『缓存』空间从哪来」？
  （L2 §7/§9：缓存不是副本，是还没被覆盖的空闲块）
- 能不能解释「为什么叶子优先驱逐比按释放顺序复用保住的前缀多」？
  （L3 §4/§8：FIFO 最先发走最早释放的脊柱块；叶子是局部损失，祖先是全局枢纽）
- 能不能解释「为什么 touch 必须先于分配，vLLM 的两阶段分配在防什么」？
  （L3 §6：命中块尚在 free queue 时，先分配会弹出命中块本身 → 一表双别名）

## 权威实现与延伸

- 对标源码：vLLM `github.com/vllm-project/vllm`（PagedAttention / block manager）；SGLang `github.com/sgl-project/sglang`（RadixAttention）
- 概念延伸：为轨道 01 RL rollout、轨道 04 agent 提供推理后端
