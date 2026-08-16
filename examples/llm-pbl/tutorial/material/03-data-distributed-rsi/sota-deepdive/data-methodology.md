# 03 轨 sota-deepdive — LLM 数据方法论：去重、质量过滤、配比与去污染

> **深挖对象**：预训练数据方法论的四个机制面——**去重（deduplication）**、**质量过滤（quality filtering）**、
> **数据配比 / 域重加权（mixture reweighting）**、**去污染（decontamination）**——以一手数据报告
> （Lee et al. / FineWeb / DCLM / DoReMi / Nemotron-CC）为唯一论据来源，以本仓库 nano 实测与
> 可运行本质模拟为实证锚点。
> **SOTA 对齐日**：2026-08-12（对齐过程与证据见 §6 与 §8；更新一代替代 = Nemotron-CC，见 §6.3）。
> **一句话**：预训练数据工程的本质不是「洗干净数据」，而是**用可测量的手段控制训练分布**——
> 去掉什么（重复/低质/背题）、留下什么（质量阈值）、各留多少（配比），每一步都是分布决策，
> 而每一步的失误都会以「条数没变、分布已污染」的静默形态反噬训练。

---

## §0 定位与可运行锚点声明

本 deepdive 不做综述式罗列。四个机制面各自回答一个「为什么」：

- **A 去重**：为什么逐字节哈希抓不住「改了几个字」的重复，而 MinHash+LSH 能用 O(1) 候选命中「≥75% 相似」？
- **B 质量过滤**：为什么「以标点结尾的行占比 ≤0.12」这类廉价启发式能切开低质文档分布？质量分类器的阈值到底是什么旋钮？
- **C 配比**：为什么用 280M 小 proxy 模型按域损失做乘性权重更新（Group DRO / minimax），能让 8B 主模型少走弯路？
- **D 去污染**：为什么 DCLM 选择「发去污染工具 + 要求披露报告」而不是直接把整个数据池清洗一遍？

**可运行锚点（本文全部 sim 数字的来源）**：

```bash
python3 -B data_methodology_sim.py   # 纯标准库，CPU 秒级，确定性输出
```

- `data_methodology_sim.py`（本目录）md5 `897abf29927e18130bda9e6f94dcc72d` / 581 行。
  纯标准库（hashlib/math/random），seed 固定、无计时行，跨运行逐字节一致。
- 2026-08-12（2026-08-12）双独立 CWD 复跑：EXIT=0 ×2、stderr 0 B、两遍 BYTE-IDENTICAL；
  输出 md5 `db8359c8c93ce1f18ef9b2c64ed3dc72` / 139 行；self-check **26/26 PASS**；
  digest（关键指标 md5）`41a90839b75f20fb115674c835201a4b`。
- sim 是**本质模拟**（课程可运行性契约）：真实算法语义（真实 MinHash/LSH、真实乘性权重重加权、
  真实 n-gram 重叠）+ toy 尺度。真实生产规模（15T token / 280M–8B 模型 / 6,000 H100 hours）
  标 `[TODO: verify on real system]`，本文只引论文一手数字。
- **[C] 配比机制面在 nano 侧无现成实测锚**，以本 sim 为可运行锚点（显式声明）；
  **[A]/[B]/[D] 的工程侧实测锚**由 nano-data-juicer / nano-ray 提供（§1.4 / §2.4 / §4.3 交叉引用，均标行号）。

**引文纪律声明**：本文全部英文引文为论文原文逐字摘录（行内引用编号如 `[71]`、公式排印如 `4\%`
按 ar5iv 纯文本归一，其余一字不动），核验方法见 §8.2。五处转录误差已按 2026-08-12 fresh 重抓
的源文纠正，纠正记录见 §8.3。

---

## §1 机制面 A — 去重：exact 子串 + MinHash/LSH 近重复 + 传递聚类

### 1.1 为什么去重是地基（Lee et al. 2107.06499）

去重之所以是 A 层经典机制，是因为它同时命中三个要害——记忆化、效率、评测可信度。
Lee et al.（去重原始提出文之一）摘要逐字：

> "We find that existing language modeling datasets contain many near-duplicate examples and long
> repetitive substrings. As a result, over 1% of the unprompted output of language models trained on
> these datasets is copied verbatim from the training data."（2107.06499 摘要）

> "removing from C4 a single 61 word English sentence that is repeated over 60,000 times"（同上）

> "Deduplication allows us to train models that emit memorized text ten times less frequently and
> require fewer training steps to achieve the same or better accuracy."（同上）

> "We can also reduce train-test overlap, which affects over 4% of the validation set of standard
> datasets"（同上）

四个数字（>1% 逐字拷贝 / 单句重复 60,000 次 / 记忆输出降 10 倍 / >4% 验证集重叠）说明：
重复不是「浪费一点算力」，而是**让模型背题、让评测失真**的分布污染。

### 1.2 两路互补：exact 子串 vs MinHash 整例

Lee et al. 的方法节逐字（两工具互补是去重机制的骨架）：

> "First, using a suffix array Manber and Myers 1993, we remove duplicate substrings from the dataset
> if they occur verbatim in more than one example. Second, we use MinHash ( Broder 1997 ), an efficient
> algorithm for estimating the n-gram similarity between all pairs of examples in a corpus, to remove
> entire examples from the dataset if they have high n-gram overlap with any other example."（2107.06499 方法节）

> "This method, which we call NearDup, is a good complement to the exact substring matching"（2107.06499 §4.2）

**为什么必须两路**：exact（suffix array）抓「逐字节相同的子串」，代价可控但抓不到改字；
MinHash 估计整例 Jaccard，抓「改了几个字的近重复」，但两两比较是 O(N²)——
于是需要 LSH 分带把候选空间压到 O(1) 桶级。

### 1.3 FineWeb 的粒度消融与 LSH 参数（一手披露最完整）

FineWeb 报告把去重做成设计空间消融，先承认选择之多：

> "While deduplication may seem as straightforward as \"removing duplicate text\", in practice many
> design choices must be made (line, paragraph, or document-level deduplication? fuzzy or exact
> matching? etc.)"（2406.17557）

三个粒度档的实测代价（逐字数字）：

> URL 去重："We explored URL deduplication, where we only kept one document per normalized (lowercased)
> URL (71.5% of tokens removed, 5.6 trillion left)"
> 行级去重："remove all but 1 (randomly chosen) occurrence of each duplicated line (77.8% of tokens
> dropped, 4.4 trillion left)"
> 行级+最小词数："only removing duplicate lines with at least 10 words and dropping documents with
> fewer than 3 sentences after deduplication (85% of tokens dropped, 2.9 trillion left)"（均 2406.17557）

**71.5% → 77.8% → 85%**：粒度越细抓得越多，但误伤风险与计算代价同步上升——
粒度选择是「去重收益 vs 多样性损失」的取舍，不是越狠越好。

MinHash/LSH 参数逐字（这是「≥75% 相似」软阈值的机制来源）：

> "We chose to collect each document's 5-grams, obtained using an English word tokenizer, and computed
> MinHashes using 112 hash functions in total, split into 14 buckets of 8 hashes each — targeting
> documents that are at least 75% similar. Documents with the same 8 MinHashes in any bucket are
> considered duplicates of each other."（2406.17557）

> "We then perform a transitive clustering step where documents A, B and C will be in the same duplicate
> cluster if A and C are duplicates and B and C are duplicates, even if A and B do not have 8 matching
> MinHashes in any bucket with each other."（2406.17557）

LSH 的数学本质：b bands×rows 分带把相似度 s 映射成候选概率 `1-(1-s^r)^b`，
S 曲线拐点 `s* = (1/b)^(1/r)`。FineWeb 的 14 buckets×8 rows 给出 `s* ≈ 0.77`——
恰好压在「targeting ≥75% similar」上：**分带参数就是软阈值的物理实现**。

### 1.4 sim 实证（[A] 区块）+ nano 实测交叉锚

sim [A] 在 toy 尺度复现全部四个机制（真实 MinHash/LSH，非伪代码）：

```text
[A1] exact-hash dedup: 4 docs -> keep 3 ['d0_base', 'd2_near', 'd3_unrel']
     removed (逐字节重复): [('d1_exact', 'd0_base')]
     near-dup 'd2_near' 仍在? 是 ← 精确哈希抓不到改字近重复
[A2] d0_base vs d2_near: true=0.643  minhash=0.641  |Δ|=0.002   <- 64 哈希签名估计 Jaccard
[A3] LSH S 曲线 (8 bands × 8 rows, 拐点 s*≈0.77):
     J≈0.50: candidate hit 4/40 = 0.10 | J≈0.75: 27/40 = 0.68 | J≈0.90: 40/40 = 1.00
[A4] 传递聚类: 边 Jaccard xy=0.82 yz=0.67 xz=0.67 (阈值 0.5)
     连通分量: [['x', 'y', 'z']]  <- x~z 即便低于阈值也经 y 同簇
```

（输出锚 md5 `db8359c8…`，§0 声明；A1–A4 全部 self-check PASS。）

**工程侧实测锚（nano 材料，交叉引用）**——去重在分布式执行下的真实失败形态与账本：

- 去重漏斗契约（三执行器逐位一致）：3360 → 2358 → 2110；重复对账本 248 对
  （同分区 12 + 跨分区 236）；naive 分区各做各的泄漏 236 条。
  锚：`nano-data-juicer/tutorial_L2.md:L53,L60-66`；`nano-ray/tutorial_L1.md:L66,L71-74`；
  `nano-ray/tutorial_L2.md:L65-66`。
- **到达顺序敏感性**（比「漏抓」更阴的失败）：反向喂 first-seen 去重，条数仍 2110 一分不差，
  但 236 个重复对的 keeper 翻成了 copy——「条数对、内容错」。min-row_id 两阶段
  （可交换聚合 + 收敛排序）对到达顺序免疫。锚：`nano-ray/tutorial_L2.md:L86-96,L317-326`；
  `nano-ray/L2_actor_dedup_index.py:L381`（`assert flipped == EXPECTED_LEAK`）。

**取舍分析**：exact-hash O(N) 廉价但只抓逐字节；MinHash/LSH 把 O(N²) 压到桶级候选但引入
假阳/假阴（sim [A3] 的 S 曲线就是误差带）；分布式下「全局 OP 当局部 OP 跑」会静默泄漏
跨分区重复（nano-data-juicer L2 的 236 条账本），而「keep-first」这类顺序语义必须显式
携带全局顺序（row_id），否则被 RPC 时序隐式决定。

---

## §2 机制面 B — 质量过滤：廉价启发式 → 质量分类器

### 2.1 启发式为什么有效：分布判别（FineWeb）

FineWeb 的起点规模与过滤池（逐字）：

> "applied quality and repetition filters from MassiveText, using the original thresholds. After
> applying this filtering to all of the WARC-based text extracted from the 96 snapshots available at
> the time of writing, we obtained roughly 36 trillion tokens of data when tokenized with the GPT-2
> tokenizer."（2406.17557）

> "a principled strategy for choosing and tuning filtering heuristics that helped produce a small set
> of effective filters out of over fifty candidate filters from past work"（2406.17557）

**五十候选里只留三个**——启发式过滤的方法论核心不是「多」，而是「可判别」：
一个特征值不值得用，看它在高质/低质语料上的分布是否分得开。选定阈值逐字：

> "the chosen filters remove documents where the fraction of lines ending with punctuation is <=0.12
> (10.14% of tokens removed vs. 30% from the original C4 terminal punctuation filter), where the
> fraction of characters in duplicated lines is >=0.1 (12.47% of tokens removed; the original
> MassiveText threshold for this ratio is >=0.2), and/or where the fraction of lines shorter than 30
> characters is >=0.67 (3.73% of tokens removed). When applying the three together, ~22% of tokens
> were removed and the aggregate score increased by about 1% in the 28B token ablations."（2406.17557）

阈值 0.12 的选取依据是分布判别（Fig 8 口径）：

> "the lower quality dataset has a much higher density of documents for values < 0.12"（2406.17557）

谱系参照（C4 的启发式过滤是这条路的源头之一）：

> "C4 was constructed from the 2019-18 crawl by applying heuristic filters, which included dropping
> lines without a terminal punctuation mark, that mentioned javascript, or that had
> \"terms-of-use\"/\"cookie policy\" statements, and dropping documents that were too short"（2406.17557）

### 2.2 质量分类器：FineWeb-Edu 与 DCLM 的两条路线

**FineWeb-Edu（LLM 标注 → 分类器过滤）**逐字：

> "we introduce FineWeb-Edu, a 1.3-trillion token collection of educational text filtered from
> FineWeb. LLMs pretrained on FineWeb-Edu exhibit dramatically better performance on knowledge- and
> reasoning-intensive benchmarks like MMLU and ARC."（2406.17557）

> "Applying the classifier to the 15 trillion tokens of FineWeb required 6,000 H100 GPU hours."（2406.17557）

> "FineWeb-Edu achieves a 33.6% accuracy on the MMLU benchmark at only 38 billion tokens,
> significantly outperforming Matrix (second best on the metric), which reaches similar accuracy at
> 300 billion tokens."（2406.17557）

**DCLM（fastText 二分类器 = 系统对比后的最优）**逐字：

> "Training a fastText classifier for filtering performs best."（2406.11794，Table 4 题注）

Table 4（1B-1x 尺度，Core/Extended 列）八种过滤策略对比，数字逐一核验在位：
RefinedWeb reproduction 27.5/14.6 | Top 20% by Pagerank 26.1/12.9 | SemDedup 27.1/13.8 |
Classifier on BGE features 27.2/14.0 | AskLLM 28.6/14.3 | Perplexity filtering 29.0/15.0 |
Top-k average logits 29.2/14.7 | **fastText OH-2.5 +ELI5 30.2/15.4**（2406.11794 Table 4）。

策略枚举（逐字节选）："1) PageRank score filtering …" / "Semantic Deduplication (SemDedup)" /
"AskLLM which prompts an LM to see if a document is helpful" / "Perplexity filtering where we retain
low perplexity sequences following CCNet" / "7) fastText binary classifiers to distinguish data
quality"（2406.11794）。

**结论的份量**：DCLM 用 416 组实验把「model-based filtering is key」打成了实证结论：

> "we conduct 416 baseline experiments with different training sets and compute scales. Our
> experiments identify model-based filtering as a key component"（2406.11794）

> "a standardized corpus of 240T tokens extracted from Common Crawl, effective pretraining recipes
> based on the OpenLM framework, and a broad suite of 53 downstream evaluations" + "model scales
> ranging from 412M to 7B parameters"（2406.11794）

> "We also release DCLM-baseline, a 3.8T token high-quality dataset from our pool that yields better
> models than prior datasets."（2406.11794）

> "The resulting dataset, DCLM-baseline, enables training a 7B parameter language model from scratch
> to 64% 5-shot accuracy on MMLU with 2.6T training tokens." + "Compared to MAP-Neo, the previous
> state-of-the-art in open-data language models, DCLM-baseline represents a 6.6 percentage point
> improvement on MMLU while being trained with 40% less compute."（2406.11794）

> "Our baseline model is also comparable to Mistral-7B-v0.3 and Llama 3 8B on MMLU (63% & 66%), and
> performs similarly on an average of 53 natural language understanding tasks while being trained
> with 6.6× less compute than Llama 3 8B."（2406.11794）

**两条路线的取舍**：FineWeb-Edu 用 LLM 标注（贵：6,000 H100 hours 打 15T token）换「教育性」
这一语义维度；DCLM 用 fastText（廉：浅层 n-gram 分类器）换可规模化——
3.8T/240T ≈ 1.6% 的保留率下仍撑起 7B 64% MMLU。共同点：**阈值 = 「质量 vs 数据量」的唯一旋钮**，
过滤激进度直接决定保留率，保留率直接决定 long-horizon 训练可行性（§6.3 Nemotron-CC 的批评正源于此）。

### 2.3 sim 实证（[B] 区块）

```text
[B1] 启发式特征分布: good×3 punct_line_frac=1.000 | bad×3 punct_line_frac=0.000
     分布判别: min(good punct)=1.000 > max(bad punct)=0.000 => 可分: True
[B2] 应用 FineWeb 阈值 (punct<=0.12 或 short>=0.67 即丢): kept=good×3  dropped=bad×3
[B3] 质量分数阈值 → 保留率 (n=200, mock 分类器分数, 右偏):
     threshold=0.1: keep=187 | 0.2: 131 | 0.3: 94 | 0.4: 50 | 0.5: 28   单调下降: True
[B4] 静默污染: 正确阈值保留 94/200, 配置 typo 掉回默认保留 200/200 —— 条数『没变』但分布已污染
```

（[B3] 的分数为 **mock 分类器分数**，显式声明——演示「阈值→保留率」的单调旋钮语义；
真实 fastText/LLM 打分见 DCLM/FineWeb-Edu 路线。）

### 2.4 nano 实测交叉锚（真实小样本 + 静默污染实证）

- rule→llm 过滤漏斗（真实 10 条医学 SFT 样本）：10→10→10→7→7——thinking_length_filter
  （4500 字符阈值）砍 3 条，LLM scorer（threshold=3）7 条全留。
  锚：`nano-data-juicer/tutorial_L1.md:L177-189,L205-210`。
- filter 接口语义与 stats 复用：text_length_filter min_len=900，3360→2358，
  第二个同参 filter 实算 0 次（stats 命名空间复用）。锚：`nano-data-juicer/tutorial_L3.md:L47,L55-70`。
- **静默污染实证**（与 sim [B4] 同构）：配置 typo `min_lne=900` 不报错、掉回默认 min_len=10，
  保留 3360/3360——条数「看起来没变」，分布已被污染。锚：同上 L55-70。

---

## §3 机制面 C — 数据配比 / 域重加权：DoReMi 的 Group DRO 本质

### 3.1 论文一手机制（2305.10429）

命题与机制逐字：

> "The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web text) greatly
> affect language model (LM) performance."（2305.10429 摘要）

> "we propose Domain Reweighting with Minimax Optimization (DoReMi), which first trains a small proxy
> model using group distributionally robust optimization (Group DRO) over domains to produce domain
> weights (mixture proportions) without knowledge of downstream tasks. We then resample a dataset with
> these domain weights and train a larger, full-sized model."（2305.10429 摘要）

尺度与结果逐字：

> "we use DoReMi on a 280M-parameter proxy model to set the domain weights for training an
> 8B-parameter model (30x larger) more efficiently."（2305.10429）

> "On The Pile, DoReMi improves perplexity across all domains, even when it downweights a domain.
> DoReMi improves average few-shot downstream accuracy by 6.5% points over a baseline model trained
> using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training
> steps."（2305.10429 摘要）

> "On the GLaM dataset, DoReMi, which has no knowledge of downstream tasks, even matches the
> performance of using domain weights tuned on downstream tasks."（2305.10429）

> "matching the proxy and main model sizes results in a 4x average speedup"（2305.10429，消融）

### 3.2 sim 实证（[C] 区块）：minimax 的三个可运行签名

**nano 侧无配比实测锚，本机制面以 sim 为可运行锚点（显式声明，§0）**。
toy 设定：4 域，逐域独立指数损失曲线 `L_d(t)=floor+(init-floor)·exp(-rate·t)`，
域 1 起点最高（3.5）、下降最慢（rate 0.5）——「最难域」；默认配比 [0.40, 0.10, 0.30, 0.20] 偏经验、亏待域 1。

**签名一（C1）：乘性权重把预算压向最难域，收敛时各域损失近似拉平**——这是 minimax 不动点：

```text
[C1] Group DRO proxy 收敛后的域权重 (vs 默认配比):
     domain 0: default=0.40 -> DRO=0.012 ↓  (收敛损失=0.509)
     domain 1: default=0.10 -> DRO=0.956 ↑  (收敛损失=0.600)   <- 最难域
     domain 2: default=0.30 -> DRO=0.030 ↓  (收敛损失=0.501)
     domain 3: default=0.20 -> DRO=0.003 ↓  (收敛损失=0.461)
     minimax 不动点签名: 收敛时各域损失近似拉平 (max=0.600 min=0.461 spread=0.139)
```

**签名二（C2）：同预算下 worst-domain 损失更低**（minimax 的直接定义）：

```text
[C2] 同预算 80 token: Group DRO worst=0.6000 avg=0.5180 | 固定默认 worst=0.6531 avg=0.5133
```

注意 avg 一栏：**DRO 的平均损失反而略高（0.5180 vs 0.5133）**——minimax 用一点平均代价换最坏域抬升，
这不是 bug，是目标函数决定的取舍。

**签名三（C3）：守住 worst-domain 质量底线所需 token 显著更少**：

```text
[C3] 把所有域拉过质量底线 (worst-domain 损失 <= 0.75) 所需总 token:
     自适应 Group DRO: 11.0 步 | 固定默认配比: 60.0 步 | speedup = 5.45x
```

固定默认的瓶颈一目了然：最难域只拿到 10% 预算，60 步里 54 步在等它。

**[C3b] toy 边界观察（不外推，必须读）**：

```text
[C3b] 静态 proxy 权重两阶段 + 平均损失目标 0.85:
     DoReMi 静态权重: 64.0 步 | 默认配比: 16.0 步 (比值 4.00)
```

在 toy 上，DoReMi 的真实部署形态（静态权重重采样 + 平均损失目标）**反而更慢**。原因有二，
都是 toy 的结构性边界：① 平均损失由易域主导，静态权重把预算压到难域、易域被饿着；
② toy 的逐域独立损失曲线**没有跨域迁移**，而真实 LM 中域间知识会迁移——这正是论文
「even when it downweights a domain」全域改善在真实尺度成立的机制之一。
因此：**论文 8B 尺度「2.6x fewer steps / 6.5pp」是经验结果，toy 只演示 minimax 机制方向，不外推**。
（sim [C4] 在 toy 内仍复现了「被降权域损失也下降」：域 [0,2,3] 从 init 下降为 True。）

### 3.3 取舍分析

- **proxy 的价值是便宜的试错**：280M 试出的权重给 8B 用（30x），消融显示 proxy/main 同尺寸
  反而有 4x 加速——「用小模型选配比」本身就是效率设计，不是精度妥协。
- **minimax ≠ 平均最优**：Group DRO 的收敛态是「各域损失拉平」，不是「平均损失最低」。
  选配比前先问目标：要下游全面（平均）还是要短板不塌（worst-case）？DoReMi 选的是后者，
  而下游收益是它在大规模上的经验副产品（§3.2 [C3b] 的 toy 证据正说明二者不可互推）。
- nano 侧的概念衔接：给数据打分再按分数筛选/加权是「数据路由/配比」的最简形态
  （`nano-data-juicer/tutorial_L1.md:L301-302`），本机制面是它的理论纵深。

---

## §4 机制面 D — 去污染：先可测量，再谈清洗

### 4.1 DCLM 的去污染哲学（一手原文）

> "Test set samples often contaminate language model training sets; however, the effect of such
> samples on downstream performance remains largely unclear."（2406.11794）

> "To allow researchers to better understand contamination, we release decontamination tooling instead
> of decontaminating DCLM-Pool directly."（2406.11794）

> "we implement our own decontamination process for two popular tasks, MMLU and Hellaswag"（2406.11794）

> "we provide tooling based on Lee et al. 2022 to examine datasets for overlap with all of our test
> sets. We ask all submissions to disclose a decontamination report"（2406.11794）

**为什么不直接清洗整个池**：因为污染对下游的影响「remains largely unclear」——
在机制没搞清之前，把 240T 池子按某一套 n-gram 阈值洗一遍，等于用一个未经论证的假设
永久改变所有人的训练分布。DCLM 的选择是**把测量工具发出去、把披露义务写进提交协议**：
污染先变成可测量、可比较的量，清洗决策留给证据。这是「数据基准」范式区别于「数据集」的关键动作。

### 4.2 探针本质与对照口径

n-gram 重叠探针的量化源头（Lee et al.，与 §1 同文）：

> "We can also reduce train-test overlap, which affects over 4% of the validation set of standard
> datasets, thus allowing for more accurate evaluation."（2107.06499 摘要）

对照口径（并非所有数据集都选择去污染——FineWeb 在 Paloma 域覆盖评测中刻意不做，
理由是评测目标不同）：

> "We use the codebase provided in [71] but intentionally do not perform decontamination, to compare
> how well each dataset covers different domains."（2406.17557；[71] 为 Paloma 引用编号）

### 4.3 sim 实证（[D] 区块）+ 已知盲区

```text
[D1] 4-gram 重叠率: clean1=0.000 clean2=0.000 exact_copy=1.000 near_copy=0.444 paraphrase=0.000
     ⚠ 改写(paraphrase) 重叠率仅 0.000 —— n-gram 探针抓得住拷贝、抓不住改写
[D2] 阈值 0.3 判污染: flagged=['exact_copy', 'near_copy']  干净文档误报: []  植入污染漏报: []
[D3] 去污染后保留: ['clean1', 'clean2', 'paraphrase']  (3/5)
```

三个可迁移结论：① 逐字拷贝重叠率=1.0、干净文档≈0，n-gram 是廉价而有效的「背题」探针；
② 近拷贝（轻微改写）仍有 0.444 的高重叠，阈值 0.3 可捕获；
③ **语义改写级污染重叠率≈0，逃过检测——这是 n-gram 法的已知盲区**，
需语义/嵌入级去污染补位（DCLM 工具谱系之外的开放问题，标 [TODO: verify] 待一手来源补充）。

---

## §5 横切视角：四个机制面是同一条主线

**主线：用可测量手段控制训练分布。**去重控制「重复质量」，过滤控制「样本质量」，
配比控制「域间比例」，去污染控制「评测独立性」——四者都是对训练分布的显式干预，
且共享同一失败形态：**干预失误不报错，只以分布污染的形式在训练末端结账**
（sim [B4] 的 typo 全保留 / nano-data-juicer L3 的 min_lne 静默掉默认 / nano-ray L2 的 236 条内容翻转）。

与数据受限标度律的咬合（为什么「有效 unique token」是真货币）：

> "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having
> unique data. However, with more repetition, the value of adding compute eventually decays to zero."
> （2305.16264 摘要）

≤4 epoch 内重复几乎无损、超过后算力价值衰减归零——这给出去重的标度律意义：
去重/质量过滤省下的每一个 unique token 都是真实购买力（Nemotron-CC 以
「four times more unique real tokens」为卖点，§6.3）。

与 03 轨 nano 阶梯的咬合：数据方法论的算子在 nano-data-juicer（OP 语义/stats 复用/静默污染）
里落地，在 nano-ray（分区/收敛点/顺序语义，收敛点搬 6.14 MB vs 索引路线 94 KB = 65×，
`nano-ray/tutorial_L2.md:L102-107`）里分布式化，最终喂给 nano-vllm-sglang 的推理侧
（paged KV 预算 32 块准入 8/8 vs 连续分配 5/8、前缀命中省 prefill 2.6×，
`nano-vllm-sglang/tutorial_L2.md:L40-54`）——数据质量决定训练，训练产物（模型）决定推理，
推理轨迹又回流成数据（RSI 闭环，轨道 03 主线）。

---

## §6 SOTA 对齐（对齐日 2026-08-12，课程的证据时效性分层三层锚点）

### 6.1 A 层经典锚点（机制仍是地基，长期保留）

- Lee et al. 2107.06499（去重两工具原始提出；记忆化/效率/评测三害的量化源头）
- RefinedWeb 2306.01116（纯 web 数据 + 过滤谱系："RefinedWeb uses trafilatura for text extraction,
  fastText for language identification, heuristic rules inspired by MassiveText (discussed below) to
  filter data, and both MinHash (fuzzy) and ExactSubstr (exact) deduplication."——2406.17557 转述句，逐字核验）
- Scaling Data-Constrained 2305.16264（数据受限标度律，§5 引文）

**当今定位**：去重/启发式过滤/n-gram 去污染仍是所有现代 pipeline 的地基层机制，
但「激进过滤换基准分」的路线已被新一代修正（§6.3）。

### 6.2 A/B 交界 → 机制面规范锚点（本 deepdive 的教学主体）

- FineWeb 2406.17557（去重/启发式/edu 分类器全流程消融，一手披露最完整）
- DCLM 2406.11794（质量分类器系统对比 + 数据基准范式 + 去污染工具哲学）
- DoReMi 2305.10429（配比重加权机制，Group DRO/minimax）

### 6.3 B 层前沿主流（更新一代替代，记录存在性 + 机制类别，不作教学主体）

- **Nemotron-CC 2412.02595**（2024-12）——对「激进 model-based filtering」的正面修正，摘要逐字：

> "Recent English Common Crawl datasets like FineWeb-Edu and DCLM achieved significant benchmark gains
> via aggressive model-based filtering, but at the cost of removing 90% of data. This limits their
> suitability for long token horizon training, such as 15T tokens for Llama 3.1."

> "a combination of classifier ensembling, synthetic data rephrasing, and reduced reliance on
> heuristic filters" + "using a high-quality subset of our data improves MMLU by 5.6 over DCLM" +
> "our full 6.3T token dataset matches DCLM on MMLU, but contains four times more unique real tokens
> than DCLM" + "an 8B parameter model trained for 15T tokens, of which 7.2T came from our dataset, is
> better than the Llama 3.1 8B model: +5 on MMLU, +3.1 on ARC-Challenge, and +0.5 on average across
> ten diverse tasks"（均 2412.02595 摘要）

  处理口径（同 02 轨 DeepSeek-V4 先例）：机制类别 = 分类器集成 / 合成改写 / 弱化启发式；
  数字以摘要为准，机制细节（各组件消融）不作教学主体，标 `[TODO: verify]` 待全文核验。

- DCLM 自述更新一代（逐字，v4 版本正文）："Since the initial release of DCLM, newer works such as
  WebOrganizer, Nemotron-CC, and Olmo-2 have also built upon our benchmark or curation strategies to
  further advance the state-of-the-art for LLM pre-training datasets."（2406.11794）

### 6.4 对齐结论与未决项

- 本 deepdive 教学主体（A/B 层四机制面）在对齐日无「被取代」风险——它们是地基层机制；
  前沿更替发生在「过滤激进度」策略层（FineWeb-Edu/DCLM 激进 → Nemotron-CC 回调），已按 B 层记录。
- `[TODO: verify]`：2025–2026 是否有更新的旗舰级数据方法论报告；本次没有完成按
  submittedDate 倒序的全量扫描，不能把已有来源集合解释为穷尽性检索。
- `[transient/单源]`：FineWeb-2（多语版）等单源条目未逐一核验，不展开。

---

## §7 费曼自检

### 7.1 讲给外行听（类比 ×4）

- **去重 = 图书馆剔旧**：完全一样的书（exact）直接剔；但「换了封面改了序言的重版书」（near-dup）
  得靠内容指纹（MinHash）+ 按指纹分柜抽查（LSH），不可能逐本两两对照。
- **质量过滤 = 安检**：金属探测门（启发式）便宜、快、能拦掉明显危险品；开包检查（分类器）
  贵但准。安检阈值调高，漏检少但通行慢（保留率低）——阈值就是那个旋钮。
- **配比 = 备考分配时间**：哪科最拉垮就把时间往哪科压（minimax），而不是哪科容易拿分就刷哪科
  （平均最优）——但「押最难科」不等于「总分最高」，这是两件事（[C3b]）。
- **去污染 = 考前防泄题**：先装监控（发检测工具 + 要求披露），而不是直接宣布「所有疑似题目
  一律从教材删掉」——因为「疑似」的标准本身还没被证明合理。

### 7.2 动手思考题 ×5

1. sim [A3] 把 bands/rows 从 8×8 改成 16×4，拐点 s* 怎么变？对「≥75% 相似」的召回/误报各有什么影响？（改 sim 跑一遍验证你的推断）
2. sim [B3] 的保留率曲线是 mock 分数。若真实分类器分数是双峰分布（高质/低质两坨），阈值旋钮的行为会有什么不同？为什么这反而让阈值更好选？
3. sim [C2] 中 DRO 的平均损失略高于默认（0.5180 vs 0.5133）。如果你的下游指标对最差域敏感（如低资源语言 perplexity），你选哪个？如果只看平均 benchmark 呢？
4. nano-ray L2 证明 first-seen 去重对到达顺序敏感（236 条翻转）。如果把「keep-first」改成「keep-random」，顺序敏感性问题消失了吗？代价是什么？
5. DCLM 选择「发工具 + 披露」而非「直接清洗」。在什么条件下（给出你的判据）直接清洗才是正确选择？

### 7.3 反例 ×5（方法何时失效）

1. **去重过度**：行级激进去重（85% token dropped 档）在代码/模板化语料上会误伤——合法重复（API 签名、法律条款）是内容本身。粒度必须匹配语料类型。
2. **启发式过滤的分布漂移**：0.12 阈值是在英文 web 上选的（Fig 8 分布判别）；换到代码/数学/多语语料，特征分布整体平移，原阈值静默失效——又是「条数没变、分布已污染」。
3. **分类器过滤的目标偏移**：fastText/Edu 分类器优化的是「像高质量参考数据」，不是「对训练有用」——过滤掉的 90% 里可能含着 long-horizon 训练必需的多样性（Nemotron-CC 的批评，§6.3）。
4. **配比权重的轨迹依赖**：DoReMi 权重是 proxy 轨迹的产物；sim [C3b] 显示静态权重换目标函数（平均→worst）结论翻转。权重不是数据的内禀属性，是「目标 × 轨迹」的函数。
5. **n-gram 去污染的盲区**：改写级污染重叠率≈0（sim [D1] paraphrase=0.000）——背题但换了说法，探针全盲。

### 7.4 局限声明

- sim 全部数字为 toy 尺度（4 文档/6 文档/4 域/5 文档），**只演示机制方向，不外推生产数字**；
  论文数字（15T token / 6,000 H100 hours / 2.6x / 6.5pp 等）为一手报告声称值，本文未独立复现。
- sim [C] 的逐域独立损失曲线无跨域迁移，平均损失口径与论文不可互推（§3.2 [C3b] 已展开）。
- 论文引文行号未标注（arXiv 无稳定行号）；节号/表号以 2026-08-12 ar5iv 抓取为准。
- nano 交叉引用锚点行号以 2026-08-12 盘上状态为准（材料冻结，漂移概率低）。

---

## §8 溯源

### 8.1 一手来源表（全部 2026-08-12 现场核验）

| arXiv ID | 标题 | 提交/更新 | 一作 | 核验通道（2026-08-12） |
|----------|------|-----------|------|------------------------|
| 2107.06499 | Deduplicating Training Data Makes Language Models Better | 2021-07-14 / 2022-03-24 | Katherine Lee | export.arxiv.org API（标题/日期/作者）+ ar5iv 全文 253,098 B |
| 2406.17557 | The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale | 2024-06-25 / 2024-10-31 | Guilherme Penedo | 同上 + ar5iv 全文 290,898 B |
| 2406.11794 | DataComp-LM: In search of the next generation of training sets for language models | 2024-06-17 / 2025-04-21 | Jeffrey Li | 同上 + ar5iv 全文 1,212,788 B |
| 2305.10429 | DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | 2023-05-17 / 2023-11-21 | Sang Michael Xie | 同上 + ar5iv 全文 331,076 B |
| 2305.16264 | Scaling Data-Constrained Language Models | 2023-05-25 / 2025-06-28 | Niklas Muennighoff | export.arxiv.org API（含摘要逐字） |
| 2306.01116 | The RefinedWeb Dataset for Falcon LLM | 2023-06-01 | Guilherme Penedo | export.arxiv.org API（标题级） |
| 2412.02595 | Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset | 2024-12-03 / 2025-05-30 | Dan Su | export.arxiv.org API（摘要六探针全 HIT） |

API 批次：`https://export.arxiv.org/api/query?id_list=…` HTTP 200 / 20,141 B / 7 entries
API 暂时返回 429 时改用 arxiv.org/abs 规范页 + ar5iv 双通道核验；API 恢复后的复核与两通道结论一致。

### 8.2 引文核验方法

本文 46 处逐字引文全部经 **2026-08-12 抓取**的 ar5iv 全文机器核验：
空白折叠后逐字匹配（EXACT）或归一化（引号/破折号变体、引用编号 `[ N ]` 剥离、
公式双写如 `4 % 4\%`→`4%`）后匹配，46/46 在位；DCLM Table 4 的 16 个数字逐一 grep HIT。
核验脚本与抓取件同目录（`verify_quotes.py`）。

### 8.3 转录误差纠正记录（早期转录 → 源文纠正，共 5 处）

1. FineWeb 传递聚类句尾：误录 "8 ma[tching hashes]" → 源文 "8 matching MinHashes in any bucket with each other"。
2. DCLM-baseline 发布句：误录 "we release DCLM-baseline" → 源文 "We **also** release DCLM-baseline"。
3. Lee 两工具句尾：误录止于 "high n-gram overlap" → 源文 "…high n-gram overlap **with any other example**"。
4. DCLM 第七策略：误录止于 "fastText binary classifiers" → 源文 "…to distinguish data quality"。
5. FineWeb Paloma 句：误录 "[Paloma]" → 源文为引用编号 "[71]"（指 Paloma）。
另两处口径说明（非误差）：F10 句尾源文尚有 "in the 28B token ablations"（本文补全）；
F12/D6 源文用双引号与 "6.6×"（本文从源文）。

### 8.4 nano 实测锚点表（跨轨相关材料，交叉引用，2026-08-12 盘上行号）

| 锚点 | 数字 | 位置 |
|------|------|------|
| 去重漏斗契约 | 3360→2358→2110；248 对（12 同分区 + 236 跨分区）；naive 泄漏 236 | nano-data-juicer/tutorial_L2.md:L53,L60-66；nano-ray/tutorial_L1.md:L66,L71-74；nano-ray/tutorial_L2.md:L65-66 |
| 到达顺序敏感性 | 反向喂翻转 236 keeper；min-row_id 零翻转 | nano-ray/tutorial_L2.md:L86-96,L317-326；L2_actor_dedup_index.py:L381 |
| rule→llm 过滤漏斗 | 10→10→10→7→7（threshold=3） | nano-data-juicer/tutorial_L1.md:L177-189,L205-210 |
| filter 语义 + 静默污染 | min_len=900 3360→2358；stats 复用 0 重算；typo min_lne→3360/3360 | nano-data-juicer/tutorial_L3.md:L47,L55-70 |
| 配比概念指针 | 打分→筛选/加权 = 数据路由最简形态 | nano-data-juicer/tutorial_L1.md:L301-302 |
| 分布式搬运账 | 收敛点 6.14 MB vs 索引 94 KB（65×）；RPC 0.37 ms | nano-ray/tutorial_L2.md:L102-107 |
| paged KV | 预算 32 块准入 8/8 vs 5/8；前缀命中 prefill 2.6× | nano-vllm-sglang/tutorial_L2.md:L40-54 |

### 8.5 四类信息区分

- **原文声称**：§1–§4 全部英文引文及其数字（§8.1/8.2 核验链）。
- **文献已有**：MinHash/LSH 数学（Broder 1997，经 Lee/FineWeb 转述）、Group DRO 框架（Sagawa et al.，经 DoReMi 转述）——本文未直接核验原始文献，标转述。
- **合理推断**：§5「四个机制面 = 分布控制主线」的归纳；§7.3 反例 1–4 的失效条件推演（基于 sim 实证 + 论文机制，非论文原文声称）。
- **猜测**：无——不确定处均已标 `[TODO: verify]`（§6.4 两项 / §4.3 一项）。
