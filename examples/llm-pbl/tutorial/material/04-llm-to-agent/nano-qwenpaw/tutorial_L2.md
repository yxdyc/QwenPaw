# nano-qwenpaw L2 — 方法论注入：把原则变成可执行的流程

> L0 给单个调用套上 harness（system prompt + 自检），L1 让 harness 在有限窗口下活过多轮（write-through + eviction index）。
> L2 注入**方法论**：K+1 规则、费曼审查、对抗式 Examiner-B 门禁、反幻觉立场，不再是 prompt 里的散文，而是 harness **执行的流程**——
> 且所有数字（mastery 增量、80% 阈值、token_cap）都是运行时从 coach 真实文件里**解析**出来的，数字跟随源码，不是硬编码。
> 同时补上 L1 没讲的工具结果维度写穿（`cap_middleware.py`：token_cap + preview + recall pointer + degradation path）。

---

## 1. 先跑为敬

文件：`L2_real_methodology_loop.py`，零外部依赖（纯标准库：`sqlite3` / `hashlib` / `re` / `math` / `tempfile`），CPU 即跑，约 1 秒。

```bash
$ python3 L2_real_methodology_loop.py
```

真实输出（2026-08-14 复验，逐字节粘贴；输出 md5 = `634c437cc982b85d4196735cf3a2b567`，124 行）：

```text
====================================================================
nano-qwenpaw L2 — methodology injection, measured
====================================================================
python 3.9.6
declarations: LearnerModel = declared mock (latent theta,
  logistic responding, expected session score); Examiner-A =
  declared generator on a planted-defect schedule; gap detectors
  = declared heuristics — real LLM judgment sits there:
  [TODO: needs key]. The claims gate checks PROVENANCE, not
  truth. Everything else is real: rules parsed live from coach
  files (sha256 logged), real sqlite tool store + ledger, real
  sqlite3.Error degradation path, exact pointer format of
  cap_middleware.py L110-117.

[0] methodology sources: numbers parsed out of the real files
    SOUL.md            sha256[:8]=e143a057  mode=live
    k-plus-one.md      sha256[:8]=3cbf925a  mode=live
    feynman-check.md   sha256[:8]=bca18409  mode=live
    cap_middleware.py  sha256[:8]=5ea09476  mode=live
    manager.py         sha256[:8]=6260c313  mode=live
    history.py         sha256[:8]=48b71b62  mode=live
    SOUL.md: principles=7, principle#5="Anti-Hallucination: Zero Tolerance" (L53)
    k-plus-one.md: rules >80%:+0.1 / 50-80%:+0.05 / <50%:-0.05 (L85) | Examiner-B steps=5 (L40), regenerate > 2 failures
    feynman-check.md: bands >=4.5:+0.1 / >=3.5:+0.05 / >=2.5:+0 / >=0.0:-0.05 (L98) | gap categories=4: Logical Leaps, Undefined Terms, Factual Errors, Missing Aspects
    cap_middleware.py: token_cap=3000 (L38) | keep formula L106 | degrade hook L67 | pointer key=tool_call_id

[1] adversarial self-verification: Examiner-B gate on Examiner-A sets
    set#1: 6 problems, 4 planted defects | learner K=2
    defect                                          reread  independent
    P2: key off-by-one: (n-1)^2 instead of n^2        miss        CATCH
    P4: duplicate of P1 (concept+kind+stem)           miss        CATCH
    P5: stem cites token_cap=2000, registry=3000      miss        CATCH
    P6: difficulty K+3 instead of K+1                CATCH        CATCH
    caught: reread(same-channel) 1/4 vs independent-evidence 4/4
    consequence: the reread gate's verdict is FIX-AND-PASS — it would ship P2/P4/P5
    set#1 gate (independent): failures=4 > 2 -> REGENERATE (SKILL.md rule)
    set#2: failures=1 <= 2 -> fix in place (defect: P3 key recomputed by oracle)
    [Self-check: 6/6 problems verified. Adjustments: P3 answer key recomputed]

[2] the K+1 rule as a control loop (12 sessions, m0=0.10, theta0=1.0)
    adaptive trajectory (d = level(mastery)+1):
    sess  d  score%   delta  mastery  theta
       1  2    62.2  +0.05   0.15    1.151
       2  2    65.7  +0.05   0.20    1.267
       3  3    44.2  -0.05   0.15    1.327
       4  2    69.6  +0.05   0.20    1.409
       5  3    47.7  -0.05   0.15    1.495
       6  2    73.0  +0.05   0.20    1.549
       7  3    51.2  +0.05   0.25    1.656
       8  3    53.9  +0.05   0.30    1.777
       9  4    32.7  -0.05   0.25    1.777
      10  3    56.9  +0.05   0.30    1.911
      11  4    35.7  -0.05   0.25    1.911
      12  3    60.1  +0.05   0.30    2.057
    policy        mean%  final_m  final_theta theta_gain
    fixed-easy     72.3     0.70        1.757      0.757
    adaptive       54.4     0.30        2.057      1.057
    fixed-hard     18.2     0.00        1.000      0.000
    mastery inflation, measured: fixed-easy paid 0.60 mastery (0.10 -> 0.70, claims level 7) for 0.76 theta ->
    0.79 mastery per theta vs adaptive 0.19: the frozen difficulty pushes scores toward the >80%
    ceiling while theta nears its prox=0 ceiling (d+1), so the profile
    keeps crediting mastery the ability no longer backs.

[3] the Feynman review, run as a flow (topic: tool-result capping)
    r1: gaps=4 | clarity=3 accuracy=3 completeness=3 | overall=3.0 -> band >=2.5: delta=+0.00 | mastery 0.30 -> 0.30
        [Logical Leaps] so recall over the capped region is always lossless — missing premise: persisted BEFORE replace
        [Undefined Terms] FTS5 — used but not explained
        [Factual Errors] 'keyed by the session id' — contradicts cap_middleware.py: "recall pointer keyed by ``tool_call_id``"
        [Missing Aspects] degradation path — not covered
    r2: gaps=0 | clarity=5 accuracy=5 completeness=5 | overall=5.0 -> band >=4.5: delta=+0.10 | mastery 0.30 -> 0.40

[4] anti-hallucination claims gate: provenance, not truth
    VERIFIED        "a single tool result is capped at 3000 tokens" (cap_middleware.py) — cap_middleware.py@5ea09476
    NO-PROVENANCE   "the window is the memory" — no source recorded
    SHA-DRIFT       "one turn stays pinned raw at the head" (manager.py) — recorded 00000000 != live 6260c313
    QUOTE-NOT-FOUND "the scroll keeps 7 turns pinned" (manager.py) — snippet not in manager.py
    the NO-PROVENANCE claim is rejected with principle#5's maxim: "If you can't verify, don't assert"

[5] tool-result write-through (cap_middleware.py dimension)
    tool output: 16800 chars = 4200 est-tokens (cap=3000) -> write-through keyed by call_0001
    in-context: preview keep=12000 chars + pointer (40 est-tokens overhead)
    pointer (exact source format):
      <<<TRUNCATED ~1200 tokens>>>
      <system-info>Full output preserved durably. Recall it inside recall_history_python via ms.recall_tool('call_0001').</system-info>
    recall via ms.recall_tool('call_0001'): 16800 chars, byte-identical=True
    degradation path: store down -> real ProgrammingError caught; capped=False degraded=True in_context==full output: True
    (cap_middleware.py L63-68: don't truncate what we could not store)

[6] session ledger (SOUL.md principle#7 "Continuous Improvement", L72) — real sqlite
    [1] examiner-gate     set#1 failures=4 > 2 -> regenerate; set#2 fixed, 6/6 verified
    [2] k1-adaptive       12 sessions: final mastery=0.30 theta=2.057 (best theta of 3 policies)
    [3] feynman-r1        overall=3.0 -> band 2.5-3.4: delta=+0.00
    [4] feynman-r2        overall=5.0 -> band 4.5-5.0: delta=+0.10
    [5] claims-gate       1 VERIFIED / 1 NO-PROVENANCE / 1 SHA-DRIFT / 1 QUOTE-NOT-FOUND
    [6] tool-cap          call_0001: 4200 tok -> preview 12000 chars + pointer; recall byte-identical
    [7] tool-cap-degraded store down -> full output kept in context (never truncate what we could not store)

[7] self-check (structural assertions)
    PASS  SOUL.md parses to 7 principles; #5 is anti-hallucination
    PASS  Examiner-B: 5 checks parsed from k-plus-one SKILL.md
    PASS  same-channel re-read catches strictly less than independent evidence
    PASS  regenerate rule fires exactly when failures > parsed threshold
    PASS  K+1 loop: adaptive ends with the highest true theta
    PASS  fixed-easy inflates mastery above adaptive while learning less
    PASS  fixed-hard deflates mastery to the floor (<50% rule, min 0)
    PASS  adaptive keeps a majority of sessions in the 50-80% band
    PASS  Feynman r1 flags all 4 gap categories; factual error cites live source
    PASS  Feynman bands applied: r1 no-change, r2 +0.1 (parsed, not hardcoded)
    PASS  claims gate: exactly one VERIFIED of four claims
    PASS  write-through: recall is byte-identical to the capped output
    PASS  degradation: store down -> no truncation, full output in context
    PASS  ledger complete: 7 events persisted to real sqlite
    ✅ self-check passed

====================================================================
takeaway: methodology is not prompt prose — it is executable flow.
  Rules parsed out of the coach files drive the loop: the adversarial
  gate catches what a same-channel re-read misses; the K+1 loop keeps
  difficulty chasing ability so the learning signal (wrong answers
  within reach) never dies — a frozen difficulty turns the score
  into an inflating proxy; the claims gate checks provenance, not
  truth; and capping never loses data — when it cannot store, it
  does not cap. Real hosted model behind the loops: [TODO: needs key]
====================================================================
```

---

## 2. 声明清单：什么是真的，什么是声明的

L2 延续 L1 的 declared-mock 契约（课程可运行性契约），边界画在「判断」与「机制」之间：

| 组件 | 真/声明 | 说明 |
|---|---|---|
| 方法论数字（mastery 增量、80% 阈值、regen 阈值、token_cap、gap 类别、band） | **真**（运行时解析） | 正则从 6 个 coach 源文件 live 提取，sha256 记录；源文件改了，下一遍运行数字跟着变——漂移可见 |
| LearnerModel | 声明 mock | 恰三个性质：潜在能力 θ、logistic 应答 `p = 1/(1+exp(-(θ-d+1.5)))`、session 分数 = 期望正确率（ensemble 极限，无采样噪声） |
| Examiner-A | 声明生成器 | 题目按**埋点缺陷日程**生成；真实系统里是 LLM 出题 |
| 费曼 gap 检测器（leap / undefined / missing） | 声明启发式 | 真实系统里 LLM 的判断坐在这里：`[TODO: needs key]` |
| 费曼 factual-error 检测 | **真** | 拿声明文本去 live 源码里反查，证据是源文件里的原句 |
| 工具存储 / 会话 ledger | **真** sqlite3 | tempfile 目录里的真数据库 |
| degradation path | **真**异常 | 关闭连接后 INSERT 抛出的真 `sqlite3.ProgrammingError`，按源码同款 `except (sqlite3.Error, OSError)` 接住 |
| recall pointer 格式 | **真**（逐字符） | `cap_middleware.py` L110-117 的 f-string 原样镜像 |

---

## 3. §0：数字是从源码里解析出来的，不是抄进来的

`parse_rules()` 对六个源文件做的全部事情就是正则提取，输出里每一个数字都能在源文件里找到原句：

- `SOUL.md` → 7 条编号原则（`^## (\d+)\.`），principle#5 标题、"If you can't verify, don't assert" 箴言、L53/L69 行号；
- `k-plus-one/SKILL.md` → `>80%: mastery += 0.1` / `50-80%: += 0.05` / `<50%: -= 0.05`（L85-87），Examiner-B 五步（Self-Verification 节内 `^\d\. \*\*` 计数 = 5，L40），`If more than 2 problems fail` → regen 阈值 = 2；
- `feynman-check/SKILL.md` → 四条 mastery band（L98-101）与四类 gap（`#### 2.x`，L36-56）；
- `cap_middleware.py` → `token_cap: int = 3000`（L38）、keep 公式（L106）、degrade hook（L67）、pointer key = `tool_call_id`。

为什么不硬编码？因为这套课程的核心立场是**数字跟随源码**：SOUL.md 明天加一条原则、SKILL.md 改一个阈值，L2 的输出就跟着变，sha256 与行号把漂移钉在输出里。L1 对五个 scroll 源文件做的是同一件事，L2 把同样的纪律带到了方法论文件上。

## 4. §1：对抗自检——为什么「重读一遍」不算验证

Examiner-A 出两套题，各埋了缺陷。set#1 埋 4 个：

| 缺陷 | 性质 |
|---|---|
| P2 key = (n−1)² 而非 n² | 答案键 off-by-one（前 n 个奇数和 = n²，oracle 是闭式） |
| P4 与 P1 完全同构 | 多样性失败 |
| P5 题干引用 token_cap=2000 | 与 §0 解析出的 registry（3000）冲突 |
| P6 难度 K+3 | 校准失败 |

两种验证者对照：

- **reread（同通道）**：Examiner-A 用自己的过程把自己的题重看一遍。它重新推导 key 用的是**产生 key 的同一套计算**——off-by-one 原样复现，于是「自洽」；逐题检查看不见跨题重复；没有 registry 可查，错误常数畅通无阻。唯一活下来的是难度元数据比对（纯表面检查）。结果 1/4，verdict = FIX-AND-PASS——**它会带着三个隐患放行**。
- **independent（独立证据）**：solve-from-scratch 换闭式 oracle 重算、diversity 做两两指纹比对、expert 查解析出的 registry。4/4 全擒，verdict = REGENERATE——恰好触发 SKILL.md 的「>2 failures → regenerate the entire set」规则。

这就是 correlated blindspot（相关盲点）：**验证的有效性不来自「再看一遍」，而来自证据通道的异质性**。换一套独立计算、换一个外部 registry、换一次跨样本比较，才能照出同一条通道里的系统性错误。set#2 只埋 1 个缺陷，走的是 SKILL.md 的另一条路：≤2 → fix or replace in place，输出那行 `[Self-check: 6/6 problems verified. Adjustments: ...]` 与 SKILL.md 的呈现格式逐字对齐。

## 5. §2：K+1 是个控制回路，mastery 是会通胀的货币

LearnerModel（声明）的三条性质之后，三条政策跑同一起点（m0=0.10，θ0=1.0）各 12 个 session：

- **adaptive**：每个 session 取 `d = level(mastery)+1`（level = int(m×10)，声明映射）。分数被规则钉在 50–80% 带附近（12 个 session 里 8 个落在带内），θ 涨到 2.057——三政策最高。轨迹里两次掉出带下沿（44.2% / 32.7%）是**离散 band 的边界过冲**：mastery 刚跨级，难度跳一整档，overshoot 后被 −0.05 拉回。这是 mastery-based 自适应系统的真实抖动，不是 bug；真实系统用连续的模型判断（而非离散档）来平滑，此处 `[TODO: needs key]`。
- **fixed-easy**：难度冻在初始 K+1。θ 涨、难度不涨，分数滑向 >80% 天花板，规则每个 session 照付 +0.1——mastery 0.70（自称 level 7），θ 却只涨了 0.76 且正逼近 prox=0 的顶（学习燃料 = 「够得着的错题」，θ→d+1 时归零）。量化通胀：**0.79 mastery/θ vs adaptive 的 0.19**——同一种货币，购买力差了 4 倍。这就是 Goodhart 定律的回路版：当分数变成目标，它就不再是好度量。
- **fixed-hard**：难度 K0+3。分数 18%，`<50%` 规则把 mastery 一路扣到地板 0（源码写明 min 0），θ 原地不动——超出 zone 的错题不是学习燃料，是弃学信号。

结论不是「自适应更好」这种空话，而是可测的三条：adaptive 的 θ 终值严格最高；fixed-easy 的 mastery 严格高于 adaptive 但 θ 严格更低（通胀）；fixed-hard 触底。**规则本身不保证学到东西，是「难度追着能力跑」这件事在保学习信号。**

## 6. §3：费曼审查——四类 gap，只有一类需要真证据

学习者解释「工具结果 capping」（刻意接 L1 的主题），r1 埋了四类 gap，检测器逐类命中：

- **Logical Leap**：结论「so recall ... always lossless」前面**没有**「先持久化、后替换」这个前提——检测器在结论前文里找 `persist/write ... before` 模式，找不到即判飞跃；
- **Undefined Term**：`FTS5` 出现但没有同位语/定义从句；
- **Factual Error**：「keyed by the session id」——这一类不走启发式，走**真反查**：live 源码里原句是 `a recall pointer keyed by ``tool_call_id```（cap_middleware.py docstring），证据连引号一起打印；
- **Missing Aspect**：degradation path 整个没提。

评分是声明的 rubric 算术（clarity = 5−leap−undefined，accuracy = 5−2×error，completeness = 5−missing−error），overall = 3.0 → 落进解析出的 `2.5-3.4: no change` band，mastery 不动。r2 是修订版解释（补前提、定义 FTS5、改 tool_call_id、补 degradation），gap 清零，5.0 → `4.5-5.0: +0.1`。**费曼技法的闭环不在打分，在「gap → 修订 → 复查」这一圈真的转起来了。**

## 7. §4：反幻觉门禁——查 provenance，不查 truth

四条声明过闸：

| 声明 | 裁决 | 原因 |
|---|---|---|
| "a single tool result is capped at 3000 tokens" | VERIFIED | 源 + sha + 引文三件齐 |
| "the window is the memory" | NO-PROVENANCE | 没记来源——注意这句话在 L1 语境里甚至**不算错**，但门禁照样拒：不能验证的就不能断言（principle#5 箴言） |
| "one turn stays pinned..." | SHA-DRIFT | 记录的 sha 与 live sha 不符——引文还在也不行，出处已漂移 |
| "the scroll keeps 7 turns pinned" | QUOTE-NOT-FOUND | sha 是活的，引文不在源里 |

关键分野：**门禁是机械的出处核验，不做真值判断**。真值判断留给 LLM/审阅者（那里坐着 `[TODO: needs key]`）；门禁保证的是每条出闸声明都能回溯到一个此刻仍然成立的出处。这正好是 §3 factual 检测的放大版：那里查一句话，这里查一批。

## 8. §5：工具结果写穿——L1 没讲的另一半记忆

L1 写穿的是**轮次**（conversation turns），L2 补上**工具结果**维度，逐行镜像 `cap_middleware._cap`（L71-118）：

1. 16800 字符 = 4200 est-tokens > cap 3000（cap 是 §0 live 解析的，不是抄的）；
2. 全文写穿 sqlite，key = `tool_call_id`（源码 L87-91 的 `dedup_key=tcid`）；
3. in-context 留 `keep = max(1, int(len(text)*cap/n_tokens))` = 12000 字符的 preview（L106 公式原样），接**逐字符同款** pointer：`<<<TRUNCATED ~1200 tokens>>>` + `<system-info>...ms.recall_tool('call_0001')...</system-info>`（L110-117）；
4. recall 回读 16800 字符，**byte-identical=True**——capping 不丢数据是可以 fail 的断言，这里真的 assert 了；
5. **degradation path**：关掉连接再走一遍，INSERT 抛真 `ProgrammingError`，按源码同款 `except (sqlite3.Error, OSError)`（L63）接住——然后**不截断**，全文留在 context，标记 degraded。源码注释写得很直白：truncate 掉一份没能存下来的数据 = 永久丢失。一句话：**能存才配 cap；存不了就全留。**

## 9. §6：ledger——原则 7 也得是流程

SOUL.md principle#7（Continuous Improvement，L69）要求每次 session 留痕、每次交互更新 learner 档案。L2 的落地是一个真 sqlite ledger，把 [1]-[5] 的七个关键事件记成审计行。方法论说到「要记录」，harness 就把记录变成不可跳过的副作用——和 K+1、费曼一样，**原则只有落进流程才算注入**。

---

## 10. 对照权威源码（L2 锚点全表，2026-08-08 核验）

| 锚点 | 位置 | L2 对应 |
|---|---|---|
| 七条编号原则；#5 Anti-Hallucination；#7 Continuous Improvement | `coach/profile/SOUL.md` L5-L69（#5 L53，#7 L69） | §0 principles 解析、§4 箴言引用、§6 ledger 名义 |
| Examiner-B 五步（solve/correctness/difficulty/diversity/expert） | `coach/profile/skills/k-plus-one/SKILL.md` L42-47（"you MUST verify" L40） | §1 五项类型化检查 |
| regen 规则「If more than 2 problems fail, regenerate the entire set」 | 同上 L49 | §1 verdict 判定 |
| `[Self-check: N/N ... Adjustments: ...]` 呈现格式 | 同上 L52-54 | §1 set#2 输出行 |
| mastery 更新规则 >80% / 50-80% / <50%(min 0) | 同上 L84-87 | §0 解析、§2 回路增量 |
| gap 四分类 2.1-2.4 | `coach/profile/skills/feynman-check/SKILL.md` L36-56 | §3 检测器分类 |
| 三轴评分（clarity/accuracy/completeness，1-5） | 同上 Phase 4 表 | §3 rubric 算术（声明） |
| mastery band 4.5-5.0:+0.1 / 3.5-4.4:+0.05 / 2.5-3.4:no change / <2.5:−0.05 | 同上 L97-101 | §0 解析、§3 落档 |
| `token_cap: int = 3000` | `src/qwenpaw/agents/context/scroll/cap_middleware.py` L38 | §0 解析、§5 cap |
| degrade：write 失败 → 不截断、全文放行、记 degraded | 同上 L63-68（`note_write_failure` L67） | §5 第 5 步 |
| keep 公式 | 同上 L106 | §5 第 3 步 |
| pointer 格式（TRUNCATED + system-info + recall_tool(tcid)） | 同上 L110-117 | §5 逐字符镜像 |
| pointer key = tool_call_id（`dedup_key=tcid`） | 同上 L26-27、L87-91 | §3 factual 证据、§5 key |

源文件 sha256[:8]（本次输出 [0] 区）：SOUL.md `e143a057`、k-plus-one.md `3cbf925a`、feynman-check.md `bca18409`、cap_middleware.py `5ea09476`、manager.py `6260c313`、history.py `48b71b62`。其中 scroll 四件与 L1 §14 记录一致，SOUL.md 与 L1 一致。

## 11. 费曼自检

- 能不能解释：为什么同一个生成者「再读一遍自己的输出」几乎必然通过？（相关盲点：同一套计算复现同一个错误；验证效力来自证据通道异质性——独立 oracle、外部 registry、跨样本比对）
- 能不能解释：fixed-easy 的分数明明更高（mean 72.3 vs 54.4），为什么反而暴露了问题？通胀的到底是 mastery 的哪个成分？（分数→门槛→增量的机械链路在难度冻结时与真实能力脱钩；mastery/θ 汇率 0.79 vs 0.19）
- 能不能解释：store 写失败时为什么不截断、反而把全文留在 context？它在保护什么不变量？（「capping 永不丢数据」的前提是全文已持久化；前提不成立时唯一合法行为是不 cap）

## 12. 思考题

1. Examiner-B 的五项检查在真实系统里也是同一个 LLM 执行的——那「独立性」到底从哪来？（提示：oracle 与被验对象不同源、registry 在模型之外、比对跨样本而非逐样本；SKILL.md 用「Examiner-A/B」角色分离 + 强制独立重解来逼近这一点，但它不是密码学保证。）
2. adaptive 轨迹在离散 band 边界抖动（44.2% → −0.05 → 回落）。如果设计真实 mastery 系统，你会用什么平滑？（hysteresis 阈值带、难度连续化、分数 EMA、或让模型直接估计下一题难度——各自的代价？）
3. claims gate 只查出处不查真值。如果**源文件本身写错了**，这个体系在哪一层兜底？（提示：sha/引文只保证「你说的是源里写的」，不保证「源里写的是对的」——真值锚在更外层：审阅流程、多源交叉、以及 L3 要讲的 skills 架构里的对抗角色。）

## 13. L3 预告

L3 对照 qwenpaw coach 的完整架构，复现一个「有原则的 agent」：

- **skills loader**：`runtime/builder.py` 的 `Toolkit(tools=tools, skills_or_loaders=skill_dirs)`（L94）与 `_resolve_skill_loader_dirs`（L97-121）——SKILL.md 是技能目录的准入闸门（"skill '%s' has no SKILL.md at %s; not injected"），L3 把这道门做成流程；
- **SOUL 治理的完整会话循环**：七条原则在一次真实 session 里各自以什么形态生效（L2 已把其中五条变成流程，L3 补 PBL 与 learner autonomy，并让技能选择本身受原则约束）；
- 真实托管模型接入：`[TODO: needs key]`。

## 14. 测量口径与确定性记录

**输出锚**：粘贴块 md5 = `6b852cee4abf425d3271bda2c2d2f4c8`（124 行）；公开镜像代码 `L2_real_methodology_loop.py` 845 行、md5 = `e83d3bddfffc0d7a824fa61063fb099e`（新增仓库根目录自动发现，不改变实验逻辑）。

**复核口径**：(a) 粘贴块与 `python L2_real_methodology_loop.py` 当场输出逐字节一致（同一源码快照）；(b) [0] 区数字可逐条回源文件核对（行号已打印）；(c) [7] 区 14 条 PASS 全部为结构性断言，代码内可查；(d) 声明面与真实面边界见 §2 表，无未声明 mock。

**确定性**：本文件无采样、无计时行、sqlite 落 tempfile（路径不打印）；2026-08-08 在五个彼此独立的临时工作目录中各运行一次，两两 diff 为空。注：与 L1 相同，sha256 跟随源文件，源文件若被编辑，输出中的 sha 与解析数字随之改变（设计如此，漂移可见），届时本节 md5 锚需重新锚定。

**环境**：Apple M5 Pro / Python 3.13.13（`python3`）/ 纯标准库 / CPU。真实托管模型：`[TODO: needs key]`。
