# nano-qwenpaw

> **抓的核心机制**：**agent harness / coach**——把 LLM 包成一个有方法论、有记忆、能自我检查的执行体（本仓库 qwenpaw_coach 同源）。
> **对应真实系统**：qwenpaw（本仓库 `coach/`）
> **轨道**：[04 LLM→Agent](../README.md) · **状态**：L0–L3 ✅（阶梯完成）

---

## 阶梯（L0–L3）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | 玩具：给一个 LLM 调用套上「system prompt + 输出自检」最小 harness | ✅ [`L0_harness_loop.py`](L0_harness_loop.py) · [`tutorial_L0.md`](tutorial_L0.md) |
| **L1** | 加记忆与上下文管理：跨轮状态、何时压缩/检索——同一真实语料（qwenpaw 源码 facts + LLM-PBL 散文）、同一预算下实测三政策损失谱：append-only 40/40%（静默、按新近度单调）/ summarize 10/20%（salience≠importance、不可预测、1041←2417 不可逆）/ write-through+evict-index 100/100%（ctx 985≤1000、store 24==24、FTS5 召回）；对照 qwenpaw scroll 源码行号级核验 | ✅ [`L1_real_memory_loop.py`](L1_real_memory_loop.py) · [`tutorial_L1.md`](tutorial_L1.md) |
| **L2** | 注入方法论：把 K+1 / 费曼 / 对抗自检变成 harness 的内置流程——方法论数字全部 live 解析自 coach 六源文件（7 原则 / Examiner-B 5 检查 / regen>2 / mastery 规则与 4 band / token_cap=3000，行号随输出打印）；实测相关盲点：同通道重读 1/4（verdict 会放行 3 个隐患）vs 独立证据 4/4（触发 regen 规则）；K+1 控制回路实测 mastery 通胀：fixed-easy 0.79 mastery/θ vs adaptive 0.19（adaptive θ 终值 2.057 三政策最高、fixed-hard 触底 0）；费曼 r1 四类 gap 全擒（factual 用 live 源码反查）3.0→no change、r2 5.0→+0.1；claims gate 1 VERIFIED/3 拒（查 provenance 不查 truth）；工具结果写穿 4200 tok→preview 12000 字符+逐字符同款 pointer、召回 byte-identical、store 宕机走真 ProgrammingError 不截断 | ✅ [`L2_real_methodology_loop.py`](L2_real_methodology_loop.py) · [`tutorial_L2.md`](tutorial_L2.md) |
| **L3** | 对照 qwenpaw coach 的 SOUL.md + skills 架构，复现一个「有原则的 agent」：方法论变成数据——manifest 定 enablement（真实 profile 实测 9 目录 vs 8 enabled，codex-delegate never effective）、channel 过滤 reach（同一 workspace 双渠道两个 agent：console 3 skills vs voice 3 skills，差集恰 feynman-check/voice-brief）、SKILL.md 准入门含 builder 原样日志（onboard 埋点实测 gated out）、skills ride the prompt 不在 tools=（整段与独立 block 的粗略 token 差在整数取整误差内一致）、capability 在注入文档不在 harness（同一解释 console 席验证后 +0.00 vs voice 席放行 +0.05，mastery 0.30 vs 0.35，10 轮投影 0.75 vs 1.25 = L2 通胀的技能缺失版）；七原则全部取形（#3 PBL/#6 autonomy 新增形态）；arch 三源行号锚 live 推导 assert>0（builder L94/L97/L117、registry L1186/L1197、store L65/L73） | ✅ [`L3_principled_agent.py`](L3_principled_agent.py) · [`tutorial_L3.md`](tutorial_L3.md) |

## 环境依赖（分级）

- **L0**：零外部依赖（纯标准库），CPU 即跑。
- **L1**：零外部依赖（纯标准库：sqlite3 / hashlib / re / math / random），CPU 即跑，约 1 秒；
  完全确定性（seed=42、无计时行）——同一源码快照上两遍运行输出逐字节一致
  （2026-08-14 复验：当前源码与环境下自检通过）。
  运行时 live 读取本仓库五个源文件（sha256 记录、漂移可检测、pinned snapshot fallback）；
  真实托管模型路径 `[TODO: needs key]`。
- **L2**：零外部依赖（纯标准库：sqlite3 / hashlib / re / math / tempfile），CPU 即跑，约 1 秒；
  完全确定性（无采样、无计时行、sqlite 落 tempfile 不打印路径）——同一源码快照上
  两遍运行输出逐字节一致（2026-08-08 核验：五个独立工作目录各运行一次，两两 diff 为空）。
  运行时 live 读取 coach 六个源文件（SOUL.md + k-plus-one/feynman-check SKILL.md +
  scroll 三件，sha256 记录、pinned snapshot fallback），方法论数字正则解析自源文本；
  degradation path 用真 `sqlite3.ProgrammingError`；真实托管模型 `[TODO: needs key]`。
- **L3**：零外部依赖（纯标准库：json / re / sqlite3 / tempfile / shutil + import 同目录
  L2，`sys.dont_write_bytecode = True` 置于 import 之前不落 pyc），CPU 即跑，秒级；
  完全确定性（seeded、无采样、无计时行、sqlite 与 fixture workspace 落 tempfile 不打印
  路径，成功运行结束自动清理 fixture）——同一源码快照上两遍运行输出逐字节一致
  （2026-08-14 复验：当前源码与环境下自检通过）。
  运行时 live 读取 coach 六源 + arch 三源（builder/registry/store，sha256[:8] 记录、
  行号锚 live 推导 assert>0、pinned snapshot fallback）；真实托管模型 `[TODO: needs key]`。

## 核心要讲清的点

- harness vs 裸调用：system prompt / 工具 / 记忆 / 自检如何改变行为
- 上下文工程：窗口有限时，什么留、什么压、什么检索——L1 实测三种损失谱：
  append-only 静默且按新近度单调地丢头部；summarize 的损失不可预测
  （salience=词元稀有度，与任务重要性无关，实测 3 个保留槽位 2 个被 padding 占走）；
  write-through+evict-index 无损且预算有界
- **窗口是 cache，store 才是 memory**（L1）：记忆可靠性是存储设计，不是模型属性——
  write-through 24==24 不变量 + FTS5 召回在真实 sqlite 上检查
- 方法论注入：把「对抗式自检」「反幻觉」变成可执行流程而非口号（L2）——
  实测四件事：同通道重读抓 1/4、独立证据 4/4（验证效力来自证据通道异质性）；
  难度冻结时 mastery 通胀 0.79 vs 0.19 mastery/θ（分数天花板照付增量，学习信号已死）；
  反幻觉门禁查 provenance 不查 truth（sha 漂移/引文缺失/无出处三种拒法）；
  capping 永不丢数据——包括存不下来时不 cap（degradation 是不变量的一部分）
- **方法论变成数据**（L3）：skill 是携带 SKILL.md 的目录，原则是可加载的工件——
  directory ≠ skill（enablement 在 manifest，真实 profile 9 目录 vs 8 enabled）；
  SKILL.md 是准入门（effective ≠ injectable，builder 原样日志）；channel 过滤 reach
  （同一 workspace 按请求组装出两个 agent）；skills ride the prompt 不在 tools=
  （文本性能力 vs 功能性能力）；capability 在注入文档不在 harness——同一解释
  console 席 +0.00 vs voice 席 +0.05，行为差完全归因缺失的文档（L2 通胀的技能缺失版）

## 费曼自检

- 能不能解释「同一个模型，套上 harness 前后，行为差异来自哪里」？
- 能不能解释「为什么 summarize 的损失不可预测，而 append-only 的损失按新近度单调」？
  （L1 §5/§9：salience=词元稀有度，facts 共享词汇互相摊低 IDF）
- 能不能解释「逐出的轮为什么不是删掉，而是留一行 `[seq N]` 索引」？
  （L1 §7/§8：seq 是全局唯一地址，nothing is lost 是可以 fail 的断言）
- 能不能解释「为什么『自己重读一遍自己的输出』几乎必然通过，换一条证据通道就能抓住」？
  （L2 §4：相关盲点——同一套计算复现同一个错误；也解释 fixed-easy 高分为什么是通胀而非学会）
- 能不能解释「同一个 workspace 为什么对 console 请求和 voice 请求是两个 agent」？
  （L3 §6：channel 过滤 reach——组装按请求发生，差别只在哪些文档被注入）
- 能不能解释「同一份解释为什么在一个渠道被验证、在另一个渠道被放行，放行的学分去了哪里」？
  （L3 §8/§10：capability 在注入文档不在 harness；放行学分进了 mastery，通胀也是渠道问题）

## 权威实现与延伸

- 对标源码：qwenpaw 本仓库 `coach/profile/SOUL.md`（7 条编号原则，2026-08-14 核验；
  principle 5 = Anti-Hallucination）与 `coach/profile/skills/`；L1 已核验锚点：
  `src/qwenpaw/agents/context/scroll/` 四件——manager 的 context configuration（pinned head）/
  `eviction_index.py:L31`（_TIER_CAP = 10）/ `history.py:L57`（write-through，
  fts5 porter unicode61）/ `cap_middleware.py:L38`（token_cap: int = 3000）——
  详见 tutorial_L1 §8/§14（sha256 + 行号级锚点全表）。L2 已核验锚点（2026-08-08）：
  `coach/profile/SOUL.md` L5-L69（七原则，#5 L53 / #7 L69）/
  `coach/profile/skills/k-plus-one/SKILL.md` L40-49（Examiner-B 五检查 + regen>2）、
  L84-87（mastery 更新规则）/ `coach/profile/skills/feynman-check/SKILL.md`
  L36-56（gap 四分类）、L97-101（mastery 四 band）/ `cap_middleware.py`
  L63-68（degradation）、L106（keep 公式）、L110-117（pointer 格式）——
  详见 tutorial_L2 §10（锚点全表）与 §14（sha256 + 确定性记录）。
  L3 已核验锚点（2026-08-14 live 抓取，行号以抓取日为准，运行时 live 重推导
  assert>0）：arch 三源 sha256[:8] = builder `abb6c3fc` / registry `9b59216a` /
  store `2f529ecd`——`src/qwenpaw/runtime/builder.py` L94（Toolkit(tools=…,
  skills_or_loaders=skill_dirs)）、L97-121（_resolve_skill_loader_dirs 准入门）、
  L116-120（not-injected 原样日志，字符串 L117）/ `src/qwenpaw/agents/skill_system/
  registry.py` L1186-1201（resolve_effective_skills）、L1197（channel test）/
  `src/qwenpaw/agents/skill_system/store.py` L65-76（get_workspace_skills_dir）、
  L73（legacy skill/ 原地 rename）——详见 tutorial_L3 §9（锚点表 + 五条取舍分析）。
- 概念延伸：可靠性 / 事务化执行见轨道 04「可靠性专题」
