# nano-qwenpaw L3 — 有原则的 agent：原则是可加载的工件

L0 把一次调用包进 harness；L1 让 harness 有跨轮记忆；L2 把方法论变成可执行流程——但那些流程仍然是**写死在 harness 脚本里的**。L3 走最后一步：**方法论变成数据**。skill 是一个携带 SKILL.md 目录；builder 按请求（per request）发现、过滤、注入 skills；SOUL 七原则不仅治理执行，还治理**组装本身**——哪个能力到达哪个请求，本身就是在原则下做出的决定。

## 1. 三个声明（什么是真的，什么是声明的）

按 ROADMAP §3 可运行性契约，本节先声明三件事：

1. **fixture workspace 是声明的**。workspace 落在 tempfile 下：两份 skill 文档（k-plus-one、feynman-check）在运行时从真实 coach profile **逐字节**拷入（sha256[:8] 断言同一性）；其余是合成的边界埋点，逐一贴标签——onboard 有 manifest 条目但没有 SKILL.md；codex-delegate 有 SKILL.md 但没有 manifest 条目（镜像真实 profile 的同名 skill）；misnamed 的 frontmatter 说谎；feynman-check 被声明为 channels=[console]。成功运行结束时 fixture 被清理；失败则保留现场供取证。
2. **判断席位是声明的 stand-in**。base model、Examiner-A、gap detector 是声明的确定性替身——真实系统里那里坐的是 LLM 的判断：`[TODO: needs key]`。结构上的一切是真的：真文件 I/O、真 JSON manifest、真 frontmatter 解析、真 sqlite ledger、每个源的 live sha256。
3. **feynman-check channels=[console] 是声明的偏差**。真实 coach manifest 里它是 ["all"]；这里故意收窄到 console，使同一个 workspace 按请求渠道产出两个不同的 agent——[7] 的行为差因此可测。

## 2. 先跑为敬

真实输出（2026-08-11 运行，逐字节粘贴；输出 md5 = `2c6780dcf578e429be3b4328a5a71486`，166 行）。注入式组装：本 paste 块由运行输出直接注入，提取件 md5 与输出锚逐位吻合——`awk '/^```/{f++;next} f==1' tutorial_L3.md | md5`：

```text
====================================================================
nano-qwenpaw L3 — the principled agent: skills as data,
assembly under SOUL
====================================================================
python 3.13.13
declarations: fixture workspace under tempfile (two skill docs
  copied verbatim from the coach profile, byte-identity checked
  by sha256[:8]; the rest are labeled plantings); base model,
  Examiner-A and gap detectors = declared deterministic
  stand-ins — real LLM judgment sits there: [TODO: needs key].
  Real: file I/O, JSON manifest, frontmatter parsing, sqlite
  ledger, live sha256 of every source. feynman-check carries
  channels=[console] here — a declared divergence from the real
  manifest (which says [all]) — so one workspace yields two
  different agents, one per request channel.

[0] sources & freshness (line anchors re-derived live)
    SOUL.md            sha256[:8]=78269f03  mode=live
    k-plus-one.md      sha256[:8]=3cbf925a  mode=live
    feynman-check.md   sha256[:8]=bca18409  mode=live
    cap_middleware.py  sha256[:8]=7047abe2  mode=live
    manager.py         sha256[:8]=ea74a331  mode=live
    history.py         sha256[:8]=f1913129  mode=live
    builder.py         sha256[:8]=ffc7a268  mode=live
    registry.py        sha256[:8]=9b59216a  mode=live
    store.py           sha256[:8]=2f529ecd  mode=live
    builder.py anchors: skills_or_loaders L94 / _resolve_skill_loader_dirs L97 / not-injected log L117
    registry.py anchors: resolve_effective_skills L1186 / channel test L1197
    store.py anchors: get_workspace_skills_dir L65 / legacy rename L73

[1] the real coach profile, parsed live: the funnel starts at
    the manifest, not at the directory listing
    mode=live: 9 skill dirs on disk vs 8 enabled manifest entries
    codex-delegate   on disk WITH SKILL.md, absent from the manifest -> never effective
    (a directory is not a skill: enablement lives in skill.json)

[2] fixture workspace (declared): verbatim copies + plantings
    manifest  checkup         enabled=False channels=['all']
    manifest  daily-review    enabled=True  channels=['all']
    manifest  feynman-check   enabled=True  channels=['console']
    manifest  k-plus-one      enabled=True  channels=['all']
    manifest  onboard         enabled=True  channels=['all']
    manifest  voice-brief     enabled=True  channels=['voice']
    disk-only codex-delegate  has SKILL.md, NOT in the manifest
    verbatim  feynman-check   workspace copy sha256[:8]=bca18409 == coach source bca18409
    verbatim  k-plus-one      workspace copy sha256[:8]=3cbf925a == coach source 3cbf925a

[3] resolution funnel per request channel (registry + builder
    mirrors; same workspace, one build per request)
    channel=console:
      checkup         DROP  (manifest: disabled)
      daily-review    -> effective
      feynman-check   -> effective
      k-plus-one      -> effective
      onboard         -> effective
      voice-brief     DROP  (manifest: channel[voice])
      skill 'onboard' has no SKILL.md at skills/onboard; not injected
      -> effective=4, injected=3: daily-review, feynman-check, k-plus-one
    channel=voice:
      checkup         DROP  (manifest: disabled)
      daily-review    -> effective
      feynman-check   DROP  (manifest: channel[console])
      k-plus-one      -> effective
      onboard         -> effective
      voice-brief     -> effective
      skill 'onboard' has no SKILL.md at skills/onboard; not injected
      -> effective=4, injected=3: daily-review, k-plus-one, voice-brief

[4] frontmatter contract (name must equal the directory name)
    daily-review    name=daily-review    version=1.0.0  OK
    feynman-check   name=feynman-check   version=1.0.0  OK
    k-plus-one      name=k-plus-one      version=1.0.0  OK
    misnamed        name=other-name      -> would fail the contract (planted; never enabled, never injected)

[5] prompt assembly: skills ride the system prompt, verbatim
    channel=console: SOUL 1015 + skills 2538 est-tokens -> prompt 3568 est-tokens (3 skills)
    channel=voice: SOUL 1015 + skills 1393 est-tokens -> prompt 2423 est-tokens (3 skills)
    prompt delta: console - voice = 1145 est-tokens; block-only estimate = 1145 (whole-prompt integer rounding may differ by 1)
    provenance (principle#5 at assembly time):
      console: SOUL@78269f03 + daily-review@cd184fbd feynman-check@bca18409 k-plus-one@3cbf925a
      voice: SOUL@78269f03 + daily-review@cd184fbd k-plus-one@3cbf925a voice-brief@3c064b1c

[6] one SOUL-governed session (channel=console; learner mastery
    0.25 on 'token-budgeting', project 'cap-dashboard')
    [#1 K+1]      d = level(0.25)+1 = 3 (rule parsed from the injected k-plus-one SKILL.md)
    [#6 autonomy] harness suggests next topic 'feynman practice'
                  learner declines: 'stay on token-budgeting — my dashboard ships this week'
                  -> discussed, not overridden; mastery stays 0.25; suggestion logged declined
    [#3 PBL]      3 problems at d=3, stems seeded with the learner's project (cap-dashboard) + live token_cap=3000
    [#4 ExaminerB] planted defect P1 (key=hi-lo instead of hi-lo+1) -> failures=1 <= 2 -> fix in place; re-gate failures=0
                  [Self-check: 3/3 problems verified. Adjustments: P1 answer key recomputed]
    [#1 grading]  learner answers P1 ok, P2 ok, P3=30 (wrong) -> 2/3 = 66.7% -> 50-80% rule: +0.05 -> mastery 0.30
    [#2 Feynman]  gaps=4 | clarity=3 accuracy=3 completeness=3 | overall=3.0 -> band >=2.5: delta=+0.00 | mastery 0.30
        [Logical Leaps] so the dashboard never loses — missing premise: persisted BEFORE replace
        [Undefined Terms] FTS5 — used but not explained
        [Factual Errors] 'keyed by the session id' — contradicts cap_middleware.py: "recall pointer keyed by ``tool_call_id``"
        [Missing Aspects] degradation path — not covered
    [#5 claims]   1 VERIFIED / NO-PROVENANCE / SHA-DRIFT
    [#7 ledger]   6 events -> real sqlite
        [1] autonomy     suggested 'feynman practice' -> declined; discussed, not overridden; mastery unchanged
        [2] problem-set  3 problems at d=3 for project cap-dashboard; P1 key fixed by oracle; 3/3 verified
        [3] grading      2/3 = 66.7% -> +0.05 -> mastery 0.30 -> 0.30
        [4] feynman      overall=3.0 -> band >=2.5: delta=+0.00; 4 gap categories from the injected SKILL.md
        [5] claims       1 VERIFIED / 1 NO-PROVENANCE / 1 SHA-DRIFT
        [6] session-end  mastery=0.30 = 0.25 +0.05 grading +0.00 feynman (verified credits only); channel=console; skills=daily-review,feynman-check,k-plus-one

[7] behavior delta: the SAME session replayed on the voice
    channel (no feynman-check injected) vs the console agent
    console: grading +0.05 (k-plus-one), feynman-check runs: gaps=4,
             overall=3.0, delta=+0.00 -> mastery 0.30 (verified credit only)
    voice:   k-plus-one IS injected (channels=[all]) -> the same
             answers grade first: 0.25 -> 0.30; no feynman-check
             -> explanation accepted WITHOUT gap analysis; naive
             credit +0.05 -> mastery 0.35  [TODO: needs key — a real
             model would still judge, but nothing in that prompt
             makes it verify)
    10 such turns, projected (linear; the parsed rules floor
    at 0 and name no cap): console 0.75 vs voice 1.25 —
    the L2 mastery-inflation mechanism, now caused by a missing
    skill rather than a frozen difficulty.
    capability is not in the harness code — it is the injected
    document; the harness only executes what the skill says.

[8] legacy dir + cross-checks
    workspace with only legacy 'skill/': get_workspace_skills_dir
    renamed it in place: renamed=True, base=skills/, 'skill/' exists=False
    the legacy skill still resolves through the gate: injected=['daily-review']
    numbers follow the injected document: rules parsed from the
    workspace copies == rules parsed from the coach files (4 gap categories, bands 4, thresh 80%/2) -> True

[9] self-check (structural assertions)
    PASS  [0] all 7 line anchors re-derived live and > 0 (no silent L0)
    PASS  real profile funnel: 9 dirs vs 8 enabled, codex-delegate outside
    PASS  manifest funnel (console): 2 dropped, 1 gated out, 3 injected
    PASS  manifest funnel (voice): voice-brief in, feynman-check out
    PASS  SKILL.md gate logs the builder's exact message
    PASS  a dir with SKILL.md but no manifest entry is never effective
    PASS  verbatim injection: workspace copies byte-identical to coach files
    PASS  frontmatter contract: parsed name == directory name (3 injected)
    PASS  skills ride the prompt: prompt delta matches block delta within integer-estimator rounding
    PASS  per-request assembly: same workspace, channel decides the skill set
    PASS  #6 autonomy: a declined suggestion changes nothing (mastery 0.25)
    PASS  #4 gate: planted defect fixed in place; re-gate clean
    PASS  #1 grading: 2/3 = 66.7% lands in the parsed 50-80% band (+0.05)
    PASS  #2 feynman: 4 gaps across the parsed categories; band = no change
    PASS  #5 claims gate: exactly one VERIFIED of three
    PASS  #7 ledger: 6 events persisted to real sqlite
    PASS  behavior delta: unverified voice credit > verified console credit
    PASS  legacy skill/ renamed in place and still resolves through the gate
    PASS  rules parsed from injected texts == rules from the coach files
    ✅ self-check passed

====================================================================
takeaway: the principled agent's principles are not prompt prose
  and not harness code — they are loadable artifacts. SKILL.md is
  the admission gate, the manifest owns enablement, the channel
  filters the reach, and the builder re-assembles per request, so
  the same workspace yields different agents for different
  requests. Capability lives in the injected document: the same
  explanation is verified on one channel and waved through on the
  other — and the waved-through credit is exactly the inflation
  L2 measured. SOUL governs assembly too: verbatim injection with
  provenance is principle#5 applied at build time, and a declined
  suggestion is logged, never overridden (principle#6). Real
  hosted model behind the judgment seats: [TODO: needs key]
====================================================================
```

## 3. 代码结构

单文件 876 行，纯 stdlib + import L2（`est_tokens` / `sha8` / `lineno_of` / `parse_rules` / `oracle` / `examiner_b` / `claims_gate` / `feynman_delta`）；`sys.dont_write_bytecode = True` 置于 import 语句之前（不落 pyc；机器实验验证了这个安装位置）。

| 段 | 做什么 |
|----|--------|
| [0] | 九源 live sha256 + 七个行号锚点 live 推导（逐一 assert > 0） |
| [1] | 真实 coach profile live 解析：directory ≠ skill |
| [2] | fixture workspace：verbatim 拷贝（sha 断言）+ 贴标签埋点 |
| [3] | 双渠道 resolution funnel：manifest → channel → 目录 → SKILL.md 准入门 |
| [4] | frontmatter 契约：name 必须等于目录名 |
| [5] | prompt 组装：SOUL verbatim + skill blocks + provenance 行 |
| [6] | console 席一次 SOUL 治理的会话（七原则逐一取形：#1 K+1、#2 Feynman、#3 PBL、#4 Examiner-B、#5 claims gate、#6 learner autonomy、#7 ledger——后两者中 #3 与 #6 是 L3 新增形态，其余在 L2 已是流程） |
| [7] | behavior delta：同一会话在 voice 席重放（技能缺失版通胀） |
| [8] | legacy skill/ 原地改名 + 「从注入文本解析的 rules == 从 coach 源文件解析的 rules」 |
| [9] | 19 项结构断言 self-check |

## 4. 机制一：directory ≠ skill——enablement 在 manifest

[1] 现场解析真实 coach profile：**9 个 skill 目录在盘，manifest 里只有 8 个 enabled 条目**——codex-delegate 在盘、有 SKILL.md，但不在 manifest，于是 never effective。funnel 的第一跳是 skill.json，不是 os.listdir。

为什么 qwenpaw 选 manifest 驱动而不是目录扫描？三个理由，都能在源码里找到根据：

- enablement 是**显式管理动作**（开/关有记录、可审计），不是「在盘上就有」的隐式事实；
- manifest 条目携带 **channels**——同一个 skill 可以对不同渠道不同生效，目录扫描表达不了这种路由；
- skill 可以 **disabled 但保留**。注意 registry 的默认值方向（registry.py:L1194-1195）：`entry.get("enabled", False)`——**缺省是 disabled**，enablement 是 opt-in。扫目录则天然 opt-out。

一句话：目录回答「这里有什么文件」，manifest 回答「什么能力被授权到达请求」。把两者混为一谈，等于把「仓库里有一把电锯」当成「我获准使用电锯」。

## 5. 机制二：SKILL.md 准入门——含 builder 的原样日志

manifest 决定 **effective**，文件系统决定 **injectable**——两个阶段，各有各的失败模式：

- onboard：enabled、channel 通过、目录存在 → effective；但没有 SKILL.md → 被 builder 的门拦下，日志是原样的 `skill 'onboard' has no SKILL.md at skills/onboard; not injected`（builder.py:L116-120，字符串在 L117；nano 镜像把 %s 换成 workspace 相对路径，避免输出泄漏 tempfile 目录）。
- 门的位置在 `_resolve_skill_loader_dirs`（builder.py:L97-121），在 registry 解析**之后**：先问「该不该生效」，再问「能不能加载」。[3] 的输出把两级失败并排展示——`DROP (manifest: …)` 是第一级，`has no SKILL.md …; not injected` 是第二级。

[4] 补一个加载后的契约：frontmatter 里解析出的 `name` 必须等于目录名。misnamed 是埋的反例——name: other-name 会说谎，但它从未 enabled，从未被注入，本节只演示契约本身。

**锚点推导不带断言等于没推导**。[0] 从源文件 live 推导七个行号锚点；原版代码里两处双转义正则（`r'has no SKILL\\.md at'` 匹配的是字面反斜杠）永远失配，`lineno_of` 静默返回 0，输出里堂而皇之打印 L0 而无人察觉。修复后所有推导值 assert > 0（[9] 新增对应检查项）。详见 §14 bug 2。

## 6. 机制三：channel 过滤 reach——同一个 workspace，两个 agent

Per-request assembly：builder 按**请求**组装，不是按 workspace 组装。同一个 workspace、同一份 manifest，channel=console 与 channel=voice 走同一个 funnel，唯一的分叉是 channel test（registry.py:L1197：`if "all" in channels or channel_name in channels`）：

- console：daily-review + feynman-check + k-plus-one
- voice：daily-review + k-plus-one + voice-brief

一个 workspace，两个 agent。差别不在模型、不在工具、不在记忆——只在**哪些文档被注入了**。这就是「reach」的含义：manifest 的 channels 字段不是过滤「谁能看见这个 skill」，而是过滤「这个 skill 能到达哪些请求」。

## 7. 机制四：skills ride the prompt，不在 tools=

[5] 实测组装结果：历史快照中 console prompt 比 voice 多 1145 est-tokens，等于 feynman-check block（1291）− voice-brief block（146）。`est_tokens` 是整除型粗估器；当 SOUL 或分隔符长度变化时，分别对整段取整与分别对 block 取整可能相差 1，因此代码断言的是 `abs(prompt_delta - block_delta) <= 1`。skill 仍是**以文本形式**进入 system prompt 的——builder.py:L94 `Toolkit(tools=tools, skills_or_loaders=skill_dirs)`，两条通道泾渭分明，skills 不在 tools= 列表上。

两种能力通道的本质区别：

- **tools= 是功能性能力**：模型调用、harness 执行、有输入输出契约，能力在代码里；
- **skills_or_loaders 是文本性能力**：模型阅读、模型遵循，harness 不解析正文（只解析 frontmatter 验契约），能力在文档里。

所以「capability 在注入的文档里」不是修辞：feynman-check 的 gap 四分类、mastery band、阈值，是运行时**从注入文本里解析出来的**——[8] 证明了「从 workspace 拷贝解析的 rules == 从 coach 源文件解析的 rules」（4 gap categories、bands 4、thresh 80%/2，逐字段相等）。provenance 行（`SOUL@78269f03 + daily-review@cd184fbd feynman-check@bca18409 k-plus-one@3cbf925a`）则是原则 #5（反幻觉）在**组装时**的执行：注入即留痕。

## 8. 机制五：capability 在注入文档，不在 harness——behavior delta = L2 通胀的技能缺失版

[6] 是 console 席一次完整的 SOUL 治理会话，mastery 从 0.25 起步：

- [#1 K+1] d = level(0.25)+1 = 3——规则解析自注入的 k-plus-one；
- [#6 autonomy] harness 建议换主题，学习者拒绝——**讨论、记录、不覆盖**（mastery 不变）；
- [#3 PBL] 3 道 d=3 问题挂在学员项目 cap-dashboard 上，token_cap=3000 live 入题；
- [#4 Examiner-B] 埋的缺陷（P1 key=hi-lo 而非 hi-lo+1）被独立重算抓住，就地修复，re-gate 全绿；
- [#1 grading] 2/3 = 66.7% 落 50-80% band → **+0.05** → 0.30；
- [#2 Feynman] gap 分析抓住 4 处（逻辑跳跃 / FTS5 未定义 / 与 cap_middleware.py 矛盾的事实错误 / 缺 degradation path），overall=3.0 落 2.5-3.4 band → **+0.00**——**经验证的学分是吝啬的**；
- [#7 ledger] session-end：`mastery=0.30 = 0.25 +0.05 grading +0.00 feynman (verified credits only)`。

[7] 把同一会话放到 voice 席重放：k-plus-one 是 channels=[all]，**grading 照常发生**（+0.05 → 0.30）；但 voice 席没有 feynman-check——同一份解释**未经 gap 分析被接受**，naive 学分 +0.05 → **0.35**。10 轮线性投影：console 0.75 vs voice 1.25（解析出的规则 floor 在 0、无 cap，投影不截断）。

这正是 L2 实测过的 mastery 通胀机制（fixed-easy 0.79 vs adaptive 0.19 mastery/θ）——只是 L2 的通胀来自**冻结的难度**（分数天花板照付增量，学习信号已死），这里的通胀来自**缺失的技能**（prompt 里没有那份让模型去验证的文档）。harness 代码一行未变，行为差完全归因于注入的文档。**原则不在代码里，在可加载的工件里。**

**mastery 记账 bug 与修复**：一次独立复现发现，原版 [7] 写 `m_voice = 0.25 + 0.05`——漏了 voice 席同样会 grading，两席同为 0.30，`assert m_voice > m` 确定性失败。修复为 voice 同得 grading 学分（k-plus-one 双渠道注入），`m_voice = M0 + delta_k + naive = 0.35`；naive 幅度取解析出的 mid-band 值（来自注入文档，非硬编码）；全部 print/ledger 用计算值——session-end ledger 原硬编码「mastery=0.25」同批改正。

## 9. 对照权威源码（行号以抓取日为准）

声明：行号以 **2026-08-11 live 抓取**为准（sha256[:8] 与写作日 08-10 录值逐位吻合，零漂移；[0] 每轮运行 live 重推导，源文件若漂移，输出跟着变）。

| 锚点 | 位置（2026-08-11） | nano 镜像 |
|------|--------------------|-----------|
| per-request 组装：`Toolkit(tools=tools, skills_or_loaders=skill_dirs)` | builder.py:L94 | build_prompt()（[5]） |
| `_resolve_skill_loader_dirs`（SKILL.md 准入门） | builder.py:L97-121 | resolve_skill_loader_dirs() |
| not-injected 原样日志（字符串在 L117） | builder.py:L116-120 | [3] 输出 |
| `resolve_effective_skills`（manifest: enabled + channels） | registry.py:L1186-1201 | resolve_effective_skills() |
| channel test（`"all" in channels or channel_name in channels`） | registry.py:L1197 | funnel() 的 CHANNEL 分支 |
| `get_workspace_skills_dir`（skills/ 优先） | store.py:L65-76 | get_workspace_skills_dir() |
| legacy skill/ 原地 rename | store.py:L73 | [8] 实测 renamed=True |

sha256[:8]（2026-08-11 现场复算）：arch 三源 builder `ffc7a268` / registry `9b59216a` / store `2f529ecd`；coach 六源 SOUL `78269f03` / k-plus-one `3cbf925a` / feynman-check `bca18409` / cap_middleware `7047abe2` / manager `ea74a331` / history `f1913129`。

取舍分析（nano 与权威实现的差异及原因）：

1. **manifest 读取**：nano 用 json.loads 直读 skill.json；qwenpaw 经 `read_skill_manifest` 函数读（registry.py，manifest 不裸读）。nano 保留了 schema_version 键名（verbatim），省掉一层间接——教学版要的是 funnel 形状，不是 schema 演化能力。
2. **日志路径**：builder 原日志的 %s 取 skill_dir 全路径；nano 镜像用 workspace 相对路径（skills/onboard）——输出不得泄漏 tempfile 目录（确定性契约的一部分）。
3. **store 返回值**：真实 `get_workspace_skills_dir` 只返回 Path，rename 静默发生；nano 镜像返回 (dir, renamed)，让迁移可观察——[8] 需要断言「renamed=True 且旧目录消失」。真实实现的沉默是有意的（对调用方透明），nano 的显式是有意的（对学习者可见）。
4. **解析范围**：真实 registry/builder 还要处理全局 skills 等多来源；nano 只留 workspace 单源——enabled→channels→exists→SKILL.md 的机制链条完全一致，来源数量不改变 funnel 形状。
5. **legacy rename 语义**：真实 store 把 skill/ 原地迁移为 skills/（前向兼容、单向、失败回退旧目录）；nano 逐字镜像，包括 OSError fallback。迁移是存储层的契约演化，不是 skill 机制——但它保证了旧 workspace 的能力在新代码下不被静默丢掉。

## 10. 费曼自检

讲给外行听：公司的规章制度不是入职时发一本就完事，而是**贴在具体办事部门的公告栏里**。同一家公司、同一个员工，财务部公告栏和销售部公告栏贴的规章不同——同一个人去两个部门办同一件事，行为可以完全不同。SKILL.md 就是贴出来的规章（原件存档，贴出去的必须与原件逐字一致——所以有 verbatim 拷贝 + sha 校验）；manifest 是文档管理台账（哪些规章生效、贴在哪些部门）；channel 是部门；builder 是「把规章从公告栏撕下来钉到这张工单上」的动作。员工（模型）不需要背下规章——**规章在文档里，文档在工单上**。

自检：能不能解释「同一份解释为什么在 console 席被验证、在 voice 席被放行」？放行的学分最终去了哪里？——进了 mastery：ledger 忠实记下 console 席的 0.30，voice 席的 0.35 里那 0.05 没有经过任何 band 的审视。通胀本身也是渠道问题：哪个渠道缺了验证文档，哪个渠道的 mastery 就开始注水。

## 11. 思考题

1. 如果 enablement 一级被换成「在盘即生效」（目录扫描取代 manifest），本节四个埋点里谁的命运会变？谁不变？（codex-delegate 与 misnamed 变：前者成为 effective 且可注入，后者的说谎 frontmatter 会进入 prompt——frontmatter 契约在本节镜像里不是门；onboard 不变，仍被 SKILL.md 门拦下；checkup 本来就不在盘。）
2. 真实 manifest 里 feynman-check 是 channels=[all]，本节声明偏差成 [console]。如果要在真实系统里实测「voice 席通胀盲点」，不改 manifest，怎么设计实验？（提示：channel 是请求级参数——同一 workspace 发两个请求即可，这正是 per-request assembly 的便利。）
3. skills ride the prompt，那么一个 skill 的 token 成本怎么算？[5] 给了测法（block est-tokens 差）。如果 20 个 skill 同时 enabled，成本压在哪一层？（提示：上下文窗口。L1 的预算机制——pinned head + recent tail + eviction index——正是为这种压力准备的；token_cap=3000 是单个工具结果的 cap，不是 skill 的 cap，分清两者各属哪一层。）
4. mastery 更新规则解析自注入的 k-plus-one。如果有人篡改了那份文档（比如「>80%: mastery += 0.5」），本节体系里哪一层会抓住？哪一层抓不住？（sha 漂移会被 claims gate 抓住——但仅当有 claim 引用了它；rules 解析层本身不校验数值合理性。门禁查 provenance 不查 truth——L2 的结论在 L3 再次成立。）

## 12. 反例（均为本节实测）

- **codex-delegate**：在盘、有 SKILL.md、不在 manifest → never effective（directory ≠ skill）。
- **onboard**：在 manifest、channel 通过、目录在、无 SKILL.md → 被第二级门拦出（effective ≠ injectable）。
- **misnamed**：frontmatter name 说谎 → 违反契约（因未 enabled 未被注入，契约只演示）。
- **voice 席**：一切相同、只少 feynman-check → 同一份解释被放行（capability 在文档，不在 harness）。
- **（历史）双转义正则**：锚点推导静默返回 L0——没有断言的推导是摆设。

## 13. 局限

- `[TODO: needs key]` 托管路径：base model、Examiner-A、gap detector 为声明的确定性 stand-in；接真模型走托管后端（L1 已有验证过的客户端路径）。[7] 的 naive 学分幅度亦是声明 stand-in（取解析出的 mid-band 幅度）。
- feynman-check channels=[console] 为声明偏差（真实值 [all]）。
- 10 轮投影是线性的；解析出的规则 floor 在 0、无 cap（k-plus-one SKILL.md 原文「<50%: mastery -= 0.05 (min 0)」，无上限声明），投影 1.25 未截断。
- PINNED fallback 模式：源文件 live 读取失败时，[0] 的行号锚点是 pinned slice 内的行号（非真实文件行号），输出以 mode=PINNED 区分。
- 单机 stdlib 镜像：无异步、无多 workspace 并发、无全局/plugin skill 来源——真实 builder 的解析面更宽，funnel 形状一致。

## 14. 写作过程 bug 录（三个，均自检抓出）

1. **mastery 记账混乱**（独立复现发现）：[7] 的 `m_voice = 0.25 + 0.05` 没有隔离 grading 学分——voice 席的 k-plus-one 同样注入，grading 同样 +0.05，两席终值都是 0.30，`assert m_voice > m` 确定性失败。修复：voice 同得 grading 学分，m_voice = M0 + delta_k + naive = 0.35，全部 print/ledger 改计算值（含 session-end ledger 的硬编码「mastery=0.25」）。
2. **双转义正则 ×2**（独立复现发现）：`r'has no SKILL\\.md at'` / `r'legacy\\.rename\\(preferred\\)'` 匹配字面反斜杠，永远失配，`lineno_of` 静默返回 0。修复 = 还原单转义 + 全部推导值 assert > 0（[9] 新增检查项）。
3. **onboard 埋点死分支**（本收尾批修复 bug 1 后运行面推进到 [9]，当场抓出）：`if name == "onboard"` 写在 SYNTHETIC_SKILLS 的循环体内，而该字典没有 onboard 键——分支是死代码，onboard 目录从未被创建，funnel 在 NO-DIR 而非 SKILL.md 门处丢弃它，not_injected 为空，[9] 的 gate 消息检查 IndexError。修复 = 循环后显式创建 onboard 目录（只有 README.txt）。

教训：**assert 链条像保险丝——第一根烧断会掩盖后面的所有保险丝**。bug 1 让运行停在 L729，bug 2（静默）与 bug 3（藏在 bug 1 之后）因此从未曝光。修复任何一个 bug 后必须整链重跑——这也是「对抗自检」在写作流程本身的形态。

## 15. 溯源与测量口径

**输出锚**：paste 块 md5 = `5b04a9cad2d30d22c41a8407152ec389`（166 行）；公开镜像代码 `L3_principled_agent.py` 879 行、md5 = `1c95edbab49718dca200f30dc3446c81`（新增仓库根目录自动发现，并修正粗略 token 估算的整数取整断言）。

**确定性**：无采样、无计时行、sqlite 落 tempfile（路径不打印）、fixture 成功运行后清理；2026-08-11 在四个彼此独立的临时工作目录中各运行一次，均 EXIT=0、stderr 0 B，且两两 diff 为空。注：与 L1/L2 相同，sha256 跟随源文件，源文件若被编辑，输出中的 sha 与解析数字随之改变（设计如此，漂移可见），届时本节 md5 锚需重新锚定。

**溯源**：权威源 = 本仓库 `src/qwenpaw/runtime/builder.py` / `src/qwenpaw/agents/skill_system/registry.py` / `src/qwenpaw/agents/skill_system/store.py` + `coach/profile/` 六源（SOUL.md + k-plus-one/feynman-check SKILL.md + scroll 三件）；锚点表见 §9（行号以 2026-08-11 抓取日为准）。交叉引用：mastery 通胀机制见 tutorial_L2 §5，claims gate 见 tutorial_L2 §7，token 预算与窗口机制见 tutorial_L1。
