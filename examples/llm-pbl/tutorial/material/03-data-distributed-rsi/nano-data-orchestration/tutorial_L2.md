# nano-data-orchestration L2 — 并行 executor + 资源池 + trigger rules + heartbeat/收养 + 并发键 + Agentic 自愈（可运行的本质模拟）

> L0 裸出状态机内核，L1 把「真实进程 / 持久化 / wall-clock」请回来。L2 回答 L1 §11 边界表留下的每一笔债——
> 做法不是功能罗列，而是把 Airflow / Dagster / GitHub Actions / GitLab CI 四个系统的**机制本质**裸成可跑代码，
> 并逐条对照一手源码说明「它那样选、我这样模拟」。本级还包含一个真实发生的反例教材：第一版代码在
> run [7] 跌进了「进程存活探测」的坑（zombie 误判），复现实验稳定得到 EXIT=1——「为什么 Airflow 从不靠父子关系
> 判断任务死活」由此从一句引文变成一个用代价换来的教训（§9）。

---

## §1 K+1：L1 留下了哪三笔债

L1 把状态机搬到了真实 subprocess 上（exit code 分类 / state.json 落盘 / kill -9 进程组续跑 / wall-clock 退避 / 幂等登场），但它在 §11 边界表里白纸黑字留下了债，每一笔都是 L2 的课题：

| L1 的债（§11 录值） | 为什么 L1 不做 | L2 怎么还 |
|---------------------|----------------|-----------|
| 孤儿收养（调度器死、子进程活） | exit code 已随宿主丢失，「完成/在跑」不可区分；L1 对此只能 raise | heartbeat / 结果通道：**liveness 与结果都不走父子关系**（§5/§9，Airflow adopt_or_reset 源码对照） |
| RUNNING 连续形态 liveness 检测 | L1 故障模型下崩溃必伴重启，重启时点状检测已完备 | 周期扫描 + heartbeat 超时（§5，Airflow purge 定时器对照） |
| 真实并行 / 资源池 / 优先级 / trigger rules / 并发键 / Agentic 自愈 | 串行规则 C 是 L1 的确定性选择 | §3–§8 逐面对照 Airflow/Dagster/GHA/GitLab 源码 |

K+1 边界声明：runs [1]–[6] 是**逻辑时钟下的本质模拟**——时长/失败日程/心跳模式都是显式实验设计（fixture 声明在代码 `sim_plan`），确定、可复现、字节级可锚；run [7] 是**真实 subprocess 并行 + 真实 kill + 真实收养**，墙钟不确定量落 elapsed 掩码行。哪些是模拟、哪些是真货、差距在哪，§11 逐条诚实声明。

---

## §2 先跑一遍

**本课程可运行性契约声明**：L2 允许「可运行的本质模拟 + 显式注明」。本文件零依赖（纯标准库），CPU 约 5s 跑完；runs [1]–[6] 为显式注明的机制模拟，run [7] 为真进程锚（模拟核心本身全部可运行）。

```bash
$ python3 L2_parallel_executor_heartbeat_and_self_heal.py
```

完整输出如下（elapsed 掩码行已按口径 `sed '/^[[:space:]]*elapsed/d'` 删除——墙钟时长与心跳重叠秒数是不确定量，不进 check 路径；掩码口径与双跑锚点见 §17。以下各节的输出块均从此同一次运行中截取，逐字子序列）：

```text
== nano-data-orchestration L2: 并行 executor + 资源池 + trigger rules + heartbeat/收养 + 并发键 + Agentic 自愈 ==
  （runs [1]–[6] = 逻辑时钟本质模拟，确定性可锚；run [7] = 真实 subprocess 并行 + 真实 kill + 真实孤儿收养）

[1] 跨级锚：L1 fixture 在 L2 状态机上退化运行（parallelism=1）——终态 digest 必须复现 L1 录值
  [seq 01] t=0.0  ingest_crm         -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  ingest_web         -> QUEUED           入 executor 队列（priority=0）
  [seq 03] t=0.0  unit_tests         -> QUEUED           入 executor 队列（priority=0）
  [seq 04] t=0.0  ingest_crm         -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=1.0  ingest_crm         -> RETRYING         attempt 1 exit=75 transient（） —— 计划退避 1 tick
  [seq 06] t=1.0  ingest_web         -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 07] t=2.0  ingest_web         -> RETRYING         attempt 1 exit=75 transient（） —— 计划退避 1 tick
  [seq 08] t=2.0  ingest_crm         -> QUEUED           入 executor 队列（priority=0）
  [seq 09] t=2.0  ingest_crm         -> RUNNING          attempt 2 启动（queued_by=sched#1）
  [seq 10] t=3.0  ingest_crm         -> SUCCESS          attempt 2 完成  <- 重试救回
  [seq 11] t=3.0  gate_crm           -> QUEUED           入 executor 队列（priority=0）
  [seq 12] t=3.0  ingest_web         -> QUEUED           入 executor 队列（priority=0）
  [seq 13] t=3.0  gate_crm           -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 14] t=4.0  gate_crm           -> SUCCESS          attempt 1 完成
  [seq 15] t=4.0  ingest_web         -> RUNNING          attempt 2 启动（queued_by=sched#1）
  [seq 16] t=5.0  ingest_web         -> RETRYING         attempt 2 exit=75 transient（） —— 计划退避 2 tick
  [seq 17] t=5.0  unit_tests         -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 18] t=6.0  unit_tests         -> SUCCESS          attempt 1 完成
  [seq 19] t=6.0  build_curated      -> QUEUED           入 executor 队列（priority=0）
  [seq 20] t=6.0  build_curated      -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 21] t=7.0  ingest_web         -> QUEUED           入 executor 队列（priority=0）
  [seq 22] t=8.0  build_curated      -> SUCCESS          attempt 1 完成
  [seq 23] t=8.0  deploy             -> QUEUED           入 executor 队列（priority=0）
  [seq 24] t=8.0  publish_report     -> QUEUED           入 executor 队列（priority=0）
  [seq 25] t=8.0  deploy             -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 26] t=9.0  deploy             -> SUCCESS          attempt 1 完成
  [seq 27] t=9.0  ingest_web         -> RUNNING          attempt 3 启动（queued_by=sched#1）
  [seq 28] t=10.0 ingest_web         -> FAILED           attempt 3 —— 重试上限 (2) 耗尽，止损
  [seq 29] t=10.0 gate_web           -> UPSTREAM_FAILED  trigger_rule=all_success: 上游失败——依赖是承诺，不在坏数据上跑
  [seq 30] t=10.0 normalize_web      -> UPSTREAM_FAILED  trigger_rule=all_success: 上游失败——依赖是承诺，不在坏数据上跑
  [seq 31] t=10.0 publish_report     -> FAILED(deny)     capability missing (default-deny): ['metrics_write'] —— 0 attempt 0 成本
  终态向量: SUCCESS=['build_curated', 'deploy', 'gate_crm', 'ingest_crm', 'unit_tests']
            FAILED=['ingest_web', 'publish_report']  UPSTREAM_FAILED=['gate_web', 'normalize_web']
  成本账本: 总 9 = 有效 5 + 重试救回 1 + 浪费 3 + 崩溃税 0
  [check 01] PASS  跨级锚: 终态向量 digest == L1 录值 ac4a0b3ac09bf47b（L1 state_digest 公式逐字同款）
  [check 02] PASS  成本恒等式 9 = 5 + 1 + 3 + 0（复现 L0 check 13 / L1 check 08）
  [check 03] PASS  default-deny 在并行状态机下不变: publish_report attempts==0（QUEUED 之前就拒）
  [check 04] PASS  事件流 seq 单调无 gap；终态 5/2/2 与 L0/L1 逐字一致
  （串行调和用时 10 tick——[2] 的并行必须更快且收敛点不变）

[2] 并行 executor + pool + priority：同一 fixture，parallelism=3，pool 'src' 槽位=1，unit_tests 优先级最高
  [seq 01] t=0.0  unit_tests         -> QUEUED           入 executor 队列（priority=0，pool=src）
  [seq 02] t=0.0  ingest_crm         -> RUNNABLE         pool 'src' 无空槽（occupied=1）——等待
  [seq 03] t=0.0  ingest_web         -> RUNNABLE         pool 'src' 无空槽（occupied=1）——等待
  [seq 04] t=0.0  unit_tests         -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=1.0  unit_tests         -> SUCCESS          attempt 1 完成
  [seq 06] t=1.0  ingest_crm         -> QUEUED           入 executor 队列（priority=5，pool=src）
  [seq 07] t=1.0  ingest_crm         -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 08] t=2.0  ingest_crm         -> RETRYING         attempt 1 exit=75 transient（） —— 计划退避 1 tick
  [seq 09] t=2.0  ingest_web         -> QUEUED           入 executor 队列（priority=5，pool=src）
  [seq 10] t=2.0  ingest_web         -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 11] t=3.0  ingest_web         -> RETRYING         attempt 1 exit=75 transient（） —— 计划退避 1 tick
  [seq 12] t=3.0  ingest_crm         -> QUEUED           入 executor 队列（priority=5，pool=src）
  [seq 13] t=3.0  ingest_crm         -> RUNNING          attempt 2 启动（queued_by=sched#1）
  [seq 14] t=4.0  ingest_crm         -> SUCCESS          attempt 2 完成  <- 重试救回
  [seq 15] t=4.0  gate_crm           -> QUEUED           入 executor 队列（priority=0）
  [seq 16] t=4.0  ingest_web         -> QUEUED           入 executor 队列（priority=5，pool=src）
  [seq 17] t=4.0  gate_crm           -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 18] t=4.0  ingest_web         -> RUNNING          attempt 2 启动（queued_by=sched#1）
  [seq 19] t=5.0  gate_crm           -> SUCCESS          attempt 1 完成
  [seq 20] t=5.0  ingest_web         -> RETRYING         attempt 2 exit=75 transient（） —— 计划退避 2 tick
  [seq 21] t=5.0  build_curated      -> QUEUED           入 executor 队列（priority=0）
  [seq 22] t=5.0  build_curated      -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 23] t=7.0  build_curated      -> SUCCESS          attempt 1 完成
  [seq 24] t=7.0  deploy             -> QUEUED           入 executor 队列（priority=0）
  [seq 25] t=7.0  publish_report     -> QUEUED           入 executor 队列（priority=0）
  [seq 26] t=7.0  ingest_web         -> QUEUED           入 executor 队列（priority=5，pool=src）
  [seq 27] t=7.0  deploy             -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 28] t=7.0  publish_report     -> FAILED(deny)     capability missing (default-deny): ['metrics_write'] —— 0 attempt 0 成本
  [seq 29] t=7.0  ingest_web         -> RUNNING          attempt 3 启动（queued_by=sched#1）
  [seq 30] t=8.0  deploy             -> SUCCESS          attempt 1 完成
  [seq 31] t=8.0  ingest_web         -> FAILED           attempt 3 —— 重试上限 (2) 耗尽，止损
  [seq 32] t=8.0  gate_web           -> UPSTREAM_FAILED  trigger_rule=all_success: 上游失败——依赖是承诺，不在坏数据上跑
  [seq 33] t=8.0  normalize_web      -> UPSTREAM_FAILED  trigger_rule=all_success: 上游失败——依赖是承诺，不在坏数据上跑
  pool 'src' 占用峰值 = 1（slots=1）；首个出队任务 = unit_tests（priority=0 先于 priority=5）
  [check 05] PASS  并行不改变收敛点: 终态 digest == [1]（调度策略是路径，不是语义）
  [check 06] PASS  pool 不变量: 'src' 占用峰值 ≤ slots（occupied = QUEUED+RUNNING，对照 Pool.occupied_slots）
  [check 07] PASS  priority 出队顺序: 权重小者先出（nano 约定；Airflow sorted by priority_weight, reverse=False 的排序事实）
  [check 08] PASS  attempts 向量逐字不变 + 本 fixture 临界路径（ingest_web 退避链）两模式同长，故 makespan 不快于串行
  [seq 01] t=0.0  x1                 -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  x2                 -> QUEUED           入 executor 队列（priority=0）
  [seq 03] t=0.0  x3                 -> QUEUED           入 executor 队列（priority=0）
  [seq 04] t=0.0  x1                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=2.0  x1                 -> SUCCESS          attempt 1 完成
  [seq 06] t=2.0  x2                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 07] t=4.0  x2                 -> SUCCESS          attempt 1 完成
  [seq 08] t=4.0  x3                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 09] t=6.0  x3                 -> SUCCESS          attempt 1 完成
  [seq 01] t=0.0  x1                 -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  x2                 -> QUEUED           入 executor 队列（priority=0）
  [seq 03] t=0.0  x3                 -> QUEUED           入 executor 队列（priority=0）
  [seq 04] t=0.0  x1                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=0.0  x2                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 06] t=0.0  x3                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 07] t=2.0  x1                 -> SUCCESS          attempt 1 完成
  [seq 08] t=2.0  x2                 -> SUCCESS          attempt 1 完成
  [seq 09] t=2.0  x3                 -> SUCCESS          attempt 1 完成
  [check 09] PASS  并行的量化证据（无共享资源探针）: 3 独立任务 × 2 tick，串行 6 tick -> 并行 2 tick（3× 加速）

[3] trigger rules：上游终态向量 -> 本任务命运（语义对照 Airflow TriggerRuleDep 分派表）
  [seq 01] t=0.0  bad                -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  branch             -> QUEUED           入 executor 队列（priority=0）
  [seq 03] t=0.0  join_always        -> QUEUED           入 executor 队列（priority=0）
  [seq 04] t=0.0  bad                -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=0.0  branch             -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 06] t=0.0  join_always        -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 07] t=1.0  bad                -> FAILED           attempt 1 permanent（校验失败）—— 立即止损，不重试
  [seq 08] t=1.0  branch             -> SUCCESS          attempt 1 完成
  [seq 09] t=1.0  join_always        -> SUCCESS          attempt 1 完成
  [seq 10] t=1.0  join_nfmos         -> UPSTREAM_FAILED  trigger_rule=none_failed_min_one_success: 上游失败——依赖是承诺，不在坏数据上跑
  [seq 11] t=1.0  right              -> SKIPPED          branch 未选中（branch 选择另一分支）
  [seq 12] t=1.0  left               -> QUEUED           入 executor 队列（priority=0）
  [seq 13] t=1.0  left               -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 14] t=2.0  left               -> SUCCESS          attempt 1 完成
  [seq 15] t=2.0  join_all_success   -> SKIPPED          trigger_rule=all_success: skipped 上游传染
  [seq 16] t=2.0  join_all_done      -> QUEUED           入 executor 队列（priority=0）
  [seq 17] t=2.0  join_none_failed   -> QUEUED           入 executor 队列（priority=0）
  [seq 18] t=2.0  join_one_success   -> QUEUED           入 executor 队列（priority=0）
  [seq 19] t=2.0  solo_all_success   -> QUEUED           入 executor 队列（priority=0）
  [seq 20] t=2.0  join_all_done      -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 21] t=2.0  join_none_failed   -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 22] t=2.0  join_one_success   -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 23] t=2.0  solo_all_success   -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 24] t=3.0  join_all_done      -> SUCCESS          attempt 1 完成
  [seq 25] t=3.0  join_none_failed   -> SUCCESS          attempt 1 完成
  [seq 26] t=3.0  join_one_success   -> SUCCESS          attempt 1 完成
  [seq 27] t=3.0  solo_all_success   -> SUCCESS          attempt 1 完成
  bad                  -> FAILED
  branch               -> SUCCESS
  join_all_done        -> SUCCESS
  join_all_success     -> SKIPPED
  join_always          -> SUCCESS
  join_nfmos           -> UPSTREAM_FAILED
  join_none_failed     -> SUCCESS
  join_one_success     -> SUCCESS
  left                 -> SUCCESS
  right                -> SKIPPED
  solo_all_success     -> SUCCESS
  [check 10] PASS  ALL_SUCCESS 被 skipped 上游染成 SKIPPED（trigger_rule_dep.py:L433-434；skipped 不算成功调度口径）
  [check 11] PASS  branching: 未选中分支 = SKIPPED（新终态，L0/L1 没有）
  [check 12] PASS  NONE_FAILED 容忍 skipped、ONE_SUCCESS 见好就跑（两者都 SUCCESS）
  [check 13] PASS  ALL_DONE 等全部终态后无条件跑（含 FAILED 上游）；NONE_FAILED_MIN_ONE_SUCCESS 见 failed 即 UPSTREAM_FAILED
  [check 14] PASS  ALWAYS 不等上游（bad 尚未终态时 join_always 已完成）

[4] heartbeat 连续形态：stuck（活着不跳）杀之重试 / 死掉不跳 = zombie 回炉；调度器重启 -> 孤儿收养
  [seq 01] t=0.0  hb_dead            -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  hb_ok              -> QUEUED           入 executor 队列（priority=0）
  [seq 03] t=0.0  hb_stuck           -> QUEUED           入 executor 队列（priority=0）
  [seq 04] t=0.0  hb_dead            -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=0.0  hb_ok              -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 06] t=0.0  hb_stuck           -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 07] t=3.0  hb_ok              -> SUCCESS          attempt 1 完成
  [seq 08] t=4.0  hb_dead            -> RETRYING         attempt 1 被杀（heartbeat 超时进程仍存活 = stuck） —— 计划退避 1 tick
  [seq 09] t=4.0  hb_stuck           -> RETRYING         attempt 1 被杀（heartbeat 超时进程仍存活 = stuck） —— 计划退避 1 tick
  [seq 10] t=5.0  hb_dead            -> QUEUED           入 executor 队列（priority=0）
  [seq 11] t=5.0  hb_stuck           -> QUEUED           入 executor 队列（priority=0）
  [seq 12] t=5.0  hb_dead            -> RUNNING          attempt 2 启动（queued_by=sched#1）
  [seq 13] t=5.0  hb_stuck           -> RUNNING          attempt 2 启动（queued_by=sched#1）
  [seq 14] t=8.0  hb_dead            -> SUCCESS          attempt 2 完成  <- 重试救回
  [seq 15] t=8.0  hb_stuck           -> SUCCESS          attempt 2 完成  <- 重试救回
  [check 15] PASS  stuck 检测: 心跳停于 t=1，超时=2 -> 恰在 t=4 被杀（检测时刻 = last_hb + timeout 后的首个事件点，确定）
  [check 16] PASS  zombie 连续形态: 进程暴毙 + 心跳陈旧 -> 同一时刻识别回炉（L1 点状语义的连续化）
  [check 17] PASS  两者重试后都收敛 SUCCESS；hb_ok 一次通过（心跳正常 = 免打扰）
  --- 调度器崩溃 @ t=2（逻辑模拟）：丢弃 sched#1 内存，sched#2 从状态重建并收养 ---
  [seq 01] t=0.0  long_a             -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  short_b            -> QUEUED           入 executor 队列（priority=0）
  [seq 03] t=0.0  long_a             -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 04] t=0.0  short_b            -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=1.0  short_b            -> SUCCESS          attempt 1 完成
  [seq 01] t=2.0  long_a             -> RUNNING          adopted（sched#2 收养：心跳新鲜 + 进程存活，attempt 1 继续，不重启）
  [seq 02] t=6.0  long_a             -> SUCCESS          attempt 1 完成
  [check 18] PASS  孤儿收养: long_a 被 sched#2 收养——queued_by 换人、attempt 不增（工作不重做）
  [check 19] PASS  收养凭据 = 心跳新鲜 + 进程存活（对照 adoptable_states + try_adopt）；short_b 已终态不受扰动

[5] 并发键 claim/release + 指数步进退避（Dagster）；concurrency group cancel-in-progress（GitHub Actions）
  [seq 01] t=0.0  c1                 -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  c2                 -> QUEUED           入 executor 队列（priority=0）
  [seq 03] t=0.0  c3                 -> RUNNABLE         并发键 'gpu' 无空槽 —— claim 被拒（第 1 次），退避 1.00 tick
  [seq 04] t=0.0  c1                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=0.0  c2                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 06] t=1.0  c3                 -> RUNNABLE         并发键 'gpu' 无空槽 —— claim 被拒（第 2 次），退避 1.10 tick
  [seq 07] t=2.0  c1                 -> SUCCESS          attempt 1 完成
  [seq 08] t=2.0  c2                 -> SUCCESS          attempt 1 完成
  [seq 09] t=2.1  c3                 -> QUEUED           入 executor 队列（priority=0）
  [seq 10] t=2.1  c3                 -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 11] t=4.1  c3                 -> SUCCESS          attempt 1 完成
  键 'gpu' slots=2: c3 被拒 2 次，退避序列 [1.0, 1.1] tick，持有峰值 2
  [check 20] PASS  槽位不变量: 持有峰值 ≤ slot_count（claim/release 账本平衡 claims==releases==3）
  [check 21] PASS  退避是指数步进（Dagster 公式 1+(1.1^n-1)，上限 15）: c3 录值 [1.0, 1.1]
  [check 22] PASS  阻塞窗口精确: c3 在 t=2.1 拿到槽（c1/c2 t=2 释放，退避到点即 claim）且最终 SUCCESS
  [seq 01] t=0.0  r1.d1              -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  r1.d2              -> QUEUED           入 executor 队列（priority=0）
  [seq 03] t=0.0  r1.d3              -> QUEUED           入 executor 队列（priority=0）
  [seq 04] t=0.0  r1.d1              -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 05] t=1.0  r1.d1              -> CANCELLED        concurrency group 被新 run 抢占（cancel-in-progress: true）——运行中被取消（attempt 已花钱）
  [seq 06] t=1.0  r1.d2              -> CANCELLED        concurrency group 被新 run 抢占（cancel-in-progress: true）——排队中被取消（0 attempt）
  [seq 07] t=1.0  r1.d3              -> CANCELLED        concurrency group 被新 run 抢占（cancel-in-progress: true）——排队中被取消（0 attempt）
  [seq 08] t=1.0  r2.e1              -> QUEUED           入 executor 队列（priority=0）
  [seq 09] t=1.0  r2.e2              -> QUEUED           入 executor 队列（priority=0）
  [seq 10] t=1.0  r2.e3              -> QUEUED           入 executor 队列（priority=0）
  [seq 11] t=1.0  r2.e1              -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 12] t=2.0  r2.e1              -> SUCCESS          attempt 1 完成
  [seq 13] t=2.0  r2.e2              -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 14] t=3.0  r2.e2              -> SUCCESS          attempt 1 完成
  [seq 15] t=3.0  r2.e3              -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 16] t=4.0  r2.e3              -> SUCCESS          attempt 1 完成
  cancel-in-progress: ['r1.d1', 'r1.d2', 'r1.d3']（运行中 1 + 排队中 2）；r2 = ['r2.e1', 'r2.e2', 'r2.e3']
  [check 23] PASS  GHA 语义: 同 group 旧 run 全部取消（RUNNING 的 attempt 已花钱记浪费，QUEUED 的 0 attempt）
  [check 24] PASS  新 run 正常跑完: r2 三个 SUCCESS（cancel 不伤新 run）

[6] Agentic 自愈：坏源误分类重试耗尽 -> playbook P1 改道 fallback；capability 缺失 -> 升级人工（不自动授权）
  [seq 01] t=0.0  ingest_bad         -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  ingest_bad         -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 03] t=1.0  ingest_bad         -> RETRYING         attempt 1 exit=75 transient（source 拒绝所有读取（永久损坏）） —— 计划退避 1 tick
  [seq 04] t=2.0  ingest_bad         -> QUEUED           入 executor 队列（priority=0）
  [seq 05] t=2.0  ingest_bad         -> RUNNING          attempt 2 启动（queued_by=sched#1）
  [seq 06] t=3.0  ingest_bad         -> RETRYING         attempt 2 exit=75 transient（source 拒绝所有读取（永久损坏）） —— 计划退避 2 tick
  [seq 07] t=5.0  ingest_bad         -> QUEUED           入 executor 队列（priority=0）
  [seq 08] t=5.0  ingest_bad         -> RUNNING          attempt 3 启动（queued_by=sched#1）
  [seq 09] t=6.0  ingest_bad         -> FAILED           attempt 3 —— 重试上限 (2) 耗尽，止损
  [seq 10] t=6.0  gate_bad           -> UPSTREAM_FAILED  trigger_rule=all_success: 上游失败——依赖是承诺，不在坏数据上跑
  [seq 11] t=6.0  report_bad         -> UPSTREAM_FAILED  trigger_rule=all_success: 上游失败——依赖是承诺，不在坏数据上跑
  诊断: {"pattern": "P1_bad_source", "task": "ingest_bad", "cone": ["gate_bad", "report_bad"], "action": "reroute_to_fallback", "patch": {"quarantine": "ingest_bad", "add": "ingest_bad_fallback", "repoint": {"gate_bad": ["ingest_bad_fallback"]}}}
  [check 25] PASS  诊断命中 P1_bad_source: 重试耗尽 + 签名一致 + 下游锥饿死（观察全部来自结构化事件日志）
  [seq 01] t=0.0  ingest_bad_fallback -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  ingest_bad_fallback -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 03] t=1.0  ingest_bad_fallback -> SUCCESS          attempt 1 完成
  [seq 04] t=1.0  gate_bad           -> QUEUED           入 executor 队列（priority=0）
  [seq 05] t=1.0  gate_bad           -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 06] t=2.0  gate_bad           -> SUCCESS          attempt 1 完成
  [seq 07] t=2.0  report_bad         -> QUEUED           入 executor 队列（priority=0）
  [seq 08] t=2.0  report_bad         -> RUNNING          attempt 1 启动（queued_by=sched#1）
  [seq 09] t=3.0  report_bad         -> SUCCESS          attempt 1 完成
  [check 26] PASS  白名单 patch 生效: 隔离坏源 + fallback 改道后全 SUCCESS（复验 = 重跑收敛，不是口头保证）
  [seq 01] t=0.0  cap_task           -> QUEUED           入 executor 队列（priority=0）
  [seq 02] t=0.0  cap_task           -> FAILED(deny)     capability missing (default-deny): ['prod_deploy'] —— 0 attempt 0 成本
  诊断（capability 缺失）: {"pattern": "P0_capability_missing", "tasks": ["cap_task"], "action": "escalate_to_human", "reason": "capability 授权是安全边界（default-deny）——agent 不得自动授权，人工审批后重跑"}
  [check 27] PASS  安全边界 first-class: capability 缺失 -> escalate_to_human，agent 无授权动作（default-deny 不可绕过，attempts 仍 0）

[7] 真进程锚：parallelism=3 真实并行；kill -9 只杀调度器 -> 子进程成为孤儿 -> 重启收养（L1 做不到的那笔债）
  干净 run: 3 任务全 SUCCESS，digest 3c23229776e5f04c；三心跳区间存在共同存活窗（overlap > 0.1s = 真实并行的机器证据）
  [check 28] PASS  真进程并行: 三任务心跳区间存在共同存活窗（overlap > 0.1s；串行模型下三区间首尾相接、重叠≈0）
  kill 点: p_long RUNNING（pid 已录盘）；调度器被 kill -9，子进程 p_long/p_short1/p_short2 成为孤儿
  [check 29] PASS  调度器死而子进程活: rc=-SIGKILL，state.json 完整在盘，p_long 留 stale RUNNING 态（L1 在此只能 raise）
  [sched] start: tasks=3 parallelism=3
  [recover] p_long: adopted（pid 存活，heartbeat 新鲜）——liveness 与结果都走文件通道，不走父子关系
  [recover] p_short1: result channel 验证 SUCCESS（原子结果文件在盘）
  [recover] p_short2: result channel 验证 SUCCESS（原子结果文件在盘）
  [done] terminal: SUCCESS=['p_long', 'p_short1', 'p_short2']
  [check 30] PASS  孤儿收养成功: p_long 被重启的调度器收养（pid 存活 + heartbeat 新鲜），attempt 不增——工作不重做
  [check 31] PASS  收敛点不变: 终态 digest == 干净 run（崩溃模型不同，收敛点相同——L1 同款不变量）
  [check 32] PASS  崩溃税 = 0（对照 L1 killpg 模型税=1）: 孤儿活下来了，收养省下了重做的 attempt（总 attempts == 3）
  [check 33] PASS  结果通道二形态: 短任务在重启前已完成 = result 文件验收（pid 已死 + 原子结果在盘 -> SUCCESS）


self-check: 33/33 PASS
```

33 项 self-check 全 PASS。接下来逐面拆机制——每一节都先回答「为什么」，再看权威实现「怎么做」。

---

## §3 机制面 [1]：并行 executor 与资源池——为什么并行不改变收敛点

L0/L1 的规则 C 是串行出队（确定性选择）。L2 放开并行，状态链变成：

```
RUNNABLE →（pool 有空槽）→ QUEUED →（executor open_slots）→ RUNNING
```

两道门槛各自对应权威实现的一个精确事实：

**门槛一：executor 空槽。** Airflow 的 `BaseExecutor.heartbeat()` 是「触发新任务的时刻」，空槽算术一行裸出：

```python
# airflow/executors/base_executor.py:L348-350（3.x 主线 fresh 抓取，2026-08-16）
def heartbeat(self) -> None:
    """Heartbeat sent to trigger new jobs."""
    open_slots = self.parallelism - len(self.running)
```

nano 的 `open_slots()` 逐字同款（`parallelism - RUNNING 数`）。注意机制本质：**心跳到达 = 调度发生的时刻**——Airflow 不为「出队」单开一个线程或定时器，心跳本身就是节拍器。nano 的 `step()` 每 tick 调和一次，同构。

**门槛二：pool 占用。** Airflow `Pool.occupied_slots` 把 **queued 也算进占用**（pool.py:L269），且 `slots=-1 → float("inf")`（pool.py:L209）。为什么 queued 要算占用？因为「已排队未启动」是对池容量的**预承诺**——若只数 running，调度器会在启动延迟窗内超额排队，池就超卖了。nano 的 `pool_occupied = QUEUED + RUNNING` 同款，check 06 用 pool_ledger 机器验证「占用峰值 ≤ slots」。

**出队顺序**按 priority 排序（base_executor.py:L428-442 `order_queued_tasks_by_priority`，`sorted by priority_weight, reverse=False`）。nano 约定「权重小者先出」，只引排序事实——Airflow 的 `priority_weight` 数值方向语义由其 priority 计算族治理，此处不展开（不锁定、不引申）。

**关键 check：并行不改变收敛点。** [1] 以 `parallelism=1` 退化运行 L1 逐字同款 fixture，终态 digest 复现 L1 录值 `ac4a0b3ac09bf47b`（跨级锚，check 01）；[2] 放开 `parallelism=3` 后终态 digest 不变（check 05）——**调度策略是路径，不是语义**。并行改变的是 makespan 与资源占用轨迹，不改变「谁成功、谁失败、谁被拒」。check 09 用无共享资源探针量化并行的收益（3 独立任务 × 2 tick：串行 6 tick → 并行 2 tick），同时 check 08 诚实录值：本 fixture 的临界路径是 `ingest_web` 的退避链（wall-clock 依赖），并行省不下——**并行的收益取决于临界路径是算力还是等待**，这是调度优化的第一性判断。

```text
  pool 'src' 占用峰值 = 1（slots=1）；首个出队任务 = unit_tests（priority=0 先于 priority=5）
  [check 05] PASS  并行不改变收敛点: 终态 digest == [1]（调度策略是路径，不是语义）
  [check 06] PASS  pool 不变量: 'src' 占用峰值 ≤ slots（occupied = QUEUED+RUNNING，对照 Pool.occupied_slots）
  [check 09] PASS  并行的量化证据（无共享资源探针）: 3 独立任务 × 2 tick，串行 6 tick -> 并行 2 tick（3× 加速）
```

---

## §4 机制面 [2]：trigger rules 配置化——统计口径与调度口径不是一回事

L0/L1 的依赖语义是写死的 all_success。L2 把「上游终态向量 → 本任务命运」抽成可配置纯函数 `trigger_decision(rule, counts, done, total)`，语义逐条对照 Airflow `TriggerRuleDep` 的分派表（trigger_rule_dep.py:L429-440 flag 分支）。Airflow 现行主线 `TriggerRule` 枚举共 **13 项**（triggerrule.py 抓取件在盘），nano 模拟其中 6 项（all_success / all_done / one_success / none_failed / none_failed_min_one_success / always），子集声明见 §17。

新终态 **SKIPPED** 随 branching 登场（未选中分支），随之而来的是一个容易踩的口径陷阱：

```python
# airflow/utils/state.py:L222-224（抓取件逐字）
success_states: frozenset[TaskInstanceState] = frozenset(
    [TaskInstanceState.SUCCESS, TaskInstanceState.SKIPPED]
)
```

**SKIPPED ∈ success_states**——但这是 **DagRun 统计口径**（「这个 run 算不算成功结束」）。而在**调度口径**（「要不要触发下游」）里，ALL_SUCCESS 遇到 skipped 上游会被染成 SKIPPED，trigger_rule_dep.py:L430-434 逐字：

```python
if trigger_rule == TR.ALL_SUCCESS:
    if upstream_failed or failed:
        new_state = TaskInstanceState.UPSTREAM_FAILED
    elif skipped:
        new_state = TaskInstanceState.SKIPPED
```

为什么两套口径必须分开？统计口径回答「账面上这算好事吗」（skipped 不是坏事，算 success），调度口径回答「下游期待的东西真的发生了吗」（skipped 意味着上游**没跑**，承诺没兑现）。若混用一套口径：按统计口径，branching 后被 skip 的分支会被下游当「成功」继续跑——坏数据就是这么进来的。check 10 机器验证这条染色规则（`join_all_success -> SKIPPED`），[3] 的 11 任务谱系把 6 条规则全部走到：

```text
  join_all_success     -> SKIPPED
  join_none_failed     -> SUCCESS
  join_one_success     -> SUCCESS
  [check 10] PASS  ALL_SUCCESS 被 skipped 上游染成 SKIPPED（trigger_rule_dep.py:L433-434；skipped 不算成功调度口径）
  [check 13] PASS  ALL_DONE 等全部终态后无条件跑（含 FAILED 上游）；NONE_FAILED_MIN_ONE_SUCCESS 见 failed 即 UPSTREAM_FAILED
```

---

## §5 机制面 [3]：heartbeat 连续形态与孤儿收养——liveness 不走父子关系

L1 的 zombie 检测是**重启时点状**的（宿主死亡模型：killpg 全组俱灭，重启后扫一遍盘上状态）。L2 请回两笔债：

**(a) 连续形态：周期扫描 RUNNING 的心跳，超时即处置。** Airflow 的实现是一个 10s 周期定时器：

```python
# airflow/jobs/scheduler_job_runner.py:L1723-1725（抓取件逐字）
timers.call_regular_interval(
    conf.getfloat("scheduler", "task_instance_heartbeat_timeout_detection_interval", fallback=10.0),
    self._find_and_purge_task_instances_without_heartbeats,
)
```

purge 条件是 `state ∈ {RUNNING, RESTARTING} 且 last_heartbeat_at < limit`（scheduler_job_runner.py:L3516-3519）。nano 的 `step()` [b] 段每 tick 扫描同款语义，逻辑层超时 2 tick。**「进程还活着但心跳停」= stuck**——活着却没有进展，杀之重试（对应 base_executor.py:L456-458 注释里「task has been killed externally and not yet been marked as failed」的路径）。check 15 精确到 tick：心跳停于 t=1、超时=2 → 恰在 t=4 被杀（检测时刻 = last_hb + timeout 后的首个事件点，确定性可锚）。

**(b) 孤儿收养：调度器死、子进程活。** 这是 L1 只能 raise 的场景。Airflow 的答案是本机制面的核心，值得逐行读（scheduler_job_runner.py `adopt_or_reset_orphaned_tasks` 块，抓取件逐字）：

```python
# L3266-3271：调度器死亡判定 = 心跳超时（Job.latest_heartbeat 走 DB）
Job.latest_heartbeat < (timezone.utcnow() - timedelta(seconds=timeout))  # → values(state=JobState.FAILED)

# L3301：行锁抢占，防两个调度器同时收养
tis_to_adopt_or_reset_query = with_row_locks(query, of=TI, session=session, skip_locked=True)

# L3310-3311：能收养则收养
to_reset.extend(executor.try_adopt_task_instances(tis))

# L3322 / L3327：不能收养的 → state=None 回炉重调；能收养的 → queued_by 换成新调度器
ti.state = None
...
ti.queued_by_job_id = self.job.id
```

机制本质三句话：

1. **调度器死没死，不看进程，看心跳**（`Job.latest_heartbeat` 超时 → 标 FAILED，job.py:L100 列定义 / L141 更新 / L168 使用）。
2. **孤儿归谁，行锁说了算**（`skip_locked=True`——另一个调度器锁走的行我跳过，双收养竞争在 DB 层消解）。
3. **收养不成的回炉重调**（`state=None`，工作可能重做，但状态机不悬空）。

而「可被收养」的状态集合 = `adoptable_states = {QUEUED, RUNNING, RESTARTING}`（state.py:L229-231）。

[4] 的逻辑层复现：sched#1 跑到 t=2 被丢弃（模拟崩溃），sched#2 从状态快照重建，`recover()` 第一件事就是收养判定——`long_a` 心跳新鲜 + 进程存活 → `queued_by` 换人、**attempt 不增（工作不重做）**；`short_b` 已终态不受扰动（check 18/19）。

```text
  [seq 01] t=2.0  long_a             -> RUNNING          adopted（sched#2 收养：心跳新鲜 + 进程存活，attempt 1 继续，不重启）
  [check 18] PASS  孤儿收养: long_a 被 sched#2 收养——queued_by 换人、attempt 不增（工作不重做）
```

注意 nano 的收养凭据是「pid 存活 + heartbeat 文件新鲜」，而 Airflow 两者都走 DB。这个差异不是偷懒——run [7] 会证明：**pid 存活探测在 OS 层就是个不可靠信号**（§9 zombie 反例）。Airflow 把 heartbeat 放进 DB，正是为了彻底绕开父子关系。

---

## §6 机制面 [4]：并发键——Dagster 的槽位账本与指数步进退避

有些资源不能按「任务数」限流，要按**逻辑键**限流（如「GPU 池最多 2 个任务同时用」）。Dagster 的 op concurrency 机制本质是一个**槽位账本**：

```python
# dagster op_concurrency_limits_counter.py:L220-225（抓取件逐字）
available_count = (
    key_info.slot_count
    - len(key_info.pending_steps)
    - self._launched_pool_counts[pool]
    - self._in_progress_pool_counts[pool]
)
# L230-231：全部根键都满才阻塞出队
# "if we reached here, then every root concurrency key is blocked, so we should not dequeue"
```

nano 的 `KeyLedger.claim/release` 同款账本（`held < slots` 才 claim 成功），check 20 机器验证不变量「持有峰值 ≤ slot_count ∧ claims==releases」。claim 失败的退避是**指数步进**，Dagster 公式逐字：

```python
# dagster instance_concurrency_context.py:L178-189 + 常数 L16-18（抓取件逐字）
INITIAL_INTERVAL_VALUE = 1; STEP_UP_BASE = 1.1; MAX_CONCURRENCY_CLAIM_BLOCKED_INTERVAL = 15
step_up_value = STEP_UP_BASE**pending_claim_count - 1
interval = INITIAL_INTERVAL_VALUE + step_up_value
return min(MAX_CONCURRENCY_CLAIM_BLOCKED_INTERVAL, interval)
```

即 `1 + (1.1^n - 1)`、上限 15s。[5] 实测 c3 被拒 2 次、退避序列 `[1.0, 1.1]` tick、t=2.1 精确拿到槽（check 21/22）——与 L1 的 wall-clock 指数退避同族（**等待是指数增长的耐心**）。

最容易被忽略的一笔诚实设计：Dagster 的并发上下文退出时**只释放 pending claims，不释放已持有槽位**，docstring 逐字：

```text
# instance_concurrency_context.py:L28-31（抓取件逐字）
It ensures that pending concurrency claims are freed upon exiting context.  It does not,
however, free active slots that have been claimed. This is because the executor (depending on
the executor type) may have launched processes that may continue to run even after the current
context is exited.
```

为什么？因为**进程可能还在跑**——提前释放已持有槽位 = 账本超卖，和 §3 pool「queued 算占用」是同一笔账：**资源账本的保守不是缺陷，是对「我看不到进程真实状态」的诚实**。这与 L1「崩溃垃圾可见」的幂等诚实一脉相承。

---

## §7 机制面 [5]：CI/CD 参照——concurrency group 与 process mode（不锁定）

本课程的比较原则：云厂商/商业平台实现**作为参照，不锁定**。CI/CD 的并发控制与编排器同构，引两条一手文档对照：

**GitHub Actions concurrency group + cancel-in-progress**（docs.github.com workflow-syntax 页，抓取件逐字）：

> You can use jobs.&lt;job_id&gt;.concurrency to ensure that only a single job or workflow using the same concurrency group will run at a time.
> To also cancel any currently running job or workflow in the same concurrency group, specify cancel-in-progress: true.

nano 的 `cancel_group()` 模拟这个语义：新 run 提交时，同 group 旧 run 的未终态任务全部 CANCELLED——**运行中的 attempt 已花钱记浪费，排队中的 0 attempt**（check 23：`r1.d1` 运行中被取消 attempts=1，`r1.d2/d3` 排队中被取消 attempts=0）。成本记账在取消时刻就分清：已花的钱进浪费账，没花的钱一分不动。

**GitLab CI resource_group**（docs.gitlab.com/ci/yaml/ 页，抓取件逐字：`resource_group — Limit job concurrency.`）的独有概念是 **process mode**（docs.gitlab.com/ci/resource_groups/ 页）：`unordered`（谁抢到谁跑）/ `oldest_first`（按 pipeline ID 升序）/ `newest_first`（降序——连续部署场景下最新的优先）。GitLab 对 `interruptible` 的定义与 GHA cancel-in-progress 同族（ci/yaml 页逐字：「Defines if a job can be canceled when made redundant by a newer run」）。nano 只模拟 cancel-in-progress 一脉，process mode 表作对照不实现——**参照的意义是看清机制谱系，不是把每家方言都搬进来**。

```text
  [seq 05] t=1.0  r1.d1              -> CANCELLED        concurrency group 被新 run 抢占（cancel-in-progress: true）——运行中被取消（attempt 已花钱）
  [seq 06] t=1.0  r1.d2              -> CANCELLED        concurrency group 被新 run 抢占（cancel-in-progress: true）——排队中被取消（0 attempt）
  [check 23] PASS  GHA 语义: 同 group 旧 run 全部取消（RUNNING 的 attempt 已花钱记浪费，QUEUED 的 0 attempt）
```

---

## §8 机制面 [6]：Agentic 管线自愈——观察 → 诊断 → 白名单行动 → 复验

本课程把「Agentic 能力」列为本模块核心机制之一。去魅之后，agent 自愈的机制骨架是四步闭环，每一步都有明确的安全边界：

1. **观察**：全部来自结构化事件日志（events 流），不读日志文本、不猜。
2. **诊断**：playbook 模式匹配。[6] 演示两条：`P1_bad_source`（重试耗尽 + 失败签名一致 + 下游锥饿死 = 坏源误分类）与 `P0_capability_missing`（capability 缺失）。
3. **行动**：**白名单 DAG patch**——隔离坏源 + fallback 改道 + 重指依赖，patch 是声明式的，可审。
4. **复验**：重跑收敛才算数，不是口头保证（check 26：patch 后全 SUCCESS）。

安全边界是 **first-class** 的：`P0_capability_missing` 的动作恒为 `escalate_to_human`——**capability 授权是安全边界（default-deny），agent 不得自动授权**（check 27：`cap_task` attempts 仍 0，拒绝发生在 QUEUED 之前，0 成本）。这条边界不可被 agent 的任何「聪明」绕过：诊断可以泛化，授权必须人工。

```text
  诊断: {"pattern": "P1_bad_source", "task": "ingest_bad", "cone": ["gate_bad", "report_bad"], "action": "reroute_to_fallback", ...}
  [check 25] PASS  诊断命中 P1_bad_source: 重试耗尽 + 签名一致 + 下游锥饿死（观察全部来自结构化事件日志）
  [check 26] PASS  白名单 patch 生效: 隔离坏源 + fallback 改道后全 SUCCESS（复验 = 重跑收敛，不是口头保证）
  [check 27] PASS  安全边界 first-class: capability 缺失 -> escalate_to_human，agent 无授权动作（default-deny 不可绕过，attempts 仍 0）
```

真实系统里 LLM 坐在 diagnose 的位置增加泛化（把「签名匹配」换成「读事件流归纳」），但机制骨架不变——**观察结构化、行动白名单、复验机器化、授权 default-deny**。nano 用确定性策略裸出骨架，正是为了看清 LLM 只是骨架上的一个可替换关节。

---

## §9 run [7] 真进程锚：zombie 反例教材——「进程存活探测」为什么不可靠

这是本级的核心，也是一个真实发生的反例：**本级第一版代码就跌进了这个坑，复现实验稳定得到 EXIT=1，修复后陷阱本身成了教材。**

### 9.1 一个 OS 层铁证：未收割的 zombie 对 kill(pid,0) 恒返回 True

先跑一个探针（当前版本现场实测，可自行复现）：

```python
import subprocess, sys, os, time
p = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
time.sleep(0.5)                      # 子进程已退出，父进程从不 waitpid → 未收割 zombie
def pid_alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
print("before reap:", pid_alive(p.pid))   # True —— 进程死了，kill(pid,0) 却成功
print("after  reap:", (p.poll(), pid_alive(p.pid)))   # (0, False) —— poll() 即收割
```

```text
before reap: True
after  reap: (0, False)
```

机制：子进程退出后，内核保留其进程表项（存 exit code 等父进程来取），这就是 **zombie**。`os.kill(pid, 0)` 只检查「进程表项存在且我有权限」——**对 zombie 返回成功**。只有 `waitpid`/`poll()` 收割后表项才消失。于是任何基于 `kill(pid,0)` 的 `pid_alive` 探测都会把「死了但没人收尸」误判成「活着」。macOS/Linux 对未收割 zombie 的这一行为一致（POSIX 语义）。

### 9.2 第一版代码是怎么跌进去的（两个缺陷，一个本质）

第一版 run [7] 的完成检测长这样（pid 探测在前，结果文件在后）：

```python
if t["pid"] and pid_alive(t["pid"]):     # zombie → 恒 True → 永远走这个分支
    if not hb_fresh(...): 杀之回炉
    continue                              # 结果文件检查永不可达
if os.path.exists(result): SUCCESS        # 死代码
```

任务正常退出 → 变 zombie → `pid_alive` 恒 True → 完成检测永走「存活」支 → 心跳停 → 超时判 stuck → SIGKILL（对 zombie 是 no-op）→ PENDING 回炉 → 无限重试。缺陷复现实测（2 遍 × 2 新建空 CWD，EXIT=1 字节级确定性复现）：events.jsonl 全是「启动 → heartbeat 超时（进程存活 = stuck）杀之回炉 → 重启」循环，**result 文件零件**——任务其实早就正常完成了，但调度器永远看不见。

同批复跑还暴露了第二个缺陷：**启动宽限缺失**。刚启动的进程还没来得及写第一跳心跳，`hb_fresh`（无宽限基准）判「不新鲜」→ 立即当 stuck 杀掉 → 无限重启。探针实测：3 秒内 attempts 爬到 422，hb 日志每任务仅 1 beat。

两个缺陷看似不同，本质是同一个：**把不可靠的信号当成了权威**——zombie 缺陷信任了「pid 存活」，宽限缺陷把「心跳文件暂不存在」等同于「心跳停了」。而讽刺的是，本代码开篇 [3] 节自己写着结论：「Airflow 的答案：liveness 与结果都不走父子关系」。第一版代码论证了正确答案，然后没按正确答案写。

### 9.3 修复：结果通道优先 + 启动宽限——正是 Airflow 的选择

修复后的完成检测（现行代码）：**原子结果文件在盘 = 完成，与 pid 活否无关**——

```python
result = os.path.join(wd, f"result_{n}.json")
if os.path.exists(result):                 # 结果通道优先：zombie 对 kill(pid,0) 恒 True，
    t["state"] = "SUCCESS"; ...            # pid 探测不可靠，结果文件才是权威完成信号
    continue
if t["pid"] and pid_alive(t["pid"]):
    if not hb_fresh(wd, n, REAL_HB_TIMEOUT, since=t.get("launched_at")):  # 启动宽限
        杀之回炉
    continue
回炉（pid 已死且无结果文件）
```

两处修复各自对照权威实现：

- **结果通道优先**：任务程序把「完成」写成原子发布的 `result_{n}.json`（tmp + fsync + `os.replace`，L1 atomic_write 同款），调度器验收结果文件而非进程状态。这与 Airflow 的哲学同构——任务完成是 DB 里可独立验证的状态变更，不是「父进程观察子进程退出」。**exit code 不再是唯一结果通道**：孤儿场景下收养方根本不是父进程，waitpid 会 ECHILD，exit code 永久丢失——结果通道是唯一跨进程边界存活的完成证据。
- **启动宽限**：启动时录 `launched_at`，首跳之前以它为新鲜度基准——对照 Airflow 把 `last_heartbeat_at` 在入队/启动时即初始化（heartbeat 更新见 job.py:L141），解释器启动延迟不算 stuck。

### 9.4 run [7] 的机器证据

修复后 run [7] 全绿（§2 paste 块 [7] 段逐字）：干净 run 三任务真并行（三心跳区间存在共同存活窗，overlap > 0.1s——串行模型下三区间首尾相接、重叠≈0，这是真实并行的机器证据）；然后 `kill -9` **只杀调度器**（对照 L1 killpg 全组俱灭 = 两种故障模型的分岔点），三个子进程成为孤儿；on-call 窗后重启调度器：

```text
  [recover] p_long: adopted（pid 存活，heartbeat 新鲜）——liveness 与结果都走文件通道，不走父子关系
  [recover] p_short1: result channel 验证 SUCCESS（原子结果文件在盘）
  [recover] p_short2: result channel 验证 SUCCESS（原子结果文件在盘）
  [check 30] PASS  孤儿收养成功: p_long 被重启的调度器收养（pid 存活 + heartbeat 新鲜），attempt 不增——工作不重做
  [check 31] PASS  收敛点不变: 终态 digest == 干净 run（崩溃模型不同，收敛点相同——L1 同款不变量）
  [check 32] PASS  崩溃税 = 0（对照 L1 killpg 模型税=1）: 孤儿活下来了，收养省下了重做的 attempt（总 attempts == 3）
  [check 33] PASS  结果通道二形态: 短任务在重启前已完成 = result 文件验收（pid 已死 + 原子结果在盘 -> SUCCESS）
```

收养判定的序也有讲究：重启恢复时**先查结果文件**（在盘即 SUCCESS），再查「pid 存活 + 心跳新鲜」（收养），最后才回炉。为什么这个序是确定的？因为 zombie 的心跳可能「尚新鲜」（刚退出不到 0.6s）——若先查 pid+心跳，已完成的短任务会被误判成 adopted，判定随时序漂移。结果文件在盘是**与时间无关的事实**，放第一位，整个恢复序就确定了（RUN1==RUN2 字节级一致，§17 锚点）。

**教材化的结论**：输出全绿 ≠ 机制正确——runs [1]–[6] 的 27 项 check 全 PASS 掩盖不了 run [7] 的确定性崩溃，因为逻辑层用 fixture 字段（`dies_at`/`hb_stop`）显式建模 liveness，**zombie 这种「假存活」在逻辑层根本无法表达**。真进程锚的价值正在于此：它把 OS 层的不可约事实请进验收。而「为什么权威实现绕开这个坑」也从一句引文变成了用 EXIT=1 换来的教训。

---

## §10 权威实现取舍表：nano 版 vs Airflow / Dagster / GHA / GitLab

行号为抓取件录值（Airflow 3.x 主线 + Dagster 主线 fresh 抓取，2026-08-16 12:2x；主线演进，±1–3 行漂移在声明容差内，见 §17）：

| 机制面 | nano L2 | 权威实现（一手源码） | 为什么它那样选 |
|--------|---------|----------------------|----------------|
| 并行门槛 | `open_slots = parallelism - RUNNING` | base_executor.py:L348-350（heartbeat() 内计算，"Heartbeat sent to trigger new jobs"） | 心跳到达 = 调度时刻，不另开调度线程 |
| 资源池 | occupied = QUEUED+RUNNING；-1→无限 | pool.py:L209（`float("inf")`）/ L269（occupied_slots） | queued 算占用 = 防启动延迟窗超卖 |
| 出队顺序 | sorted by (priority, name) | base_executor.py:L428-442（sorted by priority_weight, reverse=False） | 只引排序事实；权重方向语义归 Airflow priority 计算族 |
| trigger rules | 6 条（13 条枚举的子集） | trigger_rule_dep.py:L429-440 分派分支 + triggerrule.py 13 项枚举 | 终态向量→命运是纯函数，配置化不写死 |
| SKIPPED 双口径 | success 计数不含 SKIPPED，trigger 单独数 skipped | state.py:L222-224（success_states 含 SKIPPED）+ trigger_rule_dep.py:L430-434 | 统计口径（DagRun）≠ 调度口径（触发下游） |
| heartbeat 连续形态 | 每 tick 扫 RUNNING last_hb，超时 2 tick | scheduler_job_runner.py:L3516-3519 + L1723-1725（10s 周期定时器，fallback=10.0） | 连续 purge，不是重启时点状检测 |
| 启动宽限 | launched_at + `hb_fresh(since=launched_at)` | last_heartbeat_at 入队/启动即初始化（job.py:L141 更新心跳） | 首跳前的解释器启动延迟不算 stuck（§9 实测反例） |
| 孤儿收养 | pid 存活 + 心跳新鲜 → adopted；结果在盘 → 验收；否则回炉 | state.py:L229-231（adoptable_states）+ scheduler_job_runner.py:L3301（skip_locked）/ L3310-3311（try_adopt）/ L3322（state=None）/ L3327（queued_by_job_id 换人） | 行锁防双收养；收养不成回炉重调 |
| 调度器死亡判定 | kill -9 调度器，心跳停即死 | Job.latest_heartbeat < now - timeout → FAILED（scheduler_job_runner.py:L3266-3271；job.py:L100 列定义 / L168 使用） | liveness 走 DB，不走父子关系 |
| 完成检测序 | **结果通道优先**，再看 pid/心跳 | 任务完成 = DB 状态变更（可独立验证），非父进程观察子进程退出 | pid 探测对 zombie 不可靠（§9 铁证） |
| 并发键 | KeyLedger claim/release；退避 `1+(1.1^n-1)` cap 15 | op_concurrency_limits_counter.py:L220-225（available_count）/ L230-231（全满才阻塞）+ instance_concurrency_context.py:L178-189（常数 L16-18：1/1.1/15） | 槽位账本是 single source of truth |
| 上下文退出不释放已持有槽 | key 只在任务终态释放 | instance_concurrency_context.py:L28-31 docstring（进程可能还在跑） | 提前释放 = 账本超卖 |
| executor 容量 | REAL_PARALLELISM=3 | executor_definition.py:L426（"tells the execution engine how many processes may run"） | — |
| cancel-in-progress | cancel_group：旧 run 未终态全 CANCELLED | GHA workflow-syntax 页双引文（§7） | 新提交使旧 run 冗余 |
| process mode | 参照不实现 | GitLab resource_groups 页（unordered/oldest_first/newest_first） | 部署语义选择：最新优先 vs 最旧优先 |

---

## §11 toy vs 生产：差距的诚实声明

1. **状态后端：文件通道 vs metadata DB。** nano 用 state.json（原子写）/ events.jsonl（append-only）/ hb_*.log / result_*.json；Airflow 用 Postgres 等 metadata DB。最实质的差距：**收养行锁（skip_locked）在 nano 的单机串行场景没有竞争对手**——真多调度器的双收养竞争必须 DB 行锁消解，nano 无法复现该竞争面 `[TODO: verify on real system]`。
2. **心跳量级。** toy 用 0.15s 心跳 / 0.6s 超时压缩墙钟；检测周期 10s 是 Airflow 抓取件录值（fallback=10.0）。生产心跳周期/超时为可配置值，量级与 toy 完全不同，绝对数字不可外推。
3. **executor 形态。** nano 是单机 subprocess，调度器与任务同机；Airflow 的 CeleryExecutor/KubernetesExecutor 把任务进程分布到不同主机——孤儿问题在那里是常态而非特例，这正是「心跳走 DB」的工程必要性来源。
4. **结果通道。** nano 用原子 result 文件；Airflow 用 DB 状态 + XCom 数据通道。本质同构：完成是**可独立于 liveness 验证的权威产物**。
5. **trigger rules 子集。** 6/13（子集声明见 §17），all_failed/one_failed/none_skipped 等未模拟。
6. **healer 是确定性 playbook。** 生产形态 LLM 坐在 diagnose 位（§8），机制骨架不变。
7. **失败模型的分工。** exit code 分类（L1 的 EX_TEMPFAIL 契约）在逻辑层 runs [1]–[6] 以 fixture 声明承载；run [7] 的任务程序恒成功——真进程层的故障模型只放「调度器被杀」，两种失败模型不混在一个 run 里，各自可锚。

---

## §12 Airflow / Dagster 的当今定位（时效性声明）

本教程全部 Airflow 行锚取自 **3.x 现行主线**（fresh 抓取 2026-08-16 12:2x）：`SchedulerJobRunner` 的收养块、executor 的 workload API 形态（queued_tasks/queued_callbacks）、13 项 TriggerRule 枚举均为当前主线形态——截至对齐日（2026-08-16），Airflow（Apache 顶级项目）仍是批处理编排的主流系统之一，Dagster 的 op concurrency 亦是现行活跃 API。GHA/GitLab CI 为 CI/CD 事实标准参照。

本教程教的是**机制本质**（心跳放哪、孤儿怎么收养、并发键记在哪、完成信号凭什么权威）——这些不随版本漂移；行号是抓取件录值，漂移容差与抓取日声明见 §17。经典机制 ≠ 版本锁定：学 Airflow 的收养块不是为了用 Airflow，是为了看懂**任何**分布式调度器都必须回答的那四个问题。

---

## §13 费曼自检

**类比（讲给外行听）**：病房。调度器 = 护士长，任务进程 = 病人，heartbeat 文件 = 心电图纸带，result 文件 = 主治医生签字的出院小结，`pid_alive` = 护士探头看「床位是否被占着」。

- 为什么「看床位」不可靠：病人可能已经去世但遗体还没被接走（zombie = 未收割的进程表项，床位仍被占着）；也可能病人只是刚去做检查还没回来（进程刚启动、第一跳心跳还没落盘）——探头一看「床空着」就宣布死亡，同样会误杀。
- 护士长的规矩：不靠探头，靠两样东西——心电图纸带是否新鲜（heartbeat），出院小结是否归档（result）。**出院小结权威最大**：小结在档 = 已出院（SUCCESS），不管床位被谁占着。
- 交接班（收养）：夜班护士长倒下（调度器死），白班接手时只看病历（state.json）：心电图新鲜 + 人活着 → 接着护理（adopted，点滴不重打）；出院小结在档 → 直接记出院；两者皆无 → 重新排队检查（zombie 回炉）。
- 为什么靠病历不靠「护士亲眼所见」：护士会换班（调度器会死），病历不丢——**liveness 与结果都不走亲眼所见（父子关系），走病历（DB/文件通道）**。

**自检问（答不出就回读）**：

1. 为什么 `os.kill(pid, 0)` 对 zombie 返回成功？（§9.1——进程表项未收割）
2. 为什么结果文件比 exit code 更权威？（§9.3——孤儿场景收养方不是父进程，waitpid 会 ECHILD，exit code 永久丢失）
3. 启动宽限为什么必要？没有它会发生什么？（§9.2——首跳前被误判 stuck，无限重启）
4. SKIPPED 为什么在统计口径算 success、在调度口径不算？（§4）

---

## §14 思考题（×5）

1. 如果把「结果通道优先」换成「exit code 优先」（父进程 waitpid 收割取 exit code），在孤儿收养场景会出什么新问题？（提示：收养方与任务进程无父子关系；代码注释「poll，不 waitpid——收养的 pid 非子进程」正是此意。）
2. hb_*.log 跨 attempt 追加写。若一个任务真的 stuck 后重启，旧 attempt 的心跳残留会怎样干扰新鲜度判断？给出一个修复方案（如按 attempt 分段/截断/心跳带 attempt 号），并说明 Airflow 的 `last_heartbeat_at` 单字段覆盖写为什么天然没有这个问题。
3. Dagster 上下文退出只释放 pending claims、不释放已持有槽位（§6 引文）。如果激进地全部释放，账本会发生什么？用一个「进程还在跑但槽位已被让出」的具体时序说明超卖。
4. ALL_SUCCESS 被 skipped 上游染成 SKIPPED，而 SKIPPED ∈ success_states（§4）。设计一个真实场景（如 branching + 报表任务），演示若只保留一套口径会分别出什么错。
5. run [7] 的收养发生在 on-call 窗（0.8s）之后。如果收养太快——旧调度器可能只是短暂卡住而非死亡——会发生什么（两个调度器同时管一个任务）？Airflow 用什么机制防双收养（§5 引文 skip_locked），nano 的单机文件通道为什么复现不了这一竞争（§11）？

---

## §15 反例与边界

**核心反例（§9 教材化）**：`pid_alive`（`os.kill(pid,0)`）对未收割 zombie 误判 True = 进程存活检测不可靠的 OS 层铁证——这正是 Airflow 不走父子关系（心跳走 DB、孤儿按行锁收养、完成走独立验证）的原因。本级第一版代码因「完成检测信任 pid_alive + 启动无宽限」确定性崩溃（缺陷复现实测 EXIT=1×2），现行代码为修复形态（结果通道优先 + launched_at 宽限），探针代码在 §9.1 可复现。

**边界清单**（每一面都是显式声明，不是遗漏）：

- zombie 行为 = POSIX 语义（macOS/Linux 对未收割 zombie 的 kill(pid,0) 一致）；Windows 进程模型不同 `[TODO: verify]`。
- 单机文件通道复现不了双收养竞争（skip_locked 行锁的面），§11 已声明。
- trigger rules 6/13 子集；heartbeat 量级为 toy；healer 为确定性策略。
- runs [1]–[6] 是逻辑时钟本质模拟（显式注明），run [7] 是真进程；模拟核心全部可运行（本课程契约）。

---

## §16 阶梯预告

同轨下一篇建议阅读 **nano-rag-retrieval L2**：混合检索（向量 + 稀疏）与重排序、检索质量的量化评估（recall@k / nDCG 族），以及对照 Milvus / OpenSearch 权威实现的取舍分析。L1 已提供真实小 embedding 模型与索引持久化，跨级前提在位。

---

## §17 溯源与口径声明

| 声明 | 类型 | 来源 |
|------|------|------|
| Airflow 行锚（§3/§4/§5/§10：base_executor.py:L348-350·L428-442·L456-458 / pool.py:L209·L269 / scheduler_job_runner.py:L1723-1725·L3266-3271·L3301·L3310-3311·L3322·L3327·L3516-3519 / state.py:L222-224·L229-231 / trigger_rule_dep.py:L429-440 / triggerrule.py 13 项枚举 / job.py:L100·L141·L168） | 文献已有（源码逐字，行号为抓取件录值） | apache/airflow 3.x 主线 fresh 抓取，2026-08-16 保存的一手源码抓取件（af_base_executor_fresh.py / af_pool_fresh.py / af_scheduler_job_runner_fresh.py / af_state_fresh.py / af_trigger_rule_dep_fresh.py / af_trigger_rule_fresh.py / af_jobs_job.py），2026-08-16 12:2x 抓取；±1–3 行漂移在代码自声明容差内 |
| job.py 行锚更正（docstring 原录 L100/L190 → L100 定义/L141 更新/L168 使用） | 文献已有（现场抓取件核对） | af_jobs_job.py 抓取件 grep 录值：L100 列定义 / L141 `self.latest_heartbeat = timezone.utcnow()` / L168 使用；L190 超漂移容差，当前版本清偿（版本漂移校准） |
| Dagster 行锚（§6/§10：op_concurrency_limits_counter.py:L220-225·L230-231 / instance_concurrency_context.py:L28-31·L178-189·常数 L16-18 / executor_definition.py:L426） | 文献已有（源码逐字） | dagster-io/dagster 主线 fresh 抓取，同日保存的一手源码抓取件（dg_op_concurrency_limits_counter.py / dg_execution_plan_instance_concurrency_context.py / dg_definitions_executor_definition.py），2026-08-16 12:2x |
| GHA 双引文（§7："ensure that only a single job or workflow using the same concurrency group will run at a time" / "cancel any currently running job or workflow in the same concurrency group"） | 文献已有（逐字引文） | docs.github.com workflow-syntax concurrency 节，抓取件 gha_concurrency.html 352,578 B，2026-08-16 12:2x |
| GitLab 引文（§7："resource_group — Limit job concurrency." / interruptible "Defines if a job can be canceled when made redundant by a newer run" / process mode unordered·oldest_first·newest_first） | 文献已有（逐字引文） | docs.gitlab.com/ci/yaml/（抓取件 gitlab_ci_yaml.html 647,161 B，resource_group 与 interruptible 两条目均在此页）+ docs.gitlab.com/ci/resource_groups/（抓取件 gitlab_resource_groups.html 67,987 B，process mode 表），2026-08-16 12:2x；代码 docstring 引文 "can be canceled when made redundant by a newer run" 为源句 "Defines if a job can be canceled when made redundant by a newer run" 的中段截取，口径在此声明 |
| 404 失败件 ×3（af_sdk_trigger_rule / af_trigger_rule / pf_types_flow_run） | 抓取失败显式声明 | 三件各 14 B「404: Not Found」在盘；本教程与代码对 Prefect flow run 状态、Airflow SDK TriggerRule **零引用**——失败显式，未从失败件编造引文 |
| zombie 探针（§9.1：未收割 zombie kill(pid,0)→True / poll() 收割后→False） | 当前版本现场实测（探针代码文内在盘可复现） | CPython 3.13.13，2026-08-16；独立复现探针结论一致 |
| 第一版缺陷取证（§9.2：EXIT=1 确定性复现 / events 无限循环 / result 零件 / 3s attempts≈422） | 缺陷复现记录 + 当前版本复跑实测 | 缺陷版本 2×2 独立 CWD 复现（RUN1==RUN2 RAW BYTE-IDENTICAL）+ 当前版本 sched 探针（临时 workdir，3s 采样） |
| 全部输出数字（seq/ attempts / 退避序列 [1.0, 1.1] / 成本 9=5+1+3+0 / digest `3c23229776e5f04c` / 33/33 等） | 本实现实测（toy 设定） | `L2_parallel_executor_heartbeat_and_self_heal.py` §2 paste 块（与运行输出 BYTE-IDENTICAL）；非真实云价、时长为 demo 尺度、不可外推 |
| overlap 0.29s（§2 elapsed 行） | 本实现实测（单次运行值） | 墙钟不确定量，落掩码行不进 check 路径（check 28 只要求 >0.1s），跨跑不可复现，不作锚 |
| L1 跨级锚 `ac4a0b3ac09bf47b` / L1 输出锚 `9e1bec41…`/92 行/10,139 B / 24/24 | 前级录值（当前版本复验不变） | `tutorial_L1.md` §12（前级冻结锚）；当前版本 L1 复跑逐位吻合（2026-08-16） |
| 「liveness 与结果都不走父子关系」「结果通道优先 = 与时间无关的事实定序」 | 合理推断（机制归纳） | 由收养块源码（L3266-3328）与 §9 实测归纳，非源码原文 |
| 启动宽限对照 Airflow last_heartbeat_at 入队即初始化 | 合理推断（机制同构） | job.py:L141 为心跳更新行录值；「入队即初始化」的具体赋值点在 Airflow 多处（TI 入队/executor 心跳），未逐行引文，标推断 |

**运行锚点**（2026-08-16，CPython 3.13.13 实测）：公开代码 md5 `a1246f84dca141006b94170a9e745f7f`/776 行/56,895 B（仅 docstring 脱敏，行为不变）；自跑 2 遍 × 2 新建空独立 CWD（`-B`）全 EXIT=0、stderr 0 B，掩码口径 `sed '/^[[:space:]]*elapsed/d'` 锚 `7be252c68fff9d00c7a15d0549c2de96`/260 行/25,323 B，RUN1==RUN2 MASKED BYTE-IDENTICAL（cmp 机器证明）；self-check 33/33 PASS；run [7] 干净 run digest `3c23229776e5f04c`；L1 跨级锚复验 `9e1bec41…`/92/10,139 + 24/24 + `ac4a0b3ac09bf47b` 逐位不变。
