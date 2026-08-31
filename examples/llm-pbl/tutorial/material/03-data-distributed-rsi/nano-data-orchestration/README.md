# nano-data-orchestration

> **抓的核心机制**：工作流编排、依赖调度、失败重试、自动化测试/部署、Agent 驱动的管线自愈（本课程约定）。
> L0 用纯 Python 裸出 DAG 调度器的状态机内核：**DAG = 一等公民的依赖结构**（环 / 未知依赖执行前被拒）+ **任务状态机与调和循环**（每 tick 扫描状态、施加转移规则——状态是完整记录）+ **失败语义**（transient/permanent 分类 → 指数退避有界重试 vs 立即失败；上游失败急切传播，爆炸半径 = 下游锥）+ **治理 first-class**（capability default-deny，拒绝先于计算；attempt 成本账本，重试不是免费的）。
> **对应真实系统**：[Apache Airflow](https://github.com/apache/airflow) / [Dagster](https://github.com/dagster-io/dagster) / [Prefect](https://github.com/PrefectHQ/prefect)；CI/CD 参照（GitHub Actions / GitLab CI）只作对照不锁定。
> **轨道**：[03 数据/分布式/RSI/数据平台工程](../README.md) · **状态**：L0 ✅ · L1 ✅ · L2 ✅

---

## 为什么从「状态机 + 失败语义」开始

数据平台（nano-data-platform）回答数据住在哪、谁能碰、花了多少钱；编排回答另一组问题：**什么顺序跑、失败了怎么办、失败花了多少钱**。RSI 闭环要「持续地跑」，而持续运行的系统里失败是常态——编排器的本质不是「按序调命令」，而是把状态、失败、权限、成本都变成一等公民（tutorial §10 反例）。

L0 的选择是把调度器最内核的状态机裸出来：不碰真实进程、不碰持久化、不碰 wall-clock——它们是环境，不是机制。任务用纯函数、时钟用逻辑 tick，于是全部行为确定、可复现、可审计。

L1 把这三个「环境」请回来——然后证明它们本身就是机制：真实进程带来 exit code 分类通道与幂等问题，持久化带来 stale 态与 zombie 识别，wall-clock 带来「计划确定 / 醒来不确定」的审计载体迁移（`tutorial_L1.md` §1 的两笔债）。

---

## 阶梯（L0–L2）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | single-file 玩具（191 行，纯标准库）：DAG 静态校验（环/未知依赖 fail fast）；状态机 + 调和循环（PENDING/RUNNABLE/RUNNING/RETRYING + 三终态）；错误分类 → 指数退避有界重试 vs 立即失败；上游失败急切传播（下游锥 0 成本）；capability default-deny（attempts==0）；成本账本（9=5+1+3 恒等式）；确定性 digest | ✅ `L0_dag_scheduler_state_machine.py` + `tutorial_L0.md` |
| **L1** | 任务换成真实 subprocess（exit code 分类通道：0 / 75=EX_TEMPFAIL / 其余 permanent）：状态落盘（state.json 原子写 + events.jsonl append-only，seq 以日志为源）+ 崩溃续跑（kill -9 进程组 [宿主死亡模型] → zombie 识别 → 回重试通道，已完成工作不重做）+ wall-clock 退避（计划等待是确定算术，醒来时刻落掩码行）+ 幂等正面登场（非幂等副作用重复 vs 原子发布收敛）；复现 L0 终态向量与成本恒等式，新增崩溃税（10=5+1+3+1） | ✅ `L1_subprocess_state_and_crash_recovery.py` + `tutorial_L1.md` |
| **L2** | 对照权威实现源码做取舍分析（行锚全取自一手抓取件）：Airflow（executor open_slots / pool occupied / trigger rules 分派表 / heartbeat 连续 purge / 孤儿收养 adoptable_states+skip_locked+try_adopt / Job.latest_heartbeat）+ Dagster（并发键槽位账本 + 指数步进退避）+ GHA cancel-in-progress / GitLab resource_group 参照（不锁定）；真实并行与资源池；Agentic 管线自愈（default-deny 安全边界，本课程约定）；run [7] 真进程锚（kill -9 只杀调度器 → 孤儿收养，崩溃税 0）+ **zombie 反例教材**（pid_alive 对未收割僵尸误判 → 结果通道优先 + 启动宽限，tutorial §9）；Prefect flow run 状态抓取 404 未覆盖（显式声明，tutorial §17）；可运行的本质模拟 + 显式注明 | ✅ `L2_parallel_executor_heartbeat_and_self_heal.py` + `tutorial_L2.md` |

**环境依赖分级**：L0 零依赖（纯标准库，CPU 秒级，任意 CWD 可跑，输出确定——双独立 CWD 双跑 stdout md5 `802aac9f48d5a7c81a5e61f695c8903d`/54 行 BYTE-IDENTICAL，EXIT=0、stderr 0 B）；L1 实测纯标准库（subprocess/signal/json，Python 3.13.13，CPU ~8.7s，任意 CWD）：双新建空独立 CWD 双跑全 EXIT=0、stderr 0 B，raw 98 行/10,510 B（md5 因 elapsed 行不同），掩码口径 `sed '/^[[:space:]]*elapsed/d'` 后 md5 `9e1bec41263dca2108190e0262590914`/92 行/10,139 B，RUN1==RUN2 BYTE-IDENTICAL；L2 实测纯标准库（Python 3.13.13，CPU ~5s，任意 CWD）：2 遍 × 2 新建空独立 CWD（`-B`）全 EXIT=0、stderr 0 B，掩码口径同款，锚 md5 `7be252c68fff9d00c7a15d0549c2de96`/260 行/25,323 B，RUN1==RUN2 MASKED BYTE-IDENTICAL，self-check 33/33 PASS；按本课程可运行性契约为「可运行的本质模拟 + 显式注明」（runs [1]–[6] 逻辑时钟模拟 / run [7] 真进程锚），真实集群路径标 `[TODO: verify on real system]`。

---

## L0 快速开始

```bash
python3 L0_dag_scheduler_state_machine.py
```

预期输出（toy 指标基线）：终态向量 `5 SUCCESS / 2 FAILED / 2 UPSTREAM_FAILED`（9 任务 CI/CD 风格管线，含重试救回 / 误分类止损 / default-deny 拒绝各一例）；成本账本 `总 9 = 有效 5 + 重试救回 1 + 浪费 3`（1 coin/attempt toy 单价，非真实云价）；run digest `0e0b34e0c9eb016f`（ticks=5, coins=9）；`self-check: 15/15 PASS`。逐步拆解见 `tutorial_L0.md`。

## L1 快速开始

```bash
python3 L1_subprocess_state_and_crash_recovery.py
```

预期输出（三 run，~9s）：**Run A** 干净基线——终态向量与 L0 逐字一致（5/2/2），成本恒等式 `9 = 5 + 1 + 3 + 0`（复现 L0 check 13，成本单位 = 1 次 subprocess 启动），终态向量 digest `ac4a0b3ac09bf47b`；**Run B** 崩溃续跑——kill -9 整个进程组（宿主死亡模型）后重启，zombie 识别回重试通道，终态 digest 与 Run A 相同，成本恒等式 `10 = 5 + 1 + 3 + 1`（崩溃税恰 1）；**Run C** 幂等对照——非幂等 append 重试得 2 行逐字重复，原子发布重试收敛恰 1 行；`self-check: 24/24 PASS`。逐步拆解见 `tutorial_L1.md`。

## L2 快速开始

```bash
python3 L2_parallel_executor_heartbeat_and_self_heal.py
```

预期输出（runs [1]–[7]，~5s）：**[1]** 跨级锚——L1 fixture 退化运行（parallelism=1），终态 digest 复现 L1 录值 `ac4a0b3ac09bf47b`，成本恒等式 `9 = 5 + 1 + 3 + 0`；**[2]** 并行不改变收敛点（digest 同款）+ pool 占用峰值 ≤ slots + priority 出队 + 3× 加速探针；**[3]** trigger rules 六条全谱（含 ALL_SUCCESS 被 skipped 染 SKIPPED）；**[4]** heartbeat 连续形态（stuck 杀之 / zombie 回炉）+ 逻辑层孤儿收养（attempt 不增）；**[5]** 并发键退避序列 `[1.0, 1.1]`（Dagster 公式）+ cancel-in-progress（GHA 语义）；**[6]** Agentic 自愈（P1 坏源改道 fallback + P0 capability 缺失升级人工）；**[7]** 真进程锚——三心跳共同存活窗证真实并行，kill -9 只杀调度器 → 孤儿收养（attempt 不增、崩溃税 0、终态 digest `3c23229776e5f04c`）；`self-check: 33/33 PASS`。逐步拆解（含 §9 zombie 反例教材：本级第一版实现跌进 pid_alive 陷阱的完整复现与修复）见 `tutorial_L2.md`。

---

## 费曼自检

- 能不能用「工地总调度」一段话讲清依赖承诺 / 重试退避 / 上游失败传播 / default-deny / 成本账本各自的角色？（见 `tutorial_L0.md` §10）
- 「调度器跑脚本」与「调度器调和状态」的本质区别是什么？（进度活在调用栈里 vs 活在状态记录里。）
- 为什么重试上限不只是成本控制，还是收敛保证与误分类的最后防线？
- L0 → L1，进度的「住处」从哪搬到哪？新住处带来哪两个新问题？（调用栈 → state.json/events.jsonl；stale 态 [zombie] 与窗口问题 [状态不知道的进程]，见 `tutorial_L1.md` §10。）
- 「崩溃税 = 1」为什么是精确预期而不是大致预期？（at-least-once 的机器证明：>1 = 重做了已完成工作，0 = 账本丢失。）
- 为什么 `os.kill(pid, 0)` 不是可靠的「进程活着」信号？信任它会跌进哪两个坑？（未收割 zombie 假存活 / 首跳前误判 stuck；见 `tutorial_L2.md` §9 zombie 反例教材。）
- 孤儿场景下，为什么结果文件比 exit code 更权威？（收养方不是父进程，waitpid 会 ECHILD，exit code 永久丢失——liveness 与结果都不走父子关系。）

## 权威实现与延伸

- 对标源码（L2 已展开，行锚 = 2026-08-16 抓取件录值）：apache/airflow 3.x 主线（executor open_slots / pool / trigger rules 分派 / heartbeat purge / 孤儿收养块）、dagster-io/dagster（op concurrency 槽位账本与退避）；Prefect flow run 状态抓取 404、L2 零引用（显式声明，见 `tutorial_L2.md` §17）；Airflow tasks 概念文档（airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html，状态表与 retry policy）
- CI/CD 参照（不锁定）：GitHub Actions / GitLab CI——价目/性能数字一律以官方文档为准或标 `[TODO: verify]`
- 姊妹模块：[nano-data-platform](../nano-data-platform/)（L0 fixture 的动作序列 ingest → gate → build → deploy 即其消费侧）
- 轨道：[03 数据/分布式/RSI/数据平台工程](../README.md)
