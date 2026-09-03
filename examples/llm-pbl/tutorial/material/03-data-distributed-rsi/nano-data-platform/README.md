# nano-data-platform

> **抓的核心机制**：数据接入 → 分层存储（raw/curated）→ 治理（成本/权限/质量）→ 训练/检索消费（课程的数据系统教学约定）。
> L0 用纯 Python 裸出两个本质：**湖仓分层**（raw 不可变追加 + curated 版本化快照派生 + 质量门硬晋升）与 **infra-as-code 状态管理**（声明式 desired state + plan/apply + state 文件，幂等且最小 diff）。
> L1 还清 L0 的两笔债（重启态蒸发 + 每次全量重算）：**watermark 增量接入**（单事务游标推进 + PK 去重 = exactly-once 物化）、**持久化 catalog**（SQLite 五表，重启幸存 + 血缘 SQL）、**增量物化**（merge/upsert 语义）、**双层对账**（raw 重放 == curated 账目 + 源↔raw 对账，跨级 digest 锚字节级复现 L0 漏斗）。
> **对应真实系统**：[Apache Iceberg](https://github.com/apache/iceberg) / [Delta Lake](https://github.com/delta-io/delta) / [dbt](https://github.com/dbt-labs/dbt-core)；Terraform（HCL 到 L2 才触及）；云原生参照（AWS Glue / Redshift / Snowflake / ClickHouse / Airbyte / Fivetran）只作对照不锁定。
> **轨道**：[03 数据/分布式/RSI/数据平台工程](../README.md) · **状态**：L0–L2 ✅

---

## 为什么从「分层 + 声明式」开始

清洗算子（nano-data-juicer）回答「数据怎么处理」，数据平台回答另一组问题：数据**从哪来、存在哪、谁能碰、花了多少钱、训练用的是哪一版**。这组问题不解决，RSI 闭环在生产环境就是空转——接不进、不可复现、管不住、算不清。

L0 的选择是把「分层契约」和「声明式状态管理」这两个最本质的机制裸出来：不碰 HCL、不碰真实存储、不碰云 API——它们是语法和环境，不是机制。

---

## 阶梯（L0–L2）

| 级别 | 目标 | 状态 |
|------|------|------|
| **L0** | single-file 玩具（200 行，纯标准库）：raw 不可变追加 + 血缘；质量门硬晋升 + PII 投影；curated 版本化快照 + 钉住消费；声明式 plan/apply + state 文件（幂等/最小 diff）；secrets 认证 + default-deny 最小权限 + 相对成本账本 | ✅ `L0_lakehouse_and_iac_state.py` + `tutorial_L0.md` |
| **L1** | 接真实小数据集（公开小数据集或本地样本）：增量接入（watermark/游标）+ 增量物化 build + 持久化 catalog（SQLite/DuckDB 级）+ 全量回测对账（raw 重放 == curated 账目），复现 L0 漏斗语义 | ✅ `L1_incremental_sync_catalog.py` + `tutorial_L1.md` |
| **L2** | 对照权威实现源码做取舍分析：Iceberg/Delta 的 snapshot/manifest/commit protocol（乐观并发、time travel、schema evolution）+ dbt 的分层派生模型；Terraform HCL/provider/state locking 教学子集；可运行的本质模拟 + 显式注明 | ✅ [代码](L2_commit_protocol_schema_evolution.py) · [教程](tutorial_L2.md) |

**环境依赖分级**：L0 零依赖（纯标准库，CPU 秒级，任意 CWD 可跑，输出确定——双跑 stdout md5 `3ee6512a5f1b5c773357696ab6d7f137`/48 行 BYTE-IDENTICAL）；L1 零依赖（纯标准库 `sqlite3`，实测 Python 3.13.13 / SQLite 3.53.1，任意 CWD 可跑——双跑 stdout md5 `b02aad91a525ad34d72168f46f916477`/73 行 BYTE-IDENTICAL，self-check 25/25）；L2 按可运行性契约（课程可运行性契约）允许「可运行的本质模拟 + 显式注明」，真实集群路径标 `[TODO: verify on real system]`。

---

## L0 快速开始

```bash
python3 L0_lakehouse_and_iac_state.py
```

预期输出（toy 指标基线）：质量门漏斗 `9 → 6`（v1）与 `12 → 8`（v2，含跨源重复拦截）；成本账本 `raw 1161 B ×1.0 + curated 1021 B ×3.0 = 4224 toy-coins/月`（toy 单价，非真实云价）；platform state digest `112fb2c779d3f592`；`self-check: 13/13 PASS`。逐步拆解见 `tutorial_L0.md`。

---

## L1 快速开始

```bash
python3 L1_incremental_sync_catalog.py
```

预期输出（toy 指标基线）：v1/v2 与 L0 全量重算**字节级一致**（跨级 digest 锚 `4599c15439c026c8` / `a12337250f5d4d79`）；wave3 增量物化 v3 = 9 条（update 原位覆写，digest `8e60d023ac8e576d`）；v3 增量扫描仅 209/1370 B（15.3%）；updated_at 游标盲区探针：回填 update 被静默漏掉 → 外层对账捕获 `['t002']` → 回补自愈；catalog digest `1ad07870b421fcf3`；`self-check: 25/25 PASS`。逐步拆解见 `tutorial_L1.md`。

---

## 费曼自检

- 能不能用「中央厨房」一段话讲清 raw/curated/质量门/最小权限/声明式配置各自的角色？（见 `tutorial_L0.md` §10）
- 「curated 是 raw 的派生物」改成「curated 是独立维护的另一份数据」，会坏掉哪些机制？
- 为什么训练任务必须显式钉住数据版本号，而不是「读最新」？
- 能不能用「书店进销存 + 月度对账」讲清 watermark/持久化 catalog/双层对账各自的角色？（见 `tutorial_L1.md` §10）
- 「游标字段选对了，增量同步就永远不需要全量对账」——用 `tutorial_L1.md` §8 的探针反驳这句话。
- 内层对账（raw 重放 == curated）全绿但数据照样错，给出一个具体场景。（要点见 `tutorial_L1.md` §7/§8：接入层漏事件时 raw 自己就缺，重放当然自洽。）

## 权威实现与延伸

- 对标源码（L2 展开）：apache/iceberg（snapshot/manifest/commit）、delta-io/delta、dbt-labs/dbt-core；Terraform state 文档（developer.hashicorp.com/terraform/language/state）
- 云厂商参照（不锁定）：AWS Glue / Redshift / Snowflake / ClickHouse / Airbyte / Fivetran——价目/性能数字一律以官方文档为准或标 `[TODO: verify]`
- 轨道：[03 数据/分布式/RSI/数据平台工程](../README.md)
