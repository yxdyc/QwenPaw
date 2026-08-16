# SOTA Deep-Dive — LLM 数据方法论（去重 / 质量过滤 / 配比 / 去污染）

> **深挖对象**：预训练数据方法论四机制面——去重（Lee et al. / FineWeb）、质量过滤（FineWeb-Edu / DCLM fastText）、数据配比（DoReMi Group DRO）、去污染（DCLM 工具哲学）（首版 ✅）；Nemotron-CC 为更新一代替代（摘要级，不作教学主体）。
> **状态**：首版完成（SOTA 对齐日期 2026-08-12）
> **可运行对照**：[nano-data-juicer L0–L3](../nano-data-juicer/) + [nano-ray](../nano-ray/) + [nano-vllm-sglang](../nano-vllm-sglang/)。

---

## 阶梯状态

| 文件 | 状态 | 说明 |
|------|------|------|
| [`data-methodology.md`](data-methodology.md) | ✅ 首版（2026-08-12） | 机制面 ×4（去重 exact+MinHash/LSH+传递聚类 / 质量过滤 启发式→分类器 / 配比 DoReMi Group DRO minimax / 去污染 n-gram 探针与 DCLM 工具哲学），每面一手来源逐字引文（46 处，2026-08-12 fresh ar5iv 逐字核验）+ sim 实测双证 + nano 实测锚交叉引用；费曼四件齐备；Nemotron-CC 定位 + 2025–26 旗舰扫描未决项标注 |
| [`data_methodology_sim.py`](data_methodology_sim.py) | ✅ 可运行锚点（2026-08-12 定版） | 四个机制面的可运行本质模拟（真实 MinHash/LSH、真实乘性权重重加权、真实 n-gram 重叠）；纯标准库、CPU 秒级、seed=3 跨运行逐字节一致、self-check 26/26；[C] 配比 nano 侧无实测锚，以本 sim 为锚（显式声明），[C3b] toy 边界显式不外推 |

## 环境依赖

- **sim**：纯标准库（hashlib / math / random），零外部依赖，CPU 秒级。
- **运行**：`python3 -B data_methodology_sim.py`（任意 CWD，`-B` 防 pycache；无计时行，跨运行逐字节一致）。
- **输出锚**：md5 `db8359c8c93ce1f18ef9b2c64ed3dc72` / 139 行 / 8,478 B；digest `41a90839b75f20fb115674c835201a4b`；sim 文件 md5 `897abf29927e18130bda9e6f94dcc72d` / 581 行 / 29,433 B（2026-08-12 双独立 CWD 复跑 BYTE-IDENTICAL）。

## 深挖什么（scope）

1. **去重**（首版已覆盖）：exact 子串（suffix array）vs MinHash 整例两路互补；LSH 分带 S 曲线与拐点 s*=(1/b)^(1/r)；FineWeb 粒度消融（URL 71.5% / 行级 77.8% / 行级+词数 85%）与 112/14×8/≥75% 参数；传递聚类；分布式下的顺序语义（first-seen vs min-row_id）。
2. **质量过滤**（首版已覆盖）：启发式分布判别（FineWeb 三阈值逐字 + Fig 8 口径）；FineWeb-Edu LLM 标注路线（6,000 H100 hours / 33.6% MMLU @38B）；DCLM fastText 路线（Table 4 八策略对比 / 416 实验 / 3.8T baseline / 7B 64% MMLU）；阈值 = 质量 vs 数据量旋钮；静默污染。
3. **数据配比**（首版已覆盖）：DoReMi Group DRO 乘性权重 minimax 三签名（损失拉平 / worst 更低 / worst 目标加速 5.45x toy）；toy 边界（[C3b] 静态权重 + 平均目标反例，论文 2.6x 为经验结果不外推）。
4. **去污染**（首版已覆盖）：n-gram 重叠探针（拷贝=1.0 / 近拷贝 0.444 / 改写≈0 盲区）；DCLM「发工具 + 披露」哲学 vs FineWeb「刻意不去污染」对照口径。
5. **后续扩展（未开写，不设 placeholder 内容）**：合成数据（self-instruct / 拒绝采样 / 教师蒸馏的工程实现与陷阱）；data-model co-dev / RSI 闭环的工程化。新增内容前需先补齐可运行锚点与来源核验。

## 信息溯源要求（反幻觉硬约束）

- 数字/结论必须来自一手来源（技术报告 arXiv / 开源代码）。
- 拿不到就标 `[TODO: verify]`，绝不凭印象写配比数字。
- 区分：原文声称 / 文献已有 / 合理推断 / 猜测（见 deepdive §8.5）。

## 来源清单（首版已核验，2026-08-12 现场重抓；arXiv 经 export.arxiv.org API [HTTP 200 / 7 entries]，论文全文经 ar5iv fresh 抓取逐字核验）

- [x] **Lee et al.** `[2107.06499]`（v2，2021-07-14 / 2022-03-24）——去重两工具原始提出；摘要四数字（>1% 逐字拷贝 / 61 词句重复 60,000 次 / 记忆输出 10× / >4% 验证集重叠）+ 方法节两工具句 + §4.2 NearDup 句逐字核验（ar5iv 253,098 B）。
- [x] **FineWeb** `[2406.17557]`（v2，2024-06-25 / 2024-10-31）——机制面规范锚点；去重粒度消融 / MinHash 参数 / 传递聚类 / 三阈值 / Fig 8 分布判别 / Edu 1.3T+6,000 H100 hours+33.6% MMLU@38B 等 16 处引文逐字核验（ar5iv 290,898 B）。
- [x] **DCLM** `[2406.11794]`（v4，2024-06-17 / 2025-04-21）——机制面规范锚点；Table 4 八策略 16 数字逐一 HIT / 416 实验 / 240T+53 evals / 3.8T baseline / 7B 64% MMLU 2.6T / MAP-Neo 6.6pp+40% / Mistral·Llama3 6.6× / 去污染四句 / 自述更新一代句逐字核验（ar5iv 1,212,788 B）。
- [x] **DoReMi** `[2305.10429]`（v4，2023-05-17 / 2023-11-21）——配比机制规范锚点；命题 / Group DRO 机制 / 280M→8B 30× / 6.5pp+2.6x / GLaM 泛化 / 同尺寸 4× 六处引文逐字核验（ar5iv 331,076 B）。
- [x] **Scaling Data-Constrained** `[2305.16264]`（v5，2023-05-25 / 2025-06-28）——A 层经典锚点；≤4 epoch 重复近无损 + 算力价值衰减归零（摘要逐字，API 通道）。
- [x] **RefinedWeb** `[2306.01116]`（v1，2023-06-01）——A 层经典锚点（标题级核验；谱系句经 FineWeb 转述核验）。
- [x] **Nemotron-CC** `[2412.02595]`（v2，2024-12-03 / 2025-05-30）——更新一代替代，存在性坐实（API 摘要六探针全 HIT：90% 数据移除批评 / 分类器集成+合成改写+弱化启发式 / MMLU +5.6 over DCLM / 6.3T 四倍 unique real tokens / 15T 训练 +5 MMLU +3.1 ARC-C）；机制细节不作教学主体（课程的前沿证据层处理）。
- [ ] 2025–2026 更新旗舰数据方法论报告：`[TODO: verify]`（submittedDate 倒序全量扫描未做，见 deepdive §6.4）；FineWeb-2 等多语单源条目 `[transient/单源]` 不展开。

## 权威实现与延伸

- 轨道 [03](../README.md)；落地参照 nano-data-juicer（OP 语义 / stats 复用 / 静默污染，L0–L3）/ nano-ray（分区 / 收敛点 / 顺序语义，L0–L2）/ nano-vllm-sglang（paged KV，L0–L2）
- 一手来源：Lee et al. `[2107.06499]` / FineWeb `[2406.17557]` / DCLM `[2406.11794]` / DoReMi `[2305.10429]`（详见 deepdive §8.1/§8.2）；Data-Juicer 开源代码（github.com/modelscope/data-juicer）
