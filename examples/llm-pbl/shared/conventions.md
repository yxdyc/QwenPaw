# shared — 跨轨共用约定

> 四条轨道共用的环境约定、术语表、评测基线。新增或修改材料时统一遵循，避免各轨各说各话。

---

## 环境约定

- **Python**：示例默认 Python 3.10+。L0 玩具实现**零外部依赖**（纯标准库 + numpy 可选），CPU 即跑。
- **GPU 相关**：L1+ 涉及 GPU 的内容，须注明显存估算与最低卡型；拿不到 GPU 验证时标 `[TODO: verify on GPU]`。
- **真实小样本**：用真实小样本数据（公开小数据集或本地样本，如一份 10 条医学 SFT MCQ），不造假数据。
- **LLM API**：涉及真实模型调用时，用环境变量传 key，**绝不硬编码 key**。
- **可移植路径**：教程命令默认使用 `python3`；外部源码 checkout 用
  `DATA_JUICER_REPO` 等显式环境变量；外部数据用 `--data` 或
  `LLM_PBL_DATA_PATH` 传入。文档和代码不得写入维护者用户名、主机地址或绝对工作目录。

## 术语表（统一用词）

| 术语 | 含义 | 注意 |
|------|------|------|
| OP（算子） | 数据处理的最小可组合单元 | Data-Juicer 语境 |
| rollout | RL 中采样出的轨迹 | 区别于「部署上线」 |
| commit | 事务语义：最终确认状态变更 | 取数据库事务义，非 git 义 |
| harness | agent 脚手架（prompt/工具/记忆/自检） | 非「测试框架」义 |
| RSI | recursive self-improvement，数据-模型协同迭代 | 轨道 03 核心 |
| CPT | continual pre-training，继续预训练 | 区别于 SFT |

## 评测基线约定

- 每个 nano-* 的 L0 须给出一个**可量化的 toy 指标**（如条数变化、loss 下降、吞吐 tokens/s），作为后续级别的对照基线。
- benchmark 分数必须可溯源；无法验证标 `[TODO: verify]`。

## 文件命名

- 阶梯实现：`L0_*.py` / `L1_*.py` / `L2_*.py` / `L3_*.py`（single-file 优先）。
- notebook-style 教程：`tutorial_*.md`（叙述 + 代码块 + 输出 + 思考题交替）。
- SOTA 深挖：`sota-deepdive/<model-or-topic>.md`。
