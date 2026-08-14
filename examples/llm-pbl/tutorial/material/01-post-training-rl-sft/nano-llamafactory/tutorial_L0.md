# nano-llamafactory · L0 教程：SFT 的数据侧——chat template、loss mask 与 collator

> **本节目标（L0）**：用 209 行纯标准库的确定性实现，抓住 SFT 数据侧的最小机制：
> **模板定边界、-100 定 loss、collator 定 batch**——SFT 与预训练共用同一个
> next-token loss，唯一的差别是「哪些位置的 labels 计入 loss」，而这件事完全发生
> 在数据侧。本节还用「coasting 模型」量化了为什么没有 mask 时低 loss 是假的。
> **模块定位**：LLaMA-Factory 是全功能微调框架（SFT/DPO/PPO/评估/导出一站式），
> 本模块只取其 SFT 数据侧的最小机制——它是后续一切训练方法的地基，L2 的 DPO
> 偏好对同样用这套 template/mask 机器构造。
> **前置**：知道语言模型训练是 next-token prediction（shifted cross-entropy）、
> 知道 SFT 大概是什么。读过 nano-verl L1（PPO 训练循环）会更好衔接。
> **本节 K+1**：从「SFT 是拿问答对微调」到「说得出 SFT 的 loss 边界画在哪、
> shift 怎么对齐、pad 为什么要双层遮罩、loss 变低为什么可能是坏事」。

---

## 1. 问题：同一个 loss，不同的遮罩

预训练的训练循环一句话：在语料上算 next-token prediction 的交叉熵，**所有**
token 都计入 loss。SFT 的数据换成 (prompt, response) 对——我们希望模型学会
「怎么写回答」，而不是把梯度花在「复读问题与模板」上。

看上去这需要不同的训练代码。实际上**训练循环一个字都不用改**：还是
`logits[i]` 预测 `labels[i+1]` 的 shifted 交叉熵。唯一的差别是
**labels 里哪些位置是真实 token id、哪些位置是 `IGNORE_INDEX = -100`**——
这正是 PyTorch `nn.CrossEntropyLoss` 的默认 `ignore_index`：对应位置不进 loss、
不进梯度。

一句话：**SFT 就是「labels 被遮罩过的预训练」**。遮罩工作全部在数据侧完成，
由三件套分工：

1. **chat template**：把 messages 展开成一条模型能读的字符串，**定下 response
   从哪里开始**；
2. **loss mask**：prompt 位置 labels 置 -100，response 位置置真实 id；
3. **collator**：把变长样本 pad 成 batch——attention_mask 管注意力，labels 再
   遮掉 pad。

三件套都很琐碎，但 SFT 的常见翻车现场——loss 很低却答非所问、推理时模型不肯
停、格式漂移——几乎全能追溯到这三处某一处写错了。本节用确定性 toy 把每个边界
和不变量做成可见的。

---

## 2. 先跑起来

文件：`L0_sft_data_pipeline.py`，纯标准库，CPU 即跑。

```bash
$ python3 L0_sft_data_pipeline.py
```

真实输出（确定性 toy，连跑三遍逐字一致）：

```text
================================================================
nano-llamafactory L0 — SFT 数据侧：template / loss mask / collator
================================================================

[1] chat template：同一份 messages，训练/推理各渲染一次
    训练全文 31 tok | 推理 prompt 23 tok | Q: What is the capital of France?
    训练全文 33 tok | 推理 prompt 24 tok | Q: How many hours are in a day?
    训练全文 27 tok | 推理 prompt 20 tok | Q: Who wrote Hamlet?
    推理时模型只见到左前缀（到 <|assistant|>\n 为止），response 由它自己续写

[2] labels 构造：prompt 全遮（-100），response 进 loss（含结束符）
    例 1: input_ids=31 tok | IGNORE=23（prompt）| 监督=8（response+<|eot|>）
    边界: logits[22]（token '\n'）预测 labels[23] = 'The'
          —— 第一个 response token 由「最后一个 prompt 位置」的 logits 监督

[3] loss mask 的作用：coasting 模型下，unmasked 的低 loss 是假的
    unmasked（预训练式，88 位置）平均 NLL = 2.7447 bit
    masked  （SFT，仅 24 个 response 位置）平均 NLL = 7.7399 bit
    loss 总和构成: prompt 区   55.77 bit (23%) | response 区  185.76 bit (77%)
    token 占比: prompt 区 67/91 = 74%

[4] collator：变长样本 pad 成 batch（pad 同时被两层遮罩）
    input_ids     1 35 17 19 18 25 20 35  4 35  2 35 15 28 31 22 30  7 35  4 35  3 35 13 22 30  6 28 11 35  4  0  0  1 35 17 19 18 25 20 35  4 35  2 35 10 29 26 19 27 18 24 35  4 35  3 35 14 19  5 26 27 18 23 35  4  1 35 17 19 18 25 20 35  4 35  2 35 16 34  9 35  4 35  3 35  8 32 33 21 12 35  4  0  0  0  0  0  0
    attn_mask     1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0  0  0  0  0  0
    labels       .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. 13 22 30  6 28 11 35  4 .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. 14 19  5 26 27 18 23 35  4 .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. ..  8 32 33 21 12 35  4 .. .. .. .. .. ..
    batch loss（token 级 pooled）= 7.7399 bit | 逐样本加权平均 = 7.7399 bit | 有效 token = 24

[5] 反例：两种常见遮罩错误
    a) pad 漏进 labels（labels=input_ids）: loss 7.7399 -> 3.1213 bit（loss 反而变低是假象：pad 可预测、稀释了均值，模型还在学续写 <pad>）
    b) mask 边界多退一格: labels 位置 [23] 的监督丢失 —— 丢的正是第一个 response token 'The'（答案开头没人教）

================================================================
✅ self-check passed: 模板前缀 / mask 边界 / 稀释方向 / batch loss 不变量 / 两种反例
================================================================

takeaway: SFT 与预训练共用同一个 next-token loss，差别只在 labels 的遮罩：
          模板定边界（<|assistant|>\n 之后才算 response）、-100 定 loss、collator 定 batch。
          vocab=36，全部数字由确定性 toy 模型现场算出。
```

> **toy 口径声明**：tokenizer 是 word 级正则切分、模板是 ChatML 风格的玩具、
> 「coasting 模型」是 bigram 计数模型——全部确定性玩具，只为把 mask 的作用
> **量化**出来；真实 tokenizer / Jinja 模板 / 梯度下降在 L1 接入。本节所有数字
> 都是这个脚本的算术输出，不是 benchmark。与 nano-megatron L0「用账本代替实测」
> 的口径相同。

**L0 基线指标（toy metric）**：在 coasting 模型下，unmasked 平均 NLL
2.7447 bit vs masked 平均 NLL 7.7399 bit；prompt 区占 74% 的 token，却只贡献
23% 的 NLL 总量——mask 把训练信号从模板噪声压缩到 response 上，masked 口径
揭示的真实难度是表面数字的 7.7399/2.7447 ≈ 2.8 倍，这就是本节的量化基线。

---

## 3. 机制一：chat template——训练渲染与推理渲染是同一条字符串的两半

```python
def apply_chat_template(system, user, assistant=None):
    """ChatML 风格的 toy 模板。assistant=None 即推理用的 generation prompt。"""
    text = f"{SYS}\n{system}\n{EOT}\n{USR}\n{user}\n{EOT}\n{ASST}\n"
    if assistant is not None:
        text += f"{assistant}\n{EOT}"
    return text
```

同一份 messages 渲染两次：**训练渲染**带 assistant 回答（`assistant` 参数非空），
**推理渲染**到 `<|assistant|>\n` 为止停住——这半条串叫 generation prompt，
模型从这里开始续写。[1] 的输出给出两种渲染的长度：31/23、33/24、27/20。

真正的要点是循环里那条 assert：

```python
prompt_text = apply_chat_template(SYSTEM, q)          # add_generation_prompt=True
assert prompt_text == full[:len(prompt_text)]         # 推理 prompt 必须是训练串的真前缀
```

**推理时模型见到的 prompt，必须是训练时见过的字符串的真前缀。**模板的全部正确性
就压在这一条不变量上：如果训练用 A 模板、推理用 B 模板（或者推理时少了一个
换行、多了一个空格），模型在推理时看到的 token 序列就是它训练分布之外的东西——
train/test mismatch，格式漂移与能力跳水大多从这里开始。真实系统里用 HF
`tokenizer.apply_chat_template(..., add_generation_prompt=...)` 或 Jinja 模板，
机制完全一样：**一份模板、两种渲染、前缀关系**。

---

## 4. 机制二：loss mask——边界画在 labels 空间，shift 发生在 loss 里

```python
IGNORE_INDEX = -100          # PyTorch nn.CrossEntropyLoss 的默认 ignore_index

def build_labels(prompt_ids, response_ids):
    """HF/LlamaFactory 约定（对齐 supervised.py:L109 注释）：
    input_ids = X Y，labels = <ignore>...<ignore> Y，真实训练时 logits[i] 对 labels[i+1]。"""
    return [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)
```

构造简单到可疑，但里面藏着 SFT 最容易错的一刀：**边界画在哪里**。loss 的配对是
shifted 的——`logits[i]` 预测的是 `labels[i+1]`。看 [2] 的输出逐字读：

- 例 1 共 31 个 token：前 23 个（system + user 两整段加 `<|assistant|>\n`）是
  prompt，labels 全为 -100；后 8 个（response 正文、其后的 `\n` 与 `<|eot|>`）
  进 loss。
- **边界**：`logits[22]`（assistant 标记后的那个 `\n`，prompt 的最后一个位置）
  预测 `labels[23] = 'The'`——**第一个 response token 是由最后一个 prompt
  位置的 logits 监督的**。所以遮罩边界画在 labels 空间（被预测端）：
  `labels[k+1]` 是第一个计入 loss 的位置，`logits[k]` 就是监督它的预测端。
- `assert labels0[-1] != IGNORE_INDEX`：结束符 `<|eot|>` 也进 loss——模型必须
  学会「何时停」。漏掉 EOS 监督的模型推理时容易写个没完（真实系统里 EOS 同样
  进 labels，见 §6 的 `efficient_eos`）。

把边界画错一格会丢什么，见 §8 反例 b。

### 4.1 mask 的作用量化：coasting 实验

知道「mask 遮住 prompt」不够，要能说出**不遮会怎样**。[3] 构造了一个
**coasting 模型**：bigram 只在三条 prompt 的模板文本上统计（重复 10 遍），
**从没见过任何一个 response 词**。然后拿它评估两种 loss 口径：

- **unmasked（预训练式，88 个位置）**：平均 NLL 2.7447 bit——单看这个数字，
  像个训练得不错的模型；
- **masked（SFT，仅 24 个 response 位置）**：平均 NLL 7.7399 bit——同一个
  模型，难度立刻现出原形。

拆 loss 总和看得更清楚：prompt 区 67/91 = **74% 的 token 只贡献了 23% 的 NLL
总量**（模板文本被 coasting 模型背得滚瓜烂熟），response 区 26% 的 token 贡献了
77%。**不遮罩时，loss 均值被模板稀释**：梯度大头去学「续写 You are a helpful
assistant」，学回答的信号被淹没。mask 的本质作用就是**把梯度聚焦到你想让模型学
的那部分序列上**。

推论（也是思考题 3 的引子）：**loss 数字只有带上 mask 口径才有意义**。拿
unmasked loss 跨实验比较、或者拿 unmasked 的「好看数字」汇报 SFT 效果，都是
拿尺子量身高却读成体重。

---

## 5. 机制三：collator——pad 被双层遮罩，batch loss 不变

```python
def collate(samples, vocab):
    """samples: (input_ids, labels)。右 pad；attention_mask 遮 pad 的注意力，labels 遮 pad 的 loss。"""
    pid = vocab.index(PAD)
    L = max(len(ids) for ids, _ in samples)
    batch = []
    for ids, labels in samples:
        n = L - len(ids)
        batch.append((ids + [pid] * n,
                      [1] * len(ids) + [0] * n,
                      labels + [IGNORE_INDEX] * n))
    return batch
```

变长样本进 batch 必须 pad 到同一长度（本例 L=33），pad 会污染两处，所以要
**双层遮罩**：

- **前向**：`attention_mask` 在 pad 位置置 0——其它 token 不该把注意力花在
  `<pad>` 上；
- **反向**：`labels` 在 pad 位置置 -100——pad 不该进 loss。

[4] 的表格把三条样本横排打印（长度 31/33/27，各自 pad 到 33），对照读：
`input_ids` 里 id 0 是 `<pad>`、35 是 `\n`、1–4 是四个角色标记，样本 1 解码出来
正是 `<|system|>\nYou are a helpful assistant.\n<|eot|>\n...`；`attn_mask` 只在
两个样本的 pad 区出现 0；`labels` 的 `..` 全是 -100，唯一的非零监督段就是三条
response（含各自的 `<|eot|>`：id 4 出现在每段监督的末尾）。

不变量在最后两行：**batch loss（token 级 pooled）= 逐样本加权平均 = 7.7399**。
正确的 collate **不改变 loss 的数值，只改变它的形状**——把逐样本的账合成 batch
的账，有效 token 一个不多一个不少（24 个，正是 [3] 的 24 个 response 位置）。
这条 assert 抓的是实现层的走样：shift 对齐错一格、pad 的 labels 忘遮、
某个样本被重复计入，数值立刻对不上。HF 的口径正是 token 级 pooling
（`DataCollatorForSeq2Seq`，见 §6），注意它**不等于**「逐样本 loss 取简单平均」
——两者在 response 长度不齐时给出不同的梯度权重，见思考题 3。

---

## 6. 与权威实现的对应：LLaMA-Factory 源码锚点

LLaMA-Factory（`github.com/hiyouga/LLaMA-Factory`）的 SFT 数据侧与本节逐条
同构。以下行号全部为 main 分支现场抓取实测（2026-08-05 快照，上游迭代可能
漂移）：

| nano 实现 | LLaMA-Factory（main，2026-08-05 快照） | 说明 |
|-----------|----------------------------------------|------|
| `apply_chat_template` 两种渲染 | `data/template.py`：`_encode` L132 把格式化输入编码成 token 段、`encode_oneturn` L60 返回 (prompt_ids, response_ids) 对；`add_generation_prompt=True` 的推理渲染语义见 L631/635 | 一份模板、两种渲染、前缀关系 |
| `build_labels`（prompt 全 -100） | `data/processor/supervised.py:L109` 注释原文：*"build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`"*；实现 L88 `source_label = [IGNORE_INDEX] * source_len` | 与本节完全同一约定 |
| `IGNORE_INDEX = -100` | `extras/constants.py:L50`；与 PyTorch `nn.CrossEntropyLoss` 默认 `ignore_index` 一致 | 哨兵值取负数，不会与任何真实 token id 相撞 |
| `collate`（pad + attention_mask + labels -100） | `data/collator.py`：L137 `MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq)`、L492 `SFTDataCollatorWith4DAttentionMask`；labels pad 位填 -100 由 transformers 的 `DataCollatorForSeq2Seq` 承担（`transformers/data/data_collator.py:L487`，`label_pad_token_id = -100` 在 L526，main 分支 2026-08-05 实测） | 双层遮罩一致 |
| 结束符进 loss | `supervised.py:L102–104`（`efficient_eos` 时 input_ids 与 labels 同时追加 eos） | 学「何时停」 |
| 未覆盖：train_on_prompt / mask_history / packing / 4D attention mask / DPO pairwise collator | `supervised.py:L83–97`（`train_on_prompt` L83–84、`mask_history` L90–97）；`collator.py:L492`（4D mask）、L553 `PairwiseDataCollatorWithPadding` | L1/L2/L3 主题 `[TODO: verify source]` |

两点值得提前知道（思考题素材，都是真实存在的开关）：

- **`train_on_prompt`**（supervised.py:L83–84）：置 True 时 `source_label =
  source_ids`——prompt 也进 loss，退回预训练式口径；
- **`mask_history`**（supervised.py:L90–97）：多轮对话只训最后一轮，历史轮的
  response 也遮掉。

---

## 7. 费曼：讲给外行听

**类比：批改答题卡。**

- **chat template = 答题卡的印刷版式**：题目印在哪、作答区从哪条线开始，版式
  说了算。考试时（推理）发给学生的是**印到作答区为止**的半张卡（generation
  prompt），批改时（训练）用的是**带答案的整张卡**——两者必须是同一版式，
  学生才认得；
- **loss mask = 阅卷只批作答区**：印刷的题干再长也不给分。-100 就是阅卷机
  的「此区域不批」标记；
- **collator = 把长短不一的答题卡摞成一摞统一批**：空白补位区（pad）既不参与
  阅读（attention_mask），也不参与给分（labels 遮罩）。

类比反例版：如果**印刷的题干也算分**，全班分数最高的会是把题干抄得最熟的
学生——这正是 [3] 的 coasting 实验：没学过任何答案的模型，unmasked loss
（2.7447）反而比 masked（7.7399）低。

一句话版本：**SFT 没有换考试，只是换了阅卷范围——版式定作答区在哪，
-100 定哪些区域给分，collator 管把卷子摞齐。**

（类比边界：真实 mask 不是阅卷人的主观取舍，是 loss 函数 ignore_index 的硬规则；
「分数」这里是负对数似然，越低越好。）

---

## 8. 反例：两种遮罩错误与一种读法错误

- **a) pad 漏进 labels（实测）**：把 labels 直接取成 input_ids（pad 也进 loss），
  loss 从 7.7399 掉到 **3.1213 bit**。变低不是变好：`<pad>` 在 pad 区之后还是
  `<pad>`，极易预测，把均值稀释下来了——模型同时在学「续写 <pad>」。
  教训：**loss 下降得「过于轻松」时，先查 labels 里混进了什么**。
- **b) 边界多退一格（实测）**：把遮罩边界从 `len(prompt)` 写成
  `len(prompt)+1`，labels 位置 [23] 的监督丢失——丢的恰是第一个 response
  token `'The'`。根因是混淆了两个空间：**shift 发生在 loss 计算时
  （logits[i] 对 labels[i+1]），labels 构造时只管按被预测端画边界**。
  多想退一格「补偿 shift」，答案的开头就没人教了。
- **c) 拿 unmasked loss 评价 SFT 模型（[3] 背书）**：一个从没见过 response 的
  coasting 模型，unmasked NLL 2.7447 比它自己的 masked NLL 7.7399 还低——
  **低 loss ≠ 会回答**。loss 数字只有在同一 mask 口径下才可比较。

---

## 9. 思考题

1. **多轮只训最后一轮**：LLaMA-Factory 的 `mask_history`（supervised.py:
   L90–97）把历史轮的 response 也遮掉，只让最后一轮进 loss。这样做得失是什么？
   （提示：历史轮可能来自旧 policy 或包含工具调用结果——它们的分布和你现在
   想训的行为一致吗？遮掉后有效 token 变少，batch 内 loss 的方差会怎么变？）
2. **什么时候想让 prompt 也进 loss**：`train_on_prompt=True`
   （supervised.py:L83–84）退回预训练式口径。设想一类任务，让「prompt 也计入
   loss」是**正确**选择。（提示：你想要的模型能力是「续写整段文本」本身，
   而非「对输入做出回答」时——比如什么场景？代价仍是 §4.1 的稀释。）
3. **loss 的汇总口径**：本节用 token 级 pooling（HF 默认），代码验证了它等于
   逐样本均值按有效 token 数加权。如果改成「逐样本 loss 简单平均」（不加权），
   当 batch 里有 8 token 的短回答与 500 token 的长推理时，两种口径各把梯度
   偏向谁？长 response 的 SFT（如 CoT 数据）该选哪种？（提示：算两种口径下
   单样本的有效权重——token 级是 n_i/Σn，简单平均是 1/B。）

---

## 10. 下一步 L1

L1 把三个玩具换成真的：真实小模型 + torch 梯度下降，在本节构造的
(input_ids, attention_mask, labels) 上跑一个最小 SFT 循环——验证本节的断言：
labels 遮罩不动训练循环分毫，却完全决定模型学到什么。数据用内置微样本起步
（一键可跑），再换真实小样本；观察训练前后生成行为的变化与 loss 曲线。
L2 进入 DPO：偏好对数据用同一套 template/mask 机器构造（真实入口
`collator.py:L553` `PairwiseDataCollatorWithPadding`），加 reference-model KL
约束——按 ROADMAP §八，DPO 属 A 层经典锚点，届时须写明其当今定位。
L3 对照 LLaMA-Factory 的配置体系：一个配置切换 SFT/DPO/PPO 的抽象取舍。

---

## 11. 溯源

- **运行输出**：本机真实执行 `python3
  L0_sft_data_pipeline.py`，确定性 toy，连跑三遍逐字一致；§2 粘贴与程序输出
  经 awk 提取 + diff 机器核对。
- **LLaMA-Factory**：`github.com/hiyouga/LLaMA-Factory`，main 分支源文件于
  2026-08-05 经 raw.githubusercontent.com 现场抓取（本机对 github.com 的 HEAD
  请求超时，源文件成功抓取即可达性实证；行号 2026-08-04 初测、2026-08-05
  复测一致）：
  - `src/llamafactory/extras/constants.py:L50`：`IGNORE_INDEX = -100`；
  - `src/llamafactory/data/processor/supervised.py`：L109 `<bos> X Y <eos>` /
    `<ignore> ... <ignore> Y <eos>` 注释原文；L83–88 train_on_prompt 与
    IGNORE 遮罩；L90–97 mask_history；L102–104 efficient_eos；
  - `src/llamafactory/data/collator.py`：L137 / L492 / L553（SFT 与 pairwise
    collator）；
  - `src/llamafactory/data/template.py`：L60 `encode_oneturn`、L132 `_encode`、
    L631/635 `add_generation_prompt` 用法。
  行号均为主分支快照，上游迭代可能漂移。
- **HF transformers**（main，2026-08-05 抓取）：`src/transformers/data/
  data_collator.py:L487` `class DataCollatorForSeq2Seq`，L526
  `label_pad_token_id: int = -100`。
- **PyTorch**：`nn.CrossEntropyLoss` 默认 `ignore_index=-100`（PyTorch 文档
  标准行为）。
- 全部 NLL/占比数字为确定性 toy 脚本的算术输出（§2 toy 口径声明），非 benchmark。
- 概念交叉引用：nano-verl L1（SFT warmup → PPO 的顺序）、nano-slime L0
  （RL 数据通路）——均为本仓库已交付材料。
