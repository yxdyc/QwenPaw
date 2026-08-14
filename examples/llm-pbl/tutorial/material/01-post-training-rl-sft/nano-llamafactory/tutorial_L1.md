# nano-llamafactory L1 — 真实 torch SFT：labels 遮罩决定模型学到什么

> **核心机制**：在 L0 构造的 (input_ids, attention_mask, labels) 三件套上，用真实
> `torch` 梯度下降训练一个极小 Transformer，验证「SFT 与预训练是同一个 next-token
> loss，只是 labels 被遮罩」——以及遮罩边界画错一个 token，模型就答不出来。
>
> **运行要求**：`torch`（CPU 即可）。本机命令：
> ```bash
> python L1_minimal_sft.py
> ```
> 真实生产用 GPU/大模型，机制相同。

---

## 1. 本节目标

L0 用一个确定性 toy 展示了 chat template、loss mask 和 collator 的语义。
L1 把 coasting bigram 换成真实可训练的 Transformer：

- 复用同一套 `(input_ids, attention_mask, labels)`；
- 用 `nn.TransformerEncoderLayer` + 可学习位置编码 + causal mask 建一个约 75K 参数的 TinyLM；
- 训练前后用 greedy generation 看行为变化；
- 复现 L0 的 **off-by-one 边界错误**：第一个 response token 被遮掉时，loss 可以同样低，但模型根本不会开始回答。

---

## 2. 跑代码

```bash
python \
  tutorial/material/01-post-training-rl-sft/nano-llamafactory/L1_minimal_sft.py
```

输出：

```
================================================================
nano-llamafactory L1 — real torch SFT on a tiny Transformer
================================================================
vocab size = 36
model params = 75,776

[1] data: per-sample token counts (prompt ignored / response+<|eot|> supervised)
    sample 0: prompt 23 tok / response+<|eot|> 8 tok
    sample 1: prompt 24 tok / response+<|eot|> 9 tok
    sample 2: prompt 20 tok / response+<|eot|> 7 tok

[2] masked SFT (prompt ignored, response supervised)
    initial loss = 3.7406  ->  final loss = 0.0003
    before training:
      Q: What is the capital of France?
         -> 'hoursareinthe<|user|>byofhoursFranceHowThereWhoHamlet?assistant.Theof<|user|>assistant.hoursof'  ❌
      Q: How many hours are in a day?
         -> '<|assistant|>is\nwrote24hoursHowHamlet?24day.France?ofassistant.TheofShakespeare.assistant.hoursofhelpful'  ❌
      Q: Who wrote Hamlet?
         -> 'TheofFrance?hoursareinthe<|user|>byofhoursFranceFrance?ofhelpfulFranceassistant.Theof<|user|>'  ❌
    after training:
      Q: What is the capital of France?
         -> 'ThecapitalofFranceisParis.\n<|eot|>'  ✅
      Q: How many hours are in a day?
         -> 'Thereare24hoursinaday.\n<|eot|>'  ✅
      Q: Who wrote Hamlet?
         -> 'HamletwaswrittenbyShakespeare.\n<|eot|>'  ✅

[3] ablation: mask boundary off-by-one
    final loss = 0.0003  (masked final = 0.0003)
    off-by-one generation:
      Q: What is the capital of France?
         -> '<|eot|>'  ❌
      Q: How many hours are in a day?
         -> '<|eot|>'  ❌
      Q: Who wrote Hamlet?
         -> '<|eot|>'  ❌
    first expected response tokens = ['The', 'There', 'Hamlet']

================================================================
✅ self-check passed: masked loss drops / answers emerge / off-by-one drops first token
```

---

## 3. 输出解读

### 3.1 数据侧三件套不变

和 L0 完全一致：

```python
full = apply_chat_template(system, user, assistant)   # 训练串
prompt = apply_chat_template(system, user)             # 推理串（必须是 full 的真前缀）
input_ids = tokenize(full)
labels = [-100] * len(prompt_ids) + response_ids       # prompt 不算 loss
```

三条样本分别为 prompt 23 / 24 / 20 tok + response/`eot` 8 / 9 / 7 tok（输出 [1] 逐条实测）。
模型学到的不是「这段话怎么念」，而是「在 assistant 该说话的位置，接下来该出现什么」。

### 3.2 masked SFT：loss 低 + 生成对

- initial loss ≈ 3.74（随机猜测 36 类）；
- final loss ≈ 0.0003；
- greedy 生成把三个问题的答案完整复现，并自己补上 `<|eot|>`。

这说明：只要 labels 遮罩正确，即使模型只有 75K 参数、训练 400 步，也能把三条知识背下来。

### 3.3 off-by-one：loss 同样低，但答案没了

把 labels 整体再往前多遮一格：

```python
def build_labels_off_by_one(prompt_ids, response_ids):
    return [IGNORE_INDEX] * (len(prompt_ids) + 1) + list(response_ids[1:])
```

结果第一个 response token（`The` / `There` / `Hamlet`）没人监督。
训练 loss 仍然降到 0.0003，但生成时模型直接输出 `<|eot|>`——它从来没被教过该怎么开头。

这是 L0 确定性 toy 的真实版：**边界错误在神经网络里不会被“自动修复”，而是被精确复制**。

---

## 4. 代码结构

### 4.1 chat template：推理 prompt 必须是训练串真前缀

```python
def apply_chat_template(system, user, assistant=None):
    text = f"{SYS}\n{system}\n{EOT}\n{USR}\n{user}\n{EOT}\n{ASST}\n"
    if assistant is not None:
        text += f"{assistant}\n{EOT}"
    return text
```

`assistant=None` 时生成推理 prompt，它是训练串去掉 response 后的前缀。
如果训练和推理的 prompt 不一致，模型学到的条件分布就错位。

### 4.2 labels 遮罩：prompt 全 -100，response（含 eos）进 loss

```python
def build_labels(prompt_ids, response_ids):
    return [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)
```

`-100` 是 PyTorch `CrossEntropyLoss.ignore_index` 的默认值，也是
LLaMA-Factory / HuggingFace 的约定。

### 4.3 collator：双层遮罩

```python
def collate(samples, vocab):
    pid = vocab.index(PAD)
    L = max(len(ids) for ids, _ in samples)
    batch_ids, batch_am, batch_labels = [], [], []
    for ids, labels in samples:
        n = L - len(ids)
        batch_ids.append(ids + [pid] * n)
        batch_am.append([1] * len(ids) + [0] * n)
        batch_labels.append(labels + [IGNORE_INDEX] * n)
    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(batch_am, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )
```

- `attention_mask` 告诉模型哪里是 pad，前向不要看；
- `labels` 告诉 loss 哪里不该算梯度，反向不要学。

两层语义不同，不能互相替代。

### 4.4 TinyLM：真实但足够小

```python
class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2,
                 num_layers=2, dim_feedforward=128, max_len=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        key_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(
            torch.ones((L, L), dtype=torch.bool, device=input_ids.device),
            diagonal=1)
        h = self.blocks(h, mask=causal_mask, src_key_padding_mask=key_mask)
        return self.head(self.norm(h))
```

- 可学习位置编码（LLaMA/RoPE 之前的经典做法，原理相同：让模型知道 token 在序列中的位置）；
- `causal_mask` 保证位置 i 只能看到 `<= i`；
- `src_key_padding_mask` 让 pad 不参与 attention。

### 4.5 shifted CE：SFT 和预训练共享同一个 loss

```python
def compute_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX)
```

`logits[:, i, :]` 预测 `labels[:, i + 1]`。SFT 和 CPT 的区别不在 loss 函数，
而在 `labels` 里哪些位置被写成 `-100`。

### 4.6 greedy generation 验证训练效果

```python
@torch.no_grad()
def generate(model, prompt_ids, eos_id, max_new_tokens=20):
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long)
        am = torch.ones_like(x)
        logits = model(x, attention_mask=am)
        nxt = int(logits[0, -1].argmax().item())
        ids.append(nxt)
        if nxt == eos_id:
            break
    return ids
```

greedy 解码不是生产用法（真实场景用 sampling/top-p/beam），但足够验证模型
是否学会了条件分布的峰值。

---

## 5. 与权威实现的对应关系

| nano-llamafactory L1 | 真实 LLaMA-Factory / PyTorch / transformers |
|----------------------|---------------------------------------------|
| `IGNORE_INDEX = -100` | LLaMA-Factory `extras/constants.py`；PyTorch `CrossEntropyLoss.ignore_index` 默认值 |
| prompt 不计算 loss，response + eos 计算 loss | `data/processor/supervised.py` 中 `<bos> X Y <eos>` → `<ignore>...<ignore> Y <eos>` 的 label 构造 |
| `attention_mask` 管前向、`labels` 管反向 | `data/collator.py` / HuggingFace `DataCollatorForSeq2Seq` 双层 pad/mask 约定 |
| shifted cross-entropy | causal LM 默认 loss `ForCausalLMLoss`（transformers `loss/loss_utils.py:L49`，经 `PreTrainedModel.loss_function` 属性接入，`modeling_utils.py:L4655`；它在 labels 末尾 pad 一格 `-100` 后取 `labels[..., 1:]`，与截断 logits 语义相同）；`logits[..., :-1, :]`/`labels[..., 1:]` 的字面截断式写在 `LabelSmoother.__call__`（`trainer_pt_utils.py:L451`，label smoothing 路径，causal LM 时 `Trainer` 在 `trainer.py:L2028` 以 `shift_labels=True` 调用） |
| chat template 必须是训练串真前缀 | `data/template.py` 的模板渲染与推理 prompt 渲染共用同一函数，保证 train/test 一致 |
| causal mask + key padding mask | `transformers` 中 `LlamaModel` 的 `_prepare_4d_causal_attention_mask_for_sdpa` 类似语义 |

官方入口：

- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- PyTorch `CrossEntropyLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- transformers `ForCausalLMLoss`: https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
- transformers `Trainer` compute_loss: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

---

## 6. 费曼自检

**把 SFT 想象成“学生做填空题，老师只批改空白处的答案”**：

- 题干（prompt）印在卷子上，老师不改，学生也不需要学会“写题干”；
- 空白处（response）才是得分点，错一个字都要扣分；
- 如果老师把第一个空也当成“题干”跳过，学生永远不知道第一个空该填什么；
- 座位号（pad）既不让学生看，也不算分，它只是让不同长度的卷子能摞在一起。

**一句话结论**：SFT 不是教模型“复述整段对话”，而是教模型“在 prompt 后面续写正确 response”；
`labels` 遮罩就是划定“哪些 token 需要被批改”的红笔。

### 思考题

1. 为什么 `labels` 遮罩要在 `labels` 空间画边界，而不是在 `input_ids` 空间把 prompt token 删掉？
2. 如果把 prompt 也纳入 loss（`unmasked`），loss 会更低还是更高？模型会更会回答问题还是更会复述模板？
3. off-by-one 的模型 loss 和正确模型几乎一样低，为什么生成却完全失败？这说明“低 loss”和“做对任务”之间有什么关系？
4. 在真实 LLaMA-Factory 配置里，哪个字段（或模板）决定 prompt 和 response 的边界？如果边界写错，训练日志会出现什么现象？

---

## 7. 边界与局限

- **CPU demo**：本脚本用 CPU 在秒级跑完；真实 SFT 用 GPU + 大模型 + mixed precision，
  但 labels 遮罩、shifted CE、causal mask 的机制不变。
- **记忆而非泛化**：3 条样本、75K 参数、400 步，模型基本是在记忆；大模型 SFT 的目标
  是泛化到分布内新 prompt，但数据侧机制相同。
- **无多轮对话 / 无 system 复杂模板**：本模块聚焦单轮 SFT 数据侧最小循环；
  多轮、工具调用、角色等扩展见 L2/L3 与 `data/template.py`。
- **未使用真实 tokenizer / 未加载 LLaMA-Factory**：L1 先验证机制；L3 再对照
  框架配置体系与真实 checkpoint 加载。

---

## 8. 溯源

- 代码：`L1_minimal_sft.py`
- 运行环境：`python`，torch 2.4.1，CPU。
- 输出保真：§2 粘贴与本机实跑输出逐字节一致（连跑 3 遍 md5 相同：`44097df723130701f581753c6e0e7557`）。
- 参考：LLaMA-Factory 源码（2026-08-05 快照），路径见 §5；
  transformers 源码锚点（`loss/loss_utils.py` / `modeling_utils.py` /
  `trainer_pt_utils.py` / `trainer.py`）为 2026-08-05 main 分支现场抓取；
  L0 确定性 toy 见 `L0_sft_data_pipeline.py` / `tutorial_L0.md`。
