# nano-pretraining-loop L1 — 同一套完整状态，在真实 torch 机器上重述

> **核心问题**：L0 用纯 Python 证明了「完整状态 == 同一次训练」。换上真实的
> tokenizer / AdamW / autocast / torch.save 之后，这个命题还成立吗？哪些承诺开始失效？
>
> **先修**：[tutorial_L0](tutorial_L0.md) 全部；PyTorch 基础（Tensor / autograd / nn.Module）。
>
> **不变量**：`same full state == same training run`——机械从玩具换成真实 torch，不变量本身不变。
>
> **运行**：`python3 -B L1_real_torch_lifecycle.py`；torch CPU 版即可（`pip install torch`），
> 固定 seed + threads=1，约 2 秒，输出确定。
>
> **验收**：18/18 self-check；完整状态 resume 与连续训练参数逐位一致（diff=0.000e+00），
> 四个故障注入必须分叉；L0 输出锚在 L1 进程内复验 match=True。
>
> **边界**：toy 语料与 char vocab；CPU 单线程确定性基线。真实 GPU 训练的 bitwise exact resume
> 承诺范围见 §9，不继承到多卡。

---

## 1. 运行与输出

**可运行性契约声明**：L1 必须可跑。本节 paste 块来自真实 CPU 运行输出；2026-08-31 在
两个新建空 CWD 中以 `-B` 复验，均 EXIT=0、stderr 0 B。仅删除 elapsed 计时行后，输出锚为
`a29b3f2bae481fac33562ab4b05acb1d` / 69 行 / 3,677 B，两次逐字节一致。

```bash
python3 -B L1_real_torch_lifecycle.py
```

```text
==============================================================================
Pretraining lifecycle L1 — real torch loop: packing, AdamW, AMP, resume
==============================================================================

[0] environment & contract
    torch=2.13.0 threads=1 seed=20260818 device=cpu
    real: char tokenizer / block packing / document mask / AdamW / LambdaLR /
          bf16 autocast / torch.save serialization / RNG-state checkpoint
    toy : corpus scale, vocab, absolute positions (no per-doc position reset)

[1] document boundary: L0 pair filter == L1 mask + ignore_index
    L0 corpus replayed: 3 docs, within-doc targets=9, cross-doc leaked=2
    loss(masked policy)=18.8833  loss(naive concat)=17.7435  differ=True
    tokenizer round-trip on all 24 docs: True

[2] real training run (fp32, AdamW + LambdaLR, accum=2)
    step  train_loss   lr        val_loss
       1     17.2010  0.00200   19.4816
       5     15.4007  0.00290   14.9623
      10      9.1847  0.00191    9.8208
      15      7.7218  0.00060    7.8367
      20      6.9604  0.00000    7.4790
    val: 19.4816 -> 7.4790; best checkpoint step=20 val=7.4790
    consumed_tokens=2560  sampler=(epoch=3, cursor=70)

[3] exact resume with the real state bundle (torch.save round-trip)
    checkpoint@step8 fields: consumed_tokens,model,optimizer,py_rng,sampler,scheduler,step,torch_rng
    uninterrupted == resume: max param diff=0.000e+00  probe logits diff=0.000e+00
    failure injections (same step-8 checkpoint, one component broken):
      drop torch RNG state -> max param diff=6.419e-03
      drop scheduler state -> max param diff=1.870e-02
      drop sampler cursor  -> max param diff=1.071e-02
      weights-only resume  -> max param diff=3.569e-02

[4] AMP: bf16 autocast tracks fp32 but is not bit-identical
    fp32 final val=7.4790  bf16 final val=7.4839  |diff|=0.0049
    max param diff fp32 vs bf16 = 2.470e-02  (>0: numerics changed; val still tracks)
    fp16 range demo: 70000.0 -> inf (overflow); 1e-8 -> 0 (flushed to zero)
    subnormal band: 1e-6 -> 1.0132789611816406e-06 (still stored, but subnormal: relative precision degrading)
    loss x 2^16 first: 1e-6*65536 = 0.0655 finite in fp16 -> unscale in fp32 before step
    (CPU autocast defaults to bf16: fp32 exponent range, no GradScaler needed here)

[5] cross-level anchor: L0 invariant re-verified inside L1
    L0 stdout md5=b342b389739d1d3c04659b2349f24392  anchor match=True

[6] self-check
    PASS | tokenizer encode/decode round-trips on every doc
    PASS | mask+ignore_index reproduces L0's 9 valid / 2 leaked targets
    PASS | boundary policy changes the actual objective
    PASS | validation loss improves on this constructed corpus
    PASS | warmup raises the learning rate
    PASS | cosine decay lowers the learning rate
    PASS | best checkpoint selected by validation artifact
    PASS | consumed_tokens counter closes arithmetically
    PASS | full-state resume is bit-for-bit equal (params)
    PASS | full-state resume is bit-for-bit equal (probe logits)
    PASS | dropping RNG state (dropout masks) diverges
    PASS | dropping scheduler state diverges
    PASS | dropping data cursor diverges
    PASS | weights-only resume diverges
    PASS | bf16 autocast tracks fp32 validation loss
    PASS | bf16 is not bit-identical to fp32
    PASS | fp16 overflow / flush-to-zero / subnormal demo holds
    PASS | L0 output anchor unchanged (cross-level invariant)

SELF-CHECK: 18/18 PASS
digest: c5b1da3fad9827732fe52d02c15396b8
takeaway: the machinery changed (real tokenizer/AdamW/autocast/torch.save),
          the invariant did not: same full state == same training run.
```

---

## 2. K+1 命题：从纯 Python 状态机到真实 torch 机械

L0 用纯标准库搭了一个 bigram LM 生命周期，证明三件事：完整状态 resume == 连续跑；
丢 Adam moments ≠；丢 data cursor ≠。那个证明的可疑之处恰恰是它的干净——
玩具机械太简单，读者有理由怀疑「换成真框架还成立吗」。

L1 把同一命题搬到真实机械上重述：

| L0（纯 Python） | L1（真实 torch） |
|------|------|
| 手写 bigram 计数 | `nn.Embedding` + `nn.MultiheadAttention` 的 TinyGPT（2 层，d=32，2 头） |
| 字符串级「tokenize」 | `CharTokenizer`：真实 encode/decode round-trip（toy 词表） |
| 手写 pair 过滤 | `pack_documents` 块打包 + `document_attn_mask` + `ignore_index=-100` |
| 手写 Adam 更新 | `torch.optim.AdamW`（decoupled weight decay，arXiv:1711.05101） |
| 手写 lr 表 | `LambdaLR` warmup + cosine |
| 手写梯度累积 | `(loss / ACCUM).backward()` ×2 + `clip_grad_norm_` |
| JSON 序列化 | `torch.save` 到 BytesIO 的 8 字段 bundle |
| 无 | `torch.autocast` bf16 + torch/torch RNG 状态入 checkpoint |

**不变的是命题，变的是机械。** 确定性契约：固定 seed + `torch.set_num_threads(1)` +
无 I/O 非确定性 → 两个新建空 CWD 的输出掩码后逐字节相同。这不是玄学，是后面所有
「resume 是否 exact」判断的测量基础：输出都不确定的话，diff=0 什么也证明不了。

---

## 3. packing 边界账本：为什么 mask IS the policy

L0 §3 说过：文档不是一条无限长字符串，边界策略必须被记录。L1 把这句话变成三个互相咬合的构件。

**① 打包（`pack_documents`）**：把加权后的文档 token 拼成一条流，切成
`SEQ_LEN+1` 长、步长 `SEQ_LEN` 的重叠块（`unfold`）。重叠 1 个 token 是边界账本的关键：
块尾的 label 要预测下一个 token，必须知道「下一个 token 还是不是同一篇文档」，所以每块带一条
同形状的 doc-id 序列。

**② 标签（`boundary_labels`）**：`labels = ids[:, 1:]`，但跨文档位置置 `IGNORE=-100`，
`F.cross_entropy(..., ignore_index=-100)` 直接把这些位置从目标函数里拿掉——不贡献 loss，也不贡献梯度。

**③ 注意力（`document_attn_mask`）**：因果下三角 ∧ 同文档 才允许 attend。
**mask 就是 policy 本身**：不是训练完再加的过滤器，而是前向计算里真实改变每个 token 能看到什么的那一层。

`[1]` 节把 L0 的 3 文档语料原样重放进 L1 机械：within-doc targets=9、cross-doc leaked=2，
与 L0 的 pair 计数逐位吻合——**L0 的 pair 过滤和 L1 的 mask+ignore_index 是同一条政策的两种实现**。
而 `loss(masked policy)=18.8833 ≠ loss(naive concat)=17.7435` 证明这不是无害的记账差异：
边界策略真实改变了目标函数（未训练模型上两种口径的交叉熵不同；训练后差异会进入梯度）。

思考：如果只检查 tensor shape，9/2 这笔账永远查不出来——串的文档、错的 mask 全都 shape 合法。
**语义正确性必须用可运行的反事实证明，不能用形状断言冒充。**

---

## 4. 崩溃反例教材：一个掩码，两个契约，一次崩溃加一次静默

本节是本教程最重要的反例。这份代码的第一个版本**跑不起来**——而且修好崩溃之后才发现，
崩溃反而是运气好的那种错误。

### 4.1 崩溃现场（第一手复现，2026-08-18）

原版 `document_attn_mask` 返回 `[B, S, S]` 的 bool 掩码。第一次 forward 即死：

```text
RuntimeError: The shape of the 3D attn_mask is torch.Size([1, 11, 11]), but should be (2, 11, 11).
```

该错误来自 `multi_head_attention_forward` 的 3D 形状检查；下面的简化摘录与
[MultiheadAttention 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention)
给出的 `(N×num_heads,L,S)` 契约一致：

```python
elif attn_mask.dim() == 3:
    correct_3d_size = (bsz * num_heads, tgt_len, src_len)   # functional.py L6912
    if attn_mask.shape != correct_3d_size:
        raise RuntimeError(...)                              # functional.py L6914-6917
```

**3D 掩码契约是 `(B×num_heads, S, S)`，不是 `(B, S, S)`。** 为什么？因为 MHA 的 kernel
把 batch 和 heads **拍平成同一个并行维度**之后再施加掩码——`bsz * num_heads` 这个乘法就是
「heads 和 batch 在 kernel 视角是同一并行维度」的工程证据。掩码必须按拍平后的维度供给，
否则第 0 个 batch 的掩码会被误发给第 1 个 head。

本例 B=1、NHEAD=2：`[1, 11, 11]` 撞上 `(2, 11, 11)` 检查，fail-loud，当场死亡。

### 4.2 隐藏条件：头数

`[B, S, S]` 形状的掩码在 **NHEAD=1 时会侥幸通过**（B×1 == B）。于是这类 bug 的标准潜伏路径是：
单头原型跑通 → 换成多头模型 → 崩溃或静默错配。教训：**凡是掩码相关的单元测试，必须扫头数维度**
（NHEAD=1 和 NHEAD>1 各至少一例），就像测 batch 维一样自然。

### 4.3 第二个契约：bool 极性是反的（静默错误，比崩溃可怕）

修形状时顺手做的探针发现了第二个坑：`nn.MultiheadAttention` 的 bool 掩码里
**True = 禁止 attend**，与 `F.scaled_dot_product_attention`（True = 允许 attend）恰好相反。

探针（对角线 True 的 3×3 掩码，看返回的 attention 权重）：

```text
weights head0:
 tensor([[0.0000, 0.5595, 0.4405],
        [0.5144, 0.0000, 0.4856],
        [0.6879, 0.3121, 0.0000]])
diag-True => BLOCKED (True=BLOCK)
```

对角线为 True，权重对角线全 0——True 的位置被屏蔽了。源码链（torch 2.13.0，2026-08-18 现场核验）：

- `nn/modules/activation.py:L1293-1299` docstring 原文：「Must be of shape (L, S) or
  (N·num_heads, L, S) … For a binary mask, a ``True`` value indicates that the
  corresponding position is **not allowed to attend**.」
- `nn/modules/activation.py:L1344-1346`：forward 把 attn_mask 交给 `F._canonical_mask`；
- `nn/functional.py:L6608-6610`：`masked_fill_(mask, float("-inf"))`——**True 位置填 -inf**；
- 对照组 `nn/functional.py:L6365`（SDPA 路径）：`masked_fill_(attn_mask.logical_not(), float("-inf"))`——
  **False 位置才填 -inf**，极性相反。

如果当初只修形状、不查极性，代码会跑绿、loss 会下降、self-check 可能全过——
但模型 attend 的是「跨文档 ∧ 未来位置」，整个边界政策被静默反演。
**崩溃是契约在保护你；静默语义反转不会。写掩码前先跑一个 3 行探针确认极性，比任何文档记忆都可靠。**

修复后的构造（`L1_real_torch_lifecycle.py` `document_attn_mask`）：

```python
blocked = ~(causal.unsqueeze(0) & same)                    # 先算「允许」，再取反
return blocked.unsqueeze(1).expand(bsz, NHEAD, seq, seq).reshape(bsz * NHEAD, seq, seq)
```

### 4.4 延伸反例：fp16「下溢」不是二元开关

`[4]` 节的 fp16 演示也踩过同款「想当然」：原版用 1e-6 当「fp16 underflow」的例子，
但实测 `fp16(1e-6) = 1.0132789611816406e-06 ≠ 0`——1e-6 落在 **subnormal 区间**，
还存着，只是精度在流失。fp16 格式事实（5 位指数、10 位尾数、bias 15）：

- 最大正规数 65504（`torch.finfo(torch.float16).max` 实测），70000.0 → `inf`（overflow）；
- 最小正规数 2^-14 ≈ 6.104e-05（`finfo.tiny` 实测）；
- 其下是 subnormal 带，最小可表示值 2^-24 ≈ 5.96e-8；
- 低于 ≈2^-25 ≈ 2.98e-8（0 与最小 subnormal 的中点）才舍入为 0：`fp16(1e-8) = 0.0`。

**「下溢」是精度渐失的连续过程，不是开关**：1e-6 还能存（相对误差已到 ~1.3%），1e-8 才归零。
这正是 fp16 混合精度需要 loss scaling 的原因（arXiv:1710.03740）：小梯度落进 subnormal 带会被
精度绞杀，先把 loss 乘 2^16 推回正规区间、反向后再除回来（`[4]` 输出：1e-6×65536 = 0.0655，fp16 下有限）。
bf16 不需要 GradScaler，因为它有 8 位指数、与 fp32 同量级范围（实测 max ≈ 3.39e+38）——
代价是尾数只有 7 位。本实现显式走 CPU bfloat16 autocast，并由运行探针核验实际 dtype。

---

## 5. 全状态 bundle：L0 状态机在真实机械上的逐项对应

`[3]` 节的 checkpoint 是一个 8 字段 bundle，用真实 `torch.save` 序列化到 BytesIO：

```text
consumed_tokens, model, optimizer, py_rng, sampler, scheduler, step, torch_rng
```

与 L0 状态机逐项对应：model ↔ 权重；optimizer ↔ Adam 的 m/v/step（真实 AdamW state_dict）；
scheduler ↔ LambdaLR 的 last_epoch；sampler ↔ mixture+seed+epoch+cursor（L0 同构状态机）；
torch_rng + py_rng ↔ dropout 掩码与 shuffle 的可重放性；step + consumed_tokens ↔ 训练进度账本。

**exact resume 的判决性证据不是「loss 接得上」，而是两条逐位证据**：

```text
uninterrupted == resume: max param diff=0.000e+00  probe logits diff=0.000e+00
```

连续跑 20 步 vs「跑 8 步 → torch.save 序列化 → 反序列化 → 跑到 20 步」，参数最大差 0；
再用一个全新 sampler 取同一批数据探两个模型的前向，logits 最大差也是 0。
注意 probe 用独立的新 sampler——resume 后的模型连「没见过的输入」上的行为都逐位一致，
这比比较训练 loss 强得多。

四个故障注入共享同一个 step-8 checkpoint，每次只破坏一个成分：

```text
drop torch RNG state -> max param diff=6.419e-03
drop scheduler state -> max param diff=1.870e-02
drop sampler cursor  -> max param diff=1.071e-02
weights-only resume  -> max param diff=3.569e-02
```

读法：丢 RNG 分叉最小（只改 dropout 掩码序列，一步内影响有限）；丢 scheduler 把 lr 重置到
warmup 起点、丢 cursor 把数据重放一遍，分叉大一个量级；weights-only 把 optimizer/scheduler/
cursor/rng 全丢，分叉最大。**分叉大小排序本身就是诊断学**：线上 resume 后轨迹异常时，
按因果半径从大到小排查丢了哪一格状态。L0 的结论在真实机械上原样成立——
**checkpoint 的完整性不是配置洁癖，是「同一次训练」的定义本身。**

---

## 6. AMP 数值：bf16 追踪 fp32，但不是逐位相同

`[4]` 节用同一 seed 跑两条完整训练（fp32 vs bf16 autocast）：

```text
fp32 final val=7.4790  bf16 final val=7.4839  |diff|=0.0049
max param diff fp32 vs bf16 = 2.470e-02  (>0: numerics changed; val still tracks)
```

这是「mixed precision 改变数值但不改变结论」的机器证明：参数逐位不同（2.47e-02），
验证 loss 只差 0.0049。**bf16 训练不是 fp32 训练的 replay，而是同一条河谷里的另一条路径**——
对 AMP 的任何「应该和 fp32 一模一样」的断言都是错的，「应该在统计上追踪」才对。
self-check 用 `|diff| < 0.25` 而不是 `== 0` 编码的正是这个边界。

---

## 7. 跨级锚：L0 不变量在 L1 进程内复验

`[5]` 节在 L1 进程内用 `redirect_stdout` 重跑 L0 的 `main()`，对 stdout 取 md5：

```text
L0 stdout md5=b342b389739d1d3c04659b2349f24392  anchor match=True
```

锚值 `b342b389739d1d3c04659b2349f24392`（36 行 / 1,598 B）双通道实测：L0 独立运行的 stdout md5
== L1 进程内捕获的 md5，逐位同一。这个设计把「L1 没有破坏 L0 的命题」
变成了每次运行都执行的机器检查，而不是教程里的一句承诺——**跨级不变量要能被代码自己证明。**

---

## 8. 权威实现取舍表

L1 的权威参照就是 PyTorch 本身。源码行号容易随版本漂移，因此这里链接稳定 API 文档，
并让代码探针负责检查当前安装版本的形状、极性与 dtype 契约：

| nano L1 选择 | PyTorch / 生产实践对应 | 取舍说明 |
|------|------|------|
| `nn.MultiheadAttention` + 显式 3D 掩码 | [MultiheadAttention 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention)：3D mask 形状 `(N*num_heads,L,S)`，bool `True` 表示禁止关注 | 生产 GPT 多用 FlashAttention/SDPA 融合 kernel；这里选 MHA 是为了让 mask 契约显式可教，代价是慢 |
| `pack_documents` 重叠块 + doc-id | Megatron-LM 的 GPT dataset 同样把文档拼成长流再按块切分、记录文档边界（arXiv:1909.08053；具体 dataset 源码行锚见 [nano-megatron 溯源节](../nano-megatron/tutorial_L1.md)，此处不重复声称） | nano 版每 epoch 重打包、全量驻留内存；生产用 mmap 索引 + 预计算 shuffle 表 |
| `AdamW` + `LambdaLR` | `torch.optim.AdamW`（arXiv:1711.05101）；生产用 warmup+cosine 同款形状 | 一致，无简化 |
| `torch.save` 单文件 bundle | 生产可用 [Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)（分 rank 保存与 load-time resharding，L2 主题）或 safetensors | 单文件是为了让「完整状态」一眼可见；分片是 L2 的 K+1 |
| bf16 autocast（CPU） | [AMP 文档](https://docs.pytorch.org/docs/stable/amp.html)的 CPU 示例显式使用 `torch.autocast(..., dtype=torch.bfloat16)` | 2026-09-03 的 NVIDIA L20 参考运行显示 bf16 与 fp32 不逐位相同；硬件相关数字见 [L1_gpu_verify.py](L1_gpu_verify.py) 与 §15，不当作普遍常数 |
| weight tying（`head.weight = tok.weight`） | GPT-2 技术报告（Radford et al., 2019，OpenAI，无 arXiv 版）报告 embedding 与 softmax 层共享权重可提升性能 [TODO: verify 章节号] | 省参数、toy 词表下稳定 |

---

## 9. toy vs 生产：诚实声明

- **tokenizer**：char 级、语料内建词表。真实系统是 BPE（词表数万、merge 规则版本化），
  换 tokenizer = 换数据语义，必须进 checkpoint manifest。
- **dataloader**：这里是同步 `next_batch`，无 prefetch / 多 worker。真实系统里
  worker 数、prefetch 顺序都是数据顺序的一部分（L0 §4 的警告在真实 loader 上更尖锐）。
- **确定性**：threads=1 + 固定 seed 是**单机 CPU 基线**。真实 GPU 训练受非确定 kernel、
  collective 规约顺序、cudnn 算法选择影响，**bitwise exact resume 的承诺不继承到多卡**
  （承 L0 §6 声明）；L2 将在分布式语境下重谈「什么算 exact」。
- **规模**：24 篇合成文档、20 步训练。loss 数值（val 19.48 → 7.48）只证明「机械在学」，
  不构成任何泛化结论；best checkpoint 恰好是最后一步也只说明欠训练。
- **语料**：两个可学习的合成语法（general 短句 + DNA 样重复），为的是让 domain mixture
  有真实可学的结构，不是真实语料分布。

---

## 10. 时效性定位

pretraining lifecycle 状态机 = **A 层经典机制**：「完整状态定义同一次训练」不随算法演进过时，
无时效问题。但本教程引用的 **torch API 契约是版本敏感的**：3D 掩码形状契约与 bool 极性
锚定 torch 2.13.0（2026-08-18 核验）；未来版本若变更，以代码内探针（§4.3 的极性探针、
§7 的跨级锚）重测为准——**契约会漂移，探针永存**。

---

## 11. 费曼自检

**类比**：L0 像用积木搭了一套「游戏存档系统」并证明存档完整 == 同一局游戏；
L1 把同一套存档逻辑搬进真实游戏引擎（torch）。只存角色等级（weights）读档，技能冷却
（optimizer moments）、任务进度（data cursor）、随机事件表位置（RNG）全丢——
看起来还是同一个角色，玩下去就是另一局。完整存档（8 字段 bundle）读档后逐帧一致，
才是「同一局」。

自检问：不看上文，能不能向同事讲清——(a) 为什么 3D 掩码必须是 (B×H, S, S)？
(b) 为什么 bool 掩码极性必须用探针验、不能靠记忆？(c) 为什么 bf16 训练不承诺逐位复现
却仍然可信？三问答不出任何一问，回到 §4 / §6。

---

## 12. 思考题

1. `[B, S, S]` 掩码在 NHEAD=1 时侥幸通过、NHEAD=2 时崩溃。设计一组掩码单元测试，
   你会把「头数」放进扫描维度吗？还有哪些「=1 时侥幸」的维度（batch？seq？）值得扫？
2. MHA 与 SDPA 的 bool 掩码极性相反。如果你的团队同时用两个 API，用什么工程手段
   （探针测试 / float 掩码统一 / wrapper 类型）杜绝这类同名异义？各有什么代价？
3. fp16 的 1e-6「还在但已不准」。subnormal 带的相对误差随数值减小如何变化？
   为什么 loss scaling 乘 2^16 能救梯度、却救不了已经 underflow 的激活值？
4. 「resume 后 loss 曲线接得上」为什么不足以证明 exact resume？本脚本哪两条证据是判决性的？
   如果只比较 train loss 不比较 probe logits，会漏掉什么？
5. 把 char tokenizer 换成 HF BPE、CPU 换成 GPU 之后，本教程的哪些承诺失效？
   bundle 需要新增哪些状态（tokenizer 版本？cuda RNG？dataloader worker 状态？）？

---

## 13. 反例与边界

- **崩溃是幸运的错误**：形状契约 fail-loud，19 秒内可见；极性契约 fail-silent，
  可以跑绿一整晚。审查一段训练代码时，先问「这里哪种错误会静默」。
- **exact resume 的边界**：本教程证明的是「确定性 CPU 单线程 + 固定 seed」下的逐位相等。
  换 GPU、换 world size、换 cudnn 版本后，合理的承诺降级为「统计上不可区分」，
  判据也要从 param diff == 0 换成分布检验。
- **边界策略不是唯一正解**：本模块选「禁止跨文档 attention + mask 掉跨文档 loss」。
  允许跨文档 attention（把 EOS 当分隔符）同样是合法 policy——但必须显式选择并记录，
  不能由 packing 实现悄悄决定。
- **self-check 全绿 ≠ 机制正确**：若掩码极性反了，18 项检查照样可能全绿
  （loss 会降，只是学的是错误的目标）。绿是必要条件，探针与反事实才是充分条件。

---

## 14. 阶梯预告

L2 = distributed sampler + sharded checkpoint（README 阶梯行已声明）：
本教程的 8 字段 bundle 搬进 FSDP/TP 语境后，「完整状态」要加上分片拓扑与 rank 映射；
exact resume 的定义本身要重写。先把 [nano-fsdp L0](../nano-fsdp/tutorial_L0.md) 的
显存账本捡起来再进 L2。

---

## 15. 溯源与口径声明

- **PyTorch API**：[MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention)
  给出 3D mask 的 `(N*num_heads,L,S)` 形状与 bool `True` 禁止关注语义；
  [AMP](https://docs.pytorch.org/docs/stable/amp.html)给出 CPU bfloat16 autocast；
  [Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)给出分 rank
  保存与 load-time resharding 边界。当前安装版本的具体行为仍由 §4.3 探针判决。
- **论文**：AdamW arXiv:1711.05101；混合精度与 loss scaling arXiv:1710.03740；
  Megatron-LM arXiv:1909.08053（packing/边界背景，源码级行锚归 nano-megatron 教程）。
- **格式事实**：fp16/bf16 范围与 subnormal 边界由 `torch.finfo` 运行实测 +
  IEEE-754 half 格式推导（5 位指数 / 10 位尾数 / bias 15）；CPU autocast 默认
  CPU autocast 的 bfloat16 路径由当前运行与官方示例交叉确认。
- **运行锚**：掩码锚 `a29b3f2bae481fac33562ab4b05acb1d` / 69 行 / 3,677 B；
  digest `c5b1da3fad9827732fe52d02c15396b8`；L0 跨级锚 `b342b389739d1d3c04659b2349f24392`
  （36 行 / 1,598 B，双通道逐位同一）。
- **GPU 参考实证（硬件相关，非普遍常数）**：2026-09-03 在 NVIDIA L20 / torch 2.9.1+cu128 上对当前 [L1_gpu_verify.py](L1_gpu_verify.py) 独立运行两次：`fp32 val=7.4785`，`bf16 val=7.4682`，`|diff|=0.0103`，`max param diff=3.178e-02`；含 `cuda_rng` 的 bundle 单卡 resume 参数差 `0.000e+00`、探针 logits 差 `0.000e+00`，丢弃 `cuda_rng` 后漂移升至 `7.056e-03`；`DataLoader(num_workers=2)` 同 seed 复现批次为 True。两次均 exit 0、stderr 为空、5/5，去除 elapsed 行后输出逐字节一致，`gpu_digest=ad93f411a09c121cf3c8419f7e7d21dd`。多卡 exact resume 仍为 L2 主题；换硬件或 PyTorch 版本必须重跑，不应复用这组数值。
- **[TODO: verify 章节号]**：GPT-2 技术报告 weight tying 的具体章节。
