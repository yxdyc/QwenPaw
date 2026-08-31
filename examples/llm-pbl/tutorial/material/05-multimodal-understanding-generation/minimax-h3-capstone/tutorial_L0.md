# L0：MiniMax H3 的系统合同，而不是“大模型假跑”

> 目标：用一个可执行 surrogate 串起 `Context → TeachingContextIR → packed rows → joint video/audio flow → decode/regenerate`，
> 并让错误 row、tag、scheduler 与部署边界确定性失败。

## 0. 三层事实先分开

| 层 | 本教程怎样用 | 不能推出什么 |
|---|---|---|
| 官方公开事实 | 输入形态、33B 单流、Qwen3-VL-32B encoder、VAE/audio latent、开放权重边界 | 完整私有 Context-IR schema |
| Diffusers 实现事实 | packed full self-attention、modality-specific AdaLN/head、video shift 12、audio shift 3、CFG distilled | 托管 2K 已能本地运行 |
| L0 surrogate | row/table、token 公式、错误注入 | H3 权重已加载或输出质量已复现 |

官方/源码链接集中在 [RESEARCH.md](../RESEARCH.md)，不要从 toy 数字反推模型卡未声明的实现。

## 1. Context 与 TeachingContextIR

FL2VA checkpoint 家族允许 0/1/2 张首尾图：0 张是 T2VA，1 张是首帧或尾帧控制，2 张是首尾帧控制；
Ref2VA 则要求 reference media。课程把 prompt 结构化成：

```python
@dataclass(frozen=True)
class TeachingContextIR:
    surrogate: bool
    task: str
    shot: str
    motion: str
    sound_event: str
```

`surrogate` 固定为 `true`。这个名字刻意带 `Teaching`：它只帮助学习者看见“自然语言如何先被规划再打包”，不猜测、
不逆向冒充 MiniMax 未公开的 Context-IR schema。

## 2. 两本 token 账

视觉教学账按公开描述的 f16t4d24 VAE，再接 Transformer `1×2×2` patch。这里 `f16` 是空间压缩 16×、`t4` 是
时间压缩 4×、`d24` 是 latent channel 数，不能把 24 当空间压缩因子：

$$
N_v=\left\lceil\frac{T}{4}\right\rceil
\left\lceil\frac{\lceil H/16\rceil}{2}\right\rceil
\left\lceil\frac{\lceil W/16\rceil}{2}\right\rceil.
$$

对 4 秒、24 FPS、768×1344，脚本得到 $24\times24\times42=24{,}192$ 个教学视觉 token。音频按每声道 40 Hz、
4 秒、双声道得到 $4\times40\times2=320$ 个教学 audio latent token。这里按 channel 展开是课程账本口径；真实 tensor
布局必须在 L2 固定 revision 后对 config/source shape 再核，不能把本式当完整显存预测。

## 3. packed full self-attention 与双 scheduler

三行分别带连续 `row_index`、`tag`、token count 和 3D MM-RoPE 坐标。一次 packed Transformer full self-attention
同时看见文本、视频和音频，不是分别做两次 cross-attention。模态差异在输入/输出投影、row tag、AdaLN 与 head。

同一次 Transformer 调用后，视频与音频各自走 rectified-flow scheduler：video `shift=12`，audio `shift=3`。
首版 checkpoint 是 CFG-distilled，所以合同记录单次 forward；L0 不再伪造一次 unconditional pass。

## 4. 四类错误必须失败

1. row index 不连续：packing 的位置/行身份不再可靠；
2. modality tag 缺失：无法选择模态分支；
3. video/audio scheduler shift 对调：同流不等于同 scheduler；
4. 把 hosted 2K Regenerate 报成本地 decode：部署声明越过开放边界。

## 5. 运行与真实输出

```bash
python3 -B L0_h3_system_contract.py
```

```text
L0 MiniMax H3 system contract
pipeline=Context -> TeachingContextIR -> packed rows -> joint flow -> decode/regenerate
TeachingContextIR={"motion":"left-to-right","shot":"wide-to-close","sound_event":"impact-at-2s","surrogate":true,"task":"FL2VA"}
contract=FL2VA rows=[(0, 'text', 9), (1, 'video', 24192), (2, 'audio', 320)]
3D_MM_RoPE=[(0, 0, 0), (24, 24, 42), (160, 0, 1)]
attention=full_self_attention cross_attention=false
schedulers=video:rectified_flow/shift12 audio:rectified_flow/shift3
CFG_distilled_forward_calls=1
boundaries=768p:local_decode 2K:hosted_regenerate
contract_and_failure_checks=8/8
RESULT_JSON={"checks":{"FL2VA_contract":true,"Ref2VA_contract":true,"T2VA_via_FL2VA_contract":true,"hosted_as_local_rejected":true,"missing_tag_rejected":true,"row_index_error_rejected":true,"scheduler_mix_rejected":true,"surrogate_marked":true},"digest":"39d8de20dc55e829","evidence_boundary":"surrogate contract only; not official Context-IR, weights, hosted 2K, or production proof","metrics":{"audio_latent_tokens":320,"cfg_distilled_forward_calls":1,"packed_tokens":24521,"video_latent_tokens":24192},"module":"minimax-h3-capstone/L0","schema_version":"1.0"}
```

## 6. L3 真机预注册

真机前重新只读确认 GPU、磁盘、依赖和许可证，并固定 H3、Diffusers/SGLang 与模型 revision。首批只下载 FL2VA，跑
BF16、短边 768p、24 FPS、4 秒：直接 T2VA、同 seed 的本地结构化 prompt T2VA、首尾帧 FL2VA。记录 prompt、seed、
revision、GPU、耗时、峰值显存、SHA256、时长/FPS 与 32 kHz stereo 契约；媒体存仓库外。

不调用付费 Context-IR/2K API，不声称本地复现 2K 完整系统；Ref2VA 大权重另批批准。评测把镜头/指令遵循、端点约束、
时序稳定、视听事件对齐的代理指标与人工判断分栏。

## 7. 动手题与边界

1. 把 FL2VA 的首尾帧都关掉，确认它仍是合法 T2VA；再把任务改成 Ref2VA 且不提供 reference，确认先于 tokenization 失败。
2. 为 Ref2VA 增加 reference row，解释 row index 为什么要重新连续编号。
3. 将时长加倍，分别计算 visual/audio token 与 full-attention pairs 的增长。

**证据边界**：本脚本没有下载权重、VAE、tokenizer、Transformer 或 decoder；row/rope 数值是教学载荷。它证明课程合同能
拒绝四类错误，不证明 H3 生成质量、速度、显存、2K 服务或完整系统开放性。
