# minimax-h3-capstone

MiniMax H3 把前面三条机制合到一个系统合同：文本/参考媒体/视频/audio latent 被打包进单流 Transformer；视频和音频
共享一次 full self-attention 调用，但使用不同输入输出、row tag、AdaLN 与 rectified-flow scheduler。

## 立即运行

```bash
python3 -B L0_h3_system_contract.py
```

仅依赖 Python 3.10+ 标准库，CPU、离线可跑。先读 [tutorial_L0.md](tutorial_L0.md)，事实出处见
[上级研究账本](../RESEARCH.md)。

## L0–L3 阶梯

| 级别 | 项目 | 验收重点 | 状态 |
|---|---|---|---|
| L0 | H3 教学系统合同 | FL2VA/Ref2VA、surrogate IR、packed rows、双 scheduler、本地/托管边界 | 已完成 |
| L1 | 公开 metadata 账本 | 只取 config/tokenizer，复算结构、序列与显存；不下载大权重 | 规划中 |
| L2 | 开放实现源码对照 | Diffusers/SGLang 接口、packing、scheduler、offload；固定 revision | 规划中 |
| L3 | FL2VA 真机最小批 | 768p、24 FPS、4 秒、三案例；完整 manifest 与人工/代理分栏评测 | 规划中 |

## 硬边界

- `TeachingContextIR.surrogate=true`：课程自建 IR，绝不是官方私有 Context-IR schema。
- 本地只声称首版开放的 H3-Base FL2VA/Ref2VA 权重边界；Context-IR、2K Regenerate 与首版未附带的稀疏注意力
  不冒充本地模块。
- 统一称“开放权重”，不把模型、托管服务、许可证限制和未公开组件合写成“完整开源系统”。
- 不分发权重；不把 H3 输出用于训练其他模型。任何真机执行前重新核验官方许可证与地域/商业/安全要求。

## 文件

- [L0_h3_system_contract.py](L0_h3_system_contract.py)
- [tutorial_L0.md](tutorial_L0.md)
- [上级研究账本](../RESEARCH.md)
