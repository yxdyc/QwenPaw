# L0：从 2D DiT 扩到时空 latent DiT

> 目标：把图像 token 的 $(h,w)$ 坐标扩成 $(t,h,w)$，比较逐帧独立预测与时空联合预测，并把序列成本算清。

## 0. 一段视频为什么不是一摞图片

逐帧模型可以让每帧都“像”，却让物体位置左右抖动。只检查首帧/末帧也不够：两端完全正确，中间仍可闪烁。
因此本实验同时冻结三类 estimand：端点误差、轨迹二阶差分、速度变化造成的 flicker。

## 1. 三维 patch 与位置

6 帧、每帧 2×2 latent 产生 $N=6\times2\times2=24$ 个 token。每个 token 都携带 `(time,row,col)`：

```python
def patchify_3d(video: list[list[list[float]]]) -> list[dict]:
    """Use 1x1x1 patches so every latent token carries an explicit (t,h,w) position."""
    return [
        {"value": value, "position": (time, row, col)}
        for time, frame in enumerate(video)
        for row, line in enumerate(frame)
        for col, value in enumerate(line)
    ]
```

若对全部 token 做 full attention，关系对数为 $N^2$。空间大小不变、帧数从 2 增到 8 时，token 增 4×，
attention pairs 增 16×。真实 kernel 的耗时还受 head、dtype、内存带宽与实现影响，不能直接把 $N^2$ 当实测延迟。

## 2. 两个预测器

- **joint**：首尾帧共同决定所有中间帧，toy 中是线性轨迹。
- **independent**：端点仍正确，但中间帧加入确定性交替偏移，模拟没有时序耦合的逐帧误差。

设轨迹为 $x_0,\ldots,x_{T-1}$：

$$
\text{roughness}=\frac{1}{T-2}\sum_{t=1}^{T-2}|x_{t+1}-2x_t+x_{t-1}|,
$$

$$
\text{flicker}=\frac{1}{T-1}\sum_t |(x_{t+1}-x_t)-\overline{\Delta x}|.
$$

roughness 测加速度突变，flicker 测相邻变化偏离总体运动速度；二者相关但不是同一指标。

## 3. 运行与真实输出

```bash
python3 -B L0_spatiotemporal_latent_dit.py
```

```text
L0 spatiotemporal latent DiT
shape=(t=6,h=2,w=2) tokens=24 full_attention_pairs=576
first_position=(0, 0, 0) last_position=(5, 1, 1)
joint_trajectory=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
independent_trajectory=[0.0, 0.44, 0.16, 0.84, 0.56, 1.0]
joint endpoint_mae=0.000 roughness=0.000 flicker=0.000
independent roughness=0.840 flicker=0.384
attention_scaling=[{'frames': 2, 'tokens': 8, 'full_attention_pairs': 64}, {'frames': 4, 'tokens': 16, 'full_attention_pairs': 256}, {'frames': 8, 'tokens': 32, 'full_attention_pairs': 1024}]
checks=5/5
RESULT_JSON={"checks":{"3d_positions_present":true,"attention_is_quadratic":true,"endpoints_respected":true,"joint_is_smoother":true,"joint_reduces_flicker":true},"digest":"d9489462398bdb8d","evidence_boundary":"deterministic latent trajectory simulation; not a trained video generator","metrics":{"full_attention_pairs":576,"independent_flicker":0.384,"independent_roughness":0.84,"joint_endpoint_mae":0.0,"joint_flicker":0.0,"joint_roughness":0.0,"video_tokens":24},"module":"nano-video-dit/L0","schema_version":"1.0"}
```

两条轨迹端点都正确，所以只看 endpoint 会漏掉逐帧失败。joint 的零 flicker 是线性构造的结果，不是“真实视频应当匀速”。

## 4. 迁移到 Video DiT

真实系统用 3D causal VAE 压缩时空体，再将时空 patch 输入 DiT；3D RoPE/位置编码让 attention 知道 token 在哪一帧、
哪一位置。首尾帧或参考图成为额外条件。长序列逼迫系统考虑 tiling、offload、稀疏或序列并行，但每种优化都必须保持
时间/空间位置和条件语义。

## 5. 动手题与边界

1. 把 joint 改成 ease-in/ease-out 非线性轨迹：roughness 非零是否必然意味着坏视频？
2. 让 independent 端点也偏离，比较 endpoint 与 flicker 各捕获哪类错误。
3. 把空间从 2×2 改成 4×4，分别计算 token 与 attention pairs 增长倍数。

**证据边界**：本实验没有 VAE、神经网络、真实像素、遮挡、镜头切换或音频。轨迹平滑只是一个局部代理，不能单独证明
视觉质量、物理一致性或提示遵循。
