# L0：用 oracle 先校准 rectified flow DiT

> 目标：在训练模型之前，先把 latent/token 账、条件注入、流方向、Euler 积分和 CFG 的符号全部校准。

## 0. 为什么先用 oracle

如果一个微型 DiT 生成失败，可能是数据、优化器、网络容量，也可能只是 scheduler 正负号写反。L0 直接给出正确速度，
把“训练是否学到”从“采样合同是否正确”中剥离。因而它适合单元测试，不适合展示生成能力。

## 1. latent patch 与 token 账

8×8 像素被压成 2×2 latent，随后每个 latent cell 当作一个 DiT token：64 pixel tokens 对 4 latent tokens，比例 16×。
真实 VAE 的压缩有重建损失，toy 这里只取块锚点来聚焦序列成本。

## 2. rectified-flow 目标

取噪声 $z_0$ 与数据 $z_1$ 的直线路径：

$$
z_t=(1-t)z_0+t z_1,\qquad v^*(z_t,t,c)=\frac{d z_t}{dt}=z_1-z_0.
$$

脚本的核心目标与 Euler 更新是：

```python
def flow_target(noise: list[float], data: list[float]) -> list[float]:
    """For z_t=(1-t)z_0+t z_1, the oracle target is dz_t/dt=z_1-z_0."""
    return [target - source for source, target in zip(noise, data)]
```

时间和文本条件另经最小 AdaLN 注入归一化 token。oracle velocity 仍直接可得，所以 AdaLN probe 只验证接口发生了条件调制，
不宣称它学到了速度场。

CFG 在速度空间写成：

$$
v_{cfg}=v_u+s(v_c-v_u).
$$

$s=1$ 恰好走向条件目标；$s=3$ 会越过目标。符号若改成减法，则沿正确速度的反方向走。

## 3. 运行与真实输出

```bash
python3 -B L0_rectified_flow_dit_oracle.py
```

```text
L0 rectified-flow DiT oracle
pixel_tokens=64 latent_tokens=4 ratio=16.0x
latent_patches=[0.2, 0.8, 0.2, 0.8]
AdaLN_probe=[1.347, -1.247, 0.698, -0.598]
Euler_CFG1_reconstruction_mae=0.000000000000
condition_attribute_hit_rate=1.000
wrong_sign_mae=1.200
CFG3_overshoot_mae=0.600
checks=6/6
RESULT_JSON={"checks":{"adaln_injects_condition":true,"cfg_one_reconstructs":true,"condition_attribute_matches":true,"latent_is_cheaper":true,"strong_cfg_overshoots":true,"wrong_direction_fails":true},"digest":"2c598340fd36bc51","evidence_boundary":"explicit oracle velocity; validates contracts and direction, not learned generation","metrics":{"condition_attribute_hit_rate":1.0,"latent_reconstruction_mae":0.0,"pixel_to_latent_token_ratio":16.0,"strong_cfg_mae":0.6,"wrong_sign_mae":1.2},"module":"nano-image-dit/L0","schema_version":"1.0"}
```

零 MAE 只意味着常速度直线路径被 8 个 Euler step 精确积分；它没有 FID、美学分或真实图像样本。CFG=3 的 0.6 MAE
则展示“条件更强”不是单调更好。

## 4. 迁移到可训练 DiT

L1 会把 oracle 换成 $v_\theta(z_t,t,c)$，对随机 $t$ 最小化：

$$
\mathcal L(\theta)=\mathbb E\left[\lVert v_\theta(z_t,t,c)-(z_1-z_0)\rVert_2^2\right].
$$

此时才需要分别诊断 train loss、held-out velocity error、采样误差和条件命中。合成条件能验证控制链，但不能代表真实图像质量。

## 5. 动手题与边界

1. 把 Euler steps 从 8 改为 1：常速度为何仍精确？怎样改成非恒定速度后暴露离散误差？
2. 扫描 CFG 0、0.5、1、2、3，画条件命中与重建误差的双轴曲线。
3. 把 latent patch 改成 2×2 合并，重算 token 比和信息损失。

**证据边界**：这是带答案的流场单元测试，不是训练出的 DiT；oracle 重建、合成属性命中和 token 比都不能证明照片真实感、
文字渲染、构图遵循或人类偏好。
