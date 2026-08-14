#!/usr/bin/env python3
"""
nano-fsdp L0 — ZeRO 显存账本

目标：手算给定模型 / 卡数下，参数 + 梯度 + 优化器状态的显存占用，
      理解「Adam fp16 训练为什么是 16 × 参数量」。

零外部依赖，CPU 即跑。

注意：本脚本只统计模型状态（parameter / gradient / optimizer state），
      不计 activations、临时 buffer、通信 buffer、OS 开销等；
      那些是 L1/L2 用真实 FSDP 跑起来后再量的内容。
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 基本单位与换算
# ---------------------------------------------------------------------------

BYTES_PER_GB = 1_000_000_000   # 十进制 GB，方便心算；OS 工具常显示 GiB（1024^3），会略小。
BYTES_PER_MB = 1_000_000


def to_gb(bytes_val: int) -> float:
    """字节 → GB（十进制，与「7B 模型 ≈ 14 GB」这种口算一致）。"""
    return bytes_val / BYTES_PER_GB


def to_mb(bytes_val: int) -> float:
    """字节 → MB（十进制）。"""
    return bytes_val / BYTES_PER_MB


# ---------------------------------------------------------------------------
# 显存账本核心公式
# ---------------------------------------------------------------------------

def adam_training_state_bytes(params: int) -> int:
    """
    Adam + mixed-precision (fp16/bf16) 训练时，**单副本**模型状态字节数。

    构成：
      - fp16 参数:        2 bytes/param
      - fp16 梯度:        2 bytes/param
      - fp32 master 参数: 4 bytes/param
      - fp32 momentum(m): 4 bytes/param
      - fp32 variance(v): 4 bytes/param
    -------------------------------------------------
    合计:                16 bytes/param

    这是 DeepSpeed ZeRO 与 PyTorch FSDP 文献里常说的
    「Adam 训练 ≈ 16 × 参数量」的简化来源。
    """
    return params * 16


def zero_memory_per_gpu(params: int, gpus: int, stage: int) -> int:
    """
    计算 ZeRO stage 0/1/2/3 下，每张卡的**模型状态**显存（字节）。

    假设：
      - 数据并行度 = gpus
      - 均匀分片，忽略 padding / buffer / 通信开销
      - 仅 parameter + gradient + optimizer state

    ZeRO-0 就是普通 DDP：每张卡都存一份完整副本。
    ZeRO-1 只分片优化器状态（master + m + v）。
    ZeRO-2 再加分片梯度。
    ZeRO-3 把参数也分片，前向/反向时通过 all-gather 临时拼回。
    """
    if gpus <= 0:
        raise ValueError(f"gpus must be positive, got {gpus}")
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"ZeRO stage must be 0/1/2/3, got {stage}")

    param_mem = params * 2      # fp16 params
    grad_mem = params * 2       # fp16 grads
    opt_mem = params * 12       # fp32 master + m + v

    if stage == 0:
        per_gpu = param_mem + grad_mem + opt_mem
    elif stage == 1:
        per_gpu = param_mem + grad_mem + opt_mem // gpus
    elif stage == 2:
        per_gpu = param_mem + (grad_mem + opt_mem) // gpus
    else:  # stage == 3
        per_gpu = (param_mem + grad_mem + opt_mem) // gpus

    return per_gpu


# ---------------------------------------------------------------------------
# 自我校验：确保公式满足几个 obvious 的边界
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """在打印主结果前跑一遍不变量检查。"""
    # 1. 单卡时各 stage 应退化到同一个数
    total = adam_training_state_bytes(1_000_000)
    for stage in (0, 1, 2, 3):
        assert zero_memory_per_gpu(1_000_000, gpus=1, stage=stage) == total, \
            f"stage {stage} should degenerate to total memory with 1 GPU"

    # 2. ZeRO-3 每卡显存 ≈ 总显存 / GPU 数（允许整数除法 1 byte 误差）
    for gpus in (2, 4, 8):
        assert abs(zero_memory_per_gpu(1_000_000, gpus, stage=3) * gpus - total) <= gpus, \
            f"ZeRO-3 memory should scale as 1/{gpus}"

    # 3. 同卡数下 stage 越高，每卡显存越小或相等
    for gpus in (2, 4, 8):
        prev = float('inf')
        for stage in (0, 1, 2, 3):
            cur = zero_memory_per_gpu(1_000_000, gpus, stage)
            assert cur <= prev, f"stage {stage} should not use more memory than previous stage"
            prev = cur


# ---------------------------------------------------------------------------
# 打印表格用的辅助函数
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)


def format_bytes(bytes_val: int) -> str:
    """根据大小自动选 GB / MB。"""
    if abs(bytes_val) >= BYTES_PER_GB:
        return f"{to_gb(bytes_val):.2f} GB"
    return f"{to_mb(bytes_val):.2f} MB"


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> None:
    _self_check()

    # ------------------------------------------------------------------
    # 1. 为什么 Adam fp16 训练 = 16 × 参数量？
    # ------------------------------------------------------------------
    print_section("1. Adam + fp16 训练的显存账本（按每参数计）")
    print("""
component               dtype    bytes/param
--------------------    -----    ------------
parameter               fp16     2
gradient                fp16     2
master parameter        fp32     4
momentum (m)            fp32     4
variance (v)            fp32     4
----------------------------------------------
total                            16
""".strip())
    print(f"\n=> 结论：adam_training_state_bytes(P) = 16 × P bytes = {adam_training_state_bytes(1)} byte/param")

    # ------------------------------------------------------------------
    # 2. 一个玩具模型：1M 参数
    # ------------------------------------------------------------------
    print_section("2. 玩具模型：1M 参数，Adam fp16 训练")
    tiny_p = 1_000_000
    tiny_total = adam_training_state_bytes(tiny_p)
    print(f"params              = {tiny_p:,}")
    print(f"total model state   = {format_bytes(tiny_total)}")
    print(f"bytes/param         = {tiny_total / tiny_p:.0f}")

    # ------------------------------------------------------------------
    # 3. 7B 模型在不同 ZeRO stage / 卡数下的每卡显存
    # ------------------------------------------------------------------
    print_section("3. 7B 模型在不同 ZeRO stage 下的每卡显存")
    params_7b = 7_000_000_000
    gpu_counts = [1, 2, 4, 8]

    header = f"{'ZeRO stage':<12}"
    for g in gpu_counts:
        label = f"{g} GPU{'s' if g > 1 else ''}"
        header += f"{label:>12}"
    print(header)
    print("-" * len(header))

    for stage in (0, 1, 2, 3):
        row = f"ZeRO-{stage:<6}"
        for gpus in gpu_counts:
            mem = zero_memory_per_gpu(params_7b, gpus, stage)
            row += f"{to_gb(mem):>12.1f} GB"
        print(row)

    print("\n说明：")
    print("  - ZeRO-0 = 普通 DDP，每张卡都存完整副本，不随卡数减少。")
    print("  - ZeRO-3 把参数也分片，显存随卡数近似线性下降（7B/8卡 = 14 GB）。")
    print("  - OS 显存工具通常按 GiB（1024^3）显示，数值会比 GB 小约 7%；这里用 GB 方便心算。")
    print("  - 实际还要加 activations、通信 buffer、OS 开销；这里只算模型状态。")

    # ------------------------------------------------------------------
    # 4. 一个快速口算练习：80B 模型，64 卡，ZeRO-3
    # ------------------------------------------------------------------
    print_section("4. 口算练习：80B 参数，64 卡，ZeRO-3")
    params_80b = 80_000_000_000
    gpus = 64
    mem = zero_memory_per_gpu(params_80b, gpus, stage=3)
    print(f"params = {params_80b/1e9:.0f}B, gpus = {gpus}, ZeRO-3")
    print(f"model state per GPU ≈ {format_bytes(mem)}")
    print(f"=> 若 activations 再占 ~10–20 GB，单卡仍需 >30 GB（A100/H100 80G 才舒服）。")

    # ------------------------------------------------------------------
    # 5. 可视化 stage 越高、每卡越省（7B/8 卡）
    # ------------------------------------------------------------------
    print_section("5. 7B / 8 卡：ZeRO stage 与每卡显存")
    gpus = 8
    for stage in (0, 1, 2, 3):
        mem = zero_memory_per_gpu(params_7b, gpus, stage)
        bar_len = int(to_gb(mem) / 2)  # 1 '#' ≈ 2 GB
        print(f"ZeRO-{stage}: {to_gb(mem):>6.1f} GB | {'#' * bar_len}")

    # ------------------------------------------------------------------
    # 6. 给学习者的「改这里」入口
    # ------------------------------------------------------------------
    print_section("6. 自己试：改 params / gpus")
    print("""
在 main() 里改这两个变量：
    my_params = 13_000_000_000   # 例如 13B
    my_gpus   = 8
然后调用 zero_memory_per_gpu(my_params, my_gpus, stage=3)。
""".strip())


if __name__ == "__main__":
    main()
