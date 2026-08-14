"""nano-megatron · L0 玩具实现
====================================

目标：用 ~200 行纯 Python 抓住 Megatron 张量并行（TP）的核心机制——
    **「怎么切、在哪通信、为什么这样切」**。

三个实验 + 两个账本：
    1. dense MLP 参照：Y = GeLU(X @ W1) @ W2
    2. Megatron 式 TP：W1 按列切（column parallel）→ GeLU 逐元素、无需通信
       → W2 按行切（row parallel）→ 前向只需 **1 次 all-reduce**，与 dense 严格一致。
    3. 反例：W1 按行切时 GeLU 非线性破坏可加性，GeLU 前不多插一次 all-reduce
       结果就是错的 ⇒「W1 列切 → W2 行切」是前向只需一次 all-reduce 的切法
       （attention 同理：QKV 按 head 列切、输出投影行切）。
    4. 通信账本：all-reduce 量与卡数无关、加卡不省流量 → TP 只在机器内用。
    5. 显存账本：每卡 params = P/N；结合 nano-fsdp L0 → 每卡训练状态 16P/N。

这是 L0（玩具级）：单文件、纯标准库、CPU 即跑，自检全部带 assert。
流水线并行 PP（L2）、序列并行 SP（L3）见 README。
参考：Megatron-LM 论文 arXiv:1909.08053（Shoeybi et al., 2019），
    列/行切分与「每层 fwd 2 + bwd 2 次 all-reduce」的核算出自该论文；
    源码级对照留到 L3（标 [TODO: verify source]）。

运行：
    python L0_tp_mlp.py
"""

import math
import random
from typing import List, Tuple

Matrix = List[List[float]]


# ===== 最小矩阵工具（纯标准库，无 numpy）=====

def matmul(A: Matrix, B: Matrix) -> Matrix:
    """A[m,k] @ B[k,n] -> [m,n]"""
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def mat_sum(mats: List[Matrix]) -> Matrix:
    """同形矩阵逐元素求和——这就是 all-reduce(sum) 的语义。"""
    out = [row[:] for row in mats[0]]
    for M in mats[1:]:
        for i, row in enumerate(M):
            for j, v in enumerate(row):
                out[i][j] += v
    return out


def gelu(x: float) -> float:
    """GeLU 精确形式：0.5x(1 + erf(x / sqrt(2)))。"""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_mat(A: Matrix) -> Matrix:
    return [[gelu(x) for x in row] for row in A]


def max_abs_diff(A: Matrix, B: Matrix) -> float:
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def rand_matrix(rng: random.Random, m: int, n: int) -> Matrix:
    return [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(m)]


# ===== 实验 1/2/3：dense 参照 + Megatron 式 TP + 行切反例 =====

def mlp_dense(X: Matrix, W1: Matrix, W2: Matrix) -> Matrix:
    """参照实现：Y = GeLU(X @ W1) @ W2。"""
    return matmul(gelu_mat(matmul(X, W1)), W2)


def mlp_tp(X: Matrix, W1: Matrix, W2: Matrix, n_ranks: int) -> Matrix:
    """Megatron 式张量并行：W1 列并行 + W2 行并行，前向只有 1 次 all-reduce。"""
    f = len(W1[0])
    assert f % n_ranks == 0
    shard = f // n_ranks
    partials = []
    for r in range(n_ranks):
        # 列并行：rank r 持有 W1 的第 r 列块 [h, f/N]
        W1_r = [row[r * shard:(r + 1) * shard] for row in W1]
        # GeLU 逐元素：半份 pre-activation 可独立过非线性，无需通信
        H_r = gelu_mat(matmul(X, W1_r))
        # 行并行：rank r 持有 W2 的第 r 行块 [f/N, h]（与自己的 H_r 列数对齐）
        W2_r = W2[r * shard:(r + 1) * shard]
        partials.append(matmul(H_r, W2_r))   # 部分和 [t, h]
    return mat_sum(partials)                 # ← 前向唯一一次 all-reduce


def _row_first_partials(X: Matrix, W1: Matrix, n_ranks: int) -> List[Matrix]:
    """反例的前半段：W1 按行切（切输入维），每个 rank 得到部分 pre-activation。"""
    h = len(W1)
    assert h % n_ranks == 0
    shard = h // n_ranks
    return [matmul([row[r * shard:(r + 1) * shard] for row in X],
                   W1[r * shard:(r + 1) * shard]) for r in range(n_ranks)]


def mlp_row_first_wrong(X: Matrix, W1: Matrix, W2: Matrix, n_ranks: int) -> Matrix:
    """反例：直接对「半生」的部分 pre-activation 做 GeLU——非线性破坏可加性，结果错。"""
    H = mat_sum([gelu_mat(P) for P in _row_first_partials(X, W1, n_ranks)])
    return matmul(H, W2)


def mlp_row_first_fixed(X: Matrix, W1: Matrix, W2: Matrix, n_ranks: int) -> Matrix:
    """同样切法，但 GeLU 前先 all-reduce 拼回完整 pre-activation：结果对了，多一次通信。"""
    P_full = mat_sum(_row_first_partials(X, W1, n_ranks))   # 多出来的 all-reduce
    return matmul(gelu_mat(P_full), W2)


# ===== 两个账本：通信与显存（数字全部现场算出，可复算）=====

def comm_ledger(h: int, seq: int, batch: int, layers: int,
                dtype_bytes: int, pp_stages: int) -> Tuple[int, int, int, int]:
    """返回 (一次 all-reduce 量, TP 每层, TP 每 microbatch, PP 每 microbatch)，bytes。

    TP：每层 2 个「列并行+行并行」块（attention 输出投影 + MLP 第二线性），
    每块前向 1 次 all-reduce，bwd 对称 2 次 → 每层 4 次；PP：只在 stage 边界
    点对点传激活，fwd+bwd 共 2*(p-1) 次。
    """
    t = batch * seq
    act = t * h * dtype_bytes
    tp_per_layer = 4 * act
    tp_per_micro = tp_per_layer * layers
    pp_per_micro = 2 * (pp_stages - 1) * act
    return act, tp_per_layer, tp_per_micro, pp_per_micro


def ring_bytes_per_gpu(act: int, n_ranks: int) -> float:
    """ring all-reduce 中每张卡实际收发量 ≈ 2(N-1)/N × 消息量（经典结论）。"""
    return 2.0 * (n_ranks - 1) / n_ranks * act


def main() -> None:
    print("=" * 64)
    print("nano-megatron L0 — tensor parallel MLP: 怎么切，在哪通信")
    print("=" * 64)

    rng = random.Random(42)
    t, h, f = 3, 4, 8            # toy 维度（真实模型 ffn 常取 4h）
    X, W1, W2 = rand_matrix(rng, t, h), rand_matrix(rng, h, f), rand_matrix(rng, f, h)
    print(f"\ntoy shape: X[{t}x{h}]  W1[{h}x{f}]  W2[{f}x{h}]  (GeLU MLP)")

    Y_ref = mlp_dense(X, W1, W2)
    print(f"\n[1] dense 参照: Y_ref[0] = [{', '.join(f'{v:.6f}' for v in Y_ref[0])}]")

    Y_tp = mlp_tp(X, W1, W2, n_ranks=2)
    err_tp = max_abs_diff(Y_ref, Y_tp)
    print(f"\n[2] Megatron 式 TP (2 ranks): W1 列切 + GeLU + W2 行切 + 1 次 all-reduce")
    print(f"    max|Y_tp - Y_ref| = {err_tp:.3e}   {'✅ 数值严格一致' if err_tp < 1e-9 else '✗'}")
    assert err_tp < 1e-9, "TP 结果应与 dense 一致"

    Y_wrong = mlp_row_first_wrong(X, W1, W2, n_ranks=2)
    Y_fixed = mlp_row_first_fixed(X, W1, W2, n_ranks=2)
    err_wrong = max_abs_diff(Y_ref, Y_wrong)
    err_fixed = max_abs_diff(Y_ref, Y_fixed)
    print(f"\n[3] 反例：W1 按行切（切输入维）")
    print(f"    naive（对部分和直接 GeLU）: max|err| = {err_wrong:.6f}  ✗ 错得离谱")
    print(f"    fixed（GeLU 前先 all-reduce）: max|err| = {err_fixed:.3e}  ✅ 又对了，但多一次通信")
    assert err_wrong > 1e-3, "反例应显著出错"
    assert err_fixed < 1e-9, "fix 后应恢复一致"
    print("    => 「W1 列切 → W2 行切」是前向只需 1 次 all-reduce 的切法")

    H, SEQ, B, L, DTYPE, P = 4096, 2048, 1, 32, 2, 8   # 6.4B 级模型，fp16
    act, tp_layer, tp_micro, pp_micro = comm_ledger(H, SEQ, B, L, DTYPE, P)
    MiB = 1024 ** 2
    print(f"\n[4] 通信账本（h={H}, seq={SEQ}, batch={B}, layers={L}, fp16）")
    print(f"    一次 all-reduce 量 [b*s, h]          = {act / MiB:8.1f} MiB")
    print(f"    TP 每层 fwd+bwd（4 次 all-reduce）   = {tp_layer / MiB:8.1f} MiB")
    print(f"    TP 每 microbatch（{L} 层）           = {tp_micro / MiB:8.1f} MiB ≈ {tp_micro / 1024**3:.1f} GiB")
    print(f"    对照 PP（{P} stages）：边界点对点     = {pp_micro / MiB:8.1f} MiB")
    print(f"    ring all-reduce 每卡收发量：")
    for n in (2, 4, 8):
        print(f"      TP={n:>2}: {ring_bytes_per_gpu(act, n) / MiB:6.1f} MiB "
              f"(= {2 * (n - 1) / n:.2f} × 消息量)")
    print("    => 加卡不省流量（趋近 2× 消息量）→ TP 靠机器内 NVLink；PP 流量小 → 跨机器")

    h0, l0 = 4096, 32
    total_params = 12 * h0 * h0 * l0            # attention ~4h^2 + MLP ~8h^2，每层
    total_bytes = total_params * 2              # fp16
    print(f"\n[5] 显存账本（h={h0}, layers={l0} → 总参数 ≈ {total_params:.2e}，约 6.4B）")
    for n in (1, 2, 4, 8):
        per_rank = total_params // n
        assert per_rank * n == total_params, "参数应能被 N 整除切分"
        print(f"    TP={n}: 每卡 params = {per_rank:.3e} "
              f"→ fp16 权重 {total_bytes / n / 1024**3:5.2f} GiB")
    print("    结合 nano-fsdp L0：Adam+fp16 训练状态 16 bytes/param → TP 下每卡 16P/N")

    print("\n" + "=" * 64)
    print("✅ self-check passed: TP 数值等价 / 反例显著出错且可修复 / 显存线性切分")
    print("=" * 64)
    print("\ntakeaway: TP 把「计算」本身切进各卡，前向每块只需 1 次 all-reduce；")
    print("          切法（W1 列 → W2 行）由 GeLU 的逐元素性唯一决定；通信量不随卡数下降，所以 TP 只在机器内用，跨机器交给 PP（L2）。")


if __name__ == "__main__":
    main()
