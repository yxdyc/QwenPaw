#!/usr/bin/env python3
"""
deepseek_v3_mechanisms_sim.py — DeepSeek-V3 四个机制面的可运行本质模拟

本文件是 02 轨 sota-deepdive（deepseek-moe-mla-stability.md）的可运行锚点。
显式声明（可运行性契约，课程可运行性契约）：这是**本质模拟**——nano-megatron 阶梯
覆盖了并行/MFU 侧的实测锚点，但 MoE 路由、MLA 压缩、FP8 量化、梯度裁剪四个
机制面在 nano 侧没有现成实测锚，本文件用 toy 尺度 + 真实格式语义演示其机制。
真实系统行为见 DeepSeek-V3 官方仓库 inference/model.py（github.com/deepseek-ai/DeepSeek-V3）。

四个机制面：
  [A] MoE：sigmoid 路由 + 组限制 + aux-loss-free 偏置负载均衡（V3 §2.1.2，Eq.16）
  [B] MLA：低秩 KV 联合压缩 + absorbed 推理路径 + 解耦 RoPE 的必要性（V2 §2.1）
  [C] FP8：细粒度量化（tile/block-wise）与高精度累加（V3 §3.3）
  [D] 稳定性：梯度范数裁剪——V3 唯一披露的稳定性旋钮（V3 §4.2，clip norm = 1.0）

依赖：仅 torch（CPU 即跑，torch 2.13.0 实测；fp8 用真实 float8_e4m3fn 格式做
量化-反量化，矩阵计算仍在 fp32/fp64——模拟量化误差，不模拟硬件 kernel）。
确定性：seed=3，无计时行，跨运行逐字节一致（digest 见输出末行）。

运行：python3 deepseek_v3_mechanisms_sim.py
"""
import sys
import hashlib

import torch

sys.dont_write_bytecode = True

torch.manual_seed(3)
DEV = torch.device("cpu")

# ----------------------------------------------------------------------
# 现场核实值（github.com/deepseek-ai/DeepSeek-V3 inference/configs/config_671B.json，
# main 分支 2026-08-11 抓取；逐项见 deepseek-moe-mla-stability.md §溯源）
# ----------------------------------------------------------------------
V3 = dict(
    dim=7168, inter_dim=18432, moe_inter_dim=2048, n_layers=61, n_dense_layers=3,
    n_heads=128, n_routed_experts=256, n_shared_experts=1, n_activated_experts=8,
    n_expert_groups=8, n_limited_groups=4, route_scale=2.5,
    q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
    v_head_dim=128, vocab_size=129280,
)

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"    {'PASS' if cond else 'FAIL'}  {name}" + (f" ({detail})" if detail else ""))


def rope_pairs(x, theta=10000.0):
    """标准 RoPE：相邻两维一对做二维旋转，位置 t 的转角 t·θ^{-2i/d}。"""
    t = torch.arange(x.size(0), dtype=torch.float64).unsqueeze(1)  # [T, 1]
    d = x.size(-1)
    i = torch.arange(0, d, 2, dtype=torch.float64)
    freq = 1.0 / (theta ** (i / d))                                # [d/2]
    ang = t * freq                                                 # [T, d/2]
    c, s = torch.cos(ang), torch.sin(ang)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


# ======================================================================
print("=" * 72)
print("DeepSeek-V3 mechanisms sim — MoE routing / MLA compression / FP8")
print("=" * 72)
print(f"toy scale, seed=3, fp32 (fp8 = real E4M3 quantize-dequantize) | V3 config 现场值内嵌")

# ======================================================================
# [A] MoE：sigmoid 路由 + 组限制 + aux-loss-free 偏置负载均衡
# ======================================================================
print()
print("[A] MoE routing: sigmoid + group-limited top-K + aux-loss-free bias")

# ---- [A0] V3 尺度参数/激活账本（从 config 现场值重算，对账官方 671B/37B）----
expert_p = 3 * V3["dim"] * V3["moe_inter_dim"]                 # SwiGLU: w1,w2,w3
n_moe_layers = V3["n_layers"] - V3["n_dense_layers"]           # 61-3 = 58
routed_total = V3["n_routed_experts"] * expert_p * n_moe_layers
attn_per_layer = (V3["dim"] * V3["q_lora_rank"]                            # wq_a
                  + V3["q_lora_rank"] * V3["n_heads"] * (V3["qk_nope_head_dim"] + V3["qk_rope_head_dim"])  # wq_b
                  + V3["dim"] * (V3["kv_lora_rank"] + V3["qk_rope_head_dim"])  # wkv_a
                  + V3["kv_lora_rank"] * V3["n_heads"] * (V3["qk_nope_head_dim"] + V3["v_head_dim"])  # wkv_b
                  + V3["n_heads"] * V3["v_head_dim"] * V3["dim"])          # wo
attn_total = attn_per_layer * V3["n_layers"]
dense_mlp_total = V3["n_dense_layers"] * 3 * V3["dim"] * V3["inter_dim"]
emb_total = V3["vocab_size"] * V3["dim"]                       # 与输出头物理共享（MTP 节 §2.2）
total_recount = routed_total + attn_total + dense_mlp_total + emb_total
act_experts = (V3["n_activated_experts"] + V3["n_shared_experts"]) * expert_p * n_moe_layers
act_recount = act_experts + attn_total + dense_mlp_total + emb_total
print(f"  [A0] V3 尺度账本（config_671B 现场值重算）")
print(f"      单 expert = 3·dim·moe_inter = {expert_p:,} 参数; MoE 层数 = {n_moe_layers}")
print(f"      routed 专家总参数 = {routed_total/1e9:.1f}B | 重算总参 = {total_recount/1e9:.1f}B (官方 671B)")
print(f"      每 token 激活 = top-{V3['n_activated_experts']}+shared-{V3['n_shared_experts']} 专家 {act_experts/1e9:.1f}B + 全部 attn/dense/emb")
print(f"      重算激活 = {act_recount/1e9:.1f}B (官方 37B) | 激活/总 = {act_recount/total_recount*100:.1f}%")
check("A0 总参重算 vs 官方 671B 偏差 <1%", abs(total_recount / 671e9 - 1) < 0.01,
      f"{total_recount/1e9:.1f}B vs 671B")
check("A0 激活重算 vs 官方 37B 偏差 <3%", abs(act_recount / 37e9 - 1) < 0.03,
      f"{act_recount/1e9:.1f}B vs 37B")

# ---- toy MoE：复刻 V3 Gate.forward 的路由逻辑（model.py:L566-598）----
N_R, TOPK, D = 16, 4, 8          # 16 专家 / top-4 / token 维度 8（V3 为 256/8/7168）
N_GROUPS, TOPK_GROUPS = 4, 3     # V3: 8 组选 4 组（组内 32 专家）；toy: 4 组选 3 组（组内 4）。
                                 # 50% 组排除在 toy 尺度造成结构性半饥饿（整组专家每步零
                                 # token），控制器无法在单批上收敛，故放宽比例、机制同构。
ROUTE_SCALE = 2.5                # config: route_scale 2.5
N_CLUSTERS, BATCH = 4, 512

centroids = torch.randn(N_CLUSTERS, D)
W_gate = torch.randn(N_R, D) * 0.5


def make_batch(gen, n):
    lab = torch.randint(0, N_CLUSTERS, (n,), generator=gen)
    return centroids[lab] + 0.3 * torch.randn(n, D, generator=gen), lab


def v3_gate(x, bias, need_weights=True):
    """与 inference/model.py Gate.forward 同构：sigmoid → (+bias 只用于选路) →
    组限制（组分=组内 top-2 和，选 top-k 组，余组 mask -inf）→ top-K →
    权重取**原始分数** → 归一 → ×route_scale。"""
    scores = torch.sigmoid(x @ W_gate.T)               # [B, N_R]
    original = scores
    s = scores + bias if bias is not None else scores
    if N_GROUPS > 1:
        s = s.view(-1, N_GROUPS, N_R // N_GROUPS)    # 与 model.py:L585 同：view 回写
        group_scores = s.topk(2, dim=-1)[0].sum(dim=-1) if bias is not None else s.amax(dim=-1)
        keep = group_scores.topk(TOPK_GROUPS, dim=-1)[1]
        mask = torch.ones(s.size(0), N_GROUPS, dtype=torch.bool).scatter_(1, keep, False)
        s = s.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
    idx = torch.topk(s, TOPK, dim=-1)[1]
    if not need_weights:
        return idx, original
    w = original.gather(1, idx)
    w = w / w.sum(dim=-1, keepdim=True) * ROUTE_SCALE  # sigmoid 分数归一后 ×2.5
    return idx, w, original


gen0 = torch.Generator().manual_seed(11)
x0, _ = make_batch(gen0, BATCH)
target_load = TOPK * BATCH / N_R   # 均匀时每个专家的期望负载

# ---- [A1] 自然路由：无偏置时的负载失衡 ----
bias = torch.zeros(N_R)
idx0, w0, orig0 = v3_gate(x0, bias)
counts0 = torch.bincount(idx0.flatten(), minlength=N_R)
cv0 = (counts0.float().std() / counts0.float().mean()).item()
dead0 = int((counts0 == 0).sum())
print(f"  [A1] 自然路由（bias=0）: 每 expert 负载 min/median/max = "
      f"{int(counts0.min())}/{int(counts0.median())}/{int(counts0.max())} (均匀期望 {target_load:.0f})")
print(f"      变异系数 CV = {cv0:.3f}, 死专家（0 token）= {dead0}")
check("A1 自然路由显著失衡 (CV>0.35 或有死专家)", cv0 > 0.35 or dead0 > 0,
      f"CV={cv0:.3f}, dead={dead0}")

# ---- [A2] aux-loss-free 偏置控制器（V3 §2.1.2：overloaded 减 γ、underloaded 加 γ）----
# V3 监控「每步整个 batch 的专家负载」，生产批极大、测量低噪；toy 的 512-token
# 批测量噪声过大（±γ 规则符号每步翻转），控制器阶段提批至 2048（机制同构）。
# γ = 「bias update speed」（V3 §2.1.2 原文术语）；生产值披露于 §4.2：前 14.3T
# tokens γ=0.001，末 500B γ=0.0（现场核验）。toy 批小、分数间隙大，校准取
# γ=0.01：γ 是 bang-bang 控制器唯一旋钮，过大则相对分数间隙翻转过猛 → 极限环
# （见 [A2b] 对照：γ=0.10 终态 max/期望钉死在 ~1.8 不收敛）。
GAMMA, STEPS, BATCH2 = 0.01, 200, 2048
target_load2 = TOPK * BATCH2 / N_R
bias = torch.zeros(N_R)
gen = torch.Generator().manual_seed(12)
hist = []
for step in range(STEPS):
    xb, _ = make_batch(gen, BATCH2)
    idx, _ = v3_gate(xb, bias, need_weights=False)
    counts = torch.bincount(idx.flatten(), minlength=N_R).float()
    hist.append((counts.max() / target_load2).item())
    bias = torch.where(counts > target_load2, bias - GAMMA, bias + GAMMA)  # V3 规则
xb_last, _ = make_batch(gen, BATCH2)
idx_f, w_f, orig_f = v3_gate(xb_last, bias)
counts_f = torch.bincount(idx_f.flatten(), minlength=N_R)
ratio_max = (counts_f.max() / target_load2).item()
ratio_min = (counts_f.min() / target_load2).item()
print(f"  [A2] bias 控制器 {STEPS} 步（γ={GAMMA}, 批={BATCH2}）: max/期望负载 {hist[0]:.2f} → {ratio_max:.2f}, "
      f"min/期望负载 → {ratio_min:.2f}")
print(f"      终态负载 min/median/max = {int(counts_f.min())}/{int(counts_f.median())}/{int(counts_f.max())} "
      f"(期望 {target_load2:.0f}), bias 范围 [{bias.min():.2f}, {bias.max():.2f}]")
check("A2 控制器收敛 (终态 max/期望 <1.25 且无死专家)",
      ratio_max < 1.25 and int(counts_f.min()) > 0, f"max={ratio_max:.2f}, min={int(counts_f.min())}")

# ---- [A2b] 对照：γ 过大 → bang-bang 极限环（同一控制器、同批、只换 γ）----
GAMMA_BIG = 0.10
bias_big = torch.zeros(N_R)
gen_big = torch.Generator().manual_seed(13)
for step in range(STEPS):
    xb, _ = make_batch(gen_big, BATCH2)
    idx, _ = v3_gate(xb, bias_big, need_weights=False)
    counts = torch.bincount(idx.flatten(), minlength=N_R).float()
    bias_big = torch.where(counts > target_load2, bias_big - GAMMA_BIG, bias_big + GAMMA_BIG)
xb_big, _ = make_batch(gen_big, BATCH2)
idx_big, _ = v3_gate(xb_big, bias_big, need_weights=False)
counts_big = torch.bincount(idx_big.flatten(), minlength=N_R)
ratio_max_big = (counts_big.max() / target_load2).item()
print(f"  [A2b] 对照 γ={GAMMA_BIG}（过大）: 终态 max/期望 = {ratio_max_big:.2f} ← 极限环，不收敛"
      f"（负载 min/max = {int(counts_big.min())}/{int(counts_big.max())}）")
check("A2b γ 过大不收敛 (max/期望 >1.5，与 A2 形成对照)", ratio_max_big > 1.5,
      f"max={ratio_max_big:.2f}")

# ---- [A3] Eq.16 机制验证：bias 只改「选谁」，不改「选中者的权重」----
idx_nb, w_nb, orig_nb = v3_gate(x0, None)              # 无 bias 路由
idx_b, w_b, orig_b = v3_gate(x0, bias)                 # 有 bias 路由
s_b = torch.sigmoid(x0 @ W_gate.T) + bias              # 含 bias 的分数（若泄漏进权重应是它）
leak_w = s_b.gather(1, idx_b)
leak_w = leak_w / leak_w.sum(dim=-1, keepdim=True) * ROUTE_SCALE
proper_w = orig_b.gather(1, idx_b)
proper_w = proper_w / proper_w.sum(dim=-1, keepdim=True) * ROUTE_SCALE
same_weights = torch.allclose(w_b, proper_w, atol=0)          # 恰等于「原始分数归一」
bias_not_leaked = not torch.allclose(w_b, leak_w, atol=1e-9)  # 且不同于「含 bias 分数归一」
selection_changed = not torch.equal(torch.sort(idx_b, dim=-1)[0], torch.sort(idx_nb, dim=-1)[0])
print(f"  [A3] Eq.16 验证: 权重 == 原始 sigmoid 分数归一（不含 bias）= {bool(same_weights)}; "
      f"权重 != 含 bias 分数归一（未泄漏）= {bool(bias_not_leaked)}; "
      f"bias 改变被选集合 = {bool(selection_changed)}")
check("A3 gating 值不含 bias（权重==原始分数归一）", same_weights)
check("A3 bias 未泄漏进权重（与泄漏版不同）", bias_not_leaked)
check("A3 bias 确实改变选路（集合不同）", selection_changed)

# ======================================================================
# [B] MLA：低秩 KV 联合压缩 + absorbed 路径 + 解耦 RoPE
# ======================================================================
print()
print("[B] MLA: low-rank joint KV compression + absorbed inference + decoupled RoPE")

# ---- [B0] V3 尺度 KV cache 账本（config 现场值）----
mha_kv = V3["n_heads"] * (V3["qk_nope_head_dim"] + V3["qk_rope_head_dim"] + V3["v_head_dim"])
mla_kv = V3["kv_lora_rank"] + V3["qk_rope_head_dim"]
print(f"  [B0] V3 尺度 KV cache /token/layer: MHA 等价 = 128×(128+64+128) = {mha_kv:,} 值; "
      f"MLA = kv_lora 512 + rope 64 = {mla_kv}")
print(f"      压缩比 = {mha_kv/mla_kv:.1f}× (压缩 {(1-mla_kv/mha_kv)*100:.1f}%); "
      f"V2 摘要的 93.3% 是 vs DeepSeek 67B 的 MHA，口径不同不混用")
check("B0 压缩比 = 40960/576 = 71.1×", abs(mha_kv / mla_kv - 71.111) < 0.01, f"{mha_kv/mla_kv:.2f}x")

# ---- toy 维度（与 V3 同构缩小）----
d, n_h, d_nope, d_rope, d_c, d_v = 64, 8, 8, 4, 16, 8
T, S = 6, 5                       # query 长度 / key 长度
h = torch.randn(S, d, dtype=torch.float64)          # key 侧 hidden states
hq = torch.randn(T, d, dtype=torch.float64)         # query 侧
W_DKV = torch.randn(d_c, d, dtype=torch.float64) * (d ** -0.5)     # 联合下投影（Eq.1）
W_UK = torch.randn(n_h, d_nope, d_c, dtype=torch.float64) * (d_c ** -0.5)  # 上投影（Eq.2）
W_UV = torch.randn(n_h, d_v, d_c, dtype=torch.float64) * (d_c ** -0.5)
W_Q = torch.randn(n_h, d_nope, d, dtype=torch.float64) * (d ** -0.5)

c = h @ W_DKV.T                                     # [S, d_c] 压缩 latent（cache 里存的就是它）
q = torch.einsum("td,hnd->thn", hq, W_Q)            # [T, n_h, d_nope]

# ---- [B1] absorbed 路径 ≡ naive per-head（无 RoPE 部分）----
k_naive = torch.einsum("sc,hnc->shn", c, W_UK)      # [S, n_h, d_nope] 物化每头 key
s_naive = torch.einsum("thn,shn->ths", q, k_naive)  # 逐头点积
q_abs = torch.einsum("thn,hnc->thc", q, W_UK)       # 把 W_UK 吸收进 query（model.py:L483）
s_abs = torch.einsum("thc,sc->ths", q_abs, c)       # 直接在 latent 上 attention（L486）
d_nope_max = (s_naive - s_abs).abs().max().item()
v_naive = torch.einsum("sc,hvc->shv", c, W_UV)
o_naive = torch.einsum("ths,shv->thv", s_naive.softmax(-1), v_naive)
o_abs = torch.einsum("ths,sc->thc", s_naive.softmax(-1), c)
o_abs = torch.einsum("thc,hvc->thv", o_abs, W_UV)   # 输出侧也走 latent（model.py:L494-495）
d_out_max = (o_naive - o_abs).abs().max().item()
print(f"  [B1] absorbed ≡ naive: 分数 max|Δ| = {d_nope_max:.2e}, 输出 max|Δ| = {d_out_max:.2e} (fp64 舍入级)")
check("B1 absorbed 路径与 naive 逐头 attention 等价 (<1e-12)", d_nope_max < 1e-12 and d_out_max < 1e-12,
      f"Δscore={d_nope_max:.1e}, Δout={d_out_max:.1e}")

# ---- [B2] 为什么必须解耦 RoPE：耦合 RoPE 破坏吸收 ----
W_KR = torch.randn(d_rope, d, dtype=torch.float64) * (d ** -0.5)
k_pe = rope_pairs(h @ W_KR.T)                       # [S, d_rope] 位置键（cache 里存的第二件）
q_pe = rope_pairs(torch.einsum("td,rd->tr", hq, W_KR))  # [T, d_rope]（toy 下 query 侧同投影）

# 参照：耦合 RoPE —— 把 RoPE 直接加在压缩-上投影后的每头 key 上（即「不解耦」的世界）
k_coup = rope_pairs(k_naive.reshape(S, n_h * d_nope), theta=100.0).reshape(S, n_h, d_nope)
q_coup = rope_pairs(q.reshape(T, n_h * d_nope), theta=100.0).reshape(T, n_h, d_nope)
s_ref_coup = torch.einsum("thn,shn->ths", q_coup, k_coup)

# 尝试：耦合 RoPE 下仍想吸收 —— W_UK 已吸收进 query，位置信息无处安放（V2 §2.1.3 的困境）
s_coup_blind = torch.einsum("thc,sc->ths", q_abs, c)     # 无位置：吸收后给不出相对位置
err_coup = (s_ref_coup - s_coup_blind).abs().max().item()

# 解耦方案（V2/V3 实际设计）：语义部分走吸收，位置部分用独立小键
s_pe = torch.einsum("tr,sr->ts", q_pe, k_pe)             # 相对位置自动成立（RoPE 性质）
s_decoup = s_abs + s_pe.unsqueeze(1).expand(-1, n_h, -1)
k_decoup_naive = k_naive                                # 语义 key 不带 RoPE
s_decoup_ref = torch.einsum("thn,shn->ths", q, k_decoup_naive) + s_pe.unsqueeze(1).expand(-1, n_h, -1)
err_decoup = (s_decoup - s_decoup_ref).abs().max().item()

# RoPE 相对位置性质：q/k 位置同平移 δ=2，pe 分数不变（<R_{t+δ}q, R_{s+δ}k> = <R_t q, R_s k>）
delta = 2
q_unrot = hq @ W_KR.T
k_unrot = h @ W_KR.T
q2 = rope_pairs(torch.cat([torch.zeros(delta, d_rope, dtype=torch.float64), q_unrot]))[delta:]
k2 = rope_pairs(torch.cat([torch.zeros(delta, d_rope, dtype=torch.float64), k_unrot]))[delta:]
s_pe_shift = torch.einsum("tr,sr->ts", q2, k2)
rel_invariant = (s_pe - s_pe_shift).abs().max().item()

cache_coupled = n_h * (d_nope + d_rope)
cache_decoupled = d_c + d_rope
print(f"  [B2] 耦合 RoPE + 盲吸收: max|Δ| vs 参照 = {err_coup:.3f}  ← 位置信息丢失（V2 §2.1.3 的困境）")
print(f"      解耦 RoPE（语义吸收 + 独立 rope 键）: max|Δ| = {err_decoup:.2e} ← 吸收等价性保住")
print(f"      RoPE 相对位置性质（平移不变）: max|Δ| = {rel_invariant:.2e}")
print(f"      cache/token: 耦合须存每头 key = {cache_coupled} 值; 解耦 = d_c+d_rope = {cache_decoupled} 值")
check("B2 耦合 RoPE 使吸收失效 (Δ>0.5)", err_coup > 0.5, f"Δ={err_coup:.3f}")
check("B2 解耦 RoPE 恢复吸收等价 (<1e-12)", err_decoup < 1e-12, f"Δ={err_decoup:.1e}")
check("B2 平移不变性（相对位置）", rel_invariant < 1e-10, f"Δ={rel_invariant:.1e}")

# ======================================================================
# [C] FP8：细粒度量化 + 高精度累加（V3 §3.3；模拟量化误差，计算在 fp32/fp64）
# ======================================================================
print()
print("[C] FP8 E4M3: fine-grained scaling + high-precision accumulation")
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max                      # 448.0


def qdq(x, block):
    """分块 scale 的量化-反量化：scale = 块内 max|·| / 448（在线量化，V3 §3.3.1 口径）。"""
    shape = x.shape
    xv = x.reshape(-1, block)
    s = xv.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / FP8_MAX
    return ((xv / s).to(FP8).to(torch.float32) * s).reshape(shape)


# ---- [C1] delayed vs online scaling：新兴 outlier（V3 §3.3.1 Online Quantization）----
# FP8 浮点量化的相对误差对 scale 不变（格式自带指数）；细粒度 scale 真正修的
# 是「delayed tensor-wise scale 用历史 amax、当前步出现新 outlier 时溢出截断」。
torch.manual_seed(7)
xc = torch.randn(1024)                          # step t-1：无 outlier
s_delayed = xc.abs().max() / FP8_MAX            # delayed tensor-wise scale（历史 amax）
xa = xc.clone(); xa[0], xa[7] = 40.0, -40.0     # step t：2 元素突增（新兴 outlier）
q_delayed = (xa / s_delayed).clamp(-FP8_MAX, FP8_MAX).to(FP8).to(torch.float32) * s_delayed
q_online = qdq(xa, 128)                         # online 细粒度 1×128（V3 tile-wise 口径）
outlier_idx = torch.tensor([0, 7])
rel_d = ((q_delayed - xa)[outlier_idx].abs() / xa[outlier_idx].abs()).mean().item()
rel_o = ((q_online - xa)[outlier_idx].abs() / xa[outlier_idx].abs()).mean().item()
clip_n = int(((xa.abs() / s_delayed) > FP8_MAX).sum())
print(f"  [C1] delayed tensor-wise scale（上步 amax={xc.abs().max():.2f}）遇新兴 outlier ±40:")
print(f"      delayed: {clip_n} 元素溢出截断, outlier 相对误差 = {rel_d*100:.1f}%（被夹到 ±{FP8_MAX*s_delayed:.2f}）")
print(f"      online 1×128 细粒度: outlier 相对误差 = {rel_o*100:.2f}%（scale 随当前块在线计算）")
check("C1 online 细粒度救回新兴 outlier (delayed>50% 且 online<5%)",
      rel_d > 0.5 and rel_o < 0.05, f"delayed={rel_d:.2f}, online={rel_o:.3f}")

# ---- [C2] 累加精度：低精度累加 vs 高精度累加（V3 §3.3.2「promotion to CUDA Cores」的动机）----
torch.manual_seed(8)
K = 4096
a = torch.randn(K); b = torch.randn(K)
aq = qdq(a, 128).double(); bq = qdq(b, 128).double()
ref_dot = (aq * bq).sum()
acc_fp32 = torch.zeros(1, dtype=torch.float64)
acc_fp16 = torch.zeros(1, dtype=torch.float16)
acc_fp8 = torch.zeros(1, dtype=torch.float32)
for i in range(K):
    p = aq[i] * bq[i]
    acc_fp32 += p
    acc_fp16 = (acc_fp16 + p.float()).half()         # 每步舍入到 fp16（模拟有限累加精度）
    acc_fp8 = (acc_fp8 + p).to(FP8).to(torch.float32)  # 每步舍入到 fp8（极端情形）
e16 = abs(acc_fp16.item() - ref_dot.item()) / abs(ref_dot.item())
e8 = abs(acc_fp8.item() - ref_dot.item()) / abs(ref_dot.item())
e32 = abs(acc_fp32.item() - ref_dot.item()) / abs(ref_dot.item())
print(f"  [C2] K=4096 内积（量化后同一输入，只变累加精度）:")
print(f"      fp64 累加参照 = {ref_dot.item():.6f}")
print(f"      fp32 累加相对误差 = {e32:.2e} | fp16 累加 = {e16*100:.2f}% | fp8 累加 = {e8*100:.1f}%")
print(f"      V3 实测口径（原文声称）: Tensor Core 有限累加精度 K=4096 最大相对误差近 2%")
check("C2 累加精度越低误差越大 (fp8>fp16>fp32)", e8 > e16 > e32,
      f"fp8={e8:.3f} > fp16={e16:.4f} > fp32={e32:.1e}")

# ---- [C3] GEMM 口径：截断误差经矩阵乘传播到输出 ----
torch.manual_seed(9)
M, N, Kb = 64, 64, 512
A = torch.randn(M, Kb); B = torch.randn(Kb, N)
s_del = A.abs().max() / FP8_MAX                 # delayed tensor-wise scale（历史 amax）
A_sp = A.clone(); A_sp[::8, 63] *= 30.0         # 当前步：列 63 突增
Aq_del = (A_sp / s_del).clamp(-FP8_MAX, FP8_MAX).to(FP8).to(torch.float32) * s_del
Aq_on = qdq(A_sp, 128)                          # online 1×128 tile scale
ref_mm = A_sp.double() @ B.double()
rel_t = ((Aq_del.double() @ B.double() - ref_mm).abs() / ref_mm.abs().clamp_min(1)).mean().item()
rel_b = ((Aq_on.double() @ B.double() - ref_mm).abs() / ref_mm.abs().clamp_min(1)).mean().item()
print(f"  [C3] 64×512·512×64 矩阵乘（A 列 63 突增，E4M3 量化-反量化后 fp64 乘）:")
print(f"      delayed tensor-wise scale: 平均相对误差 = {rel_t*100:.2f}%（截断误差经 matmul 传播）")
print(f"      online 1×128 tile scale: 平均相对误差 = {rel_b*100:.2f}%  (= {rel_t/rel_b:.1f}× 改善)")
check("C3 online 细粒度降低 GEMM 误差 (>3×)", rel_t / rel_b > 3, f"{rel_t/rel_b:.2f}x")

# ======================================================================
# [D] 训练稳定性：梯度范数裁剪（V3 §4.2 原文：「The gradient clipping norm is
#     set to 1.0」——V3 唯一披露的稳定性旋钮；loss spike 本身是集群级现象，
#     toy 不可复现，此处只演示 clip 这一旋钮的机制）
# ======================================================================
print()
print("[D] Stability: gradient norm clipping (V3 §4.2 clip norm = 1.0)")
torch.manual_seed(10)
P_DIM = 1024
LR, MAX_NORM, SPIKE = 0.1, 1.0, 50.0
g_normal = 0.01 * torch.randn(P_DIM)                    # 正常步：||g|| ≈ 0.01·√P
spike_dir = torch.randn(P_DIM)
spike_dir = spike_dir / spike_dir.norm()
g_spike = g_normal + SPIKE * spike_dir                  # spike 步：梯度范数突增
n_normal, n_spike = g_normal.norm().item(), g_spike.norm().item()
upd_noclip = LR * g_spike                               # 无 clip：整步吃下 spike
g_clipped = g_spike * (MAX_NORM / g_spike.norm())       # clip：缩到范数恰为 1.0
upd_clip = LR * g_clipped
cos_dir = torch.dot(g_clipped.double(), g_spike.double()) / (g_clipped.double().norm() * g_spike.double().norm())
normal_triggers = bool(g_normal.norm() > MAX_NORM)
print(f"    正常步 ||g|| = {n_normal:.3f} → spike 步 ||g|| = {n_spike:.2f} ({n_spike/n_normal:.0f}× 突增)")
print(f"    无 clip: ||Δθ|| = {upd_noclip.norm():.3f} (单步跳 {n_spike/n_normal:.0f}× 于正常步 {LR*n_normal:.3f})")
print(f"    clip={MAX_NORM}: ||Δθ|| = {upd_clip.norm():.3f} (= lr×{MAX_NORM} 恰; 为正常步 {upd_clip.norm()/(LR*n_normal):.1f}×), 方向保持 cos = {cos_dir:.6f}")
print(f"    正常步触发 clip = {normal_triggers} ← 保险丝平时不可见，只在 spike 时熔断")
check("D1 spike 使梯度范数突增 (>30×)", n_spike / n_normal > 30, f"{n_spike/n_normal:.0f}x")
check("D2 clip 后更新范数恰为 lr×max_norm", abs(upd_clip.norm().item() - LR * MAX_NORM) < 1e-6,
      f"{upd_clip.norm().item():.6f} vs {LR*MAX_NORM}")
check("D3 clip 只缩步长不改方向 (cos>1-1e-9)", cos_dir.item() > 1 - 1e-9, f"cos={cos_dir:.9f}")
check("D4 正常步不触发 clip", not normal_triggers, f"||g_normal||={n_normal:.3f} < {MAX_NORM}")

# ======================================================================
# [E] self-check 汇总 + digest
# ======================================================================
print()
print("[E] self-check")
n_pass = sum(1 for _, ok, _ in CHECKS if ok)
for name, ok, detail in CHECKS:
    if not ok:
        print(f"    FAIL  {name} ({detail})")
ok_all = n_pass == len(CHECKS)
print(f"    {'✅' if ok_all else '❌'} self-check {'passed' if ok_all else 'FAILED'} ({n_pass}/{len(CHECKS)})")

metrics = {
    "total_recount_B": round(total_recount / 1e9, 2),
    "act_recount_B": round(act_recount / 1e9, 2),
    "cv0": round(cv0, 4), "dead0": dead0,
    "ratio_max": round(ratio_max, 4), "ratio_min": round(ratio_min, 4),
    "ratio_max_big": round(ratio_max_big, 4),
    "same_weights": bool(same_weights), "bias_not_leaked": bool(bias_not_leaked),
    "selection_changed": bool(selection_changed),
    "kv_ratio": round(mha_kv / mla_kv, 3),
    "d_nope": float(f"{d_nope_max:.3e}"), "d_out": float(f"{d_out_max:.3e}"),
    "err_coup": round(err_coup, 4), "err_decoup": float(f"{err_decoup:.3e}"),
    "rel_inv": float(f"{rel_invariant:.3e}"),
    "rel_d": round(rel_d, 5), "rel_o": round(rel_o, 5), "clip_n": clip_n,
    "e16": round(e16, 6), "e8": round(e8, 4), "e32": float(f"{e32:.3e}"),
    "rel_t": round(rel_t, 5), "rel_b": round(rel_b, 5),
    "n_normal": round(n_normal, 4), "n_spike": round(n_spike, 4),
    "upd_clip_norm": round(upd_clip.norm().item(), 6), "cos_dir": round(cos_dir.item(), 9),
    "n_pass": n_pass, "n_checks": len(CHECKS),
}
digest = hashlib.md5(repr(sorted(metrics.items())).encode()).hexdigest()
print(f"\ndigest(md5 of metrics) = {digest}")
if not ok_all:
    sys.exit(1)
