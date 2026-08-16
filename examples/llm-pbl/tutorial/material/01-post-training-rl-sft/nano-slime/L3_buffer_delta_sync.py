#!/usr/bin/env python3
"""
nano-slime L3 — data buffer 回收 × delta weight sync：把 L2 的两个抽象对象拆到源码级

L2 把 slime 的两条训练主循环（train.py / train_async.py）建模成 regime，留下两个
「只知其名」的对象：data buffer（L0 的抽象队列）与 delta weight sync（L2 §8 只论证
了它为什么存在 = 把 S 做小）。L3 对照 slime 源码（HEAD 2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d，
2026-08-16 codeload tarball 抓取）把这两个对象的**实现机制**变成可运行的本质模拟：

  A. data buffer = RolloutDataSourceWithBuffer（slime/rollout/data_source.py）：
     - buffer 就是一个 list（:L171），采样**先 buffer 后 dataset**（:L182-188），
       默认 filter 是 pop_first FIFO（:L225-229）；
     - partial rollout（--partial-rollout，arguments.py:L456-465）：一轮凑满
       target 后 abort 在途请求（sglang_rollout.py:L451），未完成的组带着
       start_rollout_id 版本戳（:L363-364）回收进 buffer（:L648），下一轮优先消费。
       长 response 的半截工作从「丢弃」变「库存」。
  B. delta weight sync = UpdateWeightFromDiskDelta
     （slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py）：
     - 首次调用只抓 baseline 快照、不发布（:L84-88）；快照种子取自 hf_checkpoint
       而非 GPU 权重（:L95-101，docs delta-weight-sync.md:L37-41）；
     - 每次 sync：diff against 上一版快照 → 编码（xor / overwrite，:L231-239）
       → zstd-1 压缩（:L242）→ 新态 checksum（:L243）→ 写成 canonical HF 目录
       （index 带 version/base_version/encoding 元数据，:L157-167）→ 引擎
       pull → pause → flush_cache → reload → continue（:L176-189）；
     - **未变化的 tensor 整块跳过**（:L240-241），快照滚动推进到新值（:L247）；
     - 两种编码的本质差异（docs:L67-78）：xor 是对合（involution）——必须对正确
       的 base 恰好施加一次，施加两次还原；overwrite 写「位置+新值」，幂等
       （disk_delta.py:L21-25 docstring 明载）。
     - delta ⊥ colocate：colocate 走 CUDA IPC，只有 handle 跨进程，delta 的
       snapshot+diff+encode 是纯开销（arguments.py:L2057-2061 原句）。

三个定量结论（全部由本文件模拟器产出，tutorial 逐条对照源码）：
  1. partial rollout 不改变「每轮训 target 组」的吞吐结构（buffer 买不到吞吐，
     承 L1/L2），但把 abort 丢弃的 token 变成 0：半截工作带版本戳回库、下轮
     FIFO 优先消费，代价是这批样本 staleness ≥ 1（版本戳就是 off-policy 度数）。
  2. delta 的 wire bytes = f(变化密度)：未变 tensor 零代价 + 压缩吃零字节，
     低密度下 wire 只占全量百分之几（实测值见输出）；但 diff/encode 要扫全量
     （memory-bandwidth-bound，disk_delta.py:L11-13），所以 delta 只在
     「字节真要过网络/盘 且 密度够低」时赢——breakeven 密度由实测标定。
  3. 把 S_delta 灌回 L2 的 max(G,T)+S：interval=k 把 (S+C_flush) 一起摊薄，
     staleness 上界仍是 k（结构性）；colocate regime 下 delta 恒亏（断言证明）。

诚实口径：本文件是**可运行的本质模拟**（本课程 L2/L3 可运行性契约）——本机没有
GPU/多机/共享文件系统，跑不了真实 SGLang/Megatron/NCCL。建模的是 slime 源码
坐实的结构事实（行号见上；来源快照与复核锚见 tutorial §13）：buffer-first
采样序、abort 回收与版本戳、diff→编码→压缩→校验→滚动基线链、xor/overwrite
的代数性质（对合 vs 幂等）、版本链与 checksum 的失败模式。压缩用标准库
zlib-1 显式代替 zstd-1（slime 实测 zstd-1 在 wire size 与解压速度上压过
lz4/gzip/snappy/brotli 故不可调，docs:L33——nano 只取「压缩吃零字节」这一
结构性质，绝对压缩率不可外推）。时间常数是 toy 口径（承 L2），结构结论可外推、
绝对值不可。真机验证 [TODO: verify on real system] 走 GPU 通道。
依赖：零（纯标准库）。CPU 瞬时（<0.2s）。
"""

import hashlib
import random
import struct
import time
import zlib

# ================= [0] data buffer：buffer-first × partial rollout 回收 =================
# 对照：data_source.py:L171（buffer=[]）/ L177-189（get_samples 先 buffer 后 dataset）/
#       L225-229（pop_first FIFO）/ sglang_rollout.py:L407（while len(data)<target）/
#       L451（abort）/ L363-364（start_rollout_id 戳）/ L648（add_samples 回收）

N_GROUPS = 24        # dataset 里的 prompt 组数（group = 一个 prompt 的 n_samples_per_prompt 条，此处按组计）
TARGET = 8           # 每轮训练批 = rollout_batch_size 组
OVER = 2             # 过采样余量（sglang_rollout.py:L407-410 的 over_sampling_batch_size 语义，toy 常数）
ROUNDS = 6           # 模拟轮数
MEAN_LEN = 128       # 每组生成满的 token 数（L1 实测同量级口径）


class DataSourceWithBuffer:
    """data_source.py RolloutDataSourceWithBuffer 的最小忠实版：
    get_samples 先 buffer（pop_first）后 dataset（游标+epoch 回绕）；add_samples 回库。"""

    def __init__(self, n_groups, seed):
        rng = random.Random(seed)
        # 长尾工作分布：多数组一轮内做得完，少数组要跨轮（deterministic）
        self.total_work = [MEAN_LEN + rng.choice([0, 0, 0, 96, 160, 224]) for _ in range(n_groups)]
        self.cursor = 0            # sample_offset（data_source.py:L57）
        self.epoch = 0
        self.buffer = []           # L171
        self.pops = 0              # buffer-first 命中计数（nano 自加，用于守恒律）
        self.fetched_dataset = 0

    def get_samples(self, n):
        out = []
        while len(out) < n and self.buffer:                 # L182-186：先 buffer
            g = self.buffer.pop(0)                          # pop_first FIFO（L225-229）
            g["from_buffer"] = True
            self.pops += 1
            out.append(g)
        while len(out) < n:                                 # L188：不足从 dataset 补
            if self.cursor >= N_GROUPS:                     # epoch 回绕（L96-103）
                self.cursor = 0
                self.epoch += 1
            out.append({"gid": self.cursor, "done": 0, "start_rollout_id": None,
                        "total": self.total_work[self.cursor], "from_buffer": False})
            self.cursor += 1
            self.fetched_dataset += 1
        return out

    def add_samples(self, groups):                          # L198-211
        self.buffer.extend(groups)


def sim_rounds(partial_rollout, seed=7):
    """连续批语义的离散事件模拟：每轮取 TARGET+OVER 组（buffer-first 过采样）→
    所有在途组每迭代各推进 1 token（continuous batching 本质）→ 第 TARGET 个完成
    即轮末（sglang_rollout.py:L407 while 循环的退出条件），剩余在途组 abort（L451）。
    partial_rollout=True：在途组带 done 进度 + start_rollout_id 戳回库（L363-364/L648）；
    partial_rollout=False：abort 的组直接丢弃（L356-357 continue，进度作废）。"""
    ds = DataSourceWithBuffer(N_GROUPS, seed)
    trained, engine_tokens, wasted, extra_waste = [], 0, 0, 0
    aborts = extras = 0
    for r in range(ROUNDS):
        pending = ds.get_samples(TARGET + OVER)             # buffer-first 过采样取组
        remain = [g["total"] - g["done"] for g in pending]
        order = sorted(range(len(pending)), key=lambda i: (remain[i], i))
        tau = remain[order[TARGET - 1]]                     # 第 TARGET 个完成的时刻（迭代数）
        # 同一迭代同时完成的组可能多于 TARGET：前 TARGET 个进训练批，超出的
        # 按 slime 语义直接丢弃、不回库（sglang_rollout.py:L439-440 NOTE 明载）
        completers = [i for i in range(len(pending)) if remain[i] <= tau]
        train_idx = set(sorted(completers, key=lambda i: (remain[i], i))[:TARGET])
        for i, g in enumerate(pending):
            g = dict(g)
            if remain[i] <= tau:                            # 完成（进度 = remain[i]）
                g["done"] = g["total"]
                engine_tokens += remain[i]
                if i in train_idx:
                    trained.append((r, g))
                else:
                    extras += 1
                    extra_waste += remain[i]                # 超额完成：训了没人要，也不回库
            else:                                           # 在途 → abort（进度 = tau）
                g["done"] += tau
                engine_tokens += tau
                aborts += 1
                if partial_rollout:
                    if g["done"] > 0 and g["start_rollout_id"] is None:
                        g["start_rollout_id"] = r           # 版本戳（L363-364）
                    ds.add_samples([g])                     # 回收（L648）
                else:
                    wasted += g["done"]                     # 丢弃：进度作废
    residual = sum(g["done"] for g in ds.buffer)
    return {"trained": trained, "engine_tokens": engine_tokens, "wasted": wasted,
            "extra_waste": extra_waste, "aborts": aborts, "extras": extras,
            "residual": residual, "ds": ds}


def run_buffer_demo():
    print("[0] data buffer：buffer-first 采样 × partial rollout 回收")
    print(f"    workload: {N_GROUPS} 组 dataset / 每轮取 {TARGET}+{OVER} 组（过采样）/ 连续批"
          f"（每迭代各组 +1 tok）/ {ROUNDS} 轮；长尾组 total∈{{128,224,288,352}}")
    on = sim_rounds(partial_rollout=True)
    off = sim_rounds(partial_rollout=False)
    n_on, n_off = len(on["trained"]), len(off["trained"])
    assert n_on == n_off == ROUNDS * TARGET, "两变体每轮都应凑满 target"
    assert on["aborts"] + on["extras"] == ROUNDS * OVER == off["aborts"] + off["extras"], \
        "守恒：每轮取 = 训 + 超额完成 + abort"
    print(f"    ✓ 两变体各训 {n_on} 组 = {ROUNDS}×{TARGET}（吞吐结构不变——buffer 买不到吞吐，"
          f"承 L1/L2）；过采样余量去向 = abort {on['aborts']} + 超额完成 {on['extras']}"
          f"（超额完成不回库：sglang_rollout.py:L439-440 NOTE 同款语义）")
    assert off["wasted"] > 0, "无 partial rollout 时 abort 进度应被丢弃"
    assert on["wasted"] == 0, "partial rollout 下不应有丢弃 token"
    # 严格守恒律（机器证明，非近似）：
    sum_tr_on = sum(g["total"] for _, g in on["trained"])
    sum_tr_off = sum(g["total"] for _, g in off["trained"])
    assert off["engine_tokens"] == sum_tr_off + off["wasted"] + off["extra_waste"], \
        "off：引擎 token = 训练组总量 + abort 丢弃 + 超额完成丢弃"
    assert on["engine_tokens"] == sum_tr_on + on["extra_waste"] + on["residual"], \
        "on：引擎 token = 训练组总量 + 超额完成丢弃 + 期末 buffer 残留进度"
    print(f"    ✓ 守恒律：off 引擎 {off['engine_tokens']} tok = 训练 {sum_tr_off} + abort 丢弃 "
          f"{off['wasted']} + 超额 {off['extra_waste']}；on = 训练 {sum_tr_on} + 超额 "
          f"{on['extra_waste']} + 残留 {on['residual']}")
    print(f"      abort 丢弃的 {off['wasted']} tok 在 off 里永久作废（dataset 槽位同失）；"
          f"on 里 abort 进度 0 丢弃——半截工作带戳回库")
    recycled = [(r, g) for r, g in on["trained"] if g["from_buffer"]]
    assert recycled, "应存在被回收再训练的组（长尾存在性）"
    stale = [r - g["start_rollout_id"] for r, g in recycled]
    assert all(s >= 1 for s in stale), "回收组的 staleness 应 ≥1（生成在旧版、训练在新版）"
    print(f"    ✓ 回收组 {len(recycled)} 个，staleness = train_round − start_rollout_id ∈ "
          f"{sorted(set(stale))}——版本戳就是 off-policy 度数（mask-offpolicy-in-partial-rollout "
          f"是算法侧对策，arguments.py:L467-474）")
    # dataset 游标守恒：off 全从 dataset 取；on 的 buffer 命中顶替 dataset 消费
    ds_on, ds_off = on["ds"], off["ds"]
    tot_fetched = ROUNDS * (TARGET + OVER)
    assert ds_off.fetched_dataset == tot_fetched and ds_off.pops == 0, "off：buffer 恒空，全从 dataset 取"
    assert ds_on.fetched_dataset + ds_on.pops == tot_fetched, "on：dataset + buffer 命中 = 总取组数"
    cur_on = ds_on.epoch * N_GROUPS + ds_on.cursor
    cur_off = ds_off.epoch * N_GROUPS + ds_off.cursor
    assert cur_off == tot_fetched and cur_on == tot_fetched - ds_on.pops
    print(f"    ✓ 游标守恒：off dataset 取 {tot_fetched} 组（游标 {cur_off}）；on buffer 命中 "
          f"{ds_on.pops} 次顶替 dataset 消费（游标 {cur_on}）——回收另省 dataset 槽位"
          f"（prompt 昂贵时 [agent 轨迹] 这是第二笔账）")
    return on, off


# ================= [1] delta chain：seed → diff → 编码 → 压缩 → 校验 → 滚动基线 =================
# 对照：update_weight_from_disk_delta.py:L84-88（首调只抓基线）/ L199-273（_encode_delta）/
#       L231-239（xor/overwrite 分支）/ L240-241（未变 tensor 跳过）/ L242（zstd-1）/
#       L243（新态 checksum）/ L247（快照滚动）/ L157-167（index 元数据）/ L286-287（metrics）

TENSOR_SIZES = [262144, 131072, 65536, 65536, 32768, 16384]   # 6 个 tensor，共 573,440 B
NAMES = [f"t{i}" for i in range(len(TENSOR_SIZES))]
TOTAL_BYTES = sum(TENSOR_SIZES)


def make_world(seed):
    rng = random.Random(seed)
    return {n: rng.randbytes(s) for n, s in zip(NAMES, TENSOR_SIZES)}


def mutate(world, spec, seed):
    """spec = {name: density}。确定性挑位置、换新值；其余 tensor 不动（测未变跳过）。"""
    rng = random.Random(seed)
    changed = {}
    for name, d in spec.items():
        buf = bytearray(world[name])
        n = len(buf)
        k = int(n * d)
        pos = rng.sample(range(n), k)
        newvals = rng.randbytes(k)
        for i, p in enumerate(pos):
            buf[p] = newvals[i]
        world[name] = bytes(buf)
        changed[name] = sorted(pos)
    return changed


def adler32_hex(b):
    return f"{zlib.adler32(b) & 0xFFFFFFFF:08x}"    # nano 用 adler32；slime 默认 xxh3-128（arguments.py:L186-198）


def xor_encode(new, old):
    return bytes(a ^ b for a, b in zip(new, old))


def xor_apply(base, payload):
    return bytes(a ^ b for a, b in zip(base, payload))


def overwrite_encode(new, old):
    """disk_delta.py:L21-25 同款布局：u4 变更数 + u4 位置序列 + 新值字节（little-endian）。"""
    pos = [i for i, (a, b) in enumerate(zip(new, old)) if a != b]
    out = struct.pack("<I", len(pos)) + b"".join(struct.pack("<I", p) for p in pos)
    return out + bytes(new[p] for p in pos)


def overwrite_apply(base, payload):
    (k,) = struct.unpack_from("<I", payload, 0)
    off = 4 + 4 * k
    vals = payload[off:]
    assert len(vals) == k, "overwrite payload 值段长度应 = 变更数"
    buf = bytearray(base)
    for i in range(k):
        (p,) = struct.unpack_from("<I", payload, 4 + 4 * i)
        buf[p] = vals[i]
    return bytes(buf)


class DeltaChain:
    """trainer 侧 publish + engine 侧 apply 的最小忠实版（共享文件系统 = 内存 dict）。
    publish：diff against 快照 → 编码 → zlib-1（nano 代替 zstd-1，见 docstring）→ 校验和
    → index{version, base_version, delta_encoding, ...}；快照滚动（L247）。
    apply：版本链校验（只允许施加在声明的 base 上，docs:L85-86）→ 解码施加 → 新态校验。"""

    def __init__(self, world, encoding):
        self.encoding = encoding
        self.snapshot = dict(world)          # seed（L95-125 的种子语义，nano 直接取当前世界）
        self.versions = {}                   # 共享文件系统上的 weight_v{N:06d}/ 目录族
        self.version = 0
        self.wire_history = []

    def publish(self, world):
        # slime 首次 update_weights 只抓基线不发布（L84-88）；nano 的基线在构造时即持有，
        # 故每次 publish 都对应一个真实版本。
        self.version += 1
        delta, checksums, wire = {}, {}, 0
        changed_bytes = total = 0
        for name in NAMES:
            new, old = world[name], self.snapshot[name]
            total += len(new)
            if new == old:                              # L240-241：未变 tensor 整块跳过
                continue
            changed_bytes += sum(a != b for a, b in zip(new, old))
            payload = xor_encode(new, old) if self.encoding == "xor" else overwrite_encode(new, old)
            blob = zlib.compress(payload, 1)            # L242 的 zstd-1 → nano 以 zlib-1 显式代替
            delta[name] = blob
            checksums[name] = adler32_hex(new)          # L243：新态 checksum
            wire += len(blob)
            self.snapshot[name] = new                   # L247：滚动基线
        index = {"version": f"{self.version:06d}", "base_version": f"{self.version - 1:06d}",
                 "delta_encoding": self.encoding, "compression_format": "zlib-1 (nano stand-in for zstd-1)",
                 "checksum_format": "adler32", "weight_map": {n: f"model-{i:05d}.blob" for i, n in enumerate(delta)}}
        wire += len(str(index))                          # index 开销（L157-167 的元数据）
        self.versions[self.version] = {"index": index, "delta": delta, "checksums": checksums}
        self.wire_history.append(wire)
        return {"version": self.version, "wire": wire, "changed_tensors": sorted(delta),
                "density": changed_bytes / max(total, 1), "total": total, "changed_bytes": changed_bytes}

    def apply(self, base_world, target_version):
        v = self.versions[target_version]
        declared_base = int(v["index"]["base_version"])
        if getattr(base_world, "version", None) != declared_base:
            raise ValueError(f"version {target_version:06d} 只允许施加在 base {declared_base:06d} 上"
                             f"（当前 {getattr(base_world, 'version', None)}）——版本链拒绝乱序（docs:L85-86）")
        out = VersionedDict(base_world)
        for name, blob in v["delta"].items():
            payload = zlib.decompress(blob)
            fn = xor_apply if self.encoding == "xor" else overwrite_apply
            new = fn(out[name], payload)
            if adler32_hex(new) != v["checksums"][name]:
                raise ValueError(f"{name} checksum 不符——delta 损坏或 base 错误，fail loud（docs:L82-86）")
            out[name] = new
        out.version = target_version
        return out


class VersionedDict(dict):
    def __init__(self, src=None, version=0):
        super().__init__(src or {})
        self.version = version


def run_delta_demo():
    print("\n[1] delta chain：seed → diff → 编码 → 压缩 → 校验 → 滚动基线")
    world = VersionedDict(make_world(11), version=0)
    chain = DeltaChain(world, encoding="xor")
    engine = VersionedDict(world, version=0)   # 引擎侧 base = seed（pull_weights(0) 物化本地基座语义）
    print(f"    世界：{len(NAMES)} tensor / 全量 {TOTAL_BYTES:,} B；基线 = seed（首调不发布，L84-88）")

    # step 1：变 {t0,t2,t4}，各 2%
    changed1 = mutate(world, {"t0": 0.02, "t2": 0.02, "t4": 0.02}, seed=21)
    pub1 = chain.publish(world)
    assert pub1["changed_tensors"] == ["t0", "t2", "t4"], "未变 tensor 不应进 delta（L240-241）"
    print(f"    ✓ v1: 变 {pub1['changed_tensors']}（t1/t3/t5 未变 → 零代价跳过），"
          f"density={pub1['density']:.4f}，wire={pub1['wire']:,} B"
          f"（全量 {TOTAL_BYTES:,} B 的 {pub1['wire'] / TOTAL_BYTES:.2%}）")

    eng1 = chain.apply(engine, 1)
    assert all(eng1[n] == world[n] for n in NAMES), "apply 后引擎世界应与 trainer 逐字节同一"
    assert hashlib.md5(b"".join(eng1[n] for n in NAMES)).hexdigest() == \
        hashlib.md5(b"".join(world[n] for n in NAMES)).hexdigest()
    print(f"    ✓ engine apply v1：md5 逐字节吻合 trainer（版本链 0→1）")

    # step 2：变 {t0,t1}，各 1% —— diff 必须对 v1 基线（滚动，L247），不是 v0
    changed2 = mutate(world, {"t0": 0.01, "t1": 0.01}, seed=22)
    pub2 = chain.publish(world)
    assert pub2["changed_tensors"] == ["t0", "t1"], "v2 只应含本轮新变的 tensor"
    print(f"    ✓ v2: 变 {pub2['changed_tensors']}，density={pub2['density']:.4f}，"
          f"wire={pub2['wire']:,} B——diff 对 v1 基线（快照滚动 L247），t2/t4 的上轮变更不再上线")

    # 版本链：把 v2 施加在独立 v0 基座上 → 必须拒绝
    try:
        chain.apply(VersionedDict(make_world(11), version=0), 2)   # v2 声明 base=000001
        raise AssertionError("乱序 apply 未被拒绝")
    except ValueError as e:
        print(f"    ✓ 版本链拒绝乱序：v2 施加在 v0 基座 → ValueError（{str(e).split('——')[0]}）")

    eng2 = chain.apply(eng1, 2)
    assert all(eng2[n] == world[n] for n in NAMES)
    print(f"    ✓ engine apply v2（基座 v1）：md5 逐字节吻合——链 0→1→2 闭合")

    # checksum fail-loud：篡改引擎基座 1 字节再施加 → 新态校验不符
    corrupt = VersionedDict(eng1, version=1)
    bad = bytearray(corrupt["t0"]); bad[0] ^= 0xFF; corrupt["t0"] = bytes(bad)
    v = chain.versions[2]
    payload = zlib.decompress(v["delta"]["t0"])
    new = xor_apply(corrupt["t0"], payload)
    assert adler32_hex(new) != v["checksums"]["t0"], "篡改后 checksum 应不符"
    print(f"    ✓ checksum fail-loud：base 被篡改 1 字节 → 新态 adler32 不符（宁可报错，不发坏权重）")
    return pub1, pub2


# ================= [2] xor vs overwrite：对合 vs 幂等 =================
# 对照：disk_delta.py:L21-25（overwrite 布局 + 幂等 docstring）/ docs:L67-78（xor 对合、
#       overwrite 幂等）/ arguments.py:L173-184（默认 xor）/ update_weight_from_disk_delta.py:L231-239

N_BIG = 262144


def run_encoding_demo():
    print("\n[2] 编码代数：xor = 对合（须恰好一次） vs overwrite = 幂等（可重复施加）")
    rng = random.Random(31)
    base = rng.randbytes(N_BIG)

    def apply_changes(buf_positions_vals):
        new = bytearray(base)
        for p, v in buf_positions_vals:
            new[p] = v
        return bytes(new)

    rows = []
    for d in (0.001, 0.01, 0.05, 0.20):
        k = int(N_BIG * d)
        pos = rng.sample(range(N_BIG), k)
        vals = rng.randbytes(k)
        new = apply_changes(zip(pos, vals))
        c = sum(a != b for a, b in zip(new, base))      # 新值可能恰等于旧值 → c ≤ k
        px, po = xor_encode(new, base), overwrite_encode(new, base)
        cx, co = len(zlib.compress(px, 1)), len(zlib.compress(po, 1))
        rows.append((c / N_BIG, c, len(px), len(po), cx, co))
    print(f"    N={N_BIG:,} B 单 tensor，**均匀随机**变更，密度 d 下 raw / 压缩后 wire（zlib-1）：")
    for d, c, rx, ro, cx, co in rows:
        print(f"      d={d:6.3%}: xor raw={rx:>8,} zip={cx:>7,} | overwrite raw={ro:>9,} zip={co:>7,}"
              f" | xor 更省：{cx < co}")
    assert all(rx == N_BIG for _, _, rx, _, _, _ in rows), "xor raw 应恒 = N（dtype-blind 字节级）"
    assert all(ro == 4 + 5 * c for _, c, _, ro, _, _ in rows), "overwrite raw 应 = 4+5c（u4 数 + u4 位置 + 值）"
    print(f"    ✓ raw 账：xor ≡ N；overwrite = 4+5c（c=实际变更字节数）——与 overwrite_encode 布局逐字节对账")
    lo, hi = rows[1], rows[3]     # d≈1% 与 d≈20%
    assert lo[5] < lo[4], "均匀稀疏变更下 overwrite 压缩后应更省（位置+值列表比散布非零字节好压）"
    assert hi[4] < hi[5], "高密度下 xor 压缩后应更省（overwrite 的 4B/位置开销主导）"
    print(f"    ✓ 均匀随机变更下交叉点在 1%–20% 之间——**overwrite 在低密度反而更省**，"
          f"与 docs:L72『xor smallest wire』相反？看下一探针")

    # 块状变更探针：同密度，变更聚集成连续块（真实权重 diff 的形态：整行/整块参数一起动）
    d = 0.01
    k = int(N_BIG * d)
    nblk, blk = 8, k // 8
    starts = rng.sample(range(0, N_BIG - blk), nblk)
    clustered = [(p + s, rng.randbytes(1)[0]) for s in starts for p in range(blk)]
    new_c = apply_changes(clustered)
    c_c = sum(a != b for a, b in zip(new_c, base))
    cx_c = len(zlib.compress(xor_encode(new_c, base), 1))
    co_c = len(zlib.compress(overwrite_encode(new_c, base), 1))
    print(f"    块状探针（d={c_c / N_BIG:.3%}，{nblk} 个连续块）：xor zip={cx_c:,} vs overwrite zip={co_c:,}"
          f" → xor 更省：{cx_c < co_c}")
    assert cx_c < co_c, "块状变更下 xor 应反超（长零游程 + 密集块都好压；overwrite 仍付 4B/位置）"
    print(f"    ✓ 交叉位置取决于**变更分布**：均匀散布利于 overwrite、块状聚集利于 xor——"
          f"slime 默认 xor（arguments.py:L175）的依据是真实 diff 结构 + zstd-1 实测"
          f"（docs:L33 profiling 压过 lz4/gzip/snappy/brotli）；nano 的均匀随机 toy "
          f"复现机制、不复现 profiling 数字（绝对压缩率不可外推）")

    # 对合 vs 幂等：同一 delta 施加两次
    d_ = 0.01
    k = int(N_BIG * d_)
    pos = rng.sample(range(N_BIG), k)
    vals = rng.randbytes(k)
    new = apply_changes(zip(pos, vals))
    dx, do = xor_encode(new, base), overwrite_encode(new, base)
    once_x, twice_x = xor_apply(base, dx), xor_apply(xor_apply(base, dx), dx)
    once_o, twice_o = overwrite_apply(base, do), overwrite_apply(overwrite_apply(base, do), do)
    assert once_x == new and twice_x == base, "xor：一次=新态，两次=还原（对合）"
    assert once_o == new and twice_o == new, "overwrite：两次=一次=新态（幂等）"
    print(f"    ✓ xor 施加两次 → 还原回 base（involution 机器证明）——『必须对正确 base 恰好一次』（docs:L72-74）")
    print(f"    ✓ overwrite 施加两次 → 仍是新态（idempotent 机器证明）——重试/断点续传安全（docs:L75-78）")
    return rows, (c_c, cx_c, co_c)


# ================= [3] regime 整合：S_delta 灌回 L2 的 max(G,T)+S =================
# 对照：L2 常数承 nano-slime L2_engine_cost_async_regimes.py（G16=168.96 / T=73.46 / S=7.35）；
#       delta ⊥ colocate：arguments.py:L2057-2061；pull→pause→flush→reload→continue：
#       update_weight_from_disk_delta.py:L176-189（flush_cache = prefix cache 失效代价）

W_READ, KV_STEP = 1.0, 0.02          # nano-vllm-sglang L0 同款（承 L2）
G16 = 1 * 128 * (W_READ + 16 * KV_STEP)     # = 168.96
T = round(G16 / 2.3, 2)                       # = 73.46（L1 实测 G/T=2.3 口径，承 L2）
S_FULL = round(0.1 * T, 2)                    # = 7.35（L2 的 S：一次全量推送 = 10% T）
BW_NET = TOTAL_BYTES / S_FULL                 # 网络带宽：全量推送恰 S_FULL
BW_SCAN = 8 * BW_NET                          # diff/encode 扫描带宽（memory-bandwidth-bound，disk_delta.py:L11-13；toy 倍率 8×）
C_FLUSH = 0.5                                 # 每次 sync 的 pause+flush_cache+reload 固定代价（toy，prefix cache 冷启动）
S_COLOC = 0.3                                 # colocate CUDA IPC：只有 handle 跨进程（toy 常数，arguments.py:L2058-2060）


def s_delta(wire):
    """delta 一次 sync 的 toy 时间 = 全量扫描（diff 躲不掉）+ wire 过网络。"""
    return TOTAL_BYTES / BW_SCAN + wire / BW_NET


def run_regime_demo(pub1):
    print("\n[3] regime 整合：S_delta 灌回 max(G,T)+S，delta ⊥ colocate")
    d, wire = pub1["density"], pub1["wire"]
    sd = s_delta(wire)
    print(f"    承 L2：G={G16:.2f} / T={T} / S_full={S_FULL}；BW_NET={BW_NET:,.0f} B/toy-t，"
          f"BW_SCAN=8×BW_NET（toy），C_flush={C_FLUSH}")
    print(f"    d={d:.4f} 时：wire={wire:,} B → S_delta={sd:.3f} vs S_full={S_FULL}"
          f"（省 {1 - sd / S_FULL:.1%}）；其中扫描底 {TOTAL_BYTES / BW_SCAN:.3f} 是 delta 躲不掉的")
    assert sd < S_FULL, "低密度下 delta 应显著快于全量（disaggregated regime）"

    # breakeven：wire 多大时 delta 与 full 打平——由 [1] 实测标定，不是解析假设
    alpha = pub1["wire"] / pub1["density"]            # toy 标定：单位密度对应的 wire（v1 实测）
    wire_be = (S_FULL - TOTAL_BYTES / BW_SCAN) * BW_NET
    d_star = wire_be / alpha
    assert 0 < d_star < 1, "breakeven 密度应落在 (0,1)"
    print(f"    ✓ breakeven（toy 实测标定）：α = wire/density = {alpha:,.0f} B（v1 标定），"
          f"wire_be = (S_full−扫描底)×BW_NET = {wire_be:,.0f} B → d* = {d_star:.1%}"
          f"——密度高于 d* 时 diff 扫描底 + wire 反超全量直推")

    # interval 摊薄：async 每步 = max(G,T) + (S + C_flush)/k（L2 闭式 + flush 摊销）
    for k in (1, 2, 4):
        step_full = max(G16, T) + (S_FULL + C_FLUSH) / k
        step_delta = max(G16, T) + (sd + C_FLUSH) / k
        print(f"      interval={k}: 每步 full={step_full:.2f} → delta={step_delta:.2f}"
              f"（省 {(S_FULL - sd) / k:.3f}；staleness 上界 = k，结构性，承 L2）")
    step1f, step1d = max(G16, T) + S_FULL + C_FLUSH, max(G16, T) + sd + C_FLUSH
    assert step1d < step1f
    print(f"    ✓ flush 与 S 同被 interval 摊薄，但都不改 staleness 上界——interval 仍是 staleness 旋钮")

    # colocate：delta 恒亏（机器证明决策规则）
    sd_coloc = S_COLOC + TOTAL_BYTES / BW_SCAN       # IPC 推送不变，白付扫描底
    assert sd_coloc > S_COLOC, "colocate 下 delta 应恒亏"
    print(f"    ✓ colocate：S={S_COLOC}（handle-only）→ delta 强加扫描底 {TOTAL_BYTES / BW_SCAN:.3f}"
          f" → {sd_coloc:.3f} > {S_COLOC}——『delta bookkeeping 是纯开销』（arguments.py:L2058-2060 原句坐实）")
    print(f"    ✓ 决策规则（机器验证）：delta 仅当 ① 字节真要过网络/盘（nccl/disk disaggregated）"
          f"且 ② 变化密度 < d* ≈ {d_star:.1%}；colocate 永远 full-IPC")
    return sd, d_star


# ================= [4] self-check 汇总 + digest =================

def main():
    t0 = time.perf_counter()
    print("=" * 68)
    print("nano-slime L3 — data buffer 回收 × delta weight sync（源码级机制模拟）")
    print("溯源：THUDM/slime @ 2fa9a442（2026-08-16 抓取）；行锚见各段注释")
    print("=" * 68)
    on, off = run_buffer_demo()
    pub1, pub2 = run_delta_demo()
    rows, (c_c, cx_c, co_c) = run_encoding_demo()
    sd, d_star = run_regime_demo(pub1)

    digest_src = "|".join([
        f"eng_tok_on={on['engine_tokens']}", f"eng_tok_off={off['engine_tokens']}",
        f"wasted_off={off['wasted']}", f"extra_on={on['extra_waste']}", f"residual_on={on['residual']}",
        f"recycled={sum(1 for _, g in on['trained'] if g['from_buffer'])}",
        f"v1_wire={pub1['wire']}", f"v1_density={pub1['density']:.4f}",
        f"v2_wire={pub2['wire']}", f"v2_density={pub2['density']:.4f}",
        *[f"unif_{i}_d={d:.6f}_cx={cx}_co={co}" for i, (d, c, rx, ro, cx, co) in enumerate(rows)],
        f"cluster_c={c_c}_cx={cx_c}_co={co_c}",
        f"s_delta={sd:.3f}", f"d_star={d_star:.3f}",
    ])
    digest = hashlib.md5(digest_src.encode()).hexdigest()
    print(f"\n    digest(metrics) = {digest}")

    print("\n" + "=" * 68)
    print("✅ self-check passed: buffer-first/回收零丢弃/版本戳 staleness≥1 · 未变跳过/滚动基线/"
          "版本链/checksum · xor 对合 vs overwrite 幂等 · delta⊥colocate/breakeven 密度")
    print("=" * 68)
    print(f"\ntakeaway: data buffer 的价值不在吞吐（L1/L2 已证 buffer 买不到吞吐），而在把")
    print(f"          abort 的半截长尾工作带版本戳回库——staleness≥1 是它的价格，")
    print(f"          mask-offpolicy 是算法侧对策。delta sync 把 S 从『全量过网』降成")
    print(f"          『扫描底 + 密度×全量过网』：本 toy 密度 {pub1['density']:.2%} 时省 "
          f"{1 - sd / S_FULL:.1%}（breakeven d*≈{d_star:.1%}，实测标定），但 diff")
    print(f"          躲不掉全量扫描，密度高于 d* 反亏；colocate 走 CUDA IPC")
    print(f"          （handle-only）时 delta 是纯开销——slime 直接 raise 禁止该组合。")
    print(f"          真机验证（SGLang /pull_weights + Megatron gather + 共享盘）[TODO: verify on real system]")
    print(f"elapsed: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
