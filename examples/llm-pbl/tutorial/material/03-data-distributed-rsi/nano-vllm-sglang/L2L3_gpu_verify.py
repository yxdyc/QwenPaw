#!/usr/bin/env python3
"""GPU empirical probe for nano-vllm-sglang L2/L3.

L2 — real-engine PagedAttention batch decode throughput (radix cache disabled
     to isolate paging).
L3 — real-engine RadixAttention prefix-cache speedup (shared system prompt).

Uses SGLang offline Engine with attention_backend="torch_native" so it runs
without flashinfer JIT; the measured numbers therefore carry a "torch_native"
backend caveat and should be treated as representative anchors, not absolute
peak-GPU figures.

Run with an explicit local model or model identifier:
    python3 -B L2L3_gpu_verify.py --model /path/to/model [--device 0] [--quick]

No log file is written unless --log is provided. Timings are observations of
the named model, SGLang revision, backend and GPU; they are not portable constants.
"""

import argparse
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

MODEL_PATH = ""
LOG_PATH: Optional[str] = None
DEVICE = "0"
ATTENTION_BACKEND = "torch_native"
MEM_FRACTION = 0.15
QUICK = False


def log(msg: str) -> None:
    print(msg)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8", buffering=1) as f:
            f.write(msg + "\n")


def gpu_memory_mb() -> int:
    """Return used GPU memory (MiB) for the visible device via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                DEVICE,
            ],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return -1


def make_engine(disable_radix_cache: bool):
    from sglang.srt.entrypoints.engine import Engine

    log(
        f"Building Engine (disable_radix_cache={disable_radix_cache}, "
        f"attention_backend={ATTENTION_BACKEND}, disable_piecewise_cuda_graph=True) ..."
    )
    llm = Engine(
        model_path=MODEL_PATH,
        tp_size=1,
        dtype="bfloat16",
        context_length=2048,
        mem_fraction_static=MEM_FRACTION,
        attention_backend=ATTENTION_BACKEND,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_radix_cache=disable_radix_cache,
        random_seed=42,
        log_level="error",
    )
    log("Engine ready.")
    return llm


def generate_batch(llm, prompts: List[str], max_new_tokens: int) -> List[Dict]:
    """Run a batched generation and return per-output metadata."""
    outputs = llm.generate(
        prompts,
        sampling_params={"temperature": 0.0, "max_new_tokens": max_new_tokens},
    )
    return outputs


def l2_paged_batch_decode(llm) -> Dict:
    """
    L2: measure decode throughput across batch sizes with a fixed prompt.
    Radix cache is disabled so the speedup comes from PagedAttention + batching.
    """
    log("=" * 60)
    log("L2: PagedAttention batch decode throughput")
    log("=" * 60)

    prompt = "The capital of France is"
    max_new_tokens = 30
    batch_sizes = [1, 2, 4, 8]
    n_repeats = 1 if QUICK else 3

    # Warmup at B=2.
    log("Warming up at B=2 ...")
    _ = generate_batch(llm, [prompt] * 2, max_new_tokens)

    results = {}
    for b in batch_sizes:
        # Dry run once to stabilise allocator state.
        _ = generate_batch(llm, [prompt] * b, max_new_tokens)

        mem_before = gpu_memory_mb()
        latencies = []
        completed_tokens = []
        throughputs = []
        for r in range(n_repeats):
            t0 = time.perf_counter()
            outs = generate_batch(llm, [prompt] * b, max_new_tokens)
            t1 = time.perf_counter()
            lat = t1 - t0
            latencies.append(lat)
            # Count actual completions: EOS may stop before max_new_tokens.
            ntoks = [o["meta_info"]["completion_tokens"] for o in outs]
            completed = sum(ntoks)
            completed_tokens.append(completed)
            throughputs.append(completed / lat)
            log(f"  B={b} repeat={r+1}/{n_repeats} latency={lat:.3f}s tokens={ntoks}")
        mem_after = gpu_memory_mb()

        median_lat = sorted(latencies)[len(latencies) // 2]
        median_tokens = sorted(completed_tokens)[len(completed_tokens) // 2]
        throughput = sorted(throughputs)[len(throughputs) // 2]
        results[b] = {
            "median_latency_s": median_lat,
            "total_tokens": median_tokens,
            "requested_tokens": b * max_new_tokens,
            "throughput_tok_s": throughput,
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "mem_delta_mb": mem_after - mem_before,
        }
        log(
            f"L2 B={b}: median_latency={median_lat:.3f}s, "
            f"throughput={throughput:.1f} tok/s, "
            f"mem_used={mem_after}MiB (delta={mem_after - mem_before}MiB)"
        )
    return results


def l3_radix_prefix_cache(llm) -> Dict:
    """
    L3: measure RadixAttention prefix-cache speedup with a shared system prompt.
    """
    log("=" * 60)
    log("L3: RadixAttention prefix-cache speedup")
    log("=" * 60)

    system_prompt = (
        "You are a helpful assistant. Answer concisely. "
        "The following facts are known: Paris is the capital of France. "
        "Rome is the capital of Italy. Berlin is the capital of Germany. "
        "Madrid is the capital of Spain. Lisbon is the capital of Portugal. "
        "Vienna is the capital of Austria. Brussels is the capital of Belgium. "
        "Amsterdam is the capital of the Netherlands. Athens is the capital of Greece. "
        "Warsaw is the capital of Poland."
    )
    user_queries = [
        "What is the capital of Germany?",
        "What is the capital of Spain?",
        "What is the capital of Italy?",
        "What is the capital of Austria?",
    ]
    max_new_tokens = 20
    n_warm = 1 if QUICK else 4

    shared_prefix = f"Variant S. {system_prompt}"
    shared_prompts = [f"{shared_prefix}\n\nUser: {q}\nAssistant:" for q in user_queries]
    unique_prefixes = [
        f"Variant {tag}. {system_prompt}" for tag in ("A", "B", "C", "D")
    ]
    unique_prompts = [
        f"{p}\n\nUser: {q}\nAssistant:"
        for p, q in zip(unique_prefixes, user_queries)
    ]

    # Cold run: single request to populate radix cache.
    log("L3 cold run (populate cache) ...")
    t0 = time.perf_counter()
    cold_out = generate_batch(llm, [shared_prompts[0]], max_new_tokens)
    cold_lat = time.perf_counter() - t0
    cold_cached = cold_out[0]["meta_info"].get("cached_tokens", 0)
    log(f"  cold latency={cold_lat:.3f}s cached_tokens={cold_cached}")

    # Warm batched run: shared prefix should be fully cached.
    warm_latencies = []
    cached_tokens_list = []
    warm_prompt_tokens = []
    for r in range(n_warm):
        t0 = time.perf_counter()
        warm_outs = generate_batch(llm, shared_prompts, max_new_tokens)
        t1 = time.perf_counter()
        lat = t1 - t0
        warm_latencies.append(lat)
        cached = sum(o["meta_info"].get("cached_tokens", 0) for o in warm_outs)
        cached_tokens_list.append(cached)
        warm_prompt_tokens.append(sum(o["meta_info"].get("prompt_tokens", 0) for o in warm_outs))
        ntoks = [o["meta_info"]["completion_tokens"] for o in warm_outs]
        log(
            f"  warm batch repeat={r+1}/{n_warm} latency={lat:.3f}s "
            f"cached_tokens={cached} completion_tokens={ntoks}"
        )

    median_warm_lat = sorted(warm_latencies)[len(warm_latencies) // 2]
    median_cached = sorted(cached_tokens_list)[len(cached_tokens_list) // 2]

    # No-cache baseline: same queries but each has a unique prefix.
    log("L3 no-cache baseline (unique prefixes) ...")
    no_cache_latencies = []
    no_cache_prompt_tokens = []
    for r in range(n_warm):
        t0 = time.perf_counter()
        outs = generate_batch(llm, unique_prompts, max_new_tokens)
        t1 = time.perf_counter()
        lat = t1 - t0
        no_cache_latencies.append(lat)
        cached = sum(o["meta_info"].get("cached_tokens", 0) for o in outs)
        no_cache_prompt_tokens.append(sum(o["meta_info"].get("prompt_tokens", 0) for o in outs))
        log(f"  no-cache repeat={r+1}/{n_warm} latency={lat:.3f}s cached_tokens={cached}")

    median_no_cache_lat = sorted(no_cache_latencies)[len(no_cache_latencies) // 2]
    median_warm_prompt_tokens = sorted(warm_prompt_tokens)[len(warm_prompt_tokens) // 2]
    median_no_cache_prompt_tokens = sorted(no_cache_prompt_tokens)[len(no_cache_prompt_tokens) // 2]

    # The speedup metric compares per-request warm (shared prefix cached) vs
    # per-request no-cache (unique prefix).  We use wall-clock per batch / 4.
    warm_per_req = median_warm_lat / len(user_queries)
    no_cache_per_req = median_no_cache_lat / len(user_queries)
    speedup = no_cache_per_req / warm_per_req if warm_per_req > 0 else 0.0

    results = {
        "cold_latency_s": cold_lat,
        "cold_cached_tokens": cold_cached,
        "warm_median_latency_s": median_warm_lat,
        "warm_median_cached_tokens": median_cached,
        "no_cache_median_latency_s": median_no_cache_lat,
        "warm_prompt_tokens": median_warm_prompt_tokens,
        "no_cache_prompt_tokens": median_no_cache_prompt_tokens,
        "warm_per_request_latency_s": warm_per_req,
        "no_cache_per_request_latency_s": no_cache_per_req,
        "prefix_speedup": speedup,
    }
    log(
        f"L3 summary: warm_per_req={warm_per_req:.3f}s, "
        f"no_cache_per_req={no_cache_per_req:.3f}s, speedup={speedup:.2f}x, "
        f"warm_cached_tokens={median_cached}, prompt_tokens="
        f"{median_warm_prompt_tokens}/{median_no_cache_prompt_tokens} (warm/control)"
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure SGLang batching and radix-prefix behavior on one visible GPU."
    )
    parser.add_argument("--model", required=True, help="local model path or explicit model identifier")
    parser.add_argument(
        "--device",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="physical CUDA device id exposed to the process",
    )
    parser.add_argument("--log", help="optional log file; stdout-only by default")
    parser.add_argument("--attention-backend", default="torch_native")
    parser.add_argument("--mem-fraction", type=float, default=0.15)
    parser.add_argument("--quick", action="store_true", help="one repeat per measurement")
    args = parser.parse_args()
    if not 0.05 <= args.mem_fraction <= 0.95:
        parser.error("--mem-fraction must be in [0.05, 0.95]")
    return args


def main():
    global MODEL_PATH, LOG_PATH, DEVICE, ATTENTION_BACKEND, MEM_FRACTION, QUICK
    args = parse_args()
    MODEL_PATH = args.model
    LOG_PATH = args.log
    DEVICE = args.device
    ATTENTION_BACKEND = args.attention_backend
    MEM_FRACTION = args.mem_fraction
    QUICK = args.quick
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

    if LOG_PATH:
        open(LOG_PATH, "w", encoding="utf-8").close()
    log("L2L3_gpu_verify starting")
    log(f"MODEL_PATH={MODEL_PATH}")
    log(f"CUDA_VISIBLE_DEVICES={DEVICE}")
    log(f"attention_backend={ATTENTION_BACKEND} mem_fraction_static={MEM_FRACTION} quick={QUICK}")

    try:
        import sglang

        log(f"sglang_version={getattr(sglang, '__version__', 'unknown')}")
        # L2: disable radix cache to isolate PagedAttention / paging effects.
        llm_l2 = make_engine(disable_radix_cache=True)
        l2_results = l2_paged_batch_decode(llm_l2)
        del llm_l2

        # L3: enable radix cache for prefix sharing measurement.
        llm_l3 = make_engine(disable_radix_cache=False)
        l3_results = l3_radix_prefix_cache(llm_l3)
        del llm_l3

        log("=" * 60)
        log("FINAL ANCHORS (torch_native backend)")
        log("=" * 60)
        for b, r in sorted(l2_results.items()):
            log(
                f"L2 B={b}: {r['throughput_tok_s']:.1f} tok/s "
                f"({r['median_latency_s']:.3f}s for {r['total_tokens']} tokens), "
                f"GPU mem ~{r['mem_after_mb']}MiB"
            )
        log(
            f"L3 prefix speedup: {l3_results['prefix_speedup']:.2f}x "
            f"({l3_results['warm_per_request_latency_s']:.3f}s vs "
            f"{l3_results['no_cache_per_request_latency_s']:.3f}s per request), "
            f"warm cached_tokens={l3_results['warm_median_cached_tokens']}"
        )
        payload = {
            "schema_version": "1.0",
            "module": "nano-vllm-sglang-l2l3-gpu",
            "environment": {
                "model": MODEL_PATH,
                "device": DEVICE,
                "sglang": getattr(sglang, "__version__", "unknown"),
                "attention_backend": ATTENTION_BACKEND,
            },
            "metrics": {
                "l2_throughput_tok_s": {
                    str(b): round(r["throughput_tok_s"], 3)
                    for b, r in sorted(l2_results.items())
                },
                "l3_prefix_speedup": round(l3_results["prefix_speedup"], 4),
                "l3_warm_cached_tokens": l3_results["warm_median_cached_tokens"],
                "l3_prompt_tokens_warm": l3_results["warm_prompt_tokens"],
                "l3_prompt_tokens_control": l3_results["no_cache_prompt_tokens"],
            },
            "checks": {
                "all_generations_completed": True,
                "warm_prefix_cache_observed": l3_results["warm_median_cached_tokens"] > 0,
                "prompt_token_accounting_available": (
                    l3_results["warm_prompt_tokens"] > 0
                    and l3_results["no_cache_prompt_tokens"] > 0
                ),
                "matched_prompt_budget": (
                    l3_results["warm_prompt_tokens"] > 0
                    and abs(
                        l3_results["warm_prompt_tokens"]
                        - l3_results["no_cache_prompt_tokens"]
                    )
                    / max(l3_results["warm_prompt_tokens"], 1)
                    <= 0.05
                ),
            },
            "evidence_boundary": (
                "single-model single-GPU SGLang observation; timings depend on model, "
                "revision, backend, driver and load"
            ),
        }
        log("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
        log("L2L3_gpu_verify completed successfully")
    except Exception as e:
        log(f"L2L3_gpu_verify FAILED: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
