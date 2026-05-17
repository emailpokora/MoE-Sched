#!/usr/bin/env python3
"""Qwen1.5-MoE ablation runs to strengthen paper Table 2.

Adds four configurations not in the original Table 2:

  1. nocache (cap=1, LRU) — isolates expert-aware-loading benefit from
     cache-hit benefit. Every dispatch is a miss, so any speedup over
     device_map='auto' is attributable to managed transfer + skeleton-on-GPU,
     not caching.

  2. per_layer_uniform (4/layer, 96 total) — per-layer caches with identical
     uniform capacity. Same total budget as the paper's "balanced" row, but
     split across 24 independent layer caches.

  3. epcb (entropy-allocated, 96 total) — EPCB end-to-end. Same total
     budget as balanced/per_layer_uniform, but Shannon-entropy-allocated.
     Measures EPCB's wall-clock benefit, not just hit-rate.

  4. adaptive_eviction (LRU then LFU) — starts with LRU, adapts to LFU when
     hit_rate < 0.08 for 30 accesses. Demonstrates the adapt mechanism on
     a non-EPCB axis.

Results saved to figures/qwen_ablation_results.json.

Usage:
    python scripts/bench_qwen_ablation.py --runs 5
    python scripts/bench_qwen_ablation.py --runs 3 --max-tokens 32
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

# Point HF cache at D: drive where Qwen weights actually live
os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/hf_cache/hub")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import psutil

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "Explain the key ideas behind mixture-of-experts models."
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Four new ablation policies. Each one targets a specific reviewer concern.
POLICIES = {
    # Exp 1: isolates expert-aware loading from caching.
    # prefetch budget=1 needed because validator enforces budget <= capacity
    # even when strategy=none (the default).
    "nocache (cap=1)": (
        "policy nocache { "
        "cache { capacity = 1  eviction = lru } "
        "prefetch { budget = 1 } "
        "}"
    ),
    # Exp 2a: per-layer caches with no entropy adaptation (uniform).
    "per_layer_uniform (4/layer)": (
        "policy per_layer_uniform { "
        "cache { capacity = 4  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 2 } "
        "per_layer { "
        "  allocation = entropy "
        "  entropy_window = 200 "
        "  min_capacity = 4  max_capacity = 4 "
        "  rebalance_interval = 999999 "
        "  total_budget = 96 "
        "} "
        "}"
    ),
    # Exp 2b: EPCB end-to-end.
    "epcb (96 total, entropy)": (
        "policy epcb_qwen { "
        "cache { capacity = 4  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 2 } "
        "per_layer { "
        "  allocation = entropy "
        "  entropy_window = 200 "
        "  min_capacity = 2  max_capacity = 12 "
        "  rebalance_interval = 500 "
        "  total_budget = 96 "
        "} "
        "}"
    ),
    # Exp 3: adaptive eviction LRU->LFU.
    "adaptive (LRU->LFU)": (
        "policy adaptive_eviction { "
        "cache { capacity = 4  eviction = lru } "
        "prefetch { strategy = history  budget = 2 } "
        "adapt { "
        "  when hit_rate < 0.08 for 30 accesses { eviction = lfu } "
        "} "
        "}"
    ),
}


def fmt(vals):
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    return f"{m:.2f} +/- {s:.2f}"


def attach_policy(model, dsl):
    """Like moe_policylang.attach() but threads num_layers/num_experts
    through build_hook so per_layer policies work.

    The library's attach() drops those, which makes per_layer policies fail.
    """
    from moe_policylang.parser import parse_policy
    from moe_policylang.compiler import compile_policy
    from moe_policylang.runtime.hooks import build_hook
    from moe_policylang.integrations.accessors import auto_accessor
    from moe_policylang.integrations.weight_placement import WeightPlacementManager

    accessor = auto_accessor(model)
    ir = parse_policy(dsl)
    compiled = compile_policy(ir)
    hook = build_hook(
        compiled,
        num_layers=accessor.num_layers,
        num_experts=accessor.num_experts,
    )
    mgr = WeightPlacementManager(hook, accessor)
    mgr.attach()
    return mgr


def reset_stats(mgr):
    """Reset stats for either a flat PolicyHook or a PerLayerHook."""
    hook = mgr.hook
    # Per-layer hook: reset each layer's cache stats
    if hasattr(hook, "_per_layer_hooks"):
        for layer_hook in hook._per_layer_hooks.values():
            layer_hook.cache.stats.hits = 0
            layer_hook.cache.stats.misses = 0
            layer_hook.cache.stats.evictions = 0
    # Adaptive wrapper: reset inner
    elif hasattr(hook, "inner") and hasattr(hook.inner, "cache"):
        hook.inner.cache.stats.hits = 0
        hook.inner.cache.stats.misses = 0
        hook.inner.cache.stats.evictions = 0
    # Plain hook
    elif hasattr(hook, "cache"):
        hook.cache.stats.hits = 0
        hook.cache.stats.misses = 0
        hook.cache.stats.evictions = 0
    # Placement stats
    if hasattr(mgr, "stats"):
        mgr.stats.cpu_to_gpu_transfers = 0
        mgr.stats.gpu_to_cpu_transfers = 0
        mgr.stats.bytes_transferred = 0
        mgr.stats.transfer_time_s = 0.0


def run_once(model, tok, mgr, max_tokens):
    reset_stats(mgr)
    inp = tok(PROMPT, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gen_tokens = out.shape[1] - inp["input_ids"].shape[1]
    tps = gen_tokens / elapsed
    gpu_gb = torch.cuda.memory_allocated() / 1e9
    stats = mgr.get_stats()
    hr = stats["policy"]["cache"]["hit_rate"]
    xfers = stats["placement"]["cpu_to_gpu_transfers"]
    return tps, gpu_gb, hr, xfers, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "-n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--only", type=str, default=None,
                    help="Run only policies whose key contains this substring")
    args = ap.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    ram = psutil.virtual_memory()
    print(f"GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")
    print(f"RAM: {ram.total/1e9:.1f} GB total, {ram.available/1e9:.1f} GB available")
    print(f"HF cache: {os.environ.get('HF_HOME')}")
    print(f"Runs per config: {args.runs} (+ {args.warmup} warmup)")
    print()

    results = {
        "model": MODEL_ID,
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "runs": args.runs,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
        "policies": {},
    }

    import moe_policylang

    print("=" * 60)
    print("Loading Qwen1.5-MoE-A2.7B with expert-aware device map...")
    print("=" * 60)
    t_load = time.perf_counter()
    model, tok = moe_policylang.load_moe_model(MODEL_ID, trust_remote_code=True)
    load_s = time.perf_counter() - t_load
    skeleton_gb = torch.cuda.memory_allocated() / 1e9
    results["skeleton_gpu_gb"] = skeleton_gb
    results["load_time_s"] = load_s
    print(f"  Skeleton: {skeleton_gb:.2f} GB on GPU (load took {load_s:.1f}s)")
    print()

    for name, dsl in POLICIES.items():
        if args.only and args.only not in name:
            continue
        print("=" * 60)
        print(f"POLICY: {name}")
        print("=" * 60)
        print(f"  DSL: {dsl.strip()}")

        mgr = attach_policy(model, dsl)

        for w in range(args.warmup):
            tps, gpu, hr, xf, _ = run_once(model, tok, mgr, args.max_tokens)
            print(f"  warmup {w+1}: {tps:.2f} tok/s  hit={hr:.1%}")

        tps_list, gpu_list, hr_list, xf_list = [], [], [], []
        last_stats = None
        for r in range(args.runs):
            tps, gpu, hr, xf, stats = run_once(model, tok, mgr, args.max_tokens)
            tps_list.append(tps)
            gpu_list.append(gpu)
            hr_list.append(hr)
            xf_list.append(xf)
            last_stats = stats
            print(f"  run {r+1}/{args.runs}: {tps:.2f} tok/s  GPU={gpu:.1f} GB  "
                  f"hit={hr:.1%}  transfers={xf}")

        entry = {
            "dsl": dsl.strip(),
            "tps_values": tps_list,
            "tps_mean": statistics.mean(tps_list),
            "tps_std": statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0,
            "gpu_gb_mean": statistics.mean(gpu_list),
            "hit_rate_values": hr_list,
            "hit_rate_mean": statistics.mean(hr_list),
            "transfers_mean": statistics.mean(xf_list),
        }
        # Capture per-layer info for EPCB / per_layer_uniform
        if last_stats and "per_layer" in last_stats.get("policy", {}):
            entry["per_layer_capacities"] = last_stats["policy"].get("capacities")
            entry["per_layer_entropies"] = last_stats["policy"].get("entropies")
        results["policies"][name] = entry

        print(f"  -> {fmt(tps_list)} tok/s   hit={statistics.mean(hr_list)*100:.1f}%")
        print()

        mgr.detach()
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<35s}  {'tok/s':<18s}  {'GPU (GB)':<10s}  {'Hit %':<10s}")
    print("-" * 80)
    for name, p in results["policies"].items():
        tps_str = f"{p['tps_mean']:.2f} +/- {p['tps_std']:.2f}"
        hr_str = f"{p['hit_rate_mean']*100:.1f}%"
        print(f"{name:<35s}  {tps_str:<18s}  {p['gpu_gb_mean']:<10.1f}  {hr_str:<10s}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "qwen_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
