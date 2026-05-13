#!/usr/bin/env python3
"""Multi-run Qwen1.5-MoE throughput benchmark for paper rigor.

Runs each policy configuration N times and reports mean ± std for
throughput (tok/s), VRAM (GB), and hit rate.  Results are saved to
figures/qwen_multirun_results.json.

Usage (close IDE first to maximize free RAM):
    python scripts/bench_qwen_multirun.py
    python scripts/bench_qwen_multirun.py --runs 5
    python scripts/bench_qwen_multirun.py --runs 3 --skip-baseline
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import psutil

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "Explain the key ideas behind mixture-of-experts models."
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

POLICIES = {
    "aggressive (cap=2, LRU)": (
        "policy a { cache { capacity = 2  eviction = lru } "
        "prefetch { strategy = history  budget = 1 } }"
    ),
    "balanced (cap=4, LFU)": (
        "policy b { cache { capacity = 4  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 2 } }"
    ),
    "generous (cap=8, LFU)": (
        "policy c { cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 4 } }"
    ),
}


def fmt(vals):
    """Format a list of floats as mean ± std."""
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    return f"{m:.2f} ± {s:.2f}"


def run_once(model, tok, mgr, max_tokens):
    """Single inference run.  Returns (tok/s, gpu_gb, hit_rate, transfers)."""
    # Reset stats for a clean measurement
    if hasattr(mgr, 'hook'):
        mgr.hook.cache.stats.hits = 0
        mgr.hook.cache.stats.misses = 0
        mgr.hook.cache.stats.evictions = 0
    if hasattr(mgr, 'stats'):
        mgr.stats.cpu_to_gpu_transfers = 0
        mgr.stats.gpu_to_cpu_transfers = 0
        mgr.stats.bytes_transferred = 0
        mgr.stats.transfer_time_s = 0.0

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
    return tps, gpu_gb, hr, xfers


def run_baseline_once(model, tok, max_tokens):
    """Single baseline inference run.  Returns (tok/s, gpu_gb)."""
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
    return tps, gpu_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "-n", type=int, default=3,
                    help="Number of runs per configuration (default: 3)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip standard loading benchmark")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Warmup runs before measurement (default: 1)")
    args = ap.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    ram = psutil.virtual_memory()
    print(f"GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")
    print(f"RAM: {ram.total/1e9:.1f} GB total, {ram.available/1e9:.1f} GB available")
    print(f"Runs per config: {args.runs} (+ {args.warmup} warmup)")
    print()

    results = {
        "model": MODEL_ID,
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "runs": args.runs,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
    }

    # ── Baseline ──────────────────────────────────────────────────────
    if not args.skip_baseline:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("=" * 60)
        print(f"BASELINE: device_map='auto' ({args.warmup}w + {args.runs}r)")
        print("=" * 60)
        try:
            full_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True)
            full_model.eval()
            btok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            btok.pad_token = btok.eos_token

            # Warmup
            for w in range(args.warmup):
                tps, gpu = run_baseline_once(full_model, btok, args.max_tokens)
                print(f"  warmup {w+1}: {tps:.2f} tok/s")

            # Measured runs
            b_tps_list, b_gpu_list = [], []
            for r in range(args.runs):
                tps, gpu = run_baseline_once(full_model, btok, args.max_tokens)
                b_tps_list.append(tps)
                b_gpu_list.append(gpu)
                print(f"  run {r+1}/{args.runs}: {tps:.2f} tok/s  GPU={gpu:.1f} GB")

            results["baseline"] = {
                "tps_values": b_tps_list,
                "tps_mean": statistics.mean(b_tps_list),
                "tps_std": statistics.stdev(b_tps_list) if len(b_tps_list) > 1 else 0.0,
                "gpu_gb": statistics.mean(b_gpu_list),
            }
            print(f"  → {fmt(b_tps_list)} tok/s  GPU={statistics.mean(b_gpu_list):.1f} GB")

            del full_model, btok
        except Exception as e:
            print(f"  FAILED: {e}")
            results["baseline"] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()
        print()

    # ── Expert-aware loading ──────────────────────────────────────────
    import moe_policylang

    print("=" * 60)
    print("Loading model with expert-aware device map...")
    print("=" * 60)
    model, tok = moe_policylang.load_moe_model(MODEL_ID, trust_remote_code=True)
    skeleton_gb = torch.cuda.memory_allocated() / 1e9
    results["skeleton_gpu_gb"] = skeleton_gb
    print(f"  Skeleton: {skeleton_gb:.1f} GB on GPU")
    print()

    # ── Policy sweep (multi-run) ──────────────────────────────────────
    results["policies"] = {}

    for name, dsl in POLICIES.items():
        print("=" * 60)
        print(f"POLICY: {name}  ({args.warmup}w + {args.runs}r)")
        print("=" * 60)

        mgr = moe_policylang.attach(model, dsl)

        # Warmup
        for w in range(args.warmup):
            tps, gpu, hr, xf = run_once(model, tok, mgr, args.max_tokens)
            print(f"  warmup {w+1}: {tps:.2f} tok/s")

        # Measured runs
        tps_list, gpu_list, hr_list, xf_list = [], [], [], []
        for r in range(args.runs):
            tps, gpu, hr, xf = run_once(model, tok, mgr, args.max_tokens)
            tps_list.append(tps)
            gpu_list.append(gpu)
            hr_list.append(hr)
            xf_list.append(xf)
            print(f"  run {r+1}/{args.runs}: {tps:.2f} tok/s  GPU={gpu:.1f} GB  "
                  f"hit={hr:.1%}  transfers={xf}")

        policy_result = {
            "dsl": dsl,
            "tps_values": tps_list,
            "tps_mean": statistics.mean(tps_list),
            "tps_std": statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0,
            "gpu_gb_mean": statistics.mean(gpu_list),
            "hit_rate_values": hr_list,
            "hit_rate_mean": statistics.mean(hr_list),
            "hit_rate_std": statistics.stdev(hr_list) if len(hr_list) > 1 else 0.0,
            "transfers_values": xf_list,
            "transfers_mean": statistics.mean(xf_list),
        }
        results["policies"][name] = policy_result

        print(f"  → tok/s: {fmt(tps_list)}")
        print(f"  → GPU:   {statistics.mean(gpu_list):.1f} GB")
        print(f"  → hits:  {fmt([h*100 for h in hr_list])}%")
        print()

        mgr.detach()
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std)")
    print("=" * 60)
    print(f"{'Config':<30s}  {'tok/s':<18s}  {'GPU (GB)':<10s}  {'Hit Rate':<18s}")
    print("-" * 80)

    if "baseline" in results and "tps_mean" in results.get("baseline", {}):
        b = results["baseline"]
        print(f"{'Baseline (auto)':<30s}  "
              f"{b['tps_mean']:.2f} ± {b['tps_std']:.2f}{'':8s}  "
              f"{b['gpu_gb']:<10.1f}  {'—':<18s}")

    for name, p in results["policies"].items():
        tps_str = f"{p['tps_mean']:.2f} ± {p['tps_std']:.2f}"
        hr_str = f"{p['hit_rate_mean']*100:.1f} ± {p['hit_rate_std']*100:.1f}%"
        print(f"{name:<30s}  {tps_str:<18s}  {p['gpu_gb_mean']:<10.1f}  {hr_str:<18s}")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "qwen_multirun_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
