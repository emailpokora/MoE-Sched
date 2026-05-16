#!/usr/bin/env python3
"""Async vs Sync expert transfer benchmark.

Compares throughput, transfer time, and overlap ratio between synchronous
and asynchronous expert weight transfers on a real MoE model.

The benchmark runs the same policy + prompt with:
  1. Sync mode (default): transfers block on each CPU→GPU copy
  2. Async mode: prefetched experts transferred on a dedicated CUDA stream
     while the current layer's expert forward runs on the compute stream

Key metrics:
  - tok/s throughput improvement
  - transfer_time_s reduction
  - overlap_ratio (fraction of transfers that completed before needed)
  - prefetch_hits vs sync_waits

Usage:
    python scripts/bench_async_transfer.py
    python scripts/bench_async_transfer.py --runs 5
    python scripts/bench_async_transfer.py --model Qwen/Qwen1.5-MoE-A2.7B
    python scripts/bench_async_transfer.py --max-tokens 128
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

# Policies with prefetching (required for async to have effect)
# Higher prefetch budget = more experts transferred ahead of time
POLICIES = {
    "budget=4": (
        "policy ab4 { "
        "cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 4 } "
        "}"
    ),
    "budget=8": (
        "policy ab8 { "
        "cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 8 } "
        "}"
    ),
    "budget=16": (
        "policy ab16 { "
        "cache { capacity = 12  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 16 } "
        "}"
    ),
}


def fmt(vals):
    """Format a list of floats as mean ± std."""
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    return f"{m:.2f} ± {s:.2f}"


def reset_stats(mgr):
    """Reset all stats for a clean measurement."""
    if hasattr(mgr, 'hook'):
        mgr.hook.cache.stats.hits = 0
        mgr.hook.cache.stats.misses = 0
        mgr.hook.cache.stats.evictions = 0
    if hasattr(mgr, 'stats'):
        mgr.stats.cpu_to_gpu_transfers = 0
        mgr.stats.gpu_to_cpu_transfers = 0
        mgr.stats.bytes_transferred = 0
        mgr.stats.transfer_time_s = 0.0
        mgr.stats.forward_calls = 0
    if mgr._atm is not None:
        from moe_policylang.integrations.async_transfer import AsyncTransferStats
        mgr._atm.stats = AsyncTransferStats()
        mgr._atm.clear()


def run_once(model, tok, mgr, max_tokens):
    """Single inference run.  Returns dict of metrics."""
    reset_stats(mgr)

    inp = tok(PROMPT, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_tokens = out.shape[1] - inp["input_ids"].shape[1]
    tps = gen_tokens / elapsed
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    stats = mgr.get_stats()

    result = {
        "tok_s": tps,
        "elapsed_s": elapsed,
        "gen_tokens": gen_tokens,
        "gpu_peak_gb": gpu_gb,
        "hit_rate": stats["policy"]["cache"]["hit_rate"],
        "cpu_to_gpu_transfers": stats["placement"]["cpu_to_gpu_transfers"],
        "transfer_time_s": stats["placement"]["transfer_time_s"],
    }
    if "async" in stats:
        result["async"] = stats["async"]
    return result


def run_mode(model, tok, mgr, args, mode_name):
    """Run warmup + measured runs for one mode. Returns list of run dicts."""
    for w in range(args.warmup):
        r = run_once(model, tok, mgr, args.max_tokens)
        print(f"  warmup {w+1}: {r['tok_s']:.2f} tok/s")

    runs = []
    for i in range(args.runs):
        r = run_once(model, tok, mgr, args.max_tokens)
        runs.append(r)
        async_info = r.get("async", {})
        line = (f"  run {i+1}/{args.runs}: {r['tok_s']:.2f} tok/s  "
                f"xfer_t={r['transfer_time_s']:.3f}s  "
                f"transfers={r['cpu_to_gpu_transfers']}  "
                f"hit={r['hit_rate']:.1%}")
        if async_info:
            line += (f"  overlap={async_info.get('overlap_ratio', 0):.1%}"
                     f"  prefetch_hits={async_info.get('prefetch_hits', 0)}"
                     f"  sync_waits={async_info.get('sync_waits', 0)}")
        print(line)
    return runs


def analyze_runs(sync_runs, async_runs):
    """Compute comparison metrics between sync and async runs."""
    sync_tps = [r["tok_s"] for r in sync_runs]
    async_tps = [r["tok_s"] for r in async_runs]
    sync_xfer_t = [r["transfer_time_s"] for r in sync_runs]
    async_xfer_t = [r["transfer_time_s"] for r in async_runs]
    sync_transfers = [r["cpu_to_gpu_transfers"] for r in sync_runs]
    async_transfers_count = [r["cpu_to_gpu_transfers"] for r in async_runs]
    async_overlap = [r.get("async", {}).get("overlap_ratio", 0) for r in async_runs]
    async_prefetch_hits = [r.get("async", {}).get("prefetch_hits", 0) for r in async_runs]
    async_sync_waits = [r.get("async", {}).get("sync_waits", 0) for r in async_runs]

    speedup = statistics.mean(async_tps) / statistics.mean(sync_tps) if statistics.mean(sync_tps) > 0 else 0
    xfer_reduction = 1.0 - (statistics.mean(async_xfer_t) / statistics.mean(sync_xfer_t)) if statistics.mean(sync_xfer_t) > 0 else 0

    return {
        "sync_tps": sync_tps,
        "async_tps": async_tps,
        "sync_xfer_t": sync_xfer_t,
        "async_xfer_t": async_xfer_t,
        "sync_transfers": sync_transfers,
        "async_transfers": async_transfers_count,
        "async_overlap": async_overlap,
        "async_prefetch_hits": async_prefetch_hits,
        "async_sync_waits": async_sync_waits,
        "speedup": speedup,
        "xfer_reduction": xfer_reduction,
    }


def main():
    ap = argparse.ArgumentParser(description="Async vs Sync transfer benchmark")
    ap.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")
    ap.add_argument("--runs", "-n", type=int, default=3, help="Measured runs per mode")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    ap.add_argument("--max-tokens", type=int, default=64, help="Tokens to generate")
    ap.add_argument("--budget", type=str, default=None,
                    help="Run only one prefetch budget (e.g. 'budget=8')")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for async transfer benchmark.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    ram = psutil.virtual_memory()
    print("=" * 70)
    print("ASYNC vs SYNC Expert Transfer Benchmark")
    print("=" * 70)
    print(f"  Model:     {args.model}")
    print(f"  GPU:       {gpu_name} ({vram_gb:.1f} GB)")
    print(f"  RAM:       {ram.total/1e9:.1f} GB total, {ram.available/1e9:.1f} GB free")
    print(f"  Tokens:    {args.max_tokens}")
    print(f"  Runs:      {args.warmup}w + {args.runs}r per mode")
    print("=" * 70)

    import moe_policylang

    # ── Load model ONCE ───────────────────────────────────────────────
    print("\nLoading model with expert-aware device map...")
    model, tok = moe_policylang.load_moe_model(args.model, trust_remote_code=True)
    skeleton_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Skeleton on GPU: {skeleton_gb:.1f} GB\n")

    # Select which policies to test
    if args.budget:
        policies_to_test = {args.budget: POLICIES[args.budget]}
    else:
        policies_to_test = POLICIES

    results = {
        "model": args.model,
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "skeleton_gpu_gb": skeleton_gb,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "warmup": args.warmup,
        "experiments": {},
    }

    # ── Run each prefetch budget with sync then async ─────────────────
    for policy_name, policy_dsl in policies_to_test.items():
        print(f"\n{'=' * 70}")
        print(f"POLICY: {policy_name}")
        print(f"  {policy_dsl}")
        print("=" * 70)

        # -- Sync --
        print(f"\n  --- SYNC ---")
        mgr = moe_policylang.attach(model, policy_dsl, async_transfers=False)
        sync_runs = run_mode(model, tok, mgr, args, "sync")
        mgr.detach()
        gc.collect()
        torch.cuda.empty_cache()

        # -- Async (same model, no reload) --
        print(f"\n  --- ASYNC ---")
        mgr = moe_policylang.attach(model, policy_dsl, async_transfers=True)
        async_runs = run_mode(model, tok, mgr, args, "async")
        mgr.detach()
        gc.collect()
        torch.cuda.empty_cache()

        # -- Analyze --
        analysis = analyze_runs(sync_runs, async_runs)
        results["experiments"][policy_name] = {
            "policy": policy_dsl,
            "sync": {
                "tps_mean": statistics.mean(analysis["sync_tps"]),
                "tps_std": statistics.stdev(analysis["sync_tps"]) if len(analysis["sync_tps"]) > 1 else 0.0,
                "transfer_time_mean": statistics.mean(analysis["sync_xfer_t"]),
                "transfers_mean": statistics.mean(analysis["sync_transfers"]),
            },
            "async": {
                "tps_mean": statistics.mean(analysis["async_tps"]),
                "tps_std": statistics.stdev(analysis["async_tps"]) if len(analysis["async_tps"]) > 1 else 0.0,
                "transfer_time_mean": statistics.mean(analysis["async_xfer_t"]),
                "transfers_mean": statistics.mean(analysis["async_transfers"]),
                "overlap_ratio_mean": statistics.mean(analysis["async_overlap"]),
                "prefetch_hits_mean": statistics.mean(analysis["async_prefetch_hits"]),
                "sync_waits_mean": statistics.mean(analysis["async_sync_waits"]),
            },
            "speedup": round(analysis["speedup"], 3),
            "transfer_time_reduction_pct": round(analysis["xfer_reduction"] * 100, 1),
        }

        # Print per-policy summary
        print(f"\n  {'Metric':<25s}  {'Sync':<14s}  {'Async':<14s}  {'Delta':<10s}")
        print(f"  {'-'*65}")
        print(f"  {'tok/s':<25s}  {fmt(analysis['sync_tps']):<14s}  {fmt(analysis['async_tps']):<14s}  {analysis['speedup']:.2f}x")
        print(f"  {'xfer time (s)':<25s}  {statistics.mean(analysis['sync_xfer_t']):<14.3f}  {statistics.mean(analysis['async_xfer_t']):<14.3f}  -{analysis['xfer_reduction']*100:.1f}%")
        print(f"  {'overlap ratio':<25s}  {'—':<14s}  {statistics.mean(analysis['async_overlap']):<14.1%}")
        print(f"  {'prefetch hits':<25s}  {'—':<14s}  {statistics.mean(analysis['async_prefetch_hits']):<14.0f}")

    # ── Final Summary ─────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<15s}  {'Sync tok/s':<14s}  {'Async tok/s':<14s}  {'Speedup':<10s}  {'Overlap':<10s}")
    print("-" * 70)
    for name, exp in results["experiments"].items():
        s_tps = f"{exp['sync']['tps_mean']:.2f}"
        a_tps = f"{exp['async']['tps_mean']:.2f}"
        spd = f"{exp['speedup']:.2f}x"
        ovl = f"{exp['async']['overlap_ratio_mean']:.1%}"
        print(f"  {name:<13s}  {s_tps:<14s}  {a_tps:<14s}  {spd:<10s}  {ovl:<10s}")
    print("=" * 70)

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "async_transfer_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
