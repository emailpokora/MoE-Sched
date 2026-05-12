#!/usr/bin/env python3
# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Live MoE inference experiment on consumer GPU.

Runs OLMoE-1B-7B (7B params, 64 experts, top-8) on an RTX 5080 Laptop GPU
with MoE-Sched hooks attached.  Compares throughput and cache statistics
across different caching policies during real autoregressive generation.

Outputs:
  traces/constrained_e2e_results.json   — raw metrics
  paper/figures/constrained_throughput.pdf — bar chart for paper
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── paths ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "traces")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── config ───────────────────────────────────────────────────────────
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
NUM_LAYERS = 16
NUM_EXPERTS = 64
TOP_K = 8
MAX_NEW_TOKENS = 64
NUM_WARMUP = 1

PROMPTS = [
    "Explain the difference between LRU and LFU caching strategies in two sentences.",
    "What is a Mixture-of-Experts model and why is it useful?",
    "Describe how GPU memory management affects large language model serving.",
    "Write a short Python function that computes factorial recursively.",
]


# ── helpers ──────────────────────────────────────────────────────────

def gpu_mem_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark(model, tokenizer, prompts, max_new_tokens, warmup=1, label=""):
    """Generate from prompts and return throughput metrics."""
    device = next(model.parameters()).device
    results = []

    # warmup
    for i in range(min(warmup, len(prompts))):
        inp = tokenizer(prompts[i], return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    reset_gpu()

    for prompt in prompts:
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inp["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_tokens = out.shape[1] - input_len
        elapsed = t1 - t0
        results.append({
            "prompt_len": input_len,
            "gen_tokens": int(gen_tokens),
            "elapsed_s": round(elapsed, 4),
            "tokens_per_sec": round(gen_tokens / elapsed, 2) if elapsed > 0 else 0,
        })

    peak_mem = gpu_mem_gb()
    total_tok = sum(r["gen_tokens"] for r in results)
    total_t = sum(r["elapsed_s"] for r in results)
    avg_tps = total_tok / total_t if total_t > 0 else 0

    summary = {
        "label": label,
        "total_tokens": total_tok,
        "total_time_s": round(total_t, 4),
        "avg_tokens_per_sec": round(avg_tps, 2),
        "peak_gpu_mem_gb": round(peak_mem, 2),
        "per_prompt": results,
    }
    print(f"  [{label}] {avg_tps:.1f} tok/s | peak {peak_mem:.1f} GB")
    return summary


# ── MoE-Sched hook wiring ───────────────────────────────────────────

def attach_hooks(model, hook_obj):
    """Attach MoE-Sched PolicyHook to OLMoE router layers."""
    handles = []
    dispatch_times: List[float] = []

    for layer_idx in range(NUM_LAYERS):
        gate_module = model.model.layers[layer_idx].mlp.gate

        def make_hook(li):
            def fwd_hook(mod, inp, out):
                # OLMoE gate returns (logits[T,64], weights[T,8], indices[T,8])
                if isinstance(out, tuple) and len(out) >= 3:
                    expert_ids = out[2]  # [tokens, top_k]
                    selected = expert_ids.cpu().tolist()
                else:
                    return out

                t0 = time.perf_counter()
                for token_experts in selected:
                    hook_obj.on_layer(li, [int(e) for e in token_experts])
                t1 = time.perf_counter()
                dispatch_times.append((t1 - t0) * 1e6)
                return out
            return fwd_hook

        h = gate_module.register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    print(f"  Attached hooks to {len(handles)} MoE router layers")
    return handles, dispatch_times


# ── policy configs ───────────────────────────────────────────────────

def get_policies():
    """Return dict of policy_name → (compiled_policy, use_per_layer)."""
    from moe_sched.ir import (
        PolicyIR, CacheIR, PrefetchIR, ScheduleIR,
        EvictionPolicy, PrefetchStrategy, ScheduleMode,
    )
    from moe_sched.compiler import compile_policy

    policies = {}

    # 1. Naive — tiny cache (simulates almost no caching)
    policies["naive_c4"] = (compile_policy(PolicyIR(
        name="naive_c4",
        cache=CacheIR(capacity=4, eviction=EvictionPolicy.LRU),
        schedule=ScheduleIR(mode=ScheduleMode.GPU_ONLY),
    )), False)

    # 2. LRU cap=16
    policies["lru_c16"] = (compile_policy(PolicyIR(
        name="lru_c16",
        cache=CacheIR(capacity=16, eviction=EvictionPolicy.LRU),
        schedule=ScheduleIR(mode=ScheduleMode.GPU_ONLY),
    )), False)

    # 3. LFU + history prefetch cap=16
    policies["lfu_hist_c16"] = (compile_policy(PolicyIR(
        name="lfu_hist_c16",
        cache=CacheIR(capacity=16, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
        prefetch=PrefetchIR(strategy=PrefetchStrategy.HISTORY, budget=4),
        schedule=ScheduleIR(mode=ScheduleMode.HYBRID),
    )), False)

    # 4. EPCB — entropy-proportional cache budgeting
    # PerLayerHook takes raw PolicyIR (it compiles internally)
    policies["epcb_c16"] = (PolicyIR(
        name="epcb_c16",
        cache=CacheIR(capacity=16, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
        prefetch=PrefetchIR(strategy=PrefetchStrategy.HISTORY, budget=4),
        schedule=ScheduleIR(mode=ScheduleMode.HYBRID),
    ), True)  # per-layer enabled

    return policies


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LIVE MoE INFERENCE EXPERIMENT")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {MODEL_ID}")
    print(f"  {NUM_LAYERS} layers × {NUM_EXPERTS} experts, top-{TOP_K}")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (fp16, full GPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    model.eval()

    gpu_used = torch.cuda.memory_allocated() / 1e9
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU memory: {gpu_used:.1f} / {gpu_total:.1f} GB")

    reset_gpu()

    # ── Baseline: vanilla (no hooks) ──
    print("\n" + "=" * 60)
    print("1. BASELINE — Vanilla HuggingFace (no MoE-Sched)")
    print("=" * 60)
    vanilla = benchmark(model, tokenizer, PROMPTS, MAX_NEW_TOKENS,
                        warmup=NUM_WARMUP, label="vanilla")

    # ── Policy experiments ──
    from moe_sched.runtime.hooks import build_hook
    from moe_sched.runtime.per_layer import PerLayerHook, PerLayerConfig

    policies = get_policies()
    all_results = {"vanilla": vanilla}

    for pname, (compiled, use_per_layer) in policies.items():
        print(f"\n{'=' * 60}")
        print(f"POLICY: {pname}" + (" [EPCB]" if use_per_layer else ""))
        print("=" * 60)

        if use_per_layer:
            config = PerLayerConfig(
                entropy_window=200,
                min_capacity=4,
                max_capacity=32,
                rebalance_interval=100,
            )
            hook = PerLayerHook(
                base_ir=compiled,
                num_layers=NUM_LAYERS,
                num_experts=NUM_EXPERTS,
                config=config,
            )
        else:
            hook = build_hook(compiled)

        handles, dispatch_times = attach_hooks(model, hook)

        result = benchmark(model, tokenizer, PROMPTS, MAX_NEW_TOKENS,
                           warmup=NUM_WARMUP, label=pname)

        # Collect cache stats
        stats = hook.stats_snapshot()
        cache_stats = stats.get("cache", stats)
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        hit_rate = hits / max(1, hits + misses)

        result["cache_hits"] = hits
        result["cache_misses"] = misses
        result["hit_rate"] = round(hit_rate, 4)
        result["total_dispatches"] = len(dispatch_times)

        if dispatch_times:
            dt = np.array(dispatch_times)
            result["dispatch_us_mean"] = round(float(np.mean(dt)), 2)
            result["dispatch_us_p99"] = round(float(np.percentile(dt, 99)), 2)

        print(f"  Cache: {hits:,} hits / {misses:,} misses ({hit_rate:.1%})")
        if dispatch_times:
            print(f"  Dispatch: {np.mean(dispatch_times):.1f} us/call avg")

        # Cleanup
        for h in handles:
            h.remove()
        del hook
        reset_gpu()

        all_results[pname] = result

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Policy':<16} {'tok/s':>8} {'Hit Rate':>10} {'Peak GB':>8}")
    print("-" * 50)
    for name, r in all_results.items():
        hr = f"{r.get('hit_rate', 0):.1%}" if "hit_rate" in r else "N/A"
        print(f"{name:<16} {r['avg_tokens_per_sec']:>8.1f} {hr:>10} "
              f"{r['peak_gpu_mem_gb']:>8.1f}")

    # ── Save results ──
    output_path = os.path.join(RESULTS_DIR, "constrained_e2e_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(gpu_total, 1),
            "model": MODEL_ID,
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_prompts": len(PROMPTS),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ── Generate figure ──
    fig_constrained_throughput(all_results)


def fig_constrained_throughput(results):
    """Two-panel bar chart: tok/s and hit rate for each policy."""
    names = list(results.keys())
    tps = [results[n]["avg_tokens_per_sec"] for n in names]
    hit_rates = [results[n].get("hit_rate", None) for n in names]

    colors = ["#888888", "#E57373", "#4C72B0", "#55A868", "#8172B2"]

    has_hr = any(hr is not None for hr in hit_rates)

    if has_hr:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.8), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(3.5, 2.8))

    # Top: tokens/sec
    bars = ax1.bar(range(len(names)), tps,
                   color=colors[:len(names)], edgecolor="white", alpha=0.9)
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{tps[i]:.1f}", ha="center", va="bottom", fontsize=6)
    ax1.set_ylabel("Tokens / sec", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    if has_hr:
        hr_vals = [hr * 100 if hr is not None else 0 for hr in hit_rates]
        bars2 = ax2.bar(range(len(names)), hr_vals,
                        color=colors[:len(names)], edgecolor="white", alpha=0.7)
        for i, bar in enumerate(bars2):
            if hit_rates[i] is not None:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{hit_rates[i]:.0%}", ha="center", va="bottom", fontsize=6)
        ax2.set_ylabel("Hit rate (%)", fontsize=8)
        ax2.set_ylim(0, 105)
        ax2.tick_params(labelsize=7)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        target_ax = ax2
    else:
        target_ax = ax1

    target_ax.set_xticks(range(len(names)))
    target_ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "constrained_throughput.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
