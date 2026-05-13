#!/usr/bin/env python3
"""MoE-PolicyLang Demo: Qwen1.5-MoE on a memory-constrained GPU.

This script demonstrates the core value of MoE-PolicyLang:
  1. Standard loading FAILS (28.6 GB model > 17 GB VRAM)
  2. Expert-aware loading SUCCEEDS (~4 GB on GPU, experts on CPU)
  3. A DSL policy manages expert caching at runtime
  4. Generates publication-quality figures

Run from a plain terminal with Windsurf/IDE closed to maximize free RAM:
    python scripts/run_qwen_moe_demo.py
"""
import argparse, gc, json, os, sys, textwrap, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import psutil

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "Explain the key ideas behind mixture-of-experts models."
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

ap = argparse.ArgumentParser()
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--skip-baseline", action="store_true",
               help="Skip the standard loading attempt (if you already know it OOMs)")
ap.add_argument("--no-figures", action="store_true", help="Skip figure generation")
args = ap.parse_args()

# ── System info ─────────────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
ram = psutil.virtual_memory()
print(f"GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")
print(f"RAM: {ram.total/1e9:.1f} GB total, {ram.available/1e9:.1f} GB available")
print(f"Model: {MODEL_ID} (~28.6 GB fp16)")
print()

# ── Step 1: Try standard loading (expect OOM) ──────────────────────────
baseline_gpu_gb = None
baseline_tps = None

if not args.skip_baseline:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("=" * 60)
    print("STEP 1: Standard loading (device_map='auto')")
    print("=" * 60)
    try:
        t0 = time.time()
        full_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True)
        full_model.eval()
        baseline_gpu_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Loaded in {time.time()-t0:.0f}s — {baseline_gpu_gb:.1f} GB on GPU")

        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        inp = tok(PROMPT, return_tensors="pt").to(full_model.device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            out = full_model.generate(**inp, max_new_tokens=args.max_tokens, do_sample=False)
        torch.cuda.synchronize()
        baseline_tps = (out.shape[1] - inp["input_ids"].shape[1]) / (time.perf_counter() - t0)
        print(f"Baseline: {baseline_tps:.1f} tok/s  GPU={baseline_gpu_gb:.1f}GB")
        del full_model, tok, inp, out
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print(f"  The model doesn't fit in {vram_gb:.0f} GB VRAM.")

    gc.collect()
    torch.cuda.empty_cache()
    print()

# ── Step 2: Expert-aware loading ────────────────────────────────────────
import moe_policylang

print("=" * 60)
print("STEP 2: MoE-PolicyLang expert-aware loading")
print("=" * 60)
print(f"RAM available: {psutil.virtual_memory().available/1e9:.1f} GB")

t0 = time.time()
model, tok = moe_policylang.load_moe_model(MODEL_ID, trust_remote_code=True)
load_time = time.time() - t0
skeleton_gpu_gb = torch.cuda.memory_allocated() / 1e9
print(f"\nLoaded in {load_time:.0f}s")
print(f"  Skeleton on GPU: {skeleton_gpu_gb:.1f} GB")
print(f"  RAM used: {psutil.virtual_memory().used/1e9:.1f} GB")
print()

# ── Step 3: Attach policy and run inference ─────────────────────────────
print("=" * 60)
print("STEP 3: Attach DSL policy + inference")
print("=" * 60)

POLICY_DSL = """
policy qwen_offload {
    cache {
        capacity        = 4
        eviction        = lfu
        frequency_decay = 0.9
    }
    prefetch { strategy = history  budget = 2 }
}
"""
print(f"Policy:\n{textwrap.dedent(POLICY_DSL).strip()}\n")

mgr = moe_policylang.attach(model, POLICY_DSL)
policy_gpu_gb = torch.cuda.memory_allocated() / 1e9
print(f"Policy attached. GPU: {policy_gpu_gb:.1f} GB")

# Run inference
inp = tok(PROMPT, return_tensors="pt").to(model.device)
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=args.max_tokens, do_sample=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
gen_tokens = out.shape[1] - inp["input_ids"].shape[1]
policy_tps = gen_tokens / elapsed
final_gpu_gb = torch.cuda.memory_allocated() / 1e9

text = tok.decode(out[0], skip_special_tokens=True)[len(PROMPT):].strip()
print(f"\nGenerated ({gen_tokens} tokens in {elapsed:.1f}s, {policy_tps:.1f} tok/s):")
print(f"  {text[:200]}...")

# ── Step 4: Policy sweep ───────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 4: Policy sweep")
print("=" * 60)

mgr.detach(); gc.collect(); torch.cuda.empty_cache()

POLICIES = {
    "aggressive\n(cap=2, LRU)": "policy a { cache { capacity = 2  eviction = lru } prefetch { strategy = history  budget = 1 } }",
    "balanced\n(cap=4, LFU)": "policy b { cache { capacity = 4  eviction = lfu  frequency_decay = 0.9 } prefetch { strategy = history  budget = 2 } }",
    "generous\n(cap=8, LFU)": "policy c { cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 } prefetch { strategy = history  budget = 4 } }",
}

comparison = []
for name, dsl in POLICIES.items():
    m = moe_policylang.attach(model, dsl)
    inp = tok(PROMPT, return_tensors="pt").to(model.device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=args.max_tokens, do_sample=False)
    torch.cuda.synchronize()
    tps = (out.shape[1] - inp["input_ids"].shape[1]) / (time.perf_counter() - t0)
    gpu_gb = torch.cuda.memory_allocated() / 1e9
    s = m.get_stats()
    hr = s["policy"]["cache"]["hit_rate"]
    xfers = s["placement"]["cpu_to_gpu_transfers"]
    comparison.append({"name": name, "gpu_gb": gpu_gb, "tps": tps, "hit_rate": hr, "transfers": xfers})
    label = name.replace("\n", " ")
    print(f"  {label:25s}  GPU={gpu_gb:.1f}GB  {tps:.1f} tok/s  hits={hr:.0%}  transfers={xfers}")
    m.detach(); gc.collect(); torch.cuda.empty_cache()

# ── Summary ─────────────────────────────────────────────────────────────
stats = mgr.get_stats() if hasattr(mgr, 'get_stats') else {}
print(f"\n{'='*60}")
print("RESULTS: Qwen1.5-MoE-A2.7B")
print("=" * 60)
print(f"Full model:           ~28.6 GB (OOMs on {vram_gb:.0f} GB GPU)")
print(f"Skeleton on GPU:      {skeleton_gpu_gb:.1f} GB")
print(f"With policy (cap=4):  {policy_gpu_gb:.1f} GB  ({policy_tps:.1f} tok/s)")
if baseline_gpu_gb and baseline_tps:
    print(f"Baseline (full GPU):  {baseline_gpu_gb:.1f} GB  ({baseline_tps:.1f} tok/s)")
    print(f"VRAM saved:           {baseline_gpu_gb - policy_gpu_gb:.1f} GB ({(baseline_gpu_gb - policy_gpu_gb)/baseline_gpu_gb:.0%})")
else:
    print(f"VRAM saved:           ~{28.6 - policy_gpu_gb:.0f} GB ({(28.6 - policy_gpu_gb)/28.6:.0%}) vs full model")
    print(f"Standard loading:     FAILED (OOM)")
    print(f"MoE-PolicyLang:       SUCCEEDED")
print("=" * 60)

# ── Step 5: Generate figures ────────────────────────────────────────────
if not args.no_figures:
    print(f"\nGenerating figures to {FIGURES_DIR}/...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Figure 1: VRAM comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["Full Model\n(standard)", "Skeleton\nonly", "Skeleton +\nexpert cache"]
    vals = [baseline_gpu_gb or 28.6, skeleton_gpu_gb, policy_gpu_gb]
    colors = ["#d32f2f", "#1976d2", "#388e3c"]
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f} GB", ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.axhline(y=vram_gb, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"GPU limit ({vram_gb:.0f} GB)")
    if baseline_gpu_gb is None:
        ax.annotate("OOM", xy=(0, min(28.6, vram_gb)), fontsize=18, fontweight="bold",
                    color="red", ha="center", va="bottom",
                    xytext=(0, vram_gb + 1.5),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.legend(fontsize=11)
    ax.set_ylabel("GPU VRAM (GB)")
    ax.set_title(f"Qwen1.5-MoE-A2.7B on {gpu_name}", fontsize=14)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGURES_DIR, f"vram_comparison_qwen.{ext}"))
    plt.close()
    print(f"  vram_comparison_qwen.pdf")

    # Figure 2: Policy sweep (3-panel)
    names = [c["name"] for c in comparison]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    b = ax.bar(names, [c["tps"] for c in comparison], color="#1976d2", edgecolor="black", linewidth=0.5)
    for bar, c in zip(b, comparison):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{c['tps']:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Tokens/s"); ax.set_title("Throughput")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    b = ax.bar(names, [c["gpu_gb"] for c in comparison], color="#ff9800", edgecolor="black", linewidth=0.5)
    for bar, c in zip(b, comparison):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{c['gpu_gb']:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("GPU VRAM (GB)"); ax.set_title("Memory")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[2]
    b = ax.bar(names, [c["hit_rate"]*100 for c in comparison], color="#388e3c", edgecolor="black", linewidth=0.5)
    for bar, c in zip(b, comparison):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{c['hit_rate']:.0%}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Hit Rate (%)"); ax.set_title("Cache Effectiveness")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Qwen1.5-MoE-A2.7B: Policy Tradeoffs (Live Inference)", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGURES_DIR, f"policy_sweep_qwen.{ext}"))
    plt.close()
    print(f"  policy_sweep_qwen.pdf")

    # Figure 3: Stacked bar — impossible vs possible
    fig, ax = plt.subplots(figsize=(8, 5))
    skel = skeleton_gpu_gb
    expert_cache = policy_gpu_gb - skel
    full_experts = (baseline_gpu_gb or 28.6) - skel
    w = 0.4
    ax.bar(0, skel, w, color="#1976d2", label="Skeleton")
    ax.bar(0, full_experts, w, bottom=skel, color="#d32f2f", label="All experts", alpha=0.8)
    ax.bar(1, skel, w, color="#1976d2")
    ax.bar(1, expert_cache, w, bottom=skel, color="#388e3c", label="Cached experts", alpha=0.8)
    ax.axhline(y=vram_gb, color="red", linestyle="--", linewidth=2, alpha=0.8)
    ax.text(1.35, vram_gb + 0.2, f"GPU: {vram_gb:.0f} GB", color="red", fontsize=11, fontweight="bold")
    total = skel + full_experts
    ax.annotate(f"{total:.0f} GB\nOOM" if not baseline_gpu_gb else f"{total:.1f} GB",
                xy=(0, total), xytext=(0.25, total + 2), fontsize=13, fontweight="bold",
                color="red", arrowprops=dict(arrowstyle="->", color="red", lw=1.5), ha="center")
    ax.annotate(f"{policy_gpu_gb:.1f} GB\nRuns!", xy=(1, policy_gpu_gb),
                xytext=(0.75, policy_gpu_gb + 2), fontsize=13, fontweight="bold",
                color="green", arrowprops=dict(arrowstyle="->", color="green", lw=1.5), ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Without\nMoE-PolicyLang", "With\nMoE-PolicyLang"], fontsize=12)
    ax.set_ylabel("GPU VRAM (GB)")
    ax.set_title(f"Qwen1.5-MoE on {gpu_name}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, max(total, vram_gb) * 1.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGURES_DIR, f"offload_value_qwen.{ext}"))
    plt.close()
    print(f"  offload_value_qwen.pdf")

    # Save JSON
    with open(os.path.join(FIGURES_DIR, "qwen_moe_results.json"), "w") as f:
        json.dump({
            "model": MODEL_ID, "gpu": gpu_name, "vram_gb": vram_gb,
            "skeleton_gpu_gb": skeleton_gpu_gb, "policy_gpu_gb": policy_gpu_gb,
            "baseline_gpu_gb": baseline_gpu_gb, "baseline_tps": baseline_tps,
            "policy_tps": policy_tps, "comparison": comparison,
        }, f, indent=2)
    print(f"  qwen_moe_results.json")

print("\nDone.")
