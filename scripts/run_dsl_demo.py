#!/usr/bin/env python3
"""MoE-PolicyLang DSL Demo: The Language Is the Interface.

This demo makes the DSL the central artifact:
  1. Loads any MoE model with expert-aware placement (experts on CPU)
  2. Auto-generates a DSL policy from model architecture + GPU
  3. Prints the DSL source so the user can inspect/edit it
  4. Compiles and attaches it to a live model
  5. Measures the result vs. a baseline with all experts on GPU

The user can also provide their own .moe file instead of auto-generating.

Usage:
    python scripts/run_dsl_demo.py                      # auto-generate (OLMoE)
    python scripts/run_dsl_demo.py --policy my.moe      # user-written policy
    python scripts/run_dsl_demo.py --model mixtral       # Mixtral (needs ~4GB GPU)
    python scripts/run_dsl_demo.py --baseline            # also run full-GPU baseline
"""
import argparse, os, sys, textwrap, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moe_policylang

ALIASES = {"olmoe": "allenai/OLMoE-1B-7B-0924",
           "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
           "qwen-moe": "Qwen/Qwen1.5-MoE-A2.7B"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="olmoe")
ap.add_argument("--policy", default=None, help="Path to a .moe policy file (omit to auto-generate)")
ap.add_argument("--max-tokens", type=int, default=32)
ap.add_argument("--baseline", action="store_true",
               help="Also load the full model on GPU for a baseline comparison. "
                    "Only works if the model fits entirely in VRAM.")
args = ap.parse_args()

model_id = ALIASES.get(args.model, args.model)
prompt = "Explain the key ideas behind mixture-of-experts models."

# ── Load model (expert-aware: skeleton on GPU, experts on CPU) ─────────
model, tok = moe_policylang.load_moe_model(model_id)
skeleton_gpu_gb = torch.cuda.memory_allocated() / 1e9

# ── Optional baseline (full model on GPU) ──────────────────────────────
base_tps = None
baseline_gpu_gb = None
if args.baseline:
    print("\nLoading full model on GPU for baseline comparison...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    full_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16)
    full_model.eval()
    full_tok = AutoTokenizer.from_pretrained(model_id)
    full_tok.pad_token = full_tok.eos_token
    baseline_gpu_gb = torch.cuda.memory_allocated() / 1e9
    inp_full = full_tok(prompt, return_tensors="pt").to(full_model.device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = full_model.generate(**inp_full, max_new_tokens=args.max_tokens, do_sample=False)
    torch.cuda.synchronize()
    base_tps = (out.shape[1] - inp_full["input_ids"].shape[1]) / (time.perf_counter() - t0)
    print(f"Baseline (all experts on GPU): {base_tps:.1f} tok/s  GPU={baseline_gpu_gb:.1f}GB")
    del full_model, full_tok, inp_full
    torch.cuda.empty_cache()
    # Reload the expert-aware model
    model, tok = moe_policylang.load_moe_model(model_id)
    skeleton_gpu_gb = torch.cuda.memory_allocated() / 1e9

# ── Step 1: Get the DSL policy ──────────────────────────────────────────
if args.policy:
    dsl_source = open(args.policy).read()
    print(f"\nLoaded policy from {args.policy}:")
else:
    policies = moe_policylang.auto_policies(model)
    dsl_source = policies["balanced"]
    print(f"\nAuto-generated DSL policy (edit and re-run with --policy):")

print(textwrap.dedent(dsl_source).strip())

# ── Step 2: Attach — DSL string goes straight to the model ──────────────
mgr = moe_policylang.attach(model, dsl_source)

# ── Measure ─────────────────────────────────────────────────────────────
inp = tok(prompt, return_tensors="pt").to(model.device)
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=args.max_tokens, do_sample=False)
torch.cuda.synchronize()
sched_tps = (out.shape[1] - inp["input_ids"].shape[1]) / (time.perf_counter() - t0)

s = mgr.get_stats()
c, p = s["policy"]["cache"], s["placement"]
policy_gpu_gb = torch.cuda.memory_allocated() / 1e9
print(f"\nMoE-PolicyLang: {sched_tps:.1f} tok/s  GPU={policy_gpu_gb:.1f}GB"
      f"  hits={c['hit_rate']:.0%}  transfers={p['cpu_to_gpu_transfers']}"
      f"  evictions={p['gpu_to_cpu_transfers']}")

# ── Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"GPU skeleton: {skeleton_gpu_gb:.1f} GB (attention, embeddings, norms)")
print(f"GPU with policy active: {policy_gpu_gb:.1f} GB (skeleton + cached experts)")
if baseline_gpu_gb and base_tps:
    saved_gb = baseline_gpu_gb - policy_gpu_gb
    print(f"VRAM savings vs full model: {saved_gb:.1f} GB freed "
          f"({saved_gb/baseline_gpu_gb:.0%} reduction)")
    print(f"Throughput: {sched_tps:.1f} vs {base_tps:.1f} tok/s "
          f"({sched_tps/base_tps:.0%} of baseline)")
else:
    print(f"Throughput: {sched_tps:.1f} tok/s")
    print(f"(Run with --baseline to compare against full-GPU loading)")
print(f"{'='*60}")

print(f"\nGenerated: {tok.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()[:200]}")
mgr.detach()
