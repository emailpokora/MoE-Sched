#!/usr/bin/env python3
"""MoE-PolicyLang DSL Demo: The Language Is the Interface.

This demo makes the DSL the central artifact:
  1. Auto-generates a DSL policy from model architecture + GPU
  2. Prints the DSL source so the user can inspect/edit it
  3. Compiles and attaches it to a live model
  4. Measures the result

The user can also provide their own .moe file instead of auto-generating.

Usage:
    python scripts/run_dsl_demo.py                      # auto-generate
    python scripts/run_dsl_demo.py --policy my.moe      # user-written
    python scripts/run_dsl_demo.py --model mixtral       # different model
"""
import argparse, os, sys, textwrap, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import moe_policylang

ALIASES = {"olmoe": "allenai/OLMoE-1B-7B-0924",
           "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
           "qwen-moe": "Qwen/Qwen1.5-MoE-A2.7B"}
QUANTIZE = {"mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Qwen/Qwen1.5-MoE-A2.7B"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="olmoe")
ap.add_argument("--policy", default=None, help="Path to a .moe policy file (omit to auto-generate)")
ap.add_argument("--max-tokens", type=int, default=32)
args = ap.parse_args()

model_id = ALIASES.get(args.model, args.model)
prompt = "Explain the key ideas behind mixture-of-experts models."

# ── Load model ──────────────────────────────────────────────────────────
print(f"Loading {model_id}...")
kw = {"device_map": "auto"}
if model_id in QUANTIZE:
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
else:
    kw["torch_dtype"] = torch.float16
model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
model.eval()
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token

# ── Baseline ────────────────────────────────────────────────────────────
inp = tok(prompt, return_tensors="pt").to(model.device)
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=args.max_tokens, do_sample=False)
torch.cuda.synchronize()
base_tps = (out.shape[1] - inp["input_ids"].shape[1]) / (time.perf_counter() - t0)
print(f"\nBaseline: {base_tps:.1f} tok/s  GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

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
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=args.max_tokens, do_sample=False)
torch.cuda.synchronize()
sched_tps = (out.shape[1] - inp["input_ids"].shape[1]) / (time.perf_counter() - t0)

s = mgr.get_stats()
c, p = s["policy"]["cache"], s["placement"]
print(f"\nMoE-PolicyLang: {sched_tps:.1f} tok/s  GPU={torch.cuda.memory_allocated()/1e9:.1f}GB"
      f"  hits={c['hit_rate']:.0%}  transfers={p['cpu_to_gpu_transfers']}"
      f"  evictions={p['gpu_to_cpu_transfers']}")
print(f"\nA: {tok.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()[:200]}")
mgr.detach()
