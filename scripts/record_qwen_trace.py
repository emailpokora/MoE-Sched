#!/usr/bin/env python3
"""Record a clean Qwen1.5-MoE expert routing trace.

Fixes the bug in bench_qwen_budget_sweep.py's recorder where the
WeightPlacementManager wraps the gate output in a tuple, confusing the
trace recorder. This script:

  - Loads the model via load_moe_model (skeleton on GPU, experts on CPU)
  - Attaches a nocache policy so inference is tractable (~4 tok/s)
  - Hooks gate modules with a recorder that computes topk ourselves from
    router_logits, ignoring whatever format upstream hooks return

Output: traces/qwen1.5_moe_a2.7b_trace.jsonl (overwrites previous broken trace)
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/hf_cache/hub")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
TRACE_PATH = os.path.join(ROOT, "traces", "qwen1.5_moe_a2.7b_trace.jsonl")
PROMPTS = [
    "Explain the difference between LRU and LFU caching strategies in two sentences.",
    "What is a Mixture-of-Experts model and why is it useful?",
    "Describe how GPU memory management affects large language model serving.",
    "Write a short Python function that computes factorial recursively.",
]


def attach_policy(model, dsl):
    from moe_policylang.parser import parse_policy
    from moe_policylang.compiler import compile_policy
    from moe_policylang.runtime.hooks import build_hook
    from moe_policylang.integrations.accessors import auto_accessor
    from moe_policylang.integrations.weight_placement import WeightPlacementManager
    accessor = auto_accessor(model)
    ir = parse_policy(dsl)
    compiled = compile_policy(ir)
    hook = build_hook(compiled, num_layers=accessor.num_layers, num_experts=accessor.num_experts)
    mgr = WeightPlacementManager(hook, accessor)
    mgr.attach()
    return mgr


class TraceRecorder:
    """Hook gate modules and capture top-k from router_logits.

    Robust to either:
      - Tensor output (vanilla Linear gate): out is router_logits [N, E]
      - Tuple output (wrapped by placement manager): out[0] is router_logits
    Always computes topk ourselves from the logits — never trusts upstream
    to have already done it correctly.
    """

    def __init__(self, num_layers, top_k):
        self.num_layers = num_layers
        self.top_k = top_k
        self.entries = []
        self._token_counter = 0
        self._recording = True

    def make_hook(self, layer_idx):
        top_k = self.top_k

        def hook_fn(module, inp, out):
            if not self._recording:
                return
            # Extract router_logits regardless of wrapping
            if isinstance(out, torch.Tensor):
                logits = out
            elif isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], torch.Tensor):
                logits = out[0]
            else:
                return
            if logits.dim() != 2:
                return
            # Sanity: logits shape should be [N, num_experts]
            # We compute top-k ourselves to be robust.
            weights_all = F.softmax(logits.float(), dim=-1)
            top_w, top_idx = torch.topk(weights_all, k=top_k, dim=-1)
            top_w_cpu = top_w.detach().cpu()
            top_idx_cpu = top_idx.detach().cpu()
            for tok_idx in range(top_idx_cpu.shape[0]):
                self.entries.append({
                    "t": self._token_counter + tok_idx,
                    "l": layer_idx,
                    "e": [int(e) for e in top_idx_cpu[tok_idx].tolist()],
                    "s": [round(float(s), 4) for s in top_w_cpu[tok_idx].tolist()],
                })
            if layer_idx == 0:
                self._token_counter += top_idx_cpu.shape[0]

        return hook_fn

    def stop(self):
        self._recording = False

    def save(self, path, model_name, num_layers, num_experts, top_k):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = {
            "model_name": model_name,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "total_entries": len(self.entries),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")
            for e in self.entries:
                f.write(json.dumps(e) + "\n")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"HF cache: {os.environ['HF_HOME']}")

    import moe_policylang
    t0 = time.perf_counter()
    model, tok = moe_policylang.load_moe_model(MODEL_ID, trust_remote_code=True)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s, skeleton {torch.cuda.memory_allocated()/1e9:.2f} GB")

    from moe_policylang.integrations.accessors import auto_accessor
    accessor = auto_accessor(model)
    num_layers = accessor.num_layers
    num_experts = accessor.num_experts
    top_k = getattr(model.config, "num_experts_per_tok", 4)
    print(f"Detected: {num_layers} layers, {num_experts} experts, top-{top_k}")

    # nocache policy for fast inference
    mgr = attach_policy(model,
        "policy nocache { cache { capacity = 1  eviction = lru } "
        "prefetch { budget = 1 } }")

    # Hook gates AFTER the policy manager attaches, so our hook runs after
    # the placement manager's. We compute topk ourselves so the policy's
    # output format doesn't matter.
    gates = [model.model.layers[i].mlp.gate for i in range(num_layers)
             if hasattr(model.model.layers[i].mlp, "gate")]
    print(f"Found {len(gates)} gate modules")

    recorder = TraceRecorder(num_layers, top_k)
    handles = [g.register_forward_hook(recorder.make_hook(i)) for i, g in enumerate(gates)]

    # Warmup (don't record)
    recorder._recording = False
    inp = tok("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=8, do_sample=False)
    recorder._recording = True
    recorder._token_counter = 0
    recorder.entries.clear()

    t_gen = time.perf_counter()
    for i, p in enumerate(PROMPTS):
        inp = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=128, do_sample=False)
        print(f"  Prompt {i+1}/{len(PROMPTS)}: {len(recorder.entries)} entries  "
              f"({time.perf_counter()-t_gen:.1f}s)")

    recorder.stop()
    for h in handles:
        h.remove()

    recorder.save(TRACE_PATH, MODEL_ID, num_layers, num_experts, top_k)
    print(f"Saved {len(recorder.entries)} entries to {TRACE_PATH}")

    # Sanity check: print unique experts at a couple layers
    from collections import Counter
    for check_layer in [0, 5, 12, 20]:
        eids = [eid for e in recorder.entries if e["l"] == check_layer for eid in e["e"]]
        c = Counter(eids)
        print(f"  L{check_layer}: {len(c)} unique experts; top-5 by count: "
              f"{c.most_common(5)}")


if __name__ == "__main__":
    main()
