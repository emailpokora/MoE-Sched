# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Record expert activation traces from a real HuggingFace MoE model.

Loads a model, hooks into each MoE router to capture expert selections
and scores during autoregressive generation, and saves a .jsonl trace.

Supports OLMoE and Qwen MoE architectures (auto-detected from config).
For models that exceed VRAM, use --max-memory to enable CPU/disk offload.

Usage:
    # OLMoE on GPU (fast)
    python scripts/record_traces.py --model allenai/OLMoE-1B-7B-0924

    # Qwen MoE with CPU offload (slow but captures routing decisions)
    python scripts/record_traces.py --model Qwen/Qwen1.5-MoE-A2.7B \
        --max-memory-gpu 10 --max-tokens 32 --max-prompts 2

    # Custom prompts
    python scripts/record_traces.py --model allenai/OLMoE-1B-7B-0924 \
        --prompts evaluation/workloads/sharegpt_sample.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT)

DEFAULT_PROMPTS = [
    "Explain the difference between LRU and LFU caching strategies in two sentences.",
    "What is a Mixture-of-Experts model and why is it useful?",
    "Describe how GPU memory management affects large language model serving.",
    "Write a short Python function that computes factorial recursively.",
]


# ── Architecture detection ───────────────────────────────────────────

def detect_architecture(model):
    """Detect MoE architecture and return (gate_modules, num_layers, num_experts, top_k)."""
    config = model.config

    # OLMoE: model.model.layers[i].mlp.gate
    if hasattr(config, "num_experts") and "olmoe" in type(model).__name__.lower():
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
        top_k = getattr(config, "num_experts_per_tok", 8)
        gates = []
        for i in range(num_layers):
            gates.append(model.model.layers[i].mlp.gate)
        return gates, num_layers, num_experts, top_k, "olmoe"

    # Qwen MoE: model.model.layers[i].mlp.gate
    if hasattr(config, "num_experts") and "qwen" in type(model).__name__.lower():
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
        top_k = getattr(config, "num_experts_per_tok", 4)
        gates = []
        for i in range(num_layers):
            mlp = model.model.layers[i].mlp
            if hasattr(mlp, "gate"):
                gates.append(mlp.gate)
        return gates, num_layers, num_experts, top_k, "qwen"

    # Mixtral: model.model.layers[i].block_sparse_moe.gate
    if hasattr(config, "num_local_experts"):
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
        top_k = getattr(config, "num_experts_per_tok", 2)
        gates = []
        for i in range(num_layers):
            layer = model.model.layers[i]
            if hasattr(layer, "block_sparse_moe"):
                gates.append(layer.block_sparse_moe.gate)
        return gates, num_layers, num_experts, top_k, "mixtral"

    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


# ── Trace recording ──────────────────────────────────────────────────

class TraceRecorder:
    """Captures expert routing decisions during generation."""

    def __init__(self, num_layers, arch_name):
        self.num_layers = num_layers
        self.arch_name = arch_name
        self.traces = []  # list of {t, l, e, s}
        self._token_counter = 0
        self._layer_counter = 0
        self._recording = True

    def make_hook(self, layer_idx):
        """Create a forward hook for a gate module at layer_idx."""
        def hook_fn(module, inp, out):
            if not self._recording:
                return

            # Extract expert indices and scores from gate output
            if self.arch_name == "olmoe":
                # OLMoE gate returns (router_logits, routing_weights, expert_indices)
                if isinstance(out, tuple) and len(out) >= 3:
                    expert_ids = out[2]  # shape: [batch*seq, top_k]
                    router_weights = out[1]  # shape: [batch*seq, top_k]
                else:
                    return
            elif self.arch_name == "qwen":
                # Qwen MoE gate: returns (topk_idx, topk_weight, aux_loss) or similar
                if isinstance(out, tuple) and len(out) >= 2:
                    expert_ids = out[0]  # topk indices
                    router_weights = out[1]  # topk weights
                else:
                    return
            elif self.arch_name == "mixtral":
                # Mixtral gate is just a linear — routing happens in MoE block
                # We need to compute it from logits
                import torch
                logits = out  # shape: [batch*seq, num_experts]
                router_weights, expert_ids = torch.topk(logits, k=2, dim=-1)
                router_weights = torch.softmax(router_weights.float(), dim=-1)
            else:
                return

            # Record each token's routing
            expert_ids_cpu = expert_ids.detach().cpu()
            weights_cpu = router_weights.detach().cpu()

            for token_idx in range(expert_ids_cpu.shape[0]):
                experts = expert_ids_cpu[token_idx].tolist()
                scores = weights_cpu[token_idx].tolist()
                self.traces.append({
                    "t": self._token_counter + token_idx,
                    "l": layer_idx,
                    "e": [int(e) for e in experts],
                    "s": [round(float(s), 4) for s in scores],
                })

            # Track token counter (increment once per full layer pass)
            if layer_idx == 0:
                self._token_counter += expert_ids_cpu.shape[0]

        return hook_fn

    def stop(self):
        self._recording = False

    def save(self, path, model_name, num_layers, num_experts, top_k):
        """Save trace as .jsonl (header line + data lines)."""
        header = {
            "model_name": model_name,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "total_entries": len(self.traces),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")
            for entry in self.traces:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {len(self.traces)} trace entries to {path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import torch

    parser = argparse.ArgumentParser(description="Record expert activation traces")
    parser.add_argument("--model", required=True, help="HuggingFace model name/path")
    parser.add_argument("--prompts", default=None, help="JSON file with prompts (optional)")
    parser.add_argument("--output", default=None, help="Output .jsonl trace path")
    parser.add_argument("--max-tokens", type=int, default=64, help="Tokens per prompt")
    parser.add_argument("--max-prompts", type=int, default=4, help="Number of prompts")
    parser.add_argument("--max-memory-gpu", type=float, default=None,
                        help="Max GPU memory in GiB (enables CPU offload)")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Auto-name output ──
    if args.output is None:
        safe_name = args.model.replace("/", "_").replace("-", "_").lower()
        args.output = os.path.join(ROOT, "traces", f"{safe_name}_trace.jsonl")

    # ── Load model ──
    print("=" * 60)
    print("TRACE RECORDER")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Tokens/prompt: {args.max_tokens}, Prompts: {args.max_prompts}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if args.max_memory_gpu is not None:
        budget = f"{args.max_memory_gpu:.0f}GiB"
        load_kwargs["max_memory"] = {0: budget, "cpu": "24GiB"}
        offload_dir = os.path.join(ROOT, "offload_tmp")
        os.makedirs(offload_dir, exist_ok=True)
        load_kwargs["offload_folder"] = offload_dir
        print(f"  GPU budget: {budget} (CPU offload enabled)")

    print("\nLoading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # ── Detect architecture and attach hooks ──
    gates, num_layers, num_experts, top_k, arch = detect_architecture(model)
    print(f"Architecture: {arch} ({num_layers} layers, {num_experts} experts, top-{top_k})")
    print(f"Found {len(gates)} gate modules")

    recorder = TraceRecorder(num_layers, arch)
    handles = []
    for layer_idx, gate in enumerate(gates):
        h = gate.register_forward_hook(recorder.make_hook(layer_idx))
        handles.append(h)

    # ── Load prompts ──
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
        if isinstance(prompts[0], dict):
            prompts = [p.get("text", p.get("prompt", "")) for p in prompts]
    else:
        prompts = DEFAULT_PROMPTS

    prompts = prompts[:args.max_prompts]
    print(f"\nGenerating {args.max_tokens} tokens for {len(prompts)} prompts...")

    # ── Generate and record ──
    t_start = time.time()
    for i, prompt in enumerate(prompts):
        inp = tokenizer(prompt, return_tensors="pt").to(0)
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=args.max_tokens, do_sample=False)
        entries_so_far = len(recorder.traces)
        elapsed = time.time() - t_start
        print(f"  Prompt {i+1}/{len(prompts)}: {entries_so_far} entries ({elapsed:.1f}s)")

    recorder.stop()

    # ── Cleanup hooks ──
    for h in handles:
        h.remove()

    # ── Save ──
    recorder.save(args.output, args.model, num_layers, num_experts, top_k)
    print(f"\nTotal: {len(recorder.traces)} routing decisions recorded")


if __name__ == "__main__":
    main()
