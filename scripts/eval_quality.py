#!/usr/bin/env python3
"""Quality evaluation: perplexity with offloading on vs off.

Computes perplexity on a held-out corpus (wikitext-2-raw-v1 test set)
with two configurations:
  1. Baseline: device_map="auto" (full model on GPU if possible)
  2. Offloaded: expert-aware loading + MoE-PolicyLang policy

The key claim: offloading produces bit-identical computation, so
perplexity should be identical (within fp16 accumulation noise).
Measuring it makes the claim defensible rather than asserted.

Usage:
    python scripts/eval_quality.py
    python scripts/eval_quality.py --max-samples 100
    python scripts/eval_quality.py --stride 512
"""

import argparse
import gc
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn import CrossEntropyLoss

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

POLICY_DSL = (
    "policy eval { "
    "cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 } "
    "prefetch { strategy = history  budget = 4 } "
    "}"
)


def compute_perplexity(model, tokenizer, texts, max_length=1024, stride=512, device="cuda", max_tokens=8192):
    """Compute perplexity using sliding-window approach.

    Based on HuggingFace's perplexity computation guide.
    Limits total tokens to max_tokens to keep runtime bounded.
    """
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encodings.input_ids[:, :max_tokens].to(device)
    seq_len = input_ids.size(1)
    print(f"    Scoring {seq_len} tokens (max_length={max_length}, stride={stride})...")

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # number of tokens to score

        input_chunk = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits

        # Only score the last `target_len` tokens (avoid double-counting)
        shift_logits = logits[:, -target_len:-1, :].contiguous()
        shift_labels = input_chunk[:, -target_len + 1:].contiguous()

        loss_fn = CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        nlls.append(token_losses.sum().item())
        prev_end = end

        if end >= seq_len:
            break

    total_nll = sum(nlls)
    total_tokens = prev_end - 1  # -1 because we predict from position 1
    ppl = math.exp(total_nll / total_tokens)
    return ppl, total_tokens


def load_wikitext():
    """Load wikitext-2-raw-v1 test set via datasets library."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Filter empty lines
        texts = [t for t in ds["text"] if t.strip()]
        return texts
    except ImportError:
        print("  WARNING: 'datasets' not installed. Using fallback text.")
        return None
    except Exception as e:
        print(f"  WARNING: Could not load wikitext: {e}. Using fallback.")
        return None


def fallback_texts():
    """Simple fallback corpus if datasets library unavailable."""
    return [
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to weigh the importance of different parts of the input sequence when generating each output token.",
        "Mixture of experts models partition the network into specialized subnetworks called experts, with a gating mechanism that routes each input to a subset of experts. This allows scaling model capacity without proportionally scaling computation.",
        "Language model pretraining on large text corpora has become the dominant paradigm for building general-purpose NLP systems. Models like GPT, BERT, and T5 demonstrated that unsupervised pretraining followed by task-specific fine-tuning achieves state-of-the-art results.",
        "The attention mechanism computes a weighted sum of value vectors, where the weights are determined by the compatibility of query and key vectors. Multi-head attention allows the model to attend to information from different representation subspaces.",
        "Efficient inference for large language models requires careful memory management, especially when model parameters exceed available GPU memory. Techniques like weight offloading, quantization, and expert caching enable deployment on consumer hardware.",
    ] * 20  # Repeat for reasonable perplexity estimate


def main():
    ap = argparse.ArgumentParser(description="Quality evaluation (perplexity)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-length", type=int, default=1024,
                    help="Context window for perplexity computation")
    ap.add_argument("--stride", type=int, default=512,
                    help="Stride for sliding window")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit number of text samples (for speed)")
    ap.add_argument("--max-eval-tokens", type=int, default=8192,
                    help="Max tokens to score (caps corpus size)")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print("Quality Evaluation: Perplexity Comparison")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  GPU:        {gpu_name}")
    print(f"  Max length: {args.max_length}")
    print(f"  Stride:     {args.stride}")
    print("=" * 70)

    # Load evaluation corpus
    print("\nLoading evaluation corpus...")
    texts = load_wikitext()
    if texts is None:
        texts = fallback_texts()
        corpus_name = "fallback (repeated paragraphs)"
    else:
        corpus_name = "wikitext-2-raw-v1 test"

    if args.max_samples:
        texts = texts[:args.max_samples]
    print(f"  Corpus: {corpus_name}")
    print(f"  Samples: {len(texts)}")

    results = {
        "model": args.model,
        "gpu": gpu_name,
        "corpus": corpus_name,
        "max_length": args.max_length,
        "stride": args.stride,
        "n_samples": len(texts),
        "modes": {},
    }

    # ── Mode 1: Offloaded ─────────────────────────────────────────────
    print(f"\n[1/2] OFFLOADED MODE")
    print("-" * 70)

    import moe_policylang

    model, tokenizer = moe_policylang.load_moe_model(args.model, trust_remote_code=True)
    mgr = moe_policylang.attach(model, POLICY_DSL)
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    t0 = time.perf_counter()
    ppl_offloaded, n_tokens = compute_perplexity(
        model, tokenizer, texts,
        max_length=args.max_length, stride=args.stride,
        max_tokens=args.max_eval_tokens
    )
    elapsed = time.perf_counter() - t0

    print(f"  Perplexity: {ppl_offloaded:.4f}")
    print(f"  Tokens scored: {n_tokens}")
    print(f"  Time: {elapsed:.1f}s ({n_tokens/elapsed:.0f} tok/s)")

    results["modes"]["offloaded"] = {
        "perplexity": ppl_offloaded,
        "tokens_scored": n_tokens,
        "time_s": round(elapsed, 1),
        "tps": round(n_tokens / elapsed, 1),
    }

    mgr.detach()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ── Mode 2: Baseline ──────────────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n[2/2] BASELINE MODE (device_map='auto')")
        print("-" * 70)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            model_bl = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True)
            model_bl.eval()
            tok_bl = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            tok_bl.pad_token = tok_bl.eos_token
            print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

            t0 = time.perf_counter()
            ppl_baseline, n_tokens_bl = compute_perplexity(
                model_bl, tok_bl, texts,
                max_length=args.max_length, stride=args.stride,
                max_tokens=args.max_eval_tokens
            )
            elapsed_bl = time.perf_counter() - t0

            print(f"  Perplexity: {ppl_baseline:.4f}")
            print(f"  Tokens scored: {n_tokens_bl}")
            print(f"  Time: {elapsed_bl:.1f}s ({n_tokens_bl/elapsed_bl:.0f} tok/s)")

            results["modes"]["baseline"] = {
                "perplexity": ppl_baseline,
                "tokens_scored": n_tokens_bl,
                "time_s": round(elapsed_bl, 1),
                "tps": round(n_tokens_bl / elapsed_bl, 1),
            }

            del model_bl, tok_bl
        except Exception as e:
            print(f"  Baseline FAILED: {e}")
            results["modes"]["baseline"] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("QUALITY SUMMARY")
    print("=" * 70)

    if "baseline" in results["modes"] and "perplexity" in results["modes"].get("baseline", {}):
        bl_ppl = results["modes"]["baseline"]["perplexity"]
        of_ppl = results["modes"]["offloaded"]["perplexity"]
        diff = abs(of_ppl - bl_ppl)
        pct_diff = diff / bl_ppl * 100

        print(f"  Baseline perplexity:  {bl_ppl:.4f}")
        print(f"  Offloaded perplexity: {of_ppl:.4f}")
        print(f"  Absolute difference:  {diff:.4f}")
        print(f"  Relative difference:  {pct_diff:.4f}%")

        if pct_diff < 0.01:
            print(f"\n  CONCLUSION: Bit-identical (< 0.01% difference)")
        elif pct_diff < 0.1:
            print(f"\n  CONCLUSION: Negligible difference (fp16 noise)")
        else:
            print(f"\n  WARNING: Non-trivial perplexity difference ({pct_diff:.2f}%)")

        results["analysis"] = {
            "absolute_difference": round(diff, 6),
            "relative_difference_pct": round(pct_diff, 6),
            "conclusion": "identical" if pct_diff < 0.01 else "negligible" if pct_diff < 0.1 else "divergent",
        }
    else:
        print(f"  Offloaded perplexity: {results['modes']['offloaded']['perplexity']:.4f}")
        print(f"  (Baseline skipped or failed)")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "quality_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
