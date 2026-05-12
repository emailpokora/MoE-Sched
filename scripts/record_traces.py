# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Record expert activation traces from a real HuggingFace MoE model.

Phase 1 entry point: Loads a real model, runs inference on prompts,
and records which experts are selected at each layer for each token.

Status: STUB — first target for Phase 1 implementation.

Usage (planned):
    python scripts/record_traces.py \
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --prompts evaluation/workloads/sharegpt_sample.json \
        --output traces/mixtral_sharegpt.jsonl \
        --max-tokens 128 \
        --max-prompts 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Record expert activation traces")
    parser.add_argument("--model", required=True, help="HuggingFace model name/path")
    parser.add_argument("--prompts", required=True, help="JSON file with prompts")
    parser.add_argument("--output", required=True, help="Output .jsonl trace path")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=100)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    # TODO (Phase 1): Implement the following:
    # 1. Load model and tokenizer
    # 2. Hook into each MoE layer to capture router outputs
    # 3. Run inference on each prompt
    # 4. Record expert selections per token per layer
    # 5. Save as .jsonl trace

    print("=" * 60)
    print("Phase 1: Record Expert Activation Traces")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Prompts:    {args.prompts}")
    print(f"  Output:     {args.output}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Max prompts: {args.max_prompts}")
    print()
    print("NOT YET IMPLEMENTED — see PROGRESS.md")
    print()
    print("Next steps:")
    print("  1. pip install torch transformers accelerate")
    print("  2. Implement trace recording hooks")
    print("  3. Run on lab GPU with real model")
    sys.exit(1)


if __name__ == "__main__":
    main()
