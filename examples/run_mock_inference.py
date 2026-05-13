"""End-to-end demo: parse a .moe policy, compile it, install a hook, and
drive it with the mock MoE model.

Run from the week3/ directory:

    python examples/run_mock_inference.py examples/lfu_policy.moe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_policylang import build_hook, compile_policy, parse_file
from moe_policylang.integrations.mock_moe import MockMoEModel, skewed_selector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy_file", type=Path, help="Path to a .moe policy file")
    ap.add_argument("--num-tokens", type=int, default=100)
    ap.add_argument("--num-layers", type=int, default=24)
    ap.add_argument("--num-experts", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    [ir] = parse_file(args.policy_file)
    print(f"Loaded policy: {ir.name}")

    compiled = compile_policy(ir)
    hook = build_hook(compiled)

    model = MockMoEModel(
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        selector=skewed_selector(
            num_experts=args.num_experts,
            top_k=args.top_k,
            seed=args.seed,
        ),
    )
    model.run(hook, num_tokens=args.num_tokens)

    print(json.dumps(hook.stats_snapshot(), indent=2))


if __name__ == "__main__":
    main()
