# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Verify all cross-system .moe policies parse, compile, and run."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_policylang.parser import parse_file
from moe_policylang.compiler import compile_policy
from moe_policylang.runtime.hooks import build_hook

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "cross_system_policies",
)
TRACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "traces", "mixtral_sample.jsonl",
)

# Load trace
lines = open(TRACE_PATH).readlines()
trace = [json.loads(l) for l in lines[1:101]]  # first 100 entries for quick test
print(f"Loaded {len(trace)} trace entries for verification\n")

moe_files = sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".moe"))
print(f"Found {len(moe_files)} cross-system policies\n")

results = []
for fname in moe_files:
    path = os.path.join(EXAMPLES_DIR, fname)
    name = fname.replace(".moe", "")
    print(f"--- {name} ---")

    # Parse
    try:
        irs = parse_file(path)
        ir = irs[0]
        print(f"  PARSE: OK ({ir.name})")
    except Exception as e:
        print(f"  PARSE: FAIL — {e}")
        results.append((name, "PARSE FAIL", str(e)))
        continue

    # Compile
    try:
        compiled = compile_policy(ir)
        print(f"  COMPILE: OK")
    except Exception as e:
        print(f"  COMPILE: FAIL — {e}")
        results.append((name, "COMPILE FAIL", str(e)))
        continue

    # Run
    try:
        hook = build_hook(compiled)
        for entry in trace:
            hook.on_layer(
                layer_idx=entry["l"],
                selected_experts=entry["e"],
                scores=entry.get("s"),
            )
        stats = hook.stats_snapshot()
        h = stats["cache"]["hits"]
        m = stats["cache"]["misses"]
        hr = h / max(1, h + m)
        print(f"  RUN: OK — {h} hits, {m} misses, {hr:.1%} hit rate")
        results.append((name, "OK", f"{hr:.1%}"))
    except Exception as e:
        print(f"  RUN: FAIL — {e}")
        results.append((name, "RUN FAIL", str(e)))

    print()

# Summary
print("=" * 50)
print("SUMMARY")
print("=" * 50)
for name, status, detail in results:
    print(f"  {name:<30} {status:<15} {detail}")

all_ok = all(s == "OK" for _, s, _ in results)
print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")
