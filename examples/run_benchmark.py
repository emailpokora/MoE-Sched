"""Full evaluation suite — runs all policies across all workloads.

Run from the week5/ directory:

    $env:PYTHONPATH = "."
    python examples/run_benchmark.py --output results/

Produces:
  * results/benchmark_results.json  — raw metrics
  * results/tables.txt              — text tables
  * results/*.png                   — charts (if matplotlib available)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_policylang.benchmark.runner import run_all, format_table, results_to_dict
from moe_policylang.benchmark.expressiveness import (
    analyse_dsl_api_policies,
    analyse_moe_files,
    format_expressiveness_table,
)
from moe_policylang.benchmark.visualize import (
    hit_rate_table,
    throughput_table,
    latency_table,
    overhead_table,
    generate_all_charts,
)


def main():
    ap = argparse.ArgumentParser(description="MoE-PolicyLang full evaluation suite")
    ap.add_argument("--output", "-o", type=Path, default=Path("results"))
    ap.add_argument("--capacity", type=int, default=16)
    args = ap.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    # ---- Run benchmarks ----
    print("=" * 70)
    print("MoE-PolicyLang Evaluation Suite — Week 5")
    print("=" * 70)
    print()

    results = run_all(capacity=args.capacity)

    # ---- Save raw results ----
    with open(out / "benchmark_results.json", "w") as f:
        json.dump(results_to_dict(results), f, indent=2)
    print(f"\nRaw results: {out / 'benchmark_results.json'}")

    # ---- Text tables ----
    tables = []
    tables.append("=" * 70)
    tables.append("FULL RESULTS TABLE")
    tables.append("=" * 70)
    tables.append(format_table(results))

    tables.append("")
    tables.append("=" * 70)
    tables.append("HIT RATE MATRIX")
    tables.append("=" * 70)
    tables.append(hit_rate_table(results))

    tables.append("")
    tables.append("=" * 70)
    tables.append("THROUGHPUT MATRIX (tokens/second)")
    tables.append("=" * 70)
    tables.append(throughput_table(results))

    tables.append("")
    tables.append("=" * 70)
    tables.append("LATENCY MATRIX (mean µs per token)")
    tables.append("=" * 70)
    tables.append(latency_table(results))

    tables.append("")
    tables.append("=" * 70)
    tables.append("DISPATCH OVERHEAD MATRIX (%)")
    tables.append("=" * 70)
    tables.append(overhead_table(results))

    # ---- Expressiveness ----
    tables.append("")
    tables.append("=" * 70)
    tables.append("DSL EXPRESSIVENESS ANALYSIS")
    tables.append("=" * 70)

    api_entries = analyse_dsl_api_policies()
    tables.append("\nProgrammatic API policies:")
    tables.append(format_expressiveness_table(api_entries))

    examples_dir = Path("examples")
    if examples_dir.exists():
        moe_entries = analyse_moe_files(examples_dir)
        if moe_entries:
            tables.append("\n.moe file policies:")
            tables.append(format_expressiveness_table(moe_entries))

    table_text = "\n".join(tables)

    with open(out / "tables.txt", "w", encoding="utf-8") as f:
        f.write(table_text)

    print()
    print(table_text)

    # ---- Charts ----
    chart_paths = generate_all_charts(results, out)
    if chart_paths:
        print(f"\nCharts written: {[str(p) for p in chart_paths]}")
    else:
        print("\n(matplotlib not available — charts skipped)")

    print(f"\nAll outputs in: {out}/")


if __name__ == "__main__":
    main()
