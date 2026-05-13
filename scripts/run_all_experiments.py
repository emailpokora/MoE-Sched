# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Orchestrate the full evaluation suite (Phase 4).

Runs all experiment configs, all baselines, and generates comparison tables.

Status: STUB — implement after Phases 1-3.

Usage (planned):
    python scripts/run_all_experiments.py --configs evaluation/configs/ --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation suite")
    parser.add_argument("--configs", default="evaluation/configs/",
                        help="Directory containing YAML experiment configs")
    parser.add_argument("--output", default="results/",
                        help="Output directory for results")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip vLLM/MoE-Infinity baselines")
    parser.add_argument("--skip-moe-policylang", action="store_true",
                        help="Skip MoE-PolicyLang policy runs")
    args = parser.parse_args()

    print("Phase 4: Full Evaluation Suite — NOT YET IMPLEMENTED")
    print()
    print("This script will:")
    print("  1. Run all MoE-PolicyLang policies × workloads × hardware configs")
    print("  2. Run all baselines (vLLM, MoE-Infinity) on same workloads")
    print("  3. Generate comparison tables and figures")
    print("  4. Export results to JSON for paper figures")
    print()
    print("See PROGRESS.md for status.")
    sys.exit(1)


if __name__ == "__main__":
    main()
