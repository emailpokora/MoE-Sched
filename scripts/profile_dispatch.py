# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Profile MoE-PolicyLang dispatch overhead on real hardware.

Phase 2: Measures per-layer dispatch time for each policy to verify
the <5% overhead claim.

Status: STUB — implement after Cython fast path (Phase 2).

Usage (planned):
    python scripts/profile_dispatch.py --policy lru_basic --iterations 10000
    python scripts/profile_dispatch.py --all --output results/dispatch_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_policy(policy_name: str, iterations: int = 10000) -> dict:
    """Profile a single policy's dispatch overhead.

    TODO (Phase 2):
        1. Compile the DSL policy
        2. Build hook
        3. Run dispatch loop (without real model) for N iterations
        4. Measure wall-clock time per on_layer() call
        5. Compare pure-Python vs Cython fast path
        6. Report: mean, p50, p99, max in microseconds
    """
    raise NotImplementedError("Phase 2 not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Profile dispatch overhead")
    parser.add_argument("--policy", help="Policy name to profile")
    parser.add_argument("--all", action="store_true", help="Profile all policies")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--compare-fast", action="store_true",
                        help="Compare Python vs Cython implementations")
    args = parser.parse_args()

    print("Phase 2: Dispatch Profiling — NOT YET IMPLEMENTED")
    print("See PROGRESS.md for status.")
    sys.exit(1)


if __name__ == "__main__":
    main()
