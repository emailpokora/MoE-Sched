"""Run a full evaluation experiment from a YAML config.

Phase 4: Orchestrates real hardware experiments.

Status: STUB — implement after Phases 1-3 are complete.

Usage (planned):
    python evaluation/run_experiment.py --config evaluation/configs/mixtral_a100.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(config: Dict[str, Any]) -> None:
    """Execute a full experiment from config.

    TODO (Phase 4):
        1. Load model with HuggingFace
        2. Install MoE-Sched hooks
        3. For each policy × workload:
            a. Compile DSL policy
            b. Build hook
            c. Run inference on trace
            d. Collect metrics (torch.cuda.Event timing)
        4. Save results to output_dir
    """
    raise NotImplementedError(
        "Phase 4 not yet implemented. "
        "See conference-paper/PROGRESS.md for status."
    )


def main():
    parser = argparse.ArgumentParser(description="Run MoE-Sched evaluation experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    run_experiment(config)


if __name__ == "__main__":
    main()
