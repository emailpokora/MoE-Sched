"""Standardised workload definitions for MoE-Sched benchmarks.

Each workload encapsulates:
  * A human-readable name and description.
  * Token count, layer count, expert count, and top-k.
  * A deterministic expert-selector factory (so results are reproducible).
  * A characteristic: temporal locality, uniform randomness, bursty, etc.

The proposal calls for "short prompts, long-context, mixed batch" workloads.
We add a *bursty* workload that cycles between hot and cold phases to stress
prefetchers and eviction triggers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass(frozen=True)
class Workload:
    """Immutable workload specification."""

    name: str
    description: str
    num_tokens: int
    num_layers: int
    num_experts: int
    top_k: int
    selector_factory: Callable[[], Callable[[int, int], List[int]]]
    seed: int = 42

    def make_selector(self) -> Callable[[int, int], List[int]]:
        return self.selector_factory()


# ---------------------------------------------------------------------------
# Selector factories
# ---------------------------------------------------------------------------

def _skewed_factory(
    num_experts: int,
    top_k: int,
    hot_fraction: float = 0.2,
    hot_weight: float = 0.8,
    seed: int = 42,
) -> Callable[[], Callable[[int, int], List[int]]]:
    """Factory that returns a factory (workload is frozen, so we defer RNG)."""
    def factory() -> Callable[[int, int], List[int]]:
        rng = random.Random(seed)
        n_hot = max(1, int(num_experts * hot_fraction))
        hot = list(range(n_hot))
        cold = list(range(n_hot, num_experts))

        def select(token_idx: int, layer_idx: int) -> List[int]:
            chosen: List[int] = []
            while len(chosen) < top_k:
                pool = hot if rng.random() < hot_weight else cold
                if not pool:
                    pool = hot or cold
                e = rng.choice(pool)
                if e not in chosen:
                    chosen.append(e)
            return chosen
        return select
    return factory



def _bursty_factory(
    num_experts: int,
    top_k: int,
    hot_fraction: float = 0.15,
    burst_len: int = 30,
    seed: int = 42,
) -> Callable[[], Callable[[int, int], List[int]]]:
    """Alternates between hot-only and uniform phases every *burst_len* tokens."""
    def factory() -> Callable[[int, int], List[int]]:
        rng = random.Random(seed)
        n_hot = max(1, int(num_experts * hot_fraction))
        hot = list(range(n_hot))
        all_experts = list(range(num_experts))

        def select(token_idx: int, layer_idx: int) -> List[int]:
            phase = (token_idx // burst_len) % 2
            pool = hot if phase == 0 else all_experts
            chosen: List[int] = []
            while len(chosen) < top_k:
                e = rng.choice(pool)
                if e not in chosen:
                    chosen.append(e)
            return chosen
        return select
    return factory


def _mixed_factory(
    num_experts: int,
    top_k: int,
    seed: int = 42,
) -> Callable[[], Callable[[int, int], List[int]]]:
    """Simulates a mixed batch: first half of tokens are short-prompt (high
    locality), second half are long-context (moderate locality)."""
    def factory() -> Callable[[int, int], List[int]]:
        rng = random.Random(seed)
        n_hot = max(1, int(num_experts * 0.2))
        hot = list(range(n_hot))
        warm = list(range(int(num_experts * 0.4)))
        cold = list(range(num_experts))

        def select(token_idx: int, layer_idx: int) -> List[int]:
            # Short-prompt regime: high locality
            if token_idx < 50:
                pool = hot if rng.random() < 0.85 else cold
            else:
                # Long-context regime: moderate locality
                pool = warm if rng.random() < 0.5 else cold
            chosen: List[int] = []
            while len(chosen) < top_k:
                e = rng.choice(pool)
                if e not in chosen:
                    chosen.append(e)
            return chosen
        return select
    return factory


# ---------------------------------------------------------------------------
# Standard workloads
# ---------------------------------------------------------------------------

def short_prompt_workload() -> Workload:
    """Short prompts: 50 tokens, strong temporal locality."""
    ne, tk = 64, 2
    return Workload(
        name="short_prompt",
        description="50 tokens, 32 layers, 64 experts, top-2, high locality (80% hot)",
        num_tokens=50,
        num_layers=32,
        num_experts=ne,
        top_k=tk,
        selector_factory=_skewed_factory(ne, tk, hot_fraction=0.2, hot_weight=0.8),
    )


def long_context_workload() -> Workload:
    """Long context: 500 tokens, moderate locality."""
    ne, tk = 64, 2
    return Workload(
        name="long_context",
        description="500 tokens, 32 layers, 64 experts, top-2, moderate locality (60% hot)",
        num_tokens=500,
        num_layers=32,
        num_experts=ne,
        top_k=tk,
        selector_factory=_skewed_factory(ne, tk, hot_fraction=0.25, hot_weight=0.6),
    )


def mixed_batch_workload() -> Workload:
    """Mixed batch: 100 tokens split between short-prompt and long-context regimes."""
    ne, tk = 64, 2
    return Workload(
        name="mixed_batch",
        description="100 tokens, 32 layers, 64 experts, top-2, mixed locality pattern",
        num_tokens=100,
        num_layers=32,
        num_experts=ne,
        top_k=tk,
        selector_factory=_mixed_factory(ne, tk),
    )


def bursty_workload() -> Workload:
    """Bursty: 200 tokens, alternating hot-only and uniform phases."""
    ne, tk = 64, 2
    return Workload(
        name="bursty",
        description="200 tokens, 32 layers, 64 experts, top-2, bursty hot/cold phases",
        num_tokens=200,
        num_layers=32,
        num_experts=ne,
        top_k=tk,
        selector_factory=_bursty_factory(ne, tk, burst_len=30),
    )


ALL_WORKLOADS = [
    short_prompt_workload(),
    long_context_workload(),
    mixed_batch_workload(),
    bursty_workload(),
]
