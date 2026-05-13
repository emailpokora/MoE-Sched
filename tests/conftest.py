"""Shared fixtures for MoE-PolicyLang tests."""

import random

import pytest

from moe_policylang.ir import (
    CacheIR,
    EvictionPolicy,
    MonitorIR,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    ScheduleIR,
    ScheduleMode,
)


# ---------------------------------------------------------------------------
# Reusable PolicyIR fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_ir() -> PolicyIR:
    """Smallest valid policy: LRU cache, all defaults."""
    return PolicyIR(
        name="minimal",
        cache=CacheIR(capacity=16),
    )


@pytest.fixture
def full_ir() -> PolicyIR:
    """Fully-specified policy with all blocks populated."""
    return PolicyIR(
        name="full",
        cache=CacheIR(
            capacity=32,
            eviction=EvictionPolicy.LFU,
            pin_experts=[0, 1],
            lfu_decay=0.9,
        ),
        prefetch=PrefetchIR(
            strategy=PrefetchStrategy.AFFINITY,
            lookahead=2,
            budget=4,
            affinity_threshold=0.25,
        ),
        schedule=ScheduleIR(
            mode=ScheduleMode.HYBRID,
            cpu_threshold_ms=40.0,
            overlap=True,
        ),
        monitor=MonitorIR(
            metrics=["hit_rate", "latency", "memory"],
            window=200,
            log_interval=50,
        ),
    )


@pytest.fixture
def hybrid_score_ir() -> PolicyIR:
    """Score-based cache + affinity prefetch + hybrid scheduling."""
    return PolicyIR(
        name="hybrid_score",
        cache=CacheIR(
            capacity=32,
            eviction=EvictionPolicy.SCORE,
            score_ema_alpha=0.5,
        ),
        prefetch=PrefetchIR(
            strategy=PrefetchStrategy.AFFINITY,
            budget=4,
        ),
        schedule=ScheduleIR(mode=ScheduleMode.HYBRID, cpu_threshold_ms=30.0),
    )


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_workload():
    """Uniform random expert access sequence (256 experts, 2000 accesses)."""
    rng = random.Random(42)
    return [rng.randint(0, 255) for _ in range(2000)]


@pytest.fixture
def skewed_workload():
    """Skewed workload: experts 0-7 are 10x more likely than 8-255."""
    rng = random.Random(42)
    weights = [10.0] * 8 + [1.0] * 248
    total = sum(weights)
    probs = [w / total for w in weights]
    population = list(range(256))
    return rng.choices(population, weights=probs, k=2000)


@pytest.fixture
def layered_workload():
    """Workload with layer structure: 32 layers × 8 experts, 100 tokens, top-2."""
    rng = random.Random(42)
    accesses = []
    for _token in range(100):
        for layer in range(32):
            experts = rng.sample(range(8), 2)
            for e in experts:
                accesses.append((layer, layer * 8 + e))
    return accesses
