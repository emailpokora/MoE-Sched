"""Pre-defined DSL policies and baselines for the evaluation suite.

Proposal requires: 3+ DSL-specified policies and 2 baselines.
We provide 5 DSL policies covering every caching strategy and 2 baselines.
"""

from __future__ import annotations

from moe_policylang.compiler import CompiledPolicy, compile_policy
from moe_policylang.dsl import MoEPolicyLang


# ---------------------------------------------------------------------------
# DSL policies (programmatic API)
# ---------------------------------------------------------------------------

def _build_policies() -> dict[str, CompiledPolicy]:
    sched = MoEPolicyLang()

    # 1. LRU + no prefetch + GPU-only  (simplest possible DSL policy)
    @sched.policy
    def lru_basic(p):
        p.cache(capacity=16, eviction="lru")
        p.schedule(mode="gpu_only")
        p.monitor(metrics=["hit_rate", "latency"], window=200, log_interval=50)

    # 2. LFU + history prefetch + CPU-fallback  (frequency awareness)
    @sched.policy
    def lfu_history(p):
        p.cache(capacity=16, eviction="lfu", lfu_decay=0.9)
        p.prefetch(strategy="history", budget=4, history_window=60)
        p.schedule(mode="cpu_fallback")
        p.monitor(metrics=["hit_rate", "latency"], window=200, log_interval=50)

    # 3. Score-based + affinity prefetch + hybrid  (HybriMoE-inspired)
    @sched.policy
    def score_affinity(p):
        p.cache(capacity=16, eviction="score", score_ema_alpha=0.3, pin=[0, 1])
        p.prefetch(strategy="affinity", budget=4, affinity_threshold=0.3)
        p.schedule(mode="hybrid", cpu_threshold_ms=40.0)
        p.monitor(metrics=["hit_rate", "latency", "memory"], window=200, log_interval=50)

    # 4. LFU + lookahead + triggers + hybrid  (full composition)
    @sched.policy
    def composed_full(p):
        p.cache(
            capacity=16,
            eviction="lfu",
            lfu_decay=0.9,
            pin=[0],
            memory_threshold=0.75,
            memory_headroom=0.4,
            memory_budget_gb=16.0,
            expert_size_gb=1.2,
            ttl=200,
        )
        p.prefetch(strategy="lookahead", lookahead=2, budget=4, history_window=40)
        p.schedule(mode="hybrid", cpu_threshold_ms=40.0)
        p.monitor(metrics=["hit_rate", "latency", "memory"], window=200, log_interval=50)

    # 5. FreqThreshold + history + GPU-only  (threshold-based caching)
    @sched.policy
    def freq_threshold(p):
        p.cache(capacity=16, eviction="frequency_threshold", freq_threshold=0.05, freq_window=100)
        p.prefetch(strategy="history", budget=4, history_window=60)
        p.schedule(mode="gpu_only")
        p.monitor(metrics=["hit_rate", "latency"], window=200, log_interval=50)

    policies = {}
    for name, ir in sched.policies.items():
        policies[name] = compile_policy(ir)
    return policies


def get_dsl_policies() -> dict[str, CompiledPolicy]:
    """Return all pre-built DSL policies (freshly compiled each call)."""
    return _build_policies()


def get_policy_names() -> list[str]:
    """Return ordered list of policy names."""
    return list(get_dsl_policies().keys())


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

from moe_policylang.baselines import HandCodedLRU, HandCodedLRUFallback

BASELINES = {
    "baseline_lru_gpu": (HandCodedLRU, "Hand-coded LRU + GPU-only"),
    "baseline_lru_cpu_fallback": (HandCodedLRUFallback, "Hand-coded LRU + CPU-fallback"),
}
