"""Validation rules for MoE-PolicyLang PolicyIR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moe_policylang.errors import ValidationError
from moe_policylang.ir import EvictionPolicy, PrefetchStrategy, ScheduleMode

if TYPE_CHECKING:
    from moe_policylang.ir import PolicyIR


# Each rule is (description, predicate_that_must_be_true).
VALIDATION_RULES: list[tuple[str, callable]] = [
    # -- Cache rules --
    (
        "cache.capacity must be between 1 and 512",
        lambda ir: 1 <= ir.cache.capacity <= 512,
    ),
    (
        "pin_experts must contain valid expert IDs (0..511)",
        lambda ir: all(0 <= e < 512 for e in ir.cache.pin_experts),
    ),
    (
        "cannot pin more experts than cache capacity",
        lambda ir: len(ir.cache.pin_experts) <= ir.cache.capacity,
    ),
    (
        "lfu_decay must be in (0, 1)",
        lambda ir: 0 < ir.cache.lfu_decay < 1,
    ),
    (
        "freq_threshold must be in [0, 1]",
        lambda ir: 0 <= ir.cache.freq_threshold <= 1,
    ),
    (
        "freq_window must be >= 1",
        lambda ir: ir.cache.freq_window >= 1,
    ),
    (
        "score_ema_alpha must be in (0, 1]",
        lambda ir: 0 < ir.cache.score_ema_alpha <= 1,
    ),
    # -- Prefetch rules --
    (
        "lookahead must be >= 1",
        lambda ir: ir.prefetch.lookahead >= 1,
    ),
    (
        "prefetch budget must be >= 1",
        lambda ir: ir.prefetch.budget >= 1,
    ),
    (
        "prefetch budget should not exceed cache capacity",
        lambda ir: ir.prefetch.budget <= ir.cache.capacity,
    ),
    (
        "affinity_threshold must be in [0, 1]",
        lambda ir: 0 <= ir.prefetch.affinity_threshold <= 1,
    ),
    (
        "history_window must be >= 1",
        lambda ir: ir.prefetch.history_window >= 1,
    ),
    # -- Schedule rules --
    (
        "cpu_threshold_ms must be > 0",
        lambda ir: ir.schedule.cpu_threshold_ms > 0,
    ),
    (
        "HYBRID mode requires overlap=True",
        lambda ir: ir.schedule.mode != ScheduleMode.HYBRID or ir.schedule.overlap,
    ),
    # -- Cross-block rules --
    (
        "SCORE eviction requires a non-NONE prefetch strategy",
        lambda ir: (
            ir.cache.eviction != EvictionPolicy.SCORE
            or ir.prefetch.strategy != PrefetchStrategy.NONE
        ),
    ),
    # -- Eviction triggers (Week 4) --
    (
        "memory_threshold must be in (0, 1]",
        lambda ir: (
            ir.cache.memory_threshold is None
            or 0 < ir.cache.memory_threshold <= 1
        ),
    ),
    (
        "memory_headroom must be in (0, memory_threshold]",
        lambda ir: (
            ir.cache.memory_threshold is None
            or 0 < ir.cache.memory_headroom <= ir.cache.memory_threshold
        ),
    ),
    (
        "memory_budget_gb must be > 0 when memory pressure is enabled",
        lambda ir: (
            ir.cache.memory_threshold is None
            or ir.cache.memory_budget_gb > 0
        ),
    ),
    (
        "expert_size_gb must be > 0",
        lambda ir: ir.cache.expert_size_gb > 0,
    ),
    (
        "ttl must be >= 1 when set",
        lambda ir: ir.cache.ttl is None or ir.cache.ttl >= 1,
    ),
]


def validate_policy(ir: "PolicyIR") -> list[str]:
    """Validate a PolicyIR, returning a list of violation descriptions.

    Raises ``ValidationError`` if any rule fails.  Returns an empty list on
    success.
    """
    violations = [desc for desc, pred in VALIDATION_RULES if not pred(ir)]
    if violations:
        raise ValidationError(violations)
    return violations
