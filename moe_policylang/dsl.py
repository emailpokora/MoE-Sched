"""DSL frontend: decorator and builder APIs for defining MoE-PolicyLang policies."""

from __future__ import annotations

from moe_policylang.adaptive import AdaptIR, AdaptRule
from moe_policylang.errors import DSLError
from moe_policylang.ir import (
    AllocationSignal,
    CacheIR,
    EvictionPolicy,
    MonitorIR,
    PerLayerIR,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    ScheduleIR,
    ScheduleMode,
)
from moe_policylang.validator import validate_policy


# ---------------------------------------------------------------------------
# Policy builder (used inside @sched.policy decorator)
# ---------------------------------------------------------------------------

class PolicyBuilder:
    """Collects DSL calls and produces a validated PolicyIR."""

    def __init__(self) -> None:
        self._cache: CacheIR | None = None
        self._prefetch: PrefetchIR | None = None
        self._schedule: ScheduleIR | None = None
        self._monitor: MonitorIR | None = None
        self._adapt: AdaptIR | None = None
        self._per_layer: PerLayerIR | None = None

    # -- DSL primitives -----------------------------------------------------

    def cache(
        self,
        capacity: int,
        eviction: EvictionPolicy | str = EvictionPolicy.LRU,
        pin: list[int] | None = None,
        lfu_decay: float = 0.95,
        freq_threshold: float = 0.05,
        freq_window: int = 100,
        score_ema_alpha: float = 0.3,
        memory_threshold: float | None = None,
        memory_headroom: float = 0.7,
        memory_budget_gb: float = 16.0,
        expert_size_gb: float = 1.2,
        ttl: int | None = None,
    ) -> None:
        if self._cache is not None:
            raise DSLError("Duplicate cache() block in policy definition")
        if isinstance(eviction, str):
            eviction = EvictionPolicy(eviction)
        self._cache = CacheIR(
            capacity=capacity,
            eviction=eviction,
            pin_experts=pin or [],
            lfu_decay=lfu_decay,
            freq_threshold=freq_threshold,
            freq_window=freq_window,
            score_ema_alpha=score_ema_alpha,
            memory_threshold=memory_threshold,
            memory_headroom=memory_headroom,
            memory_budget_gb=memory_budget_gb,
            expert_size_gb=expert_size_gb,
            ttl=ttl,
        )

    def prefetch(
        self,
        strategy: PrefetchStrategy | str = PrefetchStrategy.NONE,
        lookahead: int = 1,
        budget: int = 4,
        affinity_threshold: float = 0.3,
        history_window: int = 50,
    ) -> None:
        if self._prefetch is not None:
            raise DSLError("Duplicate prefetch() block in policy definition")
        if isinstance(strategy, str):
            strategy = PrefetchStrategy(strategy)
        self._prefetch = PrefetchIR(
            strategy=strategy,
            lookahead=lookahead,
            budget=budget,
            affinity_threshold=affinity_threshold,
            history_window=history_window,
        )

    def schedule(
        self,
        mode: ScheduleMode | str = ScheduleMode.GPU_ONLY,
        cpu_threshold_ms: float = 50.0,
        overlap: bool = True,
        priority_routing: bool = False,
    ) -> None:
        if self._schedule is not None:
            raise DSLError("Duplicate schedule() block in policy definition")
        if isinstance(mode, str):
            mode = ScheduleMode(mode)
        self._schedule = ScheduleIR(
            mode=mode,
            cpu_threshold_ms=cpu_threshold_ms,
            overlap=overlap,
            priority_routing=priority_routing,
        )

    def monitor(
        self,
        metrics: list[str] | None = None,
        window: int = 100,
        log_interval: int = 50,
    ) -> None:
        if self._monitor is not None:
            raise DSLError("Duplicate monitor() block in policy definition")
        self._monitor = MonitorIR(
            metrics=metrics or ["hit_rate"],
            window=window,
            log_interval=log_interval,
        )

    def adapt(self, rules: list[AdaptRule]) -> None:
        """Define adaptive rules that dynamically adjust the policy.

        Args:
            rules: List of :class:`~moe_policylang.adaptive.AdaptRule` objects.
        """
        if self._adapt is not None:
            raise DSLError("Duplicate adapt() block in policy definition")
        self._adapt = AdaptIR(rules=list(rules))

    def per_layer(
        self,
        allocation: AllocationSignal | str = AllocationSignal.ENTROPY,
        entropy_window: int = 200,
        min_capacity: int = 2,
        max_capacity: int = 64,
        rebalance_interval: int = 500,
        total_budget: int | None = None,
    ) -> None:
        """Enable per-layer EPCB allocation."""
        if self._per_layer is not None:
            raise DSLError("Duplicate per_layer() block in policy definition")
        if isinstance(allocation, str):
            allocation = AllocationSignal(allocation)
        self._per_layer = PerLayerIR(
            allocation=allocation,
            entropy_window=entropy_window,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            rebalance_interval=rebalance_interval,
            total_budget=total_budget,
        )

    # -- Build --------------------------------------------------------------

    def _build(self, name: str) -> PolicyIR:
        if self._cache is None:
            raise DSLError("Policy must include a cache() block")
        ir = PolicyIR(
            name=name,
            cache=self._cache,
            prefetch=self._prefetch or PrefetchIR(),
            schedule=self._schedule or ScheduleIR(),
            monitor=self._monitor,
            adapt=self._adapt,
            per_layer=self._per_layer,
        )
        validate_policy(ir)
        return ir


# ---------------------------------------------------------------------------
# Fluent builder (alternative syntax)
# ---------------------------------------------------------------------------

class FluentPolicyBuilder:
    """Method-chaining builder for policies."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._inner = PolicyBuilder()

    def cache(self, **kwargs) -> "FluentPolicyBuilder":
        self._inner.cache(**kwargs)
        return self

    def prefetch(self, **kwargs) -> "FluentPolicyBuilder":
        self._inner.prefetch(**kwargs)
        return self

    def schedule(self, **kwargs) -> "FluentPolicyBuilder":
        self._inner.schedule(**kwargs)
        return self

    def monitor(self, **kwargs) -> "FluentPolicyBuilder":
        self._inner.monitor(**kwargs)
        return self

    def adapt(self, rules: list[AdaptRule]) -> "FluentPolicyBuilder":
        self._inner.adapt(rules)
        return self

    def per_layer(self, **kwargs) -> "FluentPolicyBuilder":
        self._inner.per_layer(**kwargs)
        return self

    def done(self) -> PolicyIR:
        return self._inner._build(self._name)


# ---------------------------------------------------------------------------
# Main framework object
# ---------------------------------------------------------------------------

class MoEPolicyLang:
    """Entry point for the MoE-PolicyLang DSL."""

    def __init__(self) -> None:
        self.policies: dict[str, PolicyIR] = {}

    def policy(self, func):
        """Decorator that registers a policy definition function."""
        builder = PolicyBuilder()
        func(builder)
        ir = builder._build(func.__name__)
        self.policies[func.__name__] = ir
        return ir

    def build(self, name: str) -> FluentPolicyBuilder:
        """Start a fluent builder for a named policy."""
        return FluentPolicyBuilder(name)

    def register(self, ir: PolicyIR) -> None:
        """Register a pre-built PolicyIR."""
        self.policies[ir.name] = ir
