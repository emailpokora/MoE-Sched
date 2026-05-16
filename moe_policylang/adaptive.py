"""Adaptive policy support — runtime policy switching based on profiling data.

Extends MoE-PolicyLang with declarative ``adapt`` blocks that monitor runtime
metrics (hit_rate, eviction_rate, etc.) and dynamically adjust policy
parameters when conditions are met.  This makes the profiling strategy
part of the policy rules, enabling automatic expert placement adaptation.

Example DSL syntax:
    policy adaptive_lfu {
        cache { capacity = 16  eviction = lfu  lfu_decay = 0.9 }
        prefetch { strategy = history  budget = 4 }
        adapt {
            when hit_rate < 0.4 for 100 steps { eviction = lru }
            when memory_usage > 0.9 { trigger memory_pressure }
        }
    }

Example Python eDSL:
    sched = MoEPolicyLang()
    @sched.policy
    def my_adaptive(p):
        p.cache(capacity=16, eviction='lfu', lfu_decay=0.9)
        p.prefetch(strategy='history', budget=4)
        p.adapt([
            AdaptRule(
                condition=AdaptCondition('hit_rate', '<', 0.4, window=100),
                action=AdaptAction(param='eviction', value='lru'),
            ),
        ])
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from moe_policylang.runtime.hooks import PolicyHook


# ---------------------------------------------------------------------------
# IR dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AdaptCondition:
    """A condition that triggers policy adaptation."""

    metric: str       # 'hit_rate', 'eviction_rate'
    op: str           # '<', '>', '<=', '>='
    threshold: float
    window: int = 1   # number of consecutive evals the condition must hold


@dataclass
class AdaptAction:
    """An action taken when an adapt condition fires."""

    param: str   # 'eviction', 'capacity', 'prefetch_strategy', ...
    value: str   # 'lru', '32', 'history', 'memory_pressure', ...


@dataclass
class AdaptRule:
    """A condition → action pair with cooldown."""

    condition: AdaptCondition
    action: AdaptAction
    cooldown: int = 50   # min steps between firings of this rule


@dataclass
class AdaptIR:
    """IR extension for adaptive policies."""

    rules: List[AdaptRule] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_OPS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "!=": operator.ne,
}


def _eval_condition(value: float, op: str, threshold: float) -> bool:
    fn = _OPS.get(op)
    if fn is None:
        raise ValueError(f"Unknown comparison operator: {op}")
    return fn(value, threshold)


# ---------------------------------------------------------------------------
# AdaptiveHook — wraps a PolicyHook with runtime adaptation
# ---------------------------------------------------------------------------

class AdaptiveHook:
    """Wraps a PolicyHook, monitoring metrics and swapping components.

    After every ``on_layer`` call, the hook checks each adapt rule.
    When a condition is met for the required window of consecutive
    evaluations, the corresponding action fires and the underlying
    policy components are hot-swapped via recompilation.
    """

    def __init__(
        self,
        base_hook: "PolicyHook",
        adapt_ir: AdaptIR,
        policy_ir: object,  # PolicyIR — forward ref to avoid circular import
        compile_fn: Callable,
    ) -> None:
        self.base = base_hook
        self.rules = adapt_ir.rules
        self._ir = policy_ir
        self._compile = compile_fn
        self._step = 0
        self._last_fired: dict[int, int] = {}     # rule index → step last fired
        self._streak: dict[int, int] = {}          # rule index → consecutive True evals
        self._adaptations: list[dict] = []         # log of all adaptation events
        self._skipped: list[dict] = []                 # log of skipped (invalid) adaptations

    # -- delegate to base hook -----------------------------------------------

    def on_layer(self, layer_idx, selected_experts, **kwargs):
        plan = self.base.on_layer(layer_idx, selected_experts, **kwargs)
        self._step += 1
        self._check_rules()
        return plan

    def stats_snapshot(self) -> dict:
        snap = self.base.stats_snapshot()
        snap["adapt"] = {
            "adaptations": len(self._adaptations),
            "skipped": len(self._skipped),
            "log": list(self._adaptations),
            "skipped_log": list(self._skipped),
        }
        return snap

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def compiled(self):
        return self.base.compiled

    @property
    def cache(self):
        return self.base.cache

    @property
    def prefetcher(self):
        return self.base.prefetcher

    @property
    def scheduler(self):
        return self.base.scheduler

    @property
    def monitor(self):
        return self.base.monitor

    @property
    def triggers(self):
        return self.base.triggers

    # -- rule evaluation -----------------------------------------------------

    def _check_rules(self) -> None:
        stats = self.base.stats_snapshot()
        for i, rule in enumerate(self.rules):
            # Cooldown check
            last = self._last_fired.get(i, -rule.cooldown - 1)
            if self._step - last < rule.cooldown:
                self._streak[i] = 0
                continue

            # Evaluate condition
            val = self._get_metric(stats, rule.condition.metric)
            if val is not None and _eval_condition(val, rule.condition.op, rule.condition.threshold):
                self._streak[i] = self._streak.get(i, 0) + 1
            else:
                self._streak[i] = 0

            # Check window requirement
            if self._streak[i] >= rule.condition.window:
                self._apply(i, rule)
                self._streak[i] = 0

    def _get_metric(self, stats: dict, metric: str) -> Optional[float]:
        if metric == "hit_rate":
            return stats["cache"]["hit_rate"]
        if metric == "eviction_rate":
            h = stats["cache"]["hits"]
            m = stats["cache"]["misses"]
            e = stats["cache"]["evictions"]
            total = h + m
            return e / total if total > 0 else 0.0
        return None

    def _apply(self, rule_idx: int, rule: AdaptRule) -> None:
        """Apply an adaptation action by modifying IR and recompiling."""
        import copy
        from moe_policylang.ir import EvictionPolicy, PrefetchStrategy, ScheduleMode

        self._last_fired[rule_idx] = self._step

        action = rule.action
        new_ir = copy.deepcopy(self._ir)

        applied = False

        # -- Cache parameter changes --
        if action.param == "eviction":
            try:
                new_ir.cache.eviction = EvictionPolicy(action.value)
                applied = True
            except ValueError:
                pass
        elif action.param == "capacity":
            new_ir.cache.capacity = int(float(action.value))
            applied = True
        elif action.param == "lfu_decay":
            new_ir.cache.lfu_decay = float(action.value)
            applied = True

        # -- Prefetch parameter changes --
        elif action.param == "prefetch_strategy":
            try:
                new_ir.prefetch.strategy = PrefetchStrategy(action.value)
                applied = True
            except ValueError:
                pass
        elif action.param == "prefetch_budget":
            new_ir.prefetch.budget = int(float(action.value))
            applied = True

        # -- Schedule parameter changes --
        elif action.param == "schedule_mode":
            try:
                new_ir.schedule.mode = ScheduleMode(action.value)
                applied = True
            except ValueError:
                pass

        # -- Trigger activation --
        elif action.param == "trigger":
            if action.value == "memory_pressure":
                if new_ir.cache.memory_threshold is None:
                    new_ir.cache.memory_threshold = 0.9
                applied = True

        # -- Per-layer rebalance (EPCB) --
        elif action.param == "rebalance":
            if hasattr(self.base, "_rebalance"):
                self.base._rebalance()
                self._last_fired[rule_idx] = self._step
                self._adaptations.append({
                    "step": self._step,
                    "rule": rule_idx,
                    "param": action.param,
                    "value": action.value,
                })
            else:
                self._skipped.append({
                    "step": self._step,
                    "rule": rule_idx,
                    "param": action.param,
                    "value": action.value,
                    "reason": "base hook does not support rebalance "
                              "(requires per_layer block)",
                })
                logger.warning(
                    "Adaptive rule %d skipped at step %d: rebalance=%s "
                    "but base hook has no _rebalance method",
                    rule_idx, self._step, action.value,
                )
            return  # rebalance is handled directly, skip IR recompilation

        if applied:
            from moe_policylang.runtime.hooks import PolicyHook
            from moe_policylang.validator import validate_policy

            try:
                validate_policy(new_ir)
            except Exception as exc:
                detail = {
                    "step": self._step,
                    "rule": rule_idx,
                    "param": action.param,
                    "value": action.value,
                    "reason": str(exc),
                }
                self._skipped.append(detail)
                logger.warning(
                    "Adaptive rule %d skipped at step %d: %s=%s → %s",
                    rule_idx, self._step, action.param, action.value, exc,
                )
                return

            new_compiled = self._compile(new_ir)
            # Use PolicyHook directly — NOT build_hook — to avoid nested
            # AdaptiveHook wrapping.  The outer AdaptiveHook (self) is
            # already the adaptation layer; the base must stay a plain hook.
            new_hook = PolicyHook(new_compiled)

            self.base = new_hook
            self._ir = new_ir

            self._adaptations.append({
                "step": self._step,
                "rule": rule_idx,
                "param": action.param,
                "value": action.value,
            })
