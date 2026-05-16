"""Text-based DSL parser for MoE-PolicyLang policies.

Parses `.moe` policy files (or raw strings) into validated PolicyIR objects
using a Lark LALR grammar.  This complements the existing Python eDSL
(decorator + fluent APIs) with a standalone, language-agnostic syntax.

Usage:
    from moe_policylang.parser import parse_policies, parse_policy

    # Multiple policies from a file or string
    policies = parse_policies(Path("examples/lru_policy.moe").read_text())

    # Single policy (convenience)
    policy = parse_policy('''
        policy my_lru {
            cache { capacity = 8  eviction = lru }
        }
    ''')
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from lark import Lark, Transformer, v_args
from lark.exceptions import VisitError

from moe_policylang.errors import DSLError, ValidationError
from moe_policylang.adaptive import AdaptAction, AdaptCondition, AdaptIR, AdaptRule
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
# Grammar
# ---------------------------------------------------------------------------

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"

_parser = Lark(
    _GRAMMAR_PATH.read_text(),
    parser="lalr",
    maybe_placeholders=False,
)


# ---------------------------------------------------------------------------
# Tree → IR transformer
# ---------------------------------------------------------------------------

_INVERT_OP = {"<": ">=", ">": "<=", "<=": ">", ">=": "<", "!=": "!="}

_KNOWN_METRICS = {"hit_rate", "eviction_rate"}


def _validate_metric(name: str) -> None:
    """Raise DSLError if metric name is not recognized."""
    if name not in _KNOWN_METRICS:
        valid = ", ".join(sorted(_KNOWN_METRICS))
        raise DSLError(f"Unknown metric '{name}'. Valid metrics: {valid}")


_EVICTION_MAP = {
    "lru": EvictionPolicy.LRU,
    "lfu": EvictionPolicy.LFU,
    "score": EvictionPolicy.SCORE,
    "frequency_threshold": EvictionPolicy.FREQ_THRESHOLD,
    "fallback": EvictionPolicy.FALLBACK,
}

_PREFETCH_MAP = {
    "none": PrefetchStrategy.NONE,
    "affinity": PrefetchStrategy.AFFINITY,
    "history": PrefetchStrategy.HISTORY,
    "lookahead": PrefetchStrategy.LOOKAHEAD,
}

_SCHEDULE_MAP = {
    "gpu_only": ScheduleMode.GPU_ONLY,
    "cpu_fallback": ScheduleMode.CPU_FALLBACK,
    "hybrid": ScheduleMode.HYBRID,
}

# Combined map for adapt actions that set enum params at runtime
_ENUM_MAP = {**_EVICTION_MAP, **_PREFETCH_MAP, **_SCHEDULE_MAP}


def _resolve_enum(value, enum_map: dict, param_name: str):
    """Look up an enum value strictly, raising DSLError on unknown values."""
    key = str(value)
    result = enum_map.get(key)
    if result is None:
        valid = ", ".join(sorted(enum_map.keys()))
        line = getattr(value, "line", None)
        raise DSLError(f"Unknown {param_name} '{key}'. Valid options: {valid}", line=line)
    return result


@v_args(inline=True)
class _IRBuilder(Transformer):
    """Transforms a Lark parse tree into a list of PolicyIR objects."""

    # -- terminals ----------------------------------------------------------

    def INT(self, token):
        return int(token)

    def NUMBER(self, token):
        return float(token)

    def ENUM_VAL(self, token):
        return token  # keep as Token to preserve .line metadata

    def BOOL_VAL(self, token):
        return str(token) == "true"

    def NAME(self, token):
        return str(token)

    # -- lists --------------------------------------------------------------

    def int_list(self, *items):
        return list(items)

    def empty_int_list(self):
        return []

    def name_list(self, *items):
        return [str(i) for i in items]

    def empty_name_list(self):
        return []

    # -- cache params -------------------------------------------------------

    def cache_capacity(self, v):
        return ("capacity", v)

    def cache_eviction(self, v):
        return ("eviction", v)

    def cache_pin(self, v):
        return ("pin_experts", v)

    def cache_lfu_decay(self, v):
        return ("lfu_decay", v)

    def cache_freq_threshold(self, v):
        return ("freq_threshold", v)

    def cache_freq_window(self, v):
        return ("freq_window", v)

    def cache_score_ema_alpha(self, v):
        return ("score_ema_alpha", v)

    def cache_memory_threshold(self, v):
        return ("memory_threshold", v)

    def cache_memory_headroom(self, v):
        return ("memory_headroom", v)

    def cache_memory_budget(self, v):
        return ("memory_budget_gb", v)

    def cache_expert_size(self, v):
        return ("expert_size_gb", v)

    def cache_ttl(self, v):
        return ("ttl", v)

    # -- conditional cache params (default : value when cond) ----------------

    def _make_cond_rule(self, param, when_val, metric, op, threshold):
        """Build an AdaptRule for ``default : when_val when metric op thresh``."""
        _validate_metric(str(metric))
        return AdaptRule(
            condition=AdaptCondition(
                metric=str(metric),
                op=str(op),
                threshold=float(threshold),
                window=1,
            ),
            action=AdaptAction(param=param, value=str(when_val)),
        )

    def cache_capacity_cond(self, default_val, when_val, metric, op, threshold):
        rule = self._make_cond_rule("capacity", when_val, metric, op, threshold)
        return ("capacity", default_val, ("_implicit_rule", rule))

    def cache_eviction_fallback(self, primary_str, secondary_str):
        return ("eviction", "fallback", ("_fallback", str(primary_str), str(secondary_str), None, None))

    def cache_eviction_cond(self, default_val, when_val, metric, op, threshold):
        rule = self._make_cond_rule("eviction", when_val, metric, op, threshold)
        return ("eviction", default_val, ("_implicit_rule", rule))

    def cache_lfu_decay_cond(self, default_val, when_val, metric, op, threshold):
        rule = self._make_cond_rule("lfu_decay", when_val, metric, op, threshold)
        return ("lfu_decay", default_val, ("_implicit_rule", rule))

    # -- prefetch params ----------------------------------------------------

    def prefetch_strategy(self, v):
        return ("strategy", v)

    def prefetch_strategy_cond(self, default_val, when_val, metric, op, threshold):
        rule = self._make_cond_rule("prefetch_strategy", when_val, metric, op, threshold)
        return ("strategy", default_val, ("_implicit_rule", rule))

    def prefetch_budget_cond(self, default_val, when_val, metric, op, threshold):
        rule = self._make_cond_rule("prefetch_budget", when_val, metric, op, threshold)
        return ("budget", default_val, ("_implicit_rule", rule))

    def prefetch_lookahead(self, v):
        return ("lookahead", v)

    def prefetch_budget(self, v):
        return ("budget", v)

    def prefetch_affinity_threshold(self, v):
        return ("affinity_threshold", v)

    def prefetch_history_window(self, v):
        return ("history_window", v)

    # -- schedule params ----------------------------------------------------

    def schedule_mode(self, v):
        return ("mode", v)

    def schedule_cpu_threshold(self, v):
        return ("cpu_threshold_ms", v)

    def schedule_overlap(self, v):
        return ("overlap", v)

    def schedule_priority_routing(self, v):
        return ("priority_routing", v)

    # -- monitor params -----------------------------------------------------

    def monitor_metrics(self, v):
        return ("metrics", v)

    def monitor_window(self, v):
        return ("window", v)

    def monitor_log_interval(self, v):
        return ("log_interval", v)

    # -- adapt params -------------------------------------------------------

    def adapt_set_param(self, name, value):
        return AdaptAction(param=str(name), value=str(value))

    def adapt_set_number(self, name, value):
        return AdaptAction(param=str(name), value=str(value))

    def adapt_set_int(self, name, value):
        return AdaptAction(param=str(name), value=str(value))

    def adapt_trigger(self, value):
        return AdaptAction(param="trigger", value=str(value))

    def adapt_rebalance(self, value):
        return AdaptAction(param="rebalance", value=str(value))

    def adapt_rule_windowed(self, metric, op, threshold, window, action):
        _validate_metric(str(metric))
        return AdaptRule(
            condition=AdaptCondition(
                metric=str(metric),
                op=str(op),
                threshold=float(threshold),
                window=int(window),
            ),
            action=action,
        )

    def adapt_rule_instant(self, metric, op, threshold, action):
        _validate_metric(str(metric))
        return AdaptRule(
            condition=AdaptCondition(
                metric=str(metric),
                op=str(op),
                threshold=float(threshold),
                window=1,
            ),
            action=action,
        )

    def COMP_OP(self, token):
        return str(token)

    # -- block assembly -----------------------------------------------------

    @staticmethod
    def _check_duplicates(params):
        """Convert (key, value) pairs to dict, raising on duplicates."""
        kw = {}
        for key, value in params:
            if key in kw:
                raise DSLError(f"Duplicate parameter '{key}'")
            kw[key] = value
        return kw

    def _extract_implicit_rules(self, params):
        """Separate plain (key, value) pairs from conditional/fallback triples."""
        kw = {}
        implicit_rules = []
        fallback_info = None
        for item in params:
            key = item[0]
            if key in kw:
                raise DSLError(f"Duplicate parameter '{key}'")
            if len(item) == 3 and isinstance(item[2], tuple):
                tag = item[2][0]
                if tag == "_implicit_rule":
                    kw[key] = item[1]
                    implicit_rules.append(item[2][1])
                elif tag == "_fallback":
                    kw[key] = item[1]
                    # (primary_str, secondary_str, primary_cap_or_None, secondary_cap_or_None)
                    fallback_info = (item[2][1], item[2][2], item[2][3], item[2][4])
                else:
                    kw[key] = item[1]
            else:
                kw[key] = item[1]
        return kw, implicit_rules, fallback_info

    def cache_block(self, *params):
        kw, implicit_rules, fallback_info = self._extract_implicit_rules(params)
        if "capacity" not in kw:
            raise DSLError("cache block requires 'capacity'")
        eviction_str = kw.pop("eviction", "lru")
        if fallback_info:
            primary_str, secondary_str, primary_cap, secondary_cap = fallback_info
            kw["eviction"] = EvictionPolicy.FALLBACK
            kw["fallback_eviction"] = _resolve_enum(secondary_str, _EVICTION_MAP, "eviction")
            kw["_primary_eviction"] = _resolve_enum(primary_str, _EVICTION_MAP, "eviction")
            kw["_primary_cap"] = primary_cap
            kw["_secondary_cap"] = secondary_cap
        else:
            kw["eviction"] = _resolve_enum(eviction_str, _EVICTION_MAP, "eviction")
        # Remove private fields before constructing CacheIR
        primary_eviction = kw.pop("_primary_eviction", None)
        primary_cap = kw.pop("_primary_cap", None)
        secondary_cap = kw.pop("_secondary_cap", None)
        cache_ir = CacheIR(**kw)
        if primary_eviction is not None:
            cache_ir._primary_eviction = primary_eviction
        if primary_cap is not None:
            cache_ir._primary_cap = primary_cap
        if secondary_cap is not None:
            cache_ir._secondary_cap = secondary_cap
        result = ("cache", cache_ir)
        if implicit_rules:
            return result + (implicit_rules,)
        return result

    def prefetch_block(self, *params):
        kw, implicit_rules, _ = self._extract_implicit_rules(params)
        strategy_str = kw.pop("strategy", "none")
        kw["strategy"] = _resolve_enum(strategy_str, _PREFETCH_MAP, "prefetch strategy")
        result = ("prefetch", PrefetchIR(**kw))
        if implicit_rules:
            return result + (implicit_rules,)
        return result

    def schedule_block(self, *params):
        kw = self._check_duplicates(params)
        mode_str = kw.pop("mode", "gpu_only")
        kw["mode"] = _resolve_enum(mode_str, _SCHEDULE_MAP, "schedule mode")
        return ("schedule", ScheduleIR(**kw))

    def monitor_block(self, *params):
        kw = self._check_duplicates(params)
        return ("monitor", MonitorIR(**kw))

    def adapt_block(self, *rules):
        return ("adapt", AdaptIR(rules=list(rules)))

    # -- per_layer params ---------------------------------------------------

    _ALLOCATION_MAP = {
        "entropy": AllocationSignal.ENTROPY,
        "uniform": AllocationSignal.UNIFORM,
    }

    def per_layer_allocation(self, v):
        return ("allocation", _resolve_enum(v, self._ALLOCATION_MAP, "allocation signal"))

    def per_layer_entropy_window(self, v):
        return ("entropy_window", v)

    def per_layer_min_cap(self, v):
        return ("min_capacity", v)

    def per_layer_max_cap(self, v):
        return ("max_capacity", v)

    def per_layer_rebalance_interval(self, v):
        return ("rebalance_interval", v)

    def per_layer_total_budget(self, v):
        return ("total_budget", v)

    def per_layer_block(self, *params):
        kw = self._check_duplicates(params)
        return ("per_layer", PerLayerIR(**kw))

    def block(self, item):
        return item

    # -- policy assembly ----------------------------------------------------

    def policy(self, name, *blocks):
        block_dict = {}
        implicit_rules = []
        for block in blocks:
            key = block[0]
            value = block[1]
            # Blocks with conditional expressions carry a third element: list of AdaptRules
            if len(block) == 3:
                implicit_rules.extend(block[2])
            if key in block_dict:
                raise DSLError(f"Duplicate '{key}' block in policy '{name}'")
            block_dict[key] = value

        if "cache" not in block_dict:
            raise DSLError(f"Policy '{name}' must include a cache block")

        # Merge implicit rules from conditional expressions with explicit adapt block
        adapt = block_dict.get("adapt")
        if implicit_rules:
            if adapt is None:
                adapt = AdaptIR(rules=implicit_rules)
            else:
                adapt = AdaptIR(rules=implicit_rules + list(adapt.rules))

        ir = PolicyIR(
            name=name,
            cache=block_dict["cache"],
            prefetch=block_dict.get("prefetch", PrefetchIR()),
            schedule=block_dict.get("schedule", ScheduleIR()),
            monitor=block_dict.get("monitor"),
            adapt=adapt,
            per_layer=block_dict.get("per_layer"),
        )
        validate_policy(ir)
        return ir

    # -- version declaration -------------------------------------------------

    _CURRENT_VERSION = 0.7

    def version_decl(self, version_num):
        if float(version_num) > self._CURRENT_VERSION:
            raise DSLError(
                f"This file requires MoE-PolicyLang grammar >= {version_num}, "
                f"but the current parser supports up to {self._CURRENT_VERSION}"
            )
        return None  # consumed; not passed to start

    def start(self, *children):
        return [c for c in children if c is not None]


_transformer = _IRBuilder()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_policies(source: str) -> List[PolicyIR]:
    """Parse a MoE-PolicyLang source string and return a list of validated PolicyIR.

    Args:
        source: The policy source text (contents of a .moe file).

    Returns:
        List of PolicyIR objects, one per ``policy`` block in the source.

    Raises:
        DSLError: If the source has structural issues (missing cache, duplicates).
        ValidationError: If any policy violates semantic constraints.
        lark.exceptions.UnexpectedInput: If the source has syntax errors.
    """
    tree = _parser.parse(source)
    try:
        return _transformer.transform(tree)
    except VisitError as e:
        # Unwrap DSLError / ValidationError raised inside the transformer so
        # callers see the original exception rather than Lark's VisitError.
        if isinstance(e.orig_exc, (DSLError, ValidationError)):
            raise e.orig_exc from None
        raise


def parse_policy(source: str) -> PolicyIR:
    """Parse a single-policy source string.

    Convenience wrapper around :func:`parse_policies` that expects exactly one
    policy definition.

    Raises:
        DSLError: If the source contains zero or more than one policy.
    """
    policies = parse_policies(source)
    if len(policies) != 1:
        raise DSLError(
            f"Expected exactly 1 policy, got {len(policies)}. "
            "Use parse_policies() for multi-policy sources."
        )
    return policies[0]


def parse_file(path: str | Path) -> List[PolicyIR]:
    """Parse a .moe policy file.

    Args:
        path: Path to the .moe file.

    Returns:
        List of PolicyIR objects.

    Raises:
        DSLError: If the file cannot be read or contains invalid syntax.
    """
    p = Path(path)
    if not p.exists():
        raise DSLError(f"Policy file not found: {p}")
    if not p.suffix == ".moe":
        raise DSLError(f"Expected a .moe file, got '{p.suffix}': {p}")
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise DSLError(f"Cannot read policy file {p}: {e}") from e
    return parse_policies(source)
