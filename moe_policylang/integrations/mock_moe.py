"""Mock MoE model for driving PolicyHooks without loading real weights.

Real HuggingFace MoE models (Mixtral, Qwen1.5-MoE, DeepSeek-V3) are too heavy
to load during unit tests.  The ``MockMoEModel`` simulates the *shape* of
their expert-dispatch pattern:

    for each token:
        for each layer (1..num_layers):
            select top-k experts from a deterministic/stochastic distribution
            invoke the policy hook

This lets us verify end-to-end behavior (hit rates, dispatch decisions,
DSL-vs-baseline equivalence) without GPUs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from moe_policylang.runtime.hooks import DispatchPlan, PolicyHook


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

def uniform_selector(
    num_experts: int, top_k: int = 2, seed: int = 0
) -> Callable[[int, int], List[int]]:
    """Return a selector that picks top_k experts uniformly at random."""
    rng = random.Random(seed)

    def select(token_idx: int, layer_idx: int) -> List[int]:
        return rng.sample(range(num_experts), top_k)

    return select


def skewed_selector(
    num_experts: int,
    top_k: int = 2,
    hot_fraction: float = 0.2,
    hot_weight: float = 0.8,
    seed: int = 0,
) -> Callable[[int, int], List[int]]:
    """Return a selector that biases toward a 'hot' expert subset.

    With ``hot_fraction=0.2`` and ``hot_weight=0.8``, 80 % of selections come
    from the first 20 % of expert IDs.  Good for testing caches under locality.
    """
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


def deterministic_trace_selector(
    trace: Sequence[Sequence[int]],
) -> Callable[[int, int], List[int]]:
    """Replay a fixed trace of expert selections (one list per (token, layer))."""
    trace = list(trace)

    def select(token_idx: int, layer_idx: int) -> List[int]:
        idx = token_idx * 10_000 + layer_idx  # arbitrary composite index
        return list(trace[idx % len(trace)])

    return select


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------

@dataclass
class MockMoEModel:
    """Simulates a MoE model's per-layer expert dispatch.

    Attributes:
        num_layers: How many MoE layers the model has.
        num_experts: Total experts per layer.
        top_k: Number of experts the router activates per token.
        selector: Callable ``(token_idx, layer_idx) -> list[expert_id]``.
            Defaults to a uniform random selector.
        score_fn: Optional callable ``(experts) -> list[float]`` to supply
            router scores to the hook (needed for SCORE eviction).
        expert_size_gb: Size passed to the scheduler's cost model.
    """

    num_layers: int = 24
    num_experts: int = 60
    top_k: int = 4
    selector: Optional[Callable[[int, int], List[int]]] = None
    score_fn: Optional[Callable[[Sequence[int]], List[float]]] = None
    expert_size_gb: float = 1.2

    # populated by run()
    plans: List[DispatchPlan] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.selector is None:
            self.selector = uniform_selector(self.num_experts, self.top_k, seed=0)

    def run(self, hook: PolicyHook, num_tokens: int) -> List[DispatchPlan]:
        """Simulate ``num_tokens`` forward passes and return all dispatch plans."""
        plans: List[DispatchPlan] = []
        for t in range(num_tokens):
            for layer in range(self.num_layers):
                experts = self.selector(t, layer)
                scores = self.score_fn(experts) if self.score_fn else None
                plan = hook.on_layer(
                    layer_idx=layer,
                    selected_experts=experts,
                    scores=scores,
                    expert_size_gb=self.expert_size_gb,
                )
                plans.append(plan)
        self.plans = plans
        return plans


def run_mock_inference(
    hook: PolicyHook,
    num_tokens: int = 32,
    num_layers: int = 24,
    num_experts: int = 60,
    top_k: int = 4,
    selector: Optional[Callable[[int, int], List[int]]] = None,
) -> List[DispatchPlan]:
    """One-line helper to exercise a hook against a mock MoE model."""
    model = MockMoEModel(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        selector=selector,
    )
    return model.run(hook, num_tokens=num_tokens)
