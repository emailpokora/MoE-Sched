"""Integration adapters that connect MoE-Sched hooks to inference engines."""

from moe_sched.integrations.mock_moe import MockMoEModel, run_mock_inference
from moe_sched.integrations.weight_placement import (
    ExpertAccessor,
    WeightPlacementManager,
)
from moe_sched.integrations.accessors import auto_accessor


def auto_manage(model, hook, gpu_device=0):
    """One-line setup: auto-detect architecture and create a managed offloader.

    Usage:
        mgr = auto_manage(model, hook)
        mgr.attach()  # hooks into model, manages expert placement
        output = model.generate(...)  # use model normally
    """
    accessor = auto_accessor(model)
    return WeightPlacementManager(hook, accessor, gpu_device=gpu_device)


def attach(model, policy, gpu_device=0):
    """Attach a MoE-Sched policy to a HuggingFace MoE model.

    This is the primary user-facing API. After calling this, use the model
    normally — MoE-Sched manages expert placement based on the policy.

    ``policy`` can be:
      - A DSL string (parsed and compiled automatically)
      - A PolicyIR object (compiled automatically)

    Usage with DSL string::

        import moe_sched

        mgr = moe_sched.attach(model, '''
            policy my_lfu {
                cache { capacity = 32  eviction = lfu  frequency_decay = 0.9 }
                prefetch { strategy = history  budget = 4 }
            }
        ''')
        output = model.generate(...)
        print(mgr.get_stats())

    Returns:
        WeightPlacementManager with hooks installed.
    """
    from moe_sched.compiler import compile_policy
    from moe_sched.ir import PolicyIR
    from moe_sched.runtime.hooks import build_hook

    if isinstance(policy, str):
        from moe_sched.parser import parse_policy
        policy = parse_policy(policy)
    elif not isinstance(policy, PolicyIR):
        raise TypeError(
            f"policy must be a DSL string or PolicyIR, got {type(policy).__name__}"
        )

    compiled = compile_policy(policy)
    hook = build_hook(compiled)
    accessor = auto_accessor(model)
    mgr = WeightPlacementManager(hook, accessor, gpu_device=gpu_device)
    mgr.attach()
    return mgr


__all__ = [
    "MockMoEModel",
    "run_mock_inference",
    "ExpertAccessor",
    "WeightPlacementManager",
    "auto_accessor",
    "auto_manage",
    "attach",
]
