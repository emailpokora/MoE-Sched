"""Integration adapters that connect MoE-PolicyLang hooks to inference engines."""

from moe_policylang.integrations.mock_moe import MockMoEModel, run_mock_inference
from moe_policylang.integrations.weight_placement import (
    ExpertAccessor,
    WeightPlacementManager,
)
from moe_policylang.integrations.accessors import auto_accessor
from moe_policylang.integrations.loading import load_moe_model


def auto_manage(model, hook, gpu_device=0, async_transfers=False):
    """One-line setup: auto-detect architecture and create a managed offloader.

    Usage:
        mgr = auto_manage(model, hook)
        mgr.attach()  # hooks into model, manages expert placement
        output = model.generate(...)  # use model normally
    """
    accessor = auto_accessor(model)
    return WeightPlacementManager(hook, accessor, gpu_device=gpu_device, async_transfers=async_transfers)


def attach(model, policy, gpu_device=0, async_transfers=False):
    """Attach a MoE-PolicyLang policy to a HuggingFace MoE model.

    This is the primary user-facing API. After calling this, use the model
    normally — MoE-PolicyLang manages expert placement based on the policy.

    ``policy`` can be:
      - A DSL string (parsed and compiled automatically)
      - A PolicyIR object (compiled automatically)

    Usage with DSL string::

        import moe_policylang

        mgr = moe_policylang.attach(model, '''
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
    from moe_policylang.compiler import compile_policy
    from moe_policylang.ir import PolicyIR
    from moe_policylang.runtime.hooks import build_hook

    if isinstance(policy, str):
        from moe_policylang.parser import parse_policy
        policy = parse_policy(policy)
    elif not isinstance(policy, PolicyIR):
        raise TypeError(
            f"policy must be a DSL string or PolicyIR, got {type(policy).__name__}"
        )

    compiled = compile_policy(policy)
    hook = build_hook(compiled)
    accessor = auto_accessor(model)
    mgr = WeightPlacementManager(hook, accessor, gpu_device=gpu_device, async_transfers=async_transfers)
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
    "load_moe_model",
]
