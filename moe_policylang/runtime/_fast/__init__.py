"""Cython-accelerated fast paths for MoE-PolicyLang runtime components.

Phase 2 of the conference paper plan. These modules provide drop-in
replacements for the pure-Python cache and scheduler implementations
with lower dispatch latency.

When built (via `python setup.py build_ext --inplace`), the fast paths
are automatically used by the compiler. If not built, the system falls
back to pure-Python implementations transparently.
"""

FAST_PATH_AVAILABLE = False
FAST_HOOK_AVAILABLE = False

try:
    from moe_policylang.runtime._fast._cache import LRUCacheFast, LFUCacheFast
    from moe_policylang.runtime._fast._scheduler import (
        GPUOnlySchedulerFast,
        CPUFallbackSchedulerFast,
        HybridSchedulerFast,
    )
    FAST_PATH_AVAILABLE = True
except ImportError:
    # Cython modules not built — pure-Python fallback is used automatically.
    pass

try:
    from moe_policylang.runtime._fast._hooks import FastPolicyHook
    FAST_HOOK_AVAILABLE = True
except ImportError:
    # Full fast-path hook not built — falls back to Python PolicyHook.
    pass
