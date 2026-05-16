"""Tests for async expert weight transfers.

Tests the AsyncTransferManager in isolation (without a real GPU) using
mock tensors and events, plus integration tests with WeightPlacementManager.
"""

from __future__ import annotations

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for async transfer tests"
)


from moe_policylang.integrations.async_transfer import (
    AsyncTransferManager,
    AsyncTransferStats,
)


# ---------------------------------------------------------------------------
# AsyncTransferStats
# ---------------------------------------------------------------------------

class TestAsyncTransferStats:
    def test_defaults(self):
        s = AsyncTransferStats()
        assert s.async_transfers == 0
        assert s.overlap_ratio == 0.0

    def test_overlap_ratio(self):
        s = AsyncTransferStats(async_transfers=8, sync_transfers=2, prefetch_hits=6)
        assert s.overlap_ratio == pytest.approx(0.6)

    def test_to_dict(self):
        s = AsyncTransferStats(async_transfers=5, prefetch_hits=3)
        d = s.to_dict()
        assert "async_transfers" in d
        assert "overlap_ratio" in d


# ---------------------------------------------------------------------------
# AsyncTransferManager — real CUDA tests
# ---------------------------------------------------------------------------

class TestAsyncTransferManager:
    @pytest.fixture
    def atm(self):
        return AsyncTransferManager(gpu_device=torch.device("cuda:0"))

    def test_start_and_ensure_ready(self, atm):
        """Start a transfer, then ensure_ready returns GPU tensors."""
        cpu_tensor = torch.randn(64, 128, device="cpu")
        atm.start_transfer(0, 5, {"w": cpu_tensor}, cpu_tensor.numel() * 4)

        result = atm.ensure_ready(0, 5)
        assert result is not None
        assert "w" in result
        assert result["w"].device.type == "cuda"
        assert result["w"].shape == (64, 128)

    def test_ensure_ready_not_started(self, atm):
        """ensure_ready returns None for non-existent expert."""
        assert atm.ensure_ready(0, 99) is None

    def test_double_start_is_noop(self, atm):
        """Starting a transfer for the same expert twice doesn't duplicate."""
        t = torch.randn(32, 32, device="cpu")
        atm.start_transfer(0, 1, {"w": t}, 4096)
        atm.start_transfer(0, 1, {"w": t}, 4096)
        assert atm.stats.async_transfers == 1

    def test_ensure_ready_twice(self, atm):
        """Second ensure_ready returns from ready cache."""
        t = torch.randn(32, 32, device="cpu")
        atm.start_transfer(0, 1, {"w": t}, 4096)
        r1 = atm.ensure_ready(0, 1)
        r2 = atm.ensure_ready(0, 1)
        assert r1 is r2  # same dict
        assert atm.stats.prefetch_hits >= 1

    def test_evict(self, atm):
        """Evict removes expert from ready cache."""
        t = torch.randn(32, 32, device="cpu")
        atm.start_transfer(0, 1, {"w": t}, 4096)
        atm.ensure_ready(0, 1)
        atm.evict(0, 1)
        assert atm.ensure_ready(0, 1) is None

    def test_evict_expert_all_layers(self, atm):
        """Evict an expert from all layers."""
        t = torch.randn(8, 8, device="cpu")
        atm.start_transfer(0, 5, {"w": t}, 256)
        atm.start_transfer(1, 5, {"w": t}, 256)
        atm.start_transfer(2, 5, {"w": t}, 256)
        atm.evict_expert_all_layers(5)
        assert atm.ensure_ready(0, 5) is None
        assert atm.ensure_ready(1, 5) is None
        assert atm.ensure_ready(2, 5) is None

    def test_mark_ready(self, atm):
        """Sync-transferred experts can be registered via mark_ready."""
        gpu_t = torch.randn(8, 8, device="cuda:0")
        atm.mark_ready(0, 3, {"w": gpu_t})
        result = atm.ensure_ready(0, 3)
        assert result is not None
        assert result["w"] is gpu_t
        assert atm.stats.sync_transfers == 1

    def test_clear(self, atm):
        """Clear removes everything."""
        t = torch.randn(8, 8, device="cpu")
        atm.start_transfer(0, 1, {"w": t}, 256)
        atm.mark_ready(1, 2, {"w": torch.randn(8, 8, device="cuda:0")})
        atm.clear()
        assert atm.ensure_ready(0, 1) is None
        assert atm.ensure_ready(1, 2) is None

    def test_sync_all(self, atm):
        """sync_all moves everything from in-flight to ready."""
        t1 = torch.randn(8, 8, device="cpu")
        t2 = torch.randn(8, 8, device="cpu")
        atm.start_transfer(0, 1, {"w": t1}, 256)
        atm.start_transfer(0, 2, {"w": t2}, 256)
        atm.sync_all()
        # Both should be in ready now
        assert atm.ensure_ready(0, 1) is not None
        assert atm.ensure_ready(0, 2) is not None

    def test_multiple_tensors_per_expert(self, atm):
        """Experts with multiple weight tensors (gate_up + down)."""
        gate_up = torch.randn(64, 256, device="cpu")
        down = torch.randn(128, 64, device="cpu")
        size_bytes = (gate_up.numel() + down.numel()) * 4
        atm.start_transfer(0, 7, {"gate_up": gate_up, "down": down}, size_bytes)
        result = atm.ensure_ready(0, 7)
        assert result is not None
        assert result["gate_up"].shape == (64, 256)
        assert result["down"].shape == (128, 64)
        assert result["gate_up"].device.type == "cuda"
        assert result["down"].device.type == "cuda"

    def test_stats_tracking(self, atm):
        """Stats are accumulated correctly."""
        t = torch.randn(8, 8, device="cpu")
        atm.start_transfer(0, 1, {"w": t}, 1000)
        atm.start_transfer(0, 2, {"w": t}, 2000)
        assert atm.stats.async_transfers == 2
        assert atm.stats.bytes_transferred == 3000

        atm.ensure_ready(0, 1)  # first call: from in-flight → ready
        atm.ensure_ready(0, 1)  # second call: from ready cache (prefetch_hit)
        assert atm.stats.prefetch_hits >= 1


# ---------------------------------------------------------------------------
# Integration: WeightPlacementManager with async_transfers=True
# ---------------------------------------------------------------------------

class TestWeightPlacementManagerAsync:
    def test_atm_created_when_enabled(self):
        """async_transfers=True creates an AsyncTransferManager."""
        from moe_policylang import compile_policy, build_hook
        from moe_policylang.ir import CacheIR, EvictionPolicy, PolicyIR
        from moe_policylang.integrations.weight_placement import WeightPlacementManager

        ir = PolicyIR(
            name="test",
            cache=CacheIR(capacity=4, eviction=EvictionPolicy.LRU),
        )
        hook = build_hook(compile_policy(ir))

        # Use a minimal mock accessor
        class MockAccessor:
            num_layers = 4
            num_experts = 8
            def get_expert_params(self, l, e): return [torch.randn(8, 8)]
            def set_expert_device(self, l, e, d): pass
            def get_expert_device(self, l, e): return torch.device("cpu")
            def expert_forward(self, l, e, h): return h
            def get_router(self, l): return None
            def run_router(self, l, h): return None, None
            def expert_size_bytes(self, l=0, e=0): return 256

        mgr = WeightPlacementManager(
            hook, MockAccessor(), gpu_device=0, async_transfers=True
        )
        assert mgr._atm is not None
        assert mgr.async_transfers is True

    def test_atm_not_created_when_disabled(self):
        """async_transfers=False does not create an AsyncTransferManager."""
        from moe_policylang import compile_policy, build_hook
        from moe_policylang.ir import CacheIR, EvictionPolicy, PolicyIR
        from moe_policylang.integrations.weight_placement import WeightPlacementManager

        ir = PolicyIR(
            name="test",
            cache=CacheIR(capacity=4, eviction=EvictionPolicy.LRU),
        )
        hook = build_hook(compile_policy(ir))

        class MockAccessor:
            num_layers = 4
            num_experts = 8
            def get_expert_params(self, l, e): return [torch.randn(8, 8)]
            def set_expert_device(self, l, e, d): pass
            def get_expert_device(self, l, e): return torch.device("cpu")
            def expert_forward(self, l, e, h): return h
            def get_router(self, l): return None
            def run_router(self, l, h): return None, None
            def expert_size_bytes(self, l=0, e=0): return 256

        mgr = WeightPlacementManager(
            hook, MockAccessor(), gpu_device=0, async_transfers=False
        )
        assert mgr._atm is None

    def test_get_stats_includes_async(self):
        """get_stats includes async section when enabled."""
        from moe_policylang import compile_policy, build_hook
        from moe_policylang.ir import CacheIR, EvictionPolicy, PolicyIR
        from moe_policylang.integrations.weight_placement import WeightPlacementManager

        ir = PolicyIR(
            name="test",
            cache=CacheIR(capacity=4, eviction=EvictionPolicy.LRU),
        )
        hook = build_hook(compile_policy(ir))

        class MockAccessor:
            num_layers = 4
            num_experts = 8
            def get_expert_params(self, l, e): return [torch.randn(8, 8)]
            def set_expert_device(self, l, e, d): pass
            def get_expert_device(self, l, e): return torch.device("cpu")
            def expert_forward(self, l, e, h): return h
            def get_router(self, l): return None
            def run_router(self, l, h): return None, None
            def expert_size_bytes(self, l=0, e=0): return 256

        mgr = WeightPlacementManager(
            hook, MockAccessor(), gpu_device=0, async_transfers=True
        )
        stats = mgr.get_stats()
        assert "async" in stats
        assert "overlap_ratio" in stats["async"]
