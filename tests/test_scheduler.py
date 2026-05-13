"""Tests for CPU/GPU execution schedulers."""

import pytest

from moe_policylang.runtime.scheduler import (
    CPUFallbackScheduler,
    ExecutionDevice,
    GPUOnlyScheduler,
    HybridScheduler,
)


class TestGPUOnlyScheduler:
    def test_cached_goes_to_gpu(self):
        s = GPUOnlyScheduler()
        assert s.decide(0, is_cached=True) is ExecutionDevice.GPU
        assert s.stats.gpu_executions == 1
        assert s.stats.transfers == 0

    def test_uncached_still_goes_to_gpu(self):
        s = GPUOnlyScheduler()
        assert s.decide(0, is_cached=False) is ExecutionDevice.GPU
        assert s.stats.transfers == 1
        assert s.stats.gpu_executions == 1

    def test_no_cpu_executions(self):
        s = GPUOnlyScheduler()
        for i in range(10):
            s.decide(i, is_cached=(i % 2 == 0))
        assert s.stats.cpu_executions == 0


class TestCPUFallbackScheduler:
    def test_cached_goes_to_gpu(self):
        s = CPUFallbackScheduler()
        assert s.decide(0, is_cached=True) is ExecutionDevice.GPU
        assert s.stats.gpu_executions == 1

    def test_uncached_goes_to_cpu(self):
        s = CPUFallbackScheduler()
        assert s.decide(0, is_cached=False) is ExecutionDevice.CPU
        assert s.stats.cpu_executions == 1
        assert s.stats.gpu_executions == 0

    def test_no_transfers(self):
        s = CPUFallbackScheduler()
        for i in range(10):
            s.decide(i, is_cached=False)
        assert s.stats.transfers == 0


class TestHybridScheduler:
    def test_cached_goes_to_gpu(self):
        s = HybridScheduler(cpu_threshold_ms=50.0)
        assert s.decide(0, is_cached=True) is ExecutionDevice.GPU

    def test_small_expert_transfers_to_gpu(self):
        # 0.5 GB / 25 GB/s = 20ms < 50ms threshold → GPU
        s = HybridScheduler(cpu_threshold_ms=50.0, pcie_bandwidth_gbs=25.0)
        assert s.decide(0, is_cached=False, expert_size_gb=0.5) is ExecutionDevice.GPU
        assert s.stats.transfers == 1

    def test_large_expert_goes_to_cpu(self):
        # 2.0 GB / 25 GB/s = 80ms > 50ms threshold → CPU
        s = HybridScheduler(cpu_threshold_ms=50.0, pcie_bandwidth_gbs=25.0)
        assert s.decide(0, is_cached=False, expert_size_gb=2.0) is ExecutionDevice.CPU
        assert s.stats.cpu_executions == 1

    def test_threshold_boundary(self):
        # Exactly at threshold: 1.25 GB / 25 GB/s = 50ms == threshold → CPU (strictly >)
        s = HybridScheduler(cpu_threshold_ms=50.0, pcie_bandwidth_gbs=25.0)
        device = s.decide(0, is_cached=False, expert_size_gb=1.25)
        # 50ms is not > 50ms, so should go to GPU
        assert device is ExecutionDevice.GPU

    def test_low_threshold_sends_everything_to_cpu(self):
        s = HybridScheduler(cpu_threshold_ms=0.001)
        assert s.decide(0, is_cached=False, expert_size_gb=0.001) is ExecutionDevice.CPU

    def test_high_threshold_sends_everything_to_gpu(self):
        s = HybridScheduler(cpu_threshold_ms=9999.0)
        assert s.decide(0, is_cached=False, expert_size_gb=10.0) is ExecutionDevice.GPU

    def test_stats_accumulate(self):
        s = HybridScheduler(cpu_threshold_ms=50.0, pcie_bandwidth_gbs=25.0)
        s.decide(0, is_cached=True)                          # GPU
        s.decide(1, is_cached=False, expert_size_gb=0.5)     # GPU (transfer)
        s.decide(2, is_cached=False, expert_size_gb=2.0)     # CPU
        assert s.stats.gpu_executions == 2
        assert s.stats.cpu_executions == 1
        assert s.stats.transfers == 1
