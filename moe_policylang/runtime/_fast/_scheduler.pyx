# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated scheduler implementations.

Drop-in replacements for GPUOnlyScheduler, CPUFallbackScheduler,
and HybridScheduler.  Return ExecutionDevice values matching the Python API.
"""

from dataclasses import dataclass
from moe_policylang.runtime.scheduler import ExecutionDevice


@dataclass
class SchedulerStats:
    gpu_executions: int = 0
    cpu_executions: int = 0
    transfers: int = 0


cdef class GPUOnlySchedulerFast:
    """All experts on GPU — cache misses trigger synchronous load."""
    cdef public object stats

    def __init__(self):
        self.stats = SchedulerStats()

    cpdef object decide(self, int expert_id, bint is_cached, double expert_size_gb=1.2):
        if is_cached:
            self.stats.gpu_executions += 1
        else:
            self.stats.transfers += 1
            self.stats.gpu_executions += 1
        return ExecutionDevice.GPU


cdef class CPUFallbackSchedulerFast:
    """Cached on GPU, misses fall back to CPU."""
    cdef public object stats

    def __init__(self):
        self.stats = SchedulerStats()

    cpdef object decide(self, int expert_id, bint is_cached, double expert_size_gb=1.2):
        if is_cached:
            self.stats.gpu_executions += 1
            return ExecutionDevice.GPU
        else:
            self.stats.cpu_executions += 1
            return ExecutionDevice.CPU


cdef class HybridSchedulerFast:
    """Cost-model scheduler: GPU or CPU based on transfer time."""
    cdef public double cpu_threshold_ms
    cdef public double pcie_bandwidth_gbs
    cdef public object stats

    def __init__(self, double cpu_threshold_ms=50.0, double pcie_bandwidth_gbs=25.0):
        self.cpu_threshold_ms = cpu_threshold_ms
        self.pcie_bandwidth_gbs = pcie_bandwidth_gbs
        self.stats = SchedulerStats()

    cpdef object decide(self, int expert_id, bint is_cached, double expert_size_gb=1.2):
        if is_cached:
            self.stats.gpu_executions += 1
            return ExecutionDevice.GPU

        cdef double transfer_ms = (expert_size_gb / self.pcie_bandwidth_gbs) * 1000.0
        if transfer_ms > self.cpu_threshold_ms:
            self.stats.cpu_executions += 1
            return ExecutionDevice.CPU
        else:
            self.stats.transfers += 1
            self.stats.gpu_executions += 1
            return ExecutionDevice.GPU
