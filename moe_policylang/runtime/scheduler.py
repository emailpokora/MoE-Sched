"""CPU/GPU execution scheduler for MoE-Sched runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExecutionDevice(Enum):
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class SchedulerStats:
    gpu_executions: int = 0
    cpu_executions: int = 0
    transfers: int = 0


class GPUOnlyScheduler:
    """All experts execute on GPU; cache misses trigger synchronous load."""

    def __init__(self) -> None:
        self.stats = SchedulerStats()

    def decide(self, expert_id: int, is_cached: bool, expert_size_gb: float = 1.2) -> ExecutionDevice:
        if is_cached:
            self.stats.gpu_executions += 1
        else:
            self.stats.transfers += 1
            self.stats.gpu_executions += 1
        return ExecutionDevice.GPU


class CPUFallbackScheduler:
    """Cached experts on GPU, cache misses on CPU."""

    def __init__(self) -> None:
        self.stats = SchedulerStats()

    def decide(self, expert_id: int, is_cached: bool, expert_size_gb: float = 1.2) -> ExecutionDevice:
        if is_cached:
            self.stats.gpu_executions += 1
            return ExecutionDevice.GPU
        else:
            self.stats.cpu_executions += 1
            return ExecutionDevice.CPU


class HybridScheduler:
    """Cost-model-based: choose GPU or CPU based on transfer vs compute time."""

    def __init__(
        self,
        cpu_threshold_ms: float = 50.0,
        pcie_bandwidth_gbs: float = 25.0,
    ):
        self.cpu_threshold_ms = cpu_threshold_ms
        self.pcie_bandwidth_gbs = pcie_bandwidth_gbs
        self.stats = SchedulerStats()

    def _estimate_transfer_ms(self, expert_size_gb: float) -> float:
        return (expert_size_gb / self.pcie_bandwidth_gbs) * 1000.0

    def decide(self, expert_id: int, is_cached: bool, expert_size_gb: float = 1.2) -> ExecutionDevice:
        if is_cached:
            self.stats.gpu_executions += 1
            return ExecutionDevice.GPU

        transfer_ms = self._estimate_transfer_ms(expert_size_gb)
        if transfer_ms > self.cpu_threshold_ms:
            self.stats.cpu_executions += 1
            return ExecutionDevice.CPU
        else:
            self.stats.transfers += 1
            self.stats.gpu_executions += 1
            return ExecutionDevice.GPU
