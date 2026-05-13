"""Eviction triggers: orthogonal-to-base-policy cache-pressure rules.

The base cache algorithms (LRU, LFU, Score, FreqThreshold) evict *only* when
the cache is full and a new expert needs to be inserted.  Several systems
(MoE-Infinity, HybriMoE, FineMoE) also evict **proactively** in response to:

  * **Memory pressure** — estimated GPU memory use exceeds a threshold,
    drop cold experts down to a headroom target.
  * **TTL / staleness** — evict any expert that has not been accessed for
    a fixed number of steps.

These are expressed in MoE-PolicyLang as *triggers* that wrap a base cache.
They do not change the cache's insertion semantics; they only force
additional evictions between accesses.

The proposal requires both triggers.  Keeping them out of the base cache
classes preserves the orthogonality between *which* expert to evict (cache
policy) and *when* to evict (trigger).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass
class TriggerStats:
    fired: int = 0
    evicted: int = 0


class _CacheProtocol(Protocol):
    """Minimal surface we rely on from a cache implementation."""

    capacity: int
    pinned: set

    def _evict_one(self) -> Optional[int]: ...


# ---------------------------------------------------------------------------
# Memory-pressure trigger
# ---------------------------------------------------------------------------

class MemoryPressureTrigger:
    """Evict down to a headroom target when estimated GPU usage is high.

    The cache tracks experts as opaque IDs; it doesn't know their byte size.
    We use a user-supplied ``expert_size_gb`` and a configured GPU budget to
    estimate fill.  When ``current_gb / budget_gb >= threshold``, evict the
    coldest experts until ``current_gb / budget_gb <= headroom``.
    """

    def __init__(
        self,
        budget_gb: float,
        threshold: float = 0.9,
        headroom: float = 0.7,
        expert_size_gb: float = 1.2,
    ):
        if not 0 < headroom <= threshold <= 1:
            raise ValueError(
                "MemoryPressureTrigger requires 0 < headroom <= threshold <= 1"
            )
        self.budget_gb = budget_gb
        self.threshold = threshold
        self.headroom = headroom
        self.expert_size_gb = expert_size_gb
        self.stats = TriggerStats()

    def _usage_frac(self, cache_size: int) -> float:
        if self.budget_gb <= 0:
            return 0.0
        return (cache_size * self.expert_size_gb) / self.budget_gb

    def after_access(self, cache: _CacheProtocol) -> List[int]:
        """Called after each cache access.  Returns list of evicted IDs."""
        frac = self._usage_frac(self._size(cache))
        if frac < self.threshold:
            return []

        self.stats.fired += 1
        evicted: List[int] = []
        # Evict until we reach the headroom fraction or we can't evict more.
        while self._usage_frac(self._size(cache)) > self.headroom:
            victim = cache._evict_one()
            if victim is None:
                break
            evicted.append(victim)
            self.stats.evicted += 1
        return evicted

    @staticmethod
    def _size(cache: _CacheProtocol) -> int:
        return getattr(cache, "size", 0) or len(
            getattr(cache, "cache", getattr(cache, "freq", getattr(cache, "scores", {})))
        )


# ---------------------------------------------------------------------------
# TTL trigger
# ---------------------------------------------------------------------------

class TTLTrigger:
    """Evict any non-pinned expert unused for ``ttl`` accesses.

    We track an access counter per expert; on every access the trigger
    checks the oldest entry and evicts it if it has gone stale.  A small
    amortised cost keeps this O(1) on average.
    """

    def __init__(self, ttl: int = 200):
        if ttl < 1:
            raise ValueError("ttl must be >= 1")
        self.ttl = ttl
        self.stats = TriggerStats()
        self._last_seen: dict[int, int] = {}
        self._step = 0

    def on_access(self, expert_id: int) -> None:
        """Caller must invoke this *before* or *after* each cache.access()."""
        self._step += 1
        self._last_seen[expert_id] = self._step

    def after_access(self, cache: _CacheProtocol) -> List[int]:
        """Called after each cache access.  Returns list of evicted IDs."""
        evicted: List[int] = []
        cutoff = self._step - self.ttl
        stale = [
            eid for eid, last in self._last_seen.items()
            if last <= cutoff and eid not in cache.pinned
        ]
        if not stale:
            return []

        self.stats.fired += 1
        for eid in stale:
            if self._evict_specific(cache, eid):
                evicted.append(eid)
                self.stats.evicted += 1
            self._last_seen.pop(eid, None)
        return evicted

    @staticmethod
    def _evict_specific(cache: _CacheProtocol, expert_id: int) -> bool:
        """Remove ``expert_id`` from the cache if present.  Uses the cache's
        internal storage — works for the four cache types shipped with
        MoE-PolicyLang (LRU ordered dict, LFU/Score freq/score dicts, FreqThreshold set)."""
        for attr in ("cache", "freq", "scores"):
            store = getattr(cache, attr, None)
            if isinstance(store, dict) and expert_id in store:
                del store[expert_id]
                cache.stats.evictions += 1
                return True
            if isinstance(store, set) and expert_id in store:
                store.discard(expert_id)
                cache.stats.evictions += 1
                return True
        return False


# ---------------------------------------------------------------------------
# Trigger set
# ---------------------------------------------------------------------------

@dataclass
class TriggerSet:
    """Bundle of triggers fired together after every cache access."""

    memory_pressure: Optional[MemoryPressureTrigger] = None
    ttl: Optional[TTLTrigger] = None

    # Populated as triggers fire; callers may read this to gather stats.
    evicted: List[int] = field(default_factory=list)

    def on_access(self, expert_id: int) -> None:
        if self.ttl is not None:
            self.ttl.on_access(expert_id)

    def after_access(self, cache: _CacheProtocol) -> List[int]:
        fired: List[int] = []
        if self.memory_pressure is not None:
            fired.extend(self.memory_pressure.after_access(cache))
        if self.ttl is not None:
            fired.extend(self.ttl.after_access(cache))
        self.evicted.extend(fired)
        return fired

    @property
    def active(self) -> bool:
        return self.memory_pressure is not None or self.ttl is not None
