"""Expert cache implementations for MoE-PolicyLang runtime."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

# Type alias for eviction callbacks: receives the evicted expert_id
EvictCallback = Optional[Callable[[int], None]]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    loads: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

class LRUCache:
    """Least-Recently-Used expert cache."""

    def __init__(self, capacity: int, pin_experts: list[int] | None = None,
                 on_evict: EvictCallback = None):
        self.capacity = capacity
        self.pinned: set[int] = set(pin_experts or [])
        self.cache: OrderedDict[int, bool] = OrderedDict()
        self.stats = CacheStats()
        self.on_evict = on_evict

        for eid in self.pinned:
            self.cache[eid] = True

    def access(self, expert_id: int) -> bool:
        """Returns True on cache hit, False on miss."""
        if expert_id in self.cache:
            self.cache.move_to_end(expert_id)
            self.stats.hits += 1
            return True

        self.stats.misses += 1
        while len(self.cache) >= self.capacity:
            if self._evict_one() is None:
                break
        if len(self.cache) < self.capacity:
            self.cache[expert_id] = True
            self.stats.loads += 1
        return False

    def _evict_one(self) -> int | None:
        for key in list(self.cache.keys()):
            if key not in self.pinned:
                del self.cache[key]
                self.stats.evictions += 1
                if self.on_evict is not None:
                    self.on_evict(key)
                return key
        return None

    def is_cached(self, expert_id: int) -> bool:
        return expert_id in self.cache

    @property
    def size(self) -> int:
        return len(self.cache)


# ---------------------------------------------------------------------------
# LFU Cache
# ---------------------------------------------------------------------------

class LFUCache:
    """Least-Frequently-Used expert cache with optional decay."""

    def __init__(
        self,
        capacity: int,
        pin_experts: list[int] | None = None,
        decay: float = 1.0,
        on_evict: EvictCallback = None,
    ):
        self.capacity = capacity
        self.pinned: set[int] = set(pin_experts or [])
        self.freq: dict[int, float] = {}
        self.stats = CacheStats()
        self.decay = decay
        self._access_count = 0
        self.on_evict = on_evict

        for eid in self.pinned:
            self.freq[eid] = float("inf")

    def access(self, expert_id: int) -> bool:
        self._access_count += 1

        if self.decay < 1.0 and self._access_count % 100 == 0:
            self._apply_decay()

        if expert_id in self.freq:
            self.freq[expert_id] += 1
            self.stats.hits += 1
            return True

        self.stats.misses += 1
        while len(self.freq) >= self.capacity:
            if self._evict_one() is None:
                break
        if len(self.freq) < self.capacity:
            self.freq[expert_id] = 1
            self.stats.loads += 1
        return False

    def _evict_one(self) -> int | None:
        candidates = {k: v for k, v in self.freq.items() if k not in self.pinned}
        if not candidates:
            return None
        victim = min(candidates, key=candidates.get)
        del self.freq[victim]
        self.stats.evictions += 1
        if self.on_evict is not None:
            self.on_evict(victim)
        return victim

    def _apply_decay(self) -> None:
        for k in self.freq:
            if k not in self.pinned:
                self.freq[k] *= self.decay

    def is_cached(self, expert_id: int) -> bool:
        return expert_id in self.freq

    @property
    def size(self) -> int:
        return len(self.freq)


# ---------------------------------------------------------------------------
# Score-Based Cache
# ---------------------------------------------------------------------------

class ScoreCache:
    """Evicts expert with lowest exponential-moving-average router score."""

    def __init__(
        self,
        capacity: int,
        pin_experts: list[int] | None = None,
        ema_alpha: float = 0.3,
        on_evict: EvictCallback = None,
    ):
        self.capacity = capacity
        self.pinned: set[int] = set(pin_experts or [])
        self.scores: dict[int, float] = {}
        self.stats = CacheStats()
        self.ema_alpha = ema_alpha
        self.on_evict = on_evict

        for eid in self.pinned:
            self.scores[eid] = float("inf")

    def access(self, expert_id: int, score: float = 1.0) -> bool:
        if expert_id in self.scores:
            old = self.scores[expert_id]
            if expert_id not in self.pinned:
                self.scores[expert_id] = (
                    self.ema_alpha * score + (1 - self.ema_alpha) * old
                )
            self.stats.hits += 1
            return True

        self.stats.misses += 1
        while len(self.scores) >= self.capacity:
            if self._evict_one() is None:
                break
        if len(self.scores) < self.capacity:
            self.scores[expert_id] = score
            self.stats.loads += 1
        return False

    def _evict_one(self) -> int | None:
        candidates = {k: v for k, v in self.scores.items() if k not in self.pinned}
        if not candidates:
            return None
        victim = min(candidates, key=candidates.get)
        del self.scores[victim]
        self.stats.evictions += 1
        if self.on_evict is not None:
            self.on_evict(victim)
        return victim

    def is_cached(self, expert_id: int) -> bool:
        return expert_id in self.scores

    @property
    def size(self) -> int:
        return len(self.scores)


# ---------------------------------------------------------------------------
# Frequency-Threshold Cache
# ---------------------------------------------------------------------------

class FreqThresholdCache:
    """Keep experts whose activation frequency exceeds a threshold."""

    def __init__(
        self,
        capacity: int,
        threshold: float = 0.05,
        window: int = 100,
        pin_experts: list[int] | None = None,
        on_evict: EvictCallback = None,
    ):
        self.capacity = capacity
        self.threshold = threshold
        self.window = window
        self.pinned: set[int] = set(pin_experts or [])
        self.history: list[int] = []
        self.cache: set[int] = set(self.pinned)
        self.stats = CacheStats()
        self.on_evict = on_evict

    def access(self, expert_id: int) -> bool:
        self.history.append(expert_id)
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

        hit = expert_id in self.cache
        if hit:
            self.stats.hits += 1
        else:
            self.stats.misses += 1

        self._refresh_cache()

        if not hit and expert_id in self.cache:
            self.stats.loads += 1

        return hit

    def _refresh_cache(self) -> None:
        if not self.history:
            return
        freq: dict[int, int] = {}
        for eid in self.history:
            freq[eid] = freq.get(eid, 0) + 1
        total = len(self.history)

        qualified = {
            eid for eid, cnt in freq.items() if cnt / total >= self.threshold
        }
        new_cache = self.pinned | qualified

        evicted = self.cache - new_cache - self.pinned
        self.stats.evictions += len(evicted)
        if self.on_evict is not None:
            for eid in evicted:
                self.on_evict(eid)

        while len(new_cache) > self.capacity:
            removable = new_cache - self.pinned
            if not removable:
                break
            worst = min(removable, key=lambda e: freq.get(e, 0))
            new_cache.discard(worst)
            self.stats.evictions += 1
            if self.on_evict is not None:
                self.on_evict(worst)

        self.cache = new_cache

    def is_cached(self, expert_id: int) -> bool:
        return expert_id in self.cache

    def _evict_one(self) -> int | None:
        """Evict an arbitrary non-pinned expert.

        Required by the memory-pressure trigger, which forces eviction
        regardless of threshold/frequency state.  Picks the expert with the
        lowest activation count in the current history window; ties broken
        arbitrarily.
        """
        counts: dict[int, int] = {}
        for eid in self.history:
            if eid in self.cache and eid not in self.pinned:
                counts[eid] = counts.get(eid, 0) + 1
        for eid in self.cache - self.pinned:
            counts.setdefault(eid, 0)
        if not counts:
            return None
        victim = min(counts, key=counts.get)
        self.cache.discard(victim)
        self.stats.evictions += 1
        if self.on_evict is not None:
            self.on_evict(victim)
        return victim

    @property
    def size(self) -> int:
        return len(self.cache)


# ---------------------------------------------------------------------------
# Composition: Fallback Cache
# ---------------------------------------------------------------------------

class FallbackCache:
    """Layered cache composition: primary | fallback(secondary).

    On access, the primary cache is consulted first.  If the primary
    misses *and* the secondary would have hit, the expert is promoted
    from the secondary to the primary.  Evictions from the primary
    demote into the secondary, creating a two-tier caching hierarchy.

    Both caches share a combined capacity budget.
    """

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary
        self.stats = CacheStats()

    def access(self, expert_id: int, score: float = 1.0) -> bool:
        """Returns True on hit in either tier."""
        # Try primary first
        if self.primary.is_cached(expert_id):
            self.primary.access(expert_id)
            self.stats.hits += 1
            return True

        # Check secondary
        if self.secondary.is_cached(expert_id):
            # Promote: access secondary (hit), then insert into primary
            self.secondary.access(expert_id)
            self.primary.access(expert_id)
            self.stats.hits += 1
            return True

        # Full miss — insert into primary
        self.stats.misses += 1
        self.primary.access(expert_id)
        return False

    def is_cached(self, expert_id: int) -> bool:
        return self.primary.is_cached(expert_id) or self.secondary.is_cached(expert_id)

    @property
    def size(self) -> int:
        return self.primary.size + self.secondary.size

    @property
    def capacity(self):
        return self.primary.capacity + self.secondary.capacity
