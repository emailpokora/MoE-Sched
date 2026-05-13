# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated cache implementations.

Phase 2: Drop-in replacements for LRUCache and LFUCache with C-typed
operations for minimal dispatch overhead.  These classes maintain the same
API as their Python counterparts so they are transparent drop-ins.
"""

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    loads: int = 0

    @property
    def total(self):
        return self.hits + self.misses

    @property
    def hit_rate(self):
        return self.hits / self.total if self.total > 0 else 0.0


cdef class LRUCacheFast:
    """Cython LRU cache — drop-in replacement for runtime.cache.LRUCache."""
    cdef public int capacity
    cdef public set pinned
    cdef public object stats
    cdef public object _cache  # OrderedDict for LRU ordering
    cdef public object on_evict  # Optional callback(expert_id)

    def __init__(self, int capacity, list pin_experts=None, object on_evict=None):
        self.capacity = capacity
        self.pinned = set(pin_experts or [])
        self._cache = OrderedDict()
        self.stats = CacheStats()
        self.on_evict = on_evict
        for eid in self.pinned:
            self._cache[eid] = True

    cpdef bint access(self, int expert_id):
        """Returns True on hit, False on miss (with eviction if full)."""
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self.stats.hits += 1
            return True

        self.stats.misses += 1
        while len(self._cache) >= self.capacity:
            if self._evict_one() < 0:
                break
        if len(self._cache) < self.capacity:
            self._cache[expert_id] = True
            self.stats.loads += 1
        return False

    cpdef int _evict_one(self):
        """Evict the LRU (oldest) non-pinned entry. Returns evicted id or -1."""
        cdef int key
        for key in list(self._cache.keys()):
            if key not in self.pinned:
                del self._cache[key]
                self.stats.evictions += 1
                if self.on_evict is not None:
                    self.on_evict(key)
                return key
        return -1

    @property
    def cache(self):
        """Expose internal OrderedDict for introspection compatibility."""
        return self._cache

    cpdef bint is_cached(self, int expert_id):
        return expert_id in self._cache

    def prefetch_insert(self, int expert_id):
        """Insert without charging a miss (for prefetch warming)."""
        if expert_id not in self._cache:
            while len(self._cache) >= self.capacity:
                if self._evict_one() < 0:
                    break
            if len(self._cache) < self.capacity:
                self._cache[expert_id] = True

    @property
    def size(self):
        return len(self._cache)


cdef class LFUCacheFast:
    """Cython LFU cache — drop-in replacement for runtime.cache.LFUCache."""
    cdef public int capacity
    cdef public double decay
    cdef public set pinned
    cdef public object stats
    cdef dict _freq
    cdef int _access_count
    cdef public object on_evict  # Optional callback(expert_id)

    def __init__(self, int capacity, list pin_experts=None, double decay=1.0, object on_evict=None):
        self.capacity = capacity
        self.pinned = set(pin_experts or [])
        self._freq = {}
        self.stats = CacheStats()
        self.decay = decay
        self._access_count = 0
        self.on_evict = on_evict
        for eid in self.pinned:
            self._freq[eid] = float("inf")

    cpdef bint access(self, int expert_id):
        self._access_count += 1

        if self.decay < 1.0 and self._access_count % 100 == 0:
            self._apply_decay()

        if expert_id in self._freq:
            self._freq[expert_id] = self._freq[expert_id] + 1
            self.stats.hits += 1
            return True

        self.stats.misses += 1
        while len(self._freq) >= self.capacity:
            if self._evict_one() < 0:
                break
        if len(self._freq) < self.capacity:
            self._freq[expert_id] = 1.0
            self.stats.loads += 1
        return False

    cpdef int _evict_one(self):
        cdef int victim = -1
        cdef double min_freq = 1e30
        cdef int k
        cdef double v
        for k, v in self._freq.items():
            if k not in self.pinned and v < min_freq:
                min_freq = v
                victim = k
        if victim >= 0:
            del self._freq[victim]
            self.stats.evictions += 1
            if self.on_evict is not None:
                self.on_evict(victim)
        return victim

    @property
    def freq(self):
        """Expose frequency dict for introspection compatibility."""
        return self._freq

    cdef void _apply_decay(self):
        cdef int k
        cdef double v
        for k in list(self._freq.keys()):
            if k not in self.pinned:
                self._freq[k] = self._freq[k] * self.decay

    cpdef bint is_cached(self, int expert_id):
        return expert_id in self._freq

    def prefetch_insert(self, int expert_id):
        if expert_id not in self._freq:
            while len(self._freq) >= self.capacity:
                if self._evict_one() < 0:
                    break
            if len(self._freq) < self.capacity:
                self._freq[expert_id] = 1.0

    @property
    def size(self):
        return len(self._freq)
