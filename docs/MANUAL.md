# MoE-Sched Language Manual

A complete guide to the MoE-Sched domain-specific language for declaring
expert caching, prefetching, and scheduling policies for Mixture-of-Experts
inference.

---

## Table of Contents

1. [What Is MoE-Sched?](#1-what-is-moe-sched)
2. [Quick Start](#2-quick-start)
3. [Two Frontends](#3-two-frontends)
   - [.moe File Syntax](#31-moe-file-syntax)
   - [Python eDSL](#32-python-edsl)
   - [Fluent Builder](#33-fluent-builder)
4. [Policy Axes](#4-policy-axes)
   - [Cache](#41-cache)
   - [Prefetch](#42-prefetch)
   - [Schedule](#43-schedule)
   - [Monitor](#44-monitor)
   - [Eviction Triggers](#45-eviction-triggers)
5. [Adaptive Policies](#5-adaptive-policies)
6. [Per-Layer Adaptive Caching](#6-per-layer-adaptive-caching)
7. [Compilation and Runtime](#7-compilation-and-runtime)
8. [Autotuner](#8-autotuner)
9. [Validation Rules](#9-validation-rules)
10. [Working with Traces](#10-working-with-traces)
11. [Examples](#11-examples)
    - [Reproducing Published Systems](#111-reproducing-published-systems)
12. [API Reference](#12-api-reference)
13. [Command-Line Interface](#13-command-line-interface)

---

## 1. What Is MoE-Sched?

### The Problem

Large Mixture-of-Experts models like Mixtral-8x7B contain tens of billions
of parameters, but only activate a fraction of them per token.  Mixtral, for
instance, has ~45B total parameters across 8 experts per layer, yet the
router selects only 2 experts per token — activating ~13B parameters.

The full model rarely fits in GPU memory.  During inference, expert weight
matrices must be **swapped between CPU and GPU on-the-fly**: experts the
router needs are transferred to the GPU, computed, and eventually evicted to
make room for others.  This is the same problem an operating system faces
with virtual memory and page replacement — but specialized for MoE expert
weights.

The performance-critical questions are:

- **Which experts should stay on GPU?** (cache eviction policy)
- **Which experts will be needed next?** (prefetch prediction)
- **When a cache miss occurs, should we transfer to GPU or run on CPU?**
  (scheduling decision)

### Why Not Just Use a Bigger GPU?

Without expert scheduling you have two options, both bad:

1. **Fit everything in GPU memory.** Mixtral-8x7B at fp16 is ~90 GB.  That
   exceeds even an A100-80 GB — you would need a multi-GPU setup (2–4×
   A100s) just for inference.  DeepSeek-V2 with 160 experts is worse still.

2. **Naive on-demand swapping.** Keep all experts in CPU RAM, transfer each
   one to the GPU when the router requests it.  Every activation is a cache
   miss and a PCIe transfer (~12 GB/s).  At ~1.2 GB per expert that is
   ~100 ms per transfer; with top-2 routing across 32 layers you stall on
   64 transfers per token.  Throughput collapses.

Expert scheduling sits in the middle: keep a **working set** of the most
useful experts on GPU (e.g. 16 out of 64), predict which ones are coming
next, and make smart eviction and scheduling decisions.  The result:

- A model that "needs" 90 GB can run on a **single 24–40 GB GPU** with
  acceptable throughput.
- Cache hit rates of 70–90 %+ mean most expert activations avoid any
  transfer at all.
- Prefetching hides the remaining transfer latency by loading experts
  before they are requested.

In short, expert scheduling **turns a multi-GPU memory problem into a
single-GPU scheduling problem.**

### Prior Art and What Is New

Several systems already perform expert scheduling — but each one
**hard-codes its policy inside the runtime**:

| System | Approach | Limitation |
|--------|----------|------------|
| **MoE-Infinity** (Xue et al., 2024) | LFU cache + activation-aware prefetch | Policy baked into C++/CUDA; changing it means rewriting the system |
| **ExpertFlow** (He et al., 2024) | Score-based eviction + predictive offload | Same — policy is inseparable from the engine |
| **FasterMoE** (He et al., 2022) | Dynamic shadowing (replicate hot experts) | Training-focused, not inference |
| **DeepSpeed-MoE** | Expert parallelism across GPUs | Assumes multi-GPU, no single-GPU caching |
| **vLLM** | PagedAttention for KV-cache | No explicit expert caching policy — relies on PyTorch memory |

The novelty of MoE-Sched is not any single eviction strategy.  It is the
**separation of concerns**:

- **Policy as data, not code** — a 10-line `.moe` file replaces 300+ lines
  of C++.
- **Free composition** — mix any eviction strategy with any prefetch
  strategy and any scheduler.  Existing systems only support their one
  built-in combination.
- **Runtime adaptation** — policies can change parameters based on observed
  metrics without restarting inference.
- **Reproduce and compare** — MoE-Infinity's policy, ExpertFlow's policy,
  and vLLM's default behavior can all be expressed in the same DSL and
  benchmarked on the same routing traces.

This is the same idea as SQL vs. hand-written B-tree traversals, or CSS vs.
inline rendering code.  The individual techniques exist — the DSL
abstraction over them does not.

### What MoE-Sched Does

MoE-Sched is a **domain-specific language and runtime** that decouples
scheduling policy from the inference engine.  Instead of writing hundreds of
lines of runtime code, you write a short `.moe` policy file:

```
policy my_policy {
    cache    { capacity = 16  eviction = lfu  frequency_decay = 0.9 }
    prefetch { strategy = history  budget = 4 }
    schedule { mode = hybrid  offload_threshold_ms = 40.0 }
}
```

The MoE-Sched compiler translates this declaration into an efficient runtime
**hook** — a callback that the inference engine invokes on every MoE layer
forward pass.

### How It Works: The `on_layer` Pipeline

Each time the model's router selects experts for a layer, the hook executes
a five-stage pipeline:

```
Router selects experts  ──►  on_layer(layer_idx, experts, scores)
                                │
                    ┌───────────┼───────────────┐
                    ▼           ▼               ▼
               1. Cache     2. Scheduler    3. Prefetcher
               lookup       decision        prediction
               (hit/miss)   (GPU/CPU)       (next layer)
                    │           │               │
                    ▼           ▼               ▼
               4. Triggers  5. Monitor
               (memory/TTL) (hit rate, latency)
                    │
                    ▼
               DispatchPlan (per-expert placement decisions)
```

1. **Cache lookup** — Is the expert already resident on GPU?  If not, the
   eviction policy decides what to evict to make room.
2. **Scheduler decision** — On a cache miss: transfer the expert to GPU
   (fast compute, slow transfer) or execute on CPU (no transfer, slower
   compute)?  Hybrid mode decides per-expert based on estimated transfer
   cost.
3. **Prefetch prediction** — Based on routing history, predict which experts
   upcoming layers will need and pre-load them before they're requested.
4. **Trigger evaluation** — Fire memory-pressure eviction if GPU memory is
   critically full, or TTL eviction if experts haven't been used recently.
5. **Monitor recording** — Track hit rate, latency, and memory metrics.
   Adaptive policies use these metrics to change parameters at runtime.

The returned `DispatchPlan` tells the inference engine, for each selected
expert, whether it's a cache hit, which device to execute on, and whether a
CPU-to-GPU transfer is required.

### Four Orthogonal Policy Axes

Every MoE-Sched policy is a composition of four independent axes:

| Axis | Controls | Options |
|------|----------|---------|
| **Cache** | Which experts stay on GPU | LRU, LFU, Score, Frequency-Threshold, Fallback (two-tier) |
| **Prefetch** | Which experts to load proactively | None, Affinity, History, Lookahead |
| **Schedule** | Where to execute cache misses | GPU-only, CPU-fallback, Hybrid |
| **Triggers** | When to force eviction | Memory pressure, TTL |

These axes are orthogonal — any eviction strategy works with any prefetch
strategy and any scheduling mode.  The DSL makes this composability
explicit.

### What MoE-Sched Is *Not*

| MoE-Sched can | MoE-Sched cannot |
|---------------|-------------------|
| Manage expert caching, prefetch, and scheduling for one MoE model | Compose or route between multiple different models |
| Adaptively tune policy parameters based on runtime metrics | Perform model parallelism or tensor sharding |
| Reproduce published systems' policies in ~15 lines of DSL | Replace the inference engine itself (vLLM, TGI, etc.) |
| Integrate with HuggingFace Transformers as a hook | Manage training workloads — inference only |
| Simulate policies offline using recorded routing traces | Perform hardware-level GPU memory management |

MoE-Sched is analogous to an OS page-replacement policy, but specialized
for MoE expert weights.  It makes MoE inference faster and more
memory-efficient — it does not create a multi-model serving fabric.

---

## 2. Quick Start

### Installation

```bash
cd conference-paper
pip install -e ".[dev]"
```

### Minimal .moe Policy

Create a file `my_policy.moe`:

```
policy my_first_policy {
    cache {
        capacity = 8
        eviction = lru
    }
}
```

### Load and Run

```python
from moe_sched import parse_file, compile_policy, build_hook

# Parse the .moe file
policies = parse_file("my_policy.moe")
ir = policies[0]

# Compile to runtime hook
compiled = compile_policy(ir)
hook = build_hook(compiled)

# Use in inference loop
for layer_idx, experts, scores in inference_events:
    plan = hook.on_layer(layer_idx, experts, scores)
    for dispatch in plan.dispatches:
        print(f"Expert {dispatch.expert_id}: "
              f"{'HIT' if dispatch.cache_hit else 'MISS'}, "
              f"device={dispatch.device}")
```

### Or Use the Python eDSL

```python
from moe_sched import MoESched, compile_policy, build_hook

sched = MoESched()

@sched.policy
def my_first_policy(p):
    p.cache(capacity=8, eviction='lru')

compiled = compile_policy(sched.policies['my_first_policy'])
hook = build_hook(compiled)
```

---

## 3. Two Frontends

MoE-Sched provides two equivalent ways to define policies. Both produce
the same intermediate representation (PolicyIR) and are fully interchangeable.

### 3.1 .moe File Syntax

Standalone text files parsed by a Lark LALR grammar (v0.6).

**Structure:**
```
version 0.6   # optional — parser rejects files requiring a newer version

# Comments start with # (full-line or inline)
policy <name> {
    cache    { <params> }
    prefetch { <params> }    # optional
    schedule { <params> }    # optional
    monitor  { <params> }    # optional
    adapt    { <rules> }     # optional
}
```

**Syntax conventions:**
- Parameters are separated by whitespace (newlines or spaces).
- **Semicolons** are accepted as optional visual separators and are silently ignored:
  ```
  cache { capacity = 16; eviction = lru; }
  ```
- **Duplicate parameters** within a block are rejected with a clear error.
- **Unknown enum values** are rejected (e.g., `eviction = bogus` raises `DSLError`).

A `.moe` file can contain multiple policies:

```
policy fast_lru {
    cache { capacity = 8  eviction = lru }
}

policy careful_lfu {
    cache { capacity = 16  eviction = lfu  frequency_decay = 0.9 }
    prefetch { strategy = history  budget = 4 }
    schedule { mode = hybrid  overlap = true }
}
```

**Loading:**
```python
from moe_sched import parse_file, parse_policy, parse_policies

# From a file
policies = parse_file("path/to/policies.moe")

# From a string (multiple policies)
policies = parse_policies(source_text)

# From a string (single policy, convenience)
ir = parse_policy("""
    policy my_policy {
        cache { capacity = 8  eviction = lru }
    }
""")
```

### 3.2 Python eDSL

Define policies using Python decorators — useful for programmatic generation.

```python
from moe_sched import MoESched

sched = MoESched()

@sched.policy
def my_policy(p):
    p.cache(capacity=16, eviction='lfu', lfu_decay=0.9)
    p.prefetch(strategy='history', budget=4)
    p.schedule(mode='hybrid', cpu_threshold_ms=40.0)
    p.monitor(metrics=["hit_rate"], window=100)

# Access the compiled IR
ir = sched.policies['my_policy']
```

**Rules:**
- The function name becomes the policy name
- Each `p.<axis>(...)` call can appear at most once
- `p.cache(...)` is mandatory; all others are optional
- String values are accepted for enum parameters (e.g., `'lru'` instead of `EvictionPolicy.LRU`)

### 3.3 Fluent Builder

Method-chaining alternative for inline policy construction:

```python
sched = MoESched()

ir = (sched.build("my_policy")
    .cache(capacity=16, eviction='lfu', lfu_decay=0.9)
    .prefetch(strategy='history', budget=4)
    .schedule(mode='hybrid')
    .done())

sched.register(ir)
```

---

## 4. Policy Axes

### 4.1 Cache

Controls which expert weight matrices are kept resident on GPU.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity` | int | **required** | Number of expert slots (1–512) |
| `eviction` | enum | `lru` | Eviction strategy |
| `pin` | int list | `[]` | Expert IDs permanently cached |
| `frequency_decay` | float | `0.95` | LFU frequency decay factor (0, 1) |
| `freq_threshold` | float | `0.05` | Frequency threshold for freq_threshold eviction |
| `freq_window` | int | `100` | Window size for frequency counting |
| `score_ema_alpha` | float | `0.3` | EMA smoothing for score-based eviction |
| `memory_threshold` | float | *none* | GPU memory pressure trigger (0, 1] |
| `memory_headroom` | float | `0.7` | Target occupancy after pressure eviction |
| `memory_budget_gb` | float | `16.0` | Simulated GPU memory budget |
| `expert_size_gb` | float | `1.2` | Size of one expert in GB |
| `ttl` | int | *none* | Time-to-live in accesses (evict if unused) |

#### Eviction Strategies

**`lru`** — Least Recently Used. Evicts the expert that hasn't been accessed
for the longest time. Best for workloads with temporal locality.

```
cache { capacity = 16  eviction = lru }
```

**`lfu`** — Least Frequently Used with decay. Tracks access frequency and
applies periodic decay to prevent old popular experts from never being evicted.

```
cache { capacity = 16  eviction = lfu  frequency_decay = 0.9 }
```

**`score`** — Score-based eviction using router attention scores. Evicts the
expert with the lowest exponential moving average score. Requires a prefetch
strategy that provides scores (affinity, history, or lookahead).

```
cache { capacity = 16  eviction = score  score_ema_alpha = 0.3 }
prefetch { strategy = affinity }  # required for score eviction
```

**`frequency_threshold`** — Evicts experts whose activation frequency falls
below a threshold over a rolling window.

```
cache {
    capacity = 16
    eviction = frequency_threshold
    freq_threshold = 0.05
    freq_window = 100
}
```

#### Expert Pinning

Pin specific expert IDs so they are never evicted. Useful for shared experts
(DeepSeek) or known-hot experts.

```
cache {
    capacity = 16
    eviction = lfu
    pin = [0, 1, 2]   # experts 0–2 are always cached
}
```

**Constraint:** `len(pin) <= capacity`

### 4.2 Prefetch

Controls proactive expert loading before they are needed.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | enum | `none` | Prefetch strategy |
| `lookahead` | int | `1` | Layers to look ahead (for `lookahead`) |
| `budget` | int | `4` | Max experts to prefetch per step |
| `affinity_threshold` | float | `0.3` | Min co-activation probability |
| `history_window` | int | `50` | Size of history buffer |

#### Strategies

**`none`** — No prefetching. Only reactive cache management.

**`affinity`** — Predicts experts for the next layer based on co-activation
patterns. If expert A at layer L frequently co-occurs with expert B at layer
L+1, B is prefetched when A is activated.

```
prefetch { strategy = affinity  affinity_threshold = 0.3  budget = 4 }
```

**`history`** — Tracks recent expert activations across all layers and
prefetches the most frequent non-current experts.

```
prefetch { strategy = history  history_window = 50  budget = 4 }
```

**`lookahead`** — Per-layer pattern matching. Maintains a sliding window of
activations *per layer index* and predicts the most frequent experts for
layers L+1 through L+lookahead.

```
prefetch { strategy = lookahead  lookahead = 2  budget = 4  history_window = 40 }
```

### 4.3 Schedule

Controls where cache-missed experts execute.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | enum | `gpu_only` | Execution strategy |
| `offload_threshold_ms` | float | `50.0` | Transfer cost threshold for hybrid mode |
| `overlap` | bool | `true` | Allow overlapped CPU/GPU execution |
| `priority_routing` | bool | `false` | Route high-score experts to GPU first |

#### Modes

**`gpu_only`** — All experts execute on GPU. Cache misses trigger a CPU→GPU
transfer before execution.

```
schedule { mode = gpu_only }
```

**`cpu_fallback`** — Cache misses execute on CPU instead of transferring
to GPU. Avoids transfer latency at the cost of slower CPU compute.

```
schedule { mode = cpu_fallback }
```

**`hybrid`** — Decision is made per-expert based on estimated transfer cost.
If the transfer would take less than `offload_threshold_ms`, transfer to GPU;
otherwise execute on CPU. Requires `overlap = true`.

```
schedule {
    mode = hybrid
    offload_threshold_ms = 40.0
    overlap = true
    priority_routing = true
}
```

### 4.4 Monitor

Collects runtime metrics for introspection and adaptive rules.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | string list | `["hit_rate"]` | Metrics to track |
| `window` | int | `100` | Rolling window size |
| `log_interval` | int | `50` | Steps between log outputs |

```
monitor {
    metrics = ["hit_rate", "latency", "memory"]
    window = 200
    log_interval = 50
}
```

### 4.5 Eviction Triggers

Orthogonal to the base eviction strategy — triggers fire proactive evictions
based on external conditions.

#### Memory Pressure

When estimated GPU usage exceeds `memory_threshold`, evict non-pinned experts
until usage drops below `memory_headroom`.

```
cache {
    capacity = 24
    eviction = lfu
    memory_threshold = 0.9    # trigger at 90% usage
    memory_headroom = 0.7     # evict down to 70%
    memory_budget_gb = 16.0   # simulated GPU budget
    expert_size_gb = 1.2      # per-expert size
}
```

#### TTL (Time-to-Live)

Evict any non-pinned expert that hasn't been accessed in `ttl` accesses.
Prevents stale experts from occupying cache slots.

```
cache {
    capacity = 16
    eviction = lru
    ttl = 200     # evict if unused for 200 accesses
}
```

Both triggers can be combined with any base eviction strategy.

### 4.6 Conditional Expressions

Parameters support a default-first `when` guard using a colon separator.
These desugar into implicit `adapt` rules — the first value (before `:`)
is the default, and the second value is applied when the condition holds.

```
cache {
    capacity = 16
    eviction = lru : lfu when hit_rate > 0.5
    frequency_decay = 0.95 : 0.85 when hit_rate > 0.7
}

prefetch {
    strategy = history : lookahead when hit_rate > 0.6
    budget = 4 : 8 when hit_rate > 0.8
}
```

**Supported parameters:** `eviction`, `capacity`, `frequency_decay` (cache);
`strategy`, `budget` (prefetch).

**Semantics:** The policy starts with the default value (before `:`). When
the runtime monitor detects the condition is met, the policy dynamically
adapts to the conditional value (after `:`). This is equivalent to writing
an explicit `adapt` rule.

### 4.7 Composition Operators

Cache strategies can be layered using the pipe operator:

```
cache {
    capacity = 24
    eviction = lfu | lru
}
```

This creates a **two-tier cache**:
- **Primary tier** (LFU, 2/3 of capacity): The main cache
- **Secondary tier** (LRU, 1/3 of capacity): Fallback cache

**Behavior:**
- If the expert is in the primary tier → cache hit
- If the expert is in the secondary tier → cache hit, expert promoted to primary
- If in neither → cache miss, expert inserted into primary

The primary/secondary split is automatically computed by the compiler.
Any combination of base eviction strategies can be composed:
`lfu | lru`, `score | lfu`, etc.

---

## 5. Adaptive Policies

Adaptive policies monitor runtime metrics and dynamically adjust policy
parameters when conditions are met. This enables **profiling-driven expert
placement** without manual intervention.

### .moe Syntax

```
policy adaptive_switcher {
    cache { capacity = 16  eviction = lfu  frequency_decay = 0.9 }
    prefetch { strategy = history  budget = 4 }

    adapt {
        # Switch to LRU if hit rate drops below 40% for 100 consecutive accesses
        when hit_rate < 0.4 for 100 accesses { eviction = lru }

        # Trigger memory pressure eviction if eviction rate spikes
        when eviction_rate > 0.3 { trigger memory_pressure }

        # Increase capacity if hit rate is very low (instant, no window)
        when hit_rate < 0.2 { capacity = 32 }
    }
}
```

### Python eDSL

```python
from moe_sched import MoESched, AdaptRule, AdaptCondition, AdaptAction

sched = MoESched()

@sched.policy
def adaptive_policy(p):
    p.cache(capacity=16, eviction='lfu', lfu_decay=0.9)
    p.prefetch(strategy='history', budget=4)
    p.adapt([
        AdaptRule(
            condition=AdaptCondition('hit_rate', '<', 0.4, window=100),
            action=AdaptAction(param='eviction', value='lru'),
            cooldown=50,
        ),
        AdaptRule(
            condition=AdaptCondition('eviction_rate', '>', 0.3),
            action=AdaptAction(param='trigger', value='memory_pressure'),
        ),
    ])
```

### Comparison Operators

| Operator | Meaning |
|----------|---------|
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less than or equal |
| `>=` | Greater than or equal |
| `!=` | Not equal |

Note: `==` is intentionally excluded — exact floating-point equality is
almost never meaningful for metrics.  Use `>=` or `<=` instead.

### Available Metrics

| Metric | Description |
|--------|-------------|
| `hit_rate` | Cache hits / (hits + misses) |
| `eviction_rate` | Evictions / (hits + misses) |

Metric names are validated at parse time.  Unknown metrics (e.g., typos)
raise a `DSLError` immediately rather than silently failing at runtime.

### Available Actions

| Action | Example | Effect |
|--------|---------|--------|
| Change eviction | `eviction = lru` | Hot-swap cache implementation |
| Change capacity | `capacity = 32` | Resize cache |
| Change LFU decay | `lfu_decay = 0.8` | Adjust frequency decay (adapt action uses IR field name) |
| Change prefetch strategy | `prefetch_strategy = lookahead` | Swap prefetcher |
| Change prefetch budget | `prefetch_budget = 8` | Adjust prefetch aggressiveness |
| Change schedule mode | `schedule_mode = hybrid` | Switch execution strategy |
| Fire trigger | `trigger memory_pressure` | Force pressure eviction |

### Adaptation Semantics

1. **Window**: Condition must hold for `window` *consecutive* evaluations
2. **Cooldown**: After firing, a rule cannot fire again for `cooldown` steps (default: 50)
3. **Validation**: Every adapted configuration passes the same 17 validation rules
4. **Recompilation**: On adaptation, the policy IR is mutated and recompiled to new runtime components
5. **Non-nesting**: The adaptive wrapper is never duplicated

---

## 6. Per-Layer Adaptive Caching

For models with many experts (e.g., DeepSeek with 64), different layers may
have different routing characteristics. Per-layer adaptive caching maintains
**separate caches per layer** with capacity allocated based on routing entropy.

### Usage

```python
from moe_sched.ir import PolicyIR, CacheIR, EvictionPolicy
from moe_sched.runtime.per_layer import PerLayerHook, PerLayerConfig

# Base policy (capacity is the per-layer default)
ir = PolicyIR(
    name="entropy_adaptive",
    cache=CacheIR(capacity=16, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
)

config = PerLayerConfig(
    entropy_window=200,       # sliding window for entropy computation
    min_capacity=4,           # minimum cache slots per layer
    max_capacity=48,          # maximum cache slots per layer
    rebalance_interval=500,   # steps between rebalances
    total_budget=432,         # total slots across all layers (27 layers * 16)
)

hook = PerLayerHook(ir, num_layers=27, num_experts=64, config=config)

# Use like any other hook
plan = hook.on_layer(layer_idx=5, selected_experts=[3, 12, 45, 7, 22, 1])
```

### How It Works

1. **Entropy tracking**: Each layer's routing entropy is computed over a
   sliding window of expert activations
2. **Proportional allocation**: Higher-entropy layers (more uniform routing)
   receive more cache capacity; lower-entropy layers (concentrated routing)
   receive less
3. **Periodic rebalancing**: Every `rebalance_interval` steps, capacities are
   recalculated and caches are rebuilt for layers whose allocation changed

### PerLayerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_window` | int | `200` | Sliding window for entropy computation |
| `min_capacity` | int | `2` | Minimum cache slots per layer |
| `max_capacity` | int | `64` | Maximum cache slots per layer |
| `rebalance_interval` | int | `500` | Steps between rebalances |
| `total_budget` | int | *auto* | Total cache slots across all layers |

### Inspecting Results

```python
stats = hook.stats_snapshot()
print(stats["entropies"])   # {0: 5.84, 1: 5.93, ...}
print(stats["capacities"])  # {0: 14, 1: 18, ...}
print(stats["per_layer"])   # per-layer hit rates
```

---

## 7. Compilation and Runtime

### Pipeline

```
.moe file / Python eDSL
         ↓
     PolicyIR (dataclasses)
         ↓
     Validator (17 rules)
         ↓
     Compiler (component assembly)
         ↓
     PolicyHook / FastPolicyHook
```

### Compiling a Policy

```python
from moe_sched import compile_policy, build_hook

# From any PolicyIR (parsed or built)
compiled = compile_policy(ir)
hook = build_hook(compiled)
```

### The Dispatch Loop

`hook.on_layer()` is called once per MoE layer per token:

```python
plan = hook.on_layer(
    layer_idx=5,                    # which layer
    selected_experts=[2, 7],        # router's top-k selection
    scores=[0.62, 0.38],           # optional: router scores
    expert_size_gb=1.2,            # optional: for hybrid scheduler
)
```

### DispatchPlan

The returned `DispatchPlan` tells the inference engine what to do:

```python
plan.layer_idx      # int: which layer
plan.dispatches     # list[ExpertDispatch]: per-expert decisions
plan.prefetched     # list[int]: experts prefetched for future layers
plan.hits           # int: cache hits this layer
plan.misses         # int: cache misses this layer

for d in plan.dispatches:
    d.expert_id     # int
    d.device        # ExecutionDevice.GPU or ExecutionDevice.CPU
    d.cache_hit     # bool
    d.transferred   # bool: CPU→GPU transfer occurred
```

### Stats Introspection

```python
stats = hook.stats_snapshot()
# Returns:
# {
#     "name": "my_policy",
#     "steps": 1000,
#     "cache": {"hits": 850, "misses": 150, "evictions": 120, "hit_rate": 0.85},
#     "prefetch": {"issued": 400, "useful": 280, "accuracy": 0.7},
#     "scheduler": {"gpu": 900, "cpu": 100, "transfers": 50},
# }
```

### Cython Fast Path

When Cython extensions are built, the compiler automatically uses optimized
implementations:

```bash
python setup_cython.py build_ext --inplace
```

This provides:
- **Phase 2**: Cython cache and scheduler components (`LRUCacheFast`, etc.)
- **Phase 3**: Full Cython dispatch loop (`FastPolicyHook`)

The selection is transparent — `build_hook()` automatically picks the fastest
available implementation.

---

## 8. Autotuner

The autotuner performs grid search over the DSL parameter space to find
optimal configurations for a given workload trace.

### Basic Usage

```python
from moe_sched.autotuner import autotune
import json

# Load a trace
with open("traces/mixtral_sample.jsonl") as f:
    header = json.loads(f.readline())
    trace_data = [json.loads(line) for line in f]

# Find the best policy
best, top5 = autotune(trace_data, metric='hit_rate', top_k=5)

print(f"Best hit rate: {best.hit_rate:.1%}")
print(f"Parameters: {best.params}")
```

### Custom Grid

```python
from moe_sched.autotuner import autotune

custom_grid = {
    "capacity": [8, 16, 32, 48],
    "eviction": ["lru", "lfu"],
    "lfu_decay": [0.8, 0.9, 0.95],
    "prefetch_strategy": ["none", "history", "lookahead"],
    "prefetch_budget": [2, 4, 8],
    "schedule_mode": ["gpu_only", "hybrid"],
}

best, top5 = autotune(trace_data, grid=custom_grid, measure_latency=True)

for i, result in enumerate(top5):
    print(f"#{i+1}: {result.hit_rate:.1%} | "
          f"{result.dispatch_mean_us:.1f}µs | "
          f"{result.params}")
```

### Default Grid

| Parameter | Values |
|-----------|--------|
| `capacity` | 4, 8, 16, 32 |
| `eviction` | lru, lfu |
| `lfu_decay` | 0.8, 0.9, 0.95 |
| `prefetch_strategy` | none, history |
| `prefetch_budget` | 2, 4, 8 |
| `schedule_mode` | gpu_only, hybrid |

The autotuner prunes invalid combinations (e.g., `lfu_decay` with LRU,
`prefetch_budget` > `capacity`).

### TuningResult

```python
result.params           # dict: parameter configuration
result.hit_rate         # float: cache hit rate
result.hits             # int: total cache hits
result.misses           # int: total cache misses
result.evictions        # int: total evictions
result.dispatch_mean_us # float: mean dispatch latency (if measured)
```

---

## 9. Validation Rules

Every policy (parsed or programmatically built) is validated before
compilation. There are **17 semantic rules** that catch configuration errors
early:

### Cache Rules
1. `capacity` must be between 1 and 512
2. `pin_experts` must contain valid IDs (0–511)
3. Cannot pin more experts than cache capacity
4. `lfu_decay` must be in (0, 1)
5. `freq_threshold` must be in [0, 1]
6. `freq_window` must be ≥ 1
7. `score_ema_alpha` must be in (0, 1]

### Prefetch Rules
8. `lookahead` must be ≥ 1
9. `prefetch budget` must be ≥ 1
10. Prefetch budget should not exceed cache capacity
11. `affinity_threshold` must be in [0, 1]
12. `history_window` must be ≥ 1

### Schedule Rules
13. `offload_threshold_ms` must be > 0
14. HYBRID mode requires `overlap = true`

### Cross-Block Rules
15. SCORE eviction requires a non-NONE prefetch strategy

### Trigger Rules
16. `memory_threshold` must be in (0, 1] when set
17. `memory_headroom` must be in (0, `memory_threshold`] when set

### Parser-Level Checks (before validation)
18. Unknown enum values are rejected (eviction, prefetch strategy, schedule mode)
19. Unknown metric names in `adapt` and conditional expressions are rejected
20. Duplicate parameters within a block are rejected
21. Duplicate blocks within a policy are rejected

**Validation errors include all violated rules at once:**

```python
from moe_sched import parse_policy

try:
    ir = parse_policy("""
        policy bad {
            cache { capacity = 0  eviction = score }
        }
    """)
except Exception as e:
    print(e)
    # ValidationError: ['cache.capacity must be between 1 and 512',
    #                    'SCORE eviction requires a non-NONE prefetch strategy']
```

---

## 10. Working with Traces

MoE-Sched uses expert activation traces in JSONL format for offline
evaluation and autotuning.

### Trace Format

```json
{"model_name": "Mixtral-8x7B", "num_layers": 32, "num_experts": 8, "top_k": 2}
{"t": 0, "l": 0, "e": [2, 5], "s": [0.62, 0.38]}
{"t": 0, "l": 1, "e": [1, 7], "s": [0.55, 0.45]}
...
```

- **Line 1**: Header with model metadata
- **Subsequent lines**: One entry per layer per token
  - `t`: Token index (resets to 0 at prompt boundaries)
  - `l`: Layer index
  - `e`: List of selected expert IDs
  - `s`: Optional list of router scores (aligned with `e`)

### Loading Traces

```python
import json

with open("traces/mixtral_sample.jsonl") as f:
    header = json.loads(f.readline())
    trace_data = [json.loads(line) for line in f]

print(f"Model: {header['model_name']}")
print(f"{len(trace_data)} entries across {header['num_layers']} layers")
```

### Recording Traces (Colab)

See `notebooks/01_trace_recording.ipynb` for recording traces from real
models on GPU. The notebook hooks into HuggingFace Transformers' MoE gate
layers to capture router decisions.

### Replaying Traces

```python
from moe_sched import parse_policy, compile_policy, build_hook

ir = parse_policy("""
    policy test { cache { capacity = 16  eviction = lfu  frequency_decay = 0.9 } }
""")
compiled = compile_policy(ir)
hook = build_hook(compiled)

for entry in trace_data:
    hook.on_layer(
        layer_idx=entry['l'],
        selected_experts=entry['e'],
        scores=entry.get('s'),
    )

stats = hook.stats_snapshot()
print(f"Hit rate: {stats['cache']['hit_rate']:.1%}")
```

---

## 11. Examples

### Minimal LRU

```
policy lru_basic {
    cache { capacity = 8  eviction = lru }
}
```

### LFU with History Prefetch

```
policy lfu_history {
    cache { capacity = 16  eviction = lfu  frequency_decay = 0.9 }
    prefetch { strategy = history  budget = 4  history_window = 50 }
    schedule { mode = cpu_fallback }
}
```

### Score-Based with Affinity Prefetch and Hybrid Scheduling

```
policy score_affinity {
    cache { capacity = 16  eviction = score  score_ema_alpha = 0.3 }
    prefetch { strategy = affinity  affinity_threshold = 0.3  budget = 4 }
    schedule { mode = hybrid  offload_threshold_ms = 40.0  overlap = true }
}
```

### Full Composition (All Axes)

```
policy composed_full {
    cache {
        capacity = 24
        eviction = lfu
        pin = [0]
        frequency_decay = 0.9
        memory_threshold = 0.75
        memory_headroom = 0.4
        memory_budget_gb = 16.0
        expert_size_gb = 1.2
        ttl = 200
    }
    prefetch {
        strategy = lookahead
        lookahead = 2
        budget = 4
        history_window = 40
    }
    schedule {
        mode = hybrid
        offload_threshold_ms = 40.0
        overlap = true
        priority_routing = true
    }
    monitor {
        metrics = [hit_rate, latency, memory]
        window = 200
        log_interval = 50
    }
}
```

### Adaptive Policy

```
policy adaptive_lfu {
    cache { capacity = 16  eviction = lfu  frequency_decay = 0.9 }
    prefetch { strategy = history  budget = 4 }

    adapt {
        when hit_rate < 0.4 for 100 accesses { eviction = lru }
        when eviction_rate > 0.3 { trigger memory_pressure }
    }
}
```

### DeepSeek-Optimized (Pinning + Large Cache)

```
policy deepseek_optimized {
    cache {
        capacity = 48
        eviction = lfu
        frequency_decay = 0.9
        pin = [9, 16, 27, 22, 44, 50, 1, 30, 6, 41, 4, 15]
    }
    prefetch { strategy = history  budget = 4  history_window = 50 }
    schedule { mode = gpu_only }
}
```

### 11.1 Reproducing Published Systems

MoE-Sched can express the expert management strategies of all nine surveyed
MoE serving systems:

**ExpertFlow** (score-based + lookahead prefetch):
```
policy expertflow {
    cache { capacity = 16  eviction = score  score_ema_alpha = 0.3 }
    prefetch { strategy = lookahead  lookahead = 2  budget = 4 }
    schedule { mode = hybrid  offload_threshold_ms = 1.5 }
}
```

**FineMoE** (score-based, GPU-only):
```
policy finemoe {
    cache { capacity = 16  eviction = score  score_ema_alpha = 0.3 }
    prefetch { strategy = affinity  budget = 4 }
    schedule { mode = gpu_only }
}
```

**HybriMoE** (score + hybrid + memory pressure):
```
policy hybrimoe {
    cache {
        capacity = 16
        eviction = score
        score_ema_alpha = 0.3
        memory_threshold = 0.85
        memory_headroom = 0.6
    }
    schedule { mode = hybrid  offload_threshold_ms = 50.0  overlap = true }
}
```

**Fiddler** (LRU + hybrid):
```
policy fiddler {
    cache { capacity = 8  eviction = lru }
    schedule { mode = hybrid  offload_threshold_ms = 50.0  overlap = true }
}
```

**MoE-Infinity** (frequency-threshold, GPU-only):
```
policy moe_infinity {
    cache {
        capacity = 16
        eviction = frequency_threshold
        freq_threshold = 0.05
        freq_window = 100
    }
    schedule { mode = gpu_only }
}
```

**ProMoE** (LRU + history prefetch):
```
policy promoe {
    cache { capacity = 16  eviction = lru }
    prefetch { strategy = history  budget = 4  history_window = 50 }
    schedule { mode = gpu_only }
}
```

---

## 12. API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `parse_policy(source)` | Parse a single policy from a string |
| `parse_policies(source)` | Parse multiple policies from a string |
| `parse_file(path)` | Parse a `.moe` file |
| `compile_policy(ir)` | Compile PolicyIR to CompiledPolicy |
| `build_hook(compiled)` | Create a runtime hook (Python or Cython) |
| `validate_policy(ir)` | Validate a PolicyIR (raises on failure) |

### Classes

| Class | Description |
|-------|-------------|
| `MoESched` | Entry point — `@sched.policy` decorator + `sched.build()` |
| `PolicyIR` | Intermediate representation of a complete policy |
| `CacheIR` | Cache configuration IR |
| `PrefetchIR` | Prefetch configuration IR |
| `ScheduleIR` | Schedule configuration IR |
| `MonitorIR` | Monitor configuration IR |
| `PolicyHook` | Python dispatch orchestrator |
| `DispatchPlan` | Result of `hook.on_layer()` |
| `ExpertDispatch` | Per-expert placement decision |
| `AdaptRule` | Adaptive rule (condition → action) |
| `AdaptCondition` | Metric comparison condition |
| `AdaptAction` | Parameter mutation action |
| `PerLayerHook` | Per-layer entropy-adaptive dispatch |
| `PerLayerConfig` | Configuration for per-layer caching |
| `RoutingEntropyTracker` | Computes per-layer Shannon entropy |

### Enums

| Enum | Values |
|------|--------|
| `EvictionPolicy` | `lru`, `lfu`, `score`, `frequency_threshold` |
| `PrefetchStrategy` | `none`, `affinity`, `history`, `lookahead` |
| `ScheduleMode` | `gpu_only`, `cpu_fallback`, `hybrid` |
| `ExecutionDevice` | `GPU`, `CPU` |

### Autotuner

| Function | Description |
|----------|-------------|
| `autotune(trace_data, grid=None, metric='hit_rate', top_k=5)` | Grid-search for optimal policy |

Returns `(best_result, top_k_results)` where each result contains
`params`, `hit_rate`, `evictions`, `dispatch_mean_us`, `hits`, `misses`.

### Exceptions

| Exception | Description |
|-----------|-------------|
| `DSLError` | Raised for structural or syntax-level issues (missing cache, duplicate params, unknown enums, unknown metrics, file not found) |
| `ValidationError` | Raised when a policy violates semantic constraints (17 validation rules, reports all violations at once) |

Both are importable from the top-level package:
```python
from moe_sched import DSLError, ValidationError
```

---

## 13. Command-Line Interface

MoE-Sched provides a CLI for validating and inspecting `.moe` files.

### Usage

```bash
# Via python -m
python -m moe_sched validate examples/*.moe
python -m moe_sched parse examples/composed_policy.moe
python -m moe_sched version

# Via entry point (after pip install)
moe-sched validate examples/*.moe
```

### Commands

| Command | Description |
|---------|-------------|
| `validate FILE [FILE ...]` | Parse and validate one or more `.moe` files.  Reports pass/fail per file with policy names. |
| `parse FILE` | Parse a `.moe` file and print the IR (cache, prefetch, schedule, monitor, adapt rules). |
| `version` | Print the MoE-Sched version. |

### Example Output

```
$ moe-sched validate examples/lru_policy.moe examples/composed_policy.moe
  ✓ examples/lru_policy.moe  (1 policy: lru_baseline)
  ✓ examples/composed_policy.moe  (1 policy: composed_showcase)

2/2 files passed.
```

```
$ moe-sched parse examples/composed_policy.moe
Policy: composed_showcase
  Cache:    capacity=24  eviction=lfu
            pin=[0]
  Prefetch: strategy=lookahead  budget=4
  Schedule: mode=hybrid  cpu_threshold_ms=40.0
  Monitor:  metrics=['hit_rate', 'latency', 'memory']  window=200
```

The `validate` command exits with code 0 on success, 1 if any file fails.
