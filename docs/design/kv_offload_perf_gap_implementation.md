# KV offload performance gap: implementation sequence

This plan is grounded in vLLM `2840ca7a33` and KVCR `317e62f`. The four
`perf-gap/01-*` through `perf-gap/04-*` branches contain measurement changes.
Optional implementations for Stages 2 through 6 now live in separate commits
on `perf-gap/05-*` through `perf-gap/09-*`. Stage 1's startup and page location
probes are still pending. The implementations have static checks but have not
been run with vLLM/KVCR or GPUs on the target host. Cherry-pick in the order
below; each branch tip includes its predecessors. Do not reset the live cache
to make comparisons; the benchmark operator owns workload runs.

## Why this order

Simple CPU and native tiering have a reported performance gap even with
`secondary_tiers: []`. Their scheduler and worker paths differ in who first
populates the shared CPU region. Tiering's scheduler maps and populates the
whole region; simple CPU generally has workers populate their slices. This
can change NUMA placement and GPU-to-host copy cost without involving KVCR.
The first branch should isolate that difference before changing eviction or
secondary write traffic.

The current default LRU ranks a request's tail chunks before its head at
request completion, but it does not guarantee prefix-closed CPU residency.
Leaf-first eviction is therefore a separate, evidence-driven experiment.
Current tiering eagerly attempts a best-effort cascade after primary stores;
it is not an acknowledged write-through contract. Write-back and primary
invalidation require new correctness machinery and come after the basic
placement comparison.

## Stage 1: NUMA evidence and safe startup ordering

Pending branch: `perf-gap/numa-placement-probe`.

Add bounded startup diagnostics for the actual EngineCore and worker CPU
affinity, GPU physical index and NUMA node, cpuset-allowed memory nodes,
shared-region page locations, and the identity of the first populating
process. Keep these diagnostics off the hot path. Probe whether each enabled
NIXL backend faults or pins the primary shared mapping during KVCR
construction and whether any secondary can access it before workers finish
population. Include the page-zero population probe in the audit.

The benchmark operator should compare simple CPU and tiering with no
secondary using equal CPU capacity, layout, chunk size, and request stream.
Use the common GPU-to-CPU store, byte, CPU slot, and hook metrics from the
measurement branches. If useful work differs, resolve that first. If bytes
match but copy time and page locality differ, proceed to Stage 2. If locality
matches, profile wrapper and scheduler time before changing NUMA policy.

## Stage 2: optional NUMA placement for the CPU primary

Implemented branch: `perf-gap/05-numa-primary`.

Add a default-off option in `kv_connector_extra_config` for the ordinary
non-canonical, non-replicated layout. Map physical accelerator indices to
online, memory-bearing, cpuset-allowed NUMA nodes, with explicit overrides.
Reject conflicts with strict process-level memory binding. Apply a temporary
thread memory policy **before** any worker-owned page is faulted, including
the population-method probe; restore the exact prior policy in `finally`.
Reject worker slices that share a physical page but request different nodes.
Report the selected node, bind failures, and fallback as separate outcomes.

Do not simply remove scheduler population. If Stage 1 shows registration
faulting, choose and implement a worker-ready barrier before KVCR registers
the primary buffer, or scheduler-led per-slice population under the correct
device policy. Neither path may expose an unpopulated buffer to NIXL. Keep
the legacy allocation path with the option off. Defer canonical and
replicated layouts, KVCR local G2, and Guard pools to separate changes.

Verify page locations with `/proc/<pid>/numa_maps` and sampled page queries
on the target host. Test flag-off equivalence, multiple GPU nodes, missing
topology, conflicting bind, partial startup failure, cleanup, and secondary
registration order. The operator then compares startup time and matched
GPU-to-CPU and CPU-to-GPU byte/time distributions on local and remote nodes.

## Stage 3: lineage-aware CPU eviction, only if the trace supports it

Implemented branch: `perf-gap/06-lineage-eviction`.

First collect a bounded, sampled event trace of parent/child offload chunk
keys, CPU admission/removal, and subsequent lookup outcomes. Existing
aggregate eviction counters cannot prove ancestor-before-descendant
eviction. Keep the content-addressed `OffloadKey`; pass the preceding
*offload chunk key* and position as separate metadata. The parent is within
the same cache group and may differ from the preceding raw block hash when
`blocks_per_chunk > 1`.

If the trace shows material stranded descendants, add an optional
full-attention leaf-first LRU. Track resident child counts across shared
prefixes and branches. Evict the least-recent eligible leaf, honoring pins,
in-flight writes, and current-store protection. A multi-chunk allocation
must select enough safe leaves before committing any eviction. Define
behavior when a child is admitted without a CPU-resident parent; never
report a prefix-closed invariant that does not hold. Leave sliding-window
and recurrent groups on the existing policy until their dependency rule is
specified. Keep current LRU as the default for the first comparison.

Test shared prefixes, branching, partial admission, missing parents,
concurrent requests, protected keys, failed batch allocation, reset, and
cache groups. Measure eviction scan time and allocation failures as well as
later hit, promotion, and recomputation outcomes.

## Stage 4: explicit tracked asynchronous write-through

Implemented branch: `perf-gap/07-tracked-write-through`.

Add a per-secondary policy setting whose default preserves today's
best-effort eager cascade. For the new tracked policy, record per-key,
per-tier states such as `PRIMARY_ONLY`, `COPYING`, and `LOWER_CONFIRMED`.
Primary reads may proceed while copying, but a lower-tier hit is advertised
only after a successful per-key acknowledgement. Backpressure and failed or
`DROPPED` deposits must remain visible and must not silently count as
confirmed residency. Handle both GPU-produced and already-primary keys.
Carry per-key completion results through the tier adapter; aggregate job
success is insufficient for mixed batches. Keep source slots pinned until
NIXL reads have quiesced, including error and timeout paths.

Test mixed successes, retries, duplicate keys, cancellation, delayed
completion, inventory ordering, and secondary eviction. Start with one KVCR
local-G2 secondary. Strict acknowledgement before reporting store completion
and durable lower-tier residency are separate contracts and need separate
decisions; the first implementation is asynchronous and opportunistic.

## Stage 5: write-back with eviction reservation

Implemented branch: `perf-gap/08-write-back`.

Add a two-stage primary eviction reservation. A key lacking confirmed
lower-tier residency cannot lose its last valid copy: reserve and pin its
CPU slot, submit a deposit, and reuse the slot only after that key's
successful acknowledgement and transfer quiescence. If all candidate slots
are pinned or awaiting flush, fail admission cleanly. Optional background
flushing can reduce stalls but cannot replace this correctness path.

Cover full-pool pressure, mixed per-key outcomes, duplicate stores,
timeouts, worker death, cancellation, reset, sleep, and shutdown. A tmpfs
primary is not persistent: restart must not advertise unflushed writes as
recoverable. Compare bytes sent per useful GPU store, eviction stalls, pin
duration, and TTFT tails against the eager and tracked write-through modes.

## Stage 6: write-through with primary invalidation

Implemented branch: `perf-gap/09-write-through-invalidate`.

Add `invalidate_if_unpinned(key, generation)` to the CPU primary. After a
confirmed lower-tier deposit, retire exactly that primary residency only
when GPU loads, secondary reads, and other pins finish. During copying,
lookups continue to see the primary copy or a pending result. A failed or
`DROPPED` deposit keeps the primary copy. Never use KVCR `release` as key
invalidation; it releases a residency claim. Test promotion/invalidation
races, concurrent reads, duplicate keys, slot reuse, and hot-key
promote/store loops. Measure freed primary capacity against extra promotion
and GPU load traffic. This mode is a capacity experiment, not presumed faster.

## Follow-ons and branch rules

Only add KVCR API changes for explicit invalidation or stronger residency
contracts if the vLLM adapter cannot provide the selected policy safely.
Treat local G2 placement, Guard pool placement, canonical/replicated layout,
query batching, and idle control polling as separate branches after the
no-secondary gap is explained. Each branch should carry its own tests and
metric deltas; model-affecting changes also need output validation and model
evaluations by the operator.

Stage 1 may show that a non-NUMA cause dominates; the NUMA option should stay
off until page placement is verified on the target host. The lineage policy
should likewise stay off until traces show that stranded descendants matter.
The policy code is available for review and cherry-picking, but runtime
correctness and performance remain to be established on a full vLLM/KVCR host.
