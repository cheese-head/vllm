# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.perf import BuddyAllocator, request_mru_keys
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.metrics.cache_hit_source import CacheHitSource


@dataclass(slots=True)
class _RequestCacheAccess:
    """Cache keys observed by one request, grouped in prefix order."""

    owner: object
    cache_generation: int
    key_groups: dict[int, list[OffloadKey]] = field(default_factory=dict)
    seen_keys: set[OffloadKey] = field(default_factory=set)
    inserted_keys: set[OffloadKey] = field(default_factory=set)
    reused_keys: set[OffloadKey] = field(default_factory=set)
    store_miss_keys: set[OffloadKey] = field(default_factory=set)
    eviction_miss_keys: set[OffloadKey] = field(default_factory=set)
    finished: bool = False


class CPUOffloadingManager(OffloadingManager):
    """An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    chunk pool management, and the prepare_store/complete_store skeletons.
    Policy-specific chunk organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_chunks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        incremental_recency: bool = False,
        compact_group_units: dict[int, int] | None = None,
    ):
        self.medium: Medium = Medium.CPU
        self._num_chunks: int = num_chunks
        self._num_allocated_chunks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_chunks)
        if incremental_recency and type(self._policy) is not LRUCachePolicy:
            raise ValueError(
                "incremental_recency currently requires the built-in lru policy"
            )
        self._incremental_recency = incremental_recency
        if compact_group_units is not None and type(self._policy) is not LRUCachePolicy:
            raise ValueError("compact_groups requires the built-in lru policy")
        self._compact_group_units = compact_group_units
        if compact_group_units is not None and any(
            type(v) is not int or v <= 0 for v in compact_group_units.values()
        ):
            raise ValueError("Compact group reservations must be positive page counts")
        self._slot_allocator = (
            BuddyAllocator(num_chunks) if compact_group_units is not None else None
        )
        self._compact_read_units = self._compact_write_units = 0
        # Track the number of chunks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_chunks: int = 0
        # Track chunks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_chunks: int = 0
        self._store_allocation_failures = 0
        self._evicted_chunks = 0
        # Exact, bounded evidence that a key was previously resident here.
        self._recent_evictions: OrderedDict[OffloadKey, None] = OrderedDict()
        self._max_recent_evictions = 64_000
        self._misses_after_eviction = 0
        self._invalidated_chunks = 0
        self._pending_invalidations: dict[OffloadKey, tuple[int, int]] = {}
        self._residency_generations: dict[OffloadKey, int] = {}
        self._next_residency_generation = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []
        self._cache_generation = 0

        # Number of chunk references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # --- chunk pool ---

    def _get_num_free_chunks(self) -> int:
        if self._slot_allocator is not None:
            return self._slot_allocator.free_units
        return len(self._free_list) + self._num_chunks - self._num_allocated_chunks

    def resident_keys(self) -> set[OffloadKey]:
        chunks = getattr(self._policy, "chunks", None)
        if chunks is None:
            raise ValueError("This cache policy does not expose resident keys")
        return set(chunks)

    def evictable_keys(self) -> list[OffloadKey]:
        chunks = getattr(self._policy, "chunks", None)
        if chunks is None:
            raise ValueError("This cache policy does not expose evictable keys")
        return [key for key, chunk in chunks.items() if chunk.ref_cnt == 0]

    def residency_generation(self, key: OffloadKey) -> int:
        return self._residency_generations[key]

    def _allocate_chunks(self, keys: list[OffloadKey]) -> list[ChunkStatus]:
        num_fresh = min(len(keys), self._num_chunks - self._num_allocated_chunks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh chunks
        chunks: list[ChunkStatus] = []
        for _ in range(num_fresh):
            chunks.append(ChunkStatus(self._num_allocated_chunks))
            self._num_allocated_chunks += 1

        # allocate reused chunks
        for _ in range(num_reused):
            chunks.append(ChunkStatus(self._free_list.pop()))
        return chunks

    def _free_chunk(self, chunk: ChunkStatus) -> None:
        if self._slot_allocator is not None:
            assert chunk.ref_cnt <= 0, "Cannot free a compact source still being read"
            if chunk.ref_cnt < 0:
                self._compact_write_units -= self._slot_allocator.size(chunk.chunk_id)
            self._slot_allocator.release(chunk.chunk_id)
            self._num_allocated_chunks = (
                self._num_chunks - self._slot_allocator.free_units
            )
            return
        self._free_list.append(chunk.chunk_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        chunks: Iterable[ChunkStatus],
    ) -> CPULoadStoreSpec:
        ids = [chunk.chunk_id for chunk in chunks]
        return (
            CPULoadStoreSpec.from_page_offsets(ids)
            if self._slot_allocator is not None
            else CPULoadStoreSpec(ids)
        )

    def _evict_store_chunks(self, count, protected):
        evicted = self._policy.evict(count, protected)
        if evicted is None:
            return None
        self._num_evictable_cache_chunks -= len(evicted)
        result = []
        for key, chunk in evicted:
            self._free_chunk(chunk)
            self._pending_invalidations.pop(key, None)
            self._residency_generations.pop(key, None)
            result.append(key)
            self._recent_evictions[key] = None
            self._recent_evictions.move_to_end(key)
            if len(self._recent_evictions) > self._max_recent_evictions:
                self._recent_evictions.popitem(last=False)
        self._evicted_chunks += len(evicted)
        return result

    def _reserve_compact(self, keys, protected):
        allocator = self._slot_allocator
        assert allocator is not None and self._compact_group_units is not None
        sizes = [self._compact_group_units[get_offload_group_idx(key)] for key in keys]
        needed = sum(1 << (size - 1).bit_length() for size in sizes)
        protected_units = sum(
            allocator.size(chunk.chunk_id)
            for key in protected
            if (chunk := self._policy.get(key)) is not None and chunk.ref_cnt == 0
        )
        available = (
            self._num_chunks
            - self._compact_read_units
            - self._compact_write_units
            - protected_units
        )
        if needed > available:
            return None
        chunks: list[ChunkStatus | None] = [None] * len(keys)
        removed: list[OffloadKey] = []
        for index in sorted(range(len(keys)), key=lambda i: sizes[i], reverse=True):
            offset = allocator.reserve(sizes[index])
            while offset is None:
                evicted = self._evict_store_chunks(1, protected)
                if evicted is None:
                    for chunk in chunks:
                        if chunk is not None:
                            allocator.release(chunk.chunk_id)
                    self._num_allocated_chunks = self._num_chunks - allocator.free_units
                    if removed and self.events is not None:
                        self.events.append(
                            OffloadingEvent(
                                keys=removed, medium=self.medium, removed=True
                            )
                        )
                    return None
                removed.extend(evicted)
                offset = allocator.reserve(sizes[index])
            chunks[index] = ChunkStatus(offset)
            self._num_allocated_chunks = self._num_chunks - allocator.free_units
        return chunks, removed

    def _record_accesses(self, keys: Collection[OffloadKey]) -> None:
        """Record an offer without evicting its tracked candidates."""
        assert self.counts is not None
        protected: set[OffloadKey] = set()
        for key in keys:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
                protected.add(key)

        num_unprotected = len(self.counts) - len(protected)
        for key in keys:
            if key in self.counts:
                continue
            if len(self.counts) >= self.max_tracker_size:
                if num_unprotected == 0:
                    continue
                self.counts.popitem(last=False)
                num_unprotected -= 1
            self.counts[key] = 1

    def _get_request_cache_access(self, req_context: ReqContext) -> _RequestCacheAccess:
        state = req_context.get_state(_RequestCacheAccess)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
        ):
            state = _RequestCacheAccess(
                owner=self, cache_generation=self._cache_generation
            )
            req_context.set_state(state)
        return state

    def _record_request_cache_access(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        inserted_keys: Iterable[OffloadKey] = (),
        reused_keys: Iterable[OffloadKey] = (),
    ) -> None:
        state = self._get_request_cache_access(req_context)
        for key in keys:
            if key in state.seen_keys:
                continue
            assert not state.finished, (
                "New cache keys observed after request finalization"
            )
            group_idx = get_offload_group_idx(key)
            state.key_groups.setdefault(group_idx, []).append(key)
            state.seen_keys.add(key)
        state.inserted_keys.update(inserted_keys)
        # Re-reading a chunk inserted by this request is an internal transfer
        # (for example, a tiering cascade), not a second cache access.
        state.reused_keys.update(
            key for key in reused_keys if key not in state.inserted_keys
        )

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._get_request_cache_access(req_context)
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        chunk = self._policy.get(key)
        if chunk is None:
            if key in self._recent_evictions:
                state = self._get_request_cache_access(req_context)
                if key not in state.eviction_miss_keys:
                    state.eviction_miss_keys.add(key)
                    self._misses_after_eviction += 1
            return LookupResult.MISS
        if not chunk.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        return CacheHitSource.HOST

    def was_recently_evicted(self, key: OffloadKey) -> bool:
        """Whether this exact key is in the bounded CPU eviction history."""
        return key in self._recent_evictions

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        return self._prepare_load(keys, req_context, record_access=True)

    def _prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        record_access: bool,
    ) -> LoadStoreSpec:
        chunks = []
        for key in keys:
            chunk = self._policy.get(key)
            assert chunk is not None, f"Chunk {key!r} not found in cache"
            assert chunk.is_ready, f"Chunk {key!r} is not ready for reading"
            if chunk.ref_cnt == 0:
                if self._slot_allocator is not None:
                    self._compact_read_units += self._slot_allocator.size(
                        chunk.chunk_id
                    )
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_chunks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_chunks >= 0
            chunk.ref_cnt += 1
            chunks.append(chunk)
        if record_access:
            self._record_request_cache_access(keys, req_context, reused_keys=keys)
        return self._get_load_store_spec(keys, chunks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            chunk = self._policy.get(key)
            assert chunk is not None, f"Chunk {key!r} not found"
            assert chunk.ref_cnt > 0, f"Chunk {key!r} ref_cnt is already 0"
            chunk.ref_cnt -= 1
            if chunk.ref_cnt == 0:
                if self._slot_allocator is not None:
                    self._compact_read_units -= self._slot_allocator.size(
                        chunk.chunk_id
                    )
                self._num_evictable_cache_chunks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)
                expected = self._pending_invalidations.get(key)
                if expected is not None:
                    self.invalidate_if_unpinned(key, *expected)

    def invalidate_if_unpinned(
        self, key: OffloadKey, chunk_id: int, generation: int
    ) -> bool:
        """Retire a confirmed primary copy only if its source slot still matches."""
        chunk = self._policy.get(key)
        if (
            chunk is None
            or chunk.chunk_id != chunk_id
            or self._residency_generations.get(key) != generation
        ):
            self._pending_invalidations.pop(key, None)
            return False
        if not chunk.is_ready or chunk.ref_cnt > 0:
            self._pending_invalidations[key] = (chunk_id, generation)
            return False
        self._pending_invalidations.pop(key, None)
        self._residency_generations.pop(key, None)
        self._policy.remove(key)
        self._num_evictable_cache_chunks -= 1
        self._invalidated_chunks += 1
        self._free_chunk(chunk)
        if self.events is not None:
            self.events.append(
                OffloadingEvent(keys=[key], medium=self.medium, removed=True)
            )
        return True

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        protected_keys: Collection[OffloadKey] = (),
    ) -> PrepareStoreOutput | None:
        keys = (
            list(dict.fromkeys(keys))
            if self._slot_allocator is not None
            else list(keys)
        )
        if self.counts is not None:
            self._record_accesses(keys)
        # Partition the original offer before admission filtering. Ready
        # resident chunks are request reuses even if their tracker entry was
        # aged out; the threshold applies only to new store candidates.
        keys_to_store: list[OffloadKey] = []
        ready_existing_keys: list[OffloadKey] = []
        for key in keys:
            chunk = self._policy.get(key)
            if chunk is None:
                keys_to_store.append(key)
            else:
                if chunk.is_ready:
                    ready_existing_keys.append(key)

        if self.counts is not None:
            num_store_candidates = len(keys_to_store)
            keys_to_store = [
                key
                for key in keys_to_store
                if self.counts.get(key, 0) >= self.store_threshold
            ]
            self.stores_skipped_in_current_batch += num_store_candidates - len(
                keys_to_store
            )

        for key in keys:
            self._policy.record_parent(key, req_context.get_offload_key_parent(key))
        keys_to_store = self._policy.filter_admissible(keys_to_store)

        state = self._get_request_cache_access(req_context)
        new_store_misses = [
            key for key in keys_to_store if key not in state.store_miss_keys
        ]
        if new_store_misses:
            # ARC learns from B1/B2 before insert() removes the ghost entry.
            # Deduplication makes this one policy observation per request.
            self._policy.on_store_miss(new_store_misses, req_context)
            state.store_miss_keys.update(new_store_misses)

        recorded_keys = set(keys_to_store)
        recorded_keys.update(ready_existing_keys)
        # Record once before any allocation-related early return. Store
        # candidates are classified as insertions only after allocation succeeds.
        self._record_request_cache_access(
            (key for key in keys if key in recorded_keys),
            req_context,
            reused_keys=ready_existing_keys,
        )

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        chunks = None
        to_evict: list[OffloadKey] = []
        protected = set(keys) | set(protected_keys)
        if self._slot_allocator is not None:
            reserved = self._reserve_compact(keys_to_store, protected)
            if reserved is None:
                self._store_allocation_failures += 1
                return None
            chunks, to_evict = reserved
        else:
            num_chunks_to_evict = len(keys_to_store) - self._get_num_free_chunks()
            if num_chunks_to_evict > 0:
                if num_chunks_to_evict > self._num_evictable_cache_chunks:
                    self._store_allocation_failures += 1
                    return None
                to_evict = self._evict_store_chunks(num_chunks_to_evict, protected)
                if to_evict is None:
                    self._store_allocation_failures += 1
                    return None

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        if chunks is None:
            chunks = self._allocate_chunks(keys_to_store)
        assert len(chunks) == len(keys_to_store), (
            "Chunk pool did not allocate the expected number of chunks"
        )

        for key, chunk in zip(keys_to_store, chunks):
            self._policy.insert(key, chunk)
            self._next_residency_generation += 1
            self._residency_generations[key] = self._next_residency_generation
            self._recent_evictions.pop(key, None)
        if self._slot_allocator is not None:
            self._compact_write_units += sum(
                self._slot_allocator.size(c.chunk_id) for c in chunks
            )
        self._num_write_pending_chunks += len(keys_to_store)
        state.inserted_keys.update(keys_to_store)

        # build store specs for allocated chunks
        store_spec = self._get_load_store_spec(keys_to_store, chunks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                chunk = self._policy.get(key)
                if chunk is not None and not chunk.is_ready:
                    if self._slot_allocator is not None:
                        self._compact_write_units -= self._slot_allocator.size(
                            chunk.chunk_id
                        )
                    chunk.ref_cnt = 0
                    self._num_write_pending_chunks -= 1
                    self._num_evictable_cache_chunks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in reversed(list(keys)):
                chunk = self._policy.get(key)
                if chunk is not None and not chunk.is_ready:
                    self._num_write_pending_chunks -= 1
                    self._policy.remove(key)
                    self._pending_invalidations.pop(key, None)
                    self._residency_generations.pop(key, None)
                    self._free_chunk(chunk)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        state = req_context.get_state(_RequestCacheAccess)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
            or state.finished
        ):
            return
        state.finished = True
        if self._incremental_recency:
            keys = request_mru_keys(
                [state.key_groups[i] for i in sorted(state.key_groups)],
                req_context.get_offload_key_position,
            )
            if keys is not None:
                assert isinstance(self._policy, LRUCachePolicy)
                self._policy.on_request_finished_mru(keys)
                return
        key_groups = []
        for group_idx in sorted(state.key_groups):
            keys = state.key_groups[group_idx]
            positions = {
                key: position
                for key in keys
                if (position := req_context.get_offload_key_position(key)) is not None
            }
            if len(positions) == len(keys):
                keys = sorted(keys, key=positions.__getitem__)
            key_groups.append(tuple(keys))
        self._policy.on_request_finished(
            tuple(key_groups),
            state.inserted_keys - state.reused_keys,
            state.reused_keys,
            req_context,
        )

    @override
    def reset_cache(self) -> None:
        # Clear ALL chunks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload chunk IDs.
        self._policy.clear()
        if self._slot_allocator is not None:
            self._slot_allocator.clear()
            self._compact_read_units = self._compact_write_units = 0
        self._cache_generation += 1
        self._num_evictable_cache_chunks = 0
        self._num_write_pending_chunks = 0

        self._free_list.clear()
        self._num_allocated_chunks = 0
        self._recent_evictions.clear()
        self._pending_invalidations.clear()
        self._residency_generations.clear()

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        free = self._get_num_free_chunks()
        resident = self._num_allocated_chunks - len(self._free_list)
        evictable = self._num_evictable_cache_chunks
        write_pending = self._num_write_pending_chunks
        if self._slot_allocator is not None:
            evictable = resident - self._compact_read_units - self._compact_write_units
            write_pending = self._compact_write_units
        for state, count in (
            ("free", free),
            ("evictable", evictable),
            ("write_pending", write_pending),
            (
                "read_pinned",
                resident - evictable - write_pending,
            ),
        ):
            stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_SLOTS, count, (state,))
        if self._store_allocation_failures:
            stats.increase_counter(
                CPUOffloadingMetrics.CPU_STORE_ALLOCATION_FAILURES,
                self._store_allocation_failures,
            )
            self._store_allocation_failures = 0
        if self._evicted_chunks:
            stats.increase_counter(
                CPUOffloadingMetrics.CPU_EVICTED_CHUNKS, self._evicted_chunks
            )
            self._evicted_chunks = 0
        if self._invalidated_chunks:
            stats.increase_counter(
                CPUOffloadingMetrics.CPU_INVALIDATED_CHUNKS,
                self._invalidated_chunks,
            )
            self._invalidated_chunks = 0
        if self._misses_after_eviction:
            stats.increase_counter(
                CPUOffloadingMetrics.CPU_MISSES_AFTER_EVICTION,
                self._misses_after_eviction,
            )
            self._misses_after_eviction = 0

        # Compute cache usage.
        num_used = self._num_allocated_chunks - len(self._free_list) - evictable
        usage = num_used / self._num_chunks if self._num_chunks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = write_pending / self._num_chunks if self._num_chunks > 0 else 0.0
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
