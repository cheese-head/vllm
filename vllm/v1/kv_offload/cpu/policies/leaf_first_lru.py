# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional prefix-closed CPU cache replacement for full-attention chunks."""

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class LeafFirstLRUCachePolicy(LRUCachePolicy):
    """Evict the oldest eligible leaf before any resident ancestor."""

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self._known_parents: dict[OffloadKey, OffloadKey] = {}
        self._resident_parents: dict[OffloadKey, OffloadKey] = {}
        self._children: dict[OffloadKey, set[OffloadKey]] = {}

    @override
    def record_parent(self, key: OffloadKey, parent: OffloadKey | None) -> None:
        if parent is None:
            return
        existing = self._known_parents.get(key)
        if existing is not None and existing != parent:
            raise ValueError("One offload key has conflicting parent keys")
        self._known_parents[key] = parent
        if (
            key in self.chunks
            and key not in self._resident_parents
            and parent in self.chunks
        ):
            self._resident_parents[key] = parent
            self._children.setdefault(parent, set()).add(key)

    @override
    def filter_admissible(self, keys: list[OffloadKey]) -> list[OffloadKey]:
        selected: set[OffloadKey] = set()
        admitted: list[OffloadKey] = []
        for key in keys:
            parent = self._known_parents.get(key)
            parent_chunk = self.chunks.get(parent) if parent is not None else None
            if (
                parent is None
                or (parent_chunk is not None and parent_chunk.is_ready)
                or parent in selected
            ):
                admitted.append(key)
                selected.add(key)
            else:
                self._known_parents.pop(key, None)
        return admitted

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        parent = self._known_parents.get(key)
        if parent is not None:
            if parent not in self.chunks:
                raise RuntimeError("Cannot insert a child without its CPU parent")
            self._resident_parents[key] = parent
            self._children.setdefault(parent, set()).add(key)
        super().insert(key, chunk)

    @override
    def remove(self, key: OffloadKey) -> None:
        parent = self._resident_parents.pop(key, None)
        if parent is not None:
            children = self._children[parent]
            children.remove(key)
            if not children:
                del self._children[parent]
        if self._children.get(key):
            raise RuntimeError("Cannot remove a CPU parent with resident children")
        self._children.pop(key, None)
        self._known_parents.pop(key, None)
        super().remove(key)

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        selected: list[OffloadKey] = []
        selected_set: set[OffloadKey] = set()
        candidates = self._evictable - protected
        while len(selected) < n:
            leaves = (
                key
                for key in candidates - selected_set
                if self._children.get(key, set()) <= selected_set
            )
            leaf = min(leaves, key=self._ranks.__getitem__, default=None)
            if leaf is None:
                return None
            selected.append(leaf)
            selected_set.add(leaf)
        result = [(key, self.chunks[key]) for key in selected]
        for key in selected:
            self.remove(key)
        self._maybe_compact_heap()
        return result

    @override
    def clear(self) -> None:
        super().clear()
        self._known_parents.clear()
        self._resident_parents.clear()
        self._children.clear()
