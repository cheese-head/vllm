# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in native CPU experiments and dependency-free layout contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from heapq import merge
from typing import Any


@dataclass(frozen=True)
class CPUPerfConfig:
    early_load_ready: bool = False
    incremental_store: bool = False
    precise_batching: bool = False
    incremental_recency: bool = False
    compact_groups: bool = False

    @classmethod
    def from_extra_config(cls, extra: Mapping[str, Any]) -> "CPUPerfConfig":
        options = extra.get("cpu_perf", {})
        if not isinstance(options, Mapping):
            raise ValueError("cpu_perf must be a mapping")
        unknown = options.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown cpu_perf options: {sorted(unknown)}")
        if any(type(v) is not bool for v in options.values()):
            raise ValueError("cpu_perf flags must be boolean")
        return cls(**options)


class BuddyAllocator:
    """Bounded page allocator; allocated extents are released only by their owner."""

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("Negative compact capacity")
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.free: list[set[int]] = [
            set() for _ in range(max(1, self.capacity.bit_length()))
        ]
        self.allocated: dict[int, int] = {}
        offset = 0
        remaining = self.capacity
        while remaining:
            order = remaining.bit_length() - 1
            self.free[order].add(offset)
            offset += 1 << order
            remaining -= 1 << order
        self.free_units = self.capacity

    def reserve(self, units: int) -> int | None:
        if units <= 0:
            raise ValueError("Compact allocations must be positive")
        wanted = (units - 1).bit_length()
        for order in range(wanted, len(self.free)):
            if not self.free[order]:
                continue
            offset = self.free[order].pop()
            while order > wanted:
                order -= 1
                self.free[order].add(offset + (1 << order))
            self.allocated[offset] = wanted
            self.free_units -= 1 << wanted
            return offset
        return None

    def size(self, offset: int) -> int:
        return 1 << self.allocated[offset]

    def release(self, offset: int):
        order = self.allocated.pop(offset)
        self.free_units += 1 << order
        while order + 1 < len(self.free):
            buddy = offset ^ (1 << order)
            if buddy not in self.free[order]:
                break
            self.free[order].remove(buddy)
            offset = min(offset, buddy)
            order += 1
        self.free[order].add(offset)


def merge_intervals(intervals, *, presorted=False):
    result: list[tuple[int, int]] = []
    for start, end in intervals if presorted else sorted(intervals):
        if not 0 <= start < end <= 2**64:
            raise ValueError("Invalid copy address interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return tuple(result)


def overlaps(left, right):
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        if a < d and c < b:
            return True
        if b <= c:
            i += 1
        else:
            j += 1
    return False


@dataclass(frozen=True)
class CopyScope:
    reads: tuple[tuple[int, int], ...]
    writes: tuple[tuple[int, int], ...]

    def extend(self, candidate: "CopyScope") -> "CopyScope | None":
        if (
            overlaps(self.writes, candidate.reads)
            or overlaps(self.reads, candidate.writes)
            or overlaps(self.writes, candidate.writes)
        ):
            return None
        return CopyScope(
            merge_intervals(merge(self.reads, candidate.reads), presorted=True),
            merge_intervals(merge(self.writes, candidate.writes), presorted=True),
        )


def request_mru_keys(key_groups, position):
    """Merge per-group order without sorting the entire prefix twice.

    Missing positions retain the legacy policy path. Equal-position ties must
    reverse both the legacy group order and its stable within-group order.
    """
    ordered = []
    for keys in key_groups:
        group = [(position(key), key) for key in keys]
        if any(p is None for p, _ in group):
            return None
        if any(a[0] > b[0] for a, b in zip(group, group[1:])):
            group.sort(key=lambda item: item[0])
        ordered.append(reversed(group))
    return (
        key
        for _, key in merge(*reversed(ordered), key=lambda item: item[0], reverse=True)
    )


def packed_group_regions(
    groups: Sequence[Sequence[tuple[int, int]]],
    row_bytes: int,
    host_bytes: int | Sequence[int],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Certify padded layer intervals and map each group into a shared slot.

    Each result is (GPU byte offset, host byte offset, byte length). Slots
    belong to one OffloadKey/group at a time. GPU groups may alias: vLLM's
    global block pool assigns each live block ID to one group. No model-name
    or dtype-specific assumptions are made about page contents.
    """
    budgets = (
        [host_bytes] * len(groups) if isinstance(host_bytes, int) else list(host_bytes)
    )
    if (
        row_bytes <= 0
        or not groups
        or not any(groups)
        or len(budgets) != len(groups)
        or any(b < 0 for b in budgets)
    ):
        raise ValueError("Empty compact layout")
    result: list[tuple[tuple[int, int, int], ...]] = []
    for group_id, group in enumerate(groups):
        merged = merge_intervals((offset, offset + size) for offset, size in group)
        if not merged:
            result.append(())
            continue
        if merged[-1][1] > row_bytes:
            raise ValueError("Compact group lies outside packed row")
        host_offset = 0
        regions = []
        for start, end in merged:
            regions.append((start, host_offset, end - start))
            host_offset += end - start
        if host_offset > budgets[group_id]:
            raise ValueError(f"Compact group {group_id} exceeds scheduler slot budget")
        result.append(tuple(regions))
    return tuple(result)
