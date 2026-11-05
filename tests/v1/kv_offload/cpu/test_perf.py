# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for flags, independent DMA ranges and lossless compact pages.

Inputs are configuration/byte geometry; outputs are validated plans and order.
Guard against hidden enablement, unordered overlapping writes and omitted bytes.
Dependency-free unit tests are the cheapest level; device coverage is separate.
"""

import importlib.util
import itertools
import random
import sys
from dataclasses import fields
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "vllm/v1/kv_offload/cpu/perf.py"
spec = importlib.util.spec_from_file_location("cpu_perf_under_test", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Config = module.CPUPerfConfig
Scope = module.CopyScope


@pytest.mark.parametrize("values", list(itertools.product([False, True], repeat=5)))
def test_all_flag_combinations_are_independent(values):
    options = dict(zip((f.name for f in fields(Config)), values))
    assert Config.from_extra_config({"cpu_perf": options}) == Config(**options)
    assert Config.from_extra_config({}) == Config()


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        [],
        {"typo": True},
        {"early_load_ready": 1},
        {"compact_groups": "false"},
    ],
)
def test_invalid_flags_fail_closed(value):
    with pytest.raises(ValueError):
        Config.from_extra_config({"cpu_perf": value})


def test_precise_ranges_allow_read_sharing_but_preserve_all_write_dependencies():
    first = Scope(((0, 10), (100, 110)), ((1000, 1010),))
    assert first.extend(Scope(((0, 10),), ((1020, 1030),))) is not None
    assert first.extend(Scope(((50, 60),), ((1050, 1060),))) is not None
    for candidate in [
        Scope(((1005, 1015),), ((2000, 2010),)),
        Scope(((2000, 2010),), ((5, 15),)),
        Scope(((2000, 2010),), ((1005, 1015),)),
    ]:
        assert first.extend(candidate) is None


@pytest.mark.parametrize("world_size", [1, 2, 8, 16])
def test_compact_group_round_trip_preserves_padding_and_other_regions(world_size):
    # Contents are opaque: quantization scales, state and padding are retained.
    groups = [[(8, 7), (20, 5)], [(40, 16)]]
    plan = module.packed_group_regions(groups, 128, 16)
    assert sum(size for _, _, size in plan[0]) == 12
    rng = random.Random(42)
    for rank in range(world_size):
        for group, regions in zip(groups, plan):
            gpu = bytearray(rng.randbytes(128))
            cpu = bytearray(16)
            restored = bytearray([255] * 128)
            for start, offset, size in regions:
                cpu[offset : offset + size] = gpu[start : start + size]
                restored[start : start + size] = cpu[offset : offset + size]
            wanted = {i for start, size in group for i in range(start, start + size)}
            assert all(
                restored[i] == (gpu[i] if i in wanted else 255) for i in range(128)
            ), rank


@pytest.mark.parametrize(
    "groups,row,host",
    [
        ([[(1, 8)], [(25, 8)]], 32, 16),
        ([[(25, 8)]], 32, 16),
        ([[(0, 17)]], 32, 16),
        ([[]], 32, 16),
        ([[(0, 0)]], 32, 16),
    ],
)
def test_uncertified_compact_layout_is_rejected(groups, row, host):
    with pytest.raises(ValueError):
        module.packed_group_regions(groups, row, host)


def test_incremental_recency_matches_stable_global_order_with_ties_and_backfills():
    rng = random.Random(42)
    for _ in range(100):
        groups = [
            rng.sample(range(i * 30, (i + 1) * 30), rng.randrange(1, 25))
            for i in range(5)
        ]
        positions = {key: rng.randrange(16) for group in groups for key in group}
        baseline = [
            key for group in groups for key in sorted(group, key=positions.__getitem__)
        ]
        baseline.sort(key=positions.__getitem__)
        actual = module.request_mru_keys(groups, positions.get)
        assert list(actual) == list(reversed(baseline))
    assert module.request_mru_keys([[1, 2]], {1: 0}.get) is None


@pytest.mark.parametrize("capacity", [0, 1, 3, 16, 73])
def test_buddy_allocator_never_overlaps_live_reservations_and_coalesces(capacity):
    rng = random.Random(42)
    allocator = module.BuddyAllocator(capacity)
    live: dict[int, int] = {}
    for _ in range(1000):
        if live and rng.randrange(3) == 0:
            offset = rng.choice(list(live))
            allocator.release(offset)
            del live[offset]
        else:
            wanted = rng.randrange(1, 20)
            before = allocator.free_units
            offset = allocator.reserve(wanted)
            if offset is None:
                assert before == allocator.free_units
            else:
                size = allocator.size(offset)
                assert wanted <= size < 2 * wanted
                assert 0 <= offset < offset + size <= capacity
                assert all(
                    offset + size <= start or start + n <= offset
                    for start, n in live.items()
                )
                live[offset] = size
        assert allocator.free_units + sum(live.values()) == capacity
    for offset in live:
        allocator.release(offset)
    assert allocator.free_units == capacity
    assert not allocator.allocated


def test_compact_groups_allow_gpu_aliases_with_distinct_cache_key_ownership():
    plan = module.packed_group_regions([[(0, 16)], [(8, 8)]], 32, [16, 8])
    assert plan == (((0, 0, 16),), ((8, 0, 8),))
    assert module.packed_group_regions([[(0, 8)], []], 32, [8, 0]) == (((0, 0, 8),), ())
