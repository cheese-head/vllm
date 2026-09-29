# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread-local NUMA policy for first-touch CPU offload allocation."""

import ctypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from vllm.utils.numa_utils import get_libnuma

_MPOL_BIND = 2
_BITS = ctypes.sizeof(ctypes.c_ulong) * 8


def _node_set(text: str) -> set[int]:
    nodes: set[int] = set()
    for part in text.strip().split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            nodes.update(range(first, last + 1))
        elif part:
            nodes.add(int(part))
    return nodes


def validate_numa_nodes(nodes: list[int]) -> None:
    """Reject unavailable and cpuset-disallowed memory nodes."""
    if not nodes:
        raise ValueError("At least one CPU offload NUMA node is required")
    for node in nodes:
        if node < 0:
            raise ValueError(f"Invalid NUMA node {node}")
        node_path = Path(f"/sys/devices/system/node/node{node}")
        if not node_path.is_dir() or not (node_path / "meminfo").exists():
            raise ValueError(f"NUMA node {node} is not online with memory")
    status = Path("/proc/self/status").read_text()
    allowed_line = next(
        (line for line in status.splitlines() if line.startswith("Mems_allowed_list:")),
        None,
    )
    if allowed_line is None:
        raise RuntimeError("Cannot determine cpuset-allowed NUMA nodes")
    allowed = _node_set(allowed_line.partition(":")[2])
    if denied := set(nodes) - allowed:
        raise ValueError(f"CPU offload NUMA nodes excluded by cpuset: {denied}")


@contextmanager
def bind_memory_to_node(node: int) -> Iterator[None]:
    """Temporarily bind the calling thread's new pages to one node."""
    libnuma = get_libnuma()
    if libnuma is None:
        raise RuntimeError("libnuma is required for CPU offload NUMA placement")
    libnuma = ctypes.CDLL(libnuma._name, use_errno=True)
    max_node = max(node + 1, os.cpu_count() or 1, 64)
    word_count = (max_node + _BITS - 1) // _BITS
    mask_type = ctypes.c_ulong * word_count
    old_mask = mask_type()
    old_mode = ctypes.c_int()
    libnuma.get_mempolicy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    libnuma.get_mempolicy.restype = ctypes.c_int
    libnuma.set_mempolicy.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
    ]
    libnuma.set_mempolicy.restype = ctypes.c_int
    if libnuma.get_mempolicy(ctypes.byref(old_mode), old_mask, max_node, None, 0):
        raise OSError(ctypes.get_errno(), "get_mempolicy failed")
    if old_mode.value == _MPOL_BIND:
        old_nodes = {
            n for n in range(max_node) if old_mask[n // _BITS] & (1 << (n % _BITS))
        }
        if node not in old_nodes:
            raise ValueError(
                f"CPU offload NUMA node {node} conflicts with process membind "
                f"{sorted(old_nodes)}"
            )
    new_mask = mask_type()
    new_mask[node // _BITS] = 1 << (node % _BITS)
    if libnuma.set_mempolicy(_MPOL_BIND, new_mask, max_node):
        raise OSError(ctypes.get_errno(), "set_mempolicy failed")
    try:
        yield
    finally:
        if libnuma.set_mempolicy(old_mode.value, old_mask, max_node):
            raise OSError(ctypes.get_errno(), "restoring memory policy failed")
