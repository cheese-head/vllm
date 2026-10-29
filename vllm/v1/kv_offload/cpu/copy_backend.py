# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA adapter for native copies; uses the existing canonical/direct fillers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.cpu.copy_executor import (
    CopyChunk,
    CopyExecutorConfig,
    PhysicalCopyStats,
)


@dataclass
class _Descriptors:
    src: np.ndarray
    dst: np.ndarray
    sizes: np.ndarray

    def envelopes(self):
        if not len(self.sizes):
            return ()
        return tuple(
            (int(a.min()), int((a + self.sizes).max())) for a in (self.src, self.dst)
        )


@dataclass(eq=False)
class _Ticket:
    stream: Any
    start: Any
    end: Any
    buffers: tuple
    nbytes: int


class CudaCopyBackend:
    def __init__(self, store_handler, load_handler, allocate_buffers):
        self.handlers = {"store": store_handler, "load": load_handler}
        self.device = store_handler.src_tensors[0].device
        self.allocate_buffers = allocate_buffers
        self.streams: dict[str, Any] = {}
        self._tickets: set[_Ticket] = set()
        self._buffers: list[tuple] = []
        self._events: list[Any] = []
        self._closed = False

    def setup(self):
        current_platform.set_device(self.device)
        self.streams = {d: current_platform.Stream() for d in self.handlers}

    @staticmethod
    def ready(producer):
        return producer.query()

    @staticmethod
    def extend_batch(scope, candidate):
        # Conservative envelopes reject false positives rather than merge
        # possibly dependent copies. All four address combinations matter.
        ranges = candidate.data.envelopes()
        if scope is None or not scope:
            return ranges
        if not ranges:
            return scope
        independent = all(
            end <= a or b <= start for start, end in scope for a, b in ranges
        )
        if not independent:
            return None
        # Fixed-size conservative envelopes keep append work linear in batch
        # descriptors, rather than rescanning all previous jobs per append.
        return tuple((min(a, c), max(b, d)) for (a, b), (c, d) in zip(scope, ranges))

    def chunks(self, payload, config: CopyExecutorConfig):
        direction, src_spec, dst_spec = payload
        handler = self.handlers[direction]
        gpu_spec = src_spec if direction == "store" else dst_spec
        src = src_spec.block_ids
        dst = dst_spec.block_ids
        if any(
            ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            for ids in (src, dst)
        ):
            raise ValueError("Native copy block IDs must be integer vectors")
        groups = gpu_spec.group_sizes
        indices = gpu_spec.block_indices
        if (
            len(groups) != len(handler.layer_refs_per_group)
            or len(indices) != len(groups)
            or sum(groups) != len(gpu_spec.block_ids)
            or any(n < 0 for n in groups)
        ):
            raise ValueError("Native copy group geometry mismatch")
        nonempty = [i for i, n in enumerate(groups) if n]
        if not nonempty:
            if len(src) or len(dst):
                raise ValueError("Empty groups with nonempty block IDs")
            empty = np.empty(0, dtype=np.uint64)
            yield CopyChunk(_Descriptors(empty, empty, empty), 0, 0, True)
            return
        src_offset = dst_offset = 0
        for group, (size, index) in enumerate(zip(groups, indices)):
            if size == 0:
                continue
            if index < 0:
                raise ValueError("Negative native copy logical block index")
            src_skip = index % handler.src_blocks_per_chunk
            dst_skip = index % handler.dst_blocks_per_chunk
            src_count = cdiv(size + src_skip, handler.src_blocks_per_chunk)
            dst_count = cdiv(size + dst_skip, handler.dst_blocks_per_chunk)
            if src_offset + src_count > len(src) or dst_offset + dst_count > len(dst):
                raise ValueError("Native copy block IDs shorter than group geometry")
            refs = handler.layer_refs_per_group[group]
            for ids, tensors, offset, count in (
                (src, handler.src_tensors, src_offset, src_count),
                (dst, handler.dst_tensors, dst_offset, dst_count),
            ):
                group_ids = ids[offset : offset + count]
                limit = min((tensors[r.tensor_idx].shape[0] for r in refs), default=0)
                if len(group_ids) and (group_ids.min() < 0 or group_ids.max() >= limit):
                    raise ValueError("Native copy block ID outside its tensor")
            plans = handler._canonical_copy_plans
            per_block = (
                sum(p.num_frags for p in plans[group])
                if plans is not None
                else len(refs)
            )
            bytes_per_block = (
                sum(p.total_bytes for p in plans[group])
                if plans is not None
                else sum(r.page_size_bytes for r in refs)
            )
            # One logical block is atomic at the layout adapter. Targets may
            # be exceeded only by that single model-defined block.
            blocks = max(
                1,
                min(
                    config.max_batch_descriptors // max(1, per_block),
                    config.target_batch_bytes // max(1, bytes_per_block),
                ),
            )
            for offset in range(0, size, blocks):
                count = min(blocks, size - offset)
                src_start, source_skip = divmod(
                    src_skip + offset, handler.src_blocks_per_chunk
                )
                dst_start, target_skip = divmod(
                    dst_skip + offset, handler.dst_blocks_per_chunk
                )
                source = src[
                    src_offset + src_start : src_offset
                    + src_start
                    + cdiv(count + source_skip, handler.src_blocks_per_chunk)
                ]
                target = dst[
                    dst_offset + dst_start : dst_offset
                    + dst_start
                    + cdiv(count + target_skip, handler.dst_blocks_per_chunk)
                ]
                arrays = [np.empty(count * per_block, dtype=np.int64) for _ in range(3)]
                used, nbytes = handler._fill_group_ops(
                    group,
                    source,
                    target,
                    count,
                    source_skip,
                    target_skip,
                    *arrays,
                    0,
                )
                data = _Descriptors(*(a[:used].view(np.uint64) for a in arrays))
                final = group == nonempty[-1] and offset + count == size
                if final and (
                    src_offset + src_count != len(src)
                    or dst_offset + dst_count != len(dst)
                ):
                    raise ValueError("Extra native copy block IDs")
                if used or final:
                    yield CopyChunk(data, used, nbytes, final)
            src_offset += src_count
            dst_offset += dst_count

    def launch(self, direction, parts, producers):
        if self._closed:
            raise RuntimeError("Native copy backend is closed")
        count = sum(p.descriptors for p in parts)
        buffers = self._buffers.pop() if self._buffers else ()
        if not buffers or buffers[0].numel() < count:
            buffers = self.allocate_buffers(count)
        views = [b[:count] for b in buffers]
        arrays = [v.numpy().view(np.uint64) for v in views]
        offset = 0
        for part in parts:
            end = offset + part.descriptors
            for output, data in zip(
                arrays, (part.data.src, part.data.dst, part.data.sizes)
            ):
                output[offset:end] = data
            offset = end
        start = self._events.pop() if self._events else torch.Event(enable_timing=True)
        end = self._events.pop() if self._events else torch.Event(enable_timing=True)
        stream = self.streams[direction]
        ticket = _Ticket(stream, start, end, buffers, sum(p.nbytes for p in parts))
        # Retain buffers even if submission fails after accepting some DMA.
        self._tickets.add(ticket)
        for producer in producers:
            stream.wait_event(producer)
        with current_platform.stream(stream):
            start.record(stream)
            if count:
                self.handlers[direction]._swap_blocks_batch(
                    *views, is_src_access_order_any=direction == "load"
                )
            end.record(stream)
        return ticket

    @staticmethod
    def query(ticket):
        return ticket.end.query()

    def finish(self, ticket):
        seconds = ticket.start.elapsed_time(ticket.end) * 1e-3
        self._tickets.remove(ticket)
        self._buffers.append(ticket.buffers)
        self._events.extend((ticket.start, ticket.end))
        return PhysicalCopyStats(ticket.nbytes, seconds)

    def close(self):
        if self._closed:
            return
        # On any error, keep all buffers and views alive until this succeeds.
        for stream in self.streams.values():
            stream.synchronize()
        for handler in self.handlers.values():
            handler.shutdown()
        self._tickets.clear()
        self._buffers.clear()
        self._events.clear()
        self.streams.clear()
        self._closed = True
