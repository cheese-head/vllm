# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Descriptor equivalence across layouts; no CUDA execution is required."""

import ctypes
import threading
import time
from collections import deque
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalPageMapping,
    CopyRun,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.copy_backend import CudaCopyBackend
from vllm.v1.kv_offload.cpu.copy_executor import CopyChunk, CopyExecutorConfig
from vllm.v1.kv_offload.cpu.gpu_worker import (
    CPUOffloadingWorker,
    SingleDirectionOffloadingHandler,
    _build_copy_plan,
)


class Tensor:
    def __init__(self, base, width, stride):
        self.base = base
        self.shape = (1000, width)
        self.row_stride = stride
        self.device = "cuda:3"

    def data_ptr(self):
        return self.base

    def stride(self, dim):
        assert dim == 0
        return self.row_stride


def handler(direction, bpc, canonical, writer, writers=2):
    result = SingleDirectionOffloadingHandler.__new__(SingleDirectionOffloadingHandler)
    mapping = (
        CanonicalPageMapping(
            canonical_page_size_bytes=512,
            local_page_size_bytes=128,
            runs=(CopyRun(0, 32, 16, 4, 32, 64),),
            num_writers=writers,
            writer_index=writer,
            parallelism_agnostic=True,
        )
        if canonical
        else None
    )
    widths = [128, 128, 128] if canonical else [128, 64, 256]
    result.layer_refs_per_group = [
        [CanonicalKVCacheRef(0, 128, mapping)],
        [
            CanonicalKVCacheRef(1, widths[1], mapping),
            CanonicalKVCacheRef(2, widths[2], mapping),
        ],
        [
            CanonicalKVCacheRef(0, 128, mapping),
            CanonicalKVCacheRef(2, widths[2], mapping),
        ],
    ]
    gpu = [Tensor(2**40 + i * 2**24, w, w + 128) for i, w in enumerate(widths)]
    cpu_widths = [(512 if canonical else w) * bpc for w in widths]
    cpu = [Tensor(2**45 + i * 2**24, w, w + 4096) for i, w in enumerate(cpu_widths)]
    store = direction == "store"
    result.src_tensors = gpu if store else cpu
    result.dst_tensors = cpu if store else gpu
    result.src_blocks_per_chunk = 1 if store else bpc
    result.dst_blocks_per_chunk = bpc if store else 1
    result.gpu_to_cpu = store
    result._canonical_copy_plans = (
        [
            [_build_copy_plan(ref, store) for ref in refs]
            for refs in result.layer_refs_per_group
        ]
        if canonical
        else None
    )
    result._scratch_bases_src = np.empty(0, dtype=np.uint64)
    result._scratch_bases_dst = np.empty(0, dtype=np.uint64)
    result._fill_group_ops = (
        result._fill_canonical_ops if canonical else result._fill_direct_ops
    )
    return result


@pytest.mark.parametrize("direction", ["store", "load"])
@pytest.mark.parametrize("bpc", [1, 3])
@pytest.mark.parametrize(
    "canonical,writer,writers",
    [
        (False, 0, 1),
        (True, 0, 1),
        (True, 0, 2),
        (True, 1, 2),
        (True, 3, 4),
        (True, 7, 8),
    ],
)
@pytest.mark.parametrize(
    "groups,indices",
    [
        ([0, 3, 7], [0, 2, 4]),
        ([5, 0, 1], [1, 0, 2]),
        ([0, 0, 0], [0, 0, 0]),
    ],
)
def test_split_descriptors_match_unsplit_native_layout(
    direction,
    bpc,
    canonical,
    writer,
    writers,
    groups,
    indices,
):
    h = handler(direction, bpc, canonical, writer, writers)
    gpu_ids = [i + 20 for i in range(sum(groups))]
    cpu_count = [
        ((n + index % bpc + bpc - 1) // bpc) if n else 0
        for n, index in zip(groups, indices)
    ]
    cpu_ids = [i + 3 for i in range(sum(cpu_count))]
    gpu = GPULoadStoreSpec(gpu_ids, groups, indices)
    cpu = CPULoadStoreSpec(cpu_ids)
    source, target = (gpu, cpu) if direction == "store" else (cpu, gpu)
    backend = CudaCopyBackend.__new__(CudaCopyBackend)
    backend.handlers = {direction: h}
    chunks = list(
        backend.chunks(
            (direction, source, target),
            CopyExecutorConfig(max_batch_descriptors=3, target_batch_bytes=256),
        )
    )
    assert chunks[-1].final and sum(c.final for c in chunks) == 1
    actual = sorted(
        (int(s), int(d), int(n))
        for c in chunks
        for s, d, n in zip(c.data.src, c.data.dst, c.data.sizes)
    )
    # Whole-group fill uses the exact pre-existing native methods.
    expected: list[tuple[int, int, int]] = []
    src_offset = dst_offset = 0
    total_bytes = 0
    for group, (size, index) in enumerate(zip(groups, indices)):
        if not size:
            continue
        s_skip = index % h.src_blocks_per_chunk
        d_skip = index % h.dst_blocks_per_chunk
        ns = (size + s_skip + h.src_blocks_per_chunk - 1) // h.src_blocks_per_chunk
        nd = (size + d_skip + h.dst_blocks_per_chunk - 1) // h.dst_blocks_per_chunk
        output = [np.empty(size * 16, dtype=np.int64) for _ in range(3)]
        used, nbytes = h._fill_group_ops(
            group,
            source.block_ids[src_offset : src_offset + ns],
            target.block_ids[dst_offset : dst_offset + nd],
            size,
            s_skip,
            d_skip,
            *output,
            0,
        )
        expected.extend(zip(*(a[:used].tolist() for a in output)))
        total_bytes += nbytes
        src_offset += ns
        dst_offset += nd
    assert actual == sorted(expected)
    assert sum(c.nbytes for c in chunks) == total_bytes


def test_batch_rejects_aliasing_or_dependent_address_envelopes():
    from vllm.v1.kv_offload.cpu.copy_backend import _Descriptors

    def chunk(src, dst):
        a = [np.array([v], dtype=np.uint64) for v in (src, dst, 16)]
        return CopyChunk(_Descriptors(*a), 1, 16, True)

    first = chunk(100, 500)
    scope = CudaCopyBackend.extend_batch(None, first)
    assert CudaCopyBackend.extend_batch(scope, chunk(200, 600)) is not None
    assert CudaCopyBackend.extend_batch(scope, chunk(108, 600)) is None
    assert CudaCopyBackend.extend_batch(scope, chunk(200, 508)) is None
    assert CudaCopyBackend.extend_batch(scope, chunk(500, 600)) is None


def test_enqueue_freezes_spec_and_captures_the_calling_stream(monkeypatch):
    from vllm.v1.kv_offload.cpu import gpu_worker

    producer = MagicMock()
    stream = object()
    monkeypatch.setattr(gpu_worker.torch, "Event", lambda: producer)
    monkeypatch.setattr(gpu_worker.current_platform, "current_stream", lambda: stream)
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    worker._copy_executor = MagicMock()
    gpu = GPULoadStoreSpec([4, 5], [2], [0])
    cpu = CPULoadStoreSpec([1, 2])
    assert worker._enqueue(12, "store", gpu, cpu)
    producer.record.assert_called_once_with(stream)
    args = worker._copy_executor.submit.call_args.args
    _, snapshot_gpu, snapshot_cpu = args[2]
    gpu.block_ids[0] = 99
    cpu.block_ids[0] = 99
    gpu.group_sizes[0] = 0
    assert snapshot_gpu.block_ids.tolist() == [4, 5]
    assert snapshot_cpu.block_ids.tolist() == [1, 2]
    assert snapshot_gpu.group_sizes == (2,)
    assert not snapshot_gpu.block_ids.flags.writeable


def test_unquiesced_shutdown_does_not_unmap_host_memory():
    from vllm.v1.kv_offload.cpu import gpu_worker

    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    worker._mmap_region = MagicMock()
    worker._copy_executor = MagicMock(quiescent=False)
    worker._copy_executor.close.side_effect = TimeoutError("still writing")
    with pytest.raises(TimeoutError):
        worker.shutdown()
    worker._mmap_region.cleanup.assert_not_called()
    assert worker in gpu_worker._RETAINED_NATIVE_CPU_WORKERS
    gpu_worker._RETAINED_NATIVE_CPU_WORKERS.remove(worker)


def test_non_cuda_executor_is_rejected_before_allocation(monkeypatch):
    from vllm.v1.kv_offload.cpu import gpu_worker

    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        CPUOffloadingWorker(
            kv_caches=None,
            blocks_per_chunk=1,
            num_cpu_chunks=1,
            copy_executor_config=CopyExecutorConfig(),
        )


@pytest.mark.parametrize("bad_id", [-1, 1000])
@pytest.mark.parametrize("direction", ["load", "store"])
def test_invalid_block_ids_fail_before_device_submission(bad_id, direction):
    h = handler(direction, 1, False, 0)
    backend = CudaCopyBackend.__new__(CudaCopyBackend)
    backend.handlers = {direction: h}
    gpu = GPULoadStoreSpec([bad_id], [1, 0, 0], [0, 0, 0])
    cpu = CPULoadStoreSpec([1])
    source, target = (gpu, cpu) if direction == "store" else (cpu, gpu)
    with pytest.raises(ValueError, match="outside its tensor"):
        list(backend.chunks((direction, source, target), CopyExecutorConfig()))


def test_physical_statistics_and_logical_acknowledgments_are_independent():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.kv_offload.base import TransferResult

    worker = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    worker.worker = MagicMock()
    worker._load_jobs = {10: "request"}
    worker._connector_worker_meta = OffloadingWorkerMetadata()
    worker.worker.get_finished.return_value = [
        TransferResult(1, True, 30, physical_transfer_stats=((50, 0.1),)),
        TransferResult(2, True, 20, physical_transfer_stats=()),
        TransferResult(10, True, 9, physical_transfer_stats=((4, 0.2), (5, 0.3))),
        TransferResult(3, True, 7, 0.4),  # Legacy compatibility.
    ]
    assert worker.get_finished(set()) == (set(), {"request"})
    meta = worker.build_connector_worker_meta()
    assert meta.completed_jobs == {1: 1, 2: 1, 10: 1, 3: 1}
    assert meta.transfer_stats.store.bytes == 57
    assert meta.transfer_stats.store.sizes == [50, 7]
    assert meta.transfer_stats.load.bytes == 9
    assert meta.transfer_stats.load.sizes == [4, 5]
    assert meta.transfer_stats.load.time == pytest.approx(0.5)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("dp_replicas", [1, 2])
def test_worker_acknowledgments_remain_per_engine_and_rank(world_size, dp_replicas):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )

    engines = []
    for _ in range(dp_replicas):
        combined = OffloadingWorkerMetadata()
        for _rank in range(world_size):
            meta = OffloadingWorkerMetadata()
            meta.mark_completed(7)
            combined = combined.aggregate(meta)
        engines.append(combined)
    assert all(meta.completed_jobs == {7: world_size} for meta in engines)


@pytest.mark.parametrize("bpc", [1, 3])
def test_executor_moves_data_only_after_producer_and_before_completion(
    monkeypatch, bpc
):
    """Exercise the real adapter with a deterministic CPU stream simulator."""
    import torch

    from vllm.v1.kv_offload.cpu import copy_backend as adapter
    from vllm.v1.kv_offload.cpu.copy_executor import NativeCopyExecutor

    active = threading.local()

    class Stream:
        def __init__(self):
            self.queue: deque = deque()
            self.lock = threading.RLock()

        def wait_event(self, event):
            self.queue.append(event.query)

        def progress(self):
            with self.lock:
                while self.queue and self.queue[0]():
                    self.queue.popleft()

        def synchronize(self):
            deadline = time.monotonic() + 3
            while self.queue:
                self.progress()
                if time.monotonic() > deadline:
                    raise TimeoutError("fake stream blocked")
                time.sleep(0.001)

    class Event:
        def __init__(self, **kwargs):
            self.done = False
            self.stream = None

        def record(self, stream):
            self.done = False
            self.stream = stream

            def mark():
                self.done = True
                return True

            stream.queue.append(mark)

        def query(self):
            if self.stream is not None:
                self.stream.progress()
            return self.done

        def elapsed_time(self, other):
            assert self.done and other.done
            return 1.0

    @contextmanager
    def context(stream):
        active.stream = stream
        yield

    monkeypatch.setattr(adapter.current_platform, "set_device", lambda device: None)
    monkeypatch.setattr(adapter.current_platform, "Stream", Stream)
    monkeypatch.setattr(adapter.current_platform, "stream", context)
    monkeypatch.setattr(adapter.torch, "Event", Event)
    gpu = torch.randint(
        0,
        256,
        (32, 128),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(12),
    )
    cpu = torch.zeros((32, 128 * bpc), dtype=torch.uint8)
    store = handler("store", bpc, False, 0)
    load = handler("load", bpc, False, 0)
    for h in (store, load):
        h.src_tensors = [gpu if h.gpu_to_cpu else cpu]
        h.dst_tensors = [cpu if h.gpu_to_cpu else gpu]
        h.layer_refs_per_group = [[CanonicalKVCacheRef(0, 128)]]

        def copy(src, dst, sizes, **kwargs):
            # Consume descriptors at simulated device execution, not submit.
            def perform():
                for s, d, n in zip(src.tolist(), dst.tolist(), sizes.tolist()):
                    ctypes.memmove(d, s, n)
                return True

            active.stream.queue.append(perform)

        h._swap_blocks_batch = copy
        h.shutdown = lambda: None
    backend = CudaCopyBackend(
        store,
        load,
        lambda n: tuple(torch.empty(n, dtype=torch.int64) for _ in range(3)),
    )
    executor = NativeCopyExecutor(
        backend,
        CopyExecutorConfig(
            max_batch_descriptors=2,
            target_batch_bytes=256,
        ),
    )
    producer = Event()
    try:
        ids = [3, 4, 5, 6]
        gs = GPULoadStoreSpec(ids, [4], [1])
        count = (4 + 1 % bpc + bpc - 1) // bpc
        cs = CPULoadStoreSpec(list(range(2, 2 + count)))
        expected = gpu[ids].clone()
        executor.submit(1, "store", ("store", gs, cs), producer, 64)
        time.sleep(0.01)
        assert not cpu.any()
        producer.done = True
        executor.wait({1})
        assert executor.get_finished()[0].nbytes == 512
        gpu[ids] = 0
        ready = Event()
        ready.done = True
        executor.submit(2, "load", ("load", cs, gs), ready, 64)
        executor.wait({2})
        assert torch.equal(gpu[ids], expected)
        assert executor.get_finished()[0].nbytes == 512
    finally:
        producer.done = True
        executor.close()
