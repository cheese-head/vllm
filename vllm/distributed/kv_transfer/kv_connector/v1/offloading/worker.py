# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections import defaultdict
from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    derive_canonical_mappings,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.perf import CPUPerfConfig, packed_group_regions
from vllm.v1.kv_offload.cpu.transfer_lifecycle import (
    CapturedStoreIntent,
    DeferredStoreQueue,
)

logger = init_logger(__name__)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods."""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig,
    ):
        self.spec = spec
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.worker: OffloadingWorker | None = None
        self._captured_stores: DeferredStoreQueue | None = None
        self._selective_stores = False
        self._known_packed_layout = False
        self._retained_submission_intents: list[CapturedStoreIntent] = []
        self._next_executor_report = 0.0
        self._last_executor_counters: dict[str, float] = {}
        self._store_capacity_fallbacks = 0
        # Non-writers still ack: pending_count waits for world_size per job.
        self._is_store_writer = (
            not self.spec.replicated_layout
            or self.spec.config.canonical_layout
            or self.spec.config.parallel.rank == 0
        )

        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)
        mode = getattr(self.worker, "store_lifecycle_mode", lambda: "legacy")()
        require_selective = (
            os.environ.get("VLLM_NATIVE_CPU_LIFECYCLE_REQUIRE_SELECTIVE") == "1"
        )
        if require_selective and mode != "selective":
            raise RuntimeError("Requested selective store executor was not enabled")
        if mode not in ("capture_only", "selective"):
            return
        limits = self.worker.store_lifecycle_limits()
        self._captured_stores = DeferredStoreQueue(*limits)
        selected = get_offloading_group_ids(self.kv_cache_config)
        specs = [
            self.kv_cache_config.kv_cache_groups[i].kv_cache_spec for i in selected
        ]
        actual = [
            item
            for spec in specs
            for item in (
                spec.kv_cache_specs.values()
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else (spec,)
            )
        ]
        parallel = self.spec.config.parallel
        eligible = (
            self._known_packed_layout
            and not self.spec.config.canonical_layout
            and self.spec.blocks_per_chunk == 1
            and self.vllm_config.speculative_config is None
            and parallel.pp_size == parallel.pcp_size == parallel.dcp_size == 1
            and parallel.data_parallel_size == 1
            and bool(actual)
            and all(
                type(spec)
                in (FullAttentionSpec, MLAAttentionSpec, SlidingWindowMLASpec)
                for spec in actual
            )
        )
        self._selective_stores = mode == "selective" and eligible
        if mode == "selective" and not eligible and require_selective:
            raise RuntimeError("Requested selective store layout was not certified")
        logger.info(
            "Native CPU lifecycle mode=%s effective=%s packed=%s rank=%d",
            mode,
            "selective" if self._selective_stores else "capture_only",
            self._known_packed_layout,
            parallel.rank,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        kv_cache_config = self.kv_cache_config
        perf = CPUPerfConfig.from_extra_config(self.spec.config.extra_config)
        selected_group_ids = get_offloading_group_ids(kv_cache_config)
        selected_groups = tuple(
            kv_cache_config.kv_cache_groups[group_id] for group_id in selected_group_ids
        )
        mappings = derive_canonical_mappings(
            self.vllm_config, kv_cache_config, kv_caches
        )

        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in selected_groups:
            assert not kv_cache_group.host_resident
            num_blocks = kv_cache_config.num_blocks
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                ref = group_kernel_blocks(kv_caches[layer_name], num_blocks)
                page = layer_kv_cache_spec.page_size_bytes
                elem_size = ref.element_size()
                byte_offset = ref.storage_offset() * elem_size
                if perf.compact_groups:
                    span = elem_size * (
                        1
                        + sum(
                            (n - 1) * s for n, s in zip(ref.shape[1:], ref.stride()[1:])
                        )
                    )
                    if any(n <= 0 for n in ref.shape[1:]) or span > page:
                        raise ValueError("Compact layer view spans outside its page")
                # Packed layouts (e.g. DSv4) interleave layers per block, so the
                # attention tensor's stride(0) (the manager-block stride) exceeds
                # page_size_bytes; other layouts have stride(0) == page_size_bytes.
                block_stride_bytes = (
                    ref.stride(0) * elem_size
                    if isinstance(layer_kv_cache_spec, AttentionSpec)
                    else page
                )
                tensors_per_block[layer_name] = (
                    torch.tensor([], dtype=torch.int8, device=ref.device).set_(
                        ref.untyped_storage(),
                        byte_offset,
                        (num_blocks, page),
                        (block_stride_bytes, 1),
                    ),
                )
                page_size_bytes[layer_name] = page

                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    unpadded_page_size_bytes[layer_name] = (
                        layer_kv_cache_spec.unpadded_page_size_bytes
                    )
                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes
                else:
                    raise NotImplementedError

        # Packed layouts (e.g. DSv4) interleave all layers within each manager
        # block: a layer view's block stride exceeds its page size. Offload the
        # whole packed block as a single transfer region.
        packed_layer_name = next(
            (
                layer_name
                for layer_name, (tensor,) in tensors_per_block.items()
                if tensor.stride(0) != tensor.shape[1]
            ),
            None,
        )
        if (
            packed_layer_name is not None
            and kv_cache_config.hisparse_host_num_blocks is None
        ):
            (tensor,) = tensors_per_block[packed_layer_name]
            num_blocks = tensor.shape[0]
            block_stride = tensor.stride(0)
            storage = tensor.untyped_storage()
            self._known_packed_layout = (
                all(
                    view.untyped_storage().data_ptr() == storage.data_ptr()
                    and view.shape[0] == num_blocks
                    and view.stride(0) == block_stride
                    and view.storage_offset() >= 0
                    and view.storage_offset() + view.shape[1] <= block_stride
                    for views in tensors_per_block.values()
                    for view in views
                )
                and storage.nbytes() >= num_blocks * block_stride
            )
            packed_tensor = tensor.as_strided(
                (num_blocks, block_stride),
                (block_stride, 1),
                storage_offset=0,
            )
            if perf.compact_groups:
                if not self._known_packed_layout:
                    raise ValueError("compact_groups requires certified packed storage")
                host_bytes = self.spec.config.compact_page_size_bytes
                assert host_bytes is not None
                group_pages = self.spec.config.compact_group_page_sizes
                if group_pages is None:
                    raise ValueError("Compact group page sizes were not resolved")
                regions = packed_group_regions(
                    [
                        [
                            (view.storage_offset(), view.shape[1])
                            for name in group.layer_names
                            for view in tensors_per_block[name]
                        ]
                        for group in selected_groups
                    ],
                    block_stride,
                    group_pages,
                )
                self._init_worker(
                    CanonicalKVCaches(
                        [CanonicalKVCacheTensor(packed_tensor, block_stride)],
                        [
                            [
                                CanonicalKVCacheRef(
                                    0,
                                    size,
                                    gpu_offset_bytes=gpu_offset,
                                    host_offset_bytes=host_offset,
                                )
                                for gpu_offset, host_offset, size in group
                            ]
                            for group in regions
                        ],
                        compact_page_size_bytes=host_bytes,
                        compact_group_page_sizes=group_pages,
                    )
                )
                return
            self._init_worker(
                CanonicalKVCaches(
                    [CanonicalKVCacheTensor(packed_tensor, block_stride)],
                    [[CanonicalKVCacheRef(0, block_stride)] for _ in selected_groups],
                )
            )
            return

        if perf.compact_groups:
            raise ValueError("compact_groups requires a packed KV cache layout")
        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        # Layers that alias the same bytes (cache groups overlay each other) share
        # one transfer region: exactly the views with equal address and strides.
        aliased_layers: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
        for layer_name, layer_tensors in tensors_per_block.items():
            view = layer_tensors[0]
            aliased_layers[(view.data_ptr(), view.stride())].append(layer_name)

        for tensor_layer_names in aliased_layers.values():
            assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1

            first_layer_name = tensor_layer_names[0]
            for tensor in tensors_per_block[first_layer_name]:
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=page_size_bytes[first_layer_name],
                    )
                )

                curr_tensor_idx = len(block_tensors) - 1
                for layer_name in tensor_layer_names:
                    mapping = (
                        mappings.get(layer_name)
                        if len(tensors_per_block[first_layer_name]) == 1
                        else None
                    )
                    assert (
                        mapping is None
                        or mapping.local_page_size_bytes
                        == unpadded_page_size_bytes[layer_name]
                    )
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                            mapping=mapping,
                        )
                    )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in selected_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._init_worker(canonical_kv_caches)

    def _queue_store(self, job_id, src_spec, dst_spec, producer) -> None:
        assert self.worker is not None and self._captured_stores is not None
        intent = self.worker.capture_store(job_id, src_spec, dst_spec, producer)
        try:
            self._captured_stores.append(intent)
        except BufferError as error:
            # The CPU manager already promised this store: preserve ownership
            # and use the conservative blocking path, never silently reject it.
            self._store_capacity_fallbacks += 1
            self._retained_submission_intents.append(intent)
            if not self.worker.submit_store_intent(intent):
                raise RuntimeError(
                    "Promised store was rejected during capacity fallback"
                ) from error
            self._retained_submission_intents.pop()

    def record_load_destination_ready(self, metadata: OffloadingConnectorMetadata):
        if self.worker is not None and metadata.load_jobs:
            record = getattr(self.worker, "record_load_destination_ready", None)
            if record is not None:
                record()

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        if self._captured_stores is not None:
            # Record the original next-step issue boundary before new GPU work.
            self._captured_stores.bind_submission_gate(
                self.worker.capture_store_dependency
            )
            flush = kv_connector_metadata.jobs_to_flush or set()
            producer = None
            for job_id in flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is None:
                    continue
                if not self._is_store_writer:
                    self._connector_worker_meta.mark_completed(job_id)
                    continue
                if producer is None:
                    producer = self.worker.capture_store_dependency()
                self._queue_store(job_id, entry.src_spec, entry.dst_spec, producer)
            self._captured_stores.submit_selected(
                self.worker.submit_store_intent,
                flush if self._selective_stores else None,
            )
            if flush:
                if flush.intersection(self._captured_stores.pending_job_ids):
                    raise RuntimeError("A required store was not accepted before reuse")
                self.worker.wait(flush)
            return

        # Pop jobs_to_flush from store_jobs into _unsubmitted_store_jobs
        # so the existing submission loop below submits them before wait().
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )

        # Submit deferred stores from previous step (and jobs_to_flush above).
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        if self._captured_stores is not None:
            self._captured_stores.submit_selected(self.worker.submit_store_intent)
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        producer = None
        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                # Gate before queueing: no _unsubmitted_store_jobs entry.
                self._connector_worker_meta.mark_completed(job_id)
                continue
            if self._captured_stores is not None:
                if producer is None:
                    producer = self.worker.capture_store_dependency()
                self._queue_store(job_id, entry.src_spec, entry.dst_spec, producer)
                continue
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Returns:
        tuple of (finished_sending, finished_recving). Stores never
        emit finished_sending — the scheduler tracks store completion
        via kv_connector_worker_meta.completed_jobs and fences any
        block reuse via jobs_to_flush. Loads still emit
        finished_recving so the base scheduler can resume requests
        blocked on remote KV (and free aborted-during-load reqs).

        """
        assert self.worker is not None
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            is_load = job_id in self._load_jobs
            stats = (
                self._connector_worker_meta.transfer_stats.load
                if is_load
                else self._connector_worker_meta.transfer_stats.store
            )
            physical = getattr(transfer_result, "physical_transfer_stats", None)
            if physical is not None:
                for nbytes, seconds in physical:
                    stats.record(nbytes, seconds)
            elif (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                stats.record(
                    transfer_result.transfer_size, transfer_result.transfer_time
                )

            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        now = time.monotonic()
        if now >= self._next_executor_report:
            self._next_executor_report = now + 10.0
            snapshot = self.worker.get_executor_statistics()
            if isinstance(snapshot, dict) and snapshot:
                gauges = {
                    "pending_jobs",
                    "pending_metadata_bytes",
                    "completed_unpolled",
                    "pending_jobs_high_water",
                    "pending_bytes_high_water",
                }
                snapshot = dict(
                    snapshot,
                    capacity_fallbacks=self._store_capacity_fallbacks,
                    selective_enabled=float(self._selective_stores),
                )
                gauges.add("selective_enabled")
                values = {
                    key: float(value)
                    if key in gauges
                    else max(
                        0.0, float(value) - self._last_executor_counters.get(key, 0.0)
                    )
                    for key, value in snapshot.items()
                }
                self._last_executor_counters = {
                    key: float(value)
                    for key, value in snapshot.items()
                    if key not in gauges
                }
                self._connector_worker_meta.executor_stats[
                    self.spec.config.parallel.rank
                ] = values

        return set(), finished_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if (
            not self._connector_worker_meta.completed_jobs
            and not self._connector_worker_meta.executor_stats
        ):
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
