# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.copy_executor import CopyExecutorConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.perf import CPUPerfConfig
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)


def _all_workers_barrier() -> None:
    """Block until every worker rank has reached this point (gloo cpu group).

    A superset of the node-local mmap openers suffices: once the barrier
    releases, every worker sharing the region file has mapped it."""
    from vllm.distributed.parallel_state import (
        get_inner_dp_world_group,
        get_world_group,
    )

    try:
        group = get_inner_dp_world_group()
    except AssertionError:
        group = get_world_group()
    group.barrier()


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "reads that have not yet completed; in tiering this "
                    "includes secondary cascades (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU chunks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
            CPUOffloadingMetrics.CPU_CACHE_SLOTS: OffloadingGaugeMetadata(
                documentation=(
                    "CPU allocation units by state: fixed slots or compact 4KiB pages."
                ),
                labelnames=("state",),
            ),
            CPUOffloadingMetrics.CPU_STORE_ALLOCATION_FAILURES: (
                OffloadingCounterMetadata(
                    documentation="CPU prepare_store calls unable to allocate slots."
                )
            ),
            CPUOffloadingMetrics.CPU_EVICTED_CHUNKS: OffloadingCounterMetadata(
                documentation="CPU cache chunks evicted to admit new chunks."
            ),
            CPUOffloadingMetrics.CPU_INVALIDATED_CHUNKS: OffloadingCounterMetadata(
                documentation=(
                    "CPU chunks invalidated after a confirmed lower-tier copy."
                )
            ),
            CPUOffloadingMetrics.CPU_MISSES_AFTER_EVICTION: (
                OffloadingCounterMetadata(
                    documentation=(
                        "Distinct request/key CPU lookup misses after a recorded "
                        "eviction of that key. History is bounded to 64000 keys."
                    ),
                )
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        self.copy_executor_config = CopyExecutorConfig.from_extra_config(
            self.extra_config
        )
        self.perf_config = CPUPerfConfig.from_extra_config(self.extra_config)
        if (
            self.perf_config.early_load_ready or self.perf_config.precise_batching
        ) and self.copy_executor_config is None:
            raise ValueError(
                "early_load_ready and precise_batching require cpu_copy_executor"
            )
        if self.perf_config.compact_groups and not config.compact_page_size_bytes:
            raise ValueError("compact_groups requires a resolved compact slot size")
        if self.perf_config.compact_groups and config.compact_page_size_bytes != 4096:
            raise ValueError("compact_groups requires 4096-byte allocator pages")
        logger.info("Native CPU performance flags: %s", self.perf_config)
        self.compact_group_units = (
            {
                group.group_id: max(1, (size * self.blocks_per_chunk + 4095) // 4096)
                for group, size in zip(config.groups, config.compact_group_page_sizes)
            }
            if config.compact_group_page_sizes is not None
            else None
        )

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_chunks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        self.replicated_layout = config.replicated_layout and self._uses_shared_region()
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            num_copies = 1 if self.replicated_layout else world_size
            page_bytes = (
                config.compact_page_size_bytes or config.worker_kv_bytes_per_block
            )
            kv_bytes_per_block = page_bytes * num_copies
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // num_copies

            # calculate num_chunks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_chunks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded chunk is of the form,
            # |--- W0-C0---|---- W1-C0---| ... |---- Wn-C0---| *** maybe-pad *** |
            # or |--- C0 (single copy) ---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk
        if self.perf_config.compact_groups:
            assert config.compact_page_size_bytes is not None
            self.cpu_page_size_per_worker = config.compact_page_size_bytes
            self.kv_bytes_per_chunk = self.cpu_page_size_per_worker * world_size
            self.num_chunks = int(cpu_bytes_to_use) // self.kv_bytes_per_chunk
            logger.info(
                "Native CPU compact bank pages=%d page_bytes=%d group_units=%s",
                self.num_chunks,
                self.cpu_page_size_per_worker,
                self.compact_group_units,
            )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a chunk must be offered for
            # storage before it is eligible for CPU offloading.  Values < 2
            # disable filtering (a threshold of 1 equals no filter; 0 is the
            # default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_chunks=self.num_chunks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                incremental_recency=self.perf_config.incremental_recency,
                compact_group_units=self.compact_group_units,
            )
        return self._manager

    def _uses_shared_region(self) -> bool:
        """Whether the worker CPU buffer is the shared mmap region (vs a private
        per-rank tensor); replicated-layout dedup is gated on this being True."""
        return (
            not self.perf_config.compact_groups
            and current_platform.is_cuda_alike()
            and not current_platform.is_rocm()
        )

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        mmap_region: SharedOffloadRegion | None = None
        # num_chunks == 0 would size the region to zero bytes, which cannot be
        # mmap'd; fall back to the tensor path (empty tensors) as before.
        if self._uses_shared_region() and self.num_chunks > 0:
            # Replicated layout puts all ranks on slot 0 (single MLA copy);
            # otherwise each rank takes its own slot by physical device index.
            if self.replicated_layout:
                rank = 0
            else:
                world_size = self.config.parallel.world_size
                rank = torch.accelerator.current_device_index() % world_size
            mmap_region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_chunks=self.num_chunks,
                rank=rank,
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                barrier=_all_workers_barrier,
                diagnose_numa=bool(
                    self.extra_config.get("cpu_numa_diagnostics", False)
                ),
            )
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=self.num_chunks,
                mmap_region=mmap_region,
                copy_executor_config=self.copy_executor_config,
                perf_config=self.perf_config,
            )
        except Exception:
            if mmap_region is not None:
                mmap_region.cleanup()
            raise

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
