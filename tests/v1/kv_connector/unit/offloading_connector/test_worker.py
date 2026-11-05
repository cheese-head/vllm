# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

NUM_BLOCKS = 10
BLOCK_SIZE = 16
NUM_KV_HEADS = 4
HEAD_SIZE = 64
DTYPE = torch.float16
DEVICE_TYPE = current_platform.device_type

# Attention backends to test
ATTN_BACKENDS: list[str] = []
if current_platform.is_cuda():
    ATTN_BACKENDS = [
        "FLASH_ATTN",
        "FLEX_ATTENTION",
        "FLASHINFER",
        "TRITON_ATTN",
    ]
elif current_platform.is_rocm():
    ATTN_BACKENDS = ["TRITON_ATTN"]
elif current_platform.is_xpu():
    ATTN_BACKENDS = ["TRITON_ATTN", "FLASH_ATTN"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allocate_kv_caches(
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list],
    device: torch.device,
):
    """Allocate kv_caches exactly as the model runner does at startup."""
    from vllm.v1.worker.utils import allocate_kv_cache

    kernel_block_sizes = [BLOCK_SIZE] * len(kv_cache_config.kv_cache_groups)
    return allocate_kv_cache(
        kv_cache_config, device, KVCacheLayout.LBNHC, kernel_block_sizes
    )


def _single_rank_vllm_config(total_kv_heads: int):
    """A one-rank (TP=1) parallel config, as canonical mappings are derived
    from it."""
    vllm_config = MagicMock()
    parallel_config = vllm_config.parallel_config
    parallel_config.tensor_parallel_size = 1
    parallel_config.decode_context_parallel_size = 1
    parallel_config.prefill_context_parallel_size = 1
    parallel_config.cp_kv_cache_interleave_size = 1
    parallel_config.world_size = 1
    parallel_config.rank = 0
    vllm_config.model_config.get_total_num_kv_heads.return_value = total_kv_heads
    return vllm_config


def _make_worker(
    kv_cache_config: KVCacheConfig,
    replicated_layout: bool = False,
    rank: int = 0,
    canonical_layout: bool = False,
):
    """Create an OffloadingConnectorWorker with mocked dependencies."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    spec = MagicMock(spec=OffloadingSpec)
    spec.replicated_layout = replicated_layout
    spec.config = MagicMock()
    spec.config.canonical_layout = canonical_layout
    spec.config.extra_config = {}
    spec.config.parallel.rank = rank
    spec.get_worker.return_value = MagicMock()

    worker = OffloadingConnectorWorker(
        spec=spec,
        vllm_config=_single_rank_vllm_config(NUM_KV_HEADS),
        kv_cache_config=kv_cache_config,
    )
    worker.worker = MagicMock()

    return worker, spec


def _store_metadata(job_id: int) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={
            job_id: TransferJob(
                req_id="req",
                src_spec=GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,)),
                dst_spec=LoadStoreSpec(),
            )
        },
    )


def _load_metadata(job_id: int) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={
            job_id: TransferJob(
                req_id="req",
                src_spec=LoadStoreSpec(),
                dst_spec=GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,)),
            )
        },
        store_jobs={},
    )


def _empty_metadata() -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(load_jobs={}, store_jobs={})


def _offloading_config(rank: int = 0) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=2,
            tp_size=2,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )


class BareExternalOffloadingSpec(OffloadingSpec):
    def get_manager(self) -> OffloadingManager:
        raise NotImplementedError

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prepare_store_kv_non_writer_marks_completed_without_submit():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=True,
        rank=1,
    )

    worker.prepare_store_kv(_store_metadata(7))
    worker.start_kv_transfers(_empty_metadata())

    assert worker._unsubmitted_store_jobs == []
    assert worker.worker is not None
    worker.worker.submit_store.assert_not_called()
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {7: 1}


@pytest.mark.parametrize("rank,canonical_layout", [(0, False), (0, True), (1, True)])
def test_prepare_store_kv_writer_submits_store(rank, canonical_layout):
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=True,
        rank=rank,
        canonical_layout=canonical_layout,
    )

    worker.prepare_store_kv(_store_metadata(8))
    assert worker.build_connector_worker_meta() is None
    worker.start_kv_transfers(_empty_metadata())

    assert worker.worker is not None
    worker.worker.submit_store.assert_called_once()


def test_prepare_store_kv_non_replicated_rank_gt_zero_queues_store():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=False,
        rank=1,
    )

    worker.prepare_store_kv(_store_metadata(9))
    assert worker.build_connector_worker_meta() is None
    assert len(worker._unsubmitted_store_jobs) == 1

    worker.start_kv_transfers(_empty_metadata())

    assert worker.worker is not None
    worker.worker.submit_store.assert_called_once()
    assert worker._unsubmitted_store_jobs == []


def test_handle_preemptions_non_writer_acks_flushed_store():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=True,
        rank=1,
    )
    metadata = _store_metadata(10)
    metadata.jobs_to_flush = {10}

    worker.handle_preemptions(metadata)

    assert worker.worker is not None
    worker.worker.submit_store.assert_not_called()
    worker.worker.wait.assert_called_once_with({10})
    assert metadata.store_jobs == {}
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {10: 1}


def test_start_kv_transfers_non_writer_still_submits_load():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=True,
        rank=1,
    )

    worker.start_kv_transfers(_load_metadata(10))

    assert worker.worker is not None
    worker.worker.submit_load.assert_called_once()
    assert worker.build_connector_worker_meta() is None


def test_offloading_connector_worker_accepts_plugin_spec_default_layout():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    spec = BareExternalOffloadingSpec(_offloading_config(rank=1))

    OffloadingConnectorWorker(
        spec=spec,
        vllm_config=_single_rank_vllm_config(NUM_KV_HEADS),
        kv_cache_config=KVCacheConfig(
            num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
        ),
    )

    assert spec.replicated_layout is False


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_register_kv_caches(backend):
    """Test register_kv_caches with multiple groups covering all layer types.

    Creates one FullAttention group, one MLA group, one Mamba group, and
    one Mamba-padded group. Each group has GROUP_SIZE layers.

    KVCacheTensors mirror the real allocation in kv_cache_utils.py: each group is one
    run of the shared allocation starting at offset 0, so layer i of every group
    aliases the same bytes.

    Uses the real GPUModelRunner.initialize_kv_cache_tensors to produce
    the raw per-layer kv_caches registered by the connector.

    Verifies that the canonicalized CanonicalKVCaches has the correct
    block tensors, tensor_idx references, and page sizes across all groups.
    """
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerBackend,
    )
    from vllm.v1.worker.utils import AttentionGroup

    MLA_HEAD_SIZE = NUM_KV_HEADS * HEAD_SIZE * 2

    # padded mamba (missing HEAD_SIZE)
    CONV_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS, HEAD_SIZE)
    UNALIGNED_SSM_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS - 1, HEAD_SIZE)

    PAGE_SIZE_BYTES = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * get_dtype_size(DTYPE)
    unaligned_mamba_page_size = PAGE_SIZE_BYTES - HEAD_SIZE * get_dtype_size(DTYPE)

    # unpadded mamba (fills page exactly)
    ALIGNED_SSM_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS, HEAD_SIZE)

    backend_cls = AttentionBackendEnum[backend].get_class()

    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    mla_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=MLA_HEAD_SIZE,
        dtype=DTYPE,
    )
    unaligned_mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(CONV_STATE_SHAPE, UNALIGNED_SSM_STATE_SHAPE),
        dtypes=(DTYPE, DTYPE),
        page_size_padded=PAGE_SIZE_BYTES,
    )
    aligned_mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(CONV_STATE_SHAPE, ALIGNED_SSM_STATE_SHAPE),
        dtypes=(DTYPE, DTYPE),
        page_size_padded=PAGE_SIZE_BYTES,
    )

    assert attn_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert mla_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert unaligned_mamba_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert aligned_mamba_spec.page_size_bytes == PAGE_SIZE_BYTES

    GROUP_SIZE = 3

    # -- Build per-group layer info ----------------------------------------
    layer_idx = 0

    attn_layer_names = []
    for _ in range(GROUP_SIZE):
        attn_layer_names.append(f"model.layers.{layer_idx}.self_attn")
        layer_idx += 1

    mla_layer_names = []
    for _ in range(GROUP_SIZE):
        mla_layer_names.append(f"model.layers.{layer_idx}.self_attn")
        layer_idx += 1

    unaligned_mamba_layer_names = []
    for _ in range(GROUP_SIZE):
        unaligned_mamba_layer_names.append(f"model.layers.{layer_idx}.mamba_unpadded")
        layer_idx += 1

    aligned_mamba_layer_names = []
    for _ in range(GROUP_SIZE - 1):
        aligned_mamba_layer_names.append(f"model.layers.{layer_idx}.mamba_padded")
        layer_idx += 1

    layer_groups = [
        attn_layer_names,
        mla_layer_names,
        unaligned_mamba_layer_names,
        aligned_mamba_layer_names,
    ]

    kv_cache_tensors: list[KVCacheTensor] = [
        KVCacheTensor(
            size=PAGE_SIZE_BYTES * NUM_BLOCKS * GROUP_SIZE,
            layers=group_layer_names,
            layer_stride=PAGE_SIZE_BYTES * NUM_BLOCKS,
            block_stride=PAGE_SIZE_BYTES,
        )
        for group_layer_names in layer_groups
    ]

    kv_cache_groups = [
        KVCacheGroupSpec(layer_names=attn_layer_names, kv_cache_spec=attn_spec),
        KVCacheGroupSpec(layer_names=mla_layer_names, kv_cache_spec=mla_spec),
        KVCacheGroupSpec(
            layer_names=unaligned_mamba_layer_names, kv_cache_spec=unaligned_mamba_spec
        ),
        KVCacheGroupSpec(
            layer_names=aligned_mamba_layer_names, kv_cache_spec=aligned_mamba_spec
        ),
    ]

    attn_groups = [
        [
            AttentionGroup(
                backend=backend_cls,
                layer_names=attn_layer_names,
                kv_cache_spec=attn_spec,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,
                layer_names=mla_layer_names,
                kv_cache_spec=mla_spec,
                kv_cache_group_id=1,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,  # unused for mamba
                layer_names=unaligned_mamba_layer_names,
                kv_cache_spec=unaligned_mamba_spec,
                kv_cache_group_id=2,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,  # unused for mamba
                layer_names=aligned_mamba_layer_names,
                kv_cache_spec=aligned_mamba_spec,
                kv_cache_group_id=3,
            ),
        ]
    ]

    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )

    kv_caches = _allocate_kv_caches(
        kv_cache_config,
        attn_groups,
        device=torch.device(f"{DEVICE_TYPE}:0"),
    )

    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    canonical = spec.get_worker.call_args[0][0]
    assert isinstance(canonical, CanonicalKVCaches)

    # -- Expected block tensors ----------------------------------------------
    # All tensors have the same padded page size (PAGE_SIZE_BYTES).
    # Tensor 0: shared by attn[0], mla[0], mamba_unaligned[0], mamba_aligned[0]
    # Tensor 1: shared by attn[1], mla[1], mamba_unaligned[1], mamba_aligned[1]
    # Tensor 2: shared by attn[2], mla[2], mamba_unaligned[2]
    #           (mamba_aligned has only GROUP_SIZE-1 = 2 layers)
    expected_tensors = [
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
    ]

    # -- Expected group data refs (order matches kv_cache_groups) -------------
    ref = CanonicalKVCacheRef
    expected_group_refs = [
        # attn group: layers attn[0..2] → tensors 0,1,2 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=2, page_size_bytes=PAGE_SIZE_BYTES),
        ],
        # mla group: layers mla[0..2] → tensors 0,1,2 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=2, page_size_bytes=PAGE_SIZE_BYTES),
        ],
        # unaligned mamba group: layers [0..2] → tensors 0,1,2 with unaligned page
        [
            ref(tensor_idx=0, page_size_bytes=unaligned_mamba_page_size),
            ref(tensor_idx=1, page_size_bytes=unaligned_mamba_page_size),
            ref(tensor_idx=2, page_size_bytes=unaligned_mamba_page_size),
        ],
        # aligned mamba group: layers [0..1] → tensors 0,1 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
        ],
    ]

    # Verify block tensors
    assert len(canonical.tensors) == len(expected_tensors)
    for block_tensor, (exp_num_blocks, exp_page_size) in zip(
        canonical.tensors, expected_tensors
    ):
        tensor = block_tensor.tensor
        assert tensor.dtype == torch.int8
        assert tensor.shape == (exp_num_blocks, exp_page_size)
        assert block_tensor.page_size_bytes == exp_page_size

    # Verify group data refs
    assert len(canonical.group_data_refs) == len(expected_group_refs)
    for actual_refs, exp_refs in zip(canonical.group_data_refs, expected_group_refs):
        assert len(actual_refs) == len(exp_refs)
        for actual, expected in zip(actual_refs, exp_refs):
            assert actual.tensor_idx == expected.tensor_idx
            assert actual.page_size_bytes == expected.page_size_bytes
            # Every layer gets a canonical mapping, certified or opaque
            assert actual.mapping is not None


def test_register_packed_kv_caches_skips_scratch_group():
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    page = attn_spec.page_size_bytes
    layers = ["layer0", "layer1"]
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * page * NUM_BLOCKS,
                layers=layers,
                layer_stride=page,
                block_stride=2 * page,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layers, attn_spec),
            KVCacheGroupSpec(
                ["scratch"],
                CircularBufferSpec(
                    block_size=4, num_kv_heads=1, head_size=1, dtype=DTYPE
                ),
            ),
        ],
    )
    kv_caches = _allocate_kv_caches(
        kv_cache_config, [], device=torch.device(f"{DEVICE_TYPE}:0")
    )
    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    canonical = spec.get_worker.call_args[0][0]
    assert len(canonical.tensors) == 1
    assert canonical.tensors[0].tensor.shape == (NUM_BLOCKS, 2 * page)
    assert canonical.group_data_refs == [[CanonicalKVCacheRef(0, 2 * page)]]


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_register_kv_caches_uniform_type(backend):
    """Test register_kv_caches with UniformTypeKVCacheSpecs.

    Two attention layers use the same backend but different num_kv_heads,
    giving them different per-layer page sizes. Each has its own
    KVCacheTensor and are wrapped in a UniformTypeKVCacheSpecs group.
    Verifies that each layer gets the correct tensor_idx and
    page_size_bytes in its block data ref.
    """
    from vllm.v1.worker.utils import AttentionGroup

    backend_cls = AttentionBackendEnum[backend].get_class()

    layer_a = "model.layers.0.self_attn"
    layer_b = "model.layers.1.self_attn"
    spec_a = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    spec_b = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS * 2,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    assert spec_a.page_size_bytes != spec_b.page_size_bytes

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=BLOCK_SIZE,
        kv_cache_specs={layer_a: spec_a, layer_b: spec_b},
    )

    # The group packs its two layers densely, one run each as their pages differ:
    # layer_a occupies the first page of every block, layer_b the rest.
    window = spec_a.page_size_bytes + spec_b.page_size_bytes
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=window * NUM_BLOCKS,
                layers=[layer_a],
                layer_stride=spec_a.page_size_bytes * NUM_BLOCKS,
                block_stride=spec_a.page_size_bytes,
            ),
            KVCacheTensor(
                size=window * NUM_BLOCKS,
                layers=[layer_b],
                layer_stride=spec_b.page_size_bytes * NUM_BLOCKS,
                block_stride=spec_b.page_size_bytes,
                offset=spec_a.page_size_bytes * NUM_BLOCKS,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_a, layer_b],
                kv_cache_spec=uniform_spec,
            )
        ],
    )

    attn_groups = [
        [
            AttentionGroup(
                backend=backend_cls,
                layer_names=[layer_a],
                kv_cache_spec=spec_a,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=backend_cls,
                layer_names=[layer_b],
                kv_cache_spec=spec_b,
                kv_cache_group_id=0,
            ),
        ]
    ]

    kv_caches = _allocate_kv_caches(
        kv_cache_config,
        attn_groups,
        device=torch.device(f"{DEVICE_TYPE}:0"),
    )

    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    canonical = spec.get_worker.call_args[0][0]
    assert isinstance(canonical, CanonicalKVCaches)

    for block_tensor in canonical.tensors:
        assert block_tensor.tensor.dtype == torch.int8

    # Single group with refs from both layers
    assert len(canonical.group_data_refs) == 1
    group_refs = canonical.group_data_refs[0]
    assert len(group_refs) == 2

    assert len(canonical.tensors) == 2
    assert canonical.tensors[0].page_size_bytes == spec_a.page_size_bytes
    assert canonical.tensors[1].page_size_bytes == spec_b.page_size_bytes
    assert canonical.tensors[0].tensor.shape == (NUM_BLOCKS, spec_a.page_size_bytes)
    assert canonical.tensors[1].tensor.shape == (NUM_BLOCKS, spec_b.page_size_bytes)

    for ref, expected_tensor_idx, expected_spec in (
        (group_refs[0], 0, spec_a),
        (group_refs[1], 1, spec_b),
    ):
        assert ref.tensor_idx == expected_tensor_idx
        assert ref.page_size_bytes == expected_spec.page_size_bytes
        assert ref.mapping is not None

    # Only layer_a matches the model's total KV head count, so layer_b gets an
    # opaque mapping rather than a certified, parallelism-agnostic one
    assert group_refs[0].mapping.parallelism_agnostic
    assert not group_refs[1].mapping.parallelism_agnostic


def _captured_test_worker(selective=True, max_jobs=256):
    import copy
    from types import SimpleNamespace

    from vllm.v1.kv_offload.cpu.transfer_lifecycle import (
        CapturedStoreIntent,
        DeferredStoreQueue,
    )

    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    )
    calls, dependencies = [], []

    def capture_dependency():
        token = object()
        dependencies.append(token)
        return token

    def record_submission(kind, value):
        calls.append((kind, value))
        return True

    worker.worker = SimpleNamespace(
        capture_store_dependency=capture_dependency,
        capture_store=lambda job, src, dst, token: CapturedStoreIntent(
            job, copy.deepcopy(src), copy.deepcopy(dst), token, 8
        ),
        submit_store_intent=lambda intent: record_submission("store", intent),
        submit_store=lambda *args: record_submission("legacy", args),
        submit_load=lambda *args: record_submission("load", args),
        wait=lambda ids: calls.append(("wait", set(ids))),
    )
    worker._captured_stores = DeferredStoreQueue(max_jobs=max_jobs)
    worker._selective_stores = selective
    return worker, calls, dependencies


def test_selective_store_keeps_original_boundaries_across_next_forward():
    worker, calls, dependencies = _captured_test_worker()
    metadata = _store_metadata(61)
    worker.prepare_store_kv(metadata)
    metadata.store_jobs[61].src_spec.block_ids[0] = 999
    worker.handle_preemptions(_empty_metadata())
    assert calls == []
    producer, sampling_gate = dependencies
    worker.start_kv_transfers(_empty_metadata())
    submitted = calls[0][1]
    assert submitted.producer.events == (producer, sampling_gate)
    assert submitted.src_spec.block_ids.tolist() == [0]
    assert len(dependencies) == 2


def test_capture_only_preserves_pre_forward_submission_order():
    worker, calls, _ = _captured_test_worker(selective=False)
    worker.prepare_store_kv(_store_metadata(62))
    worker.handle_preemptions(_empty_metadata())
    assert calls[0][0] == "store"
    assert not worker._captured_stores


def test_selective_same_step_flush_precedes_wait_and_retains_other_store():
    worker, calls, _ = _captured_test_worker()
    worker.prepare_store_kv(_store_metadata(63))
    metadata = _store_metadata(64)
    metadata.jobs_to_flush = {64}
    worker.handle_preemptions(metadata)
    assert [(kind, obj.job_id if kind == "store" else obj) for kind, obj in calls] == [
        ("store", 64),
        ("wait", {64}),
    ]
    assert worker._captured_stores.pending_job_ids == (63,)
    worker.start_kv_transfers(_empty_metadata())
    assert calls[-1][1].job_id == 63


def test_capture_capacity_fallback_keeps_already_promised_store():
    worker, calls, _ = _captured_test_worker(max_jobs=1)
    worker.prepare_store_kv(_store_metadata(65))
    worker.prepare_store_kv(_store_metadata(66))
    assert calls[0][1].job_id == 66
    assert worker._captured_stores.pending_job_ids == (65,)
    assert worker._store_capacity_fallbacks == 1
    worker.handle_preemptions(_empty_metadata())
    worker.start_kv_transfers(_empty_metadata())
    assert sorted(obj.job_id for kind, obj in calls if kind == "store") == [65, 66]


def test_unaccepted_required_store_cannot_allow_reuse():
    worker, calls, _ = _captured_test_worker()
    worker.worker.submit_store_intent = lambda intent: False
    metadata = _store_metadata(67)
    metadata.jobs_to_flush = {67}
    with pytest.raises(RuntimeError, match="required store"):
        worker.handle_preemptions(metadata)
    assert calls == []
    assert worker._captured_stores.pending_job_ids == (67,)


def _cpu_lifecycle_registration(monkeypatch, *, layout="shared", uniform=False):
    """Exercise registration with real CPU storage, without creating CUDA state."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import worker as mod

    monkeypatch.delenv("VLLM_NATIVE_CPU_LIFECYCLE_REQUIRE_SELECTIVE", raising=False)
    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    page = attention.page_size_bytes
    row_stride = 2 * page
    backing = torch.empty(NUM_BLOCKS * 3 * page, dtype=torch.uint8, device="cpu")
    first = backing.as_strided((NUM_BLOCKS, page), (row_stride, 1), 0)
    second = backing.as_strided((NUM_BLOCKS, page), (row_stride, 1), page)
    if layout == "different_storage":
        other = torch.empty_like(backing)
        second = other.as_strided((NUM_BLOCKS, page), (row_stride, 1), page)
    elif layout == "different_stride":
        second = backing.as_strided((NUM_BLOCKS, page), (3 * page, 1), page)
    elif layout == "outside_row":
        second = backing.as_strided((NUM_BLOCKS, page), (row_stride, 1), row_stride)
    elif layout == "unpacked":
        first = torch.empty((NUM_BLOCKS, page), dtype=torch.uint8, device="cpu")
        second = torch.empty_like(first)
    else:
        assert layout == "shared"

    selected_spec = (
        UniformTypeKVCacheSpecs(
            block_size=BLOCK_SIZE,
            kv_cache_specs={"first": attention, "second": attention},
        )
        if uniform
        else attention
    )
    config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["first", "second"], selected_spec)],
    )
    worker, spec = _make_worker(config)
    spec.config = _offloading_config()
    spec.blocks_per_chunk = 1
    worker.vllm_config.speculative_config = None
    backend = MagicMock(spec=OffloadingWorker)
    backend.store_lifecycle_mode.return_value = "selective"
    backend.store_lifecycle_limits.return_value = (8, 4096)
    spec.get_worker.return_value = backend
    # Canonical address geometry is unrelated to this packed-storage proof.
    monkeypatch.setattr(mod, "derive_canonical_mappings", lambda *args: {})
    return worker, spec, backend, {"first": first, "second": second}


@pytest.mark.parametrize("uniform", [False, True])
def test_selective_registration_certifies_shared_packed_cpu_storage(
    monkeypatch, uniform
):
    worker, _, _, caches = _cpu_lifecycle_registration(monkeypatch, uniform=uniform)

    worker.register_kv_caches(caches)

    assert worker._known_packed_layout
    assert worker._selective_stores
    assert worker._captured_stores is not None


@pytest.mark.parametrize(
    "layout", ["different_storage", "different_stride", "outside_row", "unpacked"]
)
def test_selective_registration_rejects_unproven_packed_storage(monkeypatch, layout):
    worker, _, _, caches = _cpu_lifecycle_registration(monkeypatch, layout=layout)

    worker.register_kv_caches(caches)

    assert not worker._known_packed_layout
    assert not worker._selective_stores
    assert worker._captured_stores is not None  # Conservative capture mode remains.


@pytest.mark.parametrize(
    "unsupported",
    ["canonical", "coalesced_chunks", "speculation", "pp", "pcp", "dcp", "dp", "swa"],
)
def test_selective_registration_falls_back_for_unsupported_contract(
    monkeypatch, unsupported
):
    from dataclasses import replace

    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    worker, spec, _, caches = _cpu_lifecycle_registration(monkeypatch)
    if unsupported == "canonical":
        spec.config = replace(spec.config, canonical_layout=True)
    elif unsupported == "coalesced_chunks":
        spec.config = replace(
            spec.config, cache=replace(spec.config.cache, blocks_per_chunk=2)
        )
        spec.blocks_per_chunk = 2
    elif unsupported == "speculation":
        worker.vllm_config.speculative_config = object()
    elif unsupported in ("pp", "pcp", "dcp", "dp"):
        name = "data_parallel_size" if unsupported == "dp" else f"{unsupported}_size"
        spec.config = replace(
            spec.config, parallel=replace(spec.config.parallel, **{name: 2})
        )
    else:
        worker.kv_cache_config.kv_cache_groups[0] = replace(
            worker.kv_cache_config.kv_cache_groups[0],
            kv_cache_spec=SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_SIZE,
                dtype=DTYPE,
                sliding_window=32,
            ),
        )

    worker.register_kv_caches(caches)

    assert worker._known_packed_layout
    assert not worker._selective_stores
    assert worker._captured_stores is not None


@pytest.mark.parametrize("mode", ["legacy", "capture_only"])
def test_registration_does_not_promote_nonselective_mode(monkeypatch, mode):
    worker, _, backend, caches = _cpu_lifecycle_registration(monkeypatch)
    backend.store_lifecycle_mode.return_value = mode

    worker.register_kv_caches(caches)

    assert not worker._selective_stores
    assert (worker._captured_stores is None) == (mode == "legacy")


@pytest.mark.parametrize("rejection", ["different_storage", "legacy", "capture_only"])
def test_required_selective_registration_cannot_silently_fall_back(
    monkeypatch, rejection
):
    layout = rejection if rejection == "different_storage" else "shared"
    worker, _, backend, caches = _cpu_lifecycle_registration(monkeypatch, layout=layout)
    if rejection != "different_storage":
        backend.store_lifecycle_mode.return_value = rejection
    monkeypatch.setenv("VLLM_NATIVE_CPU_LIFECYCLE_REQUIRE_SELECTIVE", "1")

    with pytest.raises(RuntimeError, match="selective"):
        worker.register_kv_caches(caches)


def test_compact_registration_uses_group_offsets_and_smaller_slots(monkeypatch):
    from dataclasses import replace

    worker, spec, backend, caches = _cpu_lifecycle_registration(
        monkeypatch, layout="shared"
    )
    group = worker.kv_cache_config.kv_cache_groups[0]
    worker.kv_cache_config.kv_cache_groups = [
        KVCacheGroupSpec(["first"], group.kv_cache_spec),
        KVCacheGroupSpec(["second"], group.kv_cache_spec),
    ]
    page = group.kv_cache_spec.page_size_bytes
    spec.config = replace(
        spec.config,
        compact_page_size_bytes=4096,
        compact_group_page_sizes=(page, page),
        extra_config={"cpu_perf": {"compact_groups": True}},
    )
    worker.register_kv_caches(caches)
    registered = spec.get_worker.call_args.args[0]
    assert registered.compact_page_size_bytes == 4096
    assert registered.compact_group_page_sizes == (page, page)
    assert registered.tensors[0].page_size_bytes == 2 * page
    assert [
        [(r.gpu_offset_bytes, r.host_offset_bytes, r.page_size_bytes) for r in refs]
        for refs in registered.group_data_refs
    ] == [[(0, 0, page)], [(page, 0, page)]]


def test_compact_registration_rejects_layer_view_crossing_page(monkeypatch):
    from dataclasses import replace

    worker, spec, backend, caches = _cpu_lifecycle_registration(
        monkeypatch, layout="shared"
    )
    page = worker.kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    spec.config = replace(
        spec.config,
        compact_page_size_bytes=2 * page,
        extra_config={"cpu_perf": {"compact_groups": True}},
    )
    first = caches["first"]
    caches["first"] = first.as_strided(first.shape, (first.stride(0), 2))
    with pytest.raises(ValueError, match="spans outside its page"):
        worker.register_kv_caches(caches)


@pytest.mark.parametrize("layout", [KVCacheLayout.BLNHC, KVCacheLayout.BLHNC])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("empty_group", [False, True])
def test_compact_dsv4_like_five_group_alias_layout(
    monkeypatch, layout, dtype, empty_group
):
    """Certify real allocator views on CPU; no model or CUDA state is created."""
    from dataclasses import replace

    from tests.v1.kv_connector.unit.offloading_connector.test_config import (
        _make_dsv4_flash_kv_cache_config,
        _make_vllm_config,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import worker as module
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
    from vllm.v1.worker.utils import allocate_kv_cache

    cache = _make_dsv4_flash_kv_cache_config()
    for group in cache.kv_cache_groups:
        group.kv_cache_spec = replace(
            group.kv_cache_spec,
            kv_cache_specs={
                name: replace(spec, dtype=dtype)
                for name, spec in group.kv_cache_spec.kv_cache_specs.items()
            },
        )
    page = next(
        iter(cache.kv_cache_groups[0].kv_cache_spec.kv_cache_specs.values())
    ).page_size_bytes
    row = 2 * page
    cache.kv_cache_tensors = [
        KVCacheTensor(
            size=cache.num_blocks * row,
            layers=list(group.layer_names),
            layer_stride=page,
            block_stride=row,
        )
        for group in cache.kv_cache_groups
    ]
    config = _make_vllm_config(
        tensor_parallel_size=8,
        extra_config={
            "cpu_bytes_to_use": 1 << 20,
            "cpu_perf": {"compact_groups": True},
        },
    )
    config.use_v2_model_runner = True
    caches = allocate_kv_cache(cache, torch.device("cpu"), layout)
    if empty_group:
        cache.kv_cache_groups[1].layer_names = []
    normalized = build_offloading_config(config, cache)
    worker, spec = _make_worker(cache)
    worker.vllm_config = config
    spec.config = normalized
    spec.get_worker.return_value.store_lifecycle_mode.return_value = "legacy"
    monkeypatch.setattr(module, "derive_canonical_mappings", lambda *args: {})
    worker.register_kv_caches(caches)
    registered = spec.get_worker.call_args.args[0]
    assert worker._known_packed_layout
    assert len(registered.group_data_refs) == 5
    assert registered.compact_page_size_bytes == 4096
    if empty_group:
        assert registered.group_data_refs[1] == []
    for group in registered.group_data_refs:
        assert all(r.gpu_offset_bytes + r.page_size_bytes <= row for r in group)
