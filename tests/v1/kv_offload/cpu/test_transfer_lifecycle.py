# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership and ordering contracts; no CUDA visibility claims."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SOURCE = (
    Path(__file__).resolve().parents[4] / "vllm/v1/kv_offload/cpu/transfer_lifecycle.py"
)
spec = importlib.util.spec_from_file_location(
    "native_transfer_lifecycle_tested", SOURCE
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Intent = module.CapturedStoreIntent
Queue = module.DeferredStoreQueue
capture = module.capture_store_intent


def intent(job_id, charge=8, producer=None):
    return Intent(job_id, (job_id,), (job_id + 1,), producer or object(), charge)


def record_submission(recorded: list[Any], item: Any, accepted: bool = True) -> bool:
    recorded.append(item)
    return accepted


def test_capture_owns_snapshots_and_preserves_original_producer():
    source, destination = [1, 2], [3, 4]
    boundary = SimpleNamespace(current=object())
    original = boundary.current
    captured = capture(
        7,
        source,
        destination,
        snapshot_specs=lambda src, dst: (tuple(src), tuple(dst), 32),
        capture_producer=lambda: boundary.current,
    )
    source[0], destination[0] = 99, 100
    boundary.current = object()
    assert captured.src_spec == (1, 2)
    assert captured.dst_spec == (3, 4)
    assert captured.producer is original
    with pytest.raises(FrozenInstanceError):
        captured.producer = boundary.current


def test_same_boundary_can_share_owned_producer_across_intents():
    producer = object()
    queue = Queue()
    first, second = intent(1, producer=producer), intent(2, producer=producer)
    queue.append(first)
    queue.append(second)
    accepted: list[Any] = []
    assert (
        queue.submit_selected(lambda item: record_submission(accepted, item), {2}) == 1
    )
    assert accepted == [second]
    assert accepted[0].producer is first.producer is producer
    assert queue.pending_job_ids == (1,)


def test_capture_does_not_record_dependency_for_invalid_charge():
    recorded = []
    with pytest.raises(ValueError, match="metadata"):
        capture(
            1,
            (),
            (),
            snapshot_specs=lambda src, dst: (src, dst, -1),
            capture_producer=lambda: recorded.append(True),
        )
    assert recorded == []


def test_flush_selection_keeps_unrelated_stores_and_fifo_order():
    queue = Queue()
    pending = [intent(1), intent(2), intent(3)]
    for item in pending:
        queue.append(item)
    submitted: list[Any] = []
    assert (
        queue.submit_selected(lambda item: record_submission(submitted, item), {3, 1})
        == 2
    )
    assert submitted == [pending[0], pending[2]]
    assert queue.pending_job_ids == (2,)
    assert queue.metadata_bytes == pending[1].metadata_bytes
    assert queue.submit_selected(lambda item: record_submission(submitted, item)) == 1
    assert submitted == [pending[0], pending[2], pending[1]]
    assert not queue and queue.metadata_bytes == 0


def test_clean_rejection_keeps_intact_intent_and_stops_submission():
    queue = Queue()
    first, second = intent(1), intent(2)
    queue.append(first)
    queue.append(second)
    calls: list[Any] = []
    assert (
        queue.submit_selected(
            lambda item: record_submission(calls, item, accepted=False)
        )
        == 0
    )
    assert calls == [first]
    assert queue.pending_job_ids == (1, 2)
    assert queue.metadata_bytes == 16
    accepted: list[Any] = []
    assert queue.submit_selected(lambda item: record_submission(accepted, item)) == 2
    assert accepted == [first, second]


def test_exception_retains_unknown_job_and_prevents_duplicate_retry():
    queue = Queue()
    items = [intent(1), intent(2), intent(3)]
    for item in items:
        queue.append(item)
    calls: list[Any] = []

    def submit(item):
        calls.append(item)
        if item.job_id == 2:
            raise RuntimeError("possibly accepted")
        return True

    with pytest.raises(RuntimeError, match="possibly accepted"):
        queue.submit_selected(submit)
    assert calls == items[:2]
    assert queue.pending_job_ids == (2, 3)
    assert queue.metadata_bytes == 16
    with pytest.raises(RuntimeError, match="ownership is retained"):
        queue.submit_selected(submit)
    assert calls == items[:2]


@pytest.mark.parametrize("limit", ["jobs", "bytes"])
def test_budget_rejection_does_not_accept_or_release_caller_ownership(limit):
    queue = Queue(max_jobs=1 if limit == "jobs" else 2, max_metadata_bytes=8)
    first, second = intent(1), intent(2, charge=0 if limit == "jobs" else 1)
    queue.append(first)
    with pytest.raises(BufferError):
        queue.append(second)
    assert queue.pending_job_ids == (1,)
    assert queue.metadata_bytes == 8
    assert second.producer is not None
    assert queue.submit_selected(lambda _: True) == 1
    queue.append(second)
    assert queue.pending_job_ids == (2,)


def test_duplicate_and_empty_flush_do_not_change_ownership():
    queue = Queue()
    queue.append(intent(1))
    with pytest.raises(ValueError, match="Duplicate"):
        queue.append(intent(1))
    assert (
        queue.submit_selected(lambda _: pytest.fail("ordinary store submitted"), set())
        == 0
    )
    assert queue.pending_job_ids == (1,)
    assert queue.metadata_bytes == 8


def test_callback_must_distinguish_acceptance_from_unknown_outcome():
    queue = Queue()
    queue.append(intent(1))
    with pytest.raises(TypeError, match="acceptance boolean"):
        queue.submit_selected(lambda _: None)
    assert queue.pending_job_ids == (1,)
    with pytest.raises(RuntimeError, match="ownership is retained"):
        queue.submit_selected(lambda _: True)


def test_submission_gate_is_bound_once_and_survives_clean_rejection():
    """Late submission must not replace the original sampling boundary."""
    producer, gate, newer = object(), object(), object()
    queue = module.DeferredStoreQueue()
    original = module.CapturedStoreIntent(71, object(), object(), producer, 8)
    queue.append(original)
    assert queue.bind_submission_gate(lambda: gate) == 1
    assert queue.submit_selected(lambda intent: False) == 0
    assert queue.bind_submission_gate(lambda: newer) == 0
    accepted: list[Any] = []
    assert (
        queue.submit_selected(lambda intent: record_submission(accepted, intent)) == 1
    )
    assert accepted[0].producer.events == (producer, gate)
    assert accepted[0].submission_gate_bound
    assert not original.submission_gate_bound


def test_recorded_dependency_bundle_waits_for_both_original_boundaries():
    from types import SimpleNamespace

    observed: list[Any] = []
    first = SimpleNamespace(query=lambda: True)
    second = SimpleNamespace(query=lambda: False)
    deps = module.RecordedDependencies((first, second))
    assert not deps.query()
    deps.wait_on(SimpleNamespace(wait_event=observed.append))
    assert observed == [first, second]


@pytest.mark.parametrize("limits", [(0, 1), (1, 0), (-1, 10), (True, 1)])
def test_deferred_store_limits_must_be_positive_integers(limits):
    with pytest.raises(ValueError):
        module.DeferredStoreQueue(*limits)
