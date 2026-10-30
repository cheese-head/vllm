# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts: ordering, admission, accounting, failure and quiescence.

The backend simulates stream dependencies; these tests do not claim CUDA
visibility or distributed execution coverage. Load the dependency-free
production module without importing a GPU vLLM installation.
"""

import importlib.util
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "vllm/v1/kv_offload/cpu/copy_executor.py"
spec = importlib.util.spec_from_file_location("native_copy_executor_tested", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Config = module.CopyExecutorConfig
Executor = module.NativeCopyExecutor
Chunk = module.CopyChunk
Stats = module.PhysicalCopyStats


def event(ready=True):
    value = threading.Event()
    if ready:
        value.set()
    return value


def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Copy executor did not make progress")
        time.sleep(0.001)


class Backend:
    def __init__(self):
        self.gate = event()
        self.launched = []
        self.owner = None
        self.closed = False
        self.tail = {}

    def setup(self):
        self.owner = threading.get_ident()

    def chunks(self, payload, config):
        assert threading.get_ident() == self.owner
        for i, (key, size) in enumerate(payload):
            yield Chunk(key, 1, size, i == len(payload) - 1)
        if not payload:
            yield Chunk(None, 0, 0, True)

    def ready(self, producer):
        return producer.is_set()

    def extend_batch(self, scope, candidate):
        scope = scope or frozenset()
        return None if candidate.data in scope else scope | {candidate.data}

    def launch(self, direction, parts, producers):
        assert threading.get_ident() == self.owner
        ticket = NS(
            direction=direction,
            parts=parts,
            producers=producers,
            previous=self.tail.get(direction),
            retired=False,
        )
        self.tail[direction] = ticket
        self.launched.append(ticket)
        return ticket

    def query(self, ticket):
        return (
            self.gate.is_set()
            and all(p.is_set() for p in ticket.producers)
            and (ticket.previous is None or ticket.previous.retired)
        )

    def finish(self, ticket):
        assert self.query(ticket)
        ticket.retired = True
        return Stats(sum(p.nbytes for p in ticket.parts), 0.01)

    def close(self):
        assert all(t.retired for t in self.launched)
        self.closed = True


def test_chunks_complete_once_and_physical_bytes_are_not_double_counted():
    backend = Backend()
    executor = Executor(backend, Config(max_batch_descriptors=2))
    try:
        executor.submit(17, "store", [(1, 5), (2, 6), (3, 7)], event(), 24)
        executor.submit(8, "load", [(4, 9)], event(), 8)
        executor.wait({8, 17})
        results = executor.get_finished()
        assert {r.job_id: r.nbytes for r in results} == {17: 18, 8: 9}
        by_id = {r.job_id: sum(s.nbytes for s in r.physical_stats) for r in results}
        assert by_id == {17: 18, 8: 9}
        assert executor.get_finished() == []
        assert backend.owner != threading.get_ident()
    finally:
        executor.close()
    assert backend.closed and executor.quiescent


def test_later_producer_does_not_block_an_earlier_ready_copy():
    backend = Backend()
    executor = Executor(backend, Config(max_inflight_batches=1))
    blocked = event(False)
    try:
        executor.submit(99, "store", [(1, 1)], event(), 1)
        executor.wait({99})
        executor.submit(2, "store", [(2, 1)], blocked, 1)
        until(lambda: len(backend.launched) == 2)
        assert executor.get_finished()[0].job_id == 99
        assert executor.statistics()["pending_jobs"] == 1
        blocked.set()
        executor.wait({2})
        assert executor.get_finished()[0].job_id == 2
    finally:
        blocked.set()
        executor.close()


def test_full_queue_drains_without_dropping_jobs_or_requiring_result_poll():
    backend = Backend()
    backend.gate.clear()
    executor = Executor(backend, Config(max_pending_jobs=1, max_pending_bytes=4))
    accepted = threading.Event()
    try:
        executor.submit(1, "store", [(1, 1)], event(), 8)  # Oversized singleton.

        def submit():
            executor.submit(2, "load", [(2, 1)], event(), 4)
            accepted.set()

        thread = threading.Thread(target=submit)
        thread.start()
        assert not accepted.wait(0.02)
        backend.gate.set()
        assert accepted.wait(3)
        thread.join()
        executor.wait({1, 2})
        assert {r.job_id for r in executor.get_finished()} == {1, 2}
        assert executor.statistics()["pending_jobs_high_water"] == 1
    finally:
        backend.gate.set()
        executor.close()


def test_coalescing_preserves_logical_jobs_and_limits():
    backend = Backend()
    backend.gate.clear()
    executor = Executor(
        backend,
        Config(
            max_inflight_batches=1,
            max_batch_descriptors=2,
            target_batch_bytes=4,
        ),
    )
    try:
        executor.submit(0, "load", [(0, 2)], event(), 1)
        until(lambda: bool(backend.launched))
        for i in range(1, 6):
            executor.submit(i, "load", [(i, 2)], event(), 1)
        backend.gate.set()
        executor.wait(set(range(6)))
        results = executor.get_finished()
        assert len(results) == 6 and len({r.job_id for r in results}) == 6
        assert any(len(t.parts) == 2 for t in backend.launched)
        assert all(len(t.parts) <= 2 for t in backend.launched)
        assert sum(s.nbytes for r in results for s in r.physical_stats) == 12
    finally:
        backend.gate.set()
        executor.close()


def test_conflicting_chunks_are_kept_in_separate_submissions():
    backend = Backend()
    executor = Executor(backend, Config(max_batch_descriptors=8))
    try:
        executor.submit(1, "load", [(5, 1), (5, 1), (5, 1)], event(), 1)
        executor.wait({1})
        assert len(backend.launched) == 3
        assert executor.get_finished()[0].nbytes == 3
    finally:
        executor.close()


def test_fifo_wait_uses_requested_jobs_not_numeric_ids_or_unrelated_tail():
    backend = Backend()
    executor = Executor(backend, Config(max_inflight_batches=2))
    blocked = event(False)
    try:
        executor.submit(99, "store", [(1, 1)], event(), 1)
        executor.submit(2, "store", [(2, 1)], blocked, 1)
        executor.wait({99})
        assert executor.statistics()["pending_jobs"] == 1
        assert [r.job_id for r in executor.get_finished()] == [99]
        blocked.set()
        executor.wait({2})
    finally:
        blocked.set()
        executor.close()


def test_a_store_is_served_when_loads_are_also_queued():
    backend = Backend()
    backend.gate.clear()
    executor = Executor(
        backend, Config(max_inflight_batches=1, max_batch_descriptors=1)
    )
    try:
        executor.submit(0, "load", [(0, 1)], event(), 1)
        until(lambda: len(backend.launched) == 1)
        for i in range(1, 12):
            executor.submit(i, "load", [(i, 1)], event(), 1)
        executor.submit(40, "store", [(40, 1)], event(), 1)
        backend.gate.set()
        until(lambda: any(t.direction == "store" for t in backend.launched))
        assert (
            next(i for i, t in enumerate(backend.launched) if t.direction == "store")
            <= 2
        )
        executor.wait(set(range(12)) | {40})
    finally:
        backend.gate.set()
        executor.close()


def test_duplicate_ids_are_rejected_until_results_are_consumed():
    backend = Backend()
    executor = Executor(backend, Config())
    try:
        executor.submit(7, "store", [], event(), 0)
        executor.wait({7})
        with pytest.raises(ValueError, match="Duplicate"):
            executor.submit(7, "load", [], event(), 0)
        assert executor.get_finished()[0].job_id == 7
        executor.submit(7, "load", [], event(), 0)
        executor.wait({7})
    finally:
        executor.close()


def test_backend_failure_is_terminal_and_waiters_are_not_stranded():
    class Failed(Backend):
        def launch(self, *_):
            raise OSError("driver rejected submission")

    backend = Failed()
    executor = Executor(backend, Config())
    executor.submit(1, "load", [(1, 1)], event(), 1)
    with pytest.raises(RuntimeError, match="executor failed") as error:
        executor.wait({1})
    assert isinstance(error.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match="executor failed"):
        executor.close()
    assert executor.quiescent


def test_unsafe_cleanup_failure_retains_backend_and_reservations():
    class Failed(Backend):
        def launch(self, *_):
            raise OSError("partly accepted submission")

        def close(self):
            raise OSError("device did not quiesce")

    executor = Executor(Failed(), Config())
    executor.submit(1, "store", [(1, 1)], event(), 1)
    with pytest.raises(RuntimeError):
        executor.wait({1})
    with pytest.raises(RuntimeError):
        executor.close()
    assert not executor.quiescent
    assert executor in module._RETAINED_EXECUTORS
    assert executor.statistics()["pending_jobs"] == 1
    module._RETAINED_EXECUTORS.remove(executor)  # Fake backend has no DMA.


def test_close_drains_and_rejects_further_submissions():
    executor = Executor(Backend(), Config())
    executor.submit(1, "store", [(1, 1)], event(), 1)
    executor.close()
    assert executor.quiescent
    assert executor.get_finished()[0].job_id == 1
    with pytest.raises(RuntimeError, match="closing"):
        executor.submit(2, "store", [], event(), 0)
    executor.close()


@pytest.mark.parametrize(
    "options",
    [
        {"max_pending_jobs": 0},
        {"max_pending_bytes": -1},
        {"max_batch_descriptors": True},
        {"admission_timeout": float("nan")},
        {"shutdown_timeout": float("inf")},
        {"enabled": "yes"},
        {"typo": 2},
    ],
)
def test_invalid_executor_configuration_fails_before_device_initialization(options):
    with pytest.raises((ValueError, TypeError)):
        Config.from_extra_config({"cpu_copy_executor": options})


def test_configuration_is_opt_in_and_strict_even_when_disabled():
    assert Config.from_extra_config({}) is None
    assert Config.from_extra_config({"cpu_copy_executor": {"enabled": False}}) is None
    config = Config.from_extra_config(
        {
            "cpu_copy_executor": {
                "enabled": True,
                "max_pending_jobs": 8,
            }
        }
    )
    assert config.max_pending_jobs == 8


def test_many_jobs_preserve_exact_ids_bytes_and_direction_under_pressure():
    """Admission, splitting and polling must not lose a logical completion."""
    rng = random.Random(31)
    backend = Backend()
    executor = Executor(
        backend,
        Config(max_pending_jobs=7, max_pending_bytes=17, max_batch_descriptors=3),
    )
    expected: dict[int, tuple[str, int]] = {}
    actual = {}
    physical = {"load": 0, "store": 0}

    def collect():
        for result in executor.get_finished():
            assert result.job_id not in actual
            actual[result.job_id] = result.nbytes
            direction, _ = expected[result.job_id]
            physical[direction] += sum(s.nbytes for s in result.physical_stats)

    try:
        for job_id in range(1024):
            direction = rng.choice(["load", "store"])
            sizes = [rng.randrange(1, 129) for _ in range(rng.randrange(1, 9))]
            expected[job_id] = (direction, sum(sizes))
            executor.submit(job_id, direction, list(enumerate(sizes)), event(), 4)
            if job_id % 11 == 0:
                collect()
        executor.wait(set(expected))
        collect()
        assert actual == {j: size for j, (_, size) in expected.items()}
        assert physical == {
            d: sum(n for direction, n in expected.values() if direction == d)
            for d in physical
        }
        assert executor.statistics()["pending_jobs"] == 0
        assert executor.statistics()["pending_bytes_high_water"] <= 17
    finally:
        executor.close()


@pytest.mark.parametrize("operation", ["admission", "fence", "shutdown"])
def test_deadlines_do_not_acknowledge_or_discard_unfinished_work(operation):
    backend = Backend()
    backend.gate.clear()
    executor = Executor(
        backend,
        Config(
            max_pending_jobs=1,
            admission_timeout=0.01,
            fence_timeout=0.01,
            shutdown_timeout=0.5,
        ),
    )
    try:
        executor.submit(1, "load", [(1, 16)], event(), 1)
        until(lambda: bool(backend.launched))
        with pytest.raises(TimeoutError):
            if operation == "admission":
                executor.submit(2, "store", [(2, 16)], event(), 1)
            elif operation == "fence":
                executor.wait({1})
            else:
                executor.close()
        assert not executor.quiescent
        assert executor.statistics()["pending_jobs"] == 1
        assert executor.get_finished() == []
        backend.gate.set()
        until(lambda: executor.statistics()["completed_jobs"] == 1)
        assert [r.job_id for r in executor.get_finished()] == [1]
    finally:
        backend.gate.set()
        executor.close()


@pytest.mark.parametrize("mode", ["legacy", "capture_only", "selective"])
def test_lifecycle_mode_is_explicit_and_does_not_enable_executor(monkeypatch, mode):
    monkeypatch.setenv("VLLM_NATIVE_CPU_LIFECYCLE_MODE", mode)
    assert Config.from_extra_config({}) is None
    assert Config.from_extra_config({"cpu_copy_executor": {"enabled": False}}) is None
    config = Config.from_extra_config({"cpu_copy_executor": {"enabled": True}})
    assert config.lifecycle_mode == mode
    explicit = Config.from_extra_config(
        {"cpu_copy_executor": {"enabled": True, "lifecycle_mode": "legacy"}}
    )
    assert explicit.lifecycle_mode == "legacy"


def test_unknown_lifecycle_mode_is_rejected_before_executor_start(monkeypatch):
    monkeypatch.setenv("VLLM_NATIVE_CPU_LIFECYCLE_MODE", "unsafe")
    with pytest.raises(ValueError, match="lifecycle_mode"):
        Config.from_extra_config({"cpu_copy_executor": {"enabled": True}})
