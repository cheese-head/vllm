# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local copy scheduling; device and layout operations belong to a backend."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Protocol

Direction = Literal["load", "store"]
_DIRECTIONS: tuple[Direction, ...] = ("load", "store")
_RETAINED_EXECUTORS: list[Any] = []


@dataclass(frozen=True)
class CopyExecutorConfig:
    max_pending_jobs: int = 256
    max_pending_bytes: int = 64 * 1024**2
    max_inflight_batches: int = 8
    max_batch_descriptors: int = 4096
    target_batch_bytes: int = 64 * 1024**2
    admission_timeout: float = 60.0
    fence_timeout: float = 600.0
    shutdown_timeout: float = 60.0

    @classmethod
    def from_extra_config(cls, extra: Mapping[str, Any]):
        options = extra.get("cpu_copy_executor")
        if options is None:
            return None
        if not isinstance(options, Mapping):
            raise ValueError("cpu_copy_executor must be a mapping")
        options = dict(options)
        enabled = options.pop("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("cpu_copy_executor.enabled must be boolean")
        unknown = options.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown cpu_copy_executor options: {sorted(unknown)}")
        config = cls(**options)
        return config if enabled else None

    def __post_init__(self) -> None:
        for name in (
            "max_pending_jobs",
            "max_pending_bytes",
            "max_inflight_batches",
            "max_batch_descriptors",
            "target_batch_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("admission_timeout", "fence_timeout", "shutdown_timeout"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value < float("inf"):
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class CopyChunk:
    data: Any
    descriptors: int
    nbytes: int
    final: bool


@dataclass(frozen=True)
class PhysicalCopyStats:
    nbytes: int
    seconds: float


@dataclass(frozen=True)
class CopyCompletion:
    job_id: int
    nbytes: int
    physical_stats: tuple[PhysicalCopyStats, ...]


class CopyBackend(Protocol):
    def setup(self) -> None: ...
    def chunks(self, payload: Any, config: CopyExecutorConfig): ...
    def ready(self, producer: Any) -> bool: ...
    def extend_batch(self, scope: Any, candidate: CopyChunk) -> Any: ...
    def launch(self, direction: Direction, parts, producers) -> Any: ...
    def query(self, ticket: Any) -> bool: ...
    def finish(self, ticket: Any) -> PhysicalCopyStats: ...
    def close(self) -> None: ...


@dataclass(eq=False)
class _Job:
    job_id: int
    direction: Direction
    payload: Any
    producer: Any
    charge: int
    enqueued: float = field(default_factory=time.monotonic)
    iterator: Any = None
    chunk: CopyChunk | None = None
    exhausted: bool = False
    inflight: int = 0
    nbytes: int = 0
    started: bool = False


@dataclass
class _Batch:
    ticket: Any
    jobs: list[_Job]
    submitted: float


class NativeCopyExecutor:
    """One owner of device submissions per worker, with bounded preparation.

    Queue charges cover queued and in-flight immutable job metadata. An
    oversized metadata job is admitted alone to guarantee forward progress.
    Terminal acknowledgments contain no copy buffers; like the scheduler's
    job map, they remain until the model worker polls them. Callers must poll
    completions between scheduling steps. Accepted jobs are never dropped.
    """

    def __init__(self, backend: CopyBackend, config: CopyExecutorConfig):
        self.backend = backend
        self.config = config
        self._cv = threading.Condition()
        self._queues: dict[Direction, deque[_Job]] = {d: deque() for d in _DIRECTIONS}
        self._batches: dict[Direction, deque[_Batch]] = {
            d: deque() for d in _DIRECTIONS
        }
        self._live: dict[int, _Job] = {}
        self._known: set[int] = set()
        self._completed: deque[CopyCompletion] = deque()
        self._unreported_stats: dict[str, list[PhysicalCopyStats]] = {
            "load": [],
            "store": [],
        }
        self._pending_bytes = 0
        self._urgent: set[int] = set()
        self._next: Direction = "load"
        self._closing = False
        self._ready = False
        self._error: BaseException | None = None
        self.quiescent = False
        self._stats = dict(
            accepted_jobs=0,
            completed_jobs=0,
            physical_batches=0,
            queue_seconds=0.0,
            preparation_seconds=0.0,
            observed_batch_seconds=0.0,
            admission_wait_seconds=0.0,
            fence_wait_seconds=0.0,
            pending_jobs_high_water=0,
            pending_bytes_high_water=0,
        )
        self._thread = threading.Thread(
            target=self._run, name="native-cpu-copy", daemon=True
        )
        self._thread.start()
        with self._cv:
            if not self._cv.wait_for(
                lambda: self._ready or self._error is not None,
                timeout=config.shutdown_timeout,
            ):
                self._closing = True
                self._cv.notify_all()
                raise TimeoutError("Native copy executor initialization timed out")
            self._check()

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError("Native CPU copy executor failed") from self._error

    def submit(self, job_id, direction, payload, producer, charge) -> None:
        if direction not in self._queues or charge < 0:
            raise ValueError("Invalid copy direction or metadata charge")
        start = time.monotonic()
        deadline = start + self.config.admission_timeout
        with self._cv:
            self._check()
            if job_id in self._known:
                raise ValueError(f"Duplicate native copy job {job_id}")
            while self._live and (
                len(self._live) >= self.config.max_pending_jobs
                or self._pending_bytes + charge > self.config.max_pending_bytes
            ):
                self._check()
                if self._closing:
                    raise RuntimeError("Native copy executor is closing")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Native copy admission budget did not drain")
                self._cv.wait(remaining)
            self._check()
            if self._closing:
                raise RuntimeError("Native copy executor is closing")
            if job_id in self._known:
                raise ValueError(f"Duplicate native copy job {job_id}")
            job = _Job(job_id, direction, payload, producer, charge)
            self._known.add(job_id)
            self._live[job_id] = job
            self._pending_bytes += charge
            self._queues[direction].append(job)
            self._stats["accepted_jobs"] += 1
            self._stats["admission_wait_seconds"] += time.monotonic() - start
            self._stats["pending_jobs_high_water"] = max(
                self._stats["pending_jobs_high_water"], len(self._live)
            )
            self._stats["pending_bytes_high_water"] = max(
                self._stats["pending_bytes_high_water"], self._pending_bytes
            )
            self._cv.notify_all()

    def get_finished(self) -> list[CopyCompletion]:
        with self._cv:
            self._check()
            results = list(self._completed)
            self._completed.clear()
            self._known.difference_update(r.job_id for r in results)
            return results

    def wait(self, job_ids: set[int]) -> None:
        start = time.monotonic()
        deadline = start + self.config.fence_timeout
        with self._cv:
            self._urgent.update(job_ids)
            self._cv.notify_all()
            try:
                while not job_ids.isdisjoint(self._live):
                    self._check()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Native copy reuse fence did not drain")
                    self._cv.wait(remaining)
                self._check()
            finally:
                self._urgent.difference_update(job_ids)
                self._stats["fence_wait_seconds"] += time.monotonic() - start

    def statistics(self) -> dict[str, int | float]:
        with self._cv:
            return dict(
                self._stats,
                pending_jobs=len(self._live),
                pending_metadata_bytes=self._pending_bytes,
                completed_unpolled=len(self._completed),
            )

    def close(self) -> None:
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        self._thread.join(self.config.shutdown_timeout)
        if self._thread.is_alive():
            raise TimeoutError("Native copy executor has not quiesced")
        with self._cv:
            self._check()
            if not self.quiescent:
                raise RuntimeError("Native copy backend did not quiesce")

    def _peek(self, job: _Job) -> CopyChunk:
        if job.chunk is None:
            start = time.monotonic()
            if job.iterator is None:
                job.iterator = iter(self.backend.chunks(job.payload, self.config))
            job.chunk = next(job.iterator)
            if job.chunk.descriptors < 0 or job.chunk.nbytes < 0:
                raise ValueError("Negative copy chunk dimensions")
            with self._cv:
                self._stats["preparation_seconds"] += time.monotonic() - start
        return job.chunk

    def _launch_one(self, direction: Direction) -> None:
        queue = self._queues[direction]
        parts: list[CopyChunk] = []
        jobs: list[_Job] = []
        producers: list[Any] = []
        scope = None
        descriptors = nbytes = 0
        while queue:
            with self._cv:
                job = queue[0]
            # Do not delay an earlier batch for a later producer dependency.
            if jobs and job is not jobs[-1] and not self.backend.ready(job.producer):
                break
            chunk = self._peek(job)
            if parts and (
                descriptors + chunk.descriptors > self.config.max_batch_descriptors
                or nbytes + chunk.nbytes > self.config.target_batch_bytes
            ):
                break
            extended = self.backend.extend_batch(scope, chunk)
            if parts and extended is None:
                break
            scope = extended
            parts.append(chunk)
            jobs.append(job)
            producers.append(job.producer)
            descriptors += chunk.descriptors
            nbytes += chunk.nbytes
            with self._cv:
                if not job.started:
                    self._stats["queue_seconds"] += time.monotonic() - job.enqueued
                    job.started = True
                job.inflight += 1
                job.nbytes += chunk.nbytes
                job.chunk = None
                if chunk.final:
                    job.exhausted = True
                    queue.popleft()
            # A single layout block may exceed a target; submit it alone.
            if (
                descriptors >= self.config.max_batch_descriptors
                or nbytes >= self.config.target_batch_bytes
            ):
                break
        ticket = self.backend.launch(direction, parts, producers)
        with self._cv:
            self._batches[direction].append(_Batch(ticket, jobs, time.monotonic()))
            self._stats["physical_batches"] += 1

    def _poll(self) -> bool:
        progress = False
        for direction in _DIRECTIONS:
            batches = self._batches[direction]
            while batches and self.backend.query(batches[0].ticket):
                batch = batches[0]
                stats = self.backend.finish(batch.ticket)
                with self._cv:
                    batches.popleft()
                    self._unreported_stats[direction].append(stats)
                    self._stats["observed_batch_seconds"] += (
                        time.monotonic() - batch.submitted
                    )
                    for job in batch.jobs:
                        job.inflight -= 1
                        if job.exhausted and job.inflight == 0:
                            self._completed.append(
                                CopyCompletion(
                                    job.job_id,
                                    job.nbytes,
                                    tuple(self._unreported_stats[direction]),
                                )
                            )
                            self._unreported_stats[direction].clear()
                            del self._live[job.job_id]
                            self._pending_bytes -= job.charge
                            self._stats["completed_jobs"] += 1
                    self._cv.notify_all()
                progress = True
        return progress

    def _run(self) -> None:
        try:
            self.backend.setup()
            with self._cv:
                self._ready = True
                self._cv.notify_all()
            delay = 0.00005
            while True:
                progress = self._poll()
                with self._cv:
                    if self._closing and not self._live:
                        break
                    capacity = sum(map(len, self._batches.values()))
                    available = [d for d in _DIRECTIONS if self._queues[d]]
                    urgent = [
                        d
                        for d in available
                        if any(
                            self._live[j].direction == d
                            for j in self._urgent
                            if j in self._live
                        )
                    ]
                    direction = next(
                        (d for d in urgent or [self._next] if d in available),
                        available[0] if available else None,
                    )
                if direction and capacity < self.config.max_inflight_batches:
                    self._launch_one(direction)
                    self._next = "store" if direction == "load" else "load"
                    delay = 0.00005
                    continue
                with self._cv:
                    # Device completion needs polling; host producers/fence
                    # waiters use notifications and never sleep-poll here.
                    self._cv.wait(delay if self._live else None)
                delay = 0.00005 if progress else min(delay * 2, 0.001)
            self.backend.close()
            self.quiescent = True
        except BaseException as error:
            with self._cv:
                self._error = error
                self._cv.notify_all()
            try:
                self.backend.close()
                self.quiescent = True
            except BaseException as close_error:
                add_note = getattr(error, "add_note", None)
                if add_note is not None:
                    add_note(f"Copy backend did not quiesce: {close_error}")
                _RETAINED_EXECUTORS.append(self)
        finally:
            with self._cv:
                self._cv.notify_all()
