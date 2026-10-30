# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned store captures and rank-local deferral, without device scheduling.

The worker adapter snapshots specs and captures the original producer boundary.
The existing executor remains responsible for accepted jobs and DMA progress.
This queue is model-thread confined; it does not transfer ownership across ranks.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class RecordedDependencies:
    """Own event records already issued at the original production boundaries."""

    events: tuple[Any, ...]

    def query(self) -> bool:
        return all(event.query() for event in self.events)

    def wait_on(self, stream: Any) -> None:
        for event in self.events:
            if isinstance(event, RecordedDependencies):
                event.wait_on(stream)
            else:
                stream.wait_event(event)


@dataclass(frozen=True, slots=True)
class CapturedStoreIntent:
    """Own immutable spec snapshots and an already-recorded producer dependency.

    The adapter must provide independent descriptor storage, not read-only views
    of mutable backing. The opaque producer remains owned by the intent until
    the executor accepts it, and must not be rebound to a later model step.
    """

    job_id: int
    src_spec: Any
    dst_spec: Any
    producer: Any
    metadata_bytes: int
    submission_gate_bound: bool = False

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id < 0:
            raise ValueError("Store job ID must be a nonnegative integer")
        if type(self.metadata_bytes) is not int or self.metadata_bytes < 0:
            raise ValueError("Store metadata charge must be nonnegative")
        if self.producer is None:
            raise ValueError("A captured store requires its producer dependency")


def capture_store_intent(
    job_id: int,
    src_spec: Any,
    dst_spec: Any,
    *,
    snapshot_specs: Callable[[Any, Any], tuple[Any, Any, int]],
    capture_producer: Callable[[], Any],
) -> CapturedStoreIntent:
    """Snapshot descriptors before capturing their original producer boundary.

    Args:
        job_id: Existing connector job identity, local to this worker's engine.
        src_spec: Source spec owned by the caller.
        dst_spec: Destination spec owned by the caller.
        snapshot_specs: Return owned source/destination snapshots and their
            metadata byte charge. Device/layout details belong to this adapter.
        capture_producer: Record the original producer dependency, or return an
            owned dependency already captured for this same proven boundary.

    """
    if type(job_id) is not int or job_id < 0:
        raise ValueError("Store job ID must be a nonnegative integer")
    source, destination, charge = snapshot_specs(src_spec, dst_spec)
    if type(charge) is not int or charge < 0:
        raise ValueError("Store metadata charge must be nonnegative")
    return CapturedStoreIntent(job_id, source, destination, capture_producer(), charge)


class DeferredStoreQueue:
    """Bound captured stores until the existing executor accepts them.

    Selecting flush dependencies does not submit unrelated stores. Successful
    submission moves ownership to the callback; a clean rejection keeps it here.
    An exception has an unknown submission outcome: retain pending ownership and
    latch the failure rather than automatically retrying possibly accepted DMA.
    """

    def __init__(
        self, max_jobs: int = 256, max_metadata_bytes: int = 64 * 1024**2
    ) -> None:
        if type(max_jobs) is not int or max_jobs <= 0:
            raise ValueError("Deferred store job budget must be positive")
        if type(max_metadata_bytes) is not int or max_metadata_bytes <= 0:
            raise ValueError("Deferred store metadata budget must be positive")
        self.max_jobs = max_jobs
        self.max_metadata_bytes = max_metadata_bytes
        self._pending: dict[int, CapturedStoreIntent] = {}
        self._metadata_bytes = 0
        self._submission_error: Exception | None = None

    def __len__(self) -> int:
        return len(self._pending)

    @property
    def metadata_bytes(self) -> int:
        return self._metadata_bytes

    @property
    def pending_job_ids(self) -> tuple[int, ...]:
        return tuple(self._pending)

    def bind_submission_gate(self, capture_gate: Callable[[], Any]) -> int:
        """Bind each intent once to its original next-step issue boundary."""
        self._check_submission()
        if not any(
            not intent.submission_gate_bound for intent in self._pending.values()
        ):
            return 0
        gate = capture_gate()
        if gate is None:
            raise ValueError("Submission gate must already have been recorded")
        bound = 0
        for job_id, intent in self._pending.items():
            if not intent.submission_gate_bound:
                self._pending[job_id] = replace(
                    intent,
                    producer=RecordedDependencies((intent.producer, gate)),
                    submission_gate_bound=True,
                )
                bound += 1
        return bound

    def _check_submission(self) -> None:
        if self._submission_error is not None:
            raise RuntimeError(
                "Deferred store submission failed; pending ownership is retained"
            ) from self._submission_error

    def append(self, intent: CapturedStoreIntent) -> None:
        """Accept ownership, or raise without accepting/releasing the intent."""
        self._check_submission()
        if intent.job_id in self._pending:
            raise ValueError(f"Duplicate deferred store job {intent.job_id}")
        if (
            len(self._pending) >= self.max_jobs
            or self._metadata_bytes + intent.metadata_bytes > self.max_metadata_bytes
        ):
            raise BufferError("Deferred store reservation budget is exhausted")
        self._pending[intent.job_id] = intent
        self._metadata_bytes += intent.metadata_bytes

    def submit_selected(
        self,
        submit: Callable[[CapturedStoreIntent], bool],
        job_ids: Collection[int] | None = None,
    ) -> int:
        """Submit matching jobs in FIFO order, retaining all unaccepted intents.

        ``None`` selects ordinary pending work; a collection selects only those
        flush dependencies. The callback must return True only after accepting
        ownership, or False for a clean rejection. Stop at the first rejection.
        The callback must not mutate this model-thread-confined queue.
        """
        self._check_submission()
        if not self._pending or (job_ids is not None and not job_ids):
            return 0
        accepted = 0
        for job_id in tuple(self._pending):
            if job_ids is not None and job_id not in job_ids:
                continue
            intent = self._pending[job_id]
            try:
                outcome = submit(intent)
                if type(outcome) is not bool:
                    raise TypeError(
                        "Store submission must return an acceptance boolean"
                    )
            except Exception as error:
                self._submission_error = error
                raise
            if not outcome:
                break
            del self._pending[job_id]
            self._metadata_bytes -= intent.metadata_bytes
            accepted += 1
        return accepted
