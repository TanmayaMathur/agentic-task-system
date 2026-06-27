"""Manual batching of LLM requests for the Agentic AI System.

This module implements :class:`Batcher`, the component that groups individual
agent LLM requests into batches by *size* or *time window* and flushes each
batch as a single provider call. It is implemented explicitly -- as a single
background consumer loop draining an :class:`asyncio.Queue` -- so the batching
control flow is fully visible and testable, with no black-box framework hiding
the logic (Requirement 10.1).

How it works
------------
A producer calls :meth:`Batcher.submit`, which enqueues a
:class:`~agentic.models.BatchRequest` and awaits its future. A single
background task (:meth:`Batcher._run_loop`) builds batches using two
independent triggers:

* **Size trigger** -- the batch reaches ``max_batch_size`` (Requirement 6.2).
* **Time-window trigger** -- ``max_wait_ms`` has elapsed since the batch's
  *first* request, even if the batch is not full (Requirements 6.3, 6.7).

The window is fixed by the first request and is never extended by later
arrivals (Requirement 6.7). Every flushed batch therefore satisfies
``1 <= len(batch) <= max_batch_size`` (Requirement 6.4).

On flush (:meth:`Batcher._flush`), the batch's prompts are sent through the
:class:`~agentic.failure.FailureHandler` so the call benefits from
retry/backoff, the circuit breaker, and a degraded fallback. The completion at
index ``i`` is delivered to the request at index ``i`` (Requirement 6.5); if the
call raises, every pending request in the batch has its future set to that
exception. Either way, every submitted future is eventually resolved
(Requirement 6.6).

Testability
-----------
The clock is an injectable constructor parameter (defaulting to
:func:`time.monotonic`) so tests can stamp request arrival times deterministically.
A monotonic clock is used so the window math is immune to wall-clock adjustments.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Callable, Optional, Union

from .failure import FailureHandler
from .models import BatchRequest
from .providers import Completion, LLMProvider

__all__ = ["Batcher"]

#: A monotonic clock returning seconds.
Clock = Callable[[], float]

# Sensible defaults for an implicitly-created FailureHandler when the caller
# does not inject one. Small delays keep the happy path snappy while still
# exercising the retry/fallback path on transient errors.
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_BASE_DELAY_MS = 10
_DEFAULT_BREAKER_THRESHOLD = 5


class Batcher:
    """Group agent LLM requests into batches and flush them as one call.

    Args:
        provider: The :class:`~agentic.providers.LLMProvider` used to complete a
            batch's prompts. Its :meth:`complete` MUST return completions in
            input order.
        max_batch_size: Maximum number of requests in a single flushed batch.
            Must be ``>= 1``.
        max_wait_ms: Maximum time, in milliseconds, the *first* request of a
            batch waits before the batch is flushed even if not full. Must be
            ``>= 0`` (``0`` flushes as soon as the loop yields).
        failure: Optional :class:`~agentic.failure.FailureHandler` wrapping the
            provider call. When ``None``, a handler with sensible defaults is
            created so the degraded fallback path still works.
        clock: Monotonic clock returning seconds, injectable for tests. Used to
            stamp request arrival times and compute the window deadline.
            Defaults to :func:`time.monotonic`.

    Raises:
        ValueError: If ``max_batch_size < 1`` or ``max_wait_ms < 0``.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_batch_size: int,
        max_wait_ms: int,
        failure: Optional[FailureHandler] = None,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        if not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be an integer >= 1")
        if not isinstance(max_wait_ms, (int, float)) or max_wait_ms < 0:
            raise ValueError("max_wait_ms must be a number >= 0")

        self._provider = provider
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._failure = failure if failure is not None else FailureHandler(
            max_retries=_DEFAULT_MAX_RETRIES,
            base_delay_ms=_DEFAULT_BASE_DELAY_MS,
            breaker_threshold=_DEFAULT_BREAKER_THRESHOLD,
        )
        self._clock = clock

        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._counter = itertools.count()
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    # -- Introspection (useful for tests) ----------------------------------

    @property
    def running(self) -> bool:
        """True when the background consumer loop is active."""
        return self._task is not None and not self._task.done()

    # -- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the background consumer loop.

        Idempotent: calling :meth:`start` on an already-running batcher is a
        no-op. Must be called from within a running event loop.
        """
        if self.running:
            return
        self._stopped = False
        self._task = asyncio.ensure_future(self._run_loop())

    async def stop(self) -> None:
        """Stop the loop and resolve every still-pending request.

        Cancels the background task, then drains any requests left in the queue
        and flushes them so no submitted future is left unresolved
        (Requirement 6.6). Safe to call multiple times.
        """
        self._stopped = True
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Drain anything still queued so its future is resolved.
        await self._drain()

    # -- Producer API ------------------------------------------------------

    async def submit(self, request: Union[BatchRequest, str]) -> Completion:
        """Enqueue a request and await the completion from its batch flush.

        Accepts either a fully-formed :class:`~agentic.models.BatchRequest` or a
        convenience ``str`` prompt, in which case a ``BatchRequest`` is built
        with a generated ``request_id``, an ``enqueued_at`` stamped from the
        injected clock, and a fresh future from the running loop.

        The background loop is started lazily if it is not already running, so
        ``await batcher.submit(...)`` works without a prior explicit
        :meth:`start` (Requirement 6.1).

        Args:
            request: A ``BatchRequest`` to enqueue, or a non-empty prompt string
                to wrap in one.

        Returns:
            The :class:`~agentic.providers.Completion` mapped to this request
            once its batch is flushed.

        Raises:
            Exception: Whatever the provider call ultimately raised for the
                batch (when no degraded fallback resolved it).
        """
        if isinstance(request, str):
            request = self._build_request(request)
        elif not isinstance(request, BatchRequest):
            raise TypeError("submit expects a BatchRequest or a prompt string")

        # Ensure the consumer is running so the future will be resolved.
        if not self.running:
            self.start()

        await self._queue.put(request)
        return await request.future

    def _build_request(self, prompt: str) -> BatchRequest:
        """Construct a :class:`BatchRequest` for a bare prompt string."""
        loop = asyncio.get_event_loop()
        request_id = f"req-{next(self._counter)}"
        return BatchRequest(
            request_id=request_id,
            prompt=prompt,
            enqueued_at=self._clock(),
            future=loop.create_future(),
        )

    # -- Background consumer (the core "show your work" loop) --------------

    async def _run_loop(self) -> None:
        """Build and flush batches until stopped.

        Mirrors the design's Manual Batching Algorithm:

        1. Block for the first request (no busy-wait while idle).
        2. Fix the window deadline by the first request's ``enqueued_at``.
        3. Fill the batch until the size limit is reached OR the window expires,
           using :func:`asyncio.wait_for` on ``queue.get`` with the *remaining*
           window time so a lone request never starves.
        4. Flush the batch as a single provider call.

        Loop invariants: ``batch`` is non-empty after the first ``get`` and
        ``len(batch) <= max_batch_size`` at all times; the window deadline is
        fixed by the first request and never extended by later arrivals.
        """
        while not self._stopped:
            batch: list[BatchRequest] = []

            # 1. Block for the FIRST request.
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                break
            batch.append(first)

            # 2. Window deadline fixed by the first request (never extended).
            window_deadline = first.enqueued_at + (self._max_wait_ms / 1000.0)

            # 3. Fill until size limit OR time window expires.
            while len(batch) < self._max_batch_size:
                remaining = window_deadline - self._clock()
                if remaining <= 0:
                    break  # time-window trigger
                try:
                    nxt = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break  # time-window trigger
                except asyncio.CancelledError:
                    # Shutting down mid-fill: flush what we have so those
                    # futures resolve, then exit.
                    await self._flush(batch)
                    return

            # 4. Flush the batch (size trigger or window trigger).
            await self._flush(batch)

    async def _drain(self) -> None:
        """Flush any requests still queued (used during shutdown).

        Pulls everything currently in the queue without blocking and flushes it
        in ``max_batch_size`` chunks so every pending future is resolved.
        """
        leftover: list[BatchRequest] = []
        while True:
            try:
                leftover.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for start in range(0, len(leftover), self._max_batch_size):
            await self._flush(leftover[start : start + self._max_batch_size])

    # -- Flush -------------------------------------------------------------

    async def _flush(self, batch: list[BatchRequest]) -> None:
        """Complete a batch's prompts and resolve each request's future.

        The provider call is routed through :meth:`FailureHandler.call` with a
        degraded fallback, so a transient outage yields degraded completions
        rather than failing the whole batch. On success, completion ``i`` is
        delivered to request ``i`` (Requirement 6.5). On an unrecovered
        exception, every pending future in the batch is set to that exception.
        Either way every future is resolved (Requirement 6.6).

        Args:
            batch: The requests to flush. Empty batches are a no-op.
        """
        if not batch:
            return

        prompts = [r.prompt for r in batch]
        try:
            completions = await self._failure.call(
                op=lambda: self._provider.complete(prompts),
                fallback=lambda: self._degraded_completions(prompts),
            )
            # Map each completion back to its originating request by index.
            for req, comp in zip(batch, completions):
                if not req.future.done():
                    req.future.set_result(comp)
            # Defensive: if the provider returned fewer completions than
            # prompts, the contract was violated -- resolve the remainder with
            # an explicit error so no future is left dangling.
            if len(completions) < len(batch):
                shortfall = RuntimeError(
                    "provider returned fewer completions than prompts: "
                    f"{len(completions)} < {len(batch)}"
                )
                for req in batch[len(completions):]:
                    if not req.future.done():
                        req.future.set_exception(shortfall)
        except Exception as exc:  # noqa: BLE001 - propagate to every future
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)

    async def _degraded_completions(self, prompts: list[str]) -> list[Completion]:
        """Produce deterministic degraded completions for a batch's prompts.

        Used as the :class:`FailureHandler` fallback when the provider call
        cannot succeed. The output is order-preserving and derived purely from
        the prompt so the degraded path is reproducible.
        """
        return [
            Completion(text=f"[degraded] {prompt}", prompt=prompt)
            for prompt in prompts
        ]
