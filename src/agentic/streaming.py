"""Streaming layer for the Agentic AI System.

This module delivers partial outputs and lifecycle events to a subscriber as
they are produced, decoupled from execution speed. It defines two public
types:

* :class:`StreamEvent` -- an immutable event carrying a :class:`StreamEventKind`
  ``kind``, an arbitrary ``payload`` (for partial results this is the step id
  plus the :class:`~agentic.models.AgentResult`), and a ``degraded`` flag so a
  client can render degraded results distinctly. Convenience constructors
  (:meth:`StreamEvent.partial`, :meth:`StreamEvent.started`,
  :meth:`StreamEvent.done`, :meth:`StreamEvent.error`) build the common
  lifecycle events.
* :class:`StreamBus` -- a bounded, single-consumer event bus backed by an
  ``asyncio.Queue``. :meth:`StreamBus.emit` is a *synchronous, non-blocking*
  publish, and :meth:`StreamBus.subscribe` is an async generator that yields
  events in FIFO emission order until the bus is closed.

Backpressure
------------
Because :meth:`emit` is synchronous it can never ``await`` for queue space; a
"block the producer" policy is therefore impossible here by construction
(Requirement 5.3). When the bounded queue is full the bus applies a configured
:class:`BackpressurePolicy` that always makes progress without deadlocking
(Requirement 5.5):

* ``DROP_OLDEST`` (default): evict the oldest queued event to free a slot and
  enqueue the incoming event. The freshest events keep flowing -- usually what
  a live progress stream wants.
* ``DROP_NEWEST`` / ``REJECT``: keep the queued events and discard the incoming
  event.

In both cases the bus increments a dropped-event counter and, at the next
opportunity when the queue has room, inserts a single ``DROPPED`` *marker*
event reporting how many events were lost. This lets the subscriber learn that
the stream is lossy without the producer ever blocking.

Requirements: 5.1 (orchestrator emits a partial result per terminal step),
5.2 (events delivered in emission order), 5.3 (publish never blocks the
producer), 5.4 (degraded results are marked on their events), 5.5 (full queue
applies a backpressure policy without deadlocking).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

__all__ = [
    "StreamEventKind",
    "BackpressurePolicy",
    "StreamEvent",
    "StreamBus",
]


class StreamEventKind(str, Enum):
    """The category of a :class:`StreamEvent`.

    Subclassing ``str`` keeps the value JSON-friendly and easy to compare while
    still giving callers a closed enumeration of lifecycle events.
    """

    STARTED = "started"
    PARTIAL = "partial"
    DONE = "done"
    ERROR = "error"
    #: Synthetic marker emitted by the bus when events were dropped due to a
    #: full queue (see :class:`BackpressurePolicy`).
    DROPPED = "dropped"


class BackpressurePolicy(str, Enum):
    """How :class:`StreamBus` reacts when its bounded queue is full.

    A "block the producer" policy is intentionally absent: :meth:`StreamBus.emit`
    is synchronous and must not await, so blocking is not an option (it would
    risk deadlocking the producer that drives execution).
    """

    #: Evict the oldest queued event to make room for the incoming one.
    DROP_OLDEST = "drop_oldest"
    #: Discard the incoming event, keeping the already-queued events.
    DROP_NEWEST = "drop_newest"
    #: Alias of :attr:`DROP_NEWEST`.
    REJECT = "reject"


@dataclass(frozen=True)
class StreamEvent:
    """An immutable event delivered to a stream subscriber.

    Attributes:
        kind: The :class:`StreamEventKind` describing the event category.
        payload: Arbitrary event data. For :attr:`StreamEventKind.PARTIAL`
            events this is a mapping with ``"step_id"`` and ``"result"`` keys.
        degraded: True when the event represents a degraded result, so the
            client can render it distinctly (Requirement 5.4).
    """

    kind: StreamEventKind
    payload: Any = None
    degraded: bool = False

    def __post_init__(self) -> None:
        # Normalize a string kind into the enum so callers may pass either.
        if not isinstance(self.kind, StreamEventKind):
            object.__setattr__(self, "kind", StreamEventKind(self.kind))

    # -- Convenience constructors ------------------------------------------

    @classmethod
    def partial(cls, step_id: str, result: Any) -> "StreamEvent":
        """Build a PARTIAL event for a step's :class:`AgentResult`.

        The ``degraded`` flag is derived from ``result.degraded`` so degraded
        results are marked automatically (Requirement 5.4). The payload keeps
        both the step id and the full result so the subscriber can render the
        output and correlate it with the originating step.
        """
        degraded = bool(getattr(result, "degraded", False))
        return cls(
            kind=StreamEventKind.PARTIAL,
            payload={"step_id": step_id, "result": result},
            degraded=degraded,
        )

    @classmethod
    def started(cls, payload: Any = None) -> "StreamEvent":
        """Build a STARTED lifecycle event (the run has begun)."""
        return cls(kind=StreamEventKind.STARTED, payload=payload)

    @classmethod
    def done(cls, payload: Any = None) -> "StreamEvent":
        """Build a DONE lifecycle event (the run finished)."""
        return cls(kind=StreamEventKind.DONE, payload=payload)

    @classmethod
    def error(
        cls,
        error: Any,
        *,
        step_id: str | None = None,
        payload: Any = None,
    ) -> "StreamEvent":
        """Build an ERROR lifecycle event.

        When ``payload`` is not supplied a mapping is constructed from the
        ``error`` (stringified) and the optional ``step_id``.
        """
        if payload is None:
            payload = {"error": str(error), "step_id": step_id}
        return cls(kind=StreamEventKind.ERROR, payload=payload, degraded=True)

    @classmethod
    def dropped(cls, count: int) -> "StreamEvent":
        """Build a DROPPED marker reporting how many events were lost."""
        return cls(
            kind=StreamEventKind.DROPPED,
            payload={"dropped": int(count)},
            degraded=True,
        )


#: Unique sentinel pushed onto the queue by :meth:`StreamBus.close` to signal
#: the subscriber generator to terminate. Module-level so identity comparison
#: (``is``) is stable across the bus lifetime.
_CLOSE_SENTINEL = object()


@dataclass
class StreamBus:
    """A bounded, single-consumer event bus for streaming partial outputs.

    Events emitted via :meth:`emit` are delivered to a single subscriber (via
    :meth:`subscribe`) in FIFO emission order. The underlying ``asyncio.Queue``
    preserves per-task ordering for free (Requirement 5.2).

    Args:
        maxsize: Maximum number of events buffered before the backpressure
            policy applies. Must be >= 1 (a bounded queue is required so memory
            cannot grow without limit).
        policy: The :class:`BackpressurePolicy` applied when the queue is full.

    Note:
        This bus is designed for a *single* subscriber per task, matching the
        orchestrator's usage where one consumer drains the stream for a run.
        ``asyncio.Queue`` hands each event to exactly one ``get``, so it is not
        a broadcast/fan-out channel.
    """

    maxsize: int = 1024
    policy: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST
    _queue: asyncio.Queue = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _dropped: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.maxsize, int) or self.maxsize < 1:
            raise ValueError("StreamBus.maxsize must be an integer >= 1")
        if not isinstance(self.policy, BackpressurePolicy):
            self.policy = BackpressurePolicy(self.policy)
        # Bounded queue: maxsize >= 1 guarantees backpressure rather than the
        # asyncio "0 means infinite" behavior.
        self._queue = asyncio.Queue(maxsize=self.maxsize)

    # -- Introspection (useful for tests) ----------------------------------

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has been called."""
        return self._closed

    @property
    def dropped_count(self) -> int:
        """Number of dropped events not yet surfaced via a DROPPED marker."""
        return self._dropped

    def qsize(self) -> int:
        """Current number of buffered events (including any pending markers)."""
        return self._queue.qsize()

    # -- Publishing --------------------------------------------------------

    def emit(self, event: StreamEvent) -> bool:
        """Publish ``event`` without blocking or awaiting (Requirement 5.3).

        Returns True if the event was enqueued, or False if it was dropped per
        the configured :class:`BackpressurePolicy`. Emitting after
        :meth:`close` is a no-op that returns False.

        This method only ever uses ``put_nowait``/``get_nowait`` so it can be
        called from synchronous producer code and is guaranteed not to
        deadlock when the queue is full (Requirement 5.5).
        """
        if not isinstance(event, StreamEvent):
            raise TypeError("StreamBus.emit expects a StreamEvent")
        if self._closed:
            return False

        # Opportunistically surface any pending drop marker first so the
        # subscriber learns about loss as early as there is room.
        self._drain_pending_marker()

        if self._try_put(event):
            return True

        # Queue is full -> apply the backpressure policy.
        if self.policy is BackpressurePolicy.DROP_OLDEST:
            try:
                self._queue.get_nowait()  # evict the oldest event
                self._dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - race only
                pass
            if self._try_put(event):
                return True
            # Could not enqueue even after eviction: count the incoming drop.
            self._dropped += 1
            return False

        # DROP_NEWEST / REJECT: discard the incoming event, keep the buffer.
        self._dropped += 1
        return False

    def _try_put(self, item: Any) -> bool:
        """Attempt a non-blocking put; return True on success, False if full."""
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            return False

    def _drain_pending_marker(self) -> None:
        """Enqueue a single DROPPED marker if drops are pending and room exists."""
        if self._dropped <= 0:
            return
        if self._try_put(StreamEvent.dropped(self._dropped)):
            self._dropped = 0

    # -- Subscribing -------------------------------------------------------

    async def subscribe(self) -> AsyncIterator[StreamEvent]:
        """Yield events in FIFO emission order until the bus is closed.

        Terminates cleanly when :meth:`close` (or :meth:`aclose`) is called: a
        close sentinel is enqueued behind any buffered events, so the consumer
        drains everything already emitted before the generator returns.
        """
        while True:
            item = await self._queue.get()
            if item is _CLOSE_SENTINEL:
                return
            yield item

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the bus so :meth:`subscribe` terminates after draining.

        Idempotent and non-blocking. Any pending drop marker is flushed first
        so a subscriber always learns about lost events before the stream
        ends, then the close sentinel is enqueued. If the queue is full,
        buffered events are evicted to make room so the subscriber is always
        able to terminate.
        """
        if self._closed:
            return
        self._closed = True
        # Guarantee the subscriber observes a DROPPED marker even if the queue
        # stayed saturated up to close (no consumer freed space earlier).
        if self._dropped > 0:
            self._force_put(StreamEvent.dropped(self._dropped))
            self._dropped = 0
        self._force_put(_CLOSE_SENTINEL)

    def _force_put(self, item: Any) -> None:
        """Enqueue ``item``, evicting the oldest events if the queue is full.

        Non-blocking. Because eviction removes from the front (oldest first),
        the most recently forced items survive at the tail in insertion order.
        """
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - race only
                    return

    async def aclose(self) -> None:
        """Async alias of :meth:`close` for use in ``async with``-style cleanup."""
        self.close()
