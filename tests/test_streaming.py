"""Tests for the streaming layer (``agentic.streaming``).

Covers:

* Task 6.2 -- Property 11 (Stream ordering): events drain in emission order and
  degraded events keep their ``degraded`` flag. **Validates: Requirements 5.2, 5.4**
* Task 6.3 -- Unit tests for the backpressure policy: a saturated bounded queue
  never blocks or deadlocks the (synchronous) producer, drops are reported, and a
  ``DROPPED`` marker surfaces to the subscriber on drain. **Requirements: 5.3, 5.5**

The property test is a synchronous Hypothesis test that drives the async bus via
``asyncio.run`` per example (this keeps Hypothesis off pytest-asyncio's shared
event loop). The backpressure unit tests are plain async tests.
"""

from __future__ import annotations

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic.models import AgentResult, ResultStatus
from agentic.streaming import (
    BackpressurePolicy,
    StreamBus,
    StreamEvent,
    StreamEventKind,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ok_result(step_id: str, output: str = "ok-output") -> AgentResult:
    """Build a non-degraded (OK) AgentResult for a step."""
    return AgentResult(step_id=step_id, status=ResultStatus.OK, output=output)


def _degraded_result(step_id: str, output: str = "degraded-output") -> AgentResult:
    """Build a degraded AgentResult (status DEGRADED, degraded=True)."""
    return AgentResult(
        step_id=step_id,
        status=ResultStatus.DEGRADED,
        output=output,
        degraded=True,
    )


async def _drain(bus: StreamBus) -> list[StreamEvent]:
    """Collect every event yielded by ``subscribe`` until the bus closes."""
    return [event async for event in bus.subscribe()]


async def _drain_concurrently(bus: StreamBus) -> list[StreamEvent]:
    """Start a subscriber, let it drain buffered events, then close the bus.

    Draining *before* closing means the close sentinel (and any flushed
    ``DROPPED`` marker) does not have to evict still-buffered events to make
    room, so the events that survived the backpressure policy are observed
    intact and in order.
    """
    collected: list[StreamEvent] = []

    async def consume() -> None:
        async for event in bus.subscribe():
            collected.append(event)

    consumer = asyncio.create_task(consume())
    # Yield control so the consumer drains everything currently buffered and
    # then parks on an empty queue.
    await asyncio.sleep(0)
    bus.close()
    await consumer
    return collected


# --------------------------------------------------------------------------- #
# Task 6.2 -- Property 11: Stream ordering
# --------------------------------------------------------------------------- #


@settings(max_examples=200)
@given(degraded_flags=st.lists(st.booleans(), max_size=25))
def test_property_11_stream_ordering(degraded_flags: list[bool]) -> None:
    """Property 11: events are observed in emission order; degraded flags survive.

    Generates a STARTED event, a run of PARTIAL events with varying ``degraded``
    flags, and a DONE event. With a queue large enough to avoid any drops, the
    drained sequence must equal the emitted sequence exactly, and every event
    built from a degraded result must keep ``degraded is True`` (and conversely).

    **Validates: Requirements 5.2, 5.4**
    """

    async def scenario() -> tuple[list[StreamEvent], list[StreamEvent]]:
        # maxsize comfortably exceeds (started + partials + done + sentinel).
        bus = StreamBus(
            maxsize=len(degraded_flags) + 8,
            policy=BackpressurePolicy.DROP_OLDEST,
        )

        emitted: list[StreamEvent] = []

        started = StreamEvent.started(payload={"run": "r1"})
        assert bus.emit(started) is True
        emitted.append(started)

        for i, is_degraded in enumerate(degraded_flags):
            step_id = f"s{i}"
            result = _degraded_result(step_id) if is_degraded else _ok_result(step_id)
            event = StreamEvent.partial(step_id, result)
            assert bus.emit(event) is True
            emitted.append(event)

        done = StreamEvent.done(payload={"run": "r1"})
        assert bus.emit(done) is True
        emitted.append(done)

        # No events should have been dropped given the generous capacity.
        assert bus.dropped_count == 0

        bus.close()
        drained = await _drain(bus)
        return emitted, drained

    emitted, drained = asyncio.run(scenario())

    # Order preservation: the drained sequence equals the emission sequence.
    assert drained == emitted
    assert [e.kind for e in drained] == [e.kind for e in emitted]

    # Degraded flags are preserved per event (Requirement 5.4): the PARTIAL
    # events alternate per the generated flags, framed by STARTED then DONE.
    partials = [e for e in drained if e.kind is StreamEventKind.PARTIAL]
    assert [e.degraded for e in partials] == degraded_flags
    for event, is_degraded in zip(partials, degraded_flags):
        assert event.degraded is is_degraded
        assert event.payload["result"].degraded is is_degraded

    # No synthetic DROPPED markers should appear on the lossless path.
    assert all(e.kind is not StreamEventKind.DROPPED for e in drained)


# --------------------------------------------------------------------------- #
# Task 6.3 -- Backpressure policy unit tests (Requirements 5.3, 5.5)
# --------------------------------------------------------------------------- #


async def test_drop_oldest_producer_never_blocks_and_reports_drops() -> None:
    """DROP_OLDEST: emitting past capacity never blocks/raises and reports drops.

    With ``maxsize=2`` we emit five events. ``emit`` is synchronous and must
    return a ``bool`` for every call without raising (the producer is never
    blocked or deadlocked -- Requirement 5.3). The freshest events survive in
    order, the bus accounts for the lost events, and a single ``DROPPED`` marker
    surfaces to the subscriber on drain (Requirement 5.5).
    """
    bus = StreamBus(maxsize=2, policy=BackpressurePolicy.DROP_OLDEST)

    events = [StreamEvent.partial(f"s{i}", _ok_result(f"s{i}")) for i in range(5)]
    results = [bus.emit(e) for e in events]

    # Every emit returned a bool without raising -> producer not blocked.
    assert all(isinstance(r, bool) for r in results)

    # Three of the five events were evicted to keep the queue bounded at 2.
    assert bus.dropped_count == 3

    collected = await _drain_concurrently(bus)

    # DROP_OLDEST keeps the freshest events: the last two emitted, in order.
    assert collected[0] is events[3]
    assert collected[1] is events[4]

    # A DROPPED marker surfaces reporting exactly how many events were lost.
    markers = [e for e in collected if e.kind is StreamEventKind.DROPPED]
    assert len(markers) == 1
    assert markers[0].payload["dropped"] == 3
    assert markers[0].degraded is True


async def test_drop_newest_keeps_oldest_discards_newest() -> None:
    """DROP_NEWEST: keep the already-queued (oldest) events, discard incoming.

    With ``maxsize=2`` the first two events are buffered; subsequent emits while
    the queue is full are rejected (return ``False``) and counted as drops. The
    producer never blocks, the oldest events survive in order, and a ``DROPPED``
    marker reports the discarded count (Requirements 5.3, 5.5).
    """
    bus = StreamBus(maxsize=2, policy=BackpressurePolicy.DROP_NEWEST)

    e0 = StreamEvent.partial("s0", _ok_result("s0"))
    e1 = StreamEvent.partial("s1", _ok_result("s1"))
    e2 = StreamEvent.partial("s2", _ok_result("s2"))
    e3 = StreamEvent.partial("s3", _ok_result("s3"))

    assert bus.emit(e0) is True  # buffered
    assert bus.emit(e1) is True  # buffered (queue now full)
    assert bus.emit(e2) is False  # newest discarded
    assert bus.emit(e3) is False  # newest discarded

    assert bus.dropped_count == 2

    collected = await _drain_concurrently(bus)

    # Oldest two kept in emission order; the newer events are gone.
    assert collected[0] is e0
    assert collected[1] is e1
    assert e2 not in collected
    assert e3 not in collected

    markers = [e for e in collected if e.kind is StreamEventKind.DROPPED]
    assert len(markers) == 1
    assert markers[0].payload["dropped"] == 2


async def test_reject_alias_behaves_like_drop_newest() -> None:
    """REJECT is an alias of DROP_NEWEST: oldest kept, incoming discarded."""
    bus = StreamBus(maxsize=2, policy=BackpressurePolicy.REJECT)

    e0 = StreamEvent.partial("s0", _ok_result("s0"))
    e1 = StreamEvent.partial("s1", _ok_result("s1"))
    e2 = StreamEvent.partial("s2", _ok_result("s2"))

    assert bus.emit(e0) is True
    assert bus.emit(e1) is True
    assert bus.emit(e2) is False  # rejected (kept the oldest)

    assert bus.dropped_count == 1

    collected = await _drain_concurrently(bus)

    assert collected[0] is e0
    assert collected[1] is e1
    assert e2 not in collected

    markers = [e for e in collected if e.kind is StreamEventKind.DROPPED]
    assert len(markers) == 1
    assert markers[0].payload["dropped"] == 1


async def test_emit_after_close_is_noop() -> None:
    """Emitting after ``close`` is a non-blocking no-op returning False."""
    bus = StreamBus(maxsize=4, policy=BackpressurePolicy.DROP_OLDEST)
    bus.close()
    assert bus.emit(StreamEvent.started()) is False


async def test_producer_does_not_deadlock_under_saturation() -> None:
    """A burst far exceeding capacity completes promptly (no deadlock).

    Emitting hundreds of events into a tiny queue must return control to the
    producer quickly; we bound the whole burst with a timeout so a regression
    that blocks the producer fails loudly instead of hanging.
    """
    bus = StreamBus(maxsize=2, policy=BackpressurePolicy.DROP_OLDEST)

    async def burst() -> int:
        count = 0
        for i in range(500):
            bus.emit(StreamEvent.partial(f"s{i}", _ok_result(f"s{i}")))
            count += 1
        return count

    emitted = await asyncio.wait_for(burst(), timeout=5.0)
    assert emitted == 500
    assert bus.dropped_count == 498  # only the last 2 fit the bounded queue
