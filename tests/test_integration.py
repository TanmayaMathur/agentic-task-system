"""End-to-end integration tests for the Agentic AI System.

These tests wire the *entire* system together via
:func:`agentic.cli.build_orchestrator` (MockProvider -> FailureHandler ->
Batcher -> agents routed by the Dispatcher, decomposed by the Planner and
driven by the Orchestrator, with a StreamBus consumed concurrently) and run it
to completion with the deterministic, key-free ``MockProvider`` (Requirement
7.5). They cover the two scenarios from the design's *Integration Testing
Approach*:

* **Happy path** (Task 12.1): every step COMPLETED, the final Task terminal,
  and the stream produces the expected ordered events -- a ``STARTED``, then a
  ``PARTIAL`` per step in topological order (s1 -> s2 -> s3), then ``DONE``.
* **Failure path** (Task 12.2): the MockProvider is configured to fail on the
  analyzer's prompts (``fail_on={"analyze:*"}``). A *transient* failure
  exhausts retries and falls back to a degraded completion, so the analyzer
  step ends DEGRADED and its ``PARTIAL`` event is marked ``degraded=True``
  (Requirement 8.6). The writer depends on the analyzer, so its blocked branch
  is FAILED, yet the run still terminates (Requirements 5.4, 4.3). A permanent
  variant (analyzer FAILED immediately, no degradation) is also asserted.

Requirements: 4.3, 4.4, 5.1, 5.2, 5.4, 7.5, 8.6.
"""

from __future__ import annotations

import asyncio

import pytest

from agentic.cli import build_orchestrator
from agentic.models import ResultStatus, StepStatus, Task
from agentic.streaming import StreamEvent, StreamEventKind

#: A multi-part request that exercises all three agents (retriever/analyzer/writer).
TASK_TEXT = "Research async batching, analyze the tradeoffs, and write a summary."


async def _run_and_collect(
    *,
    fail: bool = False,
    failure_type: str = "transient",
    task_text: str = TASK_TEXT,
) -> tuple[Task, list[StreamEvent]]:
    """Wire the full system, run it end to end, and collect every stream event.

    Subscribes to the stream concurrently with execution, runs the orchestrator
    to termination, then tears everything down cleanly: closes the stream so the
    consumer drains and returns, awaits the consumer, and stops the batcher so
    every pending future is resolved.
    """
    orchestrator, stream, batcher = build_orchestrator(
        fail=fail, failure_type=failure_type
    )

    events: list[StreamEvent] = []

    async def _consume() -> None:
        async for event in stream.subscribe():
            events.append(event)

    consumer = asyncio.create_task(_consume())
    try:
        task = await orchestrator.run(task_text)
    finally:
        # Clean teardown regardless of how the run ended.
        stream.close()
        await consumer
        await batcher.stop()

    return task, events


def _step_by_agent_type(task: Task, agent_type: str):
    """Return the single step with the given ``agent_type``."""
    matches = [s for s in task.steps if s.agent_type == agent_type]
    assert len(matches) == 1, f"expected exactly one {agent_type} step"
    return matches[0]


def _partial_events(events: list[StreamEvent]) -> list[StreamEvent]:
    """Return only the PARTIAL events, preserving emission order."""
    return [e for e in events if e.kind is StreamEventKind.PARTIAL]


def _partial_by_step_id(events: list[StreamEvent], step_id: str) -> StreamEvent:
    """Return the single PARTIAL event whose payload references ``step_id``."""
    matches = [
        e
        for e in _partial_events(events)
        if (e.payload or {}).get("step_id") == step_id
    ]
    assert len(matches) == 1, f"expected exactly one PARTIAL event for {step_id}"
    return matches[0]


# --------------------------------------------------------------------------
# Task 12.1: happy-path integration test
# --------------------------------------------------------------------------


async def test_happy_path_completes_and_streams_ordered_events() -> None:
    """Full happy-path run: all steps COMPLETED and events stream in order.

    Validates: Requirements 4.3, 4.4, 5.1, 5.2, 7.5
    """
    task, events = await _run_and_collect()

    # The run terminates with every step terminal (Requirements 4.3, 4.4).
    assert task.is_terminal() is True

    # Every step COMPLETED on the happy path.
    assert [s.status for s in task.steps] == [StepStatus.COMPLETED] * len(task.steps)
    for step in task.steps:
        assert step.status is StepStatus.COMPLETED
        assert step.result is not None
        assert step.result.status is ResultStatus.OK

    # The stream produced the expected ordered events (Requirement 5.2):
    # STARTED, then a PARTIAL per step in topological order, then DONE.
    kinds = [e.kind for e in events]
    assert kinds[0] is StreamEventKind.STARTED
    assert kinds[-1] is StreamEventKind.DONE

    partial_step_ids = [
        (e.payload or {}).get("step_id") for e in _partial_events(events)
    ]
    # Planner emits a linear DAG s1 (retriever) -> s2 (analyzer) -> s3 (writer);
    # the orchestrator emits a PARTIAL per step as it completes (Requirement 5.1).
    assert partial_step_ids == ["s1", "s2", "s3"]

    # No degraded markers on the happy path.
    assert all(e.degraded is False for e in _partial_events(events))

    # Exactly one STARTED and one DONE, both bracketing the partials.
    assert kinds.count(StreamEventKind.STARTED) == 1
    assert kinds.count(StreamEventKind.DONE) == 1
    started_idx = kinds.index(StreamEventKind.STARTED)
    done_idx = kinds.index(StreamEventKind.DONE)
    partial_indices = [
        i for i, k in enumerate(kinds) if k is StreamEventKind.PARTIAL
    ]
    assert started_idx < min(partial_indices)
    assert max(partial_indices) < done_idx


# --------------------------------------------------------------------------
# Task 12.2: failure-path integration test
# --------------------------------------------------------------------------


async def test_failure_path_transient_degrades_analyzer_and_terminates() -> None:
    """Transient analyzer failure degrades and the run still terminates.

    The MockProvider fails on ``analyze:*`` (transient); the batcher exhausts
    retries and falls back to a degraded completion, so the analyzer step ends
    DEGRADED and its PARTIAL event is marked degraded (Requirement 8.6). The
    writer depends on the analyzer and is FAILED via the blocked branch, yet
    the run still terminates (Requirements 5.4, 4.3).

    Validates: Requirements 8.6, 5.4, 4.3
    """
    task, events = await _run_and_collect(fail=True, failure_type="transient")

    # The run still terminates despite the failure (Requirement 4.3).
    assert task.is_terminal() is True

    retriever = _step_by_agent_type(task, "retriever")
    analyzer = _step_by_agent_type(task, "analyzer")
    writer = _step_by_agent_type(task, "writer")

    # Retriever completed normally (its prompts are not configured to fail).
    assert retriever.status is StepStatus.COMPLETED

    # Analyzer degraded via the batcher's fallback (Requirement 8.6).
    assert analyzer.status is StepStatus.DEGRADED
    assert analyzer.result is not None
    assert analyzer.result.status is ResultStatus.DEGRADED
    assert analyzer.result.degraded is True

    # A DEGRADED result streamed to the consumer, marked degraded (Requirement 5.4).
    analyzer_event = _partial_by_step_id(events, analyzer.id)
    assert analyzer_event.degraded is True
    assert analyzer_event.payload["result"].status is ResultStatus.DEGRADED

    # The writer depends on the analyzer; since the analyzer is not COMPLETED,
    # the writer can never become ready and is FAILED via the blocked branch.
    assert writer.status is StepStatus.FAILED
    assert writer.result is not None
    assert writer.result.status is ResultStatus.FAILED
    assert writer.result.error is not None

    # The blocked writer's failure also streamed to the consumer.
    writer_event = _partial_by_step_id(events, writer.id)
    assert writer_event.payload["result"].status is ResultStatus.FAILED

    # Lifecycle events still bracket the run.
    kinds = [e.kind for e in events]
    assert kinds[0] is StreamEventKind.STARTED
    assert kinds[-1] is StreamEventKind.DONE


async def test_failure_path_permanent_fails_analyzer_immediately() -> None:
    """Permanent analyzer failure: analyzer FAILED (no degradation), run terminates.

    A permanent error is never retried and no fallback is used, so the analyzer
    step ends FAILED rather than DEGRADED. The writer is still blocked and the
    run still terminates.

    Validates: Requirements 8.6, 5.4, 4.3
    """
    task, events = await _run_and_collect(fail=True, failure_type="permanent")

    assert task.is_terminal() is True

    retriever = _step_by_agent_type(task, "retriever")
    analyzer = _step_by_agent_type(task, "analyzer")
    writer = _step_by_agent_type(task, "writer")

    assert retriever.status is StepStatus.COMPLETED

    # Permanent failure propagates without a degraded fallback -> FAILED.
    assert analyzer.status is StepStatus.FAILED
    assert analyzer.result is not None
    assert analyzer.result.status is ResultStatus.FAILED
    assert analyzer.result.error is not None

    # The writer is blocked and FAILED; the run still terminates.
    assert writer.status is StepStatus.FAILED
    assert writer.result is not None
    assert writer.result.status is ResultStatus.FAILED

    kinds = [e.kind for e in events]
    assert kinds[0] is StreamEventKind.STARTED
    assert kinds[-1] is StreamEventKind.DONE
