"""Unit tests for the agents and the dispatcher (task 7.4).

Covers the *Agent* execution contract and *Dispatcher* routing from the
design's *Components and Interfaces* section:

* Each specialized agent (:class:`~agentic.agents.RetrieverAgent`,
  :class:`~agentic.agents.AnalyzerAgent`, :class:`~agentic.agents.WriterAgent`)
  turns a :class:`~agentic.models.Step` into a well-formed
  :class:`~agentic.models.AgentResult` with ``status == OK`` and a non-empty
  ``output`` on the happy path (Requirement 3.4).
* Analyzer/Writer incorporate their upstream dependency outputs (read from an
  :class:`~agentic.agents.ExecutionContext`) into the prompt they submit
  (Requirement 3.3).
* A provider failure never crosses the agent boundary: a permanent error
  becomes ``status == FAILED`` with a non-null ``error``, and an exhausted
  transient error degrades (via the batcher's ``_flush`` fallback) to
  ``status == DEGRADED`` with ``degraded == True`` (Requirement 3.5).
* The :class:`~agentic.dispatcher.Dispatcher` routes a step to the agent whose
  ``agent_type`` matches, returns a ``FAILED`` result (not an exception) for an
  unknown ``agent_type``, rejects duplicate registrations, and exposes the
  registered ``agent_types`` (Requirement 3.2).

The whole suite runs key-free against the deterministic ``MockProvider``. The
event loop runs these as plain ``async def`` tests under ``asyncio_mode = auto``.

Requirements: 3.2, 3.4, 3.5.
"""

from __future__ import annotations

import pytest

from agentic.agents import (
    AnalyzerAgent,
    ExecutionContext,
    RetrieverAgent,
    WriterAgent,
)
from agentic.batcher import Batcher
from agentic.dispatcher import Dispatcher
from agentic.errors import PermanentError, TransientError
from agentic.failure import FailureHandler
from agentic.models import AgentResult, ResultStatus, Step
from agentic.providers import Completion, MockProvider

# Canned responses keyed by the prompt-prefix globs the agents emit.
_RESPONSES = {
    "retrieve:*": "sources...",
    "analyze:*": "findings...",
    "write:*": "final report...",
}


class _RecordingMockProvider(MockProvider):
    """A :class:`MockProvider` that also records every prompt it completes.

    Used to assert that Analyzer/Writer fold their dependency outputs into the
    prompt they submit (the canned glob responses are independent of the
    dependency, so we inspect the prompt directly).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prompts_seen: list[str] = []

    async def complete(self, prompts: list[str]) -> list[Completion]:
        self.prompts_seen.extend(prompts)
        return await super().complete(prompts)


def _fast_failure() -> FailureHandler:
    """A FailureHandler with zero backoff so the degraded path stays quick.

    Mirrors the batcher's default retry/breaker configuration but removes the
    backoff sleeps; the batcher still supplies the degraded fallback in
    ``_flush``, so an exhausted transient error degrades rather than fails.
    """
    return FailureHandler(max_retries=2, base_delay_ms=0, breaker_threshold=5)


def _make_batcher(provider: MockProvider) -> Batcher:
    """Build a Batcher over ``provider`` with a small window and fast retries."""
    return Batcher(
        provider,
        max_batch_size=8,
        max_wait_ms=10,
        failure=_fast_failure(),
    )


# --------------------------------------------------------------------------
# Happy path: every agent returns a well-formed OK AgentResult.
# Requirement 3.4
# --------------------------------------------------------------------------
async def test_retriever_returns_ok_result():
    """RetrieverAgent yields an OK result with the canned retrieve response."""
    provider = MockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        step = Step(id="r1", description="gather sources", agent_type="retriever")
        result = await RetrieverAgent(batcher).execute(step, ExecutionContext())
    finally:
        await batcher.stop()

    assert isinstance(result, AgentResult)
    assert result.status is ResultStatus.OK
    assert result.step_id == "r1"
    assert result.output == "sources..."
    assert result.output  # non-empty
    assert result.error is None


async def test_analyzer_returns_ok_result():
    """AnalyzerAgent yields an OK result with the canned analyze response."""
    provider = MockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        step = Step(
            id="a1",
            description="analyze the material",
            agent_type="analyzer",
            depends_on=["r1"],
        )
        context = ExecutionContext({"r1": "RETRIEVED_MATERIAL_XYZ"})
        result = await AnalyzerAgent(batcher).execute(step, context)
    finally:
        await batcher.stop()

    assert result.status is ResultStatus.OK
    assert result.step_id == "a1"
    assert result.output == "findings..."
    assert result.error is None


async def test_writer_returns_ok_result():
    """WriterAgent yields an OK result with the canned write response."""
    provider = MockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        step = Step(
            id="w1",
            description="write the summary",
            agent_type="writer",
            depends_on=["a1"],
        )
        context = ExecutionContext({"a1": "ANALYSIS_FINDINGS_ABC"})
        result = await WriterAgent(batcher).execute(step, context)
    finally:
        await batcher.stop()

    assert result.status is ResultStatus.OK
    assert result.step_id == "w1"
    assert result.output == "final report..."
    assert result.error is None


# --------------------------------------------------------------------------
# Dependency outputs are incorporated into the submitted prompt.
# Requirement 3.3
# --------------------------------------------------------------------------
async def test_analyzer_incorporates_dependency_output():
    """The analyzer's prompt reflects its retriever dependency's output."""
    provider = _RecordingMockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        step = Step(
            id="a1",
            description="analyze the material",
            agent_type="analyzer",
            depends_on=["r1"],
        )
        context = ExecutionContext({"r1": "RETRIEVED_MATERIAL_XYZ"})
        result = await AnalyzerAgent(batcher).execute(step, context)
    finally:
        await batcher.stop()

    assert result.status is ResultStatus.OK
    # The dependency output was folded into the prompt the analyzer submitted.
    assert provider.prompts_seen, "expected the analyzer to submit a prompt"
    submitted = provider.prompts_seen[0]
    assert submitted.startswith("analyze:")
    assert "RETRIEVED_MATERIAL_XYZ" in submitted
    assert "[r1]" in submitted


async def test_writer_incorporates_dependency_output():
    """The writer's prompt reflects its analyzer dependency's output."""
    provider = _RecordingMockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        step = Step(
            id="w1",
            description="write the summary",
            agent_type="writer",
            depends_on=["a1"],
        )
        context = ExecutionContext({"a1": "ANALYSIS_FINDINGS_ABC"})
        result = await WriterAgent(batcher).execute(step, context)
    finally:
        await batcher.stop()

    assert result.status is ResultStatus.OK
    assert provider.prompts_seen, "expected the writer to submit a prompt"
    submitted = provider.prompts_seen[0]
    assert submitted.startswith("write:")
    assert "ANALYSIS_FINDINGS_ABC" in submitted
    assert "[a1]" in submitted


# --------------------------------------------------------------------------
# Failure never crosses the agent boundary.
# Requirement 3.5
# --------------------------------------------------------------------------
async def test_permanent_error_becomes_failed_result():
    """A permanent provider error maps to FAILED with a non-null error.

    A ``PermanentError`` is never retried and is not routed to the batcher's
    degraded fallback; it propagates through ``_flush`` (set on the future) and
    surfaces from ``submit``. The agent catches it and converts it to a
    well-formed FAILED result rather than letting the exception escape.
    """
    provider = MockProvider(
        responses=_RESPONSES,
        fail_on={"retrieve:*": PermanentError},
    )
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        step = Step(id="r1", description="gather sources", agent_type="retriever")
        result = await RetrieverAgent(batcher).execute(step, ExecutionContext())
    finally:
        await batcher.stop()

    assert result.status is ResultStatus.FAILED
    assert result.step_id == "r1"
    assert result.error is not None
    assert result.output == ""  # empty output is allowed only when FAILED
    assert "PermanentError" in result.error


async def test_transient_error_becomes_degraded_result():
    """An exhausted transient error degrades via the batcher fallback.

    The batcher's ``_flush`` supplies a degraded fallback, so when retries are
    exhausted the provider call yields ``"[degraded] ..."`` completions; the
    agent maps that to a DEGRADED result with ``degraded == True``.
    """
    provider = MockProvider(
        responses=_RESPONSES,
        fail_on={"retrieve:*": TransientError},
    )
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        step = Step(id="r1", description="gather sources", agent_type="retriever")
        result = await RetrieverAgent(batcher).execute(step, ExecutionContext())
    finally:
        await batcher.stop()

    assert result.status is ResultStatus.DEGRADED
    assert result.degraded is True
    assert result.step_id == "r1"
    assert result.output.startswith("[degraded]")


# --------------------------------------------------------------------------
# Dispatcher routing.
# Requirement 3.2
# --------------------------------------------------------------------------
async def test_dispatch_routes_to_matching_agent():
    """dispatch() routes a step to the agent matching its agent_type."""
    provider = MockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        dispatcher = Dispatcher(
            [
                RetrieverAgent(batcher),
                AnalyzerAgent(batcher),
                WriterAgent(batcher),
            ]
        )

        retrieve_step = Step(id="r1", description="gather", agent_type="retriever")
        write_step = Step(
            id="w1", description="write", agent_type="writer", depends_on=["a1"]
        )

        retrieve_result = await dispatcher.dispatch(retrieve_step, ExecutionContext())
        write_result = await dispatcher.dispatch(
            write_step, ExecutionContext({"a1": "ANALYSIS"})
        )
    finally:
        await batcher.stop()

    # Routed to the retriever: got the retrieve canned response.
    assert retrieve_result.status is ResultStatus.OK
    assert retrieve_result.output == "sources..."
    # Routed to the writer: got the write canned response.
    assert write_result.status is ResultStatus.OK
    assert write_result.output == "final report..."


async def test_dispatch_unknown_agent_type_returns_failed_result():
    """An unknown agent_type yields a FAILED result, not an exception."""
    provider = MockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    batcher.start()
    try:
        dispatcher = Dispatcher([RetrieverAgent(batcher)])
        step = Step(id="s1", description="do something", agent_type="searcher")
        result = await dispatcher.dispatch(step, ExecutionContext())
    finally:
        await batcher.stop()

    assert isinstance(result, AgentResult)
    assert result.status is ResultStatus.FAILED
    assert result.step_id == "s1"
    assert result.error is not None
    assert "searcher" in result.error


def test_duplicate_agent_type_raises_value_error():
    """Two agents with the same agent_type are a configuration error."""
    provider = MockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    with pytest.raises(ValueError):
        Dispatcher([RetrieverAgent(batcher), RetrieverAgent(batcher)])


def test_agent_types_property_returns_registered_set():
    """agent_types exposes exactly the registered agent types."""
    provider = MockProvider(responses=_RESPONSES)
    batcher = _make_batcher(provider)
    dispatcher = Dispatcher(
        [RetrieverAgent(batcher), AnalyzerAgent(batcher), WriterAgent(batcher)]
    )
    assert dispatcher.agent_types == {"retriever", "analyzer", "writer"}
    assert dispatcher.has_agent("analyzer")
    assert not dispatcher.has_agent("searcher")
