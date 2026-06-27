"""Tests for the Orchestrator's DAG execution loop.

Covers tasks 9.2, 9.3, and 9.4 from the implementation plan:

* **9.2 -- Property 5: DAG termination** -- over randomly generated valid DAGs,
  ``_execute_dag`` terminates and the Task ends terminal
  (Validates: Requirements 4.3, 4.4, 4.5).
* **9.3 -- Property 6: Topological respect** -- a step is only dispatched after
  every one of its dependencies is COMPLETED
  (Validates: Requirements 2.2, 4.2).
* **9.4 -- unit tests** -- empty/whitespace input is rejected before a run
  begins (no events emitted), and a failed dependency marks downstream steps
  FAILED while the run still terminates
  (Validates: Requirements 1.1, 1.2, 4.5).

The orchestrator depends only on ``dispatcher.dispatch(step, context)`` to run a
step, so these tests drive ``_execute_dag`` directly over hand-built / generated
Tasks using lightweight fake dispatchers, avoiding the full agent/batcher stack.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentic.models import (
    AgentResult,
    ResultStatus,
    Step,
    StepStatus,
    Task,
)
from agentic.orchestrator import Orchestrator
from agentic.planner import Planner
from agentic.providers import MockProvider
from agentic.streaming import StreamBus

_AGENT_TYPES = ["retriever", "analyzer", "writer"]


# ---------------------------------------------------------------------------
# Fake dispatchers
# ---------------------------------------------------------------------------


class OkDispatcher:
    """Dispatcher stub whose every step succeeds.

    Returns an ``OK`` :class:`AgentResult` for any step regardless of its
    ``agent_type``, so it can drive arbitrary generated DAGs to completion.
    """

    async def dispatch(self, step: Step, context) -> AgentResult:  # noqa: ANN001
        return AgentResult(
            step_id=step.id,
            status=ResultStatus.OK,
            output=f"out:{step.id}",
        )


class TopologyCheckingDispatcher:
    """Dispatcher stub that asserts dependency completion at dispatch time.

    Holds a reference to the Task so it can inspect step statuses when a step is
    dispatched. For every dependency of the step being dispatched it verifies
    the dependency is already COMPLETED; any violation is recorded for the test
    to assert against.
    """

    def __init__(self, task: Task) -> None:
        self._steps_by_id = {s.id: s for s in task.steps}
        self.dispatch_order: list[str] = []
        self.violations: list[tuple[str, str, StepStatus]] = []

    async def dispatch(self, step: Step, context) -> AgentResult:  # noqa: ANN001
        for dep_id in step.depends_on:
            dep = self._steps_by_id[dep_id]
            if dep.status is not StepStatus.COMPLETED:
                self.violations.append((step.id, dep_id, dep.status))
        self.dispatch_order.append(step.id)
        return AgentResult(
            step_id=step.id,
            status=ResultStatus.OK,
            output=f"out:{step.id}",
        )


class FailingDispatcher:
    """Dispatcher stub that FAILS a configured set of step ids, OK otherwise."""

    def __init__(self, fail_ids: set[str]) -> None:
        self._fail_ids = set(fail_ids)
        self.dispatched: list[str] = []

    async def dispatch(self, step: Step, context) -> AgentResult:  # noqa: ANN001
        self.dispatched.append(step.id)
        if step.id in self._fail_ids:
            return AgentResult(
                step_id=step.id,
                status=ResultStatus.FAILED,
                output="",
                error=f"forced failure for {step.id}",
            )
        return AgentResult(
            step_id=step.id,
            status=ResultStatus.OK,
            output=f"out:{step.id}",
        )


# ---------------------------------------------------------------------------
# Hypothesis strategy: random valid DAGs
# ---------------------------------------------------------------------------


@st.composite
def dag_tasks(draw, max_steps: int = 6) -> Task:
    """Generate a random *valid* :class:`Task` whose steps form a DAG.

    Acyclicity is guaranteed structurally: step ``i`` may only depend on
    earlier-indexed steps (ids ``s0..s{i-1}``), so every edge points backwards
    and no cycle can form. Step ``s0`` has no possible earlier step, so there is
    always at least one dependency-free entry point. Each step's ``agent_type``
    is drawn from the known registry.
    """
    n = draw(st.integers(min_value=1, max_value=max_steps))
    steps: list[Step] = []
    for i in range(n):
        if i == 0:
            depends_on: list[str] = []
        else:
            earlier = [f"s{j}" for j in range(i)]
            depends_on = draw(
                st.lists(st.sampled_from(earlier), unique=True, max_size=len(earlier))
            )
        agent_type = draw(st.sampled_from(_AGENT_TYPES))
        steps.append(
            Step(
                id=f"s{i}",
                description=f"step {i}",
                agent_type=agent_type,
                depends_on=depends_on,
            )
        )
    return Task(id="task-generated", text="generated dag", steps=steps)


# ---------------------------------------------------------------------------
# Task 9.2 -- Property 5: DAG termination
# ---------------------------------------------------------------------------


@settings(deadline=None)
@given(task=dag_tasks())
def test_property_dag_execution_terminates(task: Task) -> None:
    """Property 5: ``_execute_dag`` terminates and the Task ends terminal.

    For any randomly generated valid DAG, running the orchestrator's execution
    loop (with a dispatcher whose steps all succeed) completes without hanging
    and leaves every step in a terminal status.

    **Validates: Requirements 4.3, 4.4, 4.5**
    """
    orchestrator = Orchestrator(planner=None, dispatcher=OkDispatcher(), stream=None)

    asyncio.run(orchestrator._execute_dag(task))

    assert task.is_terminal()
    # With an all-OK dispatcher every step should have COMPLETED.
    assert all(step.status is StepStatus.COMPLETED for step in task.steps)


def test_dag_execution_terminates_linear_chain() -> None:
    """Concrete termination example: a 3-step linear chain completes."""
    steps = [
        Step(id="s0", description="retrieve", agent_type="retriever", depends_on=[]),
        Step(id="s1", description="analyze", agent_type="analyzer", depends_on=["s0"]),
        Step(id="s2", description="write", agent_type="writer", depends_on=["s1"]),
    ]
    task = Task(id="task-linear", text="linear chain", steps=steps)
    orchestrator = Orchestrator(planner=None, dispatcher=OkDispatcher(), stream=None)

    asyncio.run(orchestrator._execute_dag(task))

    assert task.is_terminal()
    assert [s.status for s in task.steps] == [StepStatus.COMPLETED] * 3


# ---------------------------------------------------------------------------
# Task 9.3 -- Property 6: Topological respect
# ---------------------------------------------------------------------------


@settings(deadline=None)
@given(task=dag_tasks())
def test_property_topological_respect(task: Task) -> None:
    """Property 6: a step only runs after all its dependencies are COMPLETED.

    A dispatcher records, at the moment each step is dispatched, whether every
    declared dependency is already COMPLETED. After driving the DAG to
    completion there must be zero violations, and every step must have been
    dispatched exactly once.

    **Validates: Requirements 2.2, 4.2**
    """
    dispatcher = TopologyCheckingDispatcher(task)
    orchestrator = Orchestrator(planner=None, dispatcher=dispatcher, stream=None)

    asyncio.run(orchestrator._execute_dag(task))

    assert dispatcher.violations == []
    # Every step ran exactly once and the run terminated.
    assert sorted(dispatcher.dispatch_order) == sorted(s.id for s in task.steps)
    assert task.is_terminal()


# ---------------------------------------------------------------------------
# Task 9.4 -- unit tests: input rejection and blocked-dependency branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_text", ["", "   ", "\t\n ", "\n"])
async def test_run_rejects_empty_input_without_starting_run(bad_text: str) -> None:
    """Empty/whitespace text raises ValueError and emits no stream events.

    The orchestrator must reject the input *before* a run begins, so no
    lifecycle event (not even STARTED) reaches the stream.

    **Validates: Requirements 1.1, 1.2**
    """
    planner = Planner(MockProvider())
    stream = StreamBus()
    orchestrator = Orchestrator(
        planner=planner, dispatcher=OkDispatcher(), stream=stream
    )

    with pytest.raises(ValueError):
        await orchestrator.run(bad_text)

    # No events were emitted and the stream was never closed by the run.
    assert stream.qsize() == 0
    assert stream.closed is False


async def test_blocked_dependency_marks_downstream_failed_and_terminates() -> None:
    """A failed dependency blocks dependents, which are marked FAILED.

    The run still terminates: the orchestrator's blocked-dependency branch marks
    every step that can never become ready as FAILED.

    **Validates: Requirements 4.5**
    """
    steps = [
        Step(id="s0", description="retrieve", agent_type="retriever", depends_on=[]),
        Step(id="s1", description="analyze", agent_type="analyzer", depends_on=["s0"]),
        Step(id="s2", description="write", agent_type="writer", depends_on=["s1"]),
    ]
    task = Task(id="task-blocked", text="blocked chain", steps=steps)
    dispatcher = FailingDispatcher(fail_ids={"s0"})
    orchestrator = Orchestrator(planner=None, dispatcher=dispatcher, stream=None)

    await orchestrator._execute_dag(task)

    statuses = {s.id: s.status for s in task.steps}
    assert statuses["s0"] is StepStatus.FAILED
    assert statuses["s1"] is StepStatus.FAILED
    assert statuses["s2"] is StepStatus.FAILED
    assert task.is_terminal()
    # Blocked steps never reached the dispatcher; only the entry step did.
    assert dispatcher.dispatched == ["s0"]
    # Blocked steps carry an explanatory error on their results.
    assert task.steps[1].result is not None
    assert task.steps[1].result.error is not None


async def test_independent_branch_still_runs_when_sibling_fails() -> None:
    """A failed branch does not block an independent branch; run terminates.

    ``s1`` depends on the failing ``s0`` (so it is blocked -> FAILED), while
    ``s2`` is independent and should COMPLETE normally.

    **Validates: Requirements 4.5**
    """
    steps = [
        Step(id="s0", description="retrieve", agent_type="retriever", depends_on=[]),
        Step(id="s1", description="analyze", agent_type="analyzer", depends_on=["s0"]),
        Step(id="s2", description="independent", agent_type="writer", depends_on=[]),
    ]
    task = Task(id="task-mixed", text="mixed branches", steps=steps)
    dispatcher = FailingDispatcher(fail_ids={"s0"})
    orchestrator = Orchestrator(planner=None, dispatcher=dispatcher, stream=None)

    await orchestrator._execute_dag(task)

    statuses = {s.id: s.status for s in task.steps}
    assert statuses["s0"] is StepStatus.FAILED
    assert statuses["s1"] is StepStatus.FAILED
    assert statuses["s2"] is StepStatus.COMPLETED
    assert task.is_terminal()
