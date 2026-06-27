"""Unit tests for :class:`agentic.planner.Planner` DAG validity (Task 8.2).

These tests verify that :meth:`Planner.decompose` produces a Task whose Steps
form a valid DAG with registered agent types and a valid entry point, that
every dependency references an existing Step, that a topological ordering
exists, that empty/whitespace-only input is rejected, and that the original
task text is retained.

_Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
"""

from __future__ import annotations

import pytest

from agentic.models import StepStatus, Task
from agentic.planner import Planner
from agentic.providers import MockProvider


# A representative non-empty task used across the happy-path tests.
SAMPLE_TEXT = (
    "Research async batching strategies, analyze the tradeoffs, "
    "and write a concise summary."
)


def _make_planner() -> Planner:
    """Construct a Planner backed by the deterministic MockProvider."""
    return Planner(MockProvider())


def _topological_order(task: Task) -> list[str]:
    """Return a topological ordering of ``task``'s step ids (Kahn's algorithm).

    Raises:
        AssertionError: If the graph contains a cycle (no valid ordering).
    """
    ids = [step.id for step in task.steps]
    deps = {step.id: set(step.depends_on) for step in task.steps}
    # Count of unmet prerequisites per step.
    indegree = {sid: len(deps[sid]) for sid in ids}
    # Edges dep -> dependents, so completing dep can unblock dependents.
    dependents: dict[str, list[str]] = {sid: [] for sid in ids}
    for sid in ids:
        for dep in deps[sid]:
            dependents[dep].append(sid)

    ready = [sid for sid in ids if indegree[sid] == 0]
    order: list[str] = []
    while ready:
        current = ready.pop()
        order.append(current)
        for dependent in dependents[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    assert len(order) == len(ids), "graph is not acyclic: no topological order"
    return order


# --- Requirement 2.1: at least one step --------------------------------------


async def test_decompose_produces_at_least_one_step() -> None:
    """A non-empty task yields a Task with one or more Steps (Req 2.1)."""
    planner = _make_planner()

    task = await planner.decompose(SAMPLE_TEXT)

    assert isinstance(task, Task)
    assert len(task.steps) >= 1


# --- Requirement 2.4: registered agent types ---------------------------------


async def test_all_agent_types_are_registered() -> None:
    """Every step's agent_type is drawn from planner.agent_types (Req 2.4)."""
    planner = _make_planner()

    task = await planner.decompose(SAMPLE_TEXT)

    assert planner.agent_types  # registry is non-empty
    for step in task.steps:
        assert step.agent_type in planner.agent_types


# --- Requirement 2.3: a valid entry point exists -----------------------------


async def test_has_dependency_free_entry_point() -> None:
    """At least one step has no dependencies and is ready initially (Req 2.3)."""
    planner = _make_planner()

    task = await planner.decompose(SAMPLE_TEXT)

    # At least one structural entry point (no declared dependencies).
    assert any(not step.depends_on for step in task.steps)

    # All steps start PENDING, so ready_steps() must surface that entry point.
    assert all(step.status is StepStatus.PENDING for step in task.steps)
    ready = task.ready_steps()
    assert ready, "expected at least one ready step on a freshly planned task"
    assert all(not step.depends_on for step in ready)


# --- Requirement 2.5: dependencies reference existing steps -------------------


async def test_every_dependency_references_existing_step() -> None:
    """Each declared dependency references a Step in the same Task (Req 2.5)."""
    planner = _make_planner()

    task = await planner.decompose(SAMPLE_TEXT)

    existing_ids = {step.id for step in task.steps}
    for step in task.steps:
        for dep in step.depends_on:
            assert dep in existing_ids
            assert dep != step.id  # a step never depends on itself


# --- Requirement 2.2: acyclic DAG with a topological ordering ----------------


async def test_dag_is_acyclic_topological_order_exists() -> None:
    """A topological ordering exists and respects every dependency (Req 2.2)."""
    planner = _make_planner()

    task = await planner.decompose(SAMPLE_TEXT)

    order = _topological_order(task)
    position = {sid: idx for idx, sid in enumerate(order)}

    # Each step appears after all of its dependencies in the ordering.
    for step in task.steps:
        for dep in step.depends_on:
            assert position[dep] < position[step.id]


async def test_canonical_chain_ordering() -> None:
    """The canonical plan is the linear chain s1 -> s2 -> s3 (Req 2.2)."""
    planner = _make_planner()

    task = await planner.decompose(SAMPLE_TEXT)

    steps_by_id = {step.id: step for step in task.steps}
    assert set(steps_by_id) == {"s1", "s2", "s3"}

    assert steps_by_id["s1"].depends_on == []
    assert steps_by_id["s2"].depends_on == ["s1"]
    assert steps_by_id["s3"].depends_on == ["s2"]

    # The canonical chain mirrors the agent specializations.
    assert steps_by_id["s1"].agent_type == "retriever"
    assert steps_by_id["s2"].agent_type == "analyzer"
    assert steps_by_id["s3"].agent_type == "writer"

    # The only valid topological order for a linear chain.
    assert _topological_order(task) == ["s1", "s2", "s3"]


# --- Original text retained --------------------------------------------------


async def test_original_text_is_retained() -> None:
    """The Task retains the (stripped) original request text."""
    planner = _make_planner()

    task = await planner.decompose(SAMPLE_TEXT)

    assert task.text == SAMPLE_TEXT


async def test_surrounding_whitespace_is_stripped_but_text_retained() -> None:
    """Leading/trailing whitespace is trimmed while the content is retained."""
    planner = _make_planner()

    task = await planner.decompose(f"   {SAMPLE_TEXT}   ")

    assert task.text == SAMPLE_TEXT


# --- Empty / whitespace-only input rejection ---------------------------------


@pytest.mark.parametrize("bad_text", ["", "   ", "\t", "\n", "  \n\t "])
async def test_empty_or_whitespace_input_is_rejected(bad_text: str) -> None:
    """Empty/whitespace-only input raises ValueError before a Task is made."""
    planner = _make_planner()

    with pytest.raises(ValueError):
        await planner.decompose(bad_text)
