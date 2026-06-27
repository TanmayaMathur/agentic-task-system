"""Unit and property tests for :mod:`agentic.models`.

Covers:

* Task 2.2 -- unit tests for ``Task.ready_steps()`` / ``Task.is_terminal()``
  across mixed step states, plus the whole-DAG validation rejections
  (dependency cycle, duplicate ``Step.id``, dangling/unknown dependency,
  empty ``Task.text``, and no dependency-free step).
  _Requirements: 2.1, 2.3, 2.5, 4.2_

* Task 2.3 -- property test for **Property 12: AgentResult invariant**.
  _Validates: Requirements 9.1, 9.3, 3.5_
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentic.models import (
    AgentResult,
    ResultStatus,
    Step,
    StepStatus,
    Task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(
    step_id: str,
    *,
    depends_on: list[str] | None = None,
    status: StepStatus = StepStatus.PENDING,
    agent_type: str = "retriever",
) -> Step:
    """Build a Step with sensible defaults for tests."""
    return Step(
        id=step_id,
        description=f"do {step_id}",
        agent_type=agent_type,
        depends_on=list(depends_on or []),
        status=status,
    )


def _two_step_task(
    a_status: StepStatus = StepStatus.PENDING,
    b_status: StepStatus = StepStatus.PENDING,
) -> Task:
    """A minimal valid DAG: ``a`` (entry) -> ``b`` (depends on ``a``)."""
    return Task(
        id="t1",
        text="research and write",
        steps=[
            _step("a", status=a_status),
            _step("b", depends_on=["a"], status=b_status),
        ],
    )


# ---------------------------------------------------------------------------
# Task 2.2 -- ready_steps() across mixed step states
# ---------------------------------------------------------------------------

def test_ready_steps_all_pending_returns_only_entry_points():
    task = _two_step_task()
    ready_ids = {s.id for s in task.ready_steps()}
    # Only the dependency-free step is ready; ``b`` waits on ``a``.
    assert ready_ids == {"a"}


def test_ready_steps_completed_dependency_unblocks_dependent():
    task = _two_step_task(a_status=StepStatus.COMPLETED)
    ready_ids = {s.id for s in task.ready_steps()}
    # ``a`` is no longer PENDING; ``b`` is now ready because ``a`` COMPLETED.
    assert ready_ids == {"b"}


def test_ready_steps_running_dependency_blocks_dependent():
    task = _two_step_task(a_status=StepStatus.RUNNING)
    # ``a`` is RUNNING (not PENDING) and ``b`` needs ``a`` COMPLETED.
    assert task.ready_steps() == []


def test_ready_steps_degraded_dependency_does_not_unblock():
    task = _two_step_task(a_status=StepStatus.DEGRADED)
    # A DEGRADED dependency is NOT COMPLETED, so ``b`` stays blocked.
    assert task.ready_steps() == []


def test_ready_steps_failed_dependency_does_not_unblock():
    task = _two_step_task(a_status=StepStatus.FAILED)
    assert task.ready_steps() == []


def test_ready_steps_empty_when_no_pending_steps():
    task = _two_step_task(
        a_status=StepStatus.COMPLETED, b_status=StepStatus.COMPLETED
    )
    assert task.ready_steps() == []


def test_ready_steps_independent_steps_all_ready():
    task = Task(
        id="t2",
        text="parallel work",
        steps=[_step("a"), _step("b"), _step("c")],
    )
    ready_ids = {s.id for s in task.ready_steps()}
    assert ready_ids == {"a", "b", "c"}


def test_ready_steps_multiple_dependencies_require_all_completed():
    # ``c`` depends on both ``a`` and ``b``.
    def build(a: StepStatus, b: StepStatus) -> Task:
        return Task(
            id="t3",
            text="fan in",
            steps=[
                _step("a", status=a),
                _step("b", status=b),
                _step("c", depends_on=["a", "b"]),
            ],
        )

    # Only one dependency completed -> ``c`` not ready.
    partial = build(StepStatus.COMPLETED, StepStatus.RUNNING)
    assert "c" not in {s.id for s in partial.ready_steps()}

    # Both dependencies completed -> ``c`` ready.
    both = build(StepStatus.COMPLETED, StepStatus.COMPLETED)
    assert {s.id for s in both.ready_steps()} == {"c"}


# ---------------------------------------------------------------------------
# Task 2.2 -- is_terminal() across mixed step states
# ---------------------------------------------------------------------------

def test_is_terminal_false_with_pending_step():
    task = _two_step_task(
        a_status=StepStatus.COMPLETED, b_status=StepStatus.PENDING
    )
    assert task.is_terminal() is False


def test_is_terminal_false_with_running_step():
    task = _two_step_task(
        a_status=StepStatus.COMPLETED, b_status=StepStatus.RUNNING
    )
    assert task.is_terminal() is False


def test_is_terminal_true_when_all_completed():
    task = _two_step_task(
        a_status=StepStatus.COMPLETED, b_status=StepStatus.COMPLETED
    )
    assert task.is_terminal() is True


def test_is_terminal_true_with_mixed_terminal_states():
    task = Task(
        id="t4",
        text="mixed terminal",
        steps=[
            _step("a", status=StepStatus.COMPLETED),
            _step("b", status=StepStatus.DEGRADED),
            _step("c", status=StepStatus.FAILED),
        ],
    )
    assert task.is_terminal() is True


# ---------------------------------------------------------------------------
# Task 2.2 -- whole-DAG validation rejections
# ---------------------------------------------------------------------------

def test_reject_empty_text():
    with pytest.raises(ValueError, match="non-empty string"):
        Task(id="t", text="", steps=[_step("a")])


def test_reject_whitespace_only_text():
    with pytest.raises(ValueError, match="non-empty string"):
        Task(id="t", text="   \t\n", steps=[_step("a")])


def test_reject_duplicate_step_id():
    with pytest.raises(ValueError, match="Duplicate Step.id"):
        Task(
            id="t",
            text="dup ids",
            steps=[_step("a"), _step("a")],
        )


def test_reject_unknown_dependency():
    with pytest.raises(ValueError, match="unknown step"):
        Task(
            id="t",
            text="dangling dep",
            steps=[_step("a"), _step("b", depends_on=["does-not-exist"])],
        )


def test_reject_self_dependency():
    with pytest.raises(ValueError, match="cannot depend on itself"):
        Task(
            id="t",
            text="self dep",
            steps=[_step("a"), _step("b", depends_on=["b"])],
        )


def test_reject_no_dependency_free_step():
    # Mutual dependency: neither step is an entry point.
    with pytest.raises(ValueError, match="dependency-free step"):
        Task(
            id="t",
            text="no entry point",
            steps=[
                _step("a", depends_on=["b"]),
                _step("b", depends_on=["a"]),
            ],
        )


def test_reject_dependency_cycle():
    # ``a`` is a valid entry point, but ``b`` <-> ``c`` form a cycle, so the
    # acyclic check (not the entry-point check) is what must fire here.
    with pytest.raises(ValueError, match="cycle"):
        Task(
            id="t",
            text="cyclic branch",
            steps=[
                _step("a"),
                _step("b", depends_on=["c"]),
                _step("c", depends_on=["b"]),
            ],
        )


def test_reject_empty_steps():
    with pytest.raises(ValueError, match="at least one step"):
        Task(id="t", text="no steps", steps=[])


def test_valid_dag_constructs_successfully():
    # Sanity: a well-formed DAG does not raise.
    task = _two_step_task()
    assert len(task.steps) == 2


# ---------------------------------------------------------------------------
# Task 2.3 -- Property 12: AgentResult invariant
# Validates: Requirements 9.1, 9.3, 3.5
# ---------------------------------------------------------------------------

def _expected_valid(
    status: ResultStatus, output: str, error: str | None, degraded: bool
) -> bool:
    """Predict whether an AgentResult with these fields should construct.

    Encodes the three invariants directly from the design:
      * status == FAILED  => error is not None        (Req 9.1)
      * status == DEGRADED => degraded is True         (Req 9.2/9.3)
      * output may be empty only when status == FAILED (Req 9.3)
    """
    if status is ResultStatus.FAILED and error is None:
        return False
    if status is ResultStatus.DEGRADED and not degraded:
        return False
    if output == "" and status is not ResultStatus.FAILED:
        return False
    return True


@given(
    step_id=st.text(min_size=1, max_size=8),
    status=st.sampled_from(list(ResultStatus)),
    output=st.text(max_size=20),  # includes the empty string
    error=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    attempts=st.integers(min_value=1, max_value=5),
    degraded=st.booleans(),
)
def test_property_12_agent_result_invariant(
    step_id, status, output, error, attempts, degraded
):
    """For any field combination, the constructor enforces the invariants.

    Invalid combinations must raise ``ValueError``; valid combinations must
    construct successfully and hold the invariant afterwards.
    """
    should_construct = _expected_valid(status, output, error, degraded)

    if not should_construct:
        with pytest.raises(ValueError):
            AgentResult(
                step_id=step_id,
                status=status,
                output=output,
                error=error,
                attempts=attempts,
                degraded=degraded,
            )
        return

    result = AgentResult(
        step_id=step_id,
        status=status,
        output=output,
        error=error,
        attempts=attempts,
        degraded=degraded,
    )

    # Invariants must hold on every successfully-constructed result.
    if result.status is ResultStatus.FAILED:
        assert result.error is not None
    if result.status is ResultStatus.DEGRADED:
        assert result.degraded is True
    if result.output == "":
        assert result.status is ResultStatus.FAILED
    assert isinstance(result.output, str)


@given(
    step_id=st.text(min_size=1, max_size=8),
    output=st.text(min_size=1, max_size=20),
    attempts=st.integers(min_value=1, max_value=5),
)
def test_property_12_valid_ok_results_always_construct(step_id, output, attempts):
    """A focused generator over the valid OK space always constructs."""
    result = AgentResult(
        step_id=step_id,
        status=ResultStatus.OK,
        output=output,
        attempts=attempts,
    )
    assert result.status is ResultStatus.OK
    assert result.output == output


@given(
    step_id=st.text(min_size=1, max_size=8),
    output=st.text(max_size=20),
    error=st.text(min_size=1, max_size=20),
)
def test_property_12_failed_with_error_allows_empty_output(step_id, output, error):
    """FAILED results with a non-null error construct even with empty output."""
    result = AgentResult(
        step_id=step_id,
        status=ResultStatus.FAILED,
        output=output,
        error=error,
    )
    assert result.status is ResultStatus.FAILED
    assert result.error is not None
