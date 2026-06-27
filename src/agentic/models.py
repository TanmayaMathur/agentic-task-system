"""Core data models, enums, and validation for the Agentic Task System.

This module defines the foundational data structures used throughout the
engine:

* :class:`StepStatus` / :class:`ResultStatus` -- the lifecycle and result
  enumerations.
* :class:`Step` -- a single node in a task's DAG.
* :class:`Task` -- the unit of work containing the original request text and
  the steps that form a DAG. Exposes :meth:`Task.ready_steps` and
  :meth:`Task.is_terminal`.
* :class:`AgentResult` -- the structured result an agent produces for a step,
  with invariants enforced in ``__post_init__``.
* :class:`BatchRequest` -- a single LLM request submitted by an agent to the
  batcher.

Validation rules from the design's *Data Models* section are enforced via
``__post_init__`` checks (for per-object invariants) and the :meth:`Task.validate`
helper (for whole-DAG invariants).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "StepStatus",
    "ResultStatus",
    "Step",
    "Task",
    "AgentResult",
    "BatchRequest",
]


class StepStatus(Enum):
    """Lifecycle status of a single :class:`Step`.

    A step starts :attr:`PENDING`, transitions to :attr:`RUNNING` while an
    agent executes it, and ends in one of the terminal states
    :attr:`COMPLETED`, :attr:`DEGRADED`, or :attr:`FAILED`.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class ResultStatus(Enum):
    """Outcome status carried by an :class:`AgentResult`."""

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


#: Step statuses that are considered terminal (the step will not run again).
_TERMINAL_STEP_STATUSES = frozenset(
    {StepStatus.COMPLETED, StepStatus.DEGRADED, StepStatus.FAILED}
)


@dataclass
class Step:
    """A single node in a :class:`Task`'s DAG.

    Attributes:
        id: Unique identifier within the owning Task.
        description: Human-readable description of the work to perform.
        agent_type: The agent specialization that handles this step
            (e.g. ``"retriever"``, ``"analyzer"``, ``"writer"``).
        depends_on: Ids of prerequisite steps that must be COMPLETED before
            this step becomes ready.
        status: Current :class:`StepStatus` (defaults to PENDING).
        result: The :class:`AgentResult` produced for this step, if any.
    """

    id: str
    description: str
    agent_type: str
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: "AgentResult | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Step.id must be a non-empty string")
        if not isinstance(self.agent_type, str) or not self.agent_type:
            raise ValueError("Step.agent_type must be a non-empty string")
        if not isinstance(self.depends_on, list):
            raise ValueError("Step.depends_on must be a list of step ids")

    def is_terminal(self) -> bool:
        """Return True when this step is in a terminal status."""
        return self.status in _TERMINAL_STEP_STATUSES


@dataclass
class Task:
    """The unit of work: original request text plus the DAG of steps.

    Attributes:
        id: Unique identifier for the task run.
        text: The original (non-empty) user request.
        steps: The nodes of the DAG.
    """

    id: str
    text: str
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforce the whole-DAG validation rules from the design.

        Raises:
            ValueError: If any of the following hold:
                * ``text`` is empty or whitespace-only.
                * Two steps share the same ``id``.
                * A ``depends_on`` entry references a non-existent step.
                * The dependency graph contains a cycle.
                * No step is dependency-free (no valid entry point).
        """
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Task.text must be a non-empty string")

        if not self.steps:
            raise ValueError("Task.steps must contain at least one step")

        # Unique step ids.
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"Duplicate Step.id: {step.id!r}")
            seen.add(step.id)

        # Every dependency references an existing step (and not itself).
        ids = seen
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Step {step.id!r} depends on unknown step {dep!r}"
                    )
                if dep == step.id:
                    raise ValueError(
                        f"Step {step.id!r} cannot depend on itself"
                    )

        # At least one dependency-free step (a valid entry point).
        if not any(not step.depends_on for step in self.steps):
            raise ValueError(
                "Task must have at least one dependency-free step (entry point)"
            )

        # Acyclic DAG check via iterative DFS with coloring.
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        """Raise ValueError if the dependency graph contains a cycle."""
        adjacency = {step.id: list(step.depends_on) for step in self.steps}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {sid: WHITE for sid in adjacency}

        for start in adjacency:
            if color[start] != WHITE:
                continue
            # Iterative DFS: stack holds (node, iterator over its deps).
            stack: list[tuple[str, object]] = [(start, iter(adjacency[start]))]
            color[start] = GRAY
            while stack:
                node, it = stack[-1]
                advanced = False
                for dep in it:  # type: ignore[assignment]
                    if color[dep] == GRAY:
                        raise ValueError(
                            f"Dependency cycle detected involving step {dep!r}"
                        )
                    if color[dep] == WHITE:
                        color[dep] = GRAY
                        stack.append((dep, iter(adjacency[dep])))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    stack.pop()

    def ready_steps(self) -> list[Step]:
        """Return the PENDING steps whose every dependency is COMPLETED.

        Returns an empty list when no such step exists. Dependency lookups
        read only COMPLETED states: a step blocked by a DEGRADED or FAILED
        dependency is *not* ready (the orchestrator handles such steps via the
        blocked-dependency branch).
        """
        status_by_id = {step.id: step.status for step in self.steps}
        ready: list[Step] = []
        for step in self.steps:
            if step.status is not StepStatus.PENDING:
                continue
            if all(
                status_by_id.get(dep) is StepStatus.COMPLETED
                for dep in step.depends_on
            ):
                ready.append(step)
        return ready

    def is_terminal(self) -> bool:
        """Return True when every step is COMPLETED, DEGRADED, or FAILED."""
        return all(step.is_terminal() for step in self.steps)


@dataclass
class AgentResult:
    """The structured result an agent produces for a step.

    Invariants (enforced in ``__post_init__``):
        * ``status == FAILED`` implies ``error`` is non-null.
        * ``status == DEGRADED`` implies ``degraded is True``.
        * ``output`` is always a string; an empty output is permitted only
          when ``status == FAILED``.
    """

    step_id: str
    status: ResultStatus
    output: str
    error: str | None = None
    attempts: int = 1
    degraded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResultStatus):
            raise ValueError("AgentResult.status must be a ResultStatus")
        if not isinstance(self.output, str):
            raise ValueError("AgentResult.output must be a string")

        if self.status is ResultStatus.FAILED and self.error is None:
            raise ValueError("AgentResult with status FAILED must have a non-null error")

        if self.status is ResultStatus.DEGRADED and not self.degraded:
            raise ValueError(
                "AgentResult with status DEGRADED must have degraded == True"
            )

        # Empty output is only allowed when the result FAILED.
        if not self.output and self.status is not ResultStatus.FAILED:
            raise ValueError(
                "AgentResult.output may be empty only when status is FAILED"
            )


@dataclass
class BatchRequest:
    """A single LLM request submitted by an agent to the batcher.

    Attributes:
        request_id: Unique identifier within the batcher's lifetime.
        prompt: The non-empty prompt string.
        enqueued_at: Monotonic timestamp (seconds) recorded at enqueue time.
        future: The future resolved with the matching :class:`Completion`.
            Must be unresolved at enqueue time.
    """

    request_id: str
    prompt: str
    enqueued_at: float
    future: "asyncio.Future"

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("BatchRequest.request_id must be a non-empty string")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("BatchRequest.prompt must be a non-empty string")
        if not isinstance(self.future, asyncio.Future):
            raise ValueError("BatchRequest.future must be an asyncio.Future")
        if self.future.done():
            raise ValueError("BatchRequest.future must be unresolved at enqueue time")
