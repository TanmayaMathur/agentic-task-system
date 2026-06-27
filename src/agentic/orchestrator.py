"""Orchestrator: own the end-to-end lifecycle of a task run.

This module implements *Component 1 (Orchestrator)* from the design. The
orchestrator ties the system together: it asks the :class:`~agentic.planner.Planner`
to decompose free text into a :class:`~agentic.models.Task` DAG, drives the
execution loop that schedules ready steps concurrently through the
:class:`~agentic.dispatcher.Dispatcher`, feeds partial results and lifecycle
events to the optional :class:`~agentic.streaming.StreamBus`, and guarantees the
run terminates with every step in a terminal status.

Lifecycle (:meth:`Orchestrator.run`)
------------------------------------
1. Reject empty/whitespace-only text by raising ``ValueError`` *before* any run
   begins (Requirement 1.2). The planner also rejects such input, but checking
   here ensures no lifecycle event is emitted and no Task is produced.
2. Decompose the text into a Task, retaining the original text (Requirement
   1.3); the planner echoes the text into :attr:`Task.text`.
3. Emit a ``STARTED`` lifecycle event when a stream is present, drive
   :meth:`_execute_dag` to completion, emit a ``DONE`` event, and return the
   final Task (Requirement 1.1).

DAG execution (:meth:`Orchestrator._execute_dag`)
-------------------------------------------------
The loop runs while the Task is not terminal. Each iteration computes the ready
steps (PENDING with every dependency COMPLETED -- Requirement 4.2, implemented
in :meth:`Task.ready_steps`). When there are ready steps, they are marked
RUNNING and executed concurrently via ``asyncio.gather`` (Requirement 4.1);
each step is given an :class:`~agentic.agents.ExecutionContext` built from the
:class:`~agentic.models.AgentResult` objects of its completed dependencies
(Requirement 3.3). Each result's status maps to a terminal step status
(OK -> COMPLETED, DEGRADED -> DEGRADED, FAILED -> FAILED), the result is stored
on the step, and a partial :class:`~agentic.streaming.StreamEvent` is emitted
(Requirement 5.1).

When no step is ready but the Task is not terminal, some dependency reached a
non-COMPLETED terminal status (DEGRADED/FAILED) so its dependents can never
become ready. The orchestrator marks every remaining non-terminal step FAILED
with an explanatory result, emits a partial event for each, and breaks the loop
(Requirement 4.5). Because the set of COMPLETED steps grows on every productive
iteration and the blocked branch terminates the loop otherwise, the loop always
terminates with every step terminal (Requirements 4.3, 4.4).

Requirements: 1.1, 1.2, 1.3, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1.
"""

from __future__ import annotations

import asyncio

from .agents import ExecutionContext
from .dispatcher import Dispatcher
from .models import AgentResult, ResultStatus, Step, StepStatus, Task
from .planner import Planner
from .streaming import StreamBus, StreamEvent

__all__ = ["Orchestrator"]

#: Maps an :class:`AgentResult` status to the terminal :class:`StepStatus` the
#: step should adopt once its result is in hand.
_RESULT_TO_STEP_STATUS = {
    ResultStatus.OK: StepStatus.COMPLETED,
    ResultStatus.DEGRADED: StepStatus.DEGRADED,
    ResultStatus.FAILED: StepStatus.FAILED,
}


class Orchestrator:
    """Drive a task run from free text to a fully-terminal :class:`Task`.

    Args:
        planner: The :class:`~agentic.planner.Planner` that decomposes task text
            into a validated DAG of steps.
        dispatcher: The :class:`~agentic.dispatcher.Dispatcher` that routes each
            ready step to the agent matching its ``agent_type``.
        stream: Optional :class:`~agentic.streaming.StreamBus` to receive
            lifecycle (``STARTED``/``DONE``) and per-step ``PARTIAL`` events.
            When ``None``, the orchestrator runs silently (no events emitted).
    """

    def __init__(
        self,
        planner: Planner,
        dispatcher: Dispatcher,
        stream: StreamBus | None = None,
    ) -> None:
        self._planner = planner
        self._dispatcher = dispatcher
        self._stream = stream

    async def run(self, task_text: str) -> Task:
        """Decompose, execute the DAG to completion, and return the Task.

        Rejects empty/whitespace-only input up front so no run begins
        (Requirement 1.2). On valid input, decomposes the text into a Task that
        retains the original text (Requirement 1.3), emits a ``STARTED`` event
        (when a stream is present), drives the DAG to termination, emits a
        ``DONE`` event, and returns the final Task (Requirement 1.1).

        Args:
            task_text: The original, non-empty user request.

        Returns:
            The final :class:`Task` with every step in a terminal status.

        Raises:
            ValueError: If ``task_text`` is empty or whitespace-only. No
                lifecycle event is emitted and no Task is produced.
        """
        # Reject before starting a run so no STARTED event is emitted and no
        # planning/decomposition work happens (Requirement 1.2).
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("task_text must be a non-empty string")

        # Decompose into a validated DAG; the planner retains the original text
        # on Task.text (Requirement 1.3).
        task = await self._planner.decompose(task_text)

        if self._stream is not None:
            self._stream.emit(StreamEvent.started({"task_id": task.id}))

        await self._execute_dag(task)

        if self._stream is not None:
            self._stream.emit(StreamEvent.done({"task_id": task.id}))

        return task

    async def _execute_dag(self, task: Task) -> None:
        """Schedule ready steps until every step reaches a terminal status.

        Loops while the Task is not terminal. Each iteration:

        * Computes ``task.ready_steps()`` -- PENDING steps whose dependencies
          are all COMPLETED (Requirement 4.2).
        * If none are ready (and the Task is not terminal), some dependency
          reached a non-COMPLETED terminal status, so the remaining steps are
          permanently blocked: mark them FAILED, emit a partial event for each,
          and break (Requirement 4.5).
        * Otherwise mark the ready steps RUNNING, build each step's
          :class:`ExecutionContext` from its completed dependencies' results
          (Requirement 3.3), run them concurrently with ``asyncio.gather``
          (Requirement 4.1), then map each result to a terminal step status,
          store it, and emit a partial event (Requirement 5.1).

        The COMPLETED set grows on each productive iteration and the blocked
        branch otherwise breaks, so the loop always terminates with every step
        terminal (Requirements 4.3, 4.4).
        """
        while not task.is_terminal():
            ready = task.ready_steps()

            if not ready:
                # No ready steps but the task is not terminal => one or more
                # dependencies ended DEGRADED/FAILED, permanently blocking their
                # dependents. Fail the blocked steps so the DAG terminates.
                self._fail_blocked_steps(task)
                break

            # Mark ready steps RUNNING before awaiting so readiness is computed
            # only over PENDING steps on the next iteration.
            for step in ready:
                step.status = StepStatus.RUNNING

            contexts = [self._build_context(task, step) for step in ready]
            coros = [
                self._dispatcher.dispatch(step, context)
                for step, context in zip(ready, contexts)
            ]
            results = await asyncio.gather(*coros)

            for step, result in zip(ready, results):
                self._apply_result(step, result)

    def _apply_result(self, step: Step, result: AgentResult) -> None:
        """Store ``result`` on ``step``, set its terminal status, and stream it.

        Maps the result status to the step's terminal status (OK -> COMPLETED,
        DEGRADED -> DEGRADED, FAILED -> FAILED) and emits a PARTIAL event so the
        subscriber sees the step's outcome as soon as it is known
        (Requirement 5.1).
        """
        step.result = result
        step.status = _RESULT_TO_STEP_STATUS[result.status]
        if self._stream is not None:
            self._stream.emit(StreamEvent.partial(step.id, result))

    def _build_context(self, task: Task, step: Step) -> ExecutionContext:
        """Build an :class:`ExecutionContext` from ``step``'s completed deps.

        Pulls the :class:`AgentResult` stored on each dependency that has
        COMPLETED and exposes it keyed by step id, so the agent can read its
        upstream outputs (Requirement 3.3). Only COMPLETED dependencies are
        included -- a step never runs until all its dependencies are COMPLETED
        (Requirement 4.2), so this captures exactly the inputs it should see.
        """
        results_by_id = {s.id: s for s in task.steps}
        context = ExecutionContext()
        for dep_id in step.depends_on:
            dep = results_by_id.get(dep_id)
            if (
                dep is not None
                and dep.status is StepStatus.COMPLETED
                and dep.result is not None
            ):
                context.add_result(dep_id, dep.result)
        return context

    def _fail_blocked_steps(self, task: Task) -> None:
        """Mark every still-non-terminal step FAILED and stream each result.

        Invoked when the Task is not terminal yet no step is ready: the
        remaining steps depend (directly or transitively) on a step that ended
        DEGRADED or FAILED, so they can never become ready. Each is given a
        FAILED :class:`AgentResult` with an explanatory ``error`` and a partial
        event is emitted, ensuring the DAG terminates (Requirement 4.5).
        """
        for step in task.steps:
            if step.is_terminal():
                continue
            blocked_deps = self._unsatisfied_dependencies(task, step)
            result = AgentResult(
                step_id=step.id,
                status=ResultStatus.FAILED,
                output="",
                error=(
                    "Step is blocked: dependencies did not complete successfully "
                    f"({', '.join(blocked_deps)})"
                    if blocked_deps
                    else "Step is blocked: no ready steps while task is not terminal"
                ),
            )
            self._apply_result(step, result)

    @staticmethod
    def _unsatisfied_dependencies(task: Task, step: Step) -> list[str]:
        """Return the ids of ``step``'s dependencies that are not COMPLETED."""
        status_by_id = {s.id: s.status for s in task.steps}
        return [
            dep_id
            for dep_id in step.depends_on
            if status_by_id.get(dep_id) is not StepStatus.COMPLETED
        ]
