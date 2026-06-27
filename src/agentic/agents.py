"""Specialized agents and their uniform execution contract.

This module implements *Component 3* from the design: a uniform :class:`Agent`
contract plus three specializations -- :class:`RetrieverAgent`,
:class:`AnalyzerAgent`, and :class:`WriterAgent`.

Execution contract
-------------------
Every agent translates a :class:`~agentic.models.Step` (plus the outputs of its
already-completed dependencies) into a single prompt, submits that prompt to the
injected :class:`~agentic.batcher.Batcher`, and wraps the resulting
:class:`~agentic.providers.Completion` in a well-formed
:class:`~agentic.models.AgentResult` (Requirement 3.4).

The shared :meth:`Agent.execute` is the only place this happens, so the
specializations only differ in how they build their prompt
(:meth:`Agent._build_prompt`). Crucially, ``execute`` converts *any* exception
into ``AgentResult(status=FAILED, error=...)`` so no exception ever crosses the
agent boundary (Requirement 3.5). When the completion came back from the
batcher's degraded fallback (its text is prefixed ``"[degraded]"``), the agent
reports ``status=DEGRADED`` with ``degraded=True`` so the degradation is visible
downstream (supports Requirement 8.6 / 9.2).

Dependency outputs are read from an :class:`ExecutionContext`, which maps a
completed step id to its produced output (Requirement 3.3). A Retriever has no
dependencies; an Analyzer reads its Retriever's output; a Writer reads its
Analyzer's output.

Prompt prefixes (``"retrieve:"``, ``"analyze:"``, ``"write:"``) match the
``MockProvider`` glob examples in the design so the deterministic mock can map
each agent's prompt to a canned completion.

Requirements: 3.1, 3.3, 3.4, 3.5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Union

from .batcher import Batcher
from .models import AgentResult, ResultStatus, Step

__all__ = [
    "ExecutionContext",
    "Agent",
    "RetrieverAgent",
    "AnalyzerAgent",
    "WriterAgent",
]

#: Prefix the batcher's degraded fallback stamps onto completion text.
_DEGRADED_PREFIX = "[degraded]"

#: A context entry may be a full AgentResult or a bare output string.
_ContextValue = Union[AgentResult, str]


class ExecutionContext:
    """Read-only-ish view of completed-dependency outputs, keyed by step id.

    The orchestrator populates this as steps complete; an agent reads the
    outputs of its step's dependencies from it (Requirement 3.3). Each entry is
    either a full :class:`~agentic.models.AgentResult` or a bare output string,
    so callers can build a context cheaply from whichever they have on hand.

    Args:
        outputs: Optional initial mapping of ``step_id`` to either an
            :class:`AgentResult` or an output ``str``.
    """

    def __init__(self, outputs: Mapping[str, _ContextValue] | None = None) -> None:
        self._results: dict[str, _ContextValue] = dict(outputs or {})

    def add_result(self, step_id: str, result: _ContextValue) -> None:
        """Record a completed step's result (or bare output) by id."""
        self._results[step_id] = result

    def get_result(self, step_id: str) -> AgentResult | None:
        """Return the stored :class:`AgentResult` for ``step_id``, if any.

        Returns ``None`` when the id is unknown or when only a bare output
        string was recorded for it.
        """
        value = self._results.get(step_id)
        return value if isinstance(value, AgentResult) else None

    def get_output(self, step_id: str) -> str:
        """Return the output string for ``step_id`` (``""`` if unknown).

        Accepts entries stored either as an :class:`AgentResult` (its
        ``output`` is returned) or as a bare string.
        """
        value = self._results.get(step_id)
        if value is None:
            return ""
        if isinstance(value, AgentResult):
            return value.output
        return str(value)

    def dependency_outputs(self, step: Step) -> dict[str, str]:
        """Return ``{dep_id: output}`` for the step's known dependencies.

        Only dependencies that are present in the context are included, so an
        agent never sees a placeholder for a dependency that has not completed.
        Insertion order follows ``step.depends_on`` for deterministic prompts.
        """
        outputs: dict[str, str] = {}
        for dep_id in step.depends_on:
            if dep_id in self._results:
                outputs[dep_id] = self.get_output(dep_id)
        return outputs


class Agent(ABC):
    """Uniform execution contract shared by every specialized agent.

    Subclasses set the class attribute :attr:`agent_type` and implement
    :meth:`_build_prompt`. The concrete :meth:`execute` here performs the shared
    work: build the prompt, submit it through the injected
    :class:`~agentic.batcher.Batcher`, and wrap the outcome in a well-formed
    :class:`~agentic.models.AgentResult` -- never letting an exception escape
    (Requirement 3.5).

    Args:
        batcher: The :class:`~agentic.batcher.Batcher` used to submit prompts.
    """

    #: The agent specialization this class handles (overridden by subclasses).
    agent_type: str = ""

    def __init__(self, batcher: Batcher) -> None:
        self._batcher = batcher

    async def execute(self, step: Step, context: ExecutionContext) -> AgentResult:
        """Run the step's work and return a well-formed :class:`AgentResult`.

        Builds the prompt from the step plus its dependency outputs, submits it
        to the batcher, and maps the completion to a result:

        * Completion text prefixed ``"[degraded]"`` -> ``DEGRADED`` with
          ``degraded=True``.
        * Otherwise -> ``OK`` carrying the completion text as ``output``.

        Any exception raised while building the prompt, submitting, or wrapping
        the result is caught and converted into a ``FAILED`` result with a
        populated ``error`` -- no exception crosses the agent boundary
        (Requirement 3.5).
        """
        try:
            prompt = self._build_prompt(step, context)
            completion = await self._batcher.submit(prompt)
            text = completion.text

            if isinstance(text, str) and text.startswith(_DEGRADED_PREFIX):
                return AgentResult(
                    step_id=step.id,
                    status=ResultStatus.DEGRADED,
                    output=text,
                    degraded=True,
                )
            return AgentResult(
                step_id=step.id,
                status=ResultStatus.OK,
                output=text,
            )
        except Exception as exc:  # noqa: BLE001 - failures never cross the boundary
            return AgentResult(
                step_id=step.id,
                status=ResultStatus.FAILED,
                output="",
                error=f"{type(exc).__name__}: {exc}",
            )

    @abstractmethod
    def _build_prompt(self, step: Step, context: ExecutionContext) -> str:
        """Translate the step (and dependency outputs) into a single prompt.

        Implementations MUST return a non-empty string; the batcher rejects
        empty prompts. The returned prompt should carry the agent's prefix so
        the deterministic ``MockProvider`` can map it.
        """
        raise NotImplementedError


def _format_dependencies(context: ExecutionContext, step: Step) -> str:
    """Render a step's dependency outputs as a stable, labelled block.

    Produces one ``"[dep_id] output"`` line per known dependency, in
    ``depends_on`` order, or an empty string when there are none.
    """
    deps = context.dependency_outputs(step)
    return "\n".join(f"[{dep_id}] {output}" for dep_id, output in deps.items())


class RetrieverAgent(Agent):
    """Gathers source material / context for a step.

    A retriever sits at the root of the DAG and therefore has no upstream
    dependency outputs to incorporate; its prompt is built solely from the
    step description.
    """

    agent_type = "retriever"

    def _build_prompt(self, step: Step, context: ExecutionContext) -> str:
        return f"retrieve: {step.description}"


class AnalyzerAgent(Agent):
    """Reasons over the material produced by upstream retriever step(s)."""

    agent_type = "analyzer"

    def _build_prompt(self, step: Step, context: ExecutionContext) -> str:
        material = _format_dependencies(context, step)
        if material:
            return f"analyze: {step.description}\n{material}"
        return f"analyze: {step.description}"


class WriterAgent(Agent):
    """Produces final user-facing prose from upstream analyzer output(s)."""

    agent_type = "writer"

    def _build_prompt(self, step: Step, context: ExecutionContext) -> str:
        findings = _format_dependencies(context, step)
        if findings:
            return f"write: {step.description}\n{findings}"
        return f"write: {step.description}"
