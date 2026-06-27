"""Dispatcher: route each step to the agent matching its agent type.

This module implements *Component (Dispatcher)* referenced throughout the
design: it holds an agent registry keyed by :attr:`Agent.agent_type` and routes
a :class:`~agentic.models.Step` to the single agent whose type matches the
step's ``agent_type`` (Requirement 3.2).

Registry
--------
The registry is built once at construction from the supplied agents. Two agents
that report the same ``agent_type`` are a configuration error and raise
``ValueError`` immediately, so an ambiguous route can never occur at dispatch
time (Requirement 3.1 -- one Retriever, one Analyzer, one Writer).

The set of registered types is exposed via :attr:`Dispatcher.agent_types` so the
planner can validate that every step it produces names a known agent type
(Requirements 2.4 / 3.1, 3.2).

Routing contract
-----------------
:meth:`Dispatcher.dispatch` looks up the agent for ``step.agent_type`` and
delegates to its :meth:`Agent.execute`. Agents never raise past their own
boundary (they convert failures into ``AgentResult(status=FAILED)``), so the
dispatcher mirrors that discipline: when *no* agent is registered for a step's
type it returns a well-formed ``AgentResult(status=FAILED, error=...)`` rather
than raising, keeping the orchestrator's DAG flow consistent (the orchestrator
treats a FAILED result as a terminal status and routes around it).

Requirements: 3.1, 3.2.
"""

from __future__ import annotations

from .agents import Agent, ExecutionContext
from .models import AgentResult, ResultStatus, Step

__all__ = ["Dispatcher"]


class Dispatcher:
    """Route steps to specialized agents by ``agent_type``.

    Args:
        agents: The specialized agents to register. Each agent's
            :attr:`Agent.agent_type` must be unique within the list.

    Raises:
        ValueError: If two agents share the same ``agent_type``, or if an agent
            reports an empty ``agent_type``.
    """

    def __init__(self, agents: list[Agent]) -> None:
        registry: dict[str, Agent] = {}
        for agent in agents:
            agent_type = agent.agent_type
            if not agent_type:
                raise ValueError(
                    f"Agent {type(agent).__name__} has an empty agent_type"
                )
            if agent_type in registry:
                raise ValueError(
                    f"Duplicate agent_type in registry: {agent_type!r}"
                )
            registry[agent_type] = agent
        self._registry = registry

    @property
    def agent_types(self) -> set[str]:
        """Return the set of registered agent types.

        Exposed so the planner can validate that every step it produces names
        an agent type the dispatcher can actually route to (Requirements 3.1,
        3.2).
        """
        return set(self._registry)

    def has_agent(self, agent_type: str) -> bool:
        """Return True when an agent is registered for ``agent_type``."""
        return agent_type in self._registry

    async def dispatch(self, step: Step, context: ExecutionContext) -> AgentResult:
        """Route ``step`` to its matching agent and return the result.

        Selects the agent whose ``agent_type`` equals ``step.agent_type`` and
        awaits its :meth:`Agent.execute`. When no agent is registered for the
        step's type, returns a well-formed ``FAILED`` result (with the error
        populated) instead of raising, so the orchestrator's flow stays
        consistent (Requirement 3.2).
        """
        agent = self._registry.get(step.agent_type)
        if agent is None:
            return AgentResult(
                step_id=step.id,
                status=ResultStatus.FAILED,
                output="",
                error=(
                    f"No agent registered for agent_type {step.agent_type!r}; "
                    f"known types: {sorted(self._registry)}"
                ),
            )
        return await agent.execute(step, context)
