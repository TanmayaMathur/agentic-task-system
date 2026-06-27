"""Task decomposition: turn free text into a validated DAG of Steps.

This module implements :class:`Planner`, the component that converts a
free-text user request into a :class:`~agentic.models.Task` whose
:class:`~agentic.models.Step` objects form a directed acyclic graph
(Requirement 2). The decomposition is a deterministic three-stage heuristic
that mirrors the system's agent specializations:

* a **retriever** step that gathers source material (no dependencies),
* an **analyzer** step that reasons over the retrieved material (depends on
  the retriever), and
* a **writer** step that produces the final user-facing prose (depends on the
  analyzer).

The heuristic is intentionally LLM-free so planning is fully deterministic and
key-free, yet the constructor still accepts an :class:`LLMProvider` to conform
to the design's interface (and to leave room for an LLM-driven planner later).
A :class:`LLMProvider` is also where the agent-type registry would ultimately
come from; here it is supplied/defaulted directly.

The resulting plan is linear (``s1 -> s2 -> s3``), which trivially satisfies
the DAG invariants enforced by :meth:`Task.validate`: a topological ordering
exists (Req 2.2), exactly one step is dependency-free (Req 2.3), every
``agent_type`` is drawn from the registry (Req 2.4), and every dependency
references an existing step in the same Task (Req 2.5).
"""

from __future__ import annotations

import hashlib

from .models import Step, Task
from .providers import LLMProvider

__all__ = ["Planner"]

#: The default registry of valid agent types. These mirror the three
#: specialized agents described in the design (Requirement 3.1) and are used
#: when the caller does not supply an explicit registry.
_DEFAULT_AGENT_TYPES = frozenset({"retriever", "analyzer", "writer"})

#: Maximum characters of the task text echoed into a step description, keeping
#: descriptions readable without dropping information needed to identify them.
_DESCRIPTION_SNIPPET = 200


class Planner:
    """Convert free-text task input into a validated DAG of Steps.

    The planner produces a canonical three-stage plan (retriever -> analyzer
    -> writer). Each step is given a stable id (``"s1"``, ``"s2"``, ``"s3"``)
    and a description derived from the original task text, and the dependency
    edges form a simple linear chain so a topological ordering always exists.

    Args:
        provider: The LLM provider, retained to honor the design's interface.
            The default heuristic decomposition does not call it, keeping
            planning deterministic and key-free.
        agent_types: The registry of valid agent types. When ``None``, defaults
            to ``{"retriever", "analyzer", "writer"}``. Every step the planner
            emits draws its ``agent_type`` from this registry (Requirement 2.4).

    Raises:
        ValueError: If ``provider`` is not an :class:`LLMProvider`, or if a
            supplied ``agent_types`` registry is missing any of the agent types
            the canonical plan needs.
    """

    def __init__(
        self,
        provider: LLMProvider,
        agent_types: set[str] | None = None,
    ) -> None:
        if not isinstance(provider, LLMProvider):
            raise ValueError("Planner.provider must be an LLMProvider")

        self._provider = provider
        self._agent_types: frozenset[str] = (
            frozenset(agent_types) if agent_types is not None else _DEFAULT_AGENT_TYPES
        )

        # The canonical plan relies on these three specializations existing in
        # the registry; fail fast if a custom registry cannot satisfy it
        # (otherwise Req 2.4 — every assigned type is registered — is unmet).
        required = {"retriever", "analyzer", "writer"}
        missing = required - self._agent_types
        if missing:
            raise ValueError(
                "Planner agent_types registry is missing required types: "
                f"{sorted(missing)}"
            )

    @property
    def agent_types(self) -> frozenset[str]:
        """The registry of valid agent types this planner assigns from."""
        return self._agent_types

    async def decompose(self, task_text: str) -> Task:
        """Decompose ``task_text`` into a :class:`Task` whose Steps form a DAG.

        The decomposition is deterministic: a retriever step (no dependencies)
        feeds an analyzer step, which feeds a writer step. This guarantees at
        least one Step (Req 2.1), a valid topological order (Req 2.2), a
        dependency-free entry point (Req 2.3), registered agent types
        (Req 2.4), and intra-Task dependency references (Req 2.5).

        Args:
            task_text: The original, non-empty user request.

        Returns:
            A validated :class:`Task` retaining the original ``task_text`` and
            containing the three-stage DAG of Steps.

        Raises:
            ValueError: If ``task_text`` is empty or whitespace-only. The
                rejection happens before any Task is produced, supporting the
                orchestrator's input-rejection requirement (Requirement 1.2).
        """
        # Reject empty/whitespace-only input up front, before producing a Task.
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("task_text must be a non-empty string")

        normalized = task_text.strip()
        snippet = self._snippet(normalized)

        # Canonical linear plan: retriever -> analyzer -> writer. The ids are
        # stable and dependencies reference only earlier steps in this Task.
        steps = [
            Step(
                id="s1",
                description=f"Retrieve source material for: {snippet}",
                agent_type="retriever",
                depends_on=[],
            ),
            Step(
                id="s2",
                description=f"Analyze the retrieved material for: {snippet}",
                agent_type="analyzer",
                depends_on=["s1"],
            ),
            Step(
                id="s3",
                description=f"Write the final response for: {snippet}",
                agent_type="writer",
                depends_on=["s2"],
            ),
        ]

        # Task.__post_init__ validates the whole-DAG invariants (acyclicity,
        # entry point, unique ids, intra-Task dependency references).
        return Task(id=self._make_task_id(normalized), text=normalized, steps=steps)

    @staticmethod
    def _snippet(text: str) -> str:
        """Return a single-line, length-bounded snippet of ``text``."""
        collapsed = " ".join(text.split())
        if len(collapsed) <= _DESCRIPTION_SNIPPET:
            return collapsed
        return collapsed[: _DESCRIPTION_SNIPPET - 1].rstrip() + "\u2026"

    @staticmethod
    def _make_task_id(text: str) -> str:
        """Derive a stable, unique-enough task id from the task text.

        Uses a SHA-256 prefix so the same text maps to the same id across runs
        (deterministic), while distinct texts map to distinct ids in practice.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"task-{digest}"
