"""Runnable end-to-end demo for the Agentic AI System.

This module wires every component of the system together into a single
runnable program (Task 11.1) so the whole pipeline can be exercised without an
API key, using the deterministic :class:`~agentic.providers.MockProvider`
(Requirement 7.5). It demonstrates both:

* the **happy path** -- ``python -m agentic.cli`` -- where the retriever,
  analyzer, and writer steps all complete and their partial outputs stream to
  the console as they are produced (Requirement 5.1); and
* the **failure path** -- ``python -m agentic.cli --fail`` -- where the
  ``MockProvider`` is configured to fail on the analyzer's prompts
  (``fail_on={"analyze:*"}``, Requirement 7.6). A *transient* failure exhausts
  the batcher's retries and falls back to a degraded completion, so the
  analyzer step ends DEGRADED and a degraded-marked chunk streams to the user
  (Requirement 8.6). The writer depends on the analyzer, so once the analyzer
  is non-COMPLETED the orchestrator marks the writer's blocked branch FAILED --
  the run still terminates cleanly.

Wiring
------
``MockProvider`` -> ``FailureHandler`` -> ``Batcher`` -> the three agents
(``RetrieverAgent`` / ``AnalyzerAgent`` / ``WriterAgent``) routed by the
``Dispatcher``, decomposed by the ``Planner`` and driven by the
``Orchestrator``, with a ``StreamBus`` consumed concurrently by a printer task.

The stream is consumed *concurrently* with execution: a consumer task iterates
``StreamBus.subscribe()`` and prints each event, tagging degraded/failed events
distinctly. After :meth:`Orchestrator.run` returns, the bus is closed, the
consumer is awaited to drain, and ``Task.is_terminal()`` is asserted.

Requirements: 5.1, 7.5, 7.6, 8.6.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .agents import AnalyzerAgent, RetrieverAgent, WriterAgent
from .batcher import Batcher
from .dispatcher import Dispatcher
from .errors import PermanentError, TransientError
from .failure import FailureHandler
from .models import ResultStatus, StepStatus, Task
from .orchestrator import Orchestrator
from .planner import Planner
from .providers import MockProvider
from .streaming import StreamBus, StreamEvent, StreamEventKind

__all__ = ["build_orchestrator", "run_demo", "main"]

#: A sensible default multi-part request that exercises all three agents.
DEFAULT_TASK_TEXT = (
    "Research async batching, analyze the tradeoffs, and write a summary."
)

#: Canned MockProvider responses for the happy path. Keys are globs matched
#: against each agent's prefixed prompt (see agents.py), so the deterministic
#: mock returns meaningful, readable text for every step.
DEFAULT_RESPONSES = {
    "retrieve:*": (
        "Sources gathered: (1) asyncio docs on queues and gather, "
        "(2) a blog post on manual request batching, "
        "(3) notes on backpressure and circuit breakers."
    ),
    "analyze:*": (
        "Analysis: batching amortizes provider round-trips and raises "
        "throughput, but a fixed time window trades latency for efficiency; "
        "retries with backoff plus a circuit breaker keep a transient outage "
        "from aborting the whole task."
    ),
    "write:*": (
        "Summary: An async pipeline decomposes the request into a "
        "retrieve -> analyze -> write DAG, batches LLM calls by size or time "
        "window, streams partial results as they land, and degrades "
        "gracefully when a provider stumbles."
    ),
}


def build_orchestrator(
    *,
    fail: bool = False,
    failure_type: str = "transient",
    max_batch_size: int = 8,
    max_wait_ms: int = 50,
) -> tuple[Orchestrator, StreamBus, Batcher]:
    """Construct and wire every component, returning the runnable pieces.

    Args:
        fail: When True, configure the :class:`MockProvider` to fail on the
            analyzer's prompts (``fail_on={"analyze:*"}``) to reproduce the
            failure path (Requirement 7.6).
        failure_type: ``"transient"`` (default) or ``"permanent"``. A transient
            failure is retried then falls back to a degraded completion so the
            analyzer step ends DEGRADED (Requirement 8.6); a permanent failure
            propagates immediately so the analyzer step ends FAILED.
        max_batch_size: Maximum requests per flushed batch for the batcher.
        max_wait_ms: Time window (ms) the first queued request waits before a
            partial batch is flushed.

    Returns:
        A ``(orchestrator, stream, batcher)`` tuple. The caller owns the
        batcher's lifecycle (start happens lazily on first submit; stop must be
        awaited) and the stream's lifecycle (close after the run).
    """
    error_type = PermanentError if failure_type == "permanent" else TransientError
    fail_on = {"analyze:*": error_type} if fail else None

    provider = MockProvider(responses=DEFAULT_RESPONSES, fail_on=fail_on)

    # Failure layer: small delays keep the demo snappy while still exercising
    # the retry/backoff path before the degraded fallback kicks in.
    failure = FailureHandler(
        max_retries=2,
        base_delay_ms=10,
        breaker_threshold=5,
    )

    batcher = Batcher(
        provider,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        failure=failure,
    )

    stream = StreamBus()

    orchestrator = Orchestrator(
        planner=Planner(provider),
        dispatcher=Dispatcher(
            agents=[
                RetrieverAgent(batcher),
                AnalyzerAgent(batcher),
                WriterAgent(batcher),
            ]
        ),
        stream=stream,
    )
    return orchestrator, stream, batcher


def _format_event(event: StreamEvent) -> str:
    """Render a :class:`StreamEvent` as a single readable console line.

    PARTIAL events are tagged ``[DEGRADED]`` or ``[FAILED]`` distinctly so the
    degraded/failed outcomes are visible at a glance (Requirements 5.4, 8.6).
    """
    kind = event.kind

    if kind is StreamEventKind.STARTED:
        task_id = (event.payload or {}).get("task_id", "?")
        return f"  >> STARTED   task={task_id}"

    if kind is StreamEventKind.DONE:
        task_id = (event.payload or {}).get("task_id", "?")
        return f"  >> DONE      task={task_id}"

    if kind is StreamEventKind.DROPPED:
        dropped = (event.payload or {}).get("dropped", 0)
        return f"  >> DROPPED   {dropped} event(s) lost to backpressure"

    if kind is StreamEventKind.PARTIAL:
        payload = event.payload or {}
        step_id = payload.get("step_id", "?")
        result = payload.get("result")
        status = getattr(result, "status", None)
        output = getattr(result, "output", "")
        error = getattr(result, "error", None)

        if status is ResultStatus.DEGRADED:
            tag = "[DEGRADED]"
        elif status is ResultStatus.FAILED:
            tag = "[FAILED]  "
        else:
            tag = "[OK]      "

        body = output if output else (error or "")
        return f"  -- PARTIAL {tag} step={step_id}: {body}"

    # ERROR or any other kind.
    return f"  -- {kind.value.upper()}: {event.payload}"


async def _consume(stream: StreamBus) -> None:
    """Print every event from the bus until it is closed (the consumer task).

    Runs concurrently with execution so partial outputs are shown as they are
    produced rather than after the whole run completes (Requirement 5.1). The
    generator returns cleanly once :meth:`StreamBus.close` enqueues its
    sentinel.
    """
    async for event in stream.subscribe():
        print(_format_event(event))


def _print_header(task_text: str, *, fail: bool, failure_type: str) -> None:
    """Print a short banner explaining the mode and the request being run."""
    mode = "FAILURE" if fail else "HAPPY"
    print("=" * 72)
    print(f" Agentic AI System -- end-to-end demo  [mode: {mode} PATH]")
    if fail:
        print(
            f" Analyzer prompts are configured to fail ({failure_type}); "
            "watch for a degraded/failed chunk."
        )
    else:
        print(" All steps should complete; partial outputs stream as they land.")
    print(f" Task: {task_text}")
    print(" Provider: MockProvider (deterministic, no API key required)")
    print("=" * 72)
    print(" Streaming events (consumed concurrently with execution):")


def _print_summary(task: Task) -> None:
    """Print a per-step status summary after the run terminates."""
    print("-" * 72)
    print(f" Final summary for task {task.id!r} (terminal={task.is_terminal()}):")
    for step in task.steps:
        status = step.status.value.upper()
        deps = ", ".join(step.depends_on) if step.depends_on else "-"
        print(
            f"   step {step.id} [{step.agent_type:<9}] "
            f"-> {status:<9} (depends_on: {deps})"
        )
    print("=" * 72)


async def run_demo(
    task_text: str = DEFAULT_TASK_TEXT,
    *,
    fail: bool = False,
    failure_type: str = "transient",
) -> Task:
    """Run one end-to-end demo and return the final, terminal :class:`Task`.

    Wires the components, starts a concurrent stream consumer, runs the
    orchestrator, then closes the stream, awaits the consumer to drain, and
    asserts the resulting Task is terminal.

    Args:
        task_text: The free-text request to decompose and execute.
        fail: When True, reproduce the failure path (analyzer fails).
        failure_type: ``"transient"`` or ``"permanent"`` (only used with
            ``fail=True``).

    Returns:
        The final :class:`Task` with every step in a terminal status.
    """
    _print_header(task_text, fail=fail, failure_type=failure_type)

    orchestrator, stream, batcher = build_orchestrator(
        fail=fail, failure_type=failure_type
    )

    consumer = asyncio.create_task(_consume(stream))
    try:
        task = await orchestrator.run(task_text)
    finally:
        # Close the stream so the consumer drains and returns, then ensure the
        # batcher's background loop is stopped and every future is resolved.
        stream.close()
        await consumer
        await batcher.stop()

    # The orchestrator guarantees termination; assert it for the demo's sake.
    assert task.is_terminal(), "run completed but the task is not terminal"

    _print_summary(task)
    return task


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the demo entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m agentic.cli",
        description=(
            "Run the Agentic AI System end-to-end with a deterministic, "
            "key-free MockProvider. Streams partial outputs as steps finish."
        ),
    )
    parser.add_argument(
        "task_text",
        nargs="?",
        default=DEFAULT_TASK_TEXT,
        help="The complex, multi-part request to run (default: a canned example).",
    )
    parser.add_argument(
        "--fail",
        "--fail-mode",
        dest="fail",
        action="store_true",
        help=(
            "Reproduce the failure path: the MockProvider fails on the "
            "analyzer's prompts (fail_on={'analyze:*'})."
        ),
    )
    parser.add_argument(
        "--failure-type",
        choices=["transient", "permanent"],
        default="transient",
        help=(
            "Kind of analyzer failure to inject when --fail is set. "
            "'transient' (default) degrades via the batcher fallback; "
            "'permanent' fails immediately without retry."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Console entry point: parse args, run the demo, return an exit code."""
    args = _parse_args(argv)
    asyncio.run(
        run_demo(
            args.task_text,
            fail=args.fail,
            failure_type=args.failure_type,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
