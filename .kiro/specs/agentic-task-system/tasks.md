# Implementation Plan: Agentic AI System for Multi-Step Tasks

## Overview

This plan implements the async, framework-free Agentic AI System described in the design,
in **Python 3.11+ with `asyncio`**. Work proceeds bottom-up: scaffolding, data models, the
provider abstraction, then the failure/batching/streaming layers, then agents/dispatch,
planner, orchestrator, and finally a runnable demo plus the property-based and integration
test suites.

Each task is incremental and builds on prior tasks, ending with everything wired together
through a runnable CLI/demo. Test sub-tasks are marked optional with `*`. Property tests use
**Hypothesis** and are placed next to the component they validate to catch errors early.

## Tasks

- [ ] 1. Project scaffolding and tooling
  - [ ] 1.1 Create package structure and test/tooling configuration
    - Create `src/agentic/__init__.py` and a `tests/` package with `tests/__init__.py`
    - Create `pyproject.toml` (project metadata, `requires-python = ">=3.11"`) and a `requirements.txt` / dev extras listing `pytest`, `pytest-asyncio`, `hypothesis`
    - Configure `pytest` with `asyncio_mode = auto` (in `pyproject.toml` or `pytest.ini`) and register the `src` layout on the path
    - Add a top-level `README`-less note in `__init__.py` exporting the package version only
    - _Requirements: 10.1, 10.2_

- [ ] 2. Data models, enums, and validation
  - [ ] 2.1 Implement enums and core dataclasses in `src/agentic/models.py`
    - Define `StepStatus`, `ResultStatus` enums
    - Define `Step`, `Task` (with `ready_steps()` and `is_terminal()`), `AgentResult`, and `BatchRequest` dataclasses per the design
    - Implement `Task.ready_steps()` to return PENDING steps whose every dependency is COMPLETED, and `Task.is_terminal()` to return True when every step is COMPLETED/DEGRADED/FAILED
    - Add a `validate()` helper (or `__post_init__` checks) enforcing: unique `Step.id`, every `depends_on` references an existing step, acyclic DAG, at least one dependency-free step, non-empty `Task.text`, and `BatchRequest.prompt` non-empty
    - Enforce `AgentResult` invariants: FAILED implies non-null `error`; DEGRADED implies `degraded == True`; `output` always a string (empty only allowed when FAILED)
    - _Requirements: 1.3, 2.1, 2.3, 2.5, 4.2, 9.1, 9.2, 9.3_

  - [ ]* 2.2 Write unit tests for model validation and helpers
    - Test `ready_steps()`/`is_terminal()` across mixed step states and the cycle/duplicate-id/dangling-dependency rejections
    - _Requirements: 2.1, 2.3, 2.5, 4.2_

  - [ ]* 2.3 Write property test for AgentResult invariant
    - **Property 12: AgentResult invariant**
    - **Validates: Requirements 9.1, 9.3, 3.5**

- [ ] 3. Provider-agnostic LLM abstraction
  - [ ] 3.1 Implement error taxonomy in `src/agentic/errors.py`
    - Define `TransientError`, `PermanentError`, and `CircuitOpenError` (subclass of `TransientError` or a distinct error per the failure-handler contract)
    - _Requirements: 7.3, 8.4_

  - [ ] 3.2 Implement provider interface, MockProvider, and OpenAIProvider stub in `src/agentic/providers.py`
    - Define `Completion` type and the abstract `LLMProvider.complete(prompts) -> list[Completion]` contract (input order == output order)
    - Implement deterministic `MockProvider(responses, fail_on=None)`: same prompt list yields identical completions; configured `fail_on` prompts raise the configured error (default `TransientError`) to reproduce the failure path
    - Implement `OpenAIProvider` stub conforming to `LLMProvider` (constructor accepts model/client; reads API key from environment only; raises a clear error if invoked without a client). Keep any third-party client confined to this class
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 10.3_

  - [ ]* 3.3 Write property test for MockProvider determinism and order
    - **Property 10: Determinism of MockProvider**
    - **Validates: Requirements 7.4**
    - Also assert input/output order preservation for the provider contract (supports Property 3 / Requirement 7.2)

  - [ ]* 3.4 Write unit tests for MockProvider fail_on and error typing
    - Verify `fail_on` raises the configured transient/permanent error and that non-failing prompts still resolve deterministically
    - _Requirements: 7.3, 7.6_

- [ ] 4. Failure / retry layer
  - [ ] 4.1 Implement `FailureHandler` in `src/agentic/failure.py`
    - Implement `call(op, fallback=None)` with: retry only on `TransientError` up to `max_retries`; total attempts capped at `max_retries + 1`; immediate propagation of `PermanentError`
    - Implement exponential backoff with jitter (`base_delay_ms * 2^(attempt-1) + jitter`) between transient retries
    - Implement the circuit breaker: track consecutive failures, open after `breaker_threshold`, short-circuit subsequent calls, and half-open after a cooldown to probe recovery
    - On exhausted retries: invoke `fallback` when available (returning its result for a DEGRADED outcome), else re-raise the most recent error
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 4.2 Write property test for retry bound
    - **Property 7: Retry bound**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.7**

  - [ ]* 4.3 Write property test for no retry on permanent errors
    - **Property 8: No retry on permanent errors**
    - **Validates: Requirements 8.4**

  - [ ]* 4.4 Write property test for graceful degradation
    - **Property 9: Graceful degradation**
    - **Validates: Requirements 8.5, 8.6, 9.2**

  - [ ]* 4.5 Write unit tests for breaker transitions and backoff growth
    - Verify breaker open/half-open transitions and that backoff delay is non-decreasing in attempt (modulo jitter); use a patched sleep/clock
    - _Requirements: 8.3, 8.5_

- [ ] 5. Manual batcher
  - [ ] 5.1 Implement `Batcher` in `src/agentic/batcher.py`
    - Implement `submit(request)` (enqueue and await the request's future) and the `_run_loop()` background consumer that blocks for the first request, fills until `max_batch_size` or until `max_wait_ms` elapses since the first request, then flushes
    - Implement `_flush(batch)`: build prompts, call through `FailureHandler.call` with a `_degraded_completions` fallback, map completion `i` to request `i`, and set futures (result or exception) so every future is eventually resolved
    - Fix each batch window by its first request (never extended by later arrivals); guarantee `1 <= len(batch) <= max_batch_size`; provide start/stop lifecycle for the loop
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 5.2 Write property test for batch size bound
    - **Property 1: Batch size bound**
    - **Validates: Requirements 6.2, 6.4**

  - [ ]* 5.3 Write property test for no lost requests
    - **Property 2: No lost requests**
    - **Validates: Requirements 6.1, 6.6**

  - [ ]* 5.4 Write property test for order preservation in batches
    - **Property 3: Order preservation in batches**
    - **Validates: Requirements 6.5, 7.2**

  - [ ]* 5.5 Write property test for time-window guarantee
    - **Property 4: Time-window guarantee**
    - **Validates: Requirements 6.3, 6.7**

  - [ ]* 5.6 Write unit tests for size-trigger and time-window-trigger flushes
    - Verify full-batch flush, partial-batch flush on window expiry, and index mapping using a counting MockProvider and a controllable clock
    - _Requirements: 6.2, 6.3, 6.5_

- [ ] 6. Streaming layer
  - [ ] 6.1 Implement `StreamEvent` and `StreamBus` in `src/agentic/streaming.py`
    - Define `StreamEvent` (kind/payload, `degraded` flag, plus `partial(...)` and lifecycle constructors) and a `StreamBus` backed by a bounded `asyncio.Queue`
    - Implement non-blocking `emit(event)` and async `subscribe()` generator preserving per-task emission order, with a configured backpressure policy (await space or drop-oldest with a marker) that never deadlocks the producer
    - Mark degraded results on their events so the client can render them distinctly
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 6.2 Write property test for stream ordering
    - **Property 11: Stream ordering**
    - **Validates: Requirements 5.2, 5.4**

  - [ ]* 6.3 Write unit tests for backpressure policy
    - Fill the bounded queue with a slow consumer and assert the producer is never blocked/deadlocked per policy
    - _Requirements: 5.3, 5.5_

- [ ] 7. Agents and dispatcher
  - [ ] 7.1 Implement agent base class and execution context in `src/agentic/agents.py`
    - Define `ExecutionContext` (provides completed-dependency outputs by step id) and the abstract `Agent` with `agent_type` and `async execute(step, context) -> AgentResult`
    - Implement a shared helper that submits `BatchRequest`s through the injected `Batcher` and converts failures into `AgentResult(status=FAILED, error=...)` so no exception crosses the agent boundary
    - _Requirements: 3.3, 3.4, 3.5_

  - [ ] 7.2 Implement Retriever, Analyzer, and Writer agents in `src/agentic/agents.py`
    - Implement `RetrieverAgent` (`"retriever"`), `AnalyzerAgent` (`"analyzer"`), `WriterAgent` (`"writer"`), each translating its `Step` (plus upstream dependency outputs) into prompts and returning a well-formed `AgentResult`
    - _Requirements: 3.1, 3.3, 3.4, 3.5_

  - [ ] 7.3 Implement `Dispatcher` in `src/agentic/dispatcher.py`
    - Build an agent registry keyed by `agent_type`; route each step to the agent whose `agent_type` matches the step's `agent_type`; expose the registry for planner validation
    - _Requirements: 3.1, 3.2_

  - [ ]* 7.4 Write unit tests for agent results and dispatch routing
    - Verify each agent returns a well-formed result, failures become FAILED results, and dispatch selects the matching agent (and errors on unknown agent types)
    - _Requirements: 3.2, 3.4, 3.5_

- [ ] 8. Planner / decomposer
  - [ ] 8.1 Implement `Planner` in `src/agentic/planner.py`
    - Implement `async decompose(task_text)` returning a `Task` with at least one step, all `agent_type`s drawn from the registry, dependencies referencing existing steps, at least one dependency-free step, and a guaranteed acyclic structure
    - Reject empty/whitespace-only input before producing a Task (supports orchestrator input rejection)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 8.2 Write unit tests for planner DAG validity
    - Verify non-empty tasks produce valid DAGs with registered agent types and a valid entry point; verify empty input is rejected
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 9. Orchestrator
  - [ ] 9.1 Implement `Orchestrator` in `src/agentic/orchestrator.py`
    - Implement `async run(task_text)`: reject empty/whitespace-only text without starting a run; otherwise call the planner, retain original text, drive `_execute_dag`, and return the final `Task`
    - Implement `_execute_dag`: loop while not terminal, schedule all ready steps concurrently with `asyncio.gather`, map results to step statuses, and emit a partial event to the `StreamBus` as each step reaches a terminal status
    - Implement the blocked-dependency branch: when no step is ready but the task is not terminal, mark blocked steps FAILED and terminate the loop
    - _Requirements: 1.1, 1.2, 1.3, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1_

  - [ ]* 9.2 Write property test for DAG termination
    - **Property 5: DAG termination**
    - **Validates: Requirements 4.3, 4.4, 4.5**

  - [ ]* 9.3 Write property test for topological respect
    - **Property 6: Topological respect**
    - **Validates: Requirements 2.2, 4.2**

  - [ ]* 9.4 Write unit tests for input rejection and blocked-dependency branch
    - Verify empty/whitespace input starts no run, and a failed dependency marks downstream steps FAILED while the run still terminates
    - _Requirements: 1.1, 1.2, 4.5_

- [ ] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Runnable CLI / demo entry point
  - [ ] 11.1 Implement end-to-end demo in `src/agentic/cli.py`
    - Wire `MockProvider` + `Batcher` + `FailureHandler` + `StreamBus` + `Planner` + `Dispatcher`(Retriever/Analyzer/Writer) + `Orchestrator` into a runnable `main()` (e.g., `python -m agentic.cli`)
    - Consume the stream concurrently with execution and print each (possibly degraded-marked) chunk; assert the final `Task` is terminal
    - Provide a flag/mode that constructs a `MockProvider` with `fail_on` to reproduce the failure path (analyzer degrades, writer still runs, degraded chunk streams), demonstrating no-API-key end-to-end operation
    - _Requirements: 5.1, 7.5, 7.6, 8.6_

- [ ] 12. Integration tests
  - [ ]* 12.1 Write happy-path integration test
    - End-to-end run with `MockProvider` asserting the final `Task` is terminal and the stream produced the expected ordered events
    - _Requirements: 4.3, 4.4, 5.1, 5.2, 7.5_

  - [ ]* 12.2 Write failure-path integration test
    - End-to-end run with `MockProvider` configured via `fail_on` asserting a DEGRADED result streams to the consumer and the run still terminates
    - _Requirements: 8.6, 5.4, 4.3_

- [ ] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP, but they encode the 12 correctness properties and the unit/integration coverage from the design's testing strategy.
- Each task references specific requirement clauses (and property numbers where applicable) for traceability.
- Checkpoints (tasks 10 and 13) ensure incremental validation.
- Property tests validate universal correctness properties; unit tests validate specific examples and edge cases; integration tests cover the happy and failure paths end-to-end.
- The core engine is implemented directly on `asyncio` with no LangChain/CrewAI/AutoGen; any third-party LLM client is confined to `OpenAIProvider` (Requirements 10.1–10.3).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["3.2", "6.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "4.1", "6.2", "6.3", "8.1"] },
    { "id": 4, "tasks": ["3.3", "3.4", "5.1", "8.2"] },
    { "id": 5, "tasks": ["4.2", "4.3", "4.4", "4.5", "5.2", "5.3", "5.4", "5.5", "5.6", "7.1"] },
    { "id": 6, "tasks": ["7.2"] },
    { "id": 7, "tasks": ["7.3"] },
    { "id": 8, "tasks": ["7.4", "9.1"] },
    { "id": 9, "tasks": ["9.2", "9.3", "9.4", "11.1"] },
    { "id": 10, "tasks": ["12.1", "12.2"] }
  ]
}
```
