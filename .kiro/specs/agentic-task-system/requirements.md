# Requirements Document

## Introduction

This document specifies the requirements for an Agentic AI System that accepts a complex, multi-part task, decomposes it into an ordered set of discrete steps (a DAG), and executes those steps using specialized agents (Retriever, Analyzer, Writer) coordinated by an async pipeline. The system streams partial outputs to the user as they are produced and degrades gracefully when individual steps fail, using retry-with-backoff, a circuit breaker, and fallback logic.

The engine is built from first principles on Python `asyncio` and deliberately avoids black-box agent frameworks (LangChain, CrewAI, AutoGen) so that the manual batching logic and the failure-handling control flow are fully visible and testable. All LLM access is hidden behind a provider-agnostic abstraction, with a deterministic mock provider enabling key-free, reproducible runs of both the happy path and the failure path.

These requirements are derived from the approved design document and are written to be consistent with the technical approach defined there.

## Glossary

- **System**: The complete Agentic AI System, including the orchestrator, planner, agents, batcher, streaming layer, failure layer, and provider abstraction.
- **Orchestrator**: The component that owns the end-to-end lifecycle of a task run, drives DAG execution, and feeds the streaming layer.
- **Planner**: The component that converts free-text task input into a Task containing a DAG of Steps.
- **Dispatcher**: The component that selects ready Steps and routes each to the Agent matching its agent type.
- **Agent**: A component that executes a Step's work; specializations are Retriever, Analyzer, and Writer.
- **Batcher**: The component that groups individual agent LLM requests into batches by size or time window and flushes them as a single provider call.
- **Streaming_Layer**: The component (StreamBus) that delivers partial outputs and lifecycle events to the user.
- **Failure_Handler**: The component that wraps provider calls with retry-with-backoff, a circuit breaker, and fallback logic.
- **LLM_Provider**: The abstract interface decoupling the engine from any specific LLM vendor.
- **Mock_Provider**: A deterministic, key-free implementation of LLM_Provider used for reproducible testing.
- **OpenAI_Provider**: An implementation of LLM_Provider that wraps an OpenAI-style async client.
- **Task**: The unit of work containing the original request text and the Steps forming a DAG.
- **Step**: A single node in the Task DAG, with an id, description, agent type, dependencies, and status.
- **DAG**: Directed Acyclic Graph; the structure formed by Steps and their dependency edges (a topological ordering exists).
- **AgentResult**: The structured result produced by an Agent for a Step, carrying a status, output, and optional error.
- **BatchRequest**: A single LLM request submitted by an Agent to the Batcher, carrying a prompt and a future.
- **Transient_Error**: A provider failure that may succeed on retry (timeout, rate limit, 5xx).
- **Permanent_Error**: A provider failure that will not succeed on retry (auth failure, malformed request).
- **Terminal_Status**: A Step status of COMPLETED, DEGRADED, or FAILED.
- **Circuit_Breaker**: The mechanism that short-circuits provider calls after repeated consecutive failures.

## Requirements

### Requirement 1: Accept Complex Multi-Part Task Input

**User Story:** As a user, I want to submit a complex, multi-part task as a single free-text request, so that I can have the system handle the entire request without manually splitting it.

#### Acceptance Criteria

1. WHEN a user submits a non-empty task text, THE Orchestrator SHALL begin a task run for that text.
2. IF a user submits an empty or whitespace-only task text, THEN THE Orchestrator SHALL reject the input and SHALL NOT begin a task run.
3. WHEN a task run begins, THE System SHALL retain the original task text in the resulting Task.

### Requirement 2: Decompose Task Into an Ordered DAG of Steps

**User Story:** As a user, I want my task automatically decomposed into discrete, ordered steps, so that the system can execute the work in the correct sequence.

#### Acceptance Criteria

1. WHEN the Planner decomposes a non-empty task text, THE Planner SHALL produce a Task containing at least one Step.
2. WHEN the Planner produces a Task, THE Planner SHALL ensure the Steps form a DAG for which a topological ordering exists.
3. WHEN the Planner produces a Task, THE Planner SHALL ensure at least one Step has no dependencies.
4. WHEN the Planner assigns a Step, THE Planner SHALL assign an agent type that exists in the agent registry.
5. WHERE a Step declares dependencies, THE Planner SHALL ensure every declared dependency references an existing Step within the same Task.

### Requirement 3: Route Steps to Specialized Agents

**User Story:** As a user, I want each step handled by an agent specialized for that kind of work, so that retrieval, analysis, and writing are each performed appropriately.

#### Acceptance Criteria

1. THE System SHALL provide a Retriever agent, an Analyzer agent, and a Writer agent.
2. WHEN the Dispatcher routes a Step, THE Dispatcher SHALL select the Agent whose agent type matches the Step's agent type.
3. WHEN an Agent executes a Step, THE Agent SHALL read the outputs of the Step's completed dependencies from the execution context.
4. WHEN an Agent completes execution of a Step, THE Agent SHALL return a well-formed AgentResult.
5. IF an Agent encounters a failure during execution, THEN THE Agent SHALL return an AgentResult with status FAILED rather than propagating an exception past its boundary.

### Requirement 4: Execute the DAG With an Async Pipeline

**User Story:** As a user, I want independent steps to run concurrently and the overall run to always finish, so that I get results quickly and the system never hangs.

#### Acceptance Criteria

1. WHEN the Orchestrator executes a Task, THE Orchestrator SHALL schedule all currently ready Steps concurrently.
2. THE Orchestrator SHALL treat a Step as ready only WHEN the Step is PENDING and every Step in its dependencies is COMPLETED.
3. WHEN executing a Task, THE Orchestrator SHALL ensure every Step reaches a Terminal_Status.
4. WHEN executing a Task, THE Orchestrator SHALL terminate the execution loop once every Step has a Terminal_Status.
5. IF no Step is ready while the Task is not terminal, THEN THE Orchestrator SHALL mark the blocked Steps as FAILED and terminate the execution loop.

### Requirement 5: Stream Partial Outputs to the User

**User Story:** As a user, I want to see partial outputs as they are produced, so that I do not have to wait for the entire task to finish before seeing progress.

#### Acceptance Criteria

1. WHEN a Step reaches a Terminal_Status, THE Orchestrator SHALL emit a partial result event to the Streaming_Layer.
2. THE Streaming_Layer SHALL deliver events for a single Task to a subscriber in the order they were emitted.
3. WHEN the Streaming_Layer publishes an event, THE Streaming_Layer SHALL NOT block the producer.
4. WHERE a result is degraded, THE Streaming_Layer SHALL mark the corresponding event as degraded so the client can render it distinctly.
5. IF the bounded event queue is full, THEN THE Streaming_Layer SHALL apply the configured backpressure policy without deadlocking the producer.

### Requirement 6: Manual Batching of LLM Requests

**User Story:** As a developer, I want LLM requests grouped into batches by explicit, visible logic, so that the system amortizes provider round-trips without using a black-box framework.

#### Acceptance Criteria

1. WHEN an Agent submits a BatchRequest, THE Batcher SHALL enqueue the request and resolve its future when the request's batch is flushed.
2. WHEN the number of accumulated requests reaches the maximum batch size, THE Batcher SHALL flush the batch.
3. WHEN the maximum wait window elapses since the first queued request of a batch, THE Batcher SHALL flush the batch even if it is not full.
4. THE Batcher SHALL ensure every flushed batch contains at least one request and at most the maximum batch size.
5. WHEN the Batcher flushes a batch, THE Batcher SHALL deliver the completion at index i to the request at index i.
6. THE Batcher SHALL eventually resolve every submitted BatchRequest future with either a result or an exception.
7. THE Batcher SHALL fix each batch's wait window by its first request and SHALL NOT extend the window when later requests arrive.

### Requirement 7: Provider-Agnostic LLM Abstraction

**User Story:** As a developer, I want all LLM access behind a single interface with a deterministic mock, so that I can swap providers without code changes and run the system without an API key.

#### Acceptance Criteria

1. THE System SHALL access all LLM completions through the LLM_Provider interface.
2. WHEN the LLM_Provider receives a list of prompts, THE LLM_Provider SHALL return completions in the same order as the input prompts.
3. THE LLM_Provider SHALL surface transient failures and permanent failures as distinct error types.
4. WHEN the Mock_Provider receives the same prompt list across runs, THE Mock_Provider SHALL return identical completions.
5. WHERE no API key is configured, THE System SHALL run end-to-end using the Mock_Provider.
6. WHERE the Mock_Provider is configured to fail on specific prompts, THE Mock_Provider SHALL raise the configured error for those prompts to reproduce the failure path.

### Requirement 8: Failure Handling and Graceful Degradation

**User Story:** As a user, I want the system to recover from provider failures and still return useful partial results, so that a transient outage does not abort my entire task.

#### Acceptance Criteria

1. IF a provider operation raises a Transient_Error, THEN THE Failure_Handler SHALL retry the operation until it succeeds or the maximum retry count is reached.
2. THE Failure_Handler SHALL limit the total number of attempts for a provider operation to the maximum retries plus one.
3. WHEN the Failure_Handler retries after a Transient_Error, THE Failure_Handler SHALL wait using exponential backoff with jitter before the next attempt.
4. IF a provider operation raises a Permanent_Error, THEN THE Failure_Handler SHALL propagate the error immediately without retrying.
5. WHEN consecutive failures reach the breaker threshold, THE Failure_Handler SHALL open the Circuit_Breaker and short-circuit subsequent calls.
6. IF retries are exhausted AND a fallback is available, THEN THE Failure_Handler SHALL invoke the fallback and produce an AgentResult with status DEGRADED and degraded set to true.
7. IF retries are exhausted AND no fallback is available, THEN THE Failure_Handler SHALL raise the most recent error.

### Requirement 9: Well-Formed Result Invariants

**User Story:** As a developer, I want results to obey consistent invariants, so that downstream components and the client can rely on their structure.

#### Acceptance Criteria

1. IF an AgentResult has status FAILED, THEN THE AgentResult SHALL have a non-null error.
2. IF an AgentResult has status DEGRADED, THEN THE AgentResult SHALL have degraded set to true.
3. THE AgentResult SHALL always carry a string output, where an empty output is permitted only when the status is FAILED.

### Requirement 10: No Black-Box Agent Frameworks

**User Story:** As a reviewer, I want the orchestration engine implemented without black-box agent frameworks, so that the control flow demonstrates understanding of what happens under the hood.

#### Acceptance Criteria

1. THE System SHALL implement orchestration, planning, dispatch, batching, streaming, and failure handling directly on the Python asyncio standard library.
2. THE System SHALL NOT depend on LangChain, CrewAI, or AutoGen for the core engine.
3. WHERE a third-party LLM client is used, THE System SHALL confine that dependency to an implementation of the LLM_Provider interface.
