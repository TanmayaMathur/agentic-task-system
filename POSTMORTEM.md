# Post-Mortem: Agentic AI System for Multi-Step Tasks

This document reflects on building the system: a scaling issue, a design change I would
make in hindsight, and two explicit trade-offs with reasoning. It is deliberately
honest about the limits of the current implementation.

## 1. Scaling issue encountered / anticipated

**The single global Batcher with one background loop is a throughput bottleneck and a
shared failure domain.**

Today every agent submits into one `Batcher` instance with a single `_run_loop`
consumer and one `FailureHandler`. This is great for clarity and for amortizing
provider round-trips, but it does not scale cleanly:

- **One flush at a time.** The loop builds a batch, then `await`s the provider call to
  resolve all futures before assembling the next batch. Under load, prompts that arrive
  during a slow flush wait for the in-flight provider call to return. Throughput is
  capped by `batch_size / provider_latency` rather than by what the provider could
  actually handle concurrently.
- **Shared circuit breaker.** Because all agents share one `FailureHandler`, a provider
  problem triggered by one agent's prompts opens the breaker for *everyone*. A localized
  issue degrades unrelated work.
- **Mixed prompt classes in one batch.** Retriever, analyzer, and writer prompts land in
  the same batch even though they may have very different size/latency profiles, which
  makes the size/time-window tuning a compromise rather than a fit.

**How I would scale it:** decouple "build batch" from "execute batch" by dispatching each
flush as its own task (a bounded pool of in-flight provider calls), so the loop keeps
forming batches while earlier ones are still resolving. Then shard batchers per provider
or per prompt class, each with its own breaker, so failures are isolated. At that point
the bounded queues and per-task step caps already in place become the load-shedding
mechanism. None of this changes the public contracts — it is an internal evolution of
the batcher.

## 2. Design change I would make in hindsight

**Make a non-COMPLETED dependency a first-class input rather than an automatic block.**

The orchestrator currently treats a step as "ready" only when **every** dependency is
`COMPLETED`. A `DEGRADED` dependency therefore blocks its dependents, and they are marked
`FAILED` by the blocked-dependency branch. You can see this in the failure demo: a
degraded analyzer causes the writer to be `FAILED` even though a degraded analysis is
often still useful input for a writer.

In hindsight I would model **dependency-satisfaction policy** explicitly per edge (or per
step): e.g. `require=COMPLETED` vs `accept=DEGRADED`. The writer could then run on
degraded analysis and itself emit a `DEGRADED` result, which is closer to the spirit of
"graceful degradation" than failing the branch. This is a small change to
`Task.ready_steps()` and the orchestrator's context-building, but it meaningfully changes
the user-visible behavior in partial-failure scenarios — which is exactly when graceful
degradation matters most. I kept the stricter semantics for this version because they are
simpler to reason about and to verify (Property 6: topological respect over COMPLETED
states), but I would revisit it for real use.

## 3. Two explicit trade-offs

### Trade-off A: Deterministic heuristic planner vs. an LLM-driven planner

**Choice:** The `Planner` produces a fixed retriever → analyzer → writer chain rather
than calling an LLM to decompose the request into a bespoke DAG.

**Reasoning:** The brief's core is the *execution machinery* — decomposition, routing,
async pipelining, streaming, manual batching, and failure handling — not the cleverness
of the decomposition itself. A deterministic planner makes the entire system **reproducible
and key-free**, which is what lets the happy path and the failure path be demonstrated and
property-tested without flakiness or an API key. The cost is that the decomposition does
not adapt to the request (a "compare three databases" task gets the same 3-step shape as
"summarize a paper"). Because the planner sits behind a clean `decompose()` contract and
already receives an `LLMProvider`, swapping in an LLM planner later is a localized change
that does not touch the orchestrator, dispatcher, batcher, or tests for those layers.

### Trade-off B: Drop-oldest streaming backpressure vs. blocking the producer

**Choice:** `StreamBus.emit()` is **synchronous and non-blocking**; when the bounded queue
is full the default policy drops the oldest event and surfaces a `DROPPED` marker, rather
than blocking the producer until the consumer catches up.

**Reasoning:** The producer is the orchestrator's execution loop. If emitting could block,
a slow or absent consumer could stall — or deadlock — task execution itself. Decoupling
delivery speed from execution speed (Requirement 5.3) means a slow client never holds up
the actual work. The trade-off is **lossy delivery under sustained backpressure**: a client
that cannot keep up will miss intermediate events (it still learns *that* loss happened via
the marker, and the final `Task` always carries complete state). For a high-fidelity audit
log you would instead want an unbounded or persistent sink; for a live progress stream,
freshness-with-a-loss-marker is the right default. The policy is configurable
(`DROP_OLDEST` / `DROP_NEWEST` / `REJECT`) so the caller can choose per use case.

## Bonus: what went well

- **Provider abstraction paid off immediately.** The deterministic `MockProvider`
  (with `fail_on`) made the failure path reproducible and let all 12 correctness
  properties run with no external service.
- **Property-based tests caught the contracts that matter** — batch size bounds, no lost
  requests, index mapping, retry bounds, topological respect, and termination — far more
  thoroughly than example tests alone would.
- **Explicit failure layering** (transient vs. permanent as distinct types, with the
  breaker as a sibling error that is never retried) kept the retry control flow readable
  and made "no retry on permanent errors" a one-line guarantee to test.
