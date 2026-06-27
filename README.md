# Agentic AI System for Multi-Step Tasks

An async, framework-free agentic system that accepts a complex multi-part request,
decomposes it into an ordered DAG of steps, routes each step to a specialized agent
(Retriever → Analyzer → Writer), streams partial results as they land, batches LLM
calls with **explicit, hand-written batching logic**, and degrades gracefully when a
provider fails.

The engine is built from first principles on Python `asyncio`. It deliberately avoids
black-box agent frameworks (LangChain, CrewAI, AutoGen) so the control flow - the
manual batching loop and the retry / circuit-breaker / fallback logic - is fully
visible and testable. All LLM access sits behind a small provider interface, and a
deterministic `MockProvider` lets the whole system run and be verified **without an
API key**.

## Why no agent framework?

The task explicitly asks to "show what is happening under the hood." Every concern that
a framework would hide is implemented directly here:

- **Planning / decomposition** → `planner.py`
- **Dispatch / routing** → `dispatcher.py`
- **Manual batching** (size + time-window triggers, index-mapped responses) → `batcher.py`
- **Streaming** (bounded queue, backpressure) → `streaming.py`
- **Failure handling** (retry + exponential backoff + jitter, circuit breaker, fallback) → `failure.py`
- **Orchestration** (concurrent DAG execution, termination guarantees) → `orchestrator.py`

The only optional third-party piece is an OpenAI-style client, confined entirely to
`OpenAIProvider` behind the `LLMProvider` interface.

## Architecture

```
User → Orchestrator → Planner (free text → DAG of Steps)
                    → Dispatcher → Retriever / Analyzer / Writer agents
                                 → Manual Batcher (size / time-window flush)
                                 → Failure Layer (retry · breaker · fallback)
                                 → LLMProvider (MockProvider | OpenAIProvider)
       Streaming Layer ← partial results (streamed to the user as they land)
```

The Orchestrator owns the lifecycle. The Planner turns free text into a `Task` whose
`Step`s form a DAG. The Orchestrator repeatedly schedules **all ready steps**
(dependencies satisfied) concurrently via `asyncio.gather`, routes each through the
Dispatcher to the matching agent, and emits a partial event as each step finishes.
Agents express their LLM needs as `BatchRequest`s; the Batcher groups them and flushes
on a size or time-window trigger; each flush passes through the Failure layer.

See `.kiro/specs/agentic-task-system/design.md` for the full design document
(architecture, sequence diagrams, data models, algorithm pseudocode, and the 12
correctness properties), and `requirements.md` for the EARS requirements.

## Project layout

```
src/agentic/
  models.py        # Step, Task, AgentResult, BatchRequest, enums + validation
  errors.py        # TransientError / PermanentError / CircuitOpenError taxonomy
  providers.py     # LLMProvider, deterministic MockProvider, OpenAIProvider stub
  failure.py       # FailureHandler: retry/backoff/jitter + circuit breaker + fallback
  batcher.py       # Batcher: the manual size/time-window batching loop
  streaming.py     # StreamBus / StreamEvent: bounded, non-blocking, backpressure
  agents.py        # Agent base + ExecutionContext + Retriever/Analyzer/Writer
  dispatcher.py    # Dispatcher: route a step to the agent matching its type
  planner.py       # Planner: free text → validated DAG
  orchestrator.py  # Orchestrator: drive the DAG to termination, stream results
  cli.py           # Runnable end-to-end demo (happy path + failure path)
tests/             # unit tests + 12 Hypothesis property tests + integration tests
```

## Quick start

Requires **Python 3.11+**. No runtime dependencies; test dependencies only.

```bash
# Install test/dev dependencies (the core engine has zero runtime deps)
pip install -e ".[dev]"
```

### Run the demo (no API key needed)

```bash
# Happy path: all steps complete, partial outputs stream as they land
python -m agentic.cli

# Failure path: the analyzer's provider call fails (transient) → degrades,
# the dependent writer is blocked → FAILED, the run still terminates
python -m agentic.cli --fail

# Permanent failure variant (no retry, no degraded fallback)
python -m agentic.cli --fail --failure-type permanent

# Run your own request
python -m agentic.cli "Summarize the CAP theorem and write three takeaways."

# Pace the streamed events with a visible pause (useful for demos / screen recordings)
python -m agentic.cli --fail --slow      # ~0.8s between events
python -m agentic.cli --fail --delay 1.5 # custom pause in seconds
```

> Note: the demo uses a deterministic `MockProvider`, so it returns the same
> canned text regardless of the request wording — it exercises the *pipeline*,
> not a real model. Plug in `OpenAIProvider` for real answers.

If running from a checkout without installing, put `src` on the path:

```bash
# PowerShell
$env:PYTHONPATH="src"; python -m agentic.cli --fail
# bash
PYTHONPATH=src python -m agentic.cli --fail
```

### Sample output (failure path)

```
  >> STARTED   task=task-cd13be95611c
  -- PARTIAL [OK]       step=s1: Sources gathered: ...
  -- PARTIAL [DEGRADED] step=s2: [degraded] analyze: ...
  -- PARTIAL [FAILED]   step=s3: Step is blocked: dependencies did not complete successfully (s2)
  >> DONE      task=task-cd13be95611c
 Final summary (terminal=True):
   step s1 [retriever] -> COMPLETED
   step s2 [analyzer ] -> DEGRADED
   step s3 [writer   ] -> FAILED
```

## Tests

```bash
python -m pytest -q
```

The suite (91 tests) includes unit tests for every component, end-to-end integration
tests (happy + failure paths), and the **12 correctness properties** verified with
[Hypothesis](https://hypothesis.readthedocs.io/):

| # | Property | Component |
|---|----------|-----------|
| 1 | Batch size bound (`1 ≤ len ≤ max_batch_size`) | Batcher |
| 2 | No lost requests (every future resolved) | Batcher |
| 3 | Order preservation in batches (completion `i` → request `i`) | Batcher / Provider |
| 4 | Time-window guarantee (no starvation) | Batcher |
| 5 | DAG termination (always reaches terminal) | Orchestrator |
| 6 | Topological respect (deps COMPLETED before a step runs) | Orchestrator |
| 7 | Retry bound (`attempts ≤ max_retries + 1`) | FailureHandler |
| 8 | No retry on permanent errors | FailureHandler |
| 9 | Graceful degradation (fallback → DEGRADED) | FailureHandler |
| 10 | MockProvider determinism | Provider |
| 11 | Stream ordering | StreamBus |
| 12 | AgentResult invariant (FAILED ⇒ error) | Models |

## Using a real LLM

Swap the provider with zero changes elsewhere — same `LLMProvider` contract:

```python
from agentic.providers import OpenAIProvider
provider = OpenAIProvider(model="gpt-4o-mini")  # reads OPENAI_API_KEY from env
```

The API key is read from the environment only and never logged. Without a key the
system runs end-to-end on `MockProvider`.

## Design and post-mortem

- Design document: `.kiro/specs/agentic-task-system/design.md`
- Requirements (EARS): `.kiro/specs/agentic-task-system/requirements.md`
- Post-mortem (scaling issue, hindsight change, trade-offs): `POSTMORTEM.md`
