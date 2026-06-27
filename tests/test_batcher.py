"""Tests for :class:`agentic.batcher.Batcher`.

Covers the property-based correctness properties and the unit-level
trigger/index-mapping checks from the design's *Correctness Properties* and
*Testing Strategy* sections:

* Property 1 -- Batch size bound (task 5.2, Requirements 6.2, 6.4).
* Property 2 -- No lost requests (task 5.3, Requirements 6.1, 6.6).
* Property 3 -- Order preservation in batches (task 5.4, Requirements 6.5, 7.2).
* Property 4 -- Time-window guarantee (task 5.5, Requirements 6.3, 6.7).
* Unit tests for the size trigger, the time-window trigger, and index mapping
  (task 5.6, Requirements 6.2, 6.3, 6.5).

Property tests use Hypothesis. Each generated example runs its async scenario
through :func:`asyncio.run` so the Batcher's queue/loop bind to a fresh event
loop, and ``deadline`` is disabled because the scenarios rely on real (small)
``asyncio`` timing windows that would otherwise be flaky under Hypothesis's
per-example deadline.
"""

from __future__ import annotations

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentic.batcher import Batcher
from agentic.providers import Completion, LLMProvider


class RecordingProvider(LLMProvider):
    """An :class:`LLMProvider` that records each flushed batch it receives.

    Echoes every prompt straight back as a :class:`Completion` whose ``text``
    and ``prompt`` both equal the input prompt. This makes the
    request->response correspondence trivially checkable: a request that
    receives a completion whose ``prompt`` equals its own prompt was mapped to
    the right index (Requirement 6.5).

    The event loop is single-threaded, so appending to ``batches`` from inside
    ``complete`` needs no locking.
    """

    def __init__(self) -> None:
        #: One entry per flushed batch: the exact ``prompts`` list received.
        self.batches: list[list[str]] = []
        #: Running total of prompts seen across all flushes.
        self.total_prompts: int = 0

    async def complete(self, prompts: list[str]) -> list[Completion]:
        self.batches.append(list(prompts))
        self.total_prompts += len(prompts)
        return [Completion(text=p, prompt=p) for p in prompts]


async def _submit_all(batcher: Batcher, prompts: list[str]) -> list[Completion]:
    """Submit every prompt concurrently and return results in submit order.

    :func:`asyncio.gather` preserves input order in its results, so
    ``result[i]`` corresponds to ``prompts[i]``.
    """
    return await asyncio.gather(*(batcher.submit(p) for p in prompts))


# Reusable strategies. Prompts are non-empty (BatchRequest rejects empty
# prompts) and unique so each submit maps to a distinguishable completion.
_prompts = st.lists(
    st.text(min_size=1, max_size=12),
    min_size=1,
    max_size=20,
    unique=True,
)
_batch_size = st.integers(min_value=1, max_value=8)
_wait_ms = st.integers(min_value=5, max_value=40)

_PROPERTY_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------
# Task 5.2 -- Property 1: Batch size bound.
# Validates: Requirements 6.2, 6.4
# --------------------------------------------------------------------------
@_PROPERTY_SETTINGS
@given(prompts=_prompts, max_batch_size=_batch_size, max_wait_ms=_wait_ms)
def test_property_batch_size_bound(prompts, max_batch_size, max_wait_ms):
    """Every flushed batch holds between 1 and ``max_batch_size`` requests."""

    async def scenario() -> list[list[str]]:
        provider = RecordingProvider()
        batcher = Batcher(provider, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
        batcher.start()
        try:
            await _submit_all(batcher, prompts)
        finally:
            await batcher.stop()
        return provider.batches

    batches = asyncio.run(scenario())

    assert batches, "expected at least one flushed batch"
    for batch in batches:
        assert 1 <= len(batch) <= max_batch_size


# --------------------------------------------------------------------------
# Task 5.3 -- Property 2: No lost requests.
# Validates: Requirements 6.1, 6.6
# --------------------------------------------------------------------------
@_PROPERTY_SETTINGS
@given(prompts=_prompts, max_batch_size=_batch_size, max_wait_ms=_wait_ms)
def test_property_no_lost_requests(prompts, max_batch_size, max_wait_ms):
    """Every submit resolves and the flushed prompt count equals N."""

    async def scenario() -> tuple[list[Completion], int]:
        provider = RecordingProvider()
        batcher = Batcher(provider, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
        batcher.start()
        try:
            results = await _submit_all(batcher, prompts)
        finally:
            await batcher.stop()
        return results, provider.total_prompts

    results, total_flushed = asyncio.run(scenario())

    assert len(results) == len(prompts)
    assert all(isinstance(c, Completion) for c in results)
    assert total_flushed == len(prompts)


# --------------------------------------------------------------------------
# Task 5.4 -- Property 3: Order preservation in batches.
# Validates: Requirements 6.5, 7.2
# --------------------------------------------------------------------------
@_PROPERTY_SETTINGS
@given(prompts=_prompts, max_batch_size=_batch_size, max_wait_ms=_wait_ms)
def test_property_order_preservation(prompts, max_batch_size, max_wait_ms):
    """Each submit receives the completion for its own prompt (index mapping)."""

    async def scenario() -> list[Completion]:
        provider = RecordingProvider()
        batcher = Batcher(provider, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
        batcher.start()
        try:
            results = await _submit_all(batcher, prompts)
        finally:
            await batcher.stop()
        return results

    results = asyncio.run(scenario())

    assert len(results) == len(prompts)
    for prompt, completion in zip(prompts, results):
        assert completion.prompt == prompt
        assert completion.text == prompt


# --------------------------------------------------------------------------
# Task 5.5 -- Property 4: Time-window guarantee (no starvation).
# Validates: Requirements 6.3, 6.7
# --------------------------------------------------------------------------
@_PROPERTY_SETTINGS
@given(
    max_batch_size=st.integers(min_value=2, max_value=8),
    max_wait_ms=st.integers(min_value=20, max_value=50),
)
def test_property_time_window_guarantee(max_batch_size, max_wait_ms):
    """A lone request (below the size limit) is still flushed on window expiry."""

    async def scenario() -> tuple[Completion, list[list[str]]]:
        provider = RecordingProvider()
        batcher = Batcher(provider, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
        batcher.start()
        try:
            # A generous timeout relative to max_wait_ms: if the window trigger
            # never fired, this lone submit would hang and wait_for would raise.
            completion = await asyncio.wait_for(batcher.submit("lonely"), timeout=2.0)
        finally:
            await batcher.stop()
        return completion, provider.batches

    completion, batches = asyncio.run(scenario())

    assert completion.prompt == "lonely"
    assert batches == [["lonely"]]


# --------------------------------------------------------------------------
# Task 5.6 -- Unit tests for size-trigger / time-window-trigger / index mapping.
# Requirements: 6.2, 6.3, 6.5
# --------------------------------------------------------------------------
async def test_size_trigger_flushes_full_batch():
    """Submitting exactly ``max_batch_size`` flushes a single full batch."""
    provider = RecordingProvider()
    # Large window so the *size* trigger (not the window) drives the flush.
    batcher = Batcher(provider, max_batch_size=4, max_wait_ms=1000)
    batcher.start()
    try:
        results = await _submit_all(batcher, ["a", "b", "c", "d"])
    finally:
        await batcher.stop()

    assert len(results) == 4
    assert len(provider.batches) == 1
    assert len(provider.batches[0]) == 4


async def test_time_window_trigger_flushes_partial_batch():
    """A partial batch is flushed when the window expires before it fills."""
    provider = RecordingProvider()
    # Room for 8 but only 2 arrive; the short window must trigger the flush.
    batcher = Batcher(provider, max_batch_size=8, max_wait_ms=20)
    batcher.start()
    try:
        results = await _submit_all(batcher, ["x", "y"])
    finally:
        await batcher.stop()

    assert len(results) == 2
    assert len(provider.batches) == 1
    assert len(provider.batches[0]) == 2


async def test_index_mapping_within_batch():
    """Within a flushed batch, each request gets the completion for its prompt."""
    provider = RecordingProvider()
    prompts = ["alpha", "beta", "gamma"]
    batcher = Batcher(provider, max_batch_size=3, max_wait_ms=1000)
    batcher.start()
    try:
        results = await _submit_all(batcher, prompts)
    finally:
        await batcher.stop()

    # Single full batch containing exactly the submitted prompts.
    assert len(provider.batches) == 1
    assert sorted(provider.batches[0]) == sorted(prompts)

    # Every submit resolved to the completion echoing its own prompt.
    for prompt, completion in zip(prompts, results):
        assert completion.prompt == prompt
        assert completion.text == prompt
