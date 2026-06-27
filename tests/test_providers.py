"""Tests for the provider-agnostic LLM abstraction (``agentic.providers``).

This module covers two task sub-items from the implementation plan:

* **Task 3.3 -- Property 10: Determinism of MockProvider** (Validates: Req 7.4)
  plus input/output order preservation for the provider contract (supports
  Property 3 / Requirement 7.2). These are encoded as Hypothesis
  property-based tests.
* **Task 3.4 -- MockProvider ``fail_on`` and error typing** (Requirements 7.3,
  7.6). These are example-based unit tests covering collection vs. mapping
  ``fail_on`` configuration, glob-pattern matching, and deterministic
  resolution of non-failing prompts.

``MockProvider.complete`` is a coroutine. The property tests drive it with
``asyncio.run`` inside synchronous ``@given`` functions (the pattern Hypothesis
recommends, avoiding event-loop reuse across generated examples), while the
plain unit tests are ``async def`` and rely on ``asyncio_mode = auto``.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentic.errors import PermanentError, TransientError
from agentic.providers import Completion, MockProvider

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Arbitrary prompt text. ``text`` covers unicode, empty strings, whitespace,
# and characters such as ``*``/``?``/``[`` that are significant to fnmatch, so
# the determinism guarantee is exercised across the synthetic-fallback path as
# well as exact/glob resolution.
prompt_strategy = st.text(max_size=40)
prompt_lists = st.lists(prompt_strategy, max_size=12)

# A fixed, representative ``responses`` map exercising exact + glob resolution.
_RESPONSES = {
    "retrieve:*": "sources...",
    "analyze:*": "findings...",
    "write:*": "final report...",
    "exact-prompt": "exact-value",
}


# ---------------------------------------------------------------------------
# Task 3.3 -- Property 10: Determinism + order preservation
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(prompts=prompt_lists)
def test_property10_determinism_across_calls_and_instances(prompts: list[str]) -> None:
    """Property 10: identical output across repeated calls and instances.

    For any prompt list ``p``, ``MockProvider.complete(p)`` returns identical
    completion text both when called twice on the same instance and when called
    on a separately constructed instance with the same configuration.

    **Validates: Requirements 7.4**
    """
    provider_a = MockProvider(responses=_RESPONSES)
    provider_b = MockProvider(responses=_RESPONSES)

    first = asyncio.run(provider_a.complete(prompts))
    second = asyncio.run(provider_a.complete(prompts))
    other_instance = asyncio.run(provider_b.complete(prompts))

    first_text = [c.text for c in first]
    assert [c.text for c in second] == first_text
    assert [c.text for c in other_instance] == first_text


@settings(max_examples=200)
@given(prompts=prompt_lists)
def test_property3_order_and_length_preserved(prompts: list[str]) -> None:
    """Provider contract: output order == input order, lengths match.

    Each returned :class:`Completion` corresponds positionally to its input
    prompt (``completion[i].prompt == prompts[i]``), and the output length
    equals the input length.

    **Validates: Requirements 7.2** (supports Property 3)
    """
    provider = MockProvider(responses=_RESPONSES)

    out = asyncio.run(provider.complete(prompts))

    assert len(out) == len(prompts)
    assert [c.prompt for c in out] == prompts
    for completion in out:
        assert isinstance(completion, Completion)
        assert isinstance(completion.text, str)


@settings(max_examples=100)
@given(prompts=prompt_lists)
def test_property10_determinism_with_empty_responses(prompts: list[str]) -> None:
    """Determinism also holds when every prompt uses the synthetic fallback.

    With no ``responses`` configured, completions are derived from a stable
    SHA-256 digest of the prompt, so repeated calls remain identical.

    **Validates: Requirements 7.4**
    """
    provider = MockProvider()

    first = asyncio.run(provider.complete(prompts))
    second = asyncio.run(provider.complete(prompts))

    assert [c.text for c in first] == [c.text for c in second]
    assert [c.prompt for c in first] == prompts


# ---------------------------------------------------------------------------
# Task 3.4 -- fail_on configuration and error typing (Requirements 7.3, 7.6)
# ---------------------------------------------------------------------------


async def test_fail_on_collection_raises_default_transient_error() -> None:
    """A collection ``fail_on`` raises the default TransientError on a match.

    **Validates: Requirements 7.3, 7.6**
    """
    provider = MockProvider(responses={}, fail_on={"boom"})

    with pytest.raises(TransientError):
        await provider.complete(["boom"])


async def test_fail_on_mapping_raises_specified_permanent_error() -> None:
    """A mapping ``fail_on`` raises the per-pattern error type.

    **Validates: Requirements 7.3, 7.6**
    """
    provider = MockProvider(responses={}, fail_on={"analyze:*": PermanentError})

    with pytest.raises(PermanentError):
        await provider.complete(["analyze:tradeoffs"])


async def test_fail_on_mapping_distinguishes_transient_and_permanent() -> None:
    """Distinct patterns surface distinct error types from one provider.

    **Validates: Requirements 7.3, 7.6**
    """
    provider = MockProvider(
        responses={},
        fail_on={"perm:*": PermanentError, "trans:*": TransientError},
    )

    with pytest.raises(PermanentError):
        await provider.complete(["perm:auth"])
    with pytest.raises(TransientError):
        await provider.complete(["trans:timeout"])


async def test_fail_on_glob_pattern_matches() -> None:
    """Glob patterns (e.g. ``analyze:*``) match prompts via fnmatch.

    **Validates: Requirements 7.6**
    """
    provider = MockProvider(responses={}, fail_on={"analyze:*"})

    # Matches the glob.
    with pytest.raises(TransientError):
        await provider.complete(["analyze:the-data"])

    # Does not match the glob -> resolves normally.
    out = await provider.complete(["retrieve:sources"])
    assert len(out) == 1
    assert isinstance(out[0].text, str)


async def test_custom_default_error_for_collection_fail_on() -> None:
    """A collection ``fail_on`` honors a custom default ``error`` type.

    **Validates: Requirements 7.3, 7.6**
    """
    provider = MockProvider(responses={}, fail_on=["nope"], error=PermanentError)

    with pytest.raises(PermanentError):
        await provider.complete(["nope"])


async def test_non_failing_prompts_resolve_deterministically() -> None:
    """Non-matching prompts still complete, and do so deterministically.

    Even when ``fail_on`` is configured, prompts that do not match resolve to
    their configured/synthetic completion, and repeated calls are identical.

    **Validates: Requirements 7.3, 7.4, 7.6**
    """
    provider = MockProvider(
        responses={"analyze:*": "findings..."},
        fail_on={"retrieve:*": TransientError},
    )

    prompts = ["analyze:x", "write:y", "analyze:z"]
    first = await provider.complete(prompts)
    second = await provider.complete(prompts)

    # Order/length preserved and resolution is deterministic.
    assert [c.text for c in first] == [c.text for c in second]
    assert [c.prompt for c in first] == prompts
    # Glob-configured response is used for matching prompts.
    assert first[0].text == "findings..."
    assert first[2].text == "findings..."


async def test_fail_on_failure_short_circuits_whole_batch() -> None:
    """If any prompt matches ``fail_on``, the whole batch call raises.

    Mirrors a real provider call where the batched request fails as a unit.

    **Validates: Requirements 7.3, 7.6**
    """
    provider = MockProvider(responses={}, fail_on={"bad"})

    with pytest.raises(TransientError):
        await provider.complete(["good-1", "bad", "good-2"])


def test_invalid_default_error_type_rejected() -> None:
    """A non-Exception ``error`` type is rejected at construction.

    **Validates: Requirements 7.3**
    """
    with pytest.raises(ValueError):
        MockProvider(responses={}, error=str)  # type: ignore[arg-type]


def test_invalid_mapping_error_type_rejected() -> None:
    """A mapping ``fail_on`` with a non-Exception value is rejected.

    **Validates: Requirements 7.3**
    """
    with pytest.raises(ValueError):
        MockProvider(responses={}, fail_on={"x": str})  # type: ignore[dict-item]
