"""Tests for the failure / retry layer (:mod:`agentic.failure`).

Covers tasks 4.2-4.5 of the implementation plan:

* Property 7 -- Retry bound (4.2): attempts are capped at ``max_retries + 1``.
* Property 8 -- No retry on permanent errors (4.3): ``PermanentError`` is
  attempted exactly once and propagates.
* Property 9 -- Graceful degradation (4.4): an available fallback supplies the
  result when transient retries are exhausted.
* Unit tests (4.5) -- circuit-breaker open/half-open/closed transitions and
  non-decreasing deterministic backoff growth.

All tests inject a no-op (or recording) ``sleep`` and a controllable ``clock``
so the breaker cooldown and backoff growth are exercised deterministically with
no real delays. Property tests drive their async scenarios through
``asyncio.run`` so Hypothesis can replay each example on a clean event loop.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentic.errors import CircuitOpenError, PermanentError, TransientError
from agentic.failure import BreakerState, FailureHandler


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


async def _noop_sleep(_seconds: float) -> None:
    """An injectable sleep that returns immediately (no real delay)."""
    return None


class RecordingSleep:
    """An injectable sleep that records each delay it is asked to wait."""

    def __init__(self) -> None:
        self.durations: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.durations.append(seconds)


class FakeClock:
    """A controllable monotonic clock (seconds) for driving the breaker cooldown."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class CountingTransientOp:
    """An async op that always raises ``TransientError`` and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> object:
        self.calls += 1
        raise TransientError("transient boom")


class CountingPermanentOp:
    """An async op that always raises ``PermanentError`` and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> object:
        self.calls += 1
        raise PermanentError("permanent boom")


class SwitchableOp:
    """An async op that fails until ``succeed`` is set, counting its calls."""

    def __init__(self, result: str = "ok") -> None:
        self.calls = 0
        self.succeed = False
        self.result = result

    async def __call__(self) -> str:
        self.calls += 1
        if self.succeed:
            return self.result
        raise TransientError("switchable fail")


class Fallback:
    """An async fallback that returns a fixed value and counts its calls."""

    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    async def __call__(self) -> object:
        self.calls += 1
        return self.value


# ---------------------------------------------------------------------------
# Task 4.2 -- Property 7: Retry bound
# Validates: Requirements 8.1, 8.2, 8.3, 8.7
# ---------------------------------------------------------------------------


@given(max_retries=st.integers(min_value=0, max_value=6))
@settings(max_examples=50, deadline=None)
def test_property_retry_bound(max_retries: int) -> None:
    """An always-transient op (no fallback) is attempted exactly max_retries + 1
    times and then raises the most recent transient error.

    The breaker threshold is set above the attempt count so the retry bound --
    not the breaker -- is what stops the loop.

    **Validates: Requirements 8.1, 8.2, 8.3, 8.7**
    """

    async def scenario() -> int:
        op = CountingTransientOp()
        handler = FailureHandler(
            max_retries=max_retries,
            base_delay_ms=0,
            breaker_threshold=max_retries + 5,
            sleep=_noop_sleep,
        )
        with pytest.raises(TransientError):
            await handler.call(op)
        return op.calls

    assert asyncio.run(scenario()) == max_retries + 1


# ---------------------------------------------------------------------------
# Task 4.3 -- Property 8: No retry on permanent errors
# Validates: Requirements 8.4
# ---------------------------------------------------------------------------


@given(
    max_retries=st.integers(min_value=0, max_value=6),
    base_delay_ms=st.integers(min_value=0, max_value=50),
    breaker_threshold=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50, deadline=None)
def test_property_no_retry_on_permanent(
    max_retries: int, base_delay_ms: int, breaker_threshold: int
) -> None:
    """For any configuration, an op raising ``PermanentError`` is attempted
    exactly once and the error propagates without retry.

    **Validates: Requirements 8.4**
    """

    async def scenario() -> int:
        op = CountingPermanentOp()
        handler = FailureHandler(
            max_retries=max_retries,
            base_delay_ms=base_delay_ms,
            breaker_threshold=breaker_threshold,
            sleep=_noop_sleep,
        )
        with pytest.raises(PermanentError):
            await handler.call(op)
        return op.calls

    assert asyncio.run(scenario()) == 1


# ---------------------------------------------------------------------------
# Task 4.4 -- Property 9: Graceful degradation
# Validates: Requirements 8.5, 8.6, 9.2
# ---------------------------------------------------------------------------


@given(
    max_retries=st.integers(min_value=0, max_value=6),
    breaker_threshold=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50, deadline=None)
def test_property_graceful_degradation(
    max_retries: int, breaker_threshold: int
) -> None:
    """When transient retries are exhausted and a fallback is provided, ``call``
    returns the fallback's result (the DEGRADED outcome) and invokes it once.

    **Validates: Requirements 8.5, 8.6, 9.2**
    """
    sentinel = object()

    async def scenario() -> tuple[object, int, int]:
        op = CountingTransientOp()
        fallback = Fallback(sentinel)
        handler = FailureHandler(
            max_retries=max_retries,
            base_delay_ms=0,
            breaker_threshold=breaker_threshold,
            sleep=_noop_sleep,
        )
        result = await handler.call(op, fallback=fallback)
        return result, fallback.calls, op.calls

    result, fallback_calls, op_calls = asyncio.run(scenario())
    assert result is sentinel  # the fallback's value is returned
    assert fallback_calls == 1  # the fallback was invoked exactly once
    assert op_calls >= 1  # the primary op was attempted before degrading


# ---------------------------------------------------------------------------
# Task 4.5 -- Unit tests: breaker transitions and backoff growth
# Requirements: 8.3, 8.5
# ---------------------------------------------------------------------------


async def test_breaker_opens_after_threshold_and_short_circuits_without_calling_op() -> None:
    """After ``breaker_threshold`` consecutive failures the breaker opens and
    subsequent calls short-circuit to a ``CircuitOpenError`` without calling op.

    Advancing the injected clock past the cooldown half-opens the breaker to
    admit a single probe, and a successful probe closes it and resets failures.

    Requirements: 8.5
    """
    clock = FakeClock()
    op = SwitchableOp()
    handler = FailureHandler(
        max_retries=0,
        base_delay_ms=0,
        breaker_threshold=3,
        breaker_cooldown_ms=30_000,
        clock=clock,
        sleep=_noop_sleep,
    )

    # Three consecutive failing calls (max_retries=0 -> one attempt each) trip
    # the breaker on the third failure.
    for _ in range(3):
        with pytest.raises(TransientError):
            await handler.call(op)
    assert op.calls == 3
    assert handler.consecutive_failures == 3
    assert handler.state is BreakerState.OPEN

    # While open, the next call short-circuits: op is NOT invoked.
    with pytest.raises(CircuitOpenError):
        await handler.call(op)
    assert op.calls == 3  # unchanged -> provider was not touched

    # Cooldown has not elapsed yet -> still short-circuiting.
    clock.advance(10.0)  # 10_000 ms < 30_000 ms cooldown
    with pytest.raises(CircuitOpenError):
        await handler.call(op)
    assert op.calls == 3

    # Advance past the cooldown -> breaker half-opens to admit a probe.
    clock.advance(20.0)  # total 30_000 ms elapsed since opening
    assert handler._breaker_open() is False  # admits the probe
    assert handler.state is BreakerState.HALF_OPEN

    # A successful probe closes the breaker and resets the failure count.
    op.succeed = True
    result = await handler.call(op)
    assert result == "ok"
    assert op.calls == 4  # the probe call reached the provider
    assert handler.state is BreakerState.CLOSED
    assert handler.consecutive_failures == 0


async def test_open_breaker_routes_to_fallback_without_calling_op() -> None:
    """When the breaker is open and a fallback is provided, the call degrades to
    the fallback's result without invoking the primary op.

    Requirements: 8.5
    """
    clock = FakeClock()
    op = SwitchableOp()
    handler = FailureHandler(
        max_retries=0,
        base_delay_ms=0,
        breaker_threshold=2,
        breaker_cooldown_ms=30_000,
        clock=clock,
        sleep=_noop_sleep,
    )

    # Trip the breaker with two consecutive failures.
    for _ in range(2):
        with pytest.raises(TransientError):
            await handler.call(op)
    assert handler.state is BreakerState.OPEN
    assert op.calls == 2

    # With the breaker open, a fallback yields a degraded result; op untouched.
    sentinel = object()
    fallback = Fallback(sentinel)
    result = await handler.call(op, fallback=fallback)
    assert result is sentinel
    assert fallback.calls == 1
    assert op.calls == 2  # op was short-circuited


async def test_backoff_is_non_decreasing_in_attempt() -> None:
    """With jitter removed, the deterministic backoff component is
    non-decreasing in attempt number and grows as ``base * 2^(attempt-1)``.

    One sleep is recorded per retry; the final (exhausting) attempt bails out
    before sleeping.

    Requirements: 8.3
    """
    recorder = RecordingSleep()
    max_retries = 4
    base_delay_ms = 10
    handler = FailureHandler(
        max_retries=max_retries,
        base_delay_ms=base_delay_ms,
        breaker_threshold=max_retries + 5,  # keep breaker out of the way
        sleep=recorder,
        jitter=lambda _delay: 0.0,  # remove randomness
    )
    op = CountingTransientOp()

    with pytest.raises(TransientError):
        await handler.call(op)

    # max_retries sleeps (attempts 1..max_retries); attempt max_retries+1 bails.
    assert len(recorder.durations) == max_retries
    assert op.calls == max_retries + 1

    # Deterministic exponential growth: base * 2^(attempt-1), in seconds.
    expected = [base_delay_ms * (2 ** n) / 1000.0 for n in range(max_retries)]
    assert recorder.durations == pytest.approx(expected)

    # Explicitly assert the non-decreasing property (Requirement 8.3).
    assert all(
        recorder.durations[i] <= recorder.durations[i + 1]
        for i in range(len(recorder.durations) - 1)
    )
