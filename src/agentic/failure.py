"""Failure / retry layer for the Agentic AI System.

This module implements :class:`FailureHandler`, the component that wraps every
provider call with three cooperating resilience mechanisms so the system can
degrade gracefully instead of aborting a whole task on a transient outage:

* **Retry with exponential backoff + jitter** -- a :class:`~agentic.errors.TransientError`
  (timeout, rate limit, 5xx) is retried up to ``max_retries`` times, with the
  total number of attempts capped at ``max_retries + 1`` (Requirements 8.1, 8.2,
  8.3). A :class:`~agentic.errors.PermanentError` (auth failure, malformed
  request) is *never* retried and propagates immediately (Requirement 8.4).
* **Circuit breaker** -- after ``breaker_threshold`` consecutive failures the
  breaker opens and subsequent calls short-circuit straight to fallback-or-raise
  without touching the provider (Requirement 8.5). After ``breaker_cooldown_ms``
  the breaker half-opens to let a single probe call test recovery; a success
  closes it and resets the failure count, while a failure reopens it.
* **Fallback** -- when retries are exhausted (or the breaker is open) and a
  ``fallback`` is supplied, its result is returned to produce a DEGRADED outcome
  (Requirement 8.6); with no fallback the most recent error is raised
  (Requirement 8.7).

Testability
-----------
The wall clock, the sleep coroutine, and the jitter source are all injectable
constructor parameters (defaulting to :func:`time.monotonic`, :func:`asyncio.sleep`,
and a uniform random jitter). This lets unit and property tests drive the
breaker cooldown and backoff growth deterministically without real delays. A
*monotonic* clock is used for the breaker cooldown so it is immune to wall-clock
adjustments.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7.
"""

from __future__ import annotations

import asyncio
import random
import time
from enum import Enum
from typing import Awaitable, Callable, Optional, TypeVar

from .errors import CircuitOpenError, PermanentError, TransientError

__all__ = [
    "BreakerState",
    "FailureHandler",
]

T = TypeVar("T")

#: An async operation: a zero-argument callable returning an awaitable.
Op = Callable[[], Awaitable[T]]
#: A jitter function mapping the exponential delay (ms) to an extra delay (ms).
Jitter = Callable[[float], float]
#: A monotonic clock returning seconds.
Clock = Callable[[], float]
#: A sleep coroutine accepting a delay in seconds.
Sleep = Callable[[float], Awaitable[None]]

#: Default breaker cooldown (milliseconds) before an open breaker half-opens.
DEFAULT_BREAKER_COOLDOWN_MS = 30_000


class BreakerState(str, Enum):
    """The state of the circuit breaker.

    Subclassing ``str`` keeps the value readable and JSON-friendly while still
    providing a closed enumeration. Transitions:

    * ``CLOSED`` -> ``OPEN`` once ``breaker_threshold`` consecutive failures
      accumulate.
    * ``OPEN`` -> ``HALF_OPEN`` once ``breaker_cooldown_ms`` has elapsed since
      the breaker opened (a single probe call is then allowed).
    * ``HALF_OPEN`` -> ``CLOSED`` on a probe success (failure count resets), or
      ``HALF_OPEN`` -> ``OPEN`` on a probe failure (cooldown restarts).
    """

    #: Normal operation: calls flow through to the operation.
    CLOSED = "closed"
    #: Tripped: calls short-circuit to fallback-or-raise without calling op.
    OPEN = "open"
    #: Probing: a single call is permitted to test whether the provider recovered.
    HALF_OPEN = "half_open"


class FailureHandler:
    """Wrap provider calls with retry/backoff, a circuit breaker, and fallback.

    Args:
        max_retries: Maximum number of *retries* after the initial attempt for a
            :class:`~agentic.errors.TransientError`. Total attempts are capped at
            ``max_retries + 1``. Must be ``>= 0``.
        base_delay_ms: Base backoff delay in milliseconds. The exponential
            component for retry ``n`` (1-based) is ``base_delay_ms * 2**(n-1)``.
            Must be ``>= 0`` (``0`` disables waiting).
        breaker_threshold: Number of consecutive failures that opens the
            circuit breaker. Must be ``>= 1``.
        breaker_cooldown_ms: Time in milliseconds an open breaker waits before
            half-opening to probe recovery. Must be ``>= 0``.
        clock: Monotonic clock returning seconds, injectable for tests.
            Defaults to :func:`time.monotonic`.
        sleep: Awaitable sleep accepting seconds, injectable for tests.
            Defaults to :func:`asyncio.sleep`.
        jitter: Callable mapping the exponential delay (ms) to an additional
            jitter delay (ms). Defaults to ``random.uniform(0, delay)`` (full
            jitter), which spreads retry storms across a recovering provider.
    """

    def __init__(
        self,
        max_retries: int,
        base_delay_ms: int,
        breaker_threshold: int,
        breaker_cooldown_ms: int = DEFAULT_BREAKER_COOLDOWN_MS,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        jitter: Optional[Jitter] = None,
    ) -> None:
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be an integer >= 0")
        if not isinstance(base_delay_ms, (int, float)) or base_delay_ms < 0:
            raise ValueError("base_delay_ms must be a number >= 0")
        if not isinstance(breaker_threshold, int) or breaker_threshold < 1:
            raise ValueError("breaker_threshold must be an integer >= 1")
        if not isinstance(breaker_cooldown_ms, (int, float)) or breaker_cooldown_ms < 0:
            raise ValueError("breaker_cooldown_ms must be a number >= 0")

        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown_ms = breaker_cooldown_ms

        self._clock = clock
        self._sleep = sleep
        self._jitter: Jitter = jitter if jitter is not None else _default_jitter

        # Circuit-breaker state.
        self._state: BreakerState = BreakerState.CLOSED
        self._consecutive_failures: int = 0
        #: Monotonic timestamp (seconds) at which the breaker last opened.
        self._opened_at: float = 0.0

    # -- Introspection (useful for tests) ----------------------------------

    @property
    def state(self) -> BreakerState:
        """The current circuit-breaker state (does not advance the cooldown)."""
        return self._state

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive failures recorded since the last success."""
        return self._consecutive_failures

    # -- Public API --------------------------------------------------------

    async def call(
        self,
        op: Op,
        fallback: Optional[Op] = None,
    ) -> T:
        """Run ``op`` with retry/backoff, breaker short-circuiting, and fallback.

        Control flow mirrors the design's Failure Handler Algorithm:

        1. If the breaker is open, do not call ``op`` -- route straight to
           fallback-or-raise with a :class:`~agentic.errors.CircuitOpenError`.
        2. Otherwise call ``op``. On success, record it (closing/​resetting the
           breaker) and return the result.
        3. A :class:`~agentic.errors.PermanentError` propagates immediately
           without retry (so ``attempts == 1``).
        4. A :class:`~agentic.errors.TransientError` increments the attempt and
           failure counters. If retries are exhausted (``attempt > max_retries``)
           or the breaker has since opened, route to fallback-or-raise; else wait
           for the backoff delay and retry.

        Args:
            op: The async operation to execute (raises ``TransientError`` or
                ``PermanentError`` on failure).
            fallback: Optional async operation producing a degraded result when
                the primary path cannot succeed.

        Returns:
            The result of ``op`` on success, or of ``fallback`` when the primary
            path is exhausted/short-circuited and a fallback was provided.

        Raises:
            CircuitOpenError: The breaker is open and no fallback was provided.
            TransientError: Retries were exhausted and no fallback was provided.
            PermanentError: ``op`` raised a permanent error (never retried).
        """
        # 1. Breaker check BEFORE invoking op (Requirement 8.5).
        if self._breaker_open():
            return await self._use_fallback_or_raise(fallback, CircuitOpenError())

        attempt = 0
        while True:
            try:
                result = await op()
            except PermanentError:
                # Never retry permanent errors (Requirement 8.4): propagate so
                # the caller fails fast with attempts == 1.
                self._record_failure()
                raise
            except TransientError as exc:
                attempt += 1
                self._record_failure()
                # Retry bound (Requirements 8.1, 8.2): total attempts are capped
                # at max_retries + 1. Also bail out if this failure tripped the
                # breaker mid-loop.
                if attempt > self.max_retries or self._breaker_open():
                    return await self._use_fallback_or_raise(fallback, exc)
                # Exponential backoff with jitter before the next attempt
                # (Requirement 8.3).
                await self._sleep(self._backoff_seconds(attempt))
                continue
            else:
                self._record_success()
                return result

    # -- Circuit breaker ---------------------------------------------------

    def _breaker_open(self) -> bool:
        """Return True if calls should short-circuit, advancing the cooldown.

        When the breaker is ``OPEN`` and the cooldown has elapsed, this
        transitions it to ``HALF_OPEN`` and returns ``False`` so exactly one
        probe call is admitted. While ``HALF_OPEN`` or ``CLOSED`` it returns
        ``False`` (calls are allowed).
        """
        if self._state is BreakerState.OPEN:
            elapsed_ms = (self._clock() - self._opened_at) * 1000.0
            if elapsed_ms >= self.breaker_cooldown_ms:
                # Cooldown elapsed -> admit a single probe call.
                self._state = BreakerState.HALF_OPEN
                return False
            return True
        return False

    def _record_success(self) -> None:
        """Reset the failure count and close the breaker after a success."""
        self._consecutive_failures = 0
        self._state = BreakerState.CLOSED

    def _record_failure(self) -> None:
        """Account for a failure, opening the breaker once the threshold is hit.

        A failure while ``HALF_OPEN`` reopens the breaker immediately (the probe
        failed), restarting the cooldown.
        """
        self._consecutive_failures += 1
        if (
            self._state is BreakerState.HALF_OPEN
            or self._consecutive_failures >= self.breaker_threshold
        ):
            self._state = BreakerState.OPEN
            self._opened_at = self._clock()

    # -- Helpers -----------------------------------------------------------

    def _backoff_seconds(self, attempt: int) -> float:
        """Compute the backoff delay (seconds) for a 1-based retry number.

        The exponential component is ``base_delay_ms * 2**(attempt-1)``; a
        jitter term is added on top. The deterministic component is
        non-decreasing in ``attempt`` (Requirement 8.3); only the jitter varies.
        """
        exponential_ms = self.base_delay_ms * (2 ** (attempt - 1))
        delay_ms = exponential_ms + self._jitter(float(exponential_ms))
        return max(0.0, delay_ms) / 1000.0

    async def _use_fallback_or_raise(self, fallback: Optional[Op], error: Exception) -> T:
        """Return the fallback's result if provided, else raise ``error``.

        Implements the DEGRADED-vs-raise decision shared by the breaker-open and
        retries-exhausted paths (Requirements 8.6, 8.7).
        """
        if fallback is not None:
            return await fallback()
        raise error


def _default_jitter(exponential_ms: float) -> float:
    """Full-jitter default: a uniform random delay in ``[0, exponential_ms]``."""
    if exponential_ms <= 0:
        return 0.0
    return random.uniform(0.0, exponential_ms)
