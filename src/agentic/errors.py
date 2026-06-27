"""Error taxonomy for the Agentic AI System.

This module defines the exception hierarchy used by the LLM provider
abstraction and the failure-handling layer. The distinction between
transient and permanent failures drives the retry/backoff control flow in
``FailureHandler``:

* :class:`TransientError` is retried with exponential backoff.
* :class:`PermanentError` is propagated immediately (never retried).
* :class:`CircuitOpenError` short-circuits straight to fallback-or-raise.

Keeping these as distinct, well-documented types lets the failure layer make
correct routing decisions based purely on the exception class.

Requirements: 7.3 (providers surface transient vs. permanent failures as
distinct error types), 8.4 (permanent errors propagate without retry).
"""

from __future__ import annotations


class AgenticError(Exception):
    """Base class for all errors raised by the Agentic AI System.

    Provides a single root so callers can catch every system-specific error
    with one ``except`` clause while still distinguishing the concrete
    subclasses when finer control is needed.
    """


class TransientError(AgenticError):
    """A provider failure that may succeed if the operation is retried.

    Represents temporary conditions such as a request timeout, a rate-limit
    response, or a 5xx server error. The :class:`FailureHandler` retries
    operations that raise this error using exponential backoff with jitter,
    up to its configured maximum number of attempts.
    """


class PermanentError(AgenticError):
    """A provider failure that will not succeed on retry.

    Represents conditions such as an authentication failure or a malformed
    request, where repeating the same call would deterministically fail
    again. The :class:`FailureHandler` does NOT retry this error; it
    propagates immediately so the affected requests/steps fail fast with a
    populated error.
    """


class CircuitOpenError(AgenticError):
    """Raised when a call is attempted while the circuit breaker is open.

    This is intentionally a standalone subclass of :class:`AgenticError` and
    NOT a subclass of :class:`TransientError`. The failure-handler contract
    checks the breaker before invoking the operation and, when the breaker is
    open, routes directly to fallback-or-raise using this error. Making it a
    sibling of (rather than a) ``TransientError`` ensures it is never caught
    by transient-retry logic: an open breaker means "stop hammering the
    provider", so retrying here would defeat the breaker's purpose. Callers
    that supply a fallback receive a DEGRADED result; callers without a
    fallback see this error raised.
    """
