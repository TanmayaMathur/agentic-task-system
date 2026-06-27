"""Provider-agnostic LLM abstraction for the Agentic AI System.

This module hides all LLM access behind a single interface, :class:`LLMProvider`,
so the engine never depends on a specific vendor (Requirement 7.1, 10.3). Two
implementations are provided:

* :class:`MockProvider` -- a deterministic, key-free implementation used for
  reproducible testing of both the happy path and the failure path. Given the
  same prompt list it always returns identical completions (Requirement 7.4),
  and it can be configured to raise on specific prompts to reproduce the
  failure path (Requirement 7.6).
* :class:`OpenAIProvider` -- a thin stub that conforms to the same contract and
  wraps an OpenAI-style async client. The third-party dependency is imported
  lazily *inside* the method that needs it, so this package imports cleanly even
  when the ``openai`` package is not installed (Requirement 10.3).

The provider contract is intentionally tiny: a single batched
:meth:`LLMProvider.complete` call that preserves input order (Requirement 7.2).
Transient versus permanent failures are surfaced through the distinct error
types defined in :mod:`agentic.errors` (Requirement 7.3).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Type, Union

from .errors import PermanentError, TransientError

__all__ = [
    "Completion",
    "LLMProvider",
    "MockProvider",
    "OpenAIProvider",
]


@dataclass(frozen=True)
class Completion:
    """A single completion returned by an :class:`LLMProvider`.

    Attributes:
        text: The generated completion text.
        prompt: An echo of the originating prompt. This makes the
            request/response correspondence explicit and easy to assert in
            tests; it defaults to ``None`` for providers that do not echo it.
    """

    text: str
    prompt: str | None = None


class LLMProvider(ABC):
    """Abstract interface decoupling the engine from any specific LLM vendor.

    Implementations accept a list of prompts and return a list of
    :class:`Completion` objects in the *same order* as the input prompts
    (Requirement 7.2). Failures are surfaced as :class:`TransientError` (may
    succeed on retry) or :class:`PermanentError` (will not), per Requirement
    7.3.
    """

    @abstractmethod
    async def complete(self, prompts: list[str]) -> list[Completion]:
        """Return completions for ``prompts`` in input order.

        Args:
            prompts: The prompts to complete. The completion at index ``i`` of
                the returned list corresponds to ``prompts[i]``.

        Returns:
            A list of :class:`Completion` objects of the same length as
            ``prompts`` and in the same order.

        Raises:
            TransientError: For temporary failures (timeout, rate limit, 5xx).
            PermanentError: For failures that will not succeed on retry.
        """
        raise NotImplementedError


# Type aliases documenting the accepted ``responses``/``fail_on`` shapes.
_ErrorType = Type[Exception]
_FailOn = Union["set[str]", "frozenset[str]", "list[str]", "tuple[str, ...]", Mapping[str, _ErrorType]]


class MockProvider(LLMProvider):
    """Deterministic, key-free :class:`LLMProvider` for reproducible tests.

    The provider derives each completion purely from the prompt and its static
    configuration, so calling :meth:`complete` with the same prompt list always
    yields identical completions (Requirement 7.4). No network access or API
    key is required, allowing the whole system to run end-to-end with the mock
    (Requirement 7.5).

    Response resolution for a prompt ``p`` proceeds in a fixed, deterministic
    order:

    1. **Exact match** -- if ``p`` is a key in ``responses``, its value is used.
    2. **Pattern match** -- otherwise each ``responses`` key is treated as a
       shell-style glob (``fnmatch``) and tried in sorted key order; the first
       match wins. This supports the design's ``"retrieve:*"`` / ``"analyze:*"``
       / ``"write:*"`` examples.
    3. **Synthetic fallback** -- if nothing matches, a stable completion is
       derived from ``p`` using a SHA-256 digest, so unknown prompts still map
       to a fixed, reproducible output.

    Args:
        responses: Mapping from prompt (exact string or glob pattern) to the
            completion text to return. May be empty, in which case every prompt
            uses the synthetic fallback.
        fail_on: Optional configuration of prompts that should raise instead of
            completing. It may be either:

            * a collection of prompt strings/glob patterns (``set``, ``list``,
              ``tuple``, ``frozenset``) -- matching prompts raise ``error``; or
            * a mapping from prompt string/glob pattern to an exception *type*
              (e.g. ``{"analyze:*": PermanentError}``) -- matching prompts raise
              that specific type.

            Matching uses the same exact-then-glob logic as ``responses``.
        error: The default exception type raised for ``fail_on`` entries that do
            not specify their own type (i.e. when ``fail_on`` is a plain
            collection). Defaults to :class:`TransientError` so the common case
            reproduces a retryable failure.

    Raises:
        ValueError: If ``error`` is not an ``Exception`` subclass, or if a
            mapping ``fail_on`` contains a non-``Exception`` error type.
    """

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        fail_on: _FailOn | None = None,
        error: _ErrorType = TransientError,
    ) -> None:
        if not (isinstance(error, type) and issubclass(error, Exception)):
            raise ValueError("MockProvider.error must be an Exception subclass")

        self._responses: dict[str, str] = dict(responses or {})
        self._default_error: _ErrorType = error

        # Normalize fail_on into a mapping of pattern -> error type so the
        # lookup path is uniform regardless of how it was supplied.
        self._fail_on: dict[str, _ErrorType] = {}
        if fail_on is None:
            pass
        elif isinstance(fail_on, Mapping):
            for pattern, err in fail_on.items():
                if not (isinstance(err, type) and issubclass(err, Exception)):
                    raise ValueError(
                        f"fail_on[{pattern!r}] must map to an Exception subclass"
                    )
                self._fail_on[pattern] = err
        else:
            # A plain collection of prompts/patterns -> use the default error.
            for pattern in fail_on:
                self._fail_on[pattern] = self._default_error

    async def complete(self, prompts: list[str]) -> list[Completion]:
        """Return deterministic completions in input order.

        Each prompt is resolved independently; if any prompt matches a
        ``fail_on`` entry, the configured error for that prompt is raised
        immediately (the failure surfaces for the whole batch, mirroring how a
        real provider call fails). Otherwise a :class:`Completion` is produced
        for every prompt, in order.

        Args:
            prompts: The prompts to complete.

        Returns:
            A list of :class:`Completion` objects, one per prompt, in order.

        Raises:
            Exception: The configured ``fail_on`` error type (by default
                :class:`TransientError`) when a prompt is configured to fail.
        """
        completions: list[Completion] = []
        for prompt in prompts:
            failure = self._match_failure(prompt)
            if failure is not None:
                raise failure(
                    f"MockProvider configured to fail on prompt: {prompt!r}"
                )
            completions.append(Completion(text=self._resolve(prompt), prompt=prompt))
        return completions

    def _match_failure(self, prompt: str) -> _ErrorType | None:
        """Return the error type configured for ``prompt``, or ``None``.

        Exact matches take precedence over glob-pattern matches; patterns are
        tried in sorted order for deterministic behavior when several match.
        """
        if not self._fail_on:
            return None
        if prompt in self._fail_on:
            return self._fail_on[prompt]
        for pattern in sorted(self._fail_on):
            if fnmatch.fnmatchcase(prompt, pattern):
                return self._fail_on[pattern]
        return None

    def _resolve(self, prompt: str) -> str:
        """Resolve the deterministic completion text for ``prompt``."""
        # 1. Exact match.
        if prompt in self._responses:
            return self._responses[prompt]
        # 2. Glob-pattern match (sorted keys -> deterministic first match).
        for pattern in sorted(self._responses):
            if fnmatch.fnmatchcase(prompt, pattern):
                return self._responses[pattern]
        # 3. Stable synthetic fallback derived from the prompt.
        return self._synthesize(prompt)

    @staticmethod
    def _synthesize(prompt: str) -> str:
        """Derive a stable, reproducible completion from ``prompt``.

        Uses SHA-256 (not Python's salted ``hash``) so the output is identical
        across processes and runs, satisfying the determinism requirement.
        """
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return f"mock-completion[{digest}]: {prompt}"


class OpenAIProvider(LLMProvider):
    """LLMProvider stub wrapping an OpenAI-style async client.

    This confines the only optional third-party dependency to a single class
    (Requirement 10.3). The ``openai`` import is performed lazily inside
    :meth:`complete` so importing :mod:`agentic.providers` never requires the
    package to be installed.

    The API key is read from the environment only (``OPENAI_API_KEY``); it is
    never accepted as a constructor argument and never logged. For tests or
    dependency injection, a preconfigured ``client`` may be supplied directly.

    Args:
        model: The model identifier to request completions from
            (e.g. ``"gpt-4o-mini"``).
        client: An optional preconfigured OpenAI-style async client. When
            ``None``, :meth:`complete` attempts to construct one lazily from the
            environment; if neither a client nor an API key is available it
            raises a clear :class:`PermanentError`.

    Raises:
        ValueError: If ``model`` is empty.
    """

    def __init__(self, model: str, client: object | None = None) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("OpenAIProvider.model must be a non-empty string")
        self.model = model
        self._client = client

    def _ensure_client(self) -> object:
        """Return a usable client, constructing one lazily if needed.

        Raises:
            PermanentError: If no client was injected and one cannot be built
                from the environment (missing ``openai`` package or missing
                ``OPENAI_API_KEY``). A missing key/client is a permanent
                misconfiguration, not a retryable condition.
        """
        if self._client is not None:
            return self._client

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise PermanentError(
                "OpenAIProvider has no configured client and OPENAI_API_KEY is "
                "not set. Inject a client or set the environment variable; for "
                "key-free runs use MockProvider instead."
            )

        try:
            # Lazy import keeps the third-party dependency confined to this
            # class and optional for the rest of the package.
            from openai import AsyncOpenAI  # type: ignore import-not-found
        except ImportError as exc:  # pragma: no cover - depends on env
            raise PermanentError(
                "The 'openai' package is not installed. Install it to use "
                "OpenAIProvider, or use MockProvider for key-free runs."
            ) from exc

        self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    async def complete(self, prompts: list[str]) -> list[Completion]:
        """Complete ``prompts`` via the OpenAI-style client, preserving order.

        This stub issues one request per prompt and assembles the results in
        input order. A real deployment may batch differently, but the contract
        (input order == output order) is what the engine relies on.

        Args:
            prompts: The prompts to complete.

        Returns:
            Completions in the same order as ``prompts``.

        Raises:
            PermanentError: If no client/API key is configured (see
                :meth:`_ensure_client`).
        """
        client = self._ensure_client()

        completions: list[Completion] = []
        for prompt in prompts:
            response = await client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
            completions.append(Completion(text=text, prompt=prompt))
        return completions
