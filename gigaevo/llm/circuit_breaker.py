"""Circuit breaker for LLM-bearing routers.

The router calls third-party LLM endpoints. A provider outage
(OpenAI 503 storm, OpenRouter NAT timeouts, internal vLLM crash)
otherwise translates into thousands of synchronous retries and
ratelimit pile-ups that hide the actual symptom: nothing the engine
ships succeeds.

The breaker tracks consecutive failures per router instance; once a
threshold is crossed it opens, short-circuits subsequent calls with
:class:`CircuitOpenError`, and re-enters a half-open probe state
after a cooldown. A single half-open success closes the breaker; any
failure re-opens it with the cooldown timer reset.

Three states (closed / open / half-open) mirror the Hystrix-style
finite-state machine but the breaker has no out-of-process state —
keeping it process-local sidesteps the "what if Redis is the broken
upstream" problem that would otherwise create a distributed
chicken-and-egg between the breaker and the very ledger it lives on.

A failure is anything that the caller treats as "the LLM did not
produce a usable response": surfaced via :meth:`record_failure`. A
success — even on a partial response that the schema parser
recovers — closes the breaker via :meth:`record_success`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any


class CircuitState(Enum):
    """Three-state breaker FSM.

    ``CLOSED`` is the steady-state "calls go through". ``OPEN`` means
    the breaker is rejecting calls until the cooldown expires.
    ``HALF_OPEN`` admits one probe call; the next outcome decides the
    next state.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is short-circuited by an open breaker.

    Carries the breaker name and the remaining cooldown so a
    higher-level retry policy can surface a sensible error to the
    operator (e.g. "skip this generation, requeue at t+12s").
    """

    def __init__(self, name: str, remaining_cooldown_s: float):
        self.name = name
        self.remaining_cooldown_s = remaining_cooldown_s
        super().__init__(
            f"circuit breaker {name!r} is open (cooldown {remaining_cooldown_s:.1f}s)"
        )


@dataclass
class CircuitBreakerConfig:
    """Tunables for a single :class:`LLMCircuitBreaker` instance."""

    failure_threshold: int = 5
    """Consecutive failures that flip the breaker to OPEN. The default
    is intentionally conservative — most LLM ratelimit storms recover
    within a handful of retries, and a too-tight threshold makes the
    breaker thrash."""

    cooldown_s: float = 30.0
    """Seconds the breaker stays OPEN before admitting a half-open
    probe. Should comfortably exceed the upstream's typical
    Retry-After header value (60s for OpenAI, 30s for OpenRouter)."""

    half_open_max_probes: int = 1
    """Number of concurrent half-open probes allowed. Keep at 1 to
    avoid hammering a recovering upstream with N parallel probes the
    moment cooldown expires."""

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.cooldown_s <= 0:
            raise ValueError("cooldown_s must be > 0")
        if self.half_open_max_probes < 1:
            raise ValueError("half_open_max_probes must be >= 1")


@dataclass
class LLMCircuitBreaker:
    """Per-router circuit breaker.

    Thread-safe via an internal lock; safe to share across asyncio
    tasks because every transition acquires the lock atomically and
    no awaitable code runs inside the critical section.
    """

    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _half_open_inflight: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def remaining_cooldown_s(self, now: float | None = None) -> float:
        """Seconds left before the breaker becomes half-open.

        Returns ``0.0`` when the breaker is not OPEN or the cooldown
        has elapsed.
        """
        with self._lock:
            if self._state is not CircuitState.OPEN or self._opened_at is None:
                return 0.0
            current = time.monotonic() if now is None else now
            elapsed = current - self._opened_at
            return max(0.0, self.config.cooldown_s - elapsed)

    def allow_request(self, now: float | None = None) -> bool:
        """Atomic ``should_call?`` query with transition side-effects.

        Returns ``True`` and (where relevant) flips OPEN→HALF_OPEN /
        counts the half-open probe; returns ``False`` while OPEN with
        cooldown remaining or while HALF_OPEN with the probe budget
        exhausted.
        """
        with self._lock:
            current = time.monotonic() if now is None else now
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                assert self._opened_at is not None
                if (current - self._opened_at) >= self.config.cooldown_s:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_inflight = 1
                    return True
                return False
            # HALF_OPEN
            if self._half_open_inflight >= self.config.half_open_max_probes:
                return False
            self._half_open_inflight += 1
            return True

    def record_success(self) -> None:
        """Reset the failure counter and (re-)close the breaker."""
        with self._lock:
            self._consecutive_failures = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._half_open_inflight = 0

    def record_failure(self, now: float | None = None) -> None:
        """Account a failure; open the breaker once the threshold is hit.

        A failure during HALF_OPEN re-opens the breaker immediately and
        resets the cooldown clock — the upstream still isn't healthy.
        """
        with self._lock:
            current = time.monotonic() if now is None else now
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = current
                self._half_open_inflight = 0
                self._consecutive_failures += 1
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = current

    def guard(self, now: float | None = None) -> None:
        """Raise :class:`CircuitOpenError` if the breaker is open.

        Convenience for call sites that want a one-liner upstream
        check; equivalent to ``if not allow_request(): raise``.
        """
        if not self.allow_request(now=now):
            raise CircuitOpenError(self.name, self.remaining_cooldown_s(now=now))

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for metrics / logs."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "half_open_inflight": self._half_open_inflight,
                "remaining_cooldown_s": self.remaining_cooldown_s(),
            }
